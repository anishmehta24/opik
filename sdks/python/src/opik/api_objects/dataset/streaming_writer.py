"""Build dataset-item request bodies as rows arrive, instead of materialising the upload.

A row is serialised once as it is added and fed straight into a zlib stream; a finished
body goes to a callback when a size, count or time threshold trips. What the writer holds
is one body in flight, never the upload.

Serialisation here is for the wire only. Content hashes are computed elsewhere, with the
standard library, so item identity never depends on which serialiser is in use.
"""

import asyncio
import dataclasses
import datetime
import decimal
import enum
import json
import logging
import pathlib
import threading
import uuid
import zlib
from concurrent import futures
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set

import httpx
import pydantic

from ... import httpx_client
from .. import constants
from . import identifiers
from ...rest_api.core.jsonable_encoder import jsonable_encoder

LOGGER = logging.getLogger(__name__)

# gzip container rather than a raw deflate stream, matching what the server expects.
_GZIP_WBITS = 16 + zlib.MAX_WBITS

# Values the generated client accepted that a JSON serialiser rejects on its own, and that
# `jsonable_encoder` converts without looking inside anything: for these it is exact.
_FLEXIBLE_LEAVES = (
    bytes,
    enum.Enum,
    datetime.date,  # also covers datetime.datetime
    datetime.time,
    decimal.Decimal,
    uuid.UUID,
    pathlib.PurePath,
)


class ItemNotSerializableError(TypeError):
    """A dataset item could not be serialised to JSON.

    Raised explicitly rather than left to surface from whichever component happens to
    serialise first, so the failure names the item and does not depend on the wire
    serialiser in use.
    """


def _ordered_set_members(value: Any) -> list:
    """A set's members in an order that does not vary between processes.

    A set never survives the round trip: it goes out as a JSON array and comes back as a
    list, so its digest has to equal the digest of the array it becomes or it could never
    deduplicate against its own stored form. A canonical order is what makes those two
    agree, which is why this is a requirement and not tidiness. `list()` alone comes out
    differently in every process -- Python randomises string hashing -- and
    `sort_keys=True` orders a dict's keys, never a list's members.

    Natural order where the members compare, which is most sets and much the cheaper
    path. A set may legitimately mix types and `sorted` alone raises on `{1, "a"}`, so
    those fall back to the type name ahead of the repr; that order is arbitrary rather
    than meaningful, which is all a canonical form has to be.

    A row written before this was canonical holds an arbitrary order, so a set will not
    deduplicate against one. That is not recoverable from here -- the order differs per
    row and per writing process -- and it costs nothing that worked, because hashing a
    set raised `TypeError` before this change and such a row could never be deduplicated
    against at all.
    """
    try:
        return sorted(value)
    except TypeError:
        return sorted(value, key=lambda member: (type(member).__name__, repr(member)))


def encode_flexible(value: Any) -> Any:
    """One value the wire serialiser could not encode, in the form the client sent before.

    Called only for values a serialiser rejects, so ordinary JSON-native items never pay
    for the normalisation pass this restores.

    A value with an interior is handed back as its shell rather than encoded here, so the
    serialiser walks into it and comes back through this hook for each member. Letting
    `jsonable_encoder` encode the interior instead would apply its last resort to whatever
    it found there -- `vars(obj)` for an object with no JSON form, uploading it as a dict
    of its attributes rather than telling the caller it cannot be sent.
    """
    if isinstance(value, _FLEXIBLE_LEAVES):
        return jsonable_encoder(value)
    if isinstance(value, (set, frozenset)):
        return _ordered_set_members(value)
    if isinstance(value, tuple):
        # A tuple's order is the caller's, unlike a set's, so it is kept as given.
        return list(value)
    if isinstance(value, pydantic.BaseModel):
        # `model_dump`, not the deprecated `dict`, and in python mode so the members come
        # back through here rather than being encoded by pydantic on the way out.
        return value.model_dump(by_alias=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any) -> bytes:
    """The wire form of one value.

    `default=` carries the flexible types the generated client used to accept, so only a
    value that needs the normalisation pays for it.
    """
    return json.dumps(value, default=encode_flexible).encode("utf-8")


class StreamingBatchWriter:
    """Accumulate serialised items and emit complete request bodies.

    `flush_callback` receives `(body, item_count)` for each finished body. It is called on
    the adding thread, so a bounded callback is what applies back-pressure to the producer.

    `gzip_level` of None emits the body uncompressed, for a client configured with
    `enable_json_request_compression` off.
    """

    def __init__(
        self,
        *,
        envelope: Mapping[str, Any],
        flush_callback: Callable[[bytes, int], None],
        max_payload_bytes: int,
        max_items: int,
        gzip_level: Optional[int],
    ) -> None:
        self._flush_callback = flush_callback
        self._max_payload_bytes = max_payload_bytes
        self._max_items = max_items
        self._gzip_level = gzip_level

        # Serialise the envelope once and splice the item array onto it, so the dataset
        # name and group id are escaped by the same serialiser as everything else. The
        # envelope always carries fields: splicing onto an empty `{}` would produce
        # `{,"items":[`, so this is not a shape to make optional later.
        envelope_bytes = dumps(dict(envelope))
        self._prefix = envelope_bytes[:-1] + b',"items":['
        self._suffix = b"]}"

        self._start_buffer()

    def _start_buffer(self) -> None:
        self._compressor = (
            None
            if self._gzip_level is None
            else zlib.compressobj(self._gzip_level, zlib.DEFLATED, _GZIP_WBITS)
        )
        self._chunks = [self._encode(self._prefix)]
        # Seeded with the envelope, not zero: the cap bounds the request the backend
        # receives, and a reverse proxy in front of a self-hosted install measures the
        # whole body. Counting only the items lets a batch land over the limit by the
        # envelope's width.
        self._logical_bytes = len(self._prefix) + len(self._suffix)
        self._items = 0

    def _encode(self, data: bytes) -> bytes:
        return data if self._compressor is None else self._compressor.compress(data)

    def _serialize(self, item: Mapping[str, Any]) -> bytes:
        try:
            return dumps(item)
        except Exception as exception:
            # Checked here rather than relying on a hashing or encoding step elsewhere to
            # raise first: a streaming writer may be the only component that touches the
            # item, and the caller needs a consistent error either way.
            raise ItemNotSerializableError(
                f"Dataset item is not JSON-serializable: {exception}"
            ) from exception

    def add(self, item: Mapping[str, Any]) -> None:
        payload = self._serialize(item)

        # Closing before the offending item, rather than after, is what puts an item at
        # or over the cap alone in its own request: a request rejected for its size then
        # fails that one row instead of every row that shared a batch with it. The `+ 1`
        # is the comma that would join this item to the one before it.
        if self._items > 0 and self._logical_bytes + 1 + len(payload) > (
            self._max_payload_bytes
        ):
            self.flush()

        # The comma goes into the stream on its own rather than being prepended to the
        # payload: `b"," + payload` would copy the whole row to add one byte, which at
        # this row size is a fresh buffer per item for nothing. Same bytes either way.
        if self._items > 0:
            self._chunks.append(self._encode(b","))
            self._logical_bytes += 1
        self._chunks.append(self._encode(payload))
        self._logical_bytes += len(payload)
        self._items += 1

        if self._should_flush():
            self.flush()

    def _should_flush(self) -> bool:
        return (
            self._logical_bytes >= self._max_payload_bytes
            or self._items >= self._max_items
        )

    def flush(self) -> None:
        """Emit whatever is buffered. A no-op when nothing has been added."""
        if self._items == 0:
            return

        self._chunks.append(self._encode(self._suffix))
        if self._compressor is not None:
            self._chunks.append(self._compressor.flush(zlib.Z_FINISH))
        body = b"".join(self._chunks)
        item_count = self._items

        self._start_buffer()
        self._flush_callback(body, item_count)


# Given a body and the pool's client: build the request, send it, retry it, and raise
# what retrying could not fix. Leaves URL, headers and retry policy with the caller, the
# same division the thread-backed pool had.
AsyncSend = Callable[[bytes, httpx.AsyncClient], Awaitable[None]]


@dataclasses.dataclass
class _Running:
    """The loop, its thread and the client: started together, torn down together.

    One optional field rather than three, so "has the pool started" is a single question
    and no half-started combination can be represented.
    """

    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    client: httpx.AsyncClient


class BoundedSendPool:
    """Send finished bodies over asyncio, with only so many outstanding at once.

    The bound is the point: without it a producer that serialises faster than the network
    drains would turn "never materialise the upload" back into "materialise it as queued
    request bodies". `submit` blocks once `num_threads * 2` bodies are outstanding, and
    waits there on a plain `concurrent.futures.Future` per body, so the producing thread
    never touches asyncio itself. `num_threads * 2` requests may be in flight at once --
    it sizes the bound, not a pool of threads; every send runs as a coroutine on one
    background event loop.

    A single worker takes none of that: it sends inline through `inline_send`, starting
    no loop, no thread and no client, which is what a backend older than
    `MIN_BACKEND_VERSION_FOR_PARALLEL_INSERT` needs. So does an upload whose client sends
    through a transport an async client cannot use: rather than send around whatever that
    transport enforces, the pool falls back to the same inline path.

    Neither does an upload that turns out to be one request, whatever the worker count.
    The first body is held rather than sent, and goes inline at `close` if no second body
    follows -- so `insert` of a handful of items pays for no loop, no thread and no second
    connection pool, and a caller inserting in a loop keeps the sync client's keepalive
    rather than opening a connection per call. A second body starts the loop and both go
    through it, in order.

    The first failure is re-raised to the producer, from `submit` or `close`. There is no
    rollback, so bodies already accepted stay persisted.
    """

    def __init__(
        self,
        *,
        send: AsyncSend,
        inline_send: Callable[[bytes], None],
        num_threads: int,
        transport: httpx.Client,
        max_pending: int,
    ) -> None:
        self._send = send
        self._inline_send = inline_send
        self._single_worker = num_threads == 1
        self._transport = transport
        self._max_pending = max_pending
        # Touched only by the producing thread, so it needs no lock of its own.
        self._pending: Set["futures.Future[None]"] = set()

        # Nothing is started here: an insert that never produces a second body must not
        # cost a loop, and a constructor that starts one leaks it if the caller's next
        # line raises before the try that closes this.
        self._held: Optional[bytes] = None
        self._running: Optional[_Running] = None

    def _start(self) -> "_Running":
        """Open the loop, its thread and the client, or leave nothing behind trying.

        The thread is what fails here in practice -- this path exists to avoid threads, so
        a process short of them is exactly where it runs -- and an unclosed loop is a
        leaked selector descriptor, one per insert for a caller inserting in a loop.
        """
        # Built from the client the sync path sends through, so the upload keeps the same
        # identity, base URL, timeouts and TLS settings. Raises rather than returning a
        # client that would send around the caller's transport.
        client = httpx_client.async_twin(
            self._transport, max_connections=self._max_pending
        )
        loop = None
        try:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name="opik-dataset-send-pool",
                daemon=True,
            )
            thread.start()
        except BaseException:
            if loop is not None:
                loop.close()
            # The client is dropped rather than closed: `aclose` is a coroutine and no loop
            # is left to run it on. It never sent a request, so it holds no connection.
            raise

        return _Running(loop=loop, thread=thread, client=client)

    async def _guarded_send(
        self, body: bytes, client: httpx.AsyncClient, sent: "futures.Future[None]"
    ) -> None:
        # Every outcome is put on the future here rather than left to the one
        # `run_coroutine_threadsafe` returns: a `BaseException` out of a task takes the
        # loop down with it, and the future it would have resolved never resolves, so a
        # producer waiting at the bound would wait forever.
        try:
            await self._send(body, client)
        except BaseException as exception:
            sent.set_exception(exception)
        else:
            sent.set_result(None)

    def _enqueue(self, body: bytes, running: "_Running") -> None:
        # Waiting here is the back-pressure, and the only bound there is: the producer
        # cannot run ahead of the network by more than the bodies this set holds.
        if len(self._pending) >= self._max_pending:
            done, self._pending = futures.wait(
                self._pending, return_when=futures.FIRST_COMPLETED
            )
            for future in done:
                future.result()

        sent: "futures.Future[None]" = futures.Future()
        # Tracked only once the loop has accepted it: a future recorded before a failed
        # `run_coroutine_threadsafe` -- a submit after close, against a loop already shut --
        # is one nothing will ever resolve, and every later wait would block on it forever.
        asyncio.run_coroutine_threadsafe(
            self._guarded_send(body, running.client, sent), running.loop
        )
        self._pending.add(sent)

    def _hold(self, body: bytes) -> None:
        """Keep the first body back, in case it is the only one."""
        self._held = body

    def _promote_held(self, held: bytes) -> Optional["_Running"]:
        """Second body: the upload is worth a loop after all, so start and send the first.

        Returns the started state, or None when the caller's transport carries policy an
        async client cannot use -- in which case the upload stays on the sync client,
        which is slower and the only option that does not send around that policy.

        The held body is released only once its destination is settled. Clearing it first
        would drop it if `_start` raised, and there is nowhere left to send it from by then.
        """
        try:
            running = self._start()
        except httpx_client.AsyncTransportUnavailable as exception:
            LOGGER.warning(
                "Uploading dataset items sequentially: %s, so the parallel upload would "
                "have had to send around it. Other requests are unaffected.",
                exception,
            )
            self._single_worker = True
            self._held = None
            self._inline_send(held)
            return None

        self._running, self._held = running, None
        self._enqueue(held, running)
        return running

    def submit(self, body: bytes, item_count: int) -> None:
        LOGGER.debug("Sending dataset items batch of size %d", item_count)
        if self._single_worker:
            self._inline_send(body)
            return

        running = self._running
        if running is None:
            held = self._held
            if held is None:
                self._hold(body)
                return
            running = self._promote_held(held)
            if running is None:
                self._inline_send(body)
                return

        self._enqueue(body, running)

    def close(self) -> None:
        """Finish the upload: flush a body still held, then drain and tear down."""
        if self._held is not None:
            # The whole upload was one body, so it never needed a loop. Sent here rather
            # than at `submit` because only now is it known that no second body follows.
            held, self._held = self._held, None
            self._inline_send(held)

        if self._running is None:
            return
        try:
            for future in futures.as_completed(self._pending):
                future.result()
        finally:
            self._shutdown()

    def abort(self) -> None:
        """Tear down without finishing the upload, for a producer that is already failing.

        The held body is dropped rather than sent: `close` would start a fresh blocking
        request, with its retries and rate-limit waits, while an exception unwinds -- so a
        serialisation failure half way through, or a Ctrl-C, would begin an upload instead
        of ending one. Sends already in flight are still drained; abandoning them
        mid-request is not ours to do.
        """
        self._held = None
        self._shutdown()

    def _shutdown(self) -> None:
        """Stop the loop, its thread and the client. Safe to call more than once."""
        running = self._running
        if running is None or running.loop.is_closed():
            return

        # A first failure leaves the rest of the sends running; stopping the loop under
        # them would abandon them mid-request.
        futures.wait(self._pending)
        try:
            asyncio.run_coroutine_threadsafe(
                running.client.aclose(), running.loop
            ).result()
        finally:
            running.loop.call_soon_threadsafe(running.loop.stop)
            running.thread.join()
            running.loop.close()


def item_payload(
    *,
    item_id: Optional[str],
    trace_id: Optional[str],
    span_id: Optional[str],
    source: Optional[str],
    data: Mapping[str, Any],
    description: Optional[str],
    evaluators: Optional[Any],
    execution_policy: Optional[Any],
) -> Dict[str, Any]:
    """The wire form of one dataset item.

    Mirrors the fields the generated client sends for `DatasetItemWrite`, and only those:
    the model declares `tags` as well, but the conversion never sets it, and the generated
    client omits fields that were never set while serialising explicit `None`s as null.
    Anything added here that the generated client does not send would change the request.

    The three identifiers go through `identifiers.optional_canonical_id`, because
    `DatasetItem` passes whatever it was given straight through and a `uuid.UUID` is an
    identifier that simply is not a string yet. Anything that is not a UUID in either
    form is refused before this, by `identifiers.validate_identifier`.
    """
    return {
        "id": identifiers.optional_canonical_id(item_id),
        "trace_id": identifiers.optional_canonical_id(trace_id),
        "span_id": identifiers.optional_canonical_id(span_id),
        "source": source,
        "data": data,
        "description": description,
        "evaluators": evaluators,
        "execution_policy": execution_policy,
    }


def build_batch_writer(
    *,
    dataset_name: str,
    project_name: Optional[str],
    batch_group_id: str,
    flush_callback: Callable[[bytes, int], None],
    gzip_level: Optional[int],
) -> StreamingBatchWriter:
    """The writer one dataset upload sends through.

    The batch caps live here rather than at the call site, so what bounds a request is
    decided in one place for every caller instead of being passed in and possibly
    differing between them. `gzip_level` stays a parameter because only the caller knows
    whether its transport expects a compressed body.
    """
    return StreamingBatchWriter(
        envelope={
            "dataset_name": dataset_name,
            "project_name": project_name,
            "batch_group_id": batch_group_id,
        },
        flush_callback=flush_callback,
        max_payload_bytes=int(constants.DATASET_ITEMS_MAX_BATCH_SIZE_MB * 1024 * 1024),
        max_items=constants.DATASET_ITEMS_MAX_BATCH_SIZE,
        gzip_level=gzip_level,
    )


def build_send_pool(
    send: AsyncSend,
    inline_send: Callable[[bytes], None],
    num_threads: int,
    transport: httpx.Client,
) -> BoundedSendPool:
    """The upload sink for one insert.

    Owns the one derivation the pool used to make for itself: twice the worker count, so
    a send that finishes has a body waiting without the producer running arbitrarily far
    ahead. Passing it in explicitly keeps the pool free of a default that decided policy
    where it could not be seen. It bounds requests in flight rather than threads: the
    sends are coroutines on one loop, and the same number also sizes the upload client's
    connection pool.

    Both senders go in. The pool sends concurrently over an async client it builds from
    `transport`, and inline through `inline_send` when there is a single worker or when
    the whole upload turns out to be one body.
    """
    return BoundedSendPool(
        send=send,
        inline_send=inline_send,
        num_threads=num_threads,
        transport=transport,
        max_pending=num_threads * 2,
    )
