# SPDX-License-Identifier: Apache-2.0
"""KV-tunnel MARSHAL workspace: entry rendezvous + tunneled-retrieve scatter.

A leaf module owning the per-process MARSHAL -> RETRIEVE rendezvous: the
pinned workspace pool (copy-based methods), the lock, and the
``marshal_handle`` -> :class:`WorkspaceEntry` dict (workspace-owned blobs
OR zero-copy L1-borrowed chunks — see :class:`WorkspaceEntry`), plus the
H2D scatter that the tunneled-RETRIEVE path delegates to. Kept as
a leaf (it imports only allocator + pack primitives, never ``modules/`` or
the ``MPCacheEngineContext`` class) so ``engine_context.py`` can type the
``ctx.marshal_workspace`` seam without an import cycle.
"""

# Standard
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, Generator
import os
import resource
import threading

from kvtunnel.wire.header import TunneledRequestMetadata

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import check_interprocess_event_support
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_h2d
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.multiprocess.native_completion import (
    submit_callback_to_stream,
)
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # Type-only imports: keep this a true leaf so engine_context.py can
    # import MarshalWorkspace at top level (for the ctx.marshal_workspace
    # annotation) without a runtime import cycle. See module docstring.
    from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
    from lmcache.v1.multiprocess.gpu_context import GPUCacheContext

logger = init_logger(__name__)


def _max_locked_memory() -> str:
    """RLIMIT_MEMLOCK soft limit as a human string.

    Logged at workspace-pool construction so a pinned-alloc OOM at boot
    is diagnosable: the kvtunnel workspace pool needs ``ulimit -l`` >=
    its size.

    Returns:
        ``"unlimited"`` when the soft limit is infinite, else the byte
        count formatted as ``"<n> B"``.
    """
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return "unlimited" if soft == resource.RLIM_INFINITY else f"{soft} B"


def _batched_iteration(
    lst: list[MemoryObj], batch_size: int
) -> Generator[tuple[MemoryObj, ...], None, None]:
    """Iterate over a list of MemoryObjs in fixed-size batches.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.

    Yields:
        Batches of the list as tuples.

    Raises:
        ValueError: If ``batch_size`` is less than 1.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


def failure_event_handle(gpu_context: "GPUCacheContext") -> bytes:
    """A freshly recorded no-op CUDA event's IPC handle.

    The loud-failure reply for tunneled-RETRIEVE paths: the RETRIEVE
    response contract is ``(event_ipc_handle, success)`` and the worker
    opens + waits the handle, so a failure must still carry a REAL
    recorded event (an unrecorded or empty handle would crash or hang
    the worker). Recorded inside the consuming context's device scope,
    where stock retrieve creates its events.

    Args:
        gpu_context: The consuming rank's GPU context (device + stream).

    Returns:
        The recorded event's IPC handle bytes.
    """
    with (
        torch_dev.device(gpu_context.device),
        torch_dev.stream(gpu_context.stream),
    ):
        check_interprocess_event_support()
        event = torch_dev.Event(interprocess=True)
        event.record()
    return event.ipc_handle()


@dataclass
class WorkspaceEntry:
    """One KV-tunneled MARSHAL blob set, keyed by ``marshal_handle``.

    ``mem_objs_per_rank`` maps tp_rank -> (k chunk MemoryObjs, manifest)
    — per-rank because each TP worker retrieves its own KV shard (shards
    hash to different object keys; see ipc_key_to_object_keys). For
    single-GPU deployments the inner dict has one entry at rank 0.
    ``instance_id`` is the GPU instance MARSHAL resolved against.

    Two ownership regimes, discriminated by ``l1_keys_per_rank``:

    - ``None`` — workspace-OWNED (streaming_llm, stub): the chunks were
      allocated from the pinned kvtunnel pool; MARSHAL_FREE reclaims via
      the stream-ordered ``ref_count_down`` drop on the packing
      context's stream.
    - set — L1-BORROWED (packed_fp8 zero-copy): ``mem_objs_per_rank``
      holds the READ-LOCKED L1 chunks themselves and
      ``l1_keys_per_rank`` their per-rank object keys. Reclamation means
      releasing the read locks (``finish_read_prefetched``), NEVER
      ``ref_count_down`` — that would free live L1 pool bytes through
      the refcount-bypassing allocator. ``retrieve_into`` consumes a
      rank by popping BOTH dicts in one ``_lock`` critical section and
      releases its keys stream-ordered after the H2D; ``free()``
      releases only the ranks still present (never redeemed).

    Args:
        mem_objs_per_rank: tp_rank -> (chunk MemoryObjs, manifest).
        instance_id: GPU instance the entry was resolved against.
        l1_keys_per_rank: tp_rank -> read-locked L1 keys (L1-borrowed
            entries), or ``None`` (workspace-owned).
    """

    mem_objs_per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]]
    instance_id: int
    l1_keys_per_rank: dict[int, list[ObjectKey]] | None = None


class MarshalWorkspace:
    """Per-process MARSHAL -> RETRIEVE blob rendezvous + scatter.

    Owns the dedicated kvtunnel workspace allocator (a pinned
    LazyMemoryAllocator kept separate from the StorageManager's L1
    allocator so the two don't share an eviction policy), a lock, and the
    ``marshal_handle`` -> :class:`WorkspaceEntry` dict. The MARSHAL handler
    writes entries via :meth:`put`; MARSHAL_FREE reclaims them via
    :meth:`free`; the tunneled-RETRIEVE path reads membership via
    :meth:`has` (lock-free) and scatters via :meth:`retrieve_into`.

    Args:
        ctx: The shared engine context. Used at call time to resolve
            ``ctx.chunk_size`` and ``ctx.gpu_context_registry``; not
            dereferenced at construction.
    """

    def __init__(self, ctx: "MPCacheEngineContext") -> None:
        self._ctx = ctx

        # kvtunnel MARSHAL workspace pool — a dedicated LazyMemoryAllocator
        # kept separate from the StorageManager's L1 allocator so the two
        # don't share an eviction policy. Pinned at construction from a
        # fixed byte budget: KVTUNNEL_WORKSPACE_POOL_GB (default 8);
        # init=pool by default so the whole pool is pinned eagerly and
        # Lazy's background thread no-ops. The pack writes into it;
        # MARSHAL_FREE reclaims via _lock.
        pool_gb = float(os.environ.get("KVTUNNEL_WORKSPACE_POOL_GB", "8"))
        pool_bytes = int(pool_gb * (1 << 30))
        init_gb_env = os.environ.get("KVTUNNEL_WORKSPACE_INIT_GB")
        init_bytes = int(float(init_gb_env) * (1 << 30)) if init_gb_env else pool_bytes
        self.kvtunnel_workspace_allocator: MemoryAllocatorInterface = (
            LazyMemoryAllocator(init_size=init_bytes, final_size=pool_bytes)
        )
        self._lock = threading.Lock()
        # Per-process workspace for KV-tunneled MARSHAL -> RETRIEVE
        # rendezvous, keyed by marshal_handle. The OUTER dict is mutated
        # only by put() (write) and free() (pop), both under ``_lock`` —
        # retrieve_into never adds/removes outer entries (it pops rank
        # state from an ENTRY's inner dicts, also under ``_lock``), so
        # has()'s lock-free membership read stays safe.
        self._workspace: dict[str, WorkspaceEntry] = {}
        # Device indices on which free() has scheduled a stream-ordered
        # ref_count_down host callback. close() synchronizes exactly these
        # before unpinning the pool (see close()). One MP server holds
        # packing contexts on several GPUs under TP>1, so the drain must
        # cover every such device, not just the current one. set.add is
        # GIL-atomic, so free() records without taking ``_lock``.
        self._drain_device_indices: set[int] = set()
        logger.info(
            "kvtunnel workspace pool: %d B (init %d B); Max locked memory=%s",
            pool_bytes,
            init_bytes,
            _max_locked_memory(),
        )

    def has(self, handle: str) -> bool:
        """Return whether ``handle`` has a workspace entry.

        Lock-free dict membership: the tunneled-RETRIEVE detection read
        must stay lock-free (it relies on GIL atomicity; the OUTER dict
        is mutated only by put()/free() under ``_lock`` — see the
        ``_workspace`` comment). Do NOT take ``_lock`` here.

        Args:
            handle: The ``marshal_handle`` to test for membership.

        Returns:
            True if a :class:`WorkspaceEntry` is registered for ``handle``.
        """
        return handle in self._workspace

    def put(self, handle: str, entry: WorkspaceEntry) -> None:
        """Publish a workspace entry under ``handle``.

        Taken under ``_lock`` so a concurrent :meth:`free` cannot
        interleave with the write.

        Args:
            handle: The ``marshal_handle`` to key the entry by.
            entry: The :class:`WorkspaceEntry` MARSHAL packed.
        """
        with self._lock:
            self._workspace[handle] = entry

    def free(self, handle: str) -> None:
        """Reclaim the KV-tunnel workspace entry for ``handle``.

        Fired by the proxy once the request/cycle that consumed the
        entry has finished (or aborted). Pops the entry under ``_lock``,
        then reclaims by ownership regime:

        - L1-borrowed (zero-copy packed_fp8): release the read locks of
          ranks NEVER redeemed by a RETRIEVE (abort-before-RETRIEVE) —
          inline on this handler thread, since no DMA can be in flight
          for a rank that never redeemed. Consumed ranks were popped and
          released by their RETRIEVE's stream-ordered callback; the
          shared ``_lock`` makes the consumed/unconsumed split exact, so
          double-release is impossible.
        - Workspace-owned: schedule the per-chunk ``ref_count_down`` as
          a stream-ordered host callback on the packing context's stream
          (the STORE finalize idiom) so a freed chunk's pinned bytes are
          never reclaimed while an in-flight RETRIEVE H2D is still
          draining. The normal proxy path fires this only after the vLLM
          completion returns, i.e. after the H2D has drained, so it is
          already safe by timing; the stream-ordering is defense for the
          abort path. The handler does pop + enqueue ONLY — the actual
          free runs later on the cupy callback thread — so it stays
          O(us) and never blocks the shared CPU pool. Returns as soon as
          the free is *enqueued*; the ack does NOT mean the buffer is
          reclaimed.

        Unknown / already-freed handle is a no-op.

        Args:
            handle: Workspace entry to reclaim.
        """
        with self._lock:
            entry = self._workspace.pop(handle, None)
        if entry is None:
            return  # unknown / already freed — no-op

        if entry.l1_keys_per_rank is not None:
            # L1-borrowed: NEVER ref_count_down (frees live L1 pool
            # bytes through the refcount-bypassing allocator); release
            # the unconsumed ranks' read locks instead.
            unconsumed = [
                key for keys in entry.l1_keys_per_rank.values() for key in keys
            ]
            if unconsumed:
                self._ctx.storage_manager.finish_read_prefetched(
                    unconsumed, extra_count=0
                )
            return

        all_chunks = [
            mem_obj
            for chunks, _manifest in entry.mem_objs_per_rank.values()
            for mem_obj in chunks
        ]

        def _drop(objs: list[MemoryObj]) -> None:
            for mem_obj in objs:
                mem_obj.ref_count_down()  # 1 -> 0 -> parent_allocator.free

        gpu_entry = self._ctx.gpu_context_registry.get(entry.instance_id)
        if gpu_entry is None:
            # Context already unregistered (teardown) — no DMA can be in
            # flight against it, so free inline.
            _drop(all_chunks)
            return
        # Record the packing context's device so close() drains its stream
        # (the callback below runs on a cupy stream bound to this device).
        self._drain_device_indices.add(gpu_entry.gpu_context.device.index)
        gpu_entry.gpu_context.cupy_stream.launch_host_func(_drop, all_chunks)

    def _consume_rank(
        self, marshal_handle: str, tp_rank: int
    ) -> tuple[list[MemoryObj], list[ObjectKey] | None, str | None]:
        """The consume-once critical section: entry lookup + rank pop.

        ONE ``_lock`` acquisition (atomic vs a concurrent :meth:`free`
        pop) resolves the entry and, for an L1-borrowed one, pops the
        rank's chunks AND keys from both dicts together — the pop gates
        the H2D. A workspace-owned entry is a non-consuming ``.get``
        (duplicate RETRIEVEs re-read the copy).

        Args:
            marshal_handle: Key into the workspace.
            tp_rank: The consuming rank.

        Returns:
            ``(mem_objs, l1_keys, fail)`` — on success ``fail`` is None
            and ``l1_keys`` carries the popped keys (L1-borrowed) or
            None (workspace-owned); on failure ``fail`` is the error
            message and ``l1_keys`` is non-None only in the
            inconsistent-entry case (keys popped, no blob), which the
            caller must release inline.
        """
        l1_keys: list[ObjectKey] | None = None
        mem_objs: list[MemoryObj] = []
        fail: str | None = None
        with self._lock:
            entry = self._workspace.get(marshal_handle)
            if entry is None:
                fail = (
                    f"marshal_handle={marshal_handle} not in workspace "
                    "(freed by a racing MARSHAL_FREE?)"
                )
            elif entry.l1_keys_per_rank is None:
                # Workspace-owned: non-consuming read. Retrieve only
                # needs the chunks; the manifest flows through the
                # MARSHAL response to the proxy + connector.
                pair = entry.mem_objs_per_rank.get(tp_rank)
                if pair is None:
                    fail = (
                        f"marshal_handle={marshal_handle} has no blob for "
                        f"tp_rank={tp_rank}; available "
                        f"ranks={sorted(entry.mem_objs_per_rank)}"
                    )
                else:
                    mem_objs, _manifest = pair
            else:
                # L1-borrowed: pop BOTH dicts together.
                pair = entry.mem_objs_per_rank.pop(tp_rank, None)
                l1_keys = entry.l1_keys_per_rank.pop(tp_rank, None)
                if pair is None or l1_keys is None:
                    fail = (
                        f"marshal_handle={marshal_handle} tp_rank="
                        f"{tp_rank} already consumed (duplicate "
                        "RETRIEVE?)"
                    )
                else:
                    mem_objs, _manifest = pair
        return mem_objs, l1_keys, fail

    def retrieve_into(
        self,
        tp_rank: int,
        gpu_block_ids: list[int],
        marshal_handle: str,
        instance_id: int,
    ) -> tuple[bytes, bool]:
        """Scatter a workspace entry into vLLM's paged KV cache.

        Scatters the rank's k chunk-sized MemoryObjs in batches of <=
        max_batch_size, reusing the same chunk-scatter loop as regular
        RETRIEVE. For a workspace-OWNED entry the chunks are pool copies
        and the read is non-consuming (bytes' lifetime rests on the
        proxy's free-after-drain timing + the stream-ordered drop in
        :meth:`free`). For an L1-BORROWED entry (zero-copy packed_fp8)
        the H2D reads the read-locked L1 chunks directly, so:

        - CONSUME-ONCE: one ``_lock`` critical section resolves the
          entry AND pops the rank's ``(mem_objs, keys)`` from both
          dicts. A missing handle (a racing ``free()`` won) or a missing
          rank (duplicate RETRIEVE — after the first consume the locks
          are released and the bytes may already be recycled) is a LOUD
          failure return, never an H2D.
        - RELEASE: the popped keys are released stream-ordered on this
          (consuming) rank's stream via
          ``submit_callback_to_stream("finish_read_prefetched", ...)`` —
          the stock-retrieve idiom — submitted in a ``finally`` so a
          raise inside the H2D/scatter loop still releases after
          whatever was enqueued. Post-pop failures BEFORE any DMA
          enqueue release inline instead (one release path per exit).
        - Every failure exit returns ``(recorded_event, False)`` + an
          error log, never a raise: the MQ swallows handler raises
          without replying, the worker future never resolves, and the
          request wedges in WAITING_FOR_REMOTE_KVS.

        Args:
            tp_rank: Which per-rank blob to pick from the workspace entry.
                Matches the ``worker_id`` on the incoming
                ``IPCCacheEngineKey`` that STORE originally used.
            gpu_block_ids: Paged-cache block IDs that receive the blob.
            marshal_handle: Key into the workspace; the caller checked
                membership lock-free via :meth:`has` (may have raced a
                ``free()`` since — handled here).
            instance_id: GPU instance ID; must have a registered context.

        Returns:
            tuple[bytes, bool]: CUDA event IPC handle and success flag,
            same shape as regular RETRIEVE.

        Raises:
            RuntimeError: If the instance is unregistered (process-
                lifetime registration, not a runtime race).
        """
        # GPU context FIRST: every later failure then records its event
        # inside this device scope (where stock creates its events).
        gpu_entry = self._ctx.gpu_context_registry.get(instance_id)
        if gpu_entry is None:
            raise RuntimeError(f"KV cache not registered for GPU ID {instance_id}")
        gpu_context = gpu_entry.gpu_context

        mem_objs, l1_keys, fail = self._consume_rank(marshal_handle, tp_rank)
        if fail is not None:
            if l1_keys:
                # Inconsistent-entry safety (keys popped, no blob):
                # release inline — no DMA was or will be enqueued.
                self._ctx.storage_manager.finish_read_prefetched(l1_keys, extra_count=0)
            logger.error("[kvtunnel CB] tunneled RETRIEVE failed: %s", fail)
            return failure_event_handle(gpu_context), False

        # Multi-chunk scatter: the pack emits k chunk-sized MemoryObjs.
        # The kernel `multi_layer_block_kv_transfer` hard-asserts
        # `num_objects <= 4` AND `gpu_context.max_batch_size = 4`. For
        # k > 4 we issue ceil(k / batch_size) separate kernel launches,
        # each staging up to 4 chunks via `_batched_iteration` — the same
        # pattern the regular RETRIEVE uses.
        k = len(mem_objs)
        batch_size = gpu_context.max_batch_size
        kvlgm = gpu_context.kv_layer_groups_manager
        ie_block_size = kvlgm.inference_engine_logical_block_size
        blocks_per_chunk = self._ctx.chunk_size // ie_block_size
        expected_block_count = k * blocks_per_chunk
        if len(gpu_block_ids) != expected_block_count:
            if l1_keys:
                # Post-pop, pre-DMA failure: inline release (outside the
                # loop-scoped finally — one release path per exit).
                self._ctx.storage_manager.finish_read_prefetched(l1_keys, extra_count=0)
            logger.error(
                "[kvtunnel CB] tunneled RETRIEVE failed: gpu_block_ids "
                "count mismatch: got %d, expected k*blocks_per_chunk = "
                "%d*%d = %d (k chunks x chunk_size / block_size). Check "
                "the connector's num_blocks_needed math.",
                len(gpu_block_ids),
                k,
                blocks_per_chunk,
                expected_block_count,
            )
            return failure_event_handle(gpu_context), False
        logger.info(
            "[kvtunnel CB] multi-chunk retrieve handle=%s tp_rank=%d "
            "instance_id=%d k=%d batch_size=%d blocks_per_chunk=%d "
            "gpu_block_ids_count=%d num_outer_iters=%d",
            marshal_handle,
            tp_rank,
            instance_id,
            k,
            batch_size,
            blocks_per_chunk,
            len(gpu_block_ids),
            (k + batch_size - 1) // batch_size,
        )

        ok = False
        # The device-scope entry, support check, and event creation sit
        # outside the release-guaranteeing try below: a raise here can
        # only be a process-lifetime configuration failure (bad device /
        # no IPC-event support) — every retrieve would fail and
        # ``failure_event_handle`` couldn't build a reply either — not a
        # runtime race, so the popped keys' TTL backstop is acceptable
        # (design carve-out: "process-lifetime, not a runtime race").
        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)
            try:
                all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)
                num_groups = gpu_context.kv_layer_groups_manager.num_groups

                # Outer loop: iterate batches of <= batch_size chunks. Each
                # iteration stages its chunks into staging slots
                # 0..batch_len-1, then issues one scatter call per KV
                # layer group. Mirrors retrieve()'s _retrieve_loop.
                for batch_idx, mem_obj_batch in enumerate(
                    _batched_iteration(mem_objs, batch_size=batch_size)
                ):
                    batch_len = len(mem_obj_batch)
                    start_chunk_id = batch_idx * batch_size
                    end_chunk_id = start_chunk_id + batch_len
                    chunk_block_ids_gpu = all_block_ids_gpu[
                        start_chunk_id * blocks_per_chunk : end_chunk_id
                        * blocks_per_chunk
                    ]

                    # H2D: copy this batch's chunks into staging slots
                    # 0..batch_len-1. Each per-chunk MemoryObj is sized
                    # to one chunk's bytes (= tmp_chunk_bytes_), so the
                    # h2d wrapper's size-equality check passes without any
                    # slicing.
                    for chunk_idx, mem_obj in enumerate(mem_obj_batch):
                        lmcache_memcpy_async_h2d(
                            mem_obj,
                            gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                        )

                    # Scatter this batch per KV layer group.
                    for group_idx in range(num_groups):
                        tmp_buffers = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                            batch_len, group_idx
                        )
                        group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                        group_lmcache_chunk_size = gpu_context.get_physical_chunk_size(
                            group_idx
                        )
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tb.data_ptr() for tb in tmp_buffers],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.H2D,
                            gpu_context.get_shape_desc(group_idx),
                            group_lmcache_chunk_size,
                            gpu_context.gpu_kv_format_,
                            0,
                        )

                ok = True
            except Exception:
                # Loud, never re-raised: a raise in a RETRIEVE handler
                # is swallowed by the MQ without a reply and the
                # request wedges in WAITING_FOR_REMOTE_KVS.
                logger.exception(
                    "[kvtunnel CB] retrieve_from_workspace raised "
                    "handle=%s tp_rank=%d k=%d num_blocks=%d",
                    marshal_handle,
                    tp_rank,
                    k,
                    len(gpu_block_ids),
                )
            finally:
                event.record()
                if l1_keys:
                    # Consumed-rank release: stream-ordered on THIS
                    # (consuming) rank's stream, after whatever was
                    # enqueued — the stock-retrieve idiom
                    # (gpu_transfer.py finish_read_prefetched callback).
                    # In a finally so a raise mid-loop still releases.
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_read_prefetched",
                        l1_keys,
                    )
        ipc_bytes = event.ipc_handle()
        if ok:
            logger.info(
                "RETRIEVE from workspace handle=%s k=%d blocks=%d "
                "(returning ipc_handle, success)",
                marshal_handle,
                k,
                len(gpu_block_ids),
            )
        return ipc_bytes, ok

    def close(self) -> None:
        """Drain in-flight MARSHAL_FREE callbacks, then free the pool.

        :meth:`free` schedules each chunk's ``ref_count_down`` (which on the
        last ref returns it to this allocator) as a stream-ordered host
        callback via ``cupy_stream.launch_host_func``. Unpinning the pool
        while such a callback is still pending would decrement into a closed
        allocator, so every device that hosted a packing context is
        synchronized first. It must be these specific devices, not just the
        current one: one MP server holds contexts on several GPUs under TP>1
        (``free`` records each in ``_drain_device_indices``), and a no-arg
        ``synchronize`` would drain only the current device and miss
        callbacks pending on the others. Syncing the captured device set —
        rather than the per-context streams — keeps this correct even though
        ``GPUTransferModule.close`` has already cleared the GPU-context
        registry by now, and never touches a GPU this process did not use.
        Guarded so a CPU-only / torn-down context is a no-op.
        """
        if torch_dev.is_available():
            for device_index in list(self._drain_device_indices):
                torch_dev.synchronize(device_index)
        self.kvtunnel_workspace_allocator.close()
