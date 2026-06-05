# SPDX-License-Identifier: Apache-2.0
"""KV-tunnel MARSHAL workspace: blob storage + tunneled-retrieve scatter.

A leaf module owning the per-process MARSHAL -> RETRIEVE rendezvous: the
pinned workspace pool, the lock, and the ``marshal_handle`` -> blob dict,
plus the H2D scatter that the tunneled-RETRIEVE path delegates to. Kept as
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

from kvtunnel.marshal.pack import TunneledRequestMetadata

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import check_interprocess_event_support
from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_h2d
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # Type-only import: keeps this a true leaf so engine_context.py can
    # import MarshalWorkspace at top level (for the ctx.marshal_workspace
    # annotation) without a runtime import cycle. See module docstring.
    from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext

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


@dataclass
class WorkspaceEntry:
    """One KV-tunneled MARSHAL blob set, keyed by ``marshal_handle``.

    ``mem_objs_per_rank`` maps tp_rank -> (k pooled chunk MemoryObjs,
    manifest) — per-rank because each TP worker retrieves its own KV
    shard (shards hash to different object keys; see
    ipc_key_to_object_keys). For single-GPU deployments the inner dict
    has one entry at rank 0. ``instance_id`` is the GPU instance MARSHAL
    packed against; MARSHAL_FREE schedules the ``ref_count_down`` on that
    context's stream.

    Reclaimed by the MARSHAL_FREE RPC the proxy fires when the consuming
    request/cycle finishes — ``ref_count_down`` returns each chunk to the
    pinned workspace pool.

    Args:
        mem_objs_per_rank: tp_rank -> (packed chunk MemoryObjs, manifest).
        instance_id: GPU instance the blob was packed against.
    """

    mem_objs_per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]]
    instance_id: int


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
        # rendezvous, keyed by marshal_handle. Mutated by put() (write)
        # and free() (pop), both under ``_lock``; RETRIEVE only reads.
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
        must stay lock-free (it relies on GIL atomicity, exactly today's
        RETRIEVE behavior — see the RETRIEVE-only-reads contract on
        ``_workspace``). Do NOT take ``_lock`` here.

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

        Fired by the proxy once the request/cycle that consumed the blob
        has finished. Pops the entry under ``_lock``, then schedules the
        per-chunk ``ref_count_down`` as a stream-ordered host callback on
        the packing context's stream (the STORE finalize idiom) so a freed
        chunk's pinned bytes are never reclaimed while an in-flight
        RETRIEVE H2D is still draining. The normal proxy path fires this
        only after the vLLM completion returns, i.e. after the H2D has
        drained, so it is already safe by timing; the stream-ordering is
        defense for the abort path. The handler does pop + enqueue ONLY —
        the actual free runs later on the cupy callback thread — so it
        stays O(us) and never blocks the shared CPU pool. Returns as soon
        as the free is *enqueued*; the ack does NOT mean the buffer is
        reclaimed. Unknown / already-freed handle is a no-op.

        Args:
            handle: Workspace entry to reclaim.
        """
        with self._lock:
            entry = self._workspace.pop(handle, None)
        if entry is None:
            return  # unknown / already freed — no-op

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

    def retrieve_into(
        self,
        tp_rank: int,
        gpu_block_ids: list[int],
        marshal_handle: str,
        instance_id: int,
    ) -> tuple[bytes, bool]:
        """Scatter a workspace blob into vLLM's paged KV cache.

        Scatters the k chunk-sized MemoryObjs the pack emitted, in batches
        of <= max_batch_size, reusing the same chunk-scatter loop as
        regular RETRIEVE. The rank's MemoryObjs are read out of the dict
        under ``_lock`` BEFORE scattering, so a concurrent :meth:`free` of
        the same handle cannot swap the dict entry mid-read. NOTE: the lock
        only makes the *dict read* atomic — it does NOT keep the pinned
        bytes alive. :meth:`free` schedules ``ref_count_down`` on the
        packing stream; the bytes' lifetime is guaranteed by (a) the
        proxy's free-after-drain timing (MARSHAL_FREE fires only after the
        consuming RETRIEVE has drained) and (b) stream-ordering when the
        scatter is enqueued before free's drop callback — not by this lock.
        Resolves both ``ctx.chunk_size`` and the owning GPU context (via
        ``ctx.gpu_context_registry``) internally so the unregistered-id
        path raises the byte-stable RuntimeError rather than an
        AttributeError.

        Args:
            tp_rank: Which per-rank blob to pick from the workspace entry.
                Matches the ``worker_id`` on the incoming
                ``IPCCacheEngineKey`` that STORE originally used.
            gpu_block_ids: Paged-cache block IDs that receive the blob.
            marshal_handle: Key into the workspace; the caller guarantees
                it is present (checked lock-free via :meth:`has`).
            instance_id: GPU instance ID; must have a registered context.

        Returns:
            tuple[bytes, bool]: CUDA event IPC handle and success flag,
            same shape as regular RETRIEVE.

        Raises:
            RuntimeError: If the requested rank has no blob, the instance
                is unregistered, or the block count mismatches.
        """
        # Read the rank's MemoryObjs out of the dict under _lock (atomic vs
        # a concurrent free() pop/put). The lock guards the dict entry only;
        # pinned-byte lifetime rests on free-after-drain timing + stream
        # ordering (see the docstring), not on holding this lock.
        with self._lock:
            per_rank = self._workspace[marshal_handle].mem_objs_per_rank
            if tp_rank not in per_rank:
                raise RuntimeError(
                    f"marshal_handle={marshal_handle} has no blob for "
                    f"tp_rank={tp_rank}; "
                    f"available ranks={sorted(per_rank.keys())}"
                )
            # Workspace stores (chunks, metadata) per-rank. Retrieve only
            # needs the chunks here; metadata flows through the MARSHAL
            # response to the proxy + connector.
            mem_objs, _manifest = per_rank[tp_rank]
        entry = self._ctx.gpu_context_registry.get(instance_id)
        if entry is None:
            raise RuntimeError(f"KV cache not registered for GPU ID {instance_id}")
        gpu_context = entry.gpu_context

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
            raise RuntimeError(
                f"gpu_block_ids count mismatch: got {len(gpu_block_ids)}, "
                f"expected k*blocks_per_chunk = {k}*{blocks_per_chunk} = "
                f"{expected_block_count} (k chunks x chunk_size / block_size). "
                f"Check the connector's num_blocks_needed math."
            )
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

        try:
            with (
                torch_dev.device(gpu_context.device),
                torch_dev.stream(gpu_context.stream),
            ):
                all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)
                check_interprocess_event_support()
                event = torch_dev.Event(interprocess=True)
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

                event.record()
        except Exception:
            # Surface the exception explicitly so it's grep-able in the
            # MP log before mq._notify_response swallows the response.
            logger.exception(
                "[kvtunnel CB] retrieve_from_workspace raised handle=%s "
                "tp_rank=%d k=%d num_blocks=%d",
                marshal_handle,
                tp_rank,
                k,
                len(gpu_block_ids),
            )
            raise
        ipc_bytes = event.ipc_handle()
        logger.info(
            "RETRIEVE from workspace handle=%s k=%d blocks=%d "
            "(returning ipc_handle, success)",
            marshal_handle,
            k,
            len(gpu_block_ids),
        )
        return ipc_bytes, True

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
