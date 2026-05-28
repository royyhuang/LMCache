# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Generator
import argparse
import os
import resource
import threading
import time
import uuid

from kvtunnel.marshal.pack import (
    TunneledRequestMetadata,
    streaming_llm_pack,
    stub_pack_for_plumbing,
)

# Third Party
import torch
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle, StorageManager
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.mp_observability.config import (
    ObservabilityConfig,
    add_observability_args,
    init_observability,
    parse_args_to_observability_config,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace import maybe_initialize_trace_recorder
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    add_mp_server_args,
    parse_args_to_mp_server_config,
)
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.gpu_context import (
    GPUCacheContext,
)
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


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
    request/cycle finishes (closes O1) — ``ref_count_down`` returns each
    chunk to the pinned workspace pool. No more leak.
    """

    mem_objs_per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]]
    instance_id: int


# Per-process workspace for KV-tunneled MARSHAL → RETRIEVE rendezvous,
# keyed by marshal_handle. Mutated by the MARSHAL handler (write) and the
# MARSHAL_FREE handler (pop), both under ``MPCacheEngine._workspace_lock``;
# RETRIEVE only reads.
_WORKSPACE: dict[str, WorkspaceEntry] = {}


# Helper functions
def _max_locked_memory() -> str:
    """RLIMIT_MEMLOCK soft limit as a human string.

    Logged at workspace-pool construction so a pinned-alloc OOM at boot is
    diagnosable: the kvtunnel workspace pool needs ``ulimit -l`` >= its size.
    """
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    return "unlimited" if soft == resource.RLIM_INFINITY else f"{soft} B"


def compute_extra_count(
    tp_size: int,
    world_size: int,
) -> int:
    """Compute extra count for MLA multi-reader locking.

    Non-MLA: each TP worker owns a distinct KV shard,
      so each ObjectKey is retrieved by exactly 1
      worker -> extra_count = 0.
    MLA: TP does not split KV caches, all TP workers
      share the same object. vLLM passes world_size
      already divided by tp_size (e.g. world_size=1
      for TP=4 PP=1), so ipc_keys_to_object_keys
      only produces 1 ObjectKey per chunk.  All TP
      workers retrieve that same ObjectKey, hence
      extra_count = tp_size - 1.

    Detection: tp > world_size means MLA (world_size
    was divided by tp on the vLLM side).

    Fallback: old vLLM (<= 0.8.5) does not send
    tp_size (defaults to 1); we fall back to
    world_size which gives extra_count = 0
    (safe but may under-lock for MLA).

    TODO: world_size currently carries an overloaded
    meaning (total ranks for non-MLA vs total/tp for
    MLA). Consider a dedicated field in the future.

    Args:
        tp_size: Tensor-parallel size from the client.
        world_size: World size from the cache key.

    Returns:
        Number of extra count (0 for non-MLA).
    """
    tp = tp_size if tp_size > 1 else world_size
    return tp - 1 if tp > world_size else 0


def get_layout_desc(gpu_context: GPUCacheContext, num_tokens: int) -> MemoryLayoutDesc:
    """Get the memory layout description for a given GPU context and number of tokens.

    Supports multiple KV layer groups with different shapes and dtypes.

    Args:
        gpu_context: The GPU cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and dtypes.
    """
    num_groups = gpu_context.kv_layer_groups_manager.num_groups
    shapes = [
        gpu_context.get_kv_buffer_shape(num_tokens, group_idx)
        for group_idx in range(num_groups)
    ]
    dtypes = [
        gpu_context.kv_layer_groups_manager.kv_layer_groups[group_idx].dtype
        for group_idx in range(num_groups)
    ]
    return MemoryLayoutDesc(shapes=shapes, dtypes=dtypes)


def batched_iteration(lst: list, batch_size: int) -> Generator[tuple, None, None]:
    """Utility function to iterate over a list in batches.

    Args:
        lst: The list to iterate over.
        batch_size: The size of each batch.

    Yields:
        Batches of the list as tuples.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


@dataclass
class _PrefetchJob:
    handle: PrefetchHandle
    world_size: int
    request_id: str


# Main class for the mp cache engine
class MPCacheEngine:
    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ):
        # GPU ID -> KV cache tensors
        self.gpu_contexts: dict[int, GPUCacheContext] = {}

        # GPU ID -> (model name, world size) as metadata
        # NOTE: This is mainly for determining the layout desc during prefetch
        # We assume that if the (model name, world size) is the same, then
        # the layout desc returned by the gpu context is the same.
        self.gpu_context_meta: dict[int, tuple[str, int]] = {}

        # chunk size
        self.chunk_size = chunk_size

        # Lock for clear() to avoid concurrent storage manager mutations
        self.lock = threading.Lock()

        # storage manager
        self.storage_manager = StorageManager(storage_manager_config)

        # kvtunnel MARSHAL workspace pool (closes F1 + O1) — a dedicated
        # LazyMemoryAllocator inited here alongside the StorageManager's own L1
        # allocator, but kept separate from it so the two don't share an
        # eviction policy. Pinned at construction from a fixed byte budget:
        # KVTUNNEL_WORKSPACE_POOL_GB (default 8); init=pool by default so the
        # whole pool is pinned eagerly and Lazy's background thread no-ops. The
        # pack writes into it; MARSHAL_FREE reclaims via _workspace_lock.
        pool_gb = float(os.environ.get("KVTUNNEL_WORKSPACE_POOL_GB", "8"))
        pool_bytes = int(pool_gb * (1 << 30))
        init_gb_env = os.environ.get("KVTUNNEL_WORKSPACE_INIT_GB")
        init_bytes = int(float(init_gb_env) * (1 << 30)) if init_gb_env else pool_bytes
        self.kvtunnel_workspace_allocator: MemoryAllocatorInterface = (
            LazyMemoryAllocator(init_size=init_bytes, final_size=pool_bytes)
        )
        self._workspace_lock = threading.Lock()
        logger.info(
            "kvtunnel workspace pool: %d B (init %d B); Max locked memory=%s",
            pool_bytes,
            init_bytes,
            _max_locked_memory(),
        )

        # Token hasher and session manager for token-based operations
        self.token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self.session_manager = SessionManager(self.token_hasher)

        # EventBus for observability
        self._event_bus = get_event_bus()

        # Prefetch job tracking for two-phase lookup, keyed by request_id.
        # TODO: implement periodic cleanup of stale _prefetch_jobs entries
        # for crash resilience (e.g., client calls lookup but never queries)
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()

        # WAIT_STORE notifier: chunk_hash -> list[Event] of waiters.
        # finish_write (post-DMA, host-callback thread) signals the
        # waiters via the wrapped _finish_write_and_signal helper.
        # No EventBus dependency — sub-ms wakeup, no thundering herd.
        self._pending_chunk_events: dict[bytes, list[threading.Event]] = {}
        self._pending_lock = threading.Lock()

        self._setup_metrics()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        layout_hints: LayoutHints,
    ) -> None:
        """
        Registers the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
            kv_caches (KVCache): The KV cache tensor wrappers from vLLM.
            model_name (str): The name of the model associated with this KV cache.
            world_size (int): The world size associated with this KV cache.
            layout_hints: See :class:`LayoutHints`.  Forwarded to
                :class:`GPUCacheContext` for GPU KV format detection.
        """
        gpu_context = GPUCacheContext(
            kv_caches,
            self.chunk_size,
            layout_hints=layout_hints or None,
        )
        self.gpu_contexts[instance_id] = gpu_context
        self.gpu_context_meta[instance_id] = (model_name, world_size)
        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """
        Unregisters the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id (int): The GPU instance ID (such as PID).
        """
        if instance_id in self.gpu_contexts:
            del self.gpu_contexts[instance_id]
            del self.gpu_context_meta[instance_id]
            logger.info("Unregistered KV cache for GPU ID %d", instance_id)
            torch.cuda.empty_cache()
        else:
            logger.warning("No KV cache found for GPU ID %d to unregister", instance_id)

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """
        Stores the GPU KV cache blocks to CPU.

        Args:
            key (IPCCacheEngineKey): The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to store.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the store operation. The second
                element indicates whether the store operation was successful.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]

        st = time.perf_counter()

        assert key.worker_id is not None, "Must store with worker_id != None"
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        blocks_per_chunk = self.chunk_size // gpu_context.block_size

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            event = torch.cuda.Event(interprocess=True)

            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            # Wait for vLLM to finish
            vllm_event = torch.cuda.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_SESSION_END can race MP_STORE_END.
            self._event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                )
            )

            self._event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                ),
            )

            reserved_dict: dict = {}
            try:
                layout_desc = get_layout_desc(gpu_context, self.chunk_size)
                reserved_dict = self.storage_manager.reserve_write(
                    obj_keys, layout_desc, "new"
                )

                # NOTE: Store is not batched because some obj_keys may be
                # skipped (not in reserved_dict), making block_ids
                # non-contiguous. Batching would require torch.cat to
                # reassemble block_ids, negating the benefit.
                num_groups = gpu_context.kv_layer_groups_manager.num_groups
                for idx, obj_key in enumerate(obj_keys):
                    if obj_key in reserved_dict:
                        memory_obj = reserved_dict[obj_key]
                    else:
                        continue

                    chunk_block_ids_gpu = all_block_ids_gpu[
                        idx * blocks_per_chunk : (idx + 1) * blocks_per_chunk
                    ]

                    # Copy from GPU paged buffer to tmp buffer, then to CPU — per group
                    for group_idx in range(num_groups):
                        tmp_buffer = gpu_context.get_tmp_chunk_gpu_buffer(group_idx)
                        group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tmp_buffer.data_ptr()],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.D2H,
                            gpu_context.get_shape_desc(group_idx),
                            self.chunk_size,
                            gpu_context.gpu_kv_format_,
                            0,
                        )
                    # Store is not batched, so we always use chunk_idx=0 (single slot)
                    lmcache_memcpy_async_d2h(
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=0), memory_obj
                    )
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event.record()
                if reserved_dict:
                    gpu_context.cupy_stream.launch_host_func(
                        self._finish_write_and_signal,
                        list(reserved_dict.keys()),
                    )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": len(reserved_dict),
                            "device": str(gpu_context.device),
                        },
                    ),
                )

        ed = time.perf_counter()
        if length := len(reserved_dict):
            logger.info(
                "Stored %d tokens in %.3f seconds",
                length * self.chunk_size,
                ed - st,
            )
        return event.ipc_handle(), True

    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
        marshal_handle: str = "",
    ) -> tuple[bytes, bool]:
        """
        Retrieves the CPU KV cache and put into GPU blocks.

        Args:
            key (IPCCacheEngineKey): The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id (int): The GPU instance ID (such as PID).
            gpu_block_ids (list[int]): The GPU block IDs to retrieve into.
            event_ipc_handle (bytes): The IPC handle of the event to wait on.
            skip_first_n_tokens (int): Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.
            marshal_handle (str): Rendezvous key for a KV-tunneled request.
                When non-empty and present in ``_WORKSPACE``, the packed
                marshalled blob stashed there by a prior MARSHAL RPC is
                scattered into ``gpu_block_ids`` instead of reading from
                storage. Empty string (default) falls through to the
                standard storage path.

        Returns:
            tuple[bytes, bool]: The first element is the IPC handle of the event
                that signals the completion of the retrieve operation. The second
                element indicates whether the key was successfully retrieved.
        """
        if marshal_handle and marshal_handle in _WORKSPACE:
            # TP rank comes from the incoming key — each TP worker's
            # RETRIEVE carries its own worker_id, matching the per-rank
            # workspace entry produced by marshal(). See _WORKSPACE docs.
            return self._retrieve_from_workspace(
                marshal_handle=marshal_handle,
                tp_rank=key.worker_id or 0,
                instance_id=instance_id,
                gpu_block_ids=gpu_block_ids,
            )

        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]

        st = time.perf_counter()

        assert key.worker_id is not None, "Must retrieve with worker_id != None"
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        assert instance_id in self.gpu_contexts, (
            f"KV cache not registered for GPU ID {instance_id}"
        )
        gpu_context = self.gpu_contexts[instance_id]

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_SESSION_END can race MP_RETRIEVE_END.
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            )
        )

        self._event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            ),
        )

        blocks_per_chunk = self.chunk_size // gpu_context.block_size

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            _BATCH_SIZE = gpu_context.max_batch_size
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                chunk_start = batch_idx * self.chunk_size * _BATCH_SIZE
                chunk_end = chunk_start + self.chunk_size * batch_len

                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    # Entire batch is within APC range, skip it
                    continue

                skip_tokens_in_chunk = max(
                    0,
                    min(
                        effective_start - chunk_start,
                        self.chunk_size * batch_len - 1,
                    ),
                )
                if skip_tokens_in_chunk % gpu_context.block_size != 0:
                    logger.error(
                        "skip_first_n_tokens (%d) is not aligned to block_size (%d), "
                        "rounding down from %d tokens to %d blocks",
                        skip_first_n_tokens,
                        gpu_context.block_size,
                        skip_tokens_in_chunk,
                        skip_tokens_in_chunk // gpu_context.block_size,
                    )
                skip_blocks_in_chunk = skip_tokens_in_chunk // gpu_context.block_size

                start_chunk_id = batch_idx * _BATCH_SIZE
                end_chunk_id = start_chunk_id + batch_len
                chunk_block_ids_gpu = all_block_ids_gpu[
                    start_chunk_id * blocks_per_chunk : end_chunk_id * blocks_per_chunk
                ]

                # Copy from CPU to GPU tmp buffers, then scatter to paged KV — per group
                # H2D copy: each memory_obj maps to its own batch slot
                for chunk_idx, memory_obj in enumerate(memory_obj_batch):
                    lmcache_memcpy_async_h2d(
                        memory_obj,
                        gpu_context.get_tmp_gpu_buffer_flat(chunk_idx=chunk_idx),
                    )
                for group_idx in range(num_groups):
                    tmp_buffers = gpu_context.get_tmp_chunk_gpu_buffer_batched(
                        batch_len, group_idx
                    )
                    group_kv_pointers = gpu_context.get_group_kv_pointers(group_idx)

                    lmc_ops.multi_layer_block_kv_transfer(
                        group_kv_pointers,
                        [tb.data_ptr() for tb in tmp_buffers],
                        chunk_block_ids_gpu,
                        gpu_context.device,
                        lmc_ops.TransferDirection.H2D,
                        gpu_context.get_shape_desc(group_idx),
                        self.chunk_size,
                        gpu_context.gpu_kv_format_,
                        skip_blocks_in_chunk,
                    )

        with (
            torch.cuda.device(gpu_context.device),
            torch.cuda.stream(gpu_context.stream),
        ):
            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            event = torch.cuda.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            retrieve_succeeded = False
            try:
                with self.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    _retrieve_loop(obj_keys, memory_objs)
                # Only set True when with-block exits normally
                retrieve_succeeded = True
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                if retrieve_succeeded:
                    gpu_context.cupy_stream.launch_host_func(
                        self.storage_manager.finish_read_prefetched,
                        prefetched_keys,
                    )
                self._event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(gpu_context.device),
                        },
                    ),
                )
        tokens_retrieved = len(obj_keys) * self.chunk_size
        ed = time.perf_counter()
        logger.info(
            "Retrieved %d tokens in %.3f seconds",
            tokens_retrieved,
            ed - st,
        )

        return event.ipc_handle(), True

    # ------------------------------------------------------------------
    # KV tunneling — MARSHAL RPC and the workspace-driven retrieve path.
    # MARSHAL packs an already-stored prompt's KV on the fly into a
    # workspace blob; a later RETRIEVE carrying the same marshal_handle
    # scatters that blob into vLLM's paged cache instead of reading L1.
    # ------------------------------------------------------------------

    def marshal(
        self,
        marshal_handle: str,
        real_prompt: list[int],
        method_params: dict,
        worker_id: int,
    ) -> tuple[bool, int, str, dict[int, TunneledRequestMetadata]]:
        """Pack the unmarshalled KV for ``real_prompt`` into a workspace blob.

        Runs the StreamingLLM selection kernel on CPU: fetches the stored
        unmarshalled chunks for ``real_prompt``, copies the sink +
        sliding-window slots into a fresh pinned-CPU tensor with a header
        prepended, and parks the resulting MemoryObj in ``_WORKSPACE``
        keyed by ``marshal_handle``. A later RETRIEVE carrying the same
        ``marshal_handle`` scatters that blob into vLLM's paged cache.

        Args:
            marshal_handle: Rendezvous key used by the proxy to redeem the
                workspace entry via RETRIEVE.
            real_prompt: Token IDs of the real prompt whose KV is already
                stored unmarshalled in LMCache (populated by a prior normal
                completion — the miss path).
            method_params: Method-specific parameters. For MVP we honor
                only ``num_sinks`` (default 4), ``window_size`` (default
                1020), and ``cache_salt`` (default empty). Other keys are
                ignored; future methods will define their own schema.
            worker_id: GPU instance ID whose stored KV to look up. Must
                match a prior REGISTER_KV_CACHE call.

        Returns:
            ``(success, num_fake, error_message, tunneled_request_per_rank)``.
            On success ``num_fake`` is the number of fake slots the packed
            blob occupies, ``error_message`` is the empty string, and
            ``tunneled_request_per_rank`` maps ``tp_rank`` -> per-layer
            ``TunneledRequestMetadata`` manifest the connector stages on
            the scheduler so workers build attention metadata without
            re-parsing block bytes. On failure ``num_fake`` is 0, the
            manifest map is empty, and ``error_message`` describes why.
        """
        try:
            num_sinks = int(method_params.get("num_sinks", 4))
            window_size = int(method_params.get("window_size", 1020))
            cache_salt = str(method_params.get("cache_salt", ""))

            if worker_id not in self.gpu_context_meta:
                raise RuntimeError(
                    f"no GPU context registered for worker_id={worker_id}"
                )
            _, world_size = self.gpu_context_meta[worker_id]
            use_stub = os.environ.get("KVTUNNEL_STUB_MARSHAL") == "1"

            # Pack one workspace blob per TP rank. Each TP worker's
            # RETRIEVE later addresses its own blob via the worker_id
            # field on its IPCCacheEngineKey — see retrieve() dispatch.
            # For single-GPU world_size=1 this loop runs once.
            per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]] = {}
            tunneled_request_per_rank: dict[int, TunneledRequestMetadata] = {}
            num_fake = 0
            for tp_rank in range(world_size):
                mem_objs = self._fetch_unmarshalled_for_marshal(
                    real_prompt=real_prompt,
                    worker_id=worker_id,
                    tp_rank=tp_rank,
                    cache_salt=cache_salt,
                    marshal_handle=marshal_handle,
                )
                if mem_objs is None:
                    # Cold prompt — chunks not in L1. Return a clean
                    # miss so the proxy can fall back to passthrough
                    # without an exception + ERROR traceback.
                    return (
                        False,
                        0,
                        "unmarshalled KV not fully cached in L1",
                        {},
                    )
                if use_stub:
                    # Plumbing-validation mode; see stub_pack_for_plumbing
                    # for semantics. num_layers comes from the registered
                    # GPU context — needed so the stub stamps the magic
                    # header at every layer's byte-range start, not just
                    # layer 0's.
                    gpu_ctx = self.gpu_contexts[worker_id]
                    packed_list, manifest = stub_pack_for_plumbing(
                        workspace_allocator=self.kvtunnel_workspace_allocator,
                        orig_kv_obj=mem_objs,
                        chunk_size=self.chunk_size,
                        num_layers=gpu_ctx.num_layers,
                        block_size=gpu_ctx.block_size,
                    )
                    num_fake = manifest.per_layer[0].num_fake_marshalled
                else:
                    # Real pack: consume the GPU context's per-rank head
                    # geometry so the pack can validate the chunk's
                    # KV_2LTD shape and write the header's num_active_heads
                    # field. The pack emits a list of k chunk-sized
                    # MemoryObjs; the retrieve path scatters them via
                    # batched_iteration.
                    gpu_ctx = self.gpu_contexts[worker_id]
                    # max_chunks: 2× max_batch_size gives headroom past the
                    # kernel's 4-chunk-per-call cap (mp_mem_kernels.cu:262-263).
                    max_chunks = max(8, gpu_ctx.max_batch_size * 2)
                    logger.info(
                        "[kvtunnel CB] real-pack tp_rank=%d real_prompt_len=%d "
                        "chunk_size=%d block_size=%d num_sinks=%d window_size=%d "
                        "num_layers=%d num_kv_heads=%d head_size=%d "
                        "max_chunks=%d num_groups=%d is_mla=%s",
                        tp_rank,
                        len(real_prompt),
                        self.chunk_size,
                        gpu_ctx.block_size,
                        num_sinks,
                        window_size,
                        gpu_ctx.num_layers,
                        gpu_ctx.group_num_heads[0],
                        gpu_ctx.group_head_sizes[0],
                        max_chunks,
                        gpu_ctx.kv_layer_groups_manager.num_groups,
                        gpu_ctx.is_mla,
                    )
                    packed_list, manifest = streaming_llm_pack(
                        workspace_allocator=self.kvtunnel_workspace_allocator,
                        orig_kv_obj=mem_objs,
                        chunk_size=self.chunk_size,
                        real_prompt_len=len(real_prompt),
                        num_sinks=num_sinks,
                        window_size=window_size,
                        num_layers=gpu_ctx.num_layers,
                        num_kv_heads=gpu_ctx.group_num_heads[0],
                        head_size=gpu_ctx.group_head_sizes[0],
                        block_size=gpu_ctx.block_size,
                        max_chunks=max_chunks,
                        num_groups=(gpu_ctx.kv_layer_groups_manager.num_groups),
                        is_mla=gpu_ctx.is_mla,
                    )
                    num_fake = manifest.per_layer[0].num_fake_marshalled
                    logger.info(
                        "[kvtunnel CB] real-pack returned tp_rank=%d "
                        "num_fake=%d k=%d blob_logical_shape=%s "
                        "blob_dtype=%s blob_nbytes=%d",
                        tp_rank,
                        num_fake,
                        len(packed_list),
                        tuple(packed_list[0].meta.shape),
                        packed_list[0].raw_data.dtype,
                        sum(mo.get_size() for mo in packed_list),
                    )
                # allocate() already returns each chunk at ref_count=1 —
                # that single ref IS the workspace's ownership, so no
                # ref_count_up here (an extra ref would stop the Phase-3
                # MARSHAL_FREE ref_count_down from freeing). The manifest
                # sits next to the chunks in the workspace tuple — frozen
                # msgspec.Struct, no ref-count semantics, just stashed.
                per_rank[tp_rank] = (packed_list, manifest)
                tunneled_request_per_rank[tp_rank] = manifest
            with self._workspace_lock:
                _WORKSPACE[marshal_handle] = WorkspaceEntry(
                    mem_objs_per_rank=per_rank, instance_id=worker_id
                )
            logger.info(
                "MARSHAL handle=%s real_tokens=%d num_fake=%d ranks=%d%s",
                marshal_handle,
                len(real_prompt),
                num_fake,
                world_size,
                " (STUB)" if use_stub else "",
            )
            return (True, num_fake, "", tunneled_request_per_rank)
        except Exception as exc:  # noqa: BLE001 — surface error to client
            logger.exception("MARSHAL failed for handle=%s", marshal_handle)
            return (False, 0, str(exc), {})

    def _fetch_unmarshalled_for_marshal(
        self,
        real_prompt: list[int],
        worker_id: int,
        tp_rank: int,
        cache_salt: str,
        *,
        marshal_handle: str,
    ) -> list[MemoryObj] | None:
        """Fetch the unmarshalled KV chunks for one TP rank of ``real_prompt``.

        Uses the storage manager's prefetch-then-read pattern, same as the
        normal retrieve path. The caller (``marshal``) reads the chunks'
        raw_data *before* the storage locks are released — pack copies
        bytes into a fresh tensor so lifetime is safe.

        Args:
            real_prompt: Token IDs of the real prompt.
            worker_id: GPU instance ID whose stored KV to look up. Used
                only to route through the correct registered context; not
                part of the storage key.
            tp_rank: Tensor-parallel rank whose KV shard we want. Hashes
                into the object key via ``kv_rank`` — the adapter uses
                this field on ``IPCCacheEngineKey`` for the same purpose
                during STORE, so the keys must match.
            cache_salt: Per-user isolation salt matching the one used when
                the chunks were originally stored. Empty string matches
                unsalted entries.
            marshal_handle: Per-MARSHAL UUID from the proxy. Used as the
                scratch session key so concurrent MARSHAL calls can't
                collide on session-manager state. Replaces the previous
                ``id(real_prompt)`` keying, which was Python-object-id-
                based and could collide after GC reuse.

        Returns:
            The ordered list of MemoryObj chunks covering ``real_prompt``,
            or ``None`` if the prompt's chunks are not in L1 (cold
            prompt — normal operational state, not an error).

        Raises:
            RuntimeError: If the worker is unknown or its layout desc
                is missing.
        """
        if worker_id not in self.gpu_context_meta:
            raise RuntimeError(f"no GPU context registered for worker_id={worker_id}")
        model_name, world_size = self.gpu_context_meta[worker_id]
        layout_desc = self._find_layout_desc(model_name, world_size)
        if layout_desc is None:
            raise RuntimeError(
                f"no layout desc for model={model_name} world_size={world_size}"
            )

        scratch_key = f"__marshal__{marshal_handle}__{tp_rank}"
        session = self.session_manager.get_or_create(scratch_key)
        try:
            session.set_tokens(list(real_prompt))
            chunk_hashes = [
                TokenHasher.hash_to_bytes(h)
                for h in session.get_hashes(0, len(real_prompt))
            ]
            # IPCCacheEngineKey.worker_id is the TP rank, matching what the
            # worker adapter used when STOREing this shard (see
            # vllm_multi_process_adapter.py::_create_key).
            ipc_key = IPCCacheEngineKey(
                model_name=model_name,
                world_size=world_size,
                worker_id=tp_rank,
                token_ids=tuple(real_prompt),
                start=0,
                end=len(real_prompt),
                request_id=scratch_key,
                cache_salt=cache_salt,
            )
            obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)

            self.storage_manager.submit_prefetch_task(obj_keys, layout_desc)
            with self.storage_manager.read_prefetched_results(obj_keys) as mem_objs:
                if mem_objs is None:
                    # Cold prompt: chunks aren't in L1 yet. This is
                    # a normal operational state (first request for
                    # this prompt), not an error. Return None so the
                    # caller can report a clean cache-miss without
                    # an exception + ERROR-level traceback.
                    logger.info(
                        "MARSHAL miss: prompt chunks not in L1 "
                        "(cold prompt, %d chunks)",
                        len(obj_keys),
                    )
                    return None
                # streaming_llm_pack copies slot bytes into a fresh pinned-CPU
                # tensor, so the chunks' read locks can safely release after
                # this method returns. We return the list view — callers must
                # complete reads before exiting the parent `with`-block.
                return list(mem_objs)
        finally:
            # Scratch session is single-use: each MARSHAL gets a fresh
            # key (id(real_prompt) is ephemeral). Without this the
            # SessionManager dict grows unbounded until the 10-minute
            # TTL sweep — tracked as a sharp edge in
            # plan/real-streaming-llm-pack/design.md §6.4.
            self.session_manager.remove(scratch_key)

    def _retrieve_from_workspace(
        self,
        marshal_handle: str,
        tp_rank: int,
        instance_id: int,
        gpu_block_ids: list[int],
    ) -> tuple[bytes, bool]:
        """Scatter a workspace blob into vLLM's paged KV cache.

        Mirrors the chunk-scatter loop in :meth:`retrieve` but operates on
        a single MemoryObj (the marshalled blob) treated as one batched
        chunk. Integration with a real GPU context is validated in Phase 6;
        Phase 2 unit tests monkey-patch this method to assert the
        workspace lookup fires without spinning up CUDA.

        Args:
            marshal_handle: Key into ``_WORKSPACE``; the caller guarantees
                it is present.
            tp_rank: Which per-rank blob to pick from the workspace entry.
                Matches the ``worker_id`` on the incoming
                ``IPCCacheEngineKey`` that STORE originally used.
            instance_id: GPU instance ID; must have a registered context.
            gpu_block_ids: Paged-cache block IDs that receive the blob.

        Returns:
            tuple[bytes, bool]: CUDA event IPC handle and success flag,
            same shape as :meth:`retrieve`.
        """
        per_rank = _WORKSPACE[marshal_handle].mem_objs_per_rank
        if tp_rank not in per_rank:
            raise RuntimeError(
                f"marshal_handle={marshal_handle} has no blob for "
                f"tp_rank={tp_rank}; "
                f"available ranks={sorted(per_rank.keys())}"
            )
        # Workspace stores (chunks, metadata) per-rank since Phase 1
        # of plan/tunneled-metadata-for-cuda-graph/. Retrieve only
        # needs the chunks here; metadata flows through the MARSHAL
        # response to the proxy + connector.
        mem_objs, _manifest = per_rank[tp_rank]
        if instance_id not in self.gpu_contexts:
            raise RuntimeError(f"KV cache not registered for GPU ID {instance_id}")
        gpu_context = self.gpu_contexts[instance_id]

        # Multi-chunk scatter: the pack emits k chunk-sized MemoryObjs.
        # The kernel `multi_layer_block_kv_transfer` hard-asserts
        # `num_objects ≤ 4` (`mp_mem_kernels.cu:262-263`) AND
        # `gpu_context.max_batch_size = 4` (`gpu_context.py:151`). For
        # k > 4 we issue ceil(k / batch_size) separate kernel launches,
        # each staging up to 4 chunks via `batched_iteration` — the
        # same pattern the regular RETRIEVE uses at `server.py:495-557`.
        k = len(mem_objs)
        batch_size = gpu_context.max_batch_size
        blocks_per_chunk = self.chunk_size // gpu_context.block_size
        expected_block_count = k * blocks_per_chunk
        if len(gpu_block_ids) != expected_block_count:
            raise RuntimeError(
                f"gpu_block_ids count mismatch: got {len(gpu_block_ids)}, "
                f"expected k*blocks_per_chunk = {k}*{blocks_per_chunk} = "
                f"{expected_block_count} (k chunks × chunk_size / block_size). "
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
                torch.cuda.device(gpu_context.device),
                torch.cuda.stream(gpu_context.stream),
            ):
                all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)
                event = torch.cuda.Event(interprocess=True)
                num_groups = gpu_context.kv_layer_groups_manager.num_groups

                # Outer loop: iterate batches of ≤ batch_size chunks. Each
                # iteration stages its chunks into staging slots
                # 0..batch_len-1, then issues one scatter call per KV
                # layer group. Mirrors `server.py:495-557` exactly.
                for batch_idx, mem_obj_batch in enumerate(
                    batched_iteration(mem_objs, batch_size=batch_size)
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
                    # h2d wrapper's size-equality check at gpu_ops.py:30
                    # passes without any slicing.
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
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tb.data_ptr() for tb in tmp_buffers],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.H2D,
                            gpu_context.get_shape_desc(group_idx),
                            self.chunk_size,
                            gpu_context.gpu_kv_format_,
                            0,
                        )

                event.record()
        except Exception:
            # Surface the exception explicitly so it's grep-able in the
            # MP log before mq._notify_response swallows the response
            # (mq.py:418-433).
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

    def _find_layout_desc(
        self,
        model_name: str,
        world_size: int,
    ) -> MemoryLayoutDesc | None:
        """Find layout desc from a matching GPU context.

        Returns:
            The layout descriptor, or None if no context
            matches (model_name, world_size).
        """
        for gpu_id, (m, w) in self.gpu_context_meta.items():
            if m == model_name and w == world_size:
                return get_layout_desc(
                    self.gpu_contexts[gpu_id],
                    self.chunk_size,
                )
        return None

    def lookup(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Tensor-parallel size for MLA multi-reader locking.
        """
        model_name, world_size = key.model_name, key.world_size
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_desc = self._find_layout_desc(model_name, world_size)
        if layout_desc is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                )
            )
            return

        extra_count = compute_extra_count(tp_size, world_size)

        # Compute chunk hashes for all full chunks
        chunk_hashes = self.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handle=PrefetchHandle(
                        prefetch_request_id=-1,
                        external_request_id=key.request_id,
                        l1_prefix_hit_count=0,
                        total_requested_keys=0,
                        submit_time=time.monotonic(),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                )
            )
            return

        # Publish lookup event via EventBus for observability subscribers.
        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(d) for d in layout_desc.dtypes],
                        "shapes": [list(s) for s in layout_desc.shapes],
                    },
                )
            )

        # set lookup ipc key, for session manager to use and generate object keys
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key

        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        handle = self.storage_manager.submit_prefetch_task(
            obj_keys,
            layout_desc,
            extra_count=extra_count,
            external_request_id=key.request_id,
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handle=handle,
                world_size=key.world_size,
                request_id=key.request_id,
            )
        )

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def query_prefetch_lookup_hits(
        self,
        request_id: str,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_count = self.storage_manager.query_prefetch_lookup_hits(job.handle)
        if found_count is None:
            return None

        found_count = found_count // job.world_size
        return found_count

    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_count = self.storage_manager.query_prefetch_status(job.handle)
        if found_count is None:
            return None

        # NOTE(Kuntai): this assumes two things:
        # 1. the world size is the same between keys
        # 2. the lookup sort the keys in prefix order and breaks at the
        #    first failure
        found_count = found_count // job.world_size

        self._event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={"found_count": found_count},
            )
        )

        with self._prefetch_job_lock:
            self._prefetch_jobs.pop(request_id, None)

        return found_count

    def free_lookup_locks(
        self,
        key: IPCCacheEngineKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Computes the extra reader count from ``tp_size`` and
        ``world_size`` the same way :meth:`lookup` does, so
        the correct number of locks is released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA
                multi-reader locking.
        """
        chunk_hashes = self.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=key.end
        )
        if not chunk_hashes:
            return
        obj_keys = ipc_key_to_object_keys(key, chunk_hashes)

        extra_count = compute_extra_count(tp_size, key.world_size)

        self.storage_manager.finish_read_prefetched(obj_keys, extra_count=extra_count)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def ping(self) -> bool:
        """
        Respond to a ping request.

        Returns:
            bool: Always True.
        """
        return True

    def get_chunk_size(self) -> int:
        """
        Returns the chunk size used for KV cache operations.

        Returns:
            int: The chunk size.
        """
        return self.chunk_size

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self.session_manager.remove(request_id)
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_SESSION_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch", request_id
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = ipc_key_to_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self.storage_manager.touch_l1_keys(obj_keys)

    def marshal_free(self, marshal_handle: str) -> None:
        """Reclaim the KV-tunnel workspace entry for ``marshal_handle``.

        Fired by the proxy once the request/cycle that consumed the blob
        has finished (closes O1 — the never-freed-workspace leak). Pops
        the entry under ``_workspace_lock``, then schedules the per-chunk
        ``ref_count_down`` as a stream-ordered host callback on the
        packing context's stream — the STORE finalize idiom at
        ``_store_kv``'s ``launch_host_func`` — so a freed chunk's pinned
        bytes are never reclaimed while an in-flight RETRIEVE H2D is
        still draining. (The normal proxy path fires this only after the
        vLLM completion returns, i.e. after the H2D has drained, so it is
        already safe by timing; the stream-ordering is defense for the
        abort path.) The handler does pop + enqueue ONLY — the actual
        free runs later on the cupy callback thread — so it stays O(µs)
        and never blocks the shared CPU pool. Returns as soon as the free
        is *enqueued*; the ack does NOT mean the buffer is reclaimed.
        Unknown / already-freed handle is a no-op.

        Args:
            marshal_handle: Workspace entry to reclaim.
        """
        with self._workspace_lock:
            entry = _WORKSPACE.pop(marshal_handle, None)
        if entry is None:
            return  # unknown / already freed — no-op

        all_chunks = [
            mem_obj
            for chunks, _manifest in entry.mem_objs_per_rank.values()
            for mem_obj in chunks
        ]

        def _drop(objs: list[MemoryObj]) -> None:
            for mem_obj in objs:
                mem_obj.ref_count_down()  # 1 → 0 → parent_allocator.free

        gpu_context = self.gpu_contexts.get(entry.instance_id)
        if gpu_context is None:
            # Context already unregistered (teardown) — no DMA can be in
            # flight against it, so free inline.
            _drop(all_chunks)
            return
        gpu_context.cupy_stream.launch_host_func(_drop, all_chunks)

    # ----------------------------------------------------------------
    # WAIT_STORE — gate the proxy's next MARSHAL on the previous
    # cycle's STORE having committed to L1. See
    # plan/proxy-re-tunnel-cycles/design.md §"WAIT_STORE handler".
    # ----------------------------------------------------------------

    def _signal_chunk_stores(self, chunk_hashes: list[bytes]) -> None:
        """Wake any waiters registered on the given chunk hashes.

        ``pop()`` under ``_pending_lock``, then ``set()`` outside the
        lock. STORE-side and WAIT_STORE-side never deadlock on the
        notifier (no callback runs inside ``e.set()``).

        Args:
            chunk_hashes: Hashes whose pending waiters to signal.
        """
        for chunk_hash in chunk_hashes:
            with self._pending_lock:
                events = self._pending_chunk_events.pop(chunk_hash, [])
            for e in events:
                e.set()

    def _finish_write_and_signal(self, keys: list[ObjectKey]) -> None:
        """CUDA host-callback wrapper around ``finish_write``.

        Runs on the cupy stream's host-callback thread. CUDA host
        callbacks must NOT raise — use ``try/finally`` so the signal
        fires even if ``finish_write`` itself raises (waiters then
        re-check is_ready, see False, fall through to "Pending"
        instead of hanging until their deadline).

        Args:
            keys: Object keys whose writes have just finished on the
                GPU; corresponds to the ``reserved_dict.keys()``
                argument that the original launch_host_func used.
        """
        chunk_hashes = [k.chunk_hash for k in keys]
        try:
            self.storage_manager.finish_write(keys)
        except Exception:
            logger.exception("finish_write raised; signaling waiters")
        finally:
            self._signal_chunk_stores(chunk_hashes)

    def wait_store(
        self,
        token_ids: list[int],
        end_offset: int,
        worker_id: int,
        wait_timeout_ms: int,
    ) -> str:
        """Block until ``token_ids[0:end_offset]``'s last chunk is
        committed and readable on every TP rank, or timeout.

        See ``plan/proxy-re-tunnel-cycles/design.md`` §"WAIT_STORE
        handler" for the full design rationale (race walkthrough,
        per-rank expansion, exception-path handling).

        Args:
            token_ids: Running real prompt (prompt + decoded so far).
            end_offset: Length of the running prompt; the handler
                hashes ``[0:end_offset]`` and waits on the trailing
                chunk_hash.
            worker_id: GPU instance ID; used to look up
                ``gpu_context_meta`` to get model_name + world_size.
            wait_timeout_ms: ``event.wait`` deadline in milliseconds.

        Returns:
            ``"Ready"`` if the chunk is readable on every TP rank
            within the deadline; ``"Pending"`` otherwise.

        Raises:
            RuntimeError: If ``worker_id`` is not registered.
        """
        if worker_id not in self.gpu_context_meta:
            raise RuntimeError(f"no GPU context registered for worker_id={worker_id}")

        # UUID4 session key — id(token_ids) is non-unique under GC
        # reuse and can collide between concurrent waiters.
        session_uuid = uuid.uuid4().hex
        session_key = f"__wait_store__{session_uuid}__{worker_id}"
        session = self.session_manager.get_or_create(session_key)
        event: threading.Event | None = None
        target_hash: bytes | None = None
        try:
            session.set_tokens(list(token_ids))
            chunk_hashes = [
                TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0, end_offset)
            ]
            target_hash = chunk_hashes[-1]  # only the trailing chunk

            # Cross-rank expansion: worker_id=None makes
            # ipc_key_to_object_keys iterate range(world_size) and
            # emit one ObjectKey per TP rank for the same chunk
            # hash. cache_salt="" matches the connector tracker's
            # default-empty salt and the proxy's strip-helper
            # contract (no cache_salt on the wire from any cycle
            # body).
            model_name, world_size = self.gpu_context_meta[worker_id]
            ipc_key = IPCCacheEngineKey(
                model_name=model_name,
                world_size=world_size,
                worker_id=None,
                token_ids=tuple(token_ids),
                start=0,
                end=end_offset,
                request_id=session_key,
                cache_salt="",
            )
            obj_keys = ipc_key_to_object_keys(ipc_key, [target_hash])

            if all(self.storage_manager.is_ready(obj_keys)):
                return "Ready"

            # Register an Event BEFORE the second is_ready check.
            # Closes the race where finish_write completes between
            # the first is_ready (returns False) and registration:
            # even if signal_chunk_stores fires before we register,
            # the second is_ready below sees the post-finish_write
            # state and returns Ready.
            event = threading.Event()
            with self._pending_lock:
                self._pending_chunk_events.setdefault(target_hash, []).append(event)

            if all(self.storage_manager.is_ready(obj_keys)):
                return "Ready"

            if event.wait(timeout=wait_timeout_ms / 1000.0):
                # Re-check is_ready after wakeup. The Event may
                # have been set by the wrapped finish_write's
                # exception-path signal: signal fires on exception
                # but write_lock is still held → not readable.
                # Without this re-check we'd return Ready
                # spuriously and the next MARSHAL would fail.
                if all(self.storage_manager.is_ready(obj_keys)):
                    return "Ready"
                return "Pending"
            return "Pending"
        finally:
            # Always remove our Event so _pending_chunk_events
            # doesn't leak, on every exit path.
            if event is not None and target_hash is not None:
                with self._pending_lock:
                    waiters = self._pending_chunk_events.get(target_hash)
                    if waiters is not None and event in waiters:
                        waiters.remove(event)
                        if not waiters:
                            self._pending_chunk_events.pop(target_hash, None)
            self.session_manager.remove(session_key)

    def report_status(self) -> dict:
        """Return a status dict for the entire cache engine."""
        sm = self.storage_manager.report_status()

        gpu_context_meta: dict[str, dict] = {}
        for gpu_id, meta in self.gpu_context_meta.items():
            entry: dict = {
                "model_name": meta[0],
                "world_size": meta[1],
            }
            ctx = self.gpu_contexts.get(gpu_id)
            if ctx is not None:
                entry["kv_cache_layout"] = {
                    "num_layers": ctx.num_layers,
                    "block_size": ctx.block_size,
                    "hidden_dim_sizes": str(ctx.hidden_dim_sizes),
                    "dtype": str(ctx.dtype),
                    "is_mla": ctx.is_mla,
                    "num_blocks": ctx.num_blocks,
                    "gpu_kv_format": ctx.gpu_kv_format_name,
                    "gpu_kv_shape": ctx.gpu_kv_shape,
                    "gpu_kv_concrete_shape": ctx.concrete_gpu_kv_shape,
                    "attention_backend": ctx.attention_backend,
                    "cache_size_per_token": ctx.cache_size_per_token(),
                }
            gpu_context_meta[str(gpu_id)] = entry

        return {
            "is_healthy": sm["is_healthy"],
            "engine_type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "hash_algorithm": self.token_hasher.hash_algorithm_name,
            "registered_gpu_ids": list(self.gpu_contexts.keys()),
            "gpu_context_meta": gpu_context_meta,
            "active_sessions": self.session_manager.active_count(),
            "active_prefetch_jobs": self._active_prefetch_count(),
            "storage_manager": sm,
        }

    def report_block_allocations(
        self,
        instance_id: int,
        model_name: str,
        records: list[BlockAllocationRecord],
    ) -> None:
        """Publish vLLM block allocation records to the EventBus.

        Args:
            instance_id: The scheduler instance ID.
            model_name: The model name from the adapter.
            records: List of BlockAllocationRecord with per-request
                block and token allocation deltas.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_BLOCK_ALLOCATION,
                metadata={
                    "instance_id": instance_id,
                    "model_name": model_name,
                    "records": records,
                },
            )
        )

    def debug(self) -> str:
        return "OK"

    def clear(self) -> None:
        """
        Clears all stored KV cache data from the storage manager.
        """
        with self.lock:
            self.storage_manager.memcheck()
            self.storage_manager.clear(force=True)
            self.storage_manager.memcheck()

    def close(self) -> None:
        """
        Closes the MPCacheEngine and releases all resources.
        """
        # Close storage manager
        self.storage_manager.close()

        # Free the kvtunnel workspace pool (unregisters the pinned host
        # buffer; one-time torch.cuda.synchronize on teardown).
        self.kvtunnel_workspace_allocator.close()
        logger.info("MPCacheEngine closed")

        # Release GPU contexts
        self.gpu_contexts.clear()

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for MP engine metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_engine")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )


def add_handler_helper(
    server: MessageQueueServer, request_type: RequestType, handler_function
):
    payload_classes = get_payload_classes(request_type)
    handler_type = get_handler_type(request_type)
    server.add_handler(
        request_type,
        payload_classes,
        handler_type,
        handler_function,
    )


def run_cache_server(
    mp_config: MPServerConfig,
    storage_manager_config: StorageManagerConfig,
    obs_config: ObservabilityConfig,
    return_engine: bool = False,
):
    """
    Run the LMCache cache server with ZMQ message queue.

    Args:
        mp_config: Configuration for the ZMQ multiprocess server
        storage_manager_config: Configuration for the storage manager
        obs_config: Configuration for the observability stack
        return_engine: If True, return (server, engine) after starting;
                       if False, run blocking loop to keep server alive

    Returns:
        If return_engine is True: tuple of (MessageQueueServer, MPCacheEngine)
        If return_engine is False: None (blocks until interrupted)
    """
    event_bus = init_observability(obs_config)

    # Wire up the trace recorder (no-op when --trace-level is unset).
    # Registered before the engine handlers are added so any
    # storage-manager calls during engine init are captured too.
    maybe_initialize_trace_recorder(event_bus, obs_config, storage_manager_config)

    # Initialize the engine (loggers self-register with the global controller)
    engine = MPCacheEngine(
        storage_manager_config=storage_manager_config,
        chunk_size=mp_config.chunk_size,
        hash_algorithm=mp_config.hash_algorithm,
    )

    # Initialize the message queue server
    context = zmq.Context.instance()
    server = MessageQueueServer(
        bind_url=f"tcp://{mp_config.host}:{mp_config.port}",
        context=context,
    )

    # Add handlers
    add_handler_helper(server, RequestType.REGISTER_KV_CACHE, engine.register_kv_cache)
    add_handler_helper(
        server, RequestType.UNREGISTER_KV_CACHE, engine.unregister_kv_cache
    )
    add_handler_helper(server, RequestType.STORE, engine.store)
    add_handler_helper(server, RequestType.LOOKUP, engine.lookup)
    add_handler_helper(
        server, RequestType.QUERY_PREFETCH_STATUS, engine.query_prefetch_status
    )
    add_handler_helper(
        server,
        RequestType.QUERY_PREFETCH_LOOKUP_HITS,
        engine.query_prefetch_lookup_hits,
    )
    add_handler_helper(server, RequestType.FREE_LOOKUP_LOCKS, engine.free_lookup_locks)
    add_handler_helper(server, RequestType.RETRIEVE, engine.retrieve)
    add_handler_helper(server, RequestType.MARSHAL, engine.marshal)
    add_handler_helper(server, RequestType.MARSHAL_FREE, engine.marshal_free)
    add_handler_helper(server, RequestType.WAIT_STORE, engine.wait_store)
    add_handler_helper(server, RequestType.CLEAR, engine.clear)
    add_handler_helper(server, RequestType.GET_CHUNK_SIZE, engine.get_chunk_size)
    add_handler_helper(server, RequestType.PING, engine.ping)
    add_handler_helper(server, RequestType.END_SESSION, engine.end_session)
    add_handler_helper(server, RequestType.NOOP, engine.debug)
    add_handler_helper(
        server,
        RequestType.REPORT_BLOCK_ALLOCATION,
        engine.report_block_allocations,
    )

    # Assign thread pools
    server.add_affinity_thread_pool(
        [RequestType.STORE, RequestType.RETRIEVE],
        max_workers=mp_config.max_gpu_workers,
    )
    server.add_normal_thread_pool(
        [
            RequestType.LOOKUP,
            RequestType.QUERY_PREFETCH_STATUS,
            RequestType.QUERY_PREFETCH_LOOKUP_HITS,
            RequestType.FREE_LOOKUP_LOCKS,
            RequestType.END_SESSION,
            RequestType.CLEAR,
            RequestType.PING,
            RequestType.REPORT_BLOCK_ALLOCATION,
            RequestType.MARSHAL,
            RequestType.MARSHAL_FREE,
        ],
        max_workers=mp_config.max_cpu_workers,
    )
    # WAIT_STORE on its own pool so its potentially-long blocking
    # waits (up to repack_wait_store_max_timeout_ms ≈ 30 s with
    # backoff) can't head-of-line-block MARSHAL or other CPU-pool
    # work. Pool size is preliminary; Phase 5 measures queue depth +
    # busy-fraction + per-call hold-time histogram and resizes per
    # plan/proxy-re-tunnel-cycles/design.md §"Pool registration".
    server.add_normal_thread_pool(
        [RequestType.WAIT_STORE],
        max_workers=8,
    )

    logger.info(
        "LMCache ZMQ cache server is running on tcp://%s:%d",
        mp_config.host,
        mp_config.port,
    )
    # Start the ZMQ server
    torch.cuda.init()
    server.start()

    logger.info("LMCache cache server is running...")

    # Return server and engine if requested (for HTTP server integration)
    if return_engine:
        return server, engine

    # Dummy loop to keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        event_bus.stop()
        server.close()
        engine.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache ZMQ Cache Server (without HTTP)"
    )
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    add_observability_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp_config = parse_args_to_mp_server_config(args)
    storage_manager_config = parse_args_to_config(args)
    obs_config = parse_args_to_observability_config(args)
    run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )
