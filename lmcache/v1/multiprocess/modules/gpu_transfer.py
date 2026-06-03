# SPDX-License-Identifier: Apache-2.0
"""GPU-based KV cache transfer operations for the MPCacheEngine."""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from typing import Generator, Iterator
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

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
    check_interprocess_event_support,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryAllocatorInterface, MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.gpu_context import GPUCacheContext
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.c_ops as lmc_ops

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

    Raises:
        ValueError: If batch_size is less than 1.
    """
    if batch_size < 1:
        raise ValueError("batch size must be at least one")
    it = iter(lst)
    while batch := tuple(islice(it, batch_size)):
        yield batch


@dataclass
class GPUContextEntry:
    """Registered GPU context metadata for a single worker instance.

    Args:
        gpu_context: The GPU cache context managing shape and pointers
            to vLLM GPU KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
    """

    gpu_context: GPUCacheContext
    model_name: str
    world_size: int


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


class GPUTransferModule:
    """Handles GPU-based KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheEngineContext) -> None:
        self._ctx = ctx
        self._gpu_contexts: dict[int, GPUContextEntry] = {}

        # kvtunnel MARSHAL workspace pool — a dedicated LazyMemoryAllocator
        # kept separate from the StorageManager's L1 allocator so the two
        # don't share an eviction policy. Pinned at construction from a
        # fixed byte budget: KVTUNNEL_WORKSPACE_POOL_GB (default 8);
        # init=pool by default so the whole pool is pinned eagerly and
        # Lazy's background thread no-ops. The pack writes into it;
        # MARSHAL_FREE reclaims via _workspace_lock. Lives on this module
        # (GPU-mode-gated) rather than the shared context so a non-GPU
        # server never pins a pool it cannot use.
        pool_gb = float(os.environ.get("KVTUNNEL_WORKSPACE_POOL_GB", "8"))
        pool_bytes = int(pool_gb * (1 << 30))
        init_gb_env = os.environ.get("KVTUNNEL_WORKSPACE_INIT_GB")
        init_bytes = int(float(init_gb_env) * (1 << 30)) if init_gb_env else pool_bytes
        self.kvtunnel_workspace_allocator: MemoryAllocatorInterface = (
            LazyMemoryAllocator(init_size=init_bytes, final_size=pool_bytes)
        )
        self._workspace_lock = threading.Lock()
        # Per-process workspace for KV-tunneled MARSHAL -> RETRIEVE
        # rendezvous, keyed by marshal_handle. Mutated by the MARSHAL
        # handler (write) and MARSHAL_FREE (pop), both under
        # ``_workspace_lock``; RETRIEVE only reads.
        self._WORKSPACE: dict[str, WorkspaceEntry] = {}
        logger.info(
            "kvtunnel workspace pool: %d B (init %d B); Max locked memory=%s",
            pool_bytes,
            init_bytes,
            _max_locked_memory(),
        )

        # WAIT_STORE notifier: chunk_hash -> list[Event] of waiters. The
        # finish_write wrapper (post-DMA, drain thread) signals the
        # waiters. No EventBus dependency — sub-ms wakeup, no thundering
        # herd.
        self._pending_chunk_events: dict[bytes, list[threading.Event]] = {}
        self._pending_lock = threading.Lock()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        # finish_write is wrapped so WAIT_STORE waiters are signalled
        # AFTER the L1 write completes. The wrapper keeps its OWN
        # try/finally and signals in finally: the dispatcher's outer
        # except (native_completion.py) swallows a finish_write exception
        # BEFORE any signal, so relying on it would hang waiters until
        # their deadline.
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._finish_write_and_signal,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.start()

    @property
    def context(self) -> MPCacheEngineContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    @property
    def gpu_contexts(self) -> dict[int, GPUContextEntry]:
        """Per-instance GPU context registry."""
        return self._gpu_contexts

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE,
                self.register_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.STORE,
                self.store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.RETRIEVE,
                self.retrieve,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.MARSHAL,
                self.marshal,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.MARSHAL_FREE,
                self.marshal_free,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.WAIT_STORE,
                self.wait_store,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        gpu_context_meta: dict[str, dict] = {}

        for instance_id, entry in self._gpu_contexts.items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.gpu_context
            gpu_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": {
                    "num_layers": ctx.num_layers,
                    "inference_engine_logical_block_size": (
                        ctx.kv_layer_groups_manager.inference_engine_logical_block_size
                    ),
                    "group_physical_block_sizes": ctx.group_physical_block_sizes,
                    "group_compress_ratios": ctx.group_compress_ratios,
                    "hidden_dim_sizes": str(ctx.hidden_dim_sizes),
                    "dtype": str(ctx.dtype),
                    "is_mla": ctx.is_mla,
                    "num_blocks": ctx.num_blocks,
                    "gpu_kv_format": ctx.gpu_kv_format_name,
                    "gpu_kv_shape": ctx.gpu_kv_shape,
                    "gpu_kv_concrete_shape": ctx.concrete_gpu_kv_shape,
                    "attention_backend": ctx.attention_backend,
                    "cache_size_per_token": ctx.cache_size_per_token(),
                },
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "gpu_context_meta": gpu_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        # Release the pinned kvtunnel workspace pool.
        self.kvtunnel_workspace_allocator.close()

        had_contexts = len(self._gpu_contexts) > 0
        self._gpu_contexts.clear()
        if had_contexts:
            torch_dev.empty_cache()

    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
        """
        if instance_id in self._gpu_contexts:
            logger.warning(
                "Instance %s's KV cache is already registered, "
                "skipping the new registration",
                instance_id,
            )
            return

        gpu_context = GPUCacheContext(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_type=engine_type,
        )
        self._gpu_contexts[instance_id] = GPUContextEntry(
            gpu_context=gpu_context,
            model_name=model_name,
            world_size=world_size,
        )

        layout_desc = get_layout_desc(gpu_context, self._ctx.chunk_size)
        self._ctx.layout_desc_registry.register(model_name, world_size, layout_desc)

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            gpu_context.num_layers,
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        entry = self._gpu_contexts.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)
        torch_dev.empty_cache()

    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheEngineKey,
        instance_id: int,
        gpu_block_ids: list[int],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: The GPU block IDs to store.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation was successful.

        Raises:
            ValueError: If no GPU context is registered for the given instance ID.
            RuntimeError: If the backend does not support IPC event handles.
        """
        st = time.perf_counter()
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._gpu_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        gpu_context = entry.gpu_context
        model_name = entry.model_name

        # ``blocks_per_chunk`` is counted in inference-engine-side
        # blocks (each block addresses
        # ``inference_engine_logical_block_size`` *logical* tokens).
        # For compressed groups the per-group physical slot count
        # differs, but the block-id indexing is shared with the engine
        # and therefore uses the engine logical block size here.
        blocks_per_chunk = (
            self._ctx.chunk_size
            // gpu_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            if not hasattr(torch_dev.Event, "from_ipc_handle"):
                raise RuntimeError(
                    f"Backend '{torch_device_type}' does not support IPC event "
                    "handles (Event.from_ipc_handle not available). "
                    "Multiprocess IPC requires CUDA."
                )
            vllm_event = torch_dev.Event.from_ipc_handle(
                gpu_context.device, event_ipc_handle
            )
            vllm_event.wait(stream=gpu_context.stream)

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(gpu_context.device)},
                )
            )

            self._ctx.event_bus.publish_on_stream(
                gpu_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(gpu_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            try:
                layout_desc = get_layout_desc(gpu_context, self._ctx.chunk_size)
                reserved_dict = self._ctx.storage_manager.reserve_write(
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
                        # Kernel contract: ``group_lmcache_chunk_size`` here is the
                        # number of *physical* slots per chunk for this group
                        # (= logical chunk_size // compress_ratio).
                        group_lmcache_chunk_size = gpu_context.get_physical_chunk_size(
                            group_idx
                        )
                        lmc_ops.multi_layer_block_kv_transfer(
                            group_kv_pointers,
                            [tmp_buffer.data_ptr()],
                            chunk_block_ids_gpu,
                            gpu_context.device,
                            lmc_ops.TransferDirection.D2H,
                            gpu_context.get_shape_desc(group_idx),
                            group_lmcache_chunk_size,
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
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_write",
                        list(reserved_dict.keys()),
                    )
                # All reserved MemoryObjs share one layout_desc, so per-object
                # size is identical — avoid summing N identical values.
                total_bytes = (
                    next(iter(reserved_dict.values())).get_size() * len(reserved_dict)
                    if reserved_dict
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": len(reserved_dict),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                        },
                    ),
                )

        ed = time.perf_counter()
        if length := len(reserved_dict):
            logger.info(
                "Stored %d tokens in %.3f seconds",
                length * self._ctx.chunk_size,
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
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: The GPU block IDs to retrieve into.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.
            marshal_handle: Rendezvous key for a KV-tunneled request.
                When non-empty and present in ``_WORKSPACE``, the packed
                marshalled blob stashed there by a prior MARSHAL RPC is
                scattered into ``gpu_block_ids`` instead of reading from
                storage. Empty string (default) falls through to the
                standard storage path.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.

        Raises:
            ValueError: If no GPU context is registered for the given instance ID.
        """
        if marshal_handle and marshal_handle in self._WORKSPACE:
            # TP rank comes from the incoming key — each TP worker's
            # RETRIEVE carries its own worker_id, matching the per-rank
            # workspace entry produced by marshal(). See _WORKSPACE docs.
            return self._retrieve_from_workspace(
                marshal_handle=marshal_handle,
                tp_rank=key.worker_id or 0,
                instance_id=instance_id,
                gpu_block_ids=gpu_block_ids,
            )

        st = time.perf_counter()
        obj_keys = self._ctx.resolve_obj_keys(key)

        entry = self._gpu_contexts.get(instance_id)
        if entry is None:
            raise ValueError(f"No GPU context registered for instance ID {instance_id}")
        gpu_context = entry.gpu_context
        model_name = entry.model_name

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(gpu_context.device)},
            )
        )

        self._ctx.event_bus.publish_on_stream(
            gpu_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(gpu_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                },
            ),
        )

        # ``skip_*_in_chunk`` is expressed in engine-block units
        # (logical tokens), which is what the kernel's
        # ``skip_blocks_in_chunk`` argument expects regardless
        # of per-group compression.
        ie_logical_block_size = (
            gpu_context.kv_layer_groups_manager.inference_engine_logical_block_size
        )
        blocks_per_chunk = self._ctx.chunk_size // ie_logical_block_size

        def _retrieve_loop(keys: list[ObjectKey], memory_objs: list[MemoryObj]) -> None:
            _BATCH_SIZE = gpu_context.max_batch_size
            num_groups = gpu_context.kv_layer_groups_manager.num_groups
            for batch_idx, memory_obj_batch in enumerate(
                batched_iteration(memory_objs, batch_size=_BATCH_SIZE)
            ):
                batch_len = len(memory_obj_batch)
                chunk_start = batch_idx * self._ctx.chunk_size * _BATCH_SIZE
                chunk_end = chunk_start + self._ctx.chunk_size * batch_len

                effective_start = max(chunk_start, skip_first_n_tokens)
                if effective_start >= chunk_end:
                    # Entire batch is within APC range, skip it
                    continue

                skip_tokens_in_chunk = max(
                    0,
                    min(
                        effective_start - chunk_start,
                        self._ctx.chunk_size * batch_len - 1,
                    ),
                )
                if skip_tokens_in_chunk % ie_logical_block_size != 0:
                    logger.error(
                        "skip_first_n_tokens (%d) is not aligned to "
                        "inference_engine_logical_block_size (%d), "
                        "rounding down from %d tokens to %d blocks",
                        skip_first_n_tokens,
                        ie_logical_block_size,
                        skip_tokens_in_chunk,
                        skip_tokens_in_chunk // ie_logical_block_size,
                    )
                skip_blocks_in_chunk = skip_tokens_in_chunk // ie_logical_block_size

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
                        skip_blocks_in_chunk,
                    )

        with (
            torch_dev.device(gpu_context.device),
            torch_dev.stream(gpu_context.stream),
        ):
            # Stage all block_ids to GPU once before the loop
            all_block_ids_gpu = gpu_context.stage_block_ids(gpu_block_ids)

            check_interprocess_event_support()
            event = torch_dev.Event(interprocess=True)

            prefetched_keys: list[ObjectKey] = []
            retrieve_succeeded = False
            total_bytes = 0
            try:
                with self._ctx.storage_manager.read_prefetched_results(
                    obj_keys
                ) as memory_objs:
                    if not memory_objs or len(memory_objs) != len(obj_keys):
                        logger.error("Some keys not found during retrieve!")
                        return event.ipc_handle(), False

                    prefetched_keys = obj_keys[: len(memory_objs)]
                    total_bytes = sum(mo.get_size() for mo in memory_objs)
                    _retrieve_loop(obj_keys, memory_objs)
                # Only set True when with-block exits normally
                retrieve_succeeded = True
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                return event.ipc_handle(), False
            finally:
                event.record()
                if retrieve_succeeded:
                    submit_callback_to_stream(
                        gpu_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                self._ctx.event_bus.publish_on_stream(
                    gpu_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(gpu_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                        },
                    ),
                )
        tokens_retrieved = len(obj_keys) * self._ctx.chunk_size
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
        sliding-window slots into k fresh pinned-CPU chunk tensors (header
        in chunk 0), and parks the resulting list of MemoryObjs in
        ``_WORKSPACE`` keyed by ``marshal_handle``. A later RETRIEVE carrying
        the same ``marshal_handle`` scatters that blob into vLLM's paged cache.

        Args:
            marshal_handle: Rendezvous key used by the proxy to redeem the
                workspace entry via RETRIEVE.
            real_prompt: Token IDs of the real prompt whose KV is already
                stored unmarshalled in LMCache (populated by a prior normal
                completion — the miss path).
            method_params: Method-specific parameters. Only ``num_sinks``
                (default 4), ``window_size`` (default 1020), and
                ``cache_salt`` (default empty) are honored; other keys are
                ignored.
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

            entry = self._gpu_contexts.get(worker_id)
            if entry is None:
                raise RuntimeError(
                    f"no GPU context registered for worker_id={worker_id}"
                )
            world_size = entry.world_size
            use_stub = os.environ.get("KVTUNNEL_STUB_MARSHAL") == "1"

            # Pack one workspace blob per TP rank. Each TP worker's
            # RETRIEVE later addresses its own blob via the worker_id
            # field on its IPCCacheEngineKey — see retrieve() dispatch.
            # For single-GPU world_size=1 this loop runs once.
            per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]] = {}
            tunneled_request_per_rank: dict[int, TunneledRequestMetadata] = {}
            num_fake = 0
            for tp_rank in range(world_size):
                with self._fetch_unmarshalled_for_marshal(
                    real_prompt=real_prompt,
                    worker_id=worker_id,
                    tp_rank=tp_rank,
                    cache_salt=cache_salt,
                    marshal_handle=marshal_handle,
                ) as mem_objs:
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
                    gpu_ctx = entry.gpu_context
                    kvlgm = gpu_ctx.kv_layer_groups_manager
                    ie_block_size = kvlgm.inference_engine_logical_block_size
                    if use_stub:
                        # Plumbing-validation mode; see stub_pack_for_plumbing
                        # for semantics. num_layers comes from the registered
                        # GPU context — needed so the stub stamps the magic
                        # header at every layer's byte-range start, not just
                        # layer 0's.
                        packed_list, manifest = stub_pack_for_plumbing(
                            workspace_allocator=self.kvtunnel_workspace_allocator,
                            orig_kv_obj=mem_objs,
                            chunk_size=self._ctx.chunk_size,
                            num_layers=gpu_ctx.num_layers,
                            block_size=ie_block_size,
                        )
                        num_fake = manifest.per_layer[0].num_fake_marshalled
                    else:
                        # Real pack: consume the GPU context's per-rank head
                        # geometry so the pack can validate the chunk's
                        # KV_2LTD shape and write the header's num_active_heads
                        # field. The pack emits a list of k chunk-sized
                        # MemoryObjs; the retrieve path scatters them via
                        # batched_iteration.
                        shape_desc = gpu_ctx.get_shape_desc(0)
                        # max_chunks: 2x max_batch_size gives headroom past the
                        # kernel's 4-chunk-per-call cap (mp_mem_kernels.cu).
                        max_chunks = max(8, gpu_ctx.max_batch_size * 2)
                        logger.info(
                            "[kvtunnel CB] real-pack tp_rank=%d real_prompt_len=%d "
                            "chunk_size=%d block_size=%d num_sinks=%d window_size=%d "
                            "num_layers=%d num_kv_heads=%d head_size=%d "
                            "max_chunks=%d num_groups=%d is_mla=%s",
                            tp_rank,
                            len(real_prompt),
                            self._ctx.chunk_size,
                            ie_block_size,
                            num_sinks,
                            window_size,
                            gpu_ctx.num_layers,
                            shape_desc.nh,
                            shape_desc.hs,
                            max_chunks,
                            gpu_ctx.kv_layer_groups_manager.num_groups,
                            gpu_ctx.is_mla,
                        )
                        packed_list, manifest = streaming_llm_pack(
                            workspace_allocator=self.kvtunnel_workspace_allocator,
                            orig_kv_obj=mem_objs,
                            chunk_size=self._ctx.chunk_size,
                            real_prompt_len=len(real_prompt),
                            num_sinks=num_sinks,
                            window_size=window_size,
                            num_layers=gpu_ctx.num_layers,
                            num_kv_heads=shape_desc.nh,
                            head_size=shape_desc.hs,
                            block_size=ie_block_size,
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
                    # ref_count_up here (an extra ref would stop the
                    # MARSHAL_FREE ref_count_down from freeing). The manifest
                    # sits next to the chunks in the workspace tuple — frozen
                    # msgspec.Struct, no ref-count semantics, just stashed.
                    per_rank[tp_rank] = (packed_list, manifest)
                    tunneled_request_per_rank[tp_rank] = manifest
            with self._workspace_lock:
                self._WORKSPACE[marshal_handle] = WorkspaceEntry(
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

    @contextmanager
    def _fetch_unmarshalled_for_marshal(
        self,
        real_prompt: list[int],
        worker_id: int,
        tp_rank: int,
        cache_salt: str,
        *,
        marshal_handle: str,
    ) -> Iterator[list[MemoryObj] | None]:
        """Context manager: yield the unmarshalled KV chunks for one TP
        rank of ``real_prompt`` while holding their L1 read lock.

        Uses the storage manager's prefetch-then-read pattern, same as the
        normal retrieve path. The read lock is held across the caller's
        (``marshal``) with-body — the pack copies the chunks' bytes into a
        fresh pinned tensor there — and released on the success path before
        the context exits; on the cold-miss / pack-exception path the inner
        ``read_prefetched_results`` context releases it instead.

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

        Yields:
            The ordered list of MemoryObj chunks covering ``real_prompt``,
            or ``None`` if the prompt's chunks are not in L1 (cold
            prompt — normal operational state, not an error).

        Raises:
            RuntimeError: If the worker is unknown or its layout desc
                is missing.
        """
        entry = self._gpu_contexts.get(worker_id)
        if entry is None:
            raise RuntimeError(f"no GPU context registered for worker_id={worker_id}")
        model_name = entry.model_name
        world_size = entry.world_size
        layout_desc = self._ctx.layout_desc_registry.find(model_name, world_size)
        if layout_desc is None:
            raise RuntimeError(
                f"no layout desc for model={model_name} world_size={world_size}"
            )

        scratch_key = f"__marshal__{marshal_handle}__{tp_rank}"
        session = self._ctx.session_manager.get_or_create(scratch_key)
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

            self._ctx.storage_manager.submit_prefetch_task(obj_keys, layout_desc)
            with self._ctx.storage_manager.read_prefetched_results(
                obj_keys
            ) as mem_objs:
                if mem_objs is None:
                    # Cold prompt: chunks aren't in L1 yet. A normal
                    # operational state (first request for this prompt),
                    # not an error. Yield None so marshal() reports a clean
                    # cache-miss; the read_prefetched_results context
                    # releases the partial prefix on the way out.
                    logger.info(
                        "MARSHAL miss: prompt chunks not in L1 "
                        "(cold prompt, %d chunks)",
                        len(obj_keys),
                    )
                    yield None
                    return
                # The pack runs in marshal()'s with-body while this read
                # lock is held; it copies the source bytes into fresh
                # pinned-CPU workspace tensors (a synchronous CPU
                # slice-assign), so once the with-body returns the source
                # is fully copied and the lock can be released eagerly (no
                # stream deferral, unlike the async-H2D retrieve path).
                yield list(mem_objs)
                # Reached only if marshal()'s with-body completed without
                # raising: release the source read lock. Success-path only,
                # exactly-once vs read_prefetched_results' finally, which
                # releases only on the miss/exception path.
                self._ctx.storage_manager.finish_read_prefetched(
                    obj_keys, extra_count=0
                )
        finally:
            # Scratch session is single-use: each MARSHAL gets a fresh
            # key. Without this remove, the SessionManager dict grows
            # unbounded until the 10-minute TTL sweep.
            self._ctx.session_manager.remove(scratch_key)

    def _retrieve_from_workspace(
        self,
        marshal_handle: str,
        tp_rank: int,
        instance_id: int,
        gpu_block_ids: list[int],
    ) -> tuple[bytes, bool]:
        """Scatter a workspace blob into vLLM's paged KV cache.

        Scatters the k chunk-sized MemoryObjs the pack emitted, in batches
        of <= max_batch_size, reusing the same chunk-scatter loop as
        :meth:`retrieve`. Unit tests monkey-patch this method to assert the
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

        Raises:
            RuntimeError: If the requested rank has no blob, the instance
                is unregistered, or the block count mismatches.
        """
        per_rank = self._WORKSPACE[marshal_handle].mem_objs_per_rank
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
        entry = self._gpu_contexts.get(instance_id)
        if entry is None:
            raise RuntimeError(f"KV cache not registered for GPU ID {instance_id}")
        gpu_context = entry.gpu_context

        # Multi-chunk scatter: the pack emits k chunk-sized MemoryObjs.
        # The kernel `multi_layer_block_kv_transfer` hard-asserts
        # `num_objects <= 4` AND `gpu_context.max_batch_size = 4`. For
        # k > 4 we issue ceil(k / batch_size) separate kernel launches,
        # each staging up to 4 chunks via `batched_iteration` — the same
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

    def marshal_free(self, marshal_handle: str) -> None:
        """Reclaim the KV-tunnel workspace entry for ``marshal_handle``.

        Fired by the proxy once the request/cycle that consumed the blob
        has finished. Pops the entry under ``_workspace_lock``, then
        schedules the per-chunk ``ref_count_down`` as a stream-ordered host
        callback on the packing context's stream (the STORE finalize idiom)
        so a freed chunk's pinned bytes are never reclaimed while an
        in-flight RETRIEVE H2D is still draining. The normal proxy path
        fires this only after the vLLM completion returns, i.e. after the
        H2D has drained, so it is already safe by timing; the
        stream-ordering is defense for the abort path. The handler does pop
        + enqueue ONLY — the actual free runs later on the cupy callback
        thread — so it stays O(us) and never blocks the shared CPU pool.
        Returns as soon as the free is *enqueued*; the ack does NOT mean
        the buffer is reclaimed. Unknown / already-freed handle is a no-op.

        Args:
            marshal_handle: Workspace entry to reclaim.
        """
        with self._workspace_lock:
            entry = self._WORKSPACE.pop(marshal_handle, None)
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

        gpu_entry = self._gpu_contexts.get(entry.instance_id)
        if gpu_entry is None:
            # Context already unregistered (teardown) — no DMA can be in
            # flight against it, so free inline.
            _drop(all_chunks)
            return
        gpu_entry.gpu_context.cupy_stream.launch_host_func(_drop, all_chunks)

    # ----------------------------------------------------------------
    # WAIT_STORE — gate the proxy's next MARSHAL on the previous
    # cycle's STORE having committed to L1.
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
        """Host-callback wrapper around ``finish_write`` for WAIT_STORE.

        Runs on the DeviceHostFuncDispatcher drain thread. The dispatcher's
        outer except swallows a handler exception BEFORE any signal could
        fire, so this wrapper keeps its OWN try/finally and signals the
        waiters in ``finally`` even if ``finish_write`` raises (waiters then
        re-check is_ready, see False, fall through to "Pending" instead of
        hanging until their deadline).

        Args:
            keys: Object keys whose writes have just finished on the
                GPU; corresponds to the ``reserved_dict.keys()`` argument
                that store() submits to the dispatcher.
        """
        chunk_hashes = [k.chunk_hash for k in keys]
        try:
            self._ctx.storage_manager.finish_write(keys)
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

        Only the trailing chunk hash is waited on; it is expanded
        across all TP ranks (worker_id=None) since each rank stores
        its own shard. The inline comments below walk through the
        registration/is_ready race and the exception-path re-check.

        Args:
            token_ids: Running real prompt (prompt + decoded so far).
            end_offset: Length of the running prompt; the handler
                hashes ``[0:end_offset]`` and waits on the trailing
                chunk_hash.
            worker_id: GPU instance ID; used to look up the registered
                context to get model_name + world_size.
            wait_timeout_ms: ``event.wait`` deadline in milliseconds.

        Returns:
            ``"Ready"`` if the chunk is readable on every TP rank
            within the deadline; ``"Pending"`` otherwise.

        Raises:
            RuntimeError: If ``worker_id`` is not registered.
        """
        entry = self._gpu_contexts.get(worker_id)
        if entry is None:
            raise RuntimeError(f"no GPU context registered for worker_id={worker_id}")

        # UUID4 session key — id(token_ids) is non-unique under GC
        # reuse and can collide between concurrent waiters.
        session_uuid = uuid.uuid4().hex
        session_key = f"__wait_store__{session_uuid}__{worker_id}"
        session = self._ctx.session_manager.get_or_create(session_key)
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
            model_name = entry.model_name
            world_size = entry.world_size
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

            if all(self._ctx.storage_manager.is_ready(obj_keys)):
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

            if all(self._ctx.storage_manager.is_ready(obj_keys)):
                return "Ready"

            if event.wait(timeout=wait_timeout_ms / 1000.0):
                # Re-check is_ready after wakeup. The Event may
                # have been set by the wrapped finish_write's
                # exception-path signal: signal fires on exception
                # but write_lock is still held -> not readable.
                # Without this re-check we'd return Ready
                # spuriously and the next MARSHAL would fail.
                if all(self._ctx.storage_manager.is_ready(obj_keys)):
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
            self._ctx.session_manager.remove(session_key)
