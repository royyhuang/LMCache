# SPDX-License-Identifier: Apache-2.0
"""KV-tunnel MARSHAL engine module.

The kvtunnel-only half of what used to live in ``GPUTransferModule``:
the MARSHAL / MARSHAL_FREE / WAIT_STORE handlers. MARSHAL packs an
already-stored prompt's KV on the fly into a pinned workspace blob
(via :class:`MarshalWorkspace`, published on ``ctx.marshal_workspace``);
a later RETRIEVE carrying the same ``marshal_handle`` scatters that blob
into vLLM's paged cache instead of reading L1. WAIT_STORE gates the
proxy's next MARSHAL on the previous cycle's STORE having committed to
L1, waiting on ``ctx.chunk_commit_notifier`` (signalled by
``GPUTransferModule``'s finish-write callback).

This module reaches all shared state through the ctx seams
(``ctx.gpu_context_registry``, ``ctx.chunk_commit_notifier``,
``ctx.marshal_workspace``), so it depends only on ``ctx`` — same as
every other engine module.
"""

# Standard
from contextlib import contextmanager
from typing import Iterator
import os
import uuid

from kvtunnel.marshal.pack import (
    TunneledRequestMetadata,
    streaming_llm_pack,
    stub_pack_for_plumbing,
)

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ipc_key_to_object_keys
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.engine_context import MPCacheEngineContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.marshal_workspace import (
    MarshalWorkspace,
    WorkspaceEntry,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


class MarshalModule:
    """Handles KV-tunnel MARSHAL operations.

    Owns the MARSHAL workspace (publishes it onto
    ``ctx.marshal_workspace``) and provides handlers for the MARSHAL,
    MARSHAL_FREE, and WAIT_STORE request types. Reads the GPU-context
    registry and waits on the chunk-commit notifier, both via ``ctx``.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheEngineContext) -> None:
        self._ctx = ctx

        # kvtunnel MARSHAL workspace — owns the pinned pool, the
        # marshal_handle -> blob dict, and the tunneled-RETRIEVE scatter.
        # Constructed here (MarshalModule is GPU-mode-gated by
        # _build_modules, so a non-GPU server never pins a pool it cannot
        # use) and published onto the shared context so the RETRIEVE
        # delegation in GPUTransferModule reaches it via
        # ``ctx.marshal_workspace``. The GPU-context registry and the
        # WAIT_STORE chunk-commit notifier also live on ctx (see
        # GPUContextRegistry / ChunkCommitNotifier).
        self._ctx.marshal_workspace = MarshalWorkspace(self._ctx)

    @property
    def context(self) -> MPCacheEngineContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
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
        """Return MARSHAL module status information.

        Returns:
            An empty dict. Deliberately emits no keys (and in particular
            NOT ``registered_gpu_ids``, which stays GPUTransferModule's):
            MPCacheEngine.report_status merges every module's dict via
            ``status.update``, so any colliding key would clobber another
            module's value. The MARSHAL workspace exposes no public
            entry-count accessor, so there is nothing to report here.
        """
        return {}

    def close(self) -> None:
        """Free the kvtunnel workspace pool.

        Must run AFTER GPUTransferModule.close (which stops the
        DeviceHostFuncDispatcher drain thread that fires the finish-write
        signal), so the dispatcher cannot touch the pinned host buffers
        this frees. MPCacheEngine.close iterates modules in list order and
        _build_modules appends MarshalModule after GPUTransferModule, which
        is what guarantees that ordering.

        :meth:`MarshalWorkspace.close` drains any in-flight MARSHAL_FREE
        ``ref_count_down`` host callbacks (across every packing device, for
        TP>1) before unpinning the pool, so freeing here cannot decrement
        into a closed allocator.
        """
        if self._ctx.marshal_workspace is not None:
            self._ctx.marshal_workspace.close()
            self._ctx.marshal_workspace = None

    # ------------------------------------------------------------------
    # KV tunneling — MARSHAL RPC and the workspace pack path.
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
        ``ctx.marshal_workspace`` keyed by ``marshal_handle``. A later
        RETRIEVE carrying the same ``marshal_handle`` scatters that blob
        into vLLM's paged cache.

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

            entry = self._ctx.gpu_context_registry.get(worker_id)
            if entry is None:
                raise RuntimeError(
                    f"no GPU context registered for worker_id={worker_id}"
                )
            world_size = entry.world_size
            use_stub = os.environ.get("KVTUNNEL_STUB_MARSHAL") == "1"

            # MARSHAL runs only in GPU mode, where __init__ published the
            # workspace; assert non-None so the pack allocator + put() reads
            # below are type-clean.
            workspace = self._ctx.marshal_workspace
            if workspace is None:
                raise RuntimeError("MARSHAL workspace is not initialized")
            workspace_allocator = workspace.kvtunnel_workspace_allocator

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
                            workspace_allocator=workspace_allocator,
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
                            workspace_allocator=workspace_allocator,
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
            workspace.put(
                marshal_handle,
                WorkspaceEntry(mem_objs_per_rank=per_rank, instance_id=worker_id),
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
        entry = self._ctx.gpu_context_registry.get(worker_id)
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

    def marshal_free(self, marshal_handle: str) -> None:
        """Reclaim the KV-tunnel workspace entry for ``marshal_handle``.

        Fired by the proxy once the request/cycle that consumed the blob
        has finished. Delegates to :meth:`MarshalWorkspace.free` (the
        workspace owns the pop + stream-ordered ``ref_count_down``);
        unknown / already-freed handle is a no-op there. No-op as well when
        no workspace is published (non-GPU server).

        Args:
            marshal_handle: Workspace entry to reclaim.
        """
        workspace = self._ctx.marshal_workspace
        if workspace is None:
            return
        workspace.free(marshal_handle)

    # ----------------------------------------------------------------
    # WAIT_STORE — gate the proxy's next MARSHAL on the previous
    # cycle's STORE having committed to L1.
    # ----------------------------------------------------------------

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
        entry = self._ctx.gpu_context_registry.get(worker_id)
        if entry is None:
            raise RuntimeError(f"no GPU context registered for worker_id={worker_id}")

        # UUID4 session key — id(token_ids) is non-unique under GC
        # reuse and can collide between concurrent waiters.
        session_uuid = uuid.uuid4().hex
        session_key = f"__wait_store__{session_uuid}__{worker_id}"
        session = self._ctx.session_manager.get_or_create(session_key)
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

            # Register an Event BEFORE the second is_ready check (the
            # notifier's register() context manager owns the
            # registration + leak-free removal). Closes the race where
            # finish_write completes between the first is_ready (returns
            # False) and registration: even if the notifier signal fires
            # before we register, the second is_ready below sees the
            # post-finish_write state and returns Ready.
            with self._ctx.chunk_commit_notifier.register(target_hash) as event:
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
            self._ctx.session_manager.remove(session_key)
