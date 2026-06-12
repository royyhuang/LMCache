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
    MAX_HEADER_BLOCKS,
    TunneledRequestMetadata,
    streaming_llm_pack,
    stub_pack_for_plumbing,
)

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
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

# BYTE-STABLE cache-miss message: the proxy classifies MARSHAL errors by
# substring-matching a PREFIX of this text (_CACHE_MISS_MARKER in
# kvtunnel/proxy/server.py = "unmarshalled KV not fully cached") to choose
# passthrough over a 503. Qualifiers may be APPENDED, but the prefix must
# never be reworded.
_L1_MISS_MSG = "unmarshalled KV not fully cached in L1"


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
    ) -> tuple[bool, int, str, dict[int, TunneledRequestMetadata], int]:
        """Pack the unmarshalled KV for ``real_prompt``'s longest cached
        prefix into a workspace blob.

        Runs the StreamingLLM selection kernel on CPU: fetches the stored
        unmarshalled chunks for the longest chunk-aligned prefix of
        ``real_prompt`` present in L1 (the full prompt when fully cached),
        copies the sink + sliding-window slots into k fresh pinned-CPU
        chunk tensors (header in chunk 0), and parks the resulting list of
        MemoryObjs in ``ctx.marshal_workspace`` keyed by
        ``marshal_handle``. A later RETRIEVE carrying the same
        ``marshal_handle`` scatters that blob into vLLM's paged cache.

        Partial (shorter-than-prompt) matches are gated behind
        ``KVTUNNEL_PARTIAL_PREFIX=1`` (default off: the proxy cannot yet
        forward the unmatched suffix, so a partial match reports a clean
        cache-miss exactly like the pre-partial behavior). When enabled, a
        partial match at or below the compression floor (``num_sinks +
        window_size + MAX_HEADER_BLOCKS * block_size``) reports a clean
        cache-miss rather than packing. For TP>1, a probe pass takes the
        MIN matched length across ranks and every rank packs at it; a
        rank evicted between the passes reports a clean cache-miss.

        Args:
            marshal_handle: Rendezvous key used by the proxy to redeem the
                workspace entry via RETRIEVE.
            real_prompt: Token IDs of the real prompt whose KV is already
                stored unmarshalled in LMCache (populated by a prior normal
                completion — the miss path).
            method_params: Method-specific parameters. Only ``num_sinks``
                (default 4), ``window_size`` (default 1020),
                ``cache_salt`` (default empty), and ``allow_partial``
                (default False — the caller must opt in to receive a
                partial-prefix success; callers that can't forward the
                unmatched suffix, e.g. the chat and cycles proxy paths,
                never set it) are honored; other keys are ignored.
            worker_id: GPU instance ID whose stored KV to look up. Must
                match a prior REGISTER_KV_CACHE call.

        Returns:
            ``(success, num_fake, error_message, tunneled_request_per_rank,
            matched_prefix_len)``. On success ``num_fake`` is the number of
            fake slots the packed blob occupies, ``error_message`` is the
            empty string, ``tunneled_request_per_rank`` maps ``tp_rank`` ->
            per-layer ``TunneledRequestMetadata`` manifest the connector
            stages on the scheduler so workers build attention metadata
            without re-parsing block bytes, and ``matched_prefix_len`` is
            the chunk-aligned token count of the prefix that was actually
            packed (== the chunk-aligned length of ``real_prompt`` on a
            full match — the proxy pre-truncates to a chunk multiple, so
            in practice == ``len(real_prompt)``; the proxy derives the
            unmatched suffix from it). On failure ``num_fake``
            and ``matched_prefix_len`` are 0, the manifest map is empty,
            and ``error_message`` describes why.
        """
        # Hoisted above the try so the cleanup below can always reference
        # them: blobs packed for earlier ranks must be freed on EVERY
        # non-success exit (mid-loop clean miss or exception). allocate()
        # returns each chunk at ref_count=1 — that ref is the workspace's
        # ownership ONLY once workspace.put publishes the entry; until
        # then this function owns it, and dropping it without
        # ref_count_down would leak pinned pool bytes (reclaimed only by
        # the GC __del__ safety net, which warns).
        per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]] = {}
        published = False

        def _free_packed() -> None:
            for objs, _manifest in per_rank.values():
                for mem_obj in objs:
                    mem_obj.ref_count_down()
            per_rank.clear()

        try:
            num_sinks = int(method_params.get("num_sinks", 4))
            allow_partial = bool(method_params.get("allow_partial", False))
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

            # Partial-prefix tunneling is gated until the proxy can build
            # the [num_fake + K-real-suffix] dummy prompt (Phases 3-4): a
            # partial SUCCESS today would make the proxy drop every suffix
            # token but the last — a silent wrong-answer. Default-off; the
            # Phase-4 proxy flips it on.
            partial_enabled = os.environ.get("KVTUNNEL_PARTIAL_PREFIX", "0") == "1"

            gpu_ctx = entry.gpu_context
            kvlgm = gpu_ctx.kv_layer_groups_manager
            ie_block_size = kvlgm.inference_engine_logical_block_size
            # Compression floor for a PARTIAL match (see the guard below):
            # at or below it StreamingLLM retains (nearly) the whole
            # prefix, so num_fake can reach matched_prefix_len + header
            # and the pack hard-fails on delta < 0. The header term
            # matters — retained alone clearing sinks+window still leaves
            # num_fake > matched when the block-aligned header pushes
            # content into one more chunk.
            partial_floor = num_sinks + window_size + MAX_HEADER_BLOCKS * ie_block_size

            # Pack one workspace blob per TP rank. Each TP worker's
            # RETRIEVE later addresses its own blob via the worker_id
            # field on its IPCCacheEngineKey — see retrieve() dispatch.
            # For single-GPU world_size=1 this loop runs once.
            tunneled_request_per_rank: dict[int, TunneledRequestMetadata] = {}
            num_fake = 0
            total_chunks = len(real_prompt) // self._ctx.chunk_size
            # Longest-prefix match. All ranks must pack the SAME matched
            # length so num_fake is identical across ranks (the proxy
            # sends ONE num_fake). Single rank: one pass, the fetch's own
            # longest prefix IS the match. TP>1: two passes — probe every
            # rank's hit count first (locks released immediately), take
            # the min, then fetch each rank capped at that min. A rank
            # evicted between the passes surfaces as a shrink below the
            # cap -> clean miss (accepted TOCTOU).
            limit_chunks = 0
            if world_size > 1 and partial_enabled and allow_partial:
                # Gate-off OR a non-opted-in caller skips the probe:
                # all-or-nothing requires every rank to match the FULL
                # prompt, which the fetch pass checks by itself (matched
                # < total -> clean miss), so the probe would pay an extra
                # submit+hash round per rank for an identical outcome.
                rank_hits = [
                    self._probe_prefix_hit_count(
                        real_prompt=real_prompt,
                        worker_id=worker_id,
                        tp_rank=tp_rank,
                        cache_salt=cache_salt,
                        marshal_handle=marshal_handle,
                    )
                    for tp_rank in range(world_size)
                ]
                limit_chunks = min(rank_hits)
                if limit_chunks == 0:
                    return (False, 0, _L1_MISS_MSG, {}, 0)
                if limit_chunks < total_chunks:
                    logger.info(
                        "MARSHAL TP min-prefix: rank hits %s -> packing "
                        "all %d ranks at %d chunks",
                        rank_hits,
                        world_size,
                        limit_chunks,
                    )
            # Safety net: every rank's pack must emit the same num_fake
            # (guaranteed when all ranks pack the same matched length;
            # checked anyway so a divergence corrupts nothing).
            expected_num_fake = -1
            matched_prefix_len = 0
            for tp_rank in range(world_size):
                with self._fetch_unmarshalled_for_marshal(
                    real_prompt=real_prompt,
                    worker_id=worker_id,
                    tp_rank=tp_rank,
                    cache_salt=cache_salt,
                    marshal_handle=marshal_handle,
                    limit_chunks=limit_chunks,
                ) as mem_objs:
                    if mem_objs is None:
                        # Cold prompt — no leading chunk in L1. Return a
                        # clean miss so the proxy can fall back to
                        # passthrough without an exception + traceback.
                        _free_packed()
                        return (False, 0, _L1_MISS_MSG, {}, 0)
                    matched_chunks = len(mem_objs)
                    matched_prefix_len = matched_chunks * self._ctx.chunk_size
                    if limit_chunks > 0 and matched_chunks != limit_chunks:
                        # TP two-pass TOCTOU: this rank's prefix shrank
                        # (eviction) between the probe pass and this
                        # fetch — it can no longer supply the cross-rank
                        # min. Fail clean to passthrough.
                        _free_packed()
                        return (
                            False,
                            0,
                            _L1_MISS_MSG + " (rank shrank below min:"
                            f" rank {tp_rank} matched {matched_chunks}"
                            f" chunks, min {limit_chunks})",
                            {},
                            0,
                        )
                    if matched_chunks < total_chunks:
                        if not partial_enabled:
                            # Gate off: preserve today's all-or-nothing
                            # behavior exactly (partial -> clean miss).
                            _free_packed()
                            return (
                                False,
                                0,
                                _L1_MISS_MSG + " (partial-prefix tunneling disabled)",
                                {},
                                0,
                            )
                        if not allow_partial:
                            # Caller didn't opt in: a partial success
                            # would silently truncate its answer (it
                            # cannot forward the unmatched suffix).
                            # Clean miss, exactly like gate-off.
                            _free_packed()
                            return (
                                False,
                                0,
                                _L1_MISS_MSG + " (caller did not allow partial match)",
                                {},
                                0,
                            )
                        # Treat a too-short partial match (at or below the
                        # floor computed above) as a clean miss
                        # (byte-stable message — the proxy's cache-miss
                        # classifier substring-matches it) instead of a
                        # delta<0 pack exception that would surface as a
                        # 503. Full matches keep today's behavior.
                        if matched_prefix_len <= partial_floor:
                            _free_packed()
                            return (
                                False,
                                0,
                                _L1_MISS_MSG + " (partial prefix at or below"
                                " compression floor)",
                                {},
                                0,
                            )
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
                            "matched_prefix_len=%d "
                            "chunk_size=%d block_size=%d num_sinks=%d window_size=%d "
                            "num_layers=%d num_kv_heads=%d head_size=%d "
                            "max_chunks=%d num_groups=%d is_mla=%s",
                            tp_rank,
                            len(real_prompt),
                            matched_prefix_len,
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
                        # Pack the MATCHED prefix, not the full prompt:
                        # real_prompt_len anchors the StreamingLLM window
                        # at the match boundary and sets next_real_pos /
                        # delta so the forwarded suffix RoPEs at positions
                        # matched_prefix_len.. (full match: identical to
                        # the old len(real_prompt) behavior).
                        packed_list, manifest = streaming_llm_pack(
                            workspace_allocator=workspace_allocator,
                            orig_kv_obj=mem_objs,
                            chunk_size=self._ctx.chunk_size,
                            real_prompt_len=matched_prefix_len,
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
                    # Inserted BEFORE the num_fake net below so a divergence
                    # return's _free_packed() covers THIS rank's blobs too.
                    per_rank[tp_rank] = (packed_list, manifest)
                    tunneled_request_per_rank[tp_rank] = manifest
                    if expected_num_fake < 0:
                        expected_num_fake = num_fake
                    elif num_fake != expected_num_fake:
                        # Should be impossible (all ranks pack the same
                        # matched length and num_fake depends only on it)
                        # — checked so a divergence corrupts nothing: the
                        # proxy sends ONE num_fake for all ranks.
                        _free_packed()
                        return (
                            False,
                            0,
                            _L1_MISS_MSG + " (per-rank num_fake"
                            f" divergence: rank {tp_rank} packed"
                            f" {num_fake}, expected"
                            f" {expected_num_fake})",
                            {},
                            0,
                        )
            workspace.put(
                marshal_handle,
                WorkspaceEntry(mem_objs_per_rank=per_rank, instance_id=worker_id),
            )
            published = True
            logger.info(
                "MARSHAL handle=%s real_tokens=%d matched_prefix=%d "
                "num_fake=%d ranks=%d%s",
                marshal_handle,
                len(real_prompt),
                matched_prefix_len,
                num_fake,
                world_size,
                " (STUB)" if use_stub else "",
            )
            return (
                True,
                num_fake,
                "",
                tunneled_request_per_rank,
                matched_prefix_len,
            )
        except Exception as exc:  # noqa: BLE001 — surface error to client
            logger.exception("MARSHAL failed for handle=%s", marshal_handle)
            if not published:
                # e.g. rank 1's pack raised after rank 0 packed: free the
                # orphaned blobs (ownership only transfers at put()).
                _free_packed()
            return (False, 0, str(exc), {}, 0)

    @contextmanager
    def _fetch_unmarshalled_for_marshal(
        self,
        real_prompt: list[int],
        worker_id: int,
        tp_rank: int,
        cache_salt: str,
        *,
        marshal_handle: str,
        limit_chunks: int = 0,
    ) -> Iterator[list[MemoryObj] | None]:
        """Context manager: yield the unmarshalled KV chunks for one TP
        rank of ``real_prompt`` while holding their L1 read lock.

        Uses the storage manager's prefetch-then-read pattern, same as the
        normal retrieve path. The read lock is held across the caller's
        (``marshal``) with-body — the pack copies the chunks' bytes into a
        fresh pinned tensor there — and released on the success path before
        the context exits; on the eviction-race / pack-exception path the
        inner ``read_prefetched_results`` context releases it instead (a
        zero-hit cold miss never enters the read, so nothing is held).

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
            limit_chunks: Cap on how many leading chunks to read. 0 (the
                default, single-rank path) means no cap — yield the full
                longest prefix. The TP two-pass passes the cross-rank min
                here so every rank reads the SAME prefix length; surplus
                locks beyond the cap are released before the read.

        Yields:
            The ordered list of MemoryObj chunks covering the longest
            chunk-aligned PREFIX of ``real_prompt`` present in L1, capped
            at ``limit_chunks`` when set (may be shorter than the full
            prompt — the caller derives ``matched_prefix_len =
            len(chunks) * chunk_size``), or ``None`` if no leading chunk
            is in L1 (cold prompt — normal operational state, not an
            error).

        Raises:
            RuntimeError: If the worker is unknown or its layout desc
                is missing.
        """
        with self._object_keys_for_prompt(
            real_prompt=real_prompt,
            worker_id=worker_id,
            tp_rank=tp_rank,
            cache_salt=cache_salt,
            scratch_key=f"__marshal__{marshal_handle}__{tp_rank}",
        ) as (obj_keys, layout_desc):
            # Submit on the FULL key list: submit_prefetch_task walks the
            # keys in order, breaks at the first hole, returns the
            # longest-prefix hit count on the handle, and auto-releases the
            # read locks it reserved for any keys past that hole — so
            # truncating BELOW (not here) leaves no dangling locks.
            # L1-ONLY ASSUMPTION: with an L2 adapter configured,
            # submit_prefetch_task would ALSO fire an L2 prefetch for the
            # post-hole keys whose handle we drop — those land into L1
            # holding read locks nobody releases. kvtunnel servers run
            # L1-only today; revisit before enabling L2 adapters here.
            handle = self._ctx.storage_manager.submit_prefetch_task(
                obj_keys, layout_desc
            )
            hit_count = handle.l1_prefix_hit_count
            if hit_count == 0:
                # Cold prompt: no leading chunk in L1 yet. A normal
                # operational state (first request for this prompt), not
                # an error. Yield None so marshal() reports a clean
                # cache-miss. No locks held: a zero-hit submit reserved
                # nothing (or auto-released what it briefly reserved).
                logger.info(
                    "MARSHAL miss: prompt chunks not in L1 (cold prompt, %d chunks)",
                    len(obj_keys),
                )
                yield None
                return
            take = hit_count
            if 0 < limit_chunks < hit_count:
                # TP two-pass (pass 2): this rank matched more than the
                # cross-rank min — release the surplus locks NOW (submit
                # reserved [0:hit_count]; only [0:limit_chunks] will be
                # read + released below) and read only the common prefix.
                self._ctx.storage_manager.finish_read_prefetched(
                    obj_keys[limit_chunks:hit_count], extra_count=0
                )
                take = limit_chunks
            # Longest-prefix partial match: read ONLY the matched prefix.
            # Truncating before the read keeps the lock bookkeeping
            # aligned — read_prefetched_results on the truncated list
            # yields all-good (every key was L1-reserved above), and the
            # explicit finish_read below releases exactly those keys.
            matched_keys = obj_keys[:take]
            with self._ctx.storage_manager.read_prefetched_results(
                matched_keys
            ) as mem_objs:
                if mem_objs is None:
                    # Reserve-to-read eviction race on the matched prefix
                    # (rare): the inner context released the good keys.
                    # Report a clean miss, same as cold.
                    logger.info(
                        "MARSHAL miss: matched prefix evicted between "
                        "reserve and read (%d/%d chunks)",
                        take,
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
                # raising: release the source read lock for exactly the
                # truncated prefix that was read. Success-path only,
                # exactly-once vs read_prefetched_results' finally, which
                # releases only on the miss/exception path.
                self._ctx.storage_manager.finish_read_prefetched(
                    matched_keys, extra_count=0
                )

    @contextmanager
    def _object_keys_for_prompt(
        self,
        real_prompt: list[int],
        worker_id: int,
        tp_rank: int,
        cache_salt: str,
        scratch_key: str,
    ) -> Iterator[tuple[list[ObjectKey], MemoryLayoutDesc]]:
        """Context manager: resolve ``real_prompt``'s per-rank object keys.

        Owns the scratch-session lifecycle (create -> hash -> remove) the
        key derivation needs; shared by the fetch path and the TP probe
        pass so both hash the prompt identically.

        Args:
            real_prompt: Token IDs of the real prompt (chunk-aligned).
            worker_id: GPU instance ID; resolves model/world via the
                GPU-context registry.
            tp_rank: Tensor-parallel rank (becomes the object keys'
                ``worker_id`` field, matching the STORE-side keying).
            cache_salt: Per-user isolation salt.
            scratch_key: Unique session key for this resolution; removed
                on exit so the SessionManager dict doesn't grow.

        Yields:
            ``(obj_keys, layout_desc)`` for the full prompt.

        Raises:
            RuntimeError: If the worker is unknown or its layout desc is
                missing.
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
            yield ipc_key_to_object_keys(ipc_key, chunk_hashes), layout_desc
        finally:
            # Scratch session is single-use: each resolution gets a fresh
            # key. Without this remove, the SessionManager dict grows
            # unbounded until the 10-minute TTL sweep.
            self._ctx.session_manager.remove(scratch_key)

    def _probe_prefix_hit_count(
        self,
        real_prompt: list[int],
        worker_id: int,
        tp_rank: int,
        cache_salt: str,
        marshal_handle: str,
    ) -> int:
        """Pass 1 of the TP two-pass: one rank's longest L1 prefix count.

        Submits the prefetch (which reserves read locks on the hit
        prefix) and immediately releases them — the probe only needs the
        count, so it must hold nothing. Pass 2 re-reserves each rank at
        the cross-rank min; a chunk evicted between the passes surfaces
        there as a shrink -> clean miss (accepted TOCTOU).

        Args:
            real_prompt: Token IDs of the real prompt (chunk-aligned).
            worker_id: GPU instance ID whose stored KV to look up.
            tp_rank: Tensor-parallel rank to probe.
            cache_salt: Per-user isolation salt.
            marshal_handle: Per-MARSHAL UUID (namespaces the scratch
                session key).

        Returns:
            Number of leading chunks of ``real_prompt`` present in L1
            for this rank (0 = cold).

        Raises:
            RuntimeError: If the worker is unknown or its layout desc is
                missing (propagated from the key resolution).
        """
        with self._object_keys_for_prompt(
            real_prompt=real_prompt,
            worker_id=worker_id,
            tp_rank=tp_rank,
            cache_salt=cache_salt,
            scratch_key=f"__marshal_probe__{marshal_handle}__{tp_rank}",
        ) as (obj_keys, layout_desc):
            # Same L1-ONLY ASSUMPTION as the fetch path: with an L2
            # adapter configured this submit would also fire an L2
            # prefetch for the post-hole keys whose handle we drop.
            handle = self._ctx.storage_manager.submit_prefetch_task(
                obj_keys, layout_desc
            )
            hit_count = handle.l1_prefix_hit_count
            if hit_count > 0:
                # Balance the submit's reserve: the probe holds no locks.
                self._ctx.storage_manager.finish_read_prefetched(
                    obj_keys[:hit_count], extra_count=0
                )
            return hit_count

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
