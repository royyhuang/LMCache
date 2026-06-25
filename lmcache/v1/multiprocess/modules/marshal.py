# SPDX-License-Identifier: Apache-2.0
"""KV-tunnel MARSHAL engine module.

The kvtunnel-only half of what used to live in ``GPUTransferModule``:
the MARSHAL / MARSHAL_FREE / WAIT_STORE handlers. MARSHAL packs an
already-stored prompt's KV into a pinned workspace blob (via
:class:`MarshalWorkspace`, published on ``ctx.marshal_workspace``); a
later RETRIEVE carrying the same ``marshal_handle`` scatters that blob
into vLLM's paged cache instead of reading L1. WAIT_STORE gates the
proxy's next MARSHAL on the previous cycle's STORE committing to L1,
waiting on ``ctx.chunk_commit_notifier`` (signalled by
``GPUTransferModule``'s finish-write callback).

Reaches all shared state through ctx seams, so it depends only on
``ctx`` like every other engine module.
"""

# Standard
from contextlib import contextmanager
from typing import Iterator
import os
import uuid

from kvtunnel.wire.header import (
    MAX_HEADER_BLOCKS,
    TunneledRequestMetadata,
)
from kvtunnel.wire.interface import PackRequest
from kvtunnel.wire.registry import get_packer

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
    ``ctx.marshal_workspace``) and provides the MARSHAL, MARSHAL_FREE,
    and WAIT_STORE handlers. Reads the GPU-context registry and waits
    on the chunk-commit notifier, both via ``ctx``.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheEngineContext) -> None:
        self._ctx = ctx

        # kvtunnel MARSHAL workspace — owns the pinned pool, the
        # marshal_handle -> blob dict, and the tunneled-RETRIEVE scatter.
        # Constructed here (MarshalModule is GPU-mode-gated by
        # _build_modules, so a non-GPU server never pins an unusable pool)
        # and published on ctx so GPUTransferModule's RETRIEVE delegation
        # reaches it via ``ctx.marshal_workspace``.
        self._ctx.marshal_workspace = MarshalWorkspace(self._ctx)

    @property
    def context(self) -> MPCacheEngineContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for the request types this module serves."""
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
        """Return MARSHAL module status (always empty)."""
        return {}

    def close(self) -> None:
        """Free the kvtunnel workspace pool."""
        if self._ctx.marshal_workspace is not None:
            self._ctx.marshal_workspace.close()
            self._ctx.marshal_workspace = None

    # ------------------------------------------------------------------
    # KV tunneling — MARSHAL RPC and the workspace pack path.
    # ------------------------------------------------------------------

    def marshal(
        self,
        marshal_handle: str,
        real_prompt: list[int],
        extra_params: dict,
        worker_id: int,
    ) -> tuple[bool, int, str, dict[int, TunneledRequestMetadata], int]:
        """Pack ``real_prompt``'s longest cached prefix into a workspace
        blob for a later RETRIEVE to scatter into vLLM's paged cache.

        StreamingLLM sink + sliding-window selection runs on CPU; the
        packed chunks are parked in ``ctx.marshal_workspace`` keyed by
        ``marshal_handle``. The match is the longest chunk-aligned prefix
        in L1; the hash auto-rounds, so an unaligned ``real_prompt`` is
        accepted and its tail rides the proxy's forwarded suffix. Two
        floors gate every match, each returning a clean cache-miss instead
        of packing:
          - compression floor ``num_sinks + window_size +
            MAX_HEADER_BLOCKS * block_size`` (`<=`): below it the pack
            would crash on delta<0.
          - caller floor ``min_matched_tokens`` (`<`), capped at the
            aligned total when ``min_matched_tokens >= len(real_prompt)``
            ("full-aligned or miss"; the cycles k>=1 call).
        For TP>1 with a fractional floor, a probe takes the cross-rank MIN
        match and all ranks pack at it; a full-required floor skips the
        probe (the fetch enforces all-or-nothing).

        Args:
            marshal_handle: Rendezvous key the proxy redeems via RETRIEVE.
            real_prompt: Token IDs already stored unmarshalled in LMCache
                (the miss path). May be unaligned (tail excluded).
            extra_params: Honors ``num_sinks`` (4), ``window_size``
                (1020), ``cache_salt`` (""), ``min_matched_tokens`` (0);
                other keys ignored.
            worker_id: GPU instance ID whose stored KV to look up. Must
                match a prior REGISTER_KV_CACHE.

        Returns:
            ``(success, num_fake, error_message, tunneled_request_per_rank,
            matched_prefix_len)``. ``num_fake`` is the fake-slot count the
            blob occupies; ``tunneled_request_per_rank`` maps ``tp_rank``
            -> per-layer ``TunneledRequestMetadata``; ``matched_prefix_len``
            is the chunk-aligned count packed (proxy derives the suffix as
            ``real_prompt[matched_prefix_len:]``). On failure all but
            ``error_message`` are zero/empty, prefixed with ``_L1_MISS_MSG``
            on a clean cache-miss.
        """
        # allocate() returns ref_count=1; that ref becomes the workspace's
        # only at put(). Until then this fn owns the blobs, so every
        # non-success exit must _free_packed() or leak pinned pool bytes.
        per_rank: dict[int, tuple[list[MemoryObj], TunneledRequestMetadata]] = {}
        published = False

        def _free_packed() -> None:
            for objs, _manifest in per_rank.values():
                for mem_obj in objs:
                    mem_obj.ref_count_down()
            per_rank.clear()

        try:
            num_sinks = int(extra_params.get("num_sinks", 4))
            window_size = int(extra_params.get("window_size", 1020))
            cache_salt = str(extra_params.get("cache_salt", ""))
            min_matched_tokens = int(extra_params.get("min_matched_tokens", 0))

            entry = self._ctx.gpu_context_registry.get(worker_id)
            if entry is None:
                raise RuntimeError(
                    f"no GPU context registered for worker_id={worker_id}"
                )
            world_size = entry.world_size
            use_stub = os.environ.get("KVTUNNEL_STUB_MARSHAL") == "1"
            # Non-stub marshal method (a string, not a bool flag): selects
            # the Packer for the real reuse path. ``streaming_llm`` (default,
            # token drop) or ``packed_fp8`` (fp8 2:1 slot-count packing). The
            # stub byte-copy path above is orthogonal to this.
            marshal_method = os.environ.get("KVTUNNEL_MARSHAL_METHOD", "streaming_llm")

            # Always published in GPU mode (the only mode MARSHAL runs in);
            # the check is for type-narrowing.
            workspace = self._ctx.marshal_workspace
            if workspace is None:
                raise RuntimeError("MARSHAL workspace is not initialized")
            workspace_allocator = workspace.kvtunnel_workspace_allocator

            gpu_ctx = entry.gpu_context
            kvlgm = gpu_ctx.kv_layer_groups_manager
            ie_block_size = kvlgm.inference_engine_logical_block_size
            # At/below this StreamingLLM retains ~the whole prefix, so
            # num_fake > matched + header and the pack crashes on delta<0.
            compression_floor = (
                num_sinks + window_size + MAX_HEADER_BLOCKS * ie_block_size
            )

            # require_full (ask >= full len): cap at aligned_total so an
            # unaligned tail can't force a miss. Fractional asks stay raw.
            total = len(real_prompt)
            total_chunks = total // self._ctx.chunk_size
            aligned_total = total_chunks * self._ctx.chunk_size
            require_full = min_matched_tokens >= total
            match_floor = aligned_total if require_full else min_matched_tokens

            # One blob per TP rank; each rank's RETRIEVE finds its own via
            # worker_id.
            tunneled_request_per_rank: dict[int, TunneledRequestMetadata] = {}
            num_fake = 0
            # All ranks must pack the SAME length (num_fake is uniform, the
            # proxy sends one). TP>1 probes for the cross-rank min first.
            limit_chunks = 0
            if world_size > 1 and not require_full:
                # require_full needs no probe — the fetch enforces
                # all-or-nothing. Fractional floors probe (ranks differ).
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
                matched_at_min = limit_chunks * self._ctx.chunk_size
                if matched_at_min <= compression_floor or matched_at_min < match_floor:
                    # Below a floor already -> clean miss before any fetch.
                    return (
                        False,
                        0,
                        _L1_MISS_MSG + " (cross-rank min below floor)",
                        {},
                        0,
                    )
                if limit_chunks < total_chunks:
                    logger.info(
                        "MARSHAL TP min-prefix: rank hits %s -> packing "
                        "all %d ranks at %d chunks",
                        rank_hits,
                        world_size,
                        limit_chunks,
                    )
            # Equal matched length -> equal num_fake; checked below.
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
                        # No chunk in L1 -> clean miss (proxy passthrough).
                        _free_packed()
                        return (False, 0, _L1_MISS_MSG, {}, 0)
                    matched_chunks = len(mem_objs)
                    matched_prefix_len = matched_chunks * self._ctx.chunk_size
                    if limit_chunks > 0 and matched_chunks != limit_chunks:
                        # Rank shrank (eviction) since the probe -> miss.
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
                    # _L1_MISS_MSG prefix so the proxy classifies these as
                    # a miss (passthrough), not a 503.
                    if matched_prefix_len <= compression_floor:
                        _free_packed()
                        return (
                            False,
                            0,
                            _L1_MISS_MSG + " (at or below compression floor)",
                            {},
                            0,
                        )
                    if matched_prefix_len < match_floor:
                        _free_packed()
                        return (
                            False,
                            0,
                            _L1_MISS_MSG + " (below min_matched_tokens floor)",
                            {},
                            0,
                        )
                    if use_stub:
                        # Stub stamps the magic header at every layer's
                        # start (hence num_layers); see StubPacker.
                        req = PackRequest(
                            workspace_allocator=workspace_allocator,
                            orig_kv_obj=mem_objs,
                            chunk_size=self._ctx.chunk_size,
                            num_layers=gpu_ctx.num_layers,
                            block_size=ie_block_size,
                        )
                        packed_list, manifest = get_packer("stub").pack(req)
                        num_fake = manifest.per_layer[0].num_fake_marshalled
                    else:
                        # Head geometry validates the chunk's KV_2LTD shape
                        # and fills the header's num_active_heads.
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
                        # real_prompt_len = matched (not full): anchors the
                        # window and delta so the forwarded suffix RoPEs at
                        # matched_prefix_len..
                        req = PackRequest(
                            workspace_allocator=workspace_allocator,
                            orig_kv_obj=mem_objs,
                            chunk_size=self._ctx.chunk_size,
                            num_layers=gpu_ctx.num_layers,
                            block_size=ie_block_size,
                            real_prompt_len=matched_prefix_len,
                            num_kv_heads=shape_desc.nh,
                            head_size=shape_desc.hs,
                            max_chunks=max_chunks,
                            num_groups=gpu_ctx.kv_layer_groups_manager.num_groups,
                            is_mla=gpu_ctx.is_mla,
                            extra_params=extra_params,
                        )
                        packed_list, manifest = get_packer(marshal_method).pack(req)
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
                    # ref_count=1 from allocate() is the workspace's ref —
                    # no ref_count_up (would block MARSHAL_FREE). Stored
                    # before the num_fake check so divergence frees it too.
                    per_rank[tp_rank] = (packed_list, manifest)
                    tunneled_request_per_rank[tp_rank] = manifest
                    if expected_num_fake < 0:
                        expected_num_fake = num_fake
                    elif num_fake != expected_num_fake:
                        # Impossible (equal length -> equal num_fake);
                        # guarded anyway.
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
                # Ownership transfers only at put(); free orphans on a
                # mid-loop raise.
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

        Args:
            real_prompt: Token IDs of the real prompt.
            worker_id: GPU instance ID; routes through the correct
                registered context, not part of the storage key.
            tp_rank: Tensor-parallel rank whose KV shard to read. Hashes
                into the object key via ``kv_rank`` — the STORE-side
                adapter uses the same field on ``IPCCacheEngineKey``, so
                the keys must match.
            cache_salt: Per-user isolation salt matching the STORE; empty
                string matches unsalted entries.
            marshal_handle: Per-MARSHAL UUID, used as the scratch session
                key so concurrent MARSHAL calls can't collide on
                session-manager state (replaces the GC-reusable
                ``id(real_prompt)`` keying).
            limit_chunks: Cap on leading chunks to read. 0 (default,
                single-rank path) means no cap — the full longest prefix.
                The TP two-pass passes the cross-rank min so every rank
                reads the same length; surplus locks beyond the cap are
                released before the read.

        Yields:
            The ordered MemoryObj chunks covering the longest chunk-aligned
            prefix of ``real_prompt`` in L1, capped at ``limit_chunks``
            when set (caller derives ``matched_prefix_len = len(chunks) *
            chunk_size``), or ``None`` if no leading chunk is in L1 (cold
            prompt — normal state, not an error).

        Raises:
            RuntimeError: If the worker is unknown or its layout desc is
                missing.
        """
        with self._object_keys_for_prompt(
            real_prompt=real_prompt,
            worker_id=worker_id,
            tp_rank=tp_rank,
            cache_salt=cache_salt,
            scratch_key=f"__marshal__{marshal_handle}__{tp_rank}",
        ) as (obj_keys, layout_desc):
            # Submit on the FULL key list: submit_prefetch_task breaks at
            # the first hole, returns the longest-prefix hit count, and
            # auto-releases locks past the hole, so truncating below leaves
            # no dangling locks. L1-ONLY ASSUMPTION: an L2 adapter would
            # also prefetch post-hole keys into L1 holding unreleased locks
            # — kvtunnel is L1-only today; revisit before enabling L2.
            handle = self._ctx.storage_manager.submit_prefetch_task(
                obj_keys, layout_desc
            )
            hit_count = handle.l1_prefix_hit_count
            if hit_count == 0:
                # Cold prompt: no leading chunk in L1 — a normal first-
                # request state, not an error. No locks held (a zero-hit
                # submit reserved nothing). Yield None -> clean miss.
                logger.info(
                    "MARSHAL miss: prompt chunks not in L1 (cold prompt, %d chunks)",
                    len(obj_keys),
                )
                yield None
                return
            take = hit_count
            if 0 < limit_chunks < hit_count:
                # TP pass 2: this rank matched more than the cross-rank min
                # — release the surplus locks now (submit reserved
                # [0:hit_count]; only [0:limit_chunks] is read below).
                self._ctx.storage_manager.finish_read_prefetched(
                    obj_keys[limit_chunks:hit_count], extra_count=0
                )
                take = limit_chunks
            # Read ONLY the matched prefix. Truncating before the read keeps
            # lock bookkeeping aligned — read_prefetched_results yields
            # all-good and the finish_read below releases exactly these keys.
            matched_keys = obj_keys[:take]
            with self._ctx.storage_manager.read_prefetched_results(
                matched_keys
            ) as mem_objs:
                if mem_objs is None:
                    # Reserve-to-read eviction race (rare): inner context
                    # released the keys. Clean miss, same as cold.
                    logger.info(
                        "MARSHAL miss: matched prefix evicted between "
                        "reserve and read (%d/%d chunks)",
                        take,
                        len(obj_keys),
                    )
                    yield None
                    return
                # marshal()'s with-body copies these bytes into pinned-CPU
                # tensors (sync CPU slice-assign), so once it returns the
                # source is fully copied and the lock releases eagerly (no
                # stream deferral, unlike the async-H2D retrieve path).
                yield list(mem_objs)
                # Reached only if the with-body didn't raise: release the
                # read lock for exactly the truncated prefix. Success-path
                # only — read_prefetched_results' finally covers miss/raise.
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
            tp_rank: Tensor-parallel rank (becomes the keys' ``worker_id``
                field, matching the STORE-side keying).
            cache_salt: Per-user isolation salt.
            scratch_key: Unique session key for this resolution; removed on
                exit so the SessionManager dict doesn't grow.

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
            # get_hashes(0) auto-rounds to the last full-chunk boundary, so
            # an unaligned real_prompt yields an aligned prefix instead of
            # asserting. The tail is excluded (rides the forwarded suffix);
            # ipc_key.end stays full but is inert (keys derive from hashes).
            chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
            # IPCCacheEngineKey.worker_id is the TP rank, matching the
            # STORE-side adapter (vllm_multi_process_adapter.py::_create_key).
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
            # Scratch session is single-use; without this remove the
            # SessionManager dict grows until the 10-min TTL sweep.
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

        Submits the prefetch (reserving read locks on the hit prefix) and
        immediately releases them — the probe only needs the count, so it
        holds nothing. Pass 2 re-reserves each rank at the cross-rank min;
        a chunk evicted between the passes surfaces there as a shrink ->
        clean miss (accepted TOCTOU).

        Args:
            real_prompt: Token IDs of the real prompt (chunk-aligned).
            worker_id: GPU instance ID whose stored KV to look up.
            tp_rank: Tensor-parallel rank to probe.
            cache_salt: Per-user isolation salt.
            marshal_handle: Per-MARSHAL UUID (namespaces the scratch
                session key).

        Returns:
            Number of leading chunks of ``real_prompt`` in L1 for this rank
            (0 = cold).

        Raises:
            RuntimeError: If the worker is unknown or its layout desc is
                missing (propagated from key resolution).
        """
        with self._object_keys_for_prompt(
            real_prompt=real_prompt,
            worker_id=worker_id,
            tp_rank=tp_rank,
            cache_salt=cache_salt,
            scratch_key=f"__marshal_probe__{marshal_handle}__{tp_rank}",
        ) as (obj_keys, layout_desc):
            # Same L1-ONLY ASSUMPTION as the fetch path: an L2 adapter
            # would also fire an L2 prefetch for the dropped post-hole keys.
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
        has finished. Delegates to :meth:`MarshalWorkspace.free` (which
        owns the pop + stream-ordered ``ref_count_down``); an unknown /
        already-freed handle is a no-op there, as is any call when no
        workspace is published (non-GPU server).

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
        """Block until ``token_ids``'s last committed chunk is readable on
        every TP rank, or timeout.

        Only the trailing chunk hash is waited on, expanded across all TP
        ranks (worker_id=None) since each rank stores its own shard.

        Args:
            token_ids: Running real prompt (prompt + decoded so far).
            end_offset: Length of the running prompt. Carried on the
                ipc_key (inert for chunk math); the trailing-chunk hash is
                derived via ``get_hashes(0)``, which auto-rounds
                ``token_ids`` to its last full-chunk boundary, so an
                unaligned running length no longer asserts. The proxy
                passes ``end_offset == len(token_ids)`` and only calls
                WAIT_STORE after a full chunk has committed, so the rounded
                hash list is always non-empty.
            worker_id: GPU instance ID; looks up the registered context for
                model_name + world_size.
            wait_timeout_ms: ``event.wait`` deadline in milliseconds.

        Returns:
            ``"Ready"`` if the chunk is readable on every TP rank within
            the deadline; ``"Pending"`` otherwise.

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
            # get_hashes(0) auto-rounds to the last full-chunk boundary, so
            # this waits on the last COMMITTED chunk (proxy passes
            # end_offset == len(token_ids)); an unaligned length no longer
            # asserts. end_offset is inert here (used for the ipc_key).
            chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
            target_hash = chunk_hashes[-1]  # only the trailing chunk

            # worker_id=None makes ipc_key_to_object_keys emit one ObjectKey
            # per TP rank for the same chunk hash. cache_salt="" matches the
            # connector tracker's default and the proxy's strip contract (no
            # cache_salt on the wire).
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

            # Register the Event BEFORE the second is_ready check (register()
            # owns leak-free removal). Closes the race where finish_write
            # lands between the first is_ready and registration: the second
            # check sees the post-finish_write state and returns Ready.
            with self._ctx.chunk_commit_notifier.register(target_hash) as event:
                if all(self._ctx.storage_manager.is_ready(obj_keys)):
                    return "Ready"

                if event.wait(timeout=wait_timeout_ms / 1000.0):
                    # Re-check after wakeup: the Event may have been set by
                    # finish_write's exception-path signal (fires on
                    # exception while write_lock is still held -> not
                    # readable), which would otherwise return Ready falsely.
                    if all(self._ctx.storage_manager.is_ready(obj_keys)):
                        return "Ready"
                    return "Pending"
                return "Pending"
        finally:
            self._ctx.session_manager.remove(session_key)
