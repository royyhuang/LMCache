"""Packed-deployment store windows (plan/feat/packed-at-write Phase 3).

Under ``KVTUNNEL_MARSHAL_METHOD=packed_fp8`` the server consumes block
ids at HALF cadence (compress_ratio=2), so ``GetStoreMetadata`` must
derive the store window from real-chain token offsets — the stock
vLLM-cadence slice points past the packed data on every store after
the first. The three corruption scenarios the panel named (multi-RPC
cold prefill; rounding-cycle delta with ``num_fake > ceil(N/2)``;
same-request multi-store) are pinned here, plus the packed-deployment
lookup disable (non-tunnel retrieve reports a miss).
"""

from types import SimpleNamespace

import pytest

import lmcache.integration.vllm.lmcache_mp_connector as conn_mod
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPRequestMetadata,
)

_BS = 16  # vllm block size
_CHUNK = 256
_BIC = _CHUNK // _BS  # blocks_in_chunk (vLLM cadence) = 16


class _Tracker:
    """Minimal LMCacheMPRequestTracker stand-in for GetStoreMetadata."""

    def __init__(
        self,
        *,
        num_tokens: int,
        num_blocks: int,
        kv_transfer_params=None,
        num_stored_blocks: int = 0,
    ) -> None:
        self.request_id = "r0"
        self.cache_salt = ""
        self.all_token_ids = list(range(num_tokens))
        self.allocated_block_ids = list(range(1000, 1000 + num_blocks))
        self.kv_transfer_params = kv_transfer_params
        self.num_stored_blocks = num_stored_blocks
        self.num_scheduled_tokens = num_tokens
        self.num_vllm_hit_blocks = 0
        self.num_lmcache_hit_blocks = 0
        # Stock (non-tunneled) bound consults vLLM block hashes; one
        # hash per allocated block keeps the bound at the block count.
        self.block_hashes = [b"h"] * num_blocks

    def increase_num_stored_blocks(self, n: int) -> None:
        self.num_stored_blocks += n


def _store(tracker):
    return LMCacheMPRequestMetadata.GetStoreMetadata(tracker, _BIC, _BS)


@pytest.fixture(autouse=True)
def _packed_deploy(monkeypatch):
    monkeypatch.setattr(conn_mod, "_PACKED_DEPLOY", True)


def test_multi_rpc_cold_prefill_windows_are_half_cadence():
    """Cold 2-chunk prompt stored across two RPCs: each op's block ids
    cover the PACKED half-cadence extent of its chunk — RPC 2's window
    starts at chunk1's packed blocks (token 256 -> block 8), not at the
    vLLM-cadence num_stored_blocks (16)."""
    t = _Tracker(num_tokens=2 * _CHUNK, num_blocks=2 * _BIC)
    t.num_scheduled_tokens = _CHUNK  # RPC 1 sees chunk 0 only
    t.all_token_ids = list(range(_CHUNK))
    md1 = _store(t)
    assert md1 is not None
    bs2 = 2 * _BS
    assert md1.op.block_ids == t.allocated_block_ids[0 : _CHUNK // bs2]
    assert (md1.op.start, md1.op.end) == (0, _CHUNK)

    # RPC 2: chunk 1 now available.
    t.all_token_ids = list(range(2 * _CHUNK))
    t.num_scheduled_tokens = 2 * _CHUNK
    md2 = _store(t)
    assert md2 is not None
    w0 = _CHUNK // bs2  # 8 — chunk 1's packed blocks
    w1 = 2 * _CHUNK // bs2
    assert md2.op.block_ids == t.allocated_block_ids[w0:w1]
    assert (md2.op.start, md2.op.end) == (_CHUNK, 2 * _CHUNK)


def test_rounding_cycle_delta_window_from_real_chain():
    """Tunneled delta with num_fake > ceil(N/2) (rounding cycle): the
    window derives from real_chain_start // (2*bs), BELOW the
    vLLM-cadence num_fake-based slice."""
    real_n = 1280
    num_fake = 768  # > ceil(1280/2) = 640 (chunk-rounded)
    delta = _CHUNK
    total_fake_tokens = num_fake + delta
    t = _Tracker(
        num_tokens=total_fake_tokens,
        num_blocks=(total_fake_tokens + _BS - 1) // _BS,
        kv_transfer_params={
            "kv_tunnel_mvp": True,
            "num_fake": num_fake,
            "kvtunnel_real_token_ids": list(range(real_n)),
        },
        num_stored_blocks=num_fake // _BS,  # seeded past the dummies
    )
    md = _store(t)
    assert md is not None
    bs2 = 2 * _BS
    # Real chain: start at real_n (the delta appends after the prefix).
    w0 = real_n // bs2  # 40
    w1 = (real_n + delta) // bs2  # 48
    assert md.op.block_ids == t.allocated_block_ids[w0:w1]
    assert (md.op.start, md.op.end) == (real_n, real_n + delta)
    # Stock cadence would have started at num_fake // _BS = 48 — past
    # the packed data.
    assert w0 < num_fake // _BS


def test_same_request_multi_store_advances_half_cadence():
    """Two consecutive delta stores on one tunneled tracker: each
    window advances by chunk//(2*bs) packed blocks while
    num_stored_blocks advances at vLLM cadence (staging math only)."""
    real_n = 1024
    num_fake = 512
    t = _Tracker(
        num_tokens=num_fake + 2 * _CHUNK,
        num_blocks=(num_fake + 2 * _CHUNK) // _BS,
        kv_transfer_params={
            "kv_tunnel_mvp": True,
            "num_fake": num_fake,
            "kvtunnel_real_token_ids": list(range(real_n)),
        },
        num_stored_blocks=num_fake // _BS,
    )
    t.all_token_ids = list(range(num_fake + _CHUNK))
    t.num_scheduled_tokens = num_fake + _CHUNK
    md1 = _store(t)
    assert md1 is not None
    bs2 = 2 * _BS
    assert (
        md1.op.block_ids
        == t.allocated_block_ids[real_n // bs2 : (real_n + _CHUNK) // bs2]
    )

    t.all_token_ids = list(range(num_fake + 2 * _CHUNK))
    t.num_scheduled_tokens = num_fake + 2 * _CHUNK
    md2 = _store(t)
    assert md2 is not None
    assert (
        md2.op.block_ids
        == t.allocated_block_ids[
            (real_n + _CHUNK) // bs2 : (real_n + 2 * _CHUNK) // bs2
        ]
    )
    # vLLM-cadence staging accounting still advanced by 16 per chunk.
    assert t.num_stored_blocks == num_fake // _BS + 2 * _BIC


def test_packed_deploy_disables_non_tunnel_lookup(monkeypatch):
    """Under the packed deployment a PLAIN request's lookup reports a
    miss (0, False) — the stock lookup-hit retrieve window is
    full-cadence and would scatter past the packed data; the marshal
    path is the only packed reader. The tunnel short-circuit is
    unaffected."""
    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPConnector,
    )

    submitted = []
    fake_self = SimpleNamespace(
        _get_or_create_request_tracker=lambda req: _Tracker(num_tokens=8, num_blocks=1),
        scheduler_adapter=SimpleNamespace(
            maybe_submit_lookup_request=lambda *a, **k: submitted.append(1),
            check_lookup_result=lambda rid: 8,
        ),
        vllm_block_size=_BS,
    )
    plain_req = SimpleNamespace(
        status=None,
        kv_transfer_params=None,
        request_id="p0",
        all_token_ids=list(range(8)),
    )
    got = LMCacheMPConnector.get_num_new_matched_tokens(fake_self, plain_req, 0)
    assert got == (0, False)
    assert submitted == []  # lookup RPC never fired


def test_tunnel_retrieve_window_is_half_cadence():
    """The tunnel RETRIEVE's block window is half-cadence under the
    packed deployment: num_fake tokens occupy num_fake // (2*bs)
    blocks (2 fp8 tokens per bf16 slot) — the ratio-2 workspace
    scatter consumes exactly chunk//(2*bs) ids per logical chunk. A
    vLLM-cadence window (num_fake // bs ids) trips the scatter's
    count-mismatch guard."""
    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPRequestMetadata,
    )

    num_fake = 256
    t = _Tracker(
        num_tokens=num_fake + 1,
        num_blocks=(num_fake + 1 + _BS - 1) // _BS,
        kv_transfer_params={
            "kv_tunnel_mvp": True,
            "num_fake": num_fake,
            "marshal_handle": "h1",
            "kvtunnel_real_token_ids": list(range(300)),
        },
    )
    t.num_lmcache_hit_blocks = num_fake // _BS
    md = LMCacheMPRequestMetadata.GetRetrieveMetadata(t, _BIC, _BS)
    assert md is not None
    # num_fake is the packed half-slot count -> num_fake // block_size
    # blocks, matching the compress_ratio=2 scatter's k*blocks_per_chunk.
    assert len(md.op.block_ids) == num_fake // _BS
    assert md.op.block_ids == t.allocated_block_ids[: num_fake // _BS]


def test_tunnel_retrieve_window_multi_chunk_matches_scatter():
    """Multi-chunk (k>=2) retrieve: num_blocks_needed must equal k *
    blocks_per_chunk (the compress_ratio=2 scatter's expectation),
    which num_fake // block_size gives since num_fake is the packed
    half-slot count (k * chunk_size/2). Guards the k>=2 mismatch the
    single-chunk parity masked."""
    from lmcache.integration.vllm.lmcache_mp_connector import (
        LMCacheMPRequestMetadata,
    )

    # 3 logical chunks: num_fake = 3 * (chunk/2) = 3 * 128 = 384.
    num_fake = 3 * (_CHUNK // 2)
    t = _Tracker(
        num_tokens=num_fake + 1,
        num_blocks=(num_fake + 1 + _BS - 1) // _BS,
        kv_transfer_params={
            "kv_tunnel_mvp": True,
            "num_fake": num_fake,
            "marshal_handle": "h3",
            "kvtunnel_real_token_ids": list(range(3 * _CHUNK)),
        },
    )
    t.num_lmcache_hit_blocks = num_fake // _BS
    md = LMCacheMPRequestMetadata.GetRetrieveMetadata(t, _BIC, _BS)
    assert md is not None
    # k=3 half-slot chunks, compress-ratio-2 blocks_per_chunk =
    # chunk//(2*bs); scatter needs k * that = 3 * (256//32) = 24.
    k = 3
    blocks_per_chunk = _CHUNK // (2 * _BS)
    assert len(md.op.block_ids) == k * blocks_per_chunk == 24
