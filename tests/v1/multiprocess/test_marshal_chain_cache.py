# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the incremental-marshal chain cache (``_ChainCache``).

CPU-only: the cache stores the packer's opaque ``IncrementalState`` (plain
``IncrementalState()`` sentinels here) and never inspects it, so no GPU / MarshalModule
construction is needed. Pins the lifecycle the design relies on: per-rank
store/get, supersession, LRU eviction, ``None``-state skipping, and that
nothing but supersession + LRU ever evicts (``MARSHAL_FREE`` must not touch
the cache).
"""

# First Party
from kvtunnel.wire.interface import IncrementalState
from lmcache.v1.multiprocess.modules.marshal import _ChainCache


def test_store_and_get_roundtrip():
    c = _ChainCache(max_chains=4)
    s0, s1 = IncrementalState(), IncrementalState()
    c.store("h0", "", {0: s0, 1: s1})
    assert c.get("h0", 0) is s0
    assert c.get("h0", 1) is s1
    assert c.get("h0", 2) is None  # unknown rank


def test_supersession_drops_prior_chain():
    c = _ChainCache(max_chains=4)
    a, b = IncrementalState(), IncrementalState()
    c.store("h0", "", {0: a})
    c.store("h1", "h0", {0: b})  # extends h0 -> supersedes it
    assert c.get("h1", 0) is b
    assert c.get("h0", 0) is None  # superseded


def test_none_states_cache_nothing():
    c = _ChainCache(max_chains=4)
    c.store("h0", "", {0: None, 1: None})  # a non-incremental packer
    assert c.get("h0", 0) is None
    # partial: only the non-None ranks are cached
    s = IncrementalState()
    c.store("h1", "", {0: s, 1: None})
    assert c.get("h1", 0) is s
    assert c.get("h1", 1) is None


def test_empty_and_missing_handle_return_none():
    c = _ChainCache(max_chains=4)
    assert c.get("", 0) is None  # empty handle (cold cycle 0)
    assert c.get("nope", 0) is None  # never stored


def test_lru_evicts_oldest():
    c = _ChainCache(max_chains=2)
    s = [IncrementalState() for _ in range(3)]
    c.store("h0", "", {0: s[0]})
    c.store("h1", "", {0: s[1]})
    c.store("h2", "", {0: s[2]})  # over cap -> evicts h0 (oldest)
    assert c.get("h0", 0) is None
    assert c.get("h1", 0) is s[1]
    assert c.get("h2", 0) is s[2]


def test_get_marks_mru():
    c = _ChainCache(max_chains=2)
    s = [IncrementalState() for _ in range(3)]
    c.store("h0", "", {0: s[0]})
    c.store("h1", "", {0: s[1]})
    c.get("h0", 0)  # touch h0 -> now MRU
    c.store("h2", "", {0: s[2]})  # evicts h1 (now oldest), not h0
    assert c.get("h0", 0) is s[0]
    assert c.get("h1", 0) is None
    assert c.get("h2", 0) is s[2]


def test_lru_keeps_all_at_exact_capacity():
    """At exactly max_chains nothing is evicted (pins the `> max` boundary; a
    `>=` mutation would wrongly drop a live chain sitting at capacity)."""
    c = _ChainCache(max_chains=3)
    s = [IncrementalState() for _ in range(3)]
    c.store("h0", "", {0: s[0]})
    c.store("h1", "", {0: s[1]})
    c.store("h2", "", {0: s[2]})  # now exactly at cap -> nothing evicted
    assert c.get("h0", 0) is s[0]
    assert c.get("h1", 0) is s[1]
    assert c.get("h2", 0) is s[2]


def test_supersession_multi_rank():
    """Supersession pops the whole prior chain across all ranks (TP shape)."""
    c = _ChainCache(max_chains=4)
    a0, a1, b0, b1 = (IncrementalState() for _ in range(4))
    c.store("h0", "", {0: a0, 1: a1})
    c.store("h1", "h0", {0: b0, 1: b1})  # extends h0
    assert c.get("h1", 0) is b0
    assert c.get("h1", 1) is b1
    assert c.get("h0", 0) is None  # both ranks of the prior chain gone
    assert c.get("h0", 1) is None


def test_extend_from_missing_handle_still_stores():
    """Extending a handle the LRU already dropped: the pop is a safe no-op and
    the new chain is still cached (an aborted-then-resumed chain)."""
    c = _ChainCache(max_chains=4)
    s = IncrementalState()
    c.store("h1", "gone", {0: s})  # "gone" was never stored / was evicted
    assert c.get("h1", 0) is s


def test_unrelated_store_does_not_evict_live_chain():
    """A different chain's store (no extend_from) leaves an existing chain
    intact -- supersession + LRU are the ONLY eviction paths (MARSHAL_FREE
    never touches the cache)."""
    c = _ChainCache(max_chains=8)
    a, b = IncrementalState(), IncrementalState()
    c.store("h0", "", {0: a})
    c.store("other", "", {0: b})  # unrelated cold chain
    assert c.get("h0", 0) is a  # h0 survives
