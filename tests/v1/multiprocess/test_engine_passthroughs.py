# SPDX-License-Identifier: Apache-2.0
"""Regression: HTTP-layer passthroughs dropped by the engine refactor."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.gpu_transfer import (
    GPUContextEntry,
    GPUTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.server import MPCacheEngine


def test_storage_manager_returns_context_storage_manager() -> None:
    sm = MagicMock(name="storage_manager")
    ctx = MagicMock()
    ctx.storage_manager = sm

    engine = MPCacheEngine(ctx, modules=[])
    assert engine.storage_manager is sm


def test_gpu_contexts_unwraps_entries_from_ctx_registry() -> None:
    gpu0, gpu1 = MagicMock(name="gpu_ctx_0"), MagicMock(name="gpu_ctx_1")
    ctx = MagicMock()
    ctx.gpu_context_registry.items.return_value = [
        (0, GPUContextEntry(gpu_context=gpu0, model_name="m", world_size=1)),
        (7, GPUContextEntry(gpu_context=gpu1, model_name="m", world_size=1)),
    ]

    engine = MPCacheEngine(ctx, modules=[MagicMock(spec=GPUTransferModule)])
    # Values must be unwrapped GPUCacheContexts read from ctx.
    assert engine.gpu_contexts == {0: gpu0, 7: gpu1}


def test_gpu_contexts_empty_registry_on_gpu_engine_returns_empty_dict() -> None:
    # Behavior-preserving: a GPU engine with no registrations yet returns
    # {} (not None), so /kvcache/check stays 404 ("not registered") rather
    # than flipping to 501 ("not supported for this engine type").
    ctx = MagicMock()
    ctx.gpu_context_registry.items.return_value = []
    engine = MPCacheEngine(ctx, modules=[MagicMock(spec=GPUTransferModule)])
    assert engine.gpu_contexts == {}


def test_gpu_contexts_returns_none_without_a_gpu_module() -> None:
    # No GPU-transfer module present -> None (-> 501), as before.
    ctx = MagicMock()
    ctx.gpu_context_registry.items.return_value = []
    engine = MPCacheEngine(ctx, modules=[MagicMock()])
    assert engine.gpu_contexts is None


def test_clear_delegates_to_management_module() -> None:
    mgmt = MagicMock(spec=ManagementModule)
    engine = MPCacheEngine(MagicMock(), modules=[MagicMock(), mgmt])
    engine.clear()
    mgmt.clear.assert_called_once_with()


def test_clear_raises_without_management_module() -> None:
    engine = MPCacheEngine(MagicMock(), modules=[])
    with pytest.raises(RuntimeError, match="no ManagementModule"):
        engine.clear()
