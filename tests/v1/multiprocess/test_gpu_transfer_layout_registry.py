# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GPU transfer layout registration lifetime."""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock, patch
import sys
import types

# Third Party
import pytest
import torch


class _FakeGPUContext:
    """Small stand-in for GPUCacheContext used by registration tests."""

    num_layers: int = 2


class _FakeDeviceHostFuncDispatcher:
    """No-op dispatcher to avoid starting native completion threads."""

    def register(self, kind: str, handler: object, payload_type: object) -> None:
        """Record no native callback registration."""

    def start(self) -> None:
        """Start no background thread."""

    def stop(self) -> None:
        """Stop no background thread."""


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so MP server imports work in source-only test runs."""
    module = types.ModuleType("lmcache.native_storage_ops")
    module_any = cast(Any, module)
    module_any.TTLLock = type("TTLLock", (), {})
    module_any.Bitmap = type("Bitmap", (), {})
    with patch.dict(
        sys.modules,
        {
            "lmcache.native_storage_ops": module,
            "cupy": MagicMock(),
        },
    ):
        yield


def test_unregister_one_shared_gpu_layout_keeps_registry_until_last_instance(
    monkeypatch: pytest.MonkeyPatch,
    stub_native_storage_ops: Any,
) -> None:
    """Unregistering one shared GPU instance must not remove the shared layout."""
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import (
        GPUContextRegistry,
        LayoutDescRegistry,
    )
    from lmcache.v1.multiprocess.modules import gpu_transfer as gpu_transfer_mod

    layout_desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 16, 32])],
        dtypes=[torch.float32],
    )
    ctx = MagicMock()
    ctx.chunk_size = 16
    ctx.layout_desc_registry = LayoutDescRegistry()
    # SEAM 1: register/unregister now write the GPU-context registry on ctx,
    # and register_kv_cache's already-registered guard reads it. A bare
    # MagicMock would make get() return a truthy auto-mock (guard always
    # True -> every register early-returns), so use the real registry.
    ctx.gpu_context_registry = GPUContextRegistry()

    def fake_gpu_context(*args: object, **kwargs: object) -> _FakeGPUContext:
        """Return a fake GPU context without touching CUDA."""
        return _FakeGPUContext()

    def fake_layout_desc(
        gpu_context: _FakeGPUContext,
        num_tokens: int,
    ) -> MemoryLayoutDesc:
        """Return the shared layout descriptor used by both registrations."""
        return layout_desc

    monkeypatch.setattr(
        gpu_transfer_mod,
        "DeviceHostFuncDispatcher",
        _FakeDeviceHostFuncDispatcher,
    )
    monkeypatch.setattr(gpu_transfer_mod, "GPUCacheContext", fake_gpu_context)
    monkeypatch.setattr(gpu_transfer_mod, "get_layout_desc", fake_layout_desc)
    monkeypatch.setattr(
        gpu_transfer_mod.torch_dev,
        "empty_cache",
        lambda: None,
        raising=False,
    )

    module = gpu_transfer_mod.GPUTransferModule(ctx)
    module.register_kv_cache(1, [], "shared-model", 1, EngineType.VLLM, {})
    module.register_kv_cache(2, [], "shared-model", 1, EngineType.VLLM, {})
    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc
    # SEAM 1 round-trip: both instances land in the GPU-context registry.
    assert ctx.gpu_context_registry.get(1) is not None
    assert ctx.gpu_context_registry.get(2) is not None
    assert sorted(i for i, _ in ctx.gpu_context_registry.items()) == [1, 2]

    # Re-registering a live instance is a no-op via the new guard.
    module.register_kv_cache(1, [], "shared-model", 1, EngineType.VLLM, {})
    assert sorted(i for i, _ in ctx.gpu_context_registry.items()) == [1, 2]

    module.unregister_kv_cache(1)
    assert ctx.gpu_context_registry.get(1) is None
    assert ctx.gpu_context_registry.get(2) is not None
    assert ctx.layout_desc_registry.find("shared-model", 1) is layout_desc

    module.unregister_kv_cache(2)
    assert ctx.gpu_context_registry.get(2) is None
    assert ctx.gpu_context_registry.items() == []
    assert ctx.layout_desc_registry.find("shared-model", 1) is None


def test_gpu_transfer_close_stops_dispatcher_and_clears_registry(
    monkeypatch: pytest.MonkeyPatch,
    stub_native_storage_ops: Any,
) -> None:
    """GPUTransferModule.close() stops the finish-write dispatcher and clears
    the GPU-context registry. After the marshal-module-split the workspace
    pool is owned + freed by MarshalModule.close (see
    test_marshal_close_frees_workspace), so this close must NOT touch
    ctx.marshal_workspace — the cross-module ordering (dispatcher stop before
    workspace free) is enforced by _build_modules appending MarshalModule
    AFTER GPUTransferModule, not by this single close()."""
    # First Party
    from lmcache.v1.multiprocess.engine_context import GPUContextRegistry
    from lmcache.v1.multiprocess.modules import gpu_transfer as gpu_transfer_mod

    monkeypatch.setattr(
        gpu_transfer_mod.torch_dev, "empty_cache", lambda: None, raising=False
    )

    reg = GPUContextRegistry()
    reg.register(1, object())  # non-empty -> close() takes the empty_cache path
    workspace = MagicMock()
    dispatcher = MagicMock()

    # Bypass __init__ (it starts a native dispatcher thread); set only the
    # fields close() reads.
    module = gpu_transfer_mod.GPUTransferModule.__new__(
        gpu_transfer_mod.GPUTransferModule
    )
    module._ctx = types.SimpleNamespace(
        gpu_context_registry=reg, marshal_workspace=workspace
    )
    module._device_host_func_dispatcher = dispatcher

    module.close()

    dispatcher.stop.assert_called_once()
    assert module._ctx.gpu_context_registry.items() == []  # registry cleared
    # Workspace ownership moved to MarshalModule: GPUTransferModule.close must
    # leave the seam untouched (it neither frees the pool nor nulls it).
    workspace.close.assert_not_called()
    assert module._ctx.marshal_workspace is workspace


def test_marshal_close_frees_workspace(
    stub_native_storage_ops: Any,
) -> None:
    """MarshalModule.close() frees the workspace pool and nulls the
    ctx.marshal_workspace seam.

    This is the second half of the close-ordering invariant: it runs AFTER
    GPUTransferModule.close stopped the finish-write dispatcher (guaranteed by
    _build_modules' append order), so no in-flight finish-write callback can
    touch the pinned buffers this frees. The drain that protects against a
    pending MARSHAL_FREE ref_count_down decrementing into a closed allocator
    lives inside MarshalWorkspace.close (see
    test_close_drains_recorded_devices_before_free), so MarshalModule.close
    just delegates to it."""
    # First Party
    from lmcache.v1.multiprocess.modules import marshal as marshal_mod

    workspace = MagicMock()
    module = marshal_mod.MarshalModule.__new__(marshal_mod.MarshalModule)
    module._ctx = types.SimpleNamespace(marshal_workspace=workspace)

    module.close()

    workspace.close.assert_called_once()
    assert module._ctx.marshal_workspace is None  # seam dropped

    module.close()  # idempotent: seam already None -> no-op, no raise
    workspace.close.assert_called_once()
