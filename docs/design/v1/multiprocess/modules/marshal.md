# MarshalModule — KV-tunnel MARSHAL operations

> kvtunnel fork addition. The full design and the upstream-merge-conflict
> motivation live in the parent repo at
> `plan/refactor/marshal-module-split/{plan,design}.md`; this is the
> deps-side mirror per the `docs/design/` convention.

## What it is

`MarshalModule` (`modules/marshal.py`) is the standalone home of the KV-tunnel
pack/scatter subsystem that used to live inside `GPUTransferModule`. It is an
ordinary `EngineModule` (same ABI as `BlendModule` / `LookupModule`): it owns
no upstream transfer logic and reaches all shared state through three seams on
`MPCacheEngineContext`.

It handles three `RequestType`s (all on the `NORMAL` pool; `get_handlers`):
`MARSHAL`, `MARSHAL_FREE`, `WAIT_STORE`. The wire structs are defined in
`protocols/marshal.py`.

## Seams it uses (all on `ctx`)

- `ctx.gpu_context_registry` (`GPUContextRegistry`) — `instance_id ->
  GPUCacheContext`, written by `GPUTransferModule.register_kv_cache`, read here
  to resolve the source context for a MARSHAL / scatter. `get` is lock-free
  (GIL-atomic dict read), matching the pre-split per-chunk path.
- `ctx.chunk_commit_notifier` (`ChunkCommitNotifier`) — the WAIT_STORE barrier.
  `GPUTransferModule`'s finish-write callback signals it; this module's
  `wait_store` waits on it.
- `ctx.marshal_workspace` (`MarshalWorkspace`) — the `marshal_handle -> blob`
  rendezvous + pinned pool + the H2D scatter. Constructed and published here in
  `__init__`; the tunneled-RETRIEVE branch in `GPUTransferModule.retrieve`
  delegates to its `retrieve_into`.

## MarshalWorkspace (`marshal_workspace.py`)

A leaf module (imports only the allocator + pack primitives, never `modules/`
or the `MPCacheEngineContext` class), so `engine_context.py` can type the
`ctx.marshal_workspace` seam without an import cycle. Owns: the dedicated
pinned `LazyMemoryAllocator` (separate from the L1 allocator), the `_workspace`
dict (outer dict mutated by `put`/`free` under `_lock`; `has` reads lock-free;
`retrieve_into` pops rank state from an entry's inner dicts under `_lock`),
and `_drain_device_indices`.

Two entry ownership regimes (fix/tp1-ttft-overhead): workspace-OWNED entries
(copy-based methods) hold pool blobs — `free` pops the entry and schedules
each chunk's `ref_count_down` as a stream-ordered host callback via
`cupy_stream.launch_host_func`, recording the packing context's device index
in `_drain_device_indices`. L1-BORROWED entries (packed_fp8 zero-copy) hold
the read-locked L1 chunks themselves — `retrieve_into` consumes a rank
(popping chunks + keys together) and releases its read locks stream-ordered
on the consuming rank's stream; `free` releases only never-redeemed ranks'
locks inline, and NEVER `ref_count_down`s borrowed chunks.

## Close ordering + the multi-device drain

`MPCacheEngine.close` runs modules in list order. `_build_modules` appends
`MarshalModule` **after** `GPUTransferModule` (`server.py:186-197`) so that
`GPUTransferModule.close` (stops the finish-write dispatcher, clears the
GPU-context registry) runs before `MarshalModule.close` frees the pinned pool —
no in-flight finish-write callback can touch the buffers being freed.

`MarshalModule.close` delegates to `MarshalWorkspace.close`, which first drains
the stream-ordered `MARSHAL_FREE` `ref_count_down` callbacks, then unpins the
pool. The drain synchronizes **every** device recorded in
`_drain_device_indices`, not just the current one: one MP server holds packing
contexts on several GPUs under TP>1, so a no-arg `torch_dev.synchronize()`
would drain only the current device and leave a callback pending on another
rank's GPU to decrement into freed memory. Capturing devices at `free` time
keeps this correct even though the GPU-context registry is already cleared by
drain time. A narrower, pre-existing shutdown-concurrency window (in-flight
MARSHAL_FREE handler not quiesced) is tracked as O19 in
`plan/mvp-deferred-work.md`.
