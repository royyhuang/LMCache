# SPDX-License-Identifier: Apache-2.0
"""Shared context and layout descriptor registry for engine modules."""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, TypedDict
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import StorageManagerConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import EventBus, get_event_bus
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if TYPE_CHECKING:
    # Type-only imports. GPUContextEntry lives in modules/gpu_transfer.py,
    # which imports MPCacheEngineContext, so a runtime import here would
    # cycle. The registry only stores/returns the value type; it never
    # constructs it (register_kv_cache, the sole writer, builds it).
    # MarshalWorkspace is kept type-only so importing engine_context (which
    # EVERY MP server does, including non-GPU/non-tunnel ones) does NOT pull
    # in kvtunnel.marshal.pack (and torch) at runtime — only MarshalModule,
    # the sole constructor, imports it for real.
    from lmcache.v1.multiprocess.marshal_workspace import MarshalWorkspace
    from lmcache.v1.multiprocess.modules.gpu_transfer import GPUContextEntry

logger = init_logger(__name__)


class ShmPoolInfo(TypedDict):
    """Shared-memory pool metadata returned during registration."""

    shm_name: str
    pool_size: int


@dataclass
class _LayoutDescEntry:
    """Stored layout descriptor and its active registration count."""

    layout_desc: MemoryLayoutDesc
    ref_count: int


class LayoutDescRegistry:
    """Thread-safe registry mapping (model_name, world_size) to MemoryLayoutDesc.

    Modules write to this registry when KV caches are registered.
    Consumers (e.g. LookupModule) read from it to find layout descriptors
    for prefetch tasks. Multiple worker instances can share the same
    ``(model_name, world_size)`` entry, so the registry keeps the descriptor
    until the last matching registration is unregistered.
    """

    def __init__(self) -> None:
        # Key: (model_name, world_size) -> layout descriptor entry
        self._registry: dict[tuple[str, int], _LayoutDescEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_name: str,
        world_size: int,
        layout_desc: MemoryLayoutDesc,
    ) -> None:
        """Register a layout descriptor for a (model_name, world_size) pair.

        Re-registering the same pair increments the active registration
        count. The latest descriptor is retained for lookups.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                self._registry[key] = _LayoutDescEntry(
                    layout_desc=layout_desc,
                    ref_count=1,
                )
                return

            entry.layout_desc = layout_desc
            entry.ref_count += 1

    def unregister(self, model_name: str, world_size: int) -> None:
        """Unregister one layout descriptor registration for a pair.

        The descriptor is removed only when the last active registration for
        the pair is unregistered.

        Args:
            model_name: The model name.
            world_size: The world size.
        """
        key = (model_name, world_size)
        with self._lock:
            entry = self._registry.get(key)
            if entry is None:
                return

            if entry.ref_count <= 1:
                self._registry.pop(key)
                return

            entry.ref_count -= 1

    def find(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Look up a layout descriptor by (model_name, world_size).

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The layout descriptor if found, otherwise None.
        """
        with self._lock:
            entry = self._registry.get((model_name, world_size))
            if entry is None:
                return None
            return entry.layout_desc


class GPUContextRegistry:
    """Thread-safe registry mapping instance_id to its GPU context entry.

    The single per-process home for the ``instance_id -> GPUContextEntry``
    map that GPU lifecycle (``register_kv_cache`` / ``unregister_kv_cache``)
    writes and that store / retrieve plus the kvtunnel MARSHAL subsystem
    read. Promoted onto the shared context (alongside
    :class:`LayoutDescRegistry`) so MARSHAL can resolve a worker's context
    without reaching into ``GPUTransferModule`` private state.

    Intentionally diverges from :class:`LayoutDescRegistry` in two ways:

    * **Keyed by ``instance_id`` (single-owner), not ref-counted.** Each
      instance registers exactly once; there is no ``(model_name,
      world_size)`` sharing, so there is no ref count and ``unregister``
      simply pops.
    * **:meth:`get` is lock-free.** Unlike :meth:`LayoutDescRegistry.find`
      (which locks), :meth:`get` is a bare ``dict.get`` so the per-chunk
      store / retrieve read path is never serialized on this lock — exactly
      the GIL-atomic behavior the old ``GPUTransferModule._gpu_contexts``
      dict relied on. Only the writers (:meth:`register` /
      :meth:`unregister`, register-time / SYNC pool) and the snapshot /
      teardown helpers (:meth:`items` / :meth:`clear`) take the lock.

    :meth:`get` returns ``None`` on a miss; each caller raises its own
    error so the distinct, byte-stable messages at the call sites
    (``ValueError`` for store / retrieve, ``RuntimeError`` for the MARSHAL
    handlers and the tunneled-retrieve scatter) are preserved exactly.
    """

    def __init__(self) -> None:
        # Key: instance_id -> GPU context entry. Mutated only by
        # register/unregister under _lock; read lock-free via get().
        self._registry: dict[int, "GPUContextEntry"] = {}
        self._lock = threading.Lock()

    def register(self, instance_id: int, entry: "GPUContextEntry") -> None:
        """Register the GPU context entry for an instance.

        Single-owner: re-registering an ``instance_id`` overwrites the
        previous entry (matching the old dict-assignment semantics).

        Args:
            instance_id: The GPU instance ID to key the entry by.
            entry: The GPU context entry to store.
        """
        with self._lock:
            self._registry[instance_id] = entry

    def unregister(self, instance_id: int) -> None:
        """Remove the GPU context entry for an instance.

        Popping an unregistered ``instance_id`` is a no-op (matching the
        old ``dict.pop(id, None)`` semantics).

        Args:
            instance_id: The GPU instance ID to remove.
        """
        with self._lock:
            self._registry.pop(instance_id, None)

    def get(self, instance_id: int) -> "GPUContextEntry | None":
        """Look up the GPU context entry for an instance, lock-free.

        Deliberately does NOT take ``_lock`` — this is the per-chunk
        store / retrieve read path and must stay as cheap as the old bare
        dict read (GIL atomicity). This is the intentional divergence from
        :meth:`LayoutDescRegistry.find`, which locks.

        Args:
            instance_id: The GPU instance ID to look up.

        Returns:
            The registered entry, or ``None`` if the instance is not
            registered. Callers raise their own byte-stable error on
            ``None`` so the existing per-call-site messages are preserved.
        """
        return self._registry.get(instance_id)

    def items(self) -> list[tuple[int, "GPUContextEntry"]]:
        """Return a locked snapshot of (instance_id, entry) pairs.

        Copies under ``_lock`` so an iterating consumer (e.g.
        ``report_status``) cannot hit "dict changed size during iteration"
        against a concurrent register / unregister.

        Returns:
            A list copy of the current registry items.
        """
        with self._lock:
            return list(self._registry.items())

    def clear(self) -> None:
        """Drop all registered entries (locked teardown for ``close``)."""
        with self._lock:
            self._registry.clear()


class ChunkCommitNotifier:
    """WAIT_STORE barrier: wake waiters when a chunk's STORE commits.

    Owns ``chunk_hash -> list[threading.Event]`` plus its lock. The STORE
    finish-write callback calls :meth:`signal` after the L1 write commits;
    WAIT_STORE blocks in :meth:`wait`. No ``EventBus`` dependency — the
    barrier needs sub-ms wakeup precision without a thundering herd.

    Promoted onto the shared context so the signal side (STORE completion,
    owned by ``GPUTransferModule``) and the wait side (the MARSHAL
    subsystem's WAIT_STORE handler) meet through ctx rather than through
    ``GPUTransferModule`` private state.
    """

    def __init__(self) -> None:
        # chunk_hash -> waiters. Mutated by signal() (pop) and wait()
        # (register + finally-remove), both under _lock; set() runs
        # outside the lock so no waiter callback fires while holding it.
        self._pending: dict[bytes, list[threading.Event]] = {}
        self._lock = threading.Lock()

    def signal(self, chunk_hashes: list[bytes]) -> None:
        """Wake any waiters registered on the given chunk hashes.

        ``pop()`` under the lock, then ``set()`` outside it so the STORE
        side and WAIT_STORE side never deadlock on the notifier (no waiter
        callback runs inside ``event.set()`` while the lock is held).

        Args:
            chunk_hashes: Hashes whose pending waiters to signal.
        """
        for chunk_hash in chunk_hashes:
            with self._lock:
                events = self._pending.pop(chunk_hash, [])
            for event in events:
                event.set()

    @contextmanager
    def register(self, chunk_hash: bytes) -> Iterator[threading.Event]:
        """Register a waiter Event for ``chunk_hash`` for the with-body.

        Internalizes both the waiter registration and its ``finally``
        removal so callers never touch ``_pending`` directly and the
        no-event-leak invariant lives here. The Event is registered on
        entry — BEFORE the with-body — so the caller can interpose its
        post-registration ``is_ready`` re-check (which closes the
        register-after-signal race) and only then block on the yielded
        Event. The Event is removed on every exit path (signalled, timed
        out, or raised).

        Args:
            chunk_hash: The trailing chunk hash to wait on.

        Yields:
            The :class:`threading.Event` the STORE-side :meth:`signal`
            sets once the chunk commits. The caller blocks on it via
            ``event.wait(timeout)`` after its own is_ready re-check.
        """
        event = threading.Event()
        with self._lock:
            self._pending.setdefault(chunk_hash, []).append(event)
        try:
            yield event
        finally:
            # Always remove our Event so _pending doesn't leak, on every
            # exit path (signalled, timed out, or raised).
            with self._lock:
                waiters = self._pending.get(chunk_hash)
                if waiters is not None and event in waiters:
                    waiters.remove(event)
                    if not waiters:
                        self._pending.pop(chunk_hash, None)


class MPCacheEngineContext:
    """Shared infrastructure for all engine modules.

    Holds the storage manager, token hasher, session manager, event bus,
    and layout descriptor registry. Modules receive this context at init
    and use it for shared operations.

    Args:
        storage_manager_config: Configuration for the storage manager.
        chunk_size: Chunk size for KV cache operations.
        hash_algorithm: Hash algorithm for token hashing.
    """

    def __init__(
        self,
        storage_manager_config: StorageManagerConfig,
        chunk_size: int = 256,
        hash_algorithm: str = "blake3",
    ) -> None:
        self._chunk_size = chunk_size
        self.shm_pool_info: ShmPoolInfo = self._compute_shm_pool_info(
            storage_manager_config
        )
        self._storage_manager = StorageManager(storage_manager_config)
        self._token_hasher = TokenHasher(
            chunk_size=chunk_size, hash_algorithm=hash_algorithm
        )
        self._session_manager = SessionManager(self._token_hasher)
        self._event_bus = get_event_bus()
        self._layout_desc_registry = LayoutDescRegistry()
        # kvtunnel MARSHAL seams (see GPUContextRegistry /
        # ChunkCommitNotifier). The workspace is published lazily by the
        # owning module (GPU-mode-gated) so a non-GPU server never pins a
        # pool it cannot use; None until then.
        self._gpu_context_registry = GPUContextRegistry()
        self._chunk_commit_notifier = ChunkCommitNotifier()
        self._marshal_workspace: "MarshalWorkspace | None" = None

    @property
    def chunk_size(self) -> int:
        """Chunk size for KV cache operations."""
        return self._chunk_size

    @property
    def storage_manager(self) -> StorageManager:
        """The storage manager instance."""
        return self._storage_manager

    @property
    def token_hasher(self) -> TokenHasher:
        """The token hasher for computing chunk hashes."""
        return self._token_hasher

    @property
    def session_manager(self) -> SessionManager:
        """The session manager for request lifecycle tracking."""
        return self._session_manager

    @property
    def event_bus(self) -> EventBus:
        """The event bus for observability events."""
        return self._event_bus

    @property
    def layout_desc_registry(self) -> LayoutDescRegistry:
        """Registry mapping (model_name, world_size) to MemoryLayoutDesc."""
        return self._layout_desc_registry

    @property
    def gpu_context_registry(self) -> GPUContextRegistry:
        """Registry mapping instance_id to its registered GPU context."""
        return self._gpu_context_registry

    @property
    def chunk_commit_notifier(self) -> ChunkCommitNotifier:
        """WAIT_STORE notifier waking waiters on STORE commit."""
        return self._chunk_commit_notifier

    @property
    def marshal_workspace(self) -> "MarshalWorkspace | None":
        """The MARSHAL workspace, or None on a non-GPU server.

        Published by the workspace's owning module (the publish-via-setter
        seam) once the GPU-mode module is built; ``None`` before then.
        """
        return self._marshal_workspace

    @marshal_workspace.setter
    def marshal_workspace(self, workspace: "MarshalWorkspace | None") -> None:
        """Publish (or clear) the MARSHAL workspace instance on the context.

        Args:
            workspace: The per-process :class:`MarshalWorkspace` the owning
                GPU-mode module constructed, or ``None`` when that module's
                ``close`` releases the pool and drops the seam.
        """
        self._marshal_workspace = workspace

    def resolve_obj_keys(self, key: IPCCacheEngineKey) -> list[ObjectKey]:
        """Resolve object keys from an IPC cache key.

        Uses the session manager to track token state and the token hasher
        to compute chunk hashes for the requested range.

        Args:
            key: IPC cache key describing model/session/token range.

        Returns:
            Resolved object keys for the requested token range.

        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        chunk_hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        if key.worker_id is None:
            raise ValueError("Must resolve keys with worker_id != None")
        return ipc_key_to_object_keys(key, chunk_hashes)

    @staticmethod
    def _compute_shm_pool_info(
        storage_manager_config: StorageManagerConfig,
    ) -> ShmPoolInfo:
        """Compute normalized SHM pool metadata from storage config.

        Returns an empty pool (disabled SHM transport) when ``shm_name`` is
        empty or lazy memory mode is enabled. Otherwise strips any leading ``/``
        and ensures the name starts with ``lmcache_l1_pool_``.
        """
        mem_cfg = storage_manager_config.l1_manager_config.memory_config
        shm_name = mem_cfg.shm_name or ""
        if not shm_name or mem_cfg.use_lazy:
            return {"shm_name": "", "pool_size": 0}
        bare = shm_name.lstrip("/")
        if not bare.startswith("lmcache_l1_pool_"):
            shm_name = f"lmcache_l1_pool_{bare}"
        return {"shm_name": shm_name, "pool_size": mem_cfg.size_in_bytes}
