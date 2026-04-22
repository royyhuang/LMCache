# SPDX-License-Identifier: Apache-2.0
"""
Engine protocol definitions for core KV cache operations.

This module defines the protocol for:
- REGISTER_KV_CACHE: Register a KV cache instance with the server
- UNREGISTER_KV_CACHE: Unregister a KV cache instance
- STORE: Store KV cache blocks to the server
- RETRIEVE: Retrieve KV cache blocks from the server
- LOOKUP: Submit a prefix lookup and return a prefetch job ID
- QUERY_PREFETCH_STATUS: Poll a prefetch job for its result
- END_SESSION: End a session and clean up associated resources
"""

# First Party
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheEngineKey,
    KVCache,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "REGISTER_KV_CACHE",
    "UNREGISTER_KV_CACHE",
    "STORE",
    "RETRIEVE",
    "LOOKUP",
    "QUERY_PREFETCH_STATUS",
    "QUERY_PREFETCH_LOOKUP_HITS",
    "FREE_LOOKUP_LOCKS",
    "END_SESSION",
    "MARSHAL",
]

# Type alias for cache keys
KeyType = IPCCacheEngineKey


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for engine operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # Register KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - kv_cache: KVCache - The KV cache configuration
        #   - model_name: str - Name of the model associated with the engine
        #   - world_size: int - World size of the engine
        #   - layout_hints: LayoutHints - See custom_types.LayoutHints.
        # Returns: None
        "REGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int, KVCache, str, int, LayoutHints],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Unregister KV Cache
        # Payload:
        #   - instance_id: int - Unique identifier for the vLLM instance
        # Returns: None
        "UNREGISTER_KV_CACHE": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Store KV cache blocks
        # Payload:
        #   - key: KeyType - Cache key to store
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[int] - GPU block IDs containing the data
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "STORE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[int], bytes],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Retrieve KV cache blocks
        # Payload:
        #   - key: KeyType - Cache key to retrieve
        #   - instance_id: int - Unique identifier for the vLLM instance
        #   - gpu_block_ids: list[int] - GPU block IDs to store retrieved data
        #   - event_ipc_handle: bytes - CUDA event IPC handle for synchronization
        #   - skip_first_n_tokens: int - Number of tokens to skip writing at the
        #     start of the retrieve range (to avoid overwriting APC-shared blocks)
        #   - marshal_handle: str - KV tunneling rendezvous key. Non-empty means
        #     the server reads WORKSPACE[marshal_handle] instead of doing the
        #     token-hash lookup. Empty string is the stock (non-tunneled) path.
        # Returns: tuple[bytes, bool] - (CUDA event handle, success flag)
        "RETRIEVE": ProtocolDefinition(
            payload_classes=[KeyType, int, list[int], bytes, int, str],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Submit a prefix lookup; job is tracked server-side by request_id
        # Payload:
        #   - key: KeyType - Cache key to look up
        #   - tp_size: int - Tensor-parallel size for
        #       MLA multi-reader locking
        # Returns: None
        "LOOKUP": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the status of a prefetch job by request_id
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        # Returns: int | None - Chunk count when done, None if still in progress
        "QUERY_PREFETCH_STATUS": ProtocolDefinition(
            payload_classes=[str],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Query the lookup hit chunks before the prefetch is done
        # Payload:
        #   - request_id: str - The external request ID passed in the lookup key
        # Returns: int | None - Chunk count if lookup is done, None if still in progress
        "QUERY_PREFETCH_LOOKUP_HITS": ProtocolDefinition(
            payload_classes=[str],
            response_class=int | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Free locks (release read locks without a full RETRIEVE)
        # Payload:
        #   - key: KeyType - Cache key whose read locks
        #       to release
        #   - tp_size: int - Tensor-parallel size for
        #       MLA multi-reader locking
        # Returns: None
        "FREE_LOOKUP_LOCKS": ProtocolDefinition(
            payload_classes=[KeyType, int],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # End session
        # Payload:
        #   - request_id: str - Request ID of the session to end
        # Returns: None
        "END_SESSION": ProtocolDefinition(
            payload_classes=[str],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
        # KV tunneling — pack the unmarshalled KV for `real_prompt` into a
        # header-prepended CPU blob and stash it under `marshal_handle` in
        # the server's workspace dict. A subsequent RETRIEVE carrying the
        # same `marshal_handle` scatters that blob into vLLM's paged cache.
        # See design/kv-tunneling.md for the byte layout.
        # Payload:
        #   - marshal_handle: str - rendezvous key for the workspace entry
        #   - real_prompt: list[int] - token IDs of the real prompt
        #   - method_params: dict - method-specific params (num_sinks,
        #       window_size, cache_salt); current impl hardcodes StreamingLLM
        #   - worker_id: int - GPU instance ID whose KV cache holds the prompt
        # Returns: tuple[bool, int, str] - (success, num_fake, error_message).
        #   On success, error_message == "". num_fake is the number of fake
        #   slots the marshalled blob occupies, 0 on failure.
        "MARSHAL": ProtocolDefinition(
            payload_classes=[str, list[int], dict, int],
            response_class=tuple[bool, int, str],
            handler_type=HandlerType.BLOCKING,
        ),
    }
