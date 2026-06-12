# SPDX-License-Identifier: Apache-2.0
"""
MARSHAL protocol definitions for kvtunnel KV tunneling.

This module defines the protocol for:
- MARSHAL: StreamingLLM-pack a prompt's KV into a header-prepended workspace
  blob and return the per-TP-rank tunneled-request manifest.
- WAIT_STORE: block until a chunk's STORE has committed on every TP rank
  (gates the kvtunnel re-tunnel cycle on the previous cycle's STORE).
- MARSHAL_FREE: reclaim a workspace entry by its rendezvous handle.

Split out of ``engine.py`` so the kvtunnel tunneling protocol lives in its own
module (mirroring ``blend.py`` / ``observability.py``), keeping the upstream
engine protocol free of kvtunnel additions and minimizing merge conflicts.
"""

# First Party
from kvtunnel.marshal.pack import TunneledRequestMetadata

from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

# Define request names for this protocol group
REQUEST_NAMES = [
    "MARSHAL",
    "WAIT_STORE",
    "MARSHAL_FREE",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """
    Returns protocol definitions for kvtunnel MARSHAL operations.

    Returns:
        Dictionary mapping request names to their protocol definitions
    """
    return {
        # KV tunneling — pack the unmarshalled KV for `real_prompt` into a
        # header-prepended CPU blob and stash it under `marshal_handle` in
        # the server's workspace dict. A subsequent RETRIEVE carrying the
        # same `marshal_handle` scatters that blob into vLLM's paged cache.
        # Payload:
        #   - marshal_handle: str - rendezvous key for the workspace entry
        #   - real_prompt: list[int] - token IDs of the real prompt
        #   - method_params: dict - method-specific params (num_sinks,
        #       window_size, cache_salt); current impl hardcodes StreamingLLM
        #   - worker_id: int - GPU instance ID whose KV cache holds the prompt
        # Returns: tuple[bool, int, str,
        #   dict[int, TunneledRequestMetadata], int] —
        #   (success, num_fake, error_message, tunneled_request_per_rank,
        #   matched_prefix_len).
        #   On success, error_message == "" and tunneled_request_per_rank
        #   maps tp_rank -> the per-layer TunneledInfo manifest the connector
        #   stages on the scheduler so workers can build attention metadata
        #   without re-parsing block bytes. num_fake is the number of fake
        #   slots the marshalled blob occupies, 0 on failure (manifest is {}).
        #   matched_prefix_len is the chunk-aligned token count of the prefix
        #   actually packed (== the chunk-aligned length of real_prompt on a
        #   full match; the proxy pre-truncates to a chunk multiple. 0 on
        #   failure); the proxy derives the unmatched suffix from it
        #   (partial-prefix tunneling).
        "MARSHAL": ProtocolDefinition(
            payload_classes=[str, list[int], dict, int],
            response_class=tuple[
                bool, int, str, dict[int, TunneledRequestMetadata], int
            ],
            handler_type=HandlerType.BLOCKING,
        ),
        # WAIT_STORE: block until the chunk covering
        # token_ids[0:end_offset] is committed and readable on every
        # TP rank, or until wait_timeout_ms elapses. Used by the
        # kvtunnel proxy's re-tunnel cycle loop to gate the next
        # cycle's MARSHAL on the previous cycle's STORE having
        # landed in L1.
        # Payload:
        #   - token_ids: list[int] — the running real prompt
        #     (prompt + decoded so far).
        #   - end_offset: int — the running prompt length; the
        #     handler hashes [0:end_offset] and waits on the
        #     trailing chunk_hash.
        #   - worker_id: int — GPU instance ID; the handler looks up
        #     the registered GPU context for worker_id to learn the TP
        #     world size, then iterates over TP ranks via
        #     ipc_key_to_object_keys's worker_id=None expansion.
        #   - wait_timeout_ms: int — handler's event.wait deadline.
        #     Proxy supplies it per-call so the timeout is
        #     configurable (default 3000 ms on the proxy side) and
        #     supports exponential backoff on retry.
        # Returns: str — "Ready" if all per-rank chunk objects
        # were readable within the deadline, "Pending" otherwise.
        "WAIT_STORE": ProtocolDefinition(
            payload_classes=[list[int], int, int, int],
            response_class=str,
            handler_type=HandlerType.BLOCKING,
        ),
        # MARSHAL_FREE: reclaim the workspace entry stashed under
        # `marshal_handle` once the request/cycle that consumed it has
        # finished. Fired by the proxy (which mints the handle) after the
        # vLLM completion returns — by then the blob's RETRIEVE H2D has
        # drained, so no DMA reads the freed pinned bytes. The handler
        # pops the workspace entry (MarshalWorkspace.free) and schedules
        # `ref_count_down` on the gpu_context stream (stream-ordered to cover the
        # TTL/abort path); it returns once the free is *enqueued*, so the
        # ack does NOT mean the buffer is reclaimed. Unknown handle is a
        # no-op.
        # Payload:
        #   - marshal_handle: str - the workspace entry to reclaim.
        # Returns: None
        "MARSHAL_FREE": ProtocolDefinition(
            payload_classes=[str],
            response_class=None,
            handler_type=HandlerType.BLOCKING,
        ),
    }
