# SPDX-License-Identifier: Apache-2.0
"""StreamingLLM pack kernel for KV tunneling.

Selects sink + sliding-window tokens from unmarshalled KV chunks and writes
a header-prepended blob into a fresh pinned-CPU TensorMemoryObj. A later
RETRIEVE against this workspace blob scatters the bytes into vLLM's paged
cache verbatim, where the plugin attention backend reinterprets them.

Byte layout matches design/kv-tunneling.md §Fixed Header Layout.
"""

# Standard
import struct

# Third Party
import torch

# First Party
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


# Method id registry for the `compression_type` byte in the metadata block.
# StreamingLLM = no compression, post-RoPE passthrough.
STREAMING_LLM_METHOD_ID: int = 0

# Fixed 28-byte header prefix common to every method:
#   bytes 0..15   — 4 float32 magic values (+inf, -inf, +inf, -inf)
#   bytes 16..19  — pos_id_len       (int32, little-endian)
#   bytes 20..23  — metadata_len     (int32, little-endian)
#   bytes 24..27  — num_fake_marshalled (int32, little-endian)
FIXED_HEADER_SIZE: int = 28

# StreamingLLM's method-specific metadata block (12 bytes, per
# kv-tunneling.md §Example 2). compression_type=0 (none), packing_ratio=1,
# head_mask=0xFF, num_active_heads=8 by default.
STREAMING_LLM_METADATA_SIZE: int = 12


def _build_header_bytes(
    *,
    pos_ids: list[int],
    num_fake_marshalled: int,
    num_active_heads: int,
) -> bytes:
    """Serialize the full header (fixed prefix + pos_ids + StreamingLLM meta)
    as a contiguous bytes blob.

    Args:
        pos_ids: Real positions of retained tokens (sinks + window).
        num_fake_marshalled: Number of fake slots the marshalled region
            occupies in vLLM's paged view (header slots + data slots).
        num_active_heads: Number of KV heads retained. StreamingLLM keeps
            all heads, so this matches the source tensor's head dimension.

    Returns:
        The full header as a bytes object, ready to copy into the output
        buffer at offset 0.
    """
    buf = bytearray()
    # Fixed prefix: magic (4 * float32), pos_id_len, metadata_len, num_fake.
    for magic in (float("inf"), float("-inf"), float("inf"), float("-inf")):
        buf.extend(struct.pack("<f", magic))
    buf.extend(struct.pack("<i", len(pos_ids)))
    buf.extend(struct.pack("<i", STREAMING_LLM_METADATA_SIZE))
    buf.extend(struct.pack("<i", num_fake_marshalled))
    assert len(buf) == FIXED_HEADER_SIZE

    # pos_ids block (int32 each).
    for p in pos_ids:
        buf.extend(struct.pack("<i", p))

    # StreamingLLM metadata block (12 bytes).
    buf.append(STREAMING_LLM_METHOD_ID)  # compression_type (byte 0)
    buf.append(0)  # packed_dtype (byte 1, N/A for BF16)
    buf.extend(struct.pack("<i", len(pos_ids)))  # num_real_tokens (bytes 2..5)
    buf.extend(struct.pack("<H", 1))  # packing_ratio (bytes 6..7)
    buf.append(0xFF)  # head_mask (byte 8)
    buf.append(num_active_heads & 0xFF)  # num_active_heads (byte 9)
    buf.extend(struct.pack("<H", 0))  # reserved (bytes 10..11)

    return bytes(buf)


def _select_positions(
    *,
    real_prompt_len: int,
    num_sinks: int,
    window_size: int,
) -> list[int]:
    """Pick the sink + sliding-window positions retained by StreamingLLM.

    Sinks come first (positions 0..num_sinks-1), then the trailing
    `window_size` positions. Overlap is deduplicated — for short prompts
    the window may start at num_sinks and all retained positions are
    contiguous.

    Args:
        real_prompt_len: Number of tokens in the real (un-tunneled) prompt.
        num_sinks: Number of leading "attention sink" tokens to keep.
        window_size: Number of trailing tokens to keep as the sliding window.

    Returns:
        A list of retained positions in ascending order.
    """
    sink_end = min(num_sinks, real_prompt_len)
    window_start = max(sink_end, real_prompt_len - window_size)
    sinks = list(range(sink_end))
    window = list(range(window_start, real_prompt_len))
    return sinks + window


def streaming_llm_pack(
    *,
    mem_objs: list[MemoryObj],
    chunk_size: int,
    real_prompt_len: int,
    num_sinks: int,
    window_size: int,
) -> tuple[MemoryObj, int]:
    """Select retained tokens from unmarshalled chunks, prepend header.

    The output MemoryObj has the same per-slot layout as the input chunks
    (same dtype, same shape on axes [1:]) so the existing LMCache scatter
    kernel can copy it into vLLM's paged buffer verbatim. The first-axis
    length is num_fake = num_header_slots + len(pos_ids).

    Args:
        mem_objs: Unmarshalled KV chunks for the real prompt. Each chunk's
            first axis must equal ``chunk_size``.
        chunk_size: Number of slots per chunk (LMCache's tokenization stride).
        real_prompt_len: Total number of real-prompt tokens covered by
            ``mem_objs``; must satisfy
            ``real_prompt_len <= len(mem_objs) * chunk_size``.
        num_sinks: Number of leading sink tokens to retain.
        window_size: Number of trailing window tokens to retain.

    Returns:
        A tuple ``(packed, num_fake)`` where ``packed`` is a pinned-CPU
        TensorMemoryObj holding the marshalled blob and ``num_fake`` is
        the number of fake slots the blob occupies.

    Raises:
        ValueError: If ``mem_objs`` is empty or a chunk's first-axis
            length does not match ``chunk_size``.
    """
    if not mem_objs:
        raise ValueError("mem_objs must be non-empty")

    ref = mem_objs[0].raw_data
    ref_shape = list(ref.shape)
    if len(ref_shape) < 2 or ref_shape[0] != chunk_size:
        raise ValueError(
            f"expected chunk first-axis {chunk_size}, got shape {tuple(ref_shape)}"
        )
    ref_dtype = ref.dtype

    pos_ids = _select_positions(
        real_prompt_len=real_prompt_len,
        num_sinks=num_sinks,
        window_size=window_size,
    )
    retained = len(pos_ids)

    # num_active_heads is the KV head dimension of the source tensor.
    # Layout convention across LMCache is [chunk_size, ..., num_kv_heads,
    # head_size] (head axis is second-to-last), so we read index -2.
    num_active_heads = ref_shape[-2]

    slot_elements = 1
    for dim in ref_shape[1:]:
        slot_elements *= dim
    slot_bytes = slot_elements * ref_dtype.itemsize

    header_size = FIXED_HEADER_SIZE + retained * 4 + STREAMING_LLM_METADATA_SIZE
    num_header_slots = (header_size + slot_bytes - 1) // slot_bytes
    num_fake = num_header_slots + retained

    out_tensor = torch.empty(
        [num_fake] + ref_shape[1:],
        dtype=ref_dtype,
        pin_memory=True,
    )

    # Write header bytes into slots 0..num_header_slots-1. The uint8 view
    # aliases the output tensor's storage for fresh contiguous allocations.
    header_bytes = _build_header_bytes(
        pos_ids=pos_ids,
        num_fake_marshalled=num_fake,
        num_active_heads=num_active_heads,
    )
    out_bytes_view = out_tensor.view(torch.uint8).reshape(-1)
    out_bytes_view[: len(header_bytes)] = torch.frombuffer(
        bytearray(header_bytes), dtype=torch.uint8
    )

    # Copy selected real tokens into slots starting at num_header_slots.
    for i, pos in enumerate(pos_ids):
        chunk_idx, in_chunk_offset = divmod(pos, chunk_size)
        if chunk_idx >= len(mem_objs):
            raise ValueError(
                f"pos {pos} needs chunk {chunk_idx} but only "
                f"{len(mem_objs)} chunks provided"
            )
        out_tensor[num_header_slots + i] = mem_objs[chunk_idx].raw_data[in_chunk_offset]

    metadata = MemoryObjMetadata(
        shape=out_tensor.shape,
        dtype=ref_dtype,
        address=out_tensor.data_ptr(),
        phy_size=out_tensor.numel() * ref_dtype.itemsize,
        ref_count=0,
        fmt=MemoryFormat.UNDEFINED,
    )
    packed = TensorMemoryObj(
        raw_data=out_tensor,
        metadata=metadata,
        parent_allocator=None,
    )
    return packed, num_fake
