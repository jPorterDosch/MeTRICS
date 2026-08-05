"""KV-cache <-> flat-tensor-list conversion for the streaming ONNX export.

Only the aggregator's 24 GLOBAL attention blocks cache (the frame blocks are
within-frame and stateless); each cache entry is a (K, V) pair, each of shape
[B, num_heads, n_frames, P, head_dim] with the FRAME axis at dim 2 -- that is
the axis the cached attention concatenates on (layers/attention.py) and the
one exported graphs mark dynamic.

ONNX I/O is flat, ordered, and named; these helpers define that contract in
exactly one place: block-major, K before V ("past_k_00", "past_v_00",
"past_k_01", ... / "new_k_00", ...).
"""

from collections.abc import Sequence

import torch

# == Aggregator(depth=24); assert_cache_layout() checks it against the live
# model so a config drift fails loudly instead of silently mis-slicing.
NUM_GLOBAL_BLOCKS = 24


def cache_input_names() -> list[str]:
    names = []
    for i in range(NUM_GLOBAL_BLOCKS):
        names += [f"past_k_{i:02d}", f"past_v_{i:02d}"]
    return names


def cache_output_names() -> list[str]:
    names = []
    for i in range(NUM_GLOBAL_BLOCKS):
        names += [f"new_k_{i:02d}", f"new_v_{i:02d}"]
    return names


def flatten_cache(
    past_kv: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> list[torch.Tensor]:
    """[(k, v)] * 24 -> [k0, v0, k1, v1, ...] (48 tensors)."""
    if len(past_kv) != NUM_GLOBAL_BLOCKS:
        raise ValueError(
            f"expected {NUM_GLOBAL_BLOCKS} cache entries, got {len(past_kv)}"
        )
    flat: list[torch.Tensor] = []
    for kv in past_kv:
        k, v = kv
        flat += [k, v]
    return flat


def unflatten_cache(
    flat: Sequence[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """[k0, v0, ...] (48 tensors) -> [(k, v)] * 24.

    Always returns a FRESH list: the aggregator writes new entries into the
    list it is handed (aggregator.py `past_key_values[global_idx - 1] = ...`),
    so sharing one list across steps would alias state between calls.
    """
    if len(flat) != 2 * NUM_GLOBAL_BLOCKS:
        raise ValueError(f"expected {2 * NUM_GLOBAL_BLOCKS} tensors, got {len(flat)}")
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(NUM_GLOBAL_BLOCKS)]


def empty_cache(
    n_tokens: int,
    num_heads: int = 16,
    head_dim: int = 64,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
) -> list[torch.Tensor]:
    """batch_size * 48 zero-length cache tensors [batch_size, heads, 0, n_tokens, head_dim] -- the
    'no frames seen yet' state the consumer feeds at frame 0 (the graph
    concats with the empty tensor and selects the frame-0 token from the
    zero cache length). Each batch element has independent cache."""
    shape = (batch_size, num_heads, 0, n_tokens, head_dim)
    return [
        torch.zeros(shape, device=device, dtype=dtype)
        for _ in range(2 * NUM_GLOBAL_BLOCKS)
    ]


def assert_cache_layout(aggregator) -> None:
    """Fail fast if the live model disagrees with the constants above."""
    if aggregator.depth != NUM_GLOBAL_BLOCKS:
        raise AssertionError(
            f"aggregator.depth={aggregator.depth} != NUM_GLOBAL_BLOCKS="
            f"{NUM_GLOBAL_BLOCKS}; the cache I/O contract must be updated"
        )
