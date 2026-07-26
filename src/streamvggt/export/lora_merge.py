"""Fold trained LoRA adapters into the base attention weights for export.

The exported graph must not contain adapter matmuls: merging W += B @ A *
scaling makes the LoRA'd model byte-identical in weights-space to a plain
StreamVGGT, so all export variants share one graph topology and the ONNX file
carries zero extra ops. In eval mode the branch dropout is Identity, so the
merged linear equals the wrapped forward EXACTLY (verified by the parity
test's --cpu-unit check).

Reads the wrapper classes from depth_cond.lora without modifying them.
"""

import torch
import torch.nn as nn

from streamvggt.depth_cond.lora import _QKV_INDEX, LoRALinear, LoRAQKV


@torch.no_grad()
def _merged_base(branch, weight_slice: torch.Tensor) -> None:
    """Add one low-rank branch's delta into a (slice of a) base weight."""
    delta = (branch.lora_B @ branch.lora_A) * branch.scaling
    weight_slice += delta.to(weight_slice.dtype)


@torch.no_grad()
def merge_lora(aggregator: nn.Module) -> int:
    """Merge every LoRAQKV / LoRALinear in the aggregator's attention blocks
    into its base nn.Linear and swap the plain linear back in. Returns the
    number of linears restored. Idempotent on an unwrapped aggregator (0)."""
    n = 0
    for blocks in (aggregator.frame_blocks, aggregator.global_blocks):
        for block in blocks:
            attn = block.attn
            if isinstance(attn.qkv, LoRAQKV):
                wrapper = attn.qkv
                dim = wrapper.dim
                for t in wrapper.targets:
                    i = _QKV_INDEX[t]
                    _merged_base(
                        wrapper.adapters[t],
                        wrapper.base.weight[i * dim : (i + 1) * dim],
                    )
                attn.qkv = wrapper.base
                n += 1
            if isinstance(attn.proj, LoRALinear):
                wrapper = attn.proj
                _merged_base(wrapper.adapter, wrapper.base.weight)
                attn.proj = wrapper.base
                n += 1
    return n
