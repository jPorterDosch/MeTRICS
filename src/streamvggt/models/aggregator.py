# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from typing import Tuple, Union, List

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from streamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        scale_token (bool): Add a scale token after the register tokens
            (depth_cond ScaleTokenCfg). Off: the pretrained token layout.
        scale_recurrent (bool): The scale token of frame t > 0 starts from a
            zero-init projection of frame t-1's last-layer output token
            (`scale_prev`), so it survives eviction from the KV cache.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        scale_token=False,
        scale_recurrent=False,
    ):
        super().__init__()
        if scale_recurrent and not scale_token:
            raise ValueError("scale_recurrent needs scale_token")

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Activation memory scales with clip_len x tokens/frame x blocks, not
        # param count; checkpointing trades ~30% compute to cut it. Set via
        # depth_cond train.grad_checkpoint (applies to the non-cached path only).
        self.grad_checkpointing = False

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # Optional scale token, between the register and the patch tokens; the
        # same first-frame / later-frame pair as the others. MetricStreamVGGT
        # copies a pretrained register token into it after loading weights. An
        # extra token is attended to, so enabling it changes the pretrained
        # output from step 0 (register 0's attention weight doubles).
        self.scale_token = None
        self.scale_carry = None
        if scale_token:
            self.scale_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
            nn.init.normal_(self.scale_token, std=1e-6)
            if scale_recurrent:
                # LayerNorm first: the carried state is the raw last-layer
                # residual stream, and training unrolls it 9 steps while
                # inference runs it for hundreds; normalising every step keeps
                # a gain the short unroll never constrains from compounding.
                # Zero-init projection: the recurrence starts as a no-op.
                self.scale_carry = nn.Sequential(
                    OrderedDict(norm=nn.LayerNorm(2 * embed_dim), proj=nn.Linear(2 * embed_dim, embed_dim))
                )
                nn.init.zeros_(self.scale_carry.proj.weight)
                nn.init.zeros_(self.scale_carry.proj.bias)
                # [first frame, later frames]: the carry's multiplier, picked
                # like the first-frame token (from the cache's shape), so an
                # exported graph fed a zero state at frame 0 adds nothing
                # there and one graph serves every frame. Not in the state dict.
                self.register_buffer(
                    "_carry_gate", torch.tensor([0.0, 1.0]).reshape(1, 2, 1, 1), persistent=False
                )
        self.scale_token_idx = 1 + num_register_tokens if scale_token else None

        # The patch tokens start after the camera, register (and scale) tokens
        self.patch_start_idx = 1 + num_register_tokens + int(scale_token)

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Run the frozen RGB patch-embed encoder.

        Args:
            images: [B, S, 3, H, W] in range [0, 1].
        Returns:
            patch tokens [B*S, P_patch, C].

        Split out from forward() so the deterministic frozen-encoder output can
        be precomputed and cached (see streamvggt.depth_cond.cache).
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        return patch_tokens

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        patch_tokens=None,
        injected_patch_feats=None,
        scale_prev=None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            patch_tokens: optional precomputed output of embed_patches(images)
                (frozen-encoder feature cache path). Must be numerically the
                tensor embed_patches would produce for these images.
            injected_patch_feats: optional depth-conditioning features added
                residually to the RGB patch tokens BEFORE any attention block
                (i.e. before the causal K/V cache). [B, S, P_patch, C] or
                [B*S, P_patch, C]. None -> unmodified behavior.
            scale_prev: the previous frame's last-layer scale token
                ([B, 2C], read off the output list at scale_token_idx), for a
                recurrent scale token on the KV-cache path; None on frame 0.

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if use_cache and past_key_values[0] is not None:
            _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true += 1
        else:
            S_true = S

        if use_cache and S > 1:
            raise ValueError(
                f"KV-cache path expects one frame at a time (S=1), got S={S}; "
                "cached global attention appends exactly one frame's K/V per call"
            )

        if patch_tokens is None:
            patch_tokens = self.embed_patches(images)

        _, P, C = patch_tokens.shape

        if injected_patch_feats is not None:
            if injected_patch_feats.shape not in ((B, S, P, C), (B * S, P, C)):
                raise ValueError(
                    f"injected_patch_feats shape {tuple(injected_patch_feats.shape)} does not "
                    f"match patch tokens: expected {(B, S, P, C)} or {(B * S, P, C)}"
                )
            patch_tokens = patch_tokens + injected_patch_feats.reshape(B * S, P, C)

        if use_cache:
            # slice_expand_and_flatten flattens (B, S, X, C) frame-axis-fastest,
            # so the CURRENT frame sits at the end of each batch row. Slicing the
            # flat tensor with [-1:] keeps only batch element B-1 -- correct at
            # B=1, a shape error the moment B>1. Unflatten first, then take the
            # last frame of every row. The frame-0 vs later-frame token choice
            # still rides on the traced S_true, which is what keeps one exported
            # graph legal for both.
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full.reshape(B, -1, *camera_token_full.shape[1:])[:, -1]

            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full.reshape(B, -1, *register_token_full.shape[1:])[:, -1]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        special = [camera_token, register_token]
        if self.scale_token is not None:
            if use_cache:
                scale_full = slice_expand_and_flatten(self.scale_token, B, S_true)
                scale_token = scale_full.reshape(B, -1, *scale_full.shape[1:])[:, -1]
            else:
                scale_token = slice_expand_and_flatten(self.scale_token, B, S)
            if scale_prev is not None:
                if self.scale_carry is None or not use_cache:
                    raise ValueError(
                        "scale_prev needs a recurrent scale token on the KV-cache "
                        "path (one frame per call)"
                    )
                gate_full = slice_expand_and_flatten(self._carry_gate, B, S_true)
                gate = gate_full.reshape(B, -1, 1, 1)[:, -1]  # [B, 1, 1]: 0 on frame 0
                scale_token = scale_token + self.scale_carry(scale_prev).reshape(B, 1, C) * gate
            special.append(scale_token)
        elif scale_prev is not None:
            raise ValueError("scale_prev given to an aggregator without a scale token")
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([*special, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]
                        tokens, global_idx, global_intermediates, new_kv = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx
                        )
                        past_key_values[global_idx - 1] = new_kv
                    else: 
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            blk = self.frame_blocks[frame_idx]
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                # pos is captured as a default, not passed as a checkpoint input:
                # it may be None (RoPE disabled) and needs no gradient either way
                tokens = torch.utils.checkpoint.checkpoint(
                    lambda t, b=blk, p=pos: b(t, pos=p), tokens, use_reentrant=False
                )
            else:
                tokens = blk(tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None
                
            if use_cache and self.grad_checkpointing and self.training and torch.is_grad_enabled():
                # the one-frame-at-a-time training forward (depth_cond
                # train.window): checkpointed like the parallel path, the past
                # K/V passed as inputs so their gradient reaches earlier frames
                blk = self.global_blocks[global_idx]
                past = () if past_key_values_block is None else tuple(past_key_values_block)
                tokens, block_kv = torch.utils.checkpoint.checkpoint(
                    lambda t, *kv, b=blk, p=pos: b(
                        t, pos=p, past_key_values=kv if kv else None, use_cache=True
                    ),
                    tokens, *past, use_reentrant=False,
                )
            elif use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True
                )
            else:
                blk = self.global_blocks[global_idx]
                if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                    # pos/attn_mask captured as defaults (pos may be None with
                    # RoPE disabled; neither needs gradients)
                    tokens = torch.utils.checkpoint.checkpoint(
                        lambda t, b=blk, p=pos, m=attn_mask: b(t, pos=p, attn_mask=m),
                        tokens, use_reentrant=False,
                    )
                else:
                    tokens = blk(tokens, pos=pos, attn_mask=attn_mask)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
