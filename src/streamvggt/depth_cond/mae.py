"""MAE-style spatial encoder and training-only depth reconstruction head."""

import math

import torch
import torch.nn as nn


def _sincos_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Return a deterministic [length, dim] sinusoidal embedding."""
    if dim == 0:
        return torch.empty(length, 0, device=device)
    pairs = dim // 2
    if pairs:
        omega = torch.arange(pairs, device=device, dtype=torch.float32)
        omega = torch.exp(-math.log(10_000.0) * omega / max(pairs - 1, 1))
        phase = (
            torch.arange(length, device=device, dtype=torch.float32)[:, None] * omega
        )
        pos = torch.cat([phase.sin(), phase.cos()], dim=1)
    else:
        pos = torch.empty(length, 0, device=device)
    if pos.shape[1] < dim:
        pos = torch.cat([pos, pos.new_zeros(length, dim - pos.shape[1])], dim=1)
    return pos


def _sincos_2d(
    height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return a [1, height*width, dim] embedding in row-major patch order."""
    y_dim = dim // 2
    x_dim = dim - y_dim
    y = _sincos_1d(height, y_dim, device)[:, None, :].expand(height, width, y_dim)
    x = _sincos_1d(width, x_dim, device)[None, :, :].expand(height, width, x_dim)
    return torch.cat([y, x], dim=-1).reshape(1, height * width, dim).to(dtype=dtype)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention and MLP block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class MAEDepthEncoder(nn.Module):
    """Encode every spatial patch independently per frame."""

    def __init__(self, dim: int, depth: int, num_heads: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(2, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_channels = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,S,2,H,W] -> attention over P within each of the B*S frame rows.
        B, S, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"MAE input size {(H, W)} must be divisible by patch_size "
                f"{self.patch_size}"
            )
        patches = self.patch_embed(x.flatten(0, 1))
        ph, pw = patches.shape[-2:]
        latent = patches.flatten(2).transpose(1, 2)
        latent = latent + _sincos_2d(
            ph, pw, self.out_channels, latent.device, latent.dtype
        )
        for block in self.blocks:
            latent = block(latent)
        latent = self.norm(latent)
        latent = latent.transpose(1, 2).reshape(B, S, self.out_channels, ph, pw)
        return latent


class MAEDepthDecoder(nn.Module):
    """Training-only head reconstructing transformed disparity patches."""

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(encoder_dim, decoder_dim)
        decoder_heads = min(num_heads, decoder_dim)
        while decoder_dim % decoder_heads:
            decoder_heads -= 1
        self.blocks = nn.ModuleList(
            [TransformerBlock(decoder_dim, decoder_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size**2)

    def forward(self, latent: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """Decode [B*S,P,C] into [B*S,1,H,W] transformed disparity."""
        ph, pw = grid_hw
        if latent.shape[1] != ph * pw:
            raise ValueError(
                f"latent has {latent.shape[1]} patches, grid {grid_hw} has {ph * pw}"
            )
        x = self.input_proj(latent)
        x = x + _sincos_2d(ph, pw, x.shape[-1], x.device, x.dtype)
        for block in self.blocks:
            x = block(x)
        patches = self.pred(self.norm(x))
        patches = patches.reshape(-1, ph, pw, self.patch_size, self.patch_size)
        image = patches.permute(0, 1, 3, 2, 4).reshape(
            -1, 1, ph * self.patch_size, pw * self.patch_size
        )
        return image
