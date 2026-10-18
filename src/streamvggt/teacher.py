"""A frozen monocular depth teacher run online during training, for the
edge-aware loss (loss.types.LossConfig.edge_weight, gradient_loss.
teacher_edge_loss). Prompt Depth Anything takes its edge targets from a
per-scene Zip-NeRF; here Depth Anything 3 stands in, one frame at a time,
with no gradient and no stored predictions (1.5M training frames do not fit
on Lustre). Only the teacher's spatial gradient reaches the loss, after a
per-frame scale fit onto the GT, so its own metric scale does not matter and
no focal length is needed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class TeacherModel(str, enum.Enum):
    NONE = "none"
    DA3 = "da3"  # DA3METRIC-LARGE, the verified da3_metric baseline arm's weights


@dataclass
class TeacherCfg:
    model: TeacherModel = TeacherModel.NONE
    # The teacher sees the frame upsampled res_scale x (bilinear) and its depth
    # is average-pooled back to the training grid, so edge positions come from
    # a denser prediction. 1 = the training resolution. Part of the manifest.
    res_scale: int = 1
    # frames per teacher forward; bounds the teacher's activation memory
    chunk: int = 4

    def validate(self) -> "TeacherCfg":
        self.model = TeacherModel(self.model)
        if self.res_scale not in (1, 2):
            raise ValueError(f"teacher.res_scale must be 1 or 2, got {self.res_scale}")
        if self.chunk < 1:
            raise ValueError(f"teacher.chunk must be >= 1, got {self.chunk}")
        return self


class DA3Teacher:
    """Depth Anything 3 on each frame separately (N=1 views: DA3 is a
    multi-view model and would otherwise attend across the clip), frozen,
    under autocast, outputs detached. annotate() attaches view["teacher_depth"]
    [B,H,W] float32 at the view's own resolution."""

    def __init__(self, cfg: TeacherCfg, device: torch.device):
        from eval.baselines.arms import load_da3, weights_file

        self.cfg = cfg
        self.device = device
        wrapper = load_da3(str(weights_file("da3_metric")))
        self.net = wrapper.model.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    SKY_THRESHOLD = 0.3  # DA3's own (utils/alignment.compute_sky_mask)

    @torch.no_grad()
    def depth(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images: [N,3,H,W] in [0,1], H and W multiples of 14 -> (depth
        [N,H,W] in the teacher's own canonical scale, valid [N,H,W] bool:
        finite, positive and not sky by DA3's sky head). The forward runs
        under a forked RNG: DA3's sky post-process draws from the CUDA
        generator, which would otherwise shift the sparse-mask draws of a
        seeded run relative to a run without a teacher."""
        N, _, H, W = images.shape
        rs = self.cfg.res_scale
        x = images.to(self.device, dtype=torch.float32)
        if rs != 1:
            x = F.interpolate(x, scale_factor=rs, mode="bilinear", align_corners=False)
        x = (x.unsqueeze(1) - self.mean) / self.std  # [N,1,3,H',W']
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        devices = [self.device] if self.device.type == "cuda" else []
        depths, skies = [], []
        with torch.random.fork_rng(devices=devices):
            for i in range(0, N, self.cfg.chunk):
                with torch.autocast(device_type=self.device.type, dtype=dtype):
                    out = self.net(x[i : i + self.cfg.chunk])
                depths.append(out["depth"].reshape(-1, H * rs, W * rs).float())
                sky = out.get("sky")
                skies.append(
                    sky.reshape(-1, H * rs, W * rs).float()
                    if sky is not None
                    else torch.zeros_like(depths[-1])
                )
        d = torch.cat(depths, dim=0)
        sky = torch.cat(skies, dim=0)
        if rs != 1:
            d = F.avg_pool2d(d.unsqueeze(1), kernel_size=rs).squeeze(1)
            sky = F.avg_pool2d(sky.unsqueeze(1), kernel_size=rs).squeeze(1)
        valid = torch.isfinite(d) & (d > 0) & (sky < self.SKY_THRESHOLD)
        return d.detach(), valid

    def annotate(self, views: list[dict]) -> None:
        """Attach teacher_depth and teacher_valid to every view that has an
        image (in place), once per batch: a view already carrying them is
        left alone."""
        todo = [v for v in views if "img" in v and "teacher_depth" not in v]
        if not todo:
            return
        B = todo[0]["img"].shape[0]
        frames = torch.cat([v["img"] for v in todo], dim=0)  # [S*B,3,H,W]
        depth, valid = self.depth(frames)
        for i, v in enumerate(todo):
            v["teacher_depth"] = depth[i * B : (i + 1) * B]
            v["teacher_valid"] = valid[i * B : (i + 1) * B]


def build_teacher(cfg: TeacherCfg, device: torch.device):
    """The configured teacher, or None."""
    match cfg.model:
        case TeacherModel.NONE:
            return None
        case TeacherModel.DA3:
            return DA3Teacher(cfg, device)
        case _:
            raise ValueError(f"unknown teacher model {cfg.model!r}")
