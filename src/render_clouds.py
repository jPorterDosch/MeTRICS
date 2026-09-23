#!/usr/bin/env python
"""Re-render the benchmark point-cloud snapshots (bench_clouds/*.npz, written
by bench_eval at the end of training) as GLB scenes -- the same unprojection
and camera frustums as the training-time GLB export, from data that was
saved rather than from a model.

    python render_clouds.py <run_dir>/bench_clouds/*.npz
    python render_clouds.py a.npz --cameras gt --mask-to-gt --out-dir /tmp/glb

--cameras auto (default) unprojects with the GT cameras when the snapshot
carries them (every video dataset, from the manifest) and falls back to the
model's predicted cameras otherwise; gt / pred force one. The predicted
cameras come from the frozen camera head reading fine-tuned tokens, so they
are what a deployment has, not a trustworthy track. --mask-to-gt keeps only the
pixels with GT depth, so the cloud covers the scored pixels; the default keeps
every finite prediction, which is the completion output in the GT holes.

Each npz holds: rgb [S,H,W,3] uint8, pred_depth / pred_conf / sparse_depth /
gt_depth [S,H,W] float32, sparse_mask / gt_valid [S,H,W] bool, K_pred [S,3,3],
w2c_pred [S,3,4], optional K_gt [S,3,3] / pose_gt [S,4,4] (cam2world), and
the dataset / sequence / density / mode tags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from val_images import clip_predictions
from visual_util import predictions_to_glb


def _c2w_from_w2c(w2c: np.ndarray) -> np.ndarray:
    S = w2c.shape[0]
    m = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    m[:, :3, :4] = w2c
    return np.linalg.inv(m)


def render_npz(
    path: str | Path,
    cameras: str = "auto",
    mask_to_gt: bool = False,
    out_dir: str | Path | None = None,
) -> Path:
    path = Path(path)
    with np.load(path) as d:
        rgb = d["rgb"].astype(np.float32) / 255.0  # [S,H,W,3]
        depth = d["pred_depth"]
        valid = d["gt_valid"]
        has_gt = "K_gt" in d and "pose_gt" in d
        if cameras == "auto":
            cameras = "gt" if has_gt else "pred"
        if cameras == "pred":
            K = d["K_pred"]
            pose = _c2w_from_w2c(d["w2c_pred"])
        elif cameras == "gt":
            if not has_gt:
                raise ValueError(
                    f"{path.name} carries no GT cameras; use --cameras pred"
                )
            K = d["K_gt"]
            pose = d["pose_gt"]
        else:
            raise ValueError(f"cameras must be 'auto', 'pred' or 'gt', got {cameras!r}")
    predictions = clip_predictions(
        torch.from_numpy(rgb).permute(0, 3, 1, 2),
        torch.from_numpy(depth),
        torch.from_numpy(valid),
        torch.from_numpy(K.astype(np.float32)),
        torch.from_numpy(pose.astype(np.float32)),
        mask_to_gt=mask_to_gt,
    )
    scene = predictions_to_glb(
        predictions,
        conf_thres=0.0,
        show_cam=True,
        prediction_mode="Depthmap and Camera",
    )
    out_dir = Path(out_dir) if out_dir is not None else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_predcam" if cameras == "pred" else ""
    out = out_dir / f"{path.stem}{suffix}.glb"
    scene.export(file_obj=str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("npz", nargs="+", help="snapshot files from <run>/bench_clouds/")
    ap.add_argument("--cameras", choices=["auto", "pred", "gt"], default="auto")
    ap.add_argument("--mask-to-gt", action="store_true")
    ap.add_argument("--out-dir", default=None, help="default: next to each npz")
    args = ap.parse_args()
    for p in args.npz:
        out = render_npz(p, args.cameras, args.mask_to_gt, args.out_dir)
        print(f"{p} -> {out}")


if __name__ == "__main__":
    main()
