"""Single-image metric depth on NYUv2 and KITTI, for reproducing Depth Anything
3's Table 11 ("DA3-metric", delta1 / AbsRel).

The DA3 paper names no evaluation protocol and the DA3 repository ships none
for metric depth. The protocol here (`da3_table11`) is built from what the
paper does state, Sec. 4.4: the metric model is trained at base resolution 504
with aspect ratios from 1:1 to 1:2.

  * Frames wider (or taller) than 1:2 are centre-cropped to 1:2, the GT with
    them (KITTI's ~1:3.3 frame becomes 375x750; NYU's 4:3 is untouched). The
    arm then resizes the short side to 504 (DA3's lower_bound_resize).
  * Scoring is Depth Anything V2's metric-depth evaluation loop
    (third_party/depth_anything_v2_metric, README there) with its outdoor
    depth range: prediction bilinear (align_corners) to the GT grid, valid =
    GT > 0 and 0.001 <= GT <= 80 m, no clip, images with fewer than 10 valid
    pixels skipped, DA2's `eval_depth` per image, mean over images.

The probe that found it (2026-10-10/11, 25 jobs, every variable one at a
time) is summarised in the da3_metric entry of reproduction.json.

Data:
  nyu_official_test  the 654 official test frames (splits.mat testNdxs) of
                     nyu_depth_v2_labeled.mat: lossless RGB, rawDepths GT
                     in metres, the Kinect intrinsics.
  kitti_eigen_652    DA2's splits/kitti/val.txt (the 652 Eigen test frames
                     with annotated GT; same frames and order as ZoeDepth's
                     list), image_02 PNG, GT / 256, focal from each date's
                     calib_cam_to_cam.txt P_rect_02.

Used by src/bench_baselines.py reproduce only.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_DA2 = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "depth_anything_v2_metric"
    / "metric_depth"
)
RAW = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw")
NYU_MAT = RAW / "nyu" / "nyu_depth_v2_labeled.mat"
NYU_SPLITS = RAW / "nyu" / "splits.mat"
KITTI = RAW / "kitti"
# DA2's val.txt carries the authors' absolute paths
_DA2_PREFIX = "/mnt/bn/liheyang/Kitti/"

# NYUv2 Kinect RGB intrinsics (fx, fy) of the 640x480 frame, as in
# ZoeDepth's nyudepthv2_test_files_with_gt.txt and the NYU toolbox
NYU_FOCAL = (518.8579, 519.4696)

DATASETS = ("nyu_official_test", "kitti_eigen_652")
# record protocol -> (min depth, max depth) of DA2's eval loop; 80 m is the
# --max-depth of DA2's outdoor (vkitti) metric model, and every NYU GT is below it
PROTOCOLS = {"da3_table11": (0.001, 80.0)}
# widest aspect ratio (long side / short side) the metric model is trained at
MAX_ASPECT = 2.0


@dataclass
class Sample:
    name: str
    rgb: np.ndarray  # [H,W,3] uint8
    gt: np.ndarray  # [H,W] float32 metres, 0 = invalid
    focal: tuple[float, float]  # (fx, fy) in pixels of `rgb`


_METRIC = None


def _eval_depth():
    """DA2's eval_depth, loaded by path: its package is called `util`."""
    global _METRIC
    if _METRIC is None:
        path = _DA2 / "util" / "metric.py"
        if not path.is_file():
            raise FileNotFoundError(
                f"vendored DA2 metric missing: {path} "
                "(third_party/depth_anything_v2_metric/README_VENDORED.md)"
            )
        spec = importlib.util.spec_from_file_location("da2_metric", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _METRIC = module
    return _METRIC.eval_depth


def _nyu() -> Iterator[Sample]:
    import h5py
    import scipy.io

    if not NYU_MAT.is_file():
        raise FileNotFoundError(f"nyu_official_test: {NYU_MAT} missing")
    test = sorted(int(i) for i in scipy.io.loadmat(NYU_SPLITS)["testNdxs"].ravel())
    with h5py.File(NYU_MAT, "r") as f:
        images, depths = f["images"], f["rawDepths"]
        for idx in test:  # 1-based, MATLAB
            yield Sample(
                name=f"{idx:04d}",
                rgb=np.ascontiguousarray(np.transpose(images[idx - 1], (2, 1, 0))),
                gt=np.ascontiguousarray(depths[idx - 1].T, dtype=np.float32),
                focal=NYU_FOCAL,
            )


def _kitti_focal(date: str, cache: dict) -> float:
    if date not in cache:
        calib = KITTI / date / "calib_cam_to_cam.txt"
        for line in calib.read_text().splitlines():
            if line.startswith("P_rect_02:"):
                cache[date] = float(line.split()[1])
                break
        else:
            raise ValueError(f"{calib}: no P_rect_02")
    return cache[date]


def _kitti() -> Iterator[Sample]:
    split = _DA2 / "dataset" / "splits" / "kitti" / "val.txt"
    focals: dict[str, float] = {}
    for line in split.read_text().splitlines():
        img, depth = line.split()
        img = KITTI / img.replace(_DA2_PREFIX + "raw_data/", "")
        depth = KITTI / depth.replace(_DA2_PREFIX + "data_depth_annotated/", "")
        f = _kitti_focal(img.relative_to(KITTI).parts[0], focals)
        yield Sample(
            name=str(img.relative_to(KITTI)),
            rgb=np.asarray(Image.open(img).convert("RGB")),
            gt=np.asarray(Image.open(depth), dtype=np.float32) / 256.0,
            focal=(f, f),
        )


def crop_to_aspect(s: Sample, max_aspect: float = MAX_ASPECT) -> Sample:
    """Centre crop of the image and its GT to at most `max_aspect` (long side /
    short side). A crop keeps the focal length in pixels."""
    h, w = s.gt.shape
    if s.rgb.shape[:2] != (h, w):
        raise ValueError(f"{s.name}: image {s.rgb.shape[:2]} vs GT {(h, w)}")
    if w > max_aspect * h:
        cw = int(round(max_aspect * h))
        x0 = (w - cw) // 2
        return Sample(s.name, s.rgb[:, x0 : x0 + cw], s.gt[:, x0 : x0 + cw], s.focal)
    if h > max_aspect * w:
        ch = int(round(max_aspect * w))
        y0 = (h - ch) // 2
        return Sample(s.name, s.rgb[y0 : y0 + ch], s.gt[y0 : y0 + ch], s.focal)
    return s


def samples(dataset: str) -> Iterator[Sample]:
    """The dataset's frames, cropped to the trained aspect range."""
    if dataset == "nyu_official_test":
        it = _nyu()
    elif dataset == "kitti_eigen_652":
        it = _kitti()
    else:
        raise KeyError(f"unknown dataset {dataset!r}; known: {DATASETS}")
    return (crop_to_aspect(s) for s in it)


def evaluate(pred: np.ndarray, gt: np.ndarray, protocol: str) -> dict | None:
    """DA2's per-image metrics of a metric-depth prediction (any resolution)
    against `gt`, or None when fewer than 10 pixels are valid (DA2 skips the
    image). Keys as DA2 names them, `d1` renamed `delta1`."""
    min_depth, max_depth = PROTOCOLS[protocol]
    depth = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32))
    p = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[None, None]
    p = F.interpolate(p, depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]
    valid = (depth > 0) & (depth >= min_depth) & (depth <= max_depth)
    if valid.sum() < 10:
        return None
    r = _eval_depth()(p[valid], depth[valid])
    r["delta1"] = r.pop("d1")
    return r
