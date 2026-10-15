"""Marigold's single-image evaluation, for reproducing tables that follow it
(Depth Any Video's Table 2: "Following prior methods (Ke et al., 2024; Fu et
al., 2024), we evaluate affine-invariant depth predictions by optimizing for
scale and shift").

Data and frame lists are Marigold's own, kept apart from the benchmark trees:
the evaluation tars from Marigold's README (share.phys.ethz.ch/~pf/bingkedata/
marigold/evaluation_dataset/) extracted under RAW, and the split lists
vendored verbatim in third_party/marigold_eval/data_split. The ScanNet set is
the fixed 800-frame sample Marigold ships (list + tar), not a draw made here,
so the frames are theirs by construction.

Protocol, as Marigold's script/depth/eval.py with --alignment
least_square_disparity (third_party/marigold_eval, README there): GT and
valid mask as their dataset classes build them (KITTI: kb crop of image and
GT to 352x1216, 1e-5 < GT < 80, Eigen evaluation mask; ScanNet: GT png /
1000, 1e-3 < GT < 10, no crop; NYUv2: the 654 test frames, RAW depth png /
1000, 1e-3 < GT < 10, Eigen crop [45:471, 41:601]); their align_depth_least_square fits the
prediction to GT disparity over valid & GT > 0 & prediction > 0; the aligned
disparity is clipped at 1e-3, inverted, clipped to [min, max] and to > 1e-6;
their abs_relative_difference and delta1_acc per image, mean over images.

Used by src/bench_baselines.py reproduce only.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image

_MARIGOLD = Path(__file__).resolve().parents[2] / "third_party" / "marigold_eval"
RAW = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw/marigold_eval")


@dataclass(frozen=True)
class _Spec:
    tree: str  # extracted tar under RAW
    split: str  # list under third_party/marigold_eval/data_split
    min_depth: float
    max_depth: float
    depth_scale: float  # png value -> metres
    kb_crop: bool
    eval_mask: str | None  # "kitti_eigen", "nyu_eigen" or None


DATASETS: dict[str, _Spec] = {
    "marigold_kitti": _Spec(
        "kitti_eigen_split_test",
        "kitti_depth/eigen_test_files_with_gt.txt",
        1e-5,
        80.0,
        256.0,
        kb_crop=True,
        eval_mask="kitti_eigen",
    ),
    "marigold_scannet": _Spec(
        "scannet_val_sampled_800_1",
        "scannet_depth/scannet_val_sampled_list_800_1.txt",
        1e-3,
        10.0,
        1000.0,
        kb_crop=False,
        eval_mask=None,
    ),
    "marigold_nyu": _Spec(
        "nyu_labeled_extracted",
        "nyu_depth/labeled/filename_list_test.txt",
        1e-3,
        10.0,
        1000.0,
        kb_crop=False,
        eval_mask="nyu_eigen",
    ),
}
PROTOCOLS = ("marigold_disparity",)


@dataclass
class Sample:
    name: str
    rgb: np.ndarray  # [H,W,3] uint8, as the model sees it
    gt: np.ndarray  # [H,W] float32 metres
    valid: np.ndarray  # [H,W] bool, Marigold's valid_mask_raw


_MODULES: dict[str, object] = {}


def _vendored(rel: str):
    """Marigold's module, loaded by path (its package is called `src`)."""
    if rel not in _MODULES:
        path = _MARIGOLD / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"vendored Marigold code missing: {path} "
                "(third_party/marigold_eval/README_VENDORED.md)"
            )
        spec = importlib.util.spec_from_file_location(
            "marigold_" + path.stem, str(path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[rel] = module
    return _MODULES[rel]


def _kb_crop(x: np.ndarray) -> np.ndarray:
    """KITTIDepthDataset.kitti_benchmark_crop: bottom 352 rows, centre 1216."""
    h, w = x.shape[:2]
    top, left = h - 352, int((w - 1216) / 2)
    return x[top : top + 352, left : left + 1216]


def _eigen_mask(shape: tuple[int, int]) -> np.ndarray:
    """KITTIDepthDataset._get_valid_mask, valid_mask_crop='eigen'."""
    h, w = shape
    m = np.zeros(shape, bool)
    m[
        int(0.3324324 * h) : int(0.91351351 * h),
        int(0.0359477 * w) : int(0.96405229 * w),
    ] = True
    return m


def _nyu_eigen_mask(shape: tuple[int, int]) -> np.ndarray:
    """NYUDepthDataset._get_valid_mask, eigen_valid_mask=True."""
    m = np.zeros(shape, bool)
    m[45:471, 41:601] = True
    return m


def samples(dataset: str) -> Iterator[Sample]:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    root = RAW / spec.tree
    if not root.is_dir():
        raise FileNotFoundError(
            f"{dataset}: {root} missing (Marigold evaluation tar, see module docstring)"
        )
    for line in (_MARIGOLD / "data_split" / spec.split).read_text().splitlines():
        cols = line.split()
        if not cols or cols[1] == "None":  # KITTI frames without GT: skipped, as theirs
            continue
        rgb = np.asarray(Image.open(root / cols[0]).convert("RGB"))
        gt = np.asarray(Image.open(root / cols[1]), dtype=np.float32) / spec.depth_scale
        if spec.kb_crop:
            rgb, gt = _kb_crop(rgb), _kb_crop(gt)
        valid = (gt > spec.min_depth) & (gt < spec.max_depth)
        if spec.eval_mask == "kitti_eigen":
            valid &= _eigen_mask(gt.shape)
        elif spec.eval_mask == "nyu_eigen":
            valid &= _nyu_eigen_mask(gt.shape)
        yield Sample(cols[0], np.ascontiguousarray(rgb), gt, valid)


def evaluate(pred_disparity: np.ndarray, s: Sample, dataset: str) -> dict:
    """Marigold's eval.py loop body for one image, least_square_disparity."""
    if pred_disparity.shape != s.gt.shape:
        raise ValueError(
            f"{s.name}: prediction {pred_disparity.shape} vs GT {s.gt.shape}"
        )
    spec = DATASETS[dataset]
    align = _vendored("src/util/alignment.py")
    metric = _vendored("src/util/metric.py")
    gt_disparity, gt_non_neg = align.depth2disparity(depth=s.gt, return_mask=True)
    valid = s.valid & gt_non_neg & (pred_disparity > 0)
    disparity, _, _ = align.align_depth_least_square(
        gt_arr=gt_disparity,
        pred_arr=pred_disparity.astype(np.float32),
        valid_mask_arr=valid,
        return_scale_shift=True,
        max_resolution=None,
    )
    depth = align.disparity2depth(np.clip(disparity, a_min=1e-3, a_max=None))
    depth = np.clip(depth, a_min=spec.min_depth, a_max=spec.max_depth)
    depth = np.clip(depth, a_min=1e-6, a_max=None)
    pred_t = torch.from_numpy(depth)
    gt_t = torch.from_numpy(s.gt)
    valid_t = torch.from_numpy(s.valid)
    return {
        "abs_rel": float(metric.abs_relative_difference(pred_t, gt_t, valid_t)),
        "delta1": float(metric.delta1_acc(pred_t, gt_t, valid_t)),
    }
