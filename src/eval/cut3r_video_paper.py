"""The CUT3R / TTT3R long video-depth evaluation, for reproducing the tables
of InfiniteVGGT (arXiv:2601.02281, Table 3) and OVGGT (arXiv:2603.05959,
Table 3), which both score StreamVGGT-family models this way.

Data, as TTT3R's preprocessing builds it (third_party/ttt3r_prep, vendored
verbatim from github.com/Inception3D/TTT3R):
  * Bonn N (long_prepare_bonn.py): for each of the five sequences the
    evaluation lists (balloon2, crowd2, crowd3, person_tracking2,
    synchronous), the RGB files and the depth files, each SORTED ON ITS OWN,
    sliced [30 : 30 + n] with n = min(N, frames available after 30 in rgb,
    depth and groundtruth). RGB and depth are paired by that index, not by
    timestamp, as upstream does (crowd3 holds 854 RGB vs 847 depth frames).
    GT png / 5000.
  * KITTI N (long_prepare_kitti.py): for each of the 13 drives of KITTI's
    annotated-depth `val` split, the first N annotated frames
    (proj_depth/groundtruth/image_02, sorted) and the raw image_02 frame of
    the same name; full frame, no crop. GT png / 256.

Scoring, as their eval_depth.py with `--align scale`, the setting both
repos' run.sh ship (the papers name no alignment): the prediction resized
to the GT with INTER_CUBIC, tools.depth_evaluation per sequence over every
frame at once (scale-only fit in depth space, align_with_scale=True),
max_depth 70 on Bonn and none on KITTI, metrics averaged over sequences
weighted by each sequence's valid pixels. The vendored tools.py is the
repositories' own (byte-identical between InfiniteVGGT and OVGGT).

Used by src/bench_baselines.py reproduce only.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

_TOOLS = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "ovggt"
    / "src"
    / "eval"
    / "video_depth"
    / "tools.py"
)
RAW = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw")
BONN_ROOT = RAW / "bonn_full"
KITTI_ROOT = RAW / "kitti"

BONN_SEQS = ("balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous")
BONN_START = 30  # TTT3R's START_FRAME
BONN_MAX_DEPTH = 70.0  # eval_depth.py's bonn branch
KITTI_MAX_DEPTH = None  # eval_depth.py's kitti branch: GT > 0 only

BONN_LENGTHS = (100, 200, 300, 400, 500)
KITTI_LENGTHS = (100, 300, 500)
DATASETS = tuple(f"ttt3r_bonn_{n}" for n in BONN_LENGTHS) + tuple(
    f"ttt3r_kitti_{n}" for n in KITTI_LENGTHS
)
PROTOCOLS = ("cut3r_scale",)


@dataclass
class Sequence:
    name: str
    images: list[Path]
    depths: list[Path]


def parse(dataset: str) -> tuple[str, int]:
    """('bonn' | 'kitti', N) of a dataset name."""
    m = re.fullmatch(r"ttt3r_(bonn|kitti)_(\d+)", dataset)
    if m is None or dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; known: {DATASETS}")
    return m.group(1), int(m.group(2))


def _sorted_png(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no png under {folder}")
    return files


def bonn_sequences(n_target: int, root: Path = BONN_ROOT) -> list[Sequence]:
    """long_prepare_bonn.py for one TARGET_FRAMES, restricted to the five
    sequences eval_depth.py scores."""
    out = []
    for seq in BONN_SEQS:
        d = root / f"rgbd_bonn_{seq}"
        rgb, depth = _sorted_png(d / "rgb"), _sorted_png(d / "depth")
        n_gt = len(np.loadtxt(d / "groundtruth.txt"))
        n = min(
            len(rgb) - BONN_START, len(depth) - BONN_START, n_gt - BONN_START, n_target
        )
        if n <= 0:
            raise ValueError(f"{d}: no frame after {BONN_START}")
        sl = slice(BONN_START, BONN_START + n)
        out.append(Sequence(seq, rgb[sl], depth[sl]))
    return out


def kitti_sequences(n_target: int, root: Path = KITTI_ROOT) -> list[Sequence]:
    """long_prepare_kitti.py for one TARGET_FRAMES: the first N annotated
    frames of every `val` drive and their raw image_02 frames."""
    out = []
    drives = sorted((root / "val").glob("*/proj_depth/groundtruth/image_02"))
    if not drives:
        raise FileNotFoundError(f"no KITTI val drives under {root / 'val'}")
    for gt_dir in drives:
        drive = gt_dir.parts[-4]
        depths = _sorted_png(gt_dir)[:n_target]
        img_dir = root / drive[:10] / drive / "image_02" / "data"
        images = [img_dir / p.name for p in depths]
        missing = [p for p in images if not p.is_file()]
        if missing:
            # upstream prints and skips a missing image, which would shift
            # every later prediction against its GT; refuse instead
            raise FileNotFoundError(f"{drive}: {len(missing)} raw images missing")
        out.append(Sequence(f"{drive}_02", images, depths))
    return out


def sequences(dataset: str) -> list[Sequence]:
    kind, n = parse(dataset)
    return bonn_sequences(n) if kind == "bonn" else kitti_sequences(n)


def read_frames(seq: Sequence) -> np.ndarray:
    """[S,H,W,3] uint8 RGB as stored (the arm does its own resizing)."""
    return np.stack(
        [np.asarray(exif_transpose(Image.open(p)).convert("RGB")) for p in seq.images]
    )


def read_gt(dataset: str, seq: Sequence) -> np.ndarray:
    """[S,H,W] metres, -1 where the sensor has none (eval_depth.py's
    depth_read for Bonn / KITTI)."""
    kind, _ = parse(dataset)
    scale = 5000.0 if kind == "bonn" else 256.0
    out = []
    for p in seq.depths:
        png = np.asarray(Image.open(p)).astype(np.float64)
        if png.max() <= 255:
            raise ValueError(f"{p}: not a 16-bit depth png")
        d = png / scale
        d[png == 0] = -1.0
        out.append(d)
    return np.stack(out)


_DEPTH_EVALUATION = None


def _depth_evaluation():
    """tools.depth_evaluation, loaded by path (its package is `eval`, which
    would shadow ours)."""
    global _DEPTH_EVALUATION
    if _DEPTH_EVALUATION is None:
        if not _TOOLS.is_file():
            raise FileNotFoundError(f"vendored eval tools missing: {_TOOLS}")
        spec = importlib.util.spec_from_file_location("cut3r_video_tools", _TOOLS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DEPTH_EVALUATION = module.depth_evaluation
    return _DEPTH_EVALUATION


def evaluate(pred: np.ndarray, gt: np.ndarray, dataset: str) -> dict[str, float]:
    """One sequence: eval_depth.py's loop body with --align scale. Returns
    abs_rel, delta1 (their "δ < 1.25"), rmse and valid_pixels."""
    kind, _ = parse(dataset)
    if len(pred) != len(gt):
        raise ValueError(f"{len(pred)} predictions for {len(gt)} GT frames")
    H, W = gt.shape[1:]
    pr = np.stack(
        [
            cv2.resize(p.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            for p in pred
        ]
    )
    results, *_ = _depth_evaluation()(
        pr,
        gt,
        max_depth=BONN_MAX_DEPTH if kind == "bonn" else KITTI_MAX_DEPTH,
        align_with_scale=True,
        use_gpu=True,
    )
    # their keys, renamed to the record's metric names
    return {
        "abs_rel": float(results["Abs Rel"]),
        "delta1": float(results["δ < 1.25"]),
        "rmse": float(results["RMSE"]),
        "valid_pixels": float(results["valid_pixels"]),
    }


def summarize(per_seq: list[dict[str, float]]) -> dict[str, float]:
    """eval_depth.py's average: every metric weighted by valid_pixels."""
    w = [r["valid_pixels"] for r in per_seq]
    return {
        k: float(np.average([r[k] for r in per_seq], weights=w))
        for k in per_seq[0]
        if k != "valid_pixels"
    }
