"""Online Video Depth Anything's own evaluation, for reproducing its tables.

The protocol and the frame pairing are the authors' reference code, vendored
verbatim in third_party/ovda/evaluation/ (README there; sent by the first
author, 2026-10-08): per-sequence affine fit of the inverse-depth prediction
to 1/GT on the first paired frame or on all of them, GT valid in
0 < GT < limit and kept unclipped in the fit, prediction clipped to [0, 80]
m, scoring on GT < 80 m from frame 1 on, pixel-weighted over the dataset.
This module only reads the data their pairing returns and calls their
functions; nothing about the protocol is re-implemented here.

Used by src/bench_baselines.py reproduce only. Once the arm is verified it is
scored under the benchmark's own protocols like every other model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_EVAL = Path(__file__).resolve().parents[2] / "third_party" / "ovda" / "evaluation"
RAW = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw")

# dataset name in the reproduction record -> (their dataset key, pairing
# function, root it expects). KITTI's root is a layout of symlinks onto our
# raw tree (datasets_download/download_kitti_ovda.sh), so their code reads it
# unchanged.
DATASETS: dict[str, tuple[str, str, Path]] = {
    "ovda_sintel": ("sintel", "sintel_pairs", RAW / "sintel"),
    "ovda_bonn": ("bonn", "bonn_pairs", RAW / "bonn_full"),
    "ovda_kitti": ("kitti", "kitti_pairs", RAW / "kitti_ovda"),
}
# record protocol -> their `alignment`
ALIGNMENT = {"ovda_first": "first", "ovda_all": "all"}

_MODULES: dict[str, object] = {}


def _vendored(name: str):
    """Their module, loaded by path: the package is called `evaluation`, a
    name too generic to put on sys.path."""
    if name not in _MODULES:
        path = _EVAL / f"{name}.py"
        if not path.is_file():
            raise FileNotFoundError(
                f"vendored oVDA evaluation code missing: {path} (third_party/ovda/README_VENDORED.md)"
            )
        spec = importlib.util.spec_from_file_location(f"ovda_eval_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def pairs(dataset: str) -> dict[str, list[tuple[Path, Path]]]:
    """{sequence: [(rgb, depth), ...]} from their pairing code, in their order."""
    _, fn, root = DATASETS[dataset]
    if not root.exists():
        raise FileNotFoundError(f"{dataset}: {root} missing")
    return getattr(_vendored("data_sorting"), fn)(root)


def _read_dpt(path: Path) -> np.ndarray:
    """Sintel .dpt in metres (MPI-Sintel SDK format)."""
    with open(path, "rb") as f:
        if np.fromfile(f, np.float32, 1)[0] != 202021.25:
            raise ValueError(f"{path}: not a Sintel .dpt file")
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, int(w) * int(h)).reshape(h, w)


def read_gt(dataset: str, path: Path) -> np.ndarray:
    """GT in metres, float32, as their README states: Sintel .dpt as is,
    KITTI PNG / 256, Bonn PNG / 5000."""
    key = DATASETS[dataset][0]
    if key == "sintel":
        return _read_dpt(path).astype(np.float32)
    raw = np.asarray(Image.open(path), dtype=np.float32)
    return raw / (256.0 if key == "kitti" else 5000.0)


def read_sequence(dataset: str, seq_pairs) -> tuple[np.ndarray, np.ndarray]:
    """[S,H,W,3] uint8 RGB (only paired frames) and [S,H,W] float32 GT."""
    rgb = np.stack([np.asarray(Image.open(r).convert("RGB")) for r, _ in seq_pairs])
    gt = np.stack([read_gt(dataset, d) for _, d in seq_pairs])
    return rgb, gt


def to_gt_grid(pred: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear with align_corners=True, as their historical evaluation."""
    if pred.shape[1:] == tuple(hw):
        return pred.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(pred, dtype=np.float32))[:, None]
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=True)[:, 0].numpy()


def evaluate(
    pred_disp: np.ndarray, gt: np.ndarray, dataset: str, protocol: str
) -> dict:
    """Their per-sequence totals (summed later with `summarize`)."""
    key = DATASETS[dataset][0]
    return _vendored("protocol").evaluate_sequence(
        pred_disp, gt, key, alignment=ALIGNMENT[protocol]
    )


def summarize(totals: list[dict]) -> dict[str, float]:
    """{"abs_rel", "delta1"} over the dataset, pixel-weighted (theirs)."""
    s = _vendored("protocol").summarize(totals)
    return {"abs_rel": s["AbsRel"], "delta1": s["delta1"]}
