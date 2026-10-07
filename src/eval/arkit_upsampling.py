"""ARKitScenes depth-upsampling, Validation fold: the split Prompt Depth
Anything reports in its Table 1 (arXiv:2412.14015 Sec. 4.1, "the suggested
training and evaluation protocol in [ARKitScenes]").

Used only to reproduce that table before PromptDA is scored on anything else
(eval/baselines/reproduction.json). PromptDA releases no evaluation code, so
the reading and resizing repeat apple/ARKitScenes depth_upsampling/dataset.py
(commit-independent: `load_image` and `rotate_image` there):

  * laser-scan GT (`highres_depth`, uint16 mm) resized nearest to the
    evaluated size (384x512, 768x1024, or its own 1440x1920);
  * the ARKit depth (`lowres_depth`, 256x192, uint16 mm) kept at its own
    size -- it is the prompt;
  * RGB (`wide`, 1920x1440) handed over as stored: the model does its own
    input sizing, ONCE per frame, and that one prediction is resized to each
    evaluated size. The table's three columns are evaluation resolutions,
    not three inference resolutions (see reproduction.json, promptda runs);
  * everything rotated upright by the video's `sky_direction`.

Scored in metres over GT > 0: L1 and RMSE per image, mean over images.

Layout, as datasets_download/download_arkit_scenes.py leaves it (zips are
never extracted): <root>/metadata.csv and <root>/Validation/<video_id>.zip,
each zip holding <video_id>/{wide,highres_depth,lowres_depth}/<name>.png.
"""

from __future__ import annotations

import csv
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/lustre/isaac24/proj/UTK0516/metrics_data/arkit_scenes/upsampling")

# dataset name in the reproduction record -> evaluated (H, W) before rotation
SIZES: dict[str, tuple[int, int]] = {
    "arkit_upsampling_384": (384, 512),
    "arkit_upsampling_768": (768, 1024),
    "arkit_upsampling_1440": (1440, 1920),
}

_ROTATIONS = {
    "Up": None,
    "Left": cv2.ROTATE_90_CLOCKWISE,
    "Right": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "Down": cv2.ROTATE_180,
}


@dataclass
class Sample:
    name: str  # <video_id>_<timestamp>
    rgb: np.ndarray  # [H,W,3] uint8, upright, as stored (1440x1920)
    prompt: np.ndarray  # [h,w] float32 metres, the ARKit depth, upright
    gt_raw: np.ndarray  # [1440,1920] uint16 mm as stored (not rotated), 0 = no return
    sky: str

    def gt_at(self, hw: tuple[int, int]) -> np.ndarray:
        """GT in metres at evaluated size `hw` (before rotation), upright."""
        H, W = hw
        gt = self.gt_raw
        if gt.shape != (H, W):
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        return _upright(gt, self.sky).astype(np.float32) / 1000.0


def videos(root: Path = ROOT) -> list[Path]:
    """The Validation zips, sorted by video id."""
    zips = sorted((Path(root) / "Validation").glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"no ARKitScenes upsampling zips under {Path(root) / 'Validation'}: run "
            "datasets_download/download_arkit_scenes.py upsampling --split Validation "
            "--video_id_csv datasets_download/raw/upsampling_train_val_splits.csv"
        )
    return zips


def sky_directions(root: Path = ROOT) -> dict[str, str]:
    with open(Path(root) / "metadata.csv", newline="") as f:
        return {row["video_id"]: row["sky_direction"] for row in csv.DictReader(f)}


def _decode(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"{zf.filename}: cannot decode {member}")
    return img


def _upright(img: np.ndarray, sky: str) -> np.ndarray:
    if sky not in _ROTATIONS:
        raise ValueError(f"unknown sky_direction {sky!r}; known: {list(_ROTATIONS)}")
    code = _ROTATIONS[sky]
    return img if code is None else cv2.rotate(img, code)


def samples(zip_path: Path, sky: str) -> Iterator[Sample]:
    """Every frame of one video."""
    with zipfile.ZipFile(zip_path) as zf:
        wide = sorted(n for n in zf.namelist() if "/wide/" in n and n.endswith(".png"))
        if not wide:
            raise ValueError(f"{zip_path}: no wide/*.png")
        for member in wide:
            rgb = cv2.cvtColor(_decode(zf, member), cv2.COLOR_BGR2RGB)
            gt = _decode(zf, member.replace("/wide/", "/highres_depth/"))
            low = _decode(zf, member.replace("/wide/", "/lowres_depth/"))
            yield Sample(
                name=Path(member).stem,
                rgb=_upright(rgb, sky),
                prompt=_upright(low, sky).astype(np.float32) / 1000.0,
                gt_raw=gt,
                sky=sky,
            )


def l1_rmse(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(L1, RMSE) in metres over GT > 0; NaN when the image has no GT."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} != gt {gt.shape}")
    valid = gt > 0
    if not valid.any():
        return float("nan"), float("nan")
    err = pred[valid].astype(np.float64) - gt[valid].astype(np.float64)
    return float(np.abs(err).mean()), float(np.sqrt((err**2).mean()))
