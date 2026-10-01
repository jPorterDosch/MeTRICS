"""SPOT sequences as benchmark input: real sensor sparse depth, no dense GT,
no GT cameras.

Layout (metrics_data/spot_data/<seq>/):
    color/<i>.png   640x480 RGB
    depth/<i>       int32 pixel count (640*480) then HxW float32 metres,
                    0 where the other stereo camera saw nothing (~40-45% valid)
Both are mounted sideways and must be rotated 90 degrees clockwise to be
upright. After rotating, the frame is portrait. Two framings:
  portrait (default)  the whole upright frame at 392x518 -- the full scene,
                      ground and near field included. 392x518 is in the
                      training resolution list (as are the other portrait
                      sizes), and has the same patch count as 518x392.
  landscape_crop      the top 4:3 window at 518x392, as the earlier SPOT
                      evals / GIFs (experiments/eval_all.sh --landscape-crop
                      --crop-anchor top): drops the bottom ~44% of the frame.
Numbers from the two framings are not comparable (different pixels).

The rotate / crop / resize chain mirrors src/visualize_spot.py::_prep (colour
BILINEAR, depth NEAREST through the identical integer boxes, so they stay
pixel-aligned); it is restated here rather than imported because
visualize_spot pulls in finetune_depth, which imports this benchmark.

Scoring has only the sensor to go on, so a fixed share of each frame's valid
sensor pixels is held out: the model is fed the rest (the real sensor
pattern, holes and all -- the case PIXEL_FREQ simulates), and is scored on
the held-out pixels. No TAE: without GT cameras there is no reprojection to
score, and the predicted cameras are never a result.

TODO(spot-poses): future SPOT captures should record GT camera poses (and
calibrated intrinsics for the colour camera -- Spot's odometry / body frame
plus the camera extrinsic, or a mocap / SLAM track). With them, TAE (both
definitions) runs here exactly as on the other video datasets: attach
K/pose per frame in SpotFrame, then score_tae_sequence on the window, and
render the snapshots with GT cameras instead of predicted ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RAW_W, RAW_H = 640, 480
FRAMINGS = {"portrait": (392, 518), "landscape_crop": (518, 392)}  # model (W, H)
CROP_ANCHOR = 0.0  # top of the discarded band, as eval_all.sh --crop-anchor top


@dataclass
class SpotFrame:
    image: Path  # the colour file; save_cloud records it as the frame name
    depth: Path


@dataclass
class SpotSequence:
    dataset: str
    name: str  # "<seq>@<start>"
    frames: list[SpotFrame]

    def __len__(self) -> int:
        return len(self.frames)


def read_spot_depth(path: Path) -> np.ndarray:
    """int32 pixel-count header, then RAW_H x RAW_W float32 metres."""
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        if n != RAW_W * RAW_H:
            raise ValueError(f"{path}: header {n}, expected {RAW_W * RAW_H}")
        d = np.fromfile(f, dtype=np.float32, count=n)
    if d.size != n:
        raise ValueError(f"{path}: {d.size} depth values, expected {n}")
    d = d.reshape(RAW_H, RAW_W)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def model_size(framing: str, image_size: int = 518) -> tuple[int, int]:
    """(W, H) the model runs at: the framing's 518-long-side size, scaled so
    the long side is `image_size` (bench.image_size), both sides snapped to
    a multiple of 14 -- SPOT follows --bench.image-size like every dataset."""
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {list(FRAMINGS)}, got {framing!r}")
    w, h = FRAMINGS[framing]
    f = image_size / max(w, h)
    return (max(14, round(w * f / 14) * 14), max(14, round(h * f / 14) * 14))


def _prep(
    img: Image.Image, resample, framing: str, image_size: int = 518
) -> Image.Image:
    """rotate cw -> (landscape_crop: top 4:3 window) -> scale to cover the
    model size -> centre crop. One function for colour and depth: shared
    boxes."""
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {list(FRAMINGS)}, got {framing!r}")
    img = img.transpose(Image.ROTATE_270)  # 90 degrees clockwise
    if framing == "landscape_crop":
        a = RAW_W / RAW_H
        cw, ch = (
            (round(img.height * a), img.height)
            if img.width / img.height >= a
            else (img.width, round(img.width / a))
        )
        left = (img.width - cw) // 2
        top = min(max(round(CROP_ANCHOR * (img.height - ch)), 0), img.height - ch)
        img = img.crop((left, top, left + cw, top + ch))
    tw, th = model_size(framing, image_size)
    s = max(tw / img.width, th / img.height)
    nw, nh = round(img.width * s), round(img.height * s)
    if nw < tw or nh < th:
        raise ValueError(f"resize {nw}x{nh} does not cover {tw}x{th}")
    img = img.resize((nw, nh), resample)
    lx, ty = (img.width - tw) // 2, (img.height - th) // 2
    return img.crop((lx, ty, lx + tw, ty + th))


def sequence_length(seq_dir: Path) -> int:
    n = len([p for p in (seq_dir / "color").iterdir() if p.suffix == ".png"])
    if n == 0:
        raise FileNotFoundError(f"no colour frames under {seq_dir / 'color'}")
    return n


def window(seq_dir: Path, start: int, n_frames: int, stride: int) -> SpotSequence:
    """Frames start, start+stride, ... (n_frames of them), clipped to the
    sequence; the name records seq and start so windows stay apart."""
    total = sequence_length(seq_dir)
    idx = list(range(start, total, stride))[:n_frames]
    if len(idx) < 2:
        raise ValueError(f"{seq_dir.name}: window at {start} has {len(idx)} frames")
    frames = [
        SpotFrame(seq_dir / "color" / f"{i}.png", seq_dir / "depth" / str(i))
        for i in idx
    ]
    return SpotSequence("spot", f"{seq_dir.name}@{start}", frames)


def load_views(
    seq: SpotSequence,
    device: torch.device,
    framing: str = "portrait",
    image_size: int = 518,
) -> tuple[list[dict], np.ndarray]:
    """[S]-list of view dicts (img in [0,1]) and the full sensor depth
    [S,H,W] at model resolution. The views carry depthmap / valid_mask = the
    sensor depth (so snapshots and the viewer have something to draw as
    "GT"); the fed sparse depth is attached by split_sensor_depth."""
    views, depths = [], []
    for i, fr in enumerate(seq.frames):
        rgb = _prep(
            Image.open(fr.image).convert("RGB"), Image.BILINEAR, framing, image_size
        )
        img = torch.from_numpy(np.asarray(rgb).copy()).float().permute(2, 0, 1) / 255.0
        d = np.asarray(
            _prep(
                Image.fromarray(read_spot_depth(fr.depth), mode="F"),
                Image.NEAREST,
                framing,
            )
        ).copy()
        d[~np.isfinite(d)] = 0.0
        depths.append(d)
        views.append(
            {
                "img": img[None].to(device),
                "depthmap": torch.from_numpy(d)[None].to(device),
                "valid_mask": torch.from_numpy(d > 0)[None].to(device),
                "true_shape": torch.tensor([[d.shape[0], d.shape[1]]], device=device),
                "idx": i,
                "instance": str(fr.image),
                "dataset": "spot",
            }
        )
    return views, np.stack(depths, axis=0)


def split_sensor_depth(
    views: list[dict], sensor: np.ndarray, holdout: float, seed: int, block: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out `holdout` of each frame's sensor-covered `block` x `block`
    cells (seeded; one model patch by default) and feed the model the rest
    as sparse depth. Whole cells, not single pixels: an i.i.d. per-pixel
    holdout leaves fed neighbours inside the same patch as every held-out
    pixel, so the score would measure copying, not completion. Returns
    (fed_mask, held_mask), both [S,H,W] bool."""
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    rng = np.random.default_rng(seed)
    valid = sensor > 0
    S, H, W = sensor.shape
    gh, gw = -(-H // block), -(-W // block)
    held = np.zeros_like(valid)
    for i in range(S):
        pad = np.zeros((gh * block, gw * block), bool)
        pad[:H, :W] = valid[i]
        covered = pad.reshape(gh, block, gw, block).any(axis=(1, 3))
        pick = covered & (rng.random((gh, gw)) < holdout)
        held[i] = np.repeat(np.repeat(pick, block, 0), block, 1)[:H, :W] & valid[i]
    fed = valid & ~held
    for v, d, m in zip(views, sensor, fed):
        device = v["img"].device
        v["sparse_depth"] = torch.from_numpy(np.where(m, d, 0.0).astype(np.float32))[
            None
        ].to(device)
        v["sparse_depth_mask"] = torch.from_numpy(m)[None].to(device)
    return fed, held
