"""Side-by-side GIF of one SPOT window, written by the benchmark's SPOT pass.

Five columns per frame, all at the model's framing:
    RGB | sensor depth | prediction | rel. error vs sensor | confidence
Sensor and prediction share one depth colour range (robust 2-98% of the
window's sensor depth), so a scale error shows as a colour shift rather than
being normalised away; pixels without a sensor reading are grey in the sensor
and error columns. The error column is |pred - sensor| / sensor on every
sensor pixel -- the fed ones included, which a conditioned model should get
nearly right, so a bright fed pixel is itself a finding. Confidence is the
model's own head (expp1, floor 1) on a robust per-window range.

PIL + matplotlib colormaps only; composition reuses heatmaps_to_gif's
captioned grid so these GIFs look like the rest of the repo's.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from heatmaps_to_gif import compose_grid, write_gif

GREY = np.array([128, 128, 128], np.uint8)


def _colorize(
    x: np.ndarray, ok: np.ndarray, lo: float, hi: float, cmap: str
) -> np.ndarray:
    """[H,W] float -> [H,W,3] uint8 on [lo, hi], grey where not ok."""
    t = np.clip((np.nan_to_num(x) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (matplotlib.colormaps[cmap](t)[..., :3] * 255.0).astype(np.uint8)
    rgb[~ok] = GREY
    return rgb


def write_spot_gif(
    out: Path,
    rgb: np.ndarray,
    sensor: np.ndarray,
    pred: np.ndarray,
    conf: np.ndarray,
    title: str,
    rel_vmax: float = 0.3,
    fps: float = 10.0,
    scale: float = 0.75,
) -> Path:
    """rgb [S,H,W,3] uint8; sensor / pred / conf [S,H,W] float. Returns out."""
    S = rgb.shape[0]
    for name, a in (("sensor", sensor), ("pred", pred), ("conf", conf)):
        if a.shape != rgb.shape[:3]:
            raise ValueError(f"{name} shape {a.shape} != rgb frames {rgb.shape[:3]}")
    has = sensor > 0
    if has.any():
        d_lo, d_hi = (float(v) for v in np.percentile(sensor[has], [2, 98]))
    else:
        d_lo, d_hi = (float(v) for v in np.percentile(pred[np.isfinite(pred)], [2, 98]))
    finite_c = np.isfinite(conf)
    c_lo, c_hi = (
        (float(v) for v in np.percentile(conf[finite_c], [2, 98]))
        if finite_c.any()
        else (1.0, 2.0)
    )
    rel = np.abs(pred - sensor) / np.where(has, sensor, 1.0)
    labels = [
        "",  # per frame: title + frame counter
        f"sensor depth ({d_lo:.1f}-{d_hi:.1f} m)",
        "prediction (same range)",
        f"|pred-sensor|/sensor (0-{rel_vmax:g})",
        f"confidence ({c_lo:.1f}-{c_hi:.1f})",
    ]
    frames = []
    for i in range(S):
        cols = [
            rgb[i],
            _colorize(sensor[i], has[i], d_lo, d_hi, "turbo"),
            _colorize(pred[i], np.isfinite(pred[i]), d_lo, d_hi, "turbo"),
            _colorize(rel[i], has[i] & np.isfinite(pred[i]), 0.0, rel_vmax, "inferno"),
            _colorize(conf[i], finite_c[i], c_lo, c_hi, "viridis"),
        ]
        panels = [Image.fromarray(c) for c in cols]
        if scale != 1.0:
            size = (round(panels[0].width * scale), round(panels[0].height * scale))
            panels = [p.resize(size, Image.BILINEAR) for p in panels]
        frames.append(
            compose_grid(panels, [f"{title}  frame {i + 1}/{S}", *labels[1:]], rows=1)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(frames, out, fps)
    return out
