"""SPOT loader and held-out split (CPU).

python -m unittest tests/test_spot_benchmark.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import spot_benchmark as SB  # noqa: E402


def _write_seq(root: Path, n: int = 3) -> Path:
    seq = root / "seq_0"
    (seq / "color").mkdir(parents=True)
    (seq / "depth").mkdir()
    for i in range(n):
        rgb = np.zeros((SB.RAW_H, SB.RAW_W, 3), np.uint8)
        rgb[:10, :20, 0] = 255  # red block top-left of the RAW (sideways) frame
        Image.fromarray(rgb).save(seq / "color" / f"{i}.png")
        d = np.zeros((SB.RAW_H, SB.RAW_W), np.float32)
        d[:, : SB.RAW_W // 2] = 2.0 + i  # left half of the raw frame measured
        with open(seq / "depth" / str(i), "wb") as f:
            np.array([SB.RAW_W * SB.RAW_H], np.int32).tofile(f)
            d.tofile(f)
    return seq


class SpotTest(unittest.TestCase):
    def test_read_rotate_and_align_both_framings(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = SB.window(_write_seq(Path(tmp)), 0, 3, 1)
            self.assertEqual(SB.model_size("portrait", 1036), (784, 1036))
            for framing, (W, H) in SB.FRAMINGS.items():
                views, sensor = SB.load_views(seq, torch.device("cpu"), framing)
                self.assertEqual(sensor.shape, (3, H, W))
                self.assertEqual(tuple(views[0]["img"].shape), (1, 3, H, W))
                # raw left half (measured) becomes the TOP after a clockwise
                # rotation: top rows valid, bottom rows not
                self.assertTrue((sensor[0, :20] > 0).all())
                self.assertFalse((sensor[0, -20:] > 0).any())
                self.assertAlmostEqual(
                    float(sensor[2][sensor[2] > 0].mean()), 4.0, places=5
                )
            # portrait keeps the whole frame (half of it measured); the
            # landscape crop keeps only the top 4:3 window, mostly measured
            _, full = SB.load_views(seq, torch.device("cpu"), "portrait")
            _, crop = SB.load_views(seq, torch.device("cpu"), "landscape_crop")
            self.assertAlmostEqual(float((full[0] > 0).mean()), 0.5, delta=0.01)
            self.assertGreater(float((crop[0] > 0).mean()), 0.6)

    def test_split_is_disjoint_seeded_and_fed_to_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = SB.window(_write_seq(Path(tmp)), 0, 3, 1)
            views, sensor = SB.load_views(seq, torch.device("cpu"))
            fed, held = SB.split_sensor_depth(views, sensor, 0.1, seed=7)
            valid = sensor > 0
            self.assertFalse((fed & held).any())
            self.assertTrue(((fed | held) == valid).all())
            self.assertAlmostEqual(held.sum() / valid.sum(), 0.1, delta=0.05)
            # held out in whole 14px cells: no held pixel has a fed pixel in its cell
            H, W = sensor.shape[1:]
            for y in range(0, H, 14):
                for x in range(0, W, 14):
                    c_h, c_f = (
                        held[0, y : y + 14, x : x + 14],
                        fed[0, y : y + 14, x : x + 14],
                    )
                    self.assertFalse(c_h.any() and c_f.any())
            self.assertTrue(
                torch.equal(views[0]["sparse_depth_mask"][0], torch.from_numpy(fed[0]))
            )
            fed2, held2 = SB.split_sensor_depth(views, sensor, 0.1, seed=7)
            self.assertTrue((held == held2).all())

    def test_spot_gif_frames_and_layout(self):
        from eval.spot_gif import write_spot_gif

        S, H, W = 3, 40, 30
        rgb = np.zeros((S, H, W, 3), np.uint8)
        sensor = np.zeros((S, H, W), np.float32)
        sensor[:, : H // 2] = 2.0  # top half measured
        pred = np.full((S, H, W), 2.2, np.float32)
        conf = np.ones((S, H, W), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_spot_gif(
                Path(tmp) / "g.gif", rgb, sensor, pred, conf, "t", scale=1.0
            )
            im = Image.open(out)
            self.assertEqual(im.n_frames, S)
            self.assertGreaterEqual(im.size[0], 5 * W)  # five columns
            with self.assertRaises(ValueError):
                write_spot_gif(Path(tmp) / "bad.gif", rgb, sensor[:2], pred, conf, "t")

    def test_bad_header_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "0"
            with open(p, "wb") as f:
                np.array([5], np.int32).tofile(f)
            with self.assertRaises(ValueError):
                SB.read_spot_depth(p)


if __name__ == "__main__":
    unittest.main()
