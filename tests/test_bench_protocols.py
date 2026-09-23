"""Benchmark scoring protocols on synthetic sequences (CPU, no model).

    LD_LIBRARY_PATH=~/.local/lib \\
      ~/.pyenv/versions/3.11.13/envs/metrics/bin/python -m unittest tests/test_bench_protocols.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval import protocols as P  # noqa: E402
from eval import vda_benchmark as VB  # noqa: E402
import bench_eval  # noqa: E402


def _gt(seed=0, S=3, H=12, W=16, holes=0.1):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1.0, 5.0, size=(S, H, W)).astype(np.float32)
    gt[rng.uniform(size=gt.shape) < holes] = 0.0  # sensor holes
    return gt


class PublishedProtocolTest(unittest.TestCase):
    def test_disparity_affine_prediction_scores_perfectly(self):
        gt = _gt()
        # a relative-disparity model: disparity off by scale AND shift
        pred_disp = 3.0 * (1.0 / np.where(gt > 0, gt, 1.0)) + 0.2
        m, aligned = P.published_metrics(pred_disp, gt, max_depth=10.0)
        self.assertLess(m.abs_rel, 1e-5)
        self.assertGreater(m.delta1, 0.999)
        self.assertEqual(m.frames, 3)
        self.assertTrue(np.allclose(aligned[gt > 0], gt[gt > 0], rtol=1e-4))

    def test_depth_model_enters_as_inverse_depth(self):
        gt = _gt()
        pred_depth = np.where(gt > 0, gt, 2.0) * 1.5  # metric depth off by scale
        m, _ = P.published_metrics(P.depth_to_disparity(pred_depth), gt, 10.0)
        self.assertLess(m.abs_rel, 1e-5)

    def test_frames_without_valid_gt_are_dropped(self):
        gt = _gt()
        gt[1] = 0.0
        m, _ = P.published_metrics(P.depth_to_disparity(np.ones_like(gt) * 2), gt, 10.0)
        self.assertEqual(m.frames, 2)

    def test_max_depth_masks_scoring(self):
        gt = _gt(holes=0.0)
        gt[0, 0, 0] = 50.0  # beyond max_depth: excluded, not an outlier
        pred = gt.copy()
        pred[0, 0, 0] = 1.0
        m = P.metric_metrics(pred, gt, max_depth=10.0)
        self.assertLess(m.abs_rel, 1e-6)


class NonFinitePredictionTest(unittest.TestCase):
    def test_nan_pixel_is_scored_as_error_not_crash(self):
        gt = _gt(holes=0.0)
        pred = gt.copy()
        pred[0, 3, 3] = np.nan
        pred[1, 2, 2] = np.inf
        pub, aligned = P.published_metrics(P.depth_to_disparity(pred), gt, 10.0)
        self.assertTrue(np.isfinite(aligned).all())
        self.assertTrue(np.isfinite(pub.abs_rel))
        self.assertGreater(pub.abs_rel, 0.0)  # the bad pixels count against it
        self.assertLess(pub.abs_rel, 0.05)  # but do not poison the fit
        met = P.metric_metrics(pred, gt, 10.0)
        self.assertTrue(np.isfinite(met.abs_rel))
        prompt = np.zeros_like(gt, dtype=bool)
        prompt[:, ::3, ::3] = True
        prompt[0, 3, 3] = True  # a NaN pixel inside the prompt is dropped from the fit
        spa = P.sparse_aligned_metrics(pred, np.where(prompt, gt, 0), prompt, pred, prompt, gt, 10.0)
        self.assertTrue(np.isfinite(spa.abs_rel))
        self.assertLess(spa.abs_rel, 0.05)


class MetricProtocolTest(unittest.TestCase):
    def test_scale_error_is_reported_not_aligned_away(self):
        gt = _gt(holes=0.0)
        m = P.metric_metrics(gt * 1.1, gt, 10.0)
        self.assertAlmostEqual(m.abs_rel, 0.1, places=5)
        self.assertAlmostEqual(m.delta1, 1.0, places=6)
        m2 = P.metric_metrics(gt * 1.3, gt, 10.0)
        self.assertAlmostEqual(m2.delta1, 0.0, places=6)


class SparseAlignedProtocolTest(unittest.TestCase):
    def _case(self, s=0.5, t=-0.3, seed=1):
        gt = _gt(seed=seed, holes=0.0)
        S, H, W = gt.shape
        rng = np.random.default_rng(seed)
        prompt_mask = rng.uniform(size=gt.shape) < 0.1
        prompt_depth = np.where(prompt_mask, gt, 0.0)
        pred = (gt - t) / s  # so s*pred + t == gt exactly
        return gt, pred, prompt_depth, prompt_mask

    def test_prompt_fit_recovers_the_affine(self):
        gt, pred, pd, pm = self._case()
        m = P.sparse_aligned_metrics(pred, pd, pm, pred, pm, gt, 10.0)
        self.assertLess(m.abs_rel, 1e-6)
        self.assertEqual(m.frames, gt.shape[0])

    def test_prompt_pixels_are_held_out(self):
        gt, pred, pd, pm = self._case()
        # corrupt the prediction ONLY on the prompt pixels: a scored prompt
        # would show it, a held-out one cannot
        bad = pred.copy()
        bad[pm] = 100.0
        # the fit itself uses the prompt pixels, so keep those honest by
        # passing the clean native prediction for the fit and the corrupted
        # one at GT resolution
        m = P.sparse_aligned_metrics(pred, pd, pm, bad, pm, gt, 10.0)
        self.assertLess(m.abs_rel, 1e-6)

    def test_frame_without_prompt_is_dropped(self):
        gt, pred, pd, pm = self._case()
        pm[2] = False
        pd[2] = 0.0
        m = P.sparse_aligned_metrics(pred, pd, pm, pred, pm, gt, 10.0)
        self.assertEqual(m.frames, 2)

    def test_native_and_gt_resolution_may_differ(self):
        gt, pred, pd, pm = self._case()
        S, H, W = gt.shape
        # native at half resolution: fit there, score at full
        pred_n = np.stack([cv2.resize(p, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) for p in pred])
        pd_n = np.stack([cv2.resize(p, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) for p in pd])
        pm_n = np.stack([cv2.resize(p.astype(np.float32), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) for p in pm]) > 0.5
        pm_gt = VB.resize_to_gt(pm_n.astype(np.float32), (H, W), nearest=True) > 0.5
        m = P.sparse_aligned_metrics(pred_n, pd_n, pm_n, pred, pm_gt, gt, 10.0)
        self.assertLess(m.abs_rel, 1e-6)


class AffineFitTest(unittest.TestCase):
    def test_constant_prediction_falls_back_to_shift(self):
        s, t = P.affine_fit(np.full(10, 2.0), np.full(10, 5.0))
        self.assertEqual(s, 1.0)
        self.assertAlmostEqual(t, 3.0)

    def test_too_few_samples_raise(self):
        with self.assertRaises(ValueError):
            P.affine_fit(np.array([1.0]), np.array([2.0]))


class TAETest(unittest.TestCase):
    def _static(self, S=3, H=16, W=20):
        depth = np.full((S, H, W), 2.0, dtype=np.float32)
        K = np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]])
        poses = [np.eye(4) for _ in range(S)]
        return depth, [K] * S, poses

    def test_static_camera_identical_depth_is_zero_error_both_definitions(self):
        depth, Ks, poses = self._static()
        self.assertAlmostEqual(P.tae_vda(depth, Ks, poses), 0.0, places=6)
        a, sq = P.tae_ours(depth, np.ones_like(depth, dtype=bool), Ks, poses)
        self.assertAlmostEqual(a, 0.0, places=6)
        self.assertAlmostEqual(sq, 0.0, places=6)

    def test_flicker_is_measured_and_vda_is_percent(self):
        depth, Ks, poses = self._static()
        depth[1] *= 1.1  # a 10% pop on the middle frame
        vda = P.tae_vda(depth, Ks, poses)
        ours, _ = P.tae_ours(depth, np.ones_like(depth, dtype=bool), Ks, poses)
        # both pairs see a ~10% relative jump; VDA reports it x100
        self.assertGreater(vda, 8.0)
        self.assertLess(vda, 12.0)
        self.assertGreater(ours, 0.08)
        self.assertLess(ours, 0.12)

    def test_single_frame_has_no_tae(self):
        depth, Ks, poses = self._static(S=1)
        self.assertTrue(np.isnan(P.tae_vda(depth, Ks, poses)))

    def test_length_mismatch_raises(self):
        depth, Ks, poses = self._static()
        with self.assertRaises(ValueError):
            P.tae_vda(depth, Ks[:-1], poses)


class ManifestTest(unittest.TestCase):
    def test_manifest_gt_factor_crop_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = VB.BenchSpec("bonn", "bonn/bonn_video.json", 10.0, (1, 5, 2, 8), 2, True)
            seq_dir = root / "bonn" / "seq_a"
            (seq_dir / "rgb").mkdir(parents=True)
            (seq_dir / "depth").mkdir(parents=True)
            frames = []
            for i in range(3):
                rgb = np.zeros((6, 10, 3), np.uint8)
                cv2.imwrite(str(seq_dir / "rgb" / f"{i}.png"), rgb)
                depth = np.full((6, 10), 5000 * (i + 1), np.uint16)  # (i+1) metres at factor 5000
                depth[0, 0] = 0
                cv2.imwrite(str(seq_dir / "depth" / f"{i}.png"), depth)
                frames.append({"image": f"seq_a/rgb/{i}.png", "gt_depth": f"seq_a/depth/{i}.png", "factor": 5000.0})
            with open(root / "bonn" / "bonn_video.json", "w") as f:
                json.dump({"bonn": [{"seq_a": frames}]}, f)
            seqs = VB.load_manifest(root, spec)
            self.assertEqual(len(seqs), 1)
            self.assertEqual(len(seqs[0]), 2)  # max_len truncation
            gt = VB.gt_stack(seqs[0], spec)
            self.assertEqual(gt.shape, (2, 4, 6))  # crop [1:5, 2:8]
            self.assertAlmostEqual(float(gt[1].mean()), 2.0, places=6)
            self.assertEqual(VB.bench_root_ok(root, ("bonn",)), [])
            self.assertEqual(VB.bench_root_ok(root, ("sintel",)), ["sintel"])

    def test_bench_root_ok_checks_tae_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scannet").mkdir()
            (root / "scannet" / "scannet_video.json").write_text("{}")
            self.assertEqual(VB.bench_root_ok(root, ("scannet",)), [])
            self.assertEqual(VB.bench_root_ok(root, ("scannet",), ("scannet",)), ["scannet/tae"])
            (root / "scannet" / "scannet_video_tae.json").write_text("{}")
            self.assertEqual(VB.bench_root_ok(root, ("scannet",), ("scannet",)), [])

    def test_scaled_intrinsics_shifts_then_scales(self):
        K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1]])
        out = VB.scaled_intrinsics(K, (8, -8, 11, -11), (80, 100), (40, 50))
        self.assertAlmostEqual(out[0, 2], (50.0 - 11) * 0.5)
        self.assertAlmostEqual(out[1, 2], (40.0 - 8) * 0.5)
        self.assertAlmostEqual(out[0, 0], 50.0)


class BenchmarkCfgAndAggregateTest(unittest.TestCase):
    def test_cfg_validation(self):
        bench_eval.BenchmarkCfg().validate()
        with self.assertRaises(ValueError):
            bench_eval.BenchmarkCfg(datasets=("nope",)).validate()
        with self.assertRaises(ValueError):
            bench_eval.BenchmarkCfg(densities=(0.05,), cloud_density=0.4).validate()
        with self.assertRaises(ValueError):
            bench_eval.BenchmarkCfg(image_size=500).validate()

    def test_has_cameras_rejects_nonfinite_pose(self):
        K = np.eye(3)
        good = VB.Frame(Path("a"), Path("b"), 1.0, K, np.eye(4))
        bad_pose = np.eye(4); bad_pose[0, 3] = -np.inf
        bad = VB.Frame(Path("a"), Path("b"), 1.0, K, bad_pose)
        none = VB.Frame(Path("a"), Path("b"), 1.0, K, None)
        self.assertTrue(bench_eval.has_cameras(VB.Sequence("x", "s", [good, good])))
        self.assertFalse(bench_eval.has_cameras(VB.Sequence("x", "s", [good, bad])))
        self.assertFalse(bench_eval.has_cameras(VB.Sequence("x", "s", [good, none])))

    def test_density_key(self):
        self.assertEqual(bench_eval.density_key(0.05), "d5")
        self.assertEqual(bench_eval.density_key(0.4), "d40")
        self.assertEqual(bench_eval.density_key(0.01), "d1")

    def test_aggregate_means_over_sequences_and_skips_nan(self):
        def row(seq, mode, d, absrel):
            m = {"abs_rel": absrel, "rmse": 1.0, "delta1": 0.5, "frames": 10.0}
            return {
                "dataset": "bonn", "sequence": seq, "mode": mode, "density": d,
                "realized_density": d * 0.9, "frames": 10,
                "published": dict(m), "sparse_aligned": dict(m), "metric": dict(m),
            }
        rows = [row("a", "stream", 0.05, 0.1), row("b", "stream", 0.05, 0.3), row("a", "stream", 0.4, float("nan"))]
        tae = [{"dataset": "bonn", "sequence": "a", "mode": "stream", "density": 0.05,
                "realized_density": 0.04, "frames": 10, "tae_vda": 2.0, "tae_ours": 0.02, "tae_ours_sq": 0.001}]
        out = bench_eval.aggregate(rows, tae)
        self.assertAlmostEqual(out["bonn/stream/d5/published_abs_rel"], 0.2)
        self.assertEqual(out["bonn/stream/d5/n_sequences"], 2.0)
        self.assertTrue(np.isnan(out["bonn/stream/d40/published_abs_rel"]))
        self.assertAlmostEqual(out["bonn/stream/d5/tae_vda"], 2.0)
        self.assertAlmostEqual(out["bonn/stream/d5/realized_density"], 0.045)


if __name__ == "__main__":
    unittest.main()
