"""InfiniteVGGT / OVGGT arms and the CUT3R long video-depth reproduction:
frame selection, scoring plumbing, input sizing and the gate, on CPU (no
weights).

    export PYTHONPATH=/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd/baselines_pydeps
    LD_LIBRARY_PATH=~/.local/lib \\
      ~/.pyenv/versions/3.11.13/envs/metrics/bin/python -m unittest tests/test_cache_vggt_arms.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bench_baselines as BB  # noqa: E402
from eval import cut3r_video_paper as C  # noqa: E402
from eval.baselines import arms as A  # noqa: E402
from eval.baselines import record as R  # noqa: E402


def _png16(path: Path, value: int, hw=(4, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(hw, value, np.uint16)).save(path)


def _rgb(path: Path, hw=(4, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((*hw, 3), np.uint8)).save(path)


class TestFrameSelection(unittest.TestCase):
    def test_bonn_slices_from_30_capped_by_every_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seq in C.BONN_SEQS:
                d = root / f"rgbd_bonn_{seq}"
                n_rgb, n_depth = (40, 38) if seq == "crowd3" else (45, 45)
                for i in range(n_rgb):
                    _rgb(d / "rgb" / f"{1000 + i}.png")
                for i in range(n_depth):
                    _png16(d / "depth" / f"{2000 + i}.png", 5000)
                np.savetxt(d / "groundtruth.txt", np.zeros((44, 8)))
            seqs = C.bonn_sequences(100, root)
            self.assertEqual([s.name for s in seqs], list(C.BONN_SEQS))
            by = {s.name: s for s in seqs}
            # crowd3: depth has 38 -> 8 after frame 30; others: gt 44 -> 14
            self.assertEqual(len(by["crowd3"].images), 8)
            self.assertEqual(len(by["balloon2"].images), 14)
            # sorted on their own and paired by index, as upstream
            self.assertEqual(by["balloon2"].images[0].name, "1030.png")
            self.assertEqual(by["balloon2"].depths[0].name, "2030.png")
            # N smaller than what exists
            self.assertEqual(len(C.bonn_sequences(5, root)[0].images), 5)

    def test_kitti_takes_first_n_annotated_and_matching_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drive = "2011_09_26_drive_0002_sync"
            gt_dir = root / "val" / drive / "proj_depth" / "groundtruth" / "image_02"
            img_dir = root / "2011_09_26" / drive / "image_02" / "data"
            for i in range(5, 12):
                _png16(gt_dir / f"{i:010d}.png", 2560)
            for i in range(0, 15):
                _rgb(img_dir / f"{i:010d}.png")
            (seq,) = C.kitti_sequences(3, root)
            self.assertEqual(seq.name, f"{drive}_02")
            self.assertEqual(
                [p.name for p in seq.depths], [f"{i:010d}.png" for i in (5, 6, 7)]
            )
            self.assertEqual([p.name for p in seq.images], [p.name for p in seq.depths])
            (img_dir / f"{6:010d}.png").unlink()
            with self.assertRaises(FileNotFoundError):  # never silently shift
                C.kitti_sequences(3, root)

    def test_gt_units_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.png"
            arr = np.full((2, 2), 5000, np.uint16)
            arr[0, 0] = 0
            Image.fromarray(arr).save(p)
            seq = C.Sequence("s", [p], [p])
            gt = C.read_gt("ttt3r_bonn_100", seq)
            self.assertAlmostEqual(gt[0, 1, 1], 1.0)
            self.assertEqual(gt[0, 0, 0], -1.0)
            self.assertAlmostEqual(
                C.read_gt("ttt3r_kitti_100", seq)[0, 1, 1], 5000 / 256
            )


class TestScoring(unittest.TestCase):
    def test_summary_is_pixel_weighted(self):
        s = C.summarize(
            [
                {"abs_rel": 0.1, "delta1": 0.9, "rmse": 1.0, "valid_pixels": 100.0},
                {"abs_rel": 0.4, "delta1": 0.6, "rmse": 2.0, "valid_pixels": 300.0},
            ]
        )
        self.assertAlmostEqual(s["abs_rel"], 0.325)
        self.assertNotIn("valid_pixels", s)

    def test_evaluate_fits_one_scale_per_sequence(self):
        try:
            import torch

            if not torch.cuda.is_available():
                raise unittest.SkipTest(
                    "tools.depth_evaluation(use_gpu=True) needs CUDA"
                )
        except ImportError:
            raise unittest.SkipTest("torch missing")
        gt = np.random.default_rng(0).uniform(1, 5, (3, 8, 8))
        m = C.evaluate(gt / 2.0, gt, "ttt3r_bonn_100")  # a pure scale error
        self.assertLess(m["abs_rel"], 1e-4)

    def test_dataset_names(self):
        self.assertEqual(C.parse("ttt3r_kitti_300"), ("kitti", 300))
        with self.assertRaises(KeyError):
            C.parse("ttt3r_kitti_250")  # not a length either paper reports


class TestArmSizing(unittest.TestCase):
    def test_image_sizing_matches_load_images_for_eval(self):
        model_cls, _, resize, norm = A._fork_parts(
            A.InfiniteVGGTArm.ROOT, A.InfiniteVGGTArm.MODULE, A.InfiniteVGGTArm.CLASS
        )
        import PIL.Image

        arm = object.__new__(A.InfiniteVGGTArm)
        arm._resize, arm._norm, arm._pil = resize, norm, PIL.Image
        with A._vendored(A.InfiniteVGGTArm.ROOT / "src", *A._FORK_SHADOWED):
            from dust3r.utils.image import load_images_for_eval
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            # square frames crash upstream itself (a float size); none here
            for hw in ((480, 640), (375, 1242), (436, 1024), (352, 1216)):
                frame = rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
                path = os.path.join(tmp, f"f{hw[0]}.png")
                Image.fromarray(frame).save(path)
                ref = load_images_for_eval([path], size=518, crop=False, verbose=False)[
                    0
                ]["img"]
                ours = arm._image(frame)
                self.assertEqual(tuple(ours.shape), tuple(ref.shape), hw)
                self.assertTrue(np.allclose(ours.numpy(), ref.numpy()), hw)

    def test_forks_do_not_leak_their_packages(self):
        import torch

        import streamvggt

        before = streamvggt.__file__
        models_before = sys.modules.get("models")
        path_before = list(sys.path)
        tf32_before = torch.backends.cuda.matmul.allow_tf32
        for arm in (A.OVGGTArm, A.InfiniteVGGTArm):
            A._fork_parts(arm.ROOT, arm.MODULE, arm.CLASS)
        import streamvggt as again

        self.assertEqual(again.__file__, before)  # ours, not the fork's
        self.assertIs(sys.modules.get("models"), models_before)
        self.assertEqual(sys.path, path_before)  # their croco / src entries undone
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32_before)


class TestRunner(unittest.TestCase):
    def test_longest_prefix_is_predicted_once_and_sliced(self):
        calls = []

        class FakeArm:
            info = A.ArmInfo("fake", "depth", causal=True, prompted=False, metric=False)

            def predict(self, frames):
                calls.append(len(frames))
                return (
                    np.arange(len(frames), dtype=np.float32)[:, None, None]
                    * np.ones((len(frames), 4, 6), np.float32)
                    + 1.0
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seq in C.BONN_SEQS:
                d = root / f"rgbd_bonn_{seq}"
                for i in range(45):
                    _rgb(d / "rgb" / f"{1000 + i}.png")
                    _png16(d / "depth" / f"{2000 + i}.png", 5000)
                np.savetxt(d / "groundtruth.txt", np.zeros((45, 8)))
            orig = C.BONN_ROOT
            C.BONN_ROOT = root
            seen = []
            orig_eval = C.evaluate
            C.evaluate = lambda pred, gt, ds: (
                seen.append((len(pred), float(pred[-1, 0, 0])))
                or {"abs_rel": 0.0, "delta1": 1.0, "rmse": 0.0, "valid_pixels": 1.0}
            )
            try:
                # bonn_sequences reads its root as a default argument: rebind
                C.bonn_sequences.__defaults__ = (root,)
                cache, longest = {}, {"bonn": 500}
                t = [{"id": "x", "protocol": "cut3r_scale", "metric": "abs_rel"}]
                BB.run_cut3r_paper(FakeArm(), "ttt3r_bonn_500", t, 1, longest, cache)
                BB.run_cut3r_paper(FakeArm(), "ttt3r_bonn_100", t, 1, longest, cache)
                with self.assertRaises(
                    ValueError
                ):  # a protocol this runner does not score
                    BB.run_cut3r_paper(
                        FakeArm(),
                        "ttt3r_bonn_100",
                        [
                            {
                                "id": "y",
                                "protocol": "cut3r_scale_shift",
                                "metric": "abs_rel",
                            }
                        ],
                        1,
                        longest,
                        cache,
                    )
            finally:
                C.BONN_ROOT = orig
                C.bonn_sequences.__defaults__ = (orig,)
                C.evaluate = orig_eval
        self.assertEqual(calls, [15])  # one inference, 45 - 30 frames
        self.assertEqual(seen, [(15, 15.0), (15, 15.0)])  # 100 > 15 frames: same prefix

    def test_up_to_scale_depth_is_never_scored_metric(self):
        info = A.ArmInfo("x", "depth", causal=True, prompted=False, metric=False)

        class Arm:
            pass

        arm = Arm()
        arm.info = info
        with self.assertRaises(ValueError):
            BB.score_main(
                arm, np.ones((1, 2, 2)), np.ones((1, 2, 2)), {("metric", 10.0)}
            )


class TestGate(unittest.TestCase):
    def test_new_arms_route_and_gate(self):
        rec = R.load()
        for arm in ("infinite_vggt", "ovggt"):
            entry = R.arm_entry(rec, arm)
            # reproduced 2026-10-27 (jobs 6485945, 6485902)
            self.assertEqual(R.computed_status(entry, rec["tolerance"]), R.VERIFIED)
            for t in R.runnable_targets(entry):
                self.assertEqual(BB.runner_for(t["dataset"]), "cut3r")
                self.assertEqual(t["protocol"], "cut3r_scale")
            # verified: our benchmark protocols open, without --allow-close
            R.require_allowed(rec, arm, [("scannet", "published"), ("scannet", "tae")])


if __name__ == "__main__":
    unittest.main()
