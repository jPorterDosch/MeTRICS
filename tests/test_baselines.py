"""Baseline arms: the reproduction gate, the adapter contract and the
reproduction runner's plumbing, on CPU with fake models (no weights).

    export PYTHONPATH=/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd/baselines_pydeps  # easydict
    LD_LIBRARY_PATH=~/.local/lib \\
      ~/.pyenv/versions/3.11.13/envs/metrics/bin/python -m unittest tests/test_baselines.py
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import bench_baselines as BB  # noqa: E402
from eval import arkit_upsampling as ARKIT  # noqa: E402
from eval import protocols as P  # noqa: E402
from eval import vda_benchmark as VB  # noqa: E402
from eval.baselines import arms as A  # noqa: E402
from eval.baselines import record as R  # noqa: E402

TOL = {"relative": 0.02, "last_digit_units": 2}


def _target(
    published="0.100", scale=1, gate=True, dataset="bonn", protocol="published"
):
    return {
        "id": f"{dataset}_{protocol}_{published}",
        "source": "test",
        "published": published,
        "scale": scale,
        "dataset": dataset,
        "protocol": protocol,
        "metric": "abs_rel",
        "gate": gate,
    }


def _record(*targets, status=R.PENDING):
    return {
        "tolerance": TOL,
        "arms": {
            "fake": {
                "status": status,
                "status_note": "",
                "targets": list(targets),
                "runs": [],
            }
        },
    }


class ToleranceTest(unittest.TestCase):
    def test_larger_of_relative_and_last_digit(self):
        # 2% of 0.944 beats two units of the third decimal
        self.assertAlmostEqual(R.tolerance("0.944", TOL), 0.01888)
        # two units of the third decimal beat 2% of 0.053
        self.assertAlmostEqual(R.tolerance("0.053", TOL), 0.002)
        # the last digit is read off the string: "0.570" has three decimals
        self.assertAlmostEqual(R.tolerance("0.570", TOL), 0.0114)
        self.assertAlmostEqual(R.tolerance("5.1", TOL), 0.2)

    def test_within_converts_our_unit_to_the_printed_one(self):
        pct = _target("5.1", scale=100)
        self.assertTrue(R.within(pct, 0.0529, TOL))  # 5.29 vs 5.1 +- 0.2
        self.assertFalse(R.within(pct, 0.0535, TOL))
        self.assertFalse(R.within(pct, float("nan"), TOL))

    def test_boundary_is_inside(self):
        self.assertTrue(R.within(_target("0.053"), 0.055, TOL))


class GateTest(unittest.TestCase):
    def test_unverified_arm_runs_only_its_paper_settings(self):
        rec = _record(_target(dataset="sintel", protocol="published"))
        R.require_allowed(rec, "fake", [("sintel", "published")])
        for use in [
            ("sintel", "sparse_aligned"),
            ("sintel", "metric"),
            ("kitti", "published"),
            ("spot", "metric"),
        ]:
            with self.assertRaises(R.GateError):
                R.require_allowed(rec, "fake", [("sintel", "published"), use])

    def test_failed_arm_is_as_closed_as_a_pending_one(self):
        rec = _record(_target(dataset="sintel"), status=R.FAILED)
        with self.assertRaises(R.GateError):
            R.require_allowed(rec, "fake", [("kitti", "published")])

    def test_verified_arm_is_open(self):
        rec = _record(_target(dataset="sintel"), status=R.VERIFIED)
        R.require_allowed(
            rec, "fake", [("kitti", "sparse_aligned"), ("spot", "metric")]
        )

    def test_target_without_dataset_authorises_nothing(self):
        rec = _record(_target(dataset=None, gate=False))
        self.assertEqual(R.paper_uses(rec["arms"]["fake"]), set())

    def test_build_arm_raises_before_any_model_is_built(self):
        # every arm of the real record is unverified; no weights are touched
        rec = R.load()
        for name in A.ARMS:
            if rec["arms"][name]["status"] == R.VERIFIED:
                continue
            with self.assertRaises(R.GateError):
                A.build_arm(
                    name, torch.device("cpu"), [("kitti", "sparse_aligned")], rec
                )
        with self.assertRaises(KeyError):
            A.build_arm("nope", torch.device("cpu"), [("kitti", "published")], rec)
        with self.assertRaises(ValueError):
            A.build_arm("vda", torch.device("cpu"), [], rec)


class StatusTest(unittest.TestCase):
    def setUp(self):
        self.a = _target("0.100")
        self.b = _target("0.900")
        self.info = _target("0.500", gate=False)
        self.rec = _record(self.a, self.b, self.info)

    def _apply(self, **measured):
        ids = {"a": self.a["id"], "b": self.b["id"], "info": self.info["id"]}
        return R.apply_measurements(
            self.rec, "fake", {ids[k]: v for k, v in measured.items()}, {"job_id": "1"}
        )

    def test_verified_needs_every_gating_target(self):
        self.assertEqual(self._apply(a=0.1005), R.PENDING)
        self.assertEqual(self._apply(b=0.899), R.VERIFIED)
        self.assertEqual(len(self.rec["arms"]["fake"]["runs"]), 2)

    def test_one_gating_miss_fails_the_arm(self):
        self.assertEqual(self._apply(a=0.1005, b=0.80), R.FAILED)
        # and a later run that fixes it verifies
        self.assertEqual(self._apply(b=0.90), R.VERIFIED)

    def test_non_gating_miss_does_not_fail(self):
        self.assertEqual(self._apply(a=0.1, b=0.9, info=0.1), R.VERIFIED)
        self.assertFalse(self.info["within_tolerance"])

    def test_verified_needs_all_gating_targets_from_the_same_code(self):
        ids = {"a": self.a["id"], "b": self.b["id"]}
        R.apply_measurements(self.rec, "fake", {ids["a"]: 0.1}, {"commit": "c1"})
        # the other half measured by different code: not verified
        self.assertEqual(
            R.apply_measurements(self.rec, "fake", {ids["b"]: 0.9}, {"commit": "c2"}),
            R.PENDING,
        )
        # re-measured by the same code: verified
        self.assertEqual(
            R.apply_measurements(self.rec, "fake", {ids["a"]: 0.1}, {"commit": "c2"}),
            R.VERIFIED,
        )

    def test_unknown_target_id_raises(self):
        with self.assertRaises(KeyError):
            R.apply_measurements(self.rec, "fake", {"typo": 1.0}, {})


class RecordFileTest(unittest.TestCase):
    """The checked-in record is well-formed and claims nothing unmeasured."""

    def test_schema_and_no_unmeasured_verification(self):
        rec = R.load()
        self.assertEqual(set(rec["arms"]), set(A.ARMS))
        for name, entry in rec["arms"].items():
            self.assertIn(entry["status"], (R.PENDING, R.VERIFIED, R.FAILED), name)
            ids = [t["id"] for t in entry["targets"]]
            self.assertEqual(len(ids), len(set(ids)), name)
            gating = [t for t in entry["targets"] if t["gate"]]
            self.assertTrue(gating, f"{name} has no gating target")
            for t in entry["targets"]:
                float(t["published"])
                self.assertTrue(t["source"], t["id"])
                if t["gate"]:
                    self.assertIsNotNone(t["dataset"], f"{name}/{t['id']}")
                else:
                    self.assertTrue(t["why_not_gating"], f"{name}/{t['id']}")
            if entry["status"] == R.VERIFIED:
                self.assertTrue(entry["runs"], name)
                self.assertTrue(all(t.get("within_tolerance") for t in gating), name)
            # recomputing the status from the stored measurements must agree
            again = copy.deepcopy(rec)
            stored = {
                t["id"]: t["measured"] for t in entry["targets"] if "measured" in t
            }
            self.assertEqual(
                R.apply_measurements(again, name, stored, {}), entry["status"], name
            )


def _gt(S=4, H=12, W=16, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1.0, 5.0, size=(S, H, W)).astype(np.float32)
    gt[rng.uniform(size=gt.shape) < 0.1] = 0.0
    return gt


class FirstFrameProtocolTest(unittest.TestCase):
    def test_consistent_affine_disparity_scores_perfectly(self):
        gt = _gt()
        disp = 3.0 / np.where(gt > 0, gt, 1.0) + 0.2
        m = P.ovda_aligned_metrics(disp, gt, 10.0, True)
        self.assertLess(m.abs_rel, 1e-5)
        self.assertEqual(m.frames, 4)

    def test_scale_drift_is_scored_not_absorbed(self):
        gt = _gt()
        disp = 1.0 / np.where(gt > 0, gt, 1.0)
        disp[2:] *= 1.5  # the scale drifts after frame 1
        first = P.ovda_aligned_metrics(disp, gt, 10.0, True)
        per_video, _ = P.published_metrics(disp, gt, max_depth=10.0)
        self.assertGreater(first.abs_rel, 0.15)  # frames 2-3 are 1/1.5 of GT
        self.assertGreater(first.abs_rel, per_video.abs_rel)

    def test_global_fit_absorbs_drift_and_scores_negatives_as_zero_depth(self):
        gt = _gt()
        disp = 1.0 / np.where(gt > 0, gt, 1.0)
        disp[2:] *= 1.5
        first = P.ovda_aligned_metrics(disp, gt, 10.0, True)
        glob = P.ovda_aligned_metrics(disp, gt, 10.0, False)
        self.assertLess(glob.abs_rel, first.abs_rel)

    def test_negative_aligned_disparity_scores_as_zero_depth(self):
        gt = _gt()
        disp = 1.0 / np.where(gt > 0, gt, 1.0)
        disp[1, 2, 2] = -50.0  # outside the fitted first frame
        gt[1, 2, 2] = 2.0
        m = P.ovda_aligned_metrics(disp, gt, 10.0, True)
        # depth 0 -> AbsRel exactly 1 on that pixel (VDA's floor would score
        # max_depth / gt - 1 = 4): 1 / (valid pixels of frame 1) / 4 frames
        expected = 1.0 / (gt[1] > 0).sum() / 4
        self.assertAlmostEqual(m.abs_rel, expected, places=5)

    def test_gt_beyond_max_depth_is_clipped_and_scored(self):
        gt = _gt()
        gt[:, 0, :4] = 500.0  # "sky"
        disp = 1.0 / np.minimum(np.where(gt > 0, gt, 1.0), 10.0)
        m = P.ovda_aligned_metrics(disp, gt, 10.0, False)
        self.assertLess(m.abs_rel, 1e-5)  # sky predicted at the clip: no error
        disp[:, 0, :4] = 1.0  # sky predicted at 1 m: an error, because it IS scored
        self.assertGreater(P.ovda_aligned_metrics(disp, gt, 10.0, False).abs_rel, 0.01)

    def test_first_frame_without_gt_scores_nothing(self):
        gt = _gt()
        gt[0] = 0.0
        m = P.ovda_aligned_metrics(1.0 / np.where(gt > 0, gt, 1.0), gt, 10.0, True)
        self.assertEqual(m.frames, 0)

    def test_nan_pixel_is_an_error_not_a_crash(self):
        gt = _gt()
        disp = 1.0 / np.where(gt > 0, gt, 1.0)
        disp[1, 3, 3] = np.nan
        gt[1, 3, 3] = 2.0
        m = P.ovda_aligned_metrics(disp, gt, 10.0, True)
        self.assertTrue(np.isfinite(m.abs_rel))
        self.assertGreater(m.abs_rel, 0.0)


class _FakeArm:
    """Returns an affine function of the true disparity at half resolution,
    so `published` must score ~0 after registration and alignment."""

    def __init__(self, gt_of, output="disparity"):
        self.info = A.ArmInfo("fake", output, causal=False, prompted=False)
        self.gt_of = gt_of
        self.calls = []

    def predict(self, frames, prompt=None):
        self.calls.append(frames.shape)
        depth = self.gt_of(frames)
        out = 2.0 / depth + 0.1 if self.info.output == "disparity" else depth
        return np.stack(
            [
                cv2.resize(o, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                for o in out
            ]
        ).astype(np.float32)


def _fake_bonn_tree(root: Path, n_seq=2, S=3):
    """A Bonn-shaped tree (480x640, depth factor 5000) whose RGB red channel
    encodes the depth plane, so a fake arm can 'predict' from the frames."""
    H, W = 480, 640
    entries = []
    for q in range(n_seq):
        frames = []
        for i in range(S):
            level = 40 + 30 * q + 10 * i  # red value; depth = level / 40 m
            rgb = np.zeros((H, W, 3), np.uint8)
            rgb[..., 0] = level
            rgb[: H // 2, :, 0] += 40  # two depth planes per frame
            depth = (rgb[..., 0].astype(np.float32) / 40.0 * 5000).astype(np.uint16)
            d = root / "bonn" / f"seq{q}"
            (d / "rgb").mkdir(parents=True, exist_ok=True)
            (d / "depth").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "rgb" / f"{i}.png"), rgb[..., ::-1])
            cv2.imwrite(str(d / "depth" / f"{i}.png"), depth)
            frames.append(
                {
                    "image": f"seq{q}/rgb/{i}.png",
                    "gt_depth": f"seq{q}/depth/{i}.png",
                    "factor": 5000.0,
                }
            )
        entries.append({f"seq{q}": frames})
    with open(root / "bonn" / "bonn_video.json", "w") as f:
        json.dump({"bonn": entries}, f)


def _depth_from_red(frames):
    return frames[..., 0].astype(np.float32) / 40.0


class RunnerPlumbingTest(unittest.TestCase):
    def test_video_dataset_reaches_every_target_through_our_scoring(self):
        targets = [
            {
                **_target(dataset="bonn", protocol="published"),
                "id": "pub",
                "metric": "abs_rel",
            },
            {
                **_target(dataset="bonn", protocol="first_frame"),
                "id": "ff",
                "metric": "delta1",
            },
            {
                **_target(dataset="bonn", protocol="published"),
                "id": "pub80",
                "metric": "abs_rel",
                "max_depth": 80.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _fake_bonn_tree(Path(tmp))
            arm = _FakeArm(_depth_from_red)
            measured, rows = BB.run_video_dataset(arm, "bonn", targets, Path(tmp), 0)
        self.assertEqual(arm.calls, [(3, 480, 640, 3)] * 2)  # raw frames, one pass
        self.assertEqual(len(rows), 2)
        self.assertLess(measured["pub"], 1e-3)
        self.assertLess(measured["pub80"], 1e-3)
        self.assertGreater(measured["ff"], 0.999)
        self.assertEqual(
            set(rows[0]),
            {
                "dataset",
                "sequence",
                "frames",
                "published@10",
                "first_frame@10",
                "published@80",
            },
        )

    def test_max_sequences_caps_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fake_bonn_tree(Path(tmp))
            arm = _FakeArm(_depth_from_red)
            _, rows = BB.run_video_dataset(
                arm, "bonn", [{**_target(), "id": "pub"}], Path(tmp), 1
            )
        self.assertEqual(len(rows), 1)

    def test_metric_protocol_needs_a_metric_arm(self):
        gt = _gt()
        rel = _FakeArm(None, "disparity")
        with self.assertRaises(ValueError):
            BB.score_main(rel, gt, gt, {("metric", 10.0)})
        met = _FakeArm(None, "depth")
        out = BB.score_main(met, np.where(gt > 0, gt, 1.0), gt, {("metric", 10.0)})
        self.assertLess(out[("metric", 10.0)]["abs_rel"], 1e-6)

    def test_uncropped_frames_must_match_the_uncropped_gt_size(self):
        seq = VB.Sequence("scannet", "s", [], rgb_uncropped=True)
        spec = VB.SPECS["scannet"]  # crop [8:-8, 11:-11] of a 480x640 frame -> 464x618
        pred = np.zeros((2, 480, 640), np.float32)
        self.assertEqual(
            BB.register_to_gt(pred, seq, spec, (480, 640), (464, 618)).shape,
            (2, 464, 618),
        )
        with self.assertRaises(
            ValueError
        ):  # a 968x1296 export: the crop would mean other pixels
            BB.register_to_gt(
                np.zeros((2, 968, 1296), np.float32), seq, spec, (968, 1296), (464, 618)
            )

    def test_frame_count_mismatch_is_fatal(self):
        class Short(_FakeArm):
            def predict(self, frames, prompt=None):
                return super().predict(frames)[:-1]

        with tempfile.TemporaryDirectory() as tmp:
            _fake_bonn_tree(Path(tmp))
            with self.assertRaises(ValueError):
                BB.run_video_dataset(
                    Short(_depth_from_red),
                    "bonn",
                    [{**_target(), "id": "pub"}],
                    Path(tmp),
                    0,
                )


class _FakePromptDA:
    def __init__(self):
        self.seen = []

    def predict(self, image, prompt):
        self.seen.append((tuple(image.shape), tuple(prompt.shape), float(image.max())))
        return torch.nn.functional.interpolate(prompt, image.shape[-2:], mode="nearest")


def _bare(cls, info, **attrs):
    arm = cls.__new__(cls)
    arm.info = info
    for k, v in attrs.items():
        setattr(arm, k, v)
    return arm


class AdapterContractTest(unittest.TestCase):
    def test_rgb_only_arms_refuse_sparse_depth(self):
        frames = np.zeros((2, 28, 28, 3), np.uint8)
        prompt = (np.ones((2, 28, 28), np.float32), np.ones((2, 28, 28), bool))
        for cls, name in [(A.VDAArm, "vda"), (A.OVDAArm, "ovda"), (A.DAVArm, "dav")]:
            arm = _bare(cls, A.ArmInfo(name, "disparity", False, False), model=None)
            with self.assertRaises(ValueError, msg=name):
                arm.predict(frames, prompt)

    def test_frames_must_be_uint8_rgb(self):
        arm = _bare(A.VDAArm, A.ArmInfo("vda", "disparity", False, False), model=None)
        with self.assertRaises(ValueError):
            arm.predict(np.zeros((2, 28, 28, 3), np.float32))
        with self.assertRaises(ValueError):
            arm.predict(np.zeros((28, 28, 3), np.uint8))

    def test_promptda_uses_upstream_input_sizing(self):
        fake = _FakePromptDA()
        arm = _bare(
            A.PromptDAArm,
            A.ArmInfo("promptda", "depth", True, True),
            model=fake,
            device=torch.device("cpu"),
        )
        frames = np.full((2, 30, 45, 3), 255, np.uint8)
        depth = np.full((2, 6, 8), 2.0, np.float32)
        valid = np.ones((2, 6, 8), bool)
        valid[:, 0, 0] = False  # a hole: must be infilled, not passed as 0
        depth[:, 0, 0] = 0.0
        out = arm.predict(frames, (depth, valid))
        self.assertEqual(out.shape, (2, 28, 42))  # model resolution
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out, 2.0))
        # image at 28x42 (floor to 14) in [0,1]; prompt at its own resolution
        self.assertEqual(fake.seen[0][0], (1, 3, 28, 42))
        self.assertEqual(fake.seen[0][1], (1, 1, 6, 8))
        self.assertAlmostEqual(fake.seen[0][2], 1.0)
        with self.assertRaises(ValueError):
            arm.predict(frames)
        # the long side is capped at 1008 first, as upstream's load_image
        big = np.zeros((1, 1440, 1920, 3), np.uint8)
        arm.predict(big, (depth[:1], valid[:1]))
        self.assertEqual(fake.seen[-1][0], (1, 3, 756, 1008))

    def test_dav_refuses_unregistered_outputs(self):
        with A._vendored(A.THIRD_PARTY / "depth_any_video"):
            spec = importlib.util.spec_from_file_location(
                "dav_img_utils",
                A.THIRD_PARTY / "depth_any_video" / "dav" / "utils" / "img_utils.py",
            )
            img_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(img_utils)
        arm = _bare(
            A.DAVArm,
            A.ArmInfo("dav", "disparity", False, False),
            img_utils=img_utils,
            device=torch.device("cpu"),
        )
        with self.assertRaises(ValueError):  # longer than one window
            arm.predict(np.zeros((33, 64, 64, 3), np.uint8))
        with self.assertRaises(ValueError):  # 375 is not a multiple of 32
            arm.predict(np.zeros((1, 375, 1242, 3), np.uint8))


class VendoredImportTest(unittest.TestCase):
    def test_vendored_context_leaves_no_trace(self):
        marker = object()
        sys.modules["utils"] = marker  # something the process already had
        path_before = list(sys.path)
        try:
            with A._vendored(A.THIRD_PARTY / "video_depth_anything", "utils"):
                import utils.util as vendored_util
            self.assertTrue(hasattr(vendored_util, "compute_scale_and_shift"))
            self.assertIs(sys.modules["utils"], marker)
            self.assertNotIn("utils.util", sys.modules)
            self.assertEqual(sys.path, path_before)
        finally:
            del sys.modules["utils"]
        with self.assertRaises(FileNotFoundError):
            with A._vendored(A.THIRD_PARTY / "no_such_repo"):
                pass

    @unittest.skipUnless(
        importlib.util.find_spec("easydict"), "needs easydict (see module docstring)"
    )
    def test_vendored_models_build_from_their_own_code(self):
        # the process may already have its own `utils` / `models` (croco puts
        # a `models` package on sys.path): they must come back untouched
        def generic():
            return {
                k: v
                for k, v in sys.modules.items()
                if k.split(".")[0] in ("utils", "models")
            }

        before = generic()
        vda = A._vda_class()(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
        self.assertTrue(hasattr(vda, "infer_video_depth"))
        import yaml

        with open(A._OVDA_ROOT / "configs" / "oVDA_c16.yaml") as f:
            ovda = A._ovda_class()(**yaml.safe_load(f)["net"])
        self.assertEqual(ovda.cache_size, 16)
        self.assertEqual(generic(), before)


def _png(arr) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


class ArkitUpsamplingTest(unittest.TestCase):
    def _zip(self, root: Path, vid="123", n=2):
        (root / "Validation").mkdir(parents=True)
        with zipfile.ZipFile(root / "Validation" / f"{vid}.zip", "w") as zf:
            for i in range(n):
                rgb = np.zeros((1440, 1920, 3), np.uint8)
                rgb[:100, :100, 2] = 255  # BGR on disk: a RED top-left block
                gt = np.full((1440, 1920), 2000, np.uint16)
                gt[:10] = 0  # no laser return
                low = np.full((192, 256), 2100, np.uint16)
                zf.writestr(f"{vid}/wide/{vid}_{i}.000.png", _png(rgb))
                zf.writestr(f"{vid}/highres_depth/{vid}_{i}.000.png", _png(gt))
                zf.writestr(f"{vid}/lowres_depth/{vid}_{i}.000.png", _png(low))
        with open(root / "metadata.csv", "w") as f:
            f.write(f"video_id,visit_id,sky_direction,fold\n{vid},NA,Left,Validation\n")

    def test_samples_are_resized_rotated_and_in_metres(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._zip(root)
            sky = ARKIT.sky_directions(root)
            (zp,) = ARKIT.videos(root)
            got = list(ARKIT.samples(zp, sky[zp.stem]))
        self.assertEqual([s.name for s in got], ["123_0.000", "123_1.000"])
        s = got[0]
        # RGB as stored (1440x1920), rotated clockwise ("Left") -> portrait
        self.assertEqual(s.rgb.shape, (1920, 1440, 3))
        self.assertEqual(s.prompt.shape, (256, 192))  # the prompt keeps its size
        self.assertEqual(s.rgb[0, -1].tolist(), [255, 0, 0])  # RGB, block now top-right
        self.assertAlmostEqual(float(s.prompt.max()), 2.1, places=5)
        gt = s.gt_at(ARKIT.SIZES["arkit_upsampling_384"])
        self.assertEqual(gt.shape, (512, 384))  # 384x512, then rotated
        self.assertEqual(
            s.gt_at(ARKIT.SIZES["arkit_upsampling_1440"]).shape, (1920, 1440)
        )
        self.assertAlmostEqual(float(gt.max()), 2.0)
        l1, rmse = ARKIT.l1_rmse(np.full_like(gt, 2.1), gt)
        self.assertAlmostEqual(l1, 0.1, places=5)  # holes (gt == 0) are not scored
        self.assertAlmostEqual(rmse, 0.1, places=5)

    def test_runner_scores_one_prediction_at_every_size(self):
        class Copy:  # "predicts" the prompt at its own (low) resolution
            info = A.ArmInfo("promptda", "depth", True, True)
            calls = 0

            def predict(self, frames, prompt):
                depth, valid = prompt
                self.calls += 1
                self.valid_all = bool(valid.all())
                self.frame_shape = frames.shape
                return depth

        targets = [
            {"id": "l1_384", "metric": "l1", "dataset": "arkit_upsampling_384"},
            {"id": "rmse_384", "metric": "rmse", "dataset": "arkit_upsampling_384"},
            {"id": "l1_1440", "metric": "l1", "dataset": "arkit_upsampling_1440"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._zip(Path(tmp))
            arm = Copy()
            measured, rows = BB.run_arkit(arm, targets, Path(tmp), 0)
        self.assertEqual(arm.calls, 2)  # one inference per frame, not per size
        self.assertEqual(arm.frame_shape, (1, 1920, 1440, 3))  # the stored frame
        self.assertEqual(len(rows), 4)  # 2 frames x 2 evaluated sizes
        self.assertTrue(arm.valid_all)
        for key in ("l1_384", "rmse_384", "l1_1440"):
            self.assertAlmostEqual(measured[key], 0.1, places=5)

    def test_missing_tree_names_the_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                ARKIT.videos(Path(tmp))


class OurProtocolsForBaselinesTest(unittest.TestCase):
    """Step 5: a verified arm under the benchmark's own protocols."""

    def test_sparse_fit_in_native_disparity_recovers_the_affine(self):
        gt = _gt()
        depth = np.where(gt > 0, gt, 1.0)
        disp = 3.0 / depth + 0.2  # an affine-invariant disparity model
        mask = np.zeros_like(gt, bool)
        mask[:, ::3, ::4] = True
        m = P.sparse_aligned_metrics(
            disp, depth, mask, disp, mask, gt, 10.0, native_disparity=True
        )
        self.assertLess(m.abs_rel, 1e-5)
        wrong = P.sparse_aligned_metrics(
            disp, depth, mask, disp, mask, gt, 10.0
        )  # fitted as if depth
        self.assertGreater(wrong.abs_rel, 0.1)

    def test_non_finite_disparity_pixel_scores_as_an_error(self):
        gt = _gt()
        depth = np.where(gt > 0, gt, 1.0)
        disp = (1.0 / depth).astype(np.float32)
        mask = np.zeros_like(gt, bool)
        mask[:, ::3, ::4] = True
        clean = P.sparse_aligned_metrics(
            disp, depth, mask, disp, mask, gt, 10.0, native_disparity=True
        )
        disp[0, 5, 5] = np.nan
        gt[0, 5, 5] = 2.0
        mask[0, 5, 5] = False
        bad = P.sparse_aligned_metrics(
            disp, depth, mask, disp, mask, gt, 10.0, native_disparity=True
        )
        # that pixel is scored at max_depth: AbsRel (10 - 2) / 2 = 4 on one pixel of one frame
        self.assertGreater(bad.abs_rel, clean.abs_rel + 3.0 / gt[0].size / 4)

    def test_published_uses_the_gt_grid_prediction_when_given(self):
        import bench_eval as BE

        gt = _gt(S=1, H=24, W=32)
        depth = np.where(gt > 0, gt, 1.0).astype(np.float32)
        views = [
            {
                "img": torch.zeros(1, 3, 12, 16),
                "sparse_depth": torch.ones(1, 12, 16),
                "sparse_depth_mask": torch.zeros(
                    1, 12, 16, dtype=torch.bool
                ).index_fill_(1, torch.tensor([0, 1]), True),
            }
        ]
        coarse = cv2.resize(depth[0], (16, 12), interpolation=cv2.INTER_NEAREST)[None]
        spec, seq = VB.SPECS["bonn"], VB.Sequence("bonn", "s", [])
        pred = BE.Prediction(
            coarse,
            np.zeros_like(coarse),
            np.zeros((1, 3, 4)),
            np.zeros((1, 3, 3)),
            at_gt=depth,
        )
        row, _ = BE.score_sequence(spec, seq, gt, views, pred, "offline", 0.05, 0.05)
        self.assertLess(
            row["metric"]["abs_rel"], 1e-6
        )  # the GT-grid copy was used, not the upsampled coarse one
        wrong = BE.Prediction(
            coarse,
            np.zeros_like(coarse),
            np.zeros((1, 3, 4)),
            np.zeros((1, 3, 3)),
            at_gt=coarse,
        )
        with self.assertRaises(ValueError):
            BE.score_sequence(spec, seq, gt, views, wrong, "offline", 0.05, 0.05)

    def test_score_sequence_reads_a_disparity_prediction_as_such(self):
        import bench_eval as BE

        gt = _gt(S=2, H=24, W=32)
        depth = np.where(gt > 0, gt, 1.0).astype(np.float32)
        views = [
            {
                "img": torch.zeros(1, 3, 24, 32),
                "sparse_depth": torch.from_numpy(depth[i])[None],
                "sparse_depth_mask": torch.from_numpy(
                    np.indices((24, 32)).sum(0) % 5 == 0
                )[None],
            }
            for i in range(2)
        ]
        spec = VB.SPECS["bonn"]
        seq = VB.Sequence("bonn", "s", [])
        disp = BE.Prediction(
            (2.0 / depth + 0.1).astype(np.float32),
            np.zeros_like(depth),
            np.zeros((2, 3, 4)),
            np.zeros((2, 3, 3)),
            output="disparity",
        )
        row, aligned = BE.score_sequence(
            spec, seq, gt, views, disp, "offline", 0.05, 0.05
        )
        self.assertLess(row["published"]["abs_rel"], 1e-5)
        self.assertLess(row["sparse_aligned"]["abs_rel"], 1e-5)
        self.assertTrue(np.isnan(row["metric"]["abs_rel"]))  # no metric scale to score
        self.assertEqual(aligned.shape, gt.shape)
        asdepth = BE.Prediction(
            depth, np.zeros_like(depth), np.zeros((2, 3, 4)), np.zeros((2, 3, 3))
        )
        row2, _ = BE.score_sequence(spec, seq, gt, views, asdepth, "stream", 0.05, 0.05)
        self.assertLess(row2["metric"]["abs_rel"], 1e-6)

    def test_benchmark_runs_a_fake_arm_through_bench_eval(self):
        import argparse

        rec = R.load()
        rec["arms"]["fake"] = {
            "status": R.VERIFIED,
            "status_note": "",
            "targets": [],
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            _fake_bonn_tree(Path(tmp))
            rec_path = Path(tmp) / "rec.json"
            R.save(rec, rec_path)
            arm = _FakeArm(_depth_from_red)
            A.ARMS["fake"] = lambda device, record: arm
            try:
                args = argparse.Namespace(
                    record=rec_path,
                    arm="fake",
                    device="cpu",
                    bench_root=Path(tmp),
                    out=Path(tmp) / "out",
                    datasets=["bonn"],
                    tae_datasets=[],
                    densities=[0.05, 0.4],
                    image_size=518,
                    patch_size=14,
                    seed=42,
                    overwrite=False,
                    max_sequences=0,
                )
                BB.benchmark(args)
            finally:
                del A.ARMS["fake"]
            with open(Path(tmp) / "out" / "bench_fake" / "bench_results.json") as f:
                out = json.load(f)
        self.assertEqual(len(out["rows"]), 4)  # 2 sequences x 2 densities
        self.assertEqual(
            arm.calls, [(3, 480, 640, 3)] * 2
        )  # one inference per sequence, raw frames
        self.assertEqual(out["mode"], "offline")
        self.assertLess(
            out["aggregate"]["bonn/offline/d5/published_abs_rel"], 1e-3
        )  # GT-grid prediction: no model-res round trip (that gave 1.0e-3)
        self.assertLess(out["aggregate"]["bonn/offline/d5/published_abs_rel"], 3e-3)
        self.assertLess(
            out["aggregate"]["bonn/offline/d40/sparse_aligned_abs_rel"], 3e-3
        )
        self.assertTrue(np.isnan(out["aggregate"]["bonn/offline/d5/metric_abs_rel"]))
        self.assertGreater(out["aggregate"]["bonn/offline/d5/realized_density"], 0.0)

    def test_benchmark_hands_a_prompted_arm_each_draw(self):
        import argparse

        class Prompted(_FakeArm):
            def __init__(self):
                super().__init__(_depth_from_red, "depth")
                self.info = A.ArmInfo("fakep", "depth", True, True)
                self.prompts = []

            def predict(self, frames, prompt=None):
                if prompt is None:
                    raise ValueError("needs a prompt")
                depth, valid = prompt
                self.prompts.append(float(valid.mean()))
                return self.gt_of(frames)[:, ::2, ::2].astype(np.float32)

        rec = R.load()
        rec["arms"]["fakep"] = {
            "status": R.VERIFIED,
            "status_note": "",
            "targets": [],
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            _fake_bonn_tree(Path(tmp))
            rec_path = Path(tmp) / "rec.json"
            R.save(rec, rec_path)
            arm = Prompted()
            A.ARMS["fakep"] = lambda device, record: arm
            try:
                BB.benchmark(
                    argparse.Namespace(
                        record=rec_path,
                        arm="fakep",
                        device="cpu",
                        bench_root=Path(tmp),
                        out=Path(tmp) / "out",
                        datasets=["bonn"],
                        tae_datasets=[],
                        densities=[0.05, 0.4],
                        image_size=518,
                        patch_size=14,
                        seed=42,
                        overwrite=False,
                        max_sequences=1,
                    )
                )
            finally:
                del A.ARMS["fakep"]
        self.assertEqual(len(arm.prompts), 2)  # one inference PER DENSITY
        self.assertLess(
            arm.prompts[0], arm.prompts[1]
        )  # and the draw differs: 5% then 40%

    def test_benchmark_refuses_to_overwrite_results(self):
        import argparse

        rec = R.load()
        rec["arms"]["fake"] = {
            "status": R.VERIFIED,
            "status_note": "",
            "targets": [],
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            _fake_bonn_tree(Path(tmp))
            rec_path = Path(tmp) / "rec.json"
            R.save(rec, rec_path)
            out = Path(tmp) / "out" / "bench_fake"
            out.mkdir(parents=True)
            (out / "bench_results.json").write_text("{}")
            arm = _FakeArm(_depth_from_red)
            A.ARMS["fake"] = lambda device, record: arm
            try:
                args = argparse.Namespace(
                    record=rec_path,
                    arm="fake",
                    device="cpu",
                    bench_root=Path(tmp),
                    out=Path(tmp) / "out",
                    datasets=["bonn"],
                    tae_datasets=[],
                    densities=[0.05],
                    image_size=518,
                    patch_size=14,
                    seed=42,
                    max_sequences=1,
                    overwrite=False,
                )
                with self.assertRaises(FileExistsError):
                    BB.benchmark(args)
                self.assertEqual(arm.calls, [])  # refused before any inference
                args.overwrite = True
                BB.benchmark(args)
            finally:
                del A.ARMS["fake"]
        self.assertEqual(len(arm.calls), 1)

    def test_benchmark_refuses_an_unverified_arm(self):
        import argparse

        rec = R.load()
        unverified = [a for a, e in rec["arms"].items() if e["status"] != R.VERIFIED]
        self.assertTrue(unverified)
        args = argparse.Namespace(
            record=R.RECORD_PATH,
            arm=unverified[0],
            device="cpu",
            bench_root=BB.BENCH_ROOT,
            out=Path("/nonexistent"),
            datasets=["bonn"],
            tae_datasets=[],
            densities=[0.05],
            image_size=518,
            patch_size=14,
            seed=42,
            overwrite=False,
            max_sequences=1,
        )
        with self.assertRaises(R.GateError):
            BB.benchmark(args)


class FinalReviewFixesTest(unittest.TestCase):
    def test_nan_prediction_is_an_error_in_published_and_depth_sparse_fit(self):
        gt = _gt()
        gt[gt == 0] = 2.0
        disp = 1.0 / gt
        clean, _ = P.published_metrics(disp, gt, 10.0)
        bad_disp = disp.copy()
        bad_disp[0, 3, 3] = np.nan
        bad, aligned = P.published_metrics(bad_disp, gt, 10.0)
        self.assertAlmostEqual(
            float(aligned[0, 3, 3]), 10.0, places=4
        )  # max_depth: an error
        self.assertGreater(bad.abs_rel, clean.abs_rel)
        depth = gt.copy()
        depth[0, 3, 3] = np.nan
        mask = np.zeros_like(gt, bool)
        mask[:, ::3, ::4] = True
        mask[0, 3, 3] = False
        m = P.sparse_aligned_metrics(depth, gt, mask, depth, mask, gt, 10.0)
        self.assertGreater(
            m.abs_rel, 0.5 / gt[0].size / gt.shape[0]
        )  # scored at 1e-3 m

    def test_promptda_frame_without_prompt_gives_nan_not_exit(self):
        fake = _FakePromptDA()
        arm = _bare(
            A.PromptDAArm,
            A.ArmInfo("promptda", "depth", True, True),
            model=fake,
            device=torch.device("cpu"),
        )
        frames = np.full((2, 28, 28, 3), 128, np.uint8)
        depth = np.full((2, 6, 8), 2.0, np.float32)
        valid = np.ones((2, 6, 8), bool)
        valid[1] = False
        out = arm.predict(frames, (depth, valid))
        self.assertTrue(np.isfinite(out[0]).all())
        self.assertTrue(np.isnan(out[1]).all())
        self.assertEqual(len(fake.seen), 1)

    def test_tae_manifest_only_required_for_datasets_that_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "kitti").mkdir()
            (Path(tmp) / "kitti" / "kitti_video.json").write_text("{}")
            self.assertEqual(
                VB.bench_root_ok(Path(tmp), ("kitti",), ("kitti", "scannet")), []
            )
            self.assertIn(
                "scannet/tae", VB.bench_root_ok(Path(tmp), ("scannet",), ("scannet",))
            )


class BenchmarkSettingsTest(unittest.TestCase):
    def test_baseline_cli_defaults_match_benchmark_cfg(self):
        import bench_eval as BE

        cfg = BE.BenchmarkCfg()
        import argparse

        orig = argparse.ArgumentParser.parse_args
        try:
            argparse.ArgumentParser.parse_args = lambda self, *a, **k: orig(
                self, ["benchmark", "--arm", "vda"]
            )
            captured = {}
            real_benchmark = BB.benchmark
            BB.benchmark = lambda args: captured.update(vars(args))
            BB.main()
        finally:
            argparse.ArgumentParser.parse_args = orig
            BB.benchmark = real_benchmark
        self.assertEqual(captured["seed"], cfg.seed)
        self.assertEqual(captured["patch_size"], cfg.patch_size)
        self.assertEqual(captured["image_size"], cfg.image_size)
        self.assertEqual(list(captured["datasets"]), list(cfg.datasets))

    def test_tae_window_slices_every_field(self):
        import bench_eval as BE

        S = 192
        spec = VB.SPECS["scannet"]
        seq = VB.Sequence("scannet", "s", list(range(S)))
        gt = np.arange(S, dtype=np.float32)[:, None, None] * np.ones(
            (S, 2, 2), np.float32
        )
        pred = BE.Prediction(
            gt.copy(),
            gt.copy(),
            np.zeros((S, 3, 4)),
            np.zeros((S, 3, 3)),
            at_gt=gt.copy(),
        )
        win, wgt, wp = BE.tae_window(spec, seq, gt, pred)
        lo, hi = spec.tae_range
        self.assertEqual(
            (len(win.frames), len(wgt), len(wp.depth), len(wp.at_gt), len(wp.K)),
            (hi - lo,) * 5,
        )
        self.assertEqual(win.frames[0], lo)
        self.assertEqual(float(wgt[0, 0, 0]), lo)
        self.assertEqual(float(wp.at_gt[-1, 0, 0]), hi - 1)

    def test_benchmark_seed_is_not_the_training_seed(self):
        import inspect

        import bench_eval as BE

        self.assertNotIn("seed", inspect.signature(BE.run_benchmark).parameters)
        self.assertIn("amp", inspect.signature(BE.run_benchmark).parameters)
        with self.assertRaises(ValueError):
            BE.BenchmarkCfg(patch_size=0).validate()


class MetricHoldsOutFedPixelsTest(unittest.TestCase):
    def test_metric_holds_out_fed_pixels_published_does_not(self):
        import bench_eval as BE

        gt = _gt(S=1, H=24, W=32)
        gt[gt == 0] = 2.0
        fed = np.zeros((1, 24, 32), bool)
        fed[0, :12] = True
        pred_d = gt.copy()
        pred_d[~fed] *= 1.5  # perfect on the fed pixels only: a model that copies
        views = [
            {
                "img": torch.zeros(1, 3, 24, 32),
                "sparse_depth": torch.from_numpy(gt[0])[None],
                "sparse_depth_mask": torch.from_numpy(fed[0])[None],
            }
        ]
        pred = BE.Prediction(
            pred_d, np.zeros_like(pred_d), np.zeros((1, 3, 4)), np.zeros((1, 3, 3))
        )
        row, _ = BE.score_sequence(
            VB.SPECS["bonn"],
            VB.Sequence("bonn", "s", []),
            gt,
            views,
            pred,
            "stream",
            0.5,
            0.5,
        )
        self.assertAlmostEqual(
            row["metric"]["abs_rel"], 0.5, places=5
        )  # copied half not rewarded
        unmasked = P.metric_metrics(pred_d, gt, 10.0)
        self.assertLess(unmasked.abs_rel, 0.3)
        self.assertGreater(
            row["published"]["delta1"], 0.0
        )  # published still scores every valid pixel


class ParityTreesTest(unittest.TestCase):
    """The trees added after reproducing VDA: bonn_all, sintel_bgr, 100-scene TAE."""

    def test_specs(self):
        from eval.vda_benchmark import SPECS

        self.assertEqual(SPECS["scannet"].tae_scenes, 100)
        self.assertEqual(SPECS["bonn_all_500"].dirname, "bonn_all")
        self.assertEqual(SPECS["bonn_all"].max_len, 110)
        self.assertEqual(SPECS["sintel_bgr"].max_depth, SPECS["sintel"].max_depth)

    def test_benchmark_defaults_use_the_26_sequence_bonn(self):
        import bench_eval

        cfg = bench_eval.BenchmarkCfg().validate()
        self.assertIn("bonn_all", cfg.datasets)
        self.assertNotIn("bonn", cfg.datasets)
        self.assertIn("bonn_all_500", cfg.tae_datasets)
        self.assertIn(
            "bonn", cfg.tae_datasets
        )  # scored for TAE whenever it is asked for

    def test_sintel_bgr_swaps_colour_and_shares_depth(self):
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "datasets_preprocess")
        )
        import prepare_vda_benchmark as PV

        self.assertNotIn("bonn_all", ["sintel", "kitti", "bonn", "scannet", "nyuv2"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            d = out / "sintel" / "alley_1"
            (d / "clean").mkdir(parents=True)
            (d / "depth").mkdir()
            rgb = np.zeros((4, 6, 3), np.uint8)
            rgb[..., 0] = 200  # red, as RGB
            cv2.imwrite(str(d / "clean" / "f.png"), rgb[..., ::-1])
            cv2.imwrite(str(d / "depth" / "f.png"), np.full((4, 6), 7, np.uint16))
            frames = [
                {
                    "image": "alley_1/clean/f.png",
                    "gt_depth": "alley_1/depth/f.png",
                    "factor": 1.0,
                }
            ]
            with open(out / PV.MANIFEST["sintel"], "w") as f:
                json.dump({"sintel": [{"alley_1": frames}]}, f)
            PV.prepare_sintel_bgr(out, out)
            self.assertEqual(PV.check_manifest(out, "sintel_bgr"), 1)
            swapped = cv2.imread(
                str(out / "sintel_bgr" / "alley_1" / "clean" / "f.png")
            )
            self.assertEqual(swapped[0, 0].tolist(), [200, 0, 0])  # red now sits in B
            depth = out / "sintel_bgr" / "alley_1" / "depth" / "f.png"
            self.assertTrue(depth.is_symlink())
            self.assertEqual(int(cv2.imread(str(depth), -1)[0, 0]), 7)
            # the source tree is untouched
            self.assertEqual(
                cv2.imread(str(d / "clean" / "f.png"))[0, 0].tolist(), [0, 0, 200]
            )

    def test_rename_manifest(self):
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "datasets_preprocess")
        )
        import prepare_vda_benchmark as PV

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "a.json", Path(tmp) / "b.json"
            with open(src, "w") as f:
                json.dump({"bonn": [{"s": []}]}, f)
            PV._rename_manifest(src, dst, "bonn", "bonn_all")
            with open(dst) as f:
                self.assertEqual(json.load(f), {"bonn_all": [{"s": []}]})


if __name__ == "__main__":
    unittest.main()
