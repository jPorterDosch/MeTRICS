import ast
import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


ROOT = pathlib.Path(__file__).parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CustomMaskAlignmentTest(unittest.TestCase):
    def test_custom_mask_preserves_main_affine_fit_population(self):
        pred = np.array([[1.0, 2.0, 100.0]], dtype=np.float32)
        gt = np.array([[2.0, 4.0, 6.0]], dtype=np.float32)
        custom_mask = np.array([[True, True, False]])
        cases = [
            (
                "temporal",
                "src/eval/temporal_consistency/metrics.py",
                {"scale_and_shift": True},
            ),
            ("monodepth", "src/eval/monodepth/tools.py", {"align_with_lstsq": True}),
            ("video", "src/eval/video_depth/tools.py", {"align_with_lstsq": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_{name}", path)
                metrics, *_ = module.depth_evaluation(
                    pred, gt, max_depth=None, custom_mask=custom_mask, **alignment
                )
                self.assertAlmostEqual(metrics["Abs Rel"], 0.368635982, places=6)


class ExactDeltaThresholdTest(unittest.TestCase):
    def test_perfect_depth_preserves_main_strict_delta_one(self):
        depth = np.ones((1, 2), dtype=np.float32)
        cases = [
            (
                "temporal",
                "src/eval/temporal_consistency/metrics.py",
                {"metric_scale": True},
            ),
            ("monodepth", "src/eval/monodepth/tools.py", {"metric_scale": True}),
            ("video", "src/eval/video_depth/tools.py", {"metric_scale": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_delta_{name}", path)
                metrics, *_ = module.depth_evaluation(
                    depth, depth, max_depth=None, **alignment
                )
                self.assertEqual(metrics["delta < 1."], 0.0)


class ReprojectionOcclusionTest(unittest.TestCase):
    def test_reprojection_preserves_main_last_write_behavior(self):
        module = load_module(
            "area_c_reprojection", "src/eval/temporal_consistency/metrics.py"
        )
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32)
        warp_mask = np.ones((1, 1), dtype=bool)
        transform = np.eye(4, dtype=np.float32)
        near_far = module.point2depth(points, warp_mask, transform)
        far_near = module.point2depth(points[::-1], warp_mask, transform)
        np.testing.assert_array_equal(near_far, np.array([[2.0]], dtype=np.float32))
        np.testing.assert_array_equal(far_near, np.array([[1.0]], dtype=np.float32))


class ScaleOnlyValidationTest(unittest.TestCase):
    def test_zero_prediction_is_rejected(self):
        pred = np.zeros((2, 2), dtype=np.float32)
        gt = np.ones((2, 2), dtype=np.float32)
        cases = [
            (
                "temporal",
                "src/eval/temporal_consistency/metrics.py",
                {"scale_only": True},
            ),
            ("monodepth", "src/eval/monodepth/tools.py", {"align_with_scale": True}),
            ("video", "src/eval/video_depth/tools.py", {"align_with_scale": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_scale_{name}", path)
                with self.assertRaisesRegex(ValueError, "scale-only alignment"):
                    module.depth_evaluation(pred, gt, max_depth=None, **alignment)

    def test_empty_support_returns_zero_valid_record(self):
        pred = np.ones((2, 2), dtype=np.float32)
        gt = np.ones((2, 2), dtype=np.float32)
        custom_mask = np.zeros((2, 2), dtype=bool)
        cases = [
            (
                "temporal",
                "src/eval/temporal_consistency/metrics.py",
                {"scale_only": True},
            ),
            ("monodepth", "src/eval/monodepth/tools.py", {"align_with_scale": True}),
            ("video", "src/eval/video_depth/tools.py", {"align_with_scale": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_empty_{name}", path)
                metrics, *_ = module.depth_evaluation(
                    pred,
                    gt,
                    max_depth=None,
                    custom_mask=custom_mask,
                    **alignment,
                )
                self.assertEqual(metrics["valid_pixels"], 0)


class AlignmentModeValidationTest(unittest.TestCase):
    def test_contradictory_modes_preserve_main_precedence_by_default(self):
        pred = np.ones((1, 2), dtype=np.float32)
        gt = pred * 2
        cases = [
            (
                "temporal",
                "src/eval/temporal_consistency/metrics.py",
                {"metric_scale": True, "scale_and_shift": True},
            ),
            (
                "monodepth",
                "src/eval/monodepth/tools.py",
                {"metric_scale": True, "align_with_lstsq": True},
            ),
            (
                "video",
                "src/eval/video_depth/tools.py",
                {"metric_scale": True, "align_with_lstsq": True},
            ),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_modes_{name}", path)
                try:
                    metrics, *_ = module.depth_evaluation(
                        pred, gt, max_depth=None, **alignment
                    )
                    abs_rel = metrics["Abs Rel"]
                except ValueError:
                    abs_rel = None
                self.assertEqual(abs_rel, 0.5)

    def test_contradictory_modes_can_be_rejected_explicitly(self):
        pred = np.ones((1, 2), dtype=np.float32)
        cases = [
            (
                "temporal",
                "src/eval/temporal_consistency/metrics.py",
                {"metric_scale": True, "scale_and_shift": True},
            ),
            (
                "monodepth",
                "src/eval/monodepth/tools.py",
                {"metric_scale": True, "align_with_lstsq": True},
            ),
            (
                "video",
                "src/eval/video_depth/tools.py",
                {"metric_scale": True, "align_with_lstsq": True},
            ),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_modes_strict_{name}", path)
                with self.assertRaisesRegex(ValueError, "one alignment mode"):
                    module.depth_evaluation(
                        pred,
                        pred * 2,
                        max_depth=None,
                        reject_contradictory_modes=True,
                        **alignment,
                    )


class PointCloudFinitenessTest(unittest.TestCase):
    def test_unsupported_confidence_threshold_is_not_advertised(self):
        tree = ast.parse((ROOT / "src/eval/mv_recon/launch.py").read_text())
        parser_names = {
            call.args[0].value
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        }
        self.assertNotIn("--conf_thresh", parser_names)


class RankLogAggregationTest(unittest.TestCase):
    def test_main_caps_rank_logs_and_stops_at_first_gap(self):
        class EmptyDataset:
            def __init__(self, **_kwargs):
                pass

            def __len__(self):
                return 0

        class FakeAccelerator:
            device = "cpu"
            is_main_process = True
            num_processes = 9
            process_index = 0

            def split_between_processes(self, indices):
                return mock.MagicMock(
                    __enter__=mock.Mock(return_value=indices),
                    __exit__=mock.Mock(return_value=False),
                )

            def wait_for_everyone(self):
                pass

        class FakeModel:
            def load_state_dict(self, _state_dict, strict=True):
                self.assert_strict = strict

            def eval(self):
                return self

            def to(self, _device):
                return self

        fake_modules = {
            "eval.mv_recon.data": types.SimpleNamespace(
                SevenScenes=EmptyDataset, NRGBD=EmptyDataset
            ),
            "eval.mv_recon.utils": types.SimpleNamespace(
                accuracy=mock.Mock(), completion=mock.Mock()
            ),
            "streamvggt.models.streamvggt": types.SimpleNamespace(StreamVGGT=FakeModel),
            "streamvggt.utils.pose_enc": types.SimpleNamespace(
                pose_encoding_to_extri_intri=mock.Mock()
            ),
            "streamvggt.utils.geometry": types.SimpleNamespace(
                unproject_depth_map_to_point_map=mock.Mock()
            ),
            "eval.mv_recon.criterion": types.SimpleNamespace(
                Regr3D_t_ScaleShiftInv=mock.Mock(return_value=mock.Mock()),
                L21=mock.Mock(),
            ),
            "dust3r.utils.geometry": types.SimpleNamespace(geotrf=mock.Mock()),
        }
        args = SimpleNamespace(
            weights="unused", output_dir=None, size=224, model_name="StreamVGGT"
        )
        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            gap_dir = pathlib.Path(output_dir, "7scenes")
            cap_dir = pathlib.Path(output_dir, "NRGBD")
            gap_dir.mkdir()
            cap_dir.mkdir()
            (gap_dir / "logs_0.txt").write_text("gap-rank-0\n")
            (gap_dir / "logs_2.txt").write_text("gap-rank-2\n")
            for rank in range(FakeAccelerator.num_processes):
                (cap_dir / f"logs_{rank}.txt").write_text(f"cap-rank-{rank}\n")

            module = load_module("area_c_rank_logs", "src/eval/mv_recon/launch.py")
            with (
                mock.patch.dict(sys.modules, fake_modules),
                mock.patch.object(module, "Accelerator", FakeAccelerator),
                mock.patch.object(module, "add_path_to_dust3r"),
                mock.patch.object(module.torch, "load", return_value={}),
            ):
                module.main(args)

            gap_result = (gap_dir / "logs_all.txt").read_text()
            cap_result = (cap_dir / "logs_all.txt").read_text()
            self.assertIn("gap-rank-0", gap_result)
            self.assertNotIn("gap-rank-2", gap_result)
            self.assertIn("cap-rank-7", cap_result)
            self.assertNotIn("cap-rank-8", cap_result)


class PointNormalizationTest(unittest.TestCase):
    def test_average_distance_uses_each_samples_valid_count(self):
        module = load_module("area_c_criterion", "src/eval/mv_recon/criterion.py")
        points = torch.tensor([[[[2.0, 0.0, 0.0]]], [[[4.0, 0.0, 0.0]]]])
        valid = torch.ones((2, 1, 1), dtype=torch.bool)
        factors = module.get_norm_factor(
            [points, None], "avg_dis", [valid, None], fix_first=True
        )
        torch.testing.assert_close(factors.reshape(-1), torch.tensor([2.0, 4.0]))


class Co3dCliTest(unittest.TestCase):
    def test_fast_eval_read_has_parser_definition(self):
        tree = ast.parse((ROOT / "src/eval/pose_evaluation/test_co3d.py").read_text())
        parser_names = {
            call.args[0].value.lstrip("-").replace("-", "_")
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        }
        self.assertIn("fast_eval", parser_names)

    def test_unimplemented_bundle_adjustment_fails_fast(self):
        tree = ast.parse((ROOT / "src/eval/pose_evaluation/test_co3d.py").read_text())
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        guards = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.use_ba"
        ]
        self.assertEqual(len(guards), 1)
        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(guards[0]))
        )


if __name__ == "__main__":
    unittest.main()
