import ast
import importlib.util
import pathlib
import unittest

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
