import ast
import importlib.util
import pathlib
import textwrap
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
    def test_custom_mask_excludes_affine_fit_outlier(self):
        pred = np.array([[1.0, 2.0, 100.0]], dtype=np.float32)
        gt = np.array([[2.0, 4.0, 1.0]], dtype=np.float32)
        custom_mask = np.array([[True, True, False]])
        cases = [
            ("temporal", "src/eval/temporal_consistency/metrics.py", {"scale_and_shift": True}),
            ("monodepth", "src/eval/monodepth/tools.py", {"align_with_lstsq": True}),
            ("video", "src/eval/video_depth/tools.py", {"align_with_lstsq": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_{name}", path)
                metrics, _, aligned, _ = module.depth_evaluation(
                    pred, gt, max_depth=None, custom_mask=custom_mask, **alignment
                )
                self.assertAlmostEqual(metrics["Abs Rel"], 0.0, places=6)
                np.testing.assert_allclose(
                    aligned.cpu().numpy()[custom_mask], np.array([2.0, 4.0]), atol=1e-5
                )


class VideoAffineRouteTest(unittest.TestCase):
    def test_scale_and_shift_routes_to_exact_affine_solver(self):
        source = (ROOT / "src/eval/video_depth/eval_depth.py").read_text()
        tree = ast.parse(source)
        affine_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "args.align == 'scale&shift'" not in ast.unparse(node.test):
                continue
            affine_calls.extend(
                call
                for call in ast.walk(node.body[0])
                if isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "depth_evaluation"
            )
        self.assertEqual(len(affine_calls), 3)
        for call in affine_calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertTrue(ast.literal_eval(keywords["align_with_lstsq"]))


class ExactDeltaThresholdTest(unittest.TestCase):
    def test_perfect_depth_satisfies_delta_one(self):
        depth = np.ones((1, 2), dtype=np.float32)
        cases = [
            ("temporal", "src/eval/temporal_consistency/metrics.py", {"metric_scale": True}),
            ("monodepth", "src/eval/monodepth/tools.py", {"metric_scale": True}),
            ("video", "src/eval/video_depth/tools.py", {"metric_scale": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_delta_{name}", path)
                metrics, *_ = module.depth_evaluation(
                    depth, depth, max_depth=None, **alignment
                )
                self.assertEqual(metrics["delta < 1."], 1.0)


class ReprojectionOcclusionTest(unittest.TestCase):
    def test_nearest_depth_wins_independent_of_source_order(self):
        module = load_module(
            "area_c_reprojection", "src/eval/temporal_consistency/metrics.py"
        )
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32)
        warp_mask = np.ones((1, 1), dtype=bool)
        transform = np.eye(4, dtype=np.float32)
        near_far = module.point2depth(points, warp_mask, transform)
        far_near = module.point2depth(points[::-1], warp_mask, transform)
        np.testing.assert_array_equal(near_far, np.array([[1.0]], dtype=np.float32))
        np.testing.assert_array_equal(far_near, near_far)


class ScaleOnlyValidationTest(unittest.TestCase):
    def test_zero_prediction_is_rejected(self):
        pred = np.zeros((2, 2), dtype=np.float32)
        gt = np.ones((2, 2), dtype=np.float32)
        cases = [
            ("temporal", "src/eval/temporal_consistency/metrics.py", {"scale_only": True}),
            ("monodepth", "src/eval/monodepth/tools.py", {"align_with_scale": True}),
            ("video", "src/eval/video_depth/tools.py", {"align_with_scale": True}),
        ]
        for name, path, alignment in cases:
            with self.subTest(name=name):
                module = load_module(f"area_c_scale_{name}", path)
                with self.assertRaisesRegex(ValueError, "scale-only alignment"):
                    module.depth_evaluation(pred, gt, max_depth=None, **alignment)


class AlignmentModeValidationTest(unittest.TestCase):
    def test_contradictory_modes_are_rejected(self):
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
                with self.assertRaisesRegex(ValueError, "one alignment mode"):
                    module.depth_evaluation(pred, gt, max_depth=None, **alignment)


class PointCloudFinitenessTest(unittest.TestCase):
    def test_joint_point_mask_preserves_tuples_and_colors(self):
        source = (ROOT / "src/eval/mv_recon/launch.py").read_text()
        block = source.split("                    mask = np.isfinite(pts_all_masked)", 1)[1]
        block = "                    mask = np.isfinite(pts_all_masked)" + block.split(
            "                    if args.use_proj:", 1
        )[0]
        namespace = {
            "np": np,
            "pts_all_masked": np.array([[0.0, 0.0, 1.0], [1.0, np.nan, 2.0]]),
            "pts_gt_all_masked": np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]),
            "images_all_masked": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        }
        exec(textwrap.dedent(block), namespace)
        self.assertEqual(namespace["pts_all_masked"].shape, (1, 3))
        self.assertEqual(namespace["pts_gt_all_masked"].shape, (1, 3))
        self.assertEqual(namespace["images_all_masked"].shape, (1, 3))


class PointNormalizationTest(unittest.TestCase):
    def test_average_distance_uses_each_samples_valid_count(self):
        module = load_module("area_c_criterion", "src/eval/mv_recon/criterion.py")
        points = torch.tensor([[[[2.0, 0.0, 0.0]]], [[[4.0, 0.0, 0.0]]]])
        valid = torch.ones((2, 1, 1), dtype=torch.bool)
        factors = module.get_norm_factor(
            [points, None], "avg_dis", [valid, None], fix_first=True
        )
        torch.testing.assert_close(factors.reshape(-1), torch.tensor([2.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
