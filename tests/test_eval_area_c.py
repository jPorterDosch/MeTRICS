import ast
import importlib.util
import pathlib
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
