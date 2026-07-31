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


if __name__ == "__main__":
    unittest.main()
