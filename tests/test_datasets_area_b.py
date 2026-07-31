import numpy as np

from streamvggt.datasets.base.base_multiview_dataset import get_ray_map


def test_ray_directions_ignore_camera_translation() -> None:
    c2w = np.eye(4)
    c2w[0, 3] = 1.0
    ray = get_ray_map(np.eye(4), c2w, np.eye(3), 1, 1)[0, 0]
    np.testing.assert_allclose(ray[:3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ray[3:], [0.0, 0.0, 1.0])


if __name__ == "__main__":
    test_ray_directions_ignore_camera_translation()
