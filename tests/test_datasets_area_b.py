import numpy as np

from streamvggt.datasets.base.base_multiview_dataset import (
    BaseMultiViewDataset,
    get_ray_map,
)


def test_ray_directions_ignore_camera_translation() -> None:
    c2w = np.eye(4)
    c2w[0, 3] = 1.0
    ray = get_ray_map(np.eye(4), c2w, np.eye(3), 1, 1)[0, 0]
    np.testing.assert_allclose(ray[:3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ray[3:], [0.0, 0.0, 1.0])


def test_irregular_stride_respects_effective_gap_range() -> None:
    class SequenceDataset(BaseMultiViewDataset):
        def __len__(self):
            return 1

    dataset = object.__new__(SequenceDataset)
    dataset.stride_range = (2, 4)
    dataset.regular_stride = False
    positions, _ = dataset.get_seq_from_start_id(
        4, 0, list(range(7)), np.random.default_rng(1)
    )
    gaps = np.diff(positions)
    assert np.all((2 <= gaps) & (gaps <= 4)), (positions, gaps)


if __name__ == "__main__":
    test_ray_directions_ignore_camera_translation()
    test_irregular_stride_respects_effective_gap_range()
