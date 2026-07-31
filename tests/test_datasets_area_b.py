import numpy as np
from pathlib import Path

from streamvggt.datasets.arkitscenes import ARKitScenes_Multi
from streamvggt.datasets.arkitscenes_highres import ARKitScenesHighRes_Multi
from streamvggt.datasets.base.base_multiview_dataset import (
    BaseMultiViewDataset,
    get_ray_map,
)
from streamvggt.datasets.config import DatasetConfig
from streamvggt.datasets.hammer import HAMMER_Multi
from streamvggt.datasets.hypersim import HyperSim_Multi
from streamvggt.datasets.scannet import ScanNet_Multi
from streamvggt.datasets.types import DatasetName
from streamvggt.datasets.utils.corr import extract_correspondences_from_pts3d


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


def test_metric_loaders_reject_nonmetric_label() -> None:
    config = DatasetConfig(
        root=Path("."),
        dataset=DatasetName.HAMMER,
        num_views=1,
        stride_range=(1, 1),
        resolution=((1, 1),),
        is_metric=False,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "is_metric" in str(error)
    else:
        raise AssertionError("metric data was accepted with is_metric=False")

    for loader in (
        HAMMER_Multi,
        ARKitScenes_Multi,
        ARKitScenesHighRes_Multi,
        ScanNet_Multi,
        HyperSim_Multi,
    ):
        try:
            loader(ROOT="unused", num_views=1, resolution=(1, 1), is_metric=False)
        except ValueError as error:
            assert "is_metric" in str(error), (loader.__name__, error)
        else:
            raise AssertionError(f"{loader.__name__} accepted is_metric=False")


def test_nneg_is_an_absolute_count() -> None:
    y, x = np.mgrid[:3, :3]
    points = np.stack((x, y, np.ones_like(x)), axis=-1).astype(float)
    points.reshape(-1, 3)[5:] = [0.0, 0.0, 1.0]
    view = {
        "pts3d": points,
        "camera_intrinsics": np.eye(3),
        "camera_pose": np.eye(4),
        "valid_mask": np.ones((3, 3), dtype=bool),
    }
    _, _, valid = extract_correspondences_from_pts3d(
        view, view, 4, np.random.default_rng(0), nneg=1
    )
    assert valid.sum() == 3

    config = DatasetConfig(
        root=Path("."),
        dataset=DatasetName.HAMMER,
        num_views=1,
        stride_range=(1, 1),
        resolution=((1, 1),),
        n_corres=2,
        nneg=3,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "nneg" in str(error)
    else:
        raise AssertionError("nneg larger than n_corres was accepted")


if __name__ == "__main__":
    test_ray_directions_ignore_camera_translation()
    test_irregular_stride_respects_effective_gap_range()
    test_metric_loaders_reject_nonmetric_label()
    test_nneg_is_an_absolute_count()
