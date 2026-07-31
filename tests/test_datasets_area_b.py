from pathlib import Path

import numpy as np
from PIL import Image

from streamvggt.datasets.arkitscenes import ARKitScenes_Multi
from streamvggt.datasets.arkitscenes_highres import ARKitScenesHighRes_Multi
from streamvggt.datasets.base.base_multiview_dataset import (
    BaseMultiViewDataset,
    get_ray_map,
)
from streamvggt.datasets.base.easy_dataset import EasyDataset
from streamvggt.datasets.config import DatasetConfig
from streamvggt.datasets.hammer import HAMMER_Multi
from streamvggt.datasets.hypersim import HyperSim_Multi
from streamvggt.datasets.scannet import ScanNet_Multi
from streamvggt.datasets.types import DatasetName
from streamvggt.datasets.utils.corr import extract_correspondences_from_pts3d
from streamvggt.datasets.utils.cropping import rescale_image_depthmap


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


def test_invalid_depth_pixels_are_not_positive_correspondences() -> None:
    view = {
        "pts3d": np.zeros((2, 2, 3)),
        "camera_intrinsics": np.eye(3),
        "camera_pose": np.eye(4),
        "valid_mask": np.zeros((2, 2), dtype=bool),
    }
    _, _, valid = extract_correspondences_from_pts3d(
        view, view, 1, np.random.default_rng(0)
    )
    assert not valid.any()


def test_resize_intrinsics_use_floored_raster_scales() -> None:
    image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    depth = np.zeros((480, 640), dtype=np.float32)
    intrinsics = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]])
    output, _, scaled = rescale_image_depthmap(
        image, depth, intrinsics, np.array((518, 392))
    )
    np.testing.assert_allclose(scaled[0, 0], 500.0 * output.width / 640)
    np.testing.assert_allclose(scaled[1, 1], 500.0 * output.height / 480)


def test_variable_length_sampler_rejects_fewer_than_four_views() -> None:
    class TinyDataset(EasyDataset):
        _resolutions = [(1, 1)]
        num_views = 3

        def __len__(self):
            return 3

    try:
        TinyDataset().make_sampler(1, fixed_length=False)
    except ValueError as error:
        assert "num_views" in str(error)
        assert "3" in str(error)
    else:
        raise AssertionError("variable-length sampler accepted num_views=3")


def test_negative_correspondence_settings_are_rejected() -> None:
    config = DatasetConfig(
        root=Path("."),
        dataset=DatasetName.HAMMER,
        num_views=1,
        stride_range=(1, 1),
        resolution=((1, 1),),
        n_corres=-1,
    )
    try:
        config.validate()
    except ValueError as error:
        assert "n_corres" in str(error)
    else:
        raise AssertionError("negative n_corres was accepted")

    class SequenceDataset(BaseMultiViewDataset):
        pass

    try:
        SequenceDataset(num_views=1, resolution=(1, 1), nneg=-1)
    except ValueError as error:
        assert "nneg" in str(error)
    else:
        raise AssertionError("direct constructor accepted negative nneg")


if __name__ == "__main__":
    test_ray_directions_ignore_camera_translation()
    test_irregular_stride_respects_effective_gap_range()
    test_metric_loaders_reject_nonmetric_label()
    test_nneg_is_an_absolute_count()
    test_invalid_depth_pixels_are_not_positive_correspondences()
    test_resize_intrinsics_use_floored_raster_scales()
    test_variable_length_sampler_rejects_fewer_than_four_views()
    test_negative_correspondence_settings_are_rejected()
