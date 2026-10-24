"""ScanNet_Multi's official_split: which scenes each split reads."""

import tempfile
from pathlib import Path

import numpy as np

from streamvggt.datasets import scannet as scannet_module
from streamvggt.datasets.config import MultiDatasetConfig
from streamvggt.datasets.scannet import ScanNet_Multi, official_scenes
from streamvggt.datasets.types import DatasetName, Split

REPO = Path(__file__).resolve().parents[1]


def _fake_tree(root: Path) -> Path:
    """scans_train with listed train/val scenes plus one unlisted scene,
    scans_test with one scene, and official lists naming them."""
    layout = {
        "scans_train": ["scene0000_00", "scene0001_00", "scene0002_00", "scene0003_00"],
        "scans_test": ["scene0707_00"],
    }
    for subdir, scenes in layout.items():
        for scene in scenes:
            d = root / subdir / scene
            d.mkdir(parents=True)
            np.savez(
                d / "new_scene_metadata.npz",
                images=np.array([f"{i:05d}" for i in range(8)]),
            )
    lists = root / "splits"
    lists.mkdir()
    # unsorted on purpose: the loader orders scenes itself
    (lists / "scannetv2_train.txt").write_text("scene0001_00\nscene0000_00\n")
    (lists / "scannetv2_val.txt").write_text("scene0002_00\n")
    return lists


def _scenes(root: Path, split: Split, official_split) -> list[str]:
    return ScanNet_Multi(
        ROOT=str(root),
        split=split,
        num_views=4,
        resolution=(56, 42),
        stride_range=(1, 1),
        official_split=official_split,
    ).scenes


def _with_lists(fn) -> None:
    saved = scannet_module.OFFICIAL_SPLIT_DIR
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scannet_module.OFFICIAL_SPLIT_DIR = str(_fake_tree(root))
        try:
            fn(root)
        finally:
            scannet_module.OFFICIAL_SPLIT_DIR = saved


def test_official_split_reads_only_listed_scenes() -> None:
    def check(root: Path) -> None:
        assert _scenes(root, Split.TRAIN, "train") == ["scene0000_00", "scene0001_00"]
        # val is read from scans_train even with the TEST sampling
        assert _scenes(root, Split.TEST, "val") == ["scene0002_00"]

    _with_lists(check)


def test_no_official_split_keeps_original_layout() -> None:
    def check(root: Path) -> None:
        assert sorted(_scenes(root, Split.TRAIN, None)) == [
            "scene0000_00",
            "scene0001_00",
            "scene0002_00",
            "scene0003_00",
        ]
        assert _scenes(root, Split.TEST, None) == ["scene0707_00"]

    _with_lists(check)


def test_missing_listed_scene_fails() -> None:
    def check(root: Path) -> None:
        (Path(scannet_module.OFFICIAL_SPLIT_DIR) / "scannetv2_val.txt").write_text(
            "scene0002_00\nscene0999_00\n"
        )
        try:
            _scenes(root, Split.TEST, "val")
        except FileNotFoundError as error:
            assert "scene0999_00" in str(error)
        else:
            raise AssertionError("a listed scene missing on disk was skipped")

    _with_lists(check)


def test_unknown_official_split_rejected() -> None:
    for bad in ("test", "Train"):
        try:
            ScanNet_Multi(
                ROOT="unused", num_views=4, resolution=(56, 42), official_split=bad
            )
        except ValueError as error:
            assert "official_split" in str(error)
        else:
            raise AssertionError(f"official_split={bad!r} was accepted")


def test_multi_config_passes_official_split_through() -> None:
    def check(root: Path) -> None:
        mc = MultiDatasetConfig(
            root=(root,),
            dataset=(DatasetName.SCANNET,),
            stride_range=((1, 1),),
            num_views=4,
            resolution=((56, 42),),
            split=Split.TEST,
            scannet_official_split="val",
        )
        (config,) = mc.to_dataset_configs()
        assert config.scannet_official_split == "val"
        assert config.build().scenes == ["scene0002_00"]

    _with_lists(check)


def test_vendored_lists_are_disjoint_and_complete() -> None:
    train, val = official_scenes("train"), official_scenes("val")
    assert (len(train), len(val)) == (1201, 312)
    assert len(set(train)) == len(train) and len(set(val)) == len(val)
    assert not set(train) & set(val)
    test = (REPO / "assets" / "scannet_splits" / "scannetv2_test.txt").read_text()
    assert not set(test.split()) & (set(train) | set(val))
