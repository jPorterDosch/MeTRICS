"""CPU regressions for empirical pixel-frequency sparse-depth simulation."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    import pytest
except ModuleNotFoundError:
    # Pytest is optional here, so the direct driver remains the canonical runner.
    class _DirectSkip(Exception):
        pass

    class _PytestCompat:
        @staticmethod
        def skip(reason: str) -> None:
            raise _DirectSkip(reason)

    pytest = _PytestCompat()
else:

    class _DirectSkip(Exception):
        pass


from streamvggt.depth_cond.config import (  # noqa: E402
    DepthCondCfg,
    MetricCfg,
    SparseSimMode,
    experiment_manifest,
)
from streamvggt.depth_cond.sparse import (  # noqa: E402
    load_freq_map,
    simulate_sparse_depth,
)


def _assert_value_error(fragment: str, callback) -> str:
    try:
        callback()
    except ValueError as error:
        message = str(error)
        assert fragment in message, message
        return message
    raise AssertionError(f"expected ValueError containing {fragment!r}")


def test_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "freq.npz"
        np.savez(path, freq=np.full((4, 5), 0.5, dtype=np.float32))
        depth = torch.tensor(
            [
                [[1.0, 2.0, 0.0, 4.0, 5.0]] * 4,
                [[2.0, -1.0, 3.0, 4.0, 5.0]] * 4,
            ]
        )
        views = [{"img": torch.zeros(2, 3, 4, 5), "depthmap": depth}]

        returned = simulate_sparse_depth(
            views, SparseSimMode.PIXEL_FREQ, 99, 0.5, str(path)
        )

    assert returned is views
    assert views[0]["sparse_depth"].shape == (2, 4, 5)
    assert views[0]["sparse_depth_mask"].shape == (2, 4, 5)
    assert views[0]["sparse_depth_mask"].dtype is torch.bool
    assert torch.equal(views[0]["sparse_depth"], depth * views[0]["sparse_depth_mask"])
    assert torch.all(views[0]["sparse_depth_mask"] <= (depth > 0))


def test_marginals() -> None:
    freq = np.empty((300, 1000), dtype=np.float32)
    freq[:100] = 0.0
    freq[100:200] = 1.0
    freq[200:] = 0.4
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "piecewise.npz"
        np.savez(path, freq=freq)
        views = [
            {
                "img": torch.zeros(1, 3, 300, 1000),
                "depthmap": torch.ones(1, 300, 1000),
            }
        ]
        torch.manual_seed(7)
        simulate_sparse_depth(
            views,
            SparseSimMode.PIXEL_FREQ,
            14,
            1.0 - float(freq.mean()),
            str(path),
        )

    mask = views[0]["sparse_depth_mask"]
    assert not mask[:, :100].any()
    assert mask[:, 100:200].all()
    assert abs(mask[:, 200:].float().mean().item() - 0.4) <= 0.03


def test_determinism() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "freq.npz"
        np.savez(path, freq=np.full((64, 64), 0.4, dtype=np.float32))

        masks = []
        for seed in (42, 42, 43):
            # Validation forks the RNG, so repeated seeded passes are isolated.
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                views = [
                    {
                        "img": torch.zeros(2, 3, 64, 64),
                        "depthmap": torch.ones(2, 64, 64),
                    }
                ]
                simulate_sparse_depth(
                    views, SparseSimMode.PIXEL_FREQ, 3, 0.6, str(path)
                )
                masks.append(views[0]["sparse_depth_mask"].clone())

    assert torch.equal(masks[0], masks[1])
    assert not torch.equal(masks[0], masks[2])


def test_orientation() -> None:
    freq = np.zeros((4, 8), dtype=np.float32)
    freq[:2] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bands.npz"
        np.savez(path, freq=freq)
        landscape = [{"img": torch.zeros(1, 3, 4, 8), "depthmap": torch.ones(1, 4, 8)}]
        portrait = [{"img": torch.zeros(1, 3, 8, 4), "depthmap": torch.ones(1, 8, 4)}]
        simulate_sparse_depth(landscape, SparseSimMode.PIXEL_FREQ, 14, 0.5, str(path))
        simulate_sparse_depth(portrait, SparseSimMode.PIXEL_FREQ, 14, 0.5, str(path))

    landscape_mask = landscape[0]["sparse_depth_mask"]
    portrait_mask = portrait[0]["sparse_depth_mask"]
    assert landscape_mask[:, :2].all() and not landscape_mask[:, 2:].any()
    assert not portrait_mask[:, :, :2].any() and portrait_mask[:, :, 2:].all()


def test_fail_fast() -> None:
    _assert_value_error(
        "--depth-cond.sim-freq-map-path",
        lambda: DepthCondCfg(
            sim_mode=SparseSimMode.PIXEL_FREQ, sim_freq_map_path=""
        ).validate(),
    )
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "missing.npz"
        _assert_value_error(
            "--depth-cond.sim-freq-map-path",
            lambda: DepthCondCfg(
                sim_mode=SparseSimMode.PIXEL_FREQ,
                sim_freq_map_path=str(missing),
            ).validate(),
        )

        density_path = Path(directory) / "density.npz"
        np.savez(density_path, freq=np.ones((4, 4), dtype=np.float32))
        message = _assert_value_error(
            "sim-mask-ratio", lambda: load_freq_map(str(density_path), 0.5)
        )
        assert "mean invalid fraction 0.000" in message

        invalid_path = Path(directory) / "invalid.npz"
        invalid = np.ones((4, 4), dtype=np.float32)
        invalid[0, 0] = 1.1
        np.savez(invalid_path, freq=invalid)
        _assert_value_error(
            "values must be in [0, 1]", lambda: load_freq_map(str(invalid_path), 0.0)
        )


def test_valid_mask_and() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "full.npz"
        np.savez(path, freq=np.ones((4, 4), dtype=np.float32))
        depth = torch.ones(1, 4, 4)
        depth[:, 2:, 2:] = 0.0
        valid_mask = torch.ones(1, 4, 4, dtype=torch.bool)
        valid_mask[:, :2, :2] = False
        views = [
            {
                "img": torch.zeros(1, 3, 4, 4),
                "depthmap": depth,
                "valid_mask": valid_mask,
            }
        ]
        simulate_sparse_depth(views, SparseSimMode.PIXEL_FREQ, 14, 0.0, str(path))

    mask = views[0]["sparse_depth_mask"]
    assert not mask[:, :2, :2].any()
    assert not mask[:, 2:, 2:].any()
    assert torch.equal(mask, valid_mask & (depth > 0))


def test_mae_withheld_nonempty() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "freq.npz"
        np.savez(path, freq=np.full((128, 128), 0.4, dtype=np.float32))
        depth = torch.ones(1, 128, 128)
        views = [{"img": torch.zeros(1, 3, 128, 128), "depthmap": depth}]
        torch.manual_seed(19)
        simulate_sparse_depth(views, SparseSimMode.PIXEL_FREQ, 14, 0.6, str(path))

    visible = views[0]["sparse_depth_mask"]
    valid = depth > 0
    assert visible.any()
    assert (valid & ~visible).any()


def test_config_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "freq.npz"
        np.savez(path, freq=np.full((2, 2), 0.4, dtype=np.float32))
        original = DepthCondCfg(
            sim_mode=SparseSimMode.PIXEL_FREQ,
            sim_mask_ratio=0.6,
            sim_freq_map_path=str(path),
        )
        original.validate()
        manifest = experiment_manifest(MetricCfg(depth_cond=original))
        encoded = json.dumps(manifest)
        assert "depth_cond.sim_freq_map_path" in json.loads(encoded)

        prefix = "depth_cond."
        primitives = {
            key[len(prefix) :]: value
            for key, value in manifest.items()
            if key.startswith(prefix)
        }
        rebuilt = DepthCondCfg(**primitives)
        rebuilt.validate()
        assert rebuilt == original

        old_primitives = dict(primitives)
        old_primitives.pop("sim_freq_map_path")
        old_checkpoint_cfg = DepthCondCfg(**old_primitives)
        assert old_checkpoint_cfg.sim_freq_map_path == ""


def test_real_artifact() -> None:
    path = Path(ROOT) / "assets" / "spot" / "valid_freq_640x480.npz"
    if not path.is_file():
        pytest.skip(f"real SPOT artifact is absent: {path}")
    with np.load(path) as artifact:
        freq = np.asarray(artifact["freq"])
    assert freq.shape == (480, 640)
    assert np.isfinite(freq).all()
    assert ((freq >= 0.0) & (freq <= 1.0)).all()
    assert 0.25 <= float(freq.mean()) <= 0.55


if __name__ == "__main__":
    tests = [
        test_contract,
        test_marginals,
        test_determinism,
        test_orientation,
        test_fail_fast,
        test_valid_mask_and,
        test_mae_withheld_nonempty,
        test_config_roundtrip,
        test_real_artifact,
    ]
    for test in tests:
        try:
            test()
        except _DirectSkip as error:
            print(f"SKIP {test.__name__}: {error}")
        else:
            print(f"PASS {test.__name__}")
