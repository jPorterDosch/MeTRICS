"""CPU tests for the offline SPOT validity-frequency map builder."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from build_spot_freq_map import (  # noqa: E402
    N_PIXELS,
    RAW_H,
    RAW_W,
    compute_freq,
    radial_autocorr,
    read_spot_depth,
    save_artifact,
)


def _write_depth(path: Path, depth: np.ndarray, count: int = N_PIXELS) -> None:
    with open(path, "wb") as handle:
        np.asarray(count, dtype=np.int32).tofile(handle)
        np.asarray(depth, dtype=np.float32).tofile(handle)


def test_reader_roundtrip_and_wrong_header() -> None:
    depth = np.arange(N_PIXELS, dtype=np.float32).reshape(RAW_H, RAW_W)
    with tempfile.TemporaryDirectory() as directory:
        good = Path(directory) / "0"
        bad = Path(directory) / "1"
        _write_depth(good, depth)
        _write_depth(bad, depth, count=N_PIXELS - 1)
        assert np.array_equal(read_spot_depth(good), depth)
        try:
            read_spot_depth(bad)
        except ValueError as error:
            assert "expected 307200" in str(error)
        else:
            raise AssertionError("wrong header did not raise ValueError")


def test_compute_freq_exact() -> None:
    half = np.zeros((RAW_H, RAW_W), dtype=np.float32)
    half[:, : RAW_W // 2] = 2.0
    full = np.ones((RAW_H, RAW_W), dtype=np.float32)
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / "0", Path(directory) / "1"]
        _write_depth(paths[0], half)
        _write_depth(paths[1], full)
        freq, densities = compute_freq(paths)
    assert np.all(freq[:, : RAW_W // 2] == 1.0)
    assert np.all(freq[:, RAW_W // 2 :] == 0.5)
    assert np.array_equal(densities, np.array([0.5, 1.0]))


def test_artifact_roundtrip() -> None:
    freq = np.linspace(0, 1, N_PIXELS, dtype=np.float32).reshape(RAW_H, RAW_W)
    meta = {"mean_valid": float(freq.mean()), "seqs": ["synthetic"]}
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "map.npz"
        save_artifact(out, freq, meta)
        with np.load(out) as artifact:
            assert set(artifact.files) == {"freq", "meta"}
            loaded = artifact["freq"]
            loaded_meta = json.loads(str(artifact["meta"]))
    assert loaded.dtype == np.float32
    assert loaded.shape == (RAW_H, RAW_W)
    assert 0 <= loaded.min() and loaded.max() <= 1
    assert loaded_meta["mean_valid"] == float(loaded.mean())


def test_radial_autocorr_blocky_exceeds_iid() -> None:
    rng = np.random.default_rng(7)
    iid = rng.random((24, 128, 128)) < 0.4
    coarse = rng.random((24, 16, 16)) < 0.4
    blocky = np.repeat(np.repeat(coarse, 8, axis=1), 8, axis=2)
    _, iid_profile, iid_ell = radial_autocorr(iid)
    _, block_profile, block_ell = radial_autocorr(blocky)
    assert np.isclose(iid_profile[0], 1.0)
    assert np.isclose(block_profile[0], 1.0)
    assert block_ell > iid_ell


if __name__ == "__main__":
    tests = [
        test_reader_roundtrip_and_wrong_header,
        test_compute_freq_exact,
        test_artifact_roundtrip,
        test_radial_autocorr_blocky_exceeds_iid,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
