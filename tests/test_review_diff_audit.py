#!/usr/bin/env python
"""CPU regressions for the self-audit of the review diff."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_spot_geometry_protocol_is_preserved() -> None:
    source = (ROOT / "experiments/eval_all.sh").read_text()
    active = next(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("local geom=(")
    )
    assert "--landscape-crop --crop-anchor top" in active


def test_one_frame_timing_summary() -> None:
    from visualize_depth import _format_frame_timing

    assert _format_frame_timing([12.5]) == "frame0 12.5 ms"


def test_cuda_timing_is_selected_from_input_device() -> None:
    for relative in (
        "src/streamvggt/models/streamvggt.py",
        "src/streamvggt/depth_cond/model.py",
    ):
        source = (ROOT / relative).read_text()
        assert 'cuda_timing = timing and timing_device.type == "cuda"' in source


if __name__ == "__main__":
    test_spot_geometry_protocol_is_preserved()
    test_one_frame_timing_summary()
    test_cuda_timing_is_selected_from_input_device()
