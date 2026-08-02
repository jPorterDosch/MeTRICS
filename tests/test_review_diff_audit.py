#!/usr/bin/env python
"""CPU regressions for the self-audit of the review diff."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_one_frame_timing_summary() -> None:
    from visualize_depth import _format_frame_timing

    assert _format_frame_timing([12.5]) == "frame0 12.5 ms"


if __name__ == "__main__":
    test_one_frame_timing_summary()
