#!/usr/bin/env python
"""CPU regressions for the self-audit of the review diff."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).parents[1]


def test_spot_geometry_protocol_is_preserved() -> None:
    source = (ROOT / "experiments/eval_all.sh").read_text()
    active = next(
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("local geom=(")
    )
    assert "--landscape-crop --crop-anchor top" in active


if __name__ == "__main__":
    test_spot_geometry_protocol_is_preserved()
