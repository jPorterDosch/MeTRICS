#!/usr/bin/env python
"""CPU regression for timed/untimed visualizer inference equivalence."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dust3r.inference import loss_of_one_batch  # noqa: E402
from visualize_depth import _run_streaming_inference  # noqa: E402


class _DeterministicStreamingModel:
    """Small CPU stub for the GPU-scale model; both wrappers remain real."""

    def __init__(self) -> None:
        self.call_frame_times = []

    def inference(self, views, query_points, frame_times_ms=None):
        self.call_frame_times.append(frame_times_ms)
        query_total = query_points.float().sum()
        predictions = [
            {"depth": view["img"].float() + query_total + frame_index}
            for frame_index, view in enumerate(views)
        ]
        if frame_times_ms is not None:
            frame_times_ms.extend(0.0 for _ in views)
        return SimpleNamespace(views=views, ress=predictions)


def test_timing_flag_preserves_streaming_predictions() -> None:
    views = [
        {
            "img": torch.full((1, 3, 8, 8), float(frame_index)),
            "valid_mask": torch.ones((1, 8, 8), dtype=torch.bool),
        }
        for frame_index in range(2)
    ]
    model = _DeterministicStreamingModel()

    torch.manual_seed(1234)
    with patch("torch.cuda.get_device_capability", return_value=(0, 0)):
        untimed = loss_of_one_batch(
            views,
            model,
            None,
            accelerator=None,
            inference=True,
            symmetrize_batch=False,
            use_amp=True,
        )

    frame_times_ms = []
    torch.manual_seed(1234)
    timed = _run_streaming_inference(model, views, frame_times_ms)

    assert untimed["views"] == timed["views"] == views
    assert model.call_frame_times == [None, frame_times_ms]
    assert len(frame_times_ms) == len(views)
    assert len(untimed["pred"]) == len(timed["pred"])
    for untimed_pred, timed_pred in zip(untimed["pred"], timed["pred"]):
        assert torch.equal(untimed_pred["depth"], timed_pred["depth"])


if __name__ == "__main__":
    test_timing_flag_preserves_streaming_predictions()
