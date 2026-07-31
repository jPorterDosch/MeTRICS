#!/usr/bin/env python
"""CPU regression tests for Area D training and validation fixes."""

from __future__ import annotations

import os
import sys
import unittest
from collections import defaultdict

import torch
from accelerate import PartialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import finetune_depth as fd  # noqa: E402
import croco.utils.misc as misc  # noqa: E402

PartialState()


class ReturningReducer:
    num_processes = 2
    device = torch.device("cpu")

    def wait_for_everyone(self) -> None:
        pass

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        return torch.tensor([[13.0, 4.0]], dtype=tensor.dtype)


class TrainAreaDTests(unittest.TestCase):
    def test_reduce_metrics_uses_returned_tensor(self) -> None:
        original = fd.gather_object
        fd.gather_object = lambda parts: [parts[0], parts[0]]
        try:
            per_dataset, blended = fd._reduce_metrics(
                {"hammer/loss": 10.0}, {"hammer/loss": 1}, ReturningReducer()
            )
        finally:
            fd.gather_object = original
        self.assertEqual(per_dataset, {"hammer": {"loss": 3.25}})
        self.assertEqual(blended, {"loss": 3.25})

    def test_validation_loss_is_clip_weighted(self) -> None:
        sums, counts = defaultdict(float), defaultdict(int)

        def views(batch_size: int) -> list[dict]:
            return [
                {
                    "img": torch.zeros(batch_size, 3, 1, 1),
                    "dataset": ["hammer"] * batch_size,
                }
            ]

        fd._accumulate_batch_loss(views(4), 1.0, {}, sums, counts)
        fd._accumulate_batch_loss(views(1), 9.0, {}, sums, counts)
        self.assertEqual(counts["hammer/loss"], 5)
        self.assertAlmostEqual(sums["hammer/loss"] / counts["hammer/loss"], 2.6)

    def test_log_val_stats_gathers_global_median(self) -> None:
        logger = misc.MetricLogger()
        logger.update(loss=0.0)
        logger.update(loss=0.0)

        class Accel(ReturningReducer):
            trackers = []

            def log(self, values: dict, step: int) -> None:
                self.logged = values

            def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
                return tensor

        accel = Accel()
        original = fd.gather_object
        fd.gather_object = lambda value: [[0.0, 0.0], [100.0, 100.0, 100.0]]
        try:
            results = fd._log_val_stats(logger, {}, {}, accel, "val", 1)
        finally:
            fd.gather_object = original
        self.assertEqual(results["loss_med"], 100.0)
        self.assertIn("val/all/loss_med", accel.logged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
