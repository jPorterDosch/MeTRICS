#!/usr/bin/env python
"""CPU regression tests for Area D training and validation fixes."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import finetune_depth as fd  # noqa: E402


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
