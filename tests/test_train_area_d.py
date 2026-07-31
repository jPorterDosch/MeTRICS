#!/usr/bin/env python
"""CPU regression tests for Area D training and validation fixes."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import defaultdict

import torch
from accelerate import PartialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import finetune_depth as fd  # noqa: E402
import croco.utils.misc as misc  # noqa: E402
import train_utils  # noqa: E402

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

    def test_streaming_tae_does_not_bridge_invalid_frame(self) -> None:
        original_eval, original_tae = fd.depth_evaluation, fd.tae
        fd.depth_evaluation = lambda *args, **kwargs: (
            {"Abs Rel": 0.0, "delta < 1.25": 1.0, "RMSE": 0.0},
            None,
            None,
            None,
        )
        fd.tae = lambda *args: (2.0, 4.0)
        intrinsics = torch.eye(3).reshape(1, 3, 3)
        pose = torch.eye(4).reshape(1, 4, 4)
        views, preds = [], []
        for index, valid in enumerate((True, False, True), start=1):
            views.append(
                {
                    "depthmap": torch.ones(1, 1, 1),
                    "valid_mask": torch.tensor([[[valid]]]),
                    "camera_intrinsics": intrinsics,
                    "camera_pose": pose,
                    "dataset": ["hammer"],
                }
            )
            preds.append({"depth": torch.tensor([[[[float(index)]]]])})
        try:
            metrics = fd._streaming_depth_metrics(views, preds)
        finally:
            fd.depth_evaluation, fd.tae = original_eval, original_tae
        self.assertNotIn("hammer/tae", metrics)
        self.assertNotIn("hammer/tae_sq", metrics)

    def test_remote_nonfinite_loss_fails_every_rank(self) -> None:
        class RemoteFailureReducer(ReturningReducer):
            def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
                self.reduction = reduction
                return torch.zeros_like(tensor)

        accel = RemoteFailureReducer()
        with self.assertRaisesRegex(FloatingPointError, "another rank"):
            fd._check_finite_loss(1.0, {"loss": 1.0}, accel)
        self.assertEqual(accel.reduction, "min")

    def test_fresh_run_rejects_nonzero_start_epoch(self) -> None:
        cfg = fd.FinetuneDepthCfg(start_epoch=9)
        original = fd.build_manifest
        fd.build_manifest = lambda cfg: (_ for _ in ()).throw(
            AssertionError("manifest must not be built")
        )
        try:
            with self.assertRaisesRegex(ValueError, "start_epoch"):
                fd.main(cfg)
        finally:
            fd.build_manifest = original

    def test_checkpoint_commit_fsyncs_then_replaces(self) -> None:
        class Accel:
            is_main_process = True

            def wait_for_everyone(self) -> None:
                self.waited = True

        accel = Accel()
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "checkpoint-last.pth")
            temporary = os.path.join(directory, "checkpoint-.last.tmp-1.pth")
            with open(target, "wb") as handle:
                handle.write(b"old")
            with open(temporary, "wb") as handle:
                handle.write(b"new")
            original_fsync = fd.os.fsync
            calls = []
            fd.os.fsync = lambda descriptor: calls.append(descriptor)
            try:
                fd._commit_checkpoint(
                    directory, "last", ".last.tmp-1", accel
                )
            finally:
                fd.os.fsync = original_fsync
            with open(target, "rb") as handle:
                self.assertEqual(handle.read(), b"new")
            self.assertEqual(len(calls), 1)
            self.assertTrue(accel.waited)

    def test_fresh_run_atomically_claims_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = fd.FinetuneDepthCfg(save_dir=directory, exp_group="group")
            original = train_utils.is_rank_zero
            train_utils.is_rank_zero = lambda: True
            try:
                output_dir = train_utils.resolve_output_dir(cfg, "run-id")
                self.assertTrue(os.path.isdir(output_dir))
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    train_utils.resolve_output_dir(cfg, "run-id")
            finally:
                train_utils.is_rank_zero = original

    def test_empty_loader_is_rejected_before_progress_reporting(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation loader 0"):
            fd._validate_loader_lengths([object()], [[]], [[object()]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
