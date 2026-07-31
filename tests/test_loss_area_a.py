import torch

from streamvggt.loss.depth_train_loss import DepthTrainLoss
from streamvggt.loss.distill_loss import DistillLoss


def test_all_invalid_targets_contribute_no_depth_or_point_loss() -> None:
    gts = [
        {
            "depthmap": torch.zeros(1, 2, 2),
            "valid_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        }
        for _ in range(2)
    ]
    preds = [{"depth": torch.ones(1, 2, 2, 1), "depth_conf": torch.ones(1, 2, 2)} for _ in range(2)]
    _, details = DepthTrainLoss(metric=True)(gts, preds)
    assert details["Ldepth"].item() == 0.0

    distill_gts = [
        {
            "camera_pose": torch.zeros(1, 9),
            "depth": torch.zeros(1, 2, 2, 1),
            "depth_conf": torch.ones(1, 2, 2),
            "pts3d_in_other_view": torch.zeros(1, 2, 2, 3),
            "conf": torch.ones(1, 2, 2),
            "valid_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        }
    ]
    distill_preds = [
        {
            "camera_pose": torch.zeros(1, 9),
            "depth": torch.ones(1, 2, 2, 1),
            "depth_conf": torch.ones(1, 2, 2),
            "pts3d_in_other_view": torch.ones(1, 2, 2, 3),
            "conf": torch.ones(1, 2, 2),
        }
    ]
    _, details = DistillLoss()(distill_gts, distill_preds)
    assert details["Ldepth"] == 0.0
    assert details["Lpmap"] == 0.0


if __name__ == "__main__":
    test_all_invalid_targets_contribute_no_depth_or_point_loss()
