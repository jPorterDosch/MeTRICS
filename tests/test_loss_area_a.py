import torch

from streamvggt.loss.depth_train_loss import DepthTrainLoss
from streamvggt.loss.distill_loss import DistillLoss
from streamvggt.loss.head_loss import DepthOrPmapLoss
from streamvggt.loss.trimmed_loss import TrimmedMAELoss
from streamvggt.loss.types import LossConfig


def test_all_invalid_targets_contribute_no_depth_or_point_loss() -> None:
    gts = [
        {
            "depthmap": torch.zeros(1, 2, 2),
            "valid_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        }
        for _ in range(2)
    ]
    preds = [
        {"depth": torch.ones(1, 2, 2, 1), "depth_conf": torch.ones(1, 2, 2)}
        for _ in range(2)
    ]
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


def test_trimmed_mae_uses_retained_counts_per_reduction() -> None:
    prediction = torch.stack((torch.ones(2, 5), torch.full((2, 5), 2.0)), dim=0)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)

    batch_loss = TrimmedMAELoss(trim=0.2, reduction="batch-based")(
        prediction[:1], target[:1], mask[:1]
    )
    image_loss = TrimmedMAELoss(trim=0.2, reduction="image-based")(
        prediction, target, mask
    )

    assert batch_loss.item() == 1.0
    assert image_loss.item() == 1.5


def test_spatial_gradient_uses_valid_edges_without_cropping() -> None:
    criterion = DepthOrPmapLoss(metric=True)
    for size, row in ((2, 0), (10, 0), (10, 9)):
        prediction = torch.zeros(1, size, size, 1)
        target = torch.zeros_like(prediction)
        prediction[0, row, 1, 0] = 1.0
        mask = torch.zeros(1, size, size, dtype=torch.bool)
        mask[0, row, :2] = True
        loss = criterion.image_gradient_loss(prediction, target, mask)
        assert loss.item() == 0.5


def test_loss_config_rejects_unknown_reduction() -> None:
    try:
        LossConfig(reduction="typo")
    except ValueError as error:
        assert "reduction" in str(error)
        assert "typo" in str(error)
    else:
        raise AssertionError("unknown reduction was accepted")


def test_confidence_disabled_depth_does_not_require_confidence_key() -> None:
    gts = [
        {
            "depthmap": torch.zeros(1, 2, 2),
            "valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        }
        for _ in range(2)
    ]
    preds = [{"depth": torch.ones(1, 2, 2, 1)} for _ in range(2)]
    loss, _ = DepthTrainLoss(metric=True, conf_weighting=False)(gts, preds)
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_all_invalid_targets_contribute_no_depth_or_point_loss()
    test_trimmed_mae_uses_retained_counts_per_reduction()
    test_spatial_gradient_uses_valid_edges_without_cropping()
    test_loss_config_rejects_unknown_reduction()
    test_confidence_disabled_depth_does_not_require_confidence_key()
