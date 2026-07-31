import torch

from streamvggt.loss.depth_train_loss import DepthTrainLoss
from streamvggt.loss.distill_loss import DistillLoss
from streamvggt.loss.head_loss import CameraLoss, DepthOrPmapLoss
from streamvggt.loss.l_loss import L21
from streamvggt.loss.regr_3d_pose import Regr3DPose
from streamvggt.loss.trimmed_loss import TrimmedMAELoss
from streamvggt.loss.types import LossConfig
from streamvggt.loss.utils import (
    closed_form_scale_and_shift,
    normalize_prediction_robust,
)


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


def test_temporal_loss_handles_single_frame_and_rejects_bad_config() -> None:
    prediction = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = LossConfig().build().temporal_loss(prediction, target, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert prediction.grad is not None

    for field, value in (
        ("temp_grad_scales", 0),
        ("depth_trim", 1.0),
        ("diff_depth_th", -0.1),
    ):
        try:
            LossConfig(**{field: value})
        except ValueError as error:
            assert field in str(error)
            assert str(value) in str(error)
        else:
            raise AssertionError(f"invalid {field} was accepted")


def test_camera_loss_rejects_non_finite_inputs() -> None:
    for name, prediction, target in (
        ("pred_pose", torch.full((1, 9), torch.nan), torch.zeros(1, 9)),
        ("gt_pose", torch.zeros(1, 9), torch.full((1, 9), torch.inf)),
    ):
        try:
            CameraLoss()(prediction, target)
        except ValueError as error:
            assert name in str(error)
        else:
            raise AssertionError(f"non-finite {name} was accepted")


def test_robust_normalization_median_uses_valid_values_only() -> None:
    target = torch.tensor([[[10.0, 10.0, 0.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, True, False], [False, False, False]]])
    normalized, (median, _) = normalize_prediction_robust(target, mask)
    assert median.item() == 10.0
    assert torch.equal(normalized[mask], torch.zeros(2))


def test_scale_shift_fit_is_per_sample_and_masked() -> None:
    prediction = torch.tensor([[[[0.0], [1.0], [100.0]]], [[[0.0], [1.0], [200.0]]]])
    target = torch.tensor([[[[0.0], [2.0], [-999.0]]], [[[0.0], [4.0], [999.0]]]])
    mask = torch.tensor([[[True, True, False]], [[True, True, False]]])
    scale, shift = closed_form_scale_and_shift(prediction, target, mask)
    assert torch.allclose(scale, torch.tensor([2.0, 4.0]))
    assert torch.allclose(shift, torch.tensor([0.0, 0.0]))


def test_perfect_point_map_has_zero_valid_normal_loss() -> None:
    y, x = torch.meshgrid(torch.arange(3.0), torch.arange(3.0), indexing="ij")
    point_map = torch.stack((x, y, torch.ones_like(x)), dim=-1).unsqueeze(0)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    loss = DepthOrPmapLoss().normal_loss(point_map, point_map, mask)
    assert loss.item() == 0.0


def test_scale_loss_calls_point_normalizer_with_matching_signature() -> None:
    class ProbeRegr3DPose(Regr3DPose):
        def get_norm_factor_point_cloud(
            self, pts_cross, valids, conf_cross, norm_self_only=False
        ):
            raise RuntimeError("normalizer reached")

    identity = torch.eye(4).unsqueeze(0)
    points = torch.ones(1, 1, 1, 3)
    gts = [
        {
            "camera_pose": identity,
            "pts3d": points,
            "valid_mask": torch.ones(1, 1, 1, dtype=torch.bool),
            "camera_only": torch.tensor([False]),
            "is_metric": torch.tensor([False]),
        }
    ]
    preds = [
        {
            "pts3d_in_self_view": points,
            "pts3d_in_other_view": points,
            "conf_self": torch.full((1, 1, 1), 2.0),
            "conf": torch.full((1, 1, 1), 2.0),
        }
    ]
    try:
        ProbeRegr3DPose(L21).get_all_pts3d_with_scale_loss(gts, preds)
    except RuntimeError as error:
        assert str(error) == "normalizer reached"
    else:
        raise AssertionError("point normalizer was not called")


if __name__ == "__main__":
    test_all_invalid_targets_contribute_no_depth_or_point_loss()
    test_trimmed_mae_uses_retained_counts_per_reduction()
    test_spatial_gradient_uses_valid_edges_without_cropping()
    test_loss_config_rejects_unknown_reduction()
    test_confidence_disabled_depth_does_not_require_confidence_key()
    test_temporal_loss_handles_single_frame_and_rejects_bad_config()
    test_camera_loss_rejects_non_finite_inputs()
    test_robust_normalization_median_uses_valid_values_only()
    test_scale_shift_fit_is_per_sample_and_masked()
    test_perfect_point_map_has_zero_valid_normal_loss()
    test_scale_loss_calls_point_normalizer_with_matching_signature()
