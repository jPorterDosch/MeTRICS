import torch

from .trimmed_loss import TrimmedMAELoss
from .utils import gradient_loss, reduction_batch_based, reduction_image_based


class GradientLoss(torch.nn.Module):
    def __init__(self, scales: int = 4, reduction: str = "batch-based") -> None:
        super().__init__()

        if reduction not in {"batch-based", "image-based"}:
            raise ValueError(f"invalid reduction {reduction!r}")
        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        num_frame_h: int = 1,
    ) -> torch.Tensor:
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i * frame_h : (i + 1) * frame_h, :] = i + 1

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step]
                if num_frame_h > 1
                else None,
            )

        return total


class TemporalGradientMatchingLoss(torch.nn.Module):
    def __init__(
        self,
        trim: float = 0.2,
        temp_grad_scales: int = 4,
        temp_grad_decay: float = 0.5,
        reduction: str = "batch-based",
        diff_depth_th: float = 0.05,
    ) -> None:
        super().__init__()

        if temp_grad_scales < 1:
            raise ValueError(f"invalid temp_grad_scales {temp_grad_scales!r}")
        if diff_depth_th < 0:
            raise ValueError(f"invalid diff_depth_th {diff_depth_th!r}")

        self.data_loss = TrimmedMAELoss(trim=trim, reduction=reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        prediction: Shape(B, T, H, W)
        target: Shape(B, T, H, W)
        mask: Shape(B, T, H, W)
        """
        total = 0
        cnt = 0

        min_target = (
            torch.where(mask.bool(), target, torch.inf).min(-1).values.min(-1).values
        )
        max_target = (
            torch.where(mask.bool(), target, -torch.inf).max(-1).values.max(-1).values
        )
        target_th = (max_target - min_target) * self.diff_depth_th

        for scale in range(self.temp_grad_scales):
            temp_stride = pow(2, scale)
            if temp_stride < prediction.shape[1]:
                pred_temp_grad = torch.diff(prediction[:, ::temp_stride, ...], dim=1)
                target_temp_grad = torch.diff(target[:, ::temp_stride, ...], dim=1)
                temp_mask = (
                    mask[:, ::temp_stride, ...][:, 1:, ...]
                    & mask[:, ::temp_stride, ...][:, :-1, ...]
                )

                valid_mask_from_target_th = (
                    target_temp_grad.abs()
                    < target_th.unsqueeze(-1).unsqueeze(-1)[:, ::temp_stride, ...][
                        :, 1:, ...
                    ]
                )
                temp_mask = temp_mask & valid_mask_from_target_th

                total += self.data_loss(
                    prediction=pred_temp_grad.flatten(0, 1),
                    target=target_temp_grad.flatten(0, 1),
                    mask=temp_mask.flatten(0, 1),
                ) * pow(self.temp_grad_decay, scale)
                cnt += 1

        return total / cnt if cnt else prediction.sum() * 0.0


def prealign_scale(
    target: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-frame scale that puts a teacher's depth on the target's metric
    scale: the median of target / teacher over the valid pixels, [N]. A frame
    with no usable pixel gets 1.0 (its pixels are masked out downstream)."""
    usable = valid & (teacher > 0) & (target > 0)
    ratio = torch.where(usable, target / teacher.clamp(min=1e-6), torch.nan)
    s = torch.nanmedian(ratio.flatten(1), dim=1).values
    return torch.where(torch.isfinite(s), s, torch.ones_like(s))


def teacher_edge_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    prealign: bool = True,
    log_space: bool = False,
    log_eps: float = 1e-3,
    teacher_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prompt Depth Anything's edge-aware term (Lin et al. 2025, Eq. 2):
    |d/dx (D_pred - D_teacher)| + |d/dy (D_pred - D_teacher)|, i.e. the
    prediction's spatial gradient against the TEACHER's, where the teacher is
    a model with better edges than the sensor GT. Single scale, L1, mean over
    the finite differences whose two pixels are teacher-valid (dense: the
    sensor's holes sit exactly at the silhouettes this term is for, so GT
    validity must not mask it; teacher_valid excludes the teacher's sky and
    non-finite pixels). Without teacher_valid the GT mask is used instead.

    prediction, teacher, target: [N,H,W] depth; valid: [N,H,W] bool, the GT
    validity, used for the prealign fit.
    prealign: scale the teacher per frame onto the target's metric scale first
    (prealign_scale) -- in linear depth a teacher that is 20% off would
    otherwise ask for edges 20% too tall everywhere. log_space compares
    log-depth gradients instead, which is scale-free by itself (the prealign
    is then a no-op on the gradients) and makes the term relative rather
    than in metres.
    """
    if prediction.shape != teacher.shape or prediction.shape != target.shape:
        raise ValueError(
            "teacher_edge_loss wants matching [N,H,W] tensors, got "
            f"{tuple(prediction.shape)}, {tuple(teacher.shape)}, {tuple(target.shape)}"
        )
    if prealign:
        s = prealign_scale(target, teacher, valid)
        teacher = teacher * s.view(-1, 1, 1)
    if log_space:
        prediction = torch.log(prediction.clamp(min=log_eps))
        teacher = torch.log(teacher.clamp(min=log_eps))
    residual = prediction - teacher
    dx = residual[:, :, 1:] - residual[:, :, :-1]
    dy = residual[:, 1:, :] - residual[:, :-1, :]
    pair = valid if teacher_valid is None else teacher_valid
    mx = pair[:, :, 1:] & pair[:, :, :-1]
    my = pair[:, 1:, :] & pair[:, :-1, :]
    count = mx.sum() + my.sum()
    if count == 0:
        return residual.sum() * 0.0
    return (dx.abs()[mx].sum() + dy.abs()[my].sum()) / count
