import torch

from .base import Details, MultiLoss, View
from .gradient_loss import TemporalGradientMatchingLoss, teacher_edge_loss
from .head_loss import DepthOrPmapLoss
from .utils import compute_scale_and_shift


class DepthTrainLoss(MultiLoss):
    def __init__(
        self,
        weights: tuple[float, ...] | None = None,
        alpha: float = 0.1,
        trim: float = 0.2,
        temp_grad_scales: int = 4,
        temp_grad_decay: float = 0.5,
        reduction: str = "batch-based",
        diff_depth_th: float = 0.05,
        metric: bool = False,
        log_space: bool = False,
        conf_weighting: bool = True,
        edge_weight: float = 0.0,
    ) -> None:
        super().__init__()
        # edge_weight > 0 adds Prompt Depth Anything's edge-aware term
        # (gradient_loss.teacher_edge_loss) against a teacher depth that the
        # training loop attaches to every GT view as g["teacher_depth"]
        # (finetune_depth: TeacherCfg / --teacher.model). The data term keeps
        # the sensor GT; only the spatial gradient comes from the teacher.
        # It follows log_space, like the accuracy term.
        self.edge_weight = edge_weight
        if edge_weight > 0 and not metric:
            # the edge term compares the UNALIGNED prediction with a teacher
            # on GT scale; under a scale/shift-aligned data term it would
            # push the metric scale through the edges alone
            raise ValueError("DepthTrainLoss edge_weight > 0 requires metric=True")
        # metric=True supervises absolute depth end-to-end: the depth term skips
        # its scale/shift alignment and the temporal term skips its scale fit, so
        # the metric scale the model is conditioned on is actually penalized.
        self.metric = metric
        # log_space=True runs the depth ACCURACY term (DepthOrPmapLoss) on
        # log-depth so error is relative, not absolute metres -- the far
        # background stops dominating. The temporal term is left in linear depth
        # (it already masks to near-static pixels via diff_depth_th).
        self.log_space = log_space
        # conf_weighting=False drops the learned-confidence weighting AND its
        # -alpha*log(sigma) regularizer from the depth term (alpha then has no
        # effect); the temporal term never used confidence. See
        # DepthOrPmapLoss.__init__ for why the two parts must go together.
        self.conf_weighting = conf_weighting
        self.temporal_loss = TemporalGradientMatchingLoss(
            trim=trim,
            temp_grad_scales=temp_grad_scales,
            temp_grad_decay=temp_grad_decay,
            reduction=reduction,
            diff_depth_th=diff_depth_th,
        )
        self.depth_loss = DepthOrPmapLoss(
            alpha=alpha,
            metric=metric,
            log_space=log_space,
            conf_weighting=conf_weighting,
        )

        if weights is None:
            # equal weight per objective (temporal, depth) when none supplied
            weights = (1.0,) * 2

        self.weights = weights

    def get_name(self) -> str:
        # the ablation is visible in the logged criterion string; only used for
        # repr (compute_loss returns its own details dict, so this is not a
        # metric key).
        return "DepthTrainLoss" if self.conf_weighting else "DepthTrainLoss(no_conf)"

    def compute_loss(
        self,
        gts: list[View],
        preds: list[View],
        track_queries: torch.Tensor | None = None,
        track_preds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Details]:
        losses = []

        # ---------- Ltemporal (temporal depth-gradient consistency) ----------
        # GT dense depth lives in "depthmap" [B,H,W]; the model's prediction is
        # "depth" [B,H,W,1] ("depth" on a GT view would be a teacher output, not
        # ground truth -- cf. FinetuneLoss, which also reads g["depthmap"]).
        pred_depth = torch.stack([p["depth"] for p in preds], dim=1).squeeze(-1)
        gt_depth = torch.stack([g["depthmap"] for g in gts], dim=1)
        temp_mask = torch.stack([g["valid_mask"] for g in gts], dim=1)
        raw_pred_depth = pred_depth  # the edge term below wants the unaligned output

        if not self.metric:
            # Align prediction to GT scale/shift over the clip before matching
            # temporal gradients (cf. VideoDepthLoss); skipped in metric mode so
            # the absolute temporal rate-of-change is supervised. (The +shift is
            # a no-op for the temporal diff, but the *scale removal is what we
            # drop for metric.)
            scale, shift = compute_scale_and_shift(
                pred_depth.flatten(1, 2),
                gt_depth.flatten(1, 2),
                temp_mask.flatten(1, 2),
            )
            pred_depth = scale.view(-1, 1, 1, 1) * pred_depth + shift.view(-1, 1, 1, 1)
        Ltemporal = self.temporal_loss(
            prediction=pred_depth, target=gt_depth, mask=temp_mask
        )

        losses.append((Ltemporal, "Ltemporal"))

        # ---------- Ldepth ----------
        depth_terms = []
        # per-frame breakdown of Ldepth for logging: main (confidence-weighted
        # L1), grad (spatial-gradient/edge), reg (-alpha*log(sigma) confidence
        # regularizer) -- these three sum to Ldepth. main_raw (sigma-free error)
        # and conf (mean sigma) are diagnostics that DECOMPOSE main = conf*error
        # (not summands); they separate an accuracy overfit from a confidence
        # overfit when main climbs on val.
        comp_terms: dict[str, list[torch.Tensor]] = {
            "main": [],
            "grad": [],
            "reg": [],
            "main_raw": [],  # sigma-free error: main / conf, tracks accuracy
            "conf": [],  # mean confidence sigma: the weight applied to main
        }
        for g, p in zip(gts, preds, strict=True):
            if "depth" in p:
                sigma_p = p["depth_conf"] if self.conf_weighting else None
                valid_mask = g["valid_mask"]
                if not valid_mask.any():
                    continue
                term, comps = self.depth_loss(
                    p["depth"],
                    g["depthmap"].unsqueeze(-1),
                    sigma_p=sigma_p,
                    valid_mask=valid_mask,
                    return_components=True,
                )
                depth_terms.append(term)
                for k, v in comps.items():
                    comp_terms[k].append(v)
        Ldepth = (
            torch.stack(depth_terms).mean()
            if depth_terms
            else torch.zeros_like(Ltemporal)
        )

        losses.append((Ldepth, "Ldepth"))

        total = 0.0
        details = {}

        for loss, weight in zip(losses, self.weights, strict=True):
            total += weight * loss[0]
            details[loss[1]] = loss[0]

        # ---------- Ledge (teacher-gradient edge term, optional) ----------
        if self.edge_weight > 0:
            if any("teacher_depth" not in g for g in gts):
                raise ValueError(
                    "DepthTrainLoss edge_weight > 0 needs g['teacher_depth'] on "
                    "every GT view (set --teacher.model); none was attached"
                )
            teacher = torch.stack([g["teacher_depth"] for g in gts], dim=1)
            teacher_valid = (
                torch.stack([g["teacher_valid"] for g in gts], dim=1).flatten(0, 1)
                if all("teacher_valid" in g for g in gts)
                else None
            )
            Ledge = teacher_edge_loss(
                raw_pred_depth.flatten(0, 1),
                teacher.flatten(0, 1).to(raw_pred_depth.dtype),
                gt_depth.flatten(0, 1),
                temp_mask.flatten(0, 1),
                prealign=True,
                log_space=self.log_space,
                teacher_valid=teacher_valid,
            )
            total = total + self.edge_weight * Ledge
            details["Ledge"] = Ledge

        # logging-only breakdown of Ldepth (NOT added to total; detached in
        # DepthOrPmapLoss). Zeros when no frame carried a depth prediction,
        # mirroring Ldepth's own fallback.
        for name, terms in comp_terms.items():
            details[f"Ldepth_{name}"] = (
                torch.stack(terms).mean() if terms else torch.zeros_like(Ltemporal)
            )

        return total, details
