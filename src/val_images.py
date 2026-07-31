"""Validation-sample selection and figures for wandb:
[image | predicted depth | GT depth].

One panel per sampled frame, a handful per validation pass, so a run can be
inspected by eye while it trains -- the scalar metrics say a run got worse, a
picture says whether it lost scale, went flat, or blew up in the GT holes.

The same selection also drives the .glb point-cloud export, so clip N's panel
and clip N's cloud are the same sequence -- one picture to read the error off,
one scene to fly through.

Kept out of finetune_depth.py so collection and rendering are importable (and
testable) without the training stack; the entrypoint only decides when to call
them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt
import numpy as np
import torch


def clip_dataset_label(views: list[dict], b: int) -> str:
    """Dataset label of clip b (all views in a clip come from one scene, hence
    one dataset). The collated 'dataset' field is a per-sample list/tuple; a
    bare str covers the unbatched case. Lowercased so keys are consistent
    regardless of loader casing (HAMMER_Multi emits 'hammer', ScanNet_Multi
    'ScanNet')."""
    d = views[0].get("dataset", "unknown")
    label = d if isinstance(d, str) else d[b]
    return str(label).lower()


class ValImageSampler:
    """Collects a stratified subset of validation frames and renders one
    [image | prediction | GT] panel per frame.

    Stratification is over the dataset label the loader stamps on each clip,
    read at runtime rather than from the config: the label a loader emits
    ("scannet", "hammer") need not match the config enum, and under DDP a
    rank's shard need not contain every configured dataset. Each label banks up
    to n_images candidates (the most any one label could need); the round-robin
    split over the labels actually seen happens in select(), once the pass is
    over.

    One frame per clip (frame_idx, the clip's first by default): frames within
    a clip are consecutive and near-identical, so spending the budget on
    distinct clips shows strictly more. Deterministic given the val loader's
    epoch-0 pin -- the same clips are drawn every pass, so the panels are
    comparable across epochs.

    keep_clip additionally banks each candidate's WHOLE clip (every view's
    image, predicted depth, mask and camera), which is what the .glb export
    needs. That is ~S x the memory of a panel candidate, so it is opt-in: only
    a run that asked for GLBs pays it.
    """

    def __init__(self, n_images: int, frame_idx: int = 0, keep_clip: bool = False):
        self.n_images = int(n_images)
        self.frame_idx = frame_idx
        self.keep_clip = keep_clip
        self._pool: dict[str, list[dict]] = {}

    def add_batch(self, views: list[dict], preds: list[dict]) -> None:
        """Bank candidates from one batch. views/preds are the [S]-lists
        loss_of_one_batch returns. Real GT is g["depthmap"] (NOT g["depth"],
        the teacher output); p["depth"] is the prediction. Copies only the one
        frame it keeps, and nothing at all once a label's quota is met."""
        if self.n_images <= 0 or not views:
            return
        i = min(self.frame_idx, len(views) - 1)
        view, pred = views[i], preds[i]
        for b in range(view["img"].shape[0]):
            label = clip_dataset_label(views, b)
            slot = self._pool.setdefault(label, [])
            if len(slot) >= self.n_images:
                continue
            candidate = {
                "label": label,
                "frame": i,
                # img is already rescaled to [0, 1] by _prepare_batch
                "img": view["img"][b].detach().float().cpu().numpy(),
                "pred": pred["depth"][b].detach().squeeze(-1).float().cpu().numpy(),
                "gt": view["depthmap"][b].detach().float().cpu().numpy(),
                "valid": view["valid_mask"][b].detach().cpu().numpy().astype(bool),
            }
            if self.keep_clip:
                candidate["clip"] = _clip_tensors(views, preds, b)
            slot.append(candidate)

    def select(self) -> list[dict]:
        """Round-robin over the datasets seen (label order, for determinism)
        until n_images panels are picked or the pool runs dry. Round-robin
        rather than fixed per-dataset quotas so a dataset that never appeared
        on this rank donates its share instead of leaving the budget unspent."""
        picked: list[dict] = []
        labels = sorted(self._pool)
        rank = 0
        while len(picked) < self.n_images and labels:
            drew = False
            for label in labels:
                if len(picked) >= self.n_images:
                    break
                slot = self._pool[label]
                if rank < len(slot):
                    picked.append(slot[rank])
                    drew = True
            if not drew:
                break
            rank += 1
        return picked

    def render(self, epoch: int, limit: int | None = None) -> list[tuple[str, object]]:
        """(caption, matplotlib Figure) for the first `limit` selected frames.
        The caller owns the figures and must close them."""
        return [
            (
                f"{s['label']} clip{n} frame{s['frame']} (epoch {epoch})",
                render_panel(
                    s["img"], s["pred"], s["gt"], s["valid"], title=s["label"]
                ),
            )
            for n, s in self.enumerate_selected(limit)
        ]

    def clips(self, limit: int | None = None) -> list[tuple[int, str, dict]]:
        """(index, label, clip tensors) for the first `limit` selected clips --
        the .glb export's half of the same selection, so index N here and
        "clipN" in a render() caption are the same sequence. Requires
        keep_clip."""
        if not self.keep_clip:
            raise ValueError("clips() needs ValImageSampler(keep_clip=True)")
        return [(n, s["label"], s["clip"]) for n, s in self.enumerate_selected(limit)]

    def enumerate_selected(self, limit: int | None = None) -> list[tuple[int, dict]]:
        """(index, candidate) over the selection, truncated to `limit`. The
        index is the selection's own, so every consumer of a prefix agrees on
        which clip is clipN -- and because select() is round-robin, a prefix is
        still spread over the datasets."""
        picked = self.select()
        if limit is not None:
            picked = picked[:limit]
        return list(enumerate(picked))


def _clip_tensors(views: list[dict], preds: list[dict], b: int) -> dict:
    """Clip b's whole sequence, in the [S,...] cpu layout _clip_predictions
    wants: img [S,3,H,W] in [0,1], depth/valid [S,H,W], K [S,3,3],
    pose [S,4,4] cam2world."""
    return {
        "img": torch.stack([v["img"][b] for v in views]).detach().float().cpu(),
        "depth": torch.stack([p["depth"][b] for p in preds])
        .detach()
        .squeeze(-1)
        .float()
        .cpu(),
        "valid": torch.stack([v["valid_mask"][b] for v in views]).detach().cpu().bool(),
        "K": torch.stack([v["camera_intrinsics"][b] for v in views])
        .detach()
        .float()
        .cpu(),
        "pose": torch.stack([v["camera_pose"][b] for v in views])
        .detach()
        .float()
        .cpu(),
    }


def render_panel(
    img: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    title: str = "",
):
    """One [image | prediction | GT] figure. img [3,H,W] in [0,1], the depths
    [H,W] in metres, valid [H,W] bool.

    Both depth panels share one colour scale (robust 2-98 percentile of the
    valid GT), which is the whole point of the figure: on a shared scale a
    prediction that is merely mis-SCALED looks uniformly off-colour, while one
    that lost STRUCTURE looks flat. Per-panel autoscaling would hide the first
    failure entirely. Pixels with no GT are gray in the GT panel -- the
    prediction panel stays dense, since a depth-completion model's output in
    the GT holes is exactly what masking would hide."""
    gt_ok = valid & np.isfinite(gt) & (gt > 0)
    pred_ok = np.isfinite(pred)
    if gt_ok.any():
        lo, hi = np.percentile(gt[gt_ok], [2, 98])
    elif pred_ok.any():
        lo, hi = np.percentile(pred[pred_ok], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    span = max(float(hi) - float(lo), 1e-6)
    turbo = matplotlib.colormaps["turbo"]

    def colorize(d: np.ndarray, ok: np.ndarray) -> np.ndarray:
        rgba = turbo(np.clip((np.nan_to_num(d) - lo) / span, 0.0, 1.0))
        rgba[~ok] = (0.5, 0.5, 0.5, 1.0)
        return rgba

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
    axes[0].imshow(np.clip(img.transpose(1, 2, 0), 0.0, 1.0))
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(colorize(pred, pred_ok))
    axes[1].set_title("prediction", fontsize=9)
    axes[2].imshow(colorize(gt, gt_ok))
    axes[2].set_title("GT depth", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10)

    # the panels draw pre-colormapped RGBA (so invalid pixels can be gray),
    # which carries no scale for a colorbar -- hand it the normalization the
    # two depth panels were actually built with
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(float(lo), float(lo) + span), cmap=turbo
    )
    fig.colorbar(sm, ax=axes[1:].tolist(), fraction=0.046, label="depth (m)")
    return fig
