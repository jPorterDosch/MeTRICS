#!/usr/bin/env python
# --------------------------------------------------------
# Standalone visualizer for a trained depth-conditioned StreamVGGT run.
#
# Given a weights directory (or a checkpoint file) produced by
# finetune_depth.py, this rebuilds the exact model architecture from the
# checkpoint's saved config, loads the finetuned weights, runs the streaming
# (causal, per-frame KV-cache) inference path on a handful of validation
# clips, and writes each clip's predicted-depth point cloud to a .glb.
#
# It deliberately reuses the training code's model and data helpers so the
# visualized geometry is produced by the same model path that eval uses.
#
# Requires a CUDA device for end-to-end model inference.
#
# Example:
#   cd src
#   python visualize_depth.py --weights ../checkpoints/metric_depth_cond_<id> \
#       --num-clips 4 --checkpoint best
# --------------------------------------------------------
import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: heatmap export must work on compute nodes
import numpy as np
import torch
import trimesh
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

from dust3r.inference import loss_of_one_batch, sample_query_points  # noqa
from eval.temporal_consistency.metrics import depth2point, point2depth
from finetune_depth import (
    FinetuneDepthCfg,
    _clip_predictions,
    _img2lidar,
    _prepare_batch,
    _set_data_epoch,
    _stack_depth_batch,
    build_model,
    build_train_loader,
)
from streamvggt.datasets import DatasetName, MultiDatasetConfig, Split
from visual_util import apply_scene_alignment, integrate_camera_into_scene
from streamvggt.depth_cond import (
    DepthCondCfg,
    EncoderCacheCfg,
    LoRACfg,
    MetricCfg,
    TrainCondCfg,
)

# Third comparison arm: PromptDA (Prompt Depth Anything), vendored verbatim in
# third_party/promptda (see its README). promptda is a namespace package (no
# __init__.py), so the vendor DIR goes on sys.path, not the package itself --
# which is why this import must sit below the path setup (hence the noqa).
_PROMPTDA_REPO = os.environ.get(
    "PROMPTDA_REPO",
    str(Path(__file__).resolve().parent.parent / "third_party" / "promptda"),
)
if _PROMPTDA_REPO not in sys.path:
    sys.path.insert(0, _PROMPTDA_REPO)
from promptda.promptda import PromptDA  # noqa: E402

# Checkpoint basenames finetune_depth.py writes, best -> worst preference when
# --checkpoint is left at "auto". The pipeline no longer writes a separate
# checkpoint-final.pth; the end-of-training weights live in checkpoint-last.pth,
# so the "final" alias now resolves there (kept for backward compatibility).
_CKPT_NAMES = {
    "final": "checkpoint-last.pth",
    "best": "checkpoint-best.pth",
    "last": "checkpoint-last.pth",
}
_AUTO_ORDER = ("final", "best", "last")


def _format_frame_timing(frame_times_ms) -> str:
    t = np.asarray(frame_times_ms, dtype=float)
    if len(t) == 1:
        return f"frame0 {t[0]:.1f} ms"
    rest = t[1:]
    slope = (
        float(np.polyfit(np.arange(len(rest)), rest, 1)[0])
        if len(rest) >= 2
        else float("nan")
    )
    return (
        f"frame0 {t[0]:.1f} ms | frames 1-{len(t) - 1} "
        f"median {np.median(rest):.1f} ms (min {rest.min():.1f}, "
        f"max {rest.max():.1f}) | growth {slope:+.3f} ms/frame"
    )


def _run_streaming_inference(model, views, frame_times_ms=None):
    """Run the first-party streaming path, optionally collecting frame times."""
    query_pts = (
        sample_query_points(views[0]["valid_mask"], M=64).to(
            device=views[0]["img"].device
        )
        if "valid_mask" in views[0]
        else None
    )
    on_cuda = views[0]["img"].device.type == "cuda"
    dtype = (
        torch.bfloat16
        if on_cuda and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    with torch.cuda.amp.autocast(dtype=dtype, enabled=on_cuda), torch.no_grad():
        output = model.inference(views, query_pts, frame_times_ms=frame_times_ms)
    return {"views": output.views, "pred": output.ress}


# PromptDA is frame-independent (no KV cache) and consumes the same [0,1] RGB
# + sparse-depth prompt the other arms see.
_PROMPTDA_CKPT_DEFAULT = os.environ.get(
    "PROMPTDA_CKPT", "depth-anything/prompt-depth-anything-vitl"
)


def _load_promptda(ckpt: str, device) -> torch.nn.Module:
    """Build the vendored PromptDA in eval mode.

    The checkpoint is resolved here (local path, else HF hub download) and its
    existence asserted BEFORE construction: PromptDA.load_checkpoint only warns
    on a missing file and would silently run with random depth-head weights.
    """
    if os.path.exists(ckpt):
        resolved = ckpt
    else:
        try:
            resolved = hf_hub_download(
                repo_id=ckpt, repo_type="model", filename="model.ckpt"
            )
        except Exception as exc:
            raise SystemExit(
                f"could not resolve PromptDA checkpoint {ckpt!r}: {exc}\n"
                "pre-download on a login node with\n"
                '  python -c "from huggingface_hub import hf_hub_download; '
                f"hf_hub_download('{ckpt}', 'model.ckpt')\"\n"
                "or pass --promptda-ckpt /abs/path/to/model.ckpt"
            ) from exc
    if not os.path.exists(resolved):
        raise SystemExit(f"PromptDA checkpoint {resolved!r} does not exist")

    print(f"PROMPTDA model: loading weights {resolved}")
    return PromptDA(encoder="vitl", ckpt_path=resolved).to(device).eval()


def _infill_sparse_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor infill of a sparse depth map ([H,W] meters, [H,W] bool).

    PromptDA normalizes the prompt by its min/max over the WHOLE map, and the
    fusion blocks bilinearly resample it, so sparse zeros both wreck the
    normalization range and bleed into valid measurements. PromptDA's own
    pipelines densify first with exactly this distance-transform gather."""
    if not mask.any():
        raise SystemExit(
            "frame has 0 valid sparse-depth pixels; PromptDA has no prompt "
            "to normalize against"
        )
    if mask.all():
        return depth
    from scipy.ndimage import distance_transform_edt

    _, idx = distance_transform_edt(~mask, return_indices=True)
    out = depth.copy()
    out[~mask] = depth[idx[0][~mask], idx[1][~mask]]
    return out


def _run_promptda_inference(pmodel, views, frame_times_ms=None) -> list[dict]:
    """Per-frame PromptDA over an [S]-list of prepared views. Returns preds
    shaped like the other arms' (p["depth"] is [B,H,W,1]) so
    _stack_depth_batch and the glb/heatmap exports work unchanged.

    Timing is wall-clock around the WHOLE per-frame pipeline -- the CPU
    nearest-neighbor infill, the host<->device copies, and the forward -- with
    a CUDA sync on each side, so the frame_ms column stays like-for-like with
    the streaming arms, whose timings cover their full per-frame inference.
    """
    device = views[0]["img"].device
    preds = []
    with torch.no_grad():
        for view in views:
            if frame_times_ms is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            img = view["img"].float()  # [B,3,H,W] in [0,1]; norm is internal
            sd = view["sparse_depth"].float().cpu().numpy()  # [B,H,W] meters
            sm = view["sparse_depth_mask"].cpu().numpy().astype(bool)
            prompt = np.stack(
                [_infill_sparse_depth(sd[b], sm[b]) for b in range(sd.shape[0])]
            )
            prompt = torch.from_numpy(prompt).unsqueeze(1).to(device)  # [B,1,H,W]
            depth = pmodel.predict(img, prompt)
            if frame_times_ms is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                frame_times_ms.append((time.perf_counter() - t0) * 1000.0)
            preds.append({"depth": depth.permute(0, 2, 3, 1)})  # [B,H,W,1]
    return preds


def resolve_checkpoint(weights: str, which: str) -> Path:
    """Resolve --weights (+ --checkpoint) to a concrete .pth. `weights` may be
    the checkpoint file itself or the run directory that contains it."""
    p = Path(weights)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"--weights is neither a file nor a directory: {p}")
    if which != "auto":
        cand = p / _CKPT_NAMES[which]
        if not cand.is_file():
            raise FileNotFoundError(f"No {cand.name} in {p}")
        return cand
    for key in _AUTO_ORDER:
        cand = p / _CKPT_NAMES[key]
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No checkpoint ({', '.join(_CKPT_NAMES.values())}) found in {p}"
    )


def load_saved_args(ckpt: dict) -> dict:
    """The primitive config snapshot picklable_args() embedded at save time.
    Everything is builtin types (enums as strings) -- the config dataclasses'
    validate() coerces them back."""
    if "args" not in ckpt:
        raise KeyError(
            "checkpoint has no 'args' snapshot; cannot reconstruct the model "
            "architecture. Was it written by finetune_depth.py?"
        )
    saved = ckpt["args"]
    return dict(vars(saved)) if not isinstance(saved, dict) else dict(saved)


def rebuild_metric_cfg(raw: dict) -> MetricCfg:
    """Reconstruct the architecture config from the primitive snapshot. Each
    nested dataclass's validate() turns the string/list primitives back into
    enum members, so the built model matches the checkpoint's key layout."""
    return MetricCfg(
        depth_cond=DepthCondCfg(**raw["depth_cond"]),
        lora=LoRACfg(**raw["lora"]),
        encoder_cache=EncoderCacheCfg(**raw["encoder_cache"]),
        train=TrainCondCfg(**raw["train"]),
    ).validate()


# MultiDatasetConfig fields that are parallel per-dataset tuples (entry i
# describes dataset i); every other field is shared across the mixture and so
# survives narrowing untouched. Kept in sync with MultiDatasetConfig's own
# "per-dataset parallel tuples" block -- narrowing by list length instead would
# silently mangle any shared list that happened to match the dataset count.
_PER_DATASET_FIELDS = (
    "root",
    "dataset",
    "stride_range",
    "epoch_size",
    "is_metric",
    "highres_root",
)


def _dataset_names(vd: dict) -> list[str]:
    """Saved dataset names as plain strings. The checkpoint snapshot stores
    primitives, but an in-memory config holds DatasetName members."""
    return [d.value if isinstance(d, DatasetName) else str(d) for d in vd["dataset"]]


def _select_dataset_entry(vd: dict, i: int) -> None:
    """Narrow every parallel per-dataset tuple to entry i in place, leaving a
    one-dataset config. Optional tuples that the run omitted stay omitted."""
    for field in _PER_DATASET_FIELDS:
        values = vd.get(field)
        if values is not None:
            vd[field] = [values[i]]


def rebuild_val_dataset(
    raw: dict, data_root: str | None, dataset: str | None = None
) -> MultiDatasetConfig:
    """Reconstruct the validation dataset config saved with the run, narrowed to
    exactly one dataset (this tool visualizes a single clip, so a mixture would
    silently export whichever dataset the sampler happened to draw).

    --dataset does one of two things depending on whether the run already
    validates on that dataset:

    * SELECT -- the name is in the saved val config (e.g. --dataset hammer on a
      HAMMER+ScanNet run): pin the mixture to that dataset's own entry. Root,
      stride range and epoch size all come from the saved config, so --data-root
      is optional.
    * SUBSTITUTE -- the name is not in the saved config (e.g. --dataset
      arkitscenes on a HAMMER run): swap the dataset TYPE to probe out-of-domain
      behaviour. The saved roots all point at the wrong tree, so --data-root is
      required.

    --data-root on its own just relocates the tree (useful when the data lives
    somewhere other than the training CWD); it needs the config to already be
    down to one dataset, since otherwise there is no saying which root it means.
    """
    vd = dict(raw["val_dataset"])
    names = _dataset_names(vd)
    if dataset is not None:
        if dataset in names:
            _select_dataset_entry(vd, names.index(dataset))
        else:
            if data_root is None:
                raise ValueError(
                    f"--dataset {dataset} is not in this run's val config "
                    f"({', '.join(names)}), so it needs --data-root pointing at "
                    f"that dataset's processed tree"
                )
            # keep entry 0's shared-by-position fields (stride, epoch size) and
            # overwrite the type; the root is replaced by --data-root below
            _select_dataset_entry(vd, 0)
            vd["dataset"] = [dataset]
    if data_root is not None:
        if len(vd["root"]) != 1:
            raise ValueError(
                "--data-root only supports a single-dataset val config; the "
                f"saved config has {len(vd['root'])} roots ({', '.join(names)}). "
                f"Pass --dataset <name> to pin one of them first."
            )
        vd["root"] = [data_root]
        # the lowres loader's highres-exclusion root is a sibling of the old
        # tree; drop it so it does not point outside the overridden root
        if vd.get("highres_root") is not None:
            vd["highres_root"] = [None]
    # The primitive snapshot stores every Path as a plain string, but the
    # dataclass fields are Path-typed and DatasetConfig.validate() calls
    # root.exists(). MultiDatasetConfig.validate() only coerces the enum
    # fields (dataset/split/transform), so convert the path tuples back here
    # or build_all() dies with AttributeError: 'str' has no attribute 'exists'.
    vd["root"] = [Path(r) for r in vd["root"]]
    if vd.get("highres_root") is not None:
        vd["highres_root"] = [
            None if r is None else Path(r) for r in vd["highres_root"]
        ]
    if len(vd["dataset"]) > 1:
        # Left as a mixture on purpose (the caller wants the run's whole val
        # split), but anything that exports ONE clip then labels it after the
        # run cannot say which dataset it drew -- warn rather than fail, since
        # paired-arm scoring over the mixture is still valid.
        print(
            f"WARNING val config keeps {len(vd['dataset'])} datasets "
            f"({', '.join(_dataset_names(vd))}); sampled clips may come from "
            f"any of them. Pass --dataset <name> to pin one."
        )
    return MultiDatasetConfig(**vd)


def _stack_depth_conf(preds: list[dict]) -> torch.Tensor:
    """The model's predicted depth confidence for a whole clip, [B,S,H,W] cpu.

    This is the raw depth-head output. The head is built with
    conf_activation="expp1" (streamvggt.py), i.e. conf = 1 + exp(x): 1.0 is the
    hard floor and there is NO ceiling.

    Do not confuse it with the 0/1 geometry mask _clip_predictions stores under
    the same "depth_conf" key -- that one is "GT-valid AND finite", written to
    drive predictions_to_glb's conf>1e-5 filter, and carries none of the model's
    learned uncertainty. Both the training forward and the streaming inference
    path emit p["depth_conf"] as [B,H,W]."""
    return torch.stack([p["depth_conf"].detach().float().cpu() for p in preds], dim=1)


def _clip_confidences(preds: list[dict]) -> list[tuple[float, float, float]]:
    """Per-clip (mean, min, max) of the model's predicted depth confidence over
    the whole clip -- a quick read on whether the learned confidences are
    degenerate/saturating (cf. the metric-mode depth_alpha caveat: raw-metre
    residuals can swamp the -alpha*log(sigma) regularizer). Returns one tuple
    per batch element b, in b order (matches this script's own export
    order)."""
    flat = _stack_depth_conf(preds).flatten(1)
    return [
        (flat[b].mean().item(), flat[b].min().item(), flat[b].max().item())
        for b in range(flat.shape[0])
    ]


def _per_frame_scene(predictions: dict) -> trimesh.Scene:
    """Build a GLB scene whose per-frame point clouds are SEPARATE, named
    geometries ("frame_000", "frame_001", ...) instead of one fused cloud, so
    the viewer can show/step through frames individually. Cameras for every
    frame are always added (they trace the trajectory). Same world alignment as
    predictions_to_glb (aligned to frame 0), so frames overlay consistently.

    `predictions` is the dict _clip_predictions returns: world_points_from_depth
    [S,H,W,3], depth_conf [S,H,W] (0/1 valid mask here), images [S,3,H,W] in
    [0,1], extrinsic [S,3,4] world->cam."""
    world = predictions["world_points_from_depth"]  # [S,H,W,3]
    conf = predictions["depth_conf"]  # [S,H,W]
    images = predictions["images"]
    extr = predictions["extrinsic"]  # [S,3,4] world->cam
    S = world.shape[0]

    if images.ndim == 4 and images.shape[1] == 3:  # NCHW -> NHWC
        colors = np.transpose(images, (0, 2, 3, 1))
    else:
        colors = images

    extrinsics_4x4 = np.zeros((S, 4, 4), dtype=np.float32)
    extrinsics_4x4[:, :3, :4] = extr
    extrinsics_4x4[:, 3, 3] = 1.0

    # one global scene scale (from all kept points) so camera markers are sized
    # consistently and the view does not rescale as frames toggle
    kept_all = world.reshape(-1, 3)[conf.reshape(-1) > 1e-5]
    if kept_all.size:
        lo = np.percentile(kept_all, 5, axis=0)
        hi = np.percentile(kept_all, 95, axis=0)
        scene_scale = float(np.linalg.norm(hi - lo)) or 1.0
    else:
        scene_scale = 1.0

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene = trimesh.Scene()
    for i in range(S):
        mask = conf[i].reshape(-1) > 1e-5
        verts = world[i].reshape(-1, 3)[mask]
        cols = (colors[i].reshape(-1, 3)[mask] * 255).astype(np.uint8)
        if verts.size:
            # geom_name -> node name -> three.js object.name (the viewer keys
            # its frame slider off the "frame_NNN" prefix)
            scene.add_geometry(
                trimesh.PointCloud(vertices=verts, colors=cols),
                geom_name=f"frame_{i:03d}",
            )
        rgba = colormap(i / max(S, 1))
        color = tuple(int(255 * x) for x in rgba[:3])
        cam_to_world = np.linalg.inv(extrinsics_4x4[i])
        integrate_camera_into_scene(scene, cam_to_world, color, scene_scale)
        # A marker node carrying the GT cam->world pose VERBATIM, so the viewer
        # can place its view camera at frame i's real camera. The frustum mesh
        # can't be used for this: its transform has the OpenGL conversion and a
        # scene_scale-dependent offset baked in, which the viewer cannot undo.
        # One point, hidden at load; the scene-wide alignment below applies to
        # this node exactly as it does to the clouds, so they stay consistent.
        scene.add_geometry(
            trimesh.PointCloud(vertices=np.zeros((1, 3), dtype=np.float32)),
            geom_name=f"campose_{i:03d}",
            transform=cam_to_world,
        )

    return apply_scene_alignment(scene, extrinsics_4x4)


# Default absolute scale for BOTH relative-error series so base/finetuned panels
# are directly comparable (a per-image autoscale would make every panel look
# equally bad). 0.10 = 10% relative error saturates the colormap; IN-DOMAIN val
# AbsRel is ~0.05, so structure is visible rather than crushed.
#
# This default is only right in-domain. Out-of-domain (e.g. a HAMMER-trained
# checkpoint visualized on ScanNet via --dataset) relative error routinely
# exceeds 10% at nearly every pixel, every panel clips to the top colour, and
# the comparison silently carries NO information -- measured 99.8% of valid
# pixels saturated on a scannet run. Raise it with --rel-vmax (~0.5 for that
# case) and check the saturated_frac column of the summary CSV, which exists to
# make exactly this failure visible without decoding the PNGs.
_REL_VMAX = 0.10

# Colour range for the predicted-confidence panels. The depth head's
# conf_activation is "expp1" (conf = 1 + exp(x)), so 1.0 is the hard floor and
# there is no ceiling -- hence a vmin as well as a vmax.
#
# It is FIXED for the same reason the error scales are: a per-clip autoscale
# would renormalize the base and the finetuned panel independently, so two
# clips with completely different confidence levels would render identically
# and the A/B would silently carry no information. 10.0 covers the bulk of an
# in-domain clip; the conf_saturated_frac column of the summary CSV says when
# it does not, and --conf-vmax raises it.
_CONF_VMIN, _CONF_VMAX = 1.0, 10.0


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """0-based ranks with ties assigned their group mean (scipy 'average')."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    grp = np.cumsum(np.r_[True, xs[1:] != xs[:-1]]) - 1  # group id per position
    counts = np.bincount(grp)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    r = np.empty(x.size, dtype=np.float64)
    r[order] = (starts + (counts - 1) / 2.0)[grp]
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two equal-length 1-D arrays.

    Rank rather than Pearson because relative depth error is heavy-tailed: a
    few pixels at 400% error would otherwise set the coefficient by themselves.

    Ties need AVERAGE ranks, not ordinal argsort-of-argsort. conf floors at 1.0
    (expp1), so a collapsing sigma piles mass on the floor; ordinal ranks then
    attenuate the coefficient toward 0, and on a constant frame report raster
    order vs error rather than NaN. Both read as "confidence is decoration" --
    the conclusion this number is used to draw."""
    if a.size < 2:
        return float("nan")
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _export_heatmaps(
    hm_dir: str,
    tag: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    K: torch.Tensor,
    pose: torch.Tensor,
    rel_vmax: float = _REL_VMAX,
    tcons_vmax: float | None = None,
    scale: int = 1,
    conf: torch.Tensor | None = None,
    conf_vmin: float = _CONF_VMIN,
    conf_vmax: float = _CONF_VMAX,
    frame_times_ms: list[float] | None = None,
) -> int:
    """2D heatmap companions to the GLB export, from the SAME predictions.

    pred/gt/valid: [S,H,W] cpu; K: [S,3,3]; pose: [S,4,4] cam2world.
    conf: optional [S,H,W] PREDICTED depth confidence from _stack_depth_conf
    (the model's own uncertainty head, not the 0/1 GLB mask); when omitted the
    conf panels/columns are skipped and the output is byte-identical to before.
    Writes:
      {tag}_depth_XXX.png     colormapped predicted depth (one shared robust
                              range across the clip, so brightness is comparable
                              frame to frame)
      {tag}_gterr_XXX.png     |pred-gt|/gt where GT is valid (accuracy)
      {tag}_conf_XXX.png      the model's predicted confidence, DENSE (no GT
                              mask -- see below), on the fixed [conf_vmin,
                              conf_vmax] scale so base and finetuned panels are
                              directly comparable
      {tag}_tcons_XXX.png     adjacent-pair self-consistency: pred[i] warped
                              into frame i+1 via the GT camera (depth2point /
                              point2depth -- the exact machinery tae() reduces
                              to a scalar), then |pred[i+1]-warped|/pred[i+1].
                              Needs NO ground-truth depth, only poses, so the
                              same code transfers to captures without GT.
      {tag}_summary.png       per-frame mean curves of both error series, plus
                              mean confidence on a twin axis
      {tag}_summary.csv       the SAME per-frame numbers as text, plus the
                              fraction of valid pixels clipped at rel_vmax and
                              the confidence/error rank correlation.

    frame_times_ms: optional per-frame inference milliseconds (see
    StreamVGGT.inference). Adds a frame_ms CSV column and a console line
    separating frame 0 (empty kv-cache) from the steady-state frames and their
    growth slope; omitted entirely when None.
                              The PNG is a picture of these curves and cannot
                              be compared across runs by eye; the CSV is what
                              you diff between base and finetuned.
    Pixels with no valid comparison are gray. Returns #files written."""
    import csv

    import matplotlib.pyplot as plt

    # The two series live on very different scales once out of domain: accuracy
    # can run >50% while warp self-consistency stays ~1%, so one shared ceiling
    # cannot show both (0.5 flattens gterr's structure into view but crushes
    # tcons to black). tcons_vmax defaults to rel_vmax, preserving the old
    # single-scale behaviour when only one is passed.
    if tcons_vmax is None:
        tcons_vmax = rel_vmax

    os.makedirs(hm_dir, exist_ok=True)

    def _imsave(path: str, rgba: np.ndarray) -> None:
        """plt.imsave writes the array at its NATIVE pixel size, so a panel is
        exactly the model's working resolution (518x392 -> ~1.7in at 300dpi).
        `scale` block-replicates for print. NEAREST on purpose: it enlarges
        without inventing structure, so the figure still shows the true
        resolution of the prediction. It adds NO information -- it only stops a
        poster pipeline from resampling it for you."""
        if scale > 1:
            rgba = rgba.repeat(scale, axis=0).repeat(scale, axis=1)
        plt.imsave(path, rgba)

    S = pred.shape[0]
    pred_np = pred.numpy()
    gt_np = gt.numpy()
    valid_np = valid.numpy().astype(bool)
    written = 0

    def save_map(path: str, rel: np.ndarray, ok: np.ndarray, vmax: float) -> float:
        """Write one colormapped panel; return the fraction of VALID pixels at
        or above vmax (i.e. clipped to the top colour). A value near 1 means
        the panel is saturated and holds no comparable structure -- the error
        is real but the picture cannot show how large it is."""
        cmap = matplotlib.colormaps["inferno"]
        rgba = cmap(np.clip(rel / vmax, 0.0, 1.0))
        rgba[~ok] = (0.5, 0.5, 0.5, 1.0)
        _imsave(path, rgba)
        return float((rel[ok] >= vmax).mean()) if ok.any() else float("nan")

    # --- predicted depth, one robust shared range across the clip ---
    finite = np.isfinite(pred_np) & (pred_np > 0)
    lo, hi = np.percentile(pred_np[finite], [2, 98]) if finite.any() else (0.0, 1.0)
    turbo = matplotlib.colormaps["turbo"]
    for i in range(S):
        x = np.clip((pred_np[i] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        rgba = turbo(x)
        rgba[~finite[i]] = (0.5, 0.5, 0.5, 1.0)
        _imsave(os.path.join(hm_dir, f"{tag}_depth_{i:03d}.png"), rgba)
        written += 1

    # --- accuracy vs GT, and the confidence panel that shares its mask ---
    # One loop on purpose: conf_err_corr needs the per-pixel rel-error field and
    # the GT-valid mask that gterr computes here, so splitting them would mean
    # rebuilding both.
    gterr_means, gterr_sat = [], []
    conf_means, conf_sat, conf_corr = [], [], []
    conf_np = None if conf is None else conf.numpy()
    viridis = matplotlib.colormaps["viridis"]
    for i in range(S):
        ok = valid_np[i] & (gt_np[i] > 0) & finite[i]
        rel = np.zeros_like(pred_np[i])
        rel[ok] = np.abs(pred_np[i][ok] - gt_np[i][ok]) / gt_np[i][ok]
        sat = save_map(
            os.path.join(hm_dir, f"{tag}_gterr_{i:03d}.png"), rel, ok, rel_vmax
        )
        gterr_means.append(float(rel[ok].mean()) if ok.any() else np.nan)
        gterr_sat.append(sat)
        written += 1

        if conf_np is None:
            continue
        # DENSE on purpose -- not masked to GT like gterr. Confidence exists at
        # every pixel the model predicts, and the GT holes (where the gterr
        # panel is gray) are exactly where a depth-COMPLETION model's own
        # uncertainty is the only thing you have to read. Gray here means a
        # non-finite confidence, i.e. a broken head, not "no GT".
        cok = np.isfinite(conf_np[i])
        span = max(conf_vmax - conf_vmin, 1e-9)
        rgba = viridis(np.clip((conf_np[i] - conf_vmin) / span, 0.0, 1.0))
        rgba[~cok] = (0.5, 0.5, 0.5, 1.0)
        _imsave(os.path.join(hm_dir, f"{tag}_conf_{i:03d}.png"), rgba)
        written += 1
        conf_means.append(float(conf_np[i][cok].mean()) if cok.any() else np.nan)
        conf_sat.append(
            float((conf_np[i][cok] >= conf_vmax).mean()) if cok.any() else np.nan
        )
        # Does the confidence actually track the error? NEGATIVE means yes (high
        # confidence where the error is low), which is the only reading under
        # which these panels mean anything. ~0 means the map is decoration, and
        # a POSITIVE value means the head is anti-calibrated. Computed on the
        # GT-valid pixels only -- it is the one thing here that needs GT.
        both = ok & cok
        conf_corr.append(
            _spearman(conf_np[i][both], rel[both]) if both.sum() >= 2 else np.nan
        )

    # --- temporal self-consistency (adjacent pairs, GT-depth-free) ---
    tcons_means, tcons_sat = [], []
    for i in range(S - 1):
        il_a, il_b = _img2lidar(K[i], pose[i]), _img2lidar(K[i + 1], pose[i + 1])
        warped = point2depth(
            depth2point(pred_np[i], finite[i], il_a),
            np.ones_like(finite[i + 1], dtype=np.float32),
            il_b,
        )
        ok = (warped > 1e-6) & finite[i + 1]
        rel = np.zeros_like(pred_np[i + 1])
        rel[ok] = np.abs(pred_np[i + 1][ok] - warped[ok]) / pred_np[i + 1][ok]
        sat = save_map(
            os.path.join(hm_dir, f"{tag}_tcons_{i:03d}.png"), rel, ok, tcons_vmax
        )
        tcons_means.append(float(rel[ok].mean()) if ok.any() else np.nan)
        tcons_sat.append(sat)
        written += 1

    # --- the numbers, as numbers ---
    # The curves below are unreadable across two runs by eye (that is how a
    # fully-saturated gterr comparison passed for "no difference"), so the same
    # values go to CSV. tcons is a PER-PAIR series: row i holds the (i, i+1)
    # pair, and the last row's tcons cells are empty.
    #
    # The column SET depends on whether conf was passed, so consumers must key
    # off the header row rather than fixed indices (compare_heatmap_summaries.py
    # does). New columns are appended, never inserted, so older CSVs still
    # parse.
    csv_path = os.path.join(hm_dir, f"{tag}_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        # meta = whole '#' rows, header = columns of one row; both are extended
        # under the same condition, so accumulate both and write once.
        meta = [["# tag", tag], ["# rel_vmax", rel_vmax], ["# tcons_vmax", tcons_vmax]]
        header = [
            "frame",
            "gterr_mean",
            "gterr_saturated_frac",
            "tcons_mean",
            "tcons_saturated_frac",
        ]
        if conf_np is not None:
            meta += [["# conf_vmin", conf_vmin], ["# conf_vmax", conf_vmax]]
            header += ["conf_mean", "conf_saturated_frac", "conf_err_corr"]
        if frame_times_ms is not None:
            header += ["frame_ms"]
        w.writerows(meta)
        w.writerow(header)
        for i in range(S):
            has_pair = i < S - 1
            row = [
                i,
                f"{gterr_means[i]:.6f}",
                f"{gterr_sat[i]:.6f}",
                f"{tcons_means[i]:.6f}" if has_pair else "",
                f"{tcons_sat[i]:.6f}" if has_pair else "",
            ]
            if conf_np is not None:
                row += [
                    f"{conf_means[i]:.6f}",
                    f"{conf_sat[i]:.6f}",
                    f"{conf_corr[i]:.6f}",
                ]
            if frame_times_ms is not None:
                # a clip shorter than the timing list means the caller sliced
                # frames after inference; pad rather than mis-align the rows
                row += [f"{frame_times_ms[i]:.3f}" if i < len(frame_times_ms) else ""]
            w.writerow(row)
    written += 1
    mean_gterr_sat = float(np.nanmean(gterr_sat)) if gterr_sat else float("nan")
    mean_tcons_sat = float(np.nanmean(tcons_sat)) if tcons_sat else float("nan")
    mean_conf_sat = float(np.nanmean(conf_sat)) if conf_sat else float("nan")
    mean_conf_corr = float(np.nanmean(conf_corr)) if conf_corr else float("nan")
    warn = [
        ("gterr", mean_gterr_sat, rel_vmax, "--rel-vmax"),
        ("tcons", mean_tcons_sat, tcons_vmax, "--tcons-vmax"),
    ]
    if conf_np is not None:
        warn.append(("conf", mean_conf_sat, conf_vmax, "--conf-vmax"))
    for series, sat, vmax, flag in warn:
        if sat > 0.5:
            print(
                f"  WARNING [{tag}]: {series} heatmaps are saturated at "
                f"vmax={vmax} ({sat:.1%} of valid pixels clipped). The panels "
                f"cannot show how large the value is -- raise {flag} and "
                f"re-render before comparing them."
            )
    if conf_np is not None:
        # The panels are pictures of a field whose scale is arbitrary; this
        # number is what says whether the field means anything at all.
        verdict = (
            "tracks error"
            if mean_conf_corr < -0.1
            else "ANTI-calibrated"
            if mean_conf_corr > 0.1
            else "uninformative"
        )
        print(
            f"  [{tag}] conf: mean {np.nanmean(conf_means):.3f}, "
            f"Spearman(conf, |pred-gt|/gt) = {mean_conf_corr:+.3f} ({verdict})"
        )
    if frame_times_ms:
        # Frame 0 is reported apart from the rest on purpose: it runs against an
        # EMPTY kv-cache and also absorbs lazy CUDA init, so averaging it in
        # hides the very growth this measures. The slope is the answer to "does
        # per-frame cost grow with sequence length" -- fit on frames 1.. only.
        print(f"  [{tag}] time: {_format_frame_timing(frame_times_ms)}")

    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    ax.plot(gterr_means, label="|pred-gt|/gt (per frame)", marker="o", ms=3)
    ax.plot(
        np.arange(S - 1) + 0.5,
        tcons_means,
        label="warp self-consistency (per pair)",
        marker="s",
        ms=3,
    )
    # the colormap ceilings, so a curve sitting on/above one explains a flat panel
    ax.axhline(rel_vmax, color="C0", ls="--", lw=1, label=f"gterr ceiling={rel_vmax:g}")
    if tcons_vmax != rel_vmax:
        ax.axhline(
            tcons_vmax, color="C1", ls=":", lw=1, label=f"tcons ceiling={tcons_vmax:g}"
        )
    ax.set_xlabel("frame")
    ax.set_ylabel("mean relative error")
    title = f"{tag}  (clipped: gterr {mean_gterr_sat:.0%}, tcons {mean_tcons_sat:.0%}"
    handles, labels = ax.get_legend_handles_labels()
    if conf_np is not None:
        # Twin axis: confidence is 1+exp(x) in arbitrary units, nothing like the
        # relative-error scale on the left, so sharing one axis would flatten
        # whichever series happens to be smaller.
        ax2 = ax.twinx()
        ax2.plot(conf_means, color="C2", marker="^", ms=3, lw=1, label="mean conf")
        ax2.set_ylabel("mean predicted confidence")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
        title += f", conf {mean_conf_sat:.0%}) rank corr {mean_conf_corr:+.2f}"
    else:
        title += ")"
    ax.set_title(title, fontsize=9)
    # no loc= -- matplotlib's "best" placement dodges the curves; a fixed corner
    # sat on top of the gterr series
    ax.legend(handles, labels, fontsize=8)
    fig.savefig(os.path.join(hm_dir, f"{tag}_summary.png"), dpi=150)
    plt.close(fig)
    return written + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        required=True,
        help="run output dir (containing checkpoint-*.pth) or a .pth file",
    )
    ap.add_argument(
        "--checkpoint",
        choices=["auto", "final", "best", "last"],
        default="auto",
        help="which checkpoint to load from a weights dir (default: auto)",
    )
    ap.add_argument(
        "--num-clips", type=int, default=4, help="how many val clips (scenes) to export"
    )
    ap.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="frames per scene; overrides the saved val config (default: as trained, "
        "typically 4). Use e.g. --num-clips 1 --num-views 32 for one long scene.",
    )
    ap.add_argument(
        "--base",
        action="store_true",
        help="visualize the BASE model (pretrained weights in the same architecture, "
        "conditioner/LoRA at init) instead of the finetuned checkpoint. Files are "
        "tagged base_* vs finetuned_* so both can share one --out-dir for A/B.",
    )
    ap.add_argument(
        "--pretrained",
        default=None,
        help="base checkpoint for --base (default: the pretrained path saved in the "
        "run's config; pass explicitly if that relative path does not resolve here).",
    )
    ap.add_argument(
        "--promptda",
        action="store_true",
        help="visualize PromptDA (Prompt Depth Anything) as a third arm. Runs the "
        "vendored torch model on the same clips/prompts as the other arms; files "
        "are tagged promptda_* so all three arms can share one --out-dir.",
    )
    ap.add_argument(
        "--promptda-ckpt",
        default=_PROMPTDA_CKPT_DEFAULT,
        help="PromptDA checkpoint: local model.ckpt path or HF repo id "
        f"(default {_PROMPTDA_CKPT_DEFAULT}; env PROMPTDA_CKPT). The vendored "
        "package dir is env-overridable via PROMPTDA_REPO (import-time).",
    )
    ap.add_argument(
        "--sparse-seed",
        type=int,
        default=0,
        help="seed for the simulated sparse-prompt draw, applied per clip. "
        "Keep it IDENTICAL across the arms you pair (the default is fine): it "
        "is what makes all arms condition on the same sparse pattern, which "
        "the frame-paired comparison in compare_heatmap_summaries.py assumes.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output dir (default: <weights-dir>/viz). GLBs land in <out-dir>/glb",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="override the saved val dataset root (single-dataset configs only)",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="pin the val config to ONE dataset. A name the run already "
        "validates on (e.g. hammer on a HAMMER+ScanNet run) selects that "
        "entry and keeps its saved root; any other name (e.g. scannet, "
        "arkitscenes) swaps the dataset TYPE to probe out-of-domain behaviour "
        "and requires --data-root pointing at that dataset's processed tree. "
        "Sparse conditioning is re-simulated from the new dataset's GT, so the "
        "pipeline runs unchanged.",
    )
    ap.add_argument(
        "--all-pixels",
        action="store_true",
        help="export the model's dense prediction at EVERY pixel instead of only "
        "where GT depth exists. This is the depth-completion output: predictions "
        "in GT holes (sensor dropouts, transparent/specular surfaces) are exactly "
        "what the GT-masked view hides. Files are tagged *_allpx so both views can "
        "share one --out-dir. Note the prediction is unconstrained in those holes, "
        "so it may be wild -- compare against the default view before trusting it.",
    )
    # NOTE: no --sequential flag. TEST-split datasets require
    # stride_range=(1, 1) (consecutive frames, enforced at construction),
    # so cross-view "temporal" readings are not producible by mistake.
    ap.add_argument(
        "--heatmaps",
        action="store_true",
        help="also write 2D PNG heatmaps per clip to <out-dir>/heatmaps: "
        "colormapped predicted depth, |pred-gt|/gt accuracy maps, the model's "
        "own predicted CONFIDENCE maps, and adjacent-pair warp self-consistency "
        "maps (the per-pixel field that tae() reduces to a scalar), plus a "
        "per-frame summary curve and CSV. Fixed color scales so base/finetuned "
        "panels are directly comparable.",
    )
    ap.add_argument(
        "--hm-scale",
        type=int,
        default=1,
        help="integer upscale for heatmap PNGs (default 1 = native). Panels are "
        "written at the model's working resolution (518x392), ~1.7in at 300dpi. "
        "NEAREST block replication -- enlarges for print without inventing "
        "structure, and adds NO information.",
    )
    ap.add_argument(
        "--rel-vmax",
        type=float,
        default=_REL_VMAX,
        help="relative error that saturates the gterr/tcons colormaps (default "
        f"{_REL_VMAX}, tuned for IN-DOMAIN val AbsRel ~0.05). Out-of-domain runs "
        "(e.g. --dataset scannet on a HAMMER checkpoint) blow past this at nearly "
        "every pixel, which flattens both panels to one colour and makes the "
        "comparison look like 'no difference'; try ~0.5 there. Keep it IDENTICAL "
        "across the base and finetuned runs you intend to pair, or the panels are "
        "not comparable. Check the saturated_frac columns of the summary CSV.",
    )
    ap.add_argument(
        "--tcons-vmax",
        type=float,
        default=None,
        help="separate ceiling for the warp self-consistency maps (default: same "
        "as --rel-vmax). The two series diverge out of domain -- accuracy can run "
        # NOTE the doubled %%: argparse runs every help string through
        # %-expansion, so a bare % here makes --help itself raise ValueError
        ">50%% while consistency stays ~1%% -- so one shared scale renders one of "
        "them useless. Set --rel-vmax 0.5 --tcons-vmax 0.03 for a scannet run.",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="measure per-frame inference time (CUDA events, so the timed loop "
        "is the loop that normally runs) and add a frame_ms column to the "
        "summary CSV. Off by default; the model path is untouched when off.",
    )
    ap.add_argument(
        "--conf-vmin",
        type=float,
        default=_CONF_VMIN,
        help=f"confidence at the BOTTOM of the conf colormap (default {_CONF_VMIN}). "
        "The depth head uses conf_activation='expp1' (1 + exp(x)), so 1.0 is its "
        "hard floor; raise this to spread colour over a narrow high-confidence "
        "band. Must match across the base and finetuned runs you pair.",
    )
    ap.add_argument(
        "--conf-vmax",
        type=float,
        default=_CONF_VMAX,
        help=f"confidence that saturates the conf colormap (default {_CONF_VMAX}). "
        "expp1 is unbounded above, so a well-trained head can blow past this and "
        "flatten every panel to one colour -- check the conf_saturated_frac column "
        "of the summary CSV. Keep it IDENTICAL across the runs you pair, or the "
        "two confidence panels are not comparable.",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    if args.base and args.promptda:
        raise SystemExit("--base and --promptda are mutually exclusive")

    if not torch.cuda.is_available():
        raise SystemExit(
            "visualize_depth requires a CUDA device (loss_of_one_batch runs "
            "under torch.cuda.amp.autocast)."
        )

    ckpt_path = resolve_checkpoint(args.weights, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw = load_saved_args(ckpt)
    mcfg = rebuild_metric_cfg(raw)
    val_ds = rebuild_val_dataset(raw, args.data_root, args.dataset)
    if args.num_views is not None:
        # more frames per scene -> longer sequences to scrub. num_views is
        # independent of resolution; the sampler and streaming path handle any
        # S (bounded by sequence length / GPU memory).
        val_ds.num_views = args.num_views

    mode = "promptda" if args.promptda else ("base" if args.base else "finetuned")
    if args.all_pixels:
        # distinct tag so a GT-masked and an all-pixels export can share one
        # --out-dir and sit side by side in the viewer's scene dropdown
        mode += "_allpx"
    out_dir = args.out_dir or str(
        (ckpt_path.parent if ckpt_path.parent != Path("") else Path(".")) / "viz"
    )
    os.makedirs(out_dir, exist_ok=True)

    # A minimal training config: build_model reads pretrained/resume, and
    # build_train_loader reads val_dataset/num_workers/fixed_length. The rest
    # keep their defaults (train_dataset/loss are never touched -- we only
    # build the TEST loader and never run the criterion).
    # --base loads pretrained weights into the same architecture (conditioner /
    # LoRA at their zero-init, so it reproduces the pretrained model's behavior
    # -- the baseline the finetuning improves on).
    pretrained_path = ""
    if args.base:
        pretrained_path = args.pretrained or raw.get("pretrained") or ""
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise SystemExit(
                "--base needs a valid pretrained checkpoint. The path saved in the "
                f"run config was {raw.get('pretrained')!r} (unresolved from here); "
                "pass --pretrained /abs/path/to/base_checkpoint.pth."
            )

    cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        val_dataset=val_ds,
        pretrained=pretrained_path,  # only used in --base; else weights load below
        resume=None,
        num_workers=args.num_workers,
        batch_size=1,  # streaming inference needs one clip at a time
        fixed_length=bool(raw.get("fixed_length", True)),
        output_dir=out_dir,
        export_glb=True,
        export_glb_max_clips=args.num_clips,
    )

    accelerator = Accelerator()
    device = accelerator.device

    # build_model applies LoRA + freezes (harmless for eval) so the module key
    # layout matches the saved state_dict.
    pmodel = None
    if args.promptda:
        # The StreamVGGT checkpoint is still loaded above: its saved config
        # rebuilds the val dataset, so PromptDA sees the exact clips/frames and
        # simulated sparsity the other arms see.
        pmodel = _load_promptda(args.promptda_ckpt, device)
        model = None
    elif args.base:
        # load_pretrained=True folds the base StreamVGGT weights in; no
        # finetuned state_dict is applied on top.
        print(f"BASE model: loading pretrained weights {pretrained_path}")
        model, _ = build_model(cfg, mcfg, device, load_pretrained=True)
    else:
        model, _ = build_model(cfg, mcfg, device, load_pretrained=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)
    if model is not None:
        model.eval()

    # TEST-split datasets are sequential by construction, so every export here
    # is temporally ordered
    loader = build_train_loader(cfg, Split.TEST, accelerator, batch_size=1)
    loader = accelerator.prepare(loader)
    _set_data_epoch(loader, 0)  # deterministic clip set/order (see val_loop)

    glb_dir = os.path.join(out_dir, "glb")
    os.makedirs(glb_dir, exist_ok=True)

    exported = 0
    clip_idx = 0
    with torch.no_grad():
        for batch in loader:
            if exported >= args.num_clips:
                break
            if batch[0]["img"].shape[0] != 1:
                raise ValueError(
                    "expected a batch-1 loader for streaming inference; got "
                    f"batch size {batch[0]['img'].shape[0]}"
                )
            # Reseed IMMEDIATELY before the sparse-prompt draw, per clip: the
            # simulated prompt comes from the global torch RNG, and the three
            # arms run as separate processes that consume different amounts of
            # RNG state before/around this point (model init, streaming path).
            # Without this each arm conditions on a DIFFERENT random prompt and
            # the frame-paired A/B in compare_heatmap_summaries measures prompt
            # lottery, not the model. Seeded per clip (not once) so later clips
            # stay paired even if an arm's inference consumes RNG state.
            torch.manual_seed(args.sparse_seed + clip_idx)
            clip_idx += 1
            _prepare_batch(batch, mcfg)
            # inference=True -> the causal per-frame KV-cache path (deployment
            # path), no criterion. result carries views + per-view preds.
            # Fresh list per clip: timings are per-clip, and reusing one would
            # concatenate clips into a single fake ramp.
            frame_times_ms = [] if args.timing else None
            if args.promptda:
                # frame-independent model; no streaming path, no confidence head
                preds = _run_promptda_inference(pmodel, batch, frame_times_ms)
                views = batch
                confs, conf_maps = None, None
            else:
                if frame_times_ms is None:
                    result = loss_of_one_batch(
                        batch,
                        model,
                        None,
                        accelerator,
                        inference=True,
                        symmetrize_batch=False,
                        use_amp=True,
                    )
                else:
                    result = _run_streaming_inference(model, batch, frame_times_ms)
                views, preds = result["views"], result["pred"]
                del result
                confs = _clip_confidences(preds)
                conf_maps = _stack_depth_conf(preds)  # [B,S,H,W], heatmap source
            # imgs is not in _stack_depth_batch; stack it the same way
            imgs = torch.stack([v["img"] for v in views], dim=1).float().cpu()
            pred, gt, valid, K, pose = _stack_depth_batch(views, preds)
            for b in range(pred.shape[0]):
                if exported >= args.num_clips:
                    break
                predictions = _clip_predictions(
                    imgs[b],
                    pred[b],
                    valid[b],
                    K[b],
                    pose[b],
                    mask_to_gt=not args.all_pixels,
                )
                scene = _per_frame_scene(predictions)
                scene.export(os.path.join(glb_dir, f"{mode}_clip{exported}.glb"))
                if args.heatmaps:
                    n_png = _export_heatmaps(
                        os.path.join(out_dir, "heatmaps"),
                        f"{mode}_clip{exported}",
                        pred[b],
                        gt[b],
                        valid[b],
                        K[b],
                        pose[b],
                        rel_vmax=args.rel_vmax,
                        tcons_vmax=args.tcons_vmax,
                        scale=args.hm_scale,
                        conf=None if conf_maps is None else conf_maps[b],
                        conf_vmin=args.conf_vmin,
                        conf_vmax=args.conf_vmax,
                        frame_times_ms=frame_times_ms,
                    )
                    print(f"  [{mode}] clip {exported}: {n_png} heatmap PNGs")
                if confs is not None:
                    mean_c, min_c, max_c = confs[b]
                    print(
                        f"  [{mode}] clip {exported}: {pred.shape[1]} frames | "
                        f"mean depth confidence {mean_c:.4f} (min {min_c:.4f}, "
                        f"max {max_c:.4f})"
                    )
                else:
                    print(f"  [{mode}] clip {exported}: {pred.shape[1]} frames")
                exported += 1
            del preds, batch

    print(f"Wrote {exported} .glb file(s) to {glb_dir}")


if __name__ == "__main__":
    main()
