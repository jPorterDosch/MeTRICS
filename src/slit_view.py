#!/usr/bin/env python
"""Slit-window view (temporal profiles) of benchmark sequences, the figure
Video Depth Anything uses to show temporal consistency (arXiv 2501.12375,
Fig. 9: "We show changes in color and depth over time at the vertical red
line in videos"; Fig. 5: "image temporal profiles obtained by slicing images
along the timeline at the red-line positions").

One pixel column of every frame, stacked left to right in time: an [H, T]
image per source. Where the scene is static along the column, a temporally
stable prediction draws smooth horizontal bands, as the GT does; flicker
shows as vertical streaks, scale drift as the colour sliding along the time
axis. Every prediction is put on the GT grid and aligned as a benchmark
protocol scores it -- `published` (VDA's one per-video disparity fit, which
is how their figures align and which leaves per-frame flicker in) or
`metric` (raw depth) -- with the benchmark's own sparse-depth draw, so the
view shows the output the table scored. sparse_aligned is not offered: its
per-frame fit removes exactly the per-frame scale changes this view is for.
On a TAE clip (--tae) the fit runs over every predicted frame, not only the
window TAE scores.

Two stages, so a figure is re-rendered without a GPU:

    cd src
    # GPU (experiments/all_datasets_finetune/slit_view.sh)
    python slit_view.py extract --out <dir> --dataset scannet --tae --max-sequences 2 \\
        --model Ours=ckpt:<run_dir> --model StreamVGGT=base:<run_dir> \\
        --model VDA=arm:vda --model DAV=dump:<baselines>/pred_dav
    # CPU, seconds
    python slit_view.py render --out <dir>

--model LABEL=KIND:SOURCE, KIND one of
  ckpt  a finetune_depth.py run directory, its checkpoint-last (as bench_checkpoint scores it)
  base  the pretrained StreamVGGT in that run's architecture (bench_checkpoint --base)
  arm   a baseline arm of eval/baselines/arms.py (vda, vda_metric, ovda, promptda,
        da3_metric), behind the reproduction gate bench_baselines benchmark uses
  dump  predictions written by bench_baselines dump-predictions (Depth Any Video runs
        only in its own venv), behind the same gate

extract writes <out>/<dataset>[_tae]_<sequence>/scene.npz (RGB and GT
columns, the reference frame) and <label>.npz + <label>.json per model;
render draws <out>/<dataset>[_tae]_<sequence>_x<col>.png from whatever a
scene directory holds, models in the order they were extracted.
"""

from __future__ import annotations

# first: eval.provenance's audit hook records every repository file run from
# here on, with its content at that moment (each model's code_files)
import eval.provenance  # noqa: E402,F401,I001

import argparse
import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval import protocols as P
from eval.vda_benchmark import SPECS

BENCH_ROOT = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench")
KINDS = ("ckpt", "base", "arm", "dump")
PROTOCOLS = ("published", "metric")
DISP_PERCENTILES = (2.0, 98.0)  # shared display range: the GT's disparity
LOG2_ERR_MAX = 1.0  # error row: log2(pred / GT) in [-1, 1], as compare_arms_gif
# what makes two models' panels comparable: one protocol, one sparse draw,
# one input resolution, one manifest (render checks they agree)
DRAW_KEYS = ("protocol", "density", "seed", "patch_size", "image_size", "tae")
GAP = 6  # px between panels
TITLE_PX = 36  # figure title
ROW_PAD_PX = 24  # panel titles above each row


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    source: str


def parse_model(spec: str) -> ModelSpec:
    """LABEL=KIND:SOURCE."""
    label, sep, rest = spec.partition("=")
    kind, colon, source = rest.partition(":")
    if not sep or not label or not colon or not source:
        raise ValueError(f"--model wants LABEL=KIND:SOURCE, got {spec!r}")
    if kind not in KINDS:
        raise ValueError(f"--model {spec!r}: KIND must be one of {KINDS}")
    if "/" in label or label == "scene":
        raise ValueError(
            f"--model {spec!r}: the label names a file (no '/', not 'scene')"
        )
    return ModelSpec(label, kind, source)


def columns(arr: np.ndarray, xs: list[int]) -> np.ndarray:
    """[S, H, W(, C)] -> [K, H, S(, C)]: column x of every frame, time on the
    horizontal axis."""
    return np.stack([np.swapaxes(arr[:, :, x], 0, 1) for x in xs])


def column_px(fracs: list[float], width: int) -> list[int]:
    return [int(round(f * (width - 1))) for f in fracs]


def align(
    pred_gt: np.ndarray,
    output: str,
    gt: np.ndarray,
    protocol: str,
    max_depth: float,
    held_out: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """(the prediction as `protocol` scores it, [S, H, W] depth on the GT
    grid, and that protocol's AbsRel over the sequence, for the panel label),
    as score_sequence computes both: `published` scores every GT pixel,
    `metric` the GT pixels outside `held_out` (the fed ones) on the depth
    floored and clipped to the valid range, which is what is drawn."""
    if protocol == "published":
        m, aligned = P.published_metrics(P.as_disparity(pred_gt, output), gt, max_depth)
        return aligned, m.abs_rel
    if protocol == "metric":
        if output != "depth":
            raise ValueError(
                "a disparity output has no metric scale: --protocol published"
            )
        m = P.metric_metrics(pred_gt, gt, max_depth, held_out=held_out)
        # what metric_metrics scores: a non-finite pixel at the floor, an error
        drawn = np.clip(P._finite_or_floor(pred_gt), P.DEPTH_FLOOR, max_depth)
        return drawn.astype(np.float32), m.abs_rel
    raise ValueError(f"unknown protocol {protocol!r}")


def display_range(gt_cols: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """Disparity range every depth panel of one figure shares: the GT's over
    the column shown, robust percentiles."""
    disp = 1.0 / gt_cols[valid]
    if not disp.size:
        raise ValueError("no valid GT pixel on the slit: pick another --x")
    lo, hi = np.percentile(disp, DISP_PERCENTILES)
    return float(lo), float(hi)


def scene_dir(out: Path, dataset: str, tae: bool, seq_name: str) -> Path:
    return out / f"{dataset}{'_tae' if tae else ''}_{seq_name}"


# ---------------------------------------------------------------------------
# extract (GPU)
# ---------------------------------------------------------------------------
def _sequences(args, spec):
    from eval.vda_benchmark import load_manifest

    # a TAE clip in full, as the benchmark predicts it (tae_slice=False)
    seqs = load_manifest(args.bench_root, spec, tae=args.tae, tae_slice=False)
    if args.tae:
        import bench_eval as BE

        # the TAE pass skips a clip without every frame's K (run_benchmark,
        # bench_baselines benchmark): drawing it would show an unscored clip
        seqs = [s for s in seqs if BE.has_cameras(s)]
    if args.sequences:
        by_name = {s.name: s for s in seqs}
        missing = [n for n in args.sequences if n not in by_name]
        if missing:
            raise KeyError(
                f"{spec.name}{' (tae)' if args.tae else ''} has no {missing}"
            )
        return [by_name[n] for n in args.sequences]
    return seqs[: args.max_sequences]


def gt_grid_rgb(frames: np.ndarray, seq, spec, gt_hw) -> np.ndarray:
    """[S, H, W, 3] uint8: the frames on the GT grid, registered the way an
    arm's prediction is (bench_baselines.register_to_gt)."""
    from bench_baselines import register_to_gt

    chans = [
        register_to_gt(
            frames[..., c].astype(np.float32), seq, spec, frames.shape[1:3], gt_hw
        )
        for c in range(3)
    ]
    return np.clip(np.rint(np.stack(chans, axis=-1)), 0, 255).astype(np.uint8)


# every protocol bench_baselines benchmark asks the reproduction gate for: an
# arm the table refused is not drawn either
GATE_PROTOCOLS = ("published", "sparse_aligned", "metric", "tae")


class Runner:
    """One --model, loaded once; predict() returns its depth or disparity on
    the GT grid for one sequence, as the benchmark computes it."""

    def __init__(self, m: ModelSpec, args, accelerator):
        from eval.baselines import record as R

        self.m = m
        self.accelerator = accelerator
        device = accelerator.device
        # each as it was scored: bench_baselines turns TF32 off for every arm
        # (verified without it), bench_checkpoint keeps croco's process-wide
        # TF32 for our model
        self.ours = self.tf32 = m.kind in ("ckpt", "base")
        if self.ours:
            from model_io import load_run_model

            run = load_run_model(m.source, "last", device, base=m.kind == "base")
            self.net, self.amp = run.model, bool(run.raw.get("amp", 1))
            self.output = "depth"
            self.weights = (
                "pretrained (base)" if m.kind == "base" else str(run.ckpt_path)
            )
            return
        import bench_baselines as BB
        from eval.baselines import arms as A

        uses = [(args.dataset, p) for p in GATE_PROTOCOLS]
        record = R.load()
        if m.kind == "arm":
            self.arm = A.build_arm(m.source, device, uses, record, args.allow_close)
        else:
            self.arm = BB.CachedArm(Path(m.source))
            R.require_allowed(record, self.arm.info.name, uses, args.allow_close)
        self.output = self.arm.info.output
        self.weights = f"{m.kind}:{m.source}"

    def predict(self, seq, spec, gt, views, frames, kind: str) -> np.ndarray:
        """The scoring paths: bench_eval.predict_on_gt for our model,
        bench_baselines.benchmark's _predict + register_to_gt for an arm."""
        import torch

        import bench_baselines as BB
        import bench_eval as BE
        from eval.vda_benchmark import model_to_gt

        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        gt_hw = tuple(gt.shape[1:])
        if self.ours:
            pred = BE.predict_on_gt(self.net, self.accelerator, views, gt_hw, self.amp)
            if pred.at_gt is not None:
                return pred.at_gt
            return model_to_gt(pred.depth, gt_hw, views[0].get("gt_box"))
        if isinstance(self.arm, BB.CachedArm):
            self.arm.key = (spec.name, kind, seq.name)
        if self.arm.info.prompted:
            raw = self.arm.predict(frames, BE._sparse_arrays(views))
        else:
            raw = self.arm.predict(
                frames, **BB._arm_kwargs(self.arm, seq, spec, frames, gt_hw)
            )
        return BB.register_to_gt(raw, seq, spec, frames.shape[1:3], gt_hw)


def extract(args) -> None:
    import torch
    from accelerate import Accelerator

    import bench_baselines as BB
    import bench_eval as BE  # imports croco: TF32 on, process-wide
    from eval import provenance as PROV
    from eval.vda_benchmark import model_to_gt

    spec = SPECS[args.dataset]
    accelerator = Accelerator()
    device = accelerator.device
    if device.type != "cuda":
        # the sparse draws (torch.rand under a fixed seed) differ on CPU: the
        # model would be fed other pixels than the benchmark fed it
        raise SystemExit("extract draws the benchmark's sparse depth on CUDA")
    dev = PROV.device_info()
    if dev["autocast_dtype"] not in (None, "bfloat16"):
        # as run_benchmark: under fp16 autocast our model's output is not
        # the one the table scored
        raise SystemExit(
            f"needs a bf16 GPU (A40/H100); this is {dev['gpu']} "
            f"({dev['autocast_dtype']} autocast)"
        )
    seqs = _sequences(args, spec)
    # every model loaded (and its protocol checked) before any inference
    runners = [Runner(parse_model(s), args, accelerator) for s in args.model]
    for r in runners:
        if args.protocol == "metric" and r.output != "depth":
            raise ValueError(
                f"--model {r.m.label}: a disparity output has no metric scale"
            )
    kind = "tae" if args.tae else "main"
    for seq in seqs:
        tag = f"{spec.name}/tae/{seq.name}" if args.tae else f"{spec.name}/{seq.name}"
        gt, views = BE._prepare_sequence(spec, seq, args.image_size, device)
        if not 0 <= args.ref_frame < len(gt):
            raise ValueError(
                f"--ref-frame {args.ref_frame}: {seq.name} has {len(gt)} frames"
            )
        # one draw, every model: deterministic in (tag, density, seed)
        realized = BE.attach_sparse_depth(
            views, args.density, tag, args.patch_size, args.seed, device
        )
        _, sparse_mask = BE._sparse_arrays(views)
        held_out = model_to_gt(
            sparse_mask, tuple(gt.shape[1:]), views[0].get("gt_box"), mask=True
        )
        frames = BB.read_frames(seq)
        xs = column_px(args.x, gt.shape[2])
        rgb = gt_grid_rgb(frames, seq, spec, gt.shape[1:])
        d = scene_dir(args.out, spec.name, args.tae, seq.name)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "scene.npz",
            xs=np.array(xs),
            rgb=columns(rgb, xs),
            gt=columns(gt, xs).astype(np.float32),
            valid=columns(P.gt_valid_mask(gt, spec.max_depth), xs),
            ref=rgb[args.ref_frame],
            ref_frame=np.array(args.ref_frame),
        )
        for r in runners:
            pred_gt = r.predict(seq, spec, gt, views, frames, kind)
            if pred_gt.shape != gt.shape:
                raise ValueError(f"{r.m.label} {tag}: {pred_gt.shape} vs GT {gt.shape}")
            aligned, abs_rel = align(
                pred_gt, r.output, gt, args.protocol, spec.max_depth, held_out
            )
            np.savez_compressed(d / f"{r.m.label}.npz", pred=columns(aligned, xs))
            meta = {
                "label": r.m.label,
                "model": f"{r.m.kind}:{r.m.source}",
                "weights": r.weights,
                "output": r.output,
                "abs_rel": abs_rel,
                "realized_density": realized,
                "frames": int(len(gt)),
                **{k: getattr(args, k) for k in DRAW_KEYS},
                "xs": xs,
                "written": datetime.datetime.now().isoformat(timespec="seconds"),
                "gpu": dev["gpu"],
                "code_files": PROV.loaded_code(),
            }
            with open(d / f"{r.m.label}.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(
                f"[slit] {r.m.label} {tag}: {args.protocol} AbsRel {abs_rel:.4f}",
                flush=True,
            )
        del views
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# render (CPU)
# ---------------------------------------------------------------------------
def load_scene(d: Path) -> tuple[dict, list[tuple[dict, np.ndarray]]]:
    """(scene arrays, [(model meta, pred columns [K, H, T])] in extraction
    order). Every model must share the scene's columns and one protocol and
    density, or the panels would not compare."""
    with np.load(d / "scene.npz") as z:
        scene = {k: z[k] for k in z.files}
    models = []
    for js in sorted(d.glob("*.json")):
        with open(js) as f:
            meta = json.load(f)
        with np.load(js.with_suffix(".npz")) as z:
            models.append((meta, z["pred"]))
    models.sort(key=lambda mp: mp[0]["written"])
    for key in DRAW_KEYS:
        seen = {meta.get(key) for meta, _ in models}
        if len(seen) > 1:
            raise ValueError(
                f"{d}: models extracted with different {key}: {sorted(seen)}"
            )
    for meta, pred in models:
        if list(meta["xs"]) != scene["xs"].tolist() or pred.shape != scene["gt"].shape:
            raise ValueError(
                f"{d}: {meta['label']} was extracted at columns {meta['xs']}, the scene "
                f"at {scene['xs'].tolist()}; re-extract it"
            )
    return scene, models


def _stretch(img: np.ndarray, px_per_frame: int) -> np.ndarray:
    """Widen a profile [H, T(, 3)] to px_per_frame pixels per frame."""
    return np.repeat(img, px_per_frame, axis=1)


def render_scene(
    d: Path, out: Path, px_per_frame: int, error: bool, cmap: str
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eval.spot_gif import _colorize

    scene, models = load_scene(d)
    ref = scene["ref"]
    H, W = ref.shape[:2]
    T = scene["gt"].shape[2]
    paths = []
    for k, x in enumerate(scene["xs"].tolist()):
        gt, valid = scene["gt"][k], scene["valid"][k]
        lo, hi = display_range(gt, valid)
        top = [_stretch(scene["rgb"][k], px_per_frame)]
        top.append(
            _stretch(
                _colorize(1.0 / np.where(valid, gt, 1.0), valid, lo, hi, cmap),
                px_per_frame,
            )
        )
        titles = ["RGB", "GT"]
        bottom = []
        for meta, pred in models:
            p = pred[k]
            ok = np.isfinite(p) & (p > 0)
            disp = 1.0 / np.where(ok, p, 1.0)
            top.append(_stretch(_colorize(disp, ok, lo, hi, cmap), px_per_frame))
            titles.append(f"{meta['label']}  AbsRel {meta['abs_rel']:.3f}")
            m = ok & valid
            lr = np.log2(np.where(m, p, 1.0) / np.where(m, gt, 1.0))
            bottom.append(
                _stretch(
                    _colorize(lr, m, -LOG2_ERR_MAX, LOG2_ERR_MAX, "coolwarm"),
                    px_per_frame,
                )
            )
        n_rows = 2 if error and bottom else 1
        widths = [W] + [T * px_per_frame] * len(top)
        dpi = 100
        fig_w = (sum(widths) + GAP * len(widths)) / dpi
        fig_h = (TITLE_PX + n_rows * (H + ROW_PAD_PX)) / dpi
        fig, axes = plt.subplots(
            n_rows,
            len(widths),
            figsize=(fig_w, fig_h),
            dpi=dpi,
            gridspec_kw={"width_ratios": widths},
            squeeze=False,
        )
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].imshow(ref)
        axes[0, 0].axvline(x, color="red", linewidth=1.5)
        axes[0, 0].set_title(f"frame {int(scene['ref_frame'])}, x = {x}", fontsize=9)
        for j, (img, title) in enumerate(zip(top, titles), start=1):
            axes[0, j].imshow(img, aspect="auto")
            axes[0, j].set_title(title, fontsize=9)
        if n_rows == 2:
            axes[1, 2].text(
                0.5,
                0.5,
                "log2(pred / GT)\nblue: too near\nred: too far\n(range ±1)",
                ha="center",
                va="center",
                fontsize=9,
                transform=axes[1, 2].transAxes,
            )
            for j, img in enumerate(bottom, start=3):
                axes[1, j].imshow(img, aspect="auto")
        proto = models[0][0]["protocol"] if models else "-"
        dens = models[0][0]["density"] if models else "-"
        fig.suptitle(
            f"{d.name}: column x = {x} over {T} frames (time left to right)  |  "
            f"{proto} alignment, density {dens}  |  disparity, GT range",
            fontsize=10,
        )
        # fixed pixel margins (tight_layout cannot shrink panels this narrow):
        # TITLE_PX for the suptitle, ROW_PAD_PX above each row for its titles
        fig.subplots_adjust(
            left=0.002,
            right=0.998,
            bottom=0.005,
            # the first row's panel titles sit above its axes, below the title
            top=1 - (TITLE_PX + ROW_PAD_PX) / (fig_h * dpi),
            wspace=GAP * len(widths) / sum(widths),
            hspace=ROW_PAD_PX / H,
        )
        path = out / f"{d.name}_x{x}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def render(args) -> None:
    dirs = sorted(p.parent for p in args.out.glob("*/scene.npz"))
    if args.scenes:
        dirs = [d for d in dirs if d.name in set(args.scenes)]
    if not dirs:
        raise FileNotFoundError(f"no extracted scene under {args.out}")
    for d in dirs:
        for path in render_scene(d, args.out, args.px_per_frame, args.error, args.cmap):
            print(f"[slit] wrote {path}", flush=True)


# ---------------------------------------------------------------------------
def validate(args) -> None:
    """Every check that needs no GPU, before any model is loaded."""
    if args.command == "render":
        if args.px_per_frame < 1:
            raise ValueError("--px-per-frame must be >= 1")
        return
    if args.dataset not in SPECS:
        raise KeyError(f"unknown dataset {args.dataset!r}; known: {sorted(SPECS)}")
    if args.tae and not SPECS[args.dataset].tae_json:
        raise ValueError(f"{args.dataset} has no TAE manifest")
    if not args.x or any(not 0.0 <= f <= 1.0 for f in args.x):
        raise ValueError(
            "--x takes column positions as fractions of the width, in [0, 1]"
        )
    if args.max_sequences < 1 and not args.sequences:
        raise ValueError("--max-sequences must be >= 1 (or name --sequences)")
    models = [parse_model(s) for s in args.model]
    labels = [m.label for m in models]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate --model labels: {labels}")
    for m in models:
        if m.kind in ("ckpt", "base", "dump") and not Path(m.source).is_dir():
            raise FileNotFoundError(f"--model {m.label}: no directory {m.source}")
        if m.kind == "arm":
            from eval.baselines import arms as A

            if m.source not in A.ARMS:
                raise KeyError(
                    f"--model {m.label}: unknown arm {m.source!r}; {sorted(A.ARMS)}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="GPU: run the models, save the slit columns")
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--dataset", required=True)
    ex.add_argument(
        "--tae", action="store_true", help="the TAE manifest (ScanNet: 192-frame clips)"
    )
    ex.add_argument("--sequences", nargs="*", default=[])
    ex.add_argument("--max-sequences", type=int, default=1)
    ex.add_argument("--model", action="append", required=True, help="LABEL=KIND:SOURCE")
    ex.add_argument(
        "--x", type=float, nargs="+", default=[0.5], help="columns, fraction of width"
    )
    ex.add_argument(
        "--ref-frame", type=int, default=0, help="the frame drawn with the red line"
    )
    ex.add_argument("--protocol", choices=PROTOCOLS, default="published")
    # the benchmark's draw: same density / seed / patch as the table's row
    ex.add_argument("--density", type=float, default=0.05)
    ex.add_argument("--seed", type=int, default=42)
    ex.add_argument("--patch-size", type=int, default=14)
    ex.add_argument("--image-size", type=int, default=518)
    ex.add_argument("--bench-root", type=Path, default=BENCH_ROOT)
    ex.add_argument(
        "--allow-close", action="store_true", help="admit an arm the record marks close"
    )
    re_ = sub.add_parser("render", help="CPU: draw the figures from extracted scenes")
    re_.add_argument("--out", type=Path, required=True)
    re_.add_argument(
        "--scenes", nargs="*", default=[], help="scene directory names; default all"
    )
    re_.add_argument("--px-per-frame", type=int, default=3)
    re_.add_argument(
        "--no-error", dest="error", action="store_false", help="omit the log2 error row"
    )
    re_.add_argument("--cmap", default="inferno")
    args = ap.parse_args()
    validate(args)
    if args.command == "extract":
        extract(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
