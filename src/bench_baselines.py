"""Baseline arms: fetch released weights, and reproduce each arm's own paper
through our loaders and our scoring before it may be used for anything else.

    python src/bench_baselines.py fetch --arm vda          # login node, weights -> Lustre
    python src/bench_baselines.py reproduce --arm vda      # GPU job (experiments/baselines/reproduce.sh)
    python src/bench_baselines.py reproduce --arm vda --max-sequences 2   # smoke test, record untouched
    python src/bench_baselines.py reproduce --arm vda --datasets scannet_500   # one dataset's targets
    python src/bench_baselines.py benchmark --arm vda     # GPU: OUR protocols, verified arms only
    python src/bench_baselines.py status

`reproduce` runs every runnable target of the arm in eval/baselines/
reproduction.json -- the datasets, protocol and settings its paper reports,
nothing else -- writes <out>/reproduce_<arm>.json with the per-sequence rows,
and on a complete run stores the measured values, the commit and the job id
in the record, which recomputes the arm's status (record.apply_measurements).
The gate itself is in arms.build_arm.

`benchmark` scores a VERIFIED arm under the benchmark's own protocols
(bench_eval: published / sparse_aligned / metric per density, both TAEs) with
the identical sparse-depth draws our model gets -- the pixels a RGB-only arm
never sees and is fitted on post hoc. One row per (dataset, sequence,
density), mode "stream" for a causal arm and "offline" otherwise, written
like bench_eval's bench_results.json so the two sit side by side.

`fetch` / `reproduce` / `status` are deliberately independent of bench_eval
(accelerate, streamvggt, the training stack): Depth Any Video runs from its
own venv, which has none of it. `benchmark` imports it lazily.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose

from eval import arkit_upsampling as ARKIT
from eval import protocols as P
from eval.baselines import arms as A
from eval.baselines import record as R
from eval.vda_benchmark import (
    SPECS,
    BenchSpec,
    Sequence,
    crop_slices,
    gt_stack,
    load_manifest,
    resize_to_gt,
)

BENCH_ROOT = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench")
OUT_ROOT = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/baselines")


def read_frames(seq: Sequence) -> np.ndarray:
    """[S,H,W,3] uint8 RGB, exactly the manifest's images: the arm does its
    own resizing, as its released inference script does."""
    frames = []
    for fr in seq.frames:
        if not fr.image.is_file():
            raise FileNotFoundError(f"missing benchmark image {fr.image}")
        frames.append(np.asarray(exif_transpose(Image.open(fr.image)).convert("RGB")))
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"{seq.dataset}/{seq.name}: frame shapes differ: {shapes}")
    return np.stack(frames)


def register_to_gt(
    pred: np.ndarray,
    seq: Sequence,
    spec: BenchSpec,
    frame_hw: tuple[int, int],
    gt_hw: tuple[int, int],
) -> np.ndarray:
    """Prediction -> GT grid, as VDA's eval does it: bilinear resize to the
    cropped GT; for the TAE manifest, whose frames are uncropped, first to the
    frame and through the protocol crop (their --hard_crop). The pixel crop
    means the same field of view only if the frame is the uncropped GT's
    size, so that is checked (as load_rgb does for our model)."""
    if seq.rgb_uncropped:
        ys, xs = crop_slices(spec.crop)
        rows, cols = range(frame_hw[0])[ys], range(frame_hw[1])[xs]
        if (len(rows), len(cols)) != tuple(gt_hw):
            raise ValueError(
                f"{seq.dataset}/{seq.name}: frames of {frame_hw} cropped by {spec.crop} give "
                f"{(len(rows), len(cols))}, but the cropped GT is {tuple(gt_hw)}; the frames "
                "must be at the uncropped GT size for the crop to mean the same pixels"
            )
        pred = resize_to_gt(pred, frame_hw)[:, ys, xs]
    if pred.shape[1:] != gt_hw:
        pred = resize_to_gt(pred, gt_hw)
    return pred


def score_main(arm, pred_gt: np.ndarray, gt: np.ndarray, keys: set) -> dict:
    """{(protocol, max_depth): FrameMetrics dict} for one sequence."""
    out = {}
    for protocol, max_depth in sorted(keys):
        if protocol == "published":
            m, _ = P.published_metrics(
                P.as_disparity(pred_gt, arm.info.output), gt, max_depth
            )
        elif protocol in ("first_frame", "ovda_global"):
            m = P.ovda_aligned_metrics(
                P.as_disparity(pred_gt, arm.info.output),
                gt,
                max_depth,
                protocol == "first_frame",
            )
        elif protocol == "metric":
            if arm.info.output != "depth":
                raise ValueError(f"{arm.info.name} is not metric: no `metric` protocol")
            m = P.metric_metrics(pred_gt, gt, max_depth)
        else:
            raise ValueError(f"unknown protocol {protocol!r}")
        out[(protocol, max_depth)] = m.as_dict()
    return out


def score_tae(arm, pred_gt: np.ndarray, gt: np.ndarray, seq, spec) -> float:
    """VDA's TAE on the published-aligned depth, K as their manifest has it."""
    aligned = P.vda_align_disparity(
        P.as_disparity(pred_gt, arm.info.output), gt, spec.max_depth
    )
    Ks = [fr.K for fr in seq.frames]
    if any(k is None for k in Ks):
        raise ValueError(f"{spec.name}/{seq.name}: TAE manifest frames lack K")
    poses = [
        np.full((4, 4), np.nan) if fr.pose is None else fr.pose for fr in seq.frames
    ]
    return P.tae_vda(aligned, Ks, poses)


def run_video_dataset(
    arm, dataset: str, targets: list[dict], root: Path, max_sequences: int
) -> tuple[dict[str, float], list[dict]]:
    """Measured value per target id on one benchmark dataset, plus the
    per-sequence rows behind them."""
    spec = SPECS[dataset]
    main_keys = {
        (t["protocol"], float(t.get("max_depth", spec.max_depth)))
        for t in targets
        if t["protocol"] != "tae"
    }
    rows: list[dict] = []
    passes = []
    if main_keys:
        passes.append((False, load_manifest(root, spec)))
    if any(t["protocol"] == "tae" for t in targets):
        # every listed frame: VDA's infer.py runs the model on the whole TAE
        # clip (192 frames) and its eval scores frames [10:180] of the result
        passes.append((True, load_manifest(root, spec, tae=True, tae_slice=False)))
    for tae, seqs in passes:
        if max_sequences:
            seqs = seqs[:max_sequences]
        for seq in seqs:
            gt = gt_stack(seq, spec)
            frames = read_frames(seq)
            pred = arm.predict(frames)
            if len(pred) != len(frames):
                raise ValueError(
                    f"{arm.info.name} returned {len(pred)} frames for {len(frames)}"
                )
            pred_gt = register_to_gt(pred, seq, spec, frames.shape[1:3], gt.shape[1:])
            row = {"dataset": dataset, "sequence": seq.name, "frames": len(frames)}
            if tae:
                lo, hi = spec.tae_range
                seq.frames = seq.frames[lo:hi]
                row["tae"] = {
                    "tae_vda": score_tae(arm, pred_gt[lo:hi], gt[lo:hi], seq, spec)
                }
            else:
                for (protocol, max_depth), m in score_main(
                    arm, pred_gt, gt, main_keys
                ).items():
                    row[f"{protocol}@{max_depth:g}"] = m
            rows.append(row)
            print(
                f"[repro] {arm.info.name} {dataset}/{seq.name}: {len(frames)} frames",
                flush=True,
            )
    measured = {}
    for t in targets:
        key = (
            "tae"
            if t["protocol"] == "tae"
            else f"{t['protocol']}@{float(t.get('max_depth', spec.max_depth)):g}"
        )
        measured[t["id"]] = P.finite_mean(
            [r[key][t["metric"]] for r in rows if key in r]
        )
    return measured, rows


def run_arkit(
    arm, targets: list[dict], root: Path, max_sequences: int
) -> tuple[dict[str, float], list[dict]]:
    """PromptDA's ARKitScenes table: one prediction per frame, resized to each
    evaluated size the targets name; L1 / RMSE, mean over images
    (eval/arkit_upsampling.py)."""
    datasets = sorted({t["dataset"] for t in targets})
    sky = ARKIT.sky_directions(root)
    zips = ARKIT.videos(root)
    if max_sequences:
        zips = zips[:max_sequences]
    rows = []
    for zp in zips:
        for s in ARKIT.samples(zp, sky[zp.stem]):
            pred = arm.predict(s.rgb[None], (s.prompt[None], s.prompt[None] > 0))[0]
            for dataset in datasets:
                gt = s.gt_at(ARKIT.SIZES[dataset])
                at_gt = cv2.resize(
                    pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
                )
                l1, rmse = ARKIT.l1_rmse(at_gt, gt)
                rows.append(
                    {"dataset": dataset, "sequence": s.name, "l1": l1, "rmse": rmse}
                )
        print(f"[repro] {arm.info.name} arkit_upsampling/{zp.stem} done", flush=True)
    measured = {
        t["id"]: P.finite_mean(
            [r[t["metric"]] for r in rows if r["dataset"] == t["dataset"]]
        )
        for t in targets
    }
    return measured, rows


def _commit() -> str:
    """HEAD, plus a hash of the uncommitted CODE changes (src/,
    datasets_preprocess/, experiments/, the record itself excluded, since
    every run rewrites it): two runs with the same string ran the same code,
    which is what `verified` requires of its measurements."""
    repo = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "HEAD",
                "--",
                "src",
                "datasets_preprocess",
                "experiments",
                f":!{R.RECORD_PATH.relative_to(repo)}",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "unknown"
    if not diff.strip():
        return head
    return f"{head}+{hashlib.sha1(diff.encode()).hexdigest()[:8]}"


def reproduce(args) -> None:
    record = R.load(args.record)
    entry = R.arm_entry(record, args.arm)
    targets = R.runnable_targets(entry)
    if args.datasets:
        unknown = sorted(set(args.datasets) - {t["dataset"] for t in targets})
        if unknown:
            raise SystemExit(f"{args.arm}: no target on {unknown} in the record")
        targets = [t for t in targets if t["dataset"] in args.datasets]
    if not targets:
        raise SystemExit(f"{args.arm}: no runnable target in the record")
    device = torch.device(args.device)
    arm = A.build_arm(args.arm, device, sorted(R.paper_uses(entry)), record)

    measured: dict[str, float] = {}
    rows: list[dict] = []
    arkit = [t for t in targets if t["dataset"] in ARKIT.SIZES]
    if arkit:
        measured, rows = run_arkit(arm, arkit, args.arkit_root, args.max_sequences)
    for dataset in sorted({t["dataset"] for t in targets} - set(ARKIT.SIZES)):
        ts = [t for t in targets if t["dataset"] == dataset]
        m, r = run_video_dataset(arm, dataset, ts, args.bench_root, args.max_sequences)
        measured.update(m)
        rows.extend(r)

    tol = record["tolerance"]
    print(f"\n[repro] {args.arm}: published vs measured (printed unit)")
    for t in targets:
        value = measured[t["id"]] * t["scale"]
        verdict = "ok" if R.within(t, measured[t["id"]], tol) else "MISS"
        gate = "gate" if t["gate"] else "info"
        print(
            f"[repro]   {t['id']:<24} {t['published']:>8} {value:>10.4f}  "
            f"+-{R.tolerance(t['published'], tol):.4f}  {verdict}  ({gate})"
        )

    run = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _commit(),
        "job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "host": os.uname().nodename,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "_" + "_".join(sorted(args.datasets)) if args.datasets else ""
    out = args.out / f"reproduce_{args.arm}{tag}.json"
    with open(out, "w") as f:
        json.dump({"run": run, "measured": measured, "rows": rows}, f, indent=2)
    print(f"[repro] wrote {out}")

    if args.max_sequences:
        print("[repro] --max-sequences set: partial run, record NOT updated")
        return
    bad = sorted(k for k, v in measured.items() if not np.isfinite(v))
    if bad:
        raise SystemExit(f"[repro] non-finite measurements {bad}: record NOT updated")
    # re-read: another arm's job may have written the record since this one
    # loaded it, and saving the stale copy would erase that arm's result
    record = R.load(args.record)
    status = R.apply_measurements(record, args.arm, measured, run)
    R.save(record, args.record)
    print(f"[repro] {args.arm}: status -> {status} ({args.record})")


def benchmark(args) -> None:
    import bench_eval as BE  # the training stack: metrics venv only

    record = R.load(args.record)
    protocols = ("published", "sparse_aligned", "metric", "tae")
    uses = [
        (d, p) for d in set(args.datasets) | set(args.tae_datasets) for p in protocols
    ]
    device = torch.device(args.device)
    arm = A.build_arm(args.arm, device, uses, record)  # raises unless verified
    mode = BE.MODE_STREAM if arm.info.causal else "offline"
    cfg = BE.BenchmarkCfg(
        root=args.bench_root,
        datasets=tuple(args.datasets),
        densities=tuple(args.densities),
        tae_datasets=tuple(args.tae_datasets),
        image_size=args.image_size,
        max_sequences=args.max_sequences,
        clouds_per_dataset=0,
        spot=False,
    ).validate()
    missing = BE.bench_root_ok(cfg.root, cfg.datasets, cfg.tae_datasets)
    if missing:
        raise FileNotFoundError(f"benchmark manifests missing: {missing}")

    def _pred(raw, seq, spec, frame_hw, gt_hw, native_hw) -> BE.Prediction:
        """The arm's output as bench_eval scores it: on the GT grid for
        published / metric / TAE, and at OUR model resolution (where the
        sparse pixels live) for the sparse fit."""
        at_gt = register_to_gt(raw, seq, spec, frame_hw, gt_hw)
        native = resize_to_gt(raw, native_hw)
        S = len(raw)
        return BE.Prediction(
            native,
            np.zeros_like(native),
            np.full((S, 3, 4), np.nan),
            np.full((S, 3, 3), np.nan),
            output=arm.info.output,
            at_gt=at_gt,
        )

    def _predict(frames, views):
        """A prompted arm gets this draw's sparse depth (at our model
        resolution, as our model does); a RGB-only arm gets nothing."""
        if arm.info.prompted:
            return arm.predict(frames, BE._sparse_arrays(views))
        return arm.predict(frames)

    rows: list[dict] = []
    tae_rows: list[dict] = []
    for name in cfg.datasets:
        spec = SPECS[name]
        seqs = load_manifest(cfg.root, spec)
        if cfg.max_sequences:
            seqs = seqs[: cfg.max_sequences]
        for seq in seqs:
            tag = f"{spec.name}/{seq.name}"
            gt, views = BE._prepare_sequence(spec, seq, cfg.image_size, device)
            frames = read_frames(seq)
            native_hw = tuple(views[0]["img"].shape[-2:])
            pred = None
            main_tae = (
                spec.name in cfg.tae_datasets
                and spec.tae_json is None
                and BE.has_cameras(seq)
            )
            for density in cfg.densities:
                realized = BE.attach_sparse_depth(
                    views, density, tag, args.patch_size, args.seed, device
                )
                # a RGB-only arm's output does not depend on the draw: one inference
                if pred is None or arm.info.prompted:
                    raw = _predict(frames, views)
                    pred = _pred(
                        raw, seq, spec, frames.shape[1:3], gt.shape[1:], native_hw
                    )
                row, aligned = BE.score_sequence(
                    spec, seq, gt, views, pred, mode, density, realized
                )
                rows.append(row)
                if main_tae:
                    tae_rows.append(
                        BE.score_tae_sequence(
                            spec, seq, gt, pred, mode, density, realized, aligned
                        )
                    )
            del views
            print(f"[bench] {arm.info.name} {tag}: {len(frames)} frames", flush=True)
        if spec.name in cfg.tae_datasets and spec.tae_json:
            # the sliced 170 frames, exactly what our model is run on here
            # (`reproduce` feeds all 192 for parity with VDA's infer.py)
            tseqs = load_manifest(cfg.root, spec, tae=True)
            if cfg.max_sequences:
                tseqs = tseqs[: cfg.max_sequences]
            for seq in tseqs:
                if not BE.has_cameras(seq):
                    continue
                gt, views = BE._prepare_sequence(spec, seq, cfg.image_size, device)
                frames = read_frames(seq)
                native_hw = tuple(views[0]["img"].shape[-2:])
                pred = None
                for density in cfg.densities:
                    realized = BE.attach_sparse_depth(
                        views,
                        density,
                        f"{spec.name}/tae/{seq.name}",
                        args.patch_size,
                        args.seed,
                        device,
                    )
                    if pred is None or arm.info.prompted:
                        raw = _predict(frames, views)
                        pred = _pred(
                            raw, seq, spec, frames.shape[1:3], gt.shape[1:], native_hw
                        )
                    tae_rows.append(
                        BE.score_tae_sequence(
                            spec, seq, gt, pred, mode, density, realized
                        )
                    )
                del views
                print(
                    f"[bench] {arm.info.name} {spec.name}/tae/{seq.name}: {len(frames)} frames",
                    flush=True,
                )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = BE.aggregate(rows, tae_rows)
    out_dir = args.out / f"bench_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bench_results.json"
    with open(out, "w") as f:
        json.dump(
            {
                "arm": args.arm,
                "mode": mode,
                "config": {
                    k: (
                        list(v)
                        if isinstance(v, tuple)
                        else str(v)
                        if isinstance(v, Path)
                        else v
                    )
                    for k, v in vars(cfg).items()
                },
                "sparse_seed": args.seed,
                "patch_size": args.patch_size,
                "commit": _commit(),
                "job_id": os.environ.get("SLURM_JOB_ID", "none"),
                "aggregate": result,
                "rows": rows,
                "tae_rows": tae_rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[bench] wrote {out}")
    for k in sorted(result):
        if k.endswith(
            (
                "published_abs_rel",
                "sparse_aligned_abs_rel",
                "metric_abs_rel",
                "tae_vda",
                "tae_ours",
            )
        ):
            print(f"[bench] {k}: {result[k]:.4f}")


def status(args) -> None:
    record = R.load(args.record)
    for name, entry in record["arms"].items():
        gating = [t for t in entry["targets"] if t["gate"]]
        done = sum(1 for t in gating if t.get("within_tolerance"))
        print(
            f"{name:<11} {entry['status']:<12} {done}/{len(gating)} gating targets within tolerance"
        )
        if entry["status_note"]:
            print(f"{'':<11} {entry['status_note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", type=Path, default=R.RECORD_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download an arm's released weights to Lustre")
    fetch.add_argument("--arm", required=True, choices=sorted(A.ARMS))
    rep = sub.add_parser("reproduce", help="run the arm's paper targets")
    rep.add_argument("--arm", required=True, choices=sorted(A.ARMS))
    rep.add_argument("--device", default="cuda")
    rep.add_argument("--bench-root", type=Path, default=BENCH_ROOT)
    rep.add_argument("--arkit-root", type=Path, default=ARKIT.ROOT)
    rep.add_argument("--out", type=Path, default=OUT_ROOT)
    rep.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="only these datasets of the arm's targets (default: all); the other "
        "targets keep their recorded values, so an arm can be split over jobs",
    )
    rep.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="cap on sequences per dataset (smoke test; the record is not updated)",
    )
    bench = sub.add_parser("benchmark", help="score a VERIFIED arm under our protocols")
    bench.add_argument("--arm", required=True, choices=sorted(A.ARMS))
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--bench-root", type=Path, default=BENCH_ROOT)
    bench.add_argument("--out", type=Path, default=OUT_ROOT)
    bench.add_argument(
        "--datasets",
        nargs="+",
        default=["sintel", "scannet", "kitti", "bonn_all", "nyuv2"],
    )
    bench.add_argument(
        "--tae-datasets", nargs="*", default=["sintel", "scannet", "kitti", "bonn_all"]
    )
    bench.add_argument("--densities", nargs="+", type=float, default=[0.01, 0.05, 0.4])
    bench.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="OUR model's input size: where the sparse pixels are drawn",
    )
    bench.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="sparse-depth patch (depth_cond.sim_patch_size)",
    )
    bench.add_argument(
        "--seed",
        type=int,
        default=42,
        help="sparse-depth seed (bench_checkpoint.py --seed)",
    )
    bench.add_argument("--max-sequences", type=int, default=0)
    sub.add_parser("status", help="print the record's status per arm")
    args = parser.parse_args()
    if args.command == "fetch":
        print(A.fetch_weights(args.arm, R.load(args.record)))
    elif args.command == "reproduce":
        reproduce(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
