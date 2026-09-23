"""End-of-training video-depth benchmark: the Video Depth Anything datasets
(Sintel, ScanNet, KITTI, Bonn, NYUv2 stills) scored under three protocols,
over a sweep of sparse-depth densities, on the per-frame KV-cache (streaming) path.

Streaming only, on purpose. StreamVGGT.forward -- the "offline" full-sequence
pass -- applies the same causal mask the cache reproduces incrementally, so
for this model the two are one function up to kernel numerics; there is no
streaming-vs-offline gap to measure (unlike VDA, whose offline model attends
bidirectionally). The full forward also materialises an [S*P, S*P] mask
(~24 GB at 110 frames, 58 GB at the 170-frame TAE sequences) and would OOM.

Called once from finetune_depth.run() after the streaming eval (and in the
--epochs 0 pure-eval path), never per epoch. Everything is logged under
"final_bench/<dataset>/<mode>/d<density%>/<protocol>_<metric>" next to the
val_* / final_stream series, and written to <output_dir>/bench_results.json
with the per-sequence rows behind every mean.

Protocol definitions live in eval.protocols; dataset constants and loading
in eval.vda_benchmark. This module only orchestrates: sparse depth simulation,
inference, sharding over ranks, aggregation, and the point-cloud snapshots.

Cameras: TAE and the cloud snapshots use the GT cameras the preparer writes
into every manifest (benchmark_cameras.py); the model's predicted cameras
are stored in the snapshots but never scored -- the camera head is frozen
and reads fine-tuned tokens, so its output is not a result.

Sparse depth: one TUBE_MASK patch mask per sequence (the same pixels in every
frame, like a static sensor pattern, so no mask flicker leaks into the TAE),
drawn from a seed fixed by (dataset, sequence, density) -- the identical
pixel set for every mode, every checkpoint and every baseline arm.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

from dust3r.inference import loss_of_one_batch
from eval import protocols as P
from eval.vda_benchmark import (
    SPECS,
    BenchSpec,
    Sequence,
    bench_root_ok,
    build_views,
    gt_stack,
    load_manifest,
    resize_to_gt,
    scaled_intrinsics,
)
from streamvggt.depth_cond.config import SparseSimMode
from streamvggt.depth_cond.sparse import simulate_sparse_depth
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from train_utils import to_primitive

_BENCH_ROOT = Path("/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench")

PROTOCOLS = ("published", "sparse_aligned", "metric")
MODE_STREAM = "stream"  # the only mode; kept in the keys so a baseline arm's offline rows can sit next to it


@dataclass
class BenchmarkCfg:
    """The end-of-training benchmark. Not part of the experiment identity:
    it reads the trained weights and changes nothing upstream of them."""

    enabled: bool = False
    root: Path = _BENCH_ROOT
    """Benchmark tree from datasets_preprocess/prepare_vda_benchmark.py."""
    datasets: tuple[str, ...] = ("sintel", "scannet", "kitti", "bonn", "nyuv2")
    densities: tuple[float, ...] = (0.01, 0.05, 0.40)
    """Sparse depth densities (fraction of patches visible) swept per sequence. 5%
    is the training density; 40% is SPOT's real sensor."""
    tae_datasets: tuple[str, ...] = ("scannet",)
    """Datasets scored for TAE (both definitions), per density. ScanNet uses
    VDA's TAE manifest (20 x 170 consecutive frames, its own pass); any other
    video dataset is scored on the predictions of its main pass, using the
    GT cameras the preparer attached. ScanNet only by default: it is the one
    dataset VDA reports TAE on, so the others have no baseline row yet."""
    clouds_per_dataset: int = 2
    """Sequences per dataset snapshotted as re-renderable .npz (+ .glb)."""
    cloud_frames: int = 32
    cloud_density: float = 0.05
    """The density whose streaming pass is snapshotted; must be swept."""
    image_size: int = 518
    """Long side the model runs at (dust3r load_images_for_eval convention)."""
    max_sequences: int = 0
    """Cap on sequences per dataset, for smoke tests. 0 = all."""

    def validate(self) -> "BenchmarkCfg":
        unknown = [d for d in self.datasets if d not in SPECS]
        if unknown:
            raise ValueError(
                f"bench.datasets has unknown entries {unknown}; known: {list(SPECS)}"
            )
        if not self.datasets:
            raise ValueError("bench.datasets is empty")
        if not self.densities:
            raise ValueError("bench.densities is empty")
        for d in self.densities:
            if not 0.0 < d <= 1.0:
                raise ValueError(f"bench.densities entries must be in (0, 1], got {d}")
        bad_tae = [d for d in self.tae_datasets if d not in SPECS or not SPECS[d].video]
        if bad_tae:
            raise ValueError(
                f"bench.tae_datasets must be video datasets from {list(SPECS)}, got {bad_tae}"
            )
        if self.clouds_per_dataset > 0 and self.cloud_density not in self.densities:
            raise ValueError(
                f"bench.cloud_density {self.cloud_density} is not one of the swept "
                f"densities {self.densities}"
            )
        if self.cloud_frames <= 0:
            raise ValueError(
                f"bench.cloud_frames must be positive, got {self.cloud_frames}"
            )
        if self.image_size % 14 != 0:
            raise ValueError(
                f"bench.image_size must be a multiple of 14, got {self.image_size}"
            )
        if self.max_sequences < 0:
            raise ValueError(
                f"bench.max_sequences must be >= 0, got {self.max_sequences}"
            )
        return self


def density_key(density: float) -> str:
    """0.05 -> 'd5', 0.4 -> 'd40', 0.01 -> 'd1'."""
    return f"d{density * 100:g}"


# ---------------------------------------------------------------------------
# sparse depth + inference
# ---------------------------------------------------------------------------
def _sparse_seed(base_seed: int, tag: str, density: float) -> int:
    return (
        base_seed * 1_000_003 + zlib.crc32(tag.encode()) + int(round(density * 10_000))
    ) % (2**31)


def attach_sparse_depth(
    views: list[dict],
    density: float,
    tag: str,
    patch_size: int,
    base_seed: int,
    device: torch.device,
) -> float:
    """Replace the views' sparse depth with a fresh TUBE_MASK draw at
    `density`, seeded by (tag, density), and return the realized density
    (GT holes lower it below the requested value)."""
    for v in views:
        v.pop("sparse_depth", None)
        v.pop("sparse_depth_mask", None)
    seed = _sparse_seed(base_seed, tag, density)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        simulate_sparse_depth(
            views,
            mode=SparseSimMode.TUBE_MASK,
            patch_size=patch_size,
            mask_ratio=1.0 - density,
        )
    masks = torch.stack([v["sparse_depth_mask"][0] for v in views])
    return float(masks.to(torch.float32).mean())


@dataclass
class Prediction:
    depth: np.ndarray  # [S,h,w] metres, model resolution
    conf: np.ndarray  # [S,h,w]
    w2c: np.ndarray  # [S,3,4] predicted world->cam
    K: np.ndarray  # [S,3,3] predicted intrinsics at model resolution


def predict(net, accelerator: Accelerator, views: list[dict]) -> Prediction:
    """One streaming pass over a sequence: the per-frame KV-cache path
    (MetricStreamVGGT.inference via loss_of_one_batch(inference=True)),
    i.e. deployment."""
    with torch.no_grad():
        result = loss_of_one_batch(
            views,
            net,
            None,
            accelerator,
            inference=True,
            symmetrize_batch=False,
            use_amp=True,
        )
    preds = result["pred"]
    depth = (
        torch.stack([p["depth"][0] for p in preds]).squeeze(-1).float().cpu().numpy()
    )
    conf = torch.stack([p["depth_conf"][0] for p in preds]).float().cpu().numpy()
    h, w = depth.shape[-2:]
    enc = torch.stack(
        [p["camera_pose"].detach().float().cpu() for p in preds], dim=1
    )  # [1,S,9]
    extri, intri = pose_encoding_to_extri_intri(enc, (h, w))
    del result, preds
    return Prediction(depth, conf, extri[0].numpy(), intri[0].numpy())


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _sparse_arrays(views: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    depth = torch.stack([v["sparse_depth"][0] for v in views]).float().cpu().numpy()
    mask = torch.stack([v["sparse_depth_mask"][0] for v in views]).bool().cpu().numpy()
    return depth, mask


def score_sequence(
    spec: BenchSpec,
    seq: Sequence,
    gt: np.ndarray,
    views: list[dict],
    pred: Prediction,
    mode: str,
    density: float,
    realized: float,
) -> tuple[dict, np.ndarray]:
    """One row: the three protocols for one (sequence, mode, density), plus
    the published-aligned depth, which score_tae_sequence takes so the
    resize + per-video least squares run once per (sequence, density)."""
    H, W = gt.shape[1:]
    pred_gt = resize_to_gt(pred.depth, (H, W))
    sparse_depth, sparse_mask = _sparse_arrays(views)
    sparse_mask_gt = (
        resize_to_gt(sparse_mask.astype(np.float32), (H, W), nearest=True) > 0.5
    )
    published, aligned = P.published_metrics(
        P.depth_to_disparity(pred_gt), gt, spec.max_depth
    )
    sparse = P.sparse_aligned_metrics(
        pred.depth,
        sparse_depth,
        sparse_mask,
        pred_gt,
        sparse_mask_gt,
        gt,
        spec.max_depth,
    )
    metric = P.metric_metrics(pred_gt, gt, spec.max_depth)
    row = {
        "dataset": spec.name,
        "sequence": seq.name,
        "mode": mode,
        "density": density,
        "realized_density": realized,
        "frames": int(gt.shape[0]),
        "published": published.as_dict(),
        "sparse_aligned": sparse.as_dict(),
        "metric": metric.as_dict(),
    }
    return row, aligned


def has_cameras(seq: Sequence) -> bool:
    """Every frame carries K and a finite pose -- what TAE needs. VDA's own
    ScanNet TAE manifest copies -inf poses from tracking failures verbatim
    (json -Infinity), so None is not the only way a pose can be missing."""
    return all(
        fr.K is not None and fr.pose is not None and bool(np.isfinite(fr.pose).all())
        for fr in seq.frames
    )


def score_tae_sequence(
    spec: BenchSpec,
    seq: Sequence,
    gt: np.ndarray,
    pred: Prediction,
    mode: str,
    density: float,
    realized: float,
    aligned: np.ndarray | None = None,
) -> dict:
    """One TAE row on the published-aligned depth, both definitions.
    tae_vda takes VDA's K as written (their eval does not shift the
    principal point for the 8/11 crop); tae_ours takes the crop-corrected K,
    since that one is ours to get right. `aligned` is score_sequence's
    published-aligned depth when the main pass already computed it."""
    H, W = gt.shape[1:]
    if aligned is None:
        pred_gt = resize_to_gt(pred.depth, (H, W))
        _, aligned = P.published_metrics(
            P.depth_to_disparity(pred_gt), gt, spec.max_depth
        )
    Ks_raw = [fr.K for fr in seq.frames]
    poses = [fr.pose for fr in seq.frames]
    if any(k is None for k in Ks_raw) or any(p is None for p in poses):
        raise ValueError(f"{spec.name}/{seq.name}: TAE manifest frames lack K/pose")
    Ks_ours = [scaled_intrinsics(k, spec.crop, (H, W), (H, W)) for k in Ks_raw]
    valid = P.gt_valid_mask(gt, spec.max_depth)
    tae_abs, tae_sq = P.tae_ours(aligned, valid, Ks_ours, poses)
    return {
        "dataset": spec.name,
        "sequence": seq.name,
        "mode": mode,
        "density": density,
        "realized_density": realized,
        "frames": int(gt.shape[0]),
        "tae_vda": P.tae_vda(aligned, Ks_raw, poses),
        "tae_ours": tae_abs,
        "tae_ours_sq": tae_sq,
    }


# ---------------------------------------------------------------------------
# point-cloud snapshots
# ---------------------------------------------------------------------------
def save_cloud(
    path: Path,
    spec: BenchSpec,
    seq: Sequence,
    views: list[dict],
    pred: Prediction,
    density: float,
    mode: str,
    n_frames: int,
) -> Path:
    """Everything needed to re-render the first n_frames of a sequence later
    (render_clouds.py): RGB, predicted depth + confidence, the sparse depth, the GT
    at model resolution, predicted cameras and the GT cameras when the
    manifest carries them. Compressed npz, one per (sequence, density)."""
    n = min(n_frames, len(views))
    rgb = torch.stack([(v["img"][0] * 255.0).round().clamp(0, 255) for v in views[:n]])
    payload = {
        "rgb": rgb.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy(),
        "pred_depth": pred.depth[:n].astype(np.float32),
        "pred_conf": pred.conf[:n].astype(np.float32),
        "sparse_depth": torch.stack([v["sparse_depth"][0] for v in views[:n]])
        .float()
        .cpu()
        .numpy(),
        "sparse_mask": torch.stack([v["sparse_depth_mask"][0] for v in views[:n]])
        .bool()
        .cpu()
        .numpy(),
        "gt_depth": torch.stack([v["depthmap"][0] for v in views[:n]])
        .float()
        .cpu()
        .numpy(),
        "gt_valid": torch.stack([v["valid_mask"][0] for v in views[:n]])
        .bool()
        .cpu()
        .numpy(),
        "K_pred": pred.K[:n].astype(np.float32),
        "w2c_pred": pred.w2c[:n].astype(np.float32),
        "frames": np.array([str(fr.image) for fr in seq.frames[:n]]),
        "dataset": np.array(spec.name),
        "sequence": np.array(seq.name),
        "density": np.array(density, dtype=np.float32),
        "mode": np.array(mode),
    }
    if all("camera_intrinsics" in v and "camera_pose" in v for v in views[:n]):
        payload["K_gt"] = (
            torch.stack([v["camera_intrinsics"][0] for v in views[:n]])
            .float()
            .cpu()
            .numpy()
        )
        payload["pose_gt"] = (
            torch.stack([v["camera_pose"][0] for v in views[:n]]).float().cpu().numpy()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _render_cloud(path: Path) -> None:
    # trimesh via visual_util; imported here so this module stays importable
    # (and testable) without the GLB stack
    from render_clouds import render_npz

    try:
        render_npz(path)
    except Exception as e:  # the snapshot is the deliverable; a GLB is a convenience
        print(f"[bench] GLB render of {path.name} failed ({e}); npz kept", flush=True)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows: list[dict], tae_rows: list[dict]) -> dict[str, float]:
    """Per (dataset, mode, density): mean over sequences of every metric,
    keyed "<dataset>/<mode>/d<pct>/<protocol>_<metric>", plus the sequence
    count, realized density and both TAEs. NaN where nothing scored."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault(
            (r["dataset"], r["mode"], density_key(r["density"])), []
        ).append(r)
    out: dict[str, float] = {}
    for (ds, mode, dk), rs in sorted(groups.items()):
        base = f"{ds}/{mode}/{dk}"
        out[f"{base}/n_sequences"] = float(len(rs))
        out[f"{base}/realized_density"] = _mean([r["realized_density"] for r in rs])
        for proto in PROTOCOLS:
            for m in P.METRIC_NAMES:
                out[f"{base}/{proto}_{m}"] = _mean([r[proto][m] for r in rs])
            out[f"{base}/{proto}_frames"] = _mean([r[proto]["frames"] for r in rs])
    tgroups: dict[tuple[str, str, str], list[dict]] = {}
    for r in tae_rows:
        tgroups.setdefault(
            (r["dataset"], r["mode"], density_key(r["density"])), []
        ).append(r)
    for (ds, mode, dk), rs in sorted(tgroups.items()):
        base = f"{ds}/{mode}/{dk}"
        out[f"{base}/tae_n_sequences"] = float(len(rs))
        for m in ("tae_vda", "tae_ours", "tae_ours_sq"):
            out[f"{base}/{m}"] = _mean([r[m] for r in rs])
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _prepare_sequence(
    spec: BenchSpec, seq: Sequence, size: int, device
) -> tuple[np.ndarray, list[dict]]:
    gt = gt_stack(seq, spec)
    views = build_views(seq, spec, gt, size, device)
    for v in views:  # ImgNorm [-1,1] -> [0,1], as _prepare_batch does for training
        v["img"] = (v["img"] + 1.0) / 2.0
    return gt, views


def run_benchmark(
    model: torch.nn.Module,
    accelerator: Accelerator,
    cfg: BenchmarkCfg,
    mcfg,
    output_dir: str,
    step: int,
    seed: int,
) -> dict[str, float]:
    """Score every configured dataset and log/write the results. Sequences
    are sharded round-robin over ranks and the per-sequence rows gathered on
    the main process, so a multi-GPU run finishes proportionally faster."""
    cfg.validate()
    missing = bench_root_ok(cfg.root, cfg.datasets, cfg.tae_datasets)
    if missing:
        raise FileNotFoundError(
            f"bench enabled but manifests missing under {cfg.root} for {missing}; "
            "run datasets_preprocess/prepare_vda_benchmark.py"
        )
    net = accelerator.unwrap_model(model)
    net.eval()
    device = accelerator.device
    rank, world = accelerator.process_index, accelerator.num_processes
    patch = mcfg.depth_cond.sim_patch_size
    cloud_dir = Path(output_dir) / "bench_clouds"

    rows: list[dict] = []
    tae_rows: list[dict] = []
    seconds: dict[str, float] = {}
    tae_skipped: dict[str, int] = {}
    # Per-sequence failures (a corrupt frame, a CUDA OOM on a long stream)
    # are recorded and skipped rather than raised: the ranks meet at the
    # gather below, and a rank that died on its shard would leave the others
    # waiting there forever -- at the end of a multi-day run.
    failed: list[dict] = []

    def _guard(tag: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- see above
            if device.type == "cuda":
                torch.cuda.empty_cache()
            failed.append({"sequence": tag, "error": f"{type(e).__name__}: {e}"})
            print(f"[bench] FAILED {tag}: {type(e).__name__}: {e}", flush=True)
            return None

    for name in cfg.datasets:
        spec = SPECS[name]
        t0 = time.time()
        seqs = load_manifest(cfg.root, spec)
        if cfg.max_sequences:
            seqs = seqs[: cfg.max_sequences]
        for gi in range(rank, len(seqs), world):
            seq = seqs[gi]
            tag = f"{spec.name}/{seq.name}"

            def _one_sequence(seq=seq, gi=gi, tag=tag):
                gt, views = _prepare_sequence(spec, seq, cfg.image_size, device)
                # TAE on the main pass for datasets without a dedicated TAE
                # manifest (ScanNet's separate pass is below)
                main_tae = spec.name in cfg.tae_datasets and spec.tae_json is None
                if main_tae and not has_cameras(seq):
                    main_tae = False
                    tae_skipped[spec.name] = tae_skipped.get(spec.name, 0) + 1
                    accelerator.print(f"[bench] {tag}: missing GT camera(s); no TAE")
                new_rows, new_tae = [], []
                for density in cfg.densities:
                    realized = attach_sparse_depth(
                        views, density, tag, patch, seed, device
                    )
                    pred = predict(net, accelerator, views)
                    row, aligned = score_sequence(
                        spec, seq, gt, views, pred, MODE_STREAM, density, realized
                    )
                    new_rows.append(row)
                    if main_tae:
                        new_tae.append(
                            score_tae_sequence(
                                spec,
                                seq,
                                gt,
                                pred,
                                MODE_STREAM,
                                density,
                                realized,
                                aligned,
                            )
                        )
                    if gi < cfg.clouds_per_dataset and density == cfg.cloud_density:
                        path = (
                            cloud_dir
                            / f"{spec.name}_{seq.name}_{density_key(density)}_{MODE_STREAM}.npz"
                        )
                        save_cloud(
                            path,
                            spec,
                            seq,
                            views,
                            pred,
                            density,
                            MODE_STREAM,
                            cfg.cloud_frames,
                        )
                        _render_cloud(path)
                    del pred
                del views
                # rows land only once the whole sequence scored, so a failure
                # mid-sweep cannot leave a sequence with some densities and
                # not others
                rows.extend(new_rows)
                tae_rows.extend(new_tae)
                accelerator.print(
                    f"[bench] {tag}: {len(seq)} frames done ({time.time() - t0:.0f}s into {name})"
                )

            _guard(tag, _one_sequence)
        if spec.name in cfg.tae_datasets and spec.tae_json:
            tseqs = load_manifest(cfg.root, spec, tae=True)
            if cfg.max_sequences:
                tseqs = tseqs[: cfg.max_sequences]
            for gi in range(rank, len(tseqs), world):
                seq = tseqs[gi]
                tag = f"{spec.name}/tae/{seq.name}"
                if not has_cameras(seq):
                    # ScanNet writes -inf poses where tracking failed
                    tae_skipped[spec.name] = tae_skipped.get(spec.name, 0) + 1
                    accelerator.print(f"[bench] {tag}: missing GT pose(s); skipped")
                    continue

                def _one_tae_sequence(seq=seq, tag=tag):
                    gt, views = _prepare_sequence(spec, seq, cfg.image_size, device)
                    new_tae = []
                    for density in cfg.densities:
                        realized = attach_sparse_depth(
                            views, density, tag, patch, seed, device
                        )
                        pred = predict(net, accelerator, views)
                        new_tae.append(
                            score_tae_sequence(
                                spec, seq, gt, pred, MODE_STREAM, density, realized
                            )
                        )
                        del pred
                    del views
                    tae_rows.extend(new_tae)
                    accelerator.print(f"[bench] {tag}: {len(seq)} frames done")

                _guard(tag, _one_tae_sequence)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seconds[name] = time.time() - t0

    accelerator.wait_for_everyone()
    if world > 1:
        rows = [r for part in gather_object([rows]) for r in part]
        tae_rows = [r for part in gather_object([tae_rows]) for r in part]
        seconds_all = gather_object([seconds])
        seconds = {k: max(s.get(k, 0.0) for s in seconds_all) for k in cfg.datasets}
        skipped_all = gather_object([tae_skipped])
        tae_skipped = {}
        for part in skipped_all:
            for k, v in part.items():
                tae_skipped[k] = tae_skipped.get(k, 0) + v
        failed = [f for part in gather_object([failed]) for f in part]

    result = aggregate(rows, tae_rows)
    for name, sec in seconds.items():
        result[f"{name}/seconds"] = float(sec)
    for name, n in tae_skipped.items():
        result[f"{name}/tae_skipped_sequences"] = float(n)
    result["failed_sequences"] = float(len(failed))

    if accelerator.is_main_process:
        accelerator.log({f"final_bench/{k}": v for k, v in result.items()}, step=step)
        out = Path(output_dir) / "bench_results.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "config": to_primitive(cfg),
                    "step": step,
                    "aggregate": result,
                    "rows": rows,
                    "tae_rows": tae_rows,
                    "failed": failed,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        accelerator.print(f"[bench] wrote {out}")
        if failed:
            accelerator.print(
                f"[bench] {len(failed)} sequence(s) FAILED and were skipped -- see bench_results.json"
            )
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
                accelerator.print(f"[bench] {k}: {result[k]:.4f}")
    return result
