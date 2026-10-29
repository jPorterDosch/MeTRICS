"""The comparison table: our runs and the baseline arms, from their
bench_results.json files, at one sparse-depth density.

    python src/compare_table.py \\
        --ours "random 5%=<run>/bench_v3/bench_results.json" ... \\
        --baseline <eval_jd/baselines/bench_vda/bench_results.json> ... \\
        --density 0.05 --out <dir>

Writes <out>/comparison.md (with a footnote per baseline: its reproduction
status and the evaluation differences recorded below) and comparison.csv.
Every number is read from a results file and reported per dataset; nothing
is averaged across datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eval import provenance as PROV

DATASETS = ("sintel", "scannet", "kitti", "bonn_all", "nyuv2")
# (column label, aggregate key suffix, lower is better)
METRICS = (
    ("AbsRel (published)", "published_abs_rel", True),
    ("d1 (published)", "published_delta1", False),
    ("AbsRel (sparse-aligned)", "sparse_aligned_abs_rel", True),
    ("d1 (sparse-aligned)", "sparse_aligned_delta1", False),
    ("AbsRel (metric)", "metric_abs_rel", True),
    ("d1 (metric)", "metric_delta1", False),
    ("TAE", "tae_vda", True),
)
# what differs from running the released model as-is, per arm
EVAL_NOTES = {
    "vda": "VDA-L, infer.py settings (input 518, fp32).",
    "vda_metric": (
        "Metric-VDA-L without infer_video_depth's aspect shrink (input short side "
        "518 on frames wider than 16:9) and with SDPA attention. Recorded close: "
        "KITTI delta1 within tolerance, ScanNet TAE 1.06 vs README 1.09."
    ),
    "ovda": "oVDA context 16, fp32 (paper fp16).",
    "dav": (
        "20 seeded passes averaged (paper: ensemble of 20). Verified on single "
        "images (Table 2); here every clip over 32 frames runs upstream's "
        "long-video path (key frames + interpolation), padded in time so it "
        "returns every frame, and frames go through run_infer's default "
        "1024 px cap (KITTI 374x1242 -> 308x1024; the verified KITTI target "
        "ran at native 352x1216). Frames reflect-padded to multiples of 32 "
        "instead of upstream's centre crop."
    ),
    "promptda": (
        "Per-frame, upstream input sizing. Recorded close: 3 of 6 ARKitScenes "
        "targets within tolerance, all within 5.3%."
    ),
    "da3_metric": (
        "Short side 504, focal from the GT camera; trained on aspect ratios up "
        "to 1:2 and verified on a 1:2 KITTI crop, here run on the full frame "
        "(KITTI 1:3.3)."
    ),
}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _value(agg: dict, dataset: str, mode: str, dens: str, key: str):
    v = agg.get(f"{dataset}/{mode}/{dens}/{key}")
    return None if v is None else float(v)


def _modes(agg: dict) -> set[str]:
    return {k.split("/")[1] for k in agg if k.count("/") >= 3}


def row(label: str, results: dict, dens: str) -> dict:
    agg = results["aggregate"]
    modes = _modes(agg)
    if len(modes) != 1:
        raise ValueError(f"{label}: expected one inference mode, found {sorted(modes)}")
    mode = modes.pop()
    out = {"model": label}
    for ds in DATASETS:
        for name, key, _ in METRICS:
            out[f"{ds} {name}"] = _value(agg, ds, mode, dens, key)
    return out


# settings every results file in one table must share: they decide which
# pixels are fed and scored
SHARED = ("image_size", "seed", "patch_size", "max_sequences", "root")


def check_comparable(
    named: list[tuple[str, dict]],
    dens: str,
    accept_replay: dict[str, str] | None = None,
) -> None:
    """Refuse to table results that were not scored on the same benchmark:
    shared settings, the same sequence count per dataset (and per TAE set),
    and no failed sequences (run_benchmark skips a failed one, which would
    leave that model's mean over fewer sequences).

    accept_replay: {arm: predictions commit} -- a replayed arm (benchmark
    --predictions) whose dump was made by an earlier code state than the one
    scoring it, accepted for exactly that arm and that commit. Only for a
    dump whose prediction path is verified unchanged between the two."""
    accept_replay = accept_replay or {}
    problems = []
    for key in SHARED:
        vals = {label: r.get("config", {}).get(key) for label, r in named}
        present = {v for v in vals.values() if v is not None}
        if len(present) > 1:
            problems.append(f"config {key} differs: {vals}")
    for ds in DATASETS:
        for field in ("n_sequences", "tae_n_sequences"):
            counts = {}
            for label, r in named:
                agg = r["aggregate"]
                modes = _modes(agg)
                mode = next(iter(modes)) if len(modes) == 1 else None
                counts[label] = agg.get(f"{ds}/{mode}/{dens}/{field}")
            present = {v for v in counts.values() if v is not None}
            if len(present) > 1:
                problems.append(f"{ds} {field} differs: {counts}")
    for label, r in named:
        if r.get("failed"):
            problems.append(f"{label} has failed sequences: {len(r['failed'])}")
    # one scoring code for every row (a protocol change between runs would mix
    # definitions). With every row's code_files recorded, compared file by
    # file: any file two rows both imported must be identical, so a commit
    # that touched only code a run never loaded does not separate it. A row
    # without them (scored before 2026-10-22) falls back to the whole-repo
    # commit string, which must then match exactly.
    files = {label: r.get("code_files") for label, r in named}
    if all(files.values()):
        problems += PROV.code_differences(files)
    else:
        commits = {label: r.get("commit") for label, r in named}
        if (
            not all(PROV.identifiable(c) for c in commits.values())
            or len(set(commits.values())) > 1
        ):
            problems.append(f"scoring code differs or is unrecorded: {commits}")
    dtypes = {
        label: r.get("autocast_dtype") for label, r in named if "autocast_dtype" in r
    }
    if len(set(dtypes.values())) > 1:
        problems.append(f"our runs used different autocast dtypes: {dtypes}")
    # a replayed arm's predictions must come from the same code as the run
    # that scores them: file by file where the dump recorded its files (every
    # file both the dump and the scoring run imported identical), else by
    # commit, or by an explicit --accept-replay ARM=COMMIT
    for label, r in named:
        pc = r.get("predictions_commit")
        if pc is None:
            continue
        if accept_replay.get(r.get("arm")) == pc:
            continue  # the user verified the prediction path (footnoted)
        pfiles, sfiles = r.get("predictions_code_files"), r.get("code_files")
        if pfiles and sfiles:
            problems += [
                f"{label} (dump vs scoring): {p} -- if only scoring code "
                f"changed, --accept-replay {r.get('arm')}={pc}"
                for p in PROV.code_differences({"dump": pfiles, "scoring": sfiles})
            ]
        elif pc != r.get("commit"):
            problems.append(
                f"{label}: predictions made by code {pc}, scored by {r.get('commit')}"
            )
    # the same benchmark data: identical manifest bytes for every dataset any
    # row read (a re-exported tree at the same path changes them)
    manifests = {label: r.get("manifests") for label, r in named}
    unrecorded = sorted(k for k, v in manifests.items() if v is None)
    if unrecorded:
        problems.append(f"benchmark manifests unrecorded: {unrecorded}")
    else:
        for rel in sorted({k for m in manifests.values() for k in m}):
            seen = {label: m.get(rel) for label, m in manifests.items()}
            if len(set(seen.values())) > 1:
                problems.append(f"manifest {rel} differs: {seen}")
    if problems:
        raise SystemExit("results are not comparable:\n  " + "\n  ".join(problems))


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--ours", nargs="+", default=[], help="LABEL=path/to/bench_results.json"
    )
    ap.add_argument(
        "--baseline", nargs="+", default=[], help="path/to/bench_results.json"
    )
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument(
        "--accept-replay",
        nargs="+",
        default=[],
        metavar="ARM=COMMIT",
        help="accept a replayed arm's predictions made by COMMIT although a "
        "later code state scored them (only where the prediction path is "
        "verified unchanged); named in that arm's footnote",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dens = f"d{round(args.density * 100):g}"
    accept_replay = {}
    for spec in args.accept_replay:
        arm, _, commit = spec.partition("=")
        if not commit:
            raise SystemExit(f"--accept-replay wants ARM=COMMIT, got {spec!r}")
        accept_replay[arm] = commit

    rows, notes, named = [], [], []
    for spec in args.ours:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--ours wants LABEL=path, got {spec!r}")
        named.append((f"Ours ({label})", _load(Path(path))))
    for path in args.baseline:
        res = _load(Path(path))
        named.append((res["arm"], res))
    labels = [label for label, _ in named]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        # every check is keyed by label: a duplicate would hide one file
        raise SystemExit(f"two results files under one label {dup}: one per model")
    check_comparable(named, dens, accept_replay)
    for label, res in named[: len(args.ours)]:
        rows.append(row(label, res, dens))
    for i, path in enumerate(args.baseline, 1):
        res = named[len(args.ours) + i - 1][1]
        arm = res["arm"]
        status = res.get("parity_status", "unknown")
        mark = f"[{i}]"
        rows.append(row(f"{arm} {mark}", res, dens))
        code = f"code {res.get('commit', '?')}"
        if res.get("predictions_commit"):
            code = (
                f"predictions by code {res['predictions_commit']}, "
                f"scored by code {res.get('commit', '?')}"
            )
        if res.get("predictions_commit") and arm in accept_replay:
            code += " (prediction path verified unchanged between the two)"
        notes.append(
            f"{mark} {arm}: reproduction status {status}"
            + (" (compared under --allow-close)" if res.get("allow_close") else "")
            + f". {EVAL_NOTES.get(arm, '')}"
            + f" Results: {path}, {code}."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["model"] + [f"{ds} {n}" for ds in DATASETS for n, _, _ in METRICS]
    with open(args.out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [
        f"Sparse-depth density {args.density:g} (RGB-only baselines ignore it).",
        "",
    ]
    for ds in DATASETS:
        lines += [
            f"### {ds}",
            "",
            "| model | " + " | ".join(n for n, _, _ in METRICS) + " |",
        ]
        lines.append("|---|" + "---|" * len(METRICS))
        for r in rows:
            lines.append(
                f"| {r['model']} | "
                + " | ".join(_fmt(r[f"{ds} {n}"]) for n, _, _ in METRICS)
                + " |"
            )
        lines.append("")
    lines += ["Notes:", ""]
    lines += [
        "- Every model gets the same input pixels: ScanNet's TAE clips (an "
        "uncropped manifest) are fed as whole 640x480 frames to every model and "
        "the prediction is cropped onto the 464x618 GT grid afterwards, as in "
        "VDA's protocol and the verified reproductions.",
        "- Our model's input: long side 518, each side resized to the nearest "
        "multiple of 14, no crop (518x392 for 4:3 frames, its training "
        "resolution; aspect error at most 0.9% there; Sintel 518x224, 1.6%, a "
        "grid training did not use; KITTI 1.3%). "
        "StreamVGGT's own loader floors instead (518x378). The baselines use "
        "their own sizing (VDA: short side 518, aspect kept).",
        "- Sintel is in true colour for every model; VDA's Sintel parity was "
        "verified on its channel-swapped extraction (sintel_bgr), a fairness "
        "choice made on 2026-10-05.",
        "- TAE is VDA's definition verbatim on every dataset (all pixels, frame "
        "i's K), the only dataset VDA publishes TAE on being ScanNet (verified "
        "0.563 vs 0.570). Off ScanNet it also scores pixels with no GT (Sintel "
        "sky, KITTI without LiDAR), which the per-video fit does not constrain; "
        "a GT-valid-only variant was rejected because it does not reproduce "
        "ScanNet's numbers (2026-10-14).",
        "- Sintel TAE is dominated by scene motion (TAE assumes a static scene): "
        "ground truth itself scores ~55 on average, ambush_2 alone ~830 unmasked.",
        "- ScanNet is the fallback export from the processed tree (frames shift "
        "in 16 of 100 scenes vs VDA's .sens export; scene0718_00 has 326 "
        "frames); identical for every row.",
    ]
    lines += [f"- {n}" for n in notes]
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(args.out / "comparison.md")


if __name__ == "__main__":
    main()
