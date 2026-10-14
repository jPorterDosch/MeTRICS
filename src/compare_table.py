"""The comparison table: our runs and the baseline arms, from their
bench_results.json files, at one sparse-depth density.

    python src/compare_table.py \\
        --ours "random 5%=<run>/bench_v3/bench_results.json" ... \\
        --baseline <eval_jd/baselines/bench_vda/bench_results.json> ... \\
        --density 0.05 --out <dir>

Writes <out>/comparison.md (with a footnote per baseline: its reproduction
status and the evaluation differences recorded below) and comparison.csv.
Every number is read from a results file; nothing is computed here except
the mean over datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
        "518 on frames wider than 16:9) and with SDPA attention; reproduction "
        "close, not verified (ScanNet TAE 1.06 vs README 1.09)."
    ),
    "ovda": "oVDA context 16, fp32 (paper fp16).",
    "dav": (
        "20 seeded passes averaged (paper: ensemble of 20); clips over 32 frames "
        "padded in time (one copy of the first frame, copies of the last) so "
        "upstream's long-video path returns every frame; frames reflect-padded "
        "to multiples of 32 instead of upstream's centre crop."
    ),
    "promptda": (
        "Per-frame, upstream input sizing; reproduction close, not verified "
        "(3 of 6 ARKitScenes targets within tolerance, all within 5.3%)."
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
    for name, key, _ in METRICS:
        vals = [out[f"{ds} {name}"] for ds in DATASETS]
        vals = [v for v in vals if v is not None]
        out[f"mean {name}"] = sum(vals) / len(vals) if vals else None
    return out


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
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dens = f"d{round(args.density * 100):g}"

    rows, notes = [], []
    for spec in args.ours:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--ours wants LABEL=path, got {spec!r}")
        rows.append(row(f"Ours ({label})", _load(Path(path)), dens))
    for i, path in enumerate(args.baseline, 1):
        res = _load(Path(path))
        arm = res["arm"]
        status = res.get("parity_status", "unknown")
        mark = f"[{i}]"
        rows.append(row(f"{arm} {mark}", res, dens))
        notes.append(
            f"{mark} {arm}: reproduction {status}"
            + (" (run with --allow-close)" if res.get("allow_close") else "")
            + f". {EVAL_NOTES.get(arm, '')}"
            + f" Results: {path}, code {res.get('commit', '?')}."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["model"] + [
        f"{ds} {n}" for ds in (*DATASETS, "mean") for n, _, _ in METRICS
    ]
    with open(args.out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [
        f"Sparse-depth density {args.density:g} (RGB-only baselines ignore it).",
        "",
    ]
    for ds in (*DATASETS, "mean"):
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
    lines += ["Notes:", ""] + [f"- {n}" for n in notes]
    (args.out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(args.out / "comparison.md")


if __name__ == "__main__":
    main()
