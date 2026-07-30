#!/usr/bin/env python
"""CPU smoke test for the predicted-confidence heatmap export.

Exercises the whole confidence path end to end without a GPU or a checkpoint:

    _export_heatmaps(..., conf=...)   ->  {tag}_conf_NNN.png + CSV columns
    heatmaps_to_gif.py --pair         ->  pair_conf.gif
    compare_heatmap_summaries.py      ->  conf / conf_corr rows

It feeds SYNTHETIC data with a KNOWN answer, which is the point: the confidence
field is built anti-correlated with the injected depth error, so
Spearman(conf, |pred-gt|/gt) must come out strongly negative. A run that only
checks "files were written" cannot tell a correct correlation from a
sign-flipped one, and a sign flip is exactly the bug that would make every
future confidence readout say the opposite of the truth.

Also checks the mixed-schema case: an arm written WITHOUT conf (older CSVs, or
any caller passing conf=None) must still pair against a conf-carrying arm
instead of crashing on the missing columns.

    python tests/conf_heatmap_smoke.py            # writes to a temp dir
    python tests/conf_heatmap_smoke.py --keep DIR # keep the artifacts to look at
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from visualize_depth import _export_heatmaps, _spearman  # noqa: E402

S, H, W = 4, 32, 40


def synth() -> tuple[torch.Tensor, ...]:
    """A clip whose confidence is anti-correlated with its depth error by
    construction. gt is a smooth ramp (2-6 m); pred adds a spatially varying
    relative error; conf = 1 + exp(-k * rel_err), i.e. the shape an expp1 head
    would produce if it had learned perfectly. Poses translate along x so the
    tcons warp has something to do."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gt = 2.0 + 4.0 * (xx / W)  # [H,W], 2..6 m
    gt = np.broadcast_to(gt, (S, H, W)).copy()

    # relative error field: smooth + a little noise, 0..~25%
    rel = 0.02 + 0.20 * (yy / H) + 0.02 * rng.random((S, H, W)).astype(np.float32)
    pred = gt * (1.0 + rel)
    conf = 1.0 + np.exp(-8.0 * rel)  # high where rel is low -> negative Spearman

    valid = np.ones((S, H, W), dtype=bool)
    valid[:, :2, :] = False  # a GT hole, so the mask actually does something

    K = np.zeros((S, 3, 3), dtype=np.float32)
    K[:, 0, 0] = K[:, 1, 1] = 40.0
    K[:, 0, 2], K[:, 1, 2] = W / 2, H / 2
    K[:, 2, 2] = 1.0
    pose = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))  # cam2world
    pose[:, 0, 3] = 0.05 * np.arange(S)  # slide along x

    t = lambda a: torch.from_numpy(a)  # noqa: E731
    return t(pred), t(gt), t(valid), t(K), t(pose), t(conf)


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path) as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].lstrip().startswith("#")]
    return rows[0], rows[1:]


def run(hm_dir: Path) -> int:
    pred, gt, valid, K, pose, conf = synth()
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        failures += not cond

    # ---- the expected answer, computed independently of _export_heatmaps ----
    ok = valid.numpy() & (gt.numpy() > 0)
    rel_ref = np.abs(pred.numpy() - gt.numpy()) / gt.numpy()
    want = float(
        np.mean([_spearman(conf.numpy()[i][ok[i]], rel_ref[i][ok[i]]) for i in range(S)])
    )
    print(f"\nsynthetic ground truth: Spearman(conf, rel_err) = {want:+.3f}")
    check(want < -0.9, f"the SYNTHETIC data really is anti-correlated ({want:+.3f})")

    # ---- 1. conf-carrying arm ----
    print("\n[1] _export_heatmaps with conf=  (tag 'finetuned_clip0')")
    n = _export_heatmaps(
        str(hm_dir), "finetuned_clip0", pred, gt, valid, K, pose,
        conf=conf, conf_vmin=1.0, conf_vmax=2.0,
    )
    pngs = sorted(p.name for p in hm_dir.glob("finetuned_clip0_conf_*.png"))
    check(len(pngs) == S, f"{len(pngs)}/{S} conf PNGs written {pngs}")
    # depth S + gterr S + conf S + tcons S-1 + csv + summary png
    check(n == 4 * S + 1, f"reported {n} files written (expected {4 * S + 1})")

    header, rows = read_csv(hm_dir / "finetuned_clip0_summary.csv")
    for c in ("conf_mean", "conf_saturated_frac", "conf_err_corr"):
        check(c in header, f"CSV has column {c}  (header: {','.join(header)})")
    check(
        header[:5]
        == ["frame", "gterr_mean", "gterr_saturated_frac", "tcons_mean",
            "tcons_saturated_frac"],
        "the pre-existing columns kept their positions (appended, not inserted)",
    )
    got = np.array([float(r[header.index("conf_err_corr")]) for r in rows])
    check(len(got) == S, f"{len(got)}/{S} conf_err_corr rows")
    check(
        np.allclose(got.mean(), want, atol=1e-6),
        f"CSV conf_err_corr mean {got.mean():+.6f} == independent {want:+.6f}",
    )
    # conf_vmax=2.0 with conf in (1, 2]: the zero-error tail should clip a little
    sat = np.array([float(r[header.index("conf_saturated_frac")]) for r in rows])
    check(bool((sat >= 0).all() and (sat <= 1).all()), f"saturated_frac in [0,1] {sat}")
    cm = np.array([float(r[header.index("conf_mean")]) for r in rows])
    check(bool((cm > 1.0).all()), f"conf_mean above the expp1 floor of 1.0 {cm.round(3)}")

    # a rendered conf panel must not be uniform -- a constant image is what a
    # broken normalization (or a vmin==vmax) silently produces
    from PIL import Image

    px = np.asarray(Image.open(hm_dir / pngs[0]).convert("RGB"))
    check(px.std() > 5.0, f"conf panel has structure (pixel std {px.std():.1f})")
    check(px.shape[:2] == (H, W), f"conf panel is native resolution {px.shape[:2]}")

    # ---- 2. the base arm, written WITHOUT conf (old schema) ----
    print("\n[2] _export_heatmaps with conf=None  (tag 'base_clip0', old schema)")
    n0 = _export_heatmaps(str(hm_dir), "base_clip0", pred * 1.1, gt, valid, K, pose)
    check(n0 == 3 * S + 1, f"reported {n0} files (expected {3 * S + 1}, no conf series)")
    check(
        not list(hm_dir.glob("base_clip0_conf_*.png")),
        "no conf PNGs written when conf is omitted",
    )
    h0, _ = read_csv(hm_dir / "base_clip0_summary.csv")
    check("conf_mean" not in h0, f"old-schema CSV has no conf columns ({','.join(h0)})")

    # ---- 3. GIF assembly picks the new series up with no changes ----
    print("\n[3] heatmaps_to_gif.py --pair")
    r = subprocess.run(
        [sys.executable, str(REPO / "src" / "heatmaps_to_gif.py"),
         "--hm-dir", str(hm_dir), "--pair", "base_clip0", "finetuned_clip0"],
        capture_output=True, text=True,
    )
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stderr[-500:]}")
    check(
        (hm_dir / "finetuned_clip0_conf.gif").is_file(),
        "finetuned_clip0_conf.gif written",
    )
    # base has no conf series, so pair_conf cannot exist -- and the shared
    # series (depth/gterr/tcons) must still pair
    check(
        not (hm_dir / "pair_conf.gif").is_file(),
        "no pair_conf.gif when only one arm has conf (correctly skipped)",
    )
    check((hm_dir / "pair_depth.gif").is_file(), "pair_depth.gif still written")

    # ---- 4. compare, mixed schema then matched schema ----
    print("\n[4] compare_heatmap_summaries.py, MIXED schema (must not crash)")
    cmp_cmd = [sys.executable, str(REPO / "tests" / "compare_heatmap_summaries.py"),
               str(hm_dir)]
    r = subprocess.run(cmp_cmd, capture_output=True, text=True)
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stderr[-800:]}")
    check("only one arm" in r.stdout, "reports the one-sided conf series")
    check("gterr" in r.stdout and "tcons" in r.stdout, "still compares the error series")

    print("\n[5] compare_heatmap_summaries.py, BOTH arms with conf")
    both = hm_dir.parent / "matched"
    both.mkdir(exist_ok=True)
    _export_heatmaps(str(both), "finetuned_clip0", pred, gt, valid, K, pose,
                     conf=conf, conf_vmax=2.0)
    # a WORSE, less-informative base: bigger error and a conf field that no
    # longer tracks it, so conf_corr must favour the finetuned arm (more negative)
    _export_heatmaps(str(both), "base_clip0", pred * 1.1, gt, valid, K, pose,
                     conf=torch.from_numpy(
                         np.full((S, H, W), 1.5, dtype=np.float32)
                         + 0.01 * np.random.default_rng(1).random((S, H, W)).astype(np.float32)
                     ), conf_vmax=2.0)
    r = subprocess.run(
        [sys.executable, str(REPO / "tests" / "compare_heatmap_summaries.py"), str(both)],
        capture_output=True, text=True,
    )
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stderr[-800:]}")
    print("\n".join("      " + ln for ln in r.stdout.strip().splitlines()))
    check("conf " in r.stdout, "conf row present")
    check("conf_corr" in r.stdout, "conf_corr row present")
    check("diagnostic" in r.stdout, "conf is labelled a diagnostic, not a win/loss")
    corr_line = next((ln for ln in r.stdout.splitlines() if ln.startswith("conf_corr")), "")
    # columns: name, left mean, right mean, delta, rel, verdict
    check(
        bool(corr_line) and float(corr_line.split()[3]) < 0,
        f"conf_corr delta is negative (finetuned better calibrated): {corr_line!r}",
    )
    r2 = subprocess.run(
        [sys.executable, str(REPO / "src" / "heatmaps_to_gif.py"), "--hm-dir", str(both),
         "--pair", "base_clip0", "finetuned_clip0"], capture_output=True, text=True,
    )
    check(r2.returncode == 0, f"gif exit {r2.returncode}\n{r2.stderr[-500:]}")
    check((both / "pair_conf.gif").is_file(), "pair_conf.gif written for a matched pair")

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", default=None, help="write artifacts here instead of a temp dir")
    args = ap.parse_args()

    if args.keep:
        root = Path(args.keep)
        (root / "heatmaps").mkdir(parents=True, exist_ok=True)
        failures = run(root / "heatmaps")
        print(f"\nartifacts: {root}")
    else:
        with tempfile.TemporaryDirectory() as td:
            hm = Path(td) / "heatmaps"
            hm.mkdir(parents=True)
            failures = run(hm)

    print("\n" + ("=" * 60))
    if failures:
        raise SystemExit(f"{failures} CHECK(S) FAILED")
    print("all checks passed")


if __name__ == "__main__":
    main()
