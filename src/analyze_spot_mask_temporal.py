"""Test whether SPOT depth-validity masks are temporally correlated.

Reuses the validity convention from build_spot_freq_map.py. For each lag k it
compares mask_t with mask_{t+k} two ways:

  pooled phi   - all pixels pooled. Inflated by static spatial structure
                 (a permanently dead pixel agrees with itself at every lag),
                 so it is only meaningful next to the shuffled control.
  per-pixel phi - phi computed independently per pixel, then averaged over
                 pixels that are not constant. Static structure cancels, so
                 this is the pure temporal signal.

The shuffled control re-runs the same statistics on a random frame order,
which destroys time but keeps each pixel's marginal validity rate. Excess =
observed - shuffled is the part that only temporal ordering can explain.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from build_spot_freq_map import RAW_H, RAW_W, frame_paths, read_spot_depth, valid_mask


def _phi(n11, n10, n01, n00):
    """Matthews/phi coefficient for 2x2 counts; 0 where a margin is degenerate."""
    num = n11 * n00 - n10 * n01
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    out = np.zeros(np.shape(num), dtype=np.float64)
    ok = den > 0
    np.divide(num, np.sqrt(den, where=ok), out=out, where=ok)
    return out, ok


def lag_counts(paths, max_lag):
    """Stream frames once, accumulating per-pixel 2x2 counts for each lag.

    Returns (counts, densities) where counts[k] is a (4, H, W) array of
    n11/n10/n01/n00 for pairs (t, t+k+1) and densities is the per-frame valid
    fraction in the order the paths were given.
    """
    counts = np.zeros((max_lag, 4, RAW_H, RAW_W), dtype=np.float64)
    densities = np.empty(len(paths), dtype=np.float64)
    window = deque(maxlen=max_lag)  # most recent first
    for index, path in enumerate(paths):
        cur = valid_mask(read_spot_depth(path))
        densities[index] = np.mean(cur, dtype=np.float64)
        for lag_index, prev in enumerate(window):
            # prev is mask_{t-lag}, cur is mask_t
            counts[lag_index, 0] += prev & cur
            counts[lag_index, 1] += prev & ~cur
            counts[lag_index, 2] += ~prev & cur
            counts[lag_index, 3] += ~prev & ~cur
        window.appendleft(cur)
    return counts, densities


def summarize(counts, min_obs=0):
    """Pooled and mean per-pixel phi, plus transition probabilities, per lag.

    A pixel enters the per-pixel mean only if every margin of its 2x2 table
    holds at least min_obs observations. min_obs=0 keeps the old behaviour
    (drop only degenerate pixels); raising it drops near-constant pixels whose
    phi is estimated from a handful of frames.
    """
    rows = []
    for lag_index, per_pixel in enumerate(counts):
        n11, n10, n01, n00 = per_pixel
        pooled, _ = _phi(*(c.sum() for c in per_pixel))
        pixel_phi, ok = _phi(n11, n10, n01, n00)
        if min_obs:
            ok &= (
                (n11 + n10 >= min_obs)
                & (n01 + n00 >= min_obs)
                & (n11 + n01 >= min_obs)
                & (n10 + n00 >= min_obs)
            )
        total = per_pixel.sum()
        p_valid = (n11.sum() + n01.sum()) / total
        p_stay = n11.sum() / max(n11.sum() + n10.sum(), 1.0)
        rows.append(
            {
                "lag": lag_index + 1,
                "pooled_phi": float(pooled),
                "pixel_phi": float(pixel_phi[ok].mean()) if ok.any() else 0.0,
                "varying_pixels": int(ok.sum()),
                "p_valid": float(p_valid),
                "p_valid_given_prev_valid": float(p_stay),
            }
        )
    return rows


def density_autocorr(densities, max_lag):
    """Autocorrelation of the per-frame valid fraction, lags 1..max_lag."""
    centered = densities - densities.mean()
    denom = float(centered @ centered)
    if denom == 0:
        return [0.0] * max_lag
    return [
        float(centered[: len(centered) - k] @ centered[k:] / denom)
        for k in range(1, max_lag + 1)
    ]


def strata(counts, edges=(0.0, 0.01, 0.05, 0.25, 0.75, 0.95, 1.0)):
    """Lag-1 per-pixel phi split by how often the pixel is valid.

    Shows directly whether rarely-valid pixels are pulling the unweighted
    per-pixel mean around.
    """
    n11, n10, n01, n00 = counts[0]
    total = n11 + n10 + n01 + n00
    freq = np.divide(n11 + n01, total, out=np.zeros_like(n11), where=total > 0)
    pixel_phi, ok = _phi(n11, n10, n01, n00)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = ok & (freq > lo) & (freq <= hi)
        rows.append(
            {
                "band": f"{lo:g}-{hi:g}",
                "pixels": int(band.sum()),
                "pixel_phi": float(pixel_phi[band].mean()) if band.any() else 0.0,
            }
        )
    return rows


def analyze(paths, max_lag, seed, min_obs=0):
    observed, densities = lag_counts(paths, max_lag)
    order = np.random.default_rng(seed).permutation(len(paths))
    shuffled, _ = lag_counts([paths[i] for i in order], max_lag)
    return {
        "n_frames": len(paths),
        "mean_valid": float(densities.mean()),
        "min_obs": min_obs,
        "observed": summarize(observed, min_obs),
        "shuffled": summarize(shuffled, min_obs),
        "observed_unfiltered": summarize(observed),
        "strata": strata(observed),
        "density_autocorr": density_autocorr(densities, max_lag),
    }


def report(label, result):
    print(
        f"\nseq {label}: {result['n_frames']} frames, "
        f"mean valid {result['mean_valid']:.4f}"
    )
    print(f"min_obs {result['min_obs']}")
    print(
        f"{'lag':>4} {'pixel_phi':>10} {'shuf':>8} {'excess':>8} {'npix':>8} "
        f"{'pooled':>8} {'shuf':>8} {'P(v|v)':>8} {'P(v)':>7} {'dens_ac':>8}"
    )
    for obs, shuf, autocorr in zip(
        result["observed"], result["shuffled"], result["density_autocorr"]
    ):
        print(
            f"{obs['lag']:>4} {obs['pixel_phi']:>10.4f} {shuf['pixel_phi']:>8.4f} "
            f"{obs['pixel_phi'] - shuf['pixel_phi']:>8.4f} {obs['varying_pixels']:>8d} "
            f"{obs['pooled_phi']:>8.4f} {shuf['pooled_phi']:>8.4f} "
            f"{obs['p_valid_given_prev_valid']:>8.4f} {obs['p_valid']:>7.4f} "
            f"{autocorr:>8.4f}"
        )
    print("\n  lag-1 per-pixel phi by pixel validity rate:")
    for row in result["strata"]:
        print(
            f"    f in ({row['band']:>9}]  {row['pixels']:>8d} px  "
            f"phi {row['pixel_phi']:>7.4f}"
        )
    unfiltered = result["observed_unfiltered"][0]
    print(
        f"  lag-1 unfiltered pixel_phi {unfiltered['pixel_phi']:.4f} "
        f"over {unfiltered['varying_pixels']} px"
    )

    lag1 = result["observed"][0]
    excess = lag1["pixel_phi"] - result["shuffled"][0]["pixel_phi"]
    verdict = "correlated" if excess > 0.05 else "no meaningful temporal structure"
    print(f"  lag-1 per-pixel excess phi {excess:+.4f} -> {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/oscar/data/jtompki1/cli277/new_spot_data"),
    )
    parser.add_argument("--seqs", nargs="+", default=["0", "1"])
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="use only the first N frames per sequence (0 = all)",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=0,
        help="drop pixels with fewer than N observations in any 2x2 margin",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = {}
    for seq in args.seqs:
        paths = frame_paths(args.data_root, str(seq))
        if args.max_frames:
            paths = paths[: args.max_frames]
        results[str(seq)] = analyze(paths, args.max_lag, args.seed, args.min_obs)
        report(str(seq), results[str(seq)])

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
