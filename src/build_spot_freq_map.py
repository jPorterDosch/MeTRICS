"""Build the empirical SPOT depth-validity frequency map offline.

Sequence 0 overlaps evaluation starts in experiments/eval_all.sh. Only validity
geometry is extracted, never depth values, so leakage is negligible.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_W = 640
RAW_H = 480
N_PIXELS = RAW_W * RAW_H


def read_spot_depth(path: Path) -> np.ndarray:
    """Read one raw SPOT depth frame in metres."""
    # duplicated from src/visualize_spot.py::read_spot_depth so this offline
    # tool stays numpy-only (visualize_spot imports the full training stack)
    with open(path, "rb") as handle:
        n = np.frombuffer(handle.read(4), dtype=np.int32)[0]
        if n != N_PIXELS:
            raise ValueError(f"{path}: header {n}, expected {N_PIXELS}")
        depth = np.fromfile(handle, dtype=np.float32, count=n)
    return depth.reshape(RAW_H, RAW_W)


def frame_paths(data_root: Path, seq: str) -> list[Path]:
    """Return a sequence's depth frames in numeric filename order."""
    depth_dir = Path(data_root) / str(seq) / "depth"
    return sorted(
        (path for path in depth_dir.iterdir() if path.is_file()),
        key=lambda path: int(path.name),
    )


def compute_freq(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Compute a validity frequency map without retaining frame masks."""
    if not paths:
        raise ValueError("at least one depth frame is required")
    valid_counts = np.zeros((RAW_H, RAW_W), dtype=np.float64)
    densities = np.empty(len(paths), dtype=np.float64)
    for index, path in enumerate(paths):
        valid = read_spot_depth(path) > 0
        valid_counts += valid
        densities[index] = np.mean(valid, dtype=np.float64)
    return (valid_counts / len(paths)).astype(np.float32), densities


def radial_autocorr(masks) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the normalized radial autocorrelation and its 1/e length."""
    autocorr_sum = None
    count = 0
    shape = None
    for item in masks:
        mask = read_spot_depth(item) > 0 if isinstance(item, (str, Path)) else item
        centered = np.asarray(mask, dtype=np.float64)
        centered = centered - centered.mean()
        spectrum = np.fft.fft2(centered)
        autocorr = np.fft.fftshift(np.fft.ifft2(spectrum * spectrum.conjugate()).real)
        if autocorr_sum is None:
            autocorr_sum = np.zeros_like(autocorr)
            shape = autocorr.shape
        elif autocorr.shape != shape:
            raise ValueError("all masks must have the same shape")
        autocorr_sum += autocorr
        count += 1
    if count == 0 or autocorr_sum is None or shape is None:
        raise ValueError("at least one mask is required")
    center = (shape[0] // 2, shape[1] // 2)
    zero_lag = autocorr_sum[center]
    if zero_lag <= 0:
        raise ValueError("masks have no validity variation")
    normalized = autocorr_sum / zero_lag
    yy, xx = np.indices(shape)
    radius_index = np.rint(np.hypot(yy - center[0], xx - center[1])).astype(np.int32)
    max_radius = min(center)
    flat_radius = radius_index.ravel()
    radial_sum = np.bincount(flat_radius, weights=normalized.ravel())
    radial_count = np.bincount(flat_radius)
    profile = radial_sum[: max_radius + 1] / radial_count[: max_radius + 1]
    radii = np.arange(profile.size, dtype=np.float64)
    below = np.flatnonzero(profile < 1.0 / np.e)
    ell = float(radii[below[0]]) if below.size else float("nan")
    return radii, profile, ell


def save_artifact(out: Path, freq: np.ndarray, meta: dict) -> None:
    """Write the map and JSON metadata to a compressed NumPy artifact."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, freq=np.asarray(freq, dtype=np.float32), meta=json.dumps(meta)
    )


def make_plots(
    plots_dir: Path,
    seq_freqs: dict[str, np.ndarray],
    seq_densities: dict[str, np.ndarray],
    combined_freq: np.ndarray,
    radii: np.ndarray,
    profile: np.ndarray,
    ell: float,
) -> None:
    """Write density, frequency-map, and autocorrelation diagnostics."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    for seq, densities in seq_densities.items():
        ax.hist(
            densities, bins=bins, histtype="step", linewidth=1.5, label=f"seq {seq}"
        )
    combined_densities = np.concatenate(list(seq_densities.values()))
    ax.hist(
        combined_densities, bins=bins, histtype="step", linewidth=2, label="combined"
    )
    ax.set(xlabel="Valid fraction per frame", ylabel="Frames", xlim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "valid_density_histogram.png", dpi=160)
    plt.close(fig)

    maps = list(seq_freqs.items()) + [("combined", combined_freq)]
    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 4), squeeze=False)
    image = None
    for ax, (label, freq) in zip(axes[0], maps):
        image = ax.imshow(freq, vmin=0, vmax=1, cmap="viridis", origin="upper")
        ax.set_title(f"seq {label}" if label != "combined" else label)
        ax.set_axis_off()
    fig.colorbar(
        image, ax=axes.ravel().tolist(), label="Validity frequency", shrink=0.85
    )
    fig.savefig(plots_dir / "valid_frequency_maps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, profile)
    ax.axhline(1 / np.e, color="gray", linestyle="--", label="1/e")
    if np.isfinite(ell):
        ax.axvline(ell, color="tab:red", linestyle=":", label=f"ell = {ell:.1f} px")
    ax.set(
        xlabel="Radius (pixels)", ylabel="Normalized autocorrelation", ylim=(-0.2, 1.05)
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "autocorr_radial_profile.png", dpi=160)
    plt.close(fig)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/oscar/data/jtompki1/cli277/new_spot_data"),
    )
    parser.add_argument("--seqs", nargs="+", default=["0", "1"])
    parser.add_argument(
        "--out", type=Path, default=Path("assets/spot/valid_freq_640x480.npz")
    )
    parser.add_argument("--plots-dir", type=Path, default=Path("assets/spot/analysis"))
    parser.add_argument("--autocorr-stride", type=int, default=25)
    args = parser.parse_args()
    if args.autocorr_stride <= 0:
        parser.error("--autocorr-stride must be positive")

    seq_paths = {str(seq): frame_paths(args.data_root, str(seq)) for seq in args.seqs}
    seq_freqs = {}
    seq_densities = {}
    for seq, paths in seq_paths.items():
        seq_freqs[seq], seq_densities[seq] = compute_freq(paths)
        print(f"seq {seq} mean valid fraction: {seq_densities[seq].mean():.6f}")

    n_frames = sum(len(paths) for paths in seq_paths.values())
    combined_freq = (
        sum(
            seq_freqs[seq].astype(np.float64) * len(seq_paths[seq]) for seq in seq_paths
        )
        / n_frames
    )
    combined_freq = combined_freq.astype(np.float32)
    mean_valid = float(np.concatenate(list(seq_densities.values())).mean())
    sampled_paths = [
        path for paths in seq_paths.values() for path in paths[:: args.autocorr_stride]
    ]
    radii, profile, ell = radial_autocorr(sampled_paths)

    meta = {
        "seqs": list(seq_paths),
        "n_frames": n_frames,
        "mean_valid": {
            "combined": mean_valid,
            "per_seq": {
                seq: float(densities.mean()) for seq, densities in seq_densities.items()
            },
        },
        "data_root": str(args.data_root),
        "created": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_revision(),
    }
    save_artifact(args.out, combined_freq, meta)
    make_plots(
        args.plots_dir,
        seq_freqs,
        seq_densities,
        combined_freq,
        radii,
        profile,
        ell,
    )
    print(f"combined mean valid fraction: {mean_valid:.6f}")
    print(f"autocorrelation ell: {ell:.1f} px")
    print(f"set --depth-cond.sim-mask-ratio {round(1 - mean_valid, 2)}")


if __name__ == "__main__":
    main()
