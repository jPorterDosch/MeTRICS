"""Verify the selectively-downloaded Hypersim raw tree.

The vendored downloader (download_hypersim_subset.py) resumes by FILE EXISTENCE
only and extracts straight to the final path (no .tmp+rename), so an
interrupted download leaves a truncated .hdf5 that later runs silently skip.
This sweep opens every .hdf5 and reads it fully -- a tail-truncated file may
open cleanly but raises on read -- so corruption is actually caught.

    python verify_hypersim.py /path/to/hypersim            # report only
    python verify_hypersim.py /path/to/hypersim --delete   # remove corrupt files

With --delete, re-running the downloader afterwards re-fetches exactly the
removed members (existence-skip keeps the good ones). Exit code 0 iff clean.

Files modified within --min-age seconds are SKIPPED as possibly-in-flight
(default 120), so this is safe to run while a download is still finishing --
though a final clean pass should be run once the download is fully done.
"""

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hypersim_dir", help="dataset root containing scene dirs")
    parser.add_argument(
        "--delete", action="store_true", help="delete corrupt/truncated files"
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=120,
        help="skip files modified within N seconds (possibly still downloading)",
    )
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.hypersim_dir, "**", "*.hdf5"), recursive=True)
    )
    if not files:
        print(f"no .hdf5 files under {args.hypersim_dir}")
        sys.exit(1)

    now = time.time()
    bad, skipped_recent = [], 0
    for i, path in enumerate(files, 1):
        if now - os.path.getmtime(path) < args.min_age:
            skipped_recent += 1
            continue
        try:
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    # full read forces every chunk off disk -> catches a tail
                    # truncation that a metadata-only open would miss
                    np.asarray(f[key][()])
        except Exception as e:  # h5py/OSError on truncation, KeyError, etc.
            bad.append((path, type(e).__name__))
        if i % 5000 == 0:
            print(f"checked {i}/{len(files)}...", flush=True)

    print(f"hdf5 checked: {len(files) - skipped_recent}   corrupt: {len(bad)}")
    if skipped_recent:
        print(f"skipped (modified < {args.min_age}s ago, possibly in-flight): "
              f"{skipped_recent}")
    for path, err in bad[:50]:
        print(f"  {err}: {path}")
    if len(bad) > 50:
        print(f"  ... and {len(bad) - 50} more")

    if bad and args.delete:
        for path, _ in bad:
            os.remove(path)
        print(f"deleted {len(bad)} corrupt files -- re-run the downloader to refetch")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
