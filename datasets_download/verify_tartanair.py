#!/usr/bin/env python3
"""Verify a TartanAir download before preprocessing trusts it.

The upstream downloader (download_tartanair.py, kept verbatim) writes each zip
straight to its final name with no atomic rename and no size check, and its
per-file failures are printed but not propagated -- so a job killed at the
walltime can leave a truncated archive that the next run reports as "exists",
and the wrapper still exits 0. That corruption would otherwise surface much
later as a BadZipFile deep inside preprocess_tartanair.py.

This checks, for every zip the download was supposed to produce:
  * the file is present;
  * its size matches download_training_zipfiles.txt (which is exact, and is
    also what the downloader itself selects from);
  * its central directory parses (catches truncation the size check misses).

--deep additionally CRC-checks every member (hours over ~889 GB); the default
is seconds-to-minutes and catches the failure modes that actually occur.

Exit code 0 iff every expected archive is intact.
"""

import argparse
import os
import os.path as osp
import sys
import zipfile

# byte tolerance on the manifest size (it is recorded in GB, 9 decimals)
SIZE_TOL = 1024


def expected_zips(manifest, levels=("Easy", "Hard")):
    """The same selection download_tartanair.sh makes: image_left, depth_left,
    flow_flow, flow_mask for both difficulties (--only-left, no --only-flow)."""
    wanted = {"image_left.zip", "depth_left.zip", "flow_flow.zip", "flow_mask.zip"}
    out = {}
    for line in open(manifest):
        parts = line.strip().split()
        if len(parts) != 2 or not parts[0].endswith(".zip"):
            continue
        rel, gb = parts[0], float(parts[1])
        seg = rel.split("/")
        if seg[-2] in levels and seg[-1] in wanted:
            out[rel.replace("/", "_")] = int(round(gb * 1e9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="the TartanAir download dir (flat zips)")
    ap.add_argument(
        "--manifest",
        default=osp.join(
            osp.dirname(osp.abspath(__file__)), "download_training_zipfiles.txt"
        ),
    )
    ap.add_argument("--deep", action="store_true", help="CRC-check every member")
    args = ap.parse_args()

    want = expected_zips(args.manifest)
    missing, wrong_size, corrupt = [], [], []
    for name, size in sorted(want.items()):
        p = osp.join(args.root, name)
        if not osp.isfile(p):
            missing.append(name)
            continue
        actual = os.path.getsize(p)
        if abs(actual - size) > SIZE_TOL:
            wrong_size.append(f"{name} ({actual} vs ~{size})")
            continue
        try:
            with zipfile.ZipFile(p) as zf:
                if args.deep and zf.testzip() is not None:
                    corrupt.append(f"{name} (CRC)")
                elif not zf.namelist():
                    corrupt.append(f"{name} (empty)")
        except zipfile.BadZipFile as e:
            corrupt.append(f"{name} ({e})")

    ok = len(want) - len(missing) - len(wrong_size) - len(corrupt)
    print(f"TartanAir: {ok}/{len(want)} archives intact")
    for label, items in (
        ("MISSING", missing),
        ("WRONG SIZE (truncated -- delete and re-run)", wrong_size),
        ("CORRUPT (delete and re-run)", corrupt),
    ):
        if items:
            print(f"  {label}: {len(items)}")
            for i in items[:10]:
                print(f"    {i}")
    if missing or wrong_size or corrupt:
        print("\nDelete the listed files and re-run download_tartanair.sh.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
