#!/usr/bin/env python3
"""Verify a ScanNet .sens download before extraction trusts it.

download_scannet.py (kept as ScanNet ships it) collects per-scan exceptions
into a list, prints them, and returns normally -- so the wrapper exits 0 even
when scenes failed, and the Slurm job reports SUCCESS. It also creates a
tempfile stub before fetching, which is left behind if the fetch raises.

This compares what is on disk against the authoritative release scan lists and
reports anything missing, empty, or left as a tmp stub.

Exit code 0 iff every scene in the release has a non-empty .sens and no stubs
remain.
"""

import argparse
import os
import os.path as osp
import sys
import urllib.request

BASE = "http://kaldir.vc.cit.tum.de/scannet/"


def release_scans(rel):
    with urllib.request.urlopen(BASE + rel + ".txt", timeout=60) as r:
        return [ln.decode().strip() for ln in r if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="the ScanNet download dir (holds scans/, scans_test/)")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip the release-list fetch; only check what is on disk",
    )
    args = ap.parse_args()

    rc = 0
    for rel, sub in (("v2/scans", "scans"), ("v2/scans_test", "scans_test")):
        d = osp.join(args.root, sub)
        if not osp.isdir(d):
            print(f"{sub}: MISSING directory {d}")
            rc = 1
            continue
        on_disk = {s for s in os.listdir(d) if osp.isdir(osp.join(d, s))}
        if args.offline:
            expected = on_disk
        else:
            try:
                expected = set(release_scans(rel))
            except Exception as e:
                print(f"{sub}: could not fetch release list ({e}); use --offline")
                rc = 1
                continue

        missing, empty = [], []
        for s in sorted(expected):
            p = osp.join(d, s, f"{s}.sens")
            if not osp.isfile(p):
                missing.append(s)
            elif os.path.getsize(p) == 0:
                empty.append(s)
        stubs = [
            osp.join(r, f)
            for r, _, fs in os.walk(d)
            for f in fs
            if f.startswith("tmp") and not f.endswith(".sens")
        ]
        good = len(expected) - len(missing) - len(empty)
        print(f"{sub}: {good}/{len(expected)} scenes have a non-empty .sens")
        for label, items in (
            ("MISSING", missing),
            ("EMPTY (delete the scene dir and re-run)", empty),
            ("LEFTOVER tmp STUBS (safe to delete)", stubs),
        ):
            if items:
                print(f"  {label}: {len(items)}")
                for i in items[:10]:
                    print(f"    {i}")
        if missing or empty:
            rc = 1
    if rc:
        print("\nRe-run download_scannet.sh; --skip_existing resumes the gaps.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
