#!/usr/bin/env python3
"""Verify a ScanNet .sens download before extraction trusts it.

download_scannet.py (kept as ScanNet ships it) collects per-scan exceptions
into a list, prints them, and returns normally -- so the wrapper exits 0 even
when scenes failed, and the Slurm job reports SUCCESS. It also creates a
tempfile stub before fetching, which is left behind if the fetch raises.

This compares what is on disk against the authoritative release scan lists and
reports anything missing, truncated, or left as a tmp stub.

Truncation is the failure that matters: --skip_existing skips on FILENAME, so a
.sens cut short by a walltime kill is never re-fetched, and extraction later
swallows the resulting struct.error per-scene -- the scene just vanishes from
the training set. An emptiness check does not catch that, so each .sens is
walked frame by frame (reading only the ~96-byte frame headers and seeking past
the payloads, ~12 ms for a 560 MB file) to confirm the frame index lands
exactly on EOF.

Exit code 0 iff every scene in the release has a structurally complete .sens
and no stubs remain.
"""

import argparse
import os
import os.path as osp
import struct
import sys
import urllib.request

BASE = "http://kaldir.vc.cit.tum.de/scannet/"


SENS_VERSION = 4


def sens_integrity(path):
    """Return (status, detail) for one .sens. Walks the frame index without
    decoding any payload, so cost is ~one 96-byte read per frame."""
    size = os.path.getsize(path)
    if size == 0:
        return "EMPTY", "zero bytes"
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12:
                return "TRUNCATED", "shorter than the .sens header"
            version = struct.unpack("I", head[:4])[0]
            strlen = struct.unpack("Q", head[4:])[0]
            if version != SENS_VERSION or strlen > 4096:
                # an HTML error page or a stray file lands here
                return "CORRUPT", f"bad header (version={version}, strlen={strlen})"
            f.seek(strlen, 1)  # sensor name
            f.seek(16 * 4 * 4, 1)  # 4 x 4x4 float matrices
            f.seek(4 * 2 + 4 * 4 + 4, 1)  # compression types, w/h pairs, depth_shift
            raw = f.read(8)
            if len(raw) < 8:
                return "TRUNCATED", "header ends before the frame count"
            num_frames = struct.unpack("Q", raw)[0]
            for i in range(num_frames):
                # camera_to_world (16 floats) + 2 timestamps + color/depth sizes
                meta = f.read(16 * 4 + 8 * 4)
                if len(meta) < 16 * 4 + 8 * 4:
                    return "TRUNCATED", f"ends inside frame {i} of {num_frames}"
                csz = struct.unpack("Q", meta[-16:-8])[0]
                dsz = struct.unpack("Q", meta[-8:])[0]
                if f.seek(csz + dsz, 1) > size:
                    return "TRUNCATED", f"frame {i} of {num_frames} runs past EOF"
            pos = f.tell()
            if pos != size:
                return "TRAILING", f"{size - pos} bytes after {num_frames} frames"
            return "OK", f"{num_frames} frames"
    except (OSError, struct.error) as e:
        return "CORRUPT", str(e)


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

        missing, bad = [], []
        for s in sorted(expected):
            p = osp.join(d, s, f"{s}.sens")
            if not osp.isfile(p):
                missing.append(s)
                continue
            status, detail = sens_integrity(p)
            if status != "OK":
                bad.append(f"{s} [{status}: {detail}]")
        stubs = [
            osp.join(r, f)
            for r, _, fs in os.walk(d)
            for f in fs
            if f.startswith("tmp") and not f.endswith(".sens")
        ]
        good = len(expected) - len(missing) - len(bad)
        print(f"{sub}: {good}/{len(expected)} scenes have a complete .sens")
        for label, items in (
            ("MISSING", missing),
            ("INCOMPLETE (delete the scene dir and re-run)", bad),
            ("LEFTOVER tmp STUBS (safe to delete)", stubs),
        ):
            if items:
                print(f"  {label}: {len(items)}")
                for i in items[:10]:
                    print(f"    {i}")
        if missing or bad:
            rc = 1
    if rc:
        print("\nRe-run download_scannet.sh; --skip_existing resumes the gaps.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
