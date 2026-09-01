#!/usr/bin/env python3
"""Verify a ScanNet++ download before preprocessing trusts it.

download_scannetpp.py aborts the whole run on the first asset it cannot fetch
(`download_has_error`), so a partial tree is the normal result of a timed-out
job, a 404 on a scene that no longer exists in the current release, or an
expired token part-way through. It also treats any file that merely EXISTS as
complete, so nothing downstream re-checks these.

Checks, for each of the 229 scenes DUSt3R's scene_list.json names:
  * every asset in the config is present;
  * each *.zip opens (its central directory parses -- catches the truncation
    a killed transfer leaves behind);
  * each zip actually contains members under the directory it stands in for,
    since preprocess reads '<asset>.zip/<asset-dir>/...' paths;
  * no .part sidecars are left over from an interrupted download.

--deep additionally CRC-checks every member (slow); the default is quick and
catches the failure modes that actually occur.

Exit code 0 iff every scene is complete and intact.
"""

import argparse
import os
import os.path as osp
import sys
import zipfile

import yaml

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from scene_release import ScannetppScene_Release  # noqa: E402

DEFAULT_CFG = osp.join(osp.dirname(osp.abspath(__file__)), "download_scannetpp.yml")


def check_zip(path, deep=False):
    """Return None if the archive is sound, else a short reason string."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                return "empty"
            if deep and zf.testzip() is not None:
                return "CRC"
    except zipfile.BadZipFile as e:
        return f"bad zip: {e}"
    except OSError as e:
        return f"unreadable: {e}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="the ScanNet++ download dir (config data_root)")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--deep", action="store_true", help="CRC-check every member")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scenes = cfg["download_scenes"]
    assets = cfg["download_assets"]
    # Only the assets in keep_zipped stay as archives on disk. The rest of
    # zipped_assets were zip-wrapped for transport and extracted on arrival,
    # so on disk they are ordinary files/dirs -- checking them as archives
    # would report every one of them missing.
    keep = cfg.get("keep_zipped") or []
    if isinstance(keep, bool):
        keep = list(cfg["zipped_assets"]) if keep else []
    keep = set(keep)

    data_root = osp.join(args.root, "data")
    if not osp.isdir(data_root):
        print(f"ScanNet++: no data/ under {args.root} -- nothing downloaded yet")
        return 1

    missing, corrupt, empty_dir = [], [], []
    complete = 0
    for scene_id in scenes:
        scene = ScannetppScene_Release(scene_id, data_root=data_root)
        ok = True
        for asset in assets:
            tgt = getattr(scene, asset)
            if asset in keep:
                # keep_zipped leaves the archive in place of the directory
                path = tgt.with_suffix(".zip")
                if not path.is_file():
                    # tolerate an extracted tree (keep_zipped: false)
                    if tgt.is_dir():
                        continue
                    missing.append(f"{scene_id}/{asset}")
                    ok = False
                    continue
                why = check_zip(path, args.deep)
                if why:
                    corrupt.append(f"{scene_id}/{asset} ({why})")
                    ok = False
                    continue
                # preprocess addresses members as '<asset>.zip/<dirname>/...',
                # so an archive whose members sit under some other prefix
                # would read as an empty directory much later, inside
                # preprocess_scannetpp.py's load_sfm/Image.open.
                with zipfile.ZipFile(path) as zf:
                    prefix = tgt.name + "/"
                    if not any(n.startswith(prefix) for n in zf.namelist()):
                        empty_dir.append(f"{scene_id}/{asset} (no {prefix}* members)")
                        ok = False
            else:
                if not tgt.is_file():
                    missing.append(f"{scene_id}/{asset}")
                    ok = False
        complete += ok

    # a .part anywhere means a transfer was cut mid-flight
    leftovers = []
    for dirpath, _, filenames in os.walk(data_root):
        for f in filenames:
            if f.endswith(".part"):
                leftovers.append(osp.join(dirpath, f))

    print(f"ScanNet++: {complete}/{len(scenes)} scenes complete")
    for label, items in (
        ("MISSING", missing),
        ("CORRUPT (delete and re-run)", corrupt),
        ("UNEXPECTED LAYOUT", empty_dir),
        ("LEFTOVER .part (safe to delete)", leftovers),
    ):
        if items:
            print(f"  {label}: {len(items)}")
            for i in items[:10]:
                print(f"    {i}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
    if missing or corrupt or empty_dir:
        print("\nDelete any listed archive and re-run download_scannetpp.sh.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
