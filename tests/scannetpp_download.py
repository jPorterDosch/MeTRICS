"""End-to-end test of the patched download_scannetpp.py against a fake server.

Serves a synthetic ScanNet++ release out of a local directory by intercepting
urlretrieve, then exercises: keep_zipped on/off, resume after truncation,
crash-atomicity of the .part rename, and verify_scannetpp.py's verdicts.
"""

import io
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
import zipfile
import urllib.error
import yaml

DL = osp.join(osp.dirname(osp.abspath(__file__)), "..", "datasets_download")
sys.path.insert(0, DL)

SCENES = ["aaaa111111", "bbbb222222"]
# in no split list, but the server serves all its assets -- mirrors the real
# 5656608266 / 6464461276 / 7977624358
NO_SPLIT_SCENE = "deadbeef99"
# kept as archives: members live under "<dirname>/" (verified against the
# live server -- dslr/colmap.zip holds colmap/cameras.txt etc.)
ASSETS_ZIP = {
    "dslr_colmap_dir": ("dslr/colmap", ["cameras.txt", "images.txt", "points3D.txt"]),
    "dslr_resized_dir": ("dslr/resized_images", ["DSC00001.JPG", "DSC00002.JPG"]),
    "dslr_resized_mask_dir": (
        "dslr/resized_anon_masks",
        ["DSC00001.png", "DSC00002.png"],
    ),
    "iphone_colmap_dir": (
        "iphone/colmap",
        ["cameras.txt", "images.txt", "points3D.txt"],
    ),
}
# zip-wrapped for transport but NOT kept: extracted on arrival, member sits at
# the archive root (mesh_aligned_0.05.zip -> mesh_aligned_0.05.ply)
ASSETS_ZIP_EXTRACT = {"scan_mesh_path": "scans/mesh_aligned_0.05.ply"}
# served raw
ASSETS_FILE = {"iphone_video_path": "iphone/rgb.mkv"}


def build_server(root):
    """Materialise the blobs the real server would hand out, keyed by filepath."""
    blobs = {}
    for s in SCENES + [NO_SPLIT_SCENE]:
        for asset, (reldir, files) in ASSETS_ZIP.items():
            buf = io.BytesIO()
            # server archives are DEFLATED, like the real ones
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.writestr(
                        f"{osp.basename(reldir)}/{f}", f"{s}:{reldir}/{f}".encode() * 40
                    )
            blobs[f"data/{s}/{reldir}.zip"] = buf.getvalue()
        for asset, rel in ASSETS_ZIP_EXTRACT.items():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(osp.basename(rel), f"{s}:{rel}".encode() * 40)
            blobs[f"data/{s}/{osp.splitext(rel)[0]}.zip"] = buf.getvalue()
        for asset, rel in ASSETS_FILE.items():
            blobs[f"data/{s}/{rel}"] = f"{s}:{rel}".encode() * 40
    for split, ids in (("nvs_sem_train", SCENES), ("nvs_sem_val", [])):
        blobs[f"splits/{split}.txt"] = ("\n".join(ids) + "\n").encode()
    return blobs


class FakeNet:
    """Stand-in for urlretrieve. `fail_at` raises mid-transfer to model a kill."""

    def __init__(self, blobs):
        self.blobs, self.fail_at, self.n = blobs, None, 0

    def urlretrieve(self, url, filename):
        self.n += 1
        path = url.split("filepath=", 1)[1]
        if path not in self.blobs:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b""))
        data = self.blobs[path]
        if self.fail_at is not None and self.n >= self.fail_at:
            # write a partial body, then die -- exactly what a walltime kill does
            with open(filename, "wb") as f:
                f.write(data[: len(data) // 2])
            raise KeyboardInterrupt("simulated job kill")
        with open(filename, "wb") as f:
            f.write(data)
        return filename, None


def make_cfg(data_root, keep_zipped):
    cfg = yaml.safe_load(open(osp.join(DL, "download_scannetpp.yml")))
    cfg.update(
        root_url="https://fake/download?token=TOKEN&filepath=FILEPATH",
        token="tok",
        data_root=data_root,
        keep_zipped=(list(ASSETS_ZIP) if keep_zipped else []),
        zipped_assets=list(ASSETS_ZIP) + list(ASSETS_ZIP_EXTRACT),
        download_assets=list(ASSETS_ZIP) + list(ASSETS_ZIP_EXTRACT) + list(ASSETS_FILE),
        splits=["nvs_sem_train", "nvs_sem_val"],
        meta_files=["splits/nvs_sem_train.txt", "splits/nvs_sem_val.txt"],
        exclude_assets={},
        download_scenes=SCENES,
        verbose=False,
    )
    cfg.pop("download_splits", None)
    p = osp.join(data_root, "_cfg.yml")
    os.makedirs(data_root, exist_ok=True)
    yaml.safe_dump(cfg, open(p, "w"))
    return p


def run(cfg_path, net, answer="y"):
    import download_scannetpp as D

    D.urlretrieve = net.urlretrieve
    D.input = lambda *a, **k: answer  # the y/n disk-space prompt
    import argparse

    D.main(argparse.Namespace(config_file=cfg_path))


def count_inodes(root):
    return sum(len(d) + len(f) for _, d, f in os.walk(root))


def verify(root):
    r = subprocess.run(
        [
            sys.executable,
            osp.join(DL, "verify_scannetpp.py"),
            root,
            "--config",
            osp.join(root, "_cfg.yml"),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode, r.stdout.strip()


def main():
    blobs = build_server(None)
    tmp = tempfile.mkdtemp(prefix="spp_")
    fails = []

    def check(name, cond, detail=""):
        print(
            f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}"
        )
        if not cond:
            fails.append(name)

    # ---- 1. keep_zipped: true ------------------------------------------
    print("\n1. keep_zipped: true")
    zr = osp.join(tmp, "zipped")
    cfg = make_cfg(zr, True)
    net = FakeNet(blobs)
    run(cfg, net)
    z_inodes = count_inodes(zr)
    kept = [osp.join(zr, "data", SCENES[0], "dslr", "resized_images.zip")]
    check("archives kept on disk", all(osp.isfile(p) for p in kept))
    check(
        "archives NOT extracted",
        not osp.isdir(osp.join(zr, "data", SCENES[0], "dslr", "resized_images")),
    )
    check(
        "transport-zipped single file IS extracted",
        osp.isfile(osp.join(zr, "data", SCENES[0], "scans", "mesh_aligned_0.05.ply")),
    )
    check(
        "its archive is not left behind",
        not osp.isfile(
            osp.join(zr, "data", SCENES[0], "scans", "mesh_aligned_0.05.zip")
        ),
    )
    rc, out = verify(zr)
    check("verify_scannetpp passes", rc == 0, out.splitlines()[0])

    # ---- 2. keep_zipped: false (stock behaviour still works) -----------
    print("\n2. keep_zipped: false  (unchanged stock path)")
    er = osp.join(tmp, "extracted")
    cfg2 = make_cfg(er, False)
    run(cfg2, FakeNet(blobs))
    e_inodes = count_inodes(er)
    check(
        "archives extracted",
        osp.isdir(osp.join(er, "data", SCENES[0], "dslr", "resized_images")),
    )
    check(
        "zip deleted after extract",
        not osp.isfile(osp.join(er, "data", SCENES[0], "dslr", "resized_images.zip")),
    )

    # ---- 3. payload parity between the two layouts ---------------------
    print("\n3. byte parity, zipped vs extracted")
    diffs = 0
    for s in SCENES:
        for asset, (reldir, files) in ASSETS_ZIP.items():
            zp = osp.join(zr, "data", s, reldir + ".zip")
            with zipfile.ZipFile(zp) as zf:
                for f in files:
                    a = zf.read(f"{osp.basename(reldir)}/{f}")
                    b = open(osp.join(er, "data", s, reldir, f), "rb").read()
                    diffs += a != b
        for rel in list(ASSETS_ZIP_EXTRACT.values()) + list(ASSETS_FILE.values()):
            a = open(osp.join(zr, "data", s, rel), "rb").read()
            b = open(osp.join(er, "data", s, rel), "rb").read()
            diffs += a != b
    check("every member identical", diffs == 0, f"{diffs} differences")
    check(
        "zip layout uses fewer inodes", z_inodes < e_inodes, f"{z_inodes} vs {e_inodes}"
    )

    # ---- 4. resume after truncation ------------------------------------
    print("\n4. resume after a truncated archive")
    victim = osp.join(zr, "data", SCENES[1], "dslr", "resized_images.zip")
    full = os.path.getsize(victim)
    with open(victim, "r+b") as f:
        f.truncate(full // 2)
    rc, out = verify(zr)
    check("verify DETECTS truncation", rc != 0, out.splitlines()[0])
    net2 = FakeNet(blobs)
    run(cfg, net2)
    check("re-download repaired it", os.path.getsize(victim) == full)
    check("only the broken asset re-fetched", net2.n == 1, f"{net2.n} request(s)")
    rc, out = verify(zr)
    check("verify passes again", rc == 0, out.splitlines()[0])

    # ---- 5. crash mid-transfer leaves no final-named file ---------------
    print("\n5. atomicity: job killed mid-transfer")
    cr = osp.join(tmp, "crash")
    cfg3 = make_cfg(cr, True)
    net3 = FakeNet(blobs)
    net3.fail_at = 8  # crash after several assets have landed
    try:
        run(cfg3, net3)
    except KeyboardInterrupt:
        pass
    strays = [
        osp.join(dp, f)
        for dp, _, fs in os.walk(cr)
        for f in fs
        if f.endswith((".zip", ".mkv", ".ply"))
    ]
    bad = [p for p in strays if p.endswith(".zip") and not _zip_ok(p)]
    check(
        "no truncated file under a final name",
        not bad and len(strays) >= 5,
        f"{len(bad)} bad of {len(strays)} landed",
    )
    parts = [f for dp, _, fs in os.walk(cr) for f in fs if f.endswith(".part")]
    check("no .part sidecar left behind", not parts, f"{parts}")

    # ---- 6. a scene in no split warns but still downloads --------------
    # Three scenes in DUSt3R's list are in no published split yet serve every
    # asset, so this must NOT abort the run -- `split` only selects
    # exclude_assets, and None means "exclude nothing". It must still say so.
    print("\n6. scene present in no split")
    sr = osp.join(tmp, "nosplit")
    cfg4 = make_cfg(sr, True)
    import contextlib
    import io as _io
    import yaml as _y

    c = _y.safe_load(open(cfg4))
    c["download_scenes"] = SCENES + [NO_SPLIT_SCENE]
    _y.safe_dump(c, open(cfg4, "w"))
    err = _io.StringIO()
    raised = None
    try:
        with contextlib.redirect_stderr(err):
            run(cfg4, FakeNet(blobs))
    except BaseException as e:  # noqa: BLE001 -- any abort is the failure here
        raised = f"{type(e).__name__}: {e}"
    check(
        "does NOT abort the run",
        raised is None,
        raised or "completed",
    )
    check(
        "warns naming the scene",
        NO_SPLIT_SCENE in err.getvalue(),
        err.getvalue().strip().splitlines()[-1][:70] if err.getvalue() else "no warning",
    )
    # the in-split scenes must still have landed
    landed = [d for d in os.listdir(osp.join(sr, "data"))] if osp.isdir(osp.join(sr, "data")) else []
    check(
        "in-split scenes still downloaded",
        all(s in landed for s in SCENES),
        f"{sorted(landed)}",
    )

    shutil.rmtree(tmp)
    print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
    return 1 if fails else 0


def _zip_ok(p):
    try:
        with zipfile.ZipFile(p):
            return True
    except Exception:
        return False


sys.exit(main())
