"""Zip-layout parity: the inode-safe frames.zip layout is byte-for-byte
equivalent to the original loose-file layout, at every pipeline stage.

Each dataset is built twice from ONE synthetic raw input -- once with
--extracted (the original behaviour) and once with the default frames.zip --
and the two outputs are compared member by member. Anything that differs is a
regression in the conversion, because the only intended difference is where
the bytes are stored.

Checks, in order:
  1. zipio unit: virtual-path splitting, listdir/exists over both layouts,
     np_load, and the per-PID handle cache that keeps a forked DataLoader
     worker from sharing a parent's ZipFile;
  2. ScanNet: synthetic .sens -> extract_scannet_sens (both layouts) ->
     preprocess_scannet (both layouts) -> generate_set_scannet, comparing
     bytes at each hop;
  3. HAMMER: synthetic raw sequence -> preprocess_hammer (both layouts);
  4. TartanAir: synthetic raw, as downloaded zips AND as an extracted tree ->
     preprocess_tartanair (both layouts), which also proves the reader
     handles both input forms;
  5. loader equivalence: ScanNet_Multi (dust3r and streamvggt) returns
     identical views from the zip tree and the extracted tree.

Run: python tests/zip_layout_parity.py
"""

import io
import os
import os.path as osp
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "datasets_preprocess"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import imageio  # noqa: E402

from dust3r.utils.zipio import (  # noqa: E402
    SceneZipWriter,
    exists as zexists,
    frames_root,
    listdir as zlistdir,
    np_load,
    read_bytes,
    split_zip_path,
)

PY = sys.executable
FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def tree_bytes(root):
    """Map every file under `root` to its bytes, keyed by relative path, with
    frames.zip transparently expanded so the two layouts are comparable."""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            full = osp.join(dirpath, fn)
            rel = osp.relpath(full, root)
            if fn == "frames.zip":
                prefix = osp.dirname(rel)
                with zipfile.ZipFile(full) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        out[osp.join(prefix, info.filename)] = zf.read(info.filename)
                        assert info.compress_type == zipfile.ZIP_STORED, (
                            f"{rel}:{info.filename} is compressed, expected STORED"
                        )
            else:
                with open(full, "rb") as f:
                    out[rel] = f.read()
    return out


def compare_trees(label, a_root, b_root, npz_keys_only=()):
    """Compare two output trees. Entries whose name ends with one of
    `npz_keys_only` are compared by array content rather than raw bytes,
    because np.savez embeds a timestamp in its zip headers."""
    a, b = tree_bytes(a_root), tree_bytes(b_root)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        return check(
            label, False, f"member mismatch: only_a={only_a[:3]} only_b={only_b[:3]}"
        )
    diffs = []
    for k in sorted(a):
        if a[k] == b[k]:
            continue
        if k.endswith(npz_keys_only):
            xa, xb = np.load(io.BytesIO(a[k])), np.load(io.BytesIO(b[k]))
            if set(xa.files) == set(xb.files) and all(
                np.array_equal(xa[f], xb[f]) for f in xa.files
            ):
                continue
        diffs.append(k)
    return check(label, not diffs, f"{len(a)} members, differing: {diffs[:3]}")


# ----------------------------------------------------------------- 1. zipio
def test_zipio(tmp):
    print("\n[1] zipio unit")
    d = osp.join(tmp, "zipio")
    os.makedirs(osp.join(d, "scene", "color"), exist_ok=True)
    payload = b"\x89PNG-not-really"
    with open(osp.join(d, "scene", "color", "0.jpg"), "wb") as f:
        f.write(payload)
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez(osp.join(d, "scene", "cam.npz"), a=arr)

    with SceneZipWriter(osp.join(d, "scene", "frames.zip")) as w:
        w.writestr("color/0.jpg", payload)
        buf = io.BytesIO()
        np.savez(buf, a=arr)
        w.writestr("cam.npz", buf.getvalue())

    check("split: bare archive", split_zip_path("/a/f.zip") == ("/a/f.zip", ""))
    check(
        "split: member", split_zip_path("/a/f.zip/c/0.jpg") == ("/a/f.zip", "c/0.jpg")
    )
    check("split: plain", split_zip_path("/a/c/0.jpg") == (None, "/a/c/0.jpg"))

    zroot = frames_root(osp.join(d, "scene"))
    check("frames_root finds zip", zroot.endswith("frames.zip"))
    check("read_bytes via zip", read_bytes(osp.join(zroot, "color/0.jpg")) == payload)
    check(
        "read_bytes via disk",
        read_bytes(osp.join(d, "scene", "color", "0.jpg")) == payload,
    )
    check("listdir zip subdir", zlistdir(osp.join(zroot, "color")) == ["0.jpg"])
    check("listdir zip root", set(zlistdir(zroot)) == {"color", "cam.npz"})
    check("exists zip member", zexists(osp.join(zroot, "color/0.jpg")))
    check("exists zip dir prefix", zexists(osp.join(zroot, "color")))
    check("exists false", not zexists(osp.join(zroot, "color/nope.jpg")))
    check(
        "np_load zip == disk",
        np.array_equal(
            np_load(osp.join(zroot, "cam.npz"))["a"],
            np_load(osp.join(d, "scene", "cam.npz"))["a"],
        ),
    )
    try:
        read_bytes(zroot)
        check("read_bytes(archive root) raises", False)
    except IsADirectoryError:
        check("read_bytes(archive root) raises", True)

    # regression: exists() must agree with listdir() on a bare '<x>.zip' (it
    # used to return False for an archive that is plainly there) and must
    # return a bool, not raise, for a file that ends in .zip but is not a
    # readable archive -- e.g. a truncated download
    check("exists(bare archive) is True", zexists(zroot))
    bad = osp.join(d, "truncated.zip")
    with open(bad, "wb") as f:
        f.write(b"PK\x03\x04 not a real archive")
    try:
        check("exists(non-archive .zip) is False", zexists(bad) is False)
    except Exception as e:
        check("exists(non-archive .zip) is False", False, f"raised {type(e).__name__}")

    # fork safety: a child must not reuse the parent's ZipFile handle
    r, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            got = read_bytes(osp.join(zroot, "color/0.jpg"))
            os.write(w_fd, b"1" if got == payload else b"0")
        finally:
            os._exit(0)
    os.close(w_fd)
    ok = os.read(r, 1) == b"1"
    os.close(r)
    os.waitpid(pid, 0)
    check("read after fork (per-pid handle)", ok)


# --------------------------------------------------------------- 2. ScanNet
def write_sens(path, n_frames=4, w=32, h=24):
    """Write a minimal but format-valid ScanNet .sens (v4)."""
    rng = np.random.default_rng(0)
    with open(path, "wb") as f:
        f.write(struct.pack("I", 4))
        name = b"synthetic"
        f.write(struct.pack("Q", len(name)))
        f.write(name)
        K = np.eye(4, dtype=np.float32)
        K[0, 0] = K[1, 1] = 20.0
        K[0, 2], K[1, 2] = w / 2, h / 2
        for m in (K, np.eye(4, dtype=np.float32), K, np.eye(4, dtype=np.float32)):
            f.write(struct.pack("f" * 16, *m.flatten()))
        f.write(struct.pack("i", 2))  # color: jpeg
        f.write(struct.pack("i", 1))  # depth: zlib_ushort
        f.write(struct.pack("I", w))
        f.write(struct.pack("I", h))
        f.write(struct.pack("I", w))
        f.write(struct.pack("I", h))
        f.write(struct.pack("f", 1000.0))
        f.write(struct.pack("Q", n_frames))
        for i in range(n_frames):
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = [i * 0.1, 0, 0]
            f.write(struct.pack("f" * 16, *c2w.flatten()))
            f.write(struct.pack("Q", i))
            f.write(struct.pack("Q", i))
            color = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            cbuf = io.BytesIO()
            imageio.imwrite(cbuf, color, format="JPEG")
            cdata = cbuf.getvalue()
            depth = (rng.random((h, w)) * 3000 + 500).astype(np.uint16)
            ddata = zlib.compress(depth.tobytes())
            f.write(struct.pack("Q", len(cdata)))
            f.write(struct.pack("Q", len(ddata)))
            f.write(cdata)
            f.write(ddata)


def build_scannet_raw(dst, scenes=("scene0000_00", "scene0001_00")):
    for split in ("scans_train", "scans_test"):
        for scene in scenes:
            d = osp.join(dst, split, scene)
            os.makedirs(d, exist_ok=True)
            write_sens(osp.join(d, f"{scene}.sens"))


def test_scannet(tmp):
    print("\n[2] ScanNet: .sens -> extract -> preprocess -> generate_set")
    raw_zip, raw_ext = osp.join(tmp, "sn_raw_zip"), osp.join(tmp, "sn_raw_ext")
    build_scannet_raw(raw_zip)
    build_scannet_raw(raw_ext)
    ex = osp.join(ROOT, "datasets_preprocess", "extract_scannet_sens.py")
    for root, extra in ((raw_zip, []), (raw_ext, ["--extracted"])):
        subprocess.run(
            [PY, ex, "--raw-root", root] + extra, check=True, capture_output=True
        )
    # .sens inputs are identical in both trees; drop them from the comparison
    for t in (raw_zip, raw_ext):
        for dp, _, fns in os.walk(t):
            for fn in fns:
                if fn.endswith(".sens"):
                    os.remove(osp.join(dp, fn))
    compare_trees("extract: zip == extracted", raw_ext, raw_zip)

    # rebuild the .sens (preprocess needs nothing else, but keep trees clean)
    proc_zip, proc_ext = osp.join(tmp, "sn_proc_zip"), osp.join(tmp, "sn_proc_ext")
    pp = osp.join(ROOT, "datasets_preprocess", "preprocess_scannet.py")
    for src, dst, extra in (
        (raw_zip, proc_zip, []),
        (raw_ext, proc_ext, ["--extracted"]),
    ):
        subprocess.run(
            [PY, pp, "--scannet_dir", src, "--output_dir", dst] + extra,
            check=True,
            capture_output=True,
        )
    compare_trees(
        "preprocess: zip == extracted", proc_ext, proc_zip, npz_keys_only=(".npz",)
    )

    gs = osp.join(ROOT, "datasets_preprocess", "generate_set_scannet.py")
    for dst in (proc_zip, proc_ext):
        subprocess.run(
            [
                PY,
                gs,
                "--root",
                dst,
                "--splits",
                "scans_train",
                "scans_test",
                "--max_interval",
                "150",
                "--num_workers",
                "2",
            ],
            check=True,
            capture_output=True,
        )
    compare_trees(
        "generate_set: zip == extracted", proc_ext, proc_zip, npz_keys_only=(".npz",)
    )
    return proc_zip, proc_ext


# ---------------------------------------------------------------- 3. HAMMER
def build_hammer_raw(dst, seqs, as_zip):
    rng = np.random.default_rng(1)
    for seq in seqs:
        files = {}
        K = np.array([[20.0, 0, 16.0], [0, 20.0, 12.0], [0, 0, 1.0]])
        files["polarization/intrinsics.txt"] = (
            "\n".join(" ".join(f"{v:.6f}" for v in row) for row in K) + "\n"
        ).encode()
        for i in range(3):
            b = f"{i:06d}"
            rgb = (rng.random((24, 32, 3)) * 255).astype(np.uint8)
            files[f"polarization/rgb/{b}.png"] = cv2.imencode(".png", rgb)[1].tobytes()
            depth = (rng.random((24, 32)) * 2000 + 500).astype(np.uint16)
            files[f"polarization/_gt/{b}.png"] = cv2.imencode(".png", depth)[
                1
            ].tobytes()
            pose = np.eye(4)
            pose[:3, 3] = [i * 0.1, 0, 0]
            files[f"polarization/_pose/{b}.txt"] = (
                "\n".join(" ".join(f"{v:.6f}" for v in row) for row in pose) + "\n"
            ).encode()
        seq_dir = osp.join(dst, seq)
        os.makedirs(seq_dir, exist_ok=True)
        if as_zip:
            with SceneZipWriter(osp.join(seq_dir, "frames.zip")) as w:
                for name, data in files.items():
                    w.writestr(name, data)
        else:
            for name, data in files.items():
                p = osp.join(seq_dir, name)
                os.makedirs(osp.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(data)


def test_hammer(tmp):
    print("\n[3] HAMMER: raw (both input layouts) -> preprocess (both outputs)")
    import preprocess_hammer as ph

    seqs = [f"scene{s}_traj1_1" for s in (2, 12)]
    raw_zip, raw_ext = osp.join(tmp, "hm_raw_zip"), osp.join(tmp, "hm_raw_ext")
    build_hammer_raw(raw_zip, seqs, as_zip=True)
    build_hammer_raw(raw_ext, seqs, as_zip=False)

    out = {}
    for tag, src, as_zip in (
        ("zipin_zipout", raw_zip, True),
        ("zipin_extout", raw_zip, False),
        ("extin_zipout", raw_ext, True),
        ("extin_extout", raw_ext, False),
    ):
        dst = osp.join(tmp, f"hm_{tag}")
        for seq in seqs:
            os.makedirs(osp.join(dst, ph.split_of_sequence(seq), seq), exist_ok=True)
            ph.process_sequence(seq, src, dst, as_zip)
        out[tag] = dst

    compare_trees(
        "hammer: zip input == extracted input",
        out["extin_extout"],
        out["zipin_extout"],
        npz_keys_only=(".npz",),
    )
    compare_trees(
        "hammer: zip output == extracted output",
        out["extin_extout"],
        out["extin_zipout"],
        npz_keys_only=(".npz",),
    )


# ------------------------------------------------------------- 4. TartanAir
def build_tartanair_raw(dst, as_zip, env="testenv", diff="Easy", traj="P000", n=3):
    rng = np.random.default_rng(2)
    assets = {"image_left": {}, "depth_left": {}, "flow_flow": {}, "flow_mask": {}}
    poses = []
    for i in range(n):
        rgb = (rng.random((24, 32, 3)) * 255).astype(np.uint8)
        assets["image_left"][f"image_left/{i:06d}_left.png"] = cv2.imencode(
            ".png", rgb
        )[1].tobytes()
        b = io.BytesIO()
        np.save(b, rng.random((24, 32)).astype(np.float32))
        assets["depth_left"][f"depth_left/{i:06d}_left_depth.npy"] = b.getvalue()
        if i < n - 1:
            b = io.BytesIO()
            np.save(b, rng.random((24, 32, 2)).astype(np.float32))
            assets["flow_flow"][f"flow/{i:06d}_{i + 1:06d}_flow.npy"] = b.getvalue()
            b = io.BytesIO()
            np.save(b, np.ones((24, 32), dtype=np.uint8))
            assets["flow_mask"][f"flow/{i:06d}_{i + 1:06d}_mask.npy"] = b.getvalue()
        poses.append([i * 0.1, 0, 0, 0, 0, 0, 1])
    pose_txt = (
        "\n".join(" ".join(f"{v:.6f}" for v in p) for p in poses) + "\n"
    ).encode()

    os.makedirs(dst, exist_ok=True)
    if as_zip:
        for asset, members in assets.items():
            zpath = osp.join(dst, f"{env}_{diff}_{asset}.zip")
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in members.items():
                    zf.writestr(f"{env}/{diff}/{traj}/{name}", data)
                if asset == "image_left":
                    zf.writestr(f"{env}/{diff}/{traj}/pose_left.txt", pose_txt)
    else:
        base = osp.join(dst, env, diff, traj)
        for members in assets.values():
            for name, data in members.items():
                p = osp.join(base, name)
                os.makedirs(osp.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(data)
        with open(osp.join(base, "pose_left.txt"), "wb") as f:
            f.write(pose_txt)


def test_tartanair(tmp):
    print("\n[4] TartanAir: raw (zips and tree) -> preprocess (both outputs)")
    import preprocess_tartanair as pt

    raw_zip, raw_ext = osp.join(tmp, "ta_raw_zip"), osp.join(tmp, "ta_raw_ext")
    build_tartanair_raw(raw_zip, as_zip=True)
    build_tartanair_raw(raw_ext, as_zip=False)

    out = {}
    for tag, src, as_zip in (
        ("zipin_zipout", raw_zip, True),
        ("zipin_extout", raw_zip, False),
        ("extin_extout", raw_ext, False),
    ):
        dst = osp.join(tmp, f"ta_{tag}")
        pt.main(src, dst, as_zip=as_zip)
        out[tag] = dst

    compare_trees(
        "tartanair: zip input == extracted input",
        out["extin_extout"],
        out["zipin_extout"],
        npz_keys_only=(".npz",),
    )
    compare_trees(
        "tartanair: zip output == extracted output",
        out["zipin_extout"],
        out["zipin_zipout"],
        npz_keys_only=(".npz",),
    )


# ------------------------------------------------------- 5. loader equality
def test_loaders(proc_zip, proc_ext):
    print("\n[5] loader equivalence on the two trees")
    # imported inside the guard on purpose: these pull in torch/torchvision,
    # which can be absent or broken in an env that runs the pipeline fine
    # (the pipeline itself needs no torch). An ImportError here must be one
    # reported FAIL, not an exception that discards stages 1-4's results.
    try:
        from dust3r.datasets.scannet import ScanNet_Multi as D3
        from streamvggt.datasets.scannet import ScanNet_Multi as SV
        from streamvggt.datasets.types import Split
    except Exception as e:
        check("loader imports", False, f"{type(e).__name__}: {e}")
        return

    # a fixed seed is required, not cosmetic: without it each dataset draws
    # its own crop/stride choices and the two would differ for reasons that
    # have nothing to do with the storage layout
    for name, mk in (
        (
            "dust3r ScanNet_Multi",
            lambda r: D3(
                split="train",
                ROOT=r,
                resolution=(32, 24),
                num_views=2,
                aug_crop=0,
                seed=42,
            ),
        ),
        (
            "streamvggt ScanNet_Multi",
            lambda r: SV(
                split=Split.TRAIN, ROOT=r, resolution=(32, 24), num_views=2, seed=42
            ),
        ),
    ):
        try:
            a, b = mk(proc_zip), mk(proc_ext)
            va, vb = a[(0, 0, 2)], b[(0, 0, 2)]
            same = len(va) == len(vb) and all(
                np.array_equal(np.asarray(x["img"]), np.asarray(y["img"]))
                and np.array_equal(x["depthmap"], y["depthmap"])
                and np.array_equal(x["camera_pose"], y["camera_pose"])
                and np.array_equal(x["camera_intrinsics"], y["camera_intrinsics"])
                for x, y in zip(va, vb)
            )
            check(f"{name}: zip view == extracted view", same, f"{len(va)} views")
        except Exception as e:
            check(
                f"{name}: zip view == extracted view", False, f"{type(e).__name__}: {e}"
            )


def main():
    tmp = tempfile.mkdtemp(prefix="ziplayout_")
    try:
        test_zipio(tmp)
        proc_zip, proc_ext = test_scannet(tmp)
        test_hammer(tmp)
        test_tartanair(tmp)
        test_loaders(proc_zip, proc_ext)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("=" * 60))
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL ZIP-LAYOUT PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
