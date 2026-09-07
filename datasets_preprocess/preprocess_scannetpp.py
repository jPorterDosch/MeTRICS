#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Convert raw ScanNet++ into the processed per-scene layout.
#
# Derived from DUSt3R's preprocess_scannetpp.py. What changed, and why:
#
#  * Reads the download layout. download_scannetpp.sh keeps the four
#    DSLR/iPhone *directory* assets zipped and reads them in place --
#    <scene>/dslr/{colmap,resized_images,resized_anon_masks}.zip and
#    <scene>/iphone/colmap.zip -- because a scene costs 11 inodes that way
#    instead of ~1,650. Their members are prefixed with the asset name, which
#    is exactly the shape zipio.asset_root() resolves, so both that layout and
#    a hand-extracted tree work with no flag.
#
#  * Decodes the iPhone video. iPhone RGB is not a directory in either
#    layout: the release ships iphone/rgb.mkv and iphone/rgb_mask.mkv, while
#    the upstream script opens iphone/rgb/ and iphone/rgb_masks/. The frame
#    number in a COLMAP name (frame_%06d.jpg) IS the frame's index in those
#    videos, which is what makes the mapping exact. Decoding goes through
#    OpenCV's bundled FFMPEG backend, so no `module load ffmpeg` and no
#    separate extract stage.
#
#  * Ray-casts depth instead of rasterising it. Depth is not distributed; it
#    is rendered from scans/mesh_aligned_0.05.ply. Upstream uses pyrender,
#    which needs a working OpenGL. This cluster has libEGL but no swrast DRI
#    driver (/usr/lib64/dri/swrast_dri.so is absent) and no GPU on the CPU
#    partitions, so eglInitialize fails outright with EGL_NOT_INITIALIZED.
#    Depth here is ray-cast against the same mesh with Embree (trimesh +
#    embreex): CPU only, no drivers, and it runs on `campus` alongside every
#    other preprocessing job rather than queueing for a GPU.
#
#    For a first-hit query a z-buffer and a ray cast agree, and the result was
#    checked without reference to either renderer: projecting each image's own
#    COLMAP sparse 3D points with the very pose and intrinsics used for the
#    render reproduces the rendered depth to a median of 19 mm, on a mesh
#    that is itself decimated at 5 cm.
#
#  * Normalises COLMAP image names. Most scenes register iPhone frames as
#    "frame_000020.jpg", but seven register them as "video/frame_000020.jpg".
#    Upstream applies REGEXPR_IPHONE with re.match, which anchors at position
#    0, so a prefixed name matches nothing and those scenes look as though not
#    one of their selected iPhone frames was ever registered. Taking the
#    basename fixes all seven, taking the count of scenes with no usable
#    iPhone frames from 7 to 0.
#
#  * Skips the SfM data nothing reads. Upstream's load_sfm parses
#    points3D.txt and every image's sparse 2D observations, then uses neither
#    (its own comment says "we will only use the intrinsics and pose here").
#    On a DSLR scene that is a 65 MB file plus the 346 MB bulk of images.txt,
#    and slurping images.txt with .read().splitlines() peaks near 1 GB. This
#    version streams images.txt and keeps only the pose lines.
#    subsample_img_infos(), also dead upstream, is gone.
#
#  * Writes one uncompressed frames.zip per scene rather than loose files,
#    matching every other processed dataset here, then --finalize
#    concatenates the per-scene metadata into all_metadata.npz.
#
# Usage:
#   python3 datasets_preprocess/preprocess_scannetpp.py \
#       --scannetpp_dir <raw> --precomputed_pairs <scannetpp_pairs> \
#       --output_dir <out> [--shard i --num-shards N]
#   python3 datasets_preprocess/preprocess_scannetpp.py ... --finalize
# --------------------------------------------------------
import argparse
import contextlib
import io
import json
import os
import os.path as osp
import re
import sys
import zipfile

import cv2
import numpy as np
import PIL.Image as Image
import trimesh
import trimesh.exchange.ply
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import path_to_root  # noqa
from datasets_preprocess.utils.cropping import rescale_image_depthmap  # noqa

# same sibling-import workaround preprocess_arkitscenes.py uses: path_to_root
# puts the repo root on sys.path, but the packages live under src/
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))
from dust3r.utils.zipio import (  # noqa: E402
    SceneZipWriter,
    asset_root,
    read_bytes,
    split_zip_path,
)

REGEXPR_DSLR = re.compile(r"^DSC(?P<frameid>\d+).JPG$")
REGEXPR_IPHONE = re.compile(r"frame_(?P<frameid>\d+).jpg$")

# default values from
# https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
ZNEAR = 0.05
ZFAR = 20.0


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scannetpp_dir", required=True)
    parser.add_argument(
        "--precomputed_pairs",
        required=True,
        help="DUSt3R's scannetpp_pairs dir: scene_list.json plus one "
        "<scene>/selected_pairs.npz per scene. It supplies the scene list "
        "AND the per-scene image selection, so there is no default.",
    )
    parser.add_argument("--output_dir", default="data/scannetpp_processed")
    parser.add_argument(
        "--target_resolution", default=920, type=int, help="images resolution"
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="this shard index (0-based)"
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="total number of shards"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="convert nothing; concatenate the per-scene scene_metadata.npz "
        "into all_metadata.npz. Run once, after every shard has finished.",
    )
    parser.add_argument(
        "--keep-unusable-scenes",
        action="store_true",
        help="with --finalize, keep scenes that rendered images from only one "
        "of the two cameras. They are excluded by default because "
        "ScanNetpp_Multi._load_data raises on them rather than skipping them.",
    )
    return parser


# ---------------------------------------------------------------------------
# intrinsics conventions
#
# Inlined rather than imported from dust3r.utils.geometry, which pulls in
# torch: torch is not in the preprocessing env, and no other
# datasets_preprocess script depends on it (preprocess_arkitscenes.py, the
# sibling that already reads zips, avoids it the same way). These two are the
# whole of what was used.
# ---------------------------------------------------------------------------
def colmap_to_opencv_intrinsics(K):
    """Colmap puts the centre of the top-left pixel at (0.5, 0.5); OpenCV
    puts it at (0, 0)."""
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """Inverse of colmap_to_opencv_intrinsics."""
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type="dslr"):
    if cam_type == "dslr":
        regex_expr = REGEXPR_DSLR
    elif cam_type == "iphone":
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f"wrong {cam_type=} for get_frame_number")
    matches = re.match(regex_expr, name)
    return matches["frameid"]


@contextlib.contextmanager
def open_text(path):
    """Line-oriented read of a text file in either layout.

    Streams instead of slurping. A DSLR colmap images.txt is ~350 MB, almost
    all of it the per-image sparse 2D observations that nothing here needs,
    and read_bytes().decode().splitlines() peaks around 1 GB on one scene.
    The ZipFile has to outlive the member stream (closing it closes the
    underlying fd), which is what the nested `with` is for.
    """
    archive, member = split_zip_path(path)
    if archive is None:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield f
        return
    with zipfile.ZipFile(archive) as zf, zf.open(member, "r") as raw:
        yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


@contextlib.contextmanager
def open_binary_stream(path):
    """Binary stream over a file in either layout (trimesh wants a file
    object, and the meshes are ~100 MB each)."""
    archive, member = split_zip_path(path)
    if archive is None:
        with open(path, "rb") as f:
            yield f
        return
    with zipfile.ZipFile(archive) as zf, zf.open(member, "r") as raw:
        # trimesh's ply reader seeks, which a raw deflate stream cannot do
        yield io.BytesIO(raw.read())


def load_sfm(sfm_dir, cam_type="dslr"):
    """Read a COLMAP text reconstruction: intrinsics and poses only.

    Returns (img_idx, img_infos), dropping upstream's points3D and
    observations return values -- it computed both and used neither.
    """
    intrinsics = {}
    with open_text(osp.join(sfm_dir, "cameras.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            camera = line.split()
            intrinsics[int(camera[0])] = [camera[1]] + [
                float(cam) for cam in camera[2:]
            ]

    # images.txt alternates a pose line and a sparse-2D-points line per image.
    # Only the pose lines are parsed; the points lines are the file's bulk and
    # are skipped without being split.
    #
    # ONLY comment lines are skipped, never blank ones. An iPhone
    # reconstruction carries no sparse 2D observations at all ("mean
    # observations per image: 0.0"), so its POINTS2D line is the empty string
    # -- dropping blank lines desynchronises the alternation and silently
    # keeps every OTHER image. That looked exactly like a stride-20 iPhone
    # registration against DUSt3R's stride-10 selection, i.e. like a property
    # of the data, which is why the count check below is not optional.
    expected = None
    img_idx = {}
    img_infos = {}
    with open_text(osp.join(sfm_dir, "images.txt")) as f:
        expect_pose = True
        for line in f:
            if line.startswith("#"):
                m = re.search(r"Number of images:\s*(\d+)", line)
                if m:
                    expected = int(m.group(1))
                continue
            if not expect_pose:
                expect_pose = True  # this was the points2D line
                continue
            expect_pose = False
            image = line.split()
            if len(image) != 10:
                raise ValueError(
                    f"{sfm_dir}/images.txt: expected a 10-field pose line "
                    f"(IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME), got "
                    f"{len(image)} fields: {line[:120]!r}"
                )
            idx = image[0]
            # basename, because the release is not consistent about this: most
            # scenes register iPhone frames as "frame_000020.jpg", but some
            # register them as "video/frame_000020.jpg". Upstream's
            # REGEXPR_IPHONE is applied with re.match, which anchors at
            # position 0, so a prefixed name matches nothing -- those scenes
            # look as though not one of their selected iPhone frames was ever
            # registered. DSLR names carry no prefix today; normalising both
            # costs nothing and means one less thing to be surprised by.
            img_name = osp.basename(image[-1])
            if img_name in img_idx:
                raise ValueError(f"{sfm_dir}/images.txt: duplicate image {img_name}")
            img_idx[img_name] = idx
            img_infos[idx] = dict(
                intrinsics=intrinsics[int(image[-2])],
                path=img_name,
                frame_id=get_frame_number(img_name, cam_type),
                cam_to_world=pose_from_qwxyz_txyz(image[1:-2]),
            )

    # The header states how many images the reconstruction holds; anything
    # else means the alternation slipped and poses were silently dropped.
    if expected is not None and len(img_idx) != expected:
        raise ValueError(
            f"{sfm_dir}/images.txt: header declares {expected} images but "
            f"{len(img_idx)} were parsed -- the pose/points2D alternation is "
            "out of step"
        )
    return img_idx, img_infos


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        if len(distortion) != 4:
            raise ValueError(
                f"OPENCV_FISHEYE expects 4 distortion coefficients, got "
                f"{len(distortion)}: {distortion}"
            )

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (width, height), 1, (width, height), True
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )

    undistorted_image = cv2.remap(
        rgb,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    undistorted_mask = cv2.remap(
        mask,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    K = opencv_to_colmap_intrinsics(K)
    return width, height, new_K, undistorted_image, undistorted_mask


class MeshDepthRenderer:
    """Z-depth of a triangle mesh for a pinhole camera, by ray casting.

    Embree on the CPU, so this needs no OpenGL context, no GPU and no driver
    -- see the header for why pyrender is not usable here. The BVH is built
    once per scene and reused for all ~300 of its images.
    """

    def __init__(self, ply_path):
        with open_binary_stream(ply_path) as f:
            mesh_kwargs = trimesh.exchange.ply.load_ply(f)
        # process=True (the default, and what upstream relies on) merges
        # duplicate vertices and drops degenerate faces; keep it so the
        # geometry matches the reference implementation's.
        self.mesh = trimesh.Trimesh(**mesh_kwargs)
        # imported here, not at module scope: embreex is only needed by this
        # class, and a clear ImportError at the point of use beats a failure
        # during argument parsing.
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        self.intersector = RayMeshIntersector(self.mesh)

    def render(self, K, cam_to_world, width, height):
        """Depth in metres, 0 where no surface is hit or it is out of range.

        The rays are built with z == 1 in camera space rather than as unit
        vectors, so the ray parameter of a hit IS its z-depth: taking
        (hit - eye) . optical_axis recovers it exactly, with no cosine
        correction and no reliance on the hit distance.

        K is whatever the caller passes; convert_one_image passes the matrix
        it also saves, matching upstream's pyrender call. See the comment
        there for the convention and for the size of the offset that carries.
        """
        u, v = np.meshgrid(
            np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
        )
        d_cam = np.stack(
            [
                (u - K[0, 2]) / K[0, 0],
                (v - K[1, 2]) / K[1, 1],
                np.ones_like(u),
            ],
            axis=-1,
        ).reshape(-1, 3)

        R = cam_to_world[:3, :3]
        eye = cam_to_world[:3, 3]
        directions = d_cam @ R.T
        origins = np.broadcast_to(eye, directions.shape)

        _, idx_ray, locations = self.intersector.intersects_id(
            origins, directions, multiple_hits=False, return_locations=True
        )

        depth = np.zeros(width * height, dtype=np.float64)
        if len(idx_ray):
            depth[idx_ray] = (locations - eye) @ (R @ np.array([0.0, 0.0, 1.0]))
        # same near/far clipping pyrender's IntrinsicsCamera applies, so
        # out-of-range surfaces read as "no depth" rather than as garbage
        depth[(depth < ZNEAR) | (depth > ZFAR)] = 0.0
        return depth.reshape(height, width)


class VideoFrameReader:
    """Pull specific frame indices out of an mkv, in index order.

    Sequential decode with an explicit counter, deliberately NOT
    CAP_PROP_POS_FRAMES seeking: frame-accurate seeking in H.264 depends on
    keyframe placement and on the backend build, and a seek that lands a few
    frames off would silently pair an image with the wrong camera pose. The
    cost of being sure is one linear pass -- ~25 s for rgb.mkv and ~145 s for
    rgb_mask.mkv per scene, against ~250 s of ray casting for the same scene.

    grab() skips the colour conversion and copy for frames we are not
    keeping, which is most of them (COLMAP registers every 20th frame).
    """

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0

    def read_at(self, index):
        """Decode frame `index`. Indices must be requested in ascending
        order; going backwards would need a re-open, and nothing here does."""
        if index < self.pos:
            raise ValueError(
                f"{self.path}: frame {index} requested after {self.pos}; "
                "indices must ascend"
            )
        while self.pos < index:
            if not self.cap.grab():
                raise RuntimeError(
                    f"{self.path}: stream ended at frame {self.pos}, "
                    f"before the requested {index}"
                )
            self.pos += 1
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"{self.path}: failed to decode frame {index}")
        self.pos += 1
        return frame

    def close(self):
        self.cap.release()


def convert_one_image(rgb, mask, img_infos_idx, renderer, target_resolution, emit):
    """Undistort, rescale, write the jpg, render the depth, write the png.

    Updates img_infos_idx["intrinsics"] in place to the colmap-convention
    matrix that matches what was written, which is what the scene metadata
    then records.
    """
    _, _, K, rgb, mask = undistort_images(img_infos_idx["intrinsics"], rgb, mask)

    # rescale_image_depthmap assumes opencv intrinsics. The mask rides along
    # in the depthmap slot -- it is nearest-neighbour resized, which is what
    # a mask wants.
    intrinsics = colmap_to_opencv_intrinsics(K)
    image, mask, intrinsics = rescale_image_depthmap(
        rgb,
        mask,
        intrinsics,
        (target_resolution, target_resolution * 3.0 / 4),
    )
    W, H = image.size

    intrinsics = opencv_to_colmap_intrinsics(intrinsics)
    img_infos_idx["intrinsics"] = intrinsics

    # Render with the matrix that is SAVED, which is what upstream does.
    #
    # This is deliberately not the geometrically exact choice, and the size of
    # the difference is known. undistort_images returns new_K already in
    # OpenCV convention (it is built from an OpenCV-convention K by OpenCV's
    # own routines), and the undistorted image is an ideal pinhole with it --
    # measured at 8.6e-5 px worst case for both models in the release, by
    # re-distorting the pinhole ray for each destination pixel and comparing
    # against initUndistortRectifyMap's own source coordinates. So the exact
    # chain would feed new_K straight to rescale_image_depthmap, which both
    # consumes and returns OpenCV convention.
    #
    # Upstream instead applies colmap_to_opencv to new_K first and
    # opencv_to_colmap after, which leaves the saved matrix a constant
    # (1 - 0.5 * scale) px from the exact one -- about 0.7 px at the DSLR
    # scale of 0.591, 0.76 px at the iPhone's 0.479. Matching it keeps the
    # metadata numerically identical to what the previous DUSt3R/CroCo
    # preprocessing wrote, and keeps depth and intrinsics mutually consistent
    # in exactly the way that pipeline's outputs are, since it renders with
    # the saved matrix too. The whole offset is a shared shift of depth and
    # intrinsics against the image, not a disagreement between them.
    #
    # To take the exact geometry instead, pass `K` (i.e. new_K) to
    # rescale_image_depthmap above and drop both conversions -- but that
    # changes the numbers a previously-trained checkpoint saw.
    depth = renderer.render(intrinsics, img_infos_idx["cam_to_world"], W, H)

    basename = img_infos_idx["path"].rsplit(".", 1)[0]

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    emit(f"images/{basename}.jpg", buf.getvalue())

    depth = (depth * 1000).astype("uint16")
    # invalidate depth wherever the anonymisation mask is not fully opaque.
    # The mask arrives from a lossy video for iPhone frames, and interpolated
    # by the undistort remap for both, so anything short of 255 is treated as
    # masked -- the same test upstream applies.
    if mask.ndim == 3:
        mask = mask[..., 0]
    depth[mask < 255] = 0
    ok, enc = cv2.imencode(".png", depth)
    if not ok:
        raise RuntimeError(f"failed to encode depth for {basename}")
    emit(f"depth/{basename}.png", enc.tobytes())


def process_scene(scene, root, pairsdir, output_dir, target_resolution):
    """Convert one scene. Returns the number of images written."""
    data_dir = osp.join(root, "data", scene)
    dir_dslr = osp.join(data_dir, "dslr")
    dir_iphone = osp.join(data_dir, "iphone")
    dir_scans = osp.join(data_dir, "scans")

    selected = osp.join(pairsdir, scene, "selected_pairs.npz")
    if not osp.isfile(selected):
        raise FileNotFoundError(f"{scene}: no selected_pairs.npz at {selected}")
    with np.load(selected) as npz:
        selection, pairs = npz["selection"], npz["pairs"]

    output_dir_scene = osp.join(output_dir, scene)
    os.makedirs(output_dir_scene, exist_ok=True)

    # both cameras' COLMAP reconstructions, then the mesh BVH
    dslr_colmap = asset_root(dir_dslr, "colmap")
    iphone_colmap = asset_root(dir_iphone, "colmap")
    if dslr_colmap is None or iphone_colmap is None:
        raise FileNotFoundError(f"{scene}: missing a colmap asset")
    img_idx_dslr, img_infos_dslr = load_sfm(dslr_colmap, cam_type="dslr")
    img_idx_iphone, img_infos_iphone = load_sfm(iphone_colmap, cam_type="iphone")

    renderer = MeshDepthRenderer(osp.join(dir_scans, "mesh_aligned_0.05.ply"))

    dslr_rgb = asset_root(dir_dslr, "resized_images")
    dslr_mask = asset_root(dir_dslr, "resized_anon_masks")
    if dslr_rgb is None or dslr_mask is None:
        raise FileNotFoundError(f"{scene}: missing a DSLR image/mask asset")

    # Not every selected image still exists to be rendered.
    #
    # DUSt3R computed these selections against an older ScanNet++, and a
    # small tail of the images it picked is no longer registered: 63,846 of
    # the 64,923 selected images survive (98.3%) -- DSLR 33,769/33,818
    # (99.9%), iPhone 30,077/31,105 (96.7%) -- the rest being ordinary
    # per-scene registration failures. A selected name with no COLMAP entry
    # has no pose and no intrinsics, so it cannot be rendered at all --
    # upstream would die here on a bare KeyError.
    #
    # Dropping such an image from the OUTPUT while keeping it in the metadata
    # is exactly the case the rest of the pipeline is already built for:
    # `pairs` indexes into `selection`, so the selection has to keep its
    # length and order, and both generate_set_scannetpp.py (which filters
    # pairs by what is on disk) and the loader (`imgs[i] in imgs_on_disk`)
    # already treat a listed-but-absent image as normal. Their rows in
    # trajectories/intrinsics are filled with NaN rather than identity, so
    # anything that does reach for one fails loudly instead of silently
    # training on a camera at the origin.
    selection_dslr = [
        n + ".JPG"
        for n in selection
        if n.startswith("DSC") and n + ".JPG" in img_idx_dslr
    ]
    selection_iphone = [
        n + ".jpg"
        for n in selection
        if n.startswith("frame_") and n + ".jpg" in img_idx_iphone
    ]

    # frames.zip is written first and renamed only on success, and
    # scene_metadata.npz only after that, so the npz existing implies a
    # complete archive -- that is what the skip check in main() relies on.
    written = 0
    # name (without extension) -> (intrinsics 3x3, cam2world 4x4), recorded as
    # each image is converted. The metadata is built from THIS rather than
    # from img_infos, so a row can only carry real numbers if the matching
    # frame was actually written.
    converted = {}
    with SceneZipWriter(osp.join(output_dir_scene, "frames.zip")) as writer:
        for imgname in tqdm(
            selection_dslr, desc=f"{scene} dslr", position=1, leave=False
        ):
            info = img_infos_dslr[img_idx_dslr[imgname]]
            rgb = np.array(
                Image.open(io.BytesIO(read_bytes(osp.join(dslr_rgb, imgname))))
            )
            mask = np.array(
                Image.open(
                    io.BytesIO(read_bytes(osp.join(dslr_mask, imgname[:-3] + "png")))
                )
            )
            convert_one_image(
                rgb, mask, info, renderer, target_resolution, writer.writestr
            )
            converted[imgname[:-4]] = (info["intrinsics"], info["cam_to_world"])
            written += 1

        if selection_iphone:
            # ascending frame order so the two videos can be walked linearly
            ordered = sorted(
                selection_iphone,
                key=lambda n: int(get_frame_number(n, "iphone")),
            )
            rgb_video = VideoFrameReader(osp.join(dir_iphone, "rgb.mkv"))
            mask_video = VideoFrameReader(osp.join(dir_iphone, "rgb_mask.mkv"))
            try:
                if rgb_video.frame_count != mask_video.frame_count:
                    raise RuntimeError(
                        f"{scene}: rgb.mkv has {rgb_video.frame_count} frames but "
                        f"rgb_mask.mkv has {mask_video.frame_count}; the mask "
                        "index would not line up with the image index"
                    )
                for imgname in tqdm(
                    ordered, desc=f"{scene} iphone", position=1, leave=False
                ):
                    info = img_infos_iphone[img_idx_iphone[imgname]]
                    frame_no = int(get_frame_number(imgname, "iphone"))
                    # cv2 gives BGR; the DSLR path comes through PIL as RGB,
                    # so flip to match before anything touches the pixels
                    rgb = rgb_video.read_at(frame_no)[:, :, ::-1]
                    mask = mask_video.read_at(frame_no)
                    convert_one_image(
                        rgb, mask, info, renderer, target_resolution, writer.writestr
                    )
                    converted[imgname[:-4]] = (
                        info["intrinsics"],
                        info["cam_to_world"],
                    )
                    written += 1
            finally:
                rgb_video.close()
                mask_video.close()

    # metadata, in the order `selection` gives -- `pairs` indexes into it, so
    # the length and order are load-bearing and unrendered images keep their
    # slot with a NaN pose (see the comment above selection_dslr)
    nan_K = np.full((3, 3), np.nan)
    nan_pose = np.full((4, 4), np.nan)
    trajectories = []
    intrinsics = []
    for imgname in selection:
        if not (imgname.startswith("DSC") or imgname.startswith("frame_")):
            raise ValueError(f"{scene}: invalid image name {imgname!r}")
        K, pose = converted.get(imgname, (nan_K, nan_pose))
        intrinsics.append(K)
        trajectories.append(pose)

    n_dslr = sum(1 for n in selection if n.startswith("DSC"))
    n_iphone = len(selection) - n_dslr
    np.savez(
        osp.join(output_dir_scene, "scene_metadata.npz"),
        trajectories=np.stack(trajectories, axis=0),
        intrinsics=np.stack(intrinsics, axis=0),
        images=selection,
        pairs=pairs,
    )
    return dict(
        written=written,
        selected=len(selection),
        dslr=(len(selection_dslr), n_dslr),
        iphone=(len(selection_iphone), n_iphone),
    )


def scene_list(root, pairsdir):
    """The scenes to process: DUSt3R's list, minus any absent from the
    download.

    280b83fcf3 is expected to be missing -- it was removed from the release
    between v1 and v2, its mesh 404s, and download_scannetpp.yml drops it for
    that reason. Anything else missing is worth shouting about.
    """
    with open(osp.join(pairsdir, "scene_list.json"), "r") as f:
        listed = json.load(f)
    present, absent = [], []
    for scene in listed:
        if osp.isdir(osp.join(root, "data", scene)):
            present.append(scene)
        else:
            absent.append(scene)
    if absent:
        print(
            f"{len(absent)} of {len(listed)} listed scenes are not in "
            f"{osp.join(root, 'data')} and will be skipped: "
            f"{', '.join(absent)}",
            file=sys.stderr,
        )
    return present


def finalize(root, pairsdir, output_dir, keep_unusable=False):
    """Concatenate the per-scene metadata into all_metadata.npz.

    Scenes the loader cannot open are left out by default. ScanNetpp_Multi
    takes `max(dslr_ids) < min(iphone_ids)` after filtering both lists to what
    is on disk, so a scene contributing images from only one camera does not
    get skipped -- min() raises ValueError on the empty list and the whole
    dataset fails to construct. all_metadata.npz's scene list is what the
    loader iterates, so dropping them here is the fix that needs no change to
    the training code.

    With the current release nothing is actually excluded: the seven scenes
    that first looked one-camera-only were the ones whose COLMAP entries carry
    a "video/" prefix, and normalising that in load_sfm gives all of them
    usable iPhone frames. This stays as the guard for the next such surprise.
    --keep-unusable-scenes overrides it.
    """
    scenes = scene_list(root, pairsdir)
    missing = [
        s
        for s in scenes
        if not osp.isfile(osp.join(output_dir, s, "scene_metadata.npz"))
    ]
    if missing:
        raise SystemExit(
            f"cannot finalize: {len(missing)} of {len(scenes)} scenes have no "
            f"scene_metadata.npz yet ({', '.join(missing[:10])}"
            f"{', ...' if len(missing) > 10 else ''}). Run the remaining "
            "shards first."
        )

    # First pass: read every scene and count what actually rendered. A row
    # whose pose is NaN is a selected image that could not be rendered -- see
    # the comment above selection_dslr in process_scene.
    loaded = {}
    usable = {}
    for scene in tqdm(scenes, desc="finalize: read"):
        with np.load(osp.join(output_dir, scene, "scene_metadata.npz")) as data:
            entry = dict(
                images=data["images"],
                intrinsics=data["intrinsics"],
                trajectories=data["trajectories"],
                pairs=data["pairs"].copy(),  # offset in place below
            )
        loaded[scene] = entry
        traj = entry["trajectories"]
        ok = np.isfinite(traj.reshape(len(traj), -1)).all(1)
        usable[scene] = (
            sum(
                1 for n, o in zip(entry["images"], ok) if o and str(n).startswith("DSC")
            ),
            sum(
                1
                for n, o in zip(entry["images"], ok)
                if o and str(n).startswith("frame_")
            ),
        )

    unusable = [s for s in scenes if 0 in usable[s]]
    kept = scenes if keep_unusable else [s for s in scenes if s not in unusable]
    if not kept:
        raise SystemExit("no usable scenes; nothing to write")

    counts, sceneids = [], []
    images, intrinsics, trajectories, pairs = [], [], [], []
    offset = 0
    for scene_idx, scene in enumerate(kept):
        entry = loaded[scene]
        n = len(entry["images"])
        sceneids.extend([scene_idx] * n)
        images.append(entry["images"])
        intrinsics.append(entry["intrinsics"])
        trajectories.append(entry["trajectories"])
        scene_pairs = entry["pairs"]
        scene_pairs[:, 0:2] += offset
        pairs.append(scene_pairs)
        counts.append(offset)
        offset += n

    all_trajectories = np.concatenate(trajectories, axis=0)
    rendered = np.isfinite(all_trajectories.reshape(len(all_trajectories), -1)).all(1)

    np.savez(
        osp.join(output_dir, "all_metadata.npz"),
        counts=counts,
        scenes=kept,
        sceneids=sceneids,
        images=np.concatenate(images, axis=0),
        intrinsics=np.concatenate(intrinsics, axis=0),
        trajectories=all_trajectories,
        pairs=np.concatenate(pairs, axis=0),
    )

    print(
        f"all_metadata.npz written: {len(kept)} scenes, {offset} images "
        f"({int(rendered.sum())} rendered, {int((~rendered).sum())} listed but "
        f"absent), {sum(len(p) for p in pairs)} pairs"
    )
    if unusable:
        verb = "KEPT (--keep-unusable-scenes)" if keep_unusable else "excluded"
        print(
            f"{len(unusable)} scene(s) {verb}: no rendered images from one "
            "camera, which ScanNetpp_Multi._load_data cannot open -- "
            + ", ".join(
                f"{s} (dslr {usable[s][0]}, iphone {usable[s][1]})" for s in unusable
            ),
            file=sys.stderr,
        )


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.finalize:
        finalize(
            args.scannetpp_dir,
            args.precomputed_pairs,
            args.output_dir,
            keep_unusable=args.keep_unusable_scenes,
        )
        return

    # A shard id at or above the shard count selects nothing (i % n < n
    # always), which would silently convert zero scenes and exit 0.
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(
            f"--shard {args.shard} is out of range for --num-shards "
            f"{args.num_shards}; shard must be in [0, {args.num_shards})"
        )

    scenes = scene_list(args.scannetpp_dir, args.precomputed_pairs)
    mine = [s for i, s in enumerate(scenes) if i % args.num_shards == args.shard]
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(scenes)} "
        f"scenes assigned",
        flush=True,
    )

    done = skipped = failed = 0
    for scene in tqdm(mine, position=0, leave=True):
        # a complete scene_metadata.npz implies a complete frames.zip; see
        # the write order in process_scene
        if osp.isfile(osp.join(args.output_dir, scene, "scene_metadata.npz")):
            skipped += 1
            continue
        try:
            stats = process_scene(
                scene,
                args.scannetpp_dir,
                args.precomputed_pairs,
                args.output_dir,
                args.target_resolution,
            )
        except Exception as e:  # one bad scene shouldn't kill the shard
            # Same contract as extract_scannet_sens.py: a corrupt mkv or an
            # unexpected COLMAP shape costs its own scene, not the ~6 others
            # this task still has to do. --finalize refuses to run while any
            # scene is missing, so nothing downstream can silently consume a
            # short dataset; the operator re-runs the shard instead.
            failed += 1
            print(f"FAILED {scene}: {e}", file=sys.stderr, flush=True)
            continue
        d_ok, d_sel = stats["dslr"]
        i_ok, i_sel = stats["iphone"]
        print(
            f"{scene}: {stats['written']}/{stats['selected']} images written "
            f"(dslr {d_ok}/{d_sel}, iphone {i_ok}/{i_sel})",
            flush=True,
        )
        # The loader takes max(dslr_ids) < min(iphone_ids), which raises on an
        # empty list -- a scene with no usable images from one camera breaks
        # it. Say so here, and again in --finalize where the whole set is
        # visible at once.
        if d_ok == 0 or i_ok == 0:
            print(
                f"WARNING {scene}: no usable "
                f"{'DSLR' if d_ok == 0 else 'iPhone'} images; "
                "ScanNetpp_Multi._load_data will fail on this scene",
                file=sys.stderr,
                flush=True,
            )
        done += 1

    print(
        f"[shard {args.shard}] converted={done} skipped={skipped} "
        f"failed={failed}; run with --finalize once every shard is done",
        flush=True,
    )
    # Exit non-zero so Slurm marks the task FAILED. Printing a count and
    # exiting 0 would leave a task that converted nothing looking successful,
    # and the gap would only surface at --finalize.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
