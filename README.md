<div align="center">
<h1>MeTRICS: Metric Temporally-consistent Reconstruction In Causal Streaming</h1>
</div>

### [Paper]TODO  | [Project Page]TODO | [Online Demo]TODO

>Metric Temporally-consistent Reconstrution In Causal Streaming

>Porter Dosch<sup>\*</sup>, Luise Haller<sup>\*</sup>, Faisal Zaghloul, James Tompkin

<sup>*</sup> Equal contribution.

**MeTRICS**, a causal transformer architecture for **temporally consistent 4D geometry generation** built off of [StreamVGGT](https://github.com/wzzheng/StreamVGGT), delivers both fast inference and temporally-consistent 4D reconstruction.

## News
TODO: as project progresses, update this section

## Overview
TODO: add overview of method once everything finalized

### Installation

1. Clone MeTRICS
```bash
git clone https://github.com/jPorterDosch/MeTRICS.git
cd MeTRICS
```
2. Create conda environment
```bash
conda create -n MeTRICS python=3.11 cmake=3.14.0
conda activate MeTRICS
```

3. Install requirements
```bash
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

### Download Checkpoints
Pre-trained StreamVGGT is also available at both [Hugging Face](https://huggingface.co/lch01/StreamVGGT/) and [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/).

To download from huggingface, after installing `requirements.txt`, run
```
hf download lch01/StreamVGGT \
  --local-dir ./StreamVGGT
```

### Logging (Weights & Biases)

Training runs are logged to [wandb](https://wandb.ai). Set your API key
(from https://wandb.ai/authorize) before launching:

    export WANDB_API_KEY=your_key_here

To run without logging, set `WANDB_MODE=offline` or `WANDB_MODE=disabled`.

## Data Preparation

All download scripts share one storage location, defined in
`datasets_download/env.sh` and overridable from the environment:

```
$METRICS_DATA_ROOT/            # default /lustre/isaac24/proj/UTK0516/metrics_data
├── arkit_scenes/ hammer/ scannet/ scannetpp/ tartanair/   raw downloads
└── processed/                                             preprocess outputs
```

To stage the data somewhere else, set `METRICS_DATA_ROOT` (or a single
per-dataset variable such as `SCANNET_DIR`) rather than editing the scripts:

```
METRICS_DATA_ROOT=/some/other/place bash datasets_download/download_hammer.sh
```

Each script carries its own `#SBATCH` header, so the same file runs either
way — `bash <script>` on a login node, or `sbatch <script>` to queue it. On
ISAAC prefer `sbatch`; every one is resumable and safe to re-submit. Create the
`logs/` directory in whatever directory you submit from first, since Slurm
opens the job's output file before the script runs.

#### Storage layout: one uncompressed zip per scene

Lustre charges every file to the *user* who created it as well as to the
project, and the per-user inode limit is far tighter than the space limit
(check with `lfs quota -h -u $USER /lustre/isaac24`). Stored loose, these
datasets are millions of tiny per-frame files — ScanNet alone would be ~7.5M
inodes extracted and as many again after preprocessing, which overruns a
normal quota outright.

So every stage stores frames in a single **uncompressed (`ZIP_STORED`)
archive per scene** instead:

```
<scene>/frames.zip        members: color/0.jpg, depth/0.png, cam/00000.npz, ...
<scene>/*_metadata.npz    always real files — the loaders read them directly
```

Uncompressed means readers seek straight to a member's bytes with no
decompression, so this costs no CPU and ~0.1% in size, and buys a ~1000x
reduction in inodes. `src/dust3r/utils/zipio.py` implements it; the loaders
accept either layout transparently, and raw archives (ArKitScenes' per-asset
zips, TartanAir's 144 downloaded zips) are read in place and never unzipped.

Five scripts take `--extracted` to write the original loose layout instead:
`download_hammer.py`, `extract_scannet_sens.py`, and `preprocess_{scannet,
hammer,tartanair}.py`. The two layouts are byte-for-byte identical where both
exist — `python tests/zip_layout_parity.py` builds ScanNet, HAMMER and
TartanAir both ways from one input and compares every member, end to end.

The ArKitScenes and ScanNet++ preprocessing has no such flag: it writes
`frames.zip` unconditionally, and the parity test does not cover it.

### Disk budget

Measured, not estimated: per-frame byte counts taken from one real scene of
each, scaled by frame count. The Lustre quota that binds is **per-uid** and
spans every path on the filesystem, not just this project dir — check it with
`lfs quota -u $USER /lustre/isaac24` before submitting anything.

| stage | writes to | size |
| --- | --- | --- |
| ScanNet download | `scannet/` | 1073 GiB (1613 `.sens`) |
| ScanNet extract | `scannet/`, in place beside the `.sens` | ~651 GiB |
| ScanNet preprocess + generate_set | `processed/processed_scannet` | ~424 GiB |
| TartanAir download | `tartanair/` | 828 GiB (144 zips) |
| TartanAir preprocess | `processed/processed_tartanair` | 1404 GiB |
| HAMMER download | `hammer/` | 22 GiB |
| HAMMER preprocess | `processed/processed_hammer` | ~22 GiB |
| ScanNet++ download | `scannetpp/` | 155 GiB |
| ScanNet++ preprocess + generate_set | `processed/processed_scannetpp` | ~29 GiB |

Ratios are weighted means over 6-7 scenes each, through the real encoders, not
a single sample: ScanNet extract is 0.606x the `.sens` and the processed tree
0.395x, and the TartanAir figure is the exact sum of every member's
uncompressed size across all 144 archives (its output is `ZIP_STORED`).

**The whole set does not comfortably fit in a 10 TiB quota alongside the raw
downloads.** The stage-2 ScanNet `frames.zip` files are an intermediate —
nothing downstream of `preprocess_scannet.sh` reads them — so deleting them
once that job has verified reclaims ~651 GiB and is what makes the rest fit
with room to spare. Regenerating them costs ~1.5 h per shard, against ~2 h of
re-download for the `.sens` they come from, so they are the right thing to
drop. Inodes are a non-issue in this layout — the whole processed tree is
about 4k files.

### Training Datasets
#### ARKitScenes
Download the raw data using the script provided by Apple
`bash download_arkit_scenes.sh`

Download the precomputed pairs provided by DUST3R
```
source datasets_download/env.sh
mkdir -p "$ARKIT_DIR"
cd "$ARKIT_DIR"

wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/arkitscenes_pairs.zip

unzip arkitscenes_pairs.zip
```

Then run:
```
python preprocess_arkitscenes.py --arkitscenes_dir /path/to/your/raw/data --precomputed_pairs /path/to/your/pairs --output_dir /path/to/your/outdir

python generate_set_arkitscenes.py --root /path/to/your/outdir --splits Training Test --max_interval 5.0 --num_workers 8
```

#### HAMMER
`bash datasets_download/download_hammer.sh` (or `sbatch` it)
fetches the ~24 GB polarization-camera subset out of the 170 GB official zip via
HTTP range requests. Then:
```
sbatch datasets_preprocess/preprocess_hammer.sh   # or: bash it
```
There is no generate_set stage — `preprocess_hammer.py` writes the
`scene_metadata.npz` the loader reads. A few minutes for all 64 sequences,
~22 GB out.

#### ScanNet
ScanNet's own `download-scannet.py` is **not in this repo** — TUM mails it out
only after a signed [Terms of
Use](http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf) is approved, so
redistributing it here is not ours to do. Send the signed agreement from an
institutional address to `scannet@googlegroups.com`, then save the script they
reply with as `datasets_download/download_scannet.py` (gitignored). The wrapper
checks for it and prints these instructions if it is missing.

`bash datasets_download/download_scannet.sh` (or `sbatch` it) then downloads the
`.sens` subset of ScanNet v2. **Running it constitutes agreeing to those Terms
of Use.**
Then the two preprocessing jobs (the second runs preprocess + generate_set):
```
sbatch datasets_download/extract_scannet.sh      # .sens -> frames.zip (32-way array)
sbatch datasets_preprocess/preprocess_scannet.sh # frames -> processed + metadata
```

#### TartanAir
`bash datasets_download/download_tartanair.sh` (or `sbatch` it) fetches 144 zips (~889 GB: image_left, depth_left,
flow_flow, flow_mask for Easy+Hard) from the AirLab mirror. They are **not**
unzipped — `datasets_preprocess/preprocess_tartanair.py` reads members straight
out of them, so the raw data costs 144 inodes. Needs `boto3` and `colorama`.
Then:
```
sbatch datasets_preprocess/preprocess_tartanair.sh   # 36-way array
```
There is no generate_set stage — `src/dust3r/datasets/tartanair.py` walks the
output tree and reads no metadata npz. One array task per (env, difficulty),
which is the 36 `*_image_left.zip` archives; keep `--array` and `NUM_SHARDS` in
step. **This output is ~1.5 TB, not ~830 GB:** the processed archives are
`ZIP_STORED` while the upstream ones are DEFLATE, and image/depth/flow/mask
inflate 2.0x/6.4x/1.1x/82x. Check the quota before submitting.

#### ScanNet++
Needs a personal download token (register and sign the terms at
[kaldir.vc.in.tum.de/scannetpp](https://kaldir.vc.in.tum.de/scannetpp/)). Put it
in `datasets_download/.scannetpp_token` (gitignored, mode 600) or export
`SCANNETPP_TOKEN` — it is a per-user credential tied to a signed ToU, so keep it
out of the repo and off job command lines. Then:

```
bash datasets_download/download_scannetpp.sh     # or: sbatch it
```

**~112 GiB**, versus the ~1.5 TB the stock config's defaults pull. It fetches
only what `preprocess_scannetpp.py` reads: the 228 scenes in DUSt3R's
`scannetpp_pairs/scene_list.json` and the 7 assets that script opens — no
iPhone depth (depth is *rendered* from the mesh), point clouds, semantic
meshes, nerfstudio transforms, original/undistorted DSLR, or panocam.

DUSt3R's list has 229 entries; `280b83fcf3` is excluded because it was removed
from the release between v1 and v2 — its mesh 404s, and preprocessing needs
that mesh. Leaving it in aborts the whole download.

The four DSLR/iPhone *directory* archives are kept zipped and read in place, so
a scene costs **11 inodes** (7 files + 4 dirs, measured) instead of the ~1,650
loose files inside — about 2.5k inodes for the whole dataset rather than ~375k. Single-file
assets that are merely zip-wrapped for transport (the mesh, `rgb_mask`) are
extracted — keeping those zipped saves no inodes. `verify_scannetpp.py` runs at
the end; re-runs are resumable.

That script also fetches DUSt3R's precomputed pairs (26 MB, no token) into
`$SCANNETPP_DIR/scannetpp_pairs`, which supply both the scene list and the
per-scene image selection. If your download predates that, re-run it — it
skips the 155 GiB it already has — or fetch them by hand:
```
source datasets_download/env.sh
cd "$SCANNETPP_DIR"
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip
unzip scannetpp_pairs.zip
```

Then two submissions, the second after the first finishes:
```
jid=$(sbatch --parsable datasets_preprocess/preprocess_scannetpp.sh)  # 32-way
sbatch --dependency=afterok:$jid \
    datasets_preprocess/preprocess_scannetpp_finalize.sh
```
Stage 1 renders each scene to a `frames.zip`; stage 2 builds `all_metadata.npz`
and the per-scene `new_scene_metadata.npz` the loader reads. They are separate
because `all_metadata.npz` concatenates every scene and cannot be written until
the last shard is done.

Three things differ from the stock DUSt3R script, all forced by the data or the
cluster (the header of `preprocess_scannetpp.py` has the detail):

* **iPhone frames come out of a video.** The release ships `iphone/rgb.mkv` and
  `iphone/rgb_mask.mkv`, not the `iphone/rgb/` and `iphone/rgb_masks/`
  directories the stock script opens. The frame number in a COLMAP name
  (`frame_%06d.jpg`) is the frame's index in those videos, so the mapping is
  exact. Decoding uses OpenCV's bundled FFMPEG backend — no `module load
  ffmpeg`, no separate extract stage.
* **Depth is ray-cast, not rasterised.** Depth is not distributed; it is
  rendered from `scans/mesh_aligned_0.05.ply`. Upstream uses pyrender, which
  needs a working OpenGL: ISAAC has `libEGL` but no swrast DRI driver and no
  GPU on the CPU partitions, so `eglInitialize` fails outright. This renders
  with Embree on the CPU instead (`embreex`), which also means it queues on
  `campus` rather than waiting for a GPU. Validated independently of both
  renderers: projecting each image's own COLMAP sparse 3D points with the pose
  and intrinsics used for the render reproduces the rendered depth to a median
  of **19 mm**, on a mesh decimated at 5 cm.
* **A small tail of the selected images no longer exists.** DUSt3R computed
  its selections against an older ScanNet++, and some of the images it picked
  are no longer registered. A selected name with no COLMAP entry has no pose,
  so it cannot be rendered — the stock script dies on a bare `KeyError`.
  Those images keep their slot in `scene_metadata.npz` (`pairs` indexes into
  `selection`, so its length and order are load-bearing) with a **NaN** pose,
  and are simply absent from `frames.zip`. That is the case the rest of the
  pipeline was already built for: `generate_set_scannetpp.py` filters pairs by
  what is on disk, and the loader tests `imgs[i] in imgs_on_disk`. NaN rather
  than identity, so anything that does reach for one fails loudly.

  **63,846 of the 64,923 selected images (98.3%) survive** — DSLR
  33,769/33,818 (99.9%), iPhone 30,077/31,105 (96.7%); the remainder are
  ordinary per-scene registration failures. Watch for the count of scenes with
  no usable images from one camera: the loader takes
  `max(dslr_ids) < min(iphone_ids)`, which raises rather than skipping, so
  `--finalize` drops such scenes from `all_metadata.npz` and names them. With
  this release it drops none.

  Two parsing traps sit behind those numbers, both of which quietly *look*
  like properties of the data. An iPhone reconstruction carries no sparse 2D
  points at all (`mean observations per image: 0.0`), so its `POINTS2D` line
  is **empty** — skipping blank lines desynchronises `images.txt`'s
  two-lines-per-image alternation and keeps only every *other* pose, which
  reads exactly like a stride-20 registration. And seven scenes prefix their
  COLMAP entries with `video/`, which `re.match` silently refuses, making
  those scenes look DSLR-only. `load_sfm` preserves blank lines, takes the
  basename, and checks the parsed count against the header's
  `# Number of images:` so neither can recur unnoticed.

### Evaluation Datasets
Please refer to [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare Sintel, Bonn, KITTI, NYU-v2, ScanNet, 7scenes and Neural-RGBD datasets. 

For Sintel, Bonn, and KITTI, download scripts are available in `datasets_download`; preprocessing scripts are available in `datasets_preprocess`. These scripts are taken directly from the MONST3R repo for ease of use.
Download: `bash datasets_download/download_<name>.sh`
Preprocess: `python datasets_preprocess prepare_<name>.sh`
Sintel preprocessing is omitted since it is not necessary.

## Folder Structure
The overall folder structure should be organized as follows：
TODO: fill in rest of this once we have decided on directory structure
```
MeTRICS
├── ckpt/
|   ├── model.pt
|   └── checkpoints.pth
|
└── src/
    ├── ...
```

## Evaluation
TODO: add eval scripts

## Acknowledgements
Our code is based on the following repositories:

[StreamVGGT](https://github.com/wzzheng/streamvggt)
[DUSt3R](https://github.com/naver/dust3r)
[MonST3R](https://github.com/Junyi42/monst3r.git)
[Spann3R](https://github.com/HengyiWang/spann3r.git)
[CUT3R](https://github.com/CUT3R/CUT3R)
[VGGT](https://github.com/facebookresearch/vggt)
[Point3R](https://github.com/YkiWu/Point3R)

## Citation
TODO: add citation after any paper is written/uploaded
