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
HTTP range requests. Then `python datasets_preprocess/preprocess_hammer.py`.

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

Also download DUSt3R's precomputed pairs, which supply the scene list and the
per-scene image selection:
```
source datasets_download/env.sh
cd "$SCANNETPP_DIR"
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip
unzip scannetpp_pairs.zip
```

> **Preprocessing is not wired up yet.** ScanNet++ ships iPhone RGB as
> `iphone/rgb.mkv`, but `preprocess_scannetpp.py` reads `iphone/rgb/` and
> `iphone/rgb_masks/` as directories, so an ffmpeg decode stage sits between
> download and preprocess and this repo has no script for it (`module load
> ffmpeg` — it is not on PATH by default). `preprocess_scannetpp.py` and
> `generate_set_scannetpp.py` are also still the stock loose-file versions.
> Only the download side and the dataloader understand archives so far.

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
