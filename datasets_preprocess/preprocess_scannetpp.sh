#!/bin/bash
#SBATCH --job-name=scannetpp_preprocess
#SBATCH --array=0-23              # 24 shards; scenes assigned round-robin
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scannetpp_preprocess_%A_%a.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# Stage 1 of 2 for ScanNet++: raw zips + mkv + mesh -> one processed
# frames.zip per scene (images/<name>.jpg, depth/<name>.png) plus a
# scene_metadata.npz. Stage 2 is preprocess_scannetpp_finalize.sh, which
# builds all_metadata.npz and the per-scene new_scene_metadata.npz the loader
# actually reads:
#   sbatch datasets_preprocess/preprocess_scannetpp.sh           # this, 32-way
#   sbatch datasets_preprocess/preprocess_scannetpp_finalize.sh  # then this
# They are two submissions rather than one because all_metadata.npz
# concatenates every scene, so it cannot be written until the last shard has
# finished. To chain them without waiting:
#   jid=$(sbatch --parsable datasets_preprocess/preprocess_scannetpp.sh)
#   sbatch --dependency=afterok:$jid \
#       datasets_preprocess/preprocess_scannetpp_finalize.sh
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_preprocess/preprocess_scannetpp.sh   # 24 parallel shards
#   bash   datasets_preprocess/preprocess_scannetpp.sh   # one process, all scenes
# Outside a job array SLURM_ARRAY_TASK_* are unset, so the fallback below runs
# the whole set as a single shard rather than a silent 1/32nd of it.
#
# NEEDS embreex (pip install embreex). Depth is ray-cast against each scene's
# mesh on the CPU rather than rasterised with pyrender: this cluster has
# libEGL but no swrast DRI driver and no GPU on the CPU partitions, so
# pyrender's eglInitialize fails outright. See the header of
# preprocess_scannetpp.py for the full reasoning and for how the depth was
# validated (median 19 mm against each image's own COLMAP sparse points).
#
# Width is 24, not 32, so the whole pipeline fits under the campus QOS
# MaxSubmitPU of 96 in one submission (1 hammer + 24 here + 1 finalize + 32
# extract + 1 preprocess + 1 cleanup + 24 tartanair = 84 array elements).
#
# Also needs DUSt3R's scannetpp_pairs, which supplies the scene list and the
# per-scene image selection. download_scannetpp.sh fetches it into
# $SCANNETPP_DIR/scannetpp_pairs; if that predates this change, either re-run
# it (idempotent, it will skip the 155 GiB it already has) or fetch by hand:
#   cd "$SCANNETPP_DIR" && curl -fSLO \
#     https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip \
#     && unzip -q scannetpp_pairs.zip
#
# Resumable: a scene with a scene_metadata.npz is skipped, and that npz is
# written only after its frames.zip has been renamed into place, so a skipped
# scene is always complete. Re-running finishes what a timed-out task left.
#
# 280b83fcf3 is in DUSt3R's 229-scene list but not in the release (its mesh
# 404s), so 228 scenes are processed and the missing one is reported once per
# task rather than aborting the run.
#
# Sizing, measured on 036bce3393 (150 DSLR + 131 iPhone images, 112 MB mesh):
#   time  5m43s for the scene, of which ~170 s is decoding the two mkvs and
#         ~0.5 s per image is ray casting. Over 228 scenes / 63,846
#         renderable images that is ~22 core-hours, so ~55 min per task at
#         24 ways. The 12 h wall is slack, not need.
#   size  132 MB of frames.zip for 281 images (~470 KB each: a 1035x690 or
#         920x690 jpg plus a uint16 depth png), so ~29 GiB for the set.
#   mem   2.5 GB peak RSS. Dominated by the mesh and its Embree BVH, so it
#         scales with mesh size: this one is 112 MB against a worst case of
#         256 MB (47b37eb6f9), hence 16G rather than 8G.

set -eu

# Resolving this script's own directory has to survive both invocation paths.
# Under sbatch, Slurm COPIES the script to /var/spool/.../slurm_script and runs
# it from there, so BASH_SOURCE points at the spool dir and every path built
# from it is wrong -- that is why sourcing env.sh failed with "No such file or
# directory". `scontrol show job` still knows the path the job was submitted
# with, so ask Slurm inside a job and fall back to BASH_SOURCE outside one.
# (sbatch --test-only does NOT run the body, so it cannot catch a bug here.)
#
# Query SLURM_ARRAY_JOB_ID, not SLURM_JOB_ID, and keep only the first
# Command=. In a job ARRAY exactly one task inherits the array's master job id
# -- observed on 6177682_31, whose JobIDRaw was the array id 6177682 itself --
# and `scontrol show job` on that id describes the ARRAY rather than the one
# task, so the sed did not yield a single usable path. That task then exited 1
# before doing any work, which is invisible until an afterok dependent sits at
# DependencyNeverSatisfied. Every other task in the array was unaffected,
# which is what made it look like a data problem.
#
# SLURM_SUBMIT_DIR is the belt to that braces: Slurm sets it to the directory
# the job was submitted from, and every documented invocation here submits
# from the repo root.
_rel="datasets_preprocess/preprocess_scannetpp.sh"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _self="$(scontrol show job "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" 2>/dev/null \
             | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)"
    if [ -n "$_self" ] && [ -f "$_self" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$_self")" && pwd)"
    elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/$_rel" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$SLURM_SUBMIT_DIR/$_rel")" && pwd)"
    elif [ -n "${METRICS_REPO:-}" ] && [ -f "$METRICS_REPO/$_rel" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$METRICS_REPO/$_rel")" && pwd)"
    else
        echo "could not resolve this script's path: scontrol gave" \
             "'${_self:-<empty>}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}'." \
             "Set METRICS_REPO to the repo root, submit from it, or run this" \
             "with bash instead of sbatch" >&2
        exit 1
    fi
    unset _self
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
unset _rel
source "$SCRIPT_DIR/../datasets_download/env.sh"

OUT="$METRICS_PROCESSED_ROOT/processed_scannetpp"
PAIRS="${SCANNETPP_PAIRS:-$SCANNETPP_DIR/scannetpp_pairs}"

if [ ! -f "$PAIRS/scene_list.json" ]; then
    echo "no scene_list.json under $PAIRS -- see the scannetpp_pairs note in" \
         "the header of this script" >&2
    exit 1
fi

# keep tqdm from flooding the log (a bar per scene plus one per camera);
# honored by tqdm >= 4.66, harmless otherwise
export TQDM_MININTERVAL=60

# OpenCV's own thread pool fights the Slurm allocation: it sizes itself from
# the node's core count, not the cgroup, so on a shared campus node it spawns
# dozens of threads for the undistort remaps and thrashes. Embree gets the
# cores instead -- ray casting is where the time actually goes.
export OPENCV_FOR_THREADS_NUM="${SLURM_CPUS_PER_TASK:-4}"

cd "$METRICS_REPO"

# In an array, NUM_SHARDS is fixed at 24 and deliberately NOT taken from
# SLURM_ARRAY_TASK_COUNT: that reports the size of THIS submission, so
# re-running two failed tasks with `--array=7,19` would set it to 2 while the
# ids stay 7 and 19. Since the split is `i % num_shards == shard`, 7 % 2 never
# equals 7 -- both tasks would report "0 scenes assigned" and exit 0, leaving
# the gap invisible until finalize refused to run. Keep this in step with the
# --array range in the header above.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SHARD="$SLURM_ARRAY_TASK_ID"
    NUM_SHARDS="${NUM_SHARDS:-24}"
else
    # Outside an array there is no shard, so do every scene in one process.
    # Keeping NUM_SHARDS=24 with shard 0 here would convert 1/24th of the
    # scenes and exit 0. An explicit SHARD/NUM_SHARDS still wins, for
    # re-driving one shard by hand.
    SHARD="${SHARD:-0}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
fi

"$METRICS_PY" datasets_preprocess/preprocess_scannetpp.py \
    --scannetpp_dir "$SCANNETPP_DIR" \
    --precomputed_pairs "$PAIRS" \
    --output_dir "$OUT" \
    --num-shards "$NUM_SHARDS" \
    --shard "$SHARD"
