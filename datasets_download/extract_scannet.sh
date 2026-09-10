#!/bin/bash
#SBATCH --job-name=scannet_extract
#SBATCH --array=0-31              # 32 shards; scenes assigned round-robin
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/scannet_extract_%A_%a.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# Batch-extract raw ScanNet .sens -> per-frame color/depth/pose/intrinsic.
# Stage 2 of 4 (download_scannet.sh, then this, then preprocess_scannet.sh).
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/extract_scannet.sh   # 32 parallel shards
#   bash   datasets_download/extract_scannet.sh   # one process, all scenes
# Outside a job array SLURM_ARRAY_TASK_* are unset, so the fallback below runs
# the whole set as a single shard rather than dying on an unbound variable.
#
# Resumable: already-extracted scenes are skipped, so re-running continues.
#
# Sizing, measured on scene0300_00 (1747 frames, 0.51 GiB .sens):
#   time  1.5 h per shard (47.7 core-hours over 1613 scenes / 1072.6 GiB, at
#         22 frames/s -- the depth PNG encoder is pure-Python pypng)
#   size  0.46x the .sens, so ~492 GiB of frames.zip written back INTO
#         $SCANNET_DIR alongside the inputs
#   mem   SensorData.load() holds the whole .sens in RAM as compressed frame
#         bytes -- peak RSS measured at 1.13x the file. The largest scene,
#         scans_test/scene0757_00, is 4.74 GiB, so peak is ~5.4 GiB; --mem=6G
#         left too little margin, and a cgroup OOM is a SIGKILL that takes the
#         whole shard's remaining scenes with it, not a caught exception.

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
_rel="datasets_download/extract_scannet.sh"
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
source "$SCRIPT_DIR/env.sh"

cd "$METRICS_REPO"

# In an array, NUM_SHARDS is fixed at 32 and deliberately NOT taken from
# SLURM_ARRAY_TASK_COUNT: that reports the size of THIS submission, so
# re-running two failed tasks with `--array=7,19` would set it to 2 while the
# ids stay 7 and 19. Since the split is `i % num_shards == shard`, 7 % 2 never
# equals 7 -- both tasks would report "0 scenes assigned" and exit 0, leaving
# the gap invisible until preprocessing found no frames.zip. Keep this in step
# with the --array range in the header above.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SHARD="$SLURM_ARRAY_TASK_ID"
    NUM_SHARDS="${NUM_SHARDS:-32}"
else
    # Outside an array there is no shard, so do every scene in one process.
    # Keeping NUM_SHARDS=32 with shard 0 here would extract 1/32nd of the
    # scenes and exit 0 -- the `bash` invocation the header offers would look
    # like it had succeeded, and the gap would only surface as missing
    # frames.zip during preprocessing. An explicit SHARD/NUM_SHARDS in the
    # environment still wins, for re-driving one shard by hand.
    SHARD="${SHARD:-0}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
fi

"$METRICS_PY" datasets_preprocess/extract_scannet_sens.py \
    --raw-root "$SCANNET_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard "$SHARD"
