#!/bin/bash
#SBATCH --job-name=scannet_extract
#SBATCH --array=0-31              # 32 shards; scenes assigned round-robin
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
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

set -eu

# Resolving this script's own directory has to survive both invocation paths.
# Under sbatch, Slurm COPIES the script to /var/spool/.../slurm_script and runs
# it from there, so BASH_SOURCE points at the spool dir and every path built
# from it is wrong -- that is why sourcing env.sh failed with "No such file or
# directory". `scontrol show job` still knows the path the job was submitted
# with, so ask Slurm inside a job and fall back to BASH_SOURCE outside one.
# (sbatch --test-only does NOT run the body, so it cannot catch a bug here.)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _self="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
             | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')"
    if [ -z "$_self" ] || [ ! -f "$_self" ]; then
        echo "could not resolve this script's path from scontrol; set" \
             "METRICS_REPO or run it with bash instead of sbatch" >&2
        exit 1
    fi
    SCRIPT_DIR="$(cd "$(dirname "$_self")" && pwd)"
    unset _self
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$SCRIPT_DIR/env.sh"

cd "$METRICS_REPO"

# NUM_SHARDS is fixed at 32 and deliberately NOT taken from
# SLURM_ARRAY_TASK_COUNT: that reports the size of THIS submission, so
# re-running two failed tasks with `--array=7,19` would set it to 2 while the
# ids stay 7 and 19. Since the split is `i % num_shards == shard`, 7 % 2 never
# equals 7 -- both tasks would report "0 scenes assigned" and exit 0, leaving
# the gap invisible until preprocessing found no frames.zip. Keep this in step
# with the --array range in the header above.
NUM_SHARDS="${NUM_SHARDS:-32}"

"$METRICS_PY" datasets_preprocess/extract_scannet_sens.py \
    --raw-root "$SCANNET_DIR" \
    --num-shards "$NUM_SHARDS" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}"
