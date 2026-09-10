#!/bin/bash
#SBATCH --job-name=tartanair_preprocess
#SBATCH --array=0-23              # 24 shards over 36 (env, difficulty) units
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tartanair_preprocess_%A_%a.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# TartanAir raw zips -> one processed frames.zip per trajectory
# (rgb/depth/flow/mask copied verbatim + a per-frame cam npz). The raw 144
# archives are read in place and never unzipped.
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_preprocess/preprocess_tartanair.sh   # 24 parallel shards
#   bash   datasets_preprocess/preprocess_tartanair.sh   # one process, all envs
# Outside a job array SLURM_ARRAY_TASK_* are unset, so the fallback below runs
# the whole set as a single shard rather than dying on an unbound variable.
#
# There is no generate_set stage for TartanAir: src/dust3r/datasets/tartanair.py
# walks the output tree directly and reads no metadata npz, so this one script
# is the whole pipeline.
#
# Why an array at all: preprocess_tartanair.py converts trajectories in a
# single process, and the full set is 828 GiB in / 1404 GiB out. Measured at
# 70 MiB/s of output on one core (office/Hard/P003, 300 frames in 20 s), that
# is ~6.5 h in one process -- inside the campus QOS 24 h wall, but with no
# margin for a slow node and nothing to show for a timeout. One
# (env, difficulty) is one shard unit, which is the granularity discover()
# already enumerates at, so a task parses only its own archives' central
# directories. There are 36 such units and the array is 24 wide -- 12 tasks
# take two units and 12 take one -- so that the whole pipeline fits under the
# campus QOS MaxSubmitPU of 96 in a single submission. The heaviest task is
# neighborhood_Easy (76 GiB raw) plus one more, comfortably inside an hour.
#
# Resumable: a trajectory with a complete frames.zip is skipped (the writer
# renames .tmp -> final only on success, so a final-named archive is always
# complete), so re-running finishes what a timed-out task left. Tasks are
# uneven by design -- neighborhood_Easy is 76 GiB raw against
# carwelding_Hard's 6 GiB -- so expect the array to finish very unevenly.
#
# NOTE ON SIZE: the output archives are ZIP_STORED while the upstream ones are
# DEFLATE, so the processed tree is ~1.7x the raw download, not ~1.0x. Summing
# every member's uncompressed size across all 144 archives gives exactly
# 1404 GiB: flow 701, depth 351, image 264, mask 88 (float32 flow barely
# compresses; the masks are 82x). Check `lfs quota -u $USER /lustre/isaac24`
# has that much free BEFORE submitting -- the quota is per-uid and counts
# every path on the filesystem, not just this project dir. This is the
# largest single output in the pipeline; run it last.

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
_rel="datasets_preprocess/preprocess_tartanair.sh"
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

OUT="$METRICS_PROCESSED_ROOT/processed_tartanair"

# keep tqdm from flooding the log (one bar per trajectory, ~30 per task);
# honored by tqdm >= 4.66, harmless otherwise
export TQDM_MININTERVAL=60

cd "$METRICS_REPO"

# In an array, NUM_SHARDS is fixed at 24 and deliberately NOT taken from
# SLURM_ARRAY_TASK_COUNT: that reports the size
# of THIS submission, so re-running two failed tasks with `--array=7,19` would
# set it to 2 while the ids stay 7 and 19. Since the split is
# `i % num_shards == shard`, 7 % 2 never equals 7 -- both tasks would report
# "0 trajectories" and exit 0, leaving the gap invisible. Keep this in step
# with the --array range in the header above. It does NOT have to equal the
# 36 (env, difficulty) units -- units are handed out round-robin, so any width
# covers them all -- but every width must divide the work with no empty shard:
#   ls "$TARTANAIR_DIR"/*_image_left.zip | wc -l   # 36 units
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SHARD="$SLURM_ARRAY_TASK_ID"
    NUM_SHARDS="${NUM_SHARDS:-24}"
else
    # Outside an array there is no shard, so do the whole set in one process.
    # Keeping NUM_SHARDS=24 with shard 0 here would convert 1/24th of the data
    # and exit 0 -- the `bash` invocation the header offers would look like it
    # had succeeded. An explicit SHARD/NUM_SHARDS in the environment still
    # wins, for driving one unit by hand.
    SHARD="${SHARD:-0}"
    NUM_SHARDS="${NUM_SHARDS:-1}"
fi

"$METRICS_PY" datasets_preprocess/preprocess_tartanair.py \
    --tartanair_dir "$TARTANAIR_DIR" \
    --output_dir "$OUT" \
    --num-shards "$NUM_SHARDS" \
    --shard "$SHARD"
