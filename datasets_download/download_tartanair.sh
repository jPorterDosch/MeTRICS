#!/bin/bash
#SBATCH --job-name=tartanair_download
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/tartanair_download_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long

# TartanAir (https://theairlab.org/tartanair-dataset/)
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/download_tartanair.sh   # batch (needs ./logs to exist)
#   bash   datasets_download/download_tartanair.sh   # login node
#
# download_tartanair.py and download_training_zipfiles.txt are taken from the
# official tartanair_tools repo:
#   https://github.com/castacks/tartanair_tools
# (download_training.py renamed to download_tartanair.py to match this dir's
# naming; see the header of that file for the one behavioural change.)
#
# Downloads exactly the modalities preprocess_tartanair.py consumes, for both
# difficulties: image_left, depth_left, flow_flow and flow_mask. That script
# asserts
#     len(image_left) == len(depth_left) == len(flow)//2 + 1 == len(pose_left)
# and counts BOTH the flow and mask files in that flow//2, so dropping
# flow_mask (--only-flow) would fail the assert. It is only 2.6 GB of the
# total. pose_left.txt ships inside the image_left zips.
#
# Size: 144 zips, ~889 GB. The zips are NOT extracted, so this costs 144 inodes
# rather than the millions the extracted tree would.
#
# Requires boto3 + colorama in $METRICS_PY, and outbound HTTPS to
# airlab-cloud.andrew.cmu.edu:8080. Resumable, so re-run after the 48h
# walltime; expect more than one submission.

# -e so a failure here stops the script instead of falling through to the
# steps after it. NOT -o pipefail: a `yes`/`printf` feeding a prompt exits 141
# on SIGPIPE once the reader is done, and pipefail would turn that into a
# spurious failure after a successful run.
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
_rel="datasets_download/download_tartanair.sh"
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

mkdir -p "$TARTANAIR_DIR"

# the downloader opens download_training_zipfiles.txt by relative path,
# so it has to run from its own directory
cd "$SCRIPT_DIR"

# NO --unzip: the 144 zips are read in place by the preprocessing (members are
# individually deflated, so per-frame random access is cheap). Unzipping would
# turn 144 inodes into millions and blow the per-user inode quota outright.
# Transient network failures must not throw away hours of work. The first
# real run died after 1h54m on "Read timeout on endpoint URL" with 134 of 144
# archives down, and boto3's own retries do not cover a stalled body read.
# Each attempt resumes (already-complete files are skipped), so retrying the
# whole command is cheap and idempotent. If every attempt fails the verify
# step below still runs and reports exactly which files are missing.
ATTEMPTS="${ATTEMPTS:-5}"
RETRY_WAIT="${RETRY_WAIT:-120}"

for attempt in $(seq 1 "$ATTEMPTS"); do
    if "$METRICS_PY" "$SCRIPT_DIR/download_tartanair.py" \
        --output-dir "$TARTANAIR_DIR" \
        --rgb --depth --flow \
        --only-left
    then
        break
    fi
    if [ "$attempt" -lt "$ATTEMPTS" ]; then
        echo "download_tartanair: attempt $attempt/$ATTEMPTS failed;" \
             "resuming in ${RETRY_WAIT}s" >&2
        sleep "$RETRY_WAIT"
    else
        echo "download_tartanair: still failing after $ATTEMPTS attempts;" \
             "verify_tartanair.py will list what is missing" >&2
    fi
done

# belt and braces on top of the downloader's own .part+rename: check size and
# central directory before anything downstream trusts these archives.
# Exits non-zero on any gap.
"$METRICS_PY" "$SCRIPT_DIR/verify_tartanair.py" "$TARTANAIR_DIR"
