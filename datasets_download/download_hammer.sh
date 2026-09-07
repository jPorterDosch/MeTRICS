#!/bin/bash
#SBATCH --job-name=hammer_download
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hammer_download_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# HAMMER (https://github.com/Junggy/HAMMER-dataset)
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/download_hammer.sh   # batch (needs ./logs to exist)
#   bash   datasets_download/download_hammer.sh   # login node
#
# Downloads only the polarization (RGB) camera subset needed for CUT3R
# (~24 GB of the 170 GB official zip) via HTTP range requests.
# Single-stream, so this is network-bound rather than CPU-bound.
#
# Resumable: members already on disk at the expected size are skipped and every
# file is CRC32-checked, so re-running after a timeout continues safely.
#
# Requires outbound HTTPS to campar.in.tum.de. If the compute nodes are
# firewalled, run it on a login node with bash.

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
_rel="datasets_download/download_hammer.sh"
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

mkdir -p "$HAMMER_DIR"
"$METRICS_PY" "$SCRIPT_DIR/download_hammer.py" --out "$HAMMER_DIR"
