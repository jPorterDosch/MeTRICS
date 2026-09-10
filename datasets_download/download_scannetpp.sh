#!/bin/bash
#SBATCH --job-name=scannetpp_download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/scannetpp_download_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long

# ScanNet++ (https://kaldir.vc.in.tum.de/scannetpp/)
#
# Needs a personal download token: register, sign the terms, and the site mails
# you a bundle. Put the token in datasets_download/.scannetpp_token (gitignored,
# mode 600) or export SCANNETPP_TOKEN. It is a per-user credential tied to a
# signed ToU -- keep it out of the repo and out of job scripts.
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/download_scannetpp.sh   # batch (needs ./logs)
#   bash   datasets_download/download_scannetpp.sh   # login node
# Do NOT pass the token on the sbatch command line -- Slurm exposes job
# submission lines to other users via scontrol/squeue.
#
# Downloads ~112 GiB: the 228 scenes preprocess_scannetpp.py actually reads
# (DUSt3R's scannetpp_pairs/scene_list.json, less 280b83fcf3, which was dropped
# from the release and has no mesh) and only the 7 assets that script opens.
# The stock config's defaults would pull ~1.5 TB, almost none of it read.
#
# The four DSLR/iPhone directory archives are kept ZIPPED -- preprocessing
# reads members in place -- so a scene costs 11 inodes (7 files + 4 dirs)
# rather than the ~1,650 loose files inside: ~2.5k inodes for the whole
# dataset instead of ~375k. See keep_zipped in download_scannetpp.yml.
#
# Observed throughput on a login node: ~17 MB/s, so ~2h for the 112 GiB.
#
# Resumable: a complete archive is skipped, a truncated one is deleted and
# re-fetched. Transfers land on a .part file and are renamed only on success,
# so a killed job never leaves a half-file under a final name.
# NOTE: -o pipefail is deliberately NOT set. The downloader is fed on stdin
# below, and any writer that outlives the reader (`yes`) exits 141 on SIGPIPE;
# under pipefail that becomes the pipeline's status, so the script would abort
# with a failure AFTER a completely successful download -- skipping the verify
# step and reporting the Slurm job as FAILED. Without pipefail the pipeline
# reports the downloader's own status, which is what we actually care about.
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
_rel="datasets_download/download_scannetpp.sh"
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

# token: environment first, then the gitignored file
if [[ -z "${SCANNETPP_TOKEN:-}" && -f "$SCRIPT_DIR/.scannetpp_token" ]]; then
    SCANNETPP_TOKEN="$(tr -d '[:space:]' < "$SCRIPT_DIR/.scannetpp_token")"
fi
: "${SCANNETPP_TOKEN:?no token: write it to datasets_download/.scannetpp_token or export SCANNETPP_TOKEN}"

mkdir -p "$SCANNETPP_DIR"

# DUSt3R's precomputed pairs supply BOTH the scene list this config downloads
# and the per-scene image selection preprocess_scannetpp.py renders, so the
# dataset is unusable for training without them. 26 MB, no token needed, and
# idempotent. Fetched here rather than left as a manual README step because
# forgetting it surfaces hours later, as a preprocessing array whose every
# task dies on its first scene.
PAIRS_DIR="$SCANNETPP_DIR/scannetpp_pairs"
if [ ! -f "$PAIRS_DIR/scene_list.json" ]; then
    echo "fetching scannetpp_pairs (DUSt3R precomputed pairs)..."
    # .part + rename, the same contract as every other transfer here: a
    # killed job must never leave a truncated file under a final name.
    _pairs_zip="$SCANNETPP_DIR/scannetpp_pairs.zip.part"
    curl -fSL --retry 5 --retry-delay 10 -o "$_pairs_zip" \
        https://download.europe.naverlabs.com/ComputerVision/DUSt3R/scannetpp_pairs.zip
    # the archive's members are already prefixed scannetpp_pairs/, so this
    # lands as $SCANNETPP_DIR/scannetpp_pairs/
    unzip -q -o "$_pairs_zip" -d "$SCANNETPP_DIR"
    rm -f "$_pairs_zip"
    if [ ! -f "$PAIRS_DIR/scene_list.json" ]; then
        echo "scannetpp_pairs.zip did not yield scene_list.json" >&2
        exit 1
    fi
    echo "scannetpp_pairs ready: $(ls "$PAIRS_DIR" | wc -l) entries"
fi

# Render the committed config with this run's data_root and token filled in.
# The token stays out of the repo and off the process command line (which is
# world-readable via /proc and squeue); the rendered file is mode 600 and is
# removed on exit, including on failure.
RENDERED="$(mktemp "${TMPDIR:-/tmp}/scannetpp_download.XXXXXX.yml")"
chmod 600 "$RENDERED"
trap 'rm -f "$RENDERED"' EXIT

export SCANNETPP_DIR SCANNETPP_TOKEN
"$METRICS_PY" - "$SCRIPT_DIR/download_scannetpp.yml" "$RENDERED" <<'PYEOF'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
if str(cfg.get("root_url", "")).startswith("<"):
    sys.exit("download_scannetpp.yml has a placeholder root_url; copy the real "
             "line from the config in your ScanNet++ token e-mail.")
cfg["data_root"] = os.environ["SCANNETPP_DIR"]
cfg["token"] = os.environ["SCANNETPP_TOKEN"]
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
PYEOF

# The downloader asks an interactive y/n before starting (a disk-space warning
# quoting the stock 1.5 TB figure, which does not apply to this config). Under
# sbatch stdin is closed, so input() would raise EOFError and kill the job
# before a byte moved -- feed it. `yes y` is only safe because the token and
# data_root are already resolved above: those are the script's other two
# input() calls, and neither can fire now.
# A finite stream, not `yes`: three lines comfortably covers the one prompt
# that can still fire (and any the upstream script grows later), and printf
# exits 0 once its bytes are in the pipe buffer rather than dying on SIGPIPE.
# `|| true` so a failed download still reaches the verify below. The
# downloader aborts the whole run on the first failed asset -- exactly the
# case the comment on the verify step describes -- and under `set -e` that
# status would terminate this script first, skipping the one step that says
# what is actually missing. The verify exits non-zero on any gap, so the job
# still fails; it just fails with a list.
printf 'y\ny\ny\n' | "$METRICS_PY" "$SCRIPT_DIR/download_scannetpp.py" "$RENDERED" \
    || echo "download_scannetpp.py exited non-zero; running verify to report gaps" >&2

# The downloader treats a file that merely exists as complete, and aborts the
# whole run on the first failed asset, so a partial tree is the normal outcome
# of an interrupted job. Check every archive's central directory and every
# scene's asset set before anything downstream trusts this. Exits non-zero on
# any gap; re-run this script to fill them.
"$METRICS_PY" "$SCRIPT_DIR/verify_scannetpp.py" "$SCANNETPP_DIR"
