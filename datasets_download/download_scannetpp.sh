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

# token: environment first, then the gitignored file
if [[ -z "${SCANNETPP_TOKEN:-}" && -f "$SCRIPT_DIR/.scannetpp_token" ]]; then
    SCANNETPP_TOKEN="$(tr -d '[:space:]' < "$SCRIPT_DIR/.scannetpp_token")"
fi
: "${SCANNETPP_TOKEN:?no token: write it to datasets_download/.scannetpp_token or export SCANNETPP_TOKEN}"

mkdir -p "$SCANNETPP_DIR"

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
printf 'y\ny\ny\n' | "$METRICS_PY" "$SCRIPT_DIR/download_scannetpp.py" "$RENDERED"

# The downloader treats a file that merely exists as complete, and aborts the
# whole run on the first failed asset, so a partial tree is the normal outcome
# of an interrupted job. Check every archive's central directory and every
# scene's asset set before anything downstream trusts this. Exits non-zero on
# any gap; re-run this script to fill them.
"$METRICS_PY" "$SCRIPT_DIR/verify_scannetpp.py" "$SCANNETPP_DIR"
