#!/bin/bash
#SBATCH --job-name=arkit_preprocess
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/arkit_preprocess_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long

# ARKitScenes raw assets -> the two processed trees the loaders read.
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_preprocess/preprocess_arkitscenes.sh   # queued
#   bash   datasets_preprocess/preprocess_arkitscenes.sh   # login node
#
# ONE download feeds TWO variants that partition the scenes without overlap
# (download_arkit_scenes.sh fetches the union of their assets in a single
# pass). Select with ARKIT_VARIANT=lowres|highres|both, default both:
#
#   lowres   preprocess_arkitscenes.py  (vga_wide + lowres_depth LiDAR)
#            -> processed_arkitscenes/<Training|Test>/
#            THEN generate_set_arkitscenes.py, because ARKitScenes_Multi reads
#            per-scene new_scene_metadata.npz, which only generate_set writes.
#            Needs the DUSt3R precomputed pairs: preprocess reads
#            <pairs>/<split>/scene_list.json to decide which scenes to touch,
#            so this stage cannot run before that zip is unpacked.
#
#   highres  preprocess_arkitscenes_highres.py (vga_wide + highres_depth laser
#            GT, the ~2257 "upsampling" scenes)
#            -> processed_arkitscenes_highres/<Training|Validation>/
#            NO generate_set: ARKitScenesHighRes_Multi reads the raw per-scene
#            scene_metadata.npz the preprocess itself writes (there are no
#            pairs / image_collection for this variant). Needs no pairs zip.
#
# Note the split names differ: lowres is Training/Test, highres is
# Training/Validation, and each loader maps Split.TEST onto its own. That is
# why the generate_set call below passes "Training Test" and not the highres
# names -- pointing it at the highres tree would find no scene_metadata.npz
# under a "Test" directory and silently do nothing.
#
# RESUMABILITY DIFFERS, which decides how a timeout is handled:
#   * highres SKIPS a scene whose scene_metadata.npz already exists
#     (preprocess_arkitscenes_highres.py:171), so re-running continues.
#   * lowres has NO such check -- it reprocesses every scene from scratch on
#     every run. A wall-clock kill costs the whole pass, which is why this
#     asks for the full 6-day `long` QOS ceiling rather than a nominal 48 h.
# Neither script shards, so this is one process either way; if 6 days is not
# enough for lowres, the fix is sharding preprocess_arkitscenes.py, not a
# longer wall (6 days is the QOS maximum -- see `sacctmgr show qos long`).

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
# DependencyNeverSatisfied.
#
# SLURM_SUBMIT_DIR is the belt to that braces: Slurm sets it to the directory
# the job was submitted from, and every documented invocation here submits
# from the repo root.
_rel="datasets_preprocess/preprocess_arkitscenes.sh"
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

ARKIT_VARIANT="${ARKIT_VARIANT:-both}"
case "$ARKIT_VARIANT" in
    lowres|highres|both) ;;
    *) echo "ARKIT_VARIANT must be lowres, highres or both (got '$ARKIT_VARIANT')" >&2
       exit 1 ;;
esac

# The downloader appends <dataset>/<split> to its --download_dir, so the raw
# tree lives one level under ARKIT_DIR. Overridable for a staged copy that
# landed elsewhere.
RAW="${ARKIT_RAW_DIR:-$ARKIT_DIR/raw}"
PAIRS="${ARKIT_PAIRS_DIR:-$ARKIT_DIR/arkitscenes_pairs}"
OUT_LOW="$METRICS_PROCESSED_ROOT/processed_arkitscenes"
OUT_HIGH="$METRICS_PROCESSED_ROOT/processed_arkitscenes_highres"

# keep tqdm from flooding the log (one bar per split, thousands of scenes);
# honored by tqdm >= 4.66, harmless otherwise
export TQDM_MININTERVAL=60

# Fail here rather than thousands of scenes in. The raw tree is the ~5 TB
# download; the pairs are a separate ~100 MB zip from NAVER LABS and only
# lowres needs them.
[ -d "$RAW" ] || { echo "raw ARKitScenes tree not found: $RAW" >&2
                   echo "  run datasets_download/download_arkit_scenes.sh first," \
                        "or set ARKIT_RAW_DIR" >&2; exit 1; }
if [ "$ARKIT_VARIANT" != "highres" ]; then
    [ -d "$PAIRS" ] || { echo "precomputed pairs not found: $PAIRS" >&2
                         echo "  wget+unzip arkitscenes_pairs.zip into \$ARKIT_DIR," \
                              "or set ARKIT_PAIRS_DIR (see README)" >&2; exit 1; }
fi

cd "$METRICS_REPO"

if [ "$ARKIT_VARIANT" = "lowres" ] || [ "$ARKIT_VARIANT" = "both" ]; then
    echo "=== lowres: preprocess -> $OUT_LOW ==="
    "$METRICS_PY" datasets_preprocess/preprocess_arkitscenes.py \
        --arkitscenes_dir "$RAW" \
        --precomputed_pairs "$PAIRS" \
        --output_dir "$OUT_LOW"

    # ARKitScenes_Multi reads new_scene_metadata.npz, which ONLY this writes.
    # max_interval 5.0 is the DUSt3R recipe's value, in seconds of capture
    # time; num_workers matches --cpus-per-task above.
    echo "=== lowres: generate_set -> new_scene_metadata.npz ==="
    "$METRICS_PY" datasets_preprocess/generate_set_arkitscenes.py \
        --root "$OUT_LOW" \
        --splits Training Test \
        --max_interval 5.0 \
        --num_workers "${SLURM_CPUS_PER_TASK:-8}"
fi

if [ "$ARKIT_VARIANT" = "highres" ] || [ "$ARKIT_VARIANT" = "both" ]; then
    echo "=== highres: preprocess -> $OUT_HIGH (no generate_set) ==="
    "$METRICS_PY" datasets_preprocess/preprocess_arkitscenes_highres.py \
        --arkitscenes_dir "$RAW" \
        --output_dir "$OUT_HIGH"
fi

echo "=== done (variant: $ARKIT_VARIANT) ==="
