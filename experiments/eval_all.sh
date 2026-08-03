#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/eval_all/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/eval_all/%j.out

# =============================================================================
# eval_all.sh -- run one checkpoint through every evaluation we have.
#
#   NAME=<experiment-name> ./experiments/eval_all.sh <weights-dir> [stage ...]
#
# NAME files everything under viz/<NAME>/ (default: viz/eval_<ckpt-hash>, which
# is unreadable after the fact). OUT_ROOT still overrides the whole path.
#
# Stages (default: all four, in this order):
#   ablation  tests/conditioning_ablation_eval.py --include-base
#             Does the model USE its sparse measurements? Scores conditioned /
#             ablated / base_conditioned / base_ablated on the run's OWN val
#             split and decomposes the gain. The base arms MUST tie exactly --
#             that is the zero-init no-op check, and it fails loudly if any
#             injection path is not actually zero-init.
#             Catches failure mode we previously missed -- zero-init was preventing
#             effective training/convergence.
#   hammer    In-domain base-vs-finetuned GLBs + heatmaps, pinned to the HAMMER
#             half of the run's val split (--dataset hammer). Pinning matters
#             on HAMMER+ScanNet runs: only one clip is exported, so an unpinned
#             mixture would file whichever dataset the sampler drew under
#             hammer/ and the paired table would compare across datasets.
#   scannet   The same, out-of-domain on ScanNet TEST. Answers "does any of
#             this transfer off the training distribution?"
#   spot      Real robot capture with REAL sensor sparsity (not simulated
#             patch masking), landscape-cropped, on BOTH a static and a
#             dynamic window -- the static/dynamic split matters: a static
#             window makes warp self-consistency nearly free and flatters the
#             model, so a result that only holds there is not a result.
#
# Every viz stage runs THREE arms into one out dir: base (pretrained
# StreamVGGT), finetuned (ours), and promptda (vendored Prompt Depth Anything,
# third_party/promptda; disable with PROMPTDA=0, override weights with
# PROMPTDA_CKPT). Each arm writes, per frame: predicted depth, |pred-gt|/gt
# accuracy, the model's own predicted CONFIDENCE (promptda has none), and warp
# self-consistency -- as PNG series, then as GIFs, including side-by-side
# compare_*.gif (compare_depth.gif, compare_conf.gif, ...) with one column per
# arm, all arms at once.
#
# Every stage writes a summary CSV; the run ends with ONE table per stage over
# all arms (ours last = the subject each baseline is measured against) via
# tests/compare_heatmap_summaries.py. The CSV is the
# artifact to trust -- the GIFs are for looking at, and a saturated panel is
# indistinguishable from "no difference" by eye. For the confidence series the
# number that matters is conf_err_corr (Spearman of confidence against relative
# error): more negative = the confidence actually tracks the error, near 0 = the
# map is decoration regardless of how good it looks.
#
# Needs a GPU (loss_of_one_batch runs under torch.cuda.amp.autocast). Run it on
# a GPU node, or sbatch this file.
#
# Examples:
#   ./experiments/eval_all.sh checkpoints/hammer_token_zeroinit/5978e1b00c3663be
#   NAME=zeroinit_v2 ./experiments/eval_all.sh checkpoints/<run> ablation scannet
#   NAME=stride_1_1 sbatch --job-name=stride_1_1 experiments/eval_all.sh checkpoints/<run>
#   OUT_ROOT=/abs/path/myrun ./experiments/eval_all.sh checkpoints/<run> spot
# =============================================================================

set -euo pipefail

# Resolve the repo so THIS checkout's code runs (main tree or any worktree, on
# any machine), overridable with REPO=... :
#   1. this script's own location -- correct for direct ./experiments/... runs;
#   2. under sbatch the script executes from a slurm SPOOL copy, so (1) lands
#      outside a repo -- fall back to the git toplevel of $PWD, which slurm
#      sets to the SUBMIT directory: sbatch from the checkout you mean.
#      (Deliberately not SLURM_SUBMIT_DIR -- it survives into nested
#      environments like OOD sessions and can point somewhere stale.)
# The src/visualize_depth.py probe is what detects case (2).
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO/src/visualize_depth.py" ]; then
    REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ ! -f "${REPO:-/nonexistent}/src/visualize_depth.py" ]; then
    echo "cannot locate the repo (script ran from a spool dir and the working" >&2
    echo "directory is not inside a checkout); pass REPO=/path/to/checkout" >&2
    exit 1
fi
PRETRAINED="${PRETRAINED:-$REPO/ckpt/checkpoints.pth}"
HAMMER_ROOT="${HAMMER_ROOT:-/oscar/scratch/jdosch/data/processed_hammer}"
SCANNET_ROOT="${SCANNET_ROOT:-/gpfs/data/jtompki1/cli277/metric/processed_scannet}"
SPOT_SEQ="${SPOT_SEQ:-/oscar/data/jtompki1/cli277/new_spot_data/0}"
CLIPS="${CLIPS:-32}"        # ablation clips
NUM_VIEWS="${NUM_VIEWS:-32}"  # frames per visualized clip
# TIMING=1 adds per-frame inference milliseconds (frame_ms) to every summary
# CSV. Measured with CUDA events, so it does not stall the loop it measures --
# but the numbers are only meaningful on an otherwise idle GPU, which a shared
# node is not. Off by default.
TIMING_ARGS=()
[ "${TIMING:-}" = 1 ] && TIMING_ARGS=(--timing)

# SPOT windows: start 0 is the most STATIC segment of seq 0 and start 998 the
# most dynamic (measured by frame-to-frame RGB change over all 1279 frames).
# Both are evaluated because the finetuning advantage has been seen to hold on
# one and vanish on the other -- reporting only the static one overstates it.
SPOT_STATIC="${SPOT_STATIC:-0}"
SPOT_DYNAMIC="${SPOT_DYNAMIC:-998}"

# PromptDA third arm: on by default (PROMPTDA=0 disables). The checkpoint is a
# local model.ckpt path or an HF repo id; the hub download needs network, so
# pre-download on a login node (see third_party/promptda/README.md).
PROMPTDA="${PROMPTDA:-1}"
PROMPTDA_CKPT="${PROMPTDA_CKPT:-depth-anything/prompt-depth-anything-vitl}"
PROMPTDA_ARGS=(--promptda --promptda-ckpt "$PROMPTDA_CKPT")

if [ $# -lt 1 ]; then
    sed -n '12,40p' "$0" >&2
    exit 1
fi
WEIGHTS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
RUN_ID=$(basename "$WEIGHTS")
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(ablation hammer scannet spot)
# NAME is the human-readable label this eval is filed under; the checkpoint
# basename is a hash, so without one the artifacts are unidentifiable later.
NAME="${NAME:-eval_$RUN_ID}"
OUT_ROOT="${OUT_ROOT:-$REPO/viz/$NAME}"
# resolve before the cd below -- everything after it runs from $REPO/src, so a
# relative OUT_ROOT would be created here and written to somewhere else.
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="$PWD/$OUT_ROOT" ;; esac

export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
# fragmentation-class OOMs on L40S show up as small-alloc failures; every
# GPU entry point in this repo sets this for parity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$REPO/logs/eval_all" "$OUT_ROOT"
cd "$REPO/src"

echo "=============================================================="
echo "name:    $NAME"
echo "run:     $RUN_ID"
echo "weights: $WEIGHTS"
echo "stages:  ${STAGES[*]}"
echo "out:     $OUT_ROOT"
echo "=============================================================="

# --checkpoint best explicitly: "auto" resolves ("final","best","last") with
# "final" aliased to checkpoint-last.pth, so auto silently gives LAST.
CKPT_ARGS=(--weights "$WEIGHTS" --checkpoint best)

# Colour ceilings. gterr and tcons diverge by an order of magnitude out of
# domain (accuracy >50% while consistency stays ~1%), so one shared scale
# renders one of them useless. These are for the PICTURES only -- the CSV
# means are computed pre-clip and are unaffected.
#
# --conf-vmax is the same idea for the predicted-confidence panels. The depth
# head is expp1 (conf = 1 + exp(x)), so 1.0 is the floor and there is no
# ceiling; these are starting points, not measured values. Check the
# conf_saturated_frac column of each summary CSV and retune if it is near 1
# (panels all bright) or near 0 with a low conf mean (panels all dark). Keep
# the value IDENTICAL across a base/finetuned pair -- viz_pair does, by
# construction, since both arms get the same array.
HAMMER_SCALES=(--rel-vmax 0.3  --tcons-vmax 0.05 --conf-vmax 10)  # base ~0.20 AbsRel in-domain
SCANNET_SCALES=(--rel-vmax 0.5 --tcons-vmax 0.03 --conf-vmax 10)  # OOD, tcons ~0.01
SPOT_SCALES=(--rel-vmax 1.0    --tcons-vmax 0.15 --conf-vmax 10)  # sensor-deviation, runs high

# arm tag order for the side-by-side GIFs and the summary table: baselines
# first, OUR arm last (compare_heatmap_summaries treats the last tag as the
# subject every other arm is measured against)
ARM_TAGS=(base_clip0 finetuned_clip0)
[ "$PROMPTDA" = 1 ] && ARM_TAGS=(base_clip0 promptda_clip0 finetuned_clip0)

heatmap_gifs () {  # $1 = heatmaps dir
    [ -d "$1" ] && python heatmaps_to_gif.py --hm-dir "$1" \
        --compare "${ARM_TAGS[@]}" --fps 10 || true
}

viz_pair () {  # $1 = out dir, $2 = scales array name, rest = extra flags
    local out="$1"; shift
    local -n scales=$1; shift
    python visualize_depth.py "${CKPT_ARGS[@]}" --base --pretrained "$PRETRAINED" \
        --num-clips 1 --num-views "$NUM_VIEWS" --heatmaps "${scales[@]}" \
        "${TIMING_ARGS[@]}" --out-dir "$out" "$@"
    python visualize_depth.py "${CKPT_ARGS[@]}" \
        --num-clips 1 --num-views "$NUM_VIEWS" --heatmaps "${scales[@]}" \
        "${TIMING_ARGS[@]}" --out-dir "$out" "$@"
    # same ${scales[@]} on purpose: identical colour ceilings are what make the
    # three arms' panels comparable
    [ "$PROMPTDA" = 1 ] && python visualize_depth.py "${CKPT_ARGS[@]}" \
        "${PROMPTDA_ARGS[@]}" \
        --num-clips 1 --num-views "$NUM_VIEWS" --heatmaps "${scales[@]}" \
        "${TIMING_ARGS[@]}" --out-dir "$out" "$@"
    heatmap_gifs "$out/heatmaps"
}

spot_pair () {  # $1 = out dir, $2 = --start
    local out="$1" start="$2"
    # --base MUST run first: it writes pose_cache.npz, and every later run
    # (finetuned AND promptda -- PromptDA predicts no cameras at all)
    # unprojects with the CACHED track so the A/B stays attributable to depth.
    # All geometry flags are one variable -- they are hashed into the cache's
    # fingerprint, so a mismatch between the runs hard-fails by design.
    local geom=(--rotate cw --landscape-crop --crop-anchor top
                --start "$start" --stride 2 --num-views "$NUM_VIEWS"
                --seq-dir "$SPOT_SEQ" --heatmaps "${SPOT_SCALES[@]}"
                "${TIMING_ARGS[@]}" --out-dir "$out")
    python visualize_spot.py "${CKPT_ARGS[@]}" --base --pretrained "$PRETRAINED" "${geom[@]}"
    python visualize_spot.py "${CKPT_ARGS[@]}" "${geom[@]}"
    [ "$PROMPTDA" = 1 ] && python visualize_spot.py "${CKPT_ARGS[@]}" \
        "${PROMPTDA_ARGS[@]}" "${geom[@]}"
    heatmap_gifs "$out/heatmaps"
}

for stage in "${STAGES[@]}"; do
    echo ""
    echo "---------- stage: $stage ----------"
    case "$stage" in
    ablation)
        python "$REPO/tests/conditioning_ablation_eval.py" \
            --weights "$WEIGHTS" --checkpoint best \
            --clips "$CLIPS" --include-base --pretrained "$PRETRAINED" \
            | tee "$OUT_ROOT/ablation.txt"
        ;;
    hammer)
        if [ ! -d "$HAMMER_ROOT" ]; then
            echo "SKIP hammer: $HAMMER_ROOT not found" >&2
        else
            # --dataset hammer is NOT optional on a mixed-val run: without it
            # the sampler draws from the whole val mixture, so the single clip
            # filed under hammer/ (and scaled with HAMMER_SCALES) could be a
            # ScanNet clip. On a HAMMER-only run it selects entry 0 -- a no-op.
            viz_pair "$OUT_ROOT/hammer" HAMMER_SCALES \
                --dataset hammer --data-root "$HAMMER_ROOT"
        fi
        ;;
    scannet)
        if [ ! -d "$SCANNET_ROOT" ]; then
            echo "SKIP scannet: $SCANNET_ROOT not found" >&2
        else
            viz_pair "$OUT_ROOT/scannet" SCANNET_SCALES \
                --dataset scannet --data-root "$SCANNET_ROOT"
        fi
        ;;
    spot)
        if [ ! -d "$SPOT_SEQ" ]; then
            echo "SKIP spot: $SPOT_SEQ not found" >&2
        else
            spot_pair "$OUT_ROOT/spot_static_$SPOT_STATIC"   "$SPOT_STATIC"
            spot_pair "$OUT_ROOT/spot_dynamic_$SPOT_DYNAMIC" "$SPOT_DYNAMIC"
        fi
        ;;
    *) echo "unknown stage: $stage (want ablation|hammer|scannet|spot)" >&2; exit 1 ;;
    esac
done

echo ""
echo "=============================================================="
echo "PAIRED SUMMARY (${ARM_TAGS[*]}) -- $RUN_ID"
echo "=============================================================="
mapfile -t HMDIRS < <(find "$OUT_ROOT" -type d -name heatmaps | sort)
if [ ${#HMDIRS[@]} -gt 0 ]; then
    python "$REPO/tests/compare_heatmap_summaries.py" "${HMDIRS[@]}" \
        --tags "${ARM_TAGS[@]}" | tee "$OUT_ROOT/summary.txt"
else
    echo "(no heatmap dirs -- ablation-only run; see $OUT_ROOT/ablation.txt)"
fi
echo ""
echo "artifacts: $OUT_ROOT"
