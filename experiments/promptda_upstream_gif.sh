#!/bin/bash
#SBATCH --job-name=promptda_upstream_gif
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/%x.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/%x.out

# =============================================================================
# promptda_upstream_gif.sh -- run UPSTREAM PromptDA's own inference script,
# unmodified, out of a clean checkout, on the sequences eval_all.sh visualizes.
# Predictions to a local dir, then to GIFs.
#
#   ./experiments/promptda_upstream_gif.sh [weights-dir] [stage ...]
#
# Diagnostic for the frame-to-frame flicker in the promptda arm: it separates
# "our vendored copy / our wiring does this" from "this is what PromptDA does
# on these frames".
#
# Three steps per stage:
#
#   1. DUMP (ours -- tests/dump_promptda_frames.py). The only custom code in
#      the chain, and it touches no model: our loaders produce the frames and
#      the prompt, written out as <frames>/<stage>/{rgb/%06d.jpg,
#      depth/%06d.png} -- the layout upstream's scripts glob.
#
#      PROMPT_MODE picks which prompt. Default 'arkit' on HAMMER/ScanNet: dense
#      GT downsampled to 192x256, which is what PromptDA was TRAINED on
#      ("exactly the depth resolution of iPhone ARKit Depth", arXiv 2412.14015
#      §3.3) -- ~49k filled measurements in a fixed layout every frame.
#      PROMPT_MODE=patch-infill instead reproduces our promptda arm: ~52
#      surviving 14x14 blocks at sim_mask_ratio 0.95, Voronoi-filled to dense
#      and REDRAWN INDEPENDENTLY PER FRAME. SPOT is always patch-infill (real
#      sensor, no dense GT).
#
#      Running both modes and diffing the GIFs is what separates "PromptDA is
#      shaky" from "our prompt is shaky". Note the arkit prompt hands PromptDA
#      far more information than our arm gets, so it is a diagnostic, NOT a
#      like-for-like baseline for an eval_all table.
#
#   2. INFER (theirs, unmodified). promptda/scripts/infer_stray_scan.py from
#      $UPSTREAM: their load_image (max-size 1008, INTER_AREA), their
#      load_depth, their PromptDA.predict, their save_depth. Writes
#      <results>/<stage>/%06d.png (uint16 mm) + %06d_depth.jpg (their
#      colormap, normalized PER FRAME).
#
#   3. VIDEO + GIF (theirs, then a naming-agnostic GIF writer). Their
#      promptda/scripts/generate_video.py process_stray_scan builds the
#      %06d_smooth.jpg panels -- [rgb | prediction | prompt], colour range
#      TEMPORALLY SMOOTHED across frames. Two GIFs come out of each stage, and
#      the pair is the point:
#         upstream_perframe.gif -- from their _depth.jpg, per-frame colour
#                                  range. Shows flicker if it is there.
#         upstream_smooth.gif   -- from their _smooth.jpg, smoothed range.
#                                  Their own video pipeline, which smooths the
#                                  range on purpose.
#      A wobble present in the first and absent in the second is a colour-range
#      artefact, not the depth moving.
#
# NOTHING upstream is edited, and nothing is copied into the repo -- $UPSTREAM
# is run in place. Frame selection is held identical to eval_all.sh (same
# checkpoint-derived val config, clip 0, --num-views 32, --sparse-seed 0, same
# SPOT window/stride/rotation/crop), so these GIFs line up frame-for-frame with
# the promptda_clip0 series an eval_all run already wrote.
#
# The checkpoint argument is a val-config carrier only: no weights are read
# from it and no StreamVGGT is built.
#
# Stages (default: all three): hammer | scannet | spot (static + dynamic)
#
# Env:
#   UPSTREAM         clean upstream checkout (default ~/PromptDA_upstream)
#   NAME             artifact dir under viz/ (default promptda_upstream)
#   PROMPT_MODE      arkit | patch-infill (default arkit on hammer/scannet).
#                    Set NAME too when switching, or the second run overwrites
#                    the first's GIFs.
#   MAX_SIZE         their loader's long-side cap (default 1008, their own
#                    default; our frames are 518x392 so nothing is resized --
#                    lower it only to test their resize path)
#   SMOOTH_INTERVAL  key-frame spacing for their range smoothing (default 8;
#                    their own default of 60 exceeds a 32-frame clip, collapsing
#                    to two key frames -- first and last -- i.e. maximum
#                    smoothing. Set 1 to disable smoothing entirely.)
#   FPS              GIF frame rate (default 10, matching eval_all)
#
# Needs a GPU for step 2. Run on a GPU node or sbatch this file.
# =============================================================================

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO/src/visualize_depth.py" ]; then
    REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ ! -f "${REPO:-/nonexistent}/src/visualize_depth.py" ]; then
    echo "cannot locate the repo; pass REPO=/path/to/checkout" >&2
    exit 1
fi

UPSTREAM="${UPSTREAM:-/oscar/home/jdosch/PromptDA_upstream}"
NAME="${NAME:-promptda_upstream}"
FPS="${FPS:-10}"
MAX_SIZE="${MAX_SIZE:-1008}"
SMOOTH_INTERVAL="${SMOOTH_INTERVAL:-8}"
PROMPT_MODE="${PROMPT_MODE:-arkit}"
WEIGHTS_DEFAULT="$REPO/checkpoints/hs_conf_off_sweep/5338c5bfb9be9414"

HAMMER_ROOT="${HAMMER_ROOT:-/oscar/scratch/jdosch/data/processed_hammer}"
SCANNET_ROOT="${SCANNET_ROOT:-/gpfs/data/jtompki1/cli277/metric/processed_scannet}"
SPOT_SEQ="${SPOT_SEQ:-/oscar/data/jtompki1/cli277/new_spot_data/0}"
NUM_VIEWS="${NUM_VIEWS:-32}"
SPOT_STATIC="${SPOT_STATIC:-0}"
SPOT_DYNAMIC="${SPOT_DYNAMIC:-998}"

WEIGHTS="${1:-$WEIGHTS_DEFAULT}"; [ $# -gt 0 ] && shift
WEIGHTS="$(cd "$(dirname "$WEIGHTS")" && pwd)/$(basename "$WEIGHTS")"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(hammer scannet spot)
OUT_ROOT="${OUT_ROOT:-$REPO/viz/$NAME}"
case "$OUT_ROOT" in /*) ;; *) OUT_ROOT="$PWD/$OUT_ROOT" ;; esac
FRAMES="$OUT_ROOT/frames"
RESULTS="$OUT_ROOT/results"

if [ ! -d "$UPSTREAM/promptda/scripts" ]; then
    echo "no upstream checkout at $UPSTREAM; clone it with" >&2
    echo "  git clone -b inference-script git@github.com:fmz/PromptDA.git $UPSTREAM" >&2
    exit 1
fi

export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compute nodes have no route out; the PromptDA weights are already in the hub
# cache, and offline mode makes a cache MISS fail loudly instead of hanging on
# a connection attempt
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$REPO/logs" "$FRAMES" "$RESULTS"

echo "=============================================================="
echo "upstream: $UPSTREAM ($(git -C "$UPSTREAM" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "weights:  $WEIGHTS  (val-config carrier only)"
echo "stages:   ${STAGES[*]}"
echo "out:      $OUT_ROOT"
echo "=============================================================="

# One stage: dump -> their inference -> their video panels -> GIFs.
# $1 = tag (also the scene dir name), rest = extra flags for the dump step.
run_stage () {
    local tag="$1"; shift
    local in_dir="$FRAMES/$tag" out_dir="$RESULTS/$tag"

    echo ""
    echo "---------- $tag: dump frames (ours) ----------"
    python "$REPO/tests/dump_promptda_frames.py" \
        --weights "$WEIGHTS" --checkpoint best \
        --num-views "$NUM_VIEWS" --out-dir "$in_dir" "$@"

    echo ""
    echo "---------- $tag: upstream inference (theirs, unmodified) ----------"
    # run from the checkout with it on PYTHONPATH so `promptda` resolves to
    # upstream's package and never to third_party/promptda
    (cd "$UPSTREAM" && PYTHONPATH="$UPSTREAM" python -m promptda.scripts.infer_stray_scan \
        --input-path "$in_dir" --output-path "$out_dir" --max-size "$MAX_SIZE")

    echo ""
    echo "---------- $tag: upstream video panels (theirs, unmodified) ----------"
    (cd "$UPSTREAM" && PYTHONPATH="$UPSTREAM" python -m promptda.scripts.generate_video \
        process_stray_scan --input-path "$in_dir" --result-path "$out_dir" \
        --smooth-interval "$SMOOTH_INTERVAL")

    echo ""
    echo "---------- $tag: GIFs ----------"
    python "$REPO/tests/frames_to_gif.py" --fps "$FPS" \
        --glob "$out_dir/*_depth.jpg" --out "$OUT_ROOT/${tag}_upstream_perframe.gif"
    python "$REPO/tests/frames_to_gif.py" --fps "$FPS" --max-width 1400 \
        --glob "$out_dir/*_smooth.jpg" --out "$OUT_ROOT/${tag}_upstream_smooth.gif"
}

for stage in "${STAGES[@]}"; do
    case "$stage" in
    hammer|scannet)
        root="$HAMMER_ROOT"; [ "$stage" = scannet ] && root="$SCANNET_ROOT"
        if [ ! -d "$root" ]; then
            echo "SKIP $stage: $root not found" >&2
            continue
        fi
        run_stage "$stage" --stage "$stage" --data-root "$root" \
            --prompt-mode "$PROMPT_MODE"
        ;;
    spot)
        if [ ! -d "$SPOT_SEQ" ]; then
            echo "SKIP spot: $SPOT_SEQ not found" >&2
            continue
        fi
        # geometry matches eval_all.sh's spot_pair exactly
        run_stage "spot_static_$SPOT_STATIC" --stage spot --seq-dir "$SPOT_SEQ" \
            --start "$SPOT_STATIC" --stride 2 --rotate cw --landscape-crop \
            --crop-anchor top
        run_stage "spot_dynamic_$SPOT_DYNAMIC" --stage spot --seq-dir "$SPOT_SEQ" \
            --start "$SPOT_DYNAMIC" --stride 2 --rotate cw --landscape-crop \
            --crop-anchor top
        ;;
    *) echo "unknown stage: $stage (want hammer|scannet|spot)" >&2; exit 1 ;;
    esac
done

echo ""
echo "=============================================================="
echo "GIFs (upstream PromptDA, unmodified):"
find "$OUT_ROOT" -maxdepth 1 -name '*.gif' | sort
echo ""
echo "frames in:  $FRAMES"
echo "raw depth:  $RESULTS  (%06d.png, uint16 millimetres)"
