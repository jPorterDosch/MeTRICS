#!/bin/bash
#SBATCH --job-name=scale_drift
#SBATCH --account=isaac-utk0256
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/scale_drift_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/scale_drift_%j.out
# =============================================================================
# Scale drift under depth dropout (src/scale_drift.py): how long the KV cache
# carries absolute scale once the sparse depth stops. Three curves on the 100
# ScanNet TAE clips (192 frames each), KV window 20:
#
#   StreamVGGT          the pretrained backbone, RGB only
#   Ours (dropout @20)  the tube-mask run, 5% sparse depth on frames 0-19,
#                       none from frame 20 (no conditioned frame left in the
#                       cache from frame 40)
#   Ours (fed)          the same run fed on every frame (reference)
#
#   sbatch experiments/all_datasets_finetune/scale_drift.sh
#   RUN=<run dir> DROP=10 sbatch ...      # another run / dropout frame
#
# The other experiments (src/scale_drift.py docstring), ours only except
# splice, which also streams StreamVGGT:
#   EXPERIMENT=step   sbatch ...          # fed depth x1.25 from frame 40
#   EXPERIMENT=sine   sbatch ...          # periods 4..64, +-20%
#   EXPERIMENT=jitter sbatch ...          # sigma 0.1 per-frame log noise
#   EXPERIMENT=splice sbatch ...          # two clips, the second world x2
#   MAX_CLIPS=2 sbatch -p ai-tenn-debug -q ai-tenn-debug --gres=gpu:h100:1 --time=01:00:00 ...   # smoke
#
# A40 / H100 only: our model was benchmarked in bf16.
# =============================================================================
set -euo pipefail

REPO=${REPO:-/nfs/home/jdosch1/brown-visual-computing/MeTRICS}
CK=/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd/metric_all_datasets
RUN=${RUN:-$CK/50e636145fa9afbb}
DROP=${DROP:-20}
WINDOW=${WINDOW:-20}
MAX_CLIPS=${MAX_CLIPS:-0}
EXPERIMENT=${EXPERIMENT:-dropout}
EXTRA=${EXTRA:-}  # passed through, e.g. EXTRA="--sine-periods 8 32"
case "$EXPERIMENT" in
    dropout) TAG=drop${DROP} ;;
    *) TAG=$EXPERIMENT ;;
esac
# a smoke or an EXTRA variant must not land in (or block) the full run's dir
[ "$MAX_CLIPS" != 0 ] && TAG=${TAG}_clips${MAX_CLIPS}
[ -n "$EXTRA" ] && TAG=${TAG}_$(printf '%s' "$EXTRA" | tr -cs 'A-Za-z0-9.' '_' | sed 's/^_//;s/_$//')
OUT=${OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/scale_drift/$(basename "$RUN")_${TAG}_w${WINDOW}}

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

[ -f "$RUN/checkpoint-last.pth" ] || { echo "[fatal] no checkpoint-last.pth in $RUN"; exit 1; }
[ -e "$OUT/summary.json" ] && { echo "[fatal] $OUT already has results; set OUT"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/src"
COMMON=(--out "$OUT" --window "$WINDOW" --max-clips "$MAX_CLIPS")
# shellcheck disable=SC2086
case "$EXPERIMENT" in
    dropout)
        python scale_drift.py "${COMMON[@]}" --ref-frames "$DROP" --dropout-at "$DROP" \
            --curve "StreamVGGT=base" \
            --curve "Ours (dropout @${DROP})=$RUN@$DROP" \
            --curve "Ours (fed)=$RUN" $EXTRA ;;
    splice)
        python scale_drift.py "${COMMON[@]}" --experiment splice \
            --curve "StreamVGGT=base" --curve "Ours=$RUN" $EXTRA ;;
    step|sine|jitter)
        python scale_drift.py "${COMMON[@]}" --experiment "$EXPERIMENT" \
            --curve "Ours=$RUN" $EXTRA ;;
    *) echo "[fatal] unknown EXPERIMENT=$EXPERIMENT" >&2; exit 1 ;;
esac
