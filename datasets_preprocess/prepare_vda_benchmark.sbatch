#!/bin/bash
#SBATCH --job-name=vda_bench_prep
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/vda_bench_prep_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/vda_bench_prep_%j.out

# Build the video-depth benchmark tree from the raw downloads
# (download_eval_benchmarks.sbatch must have finished):
#   1. export the first 510 frames of each ScanNet scans_test scene into
#      $EVAL_RAW/scannet, colour at the depth size so VDA's [8:-8, 11:-11]
#      crop cuts the same field of view from both (SCANNET_FROM below)
#   2. run prepare_vda_benchmark.py for every dataset
#
#   sbatch datasets_preprocess/prepare_vda_benchmark.sbatch
#   DATASETS="bonn nyuv2" sbatch datasets_preprocess/prepare_vda_benchmark.sbatch
#
# CPU only; ~1-2 h with the .sens export, ~15 min from the processed tree.
# Resumable: the ScanNet export skips complete scenes and the preparer skips
# a dataset whose manifest exists.

set -eu

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
export EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
export EVAL_OUT="${EVAL_OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench}"
DATASETS="${DATASETS:-sintel kitti bonn scannet nyuv2}"

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim

cd "$REPO"
# SCANNET_FROM=sens (default): decode the raw .sens (~1-2 h), exact VDA frame
# selection. Needs read access to metrics_data/scannet/scans_test/*/*.sens
# (0600 to hqi -- ask for `chmod g+r`). SCANNET_FROM=processed: the repo's
# processed tree (minutes, readable) -- but preprocess_scannet.py dropped
# non-finite-pose frames, which shifts "the first 510 frames" on 16 of the
# 100 scenes (scene0791_00 by 1011). Fallback only; not Table-1 parity.
SCANNET_FROM="${SCANNET_FROM:-sens}"
case " $DATASETS " in
*" scannet "*)
    echo "== scannet export ($SCANNET_FROM) -> $EVAL_RAW/scannet ($(date))"
    if [ "$SCANNET_FROM" = sens ]; then
        python datasets_preprocess/extract_scannet_sens.py \
            --raw-root /lustre/isaac24/proj/UTK0516/metrics_data/scannet \
            --splits scans_test --extracted \
            --out-root "$EVAL_RAW/scannet" --max-frames 510 --color-at-depth-res
    else
        python datasets_preprocess/export_scannet_eval_from_processed.py \
            --out-root "$EVAL_RAW/scannet" --max-frames 510
    fi
    ;;
esac

echo "== prepare $DATASETS ($(date))"
# shellcheck disable=SC2086
python datasets_preprocess/prepare_vda_benchmark.py $DATASETS --raw "$EVAL_RAW" --out "$EVAL_OUT"
echo "== done ($(date))"
du -sh "$EVAL_OUT"/* 2>/dev/null || true
