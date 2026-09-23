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
#   1. export the first 510 frames of each ScanNet scans_test .sens into
#      $EVAL_RAW/scannet (the raw tree is not ours to write into), colour at
#      the depth size so VDA's [8:-8, 11:-11] crop cuts the same field of
#      view from both
#   2. run prepare_vda_benchmark.py for every dataset
#
#   sbatch datasets_preprocess/prepare_vda_benchmark.sbatch
#   DATASETS="bonn nyuv2" sbatch datasets_preprocess/prepare_vda_benchmark.sbatch
#
# CPU only; a few minutes per dataset except ScanNet (~1-2 h for the .sens
# decode). Resumable: the .sens export skips complete scenes and the
# preparer skips a dataset whose manifest exists.

set -eu

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
export EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
export EVAL_OUT="${EVAL_OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench}"
DATASETS="${DATASETS:-sintel kitti bonn scannet nyuv2}"

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim

cd "$REPO"
case " $DATASETS " in
*" scannet "*)
    echo "== scannet .sens export -> $EVAL_RAW/scannet ($(date))"
    python datasets_preprocess/extract_scannet_sens.py \
        --raw-root /lustre/isaac24/proj/UTK0516/metrics_data/scannet \
        --splits scans_test --extracted \
        --out-root "$EVAL_RAW/scannet" --max-frames 510 --color-at-depth-res
    ;;
esac

echo "== prepare $DATASETS ($(date))"
# shellcheck disable=SC2086
python datasets_preprocess/prepare_vda_benchmark.py $DATASETS --raw "$EVAL_RAW" --out "$EVAL_OUT"
echo "== done ($(date))"
du -sh "$EVAL_OUT"/* 2>/dev/null || true
