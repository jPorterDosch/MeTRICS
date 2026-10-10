#!/bin/bash
#SBATCH --job-name=kitti_ovda_download
#SBATCH --account=isaac-utk0256
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/kitti_ovda_download_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/kitti_ovda_download_%j.out

# KITTI for reproducing Online Video Depth Anything's tables: the raw colour
# frames (image_02 and image_03 only) of the 138 drives that have
# data_depth_annotated/train GT, the set the authors evaluate
# (third_party/ovda/evaluation/sequences.csv). Each ~1.3 GB raw sync zip is
# streamed: downloaded, its two colour folders extracted, deleted. raw/kitti held
# 96 GB afterwards (2026-10-08, GT and earlier drives included); peak extra
# disk one zip plus its two colour folders.
#
# Nothing existing is overwritten, and no file in $KITTI is ever partial: each
# zip is extracted into a staging directory beside it (same filesystem), and
# every PNG is then moved into place with `mv -n` -- a rename, so a file is
# either absent or complete, and one already there (the 13 validation drives)
# is kept. Resumable: a drive is skipped once its marker
# (<drive>/.ovda_rgb_complete) exists, written after all its files moved; a
# job killed before that redoes the drive from a fresh staging directory.
#
# The whole set takes about 5 h, longer than `short`'s 3 h limit: resubmit
# until the log says "all drives present". (`long` would fit it in one job,
# but its 18-submitted-jobs-per-user QOS limit is often full.)
#
# Then lays out $EVAL_RAW/kitti_ovda as the authors' kitti_pairs() expects
# (kitti_depth/data_depth_annotated/train, kitti_raw/<date>/<drive>/...),
# as two symlinks onto $EVAL_RAW/kitti -- no data is copied.
#
#   sbatch datasets_download/download_kitti_ovda.sh

set -euo pipefail
EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
REPO="${REPO:-/nfs/home/jdosch1/brown-visual-computing/MeTRICS}"
CSV="$REPO/third_party/ovda/evaluation/sequences.csv"
KITTI="$EVAL_RAW/kitti"
ZIPS="$EVAL_RAW/kitti_zips_tmp"
mkdir -p "$ZIPS"

drives=$(awk -F, '$1=="kitti"{sub(/-image_0[23]$/,"",$3); print $3}' "$CSV" | sort -u)
echo "$(echo "$drives" | wc -l) drives"
for drive_sync in $drives; do
    drive="${drive_sync%_sync}"
    date="${drive:0:10}"
    marker="$KITTI/$date/$drive_sync/.ovda_rgb_complete"
    [ -e "$marker" ] && continue
    zip="$ZIPS/${drive_sync}.zip"
    wget -q -c -O "$zip" "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${drive}/${drive_sync}.zip"
    stage="$ZIPS/${drive_sync}_stage"
    rm -rf "$stage"
    unzip -q "$zip" "*/image_02/data/*" "*/image_03/data/*" -d "$stage"
    for camera in image_02 image_03; do
        src="$stage/$date/$drive_sync/$camera/data"
        dst="$KITTI/$date/$drive_sync/$camera/data"
        mkdir -p "$dst"
        find "$src" -maxdepth 1 -type f -exec mv -n -t "$dst" {} +
    done
    rm -rf "$stage" "$zip"
    touch "$marker"
    echo "$(date '+%F %T') $drive_sync"
done
rmdir "$ZIPS" 2>/dev/null || true
echo "all drives present"

mkdir -p "$EVAL_RAW/kitti_ovda/kitti_depth"
[ -e "$EVAL_RAW/kitti_ovda/kitti_depth/data_depth_annotated" ] || ln -s "$KITTI" "$EVAL_RAW/kitti_ovda/kitti_depth/data_depth_annotated"
[ -e "$EVAL_RAW/kitti_ovda/kitti_raw" ] || ln -s "$KITTI" "$EVAL_RAW/kitti_ovda/kitti_raw"
du -sh "$KITTI"
