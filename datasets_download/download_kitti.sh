# kitti
set -eu
EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
mkdir -p "$EVAL_RAW/kitti"
cd "$EVAL_RAW/kitti"
# Every fetch is skipped when its unpacked result exists: the zips are deleted
# after unpacking, so without the guard a rerun (e.g. to add the calib files)
# would download the 13 drives again.
[ -d depth_selection ] || wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip
[ -d val ] || wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
for drive in \
    2011_09_26_drive_0002 2011_09_26_drive_0005 2011_09_26_drive_0013 \
    2011_09_26_drive_0020 2011_09_26_drive_0023 2011_09_26_drive_0036 \
    2011_09_26_drive_0079 2011_09_26_drive_0095 2011_09_26_drive_0113 \
    2011_09_28_drive_0037 2011_09_29_drive_0026 2011_09_30_drive_0016 \
    2011_10_03_drive_0047; do
    date="${drive:0:10}"
    [ -d "$date/${drive}_sync" ] || \
        wget -c "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${drive}/${drive}_sync.zip"
done
# per-date calibration (cam_to_cam / velo_to_cam / imu_to_velo, a few KB each):
# the benchmark's GT cameras come from oxts + these
for date in 2011_09_26 2011_09_28 2011_09_29 2011_09_30 2011_10_03; do
    [ -f "$date/calib_cam_to_cam.txt" ] || \
        wget -c "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}_calib.zip"
done
find . -maxdepth 1 -name "*.zip" -exec unzip -o -q {} \;
# remove all zip files
find . -maxdepth 1 -name "*.zip" -exec rm {} \;
cd ..
