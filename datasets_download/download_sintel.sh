# Download Sintel
set -eu
EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
mkdir -p "$EVAL_RAW/sintel"
cd "$EVAL_RAW/sintel"
# each fetch is skipped once its unpacked result exists (the zips are deleted
# after unpacking, so a rerun would otherwise download everything again)
# images
[ -d training/clean ] || wget -c --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip
# depth & cameras
[ -d training/depth ] || wget -c --no-proxy http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
# flow
[ -d training/flow ] || wget -c --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip
# unzip all
find . -name "*.zip" -exec unzip -o -q {} \;
# remove all zip files
find . -name "*.zip" -exec rm {} \;
cd ..
