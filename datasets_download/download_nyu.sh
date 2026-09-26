# nyu (NYU Depth v2, labeled 1449-image set; the 654-image test split is
# splits.mat's testNdxs, applied at extraction time)
set -eu
EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
mkdir -p "$EVAL_RAW/nyu"
cd "$EVAL_RAW/nyu"
wget -c http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget -c http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
cd ..
