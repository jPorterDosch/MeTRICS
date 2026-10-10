#!/bin/bash
# One-time: the venv Depth Any Video runs from (login node; ~6 GB on Lustre).
#
#   bash experiments/baselines/setup_dav_env.sh
#
# The pins are upstream's requirements.txt (third_party/depth_any_video/
# README_VENDORED.md). They cannot go into the `metrics` venv: it has no
# diffusers, and its transformers 5.0 / huggingface_hub 1.22 are far past
# what diffusers 0.30.3 imports. Nothing here touches `metrics`.
# pyyaml and safetensors are what src/eval/baselines/arms.py imports on top.

set -euo pipefail

DAV_ENV=${DAV_ENV:-/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd/dav}
BASE_PY=/nfs/home/jdosch1/.pyenv/versions/3.11.13/bin/python

[ -e "$DAV_ENV" ] && { echo "$DAV_ENV exists; remove it to rebuild" >&2; exit 1; }
"$BASE_PY" -m venv "$DAV_ENV"
"$DAV_ENV/bin/python" -m pip install --upgrade pip
"$DAV_ENV/bin/python" -m pip install \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
"$DAV_ENV/bin/python" -m pip install \
    diffusers==0.30.3 accelerate==0.31.0 transformers==4.43.2 huggingface-hub==0.24.2 \
    "numpy<2" opencv-python-headless tqdm matplotlib scipy pillow easydict pyyaml safetensors
"$DAV_ENV/bin/python" - <<'PY'
import diffusers, torch, transformers
print("dav env ok:", torch.__version__, diffusers.__version__, transformers.__version__)
PY
du -sh "$DAV_ENV"
