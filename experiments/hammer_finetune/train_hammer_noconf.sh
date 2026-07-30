#!/bin/bash
#SBATCH --job-name=hammer_noconf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out

# =============================================================================
# train_hammer_noconf.sh -- the CONFIDENCE-TERM ablation. BOTH arms run from
# this one file, back to back: control (confidence on) then arm (off).
#
#   sbatch experiments/hammer_finetune/train_hammer_noconf.sh
#
# THE ONE VARIED FACTOR is --loss.no-depth-conf-weighting. Everything else is
# identical between the two invocations below. Change anything, change it in
# BOTH, or the contrast stops being single-variable.
#
# WHAT THE TERM IS. The depth objective (loss/head_loss.py: DepthOrPmapLoss) is
# an aleatoric-uncertainty / learned-confidence loss:
#
#   Ldepth = sigma * |pred - gt|  +  grad  -  alpha * log(sigma)
#            \____ main ______/      \__/     \____ reg _______/
#
# sigma is the depth head's OWN per-pixel output (p["depth_conf"]), so the model
# picks its own per-pixel loss weights. The -alpha*log(sigma) term is what stops
# it picking zero everywhere. Both parts come out together here -- see WHY BOTH.
#
#   control arm:  Ldepth = sigma*|pred-gt| + grad - 0.1*log(sigma)
#   ablation arm: Ldepth =       |pred-gt| + grad
#
# HYPOTHESIS. Confidence weighting is a per-pixel escape hatch: the model can
# cut its loss on pixels it finds hard by lowering sigma there instead of by
# predicting them better. On HAMMER the hard pixels are exactly the ones the
# whole project is about -- transparent and specular surfaces, where the sensor
# itself fails. So the term may be spending capacity on learning WHICH pixels to
# give up on, a skill that is tuned to the training distribution and does not
# transfer, while the accuracy signal on those pixels gets down-weighted. Under
# the ablation every GT-valid pixel counts the same, which is also what every
# metric we report (AbsRel/delta1/TAE) already assumes.
#
# It can also go the other way, and that is the real risk this run measures:
# down-weighting genuinely mislabeled pixels is robust regression, and HAMMER's
# GT is not clean. If the confidence term is mostly acting as a noise filter,
# removing it should show up as WORSE in-domain accuracy, not just worse OOD.
#
# MEASURE FIRST -- 28 GPU-hours is a lot to spend on a premise that is already
# testable on an EXISTING checkpoint. tests/conditioning_ablation_eval.py
# (eval_all.sh's `ablation` stage) already prints, per arm:
#
#   conf~err   rank correlation between predicted sigma and |error|
#   conf p10 / conf p90   the spread of sigma
#
# A well-calibrated confidence head is STRONGLY NEGATIVE on conf~err (high sigma
# exactly where error is low). Near zero = sigma has stopped discriminating good
# pixels from bad, i.e. the term is not contributing, which is the premise of
# this ablation. A tight p10/p90 spread says the same thing a second way: sigma
# has collapsed to a constant, and a constant sigma makes `main` a plain rescale
# of the error, so the whole mechanism reduces to a learning-rate multiplier.
#
#   python tests/conditioning_ablation_eval.py --weights checkpoints/<run> \
#       --checkpoint best --clips 32 --include-base --pretrained ckpt/checkpoints.pth
#
# Run that on the current best checkpoint before this pair. If conf~err is
# strongly negative and the spread is wide, confidence IS doing something and
# this ablation is likely to cost accuracy -- worth knowing in advance, and it
# does not make the experiment wrong, only its expected sign.
#
# WHY BOTH PARTS AT ONCE, AND NOT alpha=0. Dropping the regularizer alone
# (--loss.depth-alpha 0) leaves min sigma*|pred-gt|, which the model minimizes
# by driving sigma -> 0: the accuracy term goes to zero and the depth head is
# then supervised only by `grad` (shape, no scale). That is a degenerate
# objective, not an ablation, so "remove the confidence regularization term"
# means removing the weighting with it. The knob enforces this -- with
# --loss.no-depth-conf-weighting, depth_alpha becomes a verified no-op
# (tests/loss_conf_ablation.py check 4).
#
# HOW TO READ IT -- THE LOSSES ARE NOT COMPARABLE ACROSS ARMS. One carries a
# sigma factor and a log term, the other does not, so total / Ldepth /
# Ldepth_main are on different scales BY CONSTRUCTION. A "lower loss" in either
# arm means nothing. What is comparable:
#
#   absrel_metric / delta1 / TAE   sigma-free metrics; the actual readout, and
#                                  already what checkpoint-best selects on
#                                  (finetune_depth.py:493-502), so both arms'
#                                  "best" checkpoints are chosen the same way
#   Ldepth_main_raw                the sigma-FREE masked error. In the ablation
#                                  arm main_raw == main; in the control it is
#                                  the accuracy hiding under the sigma weight
#   Ldepth_grad                    identical objective in both arms
#   Ltemporal                      untouched by this ablation (verified)
#
# The control's own log already answers the "does it contribute much" question
# independently of this run: if Ldepth_main climbs on val while Ldepth_main_raw
# stays flat, the control is overfitting its CONFIDENCE, not its depth -- which
# is the mechanism above, caught in a single run. Ldepth_conf (mean sigma) shows
# whether sigma is doing anything at all or has saturated. In the ablation arm
# those logged components degenerate on purpose: conf == 1, reg == 0,
# main == main_raw.
#
# SIDE EFFECT, INTENDED: nothing else in the depth_train recipe consumes
# p["depth_conf"], so under the ablation the head's confidence channel gets NO
# gradient and stays at its pretrained values. The checkpoint is still
# export/viz-compatible (export_onnx and the GLB path both still emit conf; the
# GLB's "depth_conf" is a recomputed valid&finite mask, not the model's), but
# the arm's conf OUTPUT is not trained and must not be compared against the
# control's as if it were. visualize_depth.py's per-clip (mean,min,max) conf
# readout is the cheap check that this held: it should sit at the pretrained
# values in the arm and have moved in the control.
#
# SCOPE. The knob is DEPTH_TRAIN-only, like depth_metric and depth_log_space.
# The finetune_train recipe has a SECOND, separate confidence mechanism
# (ConfLoss wrapping Regr3DPose) which this does not touch -- no script trains
# with that recipe, so it is out of scope here.
#
# BASE CONFIG is train_hammer_stride.sh's ARM (stride 1 1 + --loss.depth-log-space,
# TOKEN injection, DEPTH heads, LoRA qkvo r16 a32, depth-alpha at its 0.1
# default). depth-alpha is deliberately NOT set: 0.1 is right for HAMMER's ~1.0 m
# median scene depth (see that script's alpha-coupling note) and it only affects
# the control arm anyway. If the stride ablation lands on (1,20) instead, change
# --train-dataset.stride-range in BOTH invocations below and re-run the pair --
# do not mix this arm against a control trained at a different stride.
#
# WHAT IT CANNOT SHOW. A null result on HAMMER metrics does not clear the term:
# HAMMER is a near-static tabletop capture and in-domain, so the generalization
# question lives in the scannet and spot stages of eval_all.sh, not the hammer
# one. It also says nothing about whether alpha is merely MISTUNED rather than
# the mechanism being wrong -- an alpha sweep is a different experiment.
#
# Cost: no extra compute vs any other arm (the ablation removes work). Reference
# is 14h16m for 5 epochs on one L40S, so budget ~28h for the pair against the
# 48h wall.
#
# Outputs: checkpoints + manifest.json (which records loss.depth_conf_weighting,
# so the two run hashes differ on that field alone) under
#   ${REPO}/checkpoints/hammer_conf_on/<run_id>/    control
#   ${REPO}/checkpoints/hammer_conf_off/<run_id>/   ablation
# Then, and the OOD stages are the point:
#   NAME=hammer_conf_on  ./experiments/eval_all.sh checkpoints/hammer_conf_on/<run_id>
#   NAME=hammer_conf_off ./experiments/eval_all.sh checkpoints/hammer_conf_off/<run_id>
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
DATA=/oscar/scratch/jdosch/data/processed_hammer

# --- environment: the StreamVGGT conda env has torch/accelerate/tyro etc. ---
export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
# expandable_segments: avoids fragmentation-class OOMs on L40S; every arm sets
# it for parity (the token head-only arm OOM'd exactly this way).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# absolute path so the script works when sbatch'd from any CWD
source "$REPO/.secrets/wandb-personal.env"

mkdir -p "$REPO/logs/hammer_finetune"

[ -d "$DATA" ] || { echo "dataset root missing: $DATA" >&2; exit 1; }

# finetune_depth.py resolves relative paths against the CWD, so run from src/
cd "$REPO/src"

# ------------------------------------------------------- CONTROL: confidence ON
python finetune_depth.py \
    --exp-group "hammer_conf_on" \
    \
    `# --- model / checkpointing -------------------------------------------` \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: TOKEN + LoRA -----------------------------------` \
    --depth-cond.injection TOKEN \
    --depth-cond.heads DEPTH \
    --train.train-heads DEPTH \
    \
    `# --- train data: HAMMER, consecutive frames --------------------------` \
    --train-dataset.root "$DATA" \
    --train-dataset.dataset HAMMER \
    --train-dataset.stride-range 1 1 \
    --train-dataset.epoch-size 4500 \
    --train-dataset.highres-root None \
    \
    `# --- val data: HAMMER test split. (1,1) is REQUIRED on TEST, not a`     \
    `# choice -- DatasetConfig.validate rejects anything else.`               \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.stride-range 1 1 \
    --val-dataset.epoch-size 1000 \
    --val-dataset.highres-root None \
    \
    `# --- loss: the SHIPPED objective. Confidence weighting on, its`         \
    `# -alpha*log(sigma) regularizer at the 0.1 default. log-space accuracy`  \
    `# term, matching the stride arms this pair branches from.`               \
    --loss.depth-log-space \
    \
    `# --- optimization ----------------------------------------------------` \
    --batch-size 1 \
    --accum-iter 1 \
    --epochs 5 \
    --lr 1e-5 \
    --min-lr 1e-7 \
    --warmup-epochs 0.5 \
    --weight-decay 0.05 \
    --amp 1 \
    --seed 42 \
    \
    `# --- cadence ---------------------------------------------------------` \
    --val-freq 1 \
    --save-freq 0.1 \
    --num-workers 8 \
    --print-freq 10

# ----------------------------------------------------------- ARM: confidence OFF
# Identical to the block above except --exp-group and the ONE loss flag.
python finetune_depth.py \
    --exp-group "hammer_conf_off" \
    \
    `# --- model / checkpointing -------------------------------------------` \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: TOKEN + LoRA -----------------------------------` \
    --depth-cond.injection TOKEN \
    --depth-cond.heads DEPTH \
    --train.train-heads DEPTH \
    \
    `# --- train data: HAMMER, consecutive frames --------------------------` \
    --train-dataset.root "$DATA" \
    --train-dataset.dataset HAMMER \
    --train-dataset.stride-range 1 1 \
    --train-dataset.epoch-size 4500 \
    --train-dataset.highres-root None \
    \
    `# --- val data: identical to the control arm ---------------------------` \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.stride-range 1 1 \
    --val-dataset.epoch-size 1000 \
    --val-dataset.highres-root None \
    \
    `# --- loss: THE VARIED FACTOR. no-depth-conf-weighting drops BOTH the`   \
    `# sigma weighting and its -alpha*log(sigma) regularizer, so the depth`   \
    `# term is the plain masked error + grad. depth-alpha is a no-op here;`   \
    `# it is left unset so the two command lines differ by one flag only.`    \
    --loss.no-depth-conf-weighting \
    --loss.depth-log-space \
    \
    `# --- optimization ----------------------------------------------------` \
    --batch-size 1 \
    --accum-iter 1 \
    --epochs 5 \
    --lr 1e-5 \
    --min-lr 1e-7 \
    --warmup-epochs 0.5 \
    --weight-decay 0.05 \
    --amp 1 \
    --seed 42 \
    \
    `# --- cadence ---------------------------------------------------------` \
    --val-freq 1 \
    --save-freq 0.1 \
    --num-workers 8 \
    --print-freq 10
