#!/bin/bash
#SBATCH --job-name=hammer_scannet_noconf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out

# =============================================================================
# train_hammer_noconf.sh -- the CONFIDENCE-TERM ablation, on HAMMER + ScanNet
# at stride 1. BOTH arms run from this one file, back to back: control
# (confidence on) then arm (off).
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
# -----------------------------------------------------------------------------
# WHY HAMMER + SCANNET AND NOT HAMMER ALONE. This is the change from the
# HAMMER-only version of this script, and it is not a preference -- HAMMER-only
# cannot answer the question. Two independent reasons:
#
# 1. THE READOUT IS SATURATED. The two HAMMER-only stride arms finished at:
#
#      job 4288664 (stride 20)   AbsRel 0.0465   d1_metric 0.99634   d1_affine 0.99694
#      job 4364230 (stride  1)   AbsRel 0.0480   d1_metric 0.99633   d1_affine 0.99694
#
#    d1_metric agrees to five decimals and d1_affine is identical. There is no
#    dynamic range left in the in-domain metric, so a HAMMER-only conf ablation
#    returns a null BY CONSTRUCTION -- indistinguishable from "the term does
#    nothing". That is the same trap the gated-arm conditioning null fell into.
#
# 2. THE HAMMER-ONLY EPOCH SIZE WAS 4x ITS NO-REPEAT CEILING. HAMMER's stride-1
#    clip pool is 5,586 (num_views=10, include_naked=False). The old
#    epoch-size 4500 x 5 epochs = 22,500 draws = 4.03x every clip, across only
#    64 sequences. train_hammer_scannet_stride.sh already worked out the fix and
#    this script inherits it verbatim: 1100/epoch puts HAMMER at
#    5,500/5,586 = 0.98x, just under no-repeat, with ScanNet carrying the rest.
#
# WHAT ADDING SCANNET CHANGES ABOUT THE QUESTION -- state this in the writeup.
# The hypothesis above is specifically about transparent/specular pixels where
# the depth sensor fails. That is HAMMER's reason for existing; ScanNet has
# essentially none of it. So ScanNet is NOT here to test the hypothesis -- it is
# here to supply the measurable range and a second domain. HAMMER stays in the
# mixture because it is the only thing carrying the mechanism. Read the two
# datasets separately in eval (eval_all.sh does this), do not average them.
#
# EPOCH SIZE IS SPLIT 1100/3400 (HAMMER/ScanNet), total 4500 -- the same total
# as every arm this branches from, so wall-clock stays near the measured 14h16m
# (job 4288664) and these arms remain compute-comparable to the HAMMER-only
# runs. Do NOT "fix" the split back to uniform: at ScanNet ~4.2M vs HAMMER 5,586
# (~745x), uniform sampling over-weights the SMALL dataset into repetition,
# which is the opposite of what uniform sampling is for. Scale both up together
# for a longer run, but keep HAMMER at or below ~1100 x (epochs) / 5,586 = 1.0x.
#
# STRIDE IS FIXED AT 1 IN BOTH ARMS and is NOT the varied factor here. (1,1) is
# the value that maximizes temporal supervision (consecutive frames, which is
# what the temporal term assumes); the separate stride question lives in
# train_hammer_scannet_stride.sh. If that ablation lands on (1,4) or (1,20),
# this pair should be re-run at the winner -- change --train-dataset.stride-range
# in BOTH invocations below. Never mix this arm against a control trained at a
# different stride.
#
# -----------------------------------------------------------------------------
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
#   NAME=noconf_prescreen ./experiments/eval_all.sh \
#       checkpoints/hammer_stride_1/1caa4da680ba0b7d ablation
#
# Run that before this pair. If conf~err is strongly negative and the spread is
# wide, confidence IS doing something and this ablation is likely to cost
# accuracy -- worth knowing in advance, and it does not make the experiment
# wrong, only its expected sign.
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
# DEPTH-ALPHA IS ASYMMETRIC IN THIS EXPERIMENT -- READ THIS BEFORE RETUNING IT.
# In the stride ablation, alpha=0.1 being ~2x off for a HAMMER+ScanNet mixture
# (the ideal is ~0.05 for a 50/50 mix; see that script's alpha-coupling note) is
# harmless BECAUSE IT HITS BOTH ARMS EQUALLY and so cannot fake a stride effect.
# Here it does not: the varied factor deletes alpha's effect in the ablation arm
# entirely, so alpha tunes ONLY the control. A mistuned alpha therefore biases
# THIS contrast in a way it could not bias the stride one.
#
# It is nonetheless left at the 0.1 default, deliberately. The control arm is
# then exactly the objective we ship, which makes this a fair test of the thing
# we actually ship -- but it is NOT a test of confidence weighting at its best,
# and a control loss to the ablation must not be reported as "confidence
# weighting is worse", only as "confidence weighting as configured is worse".
# Separating those is an alpha sweep, which is a different experiment.
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
# Read HAMMER and ScanNet SEPARATELY -- the mixture's average is not a quantity
# either dataset's question is asked in, and per reason 1 above the HAMMER half
# may still be saturated even here.
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
# BASE CONFIG is train_hammer_scannet_stride.sh's stride-1 arm: TOKEN injection,
# DEPTH heads, LoRA qkvo r16 a32, 10 m norm, log-depth accuracy term, sim RANDOM
# patch14 ratio 0.95, 5 epochs, lr 1e-5, bs 1. The ONLY differences are the
# --exp-group names and the one loss flag.
#
# LOG-SPACE DEPTH LOSS IS ON IN BOTH ARMS (--loss.depth-log-space): the accuracy
# term is |log pred - log gt| so the far background stops dominating the L1
# while the metric scale penalty is kept (unlike scale-invariant SILog). Being
# in both arms, it cannot affect the confidence contrast either way. NOTE this
# is NOT depth_cond.log_depth (the conditioning-INPUT log1p, on by default) --
# same word, different knob, different purpose.
#
# WHAT IT CANNOT SHOW. Whether alpha is merely MISTUNED rather than the
# mechanism being wrong (see the alpha note above). It also cannot speak to
# domains outside the training mixture on its own -- that is what the spot stage
# of eval_all.sh is for.
#
# Cost: no extra compute vs any other arm (the ablation removes work). Reference
# is 14h16m for 5 epochs on one L40S, so budget ~28h for the pair against the
# 48h wall.
#
# Outputs: checkpoints + manifest.json (which records loss.depth_conf_weighting,
# so the two run hashes differ on that field alone) under
#   ${REPO}/checkpoints/hs_conf_on/<run_id>/    control
#   ${REPO}/checkpoints/hs_conf_off/<run_id>/   ablation
# Then, and the OOD stages are the point:
#   NAME=hs_conf_on  ./experiments/eval_all.sh checkpoints/hs_conf_on/<run_id>
#   NAME=hs_conf_off ./experiments/eval_all.sh checkpoints/hs_conf_off/<run_id>
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
HAMMER=/oscar/scratch/jdosch/data/processed_hammer
SCANNET=/gpfs/data/jtompki1/cli277/metric/processed_scannet

# --- environment: the StreamVGGT conda env has torch/accelerate/tyro etc. ---
export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
# expandable_segments: avoids fragmentation-class OOMs on L40S; every arm sets
# it for parity (the token head-only arm OOM'd exactly this way).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# absolute path so the script works when sbatch'd from any CWD
source "$REPO/.secrets/wandb-personal.env"

mkdir -p "$REPO/logs/hammer_finetune"

for d in "$HAMMER" "$SCANNET"; do
    [ -d "$d" ] || { echo "dataset root missing: $d" >&2; exit 1; }
done

# finetune_depth.py resolves relative paths against the CWD, so run from src/
cd "$REPO/src"

# ------------------------------------------------------- CONTROL: confidence ON
python finetune_depth.py \
    --exp-group "hs_conf_on" \
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
    `# --- train data: HAMMER + ScanNet as one CatDataset. The parallel`      \
    `# tuples are index-aligned: entry i of every tuple describes dataset i.` \
    `# 1100 keeps HAMMER at 0.98x its 5,586-clip no-repeat ceiling over 5`    \
    `# epochs; ScanNet's ~4.2M pool carries the remaining 3400.`              \
    --train-dataset.root "$HAMMER" "$SCANNET" \
    --train-dataset.dataset HAMMER SCANNET \
    --train-dataset.stride-range 1 1 1 1 \
    --train-dataset.epoch-size 1100 3400 \
    --train-dataset.highres-root None None \
    \
    `# --- val data: both TEST splits. (1,1) is REQUIRED on TEST, not a`      \
    `# choice -- DatasetConfig.validate rejects anything else -- so the eval`  \
    `# protocol is identical across every arm. Valing on both keeps`          \
    `# checkpoint-best from being chosen on one domain.`                      \
    --val-dataset.root "$HAMMER" "$SCANNET" \
    --val-dataset.dataset HAMMER SCANNET \
    --val-dataset.stride-range 1 1 1 1 \
    --val-dataset.epoch-size 500 500 \
    --val-dataset.highres-root None None \
    \
    `# --- loss: the SHIPPED objective. Confidence weighting on, its`         \
    `# -alpha*log(sigma) regularizer at the 0.1 default (see the asymmetry`   \
    `# note in the header -- alpha tunes THIS arm only). log-space accuracy`  \
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
    --exp-group "hs_conf_off" \
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
    `# --- train data: identical to the control arm -------------------------` \
    --train-dataset.root "$HAMMER" "$SCANNET" \
    --train-dataset.dataset HAMMER SCANNET \
    --train-dataset.stride-range 1 1 1 1 \
    --train-dataset.epoch-size 1100 3400 \
    --train-dataset.highres-root None None \
    \
    `# --- val data: identical to the control arm ---------------------------` \
    --val-dataset.root "$HAMMER" "$SCANNET" \
    --val-dataset.dataset HAMMER SCANNET \
    --val-dataset.stride-range 1 1 1 1 \
    --val-dataset.epoch-size 500 500 \
    --val-dataset.highres-root None None \
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
