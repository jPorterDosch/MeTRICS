# Code review findings

`.git` was read-only for the entire run. No fix was applied and no commit was made. The review plan's required per-commit before/after test gate was therefore unsatisfiable and is not claimed. CPU probes below verify reviewer claims only; they are not a substitute for that gate.

## Area A — `src/streamvggt/loss/`

Role: Defense, no-commit mode.

### R1-1 and R2-1 — all-invalid GT becomes all-valid supervision

**Verdict: UPHELD — FIXED in bac1cac.** Raised independently by R1 (wrong-numbers lens) and R2 (contracts-and-runtime lens); these are the same defect.

Failure scenario: with two all-invalid frames, zero stored GT depth, unit predicted depth, and unit confidence, `DepthTrainLoss(metric=True)` reports total `1.0`, `Ldepth=1.0`, and `Ltemporal=0.0`. The same GT is therefore invalid to the temporal term and fully supervised to the depth term. Equivalent `ones_like` fallbacks affect finetune depth and distillation depth/point maps.

Minimal patch, **NOT APPLIED**: in `depth_train_loss.py:117-124`, `finetune_loss.py:70-79`, and `distill_loss.py:37-43,56-66`, remove the `if not valid_mask.any(): valid_mask = torch.ones_like(...)` fallbacks. Skip each empty-mask term instead; for distillation point maps, mirror the existing empty-term zero fallback used for depth so `torch.stack([])` is not reached.

### R1-2 — scale/shift fit merges the batch and includes invalid pixels

**Verdict: UPHELD — FIXED in 0a6a819.** Raised by R1.

Failure scenario: two independently alignable samples with required scales 2 and 4 are flattened into one fit with scale 3. Invalid storage values also affect that fit because `closed_form_scale_and_shift` has no mask input.

Minimal patch, **NOT APPLIED**: in `utils.py:138-178`, retain the batch dimension when computing means and covariance, compute one scale/shift per sample, and exclude entries where a passed validity mask is false. In `head_loss.py:159-160`, pass `valid_mask` and reshape the returned per-sample scale/shift only for broadcasting over `(H,W,C)`.

### R1-3 — trimmed MAE loses its denominator and image ownership

**Verdict: UPHELD — FIXED in c03c57b.** Raised by R1.

Failure scenario: ten valid pixels with error 1 and `trim=0.2` return `0.800000011920929`, not 1, because eight retained errors are divided by the original count of ten. The globally flattened sort also makes image-based reduction associate sorted pixels with unrelated per-image counts.

Minimal patch, **NOT APPLIED**: in `trimmed_loss.py:92-103`, for batch-based reduction divide the retained sorted errors by their retained count. For image-based reduction, retain the batch index, trim each image's valid residuals independently, average each nonempty image by its retained count, then average those image losses. Do not pass a globally shortened vector with the original `M`.

### R1-4 — spatial gradients drop edge strips and average masked zeros

**Verdict: UPHELD — FIXED in c018c86.** Raised by R1.

Failure scenario: one valid horizontal edge of unit error produces `0.5` in a 2x2 image and `0.006172839552164078` in a 10x10 image. A horizontal edge on the last row is removed by cropping both directional fields to `(H-1,W-1)`.

Minimal patch, **NOT APPLIED**: in `head_loss.py:119-137`, remove the common `min_h`/`min_w` crop. Compute the horizontal mean from `(dx_pred-dx_gt).abs()[dx_mask]` and the vertical mean independently from `(dy_pred-dy_gt).abs()[dy_mask]`, using a graph-preserving zero for a direction with no valid edges, then average the two directional losses.

### R1-5 — normal loss ignores returned normal-validity masks

**Verdict: UPHELD — FIXED in c1175cd.** Raised by R1.

Failure scenario: identical planar 3x3 point maps with every source pixel valid produce normal loss `0.1111111044883728`; undefined boundary normals enter the unmasked mean.

Minimal patch, **NOT APPLIED**: in `head_loss.py:97-100`, retain the validity outputs from both `point_map_to_normal` calls, index cosine similarity by their intersection, and average only those entries; return a graph-preserving zero if the intersection is empty.

### R1-6 — teacher confidence arguments are ignored

**Verdict: STILL DEFERRED — run a one-GPU controlled DISTILL experiment on a
fixed 100-batch synthetic sequence set, comparing student-only weighting,
arithmetic-mean teacher/student confidence, and product weighting while scaling
teacher confidences by 0.1x and 10x; validation AbsRel/TAE and the sign/magnitude
of teacher-confidence-conditioned depth and track gradients must select the
combination and establish whether confidence is precision or uncertainty.** Raised by R1.

Failure scenario: changing teacher depth confidence from 1 to 100 left the depth loss at `1.0`; changing teacher track confidence from 0 to 100 left track loss at `2.10736083984375`. `sigma_g` and `w_g` are accepted but only the prediction-side values are used.

Reason deferred: the unused parameters and invariant results are real, but Area A does not establish whether teacher and student confidence should be averaged, multiplied, or whether the teacher values are intentionally ignored. The confidence docstring also conflicts with the implemented precision-like algebra. Restoring the commented averaging expressions would silently change the DISTILL objective without a verified contract. This finding is explicitly deferred because the proposed patch would need an explanatory design essay rather than a self-evident minimal correction.

### R1-7 — robust median includes invalid zeros

**Verdict: UPHELD — FIXED in 8c0caba.** Raised by R1.

Failure scenario: valid values `[10,10]` plus four invalid pixels report median 0, scale 10, and normalized valid values `[1,1]`; the center changes with mask density.

Minimal patch, **NOT APPLIED**: in `utils.py:94-96`, compute each valid batch item's median from `target[b][mask[b]]` rather than from `(mask * target).view(...)`; keep the existing all-invalid batch handling.

### R1-8 and R2-3 — temporal loss has no zero-stride or configuration-domain handling

**Verdict: UPHELD — FIXED in 0c4b18e.** R1-8 and R2-3(a) are the same `cnt == 0` defect for `T=1`; R2 additionally raised `temp_grad_scales=0`, `depth_trim=1.0`, and negative `diff_depth_th`.

Failure scenario: a one-frame tensor and a multi-frame loss configured with zero temporal scales both raise `ZeroDivisionError`. A full trim or negative threshold silently produces zero temporal signal.

Minimal patch, **NOT APPLIED**: in `gradient_loss.py:110`, return a graph-preserving zero when no stride is usable, allowing a valid one-frame input to carry no temporal signal. In `TemporalGradientMatchingLoss.__init__` (or `LossConfig.__post_init__` for the configuration boundary), raise `ValueError` when `temp_grad_scales < 1`, `trim` is outside `[0,1)`, or `diff_depth_th < 0`, naming the invalid field and value.

### R2-2 — unknown reduction strings silently select image-based reduction

**Verdict: UPHELD — FIXED in f14258c.** Raised by R2.

Failure scenario: both `GradientLoss(reduction="typo")` and `TrimmedMAELoss(reduction="typo")` select `reduction_image_based`, so a spelling error changes the objective instead of failing configuration.

Minimal patch, **NOT APPLIED**: in `types.py:166-171`, validate that `reduction` is exactly `"batch-based"` or `"image-based"` and raise `ValueError` naming any other value. This fails at config construction without changing either valid mode.

### R2-4 — non-finite camera predictions become zero error

**Verdict: UPHELD — FIXED in e2fa480.** Raised by R2.

Failure scenario: an all-NaN nine-component prediction against finite zero GT prints three replacement warnings and returns camera loss `0.0`, treating divergence as perfect.

Minimal patch, **NOT APPLIED**: at the start of `CameraLoss.forward` in `head_loss.py:19`, raise `ValueError` if `pred_pose` or `gt_pose` contains any non-finite value, naming the offending tensor; remove the three `check_and_fix_inf_nan` calls at lines 24-26 so a bad state cannot continue as zero error.

### R2-5 — `lambda_track` is a finetune no-op but changes identity

**Verdict: STILL DEFERRED — run one FINETUNE_TRAIN batch on a CUDA GPU with real
track targets and predictions present, then compare `lambda_track=0` and `0.05`:
if a defined track loss and nonzero track-head gradient are available, specify
and implement that term; if the recipe has no track supervision contract,
approve fail-fast validation of non-default `lambda_track` without changing the
frozen field or identity classification.** Raised by R2.

Failure scenario: `LossConfig` documents and passes `lambda_track` to `FinetuneLoss`, but `FinetuneLoss` neither stores it nor computes a track term. Different values therefore identify different FINETUNE experiments without changing criterion behavior.

Reason deferred: adding tracking to FINETUNE would be a larger recipe change requiring data/target and weighting decisions; removing or reclassifying the field would alter protected historical experiment identity. Neither is safe unsupervised, and the required identity fields are explicitly out of bounds for this run.

### R2-6 — confidence-disabled depth still requires `depth_conf`

**Verdict: UPHELD — FIXED in 999e1a7.** Raised by R2.

Failure scenario: `DepthOrPmapLoss(metric=True, conf_weighting=False)` accepts `sigma_p=None`, but the containing `DepthTrainLoss` raises `KeyError: 'depth_conf'` before calling it when a depth-only prediction is supplied.

Minimal patch, **NOT APPLIED**: in `depth_train_loss.py:116`, set `sigma_p = p["depth_conf"] if self.conf_weighting else None`; make no change to the confidence-enabled contract.

### R2-7 — scale-loss path has a deterministic call/signature mismatch

**Verdict: UPHELD — FIXED in 3e4d193.** Raised as a finding by R2 and previously raised as a question by R1.

Failure scenario: either call in `get_all_pts3d_with_scale_loss` passes five positional lists plus `norm_self_only=` to `get_norm_factor_point_cloud`, whose signature accepts three positional lists plus that keyword, producing `TypeError` before a scale loss. Repository search found no caller, so the path is dormant but still broken as defined.

Minimal patch, **NOT APPLIED**: in `regr_3d_pose.py:307-314`, pass only `gt_pts_self[:1]`, `valids[:1]`, and `conf_self[:1]` plus the keyword; in lines 321-328 do the same with `pr_pts_self[:1]`, `valids[:1]`, and `conf_self[:1]`. This matches the nearby “self view point maps” contract and the existing three-list signature.

### R2-8 — `CameraLoss.delta` is ignored

**Verdict: STILL DEFERRED — on CPU, obtain the camera-loss contract from the
originating recipe or checkpoint documentation and evaluate fixed pose residuals
at 0.05, 0.1, 0.2, and 100 with `delta=0.1`: evidence that the expected curve
changes at delta settles implementation of the specified robust loss; evidence
that the recipe requires plain L1 settles deprecation/removal in a compatibility
release, not this frozen-contract round.** Raised by R2.

Failure scenario: `CameraLoss(delta=0)` and `CameraLoss(delta=999)` have identical state and outputs because the constructor drops `delta`, while finetune and distill explicitly pass `0.1`.

Reason deferred: the no-op is established, but the source does not establish whether the minimal correct behavior is to remove the API parameter and call arguments or to implement a delta-based robust loss. The latter changes training numbers; the former may break external callers. Choosing between them needs contract confirmation.

### Reviewer questions (not findings)

- R1's scale-loss arity question is jointly resolved above as the UPHELD R2-7 finding.
- The `conf_loss.py` confidence prose appears reversed relative to the head activation and loss algebra, but it was submitted as a semantic question and is not assigned a finding verdict here; it contributes to deferring R1-6.
- `head_loss.py:209` describes separately logged means as a decomposition even though `mean(sigma * error)` generally differs from `mean(sigma) * mean(error)`. No downstream interpretation contract was established, so the submitted question is not promoted to a finding.
- R2's `regr_3d_pose.py:715-725` loop-variable question remains unresolved: Area A does not establish whether pose validity should use the last, anchor, or aggregated image mask. It is not promoted to a finding.

### CPU claim probes

These commands used the required interpreter and no GPU. The first attempt lacked the repository's `src` import path and failed before executing a probe. Its exact command was the corrected probe 1 command below with only the leading `PYTHONPATH=src ` omitted:

```text
$ /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
[same first probe body shown in the corrected command below]
PY
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/oscar/home/jdosch/MeTRIC/src/streamvggt/loss/__init__.py", line 1, in <module>
    from .types import LossConfig, LossName, PixelLoss, Recipe
  File "/oscar/home/jdosch/MeTRIC/src/streamvggt/loss/types.py", line 31, in <module>
    from .finetune_loss import FinetuneLoss
  File "/oscar/home/jdosch/MeTRIC/src/streamvggt/loss/finetune_loss.py", line 4, in <module>
    from streamvggt.utils.pose_enc import extri_intri_to_pose_encoding
ModuleNotFoundError: No module named 'streamvggt'
```

Corrected probe 1:

```python
PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import torch
from streamvggt.loss.gradient_loss import GradientLoss, TemporalGradientMatchingLoss
from streamvggt.loss.head_loss import CameraLoss, DepthOrPmapLoss
from streamvggt.loss.trimmed_loss import TrimmedMAELoss
from streamvggt.loss.utils import closed_form_scale_and_shift, normalize_prediction_robust

print('torch', torch.__version__, 'cuda_used', False)
pred = torch.tensor([[[[0.],[1.]]], [[[0.],[1.]]]])
gt = torch.tensor([[[[0.],[2.]]], [[[0.],[4.]]]])
print('global_scale_shift', *[x.tolist() for x in closed_form_scale_and_shift(pred, gt)])
ones = torch.ones(1, 2, 5)
print('trim_constant_batch', TrimmedMAELoss(trim=.2)(ones, torch.zeros_like(ones), torch.ones_like(ones, dtype=torch.bool)).item())
try:
    TemporalGradientMatchingLoss()(torch.zeros(1,1,2,2), torch.zeros(1,1,2,2), torch.ones(1,1,2,2,dtype=torch.bool))
except Exception as e:
    print('temporal_T1', type(e).__name__, str(e))
print('bad_reduction_impls', GradientLoss(reduction='typo')._GradientLoss__reduction.__name__, TrimmedMAELoss(reduction='typo')._TrimmedMAELoss__reduction.__name__)
plane = torch.stack(torch.meshgrid(torch.arange(3.), torch.arange(3.), indexing='ij'), -1)
plane = torch.cat([plane, torch.ones(3,3,1)], -1).unsqueeze(0)
print('perfect_normal_loss', DepthOrPmapLoss().normal_loss(plane, plane, torch.ones(1,3,3,dtype=torch.bool)).item())
target = torch.tensor([[[10.,10.,0.],[0.,0.,0.]]])
mask = torch.tensor([[[1,1,0],[0,0,0]]], dtype=torch.bool)
norm, (median, scale) = normalize_prediction_robust(target, mask)
print('robust_median_scale_validnorm', median.tolist(), scale.tolist(), norm[mask].tolist())
print('nan_camera_loss', CameraLoss()(torch.full((1,9), torch.nan), torch.zeros(1,9)).item())
loss = DepthOrPmapLoss(metric=True, conf_weighting=False)
print('conf_none_supported', loss(torch.ones(1,2,2,1), torch.zeros(1,2,2,1), sigma_p=None, valid_mask=torch.ones(1,2,2,dtype=torch.bool)).item())
PY
```

Full output:

```text
torch 2.3.1+cu121 cuda_used False
global_scale_shift [3.0] [0.0]
trim_constant_batch 0.800000011920929
temporal_T1 ZeroDivisionError division by zero
bad_reduction_impls reduction_image_based reduction_image_based
perfect_normal_loss 0.1111111044883728
robust_median_scale_validnorm [0] [10] [1.0, 1.0]
[warning] loss_T contains 3 inf/nan values, replacing with 0
[warning] loss_R contains 4 inf/nan values, replacing with 0
[warning] loss_FL contains 2 inf/nan values, replacing with 0
nan_camera_loss 0.0
conf_none_supported 1.0
```

Corrected probe 2:

```python
PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import torch
from streamvggt.loss.depth_train_loss import DepthTrainLoss
from streamvggt.loss.gradient_loss import TemporalGradientMatchingLoss
from streamvggt.loss.head_loss import DepthOrPmapLoss, TrackLoss

def views(valid=True, with_conf=True):
    gts=[]; preds=[]
    for _ in range(2):
        gts.append({'depthmap': torch.zeros(1,2,2), 'valid_mask': torch.full((1,2,2), valid)})
        p={'depth': torch.ones(1,2,2,1)}
        if with_conf: p['depth_conf']=torch.ones(1,2,2)
        preds.append(p)
    return gts,preds
loss, details = DepthTrainLoss(metric=True)(*views(valid=False))
print('all_invalid_total_depth_temporal', loss.item(), details['Ldepth'].item(), details['Ltemporal'].item())
try:
    DepthTrainLoss(metric=True, conf_weighting=False)(*views(with_conf=False))
except Exception as e:
    print('ablation_missing_conf', type(e).__name__, str(e))
d = DepthOrPmapLoss(metric=True)
pred=torch.ones(1,2,2,1); gt=torch.zeros_like(pred); mask=torch.ones(1,2,2,dtype=torch.bool); sp=torch.ones(1,2,2)
print('teacher_depth_conf_1_100', d(pred,gt,sp,torch.ones_like(sp),mask).item(), d(pred,gt,sp,torch.full_like(sp,100),mask).item())
t = TrackLoss(); ypr=torch.ones(1,1,1,2); ygt=torch.zeros_like(ypr); vpr=torch.zeros(1,1,1); vgt=torch.zeros_like(vpr); wp=torch.ones_like(vpr)
print('teacher_track_conf_0_100', t(ypr,ygt,vpr,vgt,wp,torch.zeros_like(wp)).item(), t(ypr,ygt,vpr,vgt,wp,torch.full_like(wp,100)).item())
g = DepthOrPmapLoss(metric=True)
for size in (2,10):
    pred=torch.zeros(1,size,size,1); gt=torch.zeros_like(pred); pred[0,0,1,0]=1
    m=torch.zeros(1,size,size,dtype=torch.bool); m[0,0,:2]=True
    print('spatial_grad_size', size, g.image_gradient_loss(pred,gt,m).item())
base=torch.tensor([[[[0.]],[[1.]],[[3.]]]])
target=torch.zeros_like(base); mask=torch.ones_like(base,dtype=torch.bool)
for kwargs in ({}, {'trim':1.0}, {'diff_depth_th':-0.1}, {'temp_grad_scales':0}):
    try: print('temporal_cfg', kwargs, TemporalGradientMatchingLoss(**kwargs)(base,target,mask).item())
    except Exception as e: print('temporal_cfg', kwargs, type(e).__name__, str(e))
PY
```

Full output:

```text
all_invalid_total_depth_temporal 1.0 1.0 0.0
ablation_missing_conf KeyError 'depth_conf'
teacher_depth_conf_1_100 1.0 1.0
teacher_track_conf_0_100 2.10736083984375 2.10736083984375
spatial_grad_size 2 0.5
spatial_grad_size 10 0.006172839552164078
temporal_cfg {} 0.0
temporal_cfg {'trim': 1.0} 0.0
temporal_cfg {'diff_depth_th': -0.1} 0.0
temporal_cfg {'temp_grad_scales': 0} ZeroDivisionError division by zero
```

Focused temporal configuration probe:

```python
PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import torch
from streamvggt.loss.gradient_loss import TemporalGradientMatchingLoss
prediction = torch.tensor([[[[0., 0.]], [[1., 1.]], [[3., 3.]]]])
target = torch.tensor([[[[0., 10.]], [[0., 10.]], [[0., 10.]]]])
mask = torch.ones_like(target, dtype=torch.bool)
for kwargs in ({}, {'trim': 1.0}, {'diff_depth_th': -0.1}):
    print(kwargs, TemporalGradientMatchingLoss(**kwargs)(prediction, target, mask).item())
PY
```

Full output:

```text
{} 0.875
{'trim': 1.0} 0.0
{'diff_depth_th': -0.1} 0.0
```

## Area B — `src/streamvggt/datasets/`

Role: Defense, no-commit mode.

### R1-1 — translation is added to ray directions

**Verdict: UPHELD — FIXED in e417c1c.** Raised by R1 (wrong-numbers lens).

Failure scenario: the CPU probe below gives a camera translated +1 on x an origin of `[1,0,0]` and a normalized direction of `[0.7071,0,0.7071]`. A direction must be rotation-only; the separately returned origin already carries translation.

Minimal patch, **NOT APPLIED**: in `base/base_multiview_dataset.py:70`, replace the homogeneous row of ones passed with `rd` by zeros, so the transform computes `R @ rd` rather than `R @ rd + t`; leave origin calculation unchanged.

### R1-2 and R2-2 — irregular sampling violates its configured gap range

**Verdict: UPHELD — FIXED in 4370adf.** Raised independently by R1 and R2; these are the same defect.

Failure scenario: four views from six positions with fixed requested gap 3 return `[0,3,4,5]`. Filtering the overshoot and backfilling arbitrary unused positions creates gap-1 frames despite the declared fixed gap.

Minimal patch, **NOT APPLIED**: in `base/base_multiview_dataset.py:296-314`, replace irregular overshoot filtering/backfill with sequential gap draws whose upper bound reserves the configured minimum gap for every remaining view. When the original minimum is infeasible, use the already computed feasible lower bound and warning. Every emitted gap must remain within those effective bounds; do not backfill arbitrary positions.

### R1-3 — metric loaders may falsely label metric values non-metric

**Verdict: UPHELD — FIXED in 5ac844e.** Raised by R1.

Failure scenario: HAMMER converts a raw value of 2000 to 2.0 metres, but `is_metric=False` changes only the label and downstream masking/loss branch. The numerical data do not become non-metric.

Minimal patch, **NOT APPLIED**: in `config.py:103-143`, reject `is_metric is not True` for the five currently selectable metric dataset types, naming the dataset and flag. Add the equivalent fail-fast guard to the public constructors in `hammer.py`, `arkitscenes.py`, `arkitscenes_highres.py`, `scannet.py`, and `hypersim.py` so direct construction cannot create the contradiction. Do not rescale data or change loss behavior.

### R1-4 — `nneg` is documented as a count but used as a fraction

**Verdict: UPHELD — FIXED in a068dcd.** Raised by R1. This concerns `nneg` semantics; R2-4 below is the distinct negative-domain validation defect on the same fields.

Failure scenario: `target_n_corres=4,nneg=1` does not request one negative. The implementation initially computes zero positives as if `1` meant a 100% fraction, and its shortage fallback produced three negatives in the synthetic probe.

Minimal patch, **NOT APPLIED**: in `utils/corr.py:75-78`, compute the requested positive count as `target_n_corres - nneg` and the requested negative count as `nneg`, capped only by availability under the existing exact-total policy. In `config.py:129-130`, require `0 <= nneg <= n_corres` when correspondences are enabled.

### R1-5 — invalid depth pixels can become valid correspondences

**Verdict: UPHELD — FIXED in 57108be.** Raised by R1.

Failure scenario: two 2x2 all-zero point maps with all-false validity masks yield `[0,0] -> [0,0]` marked valid. Reciprocity alone mistakes the zero-depth sentinel for supervision.

Minimal patch, **NOT APPLIED**: in `utils/corr.py:54-68`, intersect reciprocal candidates with flattened `view1["valid_mask"]`, the source pixels' projected target validity, flattened `view2["valid_mask"]`, and the reverse-projected source validity before selecting positives. Keep deliberately sampled negatives marked false.

### R1-6 — resize dimensions and intrinsics use different scale factors

**Verdict: UPHELD — FIXED in 0213aa9.** Raised by R1.

Failure scenario: 640x480 to target 518x392 produces a 522x392 raster with actual scales `(0.815625,0.8166667)`, while both focal lengths use `0.8166667`; x geometry is inconsistent with the raster.

Minimal patch, **NOT APPLIED**: in `utils/cropping.py:81-96`, after flooring the output dimensions compute `actual_scaling = output_resolution / input_resolution` and pass it to `camera_matrix_of_crop`. In that helper at lines 109-118, normalize scaling to a two-element vector and multiply each of the first two intrinsic rows by its corresponding x/y scale. No resize dimensions or crop policy change.

### R1-7 — optional ARKit exclusion can permit overlap

**Verdict: REJECTED.** Raised by R1.

The supplied scenario gives the low-resolution loader no information from which it could infer the arbitrarily named `/data/arkit-hi` tree. `highres_root=None` explicitly documents the `_highres` convention and supports legitimate lowres-only installations; an explicit different root is required when composing both variants. The first-party training entrypoint does provide it. Making an optional unrelated dataset tree mandatory would reject valid standalone lowres use rather than fix loader correctness.

### R1-8 — aspect-ratio entries have a hidden 2:1 sampling weight

**Verdict: REJECTED.** Raised by R1.

The configuration promises that listed resolutions are enabled, not that they are uniformly sampled or order-insensitive. The sampler's weighting is deterministic, inherited training policy rather than numerical corruption. Changing it to uniform would silently alter established training distribution without an equal-weight contract or evidence that the policy is accidental; that merge risk is not justified.

### R2-1 — `drop_last=True` can produce a zero-batch loader

**Verdict: REJECTED.** Raised by R2 (contracts-and-runtime lens).

For a dataset smaller than one batch, zero complete batches is exactly the explicit and standard meaning of `drop_last=True`; the probe confirms the option is honored. Callers that need the partial batch can pass `drop_last=False`, and epoch-sized training datasets are expected to exceed a batch. A guard would make a valid sampler configuration fail or override requested behavior.

### R2-3 — variable-length sampling rejects accepted small `num_views` only at iteration

**Verdict: UPHELD — FIXED in 80a8455.** Raised by R2.

Failure scenario: `num_views=3,fixed_length=False` constructs a sampler, then iteration calls NumPy with bounds `(4,4)` and raises `ValueError: low >= high`.

Minimal patch, **NOT APPLIED**: in `base/easy_dataset.py:47-59`, before constructing `CustomRandomSampler`, raise `ValueError` naming `num_views` when `fixed_length` is false and `num_views < 4`. Values below four remain valid with `fixed_length=True`; do not narrow `DatasetConfig` globally.

### R2-4 — negative correspondence settings pass validation

**Verdict: UPHELD — FIXED in f79dd55.** Raised by R2. This is distinct from R1-4: even after count semantics are corrected, negative field values must be rejected.

Failure scenario: `n_corres=-100,nneg=0` passes validation, but the `n_corres > 0` branch is false and silently disables a manifest-declared feature.

Minimal patch, **NOT APPLIED**: in `config.py:129-130`, raise `ValueError` unless `n_corres >= 0` and `nneg >= 0`, naming the invalid field and value; retain the existing requirement that nonzero `nneg` requires correspondences. Mirror the nonnegative assertions at the direct `BaseMultiViewDataset` constructor boundary.

### R2-5 — `accelerator=None` is advertised but dereferenced

**Verdict: UPHELD — FIXED in b76f4f3.** Raised by R2.

Failure scenario: calling `get_data_loader` with its declared default raises `AttributeError: 'NoneType' object has no attribute 'num_processes'` during construction.

Minimal patch, **NOT APPLIED**: in `datasets/__init__.py:68`, pass `world_size=1 if accelerator is None else accelerator.num_processes`. This implements the existing standalone default without changing accelerator-backed callers.

### R2-6 — constructors do not exhaustively validate every frame asset

**Verdict: REJECTED.** Raised by R2.

Dataset item loading is intentionally lazy, and the design note's fail-fast promise explicitly concerns configuration fields and roots, not a complete up-front integrity scan of every large dataset. HAMMER already detects partial directory counts; requiring every loader to stat and cross-name every derived asset would add startup I/O and duplicate each loader's item-path logic. The examples are hypothetical corrupt trees, with no evidence of prevalence. A dedicated dataset-integrity tool could be useful, but constructor-wide validation is not a minimal merge-blocking fix.

### R2-7 — `is_metric` is not runtime type-validated

**Verdict: REJECTED.** Raised by R2.

`DatasetConfig.is_metric` is explicitly typed `bool`, and the supported Tyro path supplies a boolean. Direct Python callers passing the string `"false"` violate that API type contract; Python dataclass annotations do not generally imply runtime type enforcement. R1-3's real semantic problem is separately upheld, but adding piecemeal runtime validation solely for this typed field is unjustified.

### Reviewer questions (not findings)

- R1's physical-time versus frame-index question needs real sequence metadata and remains a question; no dataset tree was accessed.
- R1's alternate processed-depth-unit question likewise needs deployed dataset files and remains a question.
- R2's Accelerate sharding/equal-step question needs supported multi-process configuration evidence and remains a question; no GPU or distributed run was performed.
- R2's epoch-independent deterministic seed behavior does not contradict the stated per-sample determinism contract and remains a question.

### CPU claim probes

These commands used the required interpreter, synthetic in-memory inputs, and no GPU or dataset tree. They verify claims only and do not satisfy the unavailable before/after fix gate.

Probe 1:

```python
PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import numpy as np
from streamvggt.datasets.base.base_multiview_dataset import BaseMultiViewDataset, get_ray_map
from streamvggt.datasets.base.easy_dataset import EasyDataset
from streamvggt.datasets.utils.corr import extract_correspondences_from_pts3d

class Seq(BaseMultiViewDataset):
    def __len__(self): return 1
s = object.__new__(Seq)
s.stride_range = (2, 4)
s.regular_stride = False
s._short_scene_warnings = set()
rng = np.random.default_rng(0)
print('ray', get_ray_map(np.eye(4), np.array([[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1.]]), np.eye(3), 1, 1).tolist())
print('irregular', s.get_seq_from_start_id(4, 0, list(range(7)), rng)[0])
view = {'pts3d': np.zeros((2,2,3)), 'camera_intrinsics': np.eye(3), 'camera_pose': np.eye(4), 'valid_mask': np.zeros((2,2), dtype=bool)}
print('invalid_corr', tuple(x.tolist() for x in extract_correspondences_from_pts3d(view, view, 1, np.random.default_rng(0))))
print('nneg_one', tuple(x.tolist() for x in extract_correspondences_from_pts3d(view, view, 4, np.random.default_rng(0), nneg=1)))

class Tiny(EasyDataset):
    _resolutions = [(1,1), (1,1)]
    num_views = 3
    def __len__(self): return 3
    def __getitem__(self, idx): return idx

d = Tiny()
zero = d.make_sampler(4, drop_last=True, fixed_length=True)
zero.set_epoch(0)
print('drop_last', len(zero), list(zero))
variable = d.make_sampler(2, fixed_length=False)
variable.set_epoch(0)
try: print('variable', list(variable))
except Exception as e: print('variable', type(e).__name__, str(e))
PY
```

Full output:

```text
ray [[[1.0, 0.0, 0.0, 0.7071067811865475, 0.0, 0.7071067811865475]]]
irregular [0, 2, 4, 6]
invalid_corr ([[0, 0]], [[0, 0]], [True])
nneg_one ([[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1], [1, 0]], [True, False, False, False])
drop_last 0 []
variable ValueError low >= high
```

Probe 2:

```python
PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import numpy as np
from PIL import Image
from streamvggt.datasets import get_data_loader
from streamvggt.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from streamvggt.datasets.base.easy_dataset import EasyDataset
from streamvggt.datasets.utils.cropping import rescale_image_depthmap

class Seq(BaseMultiViewDataset):
    def __len__(self): return 1
s = object.__new__(Seq); s.stride_range=(3,3); s.regular_stride=False; s._short_scene_warnings=set()
print('irregular_fixed3', s.get_seq_from_start_id(4, 0, list(range(6)), np.random.default_rng(0))[0])
K=np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]])
im=Image.fromarray(np.zeros((480,640,3), dtype=np.uint8))
d=np.zeros((480,640), dtype=np.float32)
out_im,out_d,out_K=rescale_image_depthmap(im,d,K,(518,392))
print('resize_size', out_im.size, 'Kdiag', out_K[0,0], out_K[1,1], 'actual_scales', out_im.size[0]/640, out_im.size[1]/480)
class Tiny(EasyDataset):
    _resolutions=[(1,1)]; num_views=4
    def __len__(self): return 4
    def __getitem__(self, idx): return idx
try: get_data_loader(Tiny(), 1, num_workers=0)
except Exception as e: print('accelerator_none', type(e).__name__, str(e))
PY
```

Full output:

```text
irregular_fixed3 [0, 3, 4, 5]
resize_size (522, 392) Kdiag 408.33333833333336 408.33333833333336 actual_scales 0.815625 0.8166666666666667
accelerator_none AttributeError 'NoneType' object has no attribute 'num_processes'
```

## Area C — `src/eval/`

Role: Defense, no-commit mode.

### R1-1 — video `scale&shift` does not solve an exactly affine case

**Verdict: UPHELD — FIXED in 9b39ce6.** Raised by R1.

Failure scenario: the shipped video evaluator routes all three datasets' `scale&shift` mode through 1,000 fixed-step Adam iterations on summed L1. For prediction `[1,2,3]` and GT `[3,5,7]`, the live route reports AbsRel `0.0793723538517952`; the already-available least-squares route reports `0.0`. This mode is live, not merely a bad dormant default. `monodepth/tools.py` has the duplicate implementation, but its current evaluator does not select this advertised video mode.

Minimal patch, **NOT APPLIED**: in the three `args.align == "scale&shift"` calls in `video_depth/eval_depth.py:132-142,245-254,347-356`, pass `align_with_lstsq=True` instead of `align_with_lad2=True`. Do not alter the dormant duplicate.

### R1-2 and R2-1 — affine fitting includes `custom_mask` exclusions

**Verdict: UPHELD — FIXED in db38142.** Raised independently by R1 and R2; these are the same defect.

Failure scenario: prediction `[1,2,100]`, GT `[2,4,1]`, and mask `[true,true,false]` have an exact scale of 2 on the evaluated pixels, but the excluded outlier participates in the fit. The live validation function reports AbsRel `0.37813687324523926` and aligned valid depths `[3.0049467,2.9847984]`. `finetune_depth.py` uses this path for affine headline metrics and TAE input. The legacy tool copies expose the same callable bug, although current video/monodepth scripts do not pass `custom_mask`.

Minimal patch, **NOT APPLIED**: in `temporal_consistency/metrics.py:201-209`, intersect the GT-range mask with `custom_mask` after validating its shape and before extracting tensors or fitting. Remove the later second-stage indexing at lines 271-279. Apply the same ordering-only change to the public duplicate functions in `monodepth/tools.py` and `video_depth/tools.py`; change no alignment formula or default.

### R1-3 and R2 TAE-collision question — reprojection collisions use last-write wins

**Verdict: UPHELD — FIXED in 6005530.** R1 promoted this as a finding; R2's separate question describes the same behavior.

Failure scenario: two positive camera-space points projecting to one target pixel produce depth 2 in near/far input order and depth 1 in far/near order. TAE is intended to compare the visible reprojected surface, so choosing an occluded sample by flatten order is a real wrong-number path in the live validation implementation.

Minimal patch, **NOT APPLIED**: in `temporal_consistency/metrics.py:76-78`, initialize the warp depth to infinity, use `np.minimum.at` on the projected `(y,x)` indices to retain the nearest positive depth, then replace untouched infinities with zero before applying `warp_mask`.

### R1-4 and R2-6 — MV finiteness filtering destroys point tuples

**Verdict: UPHELD — FIXED in 5ba721c.** Raised independently by R1 and R2; these are the same defect.

Failure scenario: two points `[[0,0,1],[1,NaN,2]]` produce a component mask of shape `(2,3)` and five retained scalars; reshaping raises `ValueError`. Other component counts can silently regroup coordinates, and the computed GT mask is ignored.

Minimal patch, **NOT APPLIED**: in `mv_recon/launch.py:300-308`, compute one joint point mask as `np.isfinite(pts_all_masked).all(axis=-1) & np.isfinite(pts_gt_all_masked).all(axis=-1)`, then index prediction points, GT points, and their corresponding colors with that same one-dimensional mask. Remove the unused component masks.

### R1-5 — `avg_dis` normalization uses a batch-wide denominator

**Verdict: UPHELD — FIXED in 08e3951.** Raised by R1.

Failure scenario: two samples with one valid point at distance 2 and 4 return factors `[1,2]`, not `[2,4]`, because both numerators are divided by the batch count 2. The shipped MV launcher mitigates this by constructing `Regr3D_t(..., norm_mode=False)`, but `Regr3D_t` publicly defaults to the selectable broken `avg_dis`; dormancy does not make that default correct.

Minimal patch, **NOT APPLIED**: in `mv_recon/criterion.py:177`, replace the scalar `torch.cat(nnzs).sum()` denominator with `torch.stack(nnzs).sum(dim=0)`, preserving one valid count per batch sample for broadcasting against `all_dis.sum(dim=1)`.

### R1-6 and R2-11 — alignment modes are not mutually exclusive

**Verdict: UPHELD — FIXED in 294039c.** Raised independently by R1 and R2; these are the same defect.

Failure scenario: `metric_scale=True, scale_and_shift=True` silently takes the metric branch and reports AbsRel `0.5` for prediction half of GT instead of rejecting the contradictory request. Current in-repo callers select one mode, but this is a selectable public API trap with a self-evident fail-fast fix.

Minimal patch, **NOT APPLIED**: in `temporal_consistency/metrics.py:172-179`, require `sum((metric_scale, scale_and_shift, scale_only)) == 1` and raise `ValueError` otherwise. Make the equivalent exact-one check over the five booleans in `monodepth/tools.py` and `video_depth/tools.py`.

### R2-2 — no-overlap TAE clips are silently dropped

**Verdict: REJECTED.** Raised by R2.

With no reprojection correspondences, TAE has no defined numerical value. Returning `(nan,nan)` communicates that state; `finetune_depth.py` deliberately filters non-finite pair values rather than inventing a zero or infinity. The clip still contributes metrics that do have valid support. Adding a coverage log could be a reporting enhancement, but the protected wandb layout cannot be changed here and no existing denominator contract says every clip must contribute TAE. The NumPy warnings are noisy, not evidence that a wrong TAE number was reported.

### R2-3 — failed video sequences may be omitted from a dataset score

**Verdict: STILL DEFERRED — run the video-depth launcher on one CUDA GPU against
the real Sintel and KITTI layouts with a deterministic injected OOM before output
for one named sequence and one successful sequence; then run aggregation and
observe whether the headline score is emitted from only the survivor. Benchmark
policy must choose either a nonzero/fail-fast incomplete result or an explicitly
labelled partial score before code can change.** Raised by R2.

Failure scenario: the launcher catches selected OOM/covariance/eigenvalue failures, logs them, and can leave no prediction directory; Sintel and KITTI evaluation enumerate produced prediction groups, so a surviving subset can be averaged. Static flow establishes the possibility, but the launcher deliberately supports best-effort long evaluation and retains side logs. Establishing whether completeness is a benchmark requirement, and whether these exceptions occur before usable outputs, needs the prohibited GPU launcher and real dataset files. A fail-fast change versus an explicit partial-result manifest is a workflow decision larger than a safe unsupervised patch.

### R2-4 — scale-only alignment accepts an all-zero prediction

**Verdict: UPHELD — FIXED in 2e89c94.** Raised by R2.

Failure scenario: a zero 2x2 prediction against positive GT returns NaN AbsRel/SqRel/RMSE/Log RMSE while claiming four valid pixels. A diverged or undefined scale therefore reaches headline aggregation.

Minimal patch, **NOT APPLIED**: in `temporal_consistency/metrics.py:227`, before computing `s`, raise `ValueError` naming scale-only alignment when the selected prediction is non-finite or its absolute sum is zero. Add the same guard to the two public legacy tool copies. Do not provide a fallback scale.

### R2-5 — prediction and GT files are paired only by sort position

**Verdict: STILL DEFERRED — inventory real GT and generated prediction paths for
at least two frames from each of NYUv2, Sintel, Bonn, and KITTI on CPU, derive a
dataset-specific canonical `(sequence, frame)` key, and test equal counts, a
missing prediction, and a stale extra prediction; a key that bijectively matches
all four real layouts and rejects both mutations would settle the minimal mapping.** Raised by R2.

Failure scenario: equal-length lists containing one missing and one stale prediction can pair the wrong frames and still yield plausible metrics. The absence of any count/identity assertion is concerning, but the four datasets use different GT and generated prediction naming layouts. A correct minimal identity key cannot be derived safely without inspecting the prohibited real dataset/output trees; a basename-equality assertion would reject valid layouts. This needs dataset files or a larger explicit metadata mapping, so no speculative patch is proposed.

### R2-7 — CO3D reads undefined `args.fast_eval`

**Verdict: UPHELD — FIXED in c5f82cc.** Raised by R2.

Failure scenario: any loaded category reaches `args.fast_eval`, but AST inspection found `fast_eval` as the sole `args` attribute read without a parser definition. Evaluation fails with `AttributeError` before reporting the category.

Minimal patch, **NOT APPLIED**: in `pose_evaluation/test_co3d.py:206-216`, add `parser.add_argument('--fast_eval', action='store_true')`. The existing branch then works and the default preserves full evaluation.

### R2-8 — CO3D `--use_ba` is a silent no-op

**Verdict: UPHELD — FIXED in ebc8dea.** Raised by R2.

Failure scenario: AST inspection found zero loads of the `use_ba` parameter and no BA call in `process_sequence`; both CLI modes execute the same inference branch while one claims BA.

Minimal patch, **NOT APPLIED**: immediately after argument parsing in `pose_evaluation/test_co3d.py:309`, raise `NotImplementedError("--use_ba is not implemented")` when `args.use_ba` is true. This fails fast without inventing an unverified BA integration or changing the default path.

### R2-9 — MV log aggregation hard-codes eight ranks and breaks at gaps

**Verdict: UPHELD — FIXED in 6fd172d.** Raised by R2.

Failure scenario: rank 8 from a nine-process Accelerate run is never aggregated; a missing low-index log also makes `break` discard later ranks. The shell wrapper uses one process, but the Python launcher actively shards by Accelerate rank and therefore exposes this contract.

Minimal patch, **NOT APPLIED**: in `mv_recon/launch.py:428`, iterate `range(accelerator.num_processes)` rather than `range(8)` and change the missing-log `break` to `continue`, so every configured rank that produced a log is read.

### R2-10 — empty valid support is encoded as zero metrics

**Verdict: REJECTED.** Raised by R2.

The record is not presented as valid or perfect: `valid_pixels=0` explicitly identifies empty support, and mixed evaluation weights it by zero so it has no numerical influence. When every record is empty, NumPy raises because no weighted population exists; refusing to manufacture a dataset score is correct. Changing placeholder values to NaN would not improve the weighted contract and could contaminate otherwise valid aggregation.

### R2-12 — `delta < 1.` cannot recognize an exact prediction

**Verdict: UPHELD — FIXED in cf9a799.** Raised by R2.

Failure scenario: an exact positive prediction has ratio exactly 1, yet strict `< 1.0` reports `delta < 1. = 0.0` while every conventional delta field is 1.0. The logged key is protected and need not be renamed.

Minimal patch, **NOT APPLIED**: change only the threshold-zero comparison from `< 1.0` to `<= 1.0` in `temporal_consistency/metrics.py:302`, `monodepth/tools.py:308`, and `video_depth/tools.py:308`. Leave all key names and other thresholds unchanged.

### R2-13 — launcher flags are ignored or overridden

**Verdict: UPHELD — FIXED in b12dd98, limited to `--conf_thresh`.** Raised by R2.

Failure scenario: `mv_recon/launch.py` advertises a confidence threshold, never reads it, and uses a commented hard-coded confidence comparison. A user-selected threshold therefore has no effect. The device and video crop observations are real ergonomics, but current launchers are explicitly CUDA-oriented and unconditionally choose the shipped no-crop flow; changing those paths is not needed to protect reported numbers and is rejected as unnecessary churn.

Minimal patch, **NOT APPLIED**: remove the unused `--conf_thresh` parser argument from `mv_recon/launch.py:32-34` and remove the stale commented hard-coded threshold at line 252. This stops advertising unsupported behavior without silently changing which points the default evaluation includes.

### R2-14 — one-image NRGBD input produces a zero slice step

**Verdict: REJECTED.** Raised by R2.

NRGBD is used here for multi-view reconstruction, so a one-image scene violates the input contract. The zero slice step raises before any metric is produced; it cannot silently corrupt a score. Replacing this with a bespoke error message would be nicer diagnostics, but it is not worth a merge-risk patch for a malformed, unsupported scene.

### Reviewer questions (not findings)

- R1's global-valid-pixel versus equal-sequence aggregation question is not promoted: all three evaluators consistently weight by `valid_pixels`, but no benchmark contract in Area C establishes a different population.
- R1's symmetric TAE denominator question is not promoted: equal weighting of the two directions is a coherent meaning of “symmetric,” and no correspondence-union contract is documented.
- R2's TAE-collision question is the same behavior as upheld R1-3 and is jointly resolved there.
- R2's `metric.json` naming question is not promoted: the filename can naturally mean “metrics,” and no consumer treats it as a metric-scale claim.
- R2's always-truthy `if model_name == "stream3r" or "VGGT"` question remains harmless for the parser's currently reachable StreamVGGT/VGGT models, both of which require the body.

No finding was deferred because a patch needed an explanatory comment essay. R2-3 was deferred for prohibited GPU/real-data workflow evidence and a larger completeness-policy decision; R2-5 was deferred for real filename-contract evidence or a larger dataset-specific mapping.

### CPU claim probes

These commands used the required interpreter, `PYTHONPATH=src`, synthetic in-memory values, and no GPU, dataset, or checkpoint. `PYTHONDONTWRITEBYTECODE=1` prevented these probes from creating cache files; no probe-created `__pycache__` file required removal. Pre-existing ignored cache files were observed and left untouched. These probes verify claims only and do not satisfy the unavailable before/after fix gate.

Probe 1:

```python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import ast
import importlib.util
import pathlib
import warnings
import numpy as np
import torch

def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

video = load('video_tools_probe', 'src/eval/video_depth/tools.py')
temporal = load('temporal_metrics_probe', 'src/eval/temporal_consistency/metrics.py')
print('torch', torch.__version__, 'cuda_used', False)
pred = np.array([[1., 2., 3.]], dtype=np.float32)
gt = 2 * pred + 1
lad, *_ = video.depth_evaluation(pred, gt, max_depth=None, align_with_lad2=True, use_gpu=False)
lstsq, *_ = video.depth_evaluation(pred, gt, max_depth=None, align_with_lstsq=True, use_gpu=False)
print('video_affine_lad_absrel', lad['Abs Rel'])
print('video_affine_lstsq_absrel', lstsq['Abs Rel'])

pred = np.array([[1., 2., 100.]], dtype=np.float32)
gt = np.array([[2., 4., 1.]], dtype=np.float32)
custom = np.array([[True, True, False]])
masked_fit, _, aligned, _ = temporal.depth_evaluation(pred, gt, max_depth=None, custom_mask=custom, scale_and_shift=True)
print('custom_mask_affine_absrel', masked_fit['Abs Rel'])
print('custom_mask_affine_aligned_valid', aligned[custom].tolist())

identity = np.eye(4, dtype=np.float32)
points = np.array([[0., 0., 1.], [0., 0., 2.]], dtype=np.float32)
wmask = np.ones((1, 1), dtype=bool)
print('collision_near_far', temporal.point2depth(points, wmask, identity).tolist())
print('collision_far_near', temporal.point2depth(points[::-1], wmask, identity).tolist())
depth_a = np.array([[1., 0.], [0., 0.]], dtype=np.float32)
depth_b = np.array([[0., 0.], [0., 1.]], dtype=np.float32)
mask_a, mask_b = depth_a > 0, depth_b > 0
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    print('tae_no_overlap', temporal.tae(depth_a, mask_a, identity, depth_b, mask_b, identity))
    print('tae_no_overlap_warning_count', len(caught))

zero = np.zeros((2, 2), dtype=np.float32)
pos = np.array([[2., 4.], [6., 8.]], dtype=np.float32)
scale_zero, *_ = temporal.depth_evaluation(zero, pos, max_depth=None, scale_only=True)
print('scale_only_zero', scale_zero)
both, *_ = temporal.depth_evaluation(np.ones((1, 2), dtype=np.float32), np.full((1, 2), 2., dtype=np.float32), max_depth=None, metric_scale=True, scale_and_shift=True)
print('multiple_modes_absrel', both['Abs Rel'])
perfect, *_ = temporal.depth_evaluation(np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32), max_depth=None, metric_scale=True)
print('perfect_delta_fields', {k: v for k, v in perfect.items() if k.startswith('delta')})
empty, *_ = temporal.depth_evaluation(np.ones((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32), max_depth=None, metric_scale=True)
print('empty_support', empty)
try:
    np.average([empty['Abs Rel']], weights=[empty['valid_pixels']])
except Exception as exc:
    print('empty_weight_average', type(exc).__name__, str(exc))

pts = np.array([[0., 0., 1.], [1., np.nan, 2.]])
component_mask = np.isfinite(pts)
print('mv_component_mask_shape', component_mask.shape, 'filtered_shape', pts[component_mask].shape)
try:
    pts[component_mask].reshape(-1, 3)
except Exception as exc:
    print('mv_component_reshape', type(exc).__name__, str(exc))

source = pathlib.Path('src/eval/pose_evaluation/test_co3d.py').read_text()
tree = ast.parse(source)
parser_defs = {call.args[0].value.lstrip('-').replace('-', '_') for call in ast.walk(tree) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument' and call.args and isinstance(call.args[0], ast.Constant)}
args_reads = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'args'}
print('co3d_args_read_not_defined', sorted(args_reads - parser_defs))
proc = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process_sequence')
print('process_sequence_use_ba_loads', sum(isinstance(node, ast.Name) and node.id == 'use_ba' and isinstance(node.ctx, ast.Load) for node in ast.walk(proc)))
print('process_sequence_ba_calls', [node.func.id for node in ast.walk(proc) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and 'ba' in node.func.id.lower()])
PY
```

Full output:

```text
torch 2.3.1+cu121 cuda_used False
video_affine_lad_absrel 0.0793723538517952
video_affine_lstsq_absrel 0.0
custom_mask_affine_absrel 0.37813687324523926
custom_mask_affine_aligned_valid [3.004946708679199, 2.9847984313964844]
collision_near_far [[2.0]]
collision_far_near [[1.0]]
tae_no_overlap (nan, nan)
tae_no_overlap_warning_count 8
scale_only_zero {'Abs Rel': nan, 'Sq Rel': nan, 'RMSE': nan, 'Log RMSE': nan, 'delta < 1.': 0.0, 'delta < 1.25': 0.0, 'delta < 1.25^2': 0.0, 'delta < 1.25^3': 0.0, 'valid_pixels': 4}
multiple_modes_absrel 0.5
perfect_delta_fields {'delta < 1.': 0.0, 'delta < 1.25': 1.0, 'delta < 1.25^2': 1.0, 'delta < 1.25^3': 1.0}
empty_support {'Abs Rel': 0, 'Sq Rel': 0, 'RMSE': 0, 'Log RMSE': 0, 'delta < 1.': 0, 'delta < 1.25': 0, 'delta < 1.25^2': 0, 'delta < 1.25^3': 0, 'valid_pixels': 0}
empty_weight_average ZeroDivisionError Weights sum to zero, can't be normalized
mv_component_mask_shape (2, 3) filtered_shape (5,)
mv_component_reshape ValueError cannot reshape array of size 5 into shape (3)
co3d_args_read_not_defined ['fast_eval']
process_sequence_use_ba_loads 0
process_sequence_ba_calls []
```

Probe 2 (the first call supplied only one list element, exposing the function's two-slot input assumption before reaching the reviewed arithmetic):

```python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import torch
from eval.mv_recon.criterion import get_norm_factor

pts = [torch.tensor([[[[2., 0., 0.]]], [[[4., 0., 0.]]]])]
valids = [torch.ones((2, 1, 1), dtype=torch.bool)]
print('avg_dis_two_samples', get_norm_factor(pts, 'avg_dis', valids, fix_first=True).reshape(-1).tolist())
PY
```

Full output:

```text
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/oscar/home/jdosch/MeTRIC/src/eval/mv_recon/criterion.py", line 151, in get_norm_factor
    assert pts[1] is None or (pts[1].ndim >= 3 and pts[1].shape[-1] == 3)
           ~~~^^^
IndexError: list index out of range
```

Corrected probe 2:

```python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import torch
from eval.mv_recon.criterion import get_norm_factor

pts = [torch.tensor([[[[2., 0., 0.]]], [[[4., 0., 0.]]]]), None]
valids = [torch.ones((2, 1, 1), dtype=torch.bool), None]
print('avg_dis_two_samples', get_norm_factor(pts, 'avg_dis', valids, fix_first=True).reshape(-1).tolist())
PY
```

Full output:

```text
avg_dis_two_samples [1.0, 2.0]
```

Probe 3 verified the denominator tensor shape used in the proposed R1-5 patch:

```python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
import torch
from dust3r.utils.misc import invalid_to_zeros
pt = torch.tensor([[[[2.,0.,0.]]], [[[4.,0.,0.]]]])
valid = torch.ones((2,1,1), dtype=torch.bool)
_, nnz = invalid_to_zeros(pt, valid, ndim=3)
print('nnz_shape', tuple(nnz.shape), 'nnz', nnz.tolist())
PY
```

Full output:

```text
nnz_shape (2,) nnz [1, 1]
```

## Area D — training and validation entrypoints

Role: Defense, no-commit mode. Files reviewed: `src/finetune_depth.py`,
`src/finetune.py`, `src/train.py`, `src/train_utils.py`, and
`src/val_images.py`.

Legacy-path decision: `src/finetune.py` and `src/train.py` are not supported
MeTRIC entrypoints. Both retain the CUT3R training-code header, the implementation
notes identify `finetune_depth.py` as the new MeTRIC entrypoint, and every repository
experiment launcher found invokes `finetune_depth.py`; none invokes either legacy
file. Defects that also affect the live depth trainer are judged on that live impact.
Findings whose only victim is the unsupported legacy pair are rejected because a
patch would add merge risk to a path the project does not run.

### R1-1 and R2-4 — resume can rewrite the owning run with a different identity

**Verdict: STILL DEFERRED — approve and CPU-test a synthetic owning-manifest
matrix with `epochs=10,lr=1e-5` against current `(20,1e-5)`, `(10,1e-4)`, and
`(5,1e-5)`, specifying which cases may continue and whether an accepted extension
retains the owner's manifest/hash/id or records continuation metadata; the guard
must reject every non-sanctioned case before manifest write.** Raised independently by R1 (wrong-numbers lens) and R2
(contracts-and-runtime lens); these are the same defect.

Failure scenario: changing `lr` from `1e-5` to `1e-4` changes the experiment ID
from `2e492d2cafc125e9` to `a193e68d312f1943`, but both resume configurations
resolve to the checkpoint's parent. `run()` then writes the current manifest there
before loading the checkpoint, so a structurally compatible but result-affecting
configuration can replace the owner's recorded identity and continue updating its
checkpoints.

Reason deferred: the mechanism is real, but `resolve_output_dir` explicitly promises
that increasing `epochs` extends the owning run, while `epochs` is frozen as an
identity field. Rejecting every manifest difference would silently revoke that
documented continuation behavior; allowing selected differences requires an explicit
lineage policy that this review does not define. No `FinetuneDepthCfg` field or
`_NON_IDENTITY_FIELDS` entry may change. This joint finding is explicitly deferred
because a safe patch would need an explanatory policy essay rather than a
self-evident minimal correction. With a decided policy, the fix site is editable
`src/finetune_depth.py`, before the manifest write: read the owning `manifest.json`,
compare the permitted identity fields, fail fast on prohibited drift, and use the
owner's stored experiment ID for a sanctioned continuation.

### R1-2 and R2-5 — mid-epoch resume replays the epoch from batch zero

**Verdict: FIXED in a88f5f8.** Raised independently by R1 and R2; these are the same
defect.

Failure scenario: a checkpoint saved after batch 600 stores `epoch-1` and
`step=600`. The shared loader restores the current epoch and `start_step=600`, but
the live and legacy loops enumerate their loaders from zero. The already-updated
model therefore sees the beginning of that epoch again.

Implemented patch: immediately after checkpoint load in `src/finetune_depth.py`,
reject a restored nonzero `start_step` because sampler and RNG state are not present.
Epoch-boundary checkpoints (`start_step=0`) retain their existing resume behavior;
an unsafe mid-epoch artifact now fails before training rather than replaying data.

### R1-3 — `val/all/*_med` is rank-local

**Verdict: UPHELD — FIXED in 2160b1c.** Raised by R1.

Failure scenario: rank 0's loss deque contains `[0,0]` and rank 1's contains
`[100,100,100]`. `SmoothedValue.synchronize_between_processes` does not pool the
deques, so the main rank emits `val/all/loss_med=0` although the global median is
100. The source docstring acknowledges that the median is rank-local.

Implemented patch: `_log_val_stats` gathers the supported scalar observations
from every rank and computes their genuine global median. The frozen
`val/all/loss_med` key remains present.

This is distinct from R2-1: R1-3 concerns the local deque used for medians; R2-1
concerns discarded returned tensors used for averages.

### R1-4 — validation loss averages batch means rather than clips

**Verdict: UPHELD — FIXED in f941f65.** Raised by R1.

Failure scenario: a four-clip batch with loss 1 followed by a one-clip batch with
loss 9 reports `(1+9)/2=5.0`; equal clip support is `2.6`. This affects per-dataset
and blended `loss_avg`, including the checkpoint-selection fallback.

Implemented patch: in `src/finetune_depth.py`, make
`_accumulate_batch_loss` multiply each scalar batch value by the batch clip count
and increment its count by that clip count. In `val_loop`, retain the reduced
`loss_blended` result and pass it to `_log_val_stats`, so the existing frozen
`val/<dataset>/<metric>_avg`, `val/all/<metric>_avg`, and flat returned selection
names carry clip-weighted numerators and denominators. Do not change key names or
the criterion.

### R1-5 — streaming TAE bridges across an invalid frame

**Verdict: UPHELD — FIXED in c3da4b7.** Raised by R1.

Failure scenario: frames 0 and 2 are valid, frame 1 is invalid, and predictions are
1, 2, 3. The loop retains frame 0 across the gap and emits TAE 2 and TAE-squared 4,
although there is no adjacent valid pair.

Implemented patch: in `src/finetune_depth.py`, in
`_streaming_depth_metrics`, set `prev = None` immediately before continuing on an
invalid frame. No metric definition or key changes.

### R1-6 — accumulation-boundary logging shows the current microbatch

**Verdict: REJECTED.** Raised by R1.

The logged scalar is a valid instantaneous microbatch loss sampled at a synchronization
boundary; neither the key `train/loss` nor its surrounding code claims it is the mean
loss of the gradient-accumulation window. Gradient construction is unaffected, and
the epoch-end meters separately average every microbatch. Turning this telemetry into
a window mean would introduce state and a new semantic contract solely to satisfy an
unstated interpretation, so it does not justify churn.

### R1-7 and R2-10 — legacy final artifacts do not satisfy the resume schema

**Verdict: REJECTED.** Raised independently by R1 and R2; these are the same
legacy-only defect.

The schema mismatch is statically real: the two final writers omit `optimizer`, and
the shared loader indexes that key. It affects only unsupported CUT3R-derived
`finetune.py` and `train.py`; the live MeTRIC trainer writes resumable last/best
artifacts through `misc.save_model` and does not create this reduced final schema.
Adding a second legacy load mode or expanding an unused artifact is unnecessary
merge risk.

### R2-1 — distributed reductions discard their returned tensors

**Verdict: UPHELD — FIXED in ebe25d7.** Raised by R2.

Failure scenario: a fake two-rank reducer returned global sum/count `[13,4]`, but
`_reduce_metrics` read the unchanged local `[10,1]` and returned 10. The installed
Accelerate API returns the reduced tensor; it does not promise to mutate the input.
The same discarded-return pattern exists in the vendored meter synchronization, so
loss averages are also rank-local unless the repo-owned caller supplies its own
reduced blend. This can select a checkpoint using rank 0's shard rather than global
validation.

Implemented patch: in `src/finetune_depth.py`, assign
`t = accelerator.reduce(t, reduction="sum")` in `_reduce_metrics`. In `val_loop`,
retain both values from the existing loss reduction and merge `loss_blended` with
`depth_blended` before `_log_val_stats`; that repo-owned result overwrites the unsafe
vendored meter average under the existing frozen keys. Do not edit
`src/croco/utils/misc.py`.

This is distinct from R1-3: correct global sums/counts fix averages, but do not pool
the local observation deque from which median is computed.

### R2-2 — `even_batches=False` allegedly creates unequal training lengths

**Verdict: REJECTED.** Raised by R2.

The proposed five-batch/two-rank hang does not match these loaders. Every Area D
training loader is constructed with `drop_last=True`. Accelerate 1.14's
`BatchSamplerShard._iter_with_no_split` yields a group only after every process has a
batch when `drop_last` is true, dropping the unmatched fifth batch. A CPU probe of
exactly five raw batches produced `(len, iterated) = (2,2)` for both ranks even with
`even_batches=False`. Validation loaders may be uneven, but they perform no backward
collective per batch and reduce only after every rank completes its shard.

### R2-3 — rank-local non-finite loss exits before peers reach backward

**Verdict: UPHELD — FIXED in 19e7a88.** Raised by R2.

Failure scenario: one live depth-training rank gets NaN and executes `sys.exit(1)`;
another gets a finite loss and enters DDP backward, where it can wait until the long
process-group timeout. A genuine multi-rank run was not permitted, but the divergent
branch before the collective is explicit.

Implemented patch: in `src/finetune_depth.py`, immediately
after computing `loss_value`, reduce a scalar finite flag with `min`, assign the
returned tensor, and raise the same `FloatingPointError` on every rank before any
rank calls the scaler/backward path. Include local loss details only on the offending
rank. No vendored edit is needed.

### R2-6 — fresh nonzero `start_epoch` changes work without changing identity

**Verdict: UPHELD — FIXED in 26caa22.** Raised by R2.

Failure scenario: fresh `start_epoch=0` and `start_epoch=9` configurations produce
the same experiment ID, yet the latter trains only the final epoch. The field is
correctly frozen as resume bookkeeping, but no guard prevents it from controlling a
fresh run.

Implemented patch: in `src/finetune_depth.py::main`, before
building the manifest, raise `ValueError` when `cfg.resume is None` and
`cfg.start_epoch != 0`. Leave every `FinetuneDepthCfg` field and
`_NON_IDENTITY_FIELDS` unchanged; checkpoint loading remains the only sanctioned
source of a nonzero start epoch.

### R2-7 — live checkpoint replacement is not atomic

**Verdict: UPHELD — FIXED in 08a0f49.** Raised by R2.

Failure scenario: writing a new `checkpoint-last.pth` directly over the old target
and being preempted mid-write can leave only a truncated resume artifact. The unsafe
primitive is vendored, but the live call site is editable.

Implemented patch: in the nested `save_model` function in
`src/finetune_depth.py`, ask `misc.save_model` to write a sibling temporary filename,
then on the main process call `os.replace` from that fully closed temporary file to
`checkpoint-<fname>.pth`, followed by the existing process synchronization. This
keeps the previous target intact until atomic replacement and does not edit
`src/croco/`. The unsupported legacy-only final writer is not part of this patch.

### R2-8 — resume creates a new wandb run

**Verdict: FIXED in b2e7d71.** Raised by R2.

Failure scenario: tracker initialization supplies a display name but no stable wandb
`id` and resume policy, so a same-config checkpoint continuation can fork online
history. The legacy pair additionally discovers automatic resume after tracker init,
but those paths are unsupported.

Implemented patch: tracker initialization derives a deterministic wandb ID from both
the experiment group and run ID and uses `resume="allow"`. Same-group continuations
reconnect, different unhashed groups cannot collide, and checkpoints predating stable
IDs can establish the deterministic history on their next launch.

### R2-9 — legacy validation and best selection are no-ops

**Verdict: REJECTED.** Raised by R2.

The dormant validation configuration is inherited by unsupported CUT3R-derived
entrypoints; no MeTRIC launcher invokes them. The live `finetune_depth.py` executes
validation, logs the frozen sections, and performs best-checkpoint selection.
Reactivating old evaluation code would be a substantial unsupported-path change, not
a minimal MeTRIC fix.

### R2-11 — legacy save cadence can round to zero

**Verdict: REJECTED.** Raised by R2.

The arithmetic is real and the probe produced `ZeroDivisionError` for all three
examples, but it exists only in the unsupported legacy pair. The live depth trainer
already computes `save_every` and checks `save_every > 0` before modulo. Porting that
guard solely into unused trainers is unnecessary churn.

### R2-12 — fresh-run collision check is a race

**Verdict: UPHELD — FIXED in f540818.** Raised by R2.

Failure scenario: two identical jobs can both pass `os.path.exists` before either
later calls `mkdir(..., exist_ok=True)`, then share manifest and checkpoint paths.
This is a standard check-then-create race and needs no concurrent GPU run to establish.

Implemented patch: in `src/train_utils.py::resolve_output_dir`,
replace the rank-zero fresh-run existence check with an atomic ownership claim using
`os.makedirs(output_dir, exist_ok=False)` and translate `FileExistsError` to the
current precise collision message. Nonzero ranks still only derive the path; the
later `run()` mkdir remains idempotent for ranks in the one claimed job. Resume must
not claim a new directory.

### R2-13 — empty loaders fail in vendored progress reporting

**Verdict: UPHELD — FIXED in 9c642cb.** Raised by R2.

Failure scenario: a dataset shorter than the training batch size with
`drop_last=True`, or an empty validation dataset, gives a zero-length loader.
Vendored `MetricLogger.log_every` then divides elapsed time by zero; the error does
not identify the bad dataset/configuration.

Implemented patch: in `src/finetune_depth.py::run`, immediately
after constructing the raw training, validation, and streaming loaders and before
model construction/preparation, raise `ValueError` naming any loader whose length is
zero. This is the preferred fail-fast contract and avoids modifying vendored
`src/croco/utils/misc.py`.

### Area D CPU evidence

No GPU, entrypoint, real dataset, or real checkpoint was used. The focused synthetic
probe command was:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/task-01c-mpl-probe /users/jdosch/miniconda3/envs/StreamVGGT/bin/python - <<'PY'
from collections import defaultdict
import torch
from torch.utils.data import BatchSampler, SequentialSampler
from accelerate import PartialState
from accelerate.data_loader import BatchSamplerShard
import croco.utils.misc as misc
import finetune_depth as fd
from streamvggt.depth_cond.config import experiment_id
PartialState()
class ReturningReducer:
    num_processes = 2
    device = torch.device('cpu')
    is_main_process = True
    trackers = []
    def wait_for_everyone(self): pass
    def reduce(self, tensor, reduction='sum'):
        return torch.tensor([[13.0, 4.0]], dtype=tensor.dtype)
fd.gather_object = lambda parts: [parts[0], parts[0]]
per_ds, blended = fd._reduce_metrics({'hammer/absrel_metric': 10.0}, {'hammer/absrel_metric': 1}, ReturningReducer())
print('reduce_metrics_return_discarded', per_ds, blended)
meter = misc.SmoothedValue(); meter.update(10.0)
meter.synchronize_between_processes(ReturningReducer())
print('smoothed_reduce_return_discarded', {'count': meter.count, 'total': meter.total, 'global_avg': meter.global_avg})
def views(batch):
    return [{'img': torch.zeros(batch, 3, 1, 1), 'dataset': ['hammer'] * batch}]
sums, counts = defaultdict(float), defaultdict(int)
fd._accumulate_batch_loss(views(4), 1.0, {}, sums, counts)
fd._accumulate_batch_loss(views(1), 9.0, {}, sums, counts)
print('batch_mean_loss', sums['hammer/loss'] / counts['hammer/loss'], 'clip_weighted', (4*1.0+1*9.0)/5)
old_eval, old_tae = fd.depth_evaluation, fd.tae
fd.depth_evaluation = lambda *a, **k: ({'Abs Rel': 0.0, 'delta < 1.25': 1.0, 'RMSE': 0.0}, None, None, None)
fd.tae = lambda p1, m1, x1, p2, m2, x2: (abs(float(p2.item()-p1.item())), float((p2.item()-p1.item())**2))
K = torch.eye(3).reshape(1,3,3); pose = torch.eye(4).reshape(1,4,4)
views3, preds3 = [], []
for i, valid in enumerate((True, False, True), start=1):
    views3.append({'depthmap': torch.ones(1,1,1), 'valid_mask': torch.tensor([[[valid]]]), 'camera_intrinsics': K, 'camera_pose': pose, 'dataset': ['hammer']})
    preds3.append({'depth': torch.tensor([[[[float(i)]]]])})
print('streaming_gap_metrics', fd._streaming_depth_metrics(views3, preds3))
fd.depth_evaluation, fd.tae = old_eval, old_tae
cfg_a = fd.FinetuneDepthCfg(resume='/oscar/home/jdosch/MeTRIC/fake_run/checkpoint-last.pth', lr=1e-5)
cfg_b = fd.FinetuneDepthCfg(resume='/oscar/home/jdosch/MeTRIC/fake_run/checkpoint-last.pth', lr=1e-4)
id_a, id_b = experiment_id(fd.build_manifest(cfg_a)), experiment_id(fd.build_manifest(cfg_b))
print('resume_identity_drift', id_a, id_b, fd.resolve_output_dir(cfg_a, id_a), fd.resolve_output_dir(cfg_b, id_b))
cfg_0 = fd.FinetuneDepthCfg(start_epoch=0); cfg_9 = fd.FinetuneDepthCfg(start_epoch=9)
print('fresh_start_epoch_identity', experiment_id(fd.build_manifest(cfg_0)), experiment_id(fd.build_manifest(cfg_9)))
raw = BatchSampler(SequentialSampler(range(10)), batch_size=2, drop_last=True)
lengths = []
for rank in (0, 1):
    shard = BatchSamplerShard(raw, num_processes=2, process_index=rank, split_batches=False, even_batches=False)
    lengths.append((len(shard), len(list(shard))))
print('five_train_batches_two_ranks', lengths)
for save_freq, n in ((0,100), (0.001,100), (0.1,5)):
    interval = int(save_freq*n)
    try: result = 0 % interval
    except Exception as exc: result = f'{type(exc).__name__}: {exc}'
    print('legacy_save_interval', save_freq, n, interval, result)
PY
```

Full output:

```text
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
reduce_metrics_return_discarded {'hammer': {'absrel_metric': 10.0}} {'absrel_metric': 10.0}
smoothed_reduce_return_discarded {'count': 1, 'total': 10.0, 'global_avg': 10.0}
batch_mean_loss 5.0 clip_weighted 2.6
streaming_gap_metrics {'hammer/absrel_affine': [0.0], 'hammer/delta1_affine': [1.0], 'hammer/rmse_affine': [0.0], 'hammer/absrel_metric': [0.0], 'hammer/delta1_metric': [1.0], 'hammer/rmse_metric': [0.0], 'hammer/tae': [2.0], 'hammer/tae_sq': [4.0]}
resume_identity_drift 2e492d2cafc125e9 a193e68d312f1943 /oscar/home/jdosch/MeTRIC/fake_run /oscar/home/jdosch/MeTRIC/fake_run
fresh_start_epoch_identity 2e492d2cafc125e9 2e492d2cafc125e9
five_train_batches_two_ranks [(2, 2), (2, 2)]
legacy_save_interval 0 100 0 ZeroDivisionError: integer modulo by zero
legacy_save_interval 0.001 100 0 ZeroDivisionError: integer modulo by zero
legacy_save_interval 0.1 5 0 ZeroDivisionError: integer modulo by zero
```

An initial setup attempt used the same probe body but imported
`experiment_id` from nonexistent `streamvggt.config`; it reached no assertions.
Its full output was:

```text
/oscar/home/jdosch/.config/matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /tmp/matplotlib-sdfvjbse because there was an issue with the default path ({configdir}); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
ModuleNotFoundError: No module named 'streamvggt.config'
```

The three plan-mandated CPU tests were rerun independently.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/task-01c-mpl-val-images /users/jdosch/miniconda3/envs/StreamVGGT/bin/python tests/val_images_smoke.py
```

Full output:

```text
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
val_images smoke test
  stratification: 2 datasets -> ['hammer', 'scannet', 'hammer', 'scannet', 'hammer']; 3 datasets -> {'a': 2, 'b': 2, 'c': 1}
  budget: cap 5 honoured; single-dataset shard spends all 5; 0 disables
  determinism: identical selections across two passes
  sampling: 3 distinct clips, frame 0 only, [3,H,W]/[H,W] slices
  render: GT hole gray, prediction dense, shared scale -> /tmp/val_panels/panel_scannet.png (39176B)
  _log_val_images: 1 row 'val/samples' @ step 1234, 3 figs closed
PASS
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/task-01c-mpl-metric-sections /users/jdosch/miniconda3/envs/StreamVGGT/bin/python tests/val_metric_sections.py
```

Full output:

```text
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
val metric sections
  build_val_loaders: one loader per dataset; single-dataset delegates
  _accumulate_batch_loss: per-dataset attribution, mixed batch raises
  _reduce_metrics: levels kept apart, count-weighted blend, empties dropped
  _log_val_stats: ['val/all/absrel_metric_avg', 'val/all/loss_avg', 'val/all/loss_med', 'val/hammer/absrel_metric_avg', 'val/hammer/loss_avg', 'val/scannet/absrel_metric_avg', 'val/scannet/loss_avg']
  _log_val_stats: one group per dataset + all/, flat names still returned
PASS
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/task-01c-mpl-multi-loader /users/jdosch/miniconda3/envs/StreamVGGT/bin/python tests/val_loop_multi_loader.py
```

Full output:

```text
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
val_loop multi-loader
  two loaders -> 1 row, groups ['all', 'hammer', 'scannet'], blended loss 1.8
Building GLB scene
Using Depthmap and Camera Branch
GLB Scene built
Building GLB scene
Using Depthmap and Camera Branch
GLB Scene built
  coupled selection: panels ['hammer clip0 frame0 (epoch 2)', 'scannet clip1 frame0 (epoch 2)', 'hammer clip2 frame0 (epoch 2)']; glbs ['val_clip0_hammer.glb', 'val_clip1_scannet.glb']
  bare DataLoader rejected by both val_loop and streaming_eval
PASS
```

All focused probes and tests used `PYTHONDONTWRITEBYTECODE=1`. A final search
nevertheless found three cycle-created files:
`src/__pycache__/finetune_depth.cpython-311.pyc`,
`src/__pycache__/val_images.cpython-311.pyc`, and
`src/__pycache__/visualize_depth.cpython-311.pyc`. Exactly those three files were
removed. The `src/__pycache__` directory could not and was not removed because it
pre-existed with older bytecode files, which were left untouched.

## Area E — `src/streamvggt/depth_cond/`

Role: Defense, report-only round. No source fix was applied. Area E is project-owned
code; none of its files is under the vendored `src/croco/`, `src/dust3r/`,
`src/vggt/`, or `cloud_opt/` trees.

### E-R1-1 — cache-enabled AMP changes patch-encoder arithmetic

**Verdict: UPHELD — FIXED in 9960a31.** Raised by R1
(wrong-numbers lens).

Failure scenario: with outer autocast enabled, the normal live path runs the frozen
patch encoder under autocast, while a cache miss explicitly disables autocast and
the cache stores fp32; a warm hit also returns fp32. Enabling a feature described as
a numerically identical optimization therefore changes patch tokens, attention,
losses, and gradients. R1's CPU analogue measured a maximum token difference of
`0.0228300`; a GPU is needed only to quantify the CUDA/bf16 magnitude, not to
establish the divergent execution modes.

Resolution: retain stable fp32 cache computation and storage, and correct the cache
module and call-site contract so they no longer claim numerical identity with an
autocast live path. The CPU regression records the intentional dtype contract and
the documentation now discloses the possible numerical difference.

### E-R1-2 and E-R2-2 — cache identity omits encoder, checkpoint, and schema

**Verdict: UPHELD — FIXED in 7e49385.** Raised independently
by R1 (wrong-numbers lens) and R2 (contracts-and-runtime lens); these are the same
defect.

Failure scenario: checkpoint A and checkpoint B use the same cache directory and raw
processed-frame key. `cache.py` hashes only that raw key and stores a bare tensor, so
B accepts A's same-shaped token tensor and bypasses B's encoder without warning.
Changing sparse-depth conditioning alone does not stale this RGB-only cache; the
missing ownership is specifically encoder weights/architecture and cache schema.

Resolution: `EncoderFeatureCache` requires a schema-versioned checkpoint fingerprint
and includes it in every path digest. `load_pretrained` supplies the checkpoint
SHA-256 only after strict loading succeeds; changing the checkpoint therefore makes
existing entries misses, while sparse-depth inputs remain outside the cache identity.

### E-R1-3 — sparse simulation uses nearest-integer rather than ceiling density

**Verdict: REJECTED.** Raised by R1 (wrong-numbers lens).

For a discrete patch count, `round(n_patches * (1 - mask_ratio))` is the nearest
representable density and therefore minimizes absolute density error; `ceil` would
systematically bias visibility upward. No external configuration contract promises
ceiling. The lone contrary statement is the private helper's docstring, while the
implementation and the field's public description consistently define a ratio, not
a minimum count. The `B=1,H=1,W=5,mask_ratio=0.5` result of two visible patches is
therefore defensible behavior; at most the word `ceil` in `sparse.py:25` is stale
documentation and does not justify changing the sampling arithmetic.

### E-R2-1 — concurrent cache misses share one temporary filename

**Verdict: UPHELD — FIXED in 0a8e997.** Raised as a finding
by R2 (contracts-and-runtime lens) and previously as a question by R1. R2's
two-thread reproduction promotes the question to a confirmed finding.

Failure scenario: two workers miss the same key and both write `<final>.tmp`; one
worker replaces that shared path, after which the other worker's `os.replace` raises
`FileNotFoundError`. In distributed training, that rank-local exception can leave
peers waiting at a later trainer collective even though Area E itself has no
collective.

Minimal patch, **NOT FIXED**: in `cache.py:43-48`, give each save a unique sibling
temporary path (for example, append a UUID), write to that private path, atomically
replace the common final path, and remove only that writer's temporary file in a
`finally` block if it remains. Keep the final filename and last-completed-writer
semantics unchanged.

### E-R2-3 — fixed normalization accepts zero, negative, and non-finite constants

**Verdict: UPHELD — FIXED in 711f575.** Raised by R2
(contracts-and-runtime lens).

Failure scenario: `norm=fixed,norm_constant_m=0,log_depth=true` converts every valid
distance to signal zero while retaining mask one, silently reducing conditioning to
mask-only. A negative constant can put `log1p` outside its domain, and NaN passes the
current validation and propagates.

Minimal patch, **NOT FIXED**: in `DepthCondCfg.validate` at `config.py:92-112`, when
`enabled` and `norm is NormType.FIXED`, require `norm_constant_m` to be finite and
strictly positive and raise `ValueError` naming the value otherwise. Do not mutate
the constant or validate the frozen-but-unread value in raw mode.

### E-R2-4 — enabled LoRA accepts zero or non-finite alpha

**Verdict: UPHELD — FIXED in aa60671.** Raised by R2
(contracts-and-runtime lens).

Failure scenario: `enabled=true,alpha=0` constructs nominal adapters whose scaling is
zero, so their output and A/B/input gradients remain zero; the run is labeled LoRA
enabled but cannot train LoRA. `alpha=NaN` makes even the zero-initialized branch
produce NaN through `0 * NaN`.

Minimal patch, **NOT FIXED**: in `LoRACfg.validate` at `config.py:125-130`, when LoRA
is enabled, require `alpha` to be finite and strictly positive and raise `ValueError`
otherwise. Leave disabled-mode identity behavior and all frozen fields unchanged.

### E-R2-5 — enabled head injection accepts an empty head list

**Verdict: UPHELD — FIXED in b372de1.** Raised by R2
(contracts-and-runtime lens).

Failure scenario: `enabled=true,injection=head,heads=[]` builds no injection modules
and returns `{'depth': None, 'point': None}`. The experiment is labeled as depth
conditioning while injecting depth into no output, a silent no-op rather than merely
a low-quality user choice.

Minimal patch, **NOT FIXED**: in `DepthCondCfg.validate` at `config.py:92-112`, raise
`ValueError` when depth conditioning is enabled, injection is `HEAD`, and `heads` is
empty. Do not alter token-injection handling of the inactive `heads` field.

### E-R2-6 — enabled LoRA accepts no targets and over-reports wrapping

**Verdict: UPHELD — FIXED in e4770b0.** Raised by R2
(contracts-and-runtime lens).

Failure scenario: `enabled=true,targets=[]` leaves every qkv/proj as a plain
`Linear`, but `apply_lora` increments its counter for every attention block. Startup
therefore reports wrapped adapters even though no adapter exists.

Minimal patch, **NOT FIXED**: in `LoRACfg.validate` at `config.py:125-130`, raise
`ValueError` when LoRA is enabled and `targets` is empty. This fail-fast makes the
existing counter reachable only for configurations that wrap at least one projection
per visited attention module; do not add fallback targets or alter disabled mode.

### E-R2-7 — malformed supplied cache-key cardinality silently disables caching

**Verdict: UPHELD — FIXED in 63291a8.** Raised by R2
(contracts-and-runtime lens).

Failure scenario: for batch size two, a supplied one-element cache-key value is
converted to one key, rejected only by an internal length comparison, and then
treated like an absent key. Every batch recomputes encoder tokens while the run still
reports the cache enabled.

Resolution: the documented live-path fallback remains only for an absent `cache_key`.
A supplied key count that differs from the batch size raises `ValueError` naming the
view index and the expected and received counts.

### E-R2-8 — stale gate wording promises a removed scalar gate

**Verdict: REJECTED.** Raised by R2 (contracts-and-runtime lens).

Current behavior is internally coherent: the only zero-init token component is
`token_proj`, it receives a step-zero gradient, current same-config strict reload
works, and `conditioner.py:193-203` explicitly records that the scalar gate was
removed and old gated checkpoints require the pre-removal revision. The word
“gated” in one return-shape docstring and historical stage notes does not define a
public `conditioner.gate` state-dict key. Treating those historical notes as a live
checkpoint API would duplicate the already documented compatibility limitation and
create churn without fixing current execution.

### Verified correct

- Current same-config strict self-reload succeeds after applying identical LoRA
  wrapping; there are no missing or unexpected state keys.
- `freeze_for_finetune` selects LoRA A/B, configured output heads, and the
  conditioner, while optimizer grouping omits frozen parameters.
- Enum fallthroughs raise, unsupported MAE/token-append modes raise, and cache S>1,
  injection shapes, per-frame conditioning lengths, and DPT residual counts are
  checked.
- The current zero-initialized `token_proj` closes the former scalar-gate gradient
  deadlock and can train at step zero.
- Area E contains no collective or barrier and no new Area A/B interaction defect
  was found: Area A commit `bac1cac` preserves all-invalid masks through sparse
  simulation, and Area B commit `5ac844e` rejects false metric labels for supported
  metric datasets.

### Reviewer questions (not findings)

- R1's shared-temp-path question is resolved above as the UPHELD E-R2-1 finding.
- R1's positive non-metric depth question remains hypothetical because all currently
  supported selectable metric loaders require `is_metric=True`; it is not promoted.
- R2's inactive-mode identity and processed-RGB hash questions require experiment
  identity and preprocessing policy. The frozen `FinetuneDepthCfg` and
  `_NON_IDENTITY_FIELDS` are not changed or proposed for change here.

No Area E CPU probe was run in this defense round. The verdicts use static inspection
of permitted files and the reviewers' supplied CPU reproductions. No GPU, checkpoint,
dataset, training, evaluation, or export path was used.

## Uncovered areas

- All five areas A-E have now been reviewed. Areas A-D had their UPHELD findings
  fixed with CPU regression evidence on branch `review/codebase-sweep`.
- Area E's five scoped findings were fixed with CPU regression evidence.
- Findings marked DEFERRED across all areas remain open and unfixed.
- Nothing was pushed or merged to `main`.
- Permanently out of scope: `datasets_preprocess/`, export and visualization scripts,
  `tests/`, `experiments/`, `config/`.

## F821 audit

- `pts3d` — `src/eval/mv_recon/criterion.py:132`, the fallback return in
  `get_pred_pts3d`: **DEAD CODE — unreachable, no caller**. The only repository
  call is from `Regr3D_t.get_all_pts3d_t`, and the reachable StreamVGGT evaluation
  path supplies `pts3d_in_other_view`, which returns at line 131. No binding or
  star-import supplies `pts3d`. Git `-S` and blame attribute the line to `49656da`;
  it is present at baseline `489d28f`, so it is pre-existing, not introduced by
  this review. No fix was made because the dead fallback's intended source is
  ambiguous and guessing could change evaluation values.
- `apply_distortion` — `src/eval/pose_evaluation/tensor_to_pycolmap.py:326`, the
  `extra_params is not None` branch of `img_from_cam`: **DEAD CODE — unreachable,
  no caller**. The sole repository projection call does not pass `extra_params`,
  `run_vggt_with_ba` rejects radial cameras, and the CO3D launcher rejects
  `--use_ba` before BA. No binding or star-import supplies the helper. Git `-S`
  and blame attribute the line to `49656da`; it is present at baseline `489d28f`,
  so it is pre-existing, not introduced by this review. No fix was made because
  there is no live distortion path and choosing a distortion model would require
  guessing intent.

## Self-audit — review of the review's own diff

### Verdicts

- **R1-1 — UPHELD — AMEND.** A criterion-wide masked mean is pixel-weighted;
  multiplying it by batch size cannot reconstruct equal clip weights. With clip
  errors/support `(0, 1)` and `(4, 3)`, the real `DepthOrPmapLoss` returns `3`
  for the batch, while the intended value after a later zero-loss clip is `4/3`,
  not `2`. Commit `af60361` slices the already-produced views/predictions and
  evaluates the criterion per clip. Commits `2a0d3d6` and `f026fba` preserve two
  existing test-helper contracts exposed by the full gate.
- **R1-2 / R2-2 — UPHELD — AMEND.** The active SPOT geometry array lost
  `--landscape-crop --crop-anchor top`, changing inputs and cache identity even
  with timing disabled. Commit `3419f1f` restores the established protocol.
- **R1-3 — UPHELD — AMEND.** An empty masked selection satisfied the all-zero
  predicate and raised before the established `valid_pixels=0` result path.
  Commit `091aeb5` applies the finite/nonzero check only to nonempty selections
  in all three evaluators.
- **R1-4 — UPHELD — AMEND.** Global CUDA availability is not the inference
  device and can select an empty or wrong-device stream. Commit `5eab304`
  selects the frame tensor's device and records/synchronizes that device's
  current stream. CPU selection is covered; executing CUDA events remains a
  GPU-only path.
- **R1-5 / R2-6 — UPHELD — AMEND.** For one frame, `t[1:]` is empty and NumPy
  min/max raises after inference. Commit `a549585` reports frame 0 alone and
  only computes steady-state statistics when those frames exist.
- **R2-1 — STILL DEFERRED — run two CPU ranks with `epoch_size=1`,
  `batch_size=1`, `drop_last=True`, and `even_batches=False`, recording raw and
  prepared loader lengths on both ranks; a fix is settled only when both ranks
  coherently raise before model setup whenever any prepared length is zero.** A
  single-process fake length does not verify the sharding hazard.
- **R2-3 — STILL DEFERRED — run two CPU ranks with rank 0 fault-injected to
  raise `OSError(EIO)` at `fsync` and separately at `os.replace`, while rank 1
  reaches checkpoint commit; a fix is settled only if an error collective makes
  both ranks exit with the failing operation named, without timeout, while a
  no-fault control still atomically replaces the temp file.** No unverified
  collective patch was added.
- **R2-4 — UPHELD — AMEND.** Unconditional `LossConfig` checks rejected recipes
  that do not consume temporal settings, while direct temporal/trimmed loss
  constructors accepted them. Commit `9a0ec86` gates config validation to
  `DEPTH_TRAIN` and validates at the consuming constructors.
- **R2-5 — UPHELD — AMEND.** Direct `TrimmedMAELoss` and `GradientLoss`
  construction still mapped a misspelled reduction to image-based behavior.
  Commit `441a2fb` validates both public branching constructors.

### Vendored timing dependency

The user's `src/dust3r/inference.py` edit adds optional `frame_times_ms` only to
the inference branch of `loss_of_one_batch`: `None` preserves the old model call;
a supplied list is forwarded to `model.inference` and returned in the result.
Training, teacher, and loss branches are unchanged, and existing callers omit the
argument. Timed `visualize_depth.py` and `visualize_spot.py` calls therefore form
a coupled local API fork with both `MetricStreamVGGT.inference` and
`StreamVGGT.inference`: removing either the vendor wrapper change or the model
changes while retaining the other side produces an unexpected-keyword failure.
The vendored file was carried verbatim in `196d8e2` and was not extended here.

### CPU evidence and test-quality audit

Failing-before probes used the required interpreter and `PYTHONDONTWRITEBYTECODE=1`:
the real loss probe observed `batch_loss 3.0`; all three empty-support calls
raised `ValueError`; irrelevant recipes rejected `temp_grad_scales=0`; direct
temporal/reduction constructors were accepted; the SPOT source regression,
one-frame formatter import, and input-device source regression each failed.
After their respective commits, `tests/test_train_area_d.py`,
`tests/test_eval_area_c.py`, `tests/test_loss_area_a.py`, and
`tests/test_review_diff_audit.py` pass. The final required eight-suite gate also
passes, as does the additional self-audit suite.

The audit confirms Reviewer 2's characterization: dataset tests exercise real
pure helpers; evaluation tests mix behavioral coverage with AST/source seams;
training tests rely on fake reducers and cannot establish post-shard or
rank-asymmetric behavior; the former scalar-only clip-loss test hid the real
criterion denominator and is now strengthened with `DepthOrPmapLoss`. The
scale-loss arity probe remains a seam test. Ruff reports five pre-existing unused
imports, all in the user-owned vendored `src/dust3r/inference.py`; they were not
modified. After formatting this round's non-user changes, format check reports
four pre-existing user-owned files; they were not auto-fixed.
