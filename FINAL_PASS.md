# PR #26 final pass — current reconciled position

This document states one current position for `review/codebase-sweep`. Earlier
round conclusions that later changed are retained below as explicit
retractions. `FINAL_PASS.md` and `EVAL_PARITY.md` are the tracked reports;
`CODE_REVIEW_FINDINGS.md` and `IMPLEMENTATION_NOTES.md` remain local and
untracked by user decision.

## Current verdict

**FINAL VERDICT: MERGE-READY WITH NAMED CAVEATS.** DUST-1 and FP-2 are closed.
The accepted live defects and evidence limits below remain visible and do not
claim resolution.

## Closed findings

- **DUST-1 is resolved.** `bfad0af` restored
  `src/dust3r/inference.py` byte-for-byte to `main`. `6889915` and `7e6c7a3`
  route only the visualizers' timed mode through the shared first-party helper
  `_run_streaming_inference`; untimed mode still calls the vendored
  `loss_of_one_batch`. The helper reuses the vendored query-point sampler and
  calls the model-owned per-frame timing path. No vendored implementation was
  copied.
- **FP-2 is closed by `4deba02`.** `tests/test_visualize_timing.py` calls the
  real `_run_streaming_inference` and real `loss_of_one_batch` on identical
  synthetic inputs and RNG state. It asserts elementwise-equal predictions,
  verifies that the untimed model call receives no timing list, and requires
  one timing entry per input frame from the timed route. Only the GPU-scale
  model is replaced by a small deterministic CPU stub; the vendored wrapper's
  unconditional `torch.cuda.get_device_capability` probe is patched so that
  the real wrapper can run without a GPU.
- The FP-2 test discriminates: temporarily changing the timed query-point count
  from 64 to 63 produced exit 1 and the literal failure
  `AssertionError` at
  `assert torch.equal(untimed_pred["depth"], timed_pred["depth"])`. After
  `git checkout HEAD -- src/visualize_depth.py`, the same test exited 0 and the
  tracked worktree was clean.
- `tests/test_review_diff_audit.py` still checks the one-frame timing-summary
  string. It is not redundant with the new route test: it guards formatting,
  while the new test guards inference and timing behavior. Its former status as
  the only supposed timing-contract test is retracted.

## Named caveats and accepted decisions

1. Real-weight timed/untimed numerical equivalence remains unexercised. The
   permanent regression uses a deterministic CPU model stub; no GPU, real
   checkpoint, real dataset, or end-to-end visualizer execution is claimed.
2. Component-mask tuple fabrication remains byte-comparison-only. The affine
   route is behavior-tested by `7140e32`, and rank-log cap/first-gap behavior
   is tested by `89f5a10`, but the component-mask public path has no behavioral
   regression.
3. Items 13 and 14 are accepted pre-existing live defects. Failed
   video-evaluation sequences can be omitted from the denominator, and
   prediction/GT files can be paired by independent sort position without
   identity or count assertions. Both behaviors exist at import state
   `49656da`, `main` (`489d28f`), and this branch.
4. Item 16 is accepted live: a rank-0 checkpoint-commit failure can leave peers
   waiting for the unchanged 6000-second timeout.
5. **Accepted, not resolved — independent red reproduction is limited.** An
   agent that wrote neither the test nor the fix independently reproduced the
   red for roughly 8 of about 40 commits carrying test claims. For the
   remainder, the red was reported by the same agent that wrote both the test
   and the fix. The user was told this precise limitation and accepted it; it
   remains an evidence caveat, not independent verification of every claim.
6. Items 10, 11, and 12 are declined by the user: teacher-confidence
   weighting, inert `lambda_track`, and unused `CameraLoss.delta` remain
   unchanged.
7. Item 8 remains declined. No pre-existing ruff violation was touched. The
   current changed-Python-file lint set has 20 errors; five additional
   historical errors were in `src/dust3r/inference.py` and left the changed set
   when that vendored file was restored.

## Current item ledger

1. **Resolved — durability.** The two local reports intentionally remain
   untracked; the tracked reports are `FINAL_PASS.md` and `EVAL_PARITY.md`.
2. **Resolved by `6b83815`.** The dangling tracked reference to the local
   implementation report was removed without deleting its factual zero-init
   statement.
3. **Resolved — DUST-1.** The vendored file is pristine and visualizer timing is
   first-party, as described above.
4. **Resolved by `2a6a2da`.** `reject_contradictory_modes` is documented as a
   library opt-in whose false default preserves parity.
5. **Fixed by `8717715` — TU-1.** Global-rank precedence is `RANK`,
   `SLURM_PROCID`, then `LOCAL_RANK`.
6. **Fixed by `118e683` — VS-1.** Empty sparse-depth windows report zero valid
   pixels without fabricating a statistic.
7. **Fixed by `075ed03`.** The two PR-owned format failures were corrected; no
   unrelated file was formatted.
8. **Declined.** Pre-existing lint and format findings remain untouched.
9. **Resolved.** The parity reverts remain intact at behavior-bearing regions,
   subject to the documented whitespace-only component-mask difference and
   authorized fail-fast divergence.
9b. **Partly completed.** Affine and rank-log parity have behavioral tests;
   component-mask tuple fabrication remains byte-comparison-only.
10. **Declined.** Teacher-confidence weighting was not changed.
11. **Declined.** Inert `lambda_track` was not changed.
12. **Declined.** Unused `CameraLoss.delta` was not changed.
13. **Accepted pre-existing, live defect.** Failed sequences may be omitted
    from the video-evaluation denominator.
14. **Accepted pre-existing, live defect.** Prediction/GT files may be paired
    by independent sort position without identity/count assertions.
15. **Fixed by `85163d2`.** All three prepared-loader kinds fail fast at zero
    batches; a one-batch loader does not trigger the guard.
16. **Accepted live.** The 6000-second peer timeout is unchanged.

The frozen wandb key names, `FinetuneDepthCfg`, and `_NON_IDENTITY_FIELDS`
remain unchanged.

## Explicit historical retractions

- The Phase 10 statement that DUST-1 remained open and the vendored file had 13
  changed lines is historical only. It was superseded by the Phase 11 revert
  and first-party rerouting; it is not the current state.
- The historical changed-file lint count of 25 is retracted as a current claim.
  The reproducible current count is 20 because the pristine vendored file is no
  longer in `main...HEAD`; the five omitted violations were pre-existing and
  remain untouched in that vendored file.
- The early statement that item 9b was not attempted is historical only.
  `89f5a10` and `7140e32` later added rank-log and affine behavioral coverage;
  only component-mask tuple fabrication remains byte-comparison-only.
- TU-1 and VS-1 were initially rejected as pre-existing. Those dispositions
  are retracted as current status because the user later authorized and landed
  fixes `8717715` and `118e683`.
- The Cycle 3 statement that two PR-owned format failures were outstanding is
  historical only; `075ed03` fixed them. Item 8 still declines changes to
  pre-existing violations.
- The Phase 11 verification statement that the only timing audit checked
  formatting was accurate then, but is no longer current after `4deba02`.

No production source change is part of the FP-2 closure. No GPU, training,
evaluation, export, `srun`, or `sbatch` run is claimed for this round.
