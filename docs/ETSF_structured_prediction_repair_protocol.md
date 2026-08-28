# ETSF structured-prediction repair protocol

Status: prospective development protocol for the next OOF run. It is not a
reinterpretation of the frozen development250 adequacy decision, does not alter
the action-ranking authorization guard, and provides no fresh50 evidence.

## Frozen development250 diagnosis

The shared event representation is not generally broken. Heldout next-event
prediction reaches accuracy 0.8998, NLL 0.2775, and present-class macro-F1
0.4873, with clustered comparisons strictly better than both current-event
persistence and the other-fold class prior. Failure/success outcome prediction
also beats its other-fold prior. The failures are head- and objective-specific:

- Deployment success (first four candidates only) has prevalence 0.116 but raw
  mean probability 0.418, Brier 0.1901, NLL 0.5674, and ECE 0.3018. Training
  used success `pos_weight` 6.91--8.97 by fold, while the diagnostic only fitted
  a zero-intercept temperature. Weighted BCE changes the probability prior; a
  temperature cannot undo that intercept shift.
- `near_goal`, `stationary`, and predicate `success` have only 54, 27, and 133
  positives among 8,682 transitions. Their AP/ROC-AUC show useful ranking, but
  the capped positive weight of 20 makes raw sigmoid values overconfident.
- Only 2,310 of 8,682 durations are observed; 6,372 are right-censored. The
  current duration likelihood lets the censoring term dominate even though a
  separate reach head already predicts observation with Brier 0.0457 and AUC
  0.9727. On observed rows, 63.4% of duration point predictions overshoot. For
  clock event 1, target median is 25 steps while the predicted median is 109.
- Object displacement is zero to numerical precision on roughly half of all
  transitions. There are also finite but physically implausible simulation
  excursions: maximum absolute displacement is 101.0 and nine rows exceed 5.
  Frozen factual standard deviations `[0.196, 0.136, 0.114]` turn the largest
  row into roughly `[-228, -744, -6.4]` standard deviations. Across 15 member
  logs, object Gaussian NLL peaks as high as 4,410 and total loss as high as
  2,213 before a global gradient clip, so the object head can consume the
  shared update budget. Heldout object prediction/target coordinate
  correlations are only 0.03--0.05.
- The factual checkpoint SHA is correct and the semantic encoder stays exactly
  frozen. This is not a load failure. Across the 15 OOF checkpoints, relative
  parameter drift averages 5.1% for the transition trunk, 11.8% for the
  duration mean head, 11.4% for the predicate head, and 7.1% for the object mean
  head. The combination points to objective/scale mismatch and shared-gradient
  interference rather than missing initialization.
- There are 57 recovery labels, but the OOF configuration inherited
  `recovery_supervised=false` from the factual checkpoint. The valid binary
  failure/success result is not evidence for recovery prediction.

## Probability repair without outer-fold leakage

For a weighted BCE coefficient `w`, the probability logit is recovered
analytically as `z_probability = z_weighted - log(w)`. This is permitted only
when the checkpoint records the coefficient actually used by that head before
training, its owner fold, the owner-training-group hash, and proof that the
owner holdout was excluded. Recomputing `w` from the current branch folds is
not equivalent for a frozen factual head and is forbidden.

For v5 adjusted logits, other outer-fold OOF models were trained on target fold
`f`; using their predictions to fit a temperature has indirect leakage. V6
structured diagnostics instead repeat one bit-exact frozen factual success
logit, so there is no outer-model crossfold dependence, but the factual
training `pos_weight` was not recorded and historical overlap with the old100
subset is not excluded. Therefore neither route is valid for strict all250
probability adequacy. Bias/temperature requires a separately proven nested
inner-OOF artifact inside every outer training split.

An earlier development counterfactual obtained Brier 0.0871 by reconstructing
`w` from branch folds. It is retained only as a root-cause hypothesis and is
not an admissible v6 calibration result. Without a valid recorded head-training
contract, diagnostics report uncalibrated probabilities descriptively and set
`strict_probability_calibration_evaluable=false`; Brier, NLL, and ECE checks
fail closed. PR-AUC and within-group ordering may still use raw logits because
they do not require probability calibration.

## Next-OOF training targets

Duration is factorized as `p(reach within horizon | x)` and
`p(log1p(D) | reach, x)`. Each outer training split stores medians in the
fallback order event x body, event, body, global. The duration head predicts a
residual around that median and receives an observed-only Laplace NLL, aligning
its location with median/log-MAE. Censored rows train the reach classifier but
do not push the conditional duration location past the horizon.

Object displacement uses a contract fitted independently in every outer
training split: coordinate median, robust scale, and the 99.5th percentile of
maximum absolute displacement. Non-finite rows and rows beyond that training
fence are excluded from the object likelihood. The head predicts a residual
around the robust median with a Student-t likelihood (`df=3`), whose influence
is bounded for large errors. This protects the shared trunk; it does not make
the current failed object head adequate. On frozen dev250, the crossfit mask
excludes 44/8,682 rows, and the existing model is still worse than the robust
median on the remaining rows (MAE 0.0490 vs 0.0415), so a new OOF run is
required.

Recovery is isolated in a linear adapter on `output["transition"].detach()`.
Its unweighted BCE keeps sigmoid values probabilistic and its public API
prevents gradients from reaching the shared world model. Recovery remains a
separate fail-closed domain until this adapter has its own heldout OOF metrics;
the current 57 labels only establish trainability.

## Frozen-v6 development-only shrinkage exploration

A read-only analysis used the frozen v6 structured outer predictions. For each
target outer fold, all shrinkage constants were selected using only the other
200 branch groups; target labels were used only for reporting. This excludes
branch outer-fold tuning leakage, but it does not erase possible factual-model
history on old100, so these are development repair signals rather than new
adequacy evidence.

- Duration blended the training-fold event x body log-median with the frozen
  factual log-duration. Every fold selected residual multiplier `0.375` from
  the fixed grid. Heldout observed log-MAE was 0.5531 versus 0.6271 for the
  median baseline; the equal-group paired 95% CI was
  `[-0.0902, -0.0518]`. Duration shrinkage is the only repair here with clear
  heldout skill and is suitable for preregistration in a new run.
- Object displacement selected multiplier `0` in every fold after the robust
  q99.5 quality mask. It exactly fell back to the fold median/zero baseline and
  provided no improvement. A learned object-delta claim remains unsupported.
- A frozen-logit recovery adapter reached AP 0.154 versus prevalence 0.0496,
  but its Brier and NLL paired confidence intervals both crossed zero. It shows
  ranking signal, not adequate recovery probability prediction.

Implementation: `scripts/openvla_etsf_prediction_repair.py`. The existing
diagnostic reports these repairs under `prospective_next_oof_*`, explicitly
excluded from the current preregistered adequacy decision.
