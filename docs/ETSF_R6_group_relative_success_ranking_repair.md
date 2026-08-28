# ETSF R6 group-relative success/ranking repair

## R5 diagnosis

The formal R5 development result selected `alpha=0.25` in every outer fold.
Shrinkage improved the ECE, Brier and NLL point estimates, but the strict
cluster-bootstrap confidence gates and AP gate did not pass. More importantly,
positive affine shrinkage preserves every within-fold candidate ordering and
tie, so it cannot change the selected action or task success rate. Probability
calibration alone is therefore not an action-selection repair.

## R6 model

R6 adds a detached adapter over the already frozen factual `transition` tensor.
For each logical group and each of its four deployment candidates, the feature
is either:

- candidate transition minus the deterministic candidate transition; or
- candidate transition minus the four-candidate group mean.

A fixed candidate one-hot is appended. Feature mean and scale are fitted only
on the applicable training groups. There is no trainable shared representation:

- the probability head is an independent linear head trained by unweighted
  candidate-level binary cross-entropy;
- the ranking head is an independent linear head trained by pairwise logistic
  loss or listwise successful-candidate cross-entropy.

The ranking weight is fixed at `1.0`. The nested grid is fixed to two relative
feature modes, two ranking objectives and two L2 values (`1e-3`, `1e-2`), for
eight configurations total. All are convex low-dimensional heads optimized by
deterministic full-batch LBFGS.

Probability evaluation converts logits to float64 and applies a fixed
`1e-12` open-interval clip. This only prevents CUDA float32 sigmoid saturation
from producing exact zero/one values in NLL; it is never used by the independent
ranking rule.

## Leakage boundary and fixed action rule

For each outer fold, five label-free hash-partitioned inner group folds choose
one configuration using this fixed lexicographic rule:

1. maximum inner-OOF selected-candidate success;
2. maximum equal-logical-group mean discordant-pair accuracy;
3. minimum equal-group Brier score;
4. minimum equal-group NLL;
5. lexical configuration ID.

The final adapter is then fitted on that outer fold's complete training groups.
All five signed selection contracts and self-contained adapter states must be
finished before any outer-holdout artifact is deserialized. Outer-holdout labels
never affect features, normalization, objectives, hyperparameters or weights.

Task action selection is fixed before holdout evaluation: select the maximum
`candidate_ranking_score`; an exact tie selects the lowest candidate index.
`success_probability` is evaluated independently and is never used by this
action rule.

## Strict development metrics

Probability adequacy requires all of:

- group-bootstrap Brier difference versus owner-fold training prevalence has
  upper 95% confidence bound below zero;
- analogous NLL upper bound below zero;
- AP minus evaluation prevalence has lower bound above zero;
- 10-bin ECE is at most `0.10`.

Ranking adequacy requires both group-bootstrap lower bounds above zero for:

- selected success minus deterministic-candidate success;
- equal-logical-group mean within-group pair accuracy minus `0.5`.

At least four of five outer folds must also be non-inferior to deterministic
candidate success at the point estimate.

## Claim scope

This repair was designed after inspecting the formal R5 D250 result. Reusing
D250 for R6 is therefore adaptive development analysis even though the nested
OOF computation itself is group-disjoint. A passing result cannot be described
as confirmation, cannot authorize deployment, and cannot establish a new task
success claim. Independent data not used to design R6 is required for that.
Fresh/confirmation paths and labels are rejected by the implementation.
