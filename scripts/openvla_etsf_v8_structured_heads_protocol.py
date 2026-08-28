#!/usr/bin/env python3
"""Preregistered, development-only contract for ETSF v8 structured heads.

This module is intentionally array/file agnostic.  It defines the immutable
statistical and OOF ownership contract consumed by the v8 array evaluator.  It
does not import the v7 launcher, open a rollout collection, authorize Fresh50,
or change an action-selection rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


FORMAT = "etsf_v8_structured_heads_preregistration_v1"
FOLD_COUNT = 5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260917
ALPHA = 0.05
EPS = 1e-12


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha(value: str, *, name: str) -> str:
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase sha256")
    return value


def make_preregistration(
    *,
    implementation_sha256: str,
    label_derivation_sha256: str,
    base_checkpoint_sha256: str,
    base_training_groups_sha256: str,
) -> dict[str, Any]:
    """Create the single v8 evaluation contract before target labels are read."""

    contract: dict[str, Any] = {
        "format": FORMAT,
        "development_only": True,
        "created_before_target_labels_read": True,
        "fresh50": {
            "inputs_accepted": False,
            "labels_read": False,
            "authorization_possible": False,
        },
        "scope": {
            "changes_v7_implementation": False,
            "changes_action_rank_guard": False,
            "authorizes_selector": False,
            "next_event_policy": "frozen_bit_exact_passthrough",
            "next_event_accuracy_not_re_evaluated": True,
            "old100_historical_overlap": "descriptive_separate_excluded_from_primary",
        },
        "source_sha256": {
            "implementation": _sha(implementation_sha256, name="implementation"),
            "label_derivation": _sha(label_derivation_sha256, name="label derivation"),
            "base_checkpoint": _sha(base_checkpoint_sha256, name="base checkpoint"),
            "base_training_groups": _sha(
                base_training_groups_sha256, name="base training groups"
            ),
        },
        "oof": {
            "fold_count": FOLD_COUNT,
            "split_unit": "logical_group",
            "owner_holdout_excluded_from_head_training": True,
            "baseline_and_calibration_fit_scope": "outer_training_or_nested_inner_oof_only",
            "deployment_candidate_scope": "candidate_index_less_than_4",
        },
        "statistics": {
            "resampling_unit": "logical_group_cluster",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "two_sided_alpha": ALPHA,
            "loss_skill_gate": "model_minus_baseline_upper_ci_strictly_below_zero",
            "ranking_skill_gate": "ap_minus_prevalence_lower_ci_strictly_above_zero",
            "ece_bins": 10,
        },
        "domains": {
            "duration": {
                "target": "observed_only_log1p_duration",
                "censored_rows_used": "reach_only_not_duration",
                "point_prediction": "event_body_training_median_plus_0.375_frozen_residual",
                "likelihood": "laplace",
                "minimum_observed_groups": 30,
                "minimum_outer_training_event_body_support": 20,
                "minimum_fold_point_wins": 4,
            },
            "success": {
                "candidate_scope": "first_four_only",
                "metrics": ["brier", "nll", "ece_10", "ap_minus_prevalence"],
                "minimum_positive_per_fold": 10,
                "minimum_negative_per_fold": 10,
                "maximum_ece_10": 0.10,
                "probability_loss": "unweighted_or_inverse_sampling_unbiased_bce",
            },
            "regress": {
                "metrics": ["brier", "nll", "ece_10", "ap_minus_prevalence"],
                "minimum_positive_per_fold": 10,
                "minimum_negative_per_fold": 10,
                "maximum_ece_10": 0.10,
            },
            "recovery_given_regress": {
                "evaluation_subset": "ground_truth_regress_rows_only",
                "minimum_recovery_per_fold": 10,
                "minimum_nonrecovery_per_fold": 10,
                "conditional_and_unconditional_probability_skill_required": True,
            },
            "object": {
                "minimum_groups": 30,
                "minimum_quality_valid_coverage": 0.99,
                "baselines": ["zero", "outer_training_robust_median"],
                "strict_mae_skill_required_against_both": True,
                "rmse_and_p95_noninferiority_required_against_both": True,
                "minimum_fold_point_wins": 4,
                "default_output": "fallback_not_learned",
            },
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_preregistration(value: Mapping[str, Any]) -> None:
    """Fail closed if a supposedly preregistered contract drifted."""

    if value.get("format") != FORMAT:
        raise RuntimeError("v8 preregistration format mismatch")
    unsigned = dict(value)
    digest = unsigned.pop("contract_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise RuntimeError("v8 preregistration signature mismatch")
    sources = value.get("source_sha256", {})
    required_sources = {
        "implementation",
        "label_derivation",
        "base_checkpoint",
        "base_training_groups",
    }
    if set(sources) != required_sources:
        raise RuntimeError("v8 preregistration source hashes mismatch")
    expected = make_preregistration(
        implementation_sha256=sources["implementation"],
        label_derivation_sha256=sources["label_derivation"],
        base_checkpoint_sha256=sources["base_checkpoint"],
        base_training_groups_sha256=sources["base_training_groups"],
    )
    if dict(value) != expected:
        raise RuntimeError("v8 preregistration frozen contract mismatch")
    if value.get("development_only") is not True or value.get(
        "created_before_target_labels_read"
    ) is not True:
        raise RuntimeError("v8 preregistration must precede development labels")
    fresh = value.get("fresh50", {})
    if fresh != {
        "inputs_accepted": False,
        "labels_read": False,
        "authorization_possible": False,
    }:
        raise RuntimeError("v8 must never read or authorize Fresh50")
    if value.get("statistics") != {
        "resampling_unit": "logical_group_cluster",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "two_sided_alpha": ALPHA,
        "loss_skill_gate": "model_minus_baseline_upper_ci_strictly_below_zero",
        "ranking_skill_gate": "ap_minus_prevalence_lower_ci_strictly_above_zero",
        "ece_bins": 10,
    }:
        raise RuntimeError("v8 preregistered statistics mismatch")
    if value.get("oof", {}).get("fold_count") != FOLD_COUNT:
        raise RuntimeError("v8 requires exactly five outer folds")
    if value.get("scope", {}).get("next_event_policy") != (
        "frozen_bit_exact_passthrough"
    ):
        raise RuntimeError("v8 next-event freeze contract mismatch")
    for name, digest in sources.items():
        _sha(digest, name=name)


def validate_probability_weight_provenance(
    provenance: Mapping[int | str, Mapping[str, Any]], *, head: str
) -> dict[str, Any]:
    """Validate probability semantics for one independently trained adapter."""

    normalized = {int(key): dict(item) for key, item in provenance.items()}
    if set(normalized) != set(range(FOLD_COUNT)):
        raise RuntimeError(f"{head} weight provenance must cover five folds")
    result: dict[str, Any] = {}
    allowed_losses = {"unweighted_bce", "inverse_sampling_unbiased_bce"}
    for fold_id, item in normalized.items():
        unsigned = dict(item)
        recorded_digest = unsigned.pop("weight_contract_sha256", None)
        if recorded_digest != canonical_sha256(unsigned):
            raise RuntimeError(f"{head} weight provenance signature mismatch")
        if int(item.get("owner_fold_id", -1)) != fold_id:
            raise RuntimeError(f"{head} weight provenance owner mismatch")
        if item.get("head") != head:
            raise RuntimeError(f"{head} weight provenance head binding mismatch")
        if item.get("loss") not in allowed_losses:
            raise RuntimeError(f"{head} probability loss is not calibrated by contract")
        if item.get("weights_recorded_before_training") is not True:
            raise RuntimeError(f"{head} training weights were not recorded")
        if item.get("owner_holdout_labels_used") is not False:
            raise RuntimeError(f"{head} owner labels leaked into weight/calibration fit")
        if item.get("calibration_source") not in {
            "none_unweighted_probability",
            "nested_inner_oof_only",
        }:
            raise RuntimeError(f"{head} calibration provenance is invalid")
        positive = int(item.get("outer_training_positive", -1))
        negative = int(item.get("outer_training_negative", -1))
        prevalence = float(item.get("outer_training_prevalence", float("nan")))
        if positive < 0 or negative < 0 or positive + negative <= 0:
            raise RuntimeError(f"{head} outer-training class counts are missing")
        expected_prevalence = positive / (positive + negative)
        if not math.isfinite(prevalence) or not math.isclose(
            prevalence, expected_prevalence, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"{head} outer-training prevalence is inconsistent")
        _sha(recorded_digest, name=f"{head} weights")
        result[str(fold_id)] = item
    return {
        "status": "complete_recorded_before_training",
        "head": head,
        "folds": result,
    }


def validate_oof_ownership(
    logical_group: Sequence[str],
    fold_id: Sequence[int],
    fold_contracts: Mapping[int | str, Mapping[str, Any]],
    *,
    expected_base_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Check group ownership, training exclusion, and frozen next-event hashes."""

    groups = [str(value) for value in logical_group]
    folds = [int(value) for value in fold_id]
    if not groups or len(groups) != len(folds):
        raise RuntimeError("v8 OOF group/fold arrays are empty or misaligned")
    group_owner: dict[str, int] = {}
    for group, owner in zip(groups, folds):
        if owner not in range(FOLD_COUNT):
            raise RuntimeError("v8 OOF fold id is out of range")
        if group in group_owner and group_owner[group] != owner:
            raise RuntimeError("one logical group has multiple OOF owners")
        group_owner[group] = owner
    if set(group_owner.values()) != set(range(FOLD_COUNT)):
        raise RuntimeError("v8 OOF rows must cover exactly five folds")

    normalized = {int(key): dict(item) for key, item in fold_contracts.items()}
    if set(normalized) != set(range(FOLD_COUNT)):
        raise RuntimeError("v8 OOF fold contracts must cover exactly five folds")
    expected_base = _sha(expected_base_checkpoint_sha256, name="base checkpoint")
    summaries: dict[str, Any] = {}
    for owner, item in normalized.items():
        unsigned = dict(item)
        recorded_digest = unsigned.pop("fold_contract_sha256", None)
        if recorded_digest != canonical_sha256(unsigned):
            raise RuntimeError("v8 OOF fold contract signature mismatch")
        heldout = set(map(str, item.get("heldout_logical_groups", [])))
        training = set(map(str, item.get("training_logical_groups", [])))
        actual = {group for group, fold in group_owner.items() if fold == owner}
        if int(item.get("owner_fold_id", -1)) != owner or heldout != actual:
            raise RuntimeError("v8 OOF heldout ownership mismatch")
        if heldout & training:
            raise RuntimeError("v8 OOF owner holdout leaked into head training")
        if item.get("outer_target_labels_used_for_fit") is not False:
            raise RuntimeError("v8 OOF target labels were used for fit")
        if item.get("baseline_fit_scope") != "outer_training_only" or item.get(
            "calibration_fit_scope"
        ) not in {"none", "nested_inner_oof_only"}:
            raise RuntimeError("v8 baseline/calibration fit scope is invalid")
        if _sha(item.get("base_checkpoint_sha256", ""), name="fold base") != expected_base:
            raise RuntimeError("v8 fold uses an unexpected base checkpoint")
        before = _sha(item.get("next_event_state_sha256_before", ""), name="next-event before")
        after = _sha(item.get("next_event_state_sha256_after", ""), name="next-event after")
        if before != after or item.get("next_event_trainable") is not False:
            raise RuntimeError("v8 next-event state is not bit-exact frozen")
        trainable = [str(name) for name in item.get("trainable_parameter_names", [])]
        if any("next_event" in name or "transition_core" in name for name in trainable):
            raise RuntimeError("v8 trainable parameters cross the frozen core boundary")
        if int(item.get("duration_event_body_min_training_support", -1)) < 0:
            raise RuntimeError("v8 duration cell support provenance is missing")
        if float(item.get("duration_residual_multiplier", float("nan"))) != 0.375:
            raise RuntimeError("v8 duration residual multiplier drifted")
        if item.get("duration_censored_used_for_location") is not False:
            raise RuntimeError("v8 censored duration rows entered location training")
        if item.get("duration_scale_fit_scope") != "outer_training_only":
            raise RuntimeError("v8 duration scale fit scope is not outer-training only")
        summaries[str(owner)] = {
            "heldout_groups": len(heldout),
            "training_groups": len(training),
            "next_event_bit_exact": True,
            "duration_event_body_min_training_support": int(
                item["duration_event_body_min_training_support"]
            ),
        }
    return {
        "status": "strict_group_oof_validated",
        "logical_groups": len(group_owner),
        "folds": summaries,
        "next_event_frozen_bit_exact": True,
    }


__all__ = [
    "ALPHA",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "FOLD_COUNT",
    "FORMAT",
    "canonical_sha256",
    "make_preregistration",
    "validate_oof_ownership",
    "validate_preregistration",
    "validate_probability_weight_provenance",
]
