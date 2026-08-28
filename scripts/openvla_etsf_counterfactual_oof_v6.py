#!/usr/bin/env python3
"""Frozen-core nested OOF protocol for the ETSF success-only rank head.

Version 6 is deliberately development-only.  It has no code path that can
authorize a final refit or fresh-confirmation access.  Every outer fold uses a
real four-fold crossfit on its own training groups to choose one threshold from
the single preregistered score-gain guard family.  The chosen threshold is then
applied once to that outer fold's untouched groups.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT = "etsf_counterfactual_nested_oof_v6"
SELECTION_FORMAT = "etsf_counterfactual_nested_oof_selection_v6"
OUTER_FOLDS = 5
INNER_FOLDS = 4
EXPECTED_GROUPS = 250
SPLIT_SEED = 20260827
GUARD_MARGIN_THRESHOLDS = (0.0, 0.05, 0.10, 0.20)
BOOTSTRAP_SEED = 20260903
BOOTSTRAP_SAMPLES = 10_000
MINIMUM_COVERAGE = 0.10
MINIMUM_CHANGES = 10
MAXIMUM_HARMFUL_RATE = 0.10
MINIMUM_INNER_LCB90 = 0.0
TRAINABLE_PARAMETER_NAMES = ("action_rank_head.0.weight",)
FORMAL_SEMANTIC_DIM = 96
FORMAL_TRAINABLE_PARAMETER_COUNT = 192


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_order(keys: Sequence[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        map(str, keys),
        key=lambda key: hashlib.sha256(
            f"{namespace}|{seed}|{key}".encode("utf-8")
        ).hexdigest(),
    )


def make_nested_oof_manifest(
    logical_keys: Sequence[str],
    *,
    source_contract: Mapping[str, Any],
    semantic_dim: int,
    split_seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    keys = sorted(map(str, logical_keys))
    if len(keys) != EXPECTED_GROUPS or len(set(keys)) != EXPECTED_GROUPS:
        raise RuntimeError("v6 requires exactly 250 unique development groups")
    if semantic_dim != FORMAL_SEMANTIC_DIM:
        raise ValueError("formal v6 requires semantic_dim=96")
    expected_parameters = 2 * int(semantic_dim)
    outer_width = EXPECTED_GROUPS // OUTER_FOLDS
    ordered = _hash_order(keys, split_seed, "outer")
    folds: list[dict[str, Any]] = []
    for outer_id in range(OUTER_FOLDS):
        outer_holdout = sorted(
            ordered[outer_id * outer_width : (outer_id + 1) * outer_width]
        )
        holdout_set = set(outer_holdout)
        outer_train = [key for key in keys if key not in holdout_set]
        if len(outer_train) % INNER_FOLDS:
            raise RuntimeError("outer training set is not divisible into inner folds")
        inner_width = len(outer_train) // INNER_FOLDS
        inner_order = _hash_order(
            outer_train, split_seed + outer_id + 1, f"inner_{outer_id}"
        )
        inner: list[dict[str, Any]] = []
        for inner_id in range(INNER_FOLDS):
            inner_holdout = sorted(
                inner_order[inner_id * inner_width : (inner_id + 1) * inner_width]
            )
            inner_holdout_set = set(inner_holdout)
            inner_train = [
                key for key in outer_train if key not in inner_holdout_set
            ]
            inner.append(
                {
                    "inner_fold_id": inner_id,
                    "training_groups": inner_train,
                    "selection_holdout_groups": inner_holdout,
                    "training_group_count": len(inner_train),
                    "selection_holdout_group_count": len(inner_holdout),
                }
            )
        folds.append(
            {
                "outer_fold_id": outer_id,
                "training_groups": outer_train,
                "oof_holdout_groups": outer_holdout,
                "training_group_count": len(outer_train),
                "oof_holdout_group_count": len(outer_holdout),
                "inner_folds": inner,
                "inner_selector": (
                    "real_four_fold_crossfit_on_outer_training_groups_only"
                ),
            }
        )
    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "preregistered",
        "expected_groups": EXPECTED_GROUPS,
        "outer_fold_count": OUTER_FOLDS,
        "inner_fold_count": INNER_FOLDS,
        "split_seed": split_seed,
        "split_algorithm": "sha256_nested_balanced_v1",
        "development_groups": keys,
        "source_contract": dict(source_contract),
        "model_contract": {
            "factual_core": "frozen_bit_exact",
            "action_rank_residual": True,
            "action_rank_success_only": True,
            "rank_head": "single_linear_no_bias_on_relative_action_features",
            "semantic_dim": int(semantic_dim),
            "trainable_parameter_names": list(TRAINABLE_PARAMETER_NAMES),
            "trainable_parameter_count": expected_parameters,
            "score": "frozen_base_success_logit_plus_relative_rank_residual",
            "candidate_zero_residual": 0.0,
            "event_duration_object_terms_in_rank_score": False,
        },
        "training_contract": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "groups_per_batch": 16,
            "inner_training_steps": 100,
            "outer_refit_steps": 100,
            "loss": "success_changing_pairwise_plus_baseline_contrast",
            "checkpoint_selection": "fixed_final_step_no_holdout_selection",
            "feature_gradient": "detached_before_action_rank_head",
        },
        "selector_contract": {
            "score_candidates": ["success_only"],
            "model_hyperparameter_candidates": 1,
            "guard_family": "score_gain_margin_threshold_v1",
            "gain_margin_thresholds": list(GUARD_MARGIN_THRESHOLDS),
            "selection_data": "inner_crossfit_predictions_only",
            "minimum_coverage": MINIMUM_COVERAGE,
            "minimum_changes": MINIMUM_CHANGES,
            "maximum_harmful_rate": MAXIMUM_HARMFUL_RATE,
            "minimum_paired_delta_lcb90": MINIMUM_INNER_LCB90,
            "selection_rule": (
                "eligible_then_max_lcb90_mean_delta_coverage_then_larger_margin"
            ),
        },
        "outer_evaluation_contract": {
            "threshold_source": "corresponding_outer_fold_inner_crossfit_only",
            "holdout_access": (
                "outer_labels_first_read_after_inner_selection_and_outer_refit"
            ),
            "aggregate_selection": "none_combine_each_outer_policy_once",
            "required_score_ablations": [
                "frozen_base_success_only",
                "residual_only",
                "frozen_base_plus_residual",
            ],
            "development_gate_descriptive_only": {
                "estimand": "unconditional_equal_group_success_delta_including_zeros",
                "bootstrap_confidence_interval": 0.95,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "exact_two_sided_sign_test": True,
                "pass_rule": (
                    "ci_lower_gt_zero_and_sign_p_lt_0.05_and_helpful_gt_harmful_"
                    "and_changed_groups_ge_10"
                ),
                "fresh_authorization_effect": "none",
            },
        },
        "fresh_confirmation": {
            "inputs_accepted": False,
            "data_or_labels_read": False,
            "authorization_possible": False,
            "policy": "forbidden_even_if_development_gate_passes",
        },
        "outer_folds": folds,
    }
    payload["preregistration_sha256"] = canonical_sha256(payload)
    validate_nested_oof_manifest(payload, keys)
    return payload


def validate_nested_oof_manifest(
    manifest: Mapping[str, Any], logical_keys: Sequence[str]
) -> dict[str, int]:
    if manifest.get("format") != FORMAT or manifest.get("status") != "preregistered":
        raise RuntimeError("unsupported or incomplete v6 preregistration")
    unsigned = dict(manifest)
    recorded = str(unsigned.pop("preregistration_sha256", ""))
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("v6 preregistration signature mismatch")
    keys = set(map(str, logical_keys))
    if len(keys) != EXPECTED_GROUPS or set(manifest.get("development_groups", [])) != keys:
        raise RuntimeError("v6 preregistration development identities changed")
    fresh = manifest.get("fresh_confirmation")
    if not isinstance(fresh, Mapping) or any(
        fresh.get(name) is not False
        for name in ("inputs_accepted", "data_or_labels_read", "authorization_possible")
    ) or fresh.get("policy") != "forbidden_even_if_development_gate_passes":
        raise RuntimeError("v6 must fail closed against fresh confirmation")
    model = manifest.get("model_contract")
    if not isinstance(model, Mapping):
        raise RuntimeError("v6 lacks a model contract")
    if model.get("factual_core") != "frozen_bit_exact" or model.get(
        "action_rank_success_only"
    ) is not True:
        raise RuntimeError("v6 model is not frozen-core success-only")
    if tuple(model.get("trainable_parameter_names", ())) != TRAINABLE_PARAMETER_NAMES:
        raise RuntimeError("v6 trainable parameter allowlist changed")
    if int(model.get("semantic_dim", -1)) != FORMAL_SEMANTIC_DIM or int(
        model.get("trainable_parameter_count", -1)
    ) != FORMAL_TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("v6 low-capacity parameter count changed")
    selector = manifest.get("selector_contract")
    if not isinstance(selector, Mapping) or selector.get("score_candidates") != [
        "success_only"
    ]:
        raise RuntimeError("v6 selector introduced an unregistered score")
    if tuple(selector.get("gain_margin_thresholds", ())) != GUARD_MARGIN_THRESHOLDS:
        raise RuntimeError("v6 guard family changed")
    folds = manifest.get("outer_folds")
    if not isinstance(folds, list) or len(folds) != OUTER_FOLDS:
        raise RuntimeError("v6 outer fold count changed")
    outer_seen: list[str] = []
    for outer_id, fold in enumerate(folds):
        if int(fold.get("outer_fold_id", -1)) != outer_id:
            raise RuntimeError("v6 outer fold ids are not canonical")
        train = set(map(str, fold.get("training_groups", [])))
        holdout = set(map(str, fold.get("oof_holdout_groups", [])))
        if len(train) != 200 or len(holdout) != 50 or train & holdout or train | holdout != keys:
            raise RuntimeError("v6 outer fold has leakage or wrong dimensions")
        outer_seen.extend(holdout)
        inner = fold.get("inner_folds")
        if not isinstance(inner, list) or len(inner) != INNER_FOLDS:
            raise RuntimeError("v6 lacks real inner folds")
        inner_seen: list[str] = []
        for inner_id, child in enumerate(inner):
            if int(child.get("inner_fold_id", -1)) != inner_id:
                raise RuntimeError("v6 inner fold ids are not canonical")
            child_train = set(map(str, child.get("training_groups", [])))
            child_holdout = set(map(str, child.get("selection_holdout_groups", [])))
            if (
                len(child_train) != 150
                or len(child_holdout) != 50
                or child_train & child_holdout
                or child_train | child_holdout != train
                or child_holdout & holdout
            ):
                raise RuntimeError("v6 inner fold has leakage or wrong dimensions")
            inner_seen.extend(child_holdout)
        if len(inner_seen) != 200 or set(inner_seen) != train:
            raise RuntimeError("v6 inner crossfit does not predict every outer-train group once")
    if len(outer_seen) != EXPECTED_GROUPS or set(outer_seen) != keys:
        raise RuntimeError("v6 outer OOF does not predict every development group once")
    return {
        "development_groups": EXPECTED_GROUPS,
        "outer_training_groups": 200,
        "outer_holdout_groups": 50,
        "inner_training_groups": 150,
        "inner_holdout_groups": 50,
    }


def _bootstrap_lcb90(delta: np.ndarray) -> float:
    if not len(delta):
        return -math.inf
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta))
    )
    return float(np.quantile(delta[indices].mean(1), 0.10))


def _row_decision(row: Mapping[str, Any], margin: float) -> dict[str, Any]:
    scores = np.asarray(row["success_only_scores"], dtype=np.float64)
    labels = np.asarray(row["success"], dtype=np.float64)
    baseline = int(row["baseline_index"])
    if scores.ndim != 1 or labels.shape != scores.shape or not 0 <= baseline < len(scores):
        raise RuntimeError("invalid v6 prediction row")
    proposed = int(np.argmax(scores))
    gain = float(scores[proposed] - scores[baseline])
    selected = proposed if proposed != baseline and gain >= margin else baseline
    return {
        "logical_key": str(row["logical_key"]),
        "baseline_index": baseline,
        "proposed_index": proposed,
        "selected_index": selected,
        "score_gain": gain,
        "changed": selected != baseline,
        "success_delta": float(labels[selected] - labels[baseline]),
        "oracle_headroom": bool(labels.max() > labels[baseline]),
    }


def evaluate_guard(rows: Sequence[Mapping[str, Any]], margin: float) -> dict[str, Any]:
    decisions = [_row_decision(row, margin) for row in rows]
    delta = np.asarray([row["success_delta"] for row in decisions], dtype=np.float64)
    changed = np.asarray([row["changed"] for row in decisions], dtype=bool)
    helpful = int((delta > 0).sum())
    harmful = int((delta < 0).sum())
    changed_count = int(changed.sum())
    return {
        "gain_margin": float(margin),
        "groups": len(decisions),
        "changed_groups": changed_count,
        "coverage": changed_count / max(len(decisions), 1),
        "helpful_changes": helpful,
        "harmful_changes": harmful,
        "harmful_rate_over_changes": harmful / max(changed_count, 1),
        "mean_paired_success_delta": float(delta.mean()) if len(delta) else 0.0,
        "paired_success_delta_lcb90": _bootstrap_lcb90(delta),
        "oracle_headroom_groups": int(sum(row["oracle_headroom"] for row in decisions)),
        "decisions": decisions,
    }


def select_inner_guard(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = []
    for margin in GUARD_MARGIN_THRESHOLDS:
        value = evaluate_guard(rows, margin)
        reasons = []
        if value["coverage"] < MINIMUM_COVERAGE:
            reasons.append("coverage_below_minimum")
        if value["changed_groups"] < MINIMUM_CHANGES:
            reasons.append("insufficient_changes")
        if value["harmful_rate_over_changes"] > MAXIMUM_HARMFUL_RATE:
            reasons.append("harmful_rate_above_maximum")
        if value["paired_success_delta_lcb90"] < MINIMUM_INNER_LCB90:
            reasons.append("paired_delta_lcb90_below_minimum")
        value["eligible"] = not reasons
        value["rejection_reasons"] = reasons
        value.pop("decisions")
        candidates.append(value)
    eligible = [value for value in candidates if value["eligible"]]
    if not eligible:
        return {
            "enabled": False,
            "gain_margin": None,
            "reason": "inner_crossfit_guard_not_eligible",
            "guard_family_candidates": candidates,
            "fresh_confirmation_allowed": False,
        }
    selected = max(
        eligible,
        key=lambda value: (
            value["paired_success_delta_lcb90"],
            value["mean_paired_success_delta"],
            value["coverage"],
            value["gain_margin"],
        ),
    )
    return {
        "enabled": True,
        "gain_margin": selected["gain_margin"],
        "reason": "selected_on_inner_crossfit_only",
        "guard_family_candidates": candidates,
        "fresh_confirmation_allowed": False,
    }


def apply_outer_policy(
    rows: Sequence[Mapping[str, Any]], inner_selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if inner_selection.get("enabled") is not True:
        # An ineligible inner guard must be the deterministic fallback, not an
        # unregistered threshold chosen after seeing outer labels.
        return [_row_decision(row, math.inf) for row in rows]
    margin = float(inner_selection["gain_margin"])
    if margin not in GUARD_MARGIN_THRESHOLDS:
        raise RuntimeError("outer policy received an unregistered inner threshold")
    return [_row_decision(row, margin) for row in rows]


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "EXPECTED_GROUPS",
    "FORMAL_SEMANTIC_DIM",
    "FORMAL_TRAINABLE_PARAMETER_COUNT",
    "FORMAT",
    "GUARD_MARGIN_THRESHOLDS",
    "INNER_FOLDS",
    "OUTER_FOLDS",
    "SELECTION_FORMAT",
    "TRAINABLE_PARAMETER_NAMES",
    "apply_outer_policy",
    "canonical_sha256",
    "evaluate_guard",
    "make_nested_oof_manifest",
    "select_inner_guard",
    "validate_nested_oof_manifest",
]
