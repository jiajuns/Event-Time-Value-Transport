#!/usr/bin/env python3
"""Frozen five-fold OOF development protocol for counterfactual ETSF.

This module contains only deterministic split, raw-prediction reduction and
development-authorization logic.  It does not collect data or start training.
All candidate groups are development data under this protocol; the frozen
manifest carries either the legacy 100-group or expanded 250-group dimensions.
The fresh-50 confirmation registry is absent from every input and output.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from train_openvla_etsf_counterfactual import (
    fit_success_temperature,
    predefined_scoring_grid,
    select_validation_scoring,
    tune_guard,
)


FORMAT = "etsf_counterfactual_five_fold_oof_v1"
SELECTION_FORMAT = "etsf_counterfactual_oof_selection_v1"
EXPECTED_GROUPS = 100
FOLD_COUNT = 5
GROUPS_PER_FOLD = 20
SUPPORTED_EXPECTED_GROUPS = (100, 250)
OOF_SPLIT_SEED = 20260827
MEMBER_SEEDS = (20260827, 20260828, 20260829)
# Frozen before any expanded OOF prediction is produced.  Retry2b peaked at
# 100 steps on 100 groups; expanded training scales steps with group count so
# each example receives the same expected number of optimizer exposures.  No
# heldout label selects the step count.  Fresh-50 remains confirmation-only.
FIXED_TRAINING_STEPS = 100
TRAINING_STEPS_BY_GROUPS = {100: 100, 250: 250}
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_SAMPLES = 20_000
FAMILYWISE_ALPHA = 0.05
MINIMUM_ORACLE_HEADROOM_GROUPS = 10
MINIMUM_GUARDED_CHANGES = 10
MINIMUM_HELPFUL_CHANGES = 5
DEPLOYMENT_CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)
TRAINING_ONLY_EXTRA_CANDIDATES = ("sample_blend_1.000",)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def oof_dimensions(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return frozen (total, train-per-fold, holdout-per-fold) dimensions."""

    total = int(manifest.get("expected_groups", -1))
    fold_count = int(manifest.get("fold_count", -1))
    holdout = int(manifest.get("groups_per_fold", -1))
    if total not in SUPPORTED_EXPECTED_GROUPS:
        raise RuntimeError(
            f"OOF expected_groups must be one of {SUPPORTED_EXPECTED_GROUPS}"
        )
    if fold_count != FOLD_COUNT or holdout <= 0 or holdout * fold_count != total:
        raise RuntimeError("OOF manifest dimensions are not balanced five-fold")
    return total, total - holdout, holdout


def oof_training_steps(manifest: Mapping[str, Any]) -> int:
    """Validate the preregistered constant-exposure training budget."""

    total, _, _ = oof_dimensions(manifest)
    expected = TRAINING_STEPS_BY_GROUPS[total]
    recorded = int(manifest.get("training_steps", -1))
    if recorded != expected:
        raise RuntimeError(
            f"OOF training steps changed: expected {expected} for {total} groups"
        )
    return expected


def make_oof_folds(
    logical_keys: Sequence[str],
    *,
    split_seed: int = OOF_SPLIT_SEED,
    source_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create five balanced folds using identity strings only."""

    keys = sorted(map(str, logical_keys))
    expected_groups = len(keys)
    if (
        expected_groups not in SUPPORTED_EXPECTED_GROUPS
        or len(set(keys)) != expected_groups
    ):
        raise RuntimeError(
            "formal OOF requires exactly 100 or 250 unique logical groups"
        )
    groups_per_fold = expected_groups // FOLD_COUNT
    ordered = sorted(
        keys,
        key=lambda key: hashlib.sha256(
            f"{split_seed}|{key}".encode("utf-8")
        ).hexdigest(),
    )
    folds = []
    for fold_id in range(FOLD_COUNT):
        holdout = sorted(
            ordered[
                fold_id * groups_per_fold : (fold_id + 1) * groups_per_fold
            ]
        )
        holdout_set = set(holdout)
        train = [key for key in keys if key not in holdout_set]
        folds.append(
            {
                "fold_id": fold_id,
                "training_groups": train,
                "oof_holdout_groups": holdout,
                "training_group_count": len(train),
                "oof_holdout_group_count": len(holdout),
                "checkpoint_selection": "fixed_final_step_no_holdout_early_stop",
            }
        )
    payload = {
        "format": FORMAT,
        "status": "preregistered",
        "split_algorithm": "sha256_sort_contiguous_equal_folds_v1",
        "split_seed": split_seed,
        "expected_groups": expected_groups,
        "fold_count": FOLD_COUNT,
        "groups_per_fold": groups_per_fold,
        "member_seeds": list(MEMBER_SEEDS),
        "training_steps": TRAINING_STEPS_BY_GROUPS[expected_groups],
        "training_step_rationale": (
            "constant_group_exposure_scaled_from_pre_oof_retry2b_"
            "100groups_100steps_before_any_expanded_oof_prediction"
        ),
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "groups_per_batch": 8,
            "gradient_clip": 2.0,
            "amp": "bf16",
            "device": "cuda",
            "unfreeze_semantic": False,
            "loss_weights": {
                "success": 1.0,
                "outcome": 0.2,
                "pairwise": 0.75,
                "listwise": 0.5,
                "group_centered": 1.0,
                "baseline_contrast": 1.5,
                "event": 1.0,
                "relative": 0.5,
                "destination": 0.5,
                "predicate": 0.5,
                "reach": 0.75,
                "duration": 0.5,
                "object": 0.5,
                "latent": 0.5,
            },
            "scoring_grid": "validation_scoring_grid_v1_fixed_7",
            "guard_grid": "validation_guard_quantile_grid_v1_max_9",
            "distance_weight": 0.02,
            "minimum_proposals": 10,
            "minimum_coverage": 0.10,
            "minimum_lcb90": 0.0,
            "maximum_harmful_rate": 0.10,
            "minimum_oracle_headroom_groups": MINIMUM_ORACLE_HEADROOM_GROUPS,
            "minimum_guarded_changes": MINIMUM_GUARDED_CHANGES,
            "minimum_helpful_changes": MINIMUM_HELPFUL_CHANGES,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "development_groups": keys,
        "source_contract": dict(source_contract or {}),
        "folds": folds,
        "fresh_confirmation": {
            "access": "forbidden_during_oof_and_refit",
            "required_registry": "explicit_fresh_confirmation",
            "required_group_count": 50,
            "one_shot_only_after_oof_authorization": True,
        },
    }
    payload["preregistration_sha256"] = canonical_sha256(payload)
    validate_oof_folds(payload, keys)
    return payload


def validate_oof_folds(
    manifest: Mapping[str, Any], logical_keys: Sequence[str]
) -> dict[str, int]:
    if manifest.get("format") != FORMAT:
        raise RuntimeError("unsupported OOF fold manifest format")
    expected_groups, training_groups, groups_per_fold = oof_dimensions(manifest)
    oof_training_steps(manifest)
    keys = set(map(str, logical_keys))
    if len(keys) != expected_groups:
        raise RuntimeError("OOF source group count changed")
    recorded = manifest.get("development_groups")
    if not isinstance(recorded, list) or set(map(str, recorded)) != keys:
        raise RuntimeError("OOF development group identities changed")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != FOLD_COUNT:
        raise RuntimeError("OOF fold manifest is incomplete")
    holdout_owner: dict[str, int] = {}
    for expected_id, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or int(fold.get("fold_id", -1)) != expected_id:
            raise RuntimeError("OOF fold ids/order changed")
        train = set(map(str, fold.get("training_groups", [])))
        holdout = set(map(str, fold.get("oof_holdout_groups", [])))
        if len(train) != training_groups or len(holdout) != groups_per_fold:
            raise RuntimeError(
                "OOF fold group counts differ from frozen manifest dimensions"
            )
        if int(fold.get("training_group_count", -1)) != training_groups or int(
            fold.get("oof_holdout_group_count", -1)
        ) != groups_per_fold:
            raise RuntimeError("OOF fold count mirrors changed")
        if train & holdout or train | holdout != keys:
            raise RuntimeError("OOF train/holdout leakage or omission")
        if fold.get("checkpoint_selection") != (
            "fixed_final_step_no_holdout_early_stop"
        ):
            raise RuntimeError("OOF heldout labels may not select a checkpoint")
        for key in holdout:
            if key in holdout_owner:
                raise RuntimeError("logical group appears in multiple OOF holdouts")
            holdout_owner[key] = expected_id
    if set(holdout_owner) != keys:
        raise RuntimeError("not every development group has exactly one OOF prediction")
    expected_digest = manifest.get("preregistration_sha256")
    if expected_digest:
        unsigned = dict(manifest)
        unsigned.pop("preregistration_sha256", None)
        if str(expected_digest) != canonical_sha256(unsigned):
            raise RuntimeError("OOF preregistration payload changed")
    return {"development_groups": len(keys), "unique_oof_groups": len(holdout_owner)}


def _validate_raw_rows(
    raw_rows: Sequence[Mapping[str, Any]], fold_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validate_oof_folds(fold_manifest, fold_manifest["development_groups"])
    owner = {
        str(key): int(fold["fold_id"])
        for fold in fold_manifest["folds"]
        for key in fold["oof_holdout_groups"]
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        row = dict(raw)
        key = str(row.get("logical_key", ""))
        if key not in owner or key in seen:
            raise RuntimeError("OOF raw prediction key is unknown or duplicated")
        if int(row.get("fold_id", -1)) != owner[key]:
            raise RuntimeError("OOF prediction was not produced by its heldout fold")
        logits = np.asarray(row.get("member_success_logits"), dtype=np.float64)
        event = np.asarray(row.get("member_event_progress"), dtype=np.float64)
        duration = np.asarray(row.get("member_normalized_duration"), dtype=np.float64)
        aleatoric = np.asarray(row.get("member_aleatoric"), dtype=np.float64)
        success = np.asarray(row.get("success"), dtype=np.float64)
        distance = np.asarray(row.get("candidate_distance"), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != len(MEMBER_SEEDS):
            raise RuntimeError("OOF row must contain three member predictions")
        if event.shape != logits.shape or duration.shape != logits.shape:
            raise RuntimeError("OOF score components are misaligned")
        if aleatoric.shape != logits.shape:
            raise RuntimeError("OOF uncertainty components are misaligned")
        candidate_count = logits.shape[1]
        if success.shape != (candidate_count,) or distance.shape != (candidate_count,):
            raise RuntimeError("OOF labels/distances are misaligned")
        baseline = int(row.get("baseline_index", -1))
        if baseline != 0:
            raise RuntimeError("OOF baseline index is invalid")
        if not all(
            np.isfinite(value).all()
            for value in (logits, event, duration, aleatoric, success, distance)
        ):
            raise RuntimeError("OOF raw predictions contain non-finite values")
        if np.any((success < 0) | (success > 1)):
            raise RuntimeError("OOF success labels must be binary probabilities")
        names = row.get("candidate_names")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise RuntimeError("OOF row lacks frozen candidate names")
        names = tuple(map(str, names))
        if len(names) != candidate_count or len(set(names)) != candidate_count:
            raise RuntimeError("OOF candidate names are misaligned or duplicated")
        if names[: len(DEPLOYMENT_CANDIDATE_NAMES)] != DEPLOYMENT_CANDIDATE_NAMES:
            raise RuntimeError("OOF deployment candidate names/order changed")
        extras = names[len(DEPLOYMENT_CANDIDATE_NAMES) :]
        if extras not in ((), TRAINING_ONLY_EXTRA_CANDIDATES):
            raise RuntimeError("OOF contains an unregistered candidate schedule")
        # The fifth expanded-development candidate is useful group supervision
        # but is absent from the preregistered fresh50 collector.  Calibration,
        # scoring and authorization must use the exact deployment schedule.
        deployment = np.arange(len(DEPLOYMENT_CANDIDATE_NAMES))
        row.update(
            {
                "member_success_logits": logits[:, deployment],
                "member_event_progress": event[:, deployment],
                "member_normalized_duration": duration[:, deployment],
                "member_aleatoric": aleatoric[:, deployment],
                "success": success[deployment],
                "candidate_distance": distance[deployment],
                "steps": np.asarray(
                    row.get("steps", np.zeros(candidate_count)), dtype=np.float64
                )[deployment],
                "baseline_index": 0,
                "candidate_names": list(DEPLOYMENT_CANDIDATE_NAMES),
                "training_only_candidates_excluded_from_authorization": list(extras),
            }
        )
        seen.add(key)
        rows.append(row)
    if seen != set(owner):
        raise RuntimeError(
            "OOF raw predictions do not cover every frozen group exactly once"
        )
    return sorted(rows, key=lambda row: str(row["logical_key"]))


def _exact_one_sided_sign_p(helpful: int, harmful: int) -> float:
    non_ties = helpful + harmful
    if non_ties == 0:
        return 1.0
    return float(
        sum(math.comb(non_ties, value) for value in range(helpful, non_ties + 1))
        / (2**non_ties)
    )


def _bootstrap_mean_ci(
    delta: np.ndarray,
    *,
    expected_groups: int = EXPECTED_GROUPS,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if delta.shape != (expected_groups,):
        raise RuntimeError("OOF bootstrap requires one unconditional delta per group")
    generator = np.random.default_rng(seed)
    # Chunking avoids a large temporary if the preregistered sample count grows.
    means = np.empty(samples, dtype=np.float64)
    offset = 0
    while offset < samples:
        count = min(1000, samples - offset)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        means[offset : offset + count] = delta[indices].mean(1)
        offset += count
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def reduce_oof_predictions(
    raw_rows: Sequence[Mapping[str, Any]],
    fold_manifest: Mapping[str, Any],
    *,
    distance_weight: float = 0.02,
) -> dict[str, Any]:
    """Fit OOF temperature and freeze scoring/guard development decisions."""

    rows = _validate_raw_rows(raw_rows, fold_manifest)
    expected_groups, training_groups, _ = oof_dimensions(fold_manifest)
    logits = np.concatenate(
        [row["member_success_logits"] for row in rows], axis=1
    )
    labels = np.concatenate([row["success"] for row in rows])
    calibration = fit_success_temperature(logits, labels)
    temperature = float(calibration["temperature"])
    specs = predefined_scoring_grid(distance_weight)
    candidate_rows = []
    for spec in specs:
        scored = []
        for row in rows:
            member_logits = row["member_success_logits"]
            probability = 1.0 / (1.0 + np.exp(-member_logits / temperature))
            member_score = (
                member_logits / temperature
                + float(spec["event_weight"]) * row["member_event_progress"]
                - float(spec["duration_weight"])
                * row["member_normalized_duration"]
                - float(spec["candidate_distance_weight"])
                * row["candidate_distance"][None]
            )
            scored.append(
                {
                    "logical_key": row["logical_key"],
                    "fold_id": row["fold_id"],
                    "success": row["success"].copy(),
                    "steps": np.asarray(row.get("steps", np.zeros_like(row["success"]))),
                    "baseline_index": row["baseline_index"],
                    "mean_score": member_score.mean(0),
                    "mean_success_probability": probability.mean(0),
                    "uncertainty": probability.std(0)
                    + row["member_aleatoric"].mean(0),
                }
            )
        candidate_rows.append((spec, scored))
    scoring, selected_rows, scoring_audit = select_validation_scoring(
        candidate_rows,
        minimum_proposals=10,
        minimum_coverage=0.10,
        minimum_lcb=0.0,
    )
    proposed_guard = tune_guard(
        selected_rows,
        min_guarded_groups=10,
        min_coverage=0.10,
        minimum_lcb=0.0,
        max_harmful_rate=0.10,
    )

    headroom = sum(
        int(row["success"].max() > row["success"][row["baseline_index"]])
        for row in selected_rows
    )
    unconditional_delta = np.zeros(expected_groups, dtype=np.float64)
    decision_rows = []
    for index, row in enumerate(selected_rows):
        baseline = int(row["baseline_index"])
        proposed = int(np.argmax(row["mean_score"]))
        selected = baseline
        if proposed_guard.get("enabled") is True and proposed != baseline:
            gain = float(row["mean_score"][proposed] - row["mean_score"][baseline])
            uncertainty = float(row["uncertainty"][proposed])
            if gain >= float(proposed_guard["gain_margin"]) and uncertainty <= float(
                proposed_guard["uncertainty_threshold"]
            ):
                selected = proposed
        delta = float(row["success"][selected] - row["success"][baseline])
        unconditional_delta[index] = delta
        decision_rows.append(
            {
                "logical_key": row["logical_key"],
                "fold_id": int(row["fold_id"]),
                "baseline_index": baseline,
                "selected_index": selected,
                "changed": selected != baseline,
                "success_delta": delta,
                "oracle_headroom": bool(
                    row["success"].max() > row["success"][baseline]
                ),
            }
        )
    changed = int(np.count_nonzero(unconditional_delta))
    # changed counts only outcome-changing decisions; guarded selection coverage
    # also includes ties and is read from the proposed guard.
    guarded = int(proposed_guard.get("guarded_groups", 0))
    helpful = int((unconditional_delta > 0).sum())
    harmful = int((unconditional_delta < 0).sum())
    harmful_rate = harmful / max(guarded, 1)
    exact_p = _exact_one_sided_sign_p(helpful, harmful)
    threshold_count = max(1, len(proposed_guard.get("threshold_candidates", [])))
    searched_hypotheses = len(specs) * threshold_count
    corrected_alpha = FAMILYWISE_ALPHA / searched_hypotheses
    bootstrap_low, bootstrap_high = _bootstrap_mean_ci(
        unconditional_delta, expected_groups=expected_groups
    )
    rejection_reasons = []
    if proposed_guard.get("enabled") is not True:
        rejection_reasons.append("base_guard_grid_found_no_eligible_threshold")
    if headroom < MINIMUM_ORACLE_HEADROOM_GROUPS:
        rejection_reasons.append("insufficient_total_oracle_headroom")
    if guarded < MINIMUM_GUARDED_CHANGES:
        rejection_reasons.append("insufficient_guarded_changes")
    if helpful < MINIMUM_HELPFUL_CHANGES:
        rejection_reasons.append("insufficient_helpful_changes")
    if harmful_rate > 0.10:
        rejection_reasons.append("harmful_rate_above_preregistered_maximum")
    if bootstrap_low <= 0.0:
        rejection_reasons.append("unconditional_bootstrap_ci_not_strictly_positive")
    if exact_p > corrected_alpha:
        rejection_reasons.append("exact_sign_p_fails_familywise_threshold")
    authorized = not rejection_reasons
    authorization = {
        "authorized": authorized,
        "evidence_tier": "development_oof_authorization_not_confirmation",
        "total_oof_groups": expected_groups,
        "oracle_headroom_groups": headroom,
        "guarded_groups": guarded,
        "outcome_changing_guarded_groups": changed,
        "helpful_changes": helpful,
        "harmful_changes": harmful,
        "harmful_rate_over_guarded": harmful_rate,
        "unconditional_mean_paired_success_delta": float(unconditional_delta.mean()),
        "unconditional_group_bootstrap_95_ci": [bootstrap_low, bootstrap_high],
        "exact_one_sided_sign_mcnemar_p": exact_p,
        "searched_hypotheses_upper_bound": searched_hypotheses,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_corrected_alpha": corrected_alpha,
        "rejection_reasons": rejection_reasons,
        "fresh_confirmation_allowed": authorized,
        "fresh_confirmation_policy": (
            "one_shot_fresh50_only" if authorized else "forbidden"
        ),
        "refit_shift_warning": (
            f"OOF models train on {training_groups} groups while final models "
            f"refit on {expected_groups}; "
            "temperature, score-gain and uncertainty transport are development "
            "assumptions tested only by the one-shot fresh50 confirmation"
        ),
    }
    frozen_guard = dict(proposed_guard)
    frozen_guard["oof_authorization"] = authorization
    if not authorized:
        frozen_guard["enabled"] = False
        frozen_guard["reason"] = "oof_development_evidence_gate_failed"
    result = {
        "format": SELECTION_FORMAT,
        "status": "complete",
        "oof_preregistration_sha256": fold_manifest["preregistration_sha256"],
        "oof_prediction_groups": expected_groups,
        "success_calibration": calibration,
        "scoring": {
            "candidate_id": scoring["candidate_id"],
            "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "event_weight": scoring["event_weight"],
            "duration_weight": scoring["duration_weight"],
            "candidate_distance_weight": scoring["candidate_distance_weight"],
            "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
        },
        "scoring_selection": scoring_audit,
        "guard": frozen_guard,
        "authorization": authorization,
        "decision_audit": decision_rows,
        "fresh_confirmation_labels_read": False,
        "candidate_authorization_contract": {
            "deployment_candidate_names": list(DEPLOYMENT_CANDIDATE_NAMES),
            "training_only_extra_candidates": list(TRAINING_ONLY_EXTRA_CANDIDATES),
            "calibration_scoring_guard_use_deployment_candidates_only": True,
        },
    }
    result["selection_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "DEPLOYMENT_CANDIDATE_NAMES",
    "EXPECTED_GROUPS",
    "FIXED_TRAINING_STEPS",
    "FOLD_COUNT",
    "FORMAT",
    "MEMBER_SEEDS",
    "SELECTION_FORMAT",
    "SUPPORTED_EXPECTED_GROUPS",
    "TRAINING_STEPS_BY_GROUPS",
    "canonical_sha256",
    "make_oof_folds",
    "oof_dimensions",
    "oof_training_steps",
    "reduce_oof_predictions",
    "validate_oof_folds",
]
