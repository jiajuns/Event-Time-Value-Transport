from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import calibrate_smolvla_piper_adapter_ensemble as ensemble  # noqa: E402
from smolvla_piper_paired_success_protocol import (  # noqa: E402
    file_sha256,
    validate_dependency_receipt,
)


def source_rank_member_authority(
    temperatures: list[float] | None = None,
) -> tuple[dict, str]:
    values = list(temperatures or [1.0] * 5)
    authority = {
        "source_rank_numeric_contract": ensemble.SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": index,
                "source_checkpoint_file_sha256": f"{index + 5:x}" * 64,
                "source_rank_score_contract_sha256": f"{index + 10:x}" * 64,
                "success_temperature": values[index],
            }
            for index in range(5)
        ],
    }
    return authority, ensemble.canonical_sha256(authority)


DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY, DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA = (
    source_rank_member_authority()
)


def synthetic_arrays(groups: int = 190, classes: int = 3):
    rng = np.random.default_rng(19)
    candidates = 4
    n = groups * candidates
    group_number = np.repeat(np.arange(groups), candidates)
    candidate = np.tile(np.arange(candidates), groups)
    sample_id = np.asarray([f"validation-sample-{index:04d}" for index in range(n)])
    group_id = np.asarray([f"group-{index:04d}" for index in group_number])
    # Every formal candidate is evaluated at the same initial pre-action e0
    # root.  Future post-event labels still span the reachable event classes.
    current = np.zeros(n, dtype=np.int64)
    post = (group_number + candidate) % classes
    nxt = 1 + ((group_number + candidate) % (classes - 1))
    baseline_success = group_number % 2
    success = np.select(
        [candidate == 0, candidate == 1, candidate == 2],
        [baseline_success, np.ones(n, dtype=int), (group_number % 3 != 0)],
        default=0,
    ).astype(np.int64)
    regress = np.ones(n, dtype=bool)
    recovery = ((group_number + candidate) % 2).astype(np.int64)
    recovery_observed = regress.copy()
    duration = (
        1.5 + current * 0.6 + (group_number % 5 == 0).astype(float) * 0.8
    )
    duration_observed = group_number % 4 != 0
    object_observed = np.ones(n, dtype=bool)
    object_target = np.zeros((n, 3), dtype=np.float64)
    nonbaseline = candidate != 0
    object_target[nonbaseline, 0] = 0.2 + 0.03 * candidate[nonbaseline]
    object_target[nonbaseline, 1] = -0.1 - 0.01 * (group_number[nonbaseline] % 5)
    object_target[nonbaseline, 2] = 0.02 * (group_number[nonbaseline] % 3)
    labels = {
        "sample_id": sample_id,
        "group_id": group_id,
        "group_row_ordinal": candidate.astype(np.int64),
        "current_event": current.astype(np.int64),
        "post_event": post.astype(np.int64),
        "next_event": nxt.astype(np.int64),
        "success": success.astype(np.int64),
        "regress": regress,
        "recovery": recovery.astype(np.int64),
        "recovery_observed": recovery_observed,
        "duration": duration.astype(np.float64),
        "duration_observed": duration_observed,
        "object_target": object_target,
        "object_observed": object_observed,
        "root_candidate": np.ones(n, dtype=bool),
        "candidate_index": candidate.astype(np.int64),
        "is_baseline": candidate == 0,
        "candidate_final_success": success.copy(),
        "prediction_contract": {"recovery_head_trained": True},
    }
    members = []
    severity = ((group_number * 37) % groups) / max(groups - 1, 1)
    difficult = severity > 0.90
    for member in range(5):
        post_logits = np.full((n, classes), -2.5)
        next_logits = np.full((n, classes), -2.5)
        post_logits[np.arange(n), post] = 3.2 + 0.05 * member
        next_logits[np.arange(n), nxt] = 3.4 - 0.04 * member
        post_logits[difficult] = -0.05
        next_logits[difficult] = -0.05
        wrong_post = (post[difficult] + 1) % classes
        wrong_next = 1 + (nxt[difficult] % (classes - 1))
        post_logits[np.flatnonzero(difficult), wrong_post] = 0.05
        next_logits[np.flatnonzero(difficult), wrong_next] = 0.05
        signed_success = (2 * success - 1).astype(float)
        signed_recovery = (2 * recovery - 1).astype(float)
        success_logit = signed_success * (3.0 + 0.12 * member)
        recovery_logit = signed_recovery * (2.8 + 0.1 * member)
        success_logit[difficult] = -signed_success[difficult] * 0.15
        recovery_logit[difficult] = -signed_recovery[difficult] * 0.12
        source_contract_base_rank_score = (
            np.float32(0.4) * success.astype(np.float32)
        )
        source_action_rank_residual = np.select(
            [candidate == 0, candidate == 1, candidate == 2],
            [0.0, 1.2 + 0.01 * member, 0.3],
            default=-0.2,
        ).astype(np.float32)
        source_contract_rank_score = (
            source_contract_base_rank_score
            + source_action_rank_residual / np.float32(1.0)
        )
        duration_error = 0.002 * severity + np.where(difficult, 0.158, 0.0)
        object_error = np.where(difficult[:, None], 0.08, 0.004)
        row = {
            "sample_id": sample_id.copy(),
            "post_event_logits": post_logits + rng.normal(0, 0.04, post_logits.shape),
            "next_event_logits": next_logits + rng.normal(0, 0.04, next_logits.shape),
            "success_logit": success_logit + rng.normal(0, 0.025, n),
            "source_contract_base_rank_score": source_contract_base_rank_score,
            "source_action_rank_residual": source_action_rank_residual,
            "source_contract_rank_score": source_contract_rank_score,
            "recovery_logit": recovery_logit + rng.normal(0, 0.025, n),
            "duration_log_mean": np.log1p(duration + duration_error)
            + rng.normal(0, 0.0002, n),
            "duration_log_scale": np.log(
                0.002 + 0.004 * severity + np.where(difficult, 0.002, 0.0)
                + member * 0.00002
            ),
            "object_mean": object_target + object_error
            + rng.normal(0, 0.001, object_target.shape),
            "object_log_scale": np.log(
                np.where(difficult[:, None], 0.10, 0.012)
                + np.zeros_like(object_target) + member * 0.0005
            ),
        }
        members.append(row)
    return members, labels


def root_only_calibration(
    predictions: list[dict],
    labels: dict,
    *,
    uncertainty: np.ndarray | None = None,
    quality: np.ndarray | None = None,
) -> dict:
    root = labels["root_candidate"].astype(bool)
    rank, base, residual = ensemble._source_rank_float32_arrays(
        predictions,
        root,
        DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
        DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
    )
    n = len(labels["group_id"])
    crossfit_contract = {
        "format": "etsf_smolvla_piper_formal190_complete_root_outer_nesting_v1",
        "status": "complete_outer_heldout_isolation",
        "upstream_predictions_already_group_crossfit": True,
        "outer_crossfit_folds": 5,
        "outer_heldout_labels_used_for_any_parameter_or_selection": False,
        "complete_root_pipeline_outer_nesting": True,
    }
    crossfit_contract["root_outer_nesting_contract_sha256"] = (
        ensemble.canonical_sha256(crossfit_contract)
    )
    return ensemble.calibrate_root_group_ranker(
        rank,
        base,
        residual,
        np.full(n, 0.1, dtype=np.float64)
        if uncertainty is None
        else uncertainty,
        labels,
        source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
        source_rank_member_authority_sha256=(
            DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA
        ),
        upstream_six_head_gate_passed=True,
        global_abstain_threshold_enabled=True,
        maximum_global_candidate_uncertainty=1.0,
        global_quality=(
            np.ones(n, dtype=np.float64) if quality is None else quality
        ),
        outer_fold_structured_uncertainty=np.tile(
            (
                np.full(n, 0.1, dtype=np.float64)
                if uncertainty is None
                else uncertainty
            ),
            (ensemble.CROSSFIT_FOLDS, 1),
        ),
        outer_fold_training_quality=np.tile(
            np.ones(n, dtype=np.float64) if quality is None else quality,
            (ensemble.CROSSFIT_FOLDS, 1),
        ),
        root_outer_nesting_contract=crossfit_contract,
        bootstrap_samples=100,
    )


def set_root_outcomes_equal_to_baseline(labels: dict) -> None:
    groups = labels["group_id"].astype(str)
    baseline = labels["is_baseline"].astype(bool)
    outcomes = labels["candidate_final_success"]
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        baseline_outcome = int(outcomes[rows[baseline[rows]]][0])
        outcomes[rows] = baseline_outcome


def set_fold_pseudocorrelation_case(
    predictions: list[dict], labels: dict,
) -> None:
    groups = labels["group_id"].astype(str)
    candidate = labels["candidate_index"].astype(int)
    unique = np.asarray(sorted(set(groups.tolist())))
    folds = ensemble._logical_group_folds(unique)
    fold_by_group = dict(zip(unique.tolist(), folds.tolist()))
    ordinal_in_fold: dict[str, int] = {}
    for fold in range(ensemble.CROSSFIT_FOLDS):
        names = unique[folds == fold]
        ordinal_in_fold.update(
            {name: index for index, name in enumerate(names.tolist())}
        )
    high_margin = np.asarray(
        [ordinal_in_fold[group] < 10 for group in groups], dtype=bool
    )
    fold_zero_low = np.asarray(
        [
            fold_by_group[group] == 0 and not high_margin[index]
            for index, group in enumerate(groups)
        ],
        dtype=bool,
    )
    outcomes = labels["candidate_final_success"]
    for group in unique:
        rows = np.flatnonzero(groups == group)
        is_harmful = bool(fold_zero_low[rows[0]])
        baseline_outcome = 1 if is_harmful else 0
        outcomes[rows] = baseline_outcome
        outcomes[rows[candidate[rows] == 1]] = 0 if is_harmful else 1
    for member in predictions:
        member["source_contract_base_rank_score"][:] = np.float32(0.0)
        residual = member["source_action_rank_residual"]
        residual[:] = np.float32(-0.2)
        residual[candidate == 0] = np.float32(0.0)
        residual[(candidate == 1) & high_margin] = np.float32(1.2)
        residual[(candidate == 1) & ~high_margin] = np.float32(0.2)
        member["source_contract_rank_score"] = (
            member["source_contract_base_rank_score"]
            + residual / np.float32(1.0)
        )


def test_all_heads_metrics_uncertainty_and_group_lcb_threshold() -> None:
    predictions, labels = synthetic_arrays()
    calibration, support = ensemble.calibrate_arrays(
        predictions,
        labels,
        source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
        source_rank_member_authority_sha256=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
        bootstrap_samples=400,
    )
    assert all(
        row["enabled_for_primary"] for row in support["heads"].values()
    )
    assert calibration["metrics"]["post_event"]["performance_gate_passed"] is True
    assert calibration["metrics"]["next_event"]["performance_gate_passed"] is True
    assert calibration["metrics"]["success"]["performance_gate_passed"] is True
    assert calibration["metrics"]["success"]["deployment_temperature"] > 0
    assert calibration["metrics"]["conditional_recovery"]["performance_gate_passed"] is True
    assert calibration["head_enabled_for_primary"]["recovery"] is True
    assert calibration["metrics"]["duration_lognormal_mixture"]["performance_gate_passed"] is True
    assert calibration["metrics"]["object_total_variance"]["performance_gate_passed"] is True
    assert calibration["all_six_heads_support_performance_uncertainty_gate_passed"] is True
    ranker = calibration["root_group_ranker"]
    assert calibration["source_rank_numeric_contract"] == (
        ensemble.SOURCE_RANK_NUMERIC_CONTRACT
    )
    assert ranker["source_rank_numeric_contract"] == (
        ensemble.SOURCE_RANK_NUMERIC_CONTRACT
    )
    assert ranker["source_rank_member_authority"] == (
        DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY
    )
    assert ranker["source_rank_member_authority_sha256"] == (
        DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA
    )
    assert ranker["enabled_for_primary"] is True
    assert ranker["maximum_harmful_rate_among_executed_changes"] == 0.10
    assert ranker["selected_candidate"]["paired_gain_group_bootstrap_lcb95"] > 0
    assert ranker["score_is_success_logit"] is False
    assert "minimum_group_relative_composite_rank_score_margin" in ranker[
        "selected_candidate"
    ]
    threshold = calibration["abstain_threshold"]
    assert threshold["bootstrap_unit"] == "validation_group"
    assert threshold["test_or_paired_outcomes_used_for_selection"] is False
    assert len(threshold["candidates"]) == len(ensemble.THRESHOLD_QUANTILES)


def test_selection_aware_root_oof_stable_gain_and_shared_draws() -> None:
    predictions, labels = synthetic_arrays()
    ranker = root_only_calibration(predictions, labels)
    evidence = ranker["selection_aware_oof_evidence"]
    assert ranker["enabled_for_primary"] is True
    assert evidence["passed_for_primary"] is True
    assert evidence["paired_gain_group_bootstrap_lcb95"] > 0.0
    assert evidence["harmful_rate_group_bootstrap_ucb95"] <= 0.10
    assert evidence["upstream_predictions_already_group_crossfit"] is True
    assert evidence["complete_temperature_scale_and_root_double_nesting"] is True
    assert evidence["shared_bootstrap_draws"] == ranker[
        "development_bootstrap_draws"
    ]
    assert evidence["selection_aware_oof_evidence_sha256"] == (
        ensemble.canonical_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "selection_aware_oof_evidence_sha256"
            }
        )
    )


def test_selection_aware_root_oof_null_grid_search_cannot_authorize() -> None:
    predictions, labels = synthetic_arrays()
    set_root_outcomes_equal_to_baseline(labels)
    ranker = root_only_calibration(predictions, labels)
    evidence = ranker["selection_aware_oof_evidence"]
    assert ranker["enabled_for_primary"] is False
    assert ranker["full_formal190_deployment_refit_candidate_available"] is False
    assert evidence["passed_for_primary"] is False
    assert evidence["discordant_group_count"] == 0


def test_fold_pseudocorrelation_full_refit_does_not_replace_oof_evidence() -> None:
    predictions, labels = synthetic_arrays()
    set_fold_pseudocorrelation_case(predictions, labels)
    ranker = root_only_calibration(predictions, labels)
    evidence = ranker["selection_aware_oof_evidence"]
    assert ranker["full_formal190_deployment_refit_candidate_available"] is True
    assert ranker["selected_candidate"][
        "minimum_group_relative_composite_rank_score_margin"
    ] >= 0.25
    assert evidence["harmful_group_count"] >= 20
    assert evidence["passed_for_primary"] is False
    assert ranker["enabled_for_primary"] is False


def test_heldout_labels_do_not_change_that_folds_training_parameters() -> None:
    predictions, labels = synthetic_arrays()
    original = root_only_calibration(predictions, labels)
    heldout_fold = 0
    row_folds = ensemble._logical_group_folds(labels["group_id"].astype(str))
    heldout_rows = row_folds == heldout_fold
    labels["candidate_final_success"][heldout_rows] = (
        1 - labels["candidate_final_success"][heldout_rows]
    )
    mutated = root_only_calibration(predictions, labels)
    original_fold = original["selection_aware_oof_evidence"]["outer_folds"][
        heldout_fold
    ]
    mutated_fold = mutated["selection_aware_oof_evidence"]["outer_folds"][
        heldout_fold
    ]
    assert original_fold["selected_training_candidate"] == mutated_fold[
        "selected_training_candidate"
    ]
    assert original_fold["training_global_abstain_threshold"] == mutated_fold[
        "training_global_abstain_threshold"
    ]
    assert original_fold["training_root_candidate_grid_sha256"] == mutated_fold[
        "training_root_candidate_grid_sha256"
    ]


def test_outer_heldout_labels_do_not_change_upstream_fold_parameters() -> None:
    predictions, labels = synthetic_arrays()

    def compute() -> tuple[np.ndarray, np.ndarray, dict]:
        return ensemble.outer_nested_root_calibration_inputs(
            post_logits=np.stack(
                [row["post_event_logits"] for row in predictions]
            ),
            next_logits=np.stack(
                [row["next_event_logits"] for row in predictions]
            ),
            success_logits=np.stack(
                [row["success_logit"] for row in predictions]
            ),
            duration_means=np.stack(
                [row["duration_log_mean"] for row in predictions]
            ),
            duration_log_scales=np.stack(
                [row["duration_log_scale"] for row in predictions]
            ),
            object_means=np.stack(
                [row["object_mean"] for row in predictions]
            ),
            object_log_scales=np.stack(
                [row["object_log_scale"] for row in predictions]
            ),
            labels=labels,
            all_six_head_gate_passed=True,
        )

    uncertainty_before, quality_before, contract_before = compute()
    fold = 0
    heldout = ensemble._logical_group_folds(labels["group_id"].astype(str)) == fold
    labels["post_event"][heldout] = (labels["post_event"][heldout] + 1) % 3
    labels["next_event"][heldout] = 1 + (labels["next_event"][heldout] % 2)
    labels["success"][heldout] = 1 - labels["success"][heldout]
    labels["duration"][heldout] += 17.0
    labels["object_target"][heldout] += np.asarray([9.0, -7.0, 5.0])
    uncertainty_after, quality_after, contract_after = compute()
    assert contract_before["fold_parameters"][fold] == contract_after[
        "fold_parameters"
    ][fold]
    assert np.array_equal(uncertainty_before[fold], uncertainty_after[fold])
    training = ~heldout
    assert np.array_equal(
        quality_before[fold, training], quality_after[fold, training]
    )
    assert contract_after[
        "outer_heldout_labels_used_for_any_parameter_or_selection"
    ] is False
    assert contract_after["complete_root_pipeline_outer_nesting"] is True


def test_each_formal_group_is_scored_once_in_oof_stitch() -> None:
    predictions, labels = synthetic_arrays()
    evidence = root_only_calibration(predictions, labels)[
        "selection_aware_oof_evidence"
    ]
    decisions = evidence["stitched_group_decisions"]
    identifiers = [row["logical_group_id"] for row in decisions]
    assert evidence["stitched_decision_count"] == 190
    assert evidence["unique_stitched_logical_group_count"] == 190
    assert evidence["every_formal_logical_group_scored_exactly_once"] is True
    assert len(identifiers) == len(set(identifiers)) == 190
    assert sorted(identifiers) == sorted(set(labels["group_id"].astype(str)))


def test_any_outer_fold_without_training_candidate_fails_closed() -> None:
    predictions, labels = synthetic_arrays()
    set_root_outcomes_equal_to_baseline(labels)
    ranker = root_only_calibration(predictions, labels)
    folds = ranker["selection_aware_oof_evidence"]["outer_folds"]
    assert any(row["selection_available"] is False for row in folds)
    assert ranker["selection_aware_oof_evidence"]["passed_for_primary"] is False
    assert ranker["enabled_for_primary"] is False


def test_source_rank_one_float32_ulp_and_dtype_promotion_fail_closed() -> None:
    predictions, labels = synthetic_arrays()
    root = int(np.flatnonzero(labels["root_candidate"])[0])
    predictions[0]["source_contract_rank_score"][root] = np.nextafter(
        predictions[0]["source_contract_rank_score"][root],
        np.float32(np.inf),
        dtype=np.float32,
    )
    with pytest.raises(ensemble.CalibrationError, match="float32 audit algebra"):
        ensemble.calibrate_arrays(
            predictions,
            labels,
            source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
            source_rank_member_authority_sha256=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
            bootstrap_samples=100,
        )

    predictions, labels = synthetic_arrays()
    predictions[0]["source_contract_base_rank_score"] = predictions[0][
        "source_contract_base_rank_score"
    ].astype(np.float64)
    with pytest.raises(ensemble.CalibrationError, match="float32 audit algebra"):
        ensemble.calibrate_arrays(
            predictions,
            labels,
            source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
            source_rank_member_authority_sha256=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
            bootstrap_samples=100,
        )


def test_raw_point_three_temperature_uses_float32_training_order() -> None:
    predictions, labels = synthetic_arrays()
    authority, authority_sha = source_rank_member_authority(
        [0.3, 1.0, 1.0, 1.0, 1.0]
    )
    base = predictions[0]["source_contract_base_rank_score"]
    residual = predictions[0]["source_action_rank_residual"]
    predictions[0]["source_contract_rank_score"] = (
        base + residual / np.float32(0.3)
    )
    ensemble._source_rank_float32_arrays(
        predictions,
        labels["root_candidate"],
        authority,
        authority_sha,
    )

    python_order = (
        base.astype(np.float64) + residual.astype(np.float64) / 0.3
    ).astype(np.float32)
    assert not np.array_equal(
        python_order, predictions[0]["source_contract_rank_score"]
    )
    predictions[0]["source_contract_rank_score"] = python_order
    with pytest.raises(ensemble.CalibrationError, match="float32 audit algebra"):
        ensemble._source_rank_float32_arrays(
            predictions,
            labels["root_candidate"],
            authority,
            authority_sha,
        )


@pytest.mark.parametrize(
    "mutation",
    ("member_order", "temperature", "contract_sha", "source_sha"),
)
def test_source_rank_member_authority_tamper_fails_closed(
    mutation: str,
) -> None:
    authority, authority_sha = source_rank_member_authority()
    authority = json.loads(json.dumps(authority))
    if mutation == "member_order":
        authority["members"][0], authority["members"][1] = (
            authority["members"][1], authority["members"][0]
        )
    elif mutation == "temperature":
        authority["members"][0]["success_temperature"] = 0.3
    elif mutation == "contract_sha":
        authority["members"][0]["source_rank_score_contract_sha256"] = "f" * 64
    else:
        authority["members"][0]["source_checkpoint_file_sha256"] = "f" * 64
    with pytest.raises(ensemble.CalibrationError, match="member authority"):
        ensemble.validate_source_rank_member_authority(authority, authority_sha)


def test_success_temperature_is_fixed_and_head_disabled_when_support_is_missing() -> None:
    predictions, labels = synthetic_arrays(groups=90)
    labels["success"][:] = 1
    for row in predictions:
        row["success_logit"][:] = 4.0
    calibration, support = ensemble.calibrate_arrays(
        predictions,
        labels,
        source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
        source_rank_member_authority_sha256=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
        bootstrap_samples=100,
    )
    assert support["heads"]["success"]["enabled_for_primary"] is False
    assert calibration["head_enabled_for_primary"]["success"] is False
    assert calibration["root_group_ranker"]["enabled_for_primary"] is False
    assert calibration["root_group_ranker"]["selection_aware_oof_evidence"][
        "upstream_predictions_already_group_crossfit"
    ] is False
    assert calibration["root_group_ranker"]["selection_aware_oof_evidence"][
        "passed_for_primary"
    ] is False
    success = calibration["metrics"]["success"]
    assert success["deployment_temperature"] == 1.0
    assert success["performance_gate_passed"] is False


def test_recovery_is_conditional_calibrated_and_fails_closed_without_support() -> None:
    predictions, labels = synthetic_arrays(groups=90)
    labels["recovery_observed"][:] = False
    labels["regress"][:] = False
    labels["recovery"][:] = 0
    calibration, support = ensemble.calibrate_arrays(
        predictions,
        labels,
        source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
        source_rank_member_authority_sha256=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
        bootstrap_samples=100,
    )
    assert support["heads"]["recovery"]["enabled_for_primary"] is False
    recovery = calibration["metrics"]["conditional_recovery"]
    assert recovery["deployment_temperature"] == 1.0
    assert recovery["observed_rows"] == 0
    assert recovery["crossfit_complete"] is False


def test_recovery_cannot_activate_when_checkpoint_head_was_not_trained() -> None:
    predictions, labels = synthetic_arrays(groups=90)
    labels["prediction_contract"] = {"recovery_head_trained": False}
    calibration, support = ensemble.calibrate_arrays(
        predictions,
        labels,
        source_rank_member_authority=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY,
        source_rank_member_authority_sha256=DEFAULT_SOURCE_RANK_MEMBER_AUTHORITY_SHA,
        bootstrap_samples=100,
    )
    assert support["heads"]["recovery"]["support_threshold_met"] is True
    assert support["heads"]["recovery"]["all_member_recovery_heads_trained"] is False
    assert calibration["head_enabled_for_primary"]["recovery"] is False


def test_fifty_groups_meet_support_only_when_both_sides_are_present() -> None:
    _predictions, labels = synthetic_arrays(groups=50)
    labels["duration_observed"] = (
        np.asarray([int(value.rsplit("-", 1)[1]) for value in labels["group_id"]])
        % 2
        == 0
    )
    support = ensemble.head_support(labels, event_classes=3)
    post = support["heads"]["post_event"]
    assert post["support_threshold_met"] is True
    assert post["enabled_for_primary"] is True
    assert post["minimum_required_per_side"] == 10
    assert support["heads"]["next_event"]["support_threshold_met"] is True
    assert support["heads"]["duration"]["support_threshold_met"] is True
    assert support["heads"]["duration"]["minimum_required_per_side"] == 10
    assert support["heads"]["success"]["support_threshold_met"] is True
    assert support["heads"]["success"]["minimum_required_per_side"] == 50
    assert support["heads"]["object_effect"]["support_threshold_met"] is True
    assert support["heads"]["object_effect"]["minimum_required_per_side"] == 50


def test_success_total_variance_decomposes_aleatoric_and_epistemic() -> None:
    logits = np.asarray(
        [[-2.0, 2.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]
    )
    target = np.asarray([0, 1])
    metrics, details = ensemble.fit_success_temperature(logits, target)
    probability = details["probability"]
    assert np.allclose(details["total"], probability * (1 - probability))
    assert metrics["mean_total_variance"] == pytest.approx(
        metrics["mean_aleatoric_variance"]
        + metrics["mean_epistemic_variance"]
    )


def test_sensitive_and_hdf_paths_fail_before_data_loading() -> None:
    with pytest.raises(ensemble.CalibrationError, match="sensitive"):
        ensemble.safe_existing(Path("/tmp/test_labels/labels.npz"), "labels")
    with pytest.raises(ensemble.CalibrationError, match="sensitive"):
        ensemble.safe_existing(Path("/tmp/Fresh50/labels.npz"), "labels")


def _make_authority(root: Path) -> tuple[Path, dict]:
    predictions, labels = synthetic_arrays()
    labels_path = root / "validation_labels.npz"
    np.savez(
        labels_path,
        **{key: value for key, value in labels.items() if key != "prediction_contract"},
    )
    prediction_contract = {
        "duration_target_transform": "log1p_decision_steps",
        "next_event_observation_mask": "duration_observed",
        "success_target": "eventual_final_branch_success_repeated_per_transition",
        "recovery_target": "conditional_recovery_given_operational_regress",
        "recovery_observation_mask": "recovery_observed_and_regress",
        "recovery_shared_transition_stop_gradient": True,
        "recovery_enters_primary_before_calibration": False,
        "recovery_head_trained": True,
        "object_prediction_space": "physical_delta_xyz_m",
        "object_source_normalization_sha256": "4" * 64,
        "object_observed_policy": "row_enabled_only_if_all_selected_xyz_are_valid",
    }
    shared = {
        "training_manifest_sha256": "1" * 64,
        "split_sha256": "2" * 64,
        "source_ensemble_contract_sha256": "3" * 64,
        "prediction_contract_sha256": ensemble.canonical_sha256(prediction_contract),
    }
    members = []
    for index, prediction in enumerate(predictions):
        checkpoint = root / f"adapter_member_{index}.pt"
        checkpoint.write_bytes(f"synthetic-checkpoint-{index}".encode())
        prediction_path = root / f"validation_prediction_member_{index}.npz"
        np.savez(prediction_path, **prediction)
        source_rank_contract = {
            "format": "etsf_source63_composite_candidate_rank_score_v1",
            "status": "frozen_exact_source63_training_score_scientific_rank_only",
            "source_checkpoint_file_sha256": f"{index + 5:x}" * 64,
            "source_action_rank_residual": True,
            "source_action_rank_success_only": False,
            "source_freeze_factual_core": False,
            "base_score": "candidate_rank_score",
            "event_names": ["e0", "e12", "e3", "e4", "eK"],
            "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "event_values_authority": "source_trainer_linspace_0_1_in_checkpoint_event_order",
            "duration_scale": 200.0,
            "duration_scale_authority": "source_member_checkpoint.duration_scale",
            "duration_scale_scope": "per_source_member_not_ensemble_mean",
            "duration_unit": "decision_steps",
            "success_temperature": 1.0,
            "source_rank_numeric_contract": (
                ensemble.SOURCE_RANK_NUMERIC_CONTRACT
            ),
            "event_weight": 0.25,
            "duration_weight": 0.05,
            "residual_combination": "candidate_rank_score_plus_action_rank_residual",
            "score_variant": "source_member_training_objective_defaults",
            "source_ensemble_validation_selected_scoring_consumed": False,
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "cross_embodiment_duration_scale_calibrated": False,
            "deployment_success_probability_selector_authorized": False,
        }
        source_rank_contract["contract_sha256"] = ensemble.canonical_sha256(
            source_rank_contract
        )
        members.append(
            {
                "member_index": index,
                "member_seed": 20260828 + index,
                **shared,
                "checkpoint_path": str(checkpoint),
                "checkpoint_file_sha256": file_sha256(checkpoint),
                "validation_predictions_path": str(prediction_path),
                "validation_predictions_file_sha256": file_sha256(prediction_path),
                "source_rank_score_contract": source_rank_contract,
                "source_rank_score_contract_sha256": source_rank_contract[
                    "contract_sha256"
                ],
            }
        )
    authority = {
        "format": ensemble.INPUT_FORMAT,
        "status": ensemble.INPUT_STATUS,
        "lane": "validation_only",
        "member_count": 5,
        "shared_contract": shared,
        "prediction_contract": prediction_contract,
        "source_rank_numeric_contract": ensemble.SOURCE_RANK_NUMERIC_CONTRACT,
        "validation_identity_set_sha256": ensemble.canonical_sha256(
            labels["sample_id"].astype(str).tolist()
        ),
        "labels_path": str(labels_path),
        "labels_file_sha256": file_sha256(labels_path),
        "members": members,
        "test_artifacts_read": False,
        "fresh_artifacts_read": False,
        "confirmation_artifacts_read": False,
    }
    authority["input_authority_sha256"] = ensemble.canonical_sha256(authority)
    authority_path = root / "validation_input_authority.json"
    authority_path.write_text(
        json.dumps(authority, sort_keys=True) + "\n", encoding="utf-8"
    )
    return authority_path, authority


def _make_writable(root: Path) -> None:
    for directory, names, files in os.walk(root):
        Path(directory).chmod(0o755)
        for name in names:
            (Path(directory) / name).chmod(0o755)
        for name in files:
            (Path(directory) / name).chmod(0o644)


def test_end_to_end_receipt_is_paired_dependency_compatible() -> None:
    root = Path(tempfile.mkdtemp(prefix="adapter_ensemble_validation_", dir="/tmp"))
    try:
        authority_path, _authority = _make_authority(root)
        output = root / "calibrated_output"
        receipt = ensemble.run(
            authority_path,
            file_sha256(authority_path),
            output,
            bootstrap_samples=300,
        )
        assert receipt["status"] == ensemble.RECEIPT_STATUS
        assert receipt["test_artifacts_read"] is False
        assert receipt["artifacts_frozen_read_only"] is True
        assert receipt["source_rank_numeric_contract"] == (
            ensemble.SOURCE_RANK_NUMERIC_CONTRACT
        )
        assert receipt["source_rank_member_authority_sha256"] == (
            ensemble.canonical_sha256(receipt["source_rank_member_authority"])
        )
        assert json.loads(
            (output / "ensemble_manifest.json").read_text(encoding="utf-8")
        )["source_rank_numeric_contract"] == ensemble.SOURCE_RANK_NUMERIC_CONTRACT
        dependency_spec = {
            "name": "adapter",
            "receipt_path": str(output / "final_receipt.json"),
            "receipt_file_sha256": file_sha256(output / "final_receipt.json"),
            "expected_format": ensemble.RECEIPT_FORMAT,
            "expected_status": ensemble.RECEIPT_STATUS,
            "logical_sha256_field": "receipt_sha256",
            "required_fields": {
                "member_count": 5,
                "validation_only": True,
                "test_artifacts_read": False,
                "test_hdf5_files_opened": 0,
                "fresh_paths_accepted": False,
                "paired_development_outcomes_read": False,
                "performance_or_transfer_claim_authorized": False,
                "artifacts_frozen_read_only": True,
            },
            "run_exit_path": str(output / "run.exit"),
            "run_exit_file_sha256": file_sha256(output / "run.exit"),
        }
        audit = validate_dependency_receipt(dependency_spec)
        assert audit["status"] == ensemble.RECEIPT_STATUS
        assert (output / "run.exit").read_bytes() == b"0\n"
        assert (output / "final_receipt.json").stat().st_mode & 0o222 == 0
    finally:
        if root.exists():
            _make_writable(root)
            shutil.rmtree(root)


def test_member_with_different_split_contract_is_rejected() -> None:
    root = Path(tempfile.mkdtemp(prefix="adapter_contract_validation_", dir="/tmp"))
    try:
        _path, authority = _make_authority(root)
        authority["members"][3]["split_sha256"] = "9" * 64
        authority["input_authority_sha256"] = ensemble.canonical_sha256(
            {
                key: value
                for key, value in authority.items()
                if key != "input_authority_sha256"
            }
        )
        with pytest.raises(ensemble.CalibrationError, match="do not share"):
            ensemble.validate_input_authority(authority)
    finally:
        if root.exists():
            _make_writable(root)
            shutil.rmtree(root)
