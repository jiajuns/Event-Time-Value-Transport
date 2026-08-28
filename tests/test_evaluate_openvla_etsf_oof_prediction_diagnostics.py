from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_openvla_etsf_oof_prediction_diagnostics import (  # noqa: E402
    FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS,
    FORMAT,
    RECORDED_WEIGHTED_SUCCESS_STATUS,
    STRUCTURED_ROW_FORMAT,
    SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT,
    _crossfit_event_body_median,
    _uncertainty_diagnostic,
    build_oof_prediction_diagnostics,
    validate_oof_prediction_diagnostics,
)
from openvla_etsf_counterfactual_oof import (  # noqa: E402
    canonical_sha256,
    make_oof_folds,
)


def _fixture_rows(*, structured: bool = True) -> tuple[dict, list[dict]]:
    keys = [f"move_can_pot|piper|{10000 + index}" for index in range(100)]
    manifest = make_oof_folds(keys)
    owner = {
        key: int(fold["fold_id"])
        for fold in manifest["folds"]
        for key in fold["oof_holdout_groups"]
    }
    rows: list[dict] = []
    candidate_names = [
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
    ]
    for index, key in enumerate(sorted(keys)):
        success = float(index % 2)
        success_labels = np.asarray([success, 1.0 - success, success, 0.0])
        success_logits = np.repeat(
            (success_labels * 8.0 - 4.0)[None], 3, axis=0
        )
        row = {
            "logical_key": key,
            "fold_id": owner[key],
            "member_success_logits": success_logits,
            "member_event_progress": np.zeros((3, 4), dtype=np.float64),
            "member_normalized_duration": np.zeros((3, 4), dtype=np.float64),
            "member_aleatoric": np.full((3, 4), 0.02, dtype=np.float64),
            "success": success_labels,
            "candidate_distance": np.asarray([0.0, 0.2, 0.4, 0.6]),
            "baseline_index": 0,
            "candidate_names": list(candidate_names),
        }
        if structured:
            current_event = np.asarray([0, 1, 2, 3, 1])
            next_event = np.asarray([1, 1, 2, 4, 2])
            event_logits = np.full((3, 5, 5), -5.0)
            event_logits[:, np.arange(5), next_event] = 5.0
            reached_logits = event_logits.copy()
            duration = np.asarray([2.0, 3.0, 4.0, 5.0, 3.0])
            duration_mean = np.asarray(
                [np.log1p(2.0), np.log1p(6.0), np.log1p(4.0), np.log1p(5.0), np.log1p(6.0)]
            )
            duration_mean = np.repeat(duration_mean[None], 3, axis=0)
            duration_scale = np.full((3, 5), np.log(0.2))
            object_target = np.asarray(
                [
                    [0.1, -0.2, 0.3],
                    [0.0, 0.2, -0.1],
                    [0.2, 0.0, 0.1],
                    [-0.1, 0.1, 0.0],
                    [0.05, -0.1, 0.2],
                ],
                dtype=np.float64,
            )
            object_mean = np.repeat(object_target[None], 3, axis=0)
            object_scale = np.full((3, 5, 3), np.log(0.1))
            outcome_logits = np.full((3, 5, 3), -4.0)
            structured_success = np.concatenate([success_labels, [0.0]])
            outcome_logits[:, np.arange(5), structured_success.astype(int)] = 4.0
            predicates = np.asarray(
                [
                    [1, 0, 0, 0, success],
                    [1, 1, 0, 0, 1.0 - success],
                    [1, 1, 1, 0, success],
                    [1, 1, 1, 1, 0],
                    [1, 1, 0, 0, 0],
                ],
                dtype=np.float64,
            )
            predicate_logits = np.where(predicates > 0.5, 5.0, -5.0)
            predicate_logits = np.repeat(predicate_logits[None], 3, axis=0)
            row["structured_predictions"] = {
                "format": STRUCTURED_ROW_FORMAT,
                "sample_names": [*candidate_names, "continuation_0"],
                "terminal_mask": np.asarray([True, True, True, True, False]),
                "structured_mask": np.asarray([True] * 5),
                "dense_mask": np.asarray([True] * 5),
                "duration_observed": np.asarray([True, False, True, True, False]),
                "current_event_id": current_event,
                "clock_event_id": current_event,
                "next_event_id": next_event,
                "next_reached_event_id": next_event,
                "body_id": np.zeros(5, dtype=np.int64),
                "policy_id": np.zeros(5, dtype=np.int64),
                "duration": duration,
                "success": structured_success,
                "outcome_id": structured_success.astype(np.int64),
                "trajectory_regress": np.asarray([False] * 5),
                "trajectory_recovery": np.asarray([False] * 5),
                "object_delta": object_target,
                "post_predicates": predicates,
                "predicate_names": [
                    "moved",
                    "lifted",
                    "near_goal",
                    "stationary",
                    "success",
                ],
                "recovery_supervised": False,
                "member_next_event_logits": event_logits,
                "member_next_reached_event_logits": reached_logits,
                "member_post_predicate_logits": predicate_logits,
                "member_duration_log_mean": duration_mean,
                "member_duration_log_scale": duration_scale,
                "member_reach_logit": np.asarray([[5.0, -5.0, 5.0, 5.0, -5.0]] * 3),
                "member_object_delta_mean": object_mean,
                "member_object_delta_log_scale": object_scale,
                "member_outcome_logits": outcome_logits,
            }
        rows.append(row)
    return manifest, rows


def _attach_recorded_success_contracts(
    manifest: dict, rows: list[dict], *, positive_weight: float = 1.0
) -> None:
    training_sha = {
        int(fold["fold_id"]): canonical_sha256(
            sorted(map(str, fold["training_groups"]))
        )
        for fold in manifest["folds"]
    }
    for row in rows:
        fold_id = int(row["fold_id"])
        row["success_head_training_contract"] = {
            "format": SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT,
            "status": RECORDED_WEIGHTED_SUCCESS_STATUS,
            "owner_fold_id": fold_id,
            "positive_weight": positive_weight,
            "positive_weight_source": (
                "checkpoint_loss_metadata_recorded_before_training"
            ),
            "success_head_updated_on_owner_training_groups": True,
            "owner_oof_holdout_excluded_from_training": True,
            "owner_training_groups_sha256": training_sha[fold_id],
        }


def test_complete_diagnostics_use_all_unique_heldout_rows() -> None:
    manifest, rows = _fixture_rows()
    result = build_oof_prediction_diagnostics(rows, manifest)
    assert result["format"] == FORMAT
    assert result["status"] == "complete"
    assert result["fresh_confirmation_data_or_labels_read"] is False
    assert result["authorization_guard_changed"] is False
    assert result["oof_groups"] == 100
    assert result["structured_world_model"]["support"] == {
        "transitions": 500,
        "structured_transitions": 500,
        "terminal_branches": 400,
        "observed_durations": 300,
        "right_censored_durations": 200,
    }
    assert result["structured_world_model"]["next_event"]["ensemble"][
        "top1_accuracy"
    ] == 1.0
    assert result["structured_world_model"]["object_state_change"][
        "rmse_per_coordinate"
    ] == pytest.approx(0.0)
    assert result["structured_world_model"]["outcome"]["recovery_status"] == (
        "not_evaluable_model_contract_recovery_supervised_false"
    )
    assert result["structured_world_model"]["outcome"][
        "recovery_not_trained_fail_closed"
    ] is True
    assert result["prediction_adequacy"]["domain_pass"]["recovery"] is False
    assert result["success_probability"]["candidate_scope"] == (
        "deployment_exact_first_four_only"
    )
    assert result["success_probability"]["crossfit_calibrated"][
        "pr_auc_average_precision"
    ] == pytest.approx(1.0)
    strict_success = result["success_probability"][
        "strict_probability_assessment"
    ]
    assert strict_success["usable_for_strict_adequacy"] is False
    assert strict_success["status"] == (
        "unavailable_success_head_training_contract_not_recorded"
    )
    assert result["success_probability"]["crossfit_calibrated_valid_for_strict_adequacy"] is False
    assert result["success_probability"]["crossfit_calibration_overlap_audit"][
        "outer_target_label_indirect_model_training_overlap"
    ] == "not_excluded"
    assert result["success_probability"]["within_group_success_pair_ranking"][
        "pair_weighted_accuracy_ties_half"
    ] == pytest.approx(1.0)
    assert result["structured_world_model"]["next_event"]["ensemble"][
        "macro_f1_present_classes"
    ] == pytest.approx(1.0)
    assert result["prediction_adequacy"][
        "independent_of_reranking_authorization_guard"
    ] is True
    assert result["prediction_adequacy"]["fresh50_authorization_effect"] == "none"
    assert result["prediction_adequacy"]["domain_checks"][
        "success_probability_and_within_group_ranking"
    ]["ece_at_most_0_10"] is False
    assert result["prediction_adequacy"]["domain_checks"][
        "success_probability_and_within_group_ranking"
    ]["strict_probability_calibration_evaluable"] is False
    predicate_metrics = result["structured_world_model"]["event_subheads"][
        "post_predicates"
    ]["per_predicate"]
    assert len(predicate_metrics) == 5
    assert all(item["model_macro_f1"] == pytest.approx(1.0) for item in predicate_metrics)
    assert all(
        item["prospective_next_oof_probability_repair"]["status"]
        == "unavailable_predicate_head_training_weight_not_recorded"
        for item in predicate_metrics
    )
    assert result["structured_world_model"]["duration"][
        "prospective_next_oof_training_repair"
    ]["target_fold_labels_used_for_fit"] is False
    assert result["structured_world_model"]["object_state_change"][
        "prospective_next_oof_training_repair"
    ]["target_fold_labels_used_for_fit"] is False
    assert "post_event_predicates" in result["prediction_adequacy"]["domain_pass"]
    unsigned = dict(result)
    digest = unsigned.pop("diagnostics_sha256")
    assert digest == canonical_sha256(unsigned)
    assert validate_oof_prediction_diagnostics(result, manifest) == {
        "oof_groups": 100,
        "heldout_groups_per_fold": 20,
    }


def test_signed_diagnostics_tampering_is_rejected() -> None:
    manifest, rows = _fixture_rows()
    result = build_oof_prediction_diagnostics(rows, manifest)
    result["oof_groups"] = 99
    with pytest.raises(RuntimeError, match="contract/signature"):
        validate_oof_prediction_diagnostics(result, manifest)


def test_recorded_success_head_weight_enables_strict_probability_adequacy() -> None:
    manifest, rows = _fixture_rows()
    _attach_recorded_success_contracts(manifest, rows, positive_weight=1.0)
    result = build_oof_prediction_diagnostics(rows, manifest)
    strict = result["success_probability"]["strict_probability_assessment"]
    assert strict["usable_for_strict_adequacy"] is True
    assert strict["status"] == (
        "evaluable_recorded_training_weight_prior_shift_only"
    )
    checks = result["prediction_adequacy"]["domain_checks"][
        "success_probability_and_within_group_ranking"
    ]
    assert checks["strict_probability_calibration_evaluable"] is True
    assert checks["brier_better_than_crossfit_prevalence"] is True
    assert checks["nll_better_than_crossfit_prevalence"] is True
    assert checks["ece_at_most_0_10"] is True


def test_frozen_factual_weight_unavailable_is_explicit_and_fail_closed() -> None:
    manifest, rows = _fixture_rows()
    for row in rows:
        row["success_prediction_source"] = (
            "frozen_factual_success_logit_bit_exact_no_rank_residual"
        )
        row["success_head_training_contract"] = {
            "format": SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT,
            "status": FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS,
            "owner_fold_id": int(row["fold_id"]),
            "success_head_updated_on_owner_training_groups": False,
            "factual_core_bit_exact": True,
            "positive_weight": None,
        }
        row["diagnostic_member_axis"] = (
            "single_frozen_factual_prediction_repeated_for_legacy_three_member_axis"
        )
    result = build_oof_prediction_diagnostics(rows, manifest)
    success = result["success_probability"]
    assert success["strict_probability_assessment"]["status"] == (
        FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS
    )
    assert success["strict_probability_assessment"][
        "usable_for_strict_adequacy"
    ] is False
    overlap = success["crossfit_calibration_overlap_audit"]
    assert overlap["outer_target_label_indirect_model_training_overlap"] is False
    assert overlap["factual_historical_training_overlap"] == (
        "not_excluded_for_legacy_old100"
    )


def test_existing_v6_marker_without_new_contract_is_still_factual_unavailable() -> None:
    manifest, rows = _fixture_rows()
    for row in rows:
        row["diagnostic_member_axis"] = (
            "single_frozen_factual_prediction_repeated_for_legacy_three_member_axis"
        )
    result = build_oof_prediction_diagnostics(rows, manifest)
    success = result["success_probability"]
    assert success["success_prediction_source"] == (
        "frozen_factual_success_logit_bit_exact_no_rank_residual"
    )
    assert success["strict_probability_assessment"]["status"] == (
        FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS
    )


def test_success_training_contract_group_hash_tampering_is_rejected() -> None:
    manifest, rows = _fixture_rows()
    _attach_recorded_success_contracts(manifest, rows)
    rows[0]["success_head_training_contract"][
        "owner_training_groups_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="training provenance"):
        build_oof_prediction_diagnostics(rows, manifest)


def test_legacy_raw_rows_report_structured_metrics_as_missing() -> None:
    manifest, rows = _fixture_rows(structured=False)
    result = build_oof_prediction_diagnostics(rows, manifest)
    assert result["status"] == "legacy_raw_schema_partial"
    assert result["success_probability"]["crossfit_calibrated"]["support"] == 400
    assert result["structured_world_model"]["status"].startswith("not_evaluable")


def test_mixed_structured_schema_fails_closed() -> None:
    manifest, rows = _fixture_rows()
    rows[0].pop("structured_predictions")
    with pytest.raises(RuntimeError, match="mixed legacy/new"):
        build_oof_prediction_diagnostics(rows, manifest)


def test_owner_fold_tampering_is_rejected() -> None:
    manifest, rows = _fixture_rows()
    rows[0]["fold_id"] = (rows[0]["fold_id"] + 1) % 5
    with pytest.raises(RuntimeError, match="owner-fold"):
        build_oof_prediction_diagnostics(rows, manifest)


def test_duration_event_body_baseline_never_uses_target_fold_labels() -> None:
    folds = np.repeat(np.arange(5), 4)
    observed = np.ones(20, dtype=bool)
    event = np.tile(np.asarray([0, 0, 1, 1]), 5)
    body = np.tile(np.asarray([0, 1, 0, 1]), 5)
    target = np.linspace(0.2, 2.1, 20)
    prediction, _ = _crossfit_event_body_median(
        target, observed, folds, event, body
    )
    changed = target.copy()
    changed[folds == 2] += 1000.0
    changed_prediction, _ = _crossfit_event_body_median(
        changed, observed, folds, event, body
    )
    assert np.array_equal(
        prediction[folds == 2], changed_prediction[folds == 2]
    )


def test_uncertainty_reports_random_aurc_and_fold_wins() -> None:
    uncertainty = np.tile(np.asarray([0.0, 1.0]), 5)
    error = np.tile(np.asarray([0.0, 1.0]), 5)
    folds = np.repeat(np.arange(5), 2)
    result = _uncertainty_diagnostic(uncertainty, error, fold_id=folds)
    assert result["random_order_expected_aurc"] == pytest.approx(0.5)
    assert result["selective_risk_aurc"] < result["random_order_expected_aurc"]
    assert result["folds_with_aurc_better_than_random"] == 5


def test_training_only_fifth_cannot_change_main_success_adequacy() -> None:
    manifest, rows = _fixture_rows(structured=False)
    base = build_oof_prediction_diagnostics(rows, manifest)
    for row in rows:
        row["candidate_names"].append("sample_blend_1.000")
        row["success"] = np.r_[row["success"], 1.0]
        # Make the fifth candidate maximally wrong in every member.
        row["member_success_logits"] = np.c_[
            row["member_success_logits"], np.full(3, -20.0)
        ]
        row["member_aleatoric"] = np.c_[
            row["member_aleatoric"], np.zeros(3)
        ]
    expanded = build_oof_prediction_diagnostics(rows, manifest)
    assert expanded["success_probability"]["crossfit_calibrated"] == base[
        "success_probability"
    ]["crossfit_calibrated"]
    assert expanded["success_probability_all_collected_candidates_appendix"][
        "crossfit_calibrated"
    ]["brier"] > expanded["success_probability"]["crossfit_calibrated"]["brier"]


def test_structured_terminal_success_must_match_raw_candidate_rows() -> None:
    manifest, rows = _fixture_rows()
    rows[0]["structured_predictions"]["success"][0] = (
        1.0 - rows[0]["structured_predictions"]["success"][0]
    )
    with pytest.raises(RuntimeError, match="do not match candidate"):
        build_oof_prediction_diagnostics(rows, manifest)


def test_structured_member_count_is_frozen_to_three() -> None:
    manifest, rows = _fixture_rows()
    rows[0]["structured_predictions"]["member_next_event_logits"] = rows[0][
        "structured_predictions"
    ]["member_next_event_logits"][:2]
    with pytest.raises(RuntimeError, match="next-event predictions"):
        build_oof_prediction_diagnostics(rows, manifest)


def test_predicate_without_positive_support_fails_adequacy_closed() -> None:
    manifest, rows = _fixture_rows()
    for row in rows:
        structured = row["structured_predictions"]
        structured["post_predicates"][:, 0] = 0.0
        structured["member_post_predicate_logits"][:, :, 0] = -5.0
    result = build_oof_prediction_diagnostics(rows, manifest)
    checks = result["prediction_adequacy"]["domain_checks"][
        "post_event_predicates"
    ]
    assert checks["moved.positive_support"] is False
    assert result["prediction_adequacy"]["domain_pass"][
        "post_event_predicates"
    ] is False
