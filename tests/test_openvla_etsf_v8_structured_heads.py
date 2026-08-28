from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_openvla_etsf_v8_structured_heads_arrays as evaluator  # noqa: E402
import openvla_etsf_v8_structured_heads_protocol as protocol  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.fixture(autouse=True)
def _small_test_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production preregistration remains fixed at 10,000.  Tests exercise the
    # same cluster-resampling implementation with a smaller explicit contract.
    monkeypatch.setattr(protocol, "BOOTSTRAP_SAMPLES", 300)
    monkeypatch.setattr(evaluator, "BOOTSTRAP_SAMPLES", 300)


def _preregistration() -> dict:
    return protocol.make_preregistration(
        implementation_sha256=SHA_A,
        label_derivation_sha256=SHA_B,
        base_checkpoint_sha256=SHA_C,
        base_training_groups_sha256=SHA_D,
    )


def _weight_provenance(head: str) -> dict:
    result = {}
    for fold in range(5):
        item = {
            "head": head,
            "owner_fold_id": fold,
            "loss": "unweighted_bce",
            "weights_recorded_before_training": True,
            "owner_holdout_labels_used": False,
            "calibration_source": "none_unweighted_probability",
            "outer_training_positive": 400,
            "outer_training_negative": 400,
            "outer_training_prevalence": 0.5,
        }
        item["weight_contract_sha256"] = protocol.canonical_sha256(item)
        result[str(fold)] = item
    return result


def _fixture() -> tuple[dict, dict, dict, dict, dict]:
    rows: list[dict] = []
    # 25 groups/fold: five are historical old100 overlap and twenty are clean.
    for fold in range(5):
        for local_group in range(25):
            group = f"move_can_pot|piper|f{fold}g{local_group}"
            old100 = local_group < 5
            for candidate in range(5):
                deployment = candidate < 4
                success = float((local_group + candidate) % 2 == 0)
                regress = float(deployment and candidate in (1, 3))
                recovery = float(regress and local_group % 2 == 0)
                object_delta = np.asarray(
                    [0.2 + 0.01 * candidate, -0.1, 0.05], dtype=np.float64
                )
                rows.append(
                    {
                        "logical_group": group,
                        "fold_id": fold,
                        "old100": old100,
                        "candidate": candidate,
                        "duration_observed": candidate == 0,
                        "duration": 4.0 + (local_group % 3),
                        "success": success,
                        "regress": regress,
                        "recovery": recovery,
                        "object_delta": object_delta,
                    }
                )
    length = len(rows)
    groups = np.asarray([row["logical_group"] for row in rows], dtype=object)
    folds = np.asarray([row["fold_id"] for row in rows], dtype=np.int64)
    success = np.asarray([row["success"] for row in rows], dtype=np.float64)
    regress = np.asarray([row["regress"] for row in rows], dtype=np.float64)
    recovery = np.asarray([row["recovery"] for row in rows], dtype=np.float64)
    duration = np.asarray([row["duration"] for row in rows], dtype=np.float64)
    duration_target = np.log1p(duration)
    object_delta = np.stack([row["object_delta"] for row in rows])
    arrays = {
        "logical_group": groups,
        "fold_id": folds,
        "historical_old100_overlap": np.asarray([row["old100"] for row in rows]),
        "candidate_index": np.asarray([row["candidate"] for row in rows]),
        "duration_observed": np.asarray([row["duration_observed"] for row in rows]),
        "duration_steps": duration,
        "duration_model_log_location": duration_target,
        "duration_frozen_log_location": duration_target - (0.5 / 0.375) + 0.5,
        "duration_model_log_scale": np.full(length, np.log(0.1)),
        "duration_baseline_log_location": duration_target + 0.5,
        "duration_baseline_log_scale": np.full(length, np.log(0.5)),
        "success_mask": np.ones(length, dtype=bool),
        "success_label": success,
        "success_probability": np.where(success > 0.5, 0.95, 0.05),
        "success_baseline_probability": np.full(length, 0.5),
        "regress_mask": np.asarray([row["candidate"] < 4 for row in rows]),
        "regress_label": regress,
        "regress_probability": np.where(regress > 0.5, 0.95, 0.05),
        "regress_baseline_probability": np.full(length, 0.5),
        "recovery_label": recovery,
        "recovery_probability_given_regress": np.where(recovery > 0.5, 0.95, 0.05),
        "recovery_baseline_probability_given_regress": np.full(length, 0.5),
        "object_mask": np.ones(length, dtype=bool),
        "object_pose_quality_valid": np.ones(length, dtype=bool),
        "object_delta": object_delta,
        "object_model_delta": object_delta.copy(),
        "object_robust_median_delta": np.zeros_like(object_delta),
    }
    unique_groups = sorted(set(groups.tolist()))
    fold_contracts = {}
    next_event_sha = "e" * 64
    for fold in range(5):
        heldout = sorted(set(groups[folds == fold].tolist()))
        training = sorted(set(unique_groups) - set(heldout))
        fold_contracts[str(fold)] = {
            "owner_fold_id": fold,
            "heldout_logical_groups": heldout,
            "training_logical_groups": training,
            "outer_target_labels_used_for_fit": False,
            "baseline_fit_scope": "outer_training_only",
            "calibration_fit_scope": "none",
            "base_checkpoint_sha256": SHA_C,
            "next_event_state_sha256_before": next_event_sha,
            "next_event_state_sha256_after": next_event_sha,
            "next_event_trainable": False,
            "trainable_parameter_names": [
                "duration_adapter.weight",
                "success_adapter.weight",
                "regress_adapter.weight",
                "recovery_adapter.weight",
                "object_exploratory_adapter.weight",
            ],
            "duration_event_body_min_training_support": 20,
            "duration_residual_multiplier": 0.375,
            "duration_censored_used_for_location": False,
            "duration_scale_fit_scope": "outer_training_only",
        }
        fold_contracts[str(fold)]["fold_contract_sha256"] = (
            protocol.canonical_sha256(fold_contracts[str(fold)])
        )
    weights = {
        name: _weight_provenance(name)
        for name in ("success", "regress", "recovery_given_regress")
    }
    input_contract = {
        "source_partition": "development_only",
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "old100_overlap_declared": True,
    }
    return arrays, _preregistration(), fold_contracts, weights, input_contract


def _evaluate(fixture: tuple[dict, dict, dict, dict, dict]) -> dict:
    arrays, preregistration, folds, weights, inputs = fixture
    return evaluator.evaluate_structured_heads_arrays(
        arrays,
        preregistration=preregistration,
        fold_contracts=folds,
        probability_weight_provenance=weights,
        input_contract=inputs,
    )


def test_complete_array_evaluation_passes_and_separates_old100() -> None:
    fixture = _fixture()
    result = _evaluate(fixture)
    assert result["all_domain_pass"] is True
    assert result["domains"]["next_event"]["status"] == (
        "frozen_bit_exact_only_predictive_accuracy_not_evaluated"
    )
    assert result["domains"]["next_event"]["evaluated"] is False
    assert result["domains"]["next_event"]["accuracy_not_re_evaluated"] is True
    assert "next_event_frozen" not in result["domain_pass"]
    assert result["domains"]["duration"]["right_censored_rows_excluded_from_duration_error"] > 0
    assert result["domains"]["success"]["candidate_scope"] == (
        "provenance_clean_first_four_only"
    )
    assert result["domains"]["regress"]["ap_minus_prevalence"]["strict_skill"] is True
    assert result["domains"]["recovery_given_regress"]["conditional"]["support_gate"] is True
    assert result["domains"]["object"]["activated_output"] == "learned_object_delta"
    assert "row_rmse" in result["domains"]["object"]["comparisons"]["all_recorded"]["zero"]
    assert set(result["domains"]["object"]["fold_wins"]) == {
        "all_recorded", "quality_valid"
    }
    assert result["cohorts"]["historical_old100_overlap_descriptive_only"]["rows"] == 125
    assert result["cohorts"]["provenance_clean_primary"]["rows"] == 500
    assert result["fresh50_confirmation_authorized"] is False
    evaluator.validate_result(result, fixture[1])


def test_preregistration_and_result_tampering_fail_signature_validation() -> None:
    fixture = _fixture()
    changed = copy.deepcopy(fixture[1])
    changed["domains"]["duration"]["minimum_observed_groups"] = 1
    with pytest.raises(RuntimeError, match="signature"):
        protocol.validate_preregistration(changed)
    result = _evaluate(fixture)
    result["fresh50_labels_read"] = True
    with pytest.raises(RuntimeError, match="signature"):
        evaluator.validate_result(result, fixture[1])


def test_fresh_input_contract_is_rejected_before_metrics() -> None:
    fixture = list(_fixture())
    fixture[4] = {**fixture[4], "fresh50_labels_read": True}
    with pytest.raises(RuntimeError, match="rejects Fresh50"):
        _evaluate(tuple(fixture))


@pytest.mark.parametrize("mutation,match", [
    ("training_leak", "leaked"),
    ("next_event_changed", "bit-exact"),
    ("duration_multiplier", "multiplier"),
    ("duration_scale_scope", "scale fit scope"),
])
def test_strict_oof_and_frozen_core_audit_fail_closed(mutation: str, match: str) -> None:
    fixture = list(_fixture())
    folds = copy.deepcopy(fixture[2])
    if mutation == "training_leak":
        folds["0"]["training_logical_groups"].append(
            folds["0"]["heldout_logical_groups"][0]
        )
    elif mutation == "next_event_changed":
        folds["0"]["next_event_state_sha256_after"] = "f" * 64
    else:
        if mutation == "duration_multiplier":
            folds["0"]["duration_residual_multiplier"] = 1.0
        else:
            folds["0"]["duration_scale_fit_scope"] = "all_rows"
    # Re-signing proves validation checks the frozen semantics rather than only
    # detecting accidental byte corruption.
    unsigned = dict(folds["0"])
    unsigned.pop("fold_contract_sha256")
    folds["0"]["fold_contract_sha256"] = protocol.canonical_sha256(unsigned)
    fixture[2] = folds
    with pytest.raises(RuntimeError, match=match):
        _evaluate(tuple(fixture))


def test_missing_success_weight_provenance_only_fails_probability_domain() -> None:
    fixture = list(_fixture())
    weights = copy.deepcopy(fixture[3])
    del weights["success"]
    fixture[3] = weights
    result = _evaluate(tuple(fixture))
    assert result["domains"]["success"]["passed"] is False
    assert "missing" in result["domains"]["success"]["weight_provenance_error"]
    assert result["domains"]["duration"]["passed"] is True
    assert result["all_domain_pass"] is False


def test_probability_provenance_signature_counts_and_owner_baseline_are_bound() -> None:
    fixture = list(_fixture())
    weights = copy.deepcopy(fixture[3])
    weights["success"]["0"]["outer_training_positive"] = 1
    fixture[3] = weights
    result = _evaluate(tuple(fixture))
    assert "signature mismatch" in result["domains"]["success"]["weight_provenance_error"]

    fixture = list(_fixture())
    weights = copy.deepcopy(fixture[3])
    item = weights["success"]["0"]
    item["outer_training_positive"] = 3
    item["outer_training_negative"] = 1
    item["outer_training_prevalence"] = 0.5
    unsigned = dict(item); unsigned.pop("weight_contract_sha256")
    item["weight_contract_sha256"] = protocol.canonical_sha256(unsigned)
    fixture[3] = weights
    result = _evaluate(tuple(fixture))
    assert "prevalence is inconsistent" in result["domains"]["success"]["weight_provenance_error"]

    fixture = list(_fixture())
    weights = copy.deepcopy(fixture[3])
    item = weights["success"]["0"]
    item["head"] = "regress"
    unsigned = dict(item); unsigned.pop("weight_contract_sha256")
    item["weight_contract_sha256"] = protocol.canonical_sha256(unsigned)
    fixture[3] = weights
    result = _evaluate(tuple(fixture))
    assert "head binding mismatch" in result["domains"]["success"]["weight_provenance_error"]

    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    arrays["success_baseline_probability"][arrays["fold_id"] == 0] = 0.99
    fixture[0] = arrays
    with pytest.raises(RuntimeError, match="owner-fold outer-training prevalence"):
        _evaluate(tuple(fixture))


def test_group_level_old100_and_candidate_uniqueness_are_hard_invariants() -> None:
    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    group = arrays["logical_group"][0]
    group_rows = np.flatnonzero(arrays["logical_group"] == group)
    arrays["historical_old100_overlap"][group_rows[-1]] = False
    fixture[0] = arrays
    with pytest.raises(RuntimeError, match="constant within logical group"):
        _evaluate(tuple(fixture))

    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    primary = ~arrays["historical_old100_overlap"]
    group = arrays["logical_group"][np.flatnonzero(primary)[0]]
    row = np.flatnonzero(
        (arrays["logical_group"] == group) & (arrays["candidate_index"] == 3)
    )[0]
    arrays["candidate_index"][row] = 2
    fixture[0] = arrays
    with pytest.raises(RuntimeError, match="exactly one supervised success row"):
        _evaluate(tuple(fixture))


def test_duration_fixed_shrink_formula_and_finite_scales_are_enforced() -> None:
    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    arrays["duration_model_log_location"][0] += 1e-3
    fixture[0] = arrays
    with pytest.raises(RuntimeError, match="does not equal"):
        _evaluate(tuple(fixture))

    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    arrays["duration_model_log_scale"][0] = 1000.0
    fixture[0] = arrays
    with pytest.raises(ValueError, match="scale must be positive"):
        _evaluate(tuple(fixture))


def test_fold_contract_canonical_signature_is_verified() -> None:
    fixture = list(_fixture())
    folds = copy.deepcopy(fixture[2])
    folds["0"]["duration_event_body_min_training_support"] = 999
    fixture[2] = folds
    with pytest.raises(RuntimeError, match="fold contract signature"):
        _evaluate(tuple(fixture))


def test_recovery_is_conditional_on_regress_and_insufficient_support_fails_closed() -> None:
    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    # Remove recovery positives from one clean fold.  The conditional support
    # gate must fail even though other folds remain well supported.
    clean_fold_zero = (
        (arrays["fold_id"] == 0)
        & ~arrays["historical_old100_overlap"]
        & (arrays["regress_label"] > 0.5)
    )
    arrays["recovery_label"][clean_fold_zero] = 0.0
    arrays["recovery_probability_given_regress"][clean_fold_zero] = 0.05
    fixture[0] = arrays
    result = _evaluate(tuple(fixture))
    conditional = result["domains"]["recovery_given_regress"]["conditional"]
    assert conditional["support_by_fold"]["0"]["positive"] == 0
    assert conditional["support_gate"] is False
    assert result["domains"]["recovery_given_regress"]["passed"] is False


def test_recovery_true_without_regress_is_rejected() -> None:
    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    index = np.flatnonzero(
        (arrays["regress_mask"] > 0) & (arrays["regress_label"] == 0)
    )[0]
    arrays["recovery_label"][index] = 1.0
    fixture[0] = arrays
    with pytest.raises(RuntimeError, match="must imply regress"):
        _evaluate(tuple(fixture))


def test_object_head_defaults_to_fallback_unless_both_baselines_pass() -> None:
    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    arrays["object_model_delta"] = arrays["object_delta"] + 1.0
    fixture[0] = arrays
    result = _evaluate(tuple(fixture))
    object_result = result["domains"]["object"]
    assert object_result["passed"] is False
    assert object_result["activated_output"] == (
        "outer_training_robust_median_fallback"
    )
    assert object_result["learned_multiplier"] == 0.0


def test_primary_fails_if_only_historical_overlap_remains() -> None:
    fixture = list(_fixture())
    arrays = copy.deepcopy(fixture[0])
    arrays["historical_old100_overlap"][:] = True
    fixture[0] = arrays
    with pytest.raises(RuntimeError, match="no provenance-clean rows"):
        _evaluate(tuple(fixture))
