from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_smolvla_piper_evaluation400_results_v4 as result_v4  # noqa: E402
import smolvla_piper_evaluation400_audit_contract_v1 as audit_v1  # noqa: E402


SAMPLES = 100
PROTOCOL_SHA = "1" * 64


def _signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    base = {key: child for key, child in value.items() if key != field}
    return {**base, field: result_v4.canonical_sha256(base)}


def _event_probability(target: int, classes: int = 4) -> list[float]:
    probability = [0.1 / (classes - 1)] * classes
    probability[target] = 0.9
    return probability


def _heads(local: int, terminal_success: bool) -> dict[str, Any]:
    post_target = local % 4
    next_target = 1 + local % 3
    transition_observed = local % 5 != 0
    recovery_observed = local % 5 != 0
    recovery_target = int(local % 2 == 0) if recovery_observed else 0
    duration_target = float(2 + local % 5)
    object_target = [0.0, 0.0, 0.0]
    if local % 2:
        object_target = [0.2, -0.1, 0.05]
    object_members = [list(object_target) for _ in range(audit_v1.MEMBER_COUNT)]
    return {
        "post_event": {
            "probability": _event_probability(post_target),
            "target": post_target,
            "baseline_probability": [0.25] * 4,
            "applicable": True,
            "observed": True,
            "censored": False,
            "required_classes": [0, 1, 2, 3],
        },
        "next_event": {
            "probability": _event_probability(next_target),
            "target": next_target,
            "baseline_probability": [0.25] * 4,
            "applicable": True,
            "observed": transition_observed,
            "censored": not transition_observed,
            "required_classes": [1, 2, 3],
        },
        "duration": {
            "member_log_mean": [math.log1p(duration_target)] * audit_v1.MEMBER_COUNT,
            "member_log_scale": [-2.0] * audit_v1.MEMBER_COUNT,
            "target": duration_target,
            "observed": transition_observed,
            "applicable": True,
            "censored": not transition_observed,
            "baseline_location": duration_target + 1.0,
            "baseline_scale": 1.0,
        },
        "success": {
            "probability": 0.9 if terminal_success else 0.1,
            "target": int(terminal_success),
            "baseline_probability": 0.5,
            "applicable": True,
            "observed": True,
            "censored": False,
        },
        "recovery": {
            "probability": (
                (0.9 if recovery_target else 0.1) if recovery_observed else 0.5
            ),
            "target": recovery_target,
            "baseline_probability": 0.5,
            "applicable": True,
            "observed": recovery_observed,
            "censored": not recovery_observed,
        },
        "object_effect": {
            "member_mean": object_members,
            "member_log_scale": [
                [-3.0, -3.0, -3.0] for _ in range(audit_v1.MEMBER_COUNT)
            ],
            "target": object_target,
            "baseline_robust": [0.5, 0.5, 0.5],
            "applicable": True,
            "observed": True,
            "censored": False,
            "missing": False,
        },
    }


def _terminal_success(condition: str, local: int) -> bool:
    baseline = local % 2 == 0
    failure_rank = local // 2
    conversions = {
        "baseline": 0,
        "success_only_guarded": 30,
        "composite_rank_ungated": 40,
        "etsf": 50,
    }
    return bool(baseline or (not baseline and failure_rank < conversions[condition]))


def _selected(condition: str, local: int) -> int:
    if local % 2 == 0:
        return 0
    failure_rank = local // 2
    thresholds = {
        "baseline": (0, 0),
        "success_only_guarded": (30, 1),
        "composite_rank_ungated": (40, 2),
        "etsf": (50, 3),
    }
    threshold, candidate = thresholds[condition]
    return candidate if failure_rank < threshold else 0


def _record(ordinal: int, condition: str) -> dict[str, Any]:
    local = ordinal % 200
    policy = "policy-a" if ordinal < 200 else "policy-b"
    terminal_success = _terminal_success(condition, local)
    selected = _selected(condition, local)
    base = {
        "format": result_v4.RECORD_FORMAT,
        "status": result_v4.RECORD_STATUS,
        "protocol_core_v4_sha256": PROTOCOL_SHA,
        "pair_id": f"{ordinal + 1:064x}",
        "pair_ordinal": ordinal,
        "condition_id": condition,
        "condition_position": result_v4.CONDITION_POSITION[condition],
        "embodiment_id": "target-piper",
        "embodiment_role": "target_piper",
        "policy_id": policy,
        "shared_snapshot_sha256": f"{ordinal + 401:064x}",
        "candidate_registry_sha256": f"{ordinal + 801:064x}",
        "root_prediction_commit_sha256": f"{ordinal + 1201:064x}",
        "candidate_count": 4,
        "baseline_candidate_index": 0,
        "selected_candidate_index": selected,
        "changed_from_baseline": selected != 0,
        "terminal_success": terminal_success,
        "six_head": _heads(local, terminal_success),
        "target_recomputed_by_audit_contract_v1": True,
        "target_unsealed_after_terminal": True,
    }
    return _signed(base, "record_sha256")


def _lobo_authority() -> dict[str, Any]:
    base = {
        "format": result_v4.LOBO_EVIDENCE_AUTHORITY_FORMAT,
        "status": result_v4.LOBO_EVIDENCE_AUTHORITY_STATUS,
        "source_result_sha256": "a" * 64,
        "source_result_file_sha256": "b" * 64,
        "source_evaluator_implementation_file_sha256": "c" * 64,
        "source_audit_contract_v1_implementation_file_sha256": "d" * 64,
        "predeclared_slice_inventory_sha256": "e" * 64,
        "pair_count": 200,
        "primary_task_success_promotion_passed": True,
        "six_head_accuracy_promotion_passed": True,
        "every_predeclared_embodiment_policy_slice_passed": True,
        "pooled_result_used_for_promotion": False,
    }
    return _signed(base, "authority_sha256")


def bundle(*, with_lobo_authority: bool = False) -> dict[str, Any]:
    records = [
        _record(ordinal, condition)
        for ordinal in range(result_v4.PAIR_COUNT)
        for condition in result_v4.CONDITIONS
    ]
    base = {
        "format": result_v4.BUNDLE_FORMAT,
        "status": result_v4.BUNDLE_STATUS,
        "protocol_core_v4_sha256": PROTOCOL_SHA,
        "audit_contract_v1_implementation_file_sha256": result_v4.file_sha256(
            Path(audit_v1.__file__).resolve()
        ),
        "terminal_completeness": audit_v1.build_terminal_completeness(
            terminal_receipt_sha256="f" * 64
        ),
        "required_slice_inventory": [
            {
                "embodiment_id": "target-piper",
                "embodiment_role": "target_piper",
                "policy_id": "policy-a",
                "expected_pair_count": 200,
            },
            {
                "embodiment_id": "target-piper",
                "embodiment_role": "target_piper",
                "policy_id": "policy-b",
                "expected_pair_count": 200,
            },
        ],
        "records": records,
        "targets_unsealed_after_terminal": True,
        "hdf_or_trajectory_files_opened": 0,
        "evaluation400_subset_excluded": False,
        "lobo_evidence_authority": (
            _lobo_authority() if with_lobo_authority else None
        ),
    }
    return _signed(base, "bundle_sha256")


def _resign_bundle(value: dict[str, Any]) -> dict[str, Any]:
    value["records"] = [
        _signed(dict(row), "record_sha256") for row in value["records"]
    ]
    return _signed(value, "bundle_sha256")


@pytest.fixture(scope="module")
def valid_result() -> dict[str, Any]:
    return result_v4.evaluate_bundle(bundle(), bootstrap_samples=SAMPLES)


def test_target_piper_passes_without_fabricated_lobo_records(
    valid_result: dict[str, Any],
) -> None:
    assert valid_result["condition_order"] == list(result_v4.CONDITIONS)
    assert valid_result["comparison_registry"] == [
        {"name": name, "model": model, "comparator": comparator, "family": family}
        for name, model, comparator, family in result_v4.COMPARISONS
    ]
    assert set(valid_result["by_embodiment_role"]) == {"target_piper"}
    assert valid_result["target_task_success_promotion"]["passed"] is True
    assert valid_result["target_six_head_accuracy_promotion"]["passed"] is True
    assert valid_result["cross_embodiment_promotion"]["passed"] is False
    assert valid_result["lobo_evidence_authority"] is None
    assert valid_result["pooled_result_can_authorize_promotion"] is False
    assert valid_result["result_sha256"] == result_v4.canonical_sha256(
        {key: child for key, child in valid_result.items() if key != "result_sha256"}
    )
    pooled_secondary = [
        row for row in valid_result["pooled_diagnostic"]["comparisons"]
        if row["family"] == "secondary"
    ]
    assert len(pooled_secondary) == 4
    assert all("holm_adjusted_p" in row for row in pooled_secondary)
    assert all(
        row["paired_delta_pair_cluster_bootstrap"]["unit"] == "pair_id"
        for row in valid_result["pooled_diagnostic"]["comparisons"]
    )


def test_content_addressed_lobo_authority_is_separate_cross_embodiment_gate() -> None:
    value = bundle(with_lobo_authority=True)
    evaluated = result_v4.evaluate_bundle(
        value,
        bootstrap_samples=SAMPLES,
        expected_lobo_evidence_authority_sha256=value[
            "lobo_evidence_authority"
        ]["authority_sha256"],
    )
    assert evaluated["target_task_success_promotion"]["passed"] is True
    assert evaluated["cross_embodiment_promotion"] == {
        "status": "passed",
        "passed": True,
        "requires_target_piper_task_and_six_head_gates": True,
        "requires_separate_content_addressed_lobo_evidence_authority": True,
        "lobo_evidence_authority_present": True,
        "lobo_evidence_authority_externally_bound": True,
        "lobo_evidence_authority_passed": True,
        "pooled_cannot_mask_slice_failure": True,
        "target_task_success_promotion_passed": True,
        "target_six_head_accuracy_promotion_passed": True,
    }


def test_unbound_or_wrong_lobo_content_address_cannot_promote() -> None:
    value = bundle(with_lobo_authority=True)
    unbound = result_v4.evaluate_bundle(value, bootstrap_samples=SAMPLES)
    assert unbound["lobo_evidence_authority"][
        "externally_bound_by_expected_authority_sha256"
    ] is False
    assert unbound["cross_embodiment_promotion"]["passed"] is False
    with pytest.raises(
        result_v4.Evaluation400V4Error,
        match="differs from external content address",
    ):
        result_v4.evaluate_bundle(
            value,
            bootstrap_samples=SAMPLES,
            expected_lobo_evidence_authority_sha256="9" * 64,
        )


def test_one_bad_piper_policy_cannot_be_hidden_by_pooled_gain() -> None:
    value = bundle()
    for row in value["records"]:
        if row["policy_id"] == "policy-b" and row["condition_id"] == "etsf":
            local = row["pair_ordinal"] % 200
            baseline_success = local % 2 == 0
            row["terminal_success"] = baseline_success
            row["six_head"]["success"]["target"] = int(baseline_success)
            row["six_head"]["success"]["probability"] = (
                0.9 if baseline_success else 0.1
            )
    evaluated = result_v4.evaluate_bundle(
        _resign_bundle(value), bootstrap_samples=SAMPLES
    )
    assert evaluated["pooled_diagnostic"]["primary_task_gate_passed"] is True
    assert evaluated["by_embodiment_policy"][
        "target-piper::policy-b"
    ]["primary_task_gate_passed"] is False
    assert evaluated["target_task_success_promotion"]["passed"] is False
    assert evaluated["pooled_result_can_authorize_promotion"] is False


def test_six_head_support_failure_blocks_only_accuracy_and_cross_gate() -> None:
    value = bundle(with_lobo_authority=True)
    for row in value["records"]:
        if row["policy_id"] == "policy-b" and row["condition_id"] == "etsf":
            recovery = row["six_head"]["recovery"]
            if recovery["observed"]:
                recovery["target"] = 1
                recovery["probability"] = 0.9
    evaluated = result_v4.evaluate_bundle(
        _resign_bundle(value), bootstrap_samples=SAMPLES
    )
    assert evaluated["target_task_success_promotion"]["passed"] is True
    assert evaluated["target_six_head_accuracy_promotion"]["passed"] is False
    assert evaluated["cross_embodiment_promotion"]["passed"] is False
    assert evaluated["by_embodiment_policy"]["target-piper::policy-b"][
        "six_head_metrics_by_condition"
    ]["etsf"]["heads"]["recovery"]["status"] == "insufficient_support"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("bool_ordinal", "exact integer"),
        ("mask_drift", "next-event/duration censoring diverged"),
        ("success_drift", "success-head target differs"),
        ("authority_drift", "canonical SHA mismatch"),
    ],
)
def test_fully_resigned_contract_drift_fails_closed(
    mutation: str, match: str,
) -> None:
    value = bundle(with_lobo_authority=True)
    if mutation == "bool_ordinal":
        value["records"][0]["pair_ordinal"] = False
    elif mutation == "mask_drift":
        value["records"][0]["six_head"]["next_event"]["observed"] = True
        value["records"][0]["six_head"]["next_event"]["censored"] = False
    elif mutation == "success_drift":
        value["records"][0]["six_head"]["success"]["target"] = 0
    else:
        value["lobo_evidence_authority"]["pair_count"] = 201
        # Deliberately keep the external content address unchanged.
    value = _resign_bundle(value)
    with pytest.raises(result_v4.Evaluation400V4Error, match=match):
        result_v4.evaluate_bundle(value, bootstrap_samples=SAMPLES)


def test_missing_condition_fails_instead_of_evaluating_subset() -> None:
    value = bundle()
    del value["records"][-1]
    value = _signed(value, "bundle_sha256")
    with pytest.raises(result_v4.Evaluation400V4Error, match="exact 1600"):
        result_v4.evaluate_bundle(value, bootstrap_samples=SAMPLES)
