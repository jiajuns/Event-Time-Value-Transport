from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smolvla_piper_evaluation400_audit_contract_v1 as audit  # noqa: E402
from train_smolvla_piper_schema6_embodiment_adapter import (  # noqa: E402
    derive_conditional_recovery_targets,
)


def sha(character: str) -> str:
    return character * 64


def root_predictions(candidates: int = 3) -> dict[str, np.ndarray]:
    return {
        "post_event_logits": np.zeros((5, candidates, 4), dtype=np.float32),
        "next_event_logits": np.ones((5, candidates, 4), dtype=np.float32),
        "duration_log_mean": np.full((5, candidates), 0.5, dtype=np.float32),
        "duration_log_scale": np.full((5, candidates), -1.0, dtype=np.float32),
        "success_logit": np.linspace(
            -1.0, 1.0, 5 * candidates, dtype=np.float32
        ).reshape(5, candidates),
        "object_mean": np.zeros((5, candidates, 6), dtype=np.float32),
        "object_log_scale": np.full(
            (5, candidates, 6), -2.0, dtype=np.float32
        ),
    }


def root_authority() -> dict[str, Any]:
    return {
        "five_member_checkpoint_file_sha256": [
            f"{index + 1:x}" * 64 for index in range(5)
        ],
        "source_rank_score_contract_sha256": [
            f"{index + 6:x}" * 64 for index in range(5)
        ],
        "source_rank_member_authority_sha256": "b" * 64,
        "source_rank_numeric_contract": audit.SOURCE_RANK_NUMERIC_CONTRACT,
        "calibration_sha256": "c" * 64,
        "ensemble_manifest_sha256": "d" * 64,
        "deployment_uncertainty_contract_sha256": "e" * 64,
        "deployment_uncertainty_implementation_file_sha256": "f" * 64,
        "canonical_event_spec_file_sha256": "1" * 64,
        "schema6_runtime_contract_sha256": "2" * 64,
    }


def root_precommit() -> dict[str, Any]:
    tensor = audit.build_root_tensor_commitment(root_predictions())
    ordered = ["7" * 64, "8" * 64, "9" * 64, "a" * 64]
    legal = [False, True, True, True]
    registry = audit.canonical_sha256(
        {
            "pair_id": "5" * 64,
            "candidate_count": 4,
            "ordered_candidate_sha256": ordered,
            "candidate_legal": legal,
        }
    )
    return audit.build_root_precommit(
        protocol_core_v4_sha256="3" * 64,
        parent_v3_core_sha256="4" * 64,
        pair_id="5" * 64,
        pair_ordinal=0,
        shared_snapshot_sha256="6" * 64,
        ordered_candidate_sha256=ordered,
        candidate_legal=legal,
        candidate_registry_sha256=registry,
        prediction_candidate_indices=[1, 2, 3],
        tensor_commitment=tensor,
        authority=root_authority(),
    )


def resign(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    base = {key: child for key, child in value.items() if key != field}
    return {**base, field: audit.canonical_sha256(base)}


def recovery_commit(
    *, step: int = 5, sequence: int = 0, previous: str | None = None,
) -> dict[str, Any]:
    return audit.build_recovery_pre_step_commitment(
        protocol_core_v4_sha256="3" * 64,
        root_prediction_commit_sha256=root_precommit()["commit_sha256"],
        pair_id="5" * 64,
        pair_ordinal=0,
        condition_id="etsf",
        condition_position=3,
        step_index=step,
        commit_sequence=sequence,
        previous_commit_sha256=previous,
        pre_action_snapshot_sha256="6" * 64,
        chosen_candidate_index=1,
        chosen_candidate_sha256="7" * 64,
        current_event_id=2,
        historical_peak_event_id=3,
        member_recovery_logits=[-1.0, -0.5, 0.0, 0.5, 1.0],
        conditional_recovery_temperature=0.7,
        calibration_sha256="c" * 64,
        source_rank_member_authority_sha256="b" * 64,
        schema6_runtime_contract_sha256="2" * 64,
    )


def test_root_five_head_precommit_exact_schema_and_sha() -> None:
    value = root_precommit()
    assert audit.validate_root_precommit(value) == value["commit_sha256"]
    assert value["root_head_names"] == list(audit.ROOT_HEADS)
    assert value["recovery_root"] == {
        "status": "not_applicable",
        "policy": audit.ROOT_RECOVERY_POLICY,
        "included_in_root_head_count": False,
    }
    assert value["precondition_barrier"]["simulator_step_calls"] == 0
    tampered = copy.deepcopy(value)
    tampered["candidate_legal"][0] = True
    with pytest.raises(audit.AuditContractError, match="canonical SHA"):
        audit.validate_root_precommit(tampered)


def test_root_tensor_float32_and_one_ulp_are_bound() -> None:
    predictions = root_predictions()
    first = audit.build_root_tensor_commitment(predictions)
    changed = root_predictions()
    changed["success_logit"][0, 0] = np.nextafter(
        changed["success_logit"][0, 0], np.float32(np.inf), dtype=np.float32
    )
    second = audit.build_root_tensor_commitment(changed)
    assert first["tensor_set_sha256"] != second["tensor_set_sha256"]
    invalid = root_predictions()
    invalid["success_logit"] = invalid["success_logit"].astype(np.float64)
    with pytest.raises(audit.AuditContractError, match="float32"):
        audit.build_root_tensor_commitment(invalid)


def test_root_and_recovery_numeric_fields_reject_bool() -> None:
    value = root_precommit()
    value["pair_ordinal"] = False
    value = resign(value, "commit_sha256")
    with pytest.raises(audit.AuditContractError, match="not bool"):
        audit.validate_root_precommit(value)
    with pytest.raises(audit.AuditContractError, match="not bool"):
        audit.build_recovery_pre_step_commitment(
            protocol_core_v4_sha256="3" * 64,
            root_prediction_commit_sha256=root_precommit()["commit_sha256"],
            pair_id="5" * 64,
            pair_ordinal=0,
            condition_id="etsf",
            condition_position=3,
            step_index=False,
            commit_sequence=0,
            previous_commit_sha256=None,
            pre_action_snapshot_sha256="6" * 64,
            chosen_candidate_index=1,
            chosen_candidate_sha256="7" * 64,
            current_event_id=2,
            historical_peak_event_id=3,
            member_recovery_logits=[0.0] * 5,
            conditional_recovery_temperature=1.0,
            calibration_sha256="c" * 64,
            source_rank_member_authority_sha256="b" * 64,
            schema6_runtime_contract_sha256="2" * 64,
        )


def test_recovery_broker_rejects_wrong_step_replay_and_broken_chain() -> None:
    state = audit.RecoveryBrokerState(pair_id="5" * 64, condition_id="etsf")
    first = recovery_commit()
    state.accept_commitment(first, expected_step_index=5)
    ack = audit.build_broker_ack(first, ledger_event_sha256="d" * 64)
    with pytest.raises(audit.AuditContractError, match="wrong step"):
        state.accept_ack(ack, expected_step_index=6)
    state.accept_ack(ack, expected_step_index=5)
    with pytest.raises(audit.AuditContractError, match="sequence"):
        state.accept_commitment(first, expected_step_index=5)
    second = recovery_commit(
        step=6, sequence=1, previous=first["commit_sha256"]
    )
    state.accept_commitment(second, expected_step_index=6)
    second_ack = audit.build_broker_ack(second, ledger_event_sha256="e" * 64)
    state.accept_ack(second_ack, expected_step_index=6)
    with pytest.raises(audit.AuditContractError, match="no pending"):
        state.accept_ack(second_ack, expected_step_index=6)


def test_sealed_target_roundtrip_requires_complete_terminal_and_exact_aad() -> None:
    private_key, public_key = audit.generate_x25519_keypair()
    target = {
        "root": {"post_event": 2, "success": 1},
        "first_operational_regress": {"observed": False},
    }
    envelope = audit.seal_target_envelope(
        target,
        evaluator_public_key_raw=public_key,
        protocol_core_v4_sha256="3" * 64,
        pair_id="5" * 64,
        condition_id="etsf",
        root_prediction_commit_sha256="6" * 64,
        schema6_runtime_contract_sha256="2" * 64,
    )
    assert not {
        "target_sha256", "plaintext_sha256", "target_logical_sha256"
    } & set(envelope)
    incomplete = audit.build_terminal_completeness(
        terminal_receipt_sha256="7" * 64, complete_condition_count=1599
    )
    with pytest.raises(audit.AuditContractError, match="complete_condition_count"):
        audit.open_target_envelope(
            envelope,
            evaluator_private_key_raw=private_key,
            terminal_completeness=incomplete,
        )
    complete = audit.build_terminal_completeness(
        terminal_receipt_sha256="7" * 64
    )
    assert audit.open_target_envelope(
        envelope,
        evaluator_private_key_raw=private_key,
        terminal_completeness=complete,
        expected_protocol_core_v4_sha256="3" * 64,
        expected_pair_id="5" * 64,
        expected_condition_id="etsf",
        expected_root_prediction_commit_sha256="6" * 64,
        expected_schema6_runtime_contract_sha256="2" * 64,
    ) == target
    with pytest.raises(audit.AuditContractError, match="AAD binding"):
        audit.open_target_envelope(
            envelope,
            evaluator_private_key_raw=private_key,
            terminal_completeness=complete,
            expected_pair_id="8" * 64,
        )


def test_sealed_target_ciphertext_and_key_tampering_fail_authentication() -> None:
    private_key, public_key = audit.generate_x25519_keypair()
    envelope = audit.seal_target_envelope(
        {"success": 1},
        evaluator_public_key_raw=public_key,
        protocol_core_v4_sha256="3" * 64,
        pair_id="5" * 64,
        condition_id="etsf",
        root_prediction_commit_sha256="6" * 64,
        schema6_runtime_contract_sha256="2" * 64,
    )
    complete = audit.build_terminal_completeness(
        terminal_receipt_sha256="7" * 64
    )
    changed = copy.deepcopy(envelope)
    ciphertext = bytearray.fromhex(changed["ciphertext_hex"])
    ciphertext[0] ^= 1
    changed["ciphertext_hex"] = bytes(ciphertext).hex()
    changed["ciphertext_sha256"] = hashlib.sha256(ciphertext).hexdigest()
    changed = resign(changed, "envelope_sha256")
    with pytest.raises(audit.AuditContractError, match="decryption failed"):
        audit.open_target_envelope(
            changed,
            evaluator_private_key_raw=private_key,
            terminal_completeness=complete,
        )
    wrong_private, _wrong_public = audit.generate_x25519_keypair()
    with pytest.raises(audit.AuditContractError, match="decryption failed"):
        audit.open_target_envelope(
            envelope,
            evaluator_private_key_raw=wrong_private,
            terminal_completeness=complete,
        )


def synthetic_dense_events(
    _names: list[str], _steps: list[int], terminal_step: int,
) -> dict[str, np.ndarray]:
    events = np.asarray([0, 3, 2, 2, 2, 3, 3, 3], dtype=np.int16)
    assert terminal_step == len(events) - 1
    return {
        "trajectory_event_id": events,
        "transition_next_event_id": np.asarray(
            [3, 3, 3, 3, 3, 3, 3], dtype=np.int16
        ),
        "transition_duration_decision_steps": np.asarray(
            [1, 4, 3, 2, 1, 2, 1], dtype=np.float32
        ),
        "transition_duration_observed": np.ones(terminal_step, dtype=bool),
        "transition_duration_censored": np.zeros(terminal_step, dtype=bool),
    }


def object_target(
    value: Mapping[str, Any], *, start_step: int, end_step: int,
) -> dict[str, Any]:
    assert (start_step, end_step) == (0, 1)
    poses = np.asarray(value["poses"], dtype=np.float64)
    return {
        "object_target": poses[end_step] - poses[start_step],
        "object_observed": True,
    }


def test_injected_target_recompute_selects_precommitted_first_regress_row() -> None:
    trace = {
        "event_names": ["e0"],
        "event_steps": [0],
        "terminal_step": 7,
        "terminal_success": False,
        "trajectory_success": [False] * 8,
        "right_censored": False,
        "object_trace": {"poses": [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]]},
    }
    target = audit.recompute_audit_targets(
        trace,
        dense_event_targets_fn=synthetic_dense_events,
        recovery_targets_fn=derive_conditional_recovery_targets,
        object_target_fn=object_target,
    )
    assert target["root"] == {
        "post_event": 3,
        "next_event": 3,
        "duration_decision_steps": 1.0,
        "duration_observed": True,
        "duration_censored": False,
        "success": 0,
        "object_target_physical_xyz_m": [0.1, -0.2, 0.3],
        "object_observed": True,
        "recovery": {
            "status": "not_applicable",
            "policy": audit.ROOT_RECOVERY_POLICY,
        },
    }
    assert target["first_operational_regress"] == {
        "status": "observed",
        "applicable": True,
        "observed": True,
        "censored": False,
        "step_index": 1,
        "target": 1,
    }


def test_pair_cluster_bootstrap_weights_pairs_not_rows() -> None:
    result = audit.pair_cluster_bootstrap(
        np.asarray([0.0, 2.0, 10.0]),
        np.asarray(["a", "a", "b"]),
        samples=200,
        seed=7,
    )
    assert result["pair_count"] == 2
    assert result["estimate"] == pytest.approx(5.5)


def six_head_metric_input(rows: int = 120) -> dict[str, Any]:
    pair_id = np.asarray([f"pair-{index:03d}" for index in range(rows)])
    binary = np.arange(rows, dtype=np.int64) % 2
    probability = np.column_stack(
        [np.where(binary == 0, 0.9, 0.1), np.where(binary == 1, 0.9, 0.1)]
    )
    baseline_probability = np.full((rows, 2), 0.5)
    binary_probability = np.where(binary == 1, 0.9, 0.1)
    duration_target = 1.0 + (np.arange(rows) % 4).astype(np.float64)
    duration_mean = np.repeat(
        np.log1p(duration_target)[:, None], audit.MEMBER_COUNT, axis=1
    )
    duration_scale = np.full_like(duration_mean, np.log(0.1))
    duration_observed = np.arange(rows) < rows // 2
    object_target_value = np.zeros((rows, 3), dtype=np.float64)
    object_target_value[rows // 2 :, 0] = 1.0
    object_mean = np.repeat(
        object_target_value[:, None, :], audit.MEMBER_COUNT, axis=1
    )
    object_scale = np.full_like(object_mean, np.log(0.1))
    recovery_observed = np.arange(rows) < 100
    return {
        "pair_id": pair_id,
        "post_event": {
            "probability": probability,
            "target": binary,
            "baseline_probability": baseline_probability,
            "observed": np.ones(rows, dtype=bool),
            "required_classes": [0, 1],
        },
        "next_event": {
            "probability": probability,
            "target": binary,
            "baseline_probability": baseline_probability,
            "observed": np.ones(rows, dtype=bool),
            "required_classes": [0, 1],
        },
        "success": {
            "probability": binary_probability,
            "target": binary,
            "baseline_probability": np.full(rows, 0.5),
            "observed": np.ones(rows, dtype=bool),
        },
        "recovery": {
            "probability": binary_probability,
            "target": binary,
            "baseline_probability": np.full(rows, 0.5),
            "applicable": np.ones(rows, dtype=bool),
            "observed": recovery_observed,
        },
        "duration": {
            "member_log_mean": duration_mean,
            "member_log_scale": duration_scale,
            "target": duration_target,
            "observed": duration_observed,
            "baseline_location": duration_target + 2.0,
            "baseline_scale": np.ones(rows),
        },
        "object_effect": {
            "member_mean": object_mean,
            "member_log_scale": object_scale,
            "target": object_target_value,
            "observed": np.ones(rows, dtype=bool),
            "baseline_robust": np.full((rows, 3), 2.0),
        },
    }


def test_six_head_known_metrics_support_and_censoring() -> None:
    result = audit.compute_six_head_metrics(
        six_head_metric_input(), bootstrap_samples=200, bootstrap_seed=17
    )
    assert result["status"] == "complete_all_six_heads"
    assert result["insufficient_support_heads"] == []
    assert result["heads"]["post_event"]["equal_pair_accuracy"] == 1.0
    assert result["heads"]["success"]["equal_pair_brier"] == pytest.approx(0.01)
    assert result["heads"]["recovery"]["censored_count"] == 20
    assert result["heads"]["duration"][
        "equal_pair_median_mae_observed"
    ] == pytest.approx(0.0, abs=1e-10)
    assert result["heads"]["object_effect"]["equal_pair_l2"] == 0.0
    assert result["heads"]["post_event"][
        "nll_gain_baseline_minus_model"
    ]["lower"] > 0.0


def test_six_head_metrics_report_insufficient_support_instead_of_passing() -> None:
    value = six_head_metric_input(rows=8)
    value["recovery"]["observed"] = np.zeros(8, dtype=bool)
    result = audit.compute_six_head_metrics(
        value, bootstrap_samples=100, bootstrap_seed=19
    )
    assert result["status"] == "insufficient_support"
    assert "recovery" in result["insufficient_support_heads"]
    assert result["heads"]["recovery"]["status"] == "insufficient_support"
    assert result["heads"]["recovery"]["observed_count"] == 0
