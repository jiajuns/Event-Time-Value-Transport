from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_smolvla_piper_schema6_target_validation50_ensemble as evaluator  # noqa: E402
import train_smolvla_piper_schema6_embodiment_adapter as trainer  # noqa: E402


def _frozen(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def _source_rank_contract(source_sha256: str) -> dict[str, object]:
    value: dict[str, object] = {
        "format": trainer.SOURCE_RANK_SCORE_FORMAT,
        "status": "frozen_exact_source63_training_score_scientific_rank_only",
        "source_checkpoint_file_sha256": source_sha256,
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
        "success_temperature": trainer.SOURCE_RANK_SUCCESS_TEMPERATURE,
        "source_rank_numeric_contract": (
            evaluator.SOURCE_RANK_NUMERIC_CONTRACT
        ),
        "event_weight": trainer.SOURCE_RANK_EVENT_WEIGHT,
        "duration_weight": trainer.SOURCE_RANK_DURATION_WEIGHT,
        "residual_combination": "candidate_rank_score_plus_action_rank_residual",
        "score_variant": "source_member_training_objective_defaults",
        "source_ensemble_validation_selected_scoring_consumed": False,
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "cross_embodiment_duration_scale_calibrated": False,
        "deployment_success_probability_selector_authorized": False,
    }
    value["contract_sha256"] = trainer.canonical_sha256(value)
    return value


def _authority(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest = _frozen(tmp_path / "manifest.json", b"{}\n")
    expected = _frozen(tmp_path / "expected.json", b"{}\n")
    event = _frozen(tmp_path / "event.json", b"{}\n")
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
    members = []
    for index in range(5):
        adapter = _frozen(tmp_path / f"adapter_{index}.pt", f"adapter-{index}".encode())
        source = _frozen(tmp_path / f"source_{index}.pt", f"source-{index}".encode())
        source_rank_contract = _source_rank_contract(trainer.file_sha256(source))
        member_receipt_value = {
            "status": "complete_frozen_internal_validation_predictions",
            "member_index": index,
        }
        member_receipt_value["receipt_sha256"] = trainer.canonical_sha256(
            member_receipt_value
        )
        member_receipt = _frozen(
            tmp_path / f"member_receipt_{index}.json",
            json.dumps(member_receipt_value, sort_keys=True).encode(),
        )
        members.append(
            {
                "member_index": index,
                "member_seed": 20260828 + index,
                "adapter_checkpoint": {
                    "path": str(adapter),
                    "file_sha256": trainer.file_sha256(adapter),
                },
                "source_checkpoint": {
                    "path": str(source),
                    "file_sha256": trainer.file_sha256(source),
                },
                "member_receipt": {
                    "path": str(member_receipt),
                    "file_sha256": trainer.file_sha256(member_receipt),
                    "logical_sha256": member_receipt_value["receipt_sha256"],
                },
                "training_manifest_sha256": "a" * 64,
                "split_sha256": "b" * 64,
                "source_ensemble_contract_sha256": "c" * 64,
                "prediction_contract": prediction_contract,
                "source_rank_score_contract": source_rank_contract,
                "source_rank_score_contract_sha256": source_rank_contract[
                    "contract_sha256"
                ],
            }
        )
    value: dict[str, object] = {
        "format": evaluator.INPUT_FORMAT,
        "status": evaluator.INPUT_STATUS,
        "trainer_compatible_manifest": {
            "path": str(manifest),
            "file_sha256": trainer.file_sha256(manifest),
            "logical_sha256": "a" * 64,
        },
        "expected_manifest_split_receipt": {
            "path": str(expected),
            "file_sha256": trainer.file_sha256(expected),
            "logical_sha256": "b" * 64,
        },
        "canonical_event_spec": {
            "path": str(event),
            "file_sha256": trainer.file_sha256(event),
        },
        "members": members,
        "member_count": 5,
        "target_validation_group_count": 50,
        "adapter_training_complete_before_authority": True,
        "target_validation_open_authorized": True,
        "evaluation400_membership_present": False,
        "evaluation400_open_authorized": False,
        "fresh_or_confirmation_open_authorized": False,
        "source_rank_numeric_contract": evaluator.SOURCE_RANK_NUMERIC_CONTRACT,
    }
    value["authority_sha256"] = trainer.canonical_sha256(value)
    path = _frozen(
        tmp_path / "authority.json",
        (json.dumps(value, sort_keys=True) + "\n").encode(),
    )
    return path, value


def test_authority_binds_five_frozen_members_and_prediction_space(tmp_path: Path) -> None:
    path, value = _authority(tmp_path)
    audit = evaluator.validate_input_authority(path, trainer.file_sha256(path))
    assert len(audit["members"]) == 5
    assert audit["source_rank_numeric_contract"] == (
        evaluator.SOURCE_RANK_NUMERIC_CONTRACT
    )
    assert len(set(audit["shared_contract"])) == 4
    assert audit["members"][0]["prediction_contract"]["next_event_observation_mask"] == "duration_observed"
    assert value["evaluation400_membership_present"] is False


def test_authority_rejects_evaluation400_before_checkpoint_open(tmp_path: Path) -> None:
    path, value = _authority(tmp_path)
    path.chmod(0o644)
    value["evaluation400_membership_present"] = True
    value["authority_sha256"] = trainer.canonical_sha256(
        {key: item for key, item in value.items() if key != "authority_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(evaluator.TargetValidationEvaluatorError, match="scope changed"):
        evaluator.validate_input_authority(path, trainer.file_sha256(path))


def test_sample_identity_requires_exactly_fifty_logical_groups() -> None:
    rows = [
        {"logical_group_id": f"validation/group/{index}"}
        for index in range(50)
        for _ in range(3)
    ]
    sample, groups = evaluator._sample_identity(rows)
    assert len(sample) == 150
    assert len(set(groups.tolist())) == 50
    with pytest.raises(evaluator.TargetValidationEvaluatorError, match="identity"):
        evaluator._sample_identity(rows[:-3])


def test_sample_identity_and_authority_accept_exact_development300_validation190(
    tmp_path: Path,
) -> None:
    rows = [
        {"logical_group_id": f"formal-validation/group/{index}"}
        for index in range(190)
        for _ in range(2)
    ]
    sample, groups = evaluator._sample_identity(rows, expected_groups=190)
    assert len(sample) == 380
    assert len(set(groups.tolist())) == 190

    path, value = _authority(tmp_path)
    path.chmod(0o644)
    value["target_validation_group_count"] = 190
    value["authority_sha256"] = trainer.canonical_sha256(
        {key: item for key, item in value.items() if key != "authority_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o444)
    audit = evaluator.validate_input_authority(path, trainer.file_sha256(path))
    assert audit["target_validation_group_count"] == 190
