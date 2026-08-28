from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preregister_smolvla_piper_v7_paired_development import (  # noqa: E402
    ADAPTATION_GROUPS,
    EVALUATION_GROUPS,
    EXPECTED_D250_CANDIDATES,
    LABEL_ACCESS_CONTRACT,
    TARGET_ACTOR_ID,
    TARGET_BODY,
    TARGET_SEED_FORMAT,
    TARGET_SEED_STATUS,
    TOTAL_GROUPS,
    VALIDATION_GROUPS,
    ProtocolError,
    _reject_sensitive_path,
    canonical_sha256,
    file_sha256,
    freeze_protocol,
    paired_condition_order,
    validate_d250_identity,
    validate_forward_preflight,
    validate_frozen_protocol,
    validate_target_seed_manifest,
    validate_v7_activation,
)


SHA = "1" * 64


def signed(value: dict[str, object], key: str) -> dict[str, object]:
    result = copy.deepcopy(value)
    result[key] = canonical_sha256(result)
    return result


def d250_identity(event_spec_sha: str = SHA) -> dict[str, object]:
    groups = []
    for index in range(250):
        groups.append(
            {
                "index": index,
                "seed": index,
                "requested_seed": index,
                "resolved_seed": 1000 + index,
                "path": f"groups/group_{index:03d}.hdf5",
                "candidate_names": list(EXPECTED_D250_CANDIDATES),
                "status": "collected",
            }
        )
    return {
        "format": "etsf_event_branch_collection_identity_v1",
        "schema_version": 5,
        "task": "move_can_pot",
        "body": TARGET_BODY,
        "candidate_count": 4,
        "completed": 250,
        "seed_registry": "explicit_v7_prospective_development",
        "requested_seeds": list(range(250)),
        "resolved_seeds": list(range(1000, 1250)),
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
        "v7_seed_manifest": "/non_sensitive/v7_seed_manifest.json",
        "v7_seed_manifest_sha256": "2" * 64,
        "v7_preregistration": "/non_sensitive/v7_preregistration.json",
        "v7_preregistration_sha256": "3" * 64,
        "event_spec": "/non_sensitive/event_spec.json",
        "event_spec_sha256": event_spec_sha,
        "groups": groups,
        "label_access_contract": "identity_only_no_success_steps_event_or_outcome_fields",
        "hdf5_sha256_pre_evaluation": "not_computed",
    }


def seed_row(split: str, ordinal: int, seed: int) -> dict[str, object]:
    row: dict[str, object] = {
        "ordinal": ordinal,
        "requested_seed": seed,
        "resolved_seed": seed + 100_000,
        "instruction_sha256": "4" * 64,
        "instruction_semantics_receipt_sha256": "5" * 64,
        "initial_scene_state_sha256": "6" * 64,
        "initial_measured_joint_state_sha256": "7" * 64,
        "initial_commanded_drive_target_sha256": "8" * 64,
    }
    identity = {
        "task": "move_can_pot",
        "actor_id": TARGET_ACTOR_ID,
        "target_body": TARGET_BODY,
        "split": split,
        **row,
    }
    row["pair_id"] = canonical_sha256(identity)
    return row


def target_seed_manifest(d250_file_sha: str, d250: dict[str, object]) -> dict[str, object]:
    decoded = validate_d250_identity(d250)
    counts = {
        "adaptation": ADAPTATION_GROUPS,
        "validation": VALIDATION_GROUPS,
        "evaluation": EVALUATION_GROUPS,
    }
    cursor = 10_000
    splits: dict[str, list[dict[str, object]]] = {}
    for split, count in counts.items():
        splits[split] = [seed_row(split, ordinal, cursor + ordinal) for ordinal in range(count)]
        cursor += count
    requested = [int(row["requested_seed"]) for name in counts for row in splits[name]]
    resolved = [int(row["resolved_seed"]) for name in counts for row in splits[name]]
    target_sha = canonical_sha256({"requested": requested, "resolved": resolved})
    value: dict[str, object] = {
        "format": TARGET_SEED_FORMAT,
        "status": TARGET_SEED_STATUS,
        "task": "move_can_pot",
        "actor_id": TARGET_ACTOR_ID,
        "source_body": "aloha",
        "target_body": TARGET_BODY,
        "purpose": "nonfresh_development_only_no_confirmation_claim",
        "label_access_contract": LABEL_ACCESS_CONTRACT,
        "instruction_contract": {
            "mode": "explicit_frozen_per_pair_instruction",
            "episode_info_list_used": False,
            "semantic_receipt_required": True,
            "same_instruction_for_both_conditions": True,
        },
        "splits": splits,
        "d250_exclusion": {
            "identity_manifest_file_sha256": d250_file_sha,
            "identity_sets_sha256": decoded["identity_sets_sha256"],
            "intersection_count": 0,
        },
        "heldout_exclusion_attestation": {
            "status": "verified_disjoint_without_disclosing_heldout_identities",
            "heldout_identity_set_sha256": "9" * 64,
            "target_identity_set_sha256": target_sha,
            "intersection_count": 0,
            "sensitive_identities_included": False,
        },
    }
    return signed(value, "seed_manifest_sha256")


def forward_preflight(implementation_sha: str = SHA) -> dict[str, object]:
    return {
        "format": "smolvla_piper_zero_shot_preflight_v2",
        "status": "passed_forward_only",
        "actor_id": TARGET_ACTOR_ID,
        "authorization": "forward_only",
        "environment_execution_authorized": False,
        "transfer_claim_authorized": False,
        "data_blind": True,
        "candidate_validation": {
            "shape": [4, 50, 14],
            "candidate_sha256": [str(index) * 64 for index in range(1, 5)],
            "all_candidates_distinct": True,
            "max_abs_delta_from_candidate0": 0.1,
            "piper_limits_satisfied": True,
        },
        "shared_prefix_validation": {
            "shape": [4, 960],
            "bit_exact_across_candidates": True,
            "shared_prefix_sha256": "5" * 64,
        },
        "action_mapping_validation": {
            "identity_inferred_from_equal_dimension": False,
            "kinematic_equivalence_claimed": False,
            "physical_equivalence_claimed": False,
            "execution_authorized": False,
        },
        "static_contract": {
            "authorization_ceiling": "forward_only",
            "environment_execution_authorized": False,
            "transfer_claim_authorized": False,
        },
        "implementation_sha256": implementation_sha,
    }


def v7_activation(utility_sha: str) -> dict[str, object]:
    value: dict[str, object] = {
        "format": "etsf_composite_structured_prediction_activation_v1",
        "status": "active_structured_prediction_development_only",
        "evidence_scope": "adaptive_development_only",
        "transfer_claim_authorized": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "action_selector": {
            "authority": "v7_fixed_parameter_free_selector",
            "v8_replacement_authorized": False,
            "v8_success_input_allowed": False,
            "v8_regress_input_allowed": False,
            "duration_v2_input_allowed": False,
            "deployment_candidate_count": 4,
            "implementation_sha256": utility_sha,
        },
        "inactive_or_fallback": {
            "success": {"status": "inactive"},
            "recovery": {"status": "inactive"},
            "object": {"status": "fallback_only"},
            "total_uncertainty": {"status": "unavailable"},
        },
    }
    return signed(value, "activation_sha256")


def json_file(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def artifacts(tmp_path: Path) -> dict[str, Path]:
    event_spec = json_file(
        tmp_path / "event_spec.json",
        {"chains": {"move_can_pot": {"chain": ["start", "done"]}}, "calibration": {"move_can_pot": {}}},
    )
    d250 = d250_identity(file_sha256(event_spec))
    d250_path = json_file(tmp_path / "d250_identity.json", d250)
    seeds = target_seed_manifest(file_sha256(d250_path), d250)
    utility = tmp_path / "utility.py"
    utility.write_text("UTILITY = 1\n", encoding="utf-8")
    plugin = tmp_path / "plugin.py"
    plugin.write_text("PLUGIN = 1\n", encoding="utf-8")
    collector = tmp_path / "collector.py"
    collector.write_text("COLLECTOR = 1\n", encoding="utf-8")
    preflight_implementation = tmp_path / "preflight.py"
    preflight_implementation.write_text("PREFLIGHT = 1\n", encoding="utf-8")
    return {
        "d250_identity_path": d250_path,
        "target_seed_manifest_path": json_file(tmp_path / "target_seeds.json", seeds),
        "forward_preflight_path": json_file(
            tmp_path / "forward_preflight.json",
            forward_preflight(file_sha256(preflight_implementation)),
        ),
        "v7_activation_path": json_file(tmp_path / "v7_activation.json", v7_activation(file_sha256(utility))),
        "event_spec_path": event_spec,
        "forward_preflight_implementation_path": preflight_implementation,
        "v7_utility_path": utility,
        "actor_agnostic_plugin_path": plugin,
        "smolvla_collector_path": collector,
    }


def test_d250_identity_binds_only_label_free_fields() -> None:
    result = validate_d250_identity(d250_identity())
    assert result["groups"] == 250
    assert result["labels_read"] is False
    assert result["candidate_names"] == list(EXPECTED_D250_CANDIDATES)


@pytest.mark.parametrize("key", ["success", "reward", "event_id", "outcomes", "duration"])
def test_d250_identity_rejects_any_outcome_field(key: str) -> None:
    value = d250_identity()
    value["groups"][0][key] = 1  # type: ignore[index]
    with pytest.raises(ProtocolError, match="forbidden label"):
        validate_d250_identity(value)


def test_target_seed_manifest_has_fixed_disjoint_split_sizes(tmp_path: Path) -> None:
    d250 = d250_identity()
    path = json_file(tmp_path / "identity.json", d250)
    result = validate_target_seed_manifest(
        target_seed_manifest(file_sha256(path), d250),
        d250_identity_file_sha256=file_sha256(path),
        d250=validate_d250_identity(d250),
    )
    assert len(result["requested"]) == TOTAL_GROUPS == 530
    assert list(map(len, result["splits"].values())) == [80, 50, 400]
    assert result["labels_read"] is False


def test_target_seed_manifest_rejects_overlap_and_disclosure_tamper(tmp_path: Path) -> None:
    d250 = d250_identity()
    path = json_file(tmp_path / "identity.json", d250)
    value = target_seed_manifest(file_sha256(path), d250)
    changed = copy.deepcopy(value)
    changed["splits"]["adaptation"][0]["requested_seed"] = 0  # type: ignore[index]
    changed["seed_manifest_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "seed_manifest_sha256"}
    )
    with pytest.raises(ProtocolError):
        validate_target_seed_manifest(
            changed,
            d250_identity_file_sha256=file_sha256(path),
            d250=validate_d250_identity(d250),
        )
    changed = copy.deepcopy(value)
    changed["heldout_exclusion_attestation"]["sensitive_identities_included"] = True  # type: ignore[index]
    changed["seed_manifest_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "seed_manifest_sha256"}
    )
    with pytest.raises(ProtocolError, match="attestation"):
        validate_target_seed_manifest(
            changed,
            d250_identity_file_sha256=file_sha256(path),
            d250=validate_d250_identity(d250),
        )


def test_forward_preflight_remains_forward_only() -> None:
    assert validate_forward_preflight(forward_preflight())["authorization_ceiling"] == "forward_only"
    changed = forward_preflight()
    changed["environment_execution_authorized"] = True
    with pytest.raises(ProtocolError, match="forward-only"):
        validate_forward_preflight(changed)


def test_condition_order_is_label_free_and_deterministic() -> None:
    first = paired_condition_order("a" * 64)
    second = paired_condition_order("a" * 64)
    assert first == second
    assert set(first) == {"direct_smolvla_policy", "v7_event_world_model_selector"}
    with pytest.raises(ProtocolError, match="pair_id"):
        paired_condition_order("not-a-sha")


def test_v7_activation_keeps_failed_heads_out_of_ranking() -> None:
    value = v7_activation(SHA)
    result = validate_v7_activation(value)
    assert result["success_recovery_object_inputs_allowed"] is False
    changed = copy.deepcopy(value)
    changed["action_selector"]["v8_success_input_allowed"] = True  # type: ignore[index]
    changed["activation_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "activation_sha256"}
    )
    with pytest.raises(ProtocolError, match="capability boundary"):
        validate_v7_activation(changed)


def test_freezer_records_runtime_truth_and_claim_ceiling(tmp_path: Path) -> None:
    result = freeze_protocol(**artifacts(tmp_path))
    validate_frozen_protocol(result)
    assert result["stages"]["adaptation"]["plugin_during_first_20_groups_allowed"] is False
    assert result["known_runtime_audit"]["piper_observation_state14"] == (
        "commanded_drive_target_not_measured_qpos"
    )
    assert result["known_runtime_audit"]["physical_time_duration_claim_allowed"] is False
    assert result["known_runtime_audit"]["explicit_seed_reset_refreshes_episode_info_list"] is False
    assert result["known_runtime_audit"]["current_v7_native_smolvla_960d_input_allowed"] is False
    assert result["metrics"]["event_prediction"]["duration_quantity"] == (
        "decision_step_duration_not_physical_time"
    )
    assert result["sample_size_basis"]["fixed_evaluation_pairs"] == 400
    assert "event_world_model_itself_transfers_across_bodies" in result["claim_boundary"][
        "not_authorized_by_this_experiment"
    ]
    assert result["source_readiness"]["execution_authorized_by_this_protocol"] is False


def test_freezer_rejects_utility_activation_sha_mismatch(tmp_path: Path) -> None:
    values = artifacts(tmp_path)
    values["v7_utility_path"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="implementation SHA differ"):
        freeze_protocol(**values)


def test_fresh_and_confirmation_paths_are_rejected(tmp_path: Path) -> None:
    for name in ("Fresh50", "confirmation_results"):
        path = tmp_path / name / "identity.json"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ProtocolError, match="Fresh/confirmation"):
            _reject_sensitive_path(path.resolve(), "artifact")


def test_protocol_signature_tamper_fails(tmp_path: Path) -> None:
    result = freeze_protocol(**artifacts(tmp_path))
    result["stages"]["evaluation"]["groups"] = 399
    with pytest.raises(ProtocolError, match="signature"):
        validate_frozen_protocol(result)
