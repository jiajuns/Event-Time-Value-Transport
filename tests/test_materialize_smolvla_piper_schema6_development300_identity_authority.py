from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_smolvla_piper_schema6_development300_identity_authority as identity  # noqa: E402
import smolvla_piper_schema6_runtime_adapter_v2 as runtime_adapter  # noqa: E402
from preregister_smolvla_piper_schema6_target_development300 import (  # noqa: E402
    INSTRUCTION,
    SPLIT_COUNTS,
    build_preregistration,
    canonical_sha256,
)


HELDOUT_COMMITMENT = "b" * 64


def write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def attestation(target_role: str, target_sha256: str) -> dict[str, Any]:
    return signed(
        {
            "format": "etsf_private_identity_disjoint_attestation_v1",
            "status": "verified_disjoint_without_disclosing_heldout_identities",
            "target_role": target_role,
            "heldout_identity_set_sha256": HELDOUT_COMMITMENT,
            "target_identity_set_sha256": target_sha256,
            "intersection_count": 0,
            "sensitive_identities_included": False,
        },
        "attestation_sha256",
    )


def stable_reset(requested_seed: int, instruction: str) -> dict[str, Any]:
    assert instruction == INSTRUCTION
    return {
        "setup_status": "stable",
        "requested_seed": requested_seed,
        "resolved_seed": requested_seed,
        "instruction_observed": instruction,
        "scene_state": {
            "can_pose": [1.23456789012345, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pot_pose": [2.34567890123456, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        "measured_joint_state": [3.45678901234567] * 14,
        "commanded_drive_target": [4.56789012345678] * 14,
    }


def make_runtime(tmp_path: Path, *, max_episode_steps: int = 200) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots: dict[str, str] = {}
    for role in (
        "rlinf_root",
        "robotwin_root",
        "robotwin_code",
        "lerobot_root",
        "model_path",
        "vlm_metadata_path",
    ):
        directory = tmp_path / f"synthetic-{role.replace('_', '-')}"
        directory.mkdir()
        (directory / "bound.txt").write_text(f"synthetic {role}\n", encoding="utf-8")
        roots[role] = str(directory.resolve())

    source_names = {
        "rlinf_robotwin_env": "synthetic-rlinf-env.py",
        "robotwin_vector_env": "synthetic-vector-env.py",
        "robotwin_base_task": "synthetic-base-task.py",
        "robotwin_robot_controller": "synthetic-controller.py",
        "robotwin_move_can_pot": "synthetic-task.py",
    }
    sources: dict[str, dict[str, str]] = {}
    for role, filename in source_names.items():
        path = tmp_path / filename
        path.write_text(f"# {role}\n", encoding="utf-8")
        path.chmod(0o444)
        sources[role] = {
            "path": str(path.resolve()),
            "sha256": runtime_adapter.file_sha256(path),
        }
    seed_registry = tmp_path / "synthetic-seed-registry.json"
    seed_registry.write_text("{}\n", encoding="utf-8")
    seed_registry.chmod(0o444)
    base: dict[str, Any] = {
        "format": runtime_adapter.RUNTIME_CONTRACT_FORMAT,
        "status": runtime_adapter.RUNTIME_CONTRACT_STATUS,
        "runtime_roots": roots,
        "runtime_source_artifacts": sources,
        "eval_seed_registry": {
            "path": str(seed_registry.resolve()),
            "sha256": runtime_adapter.file_sha256(seed_registry),
        },
        "measured_joint_state_channel": runtime_adapter.DEFAULT_MEASURED_CHANNEL,
        "gpu_index": 0,
        "max_episode_steps": max_episode_steps,
        "offline_model_loading": True,
        "piper_action_bounds": [[-1.0, 1.0]] * 14,
        "model_tree_sha256": runtime_adapter.directory_tree_sha256(
            Path(roots["model_path"])
        ),
        "vlm_metadata_tree_sha256": runtime_adapter.directory_tree_sha256(
            Path(roots["vlm_metadata_path"])
        ),
        "reset_scratch_path": "/tmp/etsf-dev300-synthetic-scratch",
        "test_or_evaluation_execution_authorized": False,
        "fresh_or_confirmation_inputs_accepted": False,
    }
    return signed(base, "runtime_contract_sha256")


def make_reset_authority(
    tmp_path: Path, *, max_episode_steps: int = 200
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    preregistration = build_preregistration()
    runtime = make_runtime(tmp_path, max_episode_steps=max_episode_steps)
    requested = [row["requested_seed"] for row in preregistration["groups"]]
    candidate = attestation(
        identity.CANDIDATE_TARGET_ROLE, canonical_sha256(requested)
    )
    paths = {
        "preregistration": write_json(tmp_path / "development300-prereg.json", preregistration),
        "runtime": write_json(tmp_path / "runtime-v2b.json", runtime),
        "candidate": write_json(tmp_path / "candidate-attestation.json", candidate),
    }
    adapter = tmp_path / "synthetic-reset-adapter.py"
    adapter.write_text("def build_reset_only_adapter(**kwargs):\n    raise AssertionError\n", encoding="utf-8")
    adapter.chmod(0o444)
    paths["adapter"] = adapter
    authority = identity.build_reset_authority(
        preregistration_path=paths["preregistration"],
        expected_preregistration_file_sha256=identity.file_sha256(paths["preregistration"]),
        expected_preregistration_sha256=preregistration["preregistration_sha256"],
        runtime_contract_path=paths["runtime"],
        expected_runtime_contract_file_sha256=identity.file_sha256(paths["runtime"]),
        expected_runtime_contract_sha256=runtime["runtime_contract_sha256"],
        candidate_disjoint_attestation_path=paths["candidate"],
        expected_candidate_attestation_file_sha256=identity.file_sha256(paths["candidate"]),
        reset_adapter_path=adapter,
        expected_reset_adapter_file_sha256=identity.file_sha256(adapter),
        verify_runtime_files=True,
    )
    return preregistration, runtime, authority, paths


def complete_receipt(
    preregistration: Mapping[str, Any],
    runtime: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    authority_file_sha256: str = "a" * 64,
) -> dict[str, Any]:
    return identity.resolve_identities(
        preregistration=preregistration,
        runtime_contract=runtime,
        reset_authority=authority,
        reset_once=stable_reset,
        reset_authority_file_sha256=authority_file_sha256,
        verify_runtime_files=True,
    )


def test_reset_authority_consumes_all_three_external_contracts_and_is_reset_only(
    tmp_path: Path,
) -> None:
    preregistration, runtime, authority, _ = make_reset_authority(tmp_path)
    audit = identity.validate_reset_authority(
        authority,
        preregistration=preregistration,
        runtime_contract=runtime,
        verify_runtime_files=True,
    )
    assert audit["preregistration_sha256"] == preregistration["preregistration_sha256"]
    assert audit["runtime_contract_sha256"] == runtime["runtime_contract_sha256"]
    assert audit["heldout_identity_set_sha256"] == HELDOUT_COMMITMENT
    assert authority["permissions"] == {
        "environment_construct_allowed": True,
        "environment_reset_allowed": True,
        "maximum_reset_calls": 300,
        "environment_step_allowed": False,
        "policy_import_or_forward_allowed": False,
        "reward_success_event_outcome_trajectory_or_label_read_allowed": False,
        "collection_allowed": False,
        "evaluation400_identity_or_execution_allowed": False,
    }


def test_all_stable_resolution_is_ordered_unique_zero_step_and_hash_only(
    tmp_path: Path,
) -> None:
    preregistration, runtime, authority, _ = make_reset_authority(tmp_path)
    calls: list[int] = []

    def reset_once(seed: int, instruction: str) -> dict[str, Any]:
        calls.append(seed)
        return stable_reset(seed, instruction)

    receipt = identity.resolve_identities(
        preregistration=preregistration,
        runtime_contract=runtime,
        reset_authority=authority,
        reset_once=reset_once,
        reset_authority_file_sha256="a" * 64,
        verify_runtime_files=True,
    )
    requested = [row["requested_seed"] for row in preregistration["groups"]]
    assert calls == requested
    assert receipt["status"] == identity.RESET_COMPLETE_STATUS
    assert receipt["stable_selected_count"] == 300
    assert receipt["selected_split_counts"] == SPLIT_COUNTS
    assert receipt["environment_reset_calls"] == 300
    assert receipt["environment_step_calls"] == 0
    assert receipt["policy_import_or_forward_calls"] == 0
    assert receipt["reward_success_event_outcome_trajectory_or_label_fields_read"] == 0
    serialized = json.dumps(receipt, sort_keys=True)
    for raw_value in ("1.23456789012345", "2.34567890123456", "3.45678901234567"):
        assert raw_value not in serialized
    assert identity.validate_identity_receipt(
        receipt,
        preregistration=preregistration,
        reset_authority=authority,
    )["complete"]


def test_unstable_setup_is_recorded_and_processing_advances_without_replacement(
    tmp_path: Path,
) -> None:
    preregistration, runtime, authority, _ = make_reset_authority(tmp_path)
    requested = [row["requested_seed"] for row in preregistration["groups"]]
    calls: list[int] = []

    def reset_once(seed: int, instruction: str) -> dict[str, Any]:
        calls.append(seed)
        if len(calls) == 17:
            return {"setup_status": "unstable", "requested_seed": seed}
        return stable_reset(seed, instruction)

    receipt = identity.resolve_identities(
        preregistration=preregistration,
        runtime_contract=runtime,
        reset_authority=authority,
        reset_once=reset_once,
        reset_authority_file_sha256="a" * 64,
        verify_runtime_files=True,
    )
    assert calls == requested
    assert receipt["status"] == identity.RESET_INSUFFICIENT_STATUS
    assert receipt["attempt_count"] == 300
    assert receipt["stable_selected_count"] == 299
    assert receipt["unstable_setup_count"] == 1
    assert receipt["collection_identity_membership_frozen"] is False
    selected = attestation(
        identity.SELECTED_TARGET_ROLE, receipt["selected_identity_set_sha256"]
    )
    with pytest.raises(identity.Development300IdentityError, match="all 300"):
        identity.build_collection_identity_authority(
            preregistration=preregistration,
            reset_authority=authority,
            identity_receipt=receipt,
            selected_disjoint_attestation=selected,
        )


def test_implicit_retry_and_forbidden_nested_result_fail_closed(tmp_path: Path) -> None:
    preregistration, runtime, authority, _ = make_reset_authority(tmp_path)

    def retry(seed: int, instruction: str) -> dict[str, Any]:
        row = stable_reset(seed, instruction)
        row["resolved_seed"] = seed + 1
        return row

    with pytest.raises(identity.Development300IdentityError, match="implicit seed retry"):
        identity.resolve_identities(
            preregistration=preregistration,
            runtime_contract=runtime,
            reset_authority=authority,
            reset_once=retry,
            reset_authority_file_sha256="a" * 64,
            verify_runtime_files=True,
        )

    def forbidden(seed: int, instruction: str) -> dict[str, Any]:
        row = stable_reset(seed, instruction)
        row["scene_state"]["label"] = 1
        return row

    with pytest.raises(identity.Development300IdentityError, match="forbidden"):
        identity.resolve_identities(
            preregistration=preregistration,
            runtime_contract=runtime,
            reset_authority=authority,
            reset_once=forbidden,
            reset_authority_file_sha256="a" * 64,
            verify_runtime_files=True,
        )


def test_receipt_tampering_and_selected_attestation_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    preregistration, runtime, authority, _ = make_reset_authority(tmp_path)
    receipt = complete_receipt(preregistration, runtime, authority)
    tampered = dict(receipt)
    tampered["selected_rows"] = [dict(row) for row in receipt["selected_rows"]]
    tampered["selected_rows"][0]["split"] = "formal_target_validation"
    tampered.pop("receipt_sha256")
    tampered = signed(tampered, "receipt_sha256")
    with pytest.raises(identity.Development300IdentityError):
        identity.validate_identity_receipt(
            tampered,
            preregistration=preregistration,
            reset_authority=authority,
        )

    wrong_selected = attestation(identity.SELECTED_TARGET_ROLE, "c" * 64)
    with pytest.raises(identity.Development300IdentityError, match="attestation"):
        identity.build_collection_identity_authority(
            preregistration=preregistration,
            reset_authority=authority,
            identity_receipt=receipt,
            selected_disjoint_attestation=wrong_selected,
        )


def test_materialization_freezes_80_30_190_and_300_times_four_without_execution(
    tmp_path: Path,
) -> None:
    preregistration, runtime, authority, paths = make_reset_authority(tmp_path)
    authority_path = write_json(tmp_path / "reset-authority.json", authority)
    receipt = complete_receipt(
        preregistration,
        runtime,
        authority,
        authority_file_sha256=identity.file_sha256(authority_path),
    )
    selected = attestation(
        identity.SELECTED_TARGET_ROLE, receipt["selected_identity_set_sha256"]
    )
    receipt_path = write_json(tmp_path / "identity-receipt.json", receipt)
    selected_path = write_json(tmp_path / "selected-attestation.json", selected)
    future_root = tmp_path / "future-schema6-development300"
    output = tmp_path / "frozen-development300"
    result = identity.materialize_collection(
        preregistration_path=paths["preregistration"],
        expected_preregistration_file_sha256=identity.file_sha256(paths["preregistration"]),
        reset_authority_path=authority_path,
        expected_reset_authority_file_sha256=identity.file_sha256(authority_path),
        identity_receipt_path=receipt_path,
        expected_identity_receipt_file_sha256=identity.file_sha256(receipt_path),
        selected_attestation_path=selected_path,
        expected_selected_attestation_file_sha256=identity.file_sha256(selected_path),
        future_collection_root=future_root,
        output_directory=output,
    )
    collection = json.loads(
        Path(result["collection_preregistration_path"]).read_text(encoding="utf-8")
    )
    frozen = json.loads(
        Path(result["collection_identity_authority_path"]).read_text(encoding="utf-8")
    )
    assert frozen["split_counts"] == SPLIT_COUNTS
    assert frozen["identity_freeze_evidence"] == {
        "environment_reset_calls": 300,
        "environment_step_calls": 0,
        "policy_import_or_forward_calls": 0,
        "reward_success_event_outcome_trajectory_or_label_fields_read": 0,
        "hdf5_files_opened": 0,
    }
    assert collection["command_count"] == 300
    assert collection["planned_candidate_branches"] == 1200
    assert [row["requested_seed"] for row in collection["commands"]] == [
        row["requested_seed"] for row in preregistration["groups"]
    ]
    assert all(
        row["candidate_original_indices"] == [0, 1, 2, 3]
        and row["candidate_branch_count"] == 4
        and row["capability"]["execution_authorized_by_preregistration"] is False
        and row["capability"]["evaluation400"] is False
        for row in collection["commands"]
    )
    assert collection["execution_boundary"]["separate_bound_runner_authority_required"]
    assert collection["execution_boundary"]["evaluation400_commands_generated"] == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert stat.S_IMODE(
        Path(result["collection_identity_authority_path"]).stat().st_mode
    ) == 0o444
    assert stat.S_IMODE(
        Path(result["collection_preregistration_path"]).stat().st_mode
    ) == 0o444
    assert not future_root.exists()


def test_non_full_horizon_runtime_and_candidate_hash_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(identity.Development300IdentityError, match="full-horizon"):
        make_reset_authority(tmp_path / "short", max_episode_steps=199)

    candidate_root = tmp_path / "candidate-mismatch"
    candidate_root.mkdir()
    preregistration = build_preregistration()
    runtime = make_runtime(candidate_root)
    prereg_path = write_json(candidate_root / "development300-prereg.json", preregistration)
    runtime_path = write_json(candidate_root / "runtime-v2b.json", runtime)
    bad_attestation = attestation(identity.CANDIDATE_TARGET_ROLE, "d" * 64)
    attestation_path = write_json(candidate_root / "candidate-attestation.json", bad_attestation)
    adapter_path = candidate_root / "synthetic-reset-adapter.py"
    adapter_path.write_text("# synthetic\n", encoding="utf-8")
    adapter_path.chmod(0o444)
    with pytest.raises(identity.Development300IdentityError, match="attestation"):
        identity.build_reset_authority(
            preregistration_path=prereg_path,
            expected_preregistration_file_sha256=identity.file_sha256(prereg_path),
            expected_preregistration_sha256=preregistration["preregistration_sha256"],
            runtime_contract_path=runtime_path,
            expected_runtime_contract_file_sha256=identity.file_sha256(runtime_path),
            expected_runtime_contract_sha256=runtime["runtime_contract_sha256"],
            candidate_disjoint_attestation_path=attestation_path,
            expected_candidate_attestation_file_sha256=identity.file_sha256(attestation_path),
            reset_adapter_path=adapter_path,
            expected_reset_adapter_file_sha256=identity.file_sha256(adapter_path),
            verify_runtime_files=True,
        )
