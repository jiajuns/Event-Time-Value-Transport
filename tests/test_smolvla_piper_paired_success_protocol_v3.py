from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for directory in (SCRIPTS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge  # noqa: E402
import launch_smolvla_piper_schema6_post_collection_v3 as post  # noqa: E402
import smolvla_piper_paired_success_protocol_v3 as protocol  # noqa: E402
import test_smolvla_piper_paired_success_protocol_v2 as v2_fixture  # noqa: E402


def signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result[field] = post.canonical_sha256(result)
    return result


def write_json(path: Path, value: Mapping[str, Any], *, frozen: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    path.chmod(0o444 if frozen else 0o644)
    return path


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, logical: str) -> dict[str, str]:
    return {"path": str(path), "file_sha256": file_sha(path), "logical_sha256": logical}


def stage_result(name: str, pid: int) -> dict[str, Any]:
    lifecycle = signed(
        {
            "popen_attempted": True, "popen_reached": True,
            "process_pid": pid, "process_pgid": pid,
            "process_group_isolated": True, "returncode": 0,
            "direct_process_reaped": True, "process_group_reaped": True,
            "binding_status": "bound_reaped",
        },
        "lifecycle_sha256",
    )
    base = {
        "stage": name, "returncode": 0, "command_sha256": f"{pid + 10:064x}",
        "lifecycle": lifecycle, "log_file_sha256": f"{pid + 20:064x}",
        "run_exit_file_sha256": f"{pid + 30:064x}",
    }
    return {**base, "result_sha256": post.canonical_sha256(base)}


def member_receipt(
    *, index: int, source_path: Path, adapter_path: Path,
    adapter_seed: int, shared: Mapping[str, str], prediction: Mapping[str, Any],
) -> dict[str, Any]:
    source_rank_contract = signed(
        {
            "format": "etsf_source63_composite_candidate_rank_score_v1",
            "status": "frozen_exact_source63_training_score_scientific_rank_only",
            "source_checkpoint_file_sha256": file_sha(source_path),
            "success_temperature": 1.0,
            "source_rank_numeric_contract": protocol.SOURCE_RANK_NUMERIC_CONTRACT,
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
        },
        "contract_sha256",
    )
    base = {
        "format": post.MEMBER_FORMAT,
        "status": "complete_frozen_development300_internal_validation_adapter",
        "member_index": index, "member_seed": adapter_seed,
        "split_profile": post.SPLIT_PROFILE, "split_profile_version": 3,
        "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_sha256": file_sha(source_path),
        "source_checkpoint_role": "native_r7h_individual_source_member",
        "training_manifest_sha256": shared["training_manifest_sha256"],
        "split_sha256": shared["split_sha256"],
        "source_ensemble_contract_sha256": shared["source_ensemble_contract_sha256"],
        "summary_path": f"/sealed/member_{index}/summary.json",
        "summary_file_sha256": f"{100 + index:064x}",
        "summary_sha256": f"{110 + index:064x}",
        "checkpoint_path": str(adapter_path),
        "checkpoint_file_sha256": file_sha(adapter_path),
        "validation_predictions_path": f"/sealed/member_{index}/internal_predictions.npz",
        "validation_predictions_file_sha256": f"{120 + index:064x}",
        "validation_predictions_logical_sha256": f"{130 + index:064x}",
        "validation_labels_path": f"/sealed/member_{index}/internal_validation.npz",
        "validation_labels_file_sha256": f"{140 + index:064x}",
        "validation_labels_logical_sha256": f"{150 + index:064x}",
        "validation_identity_set_sha256": "6" * 64,
        "validation_lane": "adaptation_derived_internal_validation_only",
        "internal_validation_group_count": 30,
        "sealed_formal_target_validation_group_count": 190,
        "prediction_contract": dict(prediction),
        "source_rank_score_contract": source_rank_contract,
        "source_rank_score_contract_sha256": source_rank_contract[
            "contract_sha256"
        ],
        "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_labels_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_release_condition": (
            "external_authority_after_all_five_adapter_checkpoints_are_frozen"
        ),
        "lobo_or_aggregate_checkpoint_used": False,
        "stage_result_sha256": f"{170 + index:064x}",
    }
    return signed(base, "receipt_sha256")


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    v2 = v2_fixture.strict_fixture(tmp_path)
    safe = Path(v2["root"])
    root = safe / "post_v3"
    (root / "_watcher").mkdir(parents=True)
    (root / "members").mkdir()
    (root / "formal190" / "evaluator_stage" / "result").mkdir(parents=True)
    (root / "handoff").mkdir()

    bridge_path = Path(v2["kwargs"]["identity_bridge_path"])
    bridge_value = json.loads(bridge_path.read_text(encoding="utf-8"))
    nested_runtime_contract = signed(
        {"max_episode_steps": 200}, "runtime_contract_sha256"
    )
    target_manifest_path = Path(
        bridge_value["dependencies"]["target_manifest"]["path"]
    )
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    target_manifest.pop("seed_manifest_sha256")
    target_manifest["provenance"]["runtime_contract_sha256"] = (
        nested_runtime_contract["runtime_contract_sha256"]
    )
    write_json(
        target_manifest_path,
        signed(target_manifest, "seed_manifest_sha256"),
    )
    ensemble_path = Path(bridge_value["dependencies"]["ensemble_manifest"]["path"])
    calibration_path = Path(bridge_value["dependencies"]["calibration"]["path"])
    head_path = Path(bridge_value["dependencies"]["head_support"]["path"])
    calibration_receipt_path = Path(
        bridge_value["dependencies"]["calibration_receipt"]["path"]
    )
    original_calibration_receipt_path = calibration_receipt_path
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    ensemble.pop("ensemble_manifest_sha256")
    shared = dict(ensemble["shared_contract"])
    prediction = {
        "duration_target_transform": "log1p_decision_steps",
        "next_event_observation_mask": "duration_observed",
        "success_target": "eventual_final_branch_success_repeated_per_transition",
        "recovery_target": "conditional_recovery_given_operational_regress",
        "recovery_observation_mask": "recovery_observed_and_regress",
        "recovery_shared_transition_stop_gradient": True,
        "recovery_enters_primary_before_calibration": False,
        "recovery_head_trained": True,
        "object_prediction_space": "physical_delta_xyz_m",
        "object_source_normalization_sha256": "e" * 64,
        "object_observed_policy": "row_enabled_only_if_all_selected_xyz_are_valid",
    }
    shared["prediction_contract_sha256"] = protocol.canonical_sha256(prediction)
    ensemble["shared_contract"] = shared
    ensemble["prediction_contract"] = prediction

    members: list[dict[str, Any]] = []
    member_specs: list[tuple[Path, str]] = []
    source_shas: list[str] = []
    for index, seed in enumerate(post.SOURCE_MEMBER_SEEDS):
        source_path = safe / f"r7h_source_member_{index}.pt"
        source_path.write_bytes(f"r7h-source-{index}".encode("ascii"))
        source_path.chmod(0o444)
        adapter_path = Path(ensemble["members"][index]["checkpoint_path"])
        adapter_path.chmod(0o444)
        ensemble["members"][index]["member_seed"] = seed
        receipt_value = member_receipt(
            index=index, source_path=source_path, adapter_path=adapter_path,
            adapter_seed=seed, shared=shared, prediction=prediction,
        )
        ensemble["members"][index]["source_rank_score_contract"] = dict(
            receipt_value["source_rank_score_contract"]
        )
        ensemble["members"][index]["source_rank_score_contract_sha256"] = (
            receipt_value["source_rank_score_contract_sha256"]
        )
        receipt_path = root / "members" / f"member_{index}" / "final_receipt.json"
        write_json(receipt_path, receipt_value)
        members.append(receipt_value)
        member_specs.append((receipt_path, file_sha(receipt_path)))
        source_shas.append(file_sha(source_path))

    # Derive the ordered member authority from the five rewritten ensemble
    # contracts themselves.  Every downstream mirror is then re-signed from
    # this one authority instead of retaining the v2 fixture's member SHAs.
    source_rank_member_authority = {
        "source_rank_numeric_contract": protocol.SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": index,
                "source_checkpoint_file_sha256": row[
                    "source_rank_score_contract"
                ]["source_checkpoint_file_sha256"],
                "source_rank_score_contract_sha256": row[
                    "source_rank_score_contract_sha256"
                ],
                "success_temperature": row["source_rank_score_contract"][
                    "success_temperature"
                ],
            }
            for index, row in enumerate(ensemble["members"])
        ],
    }
    source_rank_member_authority_sha256 = post.canonical_sha256(
        source_rank_member_authority
    )

    calibration_result_root = root / "calibration" / "calibrator_stage" / "result"
    old_calibration_path, old_head_path = calibration_path, head_path
    ensemble_path = calibration_result_root / "ensemble_manifest.json"
    calibration_path = calibration_result_root / "calibration.json"
    head_path = calibration_result_root / "paired_head_support.json"
    root_ranker_path = calibration_result_root / "formal190_root_group_ranker.json"
    calibration_receipt_path = calibration_result_root / "final_receipt.json"

    calibration = json.loads(old_calibration_path.read_text(encoding="utf-8"))
    calibration.pop("calibration_sha256")
    root_ranker = dict(calibration["root_group_ranker"])
    root_ranker.pop("root_group_ranker_sha256")
    root_ranker["source_rank_numeric_contract"] = (
        protocol.SOURCE_RANK_NUMERIC_CONTRACT
    )
    root_ranker["source_rank_member_authority"] = copy.deepcopy(
        source_rank_member_authority
    )
    root_ranker["source_rank_member_authority_sha256"] = (
        source_rank_member_authority_sha256
    )
    root_ranker = signed(root_ranker, "root_group_ranker_sha256")
    write_json(root_ranker_path, root_ranker)
    root_ranker_file_sha256 = file_sha(root_ranker_path)

    calibration["prediction_contract"] = prediction
    calibration["root_group_ranker"] = root_ranker
    calibration["source_rank_numeric_contract"] = (
        protocol.SOURCE_RANK_NUMERIC_CONTRACT
    )
    calibration["source_rank_member_authority"] = copy.deepcopy(
        source_rank_member_authority
    )
    calibration["source_rank_member_authority_sha256"] = (
        source_rank_member_authority_sha256
    )
    calibration = signed(calibration, "calibration_sha256")
    write_json(calibration_path, calibration)

    head = json.loads(old_head_path.read_text(encoding="utf-8"))
    write_json(head_path, head)

    ensemble["source_rank_numeric_contract"] = (
        protocol.SOURCE_RANK_NUMERIC_CONTRACT
    )
    ensemble["source_rank_member_authority"] = copy.deepcopy(
        source_rank_member_authority
    )
    ensemble["source_rank_member_authority_sha256"] = (
        source_rank_member_authority_sha256
    )
    ensemble["calibration_sha256"] = calibration["calibration_sha256"]
    ensemble["root_group_ranker"] = {
        "path": str(root_ranker_path),
        "file_sha256": root_ranker_file_sha256,
        "logical_sha256": root_ranker["root_group_ranker_sha256"],
        "enabled_for_primary": True,
    }
    ensemble["root_group_ranker_path"] = str(root_ranker_path)
    ensemble["root_group_ranker_file_sha256"] = root_ranker_file_sha256
    ensemble["root_group_ranker_sha256"] = root_ranker[
        "root_group_ranker_sha256"
    ]
    ensemble = signed(ensemble, "ensemble_manifest_sha256")
    write_json(ensemble_path, ensemble)

    calibration_authority_members = []
    evaluator_authority_members = []
    for index, member in enumerate(members):
        calibration_authority_members.append({
            "member_index": index, "member_seed": member["member_seed"],
            **{key: shared[key] for key in shared},
            "checkpoint_path": member["checkpoint_path"],
            "checkpoint_file_sha256": member["checkpoint_file_sha256"],
            "validation_predictions_path": f"/sealed/formal/member_{index}.npz",
            "validation_predictions_file_sha256": f"{200 + index:064x}",
            "source_rank_score_contract": dict(
                member["source_rank_score_contract"]
            ),
            "source_rank_score_contract_sha256": member[
                "source_rank_score_contract_sha256"
            ],
        })
        evaluator_authority_members.append({
            "member_index": index, "member_seed": member["member_seed"],
            "adapter_checkpoint": {
                "path": member["checkpoint_path"],
                "file_sha256": member["checkpoint_file_sha256"],
            },
            "source_checkpoint": {
                "path": member["source_checkpoint_path"],
                "file_sha256": member["source_checkpoint_sha256"],
            },
            "member_receipt": {
                "path": str(member_specs[index][0]),
                "file_sha256": member_specs[index][1],
                "logical_sha256": member["receipt_sha256"],
            },
            "training_manifest_sha256": shared["training_manifest_sha256"],
            "split_sha256": shared["split_sha256"],
            "source_ensemble_contract_sha256": shared["source_ensemble_contract_sha256"],
            "prediction_contract": prediction,
            "source_rank_score_contract": dict(
                member["source_rank_score_contract"]
            ),
            "source_rank_score_contract_sha256": member[
                "source_rank_score_contract_sha256"
            ],
        })
    calibration_authority = signed({
        "format": post.calibrator.INPUT_FORMAT,
        "status": post.calibrator.INPUT_STATUS, "lane": "validation_only",
        "member_count": 5, "shared_contract": shared,
        "prediction_contract": prediction,
        "source_rank_numeric_contract": protocol.SOURCE_RANK_NUMERIC_CONTRACT,
        "validation_identity_set_sha256": "d" * 64,
        "labels_path": "/sealed/formal/common_targets.npz",
        "labels_file_sha256": "9" * 64, "members": calibration_authority_members,
        "test_artifacts_read": False, "fresh_artifacts_read": False,
        "confirmation_artifacts_read": False,
    }, "input_authority_sha256")
    calibration_authority_path = root / "formal190" / "calibration_authority.json"
    write_json(calibration_authority_path, calibration_authority)

    calibration_receipt = json.loads(
        original_calibration_receipt_path.read_text(encoding="utf-8")
    )
    calibration_receipt.pop("receipt_sha256")
    calibration_receipt["input_authority_path"] = str(calibration_authority_path)
    calibration_receipt["input_authority_file_sha256"] = file_sha(calibration_authority_path)
    calibration_receipt["input_authority_sha256"] = calibration_authority[
        "input_authority_sha256"
    ]
    calibration_receipt["shared_contract"] = shared
    calibration_receipt["prediction_contract_sha256"] = shared[
        "prediction_contract_sha256"
    ]
    calibration_receipt["calibration_path"] = str(calibration_path)
    calibration_receipt["calibration_file_sha256"] = file_sha(calibration_path)
    calibration_receipt["calibration_sha256"] = calibration[
        "calibration_sha256"
    ]
    calibration_receipt["head_support_path"] = str(head_path)
    calibration_receipt["head_support_file_sha256"] = file_sha(head_path)
    calibration_receipt["head_support_sha256"] = head["head_support_sha256"]
    calibration_receipt["root_group_ranker_path"] = str(root_ranker_path)
    calibration_receipt["root_group_ranker_file_sha256"] = (
        root_ranker_file_sha256
    )
    calibration_receipt["root_group_ranker_sha256"] = root_ranker[
        "root_group_ranker_sha256"
    ]
    calibration_receipt["source_rank_numeric_contract"] = (
        protocol.SOURCE_RANK_NUMERIC_CONTRACT
    )
    calibration_receipt["source_rank_member_authority"] = copy.deepcopy(
        source_rank_member_authority
    )
    calibration_receipt["source_rank_member_authority_sha256"] = (
        source_rank_member_authority_sha256
    )
    calibration_receipt["deployment_uncertainty_contract_sha256"] = calibration[
        "deployment_uncertainty_contract_sha256"
    ]
    calibration_receipt["ensemble_manifest_path"] = str(ensemble_path)
    calibration_receipt["ensemble_manifest_file_sha256"] = file_sha(ensemble_path)
    calibration_receipt["ensemble_manifest_sha256"] = ensemble[
        "ensemble_manifest_sha256"
    ]
    calibration_receipt = signed(calibration_receipt, "receipt_sha256")
    write_json(calibration_receipt_path, calibration_receipt)

    # Rebuild bridge from the now-final produced dependencies.
    bridge_kwargs: dict[str, Any] = {}
    replacement_paths = {
        "ensemble_manifest": ensemble_path,
        "calibration": calibration_path,
        "head_support": head_path,
        "calibration_receipt": calibration_receipt_path,
    }
    for role, descriptor in bridge_value["dependencies"].items():
        path = replacement_paths.get(role, Path(descriptor["path"]))
        bridge_kwargs[f"{role}_path"] = path
        bridge_kwargs[f"{role}_file_sha256"] = file_sha(path)
    bridge_value = bridge.freeze_bridge(**bridge_kwargs)
    write_json(bridge_path, bridge_value)

    manifest_stub = safe / "trainer_manifest.json"
    expected_stub = safe / "expected_split.json"
    event_stub = safe / "event_spec.json"
    for path in (manifest_stub, expected_stub, event_stub):
        write_json(path, {})
    evaluator_authority = signed({
        "format": post.evaluator.INPUT_FORMAT, "status": post.evaluator.INPUT_STATUS,
        "trainer_compatible_manifest": record(manifest_stub, shared["training_manifest_sha256"]),
        "expected_manifest_split_receipt": record(expected_stub, "a" * 64),
        "canonical_event_spec": {"path": str(event_stub), "file_sha256": file_sha(event_stub)},
        "members": evaluator_authority_members, "member_count": 5,
        "source_rank_numeric_contract": protocol.SOURCE_RANK_NUMERIC_CONTRACT,
        "target_validation_group_count": 190,
        "adapter_training_complete_before_authority": True,
        "target_validation_open_authorized": True,
        "evaluation400_membership_present": False,
        "evaluation400_open_authorized": False,
        "fresh_or_confirmation_open_authorized": False,
    }, "authority_sha256")
    evaluator_authority_path = root / "formal190" / "evaluator_input_authority.json"
    write_json(evaluator_authority_path, evaluator_authority)
    formal_receipt = signed({
        "format": post.evaluator.RECEIPT_FORMAT, "status": post.evaluator.RECEIPT_STATUS,
        "input_authority_path": str(evaluator_authority_path),
        "input_authority_file_sha256": file_sha(evaluator_authority_path),
        "input_authority_sha256": evaluator_authority["authority_sha256"],
        "target_validation_groups": 190, "target_validation_samples": 760,
        "target_validation_hdf5_files_opened": 190,
        "target_validation_opened_after_five_adapters_frozen": True,
        "calibration_input_authority_path": str(calibration_authority_path),
        "calibration_input_authority_file_sha256": file_sha(calibration_authority_path),
        "calibration_input_authority_sha256": calibration_authority[
            "input_authority_sha256"
        ],
        "source_rank_score_contract_sha256s": [
            member["source_rank_score_contract_sha256"] for member in members
        ],
        "source_rank_numeric_contract": protocol.SOURCE_RANK_NUMERIC_CONTRACT,
        "evaluation400_membership_present": False,
        "evaluation400_hdf5_or_label_files_opened": 0,
        "fresh_or_confirmation_files_opened": 0,
        "performance_or_transfer_claim_authorized": False,
    }, "receipt_sha256")
    formal_receipt_path = root / "formal190" / "evaluator_stage" / "result" / "final_receipt.json"
    write_json(formal_receipt_path, formal_receipt)

    source_final = signed({
        "format": protocol.SOURCE_FORMAT, "status": protocol.SOURCE_STATUS,
        "target_data_read": False, "target_labels_read": False,
        "fresh_inputs_accepted": False, "fresh_labels_read": False,
        "test_labels_used": False, "test_hdf_label_datasets_opened": 0,
        "artifacts_frozen_read_only": True,
    }, "receipt_sha256")
    source_final_path = safe / "r7h_source_final.json"
    write_json(source_final_path, source_final)

    implementation_roles = (
        "launcher", "materializer", "trainer", "evaluator", "calibrator",
        "identity_bridge_v2", "r9b_watcher",
    )
    implementation_records: dict[str, dict[str, str]] = {}
    code_root = safe / "reviewed_code"
    code_root.mkdir()
    for role in implementation_roles:
        implementation_path = code_root / f"post_v3_{role}.py"
        implementation_path.write_text(
            f"# synthetic frozen post-v3 {role}\n", encoding="utf-8"
        )
        implementation_path.chmod(0o444)
        implementation_records[role] = {
            "path": str(implementation_path),
            "file_sha256": file_sha(implementation_path),
        }
    launcher_path = Path(implementation_records["launcher"]["path"])
    launcher_sha = implementation_records["launcher"]["file_sha256"]
    monkeypatch.setattr(
        protocol, "APPROVED_POST_V3_LAUNCHER_SHA256", launcher_sha
    )
    claim_root = safe / "formal190_claims"
    claim_root.mkdir()
    plan_base = {
        "format": post.PLAN_FORMAT,
        "status": "preregistered_waiting_for_exact_r7h_r8e_r9b_and_development300",
        "output_root": str(root), "source_root": str(safe), "r9b_root": str(safe),
        "r9b_final_file_sha256": "1" * 64, "r9b_final_sha256": "2" * 64,
        "r9b_static_plan_sha256": "3" * 64,
        "development300_collection_root": str(safe),
        "development300_terminal": record(source_final_path, source_final["receipt_sha256"]),
        "development300_runner_authority": record(source_final_path, source_final["receipt_sha256"]),
        "development300_target_preregistration": record(source_final_path, source_final["receipt_sha256"]),
        "development300_identity_authority": record(source_final_path, source_final["receipt_sha256"]),
        "python": {"path": sys.executable, "file_sha256": "4" * 64},
        "implementations": implementation_records,
        "python_import_closure": {
            Path(record_value["path"]).stem: dict(record_value)
            for record_value in implementation_records.values()
        },
        "canonical_event_spec": {"path": str(event_stub), "file_sha256": file_sha(event_stub)},
        "canonical_teacher": None, "split_profile": post.SPLIT_PROFILE,
        "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
        "adapter_member_count": 5, "adapter_member_seeds": list(post.SOURCE_MEMBER_SEEDS),
        "adapter_source_policy": "one_to_one_native_r7h_individual_members_only",
        "lobo_or_aggregate_checkpoint_authorized": False,
        "adapter_steps": 10, "adapter_eval_every": 5, "gpu_index": 0,
        "gpu_lock_path": str(safe / "gpu.lock"),
        "formal190_claim_root": str(claim_root),
        "formal190_open_authorized_before_five_adapters_frozen": False,
        "evaluation400_membership_or_label_open_authorized": False,
        "old_paired400_authority_path_present": False,
        "second_reserve400_authorized": False,
        "hdf5_files_opened_during_preregistration": 0,
        "labels_or_outcomes_read_during_preregistration": False,
        "create_once_nonresumable": True,
    }
    plan = signed(plan_base, "plan_sha256")
    plan_path = root / "_watcher" / "static_plan.json"
    write_json(plan_path, plan)

    development_terminal = plan["development300_terminal"]
    claim_identity = post.canonical_sha256({
        "development300_terminal_file_sha256": development_terminal["file_sha256"],
        "development300_terminal_sha256": development_terminal["logical_sha256"],
        "split_profile": post.SPLIT_PROFILE, "formal_group_count": 190,
    })
    claim = signed({
        "format": post.FORMAL190_CLAIM_FORMAT,
        "status": post.FORMAL190_CLAIM_STATUS,
        "formal190_identity_sha256": claim_identity,
        "development300_terminal_file_sha256": development_terminal["file_sha256"],
        "development300_terminal_sha256": development_terminal["logical_sha256"],
        "split_profile": post.SPLIT_PROFILE, "formal_group_count": 190,
        "post_v3_plan_sha256": plan["plan_sha256"],
        "post_v3_output_root": str(root),
        "formal190_authority_may_be_created_once": True,
        "reopen_from_second_output_authorized": False,
        "claimed_unix_ns": 123456789,
    }, "claim_sha256")
    claim_path = claim_root / f"formal190-{claim_identity}.claim.json"
    write_json(claim_path, claim)
    claim_descriptor = {
        "path": str(claim_path), "file_sha256": file_sha(claim_path),
        "logical_sha256": claim["claim_sha256"],
        "formal190_identity_sha256": claim_identity, "consumed": True,
    }

    handoff_base = {
        "format": post.HANDOFF_FORMAT,
        "status": "ready_for_external_preoutcome_identity_bridge_v2_freeze",
        "post_v3_plan_sha256": plan["plan_sha256"],
        "lineage": {
            "r7h_source_final": record(source_final_path, source_final["receipt_sha256"]),
            "r7h_member_checkpoint_sha256": source_shas,
            "r7h_member_seed": list(post.SOURCE_MEMBER_SEEDS),
            "r8e_root": str(safe), "r8e_final": record(source_final_path, source_final["receipt_sha256"]),
            "r8e_summary_sha256": "7" * 64,
            "r9b_final": record(source_final_path, source_final["receipt_sha256"]),
            "development300_terminal": record(source_final_path, source_final["receipt_sha256"]),
            "materializer_v3_receipt": record(source_final_path, source_final["receipt_sha256"]),
            "formal190_evaluator_authority": record(
                evaluator_authority_path, evaluator_authority["authority_sha256"]
            ),
            "formal190_evaluator_receipt": record(
                formal_receipt_path, formal_receipt["receipt_sha256"]
            ),
            "formal190_global_one_shot_claim": claim_descriptor,
        },
        "identity_bridge_v2": {
            "implementation": dict(implementation_records["identity_bridge_v2"]),
            "produced_dependencies": {
                role: dict(bridge_value["dependencies"][role])
                for role in ("ensemble_manifest", "calibration", "head_support", "calibration_receipt")
            },
            "external_dependencies_required": list(post.BRIDGE_EXTERNAL_DEPENDENCIES),
            "cli_argument_mapping": {
                "ensemble-manifest": "produced_dependencies.ensemble_manifest",
                "calibration": "produced_dependencies.calibration",
                "head-support": "produced_dependencies.head_support",
                "calibration-receipt": "produced_dependencies.calibration_receipt",
            },
            "bridge_execution_authorized_by_handoff": False,
            "external_identity_attestor_required": True,
        },
        "adapter_member_count": 5,
        "adapter_member_receipt_sha256": [member["receipt_sha256"] for member in members],
        "split_profile": post.SPLIT_PROFILE,
        "required_trainer_group_counts": {"train": 80, "validation": 30, "test": 190},
        "formal190_opened_by_independent_evaluator_after_five_frozen_adapters": 190,
        "formal190_opened_by_watcher_process": 0,
        "formal190_labels_opened_before_five_adapters_frozen": 0,
        "evaluation400_membership_present": False,
        "evaluation400_hdf5_trajectory_or_labels_opened": 0,
        "evaluation400_conditions_executed": 0,
        "old_paired400_authority_waited_or_generated": False,
        "second_reserve400_created": False,
        "lobo_or_aggregate_checkpoint_used_for_adapter_training": False,
        "performance_or_transfer_claim_authorized": False,
    }
    handoff = signed(handoff_base, "handoff_sha256")
    handoff_path = root / "handoff" / "evaluation400_identity_bridge_v2_handoff.json"
    write_json(handoff_path, handoff)

    detached = signed({"status": "synthetic_detached"}, "detach_proof_sha256")
    write_json(root / "_watcher" / "detached_worker_proof.json", detached)
    for name in ("gpu_idle_before_training", "gpu_idle_after_formal190"):
        write_json(root / "_watcher" / f"{name}.json", signed(
            {"status": "synthetic_idle"}, "audit_sha256"
        ))
    release = signed({"status": "synthetic_released"}, "release_sha256")
    write_json(root / "_watcher" / "gpu_lock_release.json", release)
    materializer_receipt = signed(
        {"format": "synthetic_materializer_receipt"}, "receipt_sha256"
    )
    materializer_receipt_path = (
        root / "materialization" / "materializer_stage" / "result"
        / post.MATERIALIZER_OUTPUTS["receipt"]
    )
    write_json(materializer_receipt_path, materializer_receipt)

    physical_gpu = {
        "gpu_index": 0, "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_uuid": "GPU-synthetic-4090", "checks": {}, "audit_sha256": "f" * 64,
    }
    stage_results: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(protocol.EXPECTED_STAGES):
        stage_root = post._stage_root(root, name)
        lifecycle = signed(
            {
                "popen_attempted": True, "popen_reached": True,
                "process_pid": 1000 + index, "process_pgid": 1000 + index,
                "process_group_isolated": True, "returncode": 0,
                "direct_process_reaped": True, "process_group_reaped": True,
                "binding_status": "bound_reaped",
            },
            "lifecycle_sha256",
        )
        launch = signed({"stage": name}, "launch_sha256")
        launch_path = write_json(stage_root / "launch.json", launch)
        write_json(stage_root / "lifecycle.json", lifecycle)
        log_path = stage_root / "run.log"
        log_path.write_bytes(f"synthetic-{name}\n".encode("ascii"))
        log_path.chmod(0o444)
        exit_path = stage_root / "run.exit"
        exit_path.write_bytes(b"0\n")
        exit_path.chmod(0o444)
        gpu_stage = name.startswith("train_adapter_member_") or name.startswith(
            "evaluate_frozen_five_member"
        )
        result_base = {
            "stage": name, "returncode": 0,
            "command_sha256": f"{2000 + index:064x}",
            "launch_file_sha256": file_sha(launch_path), "lifecycle": lifecycle,
            "physical_gpu": dict(physical_gpu) if gpu_stage else None,
            "log_file_sha256": file_sha(log_path),
            "run_exit_file_sha256": file_sha(exit_path),
        }
        stage_results[name] = {
            **result_base, "result_sha256": post.canonical_sha256(result_base)
        }
    artifact_closure = post.build_artifact_closure(
        root=root, plan=plan, stage_results=stage_results,
        handoff_path=handoff_path, formal190_claim=claim_descriptor,
    )
    terminal_base = {
        "format": post.FORMAT, "status": post.TERMINAL_STATUS,
        "plan_sha256": plan["plan_sha256"],
        "detach_proof_sha256": detached["detach_proof_sha256"],
        "execution_order": list(protocol.EXPECTED_STAGES), "stage_results": stage_results,
        "adapter_member_count": 5, "adapter_member_seeds": list(post.SOURCE_MEMBER_SEEDS),
        "adapter_source_policy": "one_to_one_native_r7h_individual_members_only",
        "r7h_member_checkpoint_sha256": source_shas,
        "r8e_r9b_lineage_sha256": "9" * 64,
        "development300_materializer_receipt_sha256": materializer_receipt["receipt_sha256"],
        "formal190_opened_after_five_frozen_adapters": 190,
        "formal190_labels_opened_before_five_adapters_frozen": 0,
        "calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
        "identity_bridge_v2_handoff": record(handoff_path, handoff["handoff_sha256"]),
        "formal190_global_one_shot_claim": claim_descriptor,
        "evaluation400_hdf5_trajectory_or_labels_opened": 0,
        "evaluation400_conditions_executed": 0,
        "old_paired400_authority_waited_or_generated": False,
        "second_reserve400_created": False,
        "gpu_lock_release_sha256": release["release_sha256"],
        "artifacts_frozen_read_only": True,
        "terminal_publication": "mode000_then_tree_freeze_verify_then_run_exit0444_then_final_receipt0444_last",
        "artifact_closure": artifact_closure,
        "artifact_closure_sha256": post.canonical_sha256(artifact_closure),
    }
    terminal = signed(terminal_base, "receipt_sha256")
    terminal_path = root / "final_receipt.json"
    write_json(terminal_path, terminal)

    # Freeze all bridge dependency JSON after the last reconstruction.
    for descriptor in bridge_value["dependencies"].values():
        Path(descriptor["path"]).chmod(0o444)
    bridge_path.chmod(0o444)

    private = Ed25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    execution_root = safe / "execution_stack"
    execution_root.mkdir()
    component_paths = {
        "executor_implementation": execution_root / "paired_executor_v3.py",
        "result_evaluator_implementation": execution_root / "paired_result_evaluator_v3.py",
        "simulator_implementation": execution_root / "simulator_adapter.py",
        "runtime_contract": execution_root / "runtime_contract.json",
        "collector_implementation": execution_root / "collector_adapter.py",
        "container_inventory": execution_root / "container_inventory.json",
    }
    for role, component_path in component_paths.items():
        component_path.write_text(f"synthetic immutable {role}\n", encoding="utf-8")
        component_path.chmod(0o444)
    issuer_base = {
        "format": protocol.ISSUER_ATTESTATION_FORMAT,
        "status": protocol.ISSUER_ATTESTATION_STATUS,
        "protocol_format": protocol.CORE_FORMAT,
        "issuer_key_id": "synthetic-evaluator-key-1",
        "issuer_public_key_hex": public_hex,
        "issuer_public_key_sha256": hashlib.sha256(bytes.fromhex(public_hex)).hexdigest(),
        "issuer_identity_sha256": "c" * 64,
        "allowlist_entry_active": True,
        "authorization_sequence": 1,
    }
    issuer = signed(issuer_base, "attestation_sha256")
    issuer_path = execution_root / "trusted_issuer_attestation.json"
    write_json(issuer_path, issuer)
    opaque = {
        role: {"path": str(path), "file_sha256": file_sha(path)}
        for role, path in component_paths.items()
    }
    inventory_base = {
        "format": protocol.EXECUTION_INVENTORY_FORMAT,
        "status": protocol.EXECUTION_INVENTORY_STATUS,
        "protocol_format": protocol.CORE_FORMAT,
        "execution_lane": {
            "pair_count": 400, "only_evaluation400_lane": True,
            "additional_reserve400_count": 0,
        },
        "trusted_issuer_attestation": record(
            issuer_path, issuer["attestation_sha256"]
        ),
        "executor": {
            "identity_sha256": "d" * 64,
            "implementation": opaque["executor_implementation"],
        },
        "result_evaluator": {
            "identity_sha256": "e" * 64,
            "implementation": opaque["result_evaluator_implementation"],
        },
        "execution_stack": {
            role: opaque[role] for role in (
                "simulator_implementation", "runtime_contract",
                "collector_implementation", "container_inventory",
            )
        },
        "component_inventory_complete": True,
        "real_executor_present": True,
        "real_result_evaluator_present": True,
        "outcome_or_trajectory_files_opened_during_attestation": 0,
    }
    inventory = signed(inventory_base, "attestation_sha256")
    inventory_path = execution_root / "execution_inventory_attestation.json"
    write_json(inventory_path, inventory)
    monkeypatch.setattr(
        protocol, "APPROVED_EXECUTION_INVENTORY_FILE_SHA256",
        file_sha(inventory_path),
    )
    runtime_execution_authority_path = (
        execution_root / "schema6_runtime_execution_authority.json"
    )
    write_json(
        runtime_execution_authority_path,
        {
            "format": "synthetic_schema6_runtime_execution_authority_v2",
            "runtime_contract": nested_runtime_contract,
        },
    )
    selector_implementation_path = execution_root / "root_selector_v3.py"
    selector_implementation_path.write_text(
        "# synthetic reviewed root selector v3\n", encoding="utf-8"
    )
    selector_implementation_path.chmod(0o444)
    kwargs = {
        "post_plan_path": plan_path, "post_plan_file_sha256": file_sha(plan_path),
        "post_terminal_path": terminal_path,
        "post_terminal_file_sha256": file_sha(terminal_path),
        "post_handoff_path": handoff_path, "post_handoff_file_sha256": file_sha(handoff_path),
        "member_receipts": member_specs,
        "identity_bridge_path": bridge_path,
        "identity_bridge_file_sha256": file_sha(bridge_path),
        "execution_inventory_attestation_path": inventory_path,
        "execution_inventory_attestation_file_sha256": file_sha(inventory_path),
        "execution_inventory_attestation_sha256": inventory["attestation_sha256"],
        "expected_post_launcher_sha256": launcher_sha,
        "runtime_execution_authority_path": runtime_execution_authority_path,
        "runtime_execution_authority_file_sha256": file_sha(
            runtime_execution_authority_path
        ),
        "selector_implementation_path": selector_implementation_path,
        "selector_implementation_file_sha256": file_sha(
            selector_implementation_path
        ),
    }
    return {
        "safe": safe, "root": root, "kwargs": kwargs, "private": private,
        "head_path": head_path, "bridge_path": bridge_path,
        "terminal": terminal, "members": members,
    }


def freeze_signed_bundle(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    core = protocol.freeze_core(**data["kwargs"])
    core_path = data["safe"] / "paired_core_v3.json"
    protocol.write_json_new(core_path, core)
    core_file_sha = file_sha(core_path)
    statement = protocol.expected_decision_statement(
        core, core_file_sha256=core_file_sha, decision_nonce_hex="ab" * 32
    )
    signature = data["private"].sign(protocol.decision_signing_bytes(statement)).hex()
    decision_base = {
        "format": protocol.DECISION_FORMAT, "status": protocol.DECISION_STATUS,
        "signature_algorithm": "Ed25519", "statement": statement,
        "decision_signature_ed25519_hex": signature,
    }
    decision = {**decision_base, "decision_sha256": protocol.canonical_sha256(decision_base)}
    decision_path = data["safe"] / "paired_decision_v3.json"
    protocol.write_json_new(decision_path, decision)
    bundle = protocol.freeze_bundle(
        core_path=core_path, core_file_sha256=core_file_sha,
        decision_path=decision_path, decision_file_sha256=file_sha(decision_path),
    )
    return core, decision, bundle


def test_reviewed_post_launcher_allowlist_matches_frozen_source() -> None:
    launcher = ROOT / "scripts" / "launch_smolvla_piper_schema6_post_collection_v3.py"
    assert file_sha(launcher) == protocol.APPROVED_POST_V3_LAUNCHER_SHA256


def test_core_decision_bundle_happy_path_is_execution_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    core, decision, bundle = freeze_signed_bundle(data)
    assert protocol.validate_core(core) == core["protocol_core_sha256"]
    assert protocol.validate_bundle(bundle) == bundle["bundle_sha256"]
    assert core["evaluation400"]["pair_count"] == 400
    assert core["evaluation400"]["additional_reserve400_count"] == 0
    assert core["development_and_formal190"]["group_counts"] == {
        "train": 80, "internal_validation": 30, "formal_validation": 190,
    }
    assert core["development_and_formal190"]["all_six_heads_primary"] == list(
        protocol.HEAD_NAMES
    )
    assert core["execution_authorized"] is False
    assert decision["signature_algorithm"] == "Ed25519"
    assert bundle["execution_authorized"] is True
    assert bundle["protocol_freezer_may_execute"] is False
    assert core["preexecution_capability_receipt"]["label_or_outcome_files_opened"] == 0
    deployment = core["deployment"]
    selector = deployment["selector_authority"]
    assert selector["source_rank_score_contracts"] == deployment[
        "source_rank_score_contracts"
    ]
    assert selector["source_rank_numeric_contract"] == (
        protocol.SOURCE_RANK_NUMERIC_CONTRACT
    )
    assert len(selector["source_rank_score_contracts"]) == 5
    assert selector["deployment_parameters"] == deployment["deployment_parameters"]
    assert selector["formal190_thresholds"] == deployment["formal190_thresholds"]
    runtime = deployment["runtime_execution_authority"]
    assert set(runtime) == {
        "path", "file_sha256", "nested_runtime_contract_sha256",
        "max_episode_steps",
    }
    assert runtime["nested_runtime_contract_sha256"] == deployment[
        "target_reset_runtime_contract_sha256"
    ]
    assert type(runtime["max_episode_steps"]) is int
    assert runtime["max_episode_steps"] == 200


@pytest.mark.parametrize("max_episode_steps", (199, True))
def test_runtime_authority_requires_target_bound_exact_200_step_nested_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_episode_steps: object,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    path = Path(data["kwargs"]["runtime_execution_authority_path"])
    nested = signed(
        {"max_episode_steps": max_episode_steps}, "runtime_contract_sha256"
    )
    write_json(
        path,
        {
            "format": "synthetic_schema6_runtime_execution_authority_v2",
            "runtime_contract": nested,
        },
    )
    data["kwargs"]["runtime_execution_authority_file_sha256"] = file_sha(path)
    with pytest.raises(protocol.PairedProtocolV3Error, match="runtime"):
        protocol.freeze_core(**data["kwargs"])


def test_bridge_target_reset_runtime_must_equal_reopened_target_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    path = Path(data["bridge_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("bridge_sha256")
    value["target_reset_runtime_contract_sha256"] = "f" * 64
    path.chmod(0o644)
    write_json(path, signed(value, "bridge_sha256"))
    path.chmod(0o444)
    with pytest.raises(protocol.PairedProtocolV3Error, match="bridge dependency"):
        protocol._load_bridge(path, file_sha(path))


def test_core_source_rank_numeric_contract_drift_fails_after_full_resign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    core = protocol.freeze_core(**data["kwargs"])
    changed = copy.deepcopy(core)
    changed.pop("protocol_core_sha256")
    selector = changed["deployment"]["selector_authority"]
    selector.pop("selector_authority_sha256")
    selector["source_rank_numeric_contract"] = "ieee754_float64_reassociated"
    selector["selector_authority_sha256"] = protocol.canonical_sha256(selector)
    changed["deployment"]["selector_authority_sha256"] = selector[
        "selector_authority_sha256"
    ]
    changed["protocol_core_sha256"] = protocol.canonical_sha256(changed)
    with pytest.raises(protocol.PairedProtocolV3Error, match="core v3 boundary"):
        protocol.validate_core(changed)


@pytest.mark.parametrize("tamper", ("missing_r9b_role", "missing_r9b_closure"))
def test_post_plan_requires_exact_r9b_and_complete_import_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    plan_path = Path(data["kwargs"]["post_plan_path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan.pop("plan_sha256")
    r9b = plan["implementations"]["r9b_watcher"]
    if tamper == "missing_r9b_role":
        del plan["implementations"]["r9b_watcher"]
    else:
        del plan["python_import_closure"][Path(r9b["path"]).stem]
    write_json(plan_path, signed(plan, "plan_sha256"))
    data["kwargs"]["post_plan_file_sha256"] = file_sha(plan_path)
    with pytest.raises(protocol.PairedProtocolV3Error, match="post v3"):
        protocol.freeze_core(**data["kwargs"])


def test_unapproved_execution_inventory_blocks_core_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        protocol, "APPROVED_EXECUTION_INVENTORY_FILE_SHA256", None,
    )
    with pytest.raises(protocol.PairedProtocolV3Error, match="not been independently approved"):
        protocol.freeze_core(**data["kwargs"])


def test_execution_inventory_component_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    inventory = json.loads(
        data["kwargs"]["execution_inventory_attestation_path"].read_text()
    )
    executor_path = Path(inventory["executor"]["implementation"]["path"])
    executor_path.chmod(0o644)
    executor_path.write_text("tampered executor\n", encoding="utf-8")
    executor_path.chmod(0o444)
    with pytest.raises(protocol.PairedProtocolV3Error, match="file SHA mismatch"):
        protocol.freeze_core(**data["kwargs"])


def test_canonical_sha_cannot_impersonate_ed25519_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    core = protocol.freeze_core(**data["kwargs"])
    core_path = data["safe"] / "core_v3.json"
    protocol.write_json_new(core_path, core)
    statement = protocol.expected_decision_statement(
        core, core_file_sha256=file_sha(core_path), decision_nonce_hex="ac" * 32
    )
    base = {
        "format": protocol.DECISION_FORMAT, "status": protocol.DECISION_STATUS,
        "signature_algorithm": "Ed25519", "statement": statement,
        "decision_signature_ed25519_hex": protocol.canonical_sha256(statement) * 2,
    }
    decision = {**base, "decision_sha256": protocol.canonical_sha256(base)}
    with pytest.raises(protocol.PairedProtocolV3Error, match="signature verification"):
        protocol.verify_decision(decision, core=core, core_file_sha256=file_sha(core_path))


def test_decision_for_another_core_or_key_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    core = protocol.freeze_core(**data["kwargs"])
    statement = protocol.expected_decision_statement(
        core, core_file_sha256="1" * 64, decision_nonce_hex="ad" * 32
    )
    wrong_key = Ed25519PrivateKey.generate()
    base = {
        "format": protocol.DECISION_FORMAT, "status": protocol.DECISION_STATUS,
        "signature_algorithm": "Ed25519", "statement": statement,
        "decision_signature_ed25519_hex": wrong_key.sign(
            protocol.decision_signing_bytes(statement)
        ).hex(),
    }
    decision = {**base, "decision_sha256": protocol.canonical_sha256(base)}
    with pytest.raises(protocol.PairedProtocolV3Error, match="signature verification"):
        protocol.verify_decision(decision, core=core, core_file_sha256="1" * 64)
    with pytest.raises(protocol.PairedProtocolV3Error, match="exact core"):
        protocol.verify_decision(decision, core=core, core_file_sha256="2" * 64)


def test_post_bool_as_int_fails_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    terminal_path = data["kwargs"]["post_terminal_path"]
    terminal = copy.deepcopy(data["terminal"])
    terminal.pop("receipt_sha256")
    terminal["adapter_member_count"] = True
    write_json(terminal_path, signed(terminal, "receipt_sha256"))
    data["kwargs"]["post_terminal_file_sha256"] = file_sha(terminal_path)
    with pytest.raises(protocol.PairedProtocolV3Error, match="terminal"):
        protocol.freeze_core(**data["kwargs"])


def test_duplicate_json_key_is_rejected_from_single_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    terminal_path = data["kwargs"]["post_terminal_path"]
    terminal_path.chmod(0o644)
    terminal_path.write_text('{"format":"a","format":"b"}', encoding="utf-8")
    terminal_path.chmod(0o444)
    data["kwargs"]["post_terminal_file_sha256"] = file_sha(terminal_path)
    with pytest.raises(protocol.PairedProtocolV3Error, match="duplicate JSON key"):
        protocol.freeze_core(**data["kwargs"])


def test_ancestor_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    link = data["safe"] / "plan_alias"
    link.symlink_to(data["root"] / "_watcher", target_is_directory=True)
    data["kwargs"]["post_plan_path"] = link / "static_plan.json"
    with pytest.raises(protocol.PairedProtocolV3Error, match="symlinks"):
        protocol.freeze_core(**data["kwargs"])


def test_six_head_bool_numeric_and_recovery_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    head = json.loads(data["head_path"].read_text(encoding="utf-8"))
    calibration_path = Path(
        json.loads(data["bridge_path"].read_text())["dependencies"]["calibration"]["path"]
    )
    ensemble_path = Path(
        json.loads(data["bridge_path"].read_text())["dependencies"]["ensemble_manifest"]["path"]
    )
    calibration = json.loads(calibration_path.read_text())
    ensemble = json.loads(ensemble_path.read_text())
    head["heads"]["recovery"]["independent_positive_or_observed_groups"] = True
    head.pop("head_support_sha256")
    head = signed(head, "head_support_sha256")
    with pytest.raises(protocol.PairedProtocolV3Error):
        protocol._validate_six_heads(calibration, head, ensemble)

    head = json.loads(data["head_path"].read_text())
    head["heads"]["recovery"]["all_member_recovery_heads_trained"] = False
    head["heads"]["recovery"]["enabled_for_primary"] = False
    head.pop("head_support_sha256")
    head = signed(head, "head_support_sha256")
    calibration["head_enabled_for_primary"]["recovery"] = False
    calibration["recovery_temperature_fitted_on_validation_only"] = False
    calibration.pop("calibration_sha256")
    calibration = signed(calibration, "calibration_sha256")
    ensemble["head_enabled_for_primary"]["recovery"] = False
    ensemble["head_support_sha256"] = head["head_support_sha256"]
    ensemble["calibration_sha256"] = calibration["calibration_sha256"]
    ensemble.pop("ensemble_manifest_sha256")
    ensemble = signed(ensemble, "ensemble_manifest_sha256")
    with pytest.raises(
        protocol.PairedProtocolV3Error,
        match="formal190 calibration deployment|six formal190 heads",
    ):
        protocol._validate_six_heads(calibration, head, ensemble)


def test_outputs_are_create_once_owner_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = fixture(tmp_path, monkeypatch)
    _core, _decision, bundle = freeze_signed_bundle(data)
    output = data["safe"] / "bundle_copy_v3.json"
    protocol.write_json_new(output, bundle)
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        protocol.write_json_new(output, bundle)
