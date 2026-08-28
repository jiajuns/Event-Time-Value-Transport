from __future__ import annotations

import copy
import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for directory in (SCRIPTS, TESTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import freeze_smolvla_piper_evaluation400_execution_authority_v2 as authority  # noqa: E402
import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge  # noqa: E402
import smolvla_piper_paired_success_protocol_v2 as protocol  # noqa: E402
import test_freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge_fixture  # noqa: E402


def signed(value: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(value)
    result[field] = protocol.canonical_sha256(result)
    return result


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def safe_root(tmp_path: Path) -> Path:
    # The production protocol rejects any direct path component beginning with
    # test_; pytest's function-specific tmp_path intentionally has one.
    root = tmp_path.parent / f"pairedv2_{abs(hash(tmp_path.name))}"
    root.mkdir()
    return root


def member_receipt(
    *, index: int, seed: int, checkpoint: Path, checkpoint_sha: str,
    shared: dict[str, str], source_checkpoint_sha: str,
) -> dict[str, object]:
    return signed(
        {
            "format": protocol.MEMBER_RECEIPT_FORMAT,
            "status": protocol.MEMBER_RECEIPT_STATUS,
            "member_index": index,
            "member_seed": seed,
            "source_checkpoint_sha256": source_checkpoint_sha,
            "training_manifest_sha256": shared["training_manifest_sha256"],
            "split_sha256": shared["split_sha256"],
            "source_ensemble_contract_sha256": shared[
                "source_ensemble_contract_sha256"
            ],
            "summary_path": f"/sealed_adapter/member_{index}/summary.json",
            "summary_file_sha256": f"{100 + index:064x}",
            "summary_sha256": f"{110 + index:064x}",
            "checkpoint_path": str(checkpoint),
            "checkpoint_file_sha256": checkpoint_sha,
            "validation_predictions_path": f"/sealed_adapter/member_{index}/predictions.npz",
            "validation_predictions_file_sha256": f"{120 + index:064x}",
            "validation_predictions_logical_sha256": f"{130 + index:064x}",
            "validation_labels_path": f"/sealed_adapter/member_{index}/internal_validation.npz",
            "validation_labels_file_sha256": f"{140 + index:064x}",
            "validation_labels_logical_sha256": f"{150 + index:064x}",
            "validation_identity_set_sha256": "d" * 64,
            "validation_lane": "adaptation_derived_internal_validation_only",
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
            "target_validation50_hdf5_files_opened": 0,
            "sealed_test_labels_opened": 0,
        },
        "receipt_sha256",
    )


def strict_fixture(tmp_path: Path, *, recovery_enabled: bool = True) -> dict[str, object]:
    root = safe_root(tmp_path)
    data = bridge_fixture.fixture(root)
    r7h_sha = "f" * 64
    shared = {
        "training_manifest_sha256": "1" * 64,
        "split_sha256": "2" * 64,
        "source_ensemble_contract_sha256": r7h_sha,
        "prediction_contract_sha256": "3" * 64,
    }

    head = copy.deepcopy(data["values"]["head"])
    head.pop("head_support_sha256")
    for name in protocol.HEAD_NAMES:
        head["heads"][name]["enabled_for_primary"] = True
    head["heads"]["recovery"]["support_threshold_met"] = True
    head["heads"]["recovery"]["all_member_recovery_heads_trained"] = (
        recovery_enabled
    )
    if not recovery_enabled:
        head["heads"]["recovery"]["enabled_for_primary"] = False
    bridge_fixture.resign_file(head, "head_support", "head_support_sha256", data)
    head = json.loads(data["paths"]["head_support"].read_text(encoding="utf-8"))

    calibration = copy.deepcopy(data["values"]["calibration"])
    calibration.pop("calibration_sha256")
    calibration["head_enabled_for_primary"] = {
        name: recovery_enabled if name == "recovery" else True
        for name in protocol.HEAD_NAMES
    }
    calibration["recovery_temperature_fitted_on_validation_only"] = recovery_enabled
    bridge_fixture.resign_file(
        calibration, "calibration", "calibration_sha256", data
    )
    calibration = json.loads(
        data["paths"]["calibration"].read_text(encoding="utf-8")
    )

    checkpoints: list[Path] = []
    checkpoint_shas: list[str] = []
    ensemble = copy.deepcopy(data["values"]["ensemble"])
    ensemble.pop("ensemble_manifest_sha256")
    for index, member in enumerate(ensemble["members"]):
        checkpoint = root / f"adapter_member_{index}.pt"
        checkpoint.write_bytes(f"synthetic target adapter {index}".encode("ascii"))
        checkpoint_sha = protocol.file_sha256(checkpoint)
        checkpoints.append(checkpoint)
        checkpoint_shas.append(checkpoint_sha)
        member.update(
            {
                "member_seed": 700 + index,
                "checkpoint_path": str(checkpoint),
                "checkpoint_file_sha256": checkpoint_sha,
            }
        )
    ensemble["shared_contract"] = shared
    ensemble["head_enabled_for_primary"] = calibration["head_enabled_for_primary"]
    ensemble["calibration_sha256"] = calibration["calibration_sha256"]
    ensemble["head_support_sha256"] = head["head_support_sha256"]
    ensemble["conditional_recovery_temperature"] = calibration["metrics"][
        "conditional_recovery"
    ]["deployment_temperature"]
    bridge_fixture.resign_file(
        ensemble, "ensemble_manifest", "ensemble_manifest_sha256", data
    )
    ensemble = json.loads(
        data["paths"]["ensemble_manifest"].read_text(encoding="utf-8")
    )

    receipt = copy.deepcopy(data["values"]["receipt"])
    receipt.pop("receipt_sha256")
    receipt["shared_contract"] = shared
    for role, logical_field in (
        ("calibration", "calibration_sha256"),
        ("head_support", "head_support_sha256"),
        ("ensemble_manifest", "ensemble_manifest_sha256"),
    ):
        receipt[f"{role}_file_sha256"] = bridge.file_sha256(data["paths"][role])
        receipt[logical_field] = {
            "calibration": calibration,
            "head_support": head,
            "ensemble_manifest": ensemble,
        }[role][logical_field]
    bridge_fixture.resign_file(
        receipt, "calibration_receipt", "receipt_sha256", data
    )

    bridge_value = bridge.freeze_bridge(**data["kwargs"])
    bridge_path = write_json(root / "paired_identity_bridge_v2.json", bridge_value)
    bridge_file_sha = bridge.file_sha256(bridge_path)

    member_specs: list[tuple[Path, str]] = []
    for index, (member, checkpoint, checkpoint_sha) in enumerate(
        zip(ensemble["members"], checkpoints, checkpoint_shas, strict=True)
    ):
        value = member_receipt(
            index=index, seed=member["member_seed"], checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha, shared=shared,
            source_checkpoint_sha=f"{200 + index:064x}",
        )
        path = write_json(root / f"adapter_member_{index}_receipt.json", value)
        member_specs.append((path, protocol.file_sha256(path)))

    decision = signed(
        {
            "format": authority.DECISION_FORMAT,
            "status": authority.DECISION_STATUS,
            "authority_issuer": "synthetic-independent-evaluator",
            "authority_issuer_identity_sha256": "4" * 64,
            "decision_nonce_sha256": "5" * 64,
            "identity_bridge_file_sha256": bridge_file_sha,
            "identity_bridge_sha256": bridge_value["bridge_sha256"],
            "pair_identity_set_sha256": bridge_value["pair_identity_set_sha256"],
            "deployment_binding_sha256": bridge_value["deployment"][
                "deployment_binding_sha256"
            ],
            "r7h_source_ensemble_contract_sha256": r7h_sha,
            "authorized_pair_count": 400,
            "target_manifest_evaluation400_is_only_lane": True,
            "additional_reserve400_authorized": False,
            "executor_independent_from_training_selection_and_protocol_freezer": True,
            "outcomes_or_trajectories_read_before_decision": False,
            "postfreeze_seed_candidate_or_threshold_change_authorized": False,
            "external_executor_only": True,
            "execution_authorized": True,
        },
        "decision_sha256",
    )
    decision_path = write_json(root / "independent_decision_v2.json", decision)
    decision_file_sha = protocol.file_sha256(decision_path)
    authority_value = authority.freeze_authority(
        identity_bridge_path=bridge_path,
        identity_bridge_file_sha256=bridge_file_sha,
        external_decision_path=decision_path,
        external_decision_file_sha256=decision_file_sha,
        expected_r7h_source_ensemble_contract_sha256=r7h_sha,
    )
    authority_path = write_json(root / "external_authority_v2.json", authority_value)
    kwargs = {
        "identity_bridge_path": bridge_path,
        "identity_bridge_file_sha256": bridge_file_sha,
        "external_authority_path": authority_path,
        "external_authority_file_sha256": protocol.file_sha256(authority_path),
        "head_support_path": data["paths"]["head_support"],
        "head_support_file_sha256": data["kwargs"]["head_support_file_sha256"],
        "ensemble_manifest_path": data["paths"]["ensemble_manifest"],
        "ensemble_manifest_file_sha256": data["kwargs"][
            "ensemble_manifest_file_sha256"
        ],
        "adapter_member_receipts": member_specs,
        "expected_r7h_source_ensemble_contract_sha256": r7h_sha,
    }
    return {"root": root, "kwargs": kwargs, "bridge": bridge_value,
            "authority": authority_value, "members": member_specs}


def test_freezes_only_evaluation400_with_exact_statistics_contract(tmp_path: Path) -> None:
    data = strict_fixture(tmp_path)
    value = protocol.freeze_protocol(**data["kwargs"])
    audit = protocol.validate_protocol(value)

    assert audit == {
        "protocol_sha256": value["protocol_sha256"],
        "pair_count": 400,
        "member_count": 5,
        "execution_started": False,
    }
    assert value["scope"]["additional_reserve400_count"] == 0
    assert [row["pair_id"] for row in value["pairs"]] == [
        row["pair_id"] for row in data["bridge"]["pairs"]
    ]
    assert value["deployment"]["six_primary_heads"] == list(protocol.HEAD_NAMES)
    assert value["deployment"]["single_checkpoint_accepted"] is False
    assert value["deployment"]["lobo_checkpoint_accepted"] is False
    result = value["result_protocol"]
    assert result["mcnemar"]["test"] == "exact_two_sided_binomial_on_n01_and_n10"
    assert result["paired_bootstrap"]["sampling_unit"] == "pair_id"
    assert result["paired_bootstrap"]["samples"] == 20_000
    assert result["success_rate_difference"]["direction"] == "etsf_minus_baseline"
    assert value["preexecution_capability_receipt"]["hdf5_files_opened"] == 0
    assert value["preexecution_capability_receipt"]["pair_conditions_executed"] == 0


def test_recovery_must_be_trained_supported_and_calibrated(tmp_path: Path) -> None:
    # The current identity bridge is shared by the legacy v2 fixture and the
    # v3 execution chain.  It now rejects a disabled recovery head before a
    # v2 protocol core can be materialized, which is the stricter fail-closed
    # boundary.  Do not weaken the bridge merely to reach the later v2 check.
    with pytest.raises(
        bridge.Evaluation400BridgeError,
        match="calibration/abstention boundary",
    ):
        strict_fixture(tmp_path, recovery_enabled=False)


def test_single_or_lobo_member_receipt_is_rejected(tmp_path: Path) -> None:
    data = strict_fixture(tmp_path)
    data["kwargs"]["adapter_member_receipts"] = data["members"][:1]
    with pytest.raises(protocol.PairedSuccessProtocolV2Error, match="five r7h"):
        protocol.freeze_protocol(**data["kwargs"])

    data = strict_fixture(tmp_path.parent / "second")
    member_path, _ = data["members"][0]
    member = json.loads(member_path.read_text(encoding="utf-8"))
    member.pop("receipt_sha256")
    member["format"] = "etsf_multibody_lobo_checkpoint_receipt_v1"
    write_json(member_path, signed(member, "receipt_sha256"))
    data["kwargs"]["adapter_member_receipts"][0] = (
        member_path, protocol.file_sha256(member_path)
    )
    with pytest.raises(protocol.PairedSuccessProtocolV2Error, match="contract changed"):
        protocol.freeze_protocol(**data["kwargs"])


def test_authority_bool_numeric_or_file_tamper_fails_closed(tmp_path: Path) -> None:
    data = strict_fixture(tmp_path)
    authority_path = data["kwargs"]["external_authority_path"]
    changed = copy.deepcopy(data["authority"])
    changed.pop("authority_sha256")
    changed["execution_scope"]["authorized_pair_count"] = True
    write_json(authority_path, signed(changed, "authority_sha256"))
    data["kwargs"]["external_authority_file_sha256"] = protocol.file_sha256(
        authority_path
    )
    with pytest.raises(protocol.PairedSuccessProtocolV2Error, match="authority"):
        protocol.freeze_protocol(**data["kwargs"])


def test_member_checkpoint_tamper_fails_even_with_receipt_unchanged(tmp_path: Path) -> None:
    data = strict_fixture(tmp_path)
    member_path, _ = data["members"][0]
    member = json.loads(member_path.read_text(encoding="utf-8"))
    Path(member["checkpoint_path"]).write_bytes(b"tampered")
    with pytest.raises(protocol.PairedSuccessProtocolV2Error, match="distinct"):
        protocol.freeze_protocol(**data["kwargs"])


def test_direct_sensitive_and_symlink_paths_fail_before_use(tmp_path: Path) -> None:
    data = strict_fixture(tmp_path)
    sensitive = data["root"] / "fresh_manifest.json"
    sensitive.write_text("{}", encoding="utf-8")
    data["kwargs"]["head_support_path"] = sensitive
    data["kwargs"]["head_support_file_sha256"] = protocol.file_sha256(sensitive)
    with pytest.raises(protocol.PairedSuccessProtocolV2Error, match="forbidden"):
        protocol.freeze_protocol(**data["kwargs"])

    data = strict_fixture(tmp_path.parent / "third")
    link = data["root"] / "head_link.json"
    link.symlink_to(data["kwargs"]["head_support_path"])
    data["kwargs"]["head_support_path"] = link
    with pytest.raises(protocol.PairedSuccessProtocolV2Error, match="symlink"):
        protocol.freeze_protocol(**data["kwargs"])


def test_protocol_output_is_create_once_owner_read_only(tmp_path: Path) -> None:
    data = strict_fixture(tmp_path)
    value = protocol.freeze_protocol(**data["kwargs"])
    output = data["root"] / "paired_protocol_v2.json"
    protocol.write_json_new(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        protocol.write_json_new(output, value)

    authority_output = data["root"] / "authority_copy_v2.json"
    authority.write_json_new(authority_output, data["authority"])
    assert stat.S_IMODE(authority_output.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        authority.write_json_new(authority_output, data["authority"])
