#!/usr/bin/env python3
"""Freeze the evaluation400 paired-success protocol v2 without executing it.

This protocol is deliberately a metadata/checkpoint-byte verifier.  It binds
the evaluation400 identity bridge, an independently issued execution authority,
the six-head calibrator support contract, and all five r7h-derived target
adapter members.  It never opens HDF5, trajectory, prediction, validation-label,
or outcome files and it never imports/deserializes a checkpoint or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import freeze_smolvla_piper_evaluation400_execution_authority_v2 as authority_v2
import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge_v2


FORMAT = "etsf_smolvla_piper_paired_task_success_protocol_v2"
STATUS = "frozen_evaluation400_paired_protocol_external_execution_not_started"
PAIR_RESULT_FORMAT = "etsf_smolvla_piper_paired_task_success_pair_result_v2"
EVALUATION_RESULT_FORMAT = "etsf_smolvla_piper_paired_task_success_evaluation_v2"
MEMBER_RECEIPT_FORMAT = "etsf_smolvla_piper_schema6_adapter_member_receipt_v2"
MEMBER_RECEIPT_STATUS = "complete_frozen_internal_validation_predictions"
HEAD_NAMES = (
    "post_event",
    "next_event",
    "duration",
    "success",
    "recovery",
    "object_effect",
)
PAIR_COUNT = bridge_v2.EVALUATION_GROUPS
MEMBER_COUNT = bridge_v2.MEMBER_COUNT
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_SAMPLES = 20_000
CONFIDENCE_LEVEL = 0.95
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
FORBIDDEN_PATH_TOKENS = ("fresh", "confirmation", "trajectory", "label")
SHARED_FIELDS = {
    "training_manifest_sha256",
    "split_sha256",
    "source_ensemble_contract_sha256",
    "prediction_contract_sha256",
}
MEMBER_RECEIPT_FIELDS = {
    "format", "status", "member_index", "member_seed",
    "source_checkpoint_sha256", "training_manifest_sha256", "split_sha256",
    "source_ensemble_contract_sha256", "summary_path", "summary_file_sha256",
    "summary_sha256", "checkpoint_path", "checkpoint_file_sha256",
    "validation_predictions_path", "validation_predictions_file_sha256",
    "validation_predictions_logical_sha256", "validation_labels_path",
    "validation_labels_file_sha256", "validation_labels_logical_sha256",
    "validation_identity_set_sha256", "validation_lane",
    "duration_target_transform", "next_event_observation_mask",
    "success_target", "recovery_target", "recovery_observation_mask",
    "recovery_shared_transition_stop_gradient",
    "recovery_enters_primary_before_calibration", "recovery_head_trained",
    "object_prediction_space", "object_source_normalization_sha256",
    "object_observed_policy", "target_validation50_hdf5_files_opened",
    "sealed_test_labels_opened", "receipt_sha256",
}


class PairedSuccessProtocolV2Error(RuntimeError):
    """An immutable identity, deployment, or pre-outcome boundary failed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def file_sha256(path: Path) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise PairedSuccessProtocolV2Error("HDF input is forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sensitive(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            return True
        if lowered == "test" or lowered.startswith(("test_", "test-")):
            return True
    return False


def safe_file(path: Path, role: str, *, json_only: bool) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if _sensitive(PurePath(lexical)) or lexical.is_symlink():
        raise PairedSuccessProtocolV2Error(f"{role} path is forbidden or a symlink")
    resolved = lexical.resolve(strict=True)
    if _sensitive(PurePath(resolved)) or not resolved.is_file():
        raise PairedSuccessProtocolV2Error(f"{role} must be a safe materialized file")
    if resolved.suffix.casefold() in HDF_SUFFIXES:
        raise PairedSuccessProtocolV2Error("HDF input is forbidden")
    if json_only and resolved.suffix.casefold() != ".json":
        raise PairedSuccessProtocolV2Error(f"{role} must be JSON")
    return resolved


def safe_new_json(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if _sensitive(PurePath(lexical)) or lexical.suffix.casefold() != ".json":
        raise PairedSuccessProtocolV2Error("protocol output path is forbidden")
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError(lexical)
    parent = lexical.parent.resolve(strict=True)
    if _sensitive(PurePath(parent)) or not parent.is_dir():
        raise PairedSuccessProtocolV2Error("protocol output parent is forbidden")
    return lexical


def read_bound_json(
    path: Path, expected_file_sha256: str, role: str
) -> tuple[Path, dict[str, Any]]:
    if not is_sha(expected_file_sha256):
        raise PairedSuccessProtocolV2Error(f"{role} expected SHA is invalid")
    resolved = safe_file(path, role, json_only=True)
    before = file_sha256(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairedSuccessProtocolV2Error(f"{role} is not valid JSON") from error
    after = file_sha256(resolved)
    if before != after or before != expected_file_sha256:
        raise PairedSuccessProtocolV2Error(f"{role} file SHA changed")
    if not isinstance(value, dict):
        raise PairedSuccessProtocolV2Error(f"{role} must contain an object")
    return resolved, value


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise PairedSuccessProtocolV2Error(f"{role} logical SHA mismatch")
    return str(recorded)


def _require_six_primary_heads(
    calibration: Mapping[str, Any], head_support: Mapping[str, Any]
) -> None:
    calibrated = calibration.get("head_enabled_for_primary")
    heads = head_support.get("heads")
    recovery = heads.get("recovery") if isinstance(heads, Mapping) else None
    if (
        not isinstance(calibrated, Mapping)
        or set(calibrated) != set(HEAD_NAMES)
        or any(calibrated.get(name) is not True for name in HEAD_NAMES)
        or not isinstance(heads, Mapping)
        or set(heads) != set(HEAD_NAMES)
        or any(heads[name].get("enabled_for_primary") is not True for name in HEAD_NAMES)
        or not isinstance(recovery, Mapping)
        or recovery.get("support_threshold_met") is not True
        or recovery.get("all_member_recovery_heads_trained") is not True
        or calibration.get("recovery_temperature_fitted_on_validation_only") is not True
    ):
        raise PairedSuccessProtocolV2Error(
            "all six calibrated heads, including trained recovery, are required"
        )


def _validate_member_receipt(
    value: Mapping[str, Any], *, index: int, ensemble_member: Mapping[str, Any],
    shared: Mapping[str, Any], expected_source_contract_sha256: str,
) -> str:
    logical = verify_signed(value, "receipt_sha256", f"target adapter member {index}")
    sha_fields = {
        "source_checkpoint_sha256", "training_manifest_sha256", "split_sha256",
        "source_ensemble_contract_sha256", "summary_file_sha256", "summary_sha256",
        "checkpoint_file_sha256", "validation_predictions_file_sha256",
        "validation_predictions_logical_sha256", "validation_labels_file_sha256",
        "validation_labels_logical_sha256", "validation_identity_set_sha256",
        "object_source_normalization_sha256",
    }
    if (
        set(value) != MEMBER_RECEIPT_FIELDS
        or value.get("format") != MEMBER_RECEIPT_FORMAT
        or value.get("status") != MEMBER_RECEIPT_STATUS
        or type(value.get("member_index")) is not int
        or value["member_index"] != index
        or type(value.get("member_seed")) is not int
        or value["member_seed"] != ensemble_member.get("member_seed")
        or value.get("checkpoint_path") != ensemble_member.get("checkpoint_path")
        or value.get("checkpoint_file_sha256")
        != ensemble_member.get("checkpoint_file_sha256")
        or any(not is_sha(value.get(field)) for field in sha_fields)
        or value.get("training_manifest_sha256")
        != shared.get("training_manifest_sha256")
        or value.get("split_sha256") != shared.get("split_sha256")
        or value.get("source_ensemble_contract_sha256")
        != expected_source_contract_sha256
        or value.get("source_ensemble_contract_sha256")
        != shared.get("source_ensemble_contract_sha256")
        or value.get("validation_lane")
        != "adaptation_derived_internal_validation_only"
        or value.get("duration_target_transform") != "log1p_decision_steps"
        or value.get("next_event_observation_mask") != "duration_observed"
        or value.get("success_target")
        != "eventual_final_branch_success_repeated_per_transition"
        or value.get("recovery_target")
        != "conditional_recovery_given_operational_regress"
        or value.get("recovery_observation_mask")
        != "recovery_observed_and_regress"
        or value.get("recovery_shared_transition_stop_gradient") is not True
        or value.get("recovery_enters_primary_before_calibration") is not False
        or value.get("recovery_head_trained") is not True
        or value.get("object_prediction_space") != "physical_delta_xyz_m"
        or value.get("object_observed_policy")
        != "row_enabled_only_if_all_selected_xyz_are_valid"
        or type(value.get("target_validation50_hdf5_files_opened")) is not int
        or value["target_validation50_hdf5_files_opened"] != 0
        or type(value.get("sealed_test_labels_opened")) is not int
        or value["sealed_test_labels_opened"] != 0
    ):
        raise PairedSuccessProtocolV2Error(
            f"target adapter member {index} contract changed"
        )
    return logical


def _load_target_members(
    specs: Sequence[tuple[Path, str]], *, ensemble: Mapping[str, Any],
    expected_source_contract_sha256: str,
) -> list[dict[str, Any]]:
    members = ensemble.get("members")
    shared = ensemble.get("shared_contract")
    if (
        len(specs) != MEMBER_COUNT
        or not isinstance(members, list)
        or len(members) != MEMBER_COUNT
        or not isinstance(shared, Mapping)
        or set(shared) != SHARED_FIELDS
        or any(not is_sha(shared.get(field)) for field in SHARED_FIELDS)
        or shared.get("source_ensemble_contract_sha256")
        != expected_source_contract_sha256
    ):
        raise PairedSuccessProtocolV2Error("exactly five r7h target adapters are required")
    decoded: list[dict[str, Any]] = []
    receipt_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    seeds: set[int] = set()
    for index, ((receipt_path, receipt_file_sha), ensemble_member) in enumerate(
        zip(specs, members, strict=True)
    ):
        resolved_receipt, receipt = read_bound_json(
            receipt_path, receipt_file_sha, f"target adapter member {index} receipt"
        )
        logical = _validate_member_receipt(
            receipt, index=index, ensemble_member=ensemble_member, shared=shared,
            expected_source_contract_sha256=expected_source_contract_sha256,
        )
        checkpoint = safe_file(
            Path(str(receipt["checkpoint_path"])),
            f"target adapter member {index} checkpoint", json_only=False,
        )
        checkpoint_sha = file_sha256(checkpoint)
        seed = receipt["member_seed"]
        if (
            checkpoint_sha != receipt["checkpoint_file_sha256"]
            or receipt_file_sha in receipt_hashes
            or checkpoint_sha in checkpoint_hashes
            or seed in seeds
        ):
            raise PairedSuccessProtocolV2Error("target adapter members are not distinct")
        receipt_hashes.add(receipt_file_sha)
        checkpoint_hashes.add(checkpoint_sha)
        seeds.add(seed)
        decoded.append(
            {
                "member_index": index,
                "member_seed": seed,
                "receipt_path": str(resolved_receipt),
                "receipt_file_sha256": receipt_file_sha,
                "receipt_sha256": logical,
                "checkpoint_path": str(checkpoint),
                "checkpoint_file_sha256": checkpoint_sha,
                "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
                "recovery_head_trained": True,
            }
        )
    return decoded


def _reconstruct_authority(
    path: Path, expected_file_sha256: str,
    *, bridge_path: Path, bridge_file_sha256: str,
    expected_source_contract_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    resolved, value = read_bound_json(path, expected_file_sha256, "external authority v2")
    try:
        audit = authority_v2.validate_authority(value)
    except authority_v2.ExternalAuthorityV2Error as error:
        raise PairedSuccessProtocolV2Error("external authority v2 failed") from error
    identity = value["identity_bridge"]
    decision = value["external_decision"]
    if (
        identity.get("path") != str(bridge_path)
        or audit["identity_bridge_file_sha256"] != bridge_file_sha256
        or audit["r7h_source_ensemble_contract_sha256"]
        != expected_source_contract_sha256
    ):
        raise PairedSuccessProtocolV2Error("external authority binds another deployment")
    try:
        rebuilt = authority_v2.freeze_authority(
            identity_bridge_path=bridge_path,
            identity_bridge_file_sha256=bridge_file_sha256,
            external_decision_path=Path(str(decision["path"])),
            external_decision_file_sha256=str(decision["file_sha256"]),
            expected_r7h_source_ensemble_contract_sha256=(
                expected_source_contract_sha256
            ),
        )
    except authority_v2.ExternalAuthorityV2Error as error:
        raise PairedSuccessProtocolV2Error("external authority reconstruction failed") from error
    if rebuilt != value:
        raise PairedSuccessProtocolV2Error("external authority differs from reconstruction")
    return resolved, value


def _result_contract() -> dict[str, Any]:
    return {
        "pair_result_format": PAIR_RESULT_FORMAT,
        "evaluation_result_format": EVALUATION_RESULT_FORMAT,
        "required_complete_pair_rows": PAIR_COUNT,
        "pair_row_exact_fields": [
            "ordinal", "pair_id", "condition_order_executed",
            "same_reset_identity_reverified", "baseline_success", "etsf_success",
            "baseline_execution_receipt_sha256", "etsf_execution_receipt_sha256",
        ],
        "binary_success_values": [0, 1],
        "exact_pair_identity_and_frozen_order_required": True,
        "incomplete_duplicate_or_mismatched_pair_fails_closed": True,
        "success_rate_difference": {
            "point_estimate": "mean(etsf_success-baseline_success)",
            "direction": "etsf_minus_baseline",
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_fields": ["lower", "upper", "confidence_level"],
        },
        "mcnemar": {
            "contingency_cells": ["n00", "n01", "n10", "n11"],
            "n01_semantics": "baseline_success_etsf_failure",
            "n10_semantics": "baseline_failure_etsf_success",
            "test": "exact_two_sided_binomial_on_n01_and_n10",
            "p_value_field": "exact_two_sided_p_value",
        },
        "paired_bootstrap": {
            "sampling_unit": "pair_id",
            "statistic": "mean(etsf_success-baseline_success)",
            "replacement": True,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "two_sided_percentile",
            "confidence_level": CONFIDENCE_LEVEL,
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
        },
        "posthoc_seed_candidate_threshold_or_subgroup_selection_allowed": False,
    }


def freeze_protocol(
    *, identity_bridge_path: Path, identity_bridge_file_sha256: str,
    external_authority_path: Path, external_authority_file_sha256: str,
    head_support_path: Path, head_support_file_sha256: str,
    ensemble_manifest_path: Path, ensemble_manifest_file_sha256: str,
    adapter_member_receipts: Sequence[tuple[Path, str]],
    expected_r7h_source_ensemble_contract_sha256: str,
) -> dict[str, Any]:
    supplied_shas = (
        identity_bridge_file_sha256, external_authority_file_sha256,
        head_support_file_sha256, ensemble_manifest_file_sha256,
        expected_r7h_source_ensemble_contract_sha256,
        *(sha for _, sha in adapter_member_receipts),
    )
    if any(not is_sha(value) for value in supplied_shas):
        raise PairedSuccessProtocolV2Error("all externally expected SHAs are required")
    try:
        bridge_path, bridge, dependencies = authority_v2.reconstruct_bound_bridge(
            identity_bridge_path, identity_bridge_file_sha256
        )
    except authority_v2.ExternalAuthorityV2Error as error:
        raise PairedSuccessProtocolV2Error("identity bridge reconstruction failed") from error
    bridge_head = bridge["dependencies"]["head_support"]
    bridge_ensemble = bridge["dependencies"]["ensemble_manifest"]
    supplied_head = safe_file(head_support_path, "head support v2", json_only=True)
    supplied_ensemble = safe_file(
        ensemble_manifest_path, "target adapter ensemble", json_only=True
    )
    if (
        str(supplied_head) != bridge_head["path"]
        or head_support_file_sha256 != bridge_head["file_sha256"]
        or str(supplied_ensemble) != bridge_ensemble["path"]
        or ensemble_manifest_file_sha256 != bridge_ensemble["file_sha256"]
    ):
        raise PairedSuccessProtocolV2Error("explicit calibrator inputs differ from bridge")
    calibration_value = dependencies["calibration"]
    head_value = dependencies["head_support"]
    ensemble_value = dependencies["ensemble_manifest"]
    try:
        calibration = bridge_v2.validate_calibration(calibration_value)
        head = bridge_v2.validate_head_support(head_value)
        ensemble = bridge_v2.validate_ensemble_manifest(
            ensemble_value, calibration=calibration, head=head
        )
    except bridge_v2.Evaluation400BridgeError as error:
        raise PairedSuccessProtocolV2Error("calibrator deployment contract failed") from error
    _require_six_primary_heads(calibration_value, head_value)
    try:
        authority_v2.validate_r7h_ensemble(
            ensemble_value,
            expected_source_contract_sha256=(
                expected_r7h_source_ensemble_contract_sha256
            ),
        )
    except authority_v2.ExternalAuthorityV2Error as error:
        raise PairedSuccessProtocolV2Error("ensemble is not the r7h target adapter ensemble") from error
    authority_path, authority = _reconstruct_authority(
        external_authority_path, external_authority_file_sha256,
        bridge_path=bridge_path, bridge_file_sha256=identity_bridge_file_sha256,
        expected_source_contract_sha256=expected_r7h_source_ensemble_contract_sha256,
    )
    members = _load_target_members(
        adapter_member_receipts, ensemble=ensemble_value,
        expected_source_contract_sha256=(
            expected_r7h_source_ensemble_contract_sha256
        ),
    )
    if authority["deployment"]["member_checkpoint_sha256"] != [
        row["checkpoint_file_sha256"] for row in members
    ]:
        raise PairedSuccessProtocolV2Error("authority/member checkpoint set mismatch")
    pairs = [
        {
            "ordinal": row["ordinal"],
            "pair_id": row["pair_id"],
            "target_manifest_global_ordinal": row["target_manifest_global_ordinal"],
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
            "initial_scene_state_sha256": row["initial_scene_state_sha256"],
            "initial_measured_joint_state_sha256": row[
                "initial_measured_joint_state_sha256"
            ],
            "initial_commanded_drive_target_sha256": row[
                "initial_commanded_drive_target_sha256"
            ],
            "condition_order": list(row["condition_order"]),
            "baseline_candidate_selector": row["baseline_condition"]["selector"],
            "etsf_candidate_selector": row["etsf_condition"]["selector"],
            "candidate_count": bridge_v2.CANDIDATE_COUNT,
            "same_initial_state_for_both_conditions": True,
            "result_or_outcome_fields_present": False,
        }
        for row in bridge["pairs"]
    ]
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "identity_bridge": {
            "path": str(bridge_path),
            "file_sha256": identity_bridge_file_sha256,
            "logical_sha256": bridge["bridge_sha256"],
            "pair_identity_set_sha256": bridge["pair_identity_set_sha256"],
        },
        "external_execution_authority": {
            "path": str(authority_path),
            "file_sha256": external_authority_file_sha256,
            "logical_sha256": authority["authority_sha256"],
            "independent_issuer_identity_sha256": authority["external_decision"][
                "authority_issuer_identity_sha256"
            ],
            "external_executor_only": True,
        },
        "deployment": {
            "source_lineage": "r7h",
            "r7h_source_ensemble_contract_sha256": (
                expected_r7h_source_ensemble_contract_sha256
            ),
            "target_adapter_ensemble_manifest_path": str(supplied_ensemble),
            "target_adapter_ensemble_manifest_file_sha256": (
                ensemble_manifest_file_sha256
            ),
            "target_adapter_ensemble_manifest_sha256": ensemble[
                "ensemble_manifest_sha256"
            ],
            "head_support_path": str(supplied_head),
            "head_support_file_sha256": head_support_file_sha256,
            "head_support_sha256": head["head_support_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
            "deployment_binding_sha256": bridge["deployment"][
                "deployment_binding_sha256"
            ],
            "abstention_contract_sha256": calibration[
                "abstention_contract_sha256"
            ],
            "six_primary_heads": list(HEAD_NAMES),
            "member_count": MEMBER_COUNT,
            "members": members,
            "single_checkpoint_accepted": False,
            "lobo_checkpoint_accepted": False,
            "checkpoint_deserialization_performed": False,
        },
        "scope": {
            "pair_count": PAIR_COUNT,
            "target_manifest_evaluation400_is_only_final_paired_lane": True,
            "additional_reserve400_required": False,
            "additional_reserve400_count": 0,
            "baseline_vs_event_rerank_in_same_evaluation_lane": True,
            "exact_frozen_pair_order_required": True,
            "same_initial_state_for_both_conditions_required": True,
            "postfreeze_seed_candidate_or_threshold_change_allowed": False,
            "protocol_freezer_may_execute": False,
        },
        "pairs": pairs,
        "result_protocol": _result_contract(),
        "preexecution_capability_receipt": {
            "unique_json_artifacts_bound": 15,
            "checkpoint_files_hashed_as_opaque_bytes": MEMBER_COUNT,
            "checkpoint_deserialization_calls": 0,
            "hdf5_files_opened": 0,
            "trajectory_files_opened": 0,
            "prediction_files_opened": 0,
            "validation_label_files_opened": 0,
            "evaluation_label_or_outcome_files_opened": 0,
            "outcomes_read": False,
            "policy_import_or_forward_calls": 0,
            "simulator_calls": 0,
            "pair_conditions_executed": 0,
        },
    }
    return {**base, "protocol_sha256": canonical_sha256(base)}


def validate_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = verify_signed(value, "protocol_sha256", "paired protocol v2")
    expected_root = {
        "format", "status", "identity_bridge", "external_execution_authority",
        "deployment", "scope", "pairs", "result_protocol",
        "preexecution_capability_receipt", "protocol_sha256",
    }
    deployment = value.get("deployment")
    identity = value.get("identity_bridge")
    external = value.get("external_execution_authority")
    scope = value.get("scope")
    pairs = value.get("pairs")
    capability = value.get("preexecution_capability_receipt")
    if (
        set(value) != expected_root
        or value.get("format") != FORMAT
        or value.get("status") != STATUS
        or not isinstance(identity, Mapping)
        or set(identity) != {
            "path", "file_sha256", "logical_sha256", "pair_identity_set_sha256"
        }
        or not isinstance(identity.get("path"), str)
        or any(
            not is_sha(identity.get(field))
            for field in ("file_sha256", "logical_sha256", "pair_identity_set_sha256")
        )
        or not isinstance(external, Mapping)
        or set(external) != {
            "path", "file_sha256", "logical_sha256",
            "independent_issuer_identity_sha256", "external_executor_only",
        }
        or not isinstance(external.get("path"), str)
        or any(
            not is_sha(external.get(field))
            for field in (
                "file_sha256", "logical_sha256", "independent_issuer_identity_sha256"
            )
        )
        or external.get("external_executor_only") is not True
        or not isinstance(deployment, Mapping)
        or set(deployment) != {
            "source_lineage", "r7h_source_ensemble_contract_sha256",
            "target_adapter_ensemble_manifest_path",
            "target_adapter_ensemble_manifest_file_sha256",
            "target_adapter_ensemble_manifest_sha256", "head_support_path",
            "head_support_file_sha256", "head_support_sha256",
            "calibration_sha256", "deployment_binding_sha256",
            "abstention_contract_sha256", "six_primary_heads", "member_count",
            "members", "single_checkpoint_accepted", "lobo_checkpoint_accepted",
            "checkpoint_deserialization_performed",
        }
        or deployment.get("source_lineage") != "r7h"
        or any(
            not is_sha(deployment.get(field))
            for field in (
                "r7h_source_ensemble_contract_sha256",
                "target_adapter_ensemble_manifest_file_sha256",
                "target_adapter_ensemble_manifest_sha256",
                "head_support_file_sha256", "head_support_sha256",
                "calibration_sha256", "deployment_binding_sha256",
                "abstention_contract_sha256",
            )
        )
        or type(deployment.get("member_count")) is not int
        or deployment["member_count"] != MEMBER_COUNT
        or not isinstance(deployment.get("members"), list)
        or len(deployment["members"]) != MEMBER_COUNT
        or deployment.get("six_primary_heads") != list(HEAD_NAMES)
        or deployment.get("single_checkpoint_accepted") is not False
        or deployment.get("lobo_checkpoint_accepted") is not False
        or deployment.get("checkpoint_deserialization_performed") is not False
        or not isinstance(scope, Mapping)
        or set(scope) != {
            "pair_count",
            "target_manifest_evaluation400_is_only_final_paired_lane",
            "additional_reserve400_required", "additional_reserve400_count",
            "baseline_vs_event_rerank_in_same_evaluation_lane",
            "exact_frozen_pair_order_required",
            "same_initial_state_for_both_conditions_required",
            "postfreeze_seed_candidate_or_threshold_change_allowed",
            "protocol_freezer_may_execute",
        }
        or type(scope.get("pair_count")) is not int
        or scope["pair_count"] != PAIR_COUNT
        or scope.get("target_manifest_evaluation400_is_only_final_paired_lane") is not True
        or scope.get("additional_reserve400_required") is not False
        or type(scope.get("additional_reserve400_count")) is not int
        or scope["additional_reserve400_count"] != 0
        or scope.get("baseline_vs_event_rerank_in_same_evaluation_lane") is not True
        or scope.get("exact_frozen_pair_order_required") is not True
        or scope.get("same_initial_state_for_both_conditions_required") is not True
        or scope.get("postfreeze_seed_candidate_or_threshold_change_allowed") is not False
        or scope.get("protocol_freezer_may_execute") is not False
        or not isinstance(pairs, list)
        or len(pairs) != PAIR_COUNT
        or value.get("result_protocol") != _result_contract()
        or not isinstance(capability, Mapping)
        or set(capability) != {
            "unique_json_artifacts_bound", "checkpoint_files_hashed_as_opaque_bytes",
            "checkpoint_deserialization_calls", "hdf5_files_opened",
            "trajectory_files_opened", "prediction_files_opened",
            "validation_label_files_opened",
            "evaluation_label_or_outcome_files_opened", "outcomes_read",
            "policy_import_or_forward_calls", "simulator_calls",
            "pair_conditions_executed",
        }
        or type(capability.get("unique_json_artifacts_bound")) is not int
        or capability["unique_json_artifacts_bound"] != 15
        or type(capability.get("checkpoint_files_hashed_as_opaque_bytes")) is not int
        or capability["checkpoint_files_hashed_as_opaque_bytes"] != MEMBER_COUNT
        or any(
            type(capability.get(field)) is not int or capability[field] != 0
            for field in (
                "checkpoint_deserialization_calls", "hdf5_files_opened",
                "trajectory_files_opened", "prediction_files_opened",
                "validation_label_files_opened",
                "evaluation_label_or_outcome_files_opened",
                "policy_import_or_forward_calls", "simulator_calls",
                "pair_conditions_executed",
            )
        )
        or capability.get("outcomes_read") is not False
    ):
        raise PairedSuccessProtocolV2Error("paired protocol v2 boundary changed")
    expected_member_fields = {
        "member_index", "member_seed", "receipt_path", "receipt_file_sha256",
        "receipt_sha256", "checkpoint_path", "checkpoint_file_sha256",
        "source_checkpoint_sha256", "recovery_head_trained",
    }
    member_seeds: set[int] = set()
    member_checkpoints: set[str] = set()
    member_receipts: set[str] = set()
    for index, member in enumerate(deployment["members"]):
        if (
            not isinstance(member, Mapping)
            or set(member) != expected_member_fields
            or type(member.get("member_index")) is not int
            or member["member_index"] != index
            or type(member.get("member_seed")) is not int
            or member["member_seed"] in member_seeds
            or any(
                not is_sha(member.get(field))
                for field in (
                    "receipt_file_sha256", "receipt_sha256",
                    "checkpoint_file_sha256", "source_checkpoint_sha256",
                )
            )
            or member["checkpoint_file_sha256"] in member_checkpoints
            or member["receipt_file_sha256"] in member_receipts
            or member.get("recovery_head_trained") is not True
        ):
            raise PairedSuccessProtocolV2Error("target adapter deployment changed")
        member_seeds.add(member["member_seed"])
        member_checkpoints.add(member["checkpoint_file_sha256"])
        member_receipts.add(member["receipt_file_sha256"])
    pair_ids: set[str] = set()
    expected_pair_fields = {
        "ordinal", "pair_id", "target_manifest_global_ordinal", "requested_seed",
        "resolved_seed", "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256", "condition_order",
        "baseline_candidate_selector", "etsf_candidate_selector",
        "candidate_count", "same_initial_state_for_both_conditions",
        "result_or_outcome_fields_present",
    }
    for ordinal, row in enumerate(pairs):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_pair_fields
            or type(row.get("ordinal")) is not int
            or row["ordinal"] != ordinal
            or type(row.get("target_manifest_global_ordinal")) is not int
            or row["target_manifest_global_ordinal"] != 130 + ordinal
            or type(row.get("requested_seed")) is not int
            or type(row.get("resolved_seed")) is not int
            or not is_sha(row.get("pair_id"))
            or row["pair_id"] in pair_ids
            or any(
                not is_sha(row.get(field))
                for field in (
                    "initial_scene_state_sha256",
                    "initial_measured_joint_state_sha256",
                    "initial_commanded_drive_target_sha256",
                )
            )
            or row.get("condition_order")
            != bridge_v2.paired_condition_order(str(row.get("pair_id")))
            or row.get("baseline_candidate_selector")
            != "lowest_legal_feasibility_root_candidate"
            or row.get("etsf_candidate_selector")
            != "frozen_five_member_event_world_model_with_uncertainty_abstention"
            or type(row.get("candidate_count")) is not int
            or row["candidate_count"] != bridge_v2.CANDIDATE_COUNT
            or row.get("same_initial_state_for_both_conditions") is not True
            or row.get("result_or_outcome_fields_present") is not False
        ):
            raise PairedSuccessProtocolV2Error("paired identity/order changed")
        pair_ids.add(row["pair_id"])
    return {
        "protocol_sha256": logical,
        "pair_count": PAIR_COUNT,
        "member_count": MEMBER_COUNT,
        "execution_started": False,
    }


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_new_json(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bridge", type=Path, required=True)
    parser.add_argument("--identity-bridge-file-sha256", required=True)
    parser.add_argument("--external-authority", type=Path, required=True)
    parser.add_argument("--external-authority-file-sha256", required=True)
    parser.add_argument("--head-support", type=Path, required=True)
    parser.add_argument("--head-support-file-sha256", required=True)
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--ensemble-manifest-file-sha256", required=True)
    parser.add_argument(
        "--adapter-member-receipt", nargs=2, action="append", required=True,
        metavar=("PATH", "EXPECTED_FILE_SHA256"),
    )
    parser.add_argument("--expected-r7h-source-ensemble-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    value = freeze_protocol(
        identity_bridge_path=args.identity_bridge,
        identity_bridge_file_sha256=args.identity_bridge_file_sha256,
        external_authority_path=args.external_authority,
        external_authority_file_sha256=args.external_authority_file_sha256,
        head_support_path=args.head_support,
        head_support_file_sha256=args.head_support_file_sha256,
        ensemble_manifest_path=args.ensemble_manifest,
        ensemble_manifest_file_sha256=args.ensemble_manifest_file_sha256,
        adapter_member_receipts=[
            (Path(path), sha) for path, sha in args.adapter_member_receipt
        ],
        expected_r7h_source_ensemble_contract_sha256=(
            args.expected_r7h_source_ensemble_contract_sha256
        ),
    )
    validate_protocol(value)
    write_json_new(args.output, value)
    print(json.dumps({"protocol_sha256": value["protocol_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
