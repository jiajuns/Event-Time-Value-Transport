#!/usr/bin/env python3
"""Freeze the target-manifest evaluation400 as the only paired-success lane.

The bridge consumes signed identity/deployment receipts only.  It verifies and
binds the target manifest's 400 evaluation reset identities, the independent
selected-identity attestation, the five-member validation-only ensemble,
calibration/head-support/abstention artifacts, and the exact SmolVLA policy
feature/action runtime bridge.  It opens no HDF5, trajectory, label, outcome,
checkpoint, simulator, or policy runtime and grants no execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import smolvla_piper_deployment_uncertainty_v1 as deployment_uncertainty_v1


FORMAT = "etsf_smolvla_piper_evaluation400_paired_identity_bridge_v2"
STATUS = "frozen_preoutcome_identity_bridge_external_execution_not_authorized"
TARGET_MANIFEST_FORMAT = "etsf_smolvla_piper_target_seed_manifest_v2"
TARGET_MANIFEST_STATUS = "resolved_reset_identity_only_before_policy_execution"
ATTESTATION_FORMAT = "etsf_private_identity_disjoint_attestation_v1"
ATTESTATION_STATUS = "verified_disjoint_without_disclosing_heldout_identities"
ENSEMBLE_FORMAT = "etsf_smolvla_piper_adapter_ensemble_manifest_v2"
ENSEMBLE_STATUS = "frozen_validation_only_five_member_deployment_contract"
CALIBRATION_FORMAT = "etsf_smolvla_piper_adapter_ensemble_calibration_v2"
CALIBRATION_STATUS = "complete_validation_only_metrics_and_threshold_freeze"
HEAD_SUPPORT_FORMAT = "etsf_smolvla_piper_multitask_head_support_v2"
HEAD_SUPPORT_STATUS = (
    "frozen_from_training_and_validation_only_before_paired_development"
)
CALIBRATION_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_adapter_ensemble_validation_receipt_v2"
)
CALIBRATION_RECEIPT_STATUS = "complete_validation_only_five_member_calibration"
EXTERNAL_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_evaluation400_paired_execution_authority_v2"
)
TASK = "move_can_pot"
SOURCE_BODY = "aloha"
TARGET_BODY = "piper_piper_0.6"
ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
INSTRUCTION = "move the can into the pot"
LABEL_ACCESS_CONTRACT = (
    "reset_identity_instruction_and_initial_state_hash_only_no_policy_action_"
    "reward_success_event_or_outcome"
)
ADAPTATION_GROUPS = 80
VALIDATION_GROUPS = 50
EVALUATION_GROUPS = 400
TOTAL_GROUPS = ADAPTATION_GROUPS + VALIDATION_GROUPS + EVALUATION_GROUPS
MEMBER_COUNT = 5
CANDIDATE_COUNT = 4
CONDITION_ORDER_NAMESPACE = "schema6_evaluation400_paired_condition_order_v2"
ACTION_MAPPING = "native_feature_i_to_model_slot_i_no_coordinate_transform_v1"
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)
CORE_HEADS = (
    "post_event", "next_event", "duration", "success", "recovery",
    "object_effect",
)
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
FORBIDDEN_PATH_TOKENS = ("fresh", "confirmation", "trajectory", "label")


class Evaluation400BridgeError(RuntimeError):
    """A pre-outcome identity or deployment binding failed closed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def file_sha256(path: Path) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise Evaluation400BridgeError("HDF input is forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise Evaluation400BridgeError(f"{role} logical signature mismatch")
    return str(recorded)


def validate_source_rank_member_authority(
    value: Any, recorded_sha256: Any, *, role: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"source_rank_numeric_contract", "members"}
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not is_sha(recorded_sha256)
        or canonical_sha256(value) != recorded_sha256
        or not isinstance(value.get("members"), list)
        or len(value["members"]) != MEMBER_COUNT
    ):
        raise Evaluation400BridgeError(f"{role} is invalid")
    source_shas: list[str] = []
    contract_shas: list[str] = []
    for index, member in enumerate(value["members"]):
        if (
            not isinstance(member, Mapping)
            or set(member) != {
                "member_index", "source_checkpoint_file_sha256",
                "source_rank_score_contract_sha256", "success_temperature",
            }
            or type(member.get("member_index")) is not int
            or member["member_index"] != index
            or not is_sha(member.get("source_checkpoint_file_sha256"))
            or not is_sha(member.get("source_rank_score_contract_sha256"))
            or isinstance(member.get("success_temperature"), bool)
            or not isinstance(member.get("success_temperature"), (int, float))
            or not math.isfinite(float(member["success_temperature"]))
            or float(member["success_temperature"]) <= 0.0
        ):
            raise Evaluation400BridgeError(f"{role} is invalid")
        source_shas.append(str(member["source_checkpoint_file_sha256"]))
        contract_shas.append(str(member["source_rank_score_contract_sha256"]))
    if len(set(source_shas)) != MEMBER_COUNT or len(set(contract_shas)) != MEMBER_COUNT:
        raise Evaluation400BridgeError(f"{role} is invalid")
    return {
        "source_rank_member_authority": {
            "source_rank_numeric_contract": value[
                "source_rank_numeric_contract"
            ],
            "members": [dict(member) for member in value["members"]],
        },
        "source_rank_member_authority_sha256": str(recorded_sha256),
    }


def _safe_json_path(raw: Path, role: str, *, must_exist: bool) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(raw))))
    if path.suffix.casefold() != ".json":
        raise Evaluation400BridgeError(f"{role} must be a JSON file")
    if any(
        token in component.casefold()
        for component in path.parts
        for token in FORBIDDEN_PATH_TOKENS
    ):
        raise Evaluation400BridgeError(f"{role} path is in a forbidden namespace")
    if must_exist:
        if not path.is_file() or path.is_symlink():
            raise Evaluation400BridgeError(f"{role} must be an existing regular file")
    elif path.exists() or path.is_symlink():
        raise FileExistsError(path)
    return path


def _read_bound_json(
    path: Path, expected_file_sha256: str, role: str
) -> tuple[Path, dict[str, Any]]:
    if not is_sha(expected_file_sha256):
        raise Evaluation400BridgeError(f"{role} expected file SHA is invalid")
    resolved = _safe_json_path(path, role, must_exist=True)
    if file_sha256(resolved) != expected_file_sha256:
        raise Evaluation400BridgeError(f"{role} file SHA mismatch")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Evaluation400BridgeError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise Evaluation400BridgeError(f"{role} must contain a JSON object")
    return resolved, value


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "task",
            "actor_id",
            "target_body",
            "split",
            "ordinal",
            "requested_seed",
            "resolved_seed",
            "instruction_sha256",
            "instruction_semantics_receipt_sha256",
            "initial_scene_state_sha256",
            "initial_measured_joint_state_sha256",
            "initial_commanded_drive_target_sha256",
        )
    }


def _validate_target_row(
    row: Any, *, split: str, ordinal: int, global_ordinal: int
) -> dict[str, Any]:
    exact_fields = {
        "task",
        "actor_id",
        "target_body",
        "global_ordinal",
        "split",
        "ordinal",
        "stage_role",
        "requested_seed",
        "resolved_seed",
        "instruction",
        "instruction_sha256",
        "instruction_semantics_receipt",
        "instruction_semantics_receipt_sha256",
        "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256",
        "pair_id",
    }
    if not isinstance(row, Mapping) or set(row) != exact_fields:
        raise Evaluation400BridgeError("target manifest row schema changed")
    expected_role = {
        "adaptation": (
            "direct_actor_only_operational"
            if ordinal < 20
            else "adapter_development"
        ),
        "validation": "frozen_selection_validation",
        "evaluation": "sealed_paired_evaluation",
    }[split]
    instruction_sha = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
    semantic = row.get("instruction_semantics_receipt")
    if (
        row.get("task") != TASK
        or row.get("actor_id") != ACTOR_ID
        or row.get("target_body") != TARGET_BODY
        or row.get("split") != split
        or type(row.get("ordinal")) is not int
        or row.get("ordinal") != ordinal
        or type(row.get("global_ordinal")) is not int
        or row.get("global_ordinal") != global_ordinal
        or row.get("stage_role") != expected_role
        or type(row.get("requested_seed")) is not int
        or type(row.get("resolved_seed")) is not int
        or min(row["requested_seed"], row["resolved_seed"]) < 0
        or row.get("instruction") != INSTRUCTION
        or row.get("instruction_sha256") != instruction_sha
        or not isinstance(semantic, Mapping)
        or not is_sha(row.get("instruction_semantics_receipt_sha256"))
        or semantic.get("receipt_sha256")
        != row.get("instruction_semantics_receipt_sha256")
        or any(
            not is_sha(row.get(key))
            for key in (
                "initial_scene_state_sha256",
                "initial_measured_joint_state_sha256",
                "initial_commanded_drive_target_sha256",
                "pair_id",
            )
        )
        or row.get("pair_id") != canonical_sha256(_row_identity(row))
    ):
        raise Evaluation400BridgeError("target manifest row identity changed")
    return dict(row)


def validate_target_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest_sha = verify_signed(
        value, "seed_manifest_sha256", "target seed manifest"
    )
    expected_root = {
        "format",
        "status",
        "task",
        "actor_id",
        "source_body",
        "target_body",
        "purpose",
        "label_access_contract",
        "instruction_contract",
        "splits",
        "provenance",
        "d250_exclusion",
        "heldout_exclusion_attestation",
        "capability_receipt",
        "seed_manifest_sha256",
    }
    splits = value.get("splits")
    capability = value.get("capability_receipt")
    provenance = value.get("provenance")
    d250 = value.get("d250_exclusion")
    attestation = value.get("heldout_exclusion_attestation")
    expected_provenance = {
        "plan_file_sha256",
        "plan_sha256",
        "authorization_file_sha256",
        "authorization_sha256",
        "runtime_contract_sha256",
        "reset_receipt_file_sha256",
        "reset_receipt_sha256",
        "resolver_implementation_sha256",
        "reset_adapter_implementation_sha256",
    }
    if (
        set(value) != expected_root
        or value.get("format") != TARGET_MANIFEST_FORMAT
        or value.get("status") != TARGET_MANIFEST_STATUS
        or value.get("task") != TASK
        or value.get("actor_id") != ACTOR_ID
        or value.get("source_body") != SOURCE_BODY
        or value.get("target_body") != TARGET_BODY
        or value.get("purpose")
        != "nonfresh_development_only_no_confirmation_claim"
        or value.get("label_access_contract") != LABEL_ACCESS_CONTRACT
        or not isinstance(splits, Mapping)
        or set(splits) != {"adaptation", "validation", "evaluation"}
        or not isinstance(capability, Mapping)
        or dict(capability)
        != {
            "environment_reset_only": True,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "labels_or_outcomes_read": False,
            "policy_execution_authorized_by_manifest": False,
        }
        or not isinstance(provenance, Mapping)
        or set(provenance) != expected_provenance
        or any(not is_sha(item) for item in provenance.values())
        or not isinstance(attestation, Mapping)
        or set(attestation)
        != {
            "status",
            "heldout_identity_set_sha256",
            "target_identity_set_sha256",
            "intersection_count",
            "sensitive_identities_included",
            "attestation_sha256",
        }
        or not isinstance(d250, Mapping)
        or set(d250)
        != {
            "identity_manifest_file_sha256",
            "identity_sets_sha256",
            "intersection_count",
        }
        or not is_sha(d250.get("identity_manifest_file_sha256"))
        or not is_sha(d250.get("identity_sets_sha256"))
        or d250.get("intersection_count") != 0
    ):
        raise Evaluation400BridgeError("target manifest pre-outcome boundary changed")
    counts = {
        "adaptation": ADAPTATION_GROUPS,
        "validation": VALIDATION_GROUPS,
        "evaluation": EVALUATION_GROUPS,
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    requested: set[int] = set()
    resolved: set[int] = set()
    pair_ids: set[str] = set()
    global_offset = 0
    for split, count in counts.items():
        raw_rows = splits[split]
        if not isinstance(raw_rows, list) or len(raw_rows) != count:
            raise Evaluation400BridgeError("target manifest split count changed")
        rows[split] = []
        for ordinal, raw in enumerate(raw_rows):
            row = _validate_target_row(
                raw,
                split=split,
                ordinal=ordinal,
                global_ordinal=global_offset + ordinal,
            )
            if (
                row["requested_seed"] in requested
                or row["resolved_seed"] in resolved
                or row["pair_id"] in pair_ids
            ):
                raise Evaluation400BridgeError(
                    "target requested/resolved/pair identities are not unique"
                )
            requested.add(row["requested_seed"])
            resolved.add(row["resolved_seed"])
            pair_ids.add(row["pair_id"])
            rows[split].append(row)
        global_offset += count
    requested_order = [row["requested_seed"] for split in counts for row in rows[split]]
    resolved_order = [row["resolved_seed"] for split in counts for row in rows[split]]
    identity_set_sha = canonical_sha256(
        {"requested": requested_order, "resolved": resolved_order}
    )
    if (
        attestation.get("status") != ATTESTATION_STATUS
        or attestation.get("target_identity_set_sha256") != identity_set_sha
        or attestation.get("intersection_count") != 0
        or attestation.get("sensitive_identities_included") is not False
        or not is_sha(attestation.get("heldout_identity_set_sha256"))
        or not is_sha(attestation.get("attestation_sha256"))
    ):
        raise Evaluation400BridgeError("embedded selected-identity attestation changed")
    return {
        "manifest_sha256": manifest_sha,
        "identity_set_sha256": identity_set_sha,
        "heldout_identity_set_sha256": attestation[
            "heldout_identity_set_sha256"
        ],
        "attestation_sha256": attestation["attestation_sha256"],
        "target_reset_runtime_contract_sha256": provenance[
            "runtime_contract_sha256"
        ],
        "evaluation": rows["evaluation"],
    }


def validate_selected_identity_attestation(
    value: Mapping[str, Any], *, decoded_manifest: Mapping[str, Any]
) -> str:
    attestation_sha = verify_signed(
        value, "attestation_sha256", "selected identity attestation"
    )
    if dict(value) != {
        "format": ATTESTATION_FORMAT,
        "status": ATTESTATION_STATUS,
        "target_role": "selected_requested_and_resolved_target_identities",
        "heldout_identity_set_sha256": decoded_manifest[
            "heldout_identity_set_sha256"
        ],
        "target_identity_set_sha256": decoded_manifest["identity_set_sha256"],
        "intersection_count": 0,
        "sensitive_identities_included": False,
        "attestation_sha256": attestation_sha,
    }:
        raise Evaluation400BridgeError("selected identity attestation is invalid")
    if attestation_sha != decoded_manifest["attestation_sha256"]:
        raise Evaluation400BridgeError(
            "target manifest did not bind this selected identity attestation"
        )
    return attestation_sha


def validate_deployment_uncertainty_contract(
    value: Any, *, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Evaluation400BridgeError("deployment uncertainty contract is missing")
    exact_fields = {
        "format", "status", "shared_implementation_path",
        "shared_implementation_file_sha256",
        "performance_gate_uncertainty_source", "selector_uncertainty_source",
        "included_root_heads", "excluded_root_heads",
        "root_structured_uncertainty_head_count",
        "root_recovery_uncertainty_policy", "post_event_temperature",
        "next_event_temperature", "success_temperature",
        "conditional_recovery_temperature", "duration_scale_multiplier",
        "object_scale_multiplier", "object_error_robust_scale_m",
        "object_prediction_space",
        "duration_and_object_log_scale_multiplier_application",
        "shared_function_input_scale_state", "uncertainty_range",
        "formal_root_row_count", "evaluation400_outcomes_read",
        "deployment_uncertainty_contract_sha256",
    }
    logical = verify_signed(
        value,
        "deployment_uncertainty_contract_sha256",
        "deployment uncertainty contract",
    )
    actual_path = Path(deployment_uncertainty_v1.__file__).resolve()
    numeric_fields = {
        "post_event_temperature": ("post_event", "deployment_temperature"),
        "next_event_temperature": ("next_event", "deployment_temperature"),
        "success_temperature": ("success", "deployment_temperature"),
        "conditional_recovery_temperature": (
            "conditional_recovery", "deployment_temperature"
        ),
        "duration_scale_multiplier": (
            "duration_lognormal_mixture", "deployment_scale_multiplier"
        ),
        "object_scale_multiplier": (
            "object_total_variance", "deployment_scale_multiplier"
        ),
        "object_error_robust_scale_m": (
            "object_total_variance", "deployment_object_error_robust_scale_m"
        ),
    }
    if (
        set(value) != exact_fields
        or value.get("format") != deployment_uncertainty_v1.FORMAT
        or value.get("status")
        != "frozen_full_formal190_refit_parameters_online_reproducible"
        or value.get("shared_implementation_path") != str(actual_path)
        or value.get("shared_implementation_file_sha256")
        != file_sha256(actual_path)
        or value.get("performance_gate_uncertainty_source")
        != "five_fold_group_oof_predictions"
        or value.get("selector_uncertainty_source")
        != "full_formal190_refit_deployment_parameters"
        or value.get("included_root_heads")
        != list(deployment_uncertainty_v1.ROOT_INCLUDED_HEADS)
        or value.get("excluded_root_heads") != ["recovery"]
        or type(value.get("root_structured_uncertainty_head_count")) is not int
        or value.get("root_structured_uncertainty_head_count")
        != deployment_uncertainty_v1.ROOT_HEAD_COUNT
        or value.get("root_recovery_uncertainty_policy")
        != deployment_uncertainty_v1.ROOT_RECOVERY_UNCERTAINTY_POLICY
        or value.get("object_prediction_space") != "physical_xyz_m"
        or value.get("duration_and_object_log_scale_multiplier_application")
        != "add_log_multiplier_exactly_once"
        or value.get("shared_function_input_scale_state")
        != "duration_and_object_deployment_multiplier_already_applied_exactly_once"
        or value.get("uncertainty_range") != [0.0, 1.0]
        or type(value.get("formal_root_row_count")) is not int
        or value.get("formal_root_row_count", 0) <= 0
        or value.get("evaluation400_outcomes_read") is not False
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), (int, float))
            or not math.isfinite(float(value[field]))
            or float(value[field]) <= 0.0
            or float(value[field])
            != float(metrics.get(metric, {}).get(metric_field, -1.0))
            for field, (metric, metric_field) in numeric_fields.items()
        )
    ):
        raise Evaluation400BridgeError(
            "deployment uncertainty contract changed"
        )
    return {
        "deployment_uncertainty_contract_sha256": logical,
        "shared_implementation_path": str(actual_path),
        "shared_implementation_file_sha256": file_sha256(actual_path),
        "root_recovery_uncertainty_policy": value[
            "root_recovery_uncertainty_policy"
        ],
        "root_structured_uncertainty_head_count": value[
            "root_structured_uncertainty_head_count"
        ],
        "object_error_robust_scale_m": float(
            value["object_error_robust_scale_m"]
        ),
    }


def validate_root_selection_oof_evidence(
    root_ranker: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = root_ranker.get("selection_aware_oof_evidence")
    if not isinstance(evidence, Mapping):
        raise Evaluation400BridgeError("selection-aware root OOF evidence missing")
    logical = verify_signed(
        evidence,
        "selection_aware_oof_evidence_sha256",
        "selection-aware root OOF evidence",
    )
    if root_ranker.get("selection_aware_oof_evidence_sha256") != logical:
        raise Evaluation400BridgeError("selection-aware root OOF SHA changed")
    contract = evidence.get("root_outer_nesting_contract")
    if not isinstance(contract, Mapping):
        raise Evaluation400BridgeError("root outer-nesting contract missing")
    contract_logical = verify_signed(
        contract,
        "root_outer_nesting_contract_sha256",
        "root outer-nesting contract",
    )
    fold_parameters = contract.get("fold_parameters")
    outer_folds = evidence.get("outer_folds")
    decisions = evidence.get("stitched_group_decisions")
    fold_support = evidence.get("fold_support")
    gate_components = root_ranker.get("primary_gate_components")
    draws = evidence.get("shared_bootstrap_draws")
    if (
        evidence.get("format")
        != "etsf_smolvla_piper_formal190_root_selection_oof_evidence_v1"
        or evidence.get("status") != "passed_selection_aware_root_gate"
        or evidence.get("passed_for_primary") is not True
        or evidence.get("outer_crossfit_folds") != 5
        or evidence.get("outer_fold_assignment")
        != "lexicographic_logical_group_index_modulo_five"
        or evidence.get("root_selection_nested_within_outer_training_groups")
        is not True
        or evidence.get("global_abstain_threshold_nested_within_outer_training_groups")
        is not True
        or evidence.get("upstream_predictions_already_group_crossfit") is not True
        or evidence.get("complete_temperature_scale_and_root_double_nesting")
        is not True
        or evidence.get("formal_logical_group_count") != 190
        or evidence.get("stitched_decision_count") != 190
        or evidence.get("unique_stitched_logical_group_count") != 190
        or evidence.get("every_formal_logical_group_scored_exactly_once") is not True
        or evidence.get("root_outer_nesting_contract_sha256")
        != contract_logical
        or contract.get("format")
        != "etsf_smolvla_piper_formal190_complete_root_outer_nesting_v1"
        or contract.get("status") != "complete_outer_heldout_isolation"
        or contract.get("outer_crossfit_folds") != 5
        or contract.get("outer_heldout_labels_used_for_any_parameter_or_selection")
        is not False
        or contract.get("same_outer_training_parameters_used_for_training_and_heldout_inference")
        is not True
        or contract.get("object_robust_scale_fit_on_outer_training_groups_only")
        is not True
        or contract.get("complete_root_pipeline_outer_nesting") is not True
        or contract.get("upstream_predictions_already_group_crossfit") is not True
        or not isinstance(fold_parameters, list)
        or len(fold_parameters) != 5
        or not isinstance(outer_folds, list)
        or len(outer_folds) != 5
        or not isinstance(decisions, list)
        or len(decisions) != 190
        or not isinstance(fold_support, list)
        or len(fold_support) != 5
        or not isinstance(gate_components, Mapping)
        or gate_components.get(
            "all_six_heads_support_performance_uncertainty_gate_passed"
        )
        is not True
        or gate_components.get("full_formal190_deployment_candidate_available")
        is not True
        or gate_components.get("selection_aware_oof_evidence_passed") is not True
        or root_ranker.get(
            "primary_activation_requires_selection_aware_oof_evidence"
        )
        is not True
        or root_ranker.get("full_formal190_development_metrics_are_in_sample")
        is not True
        or root_ranker.get(
            "full_formal190_deployment_refit_candidate_available"
        )
        is not True
        or root_ranker.get("upstream_predictions_already_group_crossfit") is not True
        or root_ranker.get("complete_temperature_scale_and_root_double_nesting")
        is not True
    ):
        raise Evaluation400BridgeError("selection-aware root OOF gate changed")
    for fold, parameter in enumerate(fold_parameters):
        if (
            not isinstance(parameter, Mapping)
            or verify_signed(
                parameter,
                "fold_parameter_sha256",
                f"root outer fold {fold} parameters",
            )
            != parameter.get("fold_parameter_sha256")
            or type(parameter.get("outer_fold")) is not int
            or parameter.get("outer_fold") != fold
            or parameter.get("training_logical_group_count") != 152
            or parameter.get("heldout_logical_group_count") != 38
            or not is_sha(parameter.get("training_logical_group_ids_sha256"))
            or not is_sha(parameter.get("heldout_logical_group_ids_sha256"))
            or parameter.get("parameters_complete") is not True
            or parameter.get(
                "heldout_labels_used_for_parameters_or_training_quality"
            )
            is not False
            or any(
                isinstance(parameter.get(field), bool)
                or not isinstance(parameter.get(field), (int, float))
                or not math.isfinite(float(parameter[field]))
                or float(parameter[field]) <= 0.0
                for field in (
                    "post_event_temperature",
                    "next_event_temperature",
                    "success_temperature",
                    "duration_scale_multiplier",
                    "object_scale_multiplier",
                    "object_uncertainty_robust_scale_m",
                )
            )
        ):
            raise Evaluation400BridgeError("root outer-fold parameters changed")
    identifiers: list[str] = []
    paired_gains: list[int] = []
    changed_flags: list[bool] = []
    harmful_flags: list[bool] = []
    fold_counts = [0] * 5
    changed = helpful = harmful = selected_success = baseline_success = 0
    for decision in decisions:
        if (
            not isinstance(decision, Mapping)
            or type(decision.get("outer_fold")) is not int
            or not 0 <= decision["outer_fold"] < 5
            or not isinstance(decision.get("logical_group_id"), str)
            or not decision["logical_group_id"]
            or decision.get("selection_available") is not True
            or type(decision.get("changed_from_baseline")) is not bool
            or type(decision.get("baseline_final_success")) is not int
            or decision["baseline_final_success"] not in (0, 1)
            or type(decision.get("selected_final_success")) is not int
            or decision["selected_final_success"] not in (0, 1)
            or type(decision.get("paired_gain")) is not int
            or type(decision.get("selected_candidate_index")) is not int
            or decision["paired_gain"]
            != decision["selected_final_success"]
            - decision["baseline_final_success"]
            or (
                decision["changed_from_baseline"] is False
                and decision["paired_gain"] != 0
            )
        ):
            raise Evaluation400BridgeError("root OOF stitched decision changed")
        identifiers.append(decision["logical_group_id"])
        paired_gains.append(decision["paired_gain"])
        changed_flags.append(decision["changed_from_baseline"])
        harmful_flags.append(
            decision["changed_from_baseline"] and decision["paired_gain"] < 0
        )
        fold_counts[decision["outer_fold"]] += 1
        changed += int(decision["changed_from_baseline"])
        helpful += int(decision["paired_gain"] > 0)
        harmful += int(decision["paired_gain"] < 0)
        selected_success += decision["selected_final_success"]
        baseline_success += decision["baseline_final_success"]
    discordant = helpful + harmful
    if (
        len(set(identifiers)) != 190
        or identifiers != sorted(identifiers)
        or fold_counts != [38] * 5
        or evidence.get("changed_group_count") != changed
        or evidence.get("helpful_group_count") != helpful
        or evidence.get("harmful_group_count") != harmful
        or evidence.get("discordant_group_count") != discordant
        or evidence.get("selected_success_count") != selected_success
        or evidence.get("baseline_success_count") != baseline_success
        or changed < 50
        or discordant < 20
        or any(
            isinstance(evidence.get(field), bool)
            or not isinstance(evidence.get(field), (int, float))
            or not math.isfinite(float(evidence[field]))
            for field in (
                "change_coverage",
                "selected_success_rate",
                "baseline_success_rate",
                "paired_gain",
                "harmful_rate_among_executed_changes",
            )
        )
        or float(evidence.get("change_coverage", -1.0)) != changed / 190.0
        or float(evidence.get("selected_success_rate", -1.0))
        != selected_success / 190.0
        or float(evidence.get("baseline_success_rate", -1.0))
        != baseline_success / 190.0
        or float(evidence.get("paired_gain", -2.0))
        != (selected_success - baseline_success) / 190.0
        or float(evidence.get("harmful_rate_among_executed_changes", -1.0))
        != harmful / changed
        or evidence.get("minimum_changed_groups") != 50
        or evidence.get("minimum_discordant_groups") != 20
        or evidence.get("maximum_harmful_rate_among_executed_changes") != 0.10
        or evidence.get("paired_gain_lcb_must_be_strictly_positive") is not True
        or evidence.get("bootstrap_unit") != "logical_group"
        or evidence.get("bootstrap_seed") != 20260828
        or type(evidence.get("bootstrap_samples")) is not int
        or evidence.get("bootstrap_samples", 0) < 100
        or any(
            not isinstance(row, Mapping)
            or type(row.get("fold")) is not int
            or row.get("fold") != fold
            or row.get("logical_groups") != 38
            or row.get("support_passed") is not True
            for fold, row in enumerate(fold_support)
        )
        or not isinstance(draws, Mapping)
        or set(draws)
        != {
            "algorithm",
            "seed",
            "dtype",
            "shape",
            "draws_sha256",
            "descriptor_sha256",
        }
        or verify_signed(draws, "descriptor_sha256", "root bootstrap draws")
        != draws.get("descriptor_sha256")
        or draws.get("algorithm")
        != "numpy_pcg64_fixed_seed_logical_group_indices_v1"
        or draws.get("seed") != 20260828
        or draws.get("dtype") != "little_endian_uint16"
        or draws.get("shape") != [evidence.get("bootstrap_samples"), 190]
        or not is_sha(draws.get("draws_sha256"))
        or evidence.get("evaluation400_outcomes_read") is not False
    ):
        raise Evaluation400BridgeError("root OOF evidence arithmetic changed")
    for fold, outer in enumerate(outer_folds):
        if (
            not isinstance(outer, Mapping)
            or type(outer.get("outer_fold")) is not int
            or outer.get("outer_fold") != fold
            or outer.get("training_logical_group_count") != 152
            or outer.get("heldout_logical_group_count") != 38
            or outer.get("selection_available") is not True
            or outer.get("heldout_outcomes_used_for_training_selection") is not False
            or not isinstance(outer.get("selected_training_candidate"), Mapping)
            or any(
                isinstance(outer["selected_training_candidate"].get(field), bool)
                or not isinstance(
                    outer["selected_training_candidate"].get(field),
                    (int, float),
                )
                or not math.isfinite(
                    float(outer["selected_training_candidate"][field])
                )
                or float(outer["selected_training_candidate"][field]) < 0.0
                for field in (
                    "minimum_group_relative_composite_rank_score_margin",
                    "maximum_structured_pair_uncertainty",
                    "maximum_global_candidate_uncertainty",
                )
            )
            or not isinstance(outer.get("training_global_abstain_threshold"), Mapping)
            or outer["training_global_abstain_threshold"].get("enabled") is not True
            or not isinstance(outer.get("training_root_candidate_grid"), list)
            or len(outer["training_root_candidate_grid"]) != 20
            or outer.get("training_root_candidate_grid_sha256")
            != canonical_sha256(outer["training_root_candidate_grid"])
            or not isinstance(outer.get("heldout_decisions"), list)
            or len(outer["heldout_decisions"]) != 38
            or outer["heldout_decisions"]
            != [row for row in decisions if row["outer_fold"] == fold]
        ):
            raise Evaluation400BridgeError("root OOF outer fold changed")
    bootstrap_samples = evidence["bootstrap_samples"]
    regenerated_draws = np.random.default_rng(20260828).integers(
        0,
        190,
        size=(bootstrap_samples, 190),
        dtype=np.uint16,
    )
    regenerated_bytes = np.ascontiguousarray(
        regenerated_draws.astype("<u2", copy=False)
    ).tobytes(order="C")
    gain_array = np.asarray(paired_gains, dtype=np.float64)
    changed_array = np.asarray(changed_flags, dtype=bool)
    harmful_array = np.asarray(harmful_flags, dtype=bool)
    sampled_gain = gain_array[regenerated_draws].mean(axis=1)
    sampled_changed = changed_array[regenerated_draws].sum(axis=1)
    sampled_harmful = harmful_array[regenerated_draws].sum(axis=1)
    valid_harm = sampled_changed > 0
    regenerated_gain_lcb = float(np.quantile(sampled_gain, 0.05))
    regenerated_gain_ucb = float(np.quantile(sampled_gain, 0.95))
    regenerated_harmful_ucb = float(
        np.quantile(
            sampled_harmful[valid_harm] / sampled_changed[valid_harm],
            0.95,
        )
    )
    if (
        hashlib.sha256(regenerated_bytes).hexdigest()
        != draws["draws_sha256"]
        or float(evidence.get("paired_gain_group_bootstrap_lcb95", math.nan))
        != regenerated_gain_lcb
        or float(evidence.get("paired_gain_group_bootstrap_ucb95", math.nan))
        != regenerated_gain_ucb
        or float(evidence.get("harmful_rate_group_bootstrap_ucb95", math.nan))
        != regenerated_harmful_ucb
    ):
        raise Evaluation400BridgeError("root OOF bootstrap evidence changed")
    gain_lcb = evidence.get("paired_gain_group_bootstrap_lcb95")
    harmful_ucb = evidence.get("harmful_rate_group_bootstrap_ucb95")
    if (
        isinstance(gain_lcb, bool)
        or not isinstance(gain_lcb, (int, float))
        or not math.isfinite(float(gain_lcb))
        or float(gain_lcb) <= 0.0
        or isinstance(harmful_ucb, bool)
        or not isinstance(harmful_ucb, (int, float))
        or not math.isfinite(float(harmful_ucb))
        or float(harmful_ucb) > 0.10
    ):
        raise Evaluation400BridgeError("root OOF statistical gate changed")
    return {
        "selection_aware_oof_evidence_sha256": logical,
        "root_outer_nesting_contract_sha256": contract_logical,
    }


def validate_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = verify_signed(value, "calibration_sha256", "calibration")
    exact_fields = {
        "format",
        "status",
        "member_count",
        "metrics",
        "uncertainty_decomposition",
        "deployment_root_structured_uncertainty_contract",
        "deployment_uncertainty_contract_sha256",
        "head_enabled_for_primary",
        "head_performance_gate_protocol",
        "all_six_heads_support_performance_uncertainty_gate_passed",
        "prediction_contract",
        "success_temperature_fitted_on_validation_only",
        "recovery_temperature_fitted_on_validation_only",
        "duration_scale_fitted_by_group_crossfit",
        "object_scale_fitted_by_group_crossfit",
        "recovery_enters_primary_only_if_support_and_calibration_pass",
        "abstain_threshold",
        "root_group_ranker",
        "root_group_ranker_enabled_for_primary",
        "source_rank_numeric_contract",
        "source_rank_member_authority",
        "source_rank_member_authority_sha256",
        "validation_groups",
        "validation_samples",
        "test_artifacts_read",
        "test_hdf5_files_opened",
        "fresh_artifacts_read",
        "confirmation_artifacts_read",
        "paired_development_outcomes_read",
        "performance_claim_authorized",
        "calibration_sha256",
    }
    enabled = value.get("head_enabled_for_primary")
    abstain = value.get("abstain_threshold")
    metrics = value.get("metrics")
    uncertainty_contract = validate_deployment_uncertainty_contract(
        value.get("deployment_root_structured_uncertainty_contract"),
        metrics=metrics if isinstance(metrics, Mapping) else {},
    )
    root_ranker = value.get("root_group_ranker")
    root_oof = (
        validate_root_selection_oof_evidence(root_ranker)
        if isinstance(root_ranker, Mapping)
        else None
    )
    member_authority = validate_source_rank_member_authority(
        value.get("source_rank_member_authority"),
        value.get("source_rank_member_authority_sha256"),
        role="calibration source rank member authority",
    )
    selected_ranker = (
        root_ranker.get("selected_candidate")
        if isinstance(root_ranker, Mapping) else None
    )
    threshold = abstain.get("maximum_total_uncertainty") if isinstance(abstain, Mapping) else None
    minimum_retained = abstain.get("minimum_retained_groups") if isinstance(abstain, Mapping) else None
    if (
        set(value) != exact_fields
        or value.get("format") != CALIBRATION_FORMAT
        or value.get("status") != CALIBRATION_STATUS
        or value.get("member_count") != MEMBER_COUNT
        or not isinstance(enabled, Mapping)
        or any(enabled.get(name) is not True for name in CORE_HEADS)
        or value.get("all_six_heads_support_performance_uncertainty_gate_passed")
        is not True
        or value.get("head_performance_gate_protocol")
        != "five_fold_logical_group_crossfit_group_bootstrap_zero_gain_lcb_v1"
        or not isinstance(metrics, Mapping)
        or set(metrics)
        != {
            "post_event", "next_event", "success", "conditional_recovery",
            "duration_lognormal_mixture", "object_total_variance",
        }
        or any(
            not isinstance(metrics.get(name), Mapping)
            or metrics[name].get("crossfit_folds") != 5
            or metrics[name].get("crossfit_complete") is not True
            or metrics[name].get("performance_gate_passed") is not True
            or metrics[name].get("metric_weighting") != "equal_logical_group"
            or not isinstance(metrics[name].get("uncertainty_gate"), Mapping)
            or metrics[name]["uncertainty_gate"].get("passed") is not True
            for name in metrics
        )
        or not isinstance(root_ranker, Mapping)
        or value.get("deployment_uncertainty_contract_sha256")
        != uncertainty_contract["deployment_uncertainty_contract_sha256"]
        or value.get("root_group_ranker_enabled_for_primary") is not True
        or root_ranker.get("enabled_for_primary") is not True
        or root_ranker.get("status") != "enabled_source_composite_primary_ranker"
        or root_ranker.get("formal_logical_group_count") != 190
        or root_ranker.get("member_count") != MEMBER_COUNT
        or root_ranker.get("score_is_success_logit") is not False
        or root_ranker.get("score_is_success_probability") is not False
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or root_ranker.get("source_rank_numeric_contract")
        != value.get("source_rank_numeric_contract")
        or root_ranker.get("source_rank_member_authority")
        != member_authority["source_rank_member_authority"]
        or root_ranker.get("source_rank_member_authority_sha256")
        != member_authority["source_rank_member_authority_sha256"]
        or root_ranker.get("factual_success_head_used_only_for_independent_six_head_calibration")
        is not True
        or root_ranker.get("root_recovery_uncertainty_policy")
        != uncertainty_contract["root_recovery_uncertainty_policy"]
        or root_ranker.get("root_structured_uncertainty_head_count")
        != uncertainty_contract["root_structured_uncertainty_head_count"]
        or root_ranker.get("deployment_uncertainty_contract_sha256")
        != uncertainty_contract["deployment_uncertainty_contract_sha256"]
        or root_ranker.get("shared_uncertainty_implementation_path")
        != uncertainty_contract["shared_implementation_path"]
        or root_ranker.get("shared_uncertainty_implementation_file_sha256")
        != uncertainty_contract["shared_implementation_file_sha256"]
        or not isinstance(root_oof, Mapping)
        or not isinstance(root_ranker.get("groups"), list)
        or len(root_ranker["groups"]) != 190
        or not isinstance(root_ranker.get("candidate_grid"), list)
        or len(root_ranker["candidate_grid"]) != 20
        or not isinstance(selected_ranker, Mapping)
        or type(selected_ranker.get("changed_group_count")) is not int
        or selected_ranker["changed_group_count"] < 50
        or type(selected_ranker.get("discordant_group_count")) is not int
        or selected_ranker["discordant_group_count"] < 20
        or isinstance(
            selected_ranker.get("paired_gain_group_bootstrap_lcb95"), bool
        )
        or not isinstance(
            selected_ranker.get("paired_gain_group_bootstrap_lcb95"),
            (int, float),
        )
        or not math.isfinite(
            float(selected_ranker["paired_gain_group_bootstrap_lcb95"])
        )
        or float(selected_ranker["paired_gain_group_bootstrap_lcb95"]) <= 0.0
        or isinstance(
            selected_ranker.get("harmful_rate_group_bootstrap_ucb95"), bool
        )
        or not isinstance(
            selected_ranker.get("harmful_rate_group_bootstrap_ucb95"),
            (int, float),
        )
        or not math.isfinite(
            float(selected_ranker["harmful_rate_group_bootstrap_ucb95"])
        )
        or float(selected_ranker["harmful_rate_group_bootstrap_ucb95"]) > 0.10
        or not isinstance(selected_ranker.get("fold_support"), list)
        or len(selected_ranker["fold_support"]) != 5
        or any(
            not isinstance(row, Mapping)
            or type(row.get("fold")) is not int
            or row.get("fold") != index
            or row.get("support_passed") is not True
            for index, row in enumerate(selected_ranker["fold_support"])
        )
        or root_ranker.get("maximum_harmful_rate_among_executed_changes")
        != 0.10
        or root_ranker.get("paired_gain_lcb_must_be_strictly_positive") is not True
        or root_ranker.get(
            "zero_gain_lcb_authorizes_only_noninferiority_not_primary"
        ) is not True
        or not is_sha(root_ranker.get("root_group_ranker_sha256"))
        or verify_signed(
            root_ranker, "root_group_ranker_sha256", "root group ranker"
        ) != root_ranker.get("root_group_ranker_sha256")
        or not isinstance(abstain, Mapping)
        or abstain.get("enabled") is not True
        or abstain.get("status") != "frozen_validation_group_bootstrap_lcb"
        or not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or float(threshold) < 0
        or type(minimum_retained) is not int
        or minimum_retained < 50
        or abstain.get("test_or_paired_outcomes_used_for_selection") is not False
        or value.get("test_artifacts_read") is not False
        or value.get("test_hdf5_files_opened") != 0
        or value.get("fresh_artifacts_read") is not False
        or value.get("confirmation_artifacts_read") is not False
        or value.get("paired_development_outcomes_read") is not False
        or value.get("performance_claim_authorized") is not False
    ):
        raise Evaluation400BridgeError("calibration/abstention boundary is invalid")
    if (
        isinstance(selected_ranker, Mapping)
        and (
            float(selected_ranker.get("maximum_global_candidate_uncertainty", -1.0))
            != float(threshold)
            or isinstance(
                selected_ranker.get("minimum_group_relative_composite_rank_score_margin"),
                bool,
            )
            or not isinstance(
                selected_ranker.get("minimum_group_relative_composite_rank_score_margin"),
                (int, float),
            )
            or not math.isfinite(float(
                selected_ranker["minimum_group_relative_composite_rank_score_margin"]
            ))
            or float(
                selected_ranker["minimum_group_relative_composite_rank_score_margin"]
            ) < 0.0
            or isinstance(
                selected_ranker.get("maximum_structured_pair_uncertainty"), bool
            )
            or not isinstance(
                selected_ranker.get("maximum_structured_pair_uncertainty"),
                (int, float),
            )
            or not math.isfinite(float(
                selected_ranker["maximum_structured_pair_uncertainty"]
            ))
            or float(selected_ranker["maximum_structured_pair_uncertainty"]) < 0.0
        )
    ):
        raise Evaluation400BridgeError("root selector thresholds changed")
    return {
        "calibration_sha256": logical,
        "head_enabled_for_primary": dict(enabled),
        "metrics": dict(metrics),
        "maximum_total_uncertainty": float(
            abstain["maximum_total_uncertainty"]
        ),
        "abstention_contract_sha256": canonical_sha256(abstain),
        "root_group_ranker_sha256": root_ranker[
            "root_group_ranker_sha256"
        ],
        "root_group_ranker": dict(root_ranker),
        "source_rank_numeric_contract": value[
            "source_rank_numeric_contract"
        ],
        **member_authority,
        "minimum_group_relative_composite_rank_score_margin": float(
            selected_ranker[
                "minimum_group_relative_composite_rank_score_margin"
            ]
        ),
        "maximum_structured_pair_uncertainty": float(
            selected_ranker["maximum_structured_pair_uncertainty"]
        ),
        "deployment_uncertainty_contract_sha256": uncertainty_contract[
            "deployment_uncertainty_contract_sha256"
        ],
        "deployment_uncertainty_contract": dict(
            value["deployment_root_structured_uncertainty_contract"]
        ),
        "shared_uncertainty_implementation_path": uncertainty_contract[
            "shared_implementation_path"
        ],
        "shared_uncertainty_implementation_file_sha256": uncertainty_contract[
            "shared_implementation_file_sha256"
        ],
        "root_recovery_uncertainty_policy": uncertainty_contract[
            "root_recovery_uncertainty_policy"
        ],
        "root_structured_uncertainty_head_count": uncertainty_contract[
            "root_structured_uncertainty_head_count"
        ],
        "object_error_robust_scale_m": uncertainty_contract[
            "object_error_robust_scale_m"
        ],
        "deployment_parameters": {
            "post_event_temperature": float(
                metrics["post_event"]["deployment_temperature"]
            ),
            "next_event_temperature": float(
                metrics["next_event"]["deployment_temperature"]
            ),
            "success_temperature": float(
                metrics["success"]["deployment_temperature"]
            ),
            "conditional_recovery_temperature": float(
                metrics["conditional_recovery"]["deployment_temperature"]
            ),
            "duration_scale_multiplier": float(
                metrics["duration_lognormal_mixture"][
                    "deployment_scale_multiplier"
                ]
            ),
            "object_scale_multiplier": float(
                metrics["object_total_variance"][
                    "deployment_scale_multiplier"
                ]
            ),
            "object_error_robust_scale_m": uncertainty_contract[
                "object_error_robust_scale_m"
            ],
        },
    }


def validate_head_support(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = verify_signed(value, "head_support_sha256", "head support")
    heads = value.get("heads")
    expected = {
        "post_event",
        "next_event",
        "duration",
        "success",
        "recovery",
        "object_effect",
    }
    if (
        set(value)
        != {
            "format",
            "status",
            "heads",
            "paired_development_outcomes_read",
            "sealed_evaluation_reserve_outcomes_read",
            "head_support_sha256",
        }
        or value.get("format") != HEAD_SUPPORT_FORMAT
        or value.get("status") != HEAD_SUPPORT_STATUS
        or not isinstance(heads, Mapping)
        or set(heads) != expected
        or any(
            not isinstance(heads.get(name), Mapping)
            or heads[name].get("enabled_for_primary") is not True
            for name in CORE_HEADS
        )
        or value.get("paired_development_outcomes_read") is not False
        or value.get("sealed_evaluation_reserve_outcomes_read") is not False
    ):
        raise Evaluation400BridgeError("head-support boundary is invalid")
    expected_head_fields = {
        "enabled_for_primary",
        "support_threshold_met",
        "performance_gate_passed",
        "uncertainty_gate_passed",
        "independent_positive_or_observed_groups",
        "independent_negative_or_censored_groups",
        "minimum_required_per_side",
        "support_source",
    }
    for name, row in heads.items():
        positive = row.get("independent_positive_or_observed_groups")
        negative = row.get("independent_negative_or_censored_groups")
        minimum = row.get("minimum_required_per_side")
        expected_fields = (
            expected_head_fields
            | {"all_member_recovery_heads_trained"}
            if name == "recovery"
            else expected_head_fields
        )
        if (
            set(row) != expected_fields
            or type(positive) is not int
            or type(negative) is not int
            or type(minimum) is not int
            or min(positive, negative, minimum) < 0
        ):
            raise Evaluation400BridgeError(
                f"head-support counts are invalid: {name}"
            )
        support_met = positive >= minimum and negative >= minimum
        expected_enabled = (
            support_met
            and row.get("support_threshold_met") is True
            and row.get("performance_gate_passed") is True
            and row.get("uncertainty_gate_passed") is True
            and row.get("all_member_recovery_heads_trained") is True
            if name == "recovery"
            else (
                support_met
                and row.get("support_threshold_met") is True
                and row.get("performance_gate_passed") is True
                and row.get("uncertainty_gate_passed") is True
            )
        )
        if (
            row["enabled_for_primary"] is not expected_enabled
            or (
                row["support_threshold_met"] is not support_met
            )
            or not isinstance(row["support_source"], str)
            or not row["support_source"]
        ):
            raise Evaluation400BridgeError(
                f"head-support counts are invalid: {name}"
            )
    return {"head_support_sha256": logical, "heads": dict(heads)}


def validate_ensemble_manifest(
    value: Mapping[str, Any], *, calibration: Mapping[str, Any], head: Mapping[str, Any]
) -> dict[str, Any]:
    logical = verify_signed(
        value, "ensemble_manifest_sha256", "ensemble manifest"
    )
    members = value.get("members")
    enabled = value.get("head_enabled_for_primary")
    exact_fields = {
        "format",
        "status",
        "member_count",
        "members",
        "shared_contract",
        "prediction_contract",
        "deployment_root_structured_uncertainty_contract",
        "deployment_uncertainty_contract_sha256",
        "post_event_temperature",
        "next_event_temperature",
        "success_temperature",
        "conditional_recovery_temperature",
        "duration_scale_multiplier",
        "object_scale_multiplier",
        "object_error_robust_scale_m",
        "conditional_recovery_semantics",
        "conditional_recovery_activation_requires_observed_regress",
        "head_enabled_for_primary",
        "all_six_heads_support_performance_uncertainty_gate_passed",
        "root_group_ranker",
        "source_rank_numeric_contract",
        "source_rank_member_authority",
        "source_rank_member_authority_sha256",
        "maximum_total_uncertainty",
        "abstain_threshold_enabled",
        "calibration_sha256",
        "head_support_sha256",
        "root_group_ranker_path",
        "root_group_ranker_file_sha256",
        "root_group_ranker_sha256",
        "root_group_ranker_enabled_for_primary",
        "validation_identity_set_sha256",
        "test_artifacts_read",
        "test_hdf5_files_opened",
        "fresh_artifacts_read",
        "confirmation_artifacts_read",
        "paired_development_outcomes_read",
        "ensemble_manifest_sha256",
    }
    ensemble_threshold = value.get("maximum_total_uncertainty")
    member_authority = validate_source_rank_member_authority(
        value.get("source_rank_member_authority"),
        value.get("source_rank_member_authority_sha256"),
        role="ensemble source rank member authority",
    )
    if (
        set(value) != exact_fields
        or value.get("format") != ENSEMBLE_FORMAT
        or value.get("status") != ENSEMBLE_STATUS
        or value.get("member_count") != MEMBER_COUNT
        or not isinstance(members, list)
        or len(members) != MEMBER_COUNT
        or [row.get("member_index") for row in members if isinstance(row, Mapping)]
        != list(range(MEMBER_COUNT))
        or len(
            {
                row.get("checkpoint_file_sha256")
                for row in members
                if isinstance(row, Mapping)
            }
        )
        != MEMBER_COUNT
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "member_index",
                "member_seed",
                "checkpoint_path",
                "checkpoint_file_sha256",
                "source_rank_score_contract",
                "source_rank_score_contract_sha256",
            }
            or not is_sha(row.get("checkpoint_file_sha256"))
            or not isinstance(row.get("source_rank_score_contract"), Mapping)
            or row["source_rank_score_contract"].get(
                "source_contract_rank_score_is_success_logit"
            ) is not False
            or row["source_rank_score_contract"].get(
                "source_contract_rank_score_is_success_probability"
            ) is not False
            or not is_sha(row.get("source_rank_score_contract_sha256"))
            or row.get("source_rank_score_contract_sha256")
            != row["source_rank_score_contract"].get("contract_sha256")
            or verify_signed(
                row["source_rank_score_contract"],
                "contract_sha256",
                "source rank score contract",
            )
            != row.get("source_rank_score_contract_sha256")
            or isinstance(
                row["source_rank_score_contract"].get("success_temperature"),
                bool,
            )
            or not isinstance(
                row["source_rank_score_contract"].get("success_temperature"),
                (int, float),
            )
            or not math.isfinite(float(
                row["source_rank_score_contract"]["success_temperature"]
            ))
            or float(
                row["source_rank_score_contract"]["success_temperature"]
            ) <= 0.0
            for row in members
        )
        or not isinstance(enabled, Mapping)
        or dict(enabled) != calibration["head_enabled_for_primary"]
        or value.get("calibration_sha256")
        != calibration["calibration_sha256"]
        or value.get("head_support_sha256") != head["head_support_sha256"]
        or value.get("deployment_uncertainty_contract_sha256")
        != calibration["deployment_uncertainty_contract_sha256"]
        or value.get("deployment_root_structured_uncertainty_contract")
        != calibration["deployment_uncertainty_contract"]
        or value.get("all_six_heads_support_performance_uncertainty_gate_passed")
        is not True
        or value.get("root_group_ranker_enabled_for_primary") is not True
        or value.get("root_group_ranker_sha256")
        != calibration["root_group_ranker_sha256"]
        or value.get("source_rank_numeric_contract")
        != calibration["source_rank_numeric_contract"]
        or member_authority["source_rank_member_authority"]
        != calibration["source_rank_member_authority"]
        or member_authority["source_rank_member_authority_sha256"]
        != calibration["source_rank_member_authority_sha256"]
        or any(
            authority_member["member_index"] != index
            or authority_member["source_checkpoint_file_sha256"]
            != members[index]["source_rank_score_contract"].get(
                "source_checkpoint_file_sha256"
            )
            or authority_member["source_rank_score_contract_sha256"]
            != members[index]["source_rank_score_contract_sha256"]
            or float(authority_member["success_temperature"])
            != float(members[index]["source_rank_score_contract"].get(
                "success_temperature", -1.0
            ))
            for index, authority_member in enumerate(
                member_authority["source_rank_member_authority"]["members"]
            )
        )
        or not isinstance(value.get("root_group_ranker"), Mapping)
        or value["root_group_ranker"].get("logical_sha256")
        != calibration["root_group_ranker_sha256"]
        or value["root_group_ranker"].get("enabled_for_primary") is not True
        or value["root_group_ranker"].get("path")
        != value.get("root_group_ranker_path")
        or value["root_group_ranker"].get("file_sha256")
        != value.get("root_group_ranker_file_sha256")
        or not is_sha(value.get("root_group_ranker_file_sha256"))
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), (int, float))
            or not math.isfinite(float(value[name]))
            or float(value[name]) <= 0.0
            for name in (
                "post_event_temperature", "next_event_temperature",
                "success_temperature", "conditional_recovery_temperature",
                "duration_scale_multiplier", "object_scale_multiplier",
                "object_error_robust_scale_m",
            )
        )
        or float(value["object_error_robust_scale_m"])
        != float(
            calibration.get("metrics", {})
            .get("object_total_variance", {})
            .get("deployment_object_error_robust_scale_m", -1.0)
        )
        or float(value["post_event_temperature"])
        != float(calibration["metrics"]["post_event"]["deployment_temperature"])
        or float(value["next_event_temperature"])
        != float(calibration["metrics"]["next_event"]["deployment_temperature"])
        or float(value["success_temperature"])
        != float(calibration["metrics"]["success"]["deployment_temperature"])
        or float(value["conditional_recovery_temperature"])
        != float(
            calibration["metrics"]["conditional_recovery"][
                "deployment_temperature"
            ]
        )
        or float(value["duration_scale_multiplier"])
        != float(
            calibration["metrics"]["duration_lognormal_mixture"][
                "deployment_scale_multiplier"
            ]
        )
        or float(value["object_scale_multiplier"])
        != float(
            calibration["metrics"]["object_total_variance"][
                "deployment_scale_multiplier"
            ]
        )
        or value.get("abstain_threshold_enabled") is not True
        or not isinstance(ensemble_threshold, (int, float))
        or isinstance(ensemble_threshold, bool)
        or float(ensemble_threshold) != calibration["maximum_total_uncertainty"]
        or not is_sha(value.get("validation_identity_set_sha256"))
        or value.get("test_artifacts_read") is not False
        or value.get("test_hdf5_files_opened") != 0
        or value.get("fresh_artifacts_read") is not False
        or value.get("confirmation_artifacts_read") is not False
        or value.get("paired_development_outcomes_read") is not False
    ):
        raise Evaluation400BridgeError("five-member ensemble boundary is invalid")
    return {
        "ensemble_manifest_sha256": logical,
        "member_count": MEMBER_COUNT,
        "member_checkpoint_sha256": [
            row["checkpoint_file_sha256"] for row in members
        ],
        "validation_identity_set_sha256": value[
            "validation_identity_set_sha256"
        ],
        "root_group_ranker_sha256": value["root_group_ranker_sha256"],
        "root_group_ranker": dict(value["root_group_ranker"]),
        "source_rank_score_contract_sha256": [
            row["source_rank_score_contract_sha256"] for row in members
        ],
        "source_rank_score_contracts": [
            dict(row["source_rank_score_contract"]) for row in members
        ],
        "source_rank_numeric_contract": value[
            "source_rank_numeric_contract"
        ],
        **member_authority,
        "deployment_uncertainty_contract_sha256": value[
            "deployment_uncertainty_contract_sha256"
        ],
    }


def validate_calibration_receipt(
    value: Mapping[str, Any], *, paths: Mapping[str, Path], files: Mapping[str, str],
    calibration: Mapping[str, Any], head: Mapping[str, Any], ensemble: Mapping[str, Any]
) -> str:
    logical = verify_signed(value, "receipt_sha256", "calibration receipt")
    exact_fields = {
        "format",
        "status",
        "input_authority_path",
        "input_authority_file_sha256",
        "input_authority_sha256",
        "member_count",
        "validation_only",
        "shared_contract",
        "prediction_contract_sha256",
        "calibration_path",
        "calibration_file_sha256",
        "calibration_sha256",
        "head_support_path",
        "head_support_file_sha256",
        "head_support_sha256",
        "root_group_ranker_path",
        "root_group_ranker_file_sha256",
        "root_group_ranker_sha256",
        "root_group_ranker_enabled_for_primary",
        "source_rank_numeric_contract",
        "source_rank_member_authority",
        "source_rank_member_authority_sha256",
        "deployment_uncertainty_contract_sha256",
        "ensemble_manifest_path",
        "ensemble_manifest_file_sha256",
        "ensemble_manifest_sha256",
        "abstain_threshold_enabled",
        "test_artifacts_read",
        "test_hdf5_files_opened",
        "fresh_paths_accepted",
        "confirmation_artifacts_read",
        "paired_development_outcomes_read",
        "performance_or_transfer_claim_authorized",
        "artifacts_frozen_read_only",
        "receipt_sha256",
    }
    member_authority = validate_source_rank_member_authority(
        value.get("source_rank_member_authority"),
        value.get("source_rank_member_authority_sha256"),
        role="calibration receipt source rank member authority",
    )
    if (
        set(value) != exact_fields
        or value.get("format") != CALIBRATION_RECEIPT_FORMAT
        or value.get("status") != CALIBRATION_RECEIPT_STATUS
        or value.get("member_count") != MEMBER_COUNT
        or value.get("validation_only") is not True
        or not isinstance(value.get("shared_contract"), Mapping)
        or not is_sha(value.get("input_authority_file_sha256"))
        or not is_sha(value.get("input_authority_sha256"))
        or not is_sha(value.get("prediction_contract_sha256"))
        or value.get("calibration_path") != str(paths["calibration"])
        or value.get("calibration_file_sha256") != files["calibration"]
        or value.get("calibration_sha256")
        != calibration["calibration_sha256"]
        or value.get("head_support_path") != str(paths["head_support"])
        or value.get("head_support_file_sha256") != files["head_support"]
        or value.get("head_support_sha256") != head["head_support_sha256"]
        or value.get("root_group_ranker_enabled_for_primary") is not True
        or value.get("root_group_ranker_sha256")
        != calibration["root_group_ranker_sha256"]
        or value.get("source_rank_numeric_contract")
        != calibration["source_rank_numeric_contract"]
        or value.get("source_rank_numeric_contract")
        != ensemble["source_rank_numeric_contract"]
        or member_authority["source_rank_member_authority"]
        != calibration["source_rank_member_authority"]
        or member_authority["source_rank_member_authority"]
        != ensemble["source_rank_member_authority"]
        or member_authority["source_rank_member_authority_sha256"]
        != calibration["source_rank_member_authority_sha256"]
        or member_authority["source_rank_member_authority_sha256"]
        != ensemble["source_rank_member_authority_sha256"]
        or value.get("deployment_uncertainty_contract_sha256")
        != calibration["deployment_uncertainty_contract_sha256"]
        or not is_sha(value.get("root_group_ranker_file_sha256"))
        or value.get("ensemble_manifest_path") != str(paths["ensemble_manifest"])
        or value.get("ensemble_manifest_file_sha256")
        != files["ensemble_manifest"]
        or value.get("ensemble_manifest_sha256")
        != ensemble["ensemble_manifest_sha256"]
        or value.get("abstain_threshold_enabled") is not True
        or value.get("test_artifacts_read") is not False
        or value.get("test_hdf5_files_opened") != 0
        or value.get("fresh_paths_accepted") is not False
        or value.get("confirmation_artifacts_read") is not False
        or value.get("paired_development_outcomes_read") is not False
        or value.get("performance_or_transfer_claim_authorized") is not False
        or value.get("artifacts_frozen_read_only") is not True
    ):
        raise Evaluation400BridgeError("calibration terminal receipt is invalid")
    root_path, root_value = _read_bound_json(
        Path(str(value["root_group_ranker_path"])),
        str(value["root_group_ranker_file_sha256"]),
        "formal190 root group ranker",
    )
    root_logical = verify_signed(
        root_value, "root_group_ranker_sha256", "formal190 root group ranker"
    )
    manifest_root = ensemble.get("root_group_ranker")
    if (
        root_logical != calibration["root_group_ranker_sha256"]
        or root_value != calibration.get("root_group_ranker")
        or not isinstance(manifest_root, Mapping)
        or manifest_root.get("path") != str(root_path)
        or manifest_root.get("file_sha256")
        != value["root_group_ranker_file_sha256"]
        or manifest_root.get("logical_sha256") != root_logical
    ):
        raise Evaluation400BridgeError("formal190 root ranker file binding changed")
    return logical


def validate_policy_bridge(value: Mapping[str, Any]) -> dict[str, Any]:
    base_fields = {
        "status",
        "policy",
        "checkpoint_family",
        "bridge_contract_sha256",
        "runtime_binding_sha256",
        "state_feature_source_sha256",
        "state_feature_dimension",
        "state_feature_binding_sha256",
        "action_mapping",
        "action_mapping_binding_sha256",
        "policy_row",
        "canonical_event_interface",
        "canonical_action_effect_interface",
        "cross_policy_latent_reuse_allowed",
        "verification_sha256",
    }
    optional = {"checkpoint_file_sha256", "runtime_binding_file_sha256"}
    if set(value) - base_fields - optional:
        raise Evaluation400BridgeError("policy bridge receipt schema changed")
    signed_payload = {key: value[key] for key in base_fields}
    logical = verify_signed(
        signed_payload, "verification_sha256", "policy bridge verification"
    )
    if (
        value.get("status") != "verified_exact_policy_feature_action_bridge"
        or value.get("policy") != "smolvla"
        or value.get("checkpoint_family") != "smolvla_native_event_world_model"
        or value.get("state_feature_dimension") != 960
        or value.get("action_mapping") != ACTION_MAPPING
        or value.get("canonical_event_interface")
        != "canonical_event_id_and_reversible_predicates_v1"
        or value.get("canonical_action_effect_interface")
        != "masked_canonical_action_chunk_and_feature_validity_v1"
        or value.get("cross_policy_latent_reuse_allowed") is not False
        or any(
            not is_sha(value.get(key))
            for key in (
                "bridge_contract_sha256",
                "runtime_binding_sha256",
                "state_feature_source_sha256",
                "state_feature_binding_sha256",
                "action_mapping_binding_sha256",
            )
        )
        or any(not is_sha(value.get(key)) for key in optional if key in value)
    ):
        raise Evaluation400BridgeError("policy/runtime/action bridge is invalid")
    return {
        "verification_sha256": logical,
        "bridge_contract_sha256": value["bridge_contract_sha256"],
        "runtime_binding_sha256": value["runtime_binding_sha256"],
        "action_mapping_binding_sha256": value[
            "action_mapping_binding_sha256"
        ],
        "state_feature_binding_sha256": value["state_feature_binding_sha256"],
    }


def paired_condition_order(pair_id: str) -> list[str]:
    if not is_sha(pair_id):
        raise Evaluation400BridgeError("pair_id must be a lowercase SHA256")
    digest = hashlib.sha256(
        f"{CONDITION_ORDER_NAMESPACE}:{pair_id}".encode("ascii")
    ).digest()
    return ["baseline", "etsf"] if digest[0] & 1 == 0 else ["etsf", "baseline"]


def _build_pairs(
    evaluation_rows: Sequence[Mapping[str, Any]], *, deployment_sha256: str
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for ordinal, row in enumerate(evaluation_rows):
        pair_id = str(row["pair_id"])
        pairs.append(
            {
                "ordinal": ordinal,
                "pair_id": pair_id,
                "target_manifest_split": "evaluation",
                "target_manifest_global_ordinal": row["global_ordinal"],
                "requested_seed": row["requested_seed"],
                "resolved_seed": row["resolved_seed"],
                "instruction_sha256": row["instruction_sha256"],
                "initial_scene_state_sha256": row[
                    "initial_scene_state_sha256"
                ],
                "initial_measured_joint_state_sha256": row[
                    "initial_measured_joint_state_sha256"
                ],
                "initial_commanded_drive_target_sha256": row[
                    "initial_commanded_drive_target_sha256"
                ],
                "same_initial_state_for_both_conditions": True,
                "exact_reset_identity_reverification_before_each_condition": True,
                "condition_order": paired_condition_order(pair_id),
                "baseline_condition": {
                    "condition_id": "baseline",
                    "selector": "lowest_legal_feasibility_root_candidate",
                    "candidate_count": CANDIDATE_COUNT,
                },
                "etsf_condition": {
                    "condition_id": "etsf",
                    "selector": "frozen_five_member_event_world_model_with_uncertainty_abstention",
                    "candidate_count": CANDIDATE_COUNT,
                    "deployment_binding_sha256": deployment_sha256,
                    "fallback": "baseline",
                },
                "outcome_or_trajectory_fields_present": False,
            }
        )
    return pairs


def freeze_bridge(
    *,
    target_manifest_path: Path,
    target_manifest_file_sha256: str,
    selected_identity_attestation_path: Path,
    selected_identity_attestation_file_sha256: str,
    ensemble_manifest_path: Path,
    ensemble_manifest_file_sha256: str,
    calibration_path: Path,
    calibration_file_sha256: str,
    head_support_path: Path,
    head_support_file_sha256: str,
    calibration_receipt_path: Path,
    calibration_receipt_file_sha256: str,
    policy_bridge_receipt_path: Path,
    policy_bridge_receipt_file_sha256: str,
) -> dict[str, Any]:
    raw_paths = {
        "target_manifest": target_manifest_path,
        "selected_identity_attestation": selected_identity_attestation_path,
        "ensemble_manifest": ensemble_manifest_path,
        "calibration": calibration_path,
        "head_support": head_support_path,
        "calibration_receipt": calibration_receipt_path,
        "policy_bridge_receipt": policy_bridge_receipt_path,
    }
    file_shas = {
        "target_manifest": target_manifest_file_sha256,
        "selected_identity_attestation": selected_identity_attestation_file_sha256,
        "ensemble_manifest": ensemble_manifest_file_sha256,
        "calibration": calibration_file_sha256,
        "head_support": head_support_file_sha256,
        "calibration_receipt": calibration_receipt_file_sha256,
        "policy_bridge_receipt": policy_bridge_receipt_file_sha256,
    }
    # Reject every path/SHA before the first file is opened.
    for role, raw in raw_paths.items():
        _safe_json_path(raw, role, must_exist=True)
        if not is_sha(file_shas[role]):
            raise Evaluation400BridgeError(f"{role} expected file SHA is invalid")
    paths: dict[str, Path] = {}
    values: dict[str, dict[str, Any]] = {}
    for role, raw in raw_paths.items():
        paths[role], values[role] = _read_bound_json(
            raw, file_shas[role], role
        )

    target = validate_target_manifest(values["target_manifest"])
    attestation_sha = validate_selected_identity_attestation(
        values["selected_identity_attestation"], decoded_manifest=target
    )
    calibration = validate_calibration(values["calibration"])
    head = validate_head_support(values["head_support"])
    ensemble = validate_ensemble_manifest(
        values["ensemble_manifest"], calibration=calibration, head=head
    )
    calibration_receipt_sha = validate_calibration_receipt(
        values["calibration_receipt"],
        paths=paths,
        files=file_shas,
        calibration=calibration,
        head=head,
        ensemble=ensemble,
    )
    policy = validate_policy_bridge(values["policy_bridge_receipt"])
    selector_authority_base = {
        "format": "etsf_smolvla_piper_evaluation400_root_selector_authority_v3",
        "status": "frozen_formal190_composite_rank_and_uncertainty_selector",
        "calibration_sha256": calibration["calibration_sha256"],
        "formal190_root_group_ranker_sha256": calibration[
            "root_group_ranker_sha256"
        ],
        "source_rank_score_contract_sha256": ensemble[
            "source_rank_score_contract_sha256"
        ],
        "source_rank_score_contracts": ensemble[
            "source_rank_score_contracts"
        ],
        "source_rank_numeric_contract": calibration[
            "source_rank_numeric_contract"
        ],
        "source_rank_member_authority": calibration[
            "source_rank_member_authority"
        ],
        "source_rank_member_authority_sha256": calibration[
            "source_rank_member_authority_sha256"
        ],
        "deployment_parameters": {
            **calibration["deployment_parameters"],
            "deployment_uncertainty_contract_sha256": calibration[
                "deployment_uncertainty_contract_sha256"
            ],
        },
        "formal190_thresholds": {
            "minimum_formal190_composite_margin": calibration[
                "minimum_group_relative_composite_rank_score_margin"
            ],
            "maximum_formal190_pair_uncertainty": calibration[
                "maximum_structured_pair_uncertainty"
            ],
            "maximum_global_total_uncertainty": calibration[
                "maximum_total_uncertainty"
            ],
            "root_group_ranker_sha256": calibration[
                "root_group_ranker_sha256"
            ],
        },
        "uncertainty_contract": {
            "formal190_object_error_robust_scale_m": calibration[
                "object_error_robust_scale_m"
            ],
            "duration_deployment_scale_applied_before_selector": True,
            "object_deployment_scale_applied_before_selector": True,
            "object_predictions_physical_xyz_before_selector": True,
            "root_recovery_uncertainty_policy": calibration[
                "root_recovery_uncertainty_policy"
            ],
            "root_structured_uncertainty_head_count": calibration[
                "root_structured_uncertainty_head_count"
            ],
            "deployment_uncertainty_contract_sha256": calibration[
                "deployment_uncertainty_contract_sha256"
            ],
        },
        "deployment_uncertainty_implementation": {
            "path": calibration["shared_uncertainty_implementation_path"],
            "file_sha256": calibration[
                "shared_uncertainty_implementation_file_sha256"
            ],
        },
        "margin_comparison": "strict_greater_than_formal190_threshold",
        "alternative_set_contract": (
            "all_legal_candidates_except_lowest_legal_baseline"
        ),
        "evaluation400_outcomes_read": False,
    }
    selector_authority = {
        **selector_authority_base,
        "selector_authority_sha256": canonical_sha256(selector_authority_base),
    }
    deployment_base = {
        "member_count": MEMBER_COUNT,
        "member_checkpoint_sha256": ensemble["member_checkpoint_sha256"],
        "ensemble_manifest_sha256": ensemble["ensemble_manifest_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "head_support_sha256": head["head_support_sha256"],
        "abstention_contract_sha256": calibration["abstention_contract_sha256"],
        "root_group_ranker_sha256": calibration["root_group_ranker_sha256"],
        "deployment_uncertainty_contract_sha256": calibration[
            "deployment_uncertainty_contract_sha256"
        ],
        "source_rank_score_contract_sha256": ensemble[
            "source_rank_score_contract_sha256"
        ],
        "source_rank_score_contracts": ensemble[
            "source_rank_score_contracts"
        ],
        "source_rank_member_authority": selector_authority[
            "source_rank_member_authority"
        ],
        "source_rank_member_authority_sha256": selector_authority[
            "source_rank_member_authority_sha256"
        ],
        "selector_deployment_parameters": selector_authority[
            "deployment_parameters"
        ],
        "formal190_thresholds": selector_authority[
            "formal190_thresholds"
        ],
        "selector_authority": selector_authority,
        "selector_authority_sha256": selector_authority[
            "selector_authority_sha256"
        ],
        "maximum_total_uncertainty": calibration[
            "maximum_total_uncertainty"
        ],
        "calibration_receipt_sha256": calibration_receipt_sha,
        **policy,
    }
    deployment = {
        **deployment_base,
        "deployment_binding_sha256": canonical_sha256(deployment_base),
    }
    pairs = _build_pairs(
        target["evaluation"],
        deployment_sha256=deployment["deployment_binding_sha256"],
    )
    pair_identity_rows = [
        {
            "ordinal": row["ordinal"],
            "pair_id": row["pair_id"],
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
            "initial_scene_state_sha256": row[
                "initial_scene_state_sha256"
            ],
        }
        for row in pairs
    ]
    pair_identity_set_sha = canonical_sha256(pair_identity_rows)
    dependencies = {
        role: {
            "path": str(paths[role]),
            "file_sha256": file_shas[role],
        }
        for role in raw_paths
    }
    dependencies["target_manifest"]["logical_sha256"] = target[
        "manifest_sha256"
    ]
    dependencies["selected_identity_attestation"]["logical_sha256"] = (
        attestation_sha
    )
    dependencies["ensemble_manifest"]["logical_sha256"] = ensemble[
        "ensemble_manifest_sha256"
    ]
    dependencies["calibration"]["logical_sha256"] = calibration[
        "calibration_sha256"
    ]
    dependencies["head_support"]["logical_sha256"] = head[
        "head_support_sha256"
    ]
    dependencies["calibration_receipt"]["logical_sha256"] = (
        calibration_receipt_sha
    )
    dependencies["policy_bridge_receipt"]["logical_sha256"] = policy[
        "verification_sha256"
    ]
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "scope": {
            "task": TASK,
            "source_body": SOURCE_BODY,
            "target_body": TARGET_BODY,
            "actor_id": ACTOR_ID,
            "instruction": INSTRUCTION,
            "pair_count": EVALUATION_GROUPS,
            "target_manifest_evaluation400_is_only_final_paired_lane": True,
            "additional_reserve400_required": False,
            "additional_reserve400_count": 0,
        },
        "dependencies": dependencies,
        "selected_identity_attestation_sha256": attestation_sha,
        "target_all530_identity_set_sha256": target["identity_set_sha256"],
        "target_reset_runtime_contract_sha256": target[
            "target_reset_runtime_contract_sha256"
        ],
        "deployment": deployment,
        "pair_identity_set_sha256": pair_identity_set_sha,
        "condition_order_contract": {
            "algorithm": "sha256(namespace:pair_id)_first_byte_lsb_v1",
            "namespace": CONDITION_ORDER_NAMESPACE,
            "immutable_after_bridge_freeze": True,
            "outcome_dependent": False,
        },
        "pairs": pairs,
        "preoutcome_capability_receipt": {
            "input_json_files_opened": len(raw_paths),
            "hdf5_files_opened": 0,
            "trajectory_files_opened": 0,
            "evaluation_outcome_files_opened": 0,
            "labels_or_outcomes_read": False,
            "checkpoint_files_opened": 0,
            "environment_reset_calls": 0,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "pair_conditions_executed": 0,
            "performance_or_transfer_claim_authorized": False,
        },
        "execution_gate": {
            "execution_authorized_by_this_bridge": False,
            "required_external_authority_format": EXTERNAL_AUTHORITY_FORMAT,
            "external_authority_must_bind_bridge_file_sha256": True,
            "external_authority_must_bind_bridge_sha256": True,
            "external_authority_must_bind_pair_identity_set_sha256": True,
            "all_dependency_files_must_be_rehashed_before_execution": True,
            "exact_reset_identity_must_be_reverified_before_each_condition": True,
            "baseline_and_etsf_must_use_same_root_candidates_and_continuation": True,
            "outcome_or_trajectory_open_before_external_authority": False,
        },
        "protocol_lineage": {
            "existing_paired_success_v1_modified": False,
            "frozen_schema6_v2_modified": False,
            "new_bridge_v2_only": True,
        },
    }
    return {**base, "bridge_sha256": canonical_sha256(base)}


def validate_bridge(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = verify_signed(value, "bridge_sha256", "evaluation400 bridge")
    expected_root = {
        "format",
        "status",
        "scope",
        "dependencies",
        "selected_identity_attestation_sha256",
        "target_all530_identity_set_sha256",
        "target_reset_runtime_contract_sha256",
        "deployment",
        "pair_identity_set_sha256",
        "condition_order_contract",
        "pairs",
        "preoutcome_capability_receipt",
        "execution_gate",
        "protocol_lineage",
        "bridge_sha256",
    }
    scope = value.get("scope")
    capability = value.get("preoutcome_capability_receipt")
    gate = value.get("execution_gate")
    deployment = value.get("deployment")
    pairs = value.get("pairs")
    expected_deployment_fields = {
        "member_count", "member_checkpoint_sha256",
        "ensemble_manifest_sha256", "calibration_sha256",
        "head_support_sha256", "abstention_contract_sha256",
        "root_group_ranker_sha256",
        "deployment_uncertainty_contract_sha256",
        "source_rank_score_contract_sha256", "source_rank_score_contracts",
        "source_rank_member_authority",
        "source_rank_member_authority_sha256",
        "selector_deployment_parameters", "formal190_thresholds",
        "selector_authority",
        "selector_authority_sha256", "maximum_total_uncertainty",
        "calibration_receipt_sha256", "verification_sha256",
        "bridge_contract_sha256", "runtime_binding_sha256",
        "action_mapping_binding_sha256", "state_feature_binding_sha256",
        "deployment_binding_sha256",
    }
    if (
        set(value) != expected_root
        or value.get("format") != FORMAT
        or value.get("status") != STATUS
        or not is_sha(value.get("target_reset_runtime_contract_sha256"))
        or not isinstance(scope, Mapping)
        or scope.get("pair_count") != EVALUATION_GROUPS
        or scope.get("target_manifest_evaluation400_is_only_final_paired_lane")
        is not True
        or scope.get("additional_reserve400_required") is not False
        or scope.get("additional_reserve400_count") != 0
        or not isinstance(capability, Mapping)
        or capability.get("hdf5_files_opened") != 0
        or capability.get("trajectory_files_opened") != 0
        or capability.get("evaluation_outcome_files_opened") != 0
        or capability.get("labels_or_outcomes_read") is not False
        or capability.get("policy_import_or_forward_calls") != 0
        or capability.get("pair_conditions_executed") != 0
        or not isinstance(gate, Mapping)
        or gate.get("execution_authorized_by_this_bridge") is not False
        or gate.get("required_external_authority_format")
        != EXTERNAL_AUTHORITY_FORMAT
        or not isinstance(deployment, Mapping)
        or set(deployment) != expected_deployment_fields
        or deployment.get("member_count") != MEMBER_COUNT
        or not isinstance(pairs, list)
        or len(pairs) != EVALUATION_GROUPS
    ):
        raise Evaluation400BridgeError("evaluation400 bridge boundary changed")
    expected_deployment = dict(deployment)
    recorded_deployment = expected_deployment.pop(
        "deployment_binding_sha256", None
    )
    if (
        not is_sha(recorded_deployment)
        or recorded_deployment != canonical_sha256(expected_deployment)
    ):
        raise Evaluation400BridgeError("deployment binding changed")
    selector_authority = deployment.get("selector_authority")
    expected_selector_fields = {
        "format", "status", "calibration_sha256",
        "formal190_root_group_ranker_sha256",
        "source_rank_score_contract_sha256",
        "source_rank_score_contracts", "source_rank_numeric_contract",
        "source_rank_member_authority",
        "source_rank_member_authority_sha256",
        "deployment_parameters",
        "formal190_thresholds",
        "uncertainty_contract", "deployment_uncertainty_implementation",
        "margin_comparison", "alternative_set_contract",
        "evaluation400_outcomes_read", "selector_authority_sha256",
    }
    selector_logical = (
        verify_signed(
            selector_authority,
            "selector_authority_sha256",
            "root selector authority",
        )
        if isinstance(selector_authority, Mapping)
        else None
    )
    member_authority = validate_source_rank_member_authority(
        selector_authority.get("source_rank_member_authority")
        if isinstance(selector_authority, Mapping) else None,
        selector_authority.get("source_rank_member_authority_sha256")
        if isinstance(selector_authority, Mapping) else None,
        role="root selector source rank member authority",
    )
    if (
        not is_sha(selector_logical)
        or set(selector_authority) != expected_selector_fields
        or selector_logical != deployment.get("selector_authority_sha256")
        or selector_authority.get("format")
        != "etsf_smolvla_piper_evaluation400_root_selector_authority_v3"
        or selector_authority.get("status")
        != "frozen_formal190_composite_rank_and_uncertainty_selector"
        or selector_authority.get("evaluation400_outcomes_read") is not False
        or selector_authority.get("calibration_sha256")
        != deployment.get("calibration_sha256")
        or selector_authority.get("formal190_root_group_ranker_sha256")
        != deployment.get("root_group_ranker_sha256")
        or selector_authority.get("source_rank_score_contract_sha256")
        != deployment.get("source_rank_score_contract_sha256")
        or selector_authority.get("source_rank_score_contracts")
        != deployment.get("source_rank_score_contracts")
        or selector_authority.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or member_authority["source_rank_member_authority"]
        != deployment.get("source_rank_member_authority")
        or member_authority["source_rank_member_authority_sha256"]
        != deployment.get("source_rank_member_authority_sha256")
        or selector_authority.get("deployment_parameters")
        != deployment.get("selector_deployment_parameters")
        or selector_authority.get("formal190_thresholds")
        != deployment.get("formal190_thresholds")
        or not isinstance(
            selector_authority.get("deployment_parameters"), Mapping
        )
        or set(selector_authority["deployment_parameters"]) != {
            "post_event_temperature", "next_event_temperature",
            "success_temperature", "conditional_recovery_temperature",
            "duration_scale_multiplier", "object_scale_multiplier",
            "object_error_robust_scale_m",
            "deployment_uncertainty_contract_sha256",
        }
        or not isinstance(
            selector_authority.get("formal190_thresholds"), Mapping
        )
        or set(selector_authority["formal190_thresholds"]) != {
            "minimum_formal190_composite_margin",
            "maximum_formal190_pair_uncertainty",
            "maximum_global_total_uncertainty",
            "root_group_ranker_sha256",
        }
        or selector_authority["formal190_thresholds"].get(
            "maximum_global_total_uncertainty"
        ) != deployment.get("maximum_total_uncertainty")
        or selector_authority["formal190_thresholds"].get(
            "root_group_ranker_sha256"
        ) != deployment.get("root_group_ranker_sha256")
        or selector_authority["deployment_parameters"].get(
            "deployment_uncertainty_contract_sha256"
        ) != deployment.get("deployment_uncertainty_contract_sha256")
        or any(
            isinstance(selector_authority["deployment_parameters"].get(field), bool)
            or not isinstance(
                selector_authority["deployment_parameters"].get(field),
                (int, float),
            )
            or not math.isfinite(float(
                selector_authority["deployment_parameters"][field]
            ))
            or float(selector_authority["deployment_parameters"][field]) <= 0.0
            for field in {
                "post_event_temperature", "next_event_temperature",
                "success_temperature", "conditional_recovery_temperature",
                "duration_scale_multiplier", "object_scale_multiplier",
                "object_error_robust_scale_m",
            }
        )
        or any(
            isinstance(selector_authority["formal190_thresholds"].get(field), bool)
            or not isinstance(
                selector_authority["formal190_thresholds"].get(field),
                (int, float),
            )
            or not math.isfinite(float(
                selector_authority["formal190_thresholds"][field]
            ))
            or float(selector_authority["formal190_thresholds"][field]) < 0.0
            for field in {
                "minimum_formal190_composite_margin",
                "maximum_formal190_pair_uncertainty",
                "maximum_global_total_uncertainty",
            }
        )
        or not isinstance(
            deployment.get("source_rank_score_contract_sha256"), list
        )
        or len(deployment["source_rank_score_contract_sha256"]) != MEMBER_COUNT
        or len(set(deployment["source_rank_score_contract_sha256"]))
        != MEMBER_COUNT
        or any(
            not is_sha(item)
            for item in deployment["source_rank_score_contract_sha256"]
        )
        or not isinstance(
            deployment.get("source_rank_score_contracts"), list
        )
        or len(deployment["source_rank_score_contracts"]) != MEMBER_COUNT
        or any(
            not isinstance(contract, Mapping)
            or verify_signed(
                contract, "contract_sha256", "source rank score contract"
            ) != deployment["source_rank_score_contract_sha256"][index]
            or isinstance(contract.get("success_temperature"), bool)
            or not isinstance(contract.get("success_temperature"), (int, float))
            or not math.isfinite(float(contract["success_temperature"]))
            or float(contract["success_temperature"]) <= 0.0
            for index, contract in enumerate(
                deployment["source_rank_score_contracts"]
            )
        )
        or any(
            authority_member["source_checkpoint_file_sha256"]
            != deployment["source_rank_score_contracts"][index].get(
                "source_checkpoint_file_sha256"
            )
            or authority_member["source_rank_score_contract_sha256"]
            != deployment["source_rank_score_contract_sha256"][index]
            or float(authority_member["success_temperature"])
            != float(deployment["source_rank_score_contracts"][index].get(
                "success_temperature", -1.0
            ))
            for index, authority_member in enumerate(
                member_authority["source_rank_member_authority"]["members"]
            )
        )
        or selector_authority.get("uncertainty_contract", {}).get(
            "deployment_uncertainty_contract_sha256"
        )
        != deployment.get("deployment_uncertainty_contract_sha256")
        or selector_authority.get("margin_comparison")
        != "strict_greater_than_formal190_threshold"
        or selector_authority.get("alternative_set_contract")
        != "all_legal_candidates_except_lowest_legal_baseline"
    ):
        raise Evaluation400BridgeError("root selector authority changed")
    pair_ids: set[str] = set()
    identity_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(pairs):
        exact_pair_fields = {
            "ordinal",
            "pair_id",
            "target_manifest_split",
            "target_manifest_global_ordinal",
            "requested_seed",
            "resolved_seed",
            "instruction_sha256",
            "initial_scene_state_sha256",
            "initial_measured_joint_state_sha256",
            "initial_commanded_drive_target_sha256",
            "same_initial_state_for_both_conditions",
            "exact_reset_identity_reverification_before_each_condition",
            "condition_order",
            "baseline_condition",
            "etsf_condition",
            "outcome_or_trajectory_fields_present",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != exact_pair_fields
            or row.get("ordinal") != ordinal
            or row.get("target_manifest_split") != "evaluation"
            or row.get("target_manifest_global_ordinal")
            != ADAPTATION_GROUPS + VALIDATION_GROUPS + ordinal
            or not is_sha(row.get("pair_id"))
            or row.get("pair_id") in pair_ids
            or row.get("condition_order")
            != paired_condition_order(str(row.get("pair_id")))
            or row.get("same_initial_state_for_both_conditions") is not True
            or row.get("outcome_or_trajectory_fields_present") is not False
            or row.get("etsf_condition", {}).get(
                "deployment_binding_sha256"
            )
            != recorded_deployment
        ):
            raise Evaluation400BridgeError("paired identity/order changed")
        pair_ids.add(str(row["pair_id"]))
        identity_rows.append(
            {
                "ordinal": ordinal,
                "pair_id": row["pair_id"],
                "requested_seed": row["requested_seed"],
                "resolved_seed": row["resolved_seed"],
                "initial_scene_state_sha256": row[
                    "initial_scene_state_sha256"
                ],
            }
        )
    if value.get("pair_identity_set_sha256") != canonical_sha256(identity_rows):
        raise Evaluation400BridgeError("pair identity set SHA changed")
    return {
        "status": "verified_preoutcome_evaluation400_identity_bridge",
        "bridge_sha256": logical,
        "pair_identity_set_sha256": value["pair_identity_set_sha256"],
        "pair_count": EVALUATION_GROUPS,
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "pair_conditions_executed": 0,
        "execution_authorized": False,
    }


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = _safe_json_path(path, "output", must_exist=False)
    output.parent.resolve(strict=True)
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
    for name in (
        "target-manifest",
        "selected-identity-attestation",
        "ensemble-manifest",
        "calibration",
        "head-support",
        "calibration-receipt",
        "policy-bridge-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    value = freeze_bridge(
        target_manifest_path=args.target_manifest,
        target_manifest_file_sha256=args.target_manifest_file_sha256,
        selected_identity_attestation_path=args.selected_identity_attestation,
        selected_identity_attestation_file_sha256=(
            args.selected_identity_attestation_file_sha256
        ),
        ensemble_manifest_path=args.ensemble_manifest,
        ensemble_manifest_file_sha256=args.ensemble_manifest_file_sha256,
        calibration_path=args.calibration,
        calibration_file_sha256=args.calibration_file_sha256,
        head_support_path=args.head_support,
        head_support_file_sha256=args.head_support_file_sha256,
        calibration_receipt_path=args.calibration_receipt,
        calibration_receipt_file_sha256=args.calibration_receipt_file_sha256,
        policy_bridge_receipt_path=args.policy_bridge_receipt,
        policy_bridge_receipt_file_sha256=args.policy_bridge_receipt_file_sha256,
    )
    audit = validate_bridge(value)
    write_json_new(args.output, value)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
