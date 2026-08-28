#!/usr/bin/env python3
"""Pure contracts for the evaluation400 v4 sealed prediction audit.

This module deliberately performs no filesystem, simulator, checkpoint, HDF,
trajectory, or label I/O.  Callers provide already materialized in-memory
arrays and inject the canonical target-derivation functions they have bound in
their own immutable authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT_PRECOMMIT_FORMAT = (
    "etsf_smolvla_piper_evaluation400_root_prediction_precommit_v1"
)
ROOT_PRECOMMIT_STATUS = "committed_before_any_evaluation400_condition"
ROOT_TENSOR_FORMAT = "etsf_smolvla_piper_root_five_head_tensor_commitment_v1"
ROOT_TENSOR_STATUS = "five_member_all_legal_candidate_predictions_committed"
RECOVERY_PRECOMMIT_FORMAT = (
    "etsf_smolvla_piper_evaluation400_recovery_pre_step_commitment_v1"
)
RECOVERY_PRECOMMIT_STATUS = "prediction_committed_before_bound_simulator_step"
BROKER_ACK_FORMAT = "etsf_smolvla_piper_evaluation400_recovery_broker_ack_v1"
BROKER_ACK_STATUS = "worm_ledger_commit_acknowledged_step_authorized"
TARGET_ENVELOPE_FORMAT = (
    "etsf_smolvla_piper_evaluation400_sealed_target_envelope_v1"
)
TARGET_ENVELOPE_STATUS = "encrypted_target_opaque_until_complete_terminal"
TARGET_AAD_FORMAT = "etsf_smolvla_piper_evaluation400_target_aad_v1"
COMPLETENESS_FORMAT = (
    "etsf_smolvla_piper_evaluation400_v4_terminal_completeness_gate_v1"
)
COMPLETENESS_STATUS = "all_400_pairs_and_1600_conditions_terminal"

PAIR_COUNT = 400
CONDITION_COUNT = 1600
MEMBER_COUNT = 5
ROOT_HEADS = (
    "post_event",
    "next_event",
    "duration",
    "success",
    "object_effect",
)
ALL_HEADS = (*ROOT_HEADS, "recovery")
ROOT_RECOVERY_POLICY = (
    "excluded_at_initial_e0_without_observed_operational_regress"
)
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)
ENCRYPTION_ALGORITHM = (
    "X25519+HKDF-SHA256+ChaCha20-Poly1305"
)
KDF_INFO = b"ETSF/SmolVLA/Piper/evaluation400-v4/sealed-target-v1\0"
SHA_CHARS = frozenset("0123456789abcdef")
ZERO_SHA256 = "0" * 64
ROOT_TENSOR_FIELDS = (
    "post_event_logits",
    "next_event_logits",
    "duration_log_mean",
    "duration_log_scale",
    "success_logit",
    "object_mean",
    "object_log_scale",
)
ROOT_AUTHORITY_FIELDS = {
    "five_member_checkpoint_file_sha256",
    "source_rank_score_contract_sha256",
    "source_rank_member_authority_sha256",
    "source_rank_numeric_contract",
    "calibration_sha256",
    "ensemble_manifest_sha256",
    "deployment_uncertainty_contract_sha256",
    "deployment_uncertainty_implementation_file_sha256",
    "canonical_event_spec_file_sha256",
    "schema6_runtime_contract_sha256",
}
HEAD_MINIMUM_PAIR_SUPPORT = {
    "post_event": 10,
    "next_event": 10,
    "duration": 10,
    "success": 50,
    "recovery": 10,
    "object_effect": 50,
}
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260828


class AuditContractError(RuntimeError):
    """A sealed-audit schema, chronology, crypto, or metric check failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def _require_sha(value: Any, role: str) -> str:
    if not is_sha256(value):
        raise AuditContractError(f"{role} must be exact lowercase SHA-256")
    return str(value)


def _require_int(
    value: Any, role: str, *, minimum: int | None = None,
    expected: int | None = None,
) -> int:
    if type(value) is not int:
        raise AuditContractError(f"{role} must be an exact integer, not bool")
    if minimum is not None and value < minimum:
        raise AuditContractError(f"{role} is below its minimum")
    if expected is not None and value != expected:
        raise AuditContractError(f"{role} differs from exact authority")
    return value


def _require_float(value: Any, role: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditContractError(f"{role} must be a finite numeric value, not bool")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise AuditContractError(f"{role} must be finite and valid")
    return result


def _require_bool(value: Any, expected: bool, role: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise AuditContractError(f"{role} must be exact {expected!r}")
    return value


def _signed_document(base: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    normalized = dict(base)
    return {**normalized, field_name: canonical_sha256(normalized)}


def _verify_document(
    value: Mapping[str, Any], field_name: str, expected_fields: set[str], role: str,
) -> str:
    if not isinstance(value, Mapping) or set(value) != expected_fields | {field_name}:
        raise AuditContractError(f"{role} fields changed")
    logical = value.get(field_name)
    _require_sha(logical, f"{role} logical SHA")
    base = {key: child for key, child in value.items() if key != field_name}
    if logical != canonical_sha256(base):
        raise AuditContractError(f"{role} canonical SHA mismatch")
    return str(logical)


def _float32_tensor(value: Any, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise AuditContractError(f"{role} must have exact native float32 dtype")
    if not array.flags.c_contiguous:
        raise AuditContractError(f"{role} must be C-contiguous")
    if not np.isfinite(array).all():
        raise AuditContractError(f"{role} contains non-finite values")
    return array


def _tensor_sha256(name: str, array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"ETSF/evaluation400/root-tensor-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0float32\0")
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_root_tensor_commitment(
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact online float32 tensors for the five root-applicable heads."""

    if not isinstance(predictions, Mapping) or set(predictions) != set(
        ROOT_TENSOR_FIELDS
    ):
        raise AuditContractError("root prediction tensor inventory changed")
    arrays = {
        name: _float32_tensor(predictions[name], f"root tensor {name}")
        for name in ROOT_TENSOR_FIELDS
    }
    post = arrays["post_event_logits"]
    next_event = arrays["next_event_logits"]
    success = arrays["success_logit"]
    duration_mean = arrays["duration_log_mean"]
    duration_scale = arrays["duration_log_scale"]
    object_mean = arrays["object_mean"]
    object_scale = arrays["object_log_scale"]
    if (
        post.ndim != 3
        or post.shape[0] != MEMBER_COUNT
        or post.shape[1] < 1
        or post.shape[2] < 2
        or next_event.shape != post.shape
        or success.shape != post.shape[:2]
        or duration_mean.shape != success.shape
        or duration_scale.shape != success.shape
        or object_mean.ndim != 3
        or object_mean.shape[:2] != success.shape
        or object_mean.shape[2] < 1
        or object_scale.shape != object_mean.shape
    ):
        raise AuditContractError("root five-head tensor shapes changed")
    fields = {
        name: {
            "dtype": "float32",
            "shape": list(array.shape),
            "tensor_sha256": _tensor_sha256(name, array),
        }
        for name, array in arrays.items()
    }
    base = {
        "format": ROOT_TENSOR_FORMAT,
        "status": ROOT_TENSOR_STATUS,
        "member_count": MEMBER_COUNT,
        "candidate_count": int(success.shape[1]),
        "event_class_count": int(post.shape[2]),
        "object_dimension": int(object_mean.shape[2]),
        "head_names": list(ROOT_HEADS),
        "tensor_field_order": list(ROOT_TENSOR_FIELDS),
        "fields": fields,
        "object_prediction_space": "physical_delta_xyz_m",
        "duration_prediction_space": "log1p_decision_steps",
        "duration_and_object_deployment_scale_applied_exactly_once": True,
    }
    return _signed_document(base, "tensor_set_sha256")


def validate_root_tensor_commitment(value: Mapping[str, Any]) -> str:
    fields = {
        "format", "status", "member_count", "candidate_count",
        "event_class_count", "object_dimension", "head_names",
        "tensor_field_order", "fields", "object_prediction_space",
        "duration_prediction_space",
        "duration_and_object_deployment_scale_applied_exactly_once",
    }
    logical = _verify_document(
        value, "tensor_set_sha256", fields, "root tensor commitment"
    )
    tensor_fields = value.get("fields")
    if (
        value.get("format") != ROOT_TENSOR_FORMAT
        or value.get("status") != ROOT_TENSOR_STATUS
        or _require_int(value.get("member_count"), "root member count") != MEMBER_COUNT
        or _require_int(value.get("candidate_count"), "root candidate count", minimum=1) < 1
        or _require_int(value.get("event_class_count"), "event class count", minimum=2) < 2
        or _require_int(value.get("object_dimension"), "object dimension", minimum=1) < 1
        or value.get("head_names") != list(ROOT_HEADS)
        or value.get("tensor_field_order") != list(ROOT_TENSOR_FIELDS)
        or not isinstance(tensor_fields, Mapping)
        or set(tensor_fields) != set(ROOT_TENSOR_FIELDS)
        or value.get("object_prediction_space") != "physical_delta_xyz_m"
        or value.get("duration_prediction_space") != "log1p_decision_steps"
        or value.get("duration_and_object_deployment_scale_applied_exactly_once")
        is not True
    ):
        raise AuditContractError("root tensor commitment contract changed")
    member_count = int(value["member_count"])
    candidates = int(value["candidate_count"])
    classes = int(value["event_class_count"])
    object_dim = int(value["object_dimension"])
    expected_shapes = {
        "post_event_logits": [member_count, candidates, classes],
        "next_event_logits": [member_count, candidates, classes],
        "duration_log_mean": [member_count, candidates],
        "duration_log_scale": [member_count, candidates],
        "success_logit": [member_count, candidates],
        "object_mean": [member_count, candidates, object_dim],
        "object_log_scale": [member_count, candidates, object_dim],
    }
    for name, expected_shape in expected_shapes.items():
        row = tensor_fields.get(name)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"dtype", "shape", "tensor_sha256"}
            or row.get("dtype") != "float32"
            or row.get("shape") != expected_shape
            or not is_sha256(row.get("tensor_sha256"))
        ):
            raise AuditContractError(f"root tensor descriptor changed: {name}")
    return logical


def _validate_root_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ROOT_AUTHORITY_FIELDS:
        raise AuditContractError("root prediction authority fields changed")
    checkpoints = value.get("five_member_checkpoint_file_sha256")
    rank_contracts = value.get("source_rank_score_contract_sha256")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != MEMBER_COUNT
        or len(set(checkpoints)) != MEMBER_COUNT
        or any(not is_sha256(item) for item in checkpoints)
        or not isinstance(rank_contracts, list)
        or len(rank_contracts) != MEMBER_COUNT
        or len(set(rank_contracts)) != MEMBER_COUNT
        or any(not is_sha256(item) for item in rank_contracts)
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
    ):
        raise AuditContractError("five-member root authority changed")
    for field_name in ROOT_AUTHORITY_FIELDS - {
        "five_member_checkpoint_file_sha256",
        "source_rank_score_contract_sha256",
        "source_rank_numeric_contract",
    }:
        _require_sha(value.get(field_name), f"root authority {field_name}")
    return dict(value)


def build_root_precommit(
    *,
    protocol_core_v4_sha256: str,
    parent_v3_core_sha256: str,
    pair_id: str,
    pair_ordinal: int,
    shared_snapshot_sha256: str,
    ordered_candidate_sha256: Sequence[str],
    candidate_legal: Sequence[bool],
    candidate_registry_sha256: str,
    prediction_candidate_indices: Sequence[int],
    tensor_commitment: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    tensor_sha = validate_root_tensor_commitment(tensor_commitment)
    _validate_root_authority(authority)
    ordered = list(ordered_candidate_sha256)
    legal = list(candidate_legal)
    predicted = list(prediction_candidate_indices)
    _require_int(pair_ordinal, "pair ordinal", minimum=0)
    if (
        not is_sha256(pair_id)
        or not ordered
        or any(not is_sha256(item) for item in ordered)
        or len(set(ordered)) != len(ordered)
        or len(legal) != len(ordered)
        or any(type(item) is not bool for item in legal)
        or not any(legal)
        or any(type(item) is not int for item in predicted)
        or predicted != [index for index, enabled in enumerate(legal) if enabled]
        or tensor_commitment.get("candidate_count") != len(predicted)
    ):
        raise AuditContractError("root candidate/prediction registry changed")
    for value, role in (
        (protocol_core_v4_sha256, "v4 core"),
        (parent_v3_core_sha256, "parent v3 core"),
        (shared_snapshot_sha256, "shared snapshot"),
        (candidate_registry_sha256, "candidate registry"),
    ):
        _require_sha(value, role)
    expected_registry_sha256 = canonical_sha256(
        {
            "pair_id": pair_id,
            "candidate_count": len(ordered),
            "ordered_candidate_sha256": ordered,
            "candidate_legal": legal,
        }
    )
    if candidate_registry_sha256 != expected_registry_sha256:
        raise AuditContractError("root candidate registry SHA mismatch")
    base = {
        "format": ROOT_PRECOMMIT_FORMAT,
        "status": ROOT_PRECOMMIT_STATUS,
        "protocol_core_v4_sha256": protocol_core_v4_sha256,
        "parent_v3_core_sha256": parent_v3_core_sha256,
        "pair_id": pair_id,
        "pair_ordinal": pair_ordinal,
        "shared_snapshot_sha256": shared_snapshot_sha256,
        "ordered_candidate_sha256": ordered,
        "candidate_legal": legal,
        "candidate_registry_sha256": candidate_registry_sha256,
        "prediction_candidate_indices": predicted,
        "tensor_commitment": dict(tensor_commitment),
        "tensor_set_sha256": tensor_sha,
        "authority": dict(authority),
        "root_head_names": list(ROOT_HEADS),
        "recovery_root": {
            "status": "not_applicable",
            "policy": ROOT_RECOVERY_POLICY,
            "included_in_root_head_count": False,
        },
        "precondition_barrier": {
            "simulator_step_calls": 0,
            "condition_started_count": 0,
            "committed_before_any_condition": True,
        },
    }
    return _signed_document(base, "commit_sha256")


def validate_root_precommit(value: Mapping[str, Any]) -> str:
    fields = {
        "format", "status", "protocol_core_v4_sha256", "parent_v3_core_sha256",
        "pair_id", "pair_ordinal", "shared_snapshot_sha256",
        "ordered_candidate_sha256", "candidate_legal",
        "candidate_registry_sha256", "prediction_candidate_indices",
        "tensor_commitment", "tensor_set_sha256", "authority",
        "root_head_names", "recovery_root", "precondition_barrier",
    }
    logical = _verify_document(value, "commit_sha256", fields, "root precommit")
    if (
        value.get("format") != ROOT_PRECOMMIT_FORMAT
        or value.get("status") != ROOT_PRECOMMIT_STATUS
        or value.get("root_head_names") != list(ROOT_HEADS)
    ):
        raise AuditContractError("root precommit envelope changed")
    for field_name in (
        "protocol_core_v4_sha256", "parent_v3_core_sha256", "pair_id",
        "shared_snapshot_sha256", "candidate_registry_sha256", "tensor_set_sha256",
    ):
        _require_sha(value.get(field_name), f"root precommit {field_name}")
    _require_int(value.get("pair_ordinal"), "pair ordinal", minimum=0)
    tensor = value.get("tensor_commitment")
    if not isinstance(tensor, Mapping) or validate_root_tensor_commitment(tensor) != value[
        "tensor_set_sha256"
    ]:
        raise AuditContractError("root precommit tensor binding changed")
    _validate_root_authority(value.get("authority"))
    ordered = value.get("ordered_candidate_sha256")
    legal = value.get("candidate_legal")
    predicted = value.get("prediction_candidate_indices")
    if (
        not isinstance(ordered, list)
        or not ordered
        or any(not is_sha256(item) for item in ordered)
        or len(set(ordered)) != len(ordered)
        or not isinstance(legal, list)
        or len(legal) != len(ordered)
        or any(type(item) is not bool for item in legal)
        or not any(legal)
        or not isinstance(predicted, list)
        or any(type(item) is not int for item in predicted)
        or predicted != [index for index, enabled in enumerate(legal) if enabled]
        or tensor.get("candidate_count") != len(predicted)
    ):
        raise AuditContractError("root precommit candidate binding changed")
    expected_registry_sha256 = canonical_sha256(
        {
            "pair_id": value["pair_id"],
            "candidate_count": len(ordered),
            "ordered_candidate_sha256": ordered,
            "candidate_legal": legal,
        }
    )
    if value["candidate_registry_sha256"] != expected_registry_sha256:
        raise AuditContractError("root candidate registry SHA mismatch")
    recovery = value.get("recovery_root")
    barrier = value.get("precondition_barrier")
    if recovery != {
        "status": "not_applicable",
        "policy": ROOT_RECOVERY_POLICY,
        "included_in_root_head_count": False,
    }:
        raise AuditContractError("recovery was made applicable at the e0 root")
    if not isinstance(barrier, Mapping) or set(barrier) != {
        "simulator_step_calls", "condition_started_count",
        "committed_before_any_condition",
    }:
        raise AuditContractError("root precondition barrier fields changed")
    _require_int(barrier.get("simulator_step_calls"), "root step calls", expected=0)
    _require_int(
        barrier.get("condition_started_count"), "root condition starts", expected=0
    )
    _require_bool(
        barrier.get("committed_before_any_condition"), True,
        "root precondition chronology",
    )
    return logical


def build_recovery_pre_step_commitment(
    *,
    protocol_core_v4_sha256: str,
    root_prediction_commit_sha256: str,
    pair_id: str,
    pair_ordinal: int,
    condition_id: str,
    condition_position: int,
    step_index: int,
    commit_sequence: int,
    previous_commit_sha256: str | None,
    pre_action_snapshot_sha256: str,
    chosen_candidate_index: int,
    chosen_candidate_sha256: str,
    current_event_id: int,
    historical_peak_event_id: int,
    member_recovery_logits: Sequence[float],
    conditional_recovery_temperature: float,
    calibration_sha256: str,
    source_rank_member_authority_sha256: str,
    schema6_runtime_contract_sha256: str,
) -> dict[str, Any]:
    for value, role in (
        (protocol_core_v4_sha256, "v4 core"),
        (root_prediction_commit_sha256, "root prediction commit"),
        (pair_id, "pair ID"),
        (pre_action_snapshot_sha256, "pre-action snapshot"),
        (chosen_candidate_sha256, "chosen candidate"),
        (calibration_sha256, "calibration"),
        (source_rank_member_authority_sha256, "member authority"),
        (schema6_runtime_contract_sha256, "runtime contract"),
    ):
        _require_sha(value, role)
    for value, role in (
        (pair_ordinal, "pair ordinal"),
        (condition_position, "condition position"),
        (step_index, "step index"),
        (commit_sequence, "commit sequence"),
        (chosen_candidate_index, "chosen candidate index"),
        (current_event_id, "current event"),
        (historical_peak_event_id, "historical peak event"),
    ):
        _require_int(value, role, minimum=0)
    if not isinstance(condition_id, str) or not condition_id:
        raise AuditContractError("condition ID must be nonempty text")
    if previous_commit_sha256 is not None:
        _require_sha(previous_commit_sha256, "previous recovery commit")
    if (commit_sequence == 0) is not (previous_commit_sha256 is None):
        raise AuditContractError("recovery commitment genesis/chain changed")
    if step_index < 1 or historical_peak_event_id < 1:
        raise AuditContractError("recovery precommit cannot be made at e0")
    if current_event_id > historical_peak_event_id:
        raise AuditContractError("recovery historical peak is below current event")
    logits = [
        _require_float(value, f"member recovery logit {index}")
        for index, value in enumerate(member_recovery_logits)
    ]
    if len(logits) != MEMBER_COUNT:
        raise AuditContractError("recovery commitment requires exactly five logits")
    temperature = _require_float(
        conditional_recovery_temperature, "conditional recovery temperature",
        positive=True,
    )
    base = {
        "format": RECOVERY_PRECOMMIT_FORMAT,
        "status": RECOVERY_PRECOMMIT_STATUS,
        "protocol_core_v4_sha256": protocol_core_v4_sha256,
        "root_prediction_commit_sha256": root_prediction_commit_sha256,
        "pair_id": pair_id,
        "pair_ordinal": pair_ordinal,
        "condition_id": condition_id,
        "condition_position": condition_position,
        "step_index": step_index,
        "commit_sequence": commit_sequence,
        "previous_commit_sha256": previous_commit_sha256,
        "pre_action_snapshot_sha256": pre_action_snapshot_sha256,
        "chosen_candidate_index": chosen_candidate_index,
        "chosen_candidate_sha256": chosen_candidate_sha256,
        "current_event_id": current_event_id,
        "historical_peak_event_id": historical_peak_event_id,
        "member_recovery_logits": logits,
        "conditional_recovery_temperature": temperature,
        "calibration_sha256": calibration_sha256,
        "source_rank_member_authority_sha256": (
            source_rank_member_authority_sha256
        ),
        "schema6_runtime_contract_sha256": schema6_runtime_contract_sha256,
        "timing_contract": {
            "prediction_committed_before_step": True,
            "future_regress_or_recovery_observed": False,
            "label_driven_inference": False,
        },
    }
    return _signed_document(base, "commit_sha256")


def validate_recovery_pre_step_commitment(
    value: Mapping[str, Any],
    *,
    expected_pair_id: str | None = None,
    expected_condition_id: str | None = None,
    expected_step_index: int | None = None,
    expected_commit_sequence: int | None = None,
    expected_previous_commit_sha256: str | None | object = ...,
    seen_commit_sha256: set[str] | None = None,
) -> str:
    fields = {
        "format", "status", "protocol_core_v4_sha256",
        "root_prediction_commit_sha256", "pair_id", "pair_ordinal",
        "condition_id", "condition_position", "step_index", "commit_sequence",
        "previous_commit_sha256", "pre_action_snapshot_sha256",
        "chosen_candidate_index", "chosen_candidate_sha256", "current_event_id",
        "historical_peak_event_id", "member_recovery_logits",
        "conditional_recovery_temperature", "calibration_sha256",
        "source_rank_member_authority_sha256",
        "schema6_runtime_contract_sha256", "timing_contract",
    }
    logical = _verify_document(
        value, "commit_sha256", fields, "recovery pre-step commitment"
    )
    if (
        value.get("format") != RECOVERY_PRECOMMIT_FORMAT
        or value.get("status") != RECOVERY_PRECOMMIT_STATUS
    ):
        raise AuditContractError("recovery pre-step envelope changed")
    for field_name in (
        "protocol_core_v4_sha256", "root_prediction_commit_sha256", "pair_id",
        "pre_action_snapshot_sha256", "chosen_candidate_sha256",
        "calibration_sha256", "source_rank_member_authority_sha256",
        "schema6_runtime_contract_sha256",
    ):
        _require_sha(value.get(field_name), f"recovery {field_name}")
    for field_name in (
        "pair_ordinal", "condition_position", "step_index", "commit_sequence",
        "chosen_candidate_index", "current_event_id", "historical_peak_event_id",
    ):
        _require_int(value.get(field_name), f"recovery {field_name}", minimum=0)
    if not isinstance(value.get("condition_id"), str) or not value["condition_id"]:
        raise AuditContractError("recovery condition ID is invalid")
    previous = value.get("previous_commit_sha256")
    if previous is not None:
        _require_sha(previous, "previous recovery commit")
    if (value["commit_sequence"] == 0) is not (previous is None):
        raise AuditContractError("recovery commitment genesis/chain changed")
    if value["step_index"] < 1 or value["historical_peak_event_id"] < 1:
        raise AuditContractError("recovery precommit cannot be made at e0")
    if value["current_event_id"] > value["historical_peak_event_id"]:
        raise AuditContractError("recovery historical peak is below current event")
    logits = value.get("member_recovery_logits")
    if not isinstance(logits, list) or len(logits) != MEMBER_COUNT:
        raise AuditContractError("recovery commitment does not contain five logits")
    for index, child in enumerate(logits):
        _require_float(child, f"member recovery logit {index}")
    _require_float(
        value.get("conditional_recovery_temperature"),
        "conditional recovery temperature", positive=True,
    )
    if value.get("timing_contract") != {
        "prediction_committed_before_step": True,
        "future_regress_or_recovery_observed": False,
        "label_driven_inference": False,
    }:
        raise AuditContractError("recovery prediction chronology changed")
    if expected_pair_id is not None and value.get("pair_id") != expected_pair_id:
        raise AuditContractError("recovery commitment is bound to the wrong pair")
    if expected_condition_id is not None and value.get(
        "condition_id"
    ) != expected_condition_id:
        raise AuditContractError("recovery commitment is bound to the wrong condition")
    if expected_step_index is not None:
        _require_int(expected_step_index, "expected step", minimum=0)
        if value.get("step_index") != expected_step_index:
            raise AuditContractError("recovery commitment is bound to the wrong step")
    if expected_commit_sequence is not None:
        _require_int(expected_commit_sequence, "expected commit sequence", minimum=0)
        if value.get("commit_sequence") != expected_commit_sequence:
            raise AuditContractError("recovery commitment sequence changed")
    if expected_previous_commit_sha256 is not ... and previous != expected_previous_commit_sha256:
        raise AuditContractError("recovery commitment chain changed")
    if seen_commit_sha256 is not None and logical in seen_commit_sha256:
        raise AuditContractError("recovery commitment replayed")
    return logical


def build_broker_ack(
    commitment: Mapping[str, Any], *, ledger_event_sha256: str,
) -> dict[str, Any]:
    commit_sha = validate_recovery_pre_step_commitment(commitment)
    _require_sha(ledger_event_sha256, "broker ledger event")
    base = {
        "format": BROKER_ACK_FORMAT,
        "status": BROKER_ACK_STATUS,
        "pair_id": commitment["pair_id"],
        "condition_id": commitment["condition_id"],
        "step_index": commitment["step_index"],
        "commit_sequence": commitment["commit_sequence"],
        "recovery_pre_step_commit_sha256": commit_sha,
        "ledger_event_sha256": ledger_event_sha256,
        "step_authorized": True,
    }
    return _signed_document(base, "ack_sha256")


def validate_broker_ack(
    value: Mapping[str, Any], commitment: Mapping[str, Any], *,
    expected_step_index: int | None = None,
    seen_ack_sha256: set[str] | None = None,
) -> str:
    commit_sha = validate_recovery_pre_step_commitment(commitment)
    fields = {
        "format", "status", "pair_id", "condition_id", "step_index",
        "commit_sequence", "recovery_pre_step_commit_sha256",
        "ledger_event_sha256", "step_authorized",
    }
    logical = _verify_document(value, "ack_sha256", fields, "recovery broker ACK")
    if (
        value.get("format") != BROKER_ACK_FORMAT
        or value.get("status") != BROKER_ACK_STATUS
        or value.get("pair_id") != commitment.get("pair_id")
        or value.get("condition_id") != commitment.get("condition_id")
        or value.get("step_index") != commitment.get("step_index")
        or value.get("commit_sequence") != commitment.get("commit_sequence")
        or value.get("recovery_pre_step_commit_sha256") != commit_sha
        or value.get("step_authorized") is not True
    ):
        raise AuditContractError("recovery broker ACK binding changed")
    _require_int(value.get("step_index"), "broker ACK step", minimum=0)
    _require_int(value.get("commit_sequence"), "broker ACK sequence", minimum=0)
    _require_sha(value.get("ledger_event_sha256"), "broker ledger event")
    if expected_step_index is not None and value["step_index"] != expected_step_index:
        raise AuditContractError("broker ACK authorizes the wrong step")
    if seen_ack_sha256 is not None and logical in seen_ack_sha256:
        raise AuditContractError("recovery broker ACK replayed")
    return logical


@dataclass
class RecoveryBrokerState:
    """In-memory fail-closed replay/sequence state for one condition broker."""

    pair_id: str
    condition_id: str
    next_sequence: int = 0
    last_commit_sha256: str | None = None
    pending: Mapping[str, Any] | None = None
    seen_commits: set[str] = field(default_factory=set)
    seen_acks: set[str] = field(default_factory=set)

    def accept_commitment(
        self, commitment: Mapping[str, Any], *, expected_step_index: int,
    ) -> str:
        if self.pending is not None:
            raise AuditContractError("broker already has an unacknowledged commitment")
        logical = validate_recovery_pre_step_commitment(
            commitment,
            expected_pair_id=self.pair_id,
            expected_condition_id=self.condition_id,
            expected_step_index=expected_step_index,
            expected_commit_sequence=self.next_sequence,
            expected_previous_commit_sha256=self.last_commit_sha256,
            seen_commit_sha256=self.seen_commits,
        )
        self.pending = dict(commitment)
        self.seen_commits.add(logical)
        return logical

    def accept_ack(self, ack: Mapping[str, Any], *, expected_step_index: int) -> str:
        if self.pending is None:
            raise AuditContractError("broker ACK has no pending commitment")
        logical = validate_broker_ack(
            ack,
            self.pending,
            expected_step_index=expected_step_index,
            seen_ack_sha256=self.seen_acks,
        )
        self.last_commit_sha256 = str(self.pending["commit_sha256"])
        self.pending = None
        self.next_sequence += 1
        self.seen_acks.add(logical)
        return logical


def target_aad(
    *, protocol_core_v4_sha256: str, pair_id: str, condition_id: str,
    root_prediction_commit_sha256: str, schema6_runtime_contract_sha256: str,
) -> dict[str, Any]:
    for value, role in (
        (protocol_core_v4_sha256, "v4 core"),
        (pair_id, "pair ID"),
        (root_prediction_commit_sha256, "root prediction commit"),
        (schema6_runtime_contract_sha256, "runtime contract"),
    ):
        _require_sha(value, role)
    if not isinstance(condition_id, str) or not condition_id:
        raise AuditContractError("condition ID must be nonempty text")
    return {
        "format": TARGET_AAD_FORMAT,
        "protocol_core_v4_sha256": protocol_core_v4_sha256,
        "pair_id": pair_id,
        "condition_id": condition_id,
        "root_prediction_commit_sha256": root_prediction_commit_sha256,
        "schema6_runtime_contract_sha256": schema6_runtime_contract_sha256,
    }


def generate_x25519_keypair() -> tuple[bytes, bytes]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    except ImportError as error:
        raise AuditContractError("cryptography X25519 is required") from error
    private_key = X25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return private_raw, public_raw


def _derive_envelope_key(shared_secret: bytes, aad_bytes: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError as error:
        raise AuditContractError("cryptography HKDF-SHA256 is required") from error
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=hashlib.sha256(aad_bytes).digest(),
        info=KDF_INFO,
    ).derive(shared_secret)


def seal_target_envelope(
    target: Mapping[str, Any], *, evaluator_public_key_raw: bytes,
    protocol_core_v4_sha256: str, pair_id: str, condition_id: str,
    root_prediction_commit_sha256: str, schema6_runtime_contract_sha256: str,
) -> dict[str, Any]:
    if not isinstance(target, Mapping):
        raise AuditContractError("target plaintext must be a JSON mapping")
    if not isinstance(evaluator_public_key_raw, bytes) or len(
        evaluator_public_key_raw
    ) != 32:
        raise AuditContractError("evaluator X25519 public key must be exact 32 bytes")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey, X25519PublicKey,
        )
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError as error:
        raise AuditContractError("cryptography X25519/ChaCha20Poly1305 is required") from error
    aad = target_aad(
        protocol_core_v4_sha256=protocol_core_v4_sha256,
        pair_id=pair_id,
        condition_id=condition_id,
        root_prediction_commit_sha256=root_prediction_commit_sha256,
        schema6_runtime_contract_sha256=schema6_runtime_contract_sha256,
    )
    aad_bytes = canonical_bytes(aad)
    plaintext = canonical_bytes(target)
    ephemeral_private = X25519PrivateKey.generate()
    evaluator_public = X25519PublicKey.from_public_bytes(evaluator_public_key_raw)
    key = _derive_envelope_key(
        ephemeral_private.exchange(evaluator_public), aad_bytes
    )
    nonce = os.urandom(12)
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad_bytes)
    ephemeral_public = ephemeral_private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    base = {
        "format": TARGET_ENVELOPE_FORMAT,
        "status": TARGET_ENVELOPE_STATUS,
        "encryption_algorithm": ENCRYPTION_ALGORITHM,
        "kdf_info_sha256": hashlib.sha256(KDF_INFO).hexdigest(),
        "protocol_core_v4_sha256": protocol_core_v4_sha256,
        "pair_id": pair_id,
        "condition_id": condition_id,
        "root_prediction_commit_sha256": root_prediction_commit_sha256,
        "schema6_runtime_contract_sha256": schema6_runtime_contract_sha256,
        "aad_sha256": hashlib.sha256(aad_bytes).hexdigest(),
        "evaluator_public_key_sha256": hashlib.sha256(
            evaluator_public_key_raw
        ).hexdigest(),
        "ephemeral_public_key_x25519_hex": ephemeral_public.hex(),
        "nonce_hex": nonce.hex(),
        "ciphertext_hex": ciphertext.hex(),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
    }
    return _signed_document(base, "envelope_sha256")


def validate_target_envelope(
    value: Mapping[str, Any], *,
    expected_protocol_core_v4_sha256: str | None = None,
    expected_pair_id: str | None = None,
    expected_condition_id: str | None = None,
    expected_root_prediction_commit_sha256: str | None = None,
    expected_schema6_runtime_contract_sha256: str | None = None,
) -> tuple[str, dict[str, Any]]:
    fields = {
        "format", "status", "encryption_algorithm", "kdf_info_sha256",
        "protocol_core_v4_sha256", "pair_id", "condition_id",
        "root_prediction_commit_sha256", "schema6_runtime_contract_sha256",
        "aad_sha256", "evaluator_public_key_sha256",
        "ephemeral_public_key_x25519_hex", "nonce_hex", "ciphertext_hex",
        "ciphertext_sha256",
    }
    logical = _verify_document(value, "envelope_sha256", fields, "target envelope")
    if (
        value.get("format") != TARGET_ENVELOPE_FORMAT
        or value.get("status") != TARGET_ENVELOPE_STATUS
        or value.get("encryption_algorithm") != ENCRYPTION_ALGORITHM
        or value.get("kdf_info_sha256") != hashlib.sha256(KDF_INFO).hexdigest()
    ):
        raise AuditContractError("target envelope algorithm changed")
    forbidden = {"target_sha256", "plaintext_sha256", "target_logical_sha256"}
    if forbidden & set(value):
        raise AuditContractError("target envelope leaks a plaintext target SHA")
    for field_name in (
        "protocol_core_v4_sha256", "pair_id", "root_prediction_commit_sha256",
        "schema6_runtime_contract_sha256", "aad_sha256",
        "evaluator_public_key_sha256", "ciphertext_sha256",
    ):
        _require_sha(value.get(field_name), f"target envelope {field_name}")
    if not isinstance(value.get("condition_id"), str) or not value["condition_id"]:
        raise AuditContractError("target envelope condition is invalid")
    try:
        ephemeral = bytes.fromhex(str(value.get("ephemeral_public_key_x25519_hex")))
        nonce = bytes.fromhex(str(value.get("nonce_hex")))
        ciphertext = bytes.fromhex(str(value.get("ciphertext_hex")))
    except ValueError as error:
        raise AuditContractError("target envelope hex is invalid") from error
    if len(ephemeral) != 32 or len(nonce) != 12 or len(ciphertext) < 16:
        raise AuditContractError("target envelope crypto lengths changed")
    if hashlib.sha256(ciphertext).hexdigest() != value["ciphertext_sha256"]:
        raise AuditContractError("target envelope ciphertext SHA mismatch")
    expected = {
        "protocol_core_v4_sha256": expected_protocol_core_v4_sha256,
        "pair_id": expected_pair_id,
        "condition_id": expected_condition_id,
        "root_prediction_commit_sha256": expected_root_prediction_commit_sha256,
        "schema6_runtime_contract_sha256": (
            expected_schema6_runtime_contract_sha256
        ),
    }
    for field_name, expected_value in expected.items():
        if expected_value is not None and value.get(field_name) != expected_value:
            raise AuditContractError(f"target envelope AAD binding changed: {field_name}")
    aad = target_aad(
        protocol_core_v4_sha256=str(value["protocol_core_v4_sha256"]),
        pair_id=str(value["pair_id"]),
        condition_id=str(value["condition_id"]),
        root_prediction_commit_sha256=str(value["root_prediction_commit_sha256"]),
        schema6_runtime_contract_sha256=str(
            value["schema6_runtime_contract_sha256"]
        ),
    )
    if hashlib.sha256(canonical_bytes(aad)).hexdigest() != value["aad_sha256"]:
        raise AuditContractError("target envelope AAD SHA mismatch")
    return logical, aad


def build_terminal_completeness(
    *, terminal_receipt_sha256: str,
    complete_condition_count: int = CONDITION_COUNT,
    complete_pair_count: int = PAIR_COUNT,
    retry_count: int = 0,
    incomplete_count: int = 0,
    exclusion_count: int = 0,
) -> dict[str, Any]:
    _require_sha(terminal_receipt_sha256, "terminal receipt")
    for value, role in (
        (complete_condition_count, "complete condition count"),
        (complete_pair_count, "complete pair count"),
        (retry_count, "retry count"),
        (incomplete_count, "incomplete count"),
        (exclusion_count, "exclusion count"),
    ):
        _require_int(value, role, minimum=0)
    status = (
        COMPLETENESS_STATUS
        if complete_condition_count == CONDITION_COUNT
        and complete_pair_count == PAIR_COUNT
        and retry_count == incomplete_count == exclusion_count == 0
        else "incomplete_decryption_forbidden"
    )
    base = {
        "format": COMPLETENESS_FORMAT,
        "status": status,
        "required_condition_count": CONDITION_COUNT,
        "complete_condition_count": complete_condition_count,
        "required_pair_count": PAIR_COUNT,
        "complete_pair_count": complete_pair_count,
        "retry_count": retry_count,
        "incomplete_count": incomplete_count,
        "exclusion_count": exclusion_count,
        "terminal_receipt_sha256": terminal_receipt_sha256,
    }
    return _signed_document(base, "completeness_sha256")


def validate_terminal_completeness(value: Mapping[str, Any]) -> str:
    fields = {
        "format", "status", "required_condition_count",
        "complete_condition_count", "required_pair_count", "complete_pair_count",
        "retry_count", "incomplete_count", "exclusion_count",
        "terminal_receipt_sha256",
    }
    logical = _verify_document(
        value, "completeness_sha256", fields, "terminal completeness gate"
    )
    if value.get("format") != COMPLETENESS_FORMAT:
        raise AuditContractError("terminal completeness format changed")
    for field_name, expected in (
        ("required_condition_count", CONDITION_COUNT),
        ("complete_condition_count", CONDITION_COUNT),
        ("required_pair_count", PAIR_COUNT),
        ("complete_pair_count", PAIR_COUNT),
        ("retry_count", 0),
        ("incomplete_count", 0),
        ("exclusion_count", 0),
    ):
        _require_int(value.get(field_name), field_name, expected=expected)
    _require_sha(value.get("terminal_receipt_sha256"), "terminal receipt")
    if value.get("status") != COMPLETENESS_STATUS:
        raise AuditContractError("terminal set is incomplete; decryption is forbidden")
    return logical


def open_target_envelope(
    value: Mapping[str, Any], *, evaluator_private_key_raw: bytes,
    terminal_completeness: Mapping[str, Any],
    expected_protocol_core_v4_sha256: str | None = None,
    expected_pair_id: str | None = None,
    expected_condition_id: str | None = None,
    expected_root_prediction_commit_sha256: str | None = None,
    expected_schema6_runtime_contract_sha256: str | None = None,
) -> dict[str, Any]:
    # The terminal gate is intentionally evaluated before any crypto parsing.
    validate_terminal_completeness(terminal_completeness)
    _logical, aad = validate_target_envelope(
        value,
        expected_protocol_core_v4_sha256=expected_protocol_core_v4_sha256,
        expected_pair_id=expected_pair_id,
        expected_condition_id=expected_condition_id,
        expected_root_prediction_commit_sha256=(
            expected_root_prediction_commit_sha256
        ),
        expected_schema6_runtime_contract_sha256=(
            expected_schema6_runtime_contract_sha256
        ),
    )
    if not isinstance(evaluator_private_key_raw, bytes) or len(
        evaluator_private_key_raw
    ) != 32:
        raise AuditContractError("evaluator X25519 private key must be exact 32 bytes")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey, X25519PublicKey,
        )
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        private_key = X25519PrivateKey.from_private_bytes(evaluator_private_key_raw)
        evaluator_public_raw = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        if hashlib.sha256(evaluator_public_raw).hexdigest() != value[
            "evaluator_public_key_sha256"
        ]:
            raise AuditContractError("evaluator private key differs from envelope authority")
        ephemeral = X25519PublicKey.from_public_bytes(
            bytes.fromhex(str(value["ephemeral_public_key_x25519_hex"]))
        )
        aad_bytes = canonical_bytes(aad)
        key = _derive_envelope_key(private_key.exchange(ephemeral), aad_bytes)
        plaintext = ChaCha20Poly1305(key).decrypt(
            bytes.fromhex(str(value["nonce_hex"])),
            bytes.fromhex(str(value["ciphertext_hex"])),
            aad_bytes,
        )
        decoded = json.loads(plaintext.decode("utf-8"))
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        raise AuditContractError("sealed target authentication/decryption failed") from error
    except Exception as error:
        # cryptography raises InvalidTag, which deliberately need not leak detail.
        raise AuditContractError("sealed target authentication/decryption failed") from error
    if not isinstance(decoded, Mapping):
        raise AuditContractError("decrypted target is not a JSON mapping")
    return dict(decoded)


def recompute_audit_targets(
    trace: Mapping[str, Any], *,
    dense_event_targets_fn: Callable[[Sequence[str], Sequence[int], int], Mapping[str, Any]],
    recovery_targets_fn: Callable[..., Mapping[str, Any]],
    object_target_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute targets from in-memory trace via authority-injected semantics.

    ``object_target_fn`` receives ``object_trace``, ``start_step=0`` and
    ``end_step=1`` and must return exactly ``object_target`` (1-D physical xyz)
    and exact-bool ``object_observed``.
    """

    expected = {
        "event_names", "event_steps", "terminal_step", "terminal_success",
        "trajectory_success", "right_censored", "object_trace",
    }
    if not isinstance(trace, Mapping) or set(trace) != expected:
        raise AuditContractError("in-memory target trace fields changed")
    terminal_step = _require_int(trace.get("terminal_step"), "terminal step", minimum=1)
    terminal_success = trace.get("terminal_success")
    right_censored = trace.get("right_censored")
    if type(terminal_success) is not bool or type(right_censored) is not bool:
        raise AuditContractError("terminal success/censoring must be exact bool")
    event_names = trace.get("event_names")
    event_steps = trace.get("event_steps")
    trajectory_success = trace.get("trajectory_success")
    if (
        not isinstance(event_names, list)
        or not event_names
        or any(not isinstance(item, str) or not item for item in event_names)
        or not isinstance(event_steps, list)
        or len(event_steps) != len(event_names)
        or any(type(item) is not int for item in event_steps)
        or event_steps != sorted(event_steps)
        or event_steps[0] != 0
        or event_steps[-1] > terminal_step
        or not isinstance(trajectory_success, list)
        or len(trajectory_success) != terminal_step + 1
        or any(type(item) is not bool for item in trajectory_success)
        or trajectory_success[0] is not False
        or any(
            before and not after
            for before, after in zip(trajectory_success, trajectory_success[1:])
        )
        or trajectory_success[-1] is not terminal_success
    ):
        raise AuditContractError("in-memory event/success trace changed")
    dense = dense_event_targets_fn(event_names, event_steps, terminal_step)
    expected_dense = {
        "trajectory_event_id", "transition_next_event_id",
        "transition_duration_decision_steps", "transition_duration_observed",
        "transition_duration_censored",
    }
    if not isinstance(dense, Mapping) or set(dense) != expected_dense:
        raise AuditContractError("injected dense-event target fields changed")
    events = np.asarray(dense["trajectory_event_id"])
    next_event = np.asarray(dense["transition_next_event_id"])
    duration = np.asarray(dense["transition_duration_decision_steps"])
    duration_observed = np.asarray(dense["transition_duration_observed"])
    duration_censored = np.asarray(dense["transition_duration_censored"])
    if (
        events.shape != (terminal_step + 1,)
        or next_event.shape != (terminal_step,)
        or duration.shape != (terminal_step,)
        or duration_observed.shape != (terminal_step,)
        or duration_censored.shape != (terminal_step,)
        or duration_observed.dtype != np.dtype(bool)
        or duration_censored.dtype != np.dtype(bool)
        or not np.array_equal(~duration_observed, duration_censored)
        or not np.issubdtype(events.dtype, np.integer)
        or not np.issubdtype(next_event.dtype, np.integer)
        or not np.issubdtype(duration.dtype, np.number)
        or not np.isfinite(duration.astype(np.float64)).all()
        or bool((duration.astype(np.float64) < 0.0).any())
    ):
        raise AuditContractError("injected dense-event target shapes changed")
    recovery = recovery_targets_fn(events, right_censored=right_censored)
    if not isinstance(recovery, Mapping) or set(recovery) != {
        "regress", "recovery", "recovery_observed"
    }:
        raise AuditContractError("injected recovery target fields changed")
    regress = np.asarray(recovery["regress"])
    recovery_value = np.asarray(recovery["recovery"])
    recovery_observed = np.asarray(recovery["recovery_observed"])
    if (
        regress.shape != (terminal_step,)
        or recovery_value.shape != (terminal_step,)
        or recovery_observed.shape != (terminal_step,)
        or regress.dtype != np.dtype(bool)
        or recovery_observed.dtype != np.dtype(bool)
        or not np.isin(recovery_value, [0, 1]).all()
        or bool((recovery_observed & ~regress).any())
        or bool((recovery_value.astype(bool) & ~recovery_observed).any())
    ):
        raise AuditContractError("injected recovery masks changed")
    object_result = object_target_fn(
        trace["object_trace"], start_step=0, end_step=1
    )
    if not isinstance(object_result, Mapping) or set(object_result) != {
        "object_target", "object_observed"
    }:
        raise AuditContractError("injected object target fields changed")
    object_target = np.asarray(object_result["object_target"], dtype=np.float64)
    if (
        object_target.ndim != 1
        or object_target.size < 1
        or not np.isfinite(object_target).all()
        or type(object_result["object_observed"]) is not bool
    ):
        raise AuditContractError("injected physical object target changed")
    regress_indices = np.flatnonzero(regress)
    if len(regress_indices):
        recovery_step = int(regress_indices[0])
        observed = bool(recovery_observed[recovery_step])
        recovery_row = {
            "status": "observed" if observed else "right_censored",
            "applicable": True,
            "observed": observed,
            "censored": not observed,
            "step_index": recovery_step,
            "target": int(recovery_value[recovery_step]) if observed else None,
        }
    else:
        recovery_row = {
            "status": "not_applicable_no_operational_regress",
            "applicable": False,
            "observed": False,
            "censored": False,
            "step_index": None,
            "target": None,
        }
    return {
        "root": {
            "post_event": int(events[1]),
            "next_event": int(next_event[0]),
            "duration_decision_steps": float(duration[0]),
            "duration_observed": bool(duration_observed[0]),
            "duration_censored": bool(duration_censored[0]),
            "success": int(terminal_success),
            "object_target_physical_xyz_m": object_target.tolist(),
            "object_observed": bool(object_result["object_observed"]),
            "recovery": {
                "status": "not_applicable",
                "policy": ROOT_RECOVERY_POLICY,
            },
        },
        "first_operational_regress": recovery_row,
    }


def _mask(value: Any, length: int, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype != np.dtype(bool):
        raise AuditContractError(f"{role} must be exact bool [{length}]")
    return array


def _pair_ids(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) == 0:
        raise AuditContractError("pair IDs must be a nonempty 1-D sequence")
    normalized = np.asarray([str(item) for item in array], dtype=str)
    if any(not item for item in normalized):
        raise AuditContractError("pair IDs must be nonempty")
    return normalized


def _numeric_array(value: Any, shape: tuple[int, ...], role: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype == np.dtype(bool) or raw.shape != shape or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise AuditContractError(f"{role} numeric shape changed")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise AuditContractError(f"{role} contains non-finite values")
    return result


def _targets(value: Any, length: int, role: str, *, classes: int = 2) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype == np.dtype(bool) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise AuditContractError(f"{role} targets must be exact integer labels")
    result = raw.astype(np.int64)
    if bool(((result < 0) | (result >= classes)).any()):
        raise AuditContractError(f"{role} targets are out of range")
    return result


def _pair_means(
    values: np.ndarray, pair_ids: np.ndarray, observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(sorted(set(pair_ids[observed].tolist())), dtype=str)
    means = np.asarray(
        [values[observed & (pair_ids == name)].mean() for name in names],
        dtype=np.float64,
    )
    return names, means


def pair_cluster_bootstrap(
    values: Any, pair_ids: Any, *, observed: Any | None = None,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    ids = _pair_ids(pair_ids)
    data = _numeric_array(values, (len(ids),), "bootstrap values")
    mask = np.ones(len(ids), dtype=bool) if observed is None else _mask(
        observed, len(ids), "bootstrap observation mask"
    )
    _require_int(samples, "bootstrap samples", minimum=100)
    _require_int(seed, "bootstrap seed", minimum=0)
    names, means = _pair_means(data, ids, mask)
    if len(means) == 0:
        return {
            "status": "insufficient_support", "unit": "pair_id",
            "pair_count": 0, "estimate": None, "lower": None, "upper": None,
            "samples": samples, "seed": seed,
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    block = 1000
    for start in range(0, samples, block):
        count = min(block, samples - start)
        indices = rng.integers(0, len(means), size=(count, len(means)))
        draws[start:start + count] = means[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "status": "complete",
        "unit": "pair_id",
        "pair_count": int(len(names)),
        "estimate": float(means.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "samples": samples,
        "seed": seed,
    }


def _mean_by_pair(values: np.ndarray, ids: np.ndarray, observed: np.ndarray) -> float | None:
    _names, means = _pair_means(values, ids, observed)
    return float(means.mean()) if len(means) else None


def _gain_ci(
    baseline_loss: np.ndarray, model_loss: np.ndarray, ids: np.ndarray,
    observed: np.ndarray, *, samples: int, seed: int,
) -> dict[str, Any]:
    return pair_cluster_bootstrap(
        baseline_loss - model_loss, ids, observed=observed,
        samples=samples, seed=seed,
    )


def multiclass_metrics(
    probability: Any, target: Any, pair_ids: Any, *, baseline_probability: Any,
    observed: Any, required_classes: Sequence[int] | None = None,
    minimum_pairs_per_class: int, bootstrap_samples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    ids = _pair_ids(pair_ids)
    n = len(ids)
    probability_raw = np.asarray(probability)
    baseline_raw = np.asarray(baseline_probability)
    if (
        probability_raw.dtype == np.dtype(bool)
        or baseline_raw.dtype == np.dtype(bool)
        or not np.issubdtype(probability_raw.dtype, np.number)
        or not np.issubdtype(baseline_raw.dtype, np.number)
    ):
        raise AuditContractError("multiclass probability must be numeric, not bool")
    prob = probability_raw.astype(np.float64)
    baseline = baseline_raw.astype(np.float64)
    if (
        prob.ndim != 2
        or prob.shape[0] != n
        or prob.shape[1] < 2
        or baseline.shape != prob.shape
        or not np.isfinite(prob).all()
        or not np.isfinite(baseline).all()
        or bool((prob < 0.0).any())
        or bool((baseline < 0.0).any())
        or not np.allclose(prob.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10)
        or not np.allclose(baseline.sum(axis=1), 1.0, rtol=1e-10, atol=1e-10)
    ):
        raise AuditContractError("multiclass probability contract changed")
    truth = _targets(target, n, "multiclass", classes=prob.shape[1])
    mask = _mask(observed, n, "multiclass observation mask")
    required = (
        list(range(prob.shape[1]))
        if required_classes is None
        else list(required_classes)
    )
    if (
        not required
        or any(type(label) is not int for label in required)
        or len(set(required)) != len(required)
        or any(label < 0 or label >= prob.shape[1] for label in required)
    ):
        raise AuditContractError("required multiclass label inventory changed")
    model_nll = -np.log(np.clip(prob[np.arange(n), truth], 1e-12, 1.0))
    baseline_nll = -np.log(
        np.clip(baseline[np.arange(n), truth], 1e-12, 1.0)
    )
    correct = (prob.argmax(axis=1) == truth).astype(np.float64)
    baseline_correct = (baseline.argmax(axis=1) == truth).astype(np.float64)
    class_pair_support = {
        str(label): len(set(ids[mask & (truth == label)].tolist()))
        for label in required
    }
    status = (
        "complete"
        if min(class_pair_support.values(), default=0) >= minimum_pairs_per_class
        else "insufficient_support"
    )
    return {
        "status": status,
        "applicable_count": int(mask.sum()),
        "observed_count": int(mask.sum()),
        "censored_count": 0,
        "pair_count": len(set(ids[mask].tolist())),
        "required_classes": required,
        "class_pair_support": class_pair_support,
        "equal_pair_nll": _mean_by_pair(model_nll, ids, mask),
        "equal_pair_accuracy": _mean_by_pair(correct, ids, mask),
        "nll_gain_baseline_minus_model": _gain_ci(
            baseline_nll, model_nll, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed,
        ),
        "accuracy_gain_model_minus_baseline": pair_cluster_bootstrap(
            correct - baseline_correct, ids, observed=mask,
            samples=bootstrap_samples, seed=bootstrap_seed + 1,
        ),
    }


def binary_metrics(
    probability: Any, target: Any, pair_ids: Any, *, baseline_probability: Any,
    applicable: Any, observed: Any, minimum_pairs_per_class: int,
    bootstrap_samples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    ids = _pair_ids(pair_ids)
    n = len(ids)
    prob = _numeric_array(probability, (n,), "binary probability")
    baseline = _numeric_array(
        baseline_probability, (n,), "binary baseline probability"
    )
    if bool(((prob < 0.0) | (prob > 1.0)).any()) or bool(
        ((baseline < 0.0) | (baseline > 1.0)).any()
    ):
        raise AuditContractError("binary probabilities are outside [0,1]")
    truth = _targets(target, n, "binary", classes=2)
    applies = _mask(applicable, n, "binary applicability mask")
    mask = _mask(observed, n, "binary observation mask")
    if bool((mask & ~applies).any()):
        raise AuditContractError("binary observed mask escaped applicability")
    clipped = np.clip(prob, 1e-12, 1.0 - 1e-12)
    baseline_clipped = np.clip(baseline, 1e-12, 1.0 - 1e-12)
    nll = -(truth * np.log(clipped) + (1 - truth) * np.log(1.0 - clipped))
    baseline_nll = -(
        truth * np.log(baseline_clipped)
        + (1 - truth) * np.log(1.0 - baseline_clipped)
    )
    brier = np.square(prob - truth)
    baseline_brier = np.square(baseline - truth)
    support = {
        "positive_pairs": len(set(ids[mask & (truth == 1)].tolist())),
        "negative_pairs": len(set(ids[mask & (truth == 0)].tolist())),
    }
    status = (
        "complete"
        if min(support.values()) >= minimum_pairs_per_class
        else "insufficient_support"
    )
    return {
        "status": status,
        "applicable_count": int(applies.sum()),
        "observed_count": int(mask.sum()),
        "censored_count": int((applies & ~mask).sum()),
        "pair_count": len(set(ids[mask].tolist())),
        **support,
        "equal_pair_nll": _mean_by_pair(nll, ids, mask),
        "equal_pair_brier": _mean_by_pair(brier, ids, mask),
        "nll_gain_baseline_minus_model": _gain_ci(
            baseline_nll, nll, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed,
        ),
        "brier_gain_baseline_minus_model": _gain_ci(
            baseline_brier, brier, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed + 1,
        ),
    }


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)),
        axis=axis,
    )


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf)(value / math.sqrt(2.0)) * 0.5 + 0.5


def _duration_quantile(
    means: np.ndarray, scales: np.ndarray, probability: float,
) -> np.ndarray:
    lower = np.min(means - 10.0 * scales, axis=1)
    upper = np.max(means + 10.0 * scales, axis=1)
    for _ in range(70):
        middle = (lower + upper) / 2.0
        cdf = _normal_cdf(
            (middle[:, None] - means) / scales
        ).mean(axis=1)
        lower = np.where(cdf < probability, middle, lower)
        upper = np.where(cdf >= probability, middle, upper)
    return np.maximum(np.exp(np.clip((lower + upper) / 2.0, -30, 30)) - 1.0, 0.0)


def duration_metrics(
    member_log_mean: Any, member_log_scale: Any, target: Any, pair_ids: Any, *,
    observed: Any, baseline_location: Any, baseline_scale: Any,
    minimum_observed_and_censored_pairs: int, bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    ids = _pair_ids(pair_ids)
    n = len(ids)
    means = _numeric_array(
        member_log_mean, (n, MEMBER_COUNT), "duration member log mean"
    )
    log_scales = _numeric_array(
        member_log_scale, (n, MEMBER_COUNT), "duration member log scale"
    )
    truth = _numeric_array(target, (n,), "duration target")
    mask = _mask(observed, n, "duration observation mask")
    location = _numeric_array(
        baseline_location, (n,), "duration baseline location"
    )
    baseline_scale_array = _numeric_array(
        baseline_scale, (n,), "duration baseline scale"
    )
    if bool((truth < 0.0).any()) or bool((baseline_scale_array <= 0.0).any()):
        raise AuditContractError("duration target/baseline scale changed")
    scales = np.exp(np.clip(log_scales, -8.0, 5.0))
    log_target = np.log1p(truth)
    log_pdf = (
        -0.5 * np.square((log_target[:, None] - means) / scales)
        - np.log(scales)
        - 0.5 * math.log(2.0 * math.pi)
        - log_target[:, None]
    )
    nll = -(_logsumexp(log_pdf, axis=1) - math.log(MEMBER_COUNT))
    median = _duration_quantile(means, scales, 0.5)
    lower = _duration_quantile(means, scales, 0.05)
    upper = _duration_quantile(means, scales, 0.95)
    absolute_error = np.abs(median - truth)
    coverage = ((truth >= lower) & (truth <= upper)).astype(np.float64)
    baseline_error = np.abs(location - truth)
    baseline_nll = np.log(2.0 * baseline_scale_array) + np.abs(
        truth - location
    ) / baseline_scale_array
    observed_pairs = len(set(ids[mask].tolist()))
    censored_pairs = len(set(ids[~mask].tolist()))
    status = (
        "complete"
        if min(observed_pairs, censored_pairs) >= minimum_observed_and_censored_pairs
        else "insufficient_support"
    )
    return {
        "status": status,
        "applicable_count": n,
        "observed_count": int(mask.sum()),
        "censored_count": int((~mask).sum()),
        "observed_pair_count": observed_pairs,
        "censored_pair_count": censored_pairs,
        "equal_pair_mixture_nll_observed": _mean_by_pair(nll, ids, mask),
        "equal_pair_median_mae_observed": _mean_by_pair(
            absolute_error, ids, mask
        ),
        "equal_pair_central_90_coverage_observed": _mean_by_pair(
            coverage, ids, mask
        ),
        "nll_gain_baseline_minus_model": _gain_ci(
            baseline_nll, nll, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed,
        ),
        "mae_gain_baseline_minus_model": _gain_ci(
            baseline_error, absolute_error, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed + 1,
        ),
    }


def object_effect_metrics(
    member_mean: Any, member_log_scale: Any, target: Any, pair_ids: Any, *,
    observed: Any, baseline_robust: Any, minimum_nonzero_and_near_zero_pairs: int,
    bootstrap_samples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    ids = _pair_ids(pair_ids)
    n = len(ids)
    means_raw = np.asarray(member_mean)
    if means_raw.ndim != 3 or means_raw.shape[:2] != (n, MEMBER_COUNT):
        raise AuditContractError("object member mean shape changed")
    object_dim = means_raw.shape[2]
    means = _numeric_array(
        member_mean, (n, MEMBER_COUNT, object_dim), "object member mean"
    )
    log_scales = _numeric_array(
        member_log_scale, (n, MEMBER_COUNT, object_dim),
        "object member log scale",
    )
    truth = _numeric_array(target, (n, object_dim), "object target")
    robust = _numeric_array(
        baseline_robust, (n, object_dim), "object robust baseline"
    )
    mask = _mask(observed, n, "object observation mask")
    scales = np.exp(np.clip(log_scales, -10.0, 10.0))
    log_pdf = (
        -0.5 * np.square((truth[:, None, :] - means) / scales)
        - np.log(scales)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(axis=2)
    nll = -(_logsumexp(log_pdf, axis=1) - math.log(MEMBER_COUNT))
    prediction = means.mean(axis=1)
    aleatoric = np.square(scales).mean(axis=1)
    epistemic = means.var(axis=1)
    radius = 1.6448536269514722 * np.sqrt(
        np.maximum(aleatoric + epistemic, 0.0)
    )
    covered = (truth >= prediction - radius) & (truth <= prediction + radius)
    l2 = np.linalg.norm(prediction - truth, axis=1)
    rmse = np.sqrt(np.mean(np.square(prediction - truth), axis=1))
    zero_l2 = np.linalg.norm(truth, axis=1)
    robust_l2 = np.linalg.norm(robust - truth, axis=1)
    nonzero = np.linalg.norm(truth, axis=1) > 1e-6
    nonzero_pairs = len(set(ids[mask & nonzero].tolist()))
    near_zero_pairs = len(set(ids[mask & ~nonzero].tolist()))
    status = (
        "complete"
        if min(nonzero_pairs, near_zero_pairs)
        >= minimum_nonzero_and_near_zero_pairs
        else "insufficient_support"
    )
    return {
        "status": status,
        "applicable_count": int(mask.sum()),
        "observed_count": int(mask.sum()),
        "censored_count": 0,
        "missing_count": int((~mask).sum()),
        "pair_count": len(set(ids[mask].tolist())),
        "nonzero_pair_count": nonzero_pairs,
        "near_zero_pair_count": near_zero_pairs,
        "equal_pair_mixture_nll": _mean_by_pair(nll, ids, mask),
        "equal_pair_l2": _mean_by_pair(l2, ids, mask),
        "equal_pair_rmse": _mean_by_pair(rmse, ids, mask),
        "equal_pair_central_90_marginal_coverage": _mean_by_pair(
            covered.astype(np.float64).mean(axis=1), ids, mask
        ),
        "equal_pair_central_90_joint_coverage": _mean_by_pair(
            covered.all(axis=1).astype(np.float64), ids, mask
        ),
        "zero_l2_gain_baseline_minus_model": _gain_ci(
            zero_l2, l2, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed,
        ),
        "robust_l2_gain_baseline_minus_model": _gain_ci(
            robust_l2, l2, ids, mask,
            samples=bootstrap_samples, seed=bootstrap_seed + 1,
        ),
    }


def compute_six_head_metrics(
    value: Mapping[str, Any], *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute the fixed six-head, equal-pair sealed-audit metric family."""

    expected = {
        "pair_id", "post_event", "next_event", "duration", "success",
        "recovery", "object_effect",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise AuditContractError("six-head metric input fields changed")
    ids = _pair_ids(value["pair_id"])
    n = len(ids)
    _require_int(bootstrap_samples, "bootstrap samples", minimum=100)
    _require_int(bootstrap_seed, "bootstrap seed", minimum=0)

    def exact_head(name: str, fields: set[str]) -> Mapping[str, Any]:
        row = value.get(name)
        if not isinstance(row, Mapping) or set(row) != fields:
            raise AuditContractError(f"{name} metric fields changed")
        return row

    post = exact_head(
        "post_event",
        {"probability", "target", "baseline_probability", "observed",
         "required_classes"},
    )
    next_event = exact_head(
        "next_event",
        {"probability", "target", "baseline_probability", "observed",
         "required_classes"},
    )
    success = exact_head(
        "success", {"probability", "target", "baseline_probability", "observed"}
    )
    recovery = exact_head(
        "recovery",
        {"probability", "target", "baseline_probability", "applicable", "observed"},
    )
    duration = exact_head(
        "duration",
        {"member_log_mean", "member_log_scale", "target", "observed",
         "baseline_location", "baseline_scale"},
    )
    object_effect = exact_head(
        "object_effect",
        {"member_mean", "member_log_scale", "target", "observed",
         "baseline_robust"},
    )
    all_applicable = np.ones(n, dtype=bool)
    metrics = {
        "post_event": multiclass_metrics(
            post["probability"], post["target"], ids,
            baseline_probability=post["baseline_probability"],
            observed=post["observed"],
            required_classes=post["required_classes"],
            minimum_pairs_per_class=HEAD_MINIMUM_PAIR_SUPPORT["post_event"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 10,
        ),
        "next_event": multiclass_metrics(
            next_event["probability"], next_event["target"], ids,
            baseline_probability=next_event["baseline_probability"],
            observed=next_event["observed"],
            required_classes=next_event["required_classes"],
            minimum_pairs_per_class=HEAD_MINIMUM_PAIR_SUPPORT["next_event"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 20,
        ),
        "duration": duration_metrics(
            duration["member_log_mean"], duration["member_log_scale"],
            duration["target"], ids, observed=duration["observed"],
            baseline_location=duration["baseline_location"],
            baseline_scale=duration["baseline_scale"],
            minimum_observed_and_censored_pairs=(
                HEAD_MINIMUM_PAIR_SUPPORT["duration"]
            ),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 30,
        ),
        "success": binary_metrics(
            success["probability"], success["target"], ids,
            baseline_probability=success["baseline_probability"],
            applicable=all_applicable, observed=success["observed"],
            minimum_pairs_per_class=HEAD_MINIMUM_PAIR_SUPPORT["success"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 40,
        ),
        "recovery": binary_metrics(
            recovery["probability"], recovery["target"], ids,
            baseline_probability=recovery["baseline_probability"],
            applicable=recovery["applicable"], observed=recovery["observed"],
            minimum_pairs_per_class=HEAD_MINIMUM_PAIR_SUPPORT["recovery"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 50,
        ),
        "object_effect": object_effect_metrics(
            object_effect["member_mean"], object_effect["member_log_scale"],
            object_effect["target"], ids, observed=object_effect["observed"],
            baseline_robust=object_effect["baseline_robust"],
            minimum_nonzero_and_near_zero_pairs=(
                HEAD_MINIMUM_PAIR_SUPPORT["object_effect"]
            ),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 60,
        ),
    }
    insufficient = [
        name for name in ALL_HEADS if metrics[name]["status"] != "complete"
    ]
    next_observed = _mask(
        next_event["observed"], n, "next-event observation mask"
    )
    metrics["next_event"]["applicable_count"] = n
    metrics["next_event"]["censored_count"] = int((~next_observed).sum())
    return {
        "format": "etsf_smolvla_piper_evaluation400_six_head_metrics_v1",
        "status": (
            "complete_all_six_heads"
            if not insufficient
            else "insufficient_support"
        ),
        "metric_weighting": "equal_pair_id_after_within_pair_mean",
        "bootstrap": {
            "unit": "pair_id", "samples": bootstrap_samples,
            "base_seed": bootstrap_seed, "interval": "percentile_95_percent",
            "quantile_method": "linear",
        },
        "head_minimum_pair_support": dict(HEAD_MINIMUM_PAIR_SUPPORT),
        "insufficient_support_heads": insufficient,
        "heads": metrics,
    }


__all__ = [
    "ALL_HEADS",
    "AuditContractError",
    "BROKER_ACK_FORMAT",
    "COMPLETENESS_FORMAT",
    "CONDITION_COUNT",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "HEAD_MINIMUM_PAIR_SUPPORT",
    "MEMBER_COUNT",
    "PAIR_COUNT",
    "ROOT_HEADS",
    "ROOT_PRECOMMIT_FORMAT",
    "ROOT_RECOVERY_POLICY",
    "ROOT_TENSOR_FIELDS",
    "RecoveryBrokerState",
    "TARGET_ENVELOPE_FORMAT",
    "binary_metrics",
    "build_broker_ack",
    "build_recovery_pre_step_commitment",
    "build_root_precommit",
    "build_root_tensor_commitment",
    "build_terminal_completeness",
    "canonical_bytes",
    "canonical_sha256",
    "compute_six_head_metrics",
    "duration_metrics",
    "generate_x25519_keypair",
    "multiclass_metrics",
    "object_effect_metrics",
    "open_target_envelope",
    "pair_cluster_bootstrap",
    "recompute_audit_targets",
    "seal_target_envelope",
    "target_aad",
    "validate_broker_ack",
    "validate_recovery_pre_step_commitment",
    "validate_root_precommit",
    "validate_root_tensor_commitment",
    "validate_target_envelope",
    "validate_terminal_completeness",
]
