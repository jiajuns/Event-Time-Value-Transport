#!/usr/bin/env python3
"""Content-addressed root-observation and prediction-derivation contracts.

The module is deliberately pure: it does not launch/reset/step a simulator,
load checkpoints, read target data, or accept direct event/predicate inputs.
It binds material already produced by a frozen causal observer and a realized
five-member root predictor.  Validators require the original numeric arrays so
a self-reported JSON digest is insufficient.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

import smolvla_piper_evaluation400_audit_contract_v1 as audit
import smolvla_piper_causal_event_observer_v1 as observer


ROOT_SCOPE = "__evaluation400_root_precommit__"
OBSERVATION_FORMAT = "etsf_smolvla_piper_root_actor_visible_observation_v1"
OBSERVATION_STATUS = "root_actor_visible_observer_committed_before_prediction"
ACTOR_INPUT_FORMAT = "etsf_smolvla_piper_root_actor_visible_input_v1"
CANDIDATE_FORMAT = "etsf_smolvla_piper_root_candidate_tensor_registry_v1"
DERIVATION_FORMAT = "etsf_smolvla_piper_root_prediction_derivation_v1"
DERIVATION_STATUS = "five_member_raw_tensors_and_decision_vectors_committed"
OBSERVER_RECEIPT_FORMAT = "etsf_smolvla_piper_causal_observer_query_receipt_v4"

MEMBER_COUNT = 5
STATE_DIM = 960
PROPRIO_DIM = 14
HISTORY_STEPS = 8
PREDICATE_NAMES = ("moved", "lifted", "near_goal", "stationary", "success")
AUXILIARY_TENSOR_FIELDS = (
    "member_calibrated_success_probability",
    "member_composite_rank_score",
    "candidate_structured_five_head_uncertainty",
)


class RootObservedContractError(RuntimeError):
    """The root-observed evidence failed closed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: Any, role: str) -> str:
    if not _is_sha(value):
        raise RootObservedContractError(f"{role} must be exact SHA-256")
    return str(value)


def _require_int(value: Any, role: str, *, expected: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise RootObservedContractError(f"{role} must be a non-negative integer")
    if expected is not None and value != expected:
        raise RootObservedContractError(f"{role} differs from frozen chronology")
    return value


def _require_float(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RootObservedContractError(f"{role} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RootObservedContractError(f"{role} must be finite numeric")
    return result


def _exact(value: Any, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RootObservedContractError(f"{role} fields changed")
    return value


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    normalized = dict(base)
    return {**normalized, field: canonical_sha256(normalized)}


def _verify_signed(
    value: Mapping[str, Any], *, fields: set[str], digest_field: str, role: str,
) -> str:
    _exact(value, fields | {digest_field}, role)
    digest = _require_sha(value.get(digest_field), f"{role} digest")
    logical = {name: child for name, child in value.items() if name != digest_field}
    if digest != canonical_sha256(logical):
        raise RootObservedContractError(f"{role} canonical digest changed")
    return digest


def _float32(value: Any, *, shape: tuple[int, ...] | None, role: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float32)
        or (shape is not None and array.shape != shape)
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise RootObservedContractError(
            f"{role} must be finite C-contiguous native float32 with exact shape"
        )
    return array


def _bool(value: Any, *, shape: tuple[int, ...], role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_) or array.shape != shape or not array.flags.c_contiguous:
        raise RootObservedContractError(
            f"{role} must be C-contiguous native bool with exact shape"
        )
    return array


def _tensor_sha256(name: str, array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"ETSF/evaluation400/root-observed-tensor-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _observer_tensor_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _descriptor(name: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "tensor_sha256": _tensor_sha256(name, array),
    }


def _validate_observer_input_receipt(
    value: Mapping[str, Any], *,
    actor_name: str,
    policy_family: str,
    state_feature_source_sha256: str,
    history: np.ndarray,
    history_mask: np.ndarray,
    proprio: np.ndarray,
    image_feature: np.ndarray | None,
    image_feature_extractor_file_sha256: str | None,
) -> str:
    fields = {
        "format", "status", "actor_name", "policy_family",
        "state_feature_source_sha256", "state_input_dim", "history_sha256",
        "history_mask_sha256", "proprio_sha256", "history_contract_sha256",
        "history_start_query_index", "history_end_query_index",
        "current_query_index", "valid_history_steps", "history_padding",
        "history_truncation", "state_visibility", "proprio_source",
        "object_pose_fields_present", "simulator_privileged_state_read",
        "future_features_read", "image_feature_receipt", "execution_receipt",
    }
    digest = _verify_signed(
        value, fields=fields, digest_field="receipt_sha256",
        role="root observer input receipt",
    )
    image_receipt = value["image_feature_receipt"]
    if image_feature is None:
        image_valid = image_receipt == {"present": False}
    else:
        if not _is_sha(image_feature_extractor_file_sha256):
            raise RootObservedContractError("root image extractor is not frozen")
        image_fields = {
            "present", "format", "extractor_file_sha256", "feature_dim",
            "frame_query_index", "source", "feature_sha256",
        }
        _verify_signed(
            image_receipt, fields=image_fields, digest_field="receipt_sha256",
            role="root image receipt",
        )
        image_valid = (
            image_receipt.get("present") is True
            and image_receipt.get("format") == observer.IMAGE_RECEIPT_FORMAT
            and image_receipt.get("extractor_file_sha256")
            == image_feature_extractor_file_sha256
            and image_receipt.get("feature_dim") == int(image_feature.shape[0])
            and image_receipt.get("frame_query_index") == 0
            and image_receipt.get("source") == "actor_visible_rgb_at_current_query"
            and image_receipt.get("feature_sha256")
            == _observer_tensor_sha256(image_feature)
        )
    expected_mask = np.zeros(HISTORY_STEPS, dtype=np.bool_)
    expected_mask[0] = True
    if (
        value.get("format") != observer.INPUT_RECEIPT_FORMAT
        or value.get("status") != "actor_visible_causal_inputs_only"
        or value.get("actor_name") != actor_name
        or value.get("policy_family") != policy_family
        or value.get("state_feature_source_sha256") != state_feature_source_sha256
        or value.get("state_input_dim") != STATE_DIM
        or value.get("history_sha256") != _observer_tensor_sha256(history)
        or value.get("history_mask_sha256") != _observer_tensor_sha256(history_mask)
        or value.get("proprio_sha256") != _observer_tensor_sha256(proprio)
        or value.get("history_contract_sha256")
        != observer.causal_history_contract()["contract_sha256"]
        or value.get("history_start_query_index") != 0
        or value.get("history_end_query_index") != 0
        or value.get("current_query_index") != 0
        or value.get("valid_history_steps") != 1
        or value.get("history_padding") != observer.HISTORY_PADDING
        or value.get("history_truncation") != observer.HISTORY_TRUNCATION
        or value.get("state_visibility") != observer.STATE_VISIBILITY
        or value.get("proprio_source") != observer.PROPRIO_SOURCE
        or value.get("object_pose_fields_present") is not False
        or value.get("simulator_privileged_state_read") is not False
        or value.get("future_features_read") is not False
        or value.get("execution_receipt") != {"present": False}
        or not image_valid
        or not np.array_equal(history_mask, expected_mask)
        or np.any(history[1:] != np.float32(0.0))
    ):
        raise RootObservedContractError("root observer input receipt is not causal/actor-visible")
    return digest


def _validate_observer_output_receipt(
    value: Mapping[str, Any], *, pair_id: str, observer_authority_sha256: str,
    input_receipt_sha256: str,
) -> str:
    fields = {
        "format", "status", "authority_sha256", "pair_id", "condition_id",
        "step_index", "input_receipt_sha256", "prediction_sha256",
        "calibration_sha256", "minimum_joint_confidence", "current_event_id",
        "current_predicates", "confidence", "applicable",
        "object_pose_fields_present", "simulator_privileged_state_read",
        "hardcoded_event_fallback_used",
    }
    digest = _verify_signed(
        value, fields=fields, digest_field="receipt_sha256",
        role="root observer output receipt",
    )
    confidence = _require_float(value.get("confidence"), "root observer confidence")
    minimum = _require_float(
        value.get("minimum_joint_confidence"), "root observer minimum confidence"
    )
    event_id = _require_int(value.get("current_event_id"), "root observed event")
    predicates = value.get("current_predicates")
    if (
        value.get("format") != OBSERVER_RECEIPT_FORMAT
        or value.get("status") != "actor_visible_promoted_observation_applicable"
        or value.get("authority_sha256") != observer_authority_sha256
        or value.get("pair_id") != pair_id
        or value.get("condition_id") != ROOT_SCOPE
        or value.get("step_index") != 0
        or value.get("input_receipt_sha256") != input_receipt_sha256
        or not _is_sha(value.get("prediction_sha256"))
        or not _is_sha(value.get("calibration_sha256"))
        or event_id >= len(observer.EXPECTED_EVENTS)
        or not isinstance(predicates, Mapping)
        or set(predicates) != set(PREDICATE_NAMES)
        or any(type(child) is not bool for child in predicates.values())
        or not 0.0 <= minimum <= confidence <= 1.0
        or value.get("applicable") is not True
        or value.get("object_pose_fields_present") is not False
        or value.get("simulator_privileged_state_read") is not False
        or value.get("hardcoded_event_fallback_used") is not False
    ):
        raise RootObservedContractError("root observer output failed confidence/provenance gate")
    return digest


def _observation_chronology(value: Any) -> dict[str, int]:
    fields = {
        "root_reset_calls", "root_policy_query_calls", "root_observer_calls",
        "root_world_model_member_calls", "simulator_step_calls",
        "condition_started_count", "target_read_calls",
    }
    item = _exact(value, fields, "root observation chronology")
    expected = {
        "root_reset_calls": 1,
        "root_policy_query_calls": 1,
        "root_observer_calls": 1,
        "root_world_model_member_calls": 0,
        "simulator_step_calls": 0,
        "condition_started_count": 0,
        "target_read_calls": 0,
    }
    for name, expected_value in expected.items():
        _require_int(item.get(name), name, expected=expected_value)
    return dict(item)


def _derivation_chronology(value: Any) -> dict[str, int]:
    result = _observation_chronology(
        {**dict(value), "root_world_model_member_calls": 0}
        if isinstance(value, Mapping) else value
    )
    if not isinstance(value, Mapping) or set(value) != set(result):
        raise RootObservedContractError("root derivation chronology fields changed")
    _require_int(
        value.get("root_world_model_member_calls"),
        "root world-model member calls", expected=MEMBER_COUNT,
    )
    result["root_world_model_member_calls"] = MEMBER_COUNT
    return result


def build_root_observation_commitment(
    *,
    pair_id: str,
    pair_ordinal: int,
    shared_snapshot_sha256: str,
    pre_action_snapshot_sha256: str,
    observer_authority_sha256: str,
    observer_actor_adapter_contract_sha256: str,
    observer_output_receipt: Mapping[str, Any],
    actor_visible_inputs: Mapping[str, Any],
    ordered_candidate_sha256: Sequence[str],
    candidate_legal: Sequence[bool],
    lowest_legal_original_candidate_index: int,
    mapped_actions: Any,
    chronology: Mapping[str, Any],
) -> dict[str, Any]:
    for value, role in (
        (pair_id, "root pair"),
        (shared_snapshot_sha256, "root shared snapshot"),
        (pre_action_snapshot_sha256, "root pre-action snapshot"),
        (observer_authority_sha256, "root observer authority"),
        (
            observer_actor_adapter_contract_sha256,
            "root observer actor adapter contract",
        ),
    ):
        _require_sha(value, role)
    _require_int(pair_ordinal, "root pair ordinal")
    inputs = _exact(
        actor_visible_inputs,
        {
            "actor_name", "policy_family", "state_feature_source_sha256",
            "actor_adapter_contract_sha256", "history", "history_mask",
            "proprio", "image_feature", "image_feature_extractor_file_sha256",
            "observer_input_receipt",
        },
        "root actor-visible inputs",
    )
    actor_name = inputs["actor_name"]
    policy_family = inputs["policy_family"]
    if (
        not isinstance(actor_name, str) or not actor_name
        or not isinstance(policy_family, str) or not policy_family
    ):
        raise RootObservedContractError("root actor identity is invalid")
    source_sha = _require_sha(
        inputs["state_feature_source_sha256"], "root state feature source"
    )
    adapter_sha = _require_sha(
        inputs["actor_adapter_contract_sha256"], "root actor adapter contract"
    )
    if adapter_sha != observer_actor_adapter_contract_sha256:
        raise RootObservedContractError(
            "root actor adapter is not bound to the observer authority"
        )
    history = _float32(
        inputs["history"], shape=(HISTORY_STEPS, STATE_DIM), role="root history"
    )
    history_mask = _bool(
        inputs["history_mask"], shape=(HISTORY_STEPS,), role="root history mask"
    )
    proprio = _float32(
        inputs["proprio"], shape=(PROPRIO_DIM,), role="root proprio"
    )
    image_raw = inputs["image_feature"]
    image = None if image_raw is None else _float32(
        image_raw, shape=None, role="root image feature"
    )
    if image is not None and image.ndim != 1:
        raise RootObservedContractError("root image feature must be one vector")
    if image is None and inputs["image_feature_extractor_file_sha256"] is not None:
        raise RootObservedContractError("absent root image names an extractor")
    input_receipt_sha = _validate_observer_input_receipt(
        inputs["observer_input_receipt"],
        actor_name=actor_name,
        policy_family=policy_family,
        state_feature_source_sha256=source_sha,
        history=history,
        history_mask=history_mask,
        proprio=proprio,
        image_feature=image,
        image_feature_extractor_file_sha256=inputs[
            "image_feature_extractor_file_sha256"
        ],
    )
    output_receipt_sha = _validate_observer_output_receipt(
        observer_output_receipt,
        pair_id=pair_id,
        observer_authority_sha256=observer_authority_sha256,
        input_receipt_sha256=input_receipt_sha,
    )
    ordered = list(ordered_candidate_sha256)
    legal = list(candidate_legal)
    fallback = _require_int(
        lowest_legal_original_candidate_index, "root lowest legal candidate"
    )
    if (
        not ordered
        or any(not _is_sha(item) for item in ordered)
        or len(set(ordered)) != len(ordered)
        or len(legal) != len(ordered)
        or any(type(item) is not bool for item in legal)
        or not any(legal)
        or fallback != legal.index(True)
    ):
        raise RootObservedContractError("root candidate identity/legal registry changed")
    actions = _float32(mapped_actions, shape=None, role="root mapped actions")
    if actions.ndim != 3 or actions.shape[0] != len(ordered) or min(actions.shape[1:]) < 1:
        raise RootObservedContractError("root mapped actions must be [candidate,horizon,action]")
    action_rows = [
        _descriptor(f"mapped_action_{index}", np.ascontiguousarray(actions[index]))
        for index in range(len(ordered))
    ]
    candidate_registry_sha = canonical_sha256(
        {
            "pair_id": pair_id,
            "candidate_count": len(ordered),
            "ordered_candidate_sha256": ordered,
            "candidate_legal": legal,
        }
    )
    mapped_action_set_sha = canonical_sha256(
        [row["tensor_sha256"] for row in action_rows]
    )
    input_descriptors = {
        "history": _descriptor("history", history),
        "history_mask": _descriptor("history_mask", history_mask),
        "proprio": _descriptor("proprio", proprio),
        "image_feature": (
            None if image is None else _descriptor("image_feature", image)
        ),
    }
    actor_input_base = {
        "format": ACTOR_INPUT_FORMAT,
        "actor_name": actor_name,
        "policy_family": policy_family,
        "state_feature_source_sha256": source_sha,
        "actor_adapter_contract_sha256": adapter_sha,
        "history_contract_sha256": observer.causal_history_contract()[
            "contract_sha256"
        ],
        "tensor_descriptors": input_descriptors,
        "observer_input_receipt_sha256": input_receipt_sha,
        "image_feature_extractor_file_sha256": inputs[
            "image_feature_extractor_file_sha256"
        ],
        "direct_event_or_predicate_input_present": False,
        "object_pose_input_present": False,
        "future_or_target_input_present": False,
    }
    actor_input_commitment = _signed(actor_input_base, "input_set_sha256")
    candidate_base = {
        "format": CANDIDATE_FORMAT,
        "ordered_candidate_sha256": ordered,
        "candidate_legal": legal,
        "lowest_legal_original_candidate_index": fallback,
        "candidate_registry_sha256": candidate_registry_sha,
        "mapped_action_descriptors": action_rows,
        "mapped_action_set_sha256": mapped_action_set_sha,
    }
    candidate_commitment = _signed(candidate_base, "candidate_set_sha256")
    base = {
        "format": OBSERVATION_FORMAT,
        "status": OBSERVATION_STATUS,
        "pair_id": pair_id,
        "pair_ordinal": pair_ordinal,
        "shared_snapshot_sha256": shared_snapshot_sha256,
        "pre_action_snapshot_sha256": pre_action_snapshot_sha256,
        "root_scope": ROOT_SCOPE,
        "observer_authority_sha256": observer_authority_sha256,
        "observer_actor_adapter_contract_sha256": adapter_sha,
        "observer_output_receipt": dict(observer_output_receipt),
        "observer_output_receipt_sha256": output_receipt_sha,
        "actor_visible_input_commitment": actor_input_commitment,
        "actor_visible_input_set_sha256": actor_input_commitment[
            "input_set_sha256"
        ],
        "candidate_commitment": candidate_commitment,
        "candidate_set_sha256": candidate_commitment["candidate_set_sha256"],
        "chronology": _observation_chronology(chronology),
    }
    return _signed(base, "observation_commit_sha256")


def validate_root_observation_commitment(
    value: Mapping[str, Any], *,
    expected_pair_id: str,
    expected_pair_ordinal: int,
    expected_shared_snapshot_sha256: str,
    expected_pre_action_snapshot_sha256: str,
    expected_observer_authority_sha256: str,
    expected_observer_actor_adapter_contract_sha256: str,
    observer_output_receipt: Mapping[str, Any],
    actor_visible_inputs: Mapping[str, Any],
    ordered_candidate_sha256: Sequence[str],
    candidate_legal: Sequence[bool],
    lowest_legal_original_candidate_index: int,
    mapped_actions: Any,
) -> str:
    fields = {
        "format", "status", "pair_id", "pair_ordinal",
        "shared_snapshot_sha256", "pre_action_snapshot_sha256", "root_scope",
        "observer_authority_sha256",
        "observer_actor_adapter_contract_sha256", "observer_output_receipt",
        "observer_output_receipt_sha256", "actor_visible_input_commitment",
        "actor_visible_input_set_sha256", "candidate_commitment",
        "candidate_set_sha256", "chronology",
    }
    digest = _verify_signed(
        value, fields=fields, digest_field="observation_commit_sha256",
        role="root observation commitment",
    )
    if value.get("format") != OBSERVATION_FORMAT or value.get("status") != OBSERVATION_STATUS:
        raise RootObservedContractError("root observation envelope changed")
    rebuilt = build_root_observation_commitment(
        pair_id=expected_pair_id,
        pair_ordinal=expected_pair_ordinal,
        shared_snapshot_sha256=expected_shared_snapshot_sha256,
        pre_action_snapshot_sha256=expected_pre_action_snapshot_sha256,
        observer_authority_sha256=expected_observer_authority_sha256,
        observer_actor_adapter_contract_sha256=(
            expected_observer_actor_adapter_contract_sha256
        ),
        observer_output_receipt=observer_output_receipt,
        actor_visible_inputs=actor_visible_inputs,
        ordered_candidate_sha256=ordered_candidate_sha256,
        candidate_legal=candidate_legal,
        lowest_legal_original_candidate_index=(
            lowest_legal_original_candidate_index
        ),
        mapped_actions=mapped_actions,
        chronology=value["chronology"],
    )
    if rebuilt != dict(value):
        raise RootObservedContractError("root observation differs from original arrays")
    return digest


def _auxiliary_arrays(
    value: Mapping[str, Any], *, candidate_count: int,
) -> dict[str, np.ndarray]:
    _exact(value, set(AUXILIARY_TENSOR_FIELDS), "root derivation auxiliary tensors")
    success = _float32(
        value["member_calibrated_success_probability"],
        shape=(MEMBER_COUNT, candidate_count),
        role="member calibrated success probability",
    )
    rank = _float32(
        value["member_composite_rank_score"],
        shape=(MEMBER_COUNT, candidate_count),
        role="member composite rank score",
    )
    uncertainty = _float32(
        value["candidate_structured_five_head_uncertainty"],
        shape=(candidate_count,),
        role="candidate structured uncertainty",
    )
    if (
        np.any((success < 0.0) | (success > 1.0))
        or np.any((uncertainty < 0.0) | (uncertainty > 1.0))
    ):
        raise RootObservedContractError("root success/uncertainty escaped [0,1]")
    return {
        "member_calibrated_success_probability": success,
        "member_composite_rank_score": rank,
        "candidate_structured_five_head_uncertainty": uncertainty,
    }


def _derived_vectors(auxiliary: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    success = np.mean(
        auxiliary["member_calibrated_success_probability"], axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    rank = np.mean(
        auxiliary["member_composite_rank_score"], axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    uncertainty = auxiliary[
        "candidate_structured_five_head_uncertainty"
    ].copy()
    return {
        "mean_success_probability": success.astype(float).tolist(),
        "mean_composite_rank_score": rank.astype(float).tolist(),
        "structured_five_head_uncertainty": uncertainty.astype(float).tolist(),
    }


def build_root_prediction_derivation_commitment(
    *,
    observation_commitment: Mapping[str, Any],
    raw_predictions: Mapping[str, Any],
    auxiliary_tensors: Mapping[str, Any],
    root_predictor_authority_sha256: str,
    calibration_sha256: str,
    source_rank_contract_set_sha256: str,
    uncertainty_contract_sha256: str,
    derivation_implementation_file_sha256: str,
    chronology: Mapping[str, Any],
) -> dict[str, Any]:
    observation_sha = _verify_signed(
        observation_commitment,
        fields={
            "format", "status", "pair_id", "pair_ordinal",
            "shared_snapshot_sha256", "pre_action_snapshot_sha256", "root_scope",
            "observer_authority_sha256",
            "observer_actor_adapter_contract_sha256", "observer_output_receipt",
            "observer_output_receipt_sha256", "actor_visible_input_commitment",
            "actor_visible_input_set_sha256", "candidate_commitment",
            "candidate_set_sha256", "chronology",
        },
        digest_field="observation_commit_sha256",
        role="root observation commitment",
    )
    for digest, role in (
        (root_predictor_authority_sha256, "root predictor authority"),
        (calibration_sha256, "root calibration"),
        (source_rank_contract_set_sha256, "root Source rank contract set"),
        (uncertainty_contract_sha256, "root uncertainty contract"),
        (derivation_implementation_file_sha256, "root derivation implementation"),
    ):
        _require_sha(digest, role)
    candidate = observation_commitment.get("candidate_commitment")
    if not isinstance(candidate, Mapping):
        raise RootObservedContractError("root observation candidate registry is missing")
    legal = candidate.get("candidate_legal")
    predicted_indices = [index for index, enabled in enumerate(legal) if enabled]
    tensor_commitment = audit.build_root_tensor_commitment(raw_predictions)
    if tensor_commitment["candidate_count"] != len(predicted_indices):
        raise RootObservedContractError("root raw prediction candidate coverage changed")
    auxiliary = _auxiliary_arrays(
        auxiliary_tensors, candidate_count=len(predicted_indices)
    )
    auxiliary_descriptors = {
        name: _descriptor(name, array) for name, array in auxiliary.items()
    }
    auxiliary_set_sha = canonical_sha256(
        [auxiliary_descriptors[name]["tensor_sha256"] for name in AUXILIARY_TENSOR_FIELDS]
    )
    vectors = _derived_vectors(auxiliary)
    vector_arrays = {
        name: np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        for name, values in vectors.items()
    }
    vector_descriptors = {
        name: _descriptor(name, array) for name, array in vector_arrays.items()
    }
    vector_set_sha = canonical_sha256(
        [vector_descriptors[name]["tensor_sha256"] for name in sorted(vector_descriptors)]
    )
    base = {
        "format": DERIVATION_FORMAT,
        "status": DERIVATION_STATUS,
        "pair_id": observation_commitment["pair_id"],
        "pair_ordinal": observation_commitment["pair_ordinal"],
        "observation_commit_sha256": observation_sha,
        "observer_output_receipt_sha256": observation_commitment[
            "observer_output_receipt_sha256"
        ],
        "candidate_set_sha256": observation_commitment["candidate_set_sha256"],
        "mapped_action_set_sha256": candidate["mapped_action_set_sha256"],
        "prediction_candidate_indices": predicted_indices,
        "root_predictor_authority_sha256": root_predictor_authority_sha256,
        "calibration_sha256": calibration_sha256,
        "source_rank_contract_set_sha256": source_rank_contract_set_sha256,
        "uncertainty_contract_sha256": uncertainty_contract_sha256,
        "derivation_implementation_file_sha256": (
            derivation_implementation_file_sha256
        ),
        "raw_tensor_commitment": tensor_commitment,
        "raw_tensor_set_sha256": tensor_commitment["tensor_set_sha256"],
        "auxiliary_tensor_descriptors": auxiliary_descriptors,
        "auxiliary_tensor_set_sha256": auxiliary_set_sha,
        "derived_vectors": vectors,
        "derived_vector_descriptors": vector_descriptors,
        "derived_vector_set_sha256": vector_set_sha,
        "mean_numeric_contract": "float64_accumulate_then_single_float32_round",
        "chronology": _derivation_chronology(chronology),
    }
    return _signed(base, "derivation_commit_sha256")


def validate_root_prediction_derivation_commitment(
    value: Mapping[str, Any], *,
    observation_commitment: Mapping[str, Any],
    raw_predictions: Mapping[str, Any],
    auxiliary_tensors: Mapping[str, Any],
    expected_root_predictor_authority_sha256: str,
    expected_calibration_sha256: str,
    expected_source_rank_contract_set_sha256: str,
    expected_uncertainty_contract_sha256: str,
    expected_derivation_implementation_file_sha256: str,
) -> str:
    fields = {
        "format", "status", "pair_id", "pair_ordinal",
        "observation_commit_sha256", "observer_output_receipt_sha256",
        "candidate_set_sha256", "mapped_action_set_sha256",
        "prediction_candidate_indices", "root_predictor_authority_sha256",
        "calibration_sha256", "source_rank_contract_set_sha256",
        "uncertainty_contract_sha256", "derivation_implementation_file_sha256",
        "raw_tensor_commitment", "raw_tensor_set_sha256",
        "auxiliary_tensor_descriptors", "auxiliary_tensor_set_sha256",
        "derived_vectors", "derived_vector_descriptors",
        "derived_vector_set_sha256", "mean_numeric_contract", "chronology",
    }
    digest = _verify_signed(
        value, fields=fields, digest_field="derivation_commit_sha256",
        role="root prediction derivation commitment",
    )
    if value.get("format") != DERIVATION_FORMAT or value.get("status") != DERIVATION_STATUS:
        raise RootObservedContractError("root derivation envelope changed")
    expected_provenance = {
        "root_predictor_authority_sha256": expected_root_predictor_authority_sha256,
        "calibration_sha256": expected_calibration_sha256,
        "source_rank_contract_set_sha256": (
            expected_source_rank_contract_set_sha256
        ),
        "uncertainty_contract_sha256": expected_uncertainty_contract_sha256,
        "derivation_implementation_file_sha256": (
            expected_derivation_implementation_file_sha256
        ),
    }
    for name, expected in expected_provenance.items():
        _require_sha(expected, f"expected root derivation {name}")
        if value.get(name) != expected:
            raise RootObservedContractError(
                f"root derivation {name} differs from frozen authority"
            )
    rebuilt = build_root_prediction_derivation_commitment(
        observation_commitment=observation_commitment,
        raw_predictions=raw_predictions,
        auxiliary_tensors=auxiliary_tensors,
        root_predictor_authority_sha256=expected_root_predictor_authority_sha256,
        calibration_sha256=expected_calibration_sha256,
        source_rank_contract_set_sha256=(
            expected_source_rank_contract_set_sha256
        ),
        uncertainty_contract_sha256=expected_uncertainty_contract_sha256,
        derivation_implementation_file_sha256=(
            expected_derivation_implementation_file_sha256
        ),
        chronology=value["chronology"],
    )
    if rebuilt != dict(value):
        raise RootObservedContractError(
            "root derivation differs from original tensors/vectors"
        )
    return digest


__all__ = [
    "ACTOR_INPUT_FORMAT",
    "AUXILIARY_TENSOR_FIELDS",
    "CANDIDATE_FORMAT",
    "DERIVATION_FORMAT",
    "OBSERVATION_FORMAT",
    "ROOT_SCOPE",
    "RootObservedContractError",
    "build_root_observation_commitment",
    "build_root_prediction_derivation_commitment",
    "canonical_sha256",
    "validate_root_observation_commitment",
    "validate_root_prediction_derivation_commitment",
]
