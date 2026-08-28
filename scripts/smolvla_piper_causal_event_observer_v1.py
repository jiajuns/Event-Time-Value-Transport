#!/usr/bin/env python3
"""Actor-visible causal event observer for 960-D SmolVLA-style features.

This module is intentionally not wired into the evaluation400 v3 runner.  It
defines a monitor-first observer, strict input/provenance receipts and
promotion gates without accepting simulator object poses, future states or
outcome fields as online inputs.  Privileged simulator state may derive
*offline labels only* under the frozen supervision contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


FORMAT = "etsf_actor_visible_causal_event_observer_v1"
INPUT_RECEIPT_FORMAT = "etsf_actor_visible_observer_input_receipt_v1"
TRAINING_SUPERVISION_FORMAT = "etsf_causal_event_observer_supervision_v1"
ADAPTER_CONTRACT_FORMAT = "etsf_causal_event_observer_actor_adapter_v1"
CALIBRATION_FORMAT = "etsf_causal_event_observer_calibration_v1"
DEPLOYMENT_FORMAT = "etsf_causal_event_observer_deployment_v1"
PROMOTION_EVIDENCE_FORMAT = "etsf_causal_event_observer_promotion_evidence_v1"
IMAGE_RECEIPT_FORMAT = "etsf_actor_visible_image_feature_receipt_v1"
EXECUTION_RECEIPT_FORMAT = "etsf_actor_visible_execution_receipt_v1"
HISTORY_FORMAT = "etsf_actor_visible_960d_same_branch_causal_history_v1"
CONFIG_FORMAT = "etsf_actor_visible_causal_event_observer_config_v1"
CORE_CHECKPOINT_FORMAT = "etsf_actor_visible_causal_event_observer_core_checkpoint_v1"
ADAPTER_CHECKPOINT_FORMAT = "etsf_actor_visible_causal_event_observer_adapter_checkpoint_v1"
ADAPTER_MANIFEST_FORMAT = "etsf_actor_visible_causal_event_observer_adapter_manifest_v1"
FROZEN_AUTHORITY_MANIFEST_FORMAT = "etsf_actor_visible_causal_event_observer_frozen_authority_v1"

MONITOR_ONLY_TARGET = "monitor_only"
EVALUATION400_V3_TARGET = "evaluation400_v3"
EVALUATION400_V4_TARGET = "evaluation400_v4"
INTEGRATION_TARGETS = (
    MONITOR_ONLY_TARGET,
    EVALUATION400_V3_TARGET,
    EVALUATION400_V4_TARGET,
)

STATE_DIM = 960
MAX_HISTORY_STEPS = 8
EXPECTED_EVENTS = ("e0", "e12", "e3", "e4", "eK")
EXPECTED_PREDICATES = ("moved", "lifted", "near_goal", "stationary", "success")
STATE_VISIBILITY = "actor_visible_policy_hidden_at_query_no_simulator_state"
PROPRIO_SOURCE = "actor_visible_processed_robot_state_at_query"
HISTORY_PADDING = "right_zero_padding_with_false_mask"
HISTORY_TRUNCATION = "left_truncate_keep_most_recent"

MIN_PROMOTION_GROUPS = 50
MIN_PROMOTION_GROUPS_PER_ACTOR = 10
MIN_EVENT_ACCURACY_LCB95 = 0.70
MIN_PREDICATE_F1_LCB95 = 0.65
MAX_CALIBRATION_ECE = 0.10
MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95 = 0.05


class CausalObserverContractError(RuntimeError):
    pass


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


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _exact_keys(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CausalObserverContractError(f"{role} fields changed")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_bundle_sha256(value: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(value):
        tensor = value[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise CausalObserverContractError("adapter state is not a tensor bundle")
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def float32_feature_sha256(value: torch.Tensor) -> str:
    if value.ndim != 1 or value.dtype != torch.float32:
        raise CausalObserverContractError("image feature must be one float32 vector")
    return _tensor_sha256(value)


def file_sha256(path: Path | str) -> str:
    artifact = Path(path)
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observer_config_document(config: "CausalObserverConfig") -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CONFIG_FORMAT,
        "state_input_dim": config.state_input_dim,
        "max_history_steps": config.max_history_steps,
        "proprio_dim": config.proprio_dim,
        "image_feature_dim": config.image_feature_dim,
        "hidden_dim": config.hidden_dim,
        "adapter_rank": config.adapter_rank,
        "event_names": list(config.event_names),
        "predicate_names": list(config.predicate_names),
        "dropout": float(config.dropout),
    }
    return {**value, "config_sha256": canonical_sha256(value)}


def validate_observer_config_document(
    value: Mapping[str, Any],
) -> tuple["CausalObserverConfig", dict[str, Any]]:
    expected = {
        "format", "state_input_dim", "max_history_steps", "proprio_dim",
        "image_feature_dim", "hidden_dim", "adapter_rank", "event_names",
        "predicate_names", "dropout", "config_sha256",
    }
    item = _exact_keys(value, expected, "observer config")
    logical = dict(item)
    digest = logical.pop("config_sha256")
    if item["format"] != CONFIG_FORMAT or digest != canonical_sha256(logical):
        raise CausalObserverContractError("observer config is invalid")
    try:
        config = CausalObserverConfig(
            state_input_dim=item["state_input_dim"],
            max_history_steps=item["max_history_steps"],
            proprio_dim=item["proprio_dim"],
            image_feature_dim=item["image_feature_dim"],
            hidden_dim=item["hidden_dim"],
            adapter_rank=item["adapter_rank"],
            event_names=tuple(item["event_names"]),
            predicate_names=tuple(item["predicate_names"]),
            dropout=float(item["dropout"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CausalObserverContractError("observer config is invalid") from error
    return config, dict(item)


def causal_history_contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": HISTORY_FORMAT,
        "state_input_dim": STATE_DIM,
        "max_history_steps": MAX_HISTORY_STEPS,
        "input_history": "same_branch_hidden_prefix_ending_at_current_query",
        "padding": HISTORY_PADDING,
        "truncation": HISTORY_TRUNCATION,
        "future_features_allowed": False,
        "cross_branch_or_group_history_allowed": False,
        "simulator_privileged_state_allowed": False,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


def build_causal_history_window(prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(prefix)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != STATE_DIM
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise CausalObserverContractError("actor-visible hidden prefix is invalid")
    retained = values[-MAX_HISTORY_STEPS:]
    history = np.zeros((MAX_HISTORY_STEPS, STATE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_HISTORY_STEPS, dtype=np.bool_)
    history[: len(retained)] = retained
    mask[: len(retained)] = True
    return history, mask


@dataclass(frozen=True)
class CausalObserverConfig:
    state_input_dim: int = STATE_DIM
    max_history_steps: int = MAX_HISTORY_STEPS
    proprio_dim: int = 14
    image_feature_dim: int = 0
    hidden_dim: int = 96
    adapter_rank: int = 8
    event_names: tuple[str, ...] = EXPECTED_EVENTS
    predicate_names: tuple[str, ...] = EXPECTED_PREDICATES
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.state_input_dim != STATE_DIM or self.max_history_steps != MAX_HISTORY_STEPS:
            raise ValueError("observer requires exact actor-visible 960-D/8-step input")
        if (
            self.proprio_dim < 1
            or self.image_feature_dim < 0
            or self.hidden_dim < 16
            or self.adapter_rank < 1
        ):
            raise ValueError("observer dimensions are invalid")
        if self.event_names != EXPECTED_EVENTS or self.predicate_names != EXPECTED_PREDICATES:
            raise ValueError("observer canonical event/predicate vocabulary changed")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("observer dropout must lie in [0,1)")


def training_supervision_contract(
    *,
    event_spec_sha256: str,
    dataset_manifest_sha256: str,
    actor_registry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _is_sha(event_spec_sha256) or not _is_sha(dataset_manifest_sha256):
        raise CausalObserverContractError(
            "training contract event-spec/dataset SHA is invalid"
        )
    actors: list[dict[str, str]] = []
    names: set[str] = set()
    for raw in actor_registry:
        item = _exact_keys(
            raw,
            {"actor_name", "policy_family", "state_feature_source_sha256"},
            "training actor registry",
        )
        actor_name = item["actor_name"]
        policy_family = item["policy_family"]
        source_sha = item["state_feature_source_sha256"]
        if (
            not isinstance(actor_name, str)
            or not actor_name
            or actor_name in names
            or not isinstance(policy_family, str)
            or not policy_family
            or not _is_sha(source_sha)
        ):
            raise CausalObserverContractError("training actor registry is invalid")
        names.add(actor_name)
        actors.append(
            {
                "actor_name": actor_name,
                "policy_family": policy_family,
                "state_feature_source_sha256": source_sha,
            }
        )
    if not actors:
        raise CausalObserverContractError("training contract needs at least one actor")
    value: dict[str, Any] = {
        "format": TRAINING_SUPERVISION_FORMAT,
        "event_spec_sha256": event_spec_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "event_names": list(EXPECTED_EVENTS),
        "predicate_names": list(EXPECTED_PREDICATES),
        "history_contract": causal_history_contract(),
        "actors": actors,
        "online_input_fields": [
            "actor_visible_960d_causal_history",
            "actor_visible_proprio",
            "optional_actor_visible_image_derived_features",
            "optional_prior_execution_receipt",
        ],
        "forbidden_online_input_fields": [
            "object_poses",
            "simulator_actor_pose",
            "future_hidden",
            "future_image",
            "success_or_terminal_outcome",
            "event_or_predicate_label",
        ],
        "label_derivation": (
            "offline_current_query_pose_to_atomic_predicates_and_dynamic_event_only"
        ),
        "privileged_label_source_available_to_model_inputs": False,
        "future_query_features_available_to_model_inputs": False,
        "split_unit": "logical_reset_group",
        "split_leakage_allowed": False,
        "per_actor_support_required": True,
        "promotion_required_before_rerank": True,
    }
    return {**value, "contract_sha256": canonical_sha256(value)}


class EmbodimentResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(self.down(value))


def make_actor_adapter_contract(
    *,
    actor_name: str,
    policy_family: str,
    state_feature_source_sha256: str,
    observer_core_file_sha256: str,
    training_contract_sha256: str,
    image_feature_extractor_file_sha256: str | None,
    config: CausalObserverConfig,
    adapter_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if (
        not isinstance(actor_name, str)
        or not actor_name
        or not isinstance(policy_family, str)
        or not policy_family
        or not _is_sha(state_feature_source_sha256)
        or not _is_sha(observer_core_file_sha256)
        or not _is_sha(training_contract_sha256)
        or (
            config.image_feature_dim > 0
            and not _is_sha(image_feature_extractor_file_sha256)
        )
        or (
            config.image_feature_dim == 0
            and image_feature_extractor_file_sha256 is not None
        )
    ):
        raise CausalObserverContractError("actor adapter identity is invalid")
    value: dict[str, Any] = {
        "format": ADAPTER_CONTRACT_FORMAT,
        "actor_name": actor_name,
        "policy_family": policy_family,
        "state_visibility": STATE_VISIBILITY,
        "state_feature_source_sha256": state_feature_source_sha256,
        "state_input_dim": config.state_input_dim,
        "proprio_dim": config.proprio_dim,
        "image_feature_dim": config.image_feature_dim,
        "image_feature_extractor_file_sha256": (
            image_feature_extractor_file_sha256
        ),
        "observer_core_file_sha256": observer_core_file_sha256,
        "training_supervision_contract_sha256": training_contract_sha256,
        "adapter_rank": config.adapter_rank,
        "adapter_state_sha256": tensor_bundle_sha256(adapter_state),
    }
    return {**value, "adapter_contract_sha256": canonical_sha256(value)}


def make_calibration(
    *,
    event_spec_sha256: str,
    independent_calibration_split_sha256: str,
    minimum_joint_confidence: float,
    event_temperature: float = 1.0,
    predicate_temperatures: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
    predicate_thresholds: Sequence[float] = (0.5, 0.5, 0.5, 0.5, 0.5),
    reject_all: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "event_spec_sha256": event_spec_sha256,
        "independent_calibration_split_sha256": independent_calibration_split_sha256,
        "event_temperature": float(event_temperature),
        "predicate_temperatures": [float(item) for item in predicate_temperatures],
        "predicate_thresholds": [float(item) for item in predicate_thresholds],
        "minimum_joint_confidence": float(minimum_joint_confidence),
        "reject_all": bool(reject_all),
        "selection_data": "independent_observer_calibration_no_world_model_formal_or_evaluation",
    }
    return {**value, "calibration_sha256": canonical_sha256(value)}


def validate_calibration(
    value: Mapping[str, Any], *, event_spec_sha256: str
) -> dict[str, Any]:
    expected = {
        "format", "event_spec_sha256", "independent_calibration_split_sha256",
        "event_temperature", "predicate_temperatures", "predicate_thresholds",
        "minimum_joint_confidence", "selection_data", "calibration_sha256",
        "reject_all",
    }
    item = _exact_keys(value, expected, "observer calibration")
    logical = dict(item)
    digest = logical.pop("calibration_sha256")
    temperatures = item["predicate_temperatures"]
    thresholds = item["predicate_thresholds"]
    if (
        item["format"] != CALIBRATION_FORMAT
        or item["event_spec_sha256"] != event_spec_sha256
        or not _is_sha(item["independent_calibration_split_sha256"])
        or not _finite_number(item["event_temperature"])
        or float(item["event_temperature"]) <= 0
        or not isinstance(temperatures, list)
        or len(temperatures) != len(EXPECTED_PREDICATES)
        or any(not _finite_number(v) or float(v) <= 0 for v in temperatures)
        or not isinstance(thresholds, list)
        or len(thresholds) != len(EXPECTED_PREDICATES)
        or any(not _finite_number(v) or not 0 < float(v) < 1 for v in thresholds)
        or not _finite_number(item["minimum_joint_confidence"])
        or not 0 <= float(item["minimum_joint_confidence"]) <= 1
        or type(item["reject_all"]) is not bool
        or item["selection_data"]
        != "independent_observer_calibration_no_world_model_formal_or_evaluation"
        or digest != canonical_sha256(logical)
    ):
        raise CausalObserverContractError("observer calibration is invalid")
    return dict(item)


def validate_promotion_evidence(
    value: Mapping[str, Any],
    *,
    observer_core_file_sha256: str,
    observer_checkpoint_file_sha256: str,
    observer_config_sha256: str,
    training_contract_sha256: str,
    actor_adapter_set_sha256: str,
    actor_adapter_checkpoint_set_sha256: str,
    calibration_sha256: str,
    actor_names: Sequence[str],
) -> dict[str, Any]:
    expected = {
        "format", "status", "observer_core_file_sha256",
        "observer_checkpoint_file_sha256", "observer_config_sha256",
        "training_supervision_contract_sha256",
        "actor_adapter_set_sha256", "actor_adapter_checkpoint_set_sha256",
        "calibration_sha256", "independent_calibration_split_sha256",
        "independent_validation_split_sha256", "actor_names",
        "independent_validation_groups", "per_actor_validation_groups",
        "event_macro_accuracy_lcb95", "predicate_macro_f1_lcb95",
        "maximum_event_ece", "maximum_predicate_ece",
        "low_confidence_false_accept_ucb95",
        "future_feature_perturbation_invariant",
        "cross_branch_isolation_passed", "privileged_input_static_audit_passed",
        "calibration_group_disjoint", "promotion_receipt_sha256",
    }
    item = _exact_keys(value, expected, "observer promotion evidence")
    logical = dict(item)
    digest = logical.pop("promotion_receipt_sha256")
    per_actor = item["per_actor_validation_groups"]
    expected_names = list(actor_names)
    if (
        item["format"] != PROMOTION_EVIDENCE_FORMAT
        or item["status"] != "independent_validation_passed_all_gates"
        or item["observer_core_file_sha256"] != observer_core_file_sha256
        or item["observer_checkpoint_file_sha256"]
        != observer_checkpoint_file_sha256
        or item["observer_config_sha256"] != observer_config_sha256
        or item["training_supervision_contract_sha256"] != training_contract_sha256
        or item["actor_adapter_set_sha256"] != actor_adapter_set_sha256
        or item["actor_adapter_checkpoint_set_sha256"]
        != actor_adapter_checkpoint_set_sha256
        or item["calibration_sha256"] != calibration_sha256
        or not _is_sha(item["independent_calibration_split_sha256"])
        or not _is_sha(item["independent_validation_split_sha256"])
        or item["independent_validation_split_sha256"]
        == item["independent_calibration_split_sha256"]
        or item["actor_names"] != expected_names
        or not _strict_int(item["independent_validation_groups"], minimum=MIN_PROMOTION_GROUPS)
        or not isinstance(per_actor, Mapping)
        or set(per_actor) != set(expected_names)
        or any(
            not _strict_int(per_actor[name], minimum=MIN_PROMOTION_GROUPS_PER_ACTOR)
            for name in expected_names
        )
        or not _finite_number(item["event_macro_accuracy_lcb95"])
        or float(item["event_macro_accuracy_lcb95"]) < MIN_EVENT_ACCURACY_LCB95
        or not _finite_number(item["predicate_macro_f1_lcb95"])
        or float(item["predicate_macro_f1_lcb95"]) < MIN_PREDICATE_F1_LCB95
        or not _finite_number(item["maximum_event_ece"])
        or not 0 <= float(item["maximum_event_ece"]) <= MAX_CALIBRATION_ECE
        or not _finite_number(item["maximum_predicate_ece"])
        or not 0 <= float(item["maximum_predicate_ece"]) <= MAX_CALIBRATION_ECE
        or not _finite_number(item["low_confidence_false_accept_ucb95"])
        or not 0 <= float(item["low_confidence_false_accept_ucb95"]) <= MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95
        or any(
            item[name] is not True
            for name in (
                "future_feature_perturbation_invariant",
                "cross_branch_isolation_passed",
                "privileged_input_static_audit_passed",
                "calibration_group_disjoint",
            )
        )
        or digest != canonical_sha256(logical)
    ):
        raise CausalObserverContractError("observer promotion evidence failed closed")
    return dict(item)


def make_deployment(
    *,
    promotion_enabled: bool,
    promotion_evidence: Mapping[str, Any] | None = None,
    integration_target: str = MONITOR_ONLY_TARGET,
    promotion_validation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if integration_target not in INTEGRATION_TARGETS:
        raise CausalObserverContractError("observer integration target is invalid")
    rerank_requested = bool(
        promotion_enabled and integration_target == EVALUATION400_V4_TARGET
    )
    if rerank_requested:
        context = _exact_keys(
            promotion_validation_context,
            {
                "observer_core_file_sha256", "observer_checkpoint_file_sha256",
                "observer_config_sha256", "training_contract_sha256",
                "actor_adapter_set_sha256",
                "actor_adapter_checkpoint_set_sha256", "calibration_sha256",
                "actor_names",
            },
            "v4 promotion validation context",
        )
        if promotion_evidence is None:
            raise CausalObserverContractError(
                "v4 rerank deployment lacks promotion evidence"
            )
        validate_promotion_evidence(
            promotion_evidence,
            observer_core_file_sha256=context["observer_core_file_sha256"],
            observer_checkpoint_file_sha256=context[
                "observer_checkpoint_file_sha256"
            ],
            observer_config_sha256=context["observer_config_sha256"],
            training_contract_sha256=context["training_contract_sha256"],
            actor_adapter_set_sha256=context["actor_adapter_set_sha256"],
            actor_adapter_checkpoint_set_sha256=context[
                "actor_adapter_checkpoint_set_sha256"
            ],
            calibration_sha256=context["calibration_sha256"],
            actor_names=context["actor_names"],
        )
    rerank_enabled = rerank_requested
    value: dict[str, Any] = {
        "format": DEPLOYMENT_FORMAT,
        "status": (
            "promoted_evaluation400_v4_rerank_low_confidence_fail_closed"
            if rerank_enabled
            else (
                "promoted_observer_output_monitor_only"
                if promotion_enabled
                else "monitor_only_not_integrated"
            )
        ),
        "promotion_enabled": bool(promotion_enabled),
        "rerank_enabled": rerank_enabled,
        "promotion_evidence": (
            dict(promotion_evidence) if promotion_evidence is not None else None
        ),
        "integration_target": integration_target,
        "integration_status": (
            "integrated_frozen_observer_into_evaluation400_v4"
            if rerank_enabled
            else "not_authorized_for_evaluation400_v4_rerank"
        ),
    }
    return {**value, "deployment_sha256": canonical_sha256(value)}


def make_absent_image_receipt() -> dict[str, Any]:
    return {"present": False}


def make_image_receipt(
    feature: torch.Tensor,
    *,
    extractor_file_sha256: str,
    frame_query_index: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "present": True,
        "format": IMAGE_RECEIPT_FORMAT,
        "extractor_file_sha256": extractor_file_sha256,
        "feature_dim": int(feature.numel()),
        "frame_query_index": frame_query_index,
        "source": "actor_visible_rgb_at_current_query",
        "feature_sha256": float32_feature_sha256(feature),
    }
    return {**value, "receipt_sha256": canonical_sha256(value)}


def make_absent_execution_receipt() -> dict[str, Any]:
    return {"present": False}


def make_execution_receipt(
    *,
    action_sha256: str,
    executed_control_steps: int,
    last_completed_query_index: int,
    current_query_index: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "present": True,
        "format": EXECUTION_RECEIPT_FORMAT,
        "action_sha256": action_sha256,
        "executed_control_steps": executed_control_steps,
        "last_completed_query_index": last_completed_query_index,
        "current_query_index": current_query_index,
        "terminal_or_outcome_fields_present": False,
    }
    return {**value, "receipt_sha256": canonical_sha256(value)}


def make_input_receipt(
    *,
    history: torch.Tensor,
    history_mask: torch.Tensor,
    proprio: torch.Tensor,
    actor_name: str,
    policy_family: str,
    state_feature_source_sha256: str,
    current_query_index: int,
    valid_history_steps: int,
    image_feature_receipt: Mapping[str, Any] | None = None,
    execution_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start = current_query_index - valid_history_steps + 1
    value: dict[str, Any] = {
        "format": INPUT_RECEIPT_FORMAT,
        "status": "actor_visible_causal_inputs_only",
        "actor_name": actor_name,
        "policy_family": policy_family,
        "state_feature_source_sha256": state_feature_source_sha256,
        "state_input_dim": STATE_DIM,
        "history_sha256": _tensor_sha256(history),
        "history_mask_sha256": _tensor_sha256(history_mask),
        "proprio_sha256": _tensor_sha256(proprio),
        "history_contract_sha256": causal_history_contract()["contract_sha256"],
        "history_start_query_index": start,
        "history_end_query_index": current_query_index,
        "current_query_index": current_query_index,
        "valid_history_steps": valid_history_steps,
        "history_padding": HISTORY_PADDING,
        "history_truncation": HISTORY_TRUNCATION,
        "state_visibility": STATE_VISIBILITY,
        "proprio_source": PROPRIO_SOURCE,
        "object_pose_fields_present": False,
        "simulator_privileged_state_read": False,
        "future_features_read": False,
        "image_feature_receipt": dict(
            image_feature_receipt or make_absent_image_receipt()
        ),
        "execution_receipt": dict(
            execution_receipt or make_absent_execution_receipt()
        ),
    }
    return {**value, "receipt_sha256": canonical_sha256(value)}


def _validate_image_receipt(
    value: Any,
    *,
    feature: torch.Tensor | None,
    feature_dim: int,
    current_query_index: int,
) -> str | None:
    if value == {"present": False}:
        return None
    expected = {
        "present", "format", "extractor_file_sha256", "feature_dim",
        "frame_query_index", "source", "feature_sha256", "receipt_sha256",
    }
    item = _exact_keys(value, expected, "image feature receipt")
    logical = dict(item)
    digest = logical.pop("receipt_sha256")
    if (
        item["present"] is not True
        or item["format"] != IMAGE_RECEIPT_FORMAT
        or not _is_sha(item["extractor_file_sha256"])
        or item["feature_dim"] != feature_dim
        or item["frame_query_index"] != current_query_index
        or item["source"] != "actor_visible_rgb_at_current_query"
        or feature is None
        or feature.shape != (feature_dim,)
        or item["feature_sha256"] != float32_feature_sha256(feature)
        or digest != canonical_sha256(logical)
    ):
        raise CausalObserverContractError("image feature receipt is invalid")
    return str(item["extractor_file_sha256"])


def _validate_execution_receipt(value: Any, *, current_query_index: int) -> None:
    if value == {"present": False}:
        return
    expected = {
        "present", "format", "action_sha256", "executed_control_steps",
        "last_completed_query_index", "current_query_index",
        "terminal_or_outcome_fields_present", "receipt_sha256",
    }
    item = _exact_keys(value, expected, "execution receipt")
    logical = dict(item)
    digest = logical.pop("receipt_sha256")
    expected_last = current_query_index - 1
    if (
        item["present"] is not True
        or item["format"] != EXECUTION_RECEIPT_FORMAT
        or not _is_sha(item["action_sha256"])
        or not _strict_int(item["executed_control_steps"], minimum=1)
        or item["last_completed_query_index"] != expected_last
        or item["current_query_index"] != current_query_index
        or item["terminal_or_outcome_fields_present"] is not False
        or digest != canonical_sha256(logical)
    ):
        raise CausalObserverContractError("execution receipt is invalid")


@dataclass(frozen=True)
class CausalObserverPrediction:
    current_event_id: torch.Tensor
    current_event_probability: torch.Tensor
    current_predicates: torch.Tensor
    current_predicate_probability: torch.Tensor
    confidence: torch.Tensor
    applicability: torch.Tensor
    applicability_reason: tuple[str, ...]


class ActorVisibleCausalEventObserverV1(nn.Module):
    """Shared causal observer with separately content-addressed actor adapters."""

    def __init__(
        self,
        config: CausalObserverConfig,
        *,
        training_contract: Mapping[str, Any],
        observer_core_file_sha256: str,
        observer_checkpoint_file_sha256: str,
        observer_config_sha256: str,
        actor_adapter_set_sha256: str,
        actor_adapter_checkpoint_set_sha256: str,
        adapter_contracts: Mapping[str, Mapping[str, Any]],
        adapter_states: Mapping[str, Mapping[str, torch.Tensor]],
        calibration: Mapping[str, Any],
        deployment: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.training_contract = self._validate_training_contract(training_contract)
        if not _is_sha(observer_core_file_sha256):
            raise CausalObserverContractError("observer core file SHA is invalid")
        self.observer_core_file_sha256 = observer_core_file_sha256
        for role, digest in (
            ("observer checkpoint", observer_checkpoint_file_sha256),
            ("observer config", observer_config_sha256),
            ("actor adapter set", actor_adapter_set_sha256),
            ("actor adapter checkpoint set", actor_adapter_checkpoint_set_sha256),
        ):
            if not _is_sha(digest):
                raise CausalObserverContractError(f"{role} SHA is invalid")
        self.observer_checkpoint_file_sha256 = observer_checkpoint_file_sha256
        self.observer_config_sha256 = observer_config_sha256
        self.actor_adapter_set_sha256 = actor_adapter_set_sha256
        self.actor_adapter_checkpoint_set_sha256 = (
            actor_adapter_checkpoint_set_sha256
        )
        self.calibration = validate_calibration(
            calibration, event_spec_sha256=self.training_contract["event_spec_sha256"]
        )

        self.state_bridge = nn.Sequential(
            nn.LayerNorm(config.state_input_dim),
            nn.Linear(config.state_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.history_cell = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.proprio_encoder = nn.Sequential(
            nn.LayerNorm(config.proprio_dim),
            nn.Linear(config.proprio_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.image_encoder = (
            nn.Sequential(
                nn.LayerNorm(config.image_feature_dim),
                nn.Linear(config.image_feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.hidden_dim),
            )
            if config.image_feature_dim > 0
            else None
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.event_head = nn.Linear(config.hidden_dim, len(config.event_names))
        self.predicate_head = nn.Linear(config.hidden_dim, len(config.predicate_names))

        actor_records = self.training_contract["actors"]
        self.actor_names = tuple(item["actor_name"] for item in actor_records)
        if set(adapter_contracts) != set(self.actor_names) or set(adapter_states) != set(self.actor_names):
            raise CausalObserverContractError("actor adapter registry is incomplete")
        self.actor_adapters = nn.ModuleList()
        self.adapter_contracts: dict[str, dict[str, Any]] = {}
        for record in actor_records:
            name = record["actor_name"]
            adapter = EmbodimentResidualAdapter(config.hidden_dim, config.adapter_rank)
            state = adapter_states[name]
            adapter.load_state_dict(state, strict=True)
            contract = self._validate_adapter_contract(
                adapter_contracts[name], record=record, adapter=adapter
            )
            self.actor_adapters.append(adapter)
            self.adapter_contracts[name] = contract
        self.actor_to_index = {name: index for index, name in enumerate(self.actor_names)}
        self.deployment = self._validate_deployment(deployment)

    def _validate_training_contract(self, value: Mapping[str, Any]) -> dict[str, Any]:
        expected = {
            "format", "event_spec_sha256", "dataset_manifest_sha256",
            "event_names", "predicate_names",
            "history_contract", "actors", "online_input_fields",
            "forbidden_online_input_fields", "label_derivation",
            "privileged_label_source_available_to_model_inputs",
            "future_query_features_available_to_model_inputs", "split_unit",
            "split_leakage_allowed", "per_actor_support_required",
            "promotion_required_before_rerank", "contract_sha256",
        }
        item = _exact_keys(value, expected, "training supervision contract")
        logical = dict(item)
        digest = logical.pop("contract_sha256")
        history = item["history_contract"]
        if (
            item["format"] != TRAINING_SUPERVISION_FORMAT
            or not _is_sha(item["event_spec_sha256"])
            or not _is_sha(item["dataset_manifest_sha256"])
            or item["event_names"] != list(EXPECTED_EVENTS)
            or item["predicate_names"] != list(EXPECTED_PREDICATES)
            or history != causal_history_contract()
            or item["privileged_label_source_available_to_model_inputs"] is not False
            or item["future_query_features_available_to_model_inputs"] is not False
            or item["split_unit"] != "logical_reset_group"
            or item["split_leakage_allowed"] is not False
            or item["per_actor_support_required"] is not True
            or item["promotion_required_before_rerank"] is not True
            or digest != canonical_sha256(logical)
        ):
            raise CausalObserverContractError("training supervision contract is invalid")
        # Reconstructing through the public builder validates the actor registry.
        rebuilt = training_supervision_contract(
            event_spec_sha256=item["event_spec_sha256"],
            dataset_manifest_sha256=item["dataset_manifest_sha256"],
            actor_registry=item["actors"],
        )
        if rebuilt != dict(item):
            raise CausalObserverContractError("training supervision contract is not canonical")
        return dict(item)

    def _validate_adapter_contract(
        self,
        value: Mapping[str, Any],
        *,
        record: Mapping[str, Any],
        adapter: EmbodimentResidualAdapter,
    ) -> dict[str, Any]:
        expected = {
            "format", "actor_name", "policy_family", "state_visibility",
            "state_feature_source_sha256", "state_input_dim", "proprio_dim",
            "image_feature_dim", "image_feature_extractor_file_sha256",
            "observer_core_file_sha256",
            "training_supervision_contract_sha256", "adapter_rank",
            "adapter_state_sha256", "adapter_contract_sha256",
        }
        item = _exact_keys(value, expected, "actor adapter contract")
        logical = dict(item)
        digest = logical.pop("adapter_contract_sha256")
        if (
            item["format"] != ADAPTER_CONTRACT_FORMAT
            or item["actor_name"] != record["actor_name"]
            or item["policy_family"] != record["policy_family"]
            or item["state_visibility"] != STATE_VISIBILITY
            or item["state_feature_source_sha256"] != record["state_feature_source_sha256"]
            or item["state_input_dim"] != self.config.state_input_dim
            or item["proprio_dim"] != self.config.proprio_dim
            or item["image_feature_dim"] != self.config.image_feature_dim
            or (
                self.config.image_feature_dim > 0
                and not _is_sha(item["image_feature_extractor_file_sha256"])
            )
            or (
                self.config.image_feature_dim == 0
                and item["image_feature_extractor_file_sha256"] is not None
            )
            or item["observer_core_file_sha256"] != self.observer_core_file_sha256
            or item["training_supervision_contract_sha256"] != self.training_contract["contract_sha256"]
            or item["adapter_rank"] != self.config.adapter_rank
            or item["adapter_state_sha256"] != tensor_bundle_sha256(adapter.state_dict())
            or digest != canonical_sha256(logical)
        ):
            raise CausalObserverContractError("actor adapter contract is invalid")
        return dict(item)

    def _validate_deployment(self, value: Mapping[str, Any]) -> dict[str, Any]:
        expected = {
            "format", "status", "promotion_enabled", "rerank_enabled",
            "promotion_evidence", "integration_target",
            "integration_status", "deployment_sha256",
        }
        item = _exact_keys(value, expected, "observer deployment")
        logical = dict(item)
        digest = logical.pop("deployment_sha256")
        if (
            item["format"] != DEPLOYMENT_FORMAT
            or type(item["promotion_enabled"]) is not bool
            or type(item["rerank_enabled"]) is not bool
            or item["integration_target"] not in INTEGRATION_TARGETS
            or digest != canonical_sha256(logical)
        ):
            raise CausalObserverContractError("observer deployment is invalid")
        if item["promotion_enabled"]:
            evidence = validate_promotion_evidence(
                item["promotion_evidence"],
                observer_core_file_sha256=self.observer_core_file_sha256,
                observer_checkpoint_file_sha256=(
                    self.observer_checkpoint_file_sha256
                ),
                observer_config_sha256=self.observer_config_sha256,
                training_contract_sha256=self.training_contract["contract_sha256"],
                actor_adapter_set_sha256=self.actor_adapter_set_sha256,
                actor_adapter_checkpoint_set_sha256=(
                    self.actor_adapter_checkpoint_set_sha256
                ),
                calibration_sha256=self.calibration["calibration_sha256"],
                actor_names=self.actor_names,
            )
            if evidence["independent_calibration_split_sha256"] != self.calibration[
                "independent_calibration_split_sha256"
            ]:
                raise CausalObserverContractError("promotion/calibration split differs")
        elif item["promotion_evidence"] is not None:
            raise CausalObserverContractError("monitor-only observer claims promotion")
        rerank_expected = bool(
            item["promotion_enabled"]
            and item["integration_target"] == EVALUATION400_V4_TARGET
        )
        if rerank_expected and self.calibration["reject_all"] is True:
            raise CausalObserverContractError(
                "reject-all calibration cannot authorize v4 rerank"
            )
        expected_status = (
            "promoted_evaluation400_v4_rerank_low_confidence_fail_closed"
            if rerank_expected
            else (
                "promoted_observer_output_monitor_only"
                if item["promotion_enabled"]
                else "monitor_only_not_integrated"
            )
        )
        expected_integration = (
            "integrated_frozen_observer_into_evaluation400_v4"
            if rerank_expected
            else "not_authorized_for_evaluation400_v4_rerank"
        )
        if (
            item["rerank_enabled"] is not rerank_expected
            or item["status"] != expected_status
            or item["integration_status"] != expected_integration
        ):
            raise CausalObserverContractError("observer rerank authorization changed")
        return dict(item)

    def _encode_history(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = history.new_zeros(history.shape[0], self.config.hidden_dim)
        for index in range(self.config.max_history_steps):
            proposed = self.history_cell(self.state_bridge(history[:, index]), state)
            state = torch.where(mask[:, index, None], proposed, state)
        return state

    def _validate_inputs(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        proprio: torch.Tensor,
        actor_names: Sequence[str],
        receipts: Sequence[Mapping[str, Any]],
        image_features: torch.Tensor | None,
    ) -> list[int]:
        batch = len(actor_names)
        if (
            history.dtype != torch.float32
            or history.shape != (batch, MAX_HISTORY_STEPS, STATE_DIM)
            or history_mask.dtype != torch.bool
            or history_mask.shape != (batch, MAX_HISTORY_STEPS)
            or proprio.dtype != torch.float32
            or proprio.shape != (batch, self.config.proprio_dim)
            or len(receipts) != batch
            or not bool(torch.isfinite(history).all())
            or not bool(torch.isfinite(proprio).all())
        ):
            raise CausalObserverContractError("observer tensor input contract changed")
        if image_features is not None and (
            image_features.dtype != torch.float32
            or image_features.shape != (batch, self.config.image_feature_dim)
            or not bool(torch.isfinite(image_features).all())
        ):
            raise CausalObserverContractError("observer image feature tensor is invalid")
        if image_features is None and self.config.image_feature_dim == 0:
            pass
        elif image_features is None:
            # Optional image input is represented by the zero branch below.
            pass
        indices: list[int] = []
        receipt_fields = {
            "format", "status", "actor_name", "policy_family",
            "state_feature_source_sha256", "state_input_dim",
            "history_sha256", "history_mask_sha256", "proprio_sha256",
            "history_contract_sha256", "history_start_query_index",
            "history_end_query_index", "current_query_index",
            "valid_history_steps", "history_padding", "history_truncation",
            "state_visibility", "proprio_source", "object_pose_fields_present",
            "simulator_privileged_state_read", "future_features_read",
            "image_feature_receipt", "execution_receipt", "receipt_sha256",
        }
        for row, (actor_name, receipt) in enumerate(zip(actor_names, receipts, strict=True)):
            item = _exact_keys(receipt, receipt_fields, "observer input receipt")
            logical = dict(item)
            digest = logical.pop("receipt_sha256")
            actor_index = self.actor_to_index.get(actor_name)
            if actor_index is None:
                raise CausalObserverContractError("observer actor has no bound adapter")
            contract = self.adapter_contracts[actor_name]
            mask = history_mask[row]
            valid_steps = int(mask.sum())
            expected_mask = torch.arange(MAX_HISTORY_STEPS, device=mask.device) < valid_steps
            current_query = item["current_query_index"]
            expected_start = current_query - valid_steps + 1 if _strict_int(current_query) else -1
            if (
                item["format"] != INPUT_RECEIPT_FORMAT
                or item["status"] != "actor_visible_causal_inputs_only"
                or item["actor_name"] != actor_name
                or item["policy_family"] != contract["policy_family"]
                or item["state_feature_source_sha256"] != contract["state_feature_source_sha256"]
                or item["state_input_dim"] != STATE_DIM
                or item["history_sha256"] != _tensor_sha256(history[row])
                or item["history_mask_sha256"] != _tensor_sha256(mask)
                or item["proprio_sha256"] != _tensor_sha256(proprio[row])
                or item["history_contract_sha256"] != causal_history_contract()["contract_sha256"]
                or not _strict_int(item["valid_history_steps"], minimum=1)
                or item["valid_history_steps"] != valid_steps
                or valid_steps > MAX_HISTORY_STEPS
                or not _strict_int(current_query)
                or item["history_end_query_index"] != current_query
                or item["history_start_query_index"] != expected_start
                or expected_start != max(0, current_query - MAX_HISTORY_STEPS + 1)
                or item["history_padding"] != HISTORY_PADDING
                or item["history_truncation"] != HISTORY_TRUNCATION
                or item["state_visibility"] != STATE_VISIBILITY
                or item["proprio_source"] != PROPRIO_SOURCE
                or item["object_pose_fields_present"] is not False
                or item["simulator_privileged_state_read"] is not False
                or item["future_features_read"] is not False
                or digest != canonical_sha256(logical)
                or not torch.equal(mask, expected_mask)
                or bool((history[row][~mask] != 0).any())
            ):
                raise CausalObserverContractError("observer input receipt failed closed")
            feature = None if image_features is None else image_features[row]
            image_extractor_sha = _validate_image_receipt(
                item["image_feature_receipt"],
                feature=feature,
                feature_dim=self.config.image_feature_dim,
                current_query_index=current_query,
            )
            if image_extractor_sha is not None and self.config.image_feature_dim == 0:
                raise CausalObserverContractError("observer was not configured for image features")
            if (
                image_extractor_sha is not None
                and image_extractor_sha
                != contract["image_feature_extractor_file_sha256"]
            ):
                raise CausalObserverContractError("image extractor provenance changed")
            if image_extractor_sha is None and feature is not None and bool((feature != 0).any()):
                raise CausalObserverContractError("unreceipted image feature is nonzero")
            _validate_execution_receipt(
                item["execution_receipt"], current_query_index=current_query
            )
            indices.append(actor_index)
        return indices

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        proprio: torch.Tensor,
        *,
        actor_names: Sequence[str],
        receipts: Sequence[Mapping[str, Any]],
        image_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        actor_indices = self._validate_inputs(
            history, history_mask, proprio, actor_names, receipts, image_features
        )
        state = self._encode_history(history, history_mask)
        proprio_feature = self.proprio_encoder(proprio)
        if self.image_encoder is None or image_features is None:
            image_feature = torch.zeros_like(state)
        else:
            image_feature = self.image_encoder(image_features)
        fused = self.fusion(torch.cat([state, proprio_feature, image_feature], dim=-1))
        adapted = torch.empty_like(fused)
        for actor_index, adapter in enumerate(self.actor_adapters):
            rows = torch.as_tensor(
                [index == actor_index for index in actor_indices],
                dtype=torch.bool,
                device=fused.device,
            )
            if bool(rows.any()):
                adapted[rows] = adapter(fused[rows])
        return {
            "event_logits": self.event_head(adapted),
            "predicate_logits": self.predicate_head(adapted),
        }

    @torch.inference_mode()
    def observe(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        proprio: torch.Tensor,
        *,
        actor_names: Sequence[str],
        receipts: Sequence[Mapping[str, Any]],
        image_features: torch.Tensor | None = None,
    ) -> CausalObserverPrediction:
        output = self.forward(
            history,
            history_mask,
            proprio,
            actor_names=actor_names,
            receipts=receipts,
            image_features=image_features,
        )
        event_probability = torch.softmax(
            output["event_logits"] / float(self.calibration["event_temperature"]),
            dim=-1,
        )
        predicate_temperature = output["predicate_logits"].new_tensor(
            self.calibration["predicate_temperatures"]
        )
        predicate_probability = torch.sigmoid(
            output["predicate_logits"] / predicate_temperature
        )
        thresholds = predicate_probability.new_tensor(
            self.calibration["predicate_thresholds"]
        )
        predicates = (predicate_probability >= thresholds).to(torch.float32)
        event_confidence = event_probability.amax(dim=-1)
        predicate_confidence = torch.maximum(
            predicate_probability, 1.0 - predicate_probability
        ).amin(dim=-1)
        confidence = torch.minimum(event_confidence, predicate_confidence)
        confident = (
            confidence >= float(self.calibration["minimum_joint_confidence"])
        ) & (not bool(self.calibration["reject_all"]))
        promoted = self.deployment["promotion_enabled"] is True
        rerank_authorized = self.deployment["rerank_enabled"] is True
        applicability = confident & promoted & rerank_authorized
        reasons = tuple(
            "applicable_promoted_evaluation400_v4_rerank"
            if bool(applicability[index])
            else (
                "monitor_only_not_promoted"
                if not promoted
                else (
                    "promoted_but_not_rerank_authorized"
                    if not rerank_authorized
                    else "low_confidence_fail_closed"
                )
            )
            for index in range(len(actor_names))
        )
        return CausalObserverPrediction(
            current_event_id=event_probability.argmax(dim=-1),
            current_event_probability=event_probability,
            current_predicates=predicates,
            current_predicate_probability=predicate_probability,
            confidence=confidence,
            applicability=applicability,
            applicability_reason=reasons,
        )


def observer_core_state_dict(
    model: ActorVisibleCausalEventObserverV1,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("actor_adapters.")
    }


def _safe_declared_file(root: Path, basename: Any, role: str) -> Path:
    if (
        not isinstance(basename, str)
        or not basename
        or basename in {".", ".."}
        or Path(basename).name != basename
        or "/" in basename
        or "\\" in basename
    ):
        raise CausalObserverContractError(f"{role} basename is unsafe")
    path = root / basename
    if path.is_symlink() or not path.is_file():
        raise CausalObserverContractError(f"{role} is missing or not a regular file")
    return path


def _read_json_artifact(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CausalObserverContractError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CausalObserverContractError(f"{role} must be one JSON object")
    return value


def _read_tensor_artifact(path: Path, role: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CausalObserverContractError(
            f"{role} is not a safe weights-only torch artifact"
        ) from error
    if not isinstance(value, dict):
        raise CausalObserverContractError(f"{role} must be one tensor document")
    return value


def _validate_tensor_mapping(value: Any, role: str) -> dict[str, torch.Tensor]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(not isinstance(name, str) or not torch.is_tensor(tensor)
               for name, tensor in value.items())
    ):
        raise CausalObserverContractError(f"{role} is not a tensor state dict")
    result = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in value.items()
    }
    if any(not bool(torch.isfinite(tensor).all()) for tensor in result.values()):
        raise CausalObserverContractError(f"{role} contains non-finite tensors")
    return result


@dataclass(frozen=True)
class VerifiedCausalObservation:
    current_event_id: int
    current_predicates: dict[str, bool]
    confidence: float
    minimum_joint_confidence: float
    input_receipt_sha256: str
    prediction_sha256: str
    calibration_sha256: str


class FrozenCausalObserverRuntimeV1:
    """A realized, immutable-weight observer and its causal query history."""

    def __init__(
        self,
        *,
        model: ActorVisibleCausalEventObserverV1,
        authority: Mapping[str, Any],
        frozen_root: Path,
    ) -> None:
        if model.deployment.get("rerank_enabled") is not True:
            raise CausalObserverContractError("loaded observer is not v4-rerank enabled")
        self.model = model
        self.authority = dict(authority)
        self.frozen_root = frozen_root
        self._loaded_model_tensor_set_sha256 = tensor_bundle_sha256(
            self.model.state_dict()
        )
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}

    @property
    def authority_sha256(self) -> str:
        return str(self.authority["authority_sha256"])

    def validate_frozen_realization(self) -> None:
        promotion = self.model.deployment.get("promotion_evidence")
        if (
            self.model.training
            or any(parameter.requires_grad for parameter in self.model.parameters())
            or tensor_bundle_sha256(self.model.state_dict())
            != self._loaded_model_tensor_set_sha256
            or self.authority.get("observer_core_file_sha256")
            != self.model.observer_core_file_sha256
            or self.authority.get("observer_checkpoint_file_sha256")
            != self.model.observer_checkpoint_file_sha256
            or self.authority.get("training_contract_sha256")
            != self.model.training_contract["contract_sha256"]
            or self.authority.get("actor_adapter_set_sha256")
            != self.model.actor_adapter_set_sha256
            or self.authority.get("actor_adapter_checkpoint_set_sha256")
            != self.model.actor_adapter_checkpoint_set_sha256
            or self.authority.get("calibration_sha256")
            != self.model.calibration["calibration_sha256"]
            or self.authority.get("deployment_sha256")
            != self.model.deployment["deployment_sha256"]
            or not isinstance(promotion, Mapping)
            or self.authority.get("promotion_evidence_sha256")
            != promotion.get("promotion_receipt_sha256")
        ):
            raise CausalObserverContractError(
                "loaded observer no longer matches frozen authority"
            )

    def start_condition(self, *, pair_id: str, condition_id: str) -> None:
        self.validate_frozen_realization()
        if not _is_sha(pair_id) or not isinstance(condition_id, str) or not condition_id:
            raise CausalObserverContractError("observer condition identity is invalid")
        self._sessions[(pair_id, condition_id)] = {
            "next_step_index": 0,
            "hidden_prefix": [],
        }

    def observe_actor_visible_query(
        self,
        value: Mapping[str, Any],
        *,
        pair_id: str,
        condition_id: str,
        step_index: int,
        previous_action_sha256: str | None,
        executed_control_steps: int | None,
    ) -> VerifiedCausalObservation:
        item = _exact_keys(
            value,
            {"actor_name", "current_hidden", "current_proprio", "image_feature"},
            "actor-visible observer query",
        )
        session = self._sessions.get((pair_id, condition_id))
        if session is None or session["next_step_index"] != step_index:
            raise CausalObserverContractError("observer query history is absent or out of order")
        if step_index == 0:
            if previous_action_sha256 is not None or executed_control_steps is not None:
                raise CausalObserverContractError("root observer query has prior execution")
            execution_receipt = make_absent_execution_receipt()
        else:
            if not _is_sha(previous_action_sha256) or not _strict_int(
                executed_control_steps, minimum=1
            ):
                raise CausalObserverContractError(
                    "non-root observer query lacks bound prior execution"
                )
            execution_receipt = make_execution_receipt(
                action_sha256=str(previous_action_sha256),
                executed_control_steps=int(executed_control_steps),
                last_completed_query_index=step_index - 1,
                current_query_index=step_index,
            )
        actor_name = item["actor_name"]
        if not isinstance(actor_name, str) or actor_name not in self.model.actor_to_index:
            raise CausalObserverContractError("observer actor has no frozen adapter")
        hidden = item["current_hidden"]
        proprio = item["current_proprio"]
        if not torch.is_tensor(hidden):
            hidden = torch.as_tensor(hidden)
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)
        if (
            hidden.dtype != torch.float32
            or hidden.shape != (STATE_DIM,)
            or proprio.dtype != torch.float32
            or proprio.shape != (self.model.config.proprio_dim,)
            or not bool(torch.isfinite(hidden).all())
            or not bool(torch.isfinite(proprio).all())
        ):
            raise CausalObserverContractError("actor-visible current query tensors changed")
        session["hidden_prefix"].append(hidden.detach().cpu().numpy().copy())
        prefix = np.stack(session["hidden_prefix"], axis=0).astype(np.float32, copy=False)
        history_array, mask_array = build_causal_history_window(prefix)
        history = torch.from_numpy(history_array)
        history_mask = torch.from_numpy(mask_array)
        proprio_cpu = proprio.detach().cpu().contiguous()
        image_value = item["image_feature"]
        image_feature: torch.Tensor | None
        if image_value is None:
            image_feature = None
            image_receipt = make_absent_image_receipt()
        else:
            image_feature = (
                image_value if torch.is_tensor(image_value)
                else torch.as_tensor(image_value)
            )
            if (
                image_feature.dtype != torch.float32
                or image_feature.shape != (self.model.config.image_feature_dim,)
                or not bool(torch.isfinite(image_feature).all())
            ):
                raise CausalObserverContractError("actor-visible image feature changed")
            extractor_sha = self.model.adapter_contracts[actor_name][
                "image_feature_extractor_file_sha256"
            ]
            if not _is_sha(extractor_sha):
                raise CausalObserverContractError("image extractor is not frozen")
            image_feature = image_feature.detach().cpu().contiguous()
            image_receipt = make_image_receipt(
                image_feature,
                extractor_file_sha256=str(extractor_sha),
                frame_query_index=step_index,
            )
        actor_contract = self.model.adapter_contracts[actor_name]
        receipt = make_input_receipt(
            history=history,
            history_mask=history_mask,
            proprio=proprio_cpu,
            actor_name=actor_name,
            policy_family=str(actor_contract["policy_family"]),
            state_feature_source_sha256=str(
                actor_contract["state_feature_source_sha256"]
            ),
            current_query_index=step_index,
            valid_history_steps=min(step_index + 1, MAX_HISTORY_STEPS),
            image_feature_receipt=image_receipt,
            execution_receipt=execution_receipt,
        )
        device = next(self.model.parameters()).device
        prediction = self.model.observe(
            history[None].to(device),
            history_mask[None].to(device),
            proprio_cpu[None].to(device),
            actor_names=[actor_name],
            receipts=[receipt],
            image_features=(
                None if image_feature is None else image_feature[None].to(device)
            ),
        )
        if not bool(prediction.applicability[0]):
            raise CausalObserverContractError(
                f"observer output rejected: {prediction.applicability_reason[0]}"
            )
        event_id = int(prediction.current_event_id[0].detach().cpu())
        predicates = {
            name: bool(prediction.current_predicates[0, index].detach().cpu())
            for index, name in enumerate(EXPECTED_PREDICATES)
        }
        confidence = float(prediction.confidence[0].detach().cpu())
        prediction_document = {
            "event_probability": prediction.current_event_probability[0]
            .detach().cpu().tolist(),
            "predicate_probability": prediction.current_predicate_probability[0]
            .detach().cpu().tolist(),
            "current_event_id": event_id,
            "current_predicates": predicates,
        }
        session["next_step_index"] += 1
        return VerifiedCausalObservation(
            current_event_id=event_id,
            current_predicates=predicates,
            confidence=confidence,
            minimum_joint_confidence=float(
                self.model.calibration["minimum_joint_confidence"]
            ),
            input_receipt_sha256=str(receipt["receipt_sha256"]),
            prediction_sha256=canonical_sha256(prediction_document),
            calibration_sha256=str(self.model.calibration["calibration_sha256"]),
        )


def load_frozen_causal_observer_runtime(
    frozen_artifact_root: Path | str,
) -> FrozenCausalObserverRuntimeV1:
    unresolved_root = Path(frozen_artifact_root)
    if unresolved_root.is_symlink():
        raise CausalObserverContractError("observer artifact root cannot be a symlink")
    try:
        root = unresolved_root.resolve(strict=True)
    except OSError as error:
        raise CausalObserverContractError(
            "observer artifact root does not exist"
        ) from error
    if not root.is_dir():
        raise CausalObserverContractError("observer artifact root must be a real directory")
    authority_path = _safe_declared_file(root, "authority_manifest.json", "authority manifest")
    manifest = _read_json_artifact(authority_path, "authority manifest")
    manifest_fields = {
        "format", "status", "observer_core_file_sha256", "artifacts",
        "observer_config_sha256", "observer_checkpoint_file_sha256",
        "training_contract_sha256", "actor_adapter_manifest_sha256",
        "actor_adapter_set_sha256", "actor_adapter_checkpoint_set_sha256",
        "calibration_sha256", "promotion_evidence_sha256", "deployment_sha256",
        "authority_manifest_sha256",
    }
    _exact_keys(manifest, manifest_fields, "frozen observer authority manifest")
    manifest_logical = dict(manifest)
    manifest_digest = manifest_logical.pop("authority_manifest_sha256")
    if (
        manifest["format"] != FROZEN_AUTHORITY_MANIFEST_FORMAT
        or manifest["status"] != "frozen_promoted_evaluation400_v4_rerank"
        or not _is_sha(manifest_digest)
        or manifest_digest != canonical_sha256(manifest_logical)
    ):
        raise CausalObserverContractError("frozen observer authority manifest is invalid")
    actual_core_file_sha = file_sha256(Path(__file__).resolve())
    if manifest["observer_core_file_sha256"] != actual_core_file_sha:
        raise CausalObserverContractError("installed observer core differs from frozen authority")
    artifacts = _exact_keys(
        manifest["artifacts"],
        {
            "observer_config", "observer_checkpoint", "training_contract",
            "actor_adapter_manifest", "calibration", "promotion_evidence",
            "deployment",
        },
        "frozen observer artifact registry",
    )
    resolved: dict[str, Path] = {}
    declared_file_shas: dict[str, str] = {}
    for role, descriptor in artifacts.items():
        row = _exact_keys(descriptor, {"file", "file_sha256"}, f"{role} descriptor")
        path = _safe_declared_file(root, row["file"], role)
        actual_sha = file_sha256(path)
        if not _is_sha(row["file_sha256"]) or row["file_sha256"] != actual_sha:
            raise CausalObserverContractError(f"{role} file SHA differs")
        resolved[role] = path
        declared_file_shas[role] = actual_sha
    if manifest["observer_checkpoint_file_sha256"] != declared_file_shas[
        "observer_checkpoint"
    ]:
        raise CausalObserverContractError("observer checkpoint file binding changed")
    config, config_document = validate_observer_config_document(
        _read_json_artifact(resolved["observer_config"], "observer config")
    )
    training = _read_json_artifact(resolved["training_contract"], "training contract")
    calibration = _read_json_artifact(resolved["calibration"], "calibration")
    promotion = _read_json_artifact(resolved["promotion_evidence"], "promotion evidence")
    deployment = _read_json_artifact(resolved["deployment"], "deployment")
    adapter_manifest = _read_json_artifact(
        resolved["actor_adapter_manifest"], "actor adapter manifest"
    )
    if (
        config_document["config_sha256"] != manifest["observer_config_sha256"]
        or training.get("contract_sha256") != manifest["training_contract_sha256"]
        or calibration.get("calibration_sha256") != manifest["calibration_sha256"]
        or promotion.get("promotion_receipt_sha256")
        != manifest["promotion_evidence_sha256"]
        or deployment.get("deployment_sha256") != manifest["deployment_sha256"]
        or adapter_manifest.get("manifest_sha256")
        != manifest["actor_adapter_manifest_sha256"]
    ):
        raise CausalObserverContractError("observer semantic artifact binding changed")
    adapter_fields = {
        "format", "training_contract_sha256", "ordered_adapters",
        "actor_adapter_set_sha256", "actor_adapter_checkpoint_set_sha256",
        "manifest_sha256",
    }
    _exact_keys(adapter_manifest, adapter_fields, "actor adapter manifest")
    adapter_logical = dict(adapter_manifest)
    adapter_manifest_digest = adapter_logical.pop("manifest_sha256")
    rows = adapter_manifest["ordered_adapters"]
    if (
        adapter_manifest["format"] != ADAPTER_MANIFEST_FORMAT
        or adapter_manifest["training_contract_sha256"]
        != manifest["training_contract_sha256"]
        or not isinstance(rows, list)
        or not rows
        or adapter_manifest_digest != canonical_sha256(adapter_logical)
    ):
        raise CausalObserverContractError("actor adapter manifest is invalid")
    actor_names = [record.get("actor_name") for record in training.get("actors", [])]
    if [row.get("actor_name") for row in rows if isinstance(row, Mapping)] != actor_names:
        raise CausalObserverContractError("actor adapter order differs from training actors")
    adapter_contracts: dict[str, Mapping[str, Any]] = {}
    adapter_states: dict[str, Mapping[str, torch.Tensor]] = {}
    adapter_contract_shas: list[str] = []
    adapter_checkpoint_shas: list[str] = []
    for row_value in rows:
        row = _exact_keys(
            row_value,
            {"actor_name", "adapter_contract", "checkpoint_file", "checkpoint_file_sha256"},
            "actor adapter row",
        )
        actor_name = row["actor_name"]
        if not isinstance(actor_name, str) or actor_name in adapter_contracts:
            raise CausalObserverContractError("actor adapter identity is invalid")
        checkpoint_path = _safe_declared_file(root, row["checkpoint_file"], "adapter checkpoint")
        checkpoint_file_sha = file_sha256(checkpoint_path)
        if row["checkpoint_file_sha256"] != checkpoint_file_sha:
            raise CausalObserverContractError("adapter checkpoint file SHA differs")
        checkpoint = _read_tensor_artifact(checkpoint_path, "adapter checkpoint")
        _exact_keys(
            checkpoint,
            {
                "format", "actor_name", "adapter_contract_sha256",
                "adapter_state_sha256", "adapter_state_dict",
            },
            "adapter checkpoint",
        )
        state = _validate_tensor_mapping(
            checkpoint["adapter_state_dict"], "adapter checkpoint state"
        )
        contract = row["adapter_contract"]
        if (
            checkpoint["format"] != ADAPTER_CHECKPOINT_FORMAT
            or checkpoint["actor_name"] != actor_name
            or not isinstance(contract, Mapping)
            or checkpoint["adapter_contract_sha256"]
            != contract.get("adapter_contract_sha256")
            or checkpoint["adapter_state_sha256"] != tensor_bundle_sha256(state)
            or checkpoint["adapter_state_sha256"]
            != contract.get("adapter_state_sha256")
        ):
            raise CausalObserverContractError("adapter checkpoint binding changed")
        adapter_contracts[actor_name] = dict(contract)
        adapter_states[actor_name] = state
        adapter_contract_shas.append(str(contract["adapter_contract_sha256"]))
        adapter_checkpoint_shas.append(checkpoint_file_sha)
    adapter_set_sha = canonical_sha256(adapter_contract_shas)
    adapter_checkpoint_set_sha = canonical_sha256(adapter_checkpoint_shas)
    if (
        adapter_set_sha != adapter_manifest["actor_adapter_set_sha256"]
        or adapter_checkpoint_set_sha
        != adapter_manifest["actor_adapter_checkpoint_set_sha256"]
        or adapter_set_sha != manifest["actor_adapter_set_sha256"]
        or adapter_checkpoint_set_sha
        != manifest["actor_adapter_checkpoint_set_sha256"]
    ):
        raise CausalObserverContractError("actor adapter set commitment changed")
    checkpoint = _read_tensor_artifact(
        resolved["observer_checkpoint"], "observer core checkpoint"
    )
    _exact_keys(
        checkpoint,
        {
            "format", "observer_core_file_sha256", "observer_config_sha256",
            "training_contract_sha256", "core_tensor_set_sha256",
            "core_state_dict",
        },
        "observer core checkpoint",
    )
    core_state = _validate_tensor_mapping(
        checkpoint["core_state_dict"], "observer core checkpoint state"
    )
    if (
        checkpoint["format"] != CORE_CHECKPOINT_FORMAT
        or checkpoint["observer_core_file_sha256"] != actual_core_file_sha
        or checkpoint["observer_config_sha256"] != config_document["config_sha256"]
        or checkpoint["training_contract_sha256"] != training.get("contract_sha256")
        or checkpoint["core_tensor_set_sha256"] != tensor_bundle_sha256(core_state)
        or any(name.startswith("actor_adapters.") for name in core_state)
    ):
        raise CausalObserverContractError("observer core checkpoint binding changed")
    model = ActorVisibleCausalEventObserverV1(
        config,
        training_contract=training,
        observer_core_file_sha256=actual_core_file_sha,
        observer_checkpoint_file_sha256=declared_file_shas["observer_checkpoint"],
        observer_config_sha256=config_document["config_sha256"],
        actor_adapter_set_sha256=adapter_set_sha,
        actor_adapter_checkpoint_set_sha256=adapter_checkpoint_set_sha,
        adapter_contracts=adapter_contracts,
        adapter_states=adapter_states,
        calibration=calibration,
        deployment=deployment,
    )
    if model.deployment["promotion_evidence"] != promotion:
        raise CausalObserverContractError(
            "standalone and deployed promotion evidence differ"
        )
    expected_core_names = set(observer_core_state_dict(model))
    if set(core_state) != expected_core_names:
        raise CausalObserverContractError("observer core checkpoint parameter set changed")
    merged = model.state_dict()
    for name, tensor in core_state.items():
        merged[name] = tensor
    try:
        model.load_state_dict(merged, strict=True)
    except RuntimeError as error:
        raise CausalObserverContractError("observer core checkpoint shape changed") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise CausalObserverContractError("observer weights are not frozen for inference")
    authority_base = {
        "format": "etsf_smolvla_piper_causal_observer_authority_v4",
        "status": "frozen_promoted_actor_visible_observer_rerank_authorized",
        "observer_core_file_sha256": actual_core_file_sha,
        "observer_checkpoint_file_sha256": declared_file_shas["observer_checkpoint"],
        "training_contract_sha256": str(training["contract_sha256"]),
        "actor_adapter_set_sha256": adapter_set_sha,
        "actor_adapter_checkpoint_set_sha256": adapter_checkpoint_set_sha,
        "calibration_sha256": str(calibration["calibration_sha256"]),
        "deployment_sha256": str(deployment["deployment_sha256"]),
        "promotion_evidence_sha256": str(promotion["promotion_receipt_sha256"]),
        "frozen_authority_manifest_file_sha256": file_sha256(authority_path),
        "promotion_enabled": True,
        "rerank_enabled": True,
        "object_poses_allowed_online": False,
        "simulator_predicate_reconstruction_allowed": False,
        "hardcoded_event_fallback_allowed": False,
    }
    authority = {
        **authority_base,
        "authority_sha256": canonical_sha256(authority_base),
    }
    return FrozenCausalObserverRuntimeV1(
        model=model,
        authority=authority,
        frozen_root=root,
    )


__all__ = [
    "ADAPTER_CHECKPOINT_FORMAT",
    "ADAPTER_CONTRACT_FORMAT",
    "ADAPTER_MANIFEST_FORMAT",
    "ActorVisibleCausalEventObserverV1",
    "CONFIG_FORMAT",
    "CORE_CHECKPOINT_FORMAT",
    "CausalObserverConfig",
    "CausalObserverContractError",
    "CausalObserverPrediction",
    "EmbodimentResidualAdapter",
    "EVALUATION400_V3_TARGET",
    "EVALUATION400_V4_TARGET",
    "FORMAT",
    "FROZEN_AUTHORITY_MANIFEST_FORMAT",
    "FrozenCausalObserverRuntimeV1",
    "MAX_HISTORY_STEPS",
    "MONITOR_ONLY_TARGET",
    "PROMOTION_EVIDENCE_FORMAT",
    "STATE_DIM",
    "VerifiedCausalObservation",
    "build_causal_history_window",
    "canonical_sha256",
    "causal_history_contract",
    "float32_feature_sha256",
    "file_sha256",
    "load_frozen_causal_observer_runtime",
    "make_absent_execution_receipt",
    "make_absent_image_receipt",
    "make_actor_adapter_contract",
    "make_calibration",
    "make_deployment",
    "make_execution_receipt",
    "make_image_receipt",
    "make_input_receipt",
    "observer_config_document",
    "observer_core_state_dict",
    "tensor_bundle_sha256",
    "training_supervision_contract",
    "validate_calibration",
    "validate_observer_config_document",
    "validate_promotion_evidence",
]
