#!/usr/bin/env python3
"""Dependency-injected evaluation400 v4 condition runner integration.

The module has no simulator launcher or target-data I/O.  Its authority entry
point realizes one frozen observer directory before execution; condition
execution then delegates reset, actor-visible query inputs, step, recovery
inference, and in-memory target trace construction to a caller-supplied backend.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

import smolvla_piper_evaluation400_audit_contract_v1 as audit
import smolvla_piper_causal_event_observer_v1 as causal_observer


CONDITION_NAMES = (
    "baseline",
    "success_only_guarded",
    "composite_rank_ungated",
    "etsf",
)
REQUEST_FORMAT = "etsf_smolvla_piper_evaluation400_condition_request_v4"
REQUEST_STATUS = "root_prediction_worm_acknowledged_before_condition"
DECISION_INPUT_FORMAT = "etsf_smolvla_piper_evaluation400_root_decision_input_v4"
DECISION_INPUT_STATUS = "preoutcome_four_condition_selector_input"
ROOT_ACK_FORMAT = "etsf_smolvla_piper_evaluation400_root_precommit_broker_ack_v4"
ROOT_ACK_STATUS = "worm_root_commit_acknowledged_conditions_authorized"
RESULT_FORMAT = "etsf_smolvla_piper_evaluation400_condition_result_v4"
RESULT_STATUS = "complete_encrypted_target_single_condition"
OBSERVER_AUTHORITY_FORMAT = "etsf_smolvla_piper_causal_observer_authority_v4"
OBSERVER_RECEIPT_FORMAT = "etsf_smolvla_piper_causal_observer_query_receipt_v4"
PREDICATE_NAMES = ("moved", "lifted", "near_goal", "stationary", "success")
MAX_EPISODE_STEPS = 200


class ConditionRunnerV4Error(RuntimeError):
    """The v4 runner integration failed closed."""


class ConditionBackendV4(Protocol):
    max_steps: int

    def reset(self, pair_id: str) -> tuple[Any, Mapping[str, Any]]: ...

    def query(self, observation: Any, step_index: int) -> Mapping[str, Any]: ...

    def recovery_prediction(
        self, query: Mapping[str, Any], chosen_candidate_index: int,
    ) -> Mapping[str, Any]: ...

    def step(self, action: Any) -> tuple[Any, bool, bool, Mapping[str, Any]]: ...

    def target_trace(self) -> Mapping[str, Any]: ...


RecoveryBroker = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _require_sha(value: Any, role: str) -> str:
    if not audit.is_sha256(value):
        raise ConditionRunnerV4Error(f"{role} must be exact SHA-256")
    return str(value)


def _require_int(
    value: Any, role: str, *, minimum: int = 0, expected: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise ConditionRunnerV4Error(f"{role} must be an exact non-bool integer")
    if expected is not None and value != expected:
        raise ConditionRunnerV4Error(f"{role} differs from exact authority")
    return value


def _require_float(value: Any, role: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionRunnerV4Error(f"{role} must be finite numeric, not bool")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ConditionRunnerV4Error(f"{role} is not a valid finite value")
    return result


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    normalized = dict(base)
    return {**normalized, field: audit.canonical_sha256(normalized)}


def _verify(
    value: Mapping[str, Any], *, field: str, fields: set[str], role: str,
) -> str:
    if not isinstance(value, Mapping) or set(value) != fields | {field}:
        raise ConditionRunnerV4Error(f"{role} fields changed")
    logical = value.get(field)
    _require_sha(logical, f"{role} logical SHA")
    base = {key: child for key, child in value.items() if key != field}
    if logical != audit.canonical_sha256(base):
        raise ConditionRunnerV4Error(f"{role} canonical SHA mismatch")
    return str(logical)


def build_causal_observer_authority(
    *, frozen_artifact_root: str,
) -> causal_observer.FrozenCausalObserverRuntimeV1:
    """Realize authority from files; raw caller-supplied digest labels are forbidden."""

    try:
        return causal_observer.load_frozen_causal_observer_runtime(
            frozen_artifact_root
        )
    except causal_observer.CausalObserverContractError as error:
        raise ConditionRunnerV4Error("causal observer artifact realization failed") from error


def validate_causal_observer_authority(
    value: causal_observer.FrozenCausalObserverRuntimeV1,
) -> str:
    if not isinstance(value, causal_observer.FrozenCausalObserverRuntimeV1):
        raise ConditionRunnerV4Error(
            "causal observer authority must be a realized frozen runtime"
        )
    try:
        value.validate_frozen_realization()
    except causal_observer.CausalObserverContractError as error:
        raise ConditionRunnerV4Error(
            "causal observer frozen realization changed"
        ) from error
    authority = value.authority
    fields = {
        "format", "status", "observer_core_file_sha256",
        "observer_checkpoint_file_sha256", "training_contract_sha256",
        "actor_adapter_set_sha256", "actor_adapter_checkpoint_set_sha256",
        "calibration_sha256", "deployment_sha256", "promotion_evidence_sha256",
        "frozen_authority_manifest_file_sha256",
        "promotion_enabled", "rerank_enabled", "object_poses_allowed_online",
        "simulator_predicate_reconstruction_allowed",
        "hardcoded_event_fallback_allowed",
    }
    logical = _verify(
        authority, field="authority_sha256", fields=fields,
        role="causal observer authority",
    )
    if (
        authority.get("format") != OBSERVER_AUTHORITY_FORMAT
        or authority.get("status")
        != "frozen_promoted_actor_visible_observer_rerank_authorized"
        or authority.get("promotion_enabled") is not True
        or authority.get("rerank_enabled") is not True
        or authority.get("object_poses_allowed_online") is not False
        or authority.get("simulator_predicate_reconstruction_allowed") is not False
        or authority.get("hardcoded_event_fallback_allowed") is not False
        or value.model.deployment.get("integration_target")
        != causal_observer.EVALUATION400_V4_TARGET
        or value.model.deployment.get("rerank_enabled") is not True
        or value.model.training
        or any(parameter.requires_grad for parameter in value.model.parameters())
    ):
        raise ConditionRunnerV4Error("causal observer is not production-rerank authorized")
    for name in fields - {
        "format", "status", "promotion_enabled", "rerank_enabled",
        "object_poses_allowed_online", "simulator_predicate_reconstruction_allowed",
        "hardcoded_event_fallback_allowed",
    }:
        _require_sha(authority.get(name), f"causal observer {name}")
    return logical


def build_causal_observer_receipt(
    *, runtime: causal_observer.FrozenCausalObserverRuntimeV1,
    pair_id: str, condition_id: str, step_index: int,
    observation: causal_observer.VerifiedCausalObservation,
) -> dict[str, Any]:
    authority_sha256 = validate_causal_observer_authority(runtime)
    _require_sha(authority_sha256, "observer authority")
    _require_sha(pair_id, "observer pair")
    if not isinstance(observation, causal_observer.VerifiedCausalObservation):
        raise ConditionRunnerV4Error("observer output was not produced by frozen runtime")
    _require_sha(observation.input_receipt_sha256, "observer input receipt")
    _require_sha(observation.prediction_sha256, "observer prediction")
    _require_sha(observation.calibration_sha256, "observer calibration")
    _require_int(step_index, "observer step")
    _require_int(observation.current_event_id, "observer current event")
    if observation.current_event_id >= 5:
        raise ConditionRunnerV4Error("observer event escaped canonical vocabulary")
    predicates = dict(observation.current_predicates)
    if set(predicates) != set(PREDICATE_NAMES) or any(
        type(value) is not bool for value in predicates.values()
    ):
        raise ConditionRunnerV4Error("observer predicates changed")
    confidence_value = _require_float(observation.confidence, "observer confidence")
    minimum_confidence = _require_float(
        observation.minimum_joint_confidence, "observer minimum confidence"
    )
    if (
        not 0.0 <= confidence_value <= 1.0
        or not 0.0 <= minimum_confidence <= 1.0
        or confidence_value < minimum_confidence
        or observation.calibration_sha256
        != runtime.authority["calibration_sha256"]
    ):
        raise ConditionRunnerV4Error("observer confidence/calibration failed closed")
    base = {
        "format": OBSERVER_RECEIPT_FORMAT,
        "status": "actor_visible_promoted_observation_applicable",
        "authority_sha256": authority_sha256,
        "pair_id": pair_id,
        "condition_id": condition_id,
        "step_index": step_index,
        "input_receipt_sha256": observation.input_receipt_sha256,
        "prediction_sha256": observation.prediction_sha256,
        "calibration_sha256": observation.calibration_sha256,
        "minimum_joint_confidence": minimum_confidence,
        "current_event_id": observation.current_event_id,
        "current_predicates": predicates,
        "confidence": confidence_value,
        "applicable": True,
        "object_pose_fields_present": False,
        "simulator_privileged_state_read": False,
        "hardcoded_event_fallback_used": False,
    }
    return _signed(base, "receipt_sha256")


def validate_causal_observer_receipt(
    value: Mapping[str, Any], *,
    runtime: causal_observer.FrozenCausalObserverRuntimeV1, pair_id: str,
    condition_id: str, step_index: int,
) -> tuple[int, dict[str, bool]]:
    authority_sha256 = validate_causal_observer_authority(runtime)
    fields = {
        "format", "status", "authority_sha256", "pair_id", "condition_id",
        "step_index", "input_receipt_sha256", "prediction_sha256",
        "calibration_sha256", "minimum_joint_confidence", "current_event_id",
        "current_predicates", "confidence", "applicable",
        "object_pose_fields_present", "simulator_privileged_state_read",
        "hardcoded_event_fallback_used",
    }
    _verify(value, field="receipt_sha256", fields=fields, role="observer receipt")
    if (
        value.get("format") != OBSERVER_RECEIPT_FORMAT
        or value.get("status") != "actor_visible_promoted_observation_applicable"
        or value.get("authority_sha256") != authority_sha256
        or value.get("pair_id") != pair_id
        or value.get("condition_id") != condition_id
        or value.get("step_index") != step_index
        or value.get("applicable") is not True
        or value.get("object_pose_fields_present") is not False
        or value.get("simulator_privileged_state_read") is not False
        or value.get("hardcoded_event_fallback_used") is not False
    ):
        raise ConditionRunnerV4Error("observer receipt provenance/applicability changed")
    _require_sha(value.get("input_receipt_sha256"), "observer input receipt")
    _require_sha(value.get("prediction_sha256"), "observer prediction")
    if value.get("calibration_sha256") != runtime.authority["calibration_sha256"]:
        raise ConditionRunnerV4Error("observer receipt calibration changed")
    event = _require_int(value.get("current_event_id"), "observer current event")
    if event >= 5:
        raise ConditionRunnerV4Error("observer event escaped canonical vocabulary")
    predicates = value.get("current_predicates")
    if not isinstance(predicates, Mapping) or set(predicates) != set(PREDICATE_NAMES) or any(
        type(child) is not bool for child in predicates.values()
    ):
        raise ConditionRunnerV4Error("observer predicates changed")
    confidence = _require_float(value.get("confidence"), "observer confidence")
    minimum = _require_float(
        value.get("minimum_joint_confidence"), "observer minimum confidence"
    )
    if (
        not 0.0 <= confidence <= 1.0
        or not 0.0 <= minimum <= 1.0
        or confidence < minimum
        or minimum
        != float(runtime.model.calibration["minimum_joint_confidence"])
    ):
        raise ConditionRunnerV4Error("observer receipt low confidence accepted")
    return event, dict(predicates)


def build_root_decision_input(
    *, root_precommit: Mapping[str, Any], fallback_candidate_index: int,
    mean_success_probability: Sequence[float],
    mean_composite_rank_score: Sequence[float],
    structured_uncertainty: Sequence[float],
    success_margin_threshold: float,
    composite_margin_threshold: float,
    maximum_global_uncertainty: float,
    maximum_pair_uncertainty: float,
) -> dict[str, Any]:
    root_sha = audit.validate_root_precommit(root_precommit)
    indices = list(root_precommit["prediction_candidate_indices"])
    _require_int(fallback_candidate_index, "fallback candidate")
    if fallback_candidate_index != indices[0] or len(indices) < 2:
        raise ConditionRunnerV4Error("fallback must be lowest of at least two legal candidates")

    def vector(values: Sequence[float], role: str) -> list[float]:
        result = [_require_float(item, f"{role} {index}") for index, item in enumerate(values)]
        if len(result) != len(indices):
            raise ConditionRunnerV4Error(f"{role} candidate coverage changed")
        return result

    success = vector(mean_success_probability, "mean success probability")
    rank = vector(mean_composite_rank_score, "mean composite rank")
    uncertainty = vector(structured_uncertainty, "structured uncertainty")
    if any(not 0.0 <= item <= 1.0 for item in success + uncertainty):
        raise ConditionRunnerV4Error("probability/uncertainty escaped [0,1]")
    thresholds = {
        "success_margin": _require_float(
            success_margin_threshold, "success margin", minimum=0.0
        ),
        "composite_margin": _require_float(
            composite_margin_threshold, "composite margin", minimum=0.0
        ),
        "maximum_global_uncertainty": _require_float(
            maximum_global_uncertainty, "global uncertainty", minimum=0.0
        ),
        "maximum_pair_uncertainty": _require_float(
            maximum_pair_uncertainty, "pair uncertainty", minimum=0.0
        ),
    }
    if thresholds["maximum_global_uncertainty"] > 1.0 or thresholds[
        "maximum_pair_uncertainty"
    ] > 1.0:
        raise ConditionRunnerV4Error("uncertainty threshold escaped [0,1]")
    base = {
        "format": DECISION_INPUT_FORMAT,
        "status": DECISION_INPUT_STATUS,
        "root_prediction_commit_sha256": root_sha,
        "root_tensor_set_sha256": root_precommit["tensor_set_sha256"],
        "pair_id": root_precommit["pair_id"],
        "prediction_candidate_indices": indices,
        "fallback_candidate_index": fallback_candidate_index,
        "mean_success_probability": success,
        "mean_composite_rank_score": rank,
        "structured_five_head_uncertainty": uncertainty,
        "thresholds": thresholds,
        "score_contract": {
            "success_only": "mean_five_member_calibrated_factual_success_probability",
            "composite": "mean_five_member_adjusted_source_composite_rank_score",
            "composite_is_success_logit": False,
            "composite_is_success_probability": False,
            "alternative_set": "all_legal_candidates_except_lowest_legal_baseline",
            "margin_comparison": "strict_greater_than_frozen_threshold",
        },
    }
    return _signed(base, "decision_input_sha256")


def validate_root_decision_input(
    value: Mapping[str, Any], *, root_precommit: Mapping[str, Any],
) -> str:
    root_sha = audit.validate_root_precommit(root_precommit)
    fields = {
        "format", "status", "root_prediction_commit_sha256",
        "root_tensor_set_sha256", "pair_id", "prediction_candidate_indices",
        "fallback_candidate_index", "mean_success_probability",
        "mean_composite_rank_score", "structured_five_head_uncertainty",
        "thresholds", "score_contract",
    }
    logical = _verify(
        value, field="decision_input_sha256", fields=fields,
        role="root decision input",
    )
    indices = root_precommit["prediction_candidate_indices"]
    if (
        value.get("format") != DECISION_INPUT_FORMAT
        or value.get("status") != DECISION_INPUT_STATUS
        or value.get("root_prediction_commit_sha256") != root_sha
        or value.get("root_tensor_set_sha256") != root_precommit["tensor_set_sha256"]
        or value.get("pair_id") != root_precommit["pair_id"]
        or value.get("prediction_candidate_indices") != indices
        or value.get("fallback_candidate_index") != indices[0]
        or value.get("score_contract") != {
            "success_only": "mean_five_member_calibrated_factual_success_probability",
            "composite": "mean_five_member_adjusted_source_composite_rank_score",
            "composite_is_success_logit": False,
            "composite_is_success_probability": False,
            "alternative_set": "all_legal_candidates_except_lowest_legal_baseline",
            "margin_comparison": "strict_greater_than_frozen_threshold",
        }
    ):
        raise ConditionRunnerV4Error("root decision input binding changed")
    _require_int(value.get("fallback_candidate_index"), "fallback candidate")
    if len(indices) < 2:
        raise ConditionRunnerV4Error("root decision needs at least two legal candidates")
    for field_name, bounded in (
        ("mean_success_probability", True),
        ("mean_composite_rank_score", False),
        ("structured_five_head_uncertainty", True),
    ):
        vector = value.get(field_name)
        if not isinstance(vector, list) or len(vector) != len(indices):
            raise ConditionRunnerV4Error(f"root decision vector changed: {field_name}")
        for index, child in enumerate(vector):
            numeric = _require_float(child, f"{field_name} {index}")
            if bounded and not 0.0 <= numeric <= 1.0:
                raise ConditionRunnerV4Error(f"{field_name} escaped [0,1]")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "success_margin", "composite_margin", "maximum_global_uncertainty",
        "maximum_pair_uncertainty",
    }:
        raise ConditionRunnerV4Error("root selector thresholds changed")
    for field_name in ("success_margin", "composite_margin"):
        _require_float(thresholds.get(field_name), field_name, minimum=0.0)
    for field_name in ("maximum_global_uncertainty", "maximum_pair_uncertainty"):
        child = _require_float(thresholds.get(field_name), field_name, minimum=0.0)
        if child > 1.0:
            raise ConditionRunnerV4Error("uncertainty threshold escaped [0,1]")
    return logical


def build_root_broker_ack(
    root_precommit: Mapping[str, Any], *, ledger_event_sha256: str,
    ledger_event_index: int,
) -> dict[str, Any]:
    root_sha = audit.validate_root_precommit(root_precommit)
    _require_sha(ledger_event_sha256, "root ledger event")
    _require_int(ledger_event_index, "root ledger event index")
    base = {
        "format": ROOT_ACK_FORMAT,
        "status": ROOT_ACK_STATUS,
        "pair_id": root_precommit["pair_id"],
        "pair_ordinal": root_precommit["pair_ordinal"],
        "root_prediction_commit_sha256": root_sha,
        "ledger_event_sha256": ledger_event_sha256,
        "ledger_event_index": ledger_event_index,
        "all_four_conditions_authorized": True,
    }
    return _signed(base, "ack_sha256")


def validate_root_broker_ack(
    value: Mapping[str, Any], *, root_precommit: Mapping[str, Any],
    expected_ledger_event_sha256: str | None = None,
) -> str:
    root_sha = audit.validate_root_precommit(root_precommit)
    fields = {
        "format", "status", "pair_id", "pair_ordinal",
        "root_prediction_commit_sha256", "ledger_event_sha256",
        "ledger_event_index", "all_four_conditions_authorized",
    }
    logical = _verify(
        value, field="ack_sha256", fields=fields, role="root broker ACK"
    )
    if (
        value.get("format") != ROOT_ACK_FORMAT
        or value.get("status") != ROOT_ACK_STATUS
        or value.get("pair_id") != root_precommit["pair_id"]
        or value.get("pair_ordinal") != root_precommit["pair_ordinal"]
        or value.get("root_prediction_commit_sha256") != root_sha
        or value.get("all_four_conditions_authorized") is not True
    ):
        raise ConditionRunnerV4Error("root broker ACK binding changed")
    _require_int(value.get("pair_ordinal"), "root ACK pair ordinal")
    _require_int(value.get("ledger_event_index"), "root ACK event index")
    _require_sha(value.get("ledger_event_sha256"), "root ACK ledger event")
    if expected_ledger_event_sha256 is not None and value[
        "ledger_event_sha256"
    ] != expected_ledger_event_sha256:
        raise ConditionRunnerV4Error("root ACK references the wrong ledger event")
    return logical


def build_condition_request(
    *, protocol_core_v4_sha256: str, pair_id: str, pair_ordinal: int,
    condition_id: str, shared_snapshot_sha256: str,
    root_prediction_commit_sha256: str, root_ack_sha256: str,
    schema6_runtime_contract_sha256: str,
    causal_observer_authority_sha256: str,
) -> dict[str, Any]:
    for value, role in (
        (protocol_core_v4_sha256, "v4 core"), (pair_id, "pair ID"),
        (shared_snapshot_sha256, "shared snapshot"),
        (root_prediction_commit_sha256, "root prediction commit"),
        (root_ack_sha256, "root ACK"),
        (schema6_runtime_contract_sha256, "runtime contract"),
        (causal_observer_authority_sha256, "causal observer authority"),
    ):
        _require_sha(value, role)
    _require_int(pair_ordinal, "pair ordinal")
    if condition_id not in CONDITION_NAMES:
        raise ConditionRunnerV4Error("condition is not in the frozen v4 matrix")
    base = {
        "format": REQUEST_FORMAT,
        "status": REQUEST_STATUS,
        "protocol_core_v4_sha256": protocol_core_v4_sha256,
        "pair_id": pair_id,
        "pair_ordinal": pair_ordinal,
        "condition_id": condition_id,
        "condition_position": CONDITION_NAMES.index(condition_id),
        "condition_order": list(CONDITION_NAMES),
        "shared_snapshot_sha256": shared_snapshot_sha256,
        "root_prediction_commit_sha256": root_prediction_commit_sha256,
        "root_ack_sha256": root_ack_sha256,
        "schema6_runtime_contract_sha256": schema6_runtime_contract_sha256,
        "causal_observer_authority_sha256": causal_observer_authority_sha256,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "attempt": 0,
        "retry_count": 0,
        "outcome_visible_before_condition_start": False,
    }
    return _signed(base, "request_sha256")


def validate_condition_request(value: Mapping[str, Any]) -> str:
    fields = {
        "format", "status", "protocol_core_v4_sha256", "pair_id",
        "pair_ordinal", "condition_id", "condition_position", "condition_order",
        "shared_snapshot_sha256", "root_prediction_commit_sha256",
        "root_ack_sha256", "schema6_runtime_contract_sha256",
        "causal_observer_authority_sha256",
        "max_episode_steps", "attempt", "retry_count",
        "outcome_visible_before_condition_start",
    }
    logical = _verify(
        value, field="request_sha256", fields=fields, role="condition request"
    )
    condition = value.get("condition_id")
    if (
        value.get("format") != REQUEST_FORMAT
        or value.get("status") != REQUEST_STATUS
        or condition not in CONDITION_NAMES
        or value.get("condition_order") != list(CONDITION_NAMES)
        or value.get("condition_position") != CONDITION_NAMES.index(condition)
        or value.get("outcome_visible_before_condition_start") is not False
    ):
        raise ConditionRunnerV4Error("condition request matrix changed")
    for field_name in (
        "protocol_core_v4_sha256", "pair_id", "shared_snapshot_sha256",
        "root_prediction_commit_sha256", "root_ack_sha256",
        "schema6_runtime_contract_sha256",
        "causal_observer_authority_sha256",
    ):
        _require_sha(value.get(field_name), f"request {field_name}")
    for field_name, expected in (
        ("pair_ordinal", None), ("condition_position", None),
        ("max_episode_steps", MAX_EPISODE_STEPS), ("attempt", 0),
        ("retry_count", 0),
    ):
        _require_int(value.get(field_name), f"request {field_name}", expected=expected)
    return logical


def select_condition_root(
    condition_id: str, decision_input: Mapping[str, Any], *,
    root_precommit: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    decision_input_sha = validate_root_decision_input(
        decision_input, root_precommit=root_precommit
    )
    if condition_id not in CONDITION_NAMES:
        raise ConditionRunnerV4Error("unknown v4 condition")
    indices = list(decision_input["prediction_candidate_indices"])
    fallback = int(decision_input["fallback_candidate_index"])
    baseline_row = indices.index(fallback)
    alternatives = [row for row, index in enumerate(indices) if index != fallback]
    success = np.asarray(decision_input["mean_success_probability"], dtype=np.float64)
    rank = np.asarray(decision_input["mean_composite_rank_score"], dtype=np.float64)
    uncertainty = np.asarray(
        decision_input["structured_five_head_uncertainty"], dtype=np.float64
    )
    thresholds = decision_input["thresholds"]
    score_name: str
    gate_applied: bool
    if condition_id == "baseline":
        proposed = fallback
        margin = 0.0
        accepted = False
        score_name = "lowest_legal_feasibility_root_candidate"
        gate_applied = False
    else:
        score = success if condition_id == "success_only_guarded" else rank
        margins = score - score[baseline_row]
        proposed_row = min(
            alternatives, key=lambda row: (-float(margins[row]), int(indices[row]))
        )
        proposed = int(indices[proposed_row])
        margin = float(margins[proposed_row])
        margin_threshold = float(
            thresholds[
                "success_margin"
                if condition_id == "success_only_guarded"
                else "composite_margin"
            ]
        )
        gate_applied = condition_id in {"success_only_guarded", "etsf"}
        if gate_applied:
            proposed_uncertainty = float(uncertainty[proposed_row])
            baseline_uncertainty = float(uncertainty[baseline_row])
            pair_uncertainty = max(proposed_uncertainty, baseline_uncertainty)
            accepted = bool(
                margin > margin_threshold
                and proposed_uncertainty
                <= float(thresholds["maximum_global_uncertainty"])
                and pair_uncertainty
                <= float(thresholds["maximum_pair_uncertainty"])
            )
        else:
            proposed_uncertainty = float(uncertainty[proposed_row])
            baseline_uncertainty = float(uncertainty[baseline_row])
            pair_uncertainty = max(proposed_uncertainty, baseline_uncertainty)
            accepted = bool(margin > margin_threshold)
        score_name = (
            "mean_five_member_calibrated_factual_success_probability"
            if condition_id == "success_only_guarded"
            else "mean_five_member_adjusted_source_composite_rank_score"
        )
    selected = proposed if accepted else fallback
    proof_base = {
        "condition_id": condition_id,
        "decision_input_sha256": decision_input_sha,
        "fallback_candidate_index": fallback,
        "proposed_candidate_index": proposed,
        "selected_candidate_index": selected,
        "score_contract": score_name,
        "score_margin": margin,
        "strict_margin_accepted": accepted if not gate_applied else (
            margin > float(
                thresholds[
                    "success_margin"
                    if condition_id == "success_only_guarded"
                    else "composite_margin"
                ]
            )
        ),
        "structured_uncertainty_gate_applied": gate_applied,
        "candidate_change_accepted": accepted,
        "composite_score_is_success_logit": False,
        "composite_score_is_success_probability": False,
    }
    return selected, _signed(proof_base, "selector_proof_sha256")


def _validate_query(
    query: Mapping[str, Any], *, expected_step_index: int,
) -> tuple[list[str], list[bool], int]:
    expected = {
        "ordered_candidate_sha256", "candidate_legal",
        "lowest_legal_original_candidate_index", "mapped_actions",
        "pre_action_snapshot_sha256", "actor_visible_observer_inputs",
    }
    if not isinstance(query, Mapping) or set(query) != expected:
        raise ConditionRunnerV4Error("backend query fields changed")
    ordered = query["ordered_candidate_sha256"]
    legal = query["candidate_legal"]
    fallback = query["lowest_legal_original_candidate_index"]
    if (
        not isinstance(ordered, list)
        or not ordered
        or any(not audit.is_sha256(item) for item in ordered)
        or len(set(ordered)) != len(ordered)
        or not isinstance(legal, list)
        or len(legal) != len(ordered)
        or any(type(item) is not bool for item in legal)
        or not any(legal)
    ):
        raise ConditionRunnerV4Error("backend candidate registry changed")
    _require_int(fallback, "query fallback")
    if fallback != legal.index(True):
        raise ConditionRunnerV4Error("backend fallback is not lowest legal")
    _require_sha(query["pre_action_snapshot_sha256"], "pre-action snapshot")
    observer_inputs = query["actor_visible_observer_inputs"]
    if (
        not isinstance(observer_inputs, Mapping)
        or set(observer_inputs)
        != {"actor_name", "current_hidden", "current_proprio", "image_feature"}
    ):
        raise ConditionRunnerV4Error("backend actor-visible observer inputs changed")
    actions = query["mapped_actions"]
    if not isinstance(actions, list) or len(actions) != len(ordered):
        raise ConditionRunnerV4Error("backend mapped action registry changed")
    _require_int(expected_step_index, "expected query step")
    return list(ordered), list(legal), int(fallback)


def execute_condition_v4(
    *, request: Mapping[str, Any], backend: ConditionBackendV4,
    root_precommit: Mapping[str, Any], root_ack: Mapping[str, Any],
    decision_input: Mapping[str, Any], recovery_broker: RecoveryBroker,
    evaluator_public_key_raw: bytes,
    dense_event_targets_fn: Callable[..., Mapping[str, Any]],
    recovery_targets_fn: Callable[..., Mapping[str, Any]],
    object_target_fn: Callable[..., Mapping[str, Any]],
    causal_observer_runtime: causal_observer.FrozenCausalObserverRuntimeV1,
) -> dict[str, Any]:
    request_sha = validate_condition_request(request)
    root_sha = audit.validate_root_precommit(root_precommit)
    root_ack_sha = validate_root_broker_ack(root_ack, root_precommit=root_precommit)
    decision_sha = validate_root_decision_input(
        decision_input, root_precommit=root_precommit
    )
    observer_authority_sha = validate_causal_observer_authority(
        causal_observer_runtime
    )
    if (
        request["root_prediction_commit_sha256"] != root_sha
        or request["root_ack_sha256"] != root_ack_sha
        or request["pair_id"] != root_precommit["pair_id"]
        or request["pair_ordinal"] != root_precommit["pair_ordinal"]
        or request["shared_snapshot_sha256"]
        != root_precommit["shared_snapshot_sha256"]
        or request["schema6_runtime_contract_sha256"]
        != root_precommit["authority"]["schema6_runtime_contract_sha256"]
        or request["causal_observer_authority_sha256"] != observer_authority_sha
    ):
        raise ConditionRunnerV4Error("request/root precommit/ACK identity changed")
    if type(backend.max_steps) is not int or backend.max_steps != MAX_EPISODE_STEPS:
        raise ConditionRunnerV4Error("backend is not the exact 200-step runtime")
    try:
        causal_observer_runtime.start_condition(
            pair_id=str(request["pair_id"]),
            condition_id=str(request["condition_id"]),
        )
    except causal_observer.CausalObserverContractError as error:
        raise ConditionRunnerV4Error(
            "causal observer failed before backend reset"
        ) from error
    observation, identity = backend.reset(str(request["pair_id"]))
    if identity != {
        "pair_id": request["pair_id"],
        "shared_snapshot_sha256": request["shared_snapshot_sha256"],
    }:
        raise ConditionRunnerV4Error("backend reset identity changed")
    root_query = backend.query(observation, 0)
    ordered, legal, fallback = _validate_query(root_query, expected_step_index=0)
    if (
        ordered != root_precommit["ordered_candidate_sha256"]
        or legal != root_precommit["candidate_legal"]
        or fallback != decision_input["fallback_candidate_index"]
    ):
        raise ConditionRunnerV4Error("condition root differs from precommitted root")
    selected, selector_proof = select_condition_root(
        str(request["condition_id"]), decision_input,
        root_precommit=root_precommit,
    )
    if legal[selected] is not True:
        raise ConditionRunnerV4Error("condition selected an illegal candidate")

    broker_state = audit.RecoveryBrokerState(
        pair_id=str(request["pair_id"]), condition_id=str(request["condition_id"])
    )
    recovery_records: list[dict[str, Any]] = []
    success = False
    terminated = truncated = False
    steps: list[dict[str, Any]] = []
    query = root_query
    historical_peak_event_id = 0
    previous_action_sha256: str | None = None
    executed_control_steps: int | None = None
    for step_index in range(MAX_EPISODE_STEPS):
        current_ordered, current_legal, current_fallback = _validate_query(
            query, expected_step_index=step_index
        )
        try:
            verified_observation = causal_observer_runtime.observe_actor_visible_query(
                query["actor_visible_observer_inputs"],
                pair_id=str(request["pair_id"]),
                condition_id=str(request["condition_id"]),
                step_index=step_index,
                previous_action_sha256=previous_action_sha256,
                executed_control_steps=executed_control_steps,
            )
        except causal_observer.CausalObserverContractError as error:
            raise ConditionRunnerV4Error(
                "causal observer rejected actor-visible query before action"
            ) from error
        observer_receipt = build_causal_observer_receipt(
            runtime=causal_observer_runtime,
            pair_id=str(request["pair_id"]),
            condition_id=str(request["condition_id"]),
            step_index=step_index,
            observation=verified_observation,
        )
        observed_event_id, _observed_predicates = validate_causal_observer_receipt(
            observer_receipt,
            runtime=causal_observer_runtime,
            pair_id=str(request["pair_id"]),
            condition_id=str(request["condition_id"]),
            step_index=step_index,
        )
        historical_peak_event_id = max(historical_peak_event_id, observed_event_id)
        chosen = selected if step_index == 0 else current_fallback
        if current_legal[chosen] is not True:
            raise ConditionRunnerV4Error("runner selected an illegal continuation")
        if step_index > 0:
            if historical_peak_event_id < 1:
                recovery_records.append(
                    {
                        "step_index": step_index,
                        "status": "not_applicable_no_prior_progress",
                        "commit_sha256": None,
                        "ack_sha256": None,
                        "ledger_event_sha256": None,
                    }
                )
            else:
                recovery_prediction = backend.recovery_prediction(query, chosen)
                if not isinstance(recovery_prediction, Mapping) or set(
                    recovery_prediction
                ) != {
                    "member_recovery_logits", "conditional_recovery_temperature",
                    "calibration_sha256", "source_rank_member_authority_sha256",
                }:
                    raise ConditionRunnerV4Error(
                        "recovery shadow prediction fields changed"
                    )
                commitment = audit.build_recovery_pre_step_commitment(
                    protocol_core_v4_sha256=request["protocol_core_v4_sha256"],
                    root_prediction_commit_sha256=root_sha,
                    pair_id=request["pair_id"],
                    pair_ordinal=request["pair_ordinal"],
                    condition_id=request["condition_id"],
                    condition_position=request["condition_position"],
                    step_index=step_index,
                    commit_sequence=broker_state.next_sequence,
                    previous_commit_sha256=broker_state.last_commit_sha256,
                    pre_action_snapshot_sha256=query["pre_action_snapshot_sha256"],
                    chosen_candidate_index=chosen,
                    chosen_candidate_sha256=current_ordered[chosen],
                    current_event_id=observed_event_id,
                    historical_peak_event_id=historical_peak_event_id,
                    member_recovery_logits=recovery_prediction[
                        "member_recovery_logits"
                    ],
                    conditional_recovery_temperature=recovery_prediction[
                        "conditional_recovery_temperature"
                    ],
                    calibration_sha256=recovery_prediction["calibration_sha256"],
                    source_rank_member_authority_sha256=recovery_prediction[
                        "source_rank_member_authority_sha256"
                    ],
                    schema6_runtime_contract_sha256=request[
                        "schema6_runtime_contract_sha256"
                    ],
                )
                commit_sha = broker_state.accept_commitment(
                    commitment, expected_step_index=step_index
                )
                ack = recovery_broker(commitment)
                ack_sha = broker_state.accept_ack(
                    ack, expected_step_index=step_index
                )
                recovery_records.append(
                    {
                        "step_index": step_index,
                        "status": "committed_and_broker_acknowledged_pre_step",
                        "commit_sha256": commit_sha,
                        "ack_sha256": ack_sha,
                        "ledger_event_sha256": ack["ledger_event_sha256"],
                    }
                )
        action = query["mapped_actions"][chosen]
        observation, terminated, truncated, info = backend.step(action)
        if (
            type(terminated) is not bool
            or type(truncated) is not bool
            or not isinstance(info, Mapping)
            or set(info) != {"success", "executed_control_steps"}
            or type(info["success"]) is not bool
            or type(info["executed_control_steps"]) is not int
            or info["executed_control_steps"] < 1
        ):
            raise ConditionRunnerV4Error("backend terminal/success contract changed")
        previous_action_sha256 = current_ordered[chosen]
        executed_control_steps = int(info["executed_control_steps"])
        success = success or info["success"]
        steps.append(
            {
                "step_index": step_index,
                "selected_candidate_index": chosen,
                "candidate_sha256": current_ordered[chosen],
                "observer_receipt_sha256": observer_receipt["receipt_sha256"],
                "root_condition_selection": step_index == 0,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated or step_index + 1 >= MAX_EPISODE_STEPS:
            break
        query = backend.query(observation, step_index + 1)
    if broker_state.pending is not None:
        raise ConditionRunnerV4Error("condition ended with unacknowledged recovery commit")
    expected_recovery_steps = list(range(1, len(steps)))
    if [row["step_index"] for row in recovery_records] != expected_recovery_steps:
        raise ConditionRunnerV4Error(
            "not every executed continuation step has a recovery commit/ACK"
        )
    target_trace = backend.target_trace()
    if (
        not isinstance(target_trace, Mapping)
        or target_trace.get("terminal_step") != len(steps)
        or target_trace.get("terminal_success") is not success
    ):
        raise ConditionRunnerV4Error("target trace differs from executed terminal")
    targets = audit.recompute_audit_targets(
        target_trace,
        dense_event_targets_fn=dense_event_targets_fn,
        recovery_targets_fn=recovery_targets_fn,
        object_target_fn=object_target_fn,
    )
    envelope = audit.seal_target_envelope(
        targets,
        evaluator_public_key_raw=evaluator_public_key_raw,
        protocol_core_v4_sha256=request["protocol_core_v4_sha256"],
        pair_id=request["pair_id"],
        condition_id=request["condition_id"],
        root_prediction_commit_sha256=root_sha,
        schema6_runtime_contract_sha256=request[
            "schema6_runtime_contract_sha256"
        ],
    )
    base = {
        "format": RESULT_FORMAT,
        "status": RESULT_STATUS,
        "request_sha256": request_sha,
        "protocol_core_v4_sha256": request["protocol_core_v4_sha256"],
        "pair_id": request["pair_id"],
        "pair_ordinal": request["pair_ordinal"],
        "condition_id": request["condition_id"],
        "condition_position": request["condition_position"],
        "root_prediction_commit_sha256": root_sha,
        "root_ack_sha256": root_ack_sha,
        "decision_input_sha256": decision_sha,
        "selector_proof": selector_proof,
        "selector_proof_sha256": selector_proof["selector_proof_sha256"],
        "selected_candidate_index": selected,
        "executed_step_count": len(steps),
        "steps": steps,
        "recovery_pre_step_records": recovery_records,
        "target_envelope": envelope,
        "target_envelope_sha256": envelope["envelope_sha256"],
        "plaintext_target_present_in_result": False,
        "simulator_launched_by_this_module": False,
    }
    return _signed(base, "result_sha256")


def validate_condition_result(
    value: Mapping[str, Any], *, request: Mapping[str, Any],
    root_precommit: Mapping[str, Any], root_ack: Mapping[str, Any],
    decision_input: Mapping[str, Any],
) -> str:
    request_sha = validate_condition_request(request)
    root_sha = audit.validate_root_precommit(root_precommit)
    root_ack_sha = validate_root_broker_ack(root_ack, root_precommit=root_precommit)
    decision_sha = validate_root_decision_input(
        decision_input, root_precommit=root_precommit
    )
    expected_selected, expected_selector_proof = select_condition_root(
        str(request["condition_id"]), decision_input,
        root_precommit=root_precommit,
    )
    fields = {
        "format", "status", "request_sha256", "protocol_core_v4_sha256",
        "pair_id", "pair_ordinal", "condition_id", "condition_position",
        "root_prediction_commit_sha256", "root_ack_sha256",
        "decision_input_sha256", "selector_proof", "selector_proof_sha256",
        "selected_candidate_index", "executed_step_count", "steps",
        "recovery_pre_step_records", "target_envelope", "target_envelope_sha256",
        "plaintext_target_present_in_result", "simulator_launched_by_this_module",
    }
    logical = _verify(
        value, field="result_sha256", fields=fields, role="condition result"
    )
    if (
        value.get("format") != RESULT_FORMAT
        or value.get("status") != RESULT_STATUS
        or value.get("request_sha256") != request_sha
        or value.get("root_prediction_commit_sha256") != root_sha
        or value.get("root_ack_sha256") != root_ack_sha
        or value.get("decision_input_sha256") != decision_sha
        or value.get("selector_proof") != expected_selector_proof
        or value.get("selector_proof_sha256")
        != expected_selector_proof["selector_proof_sha256"]
        or value.get("selected_candidate_index") != expected_selected
        or any(
            value.get(field_name) != request.get(field_name)
            for field_name in (
                "protocol_core_v4_sha256", "pair_id", "pair_ordinal",
                "condition_id", "condition_position",
            )
        )
        or value.get("plaintext_target_present_in_result") is not False
        or value.get("simulator_launched_by_this_module") is not False
    ):
        raise ConditionRunnerV4Error("condition result identity changed")
    _require_int(value.get("pair_ordinal"), "result pair ordinal")
    selected = _require_int(value.get("selected_candidate_index"), "selected candidate")
    steps = value.get("steps")
    recovery = value.get("recovery_pre_step_records")
    if (
        not isinstance(steps, list)
        or not steps
        or value.get("executed_step_count") != len(steps)
        or type(value.get("executed_step_count")) is not int
        or not isinstance(recovery, list)
        or [row.get("step_index") for row in recovery] != list(range(1, len(steps)))
        or steps[0].get("selected_candidate_index") != selected
    ):
        raise ConditionRunnerV4Error("condition continuation/recovery coverage changed")
    for index, row in enumerate(steps):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "step_index", "selected_candidate_index", "candidate_sha256",
                "observer_receipt_sha256",
                "root_condition_selection", "terminated", "truncated",
            }
            or type(row.get("step_index")) is not int
            or row["step_index"] != index
            or type(row.get("selected_candidate_index")) is not int
            or not audit.is_sha256(row.get("candidate_sha256"))
            or not audit.is_sha256(row.get("observer_receipt_sha256"))
            or type(row.get("root_condition_selection")) is not bool
            or row["root_condition_selection"] is not (index == 0)
            or type(row.get("terminated")) is not bool
            or type(row.get("truncated")) is not bool
        ):
            raise ConditionRunnerV4Error("condition step proof changed")
    for expected_step, row in enumerate(recovery, start=1):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "step_index", "status", "commit_sha256", "ack_sha256",
                "ledger_event_sha256",
            }
            or type(row.get("step_index")) is not int
            or row["step_index"] != expected_step
        ):
            raise ConditionRunnerV4Error("recovery pre-step coverage changed")
        if row.get("status") == "committed_and_broker_acknowledged_pre_step":
            if any(
                not audit.is_sha256(row.get(field_name))
                for field_name in (
                    "commit_sha256", "ack_sha256", "ledger_event_sha256"
                )
            ):
                raise ConditionRunnerV4Error("recovery pre-step SHA changed")
        elif row.get("status") == "not_applicable_no_prior_progress":
            if any(
                row.get(field_name) is not None
                for field_name in (
                    "commit_sha256", "ack_sha256", "ledger_event_sha256"
                )
            ):
                raise ConditionRunnerV4Error("inapplicable recovery has an artifact")
        else:
            raise ConditionRunnerV4Error("recovery applicability status changed")
    envelope = value.get("target_envelope")
    if not isinstance(envelope, Mapping):
        raise ConditionRunnerV4Error("condition target envelope is missing")
    envelope_sha, _aad = audit.validate_target_envelope(
        envelope,
        expected_protocol_core_v4_sha256=request["protocol_core_v4_sha256"],
        expected_pair_id=request["pair_id"],
        expected_condition_id=request["condition_id"],
        expected_root_prediction_commit_sha256=root_sha,
        expected_schema6_runtime_contract_sha256=request[
            "schema6_runtime_contract_sha256"
        ],
    )
    if value.get("target_envelope_sha256") != envelope_sha:
        raise ConditionRunnerV4Error("condition target envelope SHA changed")
    return logical


__all__ = [
    "CONDITION_NAMES",
    "ConditionBackendV4",
    "ConditionRunnerV4Error",
    "DECISION_INPUT_FORMAT",
    "MAX_EPISODE_STEPS",
    "RESULT_FORMAT",
    "ROOT_ACK_FORMAT",
    "build_condition_request",
    "build_causal_observer_authority",
    "build_causal_observer_receipt",
    "build_root_broker_ack",
    "build_root_decision_input",
    "execute_condition_v4",
    "select_condition_root",
    "validate_condition_request",
    "validate_condition_result",
    "validate_root_broker_ack",
    "validate_root_decision_input",
    "validate_causal_observer_authority",
    "validate_causal_observer_receipt",
]
