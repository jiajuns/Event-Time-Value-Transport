#!/usr/bin/env python3
"""Pure prediction router for the development-only composite activation.

The router cannot train, rank candidates, alter OpenVLA, or expose the failed
success head.  Inputs must already be detached; outputs are NumPy copies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import openvla_etsf_structured_event_time_utility as v7_utility
from openvla_etsf_duration_hierarchy import canonical_sha256
from openvla_etsf_duration_hierarchy_adapter import (
    predict_duration_candidates,
    sha256_path,
    validate_duration_activation,
    validate_empirical_registry_contract,
)


FORMAT = "etsf_composite_structured_prediction_activation_v1"
ACTIVE_CAPABILITIES = (
    "next_event",
    "destination_event",
    "aleatoric_uncertainty",
    "regress",
    "duration_v2",
)
INACTIVE_CAPABILITIES = (
    "success",
    "recovery",
    "object",
    "total_uncertainty",
)
EVIDENCE_SHA_FIELDS = {
    "factual_event_result": {
        "file_sha256",
        "result_sha256",
        "materialization_sha256",
    },
    "r4_adamw_regress": {
        "result_file_sha256",
        "result_sha256",
        "contracts_file_sha256",
        "contracts_sha256",
        "arrays_file_sha256",
        "bridge_bundle_sha256",
        "materialization_sha256",
        "factual_checkpoint_sha256",
        "five_checkpoint_bundle_sha256",
    },
    "r5_success_inadequacy": {
        "file_sha256",
        "result_sha256",
        "materialization_sha256",
    },
    "r5_duration_activation": {
        "file_sha256",
        "activation_sha256",
        "materialization_sha256",
        "final_hierarchy_contract_sha256",
        "empirical_registry_contract_sha256",
    },
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _reject_fresh_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def validate_composite_activation(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    recorded = unsigned.pop("activation_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("composite activation signature mismatch")
    active = value.get("active")
    inactive = value.get("inactive_or_fallback")
    selector = value.get("action_selector")
    scope = value.get("empirical_evidence_scope")
    duration = value.get("duration_v2_activation")
    if (
        value.get("format") != FORMAT
        or value.get("status")
        != "active_structured_prediction_development_only"
        or value.get("evidence_scope") != "adaptive_development_only"
        or not isinstance(active, Mapping)
        or set(active) != set(ACTIVE_CAPABILITIES)
        or any(item.get("status") != "active" for item in active.values())
        or active.get("next_event", {}).get("ranking_input") is not True
        or active.get("destination_event", {}).get("ranking_input") is not True
        or any(
            active.get(name, {}).get("ranking_input") is not False
            for name in ("aleatoric_uncertainty", "regress", "duration_v2")
        )
        or not isinstance(inactive, Mapping)
        or set(inactive) != set(INACTIVE_CAPABILITIES)
        or inactive.get("success", {}).get("status") != "inactive"
        or inactive.get("success", {}).get("reason")
        != "strict_probability_adequacy_false"
        or inactive.get("recovery", {}).get("status") != "inactive"
        or inactive.get("object", {}).get("status") != "fallback_only"
        or inactive.get("total_uncertainty", {}).get("status") != "unavailable"
        or any(item.get("ranking_input") is not False for item in inactive.values())
        or not isinstance(selector, Mapping)
        or selector.get("authority") != "v7_fixed_parameter_free_selector"
        or selector.get("v8_replacement_authorized") is not False
        or selector.get("v8_success_input_allowed") is not False
        or selector.get("v8_regress_input_allowed") is not False
        or selector.get("duration_v2_input_allowed") is not False
        or selector.get("implementation")
        != "openvla_etsf_structured_event_time_utility.py"
        or selector.get("format") != v7_utility.FORMAT
        or selector.get("formula") != v7_utility.UTILITY_FORMULA
        or selector.get("guard_margin") != v7_utility.GUARD_MARGIN
        or selector.get("deployment_candidate_count")
        != v7_utility.DEPLOYMENT_CANDIDATE_COUNT
        or not _is_sha256(selector.get("implementation_sha256"))
        or not isinstance(scope, Mapping)
        or scope.get("one_cell_only") is not True
        or scope.get("cross_body_validated") is not False
        or scope.get("cross_policy_validated") is not False
        or value.get("interface_actor_policy_agnostic") is not True
        or value.get("transfer_claim_authorized") is not False
        or value.get("fresh50_inputs_accepted") is not False
        or value.get("fresh50_labels_read") is not False
        or value.get("fresh50_confirmation_authorized") is not False
        or value.get("selector_replacement_authorized") is not False
        or value.get("openvla_gradient_path_allowed") is not False
        or not isinstance(duration, Mapping)
        or value.get("duration_v2_activation_sha256")
        != duration.get("activation_sha256")
    ):
        raise RuntimeError("composite activation capability boundary changed")
    validate_duration_activation(duration)
    if scope != duration.get("empirical_evidence_scope"):
        raise RuntimeError("composite and duration empirical scopes differ")
    if value.get("empirical_registry_contract_sha256") != duration.get(
        "empirical_registry_contract_sha256"
    ):
        raise RuntimeError("composite body registry binding changed")
    evidence = value.get("evidence")
    required_evidence = {
        "factual_event_result",
        "r4_adamw_regress",
        "r5_success_inadequacy",
        "r5_duration_activation",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_evidence:
        raise RuntimeError("composite evidence set changed")
    for name, item in evidence.items():
        expected = EVIDENCE_SHA_FIELDS[name]
        if (
            not isinstance(item, Mapping)
            or set(item.get("required_sha_fields", ())) != expected
            or any(not _is_sha256(item.get(key)) for key in expected)
        ):
            raise RuntimeError(f"composite evidence hashes are invalid: {name}")
    materialization_shas = {
        str(item["materialization_sha256"]) for item in evidence.values()
    }
    duration_evidence = duration.get("evidence", {})
    if (
        len(materialization_shas) != 1
        or next(iter(materialization_shas))
        != duration_evidence.get("materialization_sha256")
        or evidence["r4_adamw_regress"].get("factual_checkpoint_sha256")
        != duration_evidence.get("factual_checkpoint_sha256")
        or evidence["r5_duration_activation"].get("activation_sha256")
        != duration.get("activation_sha256")
        or evidence["r5_duration_activation"].get(
            "final_hierarchy_contract_sha256"
        )
        != duration.get("final_hierarchy_contract_sha256")
        or evidence["r5_duration_activation"].get(
            "empirical_registry_contract_sha256"
        )
        != duration.get("empirical_registry_contract_sha256")
    ):
        raise RuntimeError("composite evidence cross-binding changed")
    implementation = value.get("implementation_files")
    required_files = {
        "freeze_openvla_etsf_composite_structured_prediction_activation.py",
        "openvla_etsf_composite_structured_prediction_router.py",
        "openvla_etsf_duration_hierarchy.py",
        "openvla_etsf_duration_hierarchy_adapter.py",
        "evaluate_openvla_etsf_v8_factual_events.py",
        "evaluate_openvla_etsf_v8_oof_bridge.py",
        "evaluate_openvla_etsf_v8_structured_heads_arrays.py",
        "calibrate_openvla_etsf_v8_success_inner_cv.py",
        "openvla_etsf_structured_event_time_utility.py",
    }
    if not isinstance(implementation, Mapping) or set(implementation) != required_files:
        raise RuntimeError("composite implementation hash set changed")
    if any(not _is_sha256(item) for item in implementation.values()):
        raise RuntimeError("composite implementation SHA is invalid")
    root = Path(__file__).resolve().parent
    for filename in (
        "openvla_etsf_composite_structured_prediction_router.py",
        "openvla_etsf_duration_hierarchy.py",
        "openvla_etsf_duration_hierarchy_adapter.py",
        "openvla_etsf_structured_event_time_utility.py",
    ):
        if sha256_path(root / filename) != implementation[filename]:
            raise RuntimeError(f"composite runtime code hash mismatch: {filename}")
    if selector["implementation_sha256"] != implementation[
        "openvla_etsf_structured_event_time_utility.py"
    ]:
        raise RuntimeError("composite v7 selector implementation binding changed")
    return dict(value)


def load_composite_activation(path: Path) -> dict[str, Any]:
    path = _reject_fresh_path(path, role="composite activation")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError("composite activation must contain a JSON object")
    return validate_composite_activation(value)


def _detached_numpy(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f"{name} must be detached from OpenVLA gradients")
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite numeric data")
    return result.copy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def route_structured_predictions(
    activation: Mapping[str, Any],
    *,
    body_registry_contract: Mapping[str, Any],
    current_event_id: Any,
    body_id: Any,
    next_event_logits: Any,
    destination_event_logits: Any,
    aleatoric_uncertainty: Any,
    regress_probability: Any,
    frozen_duration_log_mean: Any,
) -> dict[str, Any]:
    """Route only active prediction fields; never compute a ranking score."""

    activation = validate_composite_activation(activation)
    registry = validate_empirical_registry_contract(body_registry_contract)
    if registry["registry_sha256"] != activation[
        "empirical_registry_contract_sha256"
    ]:
        raise RuntimeError("caller body registry differs from composite evidence")
    immediate = _detached_numpy(next_event_logits, name="next_event_logits")
    destination = _detached_numpy(
        destination_event_logits, name="destination_event_logits"
    )
    if immediate.ndim != 2 or destination.shape != immediate.shape:
        raise ValueError("event logits must be aligned [candidate,event] matrices")
    count = len(immediate)
    uncertainty = _detached_numpy(
        aleatoric_uncertainty, name="aleatoric_uncertainty"
    )
    regress = _detached_numpy(regress_probability, name="regress_probability")
    if (
        uncertainty.shape != (count,)
        or np.any(uncertainty < 0.0)
        or regress.shape != (count,)
        or np.any((regress < 0.0) | (regress > 1.0))
    ):
        raise ValueError("aleatoric/regress predictions are invalid or misaligned")
    frozen_duration = _detached_numpy(
        frozen_duration_log_mean, name="frozen_duration_log_mean"
    )
    if frozen_duration.shape != (count,):
        raise ValueError("frozen duration must align with candidates")
    duration = predict_duration_candidates(
        activation["duration_v2_activation"],
        body_registry_contract=registry,
        current_event_id=current_event_id,
        body_id=body_id,
        frozen_duration_log_mean=frozen_duration,
    )
    return {
        "active_predictions": {
            "next_event": _softmax(immediate),
            "destination_event": _softmax(destination),
            "aleatoric_uncertainty": uncertainty,
            "regress": regress,
            "duration_v2": duration,
        },
        "inactive_or_fallback": dict(activation["inactive_or_fallback"]),
        # Preserve v7 exactly: it sees factual logits and the original factual
        # duration, never success, regress, recovery, object, or duration-v2.
        "v7_selector_inputs": {
            "next_event_logits": immediate.copy(),
            "next_reached_event_logits": destination.copy(),
            "duration_selected_log_mean": frozen_duration.copy(),
        },
        "v7_selector_implementation_sha256": activation["action_selector"][
            "implementation_sha256"
        ],
        "ranking_score_produced": False,
        "v8_selector_replacement_authorized": False,
        "success_head_exposed": False,
        "openvla_gradient_path": False,
        "transfer_claim_authorized": False,
    }


__all__ = [
    "ACTIVE_CAPABILITIES",
    "FORMAT",
    "INACTIVE_CAPABILITIES",
    "load_composite_activation",
    "route_structured_predictions",
    "validate_composite_activation",
]
