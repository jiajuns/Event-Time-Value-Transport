#!/usr/bin/env python3
"""Actor-agnostic, fail-closed adapter for the frozen v7 utility.

This module intentionally contains no OpenVLA/SmolVLA imports or hidden-state
assumptions.  Actors/world models meet it at a small prediction mapping.  The
adapter gathers exactly four explicitly named deployment slots, moves the
caller's explicit actor fallback to local slot zero, delegates the numerical
formula to the frozen v7 implementation, and applies validity, uncertainty and
contract authorization without changing that formula.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from openvla_etsf_structured_event_time_utility import (
    DEPLOYMENT_CANDIDATE_COUNT,
    FORMAT as V7_UTILITY_FORMAT,
    GUARD_MARGIN,
    guarded_candidate_selection_torch,
    structured_event_time_utility_from_predictions,
)


FORMAT = "etsf_actor_agnostic_structured_event_time_plugin_v1"
STANDARD_PREDICTION_KEYS = (
    "next_reached_event_logits",
    "next_event_logits",
    "duration_selected_log_mean",
    "total_uncertainty",
)
DEFAULT_REQUIRED_CONTRACTS = (
    "state_contract_matched",
    "action_contract_matched",
    "embodiment_contract_matched",
    "clock_contract_matched",
    "predicate_contract_matched",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _validated_registry(
    registry: Mapping[str, Sequence[float]],
) -> tuple[dict[str, tuple[float, ...]], str]:
    if not isinstance(registry, Mapping) or not registry:
        raise ValueError("event_values_registry must be a non-empty mapping")
    result: dict[str, tuple[float, ...]] = {}
    for raw_task, raw_values in registry.items():
        task = str(raw_task)
        if not task or task != raw_task:
            raise ValueError("event_values_registry task keys must be non-empty strings")
        if isinstance(raw_values, (str, bytes)):
            raise TypeError("event value vectors must be numeric sequences")
        try:
            values = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError) as error:
            raise TypeError("event value vectors must be numeric sequences") from error
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("event value vectors must be non-empty and finite")
        result[task] = values
    payload = {task: list(result[task]) for task in sorted(result)}
    return result, _canonical_sha256(payload)


def _validated_required_contracts(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise TypeError("required_contracts must be a sequence of names")
    result = tuple(str(name) for name in names)
    if not result or any(not name for name in result) or len(set(result)) != len(result):
        raise ValueError("required_contracts must contain unique non-empty names")
    return result


def _prediction_tensors(
    predictions: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    missing = [name for name in STANDARD_PREDICTION_KEYS if name not in predictions]
    if missing:
        raise KeyError(f"standard prediction mapping is missing: {missing}")
    values = tuple(predictions[name] for name in STANDARD_PREDICTION_KEYS)
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("standard prediction values must be tensors")
    destination, immediate, duration, uncertainty = values
    assert isinstance(destination, torch.Tensor)
    assert isinstance(immediate, torch.Tensor)
    assert isinstance(duration, torch.Tensor)
    assert isinstance(uncertainty, torch.Tensor)
    if destination.ndim < 2 or immediate.shape != destination.shape:
        raise ValueError("event logits must share shape [...,C,E]")
    if duration.shape != destination.shape[:-1]:
        raise ValueError("duration_selected_log_mean must have shape [...,C]")
    if uncertainty.shape != duration.shape:
        raise ValueError("total_uncertainty must have shape [...,C]")
    if not all(value.is_floating_point() for value in values):
        raise TypeError("standard predictions must be floating-point")
    if not all(value.device == destination.device for value in values[1:]):
        raise ValueError("standard predictions must share one device")
    return destination, immediate, duration, uncertainty


def _slot_order(
    deployment_slots: Sequence[int], *, fallback_slot: int, candidate_count: int
) -> tuple[int, ...]:
    if isinstance(deployment_slots, (str, bytes)):
        raise TypeError("deployment_slots must be four integer indices")
    slots = tuple(deployment_slots)
    if (
        len(slots) != DEPLOYMENT_CANDIDATE_COUNT
        or any(isinstance(slot, bool) or not isinstance(slot, int) for slot in slots)
        or len(set(slots)) != DEPLOYMENT_CANDIDATE_COUNT
    ):
        raise ValueError("deployment_slots must contain four unique integer indices")
    if any(slot < 0 or slot >= candidate_count for slot in slots):
        raise ValueError("deployment slot is outside the candidate range")
    if isinstance(fallback_slot, bool) or not isinstance(fallback_slot, int):
        raise TypeError("fallback_slot must be an integer index")
    if fallback_slot not in slots:
        raise ValueError("fallback_slot must be one of the four deployment slots")
    return (fallback_slot, *(slot for slot in slots if slot != fallback_slot))


def _contract_reasons(
    contract_guard: Mapping[str, bool], required: Sequence[str]
) -> tuple[str, ...]:
    if not isinstance(contract_guard, Mapping):
        raise TypeError("contract_guard must be an explicit mapping")
    reasons: list[str] = []
    for name in required:
        if name not in contract_guard:
            reasons.append(f"contract_missing:{name}")
        elif contract_guard[name] is not True:
            reasons.append(f"contract_failed:{name}")
    for name, value in contract_guard.items():
        if not isinstance(name, str) or not name:
            raise ValueError("contract_guard keys must be non-empty strings")
        if not isinstance(value, bool):
            raise TypeError("contract_guard values must be bool")
        if name not in required and not value:
            reasons.append(f"contract_failed:{name}")
    return tuple(reasons)


class ActorAgnosticStructuredEventTimePlugin(nn.Module):
    """Zero-state adapter from standard predictions to guarded actor slots."""

    def __init__(
        self,
        *,
        event_values_registry: Mapping[str, Sequence[float]],
        maximum_total_uncertainty: float,
        required_contracts: Sequence[str] = DEFAULT_REQUIRED_CONTRACTS,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(maximum_total_uncertainty)
            or maximum_total_uncertainty < 0
        ):
            raise ValueError("maximum_total_uncertainty must be finite and non-negative")
        registry, registry_sha = _validated_registry(event_values_registry)
        # Prevent the explicit task ontology from being mutated while leaving
        # the recorded registry digest stale.
        self.event_values_registry = MappingProxyType(registry)
        self.event_values_registry_sha256 = registry_sha
        self.maximum_total_uncertainty = float(maximum_total_uncertainty)
        self.required_contracts = _validated_required_contracts(required_contracts)

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        *,
        task: str,
        deployment_slots: Sequence[int],
        fallback_slot: int,
        candidate_valid_mask: torch.Tensor,
        contract_guard: Mapping[str, bool],
    ) -> dict[str, Any]:
        if task not in self.event_values_registry:
            raise KeyError(f"unknown task in event_values_registry: {task!r}")
        destination, immediate, duration, uncertainty = _prediction_tensors(predictions)
        # This is a deployment selector, not a differentiable guidance loss.
        # Detaching here enforces the policy-preserving contract even if a
        # caller accidentally invokes it inside a training graph.
        destination = destination.detach()
        immediate = immediate.detach()
        duration = duration.detach()
        uncertainty = uncertainty.detach()
        if len(self.event_values_registry[task]) != destination.shape[-1]:
            raise ValueError("task event_values length does not match prediction vocabulary")
        if destination.shape[-2] < DEPLOYMENT_CANDIDATE_COUNT:
            raise ValueError("actor-agnostic plugin requires at least four candidates")
        if not torch.is_tensor(candidate_valid_mask) or candidate_valid_mask.dtype != torch.bool:
            raise TypeError("candidate_valid_mask must be an explicit bool tensor")
        if candidate_valid_mask.shape != duration.shape:
            raise ValueError("candidate_valid_mask must have shape [...,C]")
        if candidate_valid_mask.device != destination.device:
            raise ValueError("candidate_valid_mask must share the prediction device")

        local_to_actor = _slot_order(
            deployment_slots,
            fallback_slot=fallback_slot,
            candidate_count=destination.shape[-2],
        )
        slot_index = torch.tensor(
            local_to_actor, dtype=torch.long, device=destination.device
        )
        gathered_destination = destination.index_select(-2, slot_index)
        gathered_immediate = immediate.index_select(-2, slot_index)
        gathered_duration = duration.index_select(-1, slot_index)
        gathered_uncertainty = uncertainty.index_select(-1, slot_index)
        gathered_valid = candidate_valid_mask.index_select(-1, slot_index)
        if not bool(gathered_valid[..., 0].all()):
            raise RuntimeError("declared actor fallback candidate is invalid")

        finite_prediction = (
            torch.isfinite(gathered_destination).all(dim=-1)
            & torch.isfinite(gathered_immediate).all(dim=-1)
            & torch.isfinite(gathered_duration)
        )
        fallback_finite = finite_prediction[..., 0]
        if not bool(fallback_finite.all()):
            raise RuntimeError("declared actor fallback prediction is non-finite")
        effective_valid = gathered_valid & finite_prediction

        # The frozen utility rejects NaNs before scoring.  Invalid/non-finite
        # alternatives are replaced by the finite fallback solely for numerical
        # decomposition, then excluded from proposal selection below.
        fallback_destination = gathered_destination[..., :1, :].expand_as(
            gathered_destination
        )
        fallback_immediate = gathered_immediate[..., :1, :].expand_as(
            gathered_immediate
        )
        fallback_duration = gathered_duration[..., :1].expand_as(gathered_duration)
        clean_destination = torch.where(
            finite_prediction[..., None], gathered_destination, fallback_destination
        )
        clean_immediate = torch.where(
            finite_prediction[..., None], gathered_immediate, fallback_immediate
        )
        clean_duration = torch.where(
            finite_prediction, gathered_duration, fallback_duration
        )
        decomposition = structured_event_time_utility_from_predictions(
            {
                "next_reached_event_logits": clean_destination,
                "next_event_logits": clean_immediate,
                "duration_selected_log_mean": clean_duration,
            },
            event_values=self.event_values_registry[task],
        )
        utility = decomposition["utility"]
        assert isinstance(utility, torch.Tensor)
        masked_utility = torch.where(
            effective_valid, utility, torch.full_like(utility, torch.finfo(utility.dtype).min)
        )
        base_decision = guarded_candidate_selection_torch(masked_utility)
        proposed_local = base_decision["proposed_index"]
        selected_local = base_decision["selected_index"].clone()
        accepted = base_decision["accepted"].clone()
        margin = base_decision["score_margin"]
        proposed_uncertainty = gathered_uncertainty.gather(
            -1, proposed_local.unsqueeze(-1)
        ).squeeze(-1)
        uncertainty_finite = torch.isfinite(proposed_uncertainty)
        uncertainty_ok = uncertainty_finite & (
            proposed_uncertainty <= self.maximum_total_uncertainty
        )
        contract_reasons = _contract_reasons(contract_guard, self.required_contracts)
        if contract_reasons:
            accepted = torch.zeros_like(accepted)
        accepted = accepted & uncertainty_ok
        selected_local = torch.where(accepted, proposed_local, torch.zeros_like(proposed_local))

        actor_slots = torch.tensor(local_to_actor, device=destination.device, dtype=torch.long)
        proposed_actor = actor_slots[proposed_local]
        selected_actor = actor_slots[selected_local]
        group_shape = tuple(duration.shape[:-1])
        flat_proposed = proposed_local.reshape(-1)
        flat_base_accepted = base_decision["accepted"].reshape(-1)
        flat_uncertainty_finite = uncertainty_finite.reshape(-1)
        flat_uncertainty_ok = uncertainty_ok.reshape(-1)
        flat_valid = gathered_valid.reshape(-1, DEPLOYMENT_CANDIDATE_COUNT)
        flat_finite = finite_prediction.reshape(-1, DEPLOYMENT_CANDIDATE_COUNT)
        reason_codes: list[tuple[str, ...]] = []
        candidate_reason_codes: list[tuple[tuple[str, ...], ...]] = []
        for row in range(flat_proposed.numel()):
            reasons = list(contract_reasons)
            per_candidate: list[tuple[str, ...]] = []
            for local_slot in range(DEPLOYMENT_CANDIDATE_COUNT):
                candidate_reasons: list[str] = []
                if not bool(flat_valid[row, local_slot]):
                    candidate_reasons.append("candidate_invalid")
                if not bool(flat_finite[row, local_slot]):
                    candidate_reasons.append("candidate_prediction_nonfinite")
                per_candidate.append(tuple(candidate_reasons))
            if any(per_candidate):
                reasons.append("invalid_candidate_excluded")
            if int(flat_proposed[row]) == 0:
                reasons.append("utility_prefers_fallback")
            elif not bool(flat_base_accepted[row]):
                reasons.append("utility_margin_below_guard")
            if not bool(flat_uncertainty_finite[row]):
                reasons.append("nonfinite_uncertainty")
            elif not bool(flat_uncertainty_ok[row]):
                reasons.append("uncertainty_above_guard")
            reason_codes.append(tuple(reasons))
            candidate_reason_codes.append(tuple(per_candidate))

        return {
            "format": FORMAT,
            "wrapped_utility_format": V7_UTILITY_FORMAT,
            "task": task,
            "event_values": tuple(self.event_values_registry[task]),
            "event_values_registry_sha256": self.event_values_registry_sha256,
            "trainable_parameter_count": 0,
            "buffer_count": 0,
            "deployment_slots_requested": tuple(deployment_slots),
            "local_to_actor_slot": local_to_actor,
            "fallback_actor_slot": fallback_slot,
            "fallback_local_slot": 0,
            "guard_margin": GUARD_MARGIN,
            "maximum_total_uncertainty": self.maximum_total_uncertainty,
            "required_contracts": self.required_contracts,
            "contract_guard": dict(contract_guard),
            "group_shape": group_shape,
            "candidate_valid_mask": gathered_valid,
            "prediction_finite_mask": finite_prediction,
            "effective_candidate_valid_mask": effective_valid,
            "input_sanitized_mask": ~finite_prediction,
            "utility": utility,
            "selection_utility": masked_utility,
            "proposed_local_index": proposed_local,
            "selected_local_index": selected_local,
            "proposed_actor_slot": proposed_actor,
            "selected_actor_slot": selected_actor,
            "score_margin": margin,
            "proposed_uncertainty": proposed_uncertainty,
            "uncertainty_guard_passed": uncertainty_ok,
            "accepted": accepted,
            "reason_codes": tuple(reason_codes),
            "candidate_reason_codes": tuple(candidate_reason_codes),
            "decomposition": decomposition,
        }


__all__ = [
    "ActorAgnosticStructuredEventTimePlugin",
    "DEFAULT_REQUIRED_CONTRACTS",
    "FORMAT",
    "STANDARD_PREDICTION_KEYS",
]
