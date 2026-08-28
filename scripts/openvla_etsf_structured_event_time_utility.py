#!/usr/bin/env python3
"""Parameter-free v7 structured event/time utility for candidate reranking.

The utility is deliberately task-agnostic about event order: every call must
provide an explicit event-value vector.  Only the frozen four-candidate
deployment schedule participates in within-group standardization; any
training-only fifth candidate is ignored before means and variances are
computed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn


FORMAT = "etsf_structured_event_time_utility_v7"
DEPLOYMENT_CANDIDATE_COUNT = 4
BASELINE_INDEX = 0
GUARD_MARGIN = 0.05
Z_STD_EPS = 1e-8
Z_VARIANCE_EPS = Z_STD_EPS**2
UTILITY_FORMULA = (
    "z(destination_expected_progress)-z(immediate_next_event_expected_progress)"
    "+z(duration_selected_log_mean)"
)


def _numpy_inputs(
    next_reached_event_logits: Any,
    next_event_logits: Any,
    duration_selected_log_mean: Any,
    event_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    destination = np.asarray(next_reached_event_logits)
    immediate = np.asarray(next_event_logits)
    duration = np.asarray(duration_selected_log_mean)
    values = np.asarray(event_values)
    if destination.ndim < 2 or immediate.shape != destination.shape:
        raise ValueError(
            "destination and immediate logits must share shape [...,C,E]"
        )
    if destination.shape[-2] < DEPLOYMENT_CANDIDATE_COUNT:
        raise ValueError("v7 utility requires at least four candidates per group")
    if duration.shape != destination.shape[:-1]:
        raise ValueError("duration log-mean must have shape [...,C]")
    if values.ndim != 1 or values.shape[0] != destination.shape[-1]:
        raise ValueError("event_values must be an explicit length-E vector")
    if not all(
        np.issubdtype(array.dtype, np.number)
        for array in (destination, immediate, duration, values)
    ):
        raise TypeError("v7 utility inputs must be numeric")
    destination = destination.astype(np.float64, copy=False)
    immediate = immediate.astype(np.float64, copy=False)
    duration = duration.astype(np.float64, copy=False)
    values = values.astype(np.float64, copy=False)
    if not all(
        np.isfinite(array).all()
        for array in (destination, immediate, duration, values)
    ):
        raise ValueError("v7 utility inputs must be finite")
    return destination, immediate, duration, values


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def within_group_z_numpy(values: Any) -> np.ndarray:
    """Population-standardize the last axis; constant groups map exactly to 0."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != DEPLOYMENT_CANDIDATE_COUNT:
        raise ValueError("within-group z-score requires exactly four candidates")
    if not np.isfinite(array).all():
        raise ValueError("within-group values must be finite")
    centered = array - array.mean(axis=-1, keepdims=True)
    variance = np.mean(np.square(centered), axis=-1, keepdims=True)
    active = variance > Z_VARIANCE_EPS
    safe_scale = np.where(active, np.sqrt(variance), 1.0)
    standardized = centered / safe_scale
    return np.where(active, standardized, 0.0)


def structured_event_time_utility_numpy(
    next_reached_event_logits: Any,
    next_event_logits: Any,
    duration_selected_log_mean: Any,
    *,
    event_values: Any,
) -> dict[str, Any]:
    """Compute v7 utility and its complete NumPy decomposition."""

    destination, immediate, duration, values = _numpy_inputs(
        next_reached_event_logits,
        next_event_logits,
        duration_selected_log_mean,
        event_values,
    )
    deployment = slice(0, DEPLOYMENT_CANDIDATE_COUNT)
    destination = destination[..., deployment, :]
    immediate = immediate[..., deployment, :]
    duration = duration[..., deployment]
    destination_progress = (_numpy_softmax(destination) * values).sum(axis=-1)
    immediate_progress = (_numpy_softmax(immediate) * values).sum(axis=-1)
    destination_z = within_group_z_numpy(destination_progress)
    immediate_z = within_group_z_numpy(immediate_progress)
    duration_z = within_group_z_numpy(duration)
    utility = destination_z - immediate_z + duration_z
    return {
        "format": FORMAT,
        "formula": UTILITY_FORMULA,
        "trainable_parameter_count": 0,
        "deployment_candidate_count": DEPLOYMENT_CANDIDATE_COUNT,
        "event_values": values.copy(),
        "within_group_population_std_epsilon": Z_STD_EPS,
        "destination_expected_progress": destination_progress,
        "immediate_next_event_expected_progress": immediate_progress,
        "duration_selected_log_mean": duration,
        "destination_z": destination_z,
        "immediate_next_event_z": immediate_z,
        "duration_z": duration_z,
        "utility": utility,
    }


def _torch_inputs(
    next_reached_event_logits: torch.Tensor,
    next_event_logits: torch.Tensor,
    duration_selected_log_mean: torch.Tensor,
    event_values: torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(
        torch.is_tensor(value)
        for value in (
            next_reached_event_logits,
            next_event_logits,
            duration_selected_log_mean,
        )
    ):
        raise TypeError("Torch v7 logits and duration must be tensors")
    destination = next_reached_event_logits
    immediate = next_event_logits
    duration = duration_selected_log_mean
    if not (destination.is_floating_point() and immediate.is_floating_point()):
        raise TypeError("Torch v7 logits must be floating-point")
    if not duration.is_floating_point():
        raise TypeError("Torch v7 duration must be floating-point")
    if destination.ndim < 2 or immediate.shape != destination.shape:
        raise ValueError(
            "destination and immediate logits must share shape [...,C,E]"
        )
    if destination.shape[-2] < DEPLOYMENT_CANDIDATE_COUNT:
        raise ValueError("v7 utility requires at least four candidates per group")
    if duration.shape != destination.shape[:-1]:
        raise ValueError("duration log-mean must have shape [...,C]")
    if immediate.device != destination.device or duration.device != destination.device:
        raise ValueError("Torch v7 inputs must share one device")
    compute_dtype = (
        torch.float32
        if destination.dtype in (torch.float16, torch.bfloat16)
        else destination.dtype
    )
    destination = destination.to(dtype=compute_dtype)
    immediate = immediate.to(dtype=compute_dtype)
    duration = duration.to(dtype=compute_dtype)
    values = torch.as_tensor(
        event_values, device=destination.device, dtype=compute_dtype
    )
    if values.ndim != 1 or values.shape[0] != destination.shape[-1]:
        raise ValueError("event_values must be an explicit length-E vector")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (destination, immediate, duration, values)
    ):
        raise ValueError("v7 utility inputs must be finite")
    return destination, immediate, duration, values


def within_group_z_torch(values: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`within_group_z_numpy`."""

    if not torch.is_tensor(values) or not values.is_floating_point():
        raise TypeError("within-group values must be a floating-point tensor")
    if values.ndim < 1 or values.shape[-1] != DEPLOYMENT_CANDIDATE_COUNT:
        raise ValueError("within-group z-score requires exactly four candidates")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("within-group values must be finite")
    compute_values = (
        values.float()
        if values.dtype in (torch.float16, torch.bfloat16)
        else values
    )
    centered = compute_values - compute_values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    active = variance > Z_VARIANCE_EPS
    safe_scale = torch.where(active, variance.sqrt(), torch.ones_like(variance))
    standardized = centered / safe_scale
    return torch.where(active, standardized, torch.zeros_like(standardized))


def structured_event_time_utility_torch(
    next_reached_event_logits: torch.Tensor,
    next_event_logits: torch.Tensor,
    duration_selected_log_mean: torch.Tensor,
    *,
    event_values: torch.Tensor | Sequence[float],
) -> dict[str, Any]:
    """Compute v7 utility and its complete Torch decomposition."""

    destination, immediate, duration, values = _torch_inputs(
        next_reached_event_logits,
        next_event_logits,
        duration_selected_log_mean,
        event_values,
    )
    deployment = slice(0, DEPLOYMENT_CANDIDATE_COUNT)
    destination = destination[..., deployment, :]
    immediate = immediate[..., deployment, :]
    duration = duration[..., deployment]
    destination_progress = (torch.softmax(destination, dim=-1) * values).sum(dim=-1)
    immediate_progress = (torch.softmax(immediate, dim=-1) * values).sum(dim=-1)
    destination_z = within_group_z_torch(destination_progress)
    immediate_z = within_group_z_torch(immediate_progress)
    duration_z = within_group_z_torch(duration)
    utility = destination_z - immediate_z + duration_z
    return {
        "format": FORMAT,
        "formula": UTILITY_FORMULA,
        "trainable_parameter_count": 0,
        "deployment_candidate_count": DEPLOYMENT_CANDIDATE_COUNT,
        "event_values": values,
        "within_group_population_std_epsilon": Z_STD_EPS,
        "destination_expected_progress": destination_progress,
        "immediate_next_event_expected_progress": immediate_progress,
        "duration_selected_log_mean": duration,
        "destination_z": destination_z,
        "immediate_next_event_z": immediate_z,
        "duration_z": duration_z,
        "utility": utility,
    }


def structured_event_time_utility_from_predictions(
    predictions: Mapping[str, torch.Tensor],
    *,
    event_values: torch.Tensor | Sequence[float],
) -> dict[str, Any]:
    """Plugin-facing adapter for an aggregated world-model prediction mapping."""

    required = (
        "next_reached_event_logits",
        "next_event_logits",
        "duration_selected_log_mean",
    )
    missing = [name for name in required if name not in predictions]
    if missing:
        raise KeyError(f"v7 prediction mapping is missing: {missing}")
    return structured_event_time_utility_torch(
        predictions["next_reached_event_logits"],
        predictions["next_event_logits"],
        predictions["duration_selected_log_mean"],
        event_values=event_values,
    )


def _validate_numpy_utility(utility: Any) -> np.ndarray:
    score = np.asarray(utility, dtype=np.float64)
    if score.ndim < 1 or score.shape[-1] != DEPLOYMENT_CANDIDATE_COUNT:
        raise ValueError("v7 guard requires utility shape [...,4]")
    if not np.isfinite(score).all():
        raise ValueError("v7 guard utility must be finite")
    return score


def guarded_candidate_selection_numpy(utility: Any) -> dict[str, Any]:
    """Apply the fixed candidate-0 fallback and fixed 0.05 utility margin."""

    score = _validate_numpy_utility(utility)
    proposed = np.argmax(score, axis=-1).astype(np.int64)
    proposed_score = np.take_along_axis(score, proposed[..., None], axis=-1)[..., 0]
    baseline_score = score[..., BASELINE_INDEX]
    margin = proposed_score - baseline_score
    accepted = (proposed != BASELINE_INDEX) & (margin >= GUARD_MARGIN)
    selected = np.where(accepted, proposed, BASELINE_INDEX).astype(np.int64)
    return {
        "format": FORMAT,
        "baseline_index": BASELINE_INDEX,
        "guard_margin": GUARD_MARGIN,
        "proposed_index": proposed,
        "selected_index": selected,
        "score_margin": margin,
        "accepted": accepted,
    }


def guarded_candidate_selection_torch(utility: torch.Tensor) -> dict[str, Any]:
    """Torch equivalent of :func:`guarded_candidate_selection_numpy`."""

    if not torch.is_tensor(utility) or not utility.is_floating_point():
        raise TypeError("v7 guard utility must be a floating-point tensor")
    if utility.ndim < 1 or utility.shape[-1] != DEPLOYMENT_CANDIDATE_COUNT:
        raise ValueError("v7 guard requires utility shape [...,4]")
    if not bool(torch.isfinite(utility).all()):
        raise ValueError("v7 guard utility must be finite")
    proposed = utility.argmax(dim=-1)
    proposed_score = utility.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
    baseline_score = utility[..., BASELINE_INDEX]
    margin = proposed_score - baseline_score
    accepted = (proposed != BASELINE_INDEX) & (margin >= GUARD_MARGIN)
    selected = torch.where(accepted, proposed, torch.zeros_like(proposed))
    return {
        "format": FORMAT,
        "baseline_index": BASELINE_INDEX,
        "guard_margin": GUARD_MARGIN,
        "proposed_index": proposed,
        "selected_index": selected,
        "score_margin": margin,
        "accepted": accepted,
    }


class V7StructuredEventTimeUtility(nn.Module):
    """Zero-parameter Torch module suitable for direct plugin composition."""

    def forward(
        self,
        next_reached_event_logits: torch.Tensor,
        next_event_logits: torch.Tensor,
        duration_selected_log_mean: torch.Tensor,
        *,
        event_values: torch.Tensor | Sequence[float],
    ) -> dict[str, Any]:
        return structured_event_time_utility_torch(
            next_reached_event_logits,
            next_event_logits,
            duration_selected_log_mean,
            event_values=event_values,
        )


__all__ = [
    "BASELINE_INDEX",
    "DEPLOYMENT_CANDIDATE_COUNT",
    "FORMAT",
    "GUARD_MARGIN",
    "UTILITY_FORMULA",
    "Z_STD_EPS",
    "Z_VARIANCE_EPS",
    "V7StructuredEventTimeUtility",
    "guarded_candidate_selection_numpy",
    "guarded_candidate_selection_torch",
    "structured_event_time_utility_from_predictions",
    "structured_event_time_utility_numpy",
    "structured_event_time_utility_torch",
    "within_group_z_numpy",
    "within_group_z_torch",
]
