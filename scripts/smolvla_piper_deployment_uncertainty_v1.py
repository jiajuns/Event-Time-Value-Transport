#!/usr/bin/env python3
"""Pure deployment-time uncertainty algebra shared by calibration and inference.

Inputs to duration/object functions must already be in the deployment space:
duration log-scales include the formal calibration multiplier exactly once and
object means/log-scales are physical xyz with the object multiplier included
exactly once.  The initial pre-action e0 gate deliberately excludes recovery;
recovery is still computed and retained as an ablation component.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


FORMAT = "etsf_smolvla_piper_deployment_root_structured_uncertainty_v1"
ROOT_RECOVERY_UNCERTAINTY_POLICY = (
    "excluded_at_initial_e0_without_observed_operational_regress"
)
ROOT_INCLUDED_HEADS = (
    "post_event", "next_event", "success", "duration", "object_effect"
)
ROOT_HEAD_COUNT = 5


class DeploymentUncertaintyError(RuntimeError):
    """Deployment uncertainty input or contract is invalid."""


def _positive(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise DeploymentUncertaintyError(f"{role} must be finite and positive")
    return float(value)


def _finite(array: Any, role: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise DeploymentUncertaintyError(f"{role} must be finite and nonempty")
    return value


def softmax(value: Any, temperature: Any) -> np.ndarray:
    logits = _finite(value, "event logits")
    scale = _positive(temperature, "event temperature")
    shifted = logits / scale
    shifted -= shifted.max(axis=-1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -80.0, 0.0))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def event_total_uncertainty(logits: Any, temperature: Any) -> np.ndarray:
    value = _finite(logits, "event logits")
    if value.ndim != 3 or value.shape[0] < 2 or value.shape[-1] < 2:
        raise DeploymentUncertaintyError("event logits require member,row,class axes")
    probability = softmax(value, temperature).mean(axis=0)
    entropy = -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(
        axis=-1
    )
    return entropy / math.log(value.shape[-1])


def binary_total_uncertainty(logits: Any, temperature: Any) -> np.ndarray:
    value = _finite(logits, "binary logits")
    if value.ndim != 2 or value.shape[0] < 2:
        raise DeploymentUncertaintyError("binary logits require member,row axes")
    scale = _positive(temperature, "binary temperature")
    probability = 1.0 / (
        1.0 + np.exp(-np.clip(value / scale, -40.0, 40.0))
    )
    # E[p(1-p)] + Var[p] == mean(p) * (1 - mean(p)).  Keeping both
    # terms names the aleatoric/epistemic decomposition used in the contract.
    return (
        (probability * (1.0 - probability)).mean(axis=0)
        + probability.var(axis=0)
    ) / 0.25


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    decoded = np.fromiter(
        (
            0.5 * (1.0 + math.erf(float(item) / math.sqrt(2.0)))
            for item in flat
        ),
        dtype=np.float64,
        count=len(flat),
    )
    return decoded.reshape(np.asarray(value).shape)


def shifted_lognormal_duration_uncertainty(
    log_means: Any, deployment_log_scales: Any,
) -> np.ndarray:
    means = _finite(log_means, "duration log means")
    log_scales = _finite(deployment_log_scales, "duration deployment log scales")
    if means.ndim != 2 or means.shape != log_scales.shape or means.shape[0] < 2:
        raise DeploymentUncertaintyError("duration arrays require equal member,row axes")
    scales = np.exp(np.clip(log_scales, -8.0, 5.0))
    lower = np.min(means - 10.0 * scales, axis=0)
    upper = np.max(means + 10.0 * scales, axis=0)
    for _ in range(70):
        middle = (lower + upper) / 2.0
        cdf = _normal_cdf((middle[None, :] - means) / scales).mean(axis=0)
        lower = np.where(cdf < 0.5, middle, lower)
        upper = np.where(cdf >= 0.5, middle, upper)
    median = np.maximum(
        np.exp(np.clip((lower + upper) / 2.0, -30.0, 30.0)) - 1.0, 0.0
    )
    member_mean = np.exp(
        np.clip(means + 0.5 * scales**2, -30.0, 30.0)
    ) - 1.0
    member_variance = (
        np.exp(np.clip(scales**2, 0.0, 30.0)) - 1.0
    ) * np.exp(np.clip(2.0 * means + scales**2, -30.0, 30.0))
    total = member_variance.mean(axis=0) + member_mean.var(axis=0)
    relative = np.sqrt(np.maximum(total, 0.0)) / np.maximum(median, 1e-8)
    return relative / (1.0 + relative)


def physical_object_uncertainty(
    physical_means: Any,
    physical_deployment_log_scales: Any,
    robust_scale_m: Any,
) -> np.ndarray:
    means = _finite(physical_means, "physical object means")
    log_scales = _finite(
        physical_deployment_log_scales, "physical object deployment log scales"
    )
    robust = _positive(robust_scale_m, "physical object robust scale")
    if (
        means.ndim != 3
        or means.shape != log_scales.shape
        or means.shape[0] < 2
        or means.shape[-1] < 1
    ):
        raise DeploymentUncertaintyError(
            "object arrays require equal member,row,physical-component axes"
        )
    aleatoric = np.exp(np.clip(2.0 * log_scales, -20.0, 20.0)).mean(axis=0)
    epistemic = means.var(axis=0)
    total_std = np.sqrt(
        np.maximum((aleatoric + epistemic).mean(axis=1), 0.0)
    )
    return total_std / (robust + total_std)


def root_components(
    *, predictions: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Return the exact five-head root gate plus excluded recovery ablation."""

    components = {
        "post_event": event_total_uncertainty(
            predictions["post_event_logits"], parameters["post_event_temperature"]
        ),
        "next_event": event_total_uncertainty(
            predictions["next_event_logits"], parameters["next_event_temperature"]
        ),
        "success": binary_total_uncertainty(
            predictions["success_logit"], parameters["success_temperature"]
        ),
        "duration": shifted_lognormal_duration_uncertainty(
            predictions["duration_log_mean"], predictions["duration_log_scale"]
        ),
        "object_effect": physical_object_uncertainty(
            predictions["object_mean"],
            predictions["object_log_scale"],
            parameters["object_error_robust_scale_m"],
        ),
        "recovery_ablation_excluded": binary_total_uncertainty(
            predictions["recovery_logit"],
            parameters["conditional_recovery_temperature"],
        ),
    }
    shapes = {value.shape for value in components.values()}
    if len(shapes) != 1:
        raise DeploymentUncertaintyError("uncertainty component row axes changed")
    structured = np.stack(
        [components[name] for name in ROOT_INCLUDED_HEADS], axis=0
    ).mean(axis=0)
    if (
        not np.isfinite(structured).all()
        or bool(((structured < 0.0) | (structured > 1.0)).any())
    ):
        raise DeploymentUncertaintyError("root structured uncertainty is invalid")
    return {**components, "structured_five_head": structured}


def deployment_uncertainty_components(
    *, predictions: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Stable public alias used by the formal190 calibrator and online selector."""

    return root_components(predictions=predictions, parameters=parameters)


def combine_initial_e0_root_uncertainty(
    components: Mapping[str, Any],
) -> np.ndarray:
    """Recombine exactly the five heads applicable at the pre-action e0 root."""

    if set(components) != {
        *ROOT_INCLUDED_HEADS, "recovery_ablation_excluded", "structured_five_head"
    }:
        raise DeploymentUncertaintyError("root uncertainty component set changed")
    arrays = [_finite(components[name], name) for name in ROOT_INCLUDED_HEADS]
    if len({value.shape for value in arrays}) != 1:
        raise DeploymentUncertaintyError("root uncertainty component shapes changed")
    combined = np.stack(arrays, axis=0).mean(axis=0)
    recorded = _finite(components["structured_five_head"], "recorded root uncertainty")
    if recorded.shape != combined.shape or not np.allclose(
        recorded, combined, rtol=1e-12, atol=1e-12
    ):
        raise DeploymentUncertaintyError("recorded root uncertainty changed")
    return combined
