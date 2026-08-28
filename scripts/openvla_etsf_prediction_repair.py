#!/usr/bin/env python3
"""Prospective repair contracts for ETSF structured prediction heads.

The helpers in this module are deliberately independent from action ranking and
from the fresh-confirmation gate.  They are intended to be preregistered for a
new development-only OOF run:

* weighted BCE logits are converted back to probability logits with the exact
  outer-training-fold prior shift; current outer OOF predictions are never
  reused to fit bias/temperature because their training sets overlap;
* duration is represented as an observed-only log-duration residual around a
  training-fold event x body median;
* object displacement uses a training-fold robust centre/scale, an explicit
  finite/outlier quality mask, and a Student-t likelihood.

No function opens datasets or knows a fresh-data path.  Callers must pass
already materialised development arrays and fold ownership explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch


FOLD_COUNT = 5
WEIGHTED_BINARY_CALIBRATION_PROTOCOL = (
    "recorded_head_training_weight_prior_shift_only_no_crossfold_fit_v2"
)
DURATION_RESIDUAL_PROTOCOL = (
    "observed_only_laplace_log1p_residual_about_training_event_body_median_v1"
)
OBJECT_REPAIR_PROTOCOL = (
    "training_fold_robust_median_scale_q995_quality_mask_student_t_df3_v1"
)
RECOVERY_ADAPTER_PROTOCOL = (
    "frozen_transition_feature_binary_recovery_adapter_unweighted_bce_v1"
)
EPS = 1e-12


def _as_binary(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if np.any((result != 0.0) & (result != 1.0)):
        raise ValueError(f"{name} must contain binary labels")
    return result


def _as_folds(values: np.ndarray, count: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.shape != (count,) or set(np.unique(result)) != set(range(FOLD_COUNT)):
        raise ValueError("fold ids must cover exactly five folds")
    return result


def weighted_bce_positive_weight(
    labels: np.ndarray,
    *,
    cap: float | None,
    floor_at_one: bool,
) -> float:
    """Reproduce the trainer's weighted-BCE positive-class coefficient."""

    labels = _as_binary(labels, name="weight labels")
    positive = float(labels.sum())
    weight = (len(labels) - positive) / max(positive, 1.0)
    if cap is not None:
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("positive-weight cap must be finite and positive")
        weight = min(weight, float(cap))
    if floor_at_one:
        weight = max(weight, 1.0)
    return max(float(weight), EPS)


def apply_recorded_weighted_binary_prior_shift(
    member_logits: np.ndarray,
    fold_id: np.ndarray,
    *,
    recorded_positive_weights: Mapping[int | str, float],
) -> dict[str, Any]:
    """Apply only positive weights recorded by the head's actual trainer.

    Branch-fold prevalence is deliberately not accepted as an argument: it
    cannot reconstruct the coefficient of a frozen factual head.  Callers must
    first validate each weight against a signed head-training contract.
    """

    logits = np.asarray(member_logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("member logits must have shape [members,samples]")
    folds = _as_folds(fold_id, logits.shape[1])
    if logits.shape[0] < 1 or not np.isfinite(logits).all():
        raise ValueError("member logits must be finite and non-empty")
    normalized_weights = {
        int(key): float(value) for key, value in recorded_positive_weights.items()
    }
    if set(normalized_weights) != set(range(FOLD_COUNT)) or any(
        not math.isfinite(value) or value <= 0.0
        for value in normalized_weights.values()
    ):
        raise ValueError("recorded positive weights must cover five owner folds")

    prior_corrected = np.empty_like(logits)
    positive_weights: dict[str, float] = {}
    prior_log_shifts: dict[str, float] = {}
    for target_fold in range(FOLD_COUNT):
        target = folds == target_fold
        weight = normalized_weights[target_fold]
        shift = math.log(weight)
        positive_weights[str(target_fold)] = weight
        prior_log_shifts[str(target_fold)] = shift
        prior_corrected[:, target] = logits[:, target] - shift

    member_probability = 1.0 / (
        1.0 + np.exp(-np.clip(prior_corrected, -80.0, 80.0))
    )
    probability = member_probability.mean(0)
    folds_result: dict[str, Any] = {}
    for target_fold in range(FOLD_COUNT):
        target = folds == target_fold
        folds_result[str(target_fold)] = {
            "status": "analytic_training_weight_prior_shift",
            "heldout_support": int(target.sum()),
            "heldout_labels_used_for_fit": False,
            "other_outer_oof_predictions_used_for_fit": False,
            "positive_weight": positive_weights[str(target_fold)],
            "prior_log_shift": prior_log_shifts[str(target_fold)],
        }
    return {
        "protocol": WEIGHTED_BINARY_CALIBRATION_PROTOCOL,
        "probability": probability,
        "member_probability": member_probability,
        "prior_corrected_member_logits": prior_corrected,
        "folds": folds_result,
        "labels_used_to_reconstruct_weight": False,
        "other_outer_oof_predictions_used_for_fit": False,
        "bias_temperature_status": "requires_separate_nested_inner_oof_not_available",
    }


def fit_duration_residual_contract(
    duration: np.ndarray,
    observed: np.ndarray,
    event_id: np.ndarray,
    body_id: np.ndarray,
) -> dict[str, Any]:
    """Fit a serialisable observed-only event/body median lookup."""

    duration = np.asarray(duration, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    event_id = np.asarray(event_id, dtype=np.int64)
    body_id = np.asarray(body_id, dtype=np.int64)
    if not (
        duration.ndim == 1
        and duration.shape == observed.shape == event_id.shape == body_id.shape
        and len(duration)
        and np.isfinite(duration).all()
        and np.all(duration >= 0.0)
    ):
        raise ValueError("duration contract arrays are invalid or misaligned")
    if not observed.any():
        raise ValueError("duration residual contract needs observed training labels")
    target = np.log1p(duration)

    def row(mask: np.ndarray) -> dict[str, Any]:
        return {
            "median_log1p_duration": float(np.median(target[mask])),
            "support": int(mask.sum()),
        }

    exact: dict[str, Any] = {}
    event: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for event_value, body_value in sorted(
        set(zip(event_id[observed].tolist(), body_id[observed].tolist()))
    ):
        selected = observed & (event_id == event_value) & (body_id == body_value)
        exact[f"{event_value}:{body_value}"] = row(selected)
    for value in sorted(set(event_id[observed].tolist())):
        event[str(value)] = row(observed & (event_id == value))
    for value in sorted(set(body_id[observed].tolist())):
        body[str(value)] = row(observed & (body_id == value))
    return {
        "protocol": DURATION_RESIDUAL_PROTOCOL,
        "fit_labels": "observed_training_transitions_only",
        "observed_support": int(observed.sum()),
        "exact_event_body": exact,
        "event_fallback": event,
        "body_fallback": body,
        "global": row(observed),
    }


def apply_duration_residual_contract(
    contract: Mapping[str, Any], event_id: np.ndarray, body_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-duration baseline and the lookup source for each row."""

    if contract.get("protocol") != DURATION_RESIDUAL_PROTOCOL:
        raise ValueError("unknown duration residual contract")
    event_id = np.asarray(event_id, dtype=np.int64)
    body_id = np.asarray(body_id, dtype=np.int64)
    if event_id.ndim != 1 or body_id.shape != event_id.shape:
        raise ValueError("duration event/body ids are misaligned")
    baseline = np.empty(len(event_id), dtype=np.float64)
    source = np.empty(len(event_id), dtype=object)
    exact = contract["exact_event_body"]
    event = contract["event_fallback"]
    body = contract["body_fallback"]
    for index, (event_value, body_value) in enumerate(zip(event_id, body_id)):
        choices = (
            ("event_body", exact.get(f"{int(event_value)}:{int(body_value)}")),
            ("event", event.get(str(int(event_value)))),
            ("body", body.get(str(int(body_value)))),
            ("global", contract["global"]),
        )
        name, selected = next((name, value) for name, value in choices if value is not None)
        baseline[index] = float(selected["median_log1p_duration"])
        source[index] = name
    return baseline, source


def crossfit_duration_residual_contract(
    duration: np.ndarray,
    observed: np.ndarray,
    fold_id: np.ndarray,
    event_id: np.ndarray,
    body_id: np.ndarray,
) -> dict[str, Any]:
    """Create leakage-free OOF baselines and residual labels."""

    duration = np.asarray(duration, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    folds = _as_folds(fold_id, len(duration))
    event_id = np.asarray(event_id, dtype=np.int64)
    body_id = np.asarray(body_id, dtype=np.int64)
    if not (duration.shape == observed.shape == event_id.shape == body_id.shape):
        raise ValueError("duration crossfit arrays are misaligned")
    baseline = np.empty(len(duration), dtype=np.float64)
    source = np.empty(len(duration), dtype=object)
    contracts: dict[str, Any] = {}
    for target_fold in range(FOLD_COUNT):
        target = folds == target_fold
        training = ~target
        contract = fit_duration_residual_contract(
            duration[training], observed[training], event_id[training], body_id[training]
        )
        baseline[target], source[target] = apply_duration_residual_contract(
            contract, event_id[target], body_id[target]
        )
        contracts[str(target_fold)] = contract
    return {
        "protocol": DURATION_RESIDUAL_PROTOCOL,
        "baseline_log1p_duration": baseline,
        "residual_target": np.log1p(duration) - baseline,
        "supervision_mask": observed,
        "fallback_source": source,
        "fold_contracts": contracts,
        "target_fold_labels_used_for_fit": False,
    }


def observed_duration_residual_laplace_nll(
    predicted_residual: torch.Tensor,
    predicted_log_scale: torch.Tensor,
    duration: torch.Tensor,
    observed: torch.Tensor,
    baseline_log1p_duration: torch.Tensor,
) -> torch.Tensor:
    """Median-aligned conditional duration loss; censored rows train reach only."""

    if not (
        predicted_residual.shape
        == predicted_log_scale.shape
        == duration.shape
        == observed.shape
        == baseline_log1p_duration.shape
    ):
        raise ValueError("duration repair tensors are misaligned")
    mask = observed.bool()
    if not mask.any():
        return predicted_residual.sum() * 0.0
    scale = torch.exp(predicted_log_scale[mask].clamp(-5.0, 3.0)).clamp_min(1e-4)
    target_residual = torch.log1p(duration[mask].clamp_min(0.0)) - baseline_log1p_duration[mask]
    return (
        torch.abs(target_residual - predicted_residual[mask]) / scale
        + torch.log(2.0 * scale)
    ).mean()


def fit_object_repair_contract(
    object_delta: np.ndarray,
    *,
    outlier_quantile: float = 0.995,
) -> dict[str, Any]:
    """Fit fold-local robust object normalisation and a finite quality fence."""

    values = np.asarray(object_delta, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] < 1:
        raise ValueError("object deltas must have shape [transitions,coordinates]")
    finite = np.isfinite(values).all(1)
    if not finite.any():
        raise ValueError("object training deltas contain no finite rows")
    if not 0.9 <= outlier_quantile < 1.0:
        raise ValueError("object outlier quantile must lie in [0.9,1.0)")
    finite_values = values[finite]
    centre = np.median(finite_values, axis=0)
    absolute_residual = np.abs(finite_values - centre)
    mad_scale = 1.4826 * np.median(absolute_residual, axis=0)
    fallback_scale = np.quantile(absolute_residual, 0.75, axis=0)
    scale = np.maximum(np.maximum(mad_scale, fallback_scale), 1e-4)
    max_abs = np.max(np.abs(finite_values), axis=1)
    threshold = max(float(np.quantile(max_abs, outlier_quantile)), 1e-4)
    training_valid = finite & (np.max(np.abs(np.where(finite[:, None], values, 0.0)), axis=1) <= threshold)
    return {
        "protocol": OBJECT_REPAIR_PROTOCOL,
        "outlier_quantile": float(outlier_quantile),
        "coordinate_median": centre.tolist(),
        "coordinate_robust_scale": scale.tolist(),
        "max_abs_delta_quality_threshold": threshold,
        "training_support": int(len(values)),
        "training_valid_support": int(training_valid.sum()),
        "training_nonfinite_support": int((~finite).sum()),
    }


def object_quality_mask(
    object_delta: np.ndarray, contract: Mapping[str, Any]
) -> np.ndarray:
    if contract.get("protocol") != OBJECT_REPAIR_PROTOCOL:
        raise ValueError("unknown object repair contract")
    values = np.asarray(object_delta, dtype=np.float64)
    centre = np.asarray(contract["coordinate_median"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != centre.shape:
        raise ValueError("object delta shape differs from repair contract")
    threshold = float(contract["max_abs_delta_quality_threshold"])
    return np.isfinite(values).all(1) & (np.max(np.abs(values), axis=1) <= threshold)


def robust_object_student_t_nll(
    predicted_residual: torch.Tensor,
    predicted_log_scale: torch.Tensor,
    physical_object_delta: torch.Tensor,
    quality_mask: torch.Tensor,
    coordinate_median: torch.Tensor,
    coordinate_robust_scale: torch.Tensor,
    *,
    degrees_of_freedom: float = 3.0,
) -> torch.Tensor:
    """Bounded-influence object residual likelihood for the next OOF trainer."""

    if predicted_residual.shape != predicted_log_scale.shape or predicted_residual.shape != physical_object_delta.shape:
        raise ValueError("object repair predictions and labels are misaligned")
    if quality_mask.shape != predicted_residual.shape[:-1]:
        raise ValueError("object quality mask is misaligned")
    if coordinate_median.shape != predicted_residual.shape[-1:] or coordinate_robust_scale.shape != coordinate_median.shape:
        raise ValueError("object repair normalisation is misaligned")
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise ValueError("Student-t degrees of freedom must be positive")
    mask = quality_mask.bool()
    if not mask.any():
        return predicted_residual.sum() * 0.0
    robust_scale = coordinate_robust_scale.clamp_min(1e-4)
    target = (physical_object_delta[mask] - coordinate_median) / robust_scale
    scale = torch.exp(predicted_log_scale[mask].clamp(-5.0, 3.0)).clamp_min(1e-4)
    standardized = (target - predicted_residual[mask]) / scale
    coefficient = 0.5 * (degrees_of_freedom + 1.0)
    # Constants independent of model parameters are intentionally omitted.
    return (
        coefficient * torch.log1p(standardized.square() / degrees_of_freedom)
        + torch.log(scale)
    ).mean()


class FrozenFeatureRecoveryAdapter(torch.nn.Module):
    """Small recovery head whose API enforces a stop-gradient shared core."""

    def __init__(self, transition_dim: int) -> None:
        super().__init__()
        if transition_dim <= 0:
            raise ValueError("transition feature dimension must be positive")
        self.transition_dim = int(transition_dim)
        self.head = torch.nn.Linear(self.transition_dim, 1)

    def forward(self, transition_features: torch.Tensor) -> torch.Tensor:
        if transition_features.ndim != 2 or transition_features.shape[1] != self.transition_dim:
            raise ValueError("recovery transition features are misaligned")
        # This detach is part of the public contract, not a caller convention:
        # recovery supervision cannot update the shared world-model trunk.
        return self.head(transition_features.detach()).squeeze(-1)


def recovery_adapter_training_contract(
    recovery_label: np.ndarray,
    supervised_mask: np.ndarray,
    *,
    minimum_class_support: int = 10,
) -> dict[str, Any]:
    labels = np.asarray(recovery_label, dtype=np.float64)
    mask = np.asarray(supervised_mask, dtype=bool)
    if labels.ndim != 1 or mask.shape != labels.shape or not np.isfinite(labels).all():
        raise ValueError("recovery labels and supervision mask are misaligned")
    if np.any((labels != 0.0) & (labels != 1.0)):
        raise ValueError("recovery labels must be binary")
    if minimum_class_support <= 0:
        raise ValueError("minimum recovery class support must be positive")
    selected = labels[mask]
    positive = int(selected.sum())
    negative = int(len(selected) - positive)
    evaluable = positive >= minimum_class_support and negative >= minimum_class_support
    return {
        "protocol": RECOVERY_ADAPTER_PROTOCOL,
        "status": "trainable" if evaluable else "fail_closed_insufficient_class_support",
        "shared_core_trainable": False,
        "feature_source": "world_model_output_transition_detached",
        "loss": "unweighted_binary_cross_entropy_probability_head",
        "support": int(len(selected)),
        "positive_support": positive,
        "negative_support": negative,
        "minimum_class_support": int(minimum_class_support),
    }


def recovery_adapter_loss(
    adapter: FrozenFeatureRecoveryAdapter,
    transition_features: torch.Tensor,
    recovery_label: torch.Tensor,
    supervised_mask: torch.Tensor,
) -> torch.Tensor:
    if recovery_label.shape != supervised_mask.shape or recovery_label.shape != transition_features.shape[:-1]:
        raise ValueError("recovery adapter tensors are misaligned")
    logits = adapter(transition_features)
    mask = supervised_mask.bool()
    if not mask.any():
        return logits.sum() * 0.0
    # Deliberately unweighted: unlike the failed structured probabilities, the
    # sigmoid output remains a probability without a hidden prior correction.
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], recovery_label[mask].to(logits)
    )


__all__ = [
    "DURATION_RESIDUAL_PROTOCOL",
    "OBJECT_REPAIR_PROTOCOL",
    "RECOVERY_ADAPTER_PROTOCOL",
    "WEIGHTED_BINARY_CALIBRATION_PROTOCOL",
    "apply_duration_residual_contract",
    "crossfit_duration_residual_contract",
    "apply_recorded_weighted_binary_prior_shift",
    "fit_duration_residual_contract",
    "fit_object_repair_contract",
    "FrozenFeatureRecoveryAdapter",
    "object_quality_mask",
    "observed_duration_residual_laplace_nll",
    "robust_object_student_t_nll",
    "recovery_adapter_loss",
    "recovery_adapter_training_contract",
    "weighted_bce_positive_weight",
]
