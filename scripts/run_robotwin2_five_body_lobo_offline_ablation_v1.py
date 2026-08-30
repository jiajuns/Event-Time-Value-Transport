#!/usr/bin/env python3
"""Run the fixed full-8000-branch, five-fold LOBO offline ablation.

This is an explicit training orchestrator, not a small-sample diagnostic.  It
accepts only a binding containing all 2,000 four-candidate decisions, launches
the same five-member/3,000-step budget for every fold and variant, and reports
both source-validation and frozen-checkpoint posthoc heldout metrics.  Heldout
payloads remain unopened until all source-only checkpoint selection finishes
and may never train or automatically select checkpoints/variants.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer


FORMAT = "etsf_robotwin2_five_body_lobo_offline_ablation_v1"
STATUS = "complete_frozen_checkpoint_posthoc_heldout_ablation"
VARIANTS = trainer.ABLATION_VARIANTS
QUERY_INDICES = tuple(range(40))
SEEDS_PER_CONDITION_QUERY = 5
DECISIONS_PER_BODY = 400
TOTAL_DECISIONS = 2000
TOTAL_BRANCHES = 8000
SPLIT_SEED = 20260901
STEPS_PER_MEMBER = 3000
EVAL_EVERY = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
ENSEMBLE_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
ECE_BINS = 10
RISK_COVERAGE_LEVELS = (0.25, 0.5, 0.75, 1.0)
STUDENT_T_DF = 3.0


class AblationError(RuntimeError):
    """The inventory, budget, fold isolation, or result contract changed."""


def validate_complete_inventory(audit: Mapping[str, Any]) -> dict[str, Any]:
    bodies: dict[str, Any] = {}
    total = 0
    for body in trainer.BODIES:
        groups = audit["manifests"][body]["groups"]
        if len(groups) != DECISIONS_PER_BODY:
            raise AblationError(f"{body} does not contain exactly 400 decisions")
        units: dict[tuple[str, int], set[int]] = defaultdict(set)
        for row in groups:
            condition = str(row.get("condition"))
            query = row.get("root_query_index")
            seed = row.get("requested_seed")
            if (
                condition not in trainer.CONDITIONS
                or isinstance(query, bool)
                or query not in QUERY_INDICES
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed in units[(condition, int(query))]
            ):
                raise AblationError(f"{body} has an invalid condition/query/seed inventory")
            units[(condition, int(query))].add(seed)
        if set(units) != {
            (condition, query)
            for condition in trainer.CONDITIONS
            for query in QUERY_INDICES
        } or any(len(seeds) != SEEDS_PER_CONDITION_QUERY for seeds in units.values()):
            raise AblationError(f"{body} is not complete 2x40x5")
        bodies[body] = {
            "decisions": len(groups),
            "branches": len(groups) * trainer.CANDIDATE_COUNT,
            "condition_query_seed_counts": {
                f"{condition}|query={query}": len(units[(condition, query)])
                for condition in trainer.CONDITIONS
                for query in QUERY_INDICES
            },
        }
        total += len(groups)
    if total != TOTAL_DECISIONS:
        raise AblationError("ablation input is not the complete 2,000/8,000 inventory")
    return {
        "decisions": total,
        "branches": total * trainer.CANDIDATE_COUNT,
        "root_query_indices": list(QUERY_INDICES),
        "bodies": bodies,
    }


def fold_command(
    *,
    python_executable: str,
    binding: Path,
    binding_sha256: str,
    output: Path,
    held_out_body: str,
    variant: str,
) -> list[str]:
    return [
        python_executable,
        str(Path(trainer.__file__).resolve()),
        "--mode", "train-fold",
        "--binding", str(binding),
        "--binding-sha256", binding_sha256,
        "--held-out-body", held_out_body,
        "--split-seed", str(SPLIT_SEED),
        "--output", str(output),
        "--device", "cuda",
        "--steps", str(STEPS_PER_MEMBER),
        "--eval-every", str(EVAL_EVERY),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--ablation-variant", variant,
        "--ensemble-seeds", *[str(seed) for seed in ENSEMBLE_SEEDS],
    ]


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        current = current.get(key) if isinstance(current, Mapping) else None
    return current


def _member_mean(members: Sequence[Mapping[str, Any]], *path: str) -> float | None:
    values = [_nested(member["source_validation"], path) for member in members]
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


METRICS = {
    "one_deviation_best_of_4_success_gain": (
        "candidate_ranking",
        "macro_one_deviation_branch_success_gain",
    ),
    "one_deviation_branch_selected_success_rate": (
        "candidate_ranking", "macro_selected_success_rate"
    ),
    "one_deviation_branch_oracle_success_rate": (
        "candidate_ranking", "macro_oracle_success_rate"
    ),
    "one_deviation_branch_pairwise_accuracy": (
        "candidate_ranking", "pairwise_accuracy"
    ),
    "success_brier": ("success_brier",),
    "success_auroc": ("success_auroc",),
    "post_event_macro_f1": ("post_event", "macro_f1"),
    "post_event_accuracy": ("post_event", "accuracy"),
    "next_event_macro_f1": ("next_event", "macro_f1"),
    "next_event_accuracy": ("next_event", "accuracy"),
    "duration_observed_mae_seconds": ("observed_duration_mae",),
    "duration_observed_nll": ("observed_duration_nll",),
    "object_rmse": ("object_rmse",),
    "object_nll": ("object_nll",),
    "terminal_event_accuracy": (
        "terminal_consequences", "terminal_event", "accuracy"
    ),
    "terminal_event_nll": (
        "terminal_consequences", "terminal_event", "nll"
    ),
    "terminal_goal_progress_mae_meters": (
        "terminal_consequences", "terminal_goal_progress", "mae_meters"
    ),
    "terminal_goal_progress_student_t3_nll": (
        "terminal_consequences", "terminal_goal_progress", "student_t3_nll"
    ),
}

# These are predictions made by the same five-member ensemble used for
# deployment, rather than an arithmetic mean of five separately scored models.
# Candidate-ranking metrics are added from ``evaluate_candidate_ranking`` below.
POSTHOC_ENSEMBLE_METRICS = (
    "one_deviation_best_of_4_success_gain",
    "one_deviation_branch_selected_success_rate",
    "one_deviation_branch_oracle_success_rate",
    "one_deviation_branch_pairwise_accuracy",
    "success_brier",
    "success_nll",
    "success_ece_10bin",
    "success_auroc",
    "post_event_macro_f1",
    "post_event_accuracy",
    "post_event_mixture_nll",
    "next_event_macro_f1",
    "next_event_accuracy",
    "next_event_mixture_nll",
    "terminal_event_macro_f1",
    "terminal_event_accuracy",
    "terminal_event_mixture_nll",
    "terminal_event_multiclass_brier",
    "terminal_event_confidence_ece_10bin",
    "terminal_event_ordinal_mae",
    "duration_mixture_mae_seconds",
    "duration_mixture_nll_log1p",
    "duration_mixture_observed_nll_log1p",
    "duration_mixture_censored_nll_log1p",
    "object_mixture_rmse",
    "object_student_t3_mixture_nll",
    "terminal_goal_progress_mixture_mae_meters",
    "terminal_goal_progress_mixture_rmse_meters",
    "terminal_goal_progress_student_t3_mixture_nll",
    "terminal_goal_progress_mixture_90_coverage",
    "terminal_goal_progress_mixture_90_coverage_abs_error",
    "terminal_goal_progress_mixture_90_interval_mean_width_meters",
    "recovery_brier",
    "recovery_average_precision",
    "regression_brier",
    "regression_nll",
    "joint_recovery_brier",
    "joint_recovery_nll",
    "rank_success_argmax_disagreement_rate",
    "rank_success_pairwise_disagreement_rate",
    "rank_selected_failure_aurc",
    "rank_oracle_regret_aurc",
    "success_error_aurc",
    "post_event_error_aurc",
    "next_event_error_aurc",
    "terminal_event_error_aurc",
    "duration_error_aurc",
    "object_error_aurc",
    "terminal_goal_progress_error_aurc",
    "recovery_error_aurc",
    "regression_error_aurc",
    "joint_recovery_error_aurc",
)
POSTHOC_METRIC_DIRECTIONS = {
    name: (
        "target_is_0.90"
        if name == "terminal_goal_progress_mixture_90_coverage"
        else
        "higher_is_better"
        if name
        in {
            "one_deviation_best_of_4_success_gain",
            "one_deviation_branch_selected_success_rate",
            "one_deviation_branch_oracle_success_rate",
            "one_deviation_branch_pairwise_accuracy",
            "success_auroc",
            "post_event_macro_f1",
            "post_event_accuracy",
            "next_event_macro_f1",
            "next_event_accuracy",
            "terminal_event_macro_f1",
            "terminal_event_accuracy",
            "terminal_goal_progress_mixture_90_coverage",
            "recovery_average_precision",
        }
        else "lower_is_better"
    )
    for name in POSTHOC_ENSEMBLE_METRICS
}


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean_or_none(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()) if array.size else None


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return ranks


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels > 0.5
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _rankdata_average(scores)
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.float64) > 0.5
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    cumulative_positive = np.cumsum(ordered_labels)
    average_precision = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[cursor]:
            end += 1
        group_positive = int(ordered_labels[cursor:end].sum())
        if group_positive:
            average_precision += (
                group_positive
                / positive_count
                * float(cumulative_positive[end - 1])
                / end
            )
        cursor = end
    return float(average_precision)


def _binary_precision_recall_curve(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    """Return a tie-invariant empirical PR curve and average precision."""

    labels = np.asarray(labels, dtype=np.float64) > 0.5
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0:
        return {
            "status": "unavailable_no_positive_labels",
            "positive": 0,
            "negative": negative_count,
            "average_precision": None,
            "points": [],
        }
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    cumulative_positive = np.cumsum(ordered_labels)
    points = [
        {
            "threshold": None,
            "selected": 0,
            "precision": 1.0,
            "recall": 0.0,
        }
    ]
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[cursor]:
            end += 1
        true_positive = int(cumulative_positive[end - 1])
        points.append(
            {
                "threshold": float(ordered_scores[cursor]),
                "selected": end,
                "precision": float(true_positive / end),
                "recall": float(true_positive / positive_count),
            }
        )
        cursor = end
    return {
        "status": "available",
        "positive": positive_count,
        "negative": negative_count,
        "average_precision": _average_precision(labels, scores),
        "points": points,
    }


def _macro_f1(labels: np.ndarray, predictions: np.ndarray, classes: int = 5) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if not len(labels):
        return None
    values = []
    for class_id in range(classes):
        true_positive = int(((labels == class_id) & (predictions == class_id)).sum())
        false_positive = int(((labels != class_id) & (predictions == class_id)).sum())
        false_negative = int(((labels == class_id) & (predictions != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            values.append(2.0 * true_positive / denominator)
    return float(statistics.fmean(values)) if values else None


def _logmeanexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(np.mean(np.exp(array - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _student_t3_cdf(value: np.ndarray) -> np.ndarray:
    scaled = np.asarray(value, dtype=np.float64) / math.sqrt(3.0)
    return 0.5 + (
        np.arctan(scaled) + scaled / (1.0 + np.square(scaled))
    ) / math.pi


def _student_t3_mixture_quantile(
    means: np.ndarray, scales: np.ndarray, probability: float
) -> np.ndarray:
    """Exact scalar quantile of an equal five-component Student-t(3) mixture."""

    if not 0.0 < probability < 1.0:
        raise ValueError("mixture quantile probability must be inside (0,1)")
    means = np.asarray(means, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    lower = np.min(means - 100.0 * scales, axis=0)
    upper = np.max(means + 100.0 * scales, axis=0)
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        cdf = _student_t3_cdf((midpoint[None] - means) / scales).mean(axis=0)
        lower = np.where(cdf < probability, midpoint, lower)
        upper = np.where(cdf < probability, upper, midpoint)
    return 0.5 * (lower + upper)


def _binary_ece(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = ECE_BINS
) -> tuple[float | None, list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not len(labels):
        return None, []
    indices = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    rows = []
    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        selected = indices == bin_index
        count = int(selected.sum())
        confidence = float(probabilities[selected].mean()) if count else None
        frequency = float(labels[selected].mean()) if count else None
        if count:
            ece += count / total * abs(float(confidence) - float(frequency))
        rows.append(
            {
                "bin": bin_index,
                "lower_inclusive": bin_index / bins,
                "upper_inclusive_only_for_last": (bin_index + 1) / bins,
                "count": count,
                "mean_probability": confidence,
                "empirical_frequency": frequency,
            }
        )
    return float(ece), rows


def _risk_coverage(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    *,
    error_kind: str,
    uncertainty_kind: str,
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    uncertainties = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    valid = np.isfinite(errors) & np.isfinite(uncertainties)
    errors = errors[valid]
    uncertainties = uncertainties[valid]
    if not len(errors):
        return {
            "support": 0,
            "error_kind": error_kind,
            "uncertainty_kind": uncertainty_kind,
            "aurc": None,
            "full_coverage_risk": None,
            "error_uncertainty_spearman": None,
            "risk_at_coverage": [],
        }
    # Lexsort makes original row order the deterministic tie-breaker.
    order = np.lexsort((np.arange(len(uncertainties)), uncertainties))
    ordered_error = errors[order]
    cumulative_risk = np.cumsum(ordered_error) / np.arange(1, len(errors) + 1)
    if len(errors) > 1:
        error_rank = _rankdata_average(errors)
        uncertainty_rank = _rankdata_average(uncertainties)
        if error_rank.std() > 0.0 and uncertainty_rank.std() > 0.0:
            spearman = float(np.corrcoef(error_rank, uncertainty_rank)[0, 1])
        else:
            spearman = None
    else:
        spearman = None
    points = []
    for coverage in RISK_COVERAGE_LEVELS:
        count = max(1, int(math.ceil(coverage * len(errors))))
        points.append(
            {
                "coverage": coverage,
                "retained": count,
                "risk": float(cumulative_risk[count - 1]),
            }
        )
    return {
        "support": len(errors),
        "error_kind": error_kind,
        "uncertainty_kind": uncertainty_kind,
        "ordering": "retain_lowest_uncertainty_first_stable_ties",
        "aurc": float(cumulative_risk.mean()),
        "full_coverage_risk": float(cumulative_risk[-1]),
        "error_uncertainty_spearman": spearman,
        "risk_at_coverage": points,
    }


def _dependence_cluster_from_logical_group(value: str) -> tuple[str, str, int]:
    fields = value.split("|")
    if len(fields) < 3 or not fields[0] or fields[1] not in trainer.CONDITIONS:
        raise AblationError(f"logical group lacks body/condition identity: {value!r}")
    for token in value.split("|"):
        if token.startswith("seed="):
            try:
                return fields[0], fields[1], int(token.removeprefix("seed="))
            except ValueError as error:
                raise AblationError(f"invalid requested-seed token in {value!r}") from error
    raise AblationError(f"logical group lacks a requested-seed token: {value!r}")


@torch.no_grad()
def evaluate_deployed_ensemble_predictions(
    models: Sequence[trainer.EffectAlignedSharedEventHead],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Score heldout branches with the exact five-member deployed ensemble.

    Proper probabilities are mixed in probability/density space.  Rank utilities
    use the trainer-exported epistemic lower-confidence aggregation contract.
    No target is used to fit, calibrate, threshold or select any member/variant.
    """

    if len(models) != len(ENSEMBLE_SEEDS):
        raise AblationError("deployed prediction evaluation requires five members")
    value_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    member_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    logical_groups: list[str] = []
    for model in models:
        model.eval()
    for raw in loader:
        batch = trainer.core._move_batch(raw, device)
        outputs = [model(batch) for model in models]
        logical_groups.extend(str(value) for value in raw["logical_group"])
        for name, tensor in {
            "candidate_index": batch["candidate_index"],
            "post_label": batch["post_event_id"],
            "post_mask": batch["post_event_mask"],
            "next_label": batch["next_event_id"],
            "next_mask": batch["next_event_mask"],
            "duration": batch["duration"],
            "duration_observed": batch["duration_observed"],
            "duration_mask": batch["duration_mask"],
            "success": batch["success"],
            "success_mask": batch["success_mask"],
            "recovery": batch["recovery"],
            "recovery_mask": batch["recovery_mask"] * batch["action_available"],
            "object": batch["object_delta"],
            "object_mask": batch["object_delta_mask"] * batch["action_available"],
            "terminal_event_label": batch["terminal_max_event_id"],
            "terminal_event_mask": batch["terminal_event_mask"],
            "terminal_goal_label": batch["terminal_goal_progress"],
            "terminal_goal_mask": batch["terminal_goal_progress_mask"],
            "current_event": batch["current_event_id"],
        }.items():
            value_parts[name].append(tensor.detach().cpu().numpy())
        for name, tensors in {
            "rank": [output["candidate_rank_logit"] for output in outputs],
            "success_probability": [
                torch.sigmoid(output["success_logit"]) for output in outputs
            ],
            "post_probability": [
                torch.softmax(output["post_event_logits"], -1) for output in outputs
            ],
            "next_probability": [
                torch.softmax(output["next_event_logits"], -1) for output in outputs
            ],
            "duration_log_mean": [
                output["duration_selected_log_mean"] for output in outputs
            ],
            "duration_log_scale": [
                output["duration_selected_log_scale"] for output in outputs
            ],
            "recovery_probability": [
                torch.sigmoid(output["recovery_logit"]) for output in outputs
            ],
            "object_mean": [output["object_delta_mean"] for output in outputs],
            "object_log_scale": [
                output["object_delta_log_scale"] for output in outputs
            ],
            "terminal_event_probability": [
                torch.softmax(output["terminal_event_logits"], -1)
                for output in outputs
            ],
            "terminal_goal_mean": [
                output["terminal_goal_progress_mean"] for output in outputs
            ],
            "terminal_goal_log_scale": [
                output["terminal_goal_progress_log_scale"] for output in outputs
            ],
            "regression_probability": [
                output["regression_probability"] for output in outputs
            ],
            "joint_recovery_probability": [
                output["joint_recovery_probability"] for output in outputs
            ],
        }.items():
            member_parts[name].append(
                torch.stack(tensors).detach().cpu().numpy()
            )
    values = {name: np.concatenate(parts, axis=0) for name, parts in value_parts.items()}
    members = {name: np.concatenate(parts, axis=1) for name, parts in member_parts.items()}
    row_count = len(logical_groups)
    if row_count == 0 or any(len(value) != row_count for value in values.values()):
        raise AblationError("heldout deployed prediction collection is incomplete")
    if any(value.shape[1] != row_count for value in members.values()):
        raise AblationError("heldout five-member predictions are misaligned")
    if any(not np.isfinite(value).all() for value in values.values()) or any(
        not np.isfinite(value).all() for value in members.values()
    ):
        raise AblationError("heldout deployed predictions contain non-finite values")

    epsilon = 1e-12
    success_mask = values["success_mask"] > 0.5
    success_label = values["success"][success_mask].astype(np.float64)
    success_member = members["success_probability"][:, success_mask]
    success_probability = success_member.mean(axis=0)
    success_brier = _mean_or_none(np.square(success_probability - success_label))
    success_nll = _mean_or_none(
        -success_label * np.log(np.clip(success_probability, epsilon, 1.0))
        - (1.0 - success_label)
        * np.log(np.clip(1.0 - success_probability, epsilon, 1.0))
    )
    success_ece, success_ece_bins = _binary_ece(success_label, success_probability)

    event_results: dict[str, dict[str, Any]] = {}
    event_uncertainty: dict[str, np.ndarray] = {}
    event_errors: dict[str, np.ndarray] = {}
    for prefix in ("post", "next", "terminal_event"):
        mask = values[f"{prefix}_mask"] > 0.5
        labels = values[f"{prefix}_label"][mask].astype(np.int64)
        member_probability = members[f"{prefix}_probability"][:, mask]
        probability = member_probability.mean(axis=0)
        prediction = probability.argmax(axis=-1)
        selected_probability = probability[np.arange(len(labels)), labels]
        member_entropy = -np.sum(
            member_probability * np.log(np.clip(member_probability, epsilon, 1.0)),
            axis=-1,
        )
        mixture_entropy = -np.sum(
            probability * np.log(np.clip(probability, epsilon, 1.0)), axis=-1
        )
        js_divergence = np.maximum(
            mixture_entropy - member_entropy.mean(axis=0), 0.0
        )
        errors = (prediction != labels).astype(np.float64)
        confidence_ece, _confidence_bins = _binary_ece(
            (prediction == labels).astype(np.float64),
            probability.max(axis=-1),
        )
        onehot = np.eye(5, dtype=np.float64)[labels]
        event_results[prefix] = {
            "support": len(labels),
            "class_counts": np.bincount(labels, minlength=5).tolist(),
            "macro_f1": _macro_f1(labels, prediction),
            "accuracy": _mean_or_none(prediction == labels),
            "mixture_nll": _mean_or_none(-np.log(np.clip(selected_probability, epsilon, 1.0))),
            "multiclass_brier": _mean_or_none(
                np.sum(np.square(probability - onehot), axis=-1)
            ),
            "confidence_ece_10bin": confidence_ece,
            "ordinal_mae": _mean_or_none(np.abs(prediction - labels)),
        }
        event_uncertainty[prefix] = js_divergence
        event_errors[prefix] = errors

    duration = values["duration"].astype(np.float64)
    duration_mask = values["duration_mask"] > 0.5
    duration_observed = (values["duration_observed"] > 0.5) & duration_mask
    duration_mu = members["duration_log_mean"].astype(np.float64)
    duration_scale = np.exp(members["duration_log_scale"].astype(np.float64)).clip(1e-4)
    duration_member_mean = np.maximum(
        np.expm1(np.clip(duration_mu + 0.5 * np.square(duration_scale), -30.0, 50.0)),
        0.0,
    )
    duration_prediction = duration_member_mean.mean(axis=0)
    duration_uncertainty = duration_member_mean.std(axis=0)
    transformed = np.log1p(np.maximum(duration, 0.0))[None]
    duration_z = (transformed - duration_mu) / duration_scale
    normal_log_pdf = (
        -0.5 * np.square(duration_z)
        - np.log(duration_scale)
        - 0.5 * math.log(2.0 * math.pi)
    )
    duration_observed_nll_rows = -_logmeanexp(normal_log_pdf, axis=0)
    log_survival = torch.special.log_ndtr(-torch.from_numpy(duration_z)).numpy()
    censored_nll_rows = -_logmeanexp(log_survival, axis=0)
    duration_all_nll = np.where(
        values["duration_observed"] > 0.5,
        duration_observed_nll_rows,
        censored_nll_rows,
    )

    object_mask = values["object_mask"] > 0.5
    object_label = values["object"][object_mask].astype(np.float64)
    object_mu = members["object_mean"][:, object_mask].astype(np.float64)
    object_scale = np.exp(
        members["object_log_scale"][:, object_mask].astype(np.float64)
    ).clip(1e-4)
    object_prediction = object_mu.mean(axis=0)
    object_error_rows = np.mean(np.square(object_prediction - object_label), axis=-1)
    object_standardized = (object_label[None] - object_mu) / object_scale
    student_log_constant = (
        math.lgamma((STUDENT_T_DF + 1.0) / 2.0)
        - math.lgamma(STUDENT_T_DF / 2.0)
        - 0.5 * math.log(STUDENT_T_DF * math.pi)
    )
    object_log_pdf = (
        student_log_constant
        - np.log(object_scale)
        - (STUDENT_T_DF + 1.0)
        / 2.0
        * np.log1p(np.square(object_standardized) / STUDENT_T_DF)
    )
    # One mixture component is one ensemble member's joint factorized
    # Student-t(3) density; normalize the joint NLL by the six object channels.
    object_nll_rows = -_logmeanexp(object_log_pdf.sum(axis=-1), axis=0) / object_log_pdf.shape[-1]
    object_uncertainty = object_mu.var(axis=0).mean(axis=-1)

    terminal_goal_mask = values["terminal_goal_mask"] > 0.5
    terminal_goal_label = values["terminal_goal_label"][terminal_goal_mask].astype(
        np.float64
    )
    terminal_goal_mu = members["terminal_goal_mean"][:, terminal_goal_mask].astype(
        np.float64
    )
    terminal_goal_scale = np.exp(
        members["terminal_goal_log_scale"][:, terminal_goal_mask].astype(np.float64)
    ).clip(1e-5)
    terminal_goal_prediction = terminal_goal_mu.mean(axis=0)
    terminal_goal_standardized = (
        terminal_goal_label[None] - terminal_goal_mu
    ) / terminal_goal_scale
    terminal_goal_log_pdf = (
        student_log_constant
        - np.log(terminal_goal_scale)
        - (STUDENT_T_DF + 1.0)
        / 2.0
        * np.log1p(np.square(terminal_goal_standardized) / STUDENT_T_DF)
    )
    terminal_goal_nll_rows = -_logmeanexp(terminal_goal_log_pdf, axis=0)
    terminal_goal_lower = _student_t3_mixture_quantile(
        terminal_goal_mu, terminal_goal_scale, 0.05
    )
    terminal_goal_upper = _student_t3_mixture_quantile(
        terminal_goal_mu, terminal_goal_scale, 0.95
    )
    terminal_goal_error = np.square(
        terminal_goal_prediction - terminal_goal_label
    )
    terminal_goal_uncertainty = (
        (
            3.0 * np.square(terminal_goal_scale)
            + np.square(terminal_goal_mu)
        ).mean(axis=0)
        - np.square(terminal_goal_prediction)
    ).clip(min=0.0)

    recovery_mask = values["recovery_mask"] > 0.5
    recovery_label = values["recovery"][recovery_mask].astype(np.float64)
    recovery_member = members["recovery_probability"][:, recovery_mask]
    recovery_probability = recovery_member.mean(axis=0)
    recovery_error = np.square(recovery_probability - recovery_label)
    recovery_precision_recall = _binary_precision_recall_curve(
        recovery_label, recovery_probability
    )
    regression_mask = values["post_mask"] > 0.5
    regression_label = (
        values["post_label"] < values["current_event"]
    ).astype(np.float64)[regression_mask]
    regression_member = members["regression_probability"][:, regression_mask]
    regression_probability = regression_member.mean(axis=0)
    regression_brier = np.square(regression_probability - regression_label)
    joint_recovery_label = (
        (values["recovery"] > 0.5)
        & (values["post_label"] < values["current_event"])
    ).astype(np.float64)[regression_mask]
    joint_recovery_member = members["joint_recovery_probability"][:, regression_mask]
    joint_recovery_probability = joint_recovery_member.mean(axis=0)
    joint_recovery_brier = np.square(
        joint_recovery_probability - joint_recovery_label
    )
    regression_nll = (
        -regression_label
        * np.log(np.clip(regression_probability, epsilon, 1.0))
        - (1.0 - regression_label)
        * np.log(np.clip(1.0 - regression_probability, epsilon, 1.0))
    )
    joint_recovery_nll = (
        -joint_recovery_label
        * np.log(np.clip(joint_recovery_probability, epsilon, 1.0))
        - (1.0 - joint_recovery_label)
        * np.log(np.clip(1.0 - joint_recovery_probability, epsilon, 1.0))
    )

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(logical_groups):
        by_group[group].append(index)
    rank_success_disagreement = []
    rank_member_disagreement = []
    selected_failure = []
    oracle_regret = []
    rank_pair_disagreement = rank_pair_count = 0
    for group, indices in by_group.items():
        ordered = sorted(indices, key=lambda index: int(values["candidate_index"][index]))
        if len(ordered) != trainer.CANDIDATE_COUNT or [
            int(values["candidate_index"][index]) for index in ordered
        ] != list(range(trainer.CANDIDATE_COUNT)):
            raise AblationError(f"heldout decision is incomplete: {group}")
        raw_member_scores = torch.as_tensor(members["rank"][:, ordered])
        aggregate_score = trainer.aggregate_risk_adjusted_rank_scores(
            raw_member_scores
        ).detach().cpu().numpy()
        success_score = members["success_probability"][:, ordered].mean(axis=0)
        selected = int(np.argmax(aggregate_score))
        success_selected = int(np.argmax(success_score))
        labels = values["success"][ordered].astype(np.float64)
        decision_member_scores = members["rank"][:, ordered]
        member_choices = np.argmax(decision_member_scores, axis=1)
        rank_success_disagreement.append(float(selected != success_selected))
        rank_member_disagreement.append(
            1.0
            - float(np.bincount(member_choices, minlength=4).max())
            / len(member_choices)
        )
        selected_failure.append(1.0 - float(labels[selected]))
        oracle_regret.append(float(labels.max() - labels[selected]))
        for left in range(trainer.CANDIDATE_COUNT):
            for right in range(left + 1, trainer.CANDIDATE_COUNT):
                rank_difference = aggregate_score[left] - aggregate_score[right]
                success_difference = success_score[left] - success_score[right]
                if abs(rank_difference) <= 1e-12 or abs(success_difference) <= 1e-12:
                    continue
                rank_pair_count += 1
                rank_pair_disagreement += int(rank_difference * success_difference < 0.0)
    rank_selected_failure_curve = _risk_coverage(
        np.asarray(selected_failure),
        np.asarray(rank_member_disagreement),
        error_kind="selected_candidate_binary_failure",
        uncertainty_kind=(
            "one_minus_modal_fraction_of_five_bounded_utility_member_argmax"
        ),
    )
    rank_oracle_regret_curve = _risk_coverage(
        np.asarray(oracle_regret),
        np.asarray(rank_member_disagreement),
        error_kind="oracle_success_minus_selected_success",
        uncertainty_kind=(
            "one_minus_modal_fraction_of_five_bounded_utility_member_argmax"
        ),
    )
    risk = {
        "rank_selected_failure": rank_selected_failure_curve,
        "rank_oracle_regret": rank_oracle_regret_curve,
        "success": _risk_coverage(
            np.square(success_probability - success_label),
            success_member.var(axis=0),
            error_kind="ensemble_probability_brier_row",
            uncertainty_kind="five_member_success_probability_population_variance",
        ),
        "post_event": _risk_coverage(
            event_errors["post"],
            event_uncertainty["post"],
            error_kind="ensemble_argmax_zero_one_error",
            uncertainty_kind="post_event_probability_jensen_shannon_divergence",
        ),
        "next_event": _risk_coverage(
            event_errors["next"],
            event_uncertainty["next"],
            error_kind="ensemble_argmax_zero_one_error",
            uncertainty_kind="next_event_probability_jensen_shannon_divergence",
        ),
        "terminal_event": _risk_coverage(
            event_errors["terminal_event"],
            event_uncertainty["terminal_event"],
            error_kind="ensemble_argmax_zero_one_error",
            uncertainty_kind="terminal_event_probability_jensen_shannon_divergence",
        ),
        "duration": _risk_coverage(
            np.abs(duration_prediction[duration_observed] - duration[duration_observed]),
            duration_uncertainty[duration_observed],
            error_kind="absolute_observed_duration_error_seconds",
            uncertainty_kind="five_member_predicted_duration_mean_population_std_seconds",
        ),
        "object": _risk_coverage(
            object_error_rows,
            object_uncertainty,
            error_kind="mean_squared_object_delta_error_per_row",
            uncertainty_kind="mean_dimension_five_member_object_mean_population_variance",
        ),
        "terminal_goal_progress": _risk_coverage(
            terminal_goal_error,
            terminal_goal_uncertainty,
            error_kind="squared_terminal_goal_progress_error_meters2",
            uncertainty_kind="five_member_student_t3_mixture_variance_meters2",
        ),
        "recovery": _risk_coverage(
            recovery_error,
            recovery_member.var(axis=0) if recovery_member.size else np.asarray([]),
            error_kind="ensemble_probability_brier_row",
            uncertainty_kind="five_member_recovery_probability_population_variance",
        ),
        "regression": _risk_coverage(
            regression_brier,
            regression_member.var(axis=0),
            error_kind="ensemble_probability_brier_row",
            uncertainty_kind="five_member_regression_probability_population_variance",
        ),
        "joint_recovery": _risk_coverage(
            joint_recovery_brier,
            joint_recovery_member.var(axis=0),
            error_kind="ensemble_probability_brier_row",
            uncertainty_kind="five_member_joint_recovery_probability_population_variance",
        ),
    }
    requested_seed_clusters = {
        _dependence_cluster_from_logical_group(group) for group in logical_groups
    }
    metrics = {
        "success_brier": success_brier,
        "success_nll": success_nll,
        "success_ece_10bin": success_ece,
        "success_auroc": _binary_auc(success_label, success_probability),
        "post_event_macro_f1": event_results["post"]["macro_f1"],
        "post_event_accuracy": event_results["post"]["accuracy"],
        "post_event_mixture_nll": event_results["post"]["mixture_nll"],
        "next_event_macro_f1": event_results["next"]["macro_f1"],
        "next_event_accuracy": event_results["next"]["accuracy"],
        "next_event_mixture_nll": event_results["next"]["mixture_nll"],
        "terminal_event_macro_f1": event_results["terminal_event"]["macro_f1"],
        "terminal_event_accuracy": event_results["terminal_event"]["accuracy"],
        "terminal_event_mixture_nll": event_results["terminal_event"][
            "mixture_nll"
        ],
        "terminal_event_multiclass_brier": event_results["terminal_event"][
            "multiclass_brier"
        ],
        "terminal_event_confidence_ece_10bin": event_results[
            "terminal_event"
        ]["confidence_ece_10bin"],
        "terminal_event_ordinal_mae": event_results["terminal_event"][
            "ordinal_mae"
        ],
        "duration_mixture_mae_seconds": _mean_or_none(
            np.abs(duration_prediction[duration_observed] - duration[duration_observed])
        ),
        "duration_mixture_nll_log1p": _mean_or_none(
            duration_all_nll[duration_mask]
        ),
        "duration_mixture_observed_nll_log1p": _mean_or_none(
            duration_observed_nll_rows[duration_observed]
        ),
        "duration_mixture_censored_nll_log1p": _mean_or_none(
            censored_nll_rows[duration_mask & ~duration_observed]
        ),
        "object_mixture_rmse": (
            float(math.sqrt(float(object_error_rows.mean())))
            if len(object_error_rows)
            else None
        ),
        "object_student_t3_mixture_nll": _mean_or_none(object_nll_rows),
        "terminal_goal_progress_mixture_mae_meters": _mean_or_none(
            np.abs(terminal_goal_prediction - terminal_goal_label)
        ),
        "terminal_goal_progress_mixture_rmse_meters": (
            float(np.sqrt(np.mean(terminal_goal_error)))
            if len(terminal_goal_error)
            else None
        ),
        "terminal_goal_progress_student_t3_mixture_nll": _mean_or_none(
            terminal_goal_nll_rows
        ),
        "terminal_goal_progress_mixture_90_coverage": _mean_or_none(
            (terminal_goal_label >= terminal_goal_lower)
            & (terminal_goal_label <= terminal_goal_upper)
        ),
        "terminal_goal_progress_mixture_90_coverage_abs_error": (
            None
            if not len(terminal_goal_label)
            else abs(
                float(
                    np.mean(
                        (terminal_goal_label >= terminal_goal_lower)
                        & (terminal_goal_label <= terminal_goal_upper)
                    )
                )
                - 0.90
            )
        ),
        "terminal_goal_progress_mixture_90_interval_mean_width_meters": (
            _mean_or_none(terminal_goal_upper - terminal_goal_lower)
        ),
        "recovery_brier": _mean_or_none(recovery_error),
        "recovery_average_precision": recovery_precision_recall[
            "average_precision"
        ],
        "regression_brier": _mean_or_none(regression_brier),
        "regression_nll": _mean_or_none(regression_nll),
        "joint_recovery_brier": _mean_or_none(joint_recovery_brier),
        "joint_recovery_nll": _mean_or_none(joint_recovery_nll),
        "rank_success_argmax_disagreement_rate": _mean_or_none(
            np.asarray(rank_success_disagreement)
        ),
        "rank_success_pairwise_disagreement_rate": (
            float(rank_pair_disagreement / rank_pair_count) if rank_pair_count else None
        ),
        "rank_selected_failure_aurc": rank_selected_failure_curve["aurc"],
        "rank_oracle_regret_aurc": rank_oracle_regret_curve["aurc"],
        "success_error_aurc": risk["success"]["aurc"],
        "post_event_error_aurc": risk["post_event"]["aurc"],
        "next_event_error_aurc": risk["next_event"]["aurc"],
        "terminal_event_error_aurc": risk["terminal_event"]["aurc"],
        "duration_error_aurc": risk["duration"]["aurc"],
        "object_error_aurc": risk["object"]["aurc"],
        "terminal_goal_progress_error_aurc": risk[
            "terminal_goal_progress"
        ]["aurc"],
        "recovery_error_aurc": risk["recovery"]["aurc"],
        "regression_error_aurc": risk["regression"]["aurc"],
        "joint_recovery_error_aurc": risk["joint_recovery"]["aurc"],
    }
    variant = str(getattr(models[0], "ablation_variant", "full"))
    if variant == "success_only":
        for name in (
            "terminal_event_macro_f1",
            "terminal_event_accuracy",
            "terminal_event_mixture_nll",
            "terminal_event_multiclass_brier",
            "terminal_event_confidence_ece_10bin",
            "terminal_event_ordinal_mae",
            "terminal_event_error_aurc",
        ):
            metrics[name] = None
    if variant in {"success_only", "no_object_effect"}:
        for name in (
            "terminal_goal_progress_mixture_mae_meters",
            "terminal_goal_progress_mixture_rmse_meters",
            "terminal_goal_progress_student_t3_mixture_nll",
            "terminal_goal_progress_mixture_90_coverage",
            "terminal_goal_progress_mixture_90_coverage_abs_error",
            "terminal_goal_progress_mixture_90_interval_mean_width_meters",
            "terminal_goal_progress_error_aurc",
        ):
            metrics[name] = None
    return {
        "metrics": metrics,
        "support": {
            "candidate_rows": row_count,
            "complete_four_candidate_decisions": len(by_group),
            "requested_seed_clusters": len(requested_seed_clusters),
            "success": {
                "rows": len(success_label),
                "positive": int((success_label > 0.5).sum()),
                "negative": int((success_label <= 0.5).sum()),
            },
            "post_event": event_results["post"],
            "next_event": event_results["next"],
            "terminal_event": event_results["terminal_event"],
            "duration": {
                "observed": int(duration_observed.sum()),
                "censored": int((duration_mask & ~duration_observed).sum()),
            },
            "object_rows": len(object_label),
            "terminal_goal_progress_rows": len(terminal_goal_label),
            "regression_rows": len(regression_label),
            "joint_recovery_rows": len(joint_recovery_label),
            "recovery": {
                "rows": len(recovery_label),
                "positive": int((recovery_label > 0.5).sum()),
                "negative": int((recovery_label <= 0.5).sum()),
                "precision_recall_available": (
                    recovery_precision_recall["status"] == "available"
                ),
            },
            "rank_success_pairwise_comparisons": rank_pair_count,
        },
        "success_calibration_bins": success_ece_bins,
        "recovery_precision_recall": recovery_precision_recall,
        "uncertainty_risk_coverage": risk,
        "statistical_units": {
            "prediction_observation_unit": "candidate_branch",
            "ranking_observation_unit": "complete_four_candidate_decision",
            "dependence_cluster_unit": (
                "heldout_body_condition_requested_seed_all_query_decisions_and_candidates"
            ),
            "fold_unit": "heldout_body",
            "confidence_interval_computed_here": False,
            "metrics_are_descriptive_posthoc_ablation": True,
        },
    }


def summarize_fold(
    summary: Mapping[str, Any], *, held_out_body: str, variant: str
) -> dict[str, Any]:
    members = summary.get("members")
    expected_budget = {
        "steps_per_member": STEPS_PER_MEMBER,
        "eval_every_steps": EVAL_EVERY,
        "batch_size_rows": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ensemble_members": len(ENSEMBLE_SEEDS),
    }
    current_trainer_sha = trainer.sha256_file(Path(trainer.__file__).resolve())
    ensemble_selection = summary.get("ensemble_checkpoint_selection")
    direct_rank_expected = variant != "success_only"
    if (
        summary.get("status") != "source_only_checkpoint_selection_complete"
        or summary.get("held_out_body") != held_out_body
        or summary.get("ablation") != trainer.ablation_contract(variant)
        or summary.get("candidate_rank_contract")
        != trainer.summary_candidate_rank_contract(variant)
        or summary.get("training_budget") != expected_budget
        or summary.get("heldout_labels_used_for_normalization_training_or_selection") is not False
        or summary.get("heldout_group_npz_opened") != 0
        or summary.get("preflight", {}).get("split_unit")
        != "body_condition_requested_seed_all_queries"
        or not isinstance(members, list)
        or len(members) != len(ENSEMBLE_SEEDS)
        or [member.get("seed") for member in members] != list(ENSEMBLE_SEEDS)
        or summary.get("trainer_file_sha256") != current_trainer_sha
        or summary.get("rank_supervision_available")
        is not direct_rank_expected
        or summary.get("candidate_rank_parameters_received_direct_supervision")
        is not direct_rank_expected
        or summary.get("synthetic_success_labels") != 0
        or not isinstance(ensemble_selection, Mapping)
        or ensemble_selection.get("common_step_required_for_all_five_members")
        is not True
        or ensemble_selection.get("rank_aggregation")
        != trainer.risk_adjusted_rank_ensemble_contract()
        or ensemble_selection.get("heldout_rows_used") != 0
        or any(
            member.get("best_step") != ensemble_selection.get("selected_step")
            or member.get("trainer_file_sha256") != current_trainer_sha
            for member in members
        )
    ):
        raise AblationError(f"{variant}/{held_out_body} fold contract changed")
    return {
        "held_out_body": held_out_body,
        "source_bodies": summary.get("source_bodies"),
        "member_count": len(members),
        "evaluation_role": "source_validation_used_for_checkpoint_selection",
        "candidate_ranking_estimand": trainer.ONE_DEVIATION_ESTIMAND,
        "metric_aggregation": "arithmetic_mean_of_five_selected_members",
        "metrics": {
            name: _member_mean(members, *path) for name, path in METRICS.items()
        },
    }


class _FrozenRankEnsemble(trainer.RiskAdjustedRankEnsemble):
    """Match formal inference with the frozen within-decision rank contract."""

    def __init__(
        self, models: Sequence[trainer.EffectAlignedSharedEventHead],
        variant: str = "full",
    ) -> None:
        super().__init__(models, variant)


def _load_frozen_members(
    summary: Mapping[str, Any], *, held_out_body: str, variant: str,
    device: torch.device,
) -> list[trainer.EffectAlignedSharedEventHead]:
    models = []
    current_trainer_sha = trainer.sha256_file(Path(trainer.__file__).resolve())
    selected_step = summary["ensemble_checkpoint_selection"]["selected_step"]
    direct_rank_expected = variant != "success_only"
    for expected_member, item in enumerate(summary["members"]):
        checkpoint_path = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        if (
            item.get("member") != expected_member
            or not checkpoint_path.is_file()
            or trainer.sha256_file(checkpoint_path) != item.get("checkpoint_sha256")
            or item.get("best_step") != selected_step
            or item.get("trainer_file_sha256") != current_trainer_sha
        ):
            raise AblationError(f"{variant}/{held_out_body} selected checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            checkpoint.get("format") != trainer.FORMAT
            or checkpoint.get("member") != expected_member
            or checkpoint.get("seed") != ENSEMBLE_SEEDS[expected_member]
            or checkpoint.get("held_out_body") != held_out_body
            or checkpoint.get("ablation") != trainer.ablation_contract(variant)
            or checkpoint.get("candidate_rank_contract")
            != trainer.checkpoint_candidate_rank_contract(variant)
            or checkpoint.get(
                "heldout_rows_used_for_training_normalization_or_selection"
            ) != 0
            or checkpoint.get("rank_supervision_available")
            is not direct_rank_expected
            or checkpoint.get(
                "candidate_rank_parameters_received_direct_supervision"
            )
            is not direct_rank_expected
            or checkpoint.get("synthetic_success_labels") != 0
            or checkpoint.get("trainer_file_sha256") != current_trainer_sha
            or checkpoint.get("ensemble_common_selection_step") != selected_step
        ):
            raise AblationError(f"{variant}/{held_out_body} checkpoint contract changed")
        model = trainer.EffectAlignedSharedEventHead(variant).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def evaluate_posthoc_heldout_fold(
    summary: Mapping[str, Any], audit: Mapping[str, Any], *, held_out_body: str,
    variant: str, device: torch.device,
) -> dict[str, Any]:
    """Read heldout labels only after source-only checkpoints are frozen."""

    summarize_fold(summary, held_out_body=held_out_body, variant=variant)
    models = _load_frozen_members(
        summary, held_out_body=held_out_body, variant=variant, device=device
    )
    groups = audit["manifests"][held_out_body]["groups"]
    if len(groups) != DECISIONS_PER_BODY:
        raise AblationError(f"{held_out_body} heldout inventory changed")
    rows = [
        row
        for group in groups
        for row in trainer._npz_rows(group, body=held_out_body)
    ]
    loader = DataLoader(
        trainer.core.TransitionDataset(rows, {held_out_body: 0}),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=trainer.core.collate_rows,
    )
    ranking = trainer.evaluate_candidate_ranking(
        _FrozenRankEnsemble(models, variant), loader, device,
        ablation_variant=variant,
    )
    prediction = evaluate_deployed_ensemble_predictions(models, loader, device)
    metrics = dict(prediction["metrics"])
    metrics.update(
        {
            "one_deviation_best_of_4_success_gain": _as_finite_float(
                ranking["macro_one_deviation_branch_success_gain"]
            ),
            "one_deviation_branch_selected_success_rate": _as_finite_float(
                ranking["macro_selected_success_rate"]
            ),
            "one_deviation_branch_oracle_success_rate": _as_finite_float(
                ranking["macro_oracle_success_rate"]
            ),
            "one_deviation_branch_pairwise_accuracy": _as_finite_float(
                ranking["pairwise_accuracy"]
            ),
        }
    )
    if set(metrics) != set(POSTHOC_ENSEMBLE_METRICS):
        raise AblationError(
            "posthoc deployed-ensemble metric schema changed: "
            f"missing={sorted(set(POSTHOC_ENSEMBLE_METRICS) - set(metrics))}, "
            f"extra={sorted(set(metrics) - set(POSTHOC_ENSEMBLE_METRICS))}"
        )
    return {
        "held_out_body": held_out_body,
        "source_bodies": summary.get("source_bodies"),
        "evaluation_role": "posthoc_heldout_only_after_all_checkpoint_selection",
        "candidate_ranking_estimand": ranking["estimand"],
        "heldout_decisions": len(groups),
        "heldout_branches": len(rows),
        "candidate_metric_aggregation": trainer.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT,
        "prediction_metric_aggregation": (
            "five_frozen_members_mixed_in_probability_or_density_space_then_scored"
        ),
        "heldout_labels_used_for_training_checkpoint_or_variant_selection": False,
        "metrics": metrics,
        "prediction_support": prediction["support"],
        "success_calibration_bins": prediction["success_calibration_bins"],
        "recovery_precision_recall": prediction["recovery_precision_recall"],
        "uncertainty_risk_coverage": prediction["uncertainty_risk_coverage"],
        "statistical_units": prediction["statistical_units"],
    }


def aggregate_variants(
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        folds = [
            summarize_fold(
                summaries[variant][body], held_out_body=body, variant=variant
            )
            for body in trainer.BODIES
        ]
        macro = {}
        for name in METRICS:
            values = [fold["metrics"][name] for fold in folds]
            present = [float(value) for value in values if value is not None]
            macro[name] = statistics.fmean(present) if present else None
        result[variant] = {
            "ablation": trainer.ablation_contract(variant),
            "candidate_ranking_estimand": trainer.ONE_DEVIATION_ESTIMAND,
            "folds": folds,
            "equal_fold_macro": macro,
        }
    baseline = result["success_only"]["equal_fold_macro"]
    result["comparison_to_success_only"] = {
        variant: {
            name: (
                None
                if result[variant]["equal_fold_macro"][name] is None
                or baseline[name] is None
                else result[variant]["equal_fold_macro"][name] - baseline[name]
            )
            for name in METRICS
        }
        for variant in VARIANTS
        if variant != "success_only"
    }
    return result


def aggregate_posthoc_heldout(
    evaluations: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    def aggregate_risk_curves(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        endpoints = tuple(folds[0]["uncertainty_risk_coverage"])
        if any(tuple(fold["uncertainty_risk_coverage"]) != endpoints for fold in folds):
            raise AblationError("heldout risk-coverage endpoint schema changed across folds")
        result: dict[str, Any] = {}
        for endpoint in endpoints:
            curves = [fold["uncertainty_risk_coverage"][endpoint] for fold in folds]
            if any(
                curve.get("error_kind") != curves[0].get("error_kind")
                or curve.get("uncertainty_kind")
                != curves[0].get("uncertainty_kind")
                for curve in curves[1:]
            ):
                raise AblationError(
                    f"heldout risk-coverage contract changed for {endpoint}"
                )
            scalar = {}
            for name in ("aurc", "full_coverage_risk", "error_uncertainty_spearman"):
                present = [
                    float(curve[name]) for curve in curves if curve.get(name) is not None
                ]
                scalar[name] = statistics.fmean(present) if present else None
            coverage_rows = []
            for coverage in RISK_COVERAGE_LEVELS:
                risks = []
                for curve in curves:
                    row = next(
                        (
                            item
                            for item in curve["risk_at_coverage"]
                            if item["coverage"] == coverage
                        ),
                        None,
                    )
                    if row is not None:
                        risks.append(float(row["risk"]))
                coverage_rows.append(
                    {
                        "coverage": coverage,
                        "equal_fold_mean_risk": (
                            statistics.fmean(risks) if risks else None
                        ),
                        "folds_with_support": len(risks),
                    }
                )
            result[endpoint] = {
                "folds_with_support": sum(curve.get("support", 0) > 0 for curve in curves),
                "equal_fold_macro": scalar,
                "risk_at_coverage_equal_fold_macro": coverage_rows,
                "error_kind": curves[0]["error_kind"],
                "uncertainty_kind": curves[0]["uncertainty_kind"],
            }
        return result

    result: dict[str, Any] = {}
    for variant in VARIANTS:
        folds = [evaluations[variant][body] for body in trainer.BODIES]
        macro = {}
        for name in POSTHOC_ENSEMBLE_METRICS:
            values = [fold["metrics"][name] for fold in folds]
            present = [float(value) for value in values if value is not None]
            macro[name] = statistics.fmean(present) if present else None
        result[variant] = {
            "ablation": trainer.ablation_contract(variant),
            "candidate_ranking_estimand": trainer.ONE_DEVIATION_ESTIMAND,
            "folds": folds,
            "equal_fold_macro": macro,
            "metric_folds_with_support": {
                name: sum(
                    fold["metrics"][name] is not None for fold in folds
                )
                for name in POSTHOC_ENSEMBLE_METRICS
            },
            "uncertainty_risk_coverage_equal_fold_macro": aggregate_risk_curves(folds),
        }
    baseline = result["success_only"]["equal_fold_macro"]
    result["comparison_to_success_only"] = {
        variant: {
            name: (
                None
                if result[variant]["equal_fold_macro"][name] is None
                or baseline[name] is None
                else result[variant]["equal_fold_macro"][name] - baseline[name]
            )
            for name in POSTHOC_ENSEMBLE_METRICS
        }
        for variant in VARIANTS
        if variant != "success_only"
    }
    result["effect_aligned_improvement_over_success_only"] = {
        variant: {
            name: (
                None
                if result[variant]["equal_fold_macro"][name] is None
                or baseline[name] is None
                else (
                    result[variant]["equal_fold_macro"][name] - baseline[name]
                    if POSTHOC_METRIC_DIRECTIONS[name] == "higher_is_better"
                    else (
                        abs(baseline[name] - 0.90)
                        - abs(result[variant]["equal_fold_macro"][name] - 0.90)
                        if POSTHOC_METRIC_DIRECTIONS[name] == "target_is_0.90"
                        else baseline[name]
                        - result[variant]["equal_fold_macro"][name]
                    )
                )
            )
            for name in POSTHOC_ENSEMBLE_METRICS
        }
        for variant in VARIANTS
        if variant != "success_only"
    }
    result["metric_directions"] = dict(POSTHOC_METRIC_DIRECTIONS)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    binding = args.binding.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise AblationError("ablation output must be a new directory")
    audit = trainer.load_binding(binding, args.binding_sha256)
    inventory = validate_complete_inventory(audit)
    output.mkdir(parents=True)
    summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant in VARIANTS:
        summaries[variant] = {}
        for body in trainer.BODIES:
            fold_output = output / variant / f"outer_lobo_{body}"
            command = fold_command(
                python_executable=args.python_executable,
                binding=binding,
                binding_sha256=args.binding_sha256,
                output=fold_output,
                held_out_body=body,
                variant=variant,
            )
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise AblationError(f"{variant}/{body} training exited {result.returncode}")
            summary_path = fold_output / "training_summary.json"
            if not summary_path.is_file():
                raise AblationError(f"{variant}/{body} did not produce a summary")
            summaries[variant][body] = json.loads(summary_path.read_text(encoding="utf-8"))
    # Deliberate two-phase boundary: no heldout NPZ is opened until every one
    # of the 20 runs has completed source-only checkpoint selection.
    device = torch.device("cuda")
    heldout_evaluations: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant in VARIANTS:
        heldout_evaluations[variant] = {}
        for body in trainer.BODIES:
            heldout_evaluations[variant][body] = evaluate_posthoc_heldout_fold(
                summaries[variant][body], audit,
                held_out_body=body, variant=variant, device=device,
            )
    document = {
        "format": FORMAT,
        "status": STATUS,
        "binding": str(binding),
        "binding_file_sha256": args.binding_sha256,
        "inventory": inventory,
        "fixed_budget": {
            "split_seed": SPLIT_SEED,
            "root_query_indices": list(QUERY_INDICES),
            "steps_per_member": STEPS_PER_MEMBER,
            "eval_every_steps": EVAL_EVERY,
            "batch_size_rows": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "variants": list(VARIANTS),
            "folds_per_variant": len(trainer.BODIES),
            "heldout_labels_used_for_checkpoint_selection": False,
            "all_checkpoints_selected_before_any_heldout_payload_open": True,
            "variant_selection_performed": False,
            "heldout_results_reporting_only": True,
        },
        "posthoc_prediction_evaluation_contract": {
            "candidate_rank_ensemble": trainer.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT,
            "proper_prediction_aggregation": (
                "five_frozen_members_mixed_in_probability_or_density_space_before_scoring"
            ),
            "uncertainty_source": "disagreement_among_same_five_frozen_members",
            "metric_directions": dict(POSTHOC_METRIC_DIRECTIONS),
            "prediction_observation_unit": "candidate_branch",
            "ranking_observation_unit": "complete_four_candidate_decision",
            "dependence_cluster_unit": (
                "heldout_body_condition_requested_seed_all_query_decisions_and_candidates"
            ),
            "fold_unit": "heldout_body",
            "macro_aggregation": "equal_weight_across_five_heldout_body_folds",
            "confidence_interval_computed": False,
            "heldout_labels_used_for_checkpoint_or_variant_selection": False,
            "posthoc_results_may_select_variant_for_same_heldout_claim": False,
        },
        "results": {
            "source_validation_member_mean": aggregate_variants(summaries),
            "posthoc_heldout": aggregate_posthoc_heldout(heldout_evaluations),
        },
    }
    trainer.core.atomic_json(output / "offline_ablation_summary.json", document)
    print("OFFLINE_ABLATION_COMPLETE=" + json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AblationError", "METRICS", "POSTHOC_ENSEMBLE_METRICS",
    "POSTHOC_METRIC_DIRECTIONS", "QUERY_INDICES",
    "VARIANTS", "aggregate_posthoc_heldout", "aggregate_variants",
    "evaluate_deployed_ensemble_predictions", "evaluate_posthoc_heldout_fold",
    "fold_command", "summarize_fold", "validate_complete_inventory",
]
