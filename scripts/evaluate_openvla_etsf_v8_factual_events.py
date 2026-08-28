#!/usr/bin/env python3
"""Strict development-only OOF diagnostics for frozen v8 factual event heads.

The evaluator measures prediction quality from labels.  A bit-exact factual
checkpoint is useful provenance, but is deliberately never treated as evidence
of event accuracy.  Frequency baselines are fitted independently inside every
owner fold's outer-training split, and uncertainty intervals resample complete
logical groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from openvla_etsf_counterfactual_oof import canonical_sha256
from train_openvla_etsf_v8_structured_adapters import structured_payload_sha256


FORMAT = "etsf_v8_factual_event_oof_diagnostics_v1"
MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
TRAINING_FORMAT = "etsf_v8_detached_adapter_training_input_v1"
HOLDOUT_FORMAT = "etsf_v8_detached_adapter_holdout_input_v1"
FOLD_COUNT = 5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
EPS = 1e-12

OOF_REQUIRED_FIELDS = {
    "logical_group",
    "owner_fold_id",
    "current_event_id",
    "next_event_id",
    "next_reached_event_id",
    "structured_mask",
    "duration_observed",
    "next_event_logits",
    "next_reached_event_logits",
}
TRAIN_REQUIRED_FIELDS = {
    "logical_group",
    "next_event_id",
    "next_reached_event_id",
    "structured_mask",
    "duration_observed",
}
SINGLE_MEMBER_TOTAL_STATUS = (
    "unavailable_single_forward_has_aleatoric_only_requires_ensemble_fail_closed"
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    result = _as_numpy(value)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be an aligned vector")
    return result


def _binary(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _vector(value, name=name, length=length)
    if not np.isfinite(result.astype(np.float64)).all() or bool(
        ((result != 0) & (result != 1)).any()
    ):
        raise ValueError(f"{name} must contain only finite zero/one values")
    return result.astype(bool)


def _ids(value: Any, *, name: str, length: int, classes: int) -> np.ndarray:
    result = _vector(value, name=name, length=length)
    numeric = result.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, numeric.astype(np.int64)):
        raise ValueError(f"{name} must contain finite integer ids")
    result = numeric.astype(np.int64)
    if bool(((result < 0) | (result >= classes)).any()):
        raise ValueError(f"{name} ids fall outside the event vocabulary")
    return result


def _groups(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _vector(value, name=name, length=length).astype(str)
    if any(not item for item in result.tolist()):
        raise ValueError(f"{name} contains an empty logical group")
    return result


def _logits(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _as_numpy(value).astype(np.float64)
    if result.ndim != 2 or result.shape[0] != length or result.shape[1] < 2:
        raise ValueError(f"{name} must have shape [rows,event_classes]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def _validate_oof_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = sorted(OOF_REQUIRED_FIELDS - set(arrays))
    if missing:
        raise ValueError(f"factual OOF arrays missing fields: {missing}")
    groups_raw = _as_numpy(arrays["logical_group"])
    if groups_raw.ndim != 1 or not len(groups_raw):
        raise ValueError("factual OOF needs non-empty logical_group rows")
    length = len(groups_raw)
    immediate_logits = _logits(
        arrays["next_event_logits"], name="next_event_logits", length=length
    )
    destination_logits = _logits(
        arrays["next_reached_event_logits"],
        name="next_reached_event_logits",
        length=length,
    )
    if destination_logits.shape != immediate_logits.shape:
        raise ValueError("immediate and destination event vocabularies differ")
    classes = int(immediate_logits.shape[1])
    result = {
        "logical_group": _groups(
            arrays["logical_group"], name="logical_group", length=length
        ),
        "owner_fold_id": _ids(
            arrays["owner_fold_id"],
            name="owner_fold_id",
            length=length,
            classes=FOLD_COUNT,
        ),
        "current_event_id": _ids(
            arrays["current_event_id"],
            name="current_event_id",
            length=length,
            classes=classes,
        ),
        "next_event_id": _ids(
            arrays["next_event_id"],
            name="next_event_id",
            length=length,
            classes=classes,
        ),
        "next_reached_event_id": _ids(
            arrays["next_reached_event_id"],
            name="next_reached_event_id",
            length=length,
            classes=classes,
        ),
        "structured_mask": _binary(
            arrays["structured_mask"], name="structured_mask", length=length
        ),
        "duration_observed": _binary(
            arrays["duration_observed"], name="duration_observed", length=length
        ),
        "next_event_logits": immediate_logits,
        "next_reached_event_logits": destination_logits,
    }
    has_aleatoric = "aleatoric_uncertainty" in arrays
    has_total = "total_uncertainty" in arrays
    if not has_aleatoric and not has_total:
        raise ValueError(
            "factual OOF arrays need signed aleatoric_uncertainty or a provenance-bound "
            "aleatoric-only total_uncertainty alias"
        )
    if has_total:
        provenance = arrays.get("uncertainty_provenance")
        if not isinstance(provenance, Mapping) or (
            int(provenance.get("factual_members", -1)) != 1
            or provenance.get("total_uncertainty_semantics")
            != "aleatoric_only_single_factual_member_alias"
            or provenance.get("epistemic_uncertainty_available") is not False
        ):
            raise ValueError(
                "total_uncertainty requires provenance declaring a one-member "
                "aleatoric-only alias with no epistemic estimate"
            )
    uncertainty_name = "aleatoric_uncertainty" if has_aleatoric else "total_uncertainty"
    uncertainty = _vector(
        arrays[uncertainty_name], name=uncertainty_name, length=length
    ).astype(np.float64)
    if has_aleatoric and has_total:
        total = _vector(
            arrays["total_uncertainty"], name="total_uncertainty", length=length
        ).astype(np.float64)
        if not np.array_equal(total, uncertainty):
            raise ValueError(
                "single-member total_uncertainty alias differs from aleatoric_uncertainty"
            )
    if not np.isfinite(uncertainty).all() or bool((uncertainty < 0).any()):
        raise ValueError("aleatoric uncertainty must be finite and non-negative")
    result["aleatoric_uncertainty"] = uncertainty
    for group in np.unique(result["logical_group"]):
        owners = np.unique(result["owner_fold_id"][result["logical_group"] == group])
        if len(owners) != 1:
            raise ValueError("one logical group has multiple OOF owner folds")
    if set(result["owner_fold_id"].tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("factual OOF rows must cover exactly five owner folds")
    return result


def _validate_training_arrays(
    value: Mapping[str, Any],
    *,
    fold_id: int,
    classes: int,
    heldout_groups: set[str],
    expected_training_groups: set[str],
) -> dict[str, np.ndarray]:
    missing = sorted(TRAIN_REQUIRED_FIELDS - set(value))
    if missing:
        raise ValueError(f"outer-training fold {fold_id} missing fields: {missing}")
    raw_groups = _as_numpy(value["logical_group"])
    if raw_groups.ndim != 1 or not len(raw_groups):
        raise ValueError(f"outer-training fold {fold_id} is empty")
    length = len(raw_groups)
    result = {
        "logical_group": _groups(
            value["logical_group"], name="outer-training logical_group", length=length
        ),
        "next_event_id": _ids(
            value["next_event_id"],
            name="outer-training next_event_id",
            length=length,
            classes=classes,
        ),
        "next_reached_event_id": _ids(
            value["next_reached_event_id"],
            name="outer-training next_reached_event_id",
            length=length,
            classes=classes,
        ),
        "structured_mask": _binary(
            value["structured_mask"], name="outer-training structured_mask", length=length
        ),
        "duration_observed": _binary(
            value["duration_observed"],
            name="outer-training duration_observed",
            length=length,
        ),
    }
    actual_training_groups = set(result["logical_group"].tolist())
    overlap = actual_training_groups & heldout_groups
    if overlap:
        raise ValueError(
            f"outer-training fold {fold_id} overlaps its holdout groups: {sorted(overlap)}"
        )
    if actual_training_groups != expected_training_groups:
        raise ValueError(
            f"outer-training fold {fold_id} is not the exact complement of its holdout"
        )
    if not bool(result["structured_mask"].any()):
        raise ValueError(f"outer-training fold {fold_id} lacks structured event labels")
    destination = result["structured_mask"] & result["duration_observed"]
    if not bool(destination.any()):
        raise ValueError(f"outer-training fold {fold_id} lacks observed destination labels")
    return result


def _categorical_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if labels.ndim != 1 or probability.ndim != 2 or probability.shape[0] != len(labels):
        raise ValueError("categorical labels/probabilities are not aligned")
    if not len(labels):
        raise ValueError("categorical metrics require non-empty support")
    classes = probability.shape[1]
    if not np.isfinite(probability).all() or bool((probability < 0).any()):
        raise ValueError("categorical probabilities must be finite and non-negative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("categorical probabilities must sum to one")
    predicted = probability.argmax(axis=1)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    recall: list[float | None] = []
    f1: list[float | None] = []
    for class_id in range(classes):
        if support[class_id] == 0:
            recall.append(None)
            f1.append(None)
            continue
        true_positive = int(confusion[class_id, class_id])
        predicted_support = int(confusion[:, class_id].sum())
        class_recall = true_positive / int(support[class_id])
        precision = true_positive / predicted_support if predicted_support else 0.0
        recall.append(float(class_recall))
        f1.append(
            float(2.0 * precision * class_recall / (precision + class_recall))
            if precision + class_recall
            else 0.0
        )
    return {
        "support_rows": int(len(labels)),
        "class_support": support.astype(int).tolist(),
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy_present_classes": float(
            np.mean([item for item in recall if item is not None])
        ),
        "macro_f1_present_classes": float(
            np.mean([item for item in f1 if item is not None])
        ),
        "nll": float(
            -np.log(np.clip(probability[np.arange(len(labels)), labels], EPS, 1.0)).mean()
        ),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "per_class_recall": recall,
        "per_class_f1": f1,
    }


def _frequency_probability(
    labels: np.ndarray, *, classes: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=classes).astype(np.int64)
    smoothed = (counts.astype(np.float64) + 1.0) / (int(counts.sum()) + classes)
    return counts, smoothed


def _aurc(error: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if error.ndim != 1 or error.shape != uncertainty.shape or not len(error):
        raise ValueError("AURC inputs must be non-empty aligned vectors")
    order = np.argsort(uncertainty, kind="stable")
    ordered_uncertainty = uncertainty[order]
    ordered_error = error[order]
    expected_cumulative_error = np.empty(len(error), dtype=np.float64)
    errors_before = 0.0
    start = 0
    while start < len(error):
        stop = start + 1
        while stop < len(error) and ordered_uncertainty[stop] == ordered_uncertainty[start]:
            stop += 1
        block_errors = float(ordered_error[start:stop].sum())
        block_size = stop - start
        offsets = np.arange(1, block_size + 1, dtype=np.float64)
        expected_cumulative_error[start:stop] = errors_before + offsets * block_errors / block_size
        errors_before += block_errors
        start = stop
    risk = expected_cumulative_error / np.arange(1, len(error) + 1)
    coverage = {}
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
        accepted = max(1, int(math.ceil(fraction * len(error))))
        coverage[f"{fraction:.2f}"] = float(risk[accepted - 1])
    random_risk = float(error.mean())
    aurc = float(risk.mean())
    return {
        "support_rows": int(len(error)),
        "aurc": aurc,
        "random_order_expected_aurc": random_risk,
        "aurc_improvement_over_random": random_risk - aurc,
        "risk_at_coverage": coverage,
        "tie_policy": "expected_random_order_within_equal_uncertainty",
    }


def _uncertainty_bins(
    error: np.ndarray, uncertainty: np.ndarray, groups: np.ndarray
) -> list[dict[str, Any]]:
    error = np.asarray(error, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if groups.shape != error.shape:
        raise ValueError("uncertainty bins require aligned logical groups")

    def group_mean(mask: np.ndarray) -> float:
        unique = np.unique(groups[mask])
        return float(np.mean([error[mask & (groups == group)].mean() for group in unique]))

    cutpoints = np.unique(np.quantile(uncertainty, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(cutpoints) == 1:
        mask = np.ones(len(error), dtype=bool)
        return [
            {
                "bin": 0,
                "support_rows": int(len(error)),
                "support_logical_groups": int(len(np.unique(groups))),
                "uncertainty_min": float(cutpoints[0]),
                "uncertainty_max": float(cutpoints[0]),
                "top1_error_rate": float(error.mean()),
                "equal_logical_group_mean_top1_error_rate": group_mean(mask),
            }
        ]
    assignments = np.searchsorted(cutpoints[1:-1], uncertainty, side="right")
    rows = []
    for bin_id in range(len(cutpoints) - 1):
        mask = assignments == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": int(bin_id),
                "support_rows": int(mask.sum()),
                "support_logical_groups": int(len(np.unique(groups[mask]))),
                "uncertainty_min": float(uncertainty[mask].min()),
                "uncertainty_max": float(uncertainty[mask].max()),
                "top1_error_rate": float(error[mask].mean()),
                "equal_logical_group_mean_top1_error_rate": group_mean(mask),
            }
        )
    return rows


def _cluster_blocks(groups: np.ndarray) -> list[np.ndarray]:
    groups = np.asarray(groups).astype(str)
    return [np.flatnonzero(groups == group) for group in np.unique(groups)]


def _bootstrap_indices(
    blocks: Sequence[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    selected = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[index] for index in selected])


def _interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def _clustered_comparisons(
    labels: np.ndarray,
    model: np.ndarray,
    baselines: Mapping[str, np.ndarray],
    groups: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    model_metrics = _categorical_metrics(labels, model)
    blocks = _cluster_blocks(groups)
    output: dict[str, Any] = {}
    for offset, (name, baseline) in enumerate(sorted(baselines.items())):
        baseline_metrics = _categorical_metrics(labels, baseline)
        estimates = {
            "accuracy_model_minus_baseline": model_metrics["accuracy"]
            - baseline_metrics["accuracy"],
            "balanced_accuracy_model_minus_baseline": model_metrics[
                "balanced_accuracy_present_classes"
            ]
            - baseline_metrics["balanced_accuracy_present_classes"],
            "macro_f1_model_minus_baseline": model_metrics[
                "macro_f1_present_classes"
            ]
            - baseline_metrics["macro_f1_present_classes"],
            "nll_model_minus_baseline": model_metrics["nll"] - baseline_metrics["nll"],
        }
        samples = {key: [] for key in estimates}
        rng = np.random.default_rng(bootstrap_seed + offset)
        for _ in range(bootstrap_samples):
            index = _bootstrap_indices(blocks, rng)
            sampled_model = _categorical_metrics(labels[index], model[index])
            sampled_baseline = _categorical_metrics(labels[index], baseline[index])
            samples["accuracy_model_minus_baseline"].append(
                sampled_model["accuracy"] - sampled_baseline["accuracy"]
            )
            samples["balanced_accuracy_model_minus_baseline"].append(
                sampled_model["balanced_accuracy_present_classes"]
                - sampled_baseline["balanced_accuracy_present_classes"]
            )
            samples["macro_f1_model_minus_baseline"].append(
                sampled_model["macro_f1_present_classes"]
                - sampled_baseline["macro_f1_present_classes"]
            )
            samples["nll_model_minus_baseline"].append(
                sampled_model["nll"] - sampled_baseline["nll"]
            )
        output[name] = {
            key: {"estimate": float(value), **_interval(samples[key])}
            for key, value in estimates.items()
        }
    return output


def _uncertainty_report(
    labels: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    error = (probability.argmax(axis=1) != labels).astype(np.float64)
    point = _aurc(error, uncertainty)
    blocks = _cluster_blocks(groups)
    rng = np.random.default_rng(bootstrap_seed)
    improvement_samples = []
    aurc_samples = []
    for _ in range(bootstrap_samples):
        index = _bootstrap_indices(blocks, rng)
        sample = _aurc(error[index], uncertainty[index])
        aurc_samples.append(sample["aurc"])
        improvement_samples.append(sample["aurc_improvement_over_random"])
    return {
        **point,
        "score_semantics": (
            "frozen_single_member_composite_aleatoric_score_not_"
            "epistemic_and_not_destination_specific"
        ),
        "cluster_bootstrap_95": {
            "aurc": _interval(aurc_samples),
            "aurc_improvement_over_random": _interval(improvement_samples),
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "resampling_unit": "logical_group",
        },
        "uncertainty_quantile_bins": _uncertainty_bins(error, uncertainty, groups),
        "per_owner_fold": {
            str(fold_id): _aurc(error[folds == fold_id], uncertainty[folds == fold_id])
            for fold_id in range(FOLD_COUNT)
        },
    }


def _domain_report(
    *,
    name: str,
    labels: np.ndarray,
    current: np.ndarray,
    logits: np.ndarray,
    uncertainty: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    frequency_probability: np.ndarray,
    frequency_counts: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    classes = logits.shape[1]
    model = _softmax(logits)
    persistence = np.eye(classes, dtype=np.float64)[current]
    baselines = {
        "owner_fold_add_one_frequency": frequency_probability,
        "current_event_self_loop": persistence,
    }
    return {
        "target": name,
        "support_rows": int(len(labels)),
        "support_logical_groups": int(len(np.unique(groups))),
        "model": _categorical_metrics(labels, model),
        "owner_fold_frequency_baseline": {
            **_categorical_metrics(labels, frequency_probability),
            "fit_scope": "owner_fold_outer_training_only",
            "smoothing": "add_one_laplace",
            "training_counts_by_owner_fold": dict(frequency_counts),
        },
        "current_event_self_loop_baseline": {
            **_categorical_metrics(labels, persistence),
            "probability_contract": "one_hot_current_event_clipped_only_for_finite_nll",
        },
        "cluster_bootstrap_model_minus_baselines": _clustered_comparisons(
            labels,
            model,
            baselines,
            groups,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "per_owner_fold": {
            str(fold_id): {
                "model": _categorical_metrics(
                    labels[folds == fold_id], model[folds == fold_id]
                ),
                "owner_fold_frequency_baseline": _categorical_metrics(
                    labels[folds == fold_id],
                    frequency_probability[folds == fold_id],
                ),
                "current_event_self_loop_baseline": _categorical_metrics(
                    labels[folds == fold_id], persistence[folds == fold_id]
                ),
            }
            for fold_id in range(FOLD_COUNT)
        },
        "uncertainty": _uncertainty_report(
            labels,
            model,
            uncertainty,
            groups,
            folds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100,
        ),
    }


def evaluate_factual_event_arrays(
    oof_arrays: Mapping[str, Any],
    outer_training_by_fold: Mapping[int | str, Mapping[str, Any]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate frozen factual event heads without authorizing any policy use."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    oof = _validate_oof_arrays(oof_arrays)
    classes = int(oof["next_event_logits"].shape[1])
    normalized_folds = {int(key): value for key, value in outer_training_by_fold.items()}
    if set(normalized_folds) != set(range(FOLD_COUNT)):
        raise ValueError("outer-training frequency inputs must cover exactly five folds")
    training: dict[int, dict[str, np.ndarray]] = {}
    all_groups = set(oof["logical_group"].tolist())
    for fold_id in range(FOLD_COUNT):
        heldout = set(
            oof["logical_group"][oof["owner_fold_id"] == fold_id].tolist()
        )
        training[fold_id] = _validate_training_arrays(
            normalized_folds[fold_id],
            fold_id=fold_id,
            classes=classes,
            heldout_groups=heldout,
            expected_training_groups=all_groups - heldout,
        )

    immediate_mask = oof["structured_mask"]
    destination_mask = oof["structured_mask"] & oof["duration_observed"]
    for fold_id in range(FOLD_COUNT):
        fold_rows = oof["owner_fold_id"] == fold_id
        if not bool((fold_rows & immediate_mask).any()):
            raise ValueError(f"owner fold {fold_id} lacks immediate-event support")
        if not bool((fold_rows & destination_mask).any()):
            raise ValueError(f"owner fold {fold_id} lacks observed destination support")

    frequency: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, Any]] = {"immediate": {}, "destination": {}}
    for target, mask_name in (
        ("immediate", "structured_mask"),
        ("destination", "destination"),
    ):
        probability = np.empty((len(oof["logical_group"]), classes), dtype=np.float64)
        for fold_id in range(FOLD_COUNT):
            train = training[fold_id]
            mask = (
                train["structured_mask"]
                if mask_name == "structured_mask"
                else train["structured_mask"] & train["duration_observed"]
            )
            label_name = (
                "next_event_id" if target == "immediate" else "next_reached_event_id"
            )
            fold_counts, fold_probability = _frequency_probability(
                train[label_name][mask], classes=classes
            )
            probability[oof["owner_fold_id"] == fold_id] = fold_probability
            counts[target][str(fold_id)] = {
                "outer_training_rows": int(mask.sum()),
                "class_counts": fold_counts.tolist(),
                "probability": fold_probability.tolist(),
            }
        frequency[target] = probability

    def select(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return value[mask]

    immediate = _domain_report(
        name="next_event_id",
        labels=select(oof["next_event_id"], immediate_mask),
        current=select(oof["current_event_id"], immediate_mask),
        logits=select(oof["next_event_logits"], immediate_mask),
        uncertainty=select(oof["aleatoric_uncertainty"], immediate_mask),
        groups=select(oof["logical_group"], immediate_mask),
        folds=select(oof["owner_fold_id"], immediate_mask),
        frequency_probability=select(frequency["immediate"], immediate_mask),
        frequency_counts=counts["immediate"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    destination = _domain_report(
        name="next_reached_event_id_observed_only",
        labels=select(oof["next_reached_event_id"], destination_mask),
        current=select(oof["current_event_id"], destination_mask),
        logits=select(oof["next_reached_event_logits"], destination_mask),
        uncertainty=select(oof["aleatoric_uncertainty"], destination_mask),
        groups=select(oof["logical_group"], destination_mask),
        folds=select(oof["owner_fold_id"], destination_mask),
        frequency_probability=select(frequency["destination"], destination_mask),
        frequency_counts=counts["destination"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1_000,
    )
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete_adaptive_development_only",
        "evidence_scope": "D250_adaptive_development_only_not_prospective",
        "rows": int(len(oof["logical_group"])),
        "logical_groups": int(len(np.unique(oof["logical_group"]))),
        "event_classes": classes,
        "destination_mask": "structured_and_duration_observed_only_no_censored_placeholder",
        "immediate_event": immediate,
        "observed_destination_event": destination,
        "frozen_factual_state": {
            "bit_exact_is_accuracy_evidence": False,
            "accuracy_measured_from_labels_and_logits": True,
            "accuracy_status": "evaluated_not_inferred_from_freeze_hash",
        },
        "uncertainty_scope": {
            "evaluated_quantity": "single_factual_member_composite_aleatoric_score",
            "epistemic_uncertainty_available": False,
            "complete_predictive_uncertainty_claimed": False,
            "destination_specific_uncertainty_claimed": False,
            "allowed_claim": "aleatoric_risk_ordering_diagnostic_only",
        },
        "authorization": {
            "fresh50_confirmation_authorized": False,
            "selector_authorized": False,
            "deployment_authorized": False,
            "policy_success_claim_authorized": False,
        },
        "fresh_confirmation_data_or_labels_read": False,
        "input_authentication": (
            "caller_supplied_arrays_validated_semantically_use_"
            "evaluate_materialization_manifest_for_artifact_authentication"
        ),
        "bootstrap": {
            "resampling_unit": "logical_group",
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "confidence": 0.95,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _records_to_arrays(
    payload: Mapping[str, Any], *, fold_id: int, expected_role: str, include_logits: bool
) -> dict[str, np.ndarray]:
    expected_format = TRAINING_FORMAT if expected_role == "outer_training" else HOLDOUT_FORMAT
    if payload.get("format") != expected_format:
        raise ValueError(f"fold {fold_id} has the wrong {expected_role} payload format")
    if payload.get("payload_sha256") != structured_payload_sha256(payload):
        raise ValueError(f"fold {fold_id} {expected_role} payload SHA mismatch")
    provenance = payload.get("provenance")
    records = payload.get("batches")
    if not isinstance(provenance, Mapping) or int(provenance.get("outer_fold_id", -1)) != fold_id:
        raise ValueError(f"fold {fold_id} {expected_role} provenance mismatch")
    uncertainty_contract = provenance.get("uncertainty_materialization_contract")
    if not isinstance(uncertainty_contract, Mapping):
        raise ValueError("materialized factual payload lacks uncertainty provenance")
    uncertainty_unsigned = dict(uncertainty_contract)
    uncertainty_sha = uncertainty_unsigned.pop(
        "uncertainty_materialization_contract_sha256", None
    )
    if (
        uncertainty_sha != canonical_sha256(uncertainty_unsigned)
        or uncertainty_sha
        != provenance.get("uncertainty_materialization_contract_sha256")
        or uncertainty_contract.get("stored_tensor") != "aleatoric_uncertainty"
        or uncertainty_contract.get("epistemic_uncertainty")
        != "unavailable_requires_frozen_ensemble"
        or uncertainty_contract.get("total_uncertainty")
        != "unavailable_not_fabricated_fail_closed"
        or uncertainty_contract.get("ensemble_total_uncertainty_claim") is not False
    ):
        raise ValueError("materialized single-member uncertainty provenance changed")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError(f"fold {fold_id} {expected_role} records are empty")
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "logical_group",
            "current_event_id",
            "next_event_id",
            "next_reached_event_id",
            "structured_mask",
            "duration_observed",
            "next_event_logits",
            "next_reached_event_logits",
            "aleatoric_uncertainty",
            "total_uncertainty",
        )
    }
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("materialized factual record must be a mapping")
        if record.get("split_role") != expected_role or int(record.get("outer_fold_id", -1)) != fold_id:
            raise ValueError(f"fold {fold_id} record role/owner mismatch")
        if record.get("total_uncertainty_status") != SINGLE_MEMBER_TOTAL_STATUS:
            raise ValueError("materialized record fabricates single-member total uncertainty")
        batch = record.get("batch")
        factual = record.get("factual_outputs")
        if not isinstance(batch, Mapping) or not isinstance(factual, Mapping):
            raise ValueError("materialized factual record lacks batch/factual outputs")
        if "current_event_id" not in batch:
            if "clock_event_id" in batch:
                raise ValueError(
                    "materialized batch lacks dynamic current_event_id; authenticated "
                    "clock_event_id is the canonical duration-clock label and cannot "
                    "replace it; rematerialization is required"
                )
            raise ValueError("materialized batch lacks current_event_id")
        count = len(_vector(batch.get("structured_mask"), name="structured_mask"))
        group = str(record.get("logical_group_key", ""))
        if not group:
            raise ValueError("materialized factual record lacks logical_group_key")
        parts["logical_group"].append(np.repeat(group, count))
        for name in (
            "current_event_id",
            "next_event_id",
            "next_reached_event_id",
            "structured_mask",
            "duration_observed",
        ):
            if name not in batch:
                raise ValueError(f"materialized batch lacks {name}")
            parts[name].append(_as_numpy(batch[name]))
        if include_logits:
            for name in ("next_event_logits", "next_reached_event_logits"):
                if name not in factual:
                    raise ValueError(f"materialized factual outputs lack {name}")
                parts[name].append(_as_numpy(factual[name]))
            if "aleatoric_uncertainty" in factual:
                parts["aleatoric_uncertainty"].append(
                    _as_numpy(factual["aleatoric_uncertainty"])
                )
            elif "total_uncertainty" in factual:
                parts["total_uncertainty"].append(_as_numpy(factual["total_uncertainty"]))
            else:
                raise ValueError(
                    "materialized factual outputs lack aleatoric uncertainty"
                )
    result = {
        name: np.concatenate(value, axis=0)
        for name, value in parts.items()
        if value
    }
    if include_logits:
        result["owner_fold_id"] = np.full(
            len(result["logical_group"]), fold_id, dtype=np.int64
        )
        if "total_uncertainty" in result:
            uncertainty_provenance = provenance.get("uncertainty_provenance")
            if not isinstance(uncertainty_provenance, Mapping):
                raise ValueError(
                    "materialized total_uncertainty lacks uncertainty_provenance"
                )
            result["uncertainty_provenance"] = dict(uncertainty_provenance)
    return result


def evaluate_materialized_factual_events(
    training_payloads: Mapping[int | str, Mapping[str, Any]],
    holdout_payloads: Mapping[int | str, Mapping[str, Any]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    training = {int(key): value for key, value in training_payloads.items()}
    holdout = {int(key): value for key, value in holdout_payloads.items()}
    if set(training) != set(range(FOLD_COUNT)) or set(holdout) != set(range(FOLD_COUNT)):
        raise ValueError("materialized factual evaluation requires five train/holdout payload pairs")
    train_arrays = {
        fold_id: _records_to_arrays(
            training[fold_id],
            fold_id=fold_id,
            expected_role="outer_training",
            include_logits=False,
        )
        for fold_id in range(FOLD_COUNT)
    }
    holdout_arrays = [
        _records_to_arrays(
            holdout[fold_id],
            fold_id=fold_id,
            expected_role="outer_holdout",
            include_logits=True,
        )
        for fold_id in range(FOLD_COUNT)
    ]
    array_names = {
        name
        for name, value in holdout_arrays[0].items()
        if isinstance(value, np.ndarray)
    }
    if any(
        {
            name for name, value in item.items() if isinstance(value, np.ndarray)
        }
        != array_names
        for item in holdout_arrays
    ):
        raise ValueError("holdout folds disagree on aleatoric uncertainty field semantics")
    combined = {
        name: np.concatenate([item[name] for item in holdout_arrays], axis=0)
        for name in array_names
    }
    if "total_uncertainty" in combined:
        provenance_rows = [item.get("uncertainty_provenance") for item in holdout_arrays]
        if any(row != provenance_rows[0] for row in provenance_rows[1:]):
            raise ValueError("holdout folds disagree on total_uncertainty provenance")
        combined["uncertainty_provenance"] = provenance_rows[0]
    return evaluate_factual_event_arrays(
        combined,
        train_arrays,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_materialization_manifest(
    manifest_path: Path,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("materialization manifest must be a JSON object")
    unsigned = dict(manifest)
    recorded = unsigned.pop("materialization_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise ValueError("materialization manifest SHA mismatch")
    if (
        manifest.get("format") != MATERIALIZATION_FORMAT
        or manifest.get("status") != "complete_development_only"
        or manifest.get("timing_scope")
        != "adaptive_development_only_designed_after_v7_collection_started"
        or manifest.get("prospective_claim_for_v8") is not False
        or manifest.get("fresh_confirmation_data_or_labels_read") is not False
    ):
        raise ValueError("materialization manifest is not the frozen adaptive D250 contract")
    fold_rows = manifest.get("folds")
    if not isinstance(fold_rows, Sequence) or len(fold_rows) != FOLD_COUNT:
        raise ValueError("materialization manifest must contain five folds")
    training: dict[int, Mapping[str, Any]] = {}
    holdout: dict[int, Mapping[str, Any]] = {}
    root = manifest_path.parent
    for expected_fold, row in enumerate(fold_rows):
        if not isinstance(row, Mapping) or int(row.get("outer_fold_id", -1)) != expected_fold:
            raise ValueError("materialization fold ids/order changed")
        for role, target in (("train", training), ("holdout", holdout)):
            artifact = Path(str(row.get(f"{role}_artifact", ""))).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as error:
                raise ValueError("materialized artifact escaped its completed bundle") from error
            if not artifact.is_file() or _sha256_path(artifact) != row.get(
                f"{role}_artifact_sha256"
            ):
                raise ValueError(f"fold {expected_fold} {role} artifact SHA mismatch")
            value = torch.load(artifact, map_location="cpu", weights_only=True)
            if not isinstance(value, Mapping):
                raise ValueError("materialized artifact must contain a mapping")
            if value.get("payload_sha256") != row.get(f"{role}_payload_sha256"):
                raise ValueError(f"fold {expected_fold} {role} payload binding mismatch")
            target[expected_fold] = value
    result = evaluate_materialized_factual_events(
        training,
        holdout,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    result["source_materialization"] = {
        "path": str(manifest_path),
        "file_sha256": _sha256_path(manifest_path),
        "materialization_sha256": recorded,
    }
    result["input_authentication"] = (
        "completed_materialization_manifest_and_all_five_train_holdout_"
        "artifact_and_payload_hashes_verified"
    )
    result["result_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_materialization_manifest(
        args.materialization_manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "fresh50_confirmation_authorized": False,
                "selector_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
