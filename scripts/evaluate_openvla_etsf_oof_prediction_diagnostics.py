#!/usr/bin/env python3
"""Held-out prediction diagnostics for the formal ETSF OOF protocol.

The authorization guard answers whether reranking improved development success.
It does not, by itself, establish that the world-model heads are accurate.  This
module evaluates those heads from the already-held-out rows emitted by each OOF
fold.  It never reads HDF5 collections (and therefore cannot accidentally open
fresh50), does not tune or alter the authorization guard, and treats legacy raw
artifacts without structured predictions as explicitly incomplete evidence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from openvla_etsf_counterfactual_oof import (
    DEPLOYMENT_CANDIDATE_NAMES,
    FOLD_COUNT,
    MEMBER_SEEDS,
    TRAINING_ONLY_EXTRA_CANDIDATES,
    canonical_sha256,
    oof_dimensions,
    validate_oof_folds,
)
from openvla_etsf_prediction_repair import (
    DURATION_RESIDUAL_PROTOCOL,
    OBJECT_REPAIR_PROTOCOL,
    RECOVERY_ADAPTER_PROTOCOL,
    crossfit_duration_residual_contract,
    fit_object_repair_contract,
    object_quality_mask,
    recovery_adapter_training_contract,
)
from train_openvla_etsf_counterfactual import fit_success_temperature


FORMAT = "etsf_oof_heldout_prediction_diagnostics_v1"
STRUCTURED_ROW_FORMAT = "etsf_oof_structured_prediction_row_v1"
EPS = 1e-12
# Frozen, label-independent generic predictive-skill protocol.  It establishes
# skill against simple heldout baselines, not task-safety or deployment
# adequacy.  None of these constants are consumed by the reranking guard.
ADEQUACY_PROTOCOL = "etsf_development_prediction_adequacy_v1"
ADEQUACY_BOOTSTRAP_SEED = 20260903
ADEQUACY_BOOTSTRAP_SAMPLES = 10_000
ADEQUACY_ALPHA = 0.05
MIN_BINARY_CLASS_SUPPORT = 10
MIN_EVENT_PRESENT_CLASSES = 2
MIN_EVENT_CLASS_SUPPORT = 5
MAX_SUCCESS_ECE = 0.10
MIN_PREDICATE_CLASS_SUPPORT = 10
MIN_REGRESSION_GROUPS = 30
# Five of five is the smallest one-sided fold-sign result below 0.05 under a
# 0.5 random-order null (1 / 2**5 = 0.03125).  This is deliberately stricter
# than a noisy point estimate and still remains descriptive development proof.
MIN_UNCERTAINTY_FOLD_WINS = 5
SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT = "etsf_success_head_training_contract_v1"
RECORDED_WEIGHTED_SUCCESS_STATUS = "recorded_outer_fold_weighted_bce"
FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS = (
    "unavailable_factual_training_weight_not_recorded"
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value >= 0.0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def _softmax(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    shifted = value - value.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _logmeanexp(value: np.ndarray, axis: int = 0) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    maximum = value.max(axis=axis, keepdims=True)
    return (
        np.log(np.exp(value - maximum).mean(axis=axis))
        + np.squeeze(maximum, axis=axis)
    )


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels) > 0.5
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware non-interpolated average precision."""

    labels = np.asarray(labels) > 0.5
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    result = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        true_positive += int(labels[start:end].sum())
        false_positive += int((~labels[start:end]).sum())
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(result)


def _cluster_mean_comparison(
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
    group_id: np.ndarray,
) -> dict[str, Any]:
    """Equal-group paired bootstrap CI for model-minus-baseline loss."""

    model_loss = np.asarray(model_loss, dtype=np.float64)
    baseline_loss = np.asarray(baseline_loss, dtype=np.float64)
    group_id = np.asarray(group_id, dtype=np.int64)
    if not (model_loss.shape == baseline_loss.shape == group_id.shape) or not len(group_id):
        raise RuntimeError("clustered comparison arrays are empty or misaligned")
    unique = np.unique(group_id)
    group_delta = np.asarray(
        [
            float((model_loss[group_id == value] - baseline_loss[group_id == value]).mean())
            for value in unique
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(ADEQUACY_BOOTSTRAP_SEED)
    means = np.empty(ADEQUACY_BOOTSTRAP_SAMPLES, dtype=np.float64)
    offset = 0
    while offset < ADEQUACY_BOOTSTRAP_SAMPLES:
        count = min(1000, ADEQUACY_BOOTSTRAP_SAMPLES - offset)
        indices = generator.integers(0, len(group_delta), size=(count, len(group_delta)))
        means[offset : offset + count] = group_delta[indices].mean(1)
        offset += count
    low, high = np.quantile(
        means, [ADEQUACY_ALPHA / 2.0, 1.0 - ADEQUACY_ALPHA / 2.0]
    )
    return {
        "estimand": "equal_logical_group_mean_model_minus_baseline_loss",
        "groups": int(len(unique)),
        "mean_delta": float(group_delta.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
        "model_better_if_upper_ci_below_zero": bool(high < 0.0),
        "bootstrap_samples": ADEQUACY_BOOTSTRAP_SAMPLES,
        "bootstrap_seed": ADEQUACY_BOOTSTRAP_SEED,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2:
        return None
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    if first_rank.std() <= 0.0 or second_rank.std() <= 0.0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = max(len(labels), 1)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index + 1 == bins:
            mask = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            mask = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if mask.any():
            result += float(mask.sum()) / total * abs(
                float(probabilities[mask].mean() - labels[mask].mean())
            )
    return float(result)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
    if not len(labels):
        return {
            "support": 0,
            "positive_support": 0,
            "accuracy_at_0_5": None,
            "brier": None,
            "nll": None,
            "ece_10_equal_width": None,
            "roc_auc": None,
            "pr_auc_average_precision": None,
            "positive_prevalence_random_pr_auc": None,
        }
    predicted = probabilities >= 0.5
    return {
        "support": int(len(labels)),
        "positive_support": int((labels > 0.5).sum()),
        "accuracy_at_0_5": float(np.mean(predicted == (labels > 0.5))),
        "brier": float(np.mean(np.square(probabilities - labels))),
        "nll": float(
            np.mean(
                -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
            )
        ),
        "ece_10_equal_width": _ece(labels, probabilities),
        "roc_auc": _binary_auc(labels, probabilities),
        "pr_auc_average_precision": _average_precision(labels, probabilities),
        "positive_prevalence_random_pr_auc": float((labels > 0.5).mean()),
    }


def _categorical_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, classes: int
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0)
    if probabilities.shape != (len(labels), classes):
        raise RuntimeError("categorical prediction shape mismatch")
    if not len(labels):
        return {
            "support": 0,
            "class_support": [0] * classes,
            "top1_accuracy": None,
            "nll": None,
            "class_recall": [None] * classes,
            "class_precision": [None] * classes,
            "class_f1": [None] * classes,
            "macro_recall_present_classes": None,
            "macro_f1_present_classes": None,
        }
    predicted = probabilities.argmax(-1)
    support = np.bincount(labels, minlength=classes)
    recall: list[float | None] = []
    precision: list[float | None] = []
    f1: list[float | None] = []
    for class_id in range(classes):
        actual = labels == class_id
        selected = predicted == class_id
        true_positive = int((actual & selected).sum())
        class_recall = true_positive / int(actual.sum()) if actual.any() else None
        class_precision = true_positive / int(selected.sum()) if selected.any() else None
        recall.append(class_recall)
        precision.append(class_precision)
        if class_recall is None:
            f1.append(None)
        elif class_precision is None or class_precision + class_recall == 0.0:
            f1.append(0.0)
        else:
            f1.append(
                float(2.0 * class_precision * class_recall / (class_precision + class_recall))
            )
    present_recall = [value for value in recall if value is not None]
    present_f1 = [value for value in f1 if value is not None]
    return {
        "support": int(len(labels)),
        "class_support": support.astype(int).tolist(),
        "top1_accuracy": float(np.mean(predicted == labels)),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean()),
        "class_recall": recall,
        "class_precision": precision,
        "class_f1": f1,
        "macro_recall_present_classes": (
            float(np.mean(present_recall)) if present_recall else None
        ),
        "macro_f1_present_classes": (
            float(np.mean(present_f1)) if present_f1 else None
        ),
    }


def _uncertainty_diagnostic(
    uncertainty: np.ndarray,
    error: np.ndarray,
    *,
    wrong: np.ndarray | None = None,
    fold_id: np.ndarray | None = None,
) -> dict[str, Any]:
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if uncertainty.shape != error.shape or not len(error):
        raise RuntimeError("uncertainty/error arrays must be non-empty and aligned")
    order = np.argsort(uncertainty, kind="mergesort")
    cumulative_risk = np.cumsum(error[order]) / np.arange(1, len(error) + 1)
    result: dict[str, Any] = {
        "support": int(len(error)),
        "spearman_uncertainty_vs_error": _spearman(uncertainty, error),
        "selective_risk_aurc": float(cumulative_risk.mean()),
        "random_order_expected_aurc": float(error.mean()),
        "aurc_improvement_over_random": float(error.mean() - cumulative_risk.mean()),
        "risk_at_lowest_uncertainty_coverage": {
            str(coverage): float(
                error[order[: max(1, int(math.ceil(len(error) * coverage)))]].mean()
            )
            for coverage in (0.5, 0.75, 0.9, 1.0)
        },
    }
    quartile = max(1, len(error) // 4)
    result["mean_error_lowest_uncertainty_quartile"] = float(error[order[:quartile]].mean())
    result["mean_error_highest_uncertainty_quartile"] = float(error[order[-quartile:]].mean())
    if wrong is not None:
        wrong = np.asarray(wrong, dtype=bool)
        if wrong.shape != error.shape:
            raise RuntimeError("wrong/error arrays are misaligned")
        result["error_detection_roc_auc"] = _binary_auc(wrong, uncertainty)
        result["classification_error_support"] = int(wrong.sum())
    if fold_id is not None:
        fold_id = np.asarray(fold_id, dtype=np.int64)
        if fold_id.shape != error.shape:
            raise RuntimeError("fold/error arrays are misaligned")
        per_fold: dict[str, Any] = {}
        fold_wins = 0
        for value in range(FOLD_COUNT):
            fold = fold_id == value
            if not fold.any():
                per_fold[str(value)] = {"support": 0, "aurc_improvement_over_random": None}
                continue
            fold_order = np.argsort(uncertainty[fold], kind="mergesort")
            fold_error = error[fold][fold_order]
            fold_aurc = float(
                (np.cumsum(fold_error) / np.arange(1, len(fold_error) + 1)).mean()
            )
            improvement = float(error[fold].mean() - fold_aurc)
            fold_wins += int(improvement > 0.0)
            per_fold[str(value)] = {
                "support": int(fold.sum()),
                "selective_risk_aurc": fold_aurc,
                "random_order_expected_aurc": float(error[fold].mean()),
                "aurc_improvement_over_random": improvement,
            }
        result["per_fold_heldout"] = per_fold
        result["folds_with_aurc_better_than_random"] = fold_wins
    return result


def _fold_map(manifest: Mapping[str, Any]) -> dict[str, int]:
    validate_oof_folds(manifest, manifest["development_groups"])
    return {
        str(key): int(fold["fold_id"])
        for fold in manifest["folds"]
        for key in fold["oof_holdout_groups"]
    }


def _expected_success_training_group_sha256(
    manifest: Mapping[str, Any], fold_id: int
) -> str:
    fold = next(
        (row for row in manifest["folds"] if int(row["fold_id"]) == fold_id),
        None,
    )
    if not isinstance(fold, Mapping):
        raise RuntimeError("success training contract owner fold is absent")
    return canonical_sha256(sorted(map(str, fold["training_groups"])))


def _validate_success_head_training_contract(
    value: Any,
    *,
    owner_fold_id: int,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Validate recorded head provenance; absence remains explicit fail-closed."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError("success head training contract must be a mapping")
    contract = dict(value)
    if (
        contract.get("format") != SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT
        or int(contract.get("owner_fold_id", -1)) != owner_fold_id
    ):
        raise RuntimeError("success head training contract owner/format mismatch")
    status = str(contract.get("status", ""))
    if status == FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS:
        if (
            contract.get("success_head_updated_on_owner_training_groups") is not False
            or contract.get("factual_core_bit_exact") is not True
            or contract.get("positive_weight") not in (None, "")
        ):
            raise RuntimeError("frozen factual success provenance is inconsistent")
        return contract
    if status != RECORDED_WEIGHTED_SUCCESS_STATUS:
        raise RuntimeError("unsupported success head training contract status")
    weight = contract.get("positive_weight")
    try:
        weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("recorded success positive weight is invalid") from exc
    expected_training_sha = _expected_success_training_group_sha256(
        manifest, owner_fold_id
    )
    if (
        not math.isfinite(weight)
        or weight <= 0.0
        or contract.get("positive_weight_source")
        != "checkpoint_loss_metadata_recorded_before_training"
        or contract.get("success_head_updated_on_owner_training_groups") is not True
        or contract.get("owner_oof_holdout_excluded_from_training") is not True
        or str(contract.get("owner_training_groups_sha256", ""))
        != expected_training_sha
    ):
        raise RuntimeError("recorded success head training provenance is invalid")
    contract["positive_weight"] = weight
    return contract


def _validate_rows(
    raw_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_groups, _, _ = oof_dimensions(manifest)
    owner = _fold_map(manifest)
    if len(raw_rows) != expected_groups:
        raise RuntimeError("prediction diagnostics require one row per frozen OOF group")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw_rows:
        row = dict(value)
        key = str(row.get("logical_key", ""))
        if key not in owner or key in seen or int(row.get("fold_id", -1)) != owner.get(key):
            raise RuntimeError("diagnostic row is not a unique owner-fold heldout prediction")
        logits = np.asarray(row.get("member_success_logits"), dtype=np.float64)
        labels = np.asarray(row.get("success"), dtype=np.float64)
        aleatoric = np.asarray(row.get("member_aleatoric"), dtype=np.float64)
        if (
            logits.ndim != 2
            or logits.shape[0] != len(MEMBER_SEEDS)
            or labels.shape != logits.shape[1:]
            or aleatoric.shape != logits.shape
        ):
            raise RuntimeError("success prediction diagnostic arrays are misaligned")
        if not np.isfinite(logits).all() or not np.isfinite(labels).all() or not np.isfinite(aleatoric).all():
            raise RuntimeError("prediction diagnostics refuse non-finite success data")
        if np.any((labels != 0.0) & (labels != 1.0)) or np.any(aleatoric < 0.0):
            raise RuntimeError("success labels/uncertainties violate their contracts")
        names = row.get("candidate_names")
        if (
            not isinstance(names, Sequence)
            or isinstance(names, (str, bytes))
            or len(names) != len(labels)
            or len(set(map(str, names))) != len(labels)
        ):
            raise RuntimeError("success candidate names are misaligned or duplicated")
        row["candidate_names"] = list(map(str, names))
        row["member_success_logits"] = logits
        row["success"] = labels
        row["member_aleatoric"] = aleatoric
        row["success_head_training_contract"] = (
            _validate_success_head_training_contract(
                row.get("success_head_training_contract"),
                owner_fold_id=int(row["fold_id"]),
                manifest=manifest,
            )
        )
        seen.add(key)
        rows.append(row)
    if seen != set(owner):
        raise RuntimeError("prediction diagnostics do not cover every heldout group")
    return sorted(rows, key=lambda item: str(item["logical_key"]))


def _deployment_success_rows(
    rows: Sequence[Mapping[str, Any]], *, deployment_only: bool
) -> list[dict[str, Any]]:
    scoped: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        names = tuple(map(str, row["candidate_names"]))
        if names[: len(DEPLOYMENT_CANDIDATE_NAMES)] != DEPLOYMENT_CANDIDATE_NAMES:
            raise RuntimeError("diagnostic deployment candidate names/order changed")
        extras = names[len(DEPLOYMENT_CANDIDATE_NAMES) :]
        if extras not in ((), TRAINING_ONLY_EXTRA_CANDIDATES):
            raise RuntimeError("diagnostic contains an unregistered candidate schedule")
        keep = (
            np.arange(len(DEPLOYMENT_CANDIDATE_NAMES), dtype=np.int64)
            if deployment_only
            else np.arange(len(names), dtype=np.int64)
        )
        row["member_success_logits"] = row["member_success_logits"][:, keep]
        row["member_aleatoric"] = row["member_aleatoric"][:, keep]
        row["success"] = row["success"][keep]
        row["candidate_names"] = [names[index] for index in keep]
        scoped.append(row)
    return scoped


def _success_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    logits = np.concatenate([row["member_success_logits"] for row in rows], axis=1)
    labels = np.concatenate([row["success"] for row in rows])
    folds = np.concatenate(
        [np.full(len(row["success"]), int(row["fold_id"]), dtype=np.int64) for row in rows]
    )
    group_id = np.concatenate(
        [np.full(len(row["success"]), index, dtype=np.int64) for index, row in enumerate(rows)]
    )
    explicit_sources = {
        str(row["success_prediction_source"])
        for row in rows
        if row.get("success_prediction_source") not in (None, "")
    }
    repeated_factual_axis = all(
        row.get("diagnostic_member_axis")
        == "single_frozen_factual_prediction_repeated_for_legacy_three_member_axis"
        for row in rows
    )
    if len(explicit_sources) > 1:
        raise RuntimeError("success prediction sources are mixed")
    if explicit_sources:
        prediction_source = next(iter(explicit_sources))
    elif repeated_factual_axis:
        prediction_source = "frozen_factual_success_logit_bit_exact_no_rank_residual"
    else:
        prediction_source = "legacy_member_success_logits_training_source_not_recorded"
    contract_by_fold: dict[int, Mapping[str, Any] | None] = {}
    for fold_id in range(FOLD_COUNT):
        fold_contracts = [
            row.get("success_head_training_contract")
            for row in rows
            if int(row["fold_id"]) == fold_id
        ]
        digests = {
            canonical_sha256(value)
            for value in fold_contracts
            if isinstance(value, Mapping)
        }
        missing = sum(value is None for value in fold_contracts)
        if len(digests) > 1 or (missing and missing != len(fold_contracts)):
            raise RuntimeError("success head training provenance differs within owner fold")
        contract_by_fold[fold_id] = (
            next(
                (value for value in fold_contracts if isinstance(value, Mapping)),
                None,
            )
        )
    member_probabilities = _sigmoid(logits)
    uncalibrated = member_probabilities.mean(0)
    calibrated_members = np.empty_like(member_probabilities)
    calibrated = np.empty_like(uncalibrated)
    baseline = np.empty_like(uncalibrated)
    temperatures: dict[str, float] = {}
    per_fold: dict[str, Any] = {}
    for fold_id in range(FOLD_COUNT):
        heldout = folds == fold_id
        calibration_train = ~heldout
        fit = fit_success_temperature(logits[:, calibration_train], labels[calibration_train])
        temperature = float(fit["temperature"])
        temperatures[str(fold_id)] = temperature
        calibrated_members[:, heldout] = _sigmoid(
            logits[:, heldout] / temperature
        )
        calibrated[heldout] = calibrated_members[:, heldout].mean(0)
        prevalence = float(labels[calibration_train].mean())
        baseline[heldout] = prevalence
        per_fold[str(fold_id)] = {
            "heldout_only": True,
            "crossfit_temperature_trained_on_other_folds": temperature,
            "valid_for_strict_probability_adequacy": False,
            "metrics": _binary_metrics(labels[heldout], calibrated[heldout]),
            "other_fold_prevalence_baseline": _binary_metrics(
                labels[heldout], baseline[heldout]
            ),
        }
    # Strict uncertainty and pair ordering use the emitted predictions without
    # the invalid overlapping-outer-fold temperature transform.
    uncertainty = member_probabilities.std(0) + np.concatenate(
        [row["member_aleatoric"].mean(0) for row in rows]
    )
    absolute_error = np.abs(uncalibrated - labels)
    wrong = (uncalibrated >= 0.5) != (labels > 0.5)
    clipped = np.clip(calibrated, EPS, 1.0 - EPS)
    clipped_baseline = np.clip(baseline, EPS, 1.0 - EPS)
    model_nll = -(
        labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
    )
    baseline_nll = -(
        labels * np.log(clipped_baseline)
        + (1.0 - labels) * np.log(1.0 - clipped_baseline)
    )
    pair_correctness: list[float] = []
    pair_groups: list[int] = []
    per_group_pair_accuracy: list[float] = []
    offset = 0
    for index, row in enumerate(rows):
        count = len(row["success"])
        group_label = labels[offset : offset + count]
        group_probability = logits.mean(0)[offset : offset + count]
        group_pairs: list[float] = []
        for first in range(count):
            for second in range(first + 1, count):
                if group_label[first] == group_label[second]:
                    continue
                signed_score = (group_probability[first] - group_probability[second]) * (
                    group_label[first] - group_label[second]
                )
                correctness = 1.0 if signed_score > 0.0 else 0.5 if signed_score == 0.0 else 0.0
                group_pairs.append(correctness)
                pair_correctness.append(correctness)
                pair_groups.append(index)
        if group_pairs:
            per_group_pair_accuracy.append(float(np.mean(group_pairs)))
        offset += count
    if pair_correctness:
        pair_correctness_array = np.asarray(pair_correctness, dtype=np.float64)
        pair_group_array = np.asarray(pair_groups, dtype=np.int64)
        pair_comparison: Mapping[str, Any] | None = _cluster_mean_comparison(
            1.0 - pair_correctness_array,
            np.full_like(pair_correctness_array, 0.5),
            pair_group_array,
        )
        pair_accuracy: float | None = float(pair_correctness_array.mean())
        group_pair_accuracy: float | None = float(np.mean(per_group_pair_accuracy))
    else:
        pair_comparison = None
        pair_accuracy = group_pair_accuracy = None
    recorded_contracts = all(
        isinstance(contract_by_fold[fold_id], Mapping)
        and contract_by_fold[fold_id].get("status")
        == RECORDED_WEIGHTED_SUCCESS_STATUS
        for fold_id in range(FOLD_COUNT)
    )
    frozen_weight_unavailable = all(
        isinstance(contract_by_fold[fold_id], Mapping)
        and contract_by_fold[fold_id].get("status")
        == FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS
        for fold_id in range(FOLD_COUNT)
    )
    if recorded_contracts:
        probability_logits = logits.copy()
        fold_provenance: dict[str, Any] = {}
        for fold_id in range(FOLD_COUNT):
            contract = contract_by_fold[fold_id]
            assert isinstance(contract, Mapping)
            weight = float(contract["positive_weight"])
            probability_logits[:, folds == fold_id] -= math.log(weight)
            fold_provenance[str(fold_id)] = dict(contract)
        strict_member_probability = _sigmoid(probability_logits)
        strict_probability = strict_member_probability.mean(0)
        strict_clipped = np.clip(strict_probability, EPS, 1.0 - EPS)
        strict_nll = -(
            labels * np.log(strict_clipped)
            + (1.0 - labels) * np.log(1.0 - strict_clipped)
        )
        strict_assessment: Mapping[str, Any] = {
            "status": "evaluable_recorded_training_weight_prior_shift_only",
            "usable_for_strict_adequacy": True,
            "calibration": "analytic_logit_minus_log_recorded_training_pos_weight",
            "bias_temperature_fit": "none",
            "target_fold_labels_used_for_calibration": False,
            "fold_training_contracts": fold_provenance,
            "metrics": _binary_metrics(labels, strict_probability),
            "paired_skill_vs_other_fold_prevalence": {
                "brier": _cluster_mean_comparison(
                    np.square(strict_probability - labels),
                    np.square(baseline - labels),
                    group_id,
                ),
                "nll": _cluster_mean_comparison(
                    strict_nll, baseline_nll, group_id
                ),
            },
        }
    else:
        if frozen_weight_unavailable or prediction_source == (
            "frozen_factual_success_logit_bit_exact_no_rank_residual"
        ):
            unavailable_status = FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS
        elif any(contract_by_fold.values()):
            unavailable_status = "unavailable_incomplete_success_head_training_contract"
        else:
            unavailable_status = "unavailable_success_head_training_contract_not_recorded"
        strict_assessment = {
            "status": unavailable_status,
            "usable_for_strict_adequacy": False,
            "calibration": "none",
            "bias_temperature_fit": "none",
            "uncalibrated_probability_reported_descriptively": True,
            "uncalibrated_metrics": _binary_metrics(labels, uncalibrated),
            "reason": (
                "analytic prior shift requires the positive weight actually used "
                "to train this success head; branch-fold prevalence is not a substitute"
            ),
        }
    fixed_factual_source = prediction_source == (
        "frozen_factual_success_logit_bit_exact_no_rank_residual"
    )
    legacy_overlap_audit = {
        "valid_for_strict_adequacy": False,
        "outer_target_label_indirect_model_training_overlap": (
            False if fixed_factual_source else "not_excluded"
        ),
        "factual_historical_training_overlap": (
            "not_excluded_for_legacy_old100"
            if fixed_factual_source
            else "training_source_not_fully_recorded"
        ),
        "reason": (
            "fixed v6 factual logits avoid outer-model crossfold leakage, but the "
            "factual training weight and old100 historical overlap are not recorded"
            if fixed_factual_source
            else "other outer-fold models may have trained on the target fold"
        ),
    }
    return {
        "label": "terminal_branch_success",
        "success_prediction_source": prediction_source,
        "calibration_protocol": (
            "legacy_five_fold_crossfit_temperature_descriptive_not_strict"
        ),
        "crossfit_calibrated_valid_for_strict_adequacy": False,
        "crossfit_calibration_overlap_audit": legacy_overlap_audit,
        "crossfit_temperatures": temperatures,
        "crossfit_calibrated": _binary_metrics(labels, calibrated),
        "uncalibrated_ensemble": _binary_metrics(labels, uncalibrated),
        "other_fold_prevalence_baseline": _binary_metrics(labels, baseline),
        "paired_skill_vs_other_fold_prevalence": {
            "brier": _cluster_mean_comparison(
                np.square(calibrated - labels),
                np.square(baseline - labels),
                group_id,
            ),
            "nll": _cluster_mean_comparison(
                model_nll, baseline_nll, group_id
            ),
        },
        "strict_probability_assessment": strict_assessment,
        "within_group_success_pair_ranking": {
            "comparable_pairs": int(len(pair_correctness)),
            "groups_with_comparable_pairs": int(len(per_group_pair_accuracy)),
            "pair_weighted_accuracy_ties_half": pair_accuracy,
            "equal_group_accuracy_ties_half": group_pair_accuracy,
            "paired_error_vs_random_0_5": pair_comparison,
        },
        "per_fold": per_fold,
        "uncertainty_error_relation": _uncertainty_diagnostic(
            uncertainty, absolute_error, wrong=wrong, fold_id=folds
        ),
    }


def _structured_block(row: Mapping[str, Any]) -> dict[str, Any]:
    block = row.get("structured_predictions")
    if not isinstance(block, Mapping) or block.get("format") != STRUCTURED_ROW_FORMAT:
        raise RuntimeError("structured OOF prediction row is absent or unsupported")
    result = dict(block)
    sample_names = result.get("sample_names")
    predicate_names = result.get("predicate_names")
    masks = {
        name: np.asarray(result.get(name), dtype=bool)
        for name in ("terminal_mask", "structured_mask", "dense_mask", "duration_observed")
    }
    sample_count = len(masks["dense_mask"])
    if sample_count <= 0 or any(value.shape != (sample_count,) for value in masks.values()):
        raise RuntimeError("structured OOF masks are empty or misaligned")
    if (
        not isinstance(sample_names, Sequence)
        or isinstance(sample_names, (str, bytes))
        or len(sample_names) != sample_count
    ):
        raise RuntimeError("structured OOF sample names are missing or misaligned")
    vectors = {
        "current_event_id": np.asarray(result.get("current_event_id"), dtype=np.int64),
        "clock_event_id": np.asarray(result.get("clock_event_id"), dtype=np.int64),
        "next_event_id": np.asarray(result.get("next_event_id"), dtype=np.int64),
        "next_reached_event_id": np.asarray(result.get("next_reached_event_id"), dtype=np.int64),
        "body_id": np.asarray(result.get("body_id"), dtype=np.int64),
        "policy_id": np.asarray(result.get("policy_id"), dtype=np.int64),
        "duration": np.asarray(result.get("duration"), dtype=np.float64),
        "success": np.asarray(result.get("success"), dtype=np.float64),
        "outcome_id": np.asarray(result.get("outcome_id"), dtype=np.int64),
        "trajectory_regress": np.asarray(result.get("trajectory_regress"), dtype=bool),
        "trajectory_recovery": np.asarray(result.get("trajectory_recovery"), dtype=bool),
    }
    if any(value.shape != (sample_count,) for value in vectors.values()):
        raise RuntimeError("structured OOF labels are misaligned")
    member_event = np.asarray(result.get("member_next_event_logits"), dtype=np.float64)
    member_duration_mean = np.asarray(result.get("member_duration_log_mean"), dtype=np.float64)
    member_duration_scale = np.asarray(result.get("member_duration_log_scale"), dtype=np.float64)
    member_reach = np.asarray(result.get("member_reach_logit"), dtype=np.float64)
    member_object_mean = np.asarray(result.get("member_object_delta_mean"), dtype=np.float64)
    member_object_scale = np.asarray(result.get("member_object_delta_log_scale"), dtype=np.float64)
    member_outcome = np.asarray(result.get("member_outcome_logits"), dtype=np.float64)
    target_object = np.asarray(result.get("object_delta"), dtype=np.float64)
    if (
        member_event.ndim != 3
        or member_event.shape[0] != len(MEMBER_SEEDS)
        or member_event.shape[1] != sample_count
        or member_event.shape[2] < 2
    ):
        raise RuntimeError("structured next-event predictions are misaligned")
    member_count = member_event.shape[0]
    if (
        member_duration_mean.shape != (member_count, sample_count)
        or member_duration_scale.shape != (member_count, sample_count)
        or member_reach.shape != (member_count, sample_count)
        or member_object_mean.ndim != 3
        or member_object_mean.shape[:2] != (member_count, sample_count)
        or member_object_scale.shape != member_object_mean.shape
        or target_object.shape != member_object_mean.shape[1:]
        or member_outcome.ndim != 3
        or member_outcome.shape[:2] != (member_count, sample_count)
    ):
        raise RuntimeError("structured probabilistic head predictions are misaligned")
    predicate_logits = np.asarray(result.get("member_post_predicate_logits"), dtype=np.float64)
    post_predicates = np.asarray(result.get("post_predicates"), dtype=np.float64)
    reached_logits = np.asarray(result.get("member_next_reached_event_logits"), dtype=np.float64)
    if (
        predicate_logits.ndim != 3
        or predicate_logits.shape[:2] != (member_count, sample_count)
        or post_predicates.shape != predicate_logits.shape[1:]
        or reached_logits.ndim != 3
        or reached_logits.shape[:2] != (member_count, sample_count)
        or reached_logits.shape[2] != member_event.shape[2]
        or member_outcome.shape[2] < 2
    ):
        raise RuntimeError("structured event subhead predictions are misaligned")
    if (
        not isinstance(predicate_names, Sequence)
        or isinstance(predicate_names, (str, bytes))
        or len(predicate_names) != predicate_logits.shape[-1]
        or len(set(map(str, predicate_names))) != len(predicate_names)
    ):
        raise RuntimeError("structured predicate names are missing or misaligned")
    arrays = [
        *masks.values(),
        *vectors.values(),
        member_event,
        member_duration_mean,
        member_duration_scale,
        member_reach,
        member_object_mean,
        member_object_scale,
        member_outcome,
        target_object,
        predicate_logits,
        post_predicates,
        reached_logits,
    ]
    if not all(np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays):
        raise RuntimeError("structured OOF predictions contain non-finite values")
    event_classes = member_event.shape[-1]
    outcome_classes = member_outcome.shape[-1]
    if (
        np.any(vectors["current_event_id"] < 0)
        or np.any(vectors["current_event_id"] >= event_classes)
        or np.any(vectors["clock_event_id"] < 0)
        or np.any(vectors["clock_event_id"] >= event_classes)
        or np.any(vectors["next_event_id"] < 0)
        or np.any(vectors["next_event_id"] >= event_classes)
        or np.any(vectors["next_reached_event_id"] < 0)
        or np.any(vectors["next_reached_event_id"] >= event_classes)
        or np.any(vectors["outcome_id"] < 0)
        or np.any(vectors["outcome_id"] >= outcome_classes)
        or np.any(vectors["duration"] < 0.0)
        or np.any(vectors["body_id"] < 0)
        or np.any(vectors["policy_id"] < 0)
        or np.any((vectors["success"] != 0.0) & (vectors["success"] != 1.0))
        or np.any((post_predicates != 0.0) & (post_predicates != 1.0))
        or np.any(masks["duration_observed"] & ~masks["dense_mask"])
        or np.any(masks["terminal_mask"] & ~masks["dense_mask"])
        or np.any(masks["terminal_mask"] & ~masks["structured_mask"])
    ):
        raise RuntimeError("structured OOF ids, labels, durations, or masks are invalid")
    return {
        **masks,
        **vectors,
        "member_next_event_logits": member_event,
        "member_duration_log_mean": member_duration_mean,
        "member_duration_log_scale": member_duration_scale,
        "member_reach_logit": member_reach,
        "member_object_delta_mean": member_object_mean,
        "member_object_delta_log_scale": member_object_scale,
        "member_outcome_logits": member_outcome,
        "object_delta": target_object,
        "member_post_predicate_logits": predicate_logits,
        "post_predicates": post_predicates,
        "member_next_reached_event_logits": reached_logits,
        "recovery_supervised": bool(result.get("recovery_supervised", False)),
        "sample_names": list(map(str, sample_names)),
        "predicate_names": list(map(str, predicate_names)),
    }


def _concatenate_structured(
    rows: Sequence[Mapping[str, Any]], blocks: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    member_keys = {
        "member_next_event_logits",
        "member_duration_log_mean",
        "member_duration_log_scale",
        "member_reach_logit",
        "member_object_delta_mean",
        "member_object_delta_log_scale",
        "member_outcome_logits",
        "member_post_predicate_logits",
        "member_next_reached_event_logits",
    }
    metadata_keys = {"recovery_supervised", "sample_names", "predicate_names"}
    keys = [key for key in blocks[0] if key not in metadata_keys]
    for key in keys:
        axis = 1 if key in member_keys else 0
        result[key] = np.concatenate([np.asarray(block[key]) for block in blocks], axis=axis)
    result["fold_id"] = np.concatenate(
        [
            np.full(len(block["dense_mask"]), int(row["fold_id"]), dtype=np.int64)
            for row, block in zip(rows, blocks)
        ]
    )
    result["logical_group_id"] = np.concatenate(
        [
            np.full(len(block["dense_mask"]), index, dtype=np.int64)
            for index, block in enumerate(blocks)
        ]
    )
    result["predicate_names"] = np.asarray(blocks[0]["predicate_names"], dtype=object)
    return result


def _crossfit_event_prior(labels: np.ndarray, folds: np.ndarray, classes: int) -> np.ndarray:
    probabilities = np.empty((len(labels), classes), dtype=np.float64)
    for fold_id in range(FOLD_COUNT):
        heldout = folds == fold_id
        counts = np.bincount(labels[~heldout], minlength=classes).astype(np.float64) + 1.0
        probabilities[heldout] = counts / counts.sum()
    return probabilities


def _event_diagnostics(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    mask = values["structured_mask"]
    labels = values["next_event_id"][mask]
    current = values["current_event_id"][mask]
    folds = values["fold_id"][mask]
    groups = values["logical_group_id"][mask]
    member_probability = _softmax(values["member_next_event_logits"][:, mask])
    probability = member_probability.mean(0)
    classes = probability.shape[-1]
    prior = _crossfit_event_prior(labels, folds, classes)
    persistence = np.eye(classes, dtype=np.float64)[np.clip(current, 0, classes - 1)]
    predicted = probability.argmax(-1)
    entropy = -np.sum(probability * np.log(np.clip(probability, EPS, 1.0)), axis=-1)
    entropy /= max(math.log(classes), EPS)
    wrong = predicted != labels
    row = np.arange(len(labels))
    model_nll = -np.log(np.clip(probability[row, labels], EPS, 1.0))
    persistence_nll = -np.log(np.clip(persistence[row, labels], EPS, 1.0))
    prior_nll = -np.log(np.clip(prior[row, labels], EPS, 1.0))
    persistence_wrong = persistence.argmax(-1) != labels
    prior_wrong = prior.argmax(-1) != labels
    per_fold = {
        str(fold_id): _categorical_metrics(
            labels[folds == fold_id], probability[folds == fold_id], classes=classes
        )
        for fold_id in range(FOLD_COUNT)
    }
    return {
        "target": "next_event_id",
        "ensemble": _categorical_metrics(labels, probability, classes=classes),
        "current_event_persistence_baseline": _categorical_metrics(
            labels, persistence, classes=classes
        ),
        "other_fold_smoothed_prior_baseline": _categorical_metrics(
            labels, prior, classes=classes
        ),
        "paired_skill_vs_baselines": {
            "nll_vs_current_event_persistence": _cluster_mean_comparison(
                model_nll, persistence_nll, groups
            ),
            "nll_vs_other_fold_prior": _cluster_mean_comparison(
                model_nll, prior_nll, groups
            ),
            "top1_error_vs_current_event_persistence": _cluster_mean_comparison(
                wrong.astype(float), persistence_wrong.astype(float), groups
            ),
            "top1_error_vs_other_fold_prior": _cluster_mean_comparison(
                wrong.astype(float), prior_wrong.astype(float), groups
            ),
        },
        "per_fold_heldout": per_fold,
        "uncertainty_error_relation": _uncertainty_diagnostic(
            entropy, model_nll, wrong=wrong, fold_id=folds
        ),
    }


def _crossfit_event_body_median(
    target_log_duration: np.ndarray,
    observed: np.ndarray,
    folds: np.ndarray,
    event_id: np.ndarray,
    body_id: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Other-fold observed median with exact-key then conservative fallbacks."""

    prediction = np.empty(len(target_log_duration), dtype=np.float64)
    fallback_counts = {"event_body": 0, "event": 0, "body": 0, "global": 0}
    for fold_id in range(FOLD_COUNT):
        heldout = folds == fold_id
        training = (~heldout) & observed
        if not training.any():
            raise RuntimeError("duration crossfit baseline has no observed training support")
        global_median = float(np.median(target_log_duration[training]))
        for index in np.flatnonzero(heldout):
            exact = training & (event_id == event_id[index]) & (body_id == body_id[index])
            event = training & (event_id == event_id[index])
            body = training & (body_id == body_id[index])
            if exact.any():
                selected = exact
                fallback_counts["event_body"] += 1
            elif event.any():
                selected = event
                fallback_counts["event"] += 1
            elif body.any():
                selected = body
                fallback_counts["body"] += 1
            else:
                prediction[index] = global_median
                fallback_counts["global"] += 1
                continue
            prediction[index] = float(np.median(target_log_duration[selected]))
    return prediction, fallback_counts


def _duration_diagnostics(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    mask = values["dense_mask"]
    observed = values["duration_observed"][mask]
    duration = values["duration"][mask]
    folds = values["fold_id"][mask]
    groups = values["logical_group_id"][mask]
    clock_event = values["clock_event_id"][mask]
    body = values["body_id"][mask]
    target = np.log1p(np.maximum(duration, 0.0))
    baseline_log_duration, baseline_fallback = _crossfit_event_body_median(
        target, observed, folds, clock_event, body
    )
    repair_contract = crossfit_duration_residual_contract(
        duration, observed, folds, clock_event, body
    )
    if not np.allclose(
        repair_contract["baseline_log1p_duration"],
        baseline_log_duration,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("duration repair baseline differs from audited crossfit baseline")
    baseline_duration = np.expm1(baseline_log_duration)
    means = values["member_duration_log_mean"][:, mask]
    scales = np.exp(np.clip(values["member_duration_log_scale"][:, mask], -5.0, 3.0))
    z = (target[None] - means) / np.maximum(scales, 1e-4)
    observed_log_likelihood = -0.5 * z**2 - np.log(np.maximum(scales, 1e-4)) - 0.5 * math.log(2.0 * math.pi)
    censored_log_survival = torch.special.log_ndtr(
        -torch.as_tensor(z, dtype=torch.float64)
    ).numpy()
    mixture_observed_nll = -_logmeanexp(observed_log_likelihood, axis=0)
    mixture_censored_nll = -_logmeanexp(censored_log_survival, axis=0)
    predicted_duration = np.expm1(np.clip(means, 0.0, 12.0)).mean(0)
    mixture_log_variance = np.maximum(
        np.mean(scales**2 + means**2, axis=0) - np.mean(means, axis=0) ** 2,
        0.0,
    )
    uncertainty = np.sqrt(mixture_log_variance)
    reach_probability = _sigmoid(values["member_reach_logit"][:, mask]).mean(0)
    result: dict[str, Any] = {
        "support": int(len(duration)),
        "observed_support": int(observed.sum()),
        "right_censored_support": int((~observed).sum()),
        "reach_observed_classifier": _binary_metrics(observed.astype(float), reach_probability),
        "likelihood_contract": (
            "ensemble_mixture_normal_likelihood_on_log1p_duration_matching_training_loss; "
            "right_censored_items_use_survival_probability"
        ),
        "event_body_crossfit_median_baseline": {
            "fit_labels": "observed_durations_from_other_four_folds_only",
            "key": "clock_event_id_x_body_id",
            "fallback_order": ["event", "body", "global"],
            "fallback_usage": baseline_fallback,
        },
        "prospective_next_oof_training_repair": {
            "protocol": DURATION_RESIDUAL_PROTOCOL,
            "status": "development_repair_contract_not_current_preregistered_adequacy",
            "excluded_from_current_prediction_adequacy": True,
            "target_fold_labels_used_for_fit": repair_contract[
                "target_fold_labels_used_for_fit"
            ],
            "duration_supervision": "observed_only_conditional_on_reach",
            "right_censored_supervision": "reach_classifier_only",
            "location_target": "log1p_duration_minus_training_fold_event_body_median",
            "location_likelihood": "laplace_matches_median_and_log1p_mae",
            "observed_residual_quantiles": [
                float(value)
                for value in np.quantile(
                    repair_contract["residual_target"][observed],
                    [0.0, 0.1, 0.5, 0.9, 1.0],
                )
            ],
            "fallback_usage": {
                name: int((repair_contract["fallback_source"] == name).sum())
                for name in ("event_body", "event", "body", "global")
            },
        },
    }
    per_fold: dict[str, Any] = {}
    for fold_id in range(FOLD_COUNT):
        fold = folds == fold_id
        fold_observed = fold & observed
        fold_censored = fold & ~observed
        per_fold[str(fold_id)] = {
            "support": int(fold.sum()),
            "observed_support": int(fold_observed.sum()),
            "right_censored_support": int(fold_censored.sum()),
            "reach_observed_classifier": _binary_metrics(
                observed[fold].astype(float), reach_probability[fold]
            ),
            "observed_mixture_nll_log1p_scale": (
                float(mixture_observed_nll[fold_observed].mean())
                if fold_observed.any()
                else None
            ),
            "observed_model_mae_log1p_steps": (
                float(
                    np.abs(
                        np.log1p(np.maximum(predicted_duration[fold_observed], 0.0))
                        - target[fold_observed]
                    ).mean()
                )
                if fold_observed.any()
                else None
            ),
            "observed_event_body_crossfit_median_mae_log1p_steps": (
                float(
                    np.abs(
                        baseline_log_duration[fold_observed]
                        - target[fold_observed]
                    ).mean()
                )
                if fold_observed.any()
                else None
            ),
            "right_censored_mixture_survival_nll": (
                float(mixture_censored_nll[fold_censored].mean())
                if fold_censored.any()
                else None
            ),
        }
    result["per_fold_heldout"] = per_fold
    if observed.any():
        error_steps = np.abs(predicted_duration[observed] - duration[observed])
        error_log = np.abs(np.log1p(np.maximum(predicted_duration[observed], 0.0)) - target[observed])
        baseline_error_steps = np.abs(
            baseline_duration[observed] - duration[observed]
        )
        baseline_error_log = np.abs(
            baseline_log_duration[observed] - target[observed]
        )
        result["observed"] = {
            "mixture_nll_log1p_scale": float(mixture_observed_nll[observed].mean()),
            "mae_steps": float(error_steps.mean()),
            "rmse_steps": float(np.sqrt(np.mean((predicted_duration[observed] - duration[observed]) ** 2))),
            "mae_log1p_steps": float(error_log.mean()),
            "event_body_crossfit_median_baseline": {
                "mae_steps": float(baseline_error_steps.mean()),
                "rmse_steps": float(
                    np.sqrt(
                        np.mean(
                            (baseline_duration[observed] - duration[observed]) ** 2
                        )
                    )
                ),
                "mae_log1p_steps": float(baseline_error_log.mean()),
            },
            "paired_log1p_mae_skill_vs_event_body_crossfit_median": (
                _cluster_mean_comparison(
                    error_log,
                    baseline_error_log,
                    groups[observed],
                )
            ),
            "uncertainty_error_relation": _uncertainty_diagnostic(
                uncertainty[observed], error_log, fold_id=folds[observed]
            ),
        }
    else:
        result["observed"] = {"status": "no_observed_duration_labels"}
    if (~observed).any():
        survival = np.exp(_logmeanexp(censored_log_survival[:, ~observed], axis=0))
        result["right_censored"] = {
            "mixture_survival_nll": float(mixture_censored_nll[~observed].mean()),
            "mean_predicted_survival_at_censor_bound": float(survival.mean()),
            "median_proxy_below_censor_bound_rate": float(
                np.mean(predicted_duration[~observed] < duration[~observed])
            ),
        }
    else:
        result["right_censored"] = {"status": "no_right_censored_duration_labels"}
    return result


def _object_diagnostics(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    mask = values["dense_mask"]
    target = values["object_delta"][mask]
    folds = values["fold_id"][mask]
    groups = values["logical_group_id"][mask]
    member_mean = values["member_object_delta_mean"][:, mask]
    member_scale = np.exp(
        np.clip(values["member_object_delta_log_scale"][:, mask], -20.0, 20.0)
    )
    prediction = member_mean.mean(0)
    error = prediction - target
    absolute = np.abs(error)
    log_likelihood = (
        -0.5 * ((target[None] - member_mean) / np.maximum(member_scale, 1e-8)) ** 2
        - np.log(np.maximum(member_scale, 1e-8))
        - 0.5 * math.log(2.0 * math.pi)
    )
    nll = -_logmeanexp(log_likelihood, axis=0)
    mixture_variance = np.maximum(
        np.mean(member_scale**2 + member_mean**2, axis=0) - prediction[None].squeeze(0) ** 2,
        0.0,
    )
    uncertainty = np.sqrt(mixture_variance).mean(-1)
    sample_error = absolute.mean(-1)
    repair_baseline = np.empty_like(target)
    repair_quality = np.zeros(len(target), dtype=bool)
    repair_folds: dict[str, Any] = {}
    for fold_id in range(FOLD_COUNT):
        heldout = folds == fold_id
        training = ~heldout
        contract = fit_object_repair_contract(target[training])
        repair_baseline[heldout] = np.asarray(
            contract["coordinate_median"], dtype=np.float64
        )
        repair_quality[heldout] = object_quality_mask(target[heldout], contract)
        repair_folds[str(fold_id)] = {
            "heldout_labels_used_for_fit": False,
            "training_support": contract["training_support"],
            "training_valid_support": contract["training_valid_support"],
            "max_abs_delta_quality_threshold": contract[
                "max_abs_delta_quality_threshold"
            ],
            "coordinate_median": contract["coordinate_median"],
            "coordinate_robust_scale": contract["coordinate_robust_scale"],
            "heldout_valid_support": int(repair_quality[heldout].sum()),
            "heldout_excluded_support": int((~repair_quality[heldout]).sum()),
        }
    repair_baseline_absolute = np.abs(repair_baseline - target).mean(-1)
    zero_baseline_absolute = np.abs(target).mean(-1)
    quality_model_absolute = absolute.mean(-1)[repair_quality]
    quality_group = groups[repair_quality]
    return {
        "target": "physical_object_position_delta_xyz",
        "support_transitions": int(len(target)),
        "coordinate_count": int(target.shape[-1]),
        "mae_per_coordinate": float(absolute.mean()),
        "rmse_per_coordinate": float(np.sqrt(np.mean(error**2))),
        "mean_l2_error": float(np.linalg.norm(error, axis=-1).mean()),
        "mixture_gaussian_nll_per_coordinate": float(nll.mean()),
        "zero_delta_baseline_mae_per_coordinate": float(np.abs(target).mean()),
        "zero_delta_baseline_rmse_per_coordinate": float(np.sqrt(np.mean(target**2))),
        "paired_mae_skill_vs_zero_delta_baseline": _cluster_mean_comparison(
            absolute.mean(-1), np.abs(target).mean(-1), groups
        ),
        "prospective_next_oof_training_repair": {
            "protocol": OBJECT_REPAIR_PROTOCOL,
            "status": "development_repair_contract_not_current_preregistered_adequacy",
            "excluded_from_current_prediction_adequacy": True,
            "target_fold_labels_used_for_fit": False,
            "location_target": "residual_about_training_fold_coordinate_median",
            "likelihood": "student_t_df3_in_training_fold_robust_coordinates",
            "quality_mask": "finite_and_max_abs_delta_at_most_training_fold_q995",
            "heldout_valid_support": int(repair_quality.sum()),
            "heldout_excluded_support": int((~repair_quality).sum()),
            "heldout_excluded_fraction": float((~repair_quality).mean()),
            "fold_contracts": repair_folds,
            "valid_subset_existing_model_mae_per_coordinate": float(
                quality_model_absolute.mean()
            ),
            "valid_subset_training_fold_robust_median_baseline_mae_per_coordinate": float(
                repair_baseline_absolute[repair_quality].mean()
            ),
            "valid_subset_zero_delta_baseline_mae_per_coordinate": float(
                zero_baseline_absolute[repair_quality].mean()
            ),
            "valid_subset_existing_model_skill_vs_training_fold_robust_median": (
                _cluster_mean_comparison(
                    quality_model_absolute,
                    repair_baseline_absolute[repair_quality],
                    quality_group,
                )
            ),
        },
        "per_fold_heldout": {
            str(fold_id): {
                "support_transitions": int((folds == fold_id).sum()),
                "mae_per_coordinate": float(
                    absolute[folds == fold_id].mean()
                ),
                "rmse_per_coordinate": float(
                    np.sqrt(np.mean(error[folds == fold_id] ** 2))
                ),
                "mixture_gaussian_nll_per_coordinate": float(
                    nll[folds == fold_id].mean()
                ),
            }
            for fold_id in range(FOLD_COUNT)
        },
        "uncertainty_error_relation": _uncertainty_diagnostic(
            uncertainty, sample_error, fold_id=folds
        ),
    }


def _outcome_diagnostics(
    values: Mapping[str, np.ndarray], *, recovery_supervised: bool
) -> dict[str, Any]:
    terminal = values["terminal_mask"]
    success = values["success"][terminal].astype(np.int64)
    folds = values["fold_id"][terminal]
    groups = values["logical_group_id"][terminal]
    logits = values["member_outcome_logits"][:, terminal]
    outcome = values["outcome_id"][terminal]
    binary_mask = outcome < 2 if recovery_supervised else np.ones(len(outcome), dtype=bool)
    binary_label = outcome[binary_mask] if recovery_supervised else success[binary_mask]
    binary_probability = _softmax(logits[:, binary_mask, :2]).mean(0)
    binary_prior = _crossfit_event_prior(
        binary_label, folds[binary_mask], classes=2
    )
    binary_row = np.arange(len(binary_label))
    result: dict[str, Any] = {
        "failure_success": _categorical_metrics(
            binary_label, binary_probability, classes=2
        ),
        "failure_support": int((binary_label == 0).sum()),
        "success_support": int((binary_label == 1).sum()),
        "other_fold_smoothed_prior_baseline": _categorical_metrics(
            binary_label, binary_prior, classes=2
        ),
        "paired_nll_skill_vs_other_fold_prior": _cluster_mean_comparison(
            -np.log(
                np.clip(
                    binary_probability[binary_row, binary_label], EPS, 1.0
                )
            ),
            -np.log(
                np.clip(binary_prior[binary_row, binary_label], EPS, 1.0)
            ),
            groups[binary_mask],
        ),
        "per_fold_heldout_failure_success": {
            str(fold_id): _categorical_metrics(
                binary_label[folds[binary_mask] == fold_id],
                binary_probability[folds[binary_mask] == fold_id],
                classes=2,
            )
            for fold_id in range(FOLD_COUNT)
        },
    }
    recovery_support = int((outcome == 2).sum())
    if recovery_supervised:
        probability = _softmax(logits).mean(0)
        recovery_prior = _crossfit_event_prior(
            outcome, folds, classes=probability.shape[-1]
        )
        result["failure_success_recovery"] = _categorical_metrics(
            outcome, probability, classes=probability.shape[-1]
        )
        result["failure_success_recovery_other_fold_prior_baseline"] = (
            _categorical_metrics(
                outcome, recovery_prior, classes=probability.shape[-1]
            )
        )
        result["recovery_status"] = "evaluated_from_supervised_training_contract"
    else:
        result["failure_success_recovery"] = None
        result["recovery_status"] = (
            "not_evaluable_model_contract_recovery_supervised_false"
        )
    result["recovery_label_support"] = recovery_support
    result["trajectory_regression_support"] = int(
        values["trajectory_regress"][terminal].sum()
    )
    result["trajectory_recovery_support"] = int(
        values["trajectory_recovery"][terminal].sum()
    )
    result["recovery_claim_supported"] = bool(
        recovery_supervised
        and result["recovery_status"]
        == "evaluated_from_supervised_training_contract"
    )
    result["recovery_not_trained_fail_closed"] = not result[
        "recovery_claim_supported"
    ]
    recovery_contract = recovery_adapter_training_contract(
        values["trajectory_recovery"][terminal].astype(np.float64),
        values["structured_mask"][terminal],
    )
    result["prospective_frozen_feature_recovery_adapter"] = {
        **recovery_contract,
        "protocol": RECOVERY_ADAPTER_PROTOCOL,
        "status_for_current_checkpoint": (
            "not_trained_current_factual_config_recovery_supervised_false"
            if not recovery_supervised
            else "current_joint_head_was_trained_adapter_not_required"
        ),
        "excluded_from_current_prediction_adequacy": True,
    }
    return result


def _event_subhead_diagnostics(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    structured = values["structured_mask"]
    folds = values["fold_id"][structured]
    groups = values["logical_group_id"][structured]
    predicate_probability = _sigmoid(
        values["member_post_predicate_logits"][:, structured]
    ).mean(0)
    predicate_label = values["post_predicates"][structured]
    predicate_names = list(map(str, values["predicate_names"]))
    predicate_baseline = np.empty_like(predicate_probability)
    for fold_id in range(FOLD_COUNT):
        heldout = folds == fold_id
        training = ~heldout
        positive = predicate_label[training].sum(0)
        # Beta(1,1) smoothing prevents an absent other-fold class from
        # producing an artificial infinite NLL on the target fold.
        predicate_baseline[heldout] = (positive + 1.0) / (training.sum() + 2.0)
    predicate_nll = -(
        predicate_label * np.log(np.clip(predicate_probability, EPS, 1.0))
        + (1.0 - predicate_label)
        * np.log(np.clip(1.0 - predicate_probability, EPS, 1.0))
    )
    observed = values["dense_mask"] & values["duration_observed"]
    destination_probability = _softmax(
        values["member_next_reached_event_logits"][:, observed]
    ).mean(0)
    destination_label = values["next_reached_event_id"][observed]
    per_predicate: list[dict[str, Any]] = []
    for predicate_id, name in enumerate(predicate_names):
        label = predicate_label[:, predicate_id]
        probability = predicate_probability[:, predicate_id]
        baseline = predicate_baseline[:, predicate_id]
        model_categorical = np.stack([1.0 - probability, probability], axis=-1)
        baseline_categorical = np.stack([1.0 - baseline, baseline], axis=-1)
        repair_row: Mapping[str, Any] = {
            "status": "unavailable_predicate_head_training_weight_not_recorded",
            "excluded_from_current_prediction_adequacy": True,
            "reason": (
                "branch-fold prevalence cannot reconstruct the positive weight "
                "used by a frozen or historically trained predicate head"
            ),
        }
        per_predicate.append(
            {
                "predicate_id": predicate_id,
                "predicate_name": name,
                "positive_support": int((label > 0.5).sum()),
                "negative_support": int((label <= 0.5).sum()),
                "model": _binary_metrics(label, probability),
                "model_macro_f1": _categorical_metrics(
                    label.astype(np.int64), model_categorical, classes=2
                )["macro_f1_present_classes"],
                "other_fold_smoothed_prevalence_baseline": _binary_metrics(
                    label, baseline
                ),
                "baseline_macro_f1": _categorical_metrics(
                    label.astype(np.int64), baseline_categorical, classes=2
                )["macro_f1_present_classes"],
                "paired_brier_skill_vs_other_fold_prevalence": (
                    _cluster_mean_comparison(
                        np.square(probability - label),
                        np.square(baseline - label),
                        groups,
                    )
                ),
                "prospective_next_oof_probability_repair": repair_row,
            }
        )
    return {
        "post_predicates": {
            "support_transitions": int(structured.sum()),
            "binary_entries": int(predicate_label.size),
            "micro_accuracy_at_0_5": float(
                np.mean((predicate_probability >= 0.5) == (predicate_label > 0.5))
            ),
            "binary_nll": float(predicate_nll.mean()),
            "brier": float(np.mean((predicate_probability - predicate_label) ** 2)),
            "per_predicate": per_predicate,
            "baseline_protocol": (
                "each_target_fold_compared_to_beta11_smoothed_prevalence_"
                "fit_on_other_four_folds_only"
            ),
        },
        "next_reached_event_observed_only": _categorical_metrics(
            destination_label,
            destination_probability,
            classes=destination_probability.shape[-1],
        ),
    }


def _comparison_pass(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("model_better_if_upper_ci_below_zero") is True
    )


def _uncertainty_pass(value: Mapping[str, Any]) -> bool:
    spearman = value.get("spearman_uncertainty_vs_error")
    return bool(
        spearman is not None
        and float(spearman) > 0.0
        and float(value.get("aurc_improvement_over_random", -math.inf)) > 0.0
        and int(value.get("folds_with_aurc_better_than_random", 0))
        >= MIN_UNCERTAINTY_FOLD_WINS
    )


def _prediction_adequacy_decision(
    success: Mapping[str, Any],
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a frozen generic skill gate, independent of policy authorization."""

    thresholds = {
        "paired_loss_bootstrap_upper_ci": "strictly_below_zero",
        "bootstrap_alpha": ADEQUACY_ALPHA,
        "minimum_binary_support_per_class": MIN_BINARY_CLASS_SUPPORT,
        "minimum_present_event_classes": MIN_EVENT_PRESENT_CLASSES,
        "minimum_support_per_present_event_class": MIN_EVENT_CLASS_SUPPORT,
        "maximum_success_ece_10_equal_width": MAX_SUCCESS_ECE,
        "minimum_predicate_support_per_binary_class": MIN_PREDICATE_CLASS_SUPPORT,
        "minimum_regression_groups": MIN_REGRESSION_GROUPS,
        "minimum_uncertainty_folds_better_than_random": MIN_UNCERTAINTY_FOLD_WINS,
        "pr_auc": "strictly_above_positive_prevalence",
        "macro_f1": "strictly_above_each_reported_baseline",
        "within_group_pair_accuracy": "group_clustered_error_upper_ci_below_random_0_5",
    }
    if structured.get("status") != "complete":
        return {
            "protocol": ADEQUACY_PROTOCOL,
            "status": "not_evaluable_structured_predictions_missing",
            "generic_development_predictive_skill_pass": False,
            "thresholds_frozen_before_development_result": thresholds,
            "independent_of_reranking_authorization_guard": True,
            "fresh50_authorization_effect": "none",
        }

    success_rank_metrics = success["uncalibrated_ensemble"]
    success_baseline = success["other_fold_prevalence_baseline"]
    strict_success = success.get("strict_probability_assessment")
    strict_probability_evaluable = bool(
        isinstance(strict_success, Mapping)
        and strict_success.get("usable_for_strict_adequacy") is True
        and isinstance(strict_success.get("metrics"), Mapping)
        and isinstance(
            strict_success.get("paired_skill_vs_other_fold_prevalence"), Mapping
        )
    )
    success_probability_metrics = (
        strict_success["metrics"]
        if strict_probability_evaluable
        else success_rank_metrics
    )
    success_skill = (
        strict_success["paired_skill_vs_other_fold_prevalence"]
        if strict_probability_evaluable
        else None
    )
    pair = success["within_group_success_pair_ranking"]
    success_checks = {
        "positive_support": int(success_rank_metrics["positive_support"])
        >= MIN_BINARY_CLASS_SUPPORT,
        "negative_support": (
            int(success_rank_metrics["support"])
            - int(success_rank_metrics["positive_support"])
        )
        >= MIN_BINARY_CLASS_SUPPORT,
        "strict_probability_calibration_evaluable": strict_probability_evaluable,
        "brier_better_than_crossfit_prevalence": bool(
            strict_probability_evaluable
            and _comparison_pass(success_skill["brier"])
        ),
        "nll_better_than_crossfit_prevalence": bool(
            strict_probability_evaluable
            and _comparison_pass(success_skill["nll"])
        ),
        "pr_auc_above_prevalence": bool(
            success_rank_metrics["pr_auc_average_precision"] is not None
            and float(success_rank_metrics["pr_auc_average_precision"])
            > float(success_baseline["positive_prevalence_random_pr_auc"])
        ),
        "ece_at_most_0_10": bool(
            strict_probability_evaluable
            and float(success_probability_metrics["ece_10_equal_width"])
            <= MAX_SUCCESS_ECE
        ),
        "within_group_pair_skill_above_random": _comparison_pass(
            pair["paired_error_vs_random_0_5"]
        ),
    }

    event = structured["next_event"]
    event_model = event["ensemble"]
    event_persistence = event["current_event_persistence_baseline"]
    event_prior = event["other_fold_smoothed_prior_baseline"]
    event_skill = event["paired_skill_vs_baselines"]
    present_support = [count for count in event_model["class_support"] if count > 0]
    event_checks = {
        "class_support": bool(
            len(present_support) >= MIN_EVENT_PRESENT_CLASSES
            and min(present_support) >= MIN_EVENT_CLASS_SUPPORT
        ),
        "nll_better_than_persistence": _comparison_pass(
            event_skill["nll_vs_current_event_persistence"]
        ),
        "nll_better_than_crossfit_prior": _comparison_pass(
            event_skill["nll_vs_other_fold_prior"]
        ),
        "top1_error_better_than_persistence": _comparison_pass(
            event_skill["top1_error_vs_current_event_persistence"]
        ),
        "top1_error_better_than_crossfit_prior": _comparison_pass(
            event_skill["top1_error_vs_other_fold_prior"]
        ),
        "macro_f1_better_than_persistence": bool(
            float(event_model["macro_f1_present_classes"])
            > float(event_persistence["macro_f1_present_classes"])
        ),
        "macro_f1_better_than_crossfit_prior": bool(
            float(event_model["macro_f1_present_classes"])
            > float(event_prior["macro_f1_present_classes"])
        ),
    }
    predicates = structured["event_subheads"]["post_predicates"]
    predicate_checks: dict[str, bool] = {
        "predicate_vocabulary_nonempty": bool(predicates["per_predicate"])
    }
    for item in predicates["per_predicate"]:
        prefix = str(item["predicate_name"])
        predicate_checks[f"{prefix}.positive_support"] = (
            int(item["positive_support"]) >= MIN_PREDICATE_CLASS_SUPPORT
        )
        predicate_checks[f"{prefix}.negative_support"] = (
            int(item["negative_support"]) >= MIN_PREDICATE_CLASS_SUPPORT
        )
        predicate_checks[f"{prefix}.brier_better_than_crossfit_prevalence"] = (
            _comparison_pass(item["paired_brier_skill_vs_other_fold_prevalence"])
        )
        predicate_checks[f"{prefix}.macro_f1_better_than_crossfit_prevalence"] = bool(
            float(item["model_macro_f1"]) > float(item["baseline_macro_f1"])
        )

    duration = structured["duration"]
    duration_observed = duration.get("observed", {})
    duration_comparison = duration_observed.get(
        "paired_log1p_mae_skill_vs_event_body_crossfit_median"
    )
    duration_checks = {
        "observed_group_support": bool(
            isinstance(duration_comparison, Mapping)
            and int(duration_comparison.get("groups", 0)) >= MIN_REGRESSION_GROUPS
        ),
        "log1p_mae_better_than_event_body_crossfit_median": _comparison_pass(
            duration_comparison
        ),
        "right_censoring_evaluated": bool(
            int(duration.get("right_censored_support", 0)) > 0
            and isinstance(duration.get("right_censored"), Mapping)
            and duration["right_censored"].get("mixture_survival_nll") is not None
        ),
    }

    object_state = structured["object_state_change"]
    object_comparison = object_state["paired_mae_skill_vs_zero_delta_baseline"]
    object_checks = {
        "group_support": int(object_comparison.get("groups", 0))
        >= MIN_REGRESSION_GROUPS,
        "mae_better_than_zero_delta": _comparison_pass(object_comparison),
    }

    outcome = structured["outcome"]
    outcome_model = outcome["failure_success"]
    outcome_prior = outcome["other_fold_smoothed_prior_baseline"]
    outcome_checks = {
        "failure_support": int(outcome["failure_support"])
        >= MIN_BINARY_CLASS_SUPPORT,
        "success_support": int(outcome["success_support"])
        >= MIN_BINARY_CLASS_SUPPORT,
        "nll_better_than_crossfit_prior": _comparison_pass(
            outcome["paired_nll_skill_vs_other_fold_prior"]
        ),
        "macro_f1_better_than_crossfit_prior": bool(
            float(outcome_model["macro_f1_present_classes"])
            > float(outcome_prior["macro_f1_present_classes"])
        ),
    }
    recovery_checks = {
        "recovery_labels_present": int(outcome["recovery_label_support"])
        >= MIN_BINARY_CLASS_SUPPORT,
        "recovery_head_trained": bool(outcome["recovery_claim_supported"]),
        "recovery_prediction_evaluable": bool(
            outcome.get("failure_success_recovery") is not None
        ),
    }
    if outcome.get("recovery_status") == "evaluated_from_supervised_training_contract":
        recovery_model = outcome["failure_success_recovery"]
        recovery_prior = outcome[
            "failure_success_recovery_other_fold_prior_baseline"
        ]
        recovery_checks["recovery_class_support"] = bool(
            min(recovery_model["class_support"]) >= MIN_BINARY_CLASS_SUPPORT
        )
        recovery_checks["recovery_macro_f1_better_than_crossfit_prior"] = bool(
            float(recovery_model["macro_f1_present_classes"])
            > float(recovery_prior["macro_f1_present_classes"])
        )

    uncertainty_checks = {
        "success": _uncertainty_pass(success["uncertainty_error_relation"]),
        "next_event": _uncertainty_pass(
            event["uncertainty_error_relation"]
        ),
        "duration": bool(
            isinstance(duration_observed.get("uncertainty_error_relation"), Mapping)
            and _uncertainty_pass(duration_observed["uncertainty_error_relation"])
        ),
        "object_state_change": _uncertainty_pass(
            object_state["uncertainty_error_relation"]
        ),
    }
    domains = {
        "success_probability_and_within_group_ranking": success_checks,
        "next_event": event_checks,
        "post_event_predicates": predicate_checks,
        "duration": duration_checks,
        "object_state_change": object_checks,
        "failure_success_outcome": outcome_checks,
        "recovery": recovery_checks,
        "uncertainty_risk_ordering": uncertainty_checks,
    }
    domain_pass = {
        name: bool(all(checks.values())) for name, checks in domains.items()
    }
    overall = bool(all(domain_pass.values()))
    return {
        "protocol": ADEQUACY_PROTOCOL,
        "status": "pass" if overall else "fail_closed",
        "generic_development_predictive_skill_pass": overall,
        "thresholds_frozen_before_development_result": thresholds,
        "domain_checks": domains,
        "domain_pass": domain_pass,
        "failed_checks": [
            f"{domain}.{name}"
            for domain, checks in domains.items()
            for name, passed in checks.items()
            if not passed
        ],
        "candidate_scope": "deployment_exact_first_four_only_for_success_and_ranking",
        "independent_of_reranking_authorization_guard": True,
        "fresh50_authorization_effect": "none",
        "claim_boundary": (
            "pass_means_generic_heldout_skill_vs_frozen_baselines_on_current_"
            "development_distribution;not_absolute_task_safety_not_cross_body_"
            "or_cross_policy_proof_not_fresh_confirmation"
        ),
    }


def build_oof_prediction_diagnostics(
    raw_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Build descriptive diagnostics without changing any OOF decision gate."""

    rows = _validate_rows(raw_rows, manifest)
    deployment_rows = _deployment_success_rows(rows, deployment_only=True)
    all_candidate_rows = _deployment_success_rows(rows, deployment_only=False)
    expected_groups, _, heldout_per_fold = oof_dimensions(manifest)
    structured_presence = [isinstance(row.get("structured_predictions"), Mapping) for row in rows]
    if any(structured_presence) and not all(structured_presence):
        raise RuntimeError("mixed legacy/new structured OOF rows are not comparable")
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete" if all(structured_presence) else "legacy_raw_schema_partial",
        "evidence_tier": "development_five_fold_heldout_prediction_diagnostics",
        "oof_preregistration_sha256": manifest["preregistration_sha256"],
        "oof_groups": expected_groups,
        "fold_count": FOLD_COUNT,
        "heldout_groups_per_fold": heldout_per_fold,
        "prediction_source": (
            "each_row_from_unique_owner_fold_model_never_trained_on_that_group"
        ),
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
        "diagnostics_are_descriptive_not_an_authorization_or_confirmation_gate": True,
        "success_probability": {
            "candidate_scope": "deployment_exact_first_four_only",
            "deployment_candidate_names": list(DEPLOYMENT_CANDIDATE_NAMES),
            **_success_diagnostics(deployment_rows),
        },
        "success_probability_all_collected_candidates_appendix": {
            "candidate_scope": (
                "deployment_plus_registered_training_only_fifth_if_present"
            ),
            "training_only_candidate_names": list(TRAINING_ONLY_EXTRA_CANDIDATES),
            "excluded_from_main_prediction_adequacy": True,
            **_success_diagnostics(all_candidate_rows),
        },
    }
    if not all(structured_presence):
        result["structured_world_model"] = {
            "status": "not_evaluable_legacy_raw_artifact_missing_structured_predictions",
            "required_raw_row_format": STRUCTURED_ROW_FORMAT,
        }
    else:
        blocks = [_structured_block(row) for row in rows]
        for row, block in zip(rows, blocks):
            terminal_indices = np.flatnonzero(block["terminal_mask"])
            candidate_count = len(row["success"])
            if (
                not np.array_equal(
                    terminal_indices, np.arange(candidate_count, dtype=np.int64)
                )
                or block["sample_names"][:candidate_count] != row["candidate_names"]
                or not np.array_equal(
                    block["success"][terminal_indices], row["success"]
                )
            ):
                raise RuntimeError(
                    "structured terminal rows do not match candidate success rows"
                )
        if any(
            block["predicate_names"] != blocks[0]["predicate_names"]
            for block in blocks[1:]
        ):
            raise RuntimeError("OOF folds disagree on predicate vocabulary")
        recovery_flags = {bool(block["recovery_supervised"]) for block in blocks}
        if len(recovery_flags) != 1:
            raise RuntimeError("OOF folds disagree on recovery supervision contract")
        values = _concatenate_structured(rows, blocks)
        result["structured_world_model"] = {
            "status": "complete",
            "support": {
                "transitions": int(len(values["dense_mask"])),
                "structured_transitions": int(values["structured_mask"].sum()),
                "terminal_branches": int(values["terminal_mask"].sum()),
                "observed_durations": int(values["duration_observed"].sum()),
                "right_censored_durations": int((values["dense_mask"] & ~values["duration_observed"]).sum()),
            },
            "next_event": _event_diagnostics(values),
            "event_subheads": _event_subhead_diagnostics(values),
            "duration": _duration_diagnostics(values),
            "object_state_change": _object_diagnostics(values),
            "outcome": _outcome_diagnostics(
                values, recovery_supervised=recovery_flags.pop()
            ),
        }
    result["prediction_adequacy"] = _prediction_adequacy_decision(
        result["success_probability"], result["structured_world_model"]
    )
    result["diagnostics_sha256"] = canonical_sha256(result)
    return result


def validate_oof_prediction_diagnostics(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    require_structured: bool = True,
) -> dict[str, int]:
    """Validate the signed held-out diagnostic artifact without HDF5 access."""

    unsigned = dict(value)
    recorded = str(unsigned.pop("diagnostics_sha256", ""))
    expected_groups, _, heldout = oof_dimensions(manifest)
    structured = value.get("structured_world_model")
    adequacy = value.get("prediction_adequacy")
    success = value.get("success_probability")
    appendix = value.get("success_probability_all_collected_candidates_appendix")
    if (
        value.get("format") != FORMAT
        or value.get("status")
        not in ("complete", "legacy_raw_schema_partial")
        or recorded != canonical_sha256(unsigned)
        or value.get("oof_preregistration_sha256")
        != manifest.get("preregistration_sha256")
        or int(value.get("oof_groups", -1)) != expected_groups
        or int(value.get("fold_count", -1)) != FOLD_COUNT
        or int(value.get("heldout_groups_per_fold", -1)) != heldout
        or value.get("fresh_confirmation_data_or_labels_read") is not False
        or value.get("authorization_guard_changed") is not False
        or value.get(
            "diagnostics_are_descriptive_not_an_authorization_or_confirmation_gate"
        )
        is not True
        or not isinstance(success, Mapping)
        or success.get("candidate_scope") != "deployment_exact_first_four_only"
        or not isinstance(appendix, Mapping)
        or appendix.get("excluded_from_main_prediction_adequacy") is not True
        or not isinstance(structured, Mapping)
        or not isinstance(adequacy, Mapping)
        or adequacy.get("protocol") != ADEQUACY_PROTOCOL
        or adequacy.get("independent_of_reranking_authorization_guard") is not True
        or adequacy.get("fresh50_authorization_effect") != "none"
    ):
        raise RuntimeError("OOF prediction diagnostics contract/signature changed")
    if require_structured and (
        value.get("status") != "complete"
        or structured.get("status") != "complete"
    ):
        raise RuntimeError("formal schema-v5 OOF lacks complete structured diagnostics")
    return {
        "oof_groups": expected_groups,
        "heldout_groups_per_fold": heldout,
    }


__all__ = [
    "FROZEN_FACTUAL_WEIGHT_UNAVAILABLE_STATUS",
    "FORMAT",
    "RECORDED_WEIGHTED_SUCCESS_STATUS",
    "STRUCTURED_ROW_FORMAT",
    "SUCCESS_HEAD_TRAINING_CONTRACT_FORMAT",
    "build_oof_prediction_diagnostics",
    "validate_oof_prediction_diagnostics",
]
