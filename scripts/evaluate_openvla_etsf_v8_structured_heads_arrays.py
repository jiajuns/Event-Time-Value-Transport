#!/usr/bin/env python3
"""Pure-NumPy, fail-closed evaluator for preregistered ETSF v8 heads.

The public entry point consumes already-materialised arrays.  This module has
no filesystem, HDF5, launcher, server, CUDA, or Fresh50 interface.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from openvla_etsf_v8_structured_heads_protocol import (
    ALPHA,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    FOLD_COUNT,
    canonical_sha256,
    validate_oof_ownership,
    validate_preregistration,
    validate_probability_weight_provenance,
)


FORMAT = "etsf_v8_structured_heads_array_evaluation_v1"
EPS = 1e-12


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a one-dimensional aligned array")
    return result


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _vector(value, name=name, length=length).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _binary(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _finite_vector(value, name=name, length=length)
    if np.any((result != 0.0) & (result != 1.0)):
        raise ValueError(f"{name} must contain binary values")
    return result


def _probability(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _finite_vector(value, name=name, length=length)
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0,1]")
    return np.clip(result, EPS, 1.0 - EPS)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
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


def _ece(labels: np.ndarray, probability: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probability[selected].mean() - labels[selected].mean())
            )
    return float(result)


def _binary_losses(
    labels: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    brier = np.square(probability - labels)
    nll = -(
        labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability)
    )
    return brier, nll


def _group_indices(groups: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(groups == group) for group in np.unique(groups)]


def _cluster_loss_comparison(
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    if not (len(model_loss) == len(baseline_loss) == len(groups)) or not len(groups):
        raise ValueError("cluster loss arrays are empty or misaligned")
    clusters = _group_indices(groups)
    delta = np.asarray(
        [float((model_loss[index] - baseline_loss[index]).mean()) for index in clusters]
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sample = generator.integers(
        0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta))
    )
    means = delta[sample].mean(axis=1)
    low, high = np.quantile(means, [ALPHA / 2.0, 1.0 - ALPHA / 2.0])
    return {
        "estimand": "equal_logical_group_mean_model_minus_baseline_loss",
        "groups": len(clusters),
        "model": float(np.mean(model_loss)),
        "baseline": float(np.mean(baseline_loss)),
        "mean_delta": float(delta.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
        "strict_skill": bool(high < 0.0),
    }


def _cluster_statistic_comparison(
    arrays: Sequence[np.ndarray],
    groups: np.ndarray,
    statistic: Callable[..., float],
    *,
    direction: str,
) -> dict[str, Any]:
    clusters = _group_indices(groups)
    observed = float(statistic(*arrays))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for bootstrap_index in range(BOOTSTRAP_SAMPLES):
        chosen = generator.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[index] for index in chosen])
        values[bootstrap_index] = statistic(*(array[rows] for array in arrays))
    finite = values[np.isfinite(values)]
    if len(finite) != BOOTSTRAP_SAMPLES:
        return {
            "observed": observed,
            "bootstrap_95_ci": None,
            "strict_skill": False,
            "status": "fail_closed_nonfinite_cluster_resample",
        }
    low, high = np.quantile(finite, [ALPHA / 2.0, 1.0 - ALPHA / 2.0])
    if direction == "positive":
        skill = low > 0.0
    elif direction == "nonpositive":
        skill = high <= 0.0
    else:
        raise ValueError("unknown statistic comparison direction")
    return {
        "observed": observed,
        "bootstrap_95_ci": [float(low), float(high)],
        "strict_skill": bool(skill),
        "status": "complete",
    }


def _support_by_fold(labels: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fold in range(FOLD_COUNT):
        selected = labels[folds == fold]
        result[str(fold)] = {
            "support": int(len(selected)),
            "positive": int(selected.sum()),
            "negative": int(len(selected) - selected.sum()),
        }
    return result


def _binary_domain(
    *,
    name: str,
    labels: np.ndarray,
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    minimum_positive_per_fold: int,
    minimum_negative_per_fold: int,
    maximum_ece: float,
    weight_provenance: Mapping[int | str, Mapping[str, Any]] | None,
    baseline_prevalence_by_fold: Mapping[int | str, float] | None = None,
    weight_provenance_head: str | None = None,
) -> dict[str, Any]:
    support = _support_by_fold(labels, folds)
    support_pass = all(
        item["positive"] >= minimum_positive_per_fold
        and item["negative"] >= minimum_negative_per_fold
        for item in support.values()
    )
    provenance_error = None
    provenance_result = None
    try:
        if weight_provenance is None:
            raise RuntimeError(f"{name} probability weight provenance is missing")
        provenance_result = validate_probability_weight_provenance(
            weight_provenance, head=weight_provenance_head or name
        )
    except (RuntimeError, ValueError) as error:
        provenance_error = str(error)

    baseline_error = None
    if provenance_result is not None:
        expected = (
            {int(key): float(value) for key, value in baseline_prevalence_by_fold.items()}
            if baseline_prevalence_by_fold is not None
            else {
                int(key): float(item["outer_training_prevalence"])
                for key, item in provenance_result["folds"].items()
            }
        )
        if set(expected) != set(range(FOLD_COUNT)):
            raise RuntimeError(f"{name} baseline prevalence must cover five folds")
        for fold in range(FOLD_COUNT):
            selected = folds == fold
            if selected.any() and not np.allclose(
                baseline_probability[selected], expected[fold], rtol=0.0, atol=1e-15
            ):
                raise RuntimeError(
                    f"{name} baseline probability is not the owner-fold outer-training prevalence"
                )

    if not len(labels):
        return {
            "status": "fail_closed_no_support",
            "passed": False,
            "support_by_fold": support,
            "support_gate": False,
            "weight_provenance": provenance_result,
            "weight_provenance_error": provenance_error,
            "baseline_provenance_error": baseline_error,
            "brier_vs_crossfit_prevalence": None,
            "nll_vs_crossfit_prevalence": None,
            "ap_minus_prevalence": None,
            "ece_10": None,
            "ece_gate": False,
        }

    model_brier, model_nll = _binary_losses(labels, probability)
    baseline_brier, baseline_nll = _binary_losses(labels, baseline_probability)
    brier = _cluster_loss_comparison(model_brier, baseline_brier, groups)
    nll = _cluster_loss_comparison(model_nll, baseline_nll, groups)

    def ap_gap(y: np.ndarray, p: np.ndarray) -> float:
        return _average_precision(y > 0.5, p) - float(np.mean(y))

    ap = _cluster_statistic_comparison(
        [labels, probability], groups, ap_gap, direction="positive"
    )
    ece = _ece(labels, probability)
    passed = bool(
        support_pass
        and provenance_error is None
        and baseline_error is None
        and brier["strict_skill"]
        and nll["strict_skill"]
        and ap["strict_skill"]
        and ece <= maximum_ece
    )
    return {
        "status": "passed" if passed else "fail_closed",
        "passed": passed,
        "support_by_fold": support,
        "support_gate": support_pass,
        "weight_provenance": provenance_result,
        "weight_provenance_error": provenance_error,
        "baseline_provenance_error": baseline_error,
        "brier_vs_crossfit_prevalence": brier,
        "nll_vs_crossfit_prevalence": nll,
        "ap_minus_prevalence": ap,
        "ece_10": ece,
        "ece_gate": bool(ece <= maximum_ece),
    }


def _duration_domain(
    arrays: Mapping[str, Any], primary: np.ndarray, groups: np.ndarray, folds: np.ndarray,
    *, fold_contracts: Mapping[int | str, Mapping[str, Any]],
) -> dict[str, Any]:
    length = len(primary)
    observed = _binary(
        arrays["duration_observed"], name="duration observed", length=length
    ).astype(bool)
    selected = primary & observed
    if not selected.any():
        return {"status": "fail_closed_no_observed_duration", "passed": False}
    label = _finite_vector(arrays["duration_steps"], name="duration", length=length)
    if np.any(label < 0.0):
        raise ValueError("duration must be nonnegative")
    target = np.log1p(label[selected])
    model_location_all = _finite_vector(
        arrays["duration_model_log_location"], name="duration model location", length=length
    )
    baseline_location_all = _finite_vector(
        arrays["duration_baseline_log_location"], name="duration baseline location", length=length
    )
    frozen_location_all = _finite_vector(
        arrays["duration_frozen_log_location"], name="duration frozen location", length=length
    )
    expected_location = baseline_location_all + 0.375 * (
        frozen_location_all - baseline_location_all
    )
    if not np.allclose(
        model_location_all, expected_location, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError(
            "duration model location does not equal baseline + 0.375*(frozen-baseline)"
        )
    model_location = model_location_all[selected]
    baseline_location = baseline_location_all[selected]
    with np.errstate(over="ignore", invalid="ignore"):
        model_scale_all = np.exp(
            _finite_vector(arrays["duration_model_log_scale"], name="duration model scale", length=length)
        )
        baseline_scale_all = np.exp(
            _finite_vector(arrays["duration_baseline_log_scale"], name="duration baseline scale", length=length)
        )
    if (
        not np.isfinite(model_scale_all).all()
        or not np.isfinite(baseline_scale_all).all()
        or np.any(model_scale_all <= 0.0)
        or np.any(baseline_scale_all <= 0.0)
    ):
        raise ValueError("duration Laplace scale must be positive")
    model_scale = model_scale_all[selected]
    baseline_scale = baseline_scale_all[selected]
    selected_groups = groups[selected]
    selected_folds = folds[selected]
    mae = _cluster_loss_comparison(
        np.abs(target - model_location),
        np.abs(target - baseline_location),
        selected_groups,
    )
    model_nll = np.abs(target - model_location) / model_scale + np.log(2.0 * model_scale)
    baseline_nll = np.abs(target - baseline_location) / baseline_scale + np.log(2.0 * baseline_scale)
    nll = _cluster_loss_comparison(model_nll, baseline_nll, selected_groups)
    fold_wins = {
        str(fold): bool(
            np.mean(np.abs(target[selected_folds == fold] - model_location[selected_folds == fold]))
            <= np.mean(np.abs(target[selected_folds == fold] - baseline_location[selected_folds == fold]))
        ) if np.any(selected_folds == fold) else False
        for fold in range(FOLD_COUNT)
    }
    group_support = len(np.unique(selected_groups))
    cell_support = min(
        int(dict(item)["duration_event_body_min_training_support"])
        for item in fold_contracts.values()
    )
    passed = bool(
        group_support >= 30
        and cell_support >= 20
        and sum(fold_wins.values()) >= 4
        and mae["strict_skill"]
        and nll["strict_skill"]
    )
    return {
        "status": "passed" if passed else "fail_closed",
        "passed": passed,
        "observed_rows": int(selected.sum()),
        "observed_logical_groups": group_support,
        "right_censored_rows_excluded_from_duration_error": int((primary & ~observed).sum()),
        "minimum_outer_training_event_body_support": cell_support,
        "mae_log1p_vs_event_body_median": mae,
        "laplace_nll_vs_training_median_mad_laplace": nll,
        "fold_point_estimate_noninferiority": fold_wins,
        "fold_wins": int(sum(fold_wins.values())),
    }


def _object_domain(
    arrays: Mapping[str, Any], primary: np.ndarray, groups: np.ndarray, folds: np.ndarray
) -> dict[str, Any]:
    length = len(primary)
    supervised = _binary(arrays["object_mask"], name="object mask", length=length).astype(bool)
    valid = _binary(
        arrays["object_pose_quality_valid"], name="object quality", length=length
    ).astype(bool)
    label = np.asarray(arrays["object_delta"], dtype=np.float64)
    model = np.asarray(arrays["object_model_delta"], dtype=np.float64)
    robust = np.asarray(arrays["object_robust_median_delta"], dtype=np.float64)
    if not (
        label.ndim == 2
        and label.shape == model.shape == robust.shape
        and label.shape[0] == length
        and label.shape[1] > 0
        and np.isfinite(label).all()
        and np.isfinite(model).all()
        and np.isfinite(robust).all()
    ):
        raise ValueError("object arrays must be finite aligned matrices")
    base = primary & supervised
    quality_coverage = float(valid[base].mean()) if base.any() else 0.0

    def compare(mask: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
        absolute_model = np.abs(model[mask] - label[mask])
        absolute_baseline = np.abs(baseline[mask] - label[mask])
        mae = _cluster_loss_comparison(
            absolute_model.mean(1), absolute_baseline.mean(1), groups[mask]
        )
        row_rmse = _cluster_loss_comparison(
            np.sqrt(np.square(absolute_model).mean(1)),
            np.sqrt(np.square(absolute_baseline).mean(1)),
            groups[mask],
        )

        def p95_gap(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.quantile(a, 0.95) - np.quantile(b, 0.95))

        p95 = _cluster_statistic_comparison(
            [absolute_model.max(1), absolute_baseline.max(1)],
            groups[mask],
            p95_gap,
            direction="nonpositive",
        )
        return {"mae": mae, "row_rmse": row_rmse, "p95_absolute_error": p95}

    if not base.any() or not (base & valid).any():
        return {
            "status": "fail_closed_no_object_support",
            "passed": False,
            "activated_output": "outer_training_robust_median_fallback",
            "learned_multiplier": 0.0,
        }
    zero = np.zeros_like(label)
    comparisons = {
        "all_recorded": {
            "zero": compare(base, zero),
            "outer_training_robust_median": compare(base, robust),
        },
        "quality_valid": {
            "zero": compare(base & valid, zero),
            "outer_training_robust_median": compare(base & valid, robust),
        },
    }
    fold_wins: dict[str, dict[str, bool]] = {
        "all_recorded": {},
        "quality_valid": {},
    }
    for cohort_name, cohort_mask in (
        ("all_recorded", base),
        ("quality_valid", base & valid),
    ):
        for fold in range(FOLD_COUNT):
            selected = cohort_mask & (folds == fold)
            if not selected.any():
                fold_wins[cohort_name][str(fold)] = False
                continue
            model_mae = np.abs(model[selected] - label[selected]).mean()
            fold_wins[cohort_name][str(fold)] = bool(
                model_mae <= np.abs(label[selected]).mean()
                and model_mae <= np.abs(robust[selected] - label[selected]).mean()
            )
    comparison_pass = all(
        item[metric]["strict_skill"]
        for cohort in comparisons.values()
        for item in cohort.values()
        for metric in ("mae", "row_rmse", "p95_absolute_error")
    )
    fold_win_counts = {
        name: int(sum(values.values())) for name, values in fold_wins.items()
    }
    passed = bool(
        len(np.unique(groups[base])) >= 30
        and quality_coverage >= 0.99
        and all(value >= 4 for value in fold_win_counts.values())
        and comparison_pass
    )
    return {
        "status": "passed_learned_output" if passed else "fail_closed_fallback",
        "passed": passed,
        "logical_groups": len(np.unique(groups[base])),
        "quality_valid_coverage": quality_coverage,
        "quality_coverage_gate": bool(quality_coverage >= 0.99),
        "comparisons": comparisons,
        "fold_point_estimate_noninferiority_to_both": fold_wins,
        "fold_wins": fold_win_counts,
        "activated_output": (
            "learned_object_delta" if passed else "outer_training_robust_median_fallback"
        ),
        "learned_multiplier": 1.0 if passed else 0.0,
    }


def _descriptive_cohort_support(
    arrays: Mapping[str, Any], mask: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    return {
        "rows": int(mask.sum()),
        "logical_groups": int(len(np.unique(groups[mask]))),
        "success_first_four_rows": int(
            (mask & (np.asarray(arrays["candidate_index"], dtype=np.int64) < 4)).sum()
        ),
        "duration_observed_rows": int(
            (mask & np.asarray(arrays["duration_observed"], dtype=bool)).sum()
        ),
        "object_supervised_rows": int(
            (mask & np.asarray(arrays["object_mask"], dtype=bool)).sum()
        ),
    }


def _descriptive_cohort_metrics(
    arrays: Mapping[str, Any], mask: np.ndarray
) -> dict[str, Any]:
    """Point estimates only; never consumed by a gate or model selection."""

    candidate = np.asarray(arrays["candidate_index"], dtype=np.int64)
    success_mask = mask & np.asarray(arrays["success_mask"], dtype=bool) & (candidate < 4)
    regress_mask = mask & np.asarray(arrays["regress_mask"], dtype=bool)
    observed = mask & np.asarray(arrays["duration_observed"], dtype=bool)
    object_mask = mask & np.asarray(arrays["object_mask"], dtype=bool)

    def binary(prefix: str, selected: np.ndarray) -> dict[str, Any] | None:
        if not selected.any():
            return None
        labels = np.asarray(arrays[f"{prefix}_label"], dtype=np.float64)[selected]
        probability = np.clip(
            np.asarray(arrays[f"{prefix}_probability"], dtype=np.float64)[selected],
            EPS,
            1.0 - EPS,
        )
        brier, nll = _binary_losses(labels, probability)
        return {
            "support": int(selected.sum()),
            "prevalence": float(labels.mean()),
            "brier": float(brier.mean()),
            "nll": float(nll.mean()),
            "ece_10": _ece(labels, probability),
            "average_precision": _average_precision(labels > 0.5, probability),
        }

    duration_metric = None
    if observed.any():
        target = np.log1p(np.asarray(arrays["duration_steps"], dtype=np.float64)[observed])
        model = np.asarray(arrays["duration_model_log_location"], dtype=np.float64)[observed]
        baseline = np.asarray(arrays["duration_baseline_log_location"], dtype=np.float64)[observed]
        duration_metric = {
            "support": int(observed.sum()),
            "model_log1p_mae": float(np.abs(target - model).mean()),
            "event_body_median_log1p_mae": float(np.abs(target - baseline).mean()),
        }
    object_metric = None
    if object_mask.any():
        label = np.asarray(arrays["object_delta"], dtype=np.float64)[object_mask]
        model = np.asarray(arrays["object_model_delta"], dtype=np.float64)[object_mask]
        robust = np.asarray(arrays["object_robust_median_delta"], dtype=np.float64)[object_mask]
        object_metric = {
            "support": int(object_mask.sum()),
            "model_mae_per_coordinate": float(np.abs(model - label).mean()),
            "zero_mae_per_coordinate": float(np.abs(label).mean()),
            "robust_median_mae_per_coordinate": float(np.abs(robust - label).mean()),
        }
    return {
        "status": "descriptive_only_never_used_for_gate",
        "success": binary("success", success_mask),
        "regress": binary("regress", regress_mask),
        "duration": duration_metric,
        "object": object_metric,
    }


def _evaluate_structured_heads_arrays_validated(
    arrays: Mapping[str, Any],
    *,
    source_sha256: Mapping[str, Any],
    contract_sha256: str,
    evidence_design: str,
    fold_contracts: Mapping[int | str, Mapping[str, Any]],
    probability_weight_provenance: Mapping[
        str, Mapping[int | str, Mapping[str, Any]]
    ],
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Shared metric core reached only through a validated public contract."""

    if evidence_design not in {
        "prospective_preregistered_development",
        "adaptive_current_d250_after_collection_started",
    }:
        raise RuntimeError("unknown v8 evidence design")
    if input_contract != {
        "source_partition": "development_only",
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "old100_overlap_declared": True,
    }:
        raise RuntimeError("v8 input contract rejects Fresh50 or undeclared legacy overlap")

    groups = _vector(arrays["logical_group"], name="logical group").astype(str)
    length = len(groups)
    folds = _vector(arrays["fold_id"], name="fold id", length=length).astype(np.int64)
    old100 = _binary(
        arrays["historical_old100_overlap"], name="old100 overlap", length=length
    ).astype(bool)
    candidate_index = _vector(
        arrays["candidate_index"], name="candidate index", length=length
    ).astype(np.int64)
    if np.any(candidate_index < 0):
        raise ValueError("candidate index must be nonnegative")
    for group in np.unique(groups):
        if len(np.unique(old100[groups == group])) != 1:
            raise RuntimeError(
                "historical_old100_overlap must be constant within logical group"
            )
    ownership = validate_oof_ownership(
        groups,
        folds,
        fold_contracts,
        expected_base_checkpoint_sha256=source_sha256["base_checkpoint"],
    )
    primary = ~old100
    if not primary.any():
        raise RuntimeError("v8 has no provenance-clean rows after excluding old100")
    if set(np.unique(folds[primary])) != set(range(FOLD_COUNT)):
        raise RuntimeError("v8 provenance-clean primary cohort must cover five folds")

    duration = _duration_domain(
        arrays, primary, groups, folds, fold_contracts=fold_contracts
    )

    def binary_inputs(prefix: str, mask: np.ndarray) -> tuple[np.ndarray, ...]:
        label = _binary(arrays[f"{prefix}_label"], name=f"{prefix} label", length=length)
        probability = _probability(
            arrays[f"{prefix}_probability"], name=f"{prefix} probability", length=length
        )
        baseline = _probability(
            arrays[f"{prefix}_baseline_probability"],
            name=f"{prefix} baseline probability",
            length=length,
        )
        return label[mask], probability[mask], baseline[mask], groups[mask], folds[mask]

    success_mask = primary & _binary(
        arrays["success_mask"], name="success mask", length=length
    ).astype(bool) & (candidate_index < 4)
    for group in np.unique(groups[primary]):
        selected_candidates = candidate_index[success_mask & (groups == group)]
        if sorted(selected_candidates.tolist()) != [0, 1, 2, 3]:
            raise RuntimeError(
                "each primary logical group must contain exactly one supervised success row for candidates 0..3"
            )
    success_values = binary_inputs("success", success_mask)
    success = _binary_domain(
        name="success",
        labels=success_values[0], probability=success_values[1],
        baseline_probability=success_values[2], groups=success_values[3], folds=success_values[4],
        minimum_positive_per_fold=10, minimum_negative_per_fold=10, maximum_ece=0.10,
        weight_provenance=probability_weight_provenance.get("success"),
    )
    success["candidate_scope"] = "provenance_clean_first_four_only"
    success["training_only_candidate_rows_excluded"] = int(
        (primary & np.asarray(arrays["success_mask"], dtype=bool) & (candidate_index >= 4)).sum()
    )

    regress_mask = primary & _binary(
        arrays["regress_mask"], name="regress mask", length=length
    ).astype(bool)
    regress_values = binary_inputs("regress", regress_mask)
    regress = _binary_domain(
        name="regress",
        labels=regress_values[0], probability=regress_values[1],
        baseline_probability=regress_values[2], groups=regress_values[3], folds=regress_values[4],
        minimum_positive_per_fold=10, minimum_negative_per_fold=10, maximum_ece=0.10,
        weight_provenance=probability_weight_provenance.get("regress"),
    )

    regress_label_all = _binary(
        arrays["regress_label"], name="regress label", length=length
    ).astype(bool)
    recovery_subset = regress_mask & regress_label_all
    recovery_label = _binary(
        arrays["recovery_label"], name="recovery label", length=length
    )
    supervised_regress_mask = np.asarray(arrays["regress_mask"], dtype=bool)
    if np.any(
        supervised_regress_mask
        & (recovery_label > 0.5)
        & ~regress_label_all
    ):
        raise RuntimeError(
            "recovery=true must imply regress=true within the supervised mask"
        )
    recovery_probability = _probability(
        arrays["recovery_probability_given_regress"],
        name="conditional recovery probability", length=length,
    )
    recovery_baseline = _probability(
        arrays["recovery_baseline_probability_given_regress"],
        name="conditional recovery baseline", length=length,
    )
    conditional = _binary_domain(
        name="recovery_given_regress",
        labels=recovery_label[recovery_subset],
        probability=recovery_probability[recovery_subset],
        baseline_probability=recovery_baseline[recovery_subset],
        groups=groups[recovery_subset], folds=folds[recovery_subset],
        minimum_positive_per_fold=10, minimum_negative_per_fold=10, maximum_ece=0.10,
        weight_provenance=probability_weight_provenance.get("recovery_given_regress"),
    ) if recovery_subset.any() else {"status": "fail_closed_no_regress_rows", "passed": False}

    # The unconditional recovery probability uses the separately evaluated
    # regress probability; labels outside the supervised terminal mask do not
    # enter this assessment.
    unconditional_probability = _probability(
        arrays["regress_probability"], name="regress probability", length=length
    ) * recovery_probability
    unconditional_baseline = _probability(
        arrays["regress_baseline_probability"], name="regress baseline", length=length
    ) * recovery_baseline
    regress_provenance = regress.get("weight_provenance")
    conditional_provenance = conditional.get("weight_provenance")
    if regress_provenance is not None and conditional_provenance is not None:
        unconditional_baseline_prevalence = {
            fold: float(regress_provenance["folds"][str(fold)]["outer_training_prevalence"])
            * float(conditional_provenance["folds"][str(fold)]["outer_training_prevalence"])
            for fold in range(FOLD_COUNT)
        }
        unconditional = _binary_domain(
            name="unconditional_recovery",
            labels=recovery_label[regress_mask],
            probability=np.clip(unconditional_probability[regress_mask], EPS, 1.0 - EPS),
            baseline_probability=np.clip(unconditional_baseline[regress_mask], EPS, 1.0 - EPS),
            groups=groups[regress_mask], folds=folds[regress_mask],
            minimum_positive_per_fold=10, minimum_negative_per_fold=10, maximum_ece=0.10,
            weight_provenance=probability_weight_provenance.get("regress"),
            baseline_prevalence_by_fold=unconditional_baseline_prevalence,
            weight_provenance_head="regress",
        )
    else:
        unconditional = {
            "status": "fail_closed_missing_component_probability_provenance",
            "passed": False,
        }
    recovery = {
        "status": "passed" if conditional.get("passed") and unconditional.get("passed") else "fail_closed",
        "passed": bool(conditional.get("passed") and unconditional.get("passed")),
        "evaluation_subset": "ground_truth_regress_rows_only_for_conditional",
        "conditional": conditional,
        "unconditional": unconditional,
    }

    object_result = _object_domain(arrays, primary, groups, folds)
    domain_pass = {
        "duration": duration["passed"],
        "success": success["passed"],
        "regress": regress["passed"],
        "recovery_given_regress": recovery["passed"],
        "object": object_result["passed"],
    }
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "all_structured_domains_passed" if all(domain_pass.values()) else "fail_closed_one_or_more_domains",
        "development_only": True,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "fresh50_confirmation_authorized": False,
        "v7_implementation_changed": False,
        "action_selector_authorized": False,
        "evidence_design": evidence_design,
        "prospective_claim_allowed": evidence_design == (
            "prospective_preregistered_development"
        ),
        "evaluation_contract_sha256": contract_sha256,
        "oof_ownership": ownership,
        "cohorts": {
            "provenance_clean_primary": _descriptive_cohort_support(arrays, primary, groups),
            "historical_old100_overlap_descriptive_only": {
                **_descriptive_cohort_support(arrays, old100, groups),
                "metrics": _descriptive_cohort_metrics(arrays, old100),
            },
            "primary_excludes_all_old100_rows": True,
        },
        "domains": {
            "next_event": {
                "status": "frozen_bit_exact_only_predictive_accuracy_not_evaluated",
                "evaluated": False,
                "passed": None,
                "accuracy_not_re_evaluated": True,
                "frozen_bit_exact": ownership["next_event_frozen_bit_exact"],
            },
            "duration": duration,
            "success": success,
            "regress": regress,
            "recovery_given_regress": recovery,
            "object": object_result,
        },
        "domain_pass": domain_pass,
        "all_domain_pass": all(domain_pass.values()),
        "claim_scope": (
            "development_oof_only_not_prospective_confirmation"
            if evidence_design == "prospective_preregistered_development"
            else "adaptive_current_d250_development_only_never_prospective"
        ),
    }
    if evidence_design == "prospective_preregistered_development":
        result["preregistration_sha256"] = contract_sha256
    else:
        result["adaptive_development_contract_sha256"] = contract_sha256
    result["result_sha256"] = canonical_sha256(result)
    return result


def evaluate_structured_heads_arrays(
    arrays: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    fold_contracts: Mapping[int | str, Mapping[str, Any]],
    probability_weight_provenance: Mapping[
        str, Mapping[int | str, Mapping[str, Any]]
    ],
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate genuinely preregistered v8 OOF arrays."""

    validate_preregistration(preregistration)
    return _evaluate_structured_heads_arrays_validated(
        arrays,
        source_sha256=preregistration["source_sha256"],
        contract_sha256=str(preregistration["contract_sha256"]),
        evidence_design="prospective_preregistered_development",
        fold_contracts=fold_contracts,
        probability_weight_provenance=probability_weight_provenance,
        input_contract=input_contract,
    )


def evaluate_adaptive_development_structured_heads_arrays(
    arrays: Mapping[str, Any],
    *,
    adaptive_contract: Mapping[str, Any],
    fold_contracts: Mapping[int | str, Mapping[str, Any]],
    probability_weight_provenance: Mapping[
        str, Mapping[int | str, Mapping[str, Any]]
    ],
    input_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate current-D250 arrays without laundering them as prospective."""

    from openvla_etsf_v8_adaptive_development_protocol import (
        validate_adaptive_development_contract,
    )

    validate_adaptive_development_contract(adaptive_contract)
    return _evaluate_structured_heads_arrays_validated(
        arrays,
        source_sha256=adaptive_contract["source_sha256"],
        contract_sha256=str(adaptive_contract["contract_sha256"]),
        evidence_design="adaptive_current_d250_after_collection_started",
        fold_contracts=fold_contracts,
        probability_weight_provenance=probability_weight_provenance,
        input_contract=input_contract,
    )


def validate_result(result: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    validate_preregistration(preregistration)
    if result.get("format") != FORMAT:
        raise RuntimeError("v8 result format mismatch")
    unsigned = dict(result)
    digest = unsigned.pop("result_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise RuntimeError("v8 result signature mismatch")
    if result.get("preregistration_sha256") != preregistration.get("contract_sha256"):
        raise RuntimeError("v8 result is bound to a different preregistration")
    if result.get("evidence_design") != "prospective_preregistered_development" or result.get(
        "prospective_claim_allowed"
    ) is not True:
        raise RuntimeError("v8 prospective result evidence design mismatch")
    if result.get("fresh50_inputs_accepted") is not False or result.get(
        "fresh50_labels_read"
    ) is not False or result.get("fresh50_confirmation_authorized") is not False:
        raise RuntimeError("v8 result must not read or authorize Fresh50")
    if result.get("v7_implementation_changed") is not False or result.get(
        "action_selector_authorized"
    ) is not False:
        raise RuntimeError("v8 structured-head result cannot alter v7/selector")


def validate_adaptive_development_result(
    result: Mapping[str, Any], adaptive_contract: Mapping[str, Any]
) -> None:
    from openvla_etsf_v8_adaptive_development_protocol import (
        validate_adaptive_development_contract,
    )

    validate_adaptive_development_contract(adaptive_contract)
    if result.get("format") != FORMAT:
        raise RuntimeError("v8 adaptive result format mismatch")
    unsigned = dict(result)
    digest = unsigned.pop("result_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise RuntimeError("v8 adaptive result signature mismatch")
    if result.get("adaptive_development_contract_sha256") != adaptive_contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("v8 adaptive result is bound to a different contract")
    if result.get("evidence_design") != (
        "adaptive_current_d250_after_collection_started"
    ) or result.get("prospective_claim_allowed") is not False:
        raise RuntimeError("v8 adaptive result cannot claim prospective evidence")
    if result.get("claim_scope") != (
        "adaptive_current_d250_development_only_never_prospective"
    ):
        raise RuntimeError("v8 adaptive result claim scope was weakened")
    if result.get("fresh50_inputs_accepted") is not False or result.get(
        "fresh50_labels_read"
    ) is not False or result.get("fresh50_confirmation_authorized") is not False:
        raise RuntimeError("v8 adaptive result must not read or authorize Fresh50")
    if result.get("v7_implementation_changed") is not False or result.get(
        "action_selector_authorized"
    ) is not False:
        raise RuntimeError("v8 adaptive result cannot alter v7/selector")


__all__ = [
    "FORMAT",
    "evaluate_adaptive_development_structured_heads_arrays",
    "evaluate_structured_heads_arrays",
    "validate_adaptive_development_result",
    "validate_result",
]
