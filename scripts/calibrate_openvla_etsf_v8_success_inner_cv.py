#!/usr/bin/env python3
"""Leakage-safe group-inner-CV shrinkage calibration for the v8 success head.

This is an adaptive development-only analysis.  For every outer fold it trains
five success heads on group-disjoint subsets of that fold's outer-training
payload, using the same deterministic fixed-order AdamW update as the final v8
AdamW checkpoint.  Only those inner-OOF predictions select a preregistered
positive shrinkage coefficient.  The owner outer-holdout payload is loaded only
after all five signed selection contracts have been finalized.
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
from openvla_etsf_v8_structured_adapters import (
    V8DetachedStructuredAdapters,
    V8StructuredAdapterConfig,
    frozen_tensor_mapping_sha256,
    module_state_sha256,
)
from train_openvla_etsf_v8_structured_adapters import (
    V8_TRAINING_CHECKPOINT_FORMAT,
    load_authenticated_training_payload,
    structured_payload_sha256,
    validate_v8_training_payload,
)


FORMAT = "etsf_v8_success_group_inner_cv_shrinkage_oof_v1"
FOLD_CONTRACT_FORMAT = "etsf_v8_success_group_inner_cv_shrinkage_fold_v1"
PROTOCOL_FORMAT = "etsf_v8_success_group_inner_cv_shrinkage_protocol_v1"
MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
HOLDOUT_FORMAT = "etsf_v8_detached_adapter_holdout_input_v1"
FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
INNER_SPLIT_SEED = 20260827
ALPHA_GRID = (0.25, 0.50, 0.75, 1.0)
ECE_BINS = 10
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
MAXIMUM_ECE = 0.10
SELECTION_TOLERANCE = 1e-12
EPS = 1e-12
DEPLOYMENT_CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_group_list_sha256(groups: Sequence[str]) -> str:
    normalized = list(map(str, groups))
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("logical group lists must be sorted and unique")
    return canonical_sha256({"logical_groups": normalized})


def calibration_protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROTOCOL_FORMAT,
        "scope": "adaptive_development_only_no_fresh",
        "inner_fold_count": INNER_FOLD_COUNT,
        "inner_split": (
            "label_free_sha256_order_then_round_robin_by_logical_group"
        ),
        "inner_split_seed": INNER_SPLIT_SEED,
        "alpha_grid": list(ALPHA_GRID),
        "alpha_domain": "strictly_positive_and_at_most_one",
        "transform": "p_cal=train_prevalence+alpha*(p-train_prevalence)",
        "ranking_claim": (
            "strictly_positive_alpha_preserves_order_and_ties_within_each_owner_fold"
        ),
        "selection_criterion": (
            "minimum_inner_oof_equal_group_nll_then_equal_group_brier_then_largest_alpha"
        ),
        "selection_tolerance": SELECTION_TOLERANCE,
        "optimizer_match": (
            "same_fixed_record_order_epochs_lr_weight_decay_adamw_and_per_head_"
            "gradient_clip_as_final_outer_checkpoint"
        ),
        "success_loss": "unweighted_binary_cross_entropy_terminal_rows_only",
        "ece_bins": ECE_BINS,
        "bootstrap": {
            "resampling_unit": "logical_group",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "strict_probability_gates": {
            "brier_model_minus_owner_training_prevalence_ci_upper_lt_zero": True,
            "nll_model_minus_owner_training_prevalence_ci_upper_lt_zero": True,
            "average_precision_minus_evaluation_prevalence_ci_lower_gt_zero": True,
            "ece_10_equal_width_lte": MAXIMUM_ECE,
        },
        "outer_holdout_labels_used_for_alpha_selection": False,
        "fresh50_inputs_or_labels_used": False,
    }
    value["protocol_sha256"] = canonical_sha256(value)
    return value


def deterministic_inner_folds(
    logical_groups: Sequence[str], *, owner_fold_id: int
) -> list[list[str]]:
    groups = sorted(map(str, logical_groups))
    if len(groups) != len(set(groups)) or len(groups) < INNER_FOLD_COUNT:
        raise ValueError("inner CV needs unique groups and at least five groups")
    if owner_fold_id not in range(FOLD_COUNT):
        raise ValueError("owner_fold_id must be in [0,4]")
    namespace = (
        f"{PROTOCOL_FORMAT}|seed={INNER_SPLIT_SEED}|outer={owner_fold_id}|"
    )
    ordered = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{namespace}{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    folds = [sorted(ordered[index::INNER_FOLD_COUNT]) for index in range(INNER_FOLD_COUNT)]
    if sorted(group for fold in folds for group in fold) != groups:
        raise RuntimeError("inner group split lost or duplicated a logical group")
    return folds


def shrink_probabilities(
    probability: np.ndarray, *, prevalence: float | np.ndarray, alpha: float
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    prevalence = np.asarray(prevalence, dtype=np.float64)
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be strictly positive and at most one")
    if (
        probability.ndim != 1
        or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
        or not np.isfinite(prevalence).all()
        or np.any((prevalence <= 0.0) | (prevalence >= 1.0))
    ):
        raise ValueError("shrinkage inputs must be finite probabilities")
    result = prevalence + float(alpha) * (probability - prevalence)
    if not np.isfinite(result).all() or np.any((result <= 0.0) | (result >= 1.0)):
        raise RuntimeError("positive shrinkage unexpectedly left the open unit interval")
    return result


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives <= 0:
        raise ValueError("average precision needs at least one positive")
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


def _ece(labels: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    result = 0.0
    for index in range(ECE_BINS):
        lower = index / ECE_BINS
        upper = (index + 1) / ECE_BINS
        mask = (probability >= lower) & (
            probability <= upper if index == ECE_BINS - 1 else probability < upper
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask])) - float(np.mean(labels[mask]))
            )
    return float(result)


def _equal_group_mean(loss: np.ndarray, groups: np.ndarray) -> float:
    groups = np.asarray(groups).astype(str)
    unique = sorted(set(groups.tolist()))
    if not unique:
        raise ValueError("group-equal metric needs logical groups")
    return float(np.mean([np.mean(loss[groups == group]) for group in unique]))


def binary_probability_metrics(
    labels: np.ndarray, probability: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if (
        labels.ndim != 1
        or probability.shape != labels.shape
        or groups.shape != labels.shape
        or not np.isfinite(labels).all()
        or not np.isfinite(probability).all()
        or np.any((labels != 0.0) & (labels != 1.0))
        or np.any((probability <= 0.0) | (probability >= 1.0))
    ):
        raise ValueError("binary metric inputs must be aligned finite open probabilities")
    positive = int(labels.sum())
    if positive <= 0 or positive >= len(labels):
        raise ValueError("binary metrics require both classes")
    clipped = np.clip(probability, EPS, 1.0 - EPS)
    brier = (clipped - labels) ** 2
    nll = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return {
        "support": int(len(labels)),
        "positive": positive,
        "negative": int(len(labels) - positive),
        "prevalence": float(np.mean(labels)),
        "brier": float(np.mean(brier)),
        "equal_group_brier": _equal_group_mean(brier, groups),
        "nll": float(np.mean(nll)),
        "equal_group_nll": _equal_group_mean(nll, groups),
        "ece_10_equal_width": _ece(labels, clipped),
        "average_precision": _average_precision(labels > 0.5, clipped),
        "logical_groups": int(len(set(groups.tolist()))),
    }


def _cluster_bootstrap_probability_adequacy(
    labels: np.ndarray,
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fixed group-cluster bootstrap against owner-training prevalence."""

    labels = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    baseline_probability = np.asarray(baseline_probability, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if (
        labels.shape != probability.shape
        or labels.shape != baseline_probability.shape
        or labels.shape != groups.shape
        or np.any((baseline_probability <= 0.0) | (baseline_probability >= 1.0))
    ):
        raise ValueError("probability adequacy arrays are not aligned")
    unique = sorted(set(groups.tolist()))
    indices = [np.flatnonzero(groups == group) for group in unique]
    model_brier = (probability - labels) ** 2
    baseline_brier = (baseline_probability - labels) ** 2
    model_nll = -(
        labels * np.log(np.clip(probability, EPS, 1.0 - EPS))
        + (1.0 - labels) * np.log(np.clip(1.0 - probability, EPS, 1.0 - EPS))
    )
    baseline_nll = -(
        labels * np.log(np.clip(baseline_probability, EPS, 1.0 - EPS))
        + (1.0 - labels)
        * np.log(np.clip(1.0 - baseline_probability, EPS, 1.0 - EPS))
    )
    group_brier_delta = np.asarray(
        [np.mean(model_brier[index] - baseline_brier[index]) for index in indices]
    )
    group_nll_delta = np.asarray(
        [np.mean(model_nll[index] - baseline_nll[index]) for index in indices]
    )
    rng = np.random.default_rng(seed)
    brier_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    nll_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    ap_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.integers(0, len(unique), size=len(unique))
        brier_samples[sample_index] = float(
            np.mean(group_brier_delta[sampled_groups])
        )
        nll_samples[sample_index] = float(np.mean(group_nll_delta[sampled_groups]))
        sampled_rows = np.concatenate([indices[index] for index in sampled_groups])
        sampled_labels = labels[sampled_rows]
        if 0 < int(sampled_labels.sum()) < len(sampled_labels):
            ap_samples[sample_index] = _average_precision(
                sampled_labels > 0.5, probability[sampled_rows]
            ) - float(np.mean(sampled_labels))
        else:
            ap_samples[sample_index] = np.nan

    def comparison(observed: float, samples: np.ndarray, *, lower_gate: bool) -> dict[str, Any]:
        finite = samples[np.isfinite(samples)]
        if len(finite) < max(100, int(0.90 * BOOTSTRAP_SAMPLES)):
            return {
                "status": "not_evaluable_insufficient_two_class_bootstrap_samples",
                "observed": observed,
                "bootstrap_finite_samples": int(len(finite)),
                "bootstrap_95_ci": None,
                "strict_skill": False,
            }
        interval = np.quantile(finite, [0.025, 0.975]).tolist()
        return {
            "status": "complete",
            "observed": observed,
            "bootstrap_finite_samples": int(len(finite)),
            "bootstrap_95_ci": interval,
            "strict_skill": bool(interval[0] > 0.0 if lower_gate else interval[1] < 0.0),
        }

    ap_observed = _average_precision(labels > 0.5, probability) - float(
        np.mean(labels)
    )
    brier = comparison(float(np.mean(group_brier_delta)), brier_samples, lower_gate=False)
    nll = comparison(float(np.mean(group_nll_delta)), nll_samples, lower_gate=False)
    ap = comparison(ap_observed, ap_samples, lower_gate=True)
    brier.update(
        {
            "model": _equal_group_mean(model_brier, groups),
            "baseline": _equal_group_mean(baseline_brier, groups),
            "estimand": "equal_logical_group_mean_model_minus_baseline_loss",
        }
    )
    nll.update(
        {
            "model": _equal_group_mean(model_nll, groups),
            "baseline": _equal_group_mean(baseline_nll, groups),
            "estimand": "equal_logical_group_mean_model_minus_baseline_loss",
        }
    )
    ap.update(
        {
            "model_average_precision": _average_precision(
                labels > 0.5, probability
            ),
            "evaluation_prevalence": float(np.mean(labels)),
        }
    )
    ece_value = _ece(labels, probability)
    return {
        "baseline": "owner_outer_training_success_prevalence_by_prediction_fold",
        "groups": len(unique),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": int(seed),
        "brier_model_minus_baseline": brier,
        "nll_model_minus_baseline": nll,
        "average_precision_minus_evaluation_prevalence": ap,
        "ece_10_equal_width": ece_value,
        "maximum_ece_10_equal_width": MAXIMUM_ECE,
        "ece_strict_gate": bool(ece_value <= MAXIMUM_ECE),
        "strict_probability_adequacy": bool(
            brier["strict_skill"]
            and nll["strict_skill"]
            and ap["strict_skill"]
            and ece_value <= MAXIMUM_ECE
        ),
    }


def _success_rows(record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = record.get("batch")
    factual = record.get("factual_outputs")
    if not isinstance(batch, Mapping) or not isinstance(factual, Mapping):
        raise ValueError("success calibration record lacks batch/factual outputs")
    transition = factual.get("transition")
    terminal = batch.get("terminal_mask")
    success = batch.get("success")
    if (
        not torch.is_tensor(transition)
        or transition.ndim != 2
        or not torch.is_tensor(terminal)
        or not torch.is_tensor(success)
        or tuple(terminal.shape) != (len(transition),)
        or tuple(success.shape) != (len(transition),)
    ):
        raise ValueError("success calibration tensors are not aligned")
    mask = terminal.bool()
    if int(mask.sum()) != 4 or tuple(batch.get("candidate_names", ()))[:4] != (
        DEPLOYMENT_CANDIDATE_NAMES
    ):
        raise ValueError("success calibration requires the four deployment candidates")
    labels = success[mask].float()
    if bool(((labels != 0.0) & (labels != 1.0)).any()):
        raise ValueError("success labels must be binary")
    return transition[mask].detach(), labels.detach()


def _record_groups(records: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [str(record.get("logical_group_key", "")) for record in records]
    if any(not group for group in result) or len(result) != len(set(result)):
        raise ValueError("records must have unique nonempty logical_group_key values")
    return result


def _success_prevalence(records: Sequence[Mapping[str, Any]]) -> tuple[int, int, float]:
    labels = torch.cat([_success_rows(record)[1].cpu() for record in records])
    positive = int(labels.sum())
    support = int(len(labels))
    if positive <= 0 or positive >= support:
        raise ValueError("success AdamW initialization requires both classes")
    return support, positive, positive / support


def train_success_head_adamw(
    records: Sequence[Mapping[str, Any]],
    *,
    transition_dim: int,
    optimizer_contract: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    """Train exactly the final AdamW success-parameter update on a record subset."""

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for inner CV but unavailable")
    epochs = int(optimizer_contract.get("epochs", -1))
    learning_rate = float(optimizer_contract.get("learning_rate", math.nan))
    weight_decay = float(optimizer_contract.get("weight_decay", math.nan))
    maximum_gradient_norm = float(
        optimizer_contract.get("maximum_gradient_norm_per_probability_head", math.nan)
    )
    if (
        optimizer_contract.get("name") != "AdamW"
        or epochs <= 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
        or not math.isfinite(maximum_gradient_norm)
        or maximum_gradient_norm <= 0.0
    ):
        raise ValueError("inner CV optimizer contract is not fixed-order AdamW")
    groups = _record_groups(records)
    support, positive, prevalence = _success_prevalence(records)
    head = torch.nn.Linear(transition_dim, 1).to(device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_trace: list[float] = []
    factual_before = [
        frozen_tensor_mapping_sha256(record["factual_outputs"]) for record in records
    ]
    head.train()
    for _ in range(epochs):
        for record in records:
            feature, label = _success_rows(record)
            feature = feature.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                head(feature).squeeze(-1), label
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("inner success AdamW loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=maximum_gradient_norm
            )
            optimizer.step()
            loss_trace.append(float(loss.detach().cpu()))
    factual_after = [
        frozen_tensor_mapping_sha256(record["factual_outputs"]) for record in records
    ]
    if factual_before != factual_after:
        raise RuntimeError("inner success calibration mutated factual outputs")
    head.eval()
    return head, {
        "training_groups": groups,
        "training_groups_sha256": logical_group_list_sha256(sorted(groups)),
        "support": support,
        "positive": positive,
        "negative": support - positive,
        "prevalence": prevalence,
        "steps": epochs * len(records),
        "record_order_sha256": hashlib.sha256(
            "\n".join(groups).encode("utf-8")
        ).hexdigest(),
        "loss_trace_sha256": canonical_sha256(loss_trace),
        "success_head_state_sha256": module_state_sha256(head),
        "factual_outputs_bit_exact": True,
    }


def _predict_success(
    head: torch.nn.Linear,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(device)
    probability: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    candidate_indices: list[np.ndarray] = []
    with torch.no_grad():
        for record in records:
            feature, label = _success_rows(record)
            value = torch.sigmoid(
                head(feature.to(device=device, dtype=torch.float32)).squeeze(-1)
            )
            probability.append(value.detach().cpu().numpy().astype(np.float64))
            labels.append(label.cpu().numpy().astype(np.float64))
            groups.append(
                np.repeat(str(record["logical_group_key"]), len(label)).astype(str)
            )
            candidate_indices.append(np.arange(len(label), dtype=np.int64))
    return (
        np.concatenate(probability),
        np.concatenate(labels),
        np.concatenate(groups),
        np.concatenate(candidate_indices),
    )


def optimizer_contract_from_checkpoint(
    checkpoint: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    optimizer = checkpoint.get("optimizer")
    training = checkpoint.get("training_contract")
    last_step = checkpoint.get("last_step")
    groups = _record_groups(records)
    expected_record_order_sha256 = hashlib.sha256(
        "\n".join(groups).encode("utf-8")
    ).hexdigest()
    epochs = int(optimizer.get("epochs", -1)) if isinstance(optimizer, Mapping) else -1
    if (
        not isinstance(optimizer, Mapping)
        or optimizer.get("name") != "AdamW"
        or optimizer.get("initialization")
        != "zero_weights_outer_training_prevalence_biases"
        or optimizer.get("random_seed_used_for_adapter_initialization") is not None
        or list(map(str, optimizer.get("record_order", ()))) != groups
        or optimizer.get("record_order_sha256") != expected_record_order_sha256
        or checkpoint.get("steps") != epochs * len(records)
        or not isinstance(last_step, Mapping)
        or last_step.get("gradient_clip_scope")
        != "independent_per_probability_head"
        or not isinstance(training, Mapping)
        or training.get("success_loss") != "unweighted_binary_cross_entropy"
        or training.get("optimizer_parameter_scope")
        != "v8_adapter_parameters_exactly"
    ):
        raise ValueError("final checkpoint is not the fixed r4 AdamW contract")
    value = {
        "name": "AdamW",
        "epochs": epochs,
        "learning_rate": float(optimizer.get("learning_rate", math.nan)),
        "weight_decay": float(optimizer.get("weight_decay", math.nan)),
        "maximum_gradient_norm_per_probability_head": 1.0,
        "maximum_gradient_norm_provenance": (
            "frozen_v8_train_v8_adapter_one_step_default_1.0_with_checkpoint_"
            "last_step_independent_per_probability_head_scope"
        ),
        "final_checkpoint_steps": int(checkpoint["steps"]),
        "final_checkpoint_record_order_sha256": expected_record_order_sha256,
        "final_checkpoint_last_step_gradient_clip_scope": (
            "independent_per_probability_head"
        ),
        "record_order_policy": "signed_outer_training_payload_order_filtered_for_inner_train",
        "initialization": "zero_weight_inner_training_prevalence_bias",
        "success_loss": "unweighted_binary_cross_entropy_terminal_rows_only",
        "stochastic_initialization": False,
    }
    if (
        value["epochs"] <= 0
        or not math.isfinite(value["learning_rate"])
        or value["learning_rate"] <= 0.0
        or not math.isfinite(value["weight_decay"])
        or value["weight_decay"] < 0.0
    ):
        raise ValueError("final checkpoint AdamW hyperparameters are invalid")
    value["optimizer_contract_sha256"] = canonical_sha256(value)
    return value


def _choose_alpha(metrics_by_alpha: Mapping[str, Mapping[str, Any]]) -> float:
    candidates = []
    for alpha in ALPHA_GRID:
        metrics = metrics_by_alpha[f"{alpha:.2f}"]
        candidates.append(
            (
                float(metrics["equal_group_nll"]),
                float(metrics["equal_group_brier"]),
                -float(alpha),
                float(alpha),
            )
        )
    best_nll = min(value[0] for value in candidates)
    nll_tied = [value for value in candidates if value[0] <= best_nll + SELECTION_TOLERANCE]
    best_brier = min(value[1] for value in nll_tied)
    tied = [value for value in nll_tied if value[1] <= best_brier + SELECTION_TOLERANCE]
    return max(value[3] for value in tied)


def fit_outer_training_calibration_contract(
    records: Sequence[Mapping[str, Any]],
    *,
    owner_fold_id: int,
    outer_holdout_groups: Sequence[str],
    optimizer_contract: Mapping[str, Any],
    transition_dim: int,
    materialization_sha256: str,
    train_artifact_sha256: str,
    train_payload_sha256: str,
    final_checkpoint_sha256: str,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Select alpha without accepting or reading any outer-holdout payload."""

    protocol = calibration_protocol()
    if optimizer_contract.get("optimizer_contract_sha256") != canonical_sha256(
        {
            key: value
            for key, value in optimizer_contract.items()
            if key != "optimizer_contract_sha256"
        }
    ):
        raise ValueError("optimizer contract signature mismatch")
    for value in (
        materialization_sha256,
        train_artifact_sha256,
        train_payload_sha256,
        final_checkpoint_sha256,
    ):
        if not _is_sha256(value):
            raise ValueError("calibration source SHA is invalid")
    groups = _record_groups(records)
    group_to_record = dict(zip(groups, records))
    holdout = sorted(map(str, outer_holdout_groups))
    if len(holdout) != len(set(holdout)) or set(groups) & set(holdout):
        raise ValueError("outer training and holdout groups overlap")
    inner_folds = deterministic_inner_folds(groups, owner_fold_id=owner_fold_id)
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    prediction_groups: list[np.ndarray] = []
    source_prevalence: list[np.ndarray] = []
    inner_rows: list[dict[str, Any]] = []
    for inner_fold_id, validation_groups in enumerate(inner_folds):
        validation_set = set(validation_groups)
        training_records = [
            record for group, record in zip(groups, records) if group not in validation_set
        ]
        validation_records = [group_to_record[group] for group in validation_groups]
        training_groups = _record_groups(training_records)
        if set(training_groups) & validation_set:
            raise RuntimeError("inner training leaked an inner validation group")
        head, training_audit = train_success_head_adamw(
            training_records,
            transition_dim=transition_dim,
            optimizer_contract=optimizer_contract,
            device=device,
        )
        probability, target, row_groups, _ = _predict_success(
            head, validation_records, device=device
        )
        predictions.append(probability)
        labels.append(target)
        prediction_groups.append(row_groups)
        source_prevalence.append(
            np.full(len(target), training_audit["prevalence"], dtype=np.float64)
        )
        inner_rows.append(
            {
                "inner_fold_id": inner_fold_id,
                "training_groups": training_groups,
                "training_groups_sha256": logical_group_list_sha256(
                    sorted(training_groups)
                ),
                "validation_groups": validation_groups,
                "validation_groups_sha256": logical_group_list_sha256(
                    validation_groups
                ),
                "training_audit": training_audit,
                "inner_validation_labels_used_for_training": False,
            }
        )
    probability = np.concatenate(predictions)
    target = np.concatenate(labels)
    row_groups = np.concatenate(prediction_groups)
    prevalence_by_row = np.concatenate(source_prevalence)
    if set(row_groups.tolist()) != set(groups) or len(row_groups) != 4 * len(groups):
        raise RuntimeError("inner OOF predictions do not cover outer-training groups once")
    raw_metrics = binary_probability_metrics(target, probability, row_groups)
    metrics_by_alpha: dict[str, dict[str, Any]] = {}
    for alpha in ALPHA_GRID:
        adjusted = shrink_probabilities(
            probability, prevalence=prevalence_by_row, alpha=alpha
        )
        metrics_by_alpha[f"{alpha:.2f}"] = binary_probability_metrics(
            target, adjusted, row_groups
        )
    chosen_alpha = _choose_alpha(metrics_by_alpha)
    support, positive, outer_prevalence = _success_prevalence(records)
    contract: dict[str, Any] = {
        "format": FOLD_CONTRACT_FORMAT,
        "status": "selected_from_group_disjoint_inner_oof_only",
        "scope": "adaptive_development_only",
        "owner_fold_id": owner_fold_id,
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "optimizer_contract": dict(optimizer_contract),
        "optimizer_contract_sha256": optimizer_contract[
            "optimizer_contract_sha256"
        ],
        "materialization_sha256": materialization_sha256,
        "train_artifact_sha256": train_artifact_sha256,
        "train_payload_sha256": train_payload_sha256,
        "final_outer_checkpoint_sha256": final_checkpoint_sha256,
        "outer_training_groups": groups,
        "outer_training_groups_sha256": logical_group_list_sha256(sorted(groups)),
        "outer_holdout_groups": holdout,
        "outer_holdout_groups_sha256": logical_group_list_sha256(holdout),
        "inner_folds": inner_rows,
        "inner_oof_support": int(len(target)),
        "inner_oof_logical_groups": len(groups),
        "inner_oof_raw_metrics": raw_metrics,
        "alpha_metrics": metrics_by_alpha,
        "chosen_alpha": chosen_alpha,
        "outer_training_success_support": support,
        "outer_training_success_positive": positive,
        "outer_training_success_prevalence": outer_prevalence,
        "selection_completed_before_outer_holdout_payload_loaded": True,
        "outer_holdout_labels_used_for_alpha_selection": False,
        "fresh50_inputs_or_labels_used": False,
    }
    contract["calibration_contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_calibration_contract(contract: Mapping[str, Any]) -> None:
    unsigned = dict(contract)
    recorded = unsigned.pop("calibration_contract_sha256", None)
    protocol = contract.get("protocol")
    optimizer = contract.get("optimizer_contract")
    outer_training = list(map(str, contract.get("outer_training_groups", ())))
    outer_holdout = list(map(str, contract.get("outer_holdout_groups", ())))
    inner_rows = contract.get("inner_folds")
    if (
        contract.get("format") != FOLD_CONTRACT_FORMAT
        or recorded != canonical_sha256(unsigned)
        or not isinstance(protocol, Mapping)
        or protocol.get("protocol_sha256")
        != contract.get("protocol_sha256")
        or protocol.get("protocol_sha256")
        != calibration_protocol()["protocol_sha256"]
        or contract.get("chosen_alpha") not in ALPHA_GRID
        or contract.get("outer_holdout_labels_used_for_alpha_selection") is not False
        or contract.get("fresh50_inputs_or_labels_used") is not False
        or contract.get("owner_fold_id") not in range(FOLD_COUNT)
        or not isinstance(optimizer, Mapping)
        or optimizer.get("optimizer_contract_sha256")
        != contract.get("optimizer_contract_sha256")
        or optimizer.get("optimizer_contract_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in optimizer.items()
                if key != "optimizer_contract_sha256"
            }
        )
        or not outer_training
        or len(outer_training) != len(set(outer_training))
        or not outer_holdout
        or len(outer_holdout) != len(set(outer_holdout))
        or set(outer_training) & set(outer_holdout)
        or contract.get("outer_training_groups_sha256")
        != logical_group_list_sha256(sorted(outer_training))
        or contract.get("outer_holdout_groups_sha256")
        != logical_group_list_sha256(sorted(outer_holdout))
        or not isinstance(inner_rows, Sequence)
        or isinstance(inner_rows, (str, bytes))
        or len(inner_rows) != INNER_FOLD_COUNT
        or float(contract.get("chosen_alpha", math.nan))
        != _choose_alpha(contract.get("alpha_metrics", {}))
    ):
        raise ValueError("success calibration contract signature/protocol changed")
    validation_union: set[str] = set()
    for inner_fold_id, row in enumerate(inner_rows):
        if not isinstance(row, Mapping):
            raise ValueError("success calibration inner fold contract changed")
        training = list(map(str, row.get("training_groups", ())))
        validation = list(map(str, row.get("validation_groups", ())))
        if (
            row.get("inner_fold_id") != inner_fold_id
            or set(training) & set(validation)
            or set(training) | set(validation) != set(outer_training)
            or row.get("training_groups_sha256")
            != logical_group_list_sha256(sorted(training))
            or row.get("validation_groups_sha256")
            != logical_group_list_sha256(sorted(validation))
            or row.get("inner_validation_labels_used_for_training") is not False
        ):
            raise ValueError("success calibration inner fold contract changed")
        validation_union.update(validation)
    if validation_union != set(outer_training):
        raise ValueError("success calibration inner validation coverage changed")


def _load_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if "fresh" in str(path).lower():
        raise ValueError("success calibration refuses Fresh inputs")
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    recorded = unsigned.pop("materialization_sha256", None)
    folds = value.get("folds")
    if (
        value.get("format") != MATERIALIZATION_FORMAT
        or value.get("status") != "complete_development_only"
        or value.get("fresh_confirmation_data_or_labels_read") is not False
        or recorded != canonical_sha256(unsigned)
        or not isinstance(folds, list)
        or [row.get("outer_fold_id") for row in folds] != list(range(FOLD_COUNT))
    ):
        raise ValueError("materialization manifest is not a signed development bundle")
    return dict(value)


def _load_final_checkpoint(
    path: Path,
    *,
    owner_fold_id: int,
    payload: Mapping[str, Any],
    artifact_authentication: Mapping[str, Any],
) -> tuple[dict[str, Any], V8DetachedStructuredAdapters, dict[str, Any]]:
    path = path.resolve()
    if "fresh" in str(path).lower() or not path.is_file():
        raise ValueError("final AdamW checkpoint path is invalid or Fresh")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("final AdamW checkpoint must be a mapping")
    _, records, provenance = validate_v8_training_payload(payload)
    if (
        checkpoint.get("format") != V8_TRAINING_CHECKPOINT_FORMAT
        or checkpoint.get("fresh_confirmation_data_or_labels_read") is not False
        or checkpoint.get("authorization_guard_changed") is not False
        or checkpoint.get("all_steps_factual_inputs_bit_exact") is not True
        or checkpoint.get("provenance") != provenance
        or checkpoint.get("input_artifact_authentication")
        != dict(artifact_authentication)
        or provenance.get("outer_fold_id") != owner_fold_id
    ):
        raise ValueError("final AdamW checkpoint provenance/authentication mismatch")
    config = V8StructuredAdapterConfig.from_dict(checkpoint.get("config", {}))
    adapters = V8DetachedStructuredAdapters(config).eval()
    adapters.load_state_dict(checkpoint.get("state_dict", {}), strict=True)
    if module_state_sha256(adapters) != checkpoint.get("adapter_state_sha256"):
        raise ValueError("final AdamW checkpoint adapter state SHA mismatch")
    optimizer_contract = optimizer_contract_from_checkpoint(checkpoint, records)
    return dict(checkpoint), adapters, optimizer_contract


def _load_holdout(
    path: Path,
    *,
    owner_fold_id: int,
    manifest_fold: Mapping[str, Any],
    train_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.resolve()
    if "fresh" in str(path).lower() or str(path) != str(
        Path(str(manifest_fold.get("holdout_artifact", ""))).resolve()
    ):
        raise ValueError("holdout path is not the owner development artifact")
    if sha256_path(path) != manifest_fold.get("holdout_artifact_sha256"):
        raise ValueError("owner holdout artifact SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(value, Mapping)
        or value.get("format") != HOLDOUT_FORMAT
        or value.get("payload_sha256") != manifest_fold.get("holdout_payload_sha256")
        or value.get("payload_sha256") != structured_payload_sha256(value)
    ):
        raise ValueError("owner holdout payload signature/format mismatch")
    expected_provenance = {
        **dict(train_provenance),
        "split_role": "outer_holdout_evaluation_only",
        "holdout_labels_used_for_duration_or_object_fit": False,
        "holdout_labels_present_only_in_separate_artifact": True,
    }
    if value.get("provenance") != expected_provenance:
        raise ValueError("owner holdout provenance differs from outer training")
    records = value.get("batches")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("owner holdout records are invalid")
    groups = _record_groups(records)
    if sorted(groups) != sorted(map(str, manifest_fold.get("oof_holdout_groups", ()))):
        raise ValueError("owner holdout group ownership mismatch")
    for record in records:
        factual = record.get("factual_outputs")
        if (
            record.get("split_role") != "outer_holdout"
            or record.get("outer_fold_id") != owner_fold_id
            or not isinstance(factual, Mapping)
            or record.get("factual_outputs_require_grad") is not False
            or record.get("factual_outputs_sha256")
            != frozen_tensor_mapping_sha256(factual)
            or any(
                torch.is_tensor(tensor) and tensor.requires_grad
                for tensor in factual.values()
            )
        ):
            raise ValueError("owner holdout factual record authentication failed")
        _success_rows(record)
    return dict(value)


def evaluate_outer_holdout(
    adapters: V8DetachedStructuredAdapters,
    holdout_records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_calibration_contract(contract)
    adapters = adapters.to(device).eval()
    probability, labels, groups, candidate_index = _predict_success(
        adapters.success_head, holdout_records, device=device
    )
    alpha = float(contract["chosen_alpha"])
    calibrated = shrink_probabilities(
        probability,
        prevalence=float(contract["outer_training_success_prevalence"]),
        alpha=alpha,
    )
    raw_metrics = binary_probability_metrics(labels, probability, groups)
    calibrated_metrics = binary_probability_metrics(labels, calibrated, groups)
    if raw_metrics["average_precision"] != calibrated_metrics["average_precision"]:
        raise RuntimeError("positive within-fold shrinkage changed success ranking")
    baseline_probability = np.full(
        len(labels),
        float(contract["outer_training_success_prevalence"]),
        dtype=np.float64,
    )
    adequacy = _cluster_bootstrap_probability_adequacy(
        labels,
        calibrated,
        baseline_probability,
        groups,
        seed=BOOTSTRAP_SEED + int(contract["owner_fold_id"]),
    )
    report = {
        "owner_fold_id": int(contract["owner_fold_id"]),
        "calibration_contract_sha256": contract["calibration_contract_sha256"],
        "chosen_alpha": alpha,
        "outer_training_success_prevalence": contract[
            "outer_training_success_prevalence"
        ],
        "uncalibrated": raw_metrics,
        "calibrated": calibrated_metrics,
        "calibrated_minus_uncalibrated": {
            "brier": calibrated_metrics["brier"] - raw_metrics["brier"],
            "nll": calibrated_metrics["nll"] - raw_metrics["nll"],
            "ece_10_equal_width": calibrated_metrics["ece_10_equal_width"]
            - raw_metrics["ece_10_equal_width"],
            "average_precision": calibrated_metrics["average_precision"]
            - raw_metrics["average_precision"],
        },
        "within_fold_ranking_and_ties_preserved": True,
        "calibrated_probability_adequacy": adequacy,
        "outer_holdout_used_for_alpha_selection": False,
    }
    return report, {
        "labels": labels,
        "groups": groups,
        "candidate_index": candidate_index,
        "uncalibrated": probability,
        "calibrated": calibrated,
        "baseline": baseline_probability,
    }


def run_oof_calibration(
    *,
    checkpoint_paths: Sequence[Path],
    materialization_manifest_path: Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    if len(checkpoint_paths) != FOLD_COUNT:
        raise ValueError("success calibration requires five ordered checkpoints")
    manifest_path = materialization_manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    prepared: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []

    # Phase one deliberately never deserializes an owner holdout payload.
    for owner_fold_id, checkpoint_path in enumerate(checkpoint_paths):
        row = manifest["folds"][owner_fold_id]
        train_path = Path(str(row["train_artifact"])).resolve()
        payload, authentication = load_authenticated_training_payload(
            input_path=train_path,
            materialization_manifest_path=manifest_path,
            outer_fold_id=owner_fold_id,
        )
        config, records, provenance = validate_v8_training_payload(payload)
        checkpoint, adapters, optimizer_contract = _load_final_checkpoint(
            Path(checkpoint_path),
            owner_fold_id=owner_fold_id,
            payload=payload,
            artifact_authentication=authentication,
        )
        checkpoint_sha = sha256_path(Path(checkpoint_path).resolve())
        contract = fit_outer_training_calibration_contract(
            records,
            owner_fold_id=owner_fold_id,
            outer_holdout_groups=row["oof_holdout_groups"],
            optimizer_contract=optimizer_contract,
            transition_dim=config.transition_dim,
            materialization_sha256=manifest["materialization_sha256"],
            train_artifact_sha256=row["train_artifact_sha256"],
            train_payload_sha256=row["train_payload_sha256"],
            final_checkpoint_sha256=checkpoint_sha,
            device=device,
        )
        contracts.append(contract)
        prepared.append(
            {
                "adapters": adapters,
                "train_provenance": provenance,
                "checkpoint": checkpoint,
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
            }
        )

    # Phase two starts only after every alpha is selected and contract-signed.
    fold_reports: list[dict[str, Any]] = []
    pooled_parts: dict[str, list[np.ndarray]] = {
        "labels": [],
        "groups": [],
        "candidate_index": [],
        "uncalibrated": [],
        "calibrated": [],
        "baseline": [],
    }
    calibrated_oof_rows: list[dict[str, Any]] = []
    for owner_fold_id in range(FOLD_COUNT):
        row = manifest["folds"][owner_fold_id]
        holdout_path = Path(str(row["holdout_artifact"])).resolve()
        holdout = _load_holdout(
            holdout_path,
            owner_fold_id=owner_fold_id,
            manifest_fold=row,
            train_provenance=prepared[owner_fold_id]["train_provenance"],
        )
        report, arrays = evaluate_outer_holdout(
            prepared[owner_fold_id]["adapters"],
            holdout["batches"],
            contracts[owner_fold_id],
            device=device,
        )
        report.update(
            {
                "checkpoint": prepared[owner_fold_id]["checkpoint_path"],
                "checkpoint_sha256": contracts[owner_fold_id][
                    "final_outer_checkpoint_sha256"
                ],
                "holdout_artifact": str(holdout_path),
                "holdout_artifact_sha256": row["holdout_artifact_sha256"],
                "holdout_payload_sha256": row["holdout_payload_sha256"],
            }
        )
        fold_reports.append(report)
        for name, value in arrays.items():
            pooled_parts[name].append(value)
        for row_index in range(len(arrays["labels"])):
            candidate_index = int(arrays["candidate_index"][row_index])
            calibrated_oof_rows.append(
                {
                    "owner_fold_id": owner_fold_id,
                    "logical_group": str(arrays["groups"][row_index]),
                    "candidate_index": candidate_index,
                    "candidate_name": DEPLOYMENT_CANDIDATE_NAMES[candidate_index],
                    "success_label": int(arrays["labels"][row_index]),
                    "uncalibrated_success_probability": float(
                        arrays["uncalibrated"][row_index]
                    ),
                    "calibrated_success_probability": float(
                        arrays["calibrated"][row_index]
                    ),
                    "owner_training_prevalence_baseline": float(
                        arrays["baseline"][row_index]
                    ),
                    "calibration_contract_sha256": contracts[owner_fold_id][
                        "calibration_contract_sha256"
                    ],
                }
            )
    pooled = {name: np.concatenate(value) for name, value in pooled_parts.items()}
    raw_pooled = binary_probability_metrics(
        pooled["labels"], pooled["uncalibrated"], pooled["groups"]
    )
    calibrated_pooled = binary_probability_metrics(
        pooled["labels"], pooled["calibrated"], pooled["groups"]
    )
    pooled_adequacy = _cluster_bootstrap_probability_adequacy(
        pooled["labels"],
        pooled["calibrated"],
        pooled["baseline"],
        pooled["groups"],
        seed=BOOTSTRAP_SEED,
    )
    protocol = calibration_protocol()
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete_adaptive_development_only",
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_path(Path(__file__).resolve()),
        "materialization_manifest": str(manifest_path),
        "materialization_sha256": manifest["materialization_sha256"],
        "fold_calibration_contracts": contracts,
        "outer_holdout_evaluation": {
            "folds": fold_reports,
            "pooled_oof": {
                "uncalibrated": raw_pooled,
                "calibrated": calibrated_pooled,
                "calibrated_minus_uncalibrated": {
                    "brier": calibrated_pooled["brier"] - raw_pooled["brier"],
                    "nll": calibrated_pooled["nll"] - raw_pooled["nll"],
                    "ece_10_equal_width": calibrated_pooled[
                        "ece_10_equal_width"
                    ]
                    - raw_pooled["ece_10_equal_width"],
                    "average_precision": calibrated_pooled["average_precision"]
                    - raw_pooled["average_precision"],
                },
                "pooled_cross_fold_ap_may_change_because_each_owner_uses_its_own_"
                "training_prevalence": True,
                "calibrated_probability_adequacy": pooled_adequacy,
            },
        },
        "calibrated_oof_rows": calibrated_oof_rows,
        "calibrated_oof_rows_sha256": canonical_sha256(calibrated_oof_rows),
        "calibrated_oof_row_count": len(calibrated_oof_rows),
        "calibrated_oof_alignment": (
            "owner_fold_logical_group_candidate_index_candidate_name"
        ),
        "action_ranking_preserved_within_each_group": True,
        "task_success_cannot_change_from_uncalibrated_argmax": True,
        "calibration_claim_scope": (
            "probability_adequacy_only_not_action_selection_or_task_success_improvement"
        ),
        "all_alpha_selection_completed_before_holdout_payload_deserialization": True,
        "outer_holdout_labels_used_for_alpha_selection": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "authorization": {
            "selector_authorized": False,
            "deployment_authorized": False,
            "fresh50_confirmation_authorized": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if "fresh" in str(path).lower():
        raise ValueError("success calibration refuses Fresh output paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        # A hard-link publication is atomic and, unlike os.replace, fails if a
        # concurrent or previous run already owns the immutable final name.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = run_oof_calibration(
        checkpoint_paths=args.checkpoint,
        materialization_manifest_path=args.materialization_manifest,
        device=args.device,
    )
    _atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(args.output.resolve()),
                "result_sha256": value["result_sha256"],
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ALPHA_GRID",
    "FOLD_CONTRACT_FORMAT",
    "binary_probability_metrics",
    "calibration_protocol",
    "deterministic_inner_folds",
    "evaluate_outer_holdout",
    "fit_outer_training_calibration_contract",
    "optimizer_contract_from_checkpoint",
    "run_oof_calibration",
    "shrink_probabilities",
    "train_success_head_adamw",
    "validate_calibration_contract",
]
