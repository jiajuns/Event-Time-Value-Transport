#!/usr/bin/env python3
"""Strict validation-only evaluation for the five-member canonical ensemble.

The evaluator authenticates the complete training bundle and reconstructs the
label-free group split from the frozen Stage1/OpenVLA inputs.  Only validation
descriptors are materialized.  Train and sealed-test HDF5 payloads are never
opened, and Fresh/confirmation namespaces are rejected by the shared protocol
guard.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_multibody_canonical_event_world_model import (
    CANONICAL_EVENTS,
    FORMAT as TRAINING_FORMAT,
    GroupDescriptor,
    InputBinding,
    ModelConfig,
    MultibodyCanonicalEventWorldModel,
    TransitionDataset,
    action_normalization_arrays,
    body_alias_receipt,
    canonical_json_sha256,
    collate_rows,
    evaluate_train_only_baselines,
    load_rows,
    reject_forbidden_path,
    scan_schema5_groups,
    scan_stage1_groups,
    sha256_file,
    split_receipt,
    strict_group_split,
    verify_input_bindings,
)


EVALUATION_FORMAT = "etsf_multibody_canonical_ensemble_validation_v2"
ENSEMBLE_SIZE = 5
EPSILON = 1e-12
VARIANCE_IDENTITY_RTOL = 1e-10
VARIANCE_IDENTITY_ATOL = 1e-12


class EvaluationError(RuntimeError):
    """A fail-closed bundle, protocol, or metric validation failure."""


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read {label} JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{label} must contain a JSON object")
    return value


def _require_signed_mapping(
    value: Any, *, signature_key: str, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{label} must be a mapping")
    unsigned = dict(value)
    signature = unsigned.pop(signature_key, None)
    if not _is_sha256(signature) or signature != canonical_json_sha256(unsigned):
        raise EvaluationError(f"{label} SHA-256 signature mismatch")
    return value


def _assert_json_equal(left: Any, right: Any, label: str) -> None:
    if canonical_json_sha256(left) != canonical_json_sha256(right):
        raise EvaluationError(f"{label} differs across the frozen bundle")


def _assert_json_close(left: Any, right: Any, label: str) -> None:
    """Recursively compare recomputed floating metrics without hiding drift."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise EvaluationError(f"{label} keys differ")
        for key in left:
            _assert_json_close(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise EvaluationError(f"{label} lengths differ")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_json_close(left_item, right_item, f"{label}[{index}]")
        return
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
        right, (int, float)
    ) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-9):
            raise EvaluationError(f"{label} numeric value differs")
        return
    if left != right:
        raise EvaluationError(f"{label} differs")


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # torch emits several version-dependent types.
        raise EvaluationError(f"cannot safely load checkpoint: {path}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError("checkpoint must contain a mapping")
    return value


def _validate_summary_semantics(summary: Mapping[str, Any]) -> None:
    if summary.get("format") != TRAINING_FORMAT:
        raise EvaluationError("training summary format changed")
    if summary.get("status") != "training_complete":
        raise EvaluationError("training summary is not complete")
    if summary.get("sealed_test_evaluated") is not False:
        raise EvaluationError("training summary does not preserve the sealed test")
    if int(summary.get("test_group_hdf5_opened", -1)) != 0:
        raise EvaluationError("training summary reports test HDF5 access")
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("format") != TRAINING_FORMAT:
        raise EvaluationError("training summary protocol is missing or invalid")
    if int(protocol.get("sealed_test_group_hdf5_opened", -1)) != 0:
        raise EvaluationError("training protocol reports sealed-test HDF5 access")
    if protocol.get("labels_used_for_split") is not False:
        raise EvaluationError("training split was not explicitly label-free")
    if protocol.get("test_transition_count") != "unknown_not_loaded":
        raise EvaluationError("training protocol materialized test transitions")
    if int(protocol.get("test_group_hdf5_opened", -1)) != 0:
        raise EvaluationError("training protocol reports test group HDF5 access")


def authenticate_training_bundle(
    training_summary: Path,
    expected_summary_sha256: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], dict[str, Any]]:
    """Authenticate one summary and exactly five immutable best checkpoints."""

    summary_path = reject_forbidden_path(training_summary, "training summary")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not _is_sha256(expected_summary_sha256):
        raise EvaluationError("expected training-summary SHA-256 is malformed")
    actual_summary_sha = sha256_file(summary_path)
    if actual_summary_sha != expected_summary_sha256:
        raise EvaluationError("training-summary SHA-256 mismatch")
    summary = _read_json(summary_path, "training summary")
    _validate_summary_semantics(summary)

    members = summary.get("members")
    if not isinstance(members, list) or len(members) != ENSEMBLE_SIZE:
        raise EvaluationError("summary must contain exactly five members")
    ordered = sorted(members, key=lambda item: int(item.get("member", -1)))
    if [int(item.get("member", -1)) for item in ordered] != list(range(ENSEMBLE_SIZE)):
        raise EvaluationError("ensemble member ids must be exactly 0..4")
    seeds = [int(item.get("seed", -1)) for item in ordered]
    if len(set(seeds)) != ENSEMBLE_SIZE:
        raise EvaluationError("ensemble seeds must be unique")

    protocol = summary["protocol"]
    expected_config: Mapping[str, Any] | None = None
    checkpoints: list[Mapping[str, Any]] = []
    checkpoint_receipts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for position, member in enumerate(ordered):
        raw_path = member.get("checkpoint")
        if not isinstance(raw_path, str) or not raw_path:
            raise EvaluationError("summary member lacks a checkpoint path")
        path = reject_forbidden_path(Path(raw_path), f"member {position} checkpoint")
        if path in seen_paths:
            raise EvaluationError("duplicate checkpoint path")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_sha = member.get("checkpoint_sha256")
        if not _is_sha256(expected_sha):
            raise EvaluationError("summary checkpoint SHA-256 is malformed")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise EvaluationError(f"member {position} checkpoint SHA-256 mismatch")
        checkpoint = _safe_torch_load(path)
        if checkpoint.get("format") != TRAINING_FORMAT:
            raise EvaluationError("checkpoint format changed")
        if int(checkpoint.get("member", -1)) != position:
            raise EvaluationError("checkpoint member id differs from summary")
        if int(checkpoint.get("seed", -1)) != seeds[position]:
            raise EvaluationError("checkpoint seed differs from summary")
        if int(checkpoint.get("step", -1)) != int(member.get("best_step", -2)):
            raise EvaluationError("checkpoint best step differs from summary")
        if not math.isclose(
            float(checkpoint.get("selection_score", math.nan)),
            float(member.get("best_validation_selection_score", math.inf)),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise EvaluationError("checkpoint selection score differs from summary")
        _assert_json_equal(
            checkpoint.get("validation"), member.get("best_validation"),
            "best validation metrics",
        )
        _assert_json_equal(checkpoint.get("contract"), protocol, "checkpoint contract")

        config = checkpoint.get("config")
        if not isinstance(config, Mapping):
            raise EvaluationError("checkpoint model config is missing")
        try:
            validated_config = dataclasses.asdict(ModelConfig(**dict(config)))
        except (TypeError, ValueError) as error:
            raise EvaluationError("checkpoint model config is invalid") from error
        _assert_json_equal(config, validated_config, "checkpoint model config")
        if expected_config is None:
            expected_config = config
        else:
            _assert_json_equal(config, expected_config, "model config")

        normalization = checkpoint.get("action_normalization")
        _assert_json_equal(
            normalization, protocol.get("action_normalization"),
            "action normalization contract",
        )
        try:
            action_mean, action_std = action_normalization_arrays(normalization)
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError("action normalization receipt is invalid") from error
        state = checkpoint.get("model")
        if not isinstance(state, Mapping) or not state or not all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise EvaluationError("checkpoint model state is not tensor-only")
        for key, expected in (
            ("action.action_mean", action_mean),
            ("action.action_std", action_std),
        ):
            tensor = state.get(key)
            if not isinstance(tensor, torch.Tensor) or not torch.equal(
                tensor.detach().cpu(), torch.from_numpy(expected)
            ):
                raise EvaluationError(f"checkpoint {key} differs from normalization")

        baseline = checkpoint.get("train_only_baselines")
        _require_signed_mapping(
            baseline, signature_key="sha256", label="train-only baseline"
        )
        _assert_json_equal(
            baseline, protocol.get("train_only_baselines"), "train-only baseline"
        )
        _assert_json_equal(
            checkpoint.get("validation_baseline_metrics"),
            protocol.get("validation_baseline_metrics"),
            "validation baseline metrics",
        )
        _assert_json_equal(
            checkpoint.get("validation_selection_rule"),
            protocol.get("validation_selection_rule"),
            "validation selection rule",
        )
        checkpoints.append(checkpoint)
        checkpoint_receipts.append(
            {
                "member": position,
                "seed": seeds[position],
                "path": str(path),
                "sha256": actual_sha,
                "best_step": int(checkpoint["step"]),
            }
        )
    assert expected_config is not None
    return summary, checkpoints, {
        "training_summary": str(summary_path),
        "training_summary_sha256": actual_summary_sha,
        "checkpoints": checkpoint_receipts,
        "checkpoint_bundle_sha256": canonical_json_sha256(checkpoint_receipts),
        "model_config": dict(expected_config),
        "contract_sha256": canonical_json_sha256(protocol),
        "action_normalization_sha256": protocol["action_normalization"]["sha256"],
        "train_only_baseline_sha256": protocol["train_only_baselines"]["sha256"],
    }


def binding_from_paths(
    *,
    protocol: Mapping[str, Any],
    stage1_root: Path,
    stage1_source_manifest: Path,
    stage1_target_manifest: Path,
    event_spec: Path,
    openvla_schema5_manifest: Path,
) -> InputBinding:
    input_sha = protocol.get("input_sha256")
    if not isinstance(input_sha, Mapping):
        raise EvaluationError("training protocol lacks input SHA-256 bindings")
    required = (
        "stage1_source_manifest",
        "stage1_target_manifest",
        "event_spec",
        "openvla_schema5_manifest",
    )
    if any(not _is_sha256(input_sha.get(name)) for name in required):
        raise EvaluationError("training protocol input SHA-256 binding is malformed")
    if input_sha["event_spec"] != protocol.get("event_spec_sha256"):
        raise EvaluationError("training protocol event-spec SHA bindings disagree")
    return InputBinding(
        stage1_root=stage1_root,
        stage1_source_manifest=stage1_source_manifest,
        stage1_source_manifest_sha256=input_sha["stage1_source_manifest"],
        stage1_target_manifest=stage1_target_manifest,
        stage1_target_manifest_sha256=input_sha["stage1_target_manifest"],
        event_spec=event_spec,
        event_spec_sha256=input_sha["event_spec"],
        openvla_schema5_manifest=openvla_schema5_manifest,
        openvla_schema5_manifest_sha256=input_sha["openvla_schema5_manifest"],
    )


def reconstruct_frozen_split(
    binding: InputBinding,
    *,
    split_seed: int,
    expected_protocol: Mapping[str, Any],
) -> tuple[dict[str, list[GroupDescriptor]], dict[str, Any]]:
    """Rebind manifests and split identities without opening any group HDF5."""

    audit = verify_input_bindings(binding)
    descriptors = scan_stage1_groups(binding) + scan_schema5_groups(binding)
    splits = strict_group_split(descriptors, split_seed=split_seed)
    reconstructed_split = split_receipt(splits)
    for name in ("train", "validation", "test"):
        for suffix in ("groups", "identity_sha256"):
            key = f"{name}_{suffix}"
            if reconstructed_split.get(key) != expected_protocol.get(key):
                raise EvaluationError(f"reconstructed {key} differs from training")
    if expected_protocol.get("labels_used_for_split") is not False:
        raise EvaluationError("expected protocol does not guarantee label-free split")
    aliases = body_alias_receipt(descriptors)
    _assert_json_equal(aliases, expected_protocol.get("body_alias"), "body aliases")
    if len(descriptors) != int(expected_protocol.get("total_groups", -1)):
        raise EvaluationError("reconstructed total group count differs from training")
    for key, value in audit.items():
        if key in expected_protocol and expected_protocol[key] != value:
            raise EvaluationError(f"reconstructed protocol field {key} differs")

    body_to_id = {
        name: index for index, name in enumerate(sorted({item.body for item in descriptors}))
    }
    _assert_json_equal(
        body_to_id, expected_protocol.get("body_to_id"), "canonical body mapping"
    )
    return splits, {
        "input_sha256": audit["input_sha256"],
        "split_seed": int(split_seed),
        "split": reconstructed_split,
        "body_alias": aliases,
        "body_to_id": body_to_id,
        "total_groups": len(descriptors),
        "validation_group_identities": sorted(
            item.logical_group for item in splits["validation"]
        ),
    }


def load_validation_only(
    splits: Mapping[str, Sequence[GroupDescriptor]],
    event_spec: Mapping[str, Any],
    *,
    row_loader: Callable[
        [Sequence[GroupDescriptor], Mapping[str, Any]], list[dict[str, Any]]
    ] = load_rows,
) -> list[dict[str, Any]]:
    """The sole payload-loading gateway; train/test descriptors cannot enter it."""

    validation = list(splits["validation"])
    forbidden = {
        item.logical_group for split in ("train", "test") for item in splits[split]
    }
    if any(item.logical_group in forbidden for item in validation):
        raise EvaluationError("validation payload selection overlaps train/test")
    return row_loader(validation, event_spec)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, str]:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if not len(positives) or not len(negatives):
        return None, "unavailable_single_class"
    comparison = positives[:, None] - negatives[None, :]
    value = ((comparison > 0).sum() + 0.5 * (comparison == 0).sum()) / comparison.size
    return float(value), "available"


def _macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if not len(labels):
        return None
    scores = []
    for class_id in range(len(CANONICAL_EVENTS)):
        true_positive = int(((labels == class_id) & (predictions == class_id)).sum())
        false_positive = int(((labels != class_id) & (predictions == class_id)).sum())
        false_negative = int(((labels == class_id) & (predictions != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator:
            scores.append(2.0 * true_positive / denominator)
    return float(np.mean(scores)) if scores else None


def _summary_statistics(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"mean": None, "median": None, "p90": None}
    if not np.isfinite(values).all():
        raise EvaluationError("uncertainty contains non-finite values")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _variance_identity_audit(
    total: np.ndarray, expected: np.ndarray, equation: str
) -> dict[str, Any]:
    """Audit a variance identity after promoting member outputs to float64."""

    total64 = np.asarray(total, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    residual = np.abs(total64 - expected64)
    maximum = float(np.max(residual)) if residual.size else 0.0
    reference = float(np.max(np.abs(expected64))) if expected64.size else 0.0
    threshold = VARIANCE_IDENTITY_ATOL + VARIANCE_IDENTITY_RTOL * reference
    return {
        "verified": bool(maximum <= threshold),
        "equation": equation,
        "member_outputs_promoted_to": "float64_before_moment_computation",
        "max_abs_residual": maximum,
        "absolute_tolerance": VARIANCE_IDENTITY_ATOL,
        "relative_tolerance": VARIANCE_IDENTITY_RTOL,
        "effective_max_tolerance": float(threshold),
    }


def _event_metrics(
    member_probability: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    support = int(len(labels))
    if not support:
        return {
            "status": "unavailable_no_support",
            "support": 0,
            "accuracy": None,
            "macro_f1": None,
            "nll": None,
            "class_counts": [0] * len(CANONICAL_EVENTS),
            "uncertainty": None,
        }
    ensemble = member_probability.mean(axis=0)
    prediction = ensemble.argmax(axis=-1)
    selected = ensemble[np.arange(support), labels]
    member_prediction = member_probability.argmax(axis=-1)
    total = _entropy(ensemble)
    aleatoric = _entropy(member_probability).mean(axis=0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    per_member_disagreement = np.mean(
        member_prediction != prediction[None, :], axis=1
    )
    return {
        "status": "available",
        "support": support,
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": _macro_f1(labels, prediction),
        "nll": float(np.mean(-np.log(np.clip(selected, EPSILON, 1.0)))),
        "class_counts": np.bincount(
            labels, minlength=len(CANONICAL_EVENTS)
        ).tolist(),
        "uncertainty": {
            "units": "nats_predictive_entropy",
            "member_disagreement_rate": float(
                np.mean(member_prediction != prediction[None, :])
            ),
            "per_member_disagreement_rate": per_member_disagreement.tolist(),
            "epistemic": _summary_statistics(epistemic),
            "aleatoric": _summary_statistics(aleatoric),
            "total": _summary_statistics(total),
        },
    }


def _expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float | None:
    if not len(labels):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            selected = (scores >= edges[index]) & (scores < edges[index + 1])
        if selected.any():
            result += float(selected.mean()) * abs(
                float(scores[selected].mean()) - float(labels[selected].mean())
            )
    return float(result)


def _success_metrics(
    member_probability: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    support = int(len(labels))
    if not support:
        return {
            "status": "unavailable_no_support",
            "support": {"rows": 0, "positive": 0, "negative": 0},
            "auroc": None,
            "auroc_status": "unavailable_no_support",
            "brier": None,
            "ece_10_bin": None,
            "uncertainty": None,
            "error_detection": {
                "auroc": None,
                "status": "unavailable_no_support",
            },
        }
    member_probability = np.asarray(member_probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    probability = member_probability.mean(axis=0)
    prediction = probability >= 0.5
    member_prediction = member_probability >= 0.5
    epistemic = member_probability.var(axis=0)
    aleatoric = np.mean(member_probability * (1.0 - member_probability), axis=0)
    total = epistemic + aleatoric
    variance_audit = _variance_identity_audit(
        total,
        probability * (1.0 - probability),
        "E[p(1-p)] + Var[p] = E[p](1-E[p])",
    )
    auc, auc_status = _binary_auc(labels, probability)
    errors = (prediction != (labels > 0.5)).astype(np.float64)
    error_auc, error_status = _binary_auc(errors, total)
    return {
        "status": "available",
        "support": {
            "rows": support,
            "positive": int((labels > 0.5).sum()),
            "negative": int((labels <= 0.5).sum()),
        },
        "auroc": auc,
        "auroc_status": auc_status,
        "brier": float(np.mean(np.square(probability - labels))),
        "ece_10_bin": _expected_calibration_error(labels, probability),
        "uncertainty": {
            "units": "Bernoulli_probability_variance",
            "member_disagreement_rate": float(
                np.mean(member_prediction != prediction[None, :])
            ),
            "per_member_disagreement_rate": np.mean(
                member_prediction != prediction[None, :], axis=1
            ).tolist(),
            "epistemic": _summary_statistics(epistemic),
            "aleatoric": _summary_statistics(aleatoric),
            "total": _summary_statistics(total),
            "law_of_total_variance_verified": variance_audit["verified"],
            "law_of_total_variance_audit": variance_audit,
        },
        "error_detection": {
            "score": "total_predictive_variance",
            "error_definition": "threshold_0.5_misclassification",
            "errors": int(errors.sum()),
            "correct": int(len(errors) - errors.sum()),
            "auroc": error_auc,
            "status": error_status,
        },
    }


def _duration_moments(
    member_log_mean: np.ndarray, member_log_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    member_log_mean = np.asarray(member_log_mean, dtype=np.float64)
    member_log_scale = np.asarray(member_log_scale, dtype=np.float64)
    variance_log = np.exp(2.0 * np.clip(member_log_scale, -20.0, 20.0))
    mean_exponent = np.clip(member_log_mean + 0.5 * variance_log, -50.0, 50.0)
    mean = np.maximum(np.exp(mean_exponent) - 1.0, 0.0)
    variance_exponent = np.clip(
        2.0 * member_log_mean + variance_log, -50.0, 50.0
    )
    variance = np.maximum(
        np.expm1(np.clip(variance_log, 0.0, 50.0))
        * np.exp(variance_exponent),
        0.0,
    )
    return mean, variance


def _standard_normal_cdf(value: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float64))
    return (0.5 * torch.erfc(-tensor / math.sqrt(2.0))).numpy()


def equal_weight_lognormal_mixture_median(
    member_log_mean: np.ndarray,
    member_log_scale: np.ndarray,
    *,
    iterations: int = 96,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the label-independent median of a shifted-lognormal mixture.

    The physical duration boundary is zero.  We solve in ``log1p(duration)``
    space, where the equal-weight mixture CDF is monotone and each component is
    Gaussian.  No observed duration label is accepted by this function.
    """

    means = np.asarray(member_log_mean, dtype=np.float64)
    log_scales = np.asarray(member_log_scale, dtype=np.float64)
    if (
        means.ndim != 2
        or log_scales.shape != means.shape
        or means.shape[0] != ENSEMBLE_SIZE
    ):
        raise EvaluationError("duration mixture median requires [5,N] parameters")
    if iterations < 64:
        raise EvaluationError("duration mixture median requires at least 64 iterations")
    if not np.isfinite(means).all() or not np.isfinite(log_scales).all():
        raise EvaluationError("duration mixture parameters must be finite")
    # The model head is contractually clamped to [-5,2].  Reject values outside
    # that contract instead of silently changing the predictive distribution.
    if np.any(log_scales < -5.000001) or np.any(log_scales > 2.000001):
        raise EvaluationError("duration log scale violates the model contract")
    scales = np.exp(log_scales)
    rows = means.shape[1]
    if rows == 0:
        return np.empty(0, dtype=np.float64), {
            "method": "equal_weight_mixture_cdf_bisection_in_log1p_space",
            "labels_used": False,
            "iterations": int(iterations),
            "lower_zero_boundary_rows": 0,
            "upper_finite_boundary_rows": 0,
            "max_nonboundary_cdf_error": None,
        }

    def mixture_cdf(log_duration: np.ndarray) -> np.ndarray:
        standardized = (log_duration[None, :] - means) / scales
        return _standard_normal_cdf(standardized).mean(axis=0)

    lower = np.zeros(rows, dtype=np.float64)
    lower_cdf = mixture_cdf(lower)
    lower_boundary = lower_cdf >= 0.5
    # This common upper point is at least twelve standard deviations above
    # every component mean, so it is a strict finite CDF bracket.
    upper = np.maximum(
        0.0,
        np.max(means, axis=0) + 12.0 * np.max(scales, axis=0),
    )
    upper_cdf = mixture_cdf(upper)
    if not np.isfinite(upper).all() or np.any(upper_cdf < 0.5):
        raise EvaluationError("duration mixture median could not form a finite CDF bracket")
    active = ~lower_boundary
    for _ in range(iterations):
        midpoint = lower + 0.5 * (upper - lower)
        below = mixture_cdf(midpoint) < 0.5
        lower = np.where(active & below, midpoint, lower)
        upper = np.where(active & ~below, midpoint, upper)
    median_log_duration = np.where(lower_boundary, 0.0, lower + 0.5 * (upper - lower))

    maximum_log_duration = math.log(np.finfo(np.float64).max)
    upper_finite_boundary = median_log_duration >= maximum_log_duration
    safe_log_duration = np.minimum(
        median_log_duration, np.nextafter(maximum_log_duration, -math.inf)
    )
    prediction = np.expm1(safe_log_duration)
    prediction[upper_finite_boundary] = np.finfo(np.float64).max
    prediction = np.maximum(prediction, 0.0)
    if not np.isfinite(prediction).all():
        raise EvaluationError("duration mixture median is not finite")
    nonboundary = ~(lower_boundary | upper_finite_boundary)
    root_cdf = mixture_cdf(median_log_duration)
    max_cdf_error = (
        float(np.max(np.abs(root_cdf[nonboundary] - 0.5)))
        if nonboundary.any()
        else None
    )
    return prediction, {
        "method": "equal_weight_mixture_cdf_bisection_in_log1p_space",
        "mixture_members": ENSEMBLE_SIZE,
        "mixture_weights": [1.0 / ENSEMBLE_SIZE] * ENSEMBLE_SIZE,
        "target_quantile": 0.5,
        "labels_used": False,
        "physical_lower_duration_boundary": 0.0,
        "iterations": int(iterations),
        "upper_bracket_component_sigmas": 12.0,
        "lower_zero_boundary_rows": int(lower_boundary.sum()),
        "upper_finite_boundary_rows": int(upper_finite_boundary.sum()),
        "max_nonboundary_cdf_error": max_cdf_error,
    }


def _point_prediction_diagnostics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"support": 0, "mean": None, "median": None, "p90": None, "max": None}
    if not np.isfinite(array).all():
        raise EvaluationError("duration point prediction contains non-finite values")
    return {
        "support": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(np.max(array)),
    }


def _logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (
        np.squeeze(maximum, axis=axis)
        + np.log(np.mean(np.exp(values - maximum), axis=axis))
    )


def _duration_metrics(
    member_log_mean: np.ndarray,
    member_log_scale: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    support = int(len(labels))
    if not support:
        return {
            "status": "unavailable_no_observed_support",
            "support": 0,
            "mae": None,
            "mixture_median_mae": None,
            "mixture_mean_mae_heavy_tail_diagnostic": None,
            "mixture_nll": None,
            "prediction_diagnostics": None,
            "uncertainty": None,
        }
    member_log_mean = np.asarray(member_log_mean, dtype=np.float64)
    member_log_scale = np.asarray(member_log_scale, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    member_mean, member_variance = _duration_moments(
        member_log_mean, member_log_scale
    )
    ensemble_mean = member_mean.mean(axis=0)
    ensemble_median, median_solver = equal_weight_lognormal_mixture_median(
        member_log_mean, member_log_scale
    )
    aleatoric = member_variance.mean(axis=0)
    epistemic = member_mean.var(axis=0)
    total = aleatoric + epistemic
    variance_audit = _variance_identity_audit(
        total,
        member_variance.mean(axis=0) + member_mean.var(axis=0),
        "E[Var(D|member)] + Var(E[D|member]) = Var(D)",
    )
    transformed = np.log1p(np.maximum(labels, 0.0))[None, :]
    scale = np.exp(np.clip(member_log_scale, -20.0, 20.0))
    component_log_density = (
        -0.5 * np.square((transformed - member_log_mean) / scale)
        - np.log(scale)
        - 0.5 * math.log(2.0 * math.pi)
        - transformed
    )
    mixture_nll = -_logmeanexp(component_log_density, axis=0)
    member_disagreement = np.sqrt(epistemic)
    median_mae = float(np.mean(np.abs(ensemble_median - labels)))
    mean_mae = float(np.mean(np.abs(ensemble_mean - labels)))
    return {
        "status": "available",
        "support": support,
        "mae": median_mae,
        "mixture_median_mae": median_mae,
        "mixture_mean_mae_heavy_tail_diagnostic": mean_mae,
        "mixture_nll": float(np.mean(mixture_nll)),
        "prediction": "five_component_equal_weight_lognormal_mixture_median",
        "primary_mae_point_prediction": "mixture_median_label_independent",
        "prediction_diagnostics": {
            "mixture_median": _point_prediction_diagnostics(ensemble_median),
            "mixture_mean_heavy_tail_diagnostic": _point_prediction_diagnostics(
                ensemble_mean
            ),
            "mean_to_median_average_prediction_ratio": float(
                np.mean(ensemble_mean) / max(np.mean(ensemble_median), EPSILON)
            ),
            "median_solver": median_solver,
        },
        "uncertainty": {
            "units": "duration_squared",
            "member_disagreement_std": _summary_statistics(member_disagreement),
            "per_member_rmse_from_ensemble_mean": np.sqrt(
                np.mean(np.square(member_mean - ensemble_mean[None, :]), axis=1)
            ).tolist(),
            "epistemic": _summary_statistics(epistemic),
            "aleatoric": _summary_statistics(aleatoric),
            "total": _summary_statistics(total),
            "law_of_total_variance_verified": variance_audit["verified"],
            "law_of_total_variance_audit": variance_audit,
        },
    }


def _object_metrics(
    member_mean: np.ndarray,
    member_log_scale: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    support = int(len(labels))
    if not support:
        return {
            "status": "unavailable_no_support",
            "support": 0,
            "rmse": None,
            "nll": None,
            "uncertainty": None,
        }
    member_mean = np.asarray(member_mean, dtype=np.float64)
    member_log_scale = np.asarray(member_log_scale, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    ensemble_mean = member_mean.mean(axis=0)
    member_variance = np.exp(2.0 * np.clip(member_log_scale, -20.0, 20.0))
    aleatoric_element = member_variance.mean(axis=0)
    epistemic_element = member_mean.var(axis=0)
    total_element = aleatoric_element + epistemic_element
    scale = np.exp(np.clip(member_log_scale, -20.0, 20.0))
    component_log_density = (
        -0.5 * np.square((labels[None, :, :] - member_mean) / scale)
        - np.log(scale)
        - 0.5 * math.log(2.0 * math.pi)
    )
    # One member is one multivariate mixture component.  Keep the member index
    # shared across dimensions, then normalize the joint NLL per dimension to
    # preserve the trainer's reported NLL scale.
    mixture_nll = -_logmeanexp(component_log_density.sum(axis=-1), axis=0) / labels.shape[-1]
    epistemic = epistemic_element.mean(axis=-1)
    aleatoric = aleatoric_element.mean(axis=-1)
    total = total_element.mean(axis=-1)
    variance_audit = _variance_identity_audit(
        total,
        aleatoric + epistemic,
        "mean_d(E[Var(X_d|member)] + Var(E[X_d|member])) = total",
    )
    return {
        "status": "available",
        "support": support,
        "rmse": float(np.sqrt(np.mean(np.square(labels - ensemble_mean)))),
        "nll": float(np.mean(mixture_nll)),
        "prediction": "five_component_diagonal_gaussian_mixture",
        "uncertainty": {
            "units": "mean_squared_object_delta",
            "member_disagreement_rmse": _summary_statistics(np.sqrt(epistemic)),
            "per_member_rmse_from_ensemble_mean": np.sqrt(
                np.mean(
                    np.square(member_mean - ensemble_mean[None, :, :]), axis=(1, 2)
                )
            ).tolist(),
            "epistemic": _summary_statistics(epistemic),
            "aleatoric": _summary_statistics(aleatoric),
            "total": _summary_statistics(total),
            "law_of_total_variance_verified": variance_audit["verified"],
            "law_of_total_variance_audit": variance_audit,
        },
    }


@torch.no_grad()
def collect_ensemble_predictions(
    models: Sequence[MultibodyCanonicalEventWorldModel],
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    if len(models) != ENSEMBLE_SIZE:
        raise EvaluationError("prediction requires exactly five models")
    for model in models:
        model.eval()
    row_parts: dict[str, list[np.ndarray]] = {}
    member_parts: dict[str, list[np.ndarray]] = {}
    for raw in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()
        }
        outputs = [model(batch) for model in models]
        row_values = {
            "body_id": batch["body_id"],
            "post_label": batch["post_event_id"],
            "post_mask": batch["post_event_mask"],
            "next_label": batch["next_event_id"],
            "next_mask": batch["next_event_mask"],
            "success_label": batch["success"],
            "success_mask": batch["success_mask"],
            "duration_label": batch["duration"],
            "duration_observed": batch["duration_observed"],
            "duration_mask": batch["duration_mask"],
            "object_label": batch["object_delta"],
            "object_mask": batch["object_delta_mask"] * batch["action_available"],
        }
        member_values = {
            "post_probability": torch.stack(
                [torch.softmax(item["post_event_logits"], dim=-1) for item in outputs]
            ),
            "next_probability": torch.stack(
                [torch.softmax(item["next_event_logits"], dim=-1) for item in outputs]
            ),
            "success_probability": torch.stack(
                [torch.sigmoid(item["success_logit"]) for item in outputs]
            ),
            "duration_log_mean": torch.stack(
                [item["duration_selected_log_mean"] for item in outputs]
            ),
            "duration_log_scale": torch.stack(
                [item["duration_selected_log_scale"] for item in outputs]
            ),
            "object_mean": torch.stack(
                [item["object_delta_mean"] for item in outputs]
            ),
            "object_log_scale": torch.stack(
                [item["object_delta_log_scale"] for item in outputs]
            ),
        }
        for key, value in row_values.items():
            row_parts.setdefault(key, []).append(value.detach().cpu().numpy())
        for key, value in member_values.items():
            member_parts.setdefault(key, []).append(value.detach().cpu().numpy())
    if not row_parts:
        raise EvaluationError("validation loader produced no rows")
    result = {
        key: np.concatenate(parts, axis=0) for key, parts in row_parts.items()
    }
    result.update(
        {key: np.concatenate(parts, axis=1) for key, parts in member_parts.items()}
    )
    return result


def compute_ensemble_metrics(
    values: Mapping[str, np.ndarray], row_selection: np.ndarray | None = None
) -> dict[str, Any]:
    rows = len(values["body_id"])
    selected = np.ones(rows, dtype=bool) if row_selection is None else np.asarray(
        row_selection, dtype=bool
    )
    if selected.shape != (rows,):
        raise EvaluationError("metric row selection has the wrong shape")
    for name, value in values.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise EvaluationError(f"prediction array {name} contains non-finite values")

    def mask(name: str) -> np.ndarray:
        return selected & (np.asarray(values[name]) > 0.5)

    post = mask("post_mask")
    next_event = mask("next_mask")
    success = mask("success_mask")
    observed_duration = (
        mask("duration_mask") & (np.asarray(values["duration_observed"]) > 0.5)
    )
    objects = mask("object_mask")
    return {
        "rows": int(selected.sum()),
        "post_event": _event_metrics(
            values["post_probability"][:, post],
            values["post_label"][post].astype(np.int64),
        ),
        "next_event": _event_metrics(
            values["next_probability"][:, next_event],
            values["next_label"][next_event].astype(np.int64),
        ),
        "success": _success_metrics(
            values["success_probability"][:, success],
            values["success_label"][success].astype(np.float64),
        ),
        "observed_duration": _duration_metrics(
            values["duration_log_mean"][:, observed_duration],
            values["duration_log_scale"][:, observed_duration],
            values["duration_label"][observed_duration].astype(np.float64),
        ),
        "duration_support": {
            "observed": int(observed_duration.sum()),
            "censored": int(
                (selected & (values["duration_mask"] > 0.5)
                 & ~(values["duration_observed"] > 0.5)).sum()
            ),
        },
        "object": _object_metrics(
            values["object_mean"][:, objects],
            values["object_log_scale"][:, objects],
            values["object_label"][objects].astype(np.float64),
        ),
    }


def _relative_gain(higher_is_better: bool, value: Any, baseline: Any) -> dict[str, Any]:
    if value is None or baseline is None:
        return {"value": None, "status": "unavailable_missing_metric"}
    value_float = float(value)
    baseline_float = float(baseline)
    if not math.isfinite(value_float) or not math.isfinite(baseline_float):
        return {"value": None, "status": "unavailable_non_finite"}
    numerator = (
        value_float - baseline_float
        if higher_is_better
        else baseline_float - value_float
    )
    return {
        "value": float(numerator / max(abs(baseline_float), 1e-12)),
        "absolute_change": float(value_float - baseline_float),
        "status": "available",
        "positive_means_ensemble_improved": True,
    }


def compare_to_train_only_baseline(
    metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    post_f1 = metrics["post_event"]["macro_f1"]
    post_baseline_f1 = baseline["post_event"]["macro_f1"]
    next_f1 = metrics["next_event"]["macro_f1"]
    next_baseline_f1 = baseline["next_event"]["macro_f1"]
    return {
        "scope": "global_validation",
        "baseline_source": baseline.get("source"),
        "post_event_accuracy_relative_gain": _relative_gain(
            True, metrics["post_event"]["accuracy"], baseline["post_event"]["accuracy"]
        ),
        "post_event_macro_f1_relative_gain": _relative_gain(
            True, post_f1, post_baseline_f1
        ),
        "post_event_macro_error_relative_reduction": _relative_gain(
            False,
            None if post_f1 is None else 1.0 - float(post_f1),
            None if post_baseline_f1 is None else 1.0 - float(post_baseline_f1),
        ),
        "next_event_accuracy_relative_gain": _relative_gain(
            True, metrics["next_event"]["accuracy"], baseline["next_event"]["accuracy"]
        ),
        "next_event_macro_f1_relative_gain": _relative_gain(
            True, next_f1, next_baseline_f1
        ),
        "next_event_macro_error_relative_reduction": _relative_gain(
            False,
            None if next_f1 is None else 1.0 - float(next_f1),
            None if next_baseline_f1 is None else 1.0 - float(next_baseline_f1),
        ),
        "success_brier_relative_reduction": _relative_gain(
            False, metrics["success"]["brier"], baseline["success_brier"]
        ),
        "success_auroc_relative_gain": _relative_gain(
            True, metrics["success"]["auroc"], baseline["success_auroc"]
        ),
        "observed_duration_mae_relative_reduction": _relative_gain(
            False,
            metrics["observed_duration"]["mae"],
            baseline["observed_duration_mae"],
        ),
        "object_rmse_relative_reduction": _relative_gain(
            False, metrics["object"]["rmse"], baseline["object_rmse"]
        ),
        "object_nll_relative_reduction": _relative_gain(
            False, metrics["object"]["nll"], baseline["object_nll"]
        ),
    }


def run_synthetic_smoke() -> dict[str, Any]:
    """Exercise every ensemble metric on CPU without reading or writing data."""

    rows = 10
    labels = np.arange(rows) % len(CANONICAL_EVENTS)
    event_probability = np.full(
        (ENSEMBLE_SIZE, rows, len(CANONICAL_EVENTS)), 0.025, dtype=np.float64
    )
    for member in range(ENSEMBLE_SIZE):
        event_probability[member, np.arange(rows), labels] = 0.9
        event_probability[member] /= event_probability[member].sum(
            axis=-1, keepdims=True
        )
    success_labels = (np.arange(rows) % 2).astype(np.float64)
    success_probability = np.tile(
        np.where(success_labels > 0.5, 0.8, 0.2)[None, :],
        (ENSEMBLE_SIZE, 1),
    )
    success_probability[:, :2] = 1.0 - success_probability[:, :2]
    success_probability += np.linspace(-0.04, 0.04, ENSEMBLE_SIZE)[:, None]
    duration = np.linspace(1.0, 10.0, rows)
    duration_log_mean = np.tile(
        np.log1p(duration)[None, :], (ENSEMBLE_SIZE, 1)
    ) + np.linspace(-0.1, 0.1, ENSEMBLE_SIZE)[:, None]
    objects = np.linspace(-1.0, 1.0, rows * 6).reshape(rows, 6)
    object_mean = np.tile(objects[None, :, :], (ENSEMBLE_SIZE, 1, 1))
    object_mean += np.linspace(-0.1, 0.1, ENSEMBLE_SIZE)[:, None, None]
    values = {
        "body_id": np.arange(rows) % 2,
        "post_label": labels,
        "post_mask": np.ones(rows),
        "next_label": labels,
        "next_mask": np.ones(rows),
        "success_label": success_labels,
        "success_mask": np.ones(rows),
        "duration_label": duration,
        "duration_observed": np.ones(rows),
        "duration_mask": np.ones(rows),
        "object_label": objects,
        "object_mask": np.ones(rows),
        "post_probability": event_probability,
        "next_probability": event_probability.copy(),
        "success_probability": np.clip(success_probability, 0.01, 0.99),
        "duration_log_mean": duration_log_mean,
        "duration_log_scale": np.vstack(
            [
                np.full((1, rows), 1.5),
                np.full((ENSEMBLE_SIZE - 1, rows), -2.0),
            ]
        ),
        "object_mean": object_mean,
        "object_log_scale": np.full((ENSEMBLE_SIZE, rows, 6), -1.0),
    }
    metrics = compute_ensemble_metrics(values)
    if (
        metrics["post_event"]["nll"] is None
        or metrics["success"]["error_detection"]["status"] != "available"
        or metrics["observed_duration"]["uncertainty"][
            "law_of_total_variance_verified"
        ]
        is not True
        or metrics["object"]["uncertainty"]["law_of_total_variance_verified"]
        is not True
        or metrics["success"]["uncertainty"]["law_of_total_variance_verified"]
        is not True
        or metrics["observed_duration"][
            "mixture_mean_mae_heavy_tail_diagnostic"
        ]
        <= metrics["observed_duration"]["mixture_median_mae"]
        or metrics["observed_duration"]["prediction_diagnostics"]["median_solver"][
            "labels_used"
        ]
        is not False
    ):
        raise EvaluationError("synthetic ensemble metric smoke failed")
    return {
        "status": "synthetic_smoke_passed",
        "device": "cpu_numpy",
        "members": ENSEMBLE_SIZE,
        "rows": rows,
        "all_required_metric_families_available": True,
        "success_error_detection_status": metrics["success"]["error_detection"][
            "status"
        ],
        "duration_total_variance_verified": True,
        "object_total_variance_verified": True,
        "success_total_variance_verified": True,
        "heavy_tail_mean_mae_exceeds_median_mae": True,
    }


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one JSON file atomically, without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("evaluation receipt must be a new immutable path")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    required = (
        "training_summary",
        "training_summary_sha256",
        "stage1_root",
        "stage1_source_manifest",
        "stage1_target_manifest",
        "event_spec",
        "openvla_schema5_manifest",
        "output",
    )
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise EvaluationError(f"evaluate mode requires arguments: {missing}")
    output = reject_forbidden_path(args.output, "evaluation output")
    if output.exists():
        raise FileExistsError("evaluation output must be a new immutable path")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA requested but unavailable")

    summary, checkpoints, bundle = authenticate_training_bundle(
        args.training_summary, args.training_summary_sha256
    )
    protocol = summary["protocol"]
    binding = binding_from_paths(
        protocol=protocol,
        stage1_root=args.stage1_root,
        stage1_source_manifest=args.stage1_source_manifest,
        stage1_target_manifest=args.stage1_target_manifest,
        event_spec=args.event_spec,
        openvla_schema5_manifest=args.openvla_schema5_manifest,
    )
    splits, reconstructed = reconstruct_frozen_split(
        binding, split_seed=args.split_seed, expected_protocol=protocol
    )

    # Prove every tensor key/shape against the one shared config before any
    # validation payload is opened.
    body_to_id = reconstructed["body_to_id"]
    config = ModelConfig(**bundle["model_config"])
    if config.body_count != len(body_to_id):
        raise EvaluationError("model body count differs from canonical body mapping")
    device = torch.device(args.device)
    models = []
    for member, checkpoint in enumerate(checkpoints):
        model = MultibodyCanonicalEventWorldModel(config)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as error:
            raise EvaluationError(f"member {member} model state is incompatible") from error
        models.append(model.to(device).eval())

    event_spec = _read_json(
        reject_forbidden_path(args.event_spec, "event spec"), "event spec"
    )
    validation_rows = load_validation_only(splits, event_spec)
    if len(validation_rows) != int(protocol.get("validation_transitions", -1)):
        raise EvaluationError("validation transition count differs from training")

    dataset = TransitionDataset(validation_rows, body_to_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_rows,
    )
    values = collect_ensemble_predictions(models, loader, device)
    global_metrics = compute_ensemble_metrics(values)
    per_body = {
        body: compute_ensemble_metrics(values, values["body_id"] == body_id)
        for body, body_id in sorted(body_to_id.items(), key=lambda item: item[1])
    }

    train_baseline = protocol["train_only_baselines"]
    validation_baseline = evaluate_train_only_baselines(
        train_baseline, validation_rows
    )
    _assert_json_close(
        validation_baseline,
        protocol["validation_baseline_metrics"],
        "independently recomputed validation baseline",
    )
    validation_hdf5_count = sum(
        int(item.source in {"stage1_source", "stage1_target", "openvla_schema5"})
        for item in splits["validation"]
    )
    receipt: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "status": "validation_ensemble_evaluation_complete",
        "bundle_authentication": bundle,
        "protocol_reconstruction": reconstructed,
        "data_access": {
            "split_membership_labels_used": False,
            "loaded_splits": ["validation"],
            "train_rows_loaded": 0,
            "train_hdf5_files_opened": 0,
            "validation_rows_loaded": len(validation_rows),
            "validation_group_payloads_loaded": len(splits["validation"]),
            "validation_hdf5_files_opened": validation_hdf5_count,
            "test_rows_loaded": 0,
            "test_labels_used": False,
            "test_hdf5_files_opened": 0,
            "test_hdf_label_datasets_opened": 0,
            "fresh_data_or_labels_read": False,
            "confirmation_data_or_labels_read": False,
        },
        "ensemble": {
            "members": ENSEMBLE_SIZE,
            "aggregation": "arithmetic_mean_predictive_distribution",
            "metric_contract": {
                "duration_primary_absolute_error_point_prediction": (
                    "equal_weight_lognormal_mixture_median"
                ),
                "duration_point_prediction_labels_used": False,
                "duration_mixture_mean_role": "heavy_tail_diagnostic_only",
                "uncertainty_moment_precision": "float64",
                "variance_identity_failure_policy": (
                    "report_equation_residual_and_tolerance_fail_closed"
                ),
            },
            "global": global_metrics,
            "per_canonical_body": per_body,
        },
        "train_only_baseline": {
            "receipt_sha256": train_baseline["sha256"],
            "independently_recomputed_on_validation": validation_baseline,
            "relative_improvement": compare_to_train_only_baseline(
                global_metrics, validation_baseline
            ),
        },
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _atomic_json_new(output, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict validation-only evaluation of a five-member canonical ensemble"
    )
    parser.add_argument("--mode", choices=("evaluate", "synthetic-smoke"), default="evaluate")
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--training-summary-sha256")
    parser.add_argument("--stage1-root", type=Path)
    parser.add_argument("--stage1-source-manifest", type=Path)
    parser.add_argument("--stage1-target-manifest", type=Path)
    parser.add_argument("--event-spec", type=Path)
    parser.add_argument("--openvla-schema5-manifest", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260828)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic-smoke":
        print("SYNTHETIC_SMOKE=" + json.dumps(run_synthetic_smoke(), sort_keys=True))
        return
    receipt = evaluate(args)
    print(
        "VALIDATION_ENSEMBLE_EVALUATION="
        + json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATION_FORMAT",
    "EvaluationError",
    "authenticate_training_bundle",
    "binding_from_paths",
    "collect_ensemble_predictions",
    "compare_to_train_only_baseline",
    "compute_ensemble_metrics",
    "equal_weight_lognormal_mixture_median",
    "evaluate",
    "load_validation_only",
    "parse_args",
    "reconstruct_frozen_split",
    "run_synthetic_smoke",
]
