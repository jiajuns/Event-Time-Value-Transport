#!/usr/bin/env python3
"""Leakage-safe nested OOF evaluation for v9 group-relative success ranking.

This protocol was designed after inspecting R5 development results.  Therefore
even a passing D250 OOF result remains adaptive development evidence and cannot
authorize deployment, Fresh evaluation, or a task-success improvement claim.
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

from calibrate_openvla_etsf_v8_success_inner_cv import (
    _cluster_bootstrap_probability_adequacy,
    _load_holdout,
    _load_manifest,
    binary_probability_metrics,
    logical_group_list_sha256,
    sha256_path,
)
from openvla_etsf_counterfactual_oof import canonical_sha256
from openvla_etsf_v9_group_relative_success_adapter import (
    DEPLOYMENT_CANDIDATE_NAMES,
    GroupRelativeAdapterConfig,
    GroupRelativeSuccessRankingAdapter,
    adapter_protocol_contract,
    load_serialized_adapter,
    predict_group_relative_adapter,
    preregistered_config_grid,
    serialize_adapter_state,
    train_group_relative_adapter,
)
from openvla_etsf_v8_structured_adapters import module_state_sha256
from train_openvla_etsf_v8_structured_adapters import (
    load_authenticated_training_payload,
    validate_v8_training_payload,
)


FORMAT = "etsf_v9_group_relative_success_ranking_nested_oof_v1"
FOLD_CONTRACT_FORMAT = "etsf_v9_group_relative_success_ranking_outer_fold_v1"
PROTOCOL_FORMAT = "etsf_v9_group_relative_success_ranking_protocol_v1"
FOLD_COUNT = 5
INNER_FOLD_COUNT = 5
INNER_SPLIT_SEED = 20260827
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
ECE_MAXIMUM = 0.10
SELECTION_TOLERANCE = 1e-12


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _reject_scope_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def implementation_file_contract() -> dict[str, str]:
    scripts = Path(__file__).resolve().parent
    names = (
        "evaluate_openvla_etsf_v9_group_relative_success_oof.py",
        "openvla_etsf_v9_group_relative_success_adapter.py",
        "calibrate_openvla_etsf_v8_success_inner_cv.py",
        "openvla_etsf_v8_structured_adapters.py",
        "train_openvla_etsf_v8_structured_adapters.py",
    )
    result = {}
    for name in names:
        path = scripts / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256_path(path)
    return result


def oof_protocol() -> dict[str, Any]:
    adapter_protocol = adapter_protocol_contract()
    value: dict[str, Any] = {
        "format": PROTOCOL_FORMAT,
        "scope": "post_R5_adaptive_D250_development_only_no_fresh",
        "adapter_protocol": adapter_protocol,
        "adapter_protocol_sha256": adapter_protocol["protocol_sha256"],
        "outer_fold_count": FOLD_COUNT,
        "inner_fold_count": INNER_FOLD_COUNT,
        "inner_split_seed": INNER_SPLIT_SEED,
        "inner_split": "label_free_sha256_order_round_robin_by_logical_group",
        "hyperparameter_grid": {
            "relative_modes": ["deterministic_delta", "group_centered"],
            "ranking_objectives": [
                "pairwise_logistic",
                "listwise_success_cross_entropy",
            ],
            "l2_regularization": [1e-3, 1e-2],
            "ranking_loss_weight": 1.0,
            "candidate_count": 8,
        },
        "nested_selection_criterion": (
            "maximum_inner_oof_selected_success_then_maximum_within_group_pair_"
            "accuracy_then_minimum_equal_group_brier_then_minimum_equal_group_"
            "nll_then_lexical_config_id"
        ),
        "task_action_rule": (
            "argmax_independent_candidate_ranking_score_with_lowest_candidate_"
            "index_for_exact_ties"
        ),
        "probability_output_not_used_for_task_action_selection": True,
        "strict_development_adequacy_gates": {
            "probability_brier_vs_outer_train_prevalence_bootstrap_ci_upper_lt_zero": True,
            "probability_nll_vs_outer_train_prevalence_bootstrap_ci_upper_lt_zero": True,
            "probability_ap_vs_evaluation_prevalence_bootstrap_ci_lower_gt_zero": True,
            "probability_ece_lte": ECE_MAXIMUM,
            "ranking_selected_success_minus_deterministic_bootstrap_ci_lower_gt_zero": True,
            "ranking_pair_accuracy_minus_random_bootstrap_ci_lower_gt_zero": True,
            "ranking_outer_fold_noninferiority_minimum": 4,
        },
        "bootstrap": {
            "resampling_unit": "logical_group",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "all_outer_contracts_selected_before_any_outer_holdout_deserialized": True,
        "same_D250_reused_after_R5_diagnosis": True,
        "confirmatory_or_deployment_claim_allowed": False,
        "fresh_inputs_or_labels_used": False,
    }
    value["protocol_sha256"] = canonical_sha256(value)
    return value


def deterministic_inner_folds(
    groups: Sequence[str], *, owner_fold_id: int
) -> list[list[str]]:
    normalized = sorted(map(str, groups))
    if (
        owner_fold_id not in range(FOLD_COUNT)
        or len(normalized) < INNER_FOLD_COUNT
        or len(normalized) != len(set(normalized))
    ):
        raise ValueError("inner group-CV input is invalid")
    namespace = (
        f"{PROTOCOL_FORMAT}|seed={INNER_SPLIT_SEED}|outer={owner_fold_id}|"
    )
    ordered = sorted(
        normalized,
        key=lambda group: (
            hashlib.sha256(f"{namespace}{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    folds = [
        sorted(ordered[index::INNER_FOLD_COUNT])
        for index in range(INNER_FOLD_COUNT)
    ]
    if sorted(group for fold in folds for group in fold) != normalized:
        raise RuntimeError("inner group-CV lost or duplicated a group")
    return folds


def _config_from_dict(value: Mapping[str, Any]) -> GroupRelativeAdapterConfig:
    return GroupRelativeAdapterConfig(
        transition_dim=int(value["transition_dim"]),
        relative_mode=str(value["relative_mode"]),
        ranking_objective=str(value["ranking_objective"]),
        l2_regularization=float(value["l2_regularization"]),
        ranking_loss_weight=float(value["ranking_loss_weight"]),
    )


def _flatten_probability_arrays(prediction: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    probability = np.asarray(prediction["success_probability"], dtype=np.float64)
    labels = np.asarray(prediction["success_label"], dtype=np.float64)
    groups = np.asarray(prediction["logical_groups"], dtype=str)
    if (
        probability.ndim != 2
        or probability.shape[1] != 4
        or labels.shape != probability.shape
        or groups.shape != (len(probability),)
        or not np.isfinite(probability).all()
        or np.any((probability <= 0.0) | (probability >= 1.0))
    ):
        raise RuntimeError("group-relative probability prediction shape changed")
    return (
        probability.reshape(-1),
        labels.reshape(-1),
        np.repeat(groups, 4),
        np.tile(np.arange(4, dtype=np.int64), len(groups)),
    )


def ranking_group_metrics(prediction: Mapping[str, Any]) -> dict[str, Any]:
    scores = np.asarray(prediction["candidate_ranking_score"], dtype=np.float64)
    labels = np.asarray(prediction["success_label"], dtype=np.float64)
    groups = np.asarray(prediction["logical_groups"], dtype=str)
    if (
        scores.ndim != 2
        or scores.shape[1] != 4
        or labels.shape != scores.shape
        or groups.shape != (len(scores),)
        or len(set(groups.tolist())) != len(groups)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("ranking arrays are not unique aligned four-candidate groups")
    selected = np.argmax(scores, axis=1)
    selected_success = labels[np.arange(len(labels)), selected]
    deterministic_success = labels[:, 0]
    oracle_success = labels.max(axis=1)
    pair_values: list[float] = []
    pair_by_group: list[float] = []
    pair_support_by_group: list[int] = []
    for group_scores, group_labels in zip(scores, labels):
        values = []
        for preferred in range(4):
            for other in range(4):
                if group_labels[preferred] <= group_labels[other]:
                    continue
                difference = group_scores[preferred] - group_scores[other]
                values.append(1.0 if difference > 0 else 0.5 if difference == 0 else 0.0)
        pair_values.extend(values)
        pair_by_group.append(float(np.mean(values)) if values else math.nan)
        pair_support_by_group.append(len(values))
    if not pair_values:
        raise ValueError("ranking evaluation has no discordant candidate pairs")
    finite_group_pairs = np.asarray(pair_by_group, dtype=np.float64)
    finite_group_pairs = finite_group_pairs[np.isfinite(finite_group_pairs)]
    if not len(finite_group_pairs):
        raise ValueError("ranking evaluation has no group-level pair accuracy")
    return {
        "logical_groups": int(len(groups)),
        "selected_success_rate": float(np.mean(selected_success)),
        "deterministic_candidate_success_rate": float(
            np.mean(deterministic_success)
        ),
        "selected_minus_deterministic_success_rate": float(
            np.mean(selected_success - deterministic_success)
        ),
        "oracle_candidate_success_rate": float(np.mean(oracle_success)),
        "selected_oracle_regret": float(np.mean(oracle_success - selected_success)),
        # Selection and uncertainty both use the logical group as the estimand
        # unit.  Keep the pair-weighted diagnostic separate so a group with
        # four discordant pairs cannot silently outweigh one with three.
        "within_group_pair_accuracy": float(np.mean(finite_group_pairs)),
        "within_group_pair_accuracy_pair_weighted": float(np.mean(pair_values)),
        "within_group_pair_estimand": "equal_logical_group_mean",
        "within_group_pair_support": len(pair_values),
        "selected_candidate_counts": {
            str(index): int(np.sum(selected == index)) for index in range(4)
        },
        "tie_break_rule": "numpy_argmax_lowest_candidate_index",
        "_selected_success": selected_success,
        "_deterministic_success": deterministic_success,
        "_pair_by_group": np.asarray(pair_by_group, dtype=np.float64),
        "_pair_support_by_group": np.asarray(pair_support_by_group, dtype=np.int64),
    }


def _public_ranking_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def ranking_group_bootstrap_adequacy(
    prediction: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    metrics = ranking_group_metrics(prediction)
    selected = metrics["_selected_success"]
    deterministic = metrics["_deterministic_success"]
    pair = metrics["_pair_by_group"]
    rng = np.random.default_rng(seed)
    delta_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    pair_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        index = rng.integers(0, len(selected), size=len(selected))
        delta_samples[sample] = np.mean(selected[index] - deterministic[index])
        selected_pair = pair[index]
        selected_pair = selected_pair[np.isfinite(selected_pair)]
        pair_samples[sample] = (
            np.mean(selected_pair) - 0.5 if len(selected_pair) else math.nan
        )
    if not np.isfinite(pair_samples).all():
        raise RuntimeError("ranking bootstrap sampled no evaluable pairs")
    delta_ci = np.quantile(delta_samples, [0.025, 0.975])
    pair_ci = np.quantile(pair_samples, [0.025, 0.975])
    return {
        "selected_success_minus_deterministic": {
            "point": metrics["selected_minus_deterministic_success_rate"],
            "ci95": [float(delta_ci[0]), float(delta_ci[1])],
            "strict_improvement": bool(delta_ci[0] > 0.0),
        },
        "within_group_pair_accuracy_minus_random": {
            "point": float(metrics["within_group_pair_accuracy"] - 0.5),
            "ci95": [float(pair_ci[0]), float(pair_ci[1])],
            "strict_improvement": bool(pair_ci[0] > 0.0),
        },
        "strict_ranking_adequacy": bool(delta_ci[0] > 0.0 and pair_ci[0] > 0.0),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": int(seed),
        "resampling_unit": "logical_group",
    }


def evaluate_adapter_records(
    adapter: GroupRelativeSuccessRankingAdapter,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = predict_group_relative_adapter(adapter, records, device=device)
    probability, labels, groups, _ = _flatten_probability_arrays(prediction)
    probability_metrics = binary_probability_metrics(labels, probability, groups)
    ranking_metrics = ranking_group_metrics(prediction)
    report = {
        "probability": probability_metrics,
        "ranking": _public_ranking_metrics(ranking_metrics),
    }
    return report, prediction


def _choose_grid_config(grid_reports: Mapping[str, Mapping[str, Any]]) -> str:
    candidates = []
    for config_id, report in grid_reports.items():
        probability = report["probability"]
        ranking = report["ranking"]
        candidates.append(
            (
                -float(ranking["selected_success_rate"]),
                -float(ranking["within_group_pair_accuracy"]),
                float(probability["equal_group_brier"]),
                float(probability["equal_group_nll"]),
                str(config_id),
            )
        )
    if not candidates:
        raise ValueError("nested selection grid is empty")
    return min(candidates)[-1]


def fit_outer_training_contract(
    records: Sequence[Mapping[str, Any]],
    *,
    owner_fold_id: int,
    outer_holdout_groups: Sequence[str],
    transition_dim: int,
    materialization_sha256: str,
    train_artifact_sha256: str,
    train_payload_sha256: str,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, Any], GroupRelativeSuccessRankingAdapter]:
    protocol = oof_protocol()
    if owner_fold_id not in range(FOLD_COUNT) or transition_dim <= 0:
        raise ValueError("outer fold id or transition dimension is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            materialization_sha256,
            train_artifact_sha256,
            train_payload_sha256,
        )
    ):
        raise ValueError("outer-training source SHA is invalid")
    groups = [str(record.get("logical_group_key", "")) for record in records]
    if (
        any(not group for group in groups)
        or len(groups) != len(set(groups))
        or groups != sorted(groups)
    ):
        raise ValueError("outer-training records must be sorted unique groups")
    holdout = sorted(map(str, outer_holdout_groups))
    if len(holdout) != len(set(holdout)) or set(groups) & set(holdout):
        raise ValueError("outer training/holdout group ownership overlaps")
    inner_folds = deterministic_inner_folds(groups, owner_fold_id=owner_fold_id)
    group_to_record = dict(zip(groups, records))
    grid_reports: dict[str, dict[str, Any]] = {}
    grid_configs = preregistered_config_grid(transition_dim)
    for config in grid_configs:
        probability_parts = []
        label_parts = []
        group_parts = []
        score_parts = []
        inner_audits = []
        for inner_fold_id, validation_groups in enumerate(inner_folds):
            validation_set = set(validation_groups)
            training_records = [
                record
                for group, record in zip(groups, records)
                if group not in validation_set
            ]
            validation_records = [group_to_record[group] for group in validation_groups]
            if set(
                str(record["logical_group_key"]) for record in training_records
            ) & validation_set:
                raise RuntimeError("inner validation group leaked into adapter fitting")
            adapter, audit = train_group_relative_adapter(
                training_records, config=config, device=device
            )
            prediction = predict_group_relative_adapter(
                adapter, validation_records, device=device
            )
            probability_parts.append(
                np.asarray(prediction["success_probability"], dtype=np.float64)
            )
            label_parts.append(np.asarray(prediction["success_label"], dtype=np.float64))
            group_parts.extend(prediction["logical_groups"])
            score_parts.append(
                np.asarray(prediction["candidate_ranking_score"], dtype=np.float64)
            )
            inner_audits.append(
                {
                    "inner_fold_id": inner_fold_id,
                    "training_groups": audit["training_groups"],
                    "training_groups_sha256": audit["training_groups_sha256"],
                    "validation_groups": validation_groups,
                    "validation_groups_sha256": logical_group_list_sha256(
                        validation_groups
                    ),
                    "training_audit_sha256": audit["training_audit_sha256"],
                    "inner_validation_labels_used_for_training": False,
                }
            )
        probability = np.concatenate(probability_parts)
        labels = np.concatenate(label_parts)
        scores = np.concatenate(score_parts)
        prediction = {
            "success_probability": probability,
            "success_label": labels,
            "candidate_ranking_score": scores,
            "logical_groups": group_parts,
        }
        flattened_probability, flattened_labels, row_groups, _ = (
            _flatten_probability_arrays(prediction)
        )
        probability_report = binary_probability_metrics(
            flattened_labels, flattened_probability, row_groups
        )
        ranking_report = _public_ranking_metrics(ranking_group_metrics(prediction))
        grid_reports[config.config_id] = {
            "config": config.to_dict(),
            "probability": probability_report,
            "ranking": ranking_report,
            "inner_folds": inner_audits,
            "inner_oof_groups_sha256": logical_group_list_sha256(
                sorted(group_parts)
            ),
            "inner_oof_prediction_sha256": canonical_sha256(
                {
                    "groups": group_parts,
                    "probability": probability.tolist(),
                    "ranking_score": scores.tolist(),
                    "labels": labels.tolist(),
                }
            ),
        }
    chosen_id = _choose_grid_config(grid_reports)
    chosen_config = _config_from_dict(grid_reports[chosen_id]["config"])
    final_adapter, final_training = train_group_relative_adapter(
        records, config=chosen_config, device=device
    )
    contract: dict[str, Any] = {
        "format": FOLD_CONTRACT_FORMAT,
        "status": "selected_from_outer_train_group_inner_oof_only",
        "scope": "post_R5_adaptive_D250_development_only_no_fresh",
        "owner_fold_id": int(owner_fold_id),
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "materialization_sha256": materialization_sha256,
        "train_artifact_sha256": train_artifact_sha256,
        "train_payload_sha256": train_payload_sha256,
        "outer_training_groups": groups,
        "outer_training_groups_sha256": logical_group_list_sha256(groups),
        "outer_holdout_groups": holdout,
        "outer_holdout_groups_sha256": logical_group_list_sha256(holdout),
        "grid_reports": grid_reports,
        "grid_report_count": len(grid_reports),
        "chosen_config_id": chosen_id,
        "chosen_config": chosen_config.to_dict(),
        "final_outer_training_audit": final_training,
        "final_adapter_state_sha256": final_training["adapter_state_sha256"],
        "final_adapter_state": serialize_adapter_state(final_adapter),
        "all_hyperparameters_selected_before_outer_holdout_payload_loaded": True,
        "outer_holdout_labels_used_for_model_or_hyperparameter_fit": False,
        "probability_and_ranking_parameters_disjoint": True,
        "same_D250_reuse_cannot_authorize_confirmation_or_deployment": True,
        "fresh_inputs_or_labels_used": False,
    }
    contract["fold_contract_sha256"] = canonical_sha256(contract)
    return contract, final_adapter


def validate_fold_contract(contract: Mapping[str, Any]) -> None:
    unsigned = dict(contract)
    recorded = unsigned.pop("fold_contract_sha256", None)
    grid = contract.get("grid_reports")
    chosen = contract.get("chosen_config")
    training_groups = list(map(str, contract.get("outer_training_groups", ())))
    holdout_groups = list(map(str, contract.get("outer_holdout_groups", ())))
    final_training = contract.get("final_outer_training_audit")
    expected_ids: set[str] = set()
    if isinstance(chosen, Mapping):
        dimension = int(chosen.get("transition_dim", -1))
        if dimension > 0:
            expected_ids = {
                config.config_id for config in preregistered_config_grid(dimension)
            }
    if (
        recorded != canonical_sha256(unsigned)
        or contract.get("format") != FOLD_CONTRACT_FORMAT
        or contract.get("protocol") != oof_protocol()
        or contract.get("protocol_sha256") != oof_protocol()["protocol_sha256"]
        or not isinstance(grid, Mapping)
        or len(grid) != 8
        or set(map(str, grid)) != expected_ids
        or contract.get("grid_report_count") != 8
        or contract.get("chosen_config_id") != _choose_grid_config(grid)
        or not isinstance(chosen, Mapping)
        or _config_from_dict(chosen).config_id != contract.get("chosen_config_id")
        or contract.get("all_hyperparameters_selected_before_outer_holdout_payload_loaded")
        is not True
        or contract.get("outer_holdout_labels_used_for_model_or_hyperparameter_fit")
        is not False
        or contract.get("probability_and_ranking_parameters_disjoint") is not True
        or contract.get("fresh_inputs_or_labels_used") is not False
        or contract.get("owner_fold_id") not in range(FOLD_COUNT)
        or training_groups != sorted(training_groups)
        or holdout_groups != sorted(holdout_groups)
        or len(training_groups) != len(set(training_groups))
        or len(holdout_groups) != len(set(holdout_groups))
        or set(training_groups) & set(holdout_groups)
        or contract.get("outer_training_groups_sha256")
        != logical_group_list_sha256(training_groups)
        or contract.get("outer_holdout_groups_sha256")
        != logical_group_list_sha256(holdout_groups)
        or not isinstance(final_training, Mapping)
        or final_training.get("training_audit_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in final_training.items()
                if key != "training_audit_sha256"
            }
        )
        or final_training.get("training_groups") != training_groups
        or final_training.get("adapter_state_sha256")
        != contract.get("final_adapter_state_sha256")
        or final_training.get("training_groups_sha256")
        != logical_group_list_sha256(training_groups)
        or not all(
            _is_sha256(contract.get(key))
            for key in (
                "materialization_sha256",
                "train_artifact_sha256",
                "train_payload_sha256",
            )
        )
    ):
        raise ValueError("v9 group-relative outer-fold contract changed")
    for config_id, report in grid.items():
        if not isinstance(report, Mapping):
            raise ValueError("v9 nested grid report is invalid")
        config = report.get("config")
        inner_folds = report.get("inner_folds")
        if (
            not isinstance(config, Mapping)
            or _config_from_dict(config).config_id != config_id
            or not isinstance(inner_folds, Sequence)
            or isinstance(inner_folds, (str, bytes))
            or len(inner_folds) != INNER_FOLD_COUNT
            or report.get("inner_oof_groups_sha256")
            != logical_group_list_sha256(training_groups)
        ):
            raise ValueError("v9 nested grid configuration changed")
        validation_union: set[str] = set()
        for inner_fold_id, row in enumerate(inner_folds):
            if not isinstance(row, Mapping):
                raise ValueError("v9 nested inner fold is invalid")
            inner_training = list(map(str, row.get("training_groups", ())))
            inner_validation = list(map(str, row.get("validation_groups", ())))
            if (
                row.get("inner_fold_id") != inner_fold_id
                or set(inner_training) & set(inner_validation)
                or set(inner_training) | set(inner_validation)
                != set(training_groups)
                or row.get("training_groups_sha256")
                != logical_group_list_sha256(sorted(inner_training))
                or row.get("validation_groups_sha256")
                != logical_group_list_sha256(sorted(inner_validation))
                or row.get("inner_validation_labels_used_for_training") is not False
            ):
                raise ValueError("v9 nested group split/provenance changed")
            validation_union.update(inner_validation)
        if validation_union != set(training_groups):
            raise ValueError("v9 nested validation coverage changed")
    restored = load_serialized_adapter(
        _config_from_dict(chosen), contract.get("final_adapter_state", {})
    )
    if module_state_sha256(restored) != contract.get("final_adapter_state_sha256"):
        raise ValueError("v9 serialized final adapter state hash changed")


def _combine_predictions(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "success_probability": np.concatenate(
            [np.asarray(part["success_probability"]) for part in parts]
        ),
        "candidate_ranking_score": np.concatenate(
            [np.asarray(part["candidate_ranking_score"]) for part in parts]
        ),
        "success_label": np.concatenate(
            [np.asarray(part["success_label"]) for part in parts]
        ),
        "logical_groups": [
            group for part in parts for group in part["logical_groups"]
        ],
    }


def run_nested_oof(
    *,
    materialization_manifest_path: Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    manifest_path = _reject_scope_path(
        materialization_manifest_path, role="v9 materialization manifest"
    )
    manifest = _load_manifest(manifest_path)
    contracts: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    read_trace: list[dict[str, Any]] = []

    # Phase one: only outer-training artifacts may be deserialized.
    for owner_fold_id, fold in enumerate(manifest["folds"]):
        train_path = _reject_scope_path(
            Path(str(fold["train_artifact"])),
            role=f"v9 fold {owner_fold_id} outer train",
        )
        payload, authentication = load_authenticated_training_payload(
            input_path=train_path,
            materialization_manifest_path=manifest_path,
            outer_fold_id=owner_fold_id,
        )
        config, records, provenance = validate_v8_training_payload(payload)
        if list(map(str, provenance["outer_training_groups"])) != [
            str(record["logical_group_key"]) for record in records
        ]:
            raise RuntimeError("outer-training payload group order/provenance changed")
        contract, adapter = fit_outer_training_contract(
            records,
            owner_fold_id=owner_fold_id,
            outer_holdout_groups=fold["oof_holdout_groups"],
            transition_dim=config.transition_dim,
            materialization_sha256=manifest["materialization_sha256"],
            train_artifact_sha256=fold["train_artifact_sha256"],
            train_payload_sha256=fold["train_payload_sha256"],
            device=device,
        )
        validate_fold_contract(contract)
        contracts.append(contract)
        prepared.append(
            {
                "adapter": adapter,
                "provenance": provenance,
                "authentication": authentication,
            }
        )
        read_trace.append(
            {
                "phase": "outer_train_contract_fit",
                "owner_fold_id": owner_fold_id,
                "role": "train",
                "path": str(train_path),
                "file_sha256": sha256_path(train_path),
            }
        )

    if len(contracts) != FOLD_COUNT:
        raise RuntimeError("all five outer contracts must be signed before holdout reads")

    # Phase two: fixed adapters and contracts only predict owner holdouts.
    fold_reports = []
    predictions = []
    baseline_parts = []
    oof_rows: list[dict[str, Any]] = []
    for owner_fold_id, fold in enumerate(manifest["folds"]):
        holdout_path = _reject_scope_path(
            Path(str(fold["holdout_artifact"])),
            role=f"v9 fold {owner_fold_id} outer holdout",
        )
        holdout = _load_holdout(
            holdout_path,
            owner_fold_id=owner_fold_id,
            manifest_fold=fold,
            train_provenance=prepared[owner_fold_id]["provenance"],
        )
        read_trace.append(
            {
                "phase": "outer_holdout_evaluation_after_all_contracts",
                "owner_fold_id": owner_fold_id,
                "role": "holdout",
                "path": str(holdout_path),
                "file_sha256": sha256_path(holdout_path),
            }
        )
        report, prediction = evaluate_adapter_records(
            prepared[owner_fold_id]["adapter"],
            holdout["batches"],
            device=device,
        )
        contract = contracts[owner_fold_id]
        prevalence = float(
            contract["final_outer_training_audit"]["prevalence"]
        )
        report.update(
            {
                "owner_fold_id": owner_fold_id,
                "fold_contract_sha256": contract["fold_contract_sha256"],
                "chosen_config_id": contract["chosen_config_id"],
                "owner_training_prevalence_baseline": prevalence,
                "holdout_artifact_sha256": fold["holdout_artifact_sha256"],
                "holdout_payload_sha256": fold["holdout_payload_sha256"],
                "outer_holdout_used_for_fitting_or_selection": False,
            }
        )
        fold_reports.append(report)
        predictions.append(prediction)
        baseline_parts.append(
            np.full(np.asarray(prediction["success_label"]).size, prevalence)
        )
        probability = np.asarray(prediction["success_probability"])
        score = np.asarray(prediction["candidate_ranking_score"])
        labels = np.asarray(prediction["success_label"])
        for group_index, group in enumerate(prediction["logical_groups"]):
            for candidate_index, candidate_name in enumerate(
                DEPLOYMENT_CANDIDATE_NAMES
            ):
                oof_rows.append(
                    {
                        "owner_fold_id": owner_fold_id,
                        "logical_group": str(group),
                        "candidate_index": candidate_index,
                        "candidate_name": candidate_name,
                        "success_label": int(labels[group_index, candidate_index]),
                        "success_probability": float(
                            probability[group_index, candidate_index]
                        ),
                        "candidate_ranking_score": float(
                            score[group_index, candidate_index]
                        ),
                        "fold_contract_sha256": contract[
                            "fold_contract_sha256"
                        ],
                    }
                )
    pooled = _combine_predictions(predictions)
    observed_groups = list(map(str, pooled["logical_groups"]))
    if (
        len(observed_groups) != 250
        or len(set(observed_groups)) != 250
        or len(oof_rows) != 1_000
        or [row["role"] for row in read_trace]
        != ["train"] * FOLD_COUNT + ["holdout"] * FOLD_COUNT
    ):
        raise RuntimeError("v9 OOF is not five owner folds / 250 groups / 1000 rows")
    probability, labels, groups, _ = _flatten_probability_arrays(pooled)
    baseline = np.concatenate(baseline_parts)
    probability_metrics = binary_probability_metrics(labels, probability, groups)
    probability_adequacy = _cluster_bootstrap_probability_adequacy(
        labels, probability, baseline, groups, seed=BOOTSTRAP_SEED
    )
    ranking_metrics = ranking_group_metrics(pooled)
    ranking_adequacy = ranking_group_bootstrap_adequacy(
        pooled, seed=BOOTSTRAP_SEED
    )
    fold_noninferiority = sum(
        report["ranking"]["selected_success_rate"]
        >= report["ranking"]["deterministic_candidate_success_rate"]
        for report in fold_reports
    )
    strict_development_adequacy = bool(
        probability_adequacy["strict_probability_adequacy"]
        and probability_metrics["ece_10_equal_width"] <= ECE_MAXIMUM
        and ranking_adequacy["strict_ranking_adequacy"]
        and fold_noninferiority >= 4
    )
    protocol = oof_protocol()
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": (
            "passed_adaptive_development_only"
            if strict_development_adequacy
            else "fail_closed_adaptive_development_only"
        ),
        "protocol": protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_files": implementation_file_contract(),
        "materialization_manifest": str(manifest_path),
        "materialization_file_sha256": sha256_path(manifest_path),
        "materialization_sha256": manifest["materialization_sha256"],
        "fold_contracts": contracts,
        "outer_holdout_evaluation": {
            "folds": fold_reports,
            "pooled_oof": {
                "probability": probability_metrics,
                "probability_adequacy_vs_owner_training_prevalence": probability_adequacy,
                "ranking": _public_ranking_metrics(ranking_metrics),
                "ranking_adequacy": ranking_adequacy,
                "ranking_outer_fold_noninferiority_count": int(fold_noninferiority),
                "strict_development_adequacy": strict_development_adequacy,
            },
        },
        "oof_rows": oof_rows,
        "oof_rows_sha256": canonical_sha256(oof_rows),
        "oof_row_count": len(oof_rows),
        "oof_alignment": "owner_fold_logical_group_candidate_index_candidate_name",
        "task_action_rule": (
            "argmax_candidate_ranking_score_lowest_candidate_index_exact_tie"
        ),
        "probability_output_used_for_action_selection": False,
        "all_outer_contracts_selected_before_any_outer_holdout_deserialized": True,
        "read_trace": read_trace,
        "read_trace_sha256": canonical_sha256(read_trace),
        "adaptive_design_disclosure": (
            "designed_after_formal_R5_D250_result_and_reuses_D250_so_no_"
            "confirmatory_or_deployment_claim"
        ),
        "task_success_improvement_claim_authorized": False,
        "selector_deployment_authorized": False,
        "fresh_confirmation_authorized": False,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def write_immutable_result(path: Path, value: Mapping[str, Any]) -> None:
    path = _reject_scope_path(path, role="v9 OOF output")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_nested_oof(
        materialization_manifest_path=args.materialization_manifest,
        device=args.device,
    )
    write_immutable_result(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "result_sha256": result["result_sha256"],
                "fresh_inputs_accepted": False,
                "fresh_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "deterministic_inner_folds",
    "evaluate_adapter_records",
    "fit_outer_training_contract",
    "oof_protocol",
    "ranking_group_bootstrap_adequacy",
    "ranking_group_metrics",
    "run_nested_oof",
    "validate_fold_contract",
    "write_immutable_result",
]
