#!/usr/bin/env python3
"""Train a low-capacity pairwise ETSF ranker for SmolVLA candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from train_smolvla_etsf_action_q import (
    CandidateGroup,
    atomic_json,
    load_groups,
    raw_action_features,
    sha256,
)


@dataclass
class PairwiseLinearETSF:
    feature_mode: str
    action_mean: np.ndarray
    action_std: np.ndarray
    scaler: StandardScaler
    pca: PCA
    ranker: LogisticRegression

    def raw_features(self, groups: list[CandidateGroup]) -> np.ndarray:
        hidden = np.concatenate([group.hidden for group in groups])
        baseline_hidden = np.concatenate(
            [
                np.repeat(group.hidden[0:1], len(group.hidden), axis=0)
                for group in groups
            ]
        )
        hidden_delta = hidden - baseline_hidden
        if self.feature_mode == "hidden_delta":
            return hidden_delta
        actions = torch.as_tensor(
            np.concatenate([group.actions for group in groups])
        )
        baseline_actions = torch.as_tensor(
            np.concatenate(
                [
                    np.repeat(group.actions[0:1], len(group.actions), axis=0)
                    for group in groups
                ]
            )
        )
        action_features = raw_action_features(
            actions,
            baseline_actions,
            torch.as_tensor(self.action_mean),
            torch.as_tensor(self.action_std),
        ).numpy()
        if self.feature_mode == "combined":
            return np.concatenate([hidden_delta, action_features], axis=1)
        raise ValueError(f"unknown feature mode: {self.feature_mode}")

    def scores(self, groups: list[CandidateGroup]) -> np.ndarray:
        raw = self.raw_features(groups)
        transformed = self.pca.transform(self.scaler.transform(raw))
        return self.ranker.decision_function(transformed)


def model_to_state(model: PairwiseLinearETSF) -> dict[str, Any]:
    return {
        "feature_mode": model.feature_mode,
        "action_mean": model.action_mean,
        "action_std": model.action_std,
        "scaler": model.scaler,
        "pca": model.pca,
        "ranker": model.ranker,
    }


def model_from_state(state: dict[str, Any]) -> PairwiseLinearETSF:
    return PairwiseLinearETSF(
        state["feature_mode"],
        state["action_mean"],
        state["action_std"],
        state["scaler"],
        state["pca"],
        state["ranker"],
    )


def group_arrays(groups: list[CandidateGroup]):
    widths = [len(group.actions) for group in groups]
    return {
        "labels": np.concatenate([group.success for group in groups]).astype(bool),
        "group_ids": np.repeat(np.arange(len(groups)), widths),
        "candidate_ids": np.concatenate(
            [np.arange(width) for width in widths]
        ),
        "actions": np.concatenate([group.actions for group in groups]),
        "baseline_actions": np.concatenate(
            [
                np.repeat(group.actions[0:1], len(group.actions), axis=0)
                for group in groups
            ]
        ),
    }


def candidate_action_statistics(groups: list[CandidateGroup]):
    actions = np.concatenate([group.actions for group in groups])
    return (
        actions.mean((0, 1)).astype(np.float32),
        actions.std((0, 1)).clip(min=1e-2).astype(np.float32),
    )


def fit_model(
    groups: list[CandidateGroup], feature_mode: str, components: int, c_value: float
) -> PairwiseLinearETSF:
    action_mean, action_std = candidate_action_statistics(groups)
    placeholder = PairwiseLinearETSF(
        feature_mode,
        action_mean,
        action_std,
        StandardScaler(),
        PCA(),
        LogisticRegression(),
    )
    raw = placeholder.raw_features(groups)
    scaler = StandardScaler().fit(raw)
    normalized = scaler.transform(raw)
    pca = PCA(
        n_components=min(components, len(normalized) - 1, normalized.shape[1]),
        random_state=20260827,
    ).fit(normalized)
    transformed = pca.transform(normalized)
    arrays = group_arrays(groups)
    pair_features: list[np.ndarray] = []
    pair_labels: list[int] = []
    for group_id in np.unique(arrays["group_ids"]):
        indices = np.flatnonzero(arrays["group_ids"] == group_id)
        labels = arrays["labels"][indices]
        positives = indices[labels]
        negatives = indices[~labels]
        for positive in positives:
            for negative in negatives:
                difference = transformed[positive] - transformed[negative]
                pair_features.extend([difference, -difference])
                pair_labels.extend([1, 0])
    if len(pair_features) < 4:
        raise RuntimeError("too few mixed-outcome groups for pairwise training")
    ranker = LogisticRegression(
        C=c_value,
        fit_intercept=False,
        solver="liblinear",
        max_iter=4000,
        random_state=20260827,
    ).fit(np.asarray(pair_features), np.asarray(pair_labels))
    return PairwiseLinearETSF(
        feature_mode, action_mean, action_std, scaler, pca, ranker
    )


def bootstrap_interval(difference: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = difference[
        rng.integers(0, len(difference), size=(10_000, len(difference)))
    ].mean(1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def evaluate(
    model: PairwiseLinearETSF,
    groups: list[CandidateGroup],
    bootstrap_seed: int,
    minimum_probability_margin: float = 0.0,
    maximum_normalized_distance: float = math.inf,
) -> dict[str, Any]:
    arrays = group_arrays(groups)
    scores = model.scores(groups)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
    normalized_distance = np.sqrt(
        np.square(
            (arrays["actions"] - arrays["baseline_actions"])
            / model.action_std[None, None]
        ).mean((1, 2))
    )
    selected: list[int] = []
    baseline: list[int] = []
    oracle: list[int] = []
    selected_ids: list[int] = []
    pair_correct: list[bool] = []
    margins: list[float] = []
    for group_id in np.unique(arrays["group_ids"]):
        indices = np.flatnonzero(arrays["group_ids"] == group_id)
        base = indices[arrays["candidate_ids"][indices] == 0]
        if len(base) != 1:
            raise RuntimeError(f"group {group_id} lacks one baseline candidate")
        base = int(base[0])
        proposed = int(indices[np.argmax(scores[indices])])
        margin = float(probabilities[proposed] - probabilities[base])
        pick = proposed
        if arrays["candidate_ids"][proposed] != 0 and (
            margin < minimum_probability_margin
            or normalized_distance[proposed] > maximum_normalized_distance
        ):
            pick = base
        labels = arrays["labels"][indices]
        positives = scores[indices][labels]
        negatives = scores[indices][~labels]
        if len(positives) and len(negatives):
            pair_correct.extend(
                (positives[:, None] > negatives[None]).reshape(-1).tolist()
            )
        selected.append(int(arrays["labels"][pick]))
        baseline.append(int(arrays["labels"][base]))
        oracle.append(int(labels.any()))
        selected_ids.append(int(arrays["candidate_ids"][pick]))
        margins.append(margin)
    selected_array = np.asarray(selected)
    baseline_array = np.asarray(baseline)
    difference = selected_array - baseline_array
    improved = int((difference > 0).sum())
    harmed = int((difference < 0).sum())
    labels = arrays["labels"]
    return {
        "groups": len(groups),
        "successful_candidates": int(labels.sum()),
        "groups_with_outcome_variation": int(
            sum(len(np.unique(group.success)) > 1 for group in groups)
        ),
        "baseline_successes": int(baseline_array.sum()),
        "selected_successes": int(selected_array.sum()),
        "oracle_successes": int(np.sum(oracle)),
        "baseline_success_rate": float(baseline_array.mean()),
        "selected_success_rate": float(selected_array.mean()),
        "oracle_success_rate": float(np.mean(oracle)),
        "paired_success_difference": float(difference.mean()),
        "paired_difference_ci95": bootstrap_interval(difference, bootstrap_seed),
        "improved_groups": improved,
        "harmed_groups": harmed,
        "unchanged_groups": int(len(groups) - improved - harmed),
        "changed_groups": int(np.sum(np.asarray(selected_ids) != 0)),
        "selected_candidate_ids": selected_ids,
        "selected_candidate_histogram": {
            str(index): int(np.sum(np.asarray(selected_ids) == index))
            for index in sorted(set(selected_ids))
        },
        "proposal_probability_margins": margins,
        "within_group_pair_accuracy": (
            float(np.mean(pair_correct)) if pair_correct else None
        ),
        "candidate_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "candidate_brier": float(np.mean(np.square(probabilities - labels))),
        "guard": {
            "minimum_probability_margin": minimum_probability_margin,
            "maximum_normalized_distance": maximum_normalized_distance,
        },
    }


def validation_score(metrics: dict[str, Any]) -> tuple[float, ...]:
    pair_accuracy = metrics["within_group_pair_accuracy"]
    return (
        metrics["paired_success_difference"],
        -metrics["harmed_groups"],
        pair_accuracy if pair_accuracy is not None else -1.0,
        -metrics["candidate_brier"],
        -metrics["changed_groups"],
    )


def tune_guard(model, groups, seed):
    trials = []
    selected = None
    selected_score = None
    for margin in [0.0, 0.02, 0.05, 0.10, 0.15, 0.25]:
        for distance in [0.10, 0.15, 0.20, 0.30, 0.50, 1e9]:
            metrics = evaluate(model, groups, seed, margin, distance)
            trials.append(metrics)
            score = validation_score(metrics)
            if selected_score is None or score > selected_score:
                selected_score = score
                selected = metrics
    return selected, trials


def load_roots(roots: list[Path]):
    groups: list[CandidateGroup] = []
    manifests = []
    for root in roots:
        loaded, manifest = load_groups(root)
        groups.extend(loaded)
        manifests.append(manifest)
    requested = [group.seed for group in groups]
    resolved = [group.resolved_seed for group in groups]
    if len(set(requested)) != len(requested):
        raise RuntimeError("duplicate requested seeds across roots")
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("duplicate resolved scenes across roots")
    contract_keys = [
        "checkpoint",
        "task",
        "body",
        "candidate_count",
        "action_exec_steps",
        "max_steps",
        "language_contract",
    ]
    first = manifests[0]
    for manifest in manifests[1:]:
        for key in contract_keys:
            if manifest.get(key) != first.get(key):
                raise RuntimeError(f"candidate root contract mismatch: {key}")
    return groups, first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_groups, contract = load_roots(args.train)
    validation_groups, validation_contract = load_roots([args.validation])
    train_resolved = {group.resolved_seed for group in train_groups}
    validation_resolved = {group.resolved_seed for group in validation_groups}
    overlap = train_resolved & validation_resolved
    if overlap:
        raise RuntimeError(f"train/validation scene leakage: {sorted(overlap)}")
    for key in [
        "checkpoint",
        "task",
        "body",
        "candidate_count",
        "action_exec_steps",
        "max_steps",
        "language_contract",
    ]:
        if contract.get(key) != validation_contract.get(key):
            raise RuntimeError(f"train/validation contract mismatch: {key}")

    configurations = [
        {"feature_mode": "hidden_delta", "components": 32, "c_value": 1.0},
        {"feature_mode": "combined", "components": 32, "c_value": 0.01},
    ]
    runs = []
    selected_model = None
    selected_metrics = None
    selected_config = None
    selected_score = None
    selected_trials = None
    for index, config in enumerate(configurations):
        model = fit_model(train_groups, **config)
        train_metrics = evaluate(model, train_groups, 20260827 + index)
        validation_metrics, trials = tune_guard(
            model, validation_groups, 20261827 + index
        )
        run = {
            "configuration": config,
            "train": train_metrics,
            "validation": validation_metrics,
            "guard_trials": trials,
        }
        runs.append(run)
        score = validation_score(validation_metrics)
        if selected_score is None or score > selected_score:
            selected_score = score
            selected_model = model
            selected_metrics = validation_metrics
            selected_config = config
            selected_trials = trials
    if selected_model is None or selected_metrics is None:
        raise RuntimeError("no pairwise model selected")
    gate_checks = {
        "validation_has_at_least_three_mixed_groups": selected_metrics[
            "groups_with_outcome_variation"
        ]
        >= 3,
        "validation_not_worse": selected_metrics["selected_successes"]
        >= selected_metrics["baseline_successes"],
        "validation_changed_at_least_one": selected_metrics["changed_groups"] > 0,
        "validation_pair_accuracy_at_least_0_60": (
            selected_metrics["within_group_pair_accuracy"] is not None
            and selected_metrics["within_group_pair_accuracy"] >= 0.60
        ),
        "validation_no_harmed_group": selected_metrics["harmed_groups"] == 0,
    }
    authorized = all(gate_checks.values())
    checkpoint_path = args.output / "smolvla_etsf_pairwise_linear_selected.pkl"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "method": "pairwise_Q_ETSF(candidate_hidden_delta, action_chunk)",
        "model_state": model_to_state(selected_model),
        "configuration": selected_config,
        "guard": selected_metrics["guard"],
        "action_ranking_authorized": authorized,
        "gate_checks": gate_checks,
        "train_requested_seeds": sorted(group.seed for group in train_groups),
        "train_resolved_seeds": sorted(train_resolved),
        "validation_requested_seeds": sorted(
            group.seed for group in validation_groups
        ),
        "validation_resolved_seeds": sorted(validation_resolved),
        "contract": {
            key: contract.get(key)
            for key in [
                "checkpoint",
                "task",
                "body",
                "candidate_count",
                "action_exec_steps",
                "max_steps",
                "language_contract",
            ]
        },
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "status": "validation_frozen",
        "method": checkpoint["method"],
        "train_roots": [str(root) for root in args.train],
        "validation_root": str(args.validation),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "runs": runs,
        "selected_configuration": selected_config,
        "selected_validation": selected_metrics,
        "selected_guard_trials": selected_trials,
        "gate_checks": gate_checks,
        "action_ranking_authorized": authorized,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    atomic_json(args.output / "smolvla_etsf_pairwise_linear_summary.json", summary)
    print(
        "PAIRWISE_TRAINING_COMPLETE="
        + json.dumps(
            {
                "authorized": authorized,
                "configuration": selected_config,
                "validation": selected_metrics,
                "checkpoint_sha256": summary["checkpoint_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
