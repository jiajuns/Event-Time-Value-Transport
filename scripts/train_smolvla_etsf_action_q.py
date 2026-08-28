#!/usr/bin/env python3
"""Train an action-conditioned ETSF scorer for native SmolVLA candidates.

The actor stays frozen.  Each training example is one of K flow-matching
action chunks generated from the same initial RoboTwin state, paired with the
outcome obtained by executing that chunk and then returning to candidate 0.
Only train statistics are used for normalization.  Validation selects one
initialization and freezes an abstaining action-selection guard; test data is
handled by a separate script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn


ACTION_DIM = 14
ACTION_CHUNK = 50
HIDDEN_DIM = 720
BRIDGE_DIM = 96
SCHEMA_VERSION = 1


@dataclass
class CandidateGroup:
    index: int
    seed: int
    resolved_seed: int
    hidden: np.ndarray
    actions: np.ndarray
    success: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_groups(root: Path) -> tuple[list[CandidateGroup], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"candidate collection is not complete: {root}")
    if int(manifest.get("action_dim", -1)) != ACTION_DIM:
        raise RuntimeError(f"unexpected action dimension in {manifest_path}")
    if int(manifest.get("action_chunk", -1)) != ACTION_CHUNK:
        raise RuntimeError(f"unexpected action chunk in {manifest_path}")
    if int(manifest.get("hidden_dim", -1)) != HIDDEN_DIM:
        raise RuntimeError(f"unexpected hidden dimension in {manifest_path}")
    if manifest.get("candidate_generator") != "native_smolvla_flow_matching_explicit_fixed_noise":
        raise RuntimeError(f"unexpected candidate generator in {manifest_path}")
    if manifest.get("language_contract") != (
        "same_instruction_for_initial_query_and_all_candidate_branches"
    ):
        raise RuntimeError(
            f"candidate branches do not prove a fixed-language contract: {root}"
        )

    groups: list[CandidateGroup] = []
    widths: set[int] = set()
    for item in manifest["groups"]:
        path = root / "groups" / item["path"]
        with h5py.File(path, "r") as handle:
            hidden = handle["candidate_hidden"][:].astype(np.float32)
            actions = handle["candidate_actions"][:].astype(np.float32)
            success = handle["success"][:].astype(np.float32)
            seed = int(handle.attrs["seed"])
            resolved_seed = int(handle.attrs.get("resolved_seed", seed))
            branch_instruction_consistent = bool(
                handle.attrs.get("branch_instruction_consistent", False)
            )
            action_exec_steps = int(handle.attrs["action_exec_steps"])
        index = int(item["index"])
        if seed != int(item["seed"]):
            raise RuntimeError(f"manifest/HDF5 seed mismatch at {path}")
        if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_DIM:
            raise RuntimeError(f"invalid hidden shape at {path}: {hidden.shape}")
        if actions.ndim != 3 or actions.shape[1:] != (ACTION_CHUNK, ACTION_DIM):
            raise RuntimeError(f"invalid action shape at {path}: {actions.shape}")
        if success.shape != (len(actions),) or len(hidden) != len(actions):
            raise RuntimeError(f"candidate count mismatch at {path}")
        if action_exec_steps != int(manifest["action_exec_steps"]):
            raise RuntimeError(f"action execution contract mismatch at {path}")
        if not branch_instruction_consistent:
            raise RuntimeError(f"branch instruction was not held fixed at {path}")
        if not np.isfinite(hidden).all() or not np.isfinite(actions).all():
            raise RuntimeError(f"non-finite candidate features at {path}")
        widths.add(len(actions))
        groups.append(
            CandidateGroup(index, seed, resolved_seed, hidden, actions, success)
        )
    if not groups or len(widths) != 1:
        raise RuntimeError(f"empty collection or inconsistent candidate widths: {widths}")
    seeds = [group.seed for group in groups]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate scene seeds in {root}")
    resolved_seeds = [group.resolved_seed for group in groups]
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise RuntimeError(
            f"duplicate resolved RoboTwin scenes in {root}; sanitize the split first"
        )
    return groups, manifest


def load_collection_roots(
    roots: list[Path],
) -> tuple[list[CandidateGroup], dict[str, Any]]:
    all_groups: list[CandidateGroup] = []
    manifests: list[dict[str, Any]] = []
    for root in roots:
        groups, manifest = load_groups(root)
        all_groups.extend(groups)
        manifests.append(manifest)
    requested = [group.seed for group in all_groups]
    resolved = [group.resolved_seed for group in all_groups]
    if len(set(requested)) != len(requested):
        raise RuntimeError("duplicate requested seeds across candidate roots")
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("duplicate resolved scenes across candidate roots")
    keys = [
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
        for key in keys:
            if manifest.get(key) != first.get(key):
                raise RuntimeError(f"candidate root contract mismatch: {key}")
    return all_groups, first


def raw_action_features(
    actions: torch.Tensor,
    baseline: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    normalized = (actions - action_mean[None, None]) / action_std[None, None]
    delta = (actions - baseline) / action_std[None, None]
    action_stats = torch.cat(
        [
            normalized.mean(1),
            normalized.std(1, unbiased=False),
            normalized.amin(1),
            normalized.amax(1),
            normalized[:, 0],
            normalized[:, -1],
        ],
        dim=1,
    )
    delta_stats = torch.cat(
        [
            delta.mean(1),
            delta.std(1, unbiased=False),
            delta.abs().mean(1),
            delta.abs().amax(1),
            delta[:, 0],
            delta[:, -1],
        ],
        dim=1,
    )
    velocity = normalized[:, 1:] - normalized[:, :-1]
    velocity_stats = torch.cat(
        [velocity.mean(1), velocity.std(1, unbiased=False), velocity.abs().amax(1)],
        dim=1,
    )
    scalars = torch.stack(
        [
            torch.sqrt(delta.square().mean((1, 2))),
            delta.abs().amax((1, 2)),
            torch.sqrt((actions - baseline).square().mean((1, 2))),
            (actions - baseline).abs().amax((1, 2)),
        ],
        dim=1,
    )
    return torch.cat([action_stats, delta_stats, velocity_stats, scalars], dim=1)


class SmolVLAActionETSF(nn.Module):
    """A shared 720->96 event/action bridge and candidate-value head."""

    def __init__(
        self,
        hidden_mean: torch.Tensor,
        hidden_std: torch.Tensor,
        delta_hidden_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("hidden_mean", hidden_mean.float())
        self.register_buffer("hidden_std", hidden_std.float())
        self.register_buffer("delta_hidden_std", delta_hidden_std.float())
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float())
        self.semantic = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, BRIDGE_DIM),
            nn.LayerNorm(BRIDGE_DIM),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(int(feature_mean.numel()), 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, BRIDGE_DIM),
            nn.LayerNorm(BRIDGE_DIM),
        )
        self.head = nn.Sequential(
            nn.Linear(BRIDGE_DIM * 4, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        baseline_hidden: torch.Tensor,
        actions: torch.Tensor,
        baseline_actions: torch.Tensor,
    ) -> torch.Tensor:
        absolute_hidden = (hidden - self.hidden_mean[None]) / self.hidden_std[None]
        delta_hidden = (hidden - baseline_hidden) / self.delta_hidden_std[None]
        semantic = self.semantic(torch.cat([absolute_hidden, delta_hidden], dim=1))
        raw_actions = raw_action_features(
            actions, baseline_actions, self.action_mean, self.action_std
        )
        normalized_actions = (
            raw_actions - self.feature_mean[None]
        ) / self.feature_std[None]
        action = self.action_encoder(normalized_actions)
        joined = torch.cat(
            [semantic, action, semantic * action, semantic - action], dim=1
        )
        return self.head(joined).squeeze(1)


def pack(groups: list[CandidateGroup], device: torch.device) -> dict[str, torch.Tensor]:
    widths = [len(group.actions) for group in groups]
    hidden = np.concatenate([group.hidden for group in groups])
    baseline_hidden = np.concatenate(
        [np.repeat(group.hidden[0:1], len(group.hidden), axis=0) for group in groups]
    )
    actions = np.concatenate([group.actions for group in groups])
    baseline_actions = np.concatenate(
        [np.repeat(group.actions[0:1], len(group.actions), axis=0) for group in groups]
    )
    return {
        "hidden": torch.as_tensor(hidden, device=device),
        "baseline_hidden": torch.as_tensor(baseline_hidden, device=device),
        "actions": torch.as_tensor(actions, device=device),
        "baseline_actions": torch.as_tensor(baseline_actions, device=device),
        "success": torch.as_tensor(
            np.concatenate([group.success for group in groups]), device=device
        ),
        "group_ids": torch.as_tensor(
            np.repeat(np.arange(len(groups)), widths), device=device
        ),
        "candidate_ids": torch.as_tensor(
            np.concatenate([np.arange(width) for width in widths]), device=device
        ),
        "scene_seeds": torch.as_tensor(
            np.repeat([group.seed for group in groups], widths), device=device
        ),
        "resolved_scene_seeds": torch.as_tensor(
            np.repeat([group.resolved_seed for group in groups], widths), device=device
        ),
    }


def training_statistics(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    hidden = batch["hidden"]
    baseline_hidden = batch["baseline_hidden"]
    actions = batch["actions"]
    baseline_actions = batch["baseline_actions"]
    hidden_mean = hidden.mean(0)
    hidden_std = hidden.std(0, unbiased=False).clamp_min(1e-3)
    delta_hidden_std = (hidden - baseline_hidden).std(0, unbiased=False).clamp_min(1e-3)
    action_mean = actions.mean((0, 1))
    action_std = actions.std((0, 1), unbiased=False).clamp_min(1e-2)
    raw = raw_action_features(actions, baseline_actions, action_mean, action_std)
    return (
        hidden_mean,
        hidden_std,
        delta_hidden_std,
        action_mean,
        action_std,
        raw.mean(0),
        raw.std(0, unbiased=False).clamp_min(1e-3),
    )


def model_logits(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    return model(
        batch["hidden"],
        batch["baseline_hidden"],
        batch["actions"],
        batch["baseline_actions"],
    )


def pairwise_ranking_loss(
    logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        positive = logits[mask & (labels > 0.5)]
        negative = logits[mask & (labels < 0.5)]
        if len(positive) and len(negative):
            losses.append(F.softplus(-(positive[:, None] - negative[None])).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def bootstrap_paired_interval(
    difference: np.ndarray, seed: int, draws: int = 10_000
) -> list[float]:
    if not len(difference):
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    sampled = difference[
        rng.integers(0, len(difference), size=(draws, len(difference)))
    ].mean(1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    bootstrap_seed: int,
    minimum_probability_margin: float = 0.0,
    maximum_normalized_distance: float = math.inf,
) -> dict[str, Any]:
    model.eval()
    probabilities = torch.sigmoid(model_logits(model, batch)).cpu().numpy()
    labels = batch["success"].cpu().numpy().astype(bool)
    group_ids = batch["group_ids"].cpu().numpy()
    candidate_ids = batch["candidate_ids"].cpu().numpy()
    normalized_distance = torch.sqrt(
        (
            (batch["actions"] - batch["baseline_actions"])
            / model.action_std[None, None]
        )
        .square()
        .mean((1, 2))
    ).cpu().numpy()

    selected: list[float] = []
    baseline: list[float] = []
    oracle: list[float] = []
    selected_ids: list[int] = []
    margins: list[float] = []
    pair_correct: list[bool] = []
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        base_candidates = indices[candidate_ids[indices] == 0]
        if len(base_candidates) != 1:
            raise RuntimeError(f"group {group_id} lacks exactly one baseline")
        base = int(base_candidates[0])
        proposed = int(indices[np.argmax(probabilities[indices])])
        margin = float(probabilities[proposed] - probabilities[base])
        pick = proposed
        if candidate_ids[proposed] != 0 and (
            margin < minimum_probability_margin
            or normalized_distance[proposed] > maximum_normalized_distance
        ):
            pick = base
        selected.append(float(labels[pick]))
        baseline.append(float(labels[base]))
        oracle.append(float(labels[indices].any()))
        selected_ids.append(int(candidate_ids[pick]))
        margins.append(margin)
        positives = probabilities[indices][labels[indices]]
        negatives = probabilities[indices][~labels[indices]]
        if len(positives) and len(negatives):
            pair_correct.extend(
                (positives[:, None] > negatives[None]).reshape(-1).tolist()
            )

    selected_array = np.asarray(selected)
    baseline_array = np.asarray(baseline)
    difference = selected_array - baseline_array
    improved = int(np.sum(difference > 0))
    harmed = int(np.sum(difference < 0))
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    return {
        "groups": int(len(selected)),
        "successful_candidates": int(labels.sum()),
        "groups_with_outcome_variation": int(
            sum(
                len(np.unique(labels[group_ids == group_id])) > 1
                for group_id in np.unique(group_ids)
            )
        ),
        "baseline_successes": int(baseline_array.sum()),
        "selected_successes": int(selected_array.sum()),
        "oracle_successes": int(np.sum(oracle)),
        "baseline_success_rate": float(baseline_array.mean()),
        "selected_success_rate": float(selected_array.mean()),
        "oracle_success_rate": float(np.mean(oracle)),
        "paired_success_difference": float(difference.mean()),
        "paired_difference_ci95": bootstrap_paired_interval(
            difference, bootstrap_seed
        ),
        "improved_groups": improved,
        "harmed_groups": harmed,
        "unchanged_groups": int(len(difference) - improved - harmed),
        "changed_groups": int(np.sum(np.asarray(selected_ids) != 0)),
        "selected_candidate_histogram": {
            str(index): int(np.sum(np.asarray(selected_ids) == index))
            for index in sorted(set(selected_ids))
        },
        "selected_candidate_ids": selected_ids,
        "proposal_probability_margins": margins,
        "candidate_auc": auc,
        "candidate_brier": float(np.mean(np.square(probabilities - labels))),
        "within_group_pair_accuracy": (
            float(np.mean(pair_correct)) if pair_correct else None
        ),
        "guard": {
            "minimum_probability_margin": minimum_probability_margin,
            "maximum_normalized_distance": maximum_normalized_distance,
        },
    }


def validation_score(metrics: dict[str, Any]) -> tuple[float, ...]:
    pair_accuracy = metrics["within_group_pair_accuracy"]
    auc = metrics["candidate_auc"]
    return (
        metrics["paired_success_difference"],
        -metrics["harmed_groups"],
        pair_accuracy if pair_accuracy is not None else -1.0,
        auc if auc is not None else -1.0,
        -metrics["candidate_brier"],
        -metrics["changed_groups"],
    )


def tune_guard(
    model: nn.Module, validation_batch: dict[str, torch.Tensor], seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_score: tuple[float, ...] | None = None
    for margin in [0.0, 0.02, 0.05, 0.10, 0.15, 0.25]:
        for distance in [0.10, 0.15, 0.20, 0.30, 0.50, 1e9]:
            metrics = evaluate(
                model,
                validation_batch,
                seed,
                minimum_probability_margin=margin,
                maximum_normalized_distance=distance,
            )
            trials.append(metrics)
            score = validation_score(metrics)
            if selected_score is None or score > selected_score:
                selected_score = score
                selected = metrics
    if selected is None:
        raise RuntimeError("guard tuning produced no trial")
    return selected, trials


def train_one(
    seed: int,
    statistics: tuple[torch.Tensor, ...],
    train_batch: dict[str, torch.Tensor],
    validation_batch: dict[str, torch.Tensor],
    steps: int,
    learning_rate: float,
) -> tuple[SmolVLAActionETSF, dict[str, Any], int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SmolVLAActionETSF(*statistics).to(train_batch["hidden"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    labels = train_batch["success"]
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    pos_weight = torch.tensor(
        min(negatives / max(positives, 1.0), 12.0), device=labels.device
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    last_improvement = 0
    patience = max(200, steps // 4)
    for step in range(steps):
        model.train()
        logits = model_logits(model, train_batch)
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        ranking = pairwise_ranking_loss(logits, labels, train_batch["group_ids"])
        loss = bce + ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % 20 == 0 or step == steps - 1:
            metrics = evaluate(model, validation_batch, seed + step)
            score = validation_score(metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                best_state = copy.deepcopy(model.state_dict())
                last_improvement = step
            if step - last_improvement >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_metrics, step + 1


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> SmolVLAActionETSF:
    state = checkpoint["model"]
    model = SmolVLAActionETSF(
        state["hidden_mean"],
        state["hidden_std"],
        state["delta_hidden_std"],
        state["action_mean"],
        state["action_std"],
        state["feature_mean"],
        state["feature_std"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260827, 20260828, 20260829, 20260830, 20260831],
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        parser.error("--steps and --learning-rate must be positive")

    train_groups, train_manifest = load_collection_roots(args.train)
    validation_groups, validation_manifest = load_collection_roots(
        [args.validation]
    )
    train_seeds = {group.seed for group in train_groups}
    validation_seeds = {group.seed for group in validation_groups}
    overlap = train_seeds & validation_seeds
    if overlap:
        raise RuntimeError(f"train/validation seed leakage: {sorted(overlap)}")
    train_resolved_seeds = {group.resolved_seed for group in train_groups}
    validation_resolved_seeds = {
        group.resolved_seed for group in validation_groups
    }
    resolved_overlap = train_resolved_seeds & validation_resolved_seeds
    if resolved_overlap:
        raise RuntimeError(
            f"train/validation resolved-scene leakage: {sorted(resolved_overlap)}"
        )
    if train_manifest["checkpoint"] != validation_manifest["checkpoint"]:
        raise RuntimeError("train/validation use different frozen SmolVLA checkpoints")
    for key in ["task", "body", "candidate_count", "action_exec_steps", "max_steps"]:
        if train_manifest[key] != validation_manifest[key]:
            raise RuntimeError(f"train/validation contract mismatch: {key}")

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    train_batch = pack(train_groups, device)
    validation_batch = pack(validation_groups, device)
    statistics = training_statistics(train_batch)
    args.output.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    selected_model: SmolVLAActionETSF | None = None
    selected_seed: int | None = None
    selected_score: tuple[float, ...] | None = None
    for seed in args.seeds:
        model, validation_unguarded, trained_steps = train_one(
            seed,
            statistics,
            train_batch,
            validation_batch,
            args.steps,
            args.learning_rate,
        )
        train_metrics = evaluate(model, train_batch, seed + 100_000)
        validation_metrics, guard_trials = tune_guard(
            model, validation_batch, seed + 200_000
        )
        run = {
            "seed": seed,
            "trained_steps": trained_steps,
            "train": train_metrics,
            "validation_unguarded": validation_unguarded,
            "validation": validation_metrics,
            "guard_trials": guard_trials,
        }
        runs.append(run)
        score = validation_score(validation_metrics)
        if selected_score is None or score > selected_score:
            selected_score = score
            selected_model = model
            selected_seed = seed
        print("RUN=" + json.dumps(run, sort_keys=True), flush=True)
    if selected_model is None or selected_seed is None:
        raise RuntimeError("no model initialization was selected")

    selected_run = next(run for run in runs if run["seed"] == selected_seed)
    selected_validation = selected_run["validation"]
    gate_checks = {
        "validation_has_at_least_two_mixed_groups": selected_validation[
            "groups_with_outcome_variation"
        ]
        >= 2,
        "validation_not_worse": selected_validation["selected_successes"]
        >= selected_validation["baseline_successes"],
        "validation_changed_at_least_one": selected_validation["changed_groups"] > 0,
        "validation_pair_accuracy_at_least_0_60": (
            selected_validation["within_group_pair_accuracy"] is not None
            and selected_validation["within_group_pair_accuracy"] >= 0.60
        ),
        "validation_paired_ci_lower_at_least_minus_0_10": selected_validation[
            "paired_difference_ci95"
        ][0]
        >= -0.10,
    }
    authorized = all(gate_checks.values())
    checkpoint_path = args.output / "smolvla_etsf_action_q_selected.pt"
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "method": "Q_ETSF(smolvla_expert_hidden, action_chunk)",
        "model": selected_model.state_dict(),
        "selected_seed": selected_seed,
        "guard": selected_validation["guard"],
        "action_ranking_authorized": authorized,
        "gate_checks": gate_checks,
        "train_seeds": sorted(train_seeds),
        "validation_seeds": sorted(validation_seeds),
        "train_resolved_seeds": sorted(train_resolved_seeds),
        "validation_resolved_seeds": sorted(validation_resolved_seeds),
        "frozen_actor_checkpoint": train_manifest["checkpoint"],
        "task": train_manifest["task"],
        "body": train_manifest["body"],
        "candidate_count": train_manifest["candidate_count"],
        "action_exec_steps": train_manifest["action_exec_steps"],
        "max_steps": train_manifest["max_steps"],
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "validation_frozen",
        "method": checkpoint["method"],
        "actor_frozen": True,
        "candidate_generator": train_manifest["candidate_generator"],
        "intervention": train_manifest["intervention"],
        "train_roots": [str(root) for root in args.train],
        "validation_root": str(args.validation),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_successful_candidates": int(sum(group.success.sum() for group in train_groups)),
        "validation_successful_candidates": int(
            sum(group.success.sum() for group in validation_groups)
        ),
        "selected_seed": selected_seed,
        "selected_validation": selected_validation,
        "gate_checks": gate_checks,
        "action_ranking_authorized": authorized,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "runs": runs,
        "test_policy": (
            "Apply the frozen scorer and guard once to unseen offset-40 seeds; "
            "do not retune after opening outcomes."
        ),
        "limitations": [
            "This is a within-ALOHA move_can_pot test, not yet cross-embodiment evidence.",
            "Only the first 50-step candidate chunk differs; later queries use fixed candidate 0.",
            "The small validation set can authorize a pilot test but not deployment.",
        ],
    }
    atomic_json(args.output / "smolvla_etsf_action_q_summary.json", summary)
    print(
        "TRAINING_COMPLETE="
        + json.dumps(
            {
                "selected_seed": selected_seed,
                "authorized": authorized,
                "validation": selected_validation,
                "checkpoint_sha256": summary["checkpoint_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
