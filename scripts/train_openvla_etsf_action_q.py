#!/usr/bin/env python3
"""Train an action-conditioned Q_ETSF on same-seed candidate branches."""

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

from train_openvla_etsf_shadow import BRIDGE, SemanticEncoder


ACTION_DIM = 14
CHUNK = 25


@dataclass
class CandidateGroup:
    seed: int
    resolved_seed: int
    hidden: np.ndarray
    actions: np.ndarray
    source_logprobs: np.ndarray
    success: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_groups(root: Path) -> list[CandidateGroup]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"candidate collection is not complete: {root}")
    if manifest.get("language_contract") != (
        "same_instruction_for_initial_query_and_all_candidate_branches"
    ):
        raise RuntimeError(f"candidate collection lacks fixed-language proof: {root}")
    groups = []
    for item in manifest["groups"]:
        with h5py.File(root / "groups" / item["path"], "r") as handle:
            hidden = handle["initial_hidden"][:].astype(np.float32)
            actions = handle["candidate_actions"][:].astype(np.float32)
            logprobs = handle["source_logprobs"][:].astype(np.float32)
            success = handle["success"][:].astype(np.float32)
            seed = int(handle.attrs["seed"])
            resolved_seed = int(handle.attrs.get("resolved_seed", seed))
            instruction_consistent = bool(
                handle.attrs.get("branch_instruction_consistent", False)
            )
        if hidden.shape != (4096,):
            raise RuntimeError(f"invalid hidden shape for seed {seed}: {hidden.shape}")
        if actions.ndim != 3 or actions.shape[1:] != (CHUNK, ACTION_DIM):
            raise RuntimeError(f"invalid candidate actions for seed {seed}: {actions.shape}")
        if logprobs.shape != (len(actions), CHUNK * ACTION_DIM):
            raise RuntimeError(f"invalid source logprobs for seed {seed}: {logprobs.shape}")
        if success.shape != (len(actions),):
            raise RuntimeError(f"invalid outcomes for seed {seed}: {success.shape}")
        if not instruction_consistent:
            raise RuntimeError(f"branch language changed for seed {seed}")
        if not np.isfinite(hidden).all() or not np.isfinite(actions).all() or not np.isfinite(logprobs).all():
            raise RuntimeError(f"non-finite candidate data for seed {seed}")
        groups.append(
            CandidateGroup(seed, resolved_seed, hidden, actions, logprobs, success)
        )
    if not groups or len({group.seed for group in groups}) != len(groups):
        raise RuntimeError(f"empty or duplicate candidate groups in {root}")
    if len({group.resolved_seed for group in groups}) != len(groups):
        raise RuntimeError(f"duplicate resolved RoboTwin scenes in {root}")
    widths = {len(group.actions) for group in groups}
    if len(widths) != 1:
        raise RuntimeError(f"candidate count differs across groups: {widths}")
    return groups


def load_semantic_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    semantic = {
        key.removeprefix("semantic."): value
        for key, value in state.items()
        if key.startswith("semantic.")
    }
    expected = set(SemanticEncoder(4096).state_dict())
    if set(semantic) != expected:
        raise RuntimeError("shadow checkpoint does not contain a complete semantic encoder")
    return semantic


def raw_action_features(
    actions: torch.Tensor,
    baseline: torch.Tensor,
    source_logprobs: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    x = (actions - action_mean[None, None, :]) / action_std[None, None, :]
    delta = (actions - baseline) / action_std[None, None, :]
    x_stats = torch.cat(
        [
            x.mean(1),
            x.std(1, unbiased=False),
            x.amin(1),
            x.amax(1),
            x[:, 0],
            x[:, -1],
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
    velocity = x[:, 1:] - x[:, :-1]
    velocity_stats = torch.cat(
        [velocity.abs().mean(1), velocity.abs().amax(1)], dim=1
    )
    logprob_stats = torch.stack(
        [
            source_logprobs.mean(1),
            source_logprobs.std(1, unbiased=False),
            source_logprobs.amin(1),
            source_logprobs.amax(1),
        ],
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
    return torch.cat([x_stats, delta_stats, velocity_stats, logprob_stats, scalars], dim=1)


class ActionConditionedETSF(nn.Module):
    def __init__(
        self,
        semantic_state: dict[str, torch.Tensor],
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.semantic = SemanticEncoder(4096)
        self.semantic.load_state_dict(semantic_state)
        for parameter in self.semantic.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float())
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float())
        feature_dim = int(feature_mean.numel())
        self.action_encoder = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, BRIDGE),
            nn.LayerNorm(BRIDGE),
        )
        self.head = nn.Sequential(
            nn.Linear(BRIDGE * 3, 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        baseline: torch.Tensor,
        source_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.ones((len(hidden), 1), device=hidden.device, dtype=torch.bool)
        with torch.no_grad():
            semantic = self.semantic(hidden[:, None], mask)[:, 0]
        raw = raw_action_features(
            actions, baseline, source_logprobs, self.action_mean, self.action_std
        )
        features = (raw - self.feature_mean[None]) / self.feature_std[None]
        action = self.action_encoder(features)
        joined = torch.cat([semantic, action, semantic * action], dim=1)
        return self.head(joined).squeeze(1)


def action_statistics(groups: list[CandidateGroup], device: torch.device):
    actions = torch.as_tensor(
        np.concatenate([group.actions for group in groups]), device=device
    )
    action_mean = actions.mean((0, 1))
    action_std = actions.std((0, 1), unbiased=False).clamp_min(1e-2)
    baselines = torch.cat(
        [
            torch.as_tensor(group.actions[0], device=device)[None].expand(
                len(group.actions), -1, -1
            )
            for group in groups
        ]
    )
    logprobs = torch.as_tensor(
        np.concatenate([group.source_logprobs for group in groups]), device=device
    )
    raw = raw_action_features(actions, baselines, logprobs, action_mean, action_std)
    return (
        action_mean,
        action_std,
        raw.mean(0),
        raw.std(0, unbiased=False).clamp_min(1e-3),
    )


def pack(groups: list[CandidateGroup], device: torch.device) -> dict[str, torch.Tensor]:
    widths = [len(group.actions) for group in groups]
    group_ids = np.repeat(np.arange(len(groups)), widths)
    candidate_ids = np.concatenate([np.arange(width) for width in widths])
    hidden = np.repeat(
        np.stack([group.hidden for group in groups]), widths[0], axis=0
    )
    actions = np.concatenate([group.actions for group in groups])
    baseline = np.concatenate(
        [np.repeat(group.actions[0:1], len(group.actions), axis=0) for group in groups]
    )
    logprobs = np.concatenate([group.source_logprobs for group in groups])
    success = np.concatenate([group.success for group in groups])
    return {
        "hidden": torch.as_tensor(hidden, device=device),
        "actions": torch.as_tensor(actions, device=device),
        "baseline": torch.as_tensor(baseline, device=device),
        "logprobs": torch.as_tensor(logprobs, device=device),
        "success": torch.as_tensor(success, device=device),
        "group_ids": torch.as_tensor(group_ids, device=device),
        "candidate_ids": torch.as_tensor(candidate_ids, device=device),
    }


def pairwise_loss(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor):
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        positive = logits[mask & (labels > 0.5)]
        negative = logits[mask & (labels < 0.5)]
        if len(positive) and len(negative):
            losses.append(F.softplus(-(positive[:, None] - negative[None])).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    bootstrap_seed: int,
    minimum_probability_margin: float = 0.0,
    maximum_normalized_distance: float = math.inf,
):
    model.eval()
    logits = model(batch["hidden"], batch["actions"], batch["baseline"], batch["logprobs"])
    probabilities = torch.sigmoid(logits).cpu().numpy()
    labels = batch["success"].cpu().numpy().astype(bool)
    group_ids = batch["group_ids"].cpu().numpy()
    candidate_ids = batch["candidate_ids"].cpu().numpy()
    normalized_distance = torch.sqrt(
        (
            (batch["actions"] - batch["baseline"])
            / model.action_std[None, None, :]
        )
        .square()
        .mean((1, 2))
    ).cpu().numpy()
    selected = []
    baseline = []
    oracle = []
    selected_ids = []
    pair_correct = []
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        proposed = indices[np.argmax(probabilities[indices])]
        base = indices[np.flatnonzero(candidate_ids[indices] == 0)[0]]
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
        positives = probabilities[indices][labels[indices]]
        negatives = probabilities[indices][~labels[indices]]
        if len(positives) and len(negatives):
            pair_correct.extend((positives[:, None] > negatives[None]).reshape(-1).tolist())
    selected_array = np.asarray(selected)
    baseline_array = np.asarray(baseline)
    difference = selected_array - baseline_array
    improved = int(np.sum(difference > 0))
    harmed = int(np.sum(difference < 0))
    rng = np.random.default_rng(bootstrap_seed)
    boot = difference[
        rng.integers(0, len(difference), size=(5000, len(difference)))
    ].mean(1)
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
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
        "paired_difference_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "improved_groups": improved,
        "harmed_groups": harmed,
        "unchanged_groups": int(len(difference) - improved - harmed),
        "changed_groups": int(np.sum(np.asarray(selected_ids) != 0)),
        "selected_candidate_histogram": {
            str(index): int(np.sum(np.asarray(selected_ids) == index))
            for index in sorted(set(selected_ids))
        },
        "candidate_auc": auc,
        "candidate_brier": float(np.mean(np.square(probabilities - labels))),
        "within_group_pair_accuracy": float(np.mean(pair_correct)) if pair_correct else None,
        "selected_candidate_ids": selected_ids,
        "guard": {
            "minimum_probability_margin": minimum_probability_margin,
            "maximum_normalized_distance": maximum_normalized_distance,
        },
    }


def model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> ActionConditionedETSF:
    state = checkpoint["model"]
    semantic_state = {
        key.removeprefix("semantic."): value
        for key, value in state.items()
        if key.startswith("semantic.")
    }
    model = ActionConditionedETSF(
        semantic_state,
        state["action_mean"],
        state["action_std"],
        state["feature_mean"],
        state["feature_std"],
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def validation_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    pair_accuracy = metrics["within_group_pair_accuracy"]
    auc = metrics["candidate_auc"]
    return (
        metrics["selected_success_rate"],
        pair_accuracy if pair_accuracy is not None else -1.0,
        auc if auc is not None else -1.0,
        -metrics["candidate_brier"],
    )


def tune_guard(
    model: nn.Module,
    validation_batch: dict[str, torch.Tensor],
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = []
    selected = None
    selected_score = None
    for margin in [0.0, 0.02, 0.05, 0.10, 0.15]:
        for distance in [0.10, 0.15, 0.20, 0.25, 0.35, 1e9]:
            metrics = evaluate(
                model,
                validation_batch,
                bootstrap_seed,
                minimum_probability_margin=margin,
                maximum_normalized_distance=distance,
            )
            trials.append(metrics)
            score = (
                metrics["selected_success_rate"],
                metrics["paired_difference_ci95"][0],
                -metrics["changed_groups"],
                margin,
                -distance,
            )
            if selected_score is None or score > selected_score:
                selected_score = score
                selected = metrics
    assert selected is not None
    return selected, trials


def train_one(
    seed: int,
    semantic_state: dict[str, torch.Tensor],
    stats: tuple[torch.Tensor, ...],
    train_batch: dict[str, torch.Tensor],
    validation_batch: dict[str, torch.Tensor],
    steps: int,
    learning_rate: float,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ActionConditionedETSF(semantic_state, *stats).to(train_batch["hidden"].device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-3)
    labels = train_batch["success"]
    positive = float(labels.sum())
    negative = float(len(labels) - positive)
    pos_weight = torch.tensor(min(negative / max(positive, 1.0), 10.0), device=labels.device)
    best_state = None
    best_metrics = None
    best_score = None
    patience = max(200, steps // 5)
    last_improvement = 0
    for step in range(steps):
        model.train()
        logits = model(
            train_batch["hidden"],
            train_batch["actions"],
            train_batch["baseline"],
            train_batch["logprobs"],
        )
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        ranking = pairwise_loss(logits, labels, train_batch["group_ids"])
        loss = bce + ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 2.0)
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
        raise RuntimeError("action Q training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_metrics, step + 1


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--shadow-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260826, 20260827, 20260828, 20260829, 20260830])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_groups = load_groups(args.train)
    validation_groups = load_groups(args.validation)
    train_manifest = json.loads(
        (args.train / "manifest.json").read_text(encoding="utf-8")
    )
    validation_manifest = json.loads(
        (args.validation / "manifest.json").read_text(encoding="utf-8")
    )
    overlap = {group.seed for group in train_groups} & {group.seed for group in validation_groups}
    if overlap:
        raise RuntimeError(f"train/validation seed leakage: {sorted(overlap)}")
    resolved_overlap = {
        group.resolved_seed for group in train_groups
    } & {group.resolved_seed for group in validation_groups}
    if resolved_overlap:
        raise RuntimeError(
            f"train/validation resolved-scene leakage: {sorted(resolved_overlap)}"
        )
    contract_keys = [
        "task",
        "body",
        "model_path",
        "unnorm_key",
        "candidate_count",
        "blends",
        "temperature",
        "top_k",
        "preserve_grippers",
        "intervention",
        "language_contract",
    ]
    for key in contract_keys:
        if train_manifest.get(key) != validation_manifest.get(key):
            raise RuntimeError(f"train/validation contract mismatch: {key}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    semantic_state = load_semantic_state(args.shadow_checkpoint)
    stats = action_statistics(train_groups, device)
    train_batch = pack(train_groups, device)
    validation_batch = pack(validation_groups, device)
    runs = []
    selected_model = None
    selected_score = None
    selected_seed = None
    for seed in args.seeds:
        model, validation_metrics, trained_steps = train_one(
            seed,
            semantic_state,
            stats,
            train_batch,
            validation_batch,
            args.steps,
            args.learning_rate,
        )
        train_metrics = evaluate(model, train_batch, seed + 100000)
        guarded_validation, guard_trials = tune_guard(
            model, validation_batch, seed + 200000
        )
        score = validation_score(guarded_validation)
        runs.append(
            {
                "seed": seed,
                "trained_steps": trained_steps,
                "train": train_metrics,
                "validation_unguarded": validation_metrics,
                "validation": guarded_validation,
                "guard_trials": guard_trials,
            }
        )
        if selected_score is None or score > selected_score:
            selected_score = score
            selected_model = model
            selected_seed = seed
        print(json.dumps(runs[-1], sort_keys=True), flush=True)
    assert selected_model is not None and selected_seed is not None
    selected_validation_unguarded = next(
        run["validation_unguarded"] for run in runs if run["seed"] == selected_seed
    )
    selected_run = next(run for run in runs if run["seed"] == selected_seed)
    selected_validation = selected_run["validation"]
    guard_trials = selected_run["guard_trials"]
    selected_guard = selected_validation["guard"]
    gate_checks = {
        "validation_has_at_least_two_mixed_groups": selected_validation[
            "groups_with_outcome_variation"
        ]
        >= 2,
        "validation_not_worse": selected_validation["selected_success_rate"]
        >= selected_validation["baseline_success_rate"],
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
    checkpoint = args.output / "openvla_etsf_action_q_selected.pt"
    torch.save(
        {
            "model": selected_model.state_dict(),
            "selected_seed": selected_seed,
            "shadow_checkpoint": str(args.shadow_checkpoint),
            "shadow_checkpoint_sha256": sha256(args.shadow_checkpoint),
            "action_dim": ACTION_DIM,
            "action_chunk": CHUNK,
            "feature_dim": int(selected_model.feature_mean.numel()),
            "candidate_count": len(train_groups[0].actions),
            "task": train_manifest["task"],
            "body": train_manifest["body"],
            "frozen_actor_checkpoint": train_manifest["model_path"],
            "unnorm_key": train_manifest["unnorm_key"],
            "candidate_contract": {
                key: train_manifest.get(key)
                for key in contract_keys
            },
            "train_requested_seeds": sorted(group.seed for group in train_groups),
            "train_resolved_seeds": sorted(
                group.resolved_seed for group in train_groups
            ),
            "validation_requested_seeds": sorted(
                group.seed for group in validation_groups
            ),
            "validation_resolved_seeds": sorted(
                group.resolved_seed for group in validation_groups
            ),
            "guard": selected_guard,
            "action_ranking_authorized": authorized,
            "gate_checks": gate_checks,
        },
        checkpoint,
    )
    summary = {
        "schema_version": 1,
        "method": "Q_ETSF(hidden, first_action_chunk)",
        "intervention": "candidate first chunk, then frozen deterministic OpenVLA",
        "openvla_frozen": True,
        "semantic_encoder_frozen": True,
        "train_root": str(args.train),
        "validation_root": str(args.validation),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_successful_candidates": int(sum(group.success.sum() for group in train_groups)),
        "validation_successful_candidates": int(sum(group.success.sum() for group in validation_groups)),
        "selected_seed": selected_seed,
        "runs": runs,
        "selected_validation_unguarded": selected_validation_unguarded,
        "guard_trials": guard_trials,
        "selected_guard": selected_guard,
        "selected_validation": selected_validation,
        "gate_checks": gate_checks,
        "action_ranking_authorized": authorized,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "limitations": [
            "Only the first 25-step chunk is reranked in this development experiment.",
            "The action head is supervised by simulator branches, not counterfactual labels inferred from logs.",
            "Test seeds must remain unseen until the checkpoint and selection rule are frozen.",
        ],
    }
    atomic_json(args.output / "action_q_summary.json", summary)
    print("TRAINING_COMPLETE=" + json.dumps({"selected_seed": selected_seed, "authorized": authorized, "validation": selected_validation}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
