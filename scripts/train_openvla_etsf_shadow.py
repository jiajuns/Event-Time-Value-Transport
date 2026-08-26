#!/usr/bin/env python3
"""Train and gate a frozen-OpenVLA ETSF shadow model from collected rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn


EVENTS = ["e0", "e12", "e3", "e4", "eK"]
BRIDGE = 96
CLOCK = 48
SPLIT_SEED = 20260826


@dataclass
class Episode:
    index: int
    seed: int
    success: bool
    hidden: np.ndarray
    times: np.ndarray
    event_names: list[str]
    event_steps: np.ndarray


class SemanticEncoder(nn.Module):
    def __init__(self, input_dim: int = 4096) -> None:
        super().__init__()
        self.bridge = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, BRIDGE),
            nn.GELU(),
            nn.Linear(BRIDGE, BRIDGE),
            nn.LayerNorm(BRIDGE),
        )
        self.cell = nn.GRUCell(BRIDGE, BRIDGE)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = inputs.new_zeros(inputs.shape[0], BRIDGE)
        outputs = []
        for column in range(inputs.shape[1]):
            proposal = self.cell(self.bridge(inputs[:, column]), state)
            state = torch.where(mask[:, column, None], proposal, state)
            outputs.append(state)
        return torch.stack(outputs, 1)


class ClockLNNCell(nn.Module):
    """Continuous-time low-rank cell confined to the clock branch."""

    def __init__(self) -> None:
        super().__init__()
        width = BRIDGE + CLOCK
        self.candidate = nn.Linear(width, CLOCK)
        self.log_tau = nn.Linear(width, CLOCK)

    def forward(
        self, semantic: torch.Tensor, state: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined = torch.cat([semantic, state], -1)
        proposal = torch.tanh(self.candidate(joined))
        log_tau = math.log(10.0) + 2.0 * torch.tanh(self.log_tau(joined))
        decay = torch.exp(-dt[:, None] / torch.exp(log_tau))
        return decay * state + (1.0 - decay) * proposal, log_tau


class OpenVLAETSFShadow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic = SemanticEncoder(4096)
        self.event_head = nn.Linear(BRIDGE, len(EVENTS))
        self.reach_head = nn.Linear(BRIDGE, len(EVENTS))
        self.success_head = nn.Linear(BRIDGE, 1)
        self.clock_cell = ClockLNNCell()
        self.duration_mean = nn.Linear(CLOCK, 1)
        self.duration_scale = nn.Linear(CLOCK, 1)

    def forward(
        self, inputs: torch.Tensor, dts: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        semantic = self.semantic(inputs, mask)
        clock_state = inputs.new_zeros(inputs.shape[0], CLOCK)
        duration_mean = []
        duration_scale = []
        log_taus = []
        for column in range(inputs.shape[1]):
            # The stop-gradient is a hard module boundary: timing labels cannot
            # rewrite the representation used for semantic value ordering.
            proposal, log_tau = self.clock_cell(
                semantic[:, column].detach(), clock_state, dts[:, column]
            )
            clock_state = torch.where(mask[:, column, None], proposal, clock_state)
            duration_mean.append(F.softplus(self.duration_mean(clock_state)).squeeze(-1))
            duration_scale.append(torch.clamp(self.duration_scale(clock_state).squeeze(-1), -3.0, 2.0))
            log_taus.append(log_tau)
        return {
            "semantic": semantic,
            "event_logits": self.event_head(semantic),
            "reach_logits": self.reach_head(semantic),
            "success_logits": self.success_head(semantic).squeeze(-1),
            "duration_log_mean": torch.stack(duration_mean, 1),
            "duration_log_scale": torch.stack(duration_scale, 1),
            "clock_log_tau": torch.stack(log_taus, 1),
        }


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def load_episodes(root: Path) -> list[Episode]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("collection manifest is not complete")
    episodes = []
    for item in manifest["episodes"]:
        path = root / "episodes" / item["path"]
        with h5py.File(path, "r") as handle:
            query_hidden = handle["hidden"][:].astype(np.float32)
            terminal_hidden = handle["terminal_hidden"][:].astype(np.float32)[None]
            query_steps = handle["query_steps"][:].astype(np.int32)
            terminal_step = np.asarray([int(handle.attrs["steps"])], dtype=np.int32)
            hidden = np.concatenate([query_hidden, terminal_hidden], axis=0)
            times = np.concatenate([query_steps, terminal_step])
            if hidden.shape != (len(times), 4096) or not np.isfinite(hidden).all():
                raise RuntimeError(f"invalid hidden sequence in {path}")
            episodes.append(
                Episode(
                    index=int(item["index"]),
                    seed=int(handle.attrs["seed"]),
                    success=bool(handle.attrs["success"]),
                    hidden=hidden,
                    times=times,
                    event_names=decode_strings(handle["event_names"][:]),
                    event_steps=handle["event_steps"][:].astype(np.int32),
                )
            )
    if len({episode.seed for episode in episodes}) != len(episodes):
        raise RuntimeError("duplicate rollout seeds")
    return episodes


def audit_dataset(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest["episodes"]:
        path = root / "episodes" / item["path"]
        with h5py.File(path, "r") as handle:
            event_names = decode_strings(handle["event_names"][:])
            steps = int(handle.attrs["steps"])
            success = bool(handle.attrs["success"])
            rows.append(
                {
                    "seed": int(handle.attrs["seed"]),
                    "success": success,
                    "steps": steps,
                    "hidden_shape_ok": handle["hidden"].shape[1] == 4096,
                    "terminal_hidden_ok": handle["terminal_hidden"].shape == (4096,),
                    "actions_aligned": handle["executed_actions"].shape == (steps, 14),
                    "poses_aligned": handle["object_poses"].shape[0] == steps + 1,
                    "terminal_rgb_present": handle["terminal_rgb"].shape[-1] == 3,
                    "failure_terminal_flag_ok": bool(handle.attrs["failure_terminal_retained"]) == (not success),
                    "goal_event_matches_outcome": ("eK" in event_names) == success,
                    "canonical_chain_has_gap": bool(handle.attrs["canonical_chain_has_gap"]),
                }
            )
    invariant_keys = [
        "hidden_shape_ok",
        "terminal_hidden_ok",
        "actions_aligned",
        "poses_aligned",
        "terminal_rgb_present",
        "failure_terminal_flag_ok",
        "goal_event_matches_outcome",
    ]
    checks = {
        "manifest_complete": manifest.get("status") == "complete",
        "requested_count_matches": len(rows) == len(manifest.get("requested_seeds", [])),
        "unique_seeds": len({row["seed"] for row in rows}) == len(rows),
        **{key: all(row[key] for row in rows) for key in invariant_keys},
    }
    result = {
        "n_episodes": len(rows),
        "n_success": sum(row["success"] for row in rows),
        "n_failure": sum(not row["success"] for row in rows),
        "n_canonical_chain_gaps": sum(row["canonical_chain_has_gap"] for row in rows),
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"rollout audit failed: {result}")
    return result


def split_episodes(episodes: list[Episode]) -> dict[str, list[Episode]]:
    if len(episodes) < 100:
        raise RuntimeError("formal shadow training requires at least 100 rollouts")
    indices = np.arange(len(episodes))
    labels = np.asarray([episode.success for episode in episodes], dtype=np.int64)
    development, test = train_test_split(
        indices, test_size=25, random_state=SPLIT_SEED, stratify=labels
    )
    train, validation = train_test_split(
        development,
        test_size=25,
        random_state=SPLIT_SEED + 1,
        stratify=labels[development],
    )
    return {
        "train": [episodes[index] for index in sorted(train)],
        "validation": [episodes[index] for index in sorted(validation)],
        "test": [episodes[index] for index in sorted(test)],
    }


def pack(episodes: list[Episode], device: torch.device) -> dict[str, torch.Tensor]:
    width = max(len(episode.times) for episode in episodes)
    batch = len(episodes)
    inputs = np.zeros((batch, width, 4096), dtype=np.float32)
    dts = np.zeros((batch, width), dtype=np.float32)
    mask = np.zeros((batch, width), dtype=bool)
    terminal = np.zeros((batch, width), dtype=bool)
    event_ids = np.zeros((batch, width), dtype=np.int64)
    reach = np.zeros((batch, width, len(EVENTS)), dtype=np.float32)
    labels = np.zeros((batch, width), dtype=np.float32)
    duration = np.zeros((batch, width), dtype=np.float32)
    clock_mask = np.zeros((batch, width), dtype=bool)

    for row, episode in enumerate(episodes):
        length = len(episode.times)
        inputs[row, :length] = episode.hidden
        mask[row, :length] = True
        terminal[row, length - 1] = True
        labels[row, :length] = float(episode.success)
        dts[row, 0] = 1.0
        dts[row, 1:length] = np.diff(episode.times).clip(min=1)
        achieved = set(episode.event_names)
        reach[row, :length] = np.asarray([float(event in achieved) for event in EVENTS])
        event_pairs = sorted(zip(episode.event_names, episode.event_steps), key=lambda pair: pair[1])
        for column, now in enumerate(episode.times):
            current = 0
            for name, step in event_pairs:
                if step <= now:
                    current = EVENTS.index(name)
            event_ids[row, column] = current
            if episode.success and not terminal[row, column]:
                future = [step for _, step in event_pairs if step > now]
                if future:
                    duration[row, column] = float(future[0] - now)
                    clock_mask[row, column] = duration[row, column] >= 1.0
    return {
        "inputs": torch.from_numpy(inputs).to(device),
        "dts": torch.from_numpy(dts).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "terminal": torch.from_numpy(terminal).to(device),
        "event_ids": torch.from_numpy(event_ids).to(device),
        "reach": torch.from_numpy(reach).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "duration": torch.from_numpy(duration).to(device),
        "clock_mask": torch.from_numpy(clock_mask).to(device),
    }


def last_per_event(mask: torch.Tensor, event_ids: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(mask)
    for row in range(len(mask)):
        for event_id in range(len(EVENTS) - 1):
            columns = torch.nonzero(
                mask[row] & ~terminal[row] & (event_ids[row] == event_id), as_tuple=False
            ).flatten()
            if len(columns):
                selected[row, columns[-1]] = True
    return selected


def ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    event_ids: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    pieces = []
    pairs = 0
    for event_id in range(len(EVENTS) - 1):
        group = selected & (event_ids == event_id)
        positive = scores[group & (labels == 1)]
        negative = scores[group & (labels == 0)]
        if len(positive) and len(negative):
            differences = positive[:, None] - negative[None, :]
            pieces.append(F.softplus(0.2 - differences).mean())
            pairs += differences.numel()
    if not pieces:
        return scores.sum() * 0.0, 0
    return torch.stack(pieces).mean(), pairs


def loss_function(
    model: OpenVLAETSFShadow,
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor,
    positive_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    local = {key: value[indices] for key, value in batch.items()}
    output = model(local["inputs"], local["dts"], local["mask"])
    mask = local["mask"]
    semantic_mask = mask & ~local["terminal"]
    event = F.cross_entropy(
        output["event_logits"][semantic_mask], local["event_ids"][semantic_mask]
    )
    reach = F.binary_cross_entropy_with_logits(
        output["reach_logits"][mask], local["reach"][mask]
    )
    weights = torch.where(
        local["labels"] == 1,
        local["labels"].new_tensor(positive_weight),
        local["labels"].new_tensor(1.0),
    )
    success_raw = F.binary_cross_entropy_with_logits(
        output["success_logits"], local["labels"], reduction="none"
    )
    success = (success_raw * weights)[mask].mean()
    selected = last_per_event(mask, local["event_ids"], local["terminal"])
    rank, pairs = ranking_loss(
        output["success_logits"], local["labels"], local["event_ids"], selected
    )
    clock_mask = local["clock_mask"]
    if clock_mask.any():
        target = torch.log1p(local["duration"])
        mean = output["duration_log_mean"]
        scale = torch.exp(output["duration_log_scale"]).clamp(min=1e-3)
        clock_raw = 0.5 * torch.square((target - mean) / scale) + torch.log(scale)
        clock = clock_raw[clock_mask].mean()
    else:
        clock = output["duration_log_mean"].sum() * 0.0
    total = event + reach + success + 1.5 * rank + 0.5 * clock
    return total, {
        "total": float(total.detach()),
        "event": float(event.detach()),
        "reach": float(reach.detach()),
        "success": float(success.detach()),
        "rank": float(rank.detach()),
        "rank_pairs": pairs,
        "clock": float(clock.detach()),
    }


def rank_auc(labels: list[int], scores: list[float]) -> tuple[float, int]:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    correct = 0.0
    for positive in positives:
        for negative in negatives:
            correct += float(positive > negative) + 0.5 * float(positive == negative)
    pairs = len(positives) * len(negatives)
    return (correct / pairs if pairs else math.nan), pairs


def evaluate(
    model: OpenVLAETSFShadow,
    episodes: list[Episode],
    batch: dict[str, torch.Tensor],
    clock_baseline: dict[int, float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    with torch.no_grad():
        output = model(batch["inputs"], batch["dts"], batch["mask"])
    probabilities = torch.sigmoid(output["success_logits"])
    selected = last_per_event(batch["mask"], batch["event_ids"], batch["terminal"])
    per_event = []
    weighted_correct = 0.0
    same_pairs = 0
    rows = []
    for event_id, event in enumerate(EVENTS[:-1]):
        group = selected & (batch["event_ids"] == event_id)
        labels = batch["labels"][group].int().cpu().tolist()
        scores = probabilities[group].cpu().tolist()
        auc, pairs = rank_auc(labels, scores)
        if pairs:
            per_event.append(auc)
            weighted_correct += auc * pairs
            same_pairs += pairs
        rows.append({"event": event, "auc": auc, "pairs": pairs, "n": len(labels)})
    same_auc = weighted_correct / same_pairs if same_pairs else math.nan
    macro_auc = float(np.mean(per_event)) if per_event else math.nan

    start_labels = [int(episode.success) for episode in episodes]
    start_scores = probabilities[:, 0].cpu().tolist()
    start_auc, start_pairs = rank_auc(start_labels, start_scores)
    terminal_columns = batch["mask"].sum(1) - 1
    terminal_scores = [float(probabilities[row, column]) for row, column in enumerate(terminal_columns)]
    terminal_auc, terminal_pairs = rank_auc(start_labels, terminal_scores)

    selected_scores = probabilities[selected]
    selected_labels = batch["labels"][selected]
    brier = float(torch.square(selected_scores - selected_labels).mean())
    base_probabilities = []
    for event_id in batch["event_ids"][selected].cpu().tolist():
        train_rate = clock_baseline.get(100 + event_id, clock_baseline[100])
        base_probabilities.append(train_rate)
    baseline_brier = float(np.mean(np.square(np.asarray(base_probabilities) - selected_labels.cpu().numpy())))

    duration_prediction = torch.expm1(output["duration_log_mean"]).clamp(0, 200)
    observed = batch["clock_mask"]
    if observed.any():
        target = batch["duration"][observed]
        prediction = duration_prediction[observed]
        clock_mae = float(torch.abs(prediction - target).mean())
        baseline_predictions = []
        for event_id in batch["event_ids"][observed].cpu().tolist():
            baseline_predictions.append(clock_baseline.get(event_id, clock_baseline[-1]))
        baseline_clock_mae = float(np.mean(np.abs(np.asarray(baseline_predictions) - target.cpu().numpy())))
    else:
        clock_mae = math.nan
        baseline_clock_mae = math.nan

    event_prediction = output["event_logits"].argmax(-1)
    semantic_mask = batch["mask"] & ~batch["terminal"]
    event_accuracy = float((event_prediction[semantic_mask] == batch["event_ids"][semantic_mask]).float().mean())
    metrics = {
        "n_episodes": len(episodes),
        "n_success": sum(episode.success for episode in episodes),
        "n_failure": sum(not episode.success for episode in episodes),
        "same_event_micro_auc": same_auc,
        "same_event_macro_auc": macro_auc,
        "same_event_pairs": same_pairs,
        "event_only_same_event_auc": 0.5,
        "start_auc_diagnostic": start_auc,
        "start_auc_pairs": start_pairs,
        "terminal_auc_leaky_diagnostic": terminal_auc,
        "terminal_auc_pairs": terminal_pairs,
        "same_event_brier": brier,
        "event_rate_baseline_brier": baseline_brier,
        "clock_duration_mae": clock_mae,
        "event_median_duration_mae": baseline_clock_mae,
        "event_class_accuracy": event_accuracy,
        "event_index_oracle_accuracy": 1.0,
    }
    return metrics, rows


def baselines(train: list[Episode], train_batch: dict[str, torch.Tensor]) -> dict[int, float]:
    result: dict[int, float] = {}
    observed = train_batch["clock_mask"]
    for event_id in range(len(EVENTS)):
        values = train_batch["duration"][observed & (train_batch["event_ids"] == event_id)]
        if len(values):
            result[event_id] = float(values.median())
    all_duration = train_batch["duration"][observed]
    result[-1] = float(all_duration.median()) if len(all_duration) else 25.0
    labels = train_batch["labels"]
    selected = last_per_event(train_batch["mask"], train_batch["event_ids"], train_batch["terminal"])
    result[100] = float(labels[selected].mean())
    for event_id in range(len(EVENTS)):
        group = selected & (train_batch["event_ids"] == event_id)
        if group.any():
            result[100 + event_id] = float(labels[group].mean())
    return result


def bootstrap_auc_lower(
    model: OpenVLAETSFShadow,
    episodes: list[Episode],
    batch: dict[str, torch.Tensor],
    samples: int = 1000,
) -> float:
    with torch.no_grad():
        output = model(batch["inputs"], batch["dts"], batch["mask"])
    probabilities = torch.sigmoid(output["success_logits"])
    selected = last_per_event(batch["mask"], batch["event_ids"], batch["terminal"])
    episode_scores: list[dict[int, float]] = []
    for row in range(len(episodes)):
        scores = {}
        for event_id in range(len(EVENTS) - 1):
            columns = torch.nonzero(
                selected[row] & (batch["event_ids"][row] == event_id), as_tuple=False
            ).flatten()
            if len(columns):
                scores[event_id] = float(probabilities[row, columns[0]])
        episode_scores.append(scores)
    rng = np.random.default_rng(SPLIT_SEED + 9)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(episodes), len(episodes))
        weighted = 0.0
        pairs = 0
        for event_id in range(len(EVENTS) - 1):
            labels = [int(episodes[index].success) for index in indices if event_id in episode_scores[index]]
            scores = [episode_scores[index][event_id] for index in indices if event_id in episode_scores[index]]
            auc, count = rank_auc(labels, scores)
            if count:
                weighted += auc * count
                pairs += count
        if pairs:
            values.append(weighted / pairs)
    return float(np.quantile(values, 0.025)) if values else math.nan


def train_one(
    train_batch: dict[str, torch.Tensor],
    validation_episodes: list[Episode],
    validation_batch: dict[str, torch.Tensor],
    clock_baseline: dict[int, float],
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[OpenVLAETSFShadow, list[dict[str, float]], dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = OpenVLAETSFShadow().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    labels = train_batch["labels"][:, 0]
    positives = torch.nonzero(labels == 1, as_tuple=False).flatten()
    negatives = torch.nonzero(labels == 0, as_tuple=False).flatten()
    # Outcome-balanced episode sampling already corrects the class imbalance;
    # applying a second positive-class weight would over-correct it.
    positive_weight = 1.0
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    history = []
    started = time.time()
    for step in range(steps):
        size = min(64, len(labels))
        positive_count = min(size // 2, max(1, len(positives)))
        negative_count = size - positive_count
        indices = torch.cat(
            [
                positives[torch.randint(len(positives), (positive_count,), generator=generator, device=device)],
                negatives[torch.randint(len(negatives), (negative_count,), generator=generator, device=device)],
            ]
        )
        loss, train_metrics = loss_function(model, train_batch, indices, positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (step + 1) % 500 == 0 or step + 1 == steps:
            validation_metrics, _ = evaluate(
                model.eval(), validation_episodes, validation_batch, clock_baseline
            )
            row = {
                "step": step + 1,
                "wall_seconds": time.time() - started,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
            history.append(row)
            print(f"SHADOW_TRAIN={seed}:{step + 1}/{steps} " + json.dumps(row, sort_keys=True), flush=True)
            model.train()
    validation_metrics, _ = evaluate(
        model.eval(), validation_episodes, validation_batch, clock_baseline
    )
    return model.eval(), history, validation_metrics


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260826, 20260827, 20260828, 20260829, 20260830])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audit = audit_dataset(args.data)
    atomic_json(args.output / "data_audit.json", audit)
    episodes = load_episodes(args.data)
    splits = split_episodes(episodes)
    split_manifest = {
        name: [
            {"index": episode.index, "seed": episode.seed, "success": episode.success}
            for episode in group
        ]
        for name, group in splits.items()
    }
    atomic_json(args.output / "split_manifest.json", split_manifest)
    batches = {name: pack(group, device) for name, group in splits.items()}
    clock_baseline = baselines(splits["train"], batches["train"])
    candidates = []
    for seed in args.seeds:
        model, history, validation = train_one(
            batches["train"],
            splits["validation"],
            batches["validation"],
            clock_baseline,
            device,
            args.steps,
            seed,
        )
        model_path = args.output / f"shadow_seed_{seed}.pt"
        torch.save({"model": model.state_dict(), "seed": seed, "events": EVENTS}, model_path)
        atomic_json(args.output / f"history_seed_{seed}.json", history)
        score = validation["same_event_micro_auc"]
        if math.isfinite(validation["clock_duration_mae"]):
            score -= 0.05 * validation["clock_duration_mae"] / max(
                validation["event_median_duration_mae"], 1e-6
            )
        candidates.append((score, seed, model, validation))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    _, selected_seed, selected_model, validation_metrics = candidates[0]
    test_metrics, per_event = evaluate(
        selected_model, splits["test"], batches["test"], clock_baseline
    )
    lower = bootstrap_auc_lower(
        selected_model, splits["test"], batches["test"]
    )
    test_metrics["same_event_auc_episode_bootstrap_lower_95"] = lower
    checks = {
        "rollouts_at_least_100": len(episodes) >= 100,
        "heldout_has_both_outcomes": test_metrics["n_success"] >= 4 and test_metrics["n_failure"] >= 4,
        "same_event_pairs_at_least_50": test_metrics["same_event_pairs"] >= 50,
        "same_event_auc_above_event_counter": test_metrics["same_event_micro_auc"] > 0.5,
        "same_event_auc_lower_bound_above_chance": lower > 0.5,
        "semantic_brier_beats_event_rate": test_metrics["same_event_brier"] < test_metrics["event_rate_baseline_brier"],
        "clock_beats_event_median": test_metrics["clock_duration_mae"] < test_metrics["event_median_duration_mae"],
    }
    authorized = all(checks.values())
    final_checkpoint = args.output / "openvla_etsf_shadow_selected.pt"
    torch.save(
        {
            "model": selected_model.state_dict(),
            "selected_seed": selected_seed,
            "events": EVENTS,
            "input_dim": 4096,
            "bridge_dim": BRIDGE,
            "clock_dim": CLOCK,
            "openvla_frozen": True,
            "action_ranking_authorized": authorized,
        },
        final_checkpoint,
    )
    summary = {
        "status": "offline_shadow_gate_complete",
        "data_audit": audit,
        "selected_seed": selected_seed,
        "openvla_frozen": True,
        "training_scope": "bridge_semantic_heads_clocklnn_only",
        "split_counts": {name: len(group) for name, group in splits.items()},
        "split_outcomes": {
            name: {"success": sum(ep.success for ep in group), "failure": sum(not ep.success for ep in group)}
            for name, group in splits.items()
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_per_event": per_event,
        "gate_checks": checks,
        "action_ranking_authorized": authorized,
        "policy_effect_during_collection_or_shadow": False,
        "next_action": (
            "enable_candidate_scoring_in_a_separate guarded evaluation"
            if authorized
            else "keep ETSF in shadow mode and do not rank OpenVLA actions"
        ),
    }
    atomic_json(args.output / "shadow_gate_summary.json", summary)
    print("SHADOW_GATE=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
