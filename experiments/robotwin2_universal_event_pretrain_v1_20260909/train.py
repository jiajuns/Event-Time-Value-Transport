#!/usr/bin/env python3
"""Train the universal language/graph/CfC/action event world model."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from universal_event.model import FORMAT, UniversalEventWorldModel, UniversalEventWorldModelConfig, parameter_inventory
from universal_event.schema import EVENT_NAMES, MAX_NODES, RELATION_NAMES, schema_sha256


TRAINING_FORMAT = "etsf_robotwin2_universal_event_pretraining_v1"

# These are relation-preserving language augmentations, not extra simulator
# labels.  In particular, pot lid and drawer prompts share the learned
# ``open(articulated_part)`` goal with laptop/microwave trajectories.
LANGUAGE_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "place_container_plate": (
        "Put the fruit inside the plate and release it.",
        "Put the vegetable onto the plate.",
        "把水果放入盘子并松开夹爪。",
    ),
    "place_can_basket": (
        "Put the cup inside the box and release it.",
        "把杯子放进盒子。",
    ),
    "stack_blocks_two": (
        "Put block A on top of block B.",
        "把积木A堆到积木B上。",
    ),
    "open_laptop": (
        "Open the pot lid.", "Pull the drawer open.", "打开锅盖。", "打开抽屉。",
    ),
    "open_microwave": (
        "Open the pot lid.", "Pull the drawer open.", "打开锅盖。", "打开抽屉。",
    ),
}


def episode_split(path: Path, task: str, body: str, seed: int,
                  heldout_tasks: set[str], heldout_body: str | None,
                  validation_percent: int) -> str:
    if task in heldout_tasks or (heldout_body is not None and body == heldout_body):
        return "heldout"
    digest = hashlib.sha256(f"{task}|{body}|{seed}|{path.name}".encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 100 < validation_percent else "train"


class UniversalTransitionDataset(Dataset):
    def __init__(self, root: Path, split: str, *, history: int, horizon: int, stride: int,
                 heldout_tasks: set[str], heldout_body: str | None, validation_percent: int,
                 cache_episodes: int = 24) -> None:
        self.root, self.history, self.horizon = root, history, horizon
        self.cache_episodes = cache_episodes
        self.files: list[dict[str, Any]] = []
        self.samples: list[tuple[int, int]] = []
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        for path in sorted(root.rglob("*.hdf5")):
            if path.name.endswith(".partial"):
                continue
            try:
                with h5py.File(path) as handle:
                    if str(handle.attrs.get("format", "")) != "etsf_robotwin2_universal_event_episode_v1":
                        continue
                    if str(handle.attrs.get("schema_sha256", "")) != schema_sha256():
                        raise ValueError(f"schema mismatch: {path}")
                    task, body, seed = str(handle.attrs["task"]), str(handle.attrs["body"]), int(handle.attrs["seed"])
                    length = int(handle["node_features"].shape[0])
                    instruction = str(handle.attrs["instruction"])
                    condition = str(handle.attrs["condition"])
                    native_success = bool(handle.attrs.get("native_success", False))
                assigned = episode_split(path, task, body, seed, heldout_tasks, heldout_body, validation_percent)
                if assigned != split or length < history + horizon + 1:
                    continue
                file_index = len(self.files)
                self.files.append(dict(path=path, task=task, body=body, seed=seed,
                                       instruction=instruction, condition=condition,
                                       success=native_success, length=length))
                self.samples.extend((file_index, query) for query in range(history-1, length-horizon-1, stride))
            except OSError as exc:
                raise ValueError(f"cannot read dataset file {path}: {exc}") from exc
        if not self.files or not self.samples:
            raise ValueError(f"no {split} samples under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _episode(self, file_index: int) -> dict[str, np.ndarray]:
        if file_index in self._cache:
            self._cache.move_to_end(file_index)
            return self._cache[file_index]
        path = self.files[file_index]["path"]
        with h5py.File(path) as handle:
            episode = {key: np.asarray(handle[key]) for key in (
                "node_features", "node_types", "node_mask", "edge_features", "relations",
                "relation_mask", "events", "goal", "goal_mask", "actions14", "sim_times", "success"
            )}
        self._cache[file_index] = episode
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return episode

    def __getitem__(self, item: int) -> dict[str, Any]:
        file_index, query = self.samples[item]
        meta, episode = self.files[file_index], self._episode(file_index)
        start, future = query - self.history + 1, query + self.horizon
        times = episode["sim_times"]
        history_dt = np.zeros(self.history, np.float32)
        history_dt[1:] = np.diff(times[start:query+1]).astype(np.float32)
        action_dt = np.diff(times[query:future+1]).astype(np.float32)
        current_features, future_features = episode["node_features"][[query, future]]
        channels = np.asarray([0, 1, 2, 15, 17, 20, 16, 18])
        node_delta = future_features[:, channels] - current_features[:, channels]
        goal, goal_mask = episode["goal"].astype(np.float32), episode["goal_mask"].astype(bool)
        future_relations = episode["relations"][future].astype(np.float32)
        future_relation_mask = episode["relation_mask"][future].astype(bool)
        active_goal = goal_mask & future_relation_mask
        desired = np.where(goal > .5, future_relations > .5, future_relations <= .5)
        goal_satisfied = bool(active_goal.any() and desired[active_goal].all())
        language_goal = np.zeros((2, len(RELATION_NAMES)), np.float32)
        for relation in range(len(RELATION_NAMES)):
            selected = goal_mask[relation]
            language_goal[0, relation] = bool(selected.any() and (goal[relation][selected] > .5).any())
            language_goal[1, relation] = bool(selected.any() and (goal[relation][selected] <= .5).any())
        variants = (meta["instruction"], *LANGUAGE_PARAPHRASES.get(meta["task"], ()))
        language_digest = hashlib.sha256(
            f"{meta['path']}|{query}|language-v1".encode()
        ).digest()
        instruction = variants[int.from_bytes(language_digest[:4], "big") % len(variants)]
        result: dict[str, Any] = {
            "node_features": torch.from_numpy(episode["node_features"][start:query+1].astype(np.float32)),
            "node_types": torch.from_numpy(episode["node_types"][start:query+1].astype(np.int64)),
            "node_mask": torch.from_numpy(episode["node_mask"][start:query+1].astype(bool)),
            "edge_features": torch.from_numpy(episode["edge_features"][start:query+1].astype(np.float32)),
            "history_mask": torch.ones(self.history, dtype=torch.bool),
            "history_dt": torch.from_numpy(history_dt),
            "actions": torch.from_numpy(episode["actions14"][query:future].astype(np.float32)),
            "action_mask": torch.ones(self.horizon, dtype=torch.bool),
            "action_dt": torch.from_numpy(action_dt),
            "goal": torch.from_numpy(goal), "goal_mask": torch.from_numpy(goal_mask),
            "current_relations": torch.from_numpy(episode["relations"][query].astype(np.float32)),
            "current_relation_mask": torch.from_numpy(episode["relation_mask"][query].astype(bool)),
            "future_relations": torch.from_numpy(future_relations),
            "future_relation_mask": torch.from_numpy(future_relation_mask),
            "future_node_delta": torch.from_numpy(node_delta.astype(np.float32)),
            "future_node_features": torch.from_numpy(future_features.astype(np.float32)),
            "future_node_types": torch.from_numpy(episode["node_types"][future].astype(np.int64)),
            "future_node_mask": torch.from_numpy(episode["node_mask"][future].astype(bool)),
            "future_edge_features": torch.from_numpy(episode["edge_features"][future].astype(np.float32)),
            "event_target": torch.from_numpy(episode["events"][query+1:future+1].max(0).astype(np.float32)),
            "language_goal_target": torch.from_numpy(language_goal),
            "goal_satisfied": torch.tensor(float(goal_satisfied)),
            "success": torch.tensor(float(episode["success"][future])),
            "instruction": instruction, "task": meta["task"], "body": meta["body"],
            "condition": meta["condition"], "episode": str(meta["path"]), "query": query,
        }
        return result


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.bool()
    if not bool(selected.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[selected], target[selected])


def contrastive_loss(prediction: torch.Tensor, target: torch.Tensor, temperature: float = .1) -> torch.Tensor:
    prediction, target = F.normalize(prediction, dim=-1), F.normalize(target, dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def compute_loss(model: UniversalEventWorldModel, batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch)
    relation_current = masked_bce(output["current_relation_logits"], batch["current_relations"], batch["current_relation_mask"])
    relation_future = masked_bce(output["future_relation_logits"], batch["future_relations"], batch["future_relation_mask"])
    valid_nodes = batch["future_node_mask"][..., None].expand_as(batch["future_node_delta"])
    node = F.smooth_l1_loss(output["future_node_delta"][valid_nodes], batch["future_node_delta"][valid_nodes])
    event_weights = torch.tensor([2, 2, 1, 3, 3, 5, 5, 4, 4, .5], device=output["event_logits"].device)
    event = F.binary_cross_entropy_with_logits(output["event_logits"], batch["event_target"], pos_weight=event_weights)
    language = F.binary_cross_entropy_with_logits(output["language_goal_logits"], batch["language_goal_target"], pos_weight=torch.tensor(4., device=event.device))
    goal = F.binary_cross_entropy(output["goal_probability"].clamp(1e-5, 1-1e-5), batch["goal_satisfied"])
    success = F.binary_cross_entropy_with_logits(output["success_logit"], batch["success"], pos_weight=torch.tensor(5., device=event.device))
    actual_scene, _ = model.graph(batch["future_node_features"], batch["future_node_types"],
                                  batch["future_node_mask"], batch["future_edge_features"])
    contrast = contrastive_loss(output["future_features"], actual_scene)
    legacy_losses = []
    mapping = {"held_by_actor": "held_by", "supported_by_target": "supported_by", "in_target_region": "inside"}
    for legacy, relation in mapping.items():
        index = RELATION_NAMES.index(relation)
        known = batch["current_relation_mask"][:, index].flatten(1).any(1)
        if bool(known.any()):
            truth = batch["current_relations"][:, index].flatten(1).amax(1).long()
            legacy_losses.append(F.cross_entropy(output["legacy_relations"][legacy][known], truth[known]))
    legacy = torch.stack(legacy_losses).mean() if legacy_losses else event * 0
    pieces = {
        "relation_current": relation_current, "relation_future": relation_future,
        "node": node, "event": event, "language": language, "goal": goal,
        "success": success, "contrast": contrast, "legacy": legacy,
    }
    total = (0.5*relation_current + 2.0*relation_future + 0.5*node + event +
             0.5*language + goal + success + 0.25*contrast + 0.1*legacy)
    return total, {key: float(value.detach()) for key, value in pieces.items()}


@torch.no_grad()
def evaluate(model: UniversalEventWorldModel, loader: DataLoader, device: torch.device,
             maximum_batches: int) -> dict[str, Any]:
    model.eval(); totals = defaultdict(float); count = 0
    relation_tp = np.zeros(len(RELATION_NAMES)); relation_fp = relation_tp.copy(); relation_fn = relation_tp.copy()
    event_tp = np.zeros(len(EVENT_NAMES)); event_fp = event_tp.copy(); event_fn = event_tp.copy()
    goal_correct = success_correct = rows = 0
    action_sensitivity = []
    for raw in loader:
        batch = move_batch(raw, device)
        loss, pieces = compute_loss(model, batch)
        totals["loss"] += float(loss); count += 1
        for key, value in pieces.items(): totals[key] += value
        out = model(batch)
        pred = out["future_relation_logits"] > 0
        true, mask = batch["future_relations"] > .5, batch["future_relation_mask"].bool()
        for index in range(len(RELATION_NAMES)):
            selected = mask[:, index]
            relation_tp[index] += int((pred[:, index] & true[:, index] & selected).sum())
            relation_fp[index] += int((pred[:, index] & ~true[:, index] & selected).sum())
            relation_fn[index] += int((~pred[:, index] & true[:, index] & selected).sum())
        ep, et = out["event_logits"] > 0, batch["event_target"] > .5
        event_tp += (ep & et).sum(0).cpu().numpy(); event_fp += (ep & ~et).sum(0).cpu().numpy(); event_fn += (~ep & et).sum(0).cpu().numpy()
        goal_correct += int(((out["goal_probability"] >= .5) == (batch["goal_satisfied"] > .5)).sum())
        success_correct += int(((out["success_logit"] >= 0) == (batch["success"] > .5)).sum()); rows += len(ep)
        shuffled = dict(batch); shuffled["actions"] = batch["actions"].roll(1, 0)
        shuffled["action_dt"] = batch["action_dt"].roll(1, 0)
        changed = model(shuffled)["goal_probability"]
        action_sensitivity.extend((changed - out["goal_probability"]).abs().cpu().tolist())
        if count >= maximum_batches: break
    def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
        valid = tp + fn > 0
        score = 2*tp / np.maximum(2*tp + fp + fn, 1)
        return float(score[valid].mean()) if valid.any() else 0.0
    return {
        **{key: value/max(count, 1) for key, value in totals.items()},
        "relation_macro_f1": macro_f1(relation_tp, relation_fp, relation_fn),
        "event_macro_f1": macro_f1(event_tp, event_fp, event_fn),
        "goal_accuracy": goal_correct/max(rows, 1), "success_accuracy": success_correct/max(rows, 1),
        "action_shuffle_goal_probability_delta": float(np.mean(action_sensitivity)),
        "rows": rows, "batches": count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--heldout-body", choices=("none", "aloha-agilex", "arx-x5", "franka", "piper", "ur5"), default="none")
    parser.add_argument("--heldout-tasks", nargs="*", default=["open_microwave", "place_empty_cup"])
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists(): raise FileExistsError("output must be a new path")
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    heldout_body = None if args.heldout_body == "none" else args.heldout_body
    common = dict(history=args.history, horizon=args.horizon, stride=args.stride,
                  heldout_tasks=set(args.heldout_tasks), heldout_body=heldout_body,
                  validation_percent=args.validation_percent)
    train = UniversalTransitionDataset(args.data.resolve(), "train", **common)
    validation = UniversalTransitionDataset(args.data.resolve(), "validation", **common)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                              generator=generator, drop_last=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                   num_workers=max(0, args.workers//2), pin_memory=device.type == "cuda")
    model = UniversalEventWorldModel(UniversalEventWorldModelConfig()).to(device)
    migration = None
    if args.v4_checkpoint is not None:
        migration = model.load_event_v4_initialization(args.v4_checkpoint.resolve())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate*.05)
    receipt = {
        "format": TRAINING_FORMAT, "model_format": FORMAT, "schema_sha256": schema_sha256(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "v4_migration": migration, "parameters": parameter_inventory(model),
        "data": {"train_episodes": len(train.files), "train_rows": len(train),
                 "validation_episodes": len(validation.files), "validation_rows": len(validation),
                 "train_tasks": sorted({row["task"] for row in train.files}),
                 "train_bodies": sorted({row["body"] for row in train.files}),
                 "heldout_tasks": args.heldout_tasks, "heldout_body": heldout_body},
        "shared_model_receives_body_id": False, "shared_model_receives_joint_state": False,
        "action_contract": "canonical_dual_ee_se3_plus_gripper_delta14",
        "language_paraphrases": {key: list(value) for key, value in LANGUAGE_PARAPHRASES.items()},
    }
    (args.output / "protocol_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n")
    iterator = iter(train_loader); best = math.inf; history = []
    for step in range(1, args.steps+1):
        try: raw = next(iterator)
        except StopIteration: iterator = iter(train_loader); raw = next(iterator)
        batch = move_batch(raw, device); model.train()
        loss, pieces = compute_loss(model, batch)
        if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step(); scheduler.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), **pieces,
                              "lr": scheduler.get_last_lr()[0]}), flush=True)
        metrics = None
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, validation_loader, device, args.validation_batches)
            record = {"step": step, "validation": metrics}; history.append(record)
            print("VALIDATION="+json.dumps(record), flush=True)
            if metrics["loss"] < best:
                best = metrics["loss"]
                torch.save({"format": FORMAT, "config": dataclasses.asdict(model.config),
                            "model": model.state_dict(), "step": step, "validation": metrics,
                            "v4_migration": migration, "protocol": receipt}, args.output/"best.pt")
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"format": FORMAT, "config": dataclasses.asdict(model.config),
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step,
                        "validation": metrics, "protocol": receipt}, args.output/f"step_{step:06d}.pt")
            (args.output/"training_history.json").write_text(json.dumps(history, indent=2)+"\n")
    final = {**receipt, "status": "complete", "best_validation_loss": best,
             "history_entries": len(history), "best_checkpoint": str(args.output/"best.pt")}
    (args.output/"training_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
