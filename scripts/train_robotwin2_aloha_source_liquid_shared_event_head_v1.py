#!/usr/bin/env python3
"""Train the v14 Liquid-CfC head on Aloha only.

Target-embodiment payloads are not accepted by this program.  A label-blind
requested-seed split is made inside the Aloha manifest: three seed lanes train
and two select one common checkpoint step for the five-member ensemble.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import collect_robotwin2_five_body_ee_candidate_branches_v1 as collector
import robotwin2_liquid_shared_event_head_v1 as liquid_model
import train_multibody_canonical_event_world_model as core
import train_robotwin2_five_body_lobo_shared_event_head_v1 as v13


FORMAT = "etsf_robotwin2_aloha_source_liquid_training_v1"
SPLIT_FORMAT = "etsf_robotwin2_aloha_label_blind_seed_lane_split_v1"
SOURCE_BODY = liquid_model.SOURCE_BODY
HISTORY_ARRAYS = {
    "state_history",
    "state_history_mask",
    "state_history_dt",
    "event_history_id",
    "planned_action_dt",
}
REQUIRED_ARRAYS = set(v13.REQUIRED_ARRAYS) | HISTORY_ARRAYS


class LiquidTrainingError(RuntimeError):
    """The source-only liquid training contract failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_source_manifest(
    path: Path,
    expected_sha256: str,
    *,
    history_length: int,
    expected_groups: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise LiquidTrainingError("Aloha source manifest is missing or changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    logical = unsigned.pop("logical_sha256", None)
    expected_history = collector.liquid_history_contract(history_length)
    if (
        logical != canonical_sha256(unsigned)
        or manifest.get("format") != collector.LIQUID_MANIFEST_FORMAT
        or manifest.get("collector_format") != collector.LIQUID_FORMAT
        or manifest.get("body") != SOURCE_BODY
        or manifest.get("task") != collector.TASK
        or manifest.get("candidate_count") != collector.CANDIDATE_COUNT
        or manifest.get("liquid_history_contract") != expected_history
    ):
        raise LiquidTrainingError("manifest is not the frozen Aloha liquid source")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != expected_groups:
        raise LiquidTrainingError(
            f"source manifest needs exactly {expected_groups} complete decisions"
        )
    identities: set[str] = set()
    rows: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise LiquidTrainingError("manifest group descriptor is invalid")
        group_id = group.get("group_id")
        condition = group.get("condition")
        requested_seed = group.get("requested_seed")
        root_query_index = group.get("root_query_index")
        if (
            not isinstance(group_id, str)
            or group_id in identities
            or condition not in collector.CONDITIONS
            or isinstance(requested_seed, bool)
            or not isinstance(requested_seed, int)
            or isinstance(root_query_index, bool)
            or not isinstance(root_query_index, int)
            or root_query_index < 0
        ):
            raise LiquidTrainingError("manifest group identity is invalid")
        identities.add(group_id)
        payload = (path.parent / str(group.get("path", ""))).resolve()
        try:
            payload.relative_to(path.parent)
        except ValueError as error:
            raise LiquidTrainingError("group payload escapes the source root") from error
        if not payload.is_file() or sha256_file(payload) != group.get("sha256"):
            raise LiquidTrainingError(f"source group is missing/tampered: {payload}")
        rows.extend(
            _load_group_rows(
                payload,
                logical_group=f"{SOURCE_BODY}|{condition}|{group_id}",
                condition=str(condition),
                requested_seed=int(requested_seed),
                root_query_index=int(root_query_index),
                history_length=history_length,
            )
        )
    if len(rows) != expected_groups * collector.CANDIDATE_COUNT:
        raise LiquidTrainingError("source row count differs from complete decisions")
    return manifest, rows


def _load_group_rows(
    path: Path,
    *,
    logical_group: str,
    condition: str,
    requested_seed: int,
    root_query_index: int,
    history_length: int,
) -> list[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as values:
        observed = set(values.files)
        if observed != REQUIRED_ARRAYS:
            raise LiquidTrainingError(
                f"{path} array schema mismatch: "
                f"missing={sorted(REQUIRED_ARRAYS-observed)}, "
                f"extra={sorted(observed-REQUIRED_ARRAYS)}"
            )
        arrays = {name: np.asarray(values[name]) for name in values.files}
    count = collector.CANDIDATE_COUNT
    horizon = arrays["actions"].shape[1]
    shaped = {
        "state": (count, core.STATE_DIM),
        "actions": (count, horizon, core.ACTION_DIM),
        "action_mask": (count, horizon),
        "object_delta": (count, core.OBJECT_DELTA_DIM),
        "state_history": (count, history_length, core.STATE_DIM),
        "state_history_mask": (count, history_length),
        "state_history_dt": (count, history_length),
        "event_history_id": (count, history_length),
        "planned_action_dt": (count, horizon),
    }
    for name, expected in shaped.items():
        if arrays[name].shape != expected:
            raise LiquidTrainingError(f"{path} {name} must be {expected}")
    scalar = set(arrays) - set(shaped)
    if any(arrays[name].shape != (count,) for name in scalar):
        raise LiquidTrainingError(f"{path} scalar target shape changed")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise LiquidTrainingError(f"{path} contains a non-finite array")
    mask = arrays["state_history_mask"].astype(bool)
    action_mask = arrays["action_mask"].astype(bool)
    if (
        np.any(arrays["state_history_dt"] < 0.0)
        or np.any(arrays["state_history_dt"][~mask] != 0.0)
        or np.any(arrays["planned_action_dt"] < 0.0)
        or not np.array_equal(arrays["planned_action_dt"] > 0.0, action_mask)
        or not np.allclose(
            arrays["planned_action_dt"][action_mask],
            1.0 / collector.SOURCE_EVENT_SAMPLING_HZ,
            atol=1e-7,
            rtol=0.0,
        )
        or np.any(~mask.any(axis=1))
    ):
        raise LiquidTrainingError(f"{path} continuous-time masks are invalid")
    last_valid = mask.sum(axis=1) - 1 + (~mask).sum(axis=1)
    if not np.allclose(
        arrays["state_history"][np.arange(count), last_valid],
        arrays["state"],
        atol=1e-6,
        rtol=0.0,
    ):
        raise LiquidTrainingError(f"{path} history does not terminate at root state")
    history_events = arrays["event_history_id"].astype(np.int64)
    if np.any((history_events[mask] < 0) | (history_events[mask] >= 5)):
        raise LiquidTrainingError(f"{path} history event id is invalid")
    expected_onehot = np.eye(5, dtype=np.float32)[history_events[mask]]
    if not np.array_equal(arrays["state_history"][:, :, 18:23][mask], expected_onehot):
        raise LiquidTrainingError(f"{path} history event onehot is inconsistent")
    if not np.array_equal(arrays["candidate_index"], np.arange(count)):
        raise LiquidTrainingError(f"{path} candidate order changed")
    root_shared = (
        np.array_equal(arrays["state"], np.repeat(arrays["state"][:1], count, 0))
        and np.array_equal(
            arrays["state_history"],
            np.repeat(arrays["state_history"][:1], count, 0),
        )
        and np.array_equal(
            arrays["state_history_dt"],
            np.repeat(arrays["state_history_dt"][:1], count, 0),
        )
    )
    if not root_shared:
        raise LiquidTrainingError(f"{path} candidates do not share one past root")
    rows = []
    for index in range(count):
        row = {name: arrays[name][index] for name in arrays}
        row.update(
            {
                "action_available": np.float32(1.0),
                "action_schema_id": np.int64(0),
                "logical_group": logical_group,
                "requested_seed": np.int64(requested_seed),
                "root_query_index": np.int64(root_query_index),
                "body": SOURCE_BODY,
                "condition": condition,
                "policy": "frozen_native_actor",
                "task": collector.TASK,
            }
        )
        rows.append(row)
    return rows


def label_blind_seed_split(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["logical_group"])].append(row)
    strata: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for group, members in sorted(grouped.items()):
        if len(members) != collector.CANDIDATE_COUNT:
            raise LiquidTrainingError("source-only split found an incomplete decision")
        conditions = {str(row["condition"]) for row in members}
        queries = {int(row["root_query_index"]) for row in members}
        seeds = {int(row["requested_seed"]) for row in members}
        candidates = sorted(int(row["candidate_index"]) for row in members)
        if (
            len(conditions) != 1
            or len(queries) != 1
            or len(seeds) != 1
            or candidates != list(range(collector.CANDIDATE_COUNT))
        ):
            raise LiquidTrainingError("source-only decision identity is invalid")
        strata[(conditions.pop(), queries.pop())].append((seeds.pop(), group))
    if not strata:
        raise LiquidTrainingError("source-only split has no causal stratum")
    validation_groups: set[str] = set()
    stratum_receipts = []
    for (condition, query), identities in sorted(strata.items()):
        ordered = sorted(identities)
        if len(ordered) < 5 or len({seed for seed, _group in ordered}) != len(ordered):
            raise LiquidTrainingError(
                "each condition/query split needs at least five unique seed decisions"
            )
        validation_count = max(2, int(math.ceil(0.2 * len(ordered))))
        selected = ordered[-validation_count:]
        validation_groups.update(group for _seed, group in selected)
        stratum_receipts.append(
            {
                "condition": condition,
                "root_query_index": query,
                "decision_groups": len(ordered),
                "training_groups": len(ordered) - validation_count,
                "validation_groups": validation_count,
                "validation_requested_seeds": [seed for seed, _group in selected],
            }
        )
    train = [
        dict(row)
        for row in rows
        if str(row["logical_group"]) not in validation_groups
    ]
    validation = [
        dict(row)
        for row in rows
        if str(row["logical_group"]) in validation_groups
    ]
    if not train or not validation:
        raise LiquidTrainingError("label-blind source split is empty")
    for split_name, split_rows in (("train", train), ("validation", validation)):
        groups: dict[str, int] = defaultdict(int)
        for row in split_rows:
            groups[str(row["logical_group"])] += 1
        if any(count != collector.CANDIDATE_COUNT for count in groups.values()):
            raise LiquidTrainingError(f"{split_name} split cut a candidate decision")
    receipt = {
        "format": SPLIT_FORMAT,
        "source_body": SOURCE_BODY,
        "method": (
            "within_condition_query_ascending_requested_seed_last_20pct_"
            "minimum_two_validation"
        ),
        "labels_or_outcomes_read_for_assignment": False,
        "strata": stratum_receipts,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "target_body_rows_opened": 0,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return train, validation, receipt


def fit_normalization(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    action_receipt = core.fit_train_action_normalization(
        rows, required_schema_ids=(0,)
    )
    schema = action_receipt["schemas"]["aloha"]
    action_mean = np.asarray(schema["mean"], dtype=np.float32)[None]
    action_std = np.asarray(schema["std"], dtype=np.float32)[None]
    history_values = []
    for row in rows:
        state_history = np.asarray(row["state_history"], dtype=np.float32)
        history_mask = np.asarray(row["state_history_mask"], dtype=bool)
        history_values.append(state_history[history_mask])
    states = np.concatenate(history_values, axis=0)
    state_mean = np.zeros(core.STATE_DIM, dtype=np.float32)
    state_std = np.ones(core.STATE_DIM, dtype=np.float32)
    state_mean[:18] = states[:, :18].mean(axis=0)
    state_std[:18] = np.maximum(states[:, :18].std(axis=0), 1e-4)
    receipt = {
        "format": "etsf_aloha_source_liquid_train_history_normalization_v1",
        "source_split": "train_only",
        "source_body": SOURCE_BODY,
        "history_rows_used": int(len(states)),
        "target_rows_used": 0,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "binary_state_channels_18_26_normalized": False,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return state_mean, state_std, action_mean, action_std, receipt


def _liquid_semantic_parameters(
    model: liquid_model.LiquidEffectAlignedSharedEventHead,
) -> tuple[torch.nn.Parameter, ...]:
    modules = (
        model.semantic,
        model.action,
        model.transition,
        model.liquid_terminal,
        model.terminal_event,
        model.terminal_goal_progress_component_mean,
    )
    parameters = tuple(
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    if not parameters or len({id(value) for value in parameters}) != len(parameters):
        raise LiquidTrainingError("liquid comparative parameter union is invalid")
    return parameters


def _strict_record(
    terminal: Mapping[str, Any], ranking: Mapping[str, Any], step: int
) -> dict[str, Any]:
    strict = terminal["strict_proper"]
    score = float(strict["macro_score"])
    standard_error = float(strict["macro_standard_error"])
    if not math.isfinite(score) or not math.isfinite(standard_error):
        raise LiquidTrainingError("source validation strict proper score is invalid")
    return {
        "step": int(step),
        "strict_proper": strict,
        "candidate_ranking": dict(ranking),
        "strict_score": score,
        "strict_standard_error": standard_error,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise LiquidTrainingError("v14 production training is remote CUDA-only")
    if "4090" not in torch.cuda.get_device_name(0):
        raise LiquidTrainingError("v14 production training requires the RTX 4090")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("training output must be a new path")
    manifest, all_rows = load_source_manifest(
        args.manifest,
        args.manifest_sha256,
        history_length=args.history_length,
        expected_groups=args.expected_groups,
    )
    train_rows, validation_rows, split_receipt = label_blind_seed_split(all_rows)
    state_mean, state_std, action_mean, action_std, normalization = fit_normalization(
        train_rows
    )
    proper_balance, proper_balance_receipt = v13.source_causal_stratum_proper_weights(
        train_rows,
        source_bodies=(SOURCE_BODY,),
        mode=v13.DEFAULT_PROPER_BALANCE_MODE,
    )
    rank_inventory = v13.candidate_rank_supervision_inventory(train_rows)
    if int(rank_inventory["rank_supervision_groups"]) <= 0:
        raise LiquidTrainingError(
            "Aloha source data contain no mixed-success or dense rank supervision"
        )
    proper_bootstrap, proper_bootstrap_receipt = (
        v13.proper_outcome_preserving_group_bootstrap_weights(
            train_rows, members=5, seed=args.split_seed
        )
    )
    rank_bootstrap, rank_bootstrap_receipt = (
        v13.effect_preserving_group_bootstrap_weights(
            train_rows, members=5, seed=args.split_seed
        )
    )
    output.mkdir(parents=True)
    atomic_json(output / "split_receipt.json", split_receipt)
    atomic_json(output / "normalization.json", normalization)
    body_to_id = {SOURCE_BODY: 0}
    train_dataset = core.TransitionDataset(train_rows, body_to_id)
    validation_loader = DataLoader(
        core.TransitionDataset(validation_rows, body_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    device = torch.device("cuda")
    loss_weights = dict(core.DEFAULT_LOSS_WEIGHTS)
    loss_weights["object"] = 0.0
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        group_indices[str(row["logical_group"])].append(index)
    proper_group_weight = {
        group: proper_bootstrap[:, indices[0]].tolist()
        for group, indices in group_indices.items()
    }
    rank_group_weight = {
        group: rank_bootstrap[:, indices[0]].tolist()
        for group, indices in group_indices.items()
    }
    snapshot_root = output / "common_step_snapshots"
    snapshot_root.mkdir()
    eval_records: list[dict[int, dict[str, Any]]] = []
    snapshot_paths: list[dict[int, Path]] = []
    trainer_sha = sha256_file(Path(__file__).resolve())
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = liquid_model.LiquidEffectAlignedSharedEventHead().to(device)
        model.action.set_normalization(
            torch.as_tensor(action_mean, device=device),
            torch.as_tensor(action_std, device=device),
        )
        model.set_state_normalization(
            torch.as_tensor(state_mean, device=device),
            torch.as_tensor(state_std, device=device),
        )
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad],
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        utility_parameters = tuple(model.candidate_rank.parameters())
        utility_ids = {id(value) for value in utility_parameters}
        world_parameters = tuple(
            value
            for value in model.parameters()
            if value.requires_grad and id(value) not in utility_ids
        )
        proper_loader = DataLoader(
            train_dataset,
            batch_sampler=v13.CompleteDecisionBatchSampler(
                train_rows, batch_size=args.batch_size, seed=seed
            ),
            collate_fn=core.collate_rows,
        )
        rank_loader = DataLoader(
            train_dataset,
            batch_sampler=v13.MacroBalancedRankDecisionBatchSampler(
                train_rows,
                batch_size=args.batch_size,
                seed=seed,
                positive_group_weight={
                    group: float(values[member])
                    for group, values in rank_group_weight.items()
                },
                ablation_variant="full",
            ),
            collate_fn=core.collate_rows,
        )
        proper_iterator = iter(proper_loader)
        rank_iterator = iter(rank_loader)
        member_records: dict[int, dict[str, Any]] = {}
        member_paths: dict[int, Path] = {}
        for step in range(1, args.steps + 1):
            try:
                proper_raw = next(proper_iterator)
            except StopIteration:
                proper_iterator = iter(proper_loader)
                proper_raw = next(proper_iterator)
            try:
                rank_raw = next(rank_iterator)
            except StopIteration:
                rank_iterator = iter(rank_loader)
                rank_raw = next(rank_iterator)
            proper_batch = core._move_batch(proper_raw, device)
            rank_batch = core._move_batch(rank_raw, device)
            proper_weight = torch.as_tensor(
                [
                    proper_group_weight[str(group)][member]
                    * proper_balance[str(group)]
                    for group in proper_raw["logical_group"]
                ],
                device=device,
            )
            rank_weight = torch.as_tensor(
                [rank_group_weight[str(group)][member] for group in rank_raw["logical_group"]],
                device=device,
            )
            proper_prediction = model(proper_batch)
            multitask, multitask_pieces = v13._compute_shared_multitask_loss(
                proper_prediction,
                proper_batch,
                sample_weight=proper_weight,
                loss_weights=loss_weights,
            )
            object_loss, object_pieces = v13._robust_object_effect_loss(
                proper_prediction, proper_batch, proper_weight
            )
            terminal_loss, terminal_pieces = v13._terminal_consequence_loss(
                proper_prediction, proper_batch, proper_weight
            )
            proper_world = multitask + object_loss + terminal_loss
            rank_prediction = model(rank_batch)
            rank_loss, rank_pieces = v13._candidate_rank_loss(
                rank_prediction, rank_batch, rank_weight
            )
            semantic_raw, semantic_pieces = v13._semantic_comparative_loss(
                rank_prediction, rank_batch, rank_weight
            )
            semantic_scale = v13._relative_gradient_budget_scale(
                proper_world,
                semantic_raw,
                _liquid_semantic_parameters(model),
            )
            semantic_loss = semantic_scale * semantic_raw
            loss = proper_world + rank_loss + semantic_loss
            if not bool(torch.isfinite(loss)):
                raise LiquidTrainingError("v14 training loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_parameters, 2.0)
            torch.nn.utils.clip_grad_norm_(utility_parameters, 2.0)
            optimizer.step()
            if step % args.eval_every and step != args.steps:
                continue
            terminal = v13.evaluate_terminal_consequences(
                model, validation_loader, device
            )
            ranking = v13.evaluate_candidate_ranking(
                model, validation_loader, device
            )
            record = _strict_record(terminal, ranking, step)
            record["train_objective_last"] = {
                "total": float(loss.detach()),
                "semantic_comparative_scale": float(semantic_scale.detach()),
                **{
                    name: float(value.detach())
                    for parts in (
                        multitask_pieces,
                        object_pieces,
                        terminal_pieces,
                        rank_pieces,
                        semantic_pieces,
                    )
                    for name, value in parts.items()
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                },
            }
            snapshot = snapshot_root / (
                f"member_{member:02d}_seed_{seed}_step_{step:06d}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "member": member,
                    "seed": seed,
                    "step": step,
                    "trainer_file_sha256": trainer_sha,
                },
                snapshot,
            )
            member_records[step] = record
            member_paths[step] = snapshot
            atomic_json(
                output / "training_state.json",
                {
                    "format": FORMAT,
                    "status": "training",
                    "member": member,
                    "member_count": 5,
                    "step": step,
                    "steps_per_member": args.steps,
                    "latest_strict_proper": record["strict_proper"],
                    "latest_candidate_ranking": record["candidate_ranking"],
                },
            )
            model.train()
        eval_records.append(member_records)
        snapshot_paths.append(member_paths)
        del optimizer, model
        torch.cuda.empty_cache()
    common_steps = sorted(set.intersection(*(set(item) for item in eval_records)))
    if not common_steps:
        raise LiquidTrainingError("ensemble has no aligned source-validation step")
    selection_models = [
        liquid_model.LiquidEffectAlignedSharedEventHead().to(device)
        for _ in range(5)
    ]
    ensemble = v13.RiskAdjustedRankEnsemble(selection_models, "full").to(device)
    selection_records = []
    for step in common_steps:
        for member, model in enumerate(selection_models):
            snapshot = torch.load(
                snapshot_paths[member][step], map_location=device, weights_only=True
            )
            model.load_state_dict(snapshot["model"], strict=True)
            model.eval()
        ensemble_ranking = v13.evaluate_candidate_ranking(
            ensemble, validation_loader, device
        )
        strict = [eval_records[member][step]["strict_proper"] for member in range(5)]
        combined = v13.combine_primary_and_supplement_strict_proper(
            strict, [None] * 5
        )
        selection_records.append(
            {
                "step": step,
                "selection_key": list(
                    v13.candidate_checkpoint_selection_key(
                        ensemble_ranking,
                        combined["mean_member_strict_proper_score"],
                        step,
                    )
                ),
                "ensemble_candidate_ranking": ensemble_ranking,
                **combined,
            }
        )
    selected, selection_audit = v13.select_calibration_guarded_checkpoint(
        selection_records
    )
    selected_step = int(selected["step"])
    contract = liquid_model.checkpoint_contract(args.history_length)
    members = []
    for member, seed in enumerate(args.ensemble_seeds):
        snapshot = torch.load(
            snapshot_paths[member][selected_step], map_location="cpu", weights_only=True
        )
        checkpoint_path = output / f"member_{member:02d}_seed_{seed}_best.pt"
        torch.save(
            {
                "format": FORMAT,
                "model_family": liquid_model.MODEL_FAMILY,
                "model": snapshot["model"],
                "config": dataclasses.asdict(selection_models[member].config),
                "member": member,
                "seed": seed,
                "step": selected_step,
                "source_body": SOURCE_BODY,
                "sealed_target_bodies": list(liquid_model.TARGET_BODIES),
                "liquid_contract": contract,
                "parameter_inventory": liquid_model.parameter_inventory(
                    selection_models[member]
                ),
                "normalization": normalization,
                "split_receipt": split_receipt,
                "manifest_sha256": args.manifest_sha256,
                "target_rows_used": 0,
                "source_validation": eval_records[member][selected_step],
            },
            checkpoint_path,
        )
        members.append(
            {
                "member": member,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_validation": eval_records[member][selected_step],
            }
        )
    for paths in snapshot_paths:
        for path in paths.values():
            path.unlink()
    snapshot_root.rmdir()
    summary = {
        "format": FORMAT,
        "status": "source_only_training_complete_targets_still_sealed",
        "model_family": liquid_model.MODEL_FAMILY,
        "source_body": SOURCE_BODY,
        "sealed_target_bodies": list(liquid_model.TARGET_BODIES),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": args.manifest_sha256,
        "source_groups": args.expected_groups,
        "source_rows": len(all_rows),
        "target_rows_opened": 0,
        "liquid_contract": contract,
        "parameter_inventory": liquid_model.parameter_inventory(selection_models[0]),
        "split_receipt": split_receipt,
        "normalization": normalization,
        "proper_balance": proper_balance_receipt,
        "rank_inventory": rank_inventory,
        "proper_bootstrap": proper_bootstrap_receipt,
        "rank_bootstrap": rank_bootstrap_receipt,
        "selected_step": selected_step,
        "selection_audit": selection_audit,
        "selected_ensemble_candidate_ranking": selected[
            "ensemble_candidate_ranking"
        ],
        "members": members,
        "next_required_stage": (
            "paired_closed_loop_actor_vs_liquid_best_of_n_on_each_sealed_target_body"
        ),
    }
    atomic_json(output / "training_summary.json", summary)
    atomic_json(
        output / "training_state.json",
        {
            "format": FORMAT,
            "status": "complete",
            "selected_step": selected_step,
            "training_summary": str(output / "training_summary.json"),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--expected-groups", type=int, default=400)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--ensemble-seeds",
        nargs=5,
        type=int,
        default=[20260901, 20260902, 20260903, 20260904, 20260905],
    )
    args = parser.parse_args()
    if (
        args.history_length < 2
        or args.expected_groups <= 0
        or args.steps <= 0
        or args.eval_every <= 0
        or args.batch_size < collector.CANDIDATE_COUNT
        or len(set(args.ensemble_seeds)) != 5
    ):
        parser.error("invalid v14 source-only training dimensions/budget")
    return args


def main() -> None:
    print("TRAINING=" + json.dumps(train(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FORMAT",
    "HISTORY_ARRAYS",
    "LiquidTrainingError",
    "REQUIRED_ARRAYS",
    "fit_normalization",
    "label_blind_seed_split",
    "load_source_manifest",
    "train",
]
