#!/usr/bin/env python3
"""Retrain an expanded ETSF core on its frozen source split only.

``prepare_etsf_transfer_source_core.py`` can add a data-blind reserved policy or
embodiment embedding row.  The resulting checkpoint is intentionally not a
transfer-ready core: the expanded vocabulary has never existed during source
training.  This program performs the required source-only continuation while
keeping the reserved row bit-exact and refusing every batch that contains the
reserved id.

The program never discovers data recursively.  It opens only the explicitly
bound source rollout manifest, its train/validation episode files, the sealed
test files for raw hashing through the existing cache contract, and the frozen
event specification.  No target rollout, target observation, or target label is
accepted as an input.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from prepare_etsf_transfer_source_core import (
    AXIS_CONFIG,
    FORMAT as EXPANSION_FORMAT,
    RETRAIN_FORMAT,
    file_sha256,
    verify_expansion,
)
from train_openvla_etsf_event_world_model import (
    RELATIVE_TRANSITIONS,
    TransitionDataset,
    atomic_json,
    atomic_torch_save,
    class_weights,
    collate_transitions,
    compute_loss,
    evaluate,
    load_or_build_cache,
    move_batch,
    read_rollout_descriptors,
    read_split_manifest,
    transition_indices,
)


FORBIDDEN_TARGET_TOKENS = (
    "fresh50",
    "fresh_confirmation",
    "fresh-confirmation",
    "transfer_adaptation",
    "transfer_validation",
    "transfer_confirmation",
)


def _reject_target_path(path: Path, name: str) -> None:
    lowered = str(path.expanduser().resolve()).lower()
    token = next((value for value in FORBIDDEN_TARGET_TOKENS if value in lowered), None)
    if token is not None:
        raise ValueError(f"{name} references forbidden target data token {token!r}")


def load_expanded_source_core(
    expanded_path: Path,
    *,
    source_manifest: Path,
    source_split: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify a data-blind expansion against its exact source inputs."""

    expanded_path = expanded_path.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    source_split = source_split.expanduser().resolve()
    for path, name in (
        (expanded_path, "expanded checkpoint"),
        (source_manifest, "source manifest"),
        (source_split, "source split"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        _reject_target_path(path, name)
    numpy_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(numpy_globals):
        payload = torch.load(expanded_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("expanded checkpoint must contain a mapping")
    payload = dict(payload)
    lineage = payload.get("transfer_source_core_expansion")
    if not isinstance(lineage, Mapping) or lineage.get("format") != EXPANSION_FORMAT:
        raise ValueError("checkpoint is not an ETSF source-core expansion")
    if payload.get("reserved_source_retraining") is not None:
        raise ValueError("expanded checkpoint already contains a retraining proof")
    expected_paths = {
        "source_manifest_path": source_manifest,
        "source_split_path": source_split,
    }
    for path_key, expected in expected_paths.items():
        frozen = Path(str(lineage.get(path_key, ""))).expanduser().resolve()
        sha_key = path_key.removesuffix("_path") + "_sha256"
        if frozen != expected or lineage.get(sha_key) != file_sha256(expected):
            raise ValueError(f"expanded checkpoint changed its frozen {path_key}")
    parent = Path(str(lineage.get("parent_path", ""))).expanduser().resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)
    _reject_target_path(parent, "parent checkpoint")
    expansion_audit = verify_expansion(parent, expanded_path)
    return payload, expansion_audit


def validate_source_contract(
    payload: Mapping[str, Any],
    cache: Mapping[str, Any],
    splits: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Prove that cache ids contain source identities and never the reservation."""

    config = payload.get("config")
    contract = payload.get("contract")
    lineage = payload.get("transfer_source_core_expansion")
    state = payload.get("model")
    if not all(isinstance(value, Mapping) for value in (config, contract, lineage, state)):
        raise ValueError("expanded checkpoint lacks config/contract/lineage/model")
    axis = str(lineage["axis"])
    if axis not in AXIS_CONFIG:
        raise ValueError("expanded checkpoint has an invalid reserved axis")
    spec = AXIS_CONFIG[axis]
    target_row = int(lineage["target_row"])
    reservation_name = str(lineage["reservation_name"])
    embedding_name = str(lineage["embedding_parameter"])
    mapping_name = str(spec["mapping"])
    expanded_mapping = {
        str(key): int(value) for key, value in contract[mapping_name].items()
    }
    if expanded_mapping.get(reservation_name) != target_row:
        raise ValueError("reserved identity does not map to the frozen target row")
    source_axis_mapping = dict(expanded_mapping)
    source_axis_mapping.pop(reservation_name)
    cache_axis_mapping = {
        str(key): int(value) for key, value in cache[mapping_name].items()
    }
    if source_axis_mapping != cache_axis_mapping:
        raise ValueError("source cache identity registry differs from expanded core")
    other_mapping_name = "body_to_id" if mapping_name == "policy_to_id" else "policy_to_id"
    other_contract_mapping = {
        str(key): int(value) for key, value in contract[other_mapping_name].items()
    }
    other_cache_mapping = {
        str(key): int(value) for key, value in cache[other_mapping_name].items()
    }
    if other_contract_mapping != other_cache_mapping:
        raise ValueError("non-reserved source identity registry differs from cache")
    expected_dimensions = {
        "state_input_dim": int(cache["hidden_dim"]),
        "action_dim": int(cache["action_dim"]),
        "proprio_dim": int(cache["proprio_dim"]),
        "object_delta_dim": int(cache["object_delta_dim"]),
    }
    for name, expected in expected_dimensions.items():
        if int(config[name]) != expected:
            raise ValueError(f"expanded config {name} differs from source cache")
    if list(config["event_names"]) != list(cache["events"]):
        raise ValueError("expanded event vocabulary differs from source cache")
    if not bool(config.get("structured_events", False)):
        raise ValueError("reserved-source retraining requires the structured event core")
    if contract.get("source_manifest_sha256") != cache.get("source_manifest_sha256"):
        raise ValueError("source manifest digest differs from checkpoint contract")
    for split_name, contract_name in (
        ("train", "train_seeds"),
        ("validation", "validation_seeds"),
        ("test", "sealed_test_seeds"),
    ):
        if sorted(int(value) for value in splits.get(split_name, [])) != sorted(
            int(value) for value in contract.get(contract_name, [])
        ):
            raise ValueError(f"frozen source {split_name} seeds differ from checkpoint")
    id_array_name = "policy_id" if axis == "policy" else "body_id"
    ids = np.asarray(cache["arrays"][id_array_name], dtype=np.int64)
    if bool(np.any(ids == target_row)):
        raise ValueError("reserved target row is present in the source transition cache")
    if not set(int(value) for value in np.unique(ids)).issubset(
        set(cache_axis_mapping.values())
    ):
        raise ValueError("source transition cache contains an unknown identity id")
    embedding = state.get(embedding_name)
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ValueError("reserved embedding parameter is missing or malformed")
    if target_row < 0 or target_row >= embedding.shape[0]:
        raise ValueError("reserved target row is outside the embedding")
    return {
        "axis": axis,
        "target_row": target_row,
        "reservation_name": reservation_name,
        "embedding_parameter": embedding_name,
        "batch_id_field": id_array_name,
        "source_identity_count": len(cache_axis_mapping),
        "source_transition_count": int(len(ids)),
    }


def assert_reserved_row_absent(batch: Mapping[str, torch.Tensor], contract: Mapping[str, Any]) -> None:
    values = batch[str(contract["batch_id_field"])]
    target_row = int(contract["target_row"])
    if bool(torch.any(values == target_row)):
        raise RuntimeError("reserved target row entered a source training batch")


@torch.no_grad()
def restore_reserved_row(
    model: nn.Module,
    *,
    parameter_name: str,
    target_row: int,
    reference: torch.Tensor,
) -> None:
    parameters = dict(model.named_parameters())
    parameter = parameters.get(parameter_name)
    if parameter is None:
        raise RuntimeError(f"model lacks reserved parameter {parameter_name}")
    if parameter.ndim != 2 or parameter.shape[1:] != reference.shape:
        raise RuntimeError("reserved parameter/reference shape differs")
    parameter[target_row].copy_(reference.to(parameter))


def reserved_row_is_bit_exact(
    state: Mapping[str, torch.Tensor],
    *,
    parameter_name: str,
    target_row: int,
    reference: torch.Tensor,
) -> bool:
    return torch.equal(state[parameter_name][target_row].detach().cpu(), reference.cpu())


def source_parameters_changed(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    parameter_name: str,
    target_row: int,
) -> bool:
    if set(before) != set(after):
        raise ValueError("source retraining changed model state keys")
    for name, initial in before.items():
        final = after[name].detach().cpu()
        initial = initial.detach().cpu()
        if initial.shape != final.shape or initial.dtype != final.dtype:
            raise ValueError("source retraining changed tensor shape or dtype")
        if name != parameter_name:
            if not torch.equal(initial, final):
                return True
            continue
        keep = torch.ones(initial.shape[0], dtype=torch.bool)
        keep[target_row] = False
        if not torch.equal(initial[keep], final[keep]):
            return True
    return False


def make_retraining_proof(
    *,
    expanded_path: Path,
    source_manifest: Path,
    source_split: Path,
    training_steps: int,
    training_groups: int,
) -> dict[str, Any]:
    if training_steps <= 0 or training_groups <= 0:
        raise ValueError("source retraining proof requires positive steps and groups")
    return {
        "format": RETRAIN_FORMAT,
        "status": "complete_source_only",
        "input_expanded_checkpoint_sha256": file_sha256(expanded_path),
        "source_manifest_path": str(source_manifest.expanduser().resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_split_path": str(source_split.expanduser().resolve()),
        "source_split_sha256": file_sha256(source_split),
        "source_training_steps": int(training_steps),
        "source_training_groups": int(training_groups),
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_row_used_in_source_batches": False,
        "shared_core_retrained": True,
    }


def validation_selection_score(
    metrics: Mapping[str, Any], *, duration_scale: float, object_scale: float
) -> float:
    duration_mae = metrics.get("duration_observed_mae_steps")
    next_reached = metrics.get("next_reached_event_observed_macro_f1")
    if duration_mae is None or next_reached is None:
        raise RuntimeError("source validation lacks observed event/duration targets")
    score = (
        float(metrics["reach_brier"])
        + float(metrics["success_brier"])
        + (1.0 - float(metrics["event_macro_f1"]))
        + (1.0 - float(metrics["future_semantic_cosine"]))
        + 0.10 * float(duration_mae) / duration_scale
        + 0.10 * float(metrics["object_delta_mae"]) / object_scale
        + 0.50 * (1.0 - float(metrics["relative_transition_macro_f1"]))
        + 0.25 * (1.0 - float(metrics["predicate_macro_f1"]))
        + 0.10 * (1.0 - float(next_reached))
    )
    if not math.isfinite(score):
        raise RuntimeError("source validation selection score is non-finite")
    return score


def _checkpoint_payload(
    *,
    original: Mapping[str, Any],
    model: nn.Module,
    proof: Mapping[str, Any],
    step: int,
    validation: Mapping[str, Any],
    score: float,
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in original.items()
        if key not in {"model", "optimizer", "scaler", "reserved_source_retraining"}
    }
    result.update(
        {
            "model": model.state_dict(),
            "step": int(step),
            "best_step": int(step),
            "best_score": float(score),
            "validation": dict(validation),
            "reserved_source_retraining": dict(proof),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default="bf16")
    parser.add_argument("--min-relative-class-support", type=int, default=5)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=1.0)
    parser.add_argument("--destination-weight", type=float, default=0.25)
    parser.add_argument("--predicate-weight", type=float, default=0.5)
    parser.add_argument("--reach-weight", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.5)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=0.25)
    parser.add_argument("--object-weight", type=float, default=0.5)
    parser.add_argument("--latent-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.batch_size <= 0:
        raise ValueError("steps/eval-every/batch-size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    amp_enabled = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    source_manifest = (args.data / "manifest.json").expanduser().resolve()
    source_split = args.source_split.expanduser().resolve()
    expanded_path = args.expanded.expanduser().resolve()
    payload, expansion_audit = load_expanded_source_core(
        expanded_path,
        source_manifest=source_manifest,
        source_split=source_split,
    )
    manifest, descriptors = read_rollout_descriptors(args.data)
    splits = read_split_manifest(source_split, descriptors)
    by_seed = {descriptor.seed: descriptor for descriptor in descriptors}
    loaded = [
        by_seed[seed]
        for name in ("train", "validation")
        for seed in splits[name]
    ]
    sealed = [by_seed[seed] for seed in splits.get("test", [])]
    config = EventWorldModelConfig.from_dict(payload["config"])
    cache_path = args.cache or (args.output / "source_query_transitions.pt")
    cache = load_or_build_cache(
        args.data,
        cache_path,
        config.event_names,
        payload["contract"]["object_names"],
        args.rebuild_cache,
        manifest=manifest,
        episode_descriptors=loaded,
        sealed_test_descriptors=sealed,
        split_seeds=splits,
        event_spec_path=args.event_spec,
        require_predicates=True,
    )
    source_contract = validate_source_contract(payload, cache, splits)
    arrays = cache["arrays"]
    train_indices = transition_indices(arrays, splits["train"])
    validation_indices = transition_indices(arrays, splits["validation"])
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError("frozen source split lacks train or validation transitions")

    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("expanded checkpoint lacks object normalization")
    object_mean = np.asarray(normalization["object_delta_mean"], dtype=np.float32)
    object_std = np.asarray(normalization["object_delta_std"], dtype=np.float32)
    if object_mean.shape != (config.object_delta_dim,) or object_std.shape != object_mean.shape:
        raise ValueError("expanded object normalization shape differs from config")
    if not np.isfinite(object_mean).all() or not np.isfinite(object_std).all() or np.any(object_std <= 0):
        raise ValueError("expanded object normalization is invalid")
    expected_mean = arrays["object_delta"][train_indices].mean(0).astype(np.float32)
    expected_std = np.maximum(
        arrays["object_delta"][train_indices].std(0), 1e-4
    ).astype(np.float32)
    if not np.array_equal(object_mean, expected_mean) or not np.array_equal(object_std, expected_std):
        raise ValueError("expanded normalization differs from the exact frozen source split")

    train_dataset = TransitionDataset(arrays, train_indices, object_mean, object_std)
    validation_dataset = TransitionDataset(
        arrays, validation_indices, object_mean, object_std
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_transitions,
    }
    generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_kwargs,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(payload["model"], strict=True)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    parameter_name = str(source_contract["embedding_parameter"])
    target_row = int(source_contract["target_row"])
    reserved_reference = initial_state[parameter_name][target_row].clone()
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and args.amp == "fp16"
    )
    (
        reach_pos,
        success_pos,
        event_weight,
        destination_weight,
        relative_weight,
        relative_supported,
        predicate_pos_weight,
    ) = class_weights(
        arrays,
        train_indices,
        config.num_events,
        device,
        structured=True,
        min_relative_support=args.min_relative_class_support,
    )
    loss_weights = {
        "event": args.event_weight,
        "relative": args.relative_weight,
        "destination": args.destination_weight,
        "predicate": args.predicate_weight,
        "reach": args.reach_weight,
        "duration": args.duration_weight,
        "success": args.success_weight,
        "outcome": args.outcome_weight,
        "object": args.object_weight,
        "latent": args.latent_weight,
    }
    observed = arrays["duration_observed"][train_indices] > 0.5
    duration_scale = float(cache["chunk_size"])
    if observed.any():
        duration_scale = float(
            max(arrays["duration"][train_indices][observed].mean(), cache["chunk_size"])
        )
    object_scale = float(max(object_std.mean(), 1e-3))

    atomic_json(
        args.output / "source_retraining_preregistration.json",
        {
            "format": RETRAIN_FORMAT,
            "status": "running_source_only",
            "expanded_checkpoint_path": str(expanded_path),
            "expanded_checkpoint_sha256": file_sha256(expanded_path),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": file_sha256(source_manifest),
            "source_split_path": str(source_split),
            "source_split_sha256": file_sha256(source_split),
            "source_train_groups": len(splits["train"]),
            "source_validation_groups": len(splits["validation"]),
            "sealed_test_groups_not_evaluated": len(splits.get("test", [])),
            "target_data_read": False,
            "target_labels_read": False,
            "reserved_row_used_in_source_batches": False,
            "source_contract": source_contract,
            "expansion_audit": expansion_audit,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "seed": args.seed,
        },
    )

    iterator = iter(train_loader)
    best_score = math.inf
    best_step = 0
    best_validation: Mapping[str, Any] | None = None
    log_path = args.output / "source_retraining_log.jsonl"
    started = time.time()
    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            raw_batch = next(iterator)
        assert_reserved_row_absent(raw_batch, source_contract)
        batch = move_batch(raw_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            loss, pieces = compute_loss(
                model,
                batch,
                loss_weights,
                reach_pos,
                success_pos,
                event_weight,
                destination_weight,
                relative_weight,
                relative_supported,
                predicate_pos_weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite source loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # AdamW decays an entire embedding tensor even when the reserved row has
        # zero gradient.  Restore it after every optimizer step, before any save
        # or validation, so it remains bit-exact to the data-blind initializer.
        restore_reserved_row(
            model,
            parameter_name=parameter_name,
            target_row=target_row,
            reference=reserved_reference,
        )
        validate_now = step % args.eval_every == 0 or step == args.steps
        row: dict[str, Any] = {
            "step": step,
            "wall_seconds": time.time() - started,
            **{f"train_{key}": float(value.detach()) for key, value in pieces.items()},
        }
        if validate_now:
            validation = evaluate(
                model, validation_loader, device, object_mean, object_std
            )
            score = validation_selection_score(
                validation, duration_scale=duration_scale, object_scale=object_scale
            )
            row["validation"] = validation
            row["validation_selection_score"] = score
            final_state = model.state_dict()
            if not reserved_row_is_bit_exact(
                final_state,
                parameter_name=parameter_name,
                target_row=target_row,
                reference=reserved_reference,
            ):
                raise RuntimeError("reserved target row changed during source retraining")
            if not source_parameters_changed(
                initial_state,
                final_state,
                parameter_name=parameter_name,
                target_row=target_row,
            ):
                raise RuntimeError("optimizer steps changed no source parameter")
            if score < best_score:
                best_score = score
                best_step = step
                best_validation = validation
                proof = make_retraining_proof(
                    expanded_path=expanded_path,
                    source_manifest=source_manifest,
                    source_split=source_split,
                    training_steps=step,
                    training_groups=len(splits["train"]),
                )
                atomic_torch_save(
                    args.output / "event_world_model_source_retrained_best.pt",
                    _checkpoint_payload(
                        original=payload,
                        model=model,
                        proof=proof,
                        step=step,
                        validation=validation,
                        score=score,
                    ),
                )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print("RESERVED_SOURCE_CORE=" + json.dumps(row, sort_keys=True), flush=True)

    best_path = args.output / "event_world_model_source_retrained_best.pt"
    if best_step <= 0 or not best_path.is_file() or best_validation is None:
        raise RuntimeError("source retraining produced no verified best checkpoint")
    from prepare_etsf_transfer_source_core import verify_source_retraining

    verification = verify_source_retraining(
        expanded_path,
        best_path,
        source_manifest=source_manifest,
        source_split=source_split,
    )
    summary = {
        "format": RETRAIN_FORMAT,
        "status": "source_core_ready_for_protocol_freeze",
        "device": str(device),
        "amp": args.amp,
        "steps_executed": args.steps,
        "best_step": best_step,
        "best_validation_selection_score": best_score,
        "best_validation": best_validation,
        "checkpoint": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
        "sealed_test_evaluated": False,
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_row_used_in_source_batches": False,
        "verification": verification,
    }
    atomic_json(args.output / "source_retraining_summary.json", summary)
    print("SOURCE_RETRAINING_COMPLETE=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
