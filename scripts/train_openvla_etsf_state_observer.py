#!/usr/bin/env python3
"""Train a monitor-only hidden-state observer for structured ETSF inputs.

The split is resolved from label-free HDF5 identity attributes before any
target dataset is opened.  Only train and validation groups are then loaded;
the sealed-test descriptors are retained as logical keys and are never passed
to ``read_group``.  The produced artifact cannot authorize reranking.  A
separate independent calibration and explicit promotion step is required for
that deployment state.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from openvla_etsf_state_observer import (
    FORMAT,
    SOURCE,
    StateHiddenEventPredicateObserver,
    StateObserverConfig,
    observer_artifact_payload,
    sha256,
)
from train_openvla_etsf_counterfactual import (
    BranchGroup,
    atomic_json,
    atomic_torch_save,
    load_descriptor_groups,
    load_pretrained,
    make_group_splits,
    read_split_manifest,
    scan_group_descriptors,
)


LABEL_DERIVATION = "derive_atomic_predicates_v1_plus_dynamic_event_ids_v1"


def load_observer_group_splits(
    *,
    descriptors: Sequence[Any],
    splits: Mapping[str, Sequence[str]],
    world_config: Any,
    object_names: Sequence[str],
    body_to_id: Mapping[str, int],
    policy_to_id: Mapping[str, int],
    calibrations: Mapping[str, Mapping[str, Any]],
    event_spec_sha256: str,
) -> tuple[list[BranchGroup], list[BranchGroup]]:
    """Load only train/validation targets after a label-free split.

    Kept as a narrow function so tests can assert that sealed descriptors never
    cross the loader boundary.
    """

    descriptor_map = {descriptor.logical_key: descriptor for descriptor in descriptors}
    train_descriptors = [descriptor_map[key] for key in splits["train"]]
    validation_descriptors = [descriptor_map[key] for key in splits["validation"]]
    loaded = load_descriptor_groups(
        [*train_descriptors, *validation_descriptors],
        world_config,
        object_names,
        body_to_id,
        policy_to_id,
        calibrations=calibrations,
        expected_event_spec_sha256=event_spec_sha256,
    )
    group_map = {group.logical_key: group for group in loaded}
    return (
        [group_map[key] for key in splits["train"]],
        [group_map[key] for key in splits["validation"]],
    )


def observer_examples(
    groups: Sequence[BranchGroup],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return one initial query plus every schema-v5 continuation per group."""

    hidden_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    predicate_parts: list[np.ndarray] = []
    initial_count = 0
    continuation_count = 0
    for group in groups:
        if group.schema_version != 5 or group.continuation is None:
            raise RuntimeError(
                f"observer formal supervision requires schema-v5 continuation: {group.path}"
            )
        if len(group.hidden) < 1:
            raise RuntimeError(f"observer group has no initial hidden: {group.path}")
        if not np.allclose(group.hidden, group.hidden[0], rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f"candidate branches do not share one intervention hidden: {group.path}"
            )
        if not np.all(group.current_event_id == group.current_event_id[0]):
            raise RuntimeError(
                f"candidate branches disagree on current event: {group.path}"
            )
        if not np.allclose(
            group.current_predicates,
            group.current_predicates[0],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"candidate branches disagree on current predicates: {group.path}"
            )
        hidden_parts.append(np.asarray(group.hidden[0:1], dtype=np.float32))
        event_parts.append(np.asarray(group.current_event_id[0:1], dtype=np.int64))
        predicate_parts.append(
            np.asarray(group.current_predicates[0:1], dtype=np.float32)
        )
        initial_count += 1

        continuation = group.continuation
        count = len(continuation["hidden_t"])
        if count:
            hidden_parts.append(
                np.asarray(continuation["hidden_t"], dtype=np.float32)
            )
            event_parts.append(
                np.asarray(continuation["current_event_id"], dtype=np.int64)
            )
            predicate_parts.append(
                np.asarray(continuation["current_predicates"], dtype=np.float32)
            )
            continuation_count += count
    if not hidden_parts:
        raise RuntimeError("observer split has no supervised queries")
    hidden = np.concatenate(hidden_parts)
    event = np.concatenate(event_parts)
    predicates = np.concatenate(predicate_parts)
    if hidden.ndim != 2 or event.shape != (len(hidden),):
        raise RuntimeError("observer examples have invalid hidden/event shapes")
    if predicates.ndim != 2 or len(predicates) != len(hidden):
        raise RuntimeError("observer examples have invalid predicate shapes")
    if not np.isfinite(hidden).all() or not np.isfinite(predicates).all():
        raise RuntimeError("observer supervision contains non-finite values")
    if np.any((predicates < 0.0) | (predicates > 1.0)):
        raise RuntimeError("observer predicate targets must lie in [0,1]")
    return hidden, event, predicates, {
        "initial_queries": initial_count,
        "continuation_queries": continuation_count,
        "total_queries": int(len(hidden)),
    }


def loss_weights(
    events: np.ndarray, predicates: np.ndarray, num_events: int
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = np.bincount(events, minlength=num_events).astype(np.float64)
    event_weight = np.ones(num_events, dtype=np.float32)
    present = counts > 0
    event_weight[present] = len(events) / (present.sum() * counts[present])
    positive = predicates.sum(0)
    negative = len(predicates) - positive
    predicate_pos = np.clip(negative / np.maximum(positive, 1.0), 0.25, 20.0)
    return torch.from_numpy(event_weight), torch.from_numpy(
        predicate_pos.astype(np.float32)
    )


def supervised_loss(
    output: Mapping[str, torch.Tensor],
    event: torch.Tensor,
    predicates: torch.Tensor,
    event_weight: torch.Tensor,
    predicate_pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    event_loss = F.cross_entropy(
        output["event_logits"], event, weight=event_weight
    )
    predicate_loss = F.binary_cross_entropy_with_logits(
        output["predicate_logits"],
        predicates,
        pos_weight=predicate_pos_weight,
    )
    total = event_loss + predicate_loss
    return total, {
        "total": float(total.detach()),
        "event": float(event_loss.detach()),
        "predicate": float(predicate_loss.detach()),
    }


@torch.inference_mode()
def collect_logits(
    model: StateHiddenEventPredicateObserver,
    hidden: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = TensorDataset(torch.from_numpy(hidden))
    events: list[torch.Tensor] = []
    predicates: list[torch.Tensor] = []
    model.eval()
    for (batch,) in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        output = model(batch.to(device))
        events.append(output["event_logits"].cpu())
        predicates.append(output["predicate_logits"].cpu())
    return torch.cat(events), torch.cat(predicates)


def calibrate_monitor_only(
    event_logits: torch.Tensor,
    predicate_logits: torch.Tensor,
    event: np.ndarray,
    predicates: np.ndarray,
    calibration_id: str,
) -> dict[str, Any]:
    """Select validation calibration, while keeping deployment disabled."""

    event_target = torch.from_numpy(event)
    temperatures = torch.logspace(-1, 1, 81)
    losses = torch.stack(
        [F.cross_entropy(event_logits / value, event_target) for value in temperatures]
    )
    temperature = float(temperatures[int(losses.argmin())])
    probability = torch.sigmoid(predicate_logits).numpy()
    thresholds: list[float] = []
    for column in range(predicates.shape[1]):
        target = predicates[:, column] >= 0.5
        best: tuple[float, float] | None = None
        for threshold in np.linspace(0.1, 0.9, 33):
            predicted = probability[:, column] >= threshold
            positive_recall = float((predicted & target).sum()) / max(
                int(target.sum()), 1
            )
            negative_recall = float((~predicted & ~target).sum()) / max(
                int((~target).sum()), 1
            )
            balanced_accuracy = 0.5 * (positive_recall + negative_recall)
            key = (balanced_accuracy, -abs(float(threshold) - 0.5))
            if best is None or key > best:
                best = key
                chosen = float(threshold)
        thresholds.append(chosen)
    return {
        "calibration_id": calibration_id,
        "selection_data": "observer_validation_monitor_only_not_world_model_sealed_test",
        "event_temperature": temperature,
        "predicate_thresholds": thresholds,
        # The training CLI never promotes.  This maximal gate is an additional
        # defense, not the mechanism relied upon for monitor-only status.
        "minimum_joint_confidence": 1.0,
        "method": "validation_event_nll_grid_and_predicate_balanced_accuracy_grid_v1",
    }


def observer_metrics(
    event_logits: torch.Tensor,
    predicate_logits: torch.Tensor,
    event: np.ndarray,
    predicates: np.ndarray,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    event_probability = torch.softmax(
        event_logits / float(calibration["event_temperature"]), -1
    )
    predicate_probability = torch.sigmoid(predicate_logits)
    threshold = predicate_probability.new_tensor(
        calibration["predicate_thresholds"]
    )
    predicted_event = event_probability.argmax(-1).numpy()
    predicted_predicates = (predicate_probability >= threshold).numpy()
    target_predicates = predicates >= 0.5
    true_positive = (predicted_predicates & target_predicates).sum(0)
    false_positive = (predicted_predicates & ~target_predicates).sum(0)
    false_negative = (~predicted_predicates & target_predicates).sum(0)
    f1 = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    return {
        "queries": int(len(event)),
        "event_accuracy": float((predicted_event == event).mean()),
        "event_nll": float(
            F.nll_loss(event_probability.clamp_min(1e-8).log(), torch.from_numpy(event))
        ),
        "predicate_macro_f1": float(f1.mean()),
        "predicate_exact_match": float(
            np.all(predicted_predicates == target_predicates, axis=1).mean()
        ),
        "predicate_f1": [float(value) for value in f1],
    }


def train(
    *,
    train_hidden: np.ndarray,
    train_event: np.ndarray,
    train_predicates: np.ndarray,
    validation_hidden: np.ndarray,
    validation_event: np.ndarray,
    validation_predicates: np.ndarray,
    config: StateObserverConfig,
    contract: Mapping[str, Any],
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    device: torch.device,
) -> tuple[StateHiddenEventPredicateObserver, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    bootstrap_calibration = {
        "calibration_id": "training_unpromoted",
        "selection_data": "observer_validation_monitor_only_not_world_model_sealed_test",
        "event_temperature": 1.0,
        "predicate_thresholds": [0.5] * len(config.predicate_names),
        "minimum_joint_confidence": 1.0,
    }
    deployment = {
        "rerank_enabled": False,
        "promotion_status": "monitor_only_requires_independent_validation",
        "reason": "trainer_validation_cannot_authorize_online_actor_override",
    }
    model = StateHiddenEventPredicateObserver(
        config,
        contract=contract,
        calibration=bootstrap_calibration,
        deployment=deployment,
    ).to(device)
    event_weight, predicate_pos_weight = loss_weights(
        train_event, train_predicates, len(config.event_names)
    )
    event_weight = event_weight.to(device)
    predicate_pos_weight = predicate_pos_weight.to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_hidden),
        torch.from_numpy(train_event),
        torch.from_numpy(train_predicates),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            hidden, event, predicates = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            hidden, event, predicates = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(hidden.to(device))
        loss, pieces = supervised_loss(
            output,
            event.to(device),
            predicates.to(device),
            event_weight,
            predicate_pos_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite state observer loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            validation_event_logits, validation_predicate_logits = collect_logits(
                model, validation_hidden, batch_size, device
            )
            validation_output = {
                "event_logits": validation_event_logits,
                "predicate_logits": validation_predicate_logits,
            }
            validation_loss, validation_pieces = supervised_loss(
                validation_output,
                torch.from_numpy(validation_event),
                torch.from_numpy(validation_predicates),
                event_weight.cpu(),
                predicate_pos_weight.cpu(),
            )
            row = {
                "step": step,
                "train": pieces,
                "validation": validation_pieces,
            }
            history.append(row)
            if float(validation_loss) < best_loss:
                best_loss = float(validation_loss)
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("state observer produced no selected checkpoint")
    model.load_state_dict(best_state)
    validation_event_logits, validation_predicate_logits = collect_logits(
        model, validation_hidden, batch_size, device
    )
    calibration_seed = json.dumps(
        {
            "seed": seed,
            "best_step": best_step,
            "contract": contract,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    calibration_id = "observer_validation_" + hashlib.sha256(
        calibration_seed
    ).hexdigest()
    calibration = calibrate_monitor_only(
        validation_event_logits,
        validation_predicate_logits,
        validation_event,
        validation_predicates,
        calibration_id,
    )
    final = StateHiddenEventPredicateObserver(
        config,
        contract=contract,
        calibration=calibration,
        deployment=deployment,
    ).to(device)
    final.load_state_dict(best_state)
    final.eval()
    metrics = observer_metrics(
        validation_event_logits,
        validation_predicate_logits,
        validation_event,
        validation_predicates,
        calibration,
    )
    return final, {
        "status": "complete",
        "seed": seed,
        "steps_requested": steps,
        "best_step": best_step,
        "best_validation_supervised_loss": best_loss,
        "wall_seconds": time.time() - started,
        "validation_metrics_monitor_only": metrics,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-names", nargs="+")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0:
        raise ValueError("steps, batch-size, and eval-every must be positive")
    if args.output.exists():
        raise FileExistsError(
            f"observer output already exists; refusing overwrite: {args.output}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda:0"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    pretrained, world_config = load_pretrained(args.pretrained.resolve())
    if not world_config.structured_events:
        raise RuntimeError("state observer requires a structured world-model checkpoint")
    world_contract = pretrained.get("contract")
    if not isinstance(world_contract, Mapping):
        raise RuntimeError("pretrained checkpoint lacks a frozen contract")
    event_spec = args.event_spec.resolve()
    if not event_spec.is_file():
        raise FileNotFoundError(event_spec)
    event_digest = sha256(event_spec)
    if str(world_contract.get("event_spec_sha256", "")) != event_digest:
        raise RuntimeError("pretrained checkpoint and observer event-spec differ")
    event_value = json.loads(event_spec.read_text(encoding="utf-8"))
    calibrations = event_value.get("calibration")
    if not isinstance(calibrations, Mapping):
        raise RuntimeError("event spec lacks task calibration")
    body_to_id = world_contract.get("body_to_id")
    policy_to_id = world_contract.get("policy_to_id")
    if not isinstance(body_to_id, Mapping) or not isinstance(policy_to_id, Mapping):
        raise RuntimeError("pretrained checkpoint lacks body/policy registration")
    object_names = args.object_names or world_contract.get("object_names")
    if not isinstance(object_names, Sequence) or isinstance(object_names, (str, bytes)):
        raise RuntimeError("observer object names are absent from CLI/checkpoint")

    # This scan opens identity attrs only.  Split assignment therefore occurs
    # before any train/validation label dataset is loaded.
    descriptors = scan_group_descriptors([path.resolve() for path in args.data])
    if any(descriptor.schema_version != 5 for descriptor in descriptors):
        raise RuntimeError("formal observer training refuses old schemas")
    collected_policies = {descriptor.policy for descriptor in descriptors}
    if collected_policies != {"openvla"}:
        raise RuntimeError(
            "this observer trainer is bound to OpenVLA query hidden; "
            f"found policies={sorted(collected_policies)}"
        )
    splits = (
        read_split_manifest(args.split_manifest.resolve(), descriptors)
        if args.split_manifest
        else make_group_splits(descriptors)
    )
    # Deliberately do not construct/load a sealed-test BranchGroup list.
    train_groups, validation_groups = load_observer_group_splits(
        descriptors=descriptors,
        splits=splits,
        world_config=world_config,
        object_names=list(object_names),
        body_to_id=body_to_id,
        policy_to_id=policy_to_id,
        calibrations=calibrations,
        event_spec_sha256=event_digest,
    )
    loaded = [*train_groups, *validation_groups]
    train_hidden, train_event, train_predicates, train_counts = observer_examples(
        train_groups
    )
    validation_hidden, validation_event, validation_predicates, validation_counts = (
        observer_examples(validation_groups)
    )
    if train_hidden.shape[1] != world_config.state_input_dim:
        raise RuntimeError("collected hidden dimension differs from world-model contract")
    if train_predicates.shape[1] != len(world_config.predicate_names):
        raise RuntimeError("collected predicate dimension differs from world-model contract")
    if np.any(train_event < 0) or np.any(train_event >= len(world_config.event_names)):
        raise RuntimeError("train current-event labels lie outside the frozen vocabulary")
    if np.any(validation_event < 0) or np.any(
        validation_event >= len(world_config.event_names)
    ):
        raise RuntimeError("validation current-event labels lie outside the vocabulary")

    state_contracts = world_contract.get("state_contracts", {})
    if state_contracts and not isinstance(state_contracts, Mapping):
        raise RuntimeError("pretrained state contracts are invalid")
    contract = {
        "source": SOURCE,
        "state_source": "openvla_hidden_at_query",
        "label_derivation": LABEL_DERIVATION,
        "event_names": list(world_config.event_names),
        "predicate_names": list(world_config.predicate_names),
        "event_spec": str(event_spec),
        "event_spec_sha256": event_digest,
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": sha256(args.pretrained.resolve()),
        "policy_to_id": dict(policy_to_id),
        "body_to_id": dict(body_to_id),
        "state_contracts": dict(state_contracts),
        "train_groups": list(splits["train"]),
        "validation_groups": list(splits["validation"]),
        "sealed_test_groups": list(splits["test"]),
        "sealed_test_access": "identity_attrs_only_not_loaded_not_evaluated",
        "split_manifest": (
            str(args.split_manifest.resolve()) if args.split_manifest else None
        ),
        "split_manifest_sha256": (
            sha256(args.split_manifest.resolve()) if args.split_manifest else None
        ),
        "loaded_train_validation_group_files": [
            {
                "logical_key": group.logical_key,
                "path": group.path,
                "sha256": sha256(Path(group.path)),
            }
            for group in loaded
        ],
        "loaded_sealed_test_groups": 0,
    }
    observer_config = StateObserverConfig(
        state_input_dim=world_config.state_input_dim,
        hidden_dim=args.hidden_dim,
        event_names=world_config.event_names,
        predicate_names=world_config.predicate_names,
    )
    observer, training = train(
        train_hidden=train_hidden,
        train_event=train_event,
        train_predicates=train_predicates,
        validation_hidden=validation_hidden,
        validation_event=validation_event,
        validation_predicates=validation_predicates,
        config=observer_config,
        contract=contract,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        device=device,
    )
    training["data"] = {
        "train": train_counts,
        "validation": validation_counts,
        "sealed_test": {
            "logical_groups": len(splits["test"]),
            "loaded_queries": 0,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = (args.output / "state_observer.pt").resolve()
    payload = observer_artifact_payload(observer)
    atomic_torch_save(
        checkpoint_path,
        {
            **payload,
            "model": observer.cpu().state_dict(),
            "training": training,
        },
    )
    digest = sha256(checkpoint_path)
    manifest = {
        **payload,
        "checkpoint": {"path": str(checkpoint_path), "sha256": digest},
        "training": training,
    }
    manifest_path = args.output / "state_observer_manifest.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        args.output / "training_summary.json",
        {
            "format": FORMAT,
            "status": "complete",
            "deployment": payload["deployment"],
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": digest,
            "training": training,
        },
    )
    print(
        "STATE_OBSERVER="
        + json.dumps(
            {
                "status": "complete_monitor_only",
                "manifest": str(manifest_path.resolve()),
                "validation": training["validation_metrics_monitor_only"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
