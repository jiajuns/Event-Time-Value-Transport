#!/usr/bin/env python3
"""Train, calibrate, audit, and freeze causal event observer v1.

The only production data entry point is a content-addressed dataset manifest
created by ``materialize_smolvla_piper_causal_event_observer_dataset_v1``.
Training sees actor-visible causal histories and proprioception only.  The
calibration and validation groups are never used by the optimizer.  Promotion
is fail-closed: a frozen monitor-only bundle is still emitted when any real
validation gate is missing or fails, while a v4 rerank authority is emitted
only when every independent gate passes.

The tiny synthetic mode exposed through the Python API is for deterministic
CPU regression tests.  Its output must not be represented as real evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from smolvla_piper_causal_event_observer_v1 import (
    ADAPTER_CHECKPOINT_FORMAT,
    ADAPTER_MANIFEST_FORMAT,
    CORE_CHECKPOINT_FORMAT,
    EVALUATION400_V4_TARGET,
    EXPECTED_EVENTS,
    EXPECTED_PREDICATES,
    FROZEN_AUTHORITY_MANIFEST_FORMAT,
    MAX_CALIBRATION_ECE,
    MAX_HISTORY_STEPS,
    MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95,
    MIN_EVENT_ACCURACY_LCB95,
    MIN_PREDICATE_F1_LCB95,
    MIN_PROMOTION_GROUPS,
    MIN_PROMOTION_GROUPS_PER_ACTOR,
    PROMOTION_EVIDENCE_FORMAT,
    STATE_DIM,
    ActorVisibleCausalEventObserverV1,
    CausalObserverConfig,
    EmbodimentResidualAdapter,
    canonical_sha256,
    causal_history_contract,
    make_actor_adapter_contract,
    make_calibration,
    make_deployment,
    make_execution_receipt,
    make_input_receipt,
    observer_config_document,
    tensor_bundle_sha256,
    training_supervision_contract,
    validate_calibration,
    validate_promotion_evidence,
)


FORMAT = "etsf_smolvla_piper_causal_event_observer_training_v1"
FREEZE_FORMAT = "etsf_smolvla_piper_causal_event_observer_freeze_v1"
PROMOTION_DECISION_FORMAT = (
    "etsf_smolvla_piper_causal_event_observer_promotion_decision_v1"
)
DATASET_MODULE = "materialize_smolvla_piper_causal_event_observer_dataset_v1"
DATASET_FORMAT = "etsf_smolvla_piper_causal_event_observer_dataset_v1"
DATASET_SPLITS = ("train", "calibration", "validation")
METRICS_FORMAT = "etsf_causal_event_observer_independent_validation_v1"
CALIBRATION_FIT_FORMAT = "etsf_causal_event_observer_group_calibration_fit_v1"
TRAINING_RECEIPT_FORMAT = "etsf_causal_event_observer_training_receipt_v1"

PROPRIO_DIM = 14
ARRAY_FIELDS = {
    "history",
    "history_mask",
    "proprio",
    "event_label",
    "predicate_label",
    "actor_index",
    "current_query_index",
    "query_step",
    "prior_execution_present",
    "prior_executed_control_steps",
    "prior_action_sha256",
    "sample_id",
    "logical_group_id",
    "branch_id",
    "source_file_sha256",
}
FORBIDDEN_ONLINE_NAMES = {
    "object_pose",
    "object_poses",
    "simulator_actor_pose",
    "simulator_state",
    "future_hidden",
    "future_image",
    "terminal_outcome",
    "success_outcome",
}


class ObserverTrainingError(RuntimeError):
    """A dataset, chronology, training, audit, or freeze check failed."""


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 96
    adapter_rank: int = 8
    epochs: int = 30
    batch_size_per_actor: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    event_loss_weight: float = 1.0
    predicate_loss_weight: float = 1.0
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    calibration_grid_size: int = 121
    minimum_calibration_accepts: int = 20
    seed: int = 20260828
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            self.hidden_dim < 16
            or self.adapter_rank < 1
            or self.epochs < 1
            or self.batch_size_per_actor < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.event_loss_weight <= 0.0
            or self.predicate_loss_weight <= 0.0
            or self.bootstrap_samples < 100
            or not 0.5 < self.confidence_level < 1.0
            or self.calibration_grid_size < 11
            or self.minimum_calibration_accepts < 1
            or type(self.seed) is not int
            or self.device not in {"cpu", "cuda"}
        ):
            raise ValueError("observer training configuration is invalid")


@dataclass(frozen=True)
class LoadedDataset:
    manifest_path: Path
    manifest: dict[str, Any]
    splits: dict[str, dict[str, np.ndarray]]
    actor_names: tuple[str, ...]
    actor_records: tuple[dict[str, Any], ...]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(base)
    return {**value, field: canonical_sha256(value)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(value), stream, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _clone_cpu_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in sorted(state.items())
    }


def _split_model_state(
    model: ActorVisibleCausalEventObserverV1,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    core: dict[str, torch.Tensor] = {}
    adapters = {name: {} for name in model.actor_names}
    for name, tensor in model.state_dict().items():
        if name.startswith("actor_adapters."):
            parts = name.split(".", 2)
            actor_index = int(parts[1])
            adapters[model.actor_names[actor_index]][parts[2]] = (
                tensor.detach().cpu().contiguous().clone()
            )
        else:
            core[name] = tensor.detach().cpu().contiguous().clone()
    if not core or any(not state for state in adapters.values()):
        raise ObserverTrainingError("model state did not split into core and adapters")
    return core, adapters


def _strict_manifest_checks(
    manifest: Mapping[str, Any], splits: Mapping[str, Mapping[str, np.ndarray]]
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    if manifest.get("format") != DATASET_FORMAT:
        raise ObserverTrainingError("observer dataset format changed")
    exact_scalars = {
        "event_names": list(EXPECTED_EVENTS),
        "predicate_names": list(EXPECTED_PREDICATES),
        "state_dim": STATE_DIM,
        "history_steps": MAX_HISTORY_STEPS,
        "proprio_dim": PROPRIO_DIM,
        "image_feature_dim": 0,
        "split_unit": "logical_reset_group",
        "split_group_disjoint": True,
        "privileged_label_source_available_to_model_inputs": False,
        "future_query_features_available_to_model_inputs": False,
    }
    for name, expected in exact_scalars.items():
        if manifest.get(name) != expected:
            raise ObserverTrainingError(f"observer dataset {name} changed")
    if manifest.get("history_contract_sha256") != causal_history_contract()[
        "contract_sha256"
    ]:
        raise ObserverTrainingError("dataset history contract differs from observer")
    if set(splits) != set(DATASET_SPLITS):
        raise ObserverTrainingError("dataset split inventory changed")

    raw_registry = manifest.get("actor_registry")
    if not isinstance(raw_registry, list) or not raw_registry:
        raise ObserverTrainingError("dataset actor registry is empty")
    expected_actor_fields = {
        "actor_name", "policy_family", "body", "policy",
        "state_feature_source_sha256", "actor_index",
    }
    records: list[dict[str, Any]] = []
    for expected_index, raw in enumerate(raw_registry):
        if not isinstance(raw, Mapping) or set(raw) != expected_actor_fields:
            raise ObserverTrainingError("dataset actor registry fields changed")
        item = dict(raw)
        if (
            type(item["actor_index"]) is not int
            or item["actor_index"] != expected_index
            or not isinstance(item["actor_name"], str)
            or not item["actor_name"]
            or not isinstance(item["policy_family"], str)
            or not item["policy_family"]
            or not _is_sha(item["state_feature_source_sha256"])
        ):
            raise ObserverTrainingError("dataset actor registry is invalid")
        records.append(item)
    actor_names = tuple(record["actor_name"] for record in records)
    if len(set(actor_names)) != len(actor_names):
        raise ObserverTrainingError("dataset actor names are not unique")

    split_groups: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    for split_name in DATASET_SPLITS:
        arrays = splits[split_name]
        if set(arrays) != ARRAY_FIELDS:
            raise ObserverTrainingError(f"{split_name} array fields changed")
        count = int(arrays["history"].shape[0])
        exact = {
            "history": (np.dtype(np.float32), (count, MAX_HISTORY_STEPS, STATE_DIM)),
            "history_mask": (np.dtype(np.bool_), (count, MAX_HISTORY_STEPS)),
            "proprio": (np.dtype(np.float32), (count, PROPRIO_DIM)),
            "event_label": (np.dtype(np.int64), (count,)),
            "predicate_label": (
                np.dtype(np.float32), (count, len(EXPECTED_PREDICATES))
            ),
            "actor_index": (np.dtype(np.int64), (count,)),
            "current_query_index": (np.dtype(np.int64), (count,)),
            "query_step": (np.dtype(np.int64), (count,)),
            "prior_execution_present": (np.dtype(np.bool_), (count,)),
            "prior_executed_control_steps": (np.dtype(np.int64), (count,)),
        }
        if count < 1:
            raise ObserverTrainingError(f"{split_name} is empty")
        for name, (dtype, shape) in exact.items():
            value = arrays[name]
            if value.dtype != dtype or value.shape != shape:
                raise ObserverTrainingError(
                    f"{split_name}.{name} dtype/shape changed"
                )
        for name in ARRAY_FIELDS - set(exact):
            value = arrays[name]
            if value.ndim != 1 or value.shape != (count,) or value.dtype.kind != "U":
                raise ObserverTrainingError(f"{split_name}.{name} is not Unicode[N]")
        if (
            not np.isfinite(arrays["history"]).all()
            or not np.isfinite(arrays["proprio"]).all()
            or not np.isfinite(arrays["predicate_label"]).all()
            or np.any(arrays["event_label"] < 0)
            or np.any(arrays["event_label"] >= len(EXPECTED_EVENTS))
            or np.any((arrays["predicate_label"] != 0.0) & (arrays["predicate_label"] != 1.0))
            or np.any(arrays["actor_index"] < 0)
            or np.any(arrays["actor_index"] >= len(records))
        ):
            raise ObserverTrainingError(f"{split_name} contains invalid values")
        masks = arrays["history_mask"]
        valid = masks.sum(axis=1)
        expected_valid = np.minimum(arrays["current_query_index"] + 1, MAX_HISTORY_STEPS)
        expected_mask = np.arange(MAX_HISTORY_STEPS)[None, :] < valid[:, None]
        if (
            np.any(valid < 1)
            or not np.array_equal(valid, expected_valid)
            or not np.array_equal(masks, expected_mask)
            or np.any(arrays["history"][~masks] != 0.0)
            or np.any(arrays["query_step"] < 0)
            or np.any(arrays["current_query_index"] < 0)
            or np.any(arrays["prior_executed_control_steps"] < 0)
            or np.any(
                (~arrays["prior_execution_present"])
                & (arrays["prior_executed_control_steps"] != 0)
            )
        ):
            raise ObserverTrainingError(f"{split_name} chronology/padding is invalid")
        for sample, source_sha in zip(
            arrays["sample_id"], arrays["source_file_sha256"], strict=True
        ):
            if not _is_sha(str(sample)) or not _is_sha(str(source_sha)):
                raise ObserverTrainingError(f"{split_name} contains invalid content IDs")
            if str(sample) in sample_ids:
                raise ObserverTrainingError("sample ID leaked across dataset rows")
            sample_ids.add(str(sample))
        prior_actions = arrays["prior_action_sha256"]
        query_zero = arrays["current_query_index"] == 0
        query_later = ~query_zero
        if (
            np.any(arrays["prior_execution_present"][query_zero])
            or np.any(arrays["prior_executed_control_steps"][query_zero] != 0)
            or np.any(prior_actions[query_zero] != "")
            or np.any(~arrays["prior_execution_present"][query_later])
            or np.any(arrays["prior_executed_control_steps"][query_later] <= 0)
            or any(not _is_sha(str(value)) for value in prior_actions[query_later])
        ):
            raise ObserverTrainingError(
                f"{split_name} prior execution chronology/binding is invalid"
            )
        groups = {str(item) for item in arrays["logical_group_id"]}
        if not groups or "" in groups:
            raise ObserverTrainingError(f"{split_name} group IDs are invalid")
        split_groups[split_name] = groups
        for actor_index in range(len(records)):
            if not bool(np.any(arrays["actor_index"] == actor_index)):
                raise ObserverTrainingError(
                    f"{split_name} has no support for actor {actor_names[actor_index]}"
                )
    for left_index, left in enumerate(DATASET_SPLITS):
        for right in DATASET_SPLITS[left_index + 1 :]:
            if split_groups[left] & split_groups[right]:
                raise ObserverTrainingError(f"{left}/{right} group leakage detected")
    return actor_names, tuple(records)


def load_supervision_dataset(manifest_path: Path) -> LoadedDataset:
    """Load through the materializer's validating API, then recheck trainer needs."""

    path = manifest_path.resolve()
    if path.name != "manifest.json" or not path.is_file():
        raise ObserverTrainingError("dataset entry point must be an existing manifest.json")
    module = importlib.import_module(DATASET_MODULE)
    manifest = module.validate_dataset_manifest(path, verify_npz=True)
    splits = {name: module.load_split(path, name) for name in DATASET_SPLITS}
    actor_names, records = _strict_manifest_checks(manifest, splits)
    return LoadedDataset(
        manifest_path=path,
        manifest=dict(manifest),
        splits={name: dict(value) for name, value in splits.items()},
        actor_names=actor_names,
        actor_records=records,
    )


def _make_training_contract(dataset: LoadedDataset) -> dict[str, Any]:
    event_spec = dataset.manifest.get("event_spec")
    if not isinstance(event_spec, Mapping) or not _is_sha(event_spec.get("file_sha256")):
        raise ObserverTrainingError("dataset event-spec binding is invalid")
    actor_registry = [
        {
            "actor_name": item["actor_name"],
            "policy_family": item["policy_family"],
            "state_feature_source_sha256": item["state_feature_source_sha256"],
        }
        for item in dataset.actor_records
    ]
    return training_supervision_contract(
        event_spec_sha256=str(event_spec["file_sha256"]),
        dataset_manifest_sha256=str(dataset.manifest["manifest_sha256"]),
        actor_registry=actor_registry,
    )


def _initial_model(
    dataset: LoadedDataset, config: TrainingConfig,
) -> ActorVisibleCausalEventObserverV1:
    torch.manual_seed(config.seed)
    observer_config = CausalObserverConfig(
        proprio_dim=PROPRIO_DIM,
        image_feature_dim=0,
        hidden_dim=config.hidden_dim,
        adapter_rank=config.adapter_rank,
        dropout=0.0,
    )
    training_contract = _make_training_contract(dataset)
    observer_config_sha = observer_config_document(observer_config)["config_sha256"]
    observer_source = Path(inspect.getsourcefile(ActorVisibleCausalEventObserverV1) or "")
    if not observer_source.is_file():
        raise ObserverTrainingError("observer implementation source file is unavailable")
    observer_core_file_sha = file_sha256(observer_source)
    provisional_checkpoint_sha = canonical_sha256(
        {"role": "unfrozen_training_checkpoint", "seed": config.seed}
    )
    provisional_adapter_set_sha = canonical_sha256(
        {"role": "unfrozen_training_adapter_set", "actors": list(dataset.actor_names)}
    )
    provisional_adapter_checkpoint_set_sha = canonical_sha256(
        {
            "role": "unfrozen_training_adapter_checkpoint_set",
            "actors": list(dataset.actor_names),
        }
    )
    states: dict[str, dict[str, torch.Tensor]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for record in dataset.actor_records:
        actor_name = str(record["actor_name"])
        adapter = EmbodimentResidualAdapter(
            observer_config.hidden_dim, observer_config.adapter_rank
        )
        state = _clone_cpu_state(adapter.state_dict())
        states[actor_name] = state
        contracts[actor_name] = make_actor_adapter_contract(
            actor_name=actor_name,
            policy_family=str(record["policy_family"]),
            state_feature_source_sha256=str(record["state_feature_source_sha256"]),
            observer_core_file_sha256=observer_core_file_sha,
            training_contract_sha256=training_contract["contract_sha256"],
            image_feature_extractor_file_sha256=None,
            config=observer_config,
            adapter_state=state,
        )
    calibration_split_sha = dataset.manifest["splits"]["calibration"]["logical_sha256"]
    calibration = make_calibration(
        event_spec_sha256=training_contract["event_spec_sha256"],
        independent_calibration_split_sha256=calibration_split_sha,
        minimum_joint_confidence=1.0,
        reject_all=True,
    )
    return ActorVisibleCausalEventObserverV1(
        observer_config,
        training_contract=training_contract,
        observer_core_file_sha256=observer_core_file_sha,
        observer_checkpoint_file_sha256=provisional_checkpoint_sha,
        observer_config_sha256=observer_config_sha,
        actor_adapter_set_sha256=provisional_adapter_set_sha,
        actor_adapter_checkpoint_set_sha256=(
            provisional_adapter_checkpoint_set_sha
        ),
        adapter_contracts=contracts,
        adapter_states=states,
        calibration=calibration,
        deployment=make_deployment(promotion_enabled=False),
    )


def _receipts(
    dataset: LoadedDataset,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[list[str], list[dict[str, Any]]]:
    names: list[str] = []
    receipts: list[dict[str, Any]] = []
    for source_index in indices.tolist():
        actor_index = int(arrays["actor_index"][source_index])
        record = dataset.actor_records[actor_index]
        actor_name = str(record["actor_name"])
        mask = torch.from_numpy(arrays["history_mask"][source_index])
        names.append(actor_name)
        current_query_index = int(arrays["current_query_index"][source_index])
        execution_receipt = None
        if bool(arrays["prior_execution_present"][source_index]):
            execution_receipt = make_execution_receipt(
                action_sha256=str(arrays["prior_action_sha256"][source_index]),
                executed_control_steps=int(
                    arrays["prior_executed_control_steps"][source_index]
                ),
                last_completed_query_index=current_query_index - 1,
                current_query_index=current_query_index,
            )
        receipts.append(
            make_input_receipt(
                history=torch.from_numpy(arrays["history"][source_index]),
                history_mask=mask,
                proprio=torch.from_numpy(arrays["proprio"][source_index]),
                actor_name=actor_name,
                policy_family=str(record["policy_family"]),
                state_feature_source_sha256=str(
                    record["state_feature_source_sha256"]
                ),
                current_query_index=current_query_index,
                valid_history_steps=int(mask.sum()),
                execution_receipt=execution_receipt,
            )
        )
    return names, receipts


def _forward_indices(
    model: ActorVisibleCausalEventObserverV1,
    dataset: LoadedDataset,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    actors, receipts = _receipts(dataset, arrays, indices)
    return model(
        torch.from_numpy(arrays["history"][indices]).to(device),
        torch.from_numpy(arrays["history_mask"][indices]).to(device),
        torch.from_numpy(arrays["proprio"][indices]).to(device),
        actor_names=actors,
        receipts=receipts,
    )


def _balanced_epoch_indices(
    actor_index: np.ndarray, actor_count: int, generator: np.random.Generator
) -> np.ndarray:
    by_actor = [np.flatnonzero(actor_index == index) for index in range(actor_count)]
    if any(len(indices) == 0 for indices in by_actor):
        raise ObserverTrainingError("balanced sampler encountered unsupported actor")
    target = max(len(indices) for indices in by_actor)
    sampled = np.stack([
        generator.choice(indices, size=target, replace=len(indices) < target)
        for indices in by_actor
    ])
    generator.shuffle(sampled, axis=1)
    return sampled.T.reshape(-1)


def _balanced_loss(
    output: Mapping[str, torch.Tensor],
    event_target: torch.Tensor,
    predicate_target: torch.Tensor,
    actor_target: torch.Tensor,
    actor_count: int,
    config: TrainingConfig,
) -> tuple[torch.Tensor, float, float]:
    event_rows = F.cross_entropy(output["event_logits"], event_target, reduction="none")
    predicate_rows = F.binary_cross_entropy_with_logits(
        output["predicate_logits"], predicate_target, reduction="none"
    ).mean(dim=1)
    event_actor: list[torch.Tensor] = []
    predicate_actor: list[torch.Tensor] = []
    for actor in range(actor_count):
        rows = actor_target == actor
        if bool(rows.any()):
            event_actor.append(event_rows[rows].mean())
            predicate_actor.append(predicate_rows[rows].mean())
    if len(event_actor) != actor_count:
        raise ObserverTrainingError("batch omitted an actor; balanced loss cannot proceed")
    event_loss = torch.stack(event_actor).mean()
    predicate_loss = torch.stack(predicate_actor).mean()
    total = (
        config.event_loss_weight * event_loss
        + config.predicate_loss_weight * predicate_loss
    )
    return total, float(event_loss.detach()), float(predicate_loss.detach())


def train_model(
    dataset: LoadedDataset, config: TrainingConfig
) -> tuple[ActorVisibleCausalEventObserverV1, dict[str, Any]]:
    """Optimize on train split only with equal actor contribution."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ObserverTrainingError("CUDA was requested but is unavailable")
    device = torch.device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
    model = _initial_model(dataset, config).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    arrays = dataset.splits["train"]
    generator = np.random.default_rng(config.seed)
    epoch_records: list[dict[str, Any]] = []
    actor_count = len(dataset.actor_names)
    batch_size = config.batch_size_per_actor * actor_count
    for epoch in range(config.epochs):
        order = _balanced_epoch_indices(arrays["actor_index"], actor_count, generator)
        losses: list[float] = []
        event_losses: list[float] = []
        predicate_losses: list[float] = []
        # Interleaving makes every full batch contain the same number per actor.
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if len(indices) < actor_count:
                continue
            output = _forward_indices(model, dataset, arrays, indices, device)
            event_target = torch.from_numpy(arrays["event_label"][indices]).to(device)
            predicate_target = torch.from_numpy(
                arrays["predicate_label"][indices]
            ).to(device)
            actor_target = torch.from_numpy(arrays["actor_index"][indices]).to(device)
            loss, event_loss, predicate_loss = _balanced_loss(
                output, event_target, predicate_target, actor_target, actor_count, config
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            event_losses.append(event_loss)
            predicate_losses.append(predicate_loss)
        if not losses:
            raise ObserverTrainingError("training produced no optimizer step")
        epoch_records.append(
            {
                "epoch": epoch,
                "balanced_rows": int(len(order)),
                "loss": float(np.mean(losses)),
                "event_ce": float(np.mean(event_losses)),
                "predicate_bce": float(np.mean(predicate_losses)),
            }
        )
    model.eval()
    receipt = _signed(
        {
            "format": TRAINING_RECEIPT_FORMAT,
            "status": "optimizer_complete_train_split_only",
            "seed": config.seed,
            "device": config.device,
            "training_config": asdict(config),
            "actor_balanced_sampling": True,
            "actor_balanced_loss": True,
            "event_objective": "equal_actor_mean_cross_entropy",
            "predicate_objective": "equal_actor_mean_binary_cross_entropy",
            "train_split_logical_sha256": dataset.manifest["splits"]["train"][
                "logical_sha256"
            ],
            "calibration_or_validation_used_by_optimizer": False,
            "epochs": epoch_records,
        },
        "training_receipt_sha256",
    )
    return model, receipt


@torch.inference_mode()
def _predict_split(
    model: ActorVisibleCausalEventObserverV1,
    dataset: LoadedDataset,
    split_name: str,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    arrays = dataset.splits[split_name]
    device = next(model.parameters()).device
    event: list[np.ndarray] = []
    predicate: list[np.ndarray] = []
    for start in range(0, len(arrays["event_label"]), batch_size):
        indices = np.arange(
            start, min(start + batch_size, len(arrays["event_label"])), dtype=np.int64
        )
        output = _forward_indices(model, dataset, arrays, indices, device)
        event.append(output["event_logits"].detach().cpu().numpy().astype(np.float64))
        predicate.append(
            output["predicate_logits"].detach().cpu().numpy().astype(np.float64)
        )
    return {
        "event_logits": np.concatenate(event),
        "predicate_logits": np.concatenate(predicate),
    }


def _equal_group_weights(group_ids: np.ndarray) -> np.ndarray:
    groups, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    if len(groups) < 1:
        raise ObserverTrainingError("calibration has no group")
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def _temperature_grid(size: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(0.05), math.log(20.0), size))


def _fit_event_temperature(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray, grid_size: int
) -> float:
    best = (float("inf"), 1.0)
    for temperature in _temperature_grid(grid_size):
        probability = _softmax(logits / temperature)
        loss = -float(np.sum(weights * np.log(np.maximum(
            probability[np.arange(len(labels)), labels], 1.0e-12
        ))))
        candidate = (loss, abs(math.log(temperature)), float(temperature))
        if candidate < (best[0], abs(math.log(best[1])), best[1]):
            best = (loss, float(temperature))
    return best[1]


def _fit_predicate_temperature(
    logits: np.ndarray, labels: np.ndarray, weights: np.ndarray, grid_size: int
) -> float:
    best = (float("inf"), 1.0)
    for temperature in _temperature_grid(grid_size):
        probabilities = _sigmoid(logits / temperature)
        loss = -float(np.sum(weights * (
            labels * np.log(np.maximum(probabilities, 1.0e-12))
            + (1.0 - labels) * np.log(np.maximum(1.0 - probabilities, 1.0e-12))
        )))
        candidate = (loss, abs(math.log(temperature)), float(temperature))
        if candidate < (best[0], abs(math.log(best[1])), best[1]):
            best = (loss, float(temperature))
    return best[1]


def _weighted_f1(
    labels: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> float:
    true_positive = float(np.sum(weights * ((labels == 1) & (predictions == 1))))
    false_positive = float(np.sum(weights * ((labels == 0) & (predictions == 1))))
    false_negative = float(np.sum(weights * ((labels == 1) & (predictions == 0))))
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0.0 else 2.0 * true_positive / denominator


def _fit_threshold(
    probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    candidates = np.unique(np.concatenate((np.array([0.01, 0.5, 0.99]), probabilities)))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        score = _weighted_f1(labels, probabilities >= threshold, weights)
        key = (score, -abs(float(threshold) - 0.5), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total < 1:
        return 0.0, 1.0
    # NormalDist is avoided to keep this module Python-version portable; 95%
    # is the frozen production confidence and the configurable value is tested.
    if abs(confidence - 0.95) > 1.0e-12:
        raise ObserverTrainingError("only the preregistered 95% Wilson interval is supported")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _select_joint_confidence(
    event_probability: np.ndarray,
    predicate_probability: np.ndarray,
    event_labels: np.ndarray,
    predicate_labels: np.ndarray,
    predicate_thresholds: np.ndarray,
    group_ids: np.ndarray,
    minimum_accepts: int,
    confidence: float,
) -> tuple[float, dict[str, Any]]:
    event_confidence = event_probability.max(axis=1)
    predicate_confidence = np.maximum(
        predicate_probability, 1.0 - predicate_probability
    ).min(axis=1)
    joint = np.minimum(event_confidence, predicate_confidence)
    correct = (
        (event_probability.argmax(axis=1) == event_labels)
        & np.all(
            (predicate_probability >= predicate_thresholds[None, :])
            == predicate_labels.astype(bool),
            axis=1,
        )
    )
    candidates = sorted(
        {float(value) for value in joint if 0.0 <= value <= 1.0} | {1.0}
    )
    unique_groups = np.unique(group_ids)
    selected: tuple[int, float, int, float] | None = None
    for threshold in candidates:
        accepted = joint >= threshold
        accepted_groups = [
            group for group in unique_groups
            if bool(np.any(accepted & (group_ids == group)))
        ]
        errors = sum(
            bool(np.any(accepted & ~correct & (group_ids == group)))
            for group in accepted_groups
        )
        count = len(accepted_groups)
        _, error_ucb = _wilson_interval(int(errors), count, confidence)
        if count >= minimum_accepts and error_ucb <= MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95:
            candidate = (count, -threshold, -int(errors), -error_ucb)
            if selected is None or candidate > selected:
                selected = candidate
    if selected is None:
        threshold = 1.0
        accepted_rows = joint >= threshold
        accepted_count = sum(
            bool(np.any(accepted_rows & (group_ids == group)))
            for group in unique_groups
        )
        errors = sum(
            bool(np.any(accepted_rows & ~correct & (group_ids == group)))
            for group in unique_groups
        )
        _, ucb = _wilson_interval(errors, accepted_count, confidence)
        status = "no_calibration_threshold_met_false_accept_gate_reject_all"
    else:
        accepted_count, negative_threshold, negative_errors, negative_ucb = selected
        errors = -negative_errors
        ucb = -negative_ucb
        threshold = -negative_threshold
        status = "highest_coverage_threshold_meeting_false_accept_gate"
    return float(threshold), {
        "status": status,
        "accepted_groups": accepted_count,
        "false_accept_groups": errors,
        "false_accept_wilson_ucb95": float(ucb),
    }


def fit_group_calibration(
    model: ActorVisibleCausalEventObserverV1,
    dataset: LoadedDataset,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = dataset.splits["calibration"]
    output = _predict_split(model, dataset, "calibration")
    weights = _equal_group_weights(arrays["logical_group_id"])
    event_temperature = _fit_event_temperature(
        output["event_logits"], arrays["event_label"], weights,
        config.calibration_grid_size,
    )
    predicate_temperatures = np.array(
        [
            _fit_predicate_temperature(
                output["predicate_logits"][:, index],
                arrays["predicate_label"][:, index],
                weights,
                config.calibration_grid_size,
            )
            for index in range(len(EXPECTED_PREDICATES))
        ],
        dtype=np.float64,
    )
    event_probability = _softmax(output["event_logits"] / event_temperature)
    predicate_probability = _sigmoid(
        output["predicate_logits"] / predicate_temperatures[None, :]
    )
    predicate_thresholds = np.array(
        [
            _fit_threshold(
                predicate_probability[:, index],
                arrays["predicate_label"][:, index],
                weights,
            )
            for index in range(len(EXPECTED_PREDICATES))
        ],
        dtype=np.float64,
    )
    minimum_confidence, reject_fit = _select_joint_confidence(
        event_probability,
        predicate_probability,
        arrays["event_label"],
        arrays["predicate_label"],
        predicate_thresholds,
        arrays["logical_group_id"],
        config.minimum_calibration_accepts,
        config.confidence_level,
    )
    reject_all = (
        reject_fit["status"]
        != "highest_coverage_threshold_meeting_false_accept_gate"
    )
    calibration = make_calibration(
        event_spec_sha256=model.training_contract["event_spec_sha256"],
        independent_calibration_split_sha256=dataset.manifest["splits"][
            "calibration"
        ]["logical_sha256"],
        minimum_joint_confidence=minimum_confidence,
        reject_all=reject_all,
        event_temperature=event_temperature,
        predicate_temperatures=predicate_temperatures.tolist(),
        predicate_thresholds=predicate_thresholds.tolist(),
    )
    per_actor_calibration_groups = {
        actor_name: len(set(
            arrays["logical_group_id"][arrays["actor_index"] == actor_index].tolist()
        ))
        for actor_index, actor_name in enumerate(dataset.actor_names)
    }
    calibration_group_count = len(set(arrays["logical_group_id"].tolist()))
    fit_receipt = _signed(
        {
            "format": CALIBRATION_FIT_FORMAT,
            "status": "fit_on_independent_equal_group_weighted_calibration_only",
            "calibration_split_logical_sha256": dataset.manifest["splits"][
                "calibration"
            ]["logical_sha256"],
            "logical_group_count": calibration_group_count,
            "per_actor_logical_group_count": per_actor_calibration_groups,
            "promotion_calibration_support_gate_passed": (
                calibration_group_count >= MIN_PROMOTION_GROUPS
                and all(
                    value >= MIN_PROMOTION_GROUPS_PER_ACTOR
                    for value in per_actor_calibration_groups.values()
                )
            ),
            "equal_group_weighting": True,
            "world_model_formal_or_evaluation_data_used": False,
            "temperature_grid": {
                "minimum": 0.05,
                "maximum": 20.0,
                "count": config.calibration_grid_size,
                "spacing": "natural_log",
            },
            "threshold_objective": "maximum_equal_group_weighted_f1",
            "joint_confidence_definition": (
                "min(max_event_probability,min_binary_predicate_confidence)"
            ),
            "low_confidence_reject_fit": reject_fit,
            "reject_all": reject_all,
            "calibration_sha256": calibration["calibration_sha256"],
        },
        "calibration_fit_receipt_sha256",
    )
    return calibration, fit_receipt


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(confidence) == 0:
        return 1.0
    total = len(confidence)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        rows = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(rows):
            result += float(rows.sum()) / total * abs(
                float(np.mean(confidence[rows])) - float(np.mean(correct[rows]))
            )
    return result


def _event_macro_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = [
        (
            float(np.mean(predictions[labels == item] == item))
            if np.any(labels == item) else 0.0
        )
        for item in range(len(EXPECTED_EVENTS))
    ]
    return 0.0 if not scores else float(np.mean(scores))


def _predicate_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    unit = np.full(len(labels), 1.0 / max(1, len(labels)), dtype=np.float64)
    for index in range(labels.shape[1]):
        scores.append(_weighted_f1(labels[:, index], predictions[:, index], unit))
    return float(np.mean(scores))


def _group_bootstrap_bounds(
    *, groups: np.ndarray, event_labels: np.ndarray, event_predictions: np.ndarray,
    predicate_labels: np.ndarray, predicate_predictions: np.ndarray,
    samples: int, confidence: float, seed: int,
) -> tuple[float, float]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    event_scores = np.empty(samples, dtype=np.float64)
    predicate_scores = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in drawn])
        event_scores[index] = _event_macro_accuracy(
            event_labels[rows], event_predictions[rows]
        )
        predicate_scores[index] = _predicate_macro_f1(
            predicate_labels[rows], predicate_predictions[rows]
        )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(event_scores, alpha, method="lower")),
        float(np.quantile(predicate_scores, alpha, method="lower")),
    )


def _static_privileged_input_audit() -> dict[str, Any]:
    methods = ("forward", "_validate_inputs", "_encode_history", "observe")
    found: set[str] = set()
    for method_name in methods:
        method = getattr(ActorVisibleCausalEventObserverV1, method_name)
        source = inspect.getsource(method)
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_ONLINE_NAMES:
                found.add(node.id)
            if isinstance(node, ast.arg) and node.arg in FORBIDDEN_ONLINE_NAMES:
                found.add(node.arg)
    forward_parameters = set(
        inspect.signature(ActorVisibleCausalEventObserverV1.forward).parameters
    )
    expected_parameters = {
        "self", "history", "history_mask", "proprio", "actor_names",
        "receipts", "image_features",
    }
    passed = not found and forward_parameters == expected_parameters
    return {
        "passed": passed,
        "audited_methods": list(methods),
        "forbidden_identifiers_found": sorted(found),
        "forward_parameter_names": sorted(forward_parameters),
        "expected_forward_parameter_names": sorted(expected_parameters),
    }


@torch.inference_mode()
def _future_and_branch_audits(
    model: ActorVisibleCausalEventObserverV1, dataset: LoadedDataset
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = dataset.splits["validation"]
    device = next(model.parameters()).device
    row = int(np.flatnonzero(arrays["history_mask"].sum(axis=1) < MAX_HISTORY_STEPS)[0]) \
        if np.any(arrays["history_mask"].sum(axis=1) < MAX_HISTORY_STEPS) else 0
    history = torch.from_numpy(arrays["history"][[row]]).to(device)
    mask = torch.from_numpy(arrays["history_mask"][[row]]).to(device)
    original = model._encode_history(history, mask)
    perturbed = history.clone()
    false_slots = ~mask
    if bool(false_slots.any()):
        perturbation = torch.linspace(
            -1000.0, 1000.0, int(false_slots.sum()) * STATE_DIM,
            dtype=torch.float32, device=device,
        ).reshape(int(false_slots.sum()), STATE_DIM)
        perturbed[false_slots] = perturbation
    changed = model._encode_history(perturbed, mask)
    future_passed = bool(false_slots.any()) and bool(torch.equal(original, changed))
    future = {
        "passed": future_passed,
        "audit": "masked_post_current_slots_do_not_change_gru_encoding",
        "row_sample_id": str(arrays["sample_id"][row]),
        "masked_slot_count": int(false_slots.sum()),
        "exact_tensor_equality": future_passed,
        "future_argument_absent_from_forward_signature": (
            "future_hidden" not in inspect.signature(model.forward).parameters
            and "future_image" not in inspect.signature(model.forward).parameters
        ),
    }
    future["passed"] = bool(
        future["passed"] and future["future_argument_absent_from_forward_signature"]
    )

    other_candidates = np.flatnonzero(
        (arrays["logical_group_id"] != arrays["logical_group_id"][row])
        | (arrays["branch_id"] != arrays["branch_id"][row])
    )
    if len(other_candidates) == 0:
        branch = {
            "passed": False,
            "status": "insufficient_distinct_branch_for_isolation_audit",
        }
    else:
        other = int(other_candidates[0])
        alone = _forward_indices(
            model, dataset, arrays, np.array([row], dtype=np.int64), device
        )
        mixed = _forward_indices(
            model, dataset, arrays, np.array([other, row], dtype=np.int64), device
        )
        event_delta = float(torch.max(torch.abs(
            alone["event_logits"][0] - mixed["event_logits"][1]
        )))
        predicate_delta = float(torch.max(torch.abs(
            alone["predicate_logits"][0] - mixed["predicate_logits"][1]
        )))
        event_equal = event_delta <= 1.0e-6
        predicate_equal = predicate_delta <= 1.0e-6
        branch = {
            "passed": event_equal and predicate_equal,
            "status": "same_sample_prediction_invariant_to_other_branch_batch_neighbor",
            "sample_id": str(arrays["sample_id"][row]),
            "other_sample_id": str(arrays["sample_id"][other]),
            "maximum_event_logit_absolute_delta": event_delta,
            "maximum_predicate_logit_absolute_delta": predicate_delta,
            "absolute_tolerance": 1.0e-6,
        }
    return future, branch


def evaluate_independent_validation(
    model: ActorVisibleCausalEventObserverV1,
    dataset: LoadedDataset,
    calibration: Mapping[str, Any],
    calibration_fit: Mapping[str, Any],
    config: TrainingConfig,
) -> dict[str, Any]:
    calibration = validate_calibration(
        calibration, event_spec_sha256=model.training_contract["event_spec_sha256"]
    )
    fit_logical = dict(calibration_fit)
    fit_digest = fit_logical.pop("calibration_fit_receipt_sha256", None)
    if (
        calibration_fit.get("format") != CALIBRATION_FIT_FORMAT
        or fit_digest != canonical_sha256(fit_logical)
        or calibration_fit.get("calibration_sha256")
        != calibration["calibration_sha256"]
        or calibration_fit.get("calibration_split_logical_sha256")
        != calibration["independent_calibration_split_sha256"]
    ):
        raise ObserverTrainingError("calibration fit receipt is invalid")
    arrays = dataset.splits["validation"]
    output = _predict_split(model, dataset, "validation")
    event_probability = _softmax(
        output["event_logits"] / float(calibration["event_temperature"])
    )
    predicate_probability = _sigmoid(
        output["predicate_logits"]
        / np.asarray(calibration["predicate_temperatures"], dtype=np.float64)[None, :]
    )
    thresholds = np.asarray(calibration["predicate_thresholds"], dtype=np.float64)
    event_predictions = event_probability.argmax(axis=1)
    predicate_predictions = predicate_probability >= thresholds[None, :]
    event_point = _event_macro_accuracy(arrays["event_label"], event_predictions)
    predicate_point = _predicate_macro_f1(
        arrays["predicate_label"], predicate_predictions
    )
    event_lcb, predicate_lcb = _group_bootstrap_bounds(
        groups=arrays["logical_group_id"],
        event_labels=arrays["event_label"],
        event_predictions=event_predictions,
        predicate_labels=arrays["predicate_label"],
        predicate_predictions=predicate_predictions,
        samples=config.bootstrap_samples,
        confidence=config.confidence_level,
        seed=config.seed + 1009,
    )
    event_confidence = event_probability.max(axis=1)
    event_correct = event_predictions == arrays["event_label"]
    event_ece_by_actor: dict[str, float] = {}
    predicate_ece_by_actor: dict[str, float] = {}
    event_lcb_by_actor: dict[str, float] = {}
    predicate_lcb_by_actor: dict[str, float] = {}
    per_actor_groups: dict[str, int] = {}
    for actor_index, actor_name in enumerate(dataset.actor_names):
        rows = arrays["actor_index"] == actor_index
        per_actor_groups[actor_name] = len(set(arrays["logical_group_id"][rows].tolist()))
        event_ece_by_actor[actor_name] = _ece(event_confidence[rows], event_correct[rows])
        predicate_ece_by_actor[actor_name] = max(
            _ece(
                np.maximum(predicate_probability[rows, index], 1.0 - predicate_probability[rows, index]),
                predicate_predictions[rows, index] == arrays["predicate_label"][rows, index],
            )
            for index in range(len(EXPECTED_PREDICATES))
        )
        actor_event_lcb, actor_predicate_lcb = _group_bootstrap_bounds(
            groups=arrays["logical_group_id"][rows],
            event_labels=arrays["event_label"][rows],
            event_predictions=event_predictions[rows],
            predicate_labels=arrays["predicate_label"][rows],
            predicate_predictions=predicate_predictions[rows],
            samples=config.bootstrap_samples,
            confidence=config.confidence_level,
            seed=config.seed + 2003 + actor_index,
        )
        event_lcb_by_actor[actor_name] = actor_event_lcb
        predicate_lcb_by_actor[actor_name] = actor_predicate_lcb
    predicate_binary_confidence = np.maximum(
        predicate_probability, 1.0 - predicate_probability
    ).min(axis=1)
    joint_confidence = np.minimum(event_confidence, predicate_binary_confidence)
    accepted = joint_confidence >= float(calibration["minimum_joint_confidence"])
    joint_correct = event_correct & np.all(
        predicate_predictions == arrays["predicate_label"].astype(bool), axis=1
    )
    validation_group_values = np.unique(arrays["logical_group_id"])
    accepted_group_count = sum(
        bool(np.any(accepted & (arrays["logical_group_id"] == group)))
        for group in validation_group_values
    )
    false_accepts = sum(
        bool(np.any(
            accepted & ~joint_correct & (arrays["logical_group_id"] == group)
        ))
        for group in validation_group_values
    )
    _, false_accept_ucb = _wilson_interval(
        int(false_accepts), int(accepted_group_count), config.confidence_level
    )
    future, branch = _future_and_branch_audits(model, dataset)
    static = _static_privileged_input_audit()
    train_groups = set(dataset.splits["train"]["logical_group_id"].tolist())
    calibration_groups = set(dataset.splits["calibration"]["logical_group_id"].tolist())
    validation_groups = set(arrays["logical_group_id"].tolist())
    split_disjoint = not (
        train_groups & calibration_groups
        or train_groups & validation_groups
        or calibration_groups & validation_groups
    )
    validation_group_count = len(validation_groups)
    maximum_event_ece = max(event_ece_by_actor.values())
    maximum_predicate_ece = max(predicate_ece_by_actor.values())
    conservative_event_lcb = min(event_lcb_by_actor.values())
    conservative_predicate_lcb = min(predicate_lcb_by_actor.values())
    event_support = {
        name: int(np.sum(arrays["event_label"] == index))
        for index, name in enumerate(EXPECTED_EVENTS)
    }
    predicate_support = {
        name: {
            "positive": int(np.sum(arrays["predicate_label"][:, index] == 1.0)),
            "negative": int(np.sum(arrays["predicate_label"][:, index] == 0.0)),
        }
        for index, name in enumerate(EXPECTED_PREDICATES)
    }
    gates = {
        "independent_validation_groups": validation_group_count >= MIN_PROMOTION_GROUPS,
        "per_actor_validation_groups": all(
            value >= MIN_PROMOTION_GROUPS_PER_ACTOR
            for value in per_actor_groups.values()
        ),
        "event_macro_accuracy_lcb95": (
            conservative_event_lcb >= MIN_EVENT_ACCURACY_LCB95
        ),
        "predicate_macro_f1_lcb95": (
            conservative_predicate_lcb >= MIN_PREDICATE_F1_LCB95
        ),
        "maximum_event_ece": maximum_event_ece <= MAX_CALIBRATION_ECE,
        "maximum_predicate_ece": maximum_predicate_ece <= MAX_CALIBRATION_ECE,
        "low_confidence_false_accept_ucb95": (
            false_accept_ucb <= MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95
            and accepted_group_count > 0
        ),
        "canonical_label_support": (
            all(value > 0 for value in event_support.values())
            and all(
                counts["positive"] > 0 and counts["negative"] > 0
                for counts in predicate_support.values()
            )
        ),
        "future_feature_perturbation_invariant": future["passed"],
        "cross_branch_isolation_passed": branch["passed"],
        "privileged_input_static_audit_passed": static["passed"],
        "calibration_group_disjoint": split_disjoint,
        "calibration_support": calibration_fit[
            "promotion_calibration_support_gate_passed"
        ] is True,
        "calibration_low_confidence_false_accept_fit": (
            calibration_fit["low_confidence_reject_fit"]["status"]
            == "highest_coverage_threshold_meeting_false_accept_gate"
            and calibration["reject_all"] is False
        ),
    }
    base = {
        "format": METRICS_FORMAT,
        "status": (
            "independent_validation_passed_all_gates"
            if all(gates.values())
            else "monitor_only_one_or_more_real_promotion_gates_failed"
        ),
        "validation_split_logical_sha256": dataset.manifest["splits"][
            "validation"
        ]["logical_sha256"],
        "calibration_split_logical_sha256": dataset.manifest["splits"][
            "calibration"
        ]["logical_sha256"],
        "train_split_logical_sha256": dataset.manifest["splits"]["train"][
            "logical_sha256"
        ],
        "bootstrap": {
            "unit": "logical_reset_group",
            "samples": config.bootstrap_samples,
            "confidence": config.confidence_level,
            "seed": config.seed + 1009,
            "lower_quantile_method": "numpy_lower",
        },
        "event_macro_accuracy": {
            "point": event_point,
            "pooled_group_bootstrap_lcb95": event_lcb,
            "per_actor_group_bootstrap_lcb95": event_lcb_by_actor,
            "group_bootstrap_lcb95": conservative_event_lcb,
            "promotion_aggregation": "minimum_per_actor_lcb95",
        },
        "predicate_macro_f1": {
            "point": predicate_point,
            "pooled_group_bootstrap_lcb95": predicate_lcb,
            "per_actor_group_bootstrap_lcb95": predicate_lcb_by_actor,
            "group_bootstrap_lcb95": conservative_predicate_lcb,
            "promotion_aggregation": "minimum_per_actor_lcb95",
        },
        "event_ece_by_actor": event_ece_by_actor,
        "predicate_ece_by_actor": predicate_ece_by_actor,
        "maximum_event_ece": maximum_event_ece,
        "maximum_predicate_ece": maximum_predicate_ece,
        "confidence_reject": {
            "minimum_joint_confidence": float(
                calibration["minimum_joint_confidence"]
            ),
            "accepted_rows": int(accepted.sum()),
            "accepted_groups": int(accepted_group_count),
            "false_accept_rows": int(np.sum(accepted & ~joint_correct)),
            "false_accept_groups": int(false_accepts),
            "false_accept_wilson_ucb95": false_accept_ucb,
            "wilson_unit": "logical_reset_group_any_false_accept",
        },
        "canonical_event_row_support": event_support,
        "canonical_predicate_row_support": predicate_support,
        "independent_validation_groups": validation_group_count,
        "per_actor_validation_groups": per_actor_groups,
        "future_feature_perturbation": future,
        "cross_branch_isolation": branch,
        "privileged_input_static_audit": static,
        "split_group_sets_strictly_disjoint": split_disjoint,
        "promotion_thresholds": {
            "minimum_independent_validation_groups": MIN_PROMOTION_GROUPS,
            "minimum_groups_per_actor": MIN_PROMOTION_GROUPS_PER_ACTOR,
            "minimum_event_macro_accuracy_lcb95": MIN_EVENT_ACCURACY_LCB95,
            "minimum_predicate_macro_f1_lcb95": MIN_PREDICATE_F1_LCB95,
            "maximum_ece": MAX_CALIBRATION_ECE,
            "maximum_low_confidence_false_accept_ucb95": (
                MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95
            ),
        },
        "gates": gates,
        "all_promotion_gates_passed": all(gates.values()),
        "synthetic_or_test_evidence": False,
    }
    return _signed(base, "validation_receipt_sha256")


def _promotion_decision(
    *, validation: Mapping[str, Any], core_file_sha: str,
    training_contract_sha: str, actor_names: Sequence[str],
    synthetic_evidence: bool, observer_checkpoint_file_sha: str,
    observer_config_sha: str, actor_adapter_set_sha: str,
    actor_adapter_checkpoint_set_sha: str, calibration_sha: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    gates_passed = validation.get("all_promotion_gates_passed") is True
    promoted = gates_passed and not synthetic_evidence
    status = (
        "promoted_real_independent_validation_passed_all_gates"
        if promoted
        else (
            "monitor_only_synthetic_evidence_never_authorizes_promotion"
            if synthetic_evidence
            else "monitor_only_real_validation_gate_failed"
        )
    )
    decision = _signed(
        {
            "format": PROMOTION_DECISION_FORMAT,
            "status": status,
            "promotion_enabled": promoted,
            "rerank_authority_may_be_issued": promoted,
            "synthetic_or_test_evidence": synthetic_evidence,
            "validation_receipt_sha256": validation["validation_receipt_sha256"],
            "failed_gates": sorted(
                name for name, passed in validation["gates"].items() if not passed
            ),
            "real_evidence_required": True,
        },
        "promotion_decision_sha256",
    )
    if not promoted:
        return decision, None
    evidence_base = {
        "format": PROMOTION_EVIDENCE_FORMAT,
        "status": "independent_validation_passed_all_gates",
        "observer_core_file_sha256": core_file_sha,
        "observer_checkpoint_file_sha256": observer_checkpoint_file_sha,
        "observer_config_sha256": observer_config_sha,
        "training_supervision_contract_sha256": training_contract_sha,
        "actor_adapter_set_sha256": actor_adapter_set_sha,
        "actor_adapter_checkpoint_set_sha256": (
            actor_adapter_checkpoint_set_sha
        ),
        "calibration_sha256": calibration_sha,
        "independent_calibration_split_sha256": validation[
            "calibration_split_logical_sha256"
        ],
        "independent_validation_split_sha256": validation[
            "validation_split_logical_sha256"
        ],
        "actor_names": list(actor_names),
        "independent_validation_groups": validation[
            "independent_validation_groups"
        ],
        "per_actor_validation_groups": validation[
            "per_actor_validation_groups"
        ],
        "event_macro_accuracy_lcb95": validation["event_macro_accuracy"][
            "group_bootstrap_lcb95"
        ],
        "predicate_macro_f1_lcb95": validation["predicate_macro_f1"][
            "group_bootstrap_lcb95"
        ],
        "maximum_event_ece": validation["maximum_event_ece"],
        "maximum_predicate_ece": validation["maximum_predicate_ece"],
        "low_confidence_false_accept_ucb95": validation["confidence_reject"][
            "false_accept_wilson_ucb95"
        ],
        "future_feature_perturbation_invariant": validation["gates"][
            "future_feature_perturbation_invariant"
        ],
        "cross_branch_isolation_passed": validation["gates"][
            "cross_branch_isolation_passed"
        ],
        "privileged_input_static_audit_passed": validation["gates"][
            "privileged_input_static_audit_passed"
        ],
        "calibration_group_disjoint": validation["gates"][
            "calibration_group_disjoint"
        ],
    }
    evidence = {
        **evidence_base,
        "promotion_receipt_sha256": canonical_sha256(evidence_base),
    }
    validate_promotion_evidence(
        evidence,
        observer_core_file_sha256=core_file_sha,
        observer_checkpoint_file_sha256=observer_checkpoint_file_sha,
        observer_config_sha256=observer_config_sha,
        training_contract_sha256=training_contract_sha,
        actor_adapter_set_sha256=actor_adapter_set_sha,
        actor_adapter_checkpoint_set_sha256=actor_adapter_checkpoint_set_sha,
        calibration_sha256=calibration_sha,
        actor_names=actor_names,
    )
    return decision, evidence


def freeze_bundle(
    *, model: ActorVisibleCausalEventObserverV1, dataset: LoadedDataset,
    calibration: Mapping[str, Any], calibration_fit: Mapping[str, Any],
    validation: Mapping[str, Any], training_receipt: Mapping[str, Any],
    config: TrainingConfig, output_directory: Path,
    synthetic_evidence: bool = False,
) -> dict[str, Any]:
    """Freeze exact tensors/files and issue monitor or promoted authority."""

    unresolved_output = Path(output_directory)
    if unresolved_output.is_symlink():
        raise ObserverTrainingError("freeze output directory cannot be a symlink")
    output = unresolved_output.resolve()
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise ObserverTrainingError("freeze output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    core_state, adapter_states = _split_model_state(model)
    training_contract = model.training_contract
    observer_source = Path(
        inspect.getsourcefile(ActorVisibleCausalEventObserverV1) or ""
    ).resolve()
    if not observer_source.is_file():
        raise ObserverTrainingError("observer core implementation file is unavailable")
    observer_core_file_sha = file_sha256(observer_source)
    config_document = observer_config_document(model.config)
    config_path = output / "observer_config.json"
    _atomic_json(config_path, config_document)
    training_contract_path = output / "training_contract.json"
    _atomic_json(training_contract_path, training_contract)
    training_receipt_path = output / "training_receipt.json"
    _atomic_json(training_receipt_path, training_receipt)

    core_tensor_sha = tensor_bundle_sha256(core_state)
    core_path = output / "observer_core_state.pt"
    core_payload = {
        "format": CORE_CHECKPOINT_FORMAT,
        "observer_core_file_sha256": observer_core_file_sha,
        "observer_config_sha256": config_document["config_sha256"],
        "training_contract_sha256": training_contract["contract_sha256"],
        "core_tensor_set_sha256": core_tensor_sha,
        "core_state_dict": core_state,
    }
    _atomic_torch(core_path, core_payload)
    observer_checkpoint_file_sha = file_sha256(core_path)

    adapter_records: list[dict[str, Any]] = []
    for record in dataset.actor_records:
        actor_name = str(record["actor_name"])
        state = adapter_states[actor_name]
        safe_actor = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in actor_name
        ).strip("_")
        if not safe_actor:
            safe_actor = f"actor_{record['actor_index']:03d}"
        checkpoint_path = output / (
            f"adapter_{int(record['actor_index']):03d}_{safe_actor}.pt"
        )
        contract = make_actor_adapter_contract(
            actor_name=actor_name,
            policy_family=str(record["policy_family"]),
            state_feature_source_sha256=str(record["state_feature_source_sha256"]),
            observer_core_file_sha256=observer_core_file_sha,
            training_contract_sha256=training_contract["contract_sha256"],
            image_feature_extractor_file_sha256=None,
            config=model.config,
            adapter_state=state,
        )
        adapter_tensor_sha = tensor_bundle_sha256(state)
        _atomic_torch(
            checkpoint_path,
            {
                "format": ADAPTER_CHECKPOINT_FORMAT,
                "actor_name": actor_name,
                "adapter_contract_sha256": contract[
                    "adapter_contract_sha256"
                ],
                "adapter_state_sha256": adapter_tensor_sha,
                "adapter_state_dict": state,
            },
        )
        checkpoint_file_sha = file_sha256(checkpoint_path)
        adapter_records.append(
            {
                "actor_name": actor_name,
                "adapter_contract": contract,
                "checkpoint_file": checkpoint_path.name,
                "checkpoint_file_sha256": checkpoint_file_sha,
            }
        )
    actor_adapter_set_sha = canonical_sha256(
        [
            item["adapter_contract"]["adapter_contract_sha256"]
            for item in adapter_records
        ]
    )
    adapter_checkpoint_set_sha = canonical_sha256(
        [item["checkpoint_file_sha256"] for item in adapter_records]
    )
    adapter_set_base = {
        "format": ADAPTER_MANIFEST_FORMAT,
        "training_contract_sha256": training_contract["contract_sha256"],
        "ordered_adapters": adapter_records,
        "actor_adapter_set_sha256": actor_adapter_set_sha,
        "actor_adapter_checkpoint_set_sha256": adapter_checkpoint_set_sha,
    }
    adapter_set = {
        **adapter_set_base,
        "manifest_sha256": canonical_sha256(adapter_set_base),
    }
    adapter_set_path = output / "actor_adapter_manifest.json"
    _atomic_json(adapter_set_path, adapter_set)

    calibration_path = output / "calibration.json"
    _atomic_json(calibration_path, calibration)
    calibration_fit_path = output / "calibration_fit_receipt.json"
    _atomic_json(calibration_fit_path, calibration_fit)
    validation_path = output / "independent_validation.json"
    validation_document = dict(validation)
    validation_document["synthetic_or_test_evidence"] = bool(synthetic_evidence)
    unsigned_validation = dict(validation_document)
    unsigned_validation.pop("validation_receipt_sha256")
    validation_document["validation_receipt_sha256"] = canonical_sha256(
        unsigned_validation
    )
    _atomic_json(validation_path, validation_document)

    decision, promotion_evidence = _promotion_decision(
        validation=validation_document,
        core_file_sha=observer_core_file_sha,
        training_contract_sha=training_contract["contract_sha256"],
        actor_names=dataset.actor_names,
        synthetic_evidence=synthetic_evidence,
        observer_checkpoint_file_sha=observer_checkpoint_file_sha,
        observer_config_sha=config_document["config_sha256"],
        actor_adapter_set_sha=actor_adapter_set_sha,
        actor_adapter_checkpoint_set_sha=adapter_checkpoint_set_sha,
        calibration_sha=calibration["calibration_sha256"],
    )
    decision_path = output / "promotion_decision.json"
    _atomic_json(decision_path, decision)
    evidence_record: dict[str, Any] | None = None
    if promotion_evidence is not None:
        evidence_path = output / "promotion_evidence.json"
        _atomic_json(evidence_path, promotion_evidence)
        evidence_record = {
            "path": evidence_path.name,
            "file_sha256": file_sha256(evidence_path),
            "logical_sha256": promotion_evidence["promotion_receipt_sha256"],
        }

    deployment = make_deployment(
        promotion_enabled=promotion_evidence is not None,
        promotion_evidence=promotion_evidence,
        integration_target=(
            EVALUATION400_V4_TARGET
            if promotion_evidence is not None else "monitor_only"
        ),
        promotion_validation_context=(
            {
                "observer_core_file_sha256": observer_core_file_sha,
                "observer_checkpoint_file_sha256": (
                    observer_checkpoint_file_sha
                ),
                "observer_config_sha256": config_document["config_sha256"],
                "training_contract_sha256": training_contract[
                    "contract_sha256"
                ],
                "actor_adapter_set_sha256": actor_adapter_set_sha,
                "actor_adapter_checkpoint_set_sha256": (
                    adapter_checkpoint_set_sha
                ),
                "calibration_sha256": calibration["calibration_sha256"],
                "actor_names": list(dataset.actor_names),
            }
            if promotion_evidence is not None else None
        ),
    )
    deployment_path = output / "deployment.json"
    _atomic_json(deployment_path, deployment)

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "file_sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    monitor_base = {
        "format": FREEZE_FORMAT,
        "status": (
            "frozen_promoted_evaluation400_v4_rerank"
            if promotion_evidence is not None
            else "frozen_monitor_only_no_evaluation400_v4_authority"
        ),
        "dataset_manifest": {
            "path": str(dataset.manifest_path),
            "file_sha256": file_sha256(dataset.manifest_path),
            "logical_sha256": dataset.manifest["manifest_sha256"],
        },
        "tensor_bindings": {
            "observer_core_tensor_sha256": core_tensor_sha,
            "actor_adapter_tensor_sha256_by_actor": {
                item["actor_name"]: item["adapter_contract"][
                    "adapter_state_sha256"
                ] for item in adapter_records
            },
        },
        "logical_bindings": {
            "config_sha256": config_document["config_sha256"],
            "training_contract_sha256": training_contract["contract_sha256"],
            "actor_adapter_set_sha256": actor_adapter_set_sha,
            "actor_adapter_checkpoint_set_sha256": adapter_checkpoint_set_sha,
            "calibration_sha256": calibration["calibration_sha256"],
            "validation_receipt_sha256": validation_document[
                "validation_receipt_sha256"
            ],
            "promotion_decision_sha256": decision["promotion_decision_sha256"],
            "deployment_sha256": deployment["deployment_sha256"],
            "promotion_evidence_sha256": (
                promotion_evidence["promotion_receipt_sha256"]
                if promotion_evidence is not None else None
            ),
        },
        "artifacts_excluding_this_manifest": artifacts,
        "synthetic_or_test_evidence": synthetic_evidence,
        "real_task_success_or_cross_embodiment_improvement_claimed": False,
    }
    monitor_freeze = _signed(monitor_base, "freeze_manifest_sha256")
    monitor_path = output / "monitor_freeze_manifest.json"
    _atomic_json(monitor_path, monitor_freeze)

    authority: dict[str, Any] | None = None
    authority_file_sha: str | None = None
    if promotion_evidence is not None:
        authority_artifacts = {
            "observer_config": {
                "file": config_path.name,
                "file_sha256": file_sha256(config_path),
            },
            "observer_checkpoint": {
                "file": core_path.name,
                "file_sha256": observer_checkpoint_file_sha,
            },
            "training_contract": {
                "file": training_contract_path.name,
                "file_sha256": file_sha256(training_contract_path),
            },
            "actor_adapter_manifest": {
                "file": adapter_set_path.name,
                "file_sha256": file_sha256(adapter_set_path),
            },
            "calibration": {
                "file": calibration_path.name,
                "file_sha256": file_sha256(calibration_path),
            },
            "promotion_evidence": {
                "file": "promotion_evidence.json",
                "file_sha256": evidence_record["file_sha256"],
            },
            "deployment": {
                "file": deployment_path.name,
                "file_sha256": file_sha256(deployment_path),
            },
        }
        authority_base = {
            "format": FROZEN_AUTHORITY_MANIFEST_FORMAT,
            "status": "frozen_promoted_evaluation400_v4_rerank",
            "observer_core_file_sha256": observer_core_file_sha,
            "artifacts": authority_artifacts,
            "observer_config_sha256": config_document["config_sha256"],
            "observer_checkpoint_file_sha256": observer_checkpoint_file_sha,
            "training_contract_sha256": training_contract["contract_sha256"],
            "actor_adapter_manifest_sha256": adapter_set["manifest_sha256"],
            "actor_adapter_set_sha256": actor_adapter_set_sha,
            "actor_adapter_checkpoint_set_sha256": adapter_checkpoint_set_sha,
            "calibration_sha256": calibration["calibration_sha256"],
            "promotion_evidence_sha256": promotion_evidence[
                "promotion_receipt_sha256"
            ],
            "deployment_sha256": deployment["deployment_sha256"],
        }
        authority = {
            **authority_base,
            "authority_manifest_sha256": canonical_sha256(authority_base),
        }
        authority_path = output / "authority_manifest.json"
        _atomic_json(authority_path, authority)
        authority_file_sha = file_sha256(authority_path)
    return {
        "output_directory": str(output),
        "monitor_freeze_manifest": monitor_freeze,
        "monitor_freeze_manifest_file_sha256": file_sha256(monitor_path),
        "authority_manifest": authority,
        "authority_manifest_file_sha256": authority_file_sha,
        "promotion_enabled": promotion_evidence is not None,
        "v4_rerank_authority_issued": authority is not None,
    }


def train_calibrate_validate_freeze(
    *, manifest_path: Path, output_directory: Path,
    config: TrainingConfig = TrainingConfig(), synthetic_evidence: bool = False,
) -> dict[str, Any]:
    dataset = load_supervision_dataset(manifest_path)
    model, training_receipt = train_model(dataset, config)
    calibration, calibration_fit = fit_group_calibration(model, dataset, config)
    model.calibration = validate_calibration(
        calibration, event_spec_sha256=model.training_contract["event_spec_sha256"]
    )
    validation = evaluate_independent_validation(
        model, dataset, calibration, calibration_fit, config
    )
    return freeze_bundle(
        model=model,
        dataset=dataset,
        calibration=calibration,
        calibration_fit=calibration_fit,
        validation=validation,
        training_receipt=training_receipt,
        config=config,
        output_directory=output_directory,
        synthetic_evidence=synthetic_evidence,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size-per-actor", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--synthetic-evidence",
        action="store_true",
        help="Mark test-only data; permanently disables promotion in this run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_calibrate_validate_freeze(
        manifest_path=args.dataset_manifest,
        output_directory=args.output,
        config=TrainingConfig(
            hidden_dim=args.hidden_dim,
            adapter_rank=args.adapter_rank,
            epochs=args.epochs,
            batch_size_per_actor=args.batch_size_per_actor,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            device=args.device,
        ),
        synthetic_evidence=args.synthetic_evidence,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LoadedDataset",
    "ObserverTrainingError",
    "TrainingConfig",
    "evaluate_independent_validation",
    "fit_group_calibration",
    "freeze_bundle",
    "load_supervision_dataset",
    "train_calibrate_validate_freeze",
    "train_model",
]
