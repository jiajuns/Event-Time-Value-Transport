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
    MIN_EVENT_MACRO_F1_LCB95,
    MIN_EVENT_PREDICATE_CONSISTENCY,
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
PRODUCTION_DATASET_STATUS = (
    "complete_actor_visible_causal_supervision_content_addressed"
)
PROMOTION_GATE_NAMES = frozenset(
    {
        "independent_validation_groups",
        "per_actor_validation_groups",
        "event_macro_accuracy_lcb95",
        "event_macro_f1_lcb95",
        "predicate_macro_f1_lcb95",
        "event_macro_f1_gain_over_train_frequency_lcb95",
        "predicate_macro_f1_gain_over_train_constant_lcb95",
        "maximum_event_ece",
        "maximum_predicate_ece",
        "low_confidence_false_accept_ucb95",
        "canonical_label_support",
        "event_predicate_ontology_consistency",
        "future_feature_perturbation_invariant",
        "cross_branch_isolation_passed",
        "privileged_input_static_audit_passed",
        "calibration_group_disjoint",
        "calibration_support",
        "calibration_low_confidence_false_accept_fit",
    }
)

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
    event_predicate_consistency_loss_weight: float = 0.25
    class_balance_beta: float = 0.99
    maximum_class_weight: float = 5.0
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    calibration_grid_size: int = 121
    minimum_calibration_accepts: int = MIN_PROMOTION_GROUPS
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
            or self.event_predicate_consistency_loss_weight < 0.0
            or not 0.0 <= self.class_balance_beta < 1.0
            or self.maximum_class_weight < 1.0
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


def _label_support_audit(dataset: LoadedDataset) -> dict[str, Any]:
    """Describe label support without exposing labels as online inputs.

    Promotion metrics are meaningless when a canonical event or either side of
    a predicate has no independent-group support.  The audit is content
    addressed and travels with the training receipt so that a row-heavy split
    cannot hide sparse group coverage.
    """

    split_records: dict[str, Any] = {}
    all_train_actor_support = True
    for split_name in DATASET_SPLITS:
        arrays = dataset.splits[split_name]
        groups = arrays["logical_group_id"]
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        root_rows = np.flatnonzero(arrays["current_query_index"] == 0)
        root_fingerprints = {
            hashlib.sha256(
                arrays["history"][row].tobytes(order="C")
                + arrays["history_mask"][row].tobytes(order="C")
                + arrays["proprio"][row].tobytes(order="C")
            ).hexdigest()
            for row in root_rows
        }
        actor_records: dict[str, Any] = {}
        for actor_index, actor_name in enumerate(dataset.actor_names):
            actor_rows = arrays["actor_index"] == actor_index
            event_support: dict[str, Any] = {}
            for event_index, event_name in enumerate(EXPECTED_EVENTS):
                selected = actor_rows & (arrays["event_label"] == event_index)
                event_support[event_name] = {
                    "rows": int(selected.sum()),
                    "independent_groups": int(len(np.unique(groups[selected]))),
                }
            predicate_support: dict[str, Any] = {}
            for predicate_index, predicate_name in enumerate(EXPECTED_PREDICATES):
                positive = actor_rows & (
                    arrays["predicate_label"][:, predicate_index] == 1.0
                )
                negative = actor_rows & (
                    arrays["predicate_label"][:, predicate_index] == 0.0
                )
                predicate_support[predicate_name] = {
                    "positive_rows": int(positive.sum()),
                    "positive_independent_groups": int(
                        len(np.unique(groups[positive]))
                    ),
                    "negative_rows": int(negative.sum()),
                    "negative_independent_groups": int(
                        len(np.unique(groups[negative]))
                    ),
                }
            canonical_support = bool(
                all(row["rows"] > 0 and row["independent_groups"] > 0
                    for row in event_support.values())
                and all(
                    row["positive_rows"] > 0
                    and row["positive_independent_groups"] > 0
                    and row["negative_rows"] > 0
                    and row["negative_independent_groups"] > 0
                    for row in predicate_support.values()
                )
            )
            if split_name == "train":
                all_train_actor_support = all_train_actor_support and canonical_support
            actor_records[actor_name] = {
                "rows": int(actor_rows.sum()),
                "independent_groups": int(len(np.unique(groups[actor_rows]))),
                "event": event_support,
                "predicate": predicate_support,
                "canonical_binary_and_event_support_present": canonical_support,
            }
        branch_values, branch_counts = np.unique(
            arrays["branch_id"], return_counts=True
        )
        split_records[split_name] = {
            "logical_sha256": dataset.manifest["splits"][split_name][
                "logical_sha256"
            ],
            "rows": int(len(groups)),
            "independent_groups": int(len(unique_groups)),
            "rows_per_group": {
                "minimum": int(group_counts.min()),
                "median": float(np.median(group_counts)),
                "maximum": int(group_counts.max()),
            },
            "branches": {
                str(name): int(count)
                for name, count in zip(branch_values.tolist(), branch_counts.tolist(), strict=True)
            },
            "root_rows": int(len(root_rows)),
            "root_unique_actor_visible_input_fingerprints": int(
                len(root_fingerprints)
            ),
            "root_duplicate_fraction": (
                0.0
                if len(root_rows) == 0
                else float(1.0 - len(root_fingerprints) / len(root_rows))
            ),
            "terminal_success_rows": int(
                np.sum(arrays["predicate_label"][:, EXPECTED_PREDICATES.index("success")] == 1.0)
            ),
            "actors": actor_records,
        }
    base = {
        "format": "etsf_causal_event_observer_label_support_audit_v1",
        "status": "complete_offline_labels_never_available_to_online_model_inputs",
        "dataset_manifest_sha256": dataset.manifest["manifest_sha256"],
        "split_unit": "logical_reset_group",
        "splits": split_records,
        "all_train_actors_have_canonical_label_support": bool(
            all_train_actor_support
        ),
    }
    return _signed(base, "label_support_audit_sha256")


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


def _effective_number_class_weights(
    counts: np.ndarray, *, beta: float, maximum: float
) -> np.ndarray:
    """Return capped weights with unit expected row weight."""

    values = np.asarray(counts, dtype=np.float64)
    present = values > 0
    result = np.zeros_like(values)
    if not np.any(present):
        return result
    if beta == 0.0:
        raw = np.ones(int(present.sum()), dtype=np.float64)
    else:
        raw = (1.0 - beta) / np.maximum(1.0 - np.power(beta, values[present]), 1.0e-12)
    raw *= values[present].sum() / np.sum(values[present] * raw)
    raw = np.clip(raw, 1.0 / maximum, maximum)
    raw *= values[present].sum() / np.sum(values[present] * raw)
    result[present] = raw
    return result


def _class_balance_contract(
    arrays: Mapping[str, np.ndarray], actor_count: int, config: TrainingConfig
) -> dict[str, np.ndarray]:
    event_weights = np.zeros((actor_count, len(EXPECTED_EVENTS)), dtype=np.float64)
    predicate_positive_weights = np.ones(
        (actor_count, len(EXPECTED_PREDICATES)), dtype=np.float64
    )
    predicate_normalizers = np.ones_like(predicate_positive_weights)
    for actor in range(actor_count):
        rows = arrays["actor_index"] == actor
        event_counts = np.bincount(
            arrays["event_label"][rows], minlength=len(EXPECTED_EVENTS)
        )
        event_weights[actor] = _effective_number_class_weights(
            event_counts,
            beta=config.class_balance_beta,
            maximum=config.maximum_class_weight,
        )
        for predicate in range(len(EXPECTED_PREDICATES)):
            positive = float(np.sum(arrays["predicate_label"][rows, predicate] == 1.0))
            negative = float(np.sum(arrays["predicate_label"][rows, predicate] == 0.0))
            if positive > 0.0 and negative > 0.0:
                if config.class_balance_beta == 0.0:
                    effective_positive = positive
                    effective_negative = negative
                else:
                    effective_positive = (
                        1.0 - config.class_balance_beta ** positive
                    ) / (1.0 - config.class_balance_beta)
                    effective_negative = (
                        1.0 - config.class_balance_beta ** negative
                    ) / (1.0 - config.class_balance_beta)
                weight = math.sqrt(effective_negative / effective_positive)
                weight = min(
                    config.maximum_class_weight,
                    max(1.0 / config.maximum_class_weight, weight),
                )
                predicate_positive_weights[actor, predicate] = weight
                predicate_normalizers[actor, predicate] = (
                    negative + positive * weight
                ) / (negative + positive)
    return {
        "event": event_weights.astype(np.float32),
        "predicate_positive": predicate_positive_weights.astype(np.float32),
        "predicate_normalizer": predicate_normalizers.astype(np.float32),
    }


def _hierarchical_epoch_indices(
    arrays: Mapping[str, np.ndarray],
    actor_count: int,
    generator: np.random.Generator,
    balance: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Sample actor -> logical group -> rarity-weighted row."""

    actor_rows = [
        np.flatnonzero(arrays["actor_index"] == actor)
        for actor in range(actor_count)
    ]
    if any(len(rows) == 0 for rows in actor_rows):
        raise ObserverTrainingError("hierarchical sampler encountered unsupported actor")
    target = max(len(rows) for rows in actor_rows)
    sampled_by_actor: list[np.ndarray] = []
    for actor, rows in enumerate(actor_rows):
        groups = np.unique(arrays["logical_group_id"][rows])
        by_group = {
            group: rows[arrays["logical_group_id"][rows] == group]
            for group in groups
        }
        selected_groups = generator.choice(groups, size=target, replace=True)
        selected_rows: list[int] = []
        for group in selected_groups:
            candidates = by_group[group]
            event_rarity = balance["event"][
                actor, arrays["event_label"][candidates]
            ].astype(np.float64)
            positive = arrays["predicate_label"][candidates] == 1.0
            predicate_rarity = np.where(
                positive,
                balance["predicate_positive"][actor][None, :],
                1.0,
            ).max(axis=1)
            rarity = np.sqrt(np.maximum(event_rarity, 1.0e-6) * predicate_rarity)
            rarity = np.clip(rarity, 1.0 / 5.0, 5.0)
            probability = rarity / rarity.sum()
            selected_rows.append(int(generator.choice(candidates, p=probability)))
        sampled_by_actor.append(np.asarray(selected_rows, dtype=np.int64))
    sampled = np.stack(sampled_by_actor)
    generator.shuffle(sampled, axis=1)
    return sampled.T.reshape(-1)


def _structured_event_probability_torch(
    predicate_logits: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(predicate_logits)
    moved, lifted, near_goal, stationary, success = probability.unbind(dim=-1)
    not_success = 1.0 - success
    not_stationary = 1.0 - stationary
    not_near = 1.0 - near_goal
    not_moved_or_lifted = (1.0 - moved) * (1.0 - lifted)
    return torch.stack(
        [
            not_success * not_stationary * not_near * not_moved_or_lifted,
            not_success * not_stationary * not_near * (1.0 - not_moved_or_lifted),
            not_success * not_stationary * near_goal,
            not_success * stationary,
            success,
        ],
        dim=-1,
    ).clamp_min(1.0e-8)


def _balanced_loss(
    output: Mapping[str, torch.Tensor],
    event_target: torch.Tensor,
    predicate_target: torch.Tensor,
    actor_target: torch.Tensor,
    actor_count: int,
    config: TrainingConfig,
    *,
    group_target: torch.Tensor | None = None,
    class_balance: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, float, float, float]:
    if group_target is None:
        group_target = torch.arange(len(event_target), device=event_target.device)
    if class_balance is None:
        event_class_weight = torch.ones(
            (actor_count, len(EXPECTED_EVENTS)), device=event_target.device
        )
        predicate_positive_weight = torch.ones(
            (actor_count, len(EXPECTED_PREDICATES)), device=event_target.device
        )
        predicate_normalizer = torch.ones_like(predicate_positive_weight)
    else:
        event_class_weight = class_balance["event"]
        predicate_positive_weight = class_balance["predicate_positive"]
        predicate_normalizer = class_balance["predicate_normalizer"]
    event_rows = -torch.log_softmax(output["event_logits"], dim=-1).gather(
        1, event_target[:, None]
    ).squeeze(1)
    event_rows = event_rows * event_class_weight[actor_target, event_target]
    predicate_entries = F.binary_cross_entropy_with_logits(
        output["predicate_logits"], predicate_target, reduction="none"
    )
    positive_weight = predicate_positive_weight[actor_target]
    normalizer = predicate_normalizer[actor_target]
    predicate_entries = predicate_entries * torch.where(
        predicate_target == 1.0, positive_weight, torch.ones_like(positive_weight)
    ) / normalizer
    predicate_rows = predicate_entries.mean(dim=1)
    event_probability = torch.softmax(output["event_logits"], dim=-1).clamp_min(1.0e-8)
    structured_probability = _structured_event_probability_torch(
        output["predicate_logits"]
    )
    mixture = (0.5 * (event_probability + structured_probability)).clamp_min(1.0e-8)
    consistency_rows = 0.5 * (
        (event_probability * (event_probability.log() - mixture.log())).sum(dim=-1)
        + (
            structured_probability
            * (structured_probability.log() - mixture.log())
        ).sum(dim=-1)
    )
    event_actor: list[torch.Tensor] = []
    predicate_actor: list[torch.Tensor] = []
    consistency_actor: list[torch.Tensor] = []
    for actor in range(actor_count):
        rows = actor_target == actor
        if bool(rows.any()):
            actor_groups = torch.unique(group_target[rows])
            event_actor.append(torch.stack([
                event_rows[rows & (group_target == group)].mean()
                for group in actor_groups
            ]).mean())
            predicate_actor.append(torch.stack([
                predicate_rows[rows & (group_target == group)].mean()
                for group in actor_groups
            ]).mean())
            consistency_actor.append(torch.stack([
                consistency_rows[rows & (group_target == group)].mean()
                for group in actor_groups
            ]).mean())
    if len(event_actor) != actor_count:
        raise ObserverTrainingError("batch omitted an actor; balanced loss cannot proceed")
    event_loss = torch.stack(event_actor).mean()
    predicate_loss = torch.stack(predicate_actor).mean()
    consistency_loss = torch.stack(consistency_actor).mean()
    total = (
        config.event_loss_weight * event_loss
        + config.predicate_loss_weight * predicate_loss
        + config.event_predicate_consistency_loss_weight * consistency_loss
    )
    return (
        total,
        float(event_loss.detach()),
        float(predicate_loss.detach()),
        float(consistency_loss.detach()),
    )


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
    balance_numpy = _class_balance_contract(arrays, actor_count, config)
    balance_torch = {
        name: torch.from_numpy(value).to(device)
        for name, value in balance_numpy.items()
    }
    _, group_index = np.unique(
        arrays["logical_group_id"], return_inverse=True
    )
    label_support = _label_support_audit(dataset)
    for epoch in range(config.epochs):
        order = _hierarchical_epoch_indices(
            arrays, actor_count, generator, balance_numpy
        )
        losses: list[float] = []
        event_losses: list[float] = []
        predicate_losses: list[float] = []
        consistency_losses: list[float] = []
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
            group_target = torch.from_numpy(group_index[indices]).to(device)
            loss, event_loss, predicate_loss, consistency_loss = _balanced_loss(
                output,
                event_target,
                predicate_target,
                actor_target,
                actor_count,
                config,
                group_target=group_target,
                class_balance=balance_torch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            event_losses.append(event_loss)
            predicate_losses.append(predicate_loss)
            consistency_losses.append(consistency_loss)
        if not losses:
            raise ObserverTrainingError("training produced no optimizer step")
        epoch_records.append(
            {
                "epoch": epoch,
                "balanced_rows": int(len(order)),
                "loss": float(np.mean(losses)),
                "event_ce": float(np.mean(event_losses)),
                "predicate_bce": float(np.mean(predicate_losses)),
                "event_predicate_js": float(np.mean(consistency_losses)),
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
            "logical_group_balanced_sampling": True,
            "logical_group_normalized_loss": True,
            "rare_class_balanced_sampling": True,
            "event_objective": (
                "equal_actor_equal_group_effective_number_weighted_cross_entropy"
            ),
            "predicate_objective": (
                "equal_actor_equal_group_capped_effective_number_binary_cross_entropy"
            ),
            "event_predicate_consistency_objective": (
                "priority_ontology_jensen_shannon_divergence"
            ),
            "class_balance": {
                name: value.astype(np.float64).tolist()
                for name, value in balance_numpy.items()
            },
            "label_support_audit": label_support,
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


def _predicate_decision_confidence(
    probability: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probability, dtype=np.float64)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 2 or threshold_values.shape != (values.shape[1],):
        raise ObserverTrainingError("predicate confidence tensors are misaligned")
    decisions = values >= threshold_values[None, :]
    confidence = np.where(decisions, values, 1.0 - values)
    return decisions, confidence


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
    predicate_predictions, predicate_decision_confidence = (
        _predicate_decision_confidence(
            predicate_probability, predicate_thresholds
        )
    )
    return _select_joint_confidence_from_oof(
        event_confidence=event_confidence,
        predicate_decision_confidence=predicate_decision_confidence,
        event_predictions=event_probability.argmax(axis=1),
        predicate_predictions=predicate_predictions,
        event_labels=event_labels,
        predicate_labels=predicate_labels,
        group_ids=group_ids,
        minimum_accepts=minimum_accepts,
        confidence=confidence,
    )


def _select_joint_confidence_from_oof(
    *,
    event_confidence: np.ndarray,
    predicate_decision_confidence: np.ndarray,
    event_predictions: np.ndarray,
    predicate_predictions: np.ndarray,
    event_labels: np.ndarray,
    predicate_labels: np.ndarray,
    group_ids: np.ndarray,
    minimum_accepts: int,
    confidence: float,
) -> tuple[float, dict[str, Any]]:
    predicate_confidence = predicate_decision_confidence.min(axis=1)
    joint = np.minimum(event_confidence, predicate_confidence)
    correct = (
        (event_predictions == event_labels)
        & np.all(
            predicate_predictions == predicate_labels.astype(bool),
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


def _group_oof_calibration_predictions(
    *,
    event_logits: np.ndarray,
    predicate_logits: np.ndarray,
    event_labels: np.ndarray,
    predicate_labels: np.ndarray,
    group_ids: np.ndarray,
    config: TrainingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Cross-fit all head calibration before fitting the risk-control gate."""

    unique_groups = np.unique(group_ids)
    fold_count = min(5, len(unique_groups))
    if fold_count < 2:
        raise ObserverTrainingError("group-OOF calibration requires at least two groups")
    shuffled = unique_groups.copy()
    np.random.default_rng(config.seed + 4099).shuffle(shuffled)
    group_to_fold = {
        group: index % fold_count for index, group in enumerate(shuffled)
    }
    row_fold = np.asarray([group_to_fold[group] for group in group_ids], dtype=np.int64)
    event_confidence = np.full(len(group_ids), np.nan, dtype=np.float64)
    event_predictions = np.full(len(group_ids), -1, dtype=np.int64)
    predicate_decision_confidence = np.full(
        predicate_labels.shape, np.nan, dtype=np.float64
    )
    predicate_predictions = np.zeros(predicate_labels.shape, dtype=np.bool_)
    fold_records: list[dict[str, Any]] = []
    all_fit_support_complete = True
    for fold in range(fold_count):
        heldout = row_fold == fold
        fit = ~heldout
        fit_weights = _equal_group_weights(group_ids[fit])
        event_temperature = _fit_event_temperature(
            event_logits[fit], event_labels[fit], fit_weights,
            config.calibration_grid_size,
        )
        predicate_temperatures = np.asarray([
            _fit_predicate_temperature(
                predicate_logits[fit, index],
                predicate_labels[fit, index],
                fit_weights,
                config.calibration_grid_size,
            )
            for index in range(len(EXPECTED_PREDICATES))
        ], dtype=np.float64)
        fit_predicate_probability = _sigmoid(
            predicate_logits[fit] / predicate_temperatures[None, :]
        )
        predicate_thresholds = np.asarray([
            _fit_threshold(
                fit_predicate_probability[:, index],
                predicate_labels[fit, index],
                fit_weights,
            )
            for index in range(len(EXPECTED_PREDICATES))
        ], dtype=np.float64)
        heldout_event_probability = _softmax(
            event_logits[heldout] / event_temperature
        )
        heldout_predicate_probability = _sigmoid(
            predicate_logits[heldout] / predicate_temperatures[None, :]
        )
        heldout_predicate_prediction, heldout_predicate_confidence = (
            _predicate_decision_confidence(
                heldout_predicate_probability, predicate_thresholds
            )
        )
        event_confidence[heldout] = heldout_event_probability.max(axis=1)
        event_predictions[heldout] = heldout_event_probability.argmax(axis=1)
        predicate_decision_confidence[heldout] = heldout_predicate_confidence
        predicate_predictions[heldout] = heldout_predicate_prediction
        fit_event_support = [
            int(np.sum(event_labels[fit] == index))
            for index in range(len(EXPECTED_EVENTS))
        ]
        fit_predicate_support = [
            {
                "positive": int(np.sum(predicate_labels[fit, index] == 1.0)),
                "negative": int(np.sum(predicate_labels[fit, index] == 0.0)),
            }
            for index in range(len(EXPECTED_PREDICATES))
        ]
        support_complete = bool(
            all(value > 0 for value in fit_event_support)
            and all(
                value["positive"] > 0 and value["negative"] > 0
                for value in fit_predicate_support
            )
        )
        all_fit_support_complete = all_fit_support_complete and support_complete
        fit_groups = sorted(set(group_ids[fit].tolist()))
        heldout_groups = sorted(set(group_ids[heldout].tolist()))
        fold_records.append({
            "fold": fold,
            "fit_group_count": len(fit_groups),
            "heldout_group_count": len(heldout_groups),
            "fit_group_set_sha256": canonical_sha256(fit_groups),
            "heldout_group_set_sha256": canonical_sha256(heldout_groups),
            "fit_heldout_group_disjoint": not bool(set(fit_groups) & set(heldout_groups)),
            "fit_canonical_label_support_complete": support_complete,
        })
    if (
        not np.isfinite(event_confidence).all()
        or np.any(event_predictions < 0)
        or not np.isfinite(predicate_decision_confidence).all()
    ):
        raise ObserverTrainingError("group-OOF calibration did not cover every row")
    receipt = {
        "fold_count": fold_count,
        "assignment_seed": config.seed + 4099,
        "assignment_unit": "logical_reset_group",
        "every_row_predicted_exactly_once": True,
        "all_fit_partitions_have_canonical_label_support": bool(
            all_fit_support_complete
        ),
        "folds": fold_records,
    }
    return {
        "event_confidence": event_confidence,
        "event_predictions": event_predictions,
        "predicate_decision_confidence": predicate_decision_confidence,
        "predicate_predictions": predicate_predictions,
    }, receipt


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
    oof, oof_receipt = _group_oof_calibration_predictions(
        event_logits=output["event_logits"],
        predicate_logits=output["predicate_logits"],
        event_labels=arrays["event_label"],
        predicate_labels=arrays["predicate_label"],
        group_ids=arrays["logical_group_id"],
        config=config,
    )
    minimum_confidence, reject_fit = _select_joint_confidence_from_oof(
        event_confidence=oof["event_confidence"],
        predicate_decision_confidence=oof[
            "predicate_decision_confidence"
        ],
        event_predictions=oof["event_predictions"],
        predicate_predictions=oof["predicate_predictions"],
        event_labels=arrays["event_label"],
        predicate_labels=arrays["predicate_label"],
        group_ids=arrays["logical_group_id"],
        minimum_accepts=config.minimum_calibration_accepts,
        confidence=config.confidence_level,
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
    label_support = _label_support_audit(dataset)
    calibration_actor_support = label_support["splits"]["calibration"]["actors"]
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
                and all(
                    row["canonical_binary_and_event_support_present"] is True
                    for row in calibration_actor_support.values()
                )
                and oof_receipt[
                    "all_fit_partitions_have_canonical_label_support"
                ] is True
            ),
            "minimum_risk_control_accepted_groups": max(
                config.minimum_calibration_accepts, MIN_PROMOTION_GROUPS
            ),
            "wilson_zero_error_minimum_group_derivation": {
                "confidence": config.confidence_level,
                "maximum_false_accept_ucb95": (
                    MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95
                ),
                "minimum_independent_groups": MIN_PROMOTION_GROUPS,
            },
            "calibration_actor_label_support": calibration_actor_support,
            "label_support_audit_sha256": label_support[
                "label_support_audit_sha256"
            ],
            "equal_group_weighting": True,
            "risk_control_predictions": "group_oof_head_calibration",
            "group_oof_calibration": oof_receipt,
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


def _ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = 10,
    weights: np.ndarray | None = None,
) -> float:
    if len(confidence) == 0:
        return 1.0
    if weights is None:
        normalized_weights = np.full(
            len(confidence), 1.0 / len(confidence), dtype=np.float64
        )
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if (
            normalized_weights.shape != (len(confidence),)
            or not np.isfinite(normalized_weights).all()
            or np.any(normalized_weights < 0.0)
            or normalized_weights.sum() <= 0.0
        ):
            raise ObserverTrainingError("ECE weights are invalid")
        normalized_weights = normalized_weights / normalized_weights.sum()
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        rows = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(rows):
            mass = float(normalized_weights[rows].sum())
            result += mass * abs(
                float(np.average(confidence[rows], weights=normalized_weights[rows]))
                - float(np.average(correct[rows], weights=normalized_weights[rows]))
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


def _event_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores: list[float] = []
    for item in range(len(EXPECTED_EVENTS)):
        true_positive = int(np.sum((labels == item) & (predictions == item)))
        false_positive = int(np.sum((labels != item) & (predictions == item)))
        false_negative = int(np.sum((labels == item) & (predictions != item)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    return float(np.mean(scores))


def _predicate_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    unit = np.full(len(labels), 1.0 / max(1, len(labels)), dtype=np.float64)
    for index in range(labels.shape[1]):
        scores.append(_weighted_f1(labels[:, index], predictions[:, index], unit))
    return float(np.mean(scores))


def _train_frequency_baselines(
    dataset: LoadedDataset,
    validation_arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = dataset.splits["train"]
    event_prediction = np.empty(
        len(validation_arrays["event_label"]), dtype=np.int64
    )
    predicate_prediction = np.empty(
        validation_arrays["predicate_label"].shape, dtype=np.bool_
    )
    actor_contracts: dict[str, Any] = {}
    for actor_index, actor_name in enumerate(dataset.actor_names):
        train_rows = train["actor_index"] == actor_index
        validation_rows = validation_arrays["actor_index"] == actor_index
        weights = _equal_group_weights(train["logical_group_id"][train_rows])
        event_frequency = np.asarray([
            float(np.sum(weights * (train["event_label"][train_rows] == event)))
            for event in range(len(EXPECTED_EVENTS))
        ])
        majority_event = int(np.argmax(event_frequency))
        predicate_prevalence = np.asarray([
            float(np.sum(
                weights * train["predicate_label"][train_rows, predicate]
            ))
            for predicate in range(len(EXPECTED_PREDICATES))
        ])
        constant_predicates = predicate_prevalence >= 0.5
        event_prediction[validation_rows] = majority_event
        predicate_prediction[validation_rows] = constant_predicates
        actor_contracts[actor_name] = {
            "train_equal_group_event_frequency": event_frequency.tolist(),
            "majority_event": EXPECTED_EVENTS[majority_event],
            "train_equal_group_predicate_prevalence": (
                predicate_prevalence.tolist()
            ),
            "constant_predicates": {
                name: bool(constant_predicates[index])
                for index, name in enumerate(EXPECTED_PREDICATES)
            },
        }
    return event_prediction, predicate_prediction, actor_contracts


def _group_bootstrap_gain_bounds(
    *,
    groups: np.ndarray,
    event_labels: np.ndarray,
    event_predictions: np.ndarray,
    baseline_event_predictions: np.ndarray,
    predicate_labels: np.ndarray,
    predicate_predictions: np.ndarray,
    baseline_predicate_predictions: np.ndarray,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    event_gain = np.empty(samples, dtype=np.float64)
    predicate_gain = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in drawn])
        event_gain[index] = (
            _event_macro_f1(event_labels[rows], event_predictions[rows])
            - _event_macro_f1(
                event_labels[rows], baseline_event_predictions[rows]
            )
        )
        predicate_gain[index] = (
            _predicate_macro_f1(
                predicate_labels[rows], predicate_predictions[rows]
            )
            - _predicate_macro_f1(
                predicate_labels[rows], baseline_predicate_predictions[rows]
            )
        )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(event_gain, alpha, method="lower")),
        float(np.quantile(predicate_gain, alpha, method="lower")),
    )


def _group_bootstrap_bounds(
    *, groups: np.ndarray, event_labels: np.ndarray, event_predictions: np.ndarray,
    predicate_labels: np.ndarray, predicate_predictions: np.ndarray,
    samples: int, confidence: float, seed: int,
) -> tuple[float, float, float]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    event_scores = np.empty(samples, dtype=np.float64)
    event_f1_scores = np.empty(samples, dtype=np.float64)
    predicate_scores = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in drawn])
        event_scores[index] = _event_macro_accuracy(
            event_labels[rows], event_predictions[rows]
        )
        event_f1_scores[index] = _event_macro_f1(
            event_labels[rows], event_predictions[rows]
        )
        predicate_scores[index] = _predicate_macro_f1(
            predicate_labels[rows], predicate_predictions[rows]
        )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(event_scores, alpha, method="lower")),
        float(np.quantile(event_f1_scores, alpha, method="lower")),
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
    predicate_predictions, predicate_decision_confidence = (
        _predicate_decision_confidence(predicate_probability, thresholds)
    )
    event_point = _event_macro_accuracy(arrays["event_label"], event_predictions)
    event_f1_point = _event_macro_f1(
        arrays["event_label"], event_predictions
    )
    predicate_point = _predicate_macro_f1(
        arrays["predicate_label"], predicate_predictions
    )
    event_lcb, event_f1_lcb, predicate_lcb = _group_bootstrap_bounds(
        groups=arrays["logical_group_id"],
        event_labels=arrays["event_label"],
        event_predictions=event_predictions,
        predicate_labels=arrays["predicate_label"],
        predicate_predictions=predicate_predictions,
        samples=config.bootstrap_samples,
        confidence=config.confidence_level,
        seed=config.seed + 1009,
    )
    (
        baseline_event_predictions,
        baseline_predicate_predictions,
        baseline_contracts,
    ) = _train_frequency_baselines(dataset, arrays)
    baseline_event_f1 = _event_macro_f1(
        arrays["event_label"], baseline_event_predictions
    )
    baseline_predicate_f1 = _predicate_macro_f1(
        arrays["predicate_label"], baseline_predicate_predictions
    )
    pooled_event_gain_lcb, pooled_predicate_gain_lcb = (
        _group_bootstrap_gain_bounds(
            groups=arrays["logical_group_id"],
            event_labels=arrays["event_label"],
            event_predictions=event_predictions,
            baseline_event_predictions=baseline_event_predictions,
            predicate_labels=arrays["predicate_label"],
            predicate_predictions=predicate_predictions,
            baseline_predicate_predictions=baseline_predicate_predictions,
            samples=config.bootstrap_samples,
            confidence=config.confidence_level,
            seed=config.seed + 1511,
        )
    )
    event_confidence = event_probability.max(axis=1)
    event_correct = event_predictions == arrays["event_label"]
    event_ece_by_actor: dict[str, float] = {}
    predicate_ece_by_actor: dict[str, float] = {}
    event_lcb_by_actor: dict[str, float] = {}
    event_f1_lcb_by_actor: dict[str, float] = {}
    predicate_lcb_by_actor: dict[str, float] = {}
    event_gain_lcb_by_actor: dict[str, float] = {}
    predicate_gain_lcb_by_actor: dict[str, float] = {}
    per_actor_groups: dict[str, int] = {}
    for actor_index, actor_name in enumerate(dataset.actor_names):
        rows = arrays["actor_index"] == actor_index
        actor_group_weights = _equal_group_weights(
            arrays["logical_group_id"][rows]
        )
        per_actor_groups[actor_name] = len(set(arrays["logical_group_id"][rows].tolist()))
        event_ece_by_actor[actor_name] = _ece(
            event_confidence[rows], event_correct[rows],
            weights=actor_group_weights,
        )
        predicate_ece_by_actor[actor_name] = max(
            _ece(
                predicate_decision_confidence[rows, index],
                predicate_predictions[rows, index] == arrays["predicate_label"][rows, index],
                weights=actor_group_weights,
            )
            for index in range(len(EXPECTED_PREDICATES))
        )
        actor_event_lcb, actor_event_f1_lcb, actor_predicate_lcb = (
            _group_bootstrap_bounds(
            groups=arrays["logical_group_id"][rows],
            event_labels=arrays["event_label"][rows],
            event_predictions=event_predictions[rows],
            predicate_labels=arrays["predicate_label"][rows],
            predicate_predictions=predicate_predictions[rows],
            samples=config.bootstrap_samples,
            confidence=config.confidence_level,
            seed=config.seed + 2003 + actor_index,
            )
        )
        event_lcb_by_actor[actor_name] = actor_event_lcb
        event_f1_lcb_by_actor[actor_name] = actor_event_f1_lcb
        predicate_lcb_by_actor[actor_name] = actor_predicate_lcb
        actor_event_gain_lcb, actor_predicate_gain_lcb = (
            _group_bootstrap_gain_bounds(
                groups=arrays["logical_group_id"][rows],
                event_labels=arrays["event_label"][rows],
                event_predictions=event_predictions[rows],
                baseline_event_predictions=baseline_event_predictions[rows],
                predicate_labels=arrays["predicate_label"][rows],
                predicate_predictions=predicate_predictions[rows],
                baseline_predicate_predictions=(
                    baseline_predicate_predictions[rows]
                ),
                samples=config.bootstrap_samples,
                confidence=config.confidence_level,
                seed=config.seed + 3011 + actor_index,
            )
        )
        event_gain_lcb_by_actor[actor_name] = actor_event_gain_lcb
        predicate_gain_lcb_by_actor[actor_name] = actor_predicate_gain_lcb
    predicate_binary_confidence = predicate_decision_confidence.min(axis=1)
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
    conservative_event_f1_lcb = min(event_f1_lcb_by_actor.values())
    conservative_predicate_lcb = min(predicate_lcb_by_actor.values())
    conservative_event_gain_lcb = min(event_gain_lcb_by_actor.values())
    conservative_predicate_gain_lcb = min(predicate_gain_lcb_by_actor.values())
    ontology_event_predictions = np.where(
        predicate_predictions[:, EXPECTED_PREDICATES.index("success")],
        EXPECTED_EVENTS.index("eK"),
        np.where(
            predicate_predictions[:, EXPECTED_PREDICATES.index("stationary")],
            EXPECTED_EVENTS.index("e4"),
            np.where(
                predicate_predictions[:, EXPECTED_PREDICATES.index("near_goal")],
                EXPECTED_EVENTS.index("e3"),
                np.where(
                    predicate_predictions[:, EXPECTED_PREDICATES.index("moved")]
                    | predicate_predictions[:, EXPECTED_PREDICATES.index("lifted")],
                    EXPECTED_EVENTS.index("e12"),
                    EXPECTED_EVENTS.index("e0"),
                ),
            ),
        ),
    )
    validation_group_weights = _equal_group_weights(arrays["logical_group_id"])
    event_predicate_consistency = float(np.sum(
        validation_group_weights
        * (event_predictions == ontology_event_predictions).astype(np.float64)
    ))
    event_support = {
        name: {
            "rows": int(np.sum(arrays["event_label"] == index)),
            "independent_groups": int(len(np.unique(
                arrays["logical_group_id"][arrays["event_label"] == index]
            ))),
        }
        for index, name in enumerate(EXPECTED_EVENTS)
    }
    predicate_support = {
        name: {
            "positive": int(np.sum(arrays["predicate_label"][:, index] == 1.0)),
            "negative": int(np.sum(arrays["predicate_label"][:, index] == 0.0)),
            "positive_independent_groups": int(len(np.unique(
                arrays["logical_group_id"][
                    arrays["predicate_label"][:, index] == 1.0
                ]
            ))),
            "negative_independent_groups": int(len(np.unique(
                arrays["logical_group_id"][
                    arrays["predicate_label"][:, index] == 0.0
                ]
            ))),
        }
        for index, name in enumerate(EXPECTED_PREDICATES)
    }
    label_support_audit = _label_support_audit(dataset)
    validation_actor_label_support = label_support_audit["splits"][
        "validation"
    ]["actors"]
    gates = {
        "independent_validation_groups": validation_group_count >= MIN_PROMOTION_GROUPS,
        "per_actor_validation_groups": all(
            value >= MIN_PROMOTION_GROUPS_PER_ACTOR
            for value in per_actor_groups.values()
        ),
        "event_macro_accuracy_lcb95": (
            conservative_event_lcb >= MIN_EVENT_ACCURACY_LCB95
        ),
        "event_macro_f1_lcb95": (
            conservative_event_f1_lcb >= MIN_EVENT_MACRO_F1_LCB95
        ),
        "predicate_macro_f1_lcb95": (
            conservative_predicate_lcb >= MIN_PREDICATE_F1_LCB95
        ),
        "event_macro_f1_gain_over_train_frequency_lcb95": (
            conservative_event_gain_lcb > 0.0
        ),
        "predicate_macro_f1_gain_over_train_constant_lcb95": (
            conservative_predicate_gain_lcb > 0.0
        ),
        "maximum_event_ece": maximum_event_ece <= MAX_CALIBRATION_ECE,
        "maximum_predicate_ece": maximum_predicate_ece <= MAX_CALIBRATION_ECE,
        "low_confidence_false_accept_ucb95": (
            false_accept_ucb <= MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95
            and accepted_group_count > 0
        ),
        "canonical_label_support": (
            all(
                value["rows"] > 0 and value["independent_groups"] > 0
                for value in event_support.values()
            )
            and all(
                counts["positive"] > 0
                and counts["negative"] > 0
                and counts["positive_independent_groups"] > 0
                and counts["negative_independent_groups"] > 0
                for counts in predicate_support.values()
            )
            and all(
                row["canonical_binary_and_event_support_present"] is True
                for row in validation_actor_label_support.values()
            )
        ),
        "event_predicate_ontology_consistency": (
            event_predicate_consistency >= MIN_EVENT_PREDICATE_CONSISTENCY
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
        "event_macro_f1": {
            "point": event_f1_point,
            "pooled_group_bootstrap_lcb95": event_f1_lcb,
            "per_actor_group_bootstrap_lcb95": event_f1_lcb_by_actor,
            "group_bootstrap_lcb95": conservative_event_f1_lcb,
            "promotion_aggregation": "minimum_per_actor_lcb95",
        },
        "predicate_macro_f1": {
            "point": predicate_point,
            "pooled_group_bootstrap_lcb95": predicate_lcb,
            "per_actor_group_bootstrap_lcb95": predicate_lcb_by_actor,
            "group_bootstrap_lcb95": conservative_predicate_lcb,
            "promotion_aggregation": "minimum_per_actor_lcb95",
        },
        "train_only_frequency_and_constant_baselines": {
            "fit_split": "train",
            "fit_weighting": "equal_logical_group_within_actor",
            "validation_labels_used_to_fit_baseline": False,
            "actor_contracts": baseline_contracts,
            "event_macro_f1": baseline_event_f1,
            "predicate_macro_f1": baseline_predicate_f1,
            "event_macro_f1_gain_group_bootstrap_lcb95": {
                "pooled": pooled_event_gain_lcb,
                "per_actor": event_gain_lcb_by_actor,
                "promotion_aggregation": conservative_event_gain_lcb,
            },
            "predicate_macro_f1_gain_group_bootstrap_lcb95": {
                "pooled": pooled_predicate_gain_lcb,
                "per_actor": predicate_gain_lcb_by_actor,
                "promotion_aggregation": conservative_predicate_gain_lcb,
            },
        },
        "event_ece_by_actor": event_ece_by_actor,
        "predicate_ece_by_actor": predicate_ece_by_actor,
        "ece_weighting": "equal_logical_group_within_actor",
        "maximum_event_ece": maximum_event_ece,
        "maximum_predicate_ece": maximum_predicate_ece,
        "event_predicate_ontology_consistency": {
            "equal_group_agreement": event_predicate_consistency,
            "minimum_for_promotion": MIN_EVENT_PREDICATE_CONSISTENCY,
            "event_priority": ["success", "stationary", "near_goal", "moved_or_lifted", "none"],
        },
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
        "validation_actor_label_support": validation_actor_label_support,
        "label_support_audit_sha256": label_support_audit[
            "label_support_audit_sha256"
        ],
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
            "minimum_event_macro_f1_lcb95": MIN_EVENT_MACRO_F1_LCB95,
            "minimum_predicate_macro_f1_lcb95": MIN_PREDICATE_F1_LCB95,
            "minimum_event_macro_f1_gain_over_train_frequency_lcb95": 0.0,
            "minimum_predicate_macro_f1_gain_over_train_constant_lcb95": 0.0,
            "gain_comparison": "strict_greater_than_zero",
            "maximum_ece": MAX_CALIBRATION_ECE,
            "maximum_low_confidence_false_accept_ucb95": (
                MAX_LOW_CONFIDENCE_FALSE_ACCEPT_UCB95
            ),
            "minimum_event_predicate_ontology_consistency": (
                MIN_EVENT_PREDICATE_CONSISTENCY
            ),
        },
        "gates": gates,
        "all_promotion_gates_passed": all(gates.values()),
        "synthetic_or_test_evidence": (
            dataset.manifest.get("status") != PRODUCTION_DATASET_STATUS
        ),
    }
    return _signed(base, "validation_receipt_sha256")


def _validate_training_receipt_for_freeze(
    receipt: Mapping[str, Any], *, dataset: LoadedDataset, config: TrainingConfig,
) -> None:
    fields = {
        "format", "status", "seed", "device", "training_config",
        "actor_balanced_sampling", "actor_balanced_loss",
        "logical_group_balanced_sampling", "logical_group_normalized_loss",
        "rare_class_balanced_sampling", "event_objective",
        "predicate_objective", "event_predicate_consistency_objective",
        "class_balance", "label_support_audit", "train_split_logical_sha256",
        "calibration_or_validation_used_by_optimizer", "epochs",
        "training_receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != fields:
        raise ObserverTrainingError("training receipt fields changed before freeze")
    logical = dict(receipt)
    digest = logical.pop("training_receipt_sha256", None)
    expected_support = _label_support_audit(dataset)
    expected_balance = {
        name: value.astype(np.float64).tolist()
        for name, value in _class_balance_contract(
            dataset.splits["train"], len(dataset.actor_names), config
        ).items()
    }
    epochs = receipt.get("epochs")
    epoch_fields = {
        "epoch", "balanced_rows", "loss", "event_ce", "predicate_bce",
        "event_predicate_js",
    }
    epoch_valid = (
        isinstance(epochs, list)
        and len(epochs) == config.epochs
        and all(
            isinstance(row, Mapping)
            and set(row) == epoch_fields
            and row.get("epoch") == index
            and type(row.get("epoch")) is int
            and type(row.get("balanced_rows")) is int
            and row["balanced_rows"] > 0
            and all(
                not isinstance(row.get(name), bool)
                and isinstance(row.get(name), (int, float))
                and math.isfinite(float(row[name]))
                for name in ("loss", "event_ce", "predicate_bce", "event_predicate_js")
            )
            for index, row in enumerate(epochs)
        )
    )
    if (
        digest != canonical_sha256(logical)
        or receipt.get("format") != TRAINING_RECEIPT_FORMAT
        or receipt.get("status") != "optimizer_complete_train_split_only"
        or receipt.get("seed") != config.seed
        or type(receipt.get("seed")) is not int
        or receipt.get("device") != config.device
        or receipt.get("training_config") != asdict(config)
        or receipt.get("actor_balanced_sampling") is not True
        or receipt.get("actor_balanced_loss") is not True
        or receipt.get("logical_group_balanced_sampling") is not True
        or receipt.get("logical_group_normalized_loss") is not True
        or receipt.get("rare_class_balanced_sampling") is not True
        or receipt.get("calibration_or_validation_used_by_optimizer") is not False
        or receipt.get("train_split_logical_sha256")
        != dataset.manifest["splits"]["train"]["logical_sha256"]
        or receipt.get("label_support_audit") != expected_support
        or receipt.get("class_balance") != expected_balance
        or not epoch_valid
    ):
        raise ObserverTrainingError(
            "training receipt is not self-signed and content-bound"
        )


def _recompute_and_validate_freeze_receipts(
    *, model: ActorVisibleCausalEventObserverV1, dataset: LoadedDataset,
    calibration: Mapping[str, Any], calibration_fit: Mapping[str, Any],
    validation: Mapping[str, Any], config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild every promotion-bearing receipt from frozen in-memory content."""

    if not all(
        isinstance(value, Mapping)
        for value in (calibration, calibration_fit, validation)
    ):
        raise ObserverTrainingError("freeze promotion receipts must be mappings")
    if model.training:
        raise ObserverTrainingError("observer must be in eval mode before freeze")
    expected_calibration, expected_fit = fit_group_calibration(
        model, dataset, config
    )
    if dict(calibration) != expected_calibration:
        raise ObserverTrainingError(
            "calibration differs from internally recomputed calibration content"
        )
    if dict(calibration_fit) != expected_fit:
        raise ObserverTrainingError(
            "calibration fit receipt differs from internally recomputed content"
        )
    expected_validation = evaluate_independent_validation(
        model, dataset, expected_calibration, expected_fit, config
    )
    if dict(validation) != expected_validation:
        raise ObserverTrainingError(
            "validation receipt differs from internally recomputed content"
        )
    return expected_calibration, expected_fit, expected_validation


def _validate_promotion_validation_receipt(
    validation: Mapping[str, Any], *, synthetic_evidence: bool,
) -> bool:
    if not isinstance(validation, Mapping):
        raise ObserverTrainingError("promotion validation receipt is missing")
    logical = dict(validation)
    digest = logical.pop("validation_receipt_sha256", None)
    gates = validation.get("gates")
    if (
        digest != canonical_sha256(logical)
        or validation.get("format") != METRICS_FORMAT
        or not isinstance(gates, Mapping)
        or set(gates) != PROMOTION_GATE_NAMES
        or any(type(value) is not bool for value in gates.values())
        or validation.get("all_promotion_gates_passed") is not all(gates.values())
        or type(validation.get("synthetic_or_test_evidence")) is not bool
    ):
        raise ObserverTrainingError(
            "promotion validation receipt is not complete and self-consistent"
        )
    gates_passed = all(gates.values())
    expected_status = (
        "independent_validation_passed_all_gates"
        if gates_passed
        else "monitor_only_one_or_more_real_promotion_gates_failed"
    )
    if validation.get("status") != expected_status:
        raise ObserverTrainingError("promotion validation status contradicts its gates")
    return (
        gates_passed
        and not synthetic_evidence
        and validation["synthetic_or_test_evidence"] is False
    )


def _promotion_decision(
    *, validation: Mapping[str, Any], core_file_sha: str,
    training_contract_sha: str, actor_names: Sequence[str],
    synthetic_evidence: bool, observer_checkpoint_file_sha: str,
    observer_config_sha: str, actor_adapter_set_sha: str,
    actor_adapter_checkpoint_set_sha: str, calibration_sha: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if type(synthetic_evidence) is not bool:
        raise ObserverTrainingError("synthetic evidence marker must be exact boolean")
    promoted = _validate_promotion_validation_receipt(
        validation, synthetic_evidence=synthetic_evidence
    )
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
        "event_macro_f1_lcb95": validation["event_macro_f1"][
            "group_bootstrap_lcb95"
        ],
        "predicate_macro_f1_lcb95": validation["predicate_macro_f1"][
            "group_bootstrap_lcb95"
        ],
        "event_predicate_ontology_consistency": validation[
            "event_predicate_ontology_consistency"
        ]["equal_group_agreement"],
        "event_macro_f1_gain_over_train_frequency_lcb95": validation[
            "train_only_frequency_and_constant_baselines"
        ]["event_macro_f1_gain_group_bootstrap_lcb95"]["promotion_aggregation"],
        "predicate_macro_f1_gain_over_train_constant_lcb95": validation[
            "train_only_frequency_and_constant_baselines"
        ]["predicate_macro_f1_gain_group_bootstrap_lcb95"]["promotion_aggregation"],
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

    if type(synthetic_evidence) is not bool:
        raise ObserverTrainingError("synthetic evidence marker must be exact boolean")
    effective_synthetic_evidence = (
        synthetic_evidence
        or dataset.manifest.get("status") != PRODUCTION_DATASET_STATUS
    )
    if model.training_contract.get("dataset_manifest_sha256") != dataset.manifest.get(
        "manifest_sha256"
    ):
        raise ObserverTrainingError("observer training contract changed dataset authority")
    _validate_training_receipt_for_freeze(
        training_receipt, dataset=dataset, config=config
    )
    calibration, calibration_fit, validation = (
        _recompute_and_validate_freeze_receipts(
            model=model,
            dataset=dataset,
            calibration=calibration,
            calibration_fit=calibration_fit,
            validation=validation,
            config=config,
        )
    )
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
    label_support = training_receipt.get("label_support_audit")
    if not isinstance(label_support, Mapping):
        raise ObserverTrainingError("training receipt lacks label support audit")
    label_support_logical = dict(label_support)
    label_support_digest = label_support_logical.pop(
        "label_support_audit_sha256", None
    )
    if (
        label_support.get("format")
        != "etsf_causal_event_observer_label_support_audit_v1"
        or label_support.get("dataset_manifest_sha256")
        != dataset.manifest["manifest_sha256"]
        or label_support_digest != canonical_sha256(label_support_logical)
    ):
        raise ObserverTrainingError("label support audit binding is invalid")
    label_support_path = output / "label_support_audit.json"
    _atomic_json(label_support_path, label_support)

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
    _atomic_json(validation_path, validation_document)

    decision, promotion_evidence = _promotion_decision(
        validation=validation_document,
        core_file_sha=observer_core_file_sha,
        training_contract_sha=training_contract["contract_sha256"],
        actor_names=dataset.actor_names,
        synthetic_evidence=effective_synthetic_evidence,
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
            "label_support_audit_sha256": label_support_digest,
            "promotion_decision_sha256": decision["promotion_decision_sha256"],
            "deployment_sha256": deployment["deployment_sha256"],
            "promotion_evidence_sha256": (
                promotion_evidence["promotion_receipt_sha256"]
                if promotion_evidence is not None else None
            ),
        },
        "artifacts_excluding_this_manifest": artifacts,
        "synthetic_or_test_evidence": effective_synthetic_evidence,
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
    parser.add_argument(
        "--event-predicate-consistency-loss-weight", type=float, default=0.25
    )
    parser.add_argument("--class-balance-beta", type=float, default=0.99)
    parser.add_argument("--maximum-class-weight", type=float, default=5.0)
    parser.add_argument(
        "--minimum-calibration-accepts", type=int, default=MIN_PROMOTION_GROUPS
    )
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
            event_predicate_consistency_loss_weight=(
                args.event_predicate_consistency_loss_weight
            ),
            class_balance_beta=args.class_balance_beta,
            maximum_class_weight=args.maximum_class_weight,
            minimum_calibration_accepts=args.minimum_calibration_accepts,
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
