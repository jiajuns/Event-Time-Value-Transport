#!/usr/bin/env python3
"""Train only v8 detached structured-prediction adapters.

The input is a materialised schema5 training artifact.  This trainer never
loads, optimizes, or serializes a factual world model; it consumes frozen
features/outputs produced by an outer-training-owned factual checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from openvla_etsf_v8_structured_adapters import (
    V8_ADAPTER_FORMAT,
    V8_LOSS_CONTRACT,
    V8_OBJECT_MODE,
    V8_SCHEMA_VERSION,
    V8DetachedStructuredAdapters,
    V8StructuredAdapterConfig,
    frozen_tensor_mapping_sha256,
    module_state_sha256,
    train_v8_adapter_one_step,
    validate_factual_adapter_inputs,
    validate_schema5_adapter_batch,
)


V8_TRAINING_INPUT_FORMAT = "etsf_v8_detached_adapter_training_input_v1"
V8_TRAINING_CHECKPOINT_FORMAT = "etsf_v8_detached_adapter_checkpoint_v1"
REQUIRED_PROVENANCE_SHA_FIELDS = (
    "base_checkpoint_sha256",
    "outer_training_groups_sha256",
    "label_derivation_sha256",
    "duration_baseline_contract_sha256",
    "duration_laplace_scale_contract_sha256",
    "object_fallback_contract_sha256",
    "uncertainty_materialization_contract_sha256",
)
DEPLOYMENT_CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def structured_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Content-hash a nested payload, including every tensor and label."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu().reshape(-1)
            raw = tensor.view(torch.uint8).numpy().tobytes()
            update({"dtype": str(value.dtype), "shape": list(value.shape)})
            digest.update(b"tensor:")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(value, Mapping):
            digest.update(b"mapping:")
            filtered = {
                str(key): item
                for key, item in value.items()
                if str(key) not in {"payload_sha256", "_artifact_authentication"}
            }
            for key in sorted(filtered):
                update(key)
                update(filtered[key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            digest.update(b"sequence:")
            digest.update(len(value).to_bytes(8, "big"))
            for item in value:
                update(item)
        elif isinstance(value, str):
            raw = value.encode("utf-8")
            digest.update(b"string:")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        elif isinstance(value, bytes):
            digest.update(b"bytes:")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        elif value is None:
            digest.update(b"none")
        elif isinstance(value, bool):
            digest.update(b"bool:1" if value else b"bool:0")
        elif isinstance(value, int):
            digest.update(f"int:{value}".encode("ascii"))
        elif isinstance(value, float):
            digest.update(b"float:")
            digest.update(struct.pack(">d", value))
        elif hasattr(value, "item"):
            update(value.item())
        elif hasattr(value, "tolist"):
            update(value.tolist())
        else:
            raise TypeError(f"unsupported v8 payload value: {type(value).__name__}")

    update(payload)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def load_authenticated_training_payload(
    *, input_path: Path, materialization_manifest_path: Path, outer_fold_id: int
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Authenticate one fold against the signed, atomically published bundle."""

    input_path = input_path.resolve()
    manifest_path = materialization_manifest_path.resolve()
    if ".partial" in input_path.name or not input_path.is_file():
        raise RuntimeError("v8 refuses a partial or missing training artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("v8 materialization manifest must be a mapping")
    unsigned = dict(manifest)
    recorded_manifest_sha = str(unsigned.pop("materialization_sha256", ""))
    if (
        manifest.get("format") != "etsf_v8_oof_materialization_manifest_v1"
        or manifest.get("status") != "complete_development_only"
        or recorded_manifest_sha != _canonical_sha256(unsigned)
        or manifest.get("prospective_claim_for_v8") is not False
    ):
        raise RuntimeError("v8 materialization manifest signature/status changed")
    rows = manifest.get("folds")
    if not isinstance(rows, list) or [row.get("outer_fold_id") for row in rows] != list(
        range(5)
    ):
        raise RuntimeError("v8 materialization manifest lacks the complete five-fold bundle")
    bundle_hash_rows = []
    for row in rows:
        for role in ("train", "holdout"):
            artifact = Path(str(row.get(f"{role}_artifact", ""))).resolve()
            expected = row.get(f"{role}_artifact_sha256")
            if not artifact.is_file() or _sha256_path(artifact) != expected:
                raise RuntimeError(f"v8 {role} artifact SHA changed for fold {row.get('outer_fold_id')}")
            bundle_hash_rows.append(
                {
                    "outer_fold_id": row["outer_fold_id"],
                    "role": role,
                    "path": str(artifact),
                    "sha256": expected,
                    "payload_sha256": row.get(f"{role}_payload_sha256"),
                }
            )
    selected = rows[int(outer_fold_id)] if 0 <= int(outer_fold_id) < 5 else None
    if selected is None or Path(str(selected["train_artifact"])).resolve() != input_path:
        raise RuntimeError("v8 input path does not match the requested owner fold")
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("v8 authenticated training artifact must be a mapping")
    if (
        payload.get("payload_sha256") != selected.get("train_payload_sha256")
        or payload.get("payload_sha256") != structured_payload_sha256(payload)
        or payload.get("provenance", {}).get("outer_fold_id") != int(outer_fold_id)
        or payload.get("provenance", {}).get("outer_training_groups_sha256")
        != selected.get("training_groups_sha256")
    ):
        raise RuntimeError("v8 training payload/fold provenance authentication failed")
    return payload, {
        "status": "authenticated_complete_five_fold_materialization_bundle",
        "materialization_manifest": str(manifest_path),
        "materialization_sha256": recorded_manifest_sha,
        "outer_fold_id": int(outer_fold_id),
        "train_artifact_sha256": selected["train_artifact_sha256"],
        "train_payload_sha256": selected["train_payload_sha256"],
        "ten_artifact_bundle_sha256": _canonical_sha256(bundle_hash_rows),
    }


def _validate_record(
    record: Mapping[str, Any], config: V8StructuredAdapterConfig
) -> dict[str, int]:
    required = {
        "logical_group_key",
        "split_role",
        "outer_fold_id",
        "group_metadata",
        "batch",
        "factual_outputs",
        "factual_outputs_sha256",
        "factual_outputs_require_grad",
        "total_uncertainty_status",
        "duration_baseline_log1p",
        "object_fallback",
        "object_delta_physical",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"v8 training record missing fields: {missing}")
    batch = record["batch"]
    factual = record["factual_outputs"]
    if not isinstance(batch, Mapping) or not isinstance(factual, Mapping):
        raise ValueError("batch and factual_outputs must be mappings")
    if record["factual_outputs_require_grad"] is not False:
        raise ValueError("materialized factual outputs must be explicitly gradient-free")
    actual_factual_sha = frozen_tensor_mapping_sha256(factual)
    if record["factual_outputs_sha256"] != actual_factual_sha:
        raise ValueError("materialized factual_outputs_sha256 mismatch")
    if any(
        torch.is_tensor(value) and value.requires_grad for value in factual.values()
    ):
        raise ValueError("materialized factual output tensors must not require gradients")
    transition = factual.get("transition")
    if not torch.is_tensor(transition) or transition.ndim != 2:
        raise ValueError("training factual transition must be [items,features]")
    count = int(transition.shape[0])
    support = validate_schema5_adapter_batch(batch, expected_count=count)
    validate_factual_adapter_inputs(
        factual, count=count, transition_dim=config.transition_dim
    )
    for name in ("next_event_logits", "next_reached_event_logits"):
        value = factual.get(name)
        if (
            not torch.is_tensor(value)
            or value.ndim != 2
            or value.shape[0] != count
            or value.shape[1] < 2
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"materialized {name} must be finite [items,events]")
    if factual["next_reached_event_logits"].shape != factual[
        "next_event_logits"
    ].shape:
        raise ValueError("materialized next-event logit vocabularies differ")
    uncertainty = factual.get("aleatoric_uncertainty")
    if (
        not torch.is_tensor(uncertainty)
        or tuple(uncertainty.shape) != (count,)
        or not bool(torch.isfinite(uncertainty).all())
        or bool((uncertainty < 0).any())
    ):
        raise ValueError("materialized aleatoric_uncertainty must be finite non-negative [items]")
    if record.get("total_uncertainty_status") != (
        "unavailable_single_forward_has_aleatoric_only_requires_ensemble_fail_closed"
    ):
        raise ValueError("v8 may not fabricate total uncertainty from a single forward")
    metadata = record["group_metadata"]
    if not isinstance(metadata, Mapping) or (
        metadata.get("logical_group_key") != record["logical_group_key"]
        or int(metadata.get("schema_version", -1)) != V8_SCHEMA_VERSION
        or metadata.get("policy") != "openvla"
        or tuple(metadata.get("candidate_names", ())) != DEPLOYMENT_CANDIDATE_NAMES
    ):
        raise ValueError("v8 group metadata/candidate contract changed")
    group_keys = batch.get("group_keys")
    candidate_names = tuple(batch.get("candidate_names", ()))
    if group_keys != [record["logical_group_key"]] or len(candidate_names) != count:
        raise ValueError("v8 flattened group identity/candidate names changed")
    if candidate_names[:4] != DEPLOYMENT_CANDIDATE_NAMES or candidate_names[4:] != tuple(
        f"continuation_{index}" for index in range(count - 4)
    ):
        raise ValueError("v8 candidate or continuation placeholder order changed")
    terminal = batch["terminal_mask"].bool()
    group_index = batch.get("group_index")
    baseline = batch.get("baseline_mask")
    if (
        count < 4
        or not bool(terminal[:4].all())
        or bool(terminal[4:].any())
        or not torch.is_tensor(group_index)
        or not torch.equal(
            group_index.cpu(),
            torch.tensor([0] * 4 + [-1] * (count - 4), dtype=group_index.dtype),
        )
        or not torch.is_tensor(baseline)
        or not torch.equal(
            baseline.bool().cpu(),
            torch.tensor([True, False, False, False] + [False] * (count - 4)),
        )
    ):
        raise ValueError("v8 terminal/continuation/group/baseline placeholders changed")
    for label_name in (
        "current_event_id",
        "next_event_id",
        "next_reached_event_id",
    ):
        label = batch.get(label_name)
        if (
            not torch.is_tensor(label)
            or tuple(label.shape) != (count,)
            or bool((label < 0).any())
            or bool((label >= factual["next_event_logits"].shape[1]).any())
        ):
            raise ValueError(f"v8 {label_name} is invalid for the frozen event vocabulary")
    baseline = record["duration_baseline_log1p"]
    if not torch.is_tensor(baseline) or tuple(baseline.shape) != (count,):
        raise ValueError("duration_baseline_log1p must align with schema5 rows")
    if not bool(torch.isfinite(baseline).all()):
        raise ValueError("duration_baseline_log1p contains non-finite values")
    fallback = record["object_fallback"]
    if not torch.is_tensor(fallback) or fallback.ndim not in (1, 2):
        raise ValueError("object_fallback must be a tensor with one or two dimensions")
    object_dim = int(batch["object_delta"].shape[1])
    if tuple(fallback.shape) not in ((object_dim,), (count, object_dim)):
        raise ValueError("object_fallback does not match schema5 object_delta")
    if not bool(torch.isfinite(fallback).all()):
        raise ValueError("object_fallback contains non-finite values")
    physical_object = record["object_delta_physical"]
    if (
        not torch.is_tensor(physical_object)
        or tuple(physical_object.shape) != tuple(batch["object_delta"].shape)
        or not bool(torch.isfinite(physical_object).all())
    ):
        raise ValueError("object_delta_physical must be a finite schema5-aligned tensor")
    devices = {
        transition.device,
        factual["duration_selected_log_mean"].device,
        baseline.device,
        fallback.device,
    }
    if len(devices) != 1:
        raise ValueError("one v8 training record must reside on one device")
    return support


def validate_v8_training_payload(
    payload: Mapping[str, Any],
) -> tuple[V8StructuredAdapterConfig, list[Mapping[str, Any]], dict[str, Any]]:
    if payload.get("format") != V8_TRAINING_INPUT_FORMAT:
        raise ValueError("unknown v8 training input format")
    if int(payload.get("schema_version", -1)) != V8_SCHEMA_VERSION:
        raise ValueError("v8 training input must contain schema5 batches")
    if not _is_sha256(payload.get("payload_sha256")) or payload[
        "payload_sha256"
    ] != structured_payload_sha256(payload):
        raise ValueError("v8 training payload SHA is missing or invalid")
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError("v8 training input lacks an adapter config")
    config = V8StructuredAdapterConfig.from_dict(payload["config"])
    records_value = payload.get("batches")
    if (
        not isinstance(records_value, Sequence)
        or isinstance(records_value, (str, bytes))
        or not records_value
        or any(not isinstance(record, Mapping) for record in records_value)
    ):
        raise ValueError("v8 training input needs a non-empty list of batches")
    records = list(records_value)
    for record in records:
        _validate_record(record, config)

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("v8 training input lacks provenance")
    missing_sha = [
        field
        for field in REQUIRED_PROVENANCE_SHA_FIELDS
        if not _is_sha256(provenance.get(field))
    ]
    if missing_sha:
        raise ValueError(f"v8 provenance SHA fields invalid: {missing_sha}")
    outer_fold = provenance.get("outer_fold_id")
    if not isinstance(outer_fold, int) or not 0 <= outer_fold < 5:
        raise ValueError("v8 provenance outer_fold_id must be in [0,4]")
    if provenance.get("target_outer_fold_labels_used") is not False:
        raise ValueError("v8 provenance must exclude target outer-fold labels")
    if provenance.get("factual_outputs_frozen") is not True:
        raise ValueError("v8 provenance must mark factual outputs frozen")
    if provenance.get("object_mode") != V8_OBJECT_MODE:
        raise ValueError("v8 provenance must retain the object fallback contract")
    if provenance.get("object_pose_quality_status") != (
        "unavailable_schema5_collector_has_no_quality_field_fail_closed"
    ):
        raise ValueError("v8 must fail closed when object pose quality is unavailable")
    duration_scale = provenance.get("duration_laplace_scale_contract")
    if not isinstance(duration_scale, Mapping):
        raise ValueError("v8 duration Laplace scale contract is missing")
    duration_unsigned = dict(duration_scale)
    duration_recorded_sha = duration_unsigned.pop("contract_sha256", "")
    if (
        duration_scale.get("format")
        != "etsf_v8_outer_training_duration_laplace_scale_v1"
        or duration_scale.get("owner_fold_id") != outer_fold
        or duration_scale.get("fit_scope") != "outer_training_observed_only"
        or duration_scale.get("estimator")
        != "median_absolute_deviation_divided_by_log_2"
        or duration_scale.get("censored_rows_used") is not False
        or int(duration_scale.get("outer_training_observed_support", 0)) <= 0
        or duration_recorded_sha != _canonical_sha256(duration_unsigned)
        or duration_recorded_sha
        != provenance.get("duration_laplace_scale_contract_sha256")
    ):
        raise ValueError("v8 duration Laplace scale provenance changed")
    uncertainty_contract = provenance.get("uncertainty_materialization_contract")
    if not isinstance(uncertainty_contract, Mapping) or (
        uncertainty_contract.get("stored_tensor") != "aleatoric_uncertainty"
        or uncertainty_contract.get("epistemic_uncertainty")
        != "unavailable_requires_frozen_ensemble"
        or uncertainty_contract.get("total_uncertainty")
        != "unavailable_not_fabricated_fail_closed"
        or uncertainty_contract.get("ensemble_total_uncertainty_claim") is not False
        or uncertainty_contract.get("uncertainty_materialization_contract_sha256")
        != provenance.get("uncertainty_materialization_contract_sha256")
    ):
        raise ValueError("v8 single-member uncertainty provenance changed")
    for record in records:
        if (
            record.get("split_role") != "outer_training"
            or record.get("outer_fold_id") != outer_fold
        ):
            raise ValueError("v8 record role/fold differs from training provenance")
    exclusion = provenance.get("base_target_outer_fold_exclusion_status")
    if exclusion not in ("proven", "unproven_development_only"):
        raise ValueError(
            "v8 provenance must state whether the factual base excludes the target fold"
        )
    return config, records, dict(provenance)


def _move_mapping(value: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: item.to(device) if torch.is_tensor(item) else item
        for key, item in value.items()
    }


def _move_record(record: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "batch": _move_mapping(record["batch"], device),
        "factual_outputs": _move_mapping(record["factual_outputs"], device),
        "duration_baseline_log1p": record["duration_baseline_log1p"].to(device),
        "object_fallback": record["object_fallback"].to(device),
    }


def _aggregate_binary_support(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result = {
        "success_support": 0,
        "success_positive": 0,
        "regress_support": 0,
        "regress_positive": 0,
        "recovery_given_regress_support": 0,
        "recovery_given_regress_positive": 0,
    }
    for record in records:
        batch = record["batch"]
        terminal = batch["terminal_mask"].bool()
        structured = batch["structured_mask"].bool()
        regress = batch["trajectory_regress"].bool()
        recovery = batch["trajectory_recovery"].bool()
        result["success_support"] += int(terminal.sum())
        result["success_positive"] += int((batch["success"].bool() & terminal).sum())
        result["regress_support"] += int(structured.sum())
        result["regress_positive"] += int((regress & structured).sum())
        conditional = regress & structured
        result["recovery_given_regress_support"] += int(conditional.sum())
        result["recovery_given_regress_positive"] += int(
            (recovery & conditional).sum()
        )
    for name in ("success", "regress", "recovery_given_regress"):
        support = result[f"{name}_support"]
        positive = result[f"{name}_positive"]
        if support < 2 or positive <= 0 or positive >= support:
            raise ValueError(f"v8 {name} head needs both classes in outer training data")
    return result


def _prevalence(support: Mapping[str, int], name: str) -> float:
    return support[f"{name}_positive"] / support[f"{name}_support"]


def _frozen_record_sha256(record: Mapping[str, Any]) -> str:
    return frozen_tensor_mapping_sha256(
        {
            **record["factual_outputs"],
            "duration_baseline_log1p": record["duration_baseline_log1p"],
            "object_fallback": record["object_fallback"],
        }
    )


def _aggregate_frozen_record_sha256(record_hashes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(record_hashes):
        digest.update(f"{index}:{value}\n".encode("ascii"))
    return digest.hexdigest()


def train_v8_payload(
    payload: Mapping[str, Any],
    *,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer hyperparameters are invalid")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config, records, provenance = validate_v8_training_payload(payload)
    support = _aggregate_binary_support(records)
    frozen_record_hashes = [_frozen_record_sha256(record) for record in records]
    adapters = V8DetachedStructuredAdapters(config).to(device)
    adapters.initialize_probability_biases(
        success_prevalence=_prevalence(support, "success"),
        regress_prevalence=_prevalence(support, "regress"),
        recovery_given_regress_prevalence=_prevalence(
            support, "recovery_given_regress"
        ),
    )
    optimizer = torch.optim.AdamW(
        adapters.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    step_reports: list[dict[str, Any]] = []
    adapters.train()
    for _ in range(epochs):
        for record in records:
            moved = _move_record(record, device)
            step_reports.append(
                train_v8_adapter_one_step(
                    adapters,
                    optimizer,
                    moved["factual_outputs"],
                    moved["batch"],
                    duration_baseline_log1p=moved["duration_baseline_log1p"],
                    object_fallback=moved["object_fallback"],
                )
            )
    state = {key: value.detach().cpu() for key, value in adapters.state_dict().items()}
    return {
        "format": V8_TRAINING_CHECKPOINT_FORMAT,
        "adapter_format": V8_ADAPTER_FORMAT,
        "schema_version": V8_SCHEMA_VERSION,
        "config": config.to_dict(),
        "state_dict": state,
        "training_contract": {
            "factual_outputs_frozen": True,
            "shared_core_trainable": False,
            "loss": V8_LOSS_CONTRACT,
            "success_loss": "unweighted_binary_cross_entropy",
            "regress_loss": "unweighted_binary_cross_entropy",
            "recovery_loss": "unweighted_binary_cross_entropy_on_true_regress_rows_only",
            "failure_probability": "one_minus_success_probability",
            "unconditional_recovery_probability": (
                "p_regress_times_p_recovery_given_regress"
            ),
            "duration": (
                "outer_training_event_body_median_plus_0.375_times_frozen_residual"
            ),
            "duration_trainable": False,
            "object_mode": V8_OBJECT_MODE,
            "object_trainable": False,
            "optimizer_parameter_scope": "v8_adapter_parameters_exactly",
            "optimization_claim": (
                "fixed_order_AdamW_adapter_fit_not_claimed_converged_or_optimal"
            ),
        },
        "duration_laplace_scale_contract": provenance[
            "duration_laplace_scale_contract"
        ],
        "provenance": provenance,
        "strict_oof_base_exclusion_eligible": (
            provenance["base_target_outer_fold_exclusion_status"] == "proven"
        ),
        "frozen_input_sha256_by_batch": frozen_record_hashes,
        "frozen_input_aggregate_sha256": _aggregate_frozen_record_sha256(
            frozen_record_hashes
        ),
        "support": support,
        "optimizer": {
            "name": "AdamW",
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "initialization": "zero_weights_outer_training_prevalence_biases",
            "random_seed_used_for_adapter_initialization": None,
            "record_order": [
                str(record.get("logical_group_key", "")) for record in records
            ],
            "record_order_sha256": hashlib.sha256(
                "\n".join(
                    str(record.get("logical_group_key", "")) for record in records
                ).encode("utf-8")
            ).hexdigest(),
            "loss_trace": [row["loss"] for row in step_reports],
            "convergence_status": "not_assessed_fail_closed",
        },
        "steps": len(step_reports),
        "last_step": step_reports[-1],
        "all_steps_factual_inputs_bit_exact": all(
            row["factual_input_sha256_before"]
            == row["factual_input_sha256_after"]
            for row in step_reports
        ),
        "adapter_state_sha256": module_state_sha256(adapters),
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
    }


def train_v8_payload_lbfgs(
    payload: Mapping[str, Any],
    *,
    max_iter: int = 100,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, Any]:
    """Fit the three independent convex logistic heads to convergence.

    The factual transition tensors remain detached and immutable.  Each head
    owns a separate full-batch LBFGS optimizer, so success, regression and
    conditional recovery cannot exchange gradients or optimizer state.  No
    heldout row, calibration label, shared trunk or factual parameter enters
    this fit.
    """

    if max_iter <= 0 or tolerance_grad <= 0.0 or tolerance_change <= 0.0:
        raise ValueError("LBFGS convergence parameters must be positive")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config, records, provenance = validate_v8_training_payload(payload)
    support = _aggregate_binary_support(records)
    frozen_record_hashes = [_frozen_record_sha256(record) for record in records]
    frozen_before = _aggregate_frozen_record_sha256(frozen_record_hashes)
    adapters = V8DetachedStructuredAdapters(config).to(device)
    adapters.initialize_probability_biases(
        success_prevalence=_prevalence(support, "success"),
        regress_prevalence=_prevalence(support, "regress"),
        recovery_given_regress_prevalence=_prevalence(
            support, "recovery_given_regress"
        ),
    )

    transitions: list[torch.Tensor] = []
    success_labels: list[torch.Tensor] = []
    success_masks: list[torch.Tensor] = []
    regress_labels: list[torch.Tensor] = []
    regress_masks: list[torch.Tensor] = []
    recovery_labels: list[torch.Tensor] = []
    recovery_masks: list[torch.Tensor] = []
    for record in records:
        transition = record["factual_outputs"]["transition"].detach().to(
            device=device, dtype=torch.float32
        )
        batch = record["batch"]
        terminal = batch["terminal_mask"].bool().to(device)
        structured = batch["structured_mask"].bool().to(device)
        regress = batch["trajectory_regress"].bool().to(device)
        transitions.append(transition)
        success_labels.append(batch["success"].float().to(device))
        success_masks.append(terminal)
        regress_labels.append(regress.float())
        regress_masks.append(structured)
        recovery_labels.append(batch["trajectory_recovery"].float().to(device))
        recovery_masks.append(structured & regress)
    feature = torch.cat(transitions, dim=0).detach()
    head_data = {
        "success": (
            adapters.success_head,
            torch.cat(success_labels),
            torch.cat(success_masks),
        ),
        "regress": (
            adapters.regress_head,
            torch.cat(regress_labels),
            torch.cat(regress_masks),
        ),
        "recovery_given_regress": (
            adapters.recovery_given_regress_head,
            torch.cat(recovery_labels),
            torch.cat(recovery_masks),
        ),
    }
    reports: dict[str, dict[str, Any]] = {}
    adapters.train()
    for name, (head, label, mask) in head_data.items():
        selected_feature = feature[mask]
        selected_label = label[mask]
        if len(selected_label) != support[f"{name}_support"]:
            raise RuntimeError(f"{name} LBFGS support differs from signed payload")
        initial_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                head(selected_feature).squeeze(-1), selected_label
            )
            .detach()
            .cpu()
        )
        optimizer = torch.optim.LBFGS(
            head.parameters(),
            lr=1.0,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=min(100, max_iter),
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            logits = head(selected_feature).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, selected_label
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name} LBFGS loss became non-finite")
            loss.backward()
            return loss

        optimizer.step(closure)
        optimizer.zero_grad(set_to_none=True)
        final_tensor = torch.nn.functional.binary_cross_entropy_with_logits(
            head(selected_feature).squeeze(-1), selected_label
        )
        gradients = torch.autograd.grad(final_tensor, tuple(head.parameters()))
        maximum_gradient = max(
            float(gradient.detach().abs().max().cpu()) for gradient in gradients
        )
        final_loss = float(final_tensor.detach().cpu())
        if (
            not math.isfinite(final_loss)
            or final_loss > initial_loss + max(tolerance_change, 1e-10)
            or any(not bool(torch.isfinite(parameter).all()) for parameter in head.parameters())
        ):
            raise RuntimeError(f"{name} LBFGS fit failed the finite improvement contract")
        state_value = optimizer.state[next(iter(head.parameters()))]
        reports[name] = {
            "support": int(len(selected_label)),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": initial_loss - final_loss,
            "maximum_absolute_gradient": maximum_gradient,
            "closure_calls": closure_calls,
            "optimizer_iterations": int(state_value.get("n_iter", 0)),
            "optimizer_function_evaluations": int(state_value.get("func_evals", 0)),
            "convergence_status": (
                "gradient_tolerance_met"
                if maximum_gradient <= tolerance_grad
                else "finite_improved_optimizer_stopped_before_gradient_tolerance"
            ),
        }

    frozen_after_hashes = [_frozen_record_sha256(record) for record in records]
    frozen_after = _aggregate_frozen_record_sha256(frozen_after_hashes)
    if frozen_after != frozen_before:
        raise RuntimeError("factual inputs changed during LBFGS adapter training")
    state = {key: value.detach().cpu() for key, value in adapters.state_dict().items()}
    total_initial = sum(report["initial_loss"] for report in reports.values())
    total_final = sum(report["final_loss"] for report in reports.values())
    return {
        "format": V8_TRAINING_CHECKPOINT_FORMAT,
        "adapter_format": V8_ADAPTER_FORMAT,
        "schema_version": V8_SCHEMA_VERSION,
        "config": config.to_dict(),
        "state_dict": state,
        "training_contract": {
            "factual_outputs_frozen": True,
            "shared_core_trainable": False,
            "loss": V8_LOSS_CONTRACT,
            "success_loss": "unweighted_binary_cross_entropy",
            "regress_loss": "unweighted_binary_cross_entropy",
            "recovery_loss": (
                "unweighted_binary_cross_entropy_on_true_regress_rows_only"
            ),
            "failure_probability": "one_minus_success_probability",
            "unconditional_recovery_probability": (
                "p_regress_times_p_recovery_given_regress"
            ),
            "duration": (
                "outer_training_event_body_median_plus_0.375_times_frozen_residual"
            ),
            "duration_trainable": False,
            "object_mode": V8_OBJECT_MODE,
            "object_trainable": False,
            "optimizer_parameter_scope": "one_probability_head_at_a_time_exactly",
            "optimization_claim": (
                "independent_full_batch_convex_lbfgs_fit_with_reported_gradient_status"
            ),
        },
        "duration_laplace_scale_contract": provenance[
            "duration_laplace_scale_contract"
        ],
        "provenance": provenance,
        "strict_oof_base_exclusion_eligible": (
            provenance["base_target_outer_fold_exclusion_status"] == "proven"
        ),
        "frozen_input_sha256_by_batch": frozen_record_hashes,
        "frozen_input_aggregate_sha256": frozen_before,
        "support": support,
        "optimizer": {
            "name": "independent_full_batch_LBFGS",
            "max_iter_per_head": int(max_iter),
            "tolerance_grad": float(tolerance_grad),
            "tolerance_change": float(tolerance_change),
            "line_search_fn": "strong_wolfe",
            "head_reports": reports,
            "total_initial_unweighted_bce": total_initial,
            "total_final_unweighted_bce": total_final,
            "all_heads_finite_improved": all(
                report["final_loss"] <= report["initial_loss"]
                for report in reports.values()
            ),
        },
        "steps": int(sum(report["optimizer_iterations"] for report in reports.values())),
        "last_step": {
            "loss": total_final,
            "per_head": reports,
            "adapter_parameters_changed": any(
                report["loss_reduction"] > 0.0 for report in reports.values()
            ),
        },
        "all_steps_factual_inputs_bit_exact": True,
        "adapter_state_sha256": module_state_sha256(adapters),
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
    }


class _SmokeFactual(torch.nn.Module):
    def __init__(self, input_dim: int, transition_dim: int) -> None:
        super().__init__()
        self.transition = torch.nn.Linear(input_dim, transition_dim)
        self.duration = torch.nn.Linear(transition_dim, 1)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        transition = torch.tanh(self.transition(value))
        return {
            "transition": transition,
            "duration_selected_log_mean": self.duration(transition).squeeze(-1),
        }


def cpu_one_step_smoke(seed: int = 20260827) -> dict[str, Any]:
    """Run one synthetic schema5 step without filesystem or CUDA access."""

    torch.manual_seed(seed)
    count, input_dim, transition_dim, object_dim = 8, 4, 6, 3
    factual_module = _SmokeFactual(input_dim, transition_dim)
    factual_module.eval()
    factual_outputs = factual_module(torch.randn(count, input_dim))
    batch = {
        "terminal_mask": torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool),
        "structured_mask": torch.ones(count, dtype=torch.bool),
        "dense_mask": torch.ones(count, dtype=torch.bool),
        "duration": torch.tensor([4, 5, 8, 3, 9, 6, 4, 7], dtype=torch.float32),
        "duration_observed": torch.tensor([1, 1, 0, 1, 0, 1, 1, 0], dtype=torch.bool),
        "success": torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32),
        "trajectory_regress": torch.tensor([0, 1, 1, 0, 1, 0, 1, 0], dtype=torch.bool),
        "trajectory_recovery": torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.bool),
        "object_delta": torch.zeros(count, object_dim),
    }
    baseline = torch.full((count,), float(torch.log1p(torch.tensor(5.0))))
    fallback = torch.zeros(object_dim)
    adapters = V8DetachedStructuredAdapters(
        V8StructuredAdapterConfig(transition_dim=transition_dim)
    )
    adapters.initialize_probability_biases(
        success_prevalence=2 / 6,
        regress_prevalence=4 / 8,
        recovery_given_regress_prevalence=2 / 4,
    )
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=1e-2)
    report = train_v8_adapter_one_step(
        adapters,
        optimizer,
        factual_outputs,
        batch,
        duration_baseline_log1p=baseline,
        object_fallback=fallback,
        frozen_factual_module=factual_module,
    )
    if not report["adapter_parameters_changed"]:
        raise RuntimeError("CPU smoke did not update adapter parameters")
    return {
        "status": "passed",
        "device": "cpu",
        "cuda_used": False,
        **report,
    }


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--materialization-manifest", type=Path)
    parser.add_argument("--outer-fold-id", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer-mode", choices=("lbfgs", "adamw"), default="lbfgs"
    )
    parser.add_argument("--lbfgs-max-iter", type=int, default=100)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-7)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        if (
            args.input is not None
            or args.materialization_manifest is not None
            or args.outer_fold_id is not None
            or args.output is not None
            or args.device != "cpu"
        ):
            raise ValueError("--smoke is CPU-only and does not accept input/output")
        print(json.dumps(cpu_one_step_smoke(), sort_keys=True))
        return
    if (
        args.input is None
        or args.materialization_manifest is None
        or args.outer_fold_id is None
        or args.output is None
    ):
        raise ValueError(
            "training requires --input/--materialization-manifest/--outer-fold-id/--output"
        )
    payload, artifact_authentication = load_authenticated_training_payload(
        input_path=args.input,
        materialization_manifest_path=args.materialization_manifest,
        outer_fold_id=args.outer_fold_id,
    )
    if args.optimizer_mode == "lbfgs":
        checkpoint = train_v8_payload_lbfgs(
            payload,
            max_iter=args.lbfgs_max_iter,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            device=args.device,
        )
    else:
        checkpoint = train_v8_payload(
            payload,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )
    checkpoint["input_artifact_authentication"] = artifact_authentication
    _atomic_torch_save(args.output, checkpoint)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "steps": checkpoint["steps"],
                "fresh_confirmation_data_or_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "V8_TRAINING_CHECKPOINT_FORMAT",
    "V8_TRAINING_INPUT_FORMAT",
    "cpu_one_step_smoke",
    "train_v8_payload",
    "train_v8_payload_lbfgs",
    "load_authenticated_training_payload",
    "structured_payload_sha256",
    "validate_v8_training_payload",
]
