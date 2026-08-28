#!/usr/bin/env python3
"""Authenticated R5 OOF evaluation for the duration hierarchy v2 contract.

Only a signed R3 materialization manifest and its ten authenticated ``.pt``
artifacts are consumed.  For every owner fold, the training artifact is fully
authenticated and its hierarchy contract is signed before the holdout artifact
is deserialized.  No source HDF5, actor, model training, or Fresh path exists in
this interface.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from openvla_etsf_duration_hierarchy import (
    DURATION_HIERARCHY_PROTOCOL_V2,
    MINIMUM_APPLIED_SOURCE_SUPPORT,
    apply_duration_hierarchy,
    canonical_sha256,
    fit_duration_hierarchy,
    serialize_duration_hierarchy,
    validate_duration_hierarchy_contract,
)
from openvla_etsf_v8_structured_adapters import frozen_tensor_mapping_sha256
from train_openvla_etsf_v8_structured_adapters import structured_payload_sha256


FORMAT = "etsf_r5_duration_hierarchy_oof_evaluation_v1"
MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
TRAINING_FORMAT = "etsf_v8_detached_adapter_training_input_v1"
HOLDOUT_FORMAT = "etsf_v8_detached_adapter_holdout_input_v1"
FOLD_COUNT = 5
RESIDUAL_MULTIPLIER = 0.375
MINIMUM_LAPLACE_SCALE = 1e-4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
IMPLEMENTATION_FILENAMES = (
    "evaluate_openvla_etsf_duration_hierarchy_oof.py",
    "openvla_etsf_duration_hierarchy.py",
    "openvla_etsf_v8_structured_adapters.py",
    "train_openvla_etsf_v8_structured_adapters.py",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_group_list_sha256(groups: Sequence[str]) -> str:
    normalized = list(map(str, groups))
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise RuntimeError("materialization group lists must be sorted and unique")
    return canonical_sha256({"logical_groups": normalized})


def _implementation_file_contract() -> dict[str, str]:
    scripts_root = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for filename in IMPLEMENTATION_FILENAMES:
        path = scripts_root / filename
        if not path.is_file():
            raise RuntimeError(f"R5 implementation file is missing: {filename}")
        result[filename] = sha256_path(path)
    return result


def _reject_fresh_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _tensor_numpy(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    if not torch.is_tensor(value):
        raise RuntimeError(f"{name} must be a tensor")
    result = value.detach().cpu().numpy()
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise RuntimeError(f"{name} must be a one-dimensional aligned tensor")
    return result


def _audit_factual_state(payload: Mapping[str, Any], *, fold_id: int, role: str) -> str:
    audit = payload.get("materialization_audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError(f"fold {fold_id} {role} factual audit is missing")
    before = audit.get("factual_state_sha256_before")
    after = audit.get("factual_state_sha256_after")
    if (
        audit.get("factual_state_bit_exact") is not True
        or not isinstance(before, str)
        or len(before) != 64
        or before != after
    ):
        raise RuntimeError(f"fold {fold_id} {role} factual state is not bit-exact")
    return before


def _validate_payload_header(
    payload: Mapping[str, Any],
    *,
    fold_id: int,
    role: str,
    fold_row: Mapping[str, Any],
) -> None:
    expected_format = TRAINING_FORMAT if role == "train" else HOLDOUT_FORMAT
    expected_split = "outer_training" if role == "train" else "outer_holdout"
    if (
        payload.get("format") != expected_format
        or int(payload.get("schema_version", -1)) != 5
        or payload.get("payload_sha256") != fold_row.get(f"{role}_payload_sha256")
        or payload.get("payload_sha256") != structured_payload_sha256(payload)
    ):
        raise RuntimeError(f"fold {fold_id} {role} payload authentication failed")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or int(
        provenance.get("outer_fold_id", -1)
    ) != fold_id:
        raise RuntimeError(f"fold {fold_id} {role} owner provenance mismatch")
    if (
        provenance.get("target_outer_fold_labels_used") is not False
        or provenance.get("factual_outputs_frozen") is not True
    ):
        raise RuntimeError(f"fold {fold_id} {role} crossed its OOF boundary")
    if role == "holdout" and (
        provenance.get("split_role") != "outer_holdout_evaluation_only"
        or provenance.get("holdout_labels_used_for_duration_or_object_fit") is not False
        or provenance.get("holdout_labels_present_only_in_separate_artifact") is not True
    ):
        raise RuntimeError(f"fold {fold_id} holdout fit provenance changed")
    records = payload.get("batches")
    expected_groups = (
        fold_row["training_groups"]
        if role == "train"
        else fold_row["oof_holdout_groups"]
    )
    if not isinstance(records, list) or [
        str(record.get("logical_group_key", ""))
        for record in records
        if isinstance(record, Mapping)
    ] != list(expected_groups):
        raise RuntimeError(f"fold {fold_id} {role} record ownership changed")
    if provenance.get("outer_training_groups") != fold_row["training_groups"] or (
        provenance.get("outer_training_groups_sha256")
        != fold_row["training_groups_sha256"]
    ):
        raise RuntimeError(f"fold {fold_id} {role} outer-training provenance changed")
    if role == "holdout" and (
        provenance.get("oof_holdout_groups") != fold_row["oof_holdout_groups"]
        or provenance.get("oof_holdout_groups_sha256")
        != fold_row["oof_holdout_groups_sha256"]
    ):
        raise RuntimeError(f"fold {fold_id} holdout group provenance changed")
    expected_record_role = expected_split
    if any(
        not isinstance(record, Mapping)
        or record.get("split_role") != expected_record_role
        or int(record.get("outer_fold_id", -1)) != fold_id
        for record in records
    ):
        raise RuntimeError(f"fold {fold_id} {role} record role changed")


def _records_to_arrays(
    payload: Mapping[str, Any], *, fold_id: int, role: str
) -> tuple[dict[str, np.ndarray], str]:
    state_sha = _audit_factual_state(payload, fold_id=fold_id, role=role)
    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "duration",
            "duration_observed",
            "dense_mask",
            "current_event_id",
            "clock_event_id",
            "body_id",
            "frozen_log_location",
            "logical_group",
            "row_index",
        )
    }
    for record in payload["batches"]:
        batch = record.get("batch")
        factual = record.get("factual_outputs")
        if not isinstance(batch, Mapping) or not isinstance(factual, Mapping):
            raise RuntimeError(f"fold {fold_id} {role} record lacks batch/factual outputs")
        if "current_event_id" not in batch:
            if "clock_event_id" in batch:
                raise RuntimeError(
                    "clock_event_id cannot substitute for observed current_event_id"
                )
            raise RuntimeError("materialized duration row lacks current_event_id")
        if (
            record.get("factual_outputs_require_grad") is not False
            or any(
                torch.is_tensor(value) and value.requires_grad
                for value in factual.values()
            )
            or record.get("factual_outputs_sha256")
            != frozen_tensor_mapping_sha256(factual)
        ):
            raise RuntimeError(f"fold {fold_id} {role} factual tensor hash changed")
        duration = _tensor_numpy(batch.get("duration"), name="duration")
        count = len(duration)
        group = str(record.get("logical_group_key", ""))
        if not group:
            raise RuntimeError("duration record lacks logical group")
        parts["duration"].append(duration.astype(np.float64))
        for key in (
            "duration_observed",
            "dense_mask",
            "current_event_id",
            "clock_event_id",
            "body_id",
        ):
            parts[key].append(
                _tensor_numpy(batch.get(key), name=key, length=count)
            )
        parts["frozen_log_location"].append(
            _tensor_numpy(
                factual.get("duration_selected_log_mean"),
                name="duration_selected_log_mean",
                length=count,
            ).astype(np.float64)
        )
        parts["logical_group"].append(np.repeat(group, count))
        parts["row_index"].append(np.arange(count, dtype=np.int64))
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    if (
        not np.isfinite(arrays["duration"]).all()
        or np.any(arrays["duration"] < 0.0)
        or not np.isfinite(arrays["frozen_log_location"]).all()
    ):
        raise RuntimeError("duration/frozen location contains invalid values")
    for key in ("duration_observed", "dense_mask"):
        values = arrays[key].astype(np.float64)
        if not np.isfinite(values).all() or np.any((values != 0) & (values != 1)):
            raise RuntimeError(f"{key} must be binary")
        arrays[key] = values.astype(bool)
    for key in ("current_event_id", "clock_event_id", "body_id"):
        values = arrays[key]
        converted = values.astype(np.int64)
        if not np.array_equal(values, converted) or np.any(converted < 0):
            raise RuntimeError(f"{key} must contain non-negative ids")
        arrays[key] = converted
    return arrays, state_sha


def _authenticate_artifact(
    *,
    manifest_root: Path,
    fold_row: Mapping[str, Any],
    fold_id: int,
    role: str,
    trace: list[dict[str, Any]],
    signed_duration_contract: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Path]:
    if role == "holdout":
        if signed_duration_contract is None:
            raise RuntimeError("holdout cannot be opened before duration contract")
        validate_duration_hierarchy_contract(signed_duration_contract)
    artifact = Path(str(fold_row.get(f"{role}_artifact", ""))).resolve()
    try:
        artifact.relative_to(manifest_root)
    except ValueError as error:
        raise RuntimeError("duration artifact escaped its signed bundle") from error
    expected = (manifest_root / f"fold_{fold_id}_{role}.pt").resolve()
    if artifact != expected or not artifact.is_file():
        raise RuntimeError(f"fold {fold_id} {role} artifact path changed")
    artifact_bytes = artifact.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != fold_row.get(
        f"{role}_artifact_sha256"
    ):
        raise RuntimeError(f"fold {fold_id} {role} artifact SHA mismatch")
    trace.append(
        {
            "sequence": len(trace),
            "fold": fold_id,
            "event": f"{role}_artifact_hash_authenticated",
            "path": str(artifact),
        }
    )
    # Deserialize the exact authenticated bytes, closing the hash/load race.
    payload = torch.load(io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True)
    trace.append(
        {
            "sequence": len(trace),
            "fold": fold_id,
            "event": f"{role}_payload_deserialized",
            "path": str(artifact),
        }
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"fold {fold_id} {role} artifact must contain a mapping")
    _validate_payload_header(payload, fold_id=fold_id, role=role, fold_row=fold_row)
    trace.append(
        {
            "sequence": len(trace),
            "fold": fold_id,
            "event": f"{role}_payload_authenticated",
            "payload_sha256": payload["payload_sha256"],
        }
    )
    return payload, artifact


def _laplace_scale(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1 or not len(residual) or not np.isfinite(residual).all():
        raise RuntimeError("Laplace scale requires finite outer-training residuals")
    return max(
        float(np.median(np.abs(residual))) / math.log(2.0),
        MINIMUM_LAPLACE_SCALE,
    )


def _group_mean(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(sorted(set(map(str, groups.tolist()))), dtype=str)
    means = np.asarray(
        [np.mean(values[groups == name]) for name in names], dtype=np.float64
    )
    return names, means


def _cluster_delta(
    delta: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    names, means = _group_mean(delta, groups)
    if samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = generator.integers(0, len(means), size=len(means))
        draws[index] = float(np.mean(means[selected]))
    interval = np.quantile(draws, [0.025, 0.975])
    observed = float(np.mean(means))
    return {
        "model_minus_baseline": observed,
        "equal_group_bootstrap_95_ci": [float(interval[0]), float(interval[1])],
        "strict_skill": bool(float(interval[1]) < 0.0),
        "logical_groups": len(names),
    }


def evaluate_duration_hierarchy_oof(
    materialization_manifest: Path,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    manifest_path = _reject_fresh_path(
        materialization_manifest, role="R3 materialization"
    )
    manifest = _load_json(manifest_path)
    unsigned = dict(manifest)
    recorded_manifest_sha = unsigned.pop("materialization_sha256", None)
    if recorded_manifest_sha != canonical_sha256(unsigned):
        raise RuntimeError("R3 materialization signature mismatch")
    if (
        manifest.get("format") != MATERIALIZATION_FORMAT
        or manifest.get("status") != "complete_development_only"
        or manifest.get("timing_scope")
        != "adaptive_development_only_designed_after_v7_collection_started"
        or manifest.get("prospective_claim_for_v8") is not False
        or manifest.get("fresh_confirmation_data_or_labels_read") is not False
        or manifest.get("authorization_guard_changed") is not False
    ):
        raise RuntimeError("R3 materialization changed adaptive/no-Fresh semantics")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or [row.get("outer_fold_id") for row in folds] != list(
        range(FOLD_COUNT)
    ):
        raise RuntimeError("R3 materialization must contain five ordered folds")
    development_groups = manifest.get("development_groups")
    if (
        not isinstance(development_groups, list)
        or logical_group_list_sha256(development_groups)
        != manifest.get("development_groups_sha256")
    ):
        raise RuntimeError("R3 development group registry changed")

    trace: list[dict[str, Any]] = []
    row_parts: dict[str, list[np.ndarray]] = {}
    contracts: dict[str, Any] = {}
    scale_contracts: dict[str, Any] = {}
    factual_state_shas: set[str] = set()
    root = manifest_path.parent.resolve()
    holdout_owner: dict[str, int] = {}
    for fold_id, row in enumerate(folds):
        if not isinstance(row, Mapping):
            raise RuntimeError("R3 fold row must be a mapping")
        training_groups = list(map(str, row.get("training_groups", [])))
        holdout_groups = list(map(str, row.get("oof_holdout_groups", [])))
        if (
            logical_group_list_sha256(training_groups)
            != row.get("training_groups_sha256")
            or logical_group_list_sha256(holdout_groups)
            != row.get("oof_holdout_groups_sha256")
            or set(training_groups) & set(holdout_groups)
            or set(training_groups) | set(holdout_groups)
            != set(map(str, development_groups))
        ):
            raise RuntimeError(f"fold {fold_id} ownership contract changed")
        for group in holdout_groups:
            if group in holdout_owner:
                raise RuntimeError("logical group has multiple OOF owners")
            holdout_owner[group] = fold_id

        train_payload, _ = _authenticate_artifact(
            manifest_root=root,
            fold_row=row,
            fold_id=fold_id,
            role="train",
            trace=trace,
        )
        train, train_state_sha = _records_to_arrays(
            train_payload, fold_id=fold_id, role="train"
        )
        factual_state_shas.add(train_state_sha)
        train_selected = train["dense_mask"] & train["duration_observed"]
        contract = fit_duration_hierarchy(
            duration=train["duration"],
            duration_observed=train_selected,
            current_event_id=train["current_event_id"],
            body_id=train["body_id"],
            logical_group=train["logical_group"],
            split_role=np.repeat("outer_training", len(train["duration"])),
            owner_fold_id=fold_id,
        )
        serialized_contract = serialize_duration_hierarchy(contract)
        contracts[str(fold_id)] = contract
        trace.append(
            {
                "sequence": len(trace),
                "fold": fold_id,
                "event": "duration_v2_contract_signed",
                "contract_sha256": contract["contract_sha256"],
                "serialized_sha256": hashlib.sha256(
                    serialized_contract.encode("utf-8")
                ).hexdigest(),
            }
        )

        train_applied = apply_duration_hierarchy(
            contract,
            current_event_id=train["current_event_id"],
            body_id=train["body_id"],
            expected_training_logical_groups_sha256=contract[
                "outer_training_logical_groups_sha256"
            ],
        )
        train_baseline = train_applied["baseline_log1p_duration"]
        train_model = train_baseline + RESIDUAL_MULTIPLIER * (
            train["frozen_log_location"] - train_baseline
        )
        train_target = np.log1p(train["duration"])
        model_scale = _laplace_scale(
            (train_target - train_model)[train_selected]
        )
        baseline_scale = _laplace_scale(
            (train_target - train_baseline)[train_selected]
        )
        scale_contract = {
            "format": "etsf_r5_outer_training_duration_laplace_scale_v1",
            "owner_fold_id": fold_id,
            "fit_scope": "outer_training_dense_and_observed_only",
            "holdout_labels_used": False,
            "estimator": "median_absolute_fixed_location_residual_divided_by_log_2",
            "minimum_scale": MINIMUM_LAPLACE_SCALE,
            "support": int(train_selected.sum()),
            "model_scale": model_scale,
            "baseline_scale": baseline_scale,
            "duration_hierarchy_contract_sha256": contract["contract_sha256"],
        }
        scale_contract["contract_sha256"] = canonical_sha256(scale_contract)
        scale_contracts[str(fold_id)] = scale_contract

        # The signed hierarchy and both scales now exist.  Holdout loading is
        # forbidden above this point and begins only here.
        validate_duration_hierarchy_contract(contract)
        holdout_payload, _ = _authenticate_artifact(
            manifest_root=root,
            fold_row=row,
            fold_id=fold_id,
            role="holdout",
            trace=trace,
            signed_duration_contract=contract,
        )
        holdout, holdout_state_sha = _records_to_arrays(
            holdout_payload, fold_id=fold_id, role="holdout"
        )
        if holdout_state_sha != train_state_sha:
            raise RuntimeError(f"fold {fold_id} train/holdout factual state differs")
        factual_state_shas.add(holdout_state_sha)
        applied = apply_duration_hierarchy(
            contract,
            current_event_id=holdout["current_event_id"],
            body_id=holdout["body_id"],
            expected_training_logical_groups_sha256=contract[
                "outer_training_logical_groups_sha256"
            ],
        )
        baseline = applied["baseline_log1p_duration"]
        model = baseline + RESIDUAL_MULTIPLIER * (
            holdout["frozen_log_location"] - baseline
        )
        count = len(baseline)
        fold_arrays: dict[str, np.ndarray] = {
            **holdout,
            "owner_fold_id": np.full(count, fold_id, dtype=np.int64),
            "target_log1p_duration": np.log1p(holdout["duration"]),
            "baseline_log_location": baseline,
            "model_log_location": model,
            "baseline_log_scale": np.full(count, math.log(baseline_scale)),
            "model_log_scale": np.full(count, math.log(model_scale)),
            "source_kind": applied["source_kind"],
            "source_key": applied["source_key"],
            "source_support": applied["source_support"],
            "source_logical_group_support": applied[
                "source_logical_group_support"
            ],
            "current_clock_divergence": (
                holdout["current_event_id"] != holdout["clock_event_id"]
            ),
        }
        for key, value in fold_arrays.items():
            row_parts.setdefault(key, []).append(np.asarray(value))
    if set(holdout_owner) != set(map(str, development_groups)):
        raise RuntimeError("five holdouts do not cover the development registry")
    if len(factual_state_shas) != 1:
        raise RuntimeError("R3 folds do not share one frozen factual state")
    rows = {key: np.concatenate(value, axis=0) for key, value in row_parts.items()}
    selected = rows["dense_mask"].astype(bool) & rows["duration_observed"].astype(bool)
    if not selected.any():
        raise RuntimeError("R5 duration evaluation has no observed holdout rows")
    target = rows["target_log1p_duration"][selected]
    model = rows["model_log_location"][selected]
    baseline = rows["baseline_log_location"][selected]
    groups = rows["logical_group"][selected].astype(str)
    owner = rows["owner_fold_id"][selected].astype(np.int64)
    model_scale = np.exp(rows["model_log_scale"][selected])
    baseline_scale = np.exp(rows["baseline_log_scale"][selected])
    mae_delta = np.abs(target - model) - np.abs(target - baseline)
    nll_delta = (
        np.abs(target - model) / model_scale
        + np.log(2.0 * model_scale)
        - np.abs(target - baseline) / baseline_scale
        - np.log(2.0 * baseline_scale)
    )
    mae = _cluster_delta(
        mae_delta, groups, samples=bootstrap_samples, seed=bootstrap_seed
    )
    nll = _cluster_delta(
        nll_delta, groups, samples=bootstrap_samples, seed=bootstrap_seed + 1
    )
    fold_wins: dict[str, Any] = {}
    for fold_id in range(FOLD_COUNT):
        in_fold = owner == fold_id
        _, fold_mae = _group_mean(mae_delta[in_fold], groups[in_fold])
        _, fold_nll = _group_mean(nll_delta[in_fold], groups[in_fold])
        fold_wins[str(fold_id)] = {
            "mae_noninferior": bool(np.mean(fold_mae) <= 0.0),
            "nll_noninferior": bool(np.mean(fold_nll) <= 0.0),
            "observed_logical_groups": len(fold_mae),
        }
    minimum_support = int(rows["source_support"][selected].min())
    mae_fold_wins = sum(value["mae_noninferior"] for value in fold_wins.values())
    nll_fold_wins = sum(value["nll_noninferior"] for value in fold_wins.values())
    passed = bool(
        minimum_support >= MINIMUM_APPLIED_SOURCE_SUPPORT
        and mae["strict_skill"]
        and nll["strict_skill"]
        and mae_fold_wins >= 4
        and nll_fold_wins >= 4
    )
    summary: dict[str, Any] = {
        "format": FORMAT,
        "status": "passed" if passed else "fail_closed",
        "passed": passed,
        "evidence_scope": "R3_adaptive_development_OOF_only_not_prospective",
        "source_materialization": {
            "path": str(manifest_path),
            "file_sha256": sha256_path(manifest_path),
            "materialization_sha256": recorded_manifest_sha,
            "ten_artifacts_authenticated": True,
            "source_hdf5_read": False,
        },
        "protocol": DURATION_HIERARCHY_PROTOCOL_V2,
        "implementation_files": _implementation_file_contract(),
        "duration_residual_multiplier": RESIDUAL_MULTIPLIER,
        "contracts": contracts,
        "outer_training_scale_contracts": scale_contracts,
        "factual_state_sha256": next(iter(factual_state_shas)),
        "observed_rows": int(selected.sum()),
        "observed_logical_groups": len(set(groups.tolist())),
        "right_censored_or_nondense_rows_excluded": int((~selected).sum()),
        "minimum_applied_source_support": minimum_support,
        "minimum_applied_source_support_gate": bool(
            minimum_support >= MINIMUM_APPLIED_SOURCE_SUPPORT
        ),
        "mae_log1p_model_minus_hierarchical_baseline": mae,
        "laplace_nll_model_minus_hierarchical_baseline": nll,
        "fold_point_estimate_noninferiority": fold_wins,
        "mae_fold_wins": mae_fold_wins,
        "nll_fold_wins": nll_fold_wins,
        "current_clock_divergence_rows": int(
            rows["current_clock_divergence"].sum()
        ),
        "current_event_source": "authenticated_current_event_id_never_clock_proxy",
        "read_trace": trace,
        "bootstrap": {
            "resampling_unit": "equal_weight_logical_group",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "fresh50_confirmation_authorized": False,
        "selector_authorized": False,
        "prospective_claim_allowed": False,
    }
    return {"summary": summary, "arrays": rows}


def _string_safe_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object:
            array = array.astype(str)
        result[key] = array
    return result


def write_duration_hierarchy_evaluation(
    output_dir: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    output_dir = _reject_fresh_path(output_dir, role="R5 duration output")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        arrays_path = staging / "duration_hierarchy_rows.npz"
        with arrays_path.open("wb") as handle:
            np.savez_compressed(handle, **_string_safe_arrays(value["arrays"]))
        final_arrays = output_dir / arrays_path.name
        summary = {
            **dict(value["summary"]),
            "row_arrays": {
                "path": str(final_arrays),
                "file_sha256": sha256_path(arrays_path),
                "keys": sorted(value["arrays"]),
                "rows": len(next(iter(value["arrays"].values()))),
                "alignment": "owner_fold_id_logical_group_row_index",
                "source_fields": ["source_kind", "source_key", "source_support"],
            },
        }
        summary["result_sha256"] = canonical_sha256(summary)
        result_path = staging / "duration_hierarchy_evaluation.json"
        result_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
        return summary
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = evaluate_duration_hierarchy_oof(
        args.materialization_manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result = write_duration_hierarchy_evaluation(args.output, value)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_sha256": result["result_sha256"],
                "fresh50_confirmation_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
