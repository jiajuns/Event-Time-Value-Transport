#!/usr/bin/env python3
"""Authenticated five-fold OOF bridge for the v8 array evaluator.

This bridge consumes only trained v8 adapter checkpoints and their owner-fold
holdout artifacts.  It validates file/state hashes, ownership, split roles,
frozen factual state and per-head probability semantics before constructing the
arrays required by :mod:`evaluate_openvla_etsf_v8_structured_heads_arrays`.

Current D250 evidence is intentionally labelled adaptive development-only.  No
path or API accepts Fresh confirmation data, and no result can authorise the v7
selector.  Missing physical object labels or outer-training duration scale
provenance fail closed instead of being reconstructed from holdout labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluate_openvla_etsf_v8_structured_heads_arrays import (
    evaluate_adaptive_development_structured_heads_arrays,
    validate_adaptive_development_result,
)
from openvla_etsf_v8_adaptive_development_protocol import (
    make_adaptive_development_contract,
    validate_adaptive_development_contract,
)
from openvla_etsf_v8_structured_adapters import (
    V8_DURATION_RESIDUAL_MULTIPLIER,
    V8_LOSS_CONTRACT,
    V8_OBJECT_MODE,
    V8DetachedStructuredAdapters,
    V8StructuredAdapterConfig,
    frozen_tensor_mapping_sha256,
    module_state_sha256,
    validate_factual_adapter_inputs,
    validate_schema5_adapter_batch,
)
from openvla_etsf_v8_structured_heads_protocol import (
    FOLD_COUNT,
    canonical_sha256,
)
from train_openvla_etsf_v8_structured_adapters import (
    V8_TRAINING_INPUT_FORMAT,
    V8_TRAINING_CHECKPOINT_FORMAT,
    structured_payload_sha256,
)


BRIDGE_FORMAT = "etsf_v8_authenticated_oof_evaluation_bridge_v1"
OUTPUT_FORMAT = "etsf_v8_authenticated_oof_evaluation_output_v1"
MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
HOLDOUT_FORMAT = "etsf_v8_detached_adapter_holdout_input_v1"
DURATION_SCALE_FORMAT = "etsf_v8_outer_training_duration_laplace_scale_v1"
MINIMUM_DURATION_SCALE = 1e-4
INPUT_CONTRACT = {
    "source_partition": "development_only",
    "fresh50_inputs_accepted": False,
    "fresh50_labels_read": False,
    "old100_overlap_declared": True,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _signed(value: Mapping[str, Any], key: str, *, name: str) -> str:
    unsigned = dict(value)
    digest = unsigned.pop(key, None)
    if not _is_sha256(digest) or digest != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} signature mismatch")
    return str(digest)


def _reject_confirmation_path(path: Path) -> Path:
    resolved = path.resolve()
    lowered = str(resolved).lower()
    if "fresh" in lowered or "confirmation" in lowered:
        raise RuntimeError("v8 OOF bridge rejects Fresh/confirmation paths")
    return resolved


def _load_torch_mapping(path: Path, *, role: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{role} artifact must be a mapping")
    return dict(value)


def _load_json_mapping(path: Path, *, role: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must be a JSON object")
    return value


def _validate_duration_scale_contract(
    value: Any, *, owner_fold_id: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "checkpoint lacks outer-train duration_laplace_scale_contract; "
            "prediction agent must publish model/baseline MAD scales"
        )
    result = dict(value)
    _signed(result, "contract_sha256", name="duration Laplace scale contract")
    expected_fixed = {
        "format": DURATION_SCALE_FORMAT,
        "owner_fold_id": owner_fold_id,
        "fit_scope": "outer_training_observed_only",
        "estimator": "median_absolute_deviation_divided_by_log_2",
        "censored_rows_used": False,
        "model_location": "event_body_median_plus_0.375_frozen_residual",
        "baseline_location": "outer_training_event_body_median",
        "minimum_scale": MINIMUM_DURATION_SCALE,
    }
    for key, expected in expected_fixed.items():
        if result.get(key) != expected:
            raise RuntimeError(f"duration scale contract field {key} changed")
    support = result.get("outer_training_observed_support")
    if not isinstance(support, int) or support <= 0:
        raise RuntimeError("duration scale contract lacks outer-training support")
    for name in ("model_log_scale", "baseline_log_scale"):
        number = float(result.get(name, float("nan")))
        if not math.isfinite(number) or math.exp(number) < MINIMUM_DURATION_SCALE:
            raise RuntimeError("duration scale contract contains an invalid scale")
    return result


def _validate_repair_contract(
    provenance: Mapping[str, Any], *, field: str, signature_field: str
) -> dict[str, Any]:
    value = provenance.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"checkpoint provenance lacks {field}")
    result = dict(value)
    recorded = result.pop(signature_field, None)
    if not _is_sha256(recorded) or recorded != canonical_sha256(result):
        raise RuntimeError(f"{field} signature mismatch")
    if provenance.get(signature_field) != recorded:
        raise RuntimeError(f"{field} provenance SHA mismatch")
    result[signature_field] = recorded
    return result


def _checkpoint_owner(path: Path) -> int:
    checkpoint = _load_torch_mapping(path, role="adapter checkpoint")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("adapter checkpoint lacks provenance")
    owner = provenance.get("outer_fold_id")
    if not isinstance(owner, int) or owner not in range(FOLD_COUNT):
        raise RuntimeError("adapter checkpoint owner fold is invalid")
    return owner


def _holdout_owner(path: Path) -> int:
    holdout = _load_torch_mapping(path, role="holdout")
    provenance = holdout.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("holdout artifact lacks provenance")
    owner = provenance.get("outer_fold_id")
    if not isinstance(owner, int) or owner not in range(FOLD_COUNT):
        raise RuntimeError("holdout owner fold is invalid")
    return owner


def make_bridge_bundle(
    *,
    checkpoint_paths: Sequence[Path],
    holdout_paths: Sequence[Path],
    materialization_manifest_path: Path,
    adaptive_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and sign all five checkpoint/holdout file identities."""

    validate_adaptive_development_contract(adaptive_contract)
    if len(checkpoint_paths) != FOLD_COUNT or len(holdout_paths) != FOLD_COUNT:
        raise RuntimeError("v8 OOF bridge requires exactly five checkpoints and holdouts")
    materialization_path = _reject_confirmation_path(materialization_manifest_path)
    checkpoint_by_owner: dict[int, Path] = {}
    for raw in checkpoint_paths:
        path = _reject_confirmation_path(raw)
        owner = _checkpoint_owner(path)
        if owner in checkpoint_by_owner:
            raise RuntimeError("duplicate adapter checkpoint owner fold")
        checkpoint_by_owner[owner] = path
    holdout_by_owner: dict[int, Path] = {}
    for raw in holdout_paths:
        path = _reject_confirmation_path(raw)
        owner = _holdout_owner(path)
        if owner in holdout_by_owner:
            raise RuntimeError("duplicate holdout owner fold")
        holdout_by_owner[owner] = path
    if set(checkpoint_by_owner) != set(range(FOLD_COUNT)) or set(
        holdout_by_owner
    ) != set(range(FOLD_COUNT)):
        raise RuntimeError("checkpoint/holdout owners must cover folds 0..4")
    value: dict[str, Any] = {
        "format": BRIDGE_FORMAT,
        "status": "authenticated_inputs_rehashed",
        "source_partition": "development_only",
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "adaptive_development_contract_sha256": adaptive_contract[
            "contract_sha256"
        ],
        "bridge_implementation_sha256": sha256_path(Path(__file__).resolve()),
        "materialization_manifest": str(materialization_path),
        "materialization_manifest_sha256": sha256_path(materialization_path),
        "folds": [
            {
                "owner_fold_id": owner,
                "checkpoint_role": "outer_training_only_adapter_checkpoint",
                "checkpoint": str(checkpoint_by_owner[owner]),
                "checkpoint_sha256": sha256_path(checkpoint_by_owner[owner]),
                "holdout_role": "owner_outer_holdout_evaluation_only",
                "holdout_artifact": str(holdout_by_owner[owner]),
                "holdout_artifact_sha256": sha256_path(holdout_by_owner[owner]),
            }
            for owner in range(FOLD_COUNT)
        ],
    }
    value["bridge_bundle_sha256"] = canonical_sha256(value)
    return value


def _validate_materialization(
    bundle: Mapping[str, Any], adaptive_contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    path = Path(str(bundle["materialization_manifest"])).resolve()
    if sha256_path(path) != bundle.get("materialization_manifest_sha256"):
        raise RuntimeError("materialization manifest file SHA mismatch")
    manifest = _load_json_mapping(path, role="materialization manifest")
    if manifest.get("format") != MATERIALIZATION_FORMAT or manifest.get(
        "status"
    ) != "complete_development_only":
        raise RuntimeError("materialization manifest status/format mismatch")
    _signed(manifest, "materialization_sha256", name="materialization manifest")
    if manifest.get("fresh_confirmation_data_or_labels_read") is not False:
        raise RuntimeError("materialization manifest accessed confirmation data")
    sources = adaptive_contract["source_sha256"]
    if manifest.get("base_checkpoint_sha256") != sources["base_checkpoint"]:
        raise RuntimeError("materialization base checkpoint differs from contract")
    if manifest.get("label_derivation_sha256") != sources["label_derivation"]:
        raise RuntimeError("materialization label derivation differs from contract")
    groups = manifest.get("development_groups")
    if not isinstance(groups, list) or groups != sorted(map(str, groups)) or len(
        set(groups)
    ) != len(groups):
        raise RuntimeError("materialization development groups are not canonical")
    raw_folds = manifest.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != FOLD_COUNT:
        raise RuntimeError("materialization must contain five folds")
    folds: dict[int, dict[str, Any]] = {}
    owner_seen: set[str] = set()
    group_set = set(groups)
    for raw in raw_folds:
        if not isinstance(raw, Mapping):
            raise RuntimeError("materialization fold must be a mapping")
        item = dict(raw)
        owner = item.get("outer_fold_id")
        if not isinstance(owner, int) or owner not in range(FOLD_COUNT) or owner in folds:
            raise RuntimeError("materialization fold owner mismatch")
        training = list(map(str, item.get("training_groups", [])))
        holdout = list(map(str, item.get("oof_holdout_groups", [])))
        if (
            training != sorted(training)
            or holdout != sorted(holdout)
            or set(training) & set(holdout)
            or set(training) | set(holdout) != group_set
        ):
            raise RuntimeError("materialization fold partition is invalid")
        if owner_seen & set(holdout):
            raise RuntimeError("materialization group has multiple holdout owners")
        owner_seen.update(holdout)
        folds[owner] = item
    if owner_seen != group_set or set(folds) != set(range(FOLD_COUNT)):
        raise RuntimeError("materialization holdouts do not cover development groups")
    base_identity_values: set[str] = set()
    for owner in range(FOLD_COUNT):
        audit = folds[owner].get("base_exclusion_audit")
        if not isinstance(audit, Mapping) or audit.get("status") != "proven":
            raise RuntimeError("materialization base identity exclusion is not proven")
        digest = audit.get("base_identity_contract_sha256")
        if not _is_sha256(digest):
            raise RuntimeError("materialization base identity contract SHA is invalid")
        base_identity_values.add(str(digest))
    if base_identity_values != {sources.get("base_identity_contract")}:
        raise RuntimeError(
            "materialization base identity contract differs from adaptive contract"
        )
    return manifest, folds


def _expected_input_artifact_authentication(
    *,
    materialization: Mapping[str, Any],
    materialization_path: Path,
    owner: int,
    materialization_folds: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the trainer's complete ten-artifact authentication receipt."""

    bundle_hash_rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        row = materialization_folds[fold_id]
        for role in ("train", "holdout"):
            artifact = Path(str(row.get(f"{role}_artifact", ""))).resolve()
            expected_sha = row.get(f"{role}_artifact_sha256")
            payload_sha = row.get(f"{role}_payload_sha256")
            if (
                not artifact.is_file()
                or not _is_sha256(expected_sha)
                or sha256_path(artifact) != expected_sha
                or not _is_sha256(payload_sha)
            ):
                raise RuntimeError(
                    f"v8 authenticated {role} artifact changed for fold {fold_id}"
                )
            bundle_hash_rows.append(
                {
                    "outer_fold_id": fold_id,
                    "role": role,
                    "path": str(artifact),
                    "sha256": expected_sha,
                    "payload_sha256": payload_sha,
                }
            )
    selected = materialization_folds[owner]
    return {
        "status": "authenticated_complete_five_fold_materialization_bundle",
        "materialization_manifest": str(materialization_path.resolve()),
        "materialization_sha256": materialization["materialization_sha256"],
        "outer_fold_id": owner,
        "train_artifact_sha256": selected["train_artifact_sha256"],
        "train_payload_sha256": selected["train_payload_sha256"],
        "ten_artifact_bundle_sha256": canonical_sha256(bundle_hash_rows),
    }


def _validate_bundle(bundle: Mapping[str, Any], adaptive_contract: Mapping[str, Any]) -> None:
    validate_adaptive_development_contract(adaptive_contract)
    if bundle.get("format") != BRIDGE_FORMAT or bundle.get("status") != (
        "authenticated_inputs_rehashed"
    ):
        raise RuntimeError("v8 bridge bundle format/status mismatch")
    _signed(bundle, "bridge_bundle_sha256", name="v8 bridge bundle")
    if bundle.get("source_partition") != "development_only" or bundle.get(
        "fresh50_inputs_accepted"
    ) is not False or bundle.get("fresh50_labels_read") is not False:
        raise RuntimeError("v8 bridge bundle is not development-only")
    if bundle.get("adaptive_development_contract_sha256") != adaptive_contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("v8 bridge bundle contract binding mismatch")
    if bundle.get("bridge_implementation_sha256") != sha256_path(
        Path(__file__).resolve()
    ):
        raise RuntimeError("v8 bridge implementation changed after bundle creation")
    folds = bundle.get("folds")
    if not isinstance(folds, list) or len(folds) != FOLD_COUNT:
        raise RuntimeError("v8 bridge bundle must bind five folds")
    for owner, item in enumerate(folds):
        if not isinstance(item, Mapping) or item.get("owner_fold_id") != owner:
            raise RuntimeError("v8 bridge bundle fold order/owner mismatch")
        if item.get("checkpoint_role") != "outer_training_only_adapter_checkpoint" or item.get(
            "holdout_role"
        ) != "owner_outer_holdout_evaluation_only":
            raise RuntimeError("v8 bridge input role mismatch")
        for path_key, sha_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("holdout_artifact", "holdout_artifact_sha256"),
        ):
            path = _reject_confirmation_path(Path(str(item[path_key])))
            if sha256_path(path) != item.get(sha_key):
                raise RuntimeError(f"v8 bridge {path_key} file SHA mismatch")


def _probability_provenance(
    checkpoint: Mapping[str, Any], *, owner_fold_id: int
) -> dict[str, dict[str, Any]]:
    support = checkpoint.get("support")
    training = checkpoint.get("training_contract")
    if not isinstance(support, Mapping) or not isinstance(training, Mapping):
        raise RuntimeError("checkpoint lacks probability training provenance")
    if training.get("loss") != V8_LOSS_CONTRACT or any(
        training.get(key) != "unweighted_binary_cross_entropy"
        for key in ("success_loss", "regress_loss")
    ) or training.get("recovery_loss") != (
        "unweighted_binary_cross_entropy_on_true_regress_rows_only"
    ):
        raise RuntimeError("checkpoint probability losses are not fixed unweighted BCE")
    result: dict[str, dict[str, Any]] = {}
    for head in ("success", "regress", "recovery_given_regress"):
        total = support.get(f"{head}_support")
        positive = support.get(f"{head}_positive")
        if (
            not isinstance(total, int)
            or not isinstance(positive, int)
            or total <= 0
            or positive <= 0
            or positive >= total
        ):
            raise RuntimeError(f"checkpoint {head} outer-training support is invalid")
        item: dict[str, Any] = {
            "head": head,
            "owner_fold_id": owner_fold_id,
            "loss": "unweighted_bce",
            "weights_recorded_before_training": True,
            "owner_holdout_labels_used": False,
            "calibration_source": "none_unweighted_probability",
            "outer_training_positive": positive,
            "outer_training_negative": total - positive,
            "outer_training_prevalence": positive / total,
        }
        item["weight_contract_sha256"] = canonical_sha256(item)
        result[head] = item
    return result


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    owner: int,
    materialization_fold: Mapping[str, Any],
    expected_input_authentication: Mapping[str, Any],
    adaptive_contract: Mapping[str, Any],
) -> tuple[V8DetachedStructuredAdapters, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if checkpoint.get("format") != V8_TRAINING_CHECKPOINT_FORMAT or int(
        checkpoint.get("schema_version", -1)
    ) != 5:
        raise RuntimeError("v8 adapter checkpoint format/schema mismatch")
    if checkpoint.get("fresh_confirmation_data_or_labels_read") is not False or checkpoint.get(
        "authorization_guard_changed"
    ) is not False:
        raise RuntimeError("v8 adapter checkpoint crossed the development boundary")
    if checkpoint.get("all_steps_factual_inputs_bit_exact") is not True:
        raise RuntimeError("v8 adapter checkpoint factual inputs were not bit-exact")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("outer_fold_id") != owner:
        raise RuntimeError("v8 adapter checkpoint owner provenance mismatch")
    if provenance.get("target_outer_fold_labels_used") is not False or provenance.get(
        "factual_outputs_frozen"
    ) is not True:
        raise RuntimeError("v8 adapter checkpoint used owner labels or mutable factual outputs")
    authentication = checkpoint.get("input_artifact_authentication")
    if not isinstance(authentication, Mapping) or dict(authentication) != dict(
        expected_input_authentication
    ):
        raise RuntimeError(
            "v8 adapter checkpoint input artifact authentication mismatch"
        )
    sources = adaptive_contract["source_sha256"]
    if provenance.get("base_checkpoint_sha256") != sources["base_checkpoint"] or provenance.get(
        "label_derivation_sha256"
    ) != sources["label_derivation"]:
        raise RuntimeError("v8 adapter checkpoint source hash mismatch")
    if list(map(str, provenance.get("outer_training_groups", []))) != list(
        map(str, materialization_fold["training_groups"])
    ) or list(map(str, provenance.get("oof_holdout_groups", []))) != list(
        map(str, materialization_fold["oof_holdout_groups"])
    ):
        raise RuntimeError("v8 adapter checkpoint train/holdout ownership mismatch")
    duration_baseline = _validate_repair_contract(
        provenance,
        field="duration_baseline_contract",
        signature_field="duration_baseline_contract_sha256",
    )
    object_fallback = _validate_repair_contract(
        provenance,
        field="object_fallback_contract",
        signature_field="object_fallback_contract_sha256",
    )
    if object_fallback.get("object_mode") != V8_OBJECT_MODE or object_fallback.get(
        "learned_object_output_authorized"
    ) is not False:
        raise RuntimeError("object fallback contract incorrectly authorizes learned output")
    duration_scale = _validate_duration_scale_contract(
        checkpoint.get("duration_laplace_scale_contract"), owner_fold_id=owner
    )
    provenance_duration_scale = _validate_duration_scale_contract(
        provenance.get("duration_laplace_scale_contract"), owner_fold_id=owner
    )
    if (
        duration_scale != provenance_duration_scale
        or duration_scale["contract_sha256"]
        != provenance.get("duration_laplace_scale_contract_sha256")
    ):
        raise RuntimeError(
            "checkpoint top-level duration scale differs from signed provenance"
        )
    config = V8StructuredAdapterConfig.from_dict(checkpoint.get("config", {}))
    adapters = V8DetachedStructuredAdapters(config).eval()
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("v8 adapter checkpoint lacks state_dict")
    adapters.load_state_dict(state, strict=True)
    if module_state_sha256(adapters) != checkpoint.get("adapter_state_sha256"):
        raise RuntimeError("v8 adapter state SHA mismatch")
    return adapters, dict(provenance), duration_scale, object_fallback


def _validate_materialized_payload_binding(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    materialization_fold: Mapping[str, Any],
    owner: int,
    role: str,
) -> None:
    """Bind one loaded payload to its signed manifest artifact and content SHA."""

    if role not in ("train", "holdout"):
        raise RuntimeError("unknown materialized payload role")
    expected_format = V8_TRAINING_INPUT_FORMAT if role == "train" else HOLDOUT_FORMAT
    if payload.get("format") != expected_format or int(payload.get("schema_version", -1)) != 5:
        raise RuntimeError(f"materialized {role} format/schema mismatch")
    expected_path = _reject_confirmation_path(
        Path(str(materialization_fold.get(f"{role}_artifact", "")))
    )
    if artifact_path.resolve() != expected_path:
        raise RuntimeError(f"materialized {role} artifact path mismatch")
    expected_file_sha = materialization_fold.get(f"{role}_artifact_sha256")
    if not _is_sha256(expected_file_sha) or sha256_path(artifact_path) != expected_file_sha:
        raise RuntimeError(f"materialized {role} artifact file SHA mismatch")
    recorded_payload_sha = payload.get("payload_sha256")
    expected_payload_sha = materialization_fold.get(f"{role}_payload_sha256")
    actual_payload_sha = structured_payload_sha256(payload)
    if (
        not _is_sha256(recorded_payload_sha)
        or recorded_payload_sha != expected_payload_sha
        or recorded_payload_sha != actual_payload_sha
    ):
        raise RuntimeError(f"materialized {role} payload SHA mismatch")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("outer_fold_id") != owner:
        raise RuntimeError(f"materialized {role} owner provenance mismatch")
    if provenance.get("target_outer_fold_labels_used") is not False or provenance.get(
        "factual_outputs_frozen"
    ) is not True:
        raise RuntimeError(f"materialized {role} weakened factual provenance")


def _validated_factual_state_audit(
    payload: Mapping[str, Any], *, role: str
) -> tuple[str, dict[str, Any]]:
    """Validate a payload-local audit and return its immutable factual-state SHA."""

    audit = payload.get("materialization_audit")
    if not isinstance(audit, Mapping) or audit.get("factual_state_bit_exact") is not True:
        raise RuntimeError(f"materialized {role} lacks bit-exact factual-state audit")
    before = audit.get("factual_state_sha256_before")
    after = audit.get("factual_state_sha256_after")
    if not _is_sha256(before) or before != after:
        raise RuntimeError(f"materialized {role} factual state changed")
    records = payload.get("batches")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise RuntimeError(f"materialized {role} batches are empty")
    rows = 0
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("batch"), Mapping):
            raise RuntimeError(f"materialized {role} record is invalid")
        structured_mask = record["batch"].get("structured_mask")
        if not torch.is_tensor(structured_mask) or structured_mask.ndim != 1:
            raise RuntimeError(f"materialized {role} structured mask is invalid")
        rows += int(structured_mask.shape[0])
    if audit.get("records") != len(records) or audit.get("rows") != rows:
        raise RuntimeError(f"materialized {role} factual-state audit counts disagree")
    return str(before), dict(audit)


def _factual_state_sha(
    train: Mapping[str, Any],
    holdout: Mapping[str, Any],
    materialization_fold: Mapping[str, Any],
) -> str:
    """Require train and holdout payloads to preserve one identical factual state.

    Deployed r3 manifests authenticate both artifact files and payload hashes but do
    not duplicate the payload-local audits in each fold row.  The authenticated
    payloads are therefore the source of truth.  Newer manifests may duplicate the
    audits; when present, those copies must agree exactly and cannot weaken checks.
    """

    train_sha, train_audit = _validated_factual_state_audit(train, role="train")
    holdout_sha, holdout_audit = _validated_factual_state_audit(
        holdout, role="holdout"
    )
    if train_sha != holdout_sha:
        raise RuntimeError("train/holdout factual state hashes disagree")
    manifest_train = materialization_fold.get("training_materialization_audit")
    manifest_holdout = materialization_fold.get("holdout_materialization_audit")
    if (manifest_train is None) != (manifest_holdout is None):
        raise RuntimeError("materialization manifest has incomplete factual-state audits")
    if manifest_train is not None and (
        not isinstance(manifest_train, Mapping)
        or not isinstance(manifest_holdout, Mapping)
        or dict(manifest_train) != train_audit
        or dict(manifest_holdout) != holdout_audit
    ):
        raise RuntimeError("manifest/payload factual-state audits disagree")
    return train_sha


def _binary_numpy(value: Any, *, name: str, count: int) -> np.ndarray:
    if not torch.is_tensor(value) or tuple(value.shape) != (count,):
        raise RuntimeError(f"{name} must be an aligned tensor")
    result = value.detach().cpu().numpy()
    if np.any((result != 0) & (result != 1)):
        raise RuntimeError(f"{name} must be binary")
    return result.astype(bool)


def _validate_record_mapping(
    record: Mapping[str, Any], *, owner: int, expected_group: str
) -> tuple[dict[str, Any], np.ndarray, str]:
    if record.get("logical_group_key") != expected_group or record.get(
        "split_role"
    ) != "outer_holdout" or record.get("outer_fold_id") != owner:
        raise RuntimeError("holdout record identity/owner/role mismatch")
    batch = record.get("batch")
    factual = record.get("factual_outputs")
    if not isinstance(batch, Mapping) or not isinstance(factual, Mapping):
        raise RuntimeError("holdout record lacks batch/factual outputs")
    if record.get("factual_outputs_require_grad") is not False or record.get(
        "factual_outputs_sha256"
    ) != frozen_tensor_mapping_sha256(factual):
        raise RuntimeError("holdout factual tensor hash/gradient contract mismatch")
    transition = factual.get("transition")
    if not torch.is_tensor(transition) or transition.ndim != 2:
        raise RuntimeError("holdout factual transition is invalid")
    count = int(transition.shape[0])
    validate_schema5_adapter_batch(batch, expected_count=count)
    validate_factual_adapter_inputs(
        factual, count=count, transition_dim=int(transition.shape[1])
    )
    names = batch.get("candidate_names")
    group_keys = batch.get("group_keys")
    if not isinstance(names, list) or len(names) != count or group_keys != [expected_group]:
        raise RuntimeError("holdout candidate/group row mapping is incomplete")
    terminal = _binary_numpy(batch.get("terminal_mask"), name="terminal_mask", count=count)
    baseline = _binary_numpy(batch.get("baseline_mask"), name="baseline_mask", count=count)
    if count < 4 or not np.array_equal(np.flatnonzero(terminal), np.arange(4)):
        raise RuntimeError("candidate rows 0..3 must be the only terminal deployment rows")
    if not np.array_equal(np.flatnonzero(baseline), np.asarray([0])) or names[0] != (
        "deterministic"
    ):
        raise RuntimeError("candidate0 must be the unique deterministic baseline")
    if len(set(map(str, names[:4]))) != 4 or any(
        str(name).startswith("continuation_") for name in names[:4]
    ):
        raise RuntimeError("deployment candidate names are invalid")
    if list(map(str, names[4:])) != [
        f"continuation_{index}" for index in range(count - 4)
    ]:
        raise RuntimeError("continuation row mapping is not canonical")
    group_index = batch.get("group_index")
    if not torch.is_tensor(group_index) or not np.array_equal(
        group_index.detach().cpu().numpy(),
        np.concatenate(
            [np.zeros(4, dtype=np.int64), -np.ones(count - 4, dtype=np.int64)]
        ),
    ):
        raise RuntimeError("candidate/continuation group_index mapping changed")
    physical = record.get("object_delta_physical")
    normalized = batch.get("object_delta")
    if (
        not torch.is_tensor(physical)
        or not torch.is_tensor(normalized)
        or tuple(physical.shape) != tuple(normalized.shape)
        or physical.ndim != 2
        or not bool(torch.isfinite(physical).all())
    ):
        raise RuntimeError(
            "holdout lacks aligned physical object_delta; prediction agent must add "
            "object_delta_physical rather than evaluating normalized coordinates"
        )
    quality = record.get("object_pose_quality_valid")
    if quality is None:
        quality_array = np.zeros(count, dtype=bool)
        quality_status = "unavailable_all_rows_fail_closed"
    else:
        quality_array = _binary_numpy(
            quality, name="object_pose_quality_valid", count=count
        )
        quality_status = "explicit_per_row_quality_provenance"
    return dict(batch), quality_array, quality_status


def build_evaluation_inputs(
    *,
    bundle: Mapping[str, Any],
    adaptive_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build every evaluator array and signed fold/head provenance object."""

    _validate_bundle(bundle, adaptive_contract)
    materialization, materialization_folds = _validate_materialization(
        bundle, adaptive_contract
    )
    materialization_path = Path(str(bundle["materialization_manifest"])).resolve()
    arrays_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "logical_group", "fold_id", "historical_old100_overlap",
            "candidate_index", "duration_observed", "duration_steps",
            "duration_model_log_location", "duration_frozen_log_location",
            "duration_model_log_scale", "duration_baseline_log_location",
            "duration_baseline_log_scale", "success_mask", "success_label",
            "success_probability", "success_baseline_probability", "regress_mask",
            "regress_label", "regress_probability", "regress_baseline_probability",
            "recovery_label", "recovery_probability_given_regress",
            "recovery_baseline_probability_given_regress", "object_mask",
            "object_pose_quality_valid", "object_delta", "object_model_delta",
            "object_robust_median_delta",
        )
    }
    fold_contracts: dict[str, dict[str, Any]] = {}
    probability: dict[str, dict[str, Any]] = {
        "success": {}, "regress": {}, "recovery_given_regress": {}
    }
    object_quality_status_by_fold: dict[str, str] = {}
    base_exclusion_status_by_fold: dict[str, str] = {}
    old100_groups: set[str] = set()
    factual_state_shas: set[str] = set()
    for owner, bundle_fold in enumerate(bundle["folds"]):
        materialization_fold = materialization_folds[owner]
        train_path = _reject_confirmation_path(
            Path(str(materialization_fold["train_artifact"]))
        )
        holdout_path = Path(str(bundle_fold["holdout_artifact"])).resolve()
        if str(holdout_path) != str(Path(str(materialization_fold["holdout_artifact"])).resolve()) or (
            bundle_fold["holdout_artifact_sha256"]
            != materialization_fold.get("holdout_artifact_sha256")
        ):
            raise RuntimeError("bundle holdout is not the materialized owner artifact")
        checkpoint = _load_torch_mapping(
            Path(str(bundle_fold["checkpoint"])), role="adapter checkpoint"
        )
        train = _load_torch_mapping(train_path, role="outer-training materialization")
        holdout = _load_torch_mapping(holdout_path, role="holdout")
        _validate_materialized_payload_binding(
            train,
            artifact_path=train_path,
            materialization_fold=materialization_fold,
            owner=owner,
            role="train",
        )
        _validate_materialized_payload_binding(
            holdout,
            artifact_path=holdout_path,
            materialization_fold=materialization_fold,
            owner=owner,
            role="holdout",
        )
        adapters, checkpoint_provenance, duration_scale, object_fallback = (
            _validate_checkpoint(
                checkpoint,
                owner=owner,
                materialization_fold=materialization_fold,
                expected_input_authentication=_expected_input_artifact_authentication(
                    materialization=materialization,
                    materialization_path=materialization_path,
                    owner=owner,
                    materialization_folds=materialization_folds,
                ),
                adaptive_contract=adaptive_contract,
            )
        )
        train_provenance = train.get("provenance")
        if not isinstance(train_provenance, Mapping) or dict(
            train_provenance
        ) != checkpoint_provenance:
            raise RuntimeError("checkpoint/train provenance differs")
        holdout_provenance = holdout.get("provenance")
        if not isinstance(holdout_provenance, Mapping):
            raise RuntimeError("holdout lacks provenance")
        expected_holdout_provenance = {
            **checkpoint_provenance,
            "split_role": "outer_holdout_evaluation_only",
            "holdout_labels_used_for_duration_or_object_fit": False,
            "holdout_labels_present_only_in_separate_artifact": True,
        }
        if dict(holdout_provenance) != expected_holdout_provenance:
            raise RuntimeError("checkpoint/holdout provenance differs or role was weakened")
        if holdout.get("format") != HOLDOUT_FORMAT or int(
            holdout.get("schema_version", -1)
        ) != 5:
            raise RuntimeError("holdout format/schema mismatch")
        factual_sha = _factual_state_sha(train, holdout, materialization_fold)
        factual_state_shas.add(factual_sha)
        base_audit = checkpoint_provenance.get("base_exclusion_audit")
        if not isinstance(base_audit, Mapping):
            raise RuntimeError("holdout lacks historical base-overlap audit")
        if dict(base_audit) != dict(materialization_fold.get("base_exclusion_audit", {})):
            raise RuntimeError("checkpoint/materialization base-overlap audits disagree")
        base_status = str(base_audit.get("status", ""))
        if base_status not in ("proven", "unproven_development_only"):
            raise RuntimeError("historical base-overlap audit has an unknown status")
        base_exclusion_status_by_fold[str(owner)] = base_status
        fold_old100 = set(map(str, base_audit.get("legacy_old100_holdout_groups", [])))
        if not fold_old100.issubset(set(materialization_fold["oof_holdout_groups"])):
            raise RuntimeError("historical old100 overlap contains a non-owner group")
        old100_groups.update(fold_old100)
        head_provenance = _probability_provenance(checkpoint, owner_fold_id=owner)
        for head, value in head_provenance.items():
            probability[head][str(owner)] = value
        exact_cells = checkpoint_provenance["duration_baseline_contract"].get(
            "exact_event_body", {}
        )
        if not isinstance(exact_cells, Mapping) or not exact_cells:
            raise RuntimeError("duration event/body training support is missing")
        minimum_cell_support = min(int(item["support"]) for item in exact_cells.values())
        fold_contract: dict[str, Any] = {
            "owner_fold_id": owner,
            "heldout_logical_groups": list(materialization_fold["oof_holdout_groups"]),
            "training_logical_groups": list(materialization_fold["training_groups"]),
            "outer_target_labels_used_for_fit": False,
            "baseline_fit_scope": "outer_training_only",
            "calibration_fit_scope": "none",
            "base_checkpoint_sha256": adaptive_contract["source_sha256"][
                "base_checkpoint"
            ],
            "next_event_state_sha256_before": factual_sha,
            "next_event_state_sha256_after": factual_sha,
            "next_event_hash_scope": "full_factual_model_state_stronger_than_head_only",
            "next_event_trainable": False,
            "trainable_parameter_names": list(adapters.trainable_parameter_names()),
            "duration_event_body_min_training_support": minimum_cell_support,
            "duration_residual_multiplier": V8_DURATION_RESIDUAL_MULTIPLIER,
            "duration_censored_used_for_location": False,
            "duration_scale_fit_scope": "outer_training_only",
            "duration_scale_contract_sha256": duration_scale["contract_sha256"],
        }
        fold_contract["fold_contract_sha256"] = canonical_sha256(fold_contract)
        fold_contracts[str(owner)] = fold_contract
        records = holdout.get("batches")
        expected_groups = list(materialization_fold["oof_holdout_groups"])
        if not isinstance(records, list) or len(records) != len(expected_groups):
            raise RuntimeError("holdout records do not cover owner groups")
        fold_quality_status: set[str] = set()
        for record, group in zip(records, expected_groups):
            if not isinstance(record, Mapping):
                raise RuntimeError("holdout record must be a mapping")
            batch, quality, quality_status = _validate_record_mapping(
                record, owner=owner, expected_group=group
            )
            fold_quality_status.add(quality_status)
            factual = record["factual_outputs"]
            count = int(factual["transition"].shape[0])
            baseline = record.get("duration_baseline_log1p")
            fallback = record.get("object_fallback")
            if not torch.is_tensor(baseline) or tuple(baseline.shape) != (count,):
                raise RuntimeError("holdout duration baseline is misaligned")
            normalized_expected = np.asarray(
                object_fallback.get("schema5_normalized_fallback"), dtype=np.float64
            )
            if not torch.is_tensor(fallback) or not np.allclose(
                fallback.detach().cpu().numpy(), normalized_expected, rtol=0.0, atol=1e-7
            ):
                raise RuntimeError("holdout object fallback differs from outer training")
            with torch.inference_mode():
                output = adapters(
                    factual,
                    duration_baseline_log1p=baseline,
                    object_fallback=fallback,
                )
            if output.get("learned_object_output_authorized") is not False or output.get(
                "object_prediction_status"
            ) != V8_OBJECT_MODE:
                raise RuntimeError("v8 adapter incorrectly claims learned object output")
            expected_duration_native = baseline + V8_DURATION_RESIDUAL_MULTIPLIER * (
                factual["duration_selected_log_mean"] - baseline
            )
            if not torch.equal(
                output["duration_repaired_log1p_mean"], expected_duration_native
            ):
                raise RuntimeError("v8 adapter duration output changed from the frozen formula")
            training_prevalence = {
                head: probability[head][str(owner)]["outer_training_prevalence"]
                for head in probability
            }
            physical = record["object_delta_physical"].detach().cpu().numpy().astype(
                np.float64
            )
            physical_fallback = np.asarray(
                object_fallback.get("physical_coordinate_median"), dtype=np.float64
            )
            if physical_fallback.shape != physical.shape[1:] or not np.isfinite(
                physical_fallback
            ).all():
                raise RuntimeError("physical object fallback is invalid")
            terminal = batch["terminal_mask"].detach().cpu().numpy().astype(bool)
            structured = batch["structured_mask"].detach().cpu().numpy().astype(bool)
            dense = batch["dense_mask"].detach().cpu().numpy().astype(bool)
            arrays_parts["logical_group"].append(
                np.asarray([group] * count, dtype=str)
            )
            arrays_parts["fold_id"].append(np.full(count, owner, dtype=np.int64))
            arrays_parts["historical_old100_overlap"].append(
                np.full(count, group in fold_old100, dtype=bool)
            )
            arrays_parts["candidate_index"].append(np.arange(count, dtype=np.int64))
            arrays_parts["duration_observed"].append(
                batch["duration_observed"].detach().cpu().numpy().astype(bool)
            )
            arrays_parts["duration_steps"].append(
                batch["duration"].detach().cpu().numpy().astype(np.float64)
            )
            baseline_numpy = baseline.detach().cpu().numpy().astype(np.float64)
            frozen_duration_numpy = factual[
                "duration_selected_log_mean"
            ].detach().cpu().numpy().astype(np.float64)
            # Re-evaluate the already bit-exact-checked formula in float64 so
            # the pure-NumPy evaluator need not treat float32 roundoff as a
            # semantic contract violation.
            arrays_parts["duration_model_log_location"].append(
                baseline_numpy
                + V8_DURATION_RESIDUAL_MULTIPLIER
                * (frozen_duration_numpy - baseline_numpy)
            )
            arrays_parts["duration_frozen_log_location"].append(
                frozen_duration_numpy
            )
            arrays_parts["duration_model_log_scale"].append(
                np.full(count, float(duration_scale["model_log_scale"]), dtype=np.float64)
            )
            arrays_parts["duration_baseline_log_location"].append(
                baseline_numpy
            )
            arrays_parts["duration_baseline_log_scale"].append(
                np.full(count, float(duration_scale["baseline_log_scale"]), dtype=np.float64)
            )
            arrays_parts["success_mask"].append(terminal)
            arrays_parts["success_label"].append(
                batch["success"].detach().cpu().numpy().astype(np.float64)
            )
            arrays_parts["success_probability"].append(
                output["success_probability"].cpu().numpy().astype(np.float64)
            )
            arrays_parts["success_baseline_probability"].append(
                np.full(count, training_prevalence["success"], dtype=np.float64)
            )
            arrays_parts["regress_mask"].append(structured)
            arrays_parts["regress_label"].append(
                batch["trajectory_regress"].detach().cpu().numpy().astype(np.float64)
            )
            arrays_parts["regress_probability"].append(
                output["regress_probability"].cpu().numpy().astype(np.float64)
            )
            arrays_parts["regress_baseline_probability"].append(
                np.full(count, training_prevalence["regress"], dtype=np.float64)
            )
            arrays_parts["recovery_label"].append(
                batch["trajectory_recovery"].detach().cpu().numpy().astype(np.float64)
            )
            arrays_parts["recovery_probability_given_regress"].append(
                output["recovery_given_regress_probability"].cpu().numpy().astype(np.float64)
            )
            arrays_parts["recovery_baseline_probability_given_regress"].append(
                np.full(
                    count,
                    training_prevalence["recovery_given_regress"],
                    dtype=np.float64,
                )
            )
            arrays_parts["object_mask"].append(dense)
            arrays_parts["object_pose_quality_valid"].append(quality)
            arrays_parts["object_delta"].append(physical)
            robust = np.broadcast_to(physical_fallback, physical.shape).copy()
            arrays_parts["object_model_delta"].append(robust)
            arrays_parts["object_robust_median_delta"].append(robust.copy())
        object_quality_status_by_fold[str(owner)] = (
            next(iter(fold_quality_status))
            if len(fold_quality_status) == 1
            else "mixed_explicit_and_unavailable_fail_closed"
        )
    if len(factual_state_shas) != 1:
        raise RuntimeError("five owner folds do not share one frozen factual state")
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in arrays_parts.items()}
    if set(map(str, arrays["logical_group"])) != set(
        map(str, materialization["development_groups"])
    ):
        raise RuntimeError("bridged arrays do not cover all development groups")
    return {
        "arrays": arrays,
        "fold_contracts": fold_contracts,
        "probability_weight_provenance": probability,
        "input_contract": dict(INPUT_CONTRACT),
        "bridge_provenance": {
            "format": OUTPUT_FORMAT,
            "bridge_bundle_sha256": bundle["bridge_bundle_sha256"],
            "adaptive_development_contract_sha256": adaptive_contract[
                "contract_sha256"
            ],
            "materialization_sha256": materialization["materialization_sha256"],
            "historical_old100_logical_groups": sorted(old100_groups),
            "base_exclusion_status_by_fold": base_exclusion_status_by_fold,
            "all_primary_base_exclusion_proven": all(
                status == "proven"
                for status in base_exclusion_status_by_fold.values()
            ),
            "candidate_mapping": "candidate0_deterministic_candidates0_through3_deployment_continuation4plus",
            "probability_baseline": "owner_outer_training_prevalence",
            "duration_scale": "owner_outer_training_observed_only_mad_laplace",
            "next_event": "full_factual_state_bit_exact_only_accuracy_not_evaluated",
            "object_output": "outer_training_robust_physical_fallback_never_learned",
            "object_quality_status_by_fold": object_quality_status_by_fold,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "prospective_claim_allowed": False,
        },
    }


def evaluate_oof_artifacts(
    *,
    checkpoint_paths: Sequence[Path],
    holdout_paths: Sequence[Path],
    materialization_manifest_path: Path,
    base_identity_contract_sha256: str,
) -> dict[str, Any]:
    """Create the adaptive contract/bundle, build arrays, and run evaluation."""

    materialization = _load_json_mapping(
        _reject_confirmation_path(materialization_manifest_path),
        role="materialization manifest",
    )
    if not _is_sha256(base_identity_contract_sha256):
        raise RuntimeError("base identity contract SHA is invalid")
    adaptive_contract = make_adaptive_development_contract(
        implementation_sha256=sha256_path(Path(__file__).resolve()),
        label_derivation_sha256=str(materialization.get("label_derivation_sha256", "")),
        base_checkpoint_sha256=str(materialization.get("base_checkpoint_sha256", "")),
        base_identity_contract_sha256=base_identity_contract_sha256,
    )
    bundle = make_bridge_bundle(
        checkpoint_paths=checkpoint_paths,
        holdout_paths=holdout_paths,
        materialization_manifest_path=materialization_manifest_path,
        adaptive_contract=adaptive_contract,
    )
    built = build_evaluation_inputs(
        bundle=bundle, adaptive_contract=adaptive_contract
    )
    result = evaluate_adaptive_development_structured_heads_arrays(
        built["arrays"],
        adaptive_contract=adaptive_contract,
        fold_contracts=built["fold_contracts"],
        probability_weight_provenance=built["probability_weight_provenance"],
        input_contract=built["input_contract"],
    )
    validate_adaptive_development_result(result, adaptive_contract)
    return {
        **built,
        "adaptive_contract": adaptive_contract,
        "bridge_bundle": bundle,
        "evaluation_result": result,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_evaluation_output(output_dir: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = _reject_confirmation_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory {output_dir}")
    output_dir.mkdir(parents=True)
    arrays_path = output_dir / "structured_heads_arrays.npz"
    contracts_path = output_dir / "structured_heads_contracts.json"
    result_path = output_dir / "structured_heads_evaluation.json"
    _atomic_npz(arrays_path, value["arrays"])
    contracts = {
        "format": OUTPUT_FORMAT,
        "adaptive_contract": value["adaptive_contract"],
        "bridge_bundle": value["bridge_bundle"],
        "fold_contracts": value["fold_contracts"],
        "probability_weight_provenance": value["probability_weight_provenance"],
        "input_contract": value["input_contract"],
        "bridge_provenance": value["bridge_provenance"],
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_path(arrays_path),
    }
    contracts["contracts_sha256"] = canonical_sha256(contracts)
    _atomic_json(contracts_path, contracts)
    _atomic_json(result_path, value["evaluation_result"])
    return {
        "status": "complete_adaptive_development_only",
        "arrays": str(arrays_path),
        "arrays_sha256": contracts["arrays_sha256"],
        "contracts": str(contracts_path),
        "contracts_file_sha256": sha256_path(contracts_path),
        "result": str(result_path),
        "result_file_sha256": sha256_path(result_path),
        "prospective_claim_allowed": False,
        "fresh50_labels_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--base-identity-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = evaluate_oof_artifacts(
        checkpoint_paths=args.checkpoint,
        holdout_paths=args.holdout,
        materialization_manifest_path=args.materialization_manifest,
        base_identity_contract_sha256=args.base_identity_contract_sha256,
    )
    summary = write_evaluation_output(args.output, value)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BRIDGE_FORMAT",
    "DURATION_SCALE_FORMAT",
    "OUTPUT_FORMAT",
    "build_evaluation_inputs",
    "evaluate_oof_artifacts",
    "make_bridge_bundle",
    "write_evaluation_output",
]
