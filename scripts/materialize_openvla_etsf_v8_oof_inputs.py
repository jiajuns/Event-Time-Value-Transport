#!/usr/bin/env python3
"""Materialise leakage-audited schema5 inputs for v8 detached adapters.

The identity-only descriptor scan and OOF owner split happen before any label
dataset is loaded.  For each outer fold, training groups are loaded first and
are the only source for duration medians and the robust object fallback.  The
holdout groups are loaded afterwards and written to a separate artifact; their
labels can never enter the v8 training payload.

The production path reuses the existing schema5 ``scan_group_descriptors``,
``load_descriptor_groups``, ``collate_groups`` and ``forward_model`` APIs.  No
factual parameter is trainable and every materialised factual tensor is a
detached CPU clone.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

import train_openvla_etsf_counterfactual as counterfactual
from openvla_etsf_counterfactual_oof import (
    OOF_SPLIT_SEED,
    canonical_sha256,
    make_oof_folds,
)
from openvla_etsf_event_world_model import (
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from openvla_etsf_prediction_repair import (
    DURATION_RESIDUAL_PROTOCOL,
    OBJECT_REPAIR_PROTOCOL,
    apply_duration_residual_contract,
    fit_duration_residual_contract,
    fit_object_repair_contract,
)
from openvla_etsf_v7_development_confirmation import (
    validate_preregistration as validate_v7_preregistration,
    validate_seed_manifest as validate_v7_seed_manifest,
)
from openvla_etsf_v8_structured_adapters import (
    V8_DURATION_RESIDUAL_MULTIPLIER,
    V8_OBJECT_MODE,
    frozen_tensor_mapping_sha256,
    module_state_sha256,
)
from train_openvla_etsf_v8_structured_adapters import (
    V8_TRAINING_INPUT_FORMAT,
    structured_payload_sha256,
    validate_v8_training_payload,
)


V8_HOLDOUT_INPUT_FORMAT = "etsf_v8_detached_adapter_holdout_input_v1"
V8_OOF_MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
V8_BASE_EXCLUSION_FORMAT = "etsf_factual_base_exclusion_contract_v1"
V8_OWNER_MANIFEST_FORMAT = "etsf_v8_detached_adapter_owner_folds_v1"
V8_COLLECTION_SOURCE_FORMAT = "etsf_v8_signed_development_collection_source_v1"
DEPLOYMENT_CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)
FORMAL_OOF_FORMATS = {
    "etsf_counterfactual_five_fold_oof_v1",
    "etsf_counterfactual_nested_oof_v6",
    V8_OWNER_MANIFEST_FORMAT,
}
FOLD_COUNT = 5
MATERIALIZED_BATCH_TENSOR_KEYS = (
    "terminal_mask",
    "structured_mask",
    "dense_mask",
    "duration",
    "duration_observed",
    "success",
    "trajectory_regress",
    "trajectory_recovery",
    "object_delta",
    "current_event_id",
    "next_event_id",
    "next_reached_event_id",
    "clock_event_id",
    "body_id",
    "policy_id",
    "group_index",
    "baseline_mask",
)


def _fixed_outer_holdouts(
    *, manifest_format: str, logical_keys: Sequence[str], split_seed: int
) -> list[list[str]]:
    keys = sorted(map(str, logical_keys))
    namespace = "outer" if manifest_format == "etsf_counterfactual_nested_oof_v6" else None
    ordered = sorted(
        keys,
        key=lambda key: hashlib.sha256(
            (
                f"{namespace}|{split_seed}|{key}"
                if namespace is not None
                else f"{split_seed}|{key}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    if len(keys) % FOLD_COUNT:
        raise RuntimeError("OOF logical identities are not divisible into five folds")
    width = len(keys) // FOLD_COUNT
    return [
        sorted(ordered[fold_id * width : (fold_id + 1) * width])
        for fold_id in range(FOLD_COUNT)
    ]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_group_list_sha256(keys: Sequence[str]) -> str:
    normalized = list(map(str, keys))
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("logical group lists must be sorted and unique")
    return canonical_sha256({"logical_groups": normalized})


def uncertainty_materialization_contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "etsf_v8_single_factual_uncertainty_materialization_v1",
        "stored_tensor": "aleatoric_uncertainty",
        "stored_tensor_source": "factual_forward_model_aleatoric_uncertainty",
        "epistemic_uncertainty": "unavailable_requires_frozen_ensemble",
        "total_uncertainty": "unavailable_not_fabricated_fail_closed",
        "allowed_claim": "developmental_single_member_risk_coverage_only",
        "ensemble_total_uncertainty_claim": False,
    }
    value["uncertainty_materialization_contract_sha256"] = canonical_sha256(value)
    return value


def _signed_mapping_sha256(value: Mapping[str, Any], signature_key: str) -> str:
    unsigned = dict(value)
    recorded = str(unsigned.pop(signature_key, ""))
    if len(recorded) != 64 or recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{signature_key} is missing or invalid")
    return recorded


def normalize_oof_owner_folds(
    manifest: Mapping[str, Any],
    logical_keys: Sequence[str],
    *,
    require_formal_size: bool = True,
) -> list[dict[str, Any]]:
    """Normalize v5/v6 owner folds and prove train/holdout disjointness."""

    if manifest.get("format") not in FORMAL_OOF_FORMATS:
        raise RuntimeError("unsupported OOF owner manifest format")
    if manifest.get("status") != "preregistered":
        raise RuntimeError("OOF owner manifest is not preregistered")
    _signed_mapping_sha256(manifest, "preregistration_sha256")
    keys = sorted(map(str, logical_keys))
    if len(set(keys)) != len(keys) or (require_formal_size and len(keys) not in (100, 250)):
        raise RuntimeError("OOF materialization needs 100 or 250 unique development groups")
    if sorted(map(str, manifest.get("development_groups", []))) != keys:
        raise RuntimeError("OOF manifest identities differ from descriptor scan")
    if manifest.get("format") == V8_OWNER_MANIFEST_FORMAT and manifest.get(
        "development_groups_sha256"
    ) != logical_group_list_sha256(keys):
        raise RuntimeError("v8 OOF development group-list SHA changed")
    raw_folds = (
        manifest.get("outer_folds")
        if manifest.get("format") == "etsf_counterfactual_nested_oof_v6"
        else manifest.get("folds")
    )
    if not isinstance(raw_folds, Sequence) or len(raw_folds) != FOLD_COUNT:
        raise RuntimeError("OOF manifest must contain five owner folds")
    owners: dict[str, int] = {}
    normalized: list[dict[str, Any]] = []
    key_set = set(keys)
    for expected_fold, raw in enumerate(raw_folds):
        if not isinstance(raw, Mapping):
            raise RuntimeError("OOF fold must be a mapping")
        fold_id = int(raw.get("outer_fold_id", raw.get("fold_id", -1)))
        if fold_id != expected_fold:
            raise RuntimeError("OOF fold ids/order changed")
        training = sorted(map(str, raw.get("training_groups", [])))
        holdout = sorted(map(str, raw.get("oof_holdout_groups", [])))
        if (
            set(training) & set(holdout)
            or set(training) | set(holdout) != key_set
            or len(training) + len(holdout) != len(keys)
        ):
            raise RuntimeError("OOF train/holdout partition is invalid")
        if manifest.get("format") == V8_OWNER_MANIFEST_FORMAT and (
            raw.get("training_groups_sha256")
            != logical_group_list_sha256(training)
            or raw.get("oof_holdout_groups_sha256")
            != logical_group_list_sha256(holdout)
        ):
            raise RuntimeError("v8 OOF fold group-list SHA changed")
        for key in holdout:
            if key in owners:
                raise RuntimeError("a logical group has multiple OOF owners")
            owners[key] = fold_id
        normalized.append(
            {
                "outer_fold_id": fold_id,
                "training_groups": training,
                "oof_holdout_groups": holdout,
                "training_groups_sha256": logical_group_list_sha256(training),
                "oof_holdout_groups_sha256": logical_group_list_sha256(holdout),
            }
        )
    if set(owners) != key_set:
        raise RuntimeError("OOF holdouts do not cover every logical group exactly once")
    split_seed = int(manifest.get("split_seed", -1))
    if split_seed != OOF_SPLIT_SEED:
        raise RuntimeError("OOF split seed differs from the fixed protocol")
    expected_holdouts = _fixed_outer_holdouts(
        manifest_format=str(manifest["format"]),
        logical_keys=keys,
        split_seed=split_seed,
    )
    if [fold["oof_holdout_groups"] for fold in normalized] != expected_holdouts:
        raise RuntimeError("OOF owners differ from the fixed SHA256 split")
    return normalized


def reject_fresh_sources(inputs: Sequence[Path]) -> None:
    """Defense in depth: the materializer has no confirmation-data mode."""

    for source in inputs:
        resolved = source.resolve()
        if "fresh" in resolved.name.lower() or "confirmation" in resolved.name.lower():
            raise RuntimeError("Fresh confirmation sources are forbidden for v8 OOF")
        root = resolved if resolved.is_dir() else resolved.parent
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() and root.parent.joinpath("manifest.json").is_file():
            manifest_path = root.parent / "manifest.json"
        if not manifest_path.is_file():
            continue
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            value.get("seed_registry") == "explicit_fresh_confirmation"
            or value.get("fresh_seed_manifest_sha256") not in (None, "")
        ):
            raise RuntimeError("Fresh confirmation sources are forbidden for v8 OOF")


def validate_v7_collection_trust_roots(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Verify v7 signed roots without reopening any Fresh exclusion artifact."""

    seed_path = Path(str(identity.get("v7_seed_manifest", ""))).expanduser().resolve()
    prereg_path = Path(
        str(identity.get("v7_preregistration", ""))
    ).expanduser().resolve()
    if not seed_path.is_file() or not prereg_path.is_file():
        raise RuntimeError("v7 collection seed/preregistration trust roots are missing")
    if sha256_path(seed_path) != identity.get("v7_seed_manifest_sha256"):
        raise RuntimeError("v7 collection seed-manifest SHA changed")
    seed = _load_json(seed_path)
    prereg = _load_json(prereg_path)
    seed_audit = validate_v7_seed_manifest(seed, verify_files=False)
    validate_v7_preregistration(prereg)
    if (
        prereg.get("preregistration_sha256")
        != identity.get("v7_preregistration_sha256")
        or prereg.get("seed_manifest_payload_sha256")
        != seed.get("seed_manifest_payload_sha256")
        or list(map(int, identity.get("requested_seeds", [])))
        != seed_audit["requested_seeds"]
        or list(map(int, identity.get("resolved_seeds", [])))
        != seed_audit["resolved_seeds"]
    ):
        raise RuntimeError("v7 collection identities differ from signed seed/prereg roots")
    prereg_source = prereg.get("source_contract")
    if not isinstance(prereg_source, Mapping) or (
        Path(str(prereg_source.get("seed_manifest", ""))).expanduser().resolve()
        != seed_path
        or prereg_source.get("seed_manifest_file_sha256") != sha256_path(seed_path)
    ):
        raise RuntimeError("v7 preregistration does not bind the seed manifest")
    exclusion_sources = seed.get("exclusion_sources")
    if not isinstance(exclusion_sources, Mapping) or set(exclusion_sources) != {
        "official150",
        "development150",
        "fresh50",
    }:
        raise RuntimeError("v7 exclusion-registry trust roots are incomplete")
    exclusion_bindings: dict[str, dict[str, Any]] = {}
    for name, source in exclusion_sources.items():
        if not isinstance(source, Mapping):
            raise RuntimeError("v7 exclusion-registry trust root is malformed")
        binding = {
            "path": source.get("path"),
            "sha256": source.get("sha256"),
            "identity_sets_sha256": source.get("identity_sets_sha256"),
        }
        if any(not isinstance(value, str) or not value for value in binding.values()):
            raise RuntimeError("v7 exclusion-registry binding is incomplete")
        exclusion_bindings[str(name)] = binding
    return {
        "status": "signed_v7_seed_and_preregistration_verified",
        "seed_manifest": str(seed_path),
        "seed_manifest_file_sha256": sha256_path(seed_path),
        "seed_manifest_payload_sha256": seed["seed_manifest_payload_sha256"],
        "preregistration": str(prereg_path),
        "preregistration_file_sha256": sha256_path(prereg_path),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "frozen_factual_checkpoint": prereg_source.get("pretrained"),
        "frozen_factual_checkpoint_sha256": prereg_source.get("pretrained_sha256"),
        "event_spec": prereg_source.get("event_spec"),
        "event_spec_sha256": prereg_source.get("event_spec_sha256"),
        "exclusion_registry_bindings": exclusion_bindings,
        "exclusion_registry_files_reopened": False,
        "fresh_exclusion_artifact_read": False,
    }


def validate_development_collection_contract(
    *,
    data_inputs: Sequence[Path],
    descriptors: Sequence[counterfactual.GroupDescriptor],
    event_spec_sha256: str,
    oof_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind schema5 HDF5 files to the signed collection and OOF contracts.

    The collector intentionally does not duplicate ``event_spec_sha256`` in
    each HDF5 root.  Consequently the label-free collection identity, complete
    collection manifest, and every HDF5 digest must be content-bound by the
    already signed OOF source contract.  Missing bindings fail closed.
    """

    if len(data_inputs) != 1 or not data_inputs[0].resolve().is_dir():
        raise RuntimeError("formal v8 requires one signed development collection root")
    root = data_inputs[0].resolve()
    manifest_path = root / "manifest.json"
    identity_path = root / "collection_identity.json"
    if not manifest_path.is_file() or not identity_path.is_file():
        raise RuntimeError("development collection lacks manifest/collection_identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(identity, Mapping):
        raise RuntimeError("development collection manifests must be JSON objects")
    descriptor_keys = sorted(descriptor.logical_key for descriptor in descriptors)
    if (
        manifest.get("status") != "complete"
        or identity.get("status") != "complete"
        or int(manifest.get("schema_version", -1)) != 5
        or int(identity.get("schema_version", -1)) != 5
        or int(manifest.get("completed", -1)) != len(descriptors)
        or int(identity.get("completed", -1)) != len(descriptors)
    ):
        raise RuntimeError("development collection is incomplete or not schema5")
    if (
        manifest.get("seed_registry") != "explicit_v7_prospective_development"
        or identity.get("seed_registry") != "explicit_v7_prospective_development"
        or any(
            value not in (None, "")
            for value in (
                manifest.get("fresh_seed_manifest"),
                manifest.get("fresh_seed_manifest_sha256"),
                identity.get("fresh_seed_manifest"),
                identity.get("fresh_seed_manifest_sha256"),
            )
        )
    ):
        raise RuntimeError("development collection seed registry/provenance changed")
    if (
        manifest.get("event_spec_sha256") != event_spec_sha256
        or identity.get("event_spec_sha256") != event_spec_sha256
    ):
        raise RuntimeError("development collection event-spec SHA mismatch")
    if identity.get("format") != "etsf_event_branch_collection_identity_v1" or identity.get(
        "label_access_contract"
    ) != "identity_only_no_success_steps_event_or_outcome_fields":
        raise RuntimeError("development collection identity contract changed")
    identity_root_fields = (
        "schema_version",
        "status",
        "task",
        "body",
        "model_path",
        "requested_seeds",
        "resolved_seeds",
        "seed_registry",
        "event_spec_sha256",
        "candidate_count",
        "v7_seed_manifest",
        "v7_seed_manifest_sha256",
        "v7_preregistration",
        "v7_preregistration_sha256",
        "completed",
    )
    if any(identity.get(key) != manifest.get(key) for key in identity_root_fields):
        raise RuntimeError("collection manifest and label-free identity roots differ")
    v7_trust_root = validate_v7_collection_trust_roots(identity)
    task = str(manifest.get("task", ""))
    body = str(manifest.get("body", ""))
    inferred_policy = str(manifest.get("policy", "")).lower()
    if not inferred_policy:
        inferred_policy = (
            "openvla" if "openvla" in str(manifest.get("model_path", "")).lower() else ""
        )
    identity_policy = str(identity.get("policy", inferred_policy)).lower()
    if (
        not task
        or not body
        or inferred_policy != "openvla"
        or identity_policy != "openvla"
        or identity.get("task") != task
        or identity.get("body") != body
        or int(manifest.get("candidate_count", -1)) != 4
        or int(identity.get("candidate_count", -1)) != 4
        or any(
            descriptor.task != task
            or descriptor.body != body
            or descriptor.policy != "openvla"
            or descriptor.schema_version != 5
            for descriptor in descriptors
        )
    ):
        raise RuntimeError("development collection task/body/policy/candidate contract changed")

    source = oof_manifest.get("source_contract")
    if not isinstance(source, Mapping):
        raise RuntimeError("signed OOF manifest lacks a collection source contract")
    if (
        source.get("format") != V8_COLLECTION_SOURCE_FORMAT
        or source.get("event_spec_sha256") != event_spec_sha256
        or not isinstance(source.get("data_root"), str)
        or Path(str(source["data_root"])).resolve() != root
        or source.get("fresh_seed_manifest") not in (None, "")
        or source.get("fresh_labels_read") is not False
    ):
        raise RuntimeError("signed OOF collection source contract is missing or changed")
    if source.get("v7_trust_root") != v7_trust_root:
        raise RuntimeError("signed OOF v7 seed/preregistration trust root changed")
    expected_manifest_sha = source.get("collector_manifest_sha256")
    expected_identity_sha = source.get("collection_identity_sha256")
    if (
        expected_manifest_sha != sha256_path(manifest_path)
        or expected_identity_sha != sha256_path(identity_path)
    ):
        raise RuntimeError("signed OOF manifest/identity SHA binding is missing or changed")
    recorded_manifest_path = source.get("collector_manifest")
    recorded_identity_path = source.get("collection_identity")
    if (
        not isinstance(recorded_manifest_path, str)
        or Path(recorded_manifest_path).resolve() != manifest_path.resolve()
        or not isinstance(recorded_identity_path, str)
        or Path(recorded_identity_path).resolve() != identity_path.resolve()
    ):
        raise RuntimeError("signed collection manifest paths changed")
    source_files = source.get("development_group_files")
    if not isinstance(source_files, Sequence) or isinstance(source_files, (str, bytes)):
        raise RuntimeError("signed OOF source lacks development HDF5 SHA rows")
    source_by_key = {
        str(row.get("logical_key")): row
        for row in source_files
        if isinstance(row, Mapping)
    }
    if sorted(source_by_key) != descriptor_keys:
        raise RuntimeError("signed OOF HDF5 identity coverage changed")
    manifest_groups = manifest.get("groups")
    identity_groups = identity.get("groups")
    if (
        not isinstance(manifest_groups, list)
        or not isinstance(identity_groups, list)
        or len(manifest_groups) != len(descriptors)
        or len(identity_groups) != len(descriptors)
    ):
        raise RuntimeError("collection group identity/path coverage is incomplete")
    descriptor_by_key = {descriptor.logical_key: descriptor for descriptor in descriptors}
    file_rows: list[dict[str, Any]] = []
    candidate_names: tuple[str, ...] | None = None
    seen_keys: set[str] = set()
    for index, (manifest_row, identity_row) in enumerate(
        zip(manifest_groups, identity_groups)
    ):
        if not isinstance(manifest_row, Mapping) or not isinstance(identity_row, Mapping):
            raise RuntimeError("collection group rows must be mappings")
        identity_fields = (
            "index",
            "seed",
            "requested_seed",
            "resolved_seed",
            "path",
            "candidate_names",
            "status",
        )
        if any(identity_row.get(key) != manifest_row.get(key) for key in identity_fields):
            raise RuntimeError("collection manifest and label-free identity rows differ")
        if int(manifest_row.get("index", -1)) != index or manifest_row.get("status") not in (
            "collected",
            "existing",
        ):
            raise RuntimeError("collection group order/status changed")
        resolved_seed = int(manifest_row.get("resolved_seed", -1))
        logical_key = f"{task}|{body}|{resolved_seed}"
        if logical_key in seen_keys or logical_key not in descriptor_by_key:
            raise RuntimeError("collection group logical identity coverage changed")
        seen_keys.add(logical_key)
        descriptor = descriptor_by_key[logical_key]
        file_name = str(manifest_row.get("path", ""))
        group_path = root / "groups" / file_name
        if not group_path.is_file() or Path(descriptor.path).resolve() != group_path.resolve():
            raise RuntimeError("collection group resolved path changed")
        source_row = source_by_key[logical_key]
        if (
            Path(str(source_row.get("path", ""))).resolve() != group_path.resolve()
            or int(source_row.get("schema_version", -1)) != 5
            or source_row.get("sha256") != sha256_path(group_path)
        ):
            raise RuntimeError("signed development HDF5 SHA/path mismatch")
        with h5py.File(group_path, "r") as handle:
            attrs = handle.attrs
            if (
                int(attrs.get("schema_version", -1)) != 5
                or str(attrs.get("task", "")) != task
                or str(attrs.get("body", "")) != body
                or int(attrs.get("seed", -1))
                != int(manifest_row.get("requested_seed", -1))
                or int(attrs.get("requested_seed", -1))
                != int(manifest_row.get("requested_seed", -1))
                or int(attrs.get("resolved_seed", -1)) != resolved_seed
                or int(attrs.get("candidate_count", -1)) != 4
                or "candidate_names" not in handle
            ):
                raise RuntimeError("development HDF5 root identity contract changed")
            names = tuple(counterfactual.decode_strings(handle["candidate_names"][:]))
        if names != DEPLOYMENT_CANDIDATE_NAMES or tuple(
            map(str, manifest_row.get("candidate_names", []))
        ) != names:
            raise RuntimeError("development candidate names/order changed")
        if candidate_names is None:
            candidate_names = names
        elif candidate_names != names:
            raise RuntimeError("candidate names/order differ across development groups")
        file_rows.append(
            {
                "logical_key": logical_key,
                "path": str(group_path.resolve()),
                "sha256": source_row["sha256"],
                "schema_version": 5,
            }
        )
    if seen_keys != set(descriptor_keys):
        raise RuntimeError("collection group logical identities are incomplete")
    return {
        "status": "complete_schema5_signed_source_verified",
        "root": str(root),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": expected_manifest_sha,
        "collection_identity": str(identity_path.resolve()),
        "collection_identity_sha256": expected_identity_sha,
        "event_spec_sha256": event_spec_sha256,
        "seed_registry": "explicit_v7_prospective_development",
        "v7_trust_root": v7_trust_root,
        "task": task,
        "body": body,
        "policy": "openvla",
        "candidate_names": list(candidate_names or ()),
        "groups": file_rows,
        "groups_sha256": canonical_sha256(file_rows),
        "labels_used_for_owner_split": False,
    }


def preregister_v8_owner_manifest(
    *,
    data_root: Path,
    checkpoint_path: Path,
    event_spec_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create label-independent v8 owner folds bound to a complete collection.

    The split is fixed from schema5 logical identities.  The label-free
    ``collection_identity.json`` is the only parsed collection JSON before the
    folds are signed; ``manifest.json`` is content-hashed but not parsed until
    the already-fixed split is audited by
    :func:`validate_development_collection_contract`.
    """

    data_root = data_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    event_spec_path = event_spec_path.resolve()
    output_path = output_path.resolve()
    reject_fresh_sources([data_root])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    manifest_path = data_root / "manifest.json"
    identity_path = data_root / "collection_identity.json"
    for path in (manifest_path, identity_path, checkpoint_path, event_spec_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = _load_json(identity_path)
    if (
        identity.get("format") != "etsf_event_branch_collection_identity_v1"
        or identity.get("label_access_contract")
        != "identity_only_no_success_steps_event_or_outcome_fields"
        or identity.get("status") != "complete"
        or int(identity.get("schema_version", -1)) != 5
        or identity.get("seed_registry")
        != "explicit_v7_prospective_development"
        or identity.get("fresh_seed_manifest") not in (None, "")
        or identity.get("fresh_seed_manifest_sha256") not in (None, "")
        or identity.get("event_spec_sha256") != sha256_path(event_spec_path)
    ):
        raise RuntimeError("v8 owner preregistration identity contract is incomplete")
    v7_trust_root = validate_v7_collection_trust_roots(identity)
    if (
        v7_trust_root.get("frozen_factual_checkpoint_sha256")
        != sha256_path(checkpoint_path)
        or v7_trust_root.get("event_spec_sha256") != sha256_path(event_spec_path)
    ):
        raise RuntimeError("v8 checkpoint/event spec differ from the v7 signed trust root")
    descriptors = counterfactual.scan_group_descriptors([data_root])
    keys = sorted(descriptor.logical_key for descriptor in descriptors)
    if (
        len(keys) not in (100, 250)
        or len(set(keys)) != len(keys)
        or int(identity.get("completed", -1)) != len(keys)
        or len(identity.get("groups", [])) != len(keys)
        or any(
            descriptor.schema_version != 5 or descriptor.policy != "openvla"
            for descriptor in descriptors
        )
    ):
        raise RuntimeError("v8 owner preregistration requires 100 or 250 OpenVLA schema5 groups")
    source_contract = {
        "format": V8_COLLECTION_SOURCE_FORMAT,
        "data_root": str(data_root),
        "collector_manifest": str(manifest_path),
        "collector_manifest_sha256": sha256_path(manifest_path),
        "collection_identity": str(identity_path),
        "collection_identity_sha256": sha256_path(identity_path),
        "event_spec": str(event_spec_path),
        "event_spec_sha256": sha256_path(event_spec_path),
        "pretrained": str(checkpoint_path),
        "pretrained_sha256": sha256_path(checkpoint_path),
        "development_group_files": [
            {
                "logical_key": descriptor.logical_key,
                "path": str(Path(descriptor.path).resolve()),
                "sha256": sha256_path(Path(descriptor.path).resolve()),
                "schema_version": descriptor.schema_version,
            }
            for descriptor in descriptors
        ],
        "v7_trust_root": v7_trust_root,
        "identity_scan_before_owner_split": True,
        "collection_manifest_content_hashed_before_split_but_not_parsed": True,
        "fresh_seed_manifest": None,
        "fresh_labels_read": False,
    }
    legacy = make_oof_folds(
        keys, split_seed=OOF_SPLIT_SEED, source_contract=source_contract
    )
    folds: list[dict[str, Any]] = []
    for legacy_fold in legacy["folds"]:
        training = sorted(map(str, legacy_fold["training_groups"]))
        holdout = sorted(map(str, legacy_fold["oof_holdout_groups"]))
        folds.append(
            {
                "outer_fold_id": int(legacy_fold["fold_id"]),
                "training_groups": training,
                "training_groups_sha256": logical_group_list_sha256(training),
                "oof_holdout_groups": holdout,
                "oof_holdout_groups_sha256": logical_group_list_sha256(holdout),
                "checkpoint_selection": "fixed_epoch_no_holdout_selection",
            }
        )
    payload: dict[str, Any] = {
        "format": V8_OWNER_MANIFEST_FORMAT,
        "status": "preregistered",
        "split_algorithm": legacy["split_algorithm"],
        "split_seed": OOF_SPLIT_SEED,
        "fold_count": FOLD_COUNT,
        "expected_groups": len(keys),
        "development_groups": keys,
        "development_groups_sha256": logical_group_list_sha256(keys),
        "source_contract": source_contract,
        "label_access_contract": (
            "identity_only_owner_split_before_any_schema5_label_dataset_load"
        ),
        "timing_scope": "adaptive_development_only_designed_after_v7_collection_started",
        "prospective_claim_for_v8": False,
        "v7_fixed_policy_evaluation_remains_separate_and_prospective": True,
        "fresh_confirmation": {
            "inputs_accepted": False,
            "data_or_labels_read": False,
            "authorization_possible": False,
        },
        "folds": folds,
    }
    payload["preregistration_sha256"] = canonical_sha256(payload)
    normalize_oof_owner_folds(payload, keys, require_formal_size=True)
    # This parses the complete collector manifest only after owner folds and
    # their signature are fixed; it cannot influence group ownership.
    validate_development_collection_contract(
        data_inputs=[data_root],
        descriptors=descriptors,
        event_spec_sha256=sha256_path(event_spec_path),
        oof_manifest=payload,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, payload)
    return payload


def make_label_derivation_contract(
    *,
    event_spec_sha256: str,
    checkpoint_contract: Mapping[str, Any],
    regression_persistence_steps: int,
) -> dict[str, Any]:
    source_path = Path(counterfactual.__file__).resolve()
    value: dict[str, Any] = {
        "format": "etsf_schema5_label_derivation_contract_v1",
        "implementation_path": str(source_path),
        "implementation_sha256": sha256_path(source_path),
        "event_spec_sha256": event_spec_sha256,
        "regression_persistence_steps": int(regression_persistence_steps),
        "regression_recovery_derivation": "derive_regression_recovery",
        "predicate_contract": dict(checkpoint_contract.get("predicate_contract", {})),
        "labels_rederived_while_loading_schema5": True,
    }
    value["label_derivation_sha256"] = canonical_sha256(value)
    return value


def load_frozen_factual_context(
    checkpoint_path: Path,
    event_spec_path: Path,
    *,
    device: torch.device | str = torch.device("cpu"),
    regression_persistence_steps: int = counterfactual.DEFAULT_REGRESSION_PERSISTENCE_STEPS,
) -> dict[str, Any]:
    """Load one factual checkpoint in eval mode with all parameters frozen."""

    checkpoint_path = checkpoint_path.resolve()
    event_spec_path = event_spec_path.resolve()
    checkpoint_sha = sha256_path(checkpoint_path)
    event_spec_sha = sha256_path(event_spec_path)
    checkpoint, config = counterfactual.load_pretrained(checkpoint_path)
    if (
        not config.structured_events
        or config.action_rank_residual
        or config.action_rank_success_only
    ):
        raise RuntimeError("v8 requires a structured factual checkpoint without rank mode")
    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("factual checkpoint lacks a training contract")
    if contract.get("event_spec_sha256") != event_spec_sha:
        raise RuntimeError("factual checkpoint/event-spec SHA mismatch")
    event_spec = json.loads(event_spec_path.read_text(encoding="utf-8"))
    calibrations = event_spec.get("calibration")
    if not isinstance(calibrations, Mapping):
        raise RuntimeError("event spec lacks task calibration")
    object_names = contract.get("object_names")
    body_to_id = contract.get("body_to_id")
    policy_to_id = contract.get("policy_to_id")
    if (
        not isinstance(object_names, Sequence)
        or isinstance(object_names, (str, bytes))
        or not object_names
        or not isinstance(body_to_id, Mapping)
        or not isinstance(policy_to_id, Mapping)
    ):
        raise RuntimeError("factual checkpoint lacks object/body/policy mappings")
    policy_to_id = counterfactual.canonical_policy_mapping(policy_to_id)
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise RuntimeError("factual checkpoint lacks object normalization")
    object_mean = np.asarray(normalization.get("object_delta_mean"), dtype=np.float32)
    object_std = np.asarray(normalization.get("object_delta_std"), dtype=np.float32)
    if (
        object_mean.shape != (config.object_delta_dim,)
        or object_std.shape != object_mean.shape
        or not np.isfinite(object_mean).all()
        or not np.isfinite(object_std).all()
        or np.any(object_std <= 0)
    ):
        raise RuntimeError("factual object normalization is invalid")
    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(torch.device(device)).eval()
    label_contract = make_label_derivation_contract(
        event_spec_sha256=event_spec_sha,
        checkpoint_contract=contract,
        regression_persistence_steps=regression_persistence_steps,
    )
    return {
        "model": model,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_contract": dict(contract),
        "event_spec_path": str(event_spec_path),
        "event_spec_sha256": event_spec_sha,
        "calibrations": calibrations,
        "object_names": list(map(str, object_names)),
        "body_to_id": {str(key): int(value) for key, value in body_to_id.items()},
        "policy_to_id": {str(key): int(value) for key, value in policy_to_id.items()},
        "object_mean": object_mean,
        "object_std": object_std,
        "label_derivation_contract": label_contract,
        "factual_state_sha256": module_state_sha256(model),
    }


def _collate_all(
    groups: Sequence[counterfactual.BranchGroup],
    *,
    object_mean: np.ndarray,
    object_std: np.ndarray,
) -> dict[str, Any]:
    if not groups:
        raise ValueError("outer training groups cannot be empty")
    return counterfactual.collate_groups(
        groups,
        object_mean=object_mean,
        object_std=object_std,
        include_auxiliary=True,
    )


def fit_outer_training_repairs(
    training_groups: Sequence[counterfactual.BranchGroup],
    *,
    object_mean: np.ndarray,
    object_std: np.ndarray,
) -> dict[str, Any]:
    """Fit duration/object contracts from outer-training labels only."""

    batch = _collate_all(
        training_groups, object_mean=object_mean, object_std=object_std
    )
    dense = batch["dense_mask"].bool().cpu().numpy()
    observed = (
        batch["duration_observed"].bool() & batch["dense_mask"].bool()
    ).cpu().numpy()
    duration_contract = fit_duration_residual_contract(
        batch["duration"].cpu().numpy(),
        observed,
        batch["clock_event_id"].cpu().numpy(),
        batch["body_id"].cpu().numpy(),
    )
    normalized_object = batch["object_delta"].cpu().numpy().astype(np.float64)
    physical_object = normalized_object * object_std[None] + object_mean[None]
    object_contract = fit_object_repair_contract(physical_object[dense])
    physical_fallback = np.asarray(
        object_contract["coordinate_median"], dtype=np.float32
    )
    normalized_fallback = (physical_fallback - object_mean) / object_std
    object_fallback_contract: dict[str, Any] = {
        "format": "etsf_v8_outer_training_object_fallback_contract_v1",
        "repair_protocol": OBJECT_REPAIR_PROTOCOL,
        "fit_rows": "outer_training_dense_rows_only",
        "object_mode": V8_OBJECT_MODE,
        "learned_object_output_authorized": False,
        "physical_coordinate_median": physical_fallback.tolist(),
        "schema5_normalized_fallback": normalized_fallback.tolist(),
        "robust_repair_contract": object_contract,
    }
    object_fallback_contract["object_fallback_contract_sha256"] = canonical_sha256(
        object_fallback_contract
    )
    duration_contract_with_sha = dict(duration_contract)
    duration_contract_with_sha["duration_baseline_contract_sha256"] = canonical_sha256(
        duration_contract_with_sha
    )
    return {
        "duration_contract": duration_contract_with_sha,
        "object_fallback_contract": object_fallback_contract,
        "object_fallback_normalized": torch.as_tensor(
            normalized_fallback, dtype=torch.float32
        ),
        "training_dense_support": int(dense.sum()),
        "training_observed_duration_support": int(observed.sum()),
    }


def _batch_for_artifact(batch: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in MATERIALIZED_BATCH_TENSOR_KEYS:
        value = batch.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"collated schema5 batch lacks tensor {key}")
        result[key] = value.detach().cpu().clone()
    result["group_keys"] = list(map(str, batch.get("group_keys", [])))
    result["candidate_names"] = list(map(str, batch.get("candidate_names", [])))
    return result


def fit_duration_residual_uncertainty_contract(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit the bridge-compatible outer-train-only duration Laplace scales."""

    model_residuals: list[float] = []
    baseline_residuals: list[float] = []
    owner_folds = {int(record.get("outer_fold_id", -1)) for record in records}
    if len(owner_folds) != 1 or next(iter(owner_folds)) not in range(FOLD_COUNT):
        raise ValueError("duration scale records must share one valid owner fold")
    for record in records:
        if record.get("split_role") != "outer_training":
            raise ValueError("duration uncertainty scale may use outer training only")
        batch = record["batch"]
        mask = batch["dense_mask"].bool() & batch["duration_observed"].bool()
        target = torch.log1p(batch["duration"].float().clamp_min(0.0))
        baseline = record["duration_baseline_log1p"].float()
        factual = record["factual_outputs"]["duration_selected_log_mean"].float()
        repaired = baseline + V8_DURATION_RESIDUAL_MULTIPLIER * (factual - baseline)
        model_residual = (target - repaired).detach().cpu().numpy()
        baseline_residual = (target - baseline).detach().cpu().numpy()
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            model_residuals.append(float(model_residual[index]))
            baseline_residuals.append(float(baseline_residual[index]))
    if not model_residuals:
        raise RuntimeError("outer training has no observed duration residual support")

    def log_scale(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        center = float(np.median(array))
        mad = float(np.median(np.abs(array - center)))
        return float(np.log(max(mad / np.log(2.0), 1e-4)))

    value: dict[str, Any] = {
        "format": "etsf_v8_outer_training_duration_laplace_scale_v1",
        "owner_fold_id": next(iter(owner_folds)),
        "fit_scope": "outer_training_observed_only",
        "estimator": "median_absolute_deviation_divided_by_log_2",
        "censored_rows_used": False,
        "model_location": "event_body_median_plus_0.375_frozen_residual",
        "baseline_location": "outer_training_event_body_median",
        "minimum_scale": 1e-4,
        "outer_training_observed_support": len(model_residuals),
        "model_log_scale": log_scale(model_residuals),
        "baseline_log_scale": log_scale(baseline_residuals),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def materialize_group_records(
    groups: Sequence[counterfactual.BranchGroup],
    *,
    split_role: str,
    outer_fold_id: int,
    model: ActionConditionedEventWorldModel,
    duration_contract: Mapping[str, Any],
    object_fallback_normalized: torch.Tensor,
    object_mean: np.ndarray,
    object_std: np.ndarray,
    device: torch.device | str = torch.device("cpu"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split_role not in ("outer_training", "outer_holdout"):
        raise ValueError("unknown v8 materialization split role")
    if duration_contract.get("protocol") != DURATION_RESIDUAL_PROTOCOL:
        raise ValueError("duration contract protocol changed")
    device = torch.device(device)
    model_state_before = module_state_sha256(model)
    records: list[dict[str, Any]] = []
    for group in groups:
        batch_cpu = counterfactual.collate_groups(
            [group],
            object_mean=object_mean,
            object_std=object_std,
            include_auxiliary=True,
        )
        baseline, source = apply_duration_residual_contract(
            duration_contract,
            batch_cpu["clock_event_id"].cpu().numpy(),
            batch_cpu["body_id"].cpu().numpy(),
        )
        batch_device = counterfactual.move_batch(batch_cpu, device)
        with torch.inference_mode():
            factual_output = counterfactual.forward_model(model, batch_device)
        frozen_outputs = {
            "transition": factual_output["transition"].detach().cpu().clone(),
            "duration_selected_log_mean": factual_output[
                "duration_selected_log_mean"
            ].detach().cpu().clone(),
            "next_event_logits": factual_output["next_event_logits"]
            .detach()
            .cpu()
            .clone(),
            "next_reached_event_logits": factual_output[
                "next_reached_event_logits"
            ]
            .detach()
            .cpu()
            .clone(),
            "aleatoric_uncertainty": factual_output["aleatoric_uncertainty"]
            .detach()
            .cpu()
            .clone(),
        }
        frozen_sha = frozen_tensor_mapping_sha256(frozen_outputs)
        artifact_batch = _batch_for_artifact(batch_cpu)
        artifact_batch["object_delta_physical"] = (
            batch_cpu["object_delta"]
            * torch.as_tensor(object_std, dtype=batch_cpu["object_delta"].dtype)
            + torch.as_tensor(object_mean, dtype=batch_cpu["object_delta"].dtype)
        ).detach().cpu().clone()
        artifact_batch["object_pose_quality_valid"] = None
        artifact_batch["object_pose_quality_status"] = (
            "unavailable_schema5_collector_has_no_quality_field_fail_closed"
        )
        records.append(
            {
                "logical_group_key": str(group.logical_key),
                "split_role": split_role,
                "outer_fold_id": int(outer_fold_id),
                "group_metadata": {
                    "logical_group_key": str(group.logical_key),
                    "source_path": str(group.path),
                    "schema_version": int(group.schema_version),
                    "seed": int(group.seed),
                    "task": str(group.task),
                    "body": str(group.body),
                    "body_id": int(group.body_id),
                    "policy": str(group.policy),
                    "policy_id": int(group.policy_id),
                    "candidate_names": list(map(str, group.candidate_names)),
                },
                "batch": artifact_batch,
                "object_delta_physical": artifact_batch[
                    "object_delta_physical"
                ].clone(),
                "factual_outputs": frozen_outputs,
                "factual_outputs_sha256": frozen_sha,
                "duration_baseline_log1p": torch.as_tensor(
                    baseline, dtype=torch.float32
                ),
                "duration_baseline_source": list(map(str, source.tolist())),
                "object_fallback": object_fallback_normalized.detach().cpu().clone(),
                "factual_outputs_require_grad": False,
                "total_uncertainty_status": (
                    "unavailable_single_forward_has_aleatoric_only_requires_ensemble_fail_closed"
                ),
            }
        )
    model_state_after = module_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("factual model state changed during materialization")
    return records, {
        "factual_state_sha256_before": model_state_before,
        "factual_state_sha256_after": model_state_after,
        "factual_state_bit_exact": True,
        "records": len(records),
        "rows": int(sum(len(record["batch"]["duration"]) for record in records)),
    }


def _validate_split_records(
    records: Sequence[Mapping[str, Any]], expected_keys: Sequence[str], split_role: str
) -> None:
    actual = [str(record.get("logical_group_key")) for record in records]
    if actual != list(expected_keys) or len(set(actual)) != len(actual):
        raise RuntimeError(f"{split_role} materialized group identities changed")
    if any(record.get("split_role") != split_role for record in records):
        raise RuntimeError(f"{split_role} record role changed")


def base_exclusion_status(
    *,
    checkpoint_sha256: str,
    holdout_groups: Sequence[str],
    exclusion_contract: Mapping[str, Any] | None,
    legacy_old100_groups: Sequence[str] = (),
    checkpoint_contract: Mapping[str, Any] | None = None,
    authorized_target_groups: Sequence[str] = (),
    v7_trust_root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    legacy_overlap = sorted(set(map(str, holdout_groups)) & set(map(str, legacy_old100_groups)))
    if legacy_overlap:
        return {
            "status": "unproven_development_only",
            "reason": "legacy_old100_factual_training_overlap_not_excluded",
            "legacy_old100_holdout_groups": legacy_overlap,
        }
    # The factual checkpoint is itself content-bound by the v7 preregistration.
    # Its embedded split contract is therefore the authoritative source for
    # base-data identities.  The target identities independently come from the
    # signed v7 seed/preregistration chain and the content-bound collection.
    # Only this two-sided derivation may upgrade the audit to ``proven``.
    if (
        isinstance(checkpoint_contract, Mapping)
        and isinstance(v7_trust_root, Mapping)
        and authorized_target_groups
    ):
        if (
            v7_trust_root.get("status")
            != "signed_v7_seed_and_preregistration_verified"
            or v7_trust_root.get("frozen_factual_checkpoint_sha256")
            != checkpoint_sha256
        ):
            raise RuntimeError("v7 trust root does not bind the factual checkpoint")
        split_names = ("train_seeds", "validation_seeds", "sealed_test_seeds")
        split_values: dict[str, list[int]] = {}
        for name in split_names:
            raw = checkpoint_contract.get(name)
            if (
                not isinstance(raw, Sequence)
                or isinstance(raw, (str, bytes))
                or not raw
            ):
                raise RuntimeError("factual checkpoint identity split is incomplete")
            values = [int(value) for value in raw]
            if len(set(values)) != len(values) or any(value < 0 for value in values):
                raise RuntimeError("factual checkpoint identity split is invalid")
            split_values[name] = sorted(values)
        base_sets = [set(split_values[name]) for name in split_names]
        if any(
            base_sets[left] & base_sets[right]
            for left in range(len(base_sets))
            for right in range(left + 1, len(base_sets))
        ):
            raise RuntimeError("factual checkpoint identity splits overlap")
        base_seeds = set().union(*base_sets)
        # This factual checkpoint was built from the frozen 100/25/25 split.
        # Requiring complete 150-seed coverage prevents a truncated contract
        # from proving an exclusion merely by omitting source identities.
        if len(base_seeds) != 150:
            raise RuntimeError("factual checkpoint identity split does not cover 150 seeds")
        for name in ("source_manifest_sha256", "event_spec_sha256"):
            digest = checkpoint_contract.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError("factual checkpoint provenance digest is incomplete")

        authorized = sorted(map(str, authorized_target_groups))
        if len(authorized) != 250 or len(set(authorized)) != len(authorized):
            raise RuntimeError("signed v7 target registry must contain 250 unique groups")
        holdout = sorted(map(str, holdout_groups))
        if not set(holdout).issubset(set(authorized)):
            raise RuntimeError("OOF holdout is outside the signed v7 target registry")

        def group_seed(key: str) -> int:
            parts = key.rsplit("|", 2)
            if len(parts) != 3 or not parts[0] or not parts[1]:
                raise RuntimeError("logical group identity is malformed")
            try:
                value = int(parts[2])
            except ValueError as error:
                raise RuntimeError("logical group seed is not an integer") from error
            if value < 0:
                raise RuntimeError("logical group seed is negative")
            return value

        target_seeds = {group_seed(key) for key in authorized}
        if len(target_seeds) != len(authorized):
            raise RuntimeError("signed v7 target registry repeats a resolved seed")
        overlap = sorted(base_seeds & target_seeds)
        if overlap:
            return {
                "status": "unproven_development_only",
                "reason": "checkpoint_bound_base_identity_overlap_detected",
                "overlap_seeds": overlap,
                "legacy_old100_holdout_groups": [],
            }
        base_identity_contract = {
            "source_manifest_sha256": checkpoint_contract[
                "source_manifest_sha256"
            ],
            "event_spec_sha256": checkpoint_contract["event_spec_sha256"],
            **split_values,
        }
        return {
            "status": "proven",
            "reason": (
                "checkpoint_bound_150_seed_contract_disjoint_from_"
                "signed_v7_250_resolved_seed_registry"
            ),
            "base_checkpoint_sha256": checkpoint_sha256,
            "base_identity_contract_sha256": canonical_sha256(
                base_identity_contract
            ),
            "base_seed_count": len(base_seeds),
            "authorized_target_group_count": len(authorized),
            "authorized_target_groups_sha256": logical_group_list_sha256(
                authorized
            ),
            "holdout_groups_sha256": logical_group_list_sha256(holdout),
            "v7_seed_manifest_payload_sha256": v7_trust_root[
                "seed_manifest_payload_sha256"
            ],
            "v7_preregistration_sha256": v7_trust_root[
                "preregistration_sha256"
            ],
            "legacy_old100_holdout_groups": [],
        }
    if exclusion_contract is None:
        return {
            "status": "unproven_development_only",
            "reason": "no_signed_factual_base_exclusion_contract",
            "legacy_old100_holdout_groups": [],
        }
    if exclusion_contract.get("format") != V8_BASE_EXCLUSION_FORMAT:
        raise RuntimeError("unknown factual base exclusion contract")
    _signed_mapping_sha256(exclusion_contract, "contract_sha256")
    if exclusion_contract.get("base_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("factual base exclusion contract checkpoint mismatch")
    # A caller-created JSON object plus its own digest is not an independent
    # trust root.  Until the factual checkpoint itself records authoritative
    # train/validation/test identities, no standalone exclusion document can
    # upgrade a fold to ``proven``.
    return {
        "status": "unproven_development_only",
        "reason": "standalone_self_hashed_exclusion_is_not_an_authoritative_trust_root",
        "legacy_old100_holdout_groups": [],
        "submitted_base_exclusion_contract_sha256": exclusion_contract[
            "contract_sha256"
        ],
    }


def build_outer_fold_payloads(
    *,
    fold: Mapping[str, Any],
    training_groups: Sequence[counterfactual.BranchGroup],
    holdout_groups: Sequence[counterfactual.BranchGroup],
    context: Mapping[str, Any],
    exclusion_contract: Mapping[str, Any] | None = None,
    legacy_old100_groups: Sequence[str] = (),
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, Any]:
    fold_id = int(fold["outer_fold_id"])
    training_keys = sorted(map(str, fold["training_groups"]))
    holdout_keys = sorted(map(str, fold["oof_holdout_groups"]))
    if sorted(group.logical_key for group in training_groups) != training_keys:
        raise RuntimeError("outer-training loader returned the wrong logical groups")
    if sorted(group.logical_key for group in holdout_groups) != holdout_keys:
        raise RuntimeError("outer-holdout loader returned the wrong logical groups")
    repairs = fit_outer_training_repairs(
        training_groups,
        object_mean=context["object_mean"],
        object_std=context["object_std"],
    )
    train_records, train_audit = materialize_group_records(
        sorted(training_groups, key=lambda group: group.logical_key),
        split_role="outer_training",
        outer_fold_id=fold_id,
        model=context["model"],
        duration_contract=repairs["duration_contract"],
        object_fallback_normalized=repairs["object_fallback_normalized"],
        object_mean=context["object_mean"],
        object_std=context["object_std"],
        device=device,
    )
    _validate_split_records(train_records, training_keys, "outer_training")
    duration_scale_contract = fit_duration_residual_uncertainty_contract(train_records)
    # Only after all train-derived repair contracts and records are frozen do
    # we materialise target-fold labels into their separate evaluation payload.
    holdout_records, holdout_audit = materialize_group_records(
        sorted(holdout_groups, key=lambda group: group.logical_key),
        split_role="outer_holdout",
        outer_fold_id=fold_id,
        model=context["model"],
        duration_contract=repairs["duration_contract"],
        object_fallback_normalized=repairs["object_fallback_normalized"],
        object_mean=context["object_mean"],
        object_std=context["object_std"],
        device=device,
    )
    _validate_split_records(holdout_records, holdout_keys, "outer_holdout")
    exclusion = base_exclusion_status(
        checkpoint_sha256=context["checkpoint_sha256"],
        holdout_groups=holdout_keys,
        exclusion_contract=exclusion_contract,
        legacy_old100_groups=legacy_old100_groups,
        checkpoint_contract=context.get("checkpoint_contract"),
        authorized_target_groups=[
            str(row["logical_key"])
            for row in context.get("collection_audit", {}).get("groups", [])
        ],
        v7_trust_root=context.get("collection_audit", {}).get("v7_trust_root"),
    )
    label_contract = context["label_derivation_contract"]
    uncertainty_contract = uncertainty_materialization_contract()
    provenance = {
        "outer_fold_id": fold_id,
        "outer_training_groups": training_keys,
        "outer_training_groups_sha256": logical_group_list_sha256(training_keys),
        "oof_holdout_groups": holdout_keys,
        "oof_holdout_groups_sha256": logical_group_list_sha256(holdout_keys),
        "target_outer_fold_labels_used": False,
        "factual_outputs_frozen": True,
        "base_checkpoint": context["checkpoint_path"],
        "base_checkpoint_sha256": context["checkpoint_sha256"],
        "base_target_outer_fold_exclusion_status": exclusion["status"],
        "base_exclusion_audit": exclusion,
        "event_spec": context["event_spec_path"],
        "event_spec_sha256": context["event_spec_sha256"],
        "label_derivation_contract": label_contract,
        "label_derivation_sha256": label_contract["label_derivation_sha256"],
        "duration_baseline_contract": repairs["duration_contract"],
        "duration_baseline_contract_sha256": repairs["duration_contract"][
            "duration_baseline_contract_sha256"
        ],
        "duration_laplace_scale_contract": duration_scale_contract,
        "duration_laplace_scale_contract_sha256": duration_scale_contract[
            "contract_sha256"
        ],
        "object_fallback_contract": repairs["object_fallback_contract"],
        "object_fallback_contract_sha256": repairs["object_fallback_contract"][
            "object_fallback_contract_sha256"
        ],
        "object_mode": V8_OBJECT_MODE,
        "object_pose_quality_status": (
            "unavailable_schema5_collector_has_no_quality_field_fail_closed"
        ),
        "uncertainty_materialization_contract": uncertainty_contract,
        "uncertainty_materialization_contract_sha256": uncertainty_contract[
            "uncertainty_materialization_contract_sha256"
        ],
        "fresh_confirmation_data_or_labels_read": False,
    }
    training_payload = {
        "format": V8_TRAINING_INPUT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": int(context["config"].semantic_dim)},
        "batches": train_records,
        "provenance": provenance,
        "materialization_audit": train_audit,
    }
    training_payload["payload_sha256"] = structured_payload_sha256(training_payload)
    validate_v8_training_payload(training_payload)
    holdout_payload = {
        "format": V8_HOLDOUT_INPUT_FORMAT,
        "schema_version": 5,
        "config": {"transition_dim": int(context["config"].semantic_dim)},
        "batches": holdout_records,
        "provenance": {
            **provenance,
            "split_role": "outer_holdout_evaluation_only",
            "holdout_labels_used_for_duration_or_object_fit": False,
            "holdout_labels_present_only_in_separate_artifact": True,
        },
        "materialization_audit": holdout_audit,
    }
    holdout_payload["payload_sha256"] = structured_payload_sha256(holdout_payload)
    return {
        "training_payload": training_payload,
        "holdout_payload": holdout_payload,
        "fold_manifest": {
            "outer_fold_id": fold_id,
            "training_groups": training_keys,
            "training_groups_sha256": provenance["outer_training_groups_sha256"],
            "oof_holdout_groups": holdout_keys,
            "oof_holdout_groups_sha256": provenance[
                "oof_holdout_groups_sha256"
            ],
            "base_exclusion_audit": exclusion,
            "duration_baseline_contract_sha256": provenance[
                "duration_baseline_contract_sha256"
            ],
            "object_fallback_contract_sha256": provenance[
                "object_fallback_contract_sha256"
            ],
            "training_materialization_audit": train_audit,
            "holdout_materialization_audit": holdout_audit,
            "target_outer_fold_labels_used_for_training": False,
            "fresh_confirmation_data_or_labels_read": False,
        },
    }


def _extract_legacy_old100_groups(manifest: Mapping[str, Any]) -> list[str]:
    source = manifest.get("source_contract")
    if not isinstance(source, Mapping):
        return []
    for key in ("legacy_old100_groups", "old100_groups", "legacy_development_groups"):
        value = source.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return sorted(map(str, value))
    return []


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
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


def materialize_oof_inputs(
    *,
    data_inputs: Sequence[Path],
    checkpoint_path: Path,
    event_spec_path: Path,
    oof_manifest: Mapping[str, Any],
    output_dir: Path,
    exclusion_contract: Mapping[str, Any] | None = None,
    device: torch.device | str = torch.device("cpu"),
    regression_persistence_steps: int = counterfactual.DEFAULT_REGRESSION_PERSISTENCE_STEPS,
) -> dict[str, Any]:
    """Production builder: identity scan first, then fold-scoped label loads."""

    reject_fresh_sources(data_inputs)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory {output_dir}")
    descriptors = counterfactual.scan_group_descriptors(data_inputs)
    if not descriptors or any(descriptor.schema_version != 5 for descriptor in descriptors):
        raise RuntimeError("v8 formal materialization requires schema5 groups only")
    folds = normalize_oof_owner_folds(
        oof_manifest,
        [descriptor.logical_key for descriptor in descriptors],
        require_formal_size=True,
    )
    collection_audit = validate_development_collection_contract(
        data_inputs=data_inputs,
        descriptors=descriptors,
        event_spec_sha256=sha256_path(event_spec_path.resolve()),
        oof_manifest=oof_manifest,
    )
    source_contract = oof_manifest.get("source_contract")
    if not isinstance(source_contract, Mapping) or (
        source_contract.get("pretrained_sha256") != sha256_path(checkpoint_path.resolve())
        or source_contract.get("event_spec_sha256")
        != sha256_path(event_spec_path.resolve())
        or Path(str(source_contract.get("pretrained", ""))).resolve()
        != checkpoint_path.resolve()
        or Path(str(source_contract.get("event_spec", ""))).resolve()
        != event_spec_path.resolve()
    ):
        raise RuntimeError("signed OOF checkpoint/event-spec source binding changed")
    context = load_frozen_factual_context(
        checkpoint_path,
        event_spec_path,
        device=device,
        regression_persistence_steps=regression_persistence_steps,
    )
    context["collection_audit"] = collection_audit
    descriptor_by_key = {descriptor.logical_key: descriptor for descriptor in descriptors}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.materializing.", dir=output_dir.parent
        )
    )
    legacy_old100 = _extract_legacy_old100_groups(oof_manifest)
    fold_rows: list[dict[str, Any]] = []
    try:
      for fold in folds:
        # The target descriptors have still exposed identity attrs only here.
        training_groups = counterfactual.load_descriptor_groups(
            [descriptor_by_key[key] for key in fold["training_groups"]],
            context["config"],
            context["object_names"],
            context["body_to_id"],
            context["policy_to_id"],
            calibrations=context["calibrations"],
            regression_persistence_steps=regression_persistence_steps,
            expected_event_spec_sha256=context["event_spec_sha256"],
        )
        # Fit every repair and materialise the complete training artifact before
        # opening any target-fold label dataset.
        repairs = fit_outer_training_repairs(
            training_groups,
            object_mean=context["object_mean"],
            object_std=context["object_std"],
        )
        train_records, train_audit = materialize_group_records(
            sorted(training_groups, key=lambda group: group.logical_key),
            split_role="outer_training",
            outer_fold_id=fold["outer_fold_id"],
            model=context["model"],
            duration_contract=repairs["duration_contract"],
            object_fallback_normalized=repairs["object_fallback_normalized"],
            object_mean=context["object_mean"],
            object_std=context["object_std"],
            device=device,
        )
        _validate_split_records(
            train_records, fold["training_groups"], "outer_training"
        )
        duration_scale_contract = fit_duration_residual_uncertainty_contract(
            train_records
        )
        holdout_groups = counterfactual.load_descriptor_groups(
            [descriptor_by_key[key] for key in fold["oof_holdout_groups"]],
            context["config"],
            context["object_names"],
            context["body_to_id"],
            context["policy_to_id"],
            calibrations=context["calibrations"],
            regression_persistence_steps=regression_persistence_steps,
            expected_event_spec_sha256=context["event_spec_sha256"],
        )
        # Reuse the already-frozen train contracts.  build_outer_fold_payloads
        # repeats fitting, so construct the two payloads here without giving
        # that helper a chance to observe target labels during contract fitting.
        holdout_records, holdout_audit = materialize_group_records(
            sorted(holdout_groups, key=lambda group: group.logical_key),
            split_role="outer_holdout",
            outer_fold_id=fold["outer_fold_id"],
            model=context["model"],
            duration_contract=repairs["duration_contract"],
            object_fallback_normalized=repairs["object_fallback_normalized"],
            object_mean=context["object_mean"],
            object_std=context["object_std"],
            device=device,
        )
        _validate_split_records(
            holdout_records, fold["oof_holdout_groups"], "outer_holdout"
        )
        exclusion = base_exclusion_status(
            checkpoint_sha256=context["checkpoint_sha256"],
            holdout_groups=fold["oof_holdout_groups"],
            exclusion_contract=exclusion_contract,
            legacy_old100_groups=legacy_old100,
            checkpoint_contract=context["checkpoint_contract"],
            authorized_target_groups=[
                str(row["logical_key"])
                for row in collection_audit["groups"]
            ],
            v7_trust_root=collection_audit["v7_trust_root"],
        )
        label_contract = context["label_derivation_contract"]
        uncertainty_contract = uncertainty_materialization_contract()
        provenance = {
            "outer_fold_id": fold["outer_fold_id"],
            "outer_training_groups": fold["training_groups"],
            "outer_training_groups_sha256": fold["training_groups_sha256"],
            "oof_holdout_groups": fold["oof_holdout_groups"],
            "oof_holdout_groups_sha256": fold["oof_holdout_groups_sha256"],
            "target_outer_fold_labels_used": False,
            "factual_outputs_frozen": True,
            "base_checkpoint": context["checkpoint_path"],
            "base_checkpoint_sha256": context["checkpoint_sha256"],
            "base_target_outer_fold_exclusion_status": exclusion["status"],
            "base_exclusion_audit": exclusion,
            "event_spec": context["event_spec_path"],
            "event_spec_sha256": context["event_spec_sha256"],
            "label_derivation_contract": label_contract,
            "label_derivation_sha256": label_contract["label_derivation_sha256"],
            "duration_baseline_contract": repairs["duration_contract"],
            "duration_baseline_contract_sha256": repairs["duration_contract"][
                "duration_baseline_contract_sha256"
            ],
            "duration_laplace_scale_contract": duration_scale_contract,
            "duration_laplace_scale_contract_sha256": duration_scale_contract[
                "contract_sha256"
            ],
            "object_fallback_contract": repairs["object_fallback_contract"],
            "object_fallback_contract_sha256": repairs[
                "object_fallback_contract"
            ]["object_fallback_contract_sha256"],
            "object_mode": V8_OBJECT_MODE,
            "object_pose_quality_status": (
                "unavailable_schema5_collector_has_no_quality_field_fail_closed"
            ),
            "uncertainty_materialization_contract": uncertainty_contract,
            "uncertainty_materialization_contract_sha256": uncertainty_contract[
                "uncertainty_materialization_contract_sha256"
            ],
            "fresh_confirmation_data_or_labels_read": False,
            "source_collection_audit": collection_audit,
        }
        train_payload = {
            "format": V8_TRAINING_INPUT_FORMAT,
            "schema_version": 5,
            "config": {"transition_dim": int(context["config"].semantic_dim)},
            "batches": train_records,
            "provenance": provenance,
            "materialization_audit": train_audit,
        }
        train_payload["payload_sha256"] = structured_payload_sha256(train_payload)
        validate_v8_training_payload(train_payload)
        holdout_payload = {
            "format": V8_HOLDOUT_INPUT_FORMAT,
            "schema_version": 5,
            "config": {"transition_dim": int(context["config"].semantic_dim)},
            "batches": holdout_records,
            "provenance": {
                **provenance,
                "split_role": "outer_holdout_evaluation_only",
                "holdout_labels_used_for_duration_or_object_fit": False,
                "holdout_labels_present_only_in_separate_artifact": True,
            },
            "materialization_audit": holdout_audit,
        }
        holdout_payload["payload_sha256"] = structured_payload_sha256(
            holdout_payload
        )
        train_name = f"fold_{fold['outer_fold_id']}_train.pt"
        holdout_name = f"fold_{fold['outer_fold_id']}_holdout.pt"
        train_path = staging_dir / train_name
        holdout_path = staging_dir / holdout_name
        _atomic_torch_save(train_path, train_payload)
        _atomic_torch_save(holdout_path, holdout_payload)
        fold_rows.append(
            {
                "outer_fold_id": fold["outer_fold_id"],
                "training_groups": fold["training_groups"],
                "training_groups_sha256": fold["training_groups_sha256"],
                "oof_holdout_groups": fold["oof_holdout_groups"],
                "oof_holdout_groups_sha256": fold["oof_holdout_groups_sha256"],
                "train_artifact": str(output_dir / train_name),
                "train_artifact_sha256": sha256_path(train_path),
                "train_payload_sha256": train_payload["payload_sha256"],
                "holdout_artifact": str(output_dir / holdout_name),
                "holdout_artifact_sha256": sha256_path(holdout_path),
                "holdout_payload_sha256": holdout_payload["payload_sha256"],
                "base_exclusion_audit": exclusion,
                "target_outer_fold_labels_used_for_training": False,
            }
        )
      manifest = {
        "format": V8_OOF_MATERIALIZATION_FORMAT,
        "status": "complete_development_only",
        "source_oof_preregistration_sha256": oof_manifest[
            "preregistration_sha256"
        ],
        "base_checkpoint_sha256": context["checkpoint_sha256"],
        "event_spec_sha256": context["event_spec_sha256"],
        "label_derivation_sha256": context["label_derivation_contract"][
            "label_derivation_sha256"
        ],
        "source_collection_audit": collection_audit,
        "development_groups": sorted(descriptor_by_key),
        "development_groups_sha256": logical_group_list_sha256(
            sorted(descriptor_by_key)
        ),
        "folds": fold_rows,
        "fresh_confirmation_data_or_labels_read": False,
        "remote_write_performed": False,
        "authorization_guard_changed": False,
        "timing_scope": "adaptive_development_only_designed_after_v7_collection_started",
        "prospective_claim_for_v8": False,
        "final_five_checkpoint_holdout_bundle_status": (
            "pending_training_evaluator_must_bind_all_five_checkpoint_and_holdout_shas"
        ),
    }
      manifest["materialization_sha256"] = canonical_sha256(manifest)
      _atomic_json(staging_dir / "materialization_manifest.json", manifest)
      os.replace(staging_dir, output_dir)
      return manifest
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _synthetic_group(
    key: str,
    *,
    config: EventWorldModelConfig,
    duration_offset: float,
) -> counterfactual.BranchGroup:
    count, horizon = 4, 3
    synthetic_seed = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:4], "big"
    )
    rng = np.random.default_rng(synthetic_seed)
    success = np.asarray([0, 1, 0, 1], dtype=np.float32)
    regress = np.asarray([0, 1, 1, 0], dtype=bool)
    recovery = np.asarray([0, 1, 0, 0], dtype=bool)
    continuation_hidden = rng.normal(
        size=(1, config.state_input_dim)
    ).astype(np.float16)
    continuation_post_hidden = rng.normal(
        size=(1, config.state_input_dim)
    ).astype(np.float16)
    continuation_history, continuation_history_mask = (
        counterfactual.fixed_causal_hidden_window(continuation_hidden)
    )
    continuation_post_history, continuation_post_history_mask = (
        counterfactual.fixed_causal_hidden_window(
            np.concatenate(
                [continuation_hidden, continuation_post_hidden], axis=0
            )
        )
    )
    continuation = {
        "hidden_t": continuation_history[None],
        "history_mask": continuation_history_mask[None],
        "action_chunks": rng.normal(size=(1, horizon, config.action_dim)).astype(np.float32),
        "action_mask": np.ones((1, horizon), dtype=bool),
        "proprio": rng.normal(size=(1, config.proprio_dim)).astype(np.float32),
        "current_event_id": np.asarray([1], dtype=np.int64),
        "next_event_id": np.asarray([0], dtype=np.int64),
        "clock_event_id": np.asarray([1], dtype=np.int64),
        "next_reached_event_id": np.asarray([1], dtype=np.int64),
        "current_predicates": np.asarray([[1, 1, 0, 0, 0]], dtype=np.float32),
        "post_predicates": np.asarray([[1, 0, 0, 0, 0]], dtype=np.float32),
        "relative_transition_id": np.asarray([3], dtype=np.int64),
        "structured_mask": np.ones(1, dtype=bool),
        "duration": np.asarray([5 + duration_offset], dtype=np.float32),
        "duration_observed": np.ones(1, dtype=np.float32),
        "object_delta": np.asarray([[0.01, 0.0, 0.0]], dtype=np.float32),
        "post_hidden": continuation_post_history[None],
        "post_history_mask": continuation_post_history_mask[None],
        "trajectory_regress": np.ones(1, dtype=bool),
        "trajectory_recovery": np.zeros(1, dtype=bool),
    }
    hidden = rng.normal(size=(count, config.state_input_dim)).astype(np.float16)
    post_hidden = rng.normal(size=(count, config.state_input_dim)).astype(np.float16)
    root_history_pairs = [
        counterfactual.fixed_causal_hidden_window(hidden[index : index + 1])
        for index in range(count)
    ]
    root_post_history_pairs = [
        counterfactual.fixed_causal_hidden_window(
            np.stack([hidden[index], post_hidden[index]], axis=0)
        )
        for index in range(count)
    ]
    return counterfactual.BranchGroup(
        path=f"/synthetic/{key}.hdf5",
        schema_version=5,
        logical_key=key,
        seed=int(synthetic_seed % 1_000_000),
        task="move_can_pot",
        body="piper",
        policy="openvla",
        candidate_names=[
            "deterministic",
            "sample_blend_0.250",
            "sample_blend_0.500",
            "sample_blend_0.750",
        ],
        hidden=hidden,
        actions=rng.normal(size=(count, horizon, config.action_dim)).astype(np.float32),
        action_mask=np.ones((count, horizon), dtype=bool),
        proprio=rng.normal(size=(count, config.proprio_dim)).astype(np.float32),
        current_event_id=np.asarray([0, 1, 1, 0], dtype=np.int64),
        next_event_id=np.asarray([1, 2, 0, 4], dtype=np.int64),
        clock_event_id=np.asarray([0, 1, 1, 0], dtype=np.int64),
        next_reached_event_id=np.asarray([1, 2, 1, 4], dtype=np.int64),
        current_predicates=np.zeros((count, config.num_predicates), dtype=np.float32),
        post_predicates=np.zeros((count, config.num_predicates), dtype=np.float32),
        relative_transition_id=np.asarray([1, 1, 3, 2], dtype=np.int64),
        structured_mask=np.ones(count, dtype=bool),
        duration=np.asarray([4, 6, 8, 5], dtype=np.float32) + duration_offset,
        duration_observed=np.asarray([1, 1, 0, 1], dtype=np.float32),
        success=success,
        outcome_id=success.astype(np.int64),
        trajectory_regress=regress,
        trajectory_recovery=recovery,
        steps=np.asarray([10, 12, 14, 9], dtype=np.float32),
        object_delta=np.asarray(
            [[0, 0, 0], [0.02, 0, 0], [0, 0.01, 0], [0, 0, 0]],
            dtype=np.float32,
        ),
        post_hidden=post_hidden,
        dense_mask=np.ones(count, dtype=bool),
        candidate_distance=np.asarray([0, 0.2, 0.4, 0.6], dtype=np.float32),
        continuation=continuation,
        body_id=0,
        policy_id=0,
        history_hidden=np.stack([item[0] for item in root_history_pairs]),
        history_mask=np.stack([item[1] for item in root_history_pairs]),
        post_history_hidden=np.stack(
            [item[0] for item in root_post_history_pairs]
        ),
        post_history_mask=np.stack(
            [item[1] for item in root_post_history_pairs]
        ),
    )


def cpu_materializer_smoke(seed: int = 20260827) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = EventWorldModelConfig(
        state_input_dim=12,
        action_dim=4,
        proprio_dim=3,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=6,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=True,
        dropout=0.0,
    )
    model = ActionConditionedEventWorldModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    context = {
        "model": model,
        "config": config,
        "checkpoint_path": "/synthetic/factual.pt",
        "checkpoint_sha256": "a" * 64,
        "event_spec_path": "/synthetic/event_spec.json",
        "event_spec_sha256": "b" * 64,
        "object_mean": np.zeros(3, dtype=np.float32),
        "object_std": np.ones(3, dtype=np.float32),
        "label_derivation_contract": {
            "format": "synthetic_label_contract",
            "label_derivation_sha256": "c" * 64,
        },
    }
    train = [
        _synthetic_group(f"move_can_pot|piper|{index}", config=config, duration_offset=float(index))
        for index in range(4)
    ]
    holdout = [
        _synthetic_group("move_can_pot|piper|99", config=config, duration_offset=1000.0)
    ]
    fold = {
        "outer_fold_id": 0,
        "training_groups": sorted(group.logical_key for group in train),
        "oof_holdout_groups": sorted(group.logical_key for group in holdout),
    }
    result = build_outer_fold_payloads(
        fold=fold,
        training_groups=train,
        holdout_groups=holdout,
        context=context,
        legacy_old100_groups=[holdout[0].logical_key],
        device="cpu",
    )
    train_payload = result["training_payload"]
    holdout_payload = result["holdout_payload"]
    train_max_baseline = max(
        float(record["duration_baseline_log1p"].max())
        for record in train_payload["batches"]
    )
    holdout_max_baseline = max(
        float(record["duration_baseline_log1p"].max())
        for record in holdout_payload["batches"]
    )
    return {
        "status": "passed",
        "device": "cpu",
        "cuda_used": False,
        "training_groups": len(train_payload["batches"]),
        "holdout_groups": len(holdout_payload["batches"]),
        "training_payload_valid": True,
        "target_outer_fold_labels_used": False,
        "factual_state_bit_exact": result["fold_manifest"][
            "training_materialization_audit"
        ]["factual_state_bit_exact"],
        "base_exclusion_status": train_payload["provenance"][
            "base_target_outer_fold_exclusion_status"
        ],
        "train_max_duration_baseline": train_max_baseline,
        "holdout_max_duration_baseline": holdout_max_baseline,
        "extreme_holdout_duration_did_not_change_baseline": (
            holdout_max_baseline == train_max_baseline
        ),
        "learned_object_output_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preregister-owner-manifest", action="store_true")
    parser.add_argument("--data", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--event-spec", type=Path)
    parser.add_argument("--oof-manifest", type=Path)
    parser.add_argument("--base-exclusion-contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--regression-persistence-steps",
        type=int,
        default=counterfactual.DEFAULT_REGRESSION_PERSISTENCE_STEPS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        forbidden = (
            args.data,
            args.checkpoint,
            args.event_spec,
            args.oof_manifest,
            args.output,
        )
        if any(value is not None for value in forbidden) or args.device != "cpu":
            raise ValueError("materializer smoke is CPU-only and accepts no paths")
        print(json.dumps(cpu_materializer_smoke(), sort_keys=True))
        return
    if args.preregister_owner_manifest:
        if (
            args.data is None
            or len(args.data) != 1
            or args.checkpoint is None
            or args.event_spec is None
            or args.output is None
            or args.oof_manifest is not None
            or args.base_exclusion_contract is not None
            or args.device != "cpu"
        ):
            raise ValueError(
                "owner preregistration requires one data root/checkpoint/event-spec/output "
                "and is CPU-only"
            )
        result = preregister_v8_owner_manifest(
            data_root=args.data[0],
            checkpoint_path=args.checkpoint,
            event_spec_path=args.event_spec,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(args.output.resolve()),
                    "preregistration_sha256": result["preregistration_sha256"],
                    "timing_scope": result["timing_scope"],
                    "fresh_confirmation_data_or_labels_read": False,
                },
                sort_keys=True,
            )
        )
        return
    if any(
        value is None
        for value in (
            args.data,
            args.checkpoint,
            args.event_spec,
            args.oof_manifest,
            args.output,
        )
    ):
        raise ValueError("materialization requires data/checkpoint/event-spec/manifest/output")
    oof_manifest = _load_json(args.oof_manifest)
    exclusion = (
        _load_json(args.base_exclusion_contract)
        if args.base_exclusion_contract is not None
        else None
    )
    result = materialize_oof_inputs(
        data_inputs=args.data,
        checkpoint_path=args.checkpoint,
        event_spec_path=args.event_spec,
        oof_manifest=oof_manifest,
        output_dir=args.output,
        exclusion_contract=exclusion,
        device=args.device,
        regression_persistence_steps=args.regression_persistence_steps,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "materialization_sha256": result["materialization_sha256"],
                "fresh_confirmation_data_or_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "V8_BASE_EXCLUSION_FORMAT",
    "V8_HOLDOUT_INPUT_FORMAT",
    "V8_OOF_MATERIALIZATION_FORMAT",
    "V8_OWNER_MANIFEST_FORMAT",
    "base_exclusion_status",
    "build_outer_fold_payloads",
    "cpu_materializer_smoke",
    "fit_outer_training_repairs",
    "fit_duration_residual_uncertainty_contract",
    "load_frozen_factual_context",
    "logical_group_list_sha256",
    "materialize_group_records",
    "materialize_oof_inputs",
    "normalize_oof_owner_folds",
    "preregister_v8_owner_manifest",
    "reject_fresh_sources",
    "validate_development_collection_contract",
]
