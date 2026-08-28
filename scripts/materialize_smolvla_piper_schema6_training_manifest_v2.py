#!/usr/bin/env python3
"""Label-blind aggregation of frozen Piper schema-6 collection roots.

The aggregator validates JSON authority/manifest/final-receipt lineage and
per-seed registry/pose metadata.  HDF5 files are treated as opaque sealed
artifacts: only lstat/read-only/path checks and cross-contract recorded SHA256
agreement are performed.  No HDF5 byte or label is opened by this process.

Target membership is frozen before collection roots are inspected.  Exactly
adaptation80 and validation50 are eligible; target evaluation identities are
never emitted and a supplied evaluation root fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from etsf_schema6_pose_quality import (
    registry_sha256,
    spec_sha256,
    validate_registry,
    validate_spec,
)
from preregister_smolvla_piper_schema6_multiseed_collection_v2 import (
    GROUP_RECEIPT_FORMAT,
    GROUP_RECEIPT_STATUS,
    FORMAT as MULTISEED_PREREGISTRATION_FORMAT,
    validate_target_seed_manifest,
    validate_preregistration,
)
from run_smolvla_piper_schema6_multiseed_v2 import (
    RESET_RECEIPT_FORMAT,
    RESET_RECEIPT_STATUS,
)


FORMAT = "etsf_smolvla_piper_schema6_training_manifest_aggregator_v2"
COMPLETE_STATUS = "complete_label_blind_training_inputs_frozen"
INSUFFICIENT_STATUS = "insufficient_data_no_training_authorized"
TRAINER_MANIFEST_FORMAT = "etsf_smolvla_piper_schema6_training_manifest_v1"
TARGET_PARTITION_FORMAT = "etsf_smolvla_piper_schema6_target_partition_v2"
EXTERNAL_SPLIT_FORMAT = "etsf_smolvla_piper_schema6_external_group_split_v2"
EXPECTED_FORMAT = "etsf_smolvla_piper_schema6_expected_manifest_split_v2"
COLLECTION_AUTHORITY_FORMAT = "etsf_smolvla_piper_schema6_collection_authority_v2"
COLLECTION_MANIFEST_FORMAT = "etsf_smolvla_piper_schema6_collection_manifest_v2"
COLLECTION_FINAL_FORMAT = "etsf_smolvla_piper_schema6_collection_final_receipt_v2"
COLLECTION_STATUS = "complete_four_candidate_schema6_group"
TASK = "move_can_pot"
BODY = "piper"
POLICY = "smolvla"
TARGET_BODY = "piper_piper_0.6"
CANDIDATE_INDICES = [0, 1, 2, 3]
ADAPTATION_COUNT = 80
TARGET_VALIDATION_COUNT = 50
TRAIN_COUNT = 60
INTERNAL_VALIDATION_COUNT = 20
SEALED_TEST_COUNT = 50
EXTERNAL_SPLIT_SEED = 1701
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
HDF_SUFFIXES = (".h5", ".hdf", ".hdf5")
SHA_ALPHABET = frozenset("0123456789abcdef")
CAPABILITY_CONTRACT = {
    "fresh_inputs_used": False,
    "evaluation_split": False,
    "real_robot_execution": False,
    "performance_or_transfer_claim_authorized": False,
    "four_candidate_branches_complete": True,
}

NATIVE_V2_RESET_FIELDS = {
    "format", "status", "preregistration_sha256", "command_sha256",
    "split", "ordinal", "requested_seed", "resolved_seed", "pair_id",
    "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
    "initial_commanded_drive_target_sha256", "object_registry_sha256",
    "pose_spec_sha256", "identity_validation_count_before_policy_query",
    "policy_queries_before_reset_receipt", "evaluation_execution_authorized",
    "protected_inputs_read", "reset_receipt_sha256",
}
NATIVE_V2_GROUP_FIELDS = {
    "format", "status", "preregistration_sha256", "command_sha256",
    "split", "ordinal", "requested_seed", "resolved_seed", "pair_id",
    "candidate_original_indices", "branch_records",
    "per_seed_reset_receipt_sha256", "object_registry_sha256",
    "pose_spec_sha256", "group_file_sha256", "group_receipt_sha256",
}


class TrainingManifestContractError(RuntimeError):
    """The label-blind aggregation chain cannot be proved."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA_ALPHABET)
    )


def _contains_sensitive(path: PurePath) -> bool:
    return any(
        token in component.casefold()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    text = os.fspath(value)
    if not text or "\x00" in text:
        raise TrainingManifestContractError(f"{role} path is empty/invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(text)))
    if _contains_sensitive(PurePath(lexical)):
        raise TrainingManifestContractError(f"{role} path contains a forbidden component")
    resolved = lexical.resolve(strict=False)
    if _contains_sensitive(PurePath(resolved)):
        raise TrainingManifestContractError(
            f"{role} resolved path contains a forbidden component"
        )
    return resolved


def _lstat_regular(path: Path, role: str, *, read_only: bool = True) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TrainingManifestContractError(f"{role} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TrainingManifestContractError(f"{role} must be a non-symlink regular file")
    if read_only and metadata.st_mode & 0o222:
        raise TrainingManifestContractError(f"{role} is not frozen read-only")
    return metadata


def _frozen_directory(value: str | os.PathLike[str], role: str) -> Path:
    path = safe_path(value, role)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TrainingManifestContractError(f"{role} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise TrainingManifestContractError(f"{role} must be a non-symlink directory")
    if metadata.st_mode & 0o222:
        raise TrainingManifestContractError(f"{role} is not frozen read-only")
    return path


def metadata_file_sha256(path: Path, role: str) -> str:
    """Hash JSON/code metadata only; HDF paths are categorically rejected."""

    if path.suffix.casefold() in HDF_SUFFIXES:
        raise TrainingManifestContractError(f"{role} HDF byte access is forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, role: str) -> dict[str, Any]:
    _lstat_regular(path, role)
    if path.suffix.casefold() != ".json":
        raise TrainingManifestContractError(f"{role} must be JSON metadata")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingManifestContractError(f"{role} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrainingManifestContractError(f"{role} must contain a JSON object")
    _audit_embedded_paths(value, role)
    return value


def _audit_embedded_paths(value: Any, role: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _audit_embedded_paths(child, f"{role}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _audit_embedded_paths(child, f"{role}[{index}]")
    elif isinstance(value, str):
        looks_like_path = value.startswith(("/", "./", "../")) or "\\" in value
        if looks_like_path and _contains_sensitive(PurePath(value)):
            raise TrainingManifestContractError(f"{role} embeds a forbidden path")


def _verify_signature(value: Mapping[str, Any], key: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not _is_sha256(recorded) or recorded != canonical_sha256(unsigned):
        raise TrainingManifestContractError(f"{role} logical SHA256 mismatch")
    return str(recorded)


def _inside(root: Path, relative: Any, role: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise TrainingManifestContractError(f"{role} path must be relative")
    lexical = PurePath(relative)
    if ".." in lexical.parts or _contains_sensitive(lexical):
        raise TrainingManifestContractError(f"{role} path escapes or is forbidden")
    path = safe_path(root / relative, role)
    if path.parent != root and root not in path.parents:
        raise TrainingManifestContractError(f"{role} escaped collection root")
    return path


def _exact_identity(value: Any, role: str) -> dict[str, Any]:
    fields = {
        "split", "ordinal", "requested_seed", "resolved_seed", "pair_id",
        "task", "body", "policy",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrainingManifestContractError(f"{role} identity fields changed")
    identity = dict(value)
    if (
        identity["split"] not in ("adaptation", "validation", "evaluation")
        or type(identity["ordinal"]) is not int
        or identity["ordinal"] < 0
        or type(identity["requested_seed"]) is not int
        or identity["requested_seed"] < 0
        or type(identity["resolved_seed"]) is not int
        or identity["resolved_seed"] < 0
        or not _is_sha256(identity["pair_id"])
        or identity["task"] != TASK
        or identity["body"] != BODY
        or identity["policy"] != POLICY
    ):
        raise TrainingManifestContractError(f"{role} identity is invalid")
    return identity


def _record(value: Any, role: str, *, logical: bool) -> dict[str, str]:
    fields = {"path", "file_sha256"} | ({"logical_sha256"} if logical else set())
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TrainingManifestContractError(f"{role} record fields changed")
    result = {key: str(value[key]) for key in fields}
    if not _is_sha256(result["file_sha256"]) or (
        logical and not _is_sha256(result["logical_sha256"])
    ):
        raise TrainingManifestContractError(f"{role} SHA256 is invalid")
    return result


def _validate_registry_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        registry = validate_registry(value)
    except Exception as exc:
        raise TrainingManifestContractError("object registry is invalid") from exc
    objects = registry["objects"]
    expected = (
        ("can", "task_attr=can;sapien_actor_name=", "105_sauce-can/base", "manipulated"),
        ("pot", "task_attr=pot;sapien_actor_name=", "060_kitchenpot/base", "receptacle"),
    )
    if len(objects) != 2:
        raise TrainingManifestContractError("object registry must contain exactly can then pot")
    for item, (name, actor_prefix, asset_prefix, role) in zip(objects, expected, strict=True):
        suffix = str(item["asset_model_id"])[len(asset_prefix):]
        if (
            item["name"] != name
            or not str(item["stable_sim_actor_id"]).startswith(actor_prefix)
            or not str(item["stable_sim_actor_id"])[len(actor_prefix):]
            or not str(item["asset_model_id"]).startswith(asset_prefix)
            or not suffix.isdecimal()
            or str(int(suffix)) != suffix
            or item["role"] != role
            or item["is_static"] is not False
        ):
            raise TrainingManifestContractError(f"{name} actor/asset registry changed")
    return registry


@dataclass(frozen=True)
class CollectionDescriptor:
    root: Path
    identity: dict[str, Any]
    group_path: Path
    group_file_sha256: str
    authority_file_sha256: str
    authority_logical_sha256: str
    manifest_file_sha256: str
    manifest_logical_sha256: str
    final_receipt_file_sha256: str
    final_receipt_logical_sha256: str
    object_registry_file_sha256: str
    object_registry_logical_sha256: str
    pose_spec_file_sha256: str
    pose_spec_logical_sha256: str


def validate_native_v2_collection_root(
    root_value: Path,
    *,
    expected_command: Mapping[str, Any],
    expected_preregistration_file_sha256: str,
    expected_preregistration_logical_sha256: str,
    expected_event_spec_sha256: str,
    expected_collector_lineage_sha256: str,
) -> CollectionDescriptor:
    """Validate one Phase-2 native seed root without opening its HDF bytes."""

    root = _frozen_directory(root_value, "schema6 native-v2 seed root")
    command_unsigned = dict(expected_command)
    command_recorded = command_unsigned.pop("command_sha256", None)
    if (
        not _is_sha256(command_recorded)
        or command_recorded != canonical_sha256(command_unsigned)
    ):
        raise TrainingManifestContractError("native-v2 command logical SHA256 mismatch")
    outputs = expected_command.get("outputs")
    bindings = expected_command.get("bindings")
    if (
        not isinstance(outputs, Mapping)
        or set(outputs) != {
            "seed_root", "per_seed_reset_receipt", "group_hdf5",
            "completed_group_receipt",
        }
        or not isinstance(bindings, Mapping)
        or bindings.get("target_seed_manifest_file_sha256") is None
        or bindings.get("target_seed_manifest_sha256") is None
        or bindings.get("event_spec_sha256") != expected_event_spec_sha256
        or bindings.get("r6j_code_closure_sha256")
        != expected_collector_lineage_sha256
        or safe_path(str(outputs["seed_root"]), "native-v2 bound seed root") != root
    ):
        raise TrainingManifestContractError("native-v2 command bindings changed")

    reset_path = _inside(root, "per_seed_reset_receipt.json", "native-v2 reset receipt")
    receipt_path = _inside(root, "completed_group_receipt.json", "native-v2 group receipt")
    registry_path = _inside(root, "object_registry.json", "native-v2 object registry")
    pose_path = _inside(root, "pose_quality_spec.json", "native-v2 pose spec")
    group_path = _inside(root, "schema6_group.hdf5", "opaque native-v2 schema6 group")
    for bound, actual, role in (
        (outputs["per_seed_reset_receipt"], reset_path, "reset receipt"),
        (outputs["completed_group_receipt"], receipt_path, "group receipt"),
        (outputs["group_hdf5"], group_path, "group HDF"),
    ):
        if safe_path(str(bound), f"native-v2 bound {role}") != actual:
            raise TrainingManifestContractError(f"native-v2 {role} path changed")

    reset = _load_json(reset_path, "native-v2 reset receipt")
    receipt = _load_json(receipt_path, "native-v2 group receipt")
    reset_logical = _verify_signature(
        reset, "reset_receipt_sha256", "native-v2 reset receipt"
    )
    receipt_logical = _verify_signature(
        receipt, "group_receipt_sha256", "native-v2 group receipt"
    )
    reset_file = metadata_file_sha256(reset_path, "native-v2 reset receipt")
    receipt_file = metadata_file_sha256(receipt_path, "native-v2 group receipt")
    if set(reset) != NATIVE_V2_RESET_FIELDS or set(receipt) != NATIVE_V2_GROUP_FIELDS:
        raise TrainingManifestContractError("native-v2 receipt fields changed")

    identity_fields = (
        "split", "ordinal", "requested_seed", "resolved_seed", "pair_id"
    )
    command_identity = {
        "split": expected_command.get("split"),
        "ordinal": expected_command.get("ordinal"),
        "requested_seed": expected_command.get("requested_seed"),
        "resolved_seed": expected_command.get("expected_resolved_seed"),
        "pair_id": expected_command.get("pair_id"),
    }
    if (
        reset.get("format") != RESET_RECEIPT_FORMAT
        or reset.get("status") != RESET_RECEIPT_STATUS
        or receipt.get("format") != GROUP_RECEIPT_FORMAT
        or receipt.get("status") != GROUP_RECEIPT_STATUS
        or any(reset.get(field) != command_identity[field] for field in identity_fields)
        or any(receipt.get(field) != command_identity[field] for field in identity_fields)
        or reset.get("preregistration_sha256")
        != expected_preregistration_logical_sha256
        or receipt.get("preregistration_sha256")
        != expected_preregistration_logical_sha256
        or reset.get("command_sha256") != command_recorded
        or receipt.get("command_sha256") != command_recorded
        or reset.get("initial_scene_state_sha256")
        != expected_command.get("expected_initial_scene_state_sha256")
        or reset.get("identity_validation_count_before_policy_query") != 1
        or reset.get("policy_queries_before_reset_receipt") != 0
        or reset.get("evaluation_execution_authorized") is not False
        or reset.get("protected_inputs_read") is not False
        or receipt.get("candidate_original_indices") != CANDIDATE_INDICES
        or receipt.get("branch_records") != 4
        or receipt.get("per_seed_reset_receipt_sha256") != reset_logical
    ):
        raise TrainingManifestContractError("native-v2 reset/group lineage changed")

    registry_value = _load_json(registry_path, "native-v2 object registry")
    pose_value = _load_json(pose_path, "native-v2 pose spec")
    registry_file = metadata_file_sha256(registry_path, "native-v2 object registry")
    pose_file = metadata_file_sha256(pose_path, "native-v2 pose spec")
    registry = _validate_registry_semantics(registry_value)
    registry_logical = registry_sha256(registry)
    try:
        pose = validate_spec(pose_value, expected_registry_sha256=registry_logical)
        pose_logical = spec_sha256(pose, expected_registry_sha256=registry_logical)
    except Exception as exc:
        raise TrainingManifestContractError(
            "native-v2 pose spec does not bind the live registry"
        ) from exc
    metadata = _lstat_regular(group_path, "opaque native-v2 schema6 group")
    if (
        metadata.st_size < 1
        or receipt.get("object_registry_sha256") != registry_logical
        or reset.get("object_registry_sha256") != registry_logical
        or receipt.get("pose_spec_sha256") != pose_logical
        or reset.get("pose_spec_sha256") != pose_logical
        or not _is_sha256(receipt.get("group_file_sha256"))
    ):
        raise TrainingManifestContractError("native-v2 object/pose/group binding changed")
    identity = {
        **command_identity,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
    }
    return CollectionDescriptor(
        root=root,
        identity=identity,
        group_path=group_path,
        group_file_sha256=str(receipt["group_file_sha256"]),
        # The signed preregistration is the pre-collection authority and the
        # signed group receipt is both the per-group manifest and finalizer.
        authority_file_sha256=expected_preregistration_file_sha256,
        authority_logical_sha256=expected_preregistration_logical_sha256,
        manifest_file_sha256=receipt_file,
        manifest_logical_sha256=receipt_logical,
        final_receipt_file_sha256=receipt_file,
        final_receipt_logical_sha256=receipt_logical,
        object_registry_file_sha256=registry_file,
        object_registry_logical_sha256=registry_logical,
        pose_spec_file_sha256=pose_file,
        pose_spec_logical_sha256=pose_logical,
    )


def validate_collection_root(
    root_value: Path,
    *,
    expected_target_manifest_file_sha256: str,
    expected_target_manifest_logical_sha256: str,
    expected_event_spec_sha256: str,
    expected_collector_lineage_sha256: str,
) -> CollectionDescriptor:
    """Validate one frozen root without opening its HDF5 group."""

    root = _frozen_directory(root_value, "schema6 collection root")
    authority_path = _inside(root, "collection_authority.json", "collection authority")
    manifest_path = _inside(root, "manifest.json", "collection manifest")
    final_path = _inside(root, "final_receipt.json", "collection final receipt")
    authority = _load_json(authority_path, "collection authority")
    manifest = _load_json(manifest_path, "collection manifest")
    final = _load_json(final_path, "collection final receipt")
    authority_logical = _verify_signature(
        authority, "authority_sha256", "collection authority"
    )
    manifest_logical = _verify_signature(
        manifest, "manifest_sha256", "collection manifest"
    )
    final_logical = _verify_signature(final, "receipt_sha256", "collection final receipt")
    authority_file = metadata_file_sha256(authority_path, "collection authority")
    manifest_file = metadata_file_sha256(manifest_path, "collection manifest")
    final_file = metadata_file_sha256(final_path, "collection final receipt")

    expected_authority_fields = {
        "format", "status", "identity", "candidate_original_indices",
        "per_seed_live_registry_materialized", "fixed_seed_registry_reused",
        "bindings", "artifacts", "output_contract", "capability_contract",
        "authority_sha256",
    }
    expected_manifest_fields = {
        "format", "status", "identity", "candidate_original_indices",
        "branch_records", "object_registry_sha256", "pose_spec_sha256",
        "event_spec_sha256", "collector_lineage_sha256", "capability_contract",
        "group", "manifest_sha256",
    }
    expected_final_fields = expected_manifest_fields - {"manifest_sha256"} | {
        "authority", "manifest", "hdf5_content_opened_by_finalizer",
        "labels_read_by_finalizer", "receipt_sha256",
    }
    if (
        set(authority) != expected_authority_fields
        or set(manifest) != expected_manifest_fields
        or set(final) != expected_final_fields
    ):
        raise TrainingManifestContractError(
            "collection contracts contain unapproved fields (possible label disclosure)"
        )

    authority_identity = _exact_identity(authority.get("identity"), "authority")
    manifest_identity = _exact_identity(manifest.get("identity"), "manifest")
    final_identity = _exact_identity(final.get("identity"), "final receipt")
    bindings = authority.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "target_seed_manifest_file_sha256", "target_seed_manifest_sha256",
        "event_spec_sha256", "collector_lineage_sha256",
    }:
        raise TrainingManifestContractError("authority bindings changed")
    if (
        authority.get("format") != COLLECTION_AUTHORITY_FORMAT
        or authority.get("status") != "frozen_before_collection"
        or authority_identity != manifest_identity
        or authority_identity != final_identity
        or authority.get("candidate_original_indices") != CANDIDATE_INDICES
        or authority.get("per_seed_live_registry_materialized") is not True
        or authority.get("fixed_seed_registry_reused") is not False
        or authority.get("output_contract") != {
            "manifest": "manifest.json",
            "group": "schema6_group.hdf5",
            "create_once": True,
        }
        or authority.get("capability_contract") != CAPABILITY_CONTRACT
        or dict(bindings) != {
            "target_seed_manifest_file_sha256": expected_target_manifest_file_sha256,
            "target_seed_manifest_sha256": expected_target_manifest_logical_sha256,
            "event_spec_sha256": expected_event_spec_sha256,
            "collector_lineage_sha256": expected_collector_lineage_sha256,
        }
    ):
        raise TrainingManifestContractError("collection authority scope/binding changed")

    artifacts = authority.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "object_registry", "pose_quality_spec"
    }:
        raise TrainingManifestContractError("authority object/pose artifacts changed")
    registry_record = _record(artifacts["object_registry"], "object registry", logical=True)
    pose_record = _record(artifacts["pose_quality_spec"], "pose quality spec", logical=True)
    registry_path = _inside(root, registry_record["path"], "object registry")
    pose_path = _inside(root, pose_record["path"], "pose quality spec")
    registry_value = _load_json(registry_path, "object registry")
    pose_value = _load_json(pose_path, "pose quality spec")
    registry_file = metadata_file_sha256(registry_path, "object registry")
    pose_file = metadata_file_sha256(pose_path, "pose quality spec")
    registry = _validate_registry_semantics(registry_value)
    registry_logical = registry_sha256(registry)
    try:
        pose = validate_spec(pose_value, expected_registry_sha256=registry_logical)
        pose_logical = spec_sha256(pose, expected_registry_sha256=registry_logical)
    except Exception as exc:
        raise TrainingManifestContractError("pose spec does not bind the live registry") from exc
    if (
        registry_record["file_sha256"] != registry_file
        or registry_record["logical_sha256"] != registry_logical
        or pose_record["file_sha256"] != pose_file
        or pose_record["logical_sha256"] != pose_logical
    ):
        raise TrainingManifestContractError("object registry/pose artifact SHA changed")

    group_record = _record(manifest.get("group"), "manifest group", logical=False)
    final_group = _record(final.get("group"), "final group", logical=False)
    group_path = _inside(root, group_record["path"], "opaque schema6 group")
    metadata = _lstat_regular(group_path, "opaque schema6 group")
    if (
        group_record["path"] != "schema6_group.hdf5"
        or group_path.suffix.casefold() not in HDF_SUFFIXES
        or metadata.st_size < 1
    ):
        raise TrainingManifestContractError("schema6 group is not a non-empty opaque HDF artifact")

    authority_record = _record(final.get("authority"), "final authority", logical=True)
    manifest_record = _record(final.get("manifest"), "final manifest", logical=True)
    common = {
        "identity": authority_identity,
        "candidate_original_indices": CANDIDATE_INDICES,
        "branch_records": 4,
        "object_registry_sha256": registry_logical,
        "pose_spec_sha256": pose_logical,
        "event_spec_sha256": expected_event_spec_sha256,
        "collector_lineage_sha256": expected_collector_lineage_sha256,
        "capability_contract": CAPABILITY_CONTRACT,
    }
    if (
        manifest.get("format") != COLLECTION_MANIFEST_FORMAT
        or manifest.get("status") != COLLECTION_STATUS
        or any(manifest.get(key) != value for key, value in common.items())
        or final.get("format") != COLLECTION_FINAL_FORMAT
        or final.get("status") != COLLECTION_STATUS
        or any(final.get(key) != value for key, value in common.items())
        or group_record != final_group
        or authority_record != {
            "path": "collection_authority.json",
            "file_sha256": authority_file,
            "logical_sha256": authority_logical,
        }
        or manifest_record != {
            "path": "manifest.json",
            "file_sha256": manifest_file,
            "logical_sha256": manifest_logical,
        }
        or final.get("hdf5_content_opened_by_finalizer") is not False
        or final.get("labels_read_by_finalizer") is not False
    ):
        raise TrainingManifestContractError("collection final/manifest lineage changed")
    return CollectionDescriptor(
        root=root,
        identity=authority_identity,
        group_path=group_path,
        group_file_sha256=group_record["file_sha256"],
        authority_file_sha256=authority_file,
        authority_logical_sha256=authority_logical,
        manifest_file_sha256=manifest_file,
        manifest_logical_sha256=manifest_logical,
        final_receipt_file_sha256=final_file,
        final_receipt_logical_sha256=final_logical,
        object_registry_file_sha256=registry_file,
        object_registry_logical_sha256=registry_logical,
        pose_spec_file_sha256=pose_file,
        pose_spec_logical_sha256=pose_logical,
    )


def _target_key(row: Mapping[str, Any]) -> tuple[str, int, int, int, str]:
    return (
        str(row["split"]), int(row["ordinal"]), int(row["requested_seed"]),
        int(row["resolved_seed"]), str(row["pair_id"]),
    )


def _collection_key(descriptor: CollectionDescriptor) -> tuple[str, int, int, int, str]:
    identity = descriptor.identity
    return (
        str(identity["split"]), int(identity["ordinal"]),
        int(identity["requested_seed"]), int(identity["resolved_seed"]),
        str(identity["pair_id"]),
    )


def _logical_group_id(row: Mapping[str, Any]) -> str:
    return (
        f"{TASK}/{BODY}/{POLICY}/{row['split']}/{int(row['ordinal']):03d}/"
        f"{row['pair_id']}"
    )


def build_target_partition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adaptation = [_logical_group_id(row) for row in rows if row["split"] == "adaptation"]
    validation = [_logical_group_id(row) for row in rows if row["split"] == "validation"]
    if len(adaptation) != ADAPTATION_COUNT or len(validation) != TARGET_VALIDATION_COUNT:
        raise TrainingManifestContractError("target adaptation80/validation50 identity gate failed")
    value: dict[str, Any] = {
        "format": TARGET_PARTITION_FORMAT,
        "status": "frozen_from_target_seed_manifest_before_hdf_access",
        "adaptation": adaptation,
        "validation": validation,
        "evaluation": [],
        "evaluation_groups_included": 0,
        "hdf5_files_opened_before_partition_freeze": 0,
        "labels_read": False,
    }
    value["partition_sha256"] = canonical_sha256(value)
    return value


def build_external_split(partition: Mapping[str, Any]) -> dict[str, Any]:
    adaptation = list(partition["adaptation"])
    ordered = sorted(
        adaptation,
        key=lambda logical: hashlib.sha256(
            f"{EXTERNAL_SPLIT_SEED}:{logical}".encode("utf-8")
        ).hexdigest(),
    )
    internal_validation = sorted(ordered[:INTERNAL_VALIDATION_COUNT])
    train = sorted(ordered[INTERNAL_VALIDATION_COUNT:])
    sealed_test = sorted(partition["validation"])
    if not (
        len(train) == TRAIN_COUNT
        and len(internal_validation) == INTERNAL_VALIDATION_COUNT
        and len(sealed_test) == SEALED_TEST_COUNT
    ):
        raise TrainingManifestContractError("formal 60/20/50 support split failed")
    value: dict[str, Any] = {
        "format": EXTERNAL_SPLIT_FORMAT,
        "status": "frozen_label_blind_before_hdf_access",
        "algorithm": "target_validation50_to_sealed_test__sha256(seed:adaptation_logical_id)_first20_internal_validation_v2",
        "seed": EXTERNAL_SPLIT_SEED,
        "train": train,
        "validation": internal_validation,
        "test": sealed_test,
        "source_partition_sha256": partition["partition_sha256"],
        "target_validation_used_for_training_or_internal_validation": False,
        "evaluation_groups_included": 0,
        "hdf5_files_opened_before_split_freeze": 0,
        "labels_read": False,
    }
    value["split_sha256"] = canonical_sha256(value)
    return value


def _trainer_row(
    descriptor: CollectionDescriptor, *, logical_id: str, output_directory: Path
) -> dict[str, Any]:
    # A completed multi-seed collection is frozen read-only as one authority
    # tree.  Consequently a later manifest output cannot be an ancestor of the
    # collected HDF files.  Bind the already authenticated absolute path rather
    # than copying, linking, or relocating opaque HDF bytes.  The trainer
    # accepts this only from the signed manifest/split receipt and still opens
    # train/internal-validation membership exclusively.
    absolute = descriptor.group_path.resolve(strict=True)
    if not absolute.is_absolute() or _contains_sensitive(PurePath(absolute)):
        raise TrainingManifestContractError("trainer group path is unsafe")
    return {
        "logical_group_id": logical_id,
        "requested_seed": descriptor.identity["requested_seed"],
        "resolved_seed": descriptor.identity["resolved_seed"],
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "path": str(absolute),
        "file_sha256": descriptor.group_file_sha256,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, "aggregation output")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output.chmod(0o444)
    except BaseException:
        raise


def aggregate(
    *,
    target_seed_manifest_path: Path,
    expected_target_manifest_file_sha256: str,
    expected_target_manifest_logical_sha256: str,
    event_spec_sha256: str,
    collector_lineage_sha256: str,
    bound_trainer_path: Path,
    expected_bound_trainer_sha256: str,
    collection_roots: Sequence[Path],
    output_directory: Path,
    collection_preregistration_path: Path | None = None,
    expected_collection_preregistration_file_sha256: str | None = None,
    expected_collection_preregistration_logical_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate inputs and create either an insufficient or complete receipt."""

    if not all(
        _is_sha256(value)
        for value in (
            expected_target_manifest_file_sha256,
            expected_target_manifest_logical_sha256,
            event_spec_sha256,
            collector_lineage_sha256,
            expected_bound_trainer_sha256,
        )
    ):
        raise TrainingManifestContractError("expected input SHA256 bindings are invalid")
    trainer_path = safe_path(bound_trainer_path, "bound formal adapter trainer")
    _lstat_regular(trainer_path, "bound formal adapter trainer", read_only=False)
    trainer_sha = metadata_file_sha256(trainer_path, "bound formal adapter trainer")
    if trainer_sha != expected_bound_trainer_sha256:
        raise TrainingManifestContractError("bound formal adapter trainer SHA256 mismatch")
    target_path = safe_path(target_seed_manifest_path, "target seed manifest")
    _lstat_regular(target_path, "target seed manifest")
    target_file_sha = metadata_file_sha256(target_path, "target seed manifest")
    if target_file_sha != expected_target_manifest_file_sha256:
        raise TrainingManifestContractError("target seed manifest file SHA256 mismatch")
    target_value = _load_json(target_path, "target seed manifest")
    decoded = validate_target_seed_manifest(target_value)
    if decoded["seed_manifest_sha256"] != expected_target_manifest_logical_sha256:
        raise TrainingManifestContractError("target seed manifest logical SHA256 mismatch")
    target_rows = decoded["selected_rows"]
    partition = build_target_partition(target_rows)
    external_split = build_external_split(partition)
    expected_by_key = {_target_key(row): dict(row) for row in target_rows}

    preregistration: dict[str, Any] | None = None
    command_by_root: dict[Path, dict[str, Any]] = {}
    preregistration_binding: dict[str, str] | None = None
    preregistration_args = (
        collection_preregistration_path,
        expected_collection_preregistration_file_sha256,
        expected_collection_preregistration_logical_sha256,
    )
    if any(value is not None for value in preregistration_args):
        if any(value is None for value in preregistration_args):
            raise TrainingManifestContractError(
                "native-v2 collection preregistration binding is incomplete"
            )
        assert collection_preregistration_path is not None
        assert expected_collection_preregistration_file_sha256 is not None
        assert expected_collection_preregistration_logical_sha256 is not None
        if not _is_sha256(expected_collection_preregistration_file_sha256) or not _is_sha256(
            expected_collection_preregistration_logical_sha256
        ):
            raise TrainingManifestContractError(
                "native-v2 preregistration SHA256 binding is invalid"
            )
        prereg_path = safe_path(
            collection_preregistration_path, "native-v2 collection preregistration"
        )
        _lstat_regular(prereg_path, "native-v2 collection preregistration")
        prereg_file = metadata_file_sha256(
            prereg_path, "native-v2 collection preregistration"
        )
        if prereg_file != expected_collection_preregistration_file_sha256:
            raise TrainingManifestContractError(
                "native-v2 preregistration file SHA256 mismatch"
            )
        preregistration = _load_json(
            prereg_path, "native-v2 collection preregistration"
        )
        prereg_decoded = validate_preregistration(preregistration)
        if (
            preregistration.get("format") != MULTISEED_PREREGISTRATION_FORMAT
            or prereg_decoded["preregistration_sha256"]
            != expected_collection_preregistration_logical_sha256
        ):
            raise TrainingManifestContractError(
                "native-v2 preregistration logical SHA256 mismatch"
            )
        input_bindings = preregistration.get("input_bindings")
        target_binding = (
            input_bindings.get("target_seed_manifest")
            if isinstance(input_bindings, Mapping)
            else None
        )
        event_binding = (
            input_bindings.get("event_spec")
            if isinstance(input_bindings, Mapping)
            else None
        )
        r6_binding = (
            input_bindings.get("r6j_runtime_code")
            if isinstance(input_bindings, Mapping)
            else None
        )
        if (
            not isinstance(target_binding, Mapping)
            or target_binding.get("file_sha256") != target_file_sha
            or target_binding.get("logical_sha256")
            != decoded["seed_manifest_sha256"]
            or not isinstance(event_binding, Mapping)
            or event_binding.get("sha256") != event_spec_sha256
            or not isinstance(r6_binding, Mapping)
            or r6_binding.get("closure_sha256") != collector_lineage_sha256
        ):
            raise TrainingManifestContractError(
                "native-v2 preregistration target/event/collector binding changed"
            )
        for command in prereg_decoded["commands"]:
            outputs = command.get("outputs")
            if not isinstance(outputs, Mapping):
                raise TrainingManifestContractError(
                    "native-v2 preregistration command outputs are missing"
                )
            seed_root = safe_path(
                str(outputs.get("seed_root", "")), "native-v2 command seed root"
            )
            if seed_root in command_by_root:
                raise TrainingManifestContractError(
                    "native-v2 preregistration duplicates a seed root"
                )
            command_by_root[seed_root] = dict(command)
        preregistration_binding = {
            "path": str(prereg_path),
            "file_sha256": prereg_file,
            "logical_sha256": prereg_decoded["preregistration_sha256"],
        }

    roots = [safe_path(root, f"collection root {index}") for index, root in enumerate(collection_roots)]
    if len(roots) != len(set(roots)):
        raise TrainingManifestContractError("duplicate collection root supplied")
    descriptors: dict[tuple[str, int, int, int, str], CollectionDescriptor] = {}
    for root in roots:
        native_receipt = root / "completed_group_receipt.json"
        if native_receipt.exists() or native_receipt.is_symlink():
            command = command_by_root.get(root)
            if command is None or preregistration_binding is None:
                raise TrainingManifestContractError(
                    "native-v2 collection root lacks its signed preregistration command"
                )
            descriptor = validate_native_v2_collection_root(
                root,
                expected_command=command,
                expected_preregistration_file_sha256=preregistration_binding[
                    "file_sha256"
                ],
                expected_preregistration_logical_sha256=preregistration_binding[
                    "logical_sha256"
                ],
                expected_event_spec_sha256=event_spec_sha256,
                expected_collector_lineage_sha256=collector_lineage_sha256,
            )
        else:
            descriptor = validate_collection_root(
                root,
                expected_target_manifest_file_sha256=target_file_sha,
                expected_target_manifest_logical_sha256=decoded["seed_manifest_sha256"],
                expected_event_spec_sha256=event_spec_sha256,
                expected_collector_lineage_sha256=collector_lineage_sha256,
            )
        key = _collection_key(descriptor)
        if descriptor.identity["split"] == "evaluation":
            raise TrainingManifestContractError("evaluation collection root is forbidden")
        if key not in expected_by_key:
            raise TrainingManifestContractError("collection identity is not target adaptation/validation")
        if key in descriptors:
            raise TrainingManifestContractError("duplicate collection target identity")
        descriptors[key] = descriptor

    output = safe_path(output_directory, "aggregation output directory")
    if not output.is_dir():
        raise TrainingManifestContractError("aggregation output directory must already exist")
    names = {
        "receipt": output / "schema6_training_manifest_v2_receipt.json",
        "partition": output / "schema6_target_partition_v2.json",
        "split": output / "schema6_external_group_split_v2.json",
        "manifest": output / "schema6_training_manifest_v2_compat.json",
        "expected": output / "schema6_expected_manifest_split_v2.json",
    }
    if any(path.exists() for path in names.values()):
        raise TrainingManifestContractError("aggregation outputs are create-once")
    missing_keys = [key for key in expected_by_key if key not in descriptors]
    common: dict[str, Any] = {
        "format": FORMAT,
        "target_seed_manifest": {
            "path": str(target_path),
            "file_sha256": target_file_sha,
            "logical_sha256": decoded["seed_manifest_sha256"],
        },
        "event_spec_sha256": event_spec_sha256,
        "collector_lineage_sha256": collector_lineage_sha256,
        "bound_trainer_implementation": {
            "path": str(trainer_path), "file_sha256": trainer_sha,
        },
        "required_group_counts": {"adaptation": 80, "validation": 50, "evaluation": 0},
        "present_group_count": len(descriptors),
        "missing_group_count": len(missing_keys),
        "target_partition_sha256": partition["partition_sha256"],
        "expected_external_split_sha256": external_split["split_sha256"],
        "evaluation_roots_accepted": 0,
        "hdf5_content_files_opened": 0,
        "hdf5_labels_read": False,
        "hdf5_sha_validation_mode": "cross_signed_authority_manifest_final_receipt_without_hdf_open",
        "ordering_or_membership_depends_on_labels_or_outcomes": False,
    }
    if preregistration_binding is not None:
        common["native_v2_collection_preregistration"] = preregistration_binding
    if missing_keys:
        receipt = {
            **common,
            "status": INSUFFICIENT_STATUS,
            "training_authorized": False,
            "missing_identities": [
                {
                    "split": key[0], "ordinal": key[1], "requested_seed": key[2],
                    "resolved_seed": key[3], "pair_id": key[4],
                }
                for key in missing_keys
            ],
            "trainer_manifest_written": False,
            "external_split_written": False,
        }
        _audit_embedded_paths(receipt)
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_json(names["receipt"], receipt)
        return receipt

    ordered_descriptors = [descriptors[_target_key(row)] for row in target_rows]
    logical_ids = [_logical_group_id(row) for row in target_rows]
    trainer_groups = [
        _trainer_row(descriptor, logical_id=logical, output_directory=output)
        for descriptor, logical in zip(ordered_descriptors, logical_ids, strict=True)
    ]
    trainer_manifest: dict[str, Any] = {
        "format": TRAINER_MANIFEST_FORMAT,
        "status": "complete",
        "groups": trainer_groups,
        "fresh_inputs_used": False,
        "sealed_test_labels_disclosed": False,
        "event_spec_sha256": event_spec_sha256,
        "collector_lineage_sha256": collector_lineage_sha256,
        "target_seed_manifest_file_sha256": target_file_sha,
        "target_seed_manifest_logical_sha256": decoded["seed_manifest_sha256"],
        "target_partition_sha256": partition["partition_sha256"],
        "expected_external_split_sha256": external_split["split_sha256"],
        "hdf5_files_opened_during_aggregation": 0,
    }
    trainer_manifest["manifest_sha256"] = canonical_sha256(trainer_manifest)
    _audit_embedded_paths(trainer_manifest)

    _atomic_json(names["partition"], partition)
    _atomic_json(names["split"], external_split)
    _atomic_json(names["manifest"], trainer_manifest)
    partition_file_sha = metadata_file_sha256(names["partition"], "target partition output")
    split_file_sha = metadata_file_sha256(names["split"], "external split output")
    manifest_file_sha = metadata_file_sha256(names["manifest"], "trainer manifest output")
    expected: dict[str, Any] = {
        "format": EXPECTED_FORMAT,
        "status": "complete_external_manifest_and_split_expectations",
        "trainer_compatible_manifest": {
            "path": str(names["manifest"]),
            "file_sha256": manifest_file_sha,
            "logical_sha256": trainer_manifest["manifest_sha256"],
        },
        "target_partition": {
            "path": str(names["partition"]),
            "file_sha256": partition_file_sha,
            "logical_sha256": partition["partition_sha256"],
        },
        "external_split": {
            "path": str(names["split"]),
            "file_sha256": split_file_sha,
            "logical_sha256": external_split["split_sha256"],
        },
        "bound_trainer_implementation": {
            "path": str(trainer_path), "file_sha256": trainer_sha,
        },
        "required_trainer_group_counts": {
            "train": TRAIN_COUNT,
            "validation": INTERNAL_VALIDATION_COUNT,
            "test": SEALED_TEST_COUNT,
        },
        "direct_bound_trainer_execution_authorized": True,
        "hdf5_content_files_opened": 0,
        "labels_read": False,
    }
    expected["expected_receipt_sha256"] = canonical_sha256(expected)
    _atomic_json(names["expected"], expected)
    expected_file_sha = metadata_file_sha256(names["expected"], "expected SHA output")
    receipt = {
        **common,
        "status": COMPLETE_STATUS,
        "training_inputs_complete": True,
        "training_authorized": True,
        "direct_bound_trainer_execution_authorized": True,
        "trainer_compatible_manifest": expected["trainer_compatible_manifest"],
        "target_partition": expected["target_partition"],
        "external_split": expected["external_split"],
        "expected_manifest_split_receipt": {
            "path": str(names["expected"]),
            "file_sha256": expected_file_sha,
            "logical_sha256": expected["expected_receipt_sha256"],
        },
        "collection_lineage": [
            {
                "logical_group_id": logical,
                "root": str(descriptor.root),
                "authority_file_sha256": descriptor.authority_file_sha256,
                "authority_logical_sha256": descriptor.authority_logical_sha256,
                "manifest_file_sha256": descriptor.manifest_file_sha256,
                "manifest_logical_sha256": descriptor.manifest_logical_sha256,
                "final_receipt_file_sha256": descriptor.final_receipt_file_sha256,
                "final_receipt_logical_sha256": descriptor.final_receipt_logical_sha256,
                "group_recorded_file_sha256": descriptor.group_file_sha256,
                "object_registry_file_sha256": descriptor.object_registry_file_sha256,
                "object_registry_logical_sha256": descriptor.object_registry_logical_sha256,
                "pose_spec_file_sha256": descriptor.pose_spec_file_sha256,
                "pose_spec_logical_sha256": descriptor.pose_spec_logical_sha256,
            }
            for logical, descriptor in zip(logical_ids, ordered_descriptors, strict=True)
        ],
    }
    _audit_embedded_paths(receipt)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(names["receipt"], receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-seed-manifest", type=Path, required=True)
    parser.add_argument("--expected-target-manifest-file-sha256", required=True)
    parser.add_argument("--expected-target-manifest-logical-sha256", required=True)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--collector-lineage-sha256", required=True)
    parser.add_argument("--bound-trainer", type=Path, required=True)
    parser.add_argument("--expected-bound-trainer-sha256", required=True)
    parser.add_argument("--collection-root", type=Path, action="append", default=[])
    parser.add_argument("--collection-preregistration", type=Path)
    parser.add_argument("--expected-collection-preregistration-file-sha256")
    parser.add_argument("--expected-collection-preregistration-logical-sha256")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = aggregate(
        target_seed_manifest_path=args.target_seed_manifest,
        expected_target_manifest_file_sha256=args.expected_target_manifest_file_sha256,
        expected_target_manifest_logical_sha256=args.expected_target_manifest_logical_sha256,
        event_spec_sha256=args.event_spec_sha256,
        collector_lineage_sha256=args.collector_lineage_sha256,
        bound_trainer_path=args.bound_trainer,
        expected_bound_trainer_sha256=args.expected_bound_trainer_sha256,
        collection_roots=args.collection_root,
        output_directory=args.output_directory,
        collection_preregistration_path=args.collection_preregistration,
        expected_collection_preregistration_file_sha256=(
            args.expected_collection_preregistration_file_sha256
        ),
        expected_collection_preregistration_logical_sha256=(
            args.expected_collection_preregistration_logical_sha256
        ),
    )
    print(json.dumps({"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0 if receipt["status"] == COMPLETE_STATUS else 20


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPABILITY_CONTRACT", "COLLECTION_AUTHORITY_FORMAT", "COLLECTION_FINAL_FORMAT",
    "COLLECTION_MANIFEST_FORMAT", "COLLECTION_STATUS", "COMPLETE_STATUS", "FORMAT",
    "INSUFFICIENT_STATUS", "TrainingManifestContractError", "aggregate",
    "build_external_split", "build_target_partition", "canonical_sha256",
    "metadata_file_sha256", "validate_collection_root",
    "validate_native_v2_collection_root",
]
