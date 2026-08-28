#!/usr/bin/env python3
"""Materialize the frozen development300 collection into trainer inputs.

This v3 protocol is independent of the historical v2 materializer.  It opens
only signed JSON metadata and code files.  Every ``schema6_group.hdf5`` remains
opaque: the materializer performs lstat/path/permission checks and trusts its
byte SHA only through the terminal-bound sealed group receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from materialize_smolvla_piper_schema6_development300_identity_authority import (
    validate_collection_identity_authority,
)
from preregister_smolvla_piper_schema6_target_development300 import (
    BODY,
    POLICY,
    SPLIT_COUNTS,
    TASK,
    TOTAL_GROUPS,
    canonical_sha256,
    validate_preregistration,
)
from run_smolvla_piper_schema6_development300_collection import (
    AUTHORITY_FORMAT,
    AUTHORITY_SIGNATURE,
    DETACH_RECEIPT_FORMAT,
    EXPECTED_GROUP_FILES,
    GROUP_SIGNATURE,
    PLAN_SIGNATURE,
    STAGE_RECEIPT_FORMAT,
    STATIC_PLAN_FORMAT,
    TERMINAL_RECEIPT_FORMAT,
    TERMINAL_SUCCESS,
    WORKER_ACCOUNTING_FORMAT,
    WORKER_GROUP_RECEIPT_FORMAT,
    WORKER_RESET_RECEIPT_FORMAT,
    validate_runner_authority,
)


FORMAT = "etsf_smolvla_piper_schema6_training_manifest_aggregator_v3"
COMPLETE_STATUS = "complete_development300_label_blind_training_inputs_frozen"
TRAINER_MANIFEST_FORMAT = "etsf_smolvla_piper_schema6_training_manifest_v1"
TARGET_PARTITION_FORMAT = "etsf_smolvla_piper_schema6_target_partition_v3"
EXTERNAL_SPLIT_FORMAT = "etsf_smolvla_piper_schema6_external_group_split_v3"
EXPECTED_FORMAT = "etsf_smolvla_piper_schema6_expected_manifest_split_v3"
SPLIT_PROFILE = "development300_v3"
TRAIN_COUNT = 80
INTERNAL_VALIDATION_COUNT = 30
FORMAL_VALIDATION_COUNT = 190
CANDIDATE_INDICES = [0, 1, 2, 3]
HDF_SUFFIXES = {".h5", ".hdf", ".hdf5"}
SHA_CHARS = frozenset("0123456789abcdef")
SENSITIVE_COMPONENTS = {"fresh", "confirmation", "test", "tests", "testing"}
OUTPUT_NAMES = {
    "receipt": "schema6_training_manifest_v3_receipt.json",
    "partition": "schema6_target_partition_v3.json",
    "split": "schema6_external_group_split_v3.json",
    "manifest": "schema6_training_manifest_v3_compat.json",
    "expected": "schema6_expected_manifest_split_v3.json",
}


class TrainingManifestV3Error(RuntimeError):
    """The development300 terminal-to-training proof failed closed."""


@dataclass(frozen=True)
class FrozenGroup:
    logical_group_id: str
    split: str
    global_ordinal: int
    split_ordinal: int
    requested_seed: int
    resolved_seed: int
    pair_id: str
    group_path: Path
    group_file_sha256: str
    command_sha256: str
    stage_receipt_sha256: str
    group_receipt_sha256: str
    reset_receipt_sha256: str
    candidate_accounting_sha256: str
    object_registry_sha256: str
    pose_spec_sha256: str


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _require_int(value: Any, role: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingManifestV3Error(f"{role} must be an integer >= {minimum}")
    return value


def _forbidden(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if lowered in SENSITIVE_COMPONENTS:
            return True
        if lowered.startswith(("test_", "test-")):
            return True
        if "fresh" in lowered or "confirmation" in lowered:
            return True
    return False


def _reject_symlink_components(path: Path, role: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise TrainingManifestV3Error(f"{role} path is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise TrainingManifestV3Error(f"{role} path contains a symlink")


def safe_path(value: str | os.PathLike[str], role: str, *, must_exist: bool) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise TrainingManifestV3Error(f"{role} path is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(text)))
    if _forbidden(PurePath(lexical)):
        raise TrainingManifestV3Error(f"{role} enters a forbidden namespace")
    _reject_symlink_components(lexical, role)
    try:
        resolved = lexical.resolve(strict=must_exist)
    except OSError as error:
        raise TrainingManifestV3Error(f"{role} path is unavailable") from error
    if _forbidden(PurePath(resolved)):
        raise TrainingManifestV3Error(f"{role} resolves into a forbidden namespace")
    return resolved


def _regular_file(
    value: str | os.PathLike[str],
    role: str,
    *,
    frozen: bool,
    allow_hdf: bool = False,
) -> Path:
    path = safe_path(value, role, must_exist=True)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise TrainingManifestV3Error(f"{role} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise TrainingManifestV3Error(f"{role} must be a regular file")
    if not allow_hdf and path.suffix.casefold() in HDF_SUFFIXES:
        raise TrainingManifestV3Error(f"{role} HDF byte access is forbidden")
    if frozen and metadata.st_mode & 0o222:
        raise TrainingManifestV3Error(f"{role} is not frozen read-only")
    return path


def _frozen_directory(value: str | os.PathLike[str], role: str) -> Path:
    path = safe_path(value, role, must_exist=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise TrainingManifestV3Error(f"{role} must be a directory")
    if metadata.st_mode & 0o222:
        raise TrainingManifestV3Error(f"{role} is not frozen read-only")
    return path


def metadata_file_sha256(path: Path, role: str) -> str:
    source = _regular_file(path, role, frozen=False, allow_hdf=False)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_embedded_paths(value: Any, role: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _audit_embedded_paths(child, f"{role}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _audit_embedded_paths(child, f"{role}[{index}]")
    elif isinstance(value, str):
        looks_like_path = value.startswith(("/", "./", "../")) or "\\" in value
        if looks_like_path and _forbidden(PurePath(value)):
            raise TrainingManifestV3Error(f"{role} embeds a forbidden path")


def _load_json(path: Path, role: str, *, frozen: bool = True) -> dict[str, Any]:
    source = _regular_file(path, role, frozen=frozen, allow_hdf=False)
    if source.suffix.casefold() != ".json":
        raise TrainingManifestV3Error(f"{role} must be JSON metadata")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingManifestV3Error(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise TrainingManifestV3Error(f"{role} must contain an object")
    _audit_embedded_paths(value, role)
    return value


def _verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise TrainingManifestV3Error(f"{role} logical SHA changed")
    return str(recorded)


def _bound_json(
    path: Path,
    expected_file_sha256: str,
    expected_logical_sha256: str,
    signature: str,
    role: str,
) -> tuple[Path, dict[str, Any]]:
    if not _is_sha(expected_file_sha256) or not _is_sha(expected_logical_sha256):
        raise TrainingManifestV3Error(f"{role} expected SHA is invalid")
    source = _regular_file(path, role, frozen=True)
    if metadata_file_sha256(source, role) != expected_file_sha256:
        raise TrainingManifestV3Error(f"{role} file SHA changed")
    value = _load_json(source, role)
    if _verify_signed(value, signature, role) != expected_logical_sha256:
        raise TrainingManifestV3Error(f"{role} expected logical SHA changed")
    return source, value


def _record(path: Path, logical_sha256: str | None = None) -> dict[str, str]:
    result = {"path": str(path), "file_sha256": metadata_file_sha256(path, "record")}
    if logical_sha256 is not None:
        result["logical_sha256"] = logical_sha256
    return result


def _validate_record_source(
    value: Any, role: str, *, logical: bool
) -> tuple[Path, dict[str, str]]:
    expected = {"path", "file_sha256"} | ({"logical_sha256"} if logical else set())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TrainingManifestV3Error(f"{role} record fields changed")
    if not _is_sha(value.get("file_sha256")) or (
        logical and not _is_sha(value.get("logical_sha256"))
    ):
        raise TrainingManifestV3Error(f"{role} record SHA is invalid")
    source = _regular_file(Path(str(value["path"])), role, frozen=False)
    if metadata_file_sha256(source, role) != value["file_sha256"]:
        raise TrainingManifestV3Error(f"{role} record file SHA changed")
    return source, {key: str(value[key]) for key in expected}


def _validate_static_plan(
    value: Mapping[str, Any],
    *,
    root: Path,
    authority_path: Path,
    authority_file_sha256: str,
    authority: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
) -> str:
    logical = _verify_signed(value, PLAN_SIGNATURE, "runner static plan")
    expected_fields = {
        "format", "status", "runner_authority", "output_root",
        "runner_implementation", "sealed_worker_implementation",
        "runtime_python", "command_count", "candidate_accounting_records",
        "command_order_sha256", "command_sha256", "split_counts",
        "retry_or_resume_allowed", "evaluation400_commands",
        "formal_label_open_authorized", PLAN_SIGNATURE,
    }
    runner_record = value.get("runner_authority")
    if (
        set(value) != expected_fields
        or value.get("format") != STATIC_PLAN_FORMAT
        or value.get("status") != "dry_run_exact_plan_validated_output_unclaimed"
        or runner_record
        != {
            "path": str(authority_path),
            "file_sha256": authority_file_sha256,
            "logical_sha256": authority[AUTHORITY_SIGNATURE],
        }
        or value.get("output_root") != str(root)
        or value.get("runner_implementation") != authority["implementations"]["runner"]
        or value.get("sealed_worker_implementation")
        != authority["implementations"]["sealed_group_worker"]
        or value.get("runtime_python") != authority["implementations"]["runtime_python"]
        or type(value.get("command_count")) is not int
        or value.get("command_count") != TOTAL_GROUPS
        or type(value.get("candidate_accounting_records")) is not int
        or value.get("candidate_accounting_records") != TOTAL_GROUPS * 4
        or value.get("command_order_sha256")
        != authority["exact_execution"]["command_order_sha256"]
        or value.get("command_sha256")
        != [command["command_sha256"] for command in commands]
        or value.get("split_counts") != SPLIT_COUNTS
        or any(type(value["split_counts"].get(name)) is not int for name in SPLIT_COUNTS)
        or value.get("retry_or_resume_allowed") is not False
        or type(value.get("evaluation400_commands")) is not int
        or value.get("evaluation400_commands") != 0
        or value.get("formal_label_open_authorized") is not False
    ):
        raise TrainingManifestV3Error("runner static plan scope changed")
    return logical


def _validate_terminal(
    value: Mapping[str, Any],
    *,
    authority_sha256: str,
    plan_sha256: str,
    expected_logical_sha256: str,
) -> None:
    logical = _verify_signed(value, "terminal_receipt_sha256", "terminal receipt")
    expected_fields = {
        "format", "status", "runner_authority_sha256", "runner_plan_sha256",
        "completed_groups", "candidate_accounting_records", "split_counts",
        "formal_payloads_sealed", "gap_free_exact_command_order",
        "retry_replacement_additional_seed_or_resume_performed",
        "formal_label_opened_by_runner_or_watcher",
        "evaluation400_commands_executed", "stage_receipt_order_sha256",
        "terminal_receipt_sha256",
    }
    if (
        logical != expected_logical_sha256
        or set(value) != expected_fields
        or value.get("format") != TERMINAL_RECEIPT_FORMAT
        or value.get("status") != TERMINAL_SUCCESS
        or value.get("runner_authority_sha256") != authority_sha256
        or value.get("runner_plan_sha256") != plan_sha256
        or type(value.get("completed_groups")) is not int
        or value.get("completed_groups") != TOTAL_GROUPS
        or type(value.get("candidate_accounting_records")) is not int
        or value.get("candidate_accounting_records") != TOTAL_GROUPS * 4
        or value.get("split_counts") != SPLIT_COUNTS
        or any(type(value["split_counts"].get(name)) is not int for name in SPLIT_COUNTS)
        or type(value.get("formal_payloads_sealed")) is not int
        or value.get("formal_payloads_sealed") != FORMAL_VALIDATION_COUNT
        or value.get("gap_free_exact_command_order") is not True
        or value.get("retry_replacement_additional_seed_or_resume_performed") is not False
        or value.get("formal_label_opened_by_runner_or_watcher") is not False
        or type(value.get("evaluation400_commands_executed")) is not int
        or value.get("evaluation400_commands_executed") != 0
        or not _is_sha(value.get("stage_receipt_order_sha256"))
    ):
        raise TrainingManifestV3Error("collection terminal receipt is not exact success")


def _logical_group_id(command: Mapping[str, Any]) -> str:
    return (
        f"{TASK}/{BODY}/{POLICY}/{command['split']}/"
        f"{int(command['split_ordinal']):03d}/{command['pair_id']}"
    )


def _validate_registry_semantics(value: Mapping[str, Any]) -> str:
    try:
        registry = validate_registry(value)
    except Exception as error:
        raise TrainingManifestV3Error("object registry is invalid") from error
    expected = (
        ("can", "task_attr=can;sapien_actor_name=", "105_sauce-can/base", "manipulated"),
        ("pot", "task_attr=pot;sapien_actor_name=", "060_kitchenpot/base", "receptacle"),
    )
    objects = registry["objects"]
    if len(objects) != 2:
        raise TrainingManifestV3Error("object registry must contain can then pot")
    for item, (name, actor_prefix, asset_prefix, object_role) in zip(
        objects, expected, strict=True
    ):
        asset = str(item["asset_model_id"])
        suffix = asset[len(asset_prefix):]
        if (
            item["name"] != name
            or not str(item["stable_sim_actor_id"]).startswith(actor_prefix)
            or not str(item["stable_sim_actor_id"])[len(actor_prefix):]
            or not asset.startswith(asset_prefix)
            or not suffix.isdecimal()
            or str(int(suffix)) != suffix
            or item["role"] != object_role
            or item["is_static"] is not False
        ):
            raise TrainingManifestV3Error("object registry actor/asset identity changed")
    return registry_sha256(registry)


def _validate_group_receipt(
    receipt: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    authority_sha256: str,
) -> str:
    logical = _verify_signed(receipt, GROUP_SIGNATURE, "sealed group receipt")
    expected_fields = {
        "format", "status", "runner_authority_sha256", "command_sha256",
        "global_ordinal", "split", "requested_seed", "resolved_seed", "pair_id",
        "candidate_original_indices", "candidate_accounting_records",
        "candidate_accounting_sha256", "per_seed_reset_receipt_sha256",
        "object_registry_sha256", "pose_spec_sha256", "group_file_sha256",
        "formal_payload_sealed", "outcome_or_label_fields_disclosed_to_runner",
        "evaluation400", GROUP_SIGNATURE,
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("format") != WORKER_GROUP_RECEIPT_FORMAT
        or receipt.get("status") != "complete_exact_four_candidate_accounting"
        or receipt.get("runner_authority_sha256") != authority_sha256
        or receipt.get("command_sha256") != command["command_sha256"]
        or type(receipt.get("global_ordinal")) is not int
        or receipt.get("global_ordinal") != command["global_ordinal"]
        or receipt.get("split") != command["split"]
        or type(receipt.get("requested_seed")) is not int
        or receipt.get("requested_seed") != command["requested_seed"]
        or type(receipt.get("resolved_seed")) is not int
        or receipt.get("resolved_seed") != command["expected_resolved_seed"]
        or receipt.get("pair_id") != command["pair_id"]
        or receipt.get("candidate_original_indices") != CANDIDATE_INDICES
        or any(type(index) is not int for index in receipt.get("candidate_original_indices", []))
        or type(receipt.get("candidate_accounting_records")) is not int
        or receipt.get("candidate_accounting_records") != 4
        or any(
            not _is_sha(receipt.get(field))
            for field in (
                "candidate_accounting_sha256", "per_seed_reset_receipt_sha256",
                "object_registry_sha256", "pose_spec_sha256", "group_file_sha256",
            )
        )
        or receipt.get("formal_payload_sealed")
        is not (command["split"] == "formal_target_validation")
        or receipt.get("outcome_or_label_fields_disclosed_to_runner") is not False
        or receipt.get("evaluation400") is not False
    ):
        raise TrainingManifestV3Error("sealed group receipt scope changed")
    return logical


def _validate_reset(
    reset: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    identity_row: Mapping[str, Any],
    authority_sha256: str,
    plan_sha256: str,
    group_receipt: Mapping[str, Any],
) -> str:
    logical = _verify_signed(reset, "reset_receipt_sha256", "reset receipt")
    expected_fields = {
        "format", "status", "runner_authority_sha256", "runner_plan_sha256",
        "command_sha256", "global_ordinal", "split", "requested_seed",
        "resolved_seed", "pair_id", "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256", "object_registry_sha256",
        "pose_spec_sha256", "identity_validation_count_before_policy_query",
        "policy_queries_before_reset_receipt",
        "outcome_or_label_read_before_reset_receipt", "evaluation400",
        "reset_receipt_sha256",
    }
    if (
        set(reset) != expected_fields
        or reset.get("format") != WORKER_RESET_RECEIPT_FORMAT
        or reset.get("status") != "identity_verified_before_first_policy_query"
        or reset.get("runner_authority_sha256") != authority_sha256
        or reset.get("runner_plan_sha256") != plan_sha256
        or reset.get("command_sha256") != command["command_sha256"]
        or type(reset.get("global_ordinal")) is not int
        or reset.get("global_ordinal") != command["global_ordinal"]
        or reset.get("split") != command["split"]
        or type(reset.get("requested_seed")) is not int
        or reset.get("requested_seed") != command["requested_seed"]
        or type(reset.get("resolved_seed")) is not int
        or reset.get("resolved_seed") != command["expected_resolved_seed"]
        or reset.get("pair_id") != command["pair_id"]
        or any(
            reset.get(field) != identity_row[field]
            for field in (
                "initial_scene_state_sha256",
                "initial_measured_joint_state_sha256",
                "initial_commanded_drive_target_sha256",
            )
        )
        or reset.get("object_registry_sha256") != group_receipt["object_registry_sha256"]
        or reset.get("pose_spec_sha256") != group_receipt["pose_spec_sha256"]
        or type(reset.get("identity_validation_count_before_policy_query")) is not int
        or reset.get("identity_validation_count_before_policy_query") != 1
        or type(reset.get("policy_queries_before_reset_receipt")) is not int
        or reset.get("policy_queries_before_reset_receipt") != 0
        or reset.get("outcome_or_label_read_before_reset_receipt") is not False
        or reset.get("evaluation400") is not False
        or logical != group_receipt["per_seed_reset_receipt_sha256"]
    ):
        raise TrainingManifestV3Error("reset receipt binding changed")
    return logical


def _validate_accounting(
    accounting: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    group_receipt: Mapping[str, Any],
) -> str:
    logical = _verify_signed(
        accounting, "candidate_accounting_sha256", "candidate accounting"
    )
    records = accounting.get("records")
    if (
        set(accounting)
        != {
            "format", "status", "command_sha256", "candidate_original_indices",
            "records", "success_event_outcome_or_label_included",
            "candidate_accounting_sha256",
        }
        or accounting.get("format") != WORKER_ACCOUNTING_FORMAT
        or accounting.get("status") != "complete_four_original_candidate_records"
        or accounting.get("command_sha256") != command["command_sha256"]
        or accounting.get("candidate_original_indices") != CANDIDATE_INDICES
        or any(type(index) is not int for index in accounting.get("candidate_original_indices", []))
        or accounting.get("success_event_outcome_or_label_included") is not False
        or not isinstance(records, list)
        or len(records) != 4
        or logical != group_receipt["candidate_accounting_sha256"]
    ):
        raise TrainingManifestV3Error("candidate accounting scope changed")
    for index, row in enumerate(records):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "original_candidate_index", "native_action_sha256", "feasible",
                "executed", "right_censored", "execution_status",
            }
            or type(row.get("original_candidate_index")) is not int
            or row.get("original_candidate_index") != index
            or not _is_sha(row.get("native_action_sha256"))
            or type(row.get("feasible")) is not bool
            or type(row.get("executed")) is not bool
            or type(row.get("right_censored")) is not bool
            or row.get("executed") is not row.get("feasible")
            or row.get("right_censored") is row.get("feasible")
            or row.get("execution_status")
            != (
                "executed_legal_branch"
                if row.get("feasible")
                else "nonexecuted_censored_infeasible"
            )
        ):
            raise TrainingManifestV3Error("candidate accounting row changed")
    return logical


def _validate_stage(
    stage: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    group_receipt_sha256: str,
    group_file_sha256: str,
) -> str:
    logical = _verify_signed(stage, "stage_receipt_sha256", "stage receipt")
    if (
        set(stage)
        != {
            "format", "status", "global_ordinal", "command_sha256",
            "sealed_group_receipt_sha256", "group_file_sha256",
            "formal_payload_sealed", "retry_performed", "stage_receipt_sha256",
        }
        or stage.get("format")
        != "etsf_smolvla_piper_schema6_development300_collection_stage_receipt_v1"
        or stage.get("status") != "published_exact_once"
        or type(stage.get("global_ordinal")) is not int
        or stage.get("global_ordinal") != command["global_ordinal"]
        or stage.get("command_sha256") != command["command_sha256"]
        or stage.get("sealed_group_receipt_sha256") != group_receipt_sha256
        or stage.get("group_file_sha256") != group_file_sha256
        or stage.get("formal_payload_sealed")
        is not (command["split"] == "formal_target_validation")
        or stage.get("retry_performed") is not False
    ):
        raise TrainingManifestV3Error("stage success receipt changed")
    return logical


def _validate_group(
    *,
    root: Path,
    command: Mapping[str, Any],
    identity_row: Mapping[str, Any],
    authority_sha256: str,
    plan_sha256: str,
    authority: Mapping[str, Any],
    authority_path: Path,
    authority_file_sha256: str,
    plan_path: Path,
    plan_file_sha256: str,
) -> FrozenGroup:
    for field in ("global_ordinal", "split_ordinal", "requested_seed", "expected_resolved_seed"):
        _require_int(command.get(field), f"command {field}")
    for field in ("global_ordinal", "split_ordinal", "requested_seed", "resolved_seed"):
        _require_int(identity_row.get(field), f"identity {field}")
    outputs = command.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TrainingManifestV3Error("command outputs are missing")
    seed_root = _frozen_directory(Path(str(outputs.get("seed_root", ""))), "seed root")
    if root not in seed_root.parents:
        raise TrainingManifestV3Error("seed root escapes collection root")
    expected_mode = 0o500 if command["split"] == "formal_target_validation" else 0o555
    if stat.S_IMODE(seed_root.stat().st_mode) != expected_mode:
        raise TrainingManifestV3Error("seed root terminal mode changed")
    children = list(seed_root.iterdir())
    if {path.name for path in children} != EXPECTED_GROUP_FILES:
        raise TrainingManifestV3Error("sealed group file inventory changed")
    if any(path.is_symlink() for path in children):
        raise TrainingManifestV3Error("sealed group contains a symlink")
    file_mode = 0o400 if command["split"] == "formal_target_validation" else 0o444
    if any(stat.S_IMODE(path.stat().st_mode) != file_mode for path in children):
        raise TrainingManifestV3Error("sealed group file mode changed")

    receipt_path = seed_root / "completed_group_receipt.json"
    receipt = _load_json(receipt_path, "sealed group receipt")
    receipt_sha = _validate_group_receipt(
        receipt, command=command, authority_sha256=authority_sha256
    )
    reset = _load_json(seed_root / "per_seed_reset_receipt.json", "reset receipt")
    reset_sha = _validate_reset(
        reset,
        command=command,
        identity_row=identity_row,
        authority_sha256=authority_sha256,
        plan_sha256=plan_sha256,
        group_receipt=receipt,
    )
    accounting = _load_json(
        seed_root / "candidate_accounting.json", "candidate accounting"
    )
    accounting_sha = _validate_accounting(
        accounting, command=command, group_receipt=receipt
    )
    registry = _load_json(seed_root / "object_registry.json", "object registry")
    registry_logical = _validate_registry_semantics(registry)
    pose = _load_json(seed_root / "pose_quality_spec.json", "pose quality spec")
    thresholds = pose.get("thresholds")
    numeric_thresholds = (
        "quaternion_norm_abs_tolerance",
        "max_step_translation_m",
        "max_step_rotation_rad",
        "static_object_max_step_translation_m",
        "static_object_max_step_rotation_rad",
        "timestamp_step_min_s",
        "timestamp_step_max_s",
        "max_physics_substeps_per_control_step",
    )
    if (
        type(pose.get("schema_version")) is not int
        or not isinstance(thresholds, Mapping)
        or any(type(thresholds.get(field)) is bool for field in numeric_thresholds)
        or any(
            type(number) is bool
            for bounds in thresholds.get("world_aabb_m", [])
            if isinstance(bounds, list)
            for number in bounds
        )
    ):
        raise TrainingManifestV3Error("pose quality numeric fields use boolean values")
    try:
        validate_spec(pose, expected_registry_sha256=registry_logical)
        pose_logical = spec_sha256(pose, expected_registry_sha256=registry_logical)
    except Exception as error:
        raise TrainingManifestV3Error("pose quality spec is invalid") from error
    if (
        registry_logical != receipt["object_registry_sha256"]
        or pose_logical != receipt["pose_spec_sha256"]
    ):
        raise TrainingManifestV3Error("registry/pose logical SHA binding changed")

    group_path = _regular_file(
        seed_root / "schema6_group.hdf5",
        "sealed group HDF",
        frozen=True,
        allow_hdf=True,
    )
    if group_path != safe_path(outputs.get("group_hdf5", ""), "command group HDF", must_exist=True):
        raise TrainingManifestV3Error("sealed group HDF path changed")

    stage_root = _frozen_directory(
        root / "_runner" / "stages" / f"stage_{command['global_ordinal']:03d}",
        "stage root",
    )
    if {path.name for path in stage_root.iterdir()} != {
        "launch_receipt.json",
        "terminal_receipt.json",
        "worker.log",
    }:
        raise TrainingManifestV3Error("stage artifact inventory changed")
    launch = _load_json(stage_root / "launch_receipt.json", "stage launch receipt")
    launch_logical = _verify_signed(
        launch, "stage_receipt_sha256", "stage launch receipt"
    )
    del launch_logical
    expected_worker_command = [
        authority["implementations"]["runtime_python"]["path"],
        authority["implementations"]["sealed_group_worker"]["path"],
        "collect-one",
        "--authority",
        str(authority_path),
        "--authority-file-sha256",
        authority_file_sha256,
        "--static-plan",
        str(plan_path),
        "--static-plan-file-sha256",
        plan_file_sha256,
        "--global-ordinal",
        str(command["global_ordinal"]),
        "--staging-payload",
        str(
            root
            / "_runner"
            / "staging"
            / f"stage_{command['global_ordinal']:03d}"
            / "payload"
        ),
    ]
    if (
        set(launch)
        != {
            "format", "status", "global_ordinal", "command_sha256",
            "worker_command", "retry_allowed", "stage_receipt_sha256",
        }
        or launch.get("format") != STAGE_RECEIPT_FORMAT
        or launch.get("status") != "launching_exact_once"
        or type(launch.get("global_ordinal")) is not int
        or launch.get("global_ordinal") != command["global_ordinal"]
        or launch.get("command_sha256") != command["command_sha256"]
        or launch.get("worker_command") != expected_worker_command
        or launch.get("retry_allowed") is not False
    ):
        raise TrainingManifestV3Error("stage launch receipt changed")
    _regular_file(stage_root / "worker.log", "sealed worker log", frozen=True)
    stage = _load_json(stage_root / "terminal_receipt.json", "stage terminal receipt")
    stage_sha = _validate_stage(
        stage,
        command=command,
        group_receipt_sha256=receipt_sha,
        group_file_sha256=receipt["group_file_sha256"],
    )
    return FrozenGroup(
        logical_group_id=_logical_group_id(command),
        split=str(command["split"]),
        global_ordinal=int(command["global_ordinal"]),
        split_ordinal=int(command["split_ordinal"]),
        requested_seed=int(command["requested_seed"]),
        resolved_seed=int(command["expected_resolved_seed"]),
        pair_id=str(command["pair_id"]),
        group_path=group_path,
        group_file_sha256=str(receipt["group_file_sha256"]),
        command_sha256=str(command["command_sha256"]),
        stage_receipt_sha256=stage_sha,
        group_receipt_sha256=receipt_sha,
        reset_receipt_sha256=reset_sha,
        candidate_accounting_sha256=accounting_sha,
        object_registry_sha256=registry_logical,
        pose_spec_sha256=pose_logical,
    )


def _audit_frozen_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if base.is_symlink() or base.stat().st_mode & 0o222:
            raise TrainingManifestV3Error("collection terminal tree is writable or linked")
        if _forbidden(PurePath(base)):
            raise TrainingManifestV3Error("collection tree enters a forbidden namespace")
        for name in [*names, *files]:
            path = base / name
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise TrainingManifestV3Error("collection terminal tree is writable or linked")
            if _forbidden(PurePath(path)):
                raise TrainingManifestV3Error("collection tree enters a forbidden namespace")


def _validate_detached_control_plane(
    *,
    root: Path,
    authority: Mapping[str, Any],
    plan_sha256: str,
    terminal_sha256: str,
) -> None:
    runner_root = _frozen_directory(root / "_runner", "runner control root")
    expected_names = {
        "stages",
        "staging",
        "static_plan.json",
        "state.json",
        "runner.log",
        "detach_receipt.json",
        "final_receipt.json",
    }
    if {path.name for path in runner_root.iterdir()} != expected_names:
        raise TrainingManifestV3Error("detached runner control inventory changed")
    staging = _frozen_directory(runner_root / "staging", "runner staging root")
    if any(staging.iterdir()):
        raise TrainingManifestV3Error("terminal runner staging root is not empty")
    _regular_file(runner_root / "runner.log", "detached runner log", frozen=True)
    state = _load_json(runner_root / "state.json", "runner terminal state")
    if state != {"status": TERMINAL_SUCCESS, "receipt_sha256": terminal_sha256}:
        raise TrainingManifestV3Error("runner terminal state changed")
    detach = _load_json(runner_root / "detach_receipt.json", "detach receipt")
    _verify_signed(detach, "detach_receipt_sha256", "detach receipt")
    command = detach.get("command")
    if (
        set(detach)
        != {
            "format", "status", "pid", "runner_plan_sha256", "output_root",
            "command", "resume_entrypoint_exposed", "detach_receipt_sha256",
        }
        or detach.get("format") != DETACH_RECEIPT_FORMAT
        or detach.get("status")
        != "detached_new_session_ppid1_required_before_gpu_or_worker"
        or type(detach.get("pid")) is not int
        or detach.get("pid") <= 0
        or detach.get("runner_plan_sha256") != plan_sha256
        or detach.get("output_root") != str(root)
        or detach.get("resume_entrypoint_exposed") is not False
        or not isinstance(command, list)
        or len(command) != 7
        or any(not isinstance(item, str) for item in command)
        or command[:3]
        != [
            authority["implementations"]["runtime_python"]["path"],
            authority["implementations"]["runner"]["path"],
            "serve-existing",
        ]
        or command[3:6] != ["--output-root", str(root), "--idle-interval-seconds"]
    ):
        raise TrainingManifestV3Error("detached server-side runner receipt changed")
    try:
        interval = float(command[6])
    except (TypeError, ValueError) as error:
        raise TrainingManifestV3Error("detach idle interval is invalid") from error
    if not math.isfinite(interval) or interval <= 0:
        raise TrainingManifestV3Error("detach idle interval is invalid")


def _validate_identity_closure(
    *,
    target: Mapping[str, Any],
    identity: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
) -> None:
    target_rows = target.get("groups")
    identity_rows = identity.get("selected_rows")
    if not isinstance(target_rows, list) or not isinstance(identity_rows, list):
        raise TrainingManifestV3Error("development300 identity rows are missing")
    if len(target_rows) != len(identity_rows) or len(commands) != TOTAL_GROUPS:
        raise TrainingManifestV3Error("development300 identity count changed")
    requested: set[int] = set()
    resolved: set[int] = set()
    split_ordinals: set[tuple[str, int]] = set()
    pair_ids: set[str] = set()
    paths: set[str] = set()
    for index, (target_row, identity_row, command) in enumerate(
        zip(target_rows, identity_rows, commands, strict=True)
    ):
        for row_name, row, resolved_field in (
            ("target", target_row, "expected_resolved_seed"),
            ("identity", identity_row, "resolved_seed"),
            ("command", command, "expected_resolved_seed"),
        ):
            if not isinstance(row, Mapping):
                raise TrainingManifestV3Error(f"{row_name} identity row is invalid")
            _require_int(row.get("global_ordinal"), f"{row_name} global ordinal")
            _require_int(row.get("split_ordinal"), f"{row_name} split ordinal")
            _require_int(row.get("requested_seed"), f"{row_name} requested seed")
            _require_int(row.get(resolved_field), f"{row_name} resolved seed")
        if (
            target_row["global_ordinal"] != index
            or identity_row["global_ordinal"] != index
            or command["global_ordinal"] != index
            or target_row["split"] != identity_row["split"] != command["split"]
            or target_row["split"] != command["split"]
            or target_row["split_ordinal"] != identity_row["split_ordinal"]
            or target_row["split_ordinal"] != command["split_ordinal"]
            or target_row["requested_seed"] != identity_row["requested_seed"]
            or target_row["requested_seed"] != command["requested_seed"]
            or target_row["expected_resolved_seed"] != identity_row["resolved_seed"]
            or target_row["expected_resolved_seed"] != command["expected_resolved_seed"]
            or identity_row["preregistered_logical_group_id"]
            != target_row["logical_group_id"]
            or identity_row["preregistered_identity_sha256"]
            != target_row["identity_sha256"]
            or identity_row["pair_id"] != command["pair_id"]
            or command.get("candidate_original_indices") != CANDIDATE_INDICES
            or any(
                type(candidate_index) is not int
                for candidate_index in command.get("candidate_original_indices", [])
            )
            or type(command.get("candidate_branch_count")) is not int
            or command.get("candidate_branch_count") != 4
        ):
            raise TrainingManifestV3Error("development300 identity closure changed")
        req = int(command["requested_seed"])
        res = int(command["expected_resolved_seed"])
        split_key = (str(command["split"]), int(command["split_ordinal"]))
        output = command.get("outputs")
        if not isinstance(output, Mapping):
            raise TrainingManifestV3Error("command output record is missing")
        group_path = str(output.get("group_hdf5", ""))
        if (
            req in requested
            or res in resolved
            or split_key in split_ordinals
            or command["pair_id"] in pair_ids
            or group_path in paths
        ):
            raise TrainingManifestV3Error("seed/group index/path uniqueness changed")
        requested.add(req)
        resolved.add(res)
        split_ordinals.add(split_key)
        pair_ids.add(str(command["pair_id"]))
        paths.add(group_path)
    if requested != resolved:
        raise TrainingManifestV3Error("requested/resolved identity sets differ")


def build_target_partition(groups: Sequence[FrozenGroup]) -> dict[str, Any]:
    adaptation = [
        item.logical_group_id
        for item in groups
        if item.split in {"adaptation_train", "adaptation_internal_validation"}
    ]
    formal = [
        item.logical_group_id
        for item in groups
        if item.split == "formal_target_validation"
    ]
    if len(adaptation) != TRAIN_COUNT + INTERNAL_VALIDATION_COUNT or len(formal) != FORMAL_VALIDATION_COUNT:
        raise TrainingManifestV3Error("target 110/190 partition changed")
    value: dict[str, Any] = {
        "format": TARGET_PARTITION_FORMAT,
        "status": "frozen_from_target_seed_manifest_before_hdf_access",
        "split_profile": SPLIT_PROFILE,
        "required_group_counts": {
            "adaptation": TRAIN_COUNT + INTERNAL_VALIDATION_COUNT,
            "formal_target_validation": FORMAL_VALIDATION_COUNT,
        },
        "adaptation": adaptation,
        "validation": formal,
        "evaluation": [],
        "evaluation_groups_included": 0,
        "hdf5_files_opened_before_partition_freeze": 0,
        "labels_read": False,
    }
    value["partition_sha256"] = canonical_sha256(value)
    return value


def build_external_split(
    groups: Sequence[FrozenGroup], partition: Mapping[str, Any], *, seed_base: int
) -> dict[str, Any]:
    train = [item.logical_group_id for item in groups if item.split == "adaptation_train"]
    internal = [
        item.logical_group_id
        for item in groups
        if item.split == "adaptation_internal_validation"
    ]
    formal = [
        item.logical_group_id
        for item in groups
        if item.split == "formal_target_validation"
    ]
    if not (
        len(train) == TRAIN_COUNT
        and len(internal) == INTERNAL_VALIDATION_COUNT
        and len(formal) == FORMAL_VALIDATION_COUNT
        and not (set(train) & set(internal) or set(train) & set(formal) or set(internal) & set(formal))
        and set(train) | set(internal) == set(partition["adaptation"])
        and set(formal) == set(partition["validation"])
    ):
        raise TrainingManifestV3Error("external 80/30/190 split changed")
    value: dict[str, Any] = {
        "format": EXTERNAL_SPLIT_FORMAT,
        "status": "frozen_label_blind_before_hdf_access",
        "split_profile": SPLIT_PROFILE,
        "required_trainer_group_counts": {
            "train": TRAIN_COUNT,
            "validation": INTERNAL_VALIDATION_COUNT,
            "test": FORMAL_VALIDATION_COUNT,
        },
        "algorithm": "exact_development300_preregistered_physical_membership_v1",
        "seed": seed_base,
        "train": train,
        "validation": internal,
        "test": formal,
        "source_partition_sha256": partition["partition_sha256"],
        "target_validation_used_for_training_or_internal_validation": False,
        "evaluation_groups_included": 0,
        "hdf5_files_opened_before_split_freeze": 0,
        "labels_read": False,
    }
    value["split_sha256"] = canonical_sha256(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except BaseException:
        raise


def materialize(
    *,
    collection_root: Path,
    terminal_receipt_path: Path,
    expected_terminal_receipt_file_sha256: str,
    expected_terminal_receipt_sha256: str,
    runner_authority_path: Path,
    expected_runner_authority_file_sha256: str,
    expected_runner_authority_sha256: str,
    target_preregistration_path: Path,
    expected_target_preregistration_file_sha256: str,
    expected_target_preregistration_sha256: str,
    identity_authority_path: Path,
    expected_identity_authority_file_sha256: str,
    expected_identity_authority_sha256: str,
    bound_trainer_path: Path,
    expected_bound_trainer_file_sha256: str,
    output_directory: Path,
    verify_runtime_files: bool = True,
) -> dict[str, Any]:
    """Validate the exact terminal collection and create all v3 outputs once."""

    root = _frozen_directory(collection_root, "collection terminal root")
    _audit_frozen_tree(root)
    terminal_source, terminal = _bound_json(
        terminal_receipt_path,
        expected_terminal_receipt_file_sha256,
        expected_terminal_receipt_sha256,
        "terminal_receipt_sha256",
        "collection terminal receipt",
    )
    if terminal_source != root / "_runner" / "final_receipt.json":
        raise TrainingManifestV3Error("terminal receipt is outside the collection root")

    authority_source, authority = _bound_json(
        runner_authority_path,
        expected_runner_authority_file_sha256,
        expected_runner_authority_sha256,
        AUTHORITY_SIGNATURE,
        "runner authority",
    )
    if authority.get("format") != AUTHORITY_FORMAT:
        raise TrainingManifestV3Error("runner authority format changed")
    try:
        decoded_runner = validate_runner_authority(
            authority, verify_runtime_files=verify_runtime_files
        )
    except Exception as error:
        raise TrainingManifestV3Error("runner authority is invalid") from error
    if (
        decoded_runner["runner_authority_sha256"] != expected_runner_authority_sha256
        or decoded_runner["output_root"] != root
    ):
        raise TrainingManifestV3Error("runner authority terminal binding changed")

    target_source, target = _bound_json(
        target_preregistration_path,
        expected_target_preregistration_file_sha256,
        expected_target_preregistration_sha256,
        "preregistration_sha256",
        "development300 target preregistration",
    )
    try:
        target_audit = validate_preregistration(target)
    except Exception as error:
        raise TrainingManifestV3Error("development300 target preregistration is invalid") from error
    if target_audit["preregistration_sha256"] != expected_target_preregistration_sha256:
        raise TrainingManifestV3Error("development300 target preregistration changed")

    identity_source, identity_authority = _bound_json(
        identity_authority_path,
        expected_identity_authority_file_sha256,
        expected_identity_authority_sha256,
        "identity_authority_sha256",
        "collection identity authority",
    )
    try:
        identity_audit = validate_collection_identity_authority(identity_authority)
    except Exception as error:
        raise TrainingManifestV3Error("collection identity authority is invalid") from error
    authority_identity = authority.get("collection_identity_authority")
    if (
        identity_audit["identity_authority_sha256"] != expected_identity_authority_sha256
        or decoded_runner["identity_authority"] != identity_authority
        or authority_identity
        != {
            "path": str(identity_source),
            "file_sha256": expected_identity_authority_file_sha256,
            "logical_sha256": expected_identity_authority_sha256,
        }
        or identity_authority.get("development300_preregistration_sha256")
        != target_audit["preregistration_sha256"]
        or identity_authority.get("partition_sha256") != target_audit["partition_sha256"]
    ):
        raise TrainingManifestV3Error("target preregistration/identity authority closure changed")

    commands = decoded_runner["commands"]
    _validate_identity_closure(target=target, identity=identity_authority, commands=commands)

    plan_path = root / "_runner" / "static_plan.json"
    plan = _load_json(plan_path, "runner static plan")
    plan_file_sha = metadata_file_sha256(plan_path, "runner static plan")
    plan_sha = _validate_static_plan(
        plan,
        root=root,
        authority_path=authority_source,
        authority_file_sha256=expected_runner_authority_file_sha256,
        authority=authority,
        commands=commands,
    )
    _validate_terminal(
        terminal,
        authority_sha256=expected_runner_authority_sha256,
        plan_sha256=plan_sha,
        expected_logical_sha256=expected_terminal_receipt_sha256,
    )
    _validate_detached_control_plane(
        root=root,
        authority=authority,
        plan_sha256=plan_sha,
        terminal_sha256=expected_terminal_receipt_sha256,
    )

    collection_prereg_source, collection_prereg_record = _validate_record_source(
        authority["collection_preregistration"],
        "collection preregistration",
        logical=True,
    )

    if {path.name for path in root.iterdir()} != {"_runner", *SPLIT_COUNTS}:
        raise TrainingManifestV3Error("collection root inventory changed")
    expected_seed_roots: dict[str, set[Path]] = {split: set() for split in SPLIT_COUNTS}
    groups: list[FrozenGroup] = []
    for command, identity_row in zip(
        commands, identity_authority["selected_rows"], strict=True
    ):
        seed_root = safe_path(command["outputs"]["seed_root"], "command seed root", must_exist=True)
        expected_seed_roots[str(command["split"])].add(seed_root)
        groups.append(
            _validate_group(
                root=root,
                command=command,
                identity_row=identity_row,
                authority_sha256=expected_runner_authority_sha256,
                plan_sha256=plan_sha,
                authority=authority,
                authority_path=authority_source,
                authority_file_sha256=expected_runner_authority_file_sha256,
                plan_path=plan_path,
                plan_file_sha256=plan_file_sha,
            )
        )
    for split, expected_roots in expected_seed_roots.items():
        split_root = _frozen_directory(root / split, f"{split} root")
        actual = set(split_root.iterdir())
        if actual != expected_roots or any(not path.is_dir() for path in actual):
            raise TrainingManifestV3Error("collection split directory inventory changed")
    stage_root = _frozen_directory(root / "_runner" / "stages", "stage receipt root")
    expected_stage_roots = {
        stage_root / f"stage_{index:03d}" for index in range(TOTAL_GROUPS)
    }
    if set(stage_root.iterdir()) != expected_stage_roots:
        raise TrainingManifestV3Error("stage receipt inventory is not gap-free")
    if canonical_sha256([item.stage_receipt_sha256 for item in groups]) != terminal[
        "stage_receipt_order_sha256"
    ]:
        raise TrainingManifestV3Error("terminal stage receipt order changed")

    logical_ids = [item.logical_group_id for item in groups]
    if len(set(logical_ids)) != TOTAL_GROUPS:
        raise TrainingManifestV3Error("logical group ids are not unique")
    partition = build_target_partition(groups)
    seed_base = _require_int(target["seed_generation"].get("seed_base"), "seed base")
    external_split = build_external_split(groups, partition, seed_base=seed_base)

    trainer_source = _regular_file(
        bound_trainer_path, "bound trainer implementation", frozen=False
    )
    expected_trainer = Path(__file__).resolve().parent / "train_smolvla_piper_schema6_embodiment_adapter.py"
    if (
        trainer_source != expected_trainer
        or not _is_sha(expected_bound_trainer_file_sha256)
        or metadata_file_sha256(trainer_source, "bound trainer implementation")
        != expected_bound_trainer_file_sha256
    ):
        raise TrainingManifestV3Error("bound trainer implementation changed")

    output = safe_path(output_directory, "v3 output directory", must_exist=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.resolve(strict=True)
    names = {role: output / filename for role, filename in OUTPUT_NAMES.items()}

    collector_lineage = canonical_sha256(
        {
            "runner": authority["implementations"]["runner"],
            "sealed_group_worker": authority["implementations"]["sealed_group_worker"],
            "dense_collector": authority["implementations"]["dense_collector"],
            "runtime_adapter": authority["implementations"]["runtime_adapter"],
            "support_closure_sha256": authority["implementations"]["support_closure_sha256"],
        }
    )
    event_spec_sha = authority["event_specification"]["file_sha256"]
    trainer_manifest: dict[str, Any] = {
        "format": TRAINER_MANIFEST_FORMAT,
        "status": "complete",
        "groups": [
            {
                "logical_group_id": item.logical_group_id,
                "requested_seed": item.requested_seed,
                "resolved_seed": item.resolved_seed,
                "task": TASK,
                "body": BODY,
                "policy": POLICY,
                "path": str(item.group_path),
                "file_sha256": item.group_file_sha256,
            }
            for item in groups
        ],
        "fresh_inputs_used": False,
        "sealed_test_labels_disclosed": False,
        "event_spec_sha256": event_spec_sha,
        "collector_lineage_sha256": collector_lineage,
        "target_seed_manifest_file_sha256": expected_target_preregistration_file_sha256,
        "target_seed_manifest_logical_sha256": expected_target_preregistration_sha256,
        "target_partition_sha256": partition["partition_sha256"],
        "expected_external_split_sha256": external_split["split_sha256"],
        "hdf5_files_opened_during_aggregation": 0,
    }
    trainer_manifest["manifest_sha256"] = canonical_sha256(trainer_manifest)
    _audit_embedded_paths(trainer_manifest, "trainer manifest")

    output.mkdir(mode=0o700)
    _atomic_json(names["partition"], partition)
    _atomic_json(names["split"], external_split)
    _atomic_json(names["manifest"], trainer_manifest)
    expected: dict[str, Any] = {
        "format": EXPECTED_FORMAT,
        "status": "complete_external_manifest_and_split_expectations",
        "split_profile": SPLIT_PROFILE,
        "trainer_compatible_manifest": _record(
            names["manifest"], trainer_manifest["manifest_sha256"]
        ),
        "target_partition": _record(names["partition"], partition["partition_sha256"]),
        "external_split": _record(names["split"], external_split["split_sha256"]),
        "bound_trainer_implementation": {
            "path": str(trainer_source),
            "file_sha256": expected_bound_trainer_file_sha256,
        },
        "required_trainer_group_counts": {
            "train": TRAIN_COUNT,
            "validation": INTERNAL_VALIDATION_COUNT,
            "test": FORMAL_VALIDATION_COUNT,
        },
        "direct_bound_trainer_execution_authorized": True,
        "hdf5_content_files_opened": 0,
        "labels_read": False,
    }
    expected["expected_receipt_sha256"] = canonical_sha256(expected)
    _atomic_json(names["expected"], expected)

    receipt: dict[str, Any] = {
        "format": FORMAT,
        "status": COMPLETE_STATUS,
        "training_inputs_complete": True,
        "training_authorized": True,
        "direct_bound_trainer_execution_authorized": True,
        "required_physical_group_counts": dict(SPLIT_COUNTS),
        "required_trainer_group_counts": expected["required_trainer_group_counts"],
        "collection_terminal_receipt": _record(
            terminal_source, expected_terminal_receipt_sha256
        ),
        "runner_static_plan": _record(plan_path, plan_sha),
        "runner_authority": _record(authority_source, expected_runner_authority_sha256),
        "collection_preregistration": {
            "path": str(collection_prereg_source),
            **{key: value for key, value in collection_prereg_record.items() if key != "path"},
        },
        "collection_identity_authority": _record(
            identity_source, expected_identity_authority_sha256
        ),
        "development300_target_preregistration": _record(
            target_source, expected_target_preregistration_sha256
        ),
        "bound_trainer_implementation": expected["bound_trainer_implementation"],
        "trainer_compatible_manifest": expected["trainer_compatible_manifest"],
        "target_partition": expected["target_partition"],
        "external_split": expected["external_split"],
        "expected_manifest_split_receipt": _record(
            names["expected"], expected["expected_receipt_sha256"]
        ),
        "event_specification": dict(authority["event_specification"]),
        "collector_lineage_sha256": collector_lineage,
        "group_count": TOTAL_GROUPS,
        "candidate_accounting_records": TOTAL_GROUPS * 4,
        "collection_lineage_sha256": canonical_sha256(
            [
                {
                    "logical_group_id": item.logical_group_id,
                    "command_sha256": item.command_sha256,
                    "stage_receipt_sha256": item.stage_receipt_sha256,
                    "group_receipt_sha256": item.group_receipt_sha256,
                    "reset_receipt_sha256": item.reset_receipt_sha256,
                    "candidate_accounting_sha256": item.candidate_accounting_sha256,
                    "object_registry_sha256": item.object_registry_sha256,
                    "pose_spec_sha256": item.pose_spec_sha256,
                    "group_file_sha256": item.group_file_sha256,
                }
                for item in groups
            ]
        ),
        "identity_membership_disjoint_full_coverage": True,
        "requested_seed_resolved_seed_group_index_and_path_unique": True,
        "formal_target_validation_hdf5_or_labels_opened": 0,
        "fresh_confirmation_or_test_inputs_accepted": False,
        "evaluation400_identity_or_execution_authorized": False,
        "hdf5_content_files_opened": 0,
        "labels_or_outcomes_read": False,
        "hdf5_sha_validation_mode": "terminal_bound_cross_signed_receipt_lstat_only_no_hdf_byte_open",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(names["receipt"], receipt)
    output.chmod(0o555)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--terminal-receipt", type=Path, required=True)
    parser.add_argument("--expected-terminal-receipt-file-sha256", required=True)
    parser.add_argument("--expected-terminal-receipt-sha256", required=True)
    parser.add_argument("--runner-authority", type=Path, required=True)
    parser.add_argument("--expected-runner-authority-file-sha256", required=True)
    parser.add_argument("--expected-runner-authority-sha256", required=True)
    parser.add_argument("--target-preregistration", type=Path, required=True)
    parser.add_argument("--expected-target-preregistration-file-sha256", required=True)
    parser.add_argument("--expected-target-preregistration-sha256", required=True)
    parser.add_argument("--identity-authority", type=Path, required=True)
    parser.add_argument("--expected-identity-authority-file-sha256", required=True)
    parser.add_argument("--expected-identity-authority-sha256", required=True)
    parser.add_argument("--bound-trainer", type=Path, required=True)
    parser.add_argument("--expected-bound-trainer-file-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = materialize(
        collection_root=args.collection_root,
        terminal_receipt_path=args.terminal_receipt,
        expected_terminal_receipt_file_sha256=args.expected_terminal_receipt_file_sha256,
        expected_terminal_receipt_sha256=args.expected_terminal_receipt_sha256,
        runner_authority_path=args.runner_authority,
        expected_runner_authority_file_sha256=args.expected_runner_authority_file_sha256,
        expected_runner_authority_sha256=args.expected_runner_authority_sha256,
        target_preregistration_path=args.target_preregistration,
        expected_target_preregistration_file_sha256=args.expected_target_preregistration_file_sha256,
        expected_target_preregistration_sha256=args.expected_target_preregistration_sha256,
        identity_authority_path=args.identity_authority,
        expected_identity_authority_file_sha256=args.expected_identity_authority_file_sha256,
        expected_identity_authority_sha256=args.expected_identity_authority_sha256,
        bound_trainer_path=args.bound_trainer,
        expected_bound_trainer_file_sha256=args.expected_bound_trainer_file_sha256,
        output_directory=args.output_directory,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLETE_STATUS",
    "EXPECTED_FORMAT",
    "EXTERNAL_SPLIT_FORMAT",
    "FORMAT",
    "OUTPUT_NAMES",
    "SPLIT_PROFILE",
    "TARGET_PARTITION_FORMAT",
    "TRAINER_MANIFEST_FORMAT",
    "TrainingManifestV3Error",
    "build_external_split",
    "build_target_partition",
    "materialize",
    "metadata_file_sha256",
    "safe_path",
]
