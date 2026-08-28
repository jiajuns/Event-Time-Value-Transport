#!/usr/bin/env python3
"""Authorize and run the exact Schema6 development300 collection.

The public runner/watcher never imports the dense collector and never parses a
group HDF5 file.  Each group is produced in a separately bound sealed worker
process.  The watcher accepts only an outcome-free terminal receipt, publishes
the staged directory atomically, and never retries, replaces, or resumes a
failed command.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence

from materialize_smolvla_piper_schema6_development300_identity_authority import (
    COLLECTION_IDENTITY_AUTHORITY_FORMAT,
    COLLECTION_PREREGISTRATION_FORMAT,
    file_sha256,
    validate_collection_identity_authority,
)
from preregister_smolvla_piper_schema6_target_development300 import (
    CANDIDATES_PER_GROUP,
    SPLIT_COUNTS,
    TOTAL_GROUPS,
    canonical_sha256,
)
from smolvla_piper_schema6_runtime_adapter_v2 import validate_runtime_contract


AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_runner_authority_v1"
)
AUTHORITY_STATUS = (
    "authorized_exact_300_group_sealed_worker_collection_no_evaluation"
)
STATIC_PLAN_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_runner_plan_v1"
)
DETACH_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_detach_receipt_v1"
)
STAGE_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_stage_receipt_v1"
)
TERMINAL_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_terminal_receipt_v1"
)
WORKER_GROUP_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_sealed_group_receipt_v1"
)
WORKER_RESET_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_worker_reset_receipt_v1"
)
WORKER_ACCOUNTING_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_candidate_accounting_v1"
)
AUTHORITY_SIGNATURE = "runner_authority_sha256"
PLAN_SIGNATURE = "runner_plan_sha256"
GROUP_SIGNATURE = "sealed_group_receipt_sha256"
TERMINAL_SUCCESS = "complete_exact_300_group_collection_formal190_sealed"
TERMINAL_FAILURE = "failed_closed_no_retry_no_resume"
EXPECTED_GPU_FRAGMENT = "RTX 4090"
FULL_HORIZON_STEPS = 200
SHA_CHARS = frozenset("0123456789abcdef")
SENSITIVE_COMPONENTS = {"fresh", "confirmation", "evaluation", "test", "testing"}
EXPECTED_GROUP_FILES = {
    "object_registry.json",
    "pose_quality_spec.json",
    "per_seed_reset_receipt.json",
    "candidate_accounting.json",
    "schema6_group.hdf5",
    "completed_group_receipt.json",
}
SUPPORT_IMPLEMENTATION_NAMES = (
    "collect_smolvla_etsf_event_branches.py",
    "collect_openvla_etsf_rollouts.py",
    "etsf_schema6_pose_quality.py",
    "execute_smolvla_piper_r6c_simulation_smoke.py",
    "freeze_smolvla_piper_schema6_development_collection.py",
    "launch_smolvla_piper_schema6_development_collection.py",
    "materialize_smolvla_piper_schema6_reset_contract.py",
    "preregister_smolvla_piper_schema6_target_development300.py",
    "resolve_smolvla_piper_target_reset_only.py",
    "run_smolvla_piper_r6d_direct_actor_smoke.py",
    "run_smolvla_piper_r6f_feasibility_smoke.py",
    "smolvla_piper_schema6_runtime_adapter_v2.py",
    "smolvla_piper_target_seed_manifest.py",
    "verify_smolvla_piper_zero_shot_preflight.py",
)


class Development300RunnerError(RuntimeError):
    """An authority, plan, execution, or sealing invariant failed closed."""


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise Development300RunnerError("signature field already exists")
    result[field] = canonical_sha256(result)
    return result


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise Development300RunnerError(f"{role} logical SHA changed")
    return str(recorded)


def _forbidden_path(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if lowered in SENSITIVE_COMPONENTS:
            return True
        if "fresh" in lowered or "confirmation" in lowered:
            return True
    return False


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise Development300RunnerError(f"{role} path is invalid")
    path = Path(os.path.abspath(os.path.expanduser(text)))
    if _forbidden_path(PurePath(path)):
        raise Development300RunnerError(f"{role} enters a forbidden namespace")
    resolved = path.resolve(strict=False)
    if _forbidden_path(PurePath(resolved)):
        raise Development300RunnerError(f"{role} resolves into a forbidden namespace")
    return resolved


def existing_file(value: str | os.PathLike[str], role: str) -> Path:
    path = safe_path(value, role)
    if path.is_symlink():
        raise Development300RunnerError(f"{role} must not be a symlink")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise Development300RunnerError(f"{role} is unavailable") from error
    if not stat.S_ISREG(mode) or path.suffix.casefold() in {".h5", ".hdf", ".hdf5"}:
        raise Development300RunnerError(f"{role} must be a non-HDF regular file")
    return path


def load_json(path: Path, role: str) -> dict[str, Any]:
    source = existing_file(path, role)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Development300RunnerError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise Development300RunnerError(f"{role} must contain an object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def replace_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path: Path, logical_sha256: str | None = None) -> dict[str, str]:
    result = {"path": str(path), "file_sha256": file_sha256(path)}
    if logical_sha256 is not None:
        result["logical_sha256"] = logical_sha256
    return result


def opaque_file_sha256(path: Path) -> str:
    """Hash an opaque artifact without interpreting its HDF5 contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(path: Path, expected_sha256: str, role: str) -> Path:
    source = existing_file(path, role)
    if not is_sha(expected_sha256) or file_sha256(source) != expected_sha256:
        raise Development300RunnerError(f"{role} file SHA changed")
    return source


def _new_root(path: Path, role: str) -> Path:
    root = safe_path(path, role)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.resolve(strict=True)
    return root


def _validate_runtime_v2b(
    value: Mapping[str, Any], *, verify_files: bool
) -> dict[str, Any]:
    try:
        runtime = validate_runtime_contract(value, verify_files=verify_files)
    except Exception as error:
        raise Development300RunnerError("runtime contract v2b is invalid") from error
    if (
        runtime.get("max_episode_steps") != FULL_HORIZON_STEPS
        or runtime.get("gpu_index") != 0
        or runtime.get("offline_model_loading") is not True
        or runtime.get("test_or_evaluation_execution_authorized") is not False
        or runtime.get("fresh_or_confirmation_inputs_accepted") is not False
    ):
        raise Development300RunnerError("runtime contract is not the 200-step v2b scope")
    return runtime


def _validate_identity_authority_source(
    path: Path, expected_file_sha256: str, expected_logical_sha256: str
) -> tuple[Path, dict[str, Any]]:
    source = _bound_file(path, expected_file_sha256, "collection identity authority")
    value = load_json(source, "collection identity authority")
    decoded = validate_collection_identity_authority(value)
    if (
        value.get("format") != COLLECTION_IDENTITY_AUTHORITY_FORMAT
        or decoded["identity_authority_sha256"] != expected_logical_sha256
    ):
        raise Development300RunnerError("collection identity authority binding changed")
    return source, value


def validate_collection_command(
    command: Mapping[str, Any],
    *,
    identity_row: Mapping[str, Any],
    future_root: Path,
    identity_authority_sha256: str,
    runtime_contract_sha256: str,
) -> str:
    command_sha = verify_signed(command, "command_sha256", "collection command")
    expected_fields = {
        "operation",
        "global_ordinal",
        "split",
        "split_ordinal",
        "requested_seed",
        "expected_resolved_seed",
        "pair_id",
        "candidate_original_indices",
        "candidate_branch_count",
        "outputs",
        "bindings",
        "capability",
        "command_sha256",
    }
    outputs = command.get("outputs")
    bindings = command.get("bindings")
    capability = command.get("capability")
    if (
        set(command) != expected_fields
        or command.get("operation") != "collect_schema6_four_candidate_group_v1"
        or type(command.get("global_ordinal")) is not int
        or type(command.get("split_ordinal")) is not int
        or command.get("split") not in SPLIT_COUNTS
        or type(command.get("requested_seed")) is not int
        or type(command.get("expected_resolved_seed")) is not int
        or not is_sha(command.get("pair_id"))
        or command.get("candidate_original_indices") != [0, 1, 2, 3]
        or command.get("candidate_branch_count") != CANDIDATES_PER_GROUP
        or not isinstance(outputs, Mapping)
        or set(outputs)
        != {"seed_root", "per_seed_reset_receipt", "group_hdf5", "completed_group_receipt"}
        or bindings
        != {
            "collection_identity_authority_sha256": identity_authority_sha256,
            "runtime_contract_sha256": runtime_contract_sha256,
        }
        or capability
        != {
            "execution_authorized_by_preregistration": False,
            "all_four_candidate_branches_required": True,
            "outcome_based_retry_or_replacement_allowed": False,
            "evaluation400": False,
        }
    ):
        raise Development300RunnerError("collection command scope changed")
    if (
        command["global_ordinal"] != identity_row["global_ordinal"]
        or command["split"] != identity_row["split"]
        or command["split_ordinal"] != identity_row["split_ordinal"]
        or command["requested_seed"] != identity_row["requested_seed"]
        or command["expected_resolved_seed"] != identity_row["resolved_seed"]
        or command["pair_id"] != identity_row["pair_id"]
    ):
        raise Development300RunnerError("collection command identity changed")
    seed_root = safe_path(outputs["seed_root"], "command seed root")
    expected_seed_root = (
        future_root
        / str(command["split"])
        / f"group_{int(command['split_ordinal']):03d}_seed_{int(command['requested_seed'])}"
    )
    if seed_root != expected_seed_root:
        raise Development300RunnerError("collection command seed root changed")
    expected_outputs = {
        "per_seed_reset_receipt": seed_root / "per_seed_reset_receipt.json",
        "group_hdf5": seed_root / "schema6_group.hdf5",
        "completed_group_receipt": seed_root / "completed_group_receipt.json",
    }
    if any(
        safe_path(outputs[name], f"command {name}") != expected
        for name, expected in expected_outputs.items()
    ):
        raise Development300RunnerError("collection command output path changed")
    return command_sha


def validate_collection_preregistration(
    value: Mapping[str, Any],
    *,
    identity_authority: Mapping[str, Any],
    identity_authority_path: Path,
    identity_authority_file_sha256: str,
) -> dict[str, Any]:
    logical = verify_signed(
        value,
        "collection_preregistration_sha256",
        "collection preregistration",
    )
    identity_logical = identity_authority["identity_authority_sha256"]
    identity_record = value.get("collection_identity_authority")
    commands = value.get("commands")
    boundary = value.get("execution_boundary")
    audit = value.get("materialization_audit")
    expected_fields = {
        "format",
        "status",
        "collection_identity_authority",
        "collection_identity_authority_sha256",
        "development300_preregistration_sha256",
        "partition_sha256",
        "runtime_contract_sha256",
        "future_collection_root",
        "command_count",
        "candidate_branches_per_command",
        "planned_candidate_branches",
        "ordered_splits",
        "split_counts",
        "commands",
        "execution_boundary",
        "materialization_audit",
        "collection_preregistration_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != COLLECTION_PREREGISTRATION_FORMAT
        or value.get("status")
        != "preregistered_300_four_candidate_groups_execution_not_authorized"
        or identity_record
        != {
            "path": str(identity_authority_path),
            "file_sha256": identity_authority_file_sha256,
            "logical_sha256": identity_logical,
        }
        or value.get("collection_identity_authority_sha256") != identity_logical
        or value.get("development300_preregistration_sha256")
        != identity_authority["development300_preregistration_sha256"]
        or value.get("partition_sha256") != identity_authority["partition_sha256"]
        or value.get("runtime_contract_sha256")
        != identity_authority["runtime_contract_sha256"]
        or value.get("command_count") != TOTAL_GROUPS
        or value.get("candidate_branches_per_command") != CANDIDATES_PER_GROUP
        or value.get("planned_candidate_branches")
        != TOTAL_GROUPS * CANDIDATES_PER_GROUP
        or value.get("ordered_splits") != list(SPLIT_COUNTS)
        or value.get("split_counts") != SPLIT_COUNTS
        or not isinstance(commands, list)
        or len(commands) != TOTAL_GROUPS
        or boundary
        != {
            "collection_execution_authorized": False,
            "separate_bound_runner_authority_required": True,
            "outcome_dependent_stop_retry_replacement_or_split_movement_allowed": False,
            "formal_target_validation_label_open_authorized": False,
            "evaluation400_commands_generated": 0,
            "evaluation400_identity_or_membership_read": False,
            "evaluation400_execution_authorized": False,
        }
        or audit
        != {
            "environment_reset_calls": 0,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "reward_success_event_outcome_trajectory_or_label_fields_read": 0,
            "hdf5_files_opened": 0,
        }
    ):
        raise Development300RunnerError("collection preregistration scope changed")
    future_root = safe_path(value["future_collection_root"], "future collection root")
    rows = identity_authority["selected_rows"]
    command_shas = [
        validate_collection_command(
            command,
            identity_row=rows[index],
            future_root=future_root,
            identity_authority_sha256=identity_logical,
            runtime_contract_sha256=identity_authority["runtime_contract_sha256"],
        )
        for index, command in enumerate(commands)
    ]
    if (
        [command["global_ordinal"] for command in commands] != list(range(TOTAL_GROUPS))
        or len(set(command_shas)) != TOTAL_GROUPS
        or {
            split: sum(command["split"] == split for command in commands)
            for split in SPLIT_COUNTS
        }
        != SPLIT_COUNTS
    ):
        raise Development300RunnerError("collection command inventory changed")
    return {
        "collection_preregistration_sha256": logical,
        "future_collection_root": str(future_root),
        "commands": [dict(command) for command in commands],
        "command_order_sha256": canonical_sha256(command_shas),
    }


def _implementation_record(path: Path, expected_sha256: str, role: str) -> dict[str, str]:
    source = _bound_file(path, expected_sha256, role)
    return _record(source)


def _support_closure(scripts_root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name in SUPPORT_IMPLEMENTATION_NAMES:
        path = existing_file(scripts_root / name, f"support implementation {name}")
        records[name] = _record(path)
    return records


def build_runner_authority(
    *,
    identity_authority_path: Path,
    expected_identity_authority_file_sha256: str,
    expected_identity_authority_sha256: str,
    collection_preregistration_path: Path,
    expected_collection_preregistration_file_sha256: str,
    expected_collection_preregistration_sha256: str,
    runtime_contract_path: Path,
    expected_runtime_contract_file_sha256: str,
    expected_runtime_contract_sha256: str,
    collector_path: Path,
    expected_collector_file_sha256: str,
    runtime_adapter_path: Path,
    expected_runtime_adapter_file_sha256: str,
    sealed_worker_path: Path,
    expected_sealed_worker_file_sha256: str,
    event_spec_path: Path,
    expected_event_spec_file_sha256: str,
    gpu_lock_path: Path,
    verify_runtime_files: bool = True,
) -> dict[str, Any]:
    identity_path, identity_authority = _validate_identity_authority_source(
        identity_authority_path,
        expected_identity_authority_file_sha256,
        expected_identity_authority_sha256,
    )
    prereg_path = _bound_file(
        collection_preregistration_path,
        expected_collection_preregistration_file_sha256,
        "collection preregistration",
    )
    preregistration = load_json(prereg_path, "collection preregistration")
    decoded_prereg = validate_collection_preregistration(
        preregistration,
        identity_authority=identity_authority,
        identity_authority_path=identity_path,
        identity_authority_file_sha256=expected_identity_authority_file_sha256,
    )
    if (
        decoded_prereg["collection_preregistration_sha256"]
        != expected_collection_preregistration_sha256
    ):
        raise Development300RunnerError("collection preregistration logical SHA changed")
    output_root = _new_root(
        Path(decoded_prereg["future_collection_root"]), "new collection output root"
    )
    runtime_path = _bound_file(
        runtime_contract_path,
        expected_runtime_contract_file_sha256,
        "runtime contract v2b",
    )
    runtime_source = load_json(runtime_path, "runtime contract v2b")
    runtime = _validate_runtime_v2b(
        runtime_source, verify_files=verify_runtime_files
    )
    if (
        runtime["runtime_contract_sha256"] != expected_runtime_contract_sha256
        or runtime["runtime_contract_sha256"]
        != identity_authority["runtime_contract_sha256"]
        or runtime["runtime_contract_sha256"]
        != preregistration["runtime_contract_sha256"]
    ):
        raise Development300RunnerError("runtime lineage changed")
    collector = _implementation_record(
        collector_path, expected_collector_file_sha256, "dense collector"
    )
    adapter = _implementation_record(
        runtime_adapter_path,
        expected_runtime_adapter_file_sha256,
        "runtime adapter",
    )
    worker = _implementation_record(
        sealed_worker_path,
        expected_sealed_worker_file_sha256,
        "sealed group worker",
    )
    runner_path = Path(__file__).resolve()
    scripts_root = runner_path.parent
    if (
        Path(worker["path"]) != scripts_root / "execute_smolvla_piper_schema6_development300_group.py"
        or Path(collector["path"])
        != scripts_root / "collect_smolvla_piper_schema6_dense_event_branches.py"
        or Path(adapter["path"])
        != scripts_root / "smolvla_piper_schema6_runtime_adapter_v2.py"
    ):
        raise Development300RunnerError(
            "collector/adapter/worker is outside the frozen scripts closure"
        )
    event_path = _bound_file(
        event_spec_path, expected_event_spec_file_sha256, "event specification"
    )
    load_json(event_path, "event specification")
    python_path = existing_file(Path(sys.executable).resolve(), "runtime Python")
    lock_path = safe_path(gpu_lock_path, "GPU lock")
    lock_path.parent.resolve(strict=True)
    support = _support_closure(scripts_root)
    if runtime_source != runtime:
        raise Development300RunnerError(
            "runtime contract source is not already canonical v2b"
        )
    command_plan = [
        {
            "global_ordinal": index,
            "split": command["split"],
            "command_sha256": command["command_sha256"],
            "pair_id": command["pair_id"],
            "candidate_original_indices": [0, 1, 2, 3],
            "candidate_accounting_records_required": 4,
            "final_seed_root": command["outputs"]["seed_root"],
            "outcome_based_retry_or_replacement_allowed": False,
        }
        for index, command in enumerate(decoded_prereg["commands"])
    ]
    base: dict[str, Any] = {
        "format": AUTHORITY_FORMAT,
        "status": AUTHORITY_STATUS,
        "collection_identity_authority": _record(
            identity_path, identity_authority["identity_authority_sha256"]
        ),
        "collection_preregistration": _record(
            prereg_path, decoded_prereg["collection_preregistration_sha256"]
        ),
        "runtime_contract_source": _record(
            runtime_path, runtime["runtime_contract_sha256"]
        ),
        "runtime_contract": runtime,
        "implementations": {
            "runner": _record(runner_path),
            "sealed_group_worker": worker,
            "dense_collector": collector,
            "runtime_adapter": adapter,
            "runtime_python": _record(python_path),
            "support_closure": support,
            "support_closure_sha256": canonical_sha256(support),
        },
        "event_specification": _record(event_path),
        "output_root": str(output_root),
        "gpu_contract": {
            "gpu_index": 0,
            "required_name_fragment": EXPECTED_GPU_FRAGMENT,
            "exclusive_lock_path": str(lock_path),
            "two_idle_observations_required_before_first_worker": True,
        },
        "exact_execution": {
            "group_count": TOTAL_GROUPS,
            "candidate_accounting_per_group": CANDIDATES_PER_GROUP,
            "planned_candidate_accounting_records": TOTAL_GROUPS
            * CANDIDATES_PER_GROUP,
            "split_counts": dict(SPLIT_COUNTS),
            "command_order_sha256": decoded_prereg["command_order_sha256"],
            "commands": command_plan,
            "retry_failed_command_allowed": False,
            "replacement_seed_allowed": False,
            "additional_seed_allowed": False,
            "resume_after_failure_allowed": False,
            "reentrant_execution_allowed": False,
        },
        "split_bridge": {
            "adaptation_train": "adaptation",
            "adaptation_internal_validation": "adaptation",
            "formal_target_validation": "validation",
            "mapping_is_interface_only_and_does_not_change_frozen_membership": True,
        },
        "label_boundary": {
            "sealed_worker_may_generate_group_payload": True,
            "runner_or_watcher_opens_group_hdf5": False,
            "runner_or_watcher_reads_success_event_outcome_or_label": False,
            "formal_target_validation_groups_sealed_immediately": 190,
            "formal_target_validation_label_open_authorized": False,
            "formal_target_validation_checkpoint_selection_authorized": False,
        },
        "permissions": {
            "production_collection_execution_authorized": True,
            "only_exact_preregistered_commands_authorized": True,
            "detached_server_side_execution_authorized": True,
            "fresh_or_confirmation_inputs_accepted": False,
            "evaluation400_identity_read_or_command_generation_authorized": False,
            "evaluation400_execution_authorized": False,
        },
    }
    return signed(base, AUTHORITY_SIGNATURE)


def _validate_implementation_record(
    record: Mapping[str, Any], role: str
) -> Path:
    if set(record) != {"path", "file_sha256"}:
        raise Development300RunnerError(f"{role} implementation record changed")
    return _bound_file(
        Path(str(record.get("path", ""))),
        str(record.get("file_sha256", "")),
        role,
    )


def validate_runner_authority(
    value: Mapping[str, Any], *, verify_runtime_files: bool
) -> dict[str, Any]:
    logical = verify_signed(value, AUTHORITY_SIGNATURE, "runner authority")
    expected_fields = {
        "format",
        "status",
        "collection_identity_authority",
        "collection_preregistration",
        "runtime_contract_source",
        "runtime_contract",
        "implementations",
        "event_specification",
        "output_root",
        "gpu_contract",
        "exact_execution",
        "split_bridge",
        "label_boundary",
        "permissions",
        AUTHORITY_SIGNATURE,
    }
    if (
        set(value) != expected_fields
        or value.get("format") != AUTHORITY_FORMAT
        or value.get("status") != AUTHORITY_STATUS
    ):
        raise Development300RunnerError("runner authority scope changed")
    implementations = value.get("implementations")
    if not isinstance(implementations, Mapping) or set(implementations) != {
        "runner",
        "sealed_group_worker",
        "dense_collector",
        "runtime_adapter",
        "runtime_python",
        "support_closure",
        "support_closure_sha256",
    }:
        raise Development300RunnerError("runner implementation closure changed")
    paths = {
        role: _validate_implementation_record(implementations[role], role)
        for role in (
            "runner",
            "sealed_group_worker",
            "dense_collector",
            "runtime_adapter",
            "runtime_python",
        )
    }
    if paths["runner"] != Path(__file__).resolve():
        raise Development300RunnerError("authority binds a different runner")
    scripts_root = paths["runner"].parent
    if (
        paths["sealed_group_worker"]
        != scripts_root / "execute_smolvla_piper_schema6_development300_group.py"
        or paths["dense_collector"]
        != scripts_root / "collect_smolvla_piper_schema6_dense_event_branches.py"
        or paths["runtime_adapter"]
        != scripts_root / "smolvla_piper_schema6_runtime_adapter_v2.py"
    ):
        raise Development300RunnerError("bound execution implementation path changed")
    support = implementations.get("support_closure")
    if (
        not isinstance(support, Mapping)
        or set(support) != set(SUPPORT_IMPLEMENTATION_NAMES)
        or implementations.get("support_closure_sha256")
        != canonical_sha256(support)
    ):
        raise Development300RunnerError("support implementation closure changed")
    for name, record in support.items():
        _validate_implementation_record(record, f"support implementation {name}")
    runtime = _validate_runtime_v2b(
        value.get("runtime_contract", {}), verify_files=verify_runtime_files
    )
    runtime_source_record = value.get("runtime_contract_source")
    if not isinstance(runtime_source_record, Mapping):
        raise Development300RunnerError("runtime source record is missing")
    runtime_source_path = _bound_file(
        Path(str(runtime_source_record.get("path", ""))),
        str(runtime_source_record.get("file_sha256", "")),
        "runtime contract source",
    )
    if (
        runtime_source_record.get("logical_sha256")
        != runtime["runtime_contract_sha256"]
        or load_json(runtime_source_path, "runtime contract source")
        != value["runtime_contract"]
    ):
        raise Development300RunnerError("runtime source binding changed")
    identity_record = value.get("collection_identity_authority")
    prereg_record = value.get("collection_preregistration")
    if not isinstance(identity_record, Mapping) or not isinstance(prereg_record, Mapping):
        raise Development300RunnerError("collection source records are missing")
    identity_path, identity_authority = _validate_identity_authority_source(
        Path(str(identity_record.get("path", ""))),
        str(identity_record.get("file_sha256", "")),
        str(identity_record.get("logical_sha256", "")),
    )
    prereg_path = _bound_file(
        Path(str(prereg_record.get("path", ""))),
        str(prereg_record.get("file_sha256", "")),
        "collection preregistration",
    )
    preregistration = load_json(prereg_path, "collection preregistration")
    decoded_prereg = validate_collection_preregistration(
        preregistration,
        identity_authority=identity_authority,
        identity_authority_path=identity_path,
        identity_authority_file_sha256=str(identity_record["file_sha256"]),
    )
    if (
        prereg_record.get("logical_sha256")
        != decoded_prereg["collection_preregistration_sha256"]
        or runtime["runtime_contract_sha256"]
        != identity_authority["runtime_contract_sha256"]
    ):
        raise Development300RunnerError("collection/runtime lineage changed")
    event_record = value.get("event_specification")
    if not isinstance(event_record, Mapping):
        raise Development300RunnerError("event specification record is missing")
    event_path = _bound_file(
        Path(str(event_record.get("path", ""))),
        str(event_record.get("file_sha256", "")),
        "event specification",
    )
    load_json(event_path, "event specification")
    exact = value.get("exact_execution")
    expected_plan = [
        {
            "global_ordinal": index,
            "split": command["split"],
            "command_sha256": command["command_sha256"],
            "pair_id": command["pair_id"],
            "candidate_original_indices": [0, 1, 2, 3],
            "candidate_accounting_records_required": 4,
            "final_seed_root": command["outputs"]["seed_root"],
            "outcome_based_retry_or_replacement_allowed": False,
        }
        for index, command in enumerate(decoded_prereg["commands"])
    ]
    if exact != {
        "group_count": TOTAL_GROUPS,
        "candidate_accounting_per_group": CANDIDATES_PER_GROUP,
        "planned_candidate_accounting_records": TOTAL_GROUPS * CANDIDATES_PER_GROUP,
        "split_counts": dict(SPLIT_COUNTS),
        "command_order_sha256": decoded_prereg["command_order_sha256"],
        "commands": expected_plan,
        "retry_failed_command_allowed": False,
        "replacement_seed_allowed": False,
        "additional_seed_allowed": False,
        "resume_after_failure_allowed": False,
        "reentrant_execution_allowed": False,
    }:
        raise Development300RunnerError("exact execution plan changed")
    if value.get("split_bridge") != {
        "adaptation_train": "adaptation",
        "adaptation_internal_validation": "adaptation",
        "formal_target_validation": "validation",
        "mapping_is_interface_only_and_does_not_change_frozen_membership": True,
    }:
        raise Development300RunnerError("runtime split bridge changed")
    if value.get("label_boundary") != {
        "sealed_worker_may_generate_group_payload": True,
        "runner_or_watcher_opens_group_hdf5": False,
        "runner_or_watcher_reads_success_event_outcome_or_label": False,
        "formal_target_validation_groups_sealed_immediately": 190,
        "formal_target_validation_label_open_authorized": False,
        "formal_target_validation_checkpoint_selection_authorized": False,
    }:
        raise Development300RunnerError("formal label boundary changed")
    if value.get("permissions") != {
        "production_collection_execution_authorized": True,
        "only_exact_preregistered_commands_authorized": True,
        "detached_server_side_execution_authorized": True,
        "fresh_or_confirmation_inputs_accepted": False,
        "evaluation400_identity_read_or_command_generation_authorized": False,
        "evaluation400_execution_authorized": False,
    }:
        raise Development300RunnerError("runner permissions changed")
    output_root = safe_path(value["output_root"], "collection output root")
    if output_root != Path(decoded_prereg["future_collection_root"]):
        raise Development300RunnerError("collection output root changed")
    gpu = value.get("gpu_contract")
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("gpu_index") != 0
        or gpu.get("required_name_fragment") != EXPECTED_GPU_FRAGMENT
        or gpu.get("two_idle_observations_required_before_first_worker") is not True
    ):
        raise Development300RunnerError("GPU contract changed")
    safe_path(str(gpu.get("exclusive_lock_path", "")), "GPU lock")
    return {
        "runner_authority_sha256": logical,
        "identity_authority": identity_authority,
        "preregistration": preregistration,
        "commands": decoded_prereg["commands"],
        "output_root": output_root,
        "runtime_contract": runtime,
        "paths": paths,
        "event_specification_path": event_path,
    }


def build_static_plan(
    *, authority_path: Path, expected_authority_file_sha256: str
) -> dict[str, Any]:
    source = _bound_file(
        authority_path, expected_authority_file_sha256, "runner authority"
    )
    authority = load_json(source, "runner authority")
    decoded = validate_runner_authority(authority, verify_runtime_files=True)
    _new_root(decoded["output_root"], "new collection output root")
    base: dict[str, Any] = {
        "format": STATIC_PLAN_FORMAT,
        "status": "dry_run_exact_plan_validated_output_unclaimed",
        "runner_authority": _record(
            source, decoded["runner_authority_sha256"]
        ),
        "output_root": str(decoded["output_root"]),
        "runner_implementation": _record(Path(__file__).resolve()),
        "sealed_worker_implementation": dict(
            authority["implementations"]["sealed_group_worker"]
        ),
        "runtime_python": dict(authority["implementations"]["runtime_python"]),
        "command_count": TOTAL_GROUPS,
        "candidate_accounting_records": TOTAL_GROUPS * CANDIDATES_PER_GROUP,
        "command_order_sha256": authority["exact_execution"][
            "command_order_sha256"
        ],
        "command_sha256": [command["command_sha256"] for command in decoded["commands"]],
        "split_counts": dict(SPLIT_COUNTS),
        "retry_or_resume_allowed": False,
        "evaluation400_commands": 0,
        "formal_label_open_authorized": False,
    }
    return signed(base, PLAN_SIGNATURE)


def gpu_audit(
    gpu_index: int,
    run_text: Callable[[Sequence[str]], str] | None = None,
) -> dict[str, Any]:
    runner = run_text or (
        lambda command: subprocess.run(
            command, check=True, text=True, capture_output=True
        ).stdout
    )
    identity = runner(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,uuid",
            "--format=csv,noheader",
        ]
    ).strip().split(",", 1)
    if len(identity) != 2 or EXPECTED_GPU_FRAGMENT not in identity[0].strip():
        raise Development300RunnerError("designated GPU is not an RTX 4090")
    raw_pids = runner(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    pids = sorted(
        {int(line.strip()) for line in raw_pids.splitlines() if line.strip().isdigit()}
    )
    return {
        "gpu_index": gpu_index,
        "name": identity[0].strip(),
        "uuid": identity[1].strip(),
        "compute_pids": pids,
    }


def wait_two_idle(
    *, interval_seconds: float, sleep: Callable[[float], None] = time.sleep
) -> list[dict[str, Any]]:
    consecutive: list[dict[str, Any]] = []
    while len(consecutive) < 2:
        audit = gpu_audit(0)
        if audit["compute_pids"]:
            consecutive.clear()
        else:
            if consecutive and consecutive[0]["uuid"] != audit["uuid"]:
                raise Development300RunnerError("GPU identity changed")
            consecutive.append(audit)
        if len(consecutive) < 2:
            sleep(interval_seconds)
    return consecutive


def _freeze_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            path = base / name
            if path.is_symlink():
                raise Development300RunnerError("output tree contains a symlink")
            current = stat.S_IMODE(path.stat().st_mode)
            path.chmod(0o400 if current == 0o400 else 0o444)
        for name in names:
            path = base / name
            if path.is_symlink():
                raise Development300RunnerError("output tree contains a symlink")
            current = stat.S_IMODE(path.stat().st_mode)
            path.chmod(0o500 if current == 0o500 else 0o555)
        current_root = stat.S_IMODE(base.stat().st_mode)
        base.chmod(0o500 if current_root == 0o500 else 0o555)


def _seal_failed_payload(root: Path, *, formal: bool) -> None:
    if not root.exists():
        return
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            path = base / name
            if path.is_symlink():
                raise Development300RunnerError("failed payload contains a symlink")
            path.chmod(0o400 if formal else 0o444)
        for name in names:
            path = base / name
            if path.is_symlink():
                raise Development300RunnerError("failed payload contains a symlink")
            path.chmod(0o500 if formal else 0o555)
        base.chmod(0o500 if formal else 0o555)


def _verify_group_receipt(
    receipt: Mapping[str, Any],
    *, command: Mapping[str, Any],
    authority_sha256: str,
    formal: bool,
) -> str:
    logical = verify_signed(receipt, GROUP_SIGNATURE, "sealed group receipt")
    expected_fields = {
        "format",
        "status",
        "runner_authority_sha256",
        "command_sha256",
        "global_ordinal",
        "split",
        "requested_seed",
        "resolved_seed",
        "pair_id",
        "candidate_original_indices",
        "candidate_accounting_records",
        "candidate_accounting_sha256",
        "per_seed_reset_receipt_sha256",
        "object_registry_sha256",
        "pose_spec_sha256",
        "group_file_sha256",
        "formal_payload_sealed",
        "outcome_or_label_fields_disclosed_to_runner",
        "evaluation400",
        GROUP_SIGNATURE,
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("format") != WORKER_GROUP_RECEIPT_FORMAT
        or receipt.get("status") != "complete_exact_four_candidate_accounting"
        or receipt.get("runner_authority_sha256") != authority_sha256
        or receipt.get("command_sha256") != command["command_sha256"]
        or receipt.get("global_ordinal") != command["global_ordinal"]
        or receipt.get("split") != command["split"]
        or receipt.get("requested_seed") != command["requested_seed"]
        or receipt.get("resolved_seed") != command["expected_resolved_seed"]
        or receipt.get("pair_id") != command["pair_id"]
        or receipt.get("candidate_original_indices") != [0, 1, 2, 3]
        or receipt.get("candidate_accounting_records") != 4
        or not is_sha(receipt.get("candidate_accounting_sha256"))
        or not is_sha(receipt.get("per_seed_reset_receipt_sha256"))
        or not is_sha(receipt.get("object_registry_sha256"))
        or not is_sha(receipt.get("pose_spec_sha256"))
        or not is_sha(receipt.get("group_file_sha256"))
        or receipt.get("formal_payload_sealed") is not formal
        or receipt.get("outcome_or_label_fields_disclosed_to_runner") is not False
        or receipt.get("evaluation400") is not False
    ):
        raise Development300RunnerError("sealed group receipt scope changed")
    return logical


def validate_staged_group(
    payload: Path,
    *, command: Mapping[str, Any], authority_sha256: str,
) -> tuple[dict[str, Any], str]:
    if payload.is_symlink() or not payload.is_dir():
        raise Development300RunnerError("sealed worker payload is missing")
    names = {path.name for path in payload.iterdir()}
    if names != EXPECTED_GROUP_FILES or any(path.is_symlink() for path in payload.iterdir()):
        raise Development300RunnerError("sealed worker output inventory changed")
    receipt_path = payload / "completed_group_receipt.json"
    receipt = load_json(receipt_path, "sealed group receipt")
    formal = command["split"] == "formal_target_validation"
    logical = _verify_group_receipt(
        receipt,
        command=command,
        authority_sha256=authority_sha256,
        formal=formal,
    )
    group_path = payload / "schema6_group.hdf5"
    # Byte hashing binds the opaque payload; the watcher never parses HDF5.
    if opaque_file_sha256(group_path) != receipt["group_file_sha256"]:
        raise Development300RunnerError("sealed group file SHA changed")
    reset = load_json(payload / "per_seed_reset_receipt.json", "reset receipt")
    reset_sha = verify_signed(reset, "reset_receipt_sha256", "reset receipt")
    if (
        set(reset)
        != {
            "format",
            "status",
            "runner_authority_sha256",
            "runner_plan_sha256",
            "command_sha256",
            "global_ordinal",
            "split",
            "requested_seed",
            "resolved_seed",
            "pair_id",
            "initial_scene_state_sha256",
            "initial_measured_joint_state_sha256",
            "initial_commanded_drive_target_sha256",
            "object_registry_sha256",
            "pose_spec_sha256",
            "identity_validation_count_before_policy_query",
            "policy_queries_before_reset_receipt",
            "outcome_or_label_read_before_reset_receipt",
            "evaluation400",
            "reset_receipt_sha256",
        }
        or reset.get("format") != WORKER_RESET_RECEIPT_FORMAT
        or reset.get("status") != "identity_verified_before_first_policy_query"
        or reset.get("runner_authority_sha256") != authority_sha256
        or not is_sha(reset.get("runner_plan_sha256"))
        or reset.get("command_sha256") != command["command_sha256"]
        or reset.get("global_ordinal") != command["global_ordinal"]
        or reset.get("split") != command["split"]
        or reset.get("requested_seed") != command["requested_seed"]
        or reset.get("resolved_seed") != command["expected_resolved_seed"]
        or reset.get("pair_id") != command["pair_id"]
        or not is_sha(reset.get("initial_scene_state_sha256"))
        or not is_sha(reset.get("initial_measured_joint_state_sha256"))
        or not is_sha(reset.get("initial_commanded_drive_target_sha256"))
        or reset.get("object_registry_sha256") != receipt["object_registry_sha256"]
        or reset.get("pose_spec_sha256") != receipt["pose_spec_sha256"]
        or reset.get("identity_validation_count_before_policy_query") != 1
        or reset.get("policy_queries_before_reset_receipt") != 0
        or reset.get("outcome_or_label_read_before_reset_receipt") is not False
        or reset.get("evaluation400") is not False
        or reset_sha != receipt["per_seed_reset_receipt_sha256"]
    ):
        raise Development300RunnerError("reset receipt binding changed")
    accounting = load_json(payload / "candidate_accounting.json", "candidate accounting")
    accounting_sha = verify_signed(
        accounting, "candidate_accounting_sha256", "candidate accounting"
    )
    records = accounting.get("records")
    if (
        set(accounting)
        != {
            "format",
            "status",
            "command_sha256",
            "candidate_original_indices",
            "records",
            "success_event_outcome_or_label_included",
            "candidate_accounting_sha256",
        }
        or accounting.get("format") != WORKER_ACCOUNTING_FORMAT
        or accounting.get("status") != "complete_four_original_candidate_records"
        or accounting.get("command_sha256") != command["command_sha256"]
        or accounting.get("candidate_original_indices") != [0, 1, 2, 3]
        or accounting.get("success_event_outcome_or_label_included") is not False
        or not isinstance(records, list)
        or len(records) != 4
        or [row.get("original_candidate_index") for row in records]
        != [0, 1, 2, 3]
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "original_candidate_index",
                "native_action_sha256",
                "feasible",
                "executed",
                "right_censored",
                "execution_status",
            }
            or not is_sha(row.get("native_action_sha256"))
            or type(row.get("feasible")) is not bool
            or row.get("executed") is not row.get("feasible")
            or row.get("right_censored") is row.get("feasible")
            or row.get("execution_status")
            != (
                "executed_legal_branch"
                if row.get("feasible")
                else "nonexecuted_censored_infeasible"
            )
            for row in records
        )
        or accounting_sha != receipt["candidate_accounting_sha256"]
    ):
        raise Development300RunnerError("candidate accounting binding changed")
    if formal:
        if stat.S_IMODE(payload.stat().st_mode) != 0o500 or any(
            stat.S_IMODE(path.stat().st_mode) != 0o400 for path in payload.iterdir()
        ):
            raise Development300RunnerError("formal payload is not sealed")
    return receipt, logical


def _stage_worker_command(
    *,
    authority: Mapping[str, Any],
    authority_path: Path,
    authority_file_sha256: str,
    static_plan_path: Path,
    static_plan_file_sha256: str,
    index: int,
    payload: Path,
) -> list[str]:
    implementations = authority["implementations"]
    return [
        implementations["runtime_python"]["path"],
        implementations["sealed_group_worker"]["path"],
        "collect-one",
        "--authority",
        str(authority_path),
        "--authority-file-sha256",
        authority_file_sha256,
        "--static-plan",
        str(static_plan_path),
        "--static-plan-file-sha256",
        static_plan_file_sha256,
        "--global-ordinal",
        str(index),
        "--staging-payload",
        str(payload),
    ]


def _run_worker(
    *, command: Sequence[str], log_path: Path, authority_path: Path
) -> int:
    environment = dict(os.environ)
    environment.update(
        {
            "ETSF_SCHEMA6_V2_EXECUTION_AUTHORITY": str(authority_path),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
        return process.wait()


def execute_exact_sequence(
    *,
    root: Path,
    plan: Mapping[str, Any],
    authority_path: Path,
    authority_file_sha256: str,
    launch_worker: Callable[..., int] = _run_worker,
) -> dict[str, Any]:
    authority = load_json(authority_path, "runner authority")
    decoded = validate_runner_authority(authority, verify_runtime_files=True)
    commands = decoded["commands"]
    if len(commands) != TOTAL_GROUPS:
        raise Development300RunnerError("exact command count changed")
    plan_path = root / "_runner" / "static_plan.json"
    plan_file_sha = file_sha256(plan_path)
    stages = root / "_runner" / "stages"
    staging = root / "_runner" / "staging"
    successful_receipts: list[str] = []
    for index, command in enumerate(commands):
        stage = stages / f"stage_{index:03d}"
        payload_parent = staging / f"stage_{index:03d}"
        if stage.exists() or payload_parent.exists():
            raise Development300RunnerError("runner is non-reentrant")
        stage.mkdir(mode=0o755)
        payload_parent.mkdir(mode=0o700)
        payload = payload_parent / "payload"
        worker_command = _stage_worker_command(
            authority=authority,
            authority_path=authority_path,
            authority_file_sha256=authority_file_sha256,
            static_plan_path=plan_path,
            static_plan_file_sha256=plan_file_sha,
            index=index,
            payload=payload,
        )
        launch = signed(
            {
                "format": STAGE_RECEIPT_FORMAT,
                "status": "launching_exact_once",
                "global_ordinal": index,
                "command_sha256": command["command_sha256"],
                "worker_command": worker_command,
                "retry_allowed": False,
            },
            "stage_receipt_sha256",
        )
        atomic_json(stage / "launch_receipt.json", launch)
        log_path = stage / "worker.log"
        returncode = 1
        error_type: str | None = None
        try:
            returncode = launch_worker(
                command=worker_command,
                log_path=log_path,
                authority_path=authority_path,
            )
            if returncode != 0:
                raise Development300RunnerError("sealed worker returned nonzero")
            receipt, receipt_sha = validate_staged_group(
                payload,
                command=command,
                authority_sha256=decoded["runner_authority_sha256"],
            )
            final_root = safe_path(command["outputs"]["seed_root"], "final seed root")
            if final_root.exists() or final_root.is_symlink():
                raise Development300RunnerError("final seed root already exists")
            final_root.parent.mkdir(parents=True, exist_ok=True)
            # This filesystem requires owner-write on a directory inode while
            # it is renamed.  Payload files remain sealed throughout; restore
            # the exact directory mode immediately after the atomic rename.
            formal = command["split"] == "formal_target_validation"
            payload.chmod(0o700)
            os.rename(payload, final_root)
            final_root.chmod(0o500 if formal else 0o555)
            payload_parent.rmdir()
            success = signed(
                {
                    "format": STAGE_RECEIPT_FORMAT,
                    "status": "published_exact_once",
                    "global_ordinal": index,
                    "command_sha256": command["command_sha256"],
                    "sealed_group_receipt_sha256": receipt_sha,
                    "group_file_sha256": receipt["group_file_sha256"],
                    "formal_payload_sealed": receipt["formal_payload_sealed"],
                    "retry_performed": False,
                },
                "stage_receipt_sha256",
            )
            atomic_json(stage / "terminal_receipt.json", success)
            successful_receipts.append(success["stage_receipt_sha256"])
            replace_json(
                root / "_runner" / "state.json",
                {
                    "status": "running_exact_prefix",
                    "completed_groups": index + 1,
                    "last_command_sha256": command["command_sha256"],
                },
            )
        except Exception as error:
            error_type = type(error).__name__
            if payload_parent.exists():
                _seal_failed_payload(
                    payload_parent,
                    formal=command["split"] == "formal_target_validation",
                )
            failure = signed(
                {
                    "format": STAGE_RECEIPT_FORMAT,
                    "status": TERMINAL_FAILURE,
                    "global_ordinal": index,
                    "command_sha256": command["command_sha256"],
                    "returncode": returncode,
                    "error_type": error_type,
                    "error_message_disclosed": False,
                    "retry_performed": False,
                    "resume_authorized": False,
                    "worker_log_sha256": file_sha256(log_path)
                    if log_path.is_file()
                    else None,
                },
                "stage_receipt_sha256",
            )
            atomic_json(stage / "terminal_receipt.json", failure)
            terminal_failure = signed(
                {
                    "format": TERMINAL_RECEIPT_FORMAT,
                    "status": TERMINAL_FAILURE,
                    "runner_authority_sha256": decoded["runner_authority_sha256"],
                    "runner_plan_sha256": plan[PLAN_SIGNATURE],
                    "completed_groups": len(successful_receipts),
                    "failed_global_ordinal": index,
                    "failed_command_sha256": command["command_sha256"],
                    "stage_receipt_sha256": failure["stage_receipt_sha256"],
                    "retry_or_resume_authorized": False,
                    "formal_label_opened_by_runner_or_watcher": False,
                    "evaluation400_commands_executed": 0,
                },
                "terminal_receipt_sha256",
            )
            atomic_json(root / "_runner" / "final_receipt.json", terminal_failure)
            replace_json(
                root / "_runner" / "state.json",
                {
                    "status": TERMINAL_FAILURE,
                    "receipt_sha256": terminal_failure["terminal_receipt_sha256"],
                },
            )
            _freeze_tree(root)
            raise Development300RunnerError(
                "collection failed closed; retry and resume are forbidden"
            ) from None
    terminal = signed(
        {
            "format": TERMINAL_RECEIPT_FORMAT,
            "status": TERMINAL_SUCCESS,
            "runner_authority_sha256": decoded["runner_authority_sha256"],
            "runner_plan_sha256": plan[PLAN_SIGNATURE],
            "completed_groups": TOTAL_GROUPS,
            "candidate_accounting_records": TOTAL_GROUPS
            * CANDIDATES_PER_GROUP,
            "split_counts": dict(SPLIT_COUNTS),
            "formal_payloads_sealed": 190,
            "gap_free_exact_command_order": True,
            "retry_replacement_additional_seed_or_resume_performed": False,
            "formal_label_opened_by_runner_or_watcher": False,
            "evaluation400_commands_executed": 0,
            "stage_receipt_order_sha256": canonical_sha256(successful_receipts),
        },
        "terminal_receipt_sha256",
    )
    atomic_json(root / "_runner" / "final_receipt.json", terminal)
    replace_json(
        root / "_runner" / "state.json",
        {"status": TERMINAL_SUCCESS, "receipt_sha256": terminal["terminal_receipt_sha256"]},
    )
    _freeze_tree(root)
    return terminal


def wait_for_detach(
    *, getppid: Callable[[], int] = os.getppid, sleep: Callable[[float], None] = time.sleep
) -> None:
    while getppid() != 1:
        sleep(0.1)


def claim_output_and_detach(
    *,
    authority_path: Path,
    expected_authority_file_sha256: str,
    idle_interval_seconds: float,
) -> dict[str, Any]:
    plan = build_static_plan(
        authority_path=authority_path,
        expected_authority_file_sha256=expected_authority_file_sha256,
    )
    root = _new_root(Path(plan["output_root"]), "new collection output root")
    root.mkdir(mode=0o755)
    runner_root = root / "_runner"
    runner_root.mkdir(mode=0o700)
    (runner_root / "stages").mkdir(mode=0o700)
    (runner_root / "staging").mkdir(mode=0o700)
    atomic_json(runner_root / "static_plan.json", plan)
    replace_json(runner_root / "state.json", {"status": "claimed_not_started"})
    command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "serve-existing",
        "--output-root",
        str(root),
        "--idle-interval-seconds",
        str(idle_interval_seconds),
    ]
    log_path = runner_root / "runner.log"
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except Exception as error:
        failure = signed(
            {
                "format": TERMINAL_RECEIPT_FORMAT,
                "status": TERMINAL_FAILURE,
                "phase": "detach_spawn",
                "error_type": type(error).__name__,
                "error_message_disclosed": False,
                "runner_plan_sha256": plan[PLAN_SIGNATURE],
                "completed_groups": 0,
                "retry_or_resume_authorized": False,
                "formal_label_opened_by_runner_or_watcher": False,
                "evaluation400_commands_executed": 0,
            },
            "terminal_receipt_sha256",
        )
        atomic_json(runner_root / "final_receipt.json", failure)
        _freeze_tree(root)
        raise Development300RunnerError("detached runner spawn failed closed") from None
    receipt = signed(
        {
            "format": DETACH_RECEIPT_FORMAT,
            "status": "detached_new_session_ppid1_required_before_gpu_or_worker",
            "pid": process.pid,
            "runner_plan_sha256": plan[PLAN_SIGNATURE],
            "output_root": str(root),
            "command": command,
            "resume_entrypoint_exposed": False,
        },
        "detach_receipt_sha256",
    )
    atomic_json(runner_root / "detach_receipt.json", receipt)
    return receipt


def serve_existing(root: Path, *, idle_interval_seconds: float) -> dict[str, Any]:
    wait_for_detach()
    root = safe_path(root, "claimed collection output root").resolve(strict=True)
    runner_root = root / "_runner"
    plan_path = existing_file(runner_root / "static_plan.json", "runner static plan")
    plan = load_json(plan_path, "runner static plan")
    plan_sha = verify_signed(plan, PLAN_SIGNATURE, "runner static plan")
    if plan.get("format") != STATIC_PLAN_FORMAT or plan.get("output_root") != str(root):
        raise Development300RunnerError("runner static plan changed")
    detach = load_json(runner_root / "detach_receipt.json", "detach receipt")
    verify_signed(detach, "detach_receipt_sha256", "detach receipt")
    if (
        detach.get("format") != DETACH_RECEIPT_FORMAT
        or detach.get("status")
        != "detached_new_session_ppid1_required_before_gpu_or_worker"
        or detach.get("pid") != os.getpid()
        or detach.get("runner_plan_sha256") != plan_sha
        or detach.get("output_root") != str(root)
        or detach.get("resume_entrypoint_exposed") is not False
    ):
        raise Development300RunnerError("detach receipt binding changed")
    authority_record = plan["runner_authority"]
    authority_path = _bound_file(
        Path(authority_record["path"]),
        authority_record["file_sha256"],
        "runner authority",
    )
    authority = load_json(authority_path, "runner authority")
    decoded = validate_runner_authority(authority, verify_runtime_files=True)
    if decoded["runner_authority_sha256"] != authority_record["logical_sha256"]:
        raise Development300RunnerError("runner plan authority binding changed")
    run_claim = runner_root / "run_claim.json"
    claim = signed(
        {
            "status": "claimed_once_no_resume",
            "runner_plan_sha256": plan_sha,
            "pid": os.getpid(),
        },
        "run_claim_sha256",
    )
    atomic_json(run_claim, claim, mode=0o400)
    gpu_lock = safe_path(
        authority["gpu_contract"]["exclusive_lock_path"], "GPU lock"
    )
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock.open("a+", encoding="ascii") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Development300RunnerError("RTX4090 lock is already held") from error
        idle_before = wait_two_idle(interval_seconds=idle_interval_seconds)
        replace_json(
            runner_root / "state.json",
            {"status": "gpu_idle_verified", "runner_plan_sha256": plan_sha},
        )
        terminal = execute_exact_sequence(
            root=root,
            plan=plan,
            authority_path=authority_path,
            authority_file_sha256=authority_record["file_sha256"],
        )
    # The terminal receipt deliberately omits detailed GPU process telemetry.
    if len(idle_before) != 2:
        raise Development300RunnerError("GPU idle precondition changed")
    return terminal


def record_preloop_failure(root: Path, *, error_type: str) -> None:
    """Publish a sanitized immutable receipt for failures before stage zero."""

    try:
        claimed = safe_path(root, "claimed collection output root").resolve(strict=True)
        runner_root = claimed / "_runner"
        final_path = runner_root / "final_receipt.json"
        if final_path.exists() or not runner_root.is_dir():
            return
        plan_sha: str | None = None
        plan_path = runner_root / "static_plan.json"
        if plan_path.is_file() and not plan_path.is_symlink():
            plan = load_json(plan_path, "runner static plan")
            plan_sha = verify_signed(plan, PLAN_SIGNATURE, "runner static plan")
        failure = signed(
            {
                "format": TERMINAL_RECEIPT_FORMAT,
                "status": TERMINAL_FAILURE,
                "phase": "before_first_group",
                "error_type": error_type,
                "error_message_disclosed": False,
                "runner_plan_sha256": plan_sha,
                "completed_groups": 0,
                "retry_or_resume_authorized": False,
                "formal_label_opened_by_runner_or_watcher": False,
                "evaluation400_commands_executed": 0,
            },
            "terminal_receipt_sha256",
        )
        atomic_json(final_path, failure)
        replace_json(
            runner_root / "state.json",
            {
                "status": TERMINAL_FAILURE,
                "receipt_sha256": failure["terminal_receipt_sha256"],
            },
        )
        _freeze_tree(claimed)
    except Exception:
        # Never replace the original failure with a fabricated success receipt.
        return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authority = commands.add_parser("materialize-authority")
    authority.add_argument("--identity-authority", type=Path, required=True)
    authority.add_argument("--identity-authority-file-sha256", required=True)
    authority.add_argument("--identity-authority-sha256", required=True)
    authority.add_argument("--collection-preregistration", type=Path, required=True)
    authority.add_argument("--collection-preregistration-file-sha256", required=True)
    authority.add_argument("--collection-preregistration-sha256", required=True)
    authority.add_argument("--runtime-contract", type=Path, required=True)
    authority.add_argument("--runtime-contract-file-sha256", required=True)
    authority.add_argument("--runtime-contract-sha256", required=True)
    authority.add_argument("--collector", type=Path, required=True)
    authority.add_argument("--collector-file-sha256", required=True)
    authority.add_argument("--runtime-adapter", type=Path, required=True)
    authority.add_argument("--runtime-adapter-file-sha256", required=True)
    authority.add_argument("--sealed-worker", type=Path, required=True)
    authority.add_argument("--sealed-worker-file-sha256", required=True)
    authority.add_argument("--event-spec", type=Path, required=True)
    authority.add_argument("--event-spec-file-sha256", required=True)
    authority.add_argument("--gpu-lock", type=Path, required=True)
    authority.add_argument("--output", type=Path, required=True)
    dry = commands.add_parser("dry-run-plan")
    dry.add_argument("--authority", type=Path, required=True)
    dry.add_argument("--authority-file-sha256", required=True)
    detach = commands.add_parser("detach")
    detach.add_argument("--authority", type=Path, required=True)
    detach.add_argument("--authority-file-sha256", required=True)
    detach.add_argument("--idle-interval-seconds", type=float, default=30.0)
    serve = commands.add_parser("serve-existing")
    serve.add_argument("--output-root", type=Path, required=True)
    serve.add_argument("--idle-interval-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize-authority":
        authority = build_runner_authority(
            identity_authority_path=args.identity_authority,
            expected_identity_authority_file_sha256=args.identity_authority_file_sha256,
            expected_identity_authority_sha256=args.identity_authority_sha256,
            collection_preregistration_path=args.collection_preregistration,
            expected_collection_preregistration_file_sha256=args.collection_preregistration_file_sha256,
            expected_collection_preregistration_sha256=args.collection_preregistration_sha256,
            runtime_contract_path=args.runtime_contract,
            expected_runtime_contract_file_sha256=args.runtime_contract_file_sha256,
            expected_runtime_contract_sha256=args.runtime_contract_sha256,
            collector_path=args.collector,
            expected_collector_file_sha256=args.collector_file_sha256,
            runtime_adapter_path=args.runtime_adapter,
            expected_runtime_adapter_file_sha256=args.runtime_adapter_file_sha256,
            sealed_worker_path=args.sealed_worker,
            expected_sealed_worker_file_sha256=args.sealed_worker_file_sha256,
            event_spec_path=args.event_spec,
            expected_event_spec_file_sha256=args.event_spec_file_sha256,
            gpu_lock_path=args.gpu_lock,
        )
        atomic_json(args.output, authority)
        print(json.dumps({AUTHORITY_SIGNATURE: authority[AUTHORITY_SIGNATURE]}, sort_keys=True))
        return 0
    if args.command == "dry-run-plan":
        plan = build_static_plan(
            authority_path=args.authority,
            expected_authority_file_sha256=args.authority_file_sha256,
        )
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.command == "detach":
        receipt = claim_output_and_detach(
            authority_path=args.authority,
            expected_authority_file_sha256=args.authority_file_sha256,
            idle_interval_seconds=args.idle_interval_seconds,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    try:
        serve_existing(
            args.output_root, idle_interval_seconds=args.idle_interval_seconds
        )
    except Exception as error:
        # Normal failures are recorded inside execute_exact_sequence.  Failures
        # before the command loop are intentionally not converted into a fake
        # collection receipt; the immutable run claim remains fail-closed.
        record_preloop_failure(args.output_root, error_type=type(error).__name__)
        print(f"{type(error).__name__}: runner failed closed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_FORMAT",
    "AUTHORITY_STATUS",
    "Development300RunnerError",
    "GROUP_SIGNATURE",
    "STATIC_PLAN_FORMAT",
    "WORKER_GROUP_RECEIPT_FORMAT",
    "atomic_json",
    "build_runner_authority",
    "build_static_plan",
    "execute_exact_sequence",
    "file_sha256",
    "load_json",
    "opaque_file_sha256",
    "safe_path",
    "signed",
    "validate_collection_command",
    "validate_collection_preregistration",
    "validate_runner_authority",
    "verify_signed",
]
