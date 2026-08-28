#!/usr/bin/env python3
"""Freeze development300 reset identities and a four-candidate collection plan.

This is a new protocol and does not modify or project the frozen target-v2
protocol.  The only executable operation in the first authority is environment
reset.  Collection commands are materialized only after all 300 preregistered
requested seeds have produced stable, exact, identity-only reset receipts and a
second private heldout-disjoint attestation has bound the selected
requested/resolved identity set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from preregister_smolvla_piper_schema6_target_development300 import (
    ACTOR_ID,
    BODY,
    CANDIDATES_PER_GROUP,
    FORMAL_TARGET_VALIDATION_GROUPS,
    INSTRUCTION,
    POLICY,
    SPLIT_COUNTS,
    TASK,
    TOTAL_GROUPS,
    canonical_sha256,
    validate_preregistration,
)
from resolve_smolvla_piper_target_reset_only import (
    array_sha256,
    scene_sha256,
)
from smolvla_piper_schema6_runtime_adapter_v2 import validate_runtime_contract
from smolvla_piper_target_seed_manifest import validate_disjoint_attestation


RESET_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_reset_authority_v1"
)
RESET_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_identity_resolution_receipt_v1"
)
COLLECTION_IDENTITY_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_identity_authority_v1"
)
COLLECTION_PREREGISTRATION_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_collection_preregistration_v1"
)
RESET_AUTHORITY_STATUS = (
    "authorized_identity_only_resets_no_step_no_policy_no_outcome"
)
RESET_COMPLETE_STATUS = "complete_300_stable_exact_identity_only_resets"
RESET_INSUFFICIENT_STATUS = (
    "incomplete_unstable_setup_no_collection_authority"
)
COLLECTION_IDENTITY_STATUS = (
    "frozen_after_300_identity_only_resets_and_private_disjoint_attestation"
)
COLLECTION_PREREGISTRATION_STATUS = (
    "preregistered_300_four_candidate_groups_execution_not_authorized"
)
CANDIDATE_TARGET_ROLE = "preregistered_reset_candidate_pool"
SELECTED_TARGET_ROLE = "selected_requested_and_resolved_target_identities"
CANDIDATE_INDICES = (0, 1, 2, 3)
FULL_HORIZON_STEPS = 200
HDF_SUFFIXES = {".h5", ".hdf", ".hdf5"}
SHA_CHARS = frozenset("0123456789abcdef")
FORBIDDEN_RESET_FIELDS = frozenset(
    {
        "action",
        "actions",
        "reward",
        "rewards",
        "success",
        "successes",
        "event",
        "events",
        "outcome",
        "outcomes",
        "trajectory",
        "trajectories",
        "prediction",
        "predictions",
        "policy",
        "policy_query",
        "label",
        "labels",
    }
)
FROZEN_IDENTITY_ROW_FIELDS = frozenset(
    {
        "namespace",
        "task",
        "body",
        "policy",
        "actor_id",
        "split",
        "global_ordinal",
        "split_ordinal",
        "requested_seed",
        "resolved_seed",
        "preregistered_logical_group_id",
        "preregistered_identity_sha256",
        "instruction_sha256",
        "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256",
        "resolution_status",
        "pair_id",
    }
)


class Development300IdentityError(RuntimeError):
    """A reset, identity, attestation, or collection boundary failed closed."""


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def file_sha256(path: Path) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise Development300IdentityError("HDF byte access is forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, role: str) -> Path:
    supplied = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if supplied.is_symlink():
        raise Development300IdentityError(f"{role} must not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise Development300IdentityError(f"{role} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or resolved.suffix.casefold() in HDF_SUFFIXES:
        raise Development300IdentityError(f"{role} must be a non-HDF regular file")
    return resolved


def _load_json(path: Path, role: str) -> dict[str, Any]:
    source = _regular_file(path, role)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Development300IdentityError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise Development300IdentityError(f"{role} must contain an object")
    return value


def _verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise Development300IdentityError(f"{role} logical SHA changed")
    return str(recorded)


def _signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise Development300IdentityError("signature field already exists")
    result[field] = canonical_sha256(result)
    return result


def _record(path: Path, logical_sha256: str | None = None) -> dict[str, str]:
    result = {"path": str(path), "file_sha256": file_sha256(path)}
    if logical_sha256 is not None:
        result["logical_sha256"] = logical_sha256
    return result


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_file_binding(path: Path, expected: str, role: str) -> Path:
    source = _regular_file(path, role)
    if not _is_sha(expected) or file_sha256(source) != expected:
        raise Development300IdentityError(f"{role} file SHA changed")
    return source


def _runtime_v2b(
    value: Mapping[str, Any], *, verify_files: bool
) -> dict[str, Any]:
    try:
        decoded = validate_runtime_contract(value, verify_files=verify_files)
    except Exception as error:
        raise Development300IdentityError("runtime contract is invalid") from error
    if (
        decoded.get("max_episode_steps") != FULL_HORIZON_STEPS
        or decoded.get("test_or_evaluation_execution_authorized") is not False
        or decoded.get("fresh_or_confirmation_inputs_accepted") is not False
    ):
        raise Development300IdentityError(
            "runtime contract is not the full-horizon non-evaluation v2b scope"
        )
    return decoded


def _requested(preregistration: Mapping[str, Any]) -> list[int]:
    return [int(row["requested_seed"]) for row in preregistration["groups"]]


def _candidate_identity_sha(preregistration: Mapping[str, Any]) -> str:
    return canonical_sha256(_requested(preregistration))


def _selected_identity_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {
            "requested": [int(row["requested_seed"]) for row in rows],
            "resolved": [int(row["resolved_seed"]) for row in rows],
        }
    )


def _validate_attestation(
    value: Mapping[str, Any],
    *,
    heldout_sha256: str,
    target_sha256: str,
    target_role: str,
) -> str:
    try:
        return validate_disjoint_attestation(
            value,
            heldout_identity_set_sha256=heldout_sha256,
            target_identity_set_sha256=target_sha256,
            target_role=target_role,
        )
    except Exception as error:
        raise Development300IdentityError(
            "private heldout disjoint attestation is invalid"
        ) from error


def build_reset_authority(
    *,
    preregistration_path: Path,
    expected_preregistration_file_sha256: str,
    expected_preregistration_sha256: str,
    runtime_contract_path: Path,
    expected_runtime_contract_file_sha256: str,
    expected_runtime_contract_sha256: str,
    candidate_disjoint_attestation_path: Path,
    expected_candidate_attestation_file_sha256: str,
    reset_adapter_path: Path,
    expected_reset_adapter_file_sha256: str,
    reset_adapter_factory: str = "build_reset_only_adapter",
    verify_runtime_files: bool = True,
) -> dict[str, Any]:
    prereg_path = _assert_file_binding(
        preregistration_path,
        expected_preregistration_file_sha256,
        "development300 preregistration",
    )
    preregistration = _load_json(prereg_path, "development300 preregistration")
    decoded_prereg = validate_preregistration(preregistration)
    if decoded_prereg["preregistration_sha256"] != expected_preregistration_sha256:
        raise Development300IdentityError("development300 logical SHA changed")
    runtime_path = _assert_file_binding(
        runtime_contract_path,
        expected_runtime_contract_file_sha256,
        "runtime contract v2b",
    )
    runtime = _load_json(runtime_path, "runtime contract v2b")
    decoded_runtime = _runtime_v2b(runtime, verify_files=verify_runtime_files)
    if decoded_runtime["runtime_contract_sha256"] != expected_runtime_contract_sha256:
        raise Development300IdentityError("runtime contract logical SHA changed")
    attestation_path = _assert_file_binding(
        candidate_disjoint_attestation_path,
        expected_candidate_attestation_file_sha256,
        "candidate-pool disjoint attestation",
    )
    candidate_attestation = _load_json(
        attestation_path, "candidate-pool disjoint attestation"
    )
    heldout_sha = candidate_attestation.get("heldout_identity_set_sha256")
    if not _is_sha(heldout_sha):
        raise Development300IdentityError("heldout identity commitment is invalid")
    candidate_target_sha = _candidate_identity_sha(preregistration)
    candidate_attestation_sha = _validate_attestation(
        candidate_attestation,
        heldout_sha256=str(heldout_sha),
        target_sha256=candidate_target_sha,
        target_role=CANDIDATE_TARGET_ROLE,
    )
    adapter_path = _assert_file_binding(
        reset_adapter_path,
        expected_reset_adapter_file_sha256,
        "reset adapter implementation",
    )
    if not reset_adapter_factory or reset_adapter_factory.strip() != reset_adapter_factory:
        raise Development300IdentityError("reset adapter factory is invalid")
    base: dict[str, Any] = {
        "format": RESET_AUTHORITY_FORMAT,
        "status": RESET_AUTHORITY_STATUS,
        "development300_preregistration": _record(
            prereg_path, decoded_prereg["preregistration_sha256"]
        ),
        "partition_sha256": decoded_prereg["partition_sha256"],
        "runtime_contract": _record(
            runtime_path, decoded_runtime["runtime_contract_sha256"]
        ),
        "runtime_contract_payload": decoded_runtime,
        "candidate_pool_identity_set_sha256": candidate_target_sha,
        "heldout_identity_set_sha256": str(heldout_sha),
        "candidate_pool_disjoint_attestation": candidate_attestation,
        "candidate_pool_disjoint_attestation_source": _record(
            attestation_path, candidate_attestation_sha
        ),
        "reset_adapter": {
            **_record(adapter_path),
            "factory": reset_adapter_factory,
        },
        "resolver_implementation": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": file_sha256(Path(__file__).resolve()),
        },
        "requested_seed_count": TOTAL_GROUPS,
        "requested_identity_order_sha256": candidate_target_sha,
        "permissions": {
            "environment_construct_allowed": True,
            "environment_reset_allowed": True,
            "maximum_reset_calls": TOTAL_GROUPS,
            "environment_step_allowed": False,
            "policy_import_or_forward_allowed": False,
            "reward_success_event_outcome_trajectory_or_label_read_allowed": False,
            "collection_allowed": False,
            "evaluation400_identity_or_execution_allowed": False,
        },
        "resolution_rules": {
            "requested_order_must_equal_preregistration": True,
            "one_reset_call_per_requested_seed": True,
            "implicit_seed_retry_allowed": False,
            "resolved_seed_must_equal_requested_seed": True,
            "unstable_setup_must_be_recorded_then_advance": True,
            "unstable_setup_may_select_a_replacement": False,
            "all_300_stable_required_before_collection_authority": True,
        },
        "input_or_output_data_files_opened": 0,
        "labels_or_outcomes_read": False,
    }
    return _signed(base, "authority_sha256")


def validate_reset_authority(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    verify_runtime_files: bool,
) -> dict[str, Any]:
    logical = _verify_signed(value, "authority_sha256", "reset authority")
    decoded_prereg = validate_preregistration(preregistration)
    decoded_runtime = _runtime_v2b(
        runtime_contract, verify_files=verify_runtime_files
    )
    candidate_sha = _candidate_identity_sha(preregistration)
    attestation = value.get("candidate_pool_disjoint_attestation")
    permissions = value.get("permissions")
    rules = value.get("resolution_rules")
    prereg_record = value.get("development300_preregistration")
    runtime_record = value.get("runtime_contract")
    adapter = value.get("reset_adapter")
    resolver = value.get("resolver_implementation")
    attestation_source = value.get("candidate_pool_disjoint_attestation_source")
    current_resolver = Path(__file__).resolve()
    if (
        set(value)
        != {
            "format",
            "status",
            "development300_preregistration",
            "partition_sha256",
            "runtime_contract",
            "runtime_contract_payload",
            "candidate_pool_identity_set_sha256",
            "heldout_identity_set_sha256",
            "candidate_pool_disjoint_attestation",
            "candidate_pool_disjoint_attestation_source",
            "reset_adapter",
            "resolver_implementation",
            "requested_seed_count",
            "requested_identity_order_sha256",
            "permissions",
            "resolution_rules",
            "input_or_output_data_files_opened",
            "labels_or_outcomes_read",
            "authority_sha256",
        }
        or value.get("format") != RESET_AUTHORITY_FORMAT
        or value.get("status") != RESET_AUTHORITY_STATUS
        or value.get("partition_sha256") != decoded_prereg["partition_sha256"]
        or value.get("requested_seed_count") != TOTAL_GROUPS
        or value.get("requested_identity_order_sha256") != candidate_sha
        or value.get("candidate_pool_identity_set_sha256") != candidate_sha
        or not _is_sha(value.get("heldout_identity_set_sha256"))
        or not isinstance(attestation, Mapping)
        or not isinstance(prereg_record, Mapping)
        or set(prereg_record) != {"path", "file_sha256", "logical_sha256"}
        or not isinstance(prereg_record.get("path"), str)
        or not _is_sha(prereg_record.get("file_sha256"))
        or prereg_record.get("logical_sha256")
        != decoded_prereg["preregistration_sha256"]
        or not isinstance(runtime_record, Mapping)
        or set(runtime_record) != {"path", "file_sha256", "logical_sha256"}
        or not isinstance(runtime_record.get("path"), str)
        or not _is_sha(runtime_record.get("file_sha256"))
        or runtime_record.get("logical_sha256")
        != decoded_runtime["runtime_contract_sha256"]
        or value.get("runtime_contract_payload") != decoded_runtime
        or not isinstance(attestation_source, Mapping)
        or set(attestation_source)
        != {"path", "file_sha256", "logical_sha256"}
        or not isinstance(attestation_source.get("path"), str)
        or not _is_sha(attestation_source.get("file_sha256"))
        or attestation_source.get("logical_sha256")
        != attestation.get("attestation_sha256")
        or not isinstance(adapter, Mapping)
        or set(adapter) != {"path", "file_sha256", "factory"}
        or not isinstance(adapter.get("path"), str)
        or not _is_sha(adapter.get("file_sha256"))
        or not isinstance(adapter.get("factory"), str)
        or not isinstance(resolver, Mapping)
        or resolver.get("path") != str(current_resolver)
        or resolver.get("file_sha256") != file_sha256(current_resolver)
        or not _is_sha(resolver.get("file_sha256"))
        or permissions
        != {
            "environment_construct_allowed": True,
            "environment_reset_allowed": True,
            "maximum_reset_calls": TOTAL_GROUPS,
            "environment_step_allowed": False,
            "policy_import_or_forward_allowed": False,
            "reward_success_event_outcome_trajectory_or_label_read_allowed": False,
            "collection_allowed": False,
            "evaluation400_identity_or_execution_allowed": False,
        }
        or rules
        != {
            "requested_order_must_equal_preregistration": True,
            "one_reset_call_per_requested_seed": True,
            "implicit_seed_retry_allowed": False,
            "resolved_seed_must_equal_requested_seed": True,
            "unstable_setup_must_be_recorded_then_advance": True,
            "unstable_setup_may_select_a_replacement": False,
            "all_300_stable_required_before_collection_authority": True,
        }
        or value.get("input_or_output_data_files_opened") != 0
        or value.get("labels_or_outcomes_read") is not False
    ):
        raise Development300IdentityError("reset authority scope changed")
    _validate_attestation(
        attestation,
        heldout_sha256=str(value["heldout_identity_set_sha256"]),
        target_sha256=candidate_sha,
        target_role=CANDIDATE_TARGET_ROLE,
    )
    return {
        "authority_sha256": logical,
        "preregistration_sha256": decoded_prereg["preregistration_sha256"],
        "partition_sha256": decoded_prereg["partition_sha256"],
        "runtime_contract_sha256": decoded_runtime["runtime_contract_sha256"],
        "heldout_identity_set_sha256": value["heldout_identity_set_sha256"],
    }


def _forbidden_reset_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).casefold()
            location = f"{prefix}.{key}" if prefix else str(key)
            if name in FORBIDDEN_RESET_FIELDS or set(name.split("_")) & FORBIDDEN_RESET_FIELDS:
                found.append(location)
            found.extend(_forbidden_reset_paths(child, location))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden_reset_paths(child, f"{prefix}[{index}]"))
    return found


def _resolved_row(
    preregistered: Mapping[str, Any], raw: Mapping[str, Any]
) -> dict[str, Any]:
    requested = int(preregistered["requested_seed"])
    if set(raw) != {
        "setup_status",
        "requested_seed",
        "resolved_seed",
        "instruction_observed",
        "scene_state",
        "measured_joint_state",
        "commanded_drive_target",
    }:
        raise Development300IdentityError("stable reset identity fields changed")
    if raw.get("requested_seed") != requested:
        raise Development300IdentityError("reset changed requested seed")
    resolved = raw.get("resolved_seed")
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, int)
        or resolved != requested
    ):
        raise Development300IdentityError(
            "reset performed an unapproved implicit seed retry"
        )
    if raw.get("instruction_observed") != INSTRUCTION:
        raise Development300IdentityError("reset changed frozen instruction")
    row: dict[str, Any] = {
        "namespace": preregistered["namespace"],
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "actor_id": ACTOR_ID,
        "split": preregistered["split"],
        "global_ordinal": preregistered["global_ordinal"],
        "split_ordinal": preregistered["split_ordinal"],
        "requested_seed": requested,
        "resolved_seed": resolved,
        "preregistered_logical_group_id": preregistered["logical_group_id"],
        "preregistered_identity_sha256": preregistered["identity_sha256"],
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest(),
        "initial_scene_state_sha256": scene_sha256(raw["scene_state"]),
        "initial_measured_joint_state_sha256": array_sha256(
            raw["measured_joint_state"], role="measured joint state"
        ),
        "initial_commanded_drive_target_sha256": array_sha256(
            raw["commanded_drive_target"], role="commanded drive target"
        ),
        "resolution_status": "stable_exact_identity_only_zero_step",
    }
    row["pair_id"] = canonical_sha256(row)
    return row


def resolve_identities(
    *,
    preregistration: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    reset_authority: Mapping[str, Any],
    reset_once: Callable[[int, str], Mapping[str, Any]],
    reset_authority_file_sha256: str,
    verify_runtime_files: bool = True,
) -> dict[str, Any]:
    audit = validate_reset_authority(
        reset_authority,
        preregistration=preregistration,
        runtime_contract=runtime_contract,
        verify_runtime_files=verify_runtime_files,
    )
    if not _is_sha(reset_authority_file_sha256):
        raise Development300IdentityError("reset authority file SHA is invalid")
    attempts: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    selected_requested: set[int] = set()
    selected_resolved: set[int] = set()
    for preregistered in preregistration["groups"]:
        requested = int(preregistered["requested_seed"])
        try:
            raw = reset_once(requested, INSTRUCTION)
        except Exception:
            raise Development300IdentityError(
                "reset adapter failed without an admissible identity receipt"
            ) from None
        if not isinstance(raw, Mapping):
            raise Development300IdentityError("reset adapter must return a mapping")
        if _forbidden_reset_paths(raw):
            raise Development300IdentityError(
                "reset adapter exposed a forbidden result field"
            )
        setup_status = raw.get("setup_status")
        if setup_status == "unstable":
            if set(raw) != {"setup_status", "requested_seed"} or raw.get(
                "requested_seed"
            ) != requested:
                raise Development300IdentityError(
                    "unstable reset record fields changed"
                )
            attempts.append(
                {
                    "global_ordinal": preregistered["global_ordinal"],
                    "requested_seed": requested,
                    "setup_status": "unstable_recorded_advanced_without_replacement",
                }
            )
            continue
        if setup_status != "stable":
            raise Development300IdentityError(
                "reset setup status must be stable or unstable"
            )
        row = _resolved_row(preregistered, raw)
        resolved = int(row["resolved_seed"])
        if requested in selected_requested or resolved in selected_resolved:
            raise Development300IdentityError(
                "selected requested/resolved identities are duplicated"
            )
        selected_requested.add(requested)
        selected_resolved.add(resolved)
        selected.append(row)
        attempts.append(
            {
                "global_ordinal": preregistered["global_ordinal"],
                "requested_seed": requested,
                "resolved_seed": resolved,
                "setup_status": "stable_selected_identity_only",
                "pair_id": row["pair_id"],
            }
        )
    complete = len(selected) == TOTAL_GROUPS
    split_counts = {
        split: sum(row["split"] == split for row in selected)
        for split in SPLIT_COUNTS
    }
    base: dict[str, Any] = {
        "format": RESET_RECEIPT_FORMAT,
        "status": RESET_COMPLETE_STATUS if complete else RESET_INSUFFICIENT_STATUS,
        "reset_authority_file_sha256": reset_authority_file_sha256,
        "reset_authority_sha256": audit["authority_sha256"],
        "development300_preregistration_sha256": audit[
            "preregistration_sha256"
        ],
        "partition_sha256": audit["partition_sha256"],
        "runtime_contract_sha256": audit["runtime_contract_sha256"],
        "heldout_identity_set_sha256": audit["heldout_identity_set_sha256"],
        "attempts": attempts,
        "selected_rows": selected,
        "attempt_count": len(attempts),
        "stable_selected_count": len(selected),
        "unstable_setup_count": len(attempts) - len(selected),
        "selected_split_counts": split_counts,
        "selected_requested_unique": len(selected_requested) == len(selected),
        "selected_resolved_unique": len(selected_resolved) == len(selected),
        "selected_identity_set_sha256": _selected_identity_sha(selected),
        "all_preregistered_requested_processed_in_order": True,
        "preregistered_membership_or_order_depends_on_setup_or_outcome": False,
        "collection_identity_membership_frozen": complete,
        "unstable_setup_caused_replacement_or_split_movement": False,
        "environment_reset_calls": len(attempts),
        "environment_step_calls": 0,
        "policy_import_or_forward_calls": 0,
        "reward_success_event_outcome_trajectory_or_label_fields_read": 0,
        "hdf5_files_opened": 0,
        "collection_authorized": False,
        "evaluation400_identity_or_membership_read": False,
        "evaluation400_execution_authorized": False,
    }
    return _signed(base, "receipt_sha256")


def validate_identity_receipt(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    reset_authority: Mapping[str, Any],
) -> dict[str, Any]:
    logical = _verify_signed(value, "receipt_sha256", "identity receipt")
    decoded = validate_preregistration(preregistration)
    rows = value.get("selected_rows")
    attempts = value.get("attempts")
    if (
        set(value)
        != {
            "format",
            "status",
            "reset_authority_file_sha256",
            "reset_authority_sha256",
            "development300_preregistration_sha256",
            "partition_sha256",
            "runtime_contract_sha256",
            "heldout_identity_set_sha256",
            "attempts",
            "selected_rows",
            "attempt_count",
            "stable_selected_count",
            "unstable_setup_count",
            "selected_split_counts",
            "selected_requested_unique",
            "selected_resolved_unique",
            "selected_identity_set_sha256",
            "all_preregistered_requested_processed_in_order",
            "preregistered_membership_or_order_depends_on_setup_or_outcome",
            "collection_identity_membership_frozen",
            "unstable_setup_caused_replacement_or_split_movement",
            "environment_reset_calls",
            "environment_step_calls",
            "policy_import_or_forward_calls",
            "reward_success_event_outcome_trajectory_or_label_fields_read",
            "hdf5_files_opened",
            "collection_authorized",
            "evaluation400_identity_or_membership_read",
            "evaluation400_execution_authorized",
            "receipt_sha256",
        }
        or value.get("format") != RESET_RECEIPT_FORMAT
        or value.get("development300_preregistration_sha256")
        != decoded["preregistration_sha256"]
        or value.get("partition_sha256") != decoded["partition_sha256"]
        or value.get("reset_authority_sha256")
        != reset_authority.get("authority_sha256")
        or not _is_sha(value.get("reset_authority_file_sha256"))
        or value.get("runtime_contract_sha256")
        != reset_authority.get("runtime_contract", {}).get("logical_sha256")
        or value.get("heldout_identity_set_sha256")
        != reset_authority.get("heldout_identity_set_sha256")
        or not isinstance(rows, list)
        or not isinstance(attempts, list)
        or value.get("attempt_count") != TOTAL_GROUPS
        or len(attempts) != TOTAL_GROUPS
        or value.get("environment_reset_calls") != TOTAL_GROUPS
        or value.get("environment_step_calls") != 0
        or value.get("policy_import_or_forward_calls") != 0
        or value.get(
            "reward_success_event_outcome_trajectory_or_label_fields_read"
        )
        != 0
        or value.get("hdf5_files_opened") != 0
        or value.get("collection_authorized") is not False
        or value.get("evaluation400_identity_or_membership_read") is not False
        or value.get("evaluation400_execution_authorized") is not False
    ):
        raise Development300IdentityError("identity receipt scope changed")
    expected_requested = _requested(preregistration)
    try:
        attempt_requested = [int(row["requested_seed"]) for row in attempts]
    except (KeyError, TypeError, ValueError) as error:
        raise Development300IdentityError(
            "reset attempt inventory is invalid"
        ) from error
    if attempt_requested != expected_requested:
        raise Development300IdentityError("reset attempts changed requested order")
    prereg_by_ordinal = {
        int(row["global_ordinal"]): row for row in preregistration["groups"]
    }
    rebuilt_rows: list[dict[str, Any]] = []
    selected_requested: set[int] = set()
    selected_resolved: set[int] = set()
    stable_attempt_rows: list[Mapping[str, Any]] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise Development300IdentityError("reset attempt inventory is invalid")
        requested = expected_requested[index]
        stable = attempt.get("setup_status") == "stable_selected_identity_only"
        if stable:
            if set(attempt) != {
                "global_ordinal",
                "requested_seed",
                "resolved_seed",
                "setup_status",
                "pair_id",
            }:
                raise Development300IdentityError("stable reset attempt changed")
            stable_attempt_rows.append(attempt)
        elif (
            attempt
            != {
                "global_ordinal": index,
                "requested_seed": requested,
                "setup_status": "unstable_recorded_advanced_without_replacement",
            }
        ):
            raise Development300IdentityError("unstable reset attempt changed")
    if len(stable_attempt_rows) != len(rows):
        raise Development300IdentityError("stable reset attempt count changed")
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Development300IdentityError("selected identity row is invalid")
        attempt = stable_attempt_rows[row_index]
        try:
            ordinal = int(row["global_ordinal"])
            preregistered = prereg_by_ordinal[ordinal]
            requested = int(row["requested_seed"])
            resolved = int(row["resolved_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise Development300IdentityError(
                "selected identity row is invalid"
            ) from error
        unsigned_row = dict(row)
        pair_id = unsigned_row.pop("pair_id", None)
        if (
            set(row) != FROZEN_IDENTITY_ROW_FIELDS
            or requested != int(preregistered["requested_seed"])
            or resolved != requested
            or row.get("namespace") != preregistered["namespace"]
            or row.get("task") != TASK
            or row.get("body") != BODY
            or row.get("policy") != POLICY
            or row.get("actor_id") != ACTOR_ID
            or row.get("split") != preregistered["split"]
            or row.get("split_ordinal") != preregistered["split_ordinal"]
            or row.get("preregistered_logical_group_id")
            != preregistered["logical_group_id"]
            or row.get("preregistered_identity_sha256")
            != preregistered["identity_sha256"]
            or row.get("instruction_sha256")
            != hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
            or row.get("resolution_status")
            != "stable_exact_identity_only_zero_step"
            or not _is_sha(row.get("initial_scene_state_sha256"))
            or not _is_sha(row.get("initial_measured_joint_state_sha256"))
            or not _is_sha(row.get("initial_commanded_drive_target_sha256"))
            or not _is_sha(pair_id)
            or pair_id != canonical_sha256(unsigned_row)
            or attempt.get("global_ordinal") != ordinal
            or attempt.get("requested_seed") != requested
            or attempt.get("resolved_seed") != resolved
            or attempt.get("pair_id") != pair_id
            or requested in selected_requested
            or resolved in selected_resolved
        ):
            raise Development300IdentityError("selected identity row changed")
        selected_requested.add(requested)
        selected_resolved.add(resolved)
        rebuilt_rows.append(dict(row))
    complete = len(rebuilt_rows) == TOTAL_GROUPS
    split_counts = {
        split: sum(row["split"] == split for row in rebuilt_rows)
        for split in SPLIT_COUNTS
    }
    selected_sha = _selected_identity_sha(rebuilt_rows)
    if (
        value.get("status")
        != (RESET_COMPLETE_STATUS if complete else RESET_INSUFFICIENT_STATUS)
        or value.get("stable_selected_count") != len(rebuilt_rows)
        or value.get("unstable_setup_count") != TOTAL_GROUPS - len(rebuilt_rows)
        or value.get("selected_split_counts") != split_counts
        or value.get("selected_requested_unique") is not True
        or value.get("selected_resolved_unique") is not True
        or value.get("selected_identity_set_sha256") != selected_sha
        or value.get("all_preregistered_requested_processed_in_order") is not True
        or value.get(
            "preregistered_membership_or_order_depends_on_setup_or_outcome"
        )
        is not False
        or value.get("collection_identity_membership_frozen") is not complete
        or value.get("unstable_setup_caused_replacement_or_split_movement")
        is not False
    ):
        raise Development300IdentityError("identity receipt counts changed")
    return {
        "receipt_sha256": logical,
        "complete": complete,
        "rows": rebuilt_rows,
        "selected_identity_set_sha256": selected_sha,
    }


def _future_root(path: Path) -> Path:
    value = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if value.exists() or value.is_symlink():
        raise Development300IdentityError(
            "future collection root must not already exist"
        )
    value.parent.resolve(strict=True)
    return value


def build_collection_identity_authority(
    *,
    preregistration: Mapping[str, Any],
    reset_authority: Mapping[str, Any],
    identity_receipt: Mapping[str, Any],
    selected_disjoint_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    decoded = validate_preregistration(preregistration)
    runtime_payload = reset_authority.get("runtime_contract_payload")
    if not isinstance(runtime_payload, Mapping):
        raise Development300IdentityError("reset authority runtime binding is missing")
    reset_audit = validate_reset_authority(
        reset_authority,
        preregistration=preregistration,
        runtime_contract=runtime_payload,
        verify_runtime_files=False,
    )
    receipt = validate_identity_receipt(
        identity_receipt,
        preregistration=preregistration,
        reset_authority=reset_authority,
    )
    rows = receipt["rows"]
    if not receipt["complete"] or len(rows) != TOTAL_GROUPS:
        raise Development300IdentityError(
            "all 300 stable identities are required before collection freeze"
        )
    requested = [int(row["requested_seed"]) for row in rows]
    resolved = [int(row["resolved_seed"]) for row in rows]
    if (
        requested != _requested(preregistration)
        or len(set(requested)) != TOTAL_GROUPS
        or len(set(resolved)) != TOTAL_GROUPS
        or any(requested[index] != resolved[index] for index in range(TOTAL_GROUPS))
        or {
            split: sum(row["split"] == split for row in rows)
            for split in SPLIT_COUNTS
        }
        != SPLIT_COUNTS
        or receipt["selected_identity_set_sha256"]
        != _selected_identity_sha(rows)
    ):
        raise Development300IdentityError(
            "selected identities do not preserve the frozen 80/30/190 partition"
        )
    heldout_sha = str(reset_authority["heldout_identity_set_sha256"])
    attestation_sha = _validate_attestation(
        selected_disjoint_attestation,
        heldout_sha256=heldout_sha,
        target_sha256=receipt["selected_identity_set_sha256"],
        target_role=SELECTED_TARGET_ROLE,
    )
    base: dict[str, Any] = {
        "format": COLLECTION_IDENTITY_AUTHORITY_FORMAT,
        "status": COLLECTION_IDENTITY_STATUS,
        "development300_preregistration_sha256": decoded[
            "preregistration_sha256"
        ],
        "partition_sha256": decoded["partition_sha256"],
        "reset_authority_sha256": reset_audit["authority_sha256"],
        "identity_resolution_receipt_sha256": identity_receipt["receipt_sha256"],
        "runtime_contract_sha256": reset_audit["runtime_contract_sha256"],
        "heldout_identity_set_sha256": heldout_sha,
        "selected_identity_set_sha256": receipt[
            "selected_identity_set_sha256"
        ],
        "selected_disjoint_attestation": dict(selected_disjoint_attestation),
        "selected_disjoint_attestation_sha256": attestation_sha,
        "selected_rows": rows,
        "selected_group_count": TOTAL_GROUPS,
        "split_counts": dict(SPLIT_COUNTS),
        "adaptation_bucket_groups": 110,
        "formal_target_validation_groups": FORMAL_TARGET_VALIDATION_GROUPS,
        "requested_globally_unique": True,
        "resolved_globally_unique": True,
        "identity_freeze_evidence": {
            "environment_reset_calls": TOTAL_GROUPS,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "reward_success_event_outcome_trajectory_or_label_fields_read": 0,
            "hdf5_files_opened": 0,
        },
        "permissions": {
            "collection_execution_authorized": False,
            "formal_target_validation_label_open_authorized": False,
            "evaluation400_identity_read_or_command_generation_authorized": False,
            "evaluation400_execution_authorized": False,
        },
    }
    return _signed(base, "identity_authority_sha256")


def validate_collection_identity_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    logical = _verify_signed(
        value, "identity_authority_sha256", "collection identity authority"
    )
    rows = value.get("selected_rows")
    permissions = value.get("permissions")
    evidence = value.get("identity_freeze_evidence")
    if (
        set(value)
        != {
            "format",
            "status",
            "development300_preregistration_sha256",
            "partition_sha256",
            "reset_authority_sha256",
            "identity_resolution_receipt_sha256",
            "runtime_contract_sha256",
            "heldout_identity_set_sha256",
            "selected_identity_set_sha256",
            "selected_disjoint_attestation",
            "selected_disjoint_attestation_sha256",
            "selected_rows",
            "selected_group_count",
            "split_counts",
            "adaptation_bucket_groups",
            "formal_target_validation_groups",
            "requested_globally_unique",
            "resolved_globally_unique",
            "identity_freeze_evidence",
            "permissions",
            "identity_authority_sha256",
        }
        or value.get("format") != COLLECTION_IDENTITY_AUTHORITY_FORMAT
        or value.get("status") != COLLECTION_IDENTITY_STATUS
        or not _is_sha(value.get("development300_preregistration_sha256"))
        or not _is_sha(value.get("partition_sha256"))
        or not _is_sha(value.get("reset_authority_sha256"))
        or not _is_sha(value.get("identity_resolution_receipt_sha256"))
        or not _is_sha(value.get("runtime_contract_sha256"))
        or not _is_sha(value.get("heldout_identity_set_sha256"))
        or not _is_sha(value.get("selected_identity_set_sha256"))
        or not isinstance(rows, list)
        or len(rows) != TOTAL_GROUPS
        or value.get("selected_group_count") != TOTAL_GROUPS
        or value.get("split_counts") != SPLIT_COUNTS
        or value.get("adaptation_bucket_groups") != 110
        or value.get("formal_target_validation_groups")
        != FORMAL_TARGET_VALIDATION_GROUPS
        or value.get("requested_globally_unique") is not True
        or value.get("resolved_globally_unique") is not True
        or evidence
        != {
            "environment_reset_calls": TOTAL_GROUPS,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "reward_success_event_outcome_trajectory_or_label_fields_read": 0,
            "hdf5_files_opened": 0,
        }
        or permissions
        != {
            "collection_execution_authorized": False,
            "formal_target_validation_label_open_authorized": False,
            "evaluation400_identity_read_or_command_generation_authorized": False,
            "evaluation400_execution_authorized": False,
        }
    ):
        raise Development300IdentityError("collection identity authority changed")
    selected_sha = _selected_identity_sha(rows)
    if any(
        not isinstance(row, Mapping) or set(row) != FROZEN_IDENTITY_ROW_FIELDS
        for row in rows
    ):
        raise Development300IdentityError("collection identity row schema changed")
    requested = [int(row["requested_seed"]) for row in rows]
    resolved = [int(row["resolved_seed"]) for row in rows]
    if (
        selected_sha != value["selected_identity_set_sha256"]
        or len(set(requested)) != TOTAL_GROUPS
        or len(set(resolved)) != TOTAL_GROUPS
        or requested != resolved
        or {
            split: sum(row["split"] == split for row in rows)
            for split in SPLIT_COUNTS
        }
        != SPLIT_COUNTS
    ):
        raise Development300IdentityError("collection identity membership changed")
    attestation = value.get("selected_disjoint_attestation")
    if not isinstance(attestation, Mapping):
        raise Development300IdentityError("selected disjoint attestation is missing")
    attestation_sha = _validate_attestation(
        attestation,
        heldout_sha256=str(value["heldout_identity_set_sha256"]),
        target_sha256=selected_sha,
        target_role=SELECTED_TARGET_ROLE,
    )
    if value.get("selected_disjoint_attestation_sha256") != attestation_sha:
        raise Development300IdentityError("selected attestation binding changed")
    return {
        "identity_authority_sha256": logical,
        "rows": [dict(row) for row in rows],
        "runtime_contract_sha256": value["runtime_contract_sha256"],
    }


def _collection_command(
    row: Mapping[str, Any],
    *,
    future_root: Path,
    identity_authority_sha256: str,
    runtime_contract_sha256: str,
) -> dict[str, Any]:
    split = str(row["split"])
    split_ordinal = int(row["split_ordinal"])
    seed_root = (
        future_root
        / split
        / f"group_{split_ordinal:03d}_seed_{int(row['requested_seed'])}"
    )
    base: dict[str, Any] = {
        "operation": "collect_schema6_four_candidate_group_v1",
        "global_ordinal": row["global_ordinal"],
        "split": split,
        "split_ordinal": split_ordinal,
        "requested_seed": row["requested_seed"],
        "expected_resolved_seed": row["resolved_seed"],
        "pair_id": row["pair_id"],
        "candidate_original_indices": list(CANDIDATE_INDICES),
        "candidate_branch_count": CANDIDATES_PER_GROUP,
        "outputs": {
            "seed_root": str(seed_root),
            "per_seed_reset_receipt": str(seed_root / "per_seed_reset_receipt.json"),
            "group_hdf5": str(seed_root / "schema6_group.hdf5"),
            "completed_group_receipt": str(seed_root / "completed_group_receipt.json"),
        },
        "bindings": {
            "collection_identity_authority_sha256": identity_authority_sha256,
            "runtime_contract_sha256": runtime_contract_sha256,
        },
        "capability": {
            "execution_authorized_by_preregistration": False,
            "all_four_candidate_branches_required": True,
            "outcome_based_retry_or_replacement_allowed": False,
            "evaluation400": False,
        },
    }
    return _signed(base, "command_sha256")


def build_collection_preregistration(
    *,
    identity_authority: Mapping[str, Any],
    identity_authority_record: Mapping[str, Any],
    future_collection_root: Path,
) -> dict[str, Any]:
    decoded = validate_collection_identity_authority(identity_authority)
    identity_logical = decoded["identity_authority_sha256"]
    if (
        set(identity_authority_record)
        != {"path", "file_sha256", "logical_sha256"}
        or not isinstance(identity_authority_record.get("path"), str)
        or not _is_sha(identity_authority_record.get("file_sha256"))
        or identity_authority_record.get("logical_sha256") != identity_logical
    ):
        raise Development300IdentityError("collection identity authority record changed")
    authority_source = _assert_file_binding(
        Path(str(identity_authority_record["path"])),
        str(identity_authority_record["file_sha256"]),
        "collection identity authority",
    )
    if _load_json(authority_source, "collection identity authority") != dict(
        identity_authority
    ):
        raise Development300IdentityError(
            "collection identity authority source changed"
        )
    root = _future_root(future_collection_root)
    commands = [
        _collection_command(
            row,
            future_root=root,
            identity_authority_sha256=identity_logical,
            runtime_contract_sha256=identity_authority[
                "runtime_contract_sha256"
            ],
        )
        for row in decoded["rows"]
    ]
    if (
        len(commands) != TOTAL_GROUPS
        or sum(row["candidate_branch_count"] for row in commands)
        != TOTAL_GROUPS * CANDIDATES_PER_GROUP
        or [row["requested_seed"] for row in commands]
        != [
            row["requested_seed"]
            for row in decoded["rows"]
        ]
    ):
        raise Development300IdentityError("collection command inventory changed")
    base: dict[str, Any] = {
        "format": COLLECTION_PREREGISTRATION_FORMAT,
        "status": COLLECTION_PREREGISTRATION_STATUS,
        "collection_identity_authority": dict(identity_authority_record),
        "collection_identity_authority_sha256": identity_logical,
        "development300_preregistration_sha256": identity_authority[
            "development300_preregistration_sha256"
        ],
        "partition_sha256": identity_authority["partition_sha256"],
        "runtime_contract_sha256": identity_authority[
            "runtime_contract_sha256"
        ],
        "future_collection_root": str(root),
        "command_count": TOTAL_GROUPS,
        "candidate_branches_per_command": CANDIDATES_PER_GROUP,
        "planned_candidate_branches": TOTAL_GROUPS * CANDIDATES_PER_GROUP,
        "ordered_splits": list(SPLIT_COUNTS),
        "split_counts": dict(SPLIT_COUNTS),
        "commands": commands,
        "execution_boundary": {
            "collection_execution_authorized": False,
            "separate_bound_runner_authority_required": True,
            "outcome_dependent_stop_retry_replacement_or_split_movement_allowed": False,
            "formal_target_validation_label_open_authorized": False,
            "evaluation400_commands_generated": 0,
            "evaluation400_identity_or_membership_read": False,
            "evaluation400_execution_authorized": False,
        },
        "materialization_audit": {
            "environment_reset_calls": 0,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "reward_success_event_outcome_trajectory_or_label_fields_read": 0,
            "hdf5_files_opened": 0,
        },
    }
    return _signed(base, "collection_preregistration_sha256")


def materialize_collection(
    *,
    preregistration_path: Path,
    expected_preregistration_file_sha256: str,
    reset_authority_path: Path,
    expected_reset_authority_file_sha256: str,
    identity_receipt_path: Path,
    expected_identity_receipt_file_sha256: str,
    selected_attestation_path: Path,
    expected_selected_attestation_file_sha256: str,
    future_collection_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    prereg_path = _assert_file_binding(
        preregistration_path,
        expected_preregistration_file_sha256,
        "development300 preregistration",
    )
    reset_path = _assert_file_binding(
        reset_authority_path,
        expected_reset_authority_file_sha256,
        "reset authority",
    )
    receipt_path = _assert_file_binding(
        identity_receipt_path,
        expected_identity_receipt_file_sha256,
        "identity receipt",
    )
    selected_path = _assert_file_binding(
        selected_attestation_path,
        expected_selected_attestation_file_sha256,
        "selected disjoint attestation",
    )
    preregistration = _load_json(prereg_path, "development300 preregistration")
    reset_authority = _load_json(reset_path, "reset authority")
    identity_receipt = _load_json(receipt_path, "identity receipt")
    selected_attestation = _load_json(
        selected_path, "selected disjoint attestation"
    )
    if identity_receipt.get("reset_authority_file_sha256") != file_sha256(
        reset_path
    ):
        raise Development300IdentityError(
            "identity receipt reset authority file binding changed"
        )
    _future_root(future_collection_root)
    runtime_record = reset_authority.get("runtime_contract")
    if not isinstance(runtime_record, Mapping):
        raise Development300IdentityError("reset authority runtime source is missing")
    runtime_path = _assert_file_binding(
        Path(str(runtime_record.get("path", ""))),
        str(runtime_record.get("file_sha256", "")),
        "runtime contract v2b",
    )
    runtime = _load_json(runtime_path, "runtime contract v2b")
    validate_reset_authority(
        reset_authority,
        preregistration=preregistration,
        runtime_contract=runtime,
        verify_runtime_files=True,
    )
    authority = build_collection_identity_authority(
        preregistration=preregistration,
        reset_authority=reset_authority,
        identity_receipt=identity_receipt,
        selected_disjoint_attestation=selected_attestation,
    )
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.resolve(strict=True)
    output.mkdir(mode=0o755)
    authority_path = output / "collection_identity_authority.json"
    _atomic_json_new(authority_path, authority)
    authority_record = _record(
        authority_path, authority["identity_authority_sha256"]
    )
    collection = build_collection_preregistration(
        identity_authority=authority,
        identity_authority_record=authority_record,
        future_collection_root=future_collection_root,
    )
    prereg_output = output / "collection_preregistration.json"
    _atomic_json_new(prereg_output, collection)
    output.chmod(0o555)
    return {
        "collection_identity_authority_path": str(authority_path),
        "collection_identity_authority_file_sha256": file_sha256(authority_path),
        "collection_identity_authority_sha256": authority[
            "identity_authority_sha256"
        ],
        "collection_preregistration_path": str(prereg_output),
        "collection_preregistration_file_sha256": file_sha256(prereg_output),
        "collection_preregistration_sha256": collection[
            "collection_preregistration_sha256"
        ],
        "command_count": TOTAL_GROUPS,
        "planned_candidate_branches": TOTAL_GROUPS * CANDIDATES_PER_GROUP,
        "evaluation400_commands_generated": 0,
    }


def _load_reset_factory(
    path: Path, symbol: str
) -> Callable[..., Any]:
    specification = importlib.util.spec_from_file_location(
        "_etsf_development300_reset_adapter", path
    )
    if specification is None or specification.loader is None:
        raise Development300IdentityError("cannot load reset adapter")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    factory = getattr(module, symbol, None)
    if not callable(factory):
        raise Development300IdentityError("reset adapter factory is unavailable")
    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authority = commands.add_parser("reset-authority")
    authority.add_argument("--preregistration", type=Path, required=True)
    authority.add_argument("--preregistration-file-sha256", required=True)
    authority.add_argument("--preregistration-sha256", required=True)
    authority.add_argument("--runtime-contract", type=Path, required=True)
    authority.add_argument("--runtime-contract-file-sha256", required=True)
    authority.add_argument("--runtime-contract-sha256", required=True)
    authority.add_argument("--candidate-disjoint-attestation", type=Path, required=True)
    authority.add_argument("--candidate-attestation-file-sha256", required=True)
    authority.add_argument("--reset-adapter", type=Path, required=True)
    authority.add_argument("--reset-adapter-file-sha256", required=True)
    authority.add_argument("--reset-adapter-factory", default="build_reset_only_adapter")
    authority.add_argument("--output", type=Path, required=True)

    resolve = commands.add_parser("resolve-identities")
    resolve.add_argument("--authority", type=Path, required=True)
    resolve.add_argument("--authority-file-sha256", required=True)
    resolve.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser("freeze-collection")
    freeze.add_argument("--preregistration", type=Path, required=True)
    freeze.add_argument("--preregistration-file-sha256", required=True)
    freeze.add_argument("--reset-authority", type=Path, required=True)
    freeze.add_argument("--reset-authority-file-sha256", required=True)
    freeze.add_argument("--identity-receipt", type=Path, required=True)
    freeze.add_argument("--identity-receipt-file-sha256", required=True)
    freeze.add_argument("--selected-disjoint-attestation", type=Path, required=True)
    freeze.add_argument("--selected-attestation-file-sha256", required=True)
    freeze.add_argument("--future-collection-root", type=Path, required=True)
    freeze.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "reset-authority":
        value = build_reset_authority(
            preregistration_path=args.preregistration,
            expected_preregistration_file_sha256=args.preregistration_file_sha256,
            expected_preregistration_sha256=args.preregistration_sha256,
            runtime_contract_path=args.runtime_contract,
            expected_runtime_contract_file_sha256=args.runtime_contract_file_sha256,
            expected_runtime_contract_sha256=args.runtime_contract_sha256,
            candidate_disjoint_attestation_path=args.candidate_disjoint_attestation,
            expected_candidate_attestation_file_sha256=args.candidate_attestation_file_sha256,
            reset_adapter_path=args.reset_adapter,
            expected_reset_adapter_file_sha256=args.reset_adapter_file_sha256,
            reset_adapter_factory=args.reset_adapter_factory,
        )
        _atomic_json_new(args.output, value)
        print(json.dumps({"authority_sha256": value["authority_sha256"]}, sort_keys=True))
        return 0
    if args.command == "resolve-identities":
        authority_path = _assert_file_binding(
            args.authority, args.authority_file_sha256, "reset authority"
        )
        authority = _load_json(authority_path, "reset authority")
        prereg_path = _assert_file_binding(
            Path(authority["development300_preregistration"]["path"]),
            authority["development300_preregistration"]["file_sha256"],
            "development300 preregistration",
        )
        runtime_path = _assert_file_binding(
            Path(authority["runtime_contract"]["path"]),
            authority["runtime_contract"]["file_sha256"],
            "runtime contract v2b",
        )
        adapter_path = _assert_file_binding(
            Path(authority["reset_adapter"]["path"]),
            authority["reset_adapter"]["file_sha256"],
            "reset adapter",
        )
        preregistration = _load_json(prereg_path, "development300 preregistration")
        runtime = _load_json(runtime_path, "runtime contract v2b")
        validate_reset_authority(
            authority,
            preregistration=preregistration,
            runtime_contract=runtime,
            verify_runtime_files=True,
        )
        factory = _load_reset_factory(adapter_path, authority["reset_adapter"]["factory"])
        adapter = factory(
            preregistration=preregistration,
            reset_authority=authority,
            runtime_contract=runtime,
        )
        reset_once = getattr(adapter, "reset_once", None)
        close = getattr(adapter, "close", None)
        if not callable(reset_once) or not callable(close):
            raise Development300IdentityError(
                "reset adapter must expose reset_once and close"
            )
        try:
            receipt = resolve_identities(
                preregistration=preregistration,
                runtime_contract=runtime,
                reset_authority=authority,
                reset_once=reset_once,
                reset_authority_file_sha256=args.authority_file_sha256,
                verify_runtime_files=True,
            )
        finally:
            close()
        _atomic_json_new(args.output, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "receipt_sha256": receipt["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0 if receipt["status"] == RESET_COMPLETE_STATUS else 20
    result = materialize_collection(
        preregistration_path=args.preregistration,
        expected_preregistration_file_sha256=args.preregistration_file_sha256,
        reset_authority_path=args.reset_authority,
        expected_reset_authority_file_sha256=args.reset_authority_file_sha256,
        identity_receipt_path=args.identity_receipt,
        expected_identity_receipt_file_sha256=args.identity_receipt_file_sha256,
        selected_attestation_path=args.selected_disjoint_attestation,
        expected_selected_attestation_file_sha256=args.selected_attestation_file_sha256,
        future_collection_root=args.future_collection_root,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECTION_IDENTITY_AUTHORITY_FORMAT",
    "COLLECTION_PREREGISTRATION_FORMAT",
    "Development300IdentityError",
    "RESET_AUTHORITY_FORMAT",
    "RESET_COMPLETE_STATUS",
    "RESET_INSUFFICIENT_STATUS",
    "RESET_RECEIPT_FORMAT",
    "build_collection_identity_authority",
    "build_collection_preregistration",
    "build_reset_authority",
    "file_sha256",
    "materialize_collection",
    "resolve_identities",
    "validate_identity_receipt",
    "validate_collection_identity_authority",
    "validate_reset_authority",
]
