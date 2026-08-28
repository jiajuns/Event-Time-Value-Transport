#!/usr/bin/env python3
"""Per-seed Schema6 Phase-2 runner with strict four-candidate accounting.

The production CLI cannot import a runtime adapter or construct an environment
until a separately signed execution authority binds the CPU preregistration,
this runner, the adapter, and the move_can_pot source.  The pure core accepts
dependency-injected runtime/collector/registry APIs for CPU contract tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from preregister_smolvla_piper_schema6_multiseed_collection_v2 import (
    CANDIDATE_INDICES,
    GROUP_RECEIPT_FORMAT,
    GROUP_RECEIPT_STATUS,
    R6J_RUNTIME_ARTIFACTS,
    MultiSeedProtocolError,
    canonical_sha256,
    file_sha256,
    validate_preregistration,
    validate_target_seed_manifest,
)
from resolve_smolvla_piper_target_reset_only import array_sha256, scene_sha256
from smolvla_piper_schema6_runtime_adapter_v2 import (
    RuntimeAdapterError,
    validate_runtime_contract,
)


RESET_RECEIPT_FORMAT = "etsf_smolvla_piper_schema6_multiseed_reset_receipt_v2"
RESET_RECEIPT_STATUS = "complete_verified_per_seed_reset_registry_pose"
EXECUTION_AUTHORITY_FORMAT = "etsf_smolvla_piper_schema6_multiseed_execution_authority_v2"
EXECUTION_AUTHORITY_STATUS = "authorized_adaptation80_validation50_collection_only"
V2_ACCOUNTING_FORMAT = "etsf_smolvla_piper_schema6_four_candidate_accounting_v2"
SENSITIVE_TOKENS = ("fresh", "confirmation")
FORBIDDEN_DATA_COMPONENTS = {"evaluation", "test", "testing"}
SHA_CHARS = frozenset("0123456789abcdef")


class Phase2RunnerError(RuntimeError):
    """A per-seed production invariant cannot be proven."""


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise Phase2RunnerError(f"{role} logical SHA mismatch")
    return str(recorded)


def _forbidden_component(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if any(token in lowered for token in SENSITIVE_TOKENS):
            return True
        if lowered in FORBIDDEN_DATA_COMPONENTS or lowered.startswith(("test_", "test-")):
            return True
    return False


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise Phase2RunnerError(f"{role} path is invalid")
    path = Path(os.path.abspath(os.path.expanduser(text)))
    if _forbidden_component(PurePath(path)):
        raise Phase2RunnerError(f"{role} path is forbidden")
    resolved = path.resolve(strict=False)
    if _forbidden_component(PurePath(resolved)):
        raise Phase2RunnerError(f"{role} resolves into a forbidden namespace")
    return resolved


def existing_file(value: str | os.PathLike[str], role: str) -> Path:
    path = safe_path(value, role)
    if path.is_symlink():
        raise Phase2RunnerError(f"{role} must not be a symlink")
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise Phase2RunnerError(f"{role} is unavailable") from error
    if not stat.S_ISREG(mode):
        raise Phase2RunnerError(f"{role} is not a regular file")
    return path


def load_json(path: Path, role: str) -> dict[str, Any]:
    path = existing_file(path, role)
    if path.suffix.casefold() != ".json":
        raise Phase2RunnerError(f"{role} must be JSON")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2RunnerError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise Phase2RunnerError(f"{role} must contain an object")
    return value


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_command(command: Mapping[str, Any], preregistration: Mapping[str, Any]) -> str:
    exact = {
        "split", "ordinal", "requested_seed", "expected_resolved_seed", "pair_id",
        "expected_initial_scene_state_sha256", "candidate_original_indices", "argv",
        "outputs", "bindings", "command_sha256",
    }
    command_sha = verify_signed(command, "command_sha256", "per-seed command")
    outputs = command.get("outputs")
    if (
        set(command) != exact
        or command.get("split") not in {"adaptation", "validation"}
        or type(command.get("ordinal")) is not int
        or type(command.get("requested_seed")) is not int
        or type(command.get("expected_resolved_seed")) is not int
        or not _is_sha(command.get("pair_id"))
        or not _is_sha(command.get("expected_initial_scene_state_sha256"))
        or command.get("candidate_original_indices") != list(CANDIDATE_INDICES)
        or not isinstance(command.get("argv"), list)
        or not isinstance(outputs, Mapping)
        or set(outputs) != {"seed_root", "per_seed_reset_receipt", "group_hdf5", "completed_group_receipt"}
        or command.get("bindings") is None
        or command_sha != command["command_sha256"]
    ):
        raise Phase2RunnerError("per-seed command fields changed")
    planned_root = safe_path(preregistration["outputs"]["future_collection_root"], "collection root")
    seed_root = safe_path(outputs["seed_root"], "seed output root")
    if planned_root not in seed_root.parents:
        raise Phase2RunnerError("seed output escaped preregistered root")
    expected_outputs = {
        "per_seed_reset_receipt": seed_root / "per_seed_reset_receipt.json",
        "group_hdf5": seed_root / "schema6_group.hdf5",
        "completed_group_receipt": seed_root / "completed_group_receipt.json",
    }
    for name, expected in expected_outputs.items():
        if safe_path(outputs[name], name) != expected:
            raise Phase2RunnerError(f"command output path changed: {name}")
    return command_sha


def find_command(
    preregistration: Mapping[str, Any],
    *, split: str, ordinal: int, requested_seed: int,
    expected_resolved_seed: int, expected_pair_id: str, seed_output_root: Path,
) -> dict[str, Any]:
    decoded = validate_preregistration(preregistration)
    matches = [
        row for row in decoded["commands"]
        if row.get("split") == split and row.get("ordinal") == ordinal
    ]
    if len(matches) != 1:
        raise Phase2RunnerError("command identity is missing or ambiguous")
    command = matches[0]
    _validate_command(command, preregistration)
    if (
        command["requested_seed"] != requested_seed
        or command["expected_resolved_seed"] != expected_resolved_seed
        or command["pair_id"] != expected_pair_id
        or safe_path(command["outputs"]["seed_root"], "seed output root")
        != safe_path(seed_output_root, "seed output root")
    ):
        raise Phase2RunnerError("CLI identity differs from signed command")
    return command


def selected_target_row(preregistration: Mapping[str, Any], command: Mapping[str, Any]) -> dict[str, Any]:
    binding = preregistration.get("input_bindings", {}).get("target_seed_manifest", {})
    path = existing_file(str(binding.get("path", "")), "target seed manifest")
    if file_sha256(path) != binding.get("file_sha256"):
        raise Phase2RunnerError("target seed manifest file SHA changed")
    manifest = load_json(path, "target seed manifest")
    decoded = validate_target_seed_manifest(manifest)
    if decoded["seed_manifest_sha256"] != binding.get("logical_sha256"):
        raise Phase2RunnerError("target seed manifest logical SHA changed")
    rows = [
        row for row in decoded["selected_rows"]
        if row["split"] == command["split"] and row["ordinal"] == command["ordinal"]
    ]
    if len(rows) != 1:
        raise Phase2RunnerError("target manifest row is missing or ambiguous")
    row = rows[0]
    if (
        row["requested_seed"] != command["requested_seed"]
        or row["resolved_seed"] != command["expected_resolved_seed"]
        or row["pair_id"] != command["pair_id"]
        or row["initial_scene_state_sha256"] != command["expected_initial_scene_state_sha256"]
    ):
        raise Phase2RunnerError("target manifest row differs from command")
    return row


def validate_reset_identity(raw: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    if set(raw) != {"scene_state", "measured_joint_state", "commanded_drive_target"}:
        raise Phase2RunnerError("runtime reset identity fields changed")
    observed = {
        "initial_scene_state_sha256": scene_sha256(raw["scene_state"]),
        "initial_measured_joint_state_sha256": array_sha256(raw["measured_joint_state"], role="measured joint state"),
        "initial_commanded_drive_target_sha256": array_sha256(raw["commanded_drive_target"], role="commanded drive target"),
    }
    for key, value in observed.items():
        if value != target[key]:
            raise Phase2RunnerError(f"reset identity mismatch: {key}")
    return observed


def four_candidate_accounting(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if record.get("status") != "collected_development_group":
        raise Phase2RunnerError("r6j collector did not produce a complete legal-branch group")
    query = record.get("root_query")
    if not isinstance(query, Mapping):
        raise Phase2RunnerError("root candidate query is missing")
    mask = np.asarray(query.get("feasibility_mask"), dtype=bool)
    native = list(query.get("native_action_sha256", ()))
    if mask.shape != (4,) or len(native) != 4 or any(not _is_sha(item) for item in native):
        raise Phase2RunnerError("root four-candidate registry changed")
    legal = np.flatnonzero(mask).astype(int).tolist()
    branches = record.get("branches")
    if not isinstance(branches, list):
        raise Phase2RunnerError("legal branch records are missing")
    by_original = {int(row["original_candidate_index"]): row for row in branches}
    if sorted(by_original) != legal or len(by_original) != len(branches):
        raise Phase2RunnerError("r6j legal branch accounting changed")
    result = []
    for original in CANDIDATE_INDICES:
        feasible = bool(mask[original])
        branch = by_original.get(original)
        if feasible and branch is None:
            raise Phase2RunnerError("feasible candidate lacks an executed branch")
        if not feasible and branch is not None:
            raise Phase2RunnerError("infeasible candidate was executed")
        result.append(
            {
                "original_candidate_index": original,
                "feasible": feasible,
                "execution_status": "executed_legal_branch" if feasible else "nonexecuted_censored_infeasible",
                "executed": feasible,
                "right_censored": not feasible,
                "censor_reason": "" if feasible else "root_candidate_infeasible",
                "legal_branch_index": int(branch["branch_index"]) if feasible else -1,
                "native_action_sha256": native[original],
            }
        )
    return result


def append_v2_accounting(
    path: Path, *, accounting: Sequence[Mapping[str, Any]], preregistration_sha256: str,
    command_sha256: str, pair_id: str, reset_receipt_sha256: str,
) -> None:
    with h5py.File(path, "r+") as handle:
        if "candidate_accounting_v2" in handle:
            raise Phase2RunnerError("candidate accounting already exists")
        handle.attrs["schema6_multiseed_accounting_format"] = V2_ACCOUNTING_FORMAT
        handle.attrs["preregistration_sha256"] = preregistration_sha256
        handle.attrs["command_sha256"] = command_sha256
        handle.attrs["pair_id"] = pair_id
        handle.attrs["per_seed_reset_receipt_sha256"] = reset_receipt_sha256
        group = handle.create_group("candidate_accounting_v2")
        group.create_dataset("original_candidate_index", data=np.asarray([row["original_candidate_index"] for row in accounting], dtype=np.int16))
        group.create_dataset("feasible", data=np.asarray([row["feasible"] for row in accounting], dtype=bool))
        group.create_dataset("executed", data=np.asarray([row["executed"] for row in accounting], dtype=bool))
        group.create_dataset("right_censored", data=np.asarray([row["right_censored"] for row in accounting], dtype=bool))
        group.create_dataset("legal_branch_index", data=np.asarray([row["legal_branch_index"] for row in accounting], dtype=np.int16))
        string = h5py.string_dtype("utf-8")
        group.create_dataset("execution_status", data=np.asarray([row["execution_status"] for row in accounting], dtype=object), dtype=string)
        group.create_dataset("censor_reason", data=np.asarray([row["censor_reason"] for row in accounting], dtype=object), dtype=string)
        group.create_dataset("native_action_sha256", data=np.asarray([row["native_action_sha256"] for row in accounting], dtype="S64"))
        handle.flush()


def validate_v2_group(
    path: Path, *, legacy_validate: Callable[[Path], Mapping[str, Any]],
    preregistration_sha256: str, command_sha256: str, pair_id: str,
    reset_receipt_sha256: str,
) -> dict[str, Any]:
    legacy = dict(legacy_validate(path))
    with h5py.File(path, "r") as handle:
        if (
            handle.attrs.get("schema6_multiseed_accounting_format") != V2_ACCOUNTING_FORMAT
            or handle.attrs.get("preregistration_sha256") != preregistration_sha256
            or handle.attrs.get("command_sha256") != command_sha256
            or handle.attrs.get("pair_id") != pair_id
            or handle.attrs.get("per_seed_reset_receipt_sha256") != reset_receipt_sha256
        ):
            raise Phase2RunnerError("v2 group provenance changed")
        group = handle["candidate_accounting_v2"]
        indices = group["original_candidate_index"][:].astype(int).tolist()
        feasible = group["feasible"][:].astype(bool)
        executed = group["executed"][:].astype(bool)
        censored = group["right_censored"][:].astype(bool)
        status = [item.decode() if isinstance(item, bytes) else str(item) for item in group["execution_status"][:]]
        branch_index = group["legal_branch_index"][:].astype(int)
        reasons = [item.decode() if isinstance(item, bytes) else str(item) for item in group["censor_reason"][:]]
        if (
            indices != list(CANDIDATE_INDICES)
            or not np.array_equal(executed, feasible)
            or not np.array_equal(censored, ~feasible)
            or any((status[i] != "executed_legal_branch" or branch_index[i] < 0 or reasons[i] != "") for i in range(4) if feasible[i])
            or any((status[i] != "nonexecuted_censored_infeasible" or branch_index[i] != -1 or reasons[i] != "root_candidate_infeasible") for i in range(4) if not feasible[i])
        ):
            raise Phase2RunnerError("four-candidate executed/censored accounting changed")
    return {"legacy": legacy, "branch_records": 4, "candidate_original_indices": list(CANDIDATE_INDICES)}


def collect_one_core(
    *, preregistration: Mapping[str, Any], command: Mapping[str, Any], target_row: Mapping[str, Any],
    runtime: Mapping[str, Callable[..., Any]], query_fn: Callable[..., Mapping[str, Any]],
    event_spec: Mapping[str, Any], move_can_pot_source: Mapping[str, str], max_steps: int,
    collector_api: Mapping[str, Callable[..., Any]], registry_api: Mapping[str, Callable[..., Any]],
) -> dict[str, Any]:
    prereg_sha = validate_preregistration(preregistration)["preregistration_sha256"]
    command_sha = _validate_command(command, preregistration)
    seed_root = safe_path(command["outputs"]["seed_root"], "seed output root")
    if seed_root.exists() or seed_root.is_symlink():
        raise FileExistsError(seed_root)
    seed_root.parent.mkdir(parents=True, exist_ok=True)
    seed_root.mkdir(mode=0o755)
    reset_count = 0
    canonical_registry: dict[str, Any] | None = None

    def verified_reset(seed: int, instruction: str):
        nonlocal reset_count
        observation, resolved, observed_instruction = runtime["reset"](seed, instruction)
        if resolved != target_row["resolved_seed"] or observed_instruction != target_row["instruction"]:
            raise Phase2RunnerError("runtime reset seed/instruction mismatch")
        validate_reset_identity(runtime["identity_snapshot"](), target_row)
        if canonical_registry is not None:
            registry_api["assert_runtime_registry_identity"](runtime["task"](), canonical_registry)
        reset_count += 1
        return observation, resolved, observed_instruction

    verified_reset(int(command["requested_seed"]), str(target_row["instruction"]))
    canonical_registry = registry_api["build_runtime_object_registry"](runtime["task"]())
    registry_sha = registry_api["registry_sha256"](canonical_registry)
    pose_spec = registry_api["build_pose_quality_spec"](
        canonical_registry, move_can_pot_source=move_can_pot_source
    )
    pose_sha = registry_api["spec_sha256"](
        pose_spec, expected_registry_sha256=registry_sha
    )
    immutable_json(seed_root / "object_registry.json", canonical_registry)
    immutable_json(seed_root / "pose_quality_spec.json", pose_spec)
    reset_receipt: dict[str, Any] = {
        "format": RESET_RECEIPT_FORMAT,
        "status": RESET_RECEIPT_STATUS,
        "preregistration_sha256": prereg_sha,
        "command_sha256": command_sha,
        "split": command["split"],
        "ordinal": command["ordinal"],
        "requested_seed": command["requested_seed"],
        "resolved_seed": target_row["resolved_seed"],
        "pair_id": command["pair_id"],
        "initial_scene_state_sha256": target_row["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": target_row["initial_measured_joint_state_sha256"],
        "initial_commanded_drive_target_sha256": target_row["initial_commanded_drive_target_sha256"],
        "object_registry_sha256": registry_sha,
        "pose_spec_sha256": pose_sha,
        "identity_validation_count_before_policy_query": reset_count,
        "policy_queries_before_reset_receipt": 0,
        "evaluation_execution_authorized": False,
        "protected_inputs_read": False,
    }
    reset_receipt["reset_receipt_sha256"] = canonical_sha256(reset_receipt)
    immutable_json(seed_root / "per_seed_reset_receipt.json", reset_receipt)
    wrapped_runtime = dict(runtime)
    wrapped_runtime["reset"] = verified_reset
    record = collector_api["collect_dense_group"](
        runtime=wrapped_runtime,
        query_fn=query_fn,
        requested_seed=int(command["requested_seed"]),
        instruction=str(target_row["instruction"]),
        object_registry=canonical_registry,
        pose_quality_spec=pose_spec,
        event_spec=event_spec,
        max_steps=max_steps,
    )
    if record.get("resolved_seed") != target_row["resolved_seed"]:
        raise Phase2RunnerError("collected group resolved seed changed")
    accounting = four_candidate_accounting(record)
    group_path = seed_root / "schema6_group.hdf5"
    staging = seed_root / ".schema6_group.staging.hdf5"
    collector_api["save_schema6_group"](staging, record)
    append_v2_accounting(
        staging,
        accounting=accounting,
        preregistration_sha256=prereg_sha,
        command_sha256=command_sha,
        pair_id=command["pair_id"],
        reset_receipt_sha256=reset_receipt["reset_receipt_sha256"],
    )
    validate_v2_group(
        staging,
        legacy_validate=collector_api["validate_schema6_group_file"],
        preregistration_sha256=prereg_sha,
        command_sha256=command_sha,
        pair_id=command["pair_id"],
        reset_receipt_sha256=reset_receipt["reset_receipt_sha256"],
    )
    os.link(staging, group_path)
    staging.unlink()
    group_sha = file_sha256(group_path)
    receipt: dict[str, Any] = {
        "format": GROUP_RECEIPT_FORMAT,
        "status": GROUP_RECEIPT_STATUS,
        "preregistration_sha256": prereg_sha,
        "command_sha256": command_sha,
        "split": command["split"],
        "ordinal": command["ordinal"],
        "requested_seed": command["requested_seed"],
        "resolved_seed": target_row["resolved_seed"],
        "pair_id": command["pair_id"],
        "candidate_original_indices": list(CANDIDATE_INDICES),
        "branch_records": 4,
        "per_seed_reset_receipt_sha256": reset_receipt["reset_receipt_sha256"],
        "object_registry_sha256": registry_sha,
        "pose_spec_sha256": pose_sha,
        "group_file_sha256": group_sha,
    }
    receipt["group_receipt_sha256"] = canonical_sha256(receipt)
    immutable_json(seed_root / "completed_group_receipt.json", receipt)
    return receipt


def validate_execution_authority(
    value: Mapping[str, Any], *, preregistration_path: Path,
    preregistration_file_sha256: str, preregistration_sha256: str,
) -> dict[str, Any]:
    logical = verify_signed(value, "authority_sha256", "execution authority")
    exact = {
        "format", "status", "production_execution_authorized", "preregistration_path",
        "preregistration_file_sha256", "preregistration_sha256", "runner_path",
        "runner_file_sha256", "runtime_adapter_path", "runtime_adapter_file_sha256",
        "move_can_pot_source_path", "move_can_pot_source_file_sha256",
        "runtime_contract",
        "evaluation_commands_authorized", "test_inputs_read", "fresh_inputs_accepted",
        "confirmation_inputs_accepted", "authority_sha256",
    }
    if (
        set(value) != exact
        or value.get("format") != EXECUTION_AUTHORITY_FORMAT
        or value.get("status") != EXECUTION_AUTHORITY_STATUS
        or value.get("production_execution_authorized") is not True
        or safe_path(str(value.get("preregistration_path", "")), "authorized preregistration") != preregistration_path
        or value.get("preregistration_file_sha256") != preregistration_file_sha256
        or value.get("preregistration_sha256") != preregistration_sha256
        or value.get("evaluation_commands_authorized") != 0
        or value.get("test_inputs_read") is not False
        or value.get("fresh_inputs_accepted") is not False
        or value.get("confirmation_inputs_accepted") is not False
    ):
        raise Phase2RunnerError("execution authority scope changed")
    for role in ("runner", "runtime_adapter", "move_can_pot_source"):
        path = existing_file(str(value[f"{role}_path"]), role)
        if file_sha256(path) != value[f"{role}_file_sha256"]:
            raise Phase2RunnerError(f"execution authority {role} SHA changed")
    if existing_file(value["runner_path"], "runner") != Path(__file__).resolve():
        raise Phase2RunnerError("execution authority binds a different runner")
    try:
        runtime_contract = validate_runtime_contract(value["runtime_contract"])
    except (KeyError, TypeError, RuntimeAdapterError) as error:
        raise Phase2RunnerError("execution authority runtime contract is invalid") from error
    move_binding = runtime_contract["runtime_source_artifacts"]["robotwin_move_can_pot"]
    if (
        Path(move_binding["path"]).resolve()
        != existing_file(value["move_can_pot_source_path"], "move_can_pot source")
        or move_binding["sha256"] != value["move_can_pot_source_file_sha256"]
    ):
        raise Phase2RunnerError("runtime contract and authority disagree on move_can_pot")
    return {"authority_sha256": logical, **dict(value), "runtime_contract": runtime_contract}


def production_preflight(preregistration_path: Path, execution_authority_path: Path | None) -> dict[str, Any]:
    prereg_path = existing_file(preregistration_path, "preregistration")
    prereg_file_sha = file_sha256(prereg_path)
    prereg = load_json(prereg_path, "preregistration")
    decoded = validate_preregistration(prereg)
    if execution_authority_path is None:
        raise Phase2RunnerError("production execution authority dependency is absent")
    authority_path = existing_file(execution_authority_path, "execution authority")
    authority = validate_execution_authority(
        load_json(authority_path, "execution authority"),
        preregistration_path=prereg_path,
        preregistration_file_sha256=prereg_file_sha,
        preregistration_sha256=decoded["preregistration_sha256"],
    )
    # Revalidate the target identity manifest and every bound r6j byte before any import.
    target = prereg["input_bindings"]["target_seed_manifest"]
    target_path = existing_file(target["path"], "target seed manifest")
    if file_sha256(target_path) != target["file_sha256"]:
        raise Phase2RunnerError("target seed manifest dependency changed")
    r6j = prereg["input_bindings"]["r6j_runtime_code"]
    artifact_hashes = {}
    for name in R6J_RUNTIME_ARTIFACTS:
        row = r6j["artifacts"][name]
        path = existing_file(row["path"], f"r6j {name}")
        if file_sha256(path) != row["sha256"]:
            raise Phase2RunnerError(f"r6j dependency changed: {name}")
        artifact_hashes[name] = row["sha256"]
    if canonical_sha256(artifact_hashes) != r6j["closure_sha256"]:
        raise Phase2RunnerError("r6j closure SHA changed")
    return {
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": prereg_file_sha,
        "preregistration": prereg,
        "preregistration_sha256": decoded["preregistration_sha256"],
        "authority": authority,
    }


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Phase2RunnerError(f"cannot import {name}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def production_collect_one(args: argparse.Namespace) -> dict[str, Any]:
    authority_env = os.environ.get("ETSF_SCHEMA6_V2_EXECUTION_AUTHORITY")
    preflight = production_preflight(
        args.preregistration,
        Path(authority_env) if authority_env else None,
    )
    prereg = preflight["preregistration"]
    command = find_command(
        prereg,
        split=args.split,
        ordinal=args.ordinal,
        requested_seed=args.requested_seed,
        expected_resolved_seed=args.expected_resolved_seed,
        expected_pair_id=args.expected_pair_id,
        seed_output_root=args.seed_output_root,
    )
    target = selected_target_row(prereg, command)
    authority = preflight["authority"]
    r6j_root = Path(prereg["input_bindings"]["r6j_runtime_code"]["root"])
    sys.path.insert(0, str(r6j_root))
    collector = _load_module(r6j_root / "collect_smolvla_piper_schema6_dense_event_branches.py", "etsf_bound_schema6_collector_v2")
    materializer = _load_module(r6j_root / "materialize_smolvla_piper_schema6_reset_contract.py", "etsf_bound_schema6_materializer_v2")
    pose = _load_module(r6j_root / "etsf_schema6_pose_quality.py", "etsf_bound_schema6_pose_v2")
    adapter = _load_module(Path(authority["runtime_adapter_path"]), "etsf_bound_schema6_runtime_adapter_v2")
    event_spec_path = Path(prereg["input_bindings"]["event_spec"]["path"])
    event_spec = load_json(event_spec_path, "event spec")
    built = adapter.build_runtime(command=command, event_spec=event_spec)
    if not isinstance(built, Mapping) or set(built) != {"runtime", "query_fn", "max_steps", "close"}:
        raise Phase2RunnerError("runtime adapter interface changed")
    try:
        receipt = collect_one_core(
            preregistration=prereg,
            command=command,
            target_row=target,
            runtime=built["runtime"],
            query_fn=built["query_fn"],
            event_spec=event_spec,
            move_can_pot_source={
                "path": authority["move_can_pot_source_path"],
                "sha256": authority["move_can_pot_source_file_sha256"],
            },
            max_steps=int(built["max_steps"]),
            collector_api={
                "collect_dense_group": collector.collect_dense_group,
                "save_schema6_group": collector.save_schema6_group,
                "validate_schema6_group_file": collector.validate_schema6_group_file,
            },
            registry_api={
                "build_runtime_object_registry": materializer.build_runtime_object_registry,
                "assert_runtime_registry_identity": materializer.assert_runtime_registry_identity,
                "build_pose_quality_spec": materializer.build_pose_quality_spec,
                "registry_sha256": pose.registry_sha256,
                "spec_sha256": pose.spec_sha256,
            },
        )
    finally:
        built["close"]()
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect-one", "preflight"))
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--split", choices=("adaptation", "validation"))
    parser.add_argument("--ordinal", type=int)
    parser.add_argument("--requested-seed", type=int)
    parser.add_argument("--expected-resolved-seed", type=int)
    parser.add_argument("--expected-pair-id")
    parser.add_argument("--seed-output-root", type=Path)
    parser.add_argument("--execution-authority", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        result = production_preflight(args.preregistration, args.execution_authority)
        print(json.dumps({"status": "production_preflight_complete", "preregistration_sha256": result["preregistration_sha256"]}, sort_keys=True))
        return 0
    required = (args.split, args.ordinal, args.requested_seed, args.expected_resolved_seed, args.expected_pair_id, args.seed_output_root)
    if any(value is None for value in required):
        raise Phase2RunnerError("collect-one command identity arguments are incomplete")
    receipt = production_collect_one(args)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXECUTION_AUTHORITY_FORMAT", "EXECUTION_AUTHORITY_STATUS", "Phase2RunnerError",
    "V2_ACCOUNTING_FORMAT", "collect_one_core", "find_command",
    "four_candidate_accounting", "production_preflight", "selected_target_row",
    "validate_execution_authority", "validate_reset_identity", "validate_v2_group",
]
