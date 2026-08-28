#!/usr/bin/env python3
"""Fail-closed external supervisor for the single paired evaluation400 lane.

This process consumes an already frozen paired-success v3 core, its independent
Ed25519 decision, the resulting execution bundle, and the separately reviewed
execution inventory.  It owns only orchestration and immutable receipts.  The
actual simulator/policy condition is delegated to a content-addressed external
condition-runner interface.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import signal
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import smolvla_piper_paired_success_protocol_v3 as paired_v3
import evaluate_smolvla_piper_evaluation400_results_v3 as result_v3


FORMAT = "etsf_smolvla_piper_evaluation400_external_executor_supervisor_v3"
PLAN_FORMAT = "etsf_smolvla_piper_evaluation400_external_executor_plan_v3"
STATE_FORMAT = "etsf_smolvla_piper_evaluation400_external_executor_state_v3"
CLAIM_FORMAT = "etsf_smolvla_piper_evaluation400_worm_lane_claim_v3"
PAIR_STARTED_FORMAT = "etsf_smolvla_piper_evaluation400_pair_started_v3"
CONDITION_REQUEST_FORMAT = "etsf_smolvla_piper_evaluation400_condition_request_v3"
CONDITION_STARTED_FORMAT = "etsf_smolvla_piper_evaluation400_condition_started_v3"
RUNNER_RESULT_FORMAT = "etsf_smolvla_piper_evaluation400_condition_runner_result_v3"
CONDITION_RECEIPT_FORMAT = result_v3.CONDITION_FORMAT
PAIR_RECEIPT_FORMAT = result_v3.PAIR_FORMAT
EXECUTION_RECEIPT_FORMAT = result_v3.TERMINAL_FORMAT
RUNTIME_CONTRACT_FORMAT = "etsf_smolvla_piper_condition_runner_runtime_contract_v3"
STAGE_FORMAT = "etsf_smolvla_piper_evaluation400_condition_stage_v3"
DEPENDENCY_CLOSURE_FORMAT = (
    "etsf_smolvla_piper_evaluation400_local_import_dependency_closure_v3"
)
DEPENDENCY_CLOSURE_STATUS = (
    "recursive_static_imports_and_limited_dynamic_imports_content_addressed"
)
BOUND_EXTERNAL_DYNAMIC_IMPORTS = frozenset({
    "rlinf.envs.robotwin.robotwin_env",
    "robotwin.envs.vector_env",
    "envs._base_task",
    "envs.robot.robot",
    "envs.move_can_pot",
})

CLAIM_STATUS = "consumed_before_evaluation400_pair_zero"
RUNNER_RESULT_STATUS = "complete_single_condition_from_bound_snapshot"
CONDITION_COMPLETE = result_v3.CONDITION_STATUS
PAIR_COMPLETE = result_v3.PAIR_STATUS
EXECUTION_COMPLETE = result_v3.TERMINAL_STATUS
EXECUTION_FAILED = "failed_closed_evaluation400_external_execution"
PAIR_COUNT = 400
CANDIDATE_COUNT = 4
EXPECTED_GPU_FRAGMENT = "RTX 4090"
CONDITIONS = frozenset({"baseline", "etsf"})
SHA_CHARS = frozenset("0123456789abcdef")
_HELD_UNPROVEN_LOCKS: list[Any] = []


class ExecutorV3Error(RuntimeError):
    """An authority, identity, lifecycle, or WORM invariant failed closed."""


class UnprovenProcessGroup(ExecutorV3Error):
    """A Popen attempt cannot prove the entire process group reaped."""


class IncompleteLane(ExecutorV3Error):
    """A write-ahead start lacks its terminal and may never be replayed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _strict_int(value: Any, role: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExecutorV3Error(f"{role} must be an integer >= {minimum}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutorV3Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_path(
    value: str | os.PathLike[str],
    role: str,
    *,
    must_exist: bool = False,
) -> Path:
    raw = os.fspath(value)
    if not raw or "\0" in raw:
        raise ExecutorV3Error(f"{role} path is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(raw)))
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ExecutorV3Error(f"{role} path contains a symlink")
    try:
        return lexical.resolve(strict=must_exist)
    except OSError as error:
        raise ExecutorV3Error(f"{role} path is unavailable") from error


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExecutorV3Error("hashed artifact is not a regular file")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def existing_file(
    value: str | os.PathLike[str], role: str, *, frozen: bool = True
) -> Path:
    path = safe_path(value, role, must_exist=True)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ExecutorV3Error(f"{role} must be a single-link regular file")
    if frozen and metadata.st_mode & 0o222 and (
        metadata.st_uid == os.geteuid() or os.access(path, os.W_OK)
    ):
        raise ExecutorV3Error(f"{role} must be frozen read-only")
    return path


def read_json(
    value: str | os.PathLike[str],
    role: str,
    *,
    expected_file_sha256: str | None = None,
    frozen: bool = True,
) -> tuple[Path, dict[str, Any], str]:
    path = existing_file(value, role, frozen=frozen)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        first = os.fstat(descriptor)
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            payload.extend(block)
        second = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (first.st_dev, first.st_ino, first.st_size) != (
        second.st_dev,
        second.st_ino,
        second.st_size,
    ):
        raise ExecutorV3Error(f"{role} changed during read")
    file_sha = digest.hexdigest()
    if expected_file_sha256 is not None and (
        not is_sha(expected_file_sha256) or file_sha != expected_file_sha256
    ):
        raise ExecutorV3Error(f"{role} file SHA mismatch")
    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorV3Error(f"{role} is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ExecutorV3Error(f"{role} must contain a JSON object")
    return path, decoded, file_sha


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    logical = unsigned.pop(field, None)
    if not is_sha(logical) or logical != canonical_sha256(unsigned):
        raise ExecutorV3Error(f"{role} logical SHA mismatch")
    return str(logical)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def immutable_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        payload = json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8") + b"\n"
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise OSError("short immutable JSON write")
            remaining = remaining[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def immutable_bytes(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise OSError("short immutable byte write")
            remaining = remaining[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _record(path: Path, logical_sha256: str | None = None) -> dict[str, str]:
    row = {"path": str(path), "file_sha256": hash_file(path)}
    if logical_sha256 is not None:
        row["logical_sha256"] = logical_sha256
    return row


def _code_record(path: Path, expected_sha256: str, role: str) -> dict[str, str]:
    source = existing_file(path, role, frozen=True)
    if not is_sha(expected_sha256) or hash_file(source) != expected_sha256:
        raise ExecutorV3Error(f"{role} implementation SHA mismatch")
    return {"path": str(source), "file_sha256": expected_sha256}


def _local_import_target(scripts_root: Path, module: str) -> Path | None:
    top_level = module.split(".", 1)[0]
    candidate = scripts_root / f"{top_level}.py"
    return candidate if candidate.is_file() and not candidate.is_symlink() else None


def build_local_dependency_closure(
    roots: Sequence[Path], *, scripts_root: Path,
) -> dict[str, Any]:
    """Recursively bind local imports and the two reviewed dynamic-load lanes."""

    trusted_root = scripts_root.resolve(strict=True)
    pending: list[Path] = []
    root_relatives: list[str] = []
    for raw in roots:
        source = existing_file(raw, "local dependency closure root", frozen=True)
        try:
            relative = source.relative_to(trusted_root).as_posix()
        except ValueError as error:
            raise ExecutorV3Error(
                "local dependency closure root escaped the trusted scripts root"
            ) from error
        if "/" in relative or not relative.endswith(".py"):
            raise ExecutorV3Error("only flat reviewed scripts may enter local closure")
        pending.append(source)
        root_relatives.append(relative)

    files: dict[str, str] = {}
    dynamic: list[dict[str, Any]] = []
    while pending:
        source = pending.pop()
        relative = source.relative_to(trusted_root).as_posix()
        if relative in files:
            continue
        payload = source.read_bytes()
        try:
            tree = ast.parse(payload, filename=str(source))
        except (SyntaxError, ValueError) as error:
            raise ExecutorV3Error(f"cannot statically parse local dependency {relative}") from error
        files[relative] = hashlib.sha256(payload).hexdigest()
        discovered: set[Path] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ExecutorV3Error(
                        f"relative import is forbidden in dependency closure: {relative}"
                    )
                modules = [node.module] if node.module else []
            else:
                modules = []
            for module in modules:
                if not isinstance(module, str):
                    continue
                target = _local_import_target(trusted_root, module)
                if target is not None:
                    discovered.add(target)
            if not isinstance(node, ast.Call):
                continue
            dotted = ""
            function = node.func
            if isinstance(function, ast.Name):
                dotted = function.id
            elif isinstance(function, ast.Attribute):
                parts: list[str] = []
                cursor: ast.AST = function
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    parts.append(cursor.id)
                    dotted = ".".join(reversed(parts))
            if dotted == "__import__":
                raise ExecutorV3Error(
                    f"unbounded __import__ in local dependency: {relative}:{node.lineno}"
                )
            if dotted == "importlib.import_module":
                argument = node.args[0] if node.args else None
                if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                    raise ExecutorV3Error(
                        f"nonliteral dynamic import is not closed: {relative}:{node.lineno}"
                    )
                module = argument.value
                target = _local_import_target(trusted_root, module)
                if target is not None:
                    discovered.add(target)
                elif module not in BOUND_EXTERNAL_DYNAMIC_IMPORTS:
                    raise ExecutorV3Error(
                        f"dynamic external import lacks reviewed authority: {module}"
                    )
                else:
                    dynamic.append({
                        "relative_path": relative,
                        "line": node.lineno,
                        "kind": "literal_external_runtime_module",
                        "module": module,
                        "authority": "runtime_contract.runtime_source_artifacts",
                    })
            elif dotted == "importlib.util.spec_from_file_location":
                dynamic.append({
                    "relative_path": relative,
                    "line": node.lineno,
                    "kind": "content_addressed_file_module",
                    "module": None,
                    "authority": "condition_runner.bound_file_path_and_sha256",
                })
        pending.extend(sorted(discovered, key=str))

    base = {
        "format": DEPENDENCY_CLOSURE_FORMAT,
        "status": DEPENDENCY_CLOSURE_STATUS,
        "scripts_root": str(trusted_root),
        "roots": sorted(set(root_relatives)),
        "files": [
            {"relative_path": relative, "file_sha256": files[relative]}
            for relative in sorted(files)
        ],
        "dynamic_imports": sorted(
            dynamic,
            key=lambda row: (
                row["relative_path"], row["line"], row["kind"], row.get("module") or ""
            ),
        ),
        "unclosed_local_import_count": 0,
        "unbounded_dynamic_import_count": 0,
    }
    return {**base, "closure_sha256": canonical_sha256(base)}


def validate_local_dependency_closure(
    record: Mapping[str, Any], *, expected_roots: Sequence[Path],
) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {
        "path", "file_sha256", "logical_sha256"
    }:
        raise ExecutorV3Error("local dependency closure descriptor changed")
    _path, value, file_sha = read_json(
        Path(str(record["path"])),
        "local dependency closure",
        expected_file_sha256=str(record["file_sha256"]),
    )
    logical = verify_signed(value, "closure_sha256", "local dependency closure")
    scripts_root = Path(str(value.get("scripts_root")))
    rebuilt = build_local_dependency_closure(
        expected_roots, scripts_root=scripts_root
    )
    if (
        file_sha != record["file_sha256"]
        or logical != record["logical_sha256"]
        or rebuilt != value
    ):
        raise ExecutorV3Error("local dependency closure changed")
    return value


def load_executor_signing_key(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_public_key_sha256: str,
) -> tuple[Any, str, str]:
    source = existing_file(path, "executor Ed25519 private key", frozen=True)
    file_sha = hash_file(source)
    if not is_sha(expected_file_sha256) or file_sha != expected_file_sha256:
        raise ExecutorV3Error("executor private-key file SHA mismatch")
    payload = source.read_bytes()
    if len(payload) == 64:
        try:
            private_bytes = bytes.fromhex(payload.decode("ascii"))
        except (UnicodeError, ValueError) as error:
            raise ExecutorV3Error("executor private key is not canonical raw/hex") from error
    elif len(payload) == 65 and payload.endswith(b"\n"):
        try:
            private_bytes = bytes.fromhex(payload[:-1].decode("ascii"))
        except (UnicodeError, ValueError) as error:
            raise ExecutorV3Error("executor private key is not canonical raw/hex") from error
    elif len(payload) == 32:
        private_bytes = payload
    else:
        raise ExecutorV3Error("executor private key must contain exactly 32 bytes")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise ExecutorV3Error("cryptography Ed25519 is required") from error
    public_hex = public_bytes.hex()
    public_sha = hashlib.sha256(public_bytes).hexdigest()
    if public_sha != expected_public_key_sha256:
        raise ExecutorV3Error("executor public key differs from paired core identity")
    return private_key, public_hex, file_sha


def sign_executor_receipt(
    *,
    receipt_format: str,
    receipt_status: str,
    statement: Mapping[str, Any],
    private_key: Any,
) -> dict[str, Any]:
    signature = private_key.sign(
        result_v3._receipt_signing_bytes(receipt_format, statement)
    ).hex()
    base = {
        "format": receipt_format,
        "status": receipt_status,
        "signature_algorithm": "Ed25519",
        "statement": dict(statement),
        "executor_signature_ed25519_hex": signature,
    }
    return {**base, "receipt_sha256": canonical_sha256(base)}


def validate_pair_rows(core: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluation = core.get("evaluation400")
    rows = evaluation.get("pairs") if isinstance(evaluation, Mapping) else None
    expected_fields = {
        "ordinal",
        "pair_id",
        "target_manifest_global_ordinal",
        "requested_seed",
        "resolved_seed",
        "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256",
        "condition_order",
        "candidate_count",
    }
    if (
        not isinstance(evaluation, Mapping)
        or type(evaluation.get("pair_count")) is not int
        or evaluation["pair_count"] != PAIR_COUNT
        or not isinstance(rows, list)
        or len(rows) != PAIR_COUNT
    ):
        raise ExecutorV3Error("evaluation400 pair inventory changed")
    normalized: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ExecutorV3Error("evaluation400 pair fields changed")
        pair_id = raw.get("pair_id")
        order = raw.get("condition_order")
        if (
            type(raw.get("ordinal")) is not int
            or raw["ordinal"] != ordinal
            or type(raw.get("target_manifest_global_ordinal")) is not int
            or raw["target_manifest_global_ordinal"] < 0
            or type(raw.get("requested_seed")) is not int
            or type(raw.get("resolved_seed")) is not int
            or not isinstance(pair_id, str)
            or not pair_id
            or pair_id in pair_ids
            or not isinstance(order, list)
            or len(order) != 2
            or set(order) != CONDITIONS
            or raw.get("candidate_count") != CANDIDATE_COUNT
            or type(raw.get("candidate_count")) is not int
            or any(
                not is_sha(raw.get(field))
                for field in (
                    "initial_scene_state_sha256",
                    "initial_measured_joint_state_sha256",
                    "initial_commanded_drive_target_sha256",
                )
            )
        ):
            raise ExecutorV3Error("evaluation400 pair ordering/identity changed")
        pair_ids.add(pair_id)
        normalized.append(dict(raw))
    if evaluation.get("pair_identity_set_sha256") is None or not is_sha(
        evaluation.get("pair_identity_set_sha256")
    ):
        raise ExecutorV3Error("evaluation400 pair identity-set SHA is invalid")
    return normalized


def validate_runtime_contract(
    inventory: Mapping[str, Any], *, condition_runner: Mapping[str, str]
) -> dict[str, Any]:
    stack = inventory.get("execution_stack")
    if not isinstance(stack, Mapping):
        raise ExecutorV3Error("execution inventory stack is missing")
    collector = stack.get("collector_implementation")
    runtime = stack.get("runtime_contract")
    simulator = stack.get("simulator_implementation")
    if (
        not isinstance(collector, Mapping)
        or dict(collector) != dict(condition_runner)
        or not isinstance(runtime, Mapping)
        or not isinstance(simulator, Mapping)
        or dict(simulator) == dict(condition_runner)
        or Path(str(simulator.get("path", ""))).name
        != "smolvla_piper_schema6_runtime_adapter_v2.py"
    ):
        raise ExecutorV3Error("condition-runner is not the reviewed collector binding")
    runtime_path, value, runtime_file_sha = read_json(
        Path(str(runtime.get("path"))),
        "condition-runner runtime contract",
        expected_file_sha256=str(runtime.get("file_sha256")),
    )
    logical = verify_signed(value, "runtime_contract_sha256", "runtime contract")
    expected_fields = {
        "format",
        "status",
        "interface_version",
        "mode",
        "request_format",
        "result_format",
        "condition_runner_implementation_sha256",
        "simulator_implementation_sha256",
        "visible_device_contract",
        "pair_attempt",
        "candidate_count",
        "condition_names",
        "schema6_execution_authority_file_sha256",
        "schema6_runtime_contract_sha256",
        "max_episode_steps",
        "runtime_contract_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != RUNTIME_CONTRACT_FORMAT
        or value.get("status") != "externally_reviewed_condition_runner_interface"
        or value.get("interface_version") != 3
        or type(value.get("interface_version")) is not int
        or value.get("mode") != "execute-condition-v3"
        or value.get("request_format") != CONDITION_REQUEST_FORMAT
        or value.get("result_format") != RUNNER_RESULT_FORMAT
        or value.get("condition_runner_implementation_sha256")
        != condition_runner["file_sha256"]
        or value.get("simulator_implementation_sha256")
        != simulator.get("file_sha256")
        or value.get("visible_device_contract") != "exact_gpu_uuid_as_cuda_visible_devices_and_cuda0"
        or value.get("pair_attempt") != 0
        or type(value.get("pair_attempt")) is not int
        or value.get("candidate_count") != CANDIDATE_COUNT
        or type(value.get("candidate_count")) is not int
        or value.get("condition_names") != ["baseline", "etsf"]
        or not is_sha(value.get("schema6_execution_authority_file_sha256"))
        or not is_sha(value.get("schema6_runtime_contract_sha256"))
        or type(value.get("max_episode_steps")) is not int
        or value["max_episode_steps"] != 200
    ):
        raise ExecutorV3Error("condition-runner runtime contract changed")
    return {
        "path": str(runtime_path),
        "file_sha256": runtime_file_sha,
        "logical_sha256": logical,
        "schema6_execution_authority_file_sha256": value[
            "schema6_execution_authority_file_sha256"
        ],
        "schema6_runtime_contract_sha256": value[
            "schema6_runtime_contract_sha256"
        ],
        "max_episode_steps": 200,
    }


def validate_schema6_full_horizon_binding(
    runtime_interface: Mapping[str, Any],
    execution_authority_record: Mapping[str, Any],
) -> None:
    if (
        not isinstance(execution_authority_record, Mapping)
        or set(execution_authority_record) != {"path", "file_sha256"}
        or runtime_interface.get("schema6_execution_authority_file_sha256")
        != execution_authority_record.get("file_sha256")
    ):
        raise ExecutorV3Error("runtime interface does not bind schema6 authority")
    _path, authority, authority_file_sha = read_json(
        Path(str(execution_authority_record["path"])),
        "schema6 execution authority",
        expected_file_sha256=str(execution_authority_record["file_sha256"]),
    )
    contract = authority.get("runtime_contract")
    if (
        authority_file_sha != runtime_interface[
            "schema6_execution_authority_file_sha256"
        ]
        or not isinstance(contract, Mapping)
        or contract.get("runtime_contract_sha256")
        != runtime_interface.get("schema6_runtime_contract_sha256")
        or type(contract.get("max_episode_steps")) is not int
        or contract["max_episode_steps"] != 200
        or runtime_interface.get("max_episode_steps") != 200
    ):
        raise ExecutorV3Error("schema6 runtime is not the exact 200-step authority")


def validate_authority_bundle(
    *,
    core_path: Path,
    core_file_sha256: str,
    decision_path: Path,
    decision_file_sha256: str,
    bundle_path: Path,
    bundle_file_sha256: str,
    inventory_path: Path,
    inventory_file_sha256: str,
    inventory_sha256: str,
    supervisor: Mapping[str, str],
    condition_runner: Mapping[str, str],
) -> dict[str, Any]:
    core_bound, core, core_file = read_json(
        core_path, "paired core v3", expected_file_sha256=core_file_sha256
    )
    core_logical = paired_v3.validate_core(core)
    decision_bound, decision, decision_file = read_json(
        decision_path,
        "Ed25519 execution decision",
        expected_file_sha256=decision_file_sha256,
    )
    decision_logical = paired_v3.verify_decision(
        decision, core=core, core_file_sha256=core_file
    )
    bundle_bound, bundle, bundle_file = read_json(
        bundle_path, "execution bundle v3", expected_file_sha256=bundle_file_sha256
    )
    bundle_logical = paired_v3.validate_bundle(bundle)
    rebuilt = paired_v3.freeze_bundle(
        core_path=core_bound,
        core_file_sha256=core_file,
        decision_path=decision_bound,
        decision_file_sha256=decision_file,
    )
    if rebuilt != bundle:
        raise ExecutorV3Error("execution bundle differs from full dependency reconstruction")
    if (
        bundle.get("protocol_core")
        != {
            "path": str(core_bound),
            "file_sha256": core_file,
            "logical_sha256": core_logical,
        }
        or bundle.get("ed25519_decision")
        != {
            "path": str(decision_bound),
            "file_sha256": decision_file,
            "logical_sha256": decision_logical,
        }
    ):
        raise ExecutorV3Error("bundle does not bind the supplied core and Ed25519 decision")
    inventory_bound, inventory, inventory_file = read_json(
        inventory_path,
        "independent execution inventory",
        expected_file_sha256=inventory_file_sha256,
    )
    inventory_logical = verify_signed(
        inventory, "attestation_sha256", "independent execution inventory"
    )
    if inventory_logical != inventory_sha256:
        raise ExecutorV3Error("execution inventory logical SHA mismatch")
    validated_inventory, _issuer, inventory_record, stack_sha = (
        paired_v3._validate_execution_inventory(
            path=inventory_bound,
            expected_file_sha256=inventory_file,
            expected_logical_sha256=inventory_logical,
        )
    )
    core_inventory = core.get("execution_inventory")
    executor = inventory.get("executor")
    if (
        validated_inventory != inventory
        or not isinstance(core_inventory, Mapping)
        or core_inventory.get("attestation") != inventory_record
        or core_inventory.get("stack_binding_sha256") != stack_sha
        or bundle.get("execution_inventory") != core_inventory
        or not isinstance(executor, Mapping)
        or executor.get("implementation") != dict(supervisor)
        or core_inventory.get("executor_implementation_file_sha256")
        != supervisor["file_sha256"]
    ):
        raise ExecutorV3Error("supervisor/execution-inventory binding changed")
    runtime_contract = validate_runtime_contract(
        inventory, condition_runner=condition_runner
    )
    pairs = validate_pair_rows(core)
    if (
        bundle.get("pair_identity_set_sha256")
        != core["evaluation400"]["pair_identity_set_sha256"]
        or bundle.get("authorized_pair_count") != PAIR_COUNT
        or bundle.get("execution_authorized") is not True
        or bundle.get("external_executor_only") is not True
    ):
        raise ExecutorV3Error("execution bundle lane authority changed")
    return {
        "core": core,
        "decision": decision,
        "bundle": bundle,
        "inventory": inventory,
        "pairs": pairs,
        "runtime_contract": runtime_contract,
        "records": {
            "core": _record(core_bound, core_logical),
            "decision": _record(decision_bound, decision_logical),
            "bundle": _record(bundle_bound, bundle_logical),
            "inventory": _record(inventory_bound, inventory_logical),
        },
    }


def lane_identity(authority: Mapping[str, Any]) -> str:
    records = authority["records"]
    return canonical_sha256(
        {
            "lane": "single_evaluation400_paired_lane_v3",
            "core_sha256": records["core"]["logical_sha256"],
            "decision_sha256": records["decision"]["logical_sha256"],
            "bundle_sha256": records["bundle"]["logical_sha256"],
            "inventory_sha256": records["inventory"]["logical_sha256"],
            "pair_identity_set_sha256": authority["core"]["evaluation400"][
                "pair_identity_set_sha256"
            ],
            "pair_count": PAIR_COUNT,
        }
    )


def _secure_directory(path: Path, role: str) -> Path:
    root = safe_path(path, role, must_exist=True)
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise ExecutorV3Error(f"{role} must be owner-controlled")
    return root


def _ensure_private_subdirectory(path: Path, role: str) -> Path:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    return _secure_directory(path, role)


def acquire_lane_claim(
    *,
    ledger_root: Path,
    identity: str,
    plan: Mapping[str, Any],
    private_key: Any,
) -> dict[str, Any]:
    if not is_sha(identity):
        raise ExecutorV3Error("evaluation400 lane identity is invalid")
    root = _secure_directory(ledger_root, "WORM ledger root")
    claims = _ensure_private_subdirectory(root / "claims", "WORM claims root")
    lanes = _ensure_private_subdirectory(root / "lanes", "WORM lanes root")
    claim_path = claims / f"evaluation400-{identity}.claim.json"
    statement = {
        "protocol_core_sha256": plan["authority"]["core"]["logical_sha256"],
        "decision_sha256": plan["authority"]["decision"]["logical_sha256"],
        "bundle_sha256": plan["authority"]["bundle"]["logical_sha256"],
        "execution_nonce_hex": plan["execution_nonce_hex"],
        "pair_identity_set_sha256": plan["pair_identity_set_sha256"],
        "deployment_binding_sha256": plan["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": plan[
            "policy_runtime_action_binding_sha256"
        ],
        "ledger_id_sha256": plan["ledger_id_sha256"],
        "claim_ordinal": 0,
        "claim_count": 1,
        "claim_release_count": 0,
        "claimed_before_any_outcome_read": True,
        "outcome_or_success_values_read_before_claim": 0,
        "retry_or_reclaim_authorized": False,
    }
    if set(statement) != result_v3.CLAIM_STATEMENT_FIELDS:
        raise ExecutorV3Error("claim statement differs from result evaluator v3")
    claim = sign_executor_receipt(
        receipt_format=result_v3.CLAIM_FORMAT,
        receipt_status=result_v3.CLAIM_STATUS,
        statement=statement,
        private_key=private_key,
    )
    try:
        immutable_json(claim_path, claim)
    except FileExistsError as error:
        raise ExecutorV3Error("evaluation400 WORM lane was already claimed") from error
    lane_root = lanes / identity
    try:
        lane_root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ExecutorV3Error("evaluation400 WORM lane directory already exists") from error
    (lane_root / "pairs").mkdir(mode=0o700)
    (lane_root / "events").mkdir(mode=0o700)
    return {
        "path": str(claim_path),
        "file_sha256": hash_file(claim_path),
        "logical_sha256": claim["receipt_sha256"],
        "lane_root": str(lane_root),
        "lane_identity_sha256": identity,
    }


def validate_lane_claim(plan: Mapping[str, Any]) -> dict[str, Any]:
    identity = plan.get("lane_identity_sha256")
    root = _secure_directory(
        Path(str(plan.get("ledger_root"))), "WORM ledger root"
    )
    path = root / "claims" / f"evaluation400-{identity}.claim.json"
    bound, claim, file_sha = read_json(path, "evaluation400 WORM claim")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(str(plan["executor_public_key_hex"]))
        )
        statement, logical = result_v3._verify_executor_receipt(
            claim,
            expected_format=result_v3.CLAIM_FORMAT,
            expected_status=result_v3.CLAIM_STATUS,
            public_key=public_key,
            role="evaluation400 WORM claim",
        )
        ledger_id = result_v3._validate_claim_statement(
            statement,
            core={
                "protocol_core_sha256": plan["authority"]["core"]["logical_sha256"],
                "evaluation400": {
                    "pair_identity_set_sha256": plan["pair_identity_set_sha256"]
                },
                "deployment": {
                    "deployment_binding_sha256": plan["deployment_binding_sha256"],
                    "policy_runtime_action_binding_sha256": plan[
                        "policy_runtime_action_binding_sha256"
                    ],
                },
            },
            decision={"decision_sha256": plan["authority"]["decision"]["logical_sha256"]},
            bundle={"bundle_sha256": plan["authority"]["bundle"]["logical_sha256"]},
            execution_nonce_hex=str(plan["execution_nonce_hex"]),
        )
    except Exception as error:
        raise ExecutorV3Error("evaluation400 WORM claim signature/contract changed") from error
    lane_root = root / "lanes" / str(identity)
    if (
        set(claim) != result_v3.RECEIPT_FIELDS
        or ledger_id != plan.get("ledger_id_sha256")
        or not lane_root.is_dir()
        or not (lane_root / "pairs").is_dir()
        or not (lane_root / "events").is_dir()
    ):
        raise ExecutorV3Error("evaluation400 WORM claim changed")
    return {
        "path": str(bound),
        "file_sha256": file_sha,
        "logical_sha256": logical,
        "lane_root": str(lane_root),
        "lane_identity_sha256": str(identity),
    }


def reject_inherited_cuda_mapping() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ExecutorV3Error("inherited CUDA_VISIBLE_DEVICES remapping is forbidden")


def gpu_audit(index: int) -> dict[str, Any]:
    identity = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().split(",", 1)
    if len(identity) != 2 or EXPECTED_GPU_FRAGMENT not in identity[0].strip():
        raise ExecutorV3Error("designated GPU is not an RTX 4090")
    pids = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    return {
        "gpu_index": index,
        "gpu_name": identity[0].strip(),
        "gpu_uuid": identity[1].strip(),
        "compute_pids": sorted(
            int(row.strip()) for row in pids.splitlines() if row.strip().isdigit()
        ),
    }


def wait_two_idle(
    index: int,
    expected_uuid: str,
    *,
    interval: float,
    audit: Callable[[int], Mapping[str, Any]] = gpu_audit,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    while len(accepted) < 2:
        current = dict(audit(index))
        if (
            current.get("gpu_uuid") != expected_uuid
            or EXPECTED_GPU_FRAGMENT not in str(current.get("gpu_name", ""))
        ):
            raise ExecutorV3Error("physical GPU UUID/name changed")
        if current.get("compute_pids"):
            accepted.clear()
        else:
            accepted.append(current)
        if len(accepted) < 2:
            sleep(interval)
    base = {
        "gpu_index": index,
        "gpu_name": accepted[0]["gpu_name"],
        "gpu_uuid": expected_uuid,
        "checks": 2,
    }
    return {**base, "audit_sha256": canonical_sha256(base)}


def open_gpu_lock(path: Path) -> Any:
    lock_path = safe_path(path, "GPU lock")
    parent = safe_path(lock_path.parent, "GPU lock parent", must_exist=True)
    if not parent.is_dir():
        raise ExecutorV3Error("GPU lock parent is not a directory")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        os.close(descriptor)
        raise ExecutorV3Error("GPU lock inode is unsafe")
    return os.fdopen(descriptor, "r+", encoding="ascii")


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_direct(process: subprocess.Popen[bytes], timeout: float = 10.0) -> bool:
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return process.poll() is not None


def _stop_group(
    process: subprocess.Popen[bytes], pgid: int, timeout: float = 10.0
) -> tuple[bool, bool]:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None and not _process_group_alive(pgid):
            break
        time.sleep(0.02)
    if process.poll() is None or _process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + timeout
    while _process_group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return process.poll() is not None, not _process_group_alive(pgid)


def run_condition_stage(
    *,
    command: Sequence[str],
    stage_root: Path,
    gpu_uuid: str,
    pre_popen_guard: Callable[[], None],
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    getpgid: Callable[[int], int] = os.getpgid,
) -> dict[str, Any]:
    if stage_root.exists() or stage_root.is_symlink():
        raise FileExistsError(stage_root)
    stage_root.mkdir(mode=0o700)
    original_command = list(command)
    if len(original_command) < 2:
        raise ExecutorV3Error("condition command lacks runtime Python/runner")
    command_sha256 = canonical_sha256(original_command)
    launch: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    attempted = False
    direct_reaped = False
    group_reaped = False
    returncode: int | None = None
    error: BaseException | None = None
    log_path = stage_root / "run.log"
    executable_fds: list[int] = []
    try:
        with log_path.open("xb") as log:
            pre_popen_guard()
            fd_mapping: list[dict[str, Any]] = []
            for role, raw in zip(
                ("runtime_python", "condition_runner"),
                original_command[:2],
                strict=True,
            ):
                source = existing_file(Path(raw), role)
                descriptor = os.open(
                    source,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                executable_fds.append(descriptor)
                fd_mapping.append(
                    {
                        "role": role,
                        "source_path": str(source),
                        "source_file_sha256": hash_file(source),
                        "inherited_fd": descriptor,
                        "executed_path": f"/proc/self/fd/{descriptor}",
                    }
                )
            bound_command = [
                fd_mapping[0]["executed_path"],
                "-I",
                fd_mapping[1]["executed_path"],
                *original_command[2:],
            ]
            environment = {
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            launch_base = {
                "format": STAGE_FORMAT,
                "status": "fd_bound_guard_passed_immediately_before_popen",
                "original_command": original_command,
                "command_sha256": command_sha256,
                "executed_command": bound_command,
                "fd_mapping": fd_mapping,
                "isolated_python": True,
                "environment_policy": "explicit_allowlist_no_pythonpath_pythonhome_or_ld_preload",
                "environment_keys": sorted(environment),
                "forbidden_environment_keys_absent": sorted(
                    ["PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"]
                ),
                "gpu_uuid": gpu_uuid,
                "device": "cuda:0",
            }
            launch = {**launch_base, "launch_sha256": canonical_sha256(launch_base)}
            immutable_json(stage_root / "launch.json", launch)
            attempted = True
            process = popen(
                bound_command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                pass_fds=tuple(executable_fds),
                env=environment,
            )
            if type(process.pid) is not int or process.pid <= 0:
                raise UnprovenProcessGroup("condition Popen returned invalid PID")
            pgid = getpgid(process.pid)
            if type(pgid) is not int or pgid <= 0 or pgid != process.pid:
                raise UnprovenProcessGroup("condition PGID is not isolated PID=PGID")
            returncode = process.wait()
            direct_reaped = True
    except BaseException as caught:
        error = caught
    finally:
        for descriptor in executable_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if attempted:
            if process is None:
                group_reaped = False
            elif pgid is None or pgid != process.pid:
                direct_reaped = _stop_direct(process)
                group_reaped = False
                if process.poll() is not None:
                    returncode = process.returncode
            else:
                if process.poll() is not None:
                    try:
                        returncode = process.wait(timeout=0)
                        direct_reaped = True
                    except (OSError, subprocess.TimeoutExpired):
                        direct_reaped = False
                if error is not None or not direct_reaped or _process_group_alive(pgid):
                    direct_reaped, group_reaped = _stop_group(process, pgid)
                    if process.poll() is not None:
                        returncode = process.returncode
                else:
                    group_reaped = True
        lifecycle_base = {
            "popen_attempted": attempted,
            "popen_reached": process is not None,
            "process_pid": process.pid if process is not None else None,
            "process_pgid": pgid,
            "process_group_isolated": process is not None and pgid == process.pid,
            "returncode": returncode,
            "direct_process_reaped": direct_reaped,
            "process_group_reaped": group_reaped,
            "binding_status": (
                "bound_reaped"
                if process is not None and pgid == process.pid and direct_reaped and group_reaped
                else "attempted_unproven"
                if attempted
                else "not_attempted"
            ),
        }
        lifecycle = {
            **lifecycle_base,
            "lifecycle_sha256": canonical_sha256(lifecycle_base),
        }
        immutable_json(stage_root / "lifecycle.json", lifecycle)
        exit_code = returncode if type(returncode) is int else 255
        descriptor = os.open(
            stage_root / "run.exit",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{exit_code}\n")
            handle.flush()
            os.fsync(handle.fileno())
        log_path.chmod(0o444)
        stage_root.chmod(0o555)
    if not attempted:
        if error is not None:
            raise ExecutorV3Error(
                f"condition pre-Popen guard raised {type(error).__name__}"
            ) from error
        raise ExecutorV3Error("condition stage ended before Popen")
    if not (
        process is not None
        and pgid == process.pid
        and direct_reaped
        and group_reaped
    ):
        raise UnprovenProcessGroup("condition process group lifecycle is unproven")
    if error is not None:
        raise ExecutorV3Error(f"condition stage raised {type(error).__name__}") from error
    if type(returncode) is not int or returncode != 0:
        raise ExecutorV3Error(f"condition runner exited {returncode!r}")
    result = {
        "returncode": returncode,
        "command_sha256": command_sha256,
        "launch": _record(stage_root / "launch.json", launch["launch_sha256"]),
        "lifecycle": _record(
            stage_root / "lifecycle.json", lifecycle["lifecycle_sha256"]
        ),
        "log": _record(log_path),
        "exit": _record(stage_root / "run.exit"),
    }
    return {**result, "stage_result_sha256": canonical_sha256(result)}


def shared_snapshot_sha256(pair: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "pair_id": pair["pair_id"],
            "requested_seed": pair["requested_seed"],
            "resolved_seed": pair["resolved_seed"],
            "initial_scene_state_sha256": pair["initial_scene_state_sha256"],
            "initial_measured_joint_state_sha256": pair[
                "initial_measured_joint_state_sha256"
            ],
            "initial_commanded_drive_target_sha256": pair[
                "initial_commanded_drive_target_sha256"
            ],
        }
    )


def pair_identity_sha256(pair: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(pair))


def condition_request(
    *,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    pair: Mapping[str, Any],
    condition_ordinal: int,
) -> dict[str, Any]:
    condition = pair["condition_order"][condition_ordinal]
    base = {
        "format": CONDITION_REQUEST_FORMAT,
        "status": "write_ahead_before_condition_popen",
        "plan_sha256": plan["plan_sha256"],
        "bundle_sha256": plan["authority"]["bundle"]["logical_sha256"],
        "claim_sha256": claim["logical_sha256"],
        "pair_id": pair["pair_id"],
        "ordinal": pair["ordinal"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "initial_scene_state_sha256": pair["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": pair[
            "initial_measured_joint_state_sha256"
        ],
        "initial_commanded_drive_target_sha256": pair[
            "initial_commanded_drive_target_sha256"
        ],
        "attempt": 0,
        "pair_identity_sha256": pair_identity_sha256(pair),
        "condition": condition,
        "condition_ordinal": condition_ordinal,
        "condition_order": list(pair["condition_order"]),
        "shared_snapshot_sha256": shared_snapshot_sha256(pair),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_generation_contract_sha256": canonical_sha256(
            {
                "deployment_binding_sha256": plan["deployment_binding_sha256"],
                "pair_identity_sha256": pair_identity_sha256(pair),
                "shared_snapshot_sha256": shared_snapshot_sha256(pair),
                "candidate_count": CANDIDATE_COUNT,
            }
        ),
        "postfreeze_identity_or_order_change_authorized": False,
        "outcome_visible_before_condition_start": False,
    }
    return {**base, "request_sha256": canonical_sha256(base)}


def validate_runner_result(
    result_root: Path,
    *,
    request_path: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    expected_names = {
        "condition_result.json",
        "trajectory.bin",
        "continuation.bin",
        "run.exit",
    }
    if not result_root.is_dir() or {path.name for path in result_root.iterdir()} != expected_names:
        raise ExecutorV3Error("condition-runner output inventory changed")
    result_path, result, result_file = read_json(
        result_root / "condition_result.json", "condition-runner result"
    )
    result_logical = verify_signed(
        result, "result_sha256", "condition-runner result"
    )
    expected_fields = {
        "format",
        "status",
        "request",
        "pair_id",
        "ordinal",
        "attempt",
        "condition",
        "condition_ordinal",
        "shared_snapshot_sha256",
        "candidate_count",
        "ordered_candidate_sha256",
        "candidate_legal",
        "candidate_registry_sha256",
        "schema6_execution_authority_file_sha256",
        "schema6_runtime_contract_sha256",
        "max_episode_steps",
        "selected_candidate_index",
        "selector_execution_proof",
        "selector_execution_proof_sha256",
        "selector_score_contract",
        "source_rank_score_contract_sha256",
        "source_contract_rank_score_is_success_logit",
        "source_contract_rank_score_is_success_probability",
        "formal190_target_outcome_calibrated_acceptance_margin",
        "continuation_contract",
        "continuation_policy_sha256",
        "continuation_rerank_after_root",
        "candidate_replacement_count",
        "continuation_proof_sha256",
        "task_success",
        "trajectory_artifact",
        "continuation_artifact",
        "simulator_exit_code",
        "result_sha256",
    }
    trajectory = result.get("trajectory_artifact")
    continuation_artifact = result.get("continuation_artifact")
    ordered = result.get("ordered_candidate_sha256")
    legal = result.get("candidate_legal")
    if (
        set(result) != expected_fields
        or result.get("format") != RUNNER_RESULT_FORMAT
        or result.get("status") != RUNNER_RESULT_STATUS
        or result.get("request")
        != {
            "path": str(request_path),
            "file_sha256": hash_file(request_path),
            "logical_sha256": request["request_sha256"],
        }
        or any(result.get(field) != request.get(field) for field in (
            "pair_id",
            "ordinal",
            "attempt",
            "condition",
            "condition_ordinal",
            "shared_snapshot_sha256",
            "candidate_count",
        ))
        or type(result.get("selected_candidate_index")) is not int
        or not 0 <= result["selected_candidate_index"] < CANDIDATE_COUNT
        or not isinstance(ordered, list)
        or len(ordered) != CANDIDATE_COUNT
        or any(not is_sha(item) for item in ordered)
        or len(set(ordered)) != CANDIDATE_COUNT
        or not isinstance(legal, list)
        or len(legal) != CANDIDATE_COUNT
        or any(type(item) is not bool for item in legal)
        or not any(legal)
        or legal[result["selected_candidate_index"]] is not True
        or result.get("candidate_registry_sha256")
        != result_v3._candidate_registry_sha(
            str(result.get("pair_id")), ordered, legal
        )
        or not is_sha(result.get("schema6_execution_authority_file_sha256"))
        or not is_sha(result.get("schema6_runtime_contract_sha256"))
        or type(result.get("max_episode_steps")) is not int
        or result["max_episode_steps"] != 200
        or result.get("continuation_contract")
        != result_v3.CONTINUATION_CONTRACT
        or not is_sha(result.get("continuation_policy_sha256"))
        or result.get("continuation_rerank_after_root") is not False
        or type(result.get("candidate_replacement_count")) is not int
        or result["candidate_replacement_count"] != 0
        or result.get("continuation_proof_sha256")
        != result_v3._continuation_proof(result)
        or type(result.get("task_success")) is not bool
        or type(result.get("simulator_exit_code")) is not int
        or result["simulator_exit_code"] != 0
        or type(result.get("ordinal")) is not int
        or type(result.get("attempt")) is not int
        or result["attempt"] != 0
        or type(result.get("condition_ordinal")) is not int
        or type(result.get("candidate_count")) is not int
        or result["candidate_count"] != CANDIDATE_COUNT
        or not is_sha(result.get("selector_execution_proof_sha256"))
        or not isinstance(result.get("selector_execution_proof"), Mapping)
        or result["selector_execution_proof_sha256"]
        != canonical_sha256(result["selector_execution_proof"])
        or result["selector_execution_proof"].get("score_contract")
        != result.get("selector_score_contract")
        or result["selector_execution_proof"].get(
            "source_rank_score_contract_sha256"
        ) != result.get("source_rank_score_contract_sha256")
        or result["selector_execution_proof"].get(
            "source_contract_rank_score_is_success_logit"
        ) is not False
        or result["selector_execution_proof"].get(
            "source_contract_rank_score_is_success_probability"
        ) is not False
        or result["selector_execution_proof"].get(
            "formal190_target_outcome_calibrated_acceptance_margin"
        ) is not result.get("formal190_target_outcome_calibrated_acceptance_margin")
        or result.get("selector_score_contract") not in {
            "lowest_legal_feasibility_root_candidate",
            "five_member_adjusted_source_composite_candidate_rank_score_margin",
        }
        or not isinstance(result.get("source_rank_score_contract_sha256"), list)
        or any(
            not is_sha(value)
            for value in result["source_rank_score_contract_sha256"]
        )
        or result.get("source_contract_rank_score_is_success_logit") is not False
        or result.get("source_contract_rank_score_is_success_probability") is not False
        or type(
            result.get("formal190_target_outcome_calibrated_acceptance_margin")
        ) is not bool
        or (
            result.get("condition") == "baseline"
            and (
                result["source_rank_score_contract_sha256"] != []
                or result["selector_score_contract"]
                != "lowest_legal_feasibility_root_candidate"
                or result["formal190_target_outcome_calibrated_acceptance_margin"]
                is not False
            )
        )
        or (
            result.get("condition") == "etsf"
            and (
                len(result["source_rank_score_contract_sha256"]) != 5
                or result["selector_score_contract"]
                != "five_member_adjusted_source_composite_candidate_rank_score_margin"
                or result["formal190_target_outcome_calibrated_acceptance_margin"]
                is not True
            )
        )
        or (
            result.get("condition") == "baseline"
            and result.get("selected_candidate_index") != legal.index(True)
        )
    ):
        raise ExecutorV3Error("condition-runner result semantics changed")
    try:
        result_v3.validate_selector_execution_proof(
            result["selector_execution_proof"],
            condition=str(result["condition"]),
            candidate_legal=result["candidate_legal"],
            selected_candidate_index=result["selected_candidate_index"],
        )
    except result_v3.Evaluation400ResultError as error:
        raise ExecutorV3Error("condition-runner selector proof is invalid") from error
    for raw, expected_path, role in (
        (trajectory, result_root / "trajectory.bin", "trajectory artifact"),
        (
            continuation_artifact,
            result_root / "continuation.bin",
            "continuation artifact",
        ),
    ):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"path", "file_sha256"}
            or raw.get("path") != str(expected_path)
            or not is_sha(raw.get("file_sha256"))
            or hash_file(existing_file(expected_path, role)) != raw["file_sha256"]
        ):
            raise ExecutorV3Error(f"{role} binding changed")
    exit_path = existing_file(result_root / "run.exit", "runner run.exit")
    if exit_path.read_bytes() != b"0\n":
        raise ExecutorV3Error("condition-runner run.exit changed")
    return {
        "value": result,
        "record": {
            "path": str(result_path),
            "file_sha256": result_file,
            "logical_sha256": result_logical,
        },
    }


def _condition_command(
    plan: Mapping[str, Any], request_path: Path, result_root: Path
) -> list[str]:
    runtime = plan["runtime_contract"]
    inventory = plan["inventory_components"]
    return [
        str(plan["python"]["path"]),
        str(plan["condition_runner"]["path"]),
        "--mode",
        "execute-condition-v3",
        "--request",
        str(request_path),
        "--request-file-sha256",
        hash_file(request_path),
        "--output-root",
        str(result_root),
        "--runtime-contract",
        str(runtime["path"]),
        "--runtime-contract-file-sha256",
        str(runtime["file_sha256"]),
        "--condition-runner-source",
        str(plan["condition_runner"]["path"]),
        "--condition-runner-source-file-sha256",
        str(plan["condition_runner"]["file_sha256"]),
        "--paired-protocol-implementation",
        str(plan["paired_protocol"]["path"]),
        "--paired-protocol-implementation-file-sha256",
        str(plan["paired_protocol"]["file_sha256"]),
        "--simulator-implementation",
        str(inventory["simulator_implementation"]["path"]),
        "--simulator-implementation-file-sha256",
        str(inventory["simulator_implementation"]["file_sha256"]),
        "--protocol-core",
        str(plan["authority"]["core"]["path"]),
        "--protocol-core-file-sha256",
        str(plan["authority"]["core"]["file_sha256"]),
        "--ed25519-decision",
        str(plan["authority"]["decision"]["path"]),
        "--ed25519-decision-file-sha256",
        str(plan["authority"]["decision"]["file_sha256"]),
        "--execution-bundle",
        str(plan["authority"]["bundle"]["path"]),
        "--execution-bundle-file-sha256",
        str(plan["authority"]["bundle"]["file_sha256"]),
        "--canonical-event-spec",
        str(plan["runner_dependencies"]["canonical_event_spec"]["path"]),
        "--canonical-event-spec-file-sha256",
        str(plan["runner_dependencies"]["canonical_event_spec"]["file_sha256"]),
        "--schema6-execution-authority",
        str(plan["runner_dependencies"]["schema6_execution_authority"]["path"]),
        "--schema6-execution-authority-file-sha256",
        str(plan["runner_dependencies"]["schema6_execution_authority"]["file_sha256"]),
        "--dense-collector-implementation",
        str(plan["runner_dependencies"]["dense_collector_implementation"]["path"]),
        "--dense-collector-implementation-file-sha256",
        str(plan["runner_dependencies"]["dense_collector_implementation"]["file_sha256"]),
        "--adapter-trainer-implementation",
        str(plan["runner_dependencies"]["adapter_trainer_implementation"]["path"]),
        "--adapter-trainer-implementation-file-sha256",
        str(plan["runner_dependencies"]["adapter_trainer_implementation"]["file_sha256"]),
        "--local-dependency-closure",
        str(plan["runner_dependencies"]["local_dependency_closure"]["path"]),
        "--local-dependency-closure-file-sha256",
        str(plan["runner_dependencies"]["local_dependency_closure"]["file_sha256"]),
        "--local-dependency-closure-sha256",
        str(plan["runner_dependencies"]["local_dependency_closure"]["logical_sha256"]),
        "--device",
        "cuda:0",
    ]


def _worm_pair_root(claim: Mapping[str, Any], ordinal: int) -> Path:
    return Path(str(claim["lane_root"])) / "pairs" / f"{ordinal:03d}"


def scan_completed_pairs(
    *,
    claim: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    gap_seen = False
    for ordinal, pair in enumerate(pairs):
        pair_root = _worm_pair_root(claim, ordinal)
        if not pair_root.exists():
            gap_seen = True
            continue
        if gap_seen:
            raise IncompleteLane("WORM pair ordinals are not contiguous")
        terminal_path = pair_root / "pair_terminal.json"
        started_path = pair_root / "pair_started.json"
        if not started_path.is_file() or not terminal_path.is_file():
            raise IncompleteLane(
                f"pair {ordinal} was started without a terminal and cannot be replayed"
            )
        _path, terminal, file_sha = read_json(
            terminal_path, f"pair {ordinal} terminal"
        )
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(str(plan["executor_public_key_hex"]))
            )
            statement, logical = result_v3._verify_executor_receipt(
                terminal,
                expected_format=PAIR_RECEIPT_FORMAT,
                expected_status=PAIR_COMPLETE,
                public_key=public_key,
                role=f"pair {ordinal} terminal",
            )
        except Exception as error:
            raise IncompleteLane(f"pair {ordinal} terminal signature is invalid") from error
        if (
            statement.get("ordinal") != ordinal
            or statement.get("pair_id") != pair["pair_id"]
            or statement.get("condition_attempt_count") != 2
            or type(statement.get("condition_attempt_count")) is not int
            or not isinstance(statement.get("condition_receipts"), list)
            or len(statement["condition_receipts"]) != 2
        ):
            raise IncompleteLane(f"pair {ordinal} terminal is invalid")
        ledger_events: list[dict[str, Any]] = []
        expected_types = (
            "condition_started_preoutcome",
            "condition_terminal",
            "condition_started_preoutcome",
            "condition_terminal",
            "pair_terminal",
        )
        for offset, event_type in enumerate(expected_types):
            event_index = 5 * ordinal + offset
            event_path = (
                Path(str(claim["lane_root"]))
                / "events"
                / f"{event_index:04d}-{event_type}.json"
            )
            try:
                _event_path, event_receipt, event_file_sha = read_json(
                    event_path, f"ledger event {event_index}"
                )
                event_statement, event_logical = result_v3._verify_executor_receipt(
                    event_receipt,
                    expected_format=result_v3.LEDGER_EVENT_FORMAT,
                    expected_status=result_v3.LEDGER_EVENT_STATUS,
                    public_key=public_key,
                    role=f"ledger event {event_index}",
                )
            except Exception as error:
                raise IncompleteLane(
                    f"pair {ordinal} ledger event closure is invalid"
                ) from error
            expected_global = 2 * ordinal + (offset // 2) if offset < 4 else None
            if (
                event_statement.get("event_index") != event_index
                or event_statement.get("event_type") != event_type
                or event_statement.get("pair_ordinal") != ordinal
                or event_statement.get("global_condition_ordinal") != expected_global
            ):
                raise IncompleteLane(f"pair {ordinal} ledger event order changed")
            ledger_events.append(
                {
                    "event_index": event_index,
                    "event_type": event_type,
                    "pair_ordinal": ordinal,
                    "global_condition_ordinal": expected_global,
                    "path": str(event_path),
                    "file_sha256": event_file_sha,
                    "logical_sha256": event_logical,
                }
            )
        completed.append(
            {
                "ordinal": ordinal,
                "pair_id": pair["pair_id"],
                "path": str(terminal_path),
                "file_sha256": file_sha,
                "logical_sha256": logical,
                "condition_receipts": list(statement["condition_receipts"]),
                "ledger_event_receipts": ledger_events,
            }
        )
    return completed


def _common_statement_binding(
    plan: Mapping[str, Any], dependency_rehash_sha256: str
) -> dict[str, Any]:
    return {
        "protocol_core_sha256": plan["authority"]["core"]["logical_sha256"],
        "decision_sha256": plan["authority"]["decision"]["logical_sha256"],
        "bundle_sha256": plan["authority"]["bundle"]["logical_sha256"],
        "execution_nonce_hex": plan["execution_nonce_hex"],
        "pair_identity_set_sha256": plan["pair_identity_set_sha256"],
        "deployment_binding_sha256": plan["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": plan[
            "policy_runtime_action_binding_sha256"
        ],
        "preexecution_dependency_rehash_sha256": dependency_rehash_sha256,
    }


def condition_execution_statement(
    *,
    plan: Mapping[str, Any],
    pair: Mapping[str, Any],
    position: int,
    result: Mapping[str, Any],
    dependency_rehash_sha256: str,
    core: Mapping[str, Any],
    ledger_condition_start_event_sha256: str,
    execution_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    condition_id = pair["condition_order"][position]
    statement = {
        **_common_statement_binding(plan, dependency_rehash_sha256),
        "ledger_condition_start_event_sha256": ledger_condition_start_event_sha256,
        "execution_artifacts": dict(execution_artifacts),
        "global_condition_ordinal": 2 * pair["ordinal"] + position,
        "pair_ordinal": pair["ordinal"],
        "pair_id": pair["pair_id"],
        "target_manifest_global_ordinal": pair["target_manifest_global_ordinal"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "condition_position": position,
        "condition_id": condition_id,
        "condition_order": list(pair["condition_order"]),
        # Result-evaluator v3 uses a one-based execution attempt while the WORM
        # lane and runner request remain the required zero-based attempt=0.
        "attempt_index": 1,
        "retry_count": 0,
        "condition_started": True,
        "condition_terminal": True,
        "incomplete": False,
        "excluded": False,
        "initial_scene_state_sha256": pair["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": pair[
            "initial_measured_joint_state_sha256"
        ],
        "initial_commanded_drive_target_sha256": pair[
            "initial_commanded_drive_target_sha256"
        ],
        "reset_proof_sha256": shared_snapshot_sha256(pair),
        "candidate_count": CANDIDATE_COUNT,
        "ordered_candidate_sha256": list(result["ordered_candidate_sha256"]),
        "candidate_legal": list(result["candidate_legal"]),
        "candidate_registry_sha256": result["candidate_registry_sha256"],
        "continuation_contract": result["continuation_contract"],
        "continuation_policy_sha256": result["continuation_policy_sha256"],
        "continuation_rerank_after_root": result[
            "continuation_rerank_after_root"
        ],
        "candidate_replacement_count": result["candidate_replacement_count"],
        "continuation_proof_sha256": result["continuation_proof_sha256"],
        "selector": (
            core["deployment"]["baseline_selector"]
            if condition_id == "baseline"
            else core["deployment"]["etsf_selector"]
        ),
        "selected_candidate_ordinal": result["selected_candidate_index"],
        "success": result["task_success"],
        "success_source": result_v3.SUCCESS_SOURCE,
        "predicted_success_used_as_outcome": False,
    }
    if set(statement) != result_v3.CONDITION_STATEMENT_FIELDS:
        raise ExecutorV3Error("condition statement differs from result evaluator v3")
    return statement


def pair_execution_statement(
    *,
    plan: Mapping[str, Any],
    pair: Mapping[str, Any],
    condition_records: Sequence[Mapping[str, Any]],
    condition_statements: Sequence[Mapping[str, Any]],
    ledger_condition_terminal_event_sha256: Sequence[str],
    dependency_rehash_sha256: str,
) -> dict[str, Any]:
    by_condition = {row["condition_id"]: row for row in condition_statements}
    first = condition_statements[0]
    statement = {
        **_common_statement_binding(plan, dependency_rehash_sha256),
        "ordinal": pair["ordinal"],
        "pair_id": pair["pair_id"],
        "target_manifest_global_ordinal": pair["target_manifest_global_ordinal"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "condition_order": list(pair["condition_order"]),
        "condition_receipts": list(condition_records),
        "reset_proof_sha256": first["reset_proof_sha256"],
        "candidate_registry_sha256": first["candidate_registry_sha256"],
        "continuation_proof_sha256": first["continuation_proof_sha256"],
        "ledger_condition_terminal_event_sha256": list(
            ledger_condition_terminal_event_sha256
        ),
        "condition_attempt_count": 2,
        "complete_condition_count": 2,
        "retry_count": 0,
        "incomplete": False,
        "excluded": False,
        "baseline_success": by_condition["baseline"]["success"],
        "etsf_success": by_condition["etsf"]["success"],
        "success_source": result_v3.SUCCESS_SOURCE,
    }
    if set(statement) != result_v3.PAIR_STATEMENT_FIELDS:
        raise ExecutorV3Error("pair statement differs from result evaluator v3")
    return statement


def ledger_event_statement(
    *,
    plan: Mapping[str, Any],
    dependency_rehash_sha256: str,
    event_index: int,
    event_type: str,
    previous_entry_sha256: str,
    pair: Mapping[str, Any] | None,
    position: int | None,
    artifact_receipt_sha256: str | None,
) -> dict[str, Any]:
    if pair is None:
        pair_ordinal = global_ordinal = condition_id = None
    elif position is None:
        pair_ordinal = pair["ordinal"]
        global_ordinal = condition_id = None
    else:
        pair_ordinal = pair["ordinal"]
        global_ordinal = 2 * pair["ordinal"] + position
        condition_id = pair["condition_order"][position]
    statement = {
        **_common_statement_binding(plan, dependency_rehash_sha256),
        "ledger_id_sha256": plan["ledger_id_sha256"],
        "event_index": event_index,
        "event_type": event_type,
        "previous_entry_sha256": previous_entry_sha256,
        "pair_ordinal": pair_ordinal,
        "global_condition_ordinal": global_ordinal,
        "condition_position": position,
        "condition_id": condition_id,
        "artifact_receipt_sha256": artifact_receipt_sha256,
        "outcome_or_success_read_before_event": (
            event_type != "condition_started_preoutcome"
        ),
    }
    if set(statement) != result_v3.LEDGER_EVENT_STATEMENT_FIELDS:
        raise ExecutorV3Error("ledger event statement differs from result evaluator v3")
    return statement


def append_ledger_event(
    *,
    claim: Mapping[str, Any],
    plan: Mapping[str, Any],
    dependency_rehash_sha256: str,
    cursor: dict[str, Any],
    event_type: str,
    pair: Mapping[str, Any] | None,
    position: int | None,
    artifact_receipt_sha256: str | None,
    private_key: Any,
) -> dict[str, Any]:
    event_index = _strict_int(cursor.get("event_index"), "ledger event cursor")
    previous = cursor.get("previous_entry_sha256")
    if not is_sha(previous):
        raise ExecutorV3Error("ledger previous-entry SHA is invalid")
    statement = ledger_event_statement(
        plan=plan,
        dependency_rehash_sha256=dependency_rehash_sha256,
        event_index=event_index,
        event_type=event_type,
        previous_entry_sha256=str(previous),
        pair=pair,
        position=position,
        artifact_receipt_sha256=artifact_receipt_sha256,
    )
    receipt = sign_executor_receipt(
        receipt_format=result_v3.LEDGER_EVENT_FORMAT,
        receipt_status=result_v3.LEDGER_EVENT_STATUS,
        statement=statement,
        private_key=private_key,
    )
    event_path = (
        Path(str(claim["lane_root"]))
        / "events"
        / f"{event_index:04d}-{event_type}.json"
    )
    immutable_json(event_path, receipt)
    descriptor = {
        "event_index": event_index,
        "event_type": event_type,
        "pair_ordinal": None if pair is None else pair["ordinal"],
        "global_condition_ordinal": (
            None
            if pair is None or position is None
            else 2 * pair["ordinal"] + position
        ),
        **_record(event_path, receipt["receipt_sha256"]),
    }
    cursor["event_index"] = event_index + 1
    cursor["previous_entry_sha256"] = receipt["receipt_sha256"]
    cursor.setdefault("events", []).append(descriptor)
    return descriptor


def execute_pair(
    *,
    root: Path,
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    pair: Mapping[str, Any],
    physical_gpu: Mapping[str, Any],
    core: Mapping[str, Any],
    executor_private_key: Any,
    dependency_rehash_sha256: str,
    ledger_cursor: dict[str, Any],
    pre_popen_guard_factory: Callable[[Sequence[str]], Callable[[], None]],
    stage_runner: Callable[..., Mapping[str, Any]] = run_condition_stage,
    condition_idle_prover: Callable[[int, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ordinal = int(pair["ordinal"])
    pair_root = _worm_pair_root(claim, ordinal)
    try:
        pair_root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise IncompleteLane(f"pair {ordinal} was already started") from error
    snapshot_sha = shared_snapshot_sha256(pair)
    started_base = {
        "format": PAIR_STARTED_FORMAT,
        "status": "write_ahead_pair_started_no_terminal_yet",
        "claim_sha256": claim["logical_sha256"],
        "bundle_sha256": plan["authority"]["bundle"]["logical_sha256"],
        "ordinal": ordinal,
        "pair_id": pair["pair_id"],
        "attempt": 0,
        "pair_identity_sha256": pair_identity_sha256(pair),
        "shared_snapshot_sha256": snapshot_sha,
        "condition_order": list(pair["condition_order"]),
        "candidate_count": CANDIDATE_COUNT,
        "started_unix_ns": time.time_ns(),
    }
    pair_started = {
        **started_base,
        "pair_started_sha256": canonical_sha256(started_base),
    }
    immutable_json(pair_root / "pair_started.json", pair_started)
    condition_receipts: list[dict[str, Any]] = []
    condition_statements: list[dict[str, Any]] = []
    runner_values: list[dict[str, Any]] = []
    condition_terminal_event_shas: list[str] = []
    for condition_ordinal in range(2):
        request = condition_request(
            plan=plan,
            claim=claim,
            pair=pair,
            condition_ordinal=condition_ordinal,
        )
        condition = request["condition"]
        condition_root = pair_root / f"condition_{condition_ordinal}_{condition}"
        condition_root.mkdir(mode=0o700)
        request_path = condition_root / "request.json"
        immutable_json(request_path, request)
        output_condition_root = (
            root / "pairs" / f"{ordinal:03d}" / f"condition_{condition_ordinal}_{condition}"
        )
        output_condition_root.parent.mkdir(parents=True, exist_ok=True)
        output_condition_root.mkdir(mode=0o700)
        idle_prover = condition_idle_prover or (
            lambda index, uuid: wait_two_idle(index, uuid, interval=1.0)
        )
        condition_idle_before = dict(
            idle_prover(int(plan["gpu_index"]), str(physical_gpu["gpu_uuid"]))
        )
        if (
            condition_idle_before.get("gpu_uuid") != physical_gpu["gpu_uuid"]
            or type(condition_idle_before.get("checks")) is not int
            or condition_idle_before["checks"] != 2
            or not is_sha(condition_idle_before.get("audit_sha256"))
        ):
            raise ExecutorV3Error("condition did not start after two same-UUID idle checks")
        idle_before_path = output_condition_root / "gpu_idle_before_condition.json"
        immutable_json(idle_before_path, condition_idle_before)
        condition_started_base = {
            "format": CONDITION_STARTED_FORMAT,
            "status": "write_ahead_condition_started_no_terminal_yet",
            "pair_started_sha256": pair_started["pair_started_sha256"],
            "request_sha256": request["request_sha256"],
            "ordinal": ordinal,
            "attempt": 0,
            "condition": condition,
            "condition_ordinal": condition_ordinal,
            "started_unix_ns": time.time_ns(),
        }
        condition_started = {
            **condition_started_base,
            "condition_started_sha256": canonical_sha256(condition_started_base),
        }
        immutable_json(condition_root / "condition_started.json", condition_started)
        start_event = append_ledger_event(
            claim=claim,
            plan=plan,
            dependency_rehash_sha256=dependency_rehash_sha256,
            cursor=ledger_cursor,
            event_type="condition_started_preoutcome",
            pair=pair,
            position=condition_ordinal,
            artifact_receipt_sha256=None,
            private_key=executor_private_key,
        )
        stage_root = output_condition_root / "stage"
        result_root = stage_root / "result"
        command = _condition_command(plan, request_path, result_root)
        try:
            stage = dict(
                stage_runner(
                    command=command,
                    stage_root=stage_root,
                    gpu_uuid=str(physical_gpu["gpu_uuid"]),
                    pre_popen_guard=pre_popen_guard_factory(command),
                )
            )
            lifecycle_record = stage.get("lifecycle")
            if not isinstance(lifecycle_record, Mapping) or set(lifecycle_record) != {
                "path", "file_sha256", "logical_sha256"
            }:
                raise ExecutorV3Error("condition stage lacks lifecycle proof")
            _life_path, lifecycle, lifecycle_file = read_json(
                Path(str(lifecycle_record["path"])), "condition lifecycle"
            )
            lifecycle_logical = verify_signed(
                lifecycle, "lifecycle_sha256", "condition lifecycle"
            )
            if (
                lifecycle_file != lifecycle_record["file_sha256"]
                or lifecycle_logical != lifecycle_record["logical_sha256"]
                or type(lifecycle.get("process_pid")) is not int
                or type(lifecycle.get("process_pgid")) is not int
                or lifecycle["process_pid"] != lifecycle["process_pgid"]
                or lifecycle.get("process_group_isolated") is not True
                or lifecycle.get("direct_process_reaped") is not True
                or lifecycle.get("process_group_reaped") is not True
                or lifecycle.get("binding_status") != "bound_reaped"
                or type(lifecycle.get("returncode")) is not int
                or lifecycle["returncode"] != 0
            ):
                raise ExecutorV3Error("condition process-group lifecycle is not proven")
            condition_idle = dict(
                idle_prover(int(plan["gpu_index"]), str(physical_gpu["gpu_uuid"]))
            )
            if (
                condition_idle.get("gpu_uuid") != physical_gpu["gpu_uuid"]
                or type(condition_idle.get("checks")) is not int
                or condition_idle["checks"] != 2
                or not is_sha(condition_idle.get("audit_sha256"))
            ):
                raise ExecutorV3Error("condition did not end with two same-UUID idle checks")
            idle_after_path = output_condition_root / "gpu_idle_after_condition.json"
            immutable_json(idle_after_path, condition_idle)
            runner = validate_runner_result(
                result_root, request_path=request_path, request=request
            )
        except UnprovenProcessGroup:
            raise
        except BaseException as error:
            failure_base = {
                "format": CONDITION_RECEIPT_FORMAT,
                "status": "failed_condition_terminal_process_group_reaped",
                "request_sha256": request["request_sha256"],
                "ordinal": ordinal,
                "attempt": 0,
                "condition": condition,
                "condition_ordinal": condition_ordinal,
                "error_type": type(error).__name__,
                "rerun_authorized": False,
            }
            failure = {
                **failure_base,
                "condition_receipt_sha256": canonical_sha256(failure_base),
            }
            immutable_json(condition_root / "condition_terminal.json", failure)
            raise
        result = runner["value"]
        execution_artifacts = {
            "runner_result": dict(runner["record"]),
            "stage_launch": dict(stage["launch"]),
            "stage_lifecycle": dict(stage["lifecycle"]),
            "stage_log": dict(stage["log"]),
            "stage_exit": dict(stage["exit"]),
            "gpu_idle_before": _record(
                idle_before_path, condition_idle_before["audit_sha256"]
            ),
            "gpu_idle_after": _record(
                idle_after_path, condition_idle["audit_sha256"]
            ),
            "gpu_uuid": str(physical_gpu["gpu_uuid"]),
        }
        statement = condition_execution_statement(
            plan=plan,
            pair=pair,
            position=condition_ordinal,
            result=result,
            dependency_rehash_sha256=dependency_rehash_sha256,
            core=core,
            ledger_condition_start_event_sha256=start_event["logical_sha256"],
            execution_artifacts=execution_artifacts,
        )
        receipt = sign_executor_receipt(
            receipt_format=CONDITION_RECEIPT_FORMAT,
            receipt_status=CONDITION_COMPLETE,
            statement=statement,
            private_key=executor_private_key,
        )
        terminal_path = condition_root / "condition_terminal.json"
        immutable_json(terminal_path, receipt)
        condition_root.chmod(0o555)
        record = _record(terminal_path, receipt["receipt_sha256"])
        condition_receipts.append(
            {
                "global_condition_ordinal": 2 * ordinal + condition_ordinal,
                "pair_ordinal": ordinal,
                "condition_position": condition_ordinal,
                "condition_id": condition,
                **record,
            }
        )
        terminal_event = append_ledger_event(
            claim=claim,
            plan=plan,
            dependency_rehash_sha256=dependency_rehash_sha256,
            cursor=ledger_cursor,
            event_type="condition_terminal",
            pair=pair,
            position=condition_ordinal,
            artifact_receipt_sha256=receipt["receipt_sha256"],
            private_key=executor_private_key,
        )
        condition_terminal_event_shas.append(terminal_event["logical_sha256"])
        condition_statements.append(statement)
        runner_values.append(result)
    if (
        any(row["shared_snapshot_sha256"] != snapshot_sha for row in runner_values)
        or runner_values[0]["candidate_count"] != CANDIDATE_COUNT
        or runner_values[1]["candidate_count"] != CANDIDATE_COUNT
        or runner_values[0]["candidate_registry_sha256"]
        != runner_values[1]["candidate_registry_sha256"]
        or runner_values[0]["ordered_candidate_sha256"]
        != runner_values[1]["ordered_candidate_sha256"]
        or runner_values[0]["candidate_legal"]
        != runner_values[1]["candidate_legal"]
        or runner_values[0]["continuation_proof_sha256"]
        != runner_values[1]["continuation_proof_sha256"]
        or runner_values[0]["continuation_contract"]
        != runner_values[1]["continuation_contract"]
        or runner_values[0]["continuation_policy_sha256"]
        != runner_values[1]["continuation_policy_sha256"]
    ):
        raise ExecutorV3Error(
            "paired conditions did not share one snapshot and four-candidate set"
        )
    pair_statement = pair_execution_statement(
        plan=plan,
        pair=pair,
        condition_records=condition_receipts,
        condition_statements=condition_statements,
        ledger_condition_terminal_event_sha256=condition_terminal_event_shas,
        dependency_rehash_sha256=dependency_rehash_sha256,
    )
    pair_receipt = sign_executor_receipt(
        receipt_format=PAIR_RECEIPT_FORMAT,
        receipt_status=PAIR_COMPLETE,
        statement=pair_statement,
        private_key=executor_private_key,
    )
    terminal_path = pair_root / "pair_terminal.json"
    immutable_json(terminal_path, pair_receipt)
    pair_event = append_ledger_event(
        claim=claim,
        plan=plan,
        dependency_rehash_sha256=dependency_rehash_sha256,
        cursor=ledger_cursor,
        event_type="pair_terminal",
        pair=pair,
        position=None,
        artifact_receipt_sha256=pair_receipt["receipt_sha256"],
        private_key=executor_private_key,
    )
    pair_root.chmod(0o555)
    return {
        "path": str(terminal_path),
        "file_sha256": hash_file(terminal_path),
        "logical_sha256": pair_receipt["receipt_sha256"],
        "ordinal": ordinal,
        "pair_id": pair["pair_id"],
        "condition_receipts": condition_receipts,
        "ledger_event_receipts": [
            *ledger_cursor["events"][-5:-1],
            pair_event,
        ],
    }


def _inventory_component_records(inventory: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    stack = inventory["execution_stack"]
    return {
        "simulator_implementation": dict(stack["simulator_implementation"]),
        "runtime_contract": dict(stack["runtime_contract"]),
        "collector_implementation": dict(stack["collector_implementation"]),
        "container_inventory": dict(stack["container_inventory"]),
        "result_evaluator_implementation": dict(
            inventory["result_evaluator"]["implementation"]
        ),
    }


def full_revalidate(plan: Mapping[str, Any]) -> dict[str, Any]:
    reject_inherited_cuda_mapping()
    python = _code_record(
        Path(str(plan["python"]["path"])),
        str(plan["python"]["file_sha256"]),
        "runtime Python",
    )
    supervisor = _code_record(
        Path(str(plan["supervisor"]["path"])),
        str(plan["supervisor"]["file_sha256"]),
        "executor supervisor",
    )
    actual_supervisor = Path(__file__).resolve(strict=True)
    if Path(supervisor["path"]) != actual_supervisor:
        raise ExecutorV3Error("actually executed supervisor path changed")
    protocol_record = _code_record(
        Path(str(plan["paired_protocol"]["path"])),
        str(plan["paired_protocol"]["file_sha256"]),
        "actually imported paired v3 protocol",
    )
    if Path(protocol_record["path"]) != Path(str(paired_v3.__file__)).resolve(strict=True):
        raise ExecutorV3Error("actually imported paired v3 protocol path changed")
    result_evaluator_record = _code_record(
        Path(str(plan["result_evaluator"]["path"])),
        str(plan["result_evaluator"]["file_sha256"]),
        "actually imported result evaluator v3",
    )
    if Path(result_evaluator_record["path"]) != Path(str(result_v3.__file__)).resolve(strict=True):
        raise ExecutorV3Error("actually imported result evaluator path changed")
    condition_runner = _code_record(
        Path(str(plan["condition_runner"]["path"])),
        str(plan["condition_runner"]["file_sha256"]),
        "condition runner",
    )
    runner_dependencies = plan.get("runner_dependencies")
    if not isinstance(runner_dependencies, Mapping) or set(runner_dependencies) != {
        "canonical_event_spec",
        "schema6_execution_authority",
        "dense_collector_implementation",
        "adapter_trainer_implementation",
        "selector_implementation",
        "local_dependency_closure",
    }:
        raise ExecutorV3Error("runner dependency closure changed")
    for role, record in runner_dependencies.items():
        if role == "local_dependency_closure":
            continue
        if not isinstance(record, Mapping) or set(record) != {"path", "file_sha256"}:
            raise ExecutorV3Error(f"runner dependency record changed: {role}")
        source = existing_file(Path(str(record["path"])), role)
        if not is_sha(record.get("file_sha256")) or hash_file(source) != record["file_sha256"]:
            raise ExecutorV3Error(f"runner dependency changed: {role}")
    validate_local_dependency_closure(
        runner_dependencies["local_dependency_closure"],
        expected_roots=[
            Path(str(condition_runner["path"])),
            Path(str(protocol_record["path"])),
            Path(str(plan["inventory_components"]["simulator_implementation"]["path"])),
            Path(str(runner_dependencies["dense_collector_implementation"]["path"])),
            Path(str(runner_dependencies["adapter_trainer_implementation"]["path"])),
            Path(str(runner_dependencies["selector_implementation"]["path"])),
        ],
    )
    authority = validate_authority_bundle(
        core_path=Path(str(plan["authority"]["core"]["path"])),
        core_file_sha256=str(plan["authority"]["core"]["file_sha256"]),
        decision_path=Path(str(plan["authority"]["decision"]["path"])),
        decision_file_sha256=str(plan["authority"]["decision"]["file_sha256"]),
        bundle_path=Path(str(plan["authority"]["bundle"]["path"])),
        bundle_file_sha256=str(plan["authority"]["bundle"]["file_sha256"]),
        inventory_path=Path(str(plan["authority"]["inventory"]["path"])),
        inventory_file_sha256=str(plan["authority"]["inventory"]["file_sha256"]),
        inventory_sha256=str(plan["authority"]["inventory"]["logical_sha256"]),
        supervisor=supervisor,
        condition_runner=condition_runner,
    )
    validate_schema6_full_horizon_binding(
        authority["runtime_contract"],
        runner_dependencies["schema6_execution_authority"],
    )
    if authority["records"] != plan["authority"]:
        raise ExecutorV3Error("authority records changed after preregistration")
    if authority["runtime_contract"] != plan["runtime_contract"]:
        raise ExecutorV3Error("runtime contract changed after preregistration")
    current_components = _inventory_component_records(authority["inventory"])
    if current_components != plan["inventory_components"]:
        raise ExecutorV3Error("execution component inventory changed")
    for role, record in current_components.items():
        source = existing_file(Path(record["path"]), role)
        if hash_file(source) != record["file_sha256"]:
            raise ExecutorV3Error(f"execution component changed: {role}")
    if current_components["result_evaluator_implementation"] != result_evaluator_record:
        raise ExecutorV3Error("result evaluator differs from reviewed execution inventory")
    _private, public_hex, private_file_sha = load_executor_signing_key(
        Path(str(plan["executor_signing_key"]["path"])),
        str(plan["executor_signing_key"]["file_sha256"]),
        expected_public_key_sha256=str(
            authority["core"]["authority_policy"]["executor_identity_sha256"]
        ),
    )
    if (
        public_hex != plan["executor_public_key_hex"]
        or private_file_sha != plan["executor_signing_key"]["file_sha256"]
        or hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()
        != plan["executor_identity_sha256"]
    ):
        raise ExecutorV3Error("executor signing identity changed")
    bootstrap = plan.get("bootstrap_draws")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "path",
        "file_sha256",
        "logical_sha256",
    }:
        raise ExecutorV3Error("bootstrap draw binding changed")
    bootstrap_path = existing_file(
        Path(str(bootstrap["path"])), "frozen bootstrap draws"
    )
    expected_bootstrap_logical = canonical_sha256(
        {
            "format": result_v3.BOOTSTRAP_FORMAT,
            "shape": list(result_v3.BOOTSTRAP_SHAPE),
            "seed": result_v3.BOOTSTRAP_SEED,
            "generator": result_v3.BOOTSTRAP_GENERATOR,
            "file_sha256": bootstrap["file_sha256"],
        }
    )
    if (
        hash_file(bootstrap_path) != bootstrap["file_sha256"]
        or bootstrap["logical_sha256"] != expected_bootstrap_logical
        or plan.get("bootstrap_frozen_before_first_condition_started") is not True
    ):
        raise ExecutorV3Error("bootstrap draw artifact changed")
    return authority


def make_pre_popen_guard(
    plan: Mapping[str, Any], command: Sequence[str]
) -> Callable[[], None]:
    def guard() -> None:
        authority = full_revalidate(plan)
        if authority["core"]["evaluation400"]["pair_count"] != PAIR_COUNT:
            raise ExecutorV3Error("pair count changed before Popen")
        current = gpu_audit(int(plan["gpu_index"]))
        if current["gpu_uuid"] != plan["gpu_uuid"]:
            raise ExecutorV3Error("GPU UUID changed immediately before Popen")
        positions = [
            index + 1
            for index, token in enumerate(command[:-1])
            if token == "--device"
        ]
        if positions != [len(command) - 1] or command[positions[0]] != "cuda:0":
            raise ExecutorV3Error("condition runner must use the only visible cuda:0")

    return guard


def preregister(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    reject_inherited_cuda_mapping()
    output = safe_path(args.output_root, "executor output root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.resolve(strict=True)
    ledger_root = _secure_directory(args.ledger_root, "WORM ledger root")
    python = _code_record(args.python, args.python_sha256, "runtime Python")
    supervisor = _code_record(
        Path(__file__), hash_file(Path(__file__).resolve()), "executor supervisor"
    )
    protocol_record = _code_record(
        Path(str(paired_v3.__file__)),
        hash_file(Path(str(paired_v3.__file__)).resolve()),
        "actually imported paired v3 protocol",
    )
    condition_runner = _code_record(
        args.condition_runner,
        args.condition_runner_sha256,
        "condition runner",
    )
    runner_dependencies = {
        "canonical_event_spec": _record(
            existing_file(args.canonical_event_spec, "canonical event spec")
        ),
        "schema6_execution_authority": _record(
            existing_file(
                args.schema6_execution_authority,
                "schema6 v2 execution authority",
            )
        ),
        "dense_collector_implementation": _code_record(
            args.dense_collector_implementation,
            args.dense_collector_implementation_sha256,
            "dense collector implementation",
        ),
        "adapter_trainer_implementation": _code_record(
            args.adapter_trainer_implementation,
            args.adapter_trainer_implementation_sha256,
            "adapter trainer inference implementation",
        ),
    }
    if (
        runner_dependencies["canonical_event_spec"]["file_sha256"]
        != args.canonical_event_spec_file_sha256
        or runner_dependencies["schema6_execution_authority"]["file_sha256"]
        != args.schema6_execution_authority_file_sha256
    ):
        raise ExecutorV3Error("runner JSON dependency SHA mismatch")
    authority = validate_authority_bundle(
        core_path=args.core,
        core_file_sha256=args.core_file_sha256,
        decision_path=args.decision,
        decision_file_sha256=args.decision_file_sha256,
        bundle_path=args.bundle,
        bundle_file_sha256=args.bundle_file_sha256,
        inventory_path=args.execution_inventory,
        inventory_file_sha256=args.execution_inventory_file_sha256,
        inventory_sha256=args.execution_inventory_sha256,
        supervisor=supervisor,
        condition_runner=condition_runner,
    )
    validate_schema6_full_horizon_binding(
        authority["runtime_contract"],
        runner_dependencies["schema6_execution_authority"],
    )
    result_evaluator_record = _code_record(
        Path(str(result_v3.__file__)),
        hash_file(Path(str(result_v3.__file__)).resolve(strict=True)),
        "actually imported result evaluator v3",
    )
    if (
        _inventory_component_records(authority["inventory"])[
            "result_evaluator_implementation"
        ]
        != result_evaluator_record
    ):
        raise ExecutorV3Error(
            "actually imported result evaluator is not the reviewed implementation"
        )
    selector_authority = authority["core"].get("deployment", {}).get(
        "selector_authority"
    )
    selector_implementation_raw = (
        selector_authority.get("implementation")
        if isinstance(selector_authority, Mapping)
        else None
    )
    if (
        not isinstance(selector_implementation_raw, Mapping)
        or set(selector_implementation_raw) != {"path", "file_sha256"}
    ):
        raise ExecutorV3Error(
            "production selector implementation is not frozen in paired core"
        )
    runner_dependencies["selector_implementation"] = _code_record(
        Path(str(selector_implementation_raw["path"])),
        str(selector_implementation_raw["file_sha256"]),
        "production selector implementation",
    )
    executor_identity = authority["core"]["authority_policy"][
        "executor_identity_sha256"
    ]
    executor_private_key, executor_public_hex, private_key_file_sha = (
        load_executor_signing_key(
            args.executor_signing_private_key,
            args.executor_signing_private_key_file_sha256,
            expected_public_key_sha256=executor_identity,
        )
    )
    if not isinstance(args.executor_key_id, str) or not args.executor_key_id:
        raise ExecutorV3Error("executor key ID is required")
    gpu_index = _strict_int(args.gpu_index, "GPU index")
    if not isinstance(args.gpu_uuid, str) or not args.gpu_uuid.startswith("GPU-"):
        raise ExecutorV3Error("explicit physical GPU UUID is required")
    output.mkdir(mode=0o700)
    for name in ("_supervisor", "pairs"):
        (output / name).mkdir(mode=0o700)
    bootstrap_path = output / "_supervisor" / "bootstrap_draws.uint16le"
    dependency_closure = build_local_dependency_closure(
        [
            Path(str(condition_runner["path"])),
            Path(str(protocol_record["path"])),
            Path(str(_inventory_component_records(authority["inventory"])[
                "simulator_implementation"
            ]["path"])),
            Path(str(runner_dependencies["dense_collector_implementation"]["path"])),
            Path(str(runner_dependencies["adapter_trainer_implementation"]["path"])),
            Path(str(runner_dependencies["selector_implementation"]["path"])),
        ],
        scripts_root=Path(str(condition_runner["path"])).parent,
    )
    dependency_closure_path = (
        output / "_supervisor" / "local_dependency_closure.json"
    )
    immutable_json(dependency_closure_path, dependency_closure)
    runner_dependencies["local_dependency_closure"] = _record(
        dependency_closure_path, dependency_closure["closure_sha256"]
    )
    bootstrap_payload = result_v3.bootstrap_draw_bytes()
    immutable_bytes(bootstrap_path, bootstrap_payload)
    bootstrap_file_sha = hash_file(bootstrap_path)
    bootstrap_logical = canonical_sha256(
        {
            "format": result_v3.BOOTSTRAP_FORMAT,
            "shape": list(result_v3.BOOTSTRAP_SHAPE),
            "seed": result_v3.BOOTSTRAP_SEED,
            "generator": result_v3.BOOTSTRAP_GENERATOR,
            "file_sha256": bootstrap_file_sha,
        }
    )
    identity = lane_identity(authority)
    execution_nonce_hex = secrets.token_hex(32)
    ledger_id_sha256 = canonical_sha256(
        {
            "format": result_v3.LEDGER_FORMAT,
            "lane_identity_sha256": identity,
            "execution_nonce_hex": execution_nonce_hex,
            "protocol_core_sha256": authority["records"]["core"]["logical_sha256"],
            "decision_sha256": authority["records"]["decision"]["logical_sha256"],
            "bundle_sha256": authority["records"]["bundle"]["logical_sha256"],
            "pair_identity_set_sha256": authority["core"]["evaluation400"][
                "pair_identity_set_sha256"
            ],
        }
    )
    plan_base = {
        "format": PLAN_FORMAT,
        "status": "preregistered_authorized_before_pair_zero",
        "output_root": str(output),
        "ledger_root": str(ledger_root),
        "lane_identity_sha256": identity,
        "authority": authority["records"],
        "runtime_contract": authority["runtime_contract"],
        "inventory_components": _inventory_component_records(authority["inventory"]),
        "python": python,
        "supervisor": supervisor,
        "paired_protocol": protocol_record,
        "result_evaluator": result_evaluator_record,
        "condition_runner": condition_runner,
        "runner_dependencies": runner_dependencies,
        "executor_signing_key": {
            "path": str(
                existing_file(
                    args.executor_signing_private_key,
                    "executor Ed25519 private key",
                )
            ),
            "file_sha256": private_key_file_sha,
        },
        "executor_key_id": args.executor_key_id,
        "executor_public_key_hex": executor_public_hex,
        "executor_identity_sha256": executor_identity,
        "execution_nonce_hex": execution_nonce_hex,
        "ledger_id_sha256": ledger_id_sha256,
        "bootstrap_draws": {
            "path": str(bootstrap_path),
            "file_sha256": bootstrap_file_sha,
            "logical_sha256": bootstrap_logical,
        },
        "bootstrap_frozen_before_first_condition_started": True,
        "pair_identity_set_sha256": authority["core"]["evaluation400"][
            "pair_identity_set_sha256"
        ],
        "pair_count": PAIR_COUNT,
        "attempt": 0,
        "candidate_count": CANDIDATE_COUNT,
        "deployment_binding_sha256": authority["core"]["deployment"][
            "deployment_binding_sha256"
        ],
        "policy_runtime_action_binding_sha256": authority["core"]["deployment"][
            "policy_runtime_action_binding_sha256"
        ],
        "gpu_index": gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "gpu_lock_path": str(safe_path(args.gpu_lock, "GPU lock")),
        "result_evaluator_executed": False,
        "result_analysis_performed": False,
        "additional_reserve400_count": 0,
        "create_once": True,
    }
    plan = {**plan_base, "plan_sha256": canonical_sha256(plan_base)}
    plan_path = output / "_supervisor" / "static_plan.json"
    immutable_json(plan_path, plan)
    claim = acquire_lane_claim(
        ledger_root=ledger_root,
        identity=identity,
        plan=plan,
        private_key=executor_private_key,
    )
    atomic_json(
        output / "_supervisor" / "state.json",
        {
            "format": STATE_FORMAT,
            "status": "preregistered_claim_consumed_before_pair_zero",
            "plan_sha256": plan["plan_sha256"],
            "claim_sha256": claim["logical_sha256"],
            "pairs_started": 0,
            "conditions_started": 0,
            "result_analysis_performed": False,
        },
    )
    return output, plan, claim


def load_plan(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_path, plan, _file_sha = read_json(path, "executor static plan")
    logical = verify_signed(plan, "plan_sha256", "executor static plan")
    root = safe_path(plan.get("output_root", ""), "executor output root", must_exist=True)
    if (
        plan.get("format") != PLAN_FORMAT
        or plan_path != root / "_supervisor" / "static_plan.json"
        or logical != plan.get("plan_sha256")
        or type(plan.get("pair_count")) is not int
        or plan["pair_count"] != PAIR_COUNT
        or type(plan.get("attempt")) is not int
        or plan["attempt"] != 0
        or type(plan.get("candidate_count")) is not int
        or plan["candidate_count"] != CANDIDATE_COUNT
        or type(plan.get("additional_reserve400_count")) is not int
        or plan["additional_reserve400_count"] != 0
        or plan.get("result_evaluator_executed") is not False
        or plan.get("result_analysis_performed") is not False
        or not is_sha(plan.get("ledger_id_sha256"))
    ):
        raise ExecutorV3Error("executor static plan changed")
    authority = full_revalidate(plan)
    claim = validate_lane_claim(plan)
    return root, plan, authority, claim


def _create_hidden(path: Path, payload: bytes) -> tuple[int, int]:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o000,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise OSError("short hidden terminal write")
            remaining = remaining[count:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def freeze_tree(root: Path, hidden: frozenset[Path] = frozenset()) -> None:
    hidden_resolved = {path.resolve(strict=False) for path in hidden}
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise ExecutorV3Error("executor output contains a symlink")
            if item.resolve(strict=False) not in hidden_resolved:
                item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise ExecutorV3Error("executor output contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def publish_terminal(
    root: Path, receipt: Mapping[str, Any], *, success: bool
) -> None:
    terminal_path = root / ("final_receipt.json" if success else "failure_receipt.json")
    exit_path = root / "run.exit"
    opposite = root / ("failure_receipt.json" if success else "final_receipt.json")
    if any(path.exists() or path.is_symlink() for path in (terminal_path, exit_path, opposite)):
        raise FileExistsError("executor terminal already exists")
    payload = json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    exit_payload = b"0\n" if success else b"1\n"
    created: dict[Path, tuple[int, int]] = {}
    published = False
    try:
        created[terminal_path] = _create_hidden(terminal_path, payload)
        created[exit_path] = _create_hidden(exit_path, exit_payload)
        freeze_tree(root, frozenset({terminal_path, exit_path}))
        exit_path.chmod(0o444)
        _fsync_directory(root)
        terminal_path.chmod(0o444)
        _fsync_directory(root)
        published = True
    finally:
        if not published:
            try:
                root.chmod(0o700)
            except OSError:
                pass
            for path, identity in created.items():
                try:
                    metadata = path.lstat()
                    if (metadata.st_dev, metadata.st_ino) == identity:
                        path.unlink()
                except OSError:
                    pass


def publish_worm_terminal(
    lane_root: Path, name: str, receipt: Mapping[str, Any]
) -> Path:
    """Freeze the WORM lane, then expose its only terminal as the last inode."""

    terminal = lane_root / name
    if terminal.exists() or terminal.is_symlink():
        raise FileExistsError(terminal)
    payload = json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    identity = _create_hidden(terminal, payload)
    published = False
    try:
        freeze_tree(lane_root, frozenset({terminal}))
        terminal.chmod(0o444)
        _fsync_directory(lane_root)
        published = True
        return terminal
    finally:
        if not published:
            try:
                lane_root.chmod(0o700)
            except OSError:
                pass
            try:
                metadata = terminal.lstat()
                if (metadata.st_dev, metadata.st_ino) == identity:
                    terminal.unlink()
            except OSError:
                pass


def validate_completed_pair_records(
    records: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    dependency_rehash_sha256: str,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    if len(records) != PAIR_COUNT:
        raise ExecutorV3Error("execution terminal does not contain 400 pairs")
    all_conditions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_artifact_paths: set[str] = set()
    for ordinal, (record, pair) in enumerate(zip(records, pairs, strict=True)):
        if not isinstance(record, Mapping) or set(record) != {
            "ordinal", "pair_id", "path", "file_sha256", "logical_sha256",
            "condition_receipts", "ledger_event_receipts",
        }:
            raise ExecutorV3Error("pair terminal record changed")
        condition_records = record["condition_receipts"]
        if not isinstance(condition_records, list) or len(condition_records) != 2:
            raise ExecutorV3Error("pair condition descriptors changed")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(str(plan["executor_public_key_hex"]))
            )
        except (ImportError, ModuleNotFoundError, ValueError) as error:
            raise ExecutorV3Error("executor public key is invalid") from error
        conditions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for position, condition_record in enumerate(condition_records):
            exact = {
                "global_condition_ordinal", "pair_ordinal", "condition_position",
                "condition_id", "path", "file_sha256", "logical_sha256",
            }
            if not isinstance(condition_record, Mapping) or set(condition_record) != exact:
                raise ExecutorV3Error("condition terminal descriptor changed")
            _condition_path, condition_receipt, condition_file = read_json(
                Path(str(condition_record["path"])),
                f"condition {2 * ordinal + position} receipt",
            )
            condition_statement, condition_logical = result_v3._verify_executor_receipt(
                condition_receipt,
                expected_format=CONDITION_RECEIPT_FORMAT,
                expected_status=CONDITION_COMPLETE,
                public_key=public_key,
                role=f"condition {2 * ordinal + position} receipt",
            )
            if (
                condition_file != condition_record["file_sha256"]
                or condition_logical != condition_record["logical_sha256"]
            ):
                raise ExecutorV3Error("condition terminal descriptor SHA changed")
            artifact_paths = result_v3._validate_condition_statement(
                condition_statement,
                pair=pair,
                position=position,
                core=authority["core"],
                decision=authority["decision"],
                bundle=authority["bundle"],
                execution_nonce_hex=str(plan["execution_nonce_hex"]),
                dependency_rehash_sha256=dependency_rehash_sha256,
            )
            if seen_artifact_paths & artifact_paths:
                raise ExecutorV3Error("condition execution-artifact path reused")
            seen_artifact_paths.update(artifact_paths)
            conditions.append((dict(condition_record), condition_statement))
            all_conditions.append((dict(condition_record), condition_statement))
        _path, value, file_sha = read_json(
            Path(str(record["path"])), f"pair {ordinal} terminal"
        )
        statement, logical = result_v3._verify_executor_receipt(
            value,
            expected_format=PAIR_RECEIPT_FORMAT,
            expected_status=PAIR_COMPLETE,
            public_key=public_key,
            role=f"pair {ordinal} terminal",
        )
        if (
            file_sha != record["file_sha256"]
            or logical != record["logical_sha256"]
            or record["ordinal"] != ordinal
            or record["pair_id"] != pair["pair_id"]
        ):
            raise ExecutorV3Error("pair terminal closure changed")
        result_v3._validate_pair_statement(
            statement,
            pair=pair,
            conditions=conditions,
            core=authority["core"],
            decision=authority["decision"],
            bundle=authority["bundle"],
            execution_nonce_hex=str(plan["execution_nonce_hex"]),
            dependency_rehash_sha256=dependency_rehash_sha256,
        )
        all_pairs.append(
            (
                {
                    key: record[key]
                    for key in ("ordinal", "pair_id", "path", "file_sha256", "logical_sha256")
                },
                statement,
            )
        )
    return all_conditions, all_pairs


def freeze_preexecution_dependency_rehash(
    *, root: Path, plan: Mapping[str, Any], claim: Mapping[str, Any]
) -> dict[str, Any]:
    # This second full reconstruction occurs after the one-shot claim and before
    # pair 0/condition 0 receives a write-ahead start event.
    full_revalidate(plan)
    path = root / "_supervisor" / "preexecution_dependency_rehash.json"
    base = {
        "status": "full_rehash_after_claim_before_first_condition",
        "plan_sha256": plan["plan_sha256"],
        "claim_sha256": claim["logical_sha256"],
        "authority": dict(plan["authority"]),
        "python": dict(plan["python"]),
        "supervisor": dict(plan["supervisor"]),
        "paired_protocol": dict(plan["paired_protocol"]),
        "result_evaluator": dict(plan["result_evaluator"]),
        "condition_runner": dict(plan["condition_runner"]),
        "runner_dependencies": dict(plan["runner_dependencies"]),
        "runtime_contract": dict(plan["runtime_contract"]),
        "inventory_components": dict(plan["inventory_components"]),
        "executor_identity_sha256": plan["executor_identity_sha256"],
        "bootstrap_draws": dict(plan["bootstrap_draws"]),
        "rehash_unix_ns": time.time_ns(),
    }
    value = {**base, "dependency_rehash_sha256": canonical_sha256(base)}
    if path.exists() or path.is_symlink():
        _bound, persisted, file_sha = read_json(
            path, "preexecution dependency rehash"
        )
        logical = verify_signed(
            persisted,
            "dependency_rehash_sha256",
            "preexecution dependency rehash",
        )
        if persisted != value:
            # Timestamps make a newly reconstructed value intentionally differ;
            # an existing valid record is accepted only after all dependencies
            # above were revalidated and its immutable bindings match the plan.
            comparable = dict(persisted)
            comparable.pop("rehash_unix_ns", None)
            current = dict(value)
            current.pop("rehash_unix_ns", None)
            comparable.pop("dependency_rehash_sha256", None)
            current.pop("dependency_rehash_sha256", None)
            if comparable != current:
                raise ExecutorV3Error("preexecution dependency rehash changed")
        return {
            "path": str(path),
            "file_sha256": file_sha,
            "logical_sha256": logical,
        }
    immutable_json(path, value)
    return _record(path, value["dependency_rehash_sha256"])


def execute(
    plan_path: Path,
    *,
    idle_interval: float,
    hold_unproven: bool = True,
) -> dict[str, Any]:
    root, plan, authority, claim = load_plan(plan_path)
    pairs = authority["pairs"]
    dependency_rehash = freeze_preexecution_dependency_rehash(
        root=root, plan=plan, claim=claim
    )
    executor_private_key, _public_hex, _private_file_sha = (
        load_executor_signing_key(
            Path(str(plan["executor_signing_key"]["path"])),
            str(plan["executor_signing_key"]["file_sha256"]),
            expected_public_key_sha256=str(plan["executor_identity_sha256"]),
        )
    )
    completed = scan_completed_pairs(claim=claim, pairs=pairs, plan=plan)
    existing_events = [
        dict(event)
        for record in completed
        for event in record["ledger_event_receipts"]
    ]
    ledger_cursor: dict[str, Any] = {
        "event_index": len(existing_events),
        "previous_entry_sha256": (
            existing_events[-1]["logical_sha256"]
            if existing_events
            else claim["logical_sha256"]
        ),
        "events": existing_events,
    }
    lock = open_gpu_lock(Path(str(plan["gpu_lock_path"])))
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        idle_before = wait_two_idle(
            int(plan["gpu_index"]),
            str(plan["gpu_uuid"]),
            interval=idle_interval,
        )
        immutable_json(root / "_supervisor" / "gpu_idle_before.json", idle_before)
        for pair in pairs[len(completed) :]:
            atomic_json(
                root / "_supervisor" / "state.json",
                {
                    "format": STATE_FORMAT,
                    "status": "executing_exact_next_pair",
                    "plan_sha256": plan["plan_sha256"],
                    "claim_sha256": claim["logical_sha256"],
                    "next_ordinal": pair["ordinal"],
                    "attempt": 0,
                    "pairs_complete": len(completed),
                    "result_analysis_performed": False,
                },
            )
            record = execute_pair(
                root=root,
                plan=plan,
                claim=claim,
                pair=pair,
                physical_gpu=idle_before,
                core=authority["core"],
                executor_private_key=executor_private_key,
                dependency_rehash_sha256=dependency_rehash[
                    "logical_sha256"
                ],
                ledger_cursor=ledger_cursor,
                pre_popen_guard_factory=lambda command: make_pre_popen_guard(
                    plan, command
                ),
            )
            completed.append(record)
        condition_items, pair_items = validate_completed_pair_records(
            completed,
            pairs,
            plan=plan,
            authority=authority,
            dependency_rehash_sha256=dependency_rehash["logical_sha256"],
        )
        idle_after = wait_two_idle(
            int(plan["gpu_index"]),
            str(plan["gpu_uuid"]),
            interval=idle_interval,
        )
        if any(
            idle_after[field] != idle_before[field]
            for field in ("gpu_index", "gpu_name", "gpu_uuid")
        ):
            raise ExecutorV3Error("GPU identity changed across evaluation400")
        immutable_json(root / "_supervisor" / "gpu_idle_after.json", idle_after)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        release_base = {
            "status": "released_after_all_condition_groups_reaped_and_gpu_idle",
            "gpu_uuid": plan["gpu_uuid"],
            "lock_path": plan["gpu_lock_path"],
            "all_condition_process_groups_reaped": True,
            "idle_before_sha256": idle_before["audit_sha256"],
            "idle_after_sha256": idle_after["audit_sha256"],
            "released_unix_ns": time.time_ns(),
        }
        release = {
            **release_base,
            "release_sha256": canonical_sha256(release_base),
        }
        immutable_json(root / "_supervisor" / "gpu_lock_release.json", release)
        lock.close()
        lock = None
        pair_descriptors = [
            {
                key: row[key]
                for key in ("ordinal", "pair_id", "path", "file_sha256", "logical_sha256")
            }
            for row in completed
        ]
        condition_descriptors = [
            dict(condition)
            for row in completed
            for condition in row["condition_receipts"]
        ]
        final_event = append_ledger_event(
            claim=claim,
            plan=plan,
            dependency_rehash_sha256=dependency_rehash["logical_sha256"],
            cursor=ledger_cursor,
            event_type="execution_terminal",
            pair=None,
            position=None,
            artifact_receipt_sha256=None,
            private_key=executor_private_key,
        )
        ledger = {
            "format": result_v3.LEDGER_FORMAT,
            "terminal_state": result_v3.LEDGER_FINAL_STATE,
            "ledger_id_sha256": plan["ledger_id_sha256"],
            "claim_receipt_sha256": claim["logical_sha256"],
            "final_event_sha256": final_event["logical_sha256"],
            "event_count": result_v3.LEDGER_EVENT_COUNT,
            "claim_count": 1,
            "claim_release_count": 0,
            "execution_attempt_count": 1,
            "condition_attempt_count": 2 * PAIR_COUNT,
            "retry_count": 0,
            "selective_rerun_count": 0,
            "pair_exclusion_count": 0,
            "condition_exclusion_count": 0,
            "incomplete_pair_count": 0,
            "incomplete_condition_count": 0,
            "complete_pair_count": PAIR_COUNT,
            "complete_condition_count": 2 * PAIR_COUNT,
            "claim_before_outcome_read": True,
            "one_shot_consumed": True,
        }
        terminal_statement = {
            "protocol_core": dict(plan["authority"]["core"]),
            "ed25519_decision": dict(plan["authority"]["decision"]),
            "execution_bundle": dict(plan["authority"]["bundle"]),
            "executor_key_id": plan["executor_key_id"],
            "executor_public_key_hex": plan["executor_public_key_hex"],
            "executor_public_key_sha256": plan["executor_identity_sha256"],
            "executor_identity_sha256": plan["executor_identity_sha256"],
            "execution_nonce_hex": plan["execution_nonce_hex"],
            "pair_identity_set_sha256": plan["pair_identity_set_sha256"],
            "deployment_binding_sha256": plan["deployment_binding_sha256"],
            "policy_runtime_action_binding_sha256": plan[
                "policy_runtime_action_binding_sha256"
            ],
            "preexecution_dependency_rehash_sha256": dependency_rehash[
                "logical_sha256"
            ],
            "full_dependency_rehash_after_claim_before_first_condition_started": True,
            "bootstrap_draws": dict(plan["bootstrap_draws"]),
            "bootstrap_format": result_v3.BOOTSTRAP_FORMAT,
            "bootstrap_shape": list(result_v3.BOOTSTRAP_SHAPE),
            "bootstrap_seed": result_v3.BOOTSTRAP_SEED,
            "bootstrap_generator": result_v3.BOOTSTRAP_GENERATOR,
            "bootstrap_frozen_before_first_condition_started": True,
            "execution_claim": {
                key: claim[key]
                for key in ("path", "file_sha256", "logical_sha256")
            },
            "ledger_contract": ledger,
            "ledger_events": list(ledger_cursor["events"]),
            "pair_receipts": pair_descriptors,
            "condition_receipts": condition_descriptors,
            "execution_complete": True,
            "subset_statistics_authorized": False,
            "performance_claim_authorized_by_executor": False,
        }
        if set(terminal_statement) != result_v3.TERMINAL_STATEMENT_FIELDS:
            raise ExecutorV3Error(
                "execution terminal statement differs from result evaluator v3"
            )
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(str(plan["executor_public_key_hex"]))
            )
            result_v3._validate_terminal_statement(
                terminal_statement,
                core_record=plan["authority"]["core"],
                decision_record=plan["authority"]["decision"],
                bundle_record=plan["authority"]["bundle"],
                core=authority["core"],
                bootstrap_path=Path(str(plan["bootstrap_draws"]["path"])),
                bootstrap_file_sha256=str(plan["bootstrap_draws"]["file_sha256"]),
            )
            result_v3.validate_execution_ledger(
                claim_record=terminal_statement["execution_claim"],
                event_records=terminal_statement["ledger_events"],
                condition_items=condition_items,
                pair_items=pair_items,
                ledger_contract=ledger,
                core=authority["core"],
                decision=authority["decision"],
                bundle=authority["bundle"],
                execution_nonce_hex=str(plan["execution_nonce_hex"]),
                dependency_rehash_sha256=dependency_rehash["logical_sha256"],
                public_key=public_key,
            )
        except Exception as error:
            raise ExecutorV3Error(
                "result evaluator rejected execution receipt/ledger closure"
            ) from error
        lane_terminal = sign_executor_receipt(
            receipt_format=EXECUTION_RECEIPT_FORMAT,
            receipt_status=EXECUTION_COMPLETE,
            statement=terminal_statement,
            private_key=executor_private_key,
        )
        publish_worm_terminal(
            Path(claim["lane_root"]), "execution_terminal.json", lane_terminal
        )
        publish_terminal(root, lane_terminal, success=True)
        return lane_terminal
    except UnprovenProcessGroup:
        _HELD_UNPROVEN_LOCKS.append(lock)
        atomic_json(
            root / "_supervisor" / "state.json",
            {
                "format": STATE_FORMAT,
                "status": "incomplete_unproven_condition_group_gpu_lock_retained",
                "plan_sha256": plan["plan_sha256"],
                "gpu_lock_retained": True,
                "artifacts_frozen_read_only": False,
                "rerun_authorized": False,
            },
        )
        if hold_unproven:
            while True:
                time.sleep(60)
        raise
    except BaseException as error:
        if lock is not None:
            try:
                failure_idle = wait_two_idle(
                    int(plan["gpu_index"]),
                    str(plan["gpu_uuid"]),
                    interval=idle_interval,
                )
                immutable_json(
                    root / "_supervisor" / "gpu_idle_before_failure_release.json",
                    failure_idle,
                )
            except BaseException as idle_error:
                _HELD_UNPROVEN_LOCKS.append(lock)
                atomic_json(
                    root / "_supervisor" / "state.json",
                    {
                        "format": STATE_FORMAT,
                        "status": "failure_gpu_idle_unproven_lock_retained",
                        "plan_sha256": plan["plan_sha256"],
                        "gpu_lock_retained": True,
                        "artifacts_frozen_read_only": False,
                        "rerun_authorized": False,
                    },
                )
                if hold_unproven:
                    while True:
                        time.sleep(60)
                raise UnprovenProcessGroup(
                    "failure path cannot prove same-UUID GPU idle"
                ) from idle_error
            else:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                finally:
                    lock.close()
                lock = None
        failure_base = {
            "format": EXECUTION_RECEIPT_FORMAT,
            "status": EXECUTION_FAILED,
            "plan_sha256": plan["plan_sha256"],
            "claim_sha256": claim["logical_sha256"],
            "error_type": type(error).__name__,
            "pair_receipts_complete": completed,
            "rerun_authorized": False,
            "additional_reserve400_count": 0,
            "result_evaluator_executed": False,
            "result_analysis_performed": False,
            "gpu_lock_retained": False,
            "artifacts_frozen_read_only": True,
        }
        failure = {
            **failure_base,
            "execution_receipt_sha256": canonical_sha256(failure_base),
        }
        lane_failure = Path(claim["lane_root"]) / "execution_failure.json"
        if not lane_failure.exists():
            publish_worm_terminal(
                Path(claim["lane_root"]), "execution_failure.json", failure
            )
        publish_terminal(root, failure, success=False)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("preregister")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--ledger-root", type=Path, required=True)
    for name in ("core", "decision", "bundle"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
        prepare.add_argument(f"--{name}-file-sha256", required=True)
    prepare.add_argument("--execution-inventory", type=Path, required=True)
    prepare.add_argument("--execution-inventory-file-sha256", required=True)
    prepare.add_argument("--execution-inventory-sha256", required=True)
    prepare.add_argument("--python", type=Path, required=True)
    prepare.add_argument("--python-sha256", required=True)
    prepare.add_argument("--condition-runner", type=Path, required=True)
    prepare.add_argument("--condition-runner-sha256", required=True)
    prepare.add_argument("--canonical-event-spec", type=Path, required=True)
    prepare.add_argument("--canonical-event-spec-file-sha256", required=True)
    prepare.add_argument("--schema6-execution-authority", type=Path, required=True)
    prepare.add_argument("--schema6-execution-authority-file-sha256", required=True)
    prepare.add_argument("--dense-collector-implementation", type=Path, required=True)
    prepare.add_argument("--dense-collector-implementation-sha256", required=True)
    prepare.add_argument("--adapter-trainer-implementation", type=Path, required=True)
    prepare.add_argument("--adapter-trainer-implementation-sha256", required=True)
    prepare.add_argument("--executor-signing-private-key", type=Path, required=True)
    prepare.add_argument(
        "--executor-signing-private-key-file-sha256", required=True
    )
    prepare.add_argument("--executor-key-id", required=True)
    prepare.add_argument("--gpu-index", type=int, default=0)
    prepare.add_argument("--gpu-uuid", required=True)
    prepare.add_argument("--gpu-lock", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--idle-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preregister":
        root, plan, claim = preregister(args)
        print(
            json.dumps(
                {
                    "output_root": str(root),
                    "plan_sha256": plan["plan_sha256"],
                    "claim_sha256": claim["logical_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.idle_interval <= 0:
        raise ExecutorV3Error("idle interval must be positive")
    final = execute(args.plan, idle_interval=args.idle_interval)
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_COUNT",
    "CLAIM_FORMAT",
    "CONDITION_REQUEST_FORMAT",
    "CONDITION_RECEIPT_FORMAT",
    "EXECUTION_RECEIPT_FORMAT",
    "ExecutorV3Error",
    "FORMAT",
    "IncompleteLane",
    "PAIR_COUNT",
    "PAIR_RECEIPT_FORMAT",
    "PLAN_FORMAT",
    "RUNNER_RESULT_FORMAT",
    "RUNTIME_CONTRACT_FORMAT",
    "UnprovenProcessGroup",
    "acquire_lane_claim",
    "canonical_sha256",
    "condition_request",
    "execute_pair",
    "lane_identity",
    "load_plan",
    "preregister",
    "run_condition_stage",
    "scan_completed_pairs",
    "shared_snapshot_sha256",
    "validate_authority_bundle",
    "validate_pair_rows",
    "validate_runner_result",
]
