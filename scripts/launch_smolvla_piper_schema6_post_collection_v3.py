#!/usr/bin/env python3
"""Strict development300 post-collection launcher for Schema6.

The launcher is a metadata-only watcher until the exact development300
terminal has been authenticated.  It materializes the 80/30/190 profile,
trains one Piper adapter for each of the five native r7h source members, opens
formal190 only in the independent evaluator after all five adapters are frozen,
calibrates the six structured heads, and publishes a content-addressed input
handoff for the pre-outcome evaluation400 identity bridge v2.

It never discovers or opens fresh/confirmation/evaluation400 trajectories or
labels, never creates a replacement reserve400, and never accepts a LOBO or
aggregate checkpoint as an adapter-training source.
"""

from __future__ import annotations

import argparse
import ast
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

import calibrate_smolvla_piper_adapter_ensemble as calibrator
import evaluate_smolvla_piper_schema6_target_validation50_ensemble as evaluator
import launch_smolvla_piper_schema6_autonomous_watcher as r9b_watcher
import materialize_smolvla_piper_schema6_training_manifest_v3 as materializer
import train_smolvla_piper_schema6_embodiment_adapter as trainer
import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as identity_bridge
from launch_smolvla_piper_schema6_autonomous_watcher import (
    DESIGNATED_LOBO_ROOT,
    EXPECTED_LOBO_LAUNCHER_SHA256,
    EXPECTED_LOBO_STATIC_PLAN_SHA256,
    EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256,
    EXPECTED_SOURCE_LAUNCHER_SHA256,
    EXPECTED_SOURCE_PLAN_SHA256,
    EXPECTED_SOURCE_ROOT,
    EXPECTED_SOURCE_STATIC_PLAN_SHA256,
    SOURCE_MEMBER_SEEDS,
    TERMINAL_PENDING_STATUS as R9B_FROZEN_STATE_STATUS,
    TERMINAL_STATUS as R9B_TERMINAL_STATUS,
    validate_lobo_terminal_summary,
    validate_schema6_success_terminal_receipt,
    validate_source_training_summary,
)
from run_smolvla_piper_schema6_development300_collection import (
    TERMINAL_FAILURE as DEVELOPMENT300_FAILURE,
    TERMINAL_RECEIPT_FORMAT as DEVELOPMENT300_TERMINAL_FORMAT,
    TERMINAL_SUCCESS as DEVELOPMENT300_SUCCESS,
)


FORMAT = "etsf_smolvla_piper_schema6_post_collection_launcher_v3"
PLAN_FORMAT = "etsf_smolvla_piper_schema6_post_collection_plan_v3"
STATE_FORMAT = "etsf_smolvla_piper_schema6_post_collection_state_v3"
DETACH_FORMAT = "etsf_smolvla_piper_schema6_post_collection_detach_v3"
STAGE_FORMAT = "etsf_smolvla_piper_schema6_post_collection_stage_v3"
MEMBER_FORMAT = "etsf_smolvla_piper_schema6_adapter_member_receipt_v3"
HANDOFF_FORMAT = (
    "etsf_smolvla_piper_evaluation400_identity_bridge_v2_input_handoff_v1"
)
TERMINAL_STATUS = (
    "complete_development300_five_member_formal190_calibration_handoff_frozen"
)
FAILURE_STATUS = "failed_closed_schema6_post_collection_v3"
PENDING_TERMINAL_STATUS = "terminal_success_pending_frozen_publication"
FORMAL190_CLAIM_FORMAT = "etsf_schema6_formal190_global_one_shot_claim_v1"
FORMAL190_CLAIM_STATUS = "consumed_before_first_formal190_authority_publication"
MATERIALIZER_SHA256 = (
    "8a2b4bd4cff0d534e16fe9a97c0c5d52f70f387ea39334e8e4c5bbde8dfa2455"
)
R9B_WATCHER_SHA256 = (
    "40916a07ddcd98706ac90eecf5ff86709cc73123f3f58a77d3e8ddb341887428"
)
SPLIT_PROFILE = "development300_v3"
TRAIN_GROUPS = 80
INTERNAL_GROUPS = 30
FORMAL_GROUPS = 190
MEMBER_COUNT = 5
EXPECTED_GPU_FRAGMENT = "RTX 4090"
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
FORBIDDEN_INPUT_COMPONENTS = frozenset(
    {"fresh", "confirmation", "evaluation400", "reserve400", "paired400"}
)
MATERIALIZER_OUTPUTS = dict(materializer.OUTPUT_NAMES)
BRIDGE_EXTERNAL_DEPENDENCIES = (
    "target_manifest",
    "selected_identity_attestation",
    "policy_bridge_receipt",
)
BRIDGE_PRODUCED_DEPENDENCIES = (
    "ensemble_manifest",
    "calibration",
    "head_support",
    "calibration_receipt",
)
MEMBER_FIELDS = frozenset(
    {
        "format",
        "status",
        "member_index",
        "member_seed",
        "split_profile",
        "split_profile_version",
        "required_trainer_group_counts",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_checkpoint_role",
        "training_manifest_sha256",
        "split_sha256",
        "source_ensemble_contract_sha256",
        "summary_path",
        "summary_file_sha256",
        "summary_sha256",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "validation_predictions_path",
        "validation_predictions_file_sha256",
        "validation_predictions_logical_sha256",
        "validation_labels_path",
        "validation_labels_file_sha256",
        "validation_labels_logical_sha256",
        "validation_identity_set_sha256",
        "validation_lane",
        "internal_validation_group_count",
        "sealed_formal_target_validation_group_count",
        "prediction_contract",
        "source_rank_score_contract",
        "source_rank_score_contract_sha256",
        "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen",
        "formal_target_validation_labels_opened_before_five_adapters_frozen",
        "formal_target_validation_release_condition",
        "lobo_or_aggregate_checkpoint_used",
        "stage_result_sha256",
        "receipt_sha256",
    }
)
_HELD_UNPROVEN_LOCKS: list[Any] = []

FD_IMPORT_BOOTSTRAP = r"""
import importlib.abc, importlib.util, json, runpy, sys
target_fd_path, target_source_path, encoded = sys.argv[1:4]
module_rows = json.loads(encoded)
class _BoundFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        row = module_rows.get(fullname)
        return None if row is None else importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        row = module_rows[module.__name__]
        with open(row["fd_path"], "rb") as handle:
            payload = handle.read()
        module.__file__ = row["source_path"]
        exec(compile(payload, row["source_path"], "exec"), module.__dict__)
sys.meta_path.insert(0, _BoundFinder())
sys.argv = [target_source_path] + sys.argv[4:]
runpy.run_path(target_fd_path, run_name="__main__")
""".strip()


class PostCollectionV3Error(RuntimeError):
    """A lineage, process-lifecycle, or capability invariant failed closed."""


class UnprovenProcessGroup(PostCollectionV3Error):
    """A Popen attempt cannot be proved fully reaped; the GPU lock must stay held."""


class Formal190ClaimConsumed(PostCollectionV3Error):
    """The global one-shot was consumed, but this output may not open formal190."""

    def __init__(self, message: str, claim: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.claim = dict(claim)


def retain_unproven_gpu_lock(lock_handle: Any) -> None:
    if lock_handle is None or getattr(lock_handle, "closed", True):
        raise PostCollectionV3Error("cannot retain a missing GPU lock")
    _HELD_UNPROVEN_LOCKS.append(lock_handle)


def open_shared_gpu_lock(path: Path) -> Any:
    lock_path = safe_path(path, "shared GPU lock", input_scope=False)
    parent = safe_path(
        lock_path.parent,
        "shared GPU lock parent",
        input_scope=False,
        must_exist=True,
    )
    if not parent.is_dir():
        raise PostCollectionV3Error("shared GPU lock parent is not a directory")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise PostCollectionV3Error("cannot safely open shared GPU lock") from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        os.close(descriptor)
        raise PostCollectionV3Error("shared GPU lock identity is unsafe")
    return os.fdopen(descriptor, "r+", encoding="ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _require_int(value: Any, role: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PostCollectionV3Error(f"{role} must be an integer >= {minimum}")
    return value


def _is_exact_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def _forbidden_namespace(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if any(token in lowered for token in FORBIDDEN_INPUT_COMPONENTS):
            return True
    return False


def safe_path(
    value: str | os.PathLike[str],
    role: str,
    *,
    input_scope: bool = False,
    must_exist: bool = False,
) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise PostCollectionV3Error(f"{role} path is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(text)))
    if input_scope and _forbidden_namespace(PurePath(lexical)):
        raise PostCollectionV3Error(f"{role} enters a forbidden namespace")
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise PostCollectionV3Error(f"{role} path contains a symlink")
    try:
        resolved = lexical.resolve(strict=must_exist)
    except OSError as error:
        raise PostCollectionV3Error(f"{role} path is unavailable") from error
    if input_scope and _forbidden_namespace(PurePath(resolved)):
        raise PostCollectionV3Error(f"{role} resolves into a forbidden namespace")
    return resolved


def existing_file(
    value: str | os.PathLike[str],
    role: str,
    *,
    frozen: bool = True,
    input_scope: bool = True,
) -> Path:
    path = safe_path(value, role, input_scope=input_scope, must_exist=True)
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.suffix.casefold() in HDF_SUFFIXES:
        raise PostCollectionV3Error(f"{role} must be a regular non-HDF file")
    if frozen and mode & 0o222:
        raise PostCollectionV3Error(f"{role} must be frozen read-only")
    return path


def existing_directory(
    value: str | os.PathLike[str],
    role: str,
    *,
    frozen: bool = False,
    input_scope: bool = True,
) -> Path:
    path = safe_path(value, role, input_scope=input_scope, must_exist=True)
    mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode):
        raise PostCollectionV3Error(f"{role} must be a directory")
    if frozen and mode & 0o222:
        raise PostCollectionV3Error(f"{role} must be frozen read-only")
    return path


def file_sha256(path: Path) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise PostCollectionV3Error("post v3 cannot open or hash HDF bytes")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(
    path: Path, role: str, *, frozen: bool = True, input_scope: bool = True
) -> dict[str, Any]:
    source = existing_file(
        path, role, frozen=frozen, input_scope=input_scope
    )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostCollectionV3Error(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise PostCollectionV3Error(f"{role} must contain an object")
    return value


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise PostCollectionV3Error(f"{role} logical SHA mismatch")
    return str(recorded)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def freeze_tree(root: Path, *, hidden: frozenset[Path] = frozenset()) -> None:
    hidden_resolved = {path.resolve(strict=False) for path in hidden}
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise PostCollectionV3Error("output tree contains a symlink")
            if item.resolve(strict=False) not in hidden_resolved:
                item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise PostCollectionV3Error("output tree contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def _record(path: Path, logical_sha256: str | None = None) -> dict[str, str]:
    row = {"path": str(path), "file_sha256": file_sha256(path)}
    if logical_sha256 is not None:
        row["logical_sha256"] = logical_sha256
    return row


def _assert_record(
    value: Any, role: str, *, logical_field: str | None = None
) -> tuple[Path, dict[str, Any]]:
    expected = {"path", "file_sha256"}
    if logical_field is not None:
        expected.add("logical_sha256")
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PostCollectionV3Error(f"{role} record fields changed")
    source = existing_file(Path(str(value["path"])), role)
    if not _is_sha(value.get("file_sha256")) or file_sha256(source) != value["file_sha256"]:
        raise PostCollectionV3Error(f"{role} file SHA changed")
    decoded = load_json(source, role) if source.suffix.casefold() == ".json" else {}
    if logical_field is not None:
        if verify_signed(decoded, logical_field, role) != value.get("logical_sha256"):
            raise PostCollectionV3Error(f"{role} logical SHA changed")
    return source, decoded


def wait_for_ppid1(
    *,
    getppid: Callable[[], int] = os.getppid,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 120.0,
) -> None:
    deadline = time.monotonic() + timeout
    while getppid() != 1:
        if time.monotonic() >= deadline:
            raise PostCollectionV3Error("detached watcher did not reach PPID 1")
        sleep(0.05)


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(
    process: subprocess.Popen[bytes], pgid: int, *, timeout: float = 10.0
) -> tuple[bool, bool]:
    if process.poll() is None or _process_group_alive(pgid):
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


def _stop_direct_process(
    process: subprocess.Popen[bytes], *, timeout: float = 10.0
) -> bool:
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


def gpu_audit(index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=name,uuid",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, timeout=15
    )
    fields = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(fields) != 2 or EXPECTED_GPU_FRAGMENT not in fields[0]:
        raise PostCollectionV3Error("designated GPU is not an RTX 4090")
    pids = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    compute_pids = sorted(
        int(row.strip()) for row in pids.stdout.splitlines() if row.strip().isdigit()
    )
    return {"index": index, "name": fields[0], "uuid": fields[1], "compute_pids": compute_pids}


def wait_two_idle(
    index: int,
    *,
    interval: float,
    audit: Callable[[int], Mapping[str, Any]] = gpu_audit,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    identity: tuple[str, str] | None = None
    while len(observations) < 2:
        current = dict(audit(index))
        current_identity = (str(current.get("name")), str(current.get("uuid")))
        if identity is None:
            identity = current_identity
        if current_identity != identity:
            raise PostCollectionV3Error("GPU identity changed during idle gate")
        if current.get("compute_pids"):
            observations.clear()
        else:
            observations.append(current)
        if len(observations) < 2:
            sleep(interval)
    base = {"gpu_index": index, "gpu_name": identity[0], "gpu_uuid": identity[1], "checks": 2}
    return {**base, "audit_sha256": canonical_sha256(base)}


def _runtime_binding_records(
    plan: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Any]] = [
        ("runtime Python", plan.get("python")),
        ("canonical event specification", plan.get("canonical_event_spec")),
    ]
    teacher = plan.get("canonical_teacher")
    if teacher is not None:
        records.append(("canonical teacher", teacher))
    implementations = plan.get("implementations")
    expected_implementation_roles = {
        "launcher",
        "materializer",
        "trainer",
        "evaluator",
        "calibrator",
        "identity_bridge_v2",
        "r9b_watcher",
    }
    if (
        not isinstance(implementations, Mapping)
        or set(implementations) != expected_implementation_roles
    ):
        raise PostCollectionV3Error("implementation bindings are missing")
    records.extend(
        (f"{role} implementation", record)
        for role, record in implementations.items()
    )
    closure = plan.get("python_import_closure")
    if not isinstance(closure, Mapping) or not closure:
        raise PostCollectionV3Error("Python import closure is missing")
    for module_name, record in closure.items():
        if (
            not isinstance(module_name, str)
            or not module_name.isidentifier()
            or not isinstance(record, Mapping)
            or Path(str(record.get("path", ""))).stem != module_name
        ):
            raise PostCollectionV3Error("Python import closure descriptor changed")
        records.append((f"Python import {module_name}", record))
    implementation_modules = {
        Path(str(record["path"])).stem for record in implementations.values()
    }
    if not implementation_modules <= set(closure):
        raise PostCollectionV3Error("Python import closure omits a stage implementation")
    normalized: list[tuple[str, Mapping[str, Any]]] = []
    for role, record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "file_sha256"}:
            raise PostCollectionV3Error(f"{role} binding fields changed")
        if not _is_sha(record.get("file_sha256")):
            raise PostCollectionV3Error(f"{role} file SHA is invalid")
        normalized.append((role, record))
    return normalized


def _open_verified_binding_fd(
    record: Mapping[str, Any], role: str
) -> tuple[str, int]:
    """Open and hash one immutable binding without a second pathname lookup."""

    raw = str(record["path"])
    lexical = Path(os.path.abspath(os.path.expanduser(raw)))
    if raw != str(lexical):
        raise PostCollectionV3Error(f"{role} path must be canonical absolute")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(lexical.anchor, directory_flags)
    file_fd: int | None = None
    try:
        for component in lexical.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(lexical.name, file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise PostCollectionV3Error(f"{role} must be a regular file")
        if before.st_mode & 0o222:
            raise PostCollectionV3Error(f"{role} must be frozen read-only")
        digest = hashlib.sha256()
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise PostCollectionV3Error(f"{role} changed while being hashed")
        if digest.hexdigest() != record["file_sha256"]:
            raise PostCollectionV3Error(f"{role} file SHA changed")
        os.lseek(file_fd, 0, os.SEEK_SET)
        retained = file_fd
        file_fd = None
        return str(lexical), retained
    except OSError as error:
        raise PostCollectionV3Error(
            f"{role} cannot be opened without following symlinks"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def open_verified_runtime_binding_fds(plan: Mapping[str, Any]) -> dict[str, int]:
    """Return path->verified fd for the complete runtime binding closure."""

    opened: dict[str, int] = {}
    try:
        for role, record in _runtime_binding_records(plan):
            path, descriptor = _open_verified_binding_fd(record, role)
            if path in opened:
                os.close(descriptor)
                continue
            opened[path] = descriptor
        return opened
    except BaseException:
        close_runtime_binding_fds(opened)
        raise


def close_runtime_binding_fds(bindings: Mapping[str, int] | None) -> None:
    if bindings is None:
        return
    for descriptor in bindings.values():
        try:
            os.close(descriptor)
        except OSError:
            pass


def fd_bound_command(
    command: Sequence[str], bindings: Mapping[str, int],
    import_closure: Mapping[str, Mapping[str, str]],
) -> tuple[list[str], tuple[int, ...]]:
    if len(command) < 2 or command[0] not in bindings:
        raise PostCollectionV3Error("runtime Python was not fd-bound")
    if command[1] not in bindings or Path(command[1]).suffix.casefold() != ".py":
        raise PostCollectionV3Error("stage implementation was not fd-bound")
    module_rows: dict[str, dict[str, str]] = {}
    required_fds = {bindings[command[0]], bindings[command[1]]}
    for module_name, record in import_closure.items():
        source_path = str(record["path"])
        descriptor = bindings.get(source_path)
        if descriptor is None:
            raise PostCollectionV3Error(
                f"Python import closure fd is missing: {module_name}"
            )
        required_fds.add(descriptor)
        module_rows[module_name] = {
            "fd_path": f"/proc/self/fd/{descriptor}",
            "source_path": source_path,
        }
    # Data and authority paths retain their canonical names so downstream
    # validators can enforce namespace/path/SHA contracts.  The isolated
    # bootstrap serves every local import from an already-verified FD.
    bound = [
        f"/proc/self/fd/{bindings[command[0]]}",
        "-I",
        "-c",
        FD_IMPORT_BOOTSTRAP,
        f"/proc/self/fd/{bindings[command[1]]}",
        command[1],
        json.dumps(module_rows, sort_keys=True, separators=(",", ":")),
        *command[2:],
    ]
    return bound, tuple(sorted(required_fds))


def make_pre_popen_guard(
    plan: Mapping[str, Any],
    *,
    command: Sequence[str],
    physical_gpu: Mapping[str, Any] | None,
    audit: Callable[[int], Mapping[str, Any]] = gpu_audit,
) -> Callable[[], tuple[dict[str, int], Mapping[str, Mapping[str, str]]]]:
    def guard() -> tuple[dict[str, int], Mapping[str, Mapping[str, str]]]:
        reject_inherited_cuda_remapping()
        verify_runtime_bindings(plan)
        if physical_gpu is not None:
            current = dict(audit(int(plan["gpu_index"])))
            for field in ("index", "name", "uuid"):
                expected_key = f"gpu_{field}" if field != "index" else "gpu_index"
                current_key = field
                if current.get(current_key) != physical_gpu.get(expected_key):
                    raise PostCollectionV3Error(
                        "physical GPU identity changed immediately before Popen"
                    )
            joined = list(command)
            device_positions = [
                index + 1
                for index, token in enumerate(joined[:-1])
                if token == "--device"
            ]
            if (
                device_positions != [joined.index("--device") + 1]
                or joined[device_positions[0]] != "cuda:0"
            ):
                raise PostCollectionV3Error(
                    "UUID-scoped GPU subprocess must use the only visible cuda:0"
                )
        # This is deliberately the final guard operation.  The returned FDs,
        # not the pathnames just checked above, are what Popen executes/reads.
        closure = plan["python_import_closure"]
        assert isinstance(closure, Mapping)  # validated by the fd opener
        return open_verified_runtime_binding_fds(plan), closure

    return guard


def run_bound_stage(
    *,
    name: str,
    command: Sequence[str],
    stage_root: Path,
    gpu_index: int | None,
    poll_interval: float,
    pre_popen_guard: Callable[
        [], tuple[Mapping[str, int], Mapping[str, Mapping[str, str]]] | None
    ],
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    getpgid: Callable[[int], int] = os.getpgid,
    physical_gpu: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one create-once stage and prove both direct and group reaping."""

    if stage_root.exists() or stage_root.is_symlink():
        raise FileExistsError(stage_root)
    stage_root.mkdir(mode=0o700)
    launch: dict[str, Any] = {
        "format": STAGE_FORMAT,
        "status": "popen_not_attempted",
        "stage": name,
        "command": list(command),
        "command_sha256": canonical_sha256(list(command)),
        "gpu_index": gpu_index,
        "physical_gpu": dict(physical_gpu) if physical_gpu is not None else None,
        "fresh_confirmation_evaluation400_open_authorized": False,
    }
    launch["launch_sha256"] = canonical_sha256(launch)
    immutable_json(stage_root / "launch.json", launch)
    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    popen_attempted = False
    returncode: int | None = None
    direct_reaped = False
    group_reaped = False
    error: BaseException | None = None
    runtime_bindings: Mapping[str, int] | None = None
    import_closure: Mapping[str, Mapping[str, str]] | None = None
    log_path = stage_root / "run.log"
    try:
        with log_path.open("xb") as log:
            guarded = pre_popen_guard()
            if guarded is not None:
                runtime_bindings, import_closure = guarded
            gpu_uuid: str | None = None
            if physical_gpu is not None:
                gpu_uuid = physical_gpu.get("gpu_uuid")
                if not isinstance(gpu_uuid, str) or not gpu_uuid:
                    raise PostCollectionV3Error("physical GPU UUID is invalid")
            environment = isolated_subprocess_environment(
                physical_gpu_uuid=gpu_uuid
            )
            popen_attempted = True
            if runtime_bindings is not None and import_closure is not None:
                popen_command, pass_fds = fd_bound_command(
                    command, runtime_bindings, import_closure
                )
            else:
                popen_command, pass_fds = list(command), ()
            process = popen(
                popen_command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                pass_fds=pass_fds,
                env=environment,
            )
            close_runtime_binding_fds(runtime_bindings)
            runtime_bindings = None
            if type(process.pid) is not int or process.pid <= 0:
                raise UnprovenProcessGroup("Popen returned an invalid PID")
            pgid = getpgid(process.pid)
            if type(pgid) is not int or pgid <= 0 or pgid != process.pid:
                raise UnprovenProcessGroup("stage is not an isolated PID=PGID group")
            returncode = process.wait()
            direct_reaped = True
    except BaseException as caught:
        error = caught
    finally:
        close_runtime_binding_fds(runtime_bindings)
        runtime_bindings = None
        if popen_attempted:
            if process is None:
                direct_reaped = False
                group_reaped = False
            elif pgid is None or pgid != process.pid:
                # The returned PGID is not trusted.  Never signal an unrelated
                # process group; reap the direct child and retain the GPU lock
                # because descendants cannot be proved absent.
                direct_reaped = _stop_direct_process(process)
                if returncode is None and process.poll() is not None:
                    returncode = process.returncode
                group_reaped = False
            else:
                if process.poll() is not None:
                    try:
                        returncode = process.wait(timeout=0)
                        direct_reaped = True
                    except (subprocess.TimeoutExpired, OSError):
                        direct_reaped = False
                if error is not None or not direct_reaped or _process_group_alive(pgid):
                    direct_reaped, group_reaped = _stop_process_group(process, pgid)
                    if returncode is None and process.poll() is not None:
                        returncode = process.returncode
                else:
                    group_reaped = True
        lifecycle: dict[str, Any] = {
            "popen_attempted": popen_attempted,
            "popen_reached": process is not None,
            "process_pid": process.pid if process is not None else None,
            "process_pgid": pgid,
            "process_group_isolated": (
                process is not None and pgid is not None and pgid == process.pid
            ),
            "returncode": returncode,
            "direct_process_reaped": direct_reaped,
            "process_group_reaped": group_reaped,
            "binding_status": (
                "bound_reaped"
                if process is not None and pgid == process.pid and direct_reaped and group_reaped
                else "attempted_unproven"
                if popen_attempted
                else "not_attempted"
            ),
        }
        lifecycle["lifecycle_sha256"] = canonical_sha256(lifecycle)
        immutable_json(stage_root / "lifecycle.json", lifecycle)
        exit_code = returncode if type(returncode) is int else 255
        descriptor = os.open(
            stage_root / "run.exit", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{exit_code}\n")
            handle.flush()
            os.fsync(handle.fileno())
        for path in (stage_root / "launch.json", log_path, stage_root / "lifecycle.json"):
            if path.exists():
                path.chmod(0o444)
        stage_root.chmod(0o555)
    if not (direct_reaped and group_reaped and process is not None and pgid == process.pid):
        raise UnprovenProcessGroup(f"stage {name} process lifecycle is unproven")
    if error is not None:
        raise PostCollectionV3Error(f"stage {name} raised {type(error).__name__}") from error
    if type(returncode) is not int or isinstance(returncode, bool) or returncode != 0:
        raise PostCollectionV3Error(f"stage {name} exited with {returncode!r}")
    result = {
        "stage": name,
        "returncode": returncode,
        "command_sha256": launch["command_sha256"],
        "launch_file_sha256": file_sha256(stage_root / "launch.json"),
        "lifecycle": lifecycle,
        "physical_gpu": dict(physical_gpu) if physical_gpu is not None else None,
        "log_file_sha256": file_sha256(log_path),
        "run_exit_file_sha256": file_sha256(stage_root / "run.exit"),
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def validate_source_members(source_root: Path) -> dict[str, Any]:
    """Authenticate r7h and return five individual native checkpoints."""

    root = existing_directory(source_root, "r7h source root", frozen=True)
    if root != EXPECTED_SOURCE_ROOT.resolve(strict=False):
        raise PostCollectionV3Error("source root is not the designated r7h root")
    audit = validate_source_training_summary(root)
    source_final_path = root / "final_receipt.json"
    source_final = load_json(source_final_path, "r7h final receipt")
    source_final_logical = verify_signed(
        source_final, "receipt_sha256", "r7h final receipt"
    )
    if (
        audit.get("static_plan_sha256") != EXPECTED_SOURCE_STATIC_PLAN_SHA256
        or audit.get("member_seeds") != list(SOURCE_MEMBER_SEEDS)
        or len(audit.get("member_checkpoint_sha256", [])) != MEMBER_COUNT
    ):
        raise PostCollectionV3Error("r7h source audit lineage changed")
    plan = load_json(root / "launch_plan.json", "r7h launch plan")
    implementation_files = plan.get("implementation_files")
    launcher_binding = (
        implementation_files.get("scripts/launch_smolvla_schema5_source63_native_training.py")
        if isinstance(implementation_files, Mapping)
        else None
    )
    if (
        file_sha256(root / "launch_plan.json") != EXPECTED_SOURCE_PLAN_SHA256
        or plan.get("static_plan_sha256") != EXPECTED_SOURCE_STATIC_PLAN_SHA256
        or plan.get("implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
        or not isinstance(launcher_binding, Mapping)
        or launcher_binding.get("sha256") != EXPECTED_SOURCE_LAUNCHER_SHA256
    ):
        raise PostCollectionV3Error("r7h launch plan binding changed")
    manifest_path = root / "counterfactual_training" / "ensemble_manifest.json"
    manifest = load_json(manifest_path, "r7h ensemble manifest")
    rows = manifest.get("members")
    if (
        manifest.get("format") != "etsf_counterfactual_ensemble_v1"
        or not isinstance(rows, list)
        or len(rows) != MEMBER_COUNT
    ):
        raise PostCollectionV3Error("r7h ensemble manifest changed")
    members: list[dict[str, Any]] = []
    checkpoint_paths: set[str] = set()
    checkpoint_shas: set[str] = set()
    for index, (row, seed, expected_sha) in enumerate(
        zip(rows, SOURCE_MEMBER_SEEDS, audit["member_checkpoint_sha256"], strict=True)
    ):
        if not isinstance(row, Mapping) or row.get("seed") != seed:
            raise PostCollectionV3Error("r7h member ordering or seed changed")
        raw = Path(str(row.get("path", "")))
        checkpoint = raw if raw.is_absolute() else manifest_path.parent / raw
        checkpoint = existing_file(checkpoint, f"r7h member {index} checkpoint")
        digest = file_sha256(checkpoint)
        if digest != row.get("sha256") or digest != expected_sha:
            raise PostCollectionV3Error("r7h member checkpoint SHA changed")
        if str(checkpoint) in checkpoint_paths or digest in checkpoint_shas:
            raise PostCollectionV3Error("r7h member checkpoints are not one-to-one")
        checkpoint_paths.add(str(checkpoint))
        checkpoint_shas.add(digest)
        members.append(
            {
                "member_index": index,
                "member_seed": seed,
                "path": str(checkpoint),
                "file_sha256": digest,
                "checkpoint_role": "native_r7h_individual_source_member",
            }
        )
    return {
        "source_root": str(root),
        "source_final_receipt": _record(
            source_final_path, source_final_logical
        ),
        "source_launch_plan": _record(
            root / "launch_plan.json", EXPECTED_SOURCE_STATIC_PLAN_SHA256
        ),
        "source_launcher_sha256": EXPECTED_SOURCE_LAUNCHER_SHA256,
        "source_implementation_bundle_sha256": (
            EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
        ),
        "ensemble_manifest": _record(manifest_path),
        "members": members,
        "member_count": MEMBER_COUNT,
        "aggregate_checkpoint_training_source_authorized": False,
        "lobo_checkpoint_training_source_authorized": False,
        "audit_sha256": audit["summary_sha256"],
    }


def validate_r8e_and_r9b(
    r9b_root: Path,
    *,
    expected_final_file_sha256: str,
    expected_final_logical_sha256: str,
    expected_static_plan_sha256: str,
) -> dict[str, Any]:
    """Cross-check the actual r8e root against the frozen r9b gate."""

    if not all(
        _is_sha(value)
        for value in (
            expected_final_file_sha256,
            expected_final_logical_sha256,
            expected_static_plan_sha256,
        )
    ):
        raise PostCollectionV3Error("r9b expected SHA binding is invalid")
    root = existing_directory(r9b_root, "r9b gate root", frozen=True)
    final_path = existing_file(root / "final_receipt.json", "r9b final receipt")
    if file_sha256(final_path) != expected_final_file_sha256:
        raise PostCollectionV3Error("r9b final receipt file SHA changed")
    final = load_json(final_path, "r9b final receipt")
    if verify_signed(final, "receipt_sha256", "r9b final receipt") != expected_final_logical_sha256:
        raise PostCollectionV3Error("r9b final receipt logical SHA changed")
    validate_schema6_success_terminal_receipt(final)
    if (
        final.get("status") != R9B_TERMINAL_STATUS
        or final.get("static_plan_sha256") != expected_static_plan_sha256
    ):
        raise PostCollectionV3Error("r9b terminal status or plan changed")
    r9b_plan = load_json(root / "launch_plan.json", "r9b launch plan")
    launcher_record = r9b_plan.get("watcher_implementation")
    if (
        not isinstance(launcher_record, Mapping)
        or launcher_record.get("sha256") != R9B_WATCHER_SHA256
        or r9b_plan.get("static_plan_sha256") != expected_static_plan_sha256
    ):
        raise PostCollectionV3Error("r9b watcher implementation changed")
    state = load_json(root / "launch_state.json", "r9b frozen state")
    if (
        state.get("status") != R9B_FROZEN_STATE_STATUS
        or state.get("stage_results") != final.get("stage_results")
        or state.get("stages_started") != list(final.get("execution_order", []))
        or state.get("current_stage") is not None
        or state.get("stage_pid") is not None
    ):
        raise PostCollectionV3Error("r9b frozen terminal state is not closed")
    actual_lobo = validate_lobo_terminal_summary(DESIGNATED_LOBO_ROOT)
    embedded_lobo = final.get("lobo_gate")
    if (
        not isinstance(embedded_lobo, Mapping)
        or embedded_lobo.get("summary_sha256") != actual_lobo.get("summary_sha256")
        or embedded_lobo.get("lobo_launcher_sha256")
        != EXPECTED_LOBO_LAUNCHER_SHA256
        or embedded_lobo.get("static_plan_sha256")
        != EXPECTED_LOBO_STATIC_PLAN_SHA256
        or embedded_lobo.get("lobo_checkpoints_rerank_authorized") is not False
        or final.get("lobo_checkpoints_rerank_authorized") is not False
        or final.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
    ):
        raise PostCollectionV3Error("r8e/r9b lineage cross-check failed")
    return {
        "r8e_root": str(DESIGNATED_LOBO_ROOT),
        "r8e_final_receipt": {
            "path": str(DESIGNATED_LOBO_ROOT / "final_receipt.json"),
            "file_sha256": actual_lobo["final_receipt_sha256"],
            "logical_sha256": actual_lobo["final_receipt_logical_sha256"],
        },
        "r8e_summary_sha256": actual_lobo["summary_sha256"],
        "r8e_launcher_sha256": EXPECTED_LOBO_LAUNCHER_SHA256,
        "r8e_static_plan_sha256": EXPECTED_LOBO_STATIC_PLAN_SHA256,
        "r9b_root": str(root),
        "r9b_final_receipt": _record(final_path, expected_final_logical_sha256),
        "r9b_static_plan_sha256": expected_static_plan_sha256,
        "r9b_watcher_sha256": R9B_WATCHER_SHA256,
        "lobo_checkpoints_rerank_authorized": False,
    }


def validate_development300_terminal_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    terminal_record = plan.get("development300_terminal")
    authority_record = plan.get("development300_runner_authority")
    target_record = plan.get("development300_target_preregistration")
    identity_record = plan.get("development300_identity_authority")
    terminal_path, terminal = _assert_record(
        terminal_record, "development300 terminal", logical_field="terminal_receipt_sha256"
    )
    authority_path, _ = _assert_record(
        authority_record, "development300 runner authority", logical_field="runner_authority_sha256"
    )
    target_path, _ = _assert_record(
        target_record, "development300 target preregistration", logical_field="preregistration_sha256"
    )
    identity_path, _ = _assert_record(
        identity_record, "development300 identity authority", logical_field="identity_authority_sha256"
    )
    if (
        terminal.get("format") != DEVELOPMENT300_TERMINAL_FORMAT
        or terminal.get("status") != DEVELOPMENT300_SUCCESS
        or terminal.get("completed_groups") != 300
        or terminal.get("split_counts")
        != {
            "adaptation_train": TRAIN_GROUPS,
            "adaptation_internal_validation": INTERNAL_GROUPS,
            "formal_target_validation": FORMAL_GROUPS,
        }
        or terminal.get("formal_payloads_sealed") != FORMAL_GROUPS
        or terminal.get("formal_label_opened_by_runner_or_watcher") is not False
        or not _is_exact_zero(terminal.get("evaluation400_commands_executed"))
    ):
        raise PostCollectionV3Error("development300 terminal scope changed")
    return {
        "terminal_path": str(terminal_path),
        "authority_path": str(authority_path),
        "target_path": str(target_path),
        "identity_path": str(identity_path),
        "terminal_sha256": terminal["terminal_receipt_sha256"],
    }


def acquire_formal190_claim(
    plan: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    terminal = plan["development300_terminal"]
    identity_base = {
        "development300_terminal_file_sha256": terminal["file_sha256"],
        "development300_terminal_sha256": terminal["logical_sha256"],
        "split_profile": SPLIT_PROFILE,
        "formal_group_count": FORMAL_GROUPS,
    }
    identity_sha = canonical_sha256(identity_base)
    claim_root = safe_path(
        plan["formal190_claim_root"],
        "formal190 claim root",
        input_scope=False,
        must_exist=True,
    )
    terminal_path = safe_path(
        terminal["path"],
        "development300 terminal path",
        input_scope=False,
    )
    collection_root = terminal_path.parent.parent
    expected_claim_root = (
        collection_root.parent / ".etsf_schema6_formal190_global_claims_v1"
    )
    if claim_root != expected_claim_root:
        raise PostCollectionV3Error(
            "formal190 claim root must be the collection-parent global namespace"
        )
    path = claim_root / f"formal190-{identity_sha}.claim.json"
    claim_base: dict[str, Any] = {
        "format": FORMAL190_CLAIM_FORMAT,
        "status": FORMAL190_CLAIM_STATUS,
        "formal190_identity_sha256": identity_sha,
        **identity_base,
        "post_v3_plan_sha256": plan["plan_sha256"],
        "post_v3_output_root": str(output_root),
        "formal190_authority_may_be_created_once": True,
        "reopen_from_second_output_authorized": False,
        "claimed_unix_ns": time.time_ns(),
    }
    claim = {**claim_base, "claim_sha256": canonical_sha256(claim_base)}
    payload = json.dumps(claim, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as error:
        try:
            consumed = validate_formal190_claim(
                path, expected_identity_sha256=identity_sha
            )
        except Exception:
            consumed = {
                "path": str(path),
                "formal190_identity_sha256": identity_sha,
                "consumed": True,
                "validation_status": "existing_claim_invalid_or_unreadable",
            }
        raise Formal190ClaimConsumed(
            "formal190 one-shot claim was already consumed by another output",
            consumed,
        ) from error
    except OSError as error:
        raise PostCollectionV3Error("formal190 one-shot claim creation failed") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise PostCollectionV3Error("formal190 claim inode is unsafe")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while consuming formal190 claim")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    _fsync_directory(claim_root)
    try:
        return validate_formal190_claim(
            path,
            expected_identity_sha256=identity_sha,
            expected_plan_sha256=str(plan["plan_sha256"]),
            expected_output_root=output_root,
        )
    except Exception as error:
        raise Formal190ClaimConsumed(
            "formal190 one-shot claim was consumed but could not be verified",
            {
                "path": str(path),
                "formal190_identity_sha256": identity_sha,
                "consumed": True,
                "validation_status": "new_claim_invalid_or_unreadable",
            },
        ) from error


def validate_formal190_claim(
    path: Path,
    *,
    expected_identity_sha256: str,
    expected_plan_sha256: str | None = None,
    expected_output_root: Path | None = None,
) -> dict[str, Any]:
    source = existing_file(
        path, "formal190 global claim", frozen=True, input_scope=False
    )
    metadata = source.lstat()
    claim = load_json(source, "formal190 global claim", input_scope=False)
    logical = verify_signed(claim, "claim_sha256", "formal190 global claim")
    expected_fields = {
        "format",
        "status",
        "formal190_identity_sha256",
        "development300_terminal_file_sha256",
        "development300_terminal_sha256",
        "split_profile",
        "formal_group_count",
        "post_v3_plan_sha256",
        "post_v3_output_root",
        "formal190_authority_may_be_created_once",
        "reopen_from_second_output_authorized",
        "claimed_unix_ns",
        "claim_sha256",
    }
    if (
        set(claim) != expected_fields
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
        or claim.get("format") != FORMAL190_CLAIM_FORMAT
        or claim.get("status") != FORMAL190_CLAIM_STATUS
        or claim.get("formal190_identity_sha256") != expected_identity_sha256
        or not _is_sha(claim.get("development300_terminal_file_sha256"))
        or not _is_sha(claim.get("development300_terminal_sha256"))
        or not _is_sha(claim.get("post_v3_plan_sha256"))
        or claim.get("split_profile") != SPLIT_PROFILE
        or claim.get("formal_group_count") != FORMAL_GROUPS
        or claim.get("formal190_authority_may_be_created_once") is not True
        or claim.get("reopen_from_second_output_authorized") is not False
        or type(claim.get("claimed_unix_ns")) is not int
        or claim["claimed_unix_ns"] <= 0
        or (
            expected_plan_sha256 is not None
            and claim.get("post_v3_plan_sha256") != expected_plan_sha256
        )
        or (
            expected_output_root is not None
            and claim.get("post_v3_output_root") != str(expected_output_root)
        )
    ):
        raise PostCollectionV3Error("formal190 global claim semantics changed")
    return {
        "path": str(source),
        "file_sha256": file_sha256(source),
        "logical_sha256": logical,
        "formal190_identity_sha256": expected_identity_sha256,
        "consumed": True,
    }


def materializer_v3_command(
    plan: Mapping[str, Any], output: Path
) -> list[str]:
    """The only adapter from post v3 into materializer v3's stable CLI."""

    terminal = plan["development300_terminal"]
    authority = plan["development300_runner_authority"]
    target = plan["development300_target_preregistration"]
    identity = plan["development300_identity_authority"]
    return [
        str(plan["python"]["path"]),
        str(plan["implementations"]["materializer"]["path"]),
        "--collection-root", str(plan["development300_collection_root"]),
        "--terminal-receipt", str(terminal["path"]),
        "--expected-terminal-receipt-file-sha256", str(terminal["file_sha256"]),
        "--expected-terminal-receipt-sha256", str(terminal["logical_sha256"]),
        "--runner-authority", str(authority["path"]),
        "--expected-runner-authority-file-sha256", str(authority["file_sha256"]),
        "--expected-runner-authority-sha256", str(authority["logical_sha256"]),
        "--target-preregistration", str(target["path"]),
        "--expected-target-preregistration-file-sha256", str(target["file_sha256"]),
        "--expected-target-preregistration-sha256", str(target["logical_sha256"]),
        "--identity-authority", str(identity["path"]),
        "--expected-identity-authority-file-sha256", str(identity["file_sha256"]),
        "--expected-identity-authority-sha256", str(identity["logical_sha256"]),
        "--bound-trainer", str(plan["implementations"]["trainer"]["path"]),
        "--expected-bound-trainer-file-sha256", str(plan["implementations"]["trainer"]["file_sha256"]),
        "--output-directory", str(output),
    ]


def validate_materializer_v3_outputs(output: Path) -> dict[str, Any]:
    root = existing_directory(output, "materializer v3 output", frozen=True)
    expected_names = set(MATERIALIZER_OUTPUTS.values())
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise PostCollectionV3Error("materializer v3 output inventory changed")
    receipt_path = root / MATERIALIZER_OUTPUTS["receipt"]
    receipt = load_json(receipt_path, "materializer v3 receipt")
    logical = verify_signed(receipt, "receipt_sha256", "materializer v3 receipt")
    if (
        receipt.get("format") != materializer.FORMAT
        or receipt.get("status") != materializer.COMPLETE_STATUS
        or receipt.get("training_inputs_complete") is not True
        or receipt.get("training_authorized") is not True
        or receipt.get("required_trainer_group_counts")
        != {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS}
        or receipt.get("group_count") != 300
        or not _is_exact_zero(
            receipt.get("formal_target_validation_hdf5_or_labels_opened")
        )
        or receipt.get("evaluation400_identity_or_execution_authorized") is not False
        or not _is_exact_zero(receipt.get("hdf5_content_files_opened"))
        or receipt.get("labels_or_outcomes_read") is not False
    ):
        raise PostCollectionV3Error("materializer v3 receipt scope changed")
    roles = {
        "partition": ("partition_sha256", materializer.TARGET_PARTITION_FORMAT),
        "split": ("split_sha256", materializer.EXTERNAL_SPLIT_FORMAT),
        "manifest": ("manifest_sha256", materializer.TRAINER_MANIFEST_FORMAT),
        "expected": ("expected_receipt_sha256", materializer.EXPECTED_FORMAT),
    }
    decoded: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, str]] = {}
    for role, (signature, expected_format) in roles.items():
        path = root / MATERIALIZER_OUTPUTS[role]
        value = load_json(path, f"materializer v3 {role}")
        logical_sha = verify_signed(value, signature, f"materializer v3 {role}")
        if value.get("format") != expected_format:
            raise PostCollectionV3Error(f"materializer v3 {role} format changed")
        decoded[role] = value
        records[role] = _record(path, logical_sha)
    expected = decoded["expected"]
    split = decoded["split"]
    partition = decoded["partition"]
    if (
        expected.get("split_profile") != SPLIT_PROFILE
        or expected.get("required_trainer_group_counts")
        != {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS}
        or split.get("split_profile") != SPLIT_PROFILE
        or [len(split.get(name, [])) for name in ("train", "validation", "test")]
        != [TRAIN_GROUPS, INTERNAL_GROUPS, FORMAL_GROUPS]
        or partition.get("split_profile") != SPLIT_PROFILE
        or len(partition.get("adaptation", [])) != TRAIN_GROUPS + INTERNAL_GROUPS
        or len(partition.get("validation", [])) != FORMAL_GROUPS
        or partition.get("evaluation") != []
    ):
        raise PostCollectionV3Error("materializer v3 80/30/190 profile changed")
    try:
        scanned_manifest, descriptors = trainer.scan_manifest(
            Path(records["manifest"]["path"])
        )
        decoded_split, split_audit = trainer.validate_external_split_authority(
            expected_receipt_path=Path(records["expected"]["path"]),
            expected_receipt_file_sha256=records["expected"]["file_sha256"],
            manifest_path=Path(records["manifest"]["path"]),
            manifest=scanned_manifest,
            descriptors=descriptors,
        )
    except Exception as error:
        raise PostCollectionV3Error(
            "trainer's complete v3 manifest/split validator rejected materialization"
        ) from error
    if (
        split_audit.get("split_profile") != SPLIT_PROFILE
        or split_audit.get("required_trainer_group_counts")
        != {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS}
        or decoded_split.get("train") != split.get("train")
        or decoded_split.get("validation") != split.get("validation")
        or decoded_split.get("test") != split.get("test")
        or len(
            set(decoded_split["train"])
            | set(decoded_split["validation"])
            | set(decoded_split["test"])
        )
        != TRAIN_GROUPS + INTERNAL_GROUPS + FORMAL_GROUPS
        or set(decoded_split["train"]) & set(decoded_split["validation"])
        or set(decoded_split["train"]) & set(decoded_split["test"])
        or set(decoded_split["validation"]) & set(decoded_split["test"])
        or set(partition["adaptation"])
        != set(decoded_split["train"]) | set(decoded_split["validation"])
        or partition["validation"] != decoded_split["test"]
    ):
        raise PostCollectionV3Error(
            "materializer v3 identity sets are not unique/disjoint/full/order-bound"
        )
    for receipt_key, role in (
        ("trainer_compatible_manifest", "manifest"),
        ("target_partition", "partition"),
        ("external_split", "split"),
        ("expected_manifest_split_receipt", "expected"),
    ):
        if receipt.get(receipt_key) != records[role]:
            raise PostCollectionV3Error(f"materializer receipt {receipt_key} binding changed")
    return {
        "receipt": _record(receipt_path, logical),
        **records,
        "four_downstream_outputs": [
            records["partition"], records["split"], records["manifest"], records["expected"]
        ],
        "split_profile": SPLIT_PROFILE,
        "required_trainer_group_counts": {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS},
        "formal190_hdf5_or_labels_opened": 0,
    }


def _prediction_contract(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "duration_target_transform": artifacts.get("duration_target_transform"),
        "next_event_observation_mask": artifacts.get("next_event_observation_mask"),
        "success_target": artifacts.get("success_target"),
        "recovery_target": artifacts.get("recovery_target"),
        "recovery_observation_mask": artifacts.get("recovery_observation_mask"),
        "recovery_shared_transition_stop_gradient": artifacts.get(
            "recovery_shared_transition_stop_gradient"
        ),
        "recovery_enters_primary_before_calibration": artifacts.get(
            "recovery_enters_primary_utility_or_uncertainty"
        ),
        "recovery_head_trained": artifacts.get("recovery_head_trained"),
        "object_prediction_space": artifacts.get("object_prediction_space"),
        "object_source_normalization_sha256": artifacts.get(
            "object_source_normalization_sha256"
        ),
        "object_observed_policy": artifacts.get("object_observed_policy"),
    }
    if (
        contract["duration_target_transform"] != "log1p_decision_steps"
        or contract["next_event_observation_mask"] != "duration_observed"
        or contract["success_target"]
        != "eventual_final_branch_success_repeated_per_transition"
        or contract["recovery_target"]
        != "conditional_recovery_given_operational_regress"
        or contract["recovery_observation_mask"] != "recovery_observed_and_regress"
        or contract["recovery_shared_transition_stop_gradient"] is not True
        or contract["recovery_enters_primary_before_calibration"] is not False
        or type(contract["recovery_head_trained"]) is not bool
        or contract["object_prediction_space"] != "physical_delta_xyz_m"
        or not _is_sha(contract["object_source_normalization_sha256"])
        or contract["object_observed_policy"]
        != "row_enabled_only_if_all_selected_xyz_are_valid"
    ):
        raise PostCollectionV3Error("adapter prediction contract changed")
    return contract


def validate_member_receipt_v3(
    value: Mapping[str, Any], *, member_index: int
) -> dict[str, Any]:
    if set(value) != MEMBER_FIELDS:
        raise PostCollectionV3Error(
            f"adapter member {member_index} receipt fields changed"
        )
    logical = verify_signed(
        value, "receipt_sha256", f"adapter member {member_index} receipt"
    )
    contract = value.get("prediction_contract")
    if not isinstance(contract, Mapping):
        raise PostCollectionV3Error("adapter prediction contract is missing")
    normalized_contract = _prediction_contract(
        {
            **dict(contract),
            "recovery_enters_primary_utility_or_uncertainty": contract.get(
                "recovery_enters_primary_before_calibration"
            ),
        }
    )
    source_rank_contract = value.get("source_rank_score_contract")
    try:
        normalized_source_rank_contract = trainer._validate_source_rank_score_contract(
            source_rank_contract
        )
    except (trainer.AdapterContractError, TypeError, ValueError) as error:
        raise PostCollectionV3Error(
            "adapter Source composite rank contract is invalid"
        ) from error
    if (
        value.get("format") != MEMBER_FORMAT
        or value.get("status")
        != "complete_frozen_development300_internal_validation_adapter"
        or type(value.get("member_index")) is not int
        or value.get("member_index") != member_index
        or value.get("member_seed") != SOURCE_MEMBER_SEEDS[member_index]
        or value.get("split_profile") != SPLIT_PROFILE
        or value.get("split_profile_version") != 3
        or value.get("required_trainer_group_counts")
        != {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS}
        or value.get("source_checkpoint_role")
        != "native_r7h_individual_source_member"
        or value.get("validation_lane")
        != "adaptation_derived_internal_validation_only"
        or value.get("internal_validation_group_count") != INTERNAL_GROUPS
        or value.get("sealed_formal_target_validation_group_count")
        != FORMAL_GROUPS
        or dict(contract) != normalized_contract
        or dict(source_rank_contract) != normalized_source_rank_contract
        or value.get("source_rank_score_contract_sha256")
        != normalized_source_rank_contract["contract_sha256"]
        or normalized_source_rank_contract.get("source_checkpoint_file_sha256")
        != value.get("source_checkpoint_sha256")
        or not _is_exact_zero(
            value.get(
                "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen"
            )
        )
        or not _is_exact_zero(
            value.get(
                "formal_target_validation_labels_opened_before_five_adapters_frozen"
            )
        )
        or value.get("formal_target_validation_release_condition")
        != "external_authority_after_all_five_adapter_checkpoints_are_frozen"
        or value.get("lobo_or_aggregate_checkpoint_used") is not False
        or any(
            not _is_sha(value.get(field))
            for field in (
                "source_checkpoint_sha256",
                "training_manifest_sha256",
                "split_sha256",
                "source_ensemble_contract_sha256",
                "summary_file_sha256",
                "summary_sha256",
                "checkpoint_file_sha256",
                "validation_predictions_file_sha256",
                "validation_predictions_logical_sha256",
                "validation_labels_file_sha256",
                "validation_labels_logical_sha256",
                "validation_identity_set_sha256",
                "stage_result_sha256",
                "source_rank_score_contract_sha256",
            )
        )
    ):
        raise PostCollectionV3Error(
            f"adapter member {member_index} receipt semantics changed"
        )
    for role, path_field, sha_field, input_scope in (
        ("source checkpoint", "source_checkpoint_path", "source_checkpoint_sha256", True),
        ("training summary", "summary_path", "summary_file_sha256", False),
        ("adapter checkpoint", "checkpoint_path", "checkpoint_file_sha256", False),
        ("internal predictions", "validation_predictions_path", "validation_predictions_file_sha256", False),
        ("internal labels", "validation_labels_path", "validation_labels_file_sha256", False),
    ):
        artifact = existing_file(
            Path(str(value[path_field])),
            f"adapter member {member_index} {role}",
            input_scope=input_scope,
        )
        if file_sha256(artifact) != value[sha_field]:
            raise PostCollectionV3Error(
                f"adapter member {member_index} {role} SHA changed"
            )
    summary = load_json(
        Path(str(value["summary_path"])),
        f"adapter member {member_index} training summary",
        input_scope=False,
    )
    if (
        verify_signed(
            summary,
            "summary_sha256",
            f"adapter member {member_index} training summary",
        )
        != value["summary_sha256"]
    ):
        raise PostCollectionV3Error("adapter member summary logical SHA changed")
    return {**dict(value), "verified_receipt_sha256": logical}


def train_member(
    *,
    root: Path,
    plan: Mapping[str, Any],
    materialized: Mapping[str, Any],
    source: Mapping[str, Any],
    member_index: int,
    poll_interval: float,
    physical_gpu: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    member = source["members"][member_index]
    if (
        type(member.get("member_index")) is not int
        or member.get("member_index") != member_index
        or member.get("member_seed") != SOURCE_MEMBER_SEEDS[member_index]
        or member.get("checkpoint_role")
        != "native_r7h_individual_source_member"
    ):
        raise PostCollectionV3Error("adapter/source member one-to-one binding changed")
    member_root = root / "members" / f"member_{member_index}"
    if member_root.exists() or member_root.is_symlink():
        raise FileExistsError(member_root)
    member_root.mkdir(mode=0o700)
    stage = member_root / "training_stage"
    adapter_output = stage / "adapter"
    command = [
        str(plan["python"]["path"]),
        str(plan["implementations"]["trainer"]["path"]),
        "--mode", "train",
        "--source-checkpoint", str(member["path"]),
        "--schema6-manifest", str(materialized["manifest"]["path"]),
        "--expected-manifest-split-receipt", str(materialized["expected"]["path"]),
        "--expected-manifest-split-receipt-file-sha256", str(materialized["expected"]["file_sha256"]),
        "--canonical-event-spec", str(plan["canonical_event_spec"]["path"]),
        "--output", str(adapter_output),
        "--device", "cuda:0",
        "--steps", str(plan["adapter_steps"]),
        "--eval-every", str(plan["adapter_eval_every"]),
        "--training-seed", str(member["member_seed"]),
    ]
    if plan.get("canonical_teacher") is not None:
        command.extend(
            [
                "--canonical-teacher-checkpoint",
                str(plan["canonical_teacher"]["path"]),
            ]
        )
    lifecycle = run_bound_stage(
        name=f"train_adapter_member_{member_index}",
        command=command,
        stage_root=stage,
        gpu_index=int(plan["gpu_index"]),
        poll_interval=poll_interval,
        pre_popen_guard=make_pre_popen_guard(
            plan, command=command, physical_gpu=physical_gpu
        ),
        physical_gpu=physical_gpu,
    )
    summary_path = existing_file(
        adapter_output / "training_summary.json",
        f"adapter member {member_index} summary",
        input_scope=False,
    )
    summary = load_json(
        summary_path,
        f"adapter member {member_index} summary",
        input_scope=False,
    )
    summary_sha = verify_signed(
        summary, "summary_sha256", f"adapter member {member_index} summary"
    )
    artifacts = summary.get("validation_artifacts")
    if (
        summary.get("status") != "complete"
        or summary.get("split_profile") != SPLIT_PROFILE
        or summary.get("split_profile_version") != 3
        or summary.get("required_trainer_group_counts")
        != {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS}
        or summary.get("train_groups") != TRAIN_GROUPS
        or summary.get("validation_groups") != INTERNAL_GROUPS
        or summary.get("sealed_test_groups") != FORMAL_GROUPS
        or not _is_exact_zero(summary.get("test_hdf5_files_opened"))
        or not _is_exact_zero(
            summary.get(
                "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen"
            )
        )
        or not _is_exact_zero(
            summary.get(
                "formal_target_validation_labels_opened_before_five_adapters_frozen"
            )
        )
        or summary.get("source_checkpoint_sha256") != member["file_sha256"]
        or summary.get("schema6_training_manifest_sha256")
        != materialized["manifest"]["logical_sha256"]
        or summary.get("external_split_sha256")
        != materialized["split"]["logical_sha256"]
        or not isinstance(artifacts, Mapping)
        or artifacts.get("split_profile") != SPLIT_PROFILE
        or artifacts.get("split_profile_version") != 3
        or artifacts.get("validation_group_count") != INTERNAL_GROUPS
        or artifacts.get("sealed_formal_target_validation_group_count")
        != FORMAL_GROUPS
        or not _is_exact_zero(
            artifacts.get("sealed_formal_target_validation_hdf5_files_opened")
        )
        or not _is_exact_zero(artifacts.get("sealed_test_labels_opened"))
        or not _is_exact_zero(
            artifacts.get(
                "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen"
            )
        )
        or not _is_exact_zero(
            artifacts.get(
                "formal_target_validation_labels_opened_before_five_adapters_frozen"
            )
        )
    ):
        raise PostCollectionV3Error(f"adapter member {member_index} summary is incomplete")
    prediction_contract = _prediction_contract(artifacts)
    source_rank_score_contract = trainer._validate_source_rank_score_contract(
        summary.get("source_rank_score_contract")
    )
    if source_rank_score_contract.get("source_checkpoint_file_sha256") != member[
        "file_sha256"
    ]:
        raise PostCollectionV3Error(
            f"adapter member {member_index} Source rank checkpoint binding changed"
        )
    checkpoint = existing_file(
        Path(str(summary["best_checkpoint"])),
        f"adapter member {member_index} checkpoint",
        input_scope=False,
    )
    predictions = existing_file(
        Path(str(artifacts["predictions_path"])),
        f"adapter member {member_index} internal predictions",
        input_scope=False,
    )
    labels = existing_file(
        Path(str(artifacts["labels_path"])),
        f"adapter member {member_index} internal labels",
        input_scope=False,
    )
    for path, expected, role in (
        (checkpoint, summary.get("best_checkpoint_sha256"), "checkpoint"),
        (predictions, artifacts.get("predictions_file_sha256"), "predictions"),
        (labels, artifacts.get("labels_file_sha256"), "labels"),
    ):
        if not _is_sha(expected) or file_sha256(path) != expected:
            raise PostCollectionV3Error(f"adapter member {member_index} {role} SHA changed")
    receipt: dict[str, Any] = {
        "format": MEMBER_FORMAT,
        "status": "complete_frozen_development300_internal_validation_adapter",
        "member_index": member_index,
        "member_seed": member["member_seed"],
        "split_profile": SPLIT_PROFILE,
        "split_profile_version": 3,
        "required_trainer_group_counts": {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS},
        "source_checkpoint_path": str(member["path"]),
        "source_checkpoint_sha256": member["file_sha256"],
        "source_checkpoint_role": member["checkpoint_role"],
        "training_manifest_sha256": materialized["manifest"]["logical_sha256"],
        "split_sha256": materialized["split"]["logical_sha256"],
        "source_ensemble_contract_sha256": source["ensemble_manifest"]["file_sha256"],
        "summary_path": str(summary_path),
        "summary_file_sha256": file_sha256(summary_path),
        "summary_sha256": summary_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": file_sha256(checkpoint),
        "validation_predictions_path": str(predictions),
        "validation_predictions_file_sha256": file_sha256(predictions),
        "validation_predictions_logical_sha256": artifacts["predictions_logical_sha256"],
        "validation_labels_path": str(labels),
        "validation_labels_file_sha256": file_sha256(labels),
        "validation_labels_logical_sha256": artifacts["labels_logical_sha256"],
        "validation_identity_set_sha256": artifacts["validation_identity_set_sha256"],
        "validation_lane": artifacts["lane"],
        "internal_validation_group_count": INTERNAL_GROUPS,
        "sealed_formal_target_validation_group_count": FORMAL_GROUPS,
        "prediction_contract": prediction_contract,
        "source_rank_score_contract": source_rank_score_contract,
        "source_rank_score_contract_sha256": source_rank_score_contract[
            "contract_sha256"
        ],
        "formal_target_validation_hdf5_files_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_labels_opened_before_five_adapters_frozen": 0,
        "formal_target_validation_release_condition": artifacts["formal_target_validation_release_condition"],
        "lobo_or_aggregate_checkpoint_used": False,
        "stage_result_sha256": lifecycle["result_sha256"],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = member_root / "final_receipt.json"
    immutable_json(receipt_path, receipt)
    member_root.chmod(0o555)
    return receipt, lifecycle


def validate_five_members(
    root: Path,
    members: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> None:
    if len(members) != MEMBER_COUNT:
        raise PostCollectionV3Error("exactly five adapters are required")
    shared_contracts = set()
    adapter_shas: set[str] = set()
    source_shas: set[str] = set()
    for index, (member, source_member) in enumerate(
        zip(members, source["members"], strict=True)
    ):
        receipt_path = existing_file(
            root / "members" / f"member_{index}" / "final_receipt.json",
            f"adapter member {index} receipt",
            input_scope=False,
        )
        persisted = load_json(
            receipt_path, f"adapter member {index} receipt", input_scope=False
        )
        validate_member_receipt_v3(persisted, member_index=index)
        if (
            persisted != dict(member)
            or type(member.get("member_index")) is not int
            or member.get("member_index") != index
            or member.get("member_seed") != SOURCE_MEMBER_SEEDS[index]
            or member.get("source_checkpoint_sha256")
            != source_member["file_sha256"]
            or member.get("source_checkpoint_path") != source_member["path"]
            or member.get("source_checkpoint_role")
            != "native_r7h_individual_source_member"
            or member.get("lobo_or_aggregate_checkpoint_used") is not False
            or member.get("sealed_formal_target_validation_group_count")
            != FORMAL_GROUPS
            or member.get("prediction_contract", {}).get(
                "recovery_head_trained"
            )
            is not True
        ):
            raise PostCollectionV3Error("five-adapter freeze proof changed")
        adapter_shas.add(str(member["checkpoint_file_sha256"]))
        source_shas.add(str(member["source_checkpoint_sha256"]))
        shared_contracts.add(
            (
                member["training_manifest_sha256"],
                member["split_sha256"],
                member["source_ensemble_contract_sha256"],
                canonical_sha256(member["prediction_contract"]),
            )
        )
    if (
        len(adapter_shas) != MEMBER_COUNT
        or len(source_shas) != MEMBER_COUNT
        or len(shared_contracts) != 1
    ):
        raise PostCollectionV3Error("five adapters are not distinct members of one contract")


def build_evaluator_authority(
    *,
    root: Path,
    plan: Mapping[str, Any],
    materialized: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> Path:
    """Create the formal190 authority only after five frozen member receipts."""

    validate_five_members(root, members, source)
    authority_members: list[dict[str, Any]] = []
    for index, (member, source_member) in enumerate(
        zip(members, source["members"], strict=True)
    ):
        member_receipt_path = root / "members" / f"member_{index}" / "final_receipt.json"
        authority_members.append(
            {
                "member_index": index,
                "member_seed": member["member_seed"],
                "adapter_checkpoint": {
                    "path": member["checkpoint_path"],
                    "file_sha256": member["checkpoint_file_sha256"],
                },
                "source_checkpoint": {
                    "path": source_member["path"],
                    "file_sha256": source_member["file_sha256"],
                },
                "member_receipt": {
                    "path": str(member_receipt_path),
                    "file_sha256": file_sha256(member_receipt_path),
                    "logical_sha256": member["receipt_sha256"],
                },
                "training_manifest_sha256": member["training_manifest_sha256"],
                "split_sha256": member["split_sha256"],
                "source_ensemble_contract_sha256": member["source_ensemble_contract_sha256"],
                "prediction_contract": dict(member["prediction_contract"]),
                "source_rank_score_contract": dict(
                    member["source_rank_score_contract"]
                ),
                "source_rank_score_contract_sha256": member[
                    "source_rank_score_contract_sha256"
                ],
            }
        )
    numeric_contracts = {
        row["source_rank_score_contract"]["source_rank_numeric_contract"]
        for row in authority_members
    }
    if len(numeric_contracts) != 1:
        raise PostCollectionV3Error(
            "five adapter members do not share one Source rank numeric contract"
        )
    source_rank_numeric_contract = next(iter(numeric_contracts))
    authority: dict[str, Any] = {
        "format": evaluator.INPUT_FORMAT,
        "status": evaluator.INPUT_STATUS,
        "trainer_compatible_manifest": dict(materialized["manifest"]),
        "expected_manifest_split_receipt": dict(materialized["expected"]),
        "canonical_event_spec": dict(plan["canonical_event_spec"]),
        "members": authority_members,
        "member_count": MEMBER_COUNT,
        # This mirror is derived only from the five validated, signed member
        # contracts above.  It is not a downstream self-asserted constant.
        "source_rank_numeric_contract": source_rank_numeric_contract,
        "target_validation_group_count": FORMAL_GROUPS,
        "adapter_training_complete_before_authority": True,
        "target_validation_open_authorized": True,
        "evaluation400_membership_present": False,
        "evaluation400_open_authorized": False,
        "fresh_or_confirmation_open_authorized": False,
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    path = root / "formal190" / "evaluator_input_authority.json"
    immutable_json(path, authority)
    evaluator.validate_input_authority(path, file_sha256(path))
    return path


def validate_formal190_receipt(
    path: Path,
    *,
    expected_input_authority_path: Path,
    expected_input_authority_file_sha256: str,
    expected_input_authority_sha256: str,
) -> dict[str, Any]:
    receipt = load_json(path, "formal190 evaluator receipt", input_scope=False)
    verify_signed(receipt, "receipt_sha256", "formal190 evaluator receipt")
    if (
        receipt.get("format") != evaluator.RECEIPT_FORMAT
        or receipt.get("status") != evaluator.RECEIPT_STATUS
        or receipt.get("target_validation_groups") != FORMAL_GROUPS
        or receipt.get("target_validation_hdf5_files_opened") != FORMAL_GROUPS
        or type(receipt.get("target_validation_groups")) is not int
        or type(receipt.get("target_validation_hdf5_files_opened")) is not int
        or type(receipt.get("target_validation_samples")) is not int
        or receipt["target_validation_samples"] <= 0
        or receipt.get("input_authority_path")
        != str(expected_input_authority_path)
        or receipt.get("input_authority_file_sha256")
        != expected_input_authority_file_sha256
        or receipt.get("input_authority_sha256")
        != expected_input_authority_sha256
        or receipt.get("target_validation_opened_after_five_adapters_frozen") is not True
        or receipt.get("evaluation400_membership_present") is not False
        or not _is_exact_zero(
            receipt.get("evaluation400_hdf5_or_label_files_opened")
        )
        or not _is_exact_zero(receipt.get("fresh_or_confirmation_files_opened"))
        or receipt.get("performance_or_transfer_claim_authorized") is not False
        or receipt.get("source_rank_numeric_contract")
        != trainer.SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(
            receipt.get("source_rank_score_contract_sha256s"), list
        )
        or len(receipt["source_rank_score_contract_sha256s"]) != MEMBER_COUNT
        or any(
            not _is_sha(item)
            for item in receipt["source_rank_score_contract_sha256s"]
        )
    ):
        raise PostCollectionV3Error("formal190 evaluator receipt changed")
    authority = existing_file(
        Path(str(receipt["calibration_input_authority_path"])),
        "formal190 calibration input authority",
        input_scope=False,
    )
    if file_sha256(authority) != receipt.get("calibration_input_authority_file_sha256"):
        raise PostCollectionV3Error("formal190 calibration authority SHA changed")
    calibration_authority = load_json(
        authority, "formal190 calibration input authority", input_scope=False
    )
    calibration_audit = calibrator.validate_input_authority(calibration_authority)
    if (
        calibration_audit.get("logical_sha256")
        != receipt.get("calibration_input_authority_sha256")
        or verify_signed(
            calibration_authority,
            "input_authority_sha256",
            "formal190 calibration input authority",
        )
        != receipt.get("calibration_input_authority_sha256")
        or receipt["source_rank_score_contract_sha256s"]
        != [
            row.get("source_rank_score_contract_sha256")
            for row in calibration_authority.get("members", [])
            if isinstance(row, Mapping)
        ]
        or calibration_audit.get("source_rank_numeric_contract")
        != receipt.get("source_rank_numeric_contract")
    ):
        raise PostCollectionV3Error(
            "formal190 calibration authority logical SHA changed"
        )
    return receipt


def run_formal190_evaluator(
    *,
    root: Path,
    plan: Mapping[str, Any],
    authority_path: Path,
    poll_interval: float,
    physical_gpu: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = root / "formal190" / "evaluator_stage"
    result_root = stage / "result"
    command = [
        str(plan["python"]["path"]),
        str(plan["implementations"]["evaluator"]["path"]),
        "--input-authority", str(authority_path),
        "--input-authority-file-sha256", file_sha256(authority_path),
        "--output-root", str(result_root),
        "--device", "cuda:0",
    ]
    stage_result = run_bound_stage(
        name="evaluate_frozen_five_member_ensemble_on_formal190",
        command=command,
        stage_root=stage,
        gpu_index=int(plan["gpu_index"]),
        poll_interval=poll_interval,
        pre_popen_guard=make_pre_popen_guard(
            plan, command=command, physical_gpu=physical_gpu
        ),
        physical_gpu=physical_gpu,
    )
    authority_value = load_json(
        authority_path, "formal190 evaluator input authority", input_scope=False
    )
    authority_logical = verify_signed(
        authority_value,
        "authority_sha256",
        "formal190 evaluator input authority",
    )
    receipt = validate_formal190_receipt(
        result_root / "final_receipt.json",
        expected_input_authority_path=authority_path,
        expected_input_authority_file_sha256=file_sha256(authority_path),
        expected_input_authority_sha256=authority_logical,
    )
    return receipt, stage_result


def validate_calibration_result(
    path: Path,
    *,
    evaluator_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = load_json(path, "calibrator v2 receipt", input_scope=False)
    verify_signed(receipt, "receipt_sha256", "calibrator v2 receipt")
    try:
        member_authority_temperatures = (
            calibrator.validate_source_rank_member_authority(
                receipt.get("source_rank_member_authority"),
                receipt.get("source_rank_member_authority_sha256"),
            )
        )
    except (calibrator.CalibrationError, TypeError, ValueError) as error:
        raise PostCollectionV3Error(
            "calibrator Source rank member authority changed"
        ) from error
    if (
        receipt.get("format") != calibrator.RECEIPT_FORMAT
        or receipt.get("status") != calibrator.RECEIPT_STATUS
        or receipt.get("member_count") != MEMBER_COUNT
        or type(receipt.get("member_count")) is not int
        or receipt.get("validation_only") is not True
        or receipt.get("abstain_threshold_enabled") is not True
        or receipt.get("root_group_ranker_enabled_for_primary") is not True
        or receipt.get("test_artifacts_read") is not False
        or not _is_exact_zero(receipt.get("test_hdf5_files_opened"))
        or receipt.get("fresh_paths_accepted") is not False
        or receipt.get("confirmation_artifacts_read") is not False
        or receipt.get("paired_development_outcomes_read") is not False
        or receipt.get("performance_or_transfer_claim_authorized") is not False
        or receipt.get("source_rank_numeric_contract")
        != evaluator_receipt.get("source_rank_numeric_contract")
        or len(member_authority_temperatures) != MEMBER_COUNT
        or receipt.get("artifacts_frozen_read_only") is not True
        or receipt.get("input_authority_path")
        != evaluator_receipt.get("calibration_input_authority_path")
        or receipt.get("input_authority_file_sha256")
        != evaluator_receipt.get("calibration_input_authority_file_sha256")
        or receipt.get("input_authority_sha256")
        != evaluator_receipt.get("calibration_input_authority_sha256")
    ):
        raise PostCollectionV3Error("calibrator v2 terminal scope changed")
    paths = {
        "calibration": existing_file(
            Path(str(receipt["calibration_path"])), "calibration", input_scope=False
        ),
        "head_support": existing_file(
            Path(str(receipt["head_support_path"])), "head support", input_scope=False
        ),
        "ensemble_manifest": existing_file(
            Path(str(receipt["ensemble_manifest_path"])), "ensemble manifest", input_scope=False
        ),
        "root_group_ranker": existing_file(
            Path(str(receipt["root_group_ranker_path"])),
            "formal190 root group ranker",
            input_scope=False,
        ),
        "calibration_receipt": existing_file(path, "calibration receipt", input_scope=False),
    }
    for role in (
        "calibration", "head_support", "ensemble_manifest",
        "root_group_ranker",
    ):
        if file_sha256(paths[role]) != receipt.get(f"{role}_file_sha256"):
            raise PostCollectionV3Error(f"calibrator {role} file SHA changed")
    calibration_value = load_json(paths["calibration"], "calibration", input_scope=False)
    head_value = load_json(paths["head_support"], "head support", input_scope=False)
    ensemble_value = load_json(paths["ensemble_manifest"], "ensemble manifest", input_scope=False)
    root_ranker_value = load_json(
        paths["root_group_ranker"],
        "formal190 root group ranker",
        input_scope=False,
    )
    if (
        type(calibration_value.get("validation_groups")) is not int
        or calibration_value.get("validation_groups") != FORMAL_GROUPS
        or not _is_exact_zero(calibration_value.get("test_hdf5_files_opened"))
        or calibration_value.get(
            "all_six_heads_support_performance_uncertainty_gate_passed"
        ) is not True
        or calibration_value.get("root_group_ranker_enabled_for_primary") is not True
        or root_ranker_value != calibration_value.get("root_group_ranker")
    ):
        raise PostCollectionV3Error(
            "calibration did not use exactly the bound formal190 groups"
        )
    calibration_audit = identity_bridge.validate_calibration(calibration_value)
    head_audit = identity_bridge.validate_head_support(head_value)
    ensemble_audit = identity_bridge.validate_ensemble_manifest(
        ensemble_value, calibration=calibration_audit, head=head_audit
    )
    files = {name: file_sha256(item) for name, item in paths.items()}
    identity_bridge.validate_calibration_receipt(
        receipt,
        paths=paths,
        files=files,
        calibration=calibration_audit,
        head=head_audit,
        ensemble=ensemble_audit,
    )
    enabled = ensemble_value.get("head_enabled_for_primary")
    support_heads = head_value.get("heads")
    if (
        not isinstance(enabled, Mapping)
        or set(enabled)
        != {"post_event", "next_event", "duration", "success", "recovery", "object_effect"}
        or any(enabled.get(name) is not True for name in (
            "post_event", "next_event", "duration", "success", "recovery", "object_effect"
        ))
        or not isinstance(support_heads, Mapping)
        or set(support_heads)
        != {"post_event", "next_event", "duration", "success", "recovery", "object_effect"}
        or ensemble_value.get(
            "all_six_heads_support_performance_uncertainty_gate_passed"
        ) is not True
        or ensemble_value.get("root_group_ranker_enabled_for_primary") is not True
    ):
        raise PostCollectionV3Error("six-head calibration or core-head gate changed")
    recovery = support_heads["recovery"]
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("all_member_recovery_heads_trained") is not True
        or recovery.get("enabled_for_primary")
        is not (
            recovery.get("support_threshold_met") is True
            and recovery.get("performance_gate_passed") is True
            and recovery.get("uncertainty_gate_passed") is True
            and recovery.get("all_member_recovery_heads_trained") is True
        )
    ):
        raise PostCollectionV3Error("conditional recovery support gate changed")
    return {
        "receipt": receipt,
        "paths": paths,
        "files": files,
        "calibration": calibration_value,
        "head_support": head_value,
        "ensemble": ensemble_value,
        "root_group_ranker": root_ranker_value,
    }


def run_calibrator_v2(
    *,
    root: Path,
    plan: Mapping[str, Any],
    input_authority_path: Path,
    evaluator_receipt: Mapping[str, Any],
    poll_interval: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage = root / "calibration" / "calibrator_stage"
    result_root = stage / "result"
    command = [
        str(plan["python"]["path"]),
        str(plan["implementations"]["calibrator"]["path"]),
        "--input-authority", str(input_authority_path),
        "--input-authority-file-sha256", file_sha256(input_authority_path),
        "--output-root", str(result_root),
    ]
    stage_result = run_bound_stage(
        name="calibrate_six_head_formal190_ensemble",
        command=command,
        stage_root=stage,
        gpu_index=None,
        poll_interval=poll_interval,
        pre_popen_guard=make_pre_popen_guard(
            plan, command=command, physical_gpu=None
        ),
    )
    result = validate_calibration_result(
        result_root / "final_receipt.json",
        evaluator_receipt=evaluator_receipt,
    )
    return result, stage_result


def build_identity_bridge_handoff(
    *,
    root: Path,
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
    r8e_r9b: Mapping[str, Any],
    materialized: Mapping[str, Any],
    evaluator_receipt: Mapping[str, Any],
    calibration_result: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    formal190_claim: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    validate_five_members(root, members, source)
    receipt = calibration_result["receipt"]
    paths = calibration_result["paths"]
    produced = {
        "ensemble_manifest": _record(
            paths["ensemble_manifest"], receipt["ensemble_manifest_sha256"]
        ),
        "calibration": _record(paths["calibration"], receipt["calibration_sha256"]),
        "head_support": _record(paths["head_support"], receipt["head_support_sha256"]),
        "calibration_receipt": _record(
            paths["calibration_receipt"], receipt["receipt_sha256"]
        ),
    }
    evaluator_path = Path(str(evaluator_receipt["input_authority_path"]))
    formal_receipt_path = (
        root / "formal190" / "evaluator_stage" / "result" / "final_receipt.json"
    )
    handoff: dict[str, Any] = {
        "format": HANDOFF_FORMAT,
        "status": "ready_for_external_preoutcome_identity_bridge_v2_freeze",
        "post_v3_plan_sha256": plan["plan_sha256"],
        "lineage": {
            "r7h_source_final": dict(source["source_final_receipt"]),
            "r7h_member_checkpoint_sha256": [row["file_sha256"] for row in source["members"]],
            "r7h_member_seed": list(SOURCE_MEMBER_SEEDS),
            "r8e_root": r8e_r9b["r8e_root"],
            "r8e_final": dict(r8e_r9b["r8e_final_receipt"]),
            "r8e_summary_sha256": r8e_r9b["r8e_summary_sha256"],
            "r9b_final": dict(r8e_r9b["r9b_final_receipt"]),
            "development300_terminal": dict(plan["development300_terminal"]),
            "materializer_v3_receipt": dict(materialized["receipt"]),
            "formal190_evaluator_authority": _record(
                evaluator_path, evaluator_receipt["input_authority_sha256"]
            ),
            "formal190_evaluator_receipt": _record(
                formal_receipt_path, evaluator_receipt["receipt_sha256"]
            ),
            "formal190_global_one_shot_claim": dict(formal190_claim),
        },
        "identity_bridge_v2": {
            "implementation": dict(plan["implementations"]["identity_bridge_v2"]),
            "produced_dependencies": produced,
            "external_dependencies_required": list(BRIDGE_EXTERNAL_DEPENDENCIES),
            "cli_argument_mapping": {
                "ensemble-manifest": "produced_dependencies.ensemble_manifest",
                "calibration": "produced_dependencies.calibration",
                "head-support": "produced_dependencies.head_support",
                "calibration-receipt": "produced_dependencies.calibration_receipt",
            },
            "bridge_execution_authorized_by_handoff": False,
            "external_identity_attestor_required": True,
        },
        "adapter_member_count": MEMBER_COUNT,
        "adapter_member_receipt_sha256": [row["receipt_sha256"] for row in members],
        "split_profile": SPLIT_PROFILE,
        "required_trainer_group_counts": {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS},
        "formal190_opened_by_independent_evaluator_after_five_frozen_adapters": FORMAL_GROUPS,
        "formal190_opened_by_watcher_process": 0,
        "formal190_labels_opened_before_five_adapters_frozen": 0,
        "evaluation400_membership_present": False,
        "evaluation400_hdf5_trajectory_or_labels_opened": 0,
        "evaluation400_conditions_executed": 0,
        "old_paired400_authority_waited_or_generated": False,
        "second_reserve400_created": False,
        "lobo_or_aggregate_checkpoint_used_for_adapter_training": False,
        "performance_or_transfer_claim_authorized": False,
    }
    handoff["handoff_sha256"] = canonical_sha256(handoff)
    path = root / "handoff" / "evaluation400_identity_bridge_v2_handoff.json"
    immutable_json(path, handoff)
    return path, handoff


def _code_record(path: Path, expected: str, role: str) -> dict[str, str]:
    source = existing_file(path, role, frozen=True, input_scope=False)
    if not _is_sha(expected) or file_sha256(source) != expected:
        raise PostCollectionV3Error(f"{role} file SHA mismatch")
    return {"path": str(source), "file_sha256": expected}


def build_python_import_closure(
    implementations: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    """Freeze the recursively discovered local-module closure before execution."""

    pending = [Path(str(record["path"])) for record in implementations.values()]
    roots = {path.parent for path in pending}
    discovered: dict[str, dict[str, str]] = {}
    visited_paths: set[Path] = set()
    while pending:
        source = existing_file(
            pending.pop(), "Python import-closure source", frozen=True,
            input_scope=False,
        )
        if source in visited_paths:
            continue
        visited_paths.add(source)
        module_name = source.stem
        descriptor = {"path": str(source), "file_sha256": file_sha256(source)}
        previous = discovered.get(module_name)
        if previous is not None and previous != descriptor:
            raise PostCollectionV3Error(
                f"Python import closure has ambiguous module: {module_name}"
            )
        discovered[module_name] = descriptor
        try:
            _lexical, file_descriptor = _open_verified_binding_fd(
                descriptor, f"Python import {module_name}"
            )
            try:
                chunks: list[bytes] = []
                while True:
                    block = os.read(file_descriptor, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
            finally:
                os.close(file_descriptor)
            tree = ast.parse(
                b"".join(chunks).decode("utf-8"), filename=str(source)
            )
        except (OSError, UnicodeError, SyntaxError) as error:
            raise PostCollectionV3Error(
                f"Python import closure cannot parse {module_name}"
            ) from error
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                dynamic_import = (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                )
                if dynamic_import:
                    if (
                        not node.args
                        or not isinstance(node.args[0], ast.Constant)
                        or not isinstance(node.args[0].value, str)
                        or not node.args[0].value
                    ):
                        raise PostCollectionV3Error(
                            f"Python import closure has a non-literal dynamic import: "
                            f"{module_name}"
                        )
                    imported.add(node.args[0].value.split(".", 1)[0])
        for imported_name in sorted(imported):
            candidates = [root / f"{imported_name}.py" for root in roots]
            existing = [candidate for candidate in candidates if candidate.is_file()]
            if len(existing) > 1:
                raise PostCollectionV3Error(
                    f"Python import closure module is ambiguous: {imported_name}"
                )
            if existing:
                pending.append(existing[0])
    return {name: discovered[name] for name in sorted(discovered)}


def reject_inherited_cuda_remapping() -> None:
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited not in (None, ""):
        raise PostCollectionV3Error(
            "inherited CUDA_VISIBLE_DEVICES remapping is forbidden"
        )


def isolated_subprocess_environment(
    *, physical_gpu_uuid: str | None = None
) -> dict[str, str]:
    allowed = (
        "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE",
        "TZ", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME",
        "TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    )
    environment = {
        key: value for key in allowed
        if isinstance((value := os.environ.get(key)), str) and value
    }
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    if physical_gpu_uuid is not None:
        if not isinstance(physical_gpu_uuid, str) or not physical_gpu_uuid:
            raise PostCollectionV3Error("physical GPU UUID is invalid")
        environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_uuid
    return environment


def verify_runtime_bindings(plan: Mapping[str, Any]) -> None:
    """Rehash every executable/input binding before run and every Popen."""

    bindings = open_verified_runtime_binding_fds(plan)
    close_runtime_binding_fds(bindings)
    implementations = plan.get("implementations")
    assert isinstance(implementations, Mapping)  # normalized above
    r9b_record = implementations.get("r9b_watcher")
    actual_r9b_path = Path(str(r9b_watcher.__file__)).resolve(strict=True)
    if (
        not isinstance(r9b_record, Mapping)
        or Path(str(r9b_record.get("path", ""))).resolve(strict=True)
        != actual_r9b_path
        or r9b_record.get("file_sha256") != R9B_WATCHER_SHA256
        or file_sha256(actual_r9b_path) != R9B_WATCHER_SHA256
    ):
        raise PostCollectionV3Error("actually imported r9b watcher binding changed")
    if implementations.get("materializer", {}).get("file_sha256") != MATERIALIZER_SHA256:
        raise PostCollectionV3Error("materializer v3 reviewed SHA changed")


def preregister(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    reject_inherited_cuda_remapping()
    output = safe_path(args.output_root, "post v3 output", input_scope=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.resolve(strict=True)
    if safe_path(args.source_root, "r7h source root") != EXPECTED_SOURCE_ROOT.resolve(strict=False):
        raise PostCollectionV3Error("source root must be the designated r7h root")
    python_record = _code_record(args.python, args.python_sha256, "runtime Python")
    implementations = {
        "launcher": _code_record(
            Path(__file__), file_sha256(Path(__file__).resolve()), "post v3 launcher"
        ),
        "materializer": _code_record(
            args.materializer, args.materializer_sha256, "materializer v3"
        ),
        "trainer": _code_record(args.trainer, args.trainer_sha256, "adapter trainer"),
        "evaluator": _code_record(args.evaluator, args.evaluator_sha256, "formal190 evaluator"),
        "calibrator": _code_record(args.calibrator, args.calibrator_sha256, "calibrator v2"),
        "identity_bridge_v2": _code_record(
            args.identity_bridge_v2,
            args.identity_bridge_v2_sha256,
            "evaluation400 identity bridge v2",
        ),
        "r9b_watcher": _code_record(
            Path(str(r9b_watcher.__file__)),
            R9B_WATCHER_SHA256,
            "actually imported r9b watcher",
        ),
    }
    if implementations["materializer"]["file_sha256"] != MATERIALIZER_SHA256:
        raise PostCollectionV3Error("materializer is not the reviewed stable v3 build")
    python_import_closure = build_python_import_closure(implementations)
    event_spec = _code_record(
        args.canonical_event_spec,
        args.canonical_event_spec_sha256,
        "canonical event specification",
    )
    teacher = None
    if args.canonical_teacher is not None:
        teacher = _code_record(
            args.canonical_teacher,
            args.canonical_teacher_sha256,
            "canonical teacher checkpoint",
        )
    authority_path = existing_file(
        args.development300_runner_authority,
        "development300 runner authority",
    )
    authority = load_json(authority_path, "development300 runner authority")
    authority_logical = verify_signed(
        authority, "runner_authority_sha256", "development300 runner authority"
    )
    if (
        file_sha256(authority_path) != args.development300_runner_authority_file_sha256
        or authority_logical != args.development300_runner_authority_sha256
    ):
        raise PostCollectionV3Error("development300 runner authority binding changed")
    target_path = existing_file(
        args.development300_target_preregistration,
        "development300 target preregistration",
    )
    target = load_json(target_path, "development300 target preregistration")
    target_logical = verify_signed(
        target, "preregistration_sha256", "development300 target preregistration"
    )
    if (
        file_sha256(target_path) != args.development300_target_preregistration_file_sha256
        or target_logical != args.development300_target_preregistration_sha256
    ):
        raise PostCollectionV3Error("development300 target preregistration binding changed")
    identity_path = existing_file(
        args.development300_identity_authority,
        "development300 identity authority",
    )
    identity = load_json(identity_path, "development300 identity authority")
    identity_logical = verify_signed(
        identity, "identity_authority_sha256", "development300 identity authority"
    )
    if (
        file_sha256(identity_path) != args.development300_identity_authority_file_sha256
        or identity_logical != args.development300_identity_authority_sha256
    ):
        raise PostCollectionV3Error("development300 identity authority binding changed")
    terminal_path = safe_path(
        args.development300_terminal,
        "development300 terminal",
        input_scope=True,
    )
    collection_root = safe_path(
        args.development300_collection_root,
        "development300 collection root",
        input_scope=True,
    )
    if terminal_path != collection_root / "_runner" / "final_receipt.json":
        raise PostCollectionV3Error("development300 terminal is outside its exact root")
    claim_root = safe_path(
        args.formal190_claim_root,
        "formal190 global claim root",
        input_scope=False,
        must_exist=True,
    )
    expected_claim_root = (
        collection_root.parent / ".etsf_schema6_formal190_global_claims_v1"
    )
    if claim_root != expected_claim_root:
        raise PostCollectionV3Error(
            "formal190 claim root must be the collection-parent global namespace"
        )
    claim_mode = claim_root.lstat()
    if (
        not stat.S_ISDIR(claim_mode.st_mode)
        or claim_mode.st_uid != os.geteuid()
        or claim_mode.st_mode & 0o022
    ):
        raise PostCollectionV3Error(
            "formal190 claim root must be owner-controlled and not group/world writable"
        )
    for value, role in (
        (args.development300_terminal_file_sha256, "development300 terminal file SHA"),
        (args.development300_terminal_sha256, "development300 terminal logical SHA"),
        (args.r9b_final_file_sha256, "r9b final file SHA"),
        (args.r9b_final_sha256, "r9b final logical SHA"),
        (args.r9b_static_plan_sha256, "r9b static plan SHA"),
    ):
        if not _is_sha(value):
            raise PostCollectionV3Error(f"{role} is invalid")
    _require_int(args.gpu_index, "GPU index")
    _require_int(args.adapter_steps, "adapter steps", minimum=1)
    _require_int(args.adapter_eval_every, "adapter eval interval", minimum=1)
    output.mkdir(mode=0o700)
    for name in ("_watcher", "materialization", "members", "formal190", "calibration", "handoff"):
        (output / name).mkdir(mode=0o700)
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "status": "preregistered_waiting_for_exact_r7h_r8e_r9b_and_development300",
        "output_root": str(output),
        "source_root": str(EXPECTED_SOURCE_ROOT),
        "r9b_root": str(safe_path(args.r9b_root, "r9b gate root")),
        "r9b_final_file_sha256": args.r9b_final_file_sha256,
        "r9b_final_sha256": args.r9b_final_sha256,
        "r9b_static_plan_sha256": args.r9b_static_plan_sha256,
        "development300_collection_root": str(collection_root),
        "development300_terminal": {
            "path": str(terminal_path),
            "file_sha256": args.development300_terminal_file_sha256,
            "logical_sha256": args.development300_terminal_sha256,
        },
        "development300_runner_authority": _record(authority_path, authority_logical),
        "development300_target_preregistration": _record(target_path, target_logical),
        "development300_identity_authority": _record(identity_path, identity_logical),
        "python": python_record,
        "implementations": implementations,
        "python_import_closure": python_import_closure,
        "canonical_event_spec": event_spec,
        "canonical_teacher": teacher,
        "split_profile": SPLIT_PROFILE,
        "required_trainer_group_counts": {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS},
        "adapter_member_count": MEMBER_COUNT,
        "adapter_member_seeds": list(SOURCE_MEMBER_SEEDS),
        "adapter_source_policy": "one_to_one_native_r7h_individual_members_only",
        "lobo_or_aggregate_checkpoint_authorized": False,
        "adapter_steps": args.adapter_steps,
        "adapter_eval_every": args.adapter_eval_every,
        "gpu_index": args.gpu_index,
        "gpu_lock_path": str(safe_path(args.gpu_lock, "shared GPU lock", input_scope=False)),
        "formal190_claim_root": str(claim_root),
        "formal190_open_authorized_before_five_adapters_frozen": False,
        "evaluation400_membership_or_label_open_authorized": False,
        "old_paired400_authority_path_present": False,
        "second_reserve400_authorized": False,
        "hdf5_files_opened_during_preregistration": 0,
        "labels_or_outcomes_read_during_preregistration": False,
        "create_once_nonresumable": True,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    verify_runtime_bindings(plan)
    immutable_json(output / "_watcher" / "static_plan.json", plan)
    atomic_json(
        output / "_watcher" / "state.json",
        {
            "format": STATE_FORMAT,
            "status": plan["status"],
            "plan_sha256": plan["plan_sha256"],
            "fresh_confirmation_evaluation400_files_opened": 0,
        },
    )
    return output, plan


def load_bound_plan(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    expected_plan_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    reject_inherited_cuda_remapping()
    if (expected_file_sha256 is None) != (expected_plan_sha256 is None):
        raise PostCollectionV3Error("expected plan file/logical SHAs must be paired")
    if expected_file_sha256 is None:
        plan_path = existing_file(path, "post v3 static plan", input_scope=False)
        plan = load_json(plan_path, "post v3 static plan", input_scope=False)
    else:
        if not _is_sha(expected_plan_sha256):
            raise PostCollectionV3Error("expected plan logical SHA is invalid")
        lexical, descriptor = _open_verified_binding_fd(
            {
                "path": str(Path(os.path.abspath(os.path.expanduser(os.fspath(path))))),
                "file_sha256": expected_file_sha256,
            },
            "post v3 static plan",
        )
        try:
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            try:
                decoded = json.loads(b"".join(chunks).decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise PostCollectionV3Error("post v3 static plan is invalid JSON") from error
            if not isinstance(decoded, dict):
                raise PostCollectionV3Error("post v3 static plan must contain an object")
            plan = decoded
            plan_path = Path(lexical)
        finally:
            os.close(descriptor)
    logical = verify_signed(plan, "plan_sha256", "post v3 static plan")
    if expected_plan_sha256 is not None and logical != expected_plan_sha256:
        raise PostCollectionV3Error("post v3 static plan logical SHA changed")
    root = safe_path(plan.get("output_root", ""), "post v3 output", input_scope=False, must_exist=True)
    if (
        plan.get("format") != PLAN_FORMAT
        or plan_path != root / "_watcher" / "static_plan.json"
        or plan.get("split_profile") != SPLIT_PROFILE
        or plan.get("required_trainer_group_counts")
        != {"train": TRAIN_GROUPS, "validation": INTERNAL_GROUPS, "test": FORMAL_GROUPS}
        or plan.get("adapter_member_seeds") != list(SOURCE_MEMBER_SEEDS)
        or plan.get("adapter_source_policy")
        != "one_to_one_native_r7h_individual_members_only"
        or plan.get("lobo_or_aggregate_checkpoint_authorized") is not False
        or plan.get("evaluation400_membership_or_label_open_authorized") is not False
        or logical != plan["plan_sha256"]
    ):
        raise PostCollectionV3Error("post v3 static plan scope changed")
    verify_runtime_bindings(plan)
    claim_root = safe_path(
        plan.get("formal190_claim_root", ""),
        "formal190 claim root",
        input_scope=False,
        must_exist=True,
    )
    collection_root = safe_path(
        plan.get("development300_collection_root", ""),
        "development300 collection root",
        input_scope=True,
    )
    claim_metadata = claim_root.lstat()
    if (
        claim_root
        != collection_root.parent / ".etsf_schema6_formal190_global_claims_v1"
        or
        not stat.S_ISDIR(claim_metadata.st_mode)
        or claim_metadata.st_uid != os.geteuid()
        or claim_metadata.st_mode & 0o022
    ):
        raise PostCollectionV3Error("formal190 claim root security changed")
    return root, plan


def update_state(root: Path, plan: Mapping[str, Any], status: str, **extra: Any) -> None:
    atomic_json(
        root / "_watcher" / "state.json",
        {
            "format": STATE_FORMAT,
            "status": status,
            "plan_sha256": plan["plan_sha256"],
            "heartbeat_unix": time.time(),
            "formal190_hdf5_or_labels_opened_by_watcher": 0,
            "evaluation400_hdf5_trajectory_or_labels_opened": 0,
            **extra,
        },
    )


def wait_for_upstreams(
    plan: Mapping[str, Any],
    *,
    interval: float,
    heartbeat: Callable[[str], None],
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_root = Path(str(plan["source_root"]))
    r9b_root = Path(str(plan["r9b_root"]))
    terminal_path = Path(str(plan["development300_terminal"]["path"]))
    while True:
        if (source_root / "failure_receipt.json").exists():
            raise PostCollectionV3Error("r7h published a failure receipt")
        r9b_failure = r9b_root / "failure_receipt.json"
        if r9b_failure.exists():
            raise PostCollectionV3Error("r9b published a failure receipt")
        development_failure = (
            Path(str(plan["development300_collection_root"]))
            / "_runner"
            / "failure_receipt.json"
        )
        if development_failure.exists() or development_failure.is_symlink():
            failure = load_json(
                development_failure, "development300 failure receipt"
            )
            verify_signed(
                failure,
                "terminal_receipt_sha256",
                "development300 failure receipt",
            )
            if (
                failure.get("format") != DEVELOPMENT300_TERMINAL_FORMAT
                or failure.get("status") != DEVELOPMENT300_FAILURE
                or failure.get("retry_or_resume_authorized") is not False
                or failure.get("formal_label_opened_by_runner_or_watcher") is not False
                or not _is_exact_zero(
                    failure.get("evaluation400_commands_executed")
                )
            ):
                raise PostCollectionV3Error(
                    "development300 failure receipt contract is invalid"
                )
            raise PostCollectionV3Error(
                "development300 published an authenticated terminal failure"
            )
        if (source_root / "final_receipt.json").exists() and (
            r9b_root / "final_receipt.json"
        ).exists() and terminal_path.exists():
            break
        heartbeat("waiting_for_exact_r7h_r8e_r9b_and_development300_terminals")
        sleep(interval)
    source = validate_source_members(source_root)
    r8e_r9b = validate_r8e_and_r9b(
        r9b_root,
        expected_final_file_sha256=str(plan["r9b_final_file_sha256"]),
        expected_final_logical_sha256=str(plan["r9b_final_sha256"]),
        expected_static_plan_sha256=str(plan["r9b_static_plan_sha256"]),
    )
    development = validate_development300_terminal_binding(plan)
    return source, r8e_r9b, development


def _closure_record(
    role: str, path: Path, *, signature: str | None = None
) -> dict[str, Any]:
    source = existing_file(path, role, frozen=True, input_scope=False)
    logical: str | None = None
    if signature is not None:
        value = load_json(source, role, input_scope=False)
        logical = verify_signed(value, signature, role)
    return {
        "role": role,
        "path": str(source),
        "file_sha256": file_sha256(source),
        "signature": signature,
        "logical_sha256": logical,
    }


def _stage_root(root: Path, name: str) -> Path:
    if name == "materialize_development300_v3":
        return root / "materialization" / "materializer_stage"
    if name.startswith("train_adapter_member_"):
        index = int(name.rsplit("_", 1)[1])
        return root / "members" / f"member_{index}" / "training_stage"
    if name == "evaluate_frozen_five_member_ensemble_on_formal190":
        return root / "formal190" / "evaluator_stage"
    if name == "calibrate_six_head_formal190_ensemble":
        return root / "calibration" / "calibrator_stage"
    raise PostCollectionV3Error(f"unknown stage closure role: {name}")


def build_artifact_closure(
    *,
    root: Path,
    plan: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
    handoff_path: Path,
    formal190_claim: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = [
        _closure_record(
            "static_plan",
            root / "_watcher" / "static_plan.json",
            signature="plan_sha256",
        ),
        _closure_record(
            "detached_worker_proof",
            root / "_watcher" / "detached_worker_proof.json",
            signature="detach_proof_sha256",
        ),
        _closure_record(
            "gpu_idle_before_training",
            root / "_watcher" / "gpu_idle_before_training.json",
            signature="audit_sha256",
        ),
        _closure_record(
            "gpu_idle_after_formal190",
            root / "_watcher" / "gpu_idle_after_formal190.json",
            signature="audit_sha256",
        ),
        _closure_record(
            "gpu_lock_release",
            root / "_watcher" / "gpu_lock_release.json",
            signature="release_sha256",
        ),
        _closure_record(
            "materializer_v3_receipt",
            root
            / "materialization"
            / "materializer_stage"
            / "result"
            / MATERIALIZER_OUTPUTS["receipt"],
            signature="receipt_sha256",
        ),
        _closure_record(
            "formal190_evaluator_authority",
            root / "formal190" / "evaluator_input_authority.json",
            signature="authority_sha256",
        ),
        _closure_record(
            "formal190_evaluator_receipt",
            root / "formal190" / "evaluator_stage" / "result" / "final_receipt.json",
            signature="receipt_sha256",
        ),
        _closure_record(
            "calibration_receipt",
            root / "calibration" / "calibrator_stage" / "result" / "final_receipt.json",
            signature="receipt_sha256",
        ),
        _closure_record(
            "identity_bridge_v2_handoff",
            handoff_path,
            signature="handoff_sha256",
        ),
        _closure_record(
            "formal190_global_one_shot_claim",
            Path(str(formal190_claim["path"])),
            signature="claim_sha256",
        ),
    ]
    for index in range(MEMBER_COUNT):
        records.append(
            _closure_record(
                f"adapter_member_{index}_receipt",
                root / "members" / f"member_{index}" / "final_receipt.json",
                signature="receipt_sha256",
            )
        )
    for name in stage_results:
        stage = _stage_root(root, name)
        records.extend(
            [
                _closure_record(
                    f"stage:{name}:launch",
                    stage / "launch.json",
                    signature="launch_sha256",
                ),
                _closure_record(
                    f"stage:{name}:lifecycle",
                    stage / "lifecycle.json",
                    signature="lifecycle_sha256",
                ),
                _closure_record(f"stage:{name}:log", stage / "run.log"),
                _closure_record(f"stage:{name}:exit", stage / "run.exit"),
            ]
        )
    if records[0]["logical_sha256"] != plan["plan_sha256"]:
        raise PostCollectionV3Error("artifact closure binds a different plan")
    return records


def validate_artifact_closure(
    root: Path, records: Any, expected_sha256: Any
) -> None:
    if (
        not isinstance(records, list)
        or not _is_sha(expected_sha256)
        or canonical_sha256(records) != expected_sha256
    ):
        raise PostCollectionV3Error("terminal artifact closure SHA changed")
    expected_stages = [
        "materialize_development300_v3",
        *[f"train_adapter_member_{index}" for index in range(MEMBER_COUNT)],
        "evaluate_frozen_five_member_ensemble_on_formal190",
        "calibrate_six_head_formal190_ensemble",
    ]
    expected_roles = {
        "static_plan",
        "detached_worker_proof",
        "gpu_idle_before_training",
        "gpu_idle_after_formal190",
        "gpu_lock_release",
        "materializer_v3_receipt",
        "formal190_evaluator_authority",
        "formal190_evaluator_receipt",
        "calibration_receipt",
        "identity_bridge_v2_handoff",
        "formal190_global_one_shot_claim",
        *[f"adapter_member_{index}_receipt" for index in range(MEMBER_COUNT)],
        *[
            f"stage:{stage}:{suffix}"
            for stage in expected_stages
            for suffix in ("launch", "lifecycle", "log", "exit")
        ],
    }
    roles = [row.get("role") for row in records if isinstance(row, Mapping)]
    if len(roles) != len(expected_roles) or set(roles) != expected_roles:
        raise PostCollectionV3Error("terminal artifact closure inventory changed")
    for row in records:
        role = str(row.get("role", "")) if isinstance(row, Mapping) else ""
        if role == "static_plan":
            expected_signature = "plan_sha256"
        elif role == "detached_worker_proof":
            expected_signature = "detach_proof_sha256"
        elif role.startswith("gpu_idle_"):
            expected_signature = "audit_sha256"
        elif role == "gpu_lock_release":
            expected_signature = "release_sha256"
        elif role in {
            "materializer_v3_receipt",
            "formal190_evaluator_receipt",
            "calibration_receipt",
        } or role.startswith("adapter_member_"):
            expected_signature = "receipt_sha256"
        elif role == "formal190_evaluator_authority":
            expected_signature = "authority_sha256"
        elif role == "identity_bridge_v2_handoff":
            expected_signature = "handoff_sha256"
        elif role == "formal190_global_one_shot_claim":
            expected_signature = "claim_sha256"
        elif role.endswith(":launch"):
            expected_signature = "launch_sha256"
        elif role.endswith(":lifecycle"):
            expected_signature = "lifecycle_sha256"
        else:
            expected_signature = None
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"role", "path", "file_sha256", "signature", "logical_sha256"}
            or not _is_sha(row.get("file_sha256"))
            or row.get("signature") != expected_signature
            or (
                row.get("signature") is None
                and row.get("logical_sha256") is not None
            )
            or (
                row.get("signature") is not None
                and (
                    not isinstance(row.get("signature"), str)
                    or not _is_sha(row.get("logical_sha256"))
                )
            )
        ):
            raise PostCollectionV3Error("terminal artifact closure record changed")
        path = existing_file(
            Path(str(row["path"])),
            f"terminal closure {row['role']}",
            frozen=True,
            input_scope=False,
        )
        if file_sha256(path) != row["file_sha256"]:
            raise PostCollectionV3Error(
                f"terminal closure file SHA changed: {row['role']}"
            )
        if row["signature"] is not None:
            value = load_json(
                path, f"terminal closure {row['role']}", input_scope=False
            )
            if (
                verify_signed(
                    value, str(row["signature"]), f"terminal closure {row['role']}"
                )
                != row["logical_sha256"]
            ):
                raise PostCollectionV3Error(
                    f"terminal closure logical SHA changed: {row['role']}"
                )
    by_role = {str(row["role"]): row for row in records}
    expected_inside_paths = {
        "static_plan": root / "_watcher" / "static_plan.json",
        "detached_worker_proof": root / "_watcher" / "detached_worker_proof.json",
        "gpu_lock_release": root / "_watcher" / "gpu_lock_release.json",
        "gpu_idle_before_training": root / "_watcher" / "gpu_idle_before_training.json",
        "gpu_idle_after_formal190": root / "_watcher" / "gpu_idle_after_formal190.json",
        "materializer_v3_receipt": root / "materialization" / "materializer_stage" / "result" / MATERIALIZER_OUTPUTS["receipt"],
        "formal190_evaluator_authority": root / "formal190" / "evaluator_input_authority.json",
        "formal190_evaluator_receipt": root / "formal190" / "evaluator_stage" / "result" / "final_receipt.json",
        "calibration_receipt": root / "calibration" / "calibrator_stage" / "result" / "final_receipt.json",
        "identity_bridge_v2_handoff": root / "handoff" / "evaluation400_identity_bridge_v2_handoff.json",
        **{
            f"adapter_member_{index}_receipt": root / "members" / f"member_{index}" / "final_receipt.json"
            for index in range(MEMBER_COUNT)
        },
    }
    for stage in expected_stages:
        stage_root = _stage_root(root, stage)
        expected_inside_paths.update(
            {
                f"stage:{stage}:launch": stage_root / "launch.json",
                f"stage:{stage}:lifecycle": stage_root / "lifecycle.json",
                f"stage:{stage}:log": stage_root / "run.log",
                f"stage:{stage}:exit": stage_root / "run.exit",
            }
        )
    for role, expected_path in expected_inside_paths.items():
        if Path(str(by_role[role]["path"])) != expected_path:
            raise PostCollectionV3Error(f"terminal closure path changed: {role}")
    plan_value = load_json(
        Path(str(by_role["static_plan"]["path"])),
        "terminal closure static plan",
        input_scope=False,
    )
    claim_path = Path(str(by_role["formal190_global_one_shot_claim"]["path"]))
    claim_root = safe_path(
        plan_value.get("formal190_claim_root", ""),
        "terminal closure claim root",
        input_scope=False,
        must_exist=True,
    )
    claim_value = load_json(
        claim_path, "terminal closure formal190 claim", input_scope=False
    )
    expected_claim_name = (
        f"formal190-{claim_value.get('formal190_identity_sha256')}.claim.json"
    )
    if claim_path.parent != claim_root or claim_path.name != expected_claim_name:
        raise PostCollectionV3Error("terminal closure formal190 claim path changed")


def _validate_success_receipt(root: Path, value: Mapping[str, Any]) -> None:
    unsigned = dict(value)
    logical = unsigned.pop("receipt_sha256", None)
    stage_results = value.get("stage_results")
    expected_stages = [
        "materialize_development300_v3",
        *[f"train_adapter_member_{index}" for index in range(MEMBER_COUNT)],
        "evaluate_frozen_five_member_ensemble_on_formal190",
        "calibrate_six_head_formal190_ensemble",
    ]
    expected_fields = {
        "format",
        "status",
        "plan_sha256",
        "detach_proof_sha256",
        "execution_order",
        "stage_results",
        "adapter_member_count",
        "adapter_member_seeds",
        "adapter_source_policy",
        "r7h_member_checkpoint_sha256",
        "r8e_r9b_lineage_sha256",
        "development300_materializer_receipt_sha256",
        "formal190_opened_after_five_frozen_adapters",
        "formal190_labels_opened_before_five_adapters_frozen",
        "calibration_receipt_sha256",
        "identity_bridge_v2_handoff",
        "formal190_global_one_shot_claim",
        "evaluation400_hdf5_trajectory_or_labels_opened",
        "evaluation400_conditions_executed",
        "old_paired400_authority_waited_or_generated",
        "second_reserve400_created",
        "gpu_lock_release_sha256",
        "artifacts_frozen_read_only",
        "terminal_publication",
        "artifact_closure",
        "artifact_closure_sha256",
        "receipt_sha256",
    }
    if (
        set(value) != expected_fields
        or logical != canonical_sha256(unsigned)
        or value.get("format") != FORMAT
        or value.get("status") != TERMINAL_STATUS
        or value.get("plan_sha256") is None
        or value.get("execution_order") != expected_stages
        or not isinstance(stage_results, Mapping)
        or list(stage_results) != expected_stages
        or value.get("adapter_member_count") != MEMBER_COUNT
        or value.get("adapter_member_seeds") != list(SOURCE_MEMBER_SEEDS)
        or value.get("adapter_source_policy")
        != "one_to_one_native_r7h_individual_members_only"
        or value.get("formal190_opened_after_five_frozen_adapters") != FORMAL_GROUPS
        or not _is_exact_zero(
            value.get("formal190_labels_opened_before_five_adapters_frozen")
        )
        or not _is_exact_zero(
            value.get("evaluation400_hdf5_trajectory_or_labels_opened")
        )
        or not _is_exact_zero(value.get("evaluation400_conditions_executed"))
        or value.get("old_paired400_authority_waited_or_generated") is not False
        or value.get("second_reserve400_created") is not False
        or value.get("artifacts_frozen_read_only") is not True
        or value.get("terminal_publication")
        != "mode000_then_tree_freeze_verify_then_run_exit0444_then_final_receipt0444_last"
        or any(
            not _is_sha(value.get(field))
            for field in (
                "plan_sha256",
                "detach_proof_sha256",
                "r8e_r9b_lineage_sha256",
                "development300_materializer_receipt_sha256",
                "calibration_receipt_sha256",
                "gpu_lock_release_sha256",
            )
        )
        or not isinstance(value.get("r7h_member_checkpoint_sha256"), list)
        or len(value["r7h_member_checkpoint_sha256"]) != MEMBER_COUNT
        or len(set(value["r7h_member_checkpoint_sha256"])) != MEMBER_COUNT
        or any(not _is_sha(item) for item in value["r7h_member_checkpoint_sha256"])
        or not isinstance(value.get("identity_bridge_v2_handoff"), Mapping)
        or set(value["identity_bridge_v2_handoff"])
        != {"path", "file_sha256", "logical_sha256"}
        or not _is_sha(value["identity_bridge_v2_handoff"].get("file_sha256"))
        or not _is_sha(value["identity_bridge_v2_handoff"].get("logical_sha256"))
        or not isinstance(value.get("formal190_global_one_shot_claim"), Mapping)
        or value["formal190_global_one_shot_claim"].get("consumed") is not True
        or not _is_sha(
            value["formal190_global_one_shot_claim"].get("file_sha256")
        )
        or not _is_sha(
            value["formal190_global_one_shot_claim"].get("logical_sha256")
        )
    ):
        raise PostCollectionV3Error("post v3 success receipt semantics changed")
    validate_artifact_closure(
        root, value.get("artifact_closure"), value.get("artifact_closure_sha256")
    )
    closure_by_role = {
        row["role"]: row for row in value["artifact_closure"]
    }
    if (
        closure_by_role["detached_worker_proof"]["logical_sha256"]
        != value["detach_proof_sha256"]
        or closure_by_role["gpu_lock_release"]["logical_sha256"]
        != value["gpu_lock_release_sha256"]
        or closure_by_role["materializer_v3_receipt"]["logical_sha256"]
        != value["development300_materializer_receipt_sha256"]
        or closure_by_role["calibration_receipt"]["logical_sha256"]
        != value["calibration_receipt_sha256"]
        or {
            key: closure_by_role["identity_bridge_v2_handoff"][key]
            for key in ("path", "file_sha256", "logical_sha256")
        }
        != dict(value["identity_bridge_v2_handoff"])
        or any(
            closure_by_role["formal190_global_one_shot_claim"][key]
            != value["formal190_global_one_shot_claim"][key]
            for key in ("path", "file_sha256", "logical_sha256")
        )
    ):
        raise PostCollectionV3Error("terminal summary/closure binding changed")
    physical_gpu_bindings: list[dict[str, Any]] = []
    for name in expected_stages:
        result = stage_results[name]
        lifecycle = result.get("lifecycle") if isinstance(result, Mapping) else None
        if (
            not isinstance(result, Mapping)
            or set(result)
            != {
                "stage",
                "returncode",
                "command_sha256",
                "launch_file_sha256",
                "lifecycle",
                "physical_gpu",
                "log_file_sha256",
                "run_exit_file_sha256",
                "result_sha256",
            }
            or result.get("stage") != name
            or type(result.get("returncode")) is not int
            or result.get("returncode") != 0
            or not isinstance(lifecycle, Mapping)
            or lifecycle.get("popen_attempted") is not True
            or lifecycle.get("popen_reached") is not True
            or lifecycle.get("process_group_isolated") is not True
            or lifecycle.get("direct_process_reaped") is not True
            or lifecycle.get("process_group_reaped") is not True
            or lifecycle.get("binding_status") != "bound_reaped"
            or type(lifecycle.get("process_pid")) is not int
            or lifecycle.get("process_pid") <= 0
            or lifecycle.get("process_pgid") != lifecycle.get("process_pid")
            or set(lifecycle)
            != {
                "popen_attempted",
                "popen_reached",
                "process_pid",
                "process_pgid",
                "process_group_isolated",
                "returncode",
                "direct_process_reaped",
                "process_group_reaped",
                "binding_status",
                "lifecycle_sha256",
            }
            or verify_signed(lifecycle, "lifecycle_sha256", f"{name} lifecycle")
            != lifecycle.get("lifecycle_sha256")
            or any(
                not _is_sha(result.get(field))
                for field in (
                    "command_sha256",
                    "launch_file_sha256",
                    "log_file_sha256",
                    "run_exit_file_sha256",
                    "result_sha256",
                )
            )
            or result.get("result_sha256")
            != canonical_sha256(
                {key: child for key, child in result.items() if key != "result_sha256"}
            )
        ):
            raise PostCollectionV3Error(f"post v3 stage lifecycle is invalid: {name}")
        gpu_stage = name.startswith("train_adapter_member_") or name.startswith(
            "evaluate_frozen_five_member"
        )
        physical = result.get("physical_gpu")
        if gpu_stage:
            if (
                not isinstance(physical, Mapping)
                or set(physical)
                != {
                    "gpu_index",
                    "gpu_name",
                    "gpu_uuid",
                    "checks",
                    "audit_sha256",
                }
                or type(physical.get("gpu_index")) is not int
                or physical.get("gpu_index") < 0
                or not isinstance(physical.get("gpu_uuid"), str)
                or not physical["gpu_uuid"]
            ):
                raise PostCollectionV3Error(
                    f"post v3 physical GPU binding is invalid: {name}"
                )
            physical_gpu_bindings.append(dict(physical))
        elif physical is not None:
            raise PostCollectionV3Error(
                f"CPU stage falsely claims a physical GPU: {name}"
            )
        if (
            closure_by_role[f"stage:{name}:launch"]["file_sha256"]
            != result["launch_file_sha256"]
            or closure_by_role[f"stage:{name}:lifecycle"]["logical_sha256"]
            != lifecycle["lifecycle_sha256"]
            or closure_by_role[f"stage:{name}:log"]["file_sha256"]
            != result["log_file_sha256"]
            or closure_by_role[f"stage:{name}:exit"]["file_sha256"]
            != result["run_exit_file_sha256"]
        ):
            raise PostCollectionV3Error(
                f"terminal stage file closure changed: {name}"
            )
    if len({canonical_sha256(row) for row in physical_gpu_bindings}) != 1:
        raise PostCollectionV3Error("GPU stages do not share one physical UUID")


def _create_hidden(path: Path, payload: bytes) -> tuple[int, int]:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o000)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def publish_terminal(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    success: bool,
) -> None:
    terminal_name = "final_receipt.json" if success else "failure_receipt.json"
    exit_code = 0 if success else 1
    if success:
        _validate_success_receipt(root, receipt)
    elif (
        set(receipt)
        != {
            "format",
            "status",
            "plan_sha256",
            "error_type",
            "error_message_disclosed",
            "stage_results",
            "process_lifecycle_unproven",
            "gpu_lock_retained",
            "formal190_labels_opened_before_five_adapters_frozen",
            "evaluation400_hdf5_trajectory_or_labels_opened",
            "old_paired400_authority_waited_or_generated",
            "second_reserve400_created",
            "formal190_claim_consumed",
            "formal190_claim",
            "formal190_evaluator_receipt_sha256",
            "artifacts_frozen_read_only",
            "receipt_sha256",
        }
        or receipt.get("receipt_sha256")
        != canonical_sha256(
            {key: child for key, child in receipt.items() if key != "receipt_sha256"}
        )
        or receipt.get("format") != FORMAT
        or receipt.get("status") != FAILURE_STATUS
        or receipt.get("artifacts_frozen_read_only") is not True
        or receipt.get("process_lifecycle_unproven") is not False
        or receipt.get("gpu_lock_retained") is not False
        or not _is_exact_zero(
            receipt.get("formal190_labels_opened_before_five_adapters_frozen")
        )
        or not _is_exact_zero(
            receipt.get("evaluation400_hdf5_trajectory_or_labels_opened")
        )
        or receipt.get("old_paired400_authority_waited_or_generated") is not False
        or receipt.get("second_reserve400_created") is not False
        or type(receipt.get("formal190_claim_consumed")) is not bool
        or (
            receipt.get("formal190_claim_consumed") is True
            and (
                not isinstance(receipt.get("formal190_claim"), Mapping)
                or receipt["formal190_claim"].get("consumed") is not True
            )
        )
        or (
            receipt.get("formal190_claim_consumed") is False
            and receipt.get("formal190_claim") is not None
        )
        or (
            receipt.get("formal190_evaluator_receipt_sha256") is not None
            and not _is_sha(receipt["formal190_evaluator_receipt_sha256"])
        )
    ):
        raise PostCollectionV3Error("post v3 failure receipt semantics changed")
    terminal_path = root / terminal_name
    exit_path = root / "run.exit"
    opposite = root / ("failure_receipt.json" if success else "final_receipt.json")
    for path in (terminal_path, exit_path, opposite):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    terminal_payload = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    exit_payload = f"{exit_code}\n".encode("ascii")
    created: dict[Path, tuple[int, int]] = {}
    published = False
    try:
        created[terminal_path] = _create_hidden(terminal_path, terminal_payload)
        created[exit_path] = _create_hidden(exit_path, exit_payload)
        freeze_tree(root, hidden=frozenset({terminal_path, exit_path}))
        for directory, _names, files in os.walk(root, followlinks=False):
            base = Path(directory)
            if stat.S_IMODE(base.stat().st_mode) != 0o555:
                raise PostCollectionV3Error("terminal tree directory is not frozen")
            for name in files:
                path = base / name
                if path in (terminal_path, exit_path):
                    if stat.S_IMODE(path.stat().st_mode) != 0:
                        raise PostCollectionV3Error("terminal was visible before publication")
                elif stat.S_IMODE(path.stat().st_mode) != 0o444:
                    raise PostCollectionV3Error("terminal tree file is not frozen")
        _fsync_directory(root)
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
                    if stat.S_ISREG(metadata.st_mode) and (
                        metadata.st_dev,
                        metadata.st_ino,
                    ) == identity:
                        path.unlink()
                except (FileNotFoundError, OSError):
                    pass


def execute(
    plan_path: Path,
    *,
    poll_interval: float,
    idle_interval: float,
    expected_plan_file_sha256: str | None = None,
    expected_plan_sha256: str | None = None,
    require_ppid1: bool = True,
    hold_unproven: bool = True,
) -> dict[str, Any]:
    root, plan = load_bound_plan(
        plan_path,
        expected_file_sha256=expected_plan_file_sha256,
        expected_plan_sha256=expected_plan_sha256,
    )
    if require_ppid1:
        wait_for_ppid1()
    detached_proof: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "worker_detached_ppid1_verified",
        "plan_sha256": plan["plan_sha256"],
        "worker_pid": os.getpid(),
        "worker_ppid": os.getppid(),
        "ppid1_verified": os.getppid() == 1,
    }
    if require_ppid1 and detached_proof["ppid1_verified"] is not True:
        raise PostCollectionV3Error("worker is not detached at PPID 1")
    detached_proof["detach_proof_sha256"] = canonical_sha256(detached_proof)
    immutable_json(root / "_watcher" / "detached_worker_proof.json", detached_proof)
    heartbeat = lambda status: update_state(root, plan, status)
    stage_results: dict[str, dict[str, Any]] = {}
    lock_handle: Any | None = None
    formal190_claim: dict[str, Any] | None = None
    evaluator_receipt: dict[str, Any] | None = None
    try:
        source, r8e_r9b, _development = wait_for_upstreams(
            plan, interval=poll_interval, heartbeat=heartbeat
        )
        heartbeat("materializing_development300_training_inputs_v3")
        materializer_stage = root / "materialization" / "materializer_stage"
        materializer_output = materializer_stage / "result"
        materializer_result = run_bound_stage(
            name="materialize_development300_v3",
            command=materializer_v3_command(plan, materializer_output),
            stage_root=materializer_stage,
            gpu_index=None,
            poll_interval=poll_interval,
            pre_popen_guard=make_pre_popen_guard(
                plan,
                command=materializer_v3_command(plan, materializer_output),
                physical_gpu=None,
            ),
        )
        stage_results["materialize_development300_v3"] = materializer_result
        materialized = validate_materializer_v3_outputs(materializer_output)

        lock_path = safe_path(plan["gpu_lock_path"], "shared GPU lock", input_scope=False)
        lock_handle = open_shared_gpu_lock(lock_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        idle_before = wait_two_idle(
            int(plan["gpu_index"]), interval=idle_interval
        )
        immutable_json(root / "_watcher" / "gpu_idle_before_training.json", idle_before)
        members: list[dict[str, Any]] = []
        for index in range(MEMBER_COUNT):
            heartbeat(
                f"training_native_r7h_member_{index}_adapter_formal190_still_sealed"
            )
            member, stage_result = train_member(
                root=root,
                plan=plan,
                materialized=materialized,
                source=source,
                member_index=index,
                poll_interval=poll_interval,
                physical_gpu=idle_before,
            )
            members.append(member)
            stage_results[f"train_adapter_member_{index}"] = stage_result
        validate_five_members(root, members, source)

        heartbeat("five_adapters_frozen_authorizing_independent_formal190_evaluator")
        formal190_claim = acquire_formal190_claim(plan, output_root=root)
        evaluator_authority = build_evaluator_authority(
            root=root,
            plan=plan,
            materialized=materialized,
            members=members,
            source=source,
        )
        evaluator_receipt, evaluator_result = run_formal190_evaluator(
            root=root,
            plan=plan,
            authority_path=evaluator_authority,
            poll_interval=poll_interval,
            physical_gpu=idle_before,
        )
        stage_results[
            "evaluate_frozen_five_member_ensemble_on_formal190"
        ] = evaluator_result
        idle_after = wait_two_idle(
            int(plan["gpu_index"]), interval=idle_interval
        )
        if any(
            idle_after.get(field) != idle_before.get(field)
            for field in ("gpu_index", "gpu_name", "gpu_uuid")
        ):
            raise PostCollectionV3Error("physical GPU identity changed across GPU stages")
        immutable_json(root / "_watcher" / "gpu_idle_after_formal190.json", idle_after)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        released: dict[str, Any] = {
            "status": "released_after_all_gpu_stage_process_groups_reaped_and_idle",
            "lock_path": str(lock_path),
            "owner_pid": os.getpid(),
            "idle_before_sha256": idle_before["audit_sha256"],
            "idle_after_sha256": idle_after["audit_sha256"],
            "gpu_uuid": idle_before["gpu_uuid"],
            "all_gpu_stage_groups_reaped": True,
            "released_unix": time.time(),
        }
        released["release_sha256"] = canonical_sha256(released)
        immutable_json(root / "_watcher" / "gpu_lock_release.json", released)
        lock_handle.close()
        lock_handle = None

        heartbeat("calibrating_six_heads_from_formal190_only")
        calibration_authority = Path(
            str(evaluator_receipt["calibration_input_authority_path"])
        )
        calibration_result, calibration_stage = run_calibrator_v2(
            root=root,
            plan=plan,
            input_authority_path=calibration_authority,
            evaluator_receipt=evaluator_receipt,
            poll_interval=poll_interval,
        )
        stage_results["calibrate_six_head_formal190_ensemble"] = calibration_stage
        heartbeat("publishing_content_addressed_identity_bridge_v2_handoff")
        handoff_path, handoff = build_identity_bridge_handoff(
            root=root,
            plan=plan,
            source=source,
            r8e_r9b=r8e_r9b,
            materialized=materialized,
            evaluator_receipt=evaluator_receipt,
            calibration_result=calibration_result,
            members=members,
            formal190_claim=formal190_claim,
        )
        artifact_closure = build_artifact_closure(
            root=root,
            plan=plan,
            stage_results=stage_results,
            handoff_path=handoff_path,
            formal190_claim=formal190_claim,
        )
        final_base: dict[str, Any] = {
            "format": FORMAT,
            "status": TERMINAL_STATUS,
            "plan_sha256": plan["plan_sha256"],
            "detach_proof_sha256": detached_proof["detach_proof_sha256"],
            "execution_order": list(stage_results),
            "stage_results": stage_results,
            "adapter_member_count": MEMBER_COUNT,
            "adapter_member_seeds": list(SOURCE_MEMBER_SEEDS),
            "adapter_source_policy": "one_to_one_native_r7h_individual_members_only",
            "r7h_member_checkpoint_sha256": [row["file_sha256"] for row in source["members"]],
            "r8e_r9b_lineage_sha256": canonical_sha256(r8e_r9b),
            "development300_materializer_receipt_sha256": materialized["receipt"]["logical_sha256"],
            "formal190_opened_after_five_frozen_adapters": FORMAL_GROUPS,
            "formal190_labels_opened_before_five_adapters_frozen": 0,
            "calibration_receipt_sha256": calibration_result["receipt"]["receipt_sha256"],
            "identity_bridge_v2_handoff": _record(handoff_path, handoff["handoff_sha256"]),
            "formal190_global_one_shot_claim": dict(formal190_claim),
            "evaluation400_hdf5_trajectory_or_labels_opened": 0,
            "evaluation400_conditions_executed": 0,
            "old_paired400_authority_waited_or_generated": False,
            "second_reserve400_created": False,
            "gpu_lock_release_sha256": released["release_sha256"],
            "artifacts_frozen_read_only": True,
            "terminal_publication": "mode000_then_tree_freeze_verify_then_run_exit0444_then_final_receipt0444_last",
            "artifact_closure": artifact_closure,
            "artifact_closure_sha256": canonical_sha256(artifact_closure),
        }
        final = {**final_base, "receipt_sha256": canonical_sha256(final_base)}
        update_state(
            root,
            plan,
            PENDING_TERMINAL_STATUS,
            stage_results=stage_results,
            current_stage=None,
            stage_pid=None,
        )
        publish_terminal(root, receipt=final, success=True)
        return final
    except UnprovenProcessGroup:
        gpu_lock_retained = lock_handle is not None
        if gpu_lock_retained:
            retain_unproven_gpu_lock(lock_handle)
        update_state(
            root,
            plan,
            (
                "failed_unproven_process_group_gpu_lock_retained_unfrozen"
                if gpu_lock_retained
                else "failed_unproven_cpu_process_group_no_gpu_lock_unfrozen"
            ),
            stage_results=stage_results,
            process_lifecycle_unproven=True,
            gpu_lock_retained=gpu_lock_retained,
            artifacts_frozen_read_only=False,
        )
        if hold_unproven and gpu_lock_retained:
            while True:
                time.sleep(60)
        raise
    except BaseException as error:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            lock_handle = None
        if formal190_claim is None and isinstance(error, Formal190ClaimConsumed):
            formal190_claim = dict(error.claim)
        discovered_evaluator_sha = (
            evaluator_receipt.get("receipt_sha256")
            if evaluator_receipt is not None
            else None
        )
        evaluator_receipt_path = (
            root
            / "formal190"
            / "evaluator_stage"
            / "result"
            / "final_receipt.json"
        )
        if discovered_evaluator_sha is None and evaluator_receipt_path.is_file():
            try:
                discovered = load_json(
                    evaluator_receipt_path,
                    "failed-run formal190 evaluator receipt",
                    input_scope=False,
                )
                discovered_evaluator_sha = verify_signed(
                    discovered,
                    "receipt_sha256",
                    "failed-run formal190 evaluator receipt",
                )
            except Exception:
                discovered_evaluator_sha = None
        failure_base: dict[str, Any] = {
            "format": FORMAT,
            "status": FAILURE_STATUS,
            "plan_sha256": plan["plan_sha256"],
            "error_type": type(error).__name__,
            "error_message_disclosed": False,
            "stage_results": stage_results,
            "process_lifecycle_unproven": False,
            "gpu_lock_retained": False,
            "formal190_labels_opened_before_five_adapters_frozen": 0,
            "evaluation400_hdf5_trajectory_or_labels_opened": 0,
            "old_paired400_authority_waited_or_generated": False,
            "second_reserve400_created": False,
            "formal190_claim_consumed": formal190_claim is not None,
            "formal190_claim": dict(formal190_claim) if formal190_claim is not None else None,
            "formal190_evaluator_receipt_sha256": (
                discovered_evaluator_sha
            ),
            "artifacts_frozen_read_only": True,
        }
        failure = {**failure_base, "receipt_sha256": canonical_sha256(failure_base)}
        update_state(root, plan, FAILURE_STATUS, stage_results=stage_results)
        publish_terminal(root, receipt=failure, success=False)
        raise


def detach(plan_path: Path) -> dict[str, Any]:
    root, plan = load_bound_plan(plan_path)
    plan_file_sha256 = file_sha256(plan_path)
    receipt_path = root / "_watcher" / "detach_receipt.json"
    log_path = root / "_watcher" / "detached.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError("post v3 detach is create-once")
    command = [
        str(plan["python"]["path"]),
        str(plan["implementations"]["launcher"]["path"]),
        "run",
        "--plan",
        str(plan_path),
        "--expected-plan-file-sha256",
        plan_file_sha256,
        "--expected-plan-sha256",
        str(plan["plan_sha256"]),
    ]
    reject_inherited_cuda_remapping()
    verify_runtime_bindings(plan)
    environment = isolated_subprocess_environment()
    runtime_bindings: Mapping[str, int] | None = None
    try:
        # Open after the ordinary audit and execute the same verified inodes;
        # there is no pathname lookup for Python or the launcher after this.
        runtime_bindings = open_verified_runtime_binding_fds(plan)
        closure = plan["python_import_closure"]
        assert isinstance(closure, Mapping)
        popen_command, pass_fds = fd_bound_command(
            command, runtime_bindings, closure
        )
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                popen_command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                pass_fds=pass_fds,
                env=environment,
            )
    finally:
        close_runtime_binding_fds(runtime_bindings)
    try:
        pgid = os.getpgid(process.pid)
    except OSError as error:
        direct_reaped = _stop_direct_process(process)
        _record_detach_unproven(
            root=root,
            plan=plan,
            process_pid=process.pid,
            observed_pgid=None,
            direct_reaped=direct_reaped,
            reason="getpgid_failed",
        )
        raise PostCollectionV3Error("detached watcher PGID proof failed") from error
    if pgid != process.pid:
        direct_reaped = _stop_direct_process(process)
        _record_detach_unproven(
            root=root,
            plan=plan,
            process_pid=process.pid,
            observed_pgid=pgid,
            direct_reaped=direct_reaped,
            reason="pgid_mismatch",
        )
        raise PostCollectionV3Error("detached watcher is not a new process group")
    receipt: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detached_worker_started_ppid1_proof_required_before_inputs",
        "plan_sha256": plan["plan_sha256"],
        "worker_pid": process.pid,
        "worker_pgid": pgid,
        "process_group_isolated": True,
        "command_sha256": canonical_sha256(command),
        "ppid1_proof_path": str(root / "_watcher" / "detached_worker_proof.json"),
        "client_disconnect_safe_after_parent_exit": True,
        "fresh_confirmation_evaluation400_open_authorized": False,
    }
    receipt["detach_receipt_sha256"] = canonical_sha256(receipt)
    try:
        immutable_json(receipt_path, receipt)
    except BaseException as error:
        direct_reaped, group_reaped = _stop_process_group(process, pgid)
        if not (direct_reaped and group_reaped):
            try:
                _record_detach_unproven(
                    root=root,
                    plan=plan,
                    process_pid=process.pid,
                    observed_pgid=pgid,
                    direct_reaped=direct_reaped,
                    reason="detach_receipt_publication_failed",
                )
            except Exception:
                pass
        raise PostCollectionV3Error(
            "detach receipt publication failed; worker was stopped"
        ) from error
    return receipt


def _record_detach_unproven(
    *,
    root: Path,
    plan: Mapping[str, Any],
    process_pid: int,
    observed_pgid: int | None,
    direct_reaped: bool,
    reason: str,
) -> dict[str, Any]:
    failure_base: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detach_popen_reached_process_group_unproven",
        "plan_sha256": plan["plan_sha256"],
        "process_pid": process_pid,
        "observed_pgid": observed_pgid,
        "direct_process_reaped": direct_reaped,
        "process_group_reaped": False,
        "unknown_process_group_signaled": False,
        "reason": reason,
        "gpu_lock_acquired": False,
        "artifacts_frozen_read_only": False,
    }
    failure = {
        **failure_base,
        "detach_failure_sha256": canonical_sha256(failure_base),
    }
    immutable_json(root / "_watcher" / "detach_failure.json", failure)
    update_state(
        root,
        plan,
        "detach_popen_reached_process_group_unproven",
        process_pid=process_pid,
        observed_pgid=observed_pgid,
        direct_process_reaped=direct_reaped,
        process_group_reaped=False,
        gpu_lock_retained=False,
        artifacts_frozen_read_only=False,
    )
    return failure


def _add_bound_json_arguments(parser: argparse.ArgumentParser, stem: str) -> None:
    parser.add_argument(f"--{stem}", type=Path, required=True)
    parser.add_argument(f"--{stem}-file-sha256", required=True)
    parser.add_argument(f"--{stem}-sha256", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("preregister")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--source-root", type=Path, default=EXPECTED_SOURCE_ROOT)
    prepare.add_argument("--r9b-root", type=Path, required=True)
    prepare.add_argument("--r9b-final-file-sha256", required=True)
    prepare.add_argument("--r9b-final-sha256", required=True)
    prepare.add_argument("--r9b-static-plan-sha256", required=True)
    prepare.add_argument("--development300-collection-root", type=Path, required=True)
    _add_bound_json_arguments(prepare, "development300-terminal")
    _add_bound_json_arguments(prepare, "development300-runner-authority")
    _add_bound_json_arguments(prepare, "development300-target-preregistration")
    _add_bound_json_arguments(prepare, "development300-identity-authority")
    prepare.add_argument("--python", type=Path, required=True)
    prepare.add_argument("--python-sha256", required=True)
    prepare.add_argument("--materializer", type=Path, required=True)
    prepare.add_argument("--materializer-sha256", required=True)
    prepare.add_argument("--trainer", type=Path, required=True)
    prepare.add_argument("--trainer-sha256", required=True)
    prepare.add_argument("--evaluator", type=Path, required=True)
    prepare.add_argument("--evaluator-sha256", required=True)
    prepare.add_argument("--calibrator", type=Path, required=True)
    prepare.add_argument("--calibrator-sha256", required=True)
    prepare.add_argument("--identity-bridge-v2", type=Path, required=True)
    prepare.add_argument("--identity-bridge-v2-sha256", required=True)
    prepare.add_argument("--canonical-event-spec", type=Path, required=True)
    prepare.add_argument("--canonical-event-spec-sha256", required=True)
    prepare.add_argument("--canonical-teacher", type=Path)
    prepare.add_argument("--canonical-teacher-sha256")
    prepare.add_argument("--gpu-index", type=int, default=0)
    prepare.add_argument("--formal190-claim-root", type=Path, required=True)
    prepare.add_argument(
        "--gpu-lock",
        type=Path,
        default=Path("/tmp/etsf_smolvla_piper_schema6_post_v3_gpu0.lock"),
    )
    prepare.add_argument("--adapter-steps", type=int, default=3000)
    prepare.add_argument("--adapter-eval-every", type=int, default=50)

    detached = sub.add_parser("detach")
    detached.add_argument("--plan", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--expected-plan-file-sha256", required=True)
    run.add_argument("--expected-plan-sha256", required=True)
    run.add_argument("--poll-interval", type=float, default=30.0)
    run.add_argument("--idle-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preregister":
        if (args.canonical_teacher is None) != (
            args.canonical_teacher_sha256 is None
        ):
            raise PostCollectionV3Error("canonical teacher path/SHA must be paired")
        root, plan = preregister(args)
        print(
            json.dumps(
                {
                    "output_root": str(root),
                    "plan_sha256": plan["plan_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "detach":
        print(json.dumps(detach(args.plan), sort_keys=True))
        return 0
    if args.poll_interval <= 0 or args.idle_interval <= 0:
        raise PostCollectionV3Error("poll intervals must be positive")
    receipt = execute(
        args.plan,
        poll_interval=args.poll_interval,
        idle_interval=args.idle_interval,
        expected_plan_file_sha256=args.expected_plan_file_sha256,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAT",
    "HANDOFF_FORMAT",
    "MATERIALIZER_SHA256",
    "MEMBER_FORMAT",
    "PLAN_FORMAT",
    "PostCollectionV3Error",
    "SPLIT_PROFILE",
    "UnprovenProcessGroup",
    "build_evaluator_authority",
    "build_identity_bridge_handoff",
    "canonical_sha256",
    "execute",
    "file_sha256",
    "materializer_v3_command",
    "preregister",
    "publish_terminal",
    "retain_unproven_gpu_lock",
    "run_bound_stage",
    "validate_calibration_result",
    "validate_five_members",
    "validate_formal190_receipt",
    "validate_materializer_v3_outputs",
    "validate_member_receipt_v3",
]
