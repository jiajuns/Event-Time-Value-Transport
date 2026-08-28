#!/usr/bin/env python3
"""Autonomous, fail-closed source63-only causal-observer training watcher.

This server-side launcher freezes one content-addressed static plan and runs
exactly three stages: source63 request freezing, actor-visible supervision
materialization, and CUDA observer training.  It may wait for one *exactly*
identified external process to exit, but it never signals or otherwise owns
that process.  The selected RTX 4090 is identified by UUID, observed idle
twice, protected by an exclusive create-once lock, and exposed to the trainer
only through an exact ``CUDA_VISIBLE_DEVICES=<GPU UUID>`` environment.

The launcher is deliberately source-embodiment-only.  Source, split,
event-spec, group-root and output paths containing Piper, evaluation, test,
fresh or confirmation components are rejected before their contents are read.
The terminal summary always states that neither cross-embodiment improvement
nor target-task success is claimed.  A monitor-only training result is an
honest successful terminal result; no authority may exist unless the trainer
itself published a structurally and cryptographically valid promoted bundle.

``detach`` creates the output root and starts ``run`` in a new session.  The
output root and static plan are create-once.  Failed subprocess attempts remain
diagnostic-only under ``attempts``; a later ``run`` can recover by starting a
new attempt while never consuming an incomplete attempt as a stage result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence


FORMAT = "etsf_smolvla_causal_observer_source63_autonomous_launcher_v1"
PLAN_FORMAT = "etsf_smolvla_causal_observer_source63_static_plan_v1"
STATE_FORMAT = "etsf_smolvla_causal_observer_source63_state_v1"
SUMMARY_FORMAT = "etsf_smolvla_causal_observer_source63_summary_v1"
DETACH_FORMAT = "etsf_smolvla_causal_observer_source63_detach_v1"
EXTERNAL_GUARD_FORMAT = "etsf_external_process_identity_guard_v1"
GPU_AUDIT_FORMAT = "etsf_source63_causal_observer_gpu_idle_audit_v1"
GPU_LOCK_FORMAT = "etsf_source63_causal_observer_gpu_lock_v1"
REQUEST_FORMAT = (
    "etsf_smolvla_piper_causal_event_observer_materialization_request_v1"
)
REQUEST_STATUS = "frozen_before_hdf_access"
REQUEST_AUDIT_STATUS = "complete_request_frozen_before_any_hdf_open_test_excluded"
DATASET_FORMAT = "etsf_smolvla_piper_causal_event_observer_dataset_v1"
DATASET_STATUS = "complete_actor_visible_causal_supervision_content_addressed"
FREEZE_FORMAT = "etsf_smolvla_piper_causal_event_observer_freeze_v1"
PROMOTED_STATUS = "frozen_promoted_evaluation400_v4_rerank"
MONITOR_STATUS = "frozen_monitor_only_no_evaluation400_v4_authority"
TERMINAL_STATUS = "complete_source63_causal_observer_training"
FAILURE_STATUS = "failed_closed_source63_causal_observer_training"

FREEZER = "freeze_smolvla_causal_observer_source63_request_v1.py"
MATERIALIZER = "materialize_smolvla_piper_causal_event_observer_dataset_v1.py"
TRAINER = "train_smolvla_piper_causal_event_observer_v1.py"
REQUIRED_CODE_FILES = (
    "launch_smolvla_causal_observer_source63_autonomous_v1.py",
    FREEZER,
    MATERIALIZER,
    TRAINER,
    "smolvla_piper_causal_event_observer_v1.py",
    "etsf_schema6_pose_quality.py",
)
STAGES = ("request", "dataset", "training", "terminal_verification")
SHA_CHARS = frozenset("0123456789abcdef")
PROTECTED_DATA_TOKENS = (
    "piper",
    "evaluation",
    "test",
    "fresh",
    "confirmation",
    "formal_target",
)
IGNORED_CODE_COMPONENTS = frozenset({".git", "__pycache__", ".pytest_cache"})
SCRUBBED_ENV = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PYTHON_EXE",
    }
)

# This bootstrap intentionally uses only the Python standard library.  Under
# ``python -I -c`` no repository directory is importable.  It first verifies
# the signed static plan, reconstructs and authenticates the complete recursive
# code closure, and proves that the requested target is the frozen entrypoint.
# Only after all of those checks pass is the target's sibling directory added
# to ``sys.path`` for the duration of ``runpy.run_path``.
ISOLATED_RUNPY_BOOTSTRAP = r'''
import hashlib
import json
import pathlib
import runpy
import stat
import sys

def canonical_sha256(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if len(sys.argv) < 4:
    raise SystemExit("isolated runpy bootstrap arguments are incomplete")
plan_path = pathlib.Path(sys.argv[1])
expected_plan_sha256 = sys.argv[2]
requested_target = pathlib.Path(sys.argv[3])
if plan_path.is_symlink() or not plan_path.is_file():
    raise SystemExit("isolated runpy static plan is not a regular file")
try:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit("isolated runpy static plan is unreadable") from error
if not isinstance(plan, dict):
    raise SystemExit("isolated runpy static plan is not an object")
logical_plan = dict(plan)
claimed_plan_sha256 = logical_plan.pop("static_plan_sha256", None)
if (
    not isinstance(claimed_plan_sha256, str)
    or claimed_plan_sha256 != expected_plan_sha256
    or canonical_sha256(logical_plan) != claimed_plan_sha256
):
    raise SystemExit("isolated runpy static plan content address changed")
closure = plan.get("code_closure")
if not isinstance(closure, dict):
    raise SystemExit("isolated runpy code closure is missing")
try:
    root = pathlib.Path(closure["root"]).resolve(strict=True)
except (KeyError, OSError) as error:
    raise SystemExit("isolated runpy code root is unavailable") from error
if not root.is_dir() or root.is_symlink():
    raise SystemExit("isolated runpy code root is invalid")
ignored = {".git", "__pycache__", ".pytest_cache"}
observed_files = []
for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
    relative = path.relative_to(root)
    if any(part in ignored for part in relative.parts):
        continue
    if path.is_symlink():
        raise SystemExit("isolated runpy code closure contains a symlink")
    if path.is_dir():
        continue
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("isolated runpy code closure contains a non-regular file")
    observed_files.append(
        {
            "relative_path": relative.as_posix(),
            "size_bytes": metadata.st_size,
            "mode": stat.S_IMODE(metadata.st_mode),
            "file_sha256": file_sha256(path),
        }
    )
inventory = {"root": str(root), "files": observed_files}
if (
    closure.get("files") != observed_files
    or closure.get("file_count") != len(observed_files)
    or closure.get("closure_sha256") != canonical_sha256(inventory)
):
    raise SystemExit("isolated runpy recursive code closure changed")
try:
    target = requested_target.resolve(strict=True)
except OSError as error:
    raise SystemExit("isolated runpy target is unavailable") from error
if target.is_symlink() or not target.is_file() or target.parent != root / "scripts":
    raise SystemExit("isolated runpy target is outside the frozen scripts root")
entrypoint = plan.get("entrypoints", {}).get(target.name)
if (
    not isinstance(entrypoint, dict)
    or entrypoint.get("path") != str(target)
    or entrypoint.get("file_sha256") != file_sha256(target)
):
    raise SystemExit("isolated runpy target entrypoint binding changed")
sys.path.insert(0, str(target.parent))
sys.argv = [str(target), *sys.argv[4:]]
runpy.run_path(str(target), run_name="__main__")
'''.strip()


class LauncherContractError(RuntimeError):
    """An immutable plan, process, GPU, path, or terminal contract failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".partial"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(value), stream, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LauncherContractError(f"{role} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LauncherContractError(f"{role} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise LauncherContractError(f"{role} must be a JSON object")
    return value


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise LauncherContractError(f"{role} path contains a symbolic link")


def resolve_file(raw: Path | str, role: str) -> Path:
    path = Path(raw).expanduser().absolute()
    _reject_symlink_components(path, role)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LauncherContractError(f"{role} does not exist") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise LauncherContractError(f"{role} is not a regular file")
    return resolved


def resolve_directory(raw: Path | str, role: str) -> Path:
    path = Path(raw).expanduser().absolute()
    _reject_symlink_components(path, role)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LauncherContractError(f"{role} does not exist") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise LauncherContractError(f"{role} is not a directory")
    return resolved


def resolve_output(raw: Path | str) -> Path:
    path = Path(raw).expanduser().absolute()
    reject_protected_data_path(path, "output")
    _reject_symlink_components(path.parent, "output parent")
    parent = path.parent.resolve(strict=True)
    return parent / path.name


def reject_protected_data_path(path: Path, role: str) -> None:
    for component in PurePath(path).parts:
        lowered = component.casefold()
        if any(token in lowered for token in PROTECTED_DATA_TOKENS):
            raise LauncherContractError(
                f"{role} contains a protected target/test path component"
            )


def source_data_file(raw: Path | str, expected_sha: str, role: str) -> Path:
    path = Path(raw).expanduser().absolute()
    reject_protected_data_path(path, role)
    resolved = resolve_file(path, role)
    if not is_sha256(expected_sha) or file_sha256(resolved) != expected_sha:
        raise LauncherContractError(f"{role} SHA256 does not match")
    return resolved


def recursive_code_closure(code_root: Path) -> dict[str, Any]:
    """Hash every regular, non-cache file below an immutable code root."""

    root = resolve_directory(code_root, "code root")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if any(part in IGNORED_CODE_COMPONENTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise LauncherContractError("code root closure contains a symbolic link")
        if path.is_dir():
            continue
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise LauncherContractError("code root closure contains a non-regular file")
        files.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
                "file_sha256": file_sha256(path),
            }
        )
    if not files:
        raise LauncherContractError("code root closure is empty")
    inventory = {"root": str(root), "files": files}
    return {
        **inventory,
        "file_count": len(files),
        "closure_sha256": canonical_sha256(inventory),
    }


def verify_code_closure(expected: Mapping[str, Any]) -> None:
    if not isinstance(expected.get("root"), str):
        raise LauncherContractError("code closure root is invalid")
    observed = recursive_code_closure(Path(str(expected["root"])))
    if observed != dict(expected):
        raise LauncherContractError("recursive code-root closure changed")


def _proc_start_ticks(stat_bytes: bytes) -> int:
    try:
        text = stat_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise LauncherContractError("process stat is not ASCII") from error
    close = text.rfind(")")
    if close < 0:
        raise LauncherContractError("process stat is malformed")
    fields = text[close + 2 :].split()
    if len(fields) <= 19:
        raise LauncherContractError("process start ticks are missing")
    try:
        value = int(fields[19])
    except ValueError as error:
        raise LauncherContractError("process start ticks are invalid") from error
    if value <= 0:
        raise LauncherContractError("process start ticks are invalid")
    return value


def read_process_identity(
    pid: int, *, proc_root: Path = Path("/proc")
) -> dict[str, Any] | None:
    if type(pid) is not int or pid <= 0:
        raise LauncherContractError("external PID must be a positive integer")
    process = proc_root / str(pid)
    try:
        stat_bytes = (process / "stat").read_bytes()
        cmdline = (process / "cmdline").read_bytes()
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except FileNotFoundError:
        if not process.exists():
            return None
        raise LauncherContractError("external process identity is partially readable")
    except OSError as error:
        raise LauncherContractError("external process identity cannot be audited") from error
    if not cmdline or not boot_id:
        raise LauncherContractError("external process identity is incomplete")
    return {
        "pid": pid,
        "start_ticks": _proc_start_ticks(stat_bytes),
        "boot_id": boot_id,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        "cmdline_tokens": [
            token.decode("utf-8", errors="surrogateescape")
            for token in cmdline.rstrip(b"\0").split(b"\0")
        ],
    }


def freeze_external_guard(
    args: argparse.Namespace,
    *, identity_reader: Callable[[int], Mapping[str, Any] | None] = read_process_identity,
) -> dict[str, Any]:
    values = (
        args.external_pid,
        args.external_start_ticks,
        args.external_boot_id,
        args.external_cmdline_sha256,
        args.external_script,
        args.external_script_sha256,
    )
    if all(value is None for value in values):
        base = {"format": EXTERNAL_GUARD_FORMAT, "enabled": False}
        return {**base, "guard_sha256": canonical_sha256(base)}
    if any(value is None for value in values):
        raise LauncherContractError("external process guard is all-or-none")
    script = resolve_file(args.external_script, "external process script")
    if not script.is_absolute() or file_sha256(script) != args.external_script_sha256:
        raise LauncherContractError("external process script SHA256 changed")
    observed = identity_reader(args.external_pid)
    if observed is None:
        raise LauncherContractError("external process is absent during plan freeze")
    expected = {
        "pid": args.external_pid,
        "start_ticks": args.external_start_ticks,
        "boot_id": args.external_boot_id,
        "cmdline_sha256": args.external_cmdline_sha256,
    }
    if any(observed.get(key) != value for key, value in expected.items()):
        raise LauncherContractError("external process exact identity differs")
    tokens = observed.get("cmdline_tokens")
    if not isinstance(tokens, list) or tokens.count(str(script)) != 1:
        raise LauncherContractError("external script is not one exact cmdline token")
    base = {
        "format": EXTERNAL_GUARD_FORMAT,
        "enabled": True,
        **expected,
        "script": str(script),
        "script_sha256": args.external_script_sha256,
    }
    return {**base, "guard_sha256": canonical_sha256(base)}


def external_process_alive(
    guard: Mapping[str, Any],
    *, identity_reader: Callable[[int], Mapping[str, Any] | None] = read_process_identity,
) -> bool:
    base = dict(guard)
    claimed = base.pop("guard_sha256", None)
    if not is_sha256(claimed) or claimed != canonical_sha256(base):
        raise LauncherContractError("external guard content address changed")
    if guard.get("enabled") is not True:
        return False
    script = resolve_file(str(guard["script"]), "external process script")
    if file_sha256(script) != guard.get("script_sha256"):
        raise LauncherContractError("external process script changed while waiting")
    identity = identity_reader(int(guard["pid"]))
    if identity is None:
        return False
    for key in ("pid", "start_ticks", "boot_id", "cmdline_sha256"):
        if identity.get(key) != guard.get(key):
            raise LauncherContractError("external PID was reused or identity changed")
    tokens = identity.get("cmdline_tokens")
    if not isinstance(tokens, list) or tokens.count(str(script)) != 1:
        raise LauncherContractError("external process cmdline binding changed")
    return True


def query_gpu_identity(gpu_uuid: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    matches: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[1] == gpu_uuid:
            try:
                index = int(parts[0])
            except ValueError as error:
                raise LauncherContractError("GPU index is invalid") from error
            matches.append({"gpu_index": index, "gpu_uuid": parts[1], "gpu_name": parts[2]})
    if len(matches) != 1 or "4090" not in matches[0]["gpu_name"]:
        raise LauncherContractError("designated GPU UUID is not one RTX 4090")
    return matches[0]


def query_gpu_compute_pids(gpu_uuid: str) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu_uuid}", "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.casefold().startswith("no running"):
            continue
        try:
            pid = int(text)
        except ValueError as error:
            raise LauncherContractError("GPU compute PID output is invalid") from error
        if pid <= 0:
            raise LauncherContractError("GPU compute PID output is invalid")
        pids.append(pid)
    return sorted(set(pids))


def wait_for_external_exit_and_idle_gpu(
    *,
    guard: Mapping[str, Any],
    expected_gpu: Mapping[str, Any],
    timeout_seconds: float,
    poll_seconds: float,
    identity_reader: Callable[[int], Mapping[str, Any] | None] = read_process_identity,
    gpu_identity_reader: Callable[[str], Mapping[str, Any]] = query_gpu_identity,
    gpu_pid_reader: Callable[[str], Sequence[int]] = query_gpu_compute_pids,
    sleeper: Callable[[float], None] = time.sleep,
    state_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise LauncherContractError("GPU wait intervals are invalid")
    started = time.monotonic()
    checks = 0
    idle_streak = 0
    parent_gone_latched = guard.get("enabled") is not True
    observations: list[dict[str, Any]] = []
    while True:
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError("timed out waiting for external exit and idle RTX 4090")
        alive = external_process_alive(guard, identity_reader=identity_reader)
        if parent_gone_latched and alive:
            raise LauncherContractError("external process reappeared after exact exit")
        if alive:
            idle_streak = 0
            status = "waiting_for_exact_external_process_exit"
            pids: list[int] | None = None
        else:
            parent_gone_latched = True
            observed_gpu = dict(gpu_identity_reader(str(expected_gpu["gpu_uuid"])))
            if observed_gpu != dict(expected_gpu):
                raise LauncherContractError("designated GPU identity changed")
            pids = list(gpu_pid_reader(str(expected_gpu["gpu_uuid"])))
            idle_streak = idle_streak + 1 if not pids else 0
            status = "waiting_for_two_consecutive_idle_gpu_samples"
        checks += 1
        observation = {
            "check": checks,
            "external_process_alive": alive,
            "external_process_gone_latched": parent_gone_latched,
            "compute_pids": pids,
            "idle_streak": idle_streak,
        }
        observations.append(observation)
        partial = {
            "format": GPU_AUDIT_FORMAT,
            "status": status,
            "gpu_identity": dict(expected_gpu),
            "checks": checks,
            "idle_confirmations_required": 2,
            "idle_confirmations": idle_streak,
            "observations": observations,
            "external_guard_sha256": guard["guard_sha256"],
        }
        if state_callback is not None:
            state_callback(partial)
        if idle_streak == 2:
            base = {**partial, "status": "complete_external_gone_and_gpu_idle"}
            return {**base, "audit_sha256": canonical_sha256(base)}
        sleeper(poll_seconds)


def acquire_gpu_lock(path: Path, plan_sha: str, gpu: Mapping[str, Any]) -> dict[str, Any]:
    lock = path.expanduser().absolute()
    _reject_symlink_components(lock.parent, "GPU lock parent")
    lock.parent.resolve(strict=True)
    token = uuid.uuid4().hex
    base = {
        "format": GPU_LOCK_FORMAT,
        "pid": os.getpid(),
        "token": token,
        "static_plan_sha256": plan_sha,
        "gpu_identity": dict(gpu),
    }
    payload = {**base, "lock_sha256": canonical_sha256(base)}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as error:
        raise LauncherContractError("designated GPU lock already exists") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        lock.unlink(missing_ok=True)
        raise
    return {"path": str(lock), **payload}


def release_gpu_lock(lock: Mapping[str, Any]) -> None:
    path = resolve_file(str(lock["path"]), "owned GPU lock")
    observed = _load_json(path, "owned GPU lock")
    expected = {key: value for key, value in lock.items() if key != "path"}
    if observed != expected or observed.get("pid") != os.getpid():
        raise LauncherContractError("GPU lock ownership/content changed")
    path.unlink()


def canonical_cuda_environment(base: Mapping[str, str], gpu_uuid: str) -> dict[str, str]:
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise LauncherContractError("CUDA GPU UUID is invalid")
    env = {key: value for key, value in base.items() if key not in SCRUBBED_ENV}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def cpu_stage_environment(base: Mapping[str, str]) -> dict[str, str]:
    env = {key: value for key, value in base.items() if key not in SCRUBBED_ENV}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _signed_document(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise LauncherContractError(f"document already contains {field}")
    return {**dict(value), field: canonical_sha256(value)}


def _verify_signed(value: Mapping[str, Any], field: str, role: str) -> None:
    base = dict(value)
    claimed = base.pop(field, None)
    if not is_sha256(claimed) or claimed != canonical_sha256(base):
        raise LauncherContractError(f"{role} content address is invalid")


def _entrypoints(code_root: Path) -> dict[str, Path]:
    scripts = code_root / "scripts"
    result: dict[str, Path] = {}
    for name in REQUIRED_CODE_FILES:
        path = resolve_file(scripts / name, f"code entrypoint {name}")
        result[name] = path
    return result


def _assert_server(required_hostname: str) -> None:
    if not required_hostname or socket.gethostname() != required_hostname:
        raise LauncherContractError("launcher is not running on the frozen server hostname")


def build_static_plan(
    args: argparse.Namespace,
    *,
    identity_reader: Callable[[int], Mapping[str, Any] | None] = read_process_identity,
    gpu_identity_reader: Callable[[str], Mapping[str, Any]] = query_gpu_identity,
) -> dict[str, Any]:
    _assert_server(args.required_hostname)
    output = resolve_output(args.output)
    code_root = resolve_directory(args.code_root, "code root")
    python = resolve_file(args.python, "server Python")
    manifest = source_data_file(
        args.schema5_manifest, args.schema5_manifest_sha256, "schema5 source manifest"
    )
    split = source_data_file(args.frozen_split, args.frozen_split_sha256, "frozen split")
    event_spec = source_data_file(args.event_spec, args.event_spec_sha256, "event spec")
    group_root: Path | None = None
    if args.group_root is not None:
        raw_group_root = Path(args.group_root).expanduser().absolute()
        reject_protected_data_path(raw_group_root, "source group root")
        group_root = resolve_directory(raw_group_root, "source group root")
    if not isinstance(args.gpu_uuid, str) or not args.gpu_uuid.startswith("GPU-"):
        raise LauncherContractError("designated GPU UUID is invalid")
    gpu = dict(gpu_identity_reader(args.gpu_uuid))
    if gpu.get("gpu_uuid") != args.gpu_uuid or "4090" not in str(gpu.get("gpu_name")):
        raise LauncherContractError("designated GPU is not the requested RTX 4090 UUID")
    entrypoints = _entrypoints(code_root)
    closure = recursive_code_closure(code_root)
    guard = freeze_external_guard(args, identity_reader=identity_reader)
    base: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "status": "frozen_create_once_source63_only_plan",
        "server_hostname": args.required_hostname,
        "python": {"path": str(python), "file_sha256": file_sha256(python)},
        "code_closure": closure,
        "entrypoints": {
            name: {"path": str(path), "file_sha256": file_sha256(path)}
            for name, path in sorted(entrypoints.items())
        },
        "source_inputs": {
            "schema5_manifest": {"path": str(manifest), "file_sha256": args.schema5_manifest_sha256},
            "frozen_split": {"path": str(split), "file_sha256": args.frozen_split_sha256},
            "event_spec": {"path": str(event_spec), "file_sha256": args.event_spec_sha256},
            "group_root": None if group_root is None else str(group_root),
        },
        "source_identity": {
            "source_name": args.source_name,
            "actor_name": args.actor_name,
            "policy_family": args.policy_family,
            "calibration_count": args.calibration_count,
            "source_embodiment_only": True,
        },
        "output": str(output),
        "gpu_identity": gpu,
        "gpu_lock": str(Path(args.gpu_lock).expanduser().absolute()),
        "external_guard": guard,
        "training": {
            "hidden_dim": args.hidden_dim,
            "adapter_rank": args.adapter_rank,
            "epochs": args.epochs,
            "batch_size_per_actor": args.batch_size_per_actor,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "device": "cuda",
        },
        "execution_order": list(STAGES),
        "claims": {
            "cross_embodiment_claimed": False,
            "target_task_success_claimed": False,
        },
        "protected_target_paths_allowed": False,
    }
    return _signed_document(base, "static_plan_sha256")


def validate_static_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    _verify_signed(plan, "static_plan_sha256", "static plan")
    if (
        plan.get("format") != PLAN_FORMAT
        or plan.get("status") != "frozen_create_once_source63_only_plan"
        or plan.get("execution_order") != list(STAGES)
        or plan.get("protected_target_paths_allowed") is not False
        or plan.get("claims")
        != {"cross_embodiment_claimed": False, "target_task_success_claimed": False}
        or plan.get("source_identity", {}).get("source_embodiment_only") is not True
        or plan.get("training", {}).get("device") != "cuda"
    ):
        raise LauncherContractError("static plan source-only semantics changed")
    return dict(plan)


def initialize_output(plan: Mapping[str, Any]) -> Path:
    output = Path(str(plan["output"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-once output already exists: {output}")
    output.mkdir(mode=0o700)
    (output / "attempts").mkdir(mode=0o700)
    _atomic_json(output / "static_plan.json", plan)
    state = _signed_document(
        {
            "format": STATE_FORMAT,
            "status": "initialized_recoverable",
            "phase": "request",
            "static_plan_sha256": plan["static_plan_sha256"],
            "completed_stages": [],
            "active_child": None,
            "recoverable": True,
            "cross_embodiment_claimed": False,
            "target_task_success_claimed": False,
        },
        "state_sha256",
    )
    _atomic_json(output / "state.json", state)
    return output


def load_frozen_plan(output: Path, expected_sha: str | None = None) -> dict[str, Any]:
    plan = validate_static_plan(_load_json(output / "static_plan.json", "static plan"))
    if expected_sha is not None and plan["static_plan_sha256"] != expected_sha:
        raise LauncherContractError("expected static-plan SHA256 differs")
    if Path(str(plan["output"])) != output.resolve():
        raise LauncherContractError("static plan output binding changed")
    _assert_server(str(plan["server_hostname"]))
    verify_code_closure(plan["code_closure"])
    python = resolve_file(str(plan["python"]["path"]), "server Python")
    if file_sha256(python) != plan["python"]["file_sha256"]:
        raise LauncherContractError("server Python changed")
    for role, record in plan["source_inputs"].items():
        if role == "group_root" or record is None:
            continue
        source_data_file(Path(record["path"]), record["file_sha256"], role)
    return plan


def isolated_runpy_prefix(plan: Mapping[str, Any], entrypoint_name: str) -> list[str]:
    entrypoint = plan.get("entrypoints", {}).get(entrypoint_name)
    if (
        not isinstance(entrypoint, Mapping)
        or not isinstance(entrypoint.get("path"), str)
        or not is_sha256(entrypoint.get("file_sha256"))
    ):
        raise LauncherContractError("isolated stage entrypoint is not frozen")
    target = Path(str(entrypoint["path"])).resolve(strict=True)
    code_root = Path(str(plan["code_closure"]["root"])).resolve(strict=True)
    if target.parent != code_root / "scripts" or file_sha256(target) != entrypoint["file_sha256"]:
        raise LauncherContractError("isolated stage entrypoint binding changed")
    static_plan_path = Path(str(plan["output"])) / "static_plan.json"
    frozen_plan = _load_json(static_plan_path, "isolated stage static plan")
    if frozen_plan != dict(plan):
        raise LauncherContractError("isolated stage static plan file changed")
    return [
        str(plan["python"]["path"]),
        "-I",
        "-c",
        ISOLATED_RUNPY_BOOTSTRAP,
        str(static_plan_path),
        str(plan["static_plan_sha256"]),
        str(target),
    ]


def build_stage_commands(plan: Mapping[str, Any], attempt_output: Path | None = None) -> dict[str, list[str]]:
    sources = plan["source_inputs"]
    identity = plan["source_identity"]
    output = Path(str(plan["output"]))
    request_path = output / "materialization_request.json"
    dataset_output = attempt_output or output / "dataset"
    training_output = attempt_output or output / "training"
    freezer = [
        *isolated_runpy_prefix(plan, FREEZER),
        "--schema5-manifest", sources["schema5_manifest"]["path"],
        "--schema5-manifest-sha256", sources["schema5_manifest"]["file_sha256"],
        "--frozen-split", sources["frozen_split"]["path"],
        "--frozen-split-sha256", sources["frozen_split"]["file_sha256"],
        "--event-spec", sources["event_spec"]["path"],
        "--event-spec-sha256", sources["event_spec"]["file_sha256"],
        "--output", str(request_path),
        "--calibration-count", str(identity["calibration_count"]),
        "--source-name", identity["source_name"],
        "--actor-name", identity["actor_name"],
        "--policy-family", identity["policy_family"],
    ]
    if sources["group_root"] is not None:
        freezer += ["--group-root", sources["group_root"]]
    training = plan["training"]
    return {
        "request": freezer,
        "dataset": [
            *isolated_runpy_prefix(plan, MATERIALIZER),
            "--request", str(request_path),
            "--output-directory", str(dataset_output),
        ],
        "training": [
            *isolated_runpy_prefix(plan, TRAINER),
            "--dataset-manifest", str(output / "dataset" / "manifest.json"),
            "--output", str(training_output),
            "--hidden-dim", str(training["hidden_dim"]),
            "--adapter-rank", str(training["adapter_rank"]),
            "--epochs", str(training["epochs"]),
            "--batch-size-per-actor", str(training["batch_size_per_actor"]),
            "--learning-rate", str(training["learning_rate"]),
            "--weight-decay", str(training["weight_decay"]),
            "--bootstrap-samples", str(training["bootstrap_samples"]),
            "--seed", str(training["seed"]),
            "--device", "cuda",
        ],
    }


def _validate_request(plan: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(plan["output"])) / "materialization_request.json"
    value = _load_json(path, "materialization request")
    if value.get("format") != REQUEST_FORMAT or value.get("status") != REQUEST_STATUS:
        raise LauncherContractError("materialization request contract changed")
    logical = dict(value)
    request_sha = logical.pop("request_sha256", None)
    if not is_sha256(request_sha) or request_sha != canonical_sha256(logical):
        raise LauncherContractError("materialization request logical SHA is invalid")
    audit_path = Path(str(path) + ".audit.json")
    audit = _load_json(audit_path, "materialization request audit")
    if audit.get("status") != REQUEST_AUDIT_STATUS:
        raise LauncherContractError("materialization request audit is not terminal")
    _verify_signed(audit, "audit_sha256", "materialization request audit")
    raw = canonical_bytes(value)
    # The request format contains "piper" for historical compatibility.  No
    # selected source or group path may contain target/test path components.
    for source in value.get("sources", []):
        if isinstance(source, Mapping):
            for key in ("manifest_path", "group_root"):
                raw_path = source.get(key)
                if isinstance(raw_path, str):
                    reject_protected_data_path(Path(raw_path), f"request source {key}")
    actors = value.get("actors")
    sources = value.get("sources")
    expected_identity = plan["source_identity"]
    expected_manifest = plan["source_inputs"]["schema5_manifest"]
    if (
        not isinstance(actors, list)
        or len(actors) != 1
        or actors[0].get("actor_name") != expected_identity["actor_name"]
        or actors[0].get("policy_family") != expected_identity["policy_family"]
        or not isinstance(sources, list)
        or len(sources) != 1
        or sources[0].get("source_name") != expected_identity["source_name"]
        or sources[0].get("schema_version") != 5
        or sources[0].get("manifest_path") != expected_manifest["path"]
        or sources[0].get("manifest_file_sha256")
        != expected_manifest["file_sha256"]
    ):
        raise LauncherContractError("request is not the frozen source63-only actor/source")
    audit_request = audit.get("request")
    if (
        not isinstance(audit_request, Mapping)
        or audit_request.get("path") != str(path)
        or audit_request.get("file_sha256") != file_sha256(path)
        or audit_request.get("request_sha256") != request_sha
        or audit.get("data_access_audit", {}).get(
            "original_test_groups_excluded_from_all_request_splits"
        )
        is not True
        or audit.get("data_access_audit", {}).get("original_test_group_files_opened")
        != 0
        or audit.get("data_access_audit", {}).get("original_test_group_files_hashed")
        != 0
    ):
        raise LauncherContractError("request audit does not prove zero-contact test exclusion")
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "logical_sha256": request_sha,
        "audit_path": str(audit_path),
        "audit_file_sha256": file_sha256(audit_path),
    }


def _validate_dataset(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    value = _load_json(manifest_path, "observer dataset manifest")
    if value.get("format") != DATASET_FORMAT or value.get("status") != DATASET_STATUS:
        raise LauncherContractError("observer dataset is not terminal")
    base = dict(value)
    claimed = base.pop("manifest_sha256", None)
    if not is_sha256(claimed) or claimed != canonical_sha256(base):
        raise LauncherContractError("observer dataset logical SHA is invalid")
    registry = value.get("actor_registry")
    if not isinstance(registry, list) or len(registry) != 1:
        raise LauncherContractError("dataset is not source-embodiment-only")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "calibration", "validation"}:
        raise LauncherContractError("dataset split inventory changed")
    files = [manifest_path]
    for name in ("train", "calibration", "validation"):
        record = splits[name]
        path = resolve_file(directory / str(record["path"]), f"dataset {name}")
        if file_sha256(path) != record.get("file_sha256"):
            raise LauncherContractError(f"dataset {name} file SHA changed")
        files.append(path)
    return {
        "path": str(manifest_path),
        "file_sha256": file_sha256(manifest_path),
        "logical_sha256": claimed,
        "actor_registry": registry,
        "artifact_set_sha256": canonical_sha256(
            [{"name": path.name, "file_sha256": file_sha256(path)} for path in files]
        ),
    }


def _validate_training(directory: Path) -> dict[str, Any]:
    monitor_path = directory / "monitor_freeze_manifest.json"
    monitor = _load_json(monitor_path, "monitor freeze manifest")
    if monitor.get("format") != FREEZE_FORMAT or monitor.get("status") not in {
        MONITOR_STATUS, PROMOTED_STATUS,
    }:
        raise LauncherContractError("training freeze is not terminal")
    _verify_signed(monitor, "freeze_manifest_sha256", "monitor freeze manifest")
    if monitor.get("real_task_success_or_cross_embodiment_improvement_claimed") is not False:
        raise LauncherContractError("training bundle makes an unsupported claim")
    promoted = monitor.get("status") == PROMOTED_STATUS
    authority_path = directory / "authority_manifest.json"
    if promoted != authority_path.is_file():
        raise LauncherContractError("promotion and authority publication disagree")
    authority: dict[str, Any] | None = None
    if authority_path.is_file():
        authority = _load_json(authority_path, "observer authority manifest")
        _verify_signed(authority, "authority_manifest_sha256", "observer authority")
        if authority.get("status") != PROMOTED_STATUS:
            raise LauncherContractError("observer authority status changed")
    return {
        "path": str(monitor_path),
        "file_sha256": file_sha256(monitor_path),
        "logical_sha256": monitor["freeze_manifest_sha256"],
        "promotion_enabled": promoted,
        "authority_issued": authority is not None,
        "authority_file_sha256": None if authority is None else file_sha256(authority_path),
    }


def _write_state(output: Path, base: Mapping[str, Any]) -> dict[str, Any]:
    state = _signed_document(
        {
            "format": STATE_FORMAT,
            **dict(base),
            "cross_embodiment_claimed": False,
            "target_task_success_claimed": False,
        },
        "state_sha256",
    )
    _atomic_json(output / "state.json", state)
    return state


def _run_command(
    command: Sequence[str], *, environment: Mapping[str, str], log_path: Path
) -> dict[str, Any]:
    with log_path.open("xb") as stream:
        process = subprocess.Popen(
            list(command), stdout=stream, stderr=subprocess.STDOUT,
            env=dict(environment), start_new_session=False,
        )
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(command))
    return {"returncode": returncode, "log_path": str(log_path), "log_sha256": file_sha256(log_path)}


def execute_plan(
    plan: Mapping[str, Any],
    *,
    command_runner: Callable[..., Mapping[str, Any]] = _run_command,
    idle_waiter: Callable[..., Mapping[str, Any]] = wait_for_external_exit_and_idle_gpu,
) -> dict[str, Any]:
    plan = validate_static_plan(plan)
    output = Path(str(plan["output"]))
    plan_sha = str(plan["static_plan_sha256"])
    completed: list[str] = []
    lock: dict[str, Any] | None = None
    try:
        _write_state(output, {"status": "running", "phase": "request", "static_plan_sha256": plan_sha, "completed_stages": completed, "active_child": None, "recoverable": True})
        try:
            request_record = _validate_request(plan)
        except LauncherContractError:
            command = build_stage_commands(plan)["request"]
            attempt = output / "attempts" / f"request-{uuid.uuid4().hex}"
            attempt.mkdir()
            command_runner(command, environment=cpu_stage_environment(os.environ), log_path=attempt / "stage.log")
            request_record = _validate_request(plan)
        completed.append("request")

        _write_state(output, {"status": "running", "phase": "dataset", "static_plan_sha256": plan_sha, "completed_stages": completed, "active_child": None, "recoverable": True})
        dataset_final = output / "dataset"
        try:
            dataset_record = _validate_dataset(dataset_final)
        except LauncherContractError:
            attempt = output / "attempts" / f"dataset-{uuid.uuid4().hex}"
            command = build_stage_commands(plan, attempt_output=attempt)["dataset"]
            command_runner(command, environment=cpu_stage_environment(os.environ), log_path=output / "attempts" / f"dataset-log-{uuid.uuid4().hex}.log")
            dataset_record = _validate_dataset(attempt)
            if dataset_final.exists() or dataset_final.is_symlink():
                raise LauncherContractError("incomplete final dataset blocks recovery")
            os.replace(attempt, dataset_final)
            dataset_record = _validate_dataset(dataset_final)
        completed.append("dataset")

        _write_state(output, {"status": "waiting", "phase": "external_exit_and_gpu_idle", "static_plan_sha256": plan_sha, "completed_stages": completed, "active_child": None, "recoverable": True})
        audit = dict(
            idle_waiter(
                guard=plan["external_guard"],
                expected_gpu=plan["gpu_identity"],
                timeout_seconds=float(plan.get("gpu_wait_timeout_seconds", 86400.0)),
                poll_seconds=float(plan.get("gpu_poll_seconds", 10.0)),
            )
        )
        _atomic_json(output / "gpu_idle_audit.json", audit)
        lock = acquire_gpu_lock(Path(str(plan["gpu_lock"])), plan_sha, plan["gpu_identity"])
        if dict(query_gpu_identity(str(plan["gpu_identity"]["gpu_uuid"]))) != dict(plan["gpu_identity"]):
            raise LauncherContractError("GPU identity changed after lock acquisition")
        if query_gpu_compute_pids(str(plan["gpu_identity"]["gpu_uuid"])):
            raise LauncherContractError("GPU became busy after lock acquisition")

        _write_state(output, {"status": "running", "phase": "training", "static_plan_sha256": plan_sha, "completed_stages": completed, "active_child": None, "recoverable": True, "gpu_lock_sha256": lock["lock_sha256"]})
        training_final = output / "training"
        try:
            training_record = _validate_training(training_final)
        except LauncherContractError:
            attempt = output / "attempts" / f"training-{uuid.uuid4().hex}"
            command = build_stage_commands(plan, attempt_output=attempt)["training"]
            command_runner(
                command,
                environment=canonical_cuda_environment(
                    os.environ, str(plan["gpu_identity"]["gpu_uuid"])
                ),
                log_path=output / "attempts" / f"training-log-{uuid.uuid4().hex}.log",
            )
            training_record = _validate_training(attempt)
            if training_final.exists() or training_final.is_symlink():
                raise LauncherContractError("incomplete final training blocks recovery")
            os.replace(attempt, training_final)
            training_record = _validate_training(training_final)
        completed.append("training")
        release_gpu_lock(lock)
        lock = None

        summary_base = {
            "format": SUMMARY_FORMAT,
            "status": TERMINAL_STATUS,
            "static_plan_sha256": plan_sha,
            "execution_order": list(STAGES),
            "completed_stages": [*completed, "terminal_verification"],
            "source_embodiment_only": True,
            "source_identity": plan["source_identity"],
            "request": request_record,
            "dataset": dataset_record,
            "training": training_record,
            "gpu_idle_audit_sha256": audit.get("audit_sha256"),
            "cross_embodiment_claimed": False,
            "target_task_success_claimed": False,
            "target_paths_consumed": False,
        }
        summary = _signed_document(summary_base, "summary_sha256")
        _atomic_json(output / "summary.json", summary)
        _write_state(output, {"status": TERMINAL_STATUS, "phase": "complete", "static_plan_sha256": plan_sha, "completed_stages": summary["completed_stages"], "active_child": None, "recoverable": False, "summary_sha256": summary["summary_sha256"]})
        return summary
    except BaseException as error:
        lock_retained = False
        release_error: str | None = None
        if lock is not None:
            try:
                release_gpu_lock(lock)
            except BaseException as nested:
                lock_retained = True
                release_error = f"{type(nested).__name__}: {nested}"
        _write_state(
            output,
            {
                "status": FAILURE_STATUS,
                "phase": "failed",
                "static_plan_sha256": plan_sha,
                "completed_stages": completed,
                "active_child": None,
                "recoverable": not lock_retained,
                "error_type": type(error).__name__,
                "error": str(error),
                "gpu_lock_retained": lock_retained,
                "gpu_lock_release_error": release_error,
            },
        )
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve_output(args.output)
    if output.exists():
        plan = load_frozen_plan(output, args.expected_static_plan_sha256)
    else:
        plan = build_static_plan(args)
        if args.expected_static_plan_sha256 is not None and plan["static_plan_sha256"] != args.expected_static_plan_sha256:
            raise LauncherContractError("fresh static-plan SHA differs from expected")
        initialize_output(plan)
    return execute_plan(plan)


def _run_argv(args: argparse.Namespace, plan: Mapping[str, Any]) -> list[str]:
    script = plan["entrypoints"][
        "launch_smolvla_causal_observer_source63_autonomous_v1.py"
    ]["path"]
    return [
        str(plan["python"]["path"]), "-I", str(script), "run",
        "--output", str(plan["output"]),
        "--code-root", str(args.code_root),
        "--python", str(args.python),
        "--required-hostname", str(args.required_hostname),
        "--schema5-manifest", str(args.schema5_manifest),
        "--schema5-manifest-sha256", str(args.schema5_manifest_sha256),
        "--frozen-split", str(args.frozen_split),
        "--frozen-split-sha256", str(args.frozen_split_sha256),
        "--event-spec", str(args.event_spec),
        "--event-spec-sha256", str(args.event_spec_sha256),
        "--gpu-uuid", str(args.gpu_uuid),
        "--gpu-lock", str(args.gpu_lock),
        "--expected-static-plan-sha256", str(plan["static_plan_sha256"]),
    ]


def detach(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_static_plan(args)
    output = initialize_output(plan)
    command = _run_argv(args, plan)
    log_path = output / "watcher.log"
    with log_path.open("xb") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=cpu_stage_environment(os.environ),
            start_new_session=True,
        )
    receipt = _signed_document(
        {
            "format": DETACH_FORMAT,
            "status": "detached_server_side_watcher_started",
            "pid": process.pid,
            "static_plan_sha256": plan["static_plan_sha256"],
            "output": str(output),
            "log": str(log_path),
            "cross_embodiment_claimed": False,
            "target_task_success_claimed": False,
        },
        "detach_sha256",
    )
    _atomic_json(output / "detach_receipt.json", receipt)
    return receipt


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--required-hostname", required=True)
    parser.add_argument("--schema5-manifest", required=True, type=Path)
    parser.add_argument("--schema5-manifest-sha256", required=True)
    parser.add_argument("--frozen-split", required=True, type=Path)
    parser.add_argument("--frozen-split-sha256", required=True)
    parser.add_argument("--event-spec", required=True, type=Path)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--group-root", type=Path)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--source-name", default="smolvla_source63")
    parser.add_argument("--actor-name", default="smolvla_aloha_agilex")
    parser.add_argument("--policy-family", default="smolvla")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-lock", required=True, type=Path)
    parser.add_argument("--external-pid", type=int)
    parser.add_argument("--external-start-ticks", type=int)
    parser.add_argument("--external-boot-id")
    parser.add_argument("--external-cmdline-sha256")
    parser.add_argument("--external-script", type=Path)
    parser.add_argument("--external-script-sha256")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size-per-actor", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--expected-static-plan-sha256")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "detach"):
        child = subparsers.add_parser(name)
        add_common(child)
    args = parser.parse_args(argv)
    numeric_positive = (
        args.calibration_count,
        args.hidden_dim,
        args.adapter_rank,
        args.epochs,
        args.batch_size_per_actor,
        args.bootstrap_samples,
    )
    if any(type(value) is not int or value <= 0 for value in numeric_positive):
        parser.error("all count/dimension arguments must be positive integers")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("optimizer arguments are invalid")
    if args.expected_static_plan_sha256 is not None and not is_sha256(args.expected_static_plan_sha256):
        parser.error("expected static plan SHA256 is invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = detach(args) if args.command == "detach" else run(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LauncherContractError",
    "acquire_gpu_lock",
    "build_stage_commands",
    "build_static_plan",
    "canonical_cuda_environment",
    "canonical_sha256",
    "detach",
    "execute_plan",
    "external_process_alive",
    "file_sha256",
    "freeze_external_guard",
    "read_process_identity",
    "recursive_code_closure",
    "reject_protected_data_path",
    "release_gpu_lock",
    "validate_static_plan",
    "wait_for_external_exit_and_idle_gpu",
]
