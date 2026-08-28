#!/usr/bin/env python3
"""Detached, fail-closed launcher for the paired Piper development experiment.

This watcher never opens HDF5 and never receives sealed-reserve identities.  It
accepts one already frozen development protocol, revalidates its upstream
LOBO/schema6/adapter receipts and all bound artifacts, claims a new output root
before detaching, waits for an exclusively idle RTX 4090, and invokes one exact
executor.  The executor must publish per-pair *executed simulator task-success*
JSON records.  Prediction metrics cannot substitute for those outcomes.
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

from smolvla_piper_paired_success_protocol import (
    ACTOR_ID,
    BODY,
    PAIR_RESULT_FORMAT,
    TASK,
    PairedSuccessProtocolError,
    canonical_sha256,
    evaluate_pair_results,
    file_sha256,
    validate_dependency_authority,
    validate_protocol,
    verify_signed,
)


FORMAT = "etsf_smolvla_piper_paired_success_development_watcher_v1"
PLAN_FORMAT = "etsf_smolvla_piper_paired_success_development_plan_v1"
STATE_FORMAT = "etsf_smolvla_piper_paired_success_development_state_v1"
DETACH_FORMAT = "etsf_smolvla_piper_paired_success_development_detach_v1"
MANIFEST_FORMAT = "etsf_smolvla_piper_paired_task_success_collection_v1"
MANIFEST_STATUS = "complete_development_executed_branch_task_success"
TERMINAL_PASSED = "complete_development_task_success_gate_passed"
TERMINAL_NULL = "complete_development_task_success_gate_not_passed"
FAILURE_STATUS = "failed_closed_development_execution"
EXPECTED_GPU_FRAGMENT = "RTX 4090"
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
HDF_SUFFIXES = (".h5", ".hdf", ".hdf5")
SHA_CHARS = frozenset("0123456789abcdef")


class WatcherError(RuntimeError):
    """The preregistered watcher contract cannot be proved."""


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _sensitive(path: PurePath) -> bool:
    return any(t in part.casefold() for part in path.parts for t in SENSITIVE_PATH_TOKENS)


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise WatcherError(f"{role} path is invalid")
    path = Path(os.path.abspath(os.path.expanduser(text)))
    if _sensitive(PurePath(path)):
        raise WatcherError(f"{role} path is in a forbidden namespace")
    return path


def existing_file(value: str | os.PathLike[str], role: str) -> Path:
    path = safe_path(value, role)
    if path.is_symlink():
        raise WatcherError(f"{role} must not be a symlink")
    resolved = path.resolve(strict=True)
    if _sensitive(PurePath(resolved)) or not stat.S_ISREG(resolved.stat().st_mode):
        raise WatcherError(f"{role} is not a safe regular file")
    if resolved.suffix.casefold() in HDF_SUFFIXES:
        raise WatcherError("watcher is forbidden from opening HDF5")
    return resolved


def new_root(value: str | os.PathLike[str]) -> Path:
    path = safe_path(value, "output root")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    parent = path.parent.resolve(strict=True)
    if _sensitive(PurePath(parent)) or not parent.is_dir():
        raise WatcherError("output parent is invalid")
    return path


def load_json(path: Path, role: str) -> dict[str, Any]:
    path = existing_file(path, role)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WatcherError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise WatcherError(f"{role} must be an object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def immutable_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def freeze_tree_read_only(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise WatcherError("output tree contains a symlink")
            item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise WatcherError("output tree contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def revalidate_protocol(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    path = existing_file(path, "paired protocol")
    if not _is_sha(expected_file_sha256) or file_sha256(path) != expected_file_sha256:
        raise WatcherError("paired protocol file SHA mismatch")
    protocol = load_json(path, "paired protocol")
    logical = validate_protocol(protocol)
    if (
        protocol.get("scope", {}).get("task") != TASK
        or protocol.get("scope", {}).get("body") != BODY
        or protocol.get("scope", {}).get("actor_id") != ACTOR_ID
        or protocol.get("scope", {}).get("development_only") is not True
        or protocol.get("sealed_evaluation_reserve", {}).get("execution_authorized") is not False
    ):
        raise WatcherError("protocol scope changed")
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise WatcherError("protocol artifact inventory is missing")
    for name, row in artifacts.items():
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise WatcherError(f"artifact contract changed: {name}")
        artifact = existing_file(str(row["path"]), f"artifact {name}")
        if not _is_sha(row["sha256"]) or file_sha256(artifact) != row["sha256"]:
            raise WatcherError(f"artifact SHA mismatch: {name}")
    dependency_path = Path(str(artifacts["dependency_authority"]["path"]))
    current_dependencies = validate_dependency_authority(
        load_json(dependency_path, "dependency authority")
    )
    if current_dependencies != protocol.get("dependencies"):
        raise WatcherError("upstream LOBO/schema6/adapter dependency changed")
    return {
        "path": str(path),
        "file_sha256": expected_file_sha256,
        "protocol_sha256": logical,
        "protocol": protocol,
    }


def _run_text(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def gpu_audit(gpu_index: int, run_text: Callable[[Sequence[str]], str] = _run_text) -> dict[str, Any]:
    identity = run_text(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name,uuid", "--format=csv,noheader"]
    ).strip().split(",", 1)
    if len(identity) != 2 or EXPECTED_GPU_FRAGMENT not in identity[0].strip():
        raise WatcherError("designated GPU is not an RTX 4090")
    raw = run_text(
        ["nvidia-smi", f"--id={gpu_index}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    )
    pids = sorted({int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()})
    return {"gpu_index": gpu_index, "name": identity[0].strip(), "uuid": identity[1].strip(), "compute_pids": pids}


def wait_two_idle(
    gpu_index: int,
    *,
    interval: float,
    timeout: float,
    run_text: Callable[[Sequence[str]], str] = _run_text,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    consecutive: list[dict[str, Any]] = []
    while time.monotonic() <= deadline:
        audit = gpu_audit(gpu_index, run_text)
        if audit["compute_pids"]:
            consecutive.clear()
        else:
            if consecutive and audit["uuid"] != consecutive[0]["uuid"]:
                raise WatcherError("GPU identity changed while waiting")
            consecutive.append(audit)
            if len(consecutive) == 2:
                return consecutive
        sleep(interval)
    raise TimeoutError("timed out waiting for two consecutive idle GPU audits")


def _is_descendant(pid: int, ancestor: int) -> bool:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            pid = int(fields[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def validate_collection_manifest(root: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "development_collection_manifest.json"
    manifest = load_json(manifest_path, "development collection manifest")
    verify_signed(manifest, "manifest_sha256", "development collection manifest")
    expected_ids = [row["pair_id"] for row in protocol["development_pairs"]]
    rows = manifest.get("pairs")
    exact_root = {
        "format", "status", "protocol_sha256", "lane", "task", "body", "actor_id",
        "pair_count", "pairs", "task_success_source", "predicted_success_used_as_outcome",
        "sealed_evaluation_reserve_executed", "reserve_identities_read", "reserve_outcomes_read",
        "existing_sensitive_artifacts_read", "test_hdf5_opened", "manifest_sha256",
    }
    if (
        set(manifest) != exact_root
        or manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("status") != MANIFEST_STATUS
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("lane") != "development"
        or (manifest.get("task"), manifest.get("body"), manifest.get("actor_id")) != (TASK, BODY, ACTOR_ID)
        or manifest.get("pair_count") != len(expected_ids)
        or not isinstance(rows, list)
        or [row.get("pair_id") for row in rows if isinstance(row, Mapping)] != expected_ids
        or manifest.get("task_success_source") != "simulator_info_success_from_executed_schema6_branch"
        or manifest.get("predicted_success_used_as_outcome") is not False
        or manifest.get("sealed_evaluation_reserve_executed") is not False
        or manifest.get("reserve_identities_read") is not False
        or manifest.get("reserve_outcomes_read") is not False
        or manifest.get("existing_sensitive_artifacts_read") is not False
        or manifest.get("test_hdf5_opened") != 0
    ):
        raise WatcherError("development collection manifest contract changed")
    results: list[dict[str, Any]] = []
    for expected_id, row in zip(expected_ids, rows):
        if not isinstance(row, Mapping) or set(row) != {
            "pair_id", "pair_result_path", "pair_result_file_sha256", "selection_path", "selection_file_sha256"
        }:
            raise WatcherError("manifest pair fields changed")
        for field in ("pair_result", "selection"):
            item = existing_file(str(row[f"{field}_path"]), field)
            if root.resolve() not in item.parents or file_sha256(item) != row[f"{field}_file_sha256"]:
                raise WatcherError(f"{field} escaped collection or SHA changed")
        result = load_json(Path(str(row["pair_result_path"])), "pair result")
        verify_signed(result, "pair_result_sha256", "pair result")
        if result.get("format") != PAIR_RESULT_FORMAT or result.get("pair_id") != expected_id:
            raise WatcherError("pair result identity changed")
        selection = load_json(Path(str(row["selection_path"])), "pre-outcome selection")
        verify_signed(selection, "selection_record_sha256", "pre-outcome selection")
        if (
            selection.get("pair_id") != expected_id
            or selection.get("protocol_sha256") != protocol["protocol_sha256"]
            or selection.get("environment_steps_before_selection") != 0
            or selection.get("candidate_outcomes_visible_to_selector") is not False
            or selection.get("success_reward_event_or_trajectory_visible_to_selector") is not False
        ):
            raise WatcherError("selection was not frozen before outcomes")
        results.append(result)
    return manifest, results


def preregister(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = new_root(args.output_root)
    protocol_audit = revalidate_protocol(Path(args.protocol), args.protocol_file_sha256)
    executor = existing_file(args.executor, "executor")
    python = existing_file(args.python, "Python executable")
    if not os.access(python, os.X_OK):
        raise WatcherError("Python executable is not executable")
    if file_sha256(executor) != args.executor_sha256:
        raise WatcherError("executor SHA mismatch")
    root.mkdir(mode=0o755)
    collection = root / "collection"
    collection.mkdir(mode=0o755)
    (root / "stage").mkdir(mode=0o755)
    command = [
        str(python), str(executor), "--mode", "execute-development",
        "--protocol", protocol_audit["path"], "--protocol-file-sha256", protocol_audit["file_sha256"],
        "--output-root", str(collection), "--gpu-index", str(args.gpu_index),
    ]
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "status": "preregistered_before_detach_and_before_any_development_outcome",
        "output_root": str(root),
        "collection_root": str(collection),
        "protocol": {key: protocol_audit[key] for key in ("path", "file_sha256", "protocol_sha256")},
        "executor": {"path": str(executor), "sha256": args.executor_sha256},
        "watcher": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__).resolve())},
        "python": {"path": str(python), "sha256": file_sha256(python)},
        "command": command,
        "gpu_index": args.gpu_index,
        "required_gpu_name_fragment": EXPECTED_GPU_FRAGMENT,
        "lock_path": str(safe_path(args.lock_path, "GPU lock")),
        "development_only": True,
        "sealed_evaluation_reserve_execution_authorized": False,
        "existing_sensitive_artifacts_read": False,
        "watcher_hdf5_opened": 0,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    immutable_json(root / "static_plan.json", plan)
    atomic_json(root / "state.json", {"format": STATE_FORMAT, "status": "preregistered", "plan_sha256": plan["plan_sha256"]})
    return root, plan


def serve(root: Path, *, idle_interval: float, idle_timeout: float, poll_interval: float) -> dict[str, Any]:
    plan = load_json(root / "static_plan.json", "static plan")
    verify_signed(plan, "plan_sha256", "static plan")
    protocol_audit = revalidate_protocol(Path(plan["protocol"]["path"]), plan["protocol"]["file_sha256"])
    if protocol_audit["protocol_sha256"] != plan["protocol"]["protocol_sha256"]:
        raise WatcherError("protocol logical SHA changed")
    for role in ("watcher", "executor", "python"):
        implementation = existing_file(plan[role]["path"], role)
        if file_sha256(implementation) != plan[role]["sha256"]:
            raise WatcherError(f"{role} implementation SHA changed")
    lock_path = safe_path(plan["lock_path"], "GPU lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="ascii") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WatcherError("GPU lock is already held") from error
        idle = wait_two_idle(plan["gpu_index"], interval=idle_interval, timeout=idle_timeout)
        atomic_json(root / "state.json", {"format": STATE_FORMAT, "status": "running_executor", "plan_sha256": plan["plan_sha256"], "gpu_idle_before": idle})
        log_path = root / "stage" / "run.log"
        process: subprocess.Popen[bytes] | None = None
        foreign: list[int] = []
        returncode = 1
        execution_error: str | None = None
        idle_after: list[dict[str, Any]] = []
        with log_path.open("xb") as log:
            try:
                process = subprocess.Popen(plan["command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"})
                while process.poll() is None:
                    audit = gpu_audit(plan["gpu_index"])
                    foreign = [pid for pid in audit["compute_pids"] if not _is_descendant(pid, process.pid)]
                    if foreign:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=30)
                        break
                    time.sleep(poll_interval)
                returncode = process.wait()
                if returncode == 0 and not foreign:
                    idle_after = wait_two_idle(plan["gpu_index"], interval=idle_interval, timeout=idle_timeout)
            except Exception as error:
                execution_error = f"{type(error).__name__}: {error}"
                log.write(("\nWATCHER_EXECUTION_ERROR=" + execution_error + "\n").encode())
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=30)
                if process is not None and process.returncode not in (None, 0):
                    returncode = int(process.returncode)
            finally:
                log.flush()
                os.fsync(log.fileno())
        stage_exit = returncode if returncode != 0 else (1 if execution_error or foreign else 0)
        immutable_text(root / "stage" / "run.exit", f"{stage_exit}\n")
        stage: dict[str, Any] = {
            "status": "executor_complete" if stage_exit == 0 else "executor_failed_closed",
            "returncode": stage_exit,
            "foreign_gpu_compute_pids": foreign,
            "execution_error": execution_error,
            "run_log_sha256": file_sha256(log_path),
            "run_exit_sha256": file_sha256(root / "stage" / "run.exit"),
            "gpu_idle_after": idle_after,
        }
        stage["stage_receipt_sha256"] = canonical_sha256(stage)
        immutable_json(root / "stage" / "stage_receipt.json", stage)
        if stage_exit != 0:
            raise WatcherError("executor failed or foreign GPU compute process appeared")
    # The watcher reads only JSON products, never simulator HDF5/test labels.
    revalidate_protocol(Path(plan["protocol"]["path"]), plan["protocol"]["file_sha256"])
    manifest, results = validate_collection_manifest(root / "collection", protocol_audit["protocol"])
    evaluation = evaluate_pair_results(protocol_audit["protocol"], results)
    immutable_json(root / "paired_success_evaluation.json", evaluation)
    final: dict[str, Any] = {
        "format": FORMAT,
        "status": TERMINAL_PASSED if evaluation["gate_passed"] else TERMINAL_NULL,
        "plan_sha256": plan["plan_sha256"],
        "protocol_sha256": protocol_audit["protocol_sha256"],
        "stage_returncode": 0,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "scientific_task_success_gate_passed": evaluation["gate_passed"],
        "execution_success_is_not_prediction_success": True,
        "predicted_success_used_as_outcome": False,
        "reserve_identities_read": False,
        "reserve_outcomes_read": False,
        "existing_sensitive_artifacts_read": False,
        "watcher_hdf5_opened": 0,
        "artifacts_frozen_read_only": True,
    }
    final["receipt_sha256"] = canonical_sha256(final)
    immutable_json(root / "final_receipt.json", final)
    immutable_text(root / "run.exit", "0\n")
    atomic_json(root / "state.json", {"format": STATE_FORMAT, "status": final["status"], "receipt_sha256": final["receipt_sha256"]})
    freeze_tree_read_only(root)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("detach", "serve-existing"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--protocol-file-sha256")
    parser.add_argument("--executor")
    parser.add_argument("--executor-sha256")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--lock-path", default="/tmp/etsf_rtx4090_exclusive.lock")
    parser.add_argument("--idle-interval", type=float, default=30.0)
    parser.add_argument("--idle-timeout", type=float, default=604800.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "detach":
        if not all((args.protocol, args.protocol_file_sha256, args.executor, args.executor_sha256)):
            raise SystemExit("detach requires protocol and executor paths plus SHA-256")
        root, plan = preregister(args)
        watcher_log = (root / "watcher.log").open("xb")
        command = [plan["python"]["path"], plan["watcher"]["path"], "--mode", "serve-existing", "--output-root", str(root), "--idle-interval", str(args.idle_interval), "--idle-timeout", str(args.idle_timeout), "--poll-interval", str(args.poll_interval)]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=watcher_log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        receipt = {"format": DETACH_FORMAT, "status": "detached_after_create_once_preregistration", "pid": process.pid, "plan_sha256": plan["plan_sha256"], "command": command}
        receipt["detach_receipt_sha256"] = canonical_sha256(receipt)
        immutable_json(root / "detach_receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
        return
    root = safe_path(args.output_root, "output root").resolve(strict=True)
    try:
        serve(root, idle_interval=args.idle_interval, idle_timeout=args.idle_timeout, poll_interval=args.poll_interval)
    except Exception as error:
        failure = {"format": FORMAT, "status": FAILURE_STATUS, "error_type": type(error).__name__, "error": str(error), "watcher_hdf5_opened": 0, "existing_sensitive_artifacts_read": False}
        failure["receipt_sha256"] = canonical_sha256(failure)
        if not (root / "final_receipt.json").exists():
            immutable_json(root / "final_receipt.json", failure)
        if not (root / "run.exit").exists():
            immutable_text(root / "run.exit", "1\n")
        atomic_json(root / "state.json", {"format": STATE_FORMAT, "status": FAILURE_STATUS, "receipt_sha256": failure["receipt_sha256"]})
        raise


if __name__ == "__main__":
    main()
