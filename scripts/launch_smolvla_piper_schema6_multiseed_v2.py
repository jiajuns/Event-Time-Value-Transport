#!/usr/bin/env python3
"""Detached single-RTX4090 watcher for Schema6 multi-seed Phase-2.

Static preregistration and a separate production execution authority are both
required before the output root is claimed.  The current CPU preregistration is
not itself an execution authorization; absent authority/target/adapter bytes
therefore fail before detach or environment construction.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from preregister_smolvla_piper_schema6_multiseed_collection_v2 import (
    GROUP_RECEIPT_FORMAT,
    GROUP_RECEIPT_STATUS,
    canonical_sha256,
    file_sha256,
    validate_completed_prefix,
    validate_preregistration,
)
from run_smolvla_piper_schema6_multiseed_v2 import (
    Phase2RunnerError,
    _validate_command,
    existing_file,
    load_json,
    production_preflight,
    safe_path,
    verify_signed,
)


FORMAT = "etsf_smolvla_piper_schema6_multiseed_watcher_v2"
PLAN_FORMAT = "etsf_smolvla_piper_schema6_multiseed_watcher_plan_v2"
STATE_FORMAT = "etsf_smolvla_piper_schema6_multiseed_watcher_state_v2"
DETACH_FORMAT = "etsf_smolvla_piper_schema6_multiseed_detach_receipt_v2"
TERMINAL_STATUS = "complete_adaptation80_validation50_schema6_collection"
FAILURE_STATUS = "failed_closed_schema6_multiseed_collection"
EXPECTED_GPU_FRAGMENT = "RTX 4090"


class Phase2WatcherError(RuntimeError):
    """A watcher, prefix, GPU, or detached-launch invariant failed."""


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def immutable_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def freeze_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise Phase2WatcherError("output contains a symlink")
            item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise Phase2WatcherError("output contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def _run_text(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def gpu_audit(gpu_index: int, run_text: Callable[[Sequence[str]], str] = _run_text) -> dict[str, Any]:
    identity = run_text(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name,uuid", "--format=csv,noheader"]
    ).strip().split(",", 1)
    if len(identity) != 2 or EXPECTED_GPU_FRAGMENT not in identity[0].strip():
        raise Phase2WatcherError("designated GPU is not an RTX 4090")
    raw = run_text(
        ["nvidia-smi", f"--id={gpu_index}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    )
    pids = sorted({int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()})
    return {
        "gpu_index": gpu_index,
        "name": identity[0].strip(),
        "uuid": identity[1].strip(),
        "compute_pids": pids,
    }


def wait_two_idle_forever(
    gpu_index: int, *, interval: float,
    run_text: Callable[[Sequence[str]], str] = _run_text,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    consecutive: list[dict[str, Any]] = []
    while True:
        audit = gpu_audit(gpu_index, run_text)
        if audit["compute_pids"]:
            consecutive.clear()
        else:
            if consecutive and consecutive[0]["uuid"] != audit["uuid"]:
                raise Phase2WatcherError("GPU identity changed")
            consecutive.append(audit)
            if len(consecutive) == 2:
                return consecutive
        sleep(interval)


def wait_for_ppid1(
    *, getppid: Callable[[], int] = os.getppid,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    while getppid() != 1:
        sleep(0.1)


def _is_descendant(pid: int, ancestor: int) -> bool:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().split()[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def validate_completed_prefix_files(
    preregistration: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decoded = validate_preregistration(preregistration)
    receipts: list[dict[str, Any]] = []
    first_missing = False
    for index, command in enumerate(decoded["commands"]):
        outputs = command["outputs"]
        seed_root = safe_path(outputs["seed_root"], f"seed root {index}")
        receipt_path = safe_path(outputs["completed_group_receipt"], f"group receipt {index}")
        exists = receipt_path.is_file() and not receipt_path.is_symlink()
        if first_missing:
            if seed_root.exists() or receipt_path.exists():
                raise Phase2WatcherError("completed prefix has a gap or later partial output")
            continue
        if not exists:
            first_missing = True
            if seed_root.exists() or seed_root.is_symlink():
                raise Phase2WatcherError("first unfinished command has a partial seed root")
            continue
        receipt = load_json(receipt_path, f"group receipt {index}")
        if (
            receipt.get("format") != GROUP_RECEIPT_FORMAT
            or receipt.get("status") != GROUP_RECEIPT_STATUS
        ):
            raise Phase2WatcherError("completed receipt status changed")
        group_path = existing_file(outputs["group_hdf5"], f"group HDF5 {index}")
        reset_path = existing_file(outputs["per_seed_reset_receipt"], f"reset receipt {index}")
        reset = load_json(reset_path, f"reset receipt {index}")
        reset_sha = verify_signed(reset, "reset_receipt_sha256", f"reset receipt {index}")
        if (
            file_sha256(group_path) != receipt.get("group_file_sha256")
            or reset_sha != receipt.get("per_seed_reset_receipt_sha256")
            or receipt_path.parent != seed_root
            or group_path.parent != seed_root
            or reset_path.parent != seed_root
        ):
            raise Phase2WatcherError("completed group files or SHA changed")
        receipts.append(receipt)
    pending = validate_completed_prefix(preregistration, receipts)
    return receipts, pending


def static_preflight(
    *, preregistration_path: Path, expected_preregistration_file_sha256: str,
    execution_authority_path: Path | None, output_root: Path,
    gpu_index: int, gpu_lock_path: Path,
) -> dict[str, Any]:
    prereg_path = existing_file(preregistration_path, "preregistration")
    if file_sha256(prereg_path) != expected_preregistration_file_sha256:
        raise Phase2WatcherError("preregistration file SHA changed")
    preflight = production_preflight(prereg_path, execution_authority_path)
    prereg = preflight["preregistration"]
    output = safe_path(output_root, "collection output root")
    if safe_path(prereg["outputs"]["future_collection_root"], "planned collection root") != output:
        raise Phase2WatcherError("output root differs from preregistration")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if gpu_index < 0:
        raise Phase2WatcherError("GPU index is invalid")
    lock = safe_path(gpu_lock_path, "GPU lock")
    if safe_path(prereg["execution_contract"]["gpu_lock_path"], "planned GPU lock") != lock:
        raise Phase2WatcherError("GPU lock differs from preregistration")
    commands = validate_preregistration(prereg)["commands"]
    for index, command in enumerate(commands):
        _validate_command(command, prereg)
        if command["argv"][0] != prereg["input_bindings"]["runtime_python"]["path"]:
            raise Phase2WatcherError(f"command {index} Python binding changed")
        if command["argv"][1] != prereg["input_bindings"]["v2_runner"]["path"]:
            raise Phase2WatcherError(f"command {index} runner binding changed")
    return {
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": expected_preregistration_file_sha256,
        "preregistration_sha256": preflight["preregistration_sha256"],
        "execution_authority_path": str(existing_file(execution_authority_path, "execution authority")),
        "execution_authority_sha256": preflight["authority"]["authority_sha256"],
        "output_root": str(output),
        "gpu_index": gpu_index,
        "gpu_lock_path": str(lock),
        "commands": commands,
        "preregistration": prereg,
    }


def preregister_watcher(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    audit = static_preflight(
        preregistration_path=args.preregistration,
        expected_preregistration_file_sha256=args.preregistration_file_sha256,
        execution_authority_path=args.execution_authority,
        output_root=args.output_root,
        gpu_index=args.gpu_index,
        gpu_lock_path=args.gpu_lock,
    )
    root = Path(audit["output_root"])
    root.mkdir(mode=0o755)
    (root / "_watcher").mkdir(mode=0o755)
    (root / "_watcher" / "stages").mkdir(mode=0o755)
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "status": "static_preflight_complete_production_dependencies_authorized",
        "preregistration_path": audit["preregistration_path"],
        "preregistration_file_sha256": audit["preregistration_file_sha256"],
        "preregistration_sha256": audit["preregistration_sha256"],
        "execution_authority_path": audit["execution_authority_path"],
        "execution_authority_sha256": audit["execution_authority_sha256"],
        "output_root": audit["output_root"],
        "gpu_index": audit["gpu_index"],
        "gpu_lock_path": audit["gpu_lock_path"],
        "watcher_path": str(Path(__file__).resolve()),
        "watcher_file_sha256": file_sha256(Path(__file__).resolve()),
        "command_count": 130,
        "ordered_splits": ["adaptation", "validation"],
        "production_timeout_seconds": None,
        "evaluation_commands_authorized": 0,
        "test_inputs_read": False,
        "fresh_inputs_accepted": False,
        "confirmation_inputs_accepted": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    immutable_json(root / "_watcher" / "static_plan.json", plan)
    atomic_json(root / "_watcher" / "state.json", {"format": STATE_FORMAT, "status": "preregistered", "plan_sha256": plan["plan_sha256"]})
    return root, plan


def _load_bound_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json(root / "_watcher" / "static_plan.json", "watcher static plan")
    verify_signed(plan, "plan_sha256", "watcher static plan")
    if plan.get("format") != PLAN_FORMAT or plan.get("command_count") != 130:
        raise Phase2WatcherError("watcher plan scope changed")
    if file_sha256(existing_file(plan["watcher_path"], "watcher")) != plan["watcher_file_sha256"]:
        raise Phase2WatcherError("watcher implementation changed")
    preflight = production_preflight(
        Path(plan["preregistration_path"]), Path(plan["execution_authority_path"])
    )
    if (
        preflight["preregistration_file_sha256"] != plan["preregistration_file_sha256"]
        or preflight["preregistration_sha256"] != plan["preregistration_sha256"]
        or preflight["authority"]["authority_sha256"] != plan["execution_authority_sha256"]
    ):
        raise Phase2WatcherError("production dependencies changed after preregistration")
    return plan, preflight["preregistration"]


def _run_one_command(
    *, root: Path, command: Mapping[str, Any], global_index: int,
    gpu_index: int, authority_path: str, poll_interval: float,
) -> dict[str, Any]:
    stage = root / "_watcher" / "stages" / f"stage_{global_index:03d}"
    stage.mkdir(mode=0o755)
    launch: dict[str, Any] = {
        "status": "launching_exact_preregistered_command",
        "global_index": global_index,
        "command_sha256": command["command_sha256"],
        "argv": command["argv"],
        "environment_authority_path": authority_path,
    }
    launch["launch_receipt_sha256"] = canonical_sha256(launch)
    immutable_json(stage / "launch_receipt.json", launch)
    log_path = stage / "run.log"
    process: subprocess.Popen[bytes] | None = None
    foreign: list[int] = []
    execution_error: str | None = None
    returncode = 1
    with log_path.open("xb") as log:
        try:
            environment = dict(os.environ)
            environment.update(
                {
                    "ETSF_SCHEMA6_V2_EXECUTION_AUTHORITY": authority_path,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            process = subprocess.Popen(
                command["argv"], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=environment,
            )
            while process.poll() is None:
                audit = gpu_audit(gpu_index)
                foreign = [pid for pid in audit["compute_pids"] if not _is_descendant(pid, process.pid)]
                if foreign:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                    break
                time.sleep(poll_interval)
            returncode = process.wait()
        except Exception as error:
            execution_error = f"{type(error).__name__}: {error}"
            log.write(("\nWATCHER_ERROR=" + execution_error + "\n").encode())
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
    stage_exit = returncode if returncode != 0 else (1 if foreign or execution_error else 0)
    immutable_text(stage / "run.exit", f"{stage_exit}\n")
    receipt: dict[str, Any] = {
        "status": "complete" if stage_exit == 0 else "failed_closed",
        "global_index": global_index,
        "command_sha256": command["command_sha256"],
        "returncode": stage_exit,
        "foreign_gpu_compute_pids": foreign,
        "execution_error": execution_error,
        "run_log_sha256": file_sha256(log_path),
        "run_exit_sha256": file_sha256(stage / "run.exit"),
    }
    receipt["stage_receipt_sha256"] = canonical_sha256(receipt)
    immutable_json(stage / "stage_receipt.json", receipt)
    if stage_exit != 0:
        raise Phase2WatcherError("per-seed runner failed or foreign GPU PID appeared")
    return receipt


def serve(root: Path, *, idle_interval: float, poll_interval: float) -> dict[str, Any]:
    wait_for_ppid1()
    plan, prereg = _load_bound_plan(root)
    receipts, pending = validate_completed_prefix_files(prereg)
    lock_path = Path(plan["gpu_lock_path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="ascii") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Phase2WatcherError("RTX4090 lock is already held") from error
        idle_before = wait_two_idle_forever(plan["gpu_index"], interval=idle_interval)
        commands = validate_preregistration(prereg)["commands"]
        for command in pending:
            global_index = commands.index(command)
            atomic_json(root / "_watcher" / "state.json", {"format": STATE_FORMAT, "status": "running", "completed_prefix": global_index, "command_sha256": command["command_sha256"]})
            _run_one_command(
                root=root,
                command=command,
                global_index=global_index,
                gpu_index=plan["gpu_index"],
                authority_path=plan["execution_authority_path"],
                poll_interval=poll_interval,
            )
            receipts, now_pending = validate_completed_prefix_files(prereg)
            if len(receipts) != global_index + 1 or len(now_pending) != 130 - len(receipts):
                raise Phase2WatcherError("runner did not advance the signed prefix exactly once")
        idle_after = wait_two_idle_forever(plan["gpu_index"], interval=idle_interval)
    receipts, pending = validate_completed_prefix_files(prereg)
    if len(receipts) != 130 or pending:
        raise Phase2WatcherError("terminal collection is not a complete 130-group prefix")
    final: dict[str, Any] = {
        "format": FORMAT,
        "status": TERMINAL_STATUS,
        "plan_sha256": plan["plan_sha256"],
        "preregistration_sha256": plan["preregistration_sha256"],
        "execution_authority_sha256": plan["execution_authority_sha256"],
        "completed_groups": 130,
        "adaptation_groups": 80,
        "validation_groups": 50,
        "evaluation_groups": 0,
        "signed_gap_free_prefix_complete": True,
        "gpu_idle_before": idle_before,
        "gpu_idle_after": idle_after,
        "foreign_gpu_compute_pids_accepted": False,
        "production_timeout_seconds": None,
        "test_inputs_read": False,
        "fresh_inputs_accepted": False,
        "confirmation_inputs_accepted": False,
        "artifacts_frozen_read_only": True,
    }
    final["receipt_sha256"] = canonical_sha256(final)
    immutable_json(root / "_watcher" / "final_receipt.json", final)
    immutable_text(root / "_watcher" / "run.exit", "0\n")
    atomic_json(root / "_watcher" / "state.json", {"format": STATE_FORMAT, "status": TERMINAL_STATUS, "receipt_sha256": final["receipt_sha256"]})
    freeze_tree(root)
    return final


def _detach(root: Path, plan: Mapping[str, Any], *, idle_interval: float, poll_interval: float, resume_prefix: int | None = None) -> dict[str, Any]:
    command = [
        str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
        "--mode", "serve-existing", "--output-root", str(root),
        "--idle-interval", str(idle_interval), "--poll-interval", str(poll_interval),
    ]
    log_name = "watcher.log" if resume_prefix is None else f"watcher_resume_{resume_prefix:03d}.log"
    with (root / "_watcher" / log_name).open("xb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    receipt: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detached_new_session_ppid1_required_before_gpu_lock",
        "pid": process.pid,
        "plan_sha256": plan["plan_sha256"],
        "resume_completed_prefix": resume_prefix,
        "command": command,
    }
    receipt["detach_receipt_sha256"] = canonical_sha256(receipt)
    if resume_prefix is None:
        path = root / "_watcher" / "detach_receipt.json"
    else:
        directory = root / "_watcher" / "resume_detach_receipts"
        directory.mkdir(exist_ok=True)
        path = directory / f"prefix_{resume_prefix:03d}.json"
    immutable_json(path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "detach", "detach-resume", "serve-existing"), required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--preregistration-file-sha256")
    parser.add_argument("--execution-authority", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--idle-interval", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode in {"preflight", "detach"}:
        if None in (args.preregistration, args.preregistration_file_sha256, args.gpu_lock):
            raise Phase2WatcherError("static preflight arguments are incomplete")
        if args.mode == "preflight":
            audit = static_preflight(
                preregistration_path=args.preregistration,
                expected_preregistration_file_sha256=args.preregistration_file_sha256,
                execution_authority_path=args.execution_authority,
                output_root=args.output_root,
                gpu_index=args.gpu_index,
                gpu_lock_path=args.gpu_lock,
            )
            print(json.dumps({key: audit[key] for key in ("preregistration_sha256", "execution_authority_sha256", "output_root")}, sort_keys=True))
            return 0
        root, plan = preregister_watcher(args)
        print(json.dumps(_detach(root, plan, idle_interval=args.idle_interval, poll_interval=args.poll_interval), sort_keys=True))
        return 0
    root = safe_path(args.output_root, "output root").resolve(strict=True)
    if args.mode == "detach-resume":
        plan, prereg = _load_bound_plan(root)
        receipts, _pending = validate_completed_prefix_files(prereg)
        if (root / "_watcher" / "final_receipt.json").exists():
            raise Phase2WatcherError("terminal watcher cannot be resumed")
        print(json.dumps(_detach(root, plan, idle_interval=args.idle_interval, poll_interval=args.poll_interval, resume_prefix=len(receipts)), sort_keys=True))
        return 0
    try:
        serve(root, idle_interval=args.idle_interval, poll_interval=args.poll_interval)
    except Exception as error:
        failure = {
            "format": FORMAT,
            "status": FAILURE_STATUS,
            "error_type": type(error).__name__,
            "error": str(error),
            "test_inputs_read": False,
            "fresh_inputs_accepted": False,
            "confirmation_inputs_accepted": False,
        }
        failure["receipt_sha256"] = canonical_sha256(failure)
        final = root / "_watcher" / "final_receipt.json"
        if not final.exists():
            immutable_json(final, failure)
        exit_path = root / "_watcher" / "run.exit"
        if not exit_path.exists():
            immutable_text(exit_path, "1\n")
        atomic_json(root / "_watcher" / "state.json", {"format": STATE_FORMAT, "status": FAILURE_STATUS, "receipt_sha256": failure["receipt_sha256"]})
        freeze_tree(root)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Phase2WatcherError", "gpu_audit", "static_preflight",
    "validate_completed_prefix_files", "wait_for_ppid1", "wait_two_idle_forever",
]
