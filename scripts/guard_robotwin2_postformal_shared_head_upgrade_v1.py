#!/usr/bin/env python3
"""Supervise one explicitly specified postformal watcher process.

The guarded watcher's ``run.exit`` file is authoritative: ``0`` is durable
success and ``1`` is durable failure.  A child exit without that file is
treated as an interrupted watcher and may be restarted.  The guardian never
changes watcher arguments, experiment ordering, or GPU allocation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_robotwin2_postformal_shared_head_upgrade_guardian_v1"
SAME_ERROR_LIMIT = 3
MAX_UNEXPECTED_RESTARTS = 3
TERMINAL_CHILD_GRACE_SECONDS = 5.0


class PostformalGuardianError(RuntimeError):
    """The guardian contract or a durable watcher signal is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a guardian state document in its own directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PostformalGuardianError("guardian state may not be a symbolic link")
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".partial-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_run_exit(path: Path) -> int | None:
    """Return the exact durable exit value, or ``None`` when it is absent."""

    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise PostformalGuardianError("run.exit must be a real regular file")
    raw = path.read_text(encoding="utf-8").strip()
    if raw not in {"0", "1"}:
        raise PostformalGuardianError("run.exit must contain exactly 0 or 1")
    return int(raw)


def file_fingerprint(path: Path) -> str | None:
    """Read a small state-file fingerprint without interpreting its payload."""

    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise PostformalGuardianError("watcher state must be a real regular file")
    payload = path.read_bytes()
    # Include mtime so rewriting an identical deterministic failure document
    # counts as a new observation, while a stale file left by an older attempt
    # does not.
    return f"{path.stat().st_mtime_ns}:{sha256_bytes(payload)}"


def changed_failed_error(
    path: Path, before_fingerprint: str | None
) -> dict[str, str] | None:
    """Return a newly written watcher failure, never a stale prior failure."""

    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise PostformalGuardianError("watcher state must be a real regular file")
    payload = path.read_bytes()
    payload_sha256 = sha256_bytes(payload)
    fingerprint = f"{path.stat().st_mtime_ns}:{payload_sha256}"
    if fingerprint == before_fingerprint:
        return None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PostformalGuardianError("watcher state is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise PostformalGuardianError("watcher state must be a JSON object")
    error_type = value.get("error_type")
    error_message = value.get("error_message")
    if (
        value.get("status") != "failed"
        or not isinstance(error_type, str)
        or not error_type
        or not isinstance(error_message, str)
    ):
        return None
    signature_payload = json.dumps(
        [error_type, error_message],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "error_type": error_type,
        "error_message": error_message,
        "error_signature_sha256": sha256_bytes(signature_payload),
        "watcher_state_file_sha256": payload_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-exit", type=Path, required=True)
    parser.add_argument("--watcher-state", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--watcher-argv",
        nargs=argparse.REMAINDER,
        required=True,
        help="Exact child argv; this must be the final guardian option.",
    )
    arguments = parser.parse_args(argv)
    for name in ("run_exit", "watcher_state", "state", "lock"):
        setattr(arguments, name, getattr(arguments, name).expanduser().resolve())
    if arguments.poll_seconds <= 0 or arguments.restart_delay_seconds < 0:
        raise PostformalGuardianError("guardian timing values are invalid")
    if not arguments.watcher_argv:
        raise PostformalGuardianError("watcher argv may not be empty")
    executable = Path(arguments.watcher_argv[0]).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise PostformalGuardianError(
            "watcher argv[0] must be an absolute executable file"
        )
    arguments.watcher_argv[0] = str(executable.resolve())
    return arguments


def supervise(arguments: argparse.Namespace) -> int:
    """Run the child until a durable terminal signal or bounded failure."""

    base_state = {
        "format": FORMAT,
        "guardian_pid": os.getpid(),
        "run_exit": str(arguments.run_exit),
        "watcher_state": str(arguments.watcher_state),
        "watcher_argv": list(arguments.watcher_argv),
        "same_error_limit": SAME_ERROR_LIMIT,
        "max_unexpected_restarts": MAX_UNEXPECTED_RESTARTS,
    }
    attempts: list[dict[str, Any]] = []
    unexpected_restarts = 0
    prior_error_signature_sha256: str | None = None
    same_error_count = 0

    def write_state(status: str, **extra: Any) -> None:
        atomic_json(
            arguments.state,
            {
                **base_state,
                "status": status,
                "updated_at_utc": utc_now(),
                "attempts_started": len(attempts),
                "unexpected_restart_count": unexpected_restarts,
                "same_error_consecutive_count": same_error_count,
                "attempt_history": list(attempts),
                **extra,
            },
        )

    existing_terminal = read_run_exit(arguments.run_exit)
    if existing_terminal == 0:
        write_state(
            "complete",
            terminal_reason="preexisting_run_exit_0",
            child_process_started=False,
        )
        return 0
    if existing_terminal == 1:
        write_state(
            "failed",
            terminal_reason="preexisting_run_exit_1",
            child_process_started=False,
        )
        return 1

    while True:
        before_fingerprint = file_fingerprint(arguments.watcher_state)
        attempt_number = len(attempts) + 1
        process = subprocess.Popen(list(arguments.watcher_argv), shell=False)
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "child_pid": process.pid,
            "started_at_utc": utc_now(),
        }
        attempts.append(attempt)
        write_state(
            "running",
            active_child_pid=process.pid,
            active_attempt_number=attempt_number,
        )

        terminal_seen_at: float | None = None
        child_termination_sent_at: float | None = None
        child_terminated_after_terminal_signal = False
        child_killed_after_terminal_signal = False
        try:
            while process.poll() is None:
                terminal_while_running = read_run_exit(arguments.run_exit)
                if terminal_while_running is not None:
                    now = time.monotonic()
                    if terminal_seen_at is None:
                        terminal_seen_at = now
                    elif (
                        child_termination_sent_at is None
                        and now - terminal_seen_at >= TERMINAL_CHILD_GRACE_SECONDS
                    ):
                        # The durable terminal file is authoritative.  Do not
                        # leave a wedged child holding its watcher/GPU lock.
                        process.terminate()
                        child_termination_sent_at = now
                        child_terminated_after_terminal_signal = True
                    elif (
                        child_termination_sent_at is not None
                        and now - child_termination_sent_at
                        >= TERMINAL_CHILD_GRACE_SECONDS
                    ):
                        process.kill()
                        child_killed_after_terminal_signal = True
                time.sleep(arguments.poll_seconds)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

        attempt["finished_at_utc"] = utc_now()
        attempt["child_returncode"] = process.returncode
        attempt["child_terminated_after_terminal_signal"] = (
            child_terminated_after_terminal_signal
        )
        attempt["child_killed_after_terminal_signal"] = (
            child_killed_after_terminal_signal
        )
        terminal = read_run_exit(arguments.run_exit)
        if terminal == 0:
            write_state(
                "complete",
                terminal_reason="run_exit_0",
                terminal_attempt_number=attempt_number,
                child_returncode=process.returncode,
            )
            return 0
        if terminal == 1:
            terminal_error = changed_failed_error(
                arguments.watcher_state, before_fingerprint
            )
            if terminal_error is not None:
                attempt.update(terminal_error)
            write_state(
                "failed",
                terminal_reason="run_exit_1",
                terminal_attempt_number=attempt_number,
                child_returncode=process.returncode,
                watcher_failure_error=(
                    {
                        "error_type": terminal_error["error_type"],
                        "error_message": terminal_error["error_message"],
                    }
                    if terminal_error is not None
                    else None
                ),
            )
            return 1

        observed_error = changed_failed_error(
            arguments.watcher_state, before_fingerprint
        )
        if observed_error is None:
            prior_error_signature_sha256 = None
            same_error_count = 0
        else:
            attempt.update(observed_error)
            if (
                observed_error["error_signature_sha256"]
                == prior_error_signature_sha256
            ):
                same_error_count += 1
            else:
                prior_error_signature_sha256 = observed_error[
                    "error_signature_sha256"
                ]
                same_error_count = 1
            if same_error_count >= SAME_ERROR_LIMIT:
                write_state(
                    "failed",
                    terminal_reason="same_unrecoverable_error_repeated",
                    terminal_attempt_number=attempt_number,
                    repeated_error={
                        "error_type": observed_error["error_type"],
                        "error_message": observed_error["error_message"],
                    },
                    repeated_error_signature_sha256=observed_error[
                        "error_signature_sha256"
                    ],
                    child_returncode=process.returncode,
                )
                return 1

        if unexpected_restarts >= MAX_UNEXPECTED_RESTARTS:
            write_state(
                "failed",
                terminal_reason="unexpected_restart_limit_exhausted",
                terminal_attempt_number=attempt_number,
                child_returncode=process.returncode,
            )
            return 1

        unexpected_restarts += 1
        write_state(
            "restarting",
            last_child_pid=process.pid,
            last_child_returncode=process.returncode,
            next_attempt_number=attempt_number + 1,
        )
        if arguments.restart_delay_seconds:
            time.sleep(arguments.restart_delay_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    arguments.lock.parent.mkdir(parents=True, exist_ok=True)
    if arguments.lock.is_symlink():
        raise PostformalGuardianError("guardian lock may not be a symbolic link")
    lock_stream = arguments.lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise PostformalGuardianError("another guardian is active") from error
    try:
        return supervise(arguments)
    except Exception as error:
        atomic_json(
            arguments.state,
            {
                "format": FORMAT,
                "status": "failed",
                "updated_at_utc": utc_now(),
                "guardian_pid": os.getpid(),
                "run_exit": str(arguments.run_exit),
                "watcher_state": str(arguments.watcher_state),
                "watcher_argv": list(arguments.watcher_argv),
                "terminal_reason": "guardian_error",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        lock_stream.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)


__all__ = [
    "FORMAT",
    "MAX_UNEXPECTED_RESTARTS",
    "PostformalGuardianError",
    "SAME_ERROR_LIMIT",
    "TERMINAL_CHILD_GRACE_SECONDS",
    "atomic_json",
    "changed_failed_error",
    "file_fingerprint",
    "main",
    "parse_args",
    "read_run_exit",
    "supervise",
]
