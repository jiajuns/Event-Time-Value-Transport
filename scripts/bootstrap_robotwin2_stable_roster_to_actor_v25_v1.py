#!/usr/bin/env python3
"""Bridge a frozen reset-only roster to one detached actor-v2 run.

This standard-library-only bootstrap never opens actor results, labels,
rollouts, pair artifacts, or method receipts.  It waits for and authenticates
the common-stable reset roster, launches the exact runner in a new session,
waits only for the runner's immutable static deployment binding, launches the
exact guardian in a second new session, writes immutable launch receipts, and
exits.  Existing experiment output is never resumed or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "etsf_robotwin2_stable_roster_actor_v25_bootstrap_v1"
PLAN_FORMAT = "etsf_robotwin2_stable_roster_actor_v25_bootstrap_plan_v1"
RUNNER_LAUNCH_FORMAT = (
    "etsf_robotwin2_stable_roster_actor_v25_runner_launch_v1"
)
GUARDIAN_LAUNCH_FORMAT = (
    "etsf_robotwin2_stable_roster_actor_v25_guardian_launch_v1"
)
GUARDIAN_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_guardian_v1"
GUARDIAN_PLAN_FORMAT = (
    "etsf_robotwin2_actor_execute5_vs_execute50_guardian_plan_v1"
)
GUARDIAN_STATE_FORMAT = (
    "etsf_robotwin2_actor_execute5_vs_execute50_guardian_state_v1"
)
COMPLETE_FORMAT = "etsf_robotwin2_stable_roster_actor_v25_bootstrap_complete_v1"
FAILURE_FORMAT = "etsf_robotwin2_stable_roster_actor_v25_bootstrap_failure_v1"
ROSTER_FORMAT = "etsf_robotwin2_common_stable_seed_roster_v1"
RUNNER_BINDING_FORMAT = (
    "etsf_robotwin2_actor_deployment_protocol_binding_v2_stable_roster"
)
RUNNER_FORMAT = (
    "etsf_robotwin2_five_body_actor_execute5_vs_execute50_v2_stable_roster"
)
EXPECTED_BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
EXPECTED_CONDITIONS = ("clean", "randomized")
EXPECTED_SELECTED_SEEDS = 20
EXPECTED_PAIRS = 200
EXPECTED_ROLLOUTS = 400
SHA256_CHARS = frozenset("0123456789abcdef")


class ActorV25BootstrapError(RuntimeError):
    """A static authority, roster, launch, or handoff changed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_CHARS)
    )


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ActorV25BootstrapError(f"file is missing or symbolic: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ActorV25BootstrapError(f"{label} is missing or symbolic: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ActorV25BootstrapError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ActorV25BootstrapError(f"{label} must be a JSON object")
    return value


def verify_logical_sha(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise ActorV25BootstrapError(f"{label} logical SHA-256 mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_once_bytes(path: Path, payload: bytes, label: str) -> None:
    if path.is_symlink():
        raise ActorV25BootstrapError(f"{label} may not be symbolic")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ActorV25BootstrapError(f"existing {label} differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".create", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ActorV25BootstrapError(f"racing {label} differs")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def create_once_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    payload = (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    create_once_bytes(path, payload, label)


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("logical_sha256", None)
    return {**unsigned, "logical_sha256": canonical_sha256(unsigned)}


def real_path(
    path: Path,
    *,
    label: str,
    directory: bool = False,
    executable: bool = False,
    allow_symlink: bool = False,
) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() and not allow_symlink:
        raise ActorV25BootstrapError(f"{label} may not be symbolic")
    try:
        resolved = expanded.resolve(strict=True)
        mode = os.stat(
            expanded, follow_symlinks=allow_symlink
        ).st_mode
    except (FileNotFoundError, OSError) as error:
        raise ActorV25BootstrapError(f"{label} is missing: {expanded}") from error
    valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not valid or resolved.is_symlink():
        raise ActorV25BootstrapError(f"{label} has the wrong file type")
    if executable and not os.access(resolved, os.X_OK):
        raise ActorV25BootstrapError(f"{label} is not executable")
    return resolved


def validate_roster(
    value: Mapping[str, Any], *, expected_preregistration_file_sha256: str
) -> dict[str, Any]:
    """Validate only reset-probe accounting; never inspect actor data."""

    verify_logical_sha(value, "stable reset roster")
    attempts = value.get("attempts")
    selected = value.get("selected_seeds")
    if (
        value.get("format") != ROSTER_FORMAT
        or value.get("status") != "complete_first_twenty_common_stable_seeds"
        or value.get("body_order") != list(EXPECTED_BODIES)
        or value.get("condition_order") != list(EXPECTED_CONDITIONS)
        or value.get("preregistration_file_sha256")
        != expected_preregistration_file_sha256
        or not is_sha256(value.get("preregistration_logical_sha256"))
        or not isinstance(attempts, list)
        or not attempts
        or not isinstance(selected, list)
        or len(selected) != EXPECTED_SELECTED_SEEDS
        or len(set(selected)) != EXPECTED_SELECTED_SEEDS
        or value.get("pair_count") != EXPECTED_PAIRS
        or value.get("rollout_count_for_two_methods") != EXPECTED_ROLLOUTS
        or value.get("actor_inference_calls") != 0
        or value.get("task_action_calls") != 0
        or value.get("label_or_outcome_reads") != 0
    ):
        raise ActorV25BootstrapError("stable reset roster header changed")

    stable_in_order: list[int] = []
    last_seed: int | None = None
    cell_order = [
        (body, condition)
        for body in EXPECTED_BODIES
        for condition in EXPECTED_CONDITIONS
    ]
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ActorV25BootstrapError("stable reset attempt is invalid")
        seed = attempt.get("candidate_seed")
        cells = attempt.get("cells")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or (last_seed is not None and seed != last_seed + 1)
            or not isinstance(cells, list)
            or len(cells) != len(cell_order)
            or attempt.get("actor_inference_calls") != 0
            or attempt.get("task_action_calls") != 0
            or attempt.get("label_or_outcome_reads") != 0
        ):
            raise ActorV25BootstrapError("stable reset attempt accounting changed")
        last_seed = seed
        failure_seen = False
        for index, ((body, condition), cell) in enumerate(
            zip(cell_order, cells, strict=True)
        ):
            if (
                not isinstance(cell, Mapping)
                or cell.get("cell_index") != index
                or cell.get("body") != body
                or cell.get("condition") != condition
            ):
                raise ActorV25BootstrapError("stable reset cell order changed")
            status = cell.get("status")
            if failure_seen:
                if status != "not_attempted_after_first_setup_failure":
                    raise ActorV25BootstrapError("reset probe did not short-circuit")
            elif status == "setup_failed":
                failure_seen = True
            elif status != "setup_succeeded_and_closed":
                raise ActorV25BootstrapError("reset probe cell status changed")
        stable = not failure_seen
        if attempt.get("all_ten_setup_cells_stable") is not stable:
            raise ActorV25BootstrapError("reset probe stable flag changed")
        if stable:
            stable_in_order.append(seed)
    if (
        selected != stable_in_order[:EXPECTED_SELECTED_SEEDS]
        or attempts[-1].get("candidate_seed") != selected[-1]
        or value.get("candidate_attempt_count") != len(attempts)
        or value.get("stable_candidate_count_observed") != len(stable_in_order)
    ):
        raise ActorV25BootstrapError("stable reset roster selection changed")
    return dict(value)


def runner_command(args: argparse.Namespace, roster_sha256: str) -> list[str]:
    return [
        str(args.runner_python),
        str(args.runner),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--vlm-metadata-path",
        str(args.vlm_metadata_path),
        "--robotwin-root",
        str(args.robotwin_root),
        "--event-spec",
        str(args.event_spec),
        "--stable-seed-roster",
        str(args.stable_seed_roster),
        "--stable-seed-roster-sha256",
        roster_sha256,
        "--output",
        str(args.output),
    ]


def guardian_command(
    args: argparse.Namespace, *, runner_pid: int, roster_sha256: str
) -> list[str]:
    return [
        str(args.guardian_python),
        str(args.guardian),
        "--runner-pid",
        str(runner_pid),
        "--python-bin",
        str(args.runner_python),
        "--code-root",
        str(args.code_root),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--vlm-metadata-path",
        str(args.vlm_metadata_path),
        "--robotwin-root",
        str(args.robotwin_root),
        "--event-spec",
        str(args.event_spec),
        "--stable-seed-roster",
        str(args.stable_seed_roster),
        "--stable-seed-roster-sha256",
        roster_sha256,
        "--output",
        str(args.output),
        "--state-root",
        str(args.guardian_state_root),
        "--gpu-uuid",
        args.gpu_uuid,
        "--nvidia-smi",
        str(args.nvidia_smi),
        "--poll-seconds",
        str(args.guardian_poll_seconds),
    ]


def validate_runner_binding(
    path: Path,
    *,
    runner: Path,
    runner_sha256: str,
    roster: Path,
    roster_sha256: str,
) -> tuple[dict[str, Any], str]:
    value = read_json(path, "immutable deployment binding")
    verify_logical_sha(value, "immutable deployment binding")
    roster_binding = value.get("stable_seed_roster_binding")
    if (
        value.get("format") != RUNNER_BINDING_FORMAT
        or value.get("runner_format") != RUNNER_FORMAT
        or value.get("runner_path") != str(runner)
        or value.get("runner_sha256") != runner_sha256
        or not isinstance(roster_binding, Mapping)
        or roster_binding.get("path") != str(roster)
        or roster_binding.get("file_sha256") != roster_sha256
        or roster_binding.get("selection_uses_labels_or_outcomes") is not False
        or roster_binding.get("actor_inference_calls_during_selection") != 0
    ):
        raise ActorV25BootstrapError("immutable deployment binding changed")
    return dict(value), sha256_file(path)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_owned_session(
    process: subprocess.Popen[bytes] | None,
    *,
    label: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Terminate only a child still authenticated as its own session group."""

    if process is None:
        return {"label": label, "owned_child_started": False, "action": "none"}
    pid = process.pid
    returncode = process.poll()
    if returncode is not None:
        # poll() also reaps this exact Popen child.
        return {
            "label": label,
            "owned_child_started": True,
            "pid": pid,
            "action": "already_exited_and_reaped",
            "returncode": returncode,
        }
    try:
        process_group = os.getpgid(pid)
        session_id = os.getsid(pid)
    except ProcessLookupError:
        returncode = process.wait(timeout=timeout_seconds)
        return {
            "label": label,
            "owned_child_started": True,
            "pid": pid,
            "action": "disappeared_and_reaped",
            "returncode": returncode,
        }
    if process_group != pid or session_id != pid:
        raise ActorV25BootstrapError(
            f"refusing cleanup of unauthenticated {label} process group: "
            f"pid={pid}, pgid={process_group}, sid={session_id}"
        )
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        returncode = process.wait(timeout=timeout_seconds)
        return {
            "label": label,
            "owned_child_started": True,
            "pid": pid,
            "process_group": process_group,
            "session_id": session_id,
            "action": "exited_before_sigterm_and_reaped",
            "returncode": returncode,
        }
    try:
        returncode = process.wait(timeout=timeout_seconds)
        return {
            "label": label,
            "owned_child_started": True,
            "pid": pid,
            "process_group": process_group,
            "session_id": session_id,
            "action": "sigterm_and_reaped",
            "returncode": returncode,
        }
    except subprocess.TimeoutExpired:
        # Re-authenticate immediately before escalation to protect against PID
        # reuse or an unexpected group/session mutation.
        try:
            current_group = os.getpgid(pid)
            current_session = os.getsid(pid)
        except ProcessLookupError:
            returncode = process.wait(timeout=timeout_seconds)
            return {
                "label": label,
                "owned_child_started": True,
                "pid": pid,
                "action": "exited_after_sigterm_and_reaped",
                "returncode": returncode,
            }
        if current_group != pid or current_session != pid:
            raise ActorV25BootstrapError(
                f"refusing SIGKILL of unauthenticated {label} process group"
            )
        os.killpg(pid, signal.SIGKILL)
        returncode = process.wait(timeout=timeout_seconds)
        return {
            "label": label,
            "owned_child_started": True,
            "pid": pid,
            "process_group": current_group,
            "session_id": current_session,
            "action": "sigterm_then_sigkill_and_reaped",
            "returncode": returncode,
        }


def launch_child(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: Mapping[str, str],
) -> subprocess.Popen[bytes]:
    if log_path.exists() or log_path.is_symlink():
        raise ActorV25BootstrapError(f"child log already exists: {log_path}")
    with log_path.open("xb", buffering=0) as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    try:
        process_group = os.getpgid(process.pid)
        session_id = os.getsid(process.pid)
    except ProcessLookupError as error:
        process.wait()
        raise ActorV25BootstrapError("child exited before session audit") from error
    if process_group != process.pid or session_id != process.pid:
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        except BaseException as cleanup_error:
            raise ActorV25BootstrapError(
                "child did not become a new session leader and could not be "
                "safely cleaned up"
            ) from cleanup_error
        raise ActorV25BootstrapError("child did not become a new session leader")
    return process


def wait_for_roster(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    while True:
        path = args.stable_seed_roster
        if path.exists() or path.is_symlink():
            value = read_json(path, "stable reset roster")
            validated = validate_roster(
                value,
                expected_preregistration_file_sha256=(
                    args.expected_preregistration_file_sha256
                ),
            )
            return validated, sha256_file(path)
        if not process_exists(args.roster_probe_pid):
            raise ActorV25BootstrapError(
                "stable roster probe exited before publishing its roster"
            )
        time.sleep(args.poll_seconds)


def wait_for_binding(
    args: argparse.Namespace,
    *,
    runner_process: subprocess.Popen[bytes],
    runner_sha256: str,
    roster_sha256: str,
) -> tuple[dict[str, Any], str]:
    while True:
        if args.runner_binding.exists() or args.runner_binding.is_symlink():
            return validate_runner_binding(
                args.runner_binding,
                runner=args.runner,
                runner_sha256=runner_sha256,
                roster=args.stable_seed_roster,
                roster_sha256=roster_sha256,
            )
        returncode = runner_process.poll()
        if returncode is not None:
            raise ActorV25BootstrapError(
                f"runner exited {returncode} before publishing its immutable binding"
            )
        time.sleep(args.poll_seconds)


def wait_for_guardian_handoff(
    args: argparse.Namespace,
    *,
    runner_process: subprocess.Popen[bytes],
    guardian_process: subprocess.Popen[bytes],
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Authenticate the guardian's immutable plan and live monitoring state."""

    plan_path = args.guardian_state_root / "immutable_guardian_plan.json"
    state_path = args.guardian_state_root / "guardian_state.json"
    deadline = time.monotonic() + args.guardian_handoff_timeout_seconds
    while True:
        guardian_returncode = guardian_process.poll()
        if guardian_returncode is not None:
            raise ActorV25BootstrapError(
                "guardian exited during authoritative handoff with return code "
                f"{guardian_returncode}"
            )
        runner_returncode = runner_process.poll()
        if runner_returncode is not None:
            raise ActorV25BootstrapError(
                "runner exited during guardian handoff with return code "
                f"{runner_returncode}"
            )

        plan: dict[str, Any] | None = None
        plan_sha256: str | None = None
        if plan_path.exists() or plan_path.is_symlink():
            plan = read_json(plan_path, "immutable guardian plan")
            verify_logical_sha(plan, "immutable guardian plan")
            if (
                plan.get("format") != GUARDIAN_PLAN_FORMAT
                or plan.get("guardian_format") != GUARDIAN_FORMAT
                or plan.get("initial_runner_pid") != runner_process.pid
            ):
                raise ActorV25BootstrapError(
                    "immutable guardian plan changed during handoff"
                )
            plan_sha256 = sha256_file(plan_path)

        state: dict[str, Any] | None = None
        state_sha256: str | None = None
        if state_path.exists() or state_path.is_symlink():
            state = read_json(state_path, "guardian monitoring state")
            if (
                state.get("format") != GUARDIAN_STATE_FORMAT
                or state.get("status") != "monitoring"
                or state.get("guardian_pid") != guardian_process.pid
                or state.get("managed_runner_pid") != runner_process.pid
                or state.get("runner_process_alive") is not True
            ):
                raise ActorV25BootstrapError(
                    "guardian monitoring state changed during handoff"
                )
            state_sha256 = sha256_file(state_path)

        if plan is not None and state is not None:
            # Recheck after both reads so an exit racing publication cannot be
            # mistaken for a completed authority handoff.
            if guardian_process.poll() is not None:
                raise ActorV25BootstrapError(
                    "guardian exited after publishing authoritative handoff"
                )
            if runner_process.poll() is not None:
                raise ActorV25BootstrapError(
                    "runner exited after guardian authenticated it"
                )
            assert plan_sha256 is not None and state_sha256 is not None
            return plan, plan_sha256, state, state_sha256

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ActorV25BootstrapError(
                "timed out waiting for authoritative guardian handoff state"
            )
        time.sleep(min(args.poll_seconds, remaining))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-python", type=Path, required=True)
    parser.add_argument("--guardian-python", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--guardian", type=Path, required=True)
    parser.add_argument("--guardian-sha256", required=True)
    parser.add_argument("--runner-cwd", type=Path, required=True)
    parser.add_argument("--guardian-cwd", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--stable-seed-roster", type=Path, required=True)
    parser.add_argument("--roster-probe-pid", type=int, required=True)
    parser.add_argument("--expected-preregistration-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-binding", type=Path, required=True)
    parser.add_argument("--guardian-state-root", type=Path, required=True)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--child-pythonpath", required=True)
    parser.add_argument("--vulkan-driver-files", type=Path, required=True)
    parser.add_argument("--nvidia-smi", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--guardian-poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--guardian-handoff-timeout-seconds", type=float, default=300.0
    )
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "runner_python",
        "guardian_python",
        "code_root",
        "runner",
        "guardian",
        "runner_cwd",
        "guardian_cwd",
        "actor_checkpoint",
        "vlm_metadata_path",
        "robotwin_root",
        "event_spec",
        "stable_seed_roster",
        "output",
        "runner_binding",
        "guardian_state_root",
        "bootstrap_root",
        "nvidia_smi",
        "vulkan_driver_files",
    ):
        setattr(args, name, getattr(args, name).expanduser().absolute())
    if (
        args.poll_seconds <= 0
        or args.guardian_poll_seconds <= 0
        or args.guardian_handoff_timeout_seconds <= 0
    ):
        raise ActorV25BootstrapError("poll intervals must be positive")
    if args.roster_probe_pid <= 0:
        raise ActorV25BootstrapError("roster probe PID must be positive")
    for name in (
        "runner_sha256",
        "guardian_sha256",
        "expected_preregistration_file_sha256",
    ):
        if not is_sha256(getattr(args, name)):
            raise ActorV25BootstrapError(f"{name} is not a lowercase SHA-256")
    if args.runner_binding != args.output / "immutable_deployment_binding.json":
        raise ActorV25BootstrapError(
            "runner binding must be output/immutable_deployment_binding.json"
        )
    return args


def validate_static_inputs(args: argparse.Namespace) -> dict[str, str]:
    runner_python = real_path(
        args.runner_python,
        label="runner Python",
        executable=True,
        allow_symlink=True,
    )
    guardian_python = real_path(
        args.guardian_python,
        label="guardian Python",
        executable=True,
        allow_symlink=True,
    )
    code_root = real_path(args.code_root, label="v25 code root", directory=True)
    runner = real_path(args.runner, label="v25 runner")
    guardian = real_path(args.guardian, label="v25 guardian")
    for path, label in ((runner, "runner"), (guardian, "guardian")):
        try:
            path.relative_to(code_root)
        except ValueError as error:
            raise ActorV25BootstrapError(
                f"{label} is outside the immutable v25 code root"
            ) from error
    if sha256_file(runner) != args.runner_sha256:
        raise ActorV25BootstrapError("v25 runner SHA-256 mismatch")
    if sha256_file(guardian) != args.guardian_sha256:
        raise ActorV25BootstrapError("v25 guardian SHA-256 mismatch")
    for name, directory in (
        ("runner_cwd", True),
        ("guardian_cwd", True),
        ("actor_checkpoint", True),
        ("vlm_metadata_path", True),
        ("robotwin_root", True),
        ("event_spec", False),
        ("nvidia_smi", False),
        ("vulkan_driver_files", False),
    ):
        real_path(
            getattr(args, name),
            label=name,
            directory=directory,
            allow_symlink=(name == "nvidia_smi"),
        )
    if args.output.exists() or args.output.is_symlink():
        raise ActorV25BootstrapError(
            "experiment output already exists; bootstrap never resumes or reruns"
        )
    if args.guardian_state_root.exists() or args.guardian_state_root.is_symlink():
        raise ActorV25BootstrapError("guardian state root already exists")
    if args.bootstrap_root.exists() or args.bootstrap_root.is_symlink():
        raise ActorV25BootstrapError("bootstrap root already exists")
    return {
        "runner_python": str(runner_python),
        "guardian_python": str(guardian_python),
        "code_root": str(code_root),
        "runner": str(runner),
        "guardian": str(guardian),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = normalize_args(parse_args(argv))
    static = validate_static_inputs(args)
    args.bootstrap_root.mkdir(parents=True, exist_ok=False)
    bootstrap_pid = args.bootstrap_root / "bootstrap.pid"
    create_once_bytes(
        bootstrap_pid,
        f"{os.getpid()}\n".encode("ascii"),
        "bootstrap PID",
    )
    plan_path = args.bootstrap_root / "bootstrap.plan.json"
    runner_log = args.bootstrap_root / "runner.log"
    guardian_log = args.bootstrap_root / "guardian.log"
    runner_pid_path = args.bootstrap_root / "runner.pid"
    guardian_pid_path = args.bootstrap_root / "guardian.pid"
    runner_launch_path = args.bootstrap_root / "runner.launch.json"
    guardian_launch_path = args.bootstrap_root / "guardian.launch.json"
    complete_path = args.bootstrap_root / "bootstrap.complete.json"
    failure_path = args.bootstrap_root / "bootstrap.failure.json"
    environment = os.environ.copy()
    environment.update(
        {
            "ASSETS_PATH": str(args.robotwin_root),
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": args.child_pythonpath,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "VK_DRIVER_FILES": str(args.vulkan_driver_files),
            "VK_ICD_FILENAMES": str(args.vulkan_driver_files),
        }
    )
    environment_contract = {
        key: environment[key]
        for key in (
            "ASSETS_PATH",
            "CUDA_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "PYTHONUNBUFFERED",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_OFFLINE",
            "VK_DRIVER_FILES",
            "VK_ICD_FILENAMES",
        )
    }
    runner_process: subprocess.Popen[bytes] | None = None
    guardian_process: subprocess.Popen[bytes] | None = None
    handoff_complete = False
    try:
        plan = signed(
            {
                "format": PLAN_FORMAT,
                "created_at_utc": utc_now(),
                "bootstrap_format": FORMAT,
                "static": static,
                "runner_sha256": args.runner_sha256,
                "guardian_sha256": args.guardian_sha256,
                "stable_seed_roster": str(args.stable_seed_roster),
                "roster_probe_pid": args.roster_probe_pid,
                "expected_preregistration_file_sha256": (
                    args.expected_preregistration_file_sha256
                ),
                "output_must_not_preexist": True,
                "actor_results_or_labels_read": False,
                "child_start_new_session": True,
                "guardian_handoff_timeout_seconds": (
                    args.guardian_handoff_timeout_seconds
                ),
                "child_environment_contract": environment_contract,
            }
        )
        create_once_json(plan_path, plan, "bootstrap plan")
        roster, roster_sha = wait_for_roster(args)
        if args.output.exists() or args.output.is_symlink():
            raise ActorV25BootstrapError(
                "experiment output appeared while waiting; refusing to rerun"
            )
        command = runner_command(args, roster_sha)
        runner_process = launch_child(
            command,
            cwd=args.runner_cwd,
            log_path=runner_log,
            environment=environment,
        )
        create_once_bytes(
            runner_pid_path,
            f"{runner_process.pid}\n".encode("ascii"),
            "runner PID",
        )
        runner_launch = signed(
            {
                "format": RUNNER_LAUNCH_FORMAT,
                "launched_at_utc": utc_now(),
                "pid": runner_process.pid,
                "session_id": os.getsid(runner_process.pid),
                "start_new_session": True,
                "command": command,
                "cwd": str(args.runner_cwd),
                "log": str(runner_log),
                "runner_sha256": args.runner_sha256,
                "stable_seed_roster_file_sha256": roster_sha,
                "stable_seed_roster_logical_sha256": roster["logical_sha256"],
                "actor_results_or_labels_read": False,
                "child_environment_contract": environment_contract,
            }
        )
        create_once_json(runner_launch_path, runner_launch, "runner launch receipt")
        binding, binding_sha = wait_for_binding(
            args,
            runner_process=runner_process,
            runner_sha256=args.runner_sha256,
            roster_sha256=roster_sha,
        )
        if runner_process.poll() is not None:
            raise ActorV25BootstrapError(
                "runner disappeared immediately after publishing its binding"
            )
        if (
            args.guardian_state_root.exists()
            or args.guardian_state_root.is_symlink()
        ):
            raise ActorV25BootstrapError(
                "guardian state root appeared before guardian launch"
            )
        command = guardian_command(
            args, runner_pid=runner_process.pid, roster_sha256=roster_sha
        )
        guardian_process = launch_child(
            command,
            cwd=args.guardian_cwd,
            log_path=guardian_log,
            environment=environment,
        )
        create_once_bytes(
            guardian_pid_path,
            f"{guardian_process.pid}\n".encode("ascii"),
            "guardian PID",
        )
        guardian_launch = signed(
            {
                "format": GUARDIAN_LAUNCH_FORMAT,
                "launched_at_utc": utc_now(),
                "pid": guardian_process.pid,
                "session_id": os.getsid(guardian_process.pid),
                "start_new_session": True,
                "command": command,
                "cwd": str(args.guardian_cwd),
                "log": str(guardian_log),
                "guardian_sha256": args.guardian_sha256,
                "runner_pid": runner_process.pid,
                "runner_binding": str(args.runner_binding),
                "runner_binding_file_sha256": binding_sha,
                "runner_binding_logical_sha256": binding["logical_sha256"],
                "actor_results_or_labels_read": False,
                "child_environment_contract": environment_contract,
            }
        )
        create_once_json(
            guardian_launch_path, guardian_launch, "guardian launch receipt"
        )
        guardian_plan, guardian_plan_sha, guardian_state, guardian_state_sha = (
            wait_for_guardian_handoff(
                args,
                runner_process=runner_process,
                guardian_process=guardian_process,
            )
        )
        complete = signed(
            {
                "format": COMPLETE_FORMAT,
                "status": "runner_and_guardian_detached_bootstrap_exiting",
                "completed_at_utc": utc_now(),
                "runner_pid": runner_process.pid,
                "guardian_pid": guardian_process.pid,
                "runner_launch_receipt": str(runner_launch_path),
                "runner_launch_receipt_file_sha256": sha256_file(
                    runner_launch_path
                ),
                "guardian_launch_receipt": str(guardian_launch_path),
                "guardian_launch_receipt_file_sha256": sha256_file(
                    guardian_launch_path
                ),
                "runner_binding_file_sha256": binding_sha,
                "guardian_plan": str(
                    args.guardian_state_root / "immutable_guardian_plan.json"
                ),
                "guardian_plan_file_sha256": guardian_plan_sha,
                "guardian_plan_logical_sha256": guardian_plan["logical_sha256"],
                "guardian_state": str(
                    args.guardian_state_root / "guardian_state.json"
                ),
                "guardian_state_file_sha256_at_handoff": guardian_state_sha,
                "guardian_state_status_at_handoff": guardian_state["status"],
                "actor_results_or_labels_read": False,
                "bootstrap_remains_resident": False,
            }
        )
        create_once_json(complete_path, complete, "bootstrap completion receipt")
        handoff_complete = True
        print(
            "ACTOR_V25_BOOTSTRAP_COMPLETE="
            + json.dumps(
                {
                    "runner_pid": runner_process.pid,
                    "guardian_pid": guardian_process.pid,
                    "completion": str(complete_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        cleanup_audit: list[dict[str, Any]] = []
        cleanup_errors: list[dict[str, str]] = []
        if not handoff_complete:
            for label, process in (
                ("guardian", guardian_process),
                ("runner", runner_process),
            ):
                try:
                    cleanup_audit.append(
                        terminate_owned_session(process, label=label)
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(
                        {
                            "label": label,
                            "error_type": type(cleanup_error).__name__,
                            "error_message": str(cleanup_error),
                        }
                    )
        failure = signed(
            {
                "format": FAILURE_FORMAT,
                "status": "failed_closed_no_relaunch",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "actor_results_or_labels_read": False,
                "handoff_complete": handoff_complete,
                "cleanup_audit": cleanup_audit,
                "cleanup_errors": cleanup_errors,
                "owned_child_cleanup_complete": not cleanup_errors,
                "orphaned_owned_children": False if not cleanup_errors else None,
            }
        )
        try:
            create_once_json(failure_path, failure, "bootstrap failure receipt")
        except Exception:
            pass
        if cleanup_errors:
            details = "; ".join(
                f"{item['label']}: {item['error_type']}: "
                f"{item['error_message']}"
                for item in cleanup_errors
            )
            raise ActorV25BootstrapError(
                f"{type(error).__name__}: {error}; owned child cleanup failed: "
                f"{details}"
            ) from error
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


__all__ = [
    "ActorV25BootstrapError",
    "guardian_command",
    "launch_child",
    "normalize_args",
    "parse_args",
    "process_exists",
    "runner_command",
    "terminate_owned_session",
    "validate_roster",
    "validate_runner_binding",
    "validate_static_inputs",
    "wait_for_guardian_handoff",
]
