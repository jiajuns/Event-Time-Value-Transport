#!/usr/bin/env python3
"""Fail-closed guardian for the detached 200-pair actor protocol run.

The guardian is intentionally narrower than the experiment runner.  It never
opens observations, labels, method results, or pair outcomes, and it never
decides which deployment protocol is better.  It only authenticates the
already-bound static inputs, watches the exact runner process and terminal
receipts, and grants one process-level restart after a silent disappearance.

The runner's create-once resume contract remains authoritative.  In
particular, this process never edits or removes anything below ``--output``.
Any method/pair failure receipt is terminal.  A second silent disappearance is
also terminal; it is not converted into another sample or retry.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_guardian_v1"
PLAN_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_guardian_plan_v1"
STATE_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_guardian_state_v1"
LOG_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_guardian_log_v1"
EXIT_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_guardian_exit_v1"
RUNNER_FILENAME = "run_robotwin2_five_body_actor_execute5_vs_execute50_v1.py"
RUNNER_FORMAT = "etsf_robotwin2_five_body_actor_execute5_vs_execute50_v2_stable_roster"
BINDING_FORMAT = "etsf_robotwin2_actor_deployment_protocol_binding_v2_stable_roster"
COMPLETION_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_completion_v1"
COMPLETION_STATUS = "complete_200_pairs_400_rollouts_frozen"
PAIR_COUNT = 200
ROLLOUT_COUNT = 400
RESTART_LIMIT = 1
GPU_NAME_TOKEN = "RTX 4090"
SHA256_CHARS = frozenset("0123456789abcdef")
RESTART_ENVIRONMENT_KEYS = (
    "ASSETS_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "HOME",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_OFFLINE",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "XDG_CACHE_HOME",
)


class GuardianContractError(RuntimeError):
    """Static binding, process identity, or terminal receipt is invalid."""


class ExperimentFailure(RuntimeError):
    """The experiment published a failure or exhausted its one restart."""


class GuardianLockHeld(RuntimeError):
    """A live guardian already owns this state root."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    """Match the runner's ordered path/size/content checkpoint-tree hash."""

    root = path.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise GuardianContractError("bound model tree must be a real directory")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise GuardianContractError("bound model tree contains a symbolic link")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise GuardianContractError("bound model tree contains a special file")
        rows.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise GuardianContractError("bound model tree is empty")
    return canonical_sha256(rows), len(rows), sum(row["size_bytes"] for row in rows)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_CHARS)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path: Path, value: str, *, mode: int = 0o644) -> None:
    """Atomically replace guardian-owned mutable state, never experiment data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create an atomic create-once guardian receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        existing = load_json(path, role=path.name)
        if existing != dict(value):
            raise GuardianContractError(f"create-once guardian receipt changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = load_json(path, role=path.name)
            if existing != dict(value):
                raise GuardianContractError(
                    f"create-once guardian receipt race changed: {path}"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GuardianContractError(f"{role} must be a materialized regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GuardianContractError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise GuardianContractError(f"{role} must be a JSON object")
    return value


def _absolute(path: Path, *, role: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise GuardianContractError(f"{role} must use an absolute path")
    return Path(os.path.abspath(os.fspath(expanded)))


def _existing(
    path: Path, *, role: str, directory: bool, executable: bool = False
) -> tuple[Path, Path]:
    invocation = _absolute(path, role=role)
    resolved = invocation.resolve(strict=True)
    mode = resolved.stat().st_mode
    valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not valid:
        raise GuardianContractError(
            f"{role} must resolve to a {'directory' if directory else 'regular file'}"
        )
    if executable and not os.access(resolved, os.X_OK):
        raise GuardianContractError(f"{role} is not executable")
    return invocation, resolved


def build_runner_command(args: argparse.Namespace) -> list[str]:
    runner = _absolute(args.code_root, role="code root") / "scripts" / RUNNER_FILENAME
    return [
        str(_absolute(args.python_bin, role="Python executable")),
        str(runner),
        "--actor-checkpoint",
        str(_absolute(args.actor_checkpoint, role="actor checkpoint")),
        "--vlm-metadata-path",
        str(_absolute(args.vlm_metadata_path, role="VLM metadata")),
        "--robotwin-root",
        str(_absolute(args.robotwin_root, role="RoboTwin root")),
        "--event-spec",
        str(_absolute(args.event_spec, role="event spec")),
        "--stable-seed-roster",
        str(_absolute(args.stable_seed_roster, role="stable seed roster")),
        "--stable-seed-roster-sha256",
        args.stable_seed_roster_sha256,
        "--output",
        str(_absolute(args.output, role="experiment output")),
    ]


def _walk_path_hash_bindings(value: Any) -> list[tuple[str, str, int | None]]:
    rows: list[tuple[str, str, int | None]] = []
    if isinstance(value, Mapping):
        path = value.get("path")
        digest = value.get("sha256")
        size = value.get("size_bytes")
        if isinstance(path, str) and _is_sha256(digest):
            rows.append((path, str(digest), int(size) if isinstance(size, int) else None))
        for child in value.values():
            rows.extend(_walk_path_hash_bindings(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_path_hash_bindings(child))
    return rows


def validate_binding_and_static_paths(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate every command path against the runner-created binding."""

    python_invocation, python_resolved = _existing(
        args.python_bin, role="Python executable", directory=False, executable=True
    )
    code_invocation, code_resolved = _existing(
        args.code_root, role="immutable code root", directory=True
    )
    actor_invocation, actor_resolved = _existing(
        args.actor_checkpoint, role="actor checkpoint", directory=True
    )
    vlm_invocation, vlm_resolved = _existing(
        args.vlm_metadata_path, role="VLM metadata", directory=True
    )
    robotwin_invocation, robotwin_resolved = _existing(
        args.robotwin_root, role="RoboTwin root", directory=True
    )
    event_invocation, event_resolved = _existing(
        args.event_spec, role="event spec", directory=False
    )
    roster_invocation, roster_resolved = _existing(
        args.stable_seed_roster, role="stable seed roster", directory=False
    )
    _output_invocation, output_resolved = _existing(
        args.output, role="experiment output", directory=True
    )
    state_root = _absolute(args.state_root, role="guardian state root")
    state_resolved = state_root.resolve(strict=False)
    if (
        state_resolved == output_resolved
        or output_resolved in state_resolved.parents
        or state_resolved in output_resolved.parents
    ):
        raise GuardianContractError(
            "guardian state and experiment artifact trees must be disjoint"
        )

    runner_invocation = code_invocation / "scripts" / RUNNER_FILENAME
    runner_resolved = runner_invocation.resolve(strict=True)
    if not runner_resolved.is_file() or runner_resolved.is_symlink():
        raise GuardianContractError("bound actor protocol runner is invalid")
    if code_resolved not in runner_resolved.parents:
        raise GuardianContractError("actor protocol runner escaped immutable code root")

    binding_path = output_resolved / "immutable_deployment_binding.json"
    binding = load_json(binding_path, role="immutable deployment binding")
    if binding.get("format") != BINDING_FORMAT or binding.get("runner_format") != RUNNER_FORMAT:
        raise GuardianContractError("deployment binding has the wrong protocol format")
    logical = binding.get("logical_sha256")
    if not _is_sha256(logical):
        raise GuardianContractError("deployment binding logical SHA is malformed")
    base = dict(binding)
    del base["logical_sha256"]
    if canonical_sha256(base) != logical:
        raise GuardianContractError("deployment binding logical SHA changed")

    exact_paths = {
        "runner_path": runner_resolved,
        "actor_checkpoint": actor_resolved,
        "vlm_metadata_path": vlm_resolved,
        "robotwin_root": robotwin_resolved,
        "event_spec": event_resolved,
    }
    for field, expected in exact_paths.items():
        supplied = binding.get(field)
        if not isinstance(supplied, str) or Path(supplied).resolve(strict=True) != expected:
            raise GuardianContractError(f"deployment binding changed {field}")

    runner_sha = sha256_file(runner_resolved)
    event_sha = sha256_file(event_resolved)
    roster_sha = sha256_file(roster_resolved)
    if runner_sha != binding.get("runner_sha256"):
        raise GuardianContractError("actor protocol runner SHA changed")
    if event_sha != binding.get("event_spec_sha256"):
        raise GuardianContractError("event specification SHA changed")
    roster_binding = binding.get("stable_seed_roster_binding")
    if (
        not isinstance(roster_binding, Mapping)
        or roster_binding.get("path") != str(roster_resolved)
        or roster_binding.get("file_sha256") != roster_sha
        or args.stable_seed_roster_sha256 != roster_sha
        or not _is_sha256(roster_binding.get("logical_sha256"))
        or not _is_sha256(roster_binding.get("preregistration_file_sha256"))
        or not _is_sha256(roster_binding.get("preregistration_logical_sha256"))
        or roster_binding.get("selection_uses_labels_or_outcomes") is not False
        or roster_binding.get("actor_inference_calls_during_selection") != 0
    ):
        raise GuardianContractError("stable seed roster binding changed")
    materializer = Path(str(roster_binding.get("materializer_path", ""))).resolve(
        strict=True
    )
    if (
        not materializer.is_file()
        or materializer.is_symlink()
        or code_resolved not in materializer.parents
        or sha256_file(materializer) != roster_binding.get("materializer_file_sha256")
    ):
        raise GuardianContractError("stable seed roster materializer changed")

    actor_tree = sha256_tree(actor_resolved)
    vlm_tree = sha256_tree(vlm_resolved)
    expected_actor = (
        binding.get("actor_checkpoint_tree_sha256"),
        binding.get("actor_checkpoint_file_count"),
        binding.get("actor_checkpoint_size_bytes"),
    )
    expected_vlm = (
        binding.get("vlm_metadata_tree_sha256"),
        binding.get("vlm_metadata_file_count"),
        binding.get("vlm_metadata_size_bytes"),
    )
    if actor_tree != expected_actor:
        raise GuardianContractError("actor checkpoint tree changed")
    if vlm_tree != expected_vlm:
        raise GuardianContractError("VLM metadata tree changed")

    runtime_rows = _walk_path_hash_bindings(binding.get("runtime_binding"))
    if not runtime_rows:
        raise GuardianContractError("runtime binding contains no authenticated files")
    authenticated_runtime_files = []
    for raw_path, expected_sha, expected_size in runtime_rows:
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise GuardianContractError("runtime binding path is not a real file")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise GuardianContractError(f"runtime binding size changed: {path}")
        if sha256_file(path) != expected_sha:
            raise GuardianContractError(f"runtime binding SHA changed: {path}")
        authenticated_runtime_files.append(str(path))

    return {
        "python_invocation_path": str(python_invocation),
        "python_resolved_path": str(python_resolved),
        "python_sha256": sha256_file(python_resolved),
        "code_root_invocation_path": str(code_invocation),
        "code_root_resolved_path": str(code_resolved),
        "runner_path": str(runner_resolved),
        "runner_sha256": runner_sha,
        "actor_checkpoint_path": str(actor_resolved),
        "actor_checkpoint_tree_sha256": actor_tree[0],
        "actor_checkpoint_file_count": actor_tree[1],
        "actor_checkpoint_size_bytes": actor_tree[2],
        "vlm_metadata_path": str(vlm_resolved),
        "vlm_metadata_tree_sha256": vlm_tree[0],
        "vlm_metadata_file_count": vlm_tree[1],
        "vlm_metadata_size_bytes": vlm_tree[2],
        "robotwin_root": str(robotwin_resolved),
        "event_spec_path": str(event_resolved),
        "event_spec_sha256": event_sha,
        "stable_seed_roster_path": str(roster_resolved),
        "stable_seed_roster_file_sha256": roster_sha,
        "output_path": str(output_resolved),
        "binding_path": str(binding_path),
        "binding_logical_sha256": str(logical),
        "binding_file_sha256": sha256_file(binding_path),
        "authenticated_runtime_file_count": len(authenticated_runtime_files),
        "authenticated_runtime_files_sha256": canonical_sha256(
            sorted(authenticated_runtime_files)
        ),
    }


def query_gpu_contract(nvidia_smi: Path, gpu_uuid: str) -> dict[str, Any]:
    invocation, resolved = _existing(
        nvidia_smi, role="nvidia-smi", directory=False, executable=True
    )
    completed = subprocess.run(
        [
            str(invocation),
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise GuardianContractError(f"nvidia-smi failed: {completed.stderr.strip()}")
    matches = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3 and fields[1] == gpu_uuid:
            matches.append(fields)
    if len(matches) != 1:
        raise GuardianContractError("requested GPU UUID is absent or ambiguous")
    index, observed_uuid, name = matches[0]
    if GPU_NAME_TOKEN not in name:
        raise GuardianContractError("requested GPU UUID is not an RTX 4090")
    args_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selector = str(args_value)
    # This environment check is only a launch-time convenience.  The exact
    # initial runner environment is separately captured and authenticated.
    if selector and selector not in (index, observed_uuid):
        raise GuardianContractError(
            "guardian CUDA_VISIBLE_DEVICES does not select the requested GPU UUID"
        )
    return {
        "uuid": observed_uuid,
        "physical_index": index,
        "name": name,
        "nvidia_smi_invocation_path": str(invocation),
        "nvidia_smi_resolved_path": str(resolved),
        "nvidia_smi_sha256": sha256_file(resolved),
        "guardian_cuda_visible_devices": args_value,
    }


def _proc_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError as error:
        raise ProcessLookupError(pid) from error
    closing = raw.rfind(")")
    if closing < 0:
        raise GuardianContractError("runner /proc stat is malformed")
    fields_after_comm = raw[closing + 2 :].split()
    if len(fields_after_comm) <= 19:
        raise GuardianContractError("runner /proc stat lacks start time")
    if fields_after_comm[0] in ("Z", "X", "x"):
        raise ProcessLookupError(pid)
    return int(fields_after_comm[19])


def _proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError as error:
        raise ProcessLookupError(pid) from error
    if not raw:
        raise ProcessLookupError(pid)
    return [part.decode("utf-8") for part in raw.split(b"\0") if part]


def _proc_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except FileNotFoundError as error:
        raise ProcessLookupError(pid) from error
    if not raw:
        raise ProcessLookupError(pid)
    result: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode("utf-8")] = value.decode("utf-8")
    return result


def validate_process_identity(
    pid: int,
    *,
    expected_start_ticks: int | None,
    command: Sequence[str],
    cwd: Path,
    gpu_contract: Mapping[str, Any],
) -> tuple[int, dict[str, str]]:
    """Reject PID reuse, a different command/cwd, or the wrong visible GPU."""

    start_ticks = _proc_start_ticks(pid)
    if expected_start_ticks is not None and start_ticks != expected_start_ticks:
        raise ProcessLookupError(pid)
    if _proc_cmdline(pid) != list(command):
        raise GuardianContractError("managed PID command differs from frozen runner command")
    try:
        observed_cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
    except FileNotFoundError as error:
        raise ProcessLookupError(pid) from error
    if observed_cwd != cwd.resolve(strict=True):
        raise GuardianContractError("managed runner is not in the bound RoboTwin cwd")
    environment = _proc_environment(pid)
    selector = environment.get("CUDA_VISIBLE_DEVICES", "")
    if selector not in (
        str(gpu_contract["physical_index"]),
        str(gpu_contract["uuid"]),
    ):
        raise GuardianContractError("managed runner does not select the bound GPU UUID")
    restart_environment = {
        key: environment[key]
        for key in RESTART_ENVIRONMENT_KEYS
        if key in environment
    }
    return start_ticks, restart_environment


def _failure_entries(output: Path) -> list[str]:
    entries: list[str] = []
    for directory_name in ("method_failures", "pair_failures"):
        directory = output / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise GuardianContractError(
                f"experiment failure directory is missing or invalid: {directory_name}"
            )
        for candidate in sorted(directory.iterdir()):
            if candidate.is_symlink() or not candidate.is_file():
                raise GuardianContractError(
                    f"experiment failure directory contains a special entry: {candidate}"
                )
            # A staged failure is authoritative to the runner too.  Do not
            # mistake it for a restartable interruption.
            entries.append(str(candidate.resolve()))
    return entries


def _count_final_json(directory: Path, *, role: str) -> int:
    if directory.is_symlink() or not directory.is_dir():
        raise GuardianContractError(f"{role} directory is invalid")
    count = 0
    for candidate in directory.iterdir():
        if candidate.name.startswith("."):
            raise GuardianContractError(f"{role} contains an unpromoted staged artifact")
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix != ".json":
            raise GuardianContractError(f"{role} contains an unexpected artifact")
        count += 1
    return count


def validate_completion(
    output: Path, *, binding_logical_sha256: str, binding_file_sha256: str
) -> dict[str, Any] | None:
    path = output / "run.complete.json"
    if not path.exists() and not path.is_symlink():
        return None
    completion = load_json(path, role="completion receipt")
    if (
        completion.get("format") != COMPLETION_FORMAT
        or completion.get("status") != COMPLETION_STATUS
        or completion.get("pair_count") != PAIR_COUNT
        or completion.get("rollout_count") != ROLLOUT_COUNT
        or completion.get("binding_logical_sha256") != binding_logical_sha256
        or completion.get("binding_file_sha256") != binding_file_sha256
    ):
        raise GuardianContractError("completion receipt changed its frozen contract")
    logical = completion.get("logical_sha256")
    if not _is_sha256(logical):
        raise GuardianContractError("completion receipt logical SHA is malformed")
    base = dict(completion)
    del base["logical_sha256"]
    if canonical_sha256(base) != logical:
        raise GuardianContractError("completion receipt logical SHA changed")
    outcome = output / "paired_outcomes.json"
    report = output / "paired_report.json"
    if (
        not outcome.is_file()
        or outcome.is_symlink()
        or sha256_file(outcome) != completion.get("outcome_file_sha256")
        or not report.is_file()
        or report.is_symlink()
        or sha256_file(report) != completion.get("report_file_sha256")
    ):
        raise GuardianContractError("completion outcome/report file binding changed")
    expected_counts = {
        "pairs": PAIR_COUNT,
        "attempts": PAIR_COUNT,
        "initial_commitments": PAIR_COUNT,
        "method_starts": ROLLOUT_COUNT,
        "method_results": ROLLOUT_COUNT,
    }
    for directory_name, expected_count in expected_counts.items():
        observed = _count_final_json(output / directory_name, role=directory_name)
        if observed != expected_count:
            raise GuardianContractError(
                f"completion {directory_name} count is {observed}, expected {expected_count}"
            )
    return completion


def observe_artifacts(
    output: Path, *, binding_logical_sha256: str, binding_file_sha256: str
) -> dict[str, Any]:
    failures = _failure_entries(output)
    if failures:
        return {"kind": "failure", "failure_receipts": failures}
    completion = validate_completion(
        output,
        binding_logical_sha256=binding_logical_sha256,
        binding_file_sha256=binding_file_sha256,
    )
    if completion is not None:
        return {"kind": "complete", "completion": completion}
    progress_path = output / "progress.json"
    progress = None
    if progress_path.exists() or progress_path.is_symlink():
        progress = load_json(progress_path, role="runner progress")
    completed_pairs = _count_final_json(output / "pairs", role="pairs")
    return {
        "kind": "running",
        "completed_pairs": completed_pairs,
        "progress": progress,
    }


def next_action(*, artifact_kind: str, process_alive: bool, restart_count: int) -> str:
    """Pure precedence rule used by the monitor and unit tests."""

    if artifact_kind == "failure":
        return "fail_experiment_receipt"
    if artifact_kind == "complete":
        return "complete"
    if artifact_kind != "running":
        raise GuardianContractError("unknown artifact observation")
    if process_alive:
        return "wait"
    if restart_count < RESTART_LIMIT:
        return "restart_once"
    return "fail_restart_exhausted"


def _read_optional_state(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    value = load_json(path, role="guardian state")
    if value.get("format") != STATE_FORMAT:
        raise GuardianContractError("guardian state format changed")
    return value


def _append_log(log_path: Path, events: list[dict[str, Any]], event: Mapping[str, Any]) -> None:
    events.append({"time_unix": time.time(), **dict(event)})
    atomic_json(log_path, {"format": LOG_FORMAT, "events": events})


def _load_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists() and not log_path.is_symlink():
        return []
    value = load_json(log_path, role="guardian log")
    if value.get("format") != LOG_FORMAT or not isinstance(value.get("events"), list):
        raise GuardianContractError("guardian log format changed")
    if not all(isinstance(row, dict) for row in value["events"]):
        raise GuardianContractError("guardian log contains a non-object event")
    return [dict(row) for row in value["events"]]


def _capture_plan(
    args: argparse.Namespace,
    *,
    static: Mapping[str, Any],
    gpu: Mapping[str, Any],
    command: Sequence[str],
    initial_start_ticks: int | None,
    restart_environment: Mapping[str, str],
) -> dict[str, Any]:
    base = {
        "format": PLAN_FORMAT,
        "guardian_format": FORMAT,
        "runner_command": list(command),
        "runner_cwd": str(_absolute(args.robotwin_root, role="RoboTwin root")),
        "initial_runner_pid": int(args.runner_pid),
        "initial_runner_start_ticks": initial_start_ticks,
        "restart_limit": RESTART_LIMIT,
        "poll_seconds": float(args.poll_seconds),
        "static_contract": dict(static),
        "gpu_contract": dict(gpu),
        "restart_environment": dict(restart_environment),
        "experiment_output_read_only_for_guardian": True,
        "runner_resume_contract_is_final_authority": True,
    }
    return {**base, "logical_sha256": canonical_sha256(base)}


def _validate_existing_plan(
    plan: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    static: Mapping[str, Any],
    gpu: Mapping[str, Any],
    command: Sequence[str],
) -> None:
    if plan.get("format") != PLAN_FORMAT or plan.get("guardian_format") != FORMAT:
        raise GuardianContractError("guardian plan format changed")
    base = dict(plan)
    logical = base.pop("logical_sha256", None)
    if not _is_sha256(logical) or canonical_sha256(base) != logical:
        raise GuardianContractError("guardian plan logical SHA changed")
    if (
        plan.get("runner_command") != list(command)
        or plan.get("runner_cwd")
        != str(_absolute(args.robotwin_root, role="RoboTwin root"))
        or plan.get("initial_runner_pid") != int(args.runner_pid)
        or plan.get("restart_limit") != RESTART_LIMIT
        or plan.get("poll_seconds") != float(args.poll_seconds)
        or plan.get("static_contract") != dict(static)
    ):
        raise GuardianContractError("guardian relaunch changed its frozen plan")
    saved_gpu = plan.get("gpu_contract")
    if not isinstance(saved_gpu, Mapping) or any(
        saved_gpu.get(key) != gpu.get(key)
        for key in (
            "uuid",
            "physical_index",
            "name",
            "nvidia_smi_resolved_path",
            "nvidia_smi_sha256",
        )
    ):
        raise GuardianContractError("guardian relaunch changed its GPU contract")


def _write_state(state_path: Path, state: Mapping[str, Any]) -> None:
    atomic_json(state_path, {"format": STATE_FORMAT, **dict(state)})


def _write_exit(
    exit_path: Path,
    *,
    status: str,
    exit_code: int,
    message: str,
    restart_count: int,
) -> dict[str, Any]:
    base = {
        "format": EXIT_FORMAT,
        "status": status,
        "exit_code": exit_code,
        "message": message,
        "restart_count": restart_count,
    }
    value = {
        **base,
        "logical_sha256": canonical_sha256(base),
    }
    immutable_json(exit_path, value)
    return value


def execute(args: argparse.Namespace) -> int:
    if not 0.1 <= args.poll_seconds <= 60.0:
        raise GuardianContractError("poll interval must be in [0.1, 60] seconds")
    if args.runner_pid <= 0:
        raise GuardianContractError("initial runner PID must be positive")
    state_root = _absolute(args.state_root, role="guardian state root")
    state_root.mkdir(parents=True, exist_ok=True)
    if state_root.is_symlink() or not state_root.is_dir():
        raise GuardianContractError("guardian state root is invalid")
    lock_path = state_root / "guardian.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GuardianLockHeld("another guardian instance holds the lock") from error

        state_path = state_root / "guardian_state.json"
        pid_path = state_root / "guardian.pid"
        log_path = state_root / "guardian_log.json"
        exit_path = state_root / "guardian_exit.json"
        plan_path = state_root / "immutable_guardian_plan.json"
        atomic_text(pid_path, f"{os.getpid()}\n", mode=0o444)
        events = _load_log(log_path)

        static = validate_binding_and_static_paths(args)
        gpu = query_gpu_contract(args.nvidia_smi, args.gpu_uuid)
        command = build_runner_command(args)
        output = Path(static["output_path"])
        cwd = Path(static["robotwin_root"])
        observation = observe_artifacts(
            output,
            binding_logical_sha256=str(static["binding_logical_sha256"]),
            binding_file_sha256=str(static["binding_file_sha256"]),
        )

        existing_plan = None
        if plan_path.exists() or plan_path.is_symlink():
            existing_plan = load_json(plan_path, role="immutable guardian plan")
            _validate_existing_plan(
                existing_plan,
                args,
                static=static,
                gpu=gpu,
                command=command,
            )

        prior_state = _read_optional_state(state_path)
        restart_count = int(prior_state.get("restart_count", 0)) if prior_state else 0
        managed_pid = (
            int(prior_state.get("managed_runner_pid", args.runner_pid))
            if prior_state
            else int(args.runner_pid)
        )
        managed_start_ticks = (
            int(prior_state["managed_runner_start_ticks"])
            if prior_state and isinstance(prior_state.get("managed_runner_start_ticks"), int)
            else None
        )
        if restart_count < 0 or restart_count > RESTART_LIMIT:
            raise GuardianContractError("guardian state restart count is invalid")

        restart_environment: dict[str, str]
        process_alive = False
        try:
            start_ticks, observed_environment = validate_process_identity(
                managed_pid,
                expected_start_ticks=managed_start_ticks,
                command=command,
                cwd=cwd,
                gpu_contract=gpu,
            )
            process_alive = True
            managed_start_ticks = start_ticks
            restart_environment = observed_environment
        except ProcessLookupError:
            restart_environment = {}

        if existing_plan is None:
            if observation["kind"] == "running" and not process_alive:
                raise GuardianContractError(
                    "first guardian launch cannot authenticate the initial runner process"
                )
            plan = _capture_plan(
                args,
                static=static,
                gpu=gpu,
                command=command,
                initial_start_ticks=managed_start_ticks,
                restart_environment=restart_environment,
            )
            immutable_json(plan_path, plan)
        else:
            plan = existing_plan
            frozen_environment = plan.get("restart_environment")
            if not isinstance(frozen_environment, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in frozen_environment.items()
            ):
                raise GuardianContractError("guardian plan restart environment is invalid")
            restart_environment = dict(frozen_environment)

        _append_log(
            log_path,
            events,
            {
                "event": "guardian_monitor_started",
                "guardian_pid": os.getpid(),
                "managed_runner_pid": managed_pid,
                "restart_count": restart_count,
                "completed_pairs": observation.get("completed_pairs"),
            },
        )

        while True:
            observation = observe_artifacts(
                output,
                binding_logical_sha256=str(static["binding_logical_sha256"]),
                binding_file_sha256=str(static["binding_file_sha256"]),
            )
            try:
                current_start_ticks, _environment = validate_process_identity(
                    managed_pid,
                    expected_start_ticks=managed_start_ticks,
                    command=command,
                    cwd=cwd,
                    gpu_contract=gpu,
                )
                process_alive = True
                managed_start_ticks = current_start_ticks
            except ProcessLookupError:
                process_alive = False

            action = next_action(
                artifact_kind=str(observation["kind"]),
                process_alive=process_alive,
                restart_count=restart_count,
            )
            state = {
                "status": "monitoring",
                "guardian_pid": os.getpid(),
                "managed_runner_pid": managed_pid,
                "managed_runner_start_ticks": managed_start_ticks,
                "runner_process_alive": process_alive,
                "restart_count": restart_count,
                "restart_limit": RESTART_LIMIT,
                "completed_pairs": observation.get("completed_pairs"),
                "last_observation_kind": observation["kind"],
                "last_action": action,
                "last_poll_unix": time.time(),
            }
            _write_state(state_path, state)

            if action == "complete":
                _append_log(
                    log_path,
                    events,
                    {
                        "event": "experiment_complete",
                        "managed_runner_pid": managed_pid,
                        "restart_count": restart_count,
                    },
                )
                state["status"] = "complete"
                _write_state(state_path, state)
                _write_exit(
                    exit_path,
                    status="complete",
                    exit_code=0,
                    message="authenticated 200-pair/400-rollout completion receipt",
                    restart_count=restart_count,
                )
                return 0
            if action == "fail_experiment_receipt":
                receipts = observation.get("failure_receipts", [])
                raise ExperimentFailure(
                    "method/pair failure receipt exists: " + ", ".join(receipts)
                )
            if action == "fail_restart_exhausted":
                raise ExperimentFailure(
                    "managed runner disappeared a second time without completion/failure"
                )
            if action == "restart_once":
                restart_count += 1
                state.update(
                    {
                        "status": "restarting_once",
                        "restart_count": restart_count,
                        "last_action": "restart_once_committed",
                    }
                )
                _write_state(state_path, state)
                _append_log(
                    log_path,
                    events,
                    {
                        "event": "silent_disappearance_restart_committed",
                        "previous_runner_pid": managed_pid,
                        "restart_count": restart_count,
                    },
                )
                environment = os.environ.copy()
                for key in RESTART_ENVIRONMENT_KEYS:
                    environment.pop(key, None)
                environment.update(restart_environment)
                selector = environment.get("CUDA_VISIBLE_DEVICES", "")
                if selector not in (str(gpu["physical_index"]), str(gpu["uuid"])):
                    raise GuardianContractError(
                        "frozen restart environment no longer selects bound GPU UUID"
                    )
                restart_log = state_root / "runner_restart_1.stdout.log"
                descriptor = os.open(
                    restart_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
                )
                try:
                    child = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=descriptor,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        close_fds=True,
                    )
                finally:
                    os.close(descriptor)
                managed_pid = child.pid
                try:
                    managed_start_ticks, _ = validate_process_identity(
                        managed_pid,
                        expected_start_ticks=None,
                        command=command,
                        cwd=cwd,
                        gpu_contract=gpu,
                    )
                except ProcessLookupError:
                    managed_start_ticks = None
                state.update(
                    {
                        "status": "monitoring_restarted_runner",
                        "managed_runner_pid": managed_pid,
                        "managed_runner_start_ticks": managed_start_ticks,
                        "runner_process_alive": managed_start_ticks is not None,
                    }
                )
                _write_state(state_path, state)
                _append_log(
                    log_path,
                    events,
                    {
                        "event": "runner_restarted_once",
                        "managed_runner_pid": managed_pid,
                        "managed_runner_start_ticks": managed_start_ticks,
                    },
                )
            time.sleep(args.poll_seconds)
    finally:
        os.close(lock_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--stable-seed-roster", type=Path, required=True)
    parser.add_argument("--stable-seed-roster-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state_root = _absolute(args.state_root, role="guardian state root")
    output = _absolute(args.output, role="experiment output")
    output_resolved = output.resolve(strict=True)
    state_resolved = state_root.resolve(strict=False)
    if (
        state_resolved == output_resolved
        or output_resolved in state_resolved.parents
        or state_resolved in output_resolved.parents
        or state_root.is_symlink()
    ):
        print(
            "ACTOR_PROTOCOL_GUARDIAN_FAILED_GUARDIAN_CONTRACT="
            "guardian state and experiment artifact trees must be disjoint",
            file=sys.stderr,
        )
        return 3
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "guardian_state.json"
    log_path = state_root / "guardian_log.json"
    exit_path = state_root / "guardian_exit.json"
    restart_count = 0
    try:
        return execute(args)
    except GuardianLockHeld as error:
        # A losing second instance must not mutate the first guardian's
        # atomic state/log/exit receipts.
        print(f"ACTOR_PROTOCOL_GUARDIAN_LOCK_HELD={error}", file=sys.stderr)
        return 4
    except BaseException as error:
        prior = _read_optional_state(state_path)
        if prior is not None and isinstance(prior.get("restart_count"), int):
            restart_count = int(prior["restart_count"])
        experiment_failure = isinstance(error, ExperimentFailure)
        status = "failed_experiment" if experiment_failure else "failed_guardian_contract"
        exit_code = 2 if experiment_failure else 3
        events = _load_log(log_path)
        _append_log(
            log_path,
            events,
            {
                "event": status,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "restart_count": restart_count,
            },
        )
        _write_state(
            state_path,
            {
                "status": status,
                "guardian_pid": os.getpid(),
                "restart_count": restart_count,
                "restart_limit": RESTART_LIMIT,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "last_poll_unix": time.time(),
            },
        )
        _write_exit(
            exit_path,
            status=status,
            exit_code=exit_code,
            message=f"{type(error).__name__}: {error}",
            restart_count=restart_count,
        )
        print(f"ACTOR_PROTOCOL_GUARDIAN_{status.upper()}={error}", file=sys.stderr)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLETION_FORMAT",
    "COMPLETION_STATUS",
    "ExperimentFailure",
    "GuardianContractError",
    "PAIR_COUNT",
    "RESTART_LIMIT",
    "ROLLOUT_COUNT",
    "atomic_json",
    "build_runner_command",
    "canonical_sha256",
    "next_action",
    "observe_artifacts",
    "parse_args",
    "query_gpu_contract",
    "sha256_file",
    "sha256_tree",
    "validate_completion",
    "validate_process_identity",
]
