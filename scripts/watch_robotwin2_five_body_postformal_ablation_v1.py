#!/usr/bin/env python3
"""Run the fixed full-data shared-head ablation after formal paired evaluation.

The watcher is remote-only scheduling infrastructure.  It does not expose a
reduced-data or reduced-budget mode and it never starts before the 1,000-pair
formal evaluation has durably completed.
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


FORMAT = "etsf_robotwin2_five_body_postformal_ablation_watcher_v1"
SUMMARY_FORMAT = "etsf_robotwin2_five_body_lobo_offline_ablation_v1"
SUMMARY_STATUS = "complete_frozen_checkpoint_posthoc_heldout_ablation"
UPSTREAM_FORMAT = "etsf_robotwin2_five_body_lobo_to_paired_success_watcher_v1"
BINDING_FORMAT = "etsf_robotwin2_five_body_lobo_training_binding_v1"
EXPECTED_PAIRS = 1000
EXPECTED_ROLLOUTS = 2000
EXPECTED_DECISIONS = 2000
EXPECTED_BRANCHES = 8000
EXPECTED_VARIANTS = [
    "success_only",
    "no_time_duration",
    "no_object_effect",
    "full",
]
EXPECTED_FOLDS = 5
EXPECTED_STEPS_PER_MEMBER = 3000
EXPECTED_ENSEMBLE_SEEDS = [20260901, 20260902, 20260903, 20260904, 20260905]
GPU_UUID = "GPU-06f6e50e-5296-258f-dd86-8f838390a7d1"

HOME_ROOT = Path("/home/user")
UPSTREAM_STATE = HOME_ROOT / (
    "etsf_robotwin2_fivebody_paired_success_full2000_20260830_v2_analytic."
    "watcher_state.json"
)
BINDING = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v2_analytic."
    "binding.json"
)
OUTPUT_ROOT = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1"
)
WATCHER_STATE = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1.watcher_state.json"
)
WATCHER_PID = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1.watcher.pid"
)
WATCHER_LOCK = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1.watcher.lock"
)
WATCHER_LOG = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1.watcher.log"
)
RUNNER_LOG = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1.run.log"
)
RUN_EXIT = HOME_ROOT / (
    "etsf_robotwin2_fivebody_lobo_ablation_full8000_20260830_v1.run.exit"
)
TRAINING_PYTHON = HOME_ROOT / "anaconda3/envs/ETSF_RoboTwin/bin/python"


class PostformalAblationError(RuntimeError):
    """The fixed upstream, inventory, runtime, or result contract changed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_upstream(value: Mapping[str, Any]) -> None:
    if value.get("format") != UPSTREAM_FORMAT:
        raise PostformalAblationError("formal paired watcher format changed")
    if value.get("status") != "complete":
        raise PostformalAblationError("formal paired evaluation is not complete")
    if (
        value.get("completed_pairs") != EXPECTED_PAIRS
        or value.get("completed_rollouts") != EXPECTED_ROLLOUTS
    ):
        raise PostformalAblationError("formal paired result is not the full 1000/2000 run")
    report = Path(str(value.get("paired_success_report", "")))
    if (
        not report.is_file()
        or report.is_symlink()
        or sha256_file(report) != value.get("paired_success_report_file_sha256")
    ):
        raise PostformalAblationError("formal paired report is missing or changed")


def validate_binding(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise PostformalAblationError("full-8000 binding is missing or symbolic")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("format") != BINDING_FORMAT
        or value.get("heldout_labels_may_train_fit_calibrate_or_select") is not False
        or value.get("canonical_shared_body_rows") != 1
    ):
        raise PostformalAblationError("full-8000 binding contract changed")
    body_manifests = value.get("body_manifests")
    if not isinstance(body_manifests, Mapping) or len(body_manifests) != EXPECTED_FOLDS:
        raise PostformalAblationError("binding does not contain all five bodies")
    return sha256_file(path)


def validate_complete_summary(path: Path, binding_sha256: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PostformalAblationError("offline ablation summary is missing or symbolic")
    value = json.loads(path.read_text(encoding="utf-8"))
    budget = value.get("fixed_budget", {})
    inventory = value.get("inventory", {})
    if (
        value.get("format") != SUMMARY_FORMAT
        or value.get("status") != SUMMARY_STATUS
        or value.get("binding_file_sha256") != binding_sha256
        or inventory.get("decisions") != EXPECTED_DECISIONS
        or inventory.get("branches") != EXPECTED_BRANCHES
        or budget.get("variants") != EXPECTED_VARIANTS
        or budget.get("folds_per_variant") != EXPECTED_FOLDS
        or budget.get("steps_per_member") != EXPECTED_STEPS_PER_MEMBER
        or budget.get("ensemble_seeds") != EXPECTED_ENSEMBLE_SEEDS
        or budget.get("heldout_labels_used_for_checkpoint_selection") is not False
        or budget.get("all_checkpoints_selected_before_any_heldout_payload_open")
        is not True
        or budget.get("variant_selection_performed") is not False
    ):
        raise PostformalAblationError("offline ablation summary contract changed")
    return value


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise PostformalAblationError("nvidia-smi failed while waiting for the RTX 4090")
    pids = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == GPU_UUID:
            pids.append(int(fields[1]))
    return pids


def runner_command(code_root: Path, binding_sha256: str) -> list[str]:
    return [
        str(TRAINING_PYTHON),
        str(code_root / "run_robotwin2_five_body_lobo_offline_ablation_v1.py"),
        "--binding",
        str(BINDING),
        "--binding-sha256",
        binding_sha256,
        "--output",
        str(OUTPUT_ROOT),
        "--python-executable",
        str(TRAINING_PYTHON),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    if arguments.poll_seconds <= 0:
        raise PostformalAblationError("poll interval must be positive")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if Path.home().resolve() != HOME_ROOT:
        raise PostformalAblationError("this watcher may run only under /home/user")
    lock_stream = WATCHER_LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise PostformalAblationError("another postformal ablation watcher is active") from error
    atomic_text(WATCHER_PID, f"{os.getpid()}\n")
    code_root = Path(__file__).resolve().parent
    runner = code_root / "run_robotwin2_five_body_lobo_offline_ablation_v1.py"
    trainer = code_root / "train_robotwin2_five_body_lobo_shared_event_head_v1.py"
    if any(not path.is_file() or path.is_symlink() for path in (runner, trainer)):
        raise PostformalAblationError("fixed ablation runtime is incomplete")
    if not TRAINING_PYTHON.is_file() or not os.access(TRAINING_PYTHON, os.X_OK):
        raise PostformalAblationError("fixed ablation Python is missing or not executable")

    def write_state(status: str, **extra: Any) -> None:
        atomic_json(
            WATCHER_STATE,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "upstream_state": str(UPSTREAM_STATE),
                "binding": str(BINDING),
                "output_root": str(OUTPUT_ROOT),
                "runner_log": str(RUNNER_LOG),
                "gpu_uuid": GPU_UUID,
                **extra,
            },
        )

    upstream: Mapping[str, Any] | None = None
    while upstream is None:
        try:
            candidate = json.loads(UPSTREAM_STATE.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            candidate = {}
        status = candidate.get("status")
        if status == "failed":
            raise PostformalAblationError("formal paired watcher failed")
        if status == "complete":
            validate_upstream(candidate)
            upstream = candidate
            break
        write_state(
            "waiting_for_formal_paired_completion",
            upstream_status=status,
            gpu_reserved_by_watcher=False,
        )
        time.sleep(arguments.poll_seconds)

    binding_sha256 = validate_binding(BINDING)
    summary = OUTPUT_ROOT / "offline_ablation_summary.json"
    if summary.exists():
        result = validate_complete_summary(summary, binding_sha256)
        atomic_text(RUN_EXIT, "0\n")
        write_state(
            "complete",
            reused_complete_result=True,
            summary=str(summary),
            summary_file_sha256=sha256_file(summary),
            binding_file_sha256=binding_sha256,
            gpu_reserved_by_watcher=False,
        )
        lock_stream.flush()
        return 0
    if OUTPUT_ROOT.exists():
        raise PostformalAblationError("partial ablation output exists; refusing to overwrite it")

    while True:
        active = gpu_compute_pids()
        if not active:
            break
        write_state(
            "waiting_for_idle_rtx4090_after_formal_evaluation",
            active_compute_pids=active,
            binding_file_sha256=binding_sha256,
            gpu_reserved_by_watcher=False,
        )
        time.sleep(arguments.poll_seconds)

    command = runner_command(code_root, binding_sha256)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(code_root),
        }
    )
    write_state(
        "running_full8000_fivefold_fourvariant_ablation",
        command=command,
        binding_file_sha256=binding_sha256,
        gpu_reserved_by_watcher=True,
        reduced_data_or_budget_mode=False,
    )
    with RUNNER_LOG.open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=code_root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise PostformalAblationError(f"full ablation exited {result.returncode}")
    document = validate_complete_summary(summary, binding_sha256)
    atomic_text(RUN_EXIT, "0\n")
    write_state(
        "complete",
        reused_complete_result=False,
        summary=str(summary),
        summary_file_sha256=sha256_file(summary),
        binding_file_sha256=binding_sha256,
        inventory=document["inventory"],
        gpu_reserved_by_watcher=False,
    )
    lock_stream.flush()
    return 0


def record_failure(error: BaseException) -> None:
    try:
        atomic_text(RUN_EXIT, "1\n")
        atomic_json(
            WATCHER_STATE,
            {
                "format": FORMAT,
                "status": "failed",
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
                "upstream_state": str(UPSTREAM_STATE),
                "binding": str(BINDING),
                "output_root": str(OUTPUT_ROOT),
                "gpu_uuid": GPU_UUID,
            },
        )
    except Exception:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        record_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


__all__ = [
    "PostformalAblationError",
    "gpu_compute_pids",
    "runner_command",
    "sha256_file",
    "validate_binding",
    "validate_complete_summary",
    "validate_upstream",
]
