#!/usr/bin/env python3
"""Fail-closed server pipeline: frozen audit -> hard-link merge -> 4090 OOF.

The collector and development audit are deliberately separate processes.  This
orchestrator may be started while they are still running: it waits only for the
atomically published, signed ``training_ready`` audit, rechecks its source
identities, performs the immutable 250-group merge, waits for an idle RTX 4090,
and delegates formal training to the serial OOF launcher.  It never accepts a
fresh-confirmation data root and never evaluates fresh labels.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_openvla_etsf_development250 import (
    FORMAT as AUDIT_FORMAT,
    canonical_sha256,
    sha256,
)
from launch_openvla_etsf_counterfactual_oof_v5 import (
    query_compute_pids,
    query_gpu_name,
)


FORMAT = "etsf_development250_to_oof_server_pipeline_v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def wait_for_file(path: Path, timeout: float, poll: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for atomic artifact: {path}")
        time.sleep(poll)


def validate_training_ready_audit(args: argparse.Namespace) -> dict[str, Any]:
    path = args.audit.expanduser().resolve()
    value = _json(path)
    unsigned = dict(value)
    recorded = str(unsigned.pop("audit_payload_sha256", ""))
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("development250 audit signature changed")
    old = value.get("old100")
    development = value.get("development150")
    combined = value.get("combined")
    event = value.get("event_spec")
    fresh = value.get("fresh50_exclusion")
    if not all(isinstance(row, Mapping) for row in (old, development, combined, event, fresh)):
        raise RuntimeError("development250 audit is structurally incomplete")
    if (
        value.get("format") != AUDIT_FORMAT
        or value.get("status") != "training_ready"
        or value.get("training_authorized") is not True
        or int(old.get("groups", -1)) != 100
        or int(old.get("candidate_count", -1)) != 4
        or int(development.get("groups", -1)) != 150
        or int(development.get("candidate_count", -1)) != 5
        or int(combined.get("groups", -1)) != 250
        or int(combined.get("candidate_branches", -1)) != 1150
        or fresh.get("labels_read") is not False
    ):
        raise RuntimeError("development250 audit does not authorize formal training")
    expected_paths = {
        "old100": args.old_development100.expanduser().resolve(),
        "development150": args.new_development150.expanduser().resolve(),
    }
    if Path(str(old.get("root", ""))).resolve() != expected_paths["old100"] or Path(
        str(development.get("root", ""))
    ).resolve() != expected_paths["development150"]:
        raise RuntimeError("development250 audit points to different source roots")
    event_spec = args.event_spec.expanduser().resolve()
    if Path(str(event.get("path", ""))).resolve() != event_spec or str(
        event.get("sha256", "")
    ) != sha256(event_spec):
        raise RuntimeError("development250 audit event spec changed")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "audit_payload_sha256": recorded,
        "groups": 250,
        "candidate_branches": 1150,
        "fresh_confirmation_labels_read": False,
    }


def run_logged(
    command: Sequence[str], *, log: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pipeline subprocess failed with exit {completed.returncode}; see {log}"
        )
    return {
        "returncode": completed.returncode,
        "log": str(log.resolve()),
        "log_sha256": sha256(log),
        "argv": list(command),
    }


def wait_for_idle_4090(gpu_index: int, timeout: float, poll: float) -> dict[str, Any]:
    name = query_gpu_name(gpu_index)
    if "4090" not in name:
        raise RuntimeError(f"formal OOF requires RTX 4090, found {name!r}")
    deadline = time.monotonic() + timeout
    last_pids: list[int] = []
    while True:
        last_pids = query_compute_pids(gpu_index)
        if not last_pids:
            return {"gpu_index": gpu_index, "gpu_name": name, "compute_pids": []}
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for idle RTX 4090; active PIDs={last_pids}"
            )
        time.sleep(poll)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.state_root.expanduser().absolute()
    if root.exists():
        raise FileExistsError(f"pipeline state root already exists: {root}")
    if args.merged_output.exists() or args.oof_output.exists():
        raise FileExistsError("merged/OOF output must both be brand new")
    for path in (
        args.old_development100,
        args.new_development150,
        args.event_spec,
        args.pretrained,
        args.merge_script,
        args.oof_launcher,
        args.oof_trainer,
        args.python_bin,
    ):
        if not path.expanduser().absolute().exists():
            raise FileNotFoundError(path)
    if not 0 < args.poll_seconds <= 60 or args.wait_timeout_seconds <= 0:
        raise ValueError("invalid wait timeout/poll interval")

    root.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "format": FORMAT,
        "status": "waiting_for_training_ready_audit",
        "pid": os.getpid(),
        "fresh_confirmation_inputs_accepted": False,
        "fresh_confirmation_labels_read": False,
        "merged_output": str(args.merged_output.expanduser().absolute()),
        "oof_output": str(args.oof_output.expanduser().absolute()),
    }
    state_path = root / "pipeline_state.json"
    atomic_json(state_path, state)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
    try:
        wait_for_file(
            args.audit.expanduser().resolve(),
            args.wait_timeout_seconds,
            args.poll_seconds,
        )
        state["audit"] = validate_training_ready_audit(args)
        state["status"] = "merging_frozen_development250"
        atomic_json(state_path, state)
        merge_command = [
            str(args.python_bin.expanduser().absolute()),
            str(args.merge_script.expanduser().absolute()),
            "--old-development100",
            str(args.old_development100.expanduser().absolute()),
            "--new-development150",
            str(args.new_development150.expanduser().absolute()),
            "--output",
            str(args.merged_output.expanduser().absolute()),
        ]
        state["merge"] = run_logged(
            merge_command, log=root / "merge.log", environment=environment
        )
        merged_manifest = args.merged_output.expanduser().absolute() / "manifest.json"
        merged = _json(merged_manifest)
        if (
            merged.get("status") != "complete"
            or int(merged.get("completed", -1)) != 250
            or merged.get("fresh_confirmation_labels_read") is not False
            or merged.get("seed_registry")
            != "merged_official100_plus_explicit_development150"
        ):
            raise RuntimeError("merged development250 manifest failed postcondition")
        state["merge"]["manifest"] = str(merged_manifest.resolve())
        state["merge"]["manifest_sha256"] = sha256(merged_manifest)
        state["status"] = "waiting_for_idle_rtx4090"
        atomic_json(state_path, state)
        state["gpu_idle_audit"] = wait_for_idle_4090(
            args.gpu_index, args.wait_timeout_seconds, args.poll_seconds
        )
        state["status"] = "running_formal_oof"
        atomic_json(state_path, state)
        oof_command = [
            str(args.python_bin.expanduser().absolute()),
            str(args.oof_launcher.expanduser().absolute()),
            "--data",
            str(args.merged_output.expanduser().absolute()),
            "--pretrained",
            str(args.pretrained.expanduser().absolute()),
            "--event-spec",
            str(args.event_spec.expanduser().absolute()),
            "--output",
            str(args.oof_output.expanduser().absolute()),
            "--trainer",
            str(args.oof_trainer.expanduser().absolute()),
            "--python-bin",
            str(args.python_bin.expanduser().absolute()),
            "--gpu-index",
            str(args.gpu_index),
            "--num-workers",
            str(args.num_workers),
        ]
        state["oof"] = run_logged(
            oof_command, log=root / "oof.log", environment=environment
        )
        launch_state = _json(args.oof_output.expanduser().absolute() / "launch_state.json")
        oof_status = str(launch_state.get("status", ""))
        if oof_status not in (
            "stopped_guard_not_authorized",
            "complete_fresh50_ready_one_shot",
        ):
            raise RuntimeError(f"formal OOF ended in unexpected state: {oof_status}")
        state["oof"]["launch_state"] = oof_status
        state["oof"]["launch_state_sha256"] = sha256(
            args.oof_output.expanduser().absolute() / "launch_state.json"
        )
        state["status"] = (
            "complete_fresh50_ready_one_shot"
            if oof_status == "complete_fresh50_ready_one_shot"
            else "complete_guard_not_authorized_fresh_forbidden"
        )
        state["fresh_confirmation_policy"] = (
            "one_shot_only"
            if oof_status == "complete_fresh50_ready_one_shot"
            else "forbidden"
        )
        atomic_json(state_path, state)
        return state
    except BaseException as error:
        state["status"] = "failed_closed"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        atomic_json(state_path, state)
        raise


def parse_args() -> argparse.Namespace:
    scripts = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--old-development100", type=Path, required=True)
    parser.add_argument("--new-development150", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--merge-script",
        type=Path,
        default=scripts / "merge_openvla_etsf_schema5_development.py",
    )
    parser.add_argument(
        "--oof-launcher",
        type=Path,
        default=scripts / "launch_openvla_etsf_counterfactual_oof_v5.py",
    )
    parser.add_argument(
        "--oof-trainer",
        type=Path,
        default=scripts / "train_openvla_etsf_counterfactual_oof.py",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--wait-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = execute(args)
    print(
        "DEVELOPMENT250_OOF_PIPELINE_COMPLETE="
        + json.dumps(
            {
                "status": result["status"],
                "oof_output": result["oof_output"],
                "fresh_confirmation_policy": result["fresh_confirmation_policy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
