#!/usr/bin/env python3
"""Wait for OOF authorization and conditionally start the one-shot fresh50 run.

This process is intentionally label-blind.  It observes only the upstream
pipeline state.  A failed/unauthorized OOF terminates with fresh forbidden; the
fresh launcher and its manifest are not opened until the upstream state is
``complete_fresh50_ready_one_shot``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


FORMAT = "etsf_oof_guarded_fresh_watcher_v1"
READY = "complete_fresh50_ready_one_shot"
FORBIDDEN = "complete_guard_not_authorized_fresh_forbidden"


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
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def wait_for_upstream(path: Path, timeout: float, poll: float) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if path.is_file():
            try:
                value = _json(path)
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, Mapping):
                status = str(value.get("status", ""))
                if status in (READY, FORBIDDEN):
                    return value
                if status == "failed_closed":
                    raise RuntimeError(
                        f"upstream development/OOF pipeline failed: {value.get('error')}"
                    )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for OOF authorization decision")
        time.sleep(poll)


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python_bin.expanduser().absolute()),
        str(args.fresh_launcher.expanduser().absolute()),
        "--fresh-seed-manifest",
        str(args.fresh_seed_manifest.expanduser().absolute()),
        "--counterfactual-root",
        str(args.counterfactual_root.expanduser().absolute()),
        "--event-spec",
        str(args.event_spec.expanduser().absolute()),
        "--data",
        str(args.data.expanduser().absolute()),
        "--factual-root",
        str(args.factual_root.expanduser().absolute()),
        "--model-path",
        str(args.model_path.expanduser().absolute()),
        "--rlinf-root",
        str(args.rlinf_root.expanduser().absolute()),
        "--robotwin-root",
        str(args.robotwin_root.expanduser().absolute()),
        "--robotwin-code",
        str(args.robotwin_code.expanduser().absolute()),
        "--output",
        str(args.output.expanduser().absolute()),
        "--python-bin",
        str(args.python_bin.expanduser().absolute()),
        "--wait-timeout-seconds",
        str(args.child_wait_timeout_seconds),
        "--poll-seconds",
        str(args.poll_seconds),
        "--gpu-wait-timeout-seconds",
        str(args.child_wait_timeout_seconds),
        "--gpu-poll-seconds",
        str(args.poll_seconds),
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    state_root = args.state_root.expanduser().absolute()
    output = args.output.expanduser().absolute()
    if state_root.exists() or output.exists():
        raise FileExistsError("fresh watcher state/output must both be brand new")
    if not 0 < args.poll_seconds <= 60 or min(
        args.wait_timeout_seconds, args.child_wait_timeout_seconds
    ) <= 0:
        raise ValueError("invalid timeout/poll interval")
    # Only label-free executables/infrastructure are inspected before the OOF
    # decision.  The fresh seed manifest is deliberately left to the child.
    for path in (
        args.python_bin,
        args.fresh_launcher,
        args.event_spec,
        args.factual_root,
        args.model_path,
        args.rlinf_root,
        args.robotwin_root,
        args.robotwin_code,
    ):
        if not path.expanduser().absolute().exists():
            raise FileNotFoundError(path)
    state_root.mkdir(parents=True, exist_ok=False)
    state_path = state_root / "watcher_state.json"
    state: dict[str, Any] = {
        "format": FORMAT,
        "status": "waiting_for_oof_authorization",
        "pid": os.getpid(),
        "upstream_state": str(args.upstream_state.expanduser().absolute()),
        "fresh_manifest_read_by_watcher": False,
        "fresh_labels_read_by_watcher": False,
    }
    atomic_json(state_path, state)
    try:
        upstream = wait_for_upstream(
            args.upstream_state.expanduser().absolute(),
            args.wait_timeout_seconds,
            args.poll_seconds,
        )
        state["upstream_status"] = upstream["status"]
        if upstream["status"] == FORBIDDEN:
            state["status"] = "complete_upstream_guard_not_authorized_fresh_forbidden"
            state["fresh_confirmation_policy"] = "forbidden"
            atomic_json(state_path, state)
            return state
        expected_oof = Path(str(upstream.get("oof_output", ""))).resolve() / "final"
        if expected_oof != args.counterfactual_root.expanduser().absolute().resolve():
            raise RuntimeError("fresh watcher counterfactual root differs from upstream OOF")
        command = build_command(args)
        state["status"] = "running_one_shot_fresh_pipeline"
        state["fresh_command"] = command
        # The watcher still does not open the fresh manifest; access begins in
        # the already audited child launcher after authorization.
        atomic_json(state_path, state)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "8",
            }
        )
        log = state_root / "fresh_pipeline.log"
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"fresh pipeline exited {completed.returncode}; see {log}"
            )
        audit_path = output / "pipeline_audit.json"
        marker_path = output / "fresh50_evaluation" / "evaluated_once.json"
        audit = _json(audit_path)
        marker = _json(marker_path)
        if audit.get("status") != "complete" or marker.get("status") != "complete":
            raise RuntimeError("fresh pipeline returned without complete confirmatory artifacts")
        state.update(
            {
                "status": "complete_fresh50_confirmed",
                "fresh_confirmation_policy": "consumed_once",
                "pipeline_audit": str(audit_path.resolve()),
                "confirmatory_marker": str(marker_path.resolve()),
                "fresh_manifest_read_by_watcher": False,
                "fresh_labels_read_by_watcher": False,
            }
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
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--fresh-seed-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--factual-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--fresh-launcher",
        type=Path,
        default=local / "launch_openvla_etsf_fresh50_confirmation.py",
    )
    parser.add_argument("--wait-timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--child-wait-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = execute(args)
    print(
        "GUARDED_FRESH_WATCHER_COMPLETE="
        + json.dumps(
            {
                "status": result["status"],
                "fresh_confirmation_policy": result["fresh_confirmation_policy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
