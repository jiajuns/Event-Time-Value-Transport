#!/usr/bin/env python3
"""Fail-closed launcher for schema-v5 development expansion collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from launch_openvla_etsf_counterfactual_v5 import atomic_json, wait_for_gpu_idle
from robotwin_development_seed_contract import (
    EXPECTED_COUNT,
    REGISTRY,
    sha256,
    validate_development_manifest,
)


FORMAT = "etsf_openvla_schema_v5_development_expansion_launch_v1"
DEFAULT_CODE_ROOT = Path("/home/user/etsf_event_world_model_code_20260827")
DEFAULT_PYTHON = Path("/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python")
DEFAULT_MANIFEST = DEFAULT_CODE_ROOT / (
    "artifacts/protocol/development_expansion_seeds_20260827.json"
)
DEFAULT_OUTPUT = Path(
    "/home/user/etsf_openvla_event_branches_v5_development150_20260827"
)
DEFAULT_MODEL = Path(
    "/home/user/etsf_openvla_models/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot"
)
DEFAULT_RLINF_ROOT = Path("/home/user/etsf_stage0/RLinf")
DEFAULT_ROBOTWIN_ROOT = Path("/home/user/etsf_stage0/RoboTwin")
DEFAULT_ROBOTWIN_CODE = Path("/home/user/etsf_stage0/RoboTwin_RLinf_support")
DEFAULT_EVENT_SPEC = Path("/home/user/etsf_stage2_run_20260825/event_spec.json")


def command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def wait_for_manifest(path: Path, timeout: float, poll: float) -> None:
    started = time.monotonic()
    while True:
        if path.is_file():
            return
        if time.monotonic() - started >= timeout:
            raise RuntimeError(f"timed out waiting for development manifest: {path}")
        time.sleep(max(min(poll, 60.0), 0.01))


def build_command(args: argparse.Namespace, manifest: Mapping[str, Any]) -> list[str]:
    python_bin = args.python_bin.expanduser().absolute()
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise FileNotFoundError(f"python interpreter is not executable: {python_bin}")
    return [
        str(python_bin),
        str(args.collector.absolute()),
        "--model-path", str(args.model_path.absolute()),
        "--rlinf-root", str(args.rlinf_root.absolute()),
        "--robotwin-root", str(args.robotwin_root.absolute()),
        "--robotwin-code", str(args.robotwin_code.absolute()),
        "--event-spec", str(args.event_spec.absolute()),
        "--output", str(args.output.absolute()),
        "--task", args.task,
        "--seeds-file", str(Path(str(manifest["path"]))),
        "--seeds-key", "train",
        "--allow-unregistered-seeds",
        "--development-seed-manifest", str(Path(str(manifest["path"]))),
        "--blends", "0.25", "0.5", "0.75", "1.0",
        "--temperature", "0.7",
        "--top-k", "4",
    ]


def audit_complete_collection(
    root: Path, development: Mapping[str, Any], event_spec_sha256: str
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("development collector manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("groups")
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("schema_version", -1)) != 5
        or int(manifest.get("completed", -1)) != EXPECTED_COUNT
        or manifest.get("seed_registry") != REGISTRY
        or manifest.get("development_seed_manifest_sha256") != development["sha256"]
        or not str(manifest.get("development_seed_manifest", ""))
        or manifest.get("fresh_seed_manifest") not in (None, "")
        or manifest.get("fresh_seed_manifest_sha256") not in (None, "")
        or manifest.get("requested_seeds") != development["requested_seeds"]
        or manifest.get("resolved_seeds") != development["resolved_seeds"]
        or manifest.get("event_spec_sha256") != event_spec_sha256
        or int(manifest.get("candidate_count", -1)) != 5
        or not isinstance(groups, list)
        or len(groups) != EXPECTED_COUNT
    ):
        raise RuntimeError("development collection provenance/completion contract mismatch")
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "groups": len(groups),
        "seed_registry": REGISTRY,
        "fresh_registry_used": False,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.development_seed_manifest.expanduser().absolute()
    wait_for_manifest(manifest_path, args.wait_timeout_seconds, args.poll_seconds)
    development = validate_development_manifest(manifest_path, task=args.task)
    paths = {
        "collector": args.collector.absolute(),
        "model_path": args.model_path.absolute(),
        "rlinf_root": args.rlinf_root.absolute(),
        "robotwin_root": args.robotwin_root.absolute(),
        "robotwin_code": args.robotwin_code.absolute(),
        "event_spec": args.event_spec.absolute(),
        "python_bin": args.python_bin.expanduser().absolute(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    event_digest = sha256(paths["event_spec"])
    command = build_command(args, development)
    contract = {
        "development_manifest_sha256": development["sha256"],
        "development_manifest_payload_sha256": development[
            "manifest_payload_sha256"
        ],
        "official_seed_registry_sha256": development["official_seed_registry"][
            "sha256"
        ],
        "fresh_seed_manifest_exclusion_sha256": development[
            "fresh_seed_manifest"
        ]["sha256"],
        "event_spec_sha256": event_digest,
        "collector_argv_sha256": command_sha256(command),
        "seed_registry": REGISTRY,
        "fresh_confirmation_eligible": False,
    }
    return {
        "format": FORMAT,
        "status": "preflight_complete",
        "development": development,
        "event_spec": {"path": str(paths["event_spec"]), "sha256": event_digest},
        "output": str(args.output.absolute()),
        "command": command,
        "contract": contract,
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "labels_read_during_seed_selection": False,
    }


def acquire_lock(path: Path, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise RuntimeError("development collection lock exists; use --resume after audit")
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock.get("host") == socket.gethostname() and Path(
            f"/proc/{int(lock.get('pid', -1))}"
        ).exists():
            raise RuntimeError("development collection launcher is still running")
        path.unlink()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-seed-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rlinf-root", type=Path, default=DEFAULT_RLINF_ROOT)
    parser.add_argument("--robotwin-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--robotwin-code", type=Path, default=DEFAULT_ROBOTWIN_CODE)
    parser.add_argument("--event-spec", type=Path, default=DEFAULT_EVENT_SPEC)
    parser.add_argument("--collector", type=Path, default=local / "collect_openvla_etsf_event_branches.py")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", default="move_can_pot")
    parser.add_argument("--wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.wait_timeout_seconds < 0
        or args.gpu_wait_timeout_seconds < 0
        or not 0 < args.poll_seconds <= 60
        or not 0 < args.gpu_poll_seconds <= 60
    ):
        raise ValueError("timeouts must be non-negative and polls must lie in (0,60]")
    audit = preflight(args)
    output = args.output.expanduser().absolute()
    audit_path = output / "development_launch_audit.json"
    if output.exists() and (output / "manifest.json").is_file():
        try:
            complete = audit_complete_collection(
                output, audit["development"], audit["event_spec"]["sha256"]
            )
        except RuntimeError:
            if not args.resume and not args.dry_run:
                raise RuntimeError("partial development output requires --resume")
        else:
            print("DEVELOPMENT_EXPANSION_SKIP=" + json.dumps(complete, sort_keys=True))
            return
    if args.dry_run:
        print("DEVELOPMENT_EXPANSION_DRY_RUN=" + json.dumps(audit, sort_keys=True))
        return
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError("partial development output requires --resume")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("formal development collection requires RTX 4090")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "development_launch.lock"
    acquire_lock(lock, args.resume)
    try:
        audit["status"] = "collector_running_resumable"
        audit["gpu_idle"] = wait_for_gpu_idle(
            args.gpu_wait_timeout_seconds, args.gpu_poll_seconds, gpu_index=0
        )
        atomic_json(audit_path, audit)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "8",
            }
        )
        subprocess.run(audit["command"], check=True, env=environment)
        audit["collection"] = audit_complete_collection(
            output, audit["development"], audit["event_spec"]["sha256"]
        )
        audit["status"] = "complete"
        atomic_json(audit_path, audit)
        print("DEVELOPMENT_EXPANSION_COMPLETE=" + json.dumps(audit["collection"], sort_keys=True))
    except BaseException:
        audit["status"] = "collector_interrupted_resumable"
        atomic_json(audit_path, audit)
        raise
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
