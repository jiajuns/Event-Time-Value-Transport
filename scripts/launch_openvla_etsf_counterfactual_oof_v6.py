#!/usr/bin/env python3
"""Fail-closed RTX-4090 launcher for development-only v6 nested OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate_openvla_etsf_oof_prediction_diagnostics import (
    validate_oof_prediction_diagnostics,
)
from openvla_etsf_counterfactual_oof_v6 import (
    FORMAT, OUTER_FOLDS, SELECTION_FORMAT, canonical_sha256,
    validate_nested_oof_manifest,
)
from train_openvla_etsf_counterfactual import sha256


LAUNCH_FORMAT = "etsf_counterfactual_nested_oof_v6_serial_launch_v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _command_sha(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), separators=(",", ":")).encode()).hexdigest()


def build_stage_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = args.output.expanduser().absolute()
    trainer = args.trainer.absolute()
    python_bin = args.python_bin.expanduser().absolute()
    manifest = output / "nested_oof_v6.json"
    common = ["--data", str(args.data.absolute()), "--pretrained", str(args.pretrained.absolute()),
              "--event-spec", str(args.event_spec.absolute())]
    commands = [("preregister", [str(python_bin), str(trainer), "preregister", *common,
                                  "--output", str(manifest)], False)]
    for fold_id in range(OUTER_FOLDS):
        commands.append((f"fold_{fold_id}", [str(python_bin), str(trainer), "fold", *common,
                         "--oof-manifest", str(manifest), "--fold-id", str(fold_id),
                         "--output", str(output / "folds" / f"fold_{fold_id}"),
                         "--num-workers", str(args.num_workers)], True))
    commands.append(("select", [str(python_bin), str(trainer), "select",
                     "--oof-manifest", str(manifest), "--fold-root", str(output / "folds"),
                     "--output", str(output / "oof_selection_v6.json")], False))
    return [{"stage": name, "argv": argv, "argv_sha256": _command_sha(argv), "uses_gpu": gpu}
            for name, argv, gpu in commands]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"v6 requires a new non-resumable output root: {output}")
    paths = {"manifest": args.data.absolute() / "manifest.json", "pretrained": args.pretrained.absolute(),
             "event_spec": args.event_spec.absolute(), "trainer": args.trainer.absolute(),
             "python_bin": args.python_bin.expanduser().absolute()}
    for name, path in paths.items():
        if not path.is_file(): raise FileNotFoundError(f"{name}: {path}")
    root = _json(paths["manifest"])
    if "fresh" in args.data.name.lower() or root.get("seed_registry") == "explicit_fresh_confirmation":
        raise RuntimeError("fresh confirmation input is forbidden in v6")
    if root.get("fresh_seed_manifest_sha256") not in (None, ""):
        raise RuntimeError("fresh confirmation input is forbidden in v6")
    groups = root.get("groups")
    if root.get("status") != "complete" or int(root.get("schema_version", -1)) != 5:
        raise RuntimeError("v6 requires complete schema-v5 development data")
    if not isinstance(groups, list) or len(groups) != 250 or int(root.get("completed", -1)) != 250:
        raise RuntimeError("v6 requires exactly 250 development groups")
    protocol = args.trainer.absolute().with_name("openvla_etsf_counterfactual_oof_v6.py")
    if not protocol.is_file(): raise FileNotFoundError(protocol)
    plan = {
        "format": LAUNCH_FORMAT, "status": "preflight_complete", "output_root": str(output),
        "serial_execution": True, "nonresumable": True,
        "fresh_confirmation_inputs_accepted": False,
        "completion_status": "complete_development_only_fresh_forbidden",
        "source": {name: {"path": str(path), "sha256": sha256(path)}
                   for name, path in paths.items()},
        "protocol": {"path": str(protocol), "sha256": sha256(protocol)},
        "development_groups": 250,
        "gpu": {"index": args.gpu_index, "required_name_substring": "4090",
                "global_lease": str(args.gpu_lock.absolute() if args.gpu_lock else
                                    Path(f"/tmp/etsf_openvla_oof_v6_gpu{args.gpu_index}.lock")),
                "concurrent_compute_policy": "reject_before_every_fold"},
        "commands": build_stage_commands(args),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def query_gpu_name(index: int) -> str:
    try:
        done = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=name",
                               "--format=csv,noheader"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to audit GPU model") from error
    names = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if len(names) != 1: raise RuntimeError(f"unexpected GPU audit: {names}")
    return names[0]


def query_compute_pids(index: int) -> list[int]:
    try:
        done = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid",
                               "--format=csv,noheader,nounits"], check=True,
                              capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to audit GPU processes") from error
    result = []
    for line in done.stdout.splitlines():
        value = line.strip()
        if not value or value.lower().startswith("no running processes"): continue
        try: pid = int(value)
        except ValueError as error: raise RuntimeError(f"unrecognized GPU PID: {value}") from error
        if pid > 0 and pid != os.getpid(): result.append(pid)
    return sorted(set(result))


def require_exclusive_idle_gpu(index: int) -> dict[str, Any]:
    name = query_gpu_name(index)
    if "4090" not in name: raise RuntimeError(f"v6 requires RTX 4090, found {name!r}")
    pids = query_compute_pids(index)
    if pids: raise RuntimeError(f"GPU {index} has concurrent compute PIDs {pids}")
    return {"gpu_index": index, "gpu_name": name, "compute_pids": []}


def acquire_lock(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error: raise RuntimeError(f"concurrent/stale lock exists: {path}") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def validate_stage(stage: str, output: Path) -> dict[str, Any]:
    manifest_path = output / "nested_oof_v6.json"
    if stage == "preregister":
        manifest = _json(manifest_path)
        validate_nested_oof_manifest(manifest, manifest["development_groups"])
        return {"artifact": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}
    if stage.startswith("fold_"):
        fold_id = int(stage.rsplit("_", 1)[-1]); path = output / "folds" / stage / "fold_summary.json"
        value = _json(path)
        if value.get("status") != "complete" or int(value.get("fold_id", -1)) != fold_id:
            raise RuntimeError(f"{stage} incomplete")
        if value.get("outer_labels_first_loaded_after_inner_selection_and_outer_refit") is not True:
            raise RuntimeError(f"{stage} lacks ordering proof")
        audit = value.get("outer_frozen_core_audit", {})
        if audit.get("factual_core_bit_exact") is not True or audit.get("trainable_parameter_names") != ["action_rank_head.0.weight"]:
            raise RuntimeError(f"{stage} violates frozen-core allowlist")
        if int(audit.get("trainable_parameter_count", -1)) != 192:
            raise RuntimeError(f"{stage} rank-head capacity changed")
        return {"artifact": str(path.resolve()), "sha256": sha256(path)}
    if stage == "select":
        path = output / "oof_selection_v6.json"; value = _json(path)
        if value.get("format") != SELECTION_FORMAT or value.get("status") != "complete_development_only":
            raise RuntimeError("v6 selection incomplete")
        unsigned = dict(value); recorded = unsigned.pop("selection_sha256", "")
        if recorded != canonical_sha256(unsigned): raise RuntimeError("v6 selection signature mismatch")
        auth = value.get("authorization", {})
        if auth.get("authorized") is not False or auth.get("fresh_confirmation_allowed") is not False:
            raise RuntimeError("v6 must never authorize fresh confirmation")
        required = {"frozen_base_success_only", "residual_only", "frozen_base_plus_residual"}
        if set(value.get("score_ablations", {})) != required:
            raise RuntimeError("v6 selection lacks required score ablations")
        record = value.get("prediction_diagnostics")
        if not isinstance(record, Mapping) or record.get("descriptive_only_not_guard_input") is not True:
            raise RuntimeError("v6 selection lacks descriptive prediction diagnostics")
        diagnostic_path = Path(str(record.get("path", "")))
        compatibility_path = Path(str(record.get("compatibility_manifest", "")))
        if not diagnostic_path.is_file(): diagnostic_path = output / diagnostic_path.name
        if not compatibility_path.is_file(): compatibility_path = output / compatibility_path.name
        if sha256(diagnostic_path) != record.get("sha256") or sha256(
            compatibility_path
        ) != record.get("compatibility_manifest_sha256"):
            raise RuntimeError("v6 prediction diagnostics provenance changed")
        diagnostics = _json(diagnostic_path)
        compatibility = _json(compatibility_path)
        validate_oof_prediction_diagnostics(
            diagnostics, compatibility, require_structured=True
        )
        if diagnostics.get("v6_preregistration_sha256") != value.get(
            "preregistration_sha256"
        ):
            raise RuntimeError("v6 prediction diagnostics used another preregistration")
        return {"artifact": str(path.resolve()), "sha256": sha256(path),
                "authorized": False, "development_gate_pass": value.get("development_gate_pass") is True,
                "prediction_diagnostics": str(diagnostic_path.resolve()),
                "prediction_diagnostics_sha256": sha256(diagnostic_path)}
    raise AssertionError(stage)


def run_stage(stage: Mapping[str, Any], *, output: Path, environment: Mapping[str, str]):
    log = output / "logs" / f"{stage['stage']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        done = subprocess.run(list(stage["argv"]), stdout=handle, stderr=subprocess.STDOUT,
                              env=dict(environment), check=False)
    if done.returncode: raise RuntimeError(f"v6 stage {stage['stage']} failed; see {log}")
    return {"status": "complete", "returncode": done.returncode, "log": str(log.resolve()),
            "log_sha256": sha256(log), **validate_stage(str(stage["stage"]), output)}


def execute_plan(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    output = args.output.expanduser().absolute(); output.mkdir(parents=True, exist_ok=False)
    permanent = output / "launch.lock"
    global_lock = args.gpu_lock.expanduser().absolute() if args.gpu_lock else Path(
        f"/tmp/etsf_openvla_oof_v6_gpu{args.gpu_index}.lock")
    payload = {"format": LAUNCH_FORMAT, "pid": os.getpid(), "plan_sha256": plan["plan_sha256"]}
    acquire_lock(permanent, payload); acquire_lock(global_lock, payload)
    state = {**plan, "status": "running", "stage_results": {},
             "fresh_confirmation_labels_read": False}
    atomic_json(output / "launch_plan.json", plan); atomic_json(output / "launch_state.json", state)
    env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_index),
        "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "8"})
    try:
        for stage in plan["commands"]:
            name = str(stage["stage"])
            if stage["uses_gpu"]:
                state.setdefault("gpu_idle_audits", {})[name] = require_exclusive_idle_gpu(args.gpu_index)
            state["current_stage"] = name; atomic_json(output / "launch_state.json", state)
            state["stage_results"][name] = run_stage(stage, output=output, environment=env)
            state["last_completed_stage"] = name; state["current_stage"] = None
            atomic_json(output / "launch_state.json", state)
        state["status"] = "complete_development_only_fresh_forbidden"
        state["fresh_confirmation_policy"] = "forbidden_even_if_development_gate_passes"
        atomic_json(output / "launch_state.json", state)
        return state
    except BaseException as error:
        state["status"] = "failed_nonresumable_new_output_required"
        state["error_type"] = type(error).__name__; state["error"] = str(error)
        atomic_json(output / "launch_state.json", state); raise
    finally:
        try:
            if global_lock.is_file():
                value = json.loads(global_lock.read_text(encoding="utf-8"))
                if value.get("pid") == os.getpid() and value.get("plan_sha256") == plan["plan_sha256"]:
                    global_lock.unlink()
        except (OSError, ValueError, json.JSONDecodeError): pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True); parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=Path(__file__).resolve().with_name("train_openvla_etsf_counterfactual_oof_v6.py"))
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable)); parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-lock", type=Path); parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu_index < 0 or args.num_workers < 0: raise ValueError("indices/workers must be non-negative")
    plan = preflight(args)
    if args.dry_run: print("OOF_V6_DRY_RUN=" + json.dumps(plan, sort_keys=True)); return
    execute_plan(args, plan)


if __name__ == "__main__": main()
