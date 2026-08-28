#!/usr/bin/env python3
"""Fail-closed serial launcher for the formal five-fold OOF protocol.

The launcher owns one brand-new output root and executes exactly:
preregister -> fold_0 .. fold_4 -> select -> (authorized only) final.
It never accepts a fresh-confirmation input.  Every subprocess is synchronous,
logged separately, and audited before the next stage starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from evaluate_openvla_etsf_oof_prediction_diagnostics import (
    validate_oof_prediction_diagnostics,
)

from openvla_etsf_counterfactual_oof import (
    FOLD_COUNT,
    FORMAT,
    SELECTION_FORMAT,
    canonical_sha256,
    oof_dimensions,
    validate_oof_folds,
)
from train_openvla_etsf_counterfactual import canonical_policy_mapping, sha256


LAUNCH_FORMAT = "etsf_counterfactual_oof_serial_launch_v1"


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


def _command_sha(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_factual_policy_audit(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("factual checkpoint must contain a mapping")
    config = payload.get("config")
    contract = payload.get("contract")
    if not isinstance(config, Mapping) or config.get("structured_events") is not True:
        raise RuntimeError("OOF factual checkpoint must be structured")
    if not isinstance(contract, Mapping):
        raise RuntimeError("OOF factual checkpoint lacks a contract")
    raw = contract.get("policy_to_id")
    canonical = canonical_policy_mapping(raw)
    if "openvla" not in canonical:
        raise RuntimeError(
            "factual policy_to_id does not contain an OpenVLA name or path"
        )
    if int(canonical["openvla"]) < 0:
        raise RuntimeError("canonical OpenVLA policy id is invalid")
    return {
        "raw_policy_to_id": dict(raw),
        "canonical_policy_to_id": canonical,
        "canonical_openvla_id": int(canonical["openvla"]),
        "mapping_rule": "path_or_alias_containing_openvla_to_openvla_fail_on_id_collision",
    }


def build_stage_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = args.output.absolute()
    trainer = args.trainer.absolute()
    python_bin = args.python_bin.expanduser().absolute()
    common = [
        "--data",
        str(args.data.absolute()),
        "--pretrained",
        str(args.pretrained.absolute()),
        "--event-spec",
        str(args.event_spec.absolute()),
    ]
    fold_manifest = output / "oof_folds.json"
    selection = output / "oof_selection.json"
    commands: list[tuple[str, list[str], bool, Path | None]] = [
        (
            "preregister",
            [
                str(python_bin),
                str(trainer),
                "preregister",
                *common,
                "--output",
                str(fold_manifest),
            ],
            False,
            fold_manifest,
        )
    ]
    for fold_id in range(FOLD_COUNT):
        commands.append(
            (
                f"fold_{fold_id}",
                [
                    str(python_bin),
                    str(trainer),
                    "fold",
                    *common,
                    "--oof-manifest",
                    str(fold_manifest),
                    "--fold-id",
                    str(fold_id),
                    "--output",
                    str(output / "folds" / f"fold_{fold_id}"),
                    "--num-workers",
                    str(args.num_workers),
                ],
                True,
                output / "folds" / f"fold_{fold_id}" / "fold_summary.json",
            )
        )
    commands.extend(
        [
            (
                "select",
                [
                    str(python_bin),
                    str(trainer),
                    "select",
                    "--oof-manifest",
                    str(fold_manifest),
                    "--fold-root",
                    str(output / "folds"),
                    "--output",
                    str(selection),
                ],
                False,
                selection,
            ),
            (
                "final",
                [
                    str(python_bin),
                    str(trainer),
                    "final",
                    *common,
                    "--oof-manifest",
                    str(fold_manifest),
                    "--selection",
                    str(selection),
                    "--output",
                    str(output / "final"),
                    "--num-workers",
                    str(args.num_workers),
                ],
                True,
                output / "final" / "training_summary.json",
            ),
        ]
    )
    return [
        {
            "stage": name,
            "argv": argv,
            "argv_sha256": _command_sha(argv),
            "uses_gpu": uses_gpu,
            "expected_artifact": str(artifact) if artifact is not None else None,
        }
        for name, argv, uses_gpu, artifact in commands
    ]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(
            f"OOF launch requires one new unique output root: {output}"
        )
    paths = {
        "data_manifest": args.data.absolute() / "manifest.json",
        "pretrained": args.pretrained.absolute(),
        "event_spec": args.event_spec.absolute(),
        "trainer": args.trainer.absolute(),
        # Do not resolve the venv interpreter symlink.
        "python_bin": args.python_bin.expanduser().absolute(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    root_manifest = _json(paths["data_manifest"])
    if root_manifest.get("status") != "complete" or int(
        root_manifest.get("schema_version", -1)
    ) != 5:
        raise RuntimeError("OOF launcher requires a complete schema-v5 development root")
    if root_manifest.get("seed_registry") == "explicit_fresh_confirmation" or root_manifest.get(
        "fresh_seed_manifest_sha256"
    ) not in (None, ""):
        raise RuntimeError("fresh confirmation root is forbidden as OOF development data")
    groups = root_manifest.get("groups")
    if (
        not isinstance(groups, list)
        or len(groups) not in (100, 250)
        or int(root_manifest.get("completed", len(groups))) != len(groups)
    ):
        raise RuntimeError(
            "OOF launcher requires a frozen 100- or 250-group development manifest"
        )
    policy_audit = canonical_factual_policy_audit(paths["pretrained"])
    commands = build_stage_commands(args)
    plan = {
        "format": LAUNCH_FORMAT,
        "status": "preflight_complete",
        "output_root": str(output),
        "nonresumable": True,
        "existing_output_policy": "refuse_even_if_partial_or_complete",
        "serial_execution": True,
        "fresh_confirmation_inputs_accepted": False,
        "data": {
            "root": str(args.data.absolute()),
            "manifest": str(paths["data_manifest"]),
            "manifest_sha256": sha256(paths["data_manifest"]),
            "development_groups": len(groups),
            "seed_registry": root_manifest.get("seed_registry"),
        },
        "pretrained": {
            "path": str(paths["pretrained"]),
            "sha256": sha256(paths["pretrained"]),
            "policy_identity": policy_audit,
        },
        "event_spec": {
            "path": str(paths["event_spec"]),
            "sha256": sha256(paths["event_spec"]),
        },
        "trainer": {
            "path": str(paths["trainer"]),
            "sha256": sha256(paths["trainer"]),
        },
        "python_bin": str(paths["python_bin"]),
        "gpu": {
            "index": args.gpu_index,
            "required_name_substring": "4090",
            "global_lease": str(
                args.gpu_lock.absolute()
                if args.gpu_lock is not None
                else Path(f"/tmp/etsf_openvla_oof_gpu{args.gpu_index}.lock")
            ),
            "concurrent_compute_policy": "reject_before_every_gpu_stage",
        },
        "commands": commands,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def query_gpu_name(gpu_index: int) -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to audit GPU model with nvidia-smi") from error
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(names) != 1:
        raise RuntimeError(f"unexpected GPU-name audit output: {names}")
    return names[0]


def query_compute_pids(gpu_index: int) -> list[int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to audit GPU compute PIDs with nvidia-smi") from error
    result = []
    for raw in completed.stdout.splitlines():
        value = raw.strip()
        if not value or value.lower().startswith("no running processes"):
            continue
        try:
            pid = int(value)
        except ValueError as error:
            raise RuntimeError(f"unrecognized GPU PID row: {value!r}") from error
        if pid > 0 and pid != os.getpid():
            result.append(pid)
    return sorted(set(result))


def require_exclusive_idle_gpu(gpu_index: int) -> dict[str, Any]:
    name = query_gpu_name(gpu_index)
    if "4090" not in name:
        raise RuntimeError(f"formal OOF launch requires RTX 4090, found {name!r}")
    active = query_compute_pids(gpu_index)
    if active:
        raise RuntimeError(
            f"GPU {gpu_index} has concurrent compute PIDs {active}; launch refused"
        )
    return {"gpu_index": gpu_index, "gpu_name": name, "compute_pids": []}


def acquire_lock(path: Path, payload: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"concurrent/stale launch lock exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return os.getpid()


def validate_stage(stage: str, output: Path) -> dict[str, Any]:
    if stage == "preregister":
        manifest = _json(output / "oof_folds.json")
        validate_oof_folds(manifest, manifest["development_groups"])
        return {
            "artifact": str((output / "oof_folds.json").resolve()),
            "sha256": sha256(output / "oof_folds.json"),
        }
    if stage.startswith("fold_"):
        manifest = _json(output / "oof_folds.json")
        _, expected_training, expected_holdout = oof_dimensions(manifest)
        fold_id = int(stage.rsplit("_", 1)[-1])
        path = output / "folds" / stage / "fold_summary.json"
        value = _json(path)
        if value.get("status") != "complete" or int(value.get("fold_id", -1)) != fold_id:
            raise RuntimeError(f"{stage} summary is incomplete")
        if int(value.get("training_group_count", -1)) != expected_training or int(
            value.get("oof_holdout_group_count", -1)
        ) != expected_holdout:
            raise RuntimeError(
                f"{stage} does not satisfy frozen OOF dimensions "
                f"{expected_training}/{expected_holdout}"
            )
        if value.get("holdout_labels_first_loaded_after_member_checkpoints") is not True:
            raise RuntimeError(f"{stage} lacks heldout-access ordering proof")
        raw = value.get("raw_predictions")
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{stage} lacks raw prediction provenance")
        raw_path = Path(str(raw.get("path", "")))
        if not raw_path.is_file():
            raw_path = path.parent / raw_path.name
        if not raw_path.is_file() or sha256(raw_path) != str(raw.get("sha256", "")):
            raise RuntimeError(f"{stage} raw prediction SHA mismatch")
        return {"artifact": str(path.resolve()), "sha256": sha256(path)}
    if stage == "select":
        manifest = _json(output / "oof_folds.json")
        expected_groups, _, _ = oof_dimensions(manifest)
        path = output / "oof_selection.json"
        value = _json(path)
        if value.get("format") != SELECTION_FORMAT or value.get("status") != "complete":
            raise RuntimeError("OOF selection artifact is incomplete")
        unsigned = dict(value)
        recorded = str(unsigned.pop("selection_sha256", ""))
        if recorded != canonical_sha256(unsigned):
            raise RuntimeError("OOF selection artifact signature mismatch")
        authorization = value.get("authorization")
        if not isinstance(authorization, Mapping):
            raise RuntimeError("OOF selection lacks authorization audit")
        if int(value.get("oof_prediction_groups", -1)) != expected_groups or int(
            authorization.get("total_oof_groups", -1)
        ) != expected_groups:
            raise RuntimeError("OOF selection group count changed")
        diagnostics_path = output / "oof_prediction_diagnostics.json"
        diagnostics = _json(diagnostics_path)
        validate_oof_prediction_diagnostics(
            diagnostics, manifest, require_structured=True
        )
        return {
            "artifact": str(path.resolve()),
            "sha256": sha256(path),
            "authorized": authorization.get("authorized") is True,
            "rejection_reasons": list(authorization.get("rejection_reasons", [])),
            "prediction_diagnostics": str(diagnostics_path.resolve()),
            "prediction_diagnostics_sha256": sha256(diagnostics_path),
        }
    if stage == "final":
        manifest = _json(output / "oof_folds.json")
        expected_groups, _, _ = oof_dimensions(manifest)
        summary_path = output / "final" / "training_summary.json"
        summary = _json(summary_path)
        if summary.get("status") != "complete" or summary.get("oof_authorized") is not True:
            raise RuntimeError("OOF final refit summary is incomplete or unauthorized")
        if summary.get("fresh_confirmation_labels_read") is not False:
            raise RuntimeError("OOF final refit touched fresh confirmation labels")
        if int(summary.get("development_groups", -1)) != expected_groups:
            raise RuntimeError("OOF final refit group count changed")
        ensemble_path = output / "final" / "ensemble_manifest.json"
        ensemble = _json(ensemble_path)
        if ensemble.get("test_policy") != "fresh50_one_shot_only_after_oof_authorization":
            raise RuntimeError("OOF final ensemble lacks fresh50 one-shot policy")
        guard = ensemble.get("guard")
        if not isinstance(guard, Mapping) or guard.get("enabled") is not True:
            raise RuntimeError("OOF final ensemble guard is not enabled")
        return {
            "artifact": str(summary_path.resolve()),
            "sha256": sha256(summary_path),
            "ensemble_manifest": str(ensemble_path.resolve()),
            "ensemble_manifest_sha256": sha256(ensemble_path),
        }
    raise AssertionError(stage)


def run_stage(
    stage: Mapping[str, Any],
    *,
    output: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    log_path = output / "logs" / f"{stage['stage']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            list(stage["argv"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"OOF stage {stage['stage']} failed with exit {completed.returncode}; "
            f"see {log_path}"
        )
    audit = validate_stage(str(stage["stage"]), output)
    return {
        "status": "complete",
        "returncode": completed.returncode,
        "log": str(log_path.resolve()),
        "log_sha256": sha256(log_path),
        **audit,
    }


def parse_args() -> argparse.Namespace:
    local_trainer = Path(__file__).resolve().parent / "train_openvla_etsf_counterfactual_oof.py"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=local_trainer)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--gpu-lock",
        type=Path,
        default=None,
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def execute_plan(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a preflighted plan serially; exposed for CPU state-machine tests."""

    output = args.output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    output_lock = output / "launch.lock"
    global_lock = (
        args.gpu_lock.expanduser().absolute()
        if args.gpu_lock is not None
        else Path(f"/tmp/etsf_openvla_oof_gpu{args.gpu_index}.lock")
    )
    lock_payload = {
        "format": LAUNCH_FORMAT,
        "pid": os.getpid(),
        "plan_sha256": plan["plan_sha256"],
        "output": str(output),
    }
    acquire_lock(output_lock, lock_payload)
    acquire_lock(global_lock, lock_payload)
    state: dict[str, Any] = {
        **plan,
        "status": "running",
        "pid": os.getpid(),
        "output_lock": str(output_lock),
        "global_gpu_lock": str(global_lock),
        "stage_results": {},
        "fresh_confirmation_labels_read": False,
    }
    atomic_json(output / "launch_plan.json", plan)
    atomic_json(output / "launch_state.json", state)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
    try:
        for stage in plan["commands"]:
            name = str(stage["stage"])
            if name == "final":
                selection = state["stage_results"].get("select", {})
                if selection.get("authorized") is not True:
                    state["status"] = "stopped_guard_not_authorized"
                    state["fresh_confirmation_policy"] = "forbidden"
                    atomic_json(output / "launch_state.json", state)
                    print("OOF_GUARD_NOT_AUTHORIZED=" + json.dumps(selection, sort_keys=True))
                    return state
            if bool(stage["uses_gpu"]):
                state.setdefault("gpu_idle_audits", {})[name] = require_exclusive_idle_gpu(
                    args.gpu_index
                )
                atomic_json(output / "launch_state.json", state)
            state["current_stage"] = name
            atomic_json(output / "launch_state.json", state)
            state["stage_results"][name] = run_stage(
                stage, output=output, environment=environment
            )
            state["current_stage"] = None
            state["last_completed_stage"] = name
            atomic_json(output / "launch_state.json", state)
        state["status"] = "complete_fresh50_ready_one_shot"
        state["fresh_confirmation_policy"] = "one_shot_only"
        atomic_json(output / "launch_state.json", state)
        print("OOF_SERIAL_COMPLETE=" + json.dumps(state["stage_results"]["final"], sort_keys=True))
        return state
    except BaseException as error:
        state["status"] = "failed_nonresumable_new_output_required"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        atomic_json(output / "launch_state.json", state)
        raise
    finally:
        # Remove only the process-scoped GPU lease.  launch.lock remains as a
        # permanent no-resume marker inside the unique output root.
        try:
            if global_lock.is_file():
                recorded = json.loads(global_lock.read_text(encoding="utf-8"))
                if int(recorded.get("pid", -1)) == os.getpid() and recorded.get(
                    "plan_sha256"
                ) == plan["plan_sha256"]:
                    global_lock.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            # A lock that cannot be safely proven ours is intentionally left.
            pass


def main() -> None:
    args = parse_args()
    if args.gpu_index < 0 or args.num_workers < 0:
        raise ValueError("gpu-index and num-workers must be non-negative")
    plan = preflight(args)
    if args.dry_run:
        print("OOF_SERIAL_DRY_RUN=" + json.dumps(plan, sort_keys=True))
        return
    execute_plan(args, plan)


if __name__ == "__main__":
    main()
