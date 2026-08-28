#!/usr/bin/env python3
"""Strict two-phase r3 resume after an LBFGS bridge failure.

The signed r3 materialization and five completed LBFGS checkpoints are reused.
The failed r3 directory is immutable.  A new output contains only the two OOF
evaluations and five newly trained AdamW checkpoints.  Fresh inputs are never
accepted and no selector or prospective claim can be authorized.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from launch_openvla_etsf_v8_adaptive_pipeline import (
    FORMAT as R3_FORMAT,
    _base_identity_sha,
    _publish_terminal_state,
    _reject_fresh_path,
    _run_stage,
    _validate_bridge_output,
    _wait_for_gpu_idle,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_path,
)


FORMAT = "etsf_openvla_v8_r3_bridge_fix_resume_plan_v1"
STATE_FORMAT = "etsf_openvla_v8_r3_bridge_fix_resume_state_v1"
TERMINAL_STATUS = "complete_r3_resume_adaptive_development_only_no_fresh"
MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
CHECKPOINT_FORMAT = "etsf_v8_detached_adapter_checkpoint_v1"
OLD_SCRIPT_NAMES = {
    "materializer": "materialize_openvla_etsf_v8_oof_inputs.py",
    "trainer": "train_openvla_etsf_v8_structured_adapters.py",
    "bridge": "evaluate_openvla_etsf_v8_oof_bridge.py",
    "factual_events": "evaluate_openvla_etsf_v8_factual_events.py",
}
NEW_IMPLEMENTATION_NAMES = {
    "pipeline_helpers": "launch_openvla_etsf_v8_adaptive_pipeline.py",
    "trainer": "train_openvla_etsf_v8_structured_adapters.py",
    "bridge": "evaluate_openvla_etsf_v8_oof_bridge.py",
    "array_evaluator": "evaluate_openvla_etsf_v8_structured_heads_arrays.py",
    "adaptive_protocol": "openvla_etsf_v8_adaptive_development_protocol.py",
    "structured_protocol": "openvla_etsf_v8_structured_heads_protocol.py",
    "adapters": "openvla_etsf_v8_structured_adapters.py",
}


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _verify_signed(value: Mapping[str, Any], key: str, *, name: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} signature mismatch")


def _script(code_root: Path, filename: str) -> Path:
    path = (code_root / "scripts" / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_failed_r3(
    *, plan_path: Path, state_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_json(plan_path)
    state = load_json(state_path)
    _verify_signed(plan, "plan_sha256", name="failed r3 plan")
    failure_position = (
        state.get("last_completed_stage"), state.get("current_stage")
    )
    if (
        plan.get("format") != R3_FORMAT
        or plan.get("status") != "waiting_for_v7"
        or state.get("format") != R3_FORMAT
        or state.get("plan_sha256") != plan.get("plan_sha256")
        or state.get("status") != "failed_closed_no_fresh"
        or failure_position
        not in (
            ("train_lbfgs_convex_fold_4", "evaluate_lbfgs_convex"),
            ("evaluate_lbfgs_convex", None),
        )
        or state.get("fresh50_inputs_accepted") is not False
        or state.get("fresh50_labels_read") is not False
        or state.get("automatic_fresh_launch") is not False
        or not isinstance(state.get("error"), str)
    ):
        raise RuntimeError("r3 is not the expected failed LBFGS bridge no-Fresh run")
    immutable = (
        "code_root",
        "implementation_files",
        "v7_state",
        "v7_result",
        "data",
        "checkpoint",
        "checkpoint_sha256",
        "event_spec",
        "event_spec_sha256",
        "python_bin",
        "gpu_index",
        "optimizer_candidates",
        "adaptive_development_only",
        "prospective_claim_allowed",
        "fresh50_inputs_accepted",
        "fresh50_labels_read",
        "automatic_fresh_launch",
    )
    if any(state.get(key) != plan.get(key) for key in immutable):
        raise RuntimeError("failed r3 state diverges from its signed plan")
    old_scripts = plan.get("implementation_files")
    if not isinstance(old_scripts, Mapping) or set(old_scripts) != set(OLD_SCRIPT_NAMES):
        raise RuntimeError("failed r3 four-script hash contract is incomplete")
    for name, filename in OLD_SCRIPT_NAMES.items():
        record = old_scripts[name]
        if (
            not isinstance(record, Mapping)
            or Path(str(record.get("path", ""))).name != filename
            or not _is_sha256(record.get("sha256"))
            or state.get("implementation_files", {}).get(name) != record
        ):
            raise RuntimeError(f"failed r3 old code hash is invalid: {name}")
    commands = state.get("commands")
    if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
        raise RuntimeError("failed r3 command history is missing")
    expected_stages = ["materialize_oof"] + [
        f"train_lbfgs_convex_fold_{fold}" for fold in range(5)
    ]
    for stage in expected_stages:
        matches = [
            row for row in commands
            if isinstance(row, Mapping) and row.get("stage") == stage
        ]
        if len(matches) != 1 or matches[0].get("argv_sha256") != canonical_sha256(
            matches[0].get("argv")
        ):
            raise RuntimeError(f"failed r3 command binding changed: {stage}")
        log = state.get("stage_logs", {}).get(stage)
        if not isinstance(log, Mapping):
            raise RuntimeError(f"failed r3 completed-stage log is missing: {stage}")
        log_path = Path(str(log.get("path", ""))).resolve()
        if not log_path.is_file() or sha256_path(log_path) != log.get("sha256"):
            raise RuntimeError(f"failed r3 completed-stage log hash changed: {stage}")
    last_command = state.get("commands", [])[-1]
    if not isinstance(last_command, Mapping) or last_command.get("stage") != (
        "evaluate_lbfgs_convex"
    ):
        raise RuntimeError("failed r3 command history does not end at LBFGS bridge")
    failed_bridge_log = Path(str(last_command.get("log", ""))).resolve()
    if not failed_bridge_log.is_file():
        raise RuntimeError("failed r3 LBFGS bridge log is missing")
    return plan, state


def _load_torch_mapping(path: Path, *, name: str) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must contain a mapping")
    return value


def _authenticate_materialization(
    *, manifest_path: Path, failed_root: Path, failed_plan: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = _reject_fresh_path(
        manifest_path, role="r3 development materialization"
    )
    expected_manifest = failed_root / "materialized_oof" / "materialization_manifest.json"
    if manifest_path.resolve() != expected_manifest.resolve():
        raise RuntimeError("r3 materialization is outside the failed run")
    manifest = load_json(manifest_path)
    _verify_signed(manifest, "materialization_sha256", name="r3 materialization")
    if (
        manifest.get("format") != MATERIALIZATION_FORMAT
        or manifest.get("status") != "complete_development_only"
        or manifest.get("fresh_confirmation_data_or_labels_read") is not False
        or manifest.get("authorization_guard_changed") is not False
        or manifest.get("prospective_claim_for_v8") is not False
        or manifest.get("base_checkpoint_sha256")
        != failed_plan.get("checkpoint_sha256")
        or manifest.get("event_spec_sha256") != failed_plan.get("event_spec_sha256")
    ):
        raise RuntimeError("r3 materialization changed adaptive/no-Fresh provenance")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or [row.get("outer_fold_id") for row in folds] != list(
        range(5)
    ):
        raise RuntimeError("r3 materialization is not a complete five-fold bundle")
    artifacts: list[dict[str, Any]] = []
    for fold_id, row in enumerate(folds):
        if not isinstance(row, Mapping):
            raise RuntimeError("r3 materialization fold is invalid")
        for role in ("train", "holdout"):
            path = Path(str(row.get(f"{role}_artifact", ""))).resolve()
            expected_path = manifest_path.parent / f"fold_{fold_id}_{role}.pt"
            file_sha = row.get(f"{role}_artifact_sha256")
            payload_sha = row.get(f"{role}_payload_sha256")
            if (
                path != expected_path.resolve()
                or not path.is_file()
                or not _is_sha256(file_sha)
                or sha256_path(path) != file_sha
                or not _is_sha256(payload_sha)
            ):
                raise RuntimeError(f"r3 fold {fold_id} {role} artifact hash changed")
            payload = _load_torch_mapping(path, name=f"fold {fold_id} {role}")
            if payload.get("payload_sha256") != payload_sha:
                raise RuntimeError(f"r3 fold {fold_id} {role} payload binding changed")
            batches = payload.get("batches")
            if not isinstance(batches, list) or not batches or any(
                not isinstance(record, Mapping)
                or not isinstance(record.get("batch"), Mapping)
                or "current_event_id" not in record["batch"]
                for record in batches
            ):
                raise RuntimeError(
                    f"r3 fold {fold_id} {role} lacks observed current_event_id"
                )
            artifacts.append(
                {
                    "outer_fold_id": fold_id,
                    "role": role,
                    "path": str(path),
                    "file_sha256": file_sha,
                    "payload_sha256": payload_sha,
                }
            )
    return {
        "path": str(manifest_path.resolve()),
        "file_sha256": sha256_path(manifest_path),
        "materialization_sha256": manifest["materialization_sha256"],
        "base_identity_contract_sha256": _base_identity_sha(manifest_path),
        "artifacts": artifacts,
        "ten_artifact_bundle_sha256": canonical_sha256(artifacts),
        "observed_current_event_id_verified_in_all_artifacts": True,
    }


def _expected_input_authentication(
    *, materialization: Mapping[str, Any], fold_id: int
) -> dict[str, Any]:
    artifacts = materialization["artifacts"]
    rows = [
        {
            "outer_fold_id": row["outer_fold_id"],
            "role": row["role"],
            "path": row["path"],
            "sha256": row["file_sha256"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in artifacts
    ]
    selected = next(
        row for row in artifacts
        if row["outer_fold_id"] == fold_id and row["role"] == "train"
    )
    return {
        "status": "authenticated_complete_five_fold_materialization_bundle",
        "materialization_manifest": materialization["path"],
        "materialization_sha256": materialization["materialization_sha256"],
        "outer_fold_id": fold_id,
        "train_artifact_sha256": selected["file_sha256"],
        "train_payload_sha256": selected["payload_sha256"],
        "ten_artifact_bundle_sha256": canonical_sha256(rows),
    }


def _authenticate_lbfgs_checkpoints(
    *, failed_root: Path, materialization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for fold_id in range(5):
        path = (failed_root / "lbfgs_convex" / f"fold_{fold_id}.pt").resolve()
        if not path.is_file():
            raise RuntimeError(f"r3 LBFGS checkpoint is missing for fold {fold_id}")
        checkpoint = _load_torch_mapping(path, name=f"LBFGS fold {fold_id}")
        provenance = checkpoint.get("provenance")
        if (
            checkpoint.get("format") != CHECKPOINT_FORMAT
            or int(checkpoint.get("schema_version", -1)) != 5
            or checkpoint.get("fresh_confirmation_data_or_labels_read") is not False
            or checkpoint.get("authorization_guard_changed") is not False
            or checkpoint.get("all_steps_factual_inputs_bit_exact") is not True
            or checkpoint.get("strict_oof_base_exclusion_eligible") is not True
            or checkpoint.get("optimizer", {}).get("name")
            != "independent_full_batch_LBFGS"
            or not isinstance(provenance, Mapping)
            or provenance.get("outer_fold_id") != fold_id
            or provenance.get("target_outer_fold_labels_used") is not False
            or checkpoint.get("input_artifact_authentication")
            != _expected_input_authentication(
                materialization=materialization, fold_id=fold_id
            )
        ):
            raise RuntimeError(f"r3 LBFGS checkpoint authentication failed for fold {fold_id}")
        checkpoints.append(
            {
                "outer_fold_id": fold_id,
                "path": str(path),
                "file_sha256": sha256_path(path),
                "adapter_state_sha256": checkpoint.get("adapter_state_sha256"),
                "input_artifact_authentication": checkpoint[
                    "input_artifact_authentication"
                ],
            }
        )
    return checkpoints


def _build_commands(
    *,
    python_bin: Path,
    code_root: Path,
    output_root: Path,
    materialization: Mapping[str, Any],
    lbfgs_checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    trainer = _script(code_root, NEW_IMPLEMENTATION_NAMES["trainer"])
    bridge = _script(code_root, NEW_IMPLEMENTATION_NAMES["bridge"])
    manifest_path = Path(str(materialization["path"]))
    materialized = manifest_path.parent
    commands: list[dict[str, Any]] = []

    def append(stage: str, argv: list[str], uses_gpu: bool) -> None:
        commands.append(
            {
                "stage": stage,
                "argv": argv,
                "argv_sha256": canonical_sha256(argv),
                "uses_gpu": uses_gpu,
            }
        )

    python = str(python_bin)
    lbfgs_bridge = [python, str(bridge)]
    for checkpoint in lbfgs_checkpoints:
        lbfgs_bridge.extend(["--checkpoint", str(checkpoint["path"])])
    for fold_id in range(5):
        lbfgs_bridge.extend(
            ["--holdout", str(materialized / f"fold_{fold_id}_holdout.pt")]
        )
    lbfgs_bridge.extend(
        [
            "--materialization-manifest",
            str(manifest_path),
            "--base-identity-contract-sha256",
            str(materialization["base_identity_contract_sha256"]),
            "--output",
            str(output_root / "evaluation_lbfgs_convex"),
        ]
    )
    append("evaluate_lbfgs_convex", lbfgs_bridge, False)
    adamw_checkpoints: list[Path] = []
    for fold_id in range(5):
        checkpoint = output_root / "adamw_fixed" / f"fold_{fold_id}.pt"
        argv = [
            python,
            str(trainer),
            "--input",
            str(materialized / f"fold_{fold_id}_train.pt"),
            "--materialization-manifest",
            str(manifest_path),
            "--outer-fold-id",
            str(fold_id),
            "--output",
            str(checkpoint),
            "--device",
            "cuda",
            "--optimizer-mode",
            "adamw",
            "--epochs",
            "10",
            "--learning-rate",
            "0.001",
        ]
        append(f"train_adamw_fixed_fold_{fold_id}", argv, True)
        adamw_checkpoints.append(checkpoint)
    adamw_bridge = [python, str(bridge)]
    for checkpoint in adamw_checkpoints:
        adamw_bridge.extend(["--checkpoint", str(checkpoint)])
    for fold_id in range(5):
        adamw_bridge.extend(
            ["--holdout", str(materialized / f"fold_{fold_id}_holdout.pt")]
        )
    adamw_bridge.extend(
        [
            "--materialization-manifest",
            str(manifest_path),
            "--base-identity-contract-sha256",
            str(materialization["base_identity_contract_sha256"]),
            "--output",
            str(output_root / "evaluation_adamw_fixed"),
        ]
    )
    append("evaluate_adamw_fixed", adamw_bridge, False)
    return commands


def build_resume_plan(
    *,
    failed_plan_path: Path,
    failed_state_path: Path,
    materialization_manifest: Path,
    code_root: Path,
    output_root: Path,
    python_bin: Path,
    gpu_index: int,
) -> dict[str, Any]:
    failed_plan_path = failed_plan_path.resolve()
    failed_state_path = failed_state_path.resolve()
    failed_root = _reject_fresh_path(
        failed_plan_path.parent, role="failed r3 development root"
    )
    if failed_state_path.parent != failed_root:
        raise RuntimeError("failed r3 plan/state must share one immutable root")
    code_root = code_root.resolve()
    output_root = _reject_fresh_path(output_root, role="r3 resume output")
    python_bin = Path(os.path.abspath(os.fspath(python_bin)))
    if output_root.exists():
        raise FileExistsError(output_root)
    if output_root == failed_root or failed_root in output_root.parents:
        raise RuntimeError("r3 resume output must not be inside the failed r3 root")
    if gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise RuntimeError(f"python interpreter is not executable: {python_bin}")
    failed_plan, failed_state = _validate_failed_r3(
        plan_path=failed_plan_path, state_path=failed_state_path
    )
    if code_root == Path(str(failed_plan.get("code_root", ""))).resolve():
        raise RuntimeError("r3 resume requires a distinct new code root")
    materialization = _authenticate_materialization(
        manifest_path=materialization_manifest.resolve(),
        failed_root=failed_root,
        failed_plan=failed_plan,
    )
    checkpoints = _authenticate_lbfgs_checkpoints(
        failed_root=failed_root, materialization=materialization
    )
    new_implementations = {
        name: {
            "path": str(_script(code_root, filename)),
            "sha256": sha256_path(_script(code_root, filename)),
        }
        for name, filename in NEW_IMPLEMENTATION_NAMES.items()
    }
    resume_launcher = Path(__file__).resolve()
    new_implementations["resume_launcher"] = {
        "path": str(resume_launcher),
        "sha256": sha256_path(resume_launcher),
    }
    plan = {
        "format": FORMAT,
        "status": "preregistered_no_execution",
        "failed_r3": {
            "plan": str(failed_plan_path),
            "plan_file_sha256": sha256_path(failed_plan_path),
            "plan_sha256": failed_plan["plan_sha256"],
            "state": str(failed_state_path),
            "state_file_sha256": sha256_path(failed_state_path),
            "failed_status": failed_state["status"],
            "failed_current_stage": failed_state["current_stage"],
            "failed_error_type": failed_state.get("error_type"),
            "failed_bridge_log": str(
                Path(str(failed_state["commands"][-1]["log"])).resolve()
            ),
            "failed_bridge_log_sha256": sha256_path(
                Path(str(failed_state["commands"][-1]["log"])).resolve()
            ),
        },
        "old_r3_implementation_files": failed_plan["implementation_files"],
        "resume_implementation_files": new_implementations,
        "materialization": materialization,
        "lbfgs_checkpoints": checkpoints,
        "code_root": str(code_root),
        "output_root": str(output_root),
        "python_bin": str(python_bin),
        "gpu_index": int(gpu_index),
        "commands": _build_commands(
            python_bin=python_bin,
            code_root=code_root,
            output_root=output_root,
            materialization=materialization,
            lbfgs_checkpoints=checkpoints,
        ),
        "resume_scope": "evaluate_existing_lbfgs_then_train_and_evaluate_adamw",
        "rematerialization_performed": False,
        "failed_r3_mutation_allowed": False,
        "adaptive_development_only": True,
        "prospective_claim_allowed": False,
        "selector_authorized": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "automatic_fresh_launch": False,
        "terminal_status": TERMINAL_STATUS,
    }
    return _signed(plan, "resume_plan_sha256")


def _recompute_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    failed = plan.get("failed_r3")
    materialization = plan.get("materialization")
    if not isinstance(failed, Mapping) or not isinstance(materialization, Mapping):
        raise RuntimeError("r3 resume plan provenance is incomplete")
    return build_resume_plan(
        failed_plan_path=Path(str(failed.get("plan", ""))),
        failed_state_path=Path(str(failed.get("state", ""))),
        materialization_manifest=Path(str(materialization.get("path", ""))),
        code_root=Path(str(plan.get("code_root", ""))),
        output_root=Path(str(plan.get("output_root", ""))),
        python_bin=Path(str(plan.get("python_bin", ""))),
        gpu_index=int(plan.get("gpu_index", -1)),
    )


def execute(plan_path: Path, *, poll_seconds: float) -> None:
    plan_path = plan_path.resolve()
    plan = load_json(plan_path)
    _verify_signed(plan, "resume_plan_sha256", name="r3 resume plan")
    if plan.get("format") != FORMAT or plan.get("status") != "preregistered_no_execution":
        raise RuntimeError("r3 resume plan status/format mismatch")
    recomputed = _recompute_plan(plan)
    if recomputed.get("resume_plan_sha256") != plan.get("resume_plan_sha256"):
        raise RuntimeError("r3 resume provenance changed after preregistration")
    output_root = Path(str(plan["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    logs.mkdir()
    state_path = output_root / "resume_state.json"
    atomic_json(output_root / "resume_plan.json", plan)
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "running_r3_resume",
        "resume_plan_sha256": plan["resume_plan_sha256"],
        "implementation_files": plan["resume_implementation_files"],
        "materialization": plan["materialization"],
        "lbfgs_checkpoints": plan["lbfgs_checkpoints"],
        "current_stage": None,
        "last_completed_stage": None,
        "commands": [],
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "automatic_fresh_launch": False,
    }
    atomic_json(state_path, state)
    try:
        code_root = Path(str(plan["code_root"])).resolve()
        child_env = dict(os.environ)
        child_env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(plan["gpu_index"]),
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(code_root / "scripts"),
                "OMP_NUM_THREADS": "4",
            }
        )
        audits: dict[str, Any] = {}
        gpu_checked = False
        for command in plan["commands"]:
            if command["uses_gpu"] and not gpu_checked:
                state["status"] = "waiting_for_gpu_idle"
                atomic_json(state_path, state)
                _wait_for_gpu_idle(
                    state=state,
                    state_path=state_path,
                    poll_seconds=poll_seconds,
                    gpu_index=int(plan["gpu_index"]),
                )
                state["status"] = "running_r3_resume"
                atomic_json(state_path, state)
                gpu_checked = True
            _run_stage(
                stage=str(command["stage"]),
                argv=list(command["argv"]),
                state=state,
                state_path=state_path,
                logs_dir=logs,
                code_root=code_root,
                child_env=child_env,
            )
            if str(command["stage"]).startswith("evaluate_"):
                mode = str(command["stage"]).removeprefix("evaluate_")
                audits[mode] = _validate_bridge_output(
                    output_dir=output_root / f"evaluation_{mode}",
                    materialization_manifest=Path(str(plan["materialization"]["path"])),
                )
            state["last_heartbeat_unix"] = time.time()
            atomic_json(state_path, state)
        if set(audits) != {"lbfgs_convex", "adamw_fixed"}:
            raise RuntimeError("r3 resume lacks both authenticated OOF evaluations")
        adamw = []
        for fold_id in range(5):
            path = output_root / "adamw_fixed" / f"fold_{fold_id}.pt"
            adamw.append(
                {"outer_fold_id": fold_id, "path": str(path), "file_sha256": sha256_path(path)}
            )
        summary = _signed(
            {
                "format": STATE_FORMAT,
                "status": TERMINAL_STATUS,
                "resume_plan_sha256": plan["resume_plan_sha256"],
                "failed_r3": plan["failed_r3"],
                "materialization": plan["materialization"],
                "lbfgs_checkpoints": plan["lbfgs_checkpoints"],
                "adamw_checkpoints": adamw,
                "evaluation_audits": audits,
                "rematerialization_performed": False,
                "failed_r3_mutated": False,
                "adaptive_development_only": True,
                "prospective_claim_allowed": False,
                "selector_authorized": False,
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
                "automatic_fresh_launch": False,
            },
            "summary_sha256",
        )
        _publish_terminal_state(
            summary_path=output_root / "resume_summary.json",
            state_path=state_path,
            summary=summary,
            state=state,
        )
    except BaseException as error:
        state.update(
            {
                "status": "failed_closed_r3_resume_no_fresh",
                "error_type": type(error).__name__,
                "error": str(error),
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
                "automatic_fresh_launch": False,
            }
        )
        atomic_json(state_path, state)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--failed-plan", type=Path, required=True)
    preregister.add_argument("--failed-state", type=Path, required=True)
    preregister.add_argument("--materialization-manifest", type=Path, required=True)
    preregister.add_argument("--code-root", type=Path, required=True)
    preregister.add_argument("--output-root", type=Path, required=True)
    preregister.add_argument("--python-bin", type=Path, required=True)
    preregister.add_argument("--gpu-index", type=int, default=0)
    preregister.add_argument("--plan-output", type=Path, required=True)
    run = subparsers.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preregister":
        output = _reject_fresh_path(args.plan_output, role="r3 resume plan")
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        plan = build_resume_plan(
            failed_plan_path=args.failed_plan,
            failed_state_path=args.failed_state,
            materialization_manifest=args.materialization_manifest,
            code_root=args.code_root,
            output_root=args.output_root,
            python_bin=args.python_bin,
            gpu_index=args.gpu_index,
        )
        atomic_json(output, plan)
        print(
            json.dumps(
                {"plan": str(output), "resume_plan_sha256": plan["resume_plan_sha256"]},
                sort_keys=True,
            )
        )
        return
    if not 5.0 <= args.poll_seconds <= 60.0:
        raise ValueError("poll-seconds must be in [5,60]")
    execute(args.plan, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
