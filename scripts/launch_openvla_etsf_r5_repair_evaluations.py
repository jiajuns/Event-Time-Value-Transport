#!/usr/bin/env python3
"""Immutable two-phase launcher for the R5 success/duration repair evaluations.

``preregister`` authenticates and signs every input, implementation and exact
command without creating the execution root.  ``execute`` recomputes that plan,
waits for an idle GPU, runs success inner-CV calibration on CUDA and duration
hierarchy evaluation on CPU, then authenticates both outputs.  ``detach`` starts
``execute`` through nohup with a new session and immutable receipt.

This launcher accepts adaptive development artifacts only.  Any external path
containing Fresh or confirmation is rejected, and no selector is authorized.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


FORMAT = "etsf_openvla_r5_repair_evaluations_plan_v1"
STATE_FORMAT = "etsf_openvla_r5_repair_evaluations_state_v1"
SUMMARY_FORMAT = "etsf_openvla_r5_repair_evaluations_summary_v1"
DETACH_FORMAT = "etsf_openvla_r5_repair_evaluations_detach_receipt_v1"
TERMINAL_STATUS = "complete_r5_repair_evaluations_adaptive_development_no_fresh"
FAILURE_STATUS = "failed_closed_r5_repair_evaluations_no_fresh"
MATERIALIZATION_FORMAT = "etsf_v8_oof_materialization_manifest_v1"
R4_SUMMARY_FORMAT = "etsf_openvla_v8_r3_bridge_fix_resume_state_v1"
R4_TERMINAL_STATUS = "complete_r3_resume_adaptive_development_only_no_fresh"
CHECKPOINT_FORMAT = "etsf_v8_detached_adapter_checkpoint_v1"
SUCCESS_FORMAT = "etsf_v8_success_group_inner_cv_shrinkage_oof_v1"
DURATION_FORMAT = "etsf_r5_duration_hierarchy_oof_evaluation_v1"
FOLD_COUNT = 5
EXPECTED_SUCCESS_ROWS = 1_000
ENTRYPOINTS = (
    "launch_openvla_etsf_r5_repair_evaluations.py",
    "calibrate_openvla_etsf_v8_success_inner_cv.py",
    "evaluate_openvla_etsf_duration_hierarchy_oof.py",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _reject_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def _absolute_unresolved_path(path: Path, *, role: str) -> Path:
    """Keep an interpreter symlink intact while applying the path-scope guard."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if any(
        token in part.lower()
        for part in absolute.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return absolute


def _python_contract(path: Path) -> dict[str, Any]:
    invocation = _absolute_unresolved_path(path, role="R5 Python")
    if not invocation.is_file() or not os.access(invocation, os.X_OK):
        raise RuntimeError("R5 Python interpreter is not executable")
    metadata = invocation.lstat()
    resolved = invocation.resolve()
    contract: dict[str, Any] = {
        "invocation_path": str(invocation),
        "invocation_path_is_symlink": stat.S_ISLNK(metadata.st_mode),
        "invocation_file_sha256": sha256_path(invocation),
        "invocation_lstat": {
            "mode": int(metadata.st_mode),
            "size": int(metadata.st_size),
            "mtime_ns": int(metadata.st_mtime_ns),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
        },
        "resolved_target_path": str(resolved),
        "resolved_target_file_sha256": sha256_path(resolved),
    }
    if contract["invocation_path_is_symlink"]:
        raw_target = os.readlink(invocation)
        contract.update(
            {
                "symlink_target": raw_target,
                "symlink_target_sha256": hashlib.sha256(
                    os.fsencode(raw_target)
                ).hexdigest(),
            }
        )
    else:
        contract.update({"symlink_target": None, "symlink_target_sha256": None})
    return contract


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    path = _reject_path(path, role=role)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must be a JSON mapping")
    return value


def _verify_signed(value: Mapping[str, Any], key: str, *, role: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{role} signature mismatch")


def _signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _atomic_state_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _local_python_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def implementation_closure(code_root: Path) -> dict[str, dict[str, str]]:
    """Hash the complete recursive scripts-local Python dependency closure."""

    code_root = _reject_path(code_root, role="R5 code root")
    scripts = code_root / "scripts"
    queue = [scripts / name for name in ENTRYPOINTS]
    seen: set[Path] = set()
    while queue:
        path = queue.pop().resolve()
        if path in seen:
            continue
        if not path.is_file() or scripts.resolve() not in path.parents:
            raise FileNotFoundError(path)
        seen.add(path)
        for module in _local_python_imports(path):
            candidate = scripts / (module.replace(".", "/") + ".py")
            if candidate.is_file():
                queue.append(candidate)
    records = {
        str(path.relative_to(code_root)): {
            "path": str(path),
            "sha256": sha256_path(path),
        }
        for path in sorted(seen)
    }
    if any(name not in {Path(key).name for key in records} for name in ENTRYPOINTS):
        raise RuntimeError("R5 implementation closure lost an entrypoint")
    return records


def _authenticate_materialization(manifest_path: Path) -> dict[str, Any]:
    manifest_path = _reject_path(manifest_path, role="R3 materialization manifest")
    manifest = _load_json(manifest_path, role="R3 materialization manifest")
    _verify_signed(manifest, "materialization_sha256", role="R3 materialization")
    folds = manifest.get("folds")
    if (
        manifest.get("format") != MATERIALIZATION_FORMAT
        or manifest.get("status") != "complete_development_only"
        or manifest.get("fresh_confirmation_data_or_labels_read") is not False
        or manifest.get("authorization_guard_changed") is not False
        or manifest.get("prospective_claim_for_v8") is not False
        or not _is_sha256(manifest.get("base_checkpoint_sha256"))
        or not _is_sha256(manifest.get("event_spec_sha256"))
        or not isinstance(folds, list)
        or [row.get("outer_fold_id") for row in folds] != list(range(FOLD_COUNT))
    ):
        raise RuntimeError("R3 materialization contract is invalid")
    artifacts: list[dict[str, Any]] = []
    checkpoint_bundle_rows: list[dict[str, Any]] = []
    owner_by_group: dict[str, int] = {}
    group_universe: set[str] | None = None
    for fold_id, row in enumerate(folds):
        if not isinstance(row, Mapping):
            raise RuntimeError("R3 materialization fold row is invalid")
        training = list(map(str, row.get("training_groups", ())))
        holdout = list(map(str, row.get("oof_holdout_groups", ())))
        combined = set(training) | set(holdout)
        if (
            not training
            or not holdout
            or training != sorted(training)
            or holdout != sorted(holdout)
            or len(training) != len(set(training))
            or len(holdout) != len(set(holdout))
            or set(training) & set(holdout)
            or len(training) + len(holdout) != len(combined)
            or row.get("training_groups_sha256")
            != canonical_sha256({"logical_groups": training})
            or row.get("oof_holdout_groups_sha256")
            != canonical_sha256({"logical_groups": holdout})
        ):
            raise RuntimeError(f"R3 fold {fold_id} group partition changed")
        if group_universe is None:
            group_universe = combined
        elif combined != group_universe:
            raise RuntimeError("R3 fold group universes differ")
        for group in holdout:
            if group in owner_by_group:
                raise RuntimeError("R3 logical group has multiple holdout owners")
            owner_by_group[group] = fold_id
        for role in ("train", "holdout"):
            path = _reject_path(
                Path(str(row.get(f"{role}_artifact", ""))),
                role=f"R3 fold {fold_id} {role}",
            )
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
                raise RuntimeError(f"R3 fold {fold_id} {role} artifact changed")
            artifacts.append(
                {
                    "outer_fold_id": fold_id,
                    "role": role,
                    "path": str(path),
                    "file_sha256": file_sha,
                    "payload_sha256": payload_sha,
                }
            )
            checkpoint_bundle_rows.append(
                {
                    "outer_fold_id": fold_id,
                    "role": role,
                    "path": str(path),
                    "sha256": file_sha,
                    "payload_sha256": payload_sha,
                }
            )
    if (
        group_universe is None
        or len(group_universe) != 250
        or set(owner_by_group) != group_universe
        or len({row["path"] for row in artifacts}) != 10
    ):
        raise RuntimeError("R3 bundle is not one-owner 250-group development OOF")
    return {
        "path": str(manifest_path),
        "file_sha256": sha256_path(manifest_path),
        "materialization_sha256": manifest["materialization_sha256"],
        "base_checkpoint_sha256": manifest.get("base_checkpoint_sha256"),
        "event_spec_sha256": manifest.get("event_spec_sha256"),
        "artifacts": artifacts,
        "ten_artifact_sha256": canonical_sha256(artifacts),
        "checkpoint_auth_ten_artifact_bundle_sha256": canonical_sha256(
            checkpoint_bundle_rows
        ),
        "folds": [dict(row) for row in folds],
    }


def _expected_checkpoint_authentication(
    materialization: Mapping[str, Any], *, fold_id: int
) -> dict[str, Any]:
    selected = next(
        row
        for row in materialization["artifacts"]
        if row["outer_fold_id"] == fold_id and row["role"] == "train"
    )
    return {
        "status": "authenticated_complete_five_fold_materialization_bundle",
        "materialization_manifest": materialization["path"],
        "materialization_sha256": materialization["materialization_sha256"],
        "outer_fold_id": fold_id,
        "train_artifact_sha256": selected["file_sha256"],
        "train_payload_sha256": selected["payload_sha256"],
        "ten_artifact_bundle_sha256": materialization[
            "checkpoint_auth_ten_artifact_bundle_sha256"
        ],
    }


def _authenticate_r4_adamw(
    summary_path: Path, *, materialization: Mapping[str, Any]
) -> dict[str, Any]:
    summary_path = _reject_path(summary_path, role="R4 summary")
    summary = _load_json(summary_path, role="R4 summary")
    _verify_signed(summary, "summary_sha256", role="R4 summary")
    summary_materialization = summary.get("materialization")
    checkpoints = summary.get("adamw_checkpoints")
    if (
        summary.get("format") != R4_SUMMARY_FORMAT
        or summary.get("status") != R4_TERMINAL_STATUS
        or summary.get("fresh50_inputs_accepted") is not False
        or summary.get("fresh50_labels_read") is not False
        or summary.get("selector_authorized") is not False
        or summary.get("prospective_claim_allowed") is not False
        or not isinstance(summary_materialization, Mapping)
        or summary_materialization.get("path") != materialization["path"]
        or summary_materialization.get("materialization_sha256")
        != materialization["materialization_sha256"]
        or not isinstance(checkpoints, list)
        or [row.get("outer_fold_id") for row in checkpoints]
        != list(range(FOLD_COUNT))
    ):
        raise RuntimeError("R4 summary is not the expected no-Fresh AdamW run")
    authenticated: list[dict[str, Any]] = []
    for fold_id, record in enumerate(checkpoints):
        path = _reject_path(
            Path(str(record.get("path", ""))), role=f"R4 AdamW fold {fold_id}"
        )
        recorded_file_sha = record.get("file_sha256")
        if (
            not path.is_file()
            or not _is_sha256(recorded_file_sha)
            or sha256_path(path) != recorded_file_sha
        ):
            raise RuntimeError(f"R4 AdamW checkpoint file changed for fold {fold_id}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        provenance = checkpoint.get("provenance") if isinstance(checkpoint, Mapping) else None
        optimizer = checkpoint.get("optimizer") if isinstance(checkpoint, Mapping) else None
        record_order = optimizer.get("record_order") if isinstance(optimizer, Mapping) else None
        record_order_sha256 = (
            hashlib.sha256(
                "\n".join(map(str, record_order)).encode("utf-8")
            ).hexdigest()
            if isinstance(record_order, Sequence)
            and not isinstance(record_order, (str, bytes))
            else None
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("format") != CHECKPOINT_FORMAT
            or checkpoint.get("fresh_confirmation_data_or_labels_read") is not False
            or checkpoint.get("authorization_guard_changed") is not False
            or checkpoint.get("all_steps_factual_inputs_bit_exact") is not True
            or not _is_sha256(checkpoint.get("adapter_state_sha256"))
            or not isinstance(provenance, Mapping)
            or provenance.get("outer_fold_id") != fold_id
            or provenance.get("target_outer_fold_labels_used") is not False
            or provenance.get("factual_outputs_frozen") is not True
            or checkpoint.get("input_artifact_authentication")
            != _expected_checkpoint_authentication(
                materialization, fold_id=fold_id
            )
            or not isinstance(optimizer, Mapping)
            or optimizer.get("name") != "AdamW"
            or optimizer.get("epochs") != 10
            or float(optimizer.get("learning_rate", -1.0)) != 0.001
            or float(optimizer.get("weight_decay", -1.0)) != 0.0
            or not isinstance(record_order, Sequence)
            or isinstance(record_order, (str, bytes))
            or not record_order
            or len(record_order) != len(set(map(str, record_order)))
            or optimizer.get("record_order_sha256") != record_order_sha256
            or not _is_sha256(record_order_sha256)
            or checkpoint.get("steps") != 10 * len(record_order)
            or checkpoint.get("last_step", {}).get("gradient_clip_scope")
            != "independent_per_probability_head"
        ):
            raise RuntimeError(f"R4 AdamW checkpoint provenance failed for fold {fold_id}")
        fold = materialization["folds"][fold_id]
        if (
            provenance.get("outer_training_groups") != fold.get("training_groups")
            or provenance.get("outer_training_groups_sha256")
            != fold.get("training_groups_sha256")
            or provenance.get("oof_holdout_groups") != fold.get("oof_holdout_groups")
            or provenance.get("oof_holdout_groups_sha256")
            != fold.get("oof_holdout_groups_sha256")
            or list(map(str, record_order))
            != list(map(str, provenance.get("outer_training_groups", ())))
        ):
            raise RuntimeError(f"R4 AdamW owner group provenance changed for fold {fold_id}")
        authenticated.append(
            {
                "outer_fold_id": fold_id,
                "path": str(path),
                "file_sha256": recorded_file_sha,
                "adapter_state_sha256": checkpoint.get("adapter_state_sha256"),
                "provenance_sha256": canonical_sha256(provenance),
                "input_artifact_authentication": checkpoint[
                    "input_artifact_authentication"
                ],
                "optimizer": {
                    "name": "AdamW",
                    "epochs": optimizer["epochs"],
                    "learning_rate": optimizer["learning_rate"],
                    "weight_decay": optimizer["weight_decay"],
                    "record_order_sha256": optimizer.get("record_order_sha256"),
                    "steps": checkpoint["steps"],
                    "gradient_clip_scope": checkpoint["last_step"][
                        "gradient_clip_scope"
                    ],
                },
            }
        )
    return {
        "summary": str(summary_path),
        "summary_file_sha256": sha256_path(summary_path),
        "summary_sha256": summary["summary_sha256"],
        "checkpoints": authenticated,
    }


def _build_commands(
    *,
    python_bin: Path,
    code_root: Path,
    output_root: Path,
    materialization: Mapping[str, Any],
    r4: Mapping[str, Any],
) -> list[dict[str, Any]]:
    success = code_root / "scripts" / "calibrate_openvla_etsf_v8_success_inner_cv.py"
    duration = code_root / "scripts" / "evaluate_openvla_etsf_duration_hierarchy_oof.py"
    success_argv = [str(python_bin), str(success)]
    for checkpoint in r4["checkpoints"]:
        success_argv.extend(["--checkpoint", checkpoint["path"]])
    success_argv.extend(
        [
            "--materialization-manifest",
            materialization["path"],
            "--output",
            str(output_root / "success_calibration.json"),
            "--device",
            "cuda",
        ]
    )
    duration_argv = [
        str(python_bin),
        str(duration),
        "--materialization-manifest",
        materialization["path"],
        "--output",
        str(output_root / "duration_hierarchy"),
    ]
    return [
        {
            "stage": "success_calibration",
            "argv": success_argv,
            "argv_sha256": canonical_sha256(success_argv),
            "uses_gpu": True,
            "device_contract": "CUDA_visible_single_preregistered_gpu",
        },
        {
            "stage": "duration_hierarchy",
            "argv": duration_argv,
            "argv_sha256": canonical_sha256(duration_argv),
            "uses_gpu": False,
            "device_contract": "CPU_only_CUDA_VISIBLE_DEVICES_empty",
        },
    ]


def build_plan(
    *,
    code_root: Path,
    materialization_manifest: Path,
    r4_summary: Path,
    output_root: Path,
    python_bin: Path,
    gpu_index: int,
) -> dict[str, Any]:
    code_root = _reject_path(code_root, role="R5 code root")
    output_root = _reject_path(output_root, role="R5 output root")
    python_contract = _python_contract(python_bin)
    python_bin = Path(python_contract["invocation_path"])
    if output_root.exists():
        raise FileExistsError(output_root)
    if not code_root.is_dir():
        raise FileNotFoundError(code_root)
    if gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    implementations = implementation_closure(code_root)
    materialization = _authenticate_materialization(materialization_manifest)
    r4 = _authenticate_r4_adamw(r4_summary, materialization=materialization)
    plan = {
        "format": FORMAT,
        "status": "preregistered_no_execution",
        "code_root": str(code_root),
        "implementation_files": implementations,
        "implementation_bundle_sha256": canonical_sha256(implementations),
        "materialization": materialization,
        "r4_adamw": r4,
        "output_root": str(output_root),
        "python_bin": str(python_bin),
        "python_file_sha256": python_contract["invocation_file_sha256"],
        "python_contract": python_contract,
        "gpu_index": int(gpu_index),
        "commands": _build_commands(
            python_bin=python_bin,
            code_root=code_root,
            output_root=output_root,
            materialization=materialization,
            r4=r4,
        ),
        "execution_order": ["success_calibration", "duration_hierarchy"],
        "success_stage_device": "cuda",
        "duration_stage_device": "cpu",
        "output_root_must_not_exist_before_execute": True,
        "adaptive_development_only": True,
        "prospective_claim_allowed": False,
        "selector_authorized": False,
        "automatic_fresh_launch": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "terminal_status": TERMINAL_STATUS,
    }
    return _signed(plan, "plan_sha256")


def _recompute_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    materialization = plan.get("materialization")
    r4 = plan.get("r4_adamw")
    if not isinstance(materialization, Mapping) or not isinstance(r4, Mapping):
        raise RuntimeError("R5 plan provenance is incomplete")
    return build_plan(
        code_root=Path(str(plan.get("code_root", ""))),
        materialization_manifest=Path(str(materialization.get("path", ""))),
        r4_summary=Path(str(r4.get("summary", ""))),
        output_root=Path(str(plan.get("output_root", ""))),
        python_bin=Path(str(plan.get("python_bin", ""))),
        gpu_index=int(plan.get("gpu_index", -1)),
    )


def _gpu_compute_pids(gpu_index: int) -> list[int]:
    argv = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "nvidia-smi GPU-idle query timed out after 30 seconds; fail closed"
        ) from error
    result = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            result.append(int(line))
        elif line and line.lower() not in (
            "no running processes found",
            "no running processes found.",
        ):
            raise RuntimeError(f"unexpected nvidia-smi compute PID output: {line}")
    return sorted(set(result))


def _wait_for_gpu_idle(
    *, state: dict[str, Any], state_path: Path, gpu_index: int, poll_seconds: float
) -> dict[str, Any]:
    checks = 0
    while True:
        pids = _gpu_compute_pids(gpu_index)
        checks += 1
        audit = {
            "gpu_index": gpu_index,
            "compute_pids": pids,
            "checks": checks,
            "last_check_unix": time.time(),
        }
        state["gpu_idle_audit"] = audit
        state["last_heartbeat_unix"] = time.time()
        _atomic_state_json(state_path, state)
        if not pids:
            return audit
        time.sleep(poll_seconds)


def _validate_implementations(plan: Mapping[str, Any]) -> None:
    actual = implementation_closure(Path(str(plan["code_root"])))
    if (
        actual != plan.get("implementation_files")
        or canonical_sha256(actual) != plan.get("implementation_bundle_sha256")
    ):
        raise RuntimeError("R5 implementation changed after preregistration")


def _validate_runtime_bindings(plan: Mapping[str, Any]) -> None:
    """Reauthenticate every immutable input immediately before each stage."""

    _validate_implementations(plan)
    if _python_contract(Path(str(plan["python_bin"]))) != plan.get("python_contract"):
        raise RuntimeError("R5 Python invocation/target changed after preregistration")
    materialization = _authenticate_materialization(
        Path(str(plan["materialization"]["path"]))
    )
    if materialization != plan.get("materialization"):
        raise RuntimeError("R3 materialization changed after preregistration")
    r4 = _authenticate_r4_adamw(
        Path(str(plan["r4_adamw"]["summary"])),
        materialization=materialization,
    )
    if r4 != plan.get("r4_adamw"):
        raise RuntimeError("R4 AdamW bundle changed after preregistration")


def _run_stage(
    *,
    command: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
) -> None:
    stage = str(command["stage"])
    if command.get("argv_sha256") != canonical_sha256(command.get("argv")):
        raise RuntimeError(f"R5 {stage} command signature changed")
    _validate_runtime_bindings(plan)
    log = logs_dir / f"{stage}.log"
    temporary_log = logs_dir / f".{stage}.log.partial"
    if log.exists() or temporary_log.exists():
        raise FileExistsError(f"R5 immutable stage log already exists for {stage}")
    state["current_stage"] = stage
    state.setdefault("commands", []).append(dict(command))
    state["commands"][-1]["log"] = str(log)
    state["last_heartbeat_unix"] = time.time()
    _atomic_state_json(state_path, state)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path(str(plan["code_root"])) / "scripts"),
            "OMP_NUM_THREADS": "4",
            "CUDA_VISIBLE_DEVICES": str(plan["gpu_index"])
            if command.get("uses_gpu") is True
            else "",
        }
    )
    try:
        with temporary_log.open("xb") as handle:
            completed = subprocess.run(
                list(command["argv"]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=Path(str(plan["code_root"])),
                env=environment,
            )
        os.replace(temporary_log, log)
    except BaseException:
        if temporary_log.exists() and not log.exists():
            os.replace(temporary_log, log)
        raise
    if completed.returncode != 0:
        raise RuntimeError(f"R5 stage {stage} failed; see {log}")
    state["current_stage"] = None
    state["last_completed_stage"] = stage
    state.setdefault("stage_logs", {})[stage] = {
        "path": str(log),
        "sha256": sha256_path(log),
    }
    _atomic_state_json(state_path, state)


def _validate_success_output(
    path: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    value = _load_json(path, role="R5 success output")
    _verify_signed(value, "result_sha256", role="R5 success output")
    rows = value.get("calibrated_oof_rows")
    contracts = value.get("fold_calibration_contracts")
    if (
        value.get("format") != SUCCESS_FORMAT
        or value.get("status") != "complete_adaptive_development_only"
        or value.get("materialization_manifest")
        != plan["materialization"]["path"]
        or value.get("materialization_sha256")
        != plan["materialization"]["materialization_sha256"]
        or value.get("fresh50_inputs_accepted") is not False
        or value.get("fresh50_labels_read") is not False
        or value.get("outer_holdout_labels_used_for_alpha_selection") is not False
        or value.get("authorization", {}).get("selector_authorized") is not False
        or value.get("authorization", {}).get("deployment_authorized") is not False
        or value.get("action_ranking_preserved_within_each_group") is not True
        or value.get("task_success_cannot_change_from_uncalibrated_argmax") is not True
        or value.get("all_alpha_selection_completed_before_holdout_payload_deserialization")
        is not True
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_SUCCESS_ROWS
        or value.get("calibrated_oof_row_count") != EXPECTED_SUCCESS_ROWS
        or value.get("calibrated_oof_rows_sha256") != canonical_sha256(rows)
        or not isinstance(contracts, list)
        or len(contracts) != FOLD_COUNT
    ):
        raise RuntimeError("R5 success output contract failed")
    checkpoint_shas = {
        row["outer_fold_id"]: row["file_sha256"]
        for row in plan["r4_adamw"]["checkpoints"]
    }
    contract_shas: dict[int, str] = {}
    contract_holdout_groups: dict[int, set[str]] = {}
    for owner, contract in enumerate(contracts):
        if not isinstance(contract, Mapping):
            raise RuntimeError("R5 success fold contract is invalid")
        unsigned = dict(contract)
        recorded = unsigned.pop("calibration_contract_sha256", None)
        if (
            contract.get("owner_fold_id") != owner
            or recorded != canonical_sha256(unsigned)
            or contract.get("final_outer_checkpoint_sha256")
            != checkpoint_shas[owner]
            or contract.get("outer_holdout_labels_used_for_alpha_selection") is not False
            or contract.get("fresh50_inputs_or_labels_used") is not False
        ):
            raise RuntimeError("R5 success signed fold contract changed")
        contract_shas[owner] = recorded
        holdout_groups = list(map(str, contract.get("outer_holdout_groups", ())))
        expected_holdout_groups = sorted(
            map(
                str,
                plan["materialization"]["folds"][owner].get(
                    "oof_holdout_groups", ()
                ),
            )
        )
        if (
            holdout_groups != expected_holdout_groups
            or len(holdout_groups) != len(set(holdout_groups))
            or contract.get("outer_holdout_groups_sha256")
            != plan["materialization"]["folds"][owner].get(
                "oof_holdout_groups_sha256"
            )
        ):
            raise RuntimeError("R5 success holdout ownership changed")
        contract_holdout_groups[owner] = set(holdout_groups)
    observed: dict[tuple[int, str], set[int]] = {}
    candidate_names = (
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("R5 success OOF row is invalid")
        owner = row.get("owner_fold_id")
        group = str(row.get("logical_group", ""))
        candidate = row.get("candidate_index")
        if (
            owner not in range(FOLD_COUNT)
            or not group
            or candidate not in range(4)
            or row.get("candidate_name") != candidate_names[candidate]
            or group not in contract_holdout_groups[owner]
            or row.get("calibration_contract_sha256") != contract_shas[owner]
            or row.get("success_label") not in (0, 1)
            or not all(
                isinstance(row.get(key), (int, float))
                and math_isfinite(float(row[key]))
                and 0.0 < float(row[key]) < 1.0
                for key in (
                    "uncalibrated_success_probability",
                    "calibrated_success_probability",
                    "owner_training_prevalence_baseline",
                )
            )
        ):
            raise RuntimeError("R5 success OOF row alignment changed")
        observed.setdefault((owner, group), set()).add(candidate)
    if len(observed) != 250 or any(value != set(range(4)) for value in observed.values()):
        raise RuntimeError("R5 success rows are not 250 groups x four candidates")
    implementation = next(
        row
        for key, row in plan["implementation_files"].items()
        if Path(key).name == "calibrate_openvla_etsf_v8_success_inner_cv.py"
    )
    if (
        value.get("implementation") != implementation["path"]
        or value.get("implementation_sha256") != implementation["sha256"]
    ):
        raise RuntimeError("R5 success output implementation binding changed")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "result_sha256": value["result_sha256"],
        "calibrated_oof_rows_sha256": value["calibrated_oof_rows_sha256"],
        "rows": len(rows),
        "logical_groups": len(observed),
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
    }


def math_isfinite(value: float) -> bool:
    return not (value != value or value in (float("inf"), float("-inf")))


def _validate_duration_output(
    output_dir: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    output_dir = _reject_path(output_dir, role="R5 duration output")
    result_path = output_dir / "duration_hierarchy_evaluation.json"
    value = _load_json(result_path, role="R5 duration JSON")
    _verify_signed(value, "result_sha256", role="R5 duration JSON")
    arrays = value.get("row_arrays")
    source = value.get("source_materialization")
    reported_implementations = value.get("implementation_files")
    expected_implementations = {
        Path(key).name: record["sha256"]
        for key, record in plan["implementation_files"].items()
        if Path(key).name
        in {
            "evaluate_openvla_etsf_duration_hierarchy_oof.py",
            "openvla_etsf_duration_hierarchy.py",
            "openvla_etsf_v8_structured_adapters.py",
            "train_openvla_etsf_v8_structured_adapters.py",
        }
    }
    if (
        value.get("format") != DURATION_FORMAT
        or value.get("status") not in ("passed", "fail_closed")
        or value.get("fresh50_inputs_accepted") is not False
        or value.get("fresh50_labels_read") is not False
        or value.get("fresh50_confirmation_authorized") is not False
        or value.get("selector_authorized") is not False
        or value.get("prospective_claim_allowed") is not False
        or not isinstance(source, Mapping)
        or source.get("path") != plan["materialization"]["path"]
        or source.get("materialization_sha256")
        != plan["materialization"]["materialization_sha256"]
        or source.get("file_sha256")
        != plan["materialization"]["file_sha256"]
        or source.get("ten_artifacts_authenticated") is not True
        or source.get("source_hdf5_read") is not False
        or reported_implementations != expected_implementations
        or not isinstance(arrays, Mapping)
    ):
        raise RuntimeError("R5 duration JSON contract failed")
    arrays_path = _reject_path(Path(str(arrays.get("path", ""))), role="R5 duration NPZ")
    if (
        arrays_path != (output_dir / "duration_hierarchy_rows.npz").resolve()
        or not arrays_path.is_file()
        or sha256_path(arrays_path) != arrays.get("file_sha256")
    ):
        raise RuntimeError("R5 duration NPZ hash/path changed")
    with np.load(arrays_path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        lengths = {len(payload[key]) for key in keys}
    if (
        not keys
        or keys != arrays.get("keys")
        or len(lengths) != 1
        or next(iter(lengths)) != arrays.get("rows")
        or arrays.get("alignment") != "owner_fold_id_logical_group_row_index"
    ):
        raise RuntimeError("R5 duration NPZ alignment changed")
    return {
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "result_file_sha256": sha256_path(result_path),
        "result_sha256": value["result_sha256"],
        "npz_path": str(arrays_path),
        "npz_file_sha256": arrays["file_sha256"],
        "rows": arrays["rows"],
        "evaluation_status": value["status"],
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
    }


def execute(plan_path: Path, *, poll_seconds: float) -> dict[str, Any]:
    plan_path = _reject_path(plan_path, role="R5 plan")
    plan = _load_json(plan_path, role="R5 plan")
    _verify_signed(plan, "plan_sha256", role="R5 plan")
    if (
        plan.get("format") != FORMAT
        or plan.get("status") != "preregistered_no_execution"
        or plan.get("fresh50_inputs_accepted") is not False
        or plan.get("fresh50_labels_read") is not False
        or plan.get("selector_authorized") is not False
    ):
        raise RuntimeError("R5 plan status/scope changed")
    recomputed = _recompute_plan(plan)
    if recomputed.get("plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("R5 inputs or implementation changed after preregistration")
    output_root = Path(str(plan["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    logs.mkdir()
    _immutable_json(output_root / "launch_plan.json", plan)
    state_path = output_root / "launch_state.json"
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "running_r5_repair_evaluations",
        "plan_sha256": plan["plan_sha256"],
        "current_stage": None,
        "last_completed_stage": None,
        "commands": [],
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "selector_authorized": False,
    }
    _atomic_state_json(state_path, state)
    try:
        state["status"] = "waiting_for_gpu_idle"
        _atomic_state_json(state_path, state)
        gpu_audit = _wait_for_gpu_idle(
            state=state,
            state_path=state_path,
            gpu_index=int(plan["gpu_index"]),
            poll_seconds=poll_seconds,
        )
        state["status"] = "running_r5_repair_evaluations"
        _atomic_state_json(state_path, state)
        commands = list(plan["commands"])
        if (
            [row.get("stage") for row in commands]
            != ["success_calibration", "duration_hierarchy"]
            or commands[0].get("uses_gpu") is not True
            or commands[1].get("uses_gpu") is not False
        ):
            raise RuntimeError("R5 command order/device contract changed")
        _run_stage(
            command=commands[0],
            plan=plan,
            state=state,
            state_path=state_path,
            logs_dir=logs,
        )
        success_audit = _validate_success_output(
            output_root / "success_calibration.json", plan=plan
        )
        state["success_audit"] = success_audit
        _atomic_state_json(state_path, state)
        _run_stage(
            command=commands[1],
            plan=plan,
            state=state,
            state_path=state_path,
            logs_dir=logs,
        )
        duration_audit = _validate_duration_output(
            output_root / "duration_hierarchy", plan=plan
        )
        state["duration_audit"] = duration_audit
        summary = _signed(
            {
                "format": SUMMARY_FORMAT,
                "status": TERMINAL_STATUS,
                "plan_sha256": plan["plan_sha256"],
                "implementation_bundle_sha256": plan[
                    "implementation_bundle_sha256"
                ],
                "materialization_sha256": plan["materialization"][
                    "materialization_sha256"
                ],
                "r4_summary_sha256": plan["r4_adamw"]["summary_sha256"],
                "gpu_idle_audit": gpu_audit,
                "success": success_audit,
                "duration": duration_audit,
                "adaptive_development_only": True,
                "prospective_claim_allowed": False,
                "selector_authorized": False,
                "automatic_fresh_launch": False,
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
            },
            "summary_sha256",
        )
        _immutable_json(output_root / "launch_summary.json", summary)
        state.update(
            {
                "status": TERMINAL_STATUS,
                "summary": str(output_root / "launch_summary.json"),
                "summary_sha256": summary["summary_sha256"],
                "current_stage": None,
            }
        )
        _atomic_state_json(state_path, state)
        return summary
    except BaseException as error:
        state.update(
            {
                "status": FAILURE_STATUS,
                "error_type": type(error).__name__,
                "error": str(error),
                "current_stage": state.get("current_stage"),
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
                "selector_authorized": False,
            }
        )
        _atomic_state_json(state_path, state)
        raise


def detach(
    plan_path: Path,
    *,
    poll_seconds: float,
    nohup_log: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    plan_path = _reject_path(plan_path, role="R5 detached plan")
    nohup_log = _reject_path(nohup_log, role="R5 nohup log")
    receipt_path = _reject_path(receipt_path, role="R5 detach receipt")
    if nohup_log.exists() or receipt_path.exists():
        raise FileExistsError("R5 detached log/receipt must not exist")
    plan = _load_json(plan_path, role="R5 detached plan")
    _verify_signed(plan, "plan_sha256", role="R5 detached plan")
    recomputed = _recompute_plan(plan)
    if recomputed.get("plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("R5 inputs or implementation changed before detach")
    if Path(str(plan["output_root"])).exists():
        raise FileExistsError(plan["output_root"])
    launcher = next(
        Path(record["path"])
        for key, record in plan["implementation_files"].items()
        if Path(key).name == Path(__file__).name
    )
    argv = [
        "nohup",
        str(plan["python_bin"]),
        str(launcher),
        "execute",
        "--plan",
        str(plan_path),
        "--poll-seconds",
        str(poll_seconds),
    ]
    nohup_log.parent.mkdir(parents=True, exist_ok=True)
    with nohup_log.open("xb") as handle:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=Path(str(plan["code_root"])),
            start_new_session=True,
            close_fds=True,
        )
    receipt = _signed(
        {
            "format": DETACH_FORMAT,
            "status": "detached_nohup_started",
            "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "argv": argv,
            "argv_sha256": canonical_sha256(argv),
            "pid": int(process.pid),
            "nohup_log": str(nohup_log),
            "detachment": "nohup_plus_start_new_session_redirected_stdio",
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
        },
        "receipt_sha256",
    )
    _immutable_json(receipt_path, receipt)
    return receipt


def preregister(
    *,
    code_root: Path,
    materialization_manifest: Path,
    r4_summary: Path,
    output_root: Path,
    python_bin: Path,
    gpu_index: int,
    plan_output: Path,
) -> dict[str, Any]:
    plan_output = _reject_path(plan_output, role="R5 plan output")
    normalized_output_root = _reject_path(output_root, role="R5 output root")
    if plan_output == normalized_output_root or normalized_output_root in plan_output.parents:
        raise RuntimeError("R5 plan output must be outside the absent execution root")
    if plan_output.exists():
        raise FileExistsError(plan_output)
    plan = build_plan(
        code_root=code_root,
        materialization_manifest=materialization_manifest,
        r4_summary=r4_summary,
        output_root=output_root,
        python_bin=python_bin,
        gpu_index=gpu_index,
    )
    _immutable_json(plan_output, plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--code-root", type=Path, required=True)
    preregister.add_argument("--materialization-manifest", type=Path, required=True)
    preregister.add_argument("--r4-summary", type=Path, required=True)
    preregister.add_argument("--output-root", type=Path, required=True)
    preregister.add_argument("--python-bin", type=Path, required=True)
    preregister.add_argument("--gpu-index", type=int, default=0)
    preregister.add_argument("--plan-output", type=Path, required=True)
    run = subparsers.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    detached = subparsers.add_parser("detach")
    detached.add_argument("--plan", type=Path, required=True)
    detached.add_argument("--poll-seconds", type=float, default=30.0)
    detached.add_argument("--nohup-log", type=Path, required=True)
    detached.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preregister":
        plan = preregister(
            code_root=args.code_root,
            materialization_manifest=args.materialization_manifest,
            r4_summary=args.r4_summary,
            output_root=args.output_root,
            python_bin=args.python_bin,
            gpu_index=args.gpu_index,
            plan_output=args.plan_output,
        )
        print(
            json.dumps(
                {
                    "status": plan["status"],
                    "plan": str(args.plan_output.resolve()),
                    "plan_sha256": plan["plan_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if not 5.0 <= args.poll_seconds <= 60.0:
        raise ValueError("poll-seconds must be in [5,60]")
    if args.command == "execute":
        summary = execute(args.plan, poll_seconds=args.poll_seconds)
        print(json.dumps(summary, sort_keys=True))
        return
    receipt = detach(
        args.plan,
        poll_seconds=args.poll_seconds,
        nohup_log=args.nohup_log,
        receipt_path=args.receipt,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "build_plan",
    "canonical_sha256",
    "detach",
    "execute",
    "implementation_closure",
    "preregister",
    "sha256_path",
]
