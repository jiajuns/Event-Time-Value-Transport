#!/usr/bin/env python3
"""Run a frozen transfer plan only after the OpenVLA confirmation watcher ends.

The launcher reads only the terminal watcher state; it never opens the OpenVLA
confirmation seed manifest, HDF5 files, or result labels.  A plan consists of
content-addressed, argv-only stages.  GPU stages are serialized and wait for an
idle RTX 4090.  Target confirmation is unreachable until the target validation
stage publishes ``confirmation_authorized``.

Stage executables report through ``ETSF_TRANSFER_STAGE_RECEIPT``.  This keeps
the launcher independent of OpenVLA/SmolVLA runtimes while making every child
bind its output to the frozen plan, study id, and role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from launch_openvla_etsf_counterfactual_oof_v5 import (
    query_compute_pids,
    query_gpu_name,
)
from verify_etsf_transfer_protocol import (
    AUDIT_FORMAT,
    evaluate_transfer_results,
    json_sha256,
    validate_protocol,
)
from verify_etsf_transfer_asset_preflight import validate_preflight


PLAN_FORMAT = "etsf_post_openvla_transfer_plan_v1"
STATE_FORMAT = "etsf_post_openvla_transfer_pipeline_state_v1"
RECEIPT_FORMAT = "etsf_transfer_stage_receipt_v1"
UPSTREAM_FORMAT = "etsf_oof_guarded_fresh_watcher_v1"
UPSTREAM_READY = "complete_fresh50_confirmed"
UPSTREAM_NO_CONFIRMATION = "complete_upstream_guard_not_authorized_fresh_forbidden"

REQUIRED_ROLES = (
    "asset_preflight",
    "prepare_reserved_vocabulary",
    "source_retrain_with_reserved_row",
    "verify_source_retraining_provenance",
    "target_interface_smoke",
    "collect_adaptation",
    "collect_validation",
    "freeze_transfer_protocol",
    "train_actor_hidden_observer_n20",
    "evaluate_privileged_pose_upper_bound_n20",
    "train_transfer_n0",
    "train_transfer_n5",
    "train_transfer_n10",
    "train_transfer_n20",
    "train_transfer_n50",
    "train_target_from_scratch_n20",
    "train_no_factorization_n20",
    "train_full_finetune_upper_n20",
    "validate_and_freeze",
    "audit_primary_weights",
    "run_paired_confirmation",
    "build_transfer_result_summary",
)
GPU_ROLES = {
    "target_interface_smoke",
    "collect_adaptation",
    "collect_validation",
    "train_actor_hidden_observer_n20",
    "evaluate_privileged_pose_upper_bound_n20",
    "train_transfer_n0",
    "train_transfer_n5",
    "train_transfer_n10",
    "train_transfer_n20",
    "train_transfer_n50",
    "train_target_from_scratch_n20",
    "train_no_factorization_n20",
    "train_full_finetune_upper_n20",
    "validate_and_freeze",
    "run_paired_confirmation",
}
LABEL_FREE_ROLES = {
    "asset_preflight",
    "prepare_reserved_vocabulary",
    "verify_source_retraining_provenance",
    "target_interface_smoke",
    "freeze_transfer_protocol",
    "audit_primary_weights",
}
EXPECTED_STATUS = {
    "asset_preflight": "ready",
    "prepare_reserved_vocabulary": "complete",
    "source_retrain_with_reserved_row": "complete",
    "verify_source_retraining_provenance": "source_core_ready_for_protocol_freeze",
    "target_interface_smoke": "interface_verified",
    "collect_adaptation": "complete",
    "collect_validation": "complete",
    "freeze_transfer_protocol": "complete",
    "train_actor_hidden_observer_n20": "complete",
    "evaluate_privileged_pose_upper_bound_n20": "complete",
    "train_transfer_n0": "complete",
    "train_transfer_n5": "complete",
    "train_transfer_n10": "complete",
    "train_transfer_n20": "complete",
    "train_transfer_n50": "complete",
    "train_target_from_scratch_n20": "complete",
    "train_no_factorization_n20": "complete",
    "train_full_finetune_upper_n20": "complete",
    "audit_primary_weights": "authorized",
    "run_paired_confirmation": "complete",
    "build_transfer_result_summary": "complete",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields-set(value))}, "
            f"extra={sorted(set(value)-fields)}"
        )


def plan_sha256(plan: Mapping[str, Any]) -> str:
    return json_sha256(plan)


def validate_plan(plan: Mapping[str, Any]) -> None:
    _exact(
        plan,
        {
            "format",
            "study_id",
            "axis",
            "upstream",
            "protocol_output",
            "gpu",
            "stages",
            "fresh_label_access",
        },
        "plan",
    )
    if plan["format"] != PLAN_FORMAT or plan["axis"] not in ("policy", "embodiment"):
        raise ValueError("transfer plan format/axis is invalid")
    if not isinstance(plan["study_id"], str) or not plan["study_id"]:
        raise ValueError("transfer plan study_id must be non-empty")
    if plan["fresh_label_access"] != "terminal_watcher_state_only":
        raise ValueError("transfer plan may not access OpenVLA confirmation labels")
    upstream = plan["upstream"]
    if not isinstance(upstream, Mapping):
        raise ValueError("upstream must be a mapping")
    _exact(
        upstream,
        {"state_path", "format", "required_status", "forbidden_status"},
        "upstream",
    )
    if (
        upstream["format"] != UPSTREAM_FORMAT
        or upstream["required_status"] != UPSTREAM_READY
        or upstream["forbidden_status"] != UPSTREAM_NO_CONFIRMATION
        or not Path(str(upstream["state_path"])).is_absolute()
    ):
        raise ValueError("upstream watcher contract is not frozen")
    protocol_output = Path(str(plan["protocol_output"]))
    if not protocol_output.is_absolute():
        raise ValueError("protocol_output must be absolute")
    gpu = plan["gpu"]
    if not isinstance(gpu, Mapping):
        raise ValueError("gpu must be a mapping")
    _exact(gpu, {"index", "required_name", "wait_timeout_seconds", "poll_seconds"}, "gpu")
    if (
        int(gpu["index"]) < 0
        or "4090" not in str(gpu["required_name"])
        or float(gpu["wait_timeout_seconds"]) <= 0
        or not 0 < float(gpu["poll_seconds"]) <= 60
    ):
        raise ValueError("GPU serialization contract is invalid")
    stages = plan["stages"]
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise ValueError("stages must be a sequence")
    roles = tuple(str(stage.get("role")) for stage in stages if isinstance(stage, Mapping))
    if roles != REQUIRED_ROLES:
        raise ValueError("transfer stages are missing, duplicated, or reordered")
    receipts: set[str] = set()
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("stage must be a mapping")
        _exact(stage, {"role", "argv", "gpu", "command_artifacts"}, "stage")
        role = str(stage["role"])
        if bool(stage["gpu"]) != (role in GPU_ROLES):
            raise ValueError(f"stage {role} has the wrong GPU declaration")
        argv = stage["argv"]
        if (
            not isinstance(argv, Sequence)
            or isinstance(argv, (str, bytes))
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
            or not Path(argv[0]).is_absolute()
        ):
            raise ValueError(f"stage {role} argv must be non-empty and absolute")
        artifacts = stage["command_artifacts"]
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)) or not artifacts:
            raise ValueError(f"stage {role} must bind command artifacts")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError("command artifact must be a mapping")
            _exact(artifact, {"path", "sha256"}, "command artifact")
            path = Path(str(artifact["path"]))
            if not path.is_absolute() or len(str(artifact["sha256"])) != 64:
                raise ValueError("command artifact path/hash is invalid")
        n_suffix = role.removeprefix("train_transfer_n")
        if role.startswith("train_transfer_n"):
            expected_n = int(n_suffix)
            if "--n-per-task" not in argv or str(expected_n) not in argv:
                raise ValueError(f"stage {role} does not freeze its N")
        if role in {
            "train_actor_hidden_observer_n20",
            "evaluate_privileged_pose_upper_bound_n20",
            "train_target_from_scratch_n20",
            "train_no_factorization_n20",
            "train_full_finetune_upper_n20",
        } and ("--n-per-task" not in argv or "20" not in argv):
            raise ValueError(f"matched baseline {role} must use N=20")
        if role in receipts:
            raise AssertionError("duplicate role")
        receipts.add(role)


def verify_command_artifacts(stage: Mapping[str, Any]) -> None:
    for artifact in stage["command_artifacts"]:
        path = Path(str(artifact["path"])).expanduser().resolve()
        if not path.is_file() or file_sha256(path) != str(artifact["sha256"]):
            raise RuntimeError(f"command artifact changed or is missing: {path}")


def wait_for_upstream(path: Path, timeout: float, poll: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if path.is_file():
            try:
                value = _json(path)
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict):
                if value.get("format") != UPSTREAM_FORMAT:
                    raise RuntimeError("upstream watcher format changed")
                status = str(value.get("status", ""))
                if status in (UPSTREAM_READY, UPSTREAM_NO_CONFIRMATION):
                    return value
                if status == "failed_closed":
                    raise RuntimeError(f"upstream watcher failed: {value.get('error')}")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for OpenVLA confirmation watcher")
        time.sleep(poll)


def wait_for_idle_4090(index: int, timeout: float, poll: float, required: str) -> dict[str, Any]:
    name = query_gpu_name(index)
    if required not in name:
        raise RuntimeError(f"transfer pipeline requires {required}, found {name!r}")
    deadline = time.monotonic() + timeout
    while True:
        pids = query_compute_pids(index)
        if not pids:
            return {"index": index, "name": name, "compute_pids": []}
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for idle RTX 4090; PIDs={pids}")
        time.sleep(poll)


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    digest: str,
    role: str,
) -> None:
    _exact(
        receipt,
        {
            "format",
            "study_id",
            "plan_sha256",
            "role",
            "status",
            "artifact_path",
            "artifact_sha256",
            "labels_read",
        },
        f"receipt {role}",
    )
    if (
        receipt["format"] != RECEIPT_FORMAT
        or receipt["study_id"] != plan["study_id"]
        or receipt["plan_sha256"] != digest
        or receipt["role"] != role
    ):
        raise RuntimeError(f"stage receipt identity mismatch: {role}")
    if role in LABEL_FREE_ROLES and receipt["labels_read"] is not False:
        raise RuntimeError(f"label-free stage reported label access: {role}")
    artifact = Path(str(receipt["artifact_path"])).expanduser().resolve()
    if not artifact.is_file() or file_sha256(artifact) != receipt["artifact_sha256"]:
        raise RuntimeError(f"stage artifact is missing or changed: {role}")


def run_stage(
    stage: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    digest: str,
    state_root: Path,
    gpu_waiter: Callable[[int, float, float, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    role = str(stage["role"])
    verify_command_artifacts(stage)
    gpu_audit: dict[str, Any] | None = None
    if stage["gpu"]:
        gpu = plan["gpu"]
        gpu_audit = dict(
            gpu_waiter(
                int(gpu["index"]),
                float(gpu["wait_timeout_seconds"]),
                float(gpu["poll_seconds"]),
                str(gpu["required_name"]),
            )
        )
    receipt_path = state_root / "receipts" / f"{role}.json"
    log_path = state_root / "logs" / f"{role}.log"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError(f"stage state already exists: {role}")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
            "ETSF_TRANSFER_PLAN_SHA256": digest,
            "ETSF_TRANSFER_STUDY_ID": str(plan["study_id"]),
            "ETSF_TRANSFER_STAGE_ROLE": role,
            "ETSF_TRANSFER_STAGE_RECEIPT": str(receipt_path),
        }
    )
    with log_path.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(stage["argv"]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"transfer stage {role} exited {completed.returncode}; see {log_path}")
    if not receipt_path.is_file():
        raise RuntimeError(f"transfer stage did not publish an atomic receipt: {role}")
    receipt = _json(receipt_path)
    _validate_receipt(receipt, plan=plan, digest=digest, role=role)
    if role == "validate_and_freeze":
        if receipt["status"] not in ("confirmation_authorized", "confirmation_forbidden"):
            raise RuntimeError("validation stage did not publish a frozen promotion decision")
    elif receipt["status"] != EXPECTED_STATUS[role]:
        raise RuntimeError(f"stage {role} returned unexpected status {receipt['status']!r}")
    record = {
        "role": role,
        "gpu": bool(stage["gpu"]),
        "gpu_idle_audit": gpu_audit,
        "receipt": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
        "artifact_path": receipt["artifact_path"],
        "artifact_sha256": receipt["artifact_sha256"],
        "labels_read": receipt["labels_read"],
        "status": receipt["status"],
        "log": str(log_path),
        "log_sha256": file_sha256(log_path),
    }
    return record, receipt


def execute(
    plan: Mapping[str, Any],
    *,
    state_root: Path,
    upstream_timeout: float,
    upstream_poll: float,
    gpu_waiter: Callable[[int, float, float, str], Mapping[str, Any]] = wait_for_idle_4090,
) -> dict[str, Any]:
    validate_plan(plan)
    state_root = state_root.expanduser().absolute()
    if state_root.exists():
        raise FileExistsError(state_root)
    if not 0 < upstream_poll <= 60 or upstream_timeout <= 0:
        raise ValueError("invalid upstream timeout/poll")
    state_root.mkdir(parents=True, exist_ok=False)
    digest = plan_sha256(plan)
    state_path = state_root / "pipeline_state.json"
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "waiting_for_openvla_confirmation_terminal_state",
        "study_id": plan["study_id"],
        "plan_sha256": digest,
        "openvla_confirmation_labels_read": False,
        "stages": [],
    }
    atomic_json(state_path, state)
    try:
        upstream_path = Path(str(plan["upstream"]["state_path"]))
        upstream = wait_for_upstream(upstream_path, upstream_timeout, upstream_poll)
        state["upstream_status"] = upstream["status"]
        # Deliberately record only the terminal state file identity.  Paths to
        # upstream confirmation artifacts are neither opened nor copied.
        state["upstream_state_sha256"] = file_sha256(upstream_path)
        if upstream["status"] == UPSTREAM_NO_CONFIRMATION:
            state["status"] = "complete_openvla_confirmation_not_available_transfer_not_started"
            atomic_json(state_path, state)
            return state
        state["status"] = "running_transfer_preconfirmation_stages"
        atomic_json(state_path, state)
        protocol: dict[str, Any] | None = None
        audit: dict[str, Any] | None = None
        for stage in plan["stages"]:
            role = str(stage["role"])
            state["current_stage"] = role
            atomic_json(state_path, state)
            record, receipt = run_stage(
                stage,
                plan=plan,
                digest=digest,
                state_root=state_root,
                gpu_waiter=gpu_waiter,
            )
            state["stages"].append(record)
            if role == "asset_preflight":
                assert receipt is not None
                preflight = _json(Path(str(receipt["artifact_path"])))
                validate_preflight(preflight)
                if (
                    preflight["study_id"] != plan["study_id"]
                    or preflight["axis"] != plan["axis"]
                ):
                    raise RuntimeError("asset preflight identity differs from the plan")
                state["asset_preflight_sha256"] = json_sha256(preflight)
            elif role == "freeze_transfer_protocol":
                assert receipt is not None
                protocol_path = Path(str(receipt["artifact_path"])).resolve()
                if protocol_path != Path(str(plan["protocol_output"])).resolve():
                    raise RuntimeError("protocol stage wrote an unexpected path")
                protocol = _json(protocol_path)
                validate_protocol(protocol)
                if protocol["study_id"] != plan["study_id"] or protocol["axis"] != plan["axis"]:
                    raise RuntimeError("frozen protocol identity differs from the pipeline plan")
                state["protocol_sha256"] = json_sha256(protocol)
            elif role == "verify_source_retraining_provenance":
                proof = _json(Path(str(receipt["artifact_path"])))
                if (
                    proof.get("format") != "etsf_reserved_source_core_retraining_v1"
                    or proof.get("status") != "source_core_ready_for_protocol_freeze"
                    or proof.get("ready_for_protocol_freeze") is not True
                    or proof.get("target_data_read") is not False
                    or proof.get("target_labels_read") is not False
                    or proof.get("reserved_target_row_unchanged") is not True
                    or proof.get("source_parameters_changed") is not True
                ):
                    raise RuntimeError("reserved-row source retraining is not proven")
            elif role == "validate_and_freeze" and receipt["status"] == "confirmation_forbidden":
                state["status"] = "complete_target_validation_gate_forbidden_confirmation_not_run"
                state.pop("current_stage", None)
                atomic_json(state_path, state)
                return state
            elif role == "audit_primary_weights":
                if protocol is None:
                    raise AssertionError("weight audit ran before protocol freeze")
                audit = _json(Path(str(receipt["artifact_path"])))
                if (
                    audit.get("format") != AUDIT_FORMAT
                    or audit.get("authorized") is not True
                    or audit.get("protocol_sha256") != json_sha256(protocol)
                ):
                    raise RuntimeError("primary-N weight audit is not authorized")
            elif role == "build_transfer_result_summary":
                if protocol is None or audit is None:
                    raise AssertionError("result summary ran before protocol/audit")
                results = _json(Path(str(receipt["artifact_path"])))
                decision = evaluate_transfer_results(protocol, audit, results)
                decision_path = state_root / "transfer_acceptance_decision.json"
                atomic_json(decision_path, decision)
                state["acceptance_decision"] = str(decision_path)
                state["acceptance_decision_sha256"] = file_sha256(decision_path)
                state["action_ranking_authorized"] = decision["action_ranking_authorized"]
            atomic_json(state_path, state)
        state.pop("current_stage", None)
        state["status"] = (
            "complete_transfer_confirmed_authorized"
            if state.get("action_ranking_authorized") is True
            else "complete_transfer_confirmation_inconclusive_monitor_only"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--upstream-timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--upstream-poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = _json(args.plan)
    validate_plan(plan)
    if args.validate_only:
        for stage in plan["stages"]:
            verify_command_artifacts(stage)
        print(json.dumps({"status": "valid", "plan_sha256": plan_sha256(plan)}))
        return
    result = execute(
        plan,
        state_root=args.state_root,
        upstream_timeout=args.upstream_timeout_seconds,
        upstream_poll=args.upstream_poll_seconds,
    )
    print(json.dumps({"status": result["status"], "plan_sha256": plan_sha256(plan)}))


if __name__ == "__main__":
    main()
