#!/usr/bin/env python3
"""Signed, immutable, detached launcher for the R6/v9 D250 evaluation.

The launcher deliberately has two phases.  ``preregister`` authenticates the
complete materialized R3 D250 bundle, the R4 AdamW lineage bundle, the Python
interpreter and the recursive implementation closure, then writes a signed
plan outside the still-absent output directory.  ``execute`` recomputes that
plan before creating the output directory and again immediately before the
single GPU stage.  ``detach`` starts ``execute`` in a new session and writes a
signed receipt, so progress can be recovered from server-side state and logs
after the client disconnects.

R4 checkpoints are authenticated lineage evidence only: the v9 evaluator fits
new group-relative heads directly from the materialized outer-training folds,
so the checkpoint paths are intentionally not passed on its CLI.  No path
containing Fresh or confirmation is accepted and all signed inputs must attest
that no Fresh data or labels were read.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from launch_openvla_etsf_r5_repair_evaluations import (
    _authenticate_materialization,
    _authenticate_r4_adamw,
    _immutable_json,
    _load_json,
    _python_contract,
    _reject_path,
    _signed,
    _verify_signed,
    canonical_sha256,
    sha256_path,
)


FORMAT = "etsf_openvla_v9_group_relative_success_r6_plan_v1"
STATE_FORMAT = "etsf_openvla_v9_group_relative_success_r6_state_v1"
SUMMARY_FORMAT = "etsf_openvla_v9_group_relative_success_r6_summary_v1"
DETACH_FORMAT = "etsf_openvla_v9_group_relative_success_r6_detach_receipt_v1"
TERMINAL_STATUS = "complete_r6_v9_group_relative_adaptive_d250_no_fresh"
FAILURE_STATUS = "failed_closed_r6_v9_group_relative_adaptive_d250_no_fresh"
RESULT_FORMAT = "etsf_v9_group_relative_success_ranking_nested_oof_v1"
FOLD_CONTRACT_FORMAT = "etsf_v9_group_relative_success_ranking_outer_fold_v1"
FOLD_COUNT = 5
EXPECTED_GROUPS = 250
EXPECTED_ROWS = 1_000
ENTRYPOINTS = (
    "launch_openvla_etsf_v9_group_relative_success_r6.py",
    "evaluate_openvla_etsf_v9_group_relative_success_oof.py",
)
RESULT_IMPLEMENTATIONS = {
    "evaluate_openvla_etsf_v9_group_relative_success_oof.py",
    "openvla_etsf_v9_group_relative_success_adapter.py",
    "calibrate_openvla_etsf_v8_success_inner_cv.py",
    "openvla_etsf_v8_structured_adapters.py",
    "train_openvla_etsf_v8_structured_adapters.py",
}
CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)


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


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create once, fsync, and make read-only; state files use the atomic writer."""

    _immutable_json(path, value)
    path.chmod(0o444)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    """Hash every scripts-local Python module reachable from both entrypoints."""

    code_root = _reject_path(code_root, role="R6 immutable code root")
    scripts = (code_root / "scripts").resolve()
    queue = [scripts / name for name in ENTRYPOINTS]
    seen: set[Path] = set()
    while queue:
        path = queue.pop().resolve()
        if path in seen:
            continue
        if not path.is_file() or scripts not in path.parents:
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
    names = {Path(relative).name for relative in records}
    if not set(ENTRYPOINTS).issubset(names) or not RESULT_IMPLEMENTATIONS.issubset(
        names
    ):
        raise RuntimeError("R6 implementation closure is incomplete")
    return records


def _result_implementation_contract(
    implementations: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    result = {
        Path(relative).name: str(record["sha256"])
        for relative, record in implementations.items()
        if Path(relative).name in RESULT_IMPLEMENTATIONS
    }
    if set(result) != RESULT_IMPLEMENTATIONS:
        raise RuntimeError("R6 result implementation contract is incomplete")
    return result


def _build_command(
    *, python_bin: Path, code_root: Path, output_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    argv = [
        str(python_bin),
        str(code_root / "scripts" / "evaluate_openvla_etsf_v9_group_relative_success_oof.py"),
        "--materialization-manifest",
        str(manifest["path"]),
        "--output",
        str(output_root / "group_relative_success_oof.json"),
        "--device",
        "cuda",
    ]
    return {
        "stage": "v9_group_relative_success_nested_oof",
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "uses_gpu": True,
        "device_contract": "CUDA_visible_single_preregistered_RTX4090_gpu",
        "r4_checkpoints_passed_on_cli": False,
        "r4_role": "authenticated_post_R5_lineage_only_not_model_initialization",
    }


def build_plan(
    *,
    code_root: Path,
    materialization_manifest: Path,
    r4_summary: Path,
    output_root: Path,
    python_bin: Path,
    gpu_index: int,
) -> dict[str, Any]:
    code_root = _reject_path(code_root, role="R6 immutable code root")
    output_root = _reject_path(output_root, role="R6 immutable output root")
    if not code_root.is_dir():
        raise FileNotFoundError(code_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    if gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    python_contract = _python_contract(python_bin)
    python_bin = Path(python_contract["invocation_path"])
    implementations = implementation_closure(code_root)
    materialization = _authenticate_materialization(materialization_manifest)
    r4 = _authenticate_r4_adamw(r4_summary, materialization=materialization)
    command = _build_command(
        python_bin=python_bin,
        code_root=code_root,
        output_root=output_root,
        manifest=materialization,
    )
    plan: dict[str, Any] = {
        "format": FORMAT,
        "status": "preregistered_no_execution",
        "code_root": str(code_root),
        "code_root_immutable_after_preregistration": True,
        "implementation_files": implementations,
        "implementation_bundle_sha256": canonical_sha256(implementations),
        "result_implementation_files": _result_implementation_contract(
            implementations
        ),
        "materialization": materialization,
        "materialization_scope": "R3_materialized_OOF_D250_only",
        "r4_adamw_lineage": r4,
        "r4_checkpoint_count": FOLD_COUNT,
        "r4_checkpoints_are_lineage_only": True,
        "r4_checkpoints_are_evaluator_cli_inputs": False,
        "output_root": str(output_root),
        "output_root_must_not_exist_before_execute": True,
        "python_bin": str(python_bin),
        "python_file_sha256": python_contract["invocation_file_sha256"],
        "python_contract": python_contract,
        "gpu_index": int(gpu_index),
        "commands": [command],
        "execution_order": [command["stage"]],
        "adaptive_development_only": True,
        "task_success_claim_authorized": False,
        "selector_deployment_authorized": False,
        "automatic_fresh_launch": False,
        "fresh_paths_accepted": False,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "terminal_status": TERMINAL_STATUS,
    }
    return _signed(plan, "plan_sha256")


def _recompute_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    materialization = plan.get("materialization")
    r4 = plan.get("r4_adamw_lineage")
    if not isinstance(materialization, Mapping) or not isinstance(r4, Mapping):
        raise RuntimeError("R6 plan provenance is incomplete")
    return build_plan(
        code_root=Path(str(plan.get("code_root", ""))),
        materialization_manifest=Path(str(materialization.get("path", ""))),
        r4_summary=Path(str(r4.get("summary", ""))),
        output_root=Path(str(plan.get("output_root", ""))),
        python_bin=Path(str(plan.get("python_bin", ""))),
        gpu_index=int(plan.get("gpu_index", -1)),
    )


def _validate_runtime_bindings(plan: Mapping[str, Any]) -> None:
    actual = implementation_closure(Path(str(plan["code_root"])))
    if (
        actual != plan.get("implementation_files")
        or canonical_sha256(actual) != plan.get("implementation_bundle_sha256")
        or _result_implementation_contract(actual)
        != plan.get("result_implementation_files")
    ):
        raise RuntimeError("R6 implementation changed after preregistration")
    if _python_contract(Path(str(plan["python_bin"]))) != plan.get("python_contract"):
        raise RuntimeError("R6 Python invocation/target changed after preregistration")
    materialization = _authenticate_materialization(
        Path(str(plan["materialization"]["path"]))
    )
    if materialization != plan.get("materialization"):
        raise RuntimeError("R3 D250 materialization changed after preregistration")
    r4 = _authenticate_r4_adamw(
        Path(str(plan["r4_adamw_lineage"]["summary"])),
        materialization=materialization,
    )
    if r4 != plan.get("r4_adamw_lineage"):
        raise RuntimeError("R4 lineage bundle changed after preregistration")


def _gpu_compute_pids(gpu_index: int) -> list[int]:
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
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("R6 nvidia-smi idle query timed out; fail closed") from error
    result: list[int] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            result.append(int(value))
        elif value and value.lower() not in (
            "no running processes found",
            "no running processes found.",
        ):
            raise RuntimeError(f"unexpected nvidia-smi compute PID output: {value}")
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


def _deep_validate_fold_contract(contract: Mapping[str, Any]) -> None:
    # Import only during result authentication.  The imported implementation is
    # itself part of the signed recursive code closure.
    from evaluate_openvla_etsf_v9_group_relative_success_oof import (
        validate_fold_contract,
    )

    validate_fold_contract(contract)


def _validate_result(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(path, role="R6 v9 OOF result")
    _verify_signed(value, "result_sha256", role="R6 v9 OOF result")
    contracts = value.get("fold_contracts")
    rows = value.get("oof_rows")
    trace = value.get("read_trace")
    if (
        value.get("format") != RESULT_FORMAT
        or value.get("status")
        not in ("passed_adaptive_development_only", "fail_closed_adaptive_development_only")
        or value.get("implementation_files")
        != plan.get("result_implementation_files")
        or value.get("materialization_manifest") != plan["materialization"]["path"]
        or value.get("materialization_file_sha256")
        != plan["materialization"]["file_sha256"]
        or value.get("materialization_sha256")
        != plan["materialization"]["materialization_sha256"]
        or value.get("all_outer_contracts_selected_before_any_outer_holdout_deserialized")
        is not True
        or value.get("probability_output_used_for_action_selection") is not False
        or value.get("task_success_improvement_claim_authorized") is not False
        or value.get("selector_deployment_authorized") is not False
        or value.get("fresh_confirmation_authorized") is not False
        or value.get("fresh_inputs_accepted") is not False
        or value.get("fresh_labels_read") is not False
        or not isinstance(contracts, list)
        or len(contracts) != FOLD_COUNT
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_ROWS
        or value.get("oof_row_count") != EXPECTED_ROWS
        or value.get("oof_rows_sha256") != canonical_sha256(rows)
        or not isinstance(trace, list)
        or value.get("read_trace_sha256") != canonical_sha256(trace)
    ):
        raise RuntimeError("R6 v9 output top-level contract failed")

    contract_sha: dict[int, str] = {}
    holdout_by_owner: dict[int, set[str]] = {}
    for owner, contract in enumerate(contracts):
        if not isinstance(contract, Mapping):
            raise RuntimeError("R6 v9 fold contract is invalid")
        _deep_validate_fold_contract(contract)
        unsigned = dict(contract)
        recorded = unsigned.pop("fold_contract_sha256", None)
        fold = plan["materialization"]["folds"][owner]
        training = list(map(str, contract.get("outer_training_groups", ())))
        holdout = list(map(str, contract.get("outer_holdout_groups", ())))
        if (
            recorded != canonical_sha256(unsigned)
            or contract.get("format") != FOLD_CONTRACT_FORMAT
            or contract.get("owner_fold_id") != owner
            or contract.get("materialization_sha256")
            != plan["materialization"]["materialization_sha256"]
            or contract.get("train_artifact_sha256")
            != fold["train_artifact_sha256"]
            or contract.get("train_payload_sha256") != fold["train_payload_sha256"]
            or training != fold["training_groups"]
            or holdout != fold["oof_holdout_groups"]
            or contract.get("outer_training_groups_sha256")
            != fold["training_groups_sha256"]
            or contract.get("outer_holdout_groups_sha256")
            != fold["oof_holdout_groups_sha256"]
            or contract.get("all_hyperparameters_selected_before_outer_holdout_payload_loaded")
            is not True
            or contract.get("outer_holdout_labels_used_for_model_or_hyperparameter_fit")
            is not False
            or contract.get("fresh_inputs_or_labels_used") is not False
        ):
            raise RuntimeError("R6 v9 signed fold contract changed")
        contract_sha[owner] = str(recorded)
        holdout_by_owner[owner] = set(holdout)

    observed: dict[tuple[int, str], set[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("R6 v9 OOF row is invalid")
        owner = row.get("owner_fold_id")
        group = str(row.get("logical_group", ""))
        candidate = row.get("candidate_index")
        probability = row.get("success_probability")
        ranking = row.get("candidate_ranking_score")
        if (
            owner not in range(FOLD_COUNT)
            or not group
            or group not in holdout_by_owner[owner]
            or candidate not in range(4)
            or row.get("candidate_name") != CANDIDATE_NAMES[candidate]
            or row.get("success_label") not in (0, 1)
            or not isinstance(probability, (int, float))
            or not 0.0 < float(probability) < 1.0
            or not isinstance(ranking, (int, float))
            or not float("-inf") < float(ranking) < float("inf")
            or row.get("fold_contract_sha256") != contract_sha[owner]
        ):
            raise RuntimeError("R6 v9 OOF row alignment changed")
        key = (int(owner), group)
        candidates = observed.setdefault(key, set())
        if candidate in candidates:
            raise RuntimeError("R6 v9 OOF row duplicates a candidate")
        candidates.add(int(candidate))
    if (
        len(observed) != EXPECTED_GROUPS
        or any(candidates != set(range(4)) for candidates in observed.values())
        or set(group for _, group in observed)
        != set().union(*holdout_by_owner.values())
    ):
        raise RuntimeError("R6 v9 output is not 250 owner groups x four candidates")
    expected_roles = ["train"] * FOLD_COUNT + ["holdout"] * FOLD_COUNT
    expected_owners = list(range(FOLD_COUNT)) + list(range(FOLD_COUNT))
    if (
        len(trace) != 2 * FOLD_COUNT
        or not all(isinstance(row, Mapping) for row in trace)
        or [row.get("role") for row in trace] != expected_roles
        or [row.get("owner_fold_id") for row in trace] != expected_owners
    ):
        raise RuntimeError("R6 v9 read order no longer freezes all contracts first")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "result_sha256": value["result_sha256"],
        "evaluation_status": value["status"],
        "oof_rows_sha256": value["oof_rows_sha256"],
        "rows": len(rows),
        "logical_groups": len(observed),
        "strict_development_adequacy": bool(
            value.get("outer_holdout_evaluation", {})
            .get("pooled_oof", {})
            .get("strict_development_adequacy", False)
        ),
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
    }


def _run_stage(
    *, plan: Mapping[str, Any], state: dict[str, Any], state_path: Path, logs: Path
) -> None:
    command = plan["commands"][0]
    if (
        command.get("stage") != "v9_group_relative_success_nested_oof"
        or command.get("uses_gpu") is not True
        or command.get("argv_sha256") != canonical_sha256(command.get("argv"))
        or command.get("r4_checkpoints_passed_on_cli") is not False
    ):
        raise RuntimeError("R6 command contract changed")
    _validate_runtime_bindings(plan)
    log = logs / "v9_group_relative_success_nested_oof.log"
    partial = logs / ".v9_group_relative_success_nested_oof.log.partial"
    if log.exists() or partial.exists():
        raise FileExistsError("R6 immutable stage log already exists")
    state.update(
        {
            "status": "running_v9_group_relative_success_nested_oof",
            "current_stage": command["stage"],
            "command": dict(command),
            "stage_log": str(log),
            "last_heartbeat_unix": time.time(),
        }
    )
    _atomic_state_json(state_path, state)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path(str(plan["code_root"])) / "scripts"),
            "OMP_NUM_THREADS": "4",
            "CUDA_VISIBLE_DEVICES": str(plan["gpu_index"]),
        }
    )
    try:
        with partial.open("xb") as handle:
            completed = subprocess.run(
                list(command["argv"]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=Path(str(plan["code_root"])),
                env=environment,
            )
        os.replace(partial, log)
    except BaseException:
        if partial.exists() and not log.exists():
            os.replace(partial, log)
        raise
    if completed.returncode != 0:
        raise RuntimeError(f"R6 evaluation failed; see {log}")
    log.chmod(0o444)
    state.update(
        {
            "current_stage": None,
            "last_completed_stage": command["stage"],
            "stage_log_sha256": sha256_path(log),
            "last_heartbeat_unix": time.time(),
        }
    )
    _atomic_state_json(state_path, state)


def execute(plan_path: Path, *, poll_seconds: float) -> dict[str, Any]:
    plan_path = _reject_path(plan_path, role="R6 signed plan")
    plan = _load_json(plan_path, role="R6 signed plan")
    _verify_signed(plan, "plan_sha256", role="R6 signed plan")
    if (
        plan.get("format") != FORMAT
        or plan.get("status") != "preregistered_no_execution"
        or plan.get("materialization_scope") != "R3_materialized_OOF_D250_only"
        or plan.get("r4_checkpoints_are_evaluator_cli_inputs") is not False
        or plan.get("task_success_claim_authorized") is not False
        or plan.get("selector_deployment_authorized") is not False
        or plan.get("fresh_paths_accepted") is not False
        or plan.get("fresh_inputs_accepted") is not False
        or plan.get("fresh_labels_read") is not False
    ):
        raise RuntimeError("R6 plan status/scope changed")
    recomputed = _recompute_plan(plan)
    if recomputed.get("plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("R6 inputs or implementation changed after preregistration")
    output_root = _reject_path(Path(str(plan["output_root"])), role="R6 output root")
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    logs.mkdir()
    _write_immutable_json(output_root / "launch_plan.json", plan)
    state_path = output_root / "launch_state.json"
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "initializing_r6_v9_group_relative_evaluation",
        "plan": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "server_pid": os.getpid(),
        "current_stage": None,
        "last_completed_stage": None,
        "recovery_sources": [
            str(output_root / "launch_plan.json"),
            str(state_path),
            str(logs),
        ],
        "fresh_paths_accepted": False,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
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
        _run_stage(plan=plan, state=state, state_path=state_path, logs=logs)
        result = _validate_result(
            output_root / "group_relative_success_oof.json", plan=plan
        )
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
                "r4_summary_sha256": plan["r4_adamw_lineage"]["summary_sha256"],
                "r4_checkpoint_count": FOLD_COUNT,
                "r4_checkpoints_are_lineage_only": True,
                "gpu_idle_audit": gpu_audit,
                "result": result,
                "adaptive_development_only": True,
                "task_success_claim_authorized": False,
                "selector_deployment_authorized": False,
                "automatic_fresh_launch": False,
                "fresh_paths_accepted": False,
                "fresh_inputs_accepted": False,
                "fresh_labels_read": False,
            },
            "summary_sha256",
        )
        _write_immutable_json(output_root / "launch_summary.json", summary)
        state.update(
            {
                "status": TERMINAL_STATUS,
                "summary": str(output_root / "launch_summary.json"),
                "summary_sha256": summary["summary_sha256"],
                "result_audit": result,
                "current_stage": None,
                "last_heartbeat_unix": time.time(),
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
                "last_heartbeat_unix": time.time(),
                "fresh_paths_accepted": False,
                "fresh_inputs_accepted": False,
                "fresh_labels_read": False,
                "selector_deployment_authorized": False,
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
    plan_path = _reject_path(plan_path, role="R6 detached plan")
    nohup_log = _reject_path(nohup_log, role="R6 detached nohup log")
    receipt_path = _reject_path(receipt_path, role="R6 detach receipt")
    if nohup_log.exists() or receipt_path.exists():
        raise FileExistsError("R6 detached log/receipt must not exist")
    plan = _load_json(plan_path, role="R6 detached plan")
    _verify_signed(plan, "plan_sha256", role="R6 detached plan")
    if _recompute_plan(plan).get("plan_sha256") != plan.get("plan_sha256"):
        raise RuntimeError("R6 bindings changed before detach")
    if Path(str(plan["output_root"])).exists():
        raise FileExistsError(plan["output_root"])
    launcher = next(
        Path(record["path"])
        for relative, record in plan["implementation_files"].items()
        if Path(relative).name == Path(__file__).name
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
            "remote_recovery_state": str(
                Path(str(plan["output_root"])) / "launch_state.json"
            ),
            "remote_recovery_summary": str(
                Path(str(plan["output_root"])) / "launch_summary.json"
            ),
            "detachment": "nohup_plus_start_new_session_redirected_stdio",
            "fresh_paths_accepted": False,
            "fresh_inputs_accepted": False,
            "fresh_labels_read": False,
        },
        "receipt_sha256",
    )
    _write_immutable_json(receipt_path, receipt)
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
    plan_output = _reject_path(plan_output, role="R6 plan output")
    output_root = _reject_path(output_root, role="R6 output root")
    if plan_output == output_root or output_root in plan_output.parents:
        raise RuntimeError("R6 plan must be outside the absent execution root")
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
    _write_immutable_json(plan_output, plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister_parser = subparsers.add_parser("preregister")
    preregister_parser.add_argument("--code-root", type=Path, required=True)
    preregister_parser.add_argument(
        "--materialization-manifest", type=Path, required=True
    )
    preregister_parser.add_argument("--r4-summary", type=Path, required=True)
    preregister_parser.add_argument("--output-root", type=Path, required=True)
    preregister_parser.add_argument("--python-bin", type=Path, required=True)
    preregister_parser.add_argument("--gpu-index", type=int, default=0)
    preregister_parser.add_argument("--plan-output", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--poll-seconds", type=float, default=30.0)
    detach_parser = subparsers.add_parser("detach")
    detach_parser.add_argument("--plan", type=Path, required=True)
    detach_parser.add_argument("--poll-seconds", type=float, default=30.0)
    detach_parser.add_argument("--nohup-log", type=Path, required=True)
    detach_parser.add_argument("--receipt", type=Path, required=True)
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
        print(json.dumps(execute(args.plan, poll_seconds=args.poll_seconds), sort_keys=True))
        return
    print(
        json.dumps(
            detach(
                args.plan,
                poll_seconds=args.poll_seconds,
                nohup_log=args.nohup_log,
                receipt_path=args.receipt,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_plan",
    "detach",
    "execute",
    "implementation_closure",
    "preregister",
]
