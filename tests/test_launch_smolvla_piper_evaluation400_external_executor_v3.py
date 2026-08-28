from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_smolvla_piper_evaluation400_results_v3 as result_v3  # noqa: E402
import launch_smolvla_piper_evaluation400_external_executor_v3 as executor  # noqa: E402
from test_evaluate_smolvla_piper_evaluation400_results_v3 import (  # noqa: E402
    synthetic_selector_proof,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def write_json(path: Path, value: Mapping[str, Any], mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(mode)


def frozen_script(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o444)
    return path


def signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**dict(value), field: executor.canonical_sha256(value)}


def private_key() -> tuple[Any, str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private, public.hex(), hashlib.sha256(public).hexdigest()


def pair(ordinal: int = 0) -> dict[str, Any]:
    pair_id = f"pair-{ordinal}"
    return {
        "ordinal": ordinal,
        "pair_id": pair_id,
        "target_manifest_global_ordinal": 9000 + ordinal,
        "requested_seed": 100 + ordinal,
        "resolved_seed": 200 + ordinal,
        "initial_scene_state_sha256": "1" * 64,
        "initial_measured_joint_state_sha256": "2" * 64,
        "initial_commanded_drive_target_sha256": "3" * 64,
        "condition_order": ["baseline", "etsf"],
        "candidate_count": 4,
    }


def core_and_plan(
    public_hex: str, public_sha: str, root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_authority_path = root / "schema6_runtime_execution_authority.json"
    write_json(runtime_authority_path, {
        "format": "synthetic_schema6_runtime_execution_authority_v2",
        "runtime_contract": {
            "runtime_contract_sha256": "9" * 64,
            "max_episode_steps": 200,
        },
    })
    runtime_authority_sha = executor.hash_file(runtime_authority_path)
    _selector_proof, selector_authority = synthetic_selector_proof(
        is_etsf=True, legal=[True, True, True, True], selected=1,
        include_authority=True,
        runtime_authority_sha256=runtime_authority_sha,
    )
    core = {
        "protocol_core_sha256": "a" * 64,
        "authority_policy": {"executor_identity_sha256": public_sha},
        "evaluation400": {"pair_identity_set_sha256": "b" * 64},
        "deployment": {
            "deployment_binding_sha256": "c" * 64,
            "policy_runtime_action_binding_sha256": "d" * 64,
            "baseline_selector": "lowest_legal_feasibility_root_candidate",
            "etsf_selector": "frozen_five_member_event_world_model_with_uncertainty_abstention",
            "selector_authority": selector_authority,
            "runtime_execution_authority": {
                "path": str(runtime_authority_path),
                "file_sha256": runtime_authority_sha,
                "nested_runtime_contract_sha256": "9" * 64,
                "max_episode_steps": 200,
            },
            "target_reset_runtime_contract_sha256": "9" * 64,
        },
    }
    plan = {
        "plan_sha256": "e" * 64,
        "authority": {
            "core": {"path": "/metadata/core.json", "file_sha256": "1" * 64, "logical_sha256": "a" * 64},
            "decision": {"path": "/metadata/decision.json", "file_sha256": "2" * 64, "logical_sha256": "f" * 64},
            "bundle": {"path": "/metadata/bundle.json", "file_sha256": "3" * 64, "logical_sha256": "9" * 64},
            "inventory": {"path": "/metadata/inventory.json", "file_sha256": "4" * 64, "logical_sha256": "8" * 64},
        },
        "pair_identity_set_sha256": "b" * 64,
        "deployment_binding_sha256": "c" * 64,
        "policy_runtime_action_binding_sha256": "d" * 64,
        "execution_nonce_hex": "ab" * 32,
        "ledger_id_sha256": "0" * 64,
        "executor_public_key_hex": public_hex,
        "executor_identity_sha256": public_sha,
        "python": {"path": sys.executable},
        "condition_runner": {"path": "/metadata/condition_runner.py"},
        "runtime_contract": {"path": "/metadata/runtime.json", "file_sha256": "5" * 64},
        "inventory_components": {
            "simulator_implementation": {"path": "/metadata/sim.py", "file_sha256": "6" * 64}
        },
    }
    return core, plan


def runner_value(request: Mapping[str, Any], result_root: Path) -> dict[str, Any]:
    ordered = [str(index + 1) * 64 for index in range(4)]
    legal = [False, True, True, True]
    selected = 1 if request["condition"] == "baseline" else 2
    trajectory = result_root / "trajectory.bin"
    continuation = result_root / "continuation.bin"
    trajectory.write_bytes(b"")
    continuation.write_bytes(b"")
    trajectory.chmod(0o444)
    continuation.chmod(0o444)
    is_etsf = request["condition"] == "etsf"
    selector_proof = synthetic_selector_proof(
        is_etsf=is_etsf, legal=legal, selected=selected
    )
    base = {
        "format": executor.RUNNER_RESULT_FORMAT,
        "status": executor.RUNNER_RESULT_STATUS,
        "request": {
            "path": str(result_root.parents[3] / "unused.json"),
            "file_sha256": "0" * 64,
            "logical_sha256": request["request_sha256"],
        },
        "pair_id": request["pair_id"],
        "ordinal": request["ordinal"],
        "attempt": 0,
        "condition": request["condition"],
        "condition_ordinal": request["condition_ordinal"],
        "shared_snapshot_sha256": request["shared_snapshot_sha256"],
        "candidate_count": 4,
        "ordered_candidate_sha256": ordered,
        "candidate_legal": legal,
        "candidate_registry_sha256": result_v3._candidate_registry_sha(
            request["pair_id"], ordered, legal
        ),
        "schema6_execution_authority_file_sha256": "8" * 64,
        "schema6_runtime_contract_sha256": "9" * 64,
        "max_episode_steps": 200,
        "selected_candidate_index": selected,
        "selector_execution_proof": selector_proof,
        "selector_execution_proof_sha256": executor.canonical_sha256(selector_proof),
        "selector_score_contract": selector_proof["score_contract"],
        "source_rank_score_contract_sha256": selector_proof[
            "source_rank_score_contract_sha256"
        ],
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "formal190_target_outcome_calibrated_acceptance_margin": is_etsf,
        "continuation_contract": result_v3.CONTINUATION_CONTRACT,
        "continuation_policy_sha256": "7" * 64,
        "continuation_rerank_after_root": False,
        "candidate_replacement_count": 0,
        "continuation_proof_sha256": executor.canonical_sha256(
            {
                "continuation_contract": result_v3.CONTINUATION_CONTRACT,
                "continuation_policy_sha256": "7" * 64,
                "continuation_rerank_after_root": False,
                "candidate_replacement_count": 0,
            }
        ),
        "task_success": request["condition"] == "etsf",
        "trajectory_artifact": {
            "path": str(trajectory),
            "file_sha256": executor.hash_file(trajectory),
        },
        "continuation_artifact": {
            "path": str(continuation),
            "file_sha256": executor.hash_file(continuation),
        },
        "simulator_exit_code": 0,
    }
    return signed(base, "result_sha256")


def execution_artifacts(
    root: Path,
    *,
    row: Mapping[str, Any],
    position: int,
    ordered: Sequence[str],
    legal: Sequence[bool],
    selected: int,
    success: bool,
    runtime_authority_sha256: str = "8" * 64,
    runtime_contract_sha256: str = "9" * 64,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    request_path = root / "request.json"
    request_base = {
        "format": executor.CONDITION_REQUEST_FORMAT,
        "status": "write_ahead_before_condition_popen",
        "plan_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "claim_sha256": "3" * 64,
        "pair_id": row["pair_id"],
        "ordinal": row["ordinal"],
        "requested_seed": row["requested_seed"],
        "resolved_seed": row["resolved_seed"],
        "initial_scene_state_sha256": row["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": row[
            "initial_measured_joint_state_sha256"
        ],
        "initial_commanded_drive_target_sha256": row[
            "initial_commanded_drive_target_sha256"
        ],
        "attempt": 0,
        "pair_identity_sha256": "4" * 64,
        "condition": row["condition_order"][position],
        "condition_ordinal": position,
        "condition_order": list(row["condition_order"]),
        "shared_snapshot_sha256": "5" * 64,
        "candidate_count": 4,
        "candidate_generation_contract_sha256": "6" * 64,
        "postfreeze_identity_or_order_change_authorized": False,
        "outcome_visible_before_condition_start": False,
    }
    request = signed(request_base, "request_sha256")
    write_json(request_path, request)
    stage_root = root / "stage"
    output_root = stage_root / "result"
    output_root.mkdir(parents=True)
    trajectory_path = output_root / "trajectory.bin"
    continuation_path = output_root / "continuation.bin"
    trajectory_path.write_bytes(b"synthetic")
    continuation_path.write_bytes(b"synthetic")
    trajectory_path.chmod(0o444)
    continuation_path.chmod(0o444)
    is_etsf = row["condition_order"][position] == "etsf"
    selector_proof = synthetic_selector_proof(
        is_etsf=is_etsf, legal=list(legal), selected=selected
    )
    runner_base = {
        "format": executor.RUNNER_RESULT_FORMAT,
        "status": executor.RUNNER_RESULT_STATUS,
        "request": {
            "path": str(request_path),
            "file_sha256": executor.hash_file(request_path),
            "logical_sha256": request["request_sha256"],
        },
        "pair_id": row["pair_id"],
        "ordinal": row["ordinal"],
        "attempt": 0,
        "condition": row["condition_order"][position],
        "condition_ordinal": position,
        "shared_snapshot_sha256": request["shared_snapshot_sha256"],
        "candidate_count": 4,
        "candidate_registry_sha256": result_v3._candidate_registry_sha(
            row["pair_id"], ordered, legal
        ),
        "schema6_execution_authority_file_sha256": runtime_authority_sha256,
        "schema6_runtime_contract_sha256": runtime_contract_sha256,
        "max_episode_steps": 200,
        "ordered_candidate_sha256": list(ordered),
        "candidate_legal": list(legal),
        "selected_candidate_index": selected,
        "selector_execution_proof": selector_proof,
        "selector_execution_proof_sha256": executor.canonical_sha256(selector_proof),
        "selector_score_contract": selector_proof["score_contract"],
        "source_rank_score_contract_sha256": selector_proof[
            "source_rank_score_contract_sha256"
        ],
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "formal190_target_outcome_calibrated_acceptance_margin": is_etsf,
        "continuation_contract": result_v3.CONTINUATION_CONTRACT,
        "continuation_policy_sha256": "7" * 64,
        "continuation_rerank_after_root": False,
        "candidate_replacement_count": 0,
        "continuation_proof_sha256": executor.canonical_sha256(
            {
                "continuation_contract": result_v3.CONTINUATION_CONTRACT,
                "continuation_policy_sha256": "7" * 64,
                "continuation_rerank_after_root": False,
                "candidate_replacement_count": 0,
            }
        ),
        "task_success": success,
        "trajectory_artifact": {
            "path": str(trajectory_path),
            "file_sha256": executor.hash_file(trajectory_path),
        },
        "continuation_artifact": {
            "path": str(continuation_path),
            "file_sha256": executor.hash_file(continuation_path),
        },
        "simulator_exit_code": 0,
    }
    runner_result = signed(runner_base, "result_sha256")
    runner_path = output_root / "condition_result.json"
    write_json(runner_path, runner_result)
    python_path = Path(sys.executable).resolve()
    condition_runner = ROOT / "scripts" / "run_smolvla_piper_evaluation400_condition_v3.py"
    original = [
        str(python_path), str(condition_runner), "--request", str(request_path),
        "--output-root", str(output_root), "--device", "cuda:0",
    ]
    fd_mapping = [
        {
            "role": role,
            "source_path": str(path),
            "source_file_sha256": executor.hash_file(path),
            "inherited_fd": descriptor,
            "executed_path": f"/proc/self/fd/{descriptor}",
        }
        for role, path, descriptor in (
            ("runtime_python", python_path, 10),
            ("condition_runner", condition_runner, 11),
        )
    ]
    executed = [fd_mapping[0]["executed_path"], "-I", fd_mapping[1]["executed_path"], *original[2:]]
    launch_base = {
        "format": executor.STAGE_FORMAT,
        "status": "fd_bound_guard_passed_immediately_before_popen",
        "original_command": original,
        "command_sha256": executor.canonical_sha256(original),
        "executed_command": executed,
        "fd_mapping": fd_mapping,
        "isolated_python": True,
        "environment_policy": "explicit_allowlist_no_pythonpath_pythonhome_or_ld_preload",
        "environment_keys": [
            "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "LANG", "LC_ALL", "PATH",
            "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED",
            "TRANSFORMERS_OFFLINE",
        ],
        "forbidden_environment_keys_absent": sorted(
            ["PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"]
        ),
        "gpu_uuid": "GPU-synthetic",
        "device": "cuda:0",
    }
    launch = signed(launch_base, "launch_sha256")
    launch_path = stage_root / "launch.json"
    write_json(launch_path, launch)
    lifecycle = signed(
        {
            "popen_attempted": True, "popen_reached": True,
            "process_pid": 123, "process_pgid": 123,
            "process_group_isolated": True, "returncode": 0,
            "direct_process_reaped": True, "process_group_reaped": True,
            "binding_status": "bound_reaped",
        },
        "lifecycle_sha256",
    )
    lifecycle_path = stage_root / "lifecycle.json"
    write_json(lifecycle_path, lifecycle)
    idle = signed(
        {
            "gpu_index": 0, "gpu_name": "NVIDIA RTX 4090",
            "gpu_uuid": "GPU-synthetic", "checks": 2,
        },
        "audit_sha256",
    )
    before_path = root / "gpu_idle_before_condition.json"
    after_path = root / "gpu_idle_after_condition.json"
    write_json(before_path, idle)
    write_json(after_path, idle)
    log_path = stage_root / "run.log"
    exit_path = stage_root / "run.exit"
    log_path.write_bytes(b"")
    exit_path.write_bytes(b"0\n")
    log_path.chmod(0o444)
    exit_path.chmod(0o444)

    def record(path: Path, logical: str | None = None) -> dict[str, str]:
        value = {"path": str(path), "file_sha256": executor.hash_file(path)}
        if logical is not None:
            value["logical_sha256"] = logical
        return value

    return {
        "runner_result": record(runner_path, runner_result["result_sha256"]),
        "stage_launch": record(launch_path, launch["launch_sha256"]),
        "stage_lifecycle": record(lifecycle_path, lifecycle["lifecycle_sha256"]),
        "stage_log": record(log_path),
        "stage_exit": record(exit_path),
        "gpu_idle_before": record(before_path, idle["audit_sha256"]),
        "gpu_idle_after": record(after_path, idle["audit_sha256"]),
        "gpu_uuid": "GPU-synthetic",
    }


def test_executor_receipt_is_real_ed25519_and_result_evaluator_compatible() -> None:
    private, _public_hex, _public_sha = private_key()
    statement = {"hello": "world"}
    receipt = executor.sign_executor_receipt(
        receipt_format=result_v3.CONDITION_FORMAT,
        receipt_status=result_v3.CONDITION_STATUS,
        statement=statement,
        private_key=private,
    )
    decoded, logical = result_v3._verify_executor_receipt(
        receipt,
        expected_format=result_v3.CONDITION_FORMAT,
        expected_status=result_v3.CONDITION_STATUS,
        public_key=private.public_key(),
        role="synthetic condition",
    )
    assert decoded == statement
    assert logical == receipt["receipt_sha256"]


def test_condition_and_pair_statements_match_result_evaluator_exact_fields(
    tmp_path: Path,
) -> None:
    private, public_hex, public_sha = private_key()
    core, plan = core_and_plan(public_hex, public_sha, tmp_path)
    row = pair()
    request = executor.condition_request(
        plan=plan,
        claim={"logical_sha256": "4" * 64},
        pair=row,
        condition_ordinal=0,
    )
    tmp_result = Path("/tmp/unused")
    ordered = [str(index + 1) * 64 for index in range(4)]
    legal = [True, True, True, True]
    result = {
        "ordered_candidate_sha256": ordered,
        "candidate_legal": legal,
        "candidate_registry_sha256": result_v3._candidate_registry_sha(row["pair_id"], ordered, legal),
        "continuation_contract": result_v3.CONTINUATION_CONTRACT,
        "continuation_policy_sha256": "5" * 64,
        "continuation_rerank_after_root": False,
        "candidate_replacement_count": 0,
        "continuation_proof_sha256": executor.canonical_sha256({
            "continuation_contract": result_v3.CONTINUATION_CONTRACT,
            "continuation_policy_sha256": "5" * 64,
            "continuation_rerank_after_root": False,
            "candidate_replacement_count": 0,
        }),
        "selected_candidate_index": 0,
        "task_success": False,
    }
    baseline = executor.condition_execution_statement(
        plan=plan,
        pair=row,
        position=0,
        result=result,
        dependency_rehash_sha256="6" * 64,
        core=core,
        ledger_condition_start_event_sha256="7" * 64,
        execution_artifacts=execution_artifacts(
            tmp_path / "baseline",
            row=row, position=0, ordered=ordered, legal=legal,
            selected=0, success=False,
            runtime_authority_sha256=core["deployment"][
                "runtime_execution_authority"
            ]["file_sha256"],
        ),
    )
    result_v3._validate_condition_statement(
        baseline,
        pair=row,
        position=0,
        core=core,
        decision={"decision_sha256": "f" * 64},
        bundle={"bundle_sha256": "9" * 64},
        execution_nonce_hex=plan["execution_nonce_hex"],
        dependency_rehash_sha256="6" * 64,
    )
    missing_selector_core = {
        **core,
        "deployment": {
            key: value
            for key, value in core["deployment"].items()
            if key != "selector_authority"
        },
    }
    with pytest.raises(
        result_v3.Evaluation400ResultError,
        match="paired selector authority is missing",
    ):
        result_v3._validate_condition_statement(
            baseline,
            pair=row,
            position=0,
            core=missing_selector_core,
            decision={"decision_sha256": "f" * 64},
            bundle={"bundle_sha256": "9" * 64},
            execution_nonce_hex=plan["execution_nonce_hex"],
            dependency_rehash_sha256="6" * 64,
        )
    etsf = dict(baseline)
    etsf["global_condition_ordinal"] = 1
    etsf["condition_position"] = 1
    etsf["condition_id"] = "etsf"
    etsf["selector"] = core["deployment"]["etsf_selector"]
    etsf["ledger_condition_start_event_sha256"] = "8" * 64
    etsf["execution_artifacts"] = execution_artifacts(
        tmp_path / "etsf",
        row=row, position=1, ordered=ordered, legal=legal,
        selected=0, success=False,
        runtime_authority_sha256=core["deployment"][
            "runtime_execution_authority"
        ]["file_sha256"],
    )
    condition_records = [
        {
            "global_condition_ordinal": index,
            "pair_ordinal": 0,
            "condition_position": index,
            "condition_id": row["condition_order"][index],
            "path": f"/metadata/condition-{index}.json",
            "file_sha256": str(index + 1) * 64,
            "logical_sha256": str(index + 3) * 64,
        }
        for index in range(2)
    ]
    pair_statement = executor.pair_execution_statement(
        plan=plan,
        pair=row,
        condition_records=condition_records,
        condition_statements=[baseline, etsf],
        ledger_condition_terminal_event_sha256=["a" * 64, "b" * 64],
        dependency_rehash_sha256="6" * 64,
    )
    result_v3._validate_pair_statement(
        pair_statement,
        pair=row,
        conditions=list(zip(condition_records, [baseline, etsf], strict=True)),
        core=core,
        decision={"decision_sha256": "f" * 64},
        bundle={"bundle_sha256": "9" * 64},
        execution_nonce_hex=plan["execution_nonce_hex"],
        dependency_rehash_sha256="6" * 64,
    )
    receipt = executor.sign_executor_receipt(
        receipt_format=result_v3.PAIR_FORMAT,
        receipt_status=result_v3.PAIR_STATUS,
        statement=pair_statement,
        private_key=private,
    )
    assert set(receipt) == result_v3.RECEIPT_FIELDS


def test_global_worm_claim_is_unique_across_output_roots(tmp_path: Path) -> None:
    key, public_hex, public_sha = private_key()
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o700)
    plan = {
        "authority": {
            "core": {"logical_sha256": "1" * 64},
            "decision": {"logical_sha256": "2" * 64},
            "bundle": {"logical_sha256": "3" * 64},
        },
        "pair_identity_set_sha256": "3" * 64,
        "deployment_binding_sha256": "5" * 64,
        "policy_runtime_action_binding_sha256": "6" * 64,
        "execution_nonce_hex": "ab" * 32,
        "ledger_id_sha256": "7" * 64,
        "executor_public_key_hex": public_hex,
        "executor_identity_sha256": public_sha,
    }
    first = executor.acquire_lane_claim(
        ledger_root=ledger, identity="4" * 64, plan=plan, private_key=key
    )
    assert first["logical_sha256"]
    with pytest.raises(executor.ExecutorV3Error, match="already claimed"):
        executor.acquire_lane_claim(
            ledger_root=ledger, identity="4" * 64, plan=plan, private_key=key
        )


def test_started_pair_without_terminal_is_incomplete_and_never_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "PAIR_COUNT", 1)
    lane = tmp_path / "lane"
    (lane / "pairs" / "000").mkdir(parents=True)
    write_json(lane / "pairs" / "000" / "pair_started.json", {})
    with pytest.raises(executor.IncompleteLane, match="cannot be replayed"):
        executor.scan_completed_pairs(
            claim={"lane_root": str(lane)},
            pairs=[pair()],
            plan={"executor_public_key_hex": "00" * 32},
        )


def test_pair_row_requires_exact_ordinal_attempt_lane_and_four_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "PAIR_COUNT", 1)
    row = pair()
    core = {
        "evaluation400": {
            "pair_count": 1,
            "pair_identity_set_sha256": "a" * 64,
            "pairs": [row],
        }
    }
    assert executor.validate_pair_rows(core) == [row]
    core["evaluation400"]["pairs"][0]["ordinal"] = False
    with pytest.raises(executor.ExecutorV3Error, match="ordering/identity"):
        executor.validate_pair_rows(core)


def test_condition_runner_result_rejects_bool_attempt_and_candidate_drift(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request = {
        "request_sha256": "a" * 64,
        "pair_id": "pair-0",
        "ordinal": 0,
        "attempt": 0,
        "condition": "baseline",
        "condition_ordinal": 0,
        "shared_snapshot_sha256": "b" * 64,
        "candidate_count": 4,
    }
    write_json(request_path, request)
    result_root = tmp_path / "result"
    result_root.mkdir()
    value = runner_value(request, result_root)
    value["request"] = {
        "path": str(request_path),
        "file_sha256": executor.hash_file(request_path),
        "logical_sha256": request["request_sha256"],
    }
    value["result_sha256"] = executor.canonical_sha256(
        {key: child for key, child in value.items() if key != "result_sha256"}
    )
    write_json(result_root / "condition_result.json", value)
    (result_root / "run.exit").write_text("0\n", encoding="ascii")
    (result_root / "run.exit").chmod(0o444)
    assert executor.validate_runner_result(
        result_root, request_path=request_path, request=request
    )["value"]["attempt"] == 0
    value["attempt"] = False
    value["result_sha256"] = executor.canonical_sha256(
        {key: child for key, child in value.items() if key != "result_sha256"}
    )
    path = result_root / "condition_result.json"
    path.chmod(0o644)
    write_json(path, value)
    with pytest.raises(executor.ExecutorV3Error, match="semantics changed"):
        executor.validate_runner_result(
            result_root, request_path=request_path, request=request
        )


def test_condition_stage_proves_pid_pgid_and_group_reaped(tmp_path: Path) -> None:
    script = frozen_script(
        tmp_path / "runner.py",
        "import os\n"
        "assert all(name not in os.environ for name in "
        "('PYTHONPATH','PYTHONHOME','LD_PRELOAD','LD_LIBRARY_PATH'))\n",
    )
    result = executor.run_condition_stage(
        command=[str(Path(sys.executable).resolve()), str(script)],
        stage_root=tmp_path / "stage",
        gpu_uuid="GPU-synthetic",
        pre_popen_guard=lambda: None,
    )
    _path, lifecycle, _sha = executor.read_json(
        tmp_path / "stage" / "lifecycle.json", "lifecycle"
    )
    assert lifecycle["process_pid"] == lifecycle["process_pgid"]
    assert lifecycle["direct_process_reaped"] is True
    assert lifecycle["process_group_reaped"] is True
    assert result["returncode"] == 0
    _path, launch, _sha = executor.read_json(
        tmp_path / "stage" / "launch.json", "launch"
    )
    assert launch["executed_command"][1] == "-I"
    assert [row["role"] for row in launch["fd_mapping"]] == [
        "runtime_python", "condition_runner"
    ]
    assert all(row["executed_path"].startswith("/proc/self/fd/") for row in launch["fd_mapping"])
    assert not ({"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"} & set(launch["environment_keys"]))


def test_unknown_pgid_reaps_direct_child_without_signaling_foreign_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        executor.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("foreign PGID signaled")),
    )
    script = frozen_script(
        tmp_path / "runner.py", "import time\ntime.sleep(30)\n"
    )
    with pytest.raises(executor.UnprovenProcessGroup):
        executor.run_condition_stage(
            command=[str(Path(sys.executable).resolve()), str(script)],
            stage_root=tmp_path / "stage",
            gpu_uuid="GPU-synthetic",
            pre_popen_guard=lambda: None,
            getpgid=lambda pid: pid + 100000,
        )
    _path, lifecycle, _sha = executor.read_json(
        tmp_path / "stage" / "lifecycle.json", "lifecycle"
    )
    assert lifecycle["direct_process_reaped"] is True
    assert lifecycle["process_group_reaped"] is False


def test_prepopen_guard_failure_is_not_mislabeled_as_unproven(
    tmp_path: Path,
) -> None:
    def fail() -> None:
        raise executor.ExecutorV3Error("tampered implementation")

    script = frozen_script(tmp_path / "runner.py", "pass\n")
    with pytest.raises(executor.ExecutorV3Error, match="pre-Popen guard"):
        executor.run_condition_stage(
            command=[str(Path(sys.executable).resolve()), str(script)],
            stage_root=tmp_path / "stage",
            gpu_uuid="GPU-synthetic",
            pre_popen_guard=fail,
        )
    _path, lifecycle, _sha = executor.read_json(
        tmp_path / "stage" / "lifecycle.json", "lifecycle"
    )
    assert lifecycle["popen_attempted"] is False
    assert lifecycle["binding_status"] == "not_attempted"


def test_inherited_cuda_mapping_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(executor.ExecutorV3Error, match="remapping is forbidden"):
        executor.reject_inherited_cuda_mapping()


def test_recursive_local_dependency_closure_rehashes_every_reached_file(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    root = frozen_script(scripts / "root.py", "import child\n")
    child = frozen_script(scripts / "child.py", "VALUE = 1\n")
    closure = executor.build_local_dependency_closure(
        [root], scripts_root=scripts
    )
    assert [row["relative_path"] for row in closure["files"]] == [
        "child.py", "root.py"
    ]
    closure_path = scripts / "closure.json"
    write_json(closure_path, closure)
    descriptor = {
        "path": str(closure_path),
        "file_sha256": executor.hash_file(closure_path),
        "logical_sha256": closure["closure_sha256"],
    }
    assert executor.validate_local_dependency_closure(
        descriptor, expected_roots=[root]
    ) == closure
    child.chmod(0o644)
    child.write_text("VALUE = 2\n", encoding="utf-8")
    child.chmod(0o444)
    with pytest.raises(executor.ExecutorV3Error, match="closure changed"):
        executor.validate_local_dependency_closure(
            descriptor, expected_roots=[root]
        )


def test_unbounded_dynamic_import_is_rejected_from_closure(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    root = frozen_script(
        scripts / "root.py",
        "import importlib\nname = 'child'\nimportlib.import_module(name)\n",
    )
    frozen_script(scripts / "child.py", "VALUE = 1\n")
    with pytest.raises(executor.ExecutorV3Error, match="nonliteral dynamic import"):
        executor.build_local_dependency_closure([root], scripts_root=scripts)


def test_worm_terminal_is_hidden_until_tree_freezes_and_rolls_back_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = tmp_path / "lane"
    lane.mkdir(mode=0o700)
    frozen_script(lane / "event.json", "{}\n")
    terminal = executor.publish_worm_terminal(
        lane, "execution_terminal.json", {"status": "complete"}
    )
    assert stat.S_IMODE(terminal.stat().st_mode) == 0o444
    assert stat.S_IMODE(lane.stat().st_mode) == 0o555

    failed_lane = tmp_path / "failed-lane"
    failed_lane.mkdir(mode=0o700)
    frozen_script(failed_lane / "event.json", "{}\n")
    monkeypatch.setattr(
        executor, "freeze_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("freeze failed")),
    )
    with pytest.raises(OSError, match="freeze failed"):
        executor.publish_worm_terminal(
            failed_lane, "execution_failure.json", {"status": "failed"}
        )
    assert not (failed_lane / "execution_failure.json").exists()
    assert stat.S_IMODE(failed_lane.stat().st_mode) == 0o700


def test_result_evaluator_rejects_bool_lifecycle_and_nonzero_stage_exit(
    tmp_path: Path,
) -> None:
    row = pair()
    ordered = [str(index + 1) * 64 for index in range(4)]
    legal = [True, True, True, True]
    statement = {
        "pair_id": row["pair_id"],
        "pair_ordinal": 0,
        "condition_id": row["condition_order"][0],
        "condition_position": 0,
        "success": False,
        "candidate_registry_sha256": result_v3._candidate_registry_sha(
            row["pair_id"], ordered, legal
        ),
        "ordered_candidate_sha256": ordered,
        "candidate_legal": legal,
        "selected_candidate_ordinal": 0,
    }
    runtime_path = tmp_path / "runtime-authority.json"
    write_json(runtime_path, {
        "runtime_contract": {
            "runtime_contract_sha256": "9" * 64,
            "max_episode_steps": 200,
        }
    })
    runtime_binding = {
        "path": str(runtime_path),
        "file_sha256": executor.hash_file(runtime_path),
        "nested_runtime_contract_sha256": "9" * 64,
        "max_episode_steps": 200,
    }
    artifacts = execution_artifacts(
        tmp_path / "condition",
        row=row,
        position=0,
        ordered=ordered,
        legal=legal,
        selected=0,
        success=False,
        runtime_authority_sha256=runtime_binding["file_sha256"],
    )
    result_v3._validate_condition_execution_artifacts(
        artifacts, statement=statement, role="synthetic condition",
        runtime_execution_authority=runtime_binding,
        target_reset_runtime_contract_sha256="9" * 64,
    )
    lifecycle_path = Path(artifacts["stage_lifecycle"]["path"])
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    lifecycle["returncode"] = False
    lifecycle["lifecycle_sha256"] = executor.canonical_sha256({
        key: value
        for key, value in lifecycle.items()
        if key != "lifecycle_sha256"
    })
    lifecycle_path.chmod(0o644)
    write_json(lifecycle_path, lifecycle)
    artifacts["stage_lifecycle"] = {
        "path": str(lifecycle_path),
        "file_sha256": executor.hash_file(lifecycle_path),
        "logical_sha256": lifecycle["lifecycle_sha256"],
    }
    with pytest.raises(result_v3.Evaluation400ResultError, match="lifecycle"):
        result_v3._validate_condition_execution_artifacts(
            artifacts, statement=statement, role="synthetic condition",
            runtime_execution_authority=runtime_binding,
            target_reset_runtime_contract_sha256="9" * 64,
        )

    second = execution_artifacts(
        tmp_path / "condition-exit",
        row=row,
        position=0,
        ordered=ordered,
        legal=legal,
        selected=0,
        success=False,
        runtime_authority_sha256=runtime_binding["file_sha256"],
    )
    exit_path = Path(second["stage_exit"]["path"])
    exit_path.chmod(0o644)
    exit_path.write_bytes(b"1\n")
    exit_path.chmod(0o444)
    second["stage_exit"]["file_sha256"] = executor.hash_file(exit_path)
    with pytest.raises(result_v3.Evaluation400ResultError, match="exact zero"):
        result_v3._validate_condition_execution_artifacts(
            second, statement=statement, role="synthetic condition",
            runtime_execution_authority=runtime_binding,
            target_reset_runtime_contract_sha256="9" * 64,
        )
