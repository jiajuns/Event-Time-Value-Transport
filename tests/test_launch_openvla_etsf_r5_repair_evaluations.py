from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "launch_openvla_etsf_r5_repair_evaluations.py"
SPEC = importlib.util.spec_from_file_location("r5_repair_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _resign(value: dict, key: str) -> dict:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = launcher.canonical_sha256(result)
    return result


def _groups() -> list[str]:
    return [f"task|body|group_{index:03d}" for index in range(250)]


def _group_sha(groups: list[str]) -> str:
    return launcher.canonical_sha256({"logical_groups": groups})


def _make_code_root(tmp_path: Path) -> Path:
    code_root = tmp_path / "code_r5"
    scripts = code_root / "scripts"
    scripts.mkdir(parents=True)
    files = {
        "launch_openvla_etsf_r5_repair_evaluations.py": "import shared_r5_dep\n",
        "calibrate_openvla_etsf_v8_success_inner_cv.py": (
            "import shared_r5_dep\n"
            "import train_openvla_etsf_v8_structured_adapters\n"
        ),
        "evaluate_openvla_etsf_duration_hierarchy_oof.py": (
            "import openvla_etsf_duration_hierarchy\n"
            "import openvla_etsf_v8_structured_adapters\n"
            "import train_openvla_etsf_v8_structured_adapters\n"
        ),
        "shared_r5_dep.py": "VALUE = 1\n",
        "openvla_etsf_duration_hierarchy.py": "VALUE = 2\n",
        "openvla_etsf_v8_structured_adapters.py": "VALUE = 3\n",
        "train_openvla_etsf_v8_structured_adapters.py": "VALUE = 4\n",
    }
    for name, content in files.items():
        (scripts / name).write_text(content, encoding="utf-8")
    return code_root


def _make_python_symlink(tmp_path: Path) -> Path:
    target = tmp_path / "python_target"
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    invocation = tmp_path / "venv" / "bin" / "python"
    invocation.parent.mkdir(parents=True)
    invocation.symlink_to(target)
    return invocation


def _make_input_bundle(tmp_path: Path) -> dict[str, Path]:
    code_root = _make_code_root(tmp_path)
    python_bin = _make_python_symlink(tmp_path)
    materialized = tmp_path / "r3_materialized"
    materialized.mkdir()
    all_groups = _groups()
    fold_rows = []
    artifacts = []
    for fold_id in range(5):
        holdout = all_groups[fold_id * 50 : (fold_id + 1) * 50]
        training = sorted(set(all_groups) - set(holdout))
        row = {
            "outer_fold_id": fold_id,
            "training_groups": training,
            "training_groups_sha256": _group_sha(training),
            "oof_holdout_groups": holdout,
            "oof_holdout_groups_sha256": _group_sha(holdout),
        }
        for role in ("train", "holdout"):
            path = materialized / f"fold_{fold_id}_{role}.pt"
            path.write_bytes(f"fold={fold_id};role={role}\n".encode("ascii"))
            file_sha = launcher.sha256_path(path)
            payload_sha = hashlib.sha256(
                f"payload={fold_id};role={role}".encode("ascii")
            ).hexdigest()
            row[f"{role}_artifact"] = str(path.resolve())
            row[f"{role}_artifact_sha256"] = file_sha
            row[f"{role}_payload_sha256"] = payload_sha
            artifacts.append(
                {
                    "outer_fold_id": fold_id,
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": file_sha,
                    "payload_sha256": payload_sha,
                }
            )
        fold_rows.append(row)
    manifest_path = materialized / "materialization_manifest.json"
    manifest = _resign(
        {
            "format": launcher.MATERIALIZATION_FORMAT,
            "status": "complete_development_only",
            "base_checkpoint_sha256": "a" * 64,
            "event_spec_sha256": "b" * 64,
            "folds": fold_rows,
            "fresh_confirmation_data_or_labels_read": False,
            "authorization_guard_changed": False,
            "prospective_claim_for_v8": False,
        },
        "materialization_sha256",
    )
    _write_json(manifest_path, manifest)

    checkpoint_rows = []
    ten_artifact_sha = launcher.canonical_sha256(artifacts)
    for fold_id, fold in enumerate(fold_rows):
        order = fold["training_groups"]
        train_artifact = artifacts[fold_id * 2]
        input_authentication = {
            "status": "authenticated_complete_five_fold_materialization_bundle",
            "materialization_manifest": str(manifest_path.resolve()),
            "materialization_sha256": manifest["materialization_sha256"],
            "outer_fold_id": fold_id,
            "train_artifact_sha256": train_artifact["sha256"],
            "train_payload_sha256": train_artifact["payload_sha256"],
            "ten_artifact_bundle_sha256": ten_artifact_sha,
        }
        provenance = {
            "outer_fold_id": fold_id,
            "target_outer_fold_labels_used": False,
            "factual_outputs_frozen": True,
            "outer_training_groups": order,
            "outer_training_groups_sha256": fold["training_groups_sha256"],
            "oof_holdout_groups": fold["oof_holdout_groups"],
            "oof_holdout_groups_sha256": fold["oof_holdout_groups_sha256"],
        }
        order_sha = hashlib.sha256("\n".join(order).encode("utf-8")).hexdigest()
        checkpoint = {
            "format": launcher.CHECKPOINT_FORMAT,
            "fresh_confirmation_data_or_labels_read": False,
            "authorization_guard_changed": False,
            "all_steps_factual_inputs_bit_exact": True,
            "adapter_state_sha256": hashlib.sha256(
                f"adapter={fold_id}".encode("ascii")
            ).hexdigest(),
            "input_artifact_authentication": input_authentication,
            "provenance": provenance,
            "optimizer": {
                "name": "AdamW",
                "epochs": 10,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "record_order": order,
                "record_order_sha256": order_sha,
            },
            "steps": 10 * len(order),
            "last_step": {
                "gradient_clip_scope": "independent_per_probability_head"
            },
        }
        checkpoint_path = tmp_path / "r4_adamw" / f"fold_{fold_id}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        checkpoint_rows.append(
            {
                "outer_fold_id": fold_id,
                "path": str(checkpoint_path.resolve()),
                "file_sha256": launcher.sha256_path(checkpoint_path),
            }
        )
    r4_summary_path = tmp_path / "r4_adamw" / "resume_summary.json"
    r4_summary = _resign(
        {
            "format": launcher.R4_SUMMARY_FORMAT,
            "status": launcher.R4_TERMINAL_STATUS,
            "materialization": {
                "path": str(manifest_path.resolve()),
                "materialization_sha256": manifest["materialization_sha256"],
            },
            "adamw_checkpoints": checkpoint_rows,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "selector_authorized": False,
            "prospective_claim_allowed": False,
        },
        "summary_sha256",
    )
    _write_json(r4_summary_path, r4_summary)
    return {
        "code_root": code_root,
        "python_bin": python_bin,
        "manifest": manifest_path,
        "r4_summary": r4_summary_path,
        "output_root": tmp_path / "r5_output",
    }


def _build_plan(paths: dict[str, Path]) -> dict:
    return launcher.build_plan(
        code_root=paths["code_root"],
        materialization_manifest=paths["manifest"],
        r4_summary=paths["r4_summary"],
        output_root=paths["output_root"],
        python_bin=paths["python_bin"],
        gpu_index=2,
    )


def _write_success_result(path: Path, plan: dict) -> dict:
    contracts = []
    rows = []
    candidate_names = (
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
    )
    for owner in range(5):
        fold = plan["materialization"]["folds"][owner]
        contract = _resign(
            {
                "owner_fold_id": owner,
                "final_outer_checkpoint_sha256": plan["r4_adamw"]["checkpoints"][
                    owner
                ]["file_sha256"],
                "outer_holdout_groups": fold["oof_holdout_groups"],
                "outer_holdout_groups_sha256": fold["oof_holdout_groups_sha256"],
                "outer_holdout_labels_used_for_alpha_selection": False,
                "fresh50_inputs_or_labels_used": False,
            },
            "calibration_contract_sha256",
        )
        contracts.append(contract)
        for group in fold["oof_holdout_groups"]:
            for candidate_index, candidate_name in enumerate(candidate_names):
                rows.append(
                    {
                        "owner_fold_id": owner,
                        "logical_group": group,
                        "candidate_index": candidate_index,
                        "candidate_name": candidate_name,
                        "success_label": int(candidate_index == owner % 4),
                        "uncalibrated_success_probability": 0.2 + candidate_index * 0.1,
                        "calibrated_success_probability": 0.25 + candidate_index * 0.1,
                        "owner_training_prevalence_baseline": 0.3,
                        "calibration_contract_sha256": contract[
                            "calibration_contract_sha256"
                        ],
                    }
                )
    implementation = next(
        record
        for key, record in plan["implementation_files"].items()
        if Path(key).name == "calibrate_openvla_etsf_v8_success_inner_cv.py"
    )
    result = _resign(
        {
            "format": launcher.SUCCESS_FORMAT,
            "status": "complete_adaptive_development_only",
            "materialization_manifest": plan["materialization"]["path"],
            "materialization_sha256": plan["materialization"][
                "materialization_sha256"
            ],
            "implementation": implementation["path"],
            "implementation_sha256": implementation["sha256"],
            "fold_calibration_contracts": contracts,
            "calibrated_oof_rows": rows,
            "calibrated_oof_rows_sha256": launcher.canonical_sha256(rows),
            "calibrated_oof_row_count": len(rows),
            "action_ranking_preserved_within_each_group": True,
            "task_success_cannot_change_from_uncalibrated_argmax": True,
            "all_alpha_selection_completed_before_holdout_payload_deserialization": True,
            "outer_holdout_labels_used_for_alpha_selection": False,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "authorization": {
                "selector_authorized": False,
                "deployment_authorized": False,
            },
        },
        "result_sha256",
    )
    _write_json(path, result)
    return result


def _duration_implementation_contract(plan: dict) -> dict[str, str]:
    names = {
        "evaluate_openvla_etsf_duration_hierarchy_oof.py",
        "openvla_etsf_duration_hierarchy.py",
        "openvla_etsf_v8_structured_adapters.py",
        "train_openvla_etsf_v8_structured_adapters.py",
    }
    return {
        Path(key).name: value["sha256"]
        for key, value in plan["implementation_files"].items()
        if Path(key).name in names
    }


def _write_duration_result(output_dir: Path, plan: dict) -> dict:
    output_dir.mkdir(parents=True)
    arrays_path = output_dir / "duration_hierarchy_rows.npz"
    with arrays_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            owner_fold_id=np.asarray([0, 1, 2]),
            logical_group=np.asarray(["g0", "g1", "g2"]),
            row_index=np.asarray([0, 0, 0]),
        )
    keys = ["logical_group", "owner_fold_id", "row_index"]
    result = _resign(
        {
            "format": launcher.DURATION_FORMAT,
            "status": "fail_closed",
            "source_materialization": {
                "path": plan["materialization"]["path"],
                "file_sha256": plan["materialization"]["file_sha256"],
                "materialization_sha256": plan["materialization"][
                    "materialization_sha256"
                ],
                "ten_artifacts_authenticated": True,
                "source_hdf5_read": False,
            },
            "implementation_files": _duration_implementation_contract(plan),
            "row_arrays": {
                "path": str(arrays_path.resolve()),
                "file_sha256": launcher.sha256_path(arrays_path),
                "keys": keys,
                "rows": 3,
                "alignment": "owner_fold_id_logical_group_row_index",
            },
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "fresh50_confirmation_authorized": False,
            "selector_authorized": False,
            "prospective_claim_allowed": False,
        },
        "result_sha256",
    )
    _write_json(output_dir / "duration_hierarchy_evaluation.json", result)
    return result


def test_preregister_binds_every_input_and_preserves_interpreter_symlink(
    tmp_path: Path,
) -> None:
    paths = _make_input_bundle(tmp_path)
    plan_path = tmp_path / "r5_plan.json"
    plan = launcher.preregister(
        code_root=paths["code_root"],
        materialization_manifest=paths["manifest"],
        r4_summary=paths["r4_summary"],
        output_root=paths["output_root"],
        python_bin=paths["python_bin"],
        gpu_index=2,
        plan_output=plan_path,
    )
    assert not paths["output_root"].exists()
    assert plan["python_bin"] == str(paths["python_bin"].absolute())
    assert plan["python_contract"]["invocation_path_is_symlink"] is True
    assert plan["python_contract"]["resolved_target_path"] != plan["python_bin"]
    assert plan["commands"][0]["argv"][0] == str(paths["python_bin"].absolute())
    assert len(plan["materialization"]["artifacts"]) == 10
    assert [row["outer_fold_id"] for row in plan["r4_adamw"]["checkpoints"]] == list(
        range(5)
    )
    assert [row["uses_gpu"] for row in plan["commands"]] == [True, False]
    assert launcher.canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) == plan["plan_sha256"]
    with pytest.raises(FileExistsError):
        launcher.preregister(
            code_root=paths["code_root"],
            materialization_manifest=paths["manifest"],
            r4_summary=paths["r4_summary"],
            output_root=paths["output_root"],
            python_bin=paths["python_bin"],
            gpu_index=2,
            plan_output=plan_path,
        )


def test_preregister_refuses_plan_inside_execution_root(tmp_path: Path) -> None:
    paths = _make_input_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="outside"):
        launcher.preregister(
            code_root=paths["code_root"],
            materialization_manifest=paths["manifest"],
            r4_summary=paths["r4_summary"],
            output_root=paths["output_root"],
            python_bin=paths["python_bin"],
            gpu_index=0,
            plan_output=paths["output_root"] / "plan.json",
        )


def test_execute_rejects_dependency_change_before_creating_root(tmp_path: Path) -> None:
    paths = _make_input_bundle(tmp_path)
    plan = _build_plan(paths)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    dependency = paths["code_root"] / "scripts" / "shared_r5_dep.py"
    dependency.write_text("VALUE = 99\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        launcher.execute(plan_path, poll_seconds=5.0)
    assert not paths["output_root"].exists()


@pytest.mark.parametrize("target", ["manifest", "checkpoint", "summary"])
def test_input_file_tampering_is_rejected(tmp_path: Path, target: str) -> None:
    paths = _make_input_bundle(tmp_path)
    if target == "manifest":
        paths["manifest"].write_text("{}", encoding="utf-8")
    elif target == "checkpoint":
        checkpoint = tmp_path / "r4_adamw" / "fold_2.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    else:
        paths["r4_summary"].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _build_plan(paths)


def test_resigned_checkpoint_owner_provenance_change_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _make_input_bundle(tmp_path)
    checkpoint_path = tmp_path / "r4_adamw" / "fold_2.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["provenance"]["outer_fold_id"] = 3
    torch.save(checkpoint, checkpoint_path)
    summary = json.loads(paths["r4_summary"].read_text())
    summary["adamw_checkpoints"][2]["file_sha256"] = launcher.sha256_path(
        checkpoint_path
    )
    _write_json(paths["r4_summary"], _resign(summary, "summary_sha256"))
    with pytest.raises(RuntimeError, match="provenance"):
        _build_plan(paths)


def test_interpreter_retarget_is_detected_without_resolving_invocation(
    tmp_path: Path,
) -> None:
    paths = _make_input_bundle(tmp_path)
    plan = _build_plan(paths)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    replacement = tmp_path / "python_target_2"
    replacement.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    replacement.chmod(0o755)
    paths["python_bin"].unlink()
    paths["python_bin"].symlink_to(replacement)
    with pytest.raises(RuntimeError, match="changed"):
        launcher.execute(plan_path, poll_seconds=5.0)
    assert not paths["output_root"].exists()


def test_success_result_requires_exact_signed_250_by_four_alignment(
    tmp_path: Path,
) -> None:
    paths = _make_input_bundle(tmp_path)
    plan = _build_plan(paths)
    output = tmp_path / "success_result.json"
    result = _write_success_result(output, plan)
    audit = launcher._validate_success_output(output, plan=plan)
    assert audit["rows"] == 1000
    assert audit["logical_groups"] == 250

    changed = copy.deepcopy(result)
    changed["calibrated_oof_rows"][0]["candidate_name"] = "sample_blend_0.250"
    changed["calibrated_oof_rows_sha256"] = launcher.canonical_sha256(
        changed["calibrated_oof_rows"]
    )
    changed = _resign(changed, "result_sha256")
    _write_json(output, changed)
    with pytest.raises(RuntimeError, match="alignment"):
        launcher._validate_success_output(output, plan=plan)

    wrong_owner = copy.deepcopy(result)
    wrong_owner["calibrated_oof_rows"][0]["logical_group"] = plan[
        "materialization"
    ]["folds"][1]["oof_holdout_groups"][0]
    wrong_owner["calibrated_oof_rows_sha256"] = launcher.canonical_sha256(
        wrong_owner["calibrated_oof_rows"]
    )
    _write_json(output, _resign(wrong_owner, "result_sha256"))
    with pytest.raises(RuntimeError, match="alignment"):
        launcher._validate_success_output(output, plan=plan)


def test_success_result_rejects_semantic_scope_change_even_when_resigned(
    tmp_path: Path,
) -> None:
    paths = _make_input_bundle(tmp_path)
    plan = _build_plan(paths)
    output = tmp_path / "success_result.json"
    result = _write_success_result(output, plan)
    result["fresh50_labels_read"] = True
    _write_json(output, _resign(result, "result_sha256"))
    with pytest.raises(RuntimeError, match="contract"):
        launcher._validate_success_output(output, plan=plan)


def test_duration_result_authenticates_json_npz_and_implementation(
    tmp_path: Path,
) -> None:
    paths = _make_input_bundle(tmp_path)
    plan = _build_plan(paths)
    output = tmp_path / "duration_result"
    _write_duration_result(output, plan)
    audit = launcher._validate_duration_output(output, plan=plan)
    assert audit["rows"] == 3
    assert audit["evaluation_status"] == "fail_closed"
    arrays = output / "duration_hierarchy_rows.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="NPZ"):
        launcher._validate_duration_output(output, plan=plan)


def test_duration_result_rejects_resigned_scope_change(tmp_path: Path) -> None:
    paths = _make_input_bundle(tmp_path)
    plan = _build_plan(paths)
    output = tmp_path / "duration_result"
    result = _write_duration_result(output, plan)
    result["fresh50_inputs_accepted"] = True
    _write_json(
        output / "duration_hierarchy_evaluation.json",
        _resign(result, "result_sha256"),
    )
    with pytest.raises(RuntimeError, match="contract"):
        launcher._validate_duration_output(output, plan=plan)


def test_stage_runner_uses_cuda_only_for_success_and_atomically_publishes_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_root = _make_code_root(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    state_path = tmp_path / "state.json"
    state: dict = {}
    plan = {
        "code_root": str(code_root),
        "gpu_index": 7,
    }
    environments = []

    def fake_run(argv, **kwargs):
        environments.append(dict(kwargs["env"]))
        kwargs["stdout"].write(b"complete\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(launcher, "_validate_runtime_bindings", lambda plan: None)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    for stage, uses_gpu in (("success_calibration", True), ("duration_hierarchy", False)):
        argv = ["synthetic", stage]
        launcher._run_stage(
            command={
                "stage": stage,
                "argv": argv,
                "argv_sha256": launcher.canonical_sha256(argv),
                "uses_gpu": uses_gpu,
            },
            plan=plan,
            state=state,
            state_path=state_path,
            logs_dir=logs,
        )
    assert environments[0]["CUDA_VISIBLE_DEVICES"] == "7"
    assert environments[1]["CUDA_VISIBLE_DEVICES"] == ""
    assert not list(logs.glob("*.partial"))
    assert (logs / "success_calibration.log").read_text() == "complete\n"
    assert (logs / "duration_hierarchy.log").read_text() == "complete\n"


def test_gpu_pid_parser_accepts_empty_and_no_process_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(["", "No running processes found\n", "31\n42\n31\n"])

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher._gpu_compute_pids(0) == []
    assert launcher._gpu_compute_pids(0) == []
    assert launcher._gpu_compute_pids(0) == [31, 42]


def test_gpu_pid_query_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(argv, **kwargs):
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="timed out.*fail closed"):
        launcher._gpu_compute_pids(4)
    assert observed["timeout"] == 30


def _minimal_execution_plan(tmp_path: Path) -> tuple[dict, Path]:
    output = tmp_path / "execute_output"
    commands = []
    for stage, uses_gpu in (("success_calibration", True), ("duration_hierarchy", False)):
        argv = ["python", stage]
        commands.append(
            {
                "stage": stage,
                "argv": argv,
                "argv_sha256": launcher.canonical_sha256(argv),
                "uses_gpu": uses_gpu,
            }
        )
    plan = _resign(
        {
            "format": launcher.FORMAT,
            "status": "preregistered_no_execution",
            "code_root": str(tmp_path),
            "implementation_files": {},
            "implementation_bundle_sha256": launcher.canonical_sha256({}),
            "materialization": {
                "path": str(tmp_path / "materialization.json"),
                "materialization_sha256": "a" * 64,
            },
            "r4_adamw": {"summary_sha256": "b" * 64},
            "output_root": str(output),
            "python_bin": sys.executable,
            "gpu_index": 0,
            "commands": commands,
            "selector_authorized": False,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
        },
        "plan_sha256",
    )
    plan_path = tmp_path / "execute_plan.json"
    _write_json(plan_path, plan)
    return plan, plan_path


def test_execute_orders_stages_and_writes_signed_terminal_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _minimal_execution_plan(tmp_path)
    stages = []
    monkeypatch.setattr(launcher, "_recompute_plan", lambda value: copy.deepcopy(plan))
    monkeypatch.setattr(
        launcher,
        "_wait_for_gpu_idle",
        lambda **kwargs: {"gpu_index": 0, "compute_pids": [], "checks": 1},
    )

    def fake_stage(**kwargs):
        stages.append((kwargs["command"]["stage"], kwargs["command"]["uses_gpu"]))

    monkeypatch.setattr(launcher, "_run_stage", fake_stage)
    monkeypatch.setattr(
        launcher,
        "_validate_success_output",
        lambda *args, **kwargs: {"rows": 1000, "fresh50_labels_read": False},
    )
    monkeypatch.setattr(
        launcher,
        "_validate_duration_output",
        lambda *args, **kwargs: {"rows": 500, "fresh50_labels_read": False},
    )
    summary = launcher.execute(plan_path, poll_seconds=5.0)
    assert stages == [("success_calibration", True), ("duration_hierarchy", False)]
    assert summary["status"] == launcher.TERMINAL_STATUS
    unsigned = dict(summary)
    recorded = unsigned.pop("summary_sha256")
    assert recorded == launcher.canonical_sha256(unsigned)
    state = json.loads(
        (Path(plan["output_root"]) / "launch_state.json").read_text()
    )
    assert state["status"] == launcher.TERMINAL_STATUS


def test_execute_failure_writes_fail_closed_state_without_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _minimal_execution_plan(tmp_path)
    monkeypatch.setattr(launcher, "_recompute_plan", lambda value: copy.deepcopy(plan))
    monkeypatch.setattr(launcher, "_wait_for_gpu_idle", lambda **kwargs: {})

    def fail_stage(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(launcher, "_run_stage", fail_stage)
    with pytest.raises(RuntimeError, match="synthetic"):
        launcher.execute(plan_path, poll_seconds=5.0)
    root = Path(plan["output_root"])
    state = json.loads((root / "launch_state.json").read_text())
    assert state["status"] == launcher.FAILURE_STATUS
    assert not (root / "launch_summary.json").exists()


def test_detach_uses_nohup_new_session_and_signed_immutable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _minimal_execution_plan(tmp_path)
    launcher_path = tmp_path / "launcher.py"
    launcher_path.write_text("# launcher\n", encoding="utf-8")
    plan["implementation_files"] = {
        "scripts/launch_openvla_etsf_r5_repair_evaluations.py": {
            "path": str(launcher_path),
            "sha256": launcher.sha256_path(launcher_path),
        }
    }
    plan = _resign(plan, "plan_sha256")
    _write_json(plan_path, plan)
    monkeypatch.setattr(launcher, "_recompute_plan", lambda value: copy.deepcopy(plan))
    captured = {}

    class FakeProcess:
        pid = 8123

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    receipt_path = tmp_path / "detach_receipt.json"
    receipt = launcher.detach(
        plan_path,
        poll_seconds=7.0,
        nohup_log=tmp_path / "nohup.log",
        receipt_path=receipt_path,
    )
    assert captured["argv"][0] == "nohup"
    assert captured["kwargs"]["start_new_session"] is True
    assert receipt["pid"] == 8123
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256")
    assert recorded == launcher.canonical_sha256(unsigned)
    with pytest.raises(FileExistsError):
        launcher.detach(
            plan_path,
            poll_seconds=7.0,
            nohup_log=tmp_path / "nohup.log",
            receipt_path=receipt_path,
        )


@pytest.mark.parametrize("token", ["Fresh50", "confirmation_bundle"])
def test_external_scope_tokens_are_rejected(tmp_path: Path, token: str) -> None:
    with pytest.raises(RuntimeError):
        launcher._reject_path(tmp_path / token / "artifact.json", role="input")
