from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "resume_openvla_etsf_v8_r3_after_bridge_fix.py"
SPEC = importlib.util.spec_from_file_location("v8_r3_resume", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _signed(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = resume.canonical_sha256(result)
    return result


def _make_fixture(tmp_path: Path) -> dict[str, Path]:
    failed = tmp_path / "failed_r3"
    failed.mkdir(parents=True)
    code_root = tmp_path / "new_code"
    scripts = code_root / "scripts"
    scripts.mkdir(parents=True)
    all_names = set(resume.OLD_SCRIPT_NAMES.values()) | set(
        resume.NEW_IMPLEMENTATION_NAMES.values()
    )
    for name in all_names:
        (scripts / name).write_text(f"# synthetic {name}\n", encoding="utf-8")
    old_implementations = {
        name: {
            "path": str((scripts / filename).resolve()),
            "sha256": resume.sha256_path(scripts / filename),
        }
        for name, filename in resume.OLD_SCRIPT_NAMES.items()
    }
    plan_path = failed / "pipeline_plan.json"
    plan = _signed(
        {
            "format": resume.R3_FORMAT,
            "status": "waiting_for_v7",
            "code_root": "/immutable/old/code",
            "implementation_files": old_implementations,
            "v7_state": "/signed/v7/state",
            "v7_result": "/signed/v7/result",
            "data": "/signed/development250",
            "checkpoint": "/signed/factual.pt",
            "checkpoint_sha256": "1" * 64,
            "event_spec": "/signed/event.json",
            "event_spec_sha256": "2" * 64,
            "python_bin": "/signed/python",
            "gpu_index": 3,
            "optimizer_candidates": {
                "lbfgs_convex": {"max_iter": 100},
                "adamw_fixed": {"epochs": 10, "learning_rate": 0.001},
            },
            "adaptive_development_only": True,
            "prospective_claim_allowed": False,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "automatic_fresh_launch": False,
        },
        "plan_sha256",
    )
    _write_json(plan_path, plan)

    completed_stages = ["materialize_oof"] + [
        f"train_lbfgs_convex_fold_{fold}" for fold in range(5)
    ]
    stage_logs = {}
    commands = []
    for stage in completed_stages:
        log = failed / "logs" / f"{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"completed {stage}\n", encoding="utf-8")
        stage_logs[stage] = {"path": str(log), "sha256": resume.sha256_path(log)}
        argv = ["python", f"{stage}.py"]
        commands.append(
            {"stage": stage, "argv": argv, "argv_sha256": resume.canonical_sha256(argv), "log": str(log)}
        )
    failed_bridge_log = failed / "logs" / "evaluate_lbfgs_convex.log"
    failed_bridge_log.write_text("bridge failed\n", encoding="utf-8")
    bridge_argv = ["python", "evaluate_bridge.py"]
    commands.append(
        {
            "stage": "evaluate_lbfgs_convex",
            "argv": bridge_argv,
            "argv_sha256": resume.canonical_sha256(bridge_argv),
            "log": str(failed_bridge_log),
        }
    )
    state_path = failed / "pipeline_state.json"
    state = {
        **plan,
        "status": "failed_closed_no_fresh",
        "current_stage": "evaluate_lbfgs_convex",
        "last_completed_stage": "train_lbfgs_convex_fold_4",
        "commands": commands,
        "stage_logs": stage_logs,
        "error_type": "RuntimeError",
        "error": "synthetic bridge failure",
    }
    _write_json(state_path, state)

    materialized = failed / "materialized_oof"
    materialized.mkdir()
    fold_rows = []
    artifacts = []
    base_identity = "3" * 64
    for fold in range(5):
        for role in ("train", "holdout"):
            payload_sha = f"{fold + (0 if role == 'train' else 5):064x}"
            path = materialized / f"fold_{fold}_{role}.pt"
            torch.save(
                {
                    "payload_sha256": payload_sha,
                    "batches": [{"batch": {"current_event_id": torch.tensor([fold])}}],
                },
                path,
            )
            artifacts.append(
                {
                    "outer_fold_id": fold,
                    "role": role,
                    "path": str(path.resolve()),
                    "file_sha256": resume.sha256_path(path),
                    "payload_sha256": payload_sha,
                }
            )
        fold_rows.append(
            {
                "outer_fold_id": fold,
                "train_artifact": str((materialized / f"fold_{fold}_train.pt").resolve()),
                "train_artifact_sha256": resume.sha256_path(materialized / f"fold_{fold}_train.pt"),
                "train_payload_sha256": artifacts[-2]["payload_sha256"],
                "holdout_artifact": str((materialized / f"fold_{fold}_holdout.pt").resolve()),
                "holdout_artifact_sha256": resume.sha256_path(materialized / f"fold_{fold}_holdout.pt"),
                "holdout_payload_sha256": artifacts[-1]["payload_sha256"],
                "base_exclusion_audit": {
                    "status": "proven",
                    "base_identity_contract_sha256": base_identity,
                },
            }
        )
    manifest_path = materialized / "materialization_manifest.json"
    manifest = _signed(
        {
            "format": resume.MATERIALIZATION_FORMAT,
            "status": "complete_development_only",
            "base_checkpoint_sha256": plan["checkpoint_sha256"],
            "event_spec_sha256": plan["event_spec_sha256"],
            "folds": fold_rows,
            "fresh_confirmation_data_or_labels_read": False,
            "authorization_guard_changed": False,
            "prospective_claim_for_v8": False,
        },
        "materialization_sha256",
    )
    _write_json(manifest_path, manifest)
    materialization = {
        "path": str(manifest_path.resolve()),
        "materialization_sha256": manifest["materialization_sha256"],
        "artifacts": artifacts,
    }
    for fold in range(5):
        checkpoint_path = failed / "lbfgs_convex" / f"fold_{fold}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": resume.CHECKPOINT_FORMAT,
                "schema_version": 5,
                "fresh_confirmation_data_or_labels_read": False,
                "authorization_guard_changed": False,
                "all_steps_factual_inputs_bit_exact": True,
                "strict_oof_base_exclusion_eligible": True,
                "optimizer": {"name": "independent_full_batch_LBFGS"},
                "provenance": {
                    "outer_fold_id": fold,
                    "target_outer_fold_labels_used": False,
                },
                "input_artifact_authentication": resume._expected_input_authentication(
                    materialization=materialization, fold_id=fold
                ),
                "adapter_state_sha256": f"{20 + fold:064x}",
            },
            checkpoint_path,
        )
    return {
        "failed": failed,
        "plan": plan_path,
        "state": state_path,
        "manifest": manifest_path,
        "code_root": code_root,
        "output": tmp_path / "resume_output",
    }


def _build(paths: dict[str, Path]) -> dict:
    return resume.build_resume_plan(
        failed_plan_path=paths["plan"],
        failed_state_path=paths["state"],
        materialization_manifest=paths["manifest"],
        code_root=paths["code_root"],
        output_root=paths["output"],
        python_bin=Path(sys.executable),
        gpu_index=3,
    )


def test_plan_binds_complete_resume_scope_without_materializer(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    plan = _build(paths)
    assert plan["status"] == "preregistered_no_execution"
    assert len(plan["materialization"]["artifacts"]) == 10
    assert len(plan["lbfgs_checkpoints"]) == 5
    assert [row["stage"] for row in plan["commands"]] == [
        "evaluate_lbfgs_convex",
        "train_adamw_fixed_fold_0",
        "train_adamw_fixed_fold_1",
        "train_adamw_fixed_fold_2",
        "train_adamw_fixed_fold_3",
        "train_adamw_fixed_fold_4",
        "evaluate_adamw_fixed",
    ]
    assert all("materialize_openvla" not in row["argv"] for row in plan["commands"])
    assert plan["fresh50_inputs_accepted"] is False
    assert plan["selector_authorized"] is False


def test_plan_fails_when_any_materialized_artifact_changes(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    _build(paths)
    artifact = paths["failed"] / "materialized_oof" / "fold_4_holdout.pt"
    with artifact.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        _build(paths)


def test_plan_requires_observed_current_event_in_every_artifact(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    artifact = paths["failed"] / "materialized_oof" / "fold_2_holdout.pt"
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    del payload["batches"][0]["batch"]["current_event_id"]
    torch.save(payload, artifact)
    manifest = resume.load_json(paths["manifest"])
    unsigned = {key: value for key, value in manifest.items() if key != "materialization_sha256"}
    unsigned["folds"][2]["holdout_artifact_sha256"] = resume.sha256_path(artifact)
    _write_json(paths["manifest"], _signed(unsigned, "materialization_sha256"))
    with pytest.raises(RuntimeError, match="lacks observed current_event_id"):
        _build(paths)


def test_plan_fails_when_lbfgs_checkpoint_or_failed_state_changes(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    checkpoint = paths["failed"] / "lbfgs_convex" / "fold_3.pt"
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value["optimizer"]["name"] = "AdamW"
    torch.save(value, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint authentication failed"):
        _build(paths)

    paths = _make_fixture(tmp_path / "second")
    state = resume.load_json(paths["state"])
    state["current_stage"] = "train_adamw_fixed_fold_0"
    _write_json(paths["state"], state)
    with pytest.raises(RuntimeError, match="expected failed LBFGS bridge"):
        _build(paths)


def test_full_path_boundary_guard() -> None:
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        resume._reject_fresh_path(
            Path("/srv/Fresh50/unrelated/r3_resume"), role="output"
        )
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        resume._reject_fresh_path(
            Path("/srv/archive/confirmation/r3_resume"), role="output"
        )
