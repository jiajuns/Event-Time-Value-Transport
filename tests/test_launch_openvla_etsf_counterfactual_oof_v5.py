from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_counterfactual_oof_v5 as launcher  # noqa: E402
from openvla_etsf_counterfactual_oof import make_oof_folds  # noqa: E402


def args(tmp_path: Path, output: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        data=tmp_path / "data",
        pretrained=tmp_path / "factual.pt",
        event_spec=tmp_path / "event_spec.json",
        output=output or tmp_path / "oof_output",
        trainer=SCRIPTS / "train_openvla_etsf_counterfactual_oof.py",
        python_bin=Path(sys.executable),
        gpu_index=0,
        gpu_lock=tmp_path / "gpu.lock",
        num_workers=0,
        dry_run=False,
    )


def minimal_plan(output: Path) -> dict:
    stages = ["preregister", *[f"fold_{index}" for index in range(5)], "select", "final"]
    return {
        "format": launcher.LAUNCH_FORMAT,
        "status": "preflight_complete",
        "output_root": str(output),
        "plan_sha256": "a" * 64,
        "commands": [
            {
                "stage": stage,
                "argv": ["python", stage],
                "argv_sha256": stage,
                "uses_gpu": stage.startswith("fold_") or stage == "final",
            }
            for stage in stages
        ],
    }


def test_stage_commands_are_strictly_serial_and_final_is_last(tmp_path: Path) -> None:
    value = args(tmp_path)
    commands = launcher.build_stage_commands(value)
    assert [row["stage"] for row in commands] == [
        "preregister",
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "select",
        "final",
    ]
    assert [row["uses_gpu"] for row in commands] == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
    ]
    joined = " ".join(argument for row in commands for argument in row["argv"])
    assert "fresh" not in joined.lower()
    assert str(value.output / "final") in commands[-1]["argv"]


def test_factual_path_policy_is_canonicalized_to_openvla(tmp_path: Path) -> None:
    checkpoint = tmp_path / "factual.pt"
    torch.save(
        {
            "config": {"structured_events": True},
            "contract": {
                "policy_to_id": {
                    "/home/user/checkpoints/openvla-oft-7b-robotwin": 0
                }
            },
        },
        checkpoint,
    )
    audit = launcher.canonical_factual_policy_audit(checkpoint)
    assert audit["canonical_policy_to_id"] == {"openvla": 0}
    assert audit["canonical_openvla_id"] == 0


def test_factual_policy_alias_collision_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "factual.pt"
    torch.save(
        {
            "config": {"structured_events": True},
            "contract": {
                "policy_to_id": {
                    "openvla": 0,
                    "/models/openvla-oft": 1,
                }
            },
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="aliases collide"):
        launcher.canonical_factual_policy_audit(checkpoint)


def test_select_stage_requires_signed_structured_prediction_diagnostics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "oof"
    output.mkdir()
    manifest = make_oof_folds(
        [f"move_can_pot|piper|{index}" for index in range(100)]
    )
    (output / "oof_folds.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    selection = {
        "format": launcher.SELECTION_FORMAT,
        "status": "complete",
        "oof_prediction_groups": 100,
        "authorization": {
            "total_oof_groups": 100,
            "authorized": False,
            "rejection_reasons": ["test"],
        },
    }
    selection["selection_sha256"] = launcher.canonical_sha256(selection)
    (output / "oof_selection.json").write_text(
        json.dumps(selection, sort_keys=True), encoding="utf-8"
    )
    diagnostics = {
        "format": "etsf_oof_heldout_prediction_diagnostics_v1",
        "status": "complete",
        "oof_preregistration_sha256": manifest["preregistration_sha256"],
        "oof_groups": 100,
        "fold_count": 5,
        "heldout_groups_per_fold": 20,
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
        "diagnostics_are_descriptive_not_an_authorization_or_confirmation_gate": True,
        "success_probability": {
            "candidate_scope": "deployment_exact_first_four_only"
        },
        "success_probability_all_collected_candidates_appendix": {
            "excluded_from_main_prediction_adequacy": True
        },
        "structured_world_model": {"status": "complete"},
        "prediction_adequacy": {
            "protocol": "etsf_development_prediction_adequacy_v1",
            "independent_of_reranking_authorization_guard": True,
            "fresh50_authorization_effect": "none",
        },
    }
    diagnostics["diagnostics_sha256"] = launcher.canonical_sha256(diagnostics)
    (output / "oof_prediction_diagnostics.json").write_text(
        json.dumps(diagnostics, sort_keys=True), encoding="utf-8"
    )
    audit = launcher.validate_stage("select", output)
    assert audit["authorized"] is False
    assert audit["prediction_diagnostics_sha256"] == launcher.sha256(
        output / "oof_prediction_diagnostics.json"
    )


def test_lock_is_exclusive_and_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "launch.lock"
    launcher.acquire_lock(path, {"pid": 1, "plan_sha256": "a"})
    original = path.read_text()
    with pytest.raises(RuntimeError, match="lock exists"):
        launcher.acquire_lock(path, {"pid": 2, "plan_sha256": "b"})
    assert path.read_text() == original


def test_unauthorized_selection_stops_before_final(monkeypatch, tmp_path: Path) -> None:
    value = args(tmp_path)
    plan = minimal_plan(value.output)
    called = []

    def fake_stage(stage, **kwargs):
        del kwargs
        called.append(stage["stage"])
        if stage["stage"] == "select":
            return {
                "status": "complete",
                "authorized": False,
                "rejection_reasons": ["insufficient_total_oracle_headroom"],
            }
        return {"status": "complete"}

    monkeypatch.setattr(launcher, "run_stage", fake_stage)
    monkeypatch.setattr(
        launcher,
        "require_exclusive_idle_gpu",
        lambda index: {"gpu_index": index, "compute_pids": []},
    )
    state = launcher.execute_plan(value, plan)
    assert called[-1] == "select"
    assert "final" not in called
    assert state["status"] == "stopped_guard_not_authorized"
    assert state["fresh_confirmation_policy"] == "forbidden"
    assert not value.gpu_lock.exists()
    persisted = launcher._json(value.output / "launch_state.json")
    assert persisted["status"] == "stopped_guard_not_authorized"


def test_authorized_plan_runs_one_stage_at_a_time_through_final(
    monkeypatch, tmp_path: Path
) -> None:
    value = args(tmp_path)
    plan = minimal_plan(value.output)
    called = []
    gpu_checks = []

    def fake_stage(stage, **kwargs):
        del kwargs
        called.append(stage["stage"])
        if stage["stage"] == "select":
            return {"status": "complete", "authorized": True}
        return {"status": "complete"}

    monkeypatch.setattr(launcher, "run_stage", fake_stage)
    monkeypatch.setattr(
        launcher,
        "require_exclusive_idle_gpu",
        lambda index: gpu_checks.append(index) or {"gpu_index": index},
    )
    state = launcher.execute_plan(value, plan)
    assert called == [row["stage"] for row in plan["commands"]]
    assert gpu_checks == [0] * 6
    assert state["status"] == "complete_fresh50_ready_one_shot"
    assert state["fresh_confirmation_policy"] == "one_shot_only"
    assert not value.gpu_lock.exists()
