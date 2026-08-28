from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STUB = Path(__file__).resolve().parent / "fixtures" / "transfer_stage_stub.py"
sys.path.insert(0, str(SCRIPTS))

import launch_etsf_post_openvla_transfer as launcher  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _status(role: str, *, forbid_validation: bool) -> str:
    if role == "validate_and_freeze":
        return "confirmation_forbidden" if forbid_validation else "confirmation_authorized"
    return launcher.EXPECTED_STATUS[role]


def _plan(tmp_path: Path, *, forbid_validation: bool = True) -> dict[str, object]:
    upstream = tmp_path / "upstream.json"
    protocol = tmp_path / "artifacts" / "freeze_transfer_protocol.json"
    stages = []
    for role in launcher.REQUIRED_ROLES:
        artifact = tmp_path / "artifacts" / f"{role}.json"
        argv = [
            str(Path(sys.executable).resolve()),
            str(STUB.resolve()),
            "--artifact",
            str(artifact.resolve()),
            "--status",
            _status(role, forbid_validation=forbid_validation),
            "--labels-read",
            "false" if role in launcher.LABEL_FREE_ROLES else "true",
        ]
        if role.startswith("train_transfer_n"):
            argv += ["--n-per-task", role.removeprefix("train_transfer_n")]
        elif role in {
            "train_actor_hidden_observer_n20",
            "evaluate_privileged_pose_upper_bound_n20",
            "train_target_from_scratch_n20",
            "train_no_factorization_n20",
            "train_full_finetune_upper_n20",
        }:
            argv += ["--n-per-task", "20"]
        stages.append(
            {
                "role": role,
                "argv": argv,
                "gpu": role in launcher.GPU_ROLES,
                "command_artifacts": [
                    {"path": str(STUB.resolve()), "sha256": _sha(STUB)}
                ],
            }
        )
    return {
        "format": launcher.PLAN_FORMAT,
        "study_id": "smolvla_same_body_transfer_v1",
        "axis": "policy",
        "upstream": {
            "state_path": str(upstream.resolve()),
            "format": launcher.UPSTREAM_FORMAT,
            "required_status": launcher.UPSTREAM_READY,
            "forbidden_status": launcher.UPSTREAM_NO_CONFIRMATION,
        },
        "protocol_output": str(protocol.resolve()),
        "gpu": {
            "index": 0,
            "required_name": "RTX 4090",
            "wait_timeout_seconds": 60,
            "poll_seconds": 1,
        },
        "stages": stages,
        "fresh_label_access": "terminal_watcher_state_only",
    }


def test_plan_freezes_all_n_and_matched_baselines(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    launcher.validate_plan(plan)
    roles = [stage["role"] for stage in plan["stages"]]  # type: ignore[index]
    assert roles == list(launcher.REQUIRED_ROLES)
    assert [role for role in roles if role.startswith("train_transfer_n")] == [
        "train_transfer_n0",
        "train_transfer_n5",
        "train_transfer_n10",
        "train_transfer_n20",
        "train_transfer_n50",
    ]


def test_plan_rejects_unfrozen_adaptation_size(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    stage = next(
        row for row in plan["stages"] if row["role"] == "train_transfer_n20"  # type: ignore[index]
    )
    index = stage["argv"].index("--n-per-task")  # type: ignore[index]
    stage["argv"][index + 1] = "25"  # type: ignore[index]
    with pytest.raises(ValueError, match="does not freeze its N"):
        launcher.validate_plan(plan)


def test_upstream_without_confirmation_never_runs_transfer(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    _write(
        Path(plan["upstream"]["state_path"]),  # type: ignore[index]
        {
            "format": launcher.UPSTREAM_FORMAT,
            "status": launcher.UPSTREAM_NO_CONFIRMATION,
        },
    )
    state = launcher.execute(
        plan,
        state_root=tmp_path / "state",
        upstream_timeout=1,
        upstream_poll=0.01,
        gpu_waiter=lambda *_: {"name": "fake"},
    )
    assert state["status"] == (
        "complete_openvla_confirmation_not_available_transfer_not_started"
    )
    assert state["stages"] == []
    assert state["openvla_confirmation_labels_read"] is False


def test_failed_asset_preflight_stops_before_vocabulary_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    _write(
        Path(plan["upstream"]["state_path"]),  # type: ignore[index]
        {"format": launcher.UPSTREAM_FORMAT, "status": launcher.UPSTREAM_READY},
    )

    def reject(_value: object) -> None:
        raise ValueError("mixed policy/body shift")

    monkeypatch.setattr(launcher, "validate_preflight", reject)
    gpu_calls: list[object] = []
    with pytest.raises(ValueError, match="mixed policy/body shift"):
        launcher.execute(
            plan,
            state_root=tmp_path / "state",
            upstream_timeout=1,
            upstream_poll=0.01,
            gpu_waiter=lambda *args: gpu_calls.append(args) or {"name": "fake"},
        )
    state = json.loads((tmp_path / "state" / "pipeline_state.json").read_text())
    assert state["status"] == "failed_closed"
    assert [stage["role"] for stage in state["stages"]] == ["asset_preflight"]
    assert gpu_calls == []


def test_validation_forbidden_prevents_confirmation_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, forbid_validation=True)
    _write(
        Path(plan["upstream"]["state_path"]),  # type: ignore[index]
        {"format": launcher.UPSTREAM_FORMAT, "status": launcher.UPSTREAM_READY},
    )
    monkeypatch.setattr(launcher, "validate_protocol", lambda value: None)
    state = launcher.execute(
        plan,
        state_root=tmp_path / "state",
        upstream_timeout=1,
        upstream_poll=0.01,
        gpu_waiter=lambda *_: {"name": "fake-4090", "compute_pids": []},
    )
    assert state["status"] == (
        "complete_target_validation_gate_forbidden_confirmation_not_run"
    )
    completed = [stage["role"] for stage in state["stages"]]
    assert completed[-1] == "validate_and_freeze"
    assert "run_paired_confirmation" not in completed
    assert "build_transfer_result_summary" not in completed
    assert state["openvla_confirmation_labels_read"] is False


def test_command_artifact_hash_change_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    broken = copy.deepcopy(plan)
    broken["stages"][0]["command_artifacts"][0]["sha256"] = "0" * 64  # type: ignore[index]
    launcher.validate_plan(broken)
    with pytest.raises(RuntimeError, match="command artifact changed"):
        launcher.verify_command_artifacts(broken["stages"][0])  # type: ignore[index]
