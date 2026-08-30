from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_robotwin2_five_body_lobo_to_paired_success_v1.py"
SPEC = importlib.util.spec_from_file_location("lobo_to_paired_watcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
watcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watcher
SPEC.loader.exec_module(watcher)

PREREG_SCRIPT = (
    ROOT / "scripts/preregister_robotwin2_move_can_pot_five_body_lobo_v1.py"
)
PREREG_SPEC = importlib.util.spec_from_file_location(
    "paired_metrics_preregistration_v2", PREREG_SCRIPT
)
assert PREREG_SPEC is not None and PREREG_SPEC.loader is not None
metrics_prereg = importlib.util.module_from_spec(PREREG_SPEC)
sys.modules[PREREG_SPEC.name] = metrics_prereg
PREREG_SPEC.loader.exec_module(metrics_prereg)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _paths(tmp_path: Path) -> watcher.FormalPaths:
    home = tmp_path / "home"
    code = tmp_path / "code"
    prefix = home / "paired"
    return watcher.FormalPaths(
        home=home,
        code_root=code,
        upstream_state=home / "upstream.state.json",
        upstream_run_exit=home / "upstream.run.exit",
        lobo_root=home / "lobo",
        actor_checkpoint=home / "actor",
        vlm_metadata=home / "vlm",
        robotwin_root=home / "RoboTwin",
        event_spec=home / "analytic_event_spec.json",
        materialization_receipt=home / "materialization_v1.json",
        metrics_preregistration=home / "metrics_preregistration_v2.json",
        output_root=prefix,
        state=home / "paired.state.json",
        run_exit=home / "paired.run.exit",
        pid=home / "paired.pid",
        watcher_log=home / "paired.log",
        instance_lock=home / "paired.instance.lock",
        gpu_lock=home / "paired.gpu.lock",
        runner_python=home / "runner-python",
        evaluator_python=home / "evaluator-python",
        lerobot_root=home / "lerobot",
        lerobot_site=home / "lerobot-site",
        robotwin_eval_site=home / "robotwin-eval-site",
    )


def _complete_upstream(paths: watcher.FormalPaths) -> dict[str, object]:
    fold_rows = []
    for body in watcher.BODIES:
        root = paths.fold_root(body)
        root.mkdir(parents=True)
        members = []
        for member in range(watcher.EXPECTED_MEMBERS_PER_FOLD):
            checkpoint = root / f"member_{member}.pt"
            checkpoint.write_bytes(f"{body}:{member}".encode())
            members.append(
                {
                    "member": member,
                    "seed": 20260901 + member,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha(checkpoint),
                }
            )
        summary_path = root / "training_summary.json"
        _json(
            summary_path,
            {
                "format": watcher.FOLD_FORMAT,
                "status": watcher.FOLD_STATUS,
                "held_out_body": body,
                "event_spec_sha256": watcher.EXPECTED_EVENT_SPEC_SHA256,
                "event_derivation_implementation_sha256": (
                    watcher.EXPECTED_EVENT_MODULE_SHA256
                ),
                "heldout_labels_used_for_normalization_training_or_selection": False,
                "heldout_specific_trainable_parameters": 0,
                "actor_frozen": True,
                "members": members,
            },
        )
        fold_rows.append(
            {
                "held_out_body": body,
                "training_summary": str(summary_path),
                "training_summary_file_sha256": _sha(summary_path),
                "member_count": watcher.EXPECTED_MEMBERS_PER_FOLD,
            }
        )
    _json(
        paths.final_summary,
        {
            "format": watcher.UPSTREAM_FINAL_FORMAT,
            "status": watcher.UPSTREAM_FINAL_STATUS,
            "fold_count": len(watcher.BODIES),
            "members_per_fold": watcher.EXPECTED_MEMBERS_PER_FOLD,
            "heldout_task_success_measured": False,
            "cross_embodiment_task_success_claim_authorized": False,
            "outer_folds": fold_rows,
        },
    )
    paths.upstream_run_exit.write_text("0\n", encoding="utf-8")
    state = {
        "format": watcher.UPSTREAM_FORMAT,
        "status": "complete",
        "output_root": str(paths.lobo_root),
        "actor_checkpoint": str(paths.actor_checkpoint),
        "completed_folds": list(watcher.BODIES),
        "final_summary": str(paths.final_summary),
        "final_summary_file_sha256": _sha(paths.final_summary),
        "updated_at_utc": "2026-08-30T00:00:00+00:00",
    }
    _json(paths.upstream_state, state)
    return state


def test_probe_waits_for_real_upstream_complete(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    ready, progress = watcher.probe_upstream(paths)
    assert ready is None
    assert progress == {"upstream_state_present": False}

    _json(
        paths.upstream_state,
        {
            "format": watcher.UPSTREAM_FORMAT,
            "status": "training_fold",
            "updated_at_utc": "2026-08-30T00:00:00+00:00",
        },
    )
    ready, progress = watcher.probe_upstream(paths)
    assert ready is None
    assert progress["upstream_status"] == "training_fold"


def test_true_complete_requires_aggregate_all_folds_and_25_checkpoints(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    state = _complete_upstream(paths)
    audit = watcher.validate_upstream_complete(paths, state)
    assert list(audit["folds"]) == list(watcher.BODIES)
    assert sum(len(row["members"]) for row in audit["folds"].values()) == 25

    checkpoint = paths.fold_root("piper") / "member_3.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(watcher.PairedWatcherError, match="checkpoint SHA mismatch"):
        watcher.validate_upstream_complete(paths, state)


def test_upstream_failure_is_not_silently_waited(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _json(paths.upstream_state, {"status": "failed", "error": "training exit 9"})
    with pytest.raises(watcher.PairedWatcherError, match="training exit 9"):
        watcher.probe_upstream(paths)


def test_runner_command_has_only_the_fixed_formal_inputs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    command = watcher.build_runner_command(paths)
    assert command[:2] == [str(paths.runner_python), str(paths.runner)]
    assert command.count("--lobo-fold") == 5
    for body in watcher.BODIES:
        assert f"{body}={paths.fold_root(body)}" in command
    assert command[command.index("--actor-checkpoint") + 1] == str(
        paths.actor_checkpoint
    )
    assert command[command.index("--event-spec") + 1] == str(paths.event_spec)
    assert command[command.index("--preregistration") + 1] == str(
        paths.metrics_preregistration
    )
    assert command[command.index("--output") + 1] == str(paths.output_root)
    assert command[-6:] == [
        "--action-exec-steps",
        "5",
        "--max-steps",
        "200",
        "--fps",
        "15",
    ]


def test_gpu_wait_does_not_reserve_while_busy_and_requires_two_idle_audits() -> None:
    busy = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090 D",
        "uuid": watcher.EXPECTED_GPU_UUID,
        "compute_pids": [42],
    }
    idle = {**busy, "compute_pids": []}
    observations = iter((busy, idle, idle))
    states: list[tuple[str, dict[str, object]]] = []
    sleeps: list[float] = []

    audits = watcher.wait_for_idle_gpu(
        poll_seconds=60.0,
        confirmation_seconds=5.0,
        state_writer=lambda status, **extra: states.append((status, extra)),
        gpu_reader=lambda: next(observations),
        sleep=sleeps.append,
    )

    assert audits == [idle, idle]
    assert sleeps == [60.0, 5.0]
    assert states[0][0] == "waiting_for_authorized_idle_rtx4090"
    assert states[0][1]["gpu_reserved_by_watcher"] is False
    assert states[1][0] == "confirming_authorized_idle_rtx4090"
    assert states[1][1]["gpu_reserved_by_watcher"] is False


def test_report_validation_binds_outcomes_and_canonical_sha(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.output_root.mkdir(parents=True)
    outcomes_sha = "a" * 64
    base = {
        "format": watcher.REPORT_FORMAT,
        "status": watcher.REPORT_STATUS,
        "pair_count": watcher.EXPECTED_PAIRS,
        "planned_rollout_count": watcher.EXPECTED_ROLLOUTS,
        "input_binding": {"input_file_sha256": outcomes_sha},
        "prospective_improvement_gate": {"passed": False},
    }
    _json(paths.report, {**base, "report_sha256": watcher.canonical_sha256(base)})
    assert watcher.validate_report(paths, outcomes_sha)["pair_count"] == 1_000

    changed = json.loads(paths.report.read_text(encoding="utf-8"))
    changed["pair_count"] = 999
    _json(paths.report, changed)
    with pytest.raises(watcher.PairedWatcherError, match="report contract"):
        watcher.validate_report(paths, outcomes_sha)


def test_analytic_and_execution_code_identities_are_frozen() -> None:
    assert watcher.EXPECTED_EVENT_SPEC_SHA256 == (
        "4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee"
    )
    assert watcher.EXPECTED_EVENT_MODULE_SHA256 == (
        "d236036e4121232391808743a957e8ae94722ea89df223d123f8a77296f9e6d9"
    )
    assert watcher.EXPECTED_RUNNER_SHA256 == (
        "049017f53c0f9a3e462ea29db7d351075cc6f3d427f5c63a851fd1a154db9093"
    )
    assert watcher.EXPECTED_EVALUATOR_SHA256 == (
        "6e0f2a9b370f6c8fb66caf8c01e55747f4b882ced3657a1a2b32346d9bda9984"
    )


def test_metrics_preregistration_v2_is_distinct_from_materialization_v1(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    value = metrics_prereg.build_preregistration()
    assert value["preregistration_sha256"] == (
        watcher.EXPECTED_METRICS_PREREGISTRATION_SHA256
    )
    _json(paths.metrics_preregistration, value)
    audit = watcher.validate_metrics_preregistration(paths.metrics_preregistration)
    assert audit["role"].startswith("prospective_paired_execution_and_metrics")
    assert audit["preregistration_sha256"] != (
        watcher.EXPECTED_MATERIALIZATION_PREREGISTRATION_SHA256
    )
    assert paths.materialization_receipt != paths.metrics_preregistration


def test_completion_receipt_closes_contract_pair_and_outcome_chain(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.output_root.mkdir(parents=True)
    contract_base = {
        "candidate_rank_ensemble_contract": (
            watcher.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT
        )
    }
    contract = {
        **contract_base,
        "logical_sha256": watcher.canonical_sha256(contract_base),
    }
    _json(paths.execution_contract, contract)
    outcomes_base = {
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": _sha(paths.execution_contract),
        "ordered_pair_sha256s_sha256": "a" * 64,
    }
    outcomes = {
        **outcomes_base,
        "document_sha256": watcher.canonical_sha256(outcomes_base),
    }
    _json(paths.outcomes, outcomes)
    receipt_base = {
        "format": "etsf_robotwin2_paired_execution_completion_receipt_v1",
        "status": "complete_1000_pairs_2000_rollouts_frozen",
        "execution_contract_path": str(paths.execution_contract),
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": _sha(paths.execution_contract),
        "candidate_rank_ensemble_contract": (
            watcher.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT
        ),
        "pair_count": watcher.EXPECTED_PAIRS,
        "rollout_count": watcher.EXPECTED_ROLLOUTS,
        "ordered_pair_sha256s_sha256": outcomes["ordered_pair_sha256s_sha256"],
        "outcome_path": str(paths.outcomes),
        "outcome_document_sha256": outcomes["document_sha256"],
        "outcome_file_sha256": _sha(paths.outcomes),
    }
    receipt = {
        **receipt_base,
        "logical_sha256": watcher.canonical_sha256(receipt_base),
    }
    _json(paths.completion_receipt, receipt)
    assert watcher.validate_completion_receipt(paths)["pair_count"] == 1000

    receipt["ordered_pair_sha256s_sha256"] = "b" * 64
    receipt_unsigned = dict(receipt)
    receipt_unsigned.pop("logical_sha256")
    receipt["logical_sha256"] = watcher.canonical_sha256(receipt_unsigned)
    _json(paths.completion_receipt, receipt)
    with pytest.raises(watcher.PairedWatcherError):
        watcher.validate_completion_receipt(paths)


def test_waiting_watcher_has_no_torch_numpy_or_simulator_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source
    assert "from envs" not in source
