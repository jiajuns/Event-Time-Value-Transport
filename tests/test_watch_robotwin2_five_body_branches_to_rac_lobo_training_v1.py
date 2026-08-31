from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_robotwin2_five_body_lobo_relative_action_critic_v1 as rac
import watch_robotwin2_five_body_branches_to_rac_lobo_training_v1 as watcher


def _args(tmp_path: Path, *, supplement: bool = False) -> argparse.Namespace:
    supplement_path = tmp_path / "supplement.json" if supplement else None
    return argparse.Namespace(
        branches_root=tmp_path / "branches",
        actor_checkpoint=tmp_path / "actor",
        materialization_receipt=tmp_path / "materialization.json",
        actor_authority=tmp_path / "actor_authority.json",
        binding=tmp_path / "binding.json",
        supplement_binding=supplement_path,
        supplement_binding_sha256="a" * 64 if supplement else None,
        upstream_run_exit=None,
        upstream_state=None,
        output_root=tmp_path / "rac",
        state=tmp_path / "state.json",
        run_exit=tmp_path / "exit",
        trainer=SCRIPTS
        / "train_robotwin2_five_body_lobo_relative_action_critic_v1.py",
        training_python=Path(sys.executable),
        poll_seconds=0.01,
        expected_gpu_uuid="test",
    )


def _summary_with_checkpoints(output: Path, heldout: str) -> dict:
    output.mkdir(parents=True)
    members = []
    for member, seed in enumerate(rac.DEFAULT_ENSEMBLE_SEEDS):
        checkpoint = output / f"member_{member}.pt"
        checkpoint.write_bytes(f"checkpoint-{member}".encode())
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": 3000,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": watcher.base.sha256_file(checkpoint),
            }
        )
    summary = {
        "format": rac.SUMMARY_FORMAT,
        "held_out_body": heldout,
        "members": members,
    }
    summary["logical_sha256"] = rac.canonical_sha256(summary)
    watcher.base.atomic_json(output / "training_summary.json", summary)
    return summary


def _rewrite_signed(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("logical_sha256", None)
    value["logical_sha256"] = watcher.base.canonical_sha256(value)
    watcher.base.atomic_json(path, value)


def test_fold_command_is_frozen_five_by_3000_and_supplement_bound(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, supplement=True)
    output = tmp_path / "out"
    command = watcher.fold_training_command(
        args, "franka", "b" * 64, output
    )
    assert command[0] == sys.executable
    assert command[command.index("--steps") + 1] == "3000"
    assert command[command.index("--batch-size-pairs") + 1] == "96"
    seeds = command[command.index("--ensemble-seeds") + 1 :]
    assert seeds[:5] == [str(value) for value in rac.DEFAULT_ENSEMBLE_SEEDS]
    assert command[command.index("--supplement-binding-sha256") + 1] == "a" * 64


def test_create_once_attempt_resume_and_signed_rac_rebinding(tmp_path: Path) -> None:
    args = _args(tmp_path)
    attempt = watcher.create_fold_attempt(args, "franka", "b" * 64)
    Path(attempt["log"]).write_text("complete\n", encoding="utf-8")
    original = _summary_with_checkpoints(Path(attempt["training_output"]), "franka")
    final_output = args.output_root / "outer_lobo_franka"

    receipt = watcher.rebind_fold_summary_for_atomic_promotion(
        attempt, final_output
    )
    rebound = json.loads(
        (Path(attempt["training_output"]) / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["original_training_summary_logical_sha256"] == original[
        "logical_sha256"
    ]
    assert rebound["logical_sha256"] == rac.canonical_sha256(
        {key: value for key, value in rebound.items() if key != "logical_sha256"}
    )
    assert all(
        Path(member["checkpoint"]).is_relative_to(final_output)
        for member in rebound["members"]
    )

    promotion = watcher.promote_fold_attempt(attempt, final_output)
    history = watcher.validate_fold_attempt_history(args, "franka", "b" * 64)
    assert len(history) == 1
    assert history[0]["promotion"]["logical_sha256"] == promotion[
        "logical_sha256"
    ]


def test_missing_promotion_receipt_is_recovered_without_retraining(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    attempt = watcher.create_fold_attempt(args, "ur5", "b" * 64)
    Path(attempt["log"]).write_text("complete\n", encoding="utf-8")
    _summary_with_checkpoints(Path(attempt["training_output"]), "ur5")
    final_output = args.output_root / "outer_lobo_ur5"
    watcher.rebind_fold_summary_for_atomic_promotion(attempt, final_output)
    Path(attempt["training_output"]).rename(final_output)

    history = watcher.validate_fold_attempt_history(args, "ur5", "b" * 64)
    recovered = watcher.recover_missing_promotion_receipt(history, final_output)
    assert recovered["promotion_receipt_recovered_after_interruption"] is True
    history = watcher.validate_fold_attempt_history(args, "ur5", "b" * 64)
    assert history[0]["promotion"]["logical_sha256"] == recovered["logical_sha256"]


def test_attempt_command_tamper_fails_closed_even_when_resigned(tmp_path: Path) -> None:
    args = _args(tmp_path)
    attempt = watcher.create_fold_attempt(args, "piper", "b" * 64)
    manifest = Path(attempt["directory"]) / "attempt.json"
    _rewrite_signed(manifest, lambda value: value["command"].extend(["--steps", "1"]))
    with pytest.raises(watcher.RacLoboSupervisorError, match="manifest changed"):
        watcher.validate_fold_attempt_history(args, "piper", "b" * 64)


def test_checkpoint_tamper_fails_before_atomic_promotion(tmp_path: Path) -> None:
    args = _args(tmp_path)
    attempt = watcher.create_fold_attempt(args, "arx-x5", "b" * 64)
    _summary_with_checkpoints(Path(attempt["training_output"]), "arx-x5")
    checkpoint = Path(attempt["training_output"]) / "member_2.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(watcher.RacLoboSupervisorError, match="missing or changed"):
        watcher.rebind_fold_summary_for_atomic_promotion(
            attempt, args.output_root / "outer_lobo_arx-x5"
        )


def test_normalized_args_accepts_not_yet_materialized_binding_paths(
    tmp_path: Path,
) -> None:
    args = watcher.normalized_args(_args(tmp_path, supplement=True))
    assert not args.binding.exists()
    assert not args.supplement_binding.exists()
    assert args.binding.is_absolute()
    assert args.supplement_binding.is_absolute()


def test_deferred_supplement_sha_freezes_only_after_protocol_validation(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, supplement=True)
    args.supplement_binding_sha256 = None
    args = watcher.normalized_args(args)
    args.output_root.mkdir(parents=True)
    protocol = watcher.base.actor_execution.execution_protocol(5)
    protocol_binding = {
        "format": "test_protocol_binding",
        "path_root": str(tmp_path.resolve()),
        "path": "protocol.json",
        "file_sha256": "c" * 64,
        "protocol_logical_sha256": protocol["logical_sha256"],
        "protocol": protocol,
    }
    watcher.base._ACTIVE_EXECUTION_PROTOCOL = protocol
    watcher.base._ACTIVE_EXECUTION_PROTOCOL_BINDING = protocol_binding

    assert watcher.freeze_or_validate_supplement_authority(args) is None
    binding = {
        "state_action_frame_contract": watcher.base.STATE_ACTION_FRAME_CONTRACT,
        "actor_execution_protocol": protocol,
        "actor_execution_protocol_binding": protocol_binding,
        "actor_execution_protocol_file_sha256": protocol_binding["file_sha256"],
    }
    binding["logical_sha256"] = watcher.base.canonical_sha256(binding)
    watcher.base.atomic_json(args.supplement_binding, binding)
    authority = watcher.freeze_or_validate_supplement_authority(args)
    assert authority["binding_validated_before_file_sha_freeze"] is True
    assert authority["heldout_manifest_or_payload_opened"] == 0
    assert args.supplement_binding_sha256 == watcher.base.sha256_file(
        args.supplement_binding
    )

    binding["new_field"] = "tamper"
    binding.pop("logical_sha256")
    binding["logical_sha256"] = watcher.base.canonical_sha256(binding)
    watcher.base.atomic_json(args.supplement_binding, binding)
    with pytest.raises(watcher.RacLoboSupervisorError, match="explicit SHA mismatch"):
        watcher.freeze_or_validate_supplement_authority(args)


def test_upstream_idle_gap_cannot_authorize_rac_before_complete_state(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.upstream_run_exit = tmp_path / "v13_crossbody.run.exit"
    args.upstream_state = tmp_path / "v13_crossbody.watcher_state.json"
    args = watcher.normalized_args(args)
    args.output_root.mkdir(parents=True)
    args.upstream_state.write_text(
        json.dumps({"format": "upstream", "status": "running"}),
        encoding="utf-8",
    )
    # This helper never consults transient GPU idleness: only the complete
    # upstream receipt can authorize RAC fold command/attempt creation.
    assert watcher.freeze_or_validate_upstream_authority(args) is None
    args.upstream_run_exit.write_text("0\n", encoding="utf-8")
    assert watcher.freeze_or_validate_upstream_authority(args) is None


def test_upstream_complete_report_is_frozen_and_tamper_fails(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.upstream_run_exit = tmp_path / "v13_crossbody.run.exit"
    args.upstream_state = tmp_path / "v13_crossbody.watcher_state.json"
    args = watcher.normalized_args(args)
    args.output_root.mkdir(parents=True)
    report_path = tmp_path / "nested_report.json"
    report = {"format": "test_nested_report", "status": "complete"}
    report["report_sha256"] = watcher.base.canonical_sha256(report)
    watcher.base.atomic_json(report_path, report)
    state = {
        "format": "test_v13_crossbody_state",
        "status": "complete",
        "nested_actor_n4_n8_report": str(report_path),
        "nested_actor_n4_n8_report_file_sha256": watcher.base.sha256_file(
            report_path
        ),
    }
    watcher.base.atomic_json(args.upstream_state, state)
    args.upstream_run_exit.write_text("0\n", encoding="utf-8")
    authority = watcher.freeze_or_validate_upstream_authority(args)
    assert authority["upstream_completed_before_rac_training"] is True
    assert authority["final_report_logical_sha256"] == report["report_sha256"]

    report["status"] = "tampered"
    report.pop("report_sha256")
    report["report_sha256"] = watcher.base.canonical_sha256(report)
    watcher.base.atomic_json(report_path, report)
    with pytest.raises(watcher.RacLoboSupervisorError, match="file SHA changed"):
        watcher.freeze_or_validate_upstream_authority(args)
