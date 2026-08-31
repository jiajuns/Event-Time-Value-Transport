from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_robotwin2_rac_lobo_to_nested_success_v1 as watcher


def _json(path: Path, value: dict) -> None:
    watcher.rac_watch.base.atomic_json(path, value)


def _signed(value: dict) -> dict:
    return watcher.signed(value)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        rac_root=tmp_path / "rac",
        rac_state=tmp_path / "rac.state.json",
        rac_run_exit=tmp_path / "rac.run.exit",
        actor_checkpoint=tmp_path / "actor",
        vlm_metadata=tmp_path / "vlm",
        actor_authority=tmp_path / "actor_authority.json",
        robotwin_root=tmp_path / "robotwin",
        event_spec=tmp_path / "event.json",
        reference_preregistration=tmp_path / "prereg.json",
        actor_execution_protocol=tmp_path / "protocol.json",
        actor_execution_protocol_sha256="a" * 64,
        path_root=tmp_path,
        output_root=tmp_path / "rac_nested",
        state=tmp_path / "rac_nested.state.json",
        run_exit=tmp_path / "rac_nested.run.exit",
        lock=tmp_path / "rac_nested.lock",
        nested_runner=SCRIPTS
        / "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py",
        final_materializer=SCRIPTS
        / "materialize_robotwin2_nested_n1_n4_n8_final_report_v1.py",
        robotwin_python=Path(sys.executable),
        system_python=Path(sys.executable),
        expected_gpu_uuid="test",
        poll_seconds=0.01,
    )


def _frozen_v13_authority(tmp_path: Path) -> dict:
    run_exit = tmp_path / "v13.run.exit"
    run_exit.write_text("0\n", encoding="utf-8")
    state_path = tmp_path / "v13.state.json"
    _json(state_path, {"format": "v13", "status": "complete"})
    report_path = tmp_path / "v13.report.json"
    report = {"format": "report", "status": "complete"}
    report["report_sha256"] = watcher.rac_watch.base.canonical_sha256(report)
    _json(report_path, report)
    return _signed(
        {
            "format": watcher.rac_watch.UPSTREAM_AUTHORITY_FORMAT,
            "enabled": True,
            "upstream_run_exit": str(run_exit),
            "upstream_run_exit_file_sha256": watcher.rac_watch.base.sha256_file(
                run_exit
            ),
            "upstream_state": str(state_path),
            "upstream_state_file_sha256": watcher.rac_watch.base.sha256_file(
                state_path
            ),
            "upstream_state_status": "complete",
            "final_report": str(report_path),
            "final_report_file_sha256": watcher.rac_watch.base.sha256_file(
                report_path
            ),
            "final_report_logical_sha_field": "report_sha256",
            "final_report_logical_sha256": report["report_sha256"],
            "upstream_completed_before_rac_training": True,
            "heldout_payloads_opened_while_waiting": 0,
            "frozen_at_utc": "2026-08-31T00:00:00+00:00",
        }
    )


def _complete_rac_tree(tmp_path: Path, monkeypatch) -> tuple[argparse.Namespace, dict]:
    args = _args(tmp_path)
    args.rac_root.mkdir()
    authority = _frozen_v13_authority(tmp_path)
    _json(args.rac_root / "upstream_completion_authority.json", authority)
    protocol = watcher.nested.formal.actor_execution.execution_protocol(5)
    _json(args.actor_execution_protocol, protocol)
    protocol_sha = watcher.rac_watch.base.sha256_file(
        args.actor_execution_protocol
    )
    args.actor_execution_protocol_sha256 = protocol_sha
    protocol_binding = (
        watcher.nested.formal.actor_execution.execution_protocol_file_binding(
            args.actor_execution_protocol,
            protocol_sha,
            path_root=tmp_path,
        )
    )
    actor_value = _signed(
        {
            "format": watcher.rac_watch.base.ACTOR_FORMAT,
            "one_universal_actor_for_all_five_bodies": True,
            "sampling_contract": {
                "actor_execution_protocol_binding": protocol_binding,
            },
        }
    )
    _json(args.actor_authority, actor_value)
    actor_binding = {
        "path": str(args.actor_authority),
        "file_sha256": watcher.rac_watch.base.sha256_file(args.actor_authority),
        "logical_sha256": actor_value["logical_sha256"],
        "checkpoint_sha256": "c" * 64,
    }
    rows = []
    receipts = {}
    for body in watcher.BODIES:
        fold = args.rac_root / f"outer_lobo_{body}"
        fold.mkdir()
        summary = _signed(
            {
                "format": watcher.rac_watch.rac.SUMMARY_FORMAT,
                "held_out_body": body,
                "actor_execution_protocol": protocol,
                "actor_execution_protocol_binding": protocol_binding,
                "actor_execution_protocol_file_sha256": protocol_sha,
                "heldout_rows_used_for_training_normalization_or_selection": 0,
                "all_checkpoints_selected_before_any_heldout_payload_open": True,
            }
        )
        summary_path = fold / "training_summary.json"
        _json(summary_path, summary)
        receipt_base = {
            "format": "etsf_rac_fold_ensemble_load_receipt_v1",
            "held_out_body": body,
            "source_bodies": [candidate for candidate in watcher.BODIES if candidate != body],
            "selected_step": 3000,
            "members": [],
            "heldout_payloads_or_labels_opened": 0,
        }
        receipt = {
            **receipt_base,
            "logical_sha256": watcher.rac_watch.base.canonical_sha256(receipt_base),
        }
        receipts[body] = receipt
        rows.append(
            {
                "held_out_body": body,
                "training_summary": str(summary_path),
                "training_summary_file_sha256": watcher.rac_watch.base.sha256_file(
                    summary_path
                ),
                "training_summary_logical_sha256": summary["logical_sha256"],
                "ensemble_load_receipt": receipt,
            }
        )
    aggregate = _signed(
        {
            "format": watcher.rac_watch.FINAL_FORMAT,
            "status": "five_outer_lobo_rac_source_only_training_complete",
            "model_family": watcher.rac_watch.rac.MODEL_FAMILY,
            "fold_count": 5,
            "members_per_fold": 5,
            "steps_per_member": 3000,
            "supplement_binding_file_sha256": "b" * 64,
            "actor_execution_protocol": protocol,
            "actor_execution_protocol_binding": protocol_binding,
            "actor_execution_protocol_file_sha256": protocol_sha,
            "actor_authority": actor_binding,
            "upstream_completion_authority": authority,
            "outer_folds": rows,
        }
    )
    aggregate_path = args.rac_root / "five_fold_rac_training_summary.json"
    _json(aggregate_path, aggregate)
    state = {
        "format": watcher.rac_watch.FORMAT,
        "status": "complete",
        "final_summary": str(aggregate_path),
        "final_summary_file_sha256": watcher.rac_watch.base.sha256_file(
            aggregate_path
        ),
        "upstream_completion_authority": authority,
    }
    _json(args.rac_state, state)
    args.rac_run_exit.write_text("0\n", encoding="utf-8")

    def load(path, *, device, expected_held_out_body):
        assert device.type == "cpu"
        return [], receipts[expected_held_out_body]

    monkeypatch.setattr(watcher.rac_adapter, "load_fold_ensemble", load)
    return args, aggregate


def test_waits_for_both_rac_exit_and_complete_state_without_opening_folds(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.rac_run_exit.write_text("0\n", encoding="utf-8")
    _json(args.rac_state, {"format": watcher.rac_watch.FORMAT, "status": "running"})
    assert watcher.validate_rac_supervisor_completion(args) is None


def test_complete_rac_chain_binds_frozen_v13_and_all_five_fold_summaries(
    tmp_path: Path, monkeypatch
) -> None:
    args, _aggregate = _complete_rac_tree(tmp_path, monkeypatch)
    audit = watcher.validate_rac_supervisor_completion(args)
    assert audit["required_supplement_binding_sha256"] == "b" * 64
    assert set(audit["folds"]) == set(watcher.BODIES)
    assert audit["heldout_payloads_or_labels_opened_by_watcher"] == 0

    summary = args.rac_root / f"outer_lobo_{watcher.BODIES[0]}" / "training_summary.json"
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["resigned_tamper"] = True
    value.pop("logical_sha256")
    value = _signed(value)
    _json(summary, value)
    with pytest.raises(watcher.RacNestedWatcherError, match="differs from aggregate"):
        watcher.validate_rac_supervisor_completion(args)


def test_normalized_args_accepts_missing_deferred_protocol_and_actor(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.actor_execution_protocol_sha256 = None
    for directory in (
        args.actor_checkpoint,
        args.vlm_metadata,
        args.robotwin_root,
    ):
        directory.mkdir()
    for path in (args.event_spec, args.reference_preregistration):
        path.write_text("{}\n", encoding="utf-8")
    normalized = watcher.normalized_args(args)
    assert normalized.actor_execution_protocol_sha256 is None
    assert not normalized.actor_execution_protocol.exists()
    assert not normalized.actor_authority.exists()


def test_deferred_protocol_and_actor_wait_then_freeze_without_gpu(
    tmp_path: Path, monkeypatch
) -> None:
    args, _aggregate = _complete_rac_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        watcher.rac_adapter,
        "load_fold_ensemble",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint payload opened during deferred wait")
        ),
    )
    completion = watcher.validate_rac_supervisor_completion(
        args, validate_checkpoints=False
    )
    assert completion["checkpoint_payloads_validated"] is False
    protocol_bytes = args.actor_execution_protocol.read_bytes()
    actor_bytes = args.actor_authority.read_bytes()
    args.actor_execution_protocol.unlink()
    args.actor_authority.unlink()
    args.actor_execution_protocol_sha256 = None
    assert watcher.freeze_or_validate_deferred_runtime_authority(
        args, completion
    ) is None

    args.actor_execution_protocol.write_bytes(protocol_bytes)
    assert watcher.freeze_or_validate_deferred_runtime_authority(
        args, completion
    ) is None
    args.actor_authority.write_bytes(actor_bytes)
    frozen = watcher.freeze_or_validate_deferred_runtime_authority(
        args, completion
    )
    assert frozen["files_authenticated_after_rac_completion"] is True
    assert frozen["heldout_payloads_opened_while_waiting"] == 0
    assert args.actor_execution_protocol_sha256 == frozen[
        "actor_execution_protocol_file_sha256"
    ]


def test_deferred_protocol_explicit_or_file_sha_mismatch_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    args, _aggregate = _complete_rac_tree(tmp_path, monkeypatch)
    completion = watcher.validate_rac_supervisor_completion(args)
    args.actor_execution_protocol_sha256 = "d" * 64
    with pytest.raises(watcher.RacNestedWatcherError, match="explicit actor protocol SHA"):
        watcher.freeze_or_validate_deferred_runtime_authority(args, completion)

    args.actor_execution_protocol_sha256 = None
    args.actor_execution_protocol.write_text("{}\n", encoding="utf-8")
    with pytest.raises(watcher.RacNestedWatcherError, match="file SHA"):
        watcher.freeze_or_validate_deferred_runtime_authority(args, completion)


def test_deferred_actor_authority_mismatch_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    args, _aggregate = _complete_rac_tree(tmp_path, monkeypatch)
    completion = watcher.validate_rac_supervisor_completion(args)
    actor = json.loads(args.actor_authority.read_text(encoding="utf-8"))
    actor["resigned_tamper"] = True
    actor.pop("logical_sha256")
    _json(args.actor_authority, _signed(actor))
    args.actor_execution_protocol_sha256 = None
    with pytest.raises(watcher.RacNestedWatcherError, match="actor authority differs"):
        watcher.freeze_or_validate_deferred_runtime_authority(args, completion)


def test_nested_command_is_explicit_rac_same_supplement_and_five_folds(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    completion = {
        "required_supplement_binding_sha256": "b" * 64,
        "folds": {
            body: {"root": str(tmp_path / "rac" / f"outer_lobo_{body}")}
            for body in watcher.BODIES
        },
    }
    command = watcher.nested_command(args, completion)
    assert command[command.index("--critic-kind") + 1] == "rac"
    assert command[command.index("--required-supplement-binding-sha256") + 1] == "b" * 64
    assert command.count("--lobo-fold") == 5
    assert command[command.index("--output") + 1] == str(
        args.output_root / "nested_rac"
    )


def test_execution_authority_is_create_once_resume_exact_and_tamper_closed(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.output_root.mkdir()
    authority = _signed(
        {
            "format": watcher.EXECUTION_AUTHORITY_FORMAT,
            "nested_command_sha256": "a" * 64,
        }
    )
    first = watcher.create_or_validate_execution_authority(args, authority)
    second = watcher.create_or_validate_execution_authority(args, authority)
    assert first == second
    changed = copy.deepcopy(authority)
    changed["nested_command_sha256"] = "b" * 64
    changed.pop("logical_sha256")
    changed = _signed(changed)
    with pytest.raises(watcher.RacNestedWatcherError, match="authority changed"):
        watcher.create_or_validate_execution_authority(args, changed)


def test_failure_receipt_is_signed_create_once_and_outcome_blind(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output_root.mkdir()
    value = watcher.record_failure(
        args,
        stage="nested_rac",
        authority={"logical_sha256": "a" * 64},
        returncode=2,
        error=None,
    )
    watcher.verify_signed(value, "failure")
    assert value["retry_selected_using_nested_outcomes"] is False
    observed = json.loads(
        (args.output_root / "failure_receipt.json").read_text(encoding="utf-8")
    )
    assert observed == value


def test_final_policy_report_sha_and_fail_closed_status(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    args.output_root.mkdir()
    input_value = _signed({"format": "materialized_rac_input"})
    _json(args.output_root / "rac_crossbody_final_report_input.json", input_value)
    report_base = {
        "format": watcher.materializer.evaluator.REPORT_FORMAT,
        "status": watcher.materializer.evaluator.POLICY_ONLY_STATUS,
        "input_document_sha256": input_value["logical_sha256"],
        "oracle_branch_diagnostic": {
            "evidence_sufficient": False,
            "oracle_regret_reported": False,
        },
    }
    report = {
        **report_base,
        "report_sha256": watcher.rac_watch.base.canonical_sha256(report_base),
    }
    _json(args.output_root / "rac_crossbody_final_report.json", report)
    monkeypatch.setattr(
        watcher.materializer,
        "build_materialization",
        lambda **_kwargs: (input_value, report),
    )
    audit = watcher.validate_final_report(args)
    assert audit["report_sha256"] == report["report_sha256"]
    assert audit["oracle_evidence_sufficient"] is False
