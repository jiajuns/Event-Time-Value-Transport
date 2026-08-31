from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_robotwin2_v13_rac_to_wcm_lobo_training_v1 as supervisor
import watch_robotwin2_wcm_lobo_to_nested_success_v1 as downstream


def _json(path: Path, value: dict) -> None:
    supervisor.v13_watch.atomic_json(path, value)


def _signed(value: dict) -> dict:
    return supervisor.signed(value)


def _supervisor_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        binding=tmp_path / "binding.json",
        supplement_binding=tmp_path / "supplement.json",
        supplement_binding_sha256=None,
        v13_state=tmp_path / "v13.state.json",
        v13_run_exit=tmp_path / "v13.exit",
        rac_final_root=tmp_path / "rac_final",
        rac_state=tmp_path / "rac_final" / "state.json",
        rac_run_exit=tmp_path / "rac_final" / "exit",
        output_root=tmp_path / "wcm",
        state=tmp_path / "wcm.state.json",
        run_exit=tmp_path / "wcm.exit",
        lock=tmp_path / "wcm.lock",
        trainer=SCRIPTS
        / "train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py",
        training_python=Path(sys.executable),
        poll_seconds=0.01,
        expected_gpu_uuid="test",
        max_recoverable_attempts=3,
    )


def _complete_upstreams(tmp_path: Path) -> argparse.Namespace:
    args = _supervisor_args(tmp_path)
    args.output_root.mkdir()
    args.rac_final_root.mkdir()
    args.binding.write_text("primary\n", encoding="utf-8")
    protocol = supervisor.v13_watch.actor_execution.execution_protocol(5)
    protocol_binding = {
        "format": "test_binding",
        "path_root": str(tmp_path),
        "path": "protocol.json",
        "file_sha256": "a" * 64,
        "protocol_logical_sha256": protocol["logical_sha256"],
        "protocol": protocol,
    }
    training_binding = {
        "path": str(args.binding),
        "file_sha256": supervisor.v13_watch.sha256_file(args.binding),
    }
    v13_summary = _signed(
        {
            "format": supervisor.v13_watch.FINAL_FORMAT,
            "status": "five_outer_lobo_source_only_training_complete",
            "fold_count": 5,
            "members_per_fold": 5,
            "steps_per_member": 3000,
            "training_binding": training_binding,
            "actor_execution_protocol": protocol,
            "actor_execution_protocol_binding": protocol_binding,
        }
    )
    v13_summary_path = tmp_path / "v13.summary.json"
    _json(v13_summary_path, v13_summary)
    _json(
        args.v13_state,
        {
            "format": supervisor.v13_watch.FORMAT,
            "status": "complete",
            "final_summary": str(v13_summary_path),
            "final_summary_file_sha256": supervisor.v13_watch.sha256_file(
                v13_summary_path
            ),
        },
    )
    args.v13_run_exit.write_text("0\n", encoding="utf-8")

    actor_authority = {
        "path": str(tmp_path / "actor_authority.json"),
        "file_sha256": "b" * 64,
        "logical_sha256": "c" * 64,
    }
    aggregate = _signed(
        {
            "format": supervisor.rac_watch.FINAL_FORMAT,
            "status": "five_outer_lobo_rac_source_only_training_complete",
            "training_binding": training_binding,
        }
    )
    aggregate_path = tmp_path / "rac.aggregate.json"
    _json(aggregate_path, aggregate)
    completion = _signed(
        {
            "format": "etsf_robotwin2_rac_supervisor_completion_audit_v1",
            "rac_aggregate": str(aggregate_path),
            "rac_aggregate_file_sha256": supervisor.v13_watch.sha256_file(
                aggregate_path
            ),
            "rac_aggregate_logical_sha256": aggregate["logical_sha256"],
            "required_supplement_binding_sha256": "d" * 64,
            "expected_actor_execution_protocol": protocol,
            "expected_actor_execution_protocol_binding": protocol_binding,
            "expected_actor_authority": actor_authority,
            "checkpoint_payloads_validated": True,
        }
    )
    authority = _signed(
        {
            "format": supervisor.rac_final_watch.EXECUTION_AUTHORITY_FORMAT,
            "critic_kind": "rac",
            "rac_completion_audit": completion,
        }
    )
    authority_path = args.rac_final_root / "execution_authority.json"
    _json(authority_path, authority)
    report_base = {
        "format": supervisor.rac_final_watch.materializer.evaluator.REPORT_FORMAT,
        "status": supervisor.rac_final_watch.materializer.evaluator.POLICY_ONLY_STATUS,
        "oracle_branch_diagnostic": {"evidence_sufficient": False},
    }
    report = {
        **report_base,
        "report_sha256": supervisor.v13_watch.canonical_sha256(report_base),
    }
    report_path = args.rac_final_root / "rac_crossbody_final_report.json"
    _json(report_path, report)
    final_report_binding = {
        "report": str(report_path),
        "report_file_sha256": supervisor.v13_watch.sha256_file(report_path),
        "report_sha256": report["report_sha256"],
        "status": supervisor.rac_final_watch.materializer.evaluator.POLICY_ONLY_STATUS,
    }
    receipt = _signed(
        {
            "format": supervisor.rac_final_watch.FINAL_RECEIPT_FORMAT,
            "status": "complete_rac_nested_policy_transfer_report",
            "execution_authority_logical_sha256": authority["logical_sha256"],
            "rac_completion_audit_logical_sha256": completion["logical_sha256"],
            "nested_completion_audit": {
                "critic_kind": "rac",
                "validated_rac_rank_receipts": 123,
                "all_rac_rank_receipts_replayed": True,
            },
            "final_report": final_report_binding,
            "critic_kind": "rac",
            "heldout_labels_used_for_training_or_checkpoint_selection": False,
            "cross_embodiment_success_measured": True,
        }
    )
    receipt_path = args.rac_final_root / "rac_nested_success_receipt.json"
    _json(receipt_path, receipt)
    _json(
        args.rac_state,
        {
            "format": supervisor.rac_final_watch.FORMAT,
            "status": "complete",
            "final_receipt": str(receipt_path),
            "final_receipt_file_sha256": supervisor.v13_watch.sha256_file(
                receipt_path
            ),
            "final_receipt_logical_sha256": receipt["logical_sha256"],
        },
    )
    args.rac_run_exit.write_text("0\n", encoding="utf-8")
    return args


def test_wcm_training_unlocks_only_after_complete_rac_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _complete_upstreams(tmp_path)
    monkeypatch.setattr(
        supervisor.v13_watch,
        "gpu_identity",
        lambda: (_ for _ in ()).throw(AssertionError("GPU queried during authority check")),
    )
    authority = supervisor.freeze_upstream_authority(args)
    assert authority["rac_cross_embodiment_success_measured_before_wcm_training"] is True
    assert authority["rac"]["validated_rac_rank_receipts"] == 123

    # A RAC five-fold state is not a substitute for the full nested final receipt.
    args.output_root.joinpath("upstream_v13_rac_authority.json").unlink()
    _json(
        args.rac_state,
        {"format": supervisor.rac_watch.FORMAT, "status": "complete"},
    )
    with pytest.raises(supervisor.WcmLoboSupervisorError, match="format changed"):
        supervisor.freeze_upstream_authority(args)


def test_resigned_rac_final_report_tamper_cannot_unlock_wcm(tmp_path: Path) -> None:
    args = _complete_upstreams(tmp_path)
    report_path = args.rac_final_root / "rac_crossbody_final_report.json"
    changed = copy.deepcopy(supervisor.read_json(report_path, "report"))
    changed["status"] = "tampered"
    changed.pop("report_sha256")
    changed["report_sha256"] = supervisor.v13_watch.canonical_sha256(changed)
    _json(report_path, changed)
    with pytest.raises(supervisor.WcmLoboSupervisorError):
        supervisor.freeze_upstream_authority(args)


def test_wcm_downstream_command_is_explicit_and_five_fold(tmp_path: Path) -> None:
    args = argparse.Namespace(
        robotwin_python=Path(sys.executable),
        nested_runner=SCRIPTS
        / "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py",
        actor_checkpoint=tmp_path / "actor",
        vlm_metadata=tmp_path / "vlm",
        robotwin_root=tmp_path / "robotwin",
        event_spec=tmp_path / "event.json",
        reference_preregistration=tmp_path / "prereg.json",
        actor_execution_protocol=tmp_path / "protocol.json",
        actor_execution_protocol_sha256="a" * 64,
        path_root=tmp_path,
        output_root=tmp_path / "wcm_nested",
    )
    completion = {
        "required_supplement_binding_sha256": "d" * 64,
        "folds": {
            body: {"root": str(tmp_path / "wcm" / f"outer_lobo_{body}")}
            for body in downstream.BODIES
        },
    }
    command = downstream.nested_command(args, completion)
    assert command[command.index("--critic-kind") + 1] == "wcm"
    assert command[command.index("--required-supplement-binding-sha256") + 1] == "d" * 64
    assert command.count("--lobo-fold") == 5
    assert command[command.index("--output") + 1] == str(
        args.output_root / "nested_wcm"
    )


def test_wcm_execution_authority_is_create_once_and_tamper_closed(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(output_root=tmp_path / "out")
    args.output_root.mkdir()
    authority = downstream.signed(
        {"format": downstream.EXECUTION_AUTHORITY_FORMAT, "critic_kind": "wcm"}
    )
    assert downstream.create_or_validate_execution_authority(args, authority) == authority
    changed = copy.deepcopy(authority)
    changed["critic_kind"] = "rac"
    changed.pop("logical_sha256")
    changed = downstream.signed(changed)
    with pytest.raises(downstream.WcmNestedWatcherError, match="authority changed"):
        downstream.create_or_validate_execution_authority(args, changed)
