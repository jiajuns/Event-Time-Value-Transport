#!/usr/bin/env python3
"""Run the authenticated RAC five-fold result through nested N1/N4/N8.

The watcher is intentionally downstream-only.  It waits for a complete RAC
LOBO supervisor, verifies the signed five-fold/checkpoint chain and the frozen
main-v13 completion authority, then invokes the existing nested runner with
``--critic-kind rac``.  Existing create-once nested artifacts are resumed by
the runner.  A final policy-transfer report is materialized only after every
RAC rank receipt has been replayed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import materialize_robotwin2_nested_n1_n4_n8_final_report_v1 as materializer
import robotwin2_relative_action_critic_adapter_v1 as rac_adapter
import run_robotwin2_five_body_nested_n4_n8_paired_success_v1 as nested
import watch_robotwin2_five_body_branches_to_rac_lobo_training_v1 as rac_watch
import watch_robotwin2_postformal_shared_head_upgrade_v1 as postformal


FORMAT = "etsf_robotwin2_rac_lobo_to_nested_success_watcher_v1"
EXECUTION_AUTHORITY_FORMAT = "etsf_robotwin2_rac_nested_execution_authority_v1"
LAUNCH_RECEIPT_FORMAT = "etsf_robotwin2_rac_nested_launch_receipt_v1"
FAILURE_RECEIPT_FORMAT = "etsf_robotwin2_rac_nested_failure_receipt_v1"
FINAL_RECEIPT_FORMAT = "etsf_robotwin2_rac_nested_success_receipt_v1"
DEFERRED_AUTHORITY_FORMAT = (
    "etsf_robotwin2_rac_nested_deferred_runtime_authority_v1"
)
BODIES = rac_watch.BODIES


class RacNestedWatcherError(RuntimeError):
    pass


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    return rac_watch.base.signed(value)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return rac_watch.base.read_json(path, label)
    except (OSError, rac_watch.base.LoboWatcherError) as error:
        raise RacNestedWatcherError(str(error)) from error


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    try:
        rac_watch.base.verify_logical_sha(value, label)
    except rac_watch.base.LoboWatcherError as error:
        raise RacNestedWatcherError(str(error)) from error


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stable_read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = rac_watch.base.sha256_file(path)
    value = read_json(path, label)
    after = rac_watch.base.sha256_file(path)
    if before != after:
        raise RacNestedWatcherError(f"{label} changed during authentication")
    return value, after


def _verify_named_sha(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != rac_watch.base.canonical_sha256(unsigned):
        raise RacNestedWatcherError(f"{label} {field} changed")
    return str(declared)


def _validate_frozen_v13_authority(
    authority: Mapping[str, Any],
) -> None:
    verify_signed(authority, "RAC frozen upstream authority")
    if (
        authority.get("format") != rac_watch.UPSTREAM_AUTHORITY_FORMAT
        or authority.get("enabled") is not True
        or authority.get("upstream_completed_before_rac_training") is not True
        or authority.get("heldout_payloads_opened_while_waiting") != 0
    ):
        raise RacNestedWatcherError("RAC supervisor lacks frozen main-v13 authority")
    run_exit = Path(str(authority.get("upstream_run_exit", ""))).expanduser().resolve()
    state = Path(str(authority.get("upstream_state", ""))).expanduser().resolve()
    report = Path(str(authority.get("final_report", ""))).expanduser().resolve()
    if (
        run_exit.read_text(encoding="utf-8").strip() != "0"
        or rac_watch.base.sha256_file(run_exit)
        != authority.get("upstream_run_exit_file_sha256")
        or rac_watch.base.sha256_file(state)
        != authority.get("upstream_state_file_sha256")
        or rac_watch.base.sha256_file(report)
        != authority.get("final_report_file_sha256")
    ):
        raise RacNestedWatcherError("frozen main-v13 files changed")
    upstream_state = read_json(state, "frozen main-v13 state")
    final_report = read_json(report, "frozen main-v13 report")
    named_field = authority.get("final_report_logical_sha_field")
    if (
        upstream_state.get("status") != "complete"
        or named_field not in {"report_sha256", "logical_sha256"}
        or _verify_named_sha(final_report, str(named_field), "main-v13 report")
        != authority.get("final_report_logical_sha256")
    ):
        raise RacNestedWatcherError("frozen main-v13 completion changed")


def validate_rac_supervisor_completion(
    args: argparse.Namespace,
    *,
    validate_checkpoints: bool = True,
) -> dict[str, Any] | None:
    """Return a signed RAC completion audit, or None while still running."""

    if not args.rac_run_exit.is_file() or not args.rac_state.is_file():
        return None
    exit_before = rac_watch.base.sha256_file(args.rac_run_exit)
    exit_value = args.rac_run_exit.read_text(encoding="utf-8").strip()
    exit_after = rac_watch.base.sha256_file(args.rac_run_exit)
    if exit_before != exit_after:
        raise RacNestedWatcherError("RAC run-exit changed during authentication")
    if exit_value == "1":
        raise RacNestedWatcherError("RAC supervisor failed")
    if exit_value != "0":
        return None
    state, state_sha = _stable_read(args.rac_state, "RAC supervisor state")
    if state.get("status") == "failed":
        raise RacNestedWatcherError("RAC supervisor state reports failure")
    if state.get("status") != "complete":
        return None
    aggregate_path = args.rac_root / "five_fold_rac_training_summary.json"
    aggregate, aggregate_file_sha = _stable_read(
        aggregate_path, "RAC five-fold aggregate"
    )
    verify_signed(aggregate, "RAC five-fold aggregate")
    authority_path = args.rac_root / "upstream_completion_authority.json"
    authority, authority_file_sha = _stable_read(
        authority_path, "RAC frozen upstream authority"
    )
    _validate_frozen_v13_authority(authority)
    fold_rows = aggregate.get("outer_folds")
    supplement_sha = aggregate.get("supplement_binding_file_sha256")
    protocol = aggregate.get("actor_execution_protocol")
    protocol_binding = aggregate.get("actor_execution_protocol_binding")
    actor_authority = aggregate.get("actor_authority")
    if (
        state.get("format") != rac_watch.FORMAT
        or state.get("final_summary") != str(aggregate_path)
        or state.get("final_summary_file_sha256") != aggregate_file_sha
        or state.get("upstream_completion_authority") != authority
        or aggregate.get("format") != rac_watch.FINAL_FORMAT
        or aggregate.get("status")
        != "five_outer_lobo_rac_source_only_training_complete"
        or aggregate.get("model_family") != rac_watch.rac.MODEL_FAMILY
        or aggregate.get("fold_count") != len(BODIES)
        or aggregate.get("members_per_fold") != rac_watch.rac.ENSEMBLE_SIZE
        or aggregate.get("steps_per_member") != rac_watch.STEPS_PER_MEMBER
        or aggregate.get("upstream_completion_authority") != authority
        or not isinstance(fold_rows, list)
        or len(fold_rows) != len(BODIES)
        or not isinstance(supplement_sha, str)
        or len(supplement_sha) != 64
        or not isinstance(protocol, Mapping)
        or not isinstance(protocol_binding, Mapping)
        or aggregate.get("actor_execution_protocol_file_sha256")
        != protocol_binding.get("file_sha256")
        or protocol_binding.get("protocol") != protocol
        or not isinstance(actor_authority, Mapping)
        or not isinstance(actor_authority.get("path"), str)
        or not isinstance(actor_authority.get("file_sha256"), str)
        or not isinstance(actor_authority.get("logical_sha256"), str)
    ):
        raise RacNestedWatcherError("RAC supervisor completion binding changed")
    rows_by_body = {
        row.get("held_out_body"): row
        for row in fold_rows
        if isinstance(row, Mapping)
    }
    if set(rows_by_body) != set(BODIES):
        raise RacNestedWatcherError("RAC aggregate lacks five distinct folds")
    folds: dict[str, dict[str, Any]] = {}
    for body in BODIES:
        fold_root = args.rac_root / f"outer_lobo_{body}"
        summary_path = fold_root / "training_summary.json"
        summary, summary_file_sha = _stable_read(
            summary_path, f"{body} RAC summary"
        )
        verify_signed(summary, f"{body} RAC summary")
        row = rows_by_body[body]
        if (
            row.get("training_summary") != str(summary_path)
            or row.get("training_summary_file_sha256") != summary_file_sha
            or row.get("training_summary_logical_sha256")
            != summary.get("logical_sha256")
            or summary.get("format") != rac_watch.rac.SUMMARY_FORMAT
            or summary.get("held_out_body") != body
            or summary.get("heldout_rows_used_for_training_normalization_or_selection")
            != 0
            or summary.get("all_checkpoints_selected_before_any_heldout_payload_open")
            is not True
            or summary.get("actor_execution_protocol") != protocol
            or summary.get("actor_execution_protocol_binding") != protocol_binding
            or summary.get("actor_execution_protocol_file_sha256")
            != protocol_binding.get("file_sha256")
        ):
            raise RacNestedWatcherError(f"{body} RAC fold differs from aggregate")
        load_receipt = row.get("ensemble_load_receipt")
        if not isinstance(load_receipt, Mapping):
            raise RacNestedWatcherError(f"{body} RAC load receipt is missing")
        verify_signed(load_receipt, f"{body} RAC load receipt")
        if validate_checkpoints:
            try:
                models, replayed_load_receipt = rac_adapter.load_fold_ensemble(
                    summary_path,
                    device=torch.device("cpu"),
                    expected_held_out_body=body,
                )
            except rac_adapter.RelativeActionCriticAdapterError as error:
                raise RacNestedWatcherError(str(error)) from error
            del models
            if load_receipt != replayed_load_receipt:
                raise RacNestedWatcherError(f"{body} RAC load receipt changed")
        folds[body] = {
            "root": str(fold_root),
            "summary": str(summary_path),
            "summary_file_sha256": summary_file_sha,
            "summary_logical_sha256": summary["logical_sha256"],
            "ensemble_load_receipt": load_receipt,
        }
    base = {
        "format": "etsf_robotwin2_rac_supervisor_completion_audit_v1",
        "rac_run_exit": str(args.rac_run_exit),
        "rac_run_exit_file_sha256": exit_after,
        "rac_state": str(args.rac_state),
        "rac_state_file_sha256": state_sha,
        "rac_aggregate": str(aggregate_path),
        "rac_aggregate_file_sha256": aggregate_file_sha,
        "rac_aggregate_logical_sha256": aggregate["logical_sha256"],
        "frozen_v13_authority": authority,
        "frozen_v13_authority_file_sha256": authority_file_sha,
        "required_supplement_binding_sha256": supplement_sha,
        "expected_actor_execution_protocol": dict(protocol),
        "expected_actor_execution_protocol_binding": dict(protocol_binding),
        "expected_actor_authority": dict(actor_authority),
        "folds": folds,
        "heldout_payloads_or_labels_opened_by_watcher": 0,
        "checkpoint_payloads_validated": bool(validate_checkpoints),
    }
    return {**base, "logical_sha256": rac_watch.base.canonical_sha256(base)}


def freeze_or_validate_deferred_runtime_authority(
    args: argparse.Namespace,
    completion: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Freeze protocol/actor files only after RAC completion authenticates them."""

    expected_protocol = completion.get("expected_actor_execution_protocol")
    expected_binding = completion.get("expected_actor_execution_protocol_binding")
    expected_actor = completion.get("expected_actor_authority")
    if (
        not isinstance(expected_protocol, Mapping)
        or not isinstance(expected_binding, Mapping)
        or not isinstance(expected_actor, Mapping)
    ):
        raise RacNestedWatcherError("RAC completion lacks deferred authorities")
    expected_protocol_sha = expected_binding.get("file_sha256")
    if (
        not isinstance(expected_protocol_sha, str)
        or len(expected_protocol_sha) != 64
        or expected_binding.get("protocol") != expected_protocol
        or expected_binding.get("protocol_logical_sha256")
        != expected_protocol.get("logical_sha256")
    ):
        raise RacNestedWatcherError("RAC protocol authority is invalid")
    expected_protocol_path = (
        Path(str(expected_binding.get("path_root", ""))).expanduser().resolve()
        / str(expected_binding.get("path", ""))
    ).resolve()
    expected_actor_path = Path(str(expected_actor.get("path", ""))).expanduser().resolve()
    if args.actor_execution_protocol != expected_protocol_path:
        raise RacNestedWatcherError("fixed actor protocol path differs from RAC authority")
    if args.actor_authority != expected_actor_path:
        raise RacNestedWatcherError("fixed actor authority path differs from RAC authority")
    if (
        args.actor_execution_protocol_sha256 is not None
        and args.actor_execution_protocol_sha256 != expected_protocol_sha
    ):
        raise RacNestedWatcherError("explicit actor protocol SHA differs from RAC authority")

    receipt_path = args.output_root / "deferred_runtime_authority.json"
    frozen = None
    if receipt_path.exists():
        frozen = read_json(receipt_path, "deferred RAC runtime authority")
        verify_signed(frozen, "deferred RAC runtime authority")
        if frozen.get("format") != DEFERRED_AUTHORITY_FORMAT:
            raise RacNestedWatcherError("deferred runtime authority format changed")
    if not args.actor_execution_protocol.is_file() or not args.actor_authority.is_file():
        if frozen is not None:
            raise RacNestedWatcherError("frozen deferred runtime file disappeared")
        return None

    protocol_before = rac_watch.base.sha256_file(args.actor_execution_protocol)
    if protocol_before != expected_protocol_sha:
        raise RacNestedWatcherError("actor protocol file SHA differs from RAC authority")
    try:
        observed_binding = (
            nested.formal.actor_execution.execution_protocol_file_binding(
                args.actor_execution_protocol,
                expected_protocol_sha,
                path_root=Path(str(expected_binding["path_root"])),
            )
        )
    except nested.formal.actor_execution.ActorExecutionProtocolError as error:
        raise RacNestedWatcherError(str(error)) from error
    protocol_after = rac_watch.base.sha256_file(args.actor_execution_protocol)
    if protocol_before != protocol_after or observed_binding != expected_binding:
        raise RacNestedWatcherError("actor protocol changed during deferred authentication")

    actor_before = rac_watch.base.sha256_file(args.actor_authority)
    actor_value = read_json(args.actor_authority, "deferred actor authority")
    actor_after = rac_watch.base.sha256_file(args.actor_authority)
    verify_signed(actor_value, "deferred actor authority")
    sampling_contract = actor_value.get("sampling_contract")
    if (
        actor_before != actor_after
        or actor_after != expected_actor.get("file_sha256")
        or actor_value.get("logical_sha256") != expected_actor.get("logical_sha256")
        or actor_value.get("format") != rac_watch.base.ACTOR_FORMAT
        or actor_value.get("one_universal_actor_for_all_five_bodies") is not True
        or not isinstance(sampling_contract, Mapping)
        or sampling_contract.get("actor_execution_protocol_binding")
        != expected_binding
    ):
        raise RacNestedWatcherError("actor authority differs from RAC aggregate")
    base = {
        "format": DEFERRED_AUTHORITY_FORMAT,
        "rac_completion_audit_logical_sha256": completion["logical_sha256"],
        "actor_execution_protocol": str(args.actor_execution_protocol),
        "actor_execution_protocol_file_sha256": protocol_after,
        "actor_execution_protocol_logical_sha256": expected_protocol[
            "logical_sha256"
        ],
        "actor_execution_protocol_binding": dict(expected_binding),
        "actor_authority": str(args.actor_authority),
        "actor_authority_file_sha256": actor_after,
        "actor_authority_logical_sha256": actor_value["logical_sha256"],
        "files_authenticated_after_rac_completion": True,
        "heldout_payloads_opened_while_waiting": 0,
    }
    value = {**base, "logical_sha256": rac_watch.base.canonical_sha256(base)}
    if frozen is not None:
        if frozen != value:
            raise RacNestedWatcherError("deferred runtime authority changed")
    else:
        rac_watch.base.create_once_or_verify(
            receipt_path, value, "deferred RAC runtime authority"
        )
    args.actor_execution_protocol_sha256 = expected_protocol_sha
    return value


def nested_command(
    args: argparse.Namespace, completion: Mapping[str, Any]
) -> list[str]:
    command = [
        str(args.robotwin_python),
        str(args.nested_runner),
        "--actor-checkpoint", str(args.actor_checkpoint),
        "--vlm-metadata-path", str(args.vlm_metadata),
        "--robotwin-root", str(args.robotwin_root),
        "--event-spec", str(args.event_spec),
        "--reference-preregistration", str(args.reference_preregistration),
        "--actor-execution-protocol", str(args.actor_execution_protocol),
        "--actor-execution-protocol-sha256",
        str(args.actor_execution_protocol_sha256),
        "--path-root", str(args.path_root),
        "--critic-kind", "rac",
        "--required-supplement-binding-sha256",
        str(completion["required_supplement_binding_sha256"]),
        "--output", str(args.output_root / "nested_rac"),
    ]
    for body in BODIES:
        command.extend(
            ["--lobo-fold", f"{body}={completion['folds'][body]['root']}"]
        )
    return command


def report_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.system_python),
        str(args.final_materializer),
        "--nested-root", str(args.output_root / "nested_rac"),
        "--actor-authority", str(args.actor_authority),
        "--output-input", str(args.output_root / "rac_crossbody_final_report_input.json"),
        "--output-report", str(args.output_root / "rac_crossbody_final_report.json"),
    ]


def execution_authority(
    args: argparse.Namespace,
    completion: Mapping[str, Any],
    nested_cmd: Sequence[str],
    report_cmd: Sequence[str],
) -> dict[str, Any]:
    deferred_path = args.output_root / "deferred_runtime_authority.json"
    deferred = read_json(deferred_path, "deferred RAC runtime authority")
    verify_signed(deferred, "deferred RAC runtime authority")
    code_paths = (
        Path(__file__).resolve(),
        args.nested_runner,
        Path(rac_adapter.__file__).resolve(),
        Path(rac_watch.rac.__file__).resolve(),
        args.final_materializer,
        args.robotwin_python,
        args.system_python,
    )
    base = {
        "format": EXECUTION_AUTHORITY_FORMAT,
        "rac_completion_audit": dict(completion),
        "deferred_runtime_authority": deferred,
        "deferred_runtime_authority_file_sha256": rac_watch.base.sha256_file(
            deferred_path
        ),
        "actor_checkpoint": str(args.actor_checkpoint),
        "actor_checkpoint_tree_sha256": rac_watch.base.sha256_tree(
            args.actor_checkpoint
        )[0],
        "vlm_metadata": str(args.vlm_metadata),
        "vlm_metadata_tree_sha256": rac_watch.base.sha256_tree(args.vlm_metadata)[0],
        "actor_authority": str(args.actor_authority),
        "actor_authority_file_sha256": rac_watch.base.sha256_file(
            args.actor_authority
        ),
        "event_spec": str(args.event_spec),
        "event_spec_file_sha256": rac_watch.base.sha256_file(args.event_spec),
        "reference_preregistration": str(args.reference_preregistration),
        "reference_preregistration_file_sha256": rac_watch.base.sha256_file(
            args.reference_preregistration
        ),
        "actor_execution_protocol": str(args.actor_execution_protocol),
        "actor_execution_protocol_file_sha256": rac_watch.base.sha256_file(
            args.actor_execution_protocol
        ),
        "actor_execution_protocol_expected_sha256": (
            args.actor_execution_protocol_sha256
        ),
        "code_files": [
            {"path": str(path), "sha256": rac_watch.base.sha256_file(path)}
            for path in code_paths
        ],
        "nested_command": list(nested_cmd),
        "nested_command_sha256": rac_watch.base.canonical_sha256(list(nested_cmd)),
        "report_command": list(report_cmd),
        "report_command_sha256": rac_watch.base.canonical_sha256(list(report_cmd)),
        "nested_output": str(args.output_root / "nested_rac"),
        "critic_kind": "rac",
        "resume_requires_exact_same_authority": True,
        "heldout_labels_or_outcomes_used_to_select_retry": False,
    }
    return {**base, "logical_sha256": rac_watch.base.canonical_sha256(base)}


def create_or_validate_execution_authority(
    args: argparse.Namespace, expected: Mapping[str, Any]
) -> dict[str, Any]:
    verify_signed(expected, "expected RAC nested execution authority")
    path = args.output_root / "execution_authority.json"
    if path.exists():
        observed = read_json(path, "RAC nested execution authority")
        verify_signed(observed, "RAC nested execution authority")
        if observed != dict(expected):
            raise RacNestedWatcherError("RAC nested execution authority changed")
        return observed
    rac_watch.base.create_once_or_verify(
        path, expected, "RAC nested execution authority"
    )
    return dict(expected)


def create_or_validate_launch_receipt(
    args: argparse.Namespace,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "format": LAUNCH_RECEIPT_FORMAT,
        "execution_authority_logical_sha256": authority["logical_sha256"],
        "nested_command_sha256": authority["nested_command_sha256"],
        "nested_output": authority["nested_output"],
        "same_command_resume_allowed_after_unrecorded_process_interruption": True,
        "retry_selection_reads_nested_outcomes": False,
    }
    value = {**base, "logical_sha256": rac_watch.base.canonical_sha256(base)}
    rac_watch.base.create_once_or_verify(
        args.output_root / "nested_launch_receipt.json",
        value,
        "RAC nested launch receipt",
    )
    return value


def record_failure(
    args: argparse.Namespace,
    *,
    stage: str,
    authority: Mapping[str, Any] | None,
    returncode: int | None,
    error: BaseException | None,
) -> dict[str, Any]:
    base = {
        "format": FAILURE_RECEIPT_FORMAT,
        "stage": stage,
        "execution_authority_logical_sha256": (
            authority.get("logical_sha256") if authority is not None else None
        ),
        "returncode": returncode,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
        "recorded_at_utc": rac_watch.base.utc_now(),
        "retry_selected_using_nested_outcomes": False,
    }
    value = {**base, "logical_sha256": rac_watch.base.canonical_sha256(base)}
    rac_watch.base.create_once_or_verify(
        args.output_root / "failure_receipt.json", value, "RAC nested failure"
    )
    return value


def validate_complete_nested_rac(
    args: argparse.Namespace,
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    root = args.output_root / "nested_rac"
    protocol = nested.formal.actor_execution.load_execution_protocol_file(
        args.actor_execution_protocol,
        args.actor_execution_protocol_sha256,
    )
    protocol_binding = nested.formal.actor_execution.execution_protocol_file_binding(
        args.actor_execution_protocol,
        args.actor_execution_protocol_sha256,
        path_root=args.path_root,
    )
    nested.configure_actor_execution_protocol(
        protocol,
        path=args.actor_execution_protocol,
        file_sha256=args.actor_execution_protocol_sha256,
        path_root=args.path_root,
    )
    try:
        audit = postformal.validate_nested_completion(
            root,
            expected_actor_execution_protocol_binding=protocol_binding,
        )
    except postformal.SharedHeadUpgradeError as error:
        raise RacNestedWatcherError(str(error)) from error
    contract = read_json(root / "execution_contract.json", "RAC nested contract")
    verify_signed(contract, "RAC nested contract")
    expected_folds = {
        body: nested.inspect_rac_fold(
            body, Path(str(completion["folds"][body]["root"]))
        )
        for body in BODIES
    }
    if (
        contract.get("critic_kind") != "rac"
        or contract.get("method_critic_assignment")
        != {
            nested.METHOD_ACTOR: None,
            nested.METHOD_N4: "rac",
            nested.METHOD_N8: "rac",
        }
        or contract.get("rac_rank_receipt_format")
        != nested.RAC_RANK_RECEIPT_FORMAT
        or contract.get("folds") != expected_folds
        or contract.get("fold_training_regime", {}).get(
            "supplement_binding_file_sha256"
        )
        != completion["required_supplement_binding_sha256"]
        or contract.get("actor_execution_protocol") != protocol
    ):
        raise RacNestedWatcherError("nested RAC contract changed")
    pair_dir = root / "pairs"
    expected_paths = {
        pair_dir
        / f"{nested.pair_id(row['heldout_body'], row['condition'], row['requested_seed'])}.json"
        for row in nested.evaluation_schedule()
    }
    observed_paths = {path for path in pair_dir.glob("*.json") if path.is_file()}
    if observed_paths != expected_paths:
        raise RacNestedWatcherError("nested RAC pair roster changed")
    validated_rank_receipts = 0
    for expected in nested.evaluation_schedule():
        path = pair_dir / (
            nested.pair_id(
                expected["heldout_body"],
                expected["condition"],
                expected["requested_seed"],
            )
            + ".json"
        )
        pair = read_json(path, "nested RAC pair")
        _verify_named_sha(pair, "pair_sha256", "nested RAC pair")
        rollouts = pair.get("rollouts")
        if not isinstance(rollouts, Mapping) or set(rollouts) != set(nested.METHODS):
            raise RacNestedWatcherError("nested RAC pair rollout roster changed")
        normalizer_sha = contract["source_train_action_normalizer_by_heldout_body"][
            expected["heldout_body"]
        ]["logical_sha256"]
        for method in nested.METHODS:
            try:
                nested.validate_rollout(
                    rollouts[method],
                    method=method,
                    expected=expected,
                    expected_source_action_normalizer_logical_sha256=normalizer_sha,
                )
            except nested.NestedCandidatePoolError as error:
                raise RacNestedWatcherError(str(error)) from error
            if method != nested.METHOD_ACTOR:
                for decision in rollouts[method]["decisions"]:
                    if decision.get("critic_scores", {}).get("critic_kind") != "rac":
                        raise RacNestedWatcherError("nested rollout used non-RAC critic")
                    validated_rank_receipts += 1
    return {
        **audit,
        "contract_logical_sha256": contract["logical_sha256"],
        "critic_kind": "rac",
        "validated_rac_rank_receipts": validated_rank_receipts,
        "all_rac_rank_receipts_replayed": True,
    }


def validate_final_report(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.output_root / "rac_crossbody_final_report_input.json"
    report_path = args.output_root / "rac_crossbody_final_report.json"
    value, file_sha = _stable_read(report_path, "RAC final cross-body report")
    report_sha = _verify_named_sha(value, "report_sha256", "RAC final report")
    materialized_input, input_file_sha = _stable_read(
        input_path, "RAC final report input"
    )
    verify_signed(materialized_input, "RAC final report input")
    try:
        replayed_input, replayed_report = materializer.build_materialization(
            nested_root=args.output_root / "nested_rac",
            actor_authority_path=args.actor_authority,
            oracle_truth_path=None,
        )
    except materializer.FinalReportMaterializationError as error:
        raise RacNestedWatcherError(str(error)) from error
    oracle = value.get("oracle_branch_diagnostic")
    if (
        materialized_input != replayed_input
        or value != replayed_report
        or value.get("format") != materializer.evaluator.REPORT_FORMAT
        or value.get("status") != materializer.evaluator.POLICY_ONLY_STATUS
        or value.get("input_document_sha256")
        != materialized_input["logical_sha256"]
        or not isinstance(oracle, Mapping)
        or oracle.get("evidence_sufficient") is not False
        or oracle.get("oracle_regret_reported") is not False
    ):
        raise RacNestedWatcherError("RAC final report did not fail closed")
    return {
        "input": str(input_path),
        "input_file_sha256": input_file_sha,
        "input_logical_sha256": materialized_input["logical_sha256"],
        "report": str(report_path),
        "report_file_sha256": file_sha,
        "report_sha256": report_sha,
        "status": value["status"],
        "oracle_evidence_sufficient": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rac-root", type=Path, required=True)
    parser.add_argument("--rac-state", type=Path, required=True)
    parser.add_argument("--rac-run-exit", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--reference-preregistration", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol-sha256")
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-exit", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "--nested-runner",
        type=Path,
        default=Path(__file__).with_name(
            "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py"
        ),
    )
    parser.add_argument(
        "--final-materializer",
        type=Path,
        default=Path(__file__).with_name(
            "materialize_robotwin2_nested_n1_n4_n8_final_report_v1.py"
        ),
    )
    parser.add_argument(
        "--robotwin-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/RoboTwin2/bin/python"),
    )
    parser.add_argument("--system-python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument("--expected-gpu-uuid", default=rac_watch.base.EXPECTED_GPU_UUID)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    names = (
        "rac_root", "rac_state", "rac_run_exit", "actor_checkpoint",
        "vlm_metadata", "actor_authority", "robotwin_root", "event_spec",
        "reference_preregistration", "actor_execution_protocol", "path_root",
        "output_root", "state", "run_exit", "nested_runner",
        "final_materializer", "robotwin_python", "system_python",
    )
    for name in names:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.lock = (
        args.lock.expanduser().resolve()
        if args.lock is not None
        else args.output_root / "rac_nested.lock"
    )
    if args.poll_seconds <= 0:
        raise RacNestedWatcherError("poll interval must be positive")
    if args.actor_execution_protocol_sha256 is not None and (
        len(args.actor_execution_protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.actor_execution_protocol_sha256
        )
    ):
        raise RacNestedWatcherError("explicit actor protocol SHA is invalid")
    for path in (
        args.actor_checkpoint,
        args.vlm_metadata,
        args.robotwin_root,
        args.event_spec,
        args.reference_preregistration,
        args.path_root,
        args.nested_runner,
        args.final_materializer,
        args.robotwin_python,
        args.system_python,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    try:
        args.output_root.relative_to(args.path_root)
    except ValueError as error:
        raise RacNestedWatcherError("RAC nested output must be inside path_root") from error
    return args


_FAILURE_ARGS: argparse.Namespace | None = None
_FAILURE_AUTHORITY: Mapping[str, Any] | None = None


def main(argv: Sequence[str] | None = None) -> int:
    global _FAILURE_ARGS, _FAILURE_AUTHORITY
    args = normalized_args(parse_args(argv))
    _FAILURE_ARGS = args
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock = args.lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RacNestedWatcherError("another RAC nested watcher is active") from error

    def write_state(status: str, **extra: Any) -> None:
        rac_watch.base.atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": rac_watch.base.utc_now(),
                "pid": os.getpid(),
                "rac_root": str(args.rac_root),
                "output_root": str(args.output_root),
                "critic_kind": "rac",
                **extra,
            },
        )

    prior_failure = args.output_root / "failure_receipt.json"
    if prior_failure.exists():
        value = read_json(prior_failure, "prior RAC nested failure")
        verify_signed(value, "prior RAC nested failure")
        raise RacNestedWatcherError("prior immutable RAC nested failure exists")
    completion = None
    while completion is None:
        completion = validate_rac_supervisor_completion(
            args, validate_checkpoints=False
        )
        if completion is None:
            write_state(
                "waiting_for_authenticated_rac_five_fold_completion",
                gpu_reserved_by_watcher=False,
                heldout_payloads_opened_while_waiting=0,
            )
            time.sleep(args.poll_seconds)
    deferred_authority = None
    while deferred_authority is None:
        deferred_authority = freeze_or_validate_deferred_runtime_authority(
            args, completion
        )
        if deferred_authority is None:
            write_state(
                "waiting_for_deferred_protocol_and_actor_authority",
                rac_completion_audit_logical_sha256=completion["logical_sha256"],
                actor_execution_protocol_present=(
                    args.actor_execution_protocol.is_file()
                ),
                actor_authority_present=args.actor_authority.is_file(),
                gpu_reserved_by_watcher=False,
                gpu_identity_queried=False,
                heldout_payloads_opened_while_waiting=0,
            )
            time.sleep(args.poll_seconds)
    checkpoint_completion = validate_rac_supervisor_completion(
        args, validate_checkpoints=True
    )
    if (
        checkpoint_completion is None
        or checkpoint_completion.get("rac_aggregate_logical_sha256")
        != completion.get("rac_aggregate_logical_sha256")
        or checkpoint_completion.get("frozen_v13_authority")
        != completion.get("frozen_v13_authority")
        or checkpoint_completion.get("checkpoint_payloads_validated") is not True
    ):
        raise RacNestedWatcherError(
            "RAC completion changed before checkpoint authentication"
        )
    completion = checkpoint_completion
    gpu = rac_watch.base.gpu_identity()
    if "4090" not in gpu["name"] or gpu["uuid"] != args.expected_gpu_uuid:
        raise RacNestedWatcherError(f"unexpected GPU authority: {gpu}")
    nested_cmd = nested_command(args, completion)
    report_cmd = report_command(args)
    authority = create_or_validate_execution_authority(
        args,
        execution_authority(args, completion, nested_cmd, report_cmd),
    )
    _FAILURE_AUTHORITY = authority
    create_or_validate_launch_receipt(args, authority)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(args.nested_runner.parent),
        }
    )
    nested_root = args.output_root / "nested_rac"
    if not (nested_root / "completion_receipt.json").is_file():
        while True:
            pids = rac_watch.base.gpu_compute_pids()
            if not pids:
                break
            write_state(
                "waiting_for_idle_gpu_before_rac_nested",
                external_gpu_compute_pids=pids,
                gpu=gpu,
                gpu_reserved_by_watcher=False,
                execution_authority_logical_sha256=authority["logical_sha256"],
            )
            time.sleep(args.poll_seconds)
        write_state(
            "running_rac_nested_n1_n4_n8",
            command=nested_cmd,
            gpu=gpu,
            execution_authority_logical_sha256=authority["logical_sha256"],
        )
        log = args.output_root / "rac_nested.log"
        with log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(
                nested_cmd,
                cwd=args.robotwin_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            record_failure(
                args,
                stage="nested_rac",
                authority=authority,
                returncode=result.returncode,
                error=None,
            )
            raise RacNestedWatcherError(
                f"RAC nested runner exited {result.returncode}"
            )
    nested_audit = validate_complete_nested_rac(args, completion)
    report_path = args.output_root / "rac_crossbody_final_report.json"
    if not report_path.is_file():
        write_state(
            "materializing_rac_crossbody_final_report",
            nested_completion_audit=nested_audit,
            command=report_cmd,
            execution_authority_logical_sha256=authority["logical_sha256"],
        )
        log = args.output_root / "rac_final_report.log"
        with log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(
                report_cmd,
                cwd=args.final_materializer.parent,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            record_failure(
                args,
                stage="final_report",
                authority=authority,
                returncode=result.returncode,
                error=None,
            )
            raise RacNestedWatcherError(
                f"RAC final report materializer exited {result.returncode}"
            )
    final_report = validate_final_report(args)
    final_base = {
        "format": FINAL_RECEIPT_FORMAT,
        "status": "complete_rac_nested_policy_transfer_report",
        "execution_authority_logical_sha256": authority["logical_sha256"],
        "rac_completion_audit_logical_sha256": completion["logical_sha256"],
        "nested_completion_audit": nested_audit,
        "final_report": final_report,
        "critic_kind": "rac",
        "heldout_labels_used_for_training_or_checkpoint_selection": False,
        "cross_embodiment_success_measured": True,
    }
    final = {
        **final_base,
        "logical_sha256": rac_watch.base.canonical_sha256(final_base),
    }
    final_path = args.output_root / "rac_nested_success_receipt.json"
    rac_watch.base.create_once_or_verify(
        final_path, final, "RAC nested success receipt"
    )
    atomic_text(args.run_exit, "0\n")
    write_state(
        "complete",
        gpu=gpu,
        deferred_runtime_authority=deferred_authority,
        execution_authority_logical_sha256=authority["logical_sha256"],
        nested_completion_audit=nested_audit,
        final_report=final_report,
        final_receipt=str(final_path),
        final_receipt_file_sha256=rac_watch.base.sha256_file(final_path),
        final_receipt_logical_sha256=final["logical_sha256"],
        gpu_reserved_by_watcher=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if _FAILURE_ARGS is not None:
            args = _FAILURE_ARGS
            try:
                if not (args.output_root / "failure_receipt.json").exists():
                    record_failure(
                        args,
                        stage="watcher",
                        authority=_FAILURE_AUTHORITY,
                        returncode=None,
                        error=error,
                    )
                rac_watch.base.atomic_json(
                    args.state,
                    {
                        "format": FORMAT,
                        "status": "failed",
                        "updated_at_utc": rac_watch.base.utc_now(),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
                atomic_text(args.run_exit, "1\n")
            except Exception:
                pass
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


__all__ = [
    "FORMAT",
    "RacNestedWatcherError",
    "create_or_validate_execution_authority",
    "execution_authority",
    "nested_command",
    "normalized_args",
    "report_command",
    "validate_complete_nested_rac",
    "validate_final_report",
    "validate_rac_supervisor_completion",
]
