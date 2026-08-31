#!/usr/bin/env python3
"""Run authenticated WCM LOBO folds through nested N1/N4/N8 evaluation.

This downstream watcher stays CPU-only while WCM training is incomplete.  It
authenticates all five source-only folds and the upstream RAC policy-transfer
authority, freezes the actor/protocol files, then runs the existing nested
runner with ``--critic-kind wcm``.  Completion requires replaying every signed
WCM rank receipt and independently rebuilding the policy-transfer report.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

import materialize_robotwin2_nested_n1_n4_n8_final_report_v1 as materializer
import robotwin2_wcm_future_latent_adapter_v1 as wcm_adapter
import robotwin2_wcm_future_latent_baseline_v1 as wcm
import run_robotwin2_five_body_nested_n4_n8_paired_success_v1 as nested
import watch_robotwin2_postformal_shared_head_upgrade_v1 as postformal
import watch_robotwin2_v13_rac_to_wcm_lobo_training_v1 as wcm_watch


FORMAT = "etsf_robotwin2_wcm_lobo_to_nested_success_watcher_v1"
EXECUTION_AUTHORITY_FORMAT = "etsf_robotwin2_wcm_nested_execution_authority_v1"
DEFERRED_AUTHORITY_FORMAT = "etsf_robotwin2_wcm_nested_deferred_runtime_authority_v1"
LAUNCH_RECEIPT_FORMAT = "etsf_robotwin2_wcm_nested_launch_receipt_v1"
FAILURE_RECEIPT_FORMAT = "etsf_robotwin2_wcm_nested_failure_receipt_v1"
FINAL_RECEIPT_FORMAT = "etsf_robotwin2_wcm_nested_success_receipt_v1"
BODIES = wcm.BODIES


class WcmNestedWatcherError(RuntimeError):
    pass


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    return wcm_watch.signed(value)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return wcm_watch.read_json(path, label)
    except (OSError, wcm_watch.WcmLoboSupervisorError) as error:
        raise WcmNestedWatcherError(str(error)) from error


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    try:
        wcm_watch.verify_signed(value, label)
    except wcm_watch.WcmLoboSupervisorError as error:
        raise WcmNestedWatcherError(str(error)) from error


def _stable_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = wcm.sha256_file(path)
    value = read_json(path, label)
    after = wcm.sha256_file(path)
    if before != after:
        raise WcmNestedWatcherError(f"{label} changed during authentication")
    return value, after


def _verify_named_sha(value: Mapping[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != wcm.canonical_sha256(unsigned):
        raise WcmNestedWatcherError(f"{label} {field} changed")
    return str(declared)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_retained_rac_final_authority(
    upstream: Mapping[str, Any],
) -> None:
    verify_signed(upstream, "WCM frozen upstream authority")
    rac = upstream.get("rac")
    if (
        upstream.get("format") != wcm_watch.UPSTREAM_AUTHORITY_FORMAT
        or upstream.get(
            "v13_training_and_complete_rac_policy_transfer_finished_before_wcm_gpu_use"
        )
        is not True
        or upstream.get("rac_cross_embodiment_success_measured_before_wcm_training")
        is not True
        or not isinstance(rac, Mapping)
        or rac.get("cross_embodiment_success_measured") is not True
        or type(rac.get("validated_rac_rank_receipts")) is not int
        or rac.get("validated_rac_rank_receipts", 0) <= 0
    ):
        raise WcmNestedWatcherError("WCM aggregate lacks full RAC final authority")
    for path_field, sha_field, label in (
        ("final_receipt", "final_receipt_file_sha256", "RAC final receipt"),
        ("execution_authority", "execution_authority_file_sha256", "RAC authority"),
        ("final_report", "final_report_file_sha256", "RAC final report"),
        ("training_aggregate", "training_aggregate_file_sha256", "RAC aggregate"),
    ):
        path = Path(str(rac.get(path_field, ""))).expanduser().resolve()
        if not path.is_file() or path.is_symlink() or wcm.sha256_file(path) != rac.get(
            sha_field
        ):
            raise WcmNestedWatcherError(f"retained {label} changed")


def validate_wcm_supervisor_completion(
    args: argparse.Namespace,
    *,
    validate_checkpoints: bool = True,
) -> dict[str, Any] | None:
    """Return a signed five-fold audit, or None until training is complete."""

    if not args.wcm_run_exit.is_file() or not args.wcm_state.is_file():
        return None
    exit_before = wcm.sha256_file(args.wcm_run_exit)
    exit_text = args.wcm_run_exit.read_text(encoding="utf-8").strip()
    exit_after = wcm.sha256_file(args.wcm_run_exit)
    if exit_before != exit_after:
        raise WcmNestedWatcherError("WCM run-exit changed")
    if exit_text == "1":
        raise WcmNestedWatcherError("WCM five-fold supervisor failed")
    if exit_text != "0":
        return None
    state, state_file_sha = _stable_json(args.wcm_state, "WCM supervisor state")
    if state.get("status") == "failed":
        raise WcmNestedWatcherError("WCM supervisor state reports failure")
    if state.get("status") != "complete":
        return None
    aggregate_path = args.wcm_root / "five_fold_wcm_training_summary.json"
    aggregate, aggregate_file_sha = _stable_json(
        aggregate_path, "WCM five-fold aggregate"
    )
    verify_signed(aggregate, "WCM five-fold aggregate")
    upstream = aggregate.get("upstream_authority")
    supplement = aggregate.get("supplement_binding_authority")
    rows = aggregate.get("outer_folds")
    protocol = aggregate.get("actor_execution_protocol")
    protocol_binding = aggregate.get("actor_execution_protocol_binding")
    actor_authority = aggregate.get("actor_authority")
    if not isinstance(upstream, Mapping):
        raise WcmNestedWatcherError("WCM aggregate lacks upstream authority")
    _validate_retained_rac_final_authority(upstream)
    if not isinstance(supplement, Mapping):
        raise WcmNestedWatcherError("WCM aggregate lacks supplement authority")
    verify_signed(supplement, "WCM supplement authority")
    if (
        state.get("format") != wcm_watch.FORMAT
        or state.get("final_summary") != str(aggregate_path)
        or state.get("final_summary_file_sha256") != aggregate_file_sha
        or aggregate.get("format") != wcm_watch.FINAL_FORMAT
        or aggregate.get("status")
        != "five_outer_lobo_wcm_source_only_training_complete"
        or aggregate.get("model_family") != wcm.MODEL_FAMILY
        or aggregate.get("not_official_wcm_architecture_or_weights") is not True
        or aggregate.get("fold_count") != len(BODIES)
        or aggregate.get("members_per_fold") != wcm_adapter.ENSEMBLE_SIZE
        or aggregate.get("steps_per_member") != wcm_watch.trainer.DEFAULT_STEPS
        or aggregate.get("heldout_labels_used_for_training_normalization_or_selection")
        is not False
        or aggregate.get("heldout_task_success_measured") is not False
        or not isinstance(rows, list)
        or len(rows) != len(BODIES)
        or not isinstance(protocol, Mapping)
        or not isinstance(protocol_binding, Mapping)
        or protocol_binding.get("protocol") != protocol
        or aggregate.get("actor_authority") != upstream.get("actor_authority")
        or not isinstance(actor_authority, Mapping)
        or supplement.get("binding_file_sha256")
        != upstream.get("rac", {}).get("required_supplement_binding_sha256")
    ):
        raise WcmNestedWatcherError("WCM supervisor completion binding changed")
    rows_by_body = {
        row.get("held_out_body"): row for row in rows if isinstance(row, Mapping)
    }
    if set(rows_by_body) != set(BODIES):
        raise WcmNestedWatcherError("WCM aggregate lacks five distinct folds")
    folds: dict[str, dict[str, Any]] = {}
    for body in BODIES:
        root = args.wcm_root / f"outer_lobo_{body}"
        try:
            inspection = wcm_adapter.inspect_fold(
                body,
                root,
                expected_actor_protocol_binding=protocol_binding,
            )
        except wcm_adapter.WCMAdapterError as error:
            raise WcmNestedWatcherError(str(error)) from error
        row = rows_by_body[body]
        if (
            row.get("training_summary") != inspection["training_summary"]
            or row.get("training_summary_file_sha256")
            != inspection["training_summary_sha256"]
            or row.get("training_summary_logical_sha256")
            != inspection["training_summary_logical_sha256"]
            or row.get("preflight_receipt") != inspection["preflight_receipt"]
            or row.get("preflight_receipt_file_sha256")
            != inspection["preflight_receipt_sha256"]
            or row.get("selected_step")
            != inspection["ensemble_common_selection_step"]
            or row.get("supplement_binding_file_sha256")
            != supplement["binding_file_sha256"]
        ):
            raise WcmNestedWatcherError(f"{body} WCM fold differs from aggregate")
        load_receipt = None
        if validate_checkpoints:
            try:
                ensemble, load_receipt = wcm_adapter.load_fold_ensemble(
                    inspection, device=torch.device("cpu")
                )
            except wcm_adapter.WCMAdapterError as error:
                raise WcmNestedWatcherError(str(error)) from error
            del ensemble
        folds[body] = {
            "root": str(root),
            "inspection": inspection,
            "ensemble_load_receipt": load_receipt,
        }
    base = {
        "format": "etsf_robotwin2_wcm_supervisor_completion_audit_v1",
        "wcm_run_exit": str(args.wcm_run_exit),
        "wcm_run_exit_file_sha256": exit_after,
        "wcm_state": str(args.wcm_state),
        "wcm_state_file_sha256": state_file_sha,
        "wcm_aggregate": str(aggregate_path),
        "wcm_aggregate_file_sha256": aggregate_file_sha,
        "wcm_aggregate_logical_sha256": aggregate["logical_sha256"],
        "upstream_authority": dict(upstream),
        "required_supplement_binding_sha256": supplement["binding_file_sha256"],
        "expected_actor_execution_protocol": dict(protocol),
        "expected_actor_execution_protocol_binding": dict(protocol_binding),
        "expected_actor_authority": dict(actor_authority),
        "folds": folds,
        "heldout_payloads_or_labels_opened_by_watcher": 0,
        "checkpoint_payloads_validated": bool(validate_checkpoints),
    }
    return signed(base)


def freeze_or_validate_runtime_authority(
    args: argparse.Namespace,
    completion: Mapping[str, Any],
) -> dict[str, Any] | None:
    expected_protocol = completion.get("expected_actor_execution_protocol")
    expected_binding = completion.get("expected_actor_execution_protocol_binding")
    expected_actor = completion.get("expected_actor_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (expected_protocol, expected_binding, expected_actor)
    ):
        raise WcmNestedWatcherError("WCM completion lacks runtime authority")
    protocol_sha = expected_binding.get("file_sha256")
    expected_protocol_path = (
        Path(str(expected_binding.get("path_root", ""))).expanduser().resolve()
        / str(expected_binding.get("path", ""))
    ).resolve()
    expected_actor_path = Path(str(expected_actor.get("path", ""))).expanduser().resolve()
    if (
        expected_binding.get("protocol") != expected_protocol
        or expected_binding.get("protocol_logical_sha256")
        != expected_protocol.get("logical_sha256")
        or not isinstance(protocol_sha, str)
        or len(protocol_sha) != 64
        or args.actor_execution_protocol != expected_protocol_path
        or args.actor_authority != expected_actor_path
        or (
            args.actor_execution_protocol_sha256 is not None
            and args.actor_execution_protocol_sha256 != protocol_sha
        )
    ):
        raise WcmNestedWatcherError("explicit WCM runtime authority disagrees")
    path = args.output_root / "deferred_runtime_authority.json"
    prior = None
    if path.exists():
        prior = read_json(path, "WCM deferred runtime authority")
        verify_signed(prior, "WCM deferred runtime authority")
    if not args.actor_execution_protocol.is_file() or not args.actor_authority.is_file():
        if prior is not None:
            raise WcmNestedWatcherError("frozen WCM runtime file disappeared")
        return None
    protocol_before = wcm.sha256_file(args.actor_execution_protocol)
    try:
        observed_binding = nested.formal.actor_execution.execution_protocol_file_binding(
            args.actor_execution_protocol,
            str(protocol_sha),
            path_root=Path(str(expected_binding["path_root"])),
        )
    except nested.formal.actor_execution.ActorExecutionProtocolError as error:
        raise WcmNestedWatcherError(str(error)) from error
    protocol_after = wcm.sha256_file(args.actor_execution_protocol)
    actor_before = wcm.sha256_file(args.actor_authority)
    actor = read_json(args.actor_authority, "WCM actor authority")
    actor_after = wcm.sha256_file(args.actor_authority)
    verify_signed(actor, "WCM actor authority")
    if (
        protocol_before != protocol_after
        or protocol_after != protocol_sha
        or observed_binding != expected_binding
        or actor_before != actor_after
        or actor_after != expected_actor.get("file_sha256")
        or actor.get("logical_sha256") != expected_actor.get("logical_sha256")
        or actor.get("format") != wcm_watch.v13_watch.ACTOR_FORMAT
        or actor.get("one_universal_actor_for_all_five_bodies") is not True
        or actor.get("sampling_contract", {}).get("actor_execution_protocol_binding")
        != expected_binding
    ):
        raise WcmNestedWatcherError("WCM runtime authority files changed")
    value = signed(
        {
            "format": DEFERRED_AUTHORITY_FORMAT,
            "wcm_completion_audit_logical_sha256": completion["logical_sha256"],
            "actor_execution_protocol": str(args.actor_execution_protocol),
            "actor_execution_protocol_file_sha256": protocol_after,
            "actor_execution_protocol_logical_sha256": expected_protocol[
                "logical_sha256"
            ],
            "actor_execution_protocol_binding": dict(expected_binding),
            "actor_authority": str(args.actor_authority),
            "actor_authority_file_sha256": actor_after,
            "actor_authority_logical_sha256": actor["logical_sha256"],
            "files_authenticated_after_wcm_completion": True,
            "heldout_payloads_opened_while_waiting": 0,
        }
    )
    if prior is None:
        wcm_watch.v13_watch.create_once_or_verify(
            path, value, "WCM deferred runtime authority"
        )
    elif prior != value:
        raise WcmNestedWatcherError("WCM deferred runtime authority changed")
    args.actor_execution_protocol_sha256 = protocol_sha
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
        "--critic-kind", "wcm",
        "--required-supplement-binding-sha256",
        str(completion["required_supplement_binding_sha256"]),
        "--output", str(args.output_root / "nested_wcm"),
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
        "--nested-root", str(args.output_root / "nested_wcm"),
        "--actor-authority", str(args.actor_authority),
        "--output-input", str(args.output_root / "wcm_crossbody_final_report_input.json"),
        "--output-report", str(args.output_root / "wcm_crossbody_final_report.json"),
    ]


def execution_authority(
    args: argparse.Namespace,
    completion: Mapping[str, Any],
    nested_cmd: Sequence[str],
    report_cmd: Sequence[str],
) -> dict[str, Any]:
    deferred_path = args.output_root / "deferred_runtime_authority.json"
    deferred = read_json(deferred_path, "WCM deferred runtime authority")
    verify_signed(deferred, "WCM deferred runtime authority")
    code_paths = (
        Path(__file__).resolve(),
        args.nested_runner,
        Path(wcm_adapter.__file__).resolve(),
        Path(wcm.__file__).resolve(),
        args.final_materializer,
        args.robotwin_python,
        args.system_python,
    )
    return signed(
        {
            "format": EXECUTION_AUTHORITY_FORMAT,
            "wcm_completion_audit": dict(completion),
            "deferred_runtime_authority": deferred,
            "deferred_runtime_authority_file_sha256": wcm.sha256_file(
                deferred_path
            ),
            "actor_checkpoint": str(args.actor_checkpoint),
            "actor_checkpoint_tree_sha256": wcm_watch.v13_watch.sha256_tree(
                args.actor_checkpoint
            )[0],
            "vlm_metadata": str(args.vlm_metadata),
            "vlm_metadata_tree_sha256": wcm_watch.v13_watch.sha256_tree(
                args.vlm_metadata
            )[0],
            "actor_authority": str(args.actor_authority),
            "actor_authority_file_sha256": wcm.sha256_file(args.actor_authority),
            "event_spec": str(args.event_spec),
            "event_spec_file_sha256": wcm.sha256_file(args.event_spec),
            "reference_preregistration": str(args.reference_preregistration),
            "reference_preregistration_file_sha256": wcm.sha256_file(
                args.reference_preregistration
            ),
            "actor_execution_protocol": str(args.actor_execution_protocol),
            "actor_execution_protocol_file_sha256": wcm.sha256_file(
                args.actor_execution_protocol
            ),
            "code_files": [
                {"path": str(path), "sha256": wcm.sha256_file(path)}
                for path in code_paths
            ],
            "nested_command": list(nested_cmd),
            "nested_command_sha256": wcm.canonical_sha256(list(nested_cmd)),
            "report_command": list(report_cmd),
            "report_command_sha256": wcm.canonical_sha256(list(report_cmd)),
            "nested_output": str(args.output_root / "nested_wcm"),
            "critic_kind": "wcm",
            "not_official_wcm_architecture_or_weights": True,
            "resume_requires_exact_same_authority": True,
            "heldout_labels_or_outcomes_used_to_select_retry": False,
        }
    )


def create_or_validate_execution_authority(
    args: argparse.Namespace, expected: Mapping[str, Any]
) -> dict[str, Any]:
    verify_signed(expected, "expected WCM nested execution authority")
    path = args.output_root / "execution_authority.json"
    if path.exists():
        observed = read_json(path, "WCM nested execution authority")
        verify_signed(observed, "WCM nested execution authority")
        if observed != dict(expected):
            raise WcmNestedWatcherError("WCM nested execution authority changed")
        return observed
    wcm_watch.v13_watch.create_once_or_verify(
        path, expected, "WCM nested execution authority"
    )
    return dict(expected)


def validate_complete_nested_wcm(
    args: argparse.Namespace, completion: Mapping[str, Any]
) -> dict[str, Any]:
    root = args.output_root / "nested_wcm"
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
        raise WcmNestedWatcherError(str(error)) from error
    contract = read_json(root / "execution_contract.json", "WCM nested contract")
    verify_signed(contract, "WCM nested contract")
    expected_folds = {
        body: nested.inspect_wcm_fold(
            body, Path(str(completion["folds"][body]["root"]))
        )
        for body in BODIES
    }
    if (
        contract.get("critic_kind") != "wcm"
        or contract.get("method_critic_assignment")
        != {
            nested.METHOD_ACTOR: None,
            nested.METHOD_N4: "wcm",
            nested.METHOD_N8: "wcm",
        }
        or contract.get("wcm_rank_receipt_format")
        != nested.WCM_RANK_RECEIPT_FORMAT
        or contract.get("folds") != expected_folds
        or contract.get("fold_training_regime", {}).get(
            "supplement_binding_file_sha256"
        )
        != completion["required_supplement_binding_sha256"]
        or contract.get("actor_execution_protocol") != protocol
        or contract.get("not_official_wcm_architecture_or_weights") is not True
    ):
        raise WcmNestedWatcherError("nested WCM contract changed")
    pair_dir = root / "pairs"
    expected_paths = {
        pair_dir
        / f"{nested.pair_id(row['heldout_body'], row['condition'], row['requested_seed'])}.json"
        for row in nested.evaluation_schedule()
    }
    observed_paths = {path for path in pair_dir.glob("*.json") if path.is_file()}
    if observed_paths != expected_paths:
        raise WcmNestedWatcherError("nested WCM pair roster changed")
    validated = 0
    for expected in nested.evaluation_schedule():
        body = str(expected["heldout_body"])
        path = pair_dir / (
            nested.pair_id(body, expected["condition"], expected["requested_seed"])
            + ".json"
        )
        pair = read_json(path, "nested WCM pair")
        _verify_named_sha(pair, "pair_sha256", "nested WCM pair")
        rollouts = pair.get("rollouts")
        if not isinstance(rollouts, Mapping) or set(rollouts) != set(nested.METHODS):
            raise WcmNestedWatcherError("nested WCM pair rollout roster changed")
        normalizer_sha = contract["source_train_action_normalizer_by_heldout_body"][
            body
        ]["logical_sha256"]
        expected_load = completion["folds"][body]["ensemble_load_receipt"]
        if not isinstance(expected_load, Mapping):
            raise WcmNestedWatcherError("WCM completion lacks checkpoint receipt")
        for method in nested.METHODS:
            try:
                nested.validate_rollout(
                    rollouts[method],
                    method=method,
                    expected=expected,
                    expected_source_action_normalizer_logical_sha256=normalizer_sha,
                )
            except nested.NestedCandidatePoolError as error:
                raise WcmNestedWatcherError(str(error)) from error
            if method == nested.METHOD_ACTOR:
                continue
            for decision in rollouts[method]["decisions"]:
                scores = decision.get("critic_scores")
                receipt = scores.get("rank_receipt") if isinstance(scores, Mapping) else None
                if (
                    not isinstance(receipt, Mapping)
                    or scores.get("critic_kind") != "wcm"
                    or receipt.get("critic_ensemble_load_receipt") != expected_load
                ):
                    raise WcmNestedWatcherError(
                        "nested decision differs from authenticated WCM fold"
                    )
                validated += 1
    if validated <= 0:
        raise WcmNestedWatcherError("nested WCM run contains no rank receipts")
    return {
        **audit,
        "contract_logical_sha256": contract["logical_sha256"],
        "critic_kind": "wcm",
        "validated_wcm_rank_receipts": validated,
        "all_wcm_rank_receipts_replayed": True,
    }


def validate_final_report(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.output_root / "wcm_crossbody_final_report_input.json"
    report_path = args.output_root / "wcm_crossbody_final_report.json"
    report, report_file_sha = _stable_json(report_path, "WCM final report")
    report_sha = _verify_named_sha(report, "report_sha256", "WCM final report")
    materialized_input, input_file_sha = _stable_json(
        input_path, "WCM final report input"
    )
    verify_signed(materialized_input, "WCM final report input")
    try:
        replayed_input, replayed_report = materializer.build_materialization(
            nested_root=args.output_root / "nested_wcm",
            actor_authority_path=args.actor_authority,
            oracle_truth_path=None,
        )
    except materializer.FinalReportMaterializationError as error:
        raise WcmNestedWatcherError(str(error)) from error
    oracle = report.get("oracle_branch_diagnostic")
    if (
        materialized_input != replayed_input
        or report != replayed_report
        or report.get("format") != materializer.evaluator.REPORT_FORMAT
        or report.get("status") != materializer.evaluator.POLICY_ONLY_STATUS
        or report.get("input_document_sha256")
        != materialized_input["logical_sha256"]
        or not isinstance(oracle, Mapping)
        or oracle.get("evidence_sufficient") is not False
        or oracle.get("oracle_regret_reported") is not False
    ):
        raise WcmNestedWatcherError("WCM final report did not fail closed")
    return {
        "input": str(input_path),
        "input_file_sha256": input_file_sha,
        "input_logical_sha256": materialized_input["logical_sha256"],
        "report": str(report_path),
        "report_file_sha256": report_file_sha,
        "report_sha256": report_sha,
        "status": report["status"],
        "oracle_evidence_sufficient": False,
    }


def record_failure(
    args: argparse.Namespace,
    *,
    stage: str,
    authority: Mapping[str, Any] | None,
    returncode: int | None,
    error: BaseException | None,
) -> dict[str, Any]:
    value = signed(
        {
            "format": FAILURE_RECEIPT_FORMAT,
            "stage": stage,
            "execution_authority_logical_sha256": (
                authority.get("logical_sha256") if authority is not None else None
            ),
            "returncode": returncode,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
            "recorded_at_utc": wcm_watch.v13_watch.utc_now(),
            "retry_selected_using_nested_outcomes": False,
        }
    )
    wcm_watch.v13_watch.create_once_or_verify(
        args.output_root / "failure_receipt.json", value, "WCM nested failure"
    )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wcm-root", type=Path, required=True)
    parser.add_argument("--wcm-state", type=Path, required=True)
    parser.add_argument("--wcm-run-exit", type=Path, required=True)
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
    parser.add_argument(
        "--system-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python"),
    )
    parser.add_argument(
        "--expected-gpu-uuid", default=wcm_watch.v13_watch.EXPECTED_GPU_UUID
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "wcm_root", "wcm_state", "wcm_run_exit", "actor_checkpoint",
        "vlm_metadata", "actor_authority", "robotwin_root", "event_spec",
        "reference_preregistration", "actor_execution_protocol", "path_root",
        "output_root", "state", "run_exit", "nested_runner",
        "final_materializer", "robotwin_python", "system_python",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.lock = (
        args.lock.expanduser().resolve()
        if args.lock is not None
        else args.output_root / "wcm_nested.lock"
    )
    if args.poll_seconds <= 0:
        raise WcmNestedWatcherError("poll interval must be positive")
    if args.actor_execution_protocol_sha256 is not None and (
        len(args.actor_execution_protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.actor_execution_protocol_sha256
        )
    ):
        raise WcmNestedWatcherError("explicit actor protocol SHA is invalid")
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
        raise WcmNestedWatcherError("WCM nested output must be inside path_root") from error
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
        raise WcmNestedWatcherError("another WCM nested watcher is active") from error

    def write_state(status: str, **extra: Any) -> None:
        wcm_watch.v13_watch.atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": wcm_watch.v13_watch.utc_now(),
                "pid": os.getpid(),
                "wcm_root": str(args.wcm_root),
                "output_root": str(args.output_root),
                "critic_kind": "wcm",
                **extra,
            },
        )

    if (args.output_root / "failure_receipt.json").exists():
        raise WcmNestedWatcherError("prior immutable WCM nested failure exists")
    completion = None
    while completion is None:
        completion = validate_wcm_supervisor_completion(
            args, validate_checkpoints=False
        )
        if completion is None:
            write_state(
                "waiting_for_authenticated_wcm_five_fold_completion",
                gpu_reserved_by_watcher=False,
                gpu_identity_queried=False,
                heldout_payloads_opened_while_waiting=0,
            )
            time.sleep(args.poll_seconds)
    deferred = None
    while deferred is None:
        deferred = freeze_or_validate_runtime_authority(args, completion)
        if deferred is None:
            write_state(
                "waiting_for_wcm_deferred_protocol_and_actor_authority",
                wcm_completion_audit_logical_sha256=completion["logical_sha256"],
                gpu_reserved_by_watcher=False,
                gpu_identity_queried=False,
                heldout_payloads_opened_while_waiting=0,
            )
            time.sleep(args.poll_seconds)
    checked = validate_wcm_supervisor_completion(args, validate_checkpoints=True)
    if (
        checked is None
        or checked.get("wcm_aggregate_logical_sha256")
        != completion.get("wcm_aggregate_logical_sha256")
        or checked.get("upstream_authority") != completion.get("upstream_authority")
        or checked.get("checkpoint_payloads_validated") is not True
    ):
        raise WcmNestedWatcherError("WCM completion changed before checkpoint load")
    completion = checked
    gpu = wcm_watch.v13_watch.gpu_identity()
    if "4090" not in gpu["name"] or gpu["uuid"] != args.expected_gpu_uuid:
        raise WcmNestedWatcherError(f"unexpected GPU authority: {gpu}")
    nested_cmd = nested_command(args, completion)
    report_cmd = report_command(args)
    authority = create_or_validate_execution_authority(
        args, execution_authority(args, completion, nested_cmd, report_cmd)
    )
    _FAILURE_AUTHORITY = authority
    launch = signed(
        {
            "format": LAUNCH_RECEIPT_FORMAT,
            "execution_authority_logical_sha256": authority["logical_sha256"],
            "nested_command_sha256": authority["nested_command_sha256"],
            "nested_output": authority["nested_output"],
            "same_command_resume_allowed_after_unrecorded_process_interruption": True,
            "retry_selection_reads_nested_outcomes": False,
        }
    )
    wcm_watch.v13_watch.create_once_or_verify(
        args.output_root / "nested_launch_receipt.json",
        launch,
        "WCM nested launch receipt",
    )
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
    nested_root = args.output_root / "nested_wcm"
    if not (nested_root / "completion_receipt.json").is_file():
        while wcm_watch.v13_watch.gpu_compute_pids():
            write_state(
                "waiting_for_idle_gpu_before_wcm_nested",
                gpu=gpu,
                gpu_reserved_by_watcher=False,
                execution_authority_logical_sha256=authority["logical_sha256"],
            )
            time.sleep(args.poll_seconds)
        write_state(
            "running_wcm_nested_n1_n4_n8",
            command=nested_cmd,
            gpu=gpu,
            execution_authority_logical_sha256=authority["logical_sha256"],
        )
        with (args.output_root / "wcm_nested.log").open("a", encoding="utf-8") as stream:
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
                stage="nested_wcm",
                authority=authority,
                returncode=result.returncode,
                error=None,
            )
            raise WcmNestedWatcherError(
                f"WCM nested runner exited {result.returncode}"
            )
    nested_audit = validate_complete_nested_wcm(args, completion)
    report_path = args.output_root / "wcm_crossbody_final_report.json"
    if not report_path.is_file():
        write_state(
            "materializing_wcm_crossbody_final_report",
            nested_completion_audit=nested_audit,
            command=report_cmd,
            execution_authority_logical_sha256=authority["logical_sha256"],
        )
        with (args.output_root / "wcm_final_report.log").open(
            "a", encoding="utf-8"
        ) as stream:
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
            raise WcmNestedWatcherError(
                f"WCM report materializer exited {result.returncode}"
            )
    final_report = validate_final_report(args)
    final = signed(
        {
            "format": FINAL_RECEIPT_FORMAT,
            "status": "complete_wcm_nested_policy_transfer_report",
            "execution_authority_logical_sha256": authority["logical_sha256"],
            "wcm_completion_audit_logical_sha256": completion["logical_sha256"],
            "nested_completion_audit": nested_audit,
            "final_report": final_report,
            "critic_kind": "wcm",
            "not_official_wcm_architecture_or_weights": True,
            "heldout_labels_used_for_training_or_checkpoint_selection": False,
            "cross_embodiment_success_measured": True,
        }
    )
    final_path = args.output_root / "wcm_nested_success_receipt.json"
    wcm_watch.v13_watch.create_once_or_verify(
        final_path, final, "WCM nested success receipt"
    )
    _atomic_text(args.run_exit, "0\n")
    write_state(
        "complete",
        gpu=gpu,
        deferred_runtime_authority=deferred,
        execution_authority_logical_sha256=authority["logical_sha256"],
        nested_completion_audit=nested_audit,
        final_report=final_report,
        final_receipt=str(final_path),
        final_receipt_file_sha256=wcm.sha256_file(final_path),
        final_receipt_logical_sha256=final["logical_sha256"],
        gpu_reserved_by_watcher=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if _FAILURE_ARGS is not None:
            try:
                if not (_FAILURE_ARGS.output_root / "failure_receipt.json").exists():
                    record_failure(
                        _FAILURE_ARGS,
                        stage="watcher",
                        authority=_FAILURE_AUTHORITY,
                        returncode=None,
                        error=error,
                    )
                wcm_watch.v13_watch.atomic_json(
                    _FAILURE_ARGS.state,
                    {
                        "format": FORMAT,
                        "status": "failed",
                        "updated_at_utc": wcm_watch.v13_watch.utc_now(),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
                _atomic_text(_FAILURE_ARGS.run_exit, "1\n")
            except Exception:
                pass
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


__all__ = [
    "FINAL_RECEIPT_FORMAT",
    "FORMAT",
    "WcmNestedWatcherError",
    "create_or_validate_execution_authority",
    "execution_authority",
    "freeze_or_validate_runtime_authority",
    "nested_command",
    "normalized_args",
    "report_command",
    "validate_complete_nested_wcm",
    "validate_final_report",
    "validate_wcm_supervisor_completion",
]
