#!/usr/bin/env python3
"""Wait for v13 and RAC final authorities, then train five WCM LOBO folds.

The supervisor is CPU-only while waiting.  It freezes the primary and deferred
supplement bindings only after both upstream chains have completed successfully,
waits for an idle authorized RTX 4090, and promotes each validated fold from an
attempt directory by one same-filesystem rename.  Resume never selects an
attempt from model outcomes: completed folds are replay-validated, a completed
staging output can finish promotion, and only a recorded non-informative process
interruption may start a fresh retained attempt.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import robotwin2_wcm_future_latent_baseline_v1 as wcm
import train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1 as trainer
import watch_robotwin2_five_body_branches_to_lobo_training_v1 as v13_watch
import watch_robotwin2_five_body_branches_to_rac_lobo_training_v1 as rac_watch
import watch_robotwin2_rac_lobo_to_nested_success_v1 as rac_final_watch


FORMAT = "etsf_robotwin2_v13_rac_to_wcm_lobo_supervisor_v1"
FINAL_FORMAT = "etsf_robotwin2_five_body_wcm_lobo_source_validation_aggregate_v1"
UPSTREAM_AUTHORITY_FORMAT = "etsf_robotwin2_wcm_upstream_v13_rac_authority_v1"
SUPPLEMENT_AUTHORITY_FORMAT = "etsf_robotwin2_wcm_deferred_supplement_authority_v1"
ATTEMPT_FORMAT = "etsf_robotwin2_wcm_lobo_attempt_v1"
LAUNCH_FORMAT = "etsf_robotwin2_wcm_lobo_attempt_launch_v1"
FAILURE_FORMAT = "etsf_robotwin2_wcm_lobo_attempt_failure_v1"
REBIND_FORMAT = "etsf_robotwin2_wcm_lobo_summary_rebind_v1"
PROMOTION_FORMAT = "etsf_robotwin2_wcm_lobo_attempt_promotion_v1"
BODIES = v13_watch.BODIES
RECOVERABLE_SIGNALS = frozenset(
    int(value)
    for value in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
)


class WcmLoboSupervisorError(RuntimeError):
    pass


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    return v13_watch.signed(value)


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    try:
        v13_watch.verify_logical_sha(value, label)
    except v13_watch.LoboWatcherError as error:
        raise WcmLoboSupervisorError(str(error)) from error


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return v13_watch.read_json(path, label)
    except (OSError, v13_watch.LoboWatcherError) as error:
        raise WcmLoboSupervisorError(str(error)) from error


def stable_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = v13_watch.sha256_file(path)
    value = read_json(path, label)
    after = v13_watch.sha256_file(path)
    if before != after:
        raise WcmLoboSupervisorError(f"{label} changed during authentication")
    return value, after


def _dependency(
    *,
    name: str,
    run_exit: Path,
    state_path: Path,
    expected_state_format: str,
    expected_final_format: str,
    expected_final_status: str,
) -> dict[str, Any] | None:
    if not run_exit.is_file() or not state_path.is_file():
        return None
    exit_before = v13_watch.sha256_file(run_exit)
    exit_text = run_exit.read_text(encoding="utf-8").strip()
    exit_after = v13_watch.sha256_file(run_exit)
    if exit_before != exit_after:
        raise WcmLoboSupervisorError(f"{name} run-exit changed")
    if exit_text == "1":
        raise WcmLoboSupervisorError(f"{name} upstream failed")
    if exit_text != "0":
        return None
    state, state_sha = stable_json(state_path, f"{name} state")
    if state.get("status") == "failed":
        raise WcmLoboSupervisorError(f"{name} state reports failure")
    if state.get("status") != "complete":
        return None
    summary_raw = state.get("final_summary")
    if state.get("format") != expected_state_format or not isinstance(
        summary_raw, str
    ):
        raise WcmLoboSupervisorError(f"{name} completion state changed")
    summary_path = Path(summary_raw).expanduser().resolve()
    summary, summary_sha = stable_json(summary_path, f"{name} final summary")
    verify_signed(summary, f"{name} final summary")
    if (
        state.get("final_summary_file_sha256") != summary_sha
        or summary.get("format") != expected_final_format
        or summary.get("status") != expected_final_status
        or summary.get("fold_count") != len(BODIES)
        or summary.get("members_per_fold") != 5
        or summary.get("steps_per_member") != 3000
    ):
        raise WcmLoboSupervisorError(f"{name} final authority changed")
    return {
        "name": name,
        "run_exit": str(run_exit),
        "run_exit_file_sha256": exit_after,
        "state": str(state_path),
        "state_file_sha256": state_sha,
        "final_summary": str(summary_path),
        "final_summary_file_sha256": summary_sha,
        "final_summary_logical_sha256": summary["logical_sha256"],
        "training_binding": summary.get("training_binding"),
        "actor_execution_protocol": summary.get("actor_execution_protocol"),
        "actor_execution_protocol_binding": summary.get(
            "actor_execution_protocol_binding"
        ),
        "completed_before_wcm_training": True,
    }


def _verify_named_sha(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != v13_watch.canonical_sha256(unsigned):
        raise WcmLoboSupervisorError(f"{label} {field} changed")
    return str(declared)


def _rac_final_dependency(args: argparse.Namespace) -> dict[str, Any] | None:
    """Authenticate the completed RAC policy-transfer report without GPU use."""

    if not args.rac_run_exit.is_file() or not args.rac_state.is_file():
        return None
    exit_before = v13_watch.sha256_file(args.rac_run_exit)
    exit_text = args.rac_run_exit.read_text(encoding="utf-8").strip()
    exit_after = v13_watch.sha256_file(args.rac_run_exit)
    if exit_before != exit_after:
        raise WcmLoboSupervisorError("RAC final run-exit changed")
    if exit_text == "1":
        raise WcmLoboSupervisorError("RAC final policy-transfer chain failed")
    if exit_text != "0":
        return None
    state, state_sha = stable_json(args.rac_state, "RAC final watcher state")
    if state.get("status") == "failed":
        raise WcmLoboSupervisorError("RAC final watcher state reports failure")
    if state.get("status") != "complete":
        return None
    if state.get("format") != rac_final_watch.FORMAT:
        raise WcmLoboSupervisorError("RAC final watcher format changed")

    receipt_path = Path(str(state.get("final_receipt", ""))).expanduser().resolve()
    expected_receipt_path = args.rac_final_root / "rac_nested_success_receipt.json"
    authority_path = args.rac_final_root / "execution_authority.json"
    if receipt_path != expected_receipt_path:
        raise WcmLoboSupervisorError("RAC final receipt escaped its fixed root")
    receipt, receipt_file_sha = stable_json(receipt_path, "RAC final receipt")
    verify_signed(receipt, "RAC final receipt")
    authority, authority_file_sha = stable_json(
        authority_path, "RAC final execution authority"
    )
    verify_signed(authority, "RAC final execution authority")
    completion = authority.get("rac_completion_audit")
    nested_audit = receipt.get("nested_completion_audit")
    report_binding = receipt.get("final_report")
    if not isinstance(completion, Mapping):
        raise WcmLoboSupervisorError("RAC final authority lacks training audit")
    verify_signed(completion, "RAC training completion audit")
    if not isinstance(nested_audit, Mapping) or not isinstance(
        report_binding, Mapping
    ):
        raise WcmLoboSupervisorError("RAC final receipt lacks evaluation evidence")
    report_path = Path(str(report_binding.get("report", ""))).expanduser().resolve()
    expected_report_path = args.rac_final_root / "rac_crossbody_final_report.json"
    if report_path != expected_report_path:
        raise WcmLoboSupervisorError("RAC final report escaped its fixed root")
    report, report_file_sha = stable_json(report_path, "RAC final report")
    report_logical_sha = _verify_named_sha(
        report, "report_sha256", "RAC final report"
    )
    aggregate_path = Path(
        str(completion.get("rac_aggregate", ""))
    ).expanduser().resolve()
    aggregate, aggregate_file_sha = stable_json(
        aggregate_path, "RAC training aggregate"
    )
    verify_signed(aggregate, "RAC training aggregate")
    protocol = completion.get("expected_actor_execution_protocol")
    protocol_binding = completion.get("expected_actor_execution_protocol_binding")
    actor_authority = completion.get("expected_actor_authority")
    training_binding = aggregate.get("training_binding")
    validated_receipts = nested_audit.get("validated_rac_rank_receipts")
    if (
        state.get("final_receipt_file_sha256") != receipt_file_sha
        or state.get("final_receipt_logical_sha256")
        != receipt.get("logical_sha256")
        or receipt.get("format") != rac_final_watch.FINAL_RECEIPT_FORMAT
        or receipt.get("status")
        != "complete_rac_nested_policy_transfer_report"
        or receipt.get("critic_kind") != "rac"
        or receipt.get("cross_embodiment_success_measured") is not True
        or receipt.get("heldout_labels_used_for_training_or_checkpoint_selection")
        is not False
        or authority.get("format") != rac_final_watch.EXECUTION_AUTHORITY_FORMAT
        or authority.get("critic_kind") != "rac"
        or receipt.get("execution_authority_logical_sha256")
        != authority.get("logical_sha256")
        or receipt.get("rac_completion_audit_logical_sha256")
        != completion.get("logical_sha256")
        or nested_audit.get("critic_kind") != "rac"
        or nested_audit.get("all_rac_rank_receipts_replayed") is not True
        or type(validated_receipts) is not int
        or validated_receipts <= 0
        or report_binding.get("report_file_sha256") != report_file_sha
        or report_binding.get("report_sha256") != report_logical_sha
        or report_binding.get("status")
        != rac_final_watch.materializer.evaluator.POLICY_ONLY_STATUS
        or report.get("status")
        != rac_final_watch.materializer.evaluator.POLICY_ONLY_STATUS
        or report.get("oracle_branch_diagnostic", {}).get("evidence_sufficient")
        is not False
        or aggregate_file_sha != completion.get("rac_aggregate_file_sha256")
        or aggregate.get("logical_sha256")
        != completion.get("rac_aggregate_logical_sha256")
        or aggregate.get("format") != rac_watch.FINAL_FORMAT
        or aggregate.get("status")
        != "five_outer_lobo_rac_source_only_training_complete"
        or not isinstance(training_binding, Mapping)
        or not isinstance(protocol, Mapping)
        or not isinstance(protocol_binding, Mapping)
        or protocol_binding.get("protocol") != protocol
        or not isinstance(actor_authority, Mapping)
        or completion.get("checkpoint_payloads_validated") is not True
    ):
        raise WcmLoboSupervisorError("RAC final policy-transfer authority changed")
    return {
        "name": "rac_final_policy_transfer",
        "run_exit": str(args.rac_run_exit),
        "run_exit_file_sha256": exit_after,
        "state": str(args.rac_state),
        "state_file_sha256": state_sha,
        "final_receipt": str(receipt_path),
        "final_receipt_file_sha256": receipt_file_sha,
        "final_receipt_logical_sha256": receipt["logical_sha256"],
        "execution_authority": str(authority_path),
        "execution_authority_file_sha256": authority_file_sha,
        "execution_authority_logical_sha256": authority["logical_sha256"],
        "final_report": str(report_path),
        "final_report_file_sha256": report_file_sha,
        "final_report_logical_sha256": report_logical_sha,
        "training_aggregate": str(aggregate_path),
        "training_aggregate_file_sha256": aggregate_file_sha,
        "training_aggregate_logical_sha256": aggregate["logical_sha256"],
        "training_binding": dict(training_binding),
        "required_supplement_binding_sha256": completion[
            "required_supplement_binding_sha256"
        ],
        "actor_execution_protocol": dict(protocol),
        "actor_execution_protocol_binding": dict(protocol_binding),
        "actor_authority": dict(actor_authority),
        "validated_rac_rank_receipts": validated_receipts,
        "cross_embodiment_success_measured": True,
        "completed_before_wcm_training": True,
    }


def freeze_upstream_authority(args: argparse.Namespace) -> dict[str, Any] | None:
    v13 = _dependency(
        name="v13",
        run_exit=args.v13_run_exit,
        state_path=args.v13_state,
        expected_state_format=v13_watch.FORMAT,
        expected_final_format=v13_watch.FINAL_FORMAT,
        expected_final_status="five_outer_lobo_source_only_training_complete",
    )
    rac = _rac_final_dependency(args)
    if v13 is None or rac is None:
        return None
    binding_sha = v13_watch.sha256_file(args.binding)
    v13_binding = v13.get("training_binding")
    rac_binding = rac.get("training_binding")
    if (
        not isinstance(v13_binding, Mapping)
        or not isinstance(rac_binding, Mapping)
        or v13_binding.get("file_sha256") != binding_sha
        or rac_binding.get("file_sha256") != binding_sha
        or v13.get("actor_execution_protocol")
        != rac.get("actor_execution_protocol")
        or v13.get("actor_execution_protocol_binding")
        != rac.get("actor_execution_protocol_binding")
    ):
        raise WcmLoboSupervisorError("v13/RAC primary or actor authority disagrees")
    base = {
        "format": UPSTREAM_AUTHORITY_FORMAT,
        "v13": v13,
        "rac": rac,
        "primary_binding": str(args.binding),
        "primary_binding_file_sha256": binding_sha,
        "actor_execution_protocol": v13["actor_execution_protocol"],
        "actor_execution_protocol_binding": v13[
            "actor_execution_protocol_binding"
        ],
        "actor_authority": rac["actor_authority"],
        "v13_training_and_complete_rac_policy_transfer_finished_before_wcm_gpu_use": True,
        "rac_cross_embodiment_success_measured_before_wcm_training": True,
        "heldout_payloads_opened_while_waiting": 0,
    }
    value = signed(base)
    path = args.output_root / "upstream_v13_rac_authority.json"
    if path.exists():
        observed = read_json(path, "WCM upstream authority")
        verify_signed(observed, "WCM upstream authority")
        if observed != value:
            raise WcmLoboSupervisorError("frozen WCM upstream authority changed")
        return observed
    v13_watch.create_once_or_verify(path, value, "WCM upstream authority")
    return value


def freeze_supplement_authority(
    args: argparse.Namespace, upstream: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not args.supplement_binding.is_file():
        return None
    before = v13_watch.sha256_file(args.supplement_binding)
    if args.supplement_binding_sha256 is not None and (
        args.supplement_binding_sha256 != before
    ):
        raise WcmLoboSupervisorError("explicit supplement SHA changed")
    protocol = upstream["actor_execution_protocol"]
    protocol_binding = upstream["actor_execution_protocol_binding"]
    try:
        document = v13_watch.validate_supplement_frame_binding(
            args.supplement_binding,
            execution_protocol=protocol,
            execution_protocol_binding=protocol_binding,
        )
    except v13_watch.LoboWatcherError as error:
        raise WcmLoboSupervisorError(str(error)) from error
    after = v13_watch.sha256_file(args.supplement_binding)
    if before != after:
        raise WcmLoboSupervisorError("supplement changed during deferred freeze")
    if after != upstream["rac"]["required_supplement_binding_sha256"]:
        raise WcmLoboSupervisorError(
            "WCM supplement differs from completed RAC policy-transfer regime"
        )
    base = {
        "format": SUPPLEMENT_AUTHORITY_FORMAT,
        "binding": str(args.supplement_binding),
        "binding_file_sha256": after,
        "binding_logical_sha256": document["logical_sha256"],
        "upstream_authority_logical_sha256": upstream["logical_sha256"],
        "frozen_only_after_v13_and_rac_completion": True,
        "heldout_manifest_or_payload_opened": 0,
    }
    value = signed(base)
    path = args.output_root / "supplement_binding_authority.json"
    if path.exists():
        observed = read_json(path, "WCM supplement authority")
        verify_signed(observed, "WCM supplement authority")
        if observed != value:
            raise WcmLoboSupervisorError("frozen WCM supplement authority changed")
    else:
        v13_watch.create_once_or_verify(path, value, "WCM supplement authority")
    args.supplement_binding_sha256 = after
    return value


def fold_training_command(
    args: argparse.Namespace,
    body: str,
    output: Path,
    binding_sha256: str,
) -> list[str]:
    if body not in BODIES or args.supplement_binding_sha256 is None:
        raise WcmLoboSupervisorError("WCM fold command lacks frozen authority")
    return [
        str(args.training_python),
        str(args.trainer),
        "--mode", "train-fold",
        "--binding", str(args.binding),
        "--binding-sha256", binding_sha256,
        "--supplement-binding", str(args.supplement_binding),
        "--supplement-binding-sha256", args.supplement_binding_sha256,
        "--held-out-body", body,
        "--split-seed", str(trainer.DEFAULT_SPLIT_SEED),
        "--output", str(output),
        "--device", "cuda",
        "--steps", str(trainer.DEFAULT_STEPS),
        "--eval-every", str(trainer.DEFAULT_EVAL_EVERY),
        "--batch-size", str(trainer.DEFAULT_BATCH_SIZE),
        "--learning-rate", str(trainer.DEFAULT_LEARNING_RATE),
        "--ensemble-seeds",
        *[str(seed) for seed in trainer.DEFAULT_ENSEMBLE_SEEDS],
    ]


def _summary(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    verify_signed(value, label)
    return value


def validate_fold(
    root: Path,
    *,
    body: str,
    binding_sha256: str,
    supplement_sha256: str,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    preflight_path = root / "preflight_receipt.json"
    summary_path = root / "training_summary.json"
    preflight = _summary(preflight_path, f"{body} WCM preflight")
    summary = _summary(summary_path, f"{body} WCM summary")
    source_bodies = [candidate for candidate in BODIES if candidate != body]
    selection = summary.get("ensemble_checkpoint_selection")
    budget = summary.get("training_budget")
    normalization = summary.get("normalization")
    supplement = summary.get("supplement")
    members = summary.get("members")
    if (
        preflight.get("format") != trainer.FORMAT
        or preflight.get("status")
        != "matched_wcm_preflight_passed_payloads_still_unopened"
        or preflight.get("held_out_body") != body
        or preflight.get("source_bodies") != source_bodies
        or preflight.get("primary_preflight", {}).get("binding_file_sha256")
        != binding_sha256
        or preflight.get("primary_preflight", {}).get("supplement", {}).get(
            "binding_file_sha256"
        )
        != supplement_sha256
        or preflight.get("heldout_group_npz_opened") != 0
        or preflight.get("heldout_group_payload_bytes_read") != 0
        or preflight.get(
            "heldout_labels_used_for_normalization_training_or_selection"
        )
        is not False
        or summary.get("format") != trainer.SUMMARY_FORMAT
        or summary.get("status")
        != "five_member_source_only_common_step_complete"
        or summary.get("model_family") != wcm.MODEL_FAMILY
        or summary.get("held_out_body") != body
        or summary.get("source_bodies") != source_bodies
        or summary.get("primary_binding_file_sha256") != binding_sha256
        or summary.get("supplement_binding_file_sha256") != supplement_sha256
        or summary.get("trainer_file_sha256")
        != v13_watch.sha256_file(args_trainer_path())
        or not isinstance(budget, Mapping)
        or budget.get("steps_per_member") != trainer.DEFAULT_STEPS
        or budget.get("eval_every_steps") != trainer.DEFAULT_EVAL_EVERY
        or budget.get("batch_size_rows") != trainer.DEFAULT_BATCH_SIZE
        or budget.get("learning_rate") != trainer.DEFAULT_LEARNING_RATE
        or budget.get("ensemble_members") != 5
        or not isinstance(normalization, Mapping)
        or normalization.get("supplement_rows_used") != 0
        or normalization.get("validation_rows_used") != 0
        or normalization.get("heldout_rows_used") != 0
        or not isinstance(supplement, Mapping)
        or supplement.get("enabled") is not True
        or supplement.get("normalization_rows_used") != 0
        or supplement.get("heldout_manifest_or_payload_opened") != 0
        or not isinstance(selection, Mapping)
        or selection.get("common_step_required_for_all_five_members") is not True
        or selection.get("heldout_rows_used") != 0
        or not isinstance(selection.get("selected_step"), int)
        or not isinstance(members, list)
        or len(members) != 5
        or summary.get(
            "heldout_labels_used_for_normalization_training_or_selection"
        )
        is not False
        or summary.get("task_success_evaluation_authorized") is not False
    ):
        raise WcmLoboSupervisorError(f"{body} WCM fold contract changed")
    checkpoint_paths: list[Path] = []
    for index, item in enumerate(members):
        if not isinstance(item, Mapping):
            raise WcmLoboSupervisorError("WCM member is invalid")
        checkpoint = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        try:
            checkpoint.relative_to(root)
        except ValueError as error:
            raise WcmLoboSupervisorError("WCM checkpoint escapes fold root") from error
        if (
            item.get("member") != index
            or item.get("seed") != trainer.DEFAULT_ENSEMBLE_SEEDS[index]
            or item.get("best_step") != selection["selected_step"]
            or not checkpoint.is_file()
            or checkpoint.is_symlink()
            or v13_watch.sha256_file(checkpoint) != item.get("checkpoint_sha256")
        ):
            raise WcmLoboSupervisorError("WCM member binding changed")
        checkpoint_paths.append(checkpoint)
    try:
        ensemble, receipts = wcm.load_five_member_ensemble(
            checkpoint_paths, map_location="cpu"
        )
    except (wcm.WCMBaselineError, RuntimeError, ValueError) as error:
        raise WcmLoboSupervisorError(str(error)) from error
    del ensemble
    return {
        "held_out_body": body,
        "source_bodies": source_bodies,
        "model_family": wcm.MODEL_FAMILY,
        "member_count": 5,
        "steps_per_member": trainer.DEFAULT_STEPS,
        "selected_step": selection["selected_step"],
        "training_summary": str(summary_path),
        "training_summary_file_sha256": v13_watch.sha256_file(summary_path),
        "training_summary_logical_sha256": summary["logical_sha256"],
        "preflight_receipt": str(preflight_path),
        "preflight_receipt_file_sha256": v13_watch.sha256_file(preflight_path),
        "preflight_receipt_logical_sha256": preflight["logical_sha256"],
        "checkpoint_sha256s": [item["checkpoint_sha256"] for item in members],
        "source_validation": [
            item["source_validation"] for item in members
        ],
        "normalization_logical_sha256": normalization["logical_sha256"],
        "supplement_binding_file_sha256": supplement_sha256,
        "heldout_labels_used_for_training_normalization_or_selection": False,
    }


_ACTIVE_TRAINER_PATH: Path | None = None


def args_trainer_path() -> Path:
    if _ACTIVE_TRAINER_PATH is None:
        raise WcmLoboSupervisorError("trainer path is not configured")
    return _ACTIVE_TRAINER_PATH


def _attempt_root(args: argparse.Namespace, body: str) -> Path:
    return args.output_root / "wcm_fold_attempts" / body


def _attempts(args: argparse.Namespace, body: str) -> list[dict[str, Any]]:
    root = _attempt_root(args, body)
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise WcmLoboSupervisorError("WCM attempt root is invalid")
    entries = sorted(root.glob("attempt-*"))
    result = []
    for ordinal, directory in enumerate(entries, start=1):
        if directory.name != f"attempt-{ordinal:06d}" or directory.is_symlink():
            raise WcmLoboSupervisorError("WCM attempt order changed")
        manifest = _summary(directory / "attempt.json", "WCM attempt manifest")
        if (
            manifest.get("format") != ATTEMPT_FORMAT
            or manifest.get("held_out_body") != body
            or manifest.get("attempt_ordinal") != ordinal
            or manifest.get("attempt_directory") != str(directory)
        ):
            raise WcmLoboSupervisorError("WCM attempt manifest changed")
        receipts = {}
        for key, filename, expected_format in (
            ("launch", "launch.json", LAUNCH_FORMAT),
            ("failure", "failure.json", FAILURE_FORMAT),
            ("rebind", "summary_rebind.json", REBIND_FORMAT),
            ("promotion", "promotion.json", PROMOTION_FORMAT),
        ):
            path = directory / filename
            if path.exists():
                value = _summary(path, f"WCM attempt {key}")
                if (
                    value.get("format") != expected_format
                    or value.get("attempt_logical_sha256")
                    != manifest["logical_sha256"]
                ):
                    raise WcmLoboSupervisorError(f"WCM {key} receipt changed")
                receipts[key] = value
            else:
                receipts[key] = None
        result.append(
            {
                "directory": directory,
                "manifest": manifest,
                "training_output": directory / "training_output",
                "log": directory / "training.log",
                **receipts,
            }
        )
    return result


def create_attempt(
    args: argparse.Namespace,
    body: str,
    binding_sha256: str,
    upstream_sha: str,
) -> dict[str, Any]:
    history = _attempts(args, body)
    ordinal = len(history) + 1
    if ordinal > args.max_recoverable_attempts:
        raise WcmLoboSupervisorError("WCM recoverable attempt limit reached")
    root = _attempt_root(args, body)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"attempt-{ordinal:06d}"
    output = directory / "training_output"
    log = directory / "training.log"
    command = fold_training_command(args, body, output, binding_sha256)
    manifest = signed(
        {
            "format": ATTEMPT_FORMAT,
            "supervisor_format": FORMAT,
            "held_out_body": body,
            "attempt_ordinal": ordinal,
            "attempt_directory": str(directory),
            "training_output": str(output),
            "final_fold_output": str(args.output_root / f"outer_lobo_{body}"),
            "training_log": str(log),
            "command": command,
            "command_sha256": v13_watch.canonical_sha256(command),
            "trainer_file_sha256": v13_watch.sha256_file(args.trainer),
            "primary_binding_file_sha256": binding_sha256,
            "supplement_binding_file_sha256": args.supplement_binding_sha256,
            "upstream_authority_logical_sha256": upstream_sha,
            "prior_attempt_logical_sha256s": [
                item["manifest"]["logical_sha256"] for item in history
            ],
            "attempt_selected_using_model_validation_or_heldout_outcome": False,
            "heldout_payloads_opened_by_supervisor": 0,
        }
    )
    staging = root / f".attempt-{ordinal:06d}-{os.getpid()}-{time.time_ns()}"
    staging.mkdir()
    v13_watch.create_once_or_verify(staging / "attempt.json", manifest, "WCM attempt")
    os.rename(staging, directory)
    return {
        "directory": directory,
        "manifest": manifest,
        "training_output": output,
        "log": log,
        "launch": None,
        "failure": None,
        "rebind": None,
        "promotion": None,
    }


def _process_matches(receipt: Mapping[str, Any]) -> bool:
    pid = receipt.get("training_pid")
    if type(pid) is not int or pid <= 0:
        return False
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    normalized = [value.decode("utf-8") for value in command if value]
    return v13_watch.canonical_sha256(normalized) == receipt.get(
        "observed_process_command_sha256"
    )


def record_failure(
    attempt: Mapping[str, Any], *, reason: str, returncode: int | None
) -> dict[str, Any]:
    recoverable = reason in {
        "unobserved_process_interruption",
        "trainer_terminated_by_recoverable_signal",
    }
    value = signed(
        {
            "format": FAILURE_FORMAT,
            "attempt_logical_sha256": attempt["manifest"]["logical_sha256"],
            "held_out_body": attempt["manifest"]["held_out_body"],
            "attempt_ordinal": attempt["manifest"]["attempt_ordinal"],
            "reason": reason,
            "training_returncode": returncode,
            "recoverable_noninformative_interruption": recoverable,
            "retry_selection_reads_model_validation_or_heldout_outcome": False,
        }
    )
    v13_watch.create_once_or_verify(
        Path(attempt["directory"]) / "failure.json", value, "WCM attempt failure"
    )
    return value


def rebind_and_promote(
    attempt: Mapping[str, Any],
    final: Path,
    *,
    body: str,
    binding_sha256: str,
    supplement_sha256: str,
) -> dict[str, Any]:
    training_output = Path(attempt["training_output"])
    validate_fold(
        training_output,
        body=body,
        binding_sha256=binding_sha256,
        supplement_sha256=supplement_sha256,
    )
    summary_path = training_output / "training_summary.json"
    summary = _summary(summary_path, "WCM staging summary")
    original_file_sha = v13_watch.sha256_file(summary_path)
    original_logical_sha = summary["logical_sha256"]
    rebound = dict(summary)
    rebound_members = []
    mappings = []
    for item in summary["members"]:
        checkpoint = Path(item["checkpoint"]).resolve()
        relative = checkpoint.relative_to(training_output.resolve())
        changed = dict(item)
        changed["checkpoint"] = str(final / relative)
        rebound_members.append(changed)
        mappings.append({"source": str(checkpoint), "target": str(final / relative)})
    rebound["members"] = rebound_members
    rebound.pop("logical_sha256", None)
    rebound["logical_sha256"] = wcm.canonical_sha256(rebound)
    v13_watch.atomic_json(summary_path, rebound)
    rebind = signed(
        {
            "format": REBIND_FORMAT,
            "attempt_logical_sha256": attempt["manifest"]["logical_sha256"],
            "original_summary_file_sha256": original_file_sha,
            "original_summary_logical_sha256": original_logical_sha,
            "promoted_summary_file_sha256": v13_watch.sha256_file(summary_path),
            "promoted_summary_logical_sha256": rebound["logical_sha256"],
            "checkpoint_path_mappings": mappings,
            "checkpoint_payloads_modified": False,
        }
    )
    v13_watch.create_once_or_verify(
        Path(attempt["directory"]) / "summary_rebind.json",
        rebind,
        "WCM summary rebind",
    )
    if final.exists():
        raise WcmLoboSupervisorError("WCM final fold appeared before promotion")
    os.rename(training_output, final)
    promotion = signed(
        {
            "format": PROMOTION_FORMAT,
            "attempt_logical_sha256": attempt["manifest"]["logical_sha256"],
            "held_out_body": body,
            "final_fold_output": str(final),
            "training_summary_file_sha256": v13_watch.sha256_file(
                final / "training_summary.json"
            ),
            "summary_rebind_logical_sha256": rebind["logical_sha256"],
            "training_output_moved_atomically": True,
            "retry_selection_reads_model_validation_or_heldout_outcome": False,
        }
    )
    v13_watch.create_once_or_verify(
        Path(attempt["directory"]) / "promotion.json",
        promotion,
        "WCM fold promotion",
    )
    return promotion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--supplement-binding", type=Path, required=True)
    parser.add_argument("--supplement-binding-sha256")
    parser.add_argument("--v13-state", type=Path, required=True)
    parser.add_argument("--v13-run-exit", type=Path, required=True)
    parser.add_argument("--rac-final-root", type=Path, required=True)
    parser.add_argument("--rac-state", type=Path, required=True)
    parser.add_argument("--rac-run-exit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-exit", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path(__file__).with_name(
            "train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py"
        ),
    )
    parser.add_argument(
        "--training-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--expected-gpu-uuid", default=v13_watch.EXPECTED_GPU_UUID)
    parser.add_argument("--max-recoverable-attempts", type=int, default=3)
    return parser.parse_args(argv)


def normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "binding", "supplement_binding", "v13_state", "v13_run_exit",
        "rac_final_root", "rac_state", "rac_run_exit", "output_root", "state", "run_exit",
        "trainer", "training_python",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.lock = (
        args.lock.expanduser().resolve()
        if args.lock is not None
        else args.output_root / "wcm_lobo.lock"
    )
    if args.poll_seconds <= 0 or args.max_recoverable_attempts < 1:
        raise WcmLoboSupervisorError("WCM supervisor interval/attempt limit invalid")
    return args


_FAILURE_ARGS: argparse.Namespace | None = None


def main(argv: Sequence[str] | None = None) -> int:
    global _FAILURE_ARGS, _ACTIVE_TRAINER_PATH
    args = normalized_args(parse_args(argv))
    _FAILURE_ARGS = args
    _ACTIVE_TRAINER_PATH = args.trainer
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock = args.lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise WcmLoboSupervisorError("another WCM supervisor is active") from error

    def write_state(status: str, **extra: Any) -> None:
        v13_watch.atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": v13_watch.utc_now(),
                "pid": os.getpid(),
                "output_root": str(args.output_root),
                "heldout_payloads_opened_by_supervisor": 0,
                **extra,
            },
        )

    for path in (args.binding, args.trainer, args.training_python):
        if not path.exists():
            raise FileNotFoundError(path)
    upstream = None
    while upstream is None:
        upstream = freeze_upstream_authority(args)
        if upstream is None:
            write_state(
                "waiting_for_v13_and_rac_final_authorities",
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
    try:
        v13_watch.load_primary_execution_protocol(args.binding)
    except v13_watch.LoboWatcherError as error:
        raise WcmLoboSupervisorError(str(error)) from error
    supplement = None
    while supplement is None:
        supplement = freeze_supplement_authority(args, upstream)
        if supplement is None:
            write_state(
                "waiting_for_deferred_supplement_binding",
                upstream_authority_logical_sha256=upstream["logical_sha256"],
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
    binding_sha = v13_watch.sha256_file(args.binding)
    supplement_sha = str(supplement["binding_file_sha256"])
    gpu = v13_watch.gpu_identity()
    if "4090" not in gpu["name"] or gpu["uuid"] != args.expected_gpu_uuid:
        raise WcmLoboSupervisorError(f"unexpected GPU authority: {gpu}")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(args.trainer.parent),
        }
    )
    results = []
    for fold_index, body in enumerate(BODIES):
        final = args.output_root / f"outer_lobo_{body}"
        history = _attempts(args, body)
        if final.is_dir():
            result = validate_fold(
                final,
                body=body,
                binding_sha256=binding_sha,
                supplement_sha256=supplement_sha,
            )
            promoted = [item for item in history if item["promotion"] is not None]
            if len(promoted) != 1:
                raise WcmLoboSupervisorError("completed WCM fold lacks one promotion")
            results.append(result)
            continue
        if final.exists():
            raise WcmLoboSupervisorError("incomplete WCM final fold path exists")
        if history:
            last = history[-1]
            if last["promotion"] is not None:
                raise WcmLoboSupervisorError("WCM promotion exists without final fold")
            if (Path(last["training_output"]) / "training_summary.json").is_file():
                rebind_and_promote(
                    last,
                    final,
                    body=body,
                    binding_sha256=binding_sha,
                    supplement_sha256=supplement_sha,
                )
                results.append(
                    validate_fold(
                        final,
                        body=body,
                        binding_sha256=binding_sha,
                        supplement_sha256=supplement_sha,
                    )
                )
                continue
            if last["failure"] is None:
                if last["launch"] is not None and _process_matches(last["launch"]):
                    raise WcmLoboSupervisorError(
                        "orphan WCM trainer is still active; refusing duplicate"
                    )
                record_failure(
                    last, reason="unobserved_process_interruption", returncode=None
                )
                last["failure"] = _summary(
                    Path(last["directory"]) / "failure.json", "WCM recovered failure"
                )
            if last["failure"].get("recoverable_noninformative_interruption") is not True:
                raise WcmLoboSupervisorError("prior WCM attempt failed non-recoverably")
        while v13_watch.gpu_compute_pids():
            write_state(
                "waiting_for_idle_gpu_after_upstream_authorities",
                completed_folds=[item["held_out_body"] for item in results],
                next_fold=body,
                gpu_reserved_by_supervisor=False,
                gpu=gpu,
            )
            time.sleep(args.poll_seconds)
        attempt = create_attempt(args, body, binding_sha, upstream["logical_sha256"])
        command = list(attempt["manifest"]["command"])
        write_state(
            "training_wcm_fold",
            completed_folds=[item["held_out_body"] for item in results],
            fold_index=fold_index,
            current_fold=body,
            command=command,
            gpu=gpu,
        )
        with Path(attempt["log"]).open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                cwd=args.trainer.parent,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            observed_command = [
                value.decode("utf-8")
                for value in (Path("/proc") / str(process.pid) / "cmdline")
                .read_bytes()
                .split(b"\0")
                if value
            ]
            launch = signed(
                {
                    "format": LAUNCH_FORMAT,
                    "attempt_logical_sha256": attempt["manifest"]["logical_sha256"],
                    "training_pid": process.pid,
                    "observed_process_command_sha256": v13_watch.canonical_sha256(
                        observed_command
                    ),
                    "command_sha256": attempt["manifest"]["command_sha256"],
                }
            )
            v13_watch.create_once_or_verify(
                Path(attempt["directory"]) / "launch.json", launch, "WCM launch"
            )
            returncode = process.wait()
        if returncode != 0:
            recoverable = returncode < 0 and -returncode in RECOVERABLE_SIGNALS
            record_failure(
                attempt,
                reason=(
                    "trainer_terminated_by_recoverable_signal"
                    if recoverable
                    else "trainer_process_failed"
                ),
                returncode=returncode,
            )
            raise WcmLoboSupervisorError(
                f"WCM fold {body} trainer exited {returncode}"
            )
        rebind_and_promote(
            attempt,
            final,
            body=body,
            binding_sha256=binding_sha,
            supplement_sha256=supplement_sha,
        )
        results.append(
            validate_fold(
                final,
                body=body,
                binding_sha256=binding_sha,
                supplement_sha256=supplement_sha,
            )
        )
    final = signed(
        {
            "format": FINAL_FORMAT,
            "status": "five_outer_lobo_wcm_source_only_training_complete",
            "completed_at_utc": datetime.fromtimestamp(
                max(
                    (args.output_root / f"outer_lobo_{body}" / "training_summary.json")
                    .stat()
                    .st_mtime
                    for body in BODIES
                ),
                timezone.utc,
            ).isoformat(),
            "model_family": wcm.MODEL_FAMILY,
            "not_official_wcm_architecture_or_weights": True,
            "outer_folds": results,
            "fold_count": 5,
            "members_per_fold": 5,
            "steps_per_member": trainer.DEFAULT_STEPS,
            "upstream_authority": upstream,
            "supplement_binding_authority": supplement,
            "training_binding": {
                "path": str(args.binding),
                "file_sha256": binding_sha,
            },
            "actor_execution_protocol": upstream["actor_execution_protocol"],
            "actor_execution_protocol_binding": upstream[
                "actor_execution_protocol_binding"
            ],
            "actor_authority": upstream["actor_authority"],
            "rac_final_policy_transfer_authority": upstream["rac"],
            "heldout_labels_used_for_training_normalization_or_selection": False,
            "heldout_task_success_measured": False,
            "next_required_stage": "wcm_nested_n1_n4_n8_paired_live_evaluation",
        }
    )
    final_path = args.output_root / "five_fold_wcm_training_summary.json"
    v13_watch.create_once_or_verify(final_path, final, "five-fold WCM aggregate")
    args.run_exit.parent.mkdir(parents=True, exist_ok=True)
    args.run_exit.write_text("0\n", encoding="utf-8")
    write_state(
        "complete",
        completed_folds=list(BODIES),
        final_summary=str(final_path),
        final_summary_file_sha256=v13_watch.sha256_file(final_path),
        gpu=gpu,
        gpu_reserved_by_supervisor=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if _FAILURE_ARGS is not None:
            try:
                v13_watch.atomic_json(
                    _FAILURE_ARGS.state,
                    {
                        "format": FORMAT,
                        "status": "failed",
                        "updated_at_utc": v13_watch.utc_now(),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
                _FAILURE_ARGS.run_exit.parent.mkdir(parents=True, exist_ok=True)
                _FAILURE_ARGS.run_exit.write_text("1\n", encoding="utf-8")
            except Exception:
                pass
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


__all__ = [
    "FINAL_FORMAT",
    "FORMAT",
    "WcmLoboSupervisorError",
    "fold_training_command",
    "freeze_supplement_authority",
    "freeze_upstream_authority",
    "normalized_args",
    "validate_fold",
]
