#!/usr/bin/env python3
"""Wait for authenticated five-body branches, then train five RAC LOBO folds.

This is the Relative Action Critic counterpart of the v13 shared-head watcher.
It deliberately reuses the established collection, actor, execution-protocol,
GPU and supplement authorities.  In particular, the waiting phase reads only
small authority/progress JSON files; held-out NPZ payloads are first opened by
the source-only trainer after all bindings have been authenticated.
"""

from __future__ import annotations

import argparse
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

import robotwin2_relative_action_critic_adapter_v1 as adapter
import train_robotwin2_five_body_lobo_relative_action_critic_v1 as rac
import watch_robotwin2_five_body_branches_to_lobo_training_v1 as base


FORMAT = "etsf_robotwin2_five_body_branches_to_rac_lobo_supervisor_v1"
FINAL_FORMAT = "etsf_robotwin2_five_body_rac_lobo_source_validation_aggregate_v1"
FOLD_ATTEMPT_FORMAT = "etsf_robotwin2_rac_lobo_fold_training_attempt_v1"
FOLD_ATTEMPT_FAILURE_FORMAT = "etsf_robotwin2_rac_lobo_fold_attempt_failure_v1"
FOLD_SUMMARY_REBIND_FORMAT = "etsf_robotwin2_rac_lobo_fold_summary_rebind_v1"
FOLD_ATTEMPT_PROMOTION_FORMAT = "etsf_robotwin2_rac_lobo_fold_promotion_v1"
FOLD_ATTEMPT_DIRECTORY = "rac_fold_attempts"
SUPPLEMENT_AUTHORITY_FORMAT = (
    "etsf_robotwin2_rac_deferred_supplement_binding_authority_v1"
)
UPSTREAM_AUTHORITY_FORMAT = "etsf_robotwin2_rac_upstream_completion_authority_v1"
BODIES = base.BODIES
ENSEMBLE_SEEDS = rac.DEFAULT_ENSEMBLE_SEEDS
STEPS_PER_MEMBER = rac.DEFAULT_STEPS
SPLIT_SEED = base.SUPPLEMENT_SPLIT_SEED
RECOVERABLE_TRAINING_SIGNALS = base.RECOVERABLE_TRAINING_SIGNALS


class RacLoboSupervisorError(RuntimeError):
    pass


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    return base.signed(value)


def _verify_signed(value: Mapping[str, Any], label: str) -> None:
    try:
        base.verify_logical_sha(value, label)
    except base.LoboWatcherError as error:
        raise RacLoboSupervisorError(str(error)) from error


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        return base.read_json(path, label)
    except (base.LoboWatcherError, OSError) as error:
        raise RacLoboSupervisorError(str(error)) from error


def fold_training_command(
    args: argparse.Namespace,
    held_out_body: str,
    binding_sha256: str,
    output: Path,
) -> list[str]:
    if held_out_body not in BODIES:
        raise RacLoboSupervisorError("unknown RAC LOBO body")
    command = [
        str(args.training_python),
        str(args.trainer),
        "--mode", "train-fold",
        "--binding", str(args.binding),
        "--binding-sha256", binding_sha256,
        "--held-out-body", held_out_body,
        "--split-seed", str(SPLIT_SEED),
        "--output", str(output),
        "--device", "cuda",
        "--steps", str(STEPS_PER_MEMBER),
        "--eval-every", "100",
        "--batch-size-pairs", "96",
        "--learning-rate", "0.0003",
        "--focal-gamma", "2.0",
        "--model-dim", "128",
        "--transformer-layers", "3",
        "--attention-heads", "4",
        "--dropout", "0.1",
        "--ensemble-seeds", *[str(seed) for seed in ENSEMBLE_SEEDS],
    ]
    if args.supplement_binding is not None:
        if args.supplement_binding_sha256 is None:
            raise RacLoboSupervisorError(
                "RAC training command requires a frozen supplement SHA"
            )
        command.extend(
            [
                "--supplement-binding", str(args.supplement_binding),
                "--supplement-binding-sha256",
                str(args.supplement_binding_sha256),
            ]
        )
    return command


def _supplement_authority_path(args: argparse.Namespace) -> Path:
    return args.output_root / "supplement_binding_authority.json"


def freeze_or_validate_supplement_authority(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Freeze a deferred supplement SHA only after full protocol validation.

    Returning ``None`` means the requested file has not appeared yet.  No body
    manifest or NPZ payload is opened by this function.
    """

    if args.supplement_binding is None:
        return {
            "format": SUPPLEMENT_AUTHORITY_FORMAT,
            "enabled": False,
            "binding_path": None,
            "binding_file_sha256": None,
        }
    authority_path = _supplement_authority_path(args)
    frozen = None
    if authority_path.exists():
        frozen = _read(authority_path, "RAC supplement authority")
        _verify_signed(frozen, "RAC supplement authority")
        if (
            frozen.get("format") != SUPPLEMENT_AUTHORITY_FORMAT
            or frozen.get("enabled") is not True
            or frozen.get("binding_path") != str(args.supplement_binding)
            or frozen.get("actor_execution_protocol_logical_sha256")
            != base._ACTIVE_EXECUTION_PROTOCOL.get("logical_sha256")
            or frozen.get("actor_execution_protocol_file_sha256")
            != base._ACTIVE_EXECUTION_PROTOCOL_BINDING.get("file_sha256")
        ):
            raise RacLoboSupervisorError("frozen supplement authority changed")
    if not args.supplement_binding.is_file():
        if frozen is not None:
            raise RacLoboSupervisorError("frozen supplement binding disappeared")
        return None
    before_sha = base.sha256_file(args.supplement_binding)
    if args.supplement_binding_sha256 is not None and (
        before_sha != args.supplement_binding_sha256
    ):
        raise RacLoboSupervisorError("supplement binding explicit SHA mismatch")
    value = base.validate_supplement_frame_binding(
        args.supplement_binding,
        execution_protocol=base._ACTIVE_EXECUTION_PROTOCOL,
        execution_protocol_binding=base._ACTIVE_EXECUTION_PROTOCOL_BINDING,
    )
    after_sha = base.sha256_file(args.supplement_binding)
    if before_sha != after_sha:
        raise RacLoboSupervisorError("supplement binding changed during authentication")
    observed = signed(
        {
            "format": SUPPLEMENT_AUTHORITY_FORMAT,
            "enabled": True,
            "frozen_at_utc": base.utc_now(),
            "binding_path": str(args.supplement_binding),
            "binding_file_sha256": after_sha,
            "binding_logical_sha256": value["logical_sha256"],
            "actor_execution_protocol_logical_sha256": (
                base._ACTIVE_EXECUTION_PROTOCOL["logical_sha256"]
            ),
            "actor_execution_protocol_file_sha256": (
                base._ACTIVE_EXECUTION_PROTOCOL_BINDING["file_sha256"]
            ),
            "binding_validated_before_file_sha_freeze": True,
            "heldout_manifest_or_payload_opened": 0,
        }
    )
    if frozen is not None:
        stable_fields = {
            key: frozen.get(key)
            for key in observed
            if key not in {"frozen_at_utc", "logical_sha256"}
        }
        expected_fields = {
            key: observed.get(key)
            for key in observed
            if key not in {"frozen_at_utc", "logical_sha256"}
        }
        if stable_fields != expected_fields:
            raise RacLoboSupervisorError("supplement binding differs from frozen authority")
        args.supplement_binding_sha256 = str(frozen["binding_file_sha256"])
        return frozen
    base.create_once_or_verify(
        authority_path, observed, "RAC supplement binding authority"
    )
    args.supplement_binding_sha256 = after_sha
    return observed


def _upstream_authority_path(args: argparse.Namespace) -> Path:
    return args.output_root / "upstream_completion_authority.json"


def _verify_named_sha(
    value: Mapping[str, Any], field: str, label: str
) -> None:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != base.canonical_sha256(unsigned):
        raise RacLoboSupervisorError(f"{label} {field} changed")


def _current_upstream_completion(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.upstream_run_exit is None:
        return {
            "format": UPSTREAM_AUTHORITY_FORMAT,
            "enabled": False,
        }
    if not args.upstream_run_exit.is_file() or not args.upstream_state.is_file():
        return None
    run_exit_sha_before = base.sha256_file(args.upstream_run_exit)
    run_exit_text = args.upstream_run_exit.read_text(encoding="utf-8")
    run_exit_sha_after = base.sha256_file(args.upstream_run_exit)
    if run_exit_sha_before != run_exit_sha_after:
        raise RacLoboSupervisorError("upstream run-exit changed during authentication")
    if run_exit_text.strip() != "0":
        if run_exit_text.strip() == "1":
            raise RacLoboSupervisorError("upstream v13 chain failed")
        return None
    state_sha_before = base.sha256_file(args.upstream_state)
    state = _read(args.upstream_state, "upstream v13 state")
    state_sha_after = base.sha256_file(args.upstream_state)
    if state_sha_before != state_sha_after:
        raise RacLoboSupervisorError("upstream state changed during authentication")
    if state.get("status") == "failed":
        raise RacLoboSupervisorError("upstream v13 state reports failure")
    if state.get("status") != "complete":
        return None
    if state.get("final_crossbody_report") is not None:
        report_key = "final_crossbody_report"
        file_sha_key = "final_crossbody_report_file_sha256"
        declared_named_sha = state.get("final_crossbody_report_sha256")
    else:
        report_key = "nested_actor_n4_n8_report"
        file_sha_key = "nested_actor_n4_n8_report_file_sha256"
        declared_named_sha = None
    report_raw = state.get(report_key)
    if not isinstance(report_raw, str) or not report_raw:
        raise RacLoboSupervisorError("complete upstream state lacks final report")
    report_path = Path(report_raw).expanduser().resolve()
    report_sha_before = base.sha256_file(report_path)
    report = _read(report_path, "upstream final report")
    report_file_sha = base.sha256_file(report_path)
    if report_sha_before != report_file_sha:
        raise RacLoboSupervisorError("upstream final report changed during authentication")
    if state.get(file_sha_key) != report_file_sha:
        raise RacLoboSupervisorError("upstream final report file SHA changed")
    named_field = "report_sha256" if "report_sha256" in report else "logical_sha256"
    _verify_named_sha(report, named_field, "upstream final report")
    if declared_named_sha is not None and declared_named_sha != report[named_field]:
        raise RacLoboSupervisorError("upstream state report logical SHA changed")
    return {
        "format": UPSTREAM_AUTHORITY_FORMAT,
        "enabled": True,
        "upstream_run_exit": str(args.upstream_run_exit),
        "upstream_run_exit_file_sha256": run_exit_sha_after,
        "upstream_state": str(args.upstream_state),
        "upstream_state_format": state.get("format"),
        "upstream_state_file_sha256": state_sha_after,
        "upstream_state_status": "complete",
        "final_report": str(report_path),
        "final_report_file_sha256": report_file_sha,
        "final_report_logical_sha_field": named_field,
        "final_report_logical_sha256": report[named_field],
        "upstream_completed_before_rac_training": True,
        "heldout_payloads_opened_while_waiting": 0,
    }


def freeze_or_validate_upstream_authority(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Wait for and create-once bind a successful complete upstream chain."""

    current = _current_upstream_completion(args)
    authority_path = _upstream_authority_path(args)
    if authority_path.exists():
        frozen = _read(authority_path, "RAC upstream completion authority")
        _verify_signed(frozen, "RAC upstream completion authority")
        if current is None:
            raise RacLoboSupervisorError("frozen upstream completion disappeared")
        comparable_frozen = {
            key: value for key, value in frozen.items()
            if key not in {"frozen_at_utc", "logical_sha256"}
        }
        if comparable_frozen != current:
            raise RacLoboSupervisorError("upstream completion differs from authority")
        return frozen
    if current is None:
        return None
    if current.get("enabled") is False:
        return current
    authority = signed({**current, "frozen_at_utc": base.utc_now()})
    base.create_once_or_verify(
        authority_path, authority, "RAC upstream completion authority"
    )
    return authority


def _attempt_root(output_root: Path, held_out_body: str) -> Path:
    if held_out_body not in BODIES:
        raise RacLoboSupervisorError("unknown RAC LOBO attempt body")
    return output_root / FOLD_ATTEMPT_DIRECTORY / held_out_body


def _receipt(
    path: Path,
    label: str,
    expected_format: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read(path, label)
    _verify_signed(value, label)
    if value.get("format") != expected_format:
        raise RacLoboSupervisorError(f"{label} format changed")
    return value


def validate_fold_attempt_history(
    args: argparse.Namespace,
    held_out_body: str,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    root = _attempt_root(args.output_root, held_out_body)
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise RacLoboSupervisorError("RAC fold attempt root is not a real directory")
    visible: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(".precommit-attempt-"):
            if not entry.is_dir() or entry.is_symlink():
                raise RacLoboSupervisorError("RAC precommit attempt is invalid")
            allowed = {
                item.name for item in entry.iterdir()
                if item.name == "attempt.json"
                or item.name.startswith("attempt.json.create-")
            }
            if len(allowed) != len(list(entry.iterdir())):
                raise RacLoboSupervisorError("RAC precommit attempt has artifacts")
            continue
        visible.append(entry)
    if any(
        not item.is_dir()
        or item.is_symlink()
        or not item.name.startswith("attempt-")
        or not item.name.removeprefix("attempt-").isdigit()
        for item in visible
    ):
        raise RacLoboSupervisorError("RAC attempt root contains unknown entries")

    trainer_sha = base.sha256_file(args.trainer)
    final_output = args.output_root / f"outer_lobo_{held_out_body}"
    prior: list[str] = []
    records: list[dict[str, Any]] = []
    promoted = 0
    for ordinal, directory in enumerate(visible, start=1):
        if directory.name != f"attempt-{ordinal:06d}":
            raise RacLoboSupervisorError("RAC attempt ordinals are not contiguous")
        manifest = _read(directory / "attempt.json", "RAC attempt manifest")
        _verify_signed(manifest, "RAC attempt manifest")
        training_output = directory / "training_output"
        log = directory / "training.log"
        command = fold_training_command(
            args, held_out_body, binding_sha256, training_output
        )
        if (
            manifest.get("format") != FOLD_ATTEMPT_FORMAT
            or manifest.get("supervisor_format") != FORMAT
            or manifest.get("held_out_body") != held_out_body
            or manifest.get("attempt_ordinal") != ordinal
            or manifest.get("attempt_directory") != str(directory)
            or manifest.get("training_output") != str(training_output)
            or manifest.get("final_fold_output") != str(final_output)
            or manifest.get("training_log") != str(log)
            or manifest.get("binding_file_sha256") != binding_sha256
            or manifest.get("supplement_binding_file_sha256")
            != args.supplement_binding_sha256
            or manifest.get("trainer") != str(args.trainer)
            or manifest.get("trainer_file_sha256") != trainer_sha
            or manifest.get("command") != command
            or manifest.get("command_sha256") != base.canonical_sha256(command)
            or manifest.get("prior_attempt_logical_sha256s") != prior
            or manifest.get("attempt_selected_using_training_or_validation_outcome")
            is not False
            or manifest.get("heldout_payloads_opened_by_supervisor") != 0
        ):
            raise RacLoboSupervisorError("RAC attempt manifest changed")
        failure = _receipt(
            directory / "failure.json",
            "RAC attempt failure",
            FOLD_ATTEMPT_FAILURE_FORMAT,
        )
        rebinding = _receipt(
            directory / "summary_rebinding.json",
            "RAC summary rebinding",
            FOLD_SUMMARY_REBIND_FORMAT,
        )
        promotion = _receipt(
            directory / "promotion.json",
            "RAC attempt promotion",
            FOLD_ATTEMPT_PROMOTION_FORMAT,
        )
        for value, label in (
            (failure, "failure"), (rebinding, "rebinding"), (promotion, "promotion")
        ):
            if value is not None and (
                value.get("held_out_body") != held_out_body
                or value.get("attempt_ordinal") != ordinal
                or value.get("attempt_logical_sha256")
                != manifest["logical_sha256"]
            ):
                raise RacLoboSupervisorError(f"RAC {label} receipt changed")
        if failure is not None:
            returncode = failure.get("training_returncode")
            if (
                isinstance(returncode, bool)
                or not isinstance(returncode, int)
                or failure.get("retry_selection_reads_model_outcome") is not False
                or rebinding is not None
                or promotion is not None
            ):
                raise RacLoboSupervisorError("RAC failure receipt is invalid")
        if rebinding is not None and (
            rebinding.get("checkpoint_path_rebind_count") != rac.ENSEMBLE_SIZE
            or rebinding.get("checkpoint_payloads_modified") is not False
            or not isinstance(
                rebinding.get("promoted_training_summary_logical_sha256"), str
            )
        ):
            raise RacLoboSupervisorError("RAC rebinding receipt is invalid")
        if promotion is not None:
            if (
                rebinding is None
                or promotion.get("summary_rebinding_logical_sha256")
                != rebinding["logical_sha256"]
                or promotion.get("final_fold_output") != str(final_output)
                or promotion.get("training_output_moved_atomically") is not True
                or training_output.exists()
                or not final_output.is_dir()
                or base.sha256_file(final_output / "training_summary.json")
                != promotion.get("training_summary_file_sha256")
            ):
                raise RacLoboSupervisorError("RAC promotion receipt is invalid")
            promoted += 1
            if ordinal != len(visible):
                raise RacLoboSupervisorError("promoted RAC attempt is not final")
        records.append(
            {
                "directory": directory,
                "manifest": manifest,
                "training_output": training_output,
                "log": log,
                "failure": failure,
                "summary_rebinding": rebinding,
                "promotion": promotion,
            }
        )
        prior.append(str(manifest["logical_sha256"]))
    if promoted > 1:
        raise RacLoboSupervisorError("more than one RAC attempt was promoted")
    return records


def create_fold_attempt(
    args: argparse.Namespace,
    held_out_body: str,
    binding_sha256: str,
) -> dict[str, Any]:
    history = validate_fold_attempt_history(args, held_out_body, binding_sha256)
    final_output = args.output_root / f"outer_lobo_{held_out_body}"
    if final_output.exists():
        raise RacLoboSupervisorError("cannot retry an already promoted RAC fold")
    ordinal = len(history) + 1
    root = _attempt_root(args.output_root, held_out_body)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"attempt-{ordinal:06d}"
    training_output = directory / "training_output"
    log = directory / "training.log"
    command = fold_training_command(
        args, held_out_body, binding_sha256, training_output
    )
    manifest = signed(
        {
            "format": FOLD_ATTEMPT_FORMAT,
            "supervisor_format": FORMAT,
            "held_out_body": held_out_body,
            "attempt_ordinal": ordinal,
            "created_at_utc": base.utc_now(),
            "supervisor_pid": os.getpid(),
            "attempt_directory": str(directory),
            "training_output": str(training_output),
            "final_fold_output": str(final_output),
            "training_log": str(log),
            "binding_file_sha256": binding_sha256,
            "supplement_binding_file_sha256": args.supplement_binding_sha256,
            "trainer": str(args.trainer),
            "trainer_file_sha256": base.sha256_file(args.trainer),
            "command": command,
            "command_sha256": base.canonical_sha256(command),
            "prior_attempt_logical_sha256s": [
                item["manifest"]["logical_sha256"] for item in history
            ],
            "attempt_selected_using_training_or_validation_outcome": False,
            "heldout_payloads_opened_by_supervisor": 0,
        }
    )
    staging = root / (
        f".precommit-attempt-{ordinal:06d}-{os.getpid()}-{time.time_ns()}"
    )
    staging.mkdir()
    base.create_once_or_verify(
        staging / "attempt.json", manifest, "staged RAC fold attempt"
    )
    try:
        os.rename(staging, directory)
    except FileExistsError as error:
        raise RacLoboSupervisorError("RAC fold attempt appeared concurrently") from error
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "directory": directory,
        "manifest": manifest,
        "training_output": training_output,
        "log": log,
        "failure": None,
        "summary_rebinding": None,
        "promotion": None,
    }


def record_fold_attempt_failure(
    attempt: Mapping[str, Any],
    *,
    returncode: int,
    reason: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    manifest = attempt["manifest"]
    value = signed(
        {
            "format": FOLD_ATTEMPT_FAILURE_FORMAT,
            "held_out_body": manifest["held_out_body"],
            "attempt_ordinal": manifest["attempt_ordinal"],
            "attempt_logical_sha256": manifest["logical_sha256"],
            "recorded_at_utc": base.utc_now(),
            "training_returncode": int(returncode),
            "reason": reason,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
            "retry_selection_reads_model_outcome": False,
        }
    )
    base.create_once_or_verify(
        Path(attempt["directory"]) / "failure.json",
        value,
        "RAC fold attempt failure",
    )
    return value


def rebind_fold_summary_for_atomic_promotion(
    attempt: Mapping[str, Any], final_output: Path
) -> dict[str, Any]:
    training_output = Path(attempt["training_output"])
    summary_path = training_output / "training_summary.json"
    summary = _read(summary_path, "RAC attempt training summary")
    _verify_signed(summary, "RAC attempt training summary")
    members = summary.get("members")
    if not isinstance(members, list) or len(members) != rac.ENSEMBLE_SIZE:
        raise RacLoboSupervisorError("RAC attempt lacks five members")
    original_file_sha = base.sha256_file(summary_path)
    original_logical_sha = summary["logical_sha256"]
    rebound = dict(summary)
    rebound_members: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, Mapping):
            raise RacLoboSupervisorError("RAC member binding is invalid")
        declared = Path(str(member.get("checkpoint", ""))).expanduser()
        if declared.is_symlink():
            raise RacLoboSupervisorError("RAC checkpoint may not be symbolic")
        source = declared.resolve()
        try:
            relative = source.relative_to(training_output.resolve())
        except ValueError as error:
            raise RacLoboSupervisorError("RAC checkpoint escapes attempt output") from error
        if (
            not relative.parts
            or not source.is_file()
            or source.is_symlink()
            or base.sha256_file(source) != member.get("checkpoint_sha256")
        ):
            raise RacLoboSupervisorError("RAC checkpoint is missing or changed")
        target = final_output / relative
        rebound_member = dict(member)
        rebound_member["checkpoint"] = str(target)
        rebound_members.append(rebound_member)
        mappings.append(
            {
                "member": member.get("member"),
                "source": str(source),
                "target": str(target),
                "checkpoint_sha256": member.get("checkpoint_sha256"),
            }
        )
    rebound["members"] = rebound_members
    rebound.pop("logical_sha256", None)
    rebound["logical_sha256"] = rac.canonical_sha256(rebound)
    base.atomic_json(summary_path, rebound)
    promoted_file_sha = base.sha256_file(summary_path)
    receipt = signed(
        {
            "format": FOLD_SUMMARY_REBIND_FORMAT,
            "held_out_body": attempt["manifest"]["held_out_body"],
            "attempt_ordinal": attempt["manifest"]["attempt_ordinal"],
            "attempt_logical_sha256": attempt["manifest"]["logical_sha256"],
            "recorded_at_utc": base.utc_now(),
            "original_training_summary_file_sha256": original_file_sha,
            "original_training_summary_logical_sha256": original_logical_sha,
            "promoted_training_summary_file_sha256": promoted_file_sha,
            "promoted_training_summary_logical_sha256": rebound["logical_sha256"],
            "checkpoint_path_rebind_count": len(mappings),
            "checkpoint_path_mappings": mappings,
            "checkpoint_payloads_modified": False,
        }
    )
    base.create_once_or_verify(
        Path(attempt["directory"]) / "summary_rebinding.json",
        receipt,
        "RAC summary rebinding",
    )
    return receipt


def promote_fold_attempt(
    attempt: Mapping[str, Any], final_output: Path, *, recovered: bool = False
) -> dict[str, Any]:
    training_output = Path(attempt["training_output"])
    if final_output.exists():
        raise RacLoboSupervisorError("RAC final fold already exists")
    rebinding = _read(
        Path(attempt["directory"]) / "summary_rebinding.json",
        "RAC summary rebinding",
    )
    _verify_signed(rebinding, "RAC summary rebinding")
    summary_path = training_output / "training_summary.json"
    if (
        not training_output.is_dir()
        or not summary_path.is_file()
        or rebinding.get("promoted_training_summary_file_sha256")
        != base.sha256_file(summary_path)
    ):
        raise RacLoboSupervisorError("RAC output was not rebound for promotion")
    os.rename(training_output, final_output)
    fd = os.open(final_output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    value = signed(
        {
            "format": FOLD_ATTEMPT_PROMOTION_FORMAT,
            "held_out_body": attempt["manifest"]["held_out_body"],
            "attempt_ordinal": attempt["manifest"]["attempt_ordinal"],
            "attempt_logical_sha256": attempt["manifest"]["logical_sha256"],
            "promoted_at_utc": base.utc_now(),
            "final_fold_output": str(final_output),
            "training_summary_file_sha256": base.sha256_file(
                final_output / "training_summary.json"
            ),
            "summary_rebinding_logical_sha256": rebinding["logical_sha256"],
            "training_output_moved_atomically": True,
            "promotion_receipt_recovered_after_interruption": bool(recovered),
        }
    )
    base.create_once_or_verify(
        Path(attempt["directory"]) / "promotion.json",
        value,
        "RAC fold promotion",
    )
    return value


def recover_missing_promotion_receipt(
    history: Sequence[Mapping[str, Any]], final_output: Path
) -> dict[str, Any]:
    promoted = [item for item in history if item.get("promotion") is not None]
    if len(promoted) == 1:
        return dict(promoted[0]["promotion"])
    if promoted or not history:
        raise RacLoboSupervisorError("RAC promotion history is ambiguous")
    candidate = history[-1]
    rebinding_path = Path(candidate["directory"]) / "summary_rebinding.json"
    if (
        candidate.get("failure") is not None
        or Path(candidate["training_output"]).exists()
        or not Path(candidate["log"]).is_file()
        or not final_output.is_dir()
        or not rebinding_path.is_file()
    ):
        raise RacLoboSupervisorError("RAC final fold cannot be recovered")
    rebinding = _read(rebinding_path, "RAC summary rebinding")
    _verify_signed(rebinding, "RAC summary rebinding")
    summary_sha = base.sha256_file(final_output / "training_summary.json")
    if rebinding.get("promoted_training_summary_file_sha256") != summary_sha:
        raise RacLoboSupervisorError("RAC recovered summary differs from receipt")
    value = signed(
        {
            "format": FOLD_ATTEMPT_PROMOTION_FORMAT,
            "held_out_body": candidate["manifest"]["held_out_body"],
            "attempt_ordinal": candidate["manifest"]["attempt_ordinal"],
            "attempt_logical_sha256": candidate["manifest"]["logical_sha256"],
            "promoted_at_utc": base.utc_now(),
            "final_fold_output": str(final_output),
            "training_summary_file_sha256": summary_sha,
            "summary_rebinding_logical_sha256": rebinding["logical_sha256"],
            "training_output_moved_atomically": True,
            "promotion_receipt_recovered_after_interruption": True,
        }
    )
    base.create_once_or_verify(
        Path(candidate["directory"]) / "promotion.json",
        value,
        "recovered RAC fold promotion",
    )
    return value


def fold_attempt_audit(
    attempt: Mapping[str, Any], promotion: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = attempt["manifest"]
    if promotion.get("attempt_logical_sha256") != manifest["logical_sha256"]:
        raise RacLoboSupervisorError("RAC promotion does not bind its attempt")
    return {
        "attempt_directory": str(attempt["directory"]),
        "attempt_ordinal": manifest["attempt_ordinal"],
        "attempt_manifest_logical_sha256": manifest["logical_sha256"],
        "trainer_file_sha256": manifest["trainer_file_sha256"],
        "promotion_logical_sha256": promotion["logical_sha256"],
        "promotion_receipt_recovered_after_interruption": promotion[
            "promotion_receipt_recovered_after_interruption"
        ],
        "attempt_selected_using_training_or_validation_outcome": False,
        "prior_attempt_count": len(manifest["prior_attempt_logical_sha256s"]),
        "prior_attempts_retained": True,
        "heldout_payloads_opened_by_supervisor": 0,
    }


def summarize_fold(
    path: Path,
    held_out_body: str,
    expected_binding_sha256: str,
    expected_supplement_binding_sha256: str | None,
) -> dict[str, Any]:
    preflight = _read(path / "preflight_receipt.json", "RAC preflight receipt")
    summary_path = path / "training_summary.json"
    summary = _read(summary_path, "RAC training summary")
    _verify_signed(preflight, "RAC preflight receipt")
    _verify_signed(summary, "RAC training summary")
    protocol = base._ACTIVE_EXECUTION_PROTOCOL
    protocol_binding = base._ACTIVE_EXECUTION_PROTOCOL_BINDING
    supplement = preflight.get("supplement")
    expected_source = [body for body in BODIES if body != held_out_body]
    selection = summary.get("checkpoint_selection")
    budget = summary.get("training_budget")
    members = summary.get("members")
    if (
        preflight.get("format") != rac.FORMAT
        or preflight.get("status")
        != "rac_preflight_passed_payloads_still_unopened"
        or preflight.get("held_out_body") != held_out_body
        or preflight.get("source_bodies") != expected_source
        or preflight.get("primary_binding_file_sha256")
        != expected_binding_sha256
        or preflight.get("actor_execution_protocol") != protocol
        or preflight.get("actor_execution_protocol_binding") != protocol_binding
        or preflight.get("heldout_group_npz_opened") != 0
        or preflight.get("heldout_group_payload_bytes_read") != 0
        or preflight.get("heldout_labels_used_for_training_or_selection")
        is not False
        or not isinstance(supplement, Mapping)
        or supplement.get("enabled")
        != (expected_supplement_binding_sha256 is not None)
        or supplement.get("binding_file_sha256")
        != expected_supplement_binding_sha256
        or summary.get("format") != rac.SUMMARY_FORMAT
        or summary.get("status")
        != "source_only_rac_checkpoint_selection_complete"
        or summary.get("model_family") != rac.MODEL_FAMILY
        or summary.get("rac_contract") != rac.rac_contract()
        or summary.get("held_out_body") != held_out_body
        or summary.get("source_bodies") != expected_source
        or summary.get("actor_execution_protocol") != protocol
        or summary.get("actor_execution_protocol_binding") != protocol_binding
        or summary.get("actor_execution_protocol_file_sha256")
        != protocol_binding.get("file_sha256")
        or not isinstance(budget, Mapping)
        or budget.get("steps_per_member") != STEPS_PER_MEMBER
        or budget.get("eval_every_steps") != 100
        or budget.get("ensemble_members") != rac.ENSEMBLE_SIZE
        or budget.get("batch_size_pairs") != 96
        or budget.get("learning_rate") != 0.0003
        or budget.get("focal_gamma") != 2.0
        or not isinstance(selection, Mapping)
        or selection.get("scope") != "primary_source_validation_only"
        or selection.get("supplement_rows_used") != 0
        or selection.get("heldout_rows_used") != 0
        or not isinstance(members, list)
        or len(members) != rac.ENSEMBLE_SIZE
        or [item.get("seed") for item in members if isinstance(item, Mapping)]
        != list(ENSEMBLE_SEEDS)
        or any(
            not isinstance(item, Mapping)
            or item.get("best_step") != selection.get("selected_step")
            for item in members
        )
        or summary.get("heldout_rows_used_for_training_normalization_or_selection")
        != 0
        or summary.get("all_checkpoints_selected_before_any_heldout_payload_open")
        is not True
    ):
        raise RacLoboSupervisorError(
            f"{held_out_body} RAC summary violates source-only LOBO"
        )
    try:
        models, load_receipt = adapter.load_fold_ensemble(
            summary_path,
            device=torch.device("cpu"),
            expected_held_out_body=held_out_body,
        )
    except adapter.RelativeActionCriticAdapterError as error:
        raise RacLoboSupervisorError(str(error)) from error
    del models
    metrics = selection.get("selected_metrics")
    if not isinstance(metrics, Mapping) or metrics.get("heldout_rows_used") != 0:
        raise RacLoboSupervisorError("RAC selected metrics are invalid")
    return {
        "held_out_body": held_out_body,
        "model_family": rac.MODEL_FAMILY,
        "source_bodies": expected_source,
        "member_count": rac.ENSEMBLE_SIZE,
        "steps_per_member": STEPS_PER_MEMBER,
        "selected_step": selection.get("selected_step"),
        "source_validation": dict(metrics),
        "training_summary": str(summary_path),
        "training_summary_file_sha256": base.sha256_file(summary_path),
        "training_summary_logical_sha256": summary["logical_sha256"],
        "ensemble_load_receipt": load_receipt,
        "supplement_binding_file_sha256": expected_supplement_binding_sha256,
        "heldout_labels_used_for_training_normalization_or_selection": False,
        "heldout_task_success_measured": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--supplement-binding", type=Path)
    parser.add_argument("--supplement-binding-sha256")
    parser.add_argument("--upstream-run-exit", type=Path)
    parser.add_argument("--upstream-state", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-exit", type=Path, required=True)
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path(__file__).with_name(
            "train_robotwin2_five_body_lobo_relative_action_critic_v1.py"
        ),
    )
    parser.add_argument(
        "--training-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--expected-gpu-uuid", default=base.EXPECTED_GPU_UUID)
    return parser.parse_args()


def normalized_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "branches_root", "actor_checkpoint", "materialization_receipt",
        "actor_authority", "binding", "output_root", "state", "run_exit",
        "trainer", "training_python",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.supplement_binding is None and args.supplement_binding_sha256 is not None:
        raise RacLoboSupervisorError(
            "supplement SHA cannot be supplied without its binding path"
        )
    if args.supplement_binding is not None:
        args.supplement_binding = args.supplement_binding.expanduser().resolve()
    if (args.upstream_run_exit is None) != (args.upstream_state is None):
        raise RacLoboSupervisorError(
            "upstream run-exit and state must be supplied together"
        )
    if args.upstream_run_exit is not None:
        args.upstream_run_exit = args.upstream_run_exit.expanduser().resolve()
        args.upstream_state = args.upstream_state.expanduser().resolve()
    if args.poll_seconds <= 0:
        raise RacLoboSupervisorError("poll interval must be positive")
    return args


_FAILURE_STATE_PATH: Path | None = None
_FAILURE_RUN_EXIT_PATH: Path | None = None


def main() -> int:
    global _FAILURE_STATE_PATH, _FAILURE_RUN_EXIT_PATH
    args = normalized_args(parse_args())
    _FAILURE_STATE_PATH = args.state
    _FAILURE_RUN_EXIT_PATH = args.run_exit
    protocol_loaded = False
    supplement_authority: dict[str, Any] | None = None
    upstream_authority: dict[str, Any] | None = None
    args.output_root.mkdir(parents=True, exist_ok=True)

    def write_state(status: str, **extra: Any) -> None:
        base.atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": base.utc_now(),
                "pid": os.getpid(),
                "binding": str(args.binding),
                "supplement_binding": (
                    str(args.supplement_binding)
                    if args.supplement_binding is not None else None
                ),
                "supplement_binding_sha256": args.supplement_binding_sha256,
                "supplement_binding_authority": supplement_authority,
                "upstream_run_exit": (
                    str(args.upstream_run_exit)
                    if args.upstream_run_exit is not None else None
                ),
                "upstream_state": (
                    str(args.upstream_state)
                    if args.upstream_state is not None else None
                ),
                "upstream_completion_authority": upstream_authority,
                "output_root": str(args.output_root),
                "expected_decisions": base.TOTAL_DECISIONS,
                "expected_branches": base.TOTAL_BRANCHES,
                "actor_execution_protocol": (
                    base._ACTIVE_EXECUTION_PROTOCOL if protocol_loaded else None
                ),
                "actor_execution_protocol_binding": (
                    base._ACTIVE_EXECUTION_PROTOCOL_BINDING
                    if protocol_loaded else None
                ),
                "heldout_payloads_opened_by_supervisor": 0,
                **extra,
            },
        )

    for required in (args.trainer, args.training_python):
        if not required.exists():
            raise FileNotFoundError(required)
    gpu = base.gpu_identity()
    if "4090" not in gpu["name"] or gpu["uuid"] != args.expected_gpu_uuid:
        raise RacLoboSupervisorError(f"unexpected GPU authority: {gpu}")

    collection: dict[str, Any] | None = None
    while collection is None:
        if not args.binding.is_file():
            write_state(
                "waiting_for_primary_protocol_binding",
                primary_binding_present=False,
                gpu=gpu,
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
            continue
        if not protocol_loaded:
            try:
                base.load_primary_execution_protocol(args.binding)
            except base.LoboWatcherError as error:
                raise RacLoboSupervisorError(str(error)) from error
            protocol_loaded = True
        supplement_authority = freeze_or_validate_supplement_authority(args)
        if supplement_authority is None:
            write_state(
                "waiting_for_supplement_protocol_binding",
                primary_binding_present=True,
                supplement_binding_present=False,
                gpu=gpu,
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
            continue
        upstream_authority = freeze_or_validate_upstream_authority(args)
        if upstream_authority is None:
            write_state(
                "waiting_for_complete_upstream_v13_chain",
                primary_binding_present=True,
                supplement_binding_present=True,
                gpu=gpu,
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
            continue
        progress = base.collection_progress(args.branches_root)
        base.reject_irrecoverable_progress(progress)
        upstream_ready = (
            args.actor_checkpoint.is_dir()
            and args.actor_authority.is_file()
            and args.materialization_receipt.is_file()
        )
        if not base.progress_is_complete(progress) or not upstream_ready:
            write_state(
                "waiting_for_complete_public_branches",
                collection_progress=progress,
                upstream_authorities_present=upstream_ready,
                gpu=gpu,
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
            continue
        write_state(
            "validating_complete_public_branches",
            collection_progress=progress,
            gpu=gpu,
            outcome_or_event_arrays_interpreted_by_supervisor=False,
        )
        collection = base.validate_complete_collection(
            args.branches_root,
            args.actor_checkpoint,
            execution_protocol=base._ACTIVE_EXECUTION_PROTOCOL,
        )

    checkpoint_sha, checkpoint_files, checkpoint_bytes = base.sha256_tree(
        args.actor_checkpoint
    )
    actor_authority, binding = base.validate_existing_authorities(
        args, collection, checkpoint_sha, checkpoint_files, checkpoint_bytes
    )
    binding_sha = base.sha256_file(args.binding)
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
    (args.output_root / "logs").mkdir(exist_ok=True)
    fold_results: list[dict[str, Any]] = []
    for fold_index, held_out_body in enumerate(BODIES):
        fold_output = args.output_root / f"outer_lobo_{held_out_body}"
        history = validate_fold_attempt_history(args, held_out_body, binding_sha)
        if (fold_output / "training_summary.json").is_file():
            result = summarize_fold(
                fold_output,
                held_out_body,
                binding_sha,
                args.supplement_binding_sha256,
            )
            promotion = recover_missing_promotion_receipt(history, fold_output)
            attempt = next(
                item for item in history
                if item["manifest"]["logical_sha256"]
                == promotion["attempt_logical_sha256"]
            )
            result["training_attempt"] = fold_attempt_audit(attempt, promotion)
            fold_results.append(result)
            write_state(
                "fold_already_complete",
                completed_folds=[item["held_out_body"] for item in fold_results],
                current_fold=held_out_body,
                gpu=gpu,
            )
            continue
        if fold_output.exists():
            raise RacLoboSupervisorError("incomplete RAC final fold path exists")
        while True:
            compute_pids = base.gpu_compute_pids()
            if not compute_pids:
                break
            write_state(
                "waiting_for_gpu_after_complete_collection",
                completed_folds=[item["held_out_body"] for item in fold_results],
                next_fold=held_out_body,
                external_gpu_compute_pids=compute_pids,
                gpu=gpu,
                gpu_reserved_by_supervisor=False,
            )
            time.sleep(args.poll_seconds)
        attempt = create_fold_attempt(args, held_out_body, binding_sha)
        command = list(attempt["manifest"]["command"])
        log = Path(attempt["log"])
        write_state(
            "training_fold",
            completed_folds=[item["held_out_body"] for item in fold_results],
            fold_index=fold_index,
            current_fold=held_out_body,
            fold_attempt_ordinal=attempt["manifest"]["attempt_ordinal"],
            fold_attempt_manifest_logical_sha256=attempt["manifest"][
                "logical_sha256"
            ],
            command=command,
            ensemble_members=rac.ENSEMBLE_SIZE,
            steps_per_member=STEPS_PER_MEMBER,
            gpu=gpu,
        )
        with log.open("x", encoding="utf-8") as stream:
            training = subprocess.run(
                command,
                cwd=args.trainer.parent,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if training.returncode != 0:
            record_fold_attempt_failure(
                attempt,
                returncode=training.returncode,
                reason=(
                    "trainer_process_interrupted"
                    if training.returncode < 0
                    and -training.returncode in RECOVERABLE_TRAINING_SIGNALS
                    else "trainer_process_failed"
                ),
            )
            if (
                training.returncode < 0
                and -training.returncode in RECOVERABLE_TRAINING_SIGNALS
            ):
                os.kill(os.getpid(), -training.returncode)
            raise RacLoboSupervisorError(
                f"RAC fold {held_out_body} failed with exit {training.returncode}"
            )
        try:
            summarize_fold(
                Path(attempt["training_output"]),
                held_out_body,
                binding_sha,
                args.supplement_binding_sha256,
            )
        except Exception as error:
            record_fold_attempt_failure(
                attempt,
                returncode=0,
                reason="trainer_returned_zero_but_fold_validation_failed",
                error=error,
            )
            raise
        rebind_fold_summary_for_atomic_promotion(attempt, fold_output)
        promotion = promote_fold_attempt(attempt, fold_output)
        result = summarize_fold(
            fold_output,
            held_out_body,
            binding_sha,
            args.supplement_binding_sha256,
        )
        result["training_attempt"] = fold_attempt_audit(attempt, promotion)
        fold_results.append(result)
        write_state(
            "fold_complete",
            completed_folds=[item["held_out_body"] for item in fold_results],
            last_fold_result=result,
            gpu=gpu,
        )

    metrics = [item["source_validation"] for item in fold_results]
    final = signed(
        {
            "format": FINAL_FORMAT,
            "status": "five_outer_lobo_rac_source_only_training_complete",
            "completed_at_utc": datetime.fromtimestamp(
                max(
                    (args.output_root / f"outer_lobo_{body}" / "training_summary.json")
                    .stat().st_mtime for body in BODIES
                ),
                timezone.utc,
            ).isoformat(),
            "dataset_repo": base.DATASET_REPO,
            "dataset_revision": base.DATASET_REVISION,
            "task": base.TASK,
            "model_family": rac.MODEL_FAMILY,
            "rac_contract": rac.rac_contract(),
            "actor_execution_protocol": base._ACTIVE_EXECUTION_PROTOCOL,
            "actor_execution_protocol_binding": (
                base._ACTIVE_EXECUTION_PROTOCOL_BINDING
            ),
            "actor_execution_protocol_file_sha256": (
                base._ACTIVE_EXECUTION_PROTOCOL_BINDING["file_sha256"]
            ),
            "collection_audit": collection,
            "actor_authority": {
                "path": str(args.actor_authority),
                "file_sha256": base.sha256_file(args.actor_authority),
                "logical_sha256": actor_authority["logical_sha256"],
                "checkpoint_sha256": checkpoint_sha,
            },
            "training_binding": {
                "path": str(args.binding),
                "file_sha256": binding_sha,
                "logical_sha256": binding["logical_sha256"],
            },
            "supplement_binding_file_sha256": args.supplement_binding_sha256,
            "supplement_binding_authority": supplement_authority,
            "upstream_completion_authority": upstream_authority,
            "outer_folds": fold_results,
            "fold_count": len(fold_results),
            "members_per_fold": rac.ENSEMBLE_SIZE,
            "steps_per_member": STEPS_PER_MEMBER,
            "source_validation_fold_metrics": {
                key: base.numeric_summary([item.get(key) for item in metrics])
                for key in (
                    "macro_decision_pair_focal",
                    "pair_accuracy",
                    "decision_best_set_accuracy",
                )
            },
            "heldout_labels_used_for_training_normalization_or_selection": False,
            "heldout_task_success_measured": False,
            "cross_embodiment_task_success_claim_authorized": False,
            "next_required_stage": "nested_n1_n4_n8_paired_live_evaluation",
        }
    )
    final_path = args.output_root / "five_fold_rac_training_summary.json"
    base.create_once_or_verify(final_path, final, "five-fold RAC aggregate")
    args.run_exit.parent.mkdir(parents=True, exist_ok=True)
    args.run_exit.write_text("0\n", encoding="utf-8")
    write_state(
        "complete",
        completed_folds=list(BODIES),
        final_summary=str(final_path),
        final_summary_file_sha256=base.sha256_file(final_path),
        gpu=gpu,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if _FAILURE_STATE_PATH is not None:
            base.atomic_json(
                _FAILURE_STATE_PATH,
                {
                    "format": FORMAT,
                    "status": "failed",
                    "updated_at_utc": base.utc_now(),
                    "pid": os.getpid(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        if _FAILURE_RUN_EXIT_PATH is not None:
            _FAILURE_RUN_EXIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _FAILURE_RUN_EXIT_PATH.write_text("1\n", encoding="utf-8")
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


__all__ = [
    "FORMAT",
    "FINAL_FORMAT",
    "RacLoboSupervisorError",
    "create_fold_attempt",
    "fold_training_command",
    "normalized_args",
    "promote_fold_attempt",
    "rebind_fold_summary_for_atomic_promotion",
    "recover_missing_promotion_receipt",
    "summarize_fold",
    "validate_fold_attempt_history",
]
