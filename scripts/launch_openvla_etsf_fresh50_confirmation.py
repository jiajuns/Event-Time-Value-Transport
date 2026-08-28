#!/usr/bin/env python3
"""Run the frozen fresh-50 collection, one-shot evaluation, then progress suite.

The orchestrator never reads the outcome-bearing collection manifest before the
sealed evaluator has durably reserved access.  Collection readiness and safe
resume use the collector's label-free ``collection_identity.json`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch

from collect_openvla_etsf_event_branches import (
    ACTION_DIM,
    BODY,
    CHUNK,
    EVENT_VOCAB,
    HIDDEN_ANCHOR,
    IDENTITY_MANIFEST_NAME,
    INTERVENTION,
    LANGUAGE_CONTRACT,
    POST_QUERY_ACTION_CONTRACT,
    SCHEMA_VERSION,
)
from launch_openvla_etsf_counterfactual_v5 import (
    DEFAULT_CODE_ROOT,
    DEFAULT_DATA,
    DEFAULT_EVENT_SPEC,
    DEFAULT_FACTUAL_ROOT,
    DEFAULT_OUTPUT as DEFAULT_COUNTERFACTUAL_OUTPUT,
    DEFAULT_PYTHON,
    SEEDS as ENSEMBLE_SEEDS,
    atomic_json,
    sha256,
    wait_for_gpu_idle,
)
from openvla_etsf_oof_final_contract import (
    OOF_TEST_POLICY,
    validate_authorized_oof_final,
)


FORMAT = "etsf_openvla_fresh50_confirmation_pipeline_v1"
FRESH_STATUS = "fresh_confirmation_preregistered_resolved"
FRESH_LABEL_CONTRACT = (
    "reset_identity_only_no_policy_no_action_no_event_no_success_no_reward"
)
IDENTITY_FORMAT = "etsf_event_branch_collection_identity_v1"
ENSEMBLE_FORMAT = "etsf_counterfactual_ensemble_v1"
LEGACY_TEST_POLICY = (
    "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
)
EVALUATION_FORMAT = "etsf_counterfactual_sealed_evaluation_v1"
DEFAULT_MODEL = Path(
    "/home/user/etsf_openvla_models/"
    "RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot"
)
DEFAULT_RLINF_ROOT = Path("/home/user/etsf_stage0/RLinf")
DEFAULT_ROBOTWIN_ROOT = Path("/home/user/etsf_stage0/RoboTwin")
DEFAULT_ROBOTWIN_CODE = Path("/home/user/etsf_stage0/RoboTwin_RLinf_support")
DEFAULT_FRESH_MANIFEST = Path(
    "/home/user/etsf_event_world_model_code_20260827/artifacts/protocol/"
    "fresh_confirmation_seeds_20260827.json"
)
DEFAULT_OUTPUT = Path(
    "/home/user/etsf_openvla_fresh50_confirmation_move_can_pot_20260827"
)


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def json_equivalent(left: Any, right: Any) -> bool:
    def normalize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return normalize(value.detach().cpu().tolist())
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [normalize(item) for item in value]
        return value

    return normalize(left) == normalize(right)


def resolve_recorded_path(recorded: str, anchor: Path) -> Path:
    path = Path(recorded).expanduser()
    if path.is_file():
        return path.resolve()
    portable = anchor / path.name
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(path)


def wait_for_json(path: Path, timeout_seconds: float, poll_seconds: float) -> None:
    started = time.monotonic()
    while True:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, Mapping):
                return
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(f"timed out waiting for frozen JSON: {path}")
        time.sleep(max(min(poll_seconds, timeout_seconds - elapsed, 60.0), 0.01))


def wait_for_counterfactual_complete(
    root: Path, timeout_seconds: float, poll_seconds: float
) -> None:
    started = time.monotonic()
    while True:
        audit_path = root / "launch_audit.json"
        oof_summary_path = root / "training_summary.json"
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                audit = None
            if isinstance(audit, Mapping):
                status = str(audit.get("status", ""))
                if status == "launcher_complete":
                    return
                if status.startswith("failed"):
                    raise RuntimeError(
                        f"counterfactual prerequisite failed: {status}"
                    )
        if oof_summary_path.is_file():
            try:
                summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = None
            if isinstance(summary, Mapping):
                if (
                    summary.get("status") == "complete"
                    and summary.get("oof_authorized") is True
                ):
                    return
                if str(summary.get("status", "")).startswith("failed"):
                    raise RuntimeError("OOF final prerequisite failed")
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError("timed out waiting for counterfactual launcher_complete")
        time.sleep(max(min(poll_seconds, timeout_seconds - elapsed, 60.0), 0.01))


def _ordered_fresh_rows(manifest: Mapping[str, Any]) -> list[dict[str, int]]:
    rows = manifest.get("test")
    if not isinstance(rows, list) or len(rows) != 50:
        raise RuntimeError("fresh reset-only manifest must contain exactly 50 test rows")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("fresh seed row must be a mapping")
        requested = int(row.get("requested_seed", row.get("seed", -1)))
        resolved = int(row.get("resolved_seed", -1))
        if int(row.get("seed", requested)) != requested or requested < 0 or resolved < 0:
            raise RuntimeError("fresh seed row identity is invalid")
        result.append(
            {"seed": requested, "requested_seed": requested, "resolved_seed": resolved}
        )
    requested = [row["requested_seed"] for row in result]
    resolved = [row["resolved_seed"] for row in result]
    if len(set(requested)) != 50 or len(set(resolved)) != 50:
        raise RuntimeError("fresh requested/resolved seeds must each be unique")
    return result


def audit_fresh_manifest(path: Path, task: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("fresh seed manifest must contain a JSON object")
    if (
        int(manifest.get("schema_version", -1)) != 1
        or manifest.get("status") != FRESH_STATUS
        or str(manifest.get("task", "")) != task
        or manifest.get("label_access_contract") != FRESH_LABEL_CONTRACT
    ):
        raise RuntimeError("fresh seed manifest is not the frozen reset-only contract")
    rows = _ordered_fresh_rows(manifest)
    requested = [row["requested_seed"] for row in rows]
    resolved = [row["resolved_seed"] for row in rows]
    if [int(value) for value in manifest.get("requested_seeds", [])] != requested:
        raise RuntimeError("fresh requested seed mirror/order differs")
    if [int(value) for value in manifest.get("resolved_seeds", [])] != resolved:
        raise RuntimeError("fresh resolved seed mirror/order differs")

    candidate_path = resolve_recorded_path(
        str(manifest.get("candidate_manifest", "")), path.parent
    )
    official_path = resolve_recorded_path(
        str(manifest.get("official_seed_registry", "")), path.parent
    )
    if sha256(candidate_path) != str(manifest.get("candidate_manifest_sha256", "")):
        raise RuntimeError("fresh candidate manifest SHA256 mismatch")
    if sha256(official_path) != str(
        manifest.get("official_seed_registry_sha256", "")
    ):
        raise RuntimeError("official seed registry SHA256 mismatch")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("status") != "preregistered_unresolved"
        or str(candidate.get("task", "")) != task
        or manifest.get("selection_rule") != candidate.get("selection_rule")
        or manifest.get("freeze_rule") != candidate.get("freeze_rule")
    ):
        raise RuntimeError("fresh candidate/reset-only selection contract differs")
    candidate_order = [int(value) for value in candidate["candidate_requested_seeds"]]
    positions = [candidate_order.index(seed) for seed in requested]
    if positions != sorted(positions):
        raise RuntimeError("fresh selected seeds do not preserve preregistered order")
    audited = manifest.get("audit")
    if not isinstance(audited, list):
        raise RuntimeError("fresh reset-only audit is missing")
    selected_audit = [
        {
            "requested_seed": int(row["requested_seed"]),
            "resolved_seed": int(row["resolved_seed"]),
        }
        for row in audited
        if isinstance(row, Mapping) and row.get("decision") == "selected"
    ]
    if selected_audit != [
        {
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
        }
        for row in rows
    ]:
        raise RuntimeError("fresh selected rows differ from reset-only audit order")
    official = json.loads(official_path.read_text(encoding="utf-8"))
    official_seeds = {
        int(value) for value in official.get(task, {}).get("success_seeds", [])
    }
    if not official_seeds:
        raise RuntimeError("official seed registry lacks the selected task")
    overlap = (set(requested) | set(resolved)) & official_seeds
    if overlap:
        raise RuntimeError(f"fresh seeds overlap official registry: {sorted(overlap)}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "status": FRESH_STATUS,
        "task": task,
        "rows": rows,
        "requested_seeds": requested,
        "resolved_seeds": resolved,
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
        },
        "official_seed_registry": {
            "path": str(official_path),
            "sha256": sha256(official_path),
        },
        "label_access_contract": FRESH_LABEL_CONTRACT,
    }


def audit_frozen_ensemble(root: Path, event_spec_sha256: str) -> dict[str, Any]:
    audit_path = root / "launch_audit.json"
    manifest_path = root / "ensemble_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("counterfactual launcher is not complete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract")
    test_policy = manifest.get("test_policy")
    oof_audit: Mapping[str, Any] | None = None
    launch: Mapping[str, Any] | None = None
    if test_policy == LEGACY_TEST_POLICY:
        if not audit_path.is_file():
            raise RuntimeError("legacy counterfactual launch audit is missing")
        launch = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(launch, Mapping) or launch.get("status") != "launcher_complete":
            raise RuntimeError("legacy counterfactual launcher is not complete")
    elif test_policy == OOF_TEST_POLICY:
        oof_audit = validate_authorized_oof_final(manifest_path, manifest)
    else:
        raise RuntimeError("counterfactual ensemble test policy is unsupported")
    if (
        manifest.get("format") != ENSEMBLE_FORMAT
        or not isinstance(contract, Mapping)
        or str(contract.get("event_spec_sha256", "")) != event_spec_sha256
    ):
        raise RuntimeError("counterfactual ensemble is not a frozen formal output")
    selection_contract = contract.get("scoring_selection_contract")
    if test_policy == LEGACY_TEST_POLICY and (
        not isinstance(selection_contract, Mapping)
        or selection_contract.get("selection_data")
        != "validation_only_no_sealed_test"
    ):
        raise RuntimeError("legacy ensemble scoring was not frozen on validation only")
    aggregate_record = manifest.get("ensemble_checkpoint")
    if not isinstance(aggregate_record, Mapping) or not aggregate_record.get("path"):
        raise RuntimeError("ensemble aggregate checkpoint provenance is missing")
    aggregate = resolve_recorded_path(str(aggregate_record["path"]), root)
    aggregate_digest = sha256(aggregate)
    if aggregate_digest != str(aggregate_record.get("sha256", "")):
        raise RuntimeError("ensemble aggregate checkpoint SHA256 mismatch")
    payload = torch.load(aggregate, map_location="cpu", weights_only=False)
    mirrored = (
        "format",
        "config",
        "contract",
        "normalization",
        "duration_scale",
        "success_calibration",
        "scoring",
        "scoring_selection",
        "guard",
        "predicate_contract",
        "candidate_contract",
    )
    if not isinstance(payload, Mapping) or any(
        not json_equivalent(payload.get(key), manifest.get(key)) for key in mirrored
    ):
        raise RuntimeError("ensemble manifest/aggregate frozen contract mismatch")
    guard = manifest.get("guard")
    selection = manifest.get("scoring_selection")
    if not isinstance(guard, Mapping) or guard.get("enabled") is not True:
        raise RuntimeError(
            "fresh confirmation prohibited: frozen validation guard is disabled"
        )
    if not isinstance(selection, Mapping):
        raise RuntimeError("fresh confirmation scoring selection audit is missing")
    selected_id = str(selection.get("selected_candidate_id", ""))
    candidate_rows = selection.get("candidates")
    selected_rows = [
        row
        for row in candidate_rows or []
        if isinstance(row, Mapping) and str(row.get("candidate_id", "")) == selected_id
    ]
    if test_policy == LEGACY_TEST_POLICY and (
        not selected_id
        or not isinstance(candidate_rows, list)
        or len(selected_rows) != 1
        or selected_rows[0].get("passes_pre_guard_evidence_gate") is not True
    ):
        raise RuntimeError(
            "fresh confirmation prohibited: selected scoring failed the "
            "pre_guard_evidence_gate"
        )
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(ENSEMBLE_SEEDS):
        raise RuntimeError("ensemble must contain exactly three frozen members")
    member_rows = []
    for expected_seed, member in zip(ENSEMBLE_SEEDS, members):
        if not isinstance(member, Mapping) or int(member.get("seed", -1)) != expected_seed:
            raise RuntimeError("ensemble member seed/order differs")
        member_path = resolve_recorded_path(str(member.get("path", "")), root)
        digest = sha256(member_path)
        if digest != str(member.get("sha256", "")):
            raise RuntimeError("ensemble member SHA256 mismatch")
        member_rows.append(
            {"seed": expected_seed, "path": str(member_path), "sha256": digest}
        )
    return {
        "root": str(root.resolve()),
        "authorization_mode": (
            "authorized_oof_final" if oof_audit is not None else "legacy_validation_split"
        ),
        "launch_audit": str(audit_path.resolve()) if launch is not None else None,
        "launch_audit_sha256": sha256(audit_path) if launch is not None else None,
        "oof_authorization": dict(oof_audit) if oof_audit is not None else None,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "aggregate": str(aggregate),
        "aggregate_sha256": aggregate_digest,
        "members": member_rows,
        "development_seed_suffixes": sorted(
            {
                int(str(key).rsplit("|", 1)[-1])
                for name in ("train_groups", "validation_groups", "sealed_test_groups")
                for key in contract.get(name, [])
            }
        ),
    }


def validate_collection_identity(
    root: Path,
    fresh: Mapping[str, Any],
    event_spec_sha256: str,
    model_path: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    identity_path = root / IDENTITY_MANIFEST_NAME
    if not identity_path.is_file():
        raise RuntimeError("fresh collection lacks label-free identity manifest")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    forbidden = {
        "success",
        "steps",
        "post_event_id",
        "next_event_duration_observed",
        "candidate_successes",
        "candidate_success_rates",
        "groups_with_outcome_variation",
        "dense_label_counts",
    }
    groups = identity.get("groups")
    if not isinstance(groups, list) or forbidden & set(identity) or any(
        not isinstance(row, Mapping) or forbidden & set(row) for row in groups
    ):
        raise RuntimeError("collection identity manifest contains labels or invalid rows")
    expected = {
        "format": IDENTITY_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "task": fresh["task"],
        "body": BODY,
        "model_path": str(model_path),
        "requested_seeds": fresh["requested_seeds"],
        "seed_registry": "explicit_fresh_confirmation",
        "fresh_seed_manifest_sha256": fresh["sha256"],
        "candidate_count": 4,
        "blends": [0.25, 0.5, 0.75],
        "temperature": 0.7,
        "top_k": 4,
        "preserve_grippers": True,
        "intervention": INTERVENTION,
        "language_contract": LANGUAGE_CONTRACT,
        "event_vocab": list(EVENT_VOCAB),
        "event_spec_sha256": event_spec_sha256,
        "hidden_dim": 4096,
        "hidden_anchor": HIDDEN_ANCHOR,
        "action_dim": ACTION_DIM,
        "action_chunk": CHUNK,
        "label_access_contract": (
            "identity_only_no_success_steps_event_or_outcome_fields"
        ),
        "hdf5_sha256_pre_evaluation": "not_computed",
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise RuntimeError(f"fresh collection identity contract mismatch for {key}")
    status = str(identity.get("status", ""))
    if status not in {"collecting", "complete"}:
        raise RuntimeError("fresh collection identity status is invalid")
    if require_complete and status != "complete":
        raise RuntimeError("fresh collection is not complete")
    if len(groups) > 50 or (require_complete and len(groups) != 50):
        raise RuntimeError("fresh collection group count is invalid")
    expected_rows = fresh["rows"]
    candidate_names: tuple[str, ...] | None = None
    resolved_prefix = []
    for index, row in enumerate(groups):
        expected_row = expected_rows[index]
        if (
            int(row.get("index", -1)) != index
            or int(row.get("seed", -1)) != expected_row["seed"]
            or int(row.get("requested_seed", -1)) != expected_row["requested_seed"]
            or int(row.get("resolved_seed", -1)) != expected_row["resolved_seed"]
        ):
            raise RuntimeError("fresh collection group seed/order differs")
        path = root / "groups" / str(row.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            attrs = handle.attrs
            if (
                int(attrs.get("schema_version", -1)) != SCHEMA_VERSION
                or int(attrs.get("seed", -1)) != expected_row["requested_seed"]
                or int(attrs.get("requested_seed", -1)) != expected_row["requested_seed"]
                or int(attrs.get("resolved_seed", -1)) != expected_row["resolved_seed"]
                or int(attrs.get("candidate_count", -1)) != 4
                or str(attrs.get("language_contract", "")) != LANGUAGE_CONTRACT
                or not bool(attrs.get("branch_instruction_consistent", False))
                or str(attrs.get("intervention", "")) != INTERVENTION
                or str(attrs.get("post_query_action_contract", ""))
                != POST_QUERY_ACTION_CONTRACT
            ):
                raise RuntimeError("fresh HDF5 identity/language/candidate contract differs")
            names = tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["candidate_names"][:]
            )
        if len(names) != 4 or names[0] != "deterministic":
            raise RuntimeError("fresh candidate zero is not deterministic")
        if tuple(map(str, row.get("candidate_names", []))) != names:
            raise RuntimeError("fresh identity/group candidate names differ")
        if candidate_names is None:
            candidate_names = names
        elif candidate_names != names:
            raise RuntimeError("fresh candidate names/order changed across groups")
        resolved_prefix.append(expected_row["resolved_seed"])
    if identity.get("resolved_seeds", resolved_prefix) != resolved_prefix:
        raise RuntimeError("fresh collection resolved seed mirror differs")
    return {
        "path": str(identity_path.resolve()),
        "sha256": sha256(identity_path),
        "status": status,
        "completed": len(groups),
        "candidate_names": list(candidate_names or ()),
        "labels_read": False,
        "hdf5_sha256_computed": False,
    }


def validate_evaluation_result(
    output: Path, fresh: Mapping[str, Any], ensemble: Mapping[str, Any]
) -> dict[str, Any]:
    marker = output / "evaluated_once.json"
    if not marker.is_file():
        raise RuntimeError("one-shot evaluation marker is missing")
    result = json.loads(marker.read_text(encoding="utf-8"))
    protocol = result.get("evaluation_protocol")
    collection = result.get("collection_audit")
    ensemble_record = result.get("ensemble_manifest")
    group_files = collection.get("group_files") if isinstance(collection, Mapping) else None
    if (
        result.get("format") != EVALUATION_FORMAT
        or result.get("status") != "complete"
        or result.get("evaluated_once") is not True
        or result.get("sealed_labels_first_read_by")
        != "this_evaluator_after_atomic_reservation"
        or not isinstance(protocol, Mapping)
        or protocol.get("evidence_tier") != "fresh_confirmatory"
        or protocol.get("confirmatory") is not True
        or not isinstance(protocol.get("fresh_seed_manifest"), Mapping)
        or protocol["fresh_seed_manifest"].get("sha256") != fresh["sha256"]
        or not isinstance(collection, Mapping)
        or not isinstance(group_files, list)
        or len(group_files) != 50
        or not isinstance(ensemble_record, Mapping)
        or ensemble_record.get("sha256") != ensemble["manifest_sha256"]
    ):
        raise RuntimeError("fresh one-shot evaluation result contract is invalid")
    return {"path": str(marker.resolve()), "sha256": sha256(marker), "status": "complete"}


def build_commands(args: argparse.Namespace, fresh: Mapping[str, Any], ensemble: Mapping[str, Any]) -> dict[str, Any]:
    # Preserve the venv entry-point path. ``resolve()`` follows the venv
    # symlink to its base interpreter and can silently drop venv-only packages.
    python_path = args.python_bin.expanduser().absolute()
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(f"python interpreter is not executable: {python_path}")
    python = str(python_path)
    collection_root = (args.output / "fresh50_data").resolve()
    evaluation_root = (args.output / "fresh50_evaluation").resolve()
    progress_root = (args.output / "progress_baselines").resolve()
    collector = [
        python,
        str(args.collector.resolve()),
        "--model-path", str(args.model_path.resolve()),
        "--rlinf-root", str(args.rlinf_root.resolve()),
        "--robotwin-root", str(args.robotwin_root.resolve()),
        "--robotwin-code", str(args.robotwin_code.resolve()),
        "--event-spec", str(args.event_spec.resolve()),
        "--output", str(collection_root),
        "--task", args.task,
        "--seeds-file", fresh["path"],
        "--seeds-key", "test",
        "--allow-unregistered-seeds",
        "--fresh-seed-manifest", fresh["path"],
        "--blends", "0.25", "0.5", "0.75",
        "--temperature", "0.7",
        "--top-k", "4",
    ]
    evaluator = [
        python,
        str(args.evaluator.resolve()),
        "--ensemble-manifest", ensemble["manifest"],
        "--sealed-data", str(collection_root),
        "--event-spec", str(args.event_spec.resolve()),
        "--fresh-seed-manifest", fresh["path"],
        "--output", str(evaluation_root),
        "--device", "cuda",
    ]
    progress = [
        python,
        str(args.progress_launcher.resolve()),
        "--data", str(args.data.resolve()),
        "--factual-root", str(args.factual_root.resolve()),
        "--counterfactual-root", str(args.counterfactual_root.resolve()),
        "--event-spec", str(args.event_spec.resolve()),
        "--output", str(progress_root),
        "--trainer", str(args.progress_trainer.resolve()),
        "--python-bin", python,
        "--wait-timeout-seconds", str(args.wait_timeout_seconds),
        "--poll-seconds", str(args.poll_seconds),
        "--gpu-wait-timeout-seconds", str(args.gpu_wait_timeout_seconds),
        "--gpu-poll-seconds", str(args.gpu_poll_seconds),
    ]
    return {
        name: {
            "argv": command,
            "argv_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        for name, command in (
            ("collector", collector),
            ("evaluator", evaluator),
            ("progress", progress),
        )
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    wait_for_json(args.fresh_seed_manifest.resolve(), args.wait_timeout_seconds, args.poll_seconds)
    fresh = audit_fresh_manifest(args.fresh_seed_manifest.resolve(), args.task)
    event_spec = args.event_spec.resolve()
    if not event_spec.is_file():
        raise FileNotFoundError(event_spec)
    event_digest = sha256(event_spec)
    wait_for_counterfactual_complete(
        args.counterfactual_root.resolve(), args.wait_timeout_seconds, args.poll_seconds
    )
    ensemble = audit_frozen_ensemble(args.counterfactual_root.resolve(), event_digest)
    overlap = (
        set(fresh["requested_seeds"]) | set(fresh["resolved_seeds"])
    ) & set(ensemble["development_seed_suffixes"])
    if overlap:
        raise RuntimeError(f"fresh seeds overlap ensemble development splits: {sorted(overlap)}")
    for path in (
        args.python_bin,
        args.collector,
        args.evaluator,
        args.progress_launcher,
        args.progress_trainer,
        args.model_path,
        args.rlinf_root,
        args.robotwin_root,
        args.robotwin_code,
        args.data,
        args.factual_root,
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())
    commands = build_commands(args, fresh, ensemble)
    contract = {
        "fresh_seed_manifest_sha256": fresh["sha256"],
        "ensemble_manifest_sha256": ensemble["manifest_sha256"],
        "aggregate_checkpoint_sha256": ensemble["aggregate_sha256"],
        "event_spec_sha256": event_digest,
        "commands": {
            name: row["argv_sha256"] for name, row in commands.items()
        },
        "stage_order": ["collector", "evaluator_once", "progress_after_evaluation"],
        "sealed_pre_evaluation_access": (
            "label_free_collection_identity_and_hdf5_identity_attrs_only"
        ),
    }
    return {
        "format": FORMAT,
        "status": "preflight_complete",
        "contract": contract,
        "contract_sha256": payload_sha256(contract),
        "fresh": fresh,
        "ensemble": ensemble,
        "event_spec": {"path": str(event_spec), "sha256": event_digest},
        "paths": {
            "output": str(args.output.resolve()),
            "collection": str((args.output / "fresh50_data").resolve()),
            "evaluation": str((args.output / "fresh50_evaluation").resolve()),
            "progress": str((args.output / "progress_baselines").resolve()),
        },
        "commands": commands,
        "sealed_labels_read_by_launcher_before_evaluation": False,
    }


def acquire_lock(path: Path, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise RuntimeError("fresh-50 pipeline lock exists; use --resume only after audit")
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("fresh-50 pipeline lock is unreadable") from error
        if lock.get("host") == socket.gethostname() and Path(
            f"/proc/{int(lock.get('pid', -1))}"
        ).exists():
            raise RuntimeError("fresh-50 pipeline is still running")
        path.unlink()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    local = Path(__file__).resolve().parent
    default_python = DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable)
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-seed-manifest", type=Path, default=DEFAULT_FRESH_MANIFEST)
    parser.add_argument("--counterfactual-root", type=Path, default=DEFAULT_COUNTERFACTUAL_OUTPUT)
    parser.add_argument("--event-spec", type=Path, default=DEFAULT_EVENT_SPEC)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--factual-root", type=Path, default=DEFAULT_FACTUAL_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rlinf-root", type=Path, default=DEFAULT_RLINF_ROOT)
    parser.add_argument("--robotwin-root", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--robotwin-code", type=Path, default=DEFAULT_ROBOTWIN_CODE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-bin", type=Path, default=default_python)
    parser.add_argument("--collector", type=Path, default=local / "collect_openvla_etsf_event_branches.py")
    parser.add_argument("--evaluator", type=Path, default=local / "evaluate_openvla_etsf_counterfactual_sealed.py")
    parser.add_argument("--progress-launcher", type=Path, default=local / "launch_openvla_etsf_progress_v5.py")
    parser.add_argument("--progress-trainer", type=Path, default=local / "train_openvla_etsf_progress_baseline.py")
    parser.add_argument("--task", default="move_can_pot")
    parser.add_argument("--wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.wait_timeout_seconds < 0
        or args.gpu_wait_timeout_seconds < 0
        or not 0 < args.poll_seconds <= 60
        or not 0 < args.gpu_poll_seconds <= 60
    ):
        raise ValueError("timeouts must be non-negative and polling must lie in (0,60]")
    audit = preflight(args)
    if args.dry_run:
        print("FRESH50_CONFIRMATION_DRY_RUN=" + json.dumps(audit, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("fresh-50 formal pipeline requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090" not in gpu_name:
        raise RuntimeError(f"fresh-50 formal pipeline requires RTX 4090, found {gpu_name!r}")

    output = args.output.resolve()
    audit_path = output / "pipeline_audit.json"
    if output.exists() and any(output.iterdir()):
        if not audit_path.is_file():
            raise RuntimeError("partial fresh-50 output lacks pipeline audit; refusing")
        previous = json.loads(audit_path.read_text(encoding="utf-8"))
        if previous.get("contract_sha256") != audit["contract_sha256"]:
            raise RuntimeError("fresh-50 output belongs to a different frozen contract")
        if previous.get("status") == "complete":
            evaluation = validate_evaluation_result(
                Path(audit["paths"]["evaluation"]), audit["fresh"], audit["ensemble"]
            )
            progress_summary = Path(audit["paths"]["progress"]) / "progress_baseline_suite_summary.json"
            progress_audit = Path(audit["paths"]["progress"]) / "launch_audit.json"
            if (
                not progress_summary.is_file()
                or sha256(progress_summary) != previous.get("progress_suite_sha256")
                or not progress_audit.is_file()
                or sha256(progress_audit) != previous.get("progress_audit_sha256")
                or json.loads(progress_audit.read_text(encoding="utf-8")).get("status")
                != "launcher_complete"
            ):
                raise RuntimeError("completed fresh-50 progress suite/audit mismatch")
            print(
                "FRESH50_CONFIRMATION_SKIP="
                + json.dumps({"evaluation": evaluation, "progress": str(progress_summary)}, sort_keys=True)
            )
            return
        if not args.resume:
            raise RuntimeError("partial fresh-50 output requires explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "pipeline.lock"
    acquire_lock(lock_path, args.resume)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
    collection_root = Path(audit["paths"]["collection"])
    evaluation_root = Path(audit["paths"]["evaluation"])
    progress_root = Path(audit["paths"]["progress"])
    try:
        audit["runtime"] = {"gpu_name": gpu_name, "gpu_idle_checks": []}
        audit["status"] = "collection_pending"
        atomic_json(audit_path, audit)
        identity_path = collection_root / IDENTITY_MANIFEST_NAME
        if identity_path.exists():
            collection = validate_collection_identity(
                collection_root,
                audit["fresh"],
                audit["event_spec"]["sha256"],
                args.model_path.resolve(),
                require_complete=False,
            )
        else:
            collection = None
        if collection is None or collection["status"] != "complete":
            idle = wait_for_gpu_idle(
                args.gpu_wait_timeout_seconds,
                args.gpu_poll_seconds,
                gpu_index=0,
            )
            idle["stage"] = "collector"
            audit["runtime"]["gpu_idle_checks"].append(idle)
            audit["status"] = "collection_running_resumable"
            atomic_json(audit_path, audit)
            subprocess.run(audit["commands"]["collector"]["argv"], check=True, env=environment)
        collection = validate_collection_identity(
            collection_root,
            audit["fresh"],
            audit["event_spec"]["sha256"],
            args.model_path.resolve(),
            require_complete=True,
        )
        audit["collection_identity"] = collection
        audit["status"] = "collection_complete_evaluation_pending"
        atomic_json(audit_path, audit)

        marker = evaluation_root / "evaluated_once.json"
        if marker.exists():
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
            if marker_value.get("status") != "complete":
                audit["status"] = (
                    "failed_after_one_shot_reservation_rerun_prohibited"
                )
                atomic_json(audit_path, audit)
                raise RuntimeError(
                    "one-shot evaluation was already reserved/interrupted; rerun prohibited"
                )
        else:
            idle = wait_for_gpu_idle(
                args.gpu_wait_timeout_seconds,
                args.gpu_poll_seconds,
                gpu_index=0,
            )
            idle["stage"] = "evaluator_once"
            audit["runtime"]["gpu_idle_checks"].append(idle)
            audit["status"] = "one_shot_evaluation_running_no_retry_after_reservation"
            atomic_json(audit_path, audit)
            subprocess.run(audit["commands"]["evaluator"]["argv"], check=True, env=environment)
        evaluation = validate_evaluation_result(
            evaluation_root, audit["fresh"], audit["ensemble"]
        )
        audit["evaluation"] = evaluation
        audit["status"] = "evaluation_complete_progress_pending"
        atomic_json(audit_path, audit)

        # This command is reachable only after the complete confirmatory marker
        # has been verified.  The progress launcher supplies its own 4090 idle
        # gate and fail-closed completion validation.
        subprocess.run(audit["commands"]["progress"]["argv"], check=True, env=environment)
        progress_summary = progress_root / "progress_baseline_suite_summary.json"
        progress_audit = progress_root / "launch_audit.json"
        if not progress_summary.is_file() or not progress_audit.is_file():
            raise RuntimeError("progress launcher returned without a complete suite")
        if json.loads(progress_audit.read_text(encoding="utf-8")).get("status") != "launcher_complete":
            raise RuntimeError("progress launcher audit is not complete")
        audit["progress_suite"] = str(progress_summary.resolve())
        audit["progress_suite_sha256"] = sha256(progress_summary)
        audit["progress_audit"] = str(progress_audit.resolve())
        audit["progress_audit_sha256"] = sha256(progress_audit)
        audit["status"] = "complete"
        atomic_json(audit_path, audit)
        print(
            "FRESH50_CONFIRMATION_COMPLETE="
            + json.dumps(
                {
                    "evaluation": evaluation,
                    "progress_suite": str(progress_summary.resolve()),
                    "pipeline_audit": str(audit_path.resolve()),
                },
                sort_keys=True,
            )
        )
    except BaseException:
        if audit.get("status") == "one_shot_evaluation_running_no_retry_after_reservation":
            audit["status"] = "failed_after_one_shot_reservation_rerun_prohibited"
        elif audit.get("status") not in {
            "evaluation_complete_progress_pending",
            "failed_after_one_shot_reservation_rerun_prohibited",
        }:
            audit["status"] = "collection_interrupted_resumable"
        atomic_json(audit_path, audit)
        raise
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
