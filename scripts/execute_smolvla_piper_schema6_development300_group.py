#!/usr/bin/env python3
"""Sealed per-group worker for the development300 exact collection runner.

This is the only process allowed to receive the collector's label-bearing
in-memory record.  It persists that record, validates it inside the collector
boundary, seals formal-target payloads, and publishes only an outcome-free
receipt to the parent runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from preregister_smolvla_piper_schema6_target_development300 import (
    INSTRUCTION,
    canonical_sha256,
)
from resolve_smolvla_piper_target_reset_only import array_sha256, scene_sha256
from run_smolvla_piper_schema6_development300_collection import (
    GROUP_SIGNATURE,
    PLAN_SIGNATURE,
    STATIC_PLAN_FORMAT,
    WORKER_GROUP_RECEIPT_FORMAT,
    Development300RunnerError,
    atomic_json,
    file_sha256,
    is_sha,
    load_json,
    opaque_file_sha256,
    safe_path,
    signed,
    validate_runner_authority,
    verify_signed,
)


RESET_RECEIPT_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_worker_reset_receipt_v1"
)
ACCOUNTING_FORMAT = (
    "etsf_smolvla_piper_schema6_development300_candidate_accounting_v1"
)


class Development300WorkerError(RuntimeError):
    """The sealed collector worker cannot prove one exact group."""


def _load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise Development300WorkerError("bound implementation cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _identity_snapshot_sha(raw: Mapping[str, Any]) -> dict[str, str]:
    if set(raw) != {
        "scene_state",
        "measured_joint_state",
        "commanded_drive_target",
    }:
        raise Development300WorkerError("runtime reset identity fields changed")
    return {
        "initial_scene_state_sha256": scene_sha256(raw["scene_state"]),
        "initial_measured_joint_state_sha256": array_sha256(
            raw["measured_joint_state"], role="measured joint state"
        ),
        "initial_commanded_drive_target_sha256": array_sha256(
            raw["commanded_drive_target"], role="commanded drive target"
        ),
    }


def _validate_identity(raw: Mapping[str, Any], identity_row: Mapping[str, Any]) -> None:
    observed = _identity_snapshot_sha(raw)
    for field, value in observed.items():
        if value != identity_row[field]:
            raise Development300WorkerError("runtime reset identity changed")


def _candidate_accounting(
    record: Mapping[str, Any], *, command_sha256: str
) -> dict[str, Any]:
    if record.get("status") != "collected_development_group":
        raise Development300WorkerError(
            "collector did not produce a complete exact group"
        )
    root = record.get("root_query")
    branches = record.get("branches")
    if not isinstance(root, Mapping) or not isinstance(branches, list):
        raise Development300WorkerError("collector candidate inventory is missing")
    feasibility = np.asarray(root.get("feasibility_mask"), dtype=bool)
    native_sha = list(root.get("native_action_sha256", ()))
    if (
        feasibility.shape != (4,)
        or len(native_sha) != 4
        or any(not is_sha(value) for value in native_sha)
    ):
        raise Development300WorkerError("collector did not preserve four candidates")
    legal = np.flatnonzero(feasibility).astype(int).tolist()
    by_original: dict[int, Mapping[str, Any]] = {}
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise Development300WorkerError("collector branch record changed")
        original = branch.get("original_candidate_index")
        if type(original) is not int or original in by_original:
            raise Development300WorkerError("collector branch identity changed")
        by_original[original] = branch
    if len(legal) < 2 or sorted(by_original) != legal:
        raise Development300WorkerError("feasible candidate branch accounting changed")
    rows = [
        {
            "original_candidate_index": index,
            "native_action_sha256": native_sha[index],
            "feasible": bool(feasibility[index]),
            "executed": bool(feasibility[index]),
            "right_censored": not bool(feasibility[index]),
            "execution_status": "executed_legal_branch"
            if feasibility[index]
            else "nonexecuted_censored_infeasible",
        }
        for index in range(4)
    ]
    return signed(
        {
            "format": ACCOUNTING_FORMAT,
            "status": "complete_four_original_candidate_records",
            "command_sha256": command_sha256,
            "candidate_original_indices": [0, 1, 2, 3],
            "records": rows,
            "success_event_outcome_or_label_included": False,
        },
        "candidate_accounting_sha256",
    )


def _seal_payload(payload: Path, *, formal: bool) -> None:
    file_mode = 0o400 if formal else 0o444
    directory_mode = 0o500 if formal else 0o555
    for path in payload.iterdir():
        if path.is_symlink() or not path.is_file():
            raise Development300WorkerError("worker payload contains an invalid entry")
        path.chmod(file_mode)
    payload.chmod(directory_mode)


def _load_bound_inputs(
    *,
    authority_path: Path,
    expected_authority_file_sha256: str,
    plan_path: Path,
    expected_plan_file_sha256: str,
    global_ordinal: int,
    staging_payload: Path,
) -> dict[str, Any]:
    authority_path = safe_path(authority_path, "runner authority")
    if file_sha256(authority_path) != expected_authority_file_sha256:
        raise Development300WorkerError("runner authority file SHA changed")
    authority = load_json(authority_path, "runner authority")
    decoded = validate_runner_authority(authority, verify_runtime_files=True)
    plan_path = safe_path(plan_path, "runner static plan")
    if file_sha256(plan_path) != expected_plan_file_sha256:
        raise Development300WorkerError("runner static plan file SHA changed")
    plan = load_json(plan_path, "runner static plan")
    plan_sha = verify_signed(plan, PLAN_SIGNATURE, "runner static plan")
    if (
        plan.get("format") != STATIC_PLAN_FORMAT
        or plan.get("runner_authority")
        != {
            "path": str(authority_path),
            "file_sha256": expected_authority_file_sha256,
            "logical_sha256": decoded["runner_authority_sha256"],
        }
        or plan.get("command_sha256")
        != [command["command_sha256"] for command in decoded["commands"]]
    ):
        raise Development300WorkerError("runner static plan binding changed")
    if type(global_ordinal) is not int or not 0 <= global_ordinal < len(
        decoded["commands"]
    ):
        raise Development300WorkerError("worker command ordinal is invalid")
    command = decoded["commands"][global_ordinal]
    expected_payload = (
        decoded["output_root"]
        / "_runner"
        / "staging"
        / f"stage_{global_ordinal:03d}"
        / "payload"
    )
    payload = safe_path(staging_payload, "staging payload")
    if payload != expected_payload or payload.exists() or payload.is_symlink():
        raise Development300WorkerError("staging payload binding changed")
    if command["outputs"]["seed_root"] != authority["exact_execution"]["commands"][
        global_ordinal
    ]["final_seed_root"]:
        raise Development300WorkerError("final command output binding changed")
    runner_root = decoded["output_root"] / "_runner"
    run_claim = load_json(runner_root / "run_claim.json", "runner one-shot claim")
    if (
        verify_signed(run_claim, "run_claim_sha256", "runner one-shot claim")
        != run_claim.get("run_claim_sha256")
        or run_claim.get("status") != "claimed_once_no_resume"
        or run_claim.get("runner_plan_sha256") != plan_sha
        or run_claim.get("pid") != os.getppid()
    ):
        raise Development300WorkerError("worker is outside the claimed runner process")
    launch = load_json(
        runner_root
        / "stages"
        / f"stage_{global_ordinal:03d}"
        / "launch_receipt.json",
        "stage launch receipt",
    )
    verify_signed(launch, "stage_receipt_sha256", "stage launch receipt")
    if (
        launch.get("status") != "launching_exact_once"
        or launch.get("global_ordinal") != global_ordinal
        or launch.get("command_sha256") != command["command_sha256"]
        or launch.get("retry_allowed") is not False
    ):
        raise Development300WorkerError("stage launch binding changed")
    return {
        "authority": authority,
        "authority_path": authority_path,
        "decoded": decoded,
        "plan_sha256": plan_sha,
        "command": command,
        "identity_row": decoded["identity_authority"]["selected_rows"][global_ordinal],
        "payload": payload,
    }


def collect_one(
    *,
    authority_path: Path,
    expected_authority_file_sha256: str,
    plan_path: Path,
    expected_plan_file_sha256: str,
    global_ordinal: int,
    staging_payload: Path,
) -> dict[str, Any]:
    bound = _load_bound_inputs(
        authority_path=authority_path,
        expected_authority_file_sha256=expected_authority_file_sha256,
        plan_path=plan_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        global_ordinal=global_ordinal,
        staging_payload=staging_payload,
    )
    authority = bound["authority"]
    decoded = bound["decoded"]
    command = bound["command"]
    identity_row = bound["identity_row"]
    payload = bound["payload"]
    payload.mkdir(mode=0o700)
    implementations = authority["implementations"]
    scripts_root = Path(implementations["runner"]["path"]).parent
    sys.path.insert(0, str(scripts_root))
    collector = _load_module(
        Path(implementations["dense_collector"]["path"]),
        "etsf_development300_bound_collector",
    )
    adapter = _load_module(
        Path(implementations["runtime_adapter"]["path"]),
        "etsf_development300_bound_runtime_adapter",
    )
    support = implementations["support_closure"]
    materializer = _load_module(
        Path(support["materialize_smolvla_piper_schema6_reset_contract.py"]["path"]),
        "etsf_development300_bound_materializer",
    )
    pose = _load_module(
        Path(support["etsf_schema6_pose_quality.py"]["path"]),
        "etsf_development300_bound_pose",
    )
    event_spec = load_json(
        Path(authority["event_specification"]["path"]), "event specification"
    )
    bridge_split = authority["split_bridge"][command["split"]]
    runtime_command = {
        "split": bridge_split,
        "requested_seed": command["requested_seed"],
        "outputs": {"seed_root": str(payload)},
    }
    built = adapter.build_runtime(command=runtime_command, event_spec=event_spec)
    if not isinstance(built, Mapping) or set(built) != {
        "runtime",
        "query_fn",
        "max_steps",
        "close",
    }:
        raise Development300WorkerError("runtime adapter interface changed")
    if built["max_steps"] != 200:
        raise Development300WorkerError("runtime horizon changed")
    runtime = built["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "reset",
        "identity_snapshot",
        "task",
        "snapshot",
        "step",
        "derive_events",
    }:
        raise Development300WorkerError("runtime mapping changed")
    reset_count = 0
    canonical_registry: dict[str, Any] | None = None

    def verified_reset(seed: int, instruction: str):
        nonlocal reset_count
        observation, resolved, observed_instruction = runtime["reset"](
            seed, instruction
        )
        if (
            seed != command["requested_seed"]
            or resolved != command["expected_resolved_seed"]
            or observed_instruction != INSTRUCTION
        ):
            raise Development300WorkerError("runtime reset identity changed")
        _validate_identity(runtime["identity_snapshot"](), identity_row)
        if canonical_registry is not None:
            materializer.assert_runtime_registry_identity(
                runtime["task"](), canonical_registry
            )
        reset_count += 1
        return observation, resolved, observed_instruction

    try:
        verified_reset(command["requested_seed"], INSTRUCTION)
        canonical_registry = materializer.build_runtime_object_registry(
            runtime["task"]()
        )
        registry_sha = pose.registry_sha256(canonical_registry)
        pose_spec = materializer.build_pose_quality_spec(
            canonical_registry,
            move_can_pot_source=authority["runtime_contract"][
                "runtime_source_artifacts"
            ]["robotwin_move_can_pot"],
        )
        pose_sha = pose.spec_sha256(
            pose_spec, expected_registry_sha256=registry_sha
        )
        atomic_json(payload / "object_registry.json", canonical_registry)
        atomic_json(payload / "pose_quality_spec.json", pose_spec)
        reset_receipt = signed(
            {
                "format": RESET_RECEIPT_FORMAT,
                "status": "identity_verified_before_first_policy_query",
                "runner_authority_sha256": decoded["runner_authority_sha256"],
                "runner_plan_sha256": bound["plan_sha256"],
                "command_sha256": command["command_sha256"],
                "global_ordinal": global_ordinal,
                "split": command["split"],
                "requested_seed": command["requested_seed"],
                "resolved_seed": command["expected_resolved_seed"],
                "pair_id": command["pair_id"],
                "initial_scene_state_sha256": identity_row[
                    "initial_scene_state_sha256"
                ],
                "initial_measured_joint_state_sha256": identity_row[
                    "initial_measured_joint_state_sha256"
                ],
                "initial_commanded_drive_target_sha256": identity_row[
                    "initial_commanded_drive_target_sha256"
                ],
                "object_registry_sha256": registry_sha,
                "pose_spec_sha256": pose_sha,
                "identity_validation_count_before_policy_query": reset_count,
                "policy_queries_before_reset_receipt": 0,
                "outcome_or_label_read_before_reset_receipt": False,
                "evaluation400": False,
            },
            "reset_receipt_sha256",
        )
        atomic_json(payload / "per_seed_reset_receipt.json", reset_receipt)
        wrapped_runtime = dict(runtime)
        wrapped_runtime["reset"] = verified_reset
        record = collector.collect_dense_group(
            runtime=wrapped_runtime,
            query_fn=built["query_fn"],
            requested_seed=command["requested_seed"],
            instruction=INSTRUCTION,
            object_registry=canonical_registry,
            pose_quality_spec=pose_spec,
            event_spec=event_spec,
            max_steps=200,
        )
        accounting = _candidate_accounting(
            record, command_sha256=command["command_sha256"]
        )
        atomic_json(payload / "candidate_accounting.json", accounting)
        group_path = payload / "schema6_group.hdf5"
        # The label-bearing record remains inside this sealed worker boundary.
        collector.save_schema6_group(group_path, record)
        group_sha = opaque_file_sha256(group_path)
        formal = command["split"] == "formal_target_validation"
        receipt = signed(
            {
                "format": WORKER_GROUP_RECEIPT_FORMAT,
                "status": "complete_exact_four_candidate_accounting",
                "runner_authority_sha256": decoded["runner_authority_sha256"],
                "command_sha256": command["command_sha256"],
                "global_ordinal": global_ordinal,
                "split": command["split"],
                "requested_seed": command["requested_seed"],
                "resolved_seed": command["expected_resolved_seed"],
                "pair_id": command["pair_id"],
                "candidate_original_indices": [0, 1, 2, 3],
                "candidate_accounting_records": 4,
                "candidate_accounting_sha256": accounting[
                    "candidate_accounting_sha256"
                ],
                "per_seed_reset_receipt_sha256": reset_receipt[
                    "reset_receipt_sha256"
                ],
                "object_registry_sha256": registry_sha,
                "pose_spec_sha256": pose_sha,
                "group_file_sha256": group_sha,
                "formal_payload_sealed": formal,
                "outcome_or_label_fields_disclosed_to_runner": False,
                "evaluation400": False,
            },
            GROUP_SIGNATURE,
        )
        atomic_json(payload / "completed_group_receipt.json", receipt)
        _seal_payload(payload, formal=formal)
    finally:
        built["close"]()
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect-one",))
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-file-sha256", required=True)
    parser.add_argument("--static-plan", type=Path, required=True)
    parser.add_argument("--static-plan-file-sha256", required=True)
    parser.add_argument("--global-ordinal", type=int, required=True)
    parser.add_argument("--staging-payload", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = collect_one(
        authority_path=args.authority,
        expected_authority_file_sha256=args.authority_file_sha256,
        plan_path=args.static_plan,
        expected_plan_file_sha256=args.static_plan_file_sha256,
        global_ordinal=args.global_ordinal,
        staging_payload=args.staging_payload,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                GROUP_SIGNATURE: receipt[GROUP_SIGNATURE],
                "outcome_or_label_fields_disclosed_to_runner": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCOUNTING_FORMAT",
    "Development300WorkerError",
    "RESET_RECEIPT_FORMAT",
    "collect_one",
]
