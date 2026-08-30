#!/usr/bin/env python3
"""Materialize a label/outcome-blind common-stable RoboTwin2 seed roster.

The candidate interval is frozen in source and durably preregistered before
the first simulator setup.  A probe performs only fresh ``setup_demo`` and
``close_env`` calls: it never loads an actor, executes an action, evaluates a
task predicate, or reads a rollout label.  The first 20 candidate seeds whose
ten body/condition cells all set up successfully are the only legal actor-v2
evaluation roster.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PREREGISTRATION_FORMAT = "etsf_robotwin2_common_stable_seed_preregistration_v1"
ROSTER_FORMAT = "etsf_robotwin2_common_stable_seed_roster_v1"
PROGRESS_FORMAT = "etsf_robotwin2_common_stable_seed_probe_progress_v1"
TASK = "move_can_pot"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
CELL_ORDER = tuple((body, condition) for body in BODIES for condition in CONDITIONS)
CANDIDATE_SEED_START = 2026104000
CANDIDATE_SEED_STOP_EXCLUSIVE = 2026105000
SELECTED_SEED_COUNT = 20
GPU_UUID = "GPU-06f6e50e-5296-258f-dd86-8f838390a7d1"
DEFAULT_INSTRUCTION = "Move the can to the side of the pot."
BODY_EMBODIMENT = {
    "aloha-agilex": ["aloha-agilex"],
    "arx-x5": ["ARX-X5", "ARX-X5", 0.6],
    "franka": ["franka-panda", "franka-panda", 0.8],
    "piper": ["piper", "piper", 0.6],
    "ur5": ["ur5-wsg", "ur5-wsg", 0.8],
}


class StableSeedRosterError(RuntimeError):
    """The preregistration, reset-only probe, or roster changed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise StableSeedRosterError("authority file may not be symbolic")
    resolved = expanded.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StableSeedRosterError("authority file must be real")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("logical_sha256", None)
    return {**unsigned, "logical_sha256": canonical_sha256(unsigned)}


def _verify_signed(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise StableSeedRosterError(f"{label} logical SHA-256 mismatch")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_once_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise StableSeedRosterError("create-once authority may not be symbolic")
    resolved = expanded.resolve()
    if resolved.exists():
        if not resolved.is_file() or resolved.is_symlink() or resolved.read_bytes() != payload:
            raise StableSeedRosterError("existing create-once authority changed")
        return sha256_file(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".create", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, resolved)
        except FileExistsError:
            if resolved.is_symlink() or not resolved.is_file() or resolved.read_bytes() != payload:
                raise StableSeedRosterError("racing create-once authority changed")
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(resolved)


def preregistration_document() -> dict[str, Any]:
    base = {
        "format": PREREGISTRATION_FORMAT,
        "task": TASK,
        "candidate_seed_start_inclusive": CANDIDATE_SEED_START,
        "candidate_seed_stop_exclusive": CANDIDATE_SEED_STOP_EXCLUSIVE,
        "candidate_seed_order": "strictly_increasing_integer_order",
        "selected_seed_count": SELECTED_SEED_COUNT,
        "body_order": list(BODIES),
        "condition_order": list(CONDITIONS),
        "cell_order": [
            {"body": body, "condition": condition}
            for body, condition in CELL_ORDER
        ],
        "probe_contract": {
            "fresh_setup_demo_only": True,
            "actor_or_policy_loaded": False,
            "actor_inference_called": False,
            "task_action_called": False,
            "check_success_or_event_label_called": False,
            "rollout_outcome_read": False,
            "first_setup_failure_short_circuits_remaining_cells": True,
            "unattempted_cells_recorded_as_short_circuited_not_failures": True,
            "stable_seed_definition": "all_ten_fresh_setup_cells_succeed",
            "selection_rule": "first_twenty_stable_in_candidate_order",
            "selection_uses_labels_or_outcomes": False,
        },
        "disjointness_contract": {
            "prior_failed_actor_v1_seed_block_reused": False,
            "prior_nonformal_probe_2026103000_block_reused": False,
            "fresh_candidate_interval_starts_at": CANDIDATE_SEED_START,
        },
    }
    return _signed(base)


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    _verify_signed(value, "stable-seed preregistration")
    expected = preregistration_document()
    if dict(value) != expected:
        raise StableSeedRosterError("stable-seed preregistration is not frozen")
    return dict(expected)


def _validate_attempt(attempt: Mapping[str, Any], expected_seed: int) -> bool:
    cells = attempt.get("cells")
    if attempt.get("candidate_seed") != expected_seed or not isinstance(cells, list):
        raise StableSeedRosterError("probe attempt order changed")
    if len(cells) != len(CELL_ORDER):
        raise StableSeedRosterError("probe attempt does not account for all ten cells")
    failure_seen = False
    for index, ((body, condition), cell) in enumerate(zip(CELL_ORDER, cells, strict=True)):
        if (
            not isinstance(cell, Mapping)
            or cell.get("body") != body
            or cell.get("condition") != condition
            or cell.get("cell_index") != index
        ):
            raise StableSeedRosterError("probe cell order changed")
        status = cell.get("status")
        if failure_seen:
            if status != "not_attempted_after_first_setup_failure":
                raise StableSeedRosterError("post-failure cells were not label-blind short-circuited")
        elif status == "setup_failed":
            failure_seen = True
            if not isinstance(cell.get("error_type"), str):
                raise StableSeedRosterError("setup failure lacks an error type")
        elif status != "setup_succeeded_and_closed":
            raise StableSeedRosterError("probe cell has an unknown status")
    stable = not failure_seen
    if attempt.get("all_ten_setup_cells_stable") is not stable:
        raise StableSeedRosterError("probe attempt stable flag changed")
    if attempt.get("actor_inference_calls") != 0 or attempt.get("task_action_calls") != 0:
        raise StableSeedRosterError("probe attempt is not reset-only")
    if attempt.get("label_or_outcome_reads") != 0:
        raise StableSeedRosterError("probe attempt read a label/outcome")
    return stable


def validate_stable_seed_roster(
    value: Mapping[str, Any],
    *,
    expected_preregistration_file_sha256: str | None = None,
) -> dict[str, Any]:
    _verify_signed(value, "stable-seed roster")
    preregistration = preregistration_document()
    attempts = value.get("attempts")
    selected = value.get("selected_seeds")
    if (
        value.get("format") != ROSTER_FORMAT
        or value.get("status") != "complete_first_twenty_common_stable_seeds"
        or value.get("task") != TASK
        or value.get("preregistration_logical_sha256")
        != preregistration["logical_sha256"]
        or not isinstance(value.get("preregistration_file_sha256"), str)
        or (
            expected_preregistration_file_sha256 is not None
            and value.get("preregistration_file_sha256")
            != expected_preregistration_file_sha256
        )
        or value.get("body_order") != list(BODIES)
        or value.get("condition_order") != list(CONDITIONS)
        or not isinstance(attempts, list)
        or not isinstance(selected, list)
    ):
        raise StableSeedRosterError("stable-seed roster header changed")
    stable_in_order: list[int] = []
    for offset, attempt in enumerate(attempts):
        seed = CANDIDATE_SEED_START + offset
        if _validate_attempt(attempt, seed):
            stable_in_order.append(seed)
    expected_selected = stable_in_order[:SELECTED_SEED_COUNT]
    if (
        len(expected_selected) != SELECTED_SEED_COUNT
        or selected != expected_selected
        or attempts[-1].get("candidate_seed") != expected_selected[-1]
        or value.get("candidate_attempt_count") != len(attempts)
        or value.get("stable_candidate_count_observed") != len(stable_in_order)
        or value.get("pair_count") != len(BODIES) * len(CONDITIONS) * SELECTED_SEED_COUNT
        or value.get("rollout_count_for_two_methods")
        != len(BODIES) * len(CONDITIONS) * SELECTED_SEED_COUNT * 2
    ):
        raise StableSeedRosterError("stable-seed roster selection changed")
    return dict(value)


def load_stable_seed_roster_file(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_preregistration_file_sha256: str | None = None,
) -> dict[str, Any]:
    if sha256_file(path) != expected_file_sha256:
        raise StableSeedRosterError("stable-seed roster file SHA-256 mismatch")
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StableSeedRosterError("stable-seed roster must be a JSON object")
    return validate_stable_seed_roster(
        value,
        expected_preregistration_file_sha256=(
            expected_preregistration_file_sha256
        ),
    )


def _task_args(robotwin_root: Path, body: str, condition: str) -> dict[str, Any]:
    config_root = robotwin_root / "env_cfg" / "task_config"
    with (config_root / f"demo_{condition}.yml").open("r", encoding="utf-8") as stream:
        args = yaml.safe_load(stream)
    with (config_root / "_embodiment_config.yml").open("r", encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    embodiment = list(BODY_EMBODIMENT[body])
    args["embodiment"] = embodiment

    def robot_file(name: str) -> Path:
        return (robotwin_root / Path(str(registry[name]["file_path"]))).resolve()

    if len(embodiment) == 1:
        left_file = right_file = robot_file(str(embodiment[0]))
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    else:
        left_file = robot_file(str(embodiment[0]))
        right_file = robot_file(str(embodiment[1]))
        args["dual_arm_embodied"] = False
        args["embodiment_dis"] = float(embodiment[2])
        args["embodiment_name"] = f"{embodiment[0]}_{embodiment[1]}"

    def embodiment_config(path: Path) -> dict[str, Any]:
        with (path / "config.yml").open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    # The task YAML may carry a default robot declaration.  Remove it before
    # installing the embodiment-specific pair so the setup call has one
    # unambiguous left/right robot authority rather than a stale duplicate.
    args.pop("left_robot_file", None)
    args.pop("right_robot_file", None)
    args.update(
        {
            "task_name": TASK,
            "task_config": f"demo_{condition}",
            "left_robot_file": str(left_file),
            "right_robot_file": str(right_file),
            "left_embodiment_config": embodiment_config(left_file),
            "right_embodiment_config": embodiment_config(right_file),
            "eval_mode": True,
            "eval_video_log": False,
            "collect_data": False,
            "render_freq": 0,
            "save_data": False,
            "step_lim": 200,
        }
    )
    return args


def probe_one_setup(task_class: Any, args: Mapping[str, Any], seed: int) -> None:
    """Perform exactly one fresh setup and close, without any task query."""

    task = task_class()
    setup_succeeded = False
    try:
        random.seed(seed)
        np.random.seed(seed)
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **dict(args))
        setup_succeeded = True
    finally:
        if setup_succeeded:
            task.close_env(clear_cache=False)
        else:
            # A failed setup may already have allocated a partial scene.  Its
            # best-effort teardown is infrastructure hygiene only and cannot
            # turn the failed cell into a selected seed.
            with contextlib.suppress(Exception):
                task.close_env(clear_cache=False)


def probe_candidate(
    seed: int,
    probe: Callable[[str, str, int], None],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    failed = False
    for cell_index, (body, condition) in enumerate(CELL_ORDER):
        if failed:
            cells.append(
                {
                    "cell_index": cell_index,
                    "body": body,
                    "condition": condition,
                    "status": "not_attempted_after_first_setup_failure",
                    "reason": "label_outcome_blind_short_circuit",
                }
            )
            continue
        try:
            probe(body, condition, seed)
        except Exception as error:
            failed = True
            cells.append(
                {
                    "cell_index": cell_index,
                    "body": body,
                    "condition": condition,
                    "status": "setup_failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        else:
            cells.append(
                {
                    "cell_index": cell_index,
                    "body": body,
                    "condition": condition,
                    "status": "setup_succeeded_and_closed",
                }
            )
    return {
        "candidate_seed": seed,
        "cells": cells,
        "all_ten_setup_cells_stable": not failed,
        "actor_inference_calls": 0,
        "task_action_calls": 0,
        "label_or_outcome_reads": 0,
    }


def _gpu_identity() -> tuple[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise StableSeedRosterError("nvidia-smi failed")
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise StableSeedRosterError("reset-only probe requires exactly one visible GPU")
    name, uuid = [field.strip() for field in rows[0].split(",", 1)]
    if "4090" not in name or uuid != GPU_UUID:
        raise StableSeedRosterError("reset-only probe is not on the authorized RTX 4090")
    return name, uuid


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-roster", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = preregistration_document()
    preregistration_sha = _create_once_json(args.preregistration, preregistration)
    print(
        "STABLE_SEED_PREREGISTRATION="
        + json.dumps(
            {
                "path": str(args.preregistration.expanduser().resolve()),
                "file_sha256": preregistration_sha,
                "logical_sha256": preregistration["logical_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0
    gpu_name, gpu_uuid = _gpu_identity()
    robotwin_root = args.robotwin_root.expanduser().resolve()
    if not robotwin_root.is_dir() or robotwin_root.is_symlink():
        raise StableSeedRosterError("RoboTwin root must be one real directory")
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    sys.path.insert(0, str(robotwin_root))
    module = __import__(f"envs.{TASK}", fromlist=[TASK])
    task_class = getattr(module, TASK)
    arguments = {
        (body, condition): _task_args(robotwin_root, body, condition)
        for body, condition in CELL_ORDER
    }

    progress_path = args.progress.expanduser().resolve()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(progress, dict):
            raise StableSeedRosterError("probe progress must be a JSON object")
        attempts = progress.get("attempts")
        if (
            progress.get("format") != PROGRESS_FORMAT
            or progress.get("preregistration_file_sha256") != preregistration_sha
            or progress.get("preregistration_logical_sha256")
            != preregistration["logical_sha256"]
            or not isinstance(attempts, list)
        ):
            raise StableSeedRosterError("probe progress authority changed")
    else:
        progress = {
            "format": PROGRESS_FORMAT,
            "status": "in_progress_reset_only",
            "preregistration": str(args.preregistration.expanduser().resolve()),
            "preregistration_file_sha256": preregistration_sha,
            "preregistration_logical_sha256": preregistration["logical_sha256"],
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "attempts": [],
        }
        _atomic_json(progress_path, progress)
        attempts = progress["attempts"]

    stable: list[int] = []
    for offset, attempt in enumerate(attempts):
        seed = CANDIDATE_SEED_START + offset
        if _validate_attempt(attempt, seed):
            stable.append(seed)
    if len(stable) > SELECTED_SEED_COUNT:
        raise StableSeedRosterError("progress continued after selecting twenty seeds")

    def live_probe(body: str, condition: str, seed: int) -> None:
        probe_one_setup(task_class, arguments[(body, condition)], seed)

    next_seed = CANDIDATE_SEED_START + len(attempts)
    while len(stable) < SELECTED_SEED_COUNT:
        if next_seed >= CANDIDATE_SEED_STOP_EXCLUSIVE:
            raise StableSeedRosterError("candidate range exhausted before twenty stable seeds")
        attempt = probe_candidate(next_seed, live_probe)
        _validate_attempt(attempt, next_seed)
        attempts.append(attempt)
        if attempt["all_ten_setup_cells_stable"]:
            stable.append(next_seed)
        progress["attempts"] = attempts
        progress["stable_seeds_so_far"] = stable
        progress["next_candidate_seed"] = next_seed + 1
        _atomic_json(progress_path, progress)
        print(
            "RESET_ONLY_SEED_PROBED="
            + json.dumps(
                {
                    "candidate_seed": next_seed,
                    "all_ten_setup_cells_stable": attempt[
                        "all_ten_setup_cells_stable"
                    ],
                    "stable_count": len(stable),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        next_seed += 1

    roster = _signed(
        {
            "format": ROSTER_FORMAT,
            "status": "complete_first_twenty_common_stable_seeds",
            "task": TASK,
            "preregistration": str(args.preregistration.expanduser().resolve()),
            "preregistration_file_sha256": preregistration_sha,
            "preregistration_logical_sha256": preregistration["logical_sha256"],
            "body_order": list(BODIES),
            "condition_order": list(CONDITIONS),
            "candidate_attempt_count": len(attempts),
            "stable_candidate_count_observed": len(stable),
            "selected_seeds": stable,
            "pair_count": len(BODIES) * len(CONDITIONS) * len(stable),
            "rollout_count_for_two_methods": (
                len(BODIES) * len(CONDITIONS) * len(stable) * 2
            ),
            "attempts": attempts,
            "actor_inference_calls": 0,
            "task_action_calls": 0,
            "label_or_outcome_reads": 0,
        }
    )
    validate_stable_seed_roster(
        roster, expected_preregistration_file_sha256=preregistration_sha
    )
    roster_sha = _create_once_json(args.output_roster, roster)
    progress["status"] = "complete_roster_frozen"
    progress["output_roster"] = str(args.output_roster.expanduser().resolve())
    progress["output_roster_file_sha256"] = roster_sha
    progress["output_roster_logical_sha256"] = roster["logical_sha256"]
    _atomic_json(progress_path, progress)
    print(
        "STABLE_SEED_ROSTER_COMPLETE="
        + json.dumps(
            {
                "path": str(args.output_roster.expanduser().resolve()),
                "file_sha256": roster_sha,
                "logical_sha256": roster["logical_sha256"],
                "selected_seeds": stable,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BODIES",
    "CANDIDATE_SEED_START",
    "CANDIDATE_SEED_STOP_EXCLUSIVE",
    "CELL_ORDER",
    "CONDITIONS",
    "ROSTER_FORMAT",
    "SELECTED_SEED_COUNT",
    "StableSeedRosterError",
    "load_stable_seed_roster_file",
    "preregistration_document",
    "probe_candidate",
    "validate_preregistration",
    "validate_stable_seed_roster",
]
