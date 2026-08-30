#!/usr/bin/env python3
"""Execute the preregistered five-body paired RoboTwin2 success study.

For every held-out body, condition and requested seed this runner executes two
real closed-loop rollouts from the same deterministic reset: frozen EE16 actor
candidate zero, and the same actor with the corresponding LOBO shared head
reranking four candidates.  It writes detailed, resumable pair records and, only
after all 1,000 pairs are complete, the strict frozen outcome document consumed
by ``evaluate_robotwin2_cross_embodiment_paired_success_v1.py``.

This is a remote RTX-4090 simulator entry point.  It never opens official expert
archives or protected internal trajectory/label files and performs no training.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import collect_robotwin2_five_body_ee_candidate_branches_v1 as collector
import evaluate_robotwin2_cross_embodiment_paired_success_v1 as evaluator
import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as preregister
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event
import train_robotwin2_five_body_lobo_shared_event_head_v1 as shared_head


FORMAT = "etsf_robotwin2_five_body_paired_success_execution_v1"
PAIR_FORMAT = "etsf_robotwin2_move_can_pot_live_paired_execution_v1"
CONTRACT_FORMAT = "etsf_robotwin2_move_can_pot_live_paired_execution_contract_v1"
OUTCOME_FORMAT = evaluator.INPUT_FORMAT
OUTCOME_STATUS = evaluator.INPUT_STATUS
BENCHMARK = evaluator.BENCHMARK
TASK = evaluator.TASK
BODIES = evaluator.BODIES
CONDITIONS = evaluator.EVALUATION_CONDITIONS
METHODS = evaluator.METHODS
SEED_BASE = evaluator.EVALUATION_SEED_BASE
SEED_COUNT = evaluator.EVALUATION_SEED_COUNT
CANDIDATE_COUNT = 4
NATIVE_EE_DIM = 16
STAGE_DENOMINATOR = 4
ACTOR_DATASET_FPS = 15.0
ACTION_EXEC_STEPS = 5
QUERY_CANONICALIZATION_STEPS = 1
MINIMUM_CANDIDATE_HORIZON = ACTION_EXEC_STEPS
PLANNED_DT_SECONDS = ACTION_EXEC_STEPS / ACTOR_DATASET_FPS
PREREGISTRATION_SHA256 = evaluator.APPROVED_PREREGISTRATION_SHA256
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
SUPPLEMENT_SPLIT_SEED = 20260901
SUPPLEMENT_FOLD_RECEIPT_CONTRACT = {
    "format": "etsf_proper_world_supplement_outer_lobo_split_v1",
    "candidate_count": CANDIDATE_COUNT,
    "groups_per_body": 30,
    "five_body_groups": 150,
    "five_body_rows": 150 * CANDIDATE_COUNT,
    "source_train_bodies": 3,
    "source_train_groups": 90,
    "source_train_rows": 90 * CANDIDATE_COUNT,
    "proper_validation_bodies": 1,
    "proper_validation_split_seed": SUPPLEMENT_SPLIT_SEED,
    "proper_validation_groups": 30,
    "proper_validation_rows": 30 * CANDIDATE_COUNT,
    "outer_heldout_bodies": 1,
    "outer_heldout_groups_deferred": 30,
    "outer_heldout_rows_deferred": 30 * CANDIDATE_COUNT,
    "proper_validation_use": (
        "strict_proper_checkpoint_selection_only_no_rank_selection_"
        "calibration_diagnostics_without_fit"
    ),
    "rank_selection_rows": 0,
    "calibration_fit_rows": 0,
    "heldout_payload_rows_opened": 0,
}
SUPPLEMENT_STRICT_PROPER_SCORE = (
    "primary_strict_proper_plus_fixed_0.25_times_label_blind_"
    "inner_cross_body_supplement_strict_proper"
)
SUPPLEMENT_STRICT_PROPER_SE_COMBINATION = (
    "per_member_sqrt_primary_variance_plus_0.25_squared_times_"
    "supplement_variance_then_conservative_max_across_members"
)
STRICT_PROPER_SELECTION_RULE = (
    "minimize_source_body_condition_macro_proper_score_then_"
    "maximize_rank_within_one_standard_error"
)


class PairedExecutionError(RuntimeError):
    """The actor, fold, reset, paired candidate, or output contract changed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_binding(robotwin_root: Path) -> dict[str, Any]:
    files = [
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(collector) or "").resolve(),
        Path(inspect.getsourcefile(collector.canonical_adapter) or "").resolve(),
        Path(inspect.getsourcefile(shared_head) or "").resolve(),
        Path(inspect.getsourcefile(evaluator) or "").resolve(),
        robotwin_root / "envs" / f"{TASK}.py",
        robotwin_root / "envs" / "_base_task.py",
        robotwin_root / "envs" / "robot" / "robot.py",
        robotwin_root / "env_cfg" / "task_config" / "demo_clean.yml",
        robotwin_root / "env_cfg" / "task_config" / "demo_randomized.yml",
        robotwin_root / "env_cfg" / "task_config" / "_embodiment_config.yml",
    ]
    rows = []
    for path in files:
        if not path.is_file():
            raise PairedExecutionError(f"critical runtime file is missing: {path}")
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    body_configs = {}
    for body in BODIES:
        task_args = collector._load_task_args(robotwin_root, body, "clean")
        paths = sorted(
            {
                str(Path(task_args["left_robot_file"]).resolve() / "config.yml"),
                str(Path(task_args["right_robot_file"]).resolve() / "config.yml"),
            }
        )
        body_configs[body] = [
            {"path": path, "sha256": sha256_file(Path(path))} for path in paths
        ]
    return {
        "critical_files": rows,
        "critical_files_sha256": canonical_sha256(rows),
        "body_config_files": body_configs,
        "body_config_files_sha256": canonical_sha256(body_configs),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "canonical_state_schema": collector.STATE_SCHEMA,
        "canonical_action_schema": collector.ACTION_SCHEMA,
    }


def atomic_json(path: Path, value: Mapping[str, Any], *, frozen: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if frozen:
            temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0float32-le\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_vector(value: Any, *, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise PairedExecutionError(f"{label} must be a nonempty finite vector")
    return array.astype(float).tolist()


def _joint_name(joint: Any, ordinal: int) -> str:
    getter = getattr(joint, "get_name", None)
    if callable(getter):
        value = getter()
    else:
        value = getattr(joint, "name", None)
    return str(value) if value not in (None, "") else f"active_joint_{ordinal}"


def _entity_pose(entity: Any, *, label: str) -> list[float] | None:
    getter = getattr(entity, "get_pose", None)
    pose = getter() if callable(getter) else getattr(entity, "pose", None)
    if pose is None:
        return None
    position = getattr(pose, "p", None)
    quaternion = getattr(pose, "q", None)
    if position is None or quaternion is None:
        raise PairedExecutionError(f"{label} exposed an invalid root pose")
    vector = _finite_vector(
        np.r_[np.asarray(position), np.asarray(quaternion)], label=f"{label} root pose"
    )
    if len(vector) != 7:
        raise PairedExecutionError(f"{label} root pose must be 7-D")
    return vector


def _joint_drive_vector(
    joint: Any, *, getter_name: str, attribute_name: str, label: str
) -> list[float]:
    getter = getattr(joint, getter_name, None)
    if callable(getter):
        value = getter()
    elif hasattr(joint, attribute_name):
        value = getattr(joint, attribute_name)
    else:
        raise PairedExecutionError(f"{label} is not observable")
    return _finite_vector(value, label=label)


def robot_state_snapshot(task: Any) -> dict[str, Any]:
    """Capture every stable public robot-state channel exposed by RoboTwin."""

    robot = getattr(task, "robot", None)
    if robot is None:
        raise PairedExecutionError("RoboTwin task does not expose its robot")
    result: dict[str, Any] = {
        "ee_action16": _finite_vector(
            collector.current_ee_action16(task), label="robot EE state"
        ),
        "commanded_arm_state": {},
        "articulations": {},
    }
    for side in ("left", "right"):
        command_getter = getattr(robot, f"get_{side}_arm_jointState", None)
        if not callable(command_getter):
            raise PairedExecutionError(f"robot lacks {side} commanded arm state")
        result["commanded_arm_state"][side] = _finite_vector(
            command_getter(), label=f"{side} commanded arm state"
        )
        entity = getattr(robot, f"{side}_entity", None)
        if entity is None:
            raise PairedExecutionError(f"robot lacks {side} articulation")
        state: dict[str, Any] = {
            "qpos": _finite_vector(entity.get_qpos(), label=f"{side} qpos"),
            "qvel": _finite_vector(entity.get_qvel(), label=f"{side} qvel"),
            "root_pose": _entity_pose(entity, label=f"{side} articulation"),
            "optional_dynamic_channels": {},
            "unavailable_optional_dynamic_channels": [],
        }
        for channel, method_name in (
            ("qacc", "get_qacc"),
            ("qf", "get_qf"),
        ):
            getter = getattr(entity, method_name, None)
            if callable(getter):
                state["optional_dynamic_channels"][channel] = _finite_vector(
                    getter(), label=f"{side} {channel}"
                )
            else:
                state["unavailable_optional_dynamic_channels"].append(channel)
        joints_getter = getattr(entity, "get_active_joints", None)
        if not callable(joints_getter):
            raise PairedExecutionError(f"{side} articulation lacks active joints")
        active_joints = list(joints_getter())
        joint_rows = []
        for ordinal, joint in enumerate(active_joints):
            joint_rows.append(
                {
                    "ordinal": ordinal,
                    "name": _joint_name(joint, ordinal),
                    "drive_target": _joint_drive_vector(
                        joint,
                        getter_name="get_drive_target",
                        attribute_name="drive_target",
                        label=f"{side} joint {ordinal} drive target",
                    ),
                    "drive_velocity_target": _joint_drive_vector(
                        joint,
                        getter_name="get_drive_velocity_target",
                        attribute_name="drive_velocity_target",
                        label=f"{side} joint {ordinal} drive velocity target",
                    ),
                }
            )
        state["active_joints"] = joint_rows
        result["articulations"][side] = state
    return result


def capture_reset_snapshot(
    task: Any, names: Sequence[str], objects: Sequence[Any]
) -> dict[str, Any]:
    if len(names) != len(objects) or len(set(names)) != len(names):
        raise PairedExecutionError("tracked object registry is invalid")
    tracked = []
    for name, value in zip(names, objects):
        pose = _entity_pose(value, label=f"tracked object {name!r}")
        if pose is None:
            raise PairedExecutionError(f"tracked object {name!r} lacks a pose")
        tracked.append(
            {
                "name": str(name),
                "pose_xyz_wxyz": _finite_vector(
                    pose, label=f"tracked object {name!r} pose"
                ),
            }
        )
    scene = getattr(task, "scene", None)
    if not isinstance(scene, collector.SimulationClockScene):
        raise PairedExecutionError("reset snapshot lacks the physical simulation clock")
    return {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "tracked_objects": tracked,
        "robot_state": robot_state_snapshot(task),
        "simulator_clock": {
            "physical_step_count": int(scene.step_count),
            "sim_seconds": float(scene.sim_seconds),
            "timestep_seconds": float(scene.timestep_seconds),
        },
        "task_counters": {
            "take_action_count": int(getattr(task, "take_action_cnt", 0)),
            "eval_success": bool(getattr(task, "eval_success", False)),
        },
    }


def validate_candidates(candidates: np.ndarray) -> np.ndarray:
    value = np.asarray(candidates, dtype=np.float32)
    if (
        value.ndim != 3
        or value.shape[0] != CANDIDATE_COUNT
        or value.shape[1] < MINIMUM_CANDIDATE_HORIZON
        or value.shape[2] != NATIVE_EE_DIM
        or not np.isfinite(value).all()
    ):
        raise PairedExecutionError(
            "actor candidates must be finite [4,H>=5,16] before commitment"
        )
    return value


def evaluation_schedule() -> list[dict[str, Any]]:
    rows = []
    for body in BODIES:
        for condition in CONDITIONS:
            for ordinal in range(SEED_COUNT):
                rows.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": SEED_BASE + ordinal,
                        "method_order": (
                            list(METHODS)
                            if ordinal % 2 == 0
                            else list(reversed(METHODS))
                        ),
                    }
                )
    if len(rows) != evaluator.EXPECTED_PAIR_COUNT:
        raise PairedExecutionError("paired schedule cardinality changed")
    return rows


def pair_id(body: str, condition: str, seed: int) -> str:
    safe_body = body.replace("/", "_")
    return f"{safe_body}__{condition}__seed_{seed}"


def parse_fold_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        body, separator, raw_path = value.partition("=")
        if not separator or body not in BODIES or not raw_path:
            raise PairedExecutionError("--lobo-fold must be BODY=/absolute/fold/directory")
        if body in result:
            raise PairedExecutionError(f"duplicate LOBO fold for {body}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise PairedExecutionError(f"LOBO fold is not a directory: {path}")
        result[body] = path
    if set(result) != set(BODIES):
        raise PairedExecutionError("exactly one corresponding LOBO fold is required per body")
    return result


def inspect_fold(body: str, fold_root: Path) -> dict[str, Any]:
    summary_path = fold_root / "training_summary.json"
    if not summary_path.is_file():
        raise PairedExecutionError(f"missing fold summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current_event_implementation_sha = sha256_file(
        Path(analytic_event.__file__).resolve()
    )
    current_trainer_sha = sha256_file(Path(shared_head.__file__).resolve())
    ensemble_selection = summary.get("ensemble_checkpoint_selection")
    expected_source_bodies = [candidate for candidate in BODIES if candidate != body]
    if (
        summary.get("format") != shared_head.FORMAT
        or summary.get("status") != "source_only_checkpoint_selection_complete"
        or summary.get("held_out_body") != body
        or summary.get("source_bodies") != expected_source_bodies
        or summary.get("body_adapter")
        != "single_shared_row_zero_heldout_parameters"
        or summary.get("heldout_labels_used_for_normalization_training_or_selection") is not False
        or summary.get("heldout_specific_trainable_parameters") != 0
        or summary.get("actor_frozen") is not True
        or summary.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or summary.get("event_derivation_implementation_sha256")
        != current_event_implementation_sha
        or summary.get("candidate_rank_contract")
        != shared_head.summary_candidate_rank_contract("full")
        or summary.get("event_age_contract") != shared_head.event_age_contract()
        or summary.get("terminal_horizon_contract")
        != shared_head.terminal_horizon_contract()
        or summary.get("ablation") != shared_head.ablation_contract("full")
        or summary.get("trainer_file_sha256") != current_trainer_sha
        or summary.get("rank_supervision_available") is not True
        or summary.get("candidate_rank_parameters_received_direct_supervision")
        is not True
        or summary.get("synthetic_success_labels") != 0
        or summary.get("rank_supervision_mode")
        not in {
            "mixed_success_plus_informative_dense",
            "mixed_success_only",
            "informative_dense_only",
        }
        or not isinstance(ensemble_selection, Mapping)
        or ensemble_selection.get("common_step_required_for_all_five_members")
        is not True
        or ensemble_selection.get("rank_aggregation")
        != shared_head.risk_adjusted_rank_ensemble_contract()
        or not isinstance(ensemble_selection.get("selected_step"), int)
        or ensemble_selection.get("selected_step", 0) <= 0
        or ensemble_selection.get("heldout_rows_used") != 0
    ):
        raise PairedExecutionError(f"LOBO fold summary contract changed for {body}")
    members = summary.get("members")
    if not isinstance(members, list) or len(members) != 5:
        raise PairedExecutionError(f"LOBO fold must contain five members for {body}")
    normalized = []
    identities = set()
    seeds = set()
    for item in members:
        if not isinstance(item, Mapping):
            raise PairedExecutionError("fold member must be an object")
        member = int(item.get("member", -1))
        seed = item.get("seed")
        checkpoint = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        try:
            checkpoint.relative_to(fold_root)
        except ValueError as error:
            raise PairedExecutionError("fold checkpoint escapes the fold directory") from error
        if (
            member in identities
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not checkpoint.is_file()
        ):
            raise PairedExecutionError("fold member identity/checkpoint is invalid")
        observed = sha256_file(checkpoint)
        if observed != item.get("checkpoint_sha256"):
            raise PairedExecutionError("fold member checkpoint SHA-256 mismatch")
        if (
            item.get("best_step") != ensemble_selection["selected_step"]
            or item.get("trainer_file_sha256") != current_trainer_sha
        ):
            raise PairedExecutionError(
                "fold member was not selected by the common deployment ensemble"
            )
        identities.add(member)
        seeds.add(seed)
        normalized.append(
            {
                "member": member,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed,
            }
        )
    if identities != set(range(5)):
        raise PairedExecutionError("fold members must be exactly 0..4")
    if len(seeds) != 5:
        raise PairedExecutionError("fold member seeds must be exactly five unique integers")
    return {
        "heldout_body": body,
        "source_bodies": expected_source_bodies,
        "body_adapter": "single_shared_row_zero_heldout_parameters",
        "fold_root": str(fold_root),
        "training_summary": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": current_event_implementation_sha,
        "trainer_file_sha256": current_trainer_sha,
        "ensemble_common_selection_step": ensemble_selection["selected_step"],
        "members": sorted(normalized, key=lambda row: row["member"]),
    }


def supplement_proper_validation_body(held_out_body: str) -> str:
    """Replay the trainer's frozen label-blind cross-body assignment."""

    if held_out_body not in BODIES:
        raise PairedExecutionError(
            "supplement split received an unknown held-out body"
        )
    ordered = sorted(
        BODIES,
        key=lambda body: hashlib.sha256(
            (
                f"{SUPPLEMENT_SPLIT_SEED}|supplement-crossfit-order|{body}"
            ).encode()
        ).hexdigest(),
    )
    return ordered[(ordered.index(held_out_body) + 1) % len(ordered)]


def validate_augmented_strict_proper_selection(summary: Mapping[str, Any]) -> None:
    """Replay the augmented proper-score/one-SE checkpoint-selection receipt."""

    ensemble = summary.get("ensemble_checkpoint_selection")
    if not isinstance(ensemble, Mapping):
        raise PairedExecutionError("augmented fold lacks ensemble selection evidence")
    records = ensemble.get("evaluated_common_steps")
    selection = ensemble.get("strict_proper_selection")
    if (
        ensemble.get("strict_proper_score") != SUPPLEMENT_STRICT_PROPER_SCORE
        or ensemble.get("supplement_validation_never_used_for_rank_comparison")
        is not True
        or ensemble.get("calibration_diagnostics_only_no_parameter_fit") is not True
        or not isinstance(records, list)
        or not records
        or not isinstance(selection, Mapping)
        or selection.get("rule") != STRICT_PROPER_SELECTION_RULE
        or selection.get("heldout_rows_used") != 0
    ):
        raise PairedExecutionError(
            "augmented fold strict-proper selection contract changed"
        )

    def finite(value: Any, *, nonnegative: bool = False) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (nonnegative and float(value) < 0.0)
        ):
            raise PairedExecutionError(
                "augmented fold strict-proper numeric evidence changed"
            )
        return float(value)

    normalized = []
    observed_steps: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise PairedExecutionError(
                "augmented fold evaluated-step evidence changed"
            )
        step = record.get("step")
        key = record.get("selection_key")
        ranking = record.get("ensemble_candidate_ranking")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or step in observed_steps
            or not isinstance(key, list)
            or len(key) != 7
            or not isinstance(ranking, Mapping)
        ):
            raise PairedExecutionError(
                "augmented fold evaluated-step identity changed"
            )
        observed_steps.add(step)
        key_values = [finite(value) for value in key]
        if key_values[-1] != float(step):
            raise PairedExecutionError(
                "augmented fold selection key does not bind its step"
            )
        primary_score = finite(
            record.get("mean_member_primary_strict_proper_score")
        )
        supplement_score = finite(
            record.get("mean_member_supplement_strict_proper_score")
        )
        weight = finite(record.get("supplement_strict_proper_weight"))
        combined_score = finite(record.get("mean_member_strict_proper_score"))
        primary_se = finite(
            record.get("primary_conservative_strict_proper_standard_error"),
            nonnegative=True,
        )
        supplement_se = finite(
            record.get("supplement_conservative_strict_proper_standard_error"),
            nonnegative=True,
        )
        combined_se = finite(
            record.get("conservative_strict_proper_standard_error"),
            nonnegative=True,
        )
        expected_score = primary_score + 0.25 * supplement_score
        lower_se = max(primary_se, 0.25 * supplement_se)
        upper_se = math.sqrt(primary_se**2 + (0.25 * supplement_se) ** 2)
        if (
            weight != 0.25
            or not math.isclose(
                combined_score, expected_score, rel_tol=1e-9, abs_tol=1e-12
            )
            or combined_se < lower_se - 1e-12
            or combined_se > upper_se + 1e-12
            or record.get("strict_proper_standard_error_combination")
            != SUPPLEMENT_STRICT_PROPER_SE_COMBINATION
        ):
            raise PairedExecutionError(
                "augmented fold combined strict-proper evidence changed"
            )
        diagnostic = finite(record.get("mean_member_diagnostic_multitask_score"))
        normalized.append(
            {
                "record": record,
                "step": step,
                "key": (*key_values[:6], float(step)),
                "score": combined_score,
                "standard_error": combined_se,
                "diagnostic": diagnostic,
            }
        )

    proper_best = min(normalized, key=lambda row: (row["score"], row["step"]))
    threshold = proper_best["score"] + proper_best["standard_error"]
    comparative = selection.get("comparative_validation_evidence")
    if not isinstance(comparative, bool):
        raise PairedExecutionError(
            "augmented fold comparative-selection audit changed"
        )
    eligible = (
        [row for row in normalized if row["score"] <= threshold + 1e-12]
        if comparative
        else [proper_best]
    )
    selected = min(eligible, key=lambda row: row["key"])
    if (
        selection.get("eligible_steps") != [row["step"] for row in eligible]
        or selection.get("selected_step") != selected["step"]
        or ensemble.get("selected_step") != selected["step"]
        or not math.isclose(
            finite(selection.get("best_score")),
            proper_best["score"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not math.isclose(
            finite(selection.get("conservative_one_standard_error")),
            proper_best["standard_error"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not math.isclose(
            finite(selection.get("eligible_threshold")),
            threshold,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not math.isclose(
            finite(selection.get("selected_score")),
            selected["score"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or ensemble.get("selected_key") != list(selected["record"]["selection_key"])
        or ensemble.get("selected_ensemble_candidate_ranking")
        != selected["record"]["ensemble_candidate_ranking"]
        or not math.isclose(
            finite(ensemble.get("selected_mean_member_diagnostic_multitask_score")),
            selected["diagnostic"],
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise PairedExecutionError(
            "augmented fold selected strict-proper audit cannot be replayed"
        )


def inspect_fold_training_regime(
    folds: Mapping[str, Mapping[str, Any]],
    *,
    required_supplement_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind all five folds to one C-only or one exact C+supplement regime."""

    if set(folds) != set(BODIES):
        raise PairedExecutionError("training regime needs exactly five LOBO folds")
    if required_supplement_binding_sha256 is not None and (
        len(required_supplement_binding_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in required_supplement_binding_sha256
        )
    ):
        raise PairedExecutionError(
            "required supplement binding must be a lowercase SHA-256"
        )

    receipt_contract = SUPPLEMENT_FOLD_RECEIPT_CONTRACT
    rows: dict[str, dict[str, Any]] = {}
    regimes: set[tuple[bool, str | None]] = set()
    for body in BODIES:
        summary_path = Path(str(folds[body]["training_summary"]))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        supplement = summary.get("proper_world_supplement")
        if supplement is None:
            enabled = False
            binding_sha = None
            source_groups = 0
            source_rows = 0
            validation_groups = 0
            validation_rows = 0
            validation_body = None
            heldout_groups = 0
            declaration = "legacy_c_only_summary_without_supplement_field"
        elif not isinstance(supplement, Mapping):
            raise PairedExecutionError(
                f"LOBO fold supplement declaration is invalid for {body}"
            )
        else:
            enabled = supplement.get("enabled") is True
            binding_sha = supplement.get("binding_file_sha256")
            source_groups = supplement.get("source_train_groups")
            source_rows = supplement.get("source_train_rows")
            validation_groups = supplement.get("source_validation_groups")
            validation_rows = supplement.get("source_validation_rows")
            validation_body = supplement.get("source_validation_body")
            heldout_groups = supplement.get("heldout_groups_deferred")
            declaration = "explicit_c_plus_supplement" if enabled else "explicit_c_only"
            if enabled:
                validate_augmented_strict_proper_selection(summary)
                if (
                    not isinstance(binding_sha, str)
                    or len(binding_sha) != 64
                    or any(character not in "0123456789abcdef" for character in binding_sha)
                    or supplement.get("proper_loss_weight")
                    != shared_head.SUPPLEMENT_PROPER_LOSS_WEIGHT
                    or supplement.get("rank_loss_weight")
                    != shared_head.SUPPLEMENT_RANK_LOSS_WEIGHT
                    or supplement.get("rank_or_utility_loss_weight")
                    != shared_head.SUPPLEMENT_RANK_LOSS_WEIGHT
                    or supplement.get("usage_contract")
                    != shared_head.SUPPLEMENT_USAGE_CONTRACT
                    or source_groups != receipt_contract["source_train_groups"]
                    or source_rows != receipt_contract["source_train_rows"]
                    or validation_groups
                    != receipt_contract["proper_validation_groups"]
                    or validation_rows
                    != receipt_contract["proper_validation_rows"]
                    or validation_body
                    != supplement_proper_validation_body(body)
                    or supplement.get("source_validation_body_selection")
                    != (
                        "label_blind_sha256_ordered_five_body_cycle_successor_"
                        "derangement"
                    )
                    or supplement.get("source_validation_assignment_uses_labels")
                    is not False
                    or heldout_groups
                    != receipt_contract["outer_heldout_groups_deferred"]
                    or supplement.get("rank_or_utility_rows_used")
                    != receipt_contract["source_train_rows"]
                    or supplement.get(
                        "rank_or_utility_groups_with_real_comparative_supervision",
                        0,
                    )
                    <= 0
                    or supplement.get("semantic_comparative_rows_used") != 0
                    or supplement.get("normalization_rows_used") != 0
                    or supplement.get("baseline_fit_rows_used") != 0
                    or supplement.get("source_validation_rows_used")
                    != receipt_contract["proper_validation_rows"]
                    or supplement.get(
                        "proper_checkpoint_selection_rows_authorized"
                    )
                    != receipt_contract["proper_validation_rows"]
                    or supplement.get("proper_checkpoint_selection_weight")
                    != shared_head.SUPPLEMENT_PROPER_LOSS_WEIGHT
                    or supplement.get("checkpoint_selection_rows_used")
                    != receipt_contract["proper_validation_rows"]
                    or supplement.get("checkpoint_selection_use")
                    != "strict_proper_only_primary_plus_fixed_0.25_supplement"
                    or supplement.get("rank_selection_rows_authorized")
                    != receipt_contract["rank_selection_rows"]
                    or supplement.get("rank_selection_rows_used")
                    != receipt_contract["rank_selection_rows"]
                    or supplement.get("calibration_diagnostic_rows_authorized")
                    != receipt_contract["proper_validation_rows"]
                    or supplement.get("calibration_diagnostic_rows_used")
                    != receipt_contract["proper_validation_rows"]
                    or supplement.get("calibration_rows_used")
                    != receipt_contract["calibration_fit_rows"]
                    or supplement.get("calibration_fit") is not False
                    or supplement.get("proper_validation_primary_reset_overlap")
                    != 0
                    or supplement.get("heldout_group_npz_opened") != 0
                    or supplement.get("heldout_group_payload_bytes_read") != 0
                    or supplement.get("heldout_group_payload_deserialized") != 0
                    or supplement.get("heldout_manifest_file_opened") != 0
                    or supplement.get("heldout_manifest_bytes_read") != 0
                ):
                    raise PairedExecutionError(
                        f"LOBO fold augmented training contract changed for {body}"
                    )
            elif any(
                value not in (None, 0)
                for value in (binding_sha, source_groups, source_rows, heldout_groups)
            ):
                raise PairedExecutionError(
                    f"LOBO fold claims C-only but contains supplement rows for {body}"
                )
            if not enabled:
                binding_sha = None
                source_groups = 0
                source_rows = 0
                validation_groups = 0
                validation_rows = 0
                validation_body = None
                heldout_groups = 0
        regimes.add((enabled, str(binding_sha) if binding_sha is not None else None))
        rows[body] = {
            "declaration": declaration,
            "supplement_enabled": enabled,
            "supplement_binding_file_sha256": binding_sha,
            "source_train_groups": int(source_groups),
            "source_train_rows": int(source_rows),
            "proper_validation_groups": int(validation_groups),
            "proper_validation_rows": int(validation_rows),
            "proper_validation_body": validation_body,
            "proper_checkpoint_selection_rows_used": (
                int(validation_rows) if enabled else 0
            ),
            "rank_selection_rows_used": 0,
            "calibration_fit_rows_used": 0,
            "heldout_groups_deferred": int(heldout_groups),
            "heldout_rows_deferred": int(heldout_groups) * CANDIDATE_COUNT,
            "training_summary_sha256": folds[body]["training_summary_sha256"],
        }
    if len(regimes) != 1:
        raise PairedExecutionError("five LOBO folds mix different training regimes")
    enabled, binding_sha = next(iter(regimes))
    if required_supplement_binding_sha256 is not None and (
        not enabled or binding_sha != required_supplement_binding_sha256
    ):
        raise PairedExecutionError(
            "LOBO folds do not use the explicitly required supplement binding"
        )
    base = {
        "name": "c_plus_expert_root_supplement" if enabled else "c_only",
        "supplement_enabled": enabled,
        "supplement_binding_file_sha256": binding_sha,
        "required_supplement_binding_sha256": required_supplement_binding_sha256,
        "supplement_fold_receipt_contract": (
            receipt_contract if enabled else None
        ),
        "folds": rows,
    }
    return {**base, "regime_sha256": canonical_sha256(base)}


def load_ensemble(
    fold: Mapping[str, Any], device: torch.device
) -> list[shared_head.EffectAlignedSharedEventHead]:
    models = []
    expected_source_bodies = [
        body for body in BODIES if body != fold["heldout_body"]
    ]
    if (
        fold.get("source_bodies") != expected_source_bodies
        or fold.get("body_adapter")
        != "single_shared_row_zero_heldout_parameters"
    ):
        raise PairedExecutionError("LOBO fold source-body roster changed")
    for item in fold["members"]:
        checkpoint = torch.load(
            item["checkpoint"], map_location=device, weights_only=True
        )
        if (
            checkpoint.get("format") != shared_head.FORMAT
            or checkpoint.get("held_out_body") != fold["heldout_body"]
            or checkpoint.get("source_bodies") != expected_source_bodies
            or checkpoint.get("body_adapter")
            != "single_shared_row_zero_heldout_parameters"
            or checkpoint.get("body_to_id_source_only")
            != {body: 0 for body in expected_source_bodies}
            or checkpoint.get("canonical_state_schema")
            != shared_head.CANONICAL_STATE_SCHEMA
            or checkpoint.get("canonical_action_schema")
            != shared_head.CANONICAL_ACTION_SCHEMA
            or checkpoint.get("event_age_contract")
            != shared_head.event_age_contract()
            or checkpoint.get("terminal_horizon_contract")
            != shared_head.terminal_horizon_contract()
            or checkpoint.get("model_family") != shared_head.MODEL_FAMILY
            or checkpoint.get("candidate_rank_contract")
            != shared_head.checkpoint_candidate_rank_contract("full")
            or checkpoint.get("ablation") != shared_head.ablation_contract("full")
            or checkpoint.get("heldout_rows_used_for_training_normalization_or_selection") != 0
            or checkpoint.get("rank_supervision_available") is not True
            or checkpoint.get(
                "candidate_rank_parameters_received_direct_supervision"
            )
            is not True
            or checkpoint.get("synthetic_success_labels") != 0
            or checkpoint.get("rank_supervision_mode")
            not in {
                "mixed_success_plus_informative_dense",
                "mixed_success_only",
                "informative_dense_only",
            }
            or checkpoint.get("action_stem_count") != 1
            or checkpoint.get("member") != item["member"]
            or checkpoint.get("seed") != item["seed"]
            or checkpoint.get("event_spec_sha256") != EVENT_SPEC_SHA256
            or checkpoint.get("event_derivation_implementation_sha256")
            != fold["event_derivation_implementation_sha256"]
            or checkpoint.get("trainer_file_sha256")
            != fold["trainer_file_sha256"]
            or checkpoint.get("ensemble_common_selection_step")
            != fold["ensemble_common_selection_step"]
        ):
            raise PairedExecutionError("LOBO checkpoint contract changed")
        model = shared_head.EffectAlignedSharedEventHead().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
    return models


def select_candidate(scores: Sequence[float]) -> int:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (CANDIDATE_COUNT,) or not np.isfinite(values).all():
        raise PairedExecutionError("candidate scores must be finite length four")
    return int(np.argmax(values))  # NumPy breaks exact ties by lowest index.


def scoring_batch(
    *,
    state: np.ndarray,
    current_ee: np.ndarray,
    candidates: np.ndarray,
    current_event: int,
    event_age_seconds: float,
    remaining_action_budget: int,
    action_exec_steps: int,
    dt: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if state.shape != (collector.STATE_DIM,):
        raise PairedExecutionError("shared critic state must be 27-D")
    candidates = validate_candidates(candidates)
    if action_exec_steps != ACTION_EXEC_STEPS:
        raise PairedExecutionError("formal candidate execution is fixed to five actions")
    if not np.isclose(dt, 1.0 / ACTOR_DATASET_FPS, atol=1e-12, rtol=0.0):
        raise PairedExecutionError("formal actor control interval is fixed to 1/15 second")
    if not 0 <= current_event <= STAGE_DENOMINATOR:
        raise PairedExecutionError("current event id is outside 0..4")
    if not np.isfinite(event_age_seconds) or event_age_seconds < 0.0:
        raise PairedExecutionError("current event age must be finite and non-negative")
    if isinstance(remaining_action_budget, bool) or remaining_action_budget <= 0:
        raise PairedExecutionError("remaining action budget must be a positive integer")
    expected_event_onehot = np.zeros(STAGE_DENOMINATOR + 1, dtype=np.float32)
    expected_event_onehot[current_event] = 1.0
    if not np.array_equal(
        np.asarray(state[18:23], dtype=np.float32), expected_event_onehot
    ):
        raise PairedExecutionError(
            "state event onehot does not match current_event_id"
        )
    effects = np.stack(
        [collector.canonical_action_chunk(current_ee, candidate) for candidate in candidates]
    ).astype(np.float32)
    horizon = effects.shape[1]
    action_mask = np.arange(horizon)[None] < ACTION_EXEC_STEPS
    batch = {
        "state": torch.as_tensor(
            np.repeat(state[None], CANDIDATE_COUNT, axis=0), device=device
        ),
        "actions": torch.as_tensor(effects, device=device),
        "action_mask": torch.as_tensor(
            np.repeat(action_mask, CANDIDATE_COUNT, axis=0), device=device
        ),
        "action_available": torch.ones(CANDIDATE_COUNT, dtype=torch.bool, device=device),
        "action_schema_id": torch.zeros(CANDIDATE_COUNT, dtype=torch.long, device=device),
        "body_id": torch.zeros(CANDIDATE_COUNT, dtype=torch.long, device=device),
        "dt": torch.full(
            (CANDIDATE_COUNT,), PLANNED_DT_SECONDS, dtype=torch.float32, device=device
        ),
        "current_event_id": torch.full(
            (CANDIDATE_COUNT,), current_event, dtype=torch.long, device=device
        ),
        "event_age_seconds": torch.full(
            (CANDIDATE_COUNT,),
            float(event_age_seconds),
            dtype=torch.float32,
            device=device,
        ),
        "remaining_action_budget": torch.full(
            (CANDIDATE_COUNT,),
            float(remaining_action_budget),
            dtype=torch.float32,
            device=device,
        ),
    }
    return batch


@torch.no_grad()
def score_candidates(
    models: Sequence[shared_head.EffectAlignedSharedEventHead],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if len(models) != 5:
        raise PairedExecutionError("candidate scoring requires five LOBO members")
    rank_rows = []
    success_rows = []
    post_rows = []
    next_rows = []
    duration_mean_rows = []
    duration_scale_rows = []
    terminal_event_rows = []
    terminal_goal_mean_rows = []
    terminal_goal_scale_rows = []
    regression_rows = []
    joint_recovery_rows = []
    for model in models:
        output = model(batch)
        rank_rows.append(output["candidate_rank_logit"].detach().cpu().numpy())
        success_rows.append(torch.sigmoid(output["success_logit"]).cpu().numpy())
        post_rows.append(torch.softmax(output["post_event_logits"], -1).cpu().numpy())
        next_rows.append(torch.softmax(output["next_event_logits"], -1).cpu().numpy())
        duration_mean_rows.append(
            output["duration_selected_log_mean"].detach().cpu().numpy()
        )
        duration_scale_rows.append(
            output["duration_selected_log_scale"].detach().cpu().numpy()
        )
        terminal_event_rows.append(
            torch.softmax(output["terminal_event_logits"], -1).cpu().numpy()
        )
        terminal_goal_mean_rows.append(
            output["terminal_goal_progress_mean"].detach().cpu().numpy()
        )
        terminal_goal_scale_rows.append(
            output["terminal_goal_progress_log_scale"].detach().cpu().numpy()
        )
        regression_rows.append(
            output["regression_probability"].detach().cpu().numpy()
        )
        joint_recovery_rows.append(
            output["joint_recovery_probability"].detach().cpu().numpy()
        )
    ranks = np.stack(rank_rows)
    success = np.stack(success_rows)
    post = np.stack(post_rows)
    following = np.stack(next_rows)
    duration_mean = np.stack(duration_mean_rows)
    duration_scale = np.stack(duration_scale_rows)
    terminal_event = np.stack(terminal_event_rows)
    terminal_goal_mean = np.stack(terminal_goal_mean_rows)
    terminal_goal_scale = np.stack(terminal_goal_scale_rows)
    regression = np.stack(regression_rows)
    joint_recovery = np.stack(joint_recovery_rows)
    if not all(
        np.isfinite(value).all()
        for value in (
            ranks, success, post, following, duration_mean, duration_scale,
            terminal_event, terminal_goal_mean, terminal_goal_scale,
            regression, joint_recovery,
        )
    ):
        raise PairedExecutionError("LOBO ensemble produced a non-finite score")
    risk_adjusted = shared_head.aggregate_risk_adjusted_rank_scores(
        torch.as_tensor(ranks)
    ).cpu().numpy()
    raw_candidate_mean = ranks.mean(axis=0)
    raw_candidate_std = ranks.std(axis=0, ddof=0)
    raw_member_candidate_mean = ranks.mean(axis=1)
    raw_member_candidate_std = ranks.std(axis=1, ddof=0)
    selected = select_candidate(risk_adjusted.tolist())
    return {
        "selected_candidate_index": selected,
        "candidate_rank_score_epistemic_lcb_ensemble": risk_adjusted.astype(float).tolist(),
        "candidate_rank_score_mean": raw_candidate_mean.astype(float).tolist(),
        "candidate_rank_score_raw_candidate_population_std": (
            raw_candidate_std.astype(float).tolist()
        ),
        "candidate_rank_score_raw_member_candidate_mean": (
            raw_member_candidate_mean.astype(float).tolist()
        ),
        "candidate_rank_score_raw_member_candidate_population_std": (
            raw_member_candidate_std.astype(float).tolist()
        ),
        "candidate_rank_score_members": ranks.astype(float).tolist(),
        "candidate_success_probability_mean": success.mean(axis=0).astype(float).tolist(),
        "candidate_post_event_probability_mean": post.mean(axis=0).astype(float).tolist(),
        "candidate_next_event_probability_mean": following.mean(axis=0).astype(float).tolist(),
        "candidate_duration_log_mean_members": duration_mean.astype(float).tolist(),
        "candidate_duration_log_scale_members": duration_scale.astype(float).tolist(),
        "candidate_terminal_event_probability_mean": (
            terminal_event.mean(axis=0).astype(float).tolist()
        ),
        "candidate_terminal_goal_progress_mean_members": (
            terminal_goal_mean.astype(float).tolist()
        ),
        "candidate_terminal_goal_progress_log_scale_members": (
            terminal_goal_scale.astype(float).tolist()
        ),
        "candidate_regression_probability_mean": (
            regression.mean(axis=0).astype(float).tolist()
        ),
        "candidate_joint_recovery_probability_mean": (
            joint_recovery.mean(axis=0).astype(float).tolist()
        ),
    }


def stage_progress(events: np.ndarray, success: bool) -> float:
    maximum = int(np.asarray(events, dtype=np.int64).max())
    if success:
        maximum = STAGE_DENOMINATOR
    if not 0 <= maximum <= STAGE_DENOMINATOR:
        raise PairedExecutionError("canonical event id is outside 0..4")
    return maximum / STAGE_DENOMINATOR


def canonical_state_at(
    *,
    trajectory: np.ndarray,
    sim_times: np.ndarray,
    names: Sequence[str],
    ee_action: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, int, float]:
    predicates, events = collector.derive_predicates_and_events(
        trajectory, sim_times, names, False, calibration
    )
    moving_index = list(names).index(str(calibration["moving"]))
    state = collector._state27(
        poses=trajectory,
        names=names,
        step=len(trajectory) - 1,
        initial_moving_position=trajectory[0, moving_index, :3],
        ee_action=ee_action,
        event=int(events[-1]),
        predicates=predicates,
        calibration=calibration,
    )
    return state, int(events[-1]), collector.event_age_seconds(events, sim_times)


def reset_identity(snapshot: Mapping[str, Any]) -> str:
    if snapshot.get("format") != "etsf_robotwin2_observable_reset_snapshot_v2":
        raise PairedExecutionError("reset identity requires the complete v2 snapshot")
    return canonical_sha256(snapshot)


def prepare_initial_commitment(
    *,
    body: str,
    condition: str,
    seed: int,
    task_class: Any,
    task_args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    calibration: Mapping[str, Any],
    instruction: str,
    device: torch.device,
) -> dict[str, Any]:
    """Freeze the reset and ordered initial candidates before either method."""

    required_names = set(analytic_event.REQUIRED_OBJECTS)
    task = collector._new_task(task_class, task_args, seed, instruction)
    try:
        names, objects = collector.discover_pose_objects(task, required_names)
        reset_snapshot = capture_reset_snapshot(task, names, objects)
        task.scene.step()
        canonical_snapshot = capture_reset_snapshot(task, names, objects)
        candidates = validate_candidates(
            collector.generate_candidates(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                instruction=instruction,
                scene_seed=seed,
                query_index=0,
                candidate_count=CANDIDATE_COUNT,
                device=device,
            )
        )
        after = capture_reset_snapshot(task, names, objects)
        if canonical_snapshot != after:
            raise PairedExecutionError(
                "initial candidate generation changed observable simulator state"
            )
        base = {
            "format": "etsf_robotwin2_initial_candidate_commitment_v2",
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "action_exec_steps": ACTION_EXEC_STEPS,
            "planned_dt_seconds": PLANNED_DT_SECONDS,
            "candidate_count": CANDIDATE_COUNT,
            "candidate_horizon": int(candidates.shape[1]),
            "candidate_shape": list(candidates.shape),
            "ordered_candidate_set_sha256": array_sha256(candidates),
            "reset_snapshot": reset_snapshot,
            "reset_identity_sha256": reset_identity(reset_snapshot),
            "canonical_query_snapshot": canonical_snapshot,
            "canonical_query_identity_sha256": reset_identity(canonical_snapshot),
            "query_canonicalization_steps": QUERY_CANONICALIZATION_STEPS,
            "candidate_generation_advanced_simulator": False,
        }
        return {**base, "commitment_sha256": canonical_sha256(base)}
    finally:
        task.close_env(clear_cache=False)


def verify_initial_commitment(
    commitment: Mapping[str, Any],
    *,
    body: str,
    condition: str,
    seed: int,
    reset_snapshot: Mapping[str, Any],
    canonical_query_snapshot: Mapping[str, Any],
    candidates: np.ndarray,
) -> None:
    base = {key: value for key, value in commitment.items() if key != "commitment_sha256"}
    if (
        commitment.get("format") != "etsf_robotwin2_initial_candidate_commitment_v2"
        or commitment.get("heldout_body") != body
        or commitment.get("condition") != condition
        or commitment.get("requested_seed") != seed
        or commitment.get("resolved_seed") != seed
        or commitment.get("action_exec_steps") != ACTION_EXEC_STEPS
        or commitment.get("planned_dt_seconds") != PLANNED_DT_SECONDS
        or commitment.get("candidate_count") != CANDIDATE_COUNT
        or commitment.get("candidate_horizon") != int(candidates.shape[1])
        or commitment.get("candidate_shape") != list(candidates.shape)
        or commitment.get("ordered_candidate_set_sha256") != array_sha256(candidates)
        or commitment.get("reset_snapshot") != reset_snapshot
        or commitment.get("reset_identity_sha256") != reset_identity(reset_snapshot)
        or commitment.get("canonical_query_snapshot") != canonical_query_snapshot
        or commitment.get("canonical_query_identity_sha256")
        != reset_identity(canonical_query_snapshot)
        or commitment.get("query_canonicalization_steps")
        != QUERY_CANONICALIZATION_STEPS
        or commitment.get("candidate_generation_advanced_simulator") is not False
        or commitment.get("commitment_sha256") != canonical_sha256(base)
    ):
        raise PairedExecutionError(
            "method initial reset/candidates do not match the frozen commitment"
        )


def execute_rollout(
    *,
    method: str,
    body: str,
    condition: str,
    seed: int,
    task_class: Any,
    task_args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    ensemble: Sequence[shared_head.EffectAlignedSharedEventHead],
    calibration: Mapping[str, Any],
    initial_commitment: Mapping[str, Any],
    instruction: str,
    action_exec_steps: int,
    max_steps: int,
    dt: float,
    device: torch.device,
) -> dict[str, Any]:
    if action_exec_steps != ACTION_EXEC_STEPS:
        raise PairedExecutionError("formal rollout execution is fixed to five actions")
    if not np.isclose(dt, 1.0 / ACTOR_DATASET_FPS, atol=1e-12, rtol=0.0):
        raise PairedExecutionError("formal rollout control interval is fixed to 1/15 second")
    required_names = {str(calibration["moving"])}
    anchor = str(calibration.get("anchor", "")).strip()
    if anchor:
        required_names.add(anchor)
    task = collector._new_task(task_class, task_args, seed, instruction)
    decisions = []
    try:
        names, objects = collector.discover_pose_objects(task, required_names)
        initial_poses = collector.read_poses(objects)
        initial_ee = collector.current_ee_action16(task)
        initial_snapshot = capture_reset_snapshot(task, names, objects)
        trajectory = [initial_poses]
        sim_times = [collector._sim_time(task)]
        initial_identity = reset_identity(initial_snapshot)
        initial_canonical_snapshot: Mapping[str, Any] | None = None
        query_index = 0
        while not collector._episode_done(task, max_steps):
            # Match the collector's fresh-scene contact-cache reconstruction.
            # The step is applied identically to both paired methods, advances
            # no formal actor action, and is included in event age/trajectory.
            task.scene.step()
            collector._append_physical_observation(
                task, objects, trajectory, sim_times
            )
            current_ee = collector.current_ee_action16(task)
            pre_candidate_snapshot = capture_reset_snapshot(task, names, objects)
            candidates = validate_candidates(
                collector.generate_candidates(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task=task,
                    instruction=instruction,
                    scene_seed=seed,
                    query_index=query_index,
                    candidate_count=CANDIDATE_COUNT,
                    device=device,
                )
            )
            candidate_sha = array_sha256(candidates)
            if query_index == 0:
                post_candidate_snapshot = capture_reset_snapshot(task, names, objects)
                if post_candidate_snapshot != pre_candidate_snapshot:
                    raise PairedExecutionError(
                        "method initial candidate generation changed simulator state"
                    )
                initial_canonical_snapshot = pre_candidate_snapshot
                verify_initial_commitment(
                    initial_commitment,
                    body=body,
                    condition=condition,
                    seed=seed,
                    reset_snapshot=initial_snapshot,
                    canonical_query_snapshot=pre_candidate_snapshot,
                    candidates=candidates,
                )
            current_event_age: float | None = None
            if method == "actor_baseline":
                selected = 0
                score_record = None
            elif method == "etsf_best_of_4":
                trajectory_array = np.stack(trajectory).astype(np.float32)
                state, current_event, current_event_age = canonical_state_at(
                    trajectory=trajectory_array,
                    sim_times=np.asarray(sim_times, dtype=np.float64),
                    names=names,
                    ee_action=current_ee,
                    calibration=calibration,
                )
                score_record = score_candidates(
                    ensemble,
                    scoring_batch(
                        state=state,
                        current_ee=current_ee,
                        candidates=candidates,
                        current_event=current_event,
                        event_age_seconds=current_event_age,
                        remaining_action_budget=max_steps
                        - int(getattr(task, "take_action_cnt", 0)),
                        action_exec_steps=action_exec_steps,
                        dt=dt,
                        device=device,
                    ),
                )
                selected = int(score_record["selected_candidate_index"])
            else:
                raise PairedExecutionError(f"unknown method: {method}")
            executed = 0
            first_chunk_start_seconds = collector._sim_time(task)
            for action in candidates[selected, :action_exec_steps]:
                if collector._episode_done(task, max_steps):
                    break
                # Ordinary CuRobo planning failure is represented by RoboTwin
                # as a valid ``Fail`` plan without raising.  Any Python
                # exception is a simulator/runtime protocol failure and must
                # abort the formal pair rather than become binary failure.
                task.take_action(action, action_type="ee")
                executed += 1
                collector._append_physical_observation(
                    task, objects, trajectory, sim_times
                )
            decisions.append(
                {
                    "query_index": query_index,
                    "candidate_set_sha256": candidate_sha,
                    "candidate_count": CANDIDATE_COUNT,
                    "selected_candidate_index": selected,
                    "executed_action_count": executed,
                    "planned_chunk_seconds": PLANNED_DT_SECONDS,
                    "physical_sim_seconds": (
                        collector._sim_time(task) - first_chunk_start_seconds
                    ),
                    "critic_scores": score_record,
                    "event_age_seconds": (
                        None if score_record is None else float(current_event_age)
                    ),
                }
            )
            query_index += 1
        success = bool(getattr(task, "eval_success", False))
        if not success:
            # A checker exception is a protocol/runtime failure, not evidence
            # of task failure.  Let it reach the immutable no-retry receipt.
            success = bool(task.check_success())
        trajectory_array = np.stack(trajectory).astype(np.float32)
        _predicates, events = collector.derive_predicates_and_events(
            trajectory_array,
            np.asarray(sim_times, dtype=np.float64),
            names,
            success,
            calibration,
        )
        progress = stage_progress(events, success)
        return {
            "method": method,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "initial_reset_identity_sha256": initial_identity,
            "initial_reset_snapshot": initial_snapshot,
            "initial_canonical_query_snapshot": initial_canonical_snapshot,
            "initial_candidate_commitment_sha256": initial_commitment[
                "commitment_sha256"
            ],
            "tracked_object_names": list(names),
            "initial_object_poses": initial_poses.astype(float).tolist(),
            "initial_ee16": initial_ee.astype(float).tolist(),
            "binary_success": int(success),
            "stage_progress": progress,
            "max_event_id": int(events.max()),
            "executed_control_steps": int(getattr(task, "take_action_cnt", 0)),
            "physical_sim_seconds": collector._sim_time(task) - sim_times[0],
            "sim_timestep_seconds": float(task.scene.timestep_seconds),
            "policy_query_count": len(decisions),
            "action_execution_error": None,
            "decisions": decisions,
        }
    finally:
        task.close_env(clear_cache=False)


def _validate_rollout_decisions(
    rollout: Mapping[str, Any], method: str, expected: Mapping[str, Any]
) -> None:
    if (
        rollout.get("method") != method
        or rollout.get("heldout_body") != expected["heldout_body"]
        or rollout.get("condition") != expected["condition"]
        or rollout.get("requested_seed") != expected["requested_seed"]
        or type(rollout.get("binary_success")) is not int
        or rollout["binary_success"] not in (0, 1)
        or type(rollout.get("max_event_id")) is not int
        or not 0 <= rollout["max_event_id"] <= STAGE_DENOMINATOR
        or rollout.get("action_execution_error") is not None
    ):
        raise PairedExecutionError(f"{method} rollout identity/outcome changed")
    expected_progress = (
        1.0
        if rollout["binary_success"] == 1
        else rollout["max_event_id"] / float(STAGE_DENOMINATOR)
    )
    if abs(float(rollout.get("stage_progress", -1.0)) - expected_progress) > 1e-9:
        raise PairedExecutionError(f"{method} stage progress changed")
    decisions = rollout.get("decisions")
    if (
        not isinstance(decisions, list)
        or not decisions
        or rollout.get("policy_query_count") != len(decisions)
    ):
        raise PairedExecutionError(f"{method} decision roster changed")
    for query_index, decision in enumerate(decisions):
        if (
            not isinstance(decision, Mapping)
            or decision.get("query_index") != query_index
            or decision.get("candidate_count") != CANDIDATE_COUNT
            or not isinstance(decision.get("candidate_set_sha256"), str)
            or len(decision["candidate_set_sha256"]) != 64
            or type(decision.get("selected_candidate_index")) is not int
            or not 0 <= decision["selected_candidate_index"] < CANDIDATE_COUNT
        ):
            raise PairedExecutionError(f"{method} decision identity changed")
        scores = decision.get("critic_scores")
        if method == "actor_baseline":
            if (
                decision["selected_candidate_index"] != 0
                or scores is not None
                or decision.get("event_age_seconds") is not None
            ):
                raise PairedExecutionError(
                    "actor baseline must execute candidate zero without critic scores"
                )
            continue
        if not isinstance(scores, Mapping):
            raise PairedExecutionError("ETSF decision lacks five-member critic scores")
        if (
            not isinstance(decision.get("event_age_seconds"), (int, float))
            or isinstance(decision.get("event_age_seconds"), bool)
            or not math.isfinite(float(decision["event_age_seconds"]))
            or float(decision["event_age_seconds"]) < 0.0
        ):
            raise PairedExecutionError("ETSF decision lacks a valid pre-action event age")
        raw = np.asarray(scores.get("candidate_rank_score_members"), dtype=np.float64)
        recorded = np.asarray(
            scores.get("candidate_rank_score_epistemic_lcb_ensemble"),
            dtype=np.float64,
        )
        if raw.shape != (5, CANDIDATE_COUNT) or recorded.shape != (CANDIDATE_COUNT,):
            raise PairedExecutionError("ETSF critic score shape changed")
        if not np.isfinite(raw).all() or not np.isfinite(recorded).all():
            raise PairedExecutionError("ETSF critic score is non-finite")
        recomputed = shared_head.aggregate_risk_adjusted_rank_scores(
            torch.as_tensor(raw)
        ).cpu().numpy()
        raw_mean = raw.mean(axis=0)
        raw_candidate_std = raw.std(axis=0, ddof=0)
        raw_member_mean = raw.mean(axis=1)
        raw_member_std = raw.std(axis=1, ddof=0)
        recorded_arrays = (
            (scores.get("candidate_rank_score_mean"), raw_mean),
            (
                scores.get("candidate_rank_score_raw_candidate_population_std"),
                raw_candidate_std,
            ),
            (scores.get("candidate_rank_score_raw_member_candidate_mean"), raw_member_mean),
            (
                scores.get("candidate_rank_score_raw_member_candidate_population_std"),
                raw_member_std,
            ),
        )
        if not np.allclose(recorded, recomputed, atol=1e-6, rtol=0.0):
            raise PairedExecutionError("ETSF epistemic LCB score cannot be replayed")
        for observed, expected_values in recorded_arrays:
            observed_array = np.asarray(observed, dtype=np.float64)
            if observed_array.shape != expected_values.shape or not np.allclose(
                observed_array, expected_values, atol=1e-6, rtol=0.0
            ):
                raise PairedExecutionError("ETSF raw rank audit cannot be replayed")
        selected = select_candidate(recomputed.tolist())
        if (
            scores.get("selected_candidate_index") != selected
            or decision["selected_candidate_index"] != selected
        ):
            raise PairedExecutionError("ETSF selected candidate disagrees with frozen scorer")


def validate_pair_record(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_execution_contract_sha256: str | None = None,
) -> None:
    contract_sha = value.get("execution_contract_logical_sha256")
    if (
        value.get("format") != PAIR_FORMAT
        or value.get("benchmark") != BENCHMARK
        or value.get("task") != TASK
        or value.get("heldout_body") != expected["heldout_body"]
        or value.get("condition") != expected["condition"]
        or value.get("requested_seed") != expected["requested_seed"]
        or value.get("method_order") != expected["method_order"]
        or value.get("same_resolved_reset") is not True
        or value.get("same_complete_observable_reset_snapshot") is not True
        or value.get("same_canonical_query0_snapshot") is not True
        or value.get("same_initial_candidate_set") is not True
        or not isinstance(value.get("attempt_sha256"), str)
        or not isinstance(value.get("initial_candidate_commitment_sha256"), str)
        or not isinstance(contract_sha, str)
        or len(contract_sha) != 64
        or (
            expected_execution_contract_sha256 is not None
            and contract_sha != expected_execution_contract_sha256
        )
        or value.get("pair_sha256")
        != canonical_sha256({key: item for key, item in value.items() if key != "pair_sha256"})
    ):
        raise PairedExecutionError("existing pair record changed or is not complete")
    rollouts = value.get("rollouts")
    if not isinstance(rollouts, Mapping) or set(rollouts) != set(METHODS):
        raise PairedExecutionError("pair record does not contain both methods")
    for method in METHODS:
        if not isinstance(rollouts[method], Mapping):
            raise PairedExecutionError("pair rollout must be an object")
        _validate_rollout_decisions(rollouts[method], method, expected)
        if (
            rollouts[method].get("initial_candidate_commitment_sha256")
            != value["initial_candidate_commitment_sha256"]
        ):
            raise PairedExecutionError("pair rollout commitment binding changed")


def materialize_pair(
    expected: Mapping[str, Any],
    rollouts: Mapping[str, Mapping[str, Any]],
    *,
    attempt_sha256: str,
    commitment: Mapping[str, Any],
    execution_contract_logical_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(execution_contract_logical_sha256, str)
        or len(execution_contract_logical_sha256) != 64
    ):
        raise PairedExecutionError("pair lacks the frozen execution-contract SHA-256")
    baseline = rollouts["actor_baseline"]
    etsf = rollouts["etsf_best_of_4"]
    commitment_sha = commitment.get("commitment_sha256")
    same_snapshot = bool(
        baseline.get("initial_reset_snapshot")
        == etsf.get("initial_reset_snapshot")
        == commitment.get("reset_snapshot")
    )
    same_canonical_snapshot = bool(
        baseline.get("initial_canonical_query_snapshot")
        == etsf.get("initial_canonical_query_snapshot")
        == commitment.get("canonical_query_snapshot")
    )
    same_reset = bool(
        baseline["tracked_object_names"] == etsf["tracked_object_names"]
        and same_snapshot
        and same_canonical_snapshot
        and baseline["initial_reset_identity_sha256"]
        == etsf["initial_reset_identity_sha256"]
        == commitment.get("reset_identity_sha256")
        and baseline.get("initial_candidate_commitment_sha256")
        == etsf.get("initial_candidate_commitment_sha256")
        == commitment_sha
    )
    if not same_reset:
        raise PairedExecutionError("paired methods did not resolve to the same reset")
    if not baseline["decisions"] or not etsf["decisions"]:
        raise PairedExecutionError("paired methods must each query the actor")
    same_initial_candidates = (
        baseline["decisions"][0]["candidate_set_sha256"]
        == etsf["decisions"][0]["candidate_set_sha256"]
        == commitment.get("ordered_candidate_set_sha256")
    )
    if not same_initial_candidates:
        raise PairedExecutionError("initial ordered four-candidate set changed across methods")
    base = {
        "format": PAIR_FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        **dict(expected),
        "attempt_sha256": attempt_sha256,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "initial_candidate_commitment_sha256": commitment_sha,
        "same_resolved_reset": same_reset,
        "same_complete_observable_reset_snapshot": same_snapshot,
        "same_canonical_query0_snapshot": same_canonical_snapshot,
        "same_initial_candidate_set": same_initial_candidates,
        "discordance": (
            "actor_only"
            if baseline["binary_success"] > etsf["binary_success"]
            else "etsf_only"
            if etsf["binary_success"] > baseline["binary_success"]
            else "concordant_success"
            if baseline["binary_success"] == 1
            else "concordant_failure"
        ),
        "rollouts": dict(rollouts),
    }
    return {**base, "pair_sha256": canonical_sha256(base)}


def outcome_row(pair: Mapping[str, Any]) -> dict[str, Any]:
    baseline = pair["rollouts"]["actor_baseline"]
    etsf = pair["rollouts"]["etsf_best_of_4"]
    return {
        "benchmark": BENCHMARK,
        "task": TASK,
        "heldout_body": pair["heldout_body"],
        "condition": pair["condition"],
        "requested_seed": pair["requested_seed"],
        "method_order": pair["method_order"],
        "pair_sha256": pair["pair_sha256"],
        "actor_baseline_binary_success": baseline["binary_success"],
        "actor_baseline_stage_progress": baseline["stage_progress"],
        "etsf_best_of_4_binary_success": etsf["binary_success"],
        "etsf_best_of_4_stage_progress": etsf["stage_progress"],
    }


def build_outcome_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_contract_logical_sha256: str,
    execution_contract_file_sha256: str,
) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    ordered_pair_sha256s = [str(row["pair_sha256"]) for row in normalized]
    base = {
        "format": OUTCOME_FORMAT,
        "status": OUTCOME_STATUS,
        "rows": normalized,
        "rows_sha256": canonical_sha256(normalized),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "execution_contract_file_sha256": execution_contract_file_sha256,
        "ordered_pair_sha256s_sha256": canonical_sha256(ordered_pair_sha256s),
    }
    document = {**base, "document_sha256": canonical_sha256(base)}
    evaluator.validate_input_document(document)
    return document


def load_preregistration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    receipt = preregister.validate_preregistration(value)
    if receipt["preregistration_sha256"] != PREREGISTRATION_SHA256:
        raise PairedExecutionError("preregistration is not the approved five-body protocol")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--lobo-fold", action="append", required=True,
        help="Repeat exactly five times as heldout-body=/absolute/fold/root.",
    )
    parser.add_argument(
        "--required-supplement-binding-sha256",
        help=(
            "Require every fold to be the C+supplement model trained from this "
            "exact binding; omit only for a C-only evaluation."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-exec-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=float, default=ACTOR_DATASET_FPS)
    parser.add_argument("--instruction", default=collector.DEFAULT_INSTRUCTION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise PairedExecutionError("formal paired execution requires remote RTX 4090 CUDA")
    if args.action_exec_steps != ACTION_EXEC_STEPS:
        raise PairedExecutionError("formal paired execution fixes action-exec-steps=5")
    if args.max_steps != shared_head.TERMINAL_HORIZON_CONTRACT[
        "formal_episode_action_steps"
    ]:
        raise PairedExecutionError("formal paired execution fixes max-steps=200")
    if args.fps <= 0:
        raise PairedExecutionError("fps must be positive")
    if args.fps != ACTOR_DATASET_FPS:
        raise PairedExecutionError(
            "formal EE16 actor timing is fixed to its 15 Hz training dataset"
        )
    if args.instruction != collector.DEFAULT_INSTRUCTION:
        raise PairedExecutionError("formal paired execution fixes the actor instruction")
    inputs = (
        args.actor_checkpoint, args.vlm_metadata_path, args.robotwin_root,
        args.event_spec, args.preregistration,
    )
    if any(not path.expanduser().resolve().exists() for path in inputs):
        raise FileNotFoundError("one or more required public/static inputs are missing")

    random.seed(20260900)
    np.random.seed(20260900)
    torch.manual_seed(20260900)
    robotwin_root = args.robotwin_root.expanduser().resolve()
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(robotwin_root))

    preregistration_receipt = load_preregistration(args.preregistration.resolve())
    fold_paths = parse_fold_specs(args.lobo_fold)
    folds = {body: inspect_fold(body, fold_paths[body]) for body in BODIES}
    fold_training_regime = inspect_fold_training_regime(
        folds,
        required_supplement_binding_sha256=(
            args.required_supplement_binding_sha256
        ),
    )
    actor_checkpoint = args.actor_checkpoint.expanduser().resolve()
    actor_tree_sha, actor_file_count, actor_size = shared_head.sha256_tree(actor_checkpoint)
    vlm_metadata = args.vlm_metadata_path.expanduser().resolve()
    vlm_tree_sha, vlm_file_count, vlm_size = shared_head.sha256_tree(vlm_metadata)
    event_spec_path = args.event_spec.expanduser().resolve()
    if sha256_file(event_spec_path) != EVENT_SPEC_SHA256:
        raise PairedExecutionError(
            "event specification SHA differs from formal shared-head training"
        )
    try:
        _event_spec, calibration = analytic_event.load_event_spec(event_spec_path)
    except analytic_event.AnalyticEventSpecError as error:
        raise PairedExecutionError(str(error)) from error
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs_dir = output / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    attempts_dir = output / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    commitments_dir = output / "initial_commitments"
    commitments_dir.mkdir(exist_ok=True)
    failures_dir = output / "failures"
    failures_dir.mkdir(exist_ok=True)
    outcome_path = output / "paired_outcomes.json"
    contract_path = output / "execution_contract.json"
    contract_base = {
        "format": CONTRACT_FORMAT,
        "runner_format": FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        "bodies": list(BODIES),
        "conditions": list(CONDITIONS),
        "evaluation_seed_base": SEED_BASE,
        "evaluation_seed_count": SEED_COUNT,
        "pair_count": evaluator.EXPECTED_PAIR_COUNT,
        "rollout_count": evaluator.EXPECTED_PAIR_COUNT * 2,
        "candidate_count": CANDIDATE_COUNT,
        "actor_checkpoint": str(actor_checkpoint),
        "actor_checkpoint_tree_sha256": actor_tree_sha,
        "actor_checkpoint_file_count": actor_file_count,
        "actor_checkpoint_size_bytes": actor_size,
        "vlm_metadata_path": str(vlm_metadata),
        "vlm_metadata_tree_sha256": vlm_tree_sha,
        "vlm_metadata_file_count": vlm_file_count,
        "vlm_metadata_size_bytes": vlm_size,
        "folds": folds,
        "fold_training_regime": fold_training_regime,
        "candidate_rank_ensemble_contract": (
            shared_head.risk_adjusted_rank_ensemble_contract()
        ),
        "event_spec": str(event_spec_path),
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": sha256_file(
            Path(analytic_event.__file__).resolve()
        ),
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "training_and_online_event_implementation_identical": True,
        "state27_relative_goal_contract": (
            "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
            "event_labels_and_online_state27_channels_0_2"
        ),
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": preregistration_receipt["preregistration_sha256"],
        "action_exec_steps": ACTION_EXEC_STEPS,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "actor_training_dataset_fps": ACTOR_DATASET_FPS,
        "planned_first_chunk_seconds": PLANNED_DT_SECONDS,
        "query_canonicalization": {
            "raw_scene_steps_before_each_candidate_generation": (
                QUERY_CANONICALIZATION_STEPS
            ),
            "formal_action_count_advanced": False,
            "physical_time_and_event_age_advanced": True,
            "same_as_training_fresh_scene_branch_root": True,
        },
        "event_age_contract": shared_head.event_age_contract(),
        "instruction": args.instruction,
        "runtime_binding": implementation_binding(robotwin_root),
        "no_training": True,
        "official_expert_or_protected_internal_payloads_opened": False,
    }
    contract = {**contract_base, "logical_sha256": canonical_sha256(contract_base)}
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise PairedExecutionError("existing execution contract differs")
    else:
        atomic_json(contract_path, contract, frozen=True)
    contract_file_sha256 = sha256_file(contract_path)

    from envs import CONFIGS_PATH  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    module = __import__(f"envs.{TASK}", fromlist=[TASK])
    task_class = getattr(module, TASK)
    device = torch.device("cuda:0")
    actor_config = PreTrainedConfig.from_pretrained(
        actor_checkpoint, local_files_only=True
    )
    actor_config.device = str(device)
    actor_config.vlm_model_name = str(args.vlm_metadata_path.resolve())
    actor_config.load_vlm_weights = False
    if (
        actor_config.action_feature is None
        or int(actor_config.action_feature.shape[0]) != NATIVE_EE_DIM
        or actor_config.input_features.get("observation.state") is None
        or int(actor_config.input_features["observation.state"].shape[0])
        != NATIVE_EE_DIM
    ):
        raise PairedExecutionError("frozen actor is not state16/action16 EE")
    policy = SmolVLAPolicy.from_pretrained(
        actor_checkpoint, config=actor_config, local_files_only=True, strict=True
    ).eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=actor_config,
        pretrained_path=str(actor_checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_metadata_path.resolve())},
        },
    )

    schedule = evaluation_schedule()
    rows = []
    completed = 0
    active_body = None
    ensemble: list[shared_head.EffectAlignedSharedEventHead] = []
    dt = 1.0 / args.fps
    started = time.time()
    for expected in schedule:
        body = str(expected["heldout_body"])
        identity = pair_id(body, expected["condition"], expected["requested_seed"])
        if body != active_body:
            del ensemble
            gc.collect()
            torch.cuda.empty_cache()
            ensemble = load_ensemble(folds[body], device)
            active_body = body
        path = pairs_dir / f"{identity}.json"
        if path.exists():
            pair = json.loads(path.read_text(encoding="utf-8"))
            validate_pair_record(pair, expected, contract["logical_sha256"])
        else:
            attempt_path = attempts_dir / f"{identity}.json"
            commitment_path = commitments_dir / f"{identity}.json"
            failure_path = failures_dir / f"{identity}.json"
            if attempt_path.exists() or commitment_path.exists() or failure_path.exists():
                raise PairedExecutionError(
                    "an earlier incomplete/failed pair attempt exists; formal execution "
                    "will not silently rerun it"
                )
            attempt_base = {
                "format": "etsf_robotwin2_paired_attempt_v1",
                "status": "started_once_no_automatic_retry",
                "pair_id": identity,
                **dict(expected),
                "execution_contract_logical_sha256": contract["logical_sha256"],
                "attempt_number": 1,
            }
            attempt_sha = canonical_sha256(attempt_base)
            atomic_json(
                attempt_path,
                {**attempt_base, "attempt_sha256": attempt_sha},
                frozen=True,
            )
            task_args = collector._load_task_args(
                robotwin_root, body, str(expected["condition"])
            )
            task_args["step_lim"] = args.max_steps
            try:
                commitment = prepare_initial_commitment(
                    body=body,
                    condition=str(expected["condition"]),
                    seed=int(expected["requested_seed"]),
                    task_class=task_class,
                    task_args=task_args,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    calibration=calibration,
                    instruction=args.instruction,
                    device=device,
                )
                atomic_json(commitment_path, commitment, frozen=True)
                rollouts = {}
                for method in expected["method_order"]:
                    rollouts[method] = execute_rollout(
                        method=method,
                        body=body,
                        condition=str(expected["condition"]),
                        seed=int(expected["requested_seed"]),
                        task_class=task_class,
                        task_args=task_args,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        ensemble=ensemble,
                        calibration=calibration,
                        initial_commitment=commitment,
                        instruction=args.instruction,
                        action_exec_steps=ACTION_EXEC_STEPS,
                        max_steps=args.max_steps,
                        dt=dt,
                        device=device,
                    )
                pair = materialize_pair(
                    expected,
                    rollouts,
                    attempt_sha256=attempt_sha,
                    commitment=commitment,
                    execution_contract_logical_sha256=contract["logical_sha256"],
                )
                validate_pair_record(pair, expected, contract["logical_sha256"])
                atomic_json(path, pair, frozen=True)
            except Exception as error:
                failure_base = {
                    "format": "etsf_robotwin2_paired_attempt_failure_v1",
                    "status": "failed_no_automatic_retry",
                    "pair_id": identity,
                    "attempt_sha256": attempt_sha,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                if not failure_path.exists():
                    atomic_json(
                        failure_path,
                        {
                            **failure_base,
                            "failure_sha256": canonical_sha256(failure_base),
                        },
                        frozen=True,
                    )
                raise
        rows.append(outcome_row(pair))
        completed += 1
        atomic_json(
            output / "progress.json",
            {
                "format": FORMAT,
                "status": "running" if completed < len(schedule) else "rollouts_complete",
                "completed_pairs": completed,
                "completed_rollouts": completed * 2,
                "total_pairs": len(schedule),
                "last_pair": identity,
                "wall_seconds": time.time() - started,
            },
        )
        print(
            "PAIR_COMPLETE="
            + json.dumps(
                {
                    "completed": completed,
                    "total": len(schedule),
                    "heldout_body": body,
                    "condition": expected["condition"],
                    "requested_seed": expected["requested_seed"],
                    "discordance": pair["discordance"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    document = build_outcome_document(
        rows,
        execution_contract_logical_sha256=contract["logical_sha256"],
        execution_contract_file_sha256=contract_file_sha256,
    )
    if outcome_path.exists():
        existing = json.loads(outcome_path.read_text(encoding="utf-8"))
        if existing != document:
            raise PairedExecutionError("existing frozen outcome document differs")
    else:
        atomic_json(outcome_path, document, frozen=True)
    outcome_file_sha = sha256_file(outcome_path)
    completion_path = output / "paired_execution_completion_receipt.json"
    completion_base = {
        "format": "etsf_robotwin2_paired_execution_completion_receipt_v1",
        "status": "complete_1000_pairs_2000_rollouts_frozen",
        "execution_contract_path": str(contract_path),
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": contract_file_sha256,
        "candidate_rank_ensemble_contract": (
            shared_head.risk_adjusted_rank_ensemble_contract()
        ),
        "pair_count": len(rows),
        "rollout_count": len(rows) * 2,
        "ordered_pair_sha256s_sha256": document[
            "ordered_pair_sha256s_sha256"
        ],
        "outcome_path": str(outcome_path),
        "outcome_document_sha256": document["document_sha256"],
        "outcome_file_sha256": outcome_file_sha,
    }
    completion = {
        **completion_base,
        "logical_sha256": canonical_sha256(completion_base),
    }
    if completion_path.exists():
        if json.loads(completion_path.read_text(encoding="utf-8")) != completion:
            raise PairedExecutionError("existing paired completion receipt differs")
    else:
        atomic_json(completion_path, completion, frozen=True)
    print(
        "PAIRED_EXECUTION_COMPLETE="
        + json.dumps(
            {
                "pairs": len(rows),
                "rollouts": len(rows) * 2,
                "outcome": str(outcome_path),
                "outcome_file_sha256": outcome_file_sha,
                "completion_receipt": str(completion_path),
                "completion_receipt_file_sha256": sha256_file(completion_path),
                "evaluator_command": (
                    "python3 scripts/evaluate_robotwin2_cross_embodiment_paired_success_v1.py "
                    f"--input {outcome_path} --input-file-sha256 {outcome_file_sha} "
                    f"--output {output / 'paired_success_report.json'}"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BODIES", "CONDITIONS", "FORMAT", "METHODS", "PAIR_FORMAT",
    "PairedExecutionError", "array_sha256", "build_outcome_document",
    "evaluation_schedule", "materialize_pair", "outcome_row", "pair_id",
    "parse_fold_specs", "reset_identity", "select_candidate", "stage_progress",
]
