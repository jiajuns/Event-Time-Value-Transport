#!/usr/bin/env python3
"""Wait for the universal EE16 actor and collect the formal 8,000 branches.

This remote-only continuation watcher is deliberately effect-oriented: once
the public-data actor training watcher reports a real completed checkpoint, it
freezes that exact state16/action16 actor into a signed five-body authority,
collects five complete decisions at every one of the 40 online query budgets
for each body/condition stratum, and
materializes the strict LOBO training binding.  It never opens protected
internal HDF/label payloads and it performs no local training.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event

FORMAT = "etsf_robotwin2_ee16_actor_to_five_body_branches_watcher_v4_root_pose_roundtrip"
ACTOR_FORMAT = "etsf_robotwin2_frozen_native_actor_authority_v1"
BINDING_FORMAT = "etsf_robotwin2_five_body_lobo_training_binding_v1"
MANIFEST_FORMAT = "etsf_robotwin2_canonical_transition_manifest_v1"
RECEIPT_FORMAT = "etsf_robotwin2_five_body_complete_branch_collection_receipt_v1"
DATASET_REPO = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
TASK = "move_can_pot"
DEFAULT_INSTRUCTION = "Move the can to the side of the pot."
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
COLLECTION_PRIORITY = ("piper", "arx-x5", "ur5", "aloha-agilex", "franka")
CONDITIONS = ("clean", "randomized")
ROOT_QUERIES = tuple(range(40))
QUERY_BLOCK_SIZE = 4
CANDIDATE_COUNT = 4
TARGET_PER_CONDITION_QUERY = 5
BASE_SEED_START = 2026082000
BASE_SEED_COUNT = 50
SUPPLEMENTAL_SEED_START = BASE_SEED_START + BASE_SEED_COUNT
FORMAL_EVALUATION_SEED_START = 2026090000
ACTION_EXEC_STEPS = 5
MAX_STEPS = 200
ROOT_POSE_RESTORE_ATOL = 2.384185791015625e-7
EXPECTED_GROUPS_PER_BODY = (
    len(CONDITIONS) * len(ROOT_QUERIES) * TARGET_PER_CONDITION_QUERY
)
EXPECTED_BRANCHES_PER_BODY = EXPECTED_GROUPS_PER_BODY * CANDIDATE_COUNT
EXPECTED_TOTAL_BRANCHES = EXPECTED_BRANCHES_PER_BODY * len(BODIES)
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
DIAGNOSTIC_FORMAT = "etsf_robotwin2_candidate_branch_diagnostics_v1"
OBJECT_EFFECT_SCHEMA = {
    "format": "etsf_robotwin2_moving_object_se3_effect_6d_v1",
    "channels": [
        "moving_delta_x",
        "moving_delta_y",
        "moving_delta_z",
        "moving_delta_axis_angle_x",
        "moving_delta_axis_angle_y",
        "moving_delta_axis_angle_z",
    ],
    "rotation": "q_post_times_conjugate_q_root_shortest_axis_angle_wxyz",
    "redundant_relative_goal_delta_removed": True,
}
CANDIDATE_NOISE_CONTRACT = {
    "distribution": "antithetic_standard_normal_pairs_each_marginal_N_0_I",
    "candidate_indices": [0, 1, 2, 3],
    "base_noise_indices": [0, 0, 2, 2],
    "signs": [1, -1, 1, -1],
    "candidate_zero_legacy_noise_unchanged": True,
}
TERMINAL_SUPERVISION_CONTRACT = {
    "terminal_max_event_id": (
        "maximum_canonical_event_from_candidate_root_through_continuation"
    ),
    "terminal_event_mask": "finite_horizon_terminal_event_is_valid",
    "terminal_stage_progress": "one_if_success_else_terminal_max_event_id_div_4",
    "terminal_goal_distance": "euclidean_goal_residual_at_full_continuation_terminal",
    "terminal_goal_progress": "root_goal_distance_minus_terminal_goal_distance",
    "terminal_goal_progress_mask": "finite_horizon_terminal_goal_is_valid",
    "terminal_stop_reason_id": {
        "success": 0,
        "formal_action_limit": 1,
    },
    "planner_status_failure_without_exception": "valid_finite_horizon_outcome",
    "action_execution_exception": "invalidate_complete_four_candidate_decision",
    "same_stage_progress_definition_as_formal_paired_runner": True,
}
EVENT_AGE_CONTRACT = {
    "array": "event_age_seconds",
    "semantics": "elapsed_physical_seconds_since_current_canonical_event_entry",
    "clock_source": "counted_successful_sapien_scene_step_calls",
    "available_before_candidate_execution": True,
    "same_value_for_all_candidates_at_one_root": True,
}
TERMINAL_HORIZON_CONTRACT = {
    "array": "remaining_action_budget",
    "semantics": "max_episode_action_steps_minus_pre_action_take_action_count",
    "available_before_candidate_execution": True,
    "same_value_for_all_candidates_at_one_root": True,
    "conditions_only_terminal_consequence_heads": True,
    "direct_rank_path": False,
    "formal_episode_action_steps": MAX_STEPS,
    "formal_actor_query_stride_actions": ACTION_EXEC_STEPS,
    "development_remaining_action_budgets": list(
        range(MAX_STEPS, 0, -ACTION_EXEC_STEPS)
    ),
}
BRANCH_ROOT_SNAPSHOT_CONTRACT = {
    "format": "etsf_sapien_explicit_fresh_scene_branch_root_v2_float32_roundtrip",
    "physics_state": "keyed_rigid_articulation_drive_task_render_rng_snapshot",
    "candidate_scene_isolation": "one_fresh_scene_per_candidate",
    "contact_cache_reconstruction": "one_counted_raw_scene_step",
    "derived_articulation_qacc": (
        "recorded_for_provenance_not_required_pre_step_then_recomputed_and_"
        "strictly_hashed_after_canonicalization_step"
    ),
    "precanonical_restore_exact_except_articulation_root_pose_float32_roundtrip": True,
    "articulation_root_pose_component_atol": ROOT_POSE_RESTORE_ATOL,
    "articulation_root_pose_component_rtol": 0.0,
    "all_non_root_pose_restorable_fields_bit_exact": True,
    "post_canonicalization_full_snapshot_bit_exact": True,
    "simulation_clock_restored": True,
    "task_counters_restored": ["take_action_cnt", "eval_success"],
    "rng_restored": ["python", "numpy", "torch_cpu", "torch_cuda"],
    "reset_and_action_prefix_replay_used_for_candidates": False,
}
BRANCH_DIAGNOSTIC_CONTRACT = {
    "format": DIAGNOSTIC_FORMAT,
    "first_executed": "successful_or_physics_advancing_actions_in_planned_first_chunk",
    "branch_error": "all_false_execution_exception_invalidates_complete_decision",
    "candidate_action_pairwise_rms": (
        "symmetric_raw_canonical_effect_rms_over_planned_first_five_actions"
    ),
}

HOME_ROOT = Path("/home/user")
UPSTREAM_STATE = HOME_ROOT / (
    "smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830."
    "watcher_state.json"
)
ACTOR_CHECKPOINT = HOME_ROOT / (
    "etsf_smolvla_models/"
    "smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830/"
    "checkpoints/020000/pretrained_model"
)
OUTPUT_ROOT = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip"
)
WATCHER_STATE = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip."
    "watcher_state.json"
)
WATCHER_PID = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip."
    "watcher.pid"
)
WATCHER_LOG = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip."
    "watcher.log"
)
ACTOR_AUTHORITY = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee16_actor_authority_full8000_20260830_v4_root_pose_roundtrip.json"
)
TRAINING_BINDING = HOME_ROOT / (
    "etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip."
    "binding.json"
)
MATERIALIZATION_RECEIPT = HOME_ROOT / (
    "public_benchmark_receipts/"
    "robotwin2_move_can_pot_5emb_materialization_a967b852_20260830_v1.json"
)
ROBOTWIN_ROOT = HOME_ROOT / "etsf_stage0/RoboTwin"
VLM_METADATA = HOME_ROOT / "etsf_stage0/offline_assets/smolvlm2_500m_metadata"
EVENT_SPEC = HOME_ROOT / "etsf_robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json"
REMOTE_PYTHON = HOME_ROOT / "anaconda3/envs/RoboTwin2/bin/python"
LEROBOT_ROOT = HOME_ROOT / "etsf_stage0/lerobot"
LEROBOT_SITE = HOME_ROOT / (
    "etsf_stage0/.venv_lerobot_smolvla_v044/lib/python3.10/site-packages"
)
ETSF_SITE = HOME_ROOT / (
    "anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages"
)
_STATE_WRITE_LOCK = threading.Lock()
MAX_CONSECUTIVE_SUPPLEMENTAL_FAILURES = 8


class ContinuationError(RuntimeError):
    """A frozen actor, collection, or binding contract was violated."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    root = path.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContinuationError(f"tree must be a real directory: {root}")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ContinuationError(f"tree contains a symbolic link: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ContinuationError(f"tree contains a special file: {candidate}")
        rows.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise ContinuationError(f"tree is empty: {root}")
    return canonical_sha256(rows), len(rows), sum(row["size_bytes"] for row in rows)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_state(status: str, **extra: Any) -> None:
    payload = {
        "format": FORMAT,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_state": str(UPSTREAM_STATE),
        "actor_checkpoint": str(ACTOR_CHECKPOINT),
        "output_root": str(OUTPUT_ROOT),
        "actor_authority": str(ACTOR_AUTHORITY),
        "training_binding": str(TRAINING_BINDING),
        "watcher_log": str(WATCHER_LOG),
        "expected_complete_decisions": EXPECTED_TOTAL_BRANCHES // CANDIDATE_COUNT,
        "expected_candidate_branches": EXPECTED_TOTAL_BRANCHES,
        **extra,
    }
    with _STATE_WRITE_LOCK:
        atomic_json(WATCHER_STATE, payload)


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["logical_sha256"] = canonical_sha256(result)
    return result


def write_static(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != value:
            raise ContinuationError(f"existing static artifact changed: {path}")
    else:
        atomic_json(path, value)
        path.chmod(0o444)
    return sha256_file(path)


def relative_to_home(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(HOME_ROOT).as_posix()
    except ValueError as error:
        raise ContinuationError(f"authority path escapes /home/user: {resolved}") from error


def feature_shape(config: Mapping[str, Any], lane: str, key: str) -> list[int] | None:
    features = config.get(lane)
    value = features.get(key) if isinstance(features, Mapping) else None
    shape = value.get("shape") if isinstance(value, Mapping) else None
    return list(shape) if isinstance(shape, list) else None


def wait_for_frozen_actor() -> tuple[dict[str, Any], str]:
    last_status: str | None = None
    while True:
        if not UPSTREAM_STATE.is_file():
            status = "missing"
            upstream: dict[str, Any] = {}
        else:
            try:
                upstream = json.loads(UPSTREAM_STATE.read_text(encoding="utf-8"))
                status = str(upstream.get("status", "unknown"))
            except (OSError, json.JSONDecodeError):
                status = "being_replaced"
                upstream = {}
        if status != last_status:
            print(f"UPSTREAM_ACTOR_STATUS={status}", flush=True)
            write_state("waiting_for_actor", upstream_status=status)
            last_status = status
        if status in {"failed", "training_failed", "conversion_failed"}:
            raise ContinuationError(f"upstream actor watcher failed: {upstream}")
        if status == "complete":
            if Path(str(upstream.get("final_checkpoint", ""))) != ACTOR_CHECKPOINT:
                raise ContinuationError("upstream completed a different final checkpoint")
            if upstream.get("training_exit_code") != 0:
                raise ContinuationError("upstream completion lacks exit code zero")
            if not ACTOR_CHECKPOINT.is_dir():
                raise ContinuationError("upstream checkpoint directory is missing")
            return upstream, sha256_file(UPSTREAM_STATE)
        time.sleep(30)


def validate_static_inputs() -> dict[str, Any]:
    code_root = Path(__file__).resolve().parent
    collector = code_root / "collect_robotwin2_five_body_ee_candidate_branches_v1.py"
    adapter = code_root / "robotwin2_cross_body_canonical_adapter_v1.py"
    required = (
        collector,
        adapter,
        MATERIALIZATION_RECEIPT,
        ROBOTWIN_ROOT,
        VLM_METADATA,
        EVENT_SPEC,
        REMOTE_PYTHON,
        LEROBOT_ROOT,
        LEROBOT_SITE,
        ETSF_SITE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ContinuationError(f"required public/runtime inputs are missing: {missing}")
    try:
        _event_spec, calibration = analytic_event.load_event_spec(EVENT_SPEC)
    except analytic_event.AnalyticEventSpecError as error:
        raise ContinuationError(str(error)) from error
    event_sha = sha256_file(EVENT_SPEC)
    event_implementation_sha = sha256_file(Path(analytic_event.__file__).resolve())
    vlm_sha, vlm_files, vlm_bytes = sha256_tree(VLM_METADATA)
    return {
        "code_root": code_root,
        "collector": collector,
        "collector_sha256": sha256_file(collector),
        "adapter_sha256": sha256_file(adapter),
        "event_spec_sha256": event_sha,
        "event_derivation_implementation_sha256": event_implementation_sha,
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "vlm_metadata_tree_sha256": vlm_sha,
        "vlm_metadata_file_count": vlm_files,
        "vlm_metadata_size_bytes": vlm_bytes,
        "materialization_receipt_sha256": sha256_file(MATERIALIZATION_RECEIPT),
    }


def freeze_actor_authority(
    upstream: Mapping[str, Any], upstream_sha: str, static: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    config_path = ACTOR_CHECKPOINT / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ContinuationError("final actor lacks a real config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_shape = feature_shape(config, "input_features", "observation.state")
    action_shape = feature_shape(config, "output_features", "action")
    if config.get("type") != "smolvla" or state_shape != [16] or action_shape != [16]:
        raise ContinuationError(
            f"final actor is not strict SmolVLA state16/action16: "
            f"{state_shape}/{action_shape}"
        )
    checkpoint_sha, checkpoint_files, checkpoint_bytes = sha256_tree(ACTOR_CHECKPOINT)
    sampling_contract = {
        "format": "etsf_robotwin2_five_body_fixed_flow_candidate_sampling_v1",
        "frozen_actor_checkpoint_tree_sha256": checkpoint_sha,
        "collector_file_sha256": static["collector_sha256"],
        "canonical_adapter_file_sha256": static["adapter_sha256"],
        "candidate_count": CANDIDATE_COUNT,
        "candidate_indices": list(range(CANDIDATE_COUNT)),
        "candidate_zero_is_actor_baseline": True,
        "same_ordered_candidate_set_for_baseline_and_etsf": True,
        "flow_noise_distribution": (
            "antithetic_standard_normal_pairs_each_candidate_marginal_N_0_I"
        ),
        "flow_noise_seed_expression": (
            "base_index=candidate_index-candidate_index%2; "
            "(20260903 + scene_seed*1000003 + query_index*10007 + "
            "base_index*101) mod (2**63-1); odd candidates negate the draw"
        ),
        "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
        "instruction": DEFAULT_INSTRUCTION,
        "conditions": list(CONDITIONS),
        "root_query_indices": list(ROOT_QUERIES),
        "action_exec_steps": ACTION_EXEC_STEPS,
        "max_policy_action_calls": MAX_STEPS,
        "base_development_seed_start": BASE_SEED_START,
        "base_development_seed_count": BASE_SEED_COUNT,
        "supplemental_seed_rule": (
            "same_condition_and_query_until_5_complete_decisions_per_stratum"
        ),
        "formal_evaluation_seed_lower_bound_inclusive": FORMAL_EVALUATION_SEED_START,
        "critic_dt_semantics": "planned_first_candidate_chunk_seconds",
        "planned_dt_seconds": ACTION_EXEC_STEPS / 15.0,
        "event_duration_time_source": "scene.step_count_times_scene.get_timestep",
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": static[
            "event_derivation_implementation_sha256"
        ],
        "analytic_goal_rule": analytic_event.GOAL_RULE,
        "object_effect_schema": OBJECT_EFFECT_SCHEMA,
        "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
        "event_age_contract": EVENT_AGE_CONTRACT,
        "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
        "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
        "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
        "wall_clock_used_as_physical_time": False,
    }
    sampling_sha = canonical_sha256(sampling_contract)
    common = {
        "family": "smolvla_universal_dual_ee16_five_body_actor",
        "shared_checkpoint_across_all_five_bodies": True,
        "frozen": True,
        "optimizer_updates_allowed": False,
        "checkpoint_path": relative_to_home(ACTOR_CHECKPOINT),
        "checkpoint_kind": "directory_tree",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_tree_file_count": checkpoint_files,
        "checkpoint_tree_size_bytes": checkpoint_bytes,
        "config_path_within_checkpoint": "config.json",
        "config_file_sha256": sha256_file(config_path),
        "policy_type": "smolvla",
        "state_feature": "observation.state",
        "state_shape": state_shape,
        "action_feature": "action",
        "action_shape": action_shape,
        "native_action_semantics": "dual_arm_absolute_ee_xyz_quaternion_wxyz_gripper",
        "sampling_contract_sha256": sampling_sha,
        "candidate_count": CANDIDATE_COUNT,
        "candidate_zero_is_actor_baseline": True,
        "same_ordered_candidate_set_for_baseline_and_etsf": True,
    }
    actors = {body: {**common, "embodiment": body} for body in BODIES}
    authority = signed(
        {
            "format": ACTOR_FORMAT,
            "task": TASK,
            "instruction": DEFAULT_INSTRUCTION,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "public_expert_episode_count": 2750,
            "one_universal_actor_for_all_five_bodies": True,
            "upstream_training_state_path": relative_to_home(UPSTREAM_STATE),
            "upstream_training_state_file_sha256": upstream_sha,
            "upstream_training_status": upstream.get("status"),
            "checkpoint_config_contract": {
                "config_file_sha256": sha256_file(config_path),
                "state_shape": state_shape,
                "action_shape": action_shape,
            },
            "sampling_contract": sampling_contract,
            "actors": actors,
        }
    )
    authority_sha = write_static(ACTOR_AUTHORITY, authority)
    return authority, authority_sha


def collector_environment(static: Mapping[str, Any]) -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = ":".join(
        (
            str(static["code_root"]),
            str(LEROBOT_ROOT / "src"),
            str(LEROBOT_SITE),
            str(ROBOTWIN_ROOT),
            str(ROBOTWIN_ROOT / "envs/curobo/src"),
            str(ETSF_SITE),
        )
    )
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": pythonpath,
            "ASSETS_PATH": str(ROBOTWIN_ROOT),
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def collector_command(
    static: Mapping[str, Any],
    *,
    body: str,
    conditions: Sequence[str],
    seed_start: int,
    seed_count: int,
    queries: Sequence[int],
) -> list[str]:
    return [
        str(REMOTE_PYTHON),
        str(static["collector"]),
        "--body",
        body,
        "--actor-checkpoint",
        str(ACTOR_CHECKPOINT),
        "--vlm-metadata-path",
        str(VLM_METADATA),
        "--robotwin-root",
        str(ROBOTWIN_ROOT),
        "--event-spec",
        str(EVENT_SPEC),
        "--output",
        str(OUTPUT_ROOT / body),
        "--conditions",
        *conditions,
        "--seed-start",
        str(seed_start),
        "--seed-count",
        str(seed_count),
        "--root-query-indices",
        *(str(query) for query in queries),
        "--manifest-root-query-indices",
        *(str(query) for query in ROOT_QUERIES),
        "--action-exec-steps",
        str(ACTION_EXEC_STEPS),
        "--max-steps",
        str(MAX_STEPS),
    ]


def run_collector(
    static: Mapping[str, Any],
    *,
    body: str,
    conditions: Sequence[str],
    seed_start: int,
    seed_count: int,
    queries: Sequence[int],
    phase: str,
) -> None:
    command = collector_command(
        static,
        body=body,
        conditions=conditions,
        seed_start=seed_start,
        seed_count=seed_count,
        queries=queries,
    )
    log_path = OUTPUT_ROOT / "logs" / f"{body}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(
        "collecting",
        body=body,
        phase=phase,
        conditions=list(conditions),
        seed_start=seed_start,
        seed_count=seed_count,
        root_query_indices=list(queries),
        collector_log=str(log_path),
        collector_command=command,
    )
    print(
        "COLLECTOR_START="
        + json.dumps(
            {
                "body": body,
                "phase": phase,
                "conditions": list(conditions),
                "seed_start": seed_start,
                "seed_count": seed_count,
                "queries": list(queries),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\nWATCHER_INVOCATION="
            + json.dumps(command, ensure_ascii=True)
            + "\n"
        )
        stream.flush()
        result = subprocess.run(
            command,
            cwd=ROBOTWIN_ROOT,
            env=collector_environment(static),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise ContinuationError(
            f"collector failed for {body}/{phase}: exit {result.returncode}; {log_path}"
        )


def load_manifest(body: str, static: Mapping[str, Any]) -> dict[str, Any]:
    path = OUTPUT_ROOT / body / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"body manifest missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise ContinuationError(f"manifest logical SHA mismatch for {body}")
    if (
        value.get("format") != MANIFEST_FORMAT
        or value.get("dataset_repo") != DATASET_REPO
        or value.get("dataset_revision") != DATASET_REVISION
        or value.get("task") != TASK
        or value.get("instruction") != DEFAULT_INSTRUCTION
        or value.get("body") != body
        or value.get("collector_file_sha256") != static["collector_sha256"]
        or value.get("actor_checkpoint") != str(ACTOR_CHECKPOINT)
        or value.get("candidate_count") != CANDIDATE_COUNT
        or value.get("action_exec_steps") != ACTION_EXEC_STEPS
        or value.get("max_episode_action_steps") != MAX_STEPS
        or value.get("root_query_indices") != list(ROOT_QUERIES)
        or value.get("candidate_noise_contract") != CANDIDATE_NOISE_CONTRACT
        or value.get("terminal_supervision_contract")
        != TERMINAL_SUPERVISION_CONTRACT
        or value.get("event_age_contract") != EVENT_AGE_CONTRACT
        or value.get("terminal_horizon_contract") != TERMINAL_HORIZON_CONTRACT
        or value.get("branch_root_snapshot_contract")
        != BRANCH_ROOT_SNAPSHOT_CONTRACT
        or value.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
        or value.get("branch_diagnostic_contract")
        != BRANCH_DIAGNOSTIC_CONTRACT
        or value.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or value.get("event_derivation_implementation_sha256")
        != static["event_derivation_implementation_sha256"]
        or value.get("analytic_event_contract")
        != static["analytic_event_contract"]
        or value.get("state27_relative_goal_contract")
        != (
            "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
            "event_labels_and_online_state27_channels_0_2"
        )
        or value.get("schema_adapter", {}).get("implementation_sha256")
        != static["adapter_sha256"]
        or value.get("physical_time_contract")
        != {
            "source": "counted_successful_sapien_scene_step_calls",
            "simulator_timestep_source": "scene.get_timestep",
            "policy_action_call_count_used_as_time": False,
            "wall_clock_used_as_time": False,
            "dt_semantics": "planned_first_candidate_chunk_seconds",
            "planned_action_steps": ACTION_EXEC_STEPS,
            "actor_control_hz": 15.0,
            "planned_dt_seconds": ACTION_EXEC_STEPS / 15.0,
            "duration_semantics": "simulator_elapsed_seconds_to_event_boundary",
            "zero_elapsed_duration_masked": True,
            "stationary_window_seconds": analytic_event.THRESHOLDS[
                "stationary_window_seconds"
            ],
            "stationary_speed_threshold_m_per_s": analytic_event.THRESHOLDS[
                "stationary_speed_m_per_s"
            ],
        }
        or value.get("candidate_action_contract")
        != {
            "critic_observation_time": "before_candidate_execution",
            "planned_action_horizon": ACTION_EXEC_STEPS,
            "action_mask_source": "planned_first_chunk_not_executed_count",
            "executed_action_count_used_for_action_mask": False,
            "executed_action_count_used_for_sim_time_accounting_only": True,
            "planner_status_fail_is_a_valid_action_outcome": True,
            "python_execution_exception_invalidates_complete_decision": True,
        }
    ):
        raise ContinuationError(f"manifest static contract mismatch for {body}")
    groups = value.get("groups")
    if not isinstance(groups, list):
        raise ContinuationError(f"manifest groups are invalid for {body}")
    identities: set[str] = set()
    for item in groups:
        if not isinstance(item, Mapping):
            raise ContinuationError(f"manifest group is invalid for {body}")
        identity = str(item.get("group_id", ""))
        snapshot_hashes = (
            item.get("branch_root_snapshot_sha256"),
            item.get("branch_root_restorable_snapshot_sha256"),
            item.get("canonical_root_snapshot_sha256"),
        )
        if (
            not identity
            or identity in identities
            or item.get("collector_file_sha256") != static["collector_sha256"]
            or item.get("diagnostic_format") != DIAGNOSTIC_FORMAT
            or not isinstance(item.get("diagnostics_path"), str)
            or not isinstance(item.get("diagnostics_sha256"), str)
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in snapshot_hashes
            )
        ):
            raise ContinuationError(
                f"manifest group identity/snapshot/diagnostic binding is invalid for {body}"
            )
        identities.add(identity)
    return value


def stratum_counts(manifest: Mapping[str, Any]) -> dict[tuple[str, int], int]:
    counts = {(condition, query): 0 for condition in CONDITIONS for query in ROOT_QUERIES}
    for item in manifest["groups"]:
        key = (str(item.get("condition")), int(item.get("root_query_index", -1)))
        if key not in counts:
            raise ContinuationError(f"unexpected body-manifest stratum: {key}")
        seed = int(item.get("requested_seed", -1))
        if seed < BASE_SEED_START or seed >= FORMAL_EVALUATION_SEED_START:
            raise ContinuationError(f"development/evaluation seed boundary crossed: {seed}")
        counts[key] += 1
    if any(value > TARGET_PER_CONDITION_QUERY for value in counts.values()):
        raise ContinuationError(f"a collection stratum overshot its target: {counts}")
    return counts


def progress_path(body: str) -> Path:
    return OUTPUT_ROOT / body / "supplemental_progress.json"


def load_progress(body: str) -> dict[str, int]:
    path = progress_path(body)
    keys = [f"{condition}|{query}" for condition in CONDITIONS for query in ROOT_QUERIES]
    if not path.exists():
        return {key: SUPPLEMENTAL_SEED_START for key in keys}
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != set(keys) or any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < SUPPLEMENTAL_SEED_START
        or seed >= FORMAL_EVALUATION_SEED_START
        for seed in value.values()
    ):
        raise ContinuationError(f"invalid supplemental progress for {body}")
    return value


def base_collection_jobs() -> list[tuple[str, int]]:
    """Round-robin immutable base blocks across all five embodiments."""

    if set(COLLECTION_PRIORITY) != set(BODIES) or len(COLLECTION_PRIORITY) != len(
        BODIES
    ):
        raise ContinuationError("collection priority must contain every body once")
    return [
        (body, block_start)
        for block_start in range(0, len(ROOT_QUERIES), QUERY_BLOCK_SIZE)
        for body in COLLECTION_PRIORITY
    ]


def collect_base_block(
    static: Mapping[str, Any], body: str, block_start: int
) -> tuple[str, int]:
    if (
        body not in BODIES
        or block_start < 0
        or block_start >= len(ROOT_QUERIES)
        or block_start % QUERY_BLOCK_SIZE
    ):
        raise ContinuationError("invalid body/query block in collection schedule")
    block_index = block_start // QUERY_BLOCK_SIZE
    run_collector(
        static,
        body=body,
        conditions=CONDITIONS,
        seed_start=BASE_SEED_START + block_index * TARGET_PER_CONDITION_QUERY,
        seed_count=TARGET_PER_CONDITION_QUERY,
        queries=ROOT_QUERIES[block_start : block_start + QUERY_BLOCK_SIZE],
        phase=f"base_uniform_budget_block_{block_index:02d}",
    )
    return body, block_index


def complete_body(static: Mapping[str, Any], body: str) -> dict[str, Any]:
    """Fill terminal-root gaps and freeze one already base-collected body."""

    manifest = load_manifest(body, static)
    counts = stratum_counts(manifest)
    gaps = {
        f"{condition}|{query}": TARGET_PER_CONDITION_QUERY - count
        for (condition, query), count in counts.items()
        if count < TARGET_PER_CONDITION_QUERY
    }
    if gaps:
        print(
            "TERMINAL_ROOT_GAPS="
            + json.dumps({"body": body, "gaps": gaps}, sort_keys=True),
            flush=True,
        )
    progress = load_progress(body)
    for condition in CONDITIONS:
        for query in ROOT_QUERIES:
            key = f"{condition}|{query}"
            consecutive_failures = 0
            while counts[(condition, query)] < TARGET_PER_CONDITION_QUERY:
                seed = progress[key]
                if seed >= FORMAL_EVALUATION_SEED_START:
                    raise ContinuationError("supplemental seeds reached formal evaluation range")
                try:
                    run_collector(
                        static,
                        body=body,
                        conditions=(condition,),
                        seed_start=seed,
                        seed_count=1,
                        queries=(query,),
                        phase="supplement_terminal_root_gap",
                    )
                except ContinuationError as error:
                    progress[key] = seed + 1
                    atomic_json(progress_path(body), progress)
                    consecutive_failures += 1
                    write_state(
                        "supplemental_seed_failed_resumable",
                        body=body,
                        condition=condition,
                        root_query_index=query,
                        failed_seed=seed,
                        consecutive_failures=consecutive_failures,
                        maximum_consecutive_failures=(
                            MAX_CONSECUTIVE_SUPPLEMENTAL_FAILURES
                        ),
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_SUPPLEMENTAL_FAILURES:
                        raise ContinuationError(
                            f"{body}/{condition}/query={query} failed "
                            f"{consecutive_failures} consecutive supplemental seeds"
                        ) from error
                    continue
                progress[key] = seed + 1
                atomic_json(progress_path(body), progress)
                manifest = load_manifest(body, static)
                counts = stratum_counts(manifest)
                consecutive_failures = 0
    if len(manifest["groups"]) != EXPECTED_GROUPS_PER_BODY:
        raise ContinuationError(f"{body} did not reach exactly 400 complete decisions")
    return finalize_body_manifest(body, manifest, static)


def finalize_body_manifest(
    body: str, manifest: Mapping[str, Any], static: Mapping[str, Any]
) -> dict[str, Any]:
    path = OUTPUT_ROOT / body / "manifest.json"
    groups = list(manifest["groups"])
    for item in groups:
        relative = Path(str(item.get("path", "")))
        diagnostics_relative = Path(str(item.get("diagnostics_path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or diagnostics_relative.is_absolute()
            or ".." in diagnostics_relative.parts
        ):
            raise ContinuationError(f"unsafe group path for {body}: {relative}")
        payload = (path.parent / relative).resolve()
        diagnostics_payload = (path.parent / diagnostics_relative).resolve()
        if (
            not payload.is_file()
            or payload.is_symlink()
            or sha256_file(payload) != item.get("sha256")
            or not diagnostics_payload.is_file()
            or diagnostics_payload.is_symlink()
            or sha256_file(diagnostics_payload) != item.get("diagnostics_sha256")
        ):
            raise ContinuationError(f"group payload missing/tampered: {payload}")
    finalized = dict(manifest)
    finalized.update(
        {
            "status": "complete_400_decisions_1600_candidate_branches",
            "complete_decision_count": EXPECTED_GROUPS_PER_BODY,
            "complete_candidate_branch_count": EXPECTED_BRANCHES_PER_BODY,
            "complete_per_condition_query": TARGET_PER_CONDITION_QUERY,
            "collector_file_sha256": static["collector_sha256"],
            "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
            "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
            "event_age_contract": EVENT_AGE_CONTRACT,
            "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
            "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
            "object_effect_schema": OBJECT_EFFECT_SCHEMA,
            "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
            "actor_binding": {
                "authority_path": relative_to_home(ACTOR_AUTHORITY),
                "authority_file_sha256": sha256_file(ACTOR_AUTHORITY),
                "checkpoint_tree_sha256": json.loads(
                    ACTOR_AUTHORITY.read_text(encoding="utf-8")
                )["actors"][body]["checkpoint_sha256"],
                "config_file_sha256": json.loads(
                    ACTOR_AUTHORITY.read_text(encoding="utf-8")
                )["actors"][body]["config_file_sha256"],
                "state_shape": [16],
                "action_shape": [16],
                "same_universal_actor_for_all_bodies": True,
            },
            "completion_contract": {
                "conditions": list(CONDITIONS),
                "root_query_indices": list(ROOT_QUERIES),
                "complete_decisions_per_condition_query": TARGET_PER_CONDITION_QUERY,
                "base_seed_start": BASE_SEED_START,
                "supplemental_seeds_below": FORMAL_EVALUATION_SEED_START,
                "formal_evaluation_seeds_used": False,
            },
        }
    )
    finalized.pop("logical_sha256", None)
    finalized["logical_sha256"] = canonical_sha256(finalized)
    atomic_json(path, finalized)
    for item in groups:
        (path.parent / str(item["path"])).chmod(0o444)
        (path.parent / str(item["diagnostics_path"])).chmod(0o444)
    path.chmod(0o444)
    return {
        "path": relative_to_home(path),
        "sha256": sha256_file(path),
        "logical_sha256": finalized["logical_sha256"],
        "complete_decisions": EXPECTED_GROUPS_PER_BODY,
        "candidate_branches": EXPECTED_BRANCHES_PER_BODY,
    }


def body_already_complete(body: str, static: Mapping[str, Any]) -> dict[str, Any] | None:
    path = OUTPUT_ROOT / body / "manifest.json"
    if not path.exists():
        return None
    manifest = load_manifest(body, static)
    if manifest.get("status") != "complete_400_decisions_1600_candidate_branches":
        return None
    counts = stratum_counts(manifest)
    if set(counts.values()) != {TARGET_PER_CONDITION_QUERY}:
        raise ContinuationError(f"finalized stratum counts changed for {body}")
    return finalize_body_manifest(body, manifest, static)


def freeze_training_binding(
    actor_authority_sha: str, body_manifests: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], str]:
    binding = signed(
        {
            "format": BINDING_FORMAT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "instruction": DEFAULT_INSTRUCTION,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
            "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
            "event_age_contract": EVENT_AGE_CONTRACT,
            "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
            "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
            "object_effect_schema": OBJECT_EFFECT_SCHEMA,
            "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
            "heldout_labels_may_train_fit_calibrate_or_select": False,
            "canonical_shared_body_rows": 1,
            "execution_authority": {
                "explicit_user_training_request_recorded": True,
                "public_data_only": True,
                "protected_internal_data_allowed": False,
                "remote_cuda_only": True,
            },
            "materialization_receipt": {
                "path": relative_to_home(MATERIALIZATION_RECEIPT),
                "sha256": sha256_file(MATERIALIZATION_RECEIPT),
            },
            "actor_authority": {
                "path": relative_to_home(ACTOR_AUTHORITY),
                "sha256": actor_authority_sha,
            },
            "body_manifests": {
                body: {
                    "path": body_manifests[body]["path"],
                    "sha256": body_manifests[body]["sha256"],
                }
                for body in BODIES
            },
        }
    )
    return binding, write_static(TRAINING_BINDING, binding)


def main() -> int:
    if Path.home().resolve() != HOME_ROOT:
        raise ContinuationError("this continuation watcher is remote /home/user only")
    WATCHER_PID.write_text(f"{os.getpid()}\n", encoding="utf-8")
    upstream, upstream_sha = wait_for_frozen_actor()
    static = validate_static_inputs()
    actor_authority, actor_authority_sha = freeze_actor_authority(
        upstream, upstream_sha, static
    )
    write_state(
        "actor_frozen",
        upstream_status="complete",
        actor_checkpoint_tree_sha256=actor_authority["actors"][BODIES[0]][
            "checkpoint_sha256"
        ],
        actor_config_state_shape=[16],
        actor_config_action_shape=[16],
        actor_authority_file_sha256=actor_authority_sha,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    def finalize_body(body: str) -> tuple[str, dict[str, Any]]:
        completed = body_already_complete(body, static)
        return body, completed or complete_body(static, body)

    # Three collectors fit the measured 4090 memory envelope while leaving
    # enough headroom for CuRobo/Vulkan peaks.  Four would approach the 24 GiB
    # device limit and turn throughput optimization into avoidable OOM risk.
    # Scheduling fifty resumable blocks, instead of five whole-body jobs,
    # keeps all three slots occupied until the final block while never allowing
    # concurrent writers for one body manifest.
    completed_by_body: dict[str, dict[str, Any]] = {}
    already_complete = {
        body: body_already_complete(body, static) for body in BODIES
    }
    completed_by_body.update(
        {
            body: receipt
            for body, receipt in already_complete.items()
            if receipt is not None
        }
    )
    body_locks = {body: threading.Lock() for body in BODIES}

    def locked_base_block(body: str, block_start: int) -> tuple[str, int]:
        with body_locks[body]:
            return collect_base_block(static, body, block_start)

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="body-collector") as pool:
        base_futures: dict[Any, tuple[str, int]] = {}
        for job_index, (body, block_start) in enumerate(base_collection_jobs()):
            if already_complete[body] is not None:
                continue
            future = pool.submit(locked_base_block, body, block_start)
            base_futures[future] = (body, block_start // QUERY_BLOCK_SIZE)
            if job_index < 2:
                time.sleep(20.0)
        completed_base_blocks = 0
        failed_base_blocks: list[dict[str, Any]] = []
        for future in as_completed(base_futures):
            scheduled_body, scheduled_block = base_futures[future]
            try:
                body, block_index = future.result()
            except Exception as error:
                failed_base_blocks.append(
                    {
                        "body": scheduled_body,
                        "block_index": scheduled_block,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                write_state(
                    "base_block_failed_resumable",
                    failed_body=scheduled_body,
                    failed_block=scheduled_block,
                    failed_base_blocks=list(failed_base_blocks),
                    completed_base_blocks=completed_base_blocks,
                    expected_base_blocks=len(base_futures),
                    max_parallel_body_collectors=3,
                )
                continue
            completed_base_blocks += 1
            write_state(
                "collecting_base_blocks",
                last_completed_body=body,
                last_completed_block=block_index,
                completed_base_blocks=completed_base_blocks,
                expected_base_blocks=len(base_futures),
                failed_base_blocks=list(failed_base_blocks),
                max_parallel_body_collectors=3,
            )

        finalize_futures = [
            pool.submit(finalize_body, body)
            for body in COLLECTION_PRIORITY
            if already_complete[body] is None
        ]
        for future in as_completed(finalize_futures):
            body, manifest = future.result()
            completed_by_body[body] = manifest
            write_state(
                "body_complete",
                completed_body=body,
                completed_bodies=[
                    value for value in BODIES if value in completed_by_body
                ],
                completed_candidate_branches=sum(
                    value["candidate_branches"]
                    for value in completed_by_body.values()
                ),
                max_parallel_body_collectors=3,
            )
    body_manifests = {body: completed_by_body[body] for body in BODIES}
    receipt = signed(
        {
            "format": RECEIPT_FORMAT,
            "status": "complete",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "event_derivation_implementation_sha256": static[
                "event_derivation_implementation_sha256"
            ],
            "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
            "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
            "event_age_contract": EVENT_AGE_CONTRACT,
            "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
            "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
            "object_effect_schema": OBJECT_EFFECT_SCHEMA,
            "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
            "output_root": str(OUTPUT_ROOT),
            "actor_authority_file_sha256": actor_authority_sha,
            "body_manifests": body_manifests,
            "complete_decisions": EXPECTED_TOTAL_BRANCHES // CANDIDATE_COUNT,
            "candidate_branches": EXPECTED_TOTAL_BRANCHES,
            "conditions": list(CONDITIONS),
            "root_query_indices": list(ROOT_QUERIES),
            "formal_evaluation_seeds_used": False,
            "protected_internal_data_opened": False,
        }
    )
    receipt_path = OUTPUT_ROOT / "collection_receipt.json"
    receipt_sha = write_static(receipt_path, receipt)
    binding, binding_sha = freeze_training_binding(
        actor_authority_sha, body_manifests
    )
    write_state(
        "complete",
        actor_checkpoint_tree_sha256=actor_authority["actors"][BODIES[0]][
            "checkpoint_sha256"
        ],
        actor_authority_file_sha256=actor_authority_sha,
        collection_receipt=str(receipt_path),
        collection_receipt_file_sha256=receipt_sha,
        collection_receipt_logical_sha256=receipt["logical_sha256"],
        training_binding_file_sha256=binding_sha,
        training_binding_logical_sha256=binding["logical_sha256"],
        complete_decisions=EXPECTED_TOTAL_BRANCHES // CANDIDATE_COUNT,
        candidate_branches=EXPECTED_TOTAL_BRANCHES,
    )
    print(
        "FULL8000_COMPLETE="
        + json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "binding": str(TRAINING_BINDING),
                "binding_sha256": binding_sha,
                "candidate_branches": EXPECTED_TOTAL_BRANCHES,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            write_state("failed", error=f"{type(error).__name__}: {error}")
        except Exception:
            pass
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
