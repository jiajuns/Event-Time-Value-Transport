#!/usr/bin/env python3
"""Train one strict five-body RoboTwin2 LOBO shared event/effect head.

This entry point consumes *prepared public* canonical transition groups.  It
does not open RoboTwin zip files and it does not train a VLA.  A separately
audited, deterministic and label-free parser must first map every robot to the
same 27-D state, 14-D action-effect and five-event vocabulary.

The held-out body is manifest-visible for fold assignment only.  Its group
payloads are not opened, its labels never fit normalization/parameters or
select checkpoints, and the model has a single shared body row.  Task-success
evaluation is deliberately a later live-simulator stage using the frozen actor
and the same ordered candidate set for baseline and ETSF.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_multibody_canonical_event_world_model as core
import robotwin2_cross_body_canonical_adapter_v1 as canonical_adapter
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event
import verify_robotwin2_move_can_pot_public_materialization_v1 as public_materialization


FORMAT = "etsf_robotwin2_five_body_lobo_shared_event_head_v1"
MODEL_FAMILY = "terminal_consequence_utility_shared_event_head_v6"
BINDING_FORMAT = "etsf_robotwin2_five_body_lobo_training_binding_v1"
MANIFEST_FORMAT = "etsf_robotwin2_canonical_transition_manifest_v1"
ACTOR_FORMAT = "etsf_robotwin2_frozen_native_actor_authority_v1"
MATERIALIZATION_FORMAT = public_materialization.FORMAT
DATASET_REPO = "TianxingChen/RoboTwin2.0"
DATASET_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
TASK = "move_can_pot"
DEFAULT_INSTRUCTION = "Move the can to the side of the pot."
PREREGISTRATION_SHA256 = (
    "75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee"
)
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
SOURCE_EVENT_SAMPLING_HZ = 15.0
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
CANDIDATE_COUNT = 4
CANDIDATE_RANK_FEATURE_SCHEMA = {
    "post_event_probability": (0, 5),
    "next_event_probability": (5, 10),
    "success_probability": (10, 11),
    "regression_probability": (11, 12),
    "joint_recovery_probability": (12, 13),
    "duration_log1p_mean": (13, 14),
    "duration_log1p_scale": (14, 15),
    "object_delta_mean": (15, 21),
    "object_delta_scale": (21, 27),
    "predicted_goal_progress": (27, 28),
    "predicted_goal_progress_uncertainty": (28, 29),
    "terminal_event_probability": (29, 34),
    "terminal_goal_progress_mean": (34, 35),
    "terminal_goal_progress_scale": (35, 36),
}
CANDIDATE_RANK_FEATURE_DIM = max(
    stop for _start, stop in CANDIDATE_RANK_FEATURE_SCHEMA.values()
)
OBJECT_STUDENT_T_DOF = 3.0
TERMINAL_PROGRESS_STUDENT_T_DOF = 3.0
TERMINAL_EVENT_LOSS_WEIGHT = 0.5
TERMINAL_GOAL_PROGRESS_LOSS_WEIGHT = 0.5
DENSE_FAILURE_RANK_WEIGHT = 0.1
DENSE_RANK_LABEL_EQUALITY_TOLERANCE = 1e-6
ONE_DEVIATION_ESTIMAND = (
    "one_candidate_deviation_then_frozen_actor_continuation_not_"
    "recursive_closed_loop_delta_success_rate"
)
RANK_ENSEMBLE_STD_FLOOR = 1e-6
STANDARDIZED_RANK_ENSEMBLE_CONTRACT = {
    "format": "etsf_within_decision_standardized_rank_ensemble_v1",
    "member_count": 5,
    "candidate_count": CANDIDATE_COUNT,
    "member_transform": "subtract_candidate_mean_divide_population_std",
    "population_std_correction": 0,
    "std_floor": RANK_ENSEMBLE_STD_FLOOR,
    "member_with_std_at_or_below_floor": "all_zero_contribution",
    "aggregation": "equal_mean_over_exactly_five_member_contributions",
    "normalization_scope": "one_four_candidate_decision_per_member",
}
CANDIDATE_NOISE_CONTRACT = {
    "distribution": "antithetic_standard_normal_pairs_each_marginal_N_0_I",
    "candidate_indices": [0, 1, 2, 3],
    "base_noise_indices": [0, 0, 2, 2],
    "signs": [1, -1, 1, -1],
    "candidate_zero_legacy_noise_unchanged": True,
}
OBJECT_EFFECT_SCHEMA = {
    "format": "etsf_robotwin2_moving_object_se3_effect_6d_v1",
    "channels": list(canonical_adapter.OBJECT_EFFECT6_CHANNELS),
    "rotation": "q_post_times_conjugate_q_root_shortest_axis_angle_wxyz",
    "redundant_relative_goal_delta_removed": True,
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
    "formal_episode_action_steps": 200,
    "formal_actor_query_stride_actions": 5,
    "development_remaining_action_budgets": list(range(200, 0, -5)),
}
BRANCH_ROOT_SNAPSHOT_CONTRACT = {
    "format": "etsf_sapien_explicit_fresh_scene_branch_root_v1",
    "physics_state": "keyed_rigid_articulation_drive_task_render_rng_snapshot",
    "candidate_scene_isolation": "one_fresh_scene_per_candidate",
    "contact_cache_reconstruction": "one_counted_raw_scene_step",
    "derived_articulation_qacc": (
        "recorded_for_provenance_not_required_pre_step_then_recomputed_and_"
        "strictly_hashed_after_canonicalization_step"
    ),
    "simulation_clock_restored": True,
    "task_counters_restored": ["take_action_cnt", "eval_success"],
    "rng_restored": ["python", "numpy", "torch_cpu", "torch_cuda"],
    "reset_and_action_prefix_replay_used_for_candidates": False,
}
BRANCH_DIAGNOSTIC_CONTRACT = {
    "format": "etsf_robotwin2_candidate_branch_diagnostics_v1",
    "first_executed": "successful_or_physics_advancing_actions_in_planned_first_chunk",
    "branch_error": "all_false_execution_exception_invalidates_complete_decision",
    "candidate_action_pairwise_rms": (
        "symmetric_raw_canonical_effect_rms_over_planned_first_five_actions"
    ),
}
ABLATION_VARIANTS = (
    "success_only",
    "no_time_duration",
    "no_object_effect",
    "full",
)


def _dense_rank_labels_are_orderable(
    terminal_event_level: Sequence[float],
    terminal_goal_progress: Sequence[float],
    *,
    ablation_variant: str,
) -> bool:
    """Return whether an all-failure decision contains a ranking preference."""

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
    event_values = [float(value) for value in terminal_event_level]
    goal_values = [float(value) for value in terminal_goal_progress]
    if (
        len(event_values) != CANDIDATE_COUNT
        or len(goal_values) != CANDIDATE_COUNT
        or not all(math.isfinite(value) for value in (*event_values, *goal_values))
    ):
        raise FiveBodyContractError("dense rank labels have invalid shape/value")
    if ablation_variant == "success_only":
        return False
    if max(event_values) - min(event_values) > DENSE_RANK_LABEL_EQUALITY_TOLERANCE:
        return True
    return bool(
        ablation_variant != "no_object_effect"
        and max(goal_values) - min(goal_values)
        > DENSE_RANK_LABEL_EQUALITY_TOLERANCE
    )


def ablation_contract(variant: str) -> dict[str, Any]:
    if variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    return {
        "variant": variant,
        "candidate_score": (
            "proper_horizon_coherent_terminal_eK_success_logit"
            if variant == "success_only"
            else "group_listwise_candidate_rank_logit"
        ),
        "multitask_heads_enabled": (
            ["success"]
            if variant == "success_only"
            else [
                "post_event",
                "next_event",
                *([] if variant == "no_time_duration" else ["duration"]),
                "success",
                "recovery",
                "terminal_event",
                *(
                    []
                    if variant == "no_object_effect"
                    else ["terminal_goal_progress"]
                ),
            ]
        ),
        "time_duration_rank_features_enabled": variant
        not in {"success_only", "no_time_duration"},
        "terminal_horizon_context_enabled": variant != "no_time_duration",
        "object_effect_loss_and_rank_target_enabled": variant
        not in {"success_only", "no_object_effect"},
        "mixed_success_objective": (
            "none"
            if variant == "success_only"
            else "negative_log_softmax_mass_on_any_successful_candidate"
        ),
        "all_failure_dense_objective_weight": (
            0.0 if variant == "success_only" else DENSE_FAILURE_RANK_WEIGHT
        ),
        "all_failure_dense_informative_labels_only": True,
        "dense_target_order": (
            "none"
            if variant == "success_only"
            else "strict_terminal_max_event_then_soft_goal_progress_temperature_0.02m"
        ),
        "training_streams": (
            "uniform_proper_likelihood_plus_macro_balanced_rank_only"
        ),
        "rank_ensemble_aggregation": standardized_rank_ensemble_contract(),
        "same_seed_disjoint_split": True,
        "heldout_labels_used_for_training_or_selection": False,
    }


def ablation_selection_components(
    components: Mapping[str, float], variant: str
) -> dict[str, float]:
    allowed = {
        "success_only": {"success_brier_ratio"},
        "no_time_duration": {
            "post_event_macro_error_ratio",
            "next_event_macro_error_ratio",
            "success_brier_ratio",
            "object_rmse_ratio",
        },
        "no_object_effect": {
            "post_event_macro_error_ratio",
            "next_event_macro_error_ratio",
            "observed_duration_mae_ratio",
            "success_brier_ratio",
        },
        "full": set(components),
    }
    if variant not in allowed:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    selected = {
        name: float(value) for name, value in components.items() if name in allowed[variant]
    }
    if not selected:
        raise FiveBodyContractError(
            f"{variant} has no enabled validation diagnostic for checkpoint selection"
        )
    return selected


def checkpoint_candidate_rank_contract(variant: str) -> dict[str, Any]:
    """Describe the score path saved in one selected checkpoint."""

    if variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    if variant == "success_only":
        feature_blocks = [
            "proper_success_logit_exactly_from_terminal_event_eK_probability"
        ]
    elif variant == "no_time_duration":
        feature_blocks = [
            "post_event_probability",
            "next_event_probability",
            "success_probability",
            "regression_probability_from_post_event_distribution",
            "joint_recovery_probability",
            "duration_log1p_mean_forced_zero",
            "duration_log1p_scale_forced_zero",
            "object_delta_mean",
            "object_delta_scale",
            "predicted_goal_progress_from_state_relative_goal",
            "predicted_goal_progress_uncertainty_from_object_student_t_scale",
            "terminal_event_probability",
            "terminal_goal_progress_mean",
            "terminal_goal_progress_scale",
        ]
    elif variant == "no_object_effect":
        feature_blocks = [
            "post_event_probability",
            "next_event_probability",
            "success_probability",
            "regression_probability_from_post_event_distribution",
            "joint_recovery_probability",
            "duration_log1p_mean",
            "duration_log1p_scale",
            "object_delta_mean_forced_zero",
            "object_delta_scale_forced_zero",
            "predicted_goal_progress_forced_zero",
            "predicted_goal_progress_uncertainty_forced_zero",
            "terminal_event_probability",
            "terminal_goal_progress_mean_forced_zero",
            "terminal_goal_progress_scale_forced_zero",
        ]
    else:
        feature_blocks = [
            "post_event_probability",
            "next_event_probability",
            "success_probability",
            "regression_probability_from_post_event_distribution",
            "joint_recovery_probability",
            "duration_log1p_mean",
            "duration_log1p_scale",
            "object_delta_mean",
            "object_delta_scale",
            "predicted_goal_progress_from_state_relative_goal",
            "predicted_goal_progress_uncertainty_from_object_student_t_scale",
            "terminal_event_probability",
            "terminal_goal_progress_mean",
            "terminal_goal_progress_scale",
        ]
    return {
        "feature_blocks": feature_blocks,
        "feature_schema": {
            name: list(bounds)
            for name, bounds in CANDIDATE_RANK_FEATURE_SCHEMA.items()
        },
        "feature_dim": CANDIDATE_RANK_FEATURE_DIM,
        "dt_has_numeric_score_path": variant not in {"success_only", "no_time_duration"},
        "duration_conditions_on_physical_event_age": variant
        not in {"success_only", "no_time_duration"},
        "terminal_consequences_condition_on_remaining_action_budget": variant
        != "no_time_duration",
        "remaining_action_budget_has_direct_rank_path": False,
        "event_age_has_numeric_score_path": variant != "no_time_duration",
        "direct_transitioned_or_clock_hidden_rank_path": False,
        "rank_inputs_are_detached_consequence_predictions": variant != "success_only",
        "goal_progress_definition": (
            "norm(state_relative_goal_xyz)-norm(state_relative_goal_xyz-"
            "predicted_object_translation_mean)"
        ),
        "goal_progress_uncertainty_definition": (
            "delta_method_radial_std_from_student_t3_object_translation_scale"
        ),
        "pairwise_rank_loss_enabled": False,
        "group_listwise_success_mass_loss_enabled": variant != "success_only",
        "all_failure_dense_listwise_loss_enabled": variant != "success_only",
        "all_failure_dense_listwise_weight": (
            0.0 if variant == "success_only" else DENSE_FAILURE_RANK_WEIGHT
        ),
        "all_failure_dense_informative_labels_only": True,
        "dense_rank_label_equality_tolerance": DENSE_RANK_LABEL_EQUALITY_TOLERANCE,
        "dense_target_requires_full_continuation": True,
        "dense_goal_progress_temperature_meters": (
            None
            if variant in {"success_only", "no_object_effect"}
            else DENSE_GOAL_PROGRESS_TEMPERATURE_METERS
        ),
        "proper_and_rank_batches_are_separate": True,
        "mixed_rank_bootstrap": "one_plus_poisson",
        "dense_rank_bootstrap": "poisson",
        "rank_macro_strata": "body_condition_current_event",
        "rank_ensemble_aggregation": standardized_rank_ensemble_contract(),
        "rank_loss_updates_clock_or_duration_heads": False,
        "rank_loss_updates_semantic_action_transition": False,
        "rank_loss_updates_consequence_predictors": False,
        "terminal_event_loss": "proper_categorical_cross_entropy_uniform_stream",
        "terminal_goal_progress_loss": "proper_student_t3_nll_uniform_stream",
        "success_probability_definition": (
            "exact_terminal_event_probability_of_eK_with_shared_horizon_context"
        ),
        "recovery_probability_conditions_on_terminal_horizon_context": (
            variant != "no_time_duration"
        ),
    }


def summary_candidate_rank_contract(variant: str) -> dict[str, Any]:
    """Describe rank supervision without claiming disabled ablation paths."""

    checkpoint = checkpoint_candidate_rank_contract(variant)
    return {
        "candidate_score": ablation_contract(variant)["candidate_score"],
        "feature_blocks": checkpoint["feature_blocks"],
        "feature_dim": checkpoint["feature_dim"],
        "time_and_duration_effect_used": variant
        not in {"success_only", "no_time_duration"},
        "dt_has_numeric_score_path": checkpoint["dt_has_numeric_score_path"],
        "duration_conditions_on_physical_event_age": checkpoint[
            "duration_conditions_on_physical_event_age"
        ],
        "event_age_has_numeric_score_path": checkpoint[
            "event_age_has_numeric_score_path"
        ],
        "terminal_consequences_condition_on_remaining_action_budget": checkpoint[
            "terminal_consequences_condition_on_remaining_action_budget"
        ],
        "remaining_action_budget_has_direct_rank_path": checkpoint[
            "remaining_action_budget_has_direct_rank_path"
        ],
        "pairwise_rank_loss_enabled": checkpoint["pairwise_rank_loss_enabled"],
        "group_listwise_success_mass_loss_enabled": checkpoint[
            "group_listwise_success_mass_loss_enabled"
        ],
        "all_failure_dense_listwise_loss_enabled": checkpoint[
            "all_failure_dense_listwise_loss_enabled"
        ],
        "all_failure_dense_listwise_weight": checkpoint[
            "all_failure_dense_listwise_weight"
        ],
        "all_failure_dense_informative_labels_only": checkpoint[
            "all_failure_dense_informative_labels_only"
        ],
        "dense_rank_label_equality_tolerance": checkpoint[
            "dense_rank_label_equality_tolerance"
        ],
        "dense_target_requires_full_continuation": True,
        "dense_goal_progress_temperature_meters": checkpoint[
            "dense_goal_progress_temperature_meters"
        ],
        "proper_and_rank_batches_are_separate": True,
        "mixed_rank_bootstrap": "one_plus_poisson",
        "dense_rank_bootstrap": "poisson",
        "rank_macro_strata": "body_condition_current_event",
        "rank_ensemble_aggregation": standardized_rank_ensemble_contract(),
        "rank_loss_updates_clock_or_duration_heads": False,
        "rank_loss_updates_semantic_action_transition": checkpoint[
            "rank_loss_updates_semantic_action_transition"
        ],
        "rank_loss_updates_consequence_predictors": checkpoint[
            "rank_loss_updates_consequence_predictors"
        ],
        "direct_transitioned_or_clock_hidden_rank_path": checkpoint[
            "direct_transitioned_or_clock_hidden_rank_path"
        ],
        "rank_inputs_are_detached_consequence_predictions": checkpoint[
            "rank_inputs_are_detached_consequence_predictions"
        ],
        "terminal_event_loss": checkpoint["terminal_event_loss"],
        "terminal_goal_progress_loss": checkpoint[
            "terminal_goal_progress_loss"
        ],
        "success_probability_definition": checkpoint[
            "success_probability_definition"
        ],
        "recovery_probability_conditions_on_terminal_horizon_context": checkpoint[
            "recovery_probability_conditions_on_terminal_horizon_context"
        ],
        "feature_schema": checkpoint["feature_schema"],
        "goal_progress_definition": checkpoint["goal_progress_definition"],
        "goal_progress_uncertainty_definition": checkpoint[
            "goal_progress_uncertainty_definition"
        ],
    }
CANONICAL_STATE_SCHEMA = canonical_adapter.STATE_SCHEMA
CANONICAL_ACTION_SCHEMA = canonical_adapter.ACTION_SCHEMA
REQUIRED_ARRAYS = {
    "state",
    "actions",
    "action_mask",
    "current_event_id",
    "post_event_id",
    "post_event_mask",
    "next_event_id",
    "next_event_mask",
    "duration",
    "duration_observed",
    "duration_mask",
    "success",
    "success_mask",
    "recovery",
    "recovery_mask",
    "object_delta",
    "object_delta_mask",
    "terminal_max_event_id",
    "terminal_event_mask",
    "terminal_stage_progress",
    "terminal_goal_distance",
    "terminal_goal_progress",
    "terminal_goal_progress_mask",
    "terminal_stop_reason_id",
    "candidate_index",
    "event_age_seconds",
    "remaining_action_budget",
    "dt",
}


class FiveBodyContractError(RuntimeError):
    """A five-body training authority or payload failed closed."""


def standardized_rank_ensemble_contract() -> dict[str, Any]:
    """Return the frozen deployment aggregation contract without shared mutation."""

    return dict(STANDARDIZED_RANK_ENSEMBLE_CONTRACT)


def event_age_contract() -> dict[str, Any]:
    """Return the frozen physical event-age input contract."""

    return dict(EVENT_AGE_CONTRACT)


def terminal_horizon_contract() -> dict[str, Any]:
    """Return the pre-action finite-horizon conditioning contract."""

    return dict(TERMINAL_HORIZON_CONTRACT)


def aggregate_standardized_rank_scores(
    member_scores: torch.Tensor,
    *,
    std_floor: float = RANK_ENSEMBLE_STD_FLOOR,
) -> torch.Tensor:
    """Aggregate five raw rank heads without allowing one scale to dominate.

    The last axis is one complete four-candidate decision.  Leading axes after
    the member axis are permitted so callers can aggregate several already
    grouped decisions at once.  A constant member contributes an all-zero row;
    the final divisor remains five so every member keeps exactly equal weight.
    """

    if not isinstance(member_scores, torch.Tensor):
        raise TypeError("member rank scores must be a torch.Tensor")
    if (
        member_scores.ndim < 2
        or member_scores.shape[0] != 5
        or member_scores.shape[-1] != CANDIDATE_COUNT
    ):
        raise FiveBodyContractError(
            "member rank scores must be [5,...,4] with decisions on the last axis"
        )
    if not math.isclose(
        float(std_floor), RANK_ENSEMBLE_STD_FLOOR, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FiveBodyContractError("rank ensemble std floor is frozen by contract")
    if not bool(torch.isfinite(member_scores).all()):
        raise FiveBodyContractError("member rank scores contain non-finite values")
    centered = member_scores - member_scores.mean(dim=-1, keepdim=True)
    population_std = centered.square().mean(dim=-1, keepdim=True).sqrt()
    active = population_std > float(std_floor)
    denominator = population_std.clamp_min(float(std_floor))
    standardized = torch.where(active, centered / denominator, torch.zeros_like(centered))
    return standardized.mean(dim=0)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    """Hash a checkpoint directory as ordered relative path/size/file hashes."""

    root = path.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FiveBodyContractError("checkpoint tree must be a real directory")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise FiveBodyContractError("checkpoint tree contains a symbolic link")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise FiveBodyContractError("checkpoint tree contains a special file")
        rows.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise FiveBodyContractError("checkpoint tree is empty")
    return canonical_sha256(rows), len(rows), sum(row["size_bytes"] for row in rows)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _read_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FiveBodyContractError(f"{label} is not a file: {resolved}")
    if not _is_sha(expected_sha256) or sha256_file(resolved) != expected_sha256.lower():
        raise FiveBodyContractError(f"{label} SHA-256 mismatch")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FiveBodyContractError(f"{label} must be a JSON object")
    return value


def _resolve_contained(parent: Path, relative: str, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise FiveBodyContractError(f"{label} path must be relative")
    resolved_parent = parent.resolve()
    lexical = resolved_parent
    for component in raw.parts:
        lexical = lexical / component
        if lexical.is_symlink():
            raise FiveBodyContractError(f"{label} path contains a symbolic link")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(resolved_parent)
    except ValueError as error:
        raise FiveBodyContractError(f"{label} escapes its manifest directory") from error
    return resolved


def _lexical_contained_payload_path(parent: Path, relative: str, label: str) -> Path:
    """Resolve a declared payload path without stat/open/read of that payload."""

    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise FiveBodyContractError(f"{label} path must be a safe relative path")
    absolute_parent = Path(os.path.abspath(os.fspath(parent)))
    absolute = Path(os.path.abspath(os.fspath(absolute_parent / raw)))
    try:
        absolute.relative_to(absolute_parent)
    except ValueError as error:
        raise FiveBodyContractError(f"{label} escapes its manifest directory") from error
    return absolute


def _verify_signed(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    logical = unsigned.pop("logical_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise FiveBodyContractError(f"{label} logical SHA-256 mismatch")


def validate_materialization_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        public_materialization.validate_receipt(value)
    except public_materialization.PublicMaterializationError as error:
        raise FiveBodyContractError("public payload materialization is not verified") from error
    if (
        value.get("format") != MATERIALIZATION_FORMAT
        or value.get("status") != public_materialization.STATUS
        or value.get("hf_repo_id") != DATASET_REPO
        or value.get("hf_repo_revision") != DATASET_REVISION
        or value.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or value.get("official_file_count") != 11
        or value.get("all_exact_archive_payload_sha256_verified") is not True
    ):
        raise FiveBodyContractError("public receipt is not the frozen five-body slice")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 11:
        raise FiveBodyContractError("materialization must bind all 11 official slice files")
    paths = [item.get("path") for item in files if isinstance(item, Mapping)]
    if len(paths) != 11 or len(set(paths)) != 11:
        raise FiveBodyContractError("materialization file paths are incomplete or duplicated")
    for item in files:
        if (
            not isinstance(item, Mapping)
            or not _is_sha(item.get("observed_payload_sha256"))
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] <= 0
            or item.get("size_match") is not True
            or item.get("payload_sha256_match") is not True
        ):
            raise FiveBodyContractError("materialization contains an unverified payload")
    return {"file_count": 11, "file_set_sha256": canonical_sha256(files)}


def validate_actor_authority(
    value: Mapping[str, Any], *, authority_dir: Path
) -> dict[str, Any]:
    _verify_signed(value, "actor authority")
    if value.get("format") != ACTOR_FORMAT or value.get("task") != TASK:
        raise FiveBodyContractError("actor authority format/task mismatch")
    actors = value.get("actors")
    if not isinstance(actors, Mapping) or set(actors) != set(BODIES):
        raise FiveBodyContractError("actor authority must bind exactly five bodies")
    for body in BODIES:
        actor = actors[body]
        if (
            not isinstance(actor, Mapping)
            or actor.get("frozen") is not True
            or actor.get("optimizer_updates_allowed") is not False
            or actor.get("candidate_count") != CANDIDATE_COUNT
            or actor.get("candidate_zero_is_actor_baseline") is not True
            or actor.get("same_ordered_candidate_set_for_baseline_and_etsf") is not True
            or not _is_sha(actor.get("checkpoint_sha256"))
            or not _is_sha(actor.get("sampling_contract_sha256"))
            or actor.get("checkpoint_kind") not in {"file", "directory_tree"}
        ):
            raise FiveBodyContractError(f"actor authority is incomplete for {body}")
        checkpoint = _resolve_contained(
            authority_dir, str(actor.get("checkpoint_path", "")), f"{body} actor checkpoint"
        )
        if actor["checkpoint_kind"] == "file":
            observed_sha = sha256_file(checkpoint) if checkpoint.is_file() else None
        else:
            observed_sha = sha256_tree(checkpoint)[0] if checkpoint.is_dir() else None
        if observed_sha != actor["checkpoint_sha256"]:
            raise FiveBodyContractError(f"frozen actor checkpoint missing/tampered for {body}")
    return {
        "actor_frozen": True,
        "bodies": list(BODIES),
        "candidate_count": CANDIDATE_COUNT,
        "same_ordered_candidate_set": True,
    }


def validate_body_manifest(
    value: Mapping[str, Any], *, expected_body: str, manifest_dir: Path
) -> dict[str, Any]:
    _verify_signed(value, f"{expected_body} canonical manifest")
    if (
        value.get("format") != MANIFEST_FORMAT
        or value.get("dataset_repo") != DATASET_REPO
        or value.get("dataset_revision") != DATASET_REVISION
        or value.get("task") != TASK
        or value.get("instruction") != DEFAULT_INSTRUCTION
        or value.get("body") != expected_body
    ):
        raise FiveBodyContractError(f"canonical manifest identity mismatch for {expected_body}")
    adapter = value.get("schema_adapter")
    if (
        not isinstance(adapter, Mapping)
        or adapter.get("kind") != "analytic_label_free_canonical_v1"
        or adapter.get("trainable") is not False
        or adapter.get("labels_or_outcomes_used_to_fit") is not False
        or adapter.get("heldout_supervision_allowed") is not False
        or adapter.get("state_dim") != core.STATE_DIM
        or adapter.get("action_dim") != core.ACTION_DIM
        or adapter.get("state_schema") != CANONICAL_STATE_SCHEMA
        or adapter.get("action_schema") != CANONICAL_ACTION_SCHEMA
        or adapter.get("elapsed_time_unit") != "seconds"
        or adapter.get("duration_unit") != "seconds"
        or adapter.get("event_names") != list(core.CANONICAL_EVENTS)
        or not _is_sha(adapter.get("implementation_sha256"))
    ):
        raise FiveBodyContractError(f"{expected_body} adapter is not analytic/label-free")
    physical_time = value.get("physical_time_contract")
    try:
        analytic_event.validate_event_contract(value.get("analytic_event_contract"))
    except analytic_event.AnalyticEventSpecError as error:
        raise FiveBodyContractError(
            f"{expected_body} analytic event contract changed"
        ) from error
    if (
        value.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or not _is_sha(value.get("event_derivation_implementation_sha256"))
        or value.get("state27_relative_goal_contract")
        != (
            "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
            "event_labels_and_online_state27_channels_0_2"
        )
        or not isinstance(physical_time, Mapping)
        or physical_time.get("source")
        != "counted_successful_sapien_scene_step_calls"
        or physical_time.get("simulator_timestep_source") != "scene.get_timestep"
        or physical_time.get("policy_action_call_count_used_as_time") is not False
        or physical_time.get("wall_clock_used_as_time") is not False
        or physical_time.get("dt_semantics")
        != "planned_first_candidate_chunk_seconds"
        or physical_time.get("planned_action_steps") != 5
        or physical_time.get("actor_control_hz") != SOURCE_EVENT_SAMPLING_HZ
        or physical_time.get("planned_dt_seconds") != 5.0 / SOURCE_EVENT_SAMPLING_HZ
        or physical_time.get("duration_semantics")
        != "simulator_elapsed_seconds_to_event_boundary"
        or physical_time.get("zero_elapsed_duration_masked") is not True
        or physical_time.get("stationary_window_seconds")
        != analytic_event.THRESHOLDS["stationary_window_seconds"]
        or physical_time.get("stationary_speed_threshold_m_per_s")
        != analytic_event.THRESHOLDS["stationary_speed_m_per_s"]
    ):
        raise FiveBodyContractError(
            f"{expected_body} lacks the physical simulator time/event contract"
        )
    candidate_action = value.get("candidate_action_contract")
    if candidate_action != {
        "critic_observation_time": "before_candidate_execution",
        "planned_action_horizon": 5,
        "action_mask_source": "planned_first_chunk_not_executed_count",
        "executed_action_count_used_for_action_mask": False,
        "executed_action_count_used_for_sim_time_accounting_only": True,
        "planner_status_fail_is_a_valid_action_outcome": True,
        "python_execution_exception_invalidates_complete_decision": True,
    }:
        raise FiveBodyContractError(
            f"{expected_body} censors planned candidates after execution"
        )
    if (
        value.get("candidate_noise_contract") != CANDIDATE_NOISE_CONTRACT
        or value.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
        or value.get("terminal_supervision_contract")
        != TERMINAL_SUPERVISION_CONTRACT
        or value.get("event_age_contract") != EVENT_AGE_CONTRACT
        or value.get("terminal_horizon_contract") != TERMINAL_HORIZON_CONTRACT
        or value.get("branch_root_snapshot_contract")
        != BRANCH_ROOT_SNAPSHOT_CONTRACT
        or value.get("branch_diagnostic_contract")
        != BRANCH_DIAGNOSTIC_CONTRACT
    ):
        raise FiveBodyContractError(
            f"{expected_body} candidate/object/terminal/diagnostic semantics changed"
        )
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) < 4:
        raise FiveBodyContractError(f"{expected_body} needs at least four canonical groups")
    identities: set[str] = set()
    conditions: set[str] = set()
    normalized = []
    for item in groups:
        if not isinstance(item, Mapping):
            raise FiveBodyContractError("canonical group entry must be an object")
        group_id = item.get("group_id")
        condition = item.get("condition")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in identities
            or condition not in CONDITIONS
            or isinstance(item.get("requested_seed"), bool)
            or not isinstance(item.get("requested_seed"), int)
            or not _is_sha(item.get("sha256"))
            or not _is_sha(item.get("branch_root_snapshot_sha256"))
            or not _is_sha(item.get("branch_root_restorable_snapshot_sha256"))
            or not _is_sha(item.get("canonical_root_snapshot_sha256"))
            or item.get("diagnostic_format")
            != BRANCH_DIAGNOSTIC_CONTRACT["format"]
            or not _is_sha(item.get("diagnostics_sha256"))
        ):
            raise FiveBodyContractError(f"invalid canonical group for {expected_body}")
        # Manifest validation is deliberately payload-blind.  In particular,
        # a later-held-out body must not even have its group file stat'ed or
        # hashed.  Source groups are resolved, stat'ed and hashed only at the
        # source payload boundary in ``_npz_rows``.
        path = _lexical_contained_payload_path(
            manifest_dir, str(item.get("path", "")), "group"
        )
        diagnostics_path = _lexical_contained_payload_path(
            manifest_dir, str(item.get("diagnostics_path", "")), "group diagnostics"
        )
        identities.add(group_id)
        conditions.add(condition)
        normalized.append(
            {
                **dict(item),
                "resolved_path": str(path),
                "resolved_diagnostics_path": str(diagnostics_path),
            }
        )
    if conditions != set(CONDITIONS):
        raise FiveBodyContractError(f"{expected_body} lacks clean/randomized groups")
    return {
        "body": expected_body,
        "groups": normalized,
        "group_identity_sha256": canonical_sha256(sorted(identities)),
        "adapter_sha256": adapter["implementation_sha256"],
        "event_derivation_implementation_sha256": value[
            "event_derivation_implementation_sha256"
        ],
    }


def load_binding(path: Path, expected_sha256: str) -> dict[str, Any]:
    binding = _read_bound_json(path, expected_sha256, "training binding")
    _verify_signed(binding, "training binding")
    if (
        binding.get("format") != BINDING_FORMAT
        or binding.get("dataset_repo") != DATASET_REPO
        or binding.get("dataset_revision") != DATASET_REVISION
        or binding.get("task") != TASK
        or binding.get("instruction") != DEFAULT_INSTRUCTION
        or binding.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or binding.get("candidate_noise_contract") != CANDIDATE_NOISE_CONTRACT
        or binding.get("terminal_supervision_contract")
        != TERMINAL_SUPERVISION_CONTRACT
        or binding.get("event_age_contract") != EVENT_AGE_CONTRACT
        or binding.get("terminal_horizon_contract") != TERMINAL_HORIZON_CONTRACT
        or binding.get("branch_root_snapshot_contract")
        != BRANCH_ROOT_SNAPSHOT_CONTRACT
        or binding.get("object_effect_schema") != OBJECT_EFFECT_SCHEMA
        or binding.get("branch_diagnostic_contract")
        != BRANCH_DIAGNOSTIC_CONTRACT
        or binding.get("heldout_labels_may_train_fit_calibrate_or_select") is not False
        or binding.get("canonical_shared_body_rows") != 1
    ):
        raise FiveBodyContractError("training binding violates strict LOBO")
    authority = binding.get("execution_authority")
    if authority != {
        "explicit_user_training_request_recorded": True,
        "public_data_only": True,
        "protected_internal_data_allowed": False,
        "remote_cuda_only": True,
    }:
        raise FiveBodyContractError("training binding lacks explicit public remote authority")
    root = path.expanduser().resolve().parent
    materialization_path = _resolve_contained(
        root, str(binding.get("materialization_receipt", {}).get("path", "")), "materialization"
    )
    materialization = _read_bound_json(
        materialization_path,
        str(binding.get("materialization_receipt", {}).get("sha256", "")),
        "materialization receipt",
    )
    materialization_audit = validate_materialization_receipt(materialization)
    actor_path = _resolve_contained(
        root, str(binding.get("actor_authority", {}).get("path", "")), "actor authority"
    )
    actors = _read_bound_json(
        actor_path,
        str(binding.get("actor_authority", {}).get("sha256", "")),
        "actor authority",
    )
    actor_audit = validate_actor_authority(actors, authority_dir=actor_path.parent)
    body_bindings = binding.get("body_manifests")
    if not isinstance(body_bindings, Mapping) or set(body_bindings) != set(BODIES):
        raise FiveBodyContractError("binding must contain exactly five body manifests")
    manifests = {}
    for body in BODIES:
        item = body_bindings[body]
        if not isinstance(item, Mapping):
            raise FiveBodyContractError(f"body manifest binding missing for {body}")
        manifest_path = _resolve_contained(root, str(item.get("path", "")), "body manifest")
        manifest = _read_bound_json(
            manifest_path, str(item.get("sha256", "")), f"{body} body manifest"
        )
        manifests[body] = validate_body_manifest(
            manifest, expected_body=body, manifest_dir=manifest_path.parent
        )
    event_implementations = {
        item["event_derivation_implementation_sha256"] for item in manifests.values()
    }
    if len(event_implementations) != 1:
        raise FiveBodyContractError(
            "five bodies do not share one analytic event implementation"
        )
    return {
        "binding": binding,
        "binding_file_sha256": expected_sha256.lower(),
        "materialization": materialization_audit,
        "actor": actor_audit,
        "manifests": manifests,
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": event_implementations.pop(),
    }


def _declared_root_query_index(group: Mapping[str, Any]) -> int | None:
    value = group.get("root_query_index")
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 40:
        return int(value)
    identity = str(group.get("group_id", ""))
    marker = "|query="
    if marker not in identity:
        return None
    suffix = identity.rsplit(marker, 1)[-1]
    try:
        parsed = int(suffix)
    except ValueError:
        return None
    return parsed if 0 <= parsed < 40 else None


def _horizon_covering_validation_seeds(
    by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    ordered_seeds: Sequence[int],
    target_count: int,
) -> set[int]:
    """Choose label-blind validation seeds covering every formal horizon.

    Formal collection assigns different seed blocks to different query
    budgets.  A plain global 20% seed sample can therefore omit a remaining
    action budget entirely.  Greedy set cover uses only manifest query indices
    and the preregistered hash order; it also leaves at least one training seed
    for every query.  Synthetic/legacy manifests without the formal query
    declaration retain the old hash-only split.
    """

    if target_count <= 0 or target_count >= len(ordered_seeds):
        raise FiveBodyContractError("validation seed target cannot form two lanes")
    query_by_seed = {
        seed: {
            query
            for group in by_seed[seed]
            if (query := _declared_root_query_index(group)) is not None
        }
        for seed in ordered_seeds
    }
    formal_queries = set(range(40))
    if set().union(*query_by_seed.values()) != formal_queries:
        return set(ordered_seeds[:target_count])

    support = {
        query: {seed for seed, queries in query_by_seed.items() if query in queries}
        for query in formal_queries
    }
    if any(len(seeds) < 2 for seeds in support.values()):
        raise FiveBodyContractError(
            "formal source split needs train/validation seed support at every query"
        )
    priority = {seed: index for index, seed in enumerate(ordered_seeds)}
    selected: set[int] = set()
    validation_coverage: set[int] = set()

    def preserves_training_coverage(seed: int) -> bool:
        proposed = selected | {seed}
        return all(bool(seeds - proposed) for seeds in support.values())

    while validation_coverage != formal_queries:
        candidates = [
            seed
            for seed in ordered_seeds
            if seed not in selected
            and bool(query_by_seed[seed] - validation_coverage)
            and preserves_training_coverage(seed)
        ]
        if not candidates:
            raise FiveBodyContractError(
                "cannot cover every validation query while retaining training coverage"
            )
        chosen = min(
            candidates,
            key=lambda seed: (
                -len(query_by_seed[seed] - validation_coverage),
                priority[seed],
            ),
        )
        selected.add(chosen)
        validation_coverage.update(query_by_seed[chosen])

    desired = max(target_count, len(selected))
    for seed in ordered_seeds:
        if len(selected) >= desired:
            break
        if seed not in selected and preserves_training_coverage(seed):
            selected.add(seed)
    if len(selected) < target_count:
        raise FiveBodyContractError(
            "formal horizon-covering split could not reach its validation fraction"
        )
    training_coverage = set().union(
        *(query_by_seed[seed] for seed in ordered_seeds if seed not in selected)
    )
    if validation_coverage != formal_queries or training_coverage != formal_queries:
        raise FiveBodyContractError("formal source split lost horizon coverage")
    return selected


def source_group_split(
    audit: Mapping[str, Any], *, held_out_body: str, split_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if held_out_body not in BODIES:
        raise FiveBodyContractError(f"unknown held-out body {held_out_body!r}")
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for body in BODIES:
        groups = list(audit["manifests"][body]["groups"])
        if body == held_out_body:
            heldout.extend(groups)
            continue
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            by_condition[str(group["condition"])].append(group)
        for condition in CONDITIONS:
            by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for group in by_condition[condition]:
                by_seed[int(group["requested_seed"])].append(group)
            ordered_seeds = sorted(
                by_seed,
                key=lambda seed: hashlib.sha256(
                    f"{split_seed}|{body}|{condition}|{seed}".encode()
                ).hexdigest(),
            )
            if len(ordered_seeds) < 2:
                raise FiveBodyContractError(
                    f"{body}/{condition} cannot form seed-disjoint source validation"
                )
            validation_seed_count = max(1, int(round(0.2 * len(ordered_seeds))))
            validation_seeds = _horizon_covering_validation_seeds(
                by_seed,
                ordered_seeds=ordered_seeds,
                target_count=validation_seed_count,
            )
            for seed in ordered_seeds:
                target = validation if seed in validation_seeds else training
                target.extend(
                    {**row, "body": body}
                    for row in sorted(by_seed[seed], key=lambda item: item["group_id"])
                )
    if not training or not validation or not heldout:
        raise FiveBodyContractError("LOBO split contains an empty lane")
    return training, validation, heldout


def build_preflight_receipt(
    audit: Mapping[str, Any], *, held_out_body: str, split_seed: int
) -> dict[str, Any]:
    training, validation, heldout = source_group_split(
        audit, held_out_body=held_out_body, split_seed=split_seed
    )

    def identity(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        return sorted(f"{row.get('body', held_out_body)}|{row['group_id']}" for row in rows)

    def query_coverage(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[int]]:
        result: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            query = _declared_root_query_index(row)
            if query is not None:
                result[f"{row['body']}|{row['condition']}"].add(query)
        return {key: sorted(values) for key, values in sorted(result.items())}

    training_query_coverage = query_coverage(training)
    validation_query_coverage = query_coverage(validation)
    expected_units = {
        f"{body}|{condition}"
        for body in BODIES
        if body != held_out_body
        for condition in CONDITIONS
    }
    formal_horizon_coverage = bool(training_query_coverage) or bool(
        validation_query_coverage
    )
    if formal_horizon_coverage and (
        set(training_query_coverage) != expected_units
        or set(validation_query_coverage) != expected_units
        or any(values != list(range(40)) for values in training_query_coverage.values())
        or any(values != list(range(40)) for values in validation_query_coverage.values())
    ):
        raise FiveBodyContractError(
            "source train/validation split does not cover every formal query"
        )
    receipt = {
        "format": FORMAT,
        "status": "preflight_passed_payloads_still_unopened",
        "dataset_revision": DATASET_REVISION,
        "event_spec_sha256": audit["event_spec_sha256"],
        "event_derivation_implementation_sha256": audit[
            "event_derivation_implementation_sha256"
        ],
        "binding_file_sha256": audit["binding_file_sha256"],
        "held_out_body": held_out_body,
        "source_bodies": [body for body in BODIES if body != held_out_body],
        "split_seed": split_seed,
        "source_train_groups": len(training),
        "source_validation_groups": len(validation),
        "heldout_groups_deferred": len(heldout),
        "source_train_identity_sha256": canonical_sha256(identity(training)),
        "source_validation_identity_sha256": canonical_sha256(identity(validation)),
        "heldout_identity_sha256": canonical_sha256(
            sorted(f"{held_out_body}|{row['group_id']}" for row in heldout)
        ),
        "assignment_uses_labels": False,
        "split_unit": "body_condition_requested_seed_all_queries",
        "validation_seed_selection": (
            "label_blind_hash_ordered_greedy_full_horizon_cover_then_fill_20pct"
            if formal_horizon_coverage
            else "label_blind_hash_ordered_20pct_legacy_manifest"
        ),
        "formal_horizon_coverage_enforced": formal_horizon_coverage,
        "source_train_query_indices_by_body_condition": training_query_coverage,
        "source_validation_query_indices_by_body_condition": (
            validation_query_coverage
        ),
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "model_body_rows": 1,
        "heldout_specific_trainable_parameters": 0,
        "actor_frozen": True,
        "same_ordered_candidate_set_required": True,
        "candidate_zero_is_actor_baseline": True,
        "task_success_evaluation_authorized": False,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return receipt


def _npz_rows(group: Mapping[str, Any], *, body: str) -> list[dict[str, Any]]:
    path = Path(str(group["resolved_path"]))
    if not path.is_file() or sha256_file(path) != group["sha256"]:
        raise FiveBodyContractError(f"source canonical group missing/tampered: {path}")
    with np.load(path, allow_pickle=False) as values:
        observed_arrays = set(values.files)
        if observed_arrays != REQUIRED_ARRAYS:
            raise FiveBodyContractError(
                f"{path} arrays mismatch: missing={sorted(REQUIRED_ARRAYS-observed_arrays)}, "
                f"extra={sorted(observed_arrays-REQUIRED_ARRAYS)}"
            )
        arrays = {name: np.asarray(values[name]) for name in values.files}
    count = len(arrays["state"])
    if count != CANDIDATE_COUNT:
        raise FiveBodyContractError(
            f"{path} must contain one complete four-candidate decision"
        )
    shapes = {
        "state": (count, core.STATE_DIM),
        "actions": (count, arrays["actions"].shape[1], core.ACTION_DIM),
        "action_mask": (count, arrays["actions"].shape[1]),
        "object_delta": (count, core.OBJECT_DELTA_DIM),
    }
    for name, expected in shapes.items():
        if arrays[name].shape != expected:
            raise FiveBodyContractError(f"{path} array {name} shape mismatch")
    scalar = set(arrays) - set(shapes)
    if any(arrays[name].shape != (count,) for name in scalar):
        raise FiveBodyContractError(f"{path} scalar supervision array shape mismatch")
    if count == 0 or any(not np.isfinite(value).all() for value in arrays.values()):
        raise FiveBodyContractError(f"{path} contains empty/non-finite canonical rows")
    if np.any((arrays["current_event_id"] < 0) | (arrays["current_event_id"] >= 5)):
        raise FiveBodyContractError(f"{path} current event ids are invalid")
    terminal_event = arrays["terminal_max_event_id"]
    terminal_event_mask = arrays["terminal_event_mask"]
    terminal_goal_mask = arrays["terminal_goal_progress_mask"]
    if (
        not np.array_equal(terminal_event, terminal_event.astype(np.int64))
        or np.any((terminal_event < 0) | (terminal_event >= 5))
        or np.any(terminal_event < arrays["current_event_id"])
        or np.any((terminal_event_mask < 0.0) | (terminal_event_mask > 1.0))
        or np.any((terminal_goal_mask < 0.0) | (terminal_goal_mask > 1.0))
    ):
        raise FiveBodyContractError(f"{path} terminal max event ids are invalid")
    terminal_stage_progress = arrays["terminal_stage_progress"]
    if np.any((terminal_stage_progress < 0.0) | (terminal_stage_progress > 1.0)):
        raise FiveBodyContractError(f"{path} terminal stage progress is outside [0,1]")
    if not np.allclose(
        terminal_stage_progress,
        np.where(
            arrays["success"] > 0.5,
            1.0,
            terminal_event.astype(np.float32) / 4.0,
        ),
        atol=1e-6,
        rtol=0.0,
    ):
        raise FiveBodyContractError(
            f"{path} terminal max event and stage progress disagree"
        )
    terminal_goal_distance = arrays["terminal_goal_distance"]
    if np.any(terminal_goal_distance < 0.0):
        raise FiveBodyContractError(f"{path} terminal goal distance is negative")
    initial_goal_distance = np.linalg.norm(arrays["state"][:, 0:3], axis=-1)
    if not np.allclose(
        arrays["terminal_goal_progress"],
        initial_goal_distance - terminal_goal_distance,
        atol=2e-5,
        rtol=0.0,
    ):
        raise FiveBodyContractError(
            f"{path} terminal goal progress disagrees with state and terminal distance"
        )
    expected_event_onehot = np.zeros((count, 5), dtype=np.float32)
    expected_event_onehot[np.arange(count), arrays["current_event_id"].astype(int)] = 1.0
    if not np.array_equal(
        np.asarray(arrays["state"][:, 18:23], dtype=np.float32),
        expected_event_onehot,
    ):
        raise FiveBodyContractError(
            f"{path} state event onehot disagrees with current_event_id"
        )
    if np.any(arrays["dt"] <= 0):
        raise FiveBodyContractError(f"{path} contains non-positive planned dt")
    if not np.allclose(
        arrays["dt"], 5.0 / SOURCE_EVENT_SAMPLING_HZ, atol=1e-6, rtol=0.0
    ):
        raise FiveBodyContractError(f"{path} planned dt is not fixed 5/15 seconds")
    if np.any(arrays["event_age_seconds"] < 0.0) or not np.allclose(
        arrays["event_age_seconds"],
        arrays["event_age_seconds"][:1],
        atol=1e-6,
        rtol=0.0,
    ):
        raise FiveBodyContractError(
            f"{path} candidates do not share one non-negative pre-action event age"
        )
    if np.any(arrays["remaining_action_budget"] <= 0.0) or not np.allclose(
        arrays["remaining_action_budget"],
        arrays["remaining_action_budget"][:1],
        atol=0.0,
        rtol=0.0,
    ):
        raise FiveBodyContractError(
            f"{path} candidates do not share one positive remaining action budget"
        )
    if not np.all(np.isin(
        arrays["remaining_action_budget"],
        TERMINAL_HORIZON_CONTRACT["development_remaining_action_budgets"],
    )):
        raise FiveBodyContractError(
            f"{path} remaining action budget is outside the formal query grid"
        )
    stop_reason = arrays["terminal_stop_reason_id"]
    if not np.array_equal(stop_reason, stop_reason.astype(np.int64)) or np.any(
        (stop_reason < 0) | (stop_reason > 1)
    ):
        raise FiveBodyContractError(f"{path} terminal stop reason is invalid")
    if np.any(arrays["duration"] < 0):
        raise FiveBodyContractError(f"{path} contains invalid simulator duration")
    expected_duration_mask = (
        (arrays["duration"] > 0.0)
        & (
            (arrays["duration_observed"] > 0.5)
            | np.isin(arrays["terminal_stop_reason_id"], [0, 1])
        )
    )
    if not np.array_equal(
        np.asarray(arrays["duration_mask"] > 0.5), expected_duration_mask
    ):
        raise FiveBodyContractError(
            f"{path} duration mask mixes execution-error censoring with "
            "administrative finite-horizon censoring"
        )
    horizon = arrays["actions"].shape[1]
    planned_mask = np.arange(horizon) < 5
    if horizon < 5 or not np.array_equal(
        np.asarray(arrays["action_mask"], dtype=bool),
        np.repeat(planned_mask[None], CANDIDATE_COUNT, axis=0),
    ):
        raise FiveBodyContractError(
            f"{path} action mask does not expose the full planned first chunk"
        )
    if not np.array_equal(arrays["candidate_index"], np.arange(CANDIDATE_COUNT)):
        raise FiveBodyContractError(f"{path} candidate order is not [0,1,2,3]")
    if not np.array_equal(
        arrays["state"], np.repeat(arrays["state"][:1], CANDIDATE_COUNT, axis=0)
    ) or not np.all(arrays["current_event_id"] == arrays["current_event_id"][0]):
        raise FiveBodyContractError(
            f"{path} candidates do not share one root state/event"
        )
    rows = []
    for index in range(count):
        row = {name: arrays[name][index] for name in arrays}
        row.update(
            {
                "action_available": np.float32(1.0),
                "action_schema_id": np.int64(0),
                "logical_group": f"{body}|{group['condition']}|{group['group_id']}",
                "body": body,
                "policy": "frozen_native_actor",
                "task": TASK,
            }
        )
        rows.append(row)
    return rows


class CompleteDecisionBatchSampler:
    """Shuffle decisions while keeping every four-candidate set intact."""

    def __init__(
        self, rows: Sequence[Mapping[str, Any]], *, batch_size: int, seed: int
    ) -> None:
        if batch_size < CANDIDATE_COUNT:
            raise FiveBodyContractError("batch size must fit one complete decision")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["logical_group"])].append(index)
        for group, indices in grouped.items():
            ordered = sorted(indices, key=lambda index: int(rows[index]["candidate_index"]))
            if len(ordered) != CANDIDATE_COUNT or [
                int(rows[index]["candidate_index"]) for index in ordered
            ] != list(range(CANDIDATE_COUNT)):
                raise FiveBodyContractError(f"incomplete candidate decision {group}")
            grouped[group] = ordered
        self.decisions = [grouped[name] for name in sorted(grouped)]
        self.decisions_per_batch = max(1, batch_size // CANDIDATE_COUNT)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        decisions = list(self.decisions)
        generator.shuffle(decisions)
        for start in range(0, len(decisions), self.decisions_per_batch):
            yield [
                index
                for decision in decisions[start : start + self.decisions_per_batch]
                for index in decision
            ]

    def __len__(self) -> int:
        return (len(self.decisions) + self.decisions_per_batch - 1) // self.decisions_per_batch


class MacroBalancedRankDecisionBatchSampler:
    """Build rank-only batches with sparse success comparisons in every batch.

    Proper likelihoods use :class:`CompleteDecisionBatchSampler` and therefore
    retain the empirical source distribution.  This sampler is used only by
    the candidate-rank objective.  It alternates body/condition/current-event
    strata, reserves half of each batch for mixed-success decisions when that
    many distinct groups exist, and never repeats a logical decision within a
    batch.  Sparse mixed decisions may be revisited across batches; that is the
    intended oversampling needed to prevent dense failures from drowning the
    success-changing supervision.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        batch_size: int,
        seed: int,
        positive_group_weight: Mapping[str, float],
        ablation_variant: str,
    ) -> None:
        if batch_size < CANDIDATE_COUNT:
            raise FiveBodyContractError("rank batch must fit one complete decision")
        if ablation_variant not in ABLATION_VARIANTS:
            raise FiveBodyContractError(
                f"unknown ablation variant {ablation_variant!r}"
            )
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["logical_group"])].append(index)

        decisions: dict[str, list[int]] = {}
        kinds: dict[str, str] = {}
        strata: dict[str, tuple[str, str, int]] = {}
        for group, indices in sorted(grouped.items()):
            ordered = sorted(indices, key=lambda index: int(rows[index]["candidate_index"]))
            if len(ordered) != CANDIDATE_COUNT or [
                int(rows[index]["candidate_index"]) for index in ordered
            ] != list(range(CANDIDATE_COUNT)):
                raise FiveBodyContractError(f"incomplete rank decision {group}")
            group_weights = {
                float(positive_group_weight.get(group, 0.0)) for _index in ordered
            }
            if (
                len(group_weights) != 1
                or not all(math.isfinite(value) and value >= 0.0 for value in group_weights)
            ):
                raise FiveBodyContractError("rank bootstrap weight must be finite/group-constant")
            if next(iter(group_weights)) <= 0.0:
                continue
            if not all(bool(rows[index]["success_mask"]) for index in ordered):
                raise FiveBodyContractError("rank decision lacks complete success supervision")
            outcomes = {float(rows[index]["success"]) for index in ordered}
            if outcomes == {0.0, 1.0}:
                kind = "mixed"
            elif outcomes == {0.0}:
                kind = "dense"
            elif outcomes == {1.0}:
                continue
            else:
                raise FiveBodyContractError("rank decision has non-binary success labels")
            identity = group.split("|", 2)
            body = str(rows[ordered[0]]["body"])
            current_events = {int(rows[index]["current_event_id"]) for index in ordered}
            if (
                len(identity) != 3
                or identity[0] != body
                or identity[1] not in CONDITIONS
                or len(current_events) != 1
            ):
                raise FiveBodyContractError("rank macro stratum identity changed")
            if kind == "dense":
                if not all(
                    bool(rows[index]["terminal_event_mask"])
                    and bool(rows[index]["terminal_goal_progress_mask"])
                    for index in ordered
                ):
                    raise FiveBodyContractError(
                        "dense rank decision lacks complete terminal supervision"
                    )
                if not _dense_rank_labels_are_orderable(
                    [rows[index]["terminal_max_event_id"] for index in ordered],
                    [rows[index]["terminal_goal_progress"] for index in ordered],
                    ablation_variant=ablation_variant,
                ):
                    continue
            decisions[group] = ordered
            kinds[group] = kind
            strata[group] = (body, identity[1], current_events.pop())

        self.decisions = decisions
        self.kinds = kinds
        self.strata = strata
        self.mixed_groups = sorted(group for group, kind in kinds.items() if kind == "mixed")
        self.dense_groups = sorted(group for group, kind in kinds.items() if kind == "dense")
        if not self.mixed_groups:
            raise FiveBodyContractError(
                "balanced rank sampler requires a positive-weight mixed-success decision"
            )
        self.decisions_per_batch = max(1, batch_size // CANDIDATE_COUNT)
        self.batch_count = max(
            1,
            math.ceil(len(self.decisions) / self.decisions_per_batch),
        )
        self.seed = int(seed)
        self.epoch = 0

    @staticmethod
    def _stratified_cycler(
        groups: Sequence[str],
        strata: Mapping[str, tuple[str, str, int]],
        generator: random.Random,
    ) -> Iterator[str]:
        by_stratum: dict[tuple[str, str, int], list[str]] = defaultdict(list)
        for group in groups:
            by_stratum[strata[group]].append(group)
        ordered_strata = sorted(by_stratum)
        generator.shuffle(ordered_strata)
        cursors = {stratum: 0 for stratum in ordered_strata}
        for values in by_stratum.values():
            generator.shuffle(values)
        stratum_cursor = 0
        while ordered_strata:
            stratum = ordered_strata[stratum_cursor % len(ordered_strata)]
            values = by_stratum[stratum]
            cursor = cursors[stratum]
            if cursor and cursor % len(values) == 0:
                generator.shuffle(values)
            yield values[cursor % len(values)]
            cursors[stratum] = cursor + 1
            stratum_cursor += 1

    @staticmethod
    def _draw_distinct(
        cycler: Iterator[str],
        *,
        requested: int,
        available_count: int,
        used: set[str],
    ) -> list[str]:
        selected: list[str] = []
        target = min(int(requested), max(0, int(available_count)))
        attempts = 0
        maximum_attempts = max(1, available_count * 4 + target * 4)
        while len(selected) < target and attempts < maximum_attempts:
            group = next(cycler)
            attempts += 1
            if group in used:
                continue
            used.add(group)
            selected.append(group)
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        mixed = self._stratified_cycler(self.mixed_groups, self.strata, generator)
        dense = (
            self._stratified_cycler(self.dense_groups, self.strata, generator)
            if self.dense_groups
            else None
        )
        mixed_target = max(1, self.decisions_per_batch // 2)
        for _batch in range(self.batch_count):
            used: set[str] = set()
            selected = self._draw_distinct(
                mixed,
                requested=mixed_target,
                available_count=len(self.mixed_groups),
                used=used,
            )
            if dense is not None:
                selected.extend(
                    self._draw_distinct(
                        dense,
                        requested=self.decisions_per_batch - len(selected),
                        available_count=len(self.dense_groups),
                        used=used,
                    )
                )
            selected.extend(
                self._draw_distinct(
                    mixed,
                    requested=self.decisions_per_batch - len(selected),
                    available_count=len(self.mixed_groups),
                    used=used,
                )
            )
            if not selected or not any(self.kinds[group] == "mixed" for group in selected):
                raise FiveBodyContractError("rank batch lost mixed-success supervision")
            generator.shuffle(selected)
            yield [index for group in selected for index in self.decisions[group]]

    def __len__(self) -> int:
        return self.batch_count


class EffectAlignedSharedEventHead(core.MultibodyCanonicalEventWorldModel):
    """Shared event model with a scalar head trained for best-of-four choice."""

    def __init__(self, ablation_variant: str = "full") -> None:
        if ablation_variant not in ABLATION_VARIANTS:
            raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
        super().__init__(core.ModelConfig(body_count=1, action_schema_count=1))
        self.ablation_variant = ablation_variant
        # The base class exposes horizon-free terminal outcome heads.  Branch
        # success and recovery are finite-horizon labels here, so those heads
        # are frozen and replaced below by horizon-coherent predictions.
        self.success.requires_grad_(False)
        self.recovery.requires_grad_(False)
        self.register_buffer("state_mean", torch.zeros(core.STATE_DIM))
        self.register_buffer("state_std", torch.ones(core.STATE_DIM))
        self.event_age_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, self.config.clock_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.config.clock_dim, self.config.clock_dim),
        )
        self.terminal_context_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, core.SEMANTIC_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(core.SEMANTIC_DIM, core.SEMANTIC_DIM),
        )
        self.terminal_event = torch.nn.Linear(core.SEMANTIC_DIM, 5)
        self.terminal_recovery = torch.nn.Linear(core.SEMANTIC_DIM, 1)
        self.terminal_goal_progress_mean = torch.nn.Linear(core.SEMANTIC_DIM, 1)
        self.terminal_goal_progress_scale = torch.nn.Linear(core.SEMANTIC_DIM, 1)
        self.candidate_rank = torch.nn.Sequential(
            torch.nn.LayerNorm(CANDIDATE_RANK_FEATURE_DIM),
            torch.nn.Linear(CANDIDATE_RANK_FEATURE_DIM, core.SEMANTIC_DIM // 2),
            torch.nn.GELU(),
            torch.nn.Linear(core.SEMANTIC_DIM // 2, 1),
        )

    @torch.no_grad()
    def set_state_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != (core.STATE_DIM,) or std.shape != (core.STATE_DIM,):
            raise FiveBodyContractError("state normalization must be 27-D")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise FiveBodyContractError("state normalization contains non-finite values")
        if bool((std < 1e-4).any()):
            raise FiveBodyContractError("state normalization std is below the floor")
        self.state_mean.copy_(mean.to(self.state_mean))
        self.state_std.copy_(std.to(self.state_std))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        normalized_batch = dict(batch)
        normalized_batch["state"] = (
            batch["state"] - self.state_mean
        ) / self.state_std
        output = super().forward(normalized_batch)

        event_age = batch.get("event_age_seconds")
        if (
            not isinstance(event_age, torch.Tensor)
            or event_age.shape != output["success_logit"].shape
            or not bool(torch.isfinite(event_age).all())
            or bool((event_age < 0.0).any())
        ):
            raise FiveBodyContractError(
                "shared head requires one finite non-negative event_age_seconds per row"
            )
        age_clock_hidden = output["clock_hidden"] + self.event_age_encoder(
            torch.log1p(event_age.to(output["clock_hidden"]))[:, None]
        )
        duration_log_mean = self.duration_mean(age_clock_hidden)
        duration_log_scale = torch.clamp(
            self.duration_scale(age_clock_hidden), -5.0, 2.0
        )
        current = batch["current_event_id"].long()[:, None]
        output["clock_hidden"] = age_clock_hidden
        output["duration_log_mean"] = duration_log_mean
        output["duration_log_scale"] = duration_log_scale
        output["duration_selected_log_mean"] = duration_log_mean.gather(
            1, current
        ).squeeze(1)
        output["duration_selected_log_scale"] = duration_log_scale.gather(
            1, current
        ).squeeze(1)

        remaining_budget = batch.get("remaining_action_budget")
        if (
            not isinstance(remaining_budget, torch.Tensor)
            or remaining_budget.shape != event_age.shape
            or not bool(torch.isfinite(remaining_budget).all())
            or bool((remaining_budget <= 0.0).any())
        ):
            raise FiveBodyContractError(
                "terminal consequence heads require one finite positive "
                "remaining_action_budget per row"
            )
        terminal_context = torch.stack(
            (
                torch.log1p(event_age.to(output["transitioned"])),
                torch.log1p(remaining_budget.to(output["transitioned"])),
            ),
            dim=-1,
        )
        if self.ablation_variant == "no_time_duration":
            terminal_context = torch.zeros_like(terminal_context)
        terminal_hidden = output["transitioned"] + self.terminal_context_encoder(
            terminal_context
        )
        terminal_event_logits = self.terminal_event(terminal_hidden)
        event_levels = torch.arange(
            terminal_event_logits.shape[-1], device=terminal_event_logits.device
        )[None]
        terminal_event_logits = terminal_event_logits.masked_fill(
            event_levels < current, -1e4
        )
        # eK is the canonical success event.  Defining success as its exact
        # categorical probability makes the two proper predictions coherent
        # and, unlike the old base head, conditions success on remaining time.
        terminal_success_logit = terminal_event_logits[:, -1] - torch.logsumexp(
            terminal_event_logits[:, :-1], dim=-1
        )
        terminal_recovery_logit = self.terminal_recovery(terminal_hidden).squeeze(-1)
        terminal_goal_progress_mean = self.terminal_goal_progress_mean(
            terminal_hidden
        ).squeeze(-1)
        terminal_goal_progress_log_scale = torch.clamp(
            self.terminal_goal_progress_scale(terminal_hidden).squeeze(-1),
            -7.0,
            2.0,
        )
        output["terminal_event_logits"] = terminal_event_logits
        output["success_logit"] = terminal_success_logit
        output["recovery_logit"] = terminal_recovery_logit
        output["terminal_goal_progress_mean"] = terminal_goal_progress_mean
        output["terminal_goal_progress_log_scale"] = (
            terminal_goal_progress_log_scale
        )

        # The deployed score is deliberately a function of predicted
        # consequences, never a free projection of ``transitioned`` or
        # ``clock_hidden``.  Detaching the complete feature vector keeps the
        # proper event/outcome/effect likelihoods calibrated: listwise rank
        # supervision learns how to combine their predictions, not how to
        # rewrite them into an unconstrained latent critic.
        post_probability = torch.softmax(output["post_event_logits"], dim=-1)
        next_probability = torch.softmax(output["next_event_logits"], dim=-1)
        success_probability = torch.sigmoid(output["success_logit"])[:, None]
        # Recovery is conditional on an operational regression.  Compose the
        # conditional head with the predicted post-event distribution so the
        # deployed feature is the identifiable joint consequence.  At e0 the
        # regression and joint-recovery probabilities are exactly zero.
        event_index = torch.arange(
            post_probability.shape[-1], device=post_probability.device
        )[None]
        regression_probability = (
            post_probability * (event_index < current).to(post_probability)
        ).sum(dim=-1, keepdim=True)
        joint_recovery_probability = regression_probability * torch.sigmoid(
            output["recovery_logit"]
        )[:, None]
        output["regression_probability"] = regression_probability.squeeze(-1)
        output["joint_recovery_probability"] = (
            joint_recovery_probability.squeeze(-1)
        )
        duration_features = torch.stack(
            (
                output["duration_selected_log_mean"],
                torch.exp(output["duration_selected_log_scale"]),
            ),
            dim=-1,
        )
        object_mean = output["object_delta_mean"]
        object_scale = torch.exp(output["object_delta_log_scale"])
        state = batch["state"]
        if state.ndim != 2 or state.shape[-1] != core.STATE_DIM:
            raise FiveBodyContractError(
                "consequence rank utility requires one canonical 27-D root state"
            )
        relative_goal = state[:, :3].to(object_mean)
        predicted_remaining = relative_goal - object_mean[:, :3]
        current_distance = torch.linalg.vector_norm(relative_goal, dim=-1)
        predicted_distance = torch.linalg.vector_norm(predicted_remaining, dim=-1)
        predicted_goal_progress = current_distance - predicted_distance

        # Delta-method radial uncertainty for the Student-t(3) translation
        # effect.  At a zero predicted residual, fall back to the current goal
        # direction; if both vectors are zero, use the isotropic RMS scale.
        epsilon = torch.finfo(object_mean.dtype).eps**0.5
        remaining_unit = predicted_remaining / predicted_distance[:, None].clamp_min(
            epsilon
        )
        current_unit = relative_goal / current_distance[:, None].clamp_min(epsilon)
        has_remaining_direction = predicted_distance > epsilon
        direction = torch.where(
            has_remaining_direction[:, None], remaining_unit, current_unit
        )
        has_direction = has_remaining_direction | (current_distance > epsilon)
        translation_variance = object_scale[:, :3].square()
        projected_variance = (direction.square() * translation_variance).sum(dim=-1)
        isotropic_variance = translation_variance.mean(dim=-1)
        radial_variance = torch.where(
            has_direction, projected_variance, isotropic_variance
        )
        predicted_goal_progress_uncertainty = torch.sqrt(
            OBJECT_STUDENT_T_DOF * radial_variance.clamp_min(0.0)
        )

        object_features = torch.cat(
            (
                object_mean,
                object_scale,
                predicted_goal_progress[:, None],
                predicted_goal_progress_uncertainty[:, None],
            ),
            dim=-1,
        )
        terminal_event_probability = torch.softmax(terminal_event_logits, dim=-1)
        terminal_progress_features = torch.stack(
            (
                terminal_goal_progress_mean,
                torch.exp(terminal_goal_progress_log_scale),
            ),
            dim=-1,
        )
        if self.ablation_variant == "no_time_duration":
            duration_features = torch.zeros_like(duration_features)
        if self.ablation_variant == "no_object_effect":
            object_features = torch.zeros_like(object_features)
            terminal_progress_features = torch.zeros_like(
                terminal_progress_features
            )
        rank_features = torch.cat(
            (
                post_probability,
                next_probability,
                success_probability,
                regression_probability,
                joint_recovery_probability,
                duration_features,
                object_features,
                terminal_event_probability,
                terminal_progress_features,
            ),
            dim=-1,
        ).detach()
        if rank_features.shape != (
            output["success_logit"].shape[0],
            CANDIDATE_RANK_FEATURE_DIM,
        ):
            raise FiveBodyContractError("consequence rank feature schema changed")
        output["predicted_goal_progress"] = predicted_goal_progress
        output["predicted_goal_progress_uncertainty"] = (
            predicted_goal_progress_uncertainty
        )
        output["candidate_rank_features"] = rank_features
        output["candidate_rank_logit"] = (
            output["success_logit"]
            if self.ablation_variant == "success_only"
            else self.candidate_rank(rank_features).squeeze(-1)
        )
        return output


class StandardizedRankEnsemble(torch.nn.Module):
    """Source-validation wrapper identical to the formal deployment scorer."""

    def __init__(
        self,
        models: Sequence[EffectAlignedSharedEventHead],
        ablation_variant: str,
    ) -> None:
        super().__init__()
        if len(models) != 5 or ablation_variant not in ABLATION_VARIANTS:
            raise FiveBodyContractError(
                "checkpoint selection requires five same-variant ensemble members"
            )
        self.models = torch.nn.ModuleList(models)
        self.ablation_variant = ablation_variant

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        member_scores = torch.stack(
            [model(batch)["candidate_rank_logit"] for model in self.models]
        )
        if member_scores.ndim != 2:
            raise FiveBodyContractError("ensemble rank rows must be [5,N]")
        groups: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(batch["logical_group"]):
            groups[str(group)].append(index)
        aggregate = torch.empty_like(member_scores[0])
        for indices in groups.values():
            if len(indices) != CANDIDATE_COUNT:
                raise FiveBodyContractError(
                    "ensemble validation batch split a four-candidate decision"
                )
            ordered = sorted(
                indices,
                key=lambda index: int(batch["candidate_index"][index]),
            )
            if [int(batch["candidate_index"][index]) for index in ordered] != list(
                range(CANDIDATE_COUNT)
            ):
                raise FiveBodyContractError(
                    "ensemble validation candidate order is not exactly 0..3"
                )
            selected = torch.as_tensor(
                ordered, device=member_scores.device, dtype=torch.long
            )
            aggregate[selected] = aggregate_standardized_rank_scores(
                member_scores[:, selected]
            )
        return {"candidate_rank_logit": aggregate}


@torch.no_grad()
def evaluate_terminal_consequences(
    model: EffectAlignedSharedEventHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Report accuracy/calibration of finite-horizon proper consequences."""

    model.eval()
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for raw in loader:
        batch = core._move_batch(raw, device)
        output = model(batch)
        tensors = {
            "event_label": batch["terminal_max_event_id"],
            "event_mask": batch["terminal_event_mask"],
            "event_probability": torch.softmax(
                output["terminal_event_logits"], dim=-1
            ),
            "goal_label": batch["terminal_goal_progress"],
            "goal_mask": batch["terminal_goal_progress_mask"],
            "goal_mean": output["terminal_goal_progress_mean"],
            "goal_log_scale": output["terminal_goal_progress_log_scale"],
            "post_label": batch["post_event_id"],
            "post_mask": batch["post_event_mask"],
            "current_event": batch["current_event_id"],
            "recovery_label": batch["recovery"],
            "regression_probability": output["regression_probability"],
            "joint_recovery_probability": output[
                "joint_recovery_probability"
            ],
        }
        for name, tensor in tensors.items():
            collected[name].append(tensor.detach().cpu().numpy())
    values = {name: np.concatenate(parts) for name, parts in collected.items()}

    event_mask = values["event_mask"] > 0.5
    event_label = values["event_label"][event_mask].astype(np.int64)
    event_probability = values["event_probability"][event_mask]
    event_prediction = event_probability.argmax(axis=-1)
    event_nll = -np.log(
        np.clip(event_probability[np.arange(len(event_label)), event_label], 1e-12, 1.0)
    )
    event_onehot = np.eye(5, dtype=np.float64)[event_label]

    goal_mask = values["goal_mask"] > 0.5
    goal_label = values["goal_label"][goal_mask].astype(np.float64)
    goal_mean = values["goal_mean"][goal_mask].astype(np.float64)
    goal_log_scale = values["goal_log_scale"][goal_mask].astype(np.float64)
    goal_scale = np.exp(goal_log_scale).clip(min=1e-5)
    goal_standardized = (goal_label - goal_mean) / goal_scale
    dof = TERMINAL_PROGRESS_STUDENT_T_DOF
    goal_normalizer = (
        math.lgamma(dof / 2.0)
        + 0.5 * math.log(dof * math.pi)
        - math.lgamma((dof + 1.0) / 2.0)
    )
    goal_nll = (
        goal_log_scale
        + 0.5 * (dof + 1.0) * np.log1p(np.square(goal_standardized) / dof)
        + goal_normalizer
    )
    central_90_t3 = 2.3533634348018264

    regression_mask = values["post_mask"] > 0.5
    regression_label = (
        values["post_label"] < values["current_event"]
    ).astype(np.float64)[regression_mask]
    recovery_label = (
        (values["recovery_label"] > 0.5)
        & (values["post_label"] < values["current_event"])
    ).astype(np.float64)[regression_mask]

    def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
        probabilities = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
        return {
            "support": int(len(labels)),
            "positive": int((labels > 0.5).sum()),
            "brier": float(np.mean(np.square(probabilities - labels))),
            "nll": float(
                np.mean(
                    -labels * np.log(probabilities)
                    - (1.0 - labels) * np.log1p(-probabilities)
                )
            ),
        }

    event_metrics = core._event_metrics(event_label, event_prediction)
    return {
        "terminal_event": {
            **event_metrics,
            "support": int(len(event_label)),
            "class_counts": np.bincount(event_label, minlength=5).tolist(),
            "nll": float(np.mean(event_nll)),
            "multiclass_brier": float(
                np.mean(np.sum(np.square(event_probability - event_onehot), axis=-1))
            ),
            "ordinal_mae": float(np.mean(np.abs(event_prediction - event_label))),
        },
        "terminal_goal_progress": {
            "support": int(len(goal_label)),
            "mae_meters": float(np.mean(np.abs(goal_mean - goal_label))),
            "rmse_meters": float(np.sqrt(np.mean(np.square(goal_mean - goal_label)))),
            "student_t3_nll": float(np.mean(goal_nll)),
            "central_90_coverage": float(
                np.mean(np.abs(goal_label - goal_mean) <= central_90_t3 * goal_scale)
            ),
        },
        "regression": binary_metrics(
            regression_label,
            values["regression_probability"][regression_mask],
        ),
        "joint_recovery": binary_metrics(
            recovery_label,
            values["joint_recovery_probability"][regression_mask],
        ),
    }


def _dense_rank_components(
    batch: Mapping[str, Any],
    reference: torch.Tensor,
    *,
    ablation_variant: str,
) -> list[torch.Tensor]:
    """Return ordered labels; comparisons must never collapse them to weights."""

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
    components = [batch["terminal_max_event_id"].to(reference).float()]
    if ablation_variant != "no_object_effect":
        components.append(batch["terminal_goal_progress"].to(reference).float())
    return components


DENSE_GOAL_PROGRESS_TEMPERATURE_METERS = 0.02


def _dense_soft_listwise_loss(
    scores: torch.Tensor,
    terminal_event_level: torch.Tensor,
    terminal_goal_progress: torch.Tensor,
    *,
    ablation_variant: str,
) -> torch.Tensor:
    """Strictly prefer the latest event, then softly rank metric progress.

    Event level remains lexicographically absolute: target probability is zero
    outside the maximum terminal-event level.  Within that level, the public
    analytic near-goal scale (0.02 m) is the frozen softmax temperature.  This
    avoids turning replay/numerical micrometre differences into a hard one-hot
    label.  ``no_object_effect`` uses a uniform target over the maximum level.
    """

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
    if (
        scores.ndim != 1
        or terminal_event_level.shape != scores.shape
        or terminal_goal_progress.shape != scores.shape
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(terminal_goal_progress).all())
    ):
        raise FiveBodyContractError("dense soft listwise target has invalid shape/value")
    maximum_level = terminal_event_level.max()
    maximum_mask = terminal_event_level == maximum_level
    if not bool(maximum_mask.any()):
        raise FiveBodyContractError("dense soft listwise target selected no event level")
    if ablation_variant == "no_object_effect":
        target = torch.full_like(scores[maximum_mask], 1.0 / int(maximum_mask.sum()))
    else:
        target = torch.softmax(
            terminal_goal_progress[maximum_mask]
            / DENSE_GOAL_PROGRESS_TEMPERATURE_METERS,
            dim=0,
        )
    return -(target * torch.log_softmax(scores, dim=0)[maximum_mask]).sum()


def _negative_log_probability_mass(
    scores: torch.Tensor, preferred: torch.Tensor
) -> torch.Tensor:
    if scores.ndim != 1 or preferred.shape != scores.shape or not bool(preferred.any()):
        raise FiveBodyContractError("listwise probability mass target is invalid")
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[preferred], dim=0)


def _lexicographic_compare_values(
    left: Sequence[float], right: Sequence[float]
) -> int:
    if len(left) != len(right) or not left:
        raise FiveBodyContractError("dense lexicographic values have incompatible shapes")
    for left_value, right_value in zip(left, right):
        if abs(float(left_value) - float(right_value)) <= 1e-6:
            continue
        return 1 if float(left_value) > float(right_value) else -1
    return 0


def _robust_object_effect_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Student-t object likelihood for the uniform proper-likelihood stream."""

    # A Student-t(3) likelihood retains a calibrated scale head but prevents
    # a few large object errors from dominating the shared consequence model.
    object_scale = torch.exp(output["object_delta_log_scale"]).clamp_min(1e-4)
    standardized = (
        batch["object_delta"].to(output["object_delta_mean"])
        - output["object_delta_mean"]
    ) / object_scale
    object_rows = (
        output["object_delta_log_scale"]
        + 2.0 * torch.log1p(standardized.square() / 3.0)
    ).mean(dim=-1)
    object_effect = core._weighted_mean(
        object_rows,
        sample_weight
        * batch["action_available"].to(sample_weight)
        * batch["object_delta_mask"].to(sample_weight),
    )
    if ablation_variant in {"success_only", "no_object_effect"}:
        object_effect = object_effect * 0.0
    total = 0.5 * object_effect
    return total, {
        "robust_object_effect_uniform_proper": object_effect,
        "robust_object_effect_weighted_uniform_proper": total,
    }


def _terminal_consequence_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Proper finite-horizon consequence losses on the uniform stream only."""

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
    event_rows = torch.nn.functional.cross_entropy(
        output["terminal_event_logits"],
        batch["terminal_max_event_id"].long(),
        reduction="none",
    )
    event_weight = (
        sample_weight
        * batch["action_available"].to(sample_weight)
        * batch["terminal_event_mask"].to(sample_weight)
    )
    terminal_event = core._weighted_mean(event_rows, event_weight)

    goal_target = batch["terminal_goal_progress"].to(
        output["terminal_goal_progress_mean"]
    )
    goal_mean = output["terminal_goal_progress_mean"]
    goal_log_scale = output["terminal_goal_progress_log_scale"]
    goal_scale = torch.exp(goal_log_scale).clamp_min(1e-5)
    standardized = (goal_target - goal_mean) / goal_scale
    dof = TERMINAL_PROGRESS_STUDENT_T_DOF
    normalizer = (
        math.lgamma(dof / 2.0)
        + 0.5 * math.log(dof * math.pi)
        - math.lgamma((dof + 1.0) / 2.0)
    )
    goal_rows = (
        goal_log_scale
        + 0.5 * (dof + 1.0) * torch.log1p(standardized.square() / dof)
        + normalizer
    )
    goal_weight = (
        sample_weight
        * batch["action_available"].to(sample_weight)
        * batch["terminal_goal_progress_mask"].to(sample_weight)
    )
    terminal_goal = core._weighted_mean(goal_rows, goal_weight)

    if ablation_variant == "success_only":
        terminal_event = terminal_event * 0.0
    if ablation_variant in {"success_only", "no_object_effect"}:
        terminal_goal = terminal_goal * 0.0
    weighted_event = TERMINAL_EVENT_LOSS_WEIGHT * terminal_event
    weighted_goal = TERMINAL_GOAL_PROGRESS_LOSS_WEIGHT * terminal_goal
    return weighted_event + weighted_goal, {
        "terminal_event_uniform_proper": terminal_event,
        "terminal_goal_progress_uniform_proper": terminal_goal,
        "terminal_event_weighted_uniform_proper": weighted_event,
        "terminal_goal_progress_weighted_uniform_proper": weighted_goal,
    }


def _candidate_rank_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Deployment-aligned listwise loss for the balanced rank-only stream."""

    labels = batch["success"].to(output["success_logit"])
    score = output["candidate_rank_logit"]
    if ablation_variant == "success_only":
        zero = score.sum() * 0.0
        return zero, {
            "group_listwise_success_mass_balanced_rank": zero,
            "all_failure_dense_soft_listwise_balanced_rank": zero,
            "candidate_ranking_balanced_rank": zero,
            "mixed_success_groups_in_batch": score.new_tensor(0),
            "all_failure_dense_groups_in_batch": score.new_tensor(0),
            "all_failure_uninformative_groups_in_batch": score.new_tensor(0),
        }

    success_terms: list[torch.Tensor] = []
    success_weights: list[torch.Tensor] = []
    dense_terms: list[torch.Tensor] = []
    dense_weights: list[torch.Tensor] = []
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(batch["logical_group"]):
        by_group[str(group)].append(index)
    mixed_success_groups = 0
    all_failure_dense_groups = 0
    all_failure_uninformative_groups = 0
    for indices in by_group.values():
        if len(indices) != CANDIDATE_COUNT:
            raise FiveBodyContractError("training batch split a candidate decision")
        selected = torch.as_tensor(indices, device=score.device, dtype=torch.long)
        group_scores = score[selected]
        selected_weights = sample_weight[selected]
        if not bool(torch.isfinite(selected_weights).all()) or bool(
            (selected_weights < 0.0).any()
        ):
            raise FiveBodyContractError("rank batch has invalid bootstrap weight")
        if not bool(
            torch.allclose(
                selected_weights,
                selected_weights[:1].expand_as(selected_weights),
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise FiveBodyContractError("rank bootstrap weight changed within a decision")
        group_weight = selected_weights[0]
        # Outcome kind and support counters are defined only for positive-weight
        # decisions.  A Poisson-zero dense group must not masquerade as a rank
        # update in the training audit.
        if float(group_weight.detach()) <= 0.0:
            continue
        if "success_mask" in batch and not bool(
            (batch["success_mask"].to(score)[selected] > 0.5).all()
        ):
            raise FiveBodyContractError("rank decision lacks complete success supervision")
        successful = labels[selected] > 0.5
        if bool(successful.any()) and bool((~successful).any()):
            success_terms.append(
                _negative_log_probability_mass(group_scores, successful).reshape(())
            )
            success_weights.append(group_weight)
            mixed_success_groups += 1
        elif not bool(successful.any()):
            if not bool(
                (
                    batch["terminal_event_mask"].to(score)[selected] > 0.5
                ).all()
            ) or not bool(
                (
                    batch["terminal_goal_progress_mask"].to(score)[selected]
                    > 0.5
                ).all()
            ):
                raise FiveBodyContractError(
                    "dense rank decision lacks complete terminal supervision"
                )
            terminal_event_level = (
                batch["terminal_max_event_id"].to(score)[selected].float()
            )
            terminal_goal_progress = (
                batch["terminal_goal_progress"].to(score)[selected].float()
            )
            if not _dense_rank_labels_are_orderable(
                terminal_event_level.detach().cpu().tolist(),
                terminal_goal_progress.detach().cpu().tolist(),
                ablation_variant=ablation_variant,
            ):
                all_failure_uninformative_groups += 1
                continue
            dense_terms.append(
                _dense_soft_listwise_loss(
                    group_scores,
                    terminal_event_level,
                    terminal_goal_progress,
                    ablation_variant=ablation_variant,
                ).reshape(())
            )
            dense_weights.append(group_weight)
            all_failure_dense_groups += 1
    if success_terms:
        success_ranking = core._weighted_mean(
            torch.stack(success_terms), torch.stack(success_weights)
        )
    else:
        success_ranking = score.sum() * 0.0
    if dense_terms:
        dense_ranking = core._weighted_mean(
            torch.stack(dense_terms), torch.stack(dense_weights)
        )
    else:
        dense_ranking = score.sum() * 0.0
    ranking = success_ranking + DENSE_FAILURE_RANK_WEIGHT * dense_ranking
    return ranking, {
        "group_listwise_success_mass_balanced_rank": success_ranking,
        "all_failure_dense_soft_listwise_balanced_rank": dense_ranking,
        "candidate_ranking_balanced_rank": ranking,
        "mixed_success_groups_in_batch": score.new_tensor(mixed_success_groups),
        "all_failure_dense_groups_in_batch": score.new_tensor(
            all_failure_dense_groups
        ),
        "all_failure_uninformative_groups_in_batch": score.new_tensor(
            all_failure_uninformative_groups
        ),
    }


@torch.no_grad()
def evaluate_candidate_ranking(
    model: EffectAlignedSharedEventHead,
    loader: DataLoader,
    device: torch.device,
    *,
    ablation_variant: str | None = None,
) -> dict[str, Any]:
    model.eval()
    variant = ablation_variant or getattr(model, "ablation_variant", "full")
    if variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    groups: dict[
        str,
        list[tuple[int, float, float, tuple[float, ...], float, float, float]],
    ] = defaultdict(list)
    for raw in loader:
        batch = core._move_batch(raw, device)
        output = model(batch)
        dense_components = _dense_rank_components(
            batch, output["candidate_rank_logit"], ablation_variant=variant
        )
        for index, group in enumerate(raw["logical_group"]):
            groups[str(group)].append(
                (
                    int(batch["candidate_index"][index]),
                    float(output["candidate_rank_logit"][index]),
                    float(batch["success"][index]),
                    tuple(float(component[index]) for component in dense_components),
                    float(batch["terminal_stage_progress"][index]),
                    float(batch["terminal_goal_distance"][index]),
                    float(batch["terminal_goal_progress"][index]),
                )
            )
    decisions: list[dict[str, Any]] = []
    mixed_pairs: list[dict[str, Any]] = []
    dense_pairs: list[dict[str, Any]] = []
    for group, rows in groups.items():
        rows = sorted(rows)
        if len(rows) != CANDIDATE_COUNT or [row[0] for row in rows] != list(range(4)):
            raise FiveBodyContractError(f"validation decision incomplete: {group}")
        identity = group.split("|", 2)
        if len(identity) != 3 or identity[0] not in BODIES or identity[1] not in CONDITIONS:
            raise FiveBodyContractError(f"validation decision identity changed: {group}")
        selected = max(rows, key=lambda row: row[1])
        oracle_success = max(row[2] for row in rows)
        minimum_success = min(row[2] for row in rows)
        mixed_success = oracle_success > 0.5 and minimum_success <= 0.5
        all_failure = oracle_success <= 0.5
        dense_applicable = bool(
            variant != "success_only"
            and all_failure
            and any(
                _lexicographic_compare_values(row[3], rows[0][3]) != 0
                for row in rows[1:]
            )
        )
        best_dense = rows[0][3]
        for row in rows[1:]:
            if _lexicographic_compare_values(row[3], best_dense) > 0:
                best_dense = row[3]
        decision = {
            "body": identity[0],
            "condition": identity[1],
            "baseline_success": rows[0][2],
            "selected_success": selected[2],
            "oracle_success": oracle_success,
            "baseline_terminal_stage_progress": rows[0][4],
            "selected_terminal_stage_progress": selected[4],
            "oracle_terminal_stage_progress": max(row[4] for row in rows),
            "baseline_terminal_goal_distance": rows[0][5],
            "selected_terminal_goal_distance": selected[5],
            "baseline_terminal_goal_progress": rows[0][6],
            "selected_terminal_goal_progress": selected[6],
            "mixed_success": mixed_success,
            "dense_applicable": dense_applicable,
            "dense_uninformative": bool(
                variant != "success_only" and all_failure and not dense_applicable
            ),
            "selected_dense_best": bool(
                dense_applicable
                and _lexicographic_compare_values(selected[3], best_dense) == 0
            ),
        }
        decisions.append(decision)
        for left in range(4):
            for right in range(left + 1, 4):
                score_difference = rows[left][1] - rows[right][1]
                success_difference = rows[left][2] - rows[right][2]
                if abs(success_difference) > 1e-6:
                    mixed_pairs.append(
                        {
                            "body": identity[0],
                            "condition": identity[1],
                            "correct": bool(score_difference * success_difference > 0),
                        }
                    )
                elif dense_applicable and (
                    dense_comparison := _lexicographic_compare_values(
                        rows[left][3], rows[right][3]
                    )
                ):
                    dense_sign = float(dense_comparison)
                    dense_pairs.append(
                        {
                            "body": identity[0],
                            "condition": identity[1],
                            "correct": bool(score_difference * dense_sign > 0),
                        }
                    )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        baseline = float(np.mean([float(row["baseline_success"]) for row in rows]))
        selected = float(np.mean([float(row["selected_success"]) for row in rows]))
        mixed = [row for row in rows if bool(row["mixed_success"])]
        dense = [row for row in rows if bool(row["dense_applicable"])]
        dense_uninformative = [
            row for row in rows if bool(row["dense_uninformative"])
        ]
        return {
            "decision_groups": len(rows),
            "baseline_success_rate": baseline,
            "selected_success_rate": selected,
            "one_deviation_branch_success_gain": selected - baseline,
            "oracle_success_rate": float(
                np.mean([float(row["oracle_success"]) for row in rows])
            ),
            "baseline_terminal_stage_progress": float(
                np.mean([float(row["baseline_terminal_stage_progress"]) for row in rows])
            ),
            "selected_terminal_stage_progress": float(
                np.mean([float(row["selected_terminal_stage_progress"]) for row in rows])
            ),
            "delta_terminal_stage_progress": float(
                np.mean(
                    [
                        float(row["selected_terminal_stage_progress"])
                        - float(row["baseline_terminal_stage_progress"])
                        for row in rows
                    ]
                )
            ),
            "oracle_terminal_stage_progress": float(
                np.mean([float(row["oracle_terminal_stage_progress"]) for row in rows])
            ),
            "baseline_terminal_goal_distance": float(
                np.mean([float(row["baseline_terminal_goal_distance"]) for row in rows])
            ),
            "selected_terminal_goal_distance": float(
                np.mean([float(row["selected_terminal_goal_distance"]) for row in rows])
            ),
            "delta_terminal_goal_progress": float(
                np.mean(
                    [
                        float(row["selected_terminal_goal_progress"])
                        - float(row["baseline_terminal_goal_progress"])
                        for row in rows
                    ]
                )
            ),
            "mixed_success_decisions": len(mixed),
            "mixed_success_selection_accuracy": (
                float(np.mean([float(row["selected_success"]) for row in mixed]))
                if mixed
                else None
            ),
            "dense_progress_decisions": len(dense),
            "dense_uninformative_decisions": len(dense_uninformative),
            "dense_progress_selection_accuracy": (
                float(np.mean([float(row["selected_dense_best"]) for row in dense]))
                if dense
                else None
            ),
        }

    def pair_summary(
        rows: Sequence[Mapping[str, Any]], *, body: str | None = None,
        condition: str | None = None,
    ) -> tuple[float | None, int]:
        selected = [
            row for row in rows
            if (body is None or row["body"] == body)
            and (condition is None or row["condition"] == condition)
        ]
        return (
            float(np.mean([float(row["correct"]) for row in selected]))
            if selected
            else None,
            len(selected),
        )

    global_metrics = summarize(decisions)
    units: dict[str, dict[str, Any]] = {}
    for body in BODIES:
        for condition in CONDITIONS:
            selected_rows = [
                row
                for row in decisions
                if row["body"] == body and row["condition"] == condition
            ]
            if selected_rows:
                unit = summarize(selected_rows)
                mixed_pair_accuracy, mixed_pair_count = pair_summary(
                    mixed_pairs, body=body, condition=condition
                )
                dense_pair_accuracy, dense_pair_count = pair_summary(
                    dense_pairs, body=body, condition=condition
                )
                unit.update(
                    {
                        "mixed_success_pairwise_accuracy": mixed_pair_accuracy,
                        "mixed_success_pairwise_comparisons": mixed_pair_count,
                        "dense_progress_pairwise_accuracy": dense_pair_accuracy,
                        "dense_progress_pairwise_comparisons": dense_pair_count,
                    }
                )
                units[f"{body}|{condition}"] = unit

    def macro(name: str) -> float | None:
        values = [row[name] for row in units.values() if row.get(name) is not None]
        return float(np.mean([float(value) for value in values])) if values else None

    macro_gain = float(
        np.mean(
            [
                row["one_deviation_branch_success_gain"]
                for row in units.values()
            ]
        )
    )
    macro_selected = float(
        np.mean([row["selected_success_rate"] for row in units.values()])
    )
    macro_oracle = float(
        np.mean([row["oracle_success_rate"] for row in units.values()])
    )
    mixed_pair_accuracy, mixed_pair_count = pair_summary(mixed_pairs)
    dense_pair_accuracy, dense_pair_count = pair_summary(dense_pairs)
    return {
        **global_metrics,
        "body_condition_units": units,
        "macro_one_deviation_branch_success_gain": macro_gain,
        "estimand": ONE_DEVIATION_ESTIMAND,
        "macro_selected_success_rate": macro_selected,
        "macro_oracle_success_rate": macro_oracle,
        "macro_delta_terminal_stage_progress": macro(
            "delta_terminal_stage_progress"
        ),
        "macro_delta_terminal_goal_progress": macro(
            "delta_terminal_goal_progress"
        ),
        "macro_mixed_success_selection_accuracy": macro(
            "mixed_success_selection_accuracy"
        ),
        "mixed_success_pairwise_accuracy": mixed_pair_accuracy,
        "mixed_success_pairwise_comparisons": mixed_pair_count,
        "macro_mixed_success_pairwise_accuracy": macro(
            "mixed_success_pairwise_accuracy"
        ),
        "macro_dense_progress_selection_accuracy": macro(
            "dense_progress_selection_accuracy"
        ),
        "dense_progress_pairwise_accuracy": dense_pair_accuracy,
        "dense_progress_pairwise_comparisons": dense_pair_count,
        "macro_dense_progress_pairwise_accuracy": macro(
            "dense_progress_pairwise_accuracy"
        ),
        # Backward-compatible diagnostic aliases now mean success-changing
        # comparisons only; dense failure ordering is reported separately.
        "pairwise_accuracy": mixed_pair_accuracy,
        "pairwise_comparisons": mixed_pair_count,
    }


def candidate_checkpoint_selection_key(
    ranking: Mapping[str, Any], diagnostic_score: float, step: int
) -> tuple[float, float, float, float, float, float, int]:
    """Select for deployed top-1 success before any dense diagnostic."""

    def maximize(value: Any) -> float:
        if value is None or not math.isfinite(float(value)):
            return 0.0
        return -float(value)

    return (
        maximize(ranking["macro_one_deviation_branch_success_gain"]),
        maximize(ranking.get("macro_mixed_success_selection_accuracy")),
        maximize(ranking.get("macro_mixed_success_pairwise_accuracy")),
        maximize(ranking.get("macro_dense_progress_selection_accuracy")),
        maximize(ranking.get("macro_dense_progress_pairwise_accuracy")),
        float(diagnostic_score),
        int(step),
    )


def materialize_source_rows(
    groups: Sequence[Mapping[str, Any]], *, held_out_body: str
) -> list[dict[str, Any]]:
    if any(group.get("body") == held_out_body for group in groups):
        raise FiveBodyContractError("held-out group reached source payload loader")
    rows = []
    for group in groups:
        rows.extend(_npz_rows(group, body=str(group["body"])))
    if any(row["body"] == held_out_body for row in rows):
        raise FiveBodyContractError("held-out row reached source fitting")
    return rows


def effect_preserving_group_bootstrap_weights(
    rows: Sequence[Mapping[str, Any]], *, members: int, seed: int
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Use 1+Poisson for every mixed decision and Poisson for dense decisions."""

    group_order = [str(row["logical_group"]) for row in rows]
    weights = core.logical_group_bootstrap_weights(
        group_order, members=members, seed=seed
    )
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_order):
        indices_by_group[group].append(index)
    mixed_groups = []
    for group, indices in sorted(indices_by_group.items()):
        outcomes = {
            float(rows[index]["success"])
            for index in indices
            if bool(rows[index]["success_mask"])
        }
        if outcomes == {0.0, 1.0}:
            mixed_groups.append(group)
    if not mixed_groups:
        raise FiveBodyContractError(
            "effect-preserving bootstrap requires a mixed-success decision"
        )
    # Every epistemic member sees every rare success-changing comparison.  The
    # Poisson component still changes its relative influence across members;
    # all-failure decisions retain the ordinary Poisson bootstrap, including
    # genuine zero weights.
    for group in mixed_groups:
        weights[:, indices_by_group[group]] += 1.0
    audit: list[dict[str, int]] = []
    for member in range(members):
        active_mixed = [
            group
            for group in mixed_groups
            if float(weights[member, indices_by_group[group][0]]) > 0.0
        ]
        active_indices = [
            index for index in range(len(rows)) if float(weights[member, index]) > 0.0
        ]
        positives = sum(
            bool(rows[index]["success_mask"]) and float(rows[index]["success"]) > 0.5
            for index in active_indices
        )
        negatives = sum(
            bool(rows[index]["success_mask"]) and float(rows[index]["success"]) <= 0.5
            for index in active_indices
        )
        active_mixed_count = sum(
            float(weights[member, indices_by_group[group][0]]) > 0.0
            for group in mixed_groups
        )
        if not positives or not negatives or not active_mixed_count:
            raise FiveBodyContractError(
                "effect-preserving bootstrap failed to retain outcome supervision"
            )
        audit.append(
            {
                "member": member,
                "positive_rows_with_nonzero_weight": int(positives),
                "negative_rows_with_nonzero_weight": int(negatives),
                "mixed_success_groups_with_nonzero_weight": int(active_mixed_count),
                "mixed_success_groups_total": len(mixed_groups),
                "mixed_weight_minimum": 1,
                "deterministic_mixed_group_repairs": 0,
            }
        )
    return weights.astype(np.float32, copy=False), audit


def proper_outcome_preserving_group_bootstrap_weights(
    rows: Sequence[Mapping[str, Any]], *, members: int, seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep every proper-likelihood member identifiable for task success.

    The ordinary Poisson group bootstrap is retained exactly unless it drops
    every success-positive or every success-negative logical group for one
    member.  In that rare case one complete mixed-success decision is restored
    with unit weight.  Restoring a whole decision preserves the same-root
    candidate dependence and gives the coherent terminal-event/success heads
    both outcome classes without introducing a global class weight.
    """

    group_order = [str(row["logical_group"]) for row in rows]
    weights = core.logical_group_bootstrap_weights(
        group_order, members=members, seed=seed
    )
    if weights.shape != (members, len(rows)):
        raise FiveBodyContractError("proper bootstrap returned an invalid shape")

    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_order):
        indices_by_group[group].append(index)
    positive_groups: list[str] = []
    negative_groups: list[str] = []
    mixed_groups: list[str] = []
    for group, indices in sorted(indices_by_group.items()):
        if not all(bool(rows[index]["success_mask"]) for index in indices):
            raise FiveBodyContractError(
                "proper success bootstrap requires complete group supervision"
            )
        outcomes = {float(rows[index]["success"]) for index in indices}
        if not outcomes <= {0.0, 1.0}:
            raise FiveBodyContractError(
                "proper success bootstrap found a non-binary outcome"
            )
        if 1.0 in outcomes:
            positive_groups.append(group)
        if 0.0 in outcomes:
            negative_groups.append(group)
        if outcomes == {0.0, 1.0}:
            mixed_groups.append(group)
    if not positive_groups or not negative_groups or not mixed_groups:
        raise FiveBodyContractError(
            "proper outcome-preserving bootstrap requires positive, negative, "
            "and mixed-success logical groups"
        )

    audit: list[dict[str, Any]] = []
    for member in range(members):
        repaired_groups: list[str] = []

        def active(group: str) -> bool:
            return float(weights[member, indices_by_group[group][0]]) > 0.0

        if not any(active(group) for group in positive_groups) or not any(
            active(group) for group in negative_groups
        ):
            selector = int.from_bytes(
                hashlib.sha256(
                    f"{seed}|proper-outcome|{member}".encode()
                ).digest()[:8],
                "big",
            )
            repaired = mixed_groups[selector % len(mixed_groups)]
            selected = indices_by_group[repaired]
            weights[member, selected] = np.maximum(
                weights[member, selected], np.float32(1.0)
            )
            repaired_groups.append(repaired)

        active_indices = [
            index for index in range(len(rows)) if float(weights[member, index]) > 0.0
        ]
        positive_rows = sum(
            float(rows[index]["success"]) > 0.5 for index in active_indices
        )
        negative_rows = sum(
            float(rows[index]["success"]) <= 0.5 for index in active_indices
        )
        active_positive_groups = sum(active(group) for group in positive_groups)
        active_negative_groups = sum(active(group) for group in negative_groups)
        active_mixed_groups = sum(active(group) for group in mixed_groups)
        if (
            not positive_rows
            or not negative_rows
            or not active_positive_groups
            or not active_negative_groups
        ):
            raise FiveBodyContractError(
                "proper outcome-preserving bootstrap lost identifiable success support"
            )
        for group, indices in indices_by_group.items():
            group_values = weights[member, indices]
            if not np.all(group_values == group_values[0]):
                raise FiveBodyContractError(
                    f"proper bootstrap changed within logical group {group}"
                )
        audit.append(
            {
                "member": member,
                "positive_rows_with_nonzero_weight": int(positive_rows),
                "negative_rows_with_nonzero_weight": int(negative_rows),
                "positive_groups_with_nonzero_weight": int(active_positive_groups),
                "negative_groups_with_nonzero_weight": int(active_negative_groups),
                "mixed_success_groups_with_nonzero_weight": int(active_mixed_groups),
                "positive_groups_total": len(positive_groups),
                "negative_groups_total": len(negative_groups),
                "mixed_success_groups_total": len(mixed_groups),
                "deterministic_mixed_group_repairs": len(repaired_groups),
                "repaired_groups": repaired_groups,
            }
        )
    return weights.astype(np.float32, copy=False), audit


def _train_fold(args: argparse.Namespace, audit: Mapping[str, Any]) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("training output must be a new path")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise FiveBodyContractError("real fold training is remote CUDA-only")
    if "4090" not in torch.cuda.get_device_name(0):
        raise FiveBodyContractError("real fold training requires the authorized RTX 4090")
    train_groups, validation_groups, _heldout_groups = source_group_split(
        audit, held_out_body=args.held_out_body, split_seed=args.split_seed
    )
    preflight = build_preflight_receipt(
        audit, held_out_body=args.held_out_body, split_seed=args.split_seed
    )
    output.mkdir(parents=True)
    core.atomic_json(output / "preflight_receipt.json", preflight)
    train_rows = materialize_source_rows(train_groups, held_out_body=args.held_out_body)
    validation_rows = materialize_source_rows(
        validation_groups, held_out_body=args.held_out_body
    )
    successes = np.asarray(
        [float(row["success"]) for row in train_rows if bool(row["success_mask"])]
    )
    if not len(successes) or set(np.unique(successes).tolist()) != {0.0, 1.0}:
        raise FiveBodyContractError(
            "source training requires real positive and negative outcome supervision"
        )
    outcome_by_group: dict[str, set[float]] = defaultdict(set)
    for row in train_rows:
        if bool(row["success_mask"]):
            outcome_by_group[str(row["logical_group"])].add(float(row["success"]))
    mixed_outcome_decisions = sum(len(values) > 1 for values in outcome_by_group.values())
    if mixed_outcome_decisions == 0:
        raise FiveBodyContractError(
            "source training has no within-root success/failure candidate comparison"
        )
    source_normalization = core.fit_train_action_normalization(
        train_rows, required_schema_ids=(0,)
    )
    canonical_stats = dict(source_normalization["schemas"]["aloha"])
    canonical_stats["schema_id"] = 0
    normalization = {
        "format": "etsf_five_body_canonical_train_only_normalization_v2",
        "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
        "canonical_action_schema_id": 0,
        "storage_slot_origin": "legacy_schema0_relabelled_no_aloha_semantic_claim",
        "schema": canonical_stats,
        "heldout_rows_used": 0,
    }
    normalization["sha256"] = core.canonical_json_sha256(normalization)
    action_mean = np.asarray(canonical_stats["mean"], dtype=np.float32)[None]
    action_std = np.asarray(canonical_stats["std"], dtype=np.float32)[None]
    source_states = np.stack(
        [np.asarray(row["state"], dtype=np.float32) for row in train_rows]
    )
    state_mean = np.zeros(core.STATE_DIM, dtype=np.float32)
    state_std = np.ones(core.STATE_DIM, dtype=np.float32)
    # Channels 0:18 are continuous geometry/gripper/quaternion.  Event and
    # predicate channels 18:27 stay exact binary inputs.
    state_mean[:18] = source_states[:, :18].mean(axis=0)
    state_std[:18] = np.maximum(source_states[:, :18].std(axis=0), 1e-4)
    state_normalization = {
        "format": "etsf_five_body_canonical_state_train_only_normalization_v1",
        "canonical_state_schema": CANONICAL_STATE_SCHEMA,
        "continuous_channels": list(range(18)),
        "binary_channels_unchanged": list(range(18, core.STATE_DIM)),
        "mean": state_mean.tolist(),
        "std": state_std.tolist(),
        "heldout_rows_used": 0,
    }
    state_normalization["sha256"] = core.canonical_json_sha256(state_normalization)
    baseline = core.fit_train_baselines(train_rows)
    validation_baseline = core.evaluate_train_only_baselines(baseline, validation_rows)
    body_to_id = {body: 0 for body in preflight["source_bodies"]}
    train_dataset = core.TransitionDataset(train_rows, body_to_id)
    validation_loader = DataLoader(
        core.TransitionDataset(validation_rows, body_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    device = torch.device(args.device)
    group_order = [str(row["logical_group"]) for row in train_rows]
    proper_bootstrap, proper_bootstrap_support = (
        proper_outcome_preserving_group_bootstrap_weights(
            train_rows, members=5, seed=args.split_seed
        )
    )
    rank_bootstrap, bootstrap_support = effect_preserving_group_bootstrap_weights(
        train_rows, members=5, seed=args.split_seed
    )
    proper_group_weight = {
        group: proper_bootstrap[:, index].tolist()
        for index, group in enumerate(group_order)
    }
    rank_group_weight = {
        group: rank_bootstrap[:, index].tolist()
        for index, group in enumerate(group_order)
    }
    source_negative_to_positive_ratio = float(
        (successes <= 0.5).sum() / (successes > 0.5).sum()
    )
    base_loss_weights = dict(core.DEFAULT_LOSS_WEIGHTS)
    base_loss_weights["object"] = 0.0
    if args.ablation_variant == "success_only":
        base_loss_weights = {name: 0.0 for name in base_loss_weights}
        base_loss_weights["success"] = 1.0
    elif args.ablation_variant == "no_time_duration":
        base_loss_weights["duration"] = 0.0
    trainer_file_sha256 = sha256_file(Path(__file__).resolve())
    snapshot_root = output / "source_validation_common_step_snapshots"
    snapshot_root.mkdir()
    member_eval_records: list[dict[int, dict[str, Any]]] = []
    member_snapshot_paths: list[dict[int, Path]] = []
    for member, seed in enumerate(args.ensemble_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = EffectAlignedSharedEventHead(args.ablation_variant).to(device)
        model.action.set_normalization(
            torch.as_tensor(action_mean, device=device),
            torch.as_tensor(action_std, device=device),
        )
        model.set_state_normalization(
            torch.as_tensor(state_mean, device=device),
            torch.as_tensor(state_std, device=device),
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        proper_loader = DataLoader(
            train_dataset,
            batch_sampler=CompleteDecisionBatchSampler(
                train_rows, batch_size=args.batch_size, seed=seed
            ),
            collate_fn=core.collate_rows,
        )
        rank_weight_for_member = {
            group: float(weights[member]) for group, weights in rank_group_weight.items()
        }
        rank_loader = DataLoader(
            train_dataset,
            batch_sampler=MacroBalancedRankDecisionBatchSampler(
                train_rows,
                batch_size=args.batch_size,
                seed=seed,
                positive_group_weight=rank_weight_for_member,
                ablation_variant=args.ablation_variant,
            ),
            collate_fn=core.collate_rows,
        )
        proper_iterator = iter(proper_loader)
        rank_iterator = iter(rank_loader)
        eval_records: dict[int, dict[str, Any]] = {}
        snapshot_paths: dict[int, Path] = {}
        for step in range(1, args.steps + 1):
            try:
                proper_raw = next(proper_iterator)
            except StopIteration:
                proper_iterator = iter(proper_loader)
                proper_raw = next(proper_iterator)
            try:
                rank_raw = next(rank_iterator)
            except StopIteration:
                rank_iterator = iter(rank_loader)
                rank_raw = next(rank_iterator)
            proper_batch = core._move_batch(proper_raw, device)
            rank_batch = core._move_batch(rank_raw, device)
            proper_weights = torch.tensor(
                [
                    proper_group_weight[group][member]
                    for group in proper_raw["logical_group"]
                ],
                device=device,
            )
            rank_weights = torch.tensor(
                [rank_group_weight[group][member] for group in rank_raw["logical_group"]],
                device=device,
            )
            proper_prediction = model(proper_batch)
            multitask_loss, pieces = core.compute_multitask_loss(
                proper_prediction,
                proper_batch,
                sample_weight=proper_weights,
                loss_weights=base_loss_weights,
            )
            object_effect_loss, object_pieces = _robust_object_effect_loss(
                proper_prediction,
                proper_batch,
                proper_weights,
                ablation_variant=args.ablation_variant,
            )
            terminal_loss, terminal_pieces = _terminal_consequence_loss(
                proper_prediction,
                proper_batch,
                proper_weights,
                ablation_variant=args.ablation_variant,
            )
            rank_prediction = model(rank_batch)
            decision_loss, decision_pieces = _candidate_rank_loss(
                rank_prediction,
                rank_batch,
                rank_weights,
                ablation_variant=args.ablation_variant,
            )
            loss = (
                multitask_loss
                + object_effect_loss
                + terminal_loss
                + decision_loss
            )
            if not torch.isfinite(loss):
                raise FiveBodyContractError("non-finite shared-head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if step % args.eval_every and step != args.steps:
                continue
            metrics = core.evaluate_validation_model(model, validation_loader, device)
            metrics["terminal_consequences"] = evaluate_terminal_consequences(
                model, validation_loader, device
            )
            metrics["candidate_ranking"] = evaluate_candidate_ranking(
                model, validation_loader, device
            )
            diagnostic_score, components = core.validation_selection_score(
                metrics, validation_baseline
            )
            selection_components = ablation_selection_components(
                components, args.ablation_variant
            )
            diagnostic_score = float(np.mean(list(selection_components.values())))
            metrics["diagnostic_multitask_score"] = diagnostic_score
            metrics["diagnostic_multitask_components"] = components
            metrics["checkpoint_selection_diagnostic_components"] = selection_components
            metrics["train_objective_last"] = {
                "total": float(loss.detach()),
                **{name: float(value.detach()) for name, value in pieces.items() if name != "total"},
                **{name: float(value.detach()) for name, value in object_pieces.items()},
                **{name: float(value.detach()) for name, value in terminal_pieces.items()},
                **{name: float(value.detach()) for name, value in decision_pieces.items()},
            }
            snapshot = snapshot_root / (
                f"member_{member:02d}_seed_{seed}_step_{step:06d}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "member": member,
                    "seed": seed,
                    "step": step,
                    "trainer_file_sha256": trainer_file_sha256,
                },
                snapshot,
            )
            eval_records[step] = metrics
            snapshot_paths[step] = snapshot
            model.train()
        if not eval_records:
            raise FiveBodyContractError("ensemble member produced no common-step snapshot")
        member_eval_records.append(eval_records)
        member_snapshot_paths.append(snapshot_paths)
        del optimizer, model
        torch.cuda.empty_cache()

    common_steps = sorted(
        set.intersection(*(set(records) for records in member_eval_records))
    )
    expected_steps = list(range(args.eval_every, args.steps + 1, args.eval_every))
    if not expected_steps or expected_steps[-1] != args.steps:
        expected_steps.append(args.steps)
    if common_steps != expected_steps:
        raise FiveBodyContractError(
            "five members do not share every preregistered source-validation step"
        )
    selection_models = [
        EffectAlignedSharedEventHead(args.ablation_variant).to(device)
        for _ in args.ensemble_seeds
    ]
    ensemble_model = StandardizedRankEnsemble(
        selection_models, args.ablation_variant
    ).to(device)
    best_ensemble_key: tuple[float, float, float, float, float, float, int] | None = None
    best_ensemble_step = 0
    best_ensemble_ranking: dict[str, Any] | None = None
    best_ensemble_diagnostic = float("inf")
    common_step_selection_audit = []
    for step in common_steps:
        for member, model in enumerate(selection_models):
            snapshot = torch.load(
                member_snapshot_paths[member][step],
                map_location=device,
                weights_only=True,
            )
            if (
                snapshot.get("member") != member
                or snapshot.get("seed") != args.ensemble_seeds[member]
                or snapshot.get("step") != step
                or snapshot.get("trainer_file_sha256") != trainer_file_sha256
            ):
                raise FiveBodyContractError("common-step member snapshot changed")
            model.load_state_dict(snapshot["model"], strict=True)
            model.eval()
        ensemble_ranking = evaluate_candidate_ranking(
            ensemble_model,
            validation_loader,
            device,
            ablation_variant=args.ablation_variant,
        )
        diagnostic = float(
            np.mean(
                [
                    member_eval_records[member][step][
                        "diagnostic_multitask_score"
                    ]
                    for member in range(5)
                ]
            )
        )
        key = candidate_checkpoint_selection_key(
            ensemble_ranking, diagnostic, step
        )
        common_step_selection_audit.append(
            {
                "step": step,
                "selection_key": list(key),
                "ensemble_candidate_ranking": ensemble_ranking,
                "mean_member_diagnostic_multitask_score": diagnostic,
            }
        )
        if best_ensemble_key is None or key < best_ensemble_key:
            best_ensemble_key = key
            best_ensemble_step = step
            best_ensemble_ranking = ensemble_ranking
            best_ensemble_diagnostic = diagnostic
    if best_ensemble_ranking is None or best_ensemble_key is None:
        raise FiveBodyContractError("deployment-homomorphic ensemble selected no checkpoint")

    members = []
    for member, seed in enumerate(args.ensemble_seeds):
        snapshot = torch.load(
            member_snapshot_paths[member][best_ensemble_step],
            map_location="cpu",
            weights_only=True,
        )
        checkpoint = output / f"member_{member:02d}_seed_{seed}_best.pt"
        member_validation = member_eval_records[member][best_ensemble_step]
        torch.save(
            {
                "format": FORMAT,
                "model": snapshot["model"],
                "config": dataclasses.asdict(selection_models[member].config),
                "member": member,
                "seed": seed,
                "step": best_ensemble_step,
                "ensemble_common_selection_step": best_ensemble_step,
                "held_out_body": args.held_out_body,
                "source_bodies": preflight["source_bodies"],
                "body_adapter": "single_shared_row_zero_heldout_parameters",
                "model_family": MODEL_FAMILY,
                "ablation": ablation_contract(args.ablation_variant),
                "candidate_rank_contract": checkpoint_candidate_rank_contract(
                    args.ablation_variant
                ),
                "canonical_state_schema": CANONICAL_STATE_SCHEMA,
                "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
                "event_age_contract": event_age_contract(),
                "terminal_horizon_contract": terminal_horizon_contract(),
                "event_spec_sha256": EVENT_SPEC_SHA256,
                "event_derivation_implementation_sha256": preflight[
                    "event_derivation_implementation_sha256"
                ],
                "trainer_file_sha256": trainer_file_sha256,
                "action_stem_count": 1,
                "body_to_id_source_only": body_to_id,
                "heldout_rows_used_for_training_normalization_or_selection": 0,
                "actor_frozen": True,
                "action_normalization": normalization,
                "state_normalization": state_normalization,
                "preflight_logical_sha256": preflight["logical_sha256"],
                "validation": member_validation,
                "one_deviation_ensemble_source_validation": (
                    best_ensemble_ranking
                ),
            },
            checkpoint,
        )
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": best_ensemble_step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "trainer_file_sha256": trainer_file_sha256,
                "source_validation": member_validation,
            }
        )
    del ensemble_model, selection_models
    torch.cuda.empty_cache()
    for paths in member_snapshot_paths:
        for snapshot in paths.values():
            snapshot.unlink()
    snapshot_root.rmdir()
    summary = {
        "format": FORMAT,
        "status": "source_only_checkpoint_selection_complete",
        "held_out_body": args.held_out_body,
        "source_bodies": preflight["source_bodies"],
        "canonical_state_schema": CANONICAL_STATE_SCHEMA,
        "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
        "event_age_contract": event_age_contract(),
        "terminal_horizon_contract": terminal_horizon_contract(),
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": preflight[
            "event_derivation_implementation_sha256"
        ],
        "trainer_file_sha256": trainer_file_sha256,
        "candidate_rank_contract": summary_candidate_rank_contract(
            args.ablation_variant
        ),
        "ablation": ablation_contract(args.ablation_variant),
        "training_budget": {
            "steps_per_member": args.steps,
            "eval_every_steps": args.eval_every,
            "batch_size_rows": args.batch_size,
            "learning_rate": args.learning_rate,
            "ensemble_members": len(args.ensemble_seeds),
        },
        "mixed_outcome_source_decisions": mixed_outcome_decisions,
        "source_negative_to_positive_ratio": source_negative_to_positive_ratio,
        "ensemble_bootstrap_effect_support": bootstrap_support,
        "ensemble_proper_bootstrap_outcome_support": proper_bootstrap_support,
        "success_probability_training_loss": "unweighted_proper_binary_cross_entropy",
        "checkpoint_selection_primary": (
            "five_member_standardized_one_deviation_source_validation_surrogate"
        ),
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "rank_aggregation": standardized_rank_ensemble_contract(),
            "selected_step": best_ensemble_step,
            "selected_key": list(best_ensemble_key),
            "selected_ensemble_candidate_ranking": best_ensemble_ranking,
            "selected_mean_member_diagnostic_multitask_score": (
                best_ensemble_diagnostic
            ),
            "heldout_rows_used": 0,
            "evaluated_common_steps": common_step_selection_audit,
        },
        "members": members,
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "heldout_specific_trainable_parameters": 0,
        "actor_frozen": True,
        "same_ordered_candidate_set_required_for_live_evaluation": True,
        "task_success_evaluation_authorized": False,
        "next_required_stage": "label_blind_live_candidate_scoring_then_paired_success_evaluator",
        "preflight": preflight,
    }
    core.atomic_json(output / "training_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "train-fold"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--held-out-body", choices=BODIES, required=True)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--ablation-variant", choices=ABLATION_VARIANTS, default="full"
    )
    parser.add_argument(
        "--ensemble-seeds", nargs=5, type=int,
        default=[20260901, 20260902, 20260903, 20260904, 20260905],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = load_binding(args.binding, args.binding_sha256)
    if args.mode == "preflight":
        receipt = build_preflight_receipt(
            audit, held_out_body=args.held_out_body, split_seed=args.split_seed
        )
        print("PREFLIGHT=" + json.dumps(receipt, sort_keys=True))
        return
    if args.output is None:
        raise FiveBodyContractError("train-fold requires --output")
    if args.steps <= 0 or args.eval_every <= 0:
        raise FiveBodyContractError("steps/eval-every must be positive")
    print("TRAINING=" + json.dumps(_train_fold(args, audit), sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ABLATION_VARIANTS",
    "ACTOR_FORMAT", "BINDING_FORMAT", "BODIES", "CANONICAL_ACTION_SCHEMA",
    "CANONICAL_STATE_SCHEMA", "CANDIDATE_NOISE_CONTRACT",
    "CANDIDATE_RANK_FEATURE_DIM", "CANDIDATE_RANK_FEATURE_SCHEMA",
    "BRANCH_DIAGNOSTIC_CONTRACT",
    "CompleteDecisionBatchSampler", "MacroBalancedRankDecisionBatchSampler",
    "DENSE_FAILURE_RANK_WEIGHT", "DENSE_GOAL_PROGRESS_TEMPERATURE_METERS",
    "CONDITIONS", "FORMAT",
    "EffectAlignedSharedEventHead", "FiveBodyContractError", "MANIFEST_FORMAT",
    "EVENT_SPEC_SHA256", "MATERIALIZATION_FORMAT", "MODEL_FAMILY",
    "OBJECT_EFFECT_SCHEMA", "EVENT_AGE_CONTRACT", "event_age_contract",
    "REQUIRED_ARRAYS",
    "STANDARDIZED_RANK_ENSEMBLE_CONTRACT",
    "TERMINAL_SUPERVISION_CONTRACT",
    "ablation_contract", "ablation_selection_components",
    "aggregate_standardized_rank_scores",
    "build_preflight_receipt", "canonical_sha256",
    "candidate_checkpoint_selection_key", "checkpoint_candidate_rank_contract",
    "effect_preserving_group_bootstrap_weights", "load_binding",
    "proper_outcome_preserving_group_bootstrap_weights",
    "evaluate_candidate_ranking", "materialize_source_rows", "sha256_file",
    "sha256_tree", "source_group_split", "summary_candidate_rank_contract",
    "standardized_rank_ensemble_contract",
    "validate_actor_authority", "validate_body_manifest",
    "validate_materialization_receipt",
]
