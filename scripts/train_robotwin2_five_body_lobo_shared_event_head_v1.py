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
import robotwin2_move_can_pot_analytic_event_spec_v2 as analytic_event
import verify_robotwin2_move_can_pot_public_materialization_v1 as public_materialization


FORMAT = "etsf_robotwin2_five_body_lobo_shared_event_head_v1"
MODEL_FAMILY = "terminal_consequence_utility_shared_event_head_v12"
BINDING_FORMAT = "etsf_robotwin2_five_body_lobo_training_binding_v2_endpose_frame"
MANIFEST_FORMAT = "etsf_robotwin2_canonical_transition_manifest_v2_endpose_frame"
SUPPLEMENT_BINDING_FORMAT = (
    "etsf_robotwin2_five_body_proper_world_utility_rank_supplement_binding_v3_endpose_frame"
)
SUPPLEMENT_MANIFEST_FORMAT = (
    "etsf_robotwin2_proper_world_utility_rank_supplement_manifest_v3_endpose_frame"
)
SUPPLEMENT_COLLECTOR_FORMAT = (
    "etsf_robotwin2_scripted_expert_root_actor_branches_v3_endpose_frame"
)
SUPPLEMENT_MATERIALIZER_FORMAT = (
    "etsf_robotwin2_scripted_expert_root_supplement_binding_materializer_v3_endpose_frame"
)
ACTOR_FORMAT = "etsf_robotwin2_frozen_native_actor_authority_v2_endpose_frame"
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
STATE_ACTION_FRAME_CONTRACT = {
    "format": "etsf_robotwin2_native_ee16_state_action_frame_v2",
    "training_state_source": "public_hdf5_endpose_left_right_endpose",
    "runtime_state_api": "task.get_arm_pose(left/right)",
    "runtime_state_pose_semantics": "robot.get_*_ee_pose(is_endpose=False)",
    "native_action_pose_semantics": (
        "same_absolute_world_ee_frame_as_training_endpose"
    ),
    "environment_call": "task.take_action(native_ee16, action_type=ee)",
    "pose_convention": "xyz_plus_quaternion_wxyz",
    "tcp_tool_axis_offset_m_excluded": 0.12,
    "state_and_action_same_frame": True,
}
CANDIDATE_RANK_FEATURE_SCHEMA = {
    "post_expected_stage_progress": (0, 1),
    "next_event_advance_rate": (1, 2),
    "success_probability": (2, 3),
    "no_unrecovered_regression_probability": (3, 4),
    "short_goal_progress_benefit": (4, 5),
    "short_goal_progress_uncertainty_risk": (5, 6),
    "terminal_expected_stage_progress": (6, 7),
    "terminal_goal_progress_benefit": (7, 8),
    "terminal_goal_progress_uncertainty_risk": (8, 9),
}
CANDIDATE_RANK_FEATURE_DIM = max(
    stop for _start, stop in CANDIDATE_RANK_FEATURE_SCHEMA.values()
)
MONOTONE_BENEFIT_FEATURES = (
    "post_expected_stage_progress",
    "next_event_advance_rate",
    "success_probability",
    "no_unrecovered_regression_probability",
    "short_goal_progress_benefit",
    "terminal_expected_stage_progress",
    "terminal_goal_progress_benefit",
)
MONOTONE_RISK_FEATURES = (
    "short_goal_progress_uncertainty_risk",
    "terminal_goal_progress_uncertainty_risk",
)
GOAL_PROGRESS_NORMALIZATION_METERS = 0.02
OBJECT_STUDENT_T_DOF = 3.0
TERMINAL_PROGRESS_STUDENT_T_DOF = 3.0
CONSEQUENCE_LOG_SCALE_MIN = -7.0
TERMINAL_EVENT_LOSS_WEIGHT = 0.5
TERMINAL_EVENT_ORDINAL_RPS_LOSS_WEIGHT = 0.25
TERMINAL_GOAL_PROGRESS_LOSS_WEIGHT = 0.5
SUPPLEMENT_PROPER_LOSS_WEIGHT = 0.25
SUPPLEMENT_RANK_LOSS_WEIGHT = 0.25
SUPPLEMENT_USAGE_CONTRACT = {
    "outer_lobo_source_only": True,
    "source_train_bodies_per_outer_fold": 3,
    "label_blind_inner_cross_body_proper_validation": True,
    "proper_validation_bodies_per_outer_fold": 1,
    "proper_validation_body_selection": (
        "sha256_split_seed_ordered_five_body_cycle_successor_derangement"
    ),
    "multitask_proper_loss": True,
    "robust_object_effect_proper_loss": True,
    "terminal_event_proper_loss": True,
    "terminal_goal_progress_proper_loss": True,
    "normalization_or_baseline_fit": False,
    "candidate_rank_or_utility_loss": "source_train_lane_only",
    "candidate_rank_or_utility_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
    "candidate_rank_updates": (
        "bounded_monotone_utility_only_from_detached_consequence_features"
    ),
    "semantic_comparative_loss": False,
    "source_validation_or_checkpoint_selection": (
        "proper_only_fixed_weight_0.25_no_rank_selection"
    ),
    "proper_checkpoint_selection_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
    "calibration_diagnostics": True,
    "calibration_fit": False,
    "heldout_payload_access": False,
}
EXPERT_ROOT_PROVENANCE_CONTRACT = {
    "root_prefix_policy": "robotwin_scripted_expert",
    "root_definition": "fresh_frozen_actor_branch_initialized_from_expert_state",
    "expert_prefix_claimed_actor_on_policy": False,
    "candidate_policy": "same_frozen_native_actor_as_primary_binding",
    "continuation_policy": "same_frozen_native_actor_as_primary_binding",
    "expert_terminal_outcome_used_as_branch_label": False,
    "fresh_branch_horizon_starts_at_root": True,
    "formal_actor_prefix_distribution_claimed": False,
}
SUPPLEMENT_TARGET_EVENTS = ("e12", "e3", "e4")
SUPPLEMENT_HORIZON_SCHEDULE = (10, 25, 50, 100, 200)
SUPPLEMENT_RESERVE_SEED_START = 2026081000
SUPPLEMENT_RESERVE_SEEDS_PER_SLOT = 16
SUPPLEMENT_RESERVE_SEED_STOP_EXCLUSIVE = (
    SUPPLEMENT_RESERVE_SEED_START
    + len(BODIES)
    * len(CONDITIONS)
    * len(SUPPLEMENT_HORIZON_SCHEDULE)
    * SUPPLEMENT_RESERVE_SEEDS_PER_SLOT
)
SUPPLEMENT_FORMAL_PRIMARY_SEED_START = 2026082000
SUPPLEMENT_EXPECTED_DECISIONS_PER_BODY = (
    len(CONDITIONS)
    * len(SUPPLEMENT_HORIZON_SCHEDULE)
    * len(SUPPLEMENT_TARGET_EVENTS)
)
SUPPLEMENT_ROOT_SELECTION_CONTRACT = {
    "controller": "public_RoboTwin_move_can_pot.play_once",
    "observation_granularity": "every_successful_sapien_scene_step",
    "targets": list(SUPPLEMENT_TARGET_EVENTS),
    "selection": "first_physical_sample_whose_frozen_analytic_event_equals_target",
    "one_root_per_target_per_scene_seed": True,
    "adjacent_same_event_frames_used_as_additional_roots": False,
    "e4_must_be_nonterminal_simulator_success": True,
    "root_selection_reads_actor_branch_outcomes": False,
    "triplet_acceptance": (
        "e12_e3_e4_all_exist_and_fresh_restore_canonicalize_before_any_actor_"
        "candidate_outcome"
    ),
    "missing_target_policy": (
        "record_reject_and_advance_same_body_condition_horizon_slot_ordered_"
        "reserve_before_any_actor_candidate_outcome"
    ),
    "reserve_selection_reads_actor_candidate_outcomes": False,
    "cross_body_common_success_seed_selection": False,
    "planner_after_root": "scripted_expert_ends_and_is_never_used_for_continuation",
}
SUPPLEMENT_HORIZON_CONTRACT = {
    "values": list(SUPPLEMENT_HORIZON_SCHEDULE),
    "slot_count_per_condition": len(SUPPLEMENT_HORIZON_SCHEDULE),
    "binding": (
        "body_condition_horizon_slot_has_pre_registered_ordered_reserve_"
        "seeds_before_any_rollout"
    ),
    "same_horizon_for_e12_e3_e4_of_one_seed": True,
    "new_actor_branch_take_action_count_at_root": 0,
    "remaining_action_budget_at_root_equals_bound_horizon": True,
    "expert_physics_steps_or_planner_frames_used_to_compute_horizon": False,
    "candidate_or_terminal_outcomes_used_to_choose_horizon": False,
    "actor_query_stride_actions": 5,
}
SUPPLEMENT_RESERVE_ROSTER_CONTRACT = {
    "scope": "body_local_condition_local_horizon_slot_local",
    "ordered_reserve_seeds_per_slot": SUPPLEMENT_RESERVE_SEEDS_PER_SLOT,
    "selection": "first_complete_canonicalizable_e12_e3_e4_triplet",
    "selection_occurs_before_actor_candidate_outcomes": True,
    "rejected_attempts_are_audited": True,
    "rejected_seed_candidate_outcomes_executed": False,
    "one_selected_seed_per_slot": True,
    "heldout_body_availability_changes_source_body_roster": False,
    "python_rng_seeded_from_requested_scene_seed_before_each_fresh_setup": True,
    "reserve_seed_start_inclusive": SUPPLEMENT_RESERVE_SEED_START,
    "reserve_seed_stop_exclusive": SUPPLEMENT_RESERVE_SEED_STOP_EXCLUSIVE,
    "formal_primary_seed_start": SUPPLEMENT_FORMAL_PRIMARY_SEED_START,
}
DENSE_FAILURE_RANK_WEIGHT = 0.1
DENSE_ONLY_RANK_WEIGHT = 1.0
SEMANTIC_COMPARATIVE_GRADIENT_BUDGET = 0.1
SEMANTIC_GRADIENT_SCALE_CAP = 1.0
TERMINAL_FILM_MODULATION_BOUND = 0.1
EVENT_UTILITY_RESIDUAL_BOUND = 1.0
EVENT_PRIORITY_SECONDARY_SCALE = 0.05
DENSE_RANK_LABEL_EQUALITY_TOLERANCE = 1e-4
MINIMUM_COMPARATIVE_VALIDATION_SEED_CLUSTERS = 10
MINIMUM_COMPARATIVE_VALIDATION_REQUESTED_SEEDS = 2
MINIMUM_COMPARATIVE_VALIDATION_BODY_CONDITION_UNITS = 4
MINIMUM_COMPARATIVE_VALIDATION_BODIES = 2
CROSS_BODY_STANDARDIZED_INPUT_CLIP = 5.0
ONE_DEVIATION_ESTIMAND = (
    "one_candidate_deviation_then_frozen_actor_continuation_not_"
    "recursive_closed_loop_delta_success_rate"
)
EPISTEMIC_RANK_RISK_WEIGHT = 0.25
RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT = {
    "format": "etsf_bounded_utility_epistemic_lcb_ensemble_v1",
    "member_count": 5,
    "candidate_count": CANDIDATE_COUNT,
    "member_score_contract": "same_bounded_monotone_physical_utility",
    "population_std_correction": 0,
    "epistemic_risk_weight": EPISTEMIC_RANK_RISK_WEIGHT,
    "aggregation": "member_mean_minus_weight_times_member_population_std",
    "within_member_candidate_standardization": False,
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
ROOT_POSE_RESTORE_ATOL = 2.384185791015625e-7
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
    "format": "etsf_robotwin2_candidate_branch_diagnostics_v2_endpose_frame",
    "first_executed": "successful_or_physics_advancing_actions_in_planned_first_chunk",
    "branch_error": "all_false_execution_exception_invalidates_complete_decision",
    "candidate_action_pairwise_rms": (
        "symmetric_raw_canonical_effect_rms_over_planned_first_five_actions"
    ),
    "candidate_first_token_translation_norm_m": (
        "label_free_left_right_translation_norm_from_same_frame_root_state_to_"
        "candidate_token_zero"
    ),
    "candidate_later_token_translation_norm_median_m": (
        "label_free_left_right_median_translation_norm_between_subsequent_"
        "candidate_tokens"
    ),
}
SUPPLEMENT_ACTOR_BRANCH_CONTRACT = {
    "candidate_count": CANDIDATE_COUNT,
    "candidate_generator": (
        "collect_robotwin2_five_body_ee_candidate_branches_v1.generate_candidates"
    ),
    "fresh_scene_candidate_evaluator": (
        "collect_robotwin2_five_body_ee_candidate_branches_v1._evaluate_candidate"
    ),
    "snapshot_restore_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
    "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
    "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
    "expert_actions_after_root": 0,
    "continuation_controller": "same_frozen_actor_as_four_root_candidates",
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
            "proper_horizon_coherent_terminal_eK_success_probability"
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
            else "negative_log_coherent_success_probability_mass"
        ),
        "all_failure_dense_objective_weight": (
            0.0 if variant == "success_only" else DENSE_FAILURE_RANK_WEIGHT
        ),
        "dense_only_objective_weight": (
            0.0 if variant == "success_only" else DENSE_ONLY_RANK_WEIGHT
        ),
        "all_failure_dense_informative_labels_only": True,
        "candidate_rank_requires_real_comparative_supervision": (
            variant != "success_only"
        ),
        "synthetic_success_labels_allowed": False,
        "dense_target_order": (
            "none"
            if variant == "success_only"
            else "strict_terminal_max_event_then_soft_goal_progress_temperature_0.02m"
        ),
        "training_streams": (
            "uniform_proper_likelihood_plus_macro_balanced_rank_only"
        ),
        "rank_ensemble_aggregation": risk_adjusted_rank_ensemble_contract(),
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
            "proper_success_probability_exactly_terminal_event_eK_probability"
        ]
    elif variant == "no_time_duration":
        feature_blocks = [
            "post_expected_stage_progress",
            "next_event_advance_rate_forced_zero",
            "success_probability",
            "no_unrecovered_regression_probability",
            "short_goal_progress_benefit",
            "short_goal_progress_uncertainty_risk",
            "terminal_expected_stage_progress",
            "terminal_goal_progress_benefit",
            "terminal_goal_progress_uncertainty_risk",
        ]
    elif variant == "no_object_effect":
        feature_blocks = [
            "post_expected_stage_progress",
            "next_event_advance_rate",
            "success_probability",
            "no_unrecovered_regression_probability",
            "short_goal_progress_benefit_forced_zero",
            "short_goal_progress_uncertainty_risk_forced_zero",
            "terminal_expected_stage_progress",
            "terminal_goal_progress_benefit_forced_zero",
            "terminal_goal_progress_uncertainty_risk_forced_zero",
        ]
    else:
        feature_blocks = [
            "post_expected_stage_progress",
            "next_event_advance_rate",
            "success_probability",
            "no_unrecovered_regression_probability",
            "short_goal_progress_benefit",
            "short_goal_progress_uncertainty_risk",
            "terminal_expected_stage_progress",
            "terminal_goal_progress_benefit",
            "terminal_goal_progress_uncertainty_risk",
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
        "terminal_context_fusion": (
            "disabled_time_inputs_with_shared_residual_trunk"
            if variant == "no_time_duration"
            else "bounded_horizon_conditioned_film_residual_trunk"
        ),
        "terminal_candidate_relative_predictions_condition_on_horizon": (
            variant != "no_time_duration"
        ),
        "terminal_film_modulation_bound": TERMINAL_FILM_MODULATION_BOUND,
        "remaining_action_budget_has_direct_rank_path": False,
        "event_age_has_numeric_score_path": variant != "no_time_duration",
        "direct_transitioned_or_clock_hidden_rank_path": False,
        "rank_inputs_are_detached_consequence_predictions": variant != "success_only",
        "rank_utility": (
            "bounded_invariant_monotone_global_plus_canonical_event_residual_utility"
            if variant != "success_only"
            else "coherent_bounded_terminal_success_probability"
        ),
        "utility_conditioning": (
            "canonical_current_event_global_logits_plus_bounded_event_residual"
            if variant != "success_only"
            else "coherent_success_probability_only"
        ),
        "event_utility_residual_bound": (
            EVENT_UTILITY_RESIDUAL_BOUND if variant != "success_only" else 0.0
        ),
        "event_priority_primary": (
            "terminal_expected_stage_progress"
            if variant != "success_only"
            else "terminal_eK_probability"
        ),
        "event_priority_secondary_scale": (
            EVENT_PRIORITY_SECONDARY_SCALE if variant != "success_only" else 0.0
        ),
        "deterministic_one_event_step_dominates_all_secondary_features": (
            variant != "success_only"
        ),
        "strict_lexicographic_for_arbitrary_expected_stage_difference": False,
        "body_conditioned_utility_parameters": False,
        "raw_world_frame_object_axes_in_rank_input": False,
        "cross_feature_layer_normalization": False,
        "cross_body_input_normalization": (
            "source_train_only_zscore_then_fixed_symmetric_clip"
        ),
        "cross_body_standardized_input_clip": CROSS_BODY_STANDARDIZED_INPUT_CLIP,
        "heldout_statistics_used_for_input_normalization_or_clip": False,
        "goal_progress_definition": (
            "post_event_mixture_expectation_of_norm(state_relative_goal_xyz)-"
            "norm(state_relative_goal_xyz-object_translation_component_mean)"
        ),
        "goal_progress_benefit_transform": (
            "categorical_event_mixture_expectation_of_0.5_times_one_plus_"
            "softsign_conditional_location_progress_over_0.02m"
        ),
        "goal_progress_uncertainty_definition": (
            "post_event_mixture_total_std_combining_student_t3_within_event_"
            "and_between_event_goal_progress_variance_mapped_as_std_over_std_"
            "plus_0.02m"
        ),
        "object_effect_joint_distribution": (
            "p(post_event)_times_independent_student_t3_dof3_object_delta_"
            "conditioned_on_post_event"
        ),
        "terminal_goal_joint_distribution": (
            "p(terminal_event)_times_student_t3_dof3_goal_progress_conditioned_"
            "on_terminal_event"
        ),
        "conditional_consequence_proper_nll": (
            "gather_observed_event_component_on_uniform_proper_stream"
        ),
        "legacy_consequence_mean_semantics": "categorical_event_mixture_mean",
        "legacy_consequence_log_scale_semantics": (
            "log_of_total_mixture_std_divided_by_sqrt3_moment_equivalent_"
            "student_t3_scale"
        ),
        "aleatoric_consequence_risk": (
            "law_of_total_variance_within_event_student_t3_plus_between_event"
        ),
        "duration_joint_distribution": (
            "p(next_event)_times_lognormal_log1p_duration_conditioned_on_"
            "next_event"
        ),
        "duration_observed_likelihood": (
            "observed_next_event_component_nll_with_separate_next_event_ce"
        ),
        "duration_censored_likelihood": (
            "negative_log_sum_next_event_probability_times_component_survival"
        ),
        "censored_duration_updates_next_event_probability": True,
        "duration_component_clock_gradient_updates_semantic_transition": False,
        "censored_survival_event_evidence_updates_semantic_transition": True,
        "duration_legacy_scalar_interface": (
            "moment_equivalent_gaussian_in_log1p_time_preserving_first_two_"
            "mixture_moments"
        ),
        "duration_aleatoric_uncertainty": (
            "within_next_event_gaussian_plus_between_next_event_log1p_time_"
            "variance"
        ),
        "native_probability_outputs": [
            "success_probability",
            "failure_probability",
            "conditional_recovery_probability",
            "joint_recovery_probability",
        ],
        "native_aleatoric_entropy_outputs": [
            "post_event_aleatoric_entropy",
            "next_event_aleatoric_entropy",
            "terminal_event_aleatoric_entropy",
            "success_aleatoric_entropy",
            "conditional_recovery_aleatoric_entropy",
            "joint_recovery_aleatoric_entropy",
        ],
        "epistemic_uncertainty_source": "five_member_source_group_bootstrap_ensemble",
        "pairwise_rank_loss_enabled": False,
        "group_listwise_success_mass_loss_enabled": variant != "success_only",
        "all_failure_dense_listwise_loss_enabled": variant != "success_only",
        "all_failure_dense_listwise_weight": (
            0.0 if variant == "success_only" else DENSE_FAILURE_RANK_WEIGHT
        ),
        "dense_only_listwise_weight": (
            0.0 if variant == "success_only" else DENSE_ONLY_RANK_WEIGHT
        ),
        "all_failure_dense_informative_labels_only": True,
        "dense_rank_label_equality_tolerance": DENSE_RANK_LABEL_EQUALITY_TOLERANCE,
        "candidate_rank_requires_real_comparative_supervision": (
            variant != "success_only"
        ),
        "synthetic_success_labels_allowed": False,
        "dense_target_requires_full_continuation": True,
        "dense_goal_progress_temperature_meters": (
            None
            if variant in {"success_only", "no_object_effect"}
            else DENSE_GOAL_PROGRESS_TEMPERATURE_METERS
        ),
        "dense_goal_progress_target_transform": (
            "disabled"
            if variant in {"success_only", "no_object_effect"}
            else (
                "uniform_if_max_event_goal_spread_lte_1e-4_else_"
                "softmax_of_0.5_times_one_plus_softsign_progress_over_0.02m"
            )
        ),
        "dense_goal_progress_prediction_transform": (
            "disabled"
            if variant in {"success_only", "no_object_effect"}
            else (
                "0.5_times_one_plus_softsign_predicted_progress_over_0.02m"
            )
        ),
        "dense_goal_progress_unbounded_logits": False,
        "proper_and_rank_batches_are_separate": True,
        "mixed_rank_bootstrap": "one_plus_poisson",
        "dense_rank_bootstrap": "poisson",
        "rank_macro_strata": "body_condition_current_event",
        "rank_ensemble_aggregation": risk_adjusted_rank_ensemble_contract(),
        "utility_rank_loss_updates_clock_or_duration_heads": False,
        "utility_rank_loss_updates_semantic_action_transition": False,
        "utility_rank_loss_updates_consequence_predictors": False,
        "semantic_comparative_loss_updates_terminal_predictors": (
            variant != "success_only"
        ),
        "semantic_comparative_gradient_budget_relative_to_active_union_proper": (
            0.0
            if variant == "success_only"
            else SEMANTIC_COMPARATIVE_GRADIENT_BUDGET
        ),
        "semantic_comparative_gradient_budget_scope": (
            "disabled"
            if variant == "success_only"
            else "single_active_union_semantic_action_transition_terminal_trunk_and_location_heads"
        ),
        "semantic_comparative_gradient_budget_proper_reference": (
            "primary_plus_fixed_weight_source_train_supplement_proper_losses"
        ),
        "semantic_comparative_scale_heads_excluded": True,
        "semantic_comparative_gradient_cap_applications": (
            0 if variant == "success_only" else 1
        ),
        "semantic_gradient_scale_cap": SEMANTIC_GRADIENT_SCALE_CAP,
        "world_and_utility_gradient_clipping_are_separate": True,
        "checkpoint_selection_calibration_guard": (
            "primary_source_body_condition_macro_seed_clustered_strict_proper_"
            "plus_fixed_0.25_label_blind_inner_cross_body_supplement_strict_"
            "proper_with_independent_variance_one_standard_error_then_formal_"
            "only_minimum_10_comparative_seed_clusters_across_minimum_2_"
            "requested_seeds_4_body_condition_units_2_bodies"
        ),
        "supplement_proper_validation_body_selection": (
            "sha256_split_seed_ordered_five_body_cycle_successor_derangement"
        ),
        "supplement_proper_validation_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
        "supplement_validation_used_for_rank_comparison": False,
        "supplement_validation_calibration_diagnostics": True,
        "supplement_validation_calibration_fit": False,
        "strict_proper_components": (
            ["success_binary_nll"]
            if variant == "success_only"
            else [
                "post_event_categorical_nll_weight_1.0",
                "next_event_categorical_nll_weight_0.5",
                *(
                    []
                    if variant == "no_time_duration"
                    else [
                        "duration_next_event_competing_risks_censored_"
                        "lognormal_nll_weight_0.5"
                    ]
                ),
                "success_binary_nll",
                "recovery_binary_nll_weight_0.5_when_supervised",
                *(
                    []
                    if variant == "no_object_effect"
                    else ["object_given_post_event_student_t3_nll_weight_0.5"]
                ),
                "terminal_event_categorical_nll_weight_0.5",
                "terminal_event_ordinal_ranked_probability_score_weight_0.25",
                *(
                    []
                    if variant == "no_object_effect"
                    else [
                        "terminal_goal_given_terminal_event_student_t3_nll_weight_0.5"
                    ]
                ),
            ]
        ),
        "terminal_event_loss": (
            "proper_categorical_cross_entropy_plus_strictly_proper_ordinal_"
            "ranked_probability_score_uniform_stream"
        ),
        "terminal_stage_progress_loss": (
            "terminal_event_cdf_ranked_probability_score_weight_0.25"
        ),
        "terminal_goal_progress_loss": (
            "proper_observed_terminal_event_conditional_student_t3_nll_"
            "uniform_stream"
        ),
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
        "terminal_context_fusion": checkpoint["terminal_context_fusion"],
        "terminal_candidate_relative_predictions_condition_on_horizon": checkpoint[
            "terminal_candidate_relative_predictions_condition_on_horizon"
        ],
        "terminal_film_modulation_bound": checkpoint[
            "terminal_film_modulation_bound"
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
        "dense_only_listwise_weight": checkpoint[
            "dense_only_listwise_weight"
        ],
        "all_failure_dense_informative_labels_only": checkpoint[
            "all_failure_dense_informative_labels_only"
        ],
        "dense_rank_label_equality_tolerance": checkpoint[
            "dense_rank_label_equality_tolerance"
        ],
        "candidate_rank_requires_real_comparative_supervision": checkpoint[
            "candidate_rank_requires_real_comparative_supervision"
        ],
        "synthetic_success_labels_allowed": checkpoint[
            "synthetic_success_labels_allowed"
        ],
        "dense_target_requires_full_continuation": True,
        "dense_goal_progress_temperature_meters": checkpoint[
            "dense_goal_progress_temperature_meters"
        ],
        "dense_goal_progress_target_transform": checkpoint[
            "dense_goal_progress_target_transform"
        ],
        "dense_goal_progress_prediction_transform": checkpoint[
            "dense_goal_progress_prediction_transform"
        ],
        "dense_goal_progress_unbounded_logits": checkpoint[
            "dense_goal_progress_unbounded_logits"
        ],
        "proper_and_rank_batches_are_separate": True,
        "mixed_rank_bootstrap": "one_plus_poisson",
        "dense_rank_bootstrap": "poisson",
        "rank_macro_strata": "body_condition_current_event",
        "rank_ensemble_aggregation": risk_adjusted_rank_ensemble_contract(),
        "utility_rank_loss_updates_clock_or_duration_heads": checkpoint[
            "utility_rank_loss_updates_clock_or_duration_heads"
        ],
        "utility_conditioning": checkpoint["utility_conditioning"],
        "event_utility_residual_bound": checkpoint[
            "event_utility_residual_bound"
        ],
        "event_priority_primary": checkpoint["event_priority_primary"],
        "event_priority_secondary_scale": checkpoint[
            "event_priority_secondary_scale"
        ],
        "deterministic_one_event_step_dominates_all_secondary_features": checkpoint[
            "deterministic_one_event_step_dominates_all_secondary_features"
        ],
        "strict_lexicographic_for_arbitrary_expected_stage_difference": checkpoint[
            "strict_lexicographic_for_arbitrary_expected_stage_difference"
        ],
        "body_conditioned_utility_parameters": checkpoint[
            "body_conditioned_utility_parameters"
        ],
        "utility_rank_loss_updates_semantic_action_transition": checkpoint[
            "utility_rank_loss_updates_semantic_action_transition"
        ],
        "utility_rank_loss_updates_consequence_predictors": checkpoint[
            "utility_rank_loss_updates_consequence_predictors"
        ],
        "semantic_comparative_loss_updates_terminal_predictors": checkpoint[
            "semantic_comparative_loss_updates_terminal_predictors"
        ],
        "semantic_comparative_gradient_budget_relative_to_active_union_proper": checkpoint[
            "semantic_comparative_gradient_budget_relative_to_active_union_proper"
        ],
        "semantic_comparative_gradient_budget_scope": checkpoint[
            "semantic_comparative_gradient_budget_scope"
        ],
        "semantic_comparative_gradient_budget_proper_reference": checkpoint[
            "semantic_comparative_gradient_budget_proper_reference"
        ],
        "semantic_comparative_scale_heads_excluded": checkpoint[
            "semantic_comparative_scale_heads_excluded"
        ],
        "semantic_comparative_gradient_cap_applications": checkpoint[
            "semantic_comparative_gradient_cap_applications"
        ],
        "semantic_gradient_scale_cap": checkpoint[
            "semantic_gradient_scale_cap"
        ],
        "world_and_utility_gradient_clipping_are_separate": checkpoint[
            "world_and_utility_gradient_clipping_are_separate"
        ],
        "checkpoint_selection_calibration_guard": checkpoint[
            "checkpoint_selection_calibration_guard"
        ],
        "supplement_proper_validation_body_selection": checkpoint[
            "supplement_proper_validation_body_selection"
        ],
        "supplement_proper_validation_weight": checkpoint[
            "supplement_proper_validation_weight"
        ],
        "supplement_validation_used_for_rank_comparison": checkpoint[
            "supplement_validation_used_for_rank_comparison"
        ],
        "supplement_validation_calibration_diagnostics": checkpoint[
            "supplement_validation_calibration_diagnostics"
        ],
        "supplement_validation_calibration_fit": checkpoint[
            "supplement_validation_calibration_fit"
        ],
        "strict_proper_components": checkpoint[
            "strict_proper_components"
        ],
        "direct_transitioned_or_clock_hidden_rank_path": checkpoint[
            "direct_transitioned_or_clock_hidden_rank_path"
        ],
        "rank_inputs_are_detached_consequence_predictions": checkpoint[
            "rank_inputs_are_detached_consequence_predictions"
        ],
        "terminal_event_loss": checkpoint["terminal_event_loss"],
        "terminal_stage_progress_loss": checkpoint[
            "terminal_stage_progress_loss"
        ],
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
        "goal_progress_benefit_transform": checkpoint[
            "goal_progress_benefit_transform"
        ],
        "goal_progress_uncertainty_definition": checkpoint[
            "goal_progress_uncertainty_definition"
        ],
        "object_effect_joint_distribution": checkpoint[
            "object_effect_joint_distribution"
        ],
        "terminal_goal_joint_distribution": checkpoint[
            "terminal_goal_joint_distribution"
        ],
        "conditional_consequence_proper_nll": checkpoint[
            "conditional_consequence_proper_nll"
        ],
        "legacy_consequence_mean_semantics": checkpoint[
            "legacy_consequence_mean_semantics"
        ],
        "legacy_consequence_log_scale_semantics": checkpoint[
            "legacy_consequence_log_scale_semantics"
        ],
        "aleatoric_consequence_risk": checkpoint[
            "aleatoric_consequence_risk"
        ],
        "duration_joint_distribution": checkpoint[
            "duration_joint_distribution"
        ],
        "duration_observed_likelihood": checkpoint[
            "duration_observed_likelihood"
        ],
        "duration_censored_likelihood": checkpoint[
            "duration_censored_likelihood"
        ],
        "censored_duration_updates_next_event_probability": checkpoint[
            "censored_duration_updates_next_event_probability"
        ],
        "duration_component_clock_gradient_updates_semantic_transition": checkpoint[
            "duration_component_clock_gradient_updates_semantic_transition"
        ],
        "censored_survival_event_evidence_updates_semantic_transition": checkpoint[
            "censored_survival_event_evidence_updates_semantic_transition"
        ],
        "duration_legacy_scalar_interface": checkpoint[
            "duration_legacy_scalar_interface"
        ],
        "duration_aleatoric_uncertainty": checkpoint[
            "duration_aleatoric_uncertainty"
        ],
        "native_probability_outputs": checkpoint[
            "native_probability_outputs"
        ],
        "native_aleatoric_entropy_outputs": checkpoint[
            "native_aleatoric_entropy_outputs"
        ],
        "epistemic_uncertainty_source": checkpoint[
            "epistemic_uncertainty_source"
        ],
        "cross_body_input_normalization": checkpoint[
            "cross_body_input_normalization"
        ],
        "cross_body_standardized_input_clip": checkpoint[
            "cross_body_standardized_input_clip"
        ],
        "heldout_statistics_used_for_input_normalization_or_clip": checkpoint[
            "heldout_statistics_used_for_input_normalization_or_clip"
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
    "success_height_reference_z",
    "dt",
}


class FiveBodyContractError(RuntimeError):
    """A five-body training authority or payload failed closed."""


def validate_ensemble_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Freeze five distinct ensemble initializations before any fold work."""

    values = tuple(seeds)
    if (
        len(values) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values)
        or len(set(values)) != 5
    ):
        raise FiveBodyContractError(
            "ensemble seeds must be exactly five distinct integers"
        )
    return values


def supplement_horizon_slot_key(condition: str, horizon_slot: int) -> str:
    """Return one immutable body-local supplement slot identity."""

    if (
        condition not in CONDITIONS
        or isinstance(horizon_slot, bool)
        or not isinstance(horizon_slot, int)
        or horizon_slot not in range(len(SUPPLEMENT_HORIZON_SCHEDULE))
    ):
        raise FiveBodyContractError("invalid supplement horizon slot")
    return f"{condition}|horizon_slot={horizon_slot}"


def supplement_reserve_attempt_id(
    condition: str, horizon_slot: int, seed: int
) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FiveBodyContractError("invalid supplement reserve seed")
    return (
        f"{supplement_horizon_slot_key(condition, horizon_slot)}"
        f"|requested_seed={seed}"
    )


def supplement_reserve_group_id(
    condition: str, horizon_slot: int, seed: int, target: str
) -> str:
    if target not in SUPPLEMENT_TARGET_EVENTS:
        raise FiveBodyContractError("unknown scripted root event")
    return (
        f"{supplement_reserve_attempt_id(condition, horizon_slot, seed)}"
        f"|scripted_root={target}"
    )


def supplement_reserve_roster(body: str) -> list[dict[str, Any]]:
    """Build the trainer-owned 10-slot/160-seed body-local reserve roster.

    This intentionally duplicates the immutable public collector design.  The
    production trainer must remain able to validate a materialized binding
    without importing or executing collection code.
    """

    if body not in BODIES:
        raise FiveBodyContractError(f"unknown supplement body {body!r}")
    body_index = BODIES.index(body)
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for horizon_slot, horizon in enumerate(SUPPLEMENT_HORIZON_SCHEDULE):
            global_slot = (
                (body_index * len(CONDITIONS) + condition_index)
                * len(SUPPLEMENT_HORIZON_SCHEDULE)
                + horizon_slot
            )
            first = (
                SUPPLEMENT_RESERVE_SEED_START
                + global_slot * SUPPLEMENT_RESERVE_SEEDS_PER_SLOT
            )
            rows.append(
                {
                    "slot_key": supplement_horizon_slot_key(
                        condition, horizon_slot
                    ),
                    "condition": condition,
                    "horizon_slot": horizon_slot,
                    "remaining_action_budget": int(horizon),
                    "ordered_requested_seeds": list(
                        range(first, first + SUPPLEMENT_RESERVE_SEEDS_PER_SLOT)
                    ),
                }
            )
    flattened = [
        int(seed)
        for row in rows
        for seed in row["ordered_requested_seeds"]
    ]
    expected_count = (
        len(CONDITIONS)
        * len(SUPPLEMENT_HORIZON_SCHEDULE)
        * SUPPLEMENT_RESERVE_SEEDS_PER_SLOT
    )
    if (
        len(flattened) != expected_count
        or len(set(flattened)) != expected_count
        or min(flattened) < SUPPLEMENT_RESERVE_SEED_START
        or max(flattened) >= SUPPLEMENT_RESERVE_SEED_STOP_EXCLUSIVE
        or max(flattened) >= SUPPLEMENT_FORMAL_PRIMARY_SEED_START
    ):
        raise FiveBodyContractError("supplement reserve roster is invalid")
    return rows


def supplement_reserve_horizon_by_seed(body: str) -> dict[int, int]:
    return {
        int(seed): int(row["remaining_action_budget"])
        for row in supplement_reserve_roster(body)
        for seed in row["ordered_requested_seeds"]
    }


def risk_adjusted_rank_ensemble_contract() -> dict[str, Any]:
    """Return the frozen deployment aggregation contract without shared mutation."""

    return dict(RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT)


def event_age_contract() -> dict[str, Any]:
    """Return the frozen physical event-age input contract."""

    return dict(EVENT_AGE_CONTRACT)


def state_action_frame_contract() -> dict[str, Any]:
    """Return the frozen endpose-frame ABI used by actor state and actions."""

    return dict(STATE_ACTION_FRAME_CONTRACT)


def terminal_horizon_contract() -> dict[str, Any]:
    """Return the pre-action finite-horizon conditioning contract."""

    return dict(TERMINAL_HORIZON_CONTRACT)


def aggregate_risk_adjusted_rank_scores(
    member_scores: torch.Tensor,
    *,
    risk_weight: float = EPISTEMIC_RANK_RISK_WEIGHT,
) -> torch.Tensor:
    """Return a lower-confidence utility from five comparable member scores.

    The last axis is one complete four-candidate decision.  Leading axes after
    the member axis are permitted.  The structured utility has one fixed
    physical scale for every member, so cross-member population standard
    deviation is epistemic disagreement rather than an arbitrary logit scale.
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
        float(risk_weight), EPISTEMIC_RANK_RISK_WEIGHT, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FiveBodyContractError("rank ensemble epistemic risk is frozen by contract")
    if not bool(torch.isfinite(member_scores).all()):
        raise FiveBodyContractError("member rank scores contain non-finite values")
    member_mean = member_scores.mean(dim=0)
    epistemic_std = member_scores.std(dim=0, correction=0)
    return member_mean - float(risk_weight) * epistemic_std


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
    if (
        value.get("format") != ACTOR_FORMAT
        or value.get("task") != TASK
        or value.get("state_action_frame_contract")
        != STATE_ACTION_FRAME_CONTRACT
    ):
        raise FiveBodyContractError("actor authority format/task mismatch")
    actors = value.get("actors")
    if not isinstance(actors, Mapping) or set(actors) != set(BODIES):
        raise FiveBodyContractError("actor authority must bind exactly five bodies")
    checkpoint_sha256_by_body: dict[str, str] = {}
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
        checkpoint_sha256_by_body[body] = str(actor["checkpoint_sha256"])
    return {
        "actor_frozen": True,
        "bodies": list(BODIES),
        "candidate_count": CANDIDATE_COUNT,
        "same_ordered_candidate_set": True,
        "checkpoint_sha256_by_body": checkpoint_sha256_by_body,
    }


def validate_body_manifest(
    value: Mapping[str, Any],
    *,
    expected_body: str,
    manifest_dir: Path,
    expected_format: str = MANIFEST_FORMAT,
) -> dict[str, Any]:
    _verify_signed(value, f"{expected_body} canonical manifest")
    if (
        value.get("format") != expected_format
        or value.get("dataset_repo") != DATASET_REPO
        or value.get("dataset_revision") != DATASET_REVISION
        or value.get("task") != TASK
        or value.get("instruction") != DEFAULT_INSTRUCTION
        or value.get("body") != expected_body
        or value.get("state_action_frame_contract")
        != STATE_ACTION_FRAME_CONTRACT
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
        or physical_time.get("event_thresholds")
        != analytic_event.THRESHOLDS
        or physical_time.get("event_chain_success_aligned") is not True
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


def _validate_supplement_reserve_design(
    value: Mapping[str, Any], *, body: str
) -> dict[str, Any]:
    """Validate ordered reserve selection using manifest metadata only."""

    roster = supplement_reserve_roster(body)
    flattened = [
        int(seed)
        for row in roster
        for seed in row["ordered_requested_seeds"]
    ]
    expected_horizons = supplement_reserve_horizon_by_seed(body)
    declared_horizons = value.get("pre_registered_horizon_by_seed")
    try:
        normalized_horizons: dict[int, int] = {}
        for seed, horizon in declared_horizons.items():
            normalized_seed = int(seed)
            if (
                str(seed) != str(normalized_seed)
                or isinstance(horizon, bool)
                or not isinstance(horizon, int)
            ):
                raise ValueError("reserve horizon entry is not canonical integer data")
            normalized_horizons[normalized_seed] = horizon
    except (AttributeError, TypeError, ValueError) as error:
        raise FiveBodyContractError(
            f"{body} supplement reserve horizon map is invalid"
        ) from error
    selected = value.get("selected_seed_by_slot")
    groups = value.get("groups")
    attempts = value.get("attempts")
    if (
        value.get("collection_status") != "complete"
        or value.get("reserve_roster_contract")
        != SUPPLEMENT_RESERVE_ROSTER_CONTRACT
        or value.get("reserve_roster") != roster
        or value.get("pre_registered_seeds") != flattened
        or normalized_horizons != expected_horizons
        or not isinstance(selected, Mapping)
        or set(selected) != {row["slot_key"] for row in roster}
        or not isinstance(groups, list)
        or len(groups) != SUPPLEMENT_EXPECTED_DECISIONS_PER_BODY
        or not isinstance(attempts, list)
    ):
        raise FiveBodyContractError(
            f"{body} supplement reserve design is incomplete"
        )

    attempt_by_id: dict[str, Mapping[str, Any]] = {}
    declared_attempt_order: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise FiveBodyContractError("supplement reserve attempt is not an object")
        attempt_id = attempt.get("attempt_id")
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in attempt_by_id
            or attempt.get("actor_candidate_outcomes_executed_before_selection")
            is not False
        ):
            raise FiveBodyContractError("supplement reserve attempt audit is invalid")
        attempt_by_id[attempt_id] = attempt
        declared_attempt_order.append(attempt_id)

    groups_by_id: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise FiveBodyContractError("supplement group entry must be an object")
        group_id = group.get("group_id")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in groups_by_id
        ):
            raise FiveBodyContractError("supplement group identity is invalid")
        groups_by_id[group_id] = group

    expected_attempt_order: list[str] = []
    expected_group_ids: set[str] = set()
    selected_pairs: set[tuple[str, int]] = set()
    rejected_attempt_count = 0
    for row in roster:
        condition = str(row["condition"])
        slot = int(row["horizon_slot"])
        horizon = int(row["remaining_action_budget"])
        seeds = [int(seed) for seed in row["ordered_requested_seeds"]]
        selected_seed = selected.get(row["slot_key"])
        if (
            isinstance(selected_seed, bool)
            or not isinstance(selected_seed, int)
            or selected_seed not in seeds
        ):
            raise FiveBodyContractError("selected supplement reserve seed is invalid")
        selected_seed = int(selected_seed)
        selected_index = seeds.index(selected_seed)
        slot_attempt_order: list[str] = []
        for rejected_seed in seeds[:selected_index]:
            attempt_id = supplement_reserve_attempt_id(
                condition, slot, rejected_seed
            )
            rejected = attempt_by_id.get(attempt_id)
            if (
                not isinstance(rejected, Mapping)
                or rejected.get("status") != "rejected_before_actor_outcomes"
                or rejected.get("condition") != condition
                or isinstance(rejected.get("horizon_slot"), bool)
                or not isinstance(rejected.get("horizon_slot"), int)
                or rejected.get("horizon_slot") != slot
                or isinstance(rejected.get("requested_seed"), bool)
                or not isinstance(rejected.get("requested_seed"), int)
                or rejected.get("requested_seed") != rejected_seed
                or isinstance(rejected.get("pre_registered_horizon"), bool)
                or not isinstance(rejected.get("pre_registered_horizon"), int)
                or rejected.get("pre_registered_horizon") != horizon
                or not isinstance(rejected.get("reject_reason"), str)
                or not str(rejected["reject_reason"]).strip()
            ):
                raise FiveBodyContractError(
                    "ordered supplement reserve rejection history is incomplete"
                )
            slot_attempt_order.append(attempt_id)
            rejected_attempt_count += 1

        selected_attempt_id = supplement_reserve_attempt_id(
            condition, slot, selected_seed
        )
        selected_attempt = attempt_by_id.get(selected_attempt_id)
        if (
            not isinstance(selected_attempt, Mapping)
            or selected_attempt.get("status") != "complete"
            or selected_attempt.get("condition") != condition
            or isinstance(selected_attempt.get("horizon_slot"), bool)
            or not isinstance(selected_attempt.get("horizon_slot"), int)
            or selected_attempt.get("horizon_slot") != slot
            or isinstance(selected_attempt.get("requested_seed"), bool)
            or not isinstance(selected_attempt.get("requested_seed"), int)
            or selected_attempt.get("requested_seed") != selected_seed
            or isinstance(selected_attempt.get("pre_registered_horizon"), bool)
            or not isinstance(selected_attempt.get("pre_registered_horizon"), int)
            or selected_attempt.get("pre_registered_horizon") != horizon
            or selected_attempt.get("selected_before_actor_candidate_outcomes")
            is not True
            or not _is_sha(selected_attempt.get("root_triplet_bundle_sha256"))
        ):
            raise FiveBodyContractError(
                "selected supplement reserve attempt is incomplete"
            )
        slot_attempt_order.append(selected_attempt_id)
        actual_slot_order = [
            attempt_id
            for attempt_id in declared_attempt_order
            if attempt_by_id[attempt_id].get("condition") == condition
            and attempt_by_id[attempt_id].get("horizon_slot") == slot
        ]
        if actual_slot_order != slot_attempt_order:
            raise FiveBodyContractError(
                "supplement reserve attempts are out of order or continued after selection"
            )
        expected_attempt_order.extend(slot_attempt_order)
        selected_pairs.add((condition, selected_seed))

        for target in SUPPLEMENT_TARGET_EVENTS:
            expected_group_id = supplement_reserve_group_id(
                condition, slot, selected_seed, target
            )
            group = groups_by_id.get(expected_group_id)
            event_id = {"e12": 1, "e3": 2, "e4": 3}[target]
            if (
                not isinstance(group, Mapping)
                or group.get("condition") != condition
                or group.get("horizon_slot") != slot
                or group.get("requested_seed") != selected_seed
                or group.get("pre_registered_horizon") != horizon
                or group.get("scripted_root_event") != target
                or group.get("scripted_root_event_id") != event_id
                or group.get("root_event_id") != event_id
            ):
                raise FiveBodyContractError(
                    "selected supplement reserve group contract changed"
                )
            expected_group_ids.add(expected_group_id)

    if declared_attempt_order != expected_attempt_order:
        raise FiveBodyContractError(
            "supplement reserve attempts do not follow the ten-slot roster"
        )
    if set(groups_by_id) != expected_group_ids:
        raise FiveBodyContractError("supplement reserve group design is incomplete")
    return {
        "selected_pairs": selected_pairs,
        "rejected_attempt_count": rejected_attempt_count,
        "selected_seed_by_slot_sha256": canonical_sha256(dict(selected)),
        "reserve_roster_sha256": canonical_sha256(roster),
    }


def validate_supplement_body_manifest(
    value: Mapping[str, Any],
    *,
    expected_body: str,
    manifest_dir: Path,
    expected_actor_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate a complete reserve-roster manifest without opening any NPZ."""

    validate_body_manifest(
        value,
        expected_body=expected_body,
        manifest_dir=manifest_dir,
        expected_format=SUPPLEMENT_MANIFEST_FORMAT,
    )
    _verify_signed(value, f"{expected_body} supplement manifest")
    adapter = value.get("schema_adapter")
    if (
        value.get("format") != SUPPLEMENT_MANIFEST_FORMAT
        or value.get("collector_format") != SUPPLEMENT_COLLECTOR_FORMAT
        or value.get("dataset_repo") != DATASET_REPO
        or value.get("dataset_revision") != DATASET_REVISION
        or value.get("task") != TASK
        or value.get("body") != expected_body
        or value.get("conditions") != list(CONDITIONS)
        or value.get("target_events") != list(SUPPLEMENT_TARGET_EVENTS)
        or value.get("instruction") != DEFAULT_INSTRUCTION
        or value.get("candidate_count") != CANDIDATE_COUNT
        or value.get("action_exec_steps") != 5
        or value.get("supplement_role")
        != "expert_event_root_outer_source_crossfit_proper_world_and_utility_rank"
        or value.get("root_policy") != "robotwin_scripted_expert"
        or value.get("candidate_and_continuation_policy")
        != "same_frozen_native_actor_as_primary_binding"
        or value.get("proper_loss_weight") != SUPPLEMENT_PROPER_LOSS_WEIGHT
        or value.get("rank_loss_weight") != SUPPLEMENT_RANK_LOSS_WEIGHT
        or value.get("usage_contract") != SUPPLEMENT_USAGE_CONTRACT
        or value.get("expert_root_provenance_contract")
        != EXPERT_ROOT_PROVENANCE_CONTRACT
        or not _is_sha(value.get("collector_file_sha256"))
        or not _is_sha(value.get("base_collector_file_sha256"))
        or value.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or not _is_sha(value.get("event_derivation_implementation_sha256"))
        or value.get("actor_checkpoint_tree_or_file_sha256")
        != expected_actor_checkpoint_sha256
        or not _is_sha(value.get("actor_authority_sha256"))
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
        or not isinstance(adapter, Mapping)
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
        or value.get("root_selection_contract")
        != SUPPLEMENT_ROOT_SELECTION_CONTRACT
        or value.get("horizon_contract") != SUPPLEMENT_HORIZON_CONTRACT
        or value.get("actor_branch_contract")
        != SUPPLEMENT_ACTOR_BRANCH_CONTRACT
    ):
        raise FiveBodyContractError(
            f"{expected_body} raw scripted-root supplement contract changed"
        )

    design = _validate_supplement_reserve_design(value, body=expected_body)
    identities: set[str] = set()
    conditions: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for declared in value["groups"]:
        group_id = str(declared["group_id"])
        condition = declared.get("condition")
        slot = declared.get("horizon_slot")
        seed = declared.get("requested_seed")
        root_event = declared.get("scripted_root_event_id")
        event_name = {1: "e12", 2: "e3", 3: "e4"}.get(root_event)
        branch_horizon = declared.get("pre_registered_horizon")
        if (
            group_id in identities
            or condition not in CONDITIONS
            or isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot not in range(len(SUPPLEMENT_HORIZON_SCHEDULE))
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(root_event, bool)
            or not isinstance(root_event, int)
            or event_name is None
            or isinstance(branch_horizon, bool)
            or not isinstance(branch_horizon, int)
            or branch_horizon != SUPPLEMENT_HORIZON_SCHEDULE[slot]
            or declared.get("candidate_noise_query_index")
            != {1: 1, 2: 2, 3: 3}[root_event]
            or group_id
            != supplement_reserve_group_id(
                str(condition), slot, seed, event_name
            )
            or not _is_sha(declared.get("sha256"))
            or not _is_sha(declared.get("raw_expert_snapshot_sha256"))
            or not _is_sha(declared.get("branch_root_snapshot_sha256"))
            or not _is_sha(
                declared.get("branch_root_restorable_snapshot_sha256")
            )
            or not _is_sha(declared.get("canonical_root_snapshot_sha256"))
            or declared.get("diagnostic_format")
            != BRANCH_DIAGNOSTIC_CONTRACT["format"]
            or not _is_sha(declared.get("diagnostics_sha256"))
        ):
            raise FiveBodyContractError(
                f"{expected_body} supplement group is invalid"
            )
        # Lexical containment only: held-out validation must not stat, hash, or
        # deserialize either the group payload or its diagnostics.
        payload_path = _lexical_contained_payload_path(
            manifest_dir, str(declared.get("path", "")), "supplement group"
        )
        diagnostics_path = _lexical_contained_payload_path(
            manifest_dir,
            str(declared.get("diagnostics_path", "")),
            "supplement group diagnostics",
        )
        identities.add(group_id)
        conditions.add(str(condition))
        normalized.append(
            {
                **dict(declared),
                "root_event_id": int(root_event),
                "source_role": "proper_world_supplement",
                "resolved_path": str(payload_path),
                "resolved_diagnostics_path": str(diagnostics_path),
            }
        )
    if conditions != set(CONDITIONS):
        raise FiveBodyContractError(
            f"{expected_body} supplement lacks clean/randomized roots"
        )
    return {
        "body": expected_body,
        "groups": normalized,
        "group_identity_sha256": canonical_sha256(sorted(identities)),
        "event_derivation_implementation_sha256": value[
            "event_derivation_implementation_sha256"
        ],
        "raw_manifest_format": SUPPLEMENT_MANIFEST_FORMAT,
        "usage_contract": dict(SUPPLEMENT_USAGE_CONTRACT),
        "expert_root_provenance_contract": dict(
            EXPERT_ROOT_PROVENANCE_CONTRACT
        ),
        "proper_loss_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
        "rank_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
        **design,
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
        or binding.get("state_action_frame_contract")
        != STATE_ACTION_FRAME_CONTRACT
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


def load_supplement_binding(
    path: Path,
    expected_sha256: str,
    *,
    primary_audit: Mapping[str, Any],
    held_out_body: str,
) -> dict[str, Any]:
    """Load an independently bound supplement without opening group payloads."""

    if held_out_body not in BODIES:
        raise FiveBodyContractError(f"unknown held-out body {held_out_body!r}")
    binding = _read_bound_json(path, expected_sha256, "supplement binding")
    _verify_signed(binding, "supplement binding")
    primary_binding = primary_audit.get("binding")
    if not isinstance(primary_binding, Mapping):
        raise FiveBodyContractError("primary audit is missing its binding")
    actor_binding = primary_binding.get("actor_authority")
    actor_authority_sha256 = (
        actor_binding.get("sha256") if isinstance(actor_binding, Mapping) else None
    )
    materializer = binding.get("materializer_provenance")
    rejected_attempt_count = (
        materializer.get("rejected_attempt_count")
        if isinstance(materializer, Mapping)
        else None
    )
    if (
        binding.get("format") != SUPPLEMENT_BINDING_FORMAT
        or binding.get("dataset_repo") != DATASET_REPO
        or binding.get("dataset_revision") != DATASET_REVISION
        or binding.get("task") != TASK
        or binding.get("instruction") != DEFAULT_INSTRUCTION
        or binding.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or binding.get("state_action_frame_contract")
        != STATE_ACTION_FRAME_CONTRACT
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
        or binding.get("primary_binding_file_sha256")
        != primary_audit.get("binding_file_sha256")
        or not _is_sha(actor_authority_sha256)
        or binding.get("actor_authority_sha256") != actor_authority_sha256
        or binding.get("proper_loss_weight") != SUPPLEMENT_PROPER_LOSS_WEIGHT
        or binding.get("rank_loss_weight") != SUPPLEMENT_RANK_LOSS_WEIGHT
        or binding.get("usage_contract") != SUPPLEMENT_USAGE_CONTRACT
        or binding.get("expert_root_provenance_contract")
        != EXPERT_ROOT_PROVENANCE_CONTRACT
        or not isinstance(materializer, Mapping)
        or materializer.get("format") != SUPPLEMENT_MATERIALIZER_FORMAT
        or materializer.get("payload_npz_files_opened") != 0
        or materializer.get("complete_decisions")
        != len(BODIES) * SUPPLEMENT_EXPECTED_DECISIONS_PER_BODY
        or materializer.get("complete_branches")
        != len(BODIES) * SUPPLEMENT_EXPECTED_DECISIONS_PER_BODY * CANDIDATE_COUNT
        or materializer.get("seed_overlap_with_primary") != 0
        or materializer.get("selected_seed_count") != 50
        or isinstance(rejected_attempt_count, bool)
        or not isinstance(rejected_attempt_count, int)
        or rejected_attempt_count < 0
        or rejected_attempt_count
        > (
            len(BODIES)
            * len(CONDITIONS)
            * len(SUPPLEMENT_HORIZON_SCHEDULE)
            * (SUPPLEMENT_RESERVE_SEEDS_PER_SLOT - 1)
        )
        or materializer.get(
            "selection_occurs_before_actor_candidate_outcomes"
        )
        is not True
        or materializer.get("heldout_payload_npz_files_opened") != 0
    ):
        raise FiveBodyContractError(
            "supplement binding violates the outer-source train/validation contract"
        )
    body_bindings = binding.get("body_manifests")
    if not isinstance(body_bindings, Mapping) or set(body_bindings) != set(BODIES):
        raise FiveBodyContractError(
            "supplement binding must contain exactly five body manifests"
        )
    root = path.expanduser().resolve().parent
    manifests: dict[str, dict[str, Any]] = {}
    heldout_manifest_binding: dict[str, Any] | None = None
    source_rejected_attempt_count = 0
    for body in BODIES:
        item = body_bindings[body]
        if (
            not isinstance(item, Mapping)
            or not _is_sha(item.get("sha256"))
            or not _is_sha(item.get("selected_seed_by_slot_sha256"))
            or not _is_sha(item.get("reserve_roster_sha256"))
            or isinstance(item.get("group_count"), bool)
            or not isinstance(item.get("group_count"), int)
            or int(item["group_count"]) != SUPPLEMENT_EXPECTED_DECISIONS_PER_BODY
        ):
            raise FiveBodyContractError(
                f"supplement body manifest binding missing for {body}"
            )
        if body == held_out_body:
            opaque_path = _lexical_contained_payload_path(
                root,
                str(item.get("path", "")),
                "held-out supplement manifest",
            )
            heldout_manifest_binding = {
                "body": body,
                "opaque_path": str(opaque_path),
                "sha256": str(item["sha256"]),
                "declared_group_count": int(item["group_count"]),
                "selected_seed_by_slot_sha256": str(
                    item["selected_seed_by_slot_sha256"]
                ),
                "reserve_roster_sha256": str(item["reserve_roster_sha256"]),
                "manifest_file_opened": 0,
                "manifest_bytes_read": 0,
                "payload_files_opened": 0,
                "payload_bytes_read": 0,
            }
            continue
        manifest_path = _resolve_contained(
            root, str(item.get("path", "")), "supplement body manifest"
        )
        manifest = _read_bound_json(
            manifest_path,
            str(item.get("sha256", "")),
            f"{body} supplement body manifest",
        )
        if manifest.get("actor_authority_sha256") != actor_authority_sha256:
            raise FiveBodyContractError(
                f"{body} supplement did not use the bound actor authority"
            )
        manifests[body] = validate_supplement_body_manifest(
            manifest,
            expected_body=body,
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=str(
                primary_audit["actor"]["checkpoint_sha256_by_body"][body]
            ),
        )
        if len(manifests[body]["groups"]) != int(item["group_count"]):
            raise FiveBodyContractError(
                f"{body} supplement manifest group count differs from binding"
            )
        if (
            canonical_sha256(manifest.get("selected_seed_by_slot"))
            != item["selected_seed_by_slot_sha256"]
            or canonical_sha256(manifest.get("reserve_roster"))
            != item["reserve_roster_sha256"]
        ):
            raise FiveBodyContractError(
                f"{body} supplement reserve provenance differs from binding"
            )
        source_rejected_attempt_count += sum(
            1
            for attempt in manifest.get("attempts", [])
            if isinstance(attempt, Mapping)
            and attempt.get("status") == "rejected_before_actor_outcomes"
        )
        primary_manifest = primary_audit.get("manifests", {}).get(body)
        if not isinstance(primary_manifest, Mapping):
            raise FiveBodyContractError(
                f"primary audit is missing {body} reset identities"
            )
        primary_resets = {
            (str(group["condition"]), int(group["requested_seed"]))
            for group in primary_manifest["groups"]
        }
        supplement_resets = {
            (str(group["condition"]), int(group["requested_seed"]))
            for group in manifests[body]["groups"]
        }
        if primary_resets & supplement_resets:
            raise FiveBodyContractError(
                f"{body} supplement overlaps formal condition/seed resets"
            )
    if heldout_manifest_binding is None:
        raise FiveBodyContractError("supplement held-out binding was not deferred")
    heldout_rejection_capacity = (
        len(CONDITIONS)
        * len(SUPPLEMENT_HORIZON_SCHEDULE)
        * (SUPPLEMENT_RESERVE_SEEDS_PER_SLOT - 1)
    )
    if not (
        source_rejected_attempt_count
        <= rejected_attempt_count
        <= source_rejected_attempt_count + heldout_rejection_capacity
    ):
        raise FiveBodyContractError(
            "supplement rejected-attempt provenance cannot match the deferred body"
        )
    implementations = {
        value["event_derivation_implementation_sha256"]
        for value in manifests.values()
    }
    if implementations != {
        primary_audit.get("event_derivation_implementation_sha256")
    }:
        raise FiveBodyContractError(
            "supplement and primary event implementations differ"
        )
    return {
        "binding": binding,
        "binding_file_sha256": expected_sha256.lower(),
        "manifests": manifests,
        "held_out_body": held_out_body,
        "heldout_manifest_binding": heldout_manifest_binding,
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": implementations.pop(),
        "proper_loss_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
        "rank_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
        "usage_contract": dict(SUPPLEMENT_USAGE_CONTRACT),
        "expert_root_provenance_contract": dict(
            EXPERT_ROOT_PROVENANCE_CONTRACT
        ),
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


def supplement_inner_validation_body(
    *, held_out_body: str, split_seed: int
) -> str:
    """Choose one outer-source body without inspecting supplement labels."""

    if held_out_body not in BODIES:
        raise FiveBodyContractError(f"unknown held-out body {held_out_body!r}")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise FiveBodyContractError("supplement split seed must be an integer")
    ordered = sorted(
        BODIES,
        key=lambda body: hashlib.sha256(
            f"{split_seed}|supplement-crossfit-order|{body}".encode()
        ).hexdigest(),
    )
    return ordered[(ordered.index(held_out_body) + 1) % len(ordered)]


def supplement_source_train_split(
    supplement_audit: Mapping[str, Any], *, held_out_body: str, split_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split four source bodies into three train and one proper-validation lane."""

    if held_out_body not in BODIES:
        raise FiveBodyContractError(f"unknown held-out body {held_out_body!r}")
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    manifests = supplement_audit.get("manifests")
    expected_sources = set(BODIES) - {held_out_body}
    if (
        supplement_audit.get("held_out_body") != held_out_body
        or not isinstance(manifests, Mapping)
        or set(manifests) != expected_sources
    ):
        raise FiveBodyContractError(
            "supplement audit does not expose exactly four source manifests"
        )
    validation_body = supplement_inner_validation_body(
        held_out_body=held_out_body, split_seed=split_seed
    )
    for body in BODIES:
        if body == held_out_body:
            continue
        groups = list(manifests[body]["groups"])
        if len(groups) != SUPPLEMENT_EXPECTED_DECISIONS_PER_BODY:
            raise FiveBodyContractError(
                f"{body} supplement does not contain the complete 30-group design"
            )
        target = validation if body == validation_body else training
        target.extend({**group, "body": body} for group in groups)
    heldout = supplement_audit.get("heldout_manifest_binding")
    if (
        not training
        or not validation
        or not isinstance(heldout, Mapping)
        or heldout.get("body") != held_out_body
        or int(heldout.get("declared_group_count", 0)) <= 0
    ):
        raise FiveBodyContractError("supplement LOBO split contains an empty lane")
    if (
        {str(row["body"]) for row in validation} != {validation_body}
        or {str(row["body"]) for row in training}
        != expected_sources - {validation_body}
    ):
        raise FiveBodyContractError("supplement inner cross-body split changed")
    return training, validation, dict(heldout)


def build_preflight_receipt(
    audit: Mapping[str, Any],
    *,
    held_out_body: str,
    split_seed: int,
    supplement_audit: Mapping[str, Any] | None = None,
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
    if supplement_audit is None:
        supplement_training: list[dict[str, Any]] = []
        supplement_validation: list[dict[str, Any]] = []
        supplement_validation_body: str | None = None
        supplement_heldout: dict[str, Any] = {
            "declared_group_count": 0,
            "sha256": None,
        }
    else:
        (
            supplement_training,
            supplement_validation,
            supplement_heldout,
        ) = supplement_source_train_split(
            supplement_audit,
            held_out_body=held_out_body,
            split_seed=split_seed,
        )
        supplement_validation_body = supplement_inner_validation_body(
            held_out_body=held_out_body, split_seed=split_seed
        )
    receipt = {
        "format": FORMAT,
        "status": "preflight_passed_payloads_still_unopened",
        "dataset_revision": DATASET_REVISION,
        "state_action_frame_contract": state_action_frame_contract(),
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
        "supplement": {
            "enabled": supplement_audit is not None,
            "binding_file_sha256": (
                supplement_audit.get("binding_file_sha256")
                if supplement_audit is not None
                else None
            ),
            "proper_loss_weight": (
                SUPPLEMENT_PROPER_LOSS_WEIGHT
                if supplement_audit is not None
                else 0.0
            ),
            "rank_loss_weight": (
                SUPPLEMENT_RANK_LOSS_WEIGHT
                if supplement_audit is not None
                else 0.0
            ),
            "source_train_groups": len(supplement_training),
            "source_validation_groups": len(supplement_validation),
            "source_validation_body": supplement_validation_body,
            "source_validation_body_selection": (
                "label_blind_sha256_ordered_five_body_cycle_successor_"
                "derangement"
            ),
            "source_validation_assignment_uses_labels": False,
            "heldout_groups_deferred": int(
                supplement_heldout["declared_group_count"]
            ),
            "source_train_identity_sha256": canonical_sha256(
                identity(supplement_training)
            ),
            "source_validation_identity_sha256": canonical_sha256(
                identity(supplement_validation)
            ),
            "heldout_manifest_sha256": supplement_heldout.get("sha256"),
            "usage_contract": dict(SUPPLEMENT_USAGE_CONTRACT),
            "normalization_rows_used": 0,
            "baseline_fit_rows_used": 0,
            "rank_or_utility_rows_authorized": (
                len(supplement_training) * CANDIDATE_COUNT
            ),
            "source_validation_rows_used": 0,
            "checkpoint_selection_rows_used": 0,
            "proper_checkpoint_selection_rows_authorized": (
                len(supplement_validation) * CANDIDATE_COUNT
            ),
            "proper_checkpoint_selection_weight": (
                SUPPLEMENT_PROPER_LOSS_WEIGHT
                if supplement_audit is not None
                else 0.0
            ),
            "proper_validation_primary_reset_overlap": 0,
            "proper_validation_standard_error_combination": (
                "independent_primary_and_supplement_seed_variances"
            ),
            "rank_selection_rows_authorized": 0,
            "calibration_diagnostic_rows_authorized": (
                len(supplement_validation) * CANDIDATE_COUNT
            ),
            "calibration_rows_used": 0,
            "calibration_fit": False,
            "heldout_group_npz_opened": 0,
            "heldout_group_payload_bytes_read": 0,
            "heldout_group_payload_deserialized": 0,
            "heldout_manifest_file_opened": 0,
            "heldout_manifest_bytes_read": 0,
        },
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
    if (
        not np.array_equal(
            arrays["next_event_id"], arrays["next_event_id"].astype(np.int64)
        )
        or np.any((arrays["next_event_id"] < 0) | (arrays["next_event_id"] >= 5))
        or not np.all(np.isin(arrays["next_event_mask"], [0.0, 1.0]))
        or not np.all(np.isin(arrays["duration_observed"], [0.0, 1.0]))
        or np.any(
            (arrays["duration_observed"] > 0.5)
            & ~(arrays["next_event_mask"] > 0.5)
        )
    ):
        raise FiveBodyContractError(
            f"{path} next-event/duration supervision is invalid"
        )
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
    height_reference = np.asarray(
        arrays["success_height_reference_z"], dtype=np.float64
    )
    if not np.isfinite(height_reference).all() or not np.allclose(
        height_reference,
        height_reference[:1],
        atol=0.0,
        rtol=0.0,
    ):
        raise FiveBodyContractError(
            f"{path} candidates do not share one finite task.orig_z authority"
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
                "requested_seed": np.int64(group["requested_seed"]),
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
    """Build rank-only batches without requiring a success-changing decision.

    Proper likelihoods use :class:`CompleteDecisionBatchSampler` and therefore
    retain the empirical source distribution.  This sampler is used only by
    the candidate-rank objective.  It alternates body/condition/current-event
    strata, reserves half of each batch for mixed-success decisions when they
    exist, and otherwise trains from genuinely orderable all-failure decisions.
    It never repeats a logical decision within a batch.  Sparse mixed decisions
    may be revisited across batches; that is the intended oversampling needed
    to prevent dense failures from drowning success-changing supervision.
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
        if not self.mixed_groups and not self.dense_groups:
            raise FiveBodyContractError(
                "rank sampler requires mixed-success or informative dense supervision"
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
        mixed = (
            self._stratified_cycler(self.mixed_groups, self.strata, generator)
            if self.mixed_groups
            else None
        )
        dense = (
            self._stratified_cycler(self.dense_groups, self.strata, generator)
            if self.dense_groups
            else None
        )
        mixed_target = (
            max(1, self.decisions_per_batch // 2) if mixed is not None else 0
        )
        for _batch in range(self.batch_count):
            used: set[str] = set()
            selected = (
                self._draw_distinct(
                    mixed,
                    requested=mixed_target,
                    available_count=len(self.mixed_groups),
                    used=used,
                )
                if mixed is not None
                else []
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
            if mixed is not None:
                selected.extend(
                    self._draw_distinct(
                        mixed,
                        requested=self.decisions_per_batch - len(selected),
                        available_count=len(self.mixed_groups),
                        used=used,
                    )
                )
            if not selected or (
                self.mixed_groups
                and not any(self.kinds[group] == "mixed" for group in selected)
            ):
                raise FiveBodyContractError(
                    "rank batch lost all comparative supervision"
                )
            generator.shuffle(selected)
            yield [index for group in selected for index in self.decisions[group]]

    def __len__(self) -> int:
        return self.batch_count


class InvariantMonotoneConsequenceUtility(torch.nn.Module):
    """Signed utility over bounded, cross-embodiment consequences.

    Terminal event progress is the primary value.  Benefits have non-negative
    softmax weights and uncertainties have non-positive bounded coefficients in
    a small tie-break term.  Canonical-current-event residuals are shared across
    bodies.  There is no cross-feature LayerNorm, raw world-frame object axis,
    or unconstrained sign-flipping MLP.
    """

    def __init__(self) -> None:
        super().__init__()
        benefit_initial = torch.zeros(len(MONOTONE_BENEFIT_FEATURES))
        benefit_initial[MONOTONE_BENEFIT_FEATURES.index("success_probability")] = 2.0
        benefit_initial[
            MONOTONE_BENEFIT_FEATURES.index(
                "terminal_expected_stage_progress"
            )
        ] = 1.0
        self.benefit_logits = torch.nn.Parameter(benefit_initial)
        self.risk_logits = torch.nn.Parameter(
            torch.full((len(MONOTONE_RISK_FEATURES),), -2.0)
        )
        self.event_benefit_residual = torch.nn.Parameter(
            torch.zeros(len(core.CANONICAL_EVENTS), len(MONOTONE_BENEFIT_FEATURES))
        )
        self.event_risk_residual = torch.nn.Parameter(
            torch.zeros(len(core.CANONICAL_EVENTS), len(MONOTONE_RISK_FEATURES))
        )

    @staticmethod
    def _indices(names: Sequence[str]) -> list[int]:
        indices = []
        for name in names:
            start, stop = CANDIDATE_RANK_FEATURE_SCHEMA[name]
            if stop - start != 1:
                raise FiveBodyContractError(
                    f"monotone utility feature {name} is not scalar"
                )
            indices.append(start)
        return indices

    def forward(
        self, features: torch.Tensor, current_event_id: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != CANDIDATE_RANK_FEATURE_DIM:
            raise FiveBodyContractError("monotone utility feature shape changed")
        if (
            not isinstance(current_event_id, torch.Tensor)
            or current_event_id.shape != features.shape[:1]
            or current_event_id.dtype
            not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            or bool(
                (
                    (current_event_id < 0)
                    | (current_event_id >= len(core.CANONICAL_EVENTS))
                ).any()
            )
        ):
            raise FiveBodyContractError(
                "monotone utility requires one canonical current event per row"
            )
        if not bool(torch.isfinite(features).all()) or bool(
            ((features < 0.0) | (features > 1.0)).any()
        ):
            raise FiveBodyContractError(
                "monotone utility requires finite consequences in [0,1]"
            )
        benefit = features[:, self._indices(MONOTONE_BENEFIT_FEATURES)]
        risk = features[:, self._indices(MONOTONE_RISK_FEATURES)]
        event = current_event_id.long()
        benefit_residual = EVENT_UTILITY_RESIDUAL_BOUND * torch.tanh(
            self.event_benefit_residual[event]
        )
        risk_residual = EVENT_UTILITY_RESIDUAL_BOUND * torch.tanh(
            self.event_risk_residual[event]
        )
        benefit_weights = torch.softmax(
            self.benefit_logits[None] + benefit_residual, dim=-1
        )
        risk_weights = torch.sigmoid(self.risk_logits[None] + risk_residual)
        secondary = (benefit * benefit_weights).sum(dim=-1) - (
            risk * risk_weights
        ).sum(dim=-1)
        terminal_stage_index = CANDIDATE_RANK_FEATURE_SCHEMA[
            "terminal_expected_stage_progress"
        ][0]
        # ``secondary`` is strictly bounded to (-2, 1): benefits are a convex
        # combination of [0, 1] values and the two risks each have coefficients
        # in (0, 1).  At scale 0.05 its entire possible reversal is < 0.15, so a
        # deterministic canonical one-event step (0.25) remains absolute while
        # the learned consequences can still break event ties.
        return (
            features[:, terminal_stage_index]
            + EVENT_PRIORITY_SECONDARY_SCALE * secondary
        )


def _event_conditioned_student_t3_mixture_moments(
    event_probability: torch.Tensor,
    component_mean: torch.Tensor,
    component_log_scale: torch.Tensor,
    *,
    dof: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact mean/std and a moment-equivalent Student-t log scale.

    ``component_mean`` may be scalar ``[B, E]`` or vector ``[B, E, D]``.
    The returned standard deviation applies the law of total variance, so it
    contains both Student-t variance within an observed-event mode and variance
    between event modes.  The compatibility log scale is defined as
    ``log(total_std / sqrt(dof / (dof - 2)))``; for dof=3, legacy consumers that
    multiply ``exp(log_scale)`` by ``sqrt(3)`` recover the exact mixture std.
    """

    if (
        event_probability.ndim != 2
        or component_mean.shape != component_log_scale.shape
        or component_mean.ndim not in {2, 3}
        or component_mean.shape[:2] != event_probability.shape
        or not math.isfinite(float(dof))
        or float(dof) <= 2.0
    ):
        raise FiveBodyContractError("event-conditioned Student-t mixture is invalid")
    weight = event_probability
    while weight.ndim < component_mean.ndim:
        weight = weight.unsqueeze(-1)
    component_variance = (float(dof) / (float(dof) - 2.0)) * torch.exp(
        2.0 * component_log_scale
    )
    mixture_mean = (weight * component_mean).sum(dim=1)
    minimum_variance = (float(dof) / (float(dof) - 2.0)) * math.exp(
        2.0 * CONSEQUENCE_LOG_SCALE_MIN
    )
    # Use the centered law of total variance directly.  The algebraically
    # equivalent E[var + mean^2] - E[mean]^2 form loses all between-event
    # variance in float32 when component locations share a large offset.
    centered_component_mean = component_mean - mixture_mean.unsqueeze(1)
    mixture_variance = (
        weight * (component_variance + centered_component_mean.square())
    ).sum(dim=1).clamp_min(minimum_variance)
    mixture_std = torch.sqrt(mixture_variance)
    moment_equivalent_log_scale = torch.log(
        mixture_std
        / math.sqrt(float(dof) / (float(dof) - 2.0))
    )
    return mixture_mean, mixture_std, moment_equivalent_log_scale


def _next_event_duration_mixture_moments(
    next_event_probability: torch.Tensor,
    component_log_mean: torch.Tensor,
    component_log_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moment-match ``log1p(D)`` under the next-event mixture.

    Duration is not a property of the current event alone: the time to an e3
    boundary and the time to an e4 boundary can be different modes from the
    same root.  The five clock outputs therefore parameterize
    ``p(log1p(D) | next_event=e)``.  This helper returns the exact first two
    moments of that Gaussian mixture and a scalar log scale for legacy callers
    that accept one moment-equivalent lognormal per candidate.

    Keeping the moment calculation in log-time avoids the overflow and severe
    tail domination caused by taking raw shifted-lognormal moments during
    early training.  The proper likelihood below is still evaluated against
    the individual mixture components without moment matching.
    """

    if (
        next_event_probability.ndim != 2
        or component_log_mean.shape != next_event_probability.shape
        or component_log_scale.shape != next_event_probability.shape
        or not bool(torch.isfinite(next_event_probability).all())
        or not bool(torch.isfinite(component_log_mean).all())
        or not bool(torch.isfinite(component_log_scale).all())
        or bool((next_event_probability < 0.0).any())
        or not bool(
            torch.allclose(
                next_event_probability.sum(dim=-1),
                torch.ones_like(next_event_probability[:, 0]),
                atol=1e-5,
                rtol=1e-5,
            )
        )
    ):
        raise FiveBodyContractError("next-event duration mixture is invalid")
    mixture_mean = (next_event_probability * component_log_mean).sum(dim=-1)
    component_variance = torch.exp(2.0 * component_log_scale)
    centered = component_log_mean - mixture_mean[:, None]
    mixture_variance = (
        next_event_probability * (component_variance + centered.square())
    ).sum(dim=-1).clamp_min(math.exp(-10.0))
    mixture_std = torch.sqrt(mixture_variance)
    return mixture_mean, mixture_std, torch.log(mixture_std)


def _competing_risks_duration_nll_rows(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Proper duration rows for observed and right-censored event boundaries.

    Observed rows use the component belonging to the observed next event; the
    separate next-event CE supplies ``-log p(e_next)``.  On censored rows the
    destination is unknown, so the likelihood is the exact competing-risks
    survival ``sum_e p(e) S(D_censor | e)``.  Consequently censored examples
    supervise all plausible destinations instead of being mislabelled as a
    current-event duration.
    """

    logits = output["next_event_logits"]
    mean = output["duration_component_log_mean"]
    log_scale = output["duration_component_log_scale"]
    duration = batch["duration"].to(mean)
    raw_observed = batch["duration_observed"]
    next_event = batch["next_event_id"].long()
    raw_next_mask = batch["next_event_mask"]
    if (
        logits.ndim != 2
        or mean.shape != logits.shape
        or log_scale.shape != logits.shape
        or duration.shape != logits.shape[:1]
        or raw_observed.shape != duration.shape
        or next_event.shape != duration.shape
        or raw_next_mask.shape != duration.shape
        or bool(((next_event < 0) | (next_event >= logits.shape[-1])).any())
        or not bool(((raw_observed == 0) | (raw_observed == 1)).all())
        or not bool(((raw_next_mask == 0) | (raw_next_mask == 1)).all())
        or not bool(torch.isfinite(logits).all())
        or not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(log_scale).all())
        or not bool(torch.isfinite(duration).all())
        or bool((duration < 0.0).any())
    ):
        raise FiveBodyContractError(
            "competing-risks duration supervision is invalid"
        )
    observed = raw_observed.bool()
    next_mask = raw_next_mask.bool()
    if bool((observed & ~next_mask).any()):
        raise FiveBodyContractError(
            "observed duration requires an observed next-event label"
        )
    target = torch.log1p(duration)
    scale = torch.exp(log_scale).clamp_min(1e-4)
    z = (target[:, None] - mean) / scale
    component_observed_nll = (
        0.5 * z.square() + log_scale + 0.5 * math.log(2.0 * math.pi)
    )
    observed_nll = component_observed_nll.gather(
        1, next_event[:, None]
    ).squeeze(1)
    component_log_survival = torch.special.log_ndtr(-z)
    censored_nll = -torch.logsumexp(
        torch.log_softmax(logits, dim=-1) + component_log_survival,
        dim=-1,
    )
    return torch.where(observed, observed_nll, censored_nll)


def _gather_observed_event_component(
    component: torch.Tensor,
    observed_event_id: torch.Tensor,
) -> torch.Tensor:
    """Gather one categorical-event conditional component per batch row."""

    if (
        component.ndim not in {2, 3}
        or observed_event_id.shape != (component.shape[0],)
        or observed_event_id.dtype
        not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        or bool(
            (
                (observed_event_id < 0)
                | (observed_event_id >= component.shape[1])
            ).any()
        )
    ):
        raise FiveBodyContractError("observed event component gather is invalid")
    index = observed_event_id.long()
    if component.ndim == 2:
        return component.gather(1, index[:, None]).squeeze(1)
    return component.gather(
        1, index[:, None, None].expand(-1, 1, component.shape[-1])
    ).squeeze(1)


class EffectAlignedSharedEventHead(core.MultibodyCanonicalEventWorldModel):
    """Shared event model with a scalar head trained for best-of-four choice."""

    def __init__(self, ablation_variant: str = "full") -> None:
        if ablation_variant not in ABLATION_VARIANTS:
            raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
        super().__init__(core.ModelConfig(body_count=1, action_schema_count=1))
        self.ablation_variant = ablation_variant
        self.action.normalization_clip = CROSS_BODY_STANDARDIZED_INPUT_CLIP
        # The base class exposes independent horizon-free outcome/effect heads.
        # Branch success and recovery are finite-horizon labels here, while the
        # object likelihood is categorical-post-event conditional in v11.  The
        # legacy heads remain checkpoint-visible for base compatibility but are
        # frozen and excluded from every prediction and loss below.
        self.success.requires_grad_(False)
        self.recovery.requires_grad_(False)
        self.object_mean.requires_grad_(False)
        self.object_scale.requires_grad_(False)
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
            torch.nn.Linear(core.SEMANTIC_DIM, 2 * core.SEMANTIC_DIM),
        )
        self.terminal_residual = torch.nn.Sequential(
            torch.nn.LayerNorm(core.SEMANTIC_DIM),
            torch.nn.Linear(core.SEMANTIC_DIM, core.SEMANTIC_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(core.SEMANTIC_DIM, core.SEMANTIC_DIM),
        )
        # Start from the former horizon-free representation, then let proper
        # finite-horizon supervision learn bounded FiLM and residual effects.
        # This keeps initialization stable without removing the action-horizon
        # interaction from the function class.
        torch.nn.init.zeros_(self.terminal_context_encoder[-1].weight)
        torch.nn.init.zeros_(self.terminal_context_encoder[-1].bias)
        torch.nn.init.zeros_(self.terminal_residual[-1].weight)
        torch.nn.init.zeros_(self.terminal_residual[-1].bias)
        self.terminal_event = torch.nn.Linear(core.SEMANTIC_DIM, 5)
        self.terminal_recovery = torch.nn.Linear(core.SEMANTIC_DIM, 1)
        self.object_delta_component_mean = torch.nn.Linear(
            core.SEMANTIC_DIM,
            len(core.CANONICAL_EVENTS) * core.OBJECT_DELTA_DIM,
        )
        self.object_delta_component_scale = torch.nn.Linear(
            core.SEMANTIC_DIM,
            len(core.CANONICAL_EVENTS) * core.OBJECT_DELTA_DIM,
        )
        self.terminal_goal_progress_component_mean = torch.nn.Linear(
            core.SEMANTIC_DIM, len(core.CANONICAL_EVENTS)
        )
        self.terminal_goal_progress_component_scale = torch.nn.Linear(
            core.SEMANTIC_DIM, len(core.CANONICAL_EVENTS)
        )
        self.candidate_rank = InvariantMonotoneConsequenceUtility()

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
        normalized_batch["state"] = normalized_batch["state"].clamp(
            min=-CROSS_BODY_STANDARDIZED_INPUT_CLIP,
            max=CROSS_BODY_STANDARDIZED_INPUT_CLIP,
        )
        output = super().forward(normalized_batch)
        post_probability = torch.softmax(output["post_event_logits"], dim=-1)
        object_component_mean = self.object_delta_component_mean(
            output["transitioned"]
        ).reshape(
            -1, len(core.CANONICAL_EVENTS), core.OBJECT_DELTA_DIM
        )
        object_component_log_scale = torch.clamp(
            self.object_delta_component_scale(output["transitioned"]).reshape(
                -1, len(core.CANONICAL_EVENTS), core.OBJECT_DELTA_DIM
            ),
            CONSEQUENCE_LOG_SCALE_MIN,
            2.0,
        )
        (
            object_mixture_mean,
            object_mixture_std,
            object_moment_equivalent_log_scale,
        ) = _event_conditioned_student_t3_mixture_moments(
            post_probability,
            object_component_mean,
            object_component_log_scale,
            dof=OBJECT_STUDENT_T_DOF,
        )
        output["object_delta_component_mean"] = object_component_mean
        output["object_delta_component_log_scale"] = (
            object_component_log_scale
        )
        output["object_delta_mixture_probability"] = post_probability
        # Backward-compatible fields now describe the full post-event mixture.
        # ``exp(log_scale)*sqrt(3) == std`` remains exact for legacy consumers.
        output["object_delta_mean"] = object_mixture_mean
        output["object_delta_std"] = object_mixture_std
        output["object_delta_log_scale"] = object_moment_equivalent_log_scale

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
        next_event_probability = torch.softmax(
            output["next_event_logits"], dim=-1
        )
        (
            duration_mixture_log_mean,
            duration_mixture_log_std,
            duration_moment_equivalent_log_scale,
        ) = _next_event_duration_mixture_moments(
            next_event_probability,
            duration_log_mean,
            duration_log_scale,
        )
        current = batch["current_event_id"].long()[:, None]
        output["clock_hidden"] = age_clock_hidden
        # Five components now mean p(log1p(D) | next_event=e), not five
        # current-event slots.  Preserve the legacy tensor names/shapes while
        # publishing explicit semantics for new consumers.
        output["duration_log_mean"] = duration_log_mean
        output["duration_log_scale"] = duration_log_scale
        output["duration_component_log_mean"] = duration_log_mean
        output["duration_component_log_scale"] = duration_log_scale
        output["duration_mixture_probability"] = next_event_probability
        output["duration_log1p_mixture_mean"] = duration_mixture_log_mean
        output["duration_log1p_mixture_std"] = duration_mixture_log_std
        # Backward-compatible scalar interface: these are the first-two-moment
        # equivalent Gaussian in log1p-time, no longer a hard current-event
        # component.  Existing ensemble/calibration code can consume it
        # without a tensor-schema migration.
        output["duration_selected_log_mean"] = duration_mixture_log_mean
        output["duration_selected_log_scale"] = (
            duration_moment_equivalent_log_scale
        )

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
        terminal_hidden = self._terminal_hidden(
            output["transitioned"], terminal_context
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
        terminal_goal_progress_component_mean = (
            self.terminal_goal_progress_component_mean(
                terminal_hidden
            )
        )
        terminal_goal_progress_component_log_scale = torch.clamp(
            self.terminal_goal_progress_component_scale(terminal_hidden),
            CONSEQUENCE_LOG_SCALE_MIN,
            2.0,
        )
        terminal_event_probability = torch.softmax(terminal_event_logits, dim=-1)
        (
            terminal_goal_progress_mean,
            terminal_goal_progress_std,
            terminal_goal_progress_log_scale,
        ) = _event_conditioned_student_t3_mixture_moments(
            terminal_event_probability,
            terminal_goal_progress_component_mean,
            terminal_goal_progress_component_log_scale,
            dof=TERMINAL_PROGRESS_STUDENT_T_DOF,
        )
        output["terminal_event_logits"] = terminal_event_logits
        output["success_logit"] = terminal_success_logit
        output["recovery_logit"] = terminal_recovery_logit
        output["terminal_goal_progress_component_mean"] = (
            terminal_goal_progress_component_mean
        )
        output["terminal_goal_progress_component_log_scale"] = (
            terminal_goal_progress_component_log_scale
        )
        output["terminal_goal_progress_mixture_probability"] = (
            terminal_event_probability
        )
        output["terminal_goal_progress_mean"] = terminal_goal_progress_mean
        output["terminal_goal_progress_log_scale"] = (
            terminal_goal_progress_log_scale
        )
        output["terminal_goal_progress_std"] = terminal_goal_progress_std
        terminal_success_probability = terminal_event_probability[:, -1]
        conditional_recovery_probability = torch.sigmoid(
            terminal_recovery_logit
        )
        output["success_probability"] = terminal_success_probability
        output["failure_probability"] = 1.0 - terminal_success_probability
        output["conditional_recovery_probability"] = (
            conditional_recovery_probability
        )

        def categorical_entropy(probability: torch.Tensor) -> torch.Tensor:
            return -(
                probability
                * torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny))
            ).sum(dim=-1)

        def binary_entropy(probability: torch.Tensor) -> torch.Tensor:
            epsilon = torch.finfo(probability.dtype).eps
            probability = probability.clamp(epsilon, 1.0 - epsilon)
            return -(
                probability * torch.log(probability)
                + (1.0 - probability) * torch.log1p(-probability)
            )

        output["post_event_aleatoric_entropy"] = categorical_entropy(
            post_probability
        )
        output["next_event_aleatoric_entropy"] = categorical_entropy(
            next_event_probability
        )
        output["terminal_event_aleatoric_entropy"] = categorical_entropy(
            terminal_event_probability
        )
        output["success_aleatoric_entropy"] = binary_entropy(
            terminal_success_probability
        )
        output["conditional_recovery_aleatoric_entropy"] = binary_entropy(
            conditional_recovery_probability
        )

        # The deployed score is deliberately a function of predicted
        # consequences, never a free projection of ``transitioned`` or
        # ``clock_hidden``.  Detaching the complete feature vector keeps the
        # proper event/outcome/effect likelihoods calibrated: listwise rank
        # supervision learns how to combine their predictions, not how to
        # rewrite them into an unconstrained latent critic.
        next_probability = next_event_probability
        success_probability = terminal_success_probability
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
        joint_recovery_probability = (
            regression_probability * conditional_recovery_probability[:, None]
        )
        output["regression_probability"] = regression_probability.squeeze(-1)
        output["joint_recovery_probability"] = (
            joint_recovery_probability.squeeze(-1)
        )
        output["joint_recovery_aleatoric_entropy"] = binary_entropy(
            joint_recovery_probability.squeeze(-1)
        )
        duration_component_sigma = torch.exp(duration_log_scale)
        duration_component_expectation = torch.expm1(
            (
                duration_log_mean
                + 0.5 * duration_component_sigma.square()
            ).clamp(min=0.0, max=10.0)
        )
        expected_duration_seconds = (
            next_probability * duration_component_expectation
        ).sum(dim=-1)
        object_mean = output["object_delta_mean"]
        state = batch["state"]
        if state.ndim != 2 or state.shape[-1] != core.STATE_DIM:
            raise FiveBodyContractError(
                "consequence rank utility requires one canonical 27-D root state"
            )
        relative_goal = state[:, :3].to(object_mean)
        predicted_remaining_component = (
            relative_goal[:, None, :] - object_component_mean[:, :, :3]
        )
        current_distance = torch.linalg.vector_norm(relative_goal, dim=-1)
        predicted_distance_component = torch.linalg.vector_norm(
            predicted_remaining_component, dim=-1
        )
        predicted_goal_progress_component = (
            current_distance[:, None] - predicted_distance_component
        )
        predicted_goal_progress = (
            post_probability * predicted_goal_progress_component
        ).sum(dim=-1)

        # Delta-method radial uncertainty for the Student-t(3) translation
        # effect.  At a zero predicted residual, fall back to the current goal
        # direction; if both vectors are zero, use the isotropic RMS scale.
        epsilon = torch.finfo(object_mean.dtype).eps**0.5
        remaining_unit = predicted_remaining_component / (
            predicted_distance_component[:, :, None].clamp_min(epsilon)
        )
        current_unit = relative_goal / current_distance[:, None].clamp_min(epsilon)
        has_remaining_direction = predicted_distance_component > epsilon
        direction = torch.where(
            has_remaining_direction[:, :, None],
            remaining_unit,
            current_unit[:, None, :],
        )
        has_direction = has_remaining_direction | (current_distance[:, None] > epsilon)
        translation_variance = torch.exp(
            2.0 * object_component_log_scale[:, :, :3]
        )
        projected_variance = (direction.square() * translation_variance).sum(dim=-1)
        isotropic_variance = translation_variance.mean(dim=-1)
        within_component_progress_variance = OBJECT_STUDENT_T_DOF * torch.where(
            has_direction, projected_variance, isotropic_variance
        )
        centered_goal_progress_component = (
            predicted_goal_progress_component
            - predicted_goal_progress[:, None]
        )
        predicted_goal_progress_variance = (
            post_probability
            * (
                within_component_progress_variance
                + centered_goal_progress_component.square()
            )
        ).sum(dim=-1)
        predicted_goal_progress_uncertainty = torch.sqrt(
            predicted_goal_progress_variance.clamp_min(
                OBJECT_STUDENT_T_DOF
                * math.exp(2.0 * CONSEQUENCE_LOG_SCALE_MIN)
            )
        )

        normalized_event_level = event_index.to(post_probability) / float(
            len(core.CANONICAL_EVENTS) - 1
        )
        post_expected_stage = (
            post_probability * normalized_event_level
        ).sum(dim=-1)
        positive_next_delta = (
            event_index.to(next_probability) - current
        ).clamp_min(0.0) / float(len(core.CANONICAL_EVENTS) - 1)
        next_advance_probability = (
            next_probability * positive_next_delta
        ).sum(dim=-1)
        next_event_advance_rate = next_advance_probability / (
            1.0 + expected_duration_seconds
        )
        no_unrecovered_regression = 1.0 - (
            regression_probability.squeeze(-1)
            - joint_recovery_probability.squeeze(-1)
        ).clamp(0.0, 1.0)
        short_goal_benefit = (
            post_probability
            * _goal_progress_benefit_value(predicted_goal_progress_component)
        ).sum(dim=-1)
        short_goal_risk = (
            predicted_goal_progress_uncertainty
            / (
                predicted_goal_progress_uncertainty
                + GOAL_PROGRESS_NORMALIZATION_METERS
            )
        ).clamp(0.0, 1.0)
        terminal_expected_stage = (
            terminal_event_probability * normalized_event_level
        ).sum(dim=-1)
        terminal_goal_benefit = (
            terminal_event_probability
            * _goal_progress_benefit_value(
                terminal_goal_progress_component_mean
            )
        ).sum(dim=-1)
        terminal_goal_risk = (
            terminal_goal_progress_std
            / (terminal_goal_progress_std + GOAL_PROGRESS_NORMALIZATION_METERS)
        ).clamp(0.0, 1.0)
        if self.ablation_variant == "no_time_duration":
            next_event_advance_rate = torch.zeros_like(next_event_advance_rate)
        if self.ablation_variant == "no_object_effect":
            short_goal_benefit = torch.zeros_like(short_goal_benefit)
            short_goal_risk = torch.zeros_like(short_goal_risk)
            terminal_goal_benefit = torch.zeros_like(terminal_goal_benefit)
            terminal_goal_risk = torch.zeros_like(terminal_goal_risk)
        rank_features = torch.stack(
            (
                post_expected_stage,
                next_event_advance_rate,
                success_probability,
                no_unrecovered_regression,
                short_goal_benefit,
                short_goal_risk,
                terminal_expected_stage,
                terminal_goal_benefit,
                terminal_goal_risk,
            ),
            dim=-1,
        ).detach()
        if rank_features.shape != (
            output["success_logit"].shape[0],
            CANDIDATE_RANK_FEATURE_DIM,
        ):
            raise FiveBodyContractError("consequence rank feature schema changed")
        output["predicted_goal_progress"] = predicted_goal_progress
        output["predicted_goal_progress_component"] = (
            predicted_goal_progress_component
        )
        output["predicted_goal_progress_uncertainty"] = (
            predicted_goal_progress_uncertainty
        )
        output["expected_duration_seconds"] = expected_duration_seconds
        output["candidate_rank_features"] = rank_features
        if self.ablation_variant == "success_only":
            candidate_rank_logit = success_probability
        else:
            candidate_rank_logit = self.candidate_rank(
                rank_features, batch["current_event_id"]
            )
            if candidate_rank_logit.shape == (len(rank_features), 1):
                candidate_rank_logit = candidate_rank_logit[:, 0]
        if candidate_rank_logit.shape != output["success_logit"].shape:
            raise FiveBodyContractError("candidate utility output shape changed")
        output["candidate_rank_logit"] = candidate_rank_logit
        return output

    def _terminal_hidden(
        self,
        transitioned: torch.Tensor,
        terminal_context: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse event age/horizon with consequences through bounded FiLM.

        A purely additive context followed by a linear terminal head makes the
        difference between two candidates invariant to their shared horizon.
        FiLM changes the candidate representation itself, while the bounded
        residual trunk supplies a small nonlinear interaction without exposing
        a direct latent rank path.
        """

        if transitioned.ndim != 2 or transitioned.shape[-1] != core.SEMANTIC_DIM:
            raise FiveBodyContractError("terminal transitioned state must be [B,96]")
        if terminal_context.shape != (transitioned.shape[0], 2):
            raise FiveBodyContractError("terminal context must be [B,2]")
        film = self.terminal_context_encoder(terminal_context)
        film_scale, film_shift = film.chunk(2, dim=-1)
        bounded_scale = TERMINAL_FILM_MODULATION_BOUND * torch.tanh(film_scale)
        bounded_shift = TERMINAL_FILM_MODULATION_BOUND * torch.tanh(film_shift)
        modulated = (1.0 + bounded_scale) * transitioned + bounded_shift
        residual = TERMINAL_FILM_MODULATION_BOUND * torch.tanh(
            self.terminal_residual(modulated)
        )
        return modulated + residual


class RiskAdjustedRankEnsemble(torch.nn.Module):
    """Source-validation epistemic LCB identical to the deployment scorer."""

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
            aggregate[selected] = aggregate_risk_adjusted_rank_scores(
                member_scores[:, selected]
            )
        return {"candidate_rank_logit": aggregate}


@torch.no_grad()
def evaluate_terminal_consequences(
    model: EffectAlignedSharedEventHead,
    loader: DataLoader,
    device: torch.device,
    *,
    ablation_variant: str | None = None,
) -> dict[str, Any]:
    """Report accuracy/calibration of finite-horizon proper consequences."""

    model.eval()
    variant = ablation_variant or getattr(model, "ablation_variant", "full")
    if variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {variant!r}")
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    logical_groups: list[str] = []
    for raw in loader:
        batch = core._move_batch(raw, device)
        output = model(batch)
        logical_groups.extend(str(value) for value in raw["logical_group"])
        observed_terminal_goal_mean = _gather_observed_event_component(
            output["terminal_goal_progress_component_mean"],
            batch["terminal_max_event_id"].long(),
        )
        observed_terminal_goal_log_scale = _gather_observed_event_component(
            output["terminal_goal_progress_component_log_scale"],
            batch["terminal_max_event_id"].long(),
        )
        tensors = {
            "event_label": batch["terminal_max_event_id"],
            "event_mask": batch["terminal_event_mask"],
            "event_probability": torch.softmax(
                output["terminal_event_logits"], dim=-1
            ),
            "goal_label": batch["terminal_goal_progress"],
            "goal_mask": (
                batch["terminal_goal_progress_mask"]
                * batch["terminal_event_mask"]
            ),
            "goal_mean": output["terminal_goal_progress_mean"],
            "goal_log_scale": output["terminal_goal_progress_log_scale"],
            "goal_conditional_mean": observed_terminal_goal_mean,
            "goal_conditional_log_scale": observed_terminal_goal_log_scale,
            "success_label": batch["success"],
            "success_mask": batch["success_mask"],
            "success_probability": torch.sigmoid(output["success_logit"]),
            "post_label": batch["post_event_id"],
            "post_mask": batch["post_event_mask"],
            "current_event": batch["current_event_id"],
            "recovery_label": batch["recovery"],
            "regression_probability": output["regression_probability"],
            "joint_recovery_probability": output[
                "joint_recovery_probability"
            ],
            "requested_seed": batch["requested_seed"],
        }
        if variant != "success_only":
            tensors.update(
                {
                    "post_nll": torch.nn.functional.cross_entropy(
                        output["post_event_logits"],
                        batch["post_event_id"].long(),
                        reduction="none",
                    ),
                    "post_proper_mask": batch["post_event_mask"],
                    "next_nll": torch.nn.functional.cross_entropy(
                        output["next_event_logits"],
                        batch["next_event_id"].long(),
                        reduction="none",
                    ),
                    "next_proper_mask": batch["next_event_mask"],
                    "recovery_nll": (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            output["recovery_logit"],
                            batch["recovery"].to(output["recovery_logit"]),
                            reduction="none",
                        )
                    ),
                    "recovery_proper_mask": (
                        batch["recovery_mask"] * batch["action_available"]
                    ),
                }
            )
            if variant != "no_time_duration":
                tensors.update(
                    {
                        "duration_nll": _competing_risks_duration_nll_rows(
                            output, batch
                        ),
                        "duration_proper_mask": batch["duration_mask"],
                    }
                )
            if variant != "no_object_effect":
                object_mean = _gather_observed_event_component(
                    output["object_delta_component_mean"],
                    batch["post_event_id"].long(),
                )
                object_log_scale = _gather_observed_event_component(
                    output["object_delta_component_log_scale"],
                    batch["post_event_id"].long(),
                )
                object_scale = torch.exp(object_log_scale).clamp_min(1e-4)
                object_standardized = (
                    batch["object_delta"].to(object_mean)
                    - object_mean
                ) / object_scale
                tensors.update(
                    {
                        "object_student_t3_nll": (
                            object_log_scale
                            + 2.0
                            * torch.log1p(object_standardized.square() / 3.0)
                        ).mean(dim=-1),
                        "object_proper_mask": (
                            batch["object_delta_mask"]
                            * batch["action_available"]
                            * batch["post_event_mask"]
                        ),
                        "object_abs_student_t3_standardized": (
                            object_standardized.abs()
                        ),
                        "object_log_scale": object_log_scale,
                    }
                )
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
    event_cumulative_probability = np.cumsum(event_probability, axis=-1)[:, :-1]
    event_observed_cumulative = (
        event_label[:, None] <= np.arange(len(core.CANONICAL_EVENTS) - 1)[None]
    ).astype(np.float64)
    event_ordinal_rps = np.mean(
        np.square(event_cumulative_probability - event_observed_cumulative),
        axis=-1,
    )

    goal_mask = (values["goal_mask"] > 0.5) & event_mask
    goal_label = values["goal_label"][goal_mask].astype(np.float64)
    goal_mean = values["goal_mean"][goal_mask].astype(np.float64)
    goal_log_scale = values["goal_log_scale"][goal_mask].astype(np.float64)
    goal_scale = np.exp(goal_log_scale).clip(min=1e-5)
    goal_conditional_mean = values["goal_conditional_mean"][goal_mask].astype(
        np.float64
    )
    goal_conditional_log_scale = values[
        "goal_conditional_log_scale"
    ][goal_mask].astype(np.float64)
    goal_conditional_scale = np.exp(goal_conditional_log_scale).clip(min=1e-5)
    goal_standardized = (
        goal_label - goal_conditional_mean
    ) / goal_conditional_scale
    dof = TERMINAL_PROGRESS_STUDENT_T_DOF
    goal_normalizer = (
        math.lgamma(dof / 2.0)
        + 0.5 * math.log(dof * math.pi)
        - math.lgamma((dof + 1.0) / 2.0)
    )
    goal_nll = (
        goal_conditional_log_scale
        + 0.5 * (dof + 1.0) * np.log1p(np.square(goal_standardized) / dof)
        + goal_normalizer
    )
    central_90_t3 = 2.3533634348018264

    success_mask = values["success_mask"] > 0.5
    success_label = values["success_label"].astype(np.float64)
    success_probability = np.clip(
        values["success_probability"].astype(np.float64), 1e-12, 1.0 - 1e-12
    )
    success_nll_rows = (
        -success_label * np.log(success_probability)
        - (1.0 - success_label) * np.log1p(-success_probability)
    )

    if len(logical_groups) != len(values["success_label"]):
        raise FiveBodyContractError("strict proper validation group alignment changed")
    row_event_nll = np.full(len(logical_groups), np.nan, dtype=np.float64)
    row_event_nll[event_mask] = event_nll
    row_event_ordinal_rps = np.full(
        len(logical_groups), np.nan, dtype=np.float64
    )
    row_event_ordinal_rps[event_mask] = event_ordinal_rps
    row_goal_nll = np.full(len(logical_groups), np.nan, dtype=np.float64)
    row_goal_nll[goal_mask] = goal_nll
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(logical_groups):
        by_group[group].append(index)
    strict_group_rows: list[tuple[str, str, float]] = []
    component_group_rows: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for group, indices in sorted(by_group.items()):
        if len(indices) != CANDIDATE_COUNT:
            raise FiveBodyContractError(
                "strict proper validation split a complete candidate decision"
            )
        unit_parts = group.split("|", 2)
        if (
            len(unit_parts) != 3
            or unit_parts[0] not in BODIES
            or unit_parts[1] not in CONDITIONS
        ):
            raise FiveBodyContractError(
                "strict proper validation logical group identity changed"
            )
        unit = "|".join(unit_parts[:2])
        selected = np.asarray(indices, dtype=np.int64)
        requested_seeds = values["requested_seed"][selected].astype(np.int64)
        if not np.all(requested_seeds == requested_seeds[0]):
            raise FiveBodyContractError(
                "strict proper validation requested seed changed within a decision"
            )
        seed_cluster = f"{unit}|requested_seed={int(requested_seeds[0])}"

        def supervised_mean(rows: np.ndarray, mask: np.ndarray, name: str) -> float:
            active = mask[selected]
            if not active.any():
                raise FiveBodyContractError(
                    f"strict proper validation group lacks {name} supervision"
                )
            value = float(np.mean(rows[selected][active]))
            if not math.isfinite(value):
                raise FiveBodyContractError(
                    f"strict proper validation {name} is non-finite"
                )
            return value

        def optional_supervised_mean(
            rows: np.ndarray, mask: np.ndarray, name: str
        ) -> float | None:
            active = mask[selected]
            if not active.any():
                return None
            value = float(np.mean(rows[selected][active]))
            if not math.isfinite(value):
                raise FiveBodyContractError(
                    f"strict proper validation {name} is non-finite"
                )
            return value

        success_component = supervised_mean(
            success_nll_rows, success_mask, "success"
        )
        strict = success_component
        if variant != "success_only":
            auxiliary_components = (
                (
                    "post_event_nll",
                    values["post_nll"],
                    values["post_proper_mask"] > 0.5,
                    float(core.DEFAULT_LOSS_WEIGHTS["post_event"]),
                ),
                (
                    "next_event_nll",
                    values["next_nll"],
                    values["next_proper_mask"] > 0.5,
                    float(core.DEFAULT_LOSS_WEIGHTS["next_event"]),
                ),
                *(
                    ()
                    if variant == "no_time_duration"
                    else (
                        (
                            "duration_censored_lognormal_nll",
                            values["duration_nll"],
                            values["duration_proper_mask"] > 0.5,
                            float(core.DEFAULT_LOSS_WEIGHTS["duration"]),
                        ),
                    )
                ),
                (
                    "recovery_binary_nll",
                    values["recovery_nll"],
                    values["recovery_proper_mask"] > 0.5,
                    float(core.DEFAULT_LOSS_WEIGHTS["recovery"]),
                ),
                *(
                    ()
                    if variant == "no_object_effect"
                    else (
                        (
                            "object_student_t3_nll",
                            values["object_student_t3_nll"],
                            values["object_proper_mask"] > 0.5,
                            0.5,
                        ),
                    )
                ),
            )
            for name, rows, mask, weight in auxiliary_components:
                component = optional_supervised_mean(rows, mask, name)
                if component is None:
                    continue
                strict += weight * component
                component_group_rows[name].append(
                    (unit, seed_cluster, component)
                )
            event_component = supervised_mean(
                row_event_nll, values["event_mask"] > 0.5, "terminal event"
            )
            strict += TERMINAL_EVENT_LOSS_WEIGHT * event_component
            event_ordinal_rps_component = supervised_mean(
                row_event_ordinal_rps,
                values["event_mask"] > 0.5,
                "terminal event ordinal RPS",
            )
            strict += (
                TERMINAL_EVENT_ORDINAL_RPS_LOSS_WEIGHT
                * event_ordinal_rps_component
            )
        if variant not in {"success_only", "no_object_effect"}:
            goal_component = supervised_mean(
                row_goal_nll, values["goal_mask"] > 0.5, "terminal goal"
            )
            strict += TERMINAL_GOAL_PROGRESS_LOSS_WEIGHT * goal_component
        strict_group_rows.append((unit, seed_cluster, strict))
        component_group_rows["success_nll"].append(
            (unit, seed_cluster, success_component)
        )
        if variant != "success_only":
            component_group_rows["terminal_event_nll"].append(
                (unit, seed_cluster, event_component)
            )
            component_group_rows["terminal_event_ordinal_rps"].append(
                (unit, seed_cluster, event_ordinal_rps_component)
            )
        if variant not in {"success_only", "no_object_effect"}:
            component_group_rows["terminal_goal_student_t3_nll"].append(
                (unit, seed_cluster, goal_component)
            )

    def macro_and_standard_error(
        rows: Sequence[tuple[str, str, float]],
    ) -> tuple[float, float, int]:
        values_by_unit_cluster: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for unit, seed_cluster, value in rows:
            values_by_unit_cluster[unit][seed_cluster].append(float(value))
        cluster_means_by_unit: dict[str, list[float]] = {}
        for unit, clusters in values_by_unit_cluster.items():
            if len(clusters) < 2:
                raise FiveBodyContractError(
                    "strict proper one-SE validation requires at least two "
                    f"independent requested seeds for {unit}"
                )
            cluster_means_by_unit[unit] = [
                float(np.mean(cluster_rows))
                for cluster_rows in clusters.values()
            ]
        unit_means = [
            float(np.mean(cluster_means))
            for cluster_means in cluster_means_by_unit.values()
        ]
        macro = float(np.mean(unit_means))
        variance = 0.0
        for cluster_means in cluster_means_by_unit.values():
            variance += float(np.var(cluster_means, ddof=1)) / len(cluster_means)
        standard_error = math.sqrt(variance) / len(cluster_means_by_unit)
        independent_clusters = sum(
            len(clusters) for clusters in values_by_unit_cluster.values()
        )
        return macro, float(standard_error), independent_clusters

    strict_macro, strict_standard_error, independent_seed_clusters = macro_and_standard_error(
        strict_group_rows
    )

    def component_macro(rows: Sequence[tuple[str, str, float]]) -> float:
        values_by_unit_seed: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for unit, seed_cluster, value in rows:
            values_by_unit_seed[unit][seed_cluster].append(float(value))
        if not values_by_unit_seed:
            raise FiveBodyContractError("strict proper component has no support")
        return float(
            np.mean(
                [
                    float(
                        np.mean(
                            [
                                float(np.mean(seed_values))
                                for seed_values in clusters.values()
                            ]
                        )
                    )
                    for clusters in values_by_unit_seed.values()
                ]
            )
        )

    strict_components = {
        name: component_macro(rows)
        for name, rows in component_group_rows.items()
    }

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

    if len(event_label):
        event_metrics = {
            **core._event_metrics(event_label, event_prediction),
            "support": int(len(event_label)),
            "class_counts": np.bincount(event_label, minlength=5).tolist(),
            "nll": float(np.mean(event_nll)),
            "multiclass_brier": float(
                np.mean(np.sum(np.square(event_probability - event_onehot), axis=-1))
            ),
            "ordinal_ranked_probability_score": float(
                np.mean(event_ordinal_rps)
            ),
            "ordinal_mae": float(np.mean(np.abs(event_prediction - event_label))),
        }
    else:
        event_metrics = {
            "support": 0,
            "class_counts": [0] * 5,
            "macro_f1": None,
            "accuracy": None,
            "nll": None,
            "multiclass_brier": None,
            "ordinal_ranked_probability_score": None,
            "ordinal_mae": None,
        }
    if len(goal_label):
        goal_metrics = {
            "support": int(len(goal_label)),
            "mae_meters": float(np.mean(np.abs(goal_mean - goal_label))),
            "rmse_meters": float(np.sqrt(np.mean(np.square(goal_mean - goal_label)))),
            "student_t3_nll": float(np.mean(goal_nll)),
            "central_90_coverage": float(
                np.mean(np.abs(goal_label - goal_mean) <= central_90_t3 * goal_scale)
            ),
            "point_prediction": "terminal_event_mixture_mean",
            "student_t3_nll_distribution": (
                "observed_terminal_event_conditional_component"
            ),
            "central_90_coverage_interpretation": (
                "moment_matched_student_t3_approximation_to_event_mixture"
            ),
        }
    else:
        goal_metrics = {
            "support": 0,
            "mae_meters": None,
            "rmse_meters": None,
            "student_t3_nll": None,
            "central_90_coverage": None,
            "point_prediction": "terminal_event_mixture_mean",
            "student_t3_nll_distribution": (
                "observed_terminal_event_conditional_component"
            ),
            "central_90_coverage_interpretation": (
                "moment_matched_student_t3_approximation_to_event_mixture"
            ),
        }
    if variant not in {"success_only", "no_object_effect"}:
        object_mask = values["object_proper_mask"] > 0.5
        object_nll_rows = values["object_student_t3_nll"][object_mask]
        object_abs_standardized = values[
            "object_abs_student_t3_standardized"
        ][object_mask]
        object_log_scale = values["object_log_scale"][object_mask]
        object_scale = np.exp(object_log_scale)
        object_active_indices = np.flatnonzero(object_mask)

        def object_seed_cluster_macro(row_values: np.ndarray) -> float | None:
            if not len(row_values):
                return None
            by_unit_seed: dict[str, dict[int, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for active_position, row_index in enumerate(object_active_indices):
                identity = logical_groups[int(row_index)].split("|", 2)
                if len(identity) != 3:
                    raise FiveBodyContractError(
                        "object uncertainty logical group identity changed"
                    )
                unit = "|".join(identity[:2])
                seed = int(values["requested_seed"][int(row_index)])
                by_unit_seed[unit][seed].append(float(row_values[active_position]))
            return float(
                np.mean(
                    [
                        np.mean(
                            [np.mean(seed_rows) for seed_rows in seeds.values()]
                        )
                        for seeds in by_unit_seed.values()
                    ]
                )
            )

        object_transition_metrics = {
            "support_rows": int(object_mask.sum()),
            "support_components": int(object_abs_standardized.size),
            "student_t3_nll_without_additive_normalizer": (
                object_seed_cluster_macro(object_nll_rows)
            ),
            "central_90_component_coverage": (
                object_seed_cluster_macro(
                    np.mean(
                        object_abs_standardized <= central_90_t3,
                        axis=-1,
                    )
                )
            ),
            "log_scale_floor_fraction": (
                object_seed_cluster_macro(
                    np.mean(
                        object_log_scale <= CONSEQUENCE_LOG_SCALE_MIN + 1e-6,
                        axis=-1,
                    )
                )
            ),
            "scale_quantiles_05_50_95": (
                [
                    float(value)
                    for value in np.quantile(object_scale, [0.05, 0.5, 0.95])
                ]
                if object_scale.size
                else None
            ),
            "likelihood": (
                "observed_post_event_conditional_independent_student_t_dof_3"
            ),
            "deployment_prediction": (
                "post_event_mixture_mean_and_total_within_plus_between_std"
            ),
            "proper_metric_aggregation": (
                "requested_seed_cluster_then_body_condition_macro"
            ),
            "scale_quantile_aggregation": "raw_component_descriptive_only",
        }
    else:
        object_transition_metrics = {
            "support_rows": 0,
            "support_components": 0,
            "student_t3_nll_without_additive_normalizer": None,
            "central_90_component_coverage": None,
            "log_scale_floor_fraction": None,
            "scale_quantiles_05_50_95": None,
            "likelihood": "disabled_by_ablation",
            "deployment_prediction": "disabled_by_ablation",
            "proper_metric_aggregation": (
                "requested_seed_cluster_then_body_condition_macro"
            ),
            "scale_quantile_aggregation": "raw_component_descriptive_only",
        }
    return {
        "strict_proper": {
            "macro_score": strict_macro,
            "macro_standard_error": strict_standard_error,
            "logical_decisions": len(strict_group_rows),
            "body_condition_units": len(
                {unit for unit, _seed_cluster, _value in strict_group_rows}
            ),
            "independent_requested_seed_clusters": independent_seed_clusters,
            "components": strict_components,
            "selection_rule": (
                "source_body_condition_macro_seed_clustered_proper_loss_"
                "one_standard_error"
            ),
        },
        "terminal_success": binary_metrics(
            success_label[success_mask], success_probability[success_mask]
        ),
        "terminal_event": event_metrics,
        "terminal_goal_progress": goal_metrics,
        "object_transition": object_transition_metrics,
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


def _goal_progress_benefit_value(progress: torch.Tensor) -> torch.Tensor:
    """Map metric progress to the single deployed [0, 1] benefit geometry."""

    if not isinstance(progress, torch.Tensor) or not bool(
        torch.isfinite(progress).all()
    ):
        raise FiveBodyContractError("dense goal progress contains non-finite values")
    return 0.5 * (
        torch.nn.functional.softsign(
            progress / DENSE_GOAL_PROGRESS_TEMPERATURE_METERS
        )
        + 1.0
    )


def _dense_soft_target_distribution(
    terminal_event_level: torch.Tensor,
    terminal_goal_progress: torch.Tensor,
    *,
    ablation_variant: str,
) -> torch.Tensor:
    """Return the exact full-support target used by dense rank training."""

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(f"unknown ablation variant {ablation_variant!r}")
    if (
        terminal_event_level.ndim != 1
        or terminal_goal_progress.shape != terminal_event_level.shape
        or not bool(torch.isfinite(terminal_event_level).all())
        or not bool(torch.isfinite(terminal_goal_progress).all())
    ):
        raise FiveBodyContractError("dense soft target has invalid shape/value")
    maximum_mask = terminal_event_level == terminal_event_level.max()
    if not bool(maximum_mask.any()):
        raise FiveBodyContractError("dense soft target selected no event level")
    target = torch.zeros_like(terminal_goal_progress)
    maximum_goal_progress = terminal_goal_progress[maximum_mask]
    if (
        ablation_variant == "no_object_effect"
        or float(maximum_goal_progress.max() - maximum_goal_progress.min())
        <= DENSE_RANK_LABEL_EQUALITY_TOLERANCE
    ):
        preferred = torch.full_like(
            maximum_goal_progress,
            1.0 / int(maximum_mask.sum()),
        )
    else:
        preferred = torch.softmax(
            _goal_progress_benefit_value(maximum_goal_progress),
            dim=0,
        )
    target[maximum_mask] = preferred
    return target


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
    analytic near-goal scale (0.02 m) defines the same bounded [0, 1] softsign
    benefit coordinate used at deployment before softmax.  Spreads at or below
    the frozen 1e-4 equality tolerance stay uniform.  This avoids allowing
    numerical noise or an anomalous metric displacement to turn the target into
    a hard one-hot label.  ``no_object_effect`` uses a uniform target over the
    maximum level.
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
    target = _dense_soft_target_distribution(
        terminal_event_level,
        terminal_goal_progress,
        ablation_variant=ablation_variant,
    )
    return -(target * torch.log_softmax(scores, dim=0)).sum()


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
        if (
            abs(float(left_value) - float(right_value))
            <= DENSE_RANK_LABEL_EQUALITY_TOLERANCE
        ):
            continue
        return 1 if float(left_value) > float(right_value) else -1
    return 0


def _compute_shared_multitask_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    sample_weight: torch.Tensor | None = None,
    loss_weights: Mapping[str, float] = core.DEFAULT_LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Core proper heads plus next-event competing-risks duration.

    The core trainer's legacy duration term gathers a component by
    ``current_event_id``.  v12 keeps every other core loss unchanged, disables
    only that term, and inserts the destination-conditional likelihood.  This
    makes the change auditable and avoids duplicating the event, success and
    recovery implementations.
    """

    batch_size = output["success_logit"].shape[0]
    if sample_weight is None:
        sample_weight = output["success_logit"].new_ones(batch_size)
    if sample_weight.shape != (batch_size,):
        raise FiveBodyContractError("sample weight must be [B]")
    weights = dict(loss_weights)
    if set(weights) != set(core.DEFAULT_LOSS_WEIGHTS):
        raise FiveBodyContractError("shared multitask loss weight schema changed")
    action_available = batch["action_available"].to(sample_weight)
    post = core._weighted_mean(
        torch.nn.functional.cross_entropy(
            output["post_event_logits"],
            batch["post_event_id"].long(),
            reduction="none",
        ),
        sample_weight * batch["post_event_mask"].to(sample_weight),
    )
    next_event = core._weighted_mean(
        torch.nn.functional.cross_entropy(
            output["next_event_logits"],
            batch["next_event_id"].long(),
            reduction="none",
        ),
        sample_weight * batch["next_event_mask"].to(sample_weight),
    )
    success = core._weighted_mean(
        torch.nn.functional.binary_cross_entropy_with_logits(
            output["success_logit"],
            batch["success"].to(output["success_logit"]),
            reduction="none",
        ),
        sample_weight * batch["success_mask"].to(sample_weight),
    )
    recovery_weight = (
        sample_weight
        * action_available
        * batch["recovery_mask"].to(sample_weight)
    )
    recovery = core._weighted_mean(
        torch.nn.functional.binary_cross_entropy_with_logits(
            output["recovery_logit"],
            batch["recovery"].to(output["recovery_logit"]),
            reduction="none",
        ),
        recovery_weight,
    )
    object_scale = torch.exp(output["object_delta_log_scale"]).clamp_min(1e-4)
    object_nll = (
        0.5
        * (
            (batch["object_delta"] - output["object_delta_mean"])
            / object_scale
        ).square()
        + output["object_delta_log_scale"]
        + 0.5 * math.log(2.0 * math.pi)
    ).mean(-1)
    object_weight = (
        sample_weight
        * action_available
        * batch["object_delta_mask"].to(sample_weight)
    )
    object_loss = core._weighted_mean(object_nll, object_weight)
    duration_rows = _competing_risks_duration_nll_rows(output, batch)
    duration = core._weighted_mean(
        duration_rows,
        sample_weight * batch["duration_mask"].to(sample_weight),
    )
    pieces = {
        "post_event": post,
        "next_event": next_event,
        "duration": duration,
        "duration_next_event_competing_risks": duration,
        "success": success,
        "recovery": recovery,
        "object": object_loss,
    }
    total = output["success_logit"].sum() * 0.0
    for name in core.DEFAULT_LOSS_WEIGHTS:
        weight = float(weights[name])
        if weight != 0.0:
            total = total + weight * pieces[name]
    pieces["total"] = total
    pieces["recovery_supervised_rows"] = (recovery_weight > 0).sum().to(total)
    pieces["object_supervised_rows"] = (object_weight > 0).sum().to(total)
    return total, pieces


def _robust_object_effect_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Student-t object likelihood for the uniform proper-likelihood stream."""

    # Together with the post-event categorical CE, gathering the observed
    # event component is the joint p(post_event) p(delta | post_event) proper
    # likelihood.  No target event is fed to deployment predictions.
    observed_event = batch["post_event_id"].long()
    object_mean = _gather_observed_event_component(
        output["object_delta_component_mean"], observed_event
    )
    object_log_scale = _gather_observed_event_component(
        output["object_delta_component_log_scale"], observed_event
    )
    object_scale = torch.exp(object_log_scale).clamp_min(1e-4)
    standardized = (
        batch["object_delta"].to(object_mean)
        - object_mean
    ) / object_scale
    object_rows = (
        object_log_scale
        + 2.0 * torch.log1p(standardized.square() / 3.0)
    ).mean(dim=-1)
    object_effect = core._weighted_mean(
        object_rows,
        sample_weight
        * batch["action_available"].to(sample_weight)
        * batch["object_delta_mask"].to(sample_weight)
        * batch["post_event_mask"].to(sample_weight),
    )
    if ablation_variant in {"success_only", "no_object_effect"}:
        object_effect = object_effect * 0.0
    total = 0.5 * object_effect
    return total, {
        "robust_object_effect_uniform_proper": object_effect,
        "robust_object_effect_weighted_uniform_proper": total,
    }


def _terminal_event_ordinal_rps_rows(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Strictly proper ordinal score over e0<e12<e3<e4<eK.

    Categorical cross-entropy estimates the full terminal-event distribution but
    treats all wrong classes alike.  The ranked probability score compares its
    four cumulative probabilities with the observed cumulative event, allowing
    abundant all-failure e0..e4 outcomes to share ordinal supervision without
    inventing a success label or collapsing the distribution to its mean.
    """

    if (
        logits.ndim != 2
        or logits.shape[1] != len(core.CANONICAL_EVENTS)
        or target.shape != (logits.shape[0],)
        or not bool(torch.isfinite(logits).all())
        or target.dtype
        not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        or bool(((target < 0) | (target >= len(core.CANONICAL_EVENTS))).any())
    ):
        raise FiveBodyContractError(
            "terminal ordinal ranked probability score input is invalid"
        )
    cumulative_probability = torch.softmax(logits, dim=-1).cumsum(dim=-1)[:, :-1]
    thresholds = torch.arange(
        len(core.CANONICAL_EVENTS) - 1, device=logits.device
    )[None]
    observed_cumulative = (target[:, None] <= thresholds).to(logits)
    return (cumulative_probability - observed_cumulative).square().mean(dim=-1)


def _terminal_consequence_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Proper finite-horizon consequence losses for one caller-owned stream."""

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
    ordinal_event_rows = _terminal_event_ordinal_rps_rows(
        output["terminal_event_logits"],
        batch["terminal_max_event_id"].long(),
    )
    terminal_event_ordinal_rps = core._weighted_mean(
        ordinal_event_rows, event_weight
    )

    terminal_event_target = batch["terminal_max_event_id"].long()
    goal_mean = _gather_observed_event_component(
        output["terminal_goal_progress_component_mean"], terminal_event_target
    )
    goal_log_scale = _gather_observed_event_component(
        output["terminal_goal_progress_component_log_scale"],
        terminal_event_target,
    )
    goal_target = batch["terminal_goal_progress"].to(goal_mean)
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
        * batch["terminal_event_mask"].to(sample_weight)
    )
    terminal_goal = core._weighted_mean(goal_rows, goal_weight)

    if ablation_variant == "success_only":
        terminal_event = terminal_event * 0.0
        terminal_event_ordinal_rps = terminal_event_ordinal_rps * 0.0
    if ablation_variant in {"success_only", "no_object_effect"}:
        terminal_goal = terminal_goal * 0.0
    weighted_event = TERMINAL_EVENT_LOSS_WEIGHT * terminal_event
    weighted_event_ordinal_rps = (
        TERMINAL_EVENT_ORDINAL_RPS_LOSS_WEIGHT * terminal_event_ordinal_rps
    )
    weighted_goal = TERMINAL_GOAL_PROGRESS_LOSS_WEIGHT * terminal_goal
    return weighted_event + weighted_event_ordinal_rps + weighted_goal, {
        "terminal_event_uniform_proper": terminal_event,
        "terminal_event_ordinal_rps_uniform_proper": terminal_event_ordinal_rps,
        "terminal_goal_progress_uniform_proper": terminal_goal,
        "terminal_event_weighted_uniform_proper": weighted_event,
        "terminal_event_ordinal_rps_weighted_uniform_proper": (
            weighted_event_ordinal_rps
        ),
        "terminal_goal_progress_weighted_uniform_proper": weighted_goal,
    }


def _supplement_proper_world_model_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    loss_weights: Mapping[str, float],
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the fixed train-only supplement weight to proper heads only."""

    multitask, multitask_pieces = _compute_shared_multitask_loss(
        output,
        batch,
        sample_weight=sample_weight,
        loss_weights=loss_weights,
    )
    object_effect, object_pieces = _robust_object_effect_loss(
        output,
        batch,
        sample_weight,
        ablation_variant=ablation_variant,
    )
    terminal, terminal_pieces = _terminal_consequence_loss(
        output,
        batch,
        sample_weight,
        ablation_variant=ablation_variant,
    )
    unweighted = multitask + object_effect + terminal
    weighted = SUPPLEMENT_PROPER_LOSS_WEIGHT * unweighted
    pieces = {
        "supplement_proper_unweighted": unweighted,
        "supplement_proper_weighted": weighted,
        "supplement_proper_fixed_lambda": weighted.new_tensor(
            SUPPLEMENT_PROPER_LOSS_WEIGHT
        ),
    }
    pieces.update(
        {
            f"supplement_multitask_{name}": value
            for name, value in multitask_pieces.items()
            if name != "total"
        }
    )
    pieces.update(
        {f"supplement_{name}": value for name, value in object_pieces.items()}
    )
    pieces.update(
        {f"supplement_{name}": value for name, value in terminal_pieces.items()}
    )
    return weighted, pieces


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
            "dense_rank_effective_weight": zero,
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
    dense_weight = (
        DENSE_FAILURE_RANK_WEIGHT if success_terms else DENSE_ONLY_RANK_WEIGHT
    )
    ranking = success_ranking + dense_weight * dense_ranking
    return ranking, {
        "group_listwise_success_mass_balanced_rank": success_ranking,
        "all_failure_dense_soft_listwise_balanced_rank": dense_ranking,
        "candidate_ranking_balanced_rank": ranking,
        "dense_rank_effective_weight": score.new_tensor(dense_weight),
        "mixed_success_groups_in_batch": score.new_tensor(mixed_success_groups),
        "all_failure_dense_groups_in_batch": score.new_tensor(
            all_failure_dense_groups
        ),
        "all_failure_uninformative_groups_in_batch": score.new_tensor(
            all_failure_uninformative_groups
        ),
    }


def _supplement_candidate_rank_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Low-weight source-only rank loss for expert-root actor branches.

    ``candidate_rank_logit`` is built only from detached consequence features,
    so this term updates the bounded monotone utility and cannot turn the
    expert-root distribution into comparative supervision for world-model or
    uncertainty heads.
    """

    raw, raw_pieces = _candidate_rank_loss(
        output,
        batch,
        sample_weight,
        ablation_variant=ablation_variant,
    )
    weighted = SUPPLEMENT_RANK_LOSS_WEIGHT * raw
    return weighted, {
        "supplement_candidate_rank_unweighted": raw,
        "supplement_candidate_rank_weighted": weighted,
        "supplement_candidate_rank_fixed_lambda": weighted.new_tensor(
            SUPPLEMENT_RANK_LOSS_WEIGHT
        ),
        **{
            f"supplement_rank_{name}": value
            for name, value in raw_pieces.items()
        },
    }


def _semantic_comparative_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    sample_weight: torch.Tensor,
    *,
    ablation_variant: str = "full",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Make terminal event locations decision-aware without altering scales.

    The uniform stream remains the sole estimator of absolute probability and
    uncertainty.  This rank-stream loss is group-relative: mixed outcomes
    contrast coherent terminal p(eK), while all-failure decisions contrast the
    probability of reaching the best observed terminal stage and, only among
    candidates tied at that stage, the terminal goal-progress location mean.
    No synthetic outcome is introduced and no scale head receives this loss.
    """

    reference = output["success_logit"]
    if ablation_variant == "success_only":
        zero = reference.sum() * 0.0
        return zero, {
            "semantic_comparative_mixed_success": zero,
            "semantic_comparative_dense_event": zero,
            "semantic_comparative_dense_goal": zero,
            "semantic_comparative_event_raw": zero,
            "semantic_comparative_goal_raw": zero,
            "semantic_comparative_raw": zero,
            "semantic_mixed_groups_in_batch": reference.new_tensor(0),
            "semantic_dense_groups_in_batch": reference.new_tensor(0),
        }
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(batch["logical_group"]):
        by_group[str(group)].append(index)
    mixed_terms: list[torch.Tensor] = []
    mixed_weights: list[torch.Tensor] = []
    dense_event_terms: list[torch.Tensor] = []
    dense_event_weights: list[torch.Tensor] = []
    dense_goal_terms: list[torch.Tensor] = []
    dense_goal_weights: list[torch.Tensor] = []
    for indices in by_group.values():
        if len(indices) != CANDIDATE_COUNT:
            raise FiveBodyContractError(
                "semantic comparison split a complete candidate decision"
            )
        selected = torch.as_tensor(indices, device=reference.device, dtype=torch.long)
        selected_weights = sample_weight[selected]
        if not bool(torch.isfinite(selected_weights).all()) or not bool(
            torch.allclose(
                selected_weights,
                selected_weights[:1].expand_as(selected_weights),
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise FiveBodyContractError(
                "semantic comparison bootstrap weight changed within a decision"
            )
        group_weight = selected_weights[0]
        if float(group_weight.detach()) <= 0.0:
            continue
        if not bool((batch["success_mask"].to(reference)[selected] > 0.5).all()):
            raise FiveBodyContractError(
                "semantic comparison lacks complete success supervision"
            )
        successful = batch["success"].to(reference)[selected] > 0.5
        if bool(successful.any()) and bool((~successful).any()):
            mixed_terms.append(
                _negative_log_probability_mass(
                    torch.nn.functional.logsigmoid(
                        output["success_logit"][selected]
                    ),
                    successful,
                )
            )
            mixed_weights.append(group_weight)
            continue
        if bool(successful.any()):
            continue
        if not bool(
            (batch["terminal_event_mask"].to(reference)[selected] > 0.5).all()
        ) or not bool(
            (
                batch["terminal_goal_progress_mask"].to(reference)[selected]
                > 0.5
            ).all()
        ):
            raise FiveBodyContractError(
                "semantic dense comparison lacks terminal supervision"
            )
        event_label = batch["terminal_max_event_id"].to(reference)[selected].long()
        goal_label = batch["terminal_goal_progress"].to(reference)[selected].float()
        if not _dense_rank_labels_are_orderable(
            event_label.detach().cpu().tolist(),
            goal_label.detach().cpu().tolist(),
            ablation_variant=ablation_variant,
        ):
            continue
        best_event = int(event_label.max().item())
        preferred = event_label == best_event
        event_log_probability = torch.log_softmax(
            output["terminal_event_logits"][selected], dim=-1
        )
        reaches_best_log_probability = torch.logsumexp(
            event_log_probability[:, best_event:], dim=-1
        )
        event_group_loss = _negative_log_probability_mass(
            reaches_best_log_probability, preferred
        )
        if (
            ablation_variant != "no_object_effect"
            and int(preferred.sum()) > 1
            and float(goal_label[preferred].max() - goal_label[preferred].min())
            > DENSE_RANK_LABEL_EQUALITY_TOLERANCE
        ):
            target = torch.softmax(
                _goal_progress_benefit_value(goal_label[preferred]),
                dim=0,
            )
            conditional_goal_mean = _gather_observed_event_component(
                output["terminal_goal_progress_component_mean"][selected],
                event_label,
            )
            predicted = _goal_progress_benefit_value(
                conditional_goal_mean[preferred]
            )
            goal_group_loss = -(
                target * torch.log_softmax(predicted, dim=0)
            ).sum()
            dense_goal_terms.append(goal_group_loss)
            dense_goal_weights.append(group_weight)
        dense_event_terms.append(event_group_loss)
        dense_event_weights.append(group_weight)

    mixed_loss = (
        core._weighted_mean(torch.stack(mixed_terms), torch.stack(mixed_weights))
        if mixed_terms
        else reference.sum() * 0.0
    )
    dense_event_loss = (
        core._weighted_mean(
            torch.stack(dense_event_terms), torch.stack(dense_event_weights)
        )
        if dense_event_terms
        else reference.sum() * 0.0
    )
    dense_goal_loss = (
        core._weighted_mean(
            torch.stack(dense_goal_terms), torch.stack(dense_goal_weights)
        )
        if dense_goal_terms
        else reference.sum() * 0.0
    )
    event_raw = mixed_loss + dense_event_loss
    goal_raw = dense_goal_loss
    raw = event_raw + goal_raw
    return raw, {
        "semantic_comparative_mixed_success": mixed_loss,
        "semantic_comparative_dense_event": dense_event_loss,
        "semantic_comparative_dense_goal": dense_goal_loss,
        "semantic_comparative_event_raw": event_raw,
        "semantic_comparative_goal_raw": goal_raw,
        "semantic_comparative_raw": raw,
        "semantic_mixed_groups_in_batch": reference.new_tensor(len(mixed_terms)),
        "semantic_dense_groups_in_batch": reference.new_tensor(
            len(dense_event_terms)
        ),
    }


def _relative_gradient_budget_scale(
    proper_loss: torch.Tensor,
    comparative_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    budget: float = SEMANTIC_COMPARATIVE_GRADIENT_BUDGET,
    scale_cap: float = SEMANTIC_GRADIENT_SCALE_CAP,
) -> torch.Tensor:
    """Bound comparative gradients relative to proper gradients on active paths.

    Both losses are evaluated on their complete, independently sampled streams.
    The returned scale is stop-gradient, so the rank-stream objective can shape
    relative candidate predictions but cannot exceed ``budget`` times the
    current proper-gradient norm on exactly the parameters reached by a
    non-zero comparative gradient.  Unrelated proper heads therefore cannot
    inflate a shared-world gradient budget.
    """

    selected = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not selected:
        raise FiveBodyContractError("semantic gradient budget has no trainable head")
    if (
        not math.isfinite(float(budget))
        or float(budget) < 0.0
        or not math.isfinite(float(scale_cap))
        or float(scale_cap) <= 0.0
    ):
        raise FiveBodyContractError("semantic gradient budget is invalid")
    if not comparative_loss.requires_grad or not proper_loss.requires_grad:
        return proper_loss.detach().new_zeros(())
    comparative_gradients = torch.autograd.grad(
        comparative_loss,
        selected,
        retain_graph=True,
        allow_unused=True,
    )
    proper_gradients = torch.autograd.grad(
        proper_loss,
        selected,
        retain_graph=True,
        allow_unused=True,
    )

    def norm(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
        terms = [
            gradient.detach().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        if not terms:
            return proper_loss.detach().new_zeros(())
        return torch.sqrt(torch.stack(terms).sum())

    active_indices = [
        index
        for index, gradient in enumerate(comparative_gradients)
        if gradient is not None and float(gradient.detach().square().sum()) > 0.0
    ]
    if not active_indices:
        return proper_loss.detach().new_zeros(())
    proper_norm = norm(tuple(proper_gradients[index] for index in active_indices))
    comparative_norm = norm(
        tuple(comparative_gradients[index] for index in active_indices)
    )
    if float(proper_norm) <= 0.0 or float(comparative_norm) <= 0.0:
        return proper_norm.new_zeros(())
    return torch.clamp(
        float(budget) * proper_norm / comparative_norm.clamp_min(1e-12),
        max=float(scale_cap),
    ).detach()


def _semantic_comparative_active_parameters(
    model: EffectAlignedSharedEventHead,
) -> tuple[torch.nn.Parameter, ...]:
    """Return the single audited union reached by semantic comparisons.

    Comparative supervision may shape canonical state/action transitions and
    terminal location predictions, including the horizon-interaction trunk.
    Aleatoric scale heads and the detached deployment utility are deliberately
    absent from this union.
    """

    modules = (
        model.semantic,
        model.action,
        model.transition,
        model.terminal_context_encoder,
        model.terminal_residual,
        model.terminal_event,
        model.terminal_goal_progress_component_mean,
    )
    parameters = tuple(
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    if not parameters or len({id(parameter) for parameter in parameters}) != len(
        parameters
    ):
        raise FiveBodyContractError(
            "semantic comparative active union is empty or contains duplicates"
        )
    excluded = {
        id(parameter)
        for module in (
            model.object_scale,
            model.object_delta_component_scale,
            model.duration_scale,
            model.terminal_goal_progress_component_scale,
        )
        for parameter in module.parameters()
    }
    if any(id(parameter) in excluded for parameter in parameters):
        raise FiveBodyContractError(
            "semantic comparative active union contains an uncertainty scale head"
        )
    return parameters


def _bounded_semantic_comparative_loss(
    proper_loss: torch.Tensor,
    comparative_loss: torch.Tensor,
    model: EffectAlignedSharedEventHead,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exactly one proper-relative cap over the audited active union."""

    scale = _relative_gradient_budget_scale(
        proper_loss,
        comparative_loss,
        _semantic_comparative_active_parameters(model),
    )
    return scale * comparative_loss, scale


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
    group_requested_seeds: dict[str, int] = {}
    for raw in loader:
        batch = core._move_batch(raw, device)
        output = model(batch)
        dense_components = _dense_rank_components(
            batch, output["candidate_rank_logit"], ablation_variant=variant
        )
        for index, group in enumerate(raw["logical_group"]):
            group = str(group)
            if "requested_seed" not in raw:
                raise FiveBodyContractError(
                    "validation ranking rows lack requested_seed clustering"
                )
            requested_seed = int(raw["requested_seed"][index])
            previous_seed = group_requested_seeds.setdefault(group, requested_seed)
            if previous_seed != requested_seed:
                raise FiveBodyContractError(
                    f"validation decision spans requested seeds: {group}"
                )
            groups[group].append(
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
    dense_soft_pairs: list[dict[str, Any]] = []
    for group, rows in groups.items():
        rows = sorted(rows)
        if len(rows) != CANDIDATE_COUNT or [row[0] for row in rows] != list(range(4)):
            raise FiveBodyContractError(f"validation decision incomplete: {group}")
        identity = group.split("|", 2)
        if len(identity) != 3 or identity[0] not in BODIES or identity[1] not in CONDITIONS:
            raise FiveBodyContractError(f"validation decision identity changed: {group}")
        selected_position = max(range(len(rows)), key=lambda index: rows[index][1])
        selected = rows[selected_position]
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
        dense_target: np.ndarray | None = None
        if dense_applicable:
            dense_target = (
                _dense_soft_target_distribution(
                    torch.as_tensor(
                        [float(row[3][0]) for row in rows], dtype=torch.float64
                    ),
                    torch.as_tensor(
                        [float(row[6]) for row in rows], dtype=torch.float64
                    ),
                    ablation_variant=variant,
                )
                .cpu()
                .numpy()
            )
        decision = {
            "body": identity[0],
            "condition": identity[1],
            "requested_seed": group_requested_seeds[group],
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
            "selected_dense_soft_target_probability": (
                float(dense_target[selected_position])
                if dense_target is not None
                else None
            ),
            "dense_soft_target_peak_probability": (
                float(dense_target.max()) if dense_target is not None else None
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
                            "requested_seed": group_requested_seeds[group],
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
                            "requested_seed": group_requested_seeds[group],
                            "correct": bool(score_difference * dense_sign > 0),
                        }
                    )
                if dense_target is not None:
                    target_difference = float(dense_target[left] - dense_target[right])
                    if abs(target_difference) > 1e-12:
                        dense_soft_pairs.append(
                            {
                                "body": identity[0],
                                "condition": identity[1],
                                "requested_seed": group_requested_seeds[group],
                                "correct": bool(
                                    score_difference * target_difference > 0.0
                                ),
                                "preference_weight": abs(target_difference),
                            }
                        )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def seed_cluster_mean(
            selected_rows: Sequence[Mapping[str, Any]], field: str
        ) -> float:
            clusters: dict[tuple[str, str, int], list[float]] = defaultdict(list)
            for row in selected_rows:
                clusters[
                    (
                        str(row["body"]),
                        str(row["condition"]),
                        int(row["requested_seed"]),
                    )
                ].append(float(row[field]))
            if not clusters:
                raise FiveBodyContractError(
                    f"cannot aggregate empty requested-seed clusters for {field}"
                )
            return float(
                np.mean([np.mean(values) for values in clusters.values()])
            )

        baseline = seed_cluster_mean(rows, "baseline_success")
        selected = seed_cluster_mean(rows, "selected_success")
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
                seed_cluster_mean(rows, "oracle_success")
            ),
            "baseline_terminal_stage_progress": float(
                seed_cluster_mean(rows, "baseline_terminal_stage_progress")
            ),
            "selected_terminal_stage_progress": float(
                seed_cluster_mean(rows, "selected_terminal_stage_progress")
            ),
            "delta_terminal_stage_progress": float(
                seed_cluster_mean(
                    [
                        {
                            **row,
                            "stage_delta": (
                                float(row["selected_terminal_stage_progress"])
                                - float(row["baseline_terminal_stage_progress"])
                            ),
                        }
                        for row in rows
                    ],
                    "stage_delta",
                )
            ),
            "oracle_terminal_stage_progress": float(
                seed_cluster_mean(rows, "oracle_terminal_stage_progress")
            ),
            "baseline_terminal_goal_distance": float(
                seed_cluster_mean(rows, "baseline_terminal_goal_distance")
            ),
            "selected_terminal_goal_distance": float(
                seed_cluster_mean(rows, "selected_terminal_goal_distance")
            ),
            "delta_terminal_goal_progress": float(
                seed_cluster_mean(
                    [
                        {
                            **row,
                            "goal_delta": (
                                float(row["selected_terminal_goal_progress"])
                                - float(row["baseline_terminal_goal_progress"])
                            ),
                        }
                        for row in rows
                    ],
                    "goal_delta",
                )
            ),
            "mixed_success_decisions": len(mixed),
            "mixed_success_selection_accuracy": (
                seed_cluster_mean(mixed, "selected_success")
                if mixed
                else None
            ),
            "dense_progress_decisions": len(dense),
            "dense_uninformative_decisions": len(dense_uninformative),
            "dense_progress_selection_accuracy": (
                seed_cluster_mean(dense, "selected_dense_best")
                if dense
                else None
            ),
            "dense_soft_target_probability_selected": (
                float(
                    seed_cluster_mean(
                        dense, "selected_dense_soft_target_probability"
                    )
                )
                if dense
                else None
            ),
            "dense_soft_target_probability_regret": (
                float(
                    seed_cluster_mean(
                        [
                            {
                                **row,
                                "soft_regret": (
                                    float(row["dense_soft_target_peak_probability"])
                                    - float(
                                        row[
                                            "selected_dense_soft_target_probability"
                                        ]
                                    )
                                ),
                            }
                            for row in dense
                        ],
                        "soft_regret",
                    )
                )
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
        clusters: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in selected:
            clusters[
                (
                    str(row["body"]),
                    str(row["condition"]),
                    int(row["requested_seed"]),
                )
            ].append(float(row["correct"]))
        return (
            float(np.mean([np.mean(values) for values in clusters.values()]))
            if clusters
            else None,
            len(selected),
        )

    def weighted_pair_summary(
        rows: Sequence[Mapping[str, Any]], *, body: str | None = None,
        condition: str | None = None,
    ) -> tuple[float | None, int, float]:
        selected = [
            row for row in rows
            if (body is None or row["body"] == body)
            and (condition is None or row["condition"] == condition)
        ]
        total_weight = float(sum(float(row["preference_weight"]) for row in selected))
        clusters: dict[
            tuple[str, str, int], list[Mapping[str, Any]]
        ] = defaultdict(list)
        for row in selected:
            clusters[
                (
                    str(row["body"]),
                    str(row["condition"]),
                    int(row["requested_seed"]),
                )
            ].append(row)
        cluster_accuracies = []
        for cluster_rows in clusters.values():
            cluster_weight = sum(
                float(row["preference_weight"]) for row in cluster_rows
            )
            if cluster_weight > 0.0:
                cluster_accuracies.append(
                    sum(
                        float(row["preference_weight"]) * float(row["correct"])
                        for row in cluster_rows
                    )
                    / cluster_weight
                )
        return (
            (
                float(np.mean(cluster_accuracies))
                if cluster_accuracies
                else None
            ),
            len(selected),
            total_weight,
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
                (
                    dense_soft_pair_accuracy,
                    dense_soft_pair_count,
                    dense_soft_pair_weight,
                ) = weighted_pair_summary(
                    dense_soft_pairs, body=body, condition=condition
                )
                unit.update(
                    {
                        "mixed_success_pairwise_accuracy": mixed_pair_accuracy,
                        "mixed_success_pairwise_comparisons": mixed_pair_count,
                        "dense_progress_pairwise_accuracy": dense_pair_accuracy,
                        "dense_progress_pairwise_comparisons": dense_pair_count,
                        "dense_soft_target_weighted_pairwise_accuracy": (
                            dense_soft_pair_accuracy
                        ),
                        "dense_soft_target_pairwise_comparisons": (
                            dense_soft_pair_count
                        ),
                        "dense_soft_target_pairwise_preference_weight": (
                            dense_soft_pair_weight
                        ),
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
    (
        dense_soft_pair_accuracy,
        dense_soft_pair_count,
        dense_soft_pair_weight,
    ) = weighted_pair_summary(dense_soft_pairs)
    comparative_decisions = [
        row
        for row in decisions
        if bool(row["mixed_success"]) or bool(row["dense_applicable"])
    ]
    comparative_seed_clusters = {
        (str(row["body"]), str(row["condition"]), int(row["requested_seed"]))
        for row in comparative_decisions
    }
    comparative_requested_seeds = {
        int(row["requested_seed"]) for row in comparative_decisions
    }
    comparative_body_condition_units = {
        (str(row["body"]), str(row["condition"]))
        for row in comparative_decisions
    }
    comparative_bodies = {str(row["body"]) for row in comparative_decisions}
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
        "macro_dense_soft_target_probability_selected": macro(
            "dense_soft_target_probability_selected"
        ),
        "macro_dense_soft_target_probability_regret": macro(
            "dense_soft_target_probability_regret"
        ),
        "dense_soft_target_weighted_pairwise_accuracy": (
            dense_soft_pair_accuracy
        ),
        "dense_soft_target_pairwise_comparisons": dense_soft_pair_count,
        "dense_soft_target_pairwise_preference_weight": dense_soft_pair_weight,
        "macro_dense_soft_target_weighted_pairwise_accuracy": macro(
            "dense_soft_target_weighted_pairwise_accuracy"
        ),
        "comparative_validation_seed_clusters": len(comparative_seed_clusters),
        "comparative_validation_requested_seeds": len(
            comparative_requested_seeds
        ),
        "comparative_validation_body_condition_units": len(
            comparative_body_condition_units
        ),
        "comparative_validation_bodies": len(comparative_bodies),
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
        maximize(ranking.get("macro_dense_soft_target_probability_selected")),
        maximize(
            ranking.get("macro_dense_soft_target_weighted_pairwise_accuracy")
        ),
        float(diagnostic_score),
        int(step),
    )


def checkpoint_selection_evidence_mode(
    ranking: Mapping[str, Any], *, comparative_authorized: bool
) -> str:
    """Describe the evidence that actually selected the deployed checkpoint."""

    if not comparative_authorized:
        return "strict_proper_no_comparative_evidence"
    if int(ranking.get("mixed_success_decisions", 0)) > 0:
        return "mixed_success_then_dense_progress"
    if int(ranking.get("dense_progress_decisions", 0)) > 0:
        return "dense_progress_without_mixed_success"
    raise FiveBodyContractError(
        "comparative checkpoint selection was authorized without comparative rows"
    )


def combine_primary_and_supplement_strict_proper(
    primary: Sequence[Mapping[str, Any]],
    supplement: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Combine independent proper-validation lanes without fitting calibration."""

    if len(primary) != 5 or len(supplement) != 5:
        raise FiveBodyContractError(
            "strict proper evidence requires five aligned ensemble members"
        )
    supplement_enabled = all(item is not None for item in supplement)
    if any(item is not None for item in supplement) != supplement_enabled:
        raise FiveBodyContractError(
            "supplement strict proper evidence is only partially available"
        )

    def value(item: Mapping[str, Any], name: str) -> float:
        result = item.get(name)
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, float))
            or not math.isfinite(float(result))
            or (name == "macro_standard_error" and float(result) < 0.0)
        ):
            raise FiveBodyContractError(f"strict proper evidence has invalid {name}")
        return float(result)

    primary_scores = [value(item, "macro_score") for item in primary]
    primary_standard_errors = [
        value(item, "macro_standard_error") for item in primary
    ]
    supplement_scores = (
        [value(item, "macro_score") for item in supplement if item is not None]
        if supplement_enabled
        else []
    )
    supplement_standard_errors = (
        [
            value(item, "macro_standard_error")
            for item in supplement
            if item is not None
        ]
        if supplement_enabled
        else []
    )
    combined_member_scores = [
        primary_score
        + (
            SUPPLEMENT_PROPER_LOSS_WEIGHT * supplement_scores[index]
            if supplement_enabled
            else 0.0
        )
        for index, primary_score in enumerate(primary_scores)
    ]
    combined_member_standard_errors = [
        math.sqrt(
            primary_standard_error**2
            + (
                SUPPLEMENT_PROPER_LOSS_WEIGHT
                * supplement_standard_errors[index]
            )
            ** 2
        )
        if supplement_enabled
        else primary_standard_error
        for index, primary_standard_error in enumerate(primary_standard_errors)
    ]
    return {
        "mean_member_primary_strict_proper_score": float(np.mean(primary_scores)),
        "mean_member_supplement_strict_proper_score": (
            float(np.mean(supplement_scores)) if supplement_enabled else None
        ),
        "supplement_strict_proper_weight": (
            SUPPLEMENT_PROPER_LOSS_WEIGHT if supplement_enabled else 0.0
        ),
        "mean_member_strict_proper_score": float(
            np.mean(combined_member_scores)
        ),
        "primary_conservative_strict_proper_standard_error": max(
            primary_standard_errors
        ),
        "supplement_conservative_strict_proper_standard_error": (
            max(supplement_standard_errors) if supplement_enabled else None
        ),
        "conservative_strict_proper_standard_error": max(
            combined_member_standard_errors
        ),
        "strict_proper_standard_error_combination": (
            "per_member_sqrt_primary_variance_plus_0.25_squared_times_"
            "supplement_variance_then_conservative_max_across_members"
            if supplement_enabled
            else "primary_only_conservative_max_across_members"
        ),
    }


def select_calibration_guarded_checkpoint(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Select rank performance only inside the proper one-SE confidence set."""

    if not records:
        raise FiveBodyContractError("checkpoint selection has no evaluated steps")
    normalized = []
    observed_steps: set[int] = set()
    comparative_support: tuple[int, int, int, int, int, int] | None = None
    for record in records:
        raw_score = record.get("mean_member_strict_proper_score")
        raw_standard_error = record.get(
            "conservative_strict_proper_standard_error"
        )
        step = record.get("step")
        key = record.get("selection_key")
        ranking = record.get("ensemble_candidate_ranking")

        def finite_numeric(value: Any) -> bool:
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and math.isfinite(float(value))
            )

        score = float(raw_score) if finite_numeric(raw_score) else math.nan
        standard_error = (
            float(raw_standard_error)
            if finite_numeric(raw_standard_error)
            else math.nan
        )
        finite_key = (
            tuple(float(value) for value in key)
            if isinstance(key, Sequence)
            and not isinstance(key, (str, bytes))
            and all(finite_numeric(value) for value in key)
            else ()
        )
        mixed_support = (
            ranking.get("mixed_success_decisions")
            if isinstance(ranking, Mapping)
            else None
        )
        dense_support = (
            ranking.get("dense_progress_decisions")
            if isinstance(ranking, Mapping)
            else None
        )
        seed_cluster_support = (
            ranking.get("comparative_validation_seed_clusters")
            if isinstance(ranking, Mapping)
            else None
        )
        requested_seed_support = (
            ranking.get("comparative_validation_requested_seeds")
            if isinstance(ranking, Mapping)
            else None
        )
        body_condition_support = (
            ranking.get("comparative_validation_body_condition_units")
            if isinstance(ranking, Mapping)
            else None
        )
        body_support = (
            ranking.get("comparative_validation_bodies")
            if isinstance(ranking, Mapping)
            else None
        )
        if (
            not math.isfinite(score)
            or not math.isfinite(standard_error)
            or standard_error < 0.0
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or not isinstance(key, Sequence)
            or isinstance(key, (str, bytes))
            or len(key) != 7
            or not isinstance(ranking, Mapping)
            or len(finite_key) != 7
            or not all(math.isfinite(value) for value in finite_key)
            or isinstance(key[-1], bool)
            or finite_key[-1] != float(step)
            or isinstance(mixed_support, bool)
            or not isinstance(mixed_support, int)
            or mixed_support < 0
            or isinstance(dense_support, bool)
            or not isinstance(dense_support, int)
            or dense_support < 0
            or isinstance(seed_cluster_support, bool)
            or not isinstance(seed_cluster_support, int)
            or seed_cluster_support < 0
            or isinstance(requested_seed_support, bool)
            or not isinstance(requested_seed_support, int)
            or requested_seed_support < 0
            or seed_cluster_support > mixed_support + dense_support
            or (seed_cluster_support == 0) != (requested_seed_support == 0)
            or isinstance(body_condition_support, bool)
            or not isinstance(body_condition_support, int)
            or body_condition_support < 0
            or body_condition_support > len(BODIES) * len(CONDITIONS)
            or isinstance(body_support, bool)
            or not isinstance(body_support, int)
            or body_support < 0
            or body_support > len(BODIES)
            or body_condition_support > body_support * len(CONDITIONS)
            or (seed_cluster_support == 0) != (body_condition_support == 0)
            or (seed_cluster_support == 0) != (body_support == 0)
        ):
            raise FiveBodyContractError(
                "checkpoint selection record violates the proper/rank contract"
            )
        if step in observed_steps:
            raise FiveBodyContractError("checkpoint selection steps are not unique")
        observed_steps.add(step)
        support = (
            mixed_support,
            dense_support,
            seed_cluster_support,
            requested_seed_support,
            body_condition_support,
            body_support,
        )
        if comparative_support is None:
            comparative_support = support
        elif support != comparative_support:
            raise FiveBodyContractError(
                "checkpoint comparative validation support changed across steps"
            )
        normalized.append(
            (record, score, standard_error, step, (*finite_key[:6], step))
        )
    proper_best, best_score, best_standard_error, _best_step, _best_key = min(
        normalized,
        key=lambda item: (item[1], item[3]),
    )
    threshold = best_score + best_standard_error
    comparative_decisions = (
        int(comparative_support[0] + comparative_support[1])
        if comparative_support is not None
        else 0
    )
    comparative_seed_clusters = (
        int(comparative_support[2]) if comparative_support is not None else 0
    )
    comparative_requested_seeds = (
        int(comparative_support[3]) if comparative_support is not None else 0
    )
    comparative_body_condition_units = (
        int(comparative_support[4]) if comparative_support is not None else 0
    )
    comparative_bodies = (
        int(comparative_support[5]) if comparative_support is not None else 0
    )
    comparative = (
        comparative_seed_clusters
        >= MINIMUM_COMPARATIVE_VALIDATION_SEED_CLUSTERS
        and comparative_requested_seeds
        >= MINIMUM_COMPARATIVE_VALIDATION_REQUESTED_SEEDS
        and comparative_body_condition_units
        >= MINIMUM_COMPARATIVE_VALIDATION_BODY_CONDITION_UNITS
        and comparative_bodies >= MINIMUM_COMPARATIVE_VALIDATION_BODIES
    )
    if comparative:
        eligible = [item for item in normalized if item[1] <= threshold + 1e-12]
        selected = min(eligible, key=lambda item: item[4])[0]
    else:
        eligible = [item for item in normalized if item[0] is proper_best]
        selected = proper_best
    return selected, {
        "rule": (
            "minimize_source_body_condition_macro_proper_score_then_"
            "maximize_rank_within_one_standard_error"
        ),
        "comparative_validation_evidence": comparative,
        "comparative_validation_decisions": comparative_decisions,
        "comparative_validation_seed_clusters": comparative_seed_clusters,
        "comparative_validation_requested_seeds": comparative_requested_seeds,
        "comparative_validation_body_condition_units": (
            comparative_body_condition_units
        ),
        "comparative_validation_bodies": comparative_bodies,
        "minimum_comparative_validation_seed_clusters": (
            MINIMUM_COMPARATIVE_VALIDATION_SEED_CLUSTERS
        ),
        "minimum_comparative_validation_requested_seeds": (
            MINIMUM_COMPARATIVE_VALIDATION_REQUESTED_SEEDS
        ),
        "minimum_comparative_validation_body_condition_units": (
            MINIMUM_COMPARATIVE_VALIDATION_BODY_CONDITION_UNITS
        ),
        "minimum_comparative_validation_bodies": (
            MINIMUM_COMPARATIVE_VALIDATION_BODIES
        ),
        "best_score": best_score,
        "conservative_one_standard_error": best_standard_error,
        "eligible_threshold": threshold,
        "eligible_steps": [int(item[3]) for item in eligible],
        "selected_step": int(selected["step"]),
        "selected_score": float(selected["mean_member_strict_proper_score"]),
        "heldout_rows_used": 0,
    }


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


def materialize_supplement_rows(
    groups: Sequence[Mapping[str, Any]], *, held_out_body: str
) -> list[dict[str, Any]]:
    """Open only source-body supplements and bind each row to its declared root."""

    if any(group.get("body") == held_out_body for group in groups):
        raise FiveBodyContractError(
            "held-out supplement group reached source payload loader"
        )
    rows: list[dict[str, Any]] = []
    for group in groups:
        body = str(group.get("body", ""))
        if body not in BODIES or body == held_out_body:
            raise FiveBodyContractError("supplement source body is invalid")
        if group.get("source_role") != "proper_world_supplement":
            raise FiveBodyContractError("supplement source role changed")
        root_event = int(group["root_event_id"])
        declared_horizon = int(group["pre_registered_horizon"])
        loaded = _npz_rows(group, body=body)
        if any(int(row["current_event_id"]) != root_event for row in loaded):
            raise FiveBodyContractError(
                "supplement payload current event differs from its e12/e3/e4 manifest"
            )
        if any(
            not math.isclose(
                float(row["remaining_action_budget"]),
                float(declared_horizon),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for row in loaded
        ):
            raise FiveBodyContractError(
                "supplement payload remaining budget differs from its bound horizon"
            )
        namespace = (
            f"{body}|{group['condition']}|proper-world-supplement|"
            f"{group['group_id']}"
        )
        for row in loaded:
            row["logical_group"] = namespace
        rows.extend(loaded)
    if any(row["body"] == held_out_body for row in rows):
        raise FiveBodyContractError("held-out supplement row reached source fitting")
    return rows


def standardized_input_clip_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    clip: float = CROSS_BODY_STANDARDIZED_INPUT_CLIP,
) -> dict[str, Any]:
    """Audit cross-body clipping without using held-out rows for fitting."""

    if not rows:
        raise FiveBodyContractError("input clip diagnostics require source rows")
    state_mean = np.asarray(state_mean, dtype=np.float64)
    state_std = np.asarray(state_std, dtype=np.float64)
    action_mean = np.asarray(action_mean, dtype=np.float64).reshape(-1)
    action_std = np.asarray(action_std, dtype=np.float64).reshape(-1)
    if (
        state_mean.shape != (core.STATE_DIM,)
        or state_std.shape != (core.STATE_DIM,)
        or action_mean.shape != (core.ACTION_DIM,)
        or action_std.shape != (core.ACTION_DIM,)
        or not math.isfinite(float(clip))
        or clip <= 0.0
        or np.any(state_std <= 0.0)
        or np.any(action_std <= 0.0)
    ):
        raise FiveBodyContractError("input clip diagnostic normalization changed")

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        states = np.stack(
            [np.asarray(row["state"], dtype=np.float64) for row in selected]
        )
        state_z = (states[:, :18] - state_mean[None, :18]) / state_std[None, :18]
        state_clipped = np.abs(state_z) > clip
        action_total = np.zeros(core.ACTION_DIM, dtype=np.int64)
        action_clipped = np.zeros(core.ACTION_DIM, dtype=np.int64)
        normalized_actions: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = (
            defaultdict(list)
        )
        unavailable_action_rows = 0
        unavailable_action_groups: set[str] = set()
        for row in selected:
            actions = np.asarray(row["actions"], dtype=np.float64)
            mask = np.asarray(row["action_mask"], dtype=bool)
            group = str(row["logical_group"])
            if not bool(row.get("action_available", 1.0)):
                unavailable_action_rows += 1
                unavailable_action_groups.add(group)
                continue
            if (
                actions.ndim != 2
                or actions.shape[-1] != core.ACTION_DIM
                or mask.shape != actions.shape[:1]
                or not mask.any()
            ):
                raise FiveBodyContractError(
                    "input clip diagnostics received an invalid action prefix"
                )
            z = (actions - action_mean[None]) / action_std[None]
            action_total += int(mask.sum())
            action_clipped += (np.abs(z[mask]) > clip).sum(axis=0)
            normalized_actions[group].append(
                (int(row["candidate_index"]), z, mask)
            )
        retention = []
        collisions = 0
        distinct_pairs = 0
        for group, candidates in normalized_actions.items():
            if group in unavailable_action_groups:
                continue
            candidates = sorted(candidates, key=lambda value: value[0])
            if [value[0] for value in candidates] != list(range(CANDIDATE_COUNT)):
                raise FiveBodyContractError(
                    f"input clip diagnostics split decision {group}"
                )
            for left in range(CANDIDATE_COUNT):
                for right in range(left + 1, CANDIDATE_COUNT):
                    common = candidates[left][2] & candidates[right][2]
                    raw_difference = (
                        candidates[left][1][common] - candidates[right][1][common]
                    )
                    clipped_difference = (
                        np.clip(candidates[left][1][common], -clip, clip)
                        - np.clip(candidates[right][1][common], -clip, clip)
                    )
                    raw_distance = float(np.linalg.norm(raw_difference))
                    if raw_distance <= 1e-12:
                        continue
                    clipped_distance = float(np.linalg.norm(clipped_difference))
                    distinct_pairs += 1
                    collisions += int(clipped_distance <= 1e-12)
                    retention.append(clipped_distance / raw_distance)
        return {
            "rows": len(selected),
            "state_continuous_component_count": int(state_z.size),
            "state_continuous_clip_fraction": float(state_clipped.mean()),
            "state_continuous_clip_fraction_by_channel": [
                float(value) for value in state_clipped.mean(axis=0)
            ],
            "action_valid_component_count": int(action_total.sum()),
            "action_unavailable_rows_excluded": unavailable_action_rows,
            "candidate_groups_excluded_for_unavailable_action": len(
                unavailable_action_groups
            ),
            "action_valid_clip_fraction": float(
                action_clipped.sum() / max(int(action_total.sum()), 1)
            ),
            "action_valid_clip_fraction_by_channel": [
                (
                    float(action_clipped[index] / action_total[index])
                    if action_total[index] > 0
                    else None
                )
                for index in range(core.ACTION_DIM)
            ],
            "candidate_standardized_distinct_pair_count": distinct_pairs,
            "candidate_pair_collision_count_after_clip": collisions,
            "candidate_pair_collision_fraction_after_clip": (
                float(collisions / distinct_pairs) if distinct_pairs else 0.0
            ),
            "candidate_pair_distance_retention_quantiles_05_50_95": (
                [
                    float(value)
                    for value in np.quantile(retention, [0.05, 0.5, 0.95])
                ]
                if retention
                else None
            ),
        }

    by_body = {
        body: summarize([row for row in rows if row["body"] == body])
        for body in BODIES
        if any(row["body"] == body for row in rows)
    }
    return {
        "format": "etsf_cross_body_standardized_input_clip_diagnostics_v1",
        "clip_absolute_z": float(clip),
        "normalization_fit_scope": "source_train_only",
        "heldout_rows_used_for_normalization_or_diagnostics": False,
        "overall": summarize(rows),
        "by_body": by_body,
    }


def supplement_group_bootstrap_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    members: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Ordinary member-specific Poisson bootstrap over real supplement groups."""

    if members != 5:
        raise FiveBodyContractError(
            "formal supplement epistemic bootstrap requires five members"
        )
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_group[str(row["logical_group"])].append(index)
    if not indices_by_group:
        raise FiveBodyContractError("cannot bootstrap an empty supplement stream")
    for group, indices in indices_by_group.items():
        candidates = sorted(int(rows[index]["candidate_index"]) for index in indices)
        if candidates != list(range(CANDIDATE_COUNT)):
            raise FiveBodyContractError(
                f"supplement bootstrap received incomplete decision {group}"
            )
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(
            f"{seed}|proper-world-utility-rank-supplement-bootstrap-v2".encode()
        ).digest()[:8],
        "big",
    )
    group_order = [str(row["logical_group"]) for row in rows]
    weights = core.logical_group_bootstrap_weights(
        group_order, members=members, seed=bootstrap_seed
    )
    audit: list[dict[str, Any]] = []
    for member in range(members):
        for group, indices in indices_by_group.items():
            group_values = weights[member, indices]
            if not np.all(group_values == group_values[0]):
                raise FiveBodyContractError(
                    f"supplement bootstrap changed within logical group {group}"
                )
        audit.append(
            {
                "member": member,
                "real_groups_with_nonzero_weight": sum(
                    float(weights[member, indices[0]]) > 0.0
                    for indices in indices_by_group.values()
                ),
                "real_groups_total": len(indices_by_group),
                "class_balancing_used": False,
                "synthetic_groups_or_labels": 0,
                "bootstrap_seed": bootstrap_seed,
            }
        )
    return weights.astype(np.float32, copy=False), audit, bootstrap_seed


def candidate_rank_supervision_inventory(
    rows: Sequence[Mapping[str, Any]], *, ablation_variant: str = "full"
) -> dict[str, int]:
    """Count only real complete decisions that can identify listwise utility."""

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(
            f"unknown ablation variant {ablation_variant!r}"
        )
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_group[str(row["logical_group"])].append(index)
    mixed = 0
    dense = 0
    for group, indices in indices_by_group.items():
        if sorted(int(rows[index]["candidate_index"]) for index in indices) != list(
            range(CANDIDATE_COUNT)
        ):
            raise FiveBodyContractError(
                f"rank inventory received incomplete decision {group}"
            )
        if not all(bool(rows[index]["success_mask"]) for index in indices):
            raise FiveBodyContractError(
                f"rank inventory lacks complete success supervision {group}"
            )
        outcomes = {float(rows[index]["success"]) for index in indices}
        if outcomes == {0.0, 1.0}:
            mixed += 1
        elif (
            outcomes == {0.0}
            and all(
                bool(rows[index].get("terminal_event_mask", 0.0))
                and bool(rows[index].get("terminal_goal_progress_mask", 0.0))
                for index in indices
            )
            and _dense_rank_labels_are_orderable(
                [rows[index]["terminal_max_event_id"] for index in indices],
                [rows[index]["terminal_goal_progress"] for index in indices],
                ablation_variant=ablation_variant,
            )
        ):
            dense += 1
    if ablation_variant == "success_only":
        mixed = 0
        dense = 0
    return {
        "mixed_success_groups": mixed,
        "informative_dense_groups": dense,
        "rank_supervision_groups": mixed + dense,
    }


def effect_preserving_group_bootstrap_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    members: int,
    seed: int,
    ablation_variant: str = "full",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep any available comparative rank supervision in every member."""

    if ablation_variant not in ABLATION_VARIANTS:
        raise FiveBodyContractError(
            f"unknown ablation variant {ablation_variant!r}"
        )

    group_order = [str(row["logical_group"]) for row in rows]
    weights = core.logical_group_bootstrap_weights(
        group_order, members=members, seed=seed
    )
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_order):
        indices_by_group[group].append(index)
    mixed_groups: list[str] = []
    informative_dense_groups: list[str] = []
    for group, indices in sorted(indices_by_group.items()):
        outcomes = {
            float(rows[index]["success"])
            for index in indices
            if bool(rows[index]["success_mask"])
        }
        if outcomes == {0.0, 1.0}:
            mixed_groups.append(group)
        elif (
            outcomes == {0.0}
            and all(
                bool(rows[index].get("terminal_event_mask", 0.0))
                and bool(rows[index].get("terminal_goal_progress_mask", 0.0))
                for index in indices
            )
            and _dense_rank_labels_are_orderable(
                [rows[index]["terminal_max_event_id"] for index in indices],
                [rows[index]["terminal_goal_progress"] for index in indices],
                ablation_variant=ablation_variant,
            )
        ):
            informative_dense_groups.append(group)
    # Every epistemic member sees every rare success-changing comparison.  The
    # Poisson component still changes its relative influence across members.
    # Dense decisions retain ordinary Poisson weights unless a dense-only
    # member would otherwise lose every real comparative group.
    if ablation_variant != "success_only":
        for group in mixed_groups:
            weights[:, indices_by_group[group]] += 1.0
    rank_groups = (
        []
        if ablation_variant == "success_only"
        else mixed_groups + informative_dense_groups
    )
    audit: list[dict[str, Any]] = []
    for member in range(members):
        repaired_rank_groups: list[str] = []

        def active(group: str) -> bool:
            return float(weights[member, indices_by_group[group][0]]) > 0.0

        if rank_groups and not any(active(group) for group in rank_groups):
            selector = int.from_bytes(
                hashlib.sha256(
                    f"{seed}|rank-support|{member}".encode()
                ).digest()[:8],
                "big",
            )
            repaired = rank_groups[selector % len(rank_groups)]
            weights[member, indices_by_group[repaired]] = 1.0
            repaired_rank_groups.append(repaired)
        active_mixed = [
            group
            for group in mixed_groups
            if active(group)
        ]
        active_dense = [
            group
            for group in informative_dense_groups
            if active(group)
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
        if (
            ablation_variant != "success_only"
            and mixed_groups
            and not active_mixed_count
        ):
            raise FiveBodyContractError(
                "effect-preserving bootstrap lost mixed-success supervision"
            )
        if rank_groups and not (active_mixed or active_dense):
            raise FiveBodyContractError(
                "effect-preserving bootstrap lost all comparative supervision"
            )
        audit.append(
            {
                "member": member,
                "positive_rows_with_nonzero_weight": int(positives),
                "negative_rows_with_nonzero_weight": int(negatives),
                "mixed_success_groups_with_nonzero_weight": int(active_mixed_count),
                "mixed_success_groups_total": len(mixed_groups),
                "informative_dense_groups_with_nonzero_weight": len(active_dense),
                "informative_dense_groups_total": len(informative_dense_groups),
                "rank_supervision_groups_with_nonzero_weight": (
                    len(active_mixed) + len(active_dense)
                    if ablation_variant != "success_only"
                    else 0
                ),
                "rank_supervision_groups_total": len(rank_groups),
                "mixed_weight_minimum": (
                    1 if ablation_variant != "success_only" and mixed_groups else 0
                ),
                "deterministic_mixed_group_repairs": 0,
                "deterministic_rank_group_repairs": len(repaired_rank_groups),
                "repaired_rank_groups": repaired_rank_groups,
            }
        )
    return weights.astype(np.float32, copy=False), audit


def proper_outcome_preserving_group_bootstrap_weights(
    rows: Sequence[Mapping[str, Any]], *, members: int, seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Retain every outcome class that really exists without synthesizing one.

    Ordinary logical-group Poisson weights are unchanged unless a member loses
    every group from an observed success class.  A complete real group from
    only that observed class is then restored at unit weight.  Single-class
    source data remain single-class: no positive label, negative label, mixed
    decision, class weight, or pseudo-outcome is ever fabricated.
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
    all_groups: list[str] = []
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
        all_groups.append(group)
        if 1.0 in outcomes:
            positive_groups.append(group)
        if 0.0 in outcomes:
            negative_groups.append(group)
        if outcomes == {0.0, 1.0}:
            mixed_groups.append(group)
    if not all_groups:
        raise FiveBodyContractError(
            "proper outcome-preserving bootstrap requires a supervised group"
        )

    audit: list[dict[str, Any]] = []
    for member in range(members):
        repaired_groups: list[str] = []

        def active(group: str) -> bool:
            return float(weights[member, indices_by_group[group][0]]) > 0.0

        def restore_one(candidates: Sequence[str], purpose: str) -> None:
            selector = int.from_bytes(
                hashlib.sha256(
                    f"{seed}|proper-outcome|{purpose}|{member}".encode()
                ).digest()[:8],
                "big",
            )
            repaired = candidates[selector % len(candidates)]
            selected = indices_by_group[repaired]
            weights[member, selected] = np.maximum(
                weights[member, selected], np.float32(1.0)
            )
            if repaired not in repaired_groups:
                repaired_groups.append(repaired)

        if positive_groups and not any(active(group) for group in positive_groups):
            restore_one(positive_groups, "positive")
        if negative_groups and not any(active(group) for group in negative_groups):
            restore_one(negative_groups, "negative")
        if not any(active(group) for group in all_groups):
            restore_one(all_groups, "any")

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
            (positive_groups and (not positive_rows or not active_positive_groups))
            or (negative_groups and (not negative_rows or not active_negative_groups))
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
                "positive_class_present": bool(positive_groups),
                "negative_class_present": bool(negative_groups),
                "deterministic_outcome_group_repairs": len(repaired_groups),
                "deterministic_mixed_group_repairs": sum(
                    group in mixed_groups for group in repaired_groups
                ),
                "success_labels_synthesized": 0,
                "repaired_groups": repaired_groups,
            }
        )
    return weights.astype(np.float32, copy=False), audit


def _train_fold(
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    supplement_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
        audit,
        held_out_body=args.held_out_body,
        split_seed=args.split_seed,
        supplement_audit=supplement_audit,
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
    if (
        not len(successes)
        or not set(np.unique(successes).tolist()) <= {0.0, 1.0}
    ):
        raise FiveBodyContractError(
            "source training requires real binary outcome supervision"
        )
    outcome_by_group: dict[str, set[float]] = defaultdict(set)
    for row in train_rows:
        if bool(row["success_mask"]):
            outcome_by_group[str(row["logical_group"])].add(float(row["success"]))
    mixed_outcome_decisions = sum(len(values) > 1 for values in outcome_by_group.values())
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
    if supplement_audit is None:
        supplement_groups: list[dict[str, Any]] = []
        supplement_validation_groups: list[dict[str, Any]] = []
        supplement_rows: list[dict[str, Any]] = []
        supplement_validation_rows: list[dict[str, Any]] = []
    else:
        (
            supplement_groups,
            supplement_validation_groups,
            _supplement_heldout_groups,
        ) = (
            supplement_source_train_split(
                supplement_audit,
                held_out_body=args.held_out_body,
                split_seed=args.split_seed,
            )
        )
        supplement_rows = materialize_supplement_rows(
            supplement_groups, held_out_body=args.held_out_body
        )
        supplement_validation_rows = materialize_supplement_rows(
            supplement_validation_groups, held_out_body=args.held_out_body
        )
    input_clip_diagnostics = {
        "source_train": standardized_input_clip_diagnostics(
            train_rows,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
        ),
        "source_validation": standardized_input_clip_diagnostics(
            validation_rows,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
        ),
        "proper_world_supplement_source_train": (
            standardized_input_clip_diagnostics(
                supplement_rows,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
            )
            if supplement_rows
            else None
        ),
        "proper_world_supplement_source_validation": (
            standardized_input_clip_diagnostics(
                supplement_validation_rows,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
            )
            if supplement_validation_rows
            else None
        ),
    }
    supplement_rank_inventory = candidate_rank_supervision_inventory(
        supplement_rows,
        ablation_variant=args.ablation_variant,
    )
    body_to_id = {body: 0 for body in preflight["source_bodies"]}
    train_dataset = core.TransitionDataset(train_rows, body_to_id)
    supplement_dataset = (
        core.TransitionDataset(supplement_rows, body_to_id)
        if supplement_rows
        else None
    )
    validation_loader = DataLoader(
        core.TransitionDataset(validation_rows, body_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    supplement_validation_loader = (
        DataLoader(
            core.TransitionDataset(supplement_validation_rows, body_to_id),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=core.collate_rows,
        )
        if supplement_validation_rows
        else None
    )
    device = torch.device(args.device)
    group_order = [str(row["logical_group"]) for row in train_rows]
    supplement_group_order = [
        str(row["logical_group"]) for row in supplement_rows
    ]
    if supplement_group_order:
        (
            supplement_bootstrap,
            supplement_bootstrap_support,
            supplement_bootstrap_seed,
        ) = supplement_group_bootstrap_weights(
            supplement_rows, members=5, seed=args.split_seed
        )
        supplement_indices_by_group: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(supplement_group_order):
            supplement_indices_by_group[group].append(index)
        supplement_group_weight = {
            group: supplement_bootstrap[:, indices[0]].tolist()
            for group, indices in supplement_indices_by_group.items()
        }
    else:
        supplement_bootstrap_seed = None
        supplement_group_weight = {}
        supplement_bootstrap_support = []
    proper_bootstrap, proper_bootstrap_support = (
        proper_outcome_preserving_group_bootstrap_weights(
            train_rows, members=5, seed=args.split_seed
        )
    )
    rank_bootstrap, bootstrap_support = effect_preserving_group_bootstrap_weights(
        train_rows,
        members=5,
        seed=args.split_seed,
        ablation_variant=args.ablation_variant,
    )
    rank_supervision_groups = int(
        bootstrap_support[0]["rank_supervision_groups_total"]
    )
    if any(
        int(item["rank_supervision_groups_total"]) != rank_supervision_groups
        for item in bootstrap_support
    ):
        raise FiveBodyContractError("rank supervision inventory changed by member")
    formal_rank_supervision_available = rank_supervision_groups > 0
    supplement_rank_supervision_available = (
        supplement_rank_inventory["rank_supervision_groups"] > 0
    )
    rank_supervision_available = (
        formal_rank_supervision_available or supplement_rank_supervision_available
    )
    if args.ablation_variant != "success_only" and not rank_supervision_available:
        raise FiveBodyContractError(
            "candidate-rank utility has no real mixed-success or informative "
            "dense comparison in the formal or source-only supplement stream"
        )
    mixed_rank_groups = int(bootstrap_support[0]["mixed_success_groups_total"])
    informative_dense_groups = int(
        bootstrap_support[0]["informative_dense_groups_total"]
    )
    total_mixed_rank_groups = (
        mixed_rank_groups + supplement_rank_inventory["mixed_success_groups"]
    )
    total_informative_dense_groups = (
        informative_dense_groups
        + supplement_rank_inventory["informative_dense_groups"]
    )
    total_rank_supervision_groups = (
        rank_supervision_groups
        + supplement_rank_inventory["rank_supervision_groups"]
    )
    if args.ablation_variant == "success_only":
        rank_supervision_mode = "proper_coherent_terminal_success_only"
    elif total_mixed_rank_groups and total_informative_dense_groups:
        rank_supervision_mode = "mixed_success_plus_informative_dense"
    elif total_mixed_rank_groups:
        rank_supervision_mode = "mixed_success_only"
    else:
        rank_supervision_mode = "informative_dense_only"
    proper_group_weight = {
        group: proper_bootstrap[:, index].tolist()
        for index, group in enumerate(group_order)
    }
    rank_group_weight = {
        group: rank_bootstrap[:, index].tolist()
        for index, group in enumerate(group_order)
    }
    positive_count = int((successes > 0.5).sum())
    source_negative_to_positive_ratio = (
        float((successes <= 0.5).sum() / positive_count)
        if positive_count
        else None
    )
    observed_success_classes = sorted(
        float(value) for value in np.unique(successes).tolist()
    )
    if observed_success_classes == [0.0, 1.0]:
        success_probability_identifiability = "binary_observed"
    elif observed_success_classes == [0.0]:
        success_probability_identifiability = (
            "negative_only_positive_class_unidentified"
        )
    else:
        success_probability_identifiability = (
            "positive_only_negative_class_unidentified"
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
        utility_parameters = [
            parameter
            for parameter in model.candidate_rank.parameters()
            if parameter.requires_grad
        ]
        utility_parameter_ids = {id(parameter) for parameter in utility_parameters}
        world_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in utility_parameter_ids
        ]
        if not utility_parameters or not world_parameters:
            raise FiveBodyContractError(
                "world and monotone utility parameter partitions must both be nonempty"
            )
        world_parameter_ids = {id(parameter) for parameter in world_parameters}
        trainable_parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        if (
            utility_parameter_ids & world_parameter_ids
            or utility_parameter_ids | world_parameter_ids
            != trainable_parameter_ids
        ):
            raise FiveBodyContractError(
                "world and monotone utility parameter partition changed"
            )
        proper_loader = DataLoader(
            train_dataset,
            batch_sampler=CompleteDecisionBatchSampler(
                train_rows, batch_size=args.batch_size, seed=seed
            ),
            collate_fn=core.collate_rows,
        )
        supplement_loader = (
            DataLoader(
                supplement_dataset,
                batch_sampler=CompleteDecisionBatchSampler(
                    supplement_rows,
                    batch_size=args.batch_size,
                    seed=seed,
                ),
                collate_fn=core.collate_rows,
            )
            if supplement_dataset is not None
            else None
        )
        rank_weight_for_member = {
            group: float(weights[member]) for group, weights in rank_group_weight.items()
        }
        rank_loader = DataLoader(
            train_dataset,
            batch_sampler=(
                MacroBalancedRankDecisionBatchSampler(
                    train_rows,
                    batch_size=args.batch_size,
                    seed=seed,
                    positive_group_weight=rank_weight_for_member,
                    ablation_variant=args.ablation_variant,
                )
                if formal_rank_supervision_available
                else CompleteDecisionBatchSampler(
                    train_rows, batch_size=args.batch_size, seed=seed
                )
            ),
            collate_fn=core.collate_rows,
        )
        proper_iterator = iter(proper_loader)
        supplement_iterator = (
            iter(supplement_loader) if supplement_loader is not None else None
        )
        rank_iterator = iter(rank_loader)
        eval_records: dict[int, dict[str, Any]] = {}
        snapshot_paths: dict[int, Path] = {}
        for step in range(1, args.steps + 1):
            try:
                proper_raw = next(proper_iterator)
            except StopIteration:
                proper_iterator = iter(proper_loader)
                proper_raw = next(proper_iterator)
            supplement_raw = None
            if supplement_loader is not None:
                assert supplement_iterator is not None
                try:
                    supplement_raw = next(supplement_iterator)
                except StopIteration:
                    supplement_iterator = iter(supplement_loader)
                    supplement_raw = next(supplement_iterator)
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
            multitask_loss, pieces = _compute_shared_multitask_loss(
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
            if supplement_raw is None:
                supplement_loss = multitask_loss.new_zeros(())
                supplement_rank_loss = multitask_loss.new_zeros(())
                supplement_pieces = {
                    "supplement_proper_unweighted": supplement_loss,
                    "supplement_proper_weighted": supplement_loss,
                    "supplement_proper_fixed_lambda": supplement_loss,
                }
                supplement_rank_pieces = {
                    "supplement_candidate_rank_unweighted": supplement_rank_loss,
                    "supplement_candidate_rank_weighted": supplement_rank_loss,
                    "supplement_candidate_rank_fixed_lambda": supplement_rank_loss,
                }
            else:
                supplement_batch = core._move_batch(supplement_raw, device)
                supplement_weights = torch.tensor(
                    [
                        supplement_group_weight[group][member]
                        for group in supplement_raw["logical_group"]
                    ],
                    device=device,
                )
                supplement_prediction = model(supplement_batch)
                supplement_loss, supplement_pieces = (
                    _supplement_proper_world_model_loss(
                        supplement_prediction,
                        supplement_batch,
                        supplement_weights,
                        loss_weights=base_loss_weights,
                        ablation_variant=args.ablation_variant,
                    )
                )
                supplement_rank_loss, supplement_rank_pieces = (
                    _supplement_candidate_rank_loss(
                        supplement_prediction,
                        supplement_batch,
                        supplement_weights,
                        ablation_variant=args.ablation_variant,
                    )
                )
            rank_prediction = model(rank_batch)
            decision_loss, decision_pieces = _candidate_rank_loss(
                rank_prediction,
                rank_batch,
                rank_weights,
                ablation_variant=args.ablation_variant,
            )
            _semantic_comparative_raw, semantic_comparative_pieces = (
                _semantic_comparative_loss(
                    rank_prediction,
                    rank_batch,
                    rank_weights,
                    ablation_variant=args.ablation_variant,
                )
            )
            if args.ablation_variant == "success_only":
                semantic_comparative_scale = multitask_loss.new_zeros(())
                semantic_comparative_loss = multitask_loss.new_zeros(())
            else:
                proper_world_reference = (
                    multitask_loss
                    + object_effect_loss
                    + terminal_loss
                    + supplement_loss
                )
                (
                    semantic_comparative_loss,
                    semantic_comparative_scale,
                ) = _bounded_semantic_comparative_loss(
                    proper_world_reference,
                    _semantic_comparative_raw,
                    model,
                )
            semantic_comparative_pieces = {
                **semantic_comparative_pieces,
                "semantic_comparative_active_union_gradient_scale": (
                    semantic_comparative_scale
                ),
                "semantic_comparative_gradient_bounded": (
                    semantic_comparative_loss
                ),
            }
            loss = (
                multitask_loss
                + object_effect_loss
                + terminal_loss
                + supplement_loss
                + supplement_rank_loss
                + decision_loss
                + semantic_comparative_loss
            )
            if not torch.isfinite(loss):
                raise FiveBodyContractError("non-finite shared-head training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_parameters, 2.0)
            torch.nn.utils.clip_grad_norm_(utility_parameters, 2.0)
            optimizer.step()
            if step % args.eval_every and step != args.steps:
                continue
            metrics = core.evaluate_validation_model(model, validation_loader, device)
            terminal_consequences = evaluate_terminal_consequences(
                model,
                validation_loader,
                device,
                ablation_variant=args.ablation_variant,
            )
            metrics["terminal_consequences"] = terminal_consequences
            supplement_validation_consequences = (
                evaluate_terminal_consequences(
                    model,
                    supplement_validation_loader,
                    device,
                    ablation_variant=args.ablation_variant,
                )
                if supplement_validation_loader is not None
                else None
            )
            metrics["supplement_proper_validation"] = (
                supplement_validation_consequences
            )
            if args.ablation_variant not in {"success_only", "no_object_effect"}:
                metrics["legacy_gaussian_object_nll_not_model_distribution"] = (
                    metrics["object_nll"]
                )
                metrics["object_nll"] = terminal_consequences[
                    "object_transition"
                ]["student_t3_nll_without_additive_normalizer"]
                metrics["object_nll_distribution"] = (
                    "observed_post_event_conditional_independent_student_t_dof_3"
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
            success_brier_used_for_selection = bool(
                "success_brier_ratio" in selection_components
            )
            if (
                success_probability_identifiability != "binary_observed"
                and args.ablation_variant != "success_only"
            ):
                selection_components.pop("success_brier_ratio", None)
                success_brier_used_for_selection = False
            if not selection_components:
                raise FiveBodyContractError(
                    "checkpoint selection has no identifiable proper component"
                )
            diagnostic_score = float(np.mean(list(selection_components.values())))
            metrics["diagnostic_multitask_score"] = diagnostic_score
            metrics["diagnostic_multitask_components"] = components
            metrics["checkpoint_selection_diagnostic_components"] = selection_components
            metrics["success_brier_used_for_checkpoint_selection"] = (
                success_brier_used_for_selection
            )
            metrics["train_objective_last"] = {
                "total": float(loss.detach()),
                **{name: float(value.detach()) for name, value in pieces.items() if name != "total"},
                **{name: float(value.detach()) for name, value in object_pieces.items()},
                **{name: float(value.detach()) for name, value in terminal_pieces.items()},
                **{
                    name: float(value.detach())
                    for name, value in supplement_pieces.items()
                },
                **{
                    name: float(value.detach())
                    for name, value in supplement_rank_pieces.items()
                },
                **{name: float(value.detach()) for name, value in decision_pieces.items()},
                **{
                    name: float(value.detach())
                    for name, value in semantic_comparative_pieces.items()
                },
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
    ensemble_model = RiskAdjustedRankEnsemble(
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
        member_strict_proper = [
            member_eval_records[member][step]["terminal_consequences"][
                "strict_proper"
            ]
            for member in range(5)
        ]
        member_supplement_strict_proper = [
            (
                member_eval_records[member][step]["supplement_proper_validation"][
                    "strict_proper"
                ]
                if member_eval_records[member][step][
                    "supplement_proper_validation"
                ]
                is not None
                else None
            )
            for member in range(5)
        ]
        # All members share the same validation decisions, so their sampling
        # errors are correlated across members.  Primary and supplement lanes
        # are independent, so their per-member variances add before the
        # conservative maximum is taken across members.
        combined_strict_proper = combine_primary_and_supplement_strict_proper(
            member_strict_proper, member_supplement_strict_proper
        )
        common_step_selection_audit.append(
            {
                "step": step,
                "selection_key": list(key),
                "ensemble_candidate_ranking": ensemble_ranking,
                "mean_member_diagnostic_multitask_score": diagnostic,
                **combined_strict_proper,
            }
        )
    if not common_step_selection_audit:
        raise FiveBodyContractError(
            "deployment-homomorphic ensemble evaluated no checkpoint"
        )
    selected_record, strict_proper_selection = (
        select_calibration_guarded_checkpoint(common_step_selection_audit)
    )
    validation_has_comparative_rank_evidence = bool(
        strict_proper_selection["comparative_validation_evidence"]
    )
    best_ensemble_step = int(selected_record["step"])
    best_ensemble_ranking = selected_record["ensemble_candidate_ranking"]
    best_ensemble_diagnostic = float(
        selected_record["mean_member_diagnostic_multitask_score"]
    )
    best_ensemble_key = tuple(selected_record["selection_key"])
    selection_evidence_mode = checkpoint_selection_evidence_mode(
        best_ensemble_ranking,
        comparative_authorized=validation_has_comparative_rank_evidence,
    )

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
                "state_action_frame_contract": state_action_frame_contract(),
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
                "proper_world_supplement": {
                    **preflight["supplement"],
                    "source_train_rows": len(supplement_rows),
                    "source_validation_rows": len(supplement_validation_rows),
                    "source_validation_rows_used": len(
                        supplement_validation_rows
                    ),
                    "checkpoint_selection_rows_used": len(
                        supplement_validation_rows
                    ),
                    "rank_selection_rows_used": 0,
                    "calibration_diagnostic_rows_used": len(
                        supplement_validation_rows
                    ),
                    "calibration_rows_used": 0,
                    "calibration_fit": False,
                },
                "rank_supervision_available": rank_supervision_available,
                "rank_supervision_mode": rank_supervision_mode,
                "candidate_rank_parameters_received_direct_supervision": (
                    rank_supervision_available
                ),
                "synthetic_success_labels": 0,
                "selection_evidence_mode": selection_evidence_mode,
                "success_probability_identifiability": (
                    success_probability_identifiability
                ),
                "actor_frozen": True,
                "action_normalization": normalization,
                "state_normalization": state_normalization,
                "input_clip_diagnostics": input_clip_diagnostics,
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
                "supplement_proper_validation": member_validation[
                    "supplement_proper_validation"
                ],
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
        "body_adapter": "single_shared_row_zero_heldout_parameters",
        "canonical_state_schema": CANONICAL_STATE_SCHEMA,
        "canonical_action_schema": CANONICAL_ACTION_SCHEMA,
        "state_action_frame_contract": state_action_frame_contract(),
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
        "observed_success_classes": observed_success_classes,
        "success_probability_identifiability": (
            success_probability_identifiability
        ),
        "source_success_rows": positive_count,
        "source_failure_rows": int((successes <= 0.5).sum()),
        "proper_world_supplement": {
            **preflight["supplement"],
            "source_train_rows": len(supplement_rows),
            "source_validation_rows": len(supplement_validation_rows),
            "loss": (
                "fixed_lambda_proper_world_plus_detached_utility_candidate_rank"
                if supplement_rows
                else "disabled"
            ),
            "rank_or_utility_loss_weight": (
                SUPPLEMENT_RANK_LOSS_WEIGHT if supplement_rows else 0.0
            ),
            "rank_or_utility_rows_used": (
                len(supplement_rows)
                if supplement_rows and args.ablation_variant != "success_only"
                else 0
            ),
            "rank_or_utility_groups_with_real_comparative_supervision": (
                supplement_rank_inventory["rank_supervision_groups"]
            ),
            "rank_supervision_inventory": supplement_rank_inventory,
            "semantic_comparative_rows_used": 0,
            "normalization_rows_used": 0,
            "source_validation_rows_used": len(supplement_validation_rows),
            "checkpoint_selection_rows_used": len(
                supplement_validation_rows
            ),
            "checkpoint_selection_use": (
                "strict_proper_only_primary_plus_fixed_0.25_supplement"
                if supplement_validation_rows
                else "disabled"
            ),
            "rank_selection_rows_used": 0,
            "calibration_diagnostic_rows_used": len(
                supplement_validation_rows
            ),
            "calibration_rows_used": 0,
            "calibration_fit": False,
            "ensemble_logical_group_poisson_bootstrap": (
                supplement_bootstrap_support
            ),
            "ensemble_logical_group_poisson_bootstrap_seed": (
                supplement_bootstrap_seed
            ),
            "class_balancing_used": False,
            "synthetic_groups_or_labels": 0,
        },
        "rank_supervision_available": rank_supervision_available,
        "rank_supervision_groups": total_rank_supervision_groups,
        "formal_rank_supervision_groups": rank_supervision_groups,
        "supplement_rank_supervision_groups": supplement_rank_inventory[
            "rank_supervision_groups"
        ],
        "mixed_success_rank_groups": total_mixed_rank_groups,
        "formal_mixed_success_rank_groups": mixed_rank_groups,
        "supplement_mixed_success_rank_groups": supplement_rank_inventory[
            "mixed_success_groups"
        ],
        "informative_dense_rank_groups": total_informative_dense_groups,
        "formal_informative_dense_rank_groups": informative_dense_groups,
        "supplement_informative_dense_rank_groups": supplement_rank_inventory[
            "informative_dense_groups"
        ],
        "rank_supervision_mode": rank_supervision_mode,
        "candidate_rank_parameters_received_direct_supervision": (
            rank_supervision_available
        ),
        "synthetic_success_labels": 0,
        "selection_evidence_mode": selection_evidence_mode,
        "source_negative_to_positive_ratio": source_negative_to_positive_ratio,
        "ensemble_bootstrap_effect_support": bootstrap_support,
        "ensemble_proper_bootstrap_outcome_support": proper_bootstrap_support,
        "success_probability_training_loss": "unweighted_proper_binary_cross_entropy",
        "terminal_stage_progress_training_loss": (
            "categorical_cross_entropy_plus_weight_0.25_strictly_proper_"
            "ordinal_ranked_probability_score"
        ),
        "checkpoint_selection_primary": {
            "mixed_success_then_dense_progress": (
                "five_member_epistemic_lcb_one_deviation_success_then_dense"
            ),
            "dense_progress_without_mixed_success": (
                "five_member_epistemic_lcb_terminal_event_then_goal_progress"
            ),
            "strict_proper_no_comparative_evidence": (
                (
                    "minimum_primary_plus_fixed_0.25_supplement_macro_strict_"
                    "proper_validation_score"
                )
                if supplement_validation_rows
                else "minimum_primary_macro_strict_proper_validation_score"
            ),
        }[selection_evidence_mode],
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "rank_aggregation": risk_adjusted_rank_ensemble_contract(),
            "strict_proper_score": (
                "primary_strict_proper_plus_fixed_0.25_times_label_blind_"
                "inner_cross_body_supplement_strict_proper"
                if supplement_validation_rows
                else "primary_strict_proper_only"
            ),
            "supplement_validation_never_used_for_rank_comparison": True,
            "calibration_diagnostics_only_no_parameter_fit": True,
            "selected_step": best_ensemble_step,
            "selected_key": list(best_ensemble_key),
            "selected_ensemble_candidate_ranking": best_ensemble_ranking,
            "selected_mean_member_diagnostic_multitask_score": (
                best_ensemble_diagnostic
            ),
            "strict_proper_selection": strict_proper_selection,
            "heldout_rows_used": 0,
            "evaluated_common_steps": common_step_selection_audit,
        },
        "members": members,
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "input_clip_diagnostics": input_clip_diagnostics,
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
    parser.add_argument("--supplement-binding", type=Path)
    parser.add_argument("--supplement-binding-sha256")
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
    validate_ensemble_seeds(args.ensemble_seeds)
    audit = load_binding(args.binding, args.binding_sha256)
    if (args.supplement_binding is None) != (
        args.supplement_binding_sha256 is None
    ):
        raise FiveBodyContractError(
            "supplement binding path and SHA-256 must be supplied together"
        )
    supplement_audit = (
        load_supplement_binding(
            args.supplement_binding,
            args.supplement_binding_sha256,
            primary_audit=audit,
            held_out_body=args.held_out_body,
        )
        if args.supplement_binding is not None
        else None
    )
    if args.mode == "preflight":
        receipt = build_preflight_receipt(
            audit,
            held_out_body=args.held_out_body,
            split_seed=args.split_seed,
            supplement_audit=supplement_audit,
        )
        print("PREFLIGHT=" + json.dumps(receipt, sort_keys=True))
        return
    if args.output is None:
        raise FiveBodyContractError("train-fold requires --output")
    if args.steps <= 0 or args.eval_every <= 0:
        raise FiveBodyContractError("steps/eval-every must be positive")
    print(
        "TRAINING="
        + json.dumps(
            _train_fold(args, audit, supplement_audit=supplement_audit),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ABLATION_VARIANTS",
    "ACTOR_FORMAT", "BINDING_FORMAT", "BODIES", "CANONICAL_ACTION_SCHEMA",
    "SUPPLEMENT_BINDING_FORMAT", "SUPPLEMENT_MANIFEST_FORMAT",
    "SUPPLEMENT_COLLECTOR_FORMAT",
    "SUPPLEMENT_MATERIALIZER_FORMAT",
    "SUPPLEMENT_PROPER_LOSS_WEIGHT", "SUPPLEMENT_RANK_LOSS_WEIGHT",
    "SUPPLEMENT_USAGE_CONTRACT",
    "EXPERT_ROOT_PROVENANCE_CONTRACT",
    "CANONICAL_STATE_SCHEMA", "CANDIDATE_NOISE_CONTRACT",
    "CANDIDATE_RANK_FEATURE_DIM", "CANDIDATE_RANK_FEATURE_SCHEMA",
    "BRANCH_DIAGNOSTIC_CONTRACT",
    "CompleteDecisionBatchSampler", "MacroBalancedRankDecisionBatchSampler",
    "DENSE_FAILURE_RANK_WEIGHT", "DENSE_ONLY_RANK_WEIGHT",
    "DENSE_GOAL_PROGRESS_TEMPERATURE_METERS",
    "DENSE_RANK_LABEL_EQUALITY_TOLERANCE",
    "EPISTEMIC_RANK_RISK_WEIGHT", "GOAL_PROGRESS_NORMALIZATION_METERS",
    "EVENT_PRIORITY_SECONDARY_SCALE", "EVENT_UTILITY_RESIDUAL_BOUND",
    "CONDITIONS", "FORMAT",
    "EffectAlignedSharedEventHead", "FiveBodyContractError",
    "InvariantMonotoneConsequenceUtility", "MANIFEST_FORMAT",
    "MONOTONE_BENEFIT_FEATURES", "MONOTONE_RISK_FEATURES",
    "EVENT_SPEC_SHA256", "MATERIALIZATION_FORMAT", "MODEL_FAMILY",
    "OBJECT_EFFECT_SCHEMA", "EVENT_AGE_CONTRACT", "event_age_contract",
    "STATE_ACTION_FRAME_CONTRACT", "state_action_frame_contract",
    "REQUIRED_ARRAYS",
    "RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT",
    "RiskAdjustedRankEnsemble", "SEMANTIC_COMPARATIVE_GRADIENT_BUDGET",
    "SEMANTIC_GRADIENT_SCALE_CAP",
    "TERMINAL_EVENT_ORDINAL_RPS_LOSS_WEIGHT",
    "TERMINAL_FILM_MODULATION_BOUND",
    "TERMINAL_SUPERVISION_CONTRACT",
    "ablation_contract", "ablation_selection_components",
    "aggregate_risk_adjusted_rank_scores",
    "build_preflight_receipt", "canonical_sha256",
    "combine_primary_and_supplement_strict_proper",
    "candidate_checkpoint_selection_key", "checkpoint_candidate_rank_contract",
    "candidate_rank_supervision_inventory",
    "effect_preserving_group_bootstrap_weights", "load_binding",
    "load_supplement_binding",
    "proper_outcome_preserving_group_bootstrap_weights",
    "evaluate_candidate_ranking", "materialize_source_rows",
    "materialize_supplement_rows", "sha256_file",
    "sha256_tree", "source_group_split", "supplement_inner_validation_body",
    "supplement_source_train_split",
    "supplement_group_bootstrap_weights",
    "summary_candidate_rank_contract",
    "risk_adjusted_rank_ensemble_contract",
    "select_calibration_guarded_checkpoint",
    "validate_actor_authority", "validate_body_manifest",
    "validate_supplement_body_manifest",
    "validate_materialization_receipt",
    "validate_ensemble_seeds",
]
