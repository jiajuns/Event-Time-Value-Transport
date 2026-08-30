from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_multibody_canonical_event_world_model as core  # noqa: E402
import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer_entry  # noqa: E402
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event  # noqa: E402
import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as prereg  # noqa: E402
import run_robotwin2_five_body_lobo_offline_ablation_v1 as ablation  # noqa: E402
import verify_robotwin2_move_can_pot_public_materialization_v1 as verifier  # noqa: E402
from train_robotwin2_five_body_lobo_shared_event_head_v1 import (  # noqa: E402
    _bounded_semantic_comparative_loss,
    _candidate_rank_loss,
    _dense_soft_listwise_loss,
    _relative_gradient_budget_scale,
    _robust_object_effect_loss,
    _semantic_comparative_loss,
    _semantic_comparative_active_parameters,
    _supplement_candidate_rank_loss,
    _supplement_proper_world_model_loss,
    _terminal_consequence_loss,
    _terminal_event_ordinal_rps_rows,
    ABLATION_VARIANTS,
    ACTOR_FORMAT,
    BINDING_FORMAT,
    BODIES,
    CANONICAL_ACTION_SCHEMA,
    CANONICAL_STATE_SCHEMA,
    CANDIDATE_NOISE_CONTRACT,
    DATASET_REPO,
    DATASET_REVISION,
    DEFAULT_INSTRUCTION,
    EVENT_SPEC_SHA256,
    EVENT_AGE_CONTRACT,
    GOAL_PROGRESS_NORMALIZATION_METERS,
    TERMINAL_HORIZON_CONTRACT,
    BRANCH_ROOT_SNAPSHOT_CONTRACT,
    CANDIDATE_RANK_FEATURE_DIM,
    CANDIDATE_RANK_FEATURE_SCHEMA,
    DENSE_FAILURE_RANK_WEIGHT,
    DENSE_ONLY_RANK_WEIGHT,
    DENSE_GOAL_PROGRESS_TEMPERATURE_METERS,
    EPISTEMIC_RANK_RISK_WEIGHT,
    BRANCH_DIAGNOSTIC_CONTRACT,
    EffectAlignedSharedEventHead,
    InvariantMonotoneConsequenceUtility,
    MANIFEST_FORMAT,
    MATERIALIZATION_FORMAT,
    MacroBalancedRankDecisionBatchSampler,
    MODEL_FAMILY,
    OBJECT_EFFECT_SCHEMA,
    PREREGISTRATION_SHA256,
    RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT,
    MONOTONE_BENEFIT_FEATURES,
    MONOTONE_RISK_FEATURES,
    SEMANTIC_COMPARATIVE_GRADIENT_BUDGET,
    SEMANTIC_GRADIENT_SCALE_CAP,
    TERMINAL_FILM_MODULATION_BOUND,
    TERMINAL_EVENT_ORDINAL_RPS_LOSS_WEIGHT,
    TERMINAL_SUPERVISION_CONTRACT,
    SOURCE_EVENT_SAMPLING_HZ,
    SUPPLEMENT_ACTOR_BRANCH_CONTRACT,
    SUPPLEMENT_BINDING_FORMAT,
    SUPPLEMENT_COLLECTOR_FORMAT,
    SUPPLEMENT_HORIZON_CONTRACT,
    SUPPLEMENT_MANIFEST_FORMAT,
    SUPPLEMENT_MATERIALIZER_FORMAT,
    SUPPLEMENT_PROPER_LOSS_WEIGHT,
    SUPPLEMENT_RANK_LOSS_WEIGHT,
    SUPPLEMENT_RESERVE_ROSTER_CONTRACT,
    SUPPLEMENT_ROOT_SELECTION_CONTRACT,
    SUPPLEMENT_USAGE_CONTRACT,
    EXPERT_ROOT_PROVENANCE_CONTRACT,
    TASK,
    FiveBodyContractError,
    ablation_contract,
    ablation_selection_components,
    aggregate_risk_adjusted_rank_scores,
    build_preflight_receipt,
    canonical_sha256,
    checkpoint_candidate_rank_contract,
    candidate_checkpoint_selection_key,
    effect_preserving_group_bootstrap_weights,
    proper_outcome_preserving_group_bootstrap_weights,
    evaluate_candidate_ranking,
    evaluate_terminal_consequences,
    load_binding,
    load_supplement_binding,
    materialize_source_rows,
    materialize_supplement_rows,
    select_calibration_guarded_checkpoint,
    sha256_file,
    sha256_tree,
    source_group_split,
    supplement_group_bootstrap_weights,
    supplement_reserve_attempt_id,
    supplement_reserve_group_id,
    supplement_reserve_horizon_by_seed,
    supplement_reserve_roster,
    supplement_source_train_split,
    summary_candidate_rank_contract,
    validate_ensemble_seeds,
    validate_supplement_body_manifest,
)


def _signed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["logical_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha256_file(path)


def _group(
    path: Path,
    offset: float,
    *,
    current_event: int = 0,
    remaining_action_budget: int = 175,
) -> None:
    count, horizon = 4, 5
    state = np.full((count, core.STATE_DIM), offset, dtype=np.float32)
    state[:, 18:27] = 0.0
    state[:, 18 + current_event] = 1.0
    terminal_goal_progress = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    terminal_goal_distance = (
        np.linalg.norm(state[:, 0:3], axis=-1) - terminal_goal_progress
    ).astype(np.float32)
    np.savez(
        path,
        state=state,
        actions=np.full((count, horizon, core.ACTION_DIM), offset, dtype=np.float32),
        action_mask=np.ones((count, horizon), dtype=np.float32),
        current_event_id=np.full(count, current_event, dtype=np.int64),
        post_event_id=np.maximum(
            np.asarray([1, 2, 3, 4], dtype=np.int64), current_event
        ),
        post_event_mask=np.ones(count, dtype=np.float32),
        next_event_id=np.maximum(
            np.asarray([1, 2, 3, 4], dtype=np.int64), current_event
        ),
        next_event_mask=np.ones(count, dtype=np.float32),
        duration=np.asarray([0, 2, 3, 4], dtype=np.float32),
        duration_observed=np.asarray([0, 1, 1, 1], dtype=np.float32),
        duration_mask=np.asarray([0, 1, 1, 1], dtype=np.float32),
        success=np.asarray([0, 1, 0, 1], dtype=np.float32),
        success_mask=np.ones(count, dtype=np.float32),
        recovery=np.asarray([0, 0, 1, 0], dtype=np.float32),
        recovery_mask=np.ones(count, dtype=np.float32),
        object_delta=np.full((count, core.OBJECT_DELTA_DIM), 0.1, dtype=np.float32),
        object_delta_mask=np.ones(count, dtype=np.float32),
        terminal_max_event_id=np.maximum(
            np.asarray([1, 4, 3, 4], dtype=np.int64), current_event
        ),
        terminal_event_mask=np.ones(count, dtype=np.float32),
        terminal_stage_progress=np.where(
            np.asarray([0, 1, 0, 1], dtype=bool),
            1.0,
            np.maximum(
                np.asarray([1, 4, 3, 4], dtype=np.float32), current_event
            )
            / 4.0,
        ).astype(np.float32),
        terminal_goal_distance=terminal_goal_distance,
        terminal_goal_progress=terminal_goal_progress,
        terminal_goal_progress_mask=np.ones(count, dtype=np.float32),
        terminal_stop_reason_id=np.asarray([1, 0, 1, 0], dtype=np.int64),
        candidate_index=np.arange(count, dtype=np.int64),
        event_age_seconds=np.full(count, 0.4, dtype=np.float32),
        remaining_action_budget=np.full(
            count, float(remaining_action_budget), dtype=np.float32
        ),
        dt=np.full(count, 5.0 / SOURCE_EVENT_SAMPLING_HZ, dtype=np.float32),
    )


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    files = [
        {
            "path": f"dataset/move_can_pot/file-{index}.zip",
            "size_bytes": index + 1,
            "expected_payload_sha256": f"{index + 1:064x}",
            "observed_payload_sha256": f"{index + 1:064x}",
            "size_match": True,
            "payload_sha256_match": True,
            "zip_central_directory_audit": {
                "central_directory_read_only": True,
                "member_payload_bytes_read": 0,
            },
        }
        for index in range(11)
    ]
    materialization_unsigned = {
        "format": MATERIALIZATION_FORMAT,
        "status": verifier.STATUS,
        "materialized": True,
        "no_missing_or_extra_official_task_files": True,
        "all_exact_sizes_verified": True,
        "all_exact_archive_payload_sha256_verified": True,
        "hf_repo_id": DATASET_REPO,
        "hf_repo_revision": DATASET_REVISION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "official_file_count": 11,
        "official_total_size_bytes": sum(range(1, 12)),
        "files": files,
        "read_boundary": {
            "zip_member_payload_bytes_read": 0,
            "archive_extracted": False,
            "pickle_payload_opened_or_deserialized": False,
            "numpy_payload_opened_or_deserialized": False,
            "torch_payload_opened_or_deserialized": False,
        },
        "implementation_binding": {
            "verifier_module": Path(verifier.__file__).name,
            "verifier_file_sha256": verifier._file_sha256(Path(verifier.__file__).resolve()),
            "preregistration_module": Path(prereg.__file__).name,
            "preregistration_module_file_sha256": verifier._file_sha256(
                Path(prereg.__file__).resolve()
            ),
        },
        "authority": {
            "download_completeness_attested": True,
            "training_authorized": False,
            "evaluation_authorized": False,
            "simulator_execution_authorized": False,
            "checkpoint_selection_or_promotion_authorized": False,
            "deployment_authorized": False,
            "cross_embodiment_performance_claim_authorized": False,
        },
    }
    materialization = {
        **materialization_unsigned,
        "materialization_receipt_sha256": verifier.canonical_sha256(
            materialization_unsigned
        ),
    }
    materialization_path = tmp_path / "materialization.json"
    materialization_sha = _write_json(materialization_path, materialization)

    actors = {}
    for index, body in enumerate(BODIES):
        checkpoint_path = tmp_path / f"{body}-actor.ckpt"
        if body == "piper":
            checkpoint_path.mkdir()
            (checkpoint_path / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint_path / "model.safetensors").write_bytes(b"frozen-piper")
            checkpoint_kind = "directory_tree"
            checkpoint_sha256 = sha256_tree(checkpoint_path)[0]
        else:
            checkpoint_path.write_bytes(f"frozen-{body}".encode())
            checkpoint_kind = "file"
            checkpoint_sha256 = sha256_file(checkpoint_path)
        actors[body] = {
            "family": "synthetic-test-native-actor",
            "frozen": True,
            "optimizer_updates_allowed": False,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_sha256": checkpoint_sha256,
            "sampling_contract_sha256": f"{index + 40:064x}",
            "candidate_count": 4,
            "candidate_zero_is_actor_baseline": True,
            "same_ordered_candidate_set_for_baseline_and_etsf": True,
        }
    actor_authority = _signed(
        {
            "format": ACTOR_FORMAT,
            "task": TASK,
            "actors": actors,
        }
    )
    actor_path = tmp_path / "actors.json"
    actor_sha = _write_json(actor_path, actor_authority)

    body_bindings = {}
    for body_index, body in enumerate(BODIES):
        groups = []
        for condition in ("clean", "randomized"):
            for group_index in range(2):
                name = f"{body}-{condition}-{group_index}.npz"
                group_path = tmp_path / name
                _group(group_path, float(body_index + group_index + 1))
                groups.append(
                    {
                        "group_id": name.removesuffix(".npz"),
                        "condition": condition,
                        "requested_seed": body_index * 100 + group_index,
                        "path": name,
                        "sha256": sha256_file(group_path),
                        "branch_root_snapshot_sha256": "a" * 64,
                        "branch_root_restorable_snapshot_sha256": "b" * 64,
                        "canonical_root_snapshot_sha256": "c" * 64,
                        "diagnostic_format": BRANCH_DIAGNOSTIC_CONTRACT["format"],
                        "diagnostics_path": name.replace(".npz", ".diagnostics.npz"),
                        "diagnostics_sha256": "d" * 64,
                    }
                )
        manifest = _signed(
            {
                "format": MANIFEST_FORMAT,
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "task": TASK,
                "instruction": DEFAULT_INSTRUCTION,
                "body": body,
                "schema_adapter": {
                    "kind": "analytic_label_free_canonical_v1",
                    "trainable": False,
                    "labels_or_outcomes_used_to_fit": False,
                    "heldout_supervision_allowed": False,
                    "state_dim": core.STATE_DIM,
                    "action_dim": core.ACTION_DIM,
                    "state_schema": CANONICAL_STATE_SCHEMA,
                    "action_schema": CANONICAL_ACTION_SCHEMA,
                    "elapsed_time_unit": "seconds",
                    "duration_unit": "seconds",
                    "event_names": list(core.CANONICAL_EVENTS),
                    "implementation_sha256": f"{body_index + 60:064x}",
                },
                "event_spec_sha256": EVENT_SPEC_SHA256,
                "analytic_event_contract": analytic_event.event_contract(
                    {
                        "moving": "can",
                        "anchor": "pot",
                        "required_objects": list(analytic_event.REQUIRED_OBJECTS),
                        "goal_rule": dict(analytic_event.GOAL_RULE),
                        "thresholds": dict(analytic_event.THRESHOLDS),
                        "event_rules": dict(analytic_event.EVENT_RULES),
                    }
                ),
                "event_derivation_implementation_sha256": "7" * 64,
                "state27_relative_goal_contract": (
                    "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
                    "event_labels_and_online_state27_channels_0_2"
                ),
                "physical_time_contract": {
                    "source": "counted_successful_sapien_scene_step_calls",
                    "simulator_timestep_source": "scene.get_timestep",
                    "policy_action_call_count_used_as_time": False,
                    "wall_clock_used_as_time": False,
                    "dt_semantics": "planned_first_candidate_chunk_seconds",
                    "planned_action_steps": 5,
                    "actor_control_hz": SOURCE_EVENT_SAMPLING_HZ,
                    "planned_dt_seconds": 5.0 / SOURCE_EVENT_SAMPLING_HZ,
                    "duration_semantics": "simulator_elapsed_seconds_to_event_boundary",
                    "zero_elapsed_duration_masked": True,
                    "stationary_source_sampling_hz": SOURCE_EVENT_SAMPLING_HZ,
                    "stationary_window_seconds": analytic_event.THRESHOLDS[
                        "stationary_window_seconds"
                    ],
                    "stationary_speed_threshold_m_per_s": analytic_event.THRESHOLDS[
                        "stationary_speed_m_per_s"
                    ],
                },
                "candidate_action_contract": {
                    "critic_observation_time": "before_candidate_execution",
                    "planned_action_horizon": 5,
                    "action_mask_source": "planned_first_chunk_not_executed_count",
                    "executed_action_count_used_for_action_mask": False,
                    "executed_action_count_used_for_sim_time_accounting_only": True,
                    "planner_status_fail_is_a_valid_action_outcome": True,
                    "python_execution_exception_invalidates_complete_decision": True,
                },
                "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
                "object_effect_schema": OBJECT_EFFECT_SCHEMA,
                "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
                "event_age_contract": EVENT_AGE_CONTRACT,
                "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
                "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
                "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
                "groups": groups,
            }
        )
        manifest_path = tmp_path / f"{body}-manifest.json"
        body_bindings[body] = {
            "path": manifest_path.name,
            "sha256": _write_json(manifest_path, manifest),
        }

    binding = _signed(
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
                "path": materialization_path.name,
                "sha256": materialization_sha,
            },
            "actor_authority": {"path": actor_path.name, "sha256": actor_sha},
            "body_manifests": body_bindings,
        }
    )
    binding_path = tmp_path / "binding.json"
    return binding_path, _write_json(binding_path, binding)


def _supplement_fixture(
    tmp_path: Path,
    primary_binding_path: Path,
    primary_binding_sha256: str,
) -> tuple[Path, str]:
    primary_binding = json.loads(primary_binding_path.read_text(encoding="utf-8"))
    actor_authority_sha256 = primary_binding["actor_authority"]["sha256"]
    actor_authority = json.loads(
        (tmp_path / primary_binding["actor_authority"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    body_bindings = {}
    rejected_attempt_count = 0
    for body_index, body in enumerate(BODIES):
        primary_manifest_path = tmp_path / primary_binding["body_manifests"][body]["path"]
        primary_manifest = json.loads(
            primary_manifest_path.read_text(encoding="utf-8")
        )
        roster = supplement_reserve_roster(body)
        seeds = [
            seed
            for row in roster
            for seed in row["ordered_requested_seeds"]
        ]
        horizon_by_seed = supplement_reserve_horizon_by_seed(body)
        selected_seed_by_slot = {}
        attempts = []
        groups = []
        for row in roster:
            condition = row["condition"]
            condition_index = ("clean", "randomized").index(condition)
            slot = row["horizon_slot"]
            horizon = row["remaining_action_budget"]
            ordered_seeds = row["ordered_requested_seeds"]
            # Exercise one genuine ordered rejection per body in the positive
            # fixture; every other slot selects its first reserve seed.
            selected_index = 1 if condition == "clean" and slot == 0 else 0
            selected_seed = ordered_seeds[selected_index]
            selected_seed_by_slot[row["slot_key"]] = selected_seed
            for rejected_seed in ordered_seeds[:selected_index]:
                attempts.append(
                    {
                        "attempt_id": supplement_reserve_attempt_id(
                            condition, slot, rejected_seed
                        ),
                        "status": "rejected_before_actor_outcomes",
                        "condition": condition,
                        "horizon_slot": slot,
                        "requested_seed": rejected_seed,
                        "pre_registered_horizon": horizon,
                        "reject_reason": "missing_e4",
                        "actor_candidate_outcomes_executed_before_selection": False,
                    }
                )
                rejected_attempt_count += 1
            attempts.append(
                {
                    "attempt_id": supplement_reserve_attempt_id(
                        condition, slot, selected_seed
                    ),
                    "status": "complete",
                    "condition": condition,
                    "horizon_slot": slot,
                    "requested_seed": selected_seed,
                    "pre_registered_horizon": horizon,
                    "selected_before_actor_candidate_outcomes": True,
                    "actor_candidate_outcomes_executed_before_selection": False,
                    "root_triplet_bundle_sha256": "5" * 64,
                }
            )
            for root_event, event_name in ((1, "e12"), (2, "e3"), (3, "e4")):
                name = (
                    f"supplement-{body}-{condition}-h{slot}-{event_name}-"
                    f"seed{selected_seed}.npz"
                )
                path = tmp_path / name
                _group(
                    path,
                    float(body_index + condition_index + root_event + 1),
                    current_event=root_event,
                    remaining_action_budget=horizon,
                )
                groups.append(
                    {
                        "group_id": supplement_reserve_group_id(
                            condition, slot, selected_seed, event_name
                        ),
                        "collector_file_sha256": "8" * 64,
                        "base_collector_file_sha256": "9" * 64,
                        "condition": condition,
                        "horizon_slot": slot,
                        "requested_seed": selected_seed,
                        "scripted_root_event": event_name,
                        "scripted_root_event_id": root_event,
                        "root_event_id": root_event,
                        "pre_registered_horizon": horizon,
                        "candidate_noise_query_index": {1: 1, 2: 2, 3: 3}[
                            root_event
                        ],
                        "raw_expert_snapshot_sha256": "0" * 64,
                        "branch_root_snapshot_sha256": "1" * 64,
                        "branch_root_restorable_snapshot_sha256": "2" * 64,
                        "canonical_root_snapshot_sha256": "3" * 64,
                        "path": name,
                        "sha256": sha256_file(path),
                        "diagnostic_format": BRANCH_DIAGNOSTIC_CONTRACT["format"],
                        "diagnostics_path": name.replace(
                            ".npz", ".diagnostics.npz"
                        ),
                        "diagnostics_sha256": "4" * 64,
                    }
                )
        manifest = {
            "format": SUPPLEMENT_MANIFEST_FORMAT,
            "collector_format": SUPPLEMENT_COLLECTOR_FORMAT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "body": body,
            "conditions": ["clean", "randomized"],
            "collection_status": "complete",
            "reserve_roster_contract": dict(
                SUPPLEMENT_RESERVE_ROSTER_CONTRACT
            ),
            "reserve_roster": roster,
            "pre_registered_seeds": seeds,
            "pre_registered_horizon_by_seed": {
                str(seed): horizon for seed, horizon in horizon_by_seed.items()
            },
            "selected_seed_by_slot": selected_seed_by_slot,
            "attempts": attempts,
            "target_events": ["e12", "e3", "e4"],
            "collector_file_sha256": "8" * 64,
            "base_collector_file_sha256": "9" * 64,
            "actor_checkpoint_tree_or_file_sha256": actor_authority["actors"][body][
                "checkpoint_sha256"
            ],
            "actor_authority_sha256": actor_authority_sha256,
            "instruction": DEFAULT_INSTRUCTION,
            "candidate_count": 4,
            "action_exec_steps": 5,
            "supplement_role": (
                "expert_event_root_proper_world_and_utility_rank_source_train_only"
            ),
            "root_policy": "robotwin_scripted_expert",
            "candidate_and_continuation_policy": (
                "same_frozen_native_actor_as_primary_binding"
            ),
            "proper_loss_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
            "rank_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
            "usage_contract": dict(SUPPLEMENT_USAGE_CONTRACT),
            "expert_root_provenance_contract": dict(
                EXPERT_ROOT_PROVENANCE_CONTRACT
            ),
            "root_selection_contract": dict(SUPPLEMENT_ROOT_SELECTION_CONTRACT),
            "horizon_contract": dict(SUPPLEMENT_HORIZON_CONTRACT),
            "actor_branch_contract": dict(SUPPLEMENT_ACTOR_BRANCH_CONTRACT),
            "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
            "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
            "event_age_contract": EVENT_AGE_CONTRACT,
            "terminal_horizon_contract": TERMINAL_HORIZON_CONTRACT,
            "branch_root_snapshot_contract": BRANCH_ROOT_SNAPSHOT_CONTRACT,
            "object_effect_schema": OBJECT_EFFECT_SCHEMA,
            "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "event_derivation_implementation_sha256": primary_manifest[
                "event_derivation_implementation_sha256"
            ],
            "schema_adapter": dict(primary_manifest["schema_adapter"]),
            "analytic_event_contract": dict(
                primary_manifest["analytic_event_contract"]
            ),
            "state27_relative_goal_contract": primary_manifest[
                "state27_relative_goal_contract"
            ],
            "physical_time_contract": dict(
                primary_manifest["physical_time_contract"]
            ),
            "candidate_action_contract": dict(
                primary_manifest["candidate_action_contract"]
            ),
            "groups": groups,
        }
        manifest = _signed(manifest)
        manifest_path = tmp_path / f"supplement-{body}-manifest.json"
        body_bindings[body] = {
            "path": manifest_path.name,
            "sha256": _write_json(manifest_path, manifest),
            "group_count": len(groups),
            "selected_seed_by_slot_sha256": canonical_sha256(
                selected_seed_by_slot
            ),
            "reserve_roster_sha256": canonical_sha256(roster),
        }
    binding = _signed(
        {
            "format": SUPPLEMENT_BINDING_FORMAT,
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
            "primary_binding_file_sha256": primary_binding_sha256,
            "actor_authority_sha256": actor_authority_sha256,
            "proper_loss_weight": SUPPLEMENT_PROPER_LOSS_WEIGHT,
            "rank_loss_weight": SUPPLEMENT_RANK_LOSS_WEIGHT,
            "usage_contract": dict(SUPPLEMENT_USAGE_CONTRACT),
            "expert_root_provenance_contract": dict(
                EXPERT_ROOT_PROVENANCE_CONTRACT
            ),
            "body_manifests": body_bindings,
            "materializer_provenance": {
                "format": SUPPLEMENT_MATERIALIZER_FORMAT,
                "payload_npz_files_opened": 0,
                "complete_decisions": 150,
                "complete_branches": 600,
                "seed_overlap_with_primary": 0,
                "selected_seed_count": 50,
                "rejected_attempt_count": rejected_attempt_count,
                "selection_occurs_before_actor_candidate_outcomes": True,
                "heldout_payload_npz_files_opened": 0,
            },
        }
    )
    path = tmp_path / "supplement-binding.json"
    return path, _write_json(path, binding)


def test_preflight_is_five_fold_source_only_and_payload_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("preflight opened a transition NPZ")
    ))
    for heldout in BODIES:
        receipt = build_preflight_receipt(audit, held_out_body=heldout, split_seed=17)
        assert heldout not in receipt["source_bodies"]
        assert len(receipt["source_bodies"]) == 4
        assert receipt["heldout_group_npz_opened"] == 0
        assert receipt["heldout_specific_trainable_parameters"] == 0
        assert receipt["model_body_rows"] == 1
        assert receipt["actor_frozen"] is True
        assert receipt["split_unit"] == "body_condition_requested_seed_all_queries"
        assert receipt["supplement"]["enabled"] is False
        assert receipt["supplement"]["proper_loss_weight"] == 0.0
        assert receipt["supplement"]["rank_loss_weight"] == 0.0
        assert receipt["supplement"]["rank_or_utility_rows_authorized"] == 0
    assert trainer_entry.candidate_rank_supervision_inventory([]) == {
        "mixed_success_groups": 0,
        "informative_dense_groups": 0,
        "rank_supervision_groups": 0,
    }


def test_source_split_keeps_all_queries_from_one_seed_in_one_lane() -> None:
    manifests = {}
    for body in BODIES:
        groups = []
        for condition in ("clean", "randomized"):
            for seed in (101, 102, 103, 104, 105):
                for query in (0, 10, 20, 30):
                    groups.append(
                        {
                            "condition": condition,
                            "requested_seed": seed,
                            "group_id": f"{condition}-seed{seed}-query{query}",
                        }
                    )
        manifests[body] = {"groups": groups}
    train, validation, _heldout = source_group_split(
        {"manifests": manifests}, held_out_body="franka", split_seed=19
    )
    for body in BODIES:
        if body == "franka":
            continue
        for condition in ("clean", "randomized"):
            train_seeds = {
                row["requested_seed"]
                for row in train
                if row["body"] == body and row["condition"] == condition
            }
            validation_seeds = {
                row["requested_seed"]
                for row in validation
                if row["body"] == body and row["condition"] == condition
            }
            assert train_seeds and validation_seeds
            assert train_seeds.isdisjoint(validation_seeds)
            for seed in train_seeds | validation_seeds:
                train_count = sum(
                    row["requested_seed"] == seed
                    and row["body"] == body
                    and row["condition"] == condition
                    for row in train
                )
                validation_count = sum(
                    row["requested_seed"] == seed
                    and row["body"] == body
                    and row["condition"] == condition
                    for row in validation
                )
                assert (train_count, validation_count) in {(4, 0), (0, 4)}


def test_source_split_validation_and_training_cover_every_formal_horizon() -> None:
    manifests = {}
    for body in BODIES:
        groups = []
        for condition in ("clean", "randomized"):
            for block in range(10):
                for seed_offset in range(5):
                    seed = 10_000 + 5 * block + seed_offset
                    for query in range(4 * block, 4 * block + 4):
                        groups.append(
                            {
                                "condition": condition,
                                "requested_seed": seed,
                                "root_query_index": query,
                                "group_id": (
                                    f"{condition}|seed={seed}|query={query}"
                                ),
                            }
                        )
        manifests[body] = {"groups": groups}
    train, validation, _heldout = source_group_split(
        {"manifests": manifests}, held_out_body="franka", split_seed=19
    )
    for body in BODIES:
        if body == "franka":
            continue
        for condition in ("clean", "randomized"):
            train_rows = [
                row
                for row in train
                if row["body"] == body and row["condition"] == condition
            ]
            validation_rows = [
                row
                for row in validation
                if row["body"] == body and row["condition"] == condition
            ]
            train_seeds = {row["requested_seed"] for row in train_rows}
            validation_seeds = {
                row["requested_seed"] for row in validation_rows
            }
            assert len(validation_seeds) == 10
            assert train_seeds.isdisjoint(validation_seeds)
            assert {row["root_query_index"] for row in train_rows} == set(range(40))
            assert {
                row["root_query_index"] for row in validation_rows
            } == set(range(40))


def test_heldout_payload_is_not_stat_hashed_or_deserialized_in_preflight(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    manifest_path = tmp_path / decoded["body_manifests"]["franka"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in manifest["groups"]:
        (tmp_path / group["path"]).unlink()
    # A strict Franka-heldout preflight consumes only the declared paths and
    # commitments.  Missing target payloads therefore cannot be observed.
    audit = load_binding(binding, digest)
    receipt = build_preflight_receipt(
        audit, held_out_body="franka", split_seed=23
    )
    assert receipt["heldout_group_payload_bytes_read"] == 0
    assert receipt["heldout_group_payload_deserialized"] == 0

    # The same missing files fail once Franka becomes a source body and its
    # source payload boundary is deliberately crossed.
    train, _, _ = source_group_split(audit, held_out_body="piper", split_seed=23)
    franka = [group for group in train if group["body"] == "franka"]
    with pytest.raises(FiveBodyContractError, match="missing/tampered"):
        materialize_source_rows(franka, held_out_body="piper")


def test_supplement_heldout_manifest_and_payload_are_zero_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    declared = json.loads(supplement_binding.read_text(encoding="utf-8"))
    heldout_manifest_path = (
        tmp_path / declared["body_manifests"]["franka"]["path"]
    )
    heldout_manifest = json.loads(heldout_manifest_path.read_text(encoding="utf-8"))
    for group in heldout_manifest["groups"]:
        (tmp_path / group["path"]).unlink()
    heldout_manifest_path.unlink()

    supplement = load_supplement_binding(
        supplement_binding,
        supplement_digest,
        primary_audit=primary,
        held_out_body="franka",
    )
    assert set(supplement["manifests"]) == set(BODIES) - {"franka"}
    assert supplement["heldout_manifest_binding"]["manifest_file_opened"] == 0
    assert supplement["heldout_manifest_binding"]["payload_files_opened"] == 0
    monkeypatch.setattr(
        np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight opened a supplement transition NPZ")
        ),
    )
    receipt = build_preflight_receipt(
        primary,
        held_out_body="franka",
        split_seed=23,
        supplement_audit=supplement,
    )
    assert receipt["supplement"]["source_train_groups"] == 120
    assert receipt["supplement"]["heldout_groups_deferred"] == 30
    assert receipt["supplement"]["heldout_manifest_file_opened"] == 0
    assert receipt["supplement"]["heldout_group_npz_opened"] == 0


def test_supplement_binding_reserve_provenance_fails_closed(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, _supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    original = json.loads(supplement_binding.read_text(encoding="utf-8"))

    changed_body_hash = json.loads(json.dumps(original))
    changed_body_hash["body_manifests"]["aloha-agilex"][
        "reserve_roster_sha256"
    ] = "f" * 64
    changed_body_hash.pop("logical_sha256")
    changed_body_hash = _signed(changed_body_hash)
    changed_body_path = tmp_path / "supplement-binding-changed-body.json"
    changed_body_digest = _write_json(changed_body_path, changed_body_hash)
    with pytest.raises(FiveBodyContractError, match="reserve provenance"):
        load_supplement_binding(
            changed_body_path,
            changed_body_digest,
            primary_audit=primary,
            held_out_body="franka",
        )

    changed_count = json.loads(json.dumps(original))
    changed_count["materializer_provenance"]["rejected_attempt_count"] = 0
    changed_count.pop("logical_sha256")
    changed_count = _signed(changed_count)
    changed_count_path = tmp_path / "supplement-binding-changed-count.json"
    changed_count_digest = _write_json(changed_count_path, changed_count)
    with pytest.raises(FiveBodyContractError, match="rejected-attempt provenance"):
        load_supplement_binding(
            changed_count_path,
            changed_count_digest,
            primary_audit=primary,
            held_out_body="franka",
        )


def test_supplement_is_source_train_only_and_never_enters_validation(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    supplement = load_supplement_binding(
        supplement_binding,
        supplement_digest,
        primary_audit=primary,
        held_out_body="franka",
    )
    _primary_train, primary_validation, _primary_heldout = source_group_split(
        primary, held_out_body="franka", split_seed=19
    )
    supplement_train, supplement_heldout = supplement_source_train_split(
        supplement, held_out_body="franka"
    )
    assert supplement_heldout["declared_group_count"] == 30
    assert all(group["body"] != "franka" for group in supplement_train)
    assert all(
        group.get("source_role") == "proper_world_supplement"
        for group in supplement_train
    )
    validation_identities = {
        (row["body"], row["condition"], row["group_id"])
        for row in primary_validation
    }
    supplement_identities = {
        (row["body"], row["condition"], row["group_id"])
        for row in supplement_train
    }
    assert validation_identities.isdisjoint(supplement_identities)
    rows = materialize_supplement_rows(
        supplement_train, held_out_body="franka"
    )
    assert len(rows) == 480
    assert {int(row["current_event_id"]) for row in rows} == {1, 2, 3}
    assert all(
        "|proper-world-supplement|" in row["logical_group"] for row in rows
    )
    receipt = build_preflight_receipt(
        primary,
        held_out_body="franka",
        split_seed=19,
        supplement_audit=supplement,
    )
    assert receipt["supplement"]["source_validation_groups"] == 0
    assert receipt["supplement"]["normalization_rows_used"] == 0
    assert receipt["supplement"]["rank_or_utility_rows_authorized"] == 480


def test_supplement_seed_and_expert_root_contracts_fail_closed(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, _supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    declared = json.loads(supplement_binding.read_text(encoding="utf-8"))
    manifest_path = tmp_path / declared["body_manifests"]["aloha-agilex"]["path"]
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha = primary["actor"]["checkpoint_sha256_by_body"]["aloha-agilex"]

    changed_seed = json.loads(json.dumps(original))
    changed_seed["pre_registered_seeds"][-1] += 1
    changed_seed.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="reserve design"):
        validate_supplement_body_manifest(
            _signed(changed_seed),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    incomplete = json.loads(json.dumps(original))
    incomplete["collection_status"] = "collecting"
    incomplete.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="reserve design"):
        validate_supplement_body_manifest(
            _signed(incomplete),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    changed_horizon = json.loads(json.dumps(original))
    first_seed = str(changed_horizon["pre_registered_seeds"][0])
    changed_horizon["pre_registered_horizon_by_seed"][first_seed] += 1
    changed_horizon.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="reserve design"):
        validate_supplement_body_manifest(
            _signed(changed_horizon),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    missing_slot = json.loads(json.dumps(original))
    missing_slot["selected_seed_by_slot"].pop("clean|horizon_slot=1")
    missing_slot.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="reserve design"):
        validate_supplement_body_manifest(
            _signed(missing_slot),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    missing_rejection = json.loads(json.dumps(original))
    missing_rejection["attempts"].pop(0)
    missing_rejection.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="rejection history"):
        validate_supplement_body_manifest(
            _signed(missing_rejection),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    late_selection = json.loads(json.dumps(original))
    late_selection["attempts"][1][
        "selected_before_actor_candidate_outcomes"
    ] = False
    late_selection.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="selected.*incomplete"):
        validate_supplement_body_manifest(
            _signed(late_selection),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    wrong_group_identity = json.loads(json.dumps(original))
    wrong_group_identity["groups"][0]["group_id"] += "|tampered"
    wrong_group_identity.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="group"):
        validate_supplement_body_manifest(
            _signed(wrong_group_identity),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )

    changed_root = json.loads(json.dumps(original))
    changed_root["root_selection_contract"][
        "e4_must_be_nonterminal_simulator_success"
    ] = False
    changed_root.pop("logical_sha256")
    with pytest.raises(FiveBodyContractError, match="contract changed"):
        validate_supplement_body_manifest(
            _signed(changed_root),
            expected_body="aloha-agilex",
            manifest_dir=manifest_path.parent,
            expected_actor_checkpoint_sha256=checkpoint_sha,
        )


def test_supplement_fixed_lambda_updates_only_proper_world_objectives(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    supplement = load_supplement_binding(
        supplement_binding,
        supplement_digest,
        primary_audit=primary,
        held_out_body="franka",
    )
    groups, _heldout = supplement_source_train_split(
        supplement, held_out_body="franka"
    )
    rows = materialize_supplement_rows(groups[:1], held_out_body="franka")
    mapping = {body: 0 for body in BODIES if body != "franka"}
    dataset = core.TransitionDataset(rows, mapping)
    batch = core.collate_rows([dataset[index] for index in range(4)])
    model = EffectAlignedSharedEventHead().train()
    output = model(batch)
    loss_weights = dict(core.DEFAULT_LOSS_WEIGHTS)
    loss_weights["object"] = 0.0
    loss, pieces = _supplement_proper_world_model_loss(
        output,
        batch,
        torch.ones(4),
        loss_weights=loss_weights,
    )
    assert torch.allclose(
        loss,
        SUPPLEMENT_PROPER_LOSS_WEIGHT * pieces["supplement_proper_unweighted"],
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.semantic.parameters())
    assert all(
        parameter.grad is None for parameter in model.candidate_rank.parameters()
    )


def test_supplement_rank_fixed_lambda_updates_only_detached_utility(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    supplement = load_supplement_binding(
        supplement_binding,
        supplement_digest,
        primary_audit=primary,
        held_out_body="franka",
    )
    groups, _heldout = supplement_source_train_split(
        supplement, held_out_body="franka"
    )
    rows = materialize_supplement_rows(groups[:1], held_out_body="franka")
    mapping = {body: 0 for body in BODIES if body != "franka"}
    dataset = core.TransitionDataset(rows, mapping)
    batch = core.collate_rows([dataset[index] for index in range(4)])
    model = EffectAlignedSharedEventHead().train()
    output = model(batch)
    loss, pieces = _supplement_candidate_rank_loss(
        output,
        batch,
        torch.ones(4),
    )
    torch.testing.assert_close(
        loss,
        SUPPLEMENT_RANK_LOSS_WEIGHT
        * pieces["supplement_candidate_rank_unweighted"],
    )
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.candidate_rank.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("candidate_rank.")
    )


def test_supplement_bootstrap_is_member_specific_and_complete_group_constant(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    primary = load_binding(binding, digest)
    supplement_binding, supplement_digest = _supplement_fixture(
        tmp_path, binding, digest
    )
    supplement = load_supplement_binding(
        supplement_binding,
        supplement_digest,
        primary_audit=primary,
        held_out_body="franka",
    )
    groups, _heldout = supplement_source_train_split(
        supplement, held_out_body="franka"
    )
    rows = materialize_supplement_rows(groups[:8], held_out_body="franka")
    weights, audit, bootstrap_seed = supplement_group_bootstrap_weights(
        rows, members=5, seed=20260901
    )
    assert weights.shape == (5, len(rows))
    assert len(audit) == 5
    assert all(item["bootstrap_seed"] == bootstrap_seed for item in audit)
    assert all(item["class_balancing_used"] is False for item in audit)
    assert all(item["synthetic_groups_or_labels"] == 0 for item in audit)
    indices_by_group: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_group.setdefault(str(row["logical_group"]), []).append(index)
    for member in range(5):
        for indices in indices_by_group.values():
            assert len(indices) == 4
            assert np.all(weights[member, indices] == weights[member, indices[0]])
    member_group_weights = {
        tuple(float(weights[member, indices[0]]) for indices in indices_by_group.values())
        for member in range(5)
    }
    assert len(member_group_weights) > 1


def test_source_materialization_never_accepts_heldout_and_model_has_one_body_row(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    train, validation, heldout = source_group_split(
        audit, held_out_body="franka", split_seed=19
    )
    assert all(row["body"] != "franka" for row in train + validation)
    assert all("body" not in row for row in heldout)
    rows = materialize_source_rows(train, held_out_body="franka")
    assert rows and all(row["body"] != "franka" for row in rows)
    mapping = {body: 0 for body in BODIES if body != "franka"}
    dataset = core.TransitionDataset(rows, mapping)
    batch = core.collate_rows([dataset[0], dataset[1]])
    model = core.MultibodyCanonicalEventWorldModel(
        core.ModelConfig(body_count=1, action_schema_count=1, dropout=0.0)
    ).eval()
    assert model.clock.body_beta.weight.shape[0] == 1
    assert model.action.schema_count == 1
    assert set(batch["body_id"].tolist()) == {0}
    with torch.no_grad():
        output = model(batch)
    assert output["success_logit"].shape == (2,)
    with pytest.raises(FiveBodyContractError, match="held-out group"):
        materialize_source_rows(
            [{**audit["manifests"]["franka"]["groups"][0], "body": "franka"}],
            held_out_body="franka",
        )


def test_tampered_or_supervised_adapter_fails_closed(tmp_path: Path) -> None:
    binding, digest = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    manifest_binding = decoded["body_manifests"]["piper"]
    manifest_path = tmp_path / manifest_binding["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_adapter"]["labels_or_outcomes_used_to_fit"] = True
    unsigned = dict(manifest)
    unsigned.pop("logical_sha256")
    manifest["logical_sha256"] = canonical_sha256(unsigned)
    manifest_binding["sha256"] = _write_json(manifest_path, manifest)
    binding_unsigned = dict(decoded)
    binding_unsigned.pop("logical_sha256")
    decoded["logical_sha256"] = canonical_sha256(binding_unsigned)
    digest = _write_json(binding, decoded)
    with pytest.raises(FiveBodyContractError, match="analytic/label-free"):
        load_binding(binding, digest)


def test_incomplete_public_download_receipt_fails_closed(tmp_path: Path) -> None:
    binding, _ = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    receipt_path = tmp_path / decoded["materialization_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "download_incomplete"
    unsigned = dict(receipt)
    unsigned.pop("materialization_receipt_sha256")
    receipt["materialization_receipt_sha256"] = verifier.canonical_sha256(unsigned)
    decoded["materialization_receipt"]["sha256"] = _write_json(receipt_path, receipt)
    unsigned_binding = dict(decoded)
    unsigned_binding.pop("logical_sha256")
    decoded["logical_sha256"] = canonical_sha256(unsigned_binding)
    digest = _write_json(binding, decoded)
    with pytest.raises(FiveBodyContractError, match="not verified"):
        load_binding(binding, digest)


def _model_batch(dt: torch.Tensor) -> dict[str, torch.Tensor]:
    count = len(dt)
    state = torch.zeros(count, core.STATE_DIM)
    state[:, 18] = 1.0
    return {
        "state": state,
        "actions": torch.randn(count, 5, core.ACTION_DIM),
        "action_mask": torch.ones(count, 5, dtype=torch.bool),
        "action_available": torch.ones(count, dtype=torch.bool),
        "action_schema_id": torch.zeros(count, dtype=torch.long),
        "body_id": torch.zeros(count, dtype=torch.long),
        "dt": dt,
        "event_age_seconds": torch.zeros(len(dt), dtype=torch.float32),
        "remaining_action_budget": torch.full(
            (count,), 175.0, dtype=torch.float32
        ),
        "current_event_id": torch.zeros(count, dtype=torch.long),
    }


def test_rank_gradient_updates_only_consequence_utility_not_world_model() -> None:
    torch.manual_seed(7)
    model = EffectAlignedSharedEventHead().eval()
    output = model(_model_batch(torch.full((4,), 5.0 / 15.0)))
    output["candidate_rank_logit"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.candidate_rank.parameters())
    assert all(parameter.grad is None for parameter in model.semantic.parameters())
    assert all(parameter.grad is None for parameter in model.action.parameters())
    assert all(parameter.grad is None for parameter in model.transition.parameters())
    assert all(parameter.grad is None for parameter in model.clock.parameters())
    assert all(parameter.grad is None for parameter in model.event_age_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.duration_mean.parameters())
    assert all(parameter.grad is None for parameter in model.duration_scale.parameters())
    assert all(parameter.grad is None for parameter in model.post_event.parameters())
    assert all(parameter.grad is None for parameter in model.next_event.parameters())
    assert all(parameter.grad is None for parameter in model.success.parameters())
    assert all(parameter.grad is None for parameter in model.recovery.parameters())
    assert all(parameter.grad is None for parameter in model.object_mean.parameters())
    assert all(parameter.grad is None for parameter in model.object_scale.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.terminal_context_encoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.terminal_residual.parameters()
    )
    assert all(parameter.grad is None for parameter in model.terminal_event.parameters())
    assert all(
        parameter.grad is None for parameter in model.terminal_recovery.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.terminal_goal_progress_mean.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.terminal_goal_progress_scale.parameters()
    )


def test_terminal_proper_loss_updates_long_horizon_predictors_and_backbone() -> None:
    torch.manual_seed(9)
    model = EffectAlignedSharedEventHead().train()
    batch = _model_batch(torch.full((4,), 5.0 / 15.0))
    batch.update(
        {
            "terminal_max_event_id": torch.tensor([1, 1, 3, 4]),
            "terminal_event_mask": torch.ones(4),
            "terminal_goal_progress": torch.tensor([-0.20, 0.05, 0.10, 0.25]),
            "terminal_goal_progress_mask": torch.ones(4),
        }
    )
    output = model(batch)
    loss, pieces = _terminal_consequence_loss(
        output, batch, torch.ones(4), ablation_variant="full"
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert pieces["terminal_event_uniform_proper"].requires_grad
    assert pieces["terminal_event_ordinal_rps_uniform_proper"].requires_grad
    torch.testing.assert_close(
        pieces["terminal_event_ordinal_rps_weighted_uniform_proper"],
        TERMINAL_EVENT_ORDINAL_RPS_LOSS_WEIGHT
        * pieces["terminal_event_ordinal_rps_uniform_proper"],
    )
    assert any(parameter.grad is not None for parameter in model.terminal_event.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.terminal_goal_progress_mean.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.action.parameters())
    assert any(parameter.grad is not None for parameter in model.transition.parameters())


def test_terminal_event_ordinal_rps_rewards_nearby_stage_mass_and_is_differentiable(
) -> None:
    target = torch.tensor([3], dtype=torch.long)
    near_logits = torch.nn.Parameter(
        torch.log(torch.tensor([[0.01, 0.01, 0.03, 0.85, 0.10]]))
    )
    far_logits = torch.log(
        torch.tensor([[0.85, 0.03, 0.01, 0.01, 0.10]])
    )
    near = _terminal_event_ordinal_rps_rows(near_logits, target)
    far = _terminal_event_ordinal_rps_rows(far_logits, target)
    assert near.shape == (1,)
    assert near.item() < far.item()
    near.sum().backward()
    assert near_logits.grad is not None
    assert torch.isfinite(near_logits.grad).all()


def test_semantic_comparative_loss_updates_terminal_locations_not_utility_or_scales(
) -> None:
    torch.manual_seed(12)
    model = EffectAlignedSharedEventHead().eval()
    batch = _model_batch(torch.full((8,), 5.0 / 15.0))
    batch.update(
        {
            "success": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]).float(),
            "success_mask": torch.ones(8),
            "terminal_max_event_id": torch.tensor([1, 4, 2, 3, 1, 2, 2, 1]),
            "terminal_event_mask": torch.ones(8),
            "terminal_goal_progress": torch.tensor(
                [0.00, 0.20, 0.05, 0.10, -0.05, 0.02, 0.12, -0.02]
            ),
            "terminal_goal_progress_mask": torch.ones(8),
            "logical_group": ["mixed"] * 4 + ["dense"] * 4,
        }
    )
    original_success = batch["success"].clone()
    output = model(batch)
    raw_loss, pieces = _semantic_comparative_loss(
        output, batch, torch.ones(8), ablation_variant="full"
    )
    raw_loss.backward()

    def has_nonzero_gradient(module: torch.nn.Module) -> bool:
        return any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )

    assert torch.isfinite(raw_loss)
    assert pieces["semantic_mixed_groups_in_batch"] == 1
    assert pieces["semantic_dense_groups_in_batch"] == 1
    torch.testing.assert_close(
        pieces["semantic_comparative_event_raw"],
        pieces["semantic_comparative_mixed_success"]
        + pieces["semantic_comparative_dense_event"],
    )
    torch.testing.assert_close(
        pieces["semantic_comparative_goal_raw"],
        pieces["semantic_comparative_dense_goal"],
    )
    torch.testing.assert_close(
        raw_loss,
        pieces["semantic_comparative_event_raw"]
        + pieces["semantic_comparative_goal_raw"],
    )
    torch.testing.assert_close(raw_loss, pieces["semantic_comparative_raw"])
    assert has_nonzero_gradient(model.terminal_event)
    assert has_nonzero_gradient(model.terminal_goal_progress_mean)
    assert has_nonzero_gradient(model.terminal_context_encoder)
    assert has_nonzero_gradient(model.terminal_residual)
    assert has_nonzero_gradient(model.transition)
    assert has_nonzero_gradient(model.action)
    assert has_nonzero_gradient(model.semantic)
    assert not has_nonzero_gradient(model.candidate_rank)
    assert not has_nonzero_gradient(model.terminal_goal_progress_scale)
    assert not has_nonzero_gradient(model.object_scale)
    assert not has_nonzero_gradient(model.clock)
    assert not has_nonzero_gradient(model.duration_mean)
    assert not has_nonzero_gradient(model.duration_scale)
    assert torch.equal(batch["success"], original_success)


def test_relative_gradient_budget_caps_comparative_head_gradient() -> None:
    head = torch.nn.Linear(2, 1)
    inputs = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    prediction = head(inputs).squeeze(-1)
    proper_loss = prediction.square().mean()
    comparative_loss = 100.0 * prediction.mean()
    parameters = tuple(head.parameters())

    scale = _relative_gradient_budget_scale(
        proper_loss,
        comparative_loss,
        parameters,
    )
    proper_gradients = torch.autograd.grad(
        proper_loss, parameters, retain_graph=True
    )
    scaled_comparative_gradients = torch.autograd.grad(
        scale * comparative_loss, parameters
    )

    def gradient_norm(gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.sqrt(
            torch.stack([gradient.square().sum() for gradient in gradients]).sum()
        )

    proper_norm = gradient_norm(proper_gradients)
    scaled_comparative_norm = gradient_norm(scaled_comparative_gradients)
    assert not scale.requires_grad
    assert 0.0 <= float(scale) <= SEMANTIC_GRADIENT_SCALE_CAP
    assert float(scale) < SEMANTIC_GRADIENT_SCALE_CAP
    assert scaled_comparative_norm <= (
        SEMANTIC_COMPARATIVE_GRADIENT_BUDGET * proper_norm + 1e-6
    )


def test_single_active_union_budget_excludes_scales_and_caps_once() -> None:
    torch.manual_seed(14)
    model = EffectAlignedSharedEventHead().eval()
    active = _semantic_comparative_active_parameters(model)
    active_ids = {id(parameter) for parameter in active}
    expected_modules = (
        model.semantic,
        model.action,
        model.transition,
        model.terminal_context_encoder,
        model.terminal_residual,
        model.terminal_event,
        model.terminal_goal_progress_mean,
    )
    assert active_ids == {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    for excluded_module in (
        model.object_scale,
        model.duration_scale,
        model.terminal_goal_progress_scale,
        model.candidate_rank,
    ):
        assert active_ids.isdisjoint(
            id(parameter) for parameter in excluded_module.parameters()
        )

    proper_loss = sum(parameter.square().mean() for parameter in active)
    comparative_loss = 100.0 * sum(parameter.mean() for parameter in active)
    bounded, scale = _bounded_semantic_comparative_loss(
        proper_loss, comparative_loss, model
    )
    proper_gradients = torch.autograd.grad(
        proper_loss, active, retain_graph=True
    )
    bounded_gradients = torch.autograd.grad(bounded, active)

    def gradient_norm(gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.sqrt(
            torch.stack([gradient.square().sum() for gradient in gradients]).sum()
        )

    proper_norm = gradient_norm(proper_gradients)
    bounded_norm = gradient_norm(bounded_gradients)
    assert not scale.requires_grad
    assert 0.0 < float(scale) < SEMANTIC_GRADIENT_SCALE_CAP
    assert bounded_norm <= (
        SEMANTIC_COMPARATIVE_GRADIENT_BUDGET * proper_norm + 1e-6
    )


def _fixed_terminal_metrics(
    group_specs: list[tuple[str, int, float]],
    *,
    variant: str,
    event_supervised: bool = True,
    goal_supervised: bool = True,
) -> dict[str, object]:
    row_probabilities = torch.tensor(
        [probability for _group, _seed, probability in group_specs]
    ).repeat_interleave(4)

    class FixedTerminalModel(torch.nn.Module):
        def __init__(self, success_probability: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("success_probability", success_probability)
            self.ablation_variant = variant

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            count = len(batch["success"])
            probability = self.success_probability[:count]
            nonterminal = ((1.0 - probability) / 4.0).unsqueeze(-1).expand(-1, 4)
            terminal_probability = torch.cat(
                (nonterminal, probability.unsqueeze(-1)), dim=-1
            )
            zero = probability.new_zeros(count)
            return {
                "terminal_event_logits": terminal_probability.log(),
                "terminal_goal_progress_mean": zero,
                "terminal_goal_progress_log_scale": zero,
                "success_logit": torch.logit(probability),
                "regression_probability": zero,
                "joint_recovery_probability": zero,
            }

    count = len(row_probabilities)
    batch = {
        "success": torch.ones(count),
        "success_mask": torch.ones(count),
        "terminal_max_event_id": torch.full((count,), 4, dtype=torch.long),
        "terminal_event_mask": torch.full((count,), float(event_supervised)),
        "terminal_goal_progress": torch.zeros(count),
        "terminal_goal_progress_mask": torch.full(
            (count,), float(goal_supervised)
        ),
        "post_event_id": torch.zeros(count, dtype=torch.long),
        "post_event_mask": torch.ones(count),
        "current_event_id": torch.zeros(count, dtype=torch.long),
        "recovery": torch.zeros(count),
        "requested_seed": torch.tensor(
            [seed for _group, seed, _probability in group_specs]
        ).repeat_interleave(4),
        "logical_group": [
            group for group, _seed, _probability in group_specs for _ in range(4)
        ],
    }
    return evaluate_terminal_consequences(
        FixedTerminalModel(row_probabilities),
        [batch],
        torch.device("cpu"),
        ablation_variant=variant,
    )


def test_terminal_consequence_metrics_report_strict_proper_macro_se_and_success_nll(
) -> None:
    group_specs = [
        (f"{BODIES[0]}|clean|a", 10, 0.8),
        (f"{BODIES[0]}|clean|b", 11, 0.4),
        (f"{BODIES[1]}|randomized|a", 20, 0.6),
        (f"{BODIES[1]}|randomized|b", 21, 0.2),
    ]
    metrics = _fixed_terminal_metrics(group_specs, variant="success_only")
    group_nll = -np.log(np.asarray([0.8, 0.4, 0.6, 0.2]))
    expected_macro = float(
        np.mean([np.mean(group_nll[:2]), np.mean(group_nll[2:])])
    )
    expected_se = float(
        np.sqrt(
            np.var(group_nll[:2], ddof=1) / 2
            + np.var(group_nll[2:], ddof=1) / 2
        )
        / 2
    )
    strict = metrics["strict_proper"]
    assert strict["logical_decisions"] == 4
    assert strict["body_condition_units"] == 2
    assert strict["independent_requested_seed_clusters"] == 4
    assert strict["selection_rule"] == (
        "source_body_condition_macro_seed_clustered_proper_loss_"
        "one_standard_error"
    )
    assert strict["macro_score"] == pytest.approx(expected_macro)
    assert strict["macro_standard_error"] == pytest.approx(expected_se)
    assert strict["components"] == {
        "success_nll": pytest.approx(expected_macro)
    }
    expected_support = 4 * len(group_specs)
    assert metrics["terminal_success"]["support"] == expected_support
    assert metrics["terminal_success"]["positive"] == expected_support
    assert metrics["terminal_success"]["nll"] == pytest.approx(expected_macro)


def test_success_only_strict_evaluation_does_not_require_event_or_goal_labels(
) -> None:
    metrics = _fixed_terminal_metrics(
        [
            (f"{BODIES[0]}|clean|a", 10, 0.8),
            (f"{BODIES[0]}|clean|b", 11, 0.4),
        ],
        variant="success_only",
        event_supervised=False,
        goal_supervised=False,
    )
    strict = metrics["strict_proper"]
    assert np.isfinite(strict["macro_score"])
    assert set(strict["components"]) == {"success_nll"}
    assert metrics["terminal_event"]["support"] == 0
    assert metrics["terminal_event"]["nll"] is None
    assert metrics["terminal_goal_progress"]["support"] == 0
    assert metrics["terminal_goal_progress"]["student_t3_nll"] is None


def test_no_object_effect_strict_evaluation_does_not_require_goal_labels() -> None:
    metrics = _fixed_terminal_metrics(
        [
            (f"{BODIES[0]}|clean|a", 10, 0.8),
            (f"{BODIES[0]}|clean|b", 11, 0.4),
        ],
        variant="no_object_effect",
        goal_supervised=False,
    )
    strict = metrics["strict_proper"]
    assert np.isfinite(strict["macro_score"])
    assert set(strict["components"]) == {
        "success_nll",
        "terminal_event_nll",
        "terminal_event_ordinal_rps",
    }
    assert metrics["terminal_event"]["support"] == 8
    assert metrics["terminal_goal_progress"]["support"] == 0
    assert metrics["terminal_goal_progress"]["student_t3_nll"] is None


def test_strict_proper_se_clusters_queries_by_requested_seed() -> None:
    one_query_per_seed = _fixed_terminal_metrics(
        [
            (f"{BODIES[0]}|clean|seed10-query0", 10, 0.8),
            (f"{BODIES[0]}|clean|seed20-query0", 20, 0.2),
        ],
        variant="success_only",
    )["strict_proper"]
    two_queries_per_seed = _fixed_terminal_metrics(
        [
            (f"{BODIES[0]}|clean|seed10-query0", 10, 0.8),
            (f"{BODIES[0]}|clean|seed10-query1", 10, 0.8),
            (f"{BODIES[0]}|clean|seed20-query0", 20, 0.2),
            (f"{BODIES[0]}|clean|seed20-query1", 20, 0.2),
        ],
        variant="success_only",
    )["strict_proper"]
    assert one_query_per_seed["independent_requested_seed_clusters"] == 2
    assert two_queries_per_seed["independent_requested_seed_clusters"] == 2
    assert two_queries_per_seed["macro_score"] == pytest.approx(
        one_query_per_seed["macro_score"]
    )
    assert two_queries_per_seed["macro_standard_error"] == pytest.approx(
        one_query_per_seed["macro_standard_error"]
    )


def test_strict_proper_se_fails_closed_with_fewer_than_two_requested_seeds() -> None:
    with pytest.raises(
        FiveBodyContractError,
        match="requires at least two independent requested seeds",
    ):
        _fixed_terminal_metrics(
            [
                (f"{BODIES[0]}|clean|query0", 10, 0.8),
                (f"{BODIES[0]}|clean|query1", 10, 0.4),
            ],
            variant="success_only",
        )


def test_single_failure_class_trains_coherent_success_without_synthetic_positive() -> None:
    torch.manual_seed(10)
    model = EffectAlignedSharedEventHead().train()
    batch = _model_batch(torch.full((4,), 5.0 / 15.0))
    batch.update(
        {
            "post_event_id": torch.zeros(4, dtype=torch.long),
            "post_event_mask": torch.zeros(4),
            "next_event_id": torch.zeros(4, dtype=torch.long),
            "next_event_mask": torch.zeros(4),
            "duration": torch.ones(4),
            "duration_observed": torch.ones(4),
            "duration_mask": torch.zeros(4),
            "success": torch.zeros(4),
            "success_mask": torch.ones(4),
            "recovery": torch.zeros(4),
            "recovery_mask": torch.zeros(4),
            "object_delta": torch.zeros(4, core.OBJECT_DELTA_DIM),
            "object_delta_mask": torch.zeros(4),
        }
    )
    original_success = batch["success"].clone()
    output = model(batch)
    weights = {name: 0.0 for name in core.DEFAULT_LOSS_WEIGHTS}
    weights["success"] = 1.0
    loss, pieces = core.compute_multitask_loss(
        output, batch, sample_weight=torch.ones(4), loss_weights=weights
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(pieces["success"])
    assert model.terminal_event.bias.grad is not None
    assert model.terminal_event.bias.grad[-1] > 0
    assert bool((model.terminal_event.bias.grad[:-1] < 0).any())
    assert torch.equal(batch["success"], original_success)


def test_rank_features_are_only_explicit_predicted_consequences() -> None:
    torch.manual_seed(11)
    model = EffectAlignedSharedEventHead().eval()
    batch = _model_batch(torch.full((4,), 5.0 / 15.0))
    batch["state"][:, :3] = torch.tensor(
        [[0.4, -0.2, 0.1], [0.3, 0.2, -0.1], [0.2, 0.1, 0.3], [0.1, -0.3, 0.2]]
    )
    output = model(batch)
    features = output["candidate_rank_features"]
    assert features.shape == (4, CANDIDATE_RANK_FEATURE_DIM)
    assert features.requires_grad is False

    def block(name: str) -> torch.Tensor:
        start, stop = CANDIDATE_RANK_FEATURE_SCHEMA[name]
        return features[:, start:stop]

    event_level = torch.arange(5, dtype=features.dtype)[None] / 4.0
    current = batch["current_event_id"][:, None]
    post_probability = torch.softmax(output["post_event_logits"], dim=-1)
    next_probability = torch.softmax(output["next_event_logits"], dim=-1)
    terminal_probability = torch.softmax(output["terminal_event_logits"], dim=-1)
    assert torch.allclose(
        block("post_expected_stage_progress")[:, 0],
        (post_probability * event_level).sum(dim=-1),
    )
    positive_next_delta = (
        torch.arange(5)[None] - current
    ).clamp_min(0).to(features) / 4.0
    assert torch.allclose(
        block("next_event_advance_rate")[:, 0],
        (next_probability * positive_next_delta).sum(dim=-1)
        / (1.0 + output["expected_duration_seconds"]),
    )
    assert torch.allclose(
        block("success_probability")[:, 0], torch.sigmoid(output["success_logit"])
    )
    assert torch.allclose(
        block("no_unrecovered_regression_probability")[:, 0],
        1.0
        - (
            output["regression_probability"]
            - output["joint_recovery_probability"]
        ).clamp(0.0, 1.0),
    )
    expected_progress = torch.linalg.vector_norm(batch["state"][:, :3], dim=-1) - (
        torch.linalg.vector_norm(
            batch["state"][:, :3] - output["object_delta_mean"][:, :3], dim=-1
        )
    )
    assert torch.allclose(
        block("short_goal_progress_benefit")[:, 0],
        0.5
        * (
            torch.tanh(expected_progress / GOAL_PROGRESS_NORMALIZATION_METERS) + 1.0
        ),
    )
    assert torch.allclose(
        block("short_goal_progress_uncertainty_risk")[:, 0],
        torch.tanh(
            output["predicted_goal_progress_uncertainty"]
            / GOAL_PROGRESS_NORMALIZATION_METERS
        ).clamp(0.0, 1.0),
    )
    assert torch.allclose(
        block("terminal_expected_stage_progress")[:, 0],
        (terminal_probability * event_level).sum(dim=-1),
    )
    assert torch.allclose(
        block("terminal_goal_progress_benefit")[:, 0],
        0.5
        * (
            torch.tanh(
                output["terminal_goal_progress_mean"]
                / GOAL_PROGRESS_NORMALIZATION_METERS
            )
            + 1.0
        ),
    )
    assert torch.allclose(
        block("terminal_goal_progress_uncertainty_risk")[:, 0],
        torch.tanh(
            output["terminal_goal_progress_std"]
            / GOAL_PROGRESS_NORMALIZATION_METERS
        ).clamp(0.0, 1.0),
    )
    assert bool(((features >= 0.0) & (features <= 1.0)).all())


def test_bounded_utility_is_monotone_and_has_no_raw_world_axis() -> None:
    assert CANDIDATE_RANK_FEATURE_DIM == 9
    assert set(CANDIDATE_RANK_FEATURE_SCHEMA) == set(MONOTONE_BENEFIT_FEATURES) | set(
        MONOTONE_RISK_FEATURES
    )
    assert set(MONOTONE_BENEFIT_FEATURES).isdisjoint(MONOTONE_RISK_FEATURES)
    assert all(
        stop - start == 1 for start, stop in CANDIDATE_RANK_FEATURE_SCHEMA.values()
    )
    assert not any(
        token in name
        for name in CANDIDATE_RANK_FEATURE_SCHEMA
        for token in ("object_delta", "world_axis", "axis_angle")
    )

    utility = InvariantMonotoneConsequenceUtility().eval()
    assert not any(
        isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm))
        for module in utility.modules()
    )
    baseline_features = torch.full((1, CANDIDATE_RANK_FEATURE_DIM), 0.4)
    baseline_score = utility(baseline_features)[0]
    for name in MONOTONE_BENEFIT_FEATURES:
        improved = baseline_features.clone()
        improved[:, CANDIDATE_RANK_FEATURE_SCHEMA[name][0]] += 0.1
        assert utility(improved)[0] > baseline_score
    for name in MONOTONE_RISK_FEATURES:
        riskier = baseline_features.clone()
        riskier[:, CANDIDATE_RANK_FEATURE_SCHEMA[name][0]] += 0.1
        assert utility(riskier)[0] < baseline_score

    # Moving 0.1 terminal probability mass from e4 (level .75) to eK (1.0)
    # raises both expected stage by .025 and coherent success by .1.  Even a
    # utility dominated by stage progress must therefore prefer the shift.
    with torch.no_grad():
        utility.benefit_logits.fill_(-10.0)
        terminal_stage_index = MONOTONE_BENEFIT_FEATURES.index(
            "terminal_expected_stage_progress"
        )
        utility.benefit_logits[terminal_stage_index] = 10.0
    before_success_shift = baseline_features.clone()
    after_success_shift = baseline_features.clone()
    after_success_shift[
        :, CANDIDATE_RANK_FEATURE_SCHEMA["terminal_expected_stage_progress"][0]
    ] += 0.025
    after_success_shift[
        :, CANDIDATE_RANK_FEATURE_SCHEMA["success_probability"][0]
    ] += 0.1
    assert utility(after_success_shift)[0] > utility(before_success_shift)[0]


def test_rank_score_has_explicit_numeric_dt_path_through_clock() -> None:
    model = EffectAlignedSharedEventHead().eval()
    linear = torch.nn.Linear(CANDIDATE_RANK_FEATURE_DIM, 1, bias=False)
    with torch.no_grad():
        linear.weight.zero_()
        advance_start, _advance_stop = CANDIDATE_RANK_FEATURE_SCHEMA[
            "next_event_advance_rate"
        ]
        linear.weight[0, advance_start] = 1.0
        model.clock.body_beta.weight.zero_()
        model.clock.base_tau.weight.zero_()
        model.clock.base_tau.bias.zero_()
        model.clock.candidate.weight.zero_()
        model.clock.candidate.bias.zero_()
        model.clock.candidate.bias[0] = 1.0
        model.duration_mean.weight.zero_()
        model.duration_mean.bias.zero_()
        model.duration_mean.weight[0, 0] = 1.0
    model.candidate_rank = linear
    batch = _model_batch(torch.tensor([1.0 / 15.0, 5.0 / 15.0]))
    batch["state"][1] = batch["state"][0]
    batch["actions"][1] = batch["actions"][0]
    output = model(batch)
    assert output["clock_hidden"][0, 0] != output["clock_hidden"][1, 0]
    assert output["candidate_rank_logit"][0] != output["candidate_rank_logit"][1]


def test_duration_prediction_conditions_on_physical_event_age() -> None:
    model = EffectAlignedSharedEventHead().eval()
    with torch.no_grad():
        for parameter in model.event_age_encoder.parameters():
            parameter.zero_()
        model.event_age_encoder[0].weight[0, 0] = 1.0
        model.event_age_encoder[2].weight[0, 0] = 1.0
        model.duration_mean.weight.zero_()
        model.duration_mean.bias.zero_()
        model.duration_mean.weight[0, 0] = 1.0
    batch = _model_batch(torch.full((2,), 5.0 / 15.0))
    batch["state"][1] = batch["state"][0]
    batch["actions"][1] = batch["actions"][0]
    batch["event_age_seconds"] = torch.tensor([0.0, 2.0])
    output = model(batch)
    assert output["duration_selected_log_mean"][0] != output[
        "duration_selected_log_mean"
    ][1]


def test_terminal_prediction_conditions_on_remaining_action_budget() -> None:
    model = EffectAlignedSharedEventHead().eval()
    with torch.no_grad():
        for parameter in model.terminal_context_encoder.parameters():
            parameter.zero_()
        model.terminal_context_encoder[0].weight[0, 1] = 0.2
        model.terminal_context_encoder[2].weight[
            core.SEMANTIC_DIM, 0
        ] = 1.0
        model.terminal_goal_progress_mean.weight.zero_()
        model.terminal_goal_progress_mean.bias.zero_()
        model.terminal_goal_progress_mean.weight[0, 0] = 1.0
        model.terminal_event.weight.zero_()
        model.terminal_event.bias.zero_()
        model.terminal_event.weight[-1, 0] = 1.0
        model.terminal_recovery.weight.zero_()
        model.terminal_recovery.bias.zero_()
        model.terminal_recovery.weight[0, 0] = 1.0
    batch = _model_batch(torch.full((2,), 5.0 / 15.0))
    batch["state"][1] = batch["state"][0]
    batch["actions"][1] = batch["actions"][0]
    batch["remaining_action_budget"] = torch.tensor([10.0, 200.0])
    output = model(batch)
    assert output["terminal_goal_progress_mean"][0] != output[
        "terminal_goal_progress_mean"
    ][1]
    assert output["success_logit"][0] != output["success_logit"][1]
    assert output["recovery_logit"][0] != output["recovery_logit"][1]
    terminal_probability = torch.softmax(output["terminal_event_logits"], dim=-1)
    torch.testing.assert_close(
        torch.sigmoid(output["success_logit"]), terminal_probability[:, -1]
    )


@pytest.mark.parametrize(
    ("context_channel", "low", "high"),
    ((0, 0.0, 9.0), (1, 10.0, 200.0)),
)
def test_terminal_candidate_difference_changes_with_shared_horizon_context(
    context_channel: int,
    low: float,
    high: float,
) -> None:
    """FiLM must break the additive head's candidate-difference invariance."""

    model = EffectAlignedSharedEventHead().eval()
    with torch.no_grad():
        for parameter in model.terminal_context_encoder.parameters():
            parameter.zero_()
        for parameter in model.terminal_residual.parameters():
            parameter.zero_()
        model.terminal_context_encoder[0].weight[0, context_channel] = 0.2
        # The first half of the FiLM vector modulates transitioned features.
        model.terminal_context_encoder[2].weight[0, 0] = 1.0
        model.terminal_goal_progress_mean.weight.zero_()
        model.terminal_goal_progress_mean.bias.zero_()
        model.terminal_goal_progress_mean.weight[0, 0] = 1.0

    candidate_a = torch.zeros(core.SEMANTIC_DIM)
    candidate_b = torch.zeros(core.SEMANTIC_DIM)
    candidate_a[0] = 1.0
    candidate_b[0] = -1.0
    transitioned = torch.stack(
        (candidate_a, candidate_b, candidate_a, candidate_b)
    )
    base_context = torch.tensor([0.4, 175.0]).log1p()
    low_context = base_context.clone()
    high_context = base_context.clone()
    low_context[context_channel] = torch.tensor(low).log1p()
    high_context[context_channel] = torch.tensor(high).log1p()
    context = torch.stack(
        (low_context, low_context, high_context, high_context)
    )
    terminal_hidden = model._terminal_hidden(transitioned, context)
    prediction = model.terminal_goal_progress_mean(terminal_hidden).squeeze(-1)
    low_difference = prediction[0] - prediction[1]
    high_difference = prediction[2] - prediction[3]

    assert torch.isfinite(prediction).all()
    assert abs(float((high_difference - low_difference).detach())) > 1e-4


def test_rank_ensemble_uses_mean_minus_quarter_population_std() -> None:
    scores = torch.tensor(
        [
            [0.6, 1.0, 0.4, 0.0],
            [0.6, 1.0, 0.4, 0.0],
            [0.6, 1.0, 0.4, 0.0],
            [0.6, 0.0, 0.4, 0.0],
            [0.6, 0.0, 0.4, 0.0],
        ]
    )
    assert scores.mean(0)[0] == scores.mean(0)[1]
    aggregate = aggregate_risk_adjusted_rank_scores(scores)
    expected = scores.mean(dim=0) - 0.25 * scores.std(dim=0, correction=0)
    assert aggregate.shape == (4,)
    torch.testing.assert_close(aggregate, expected)
    assert int(aggregate.argmax()) == 0
    assert aggregate[0] > aggregate[1]
    assert EPISTEMIC_RANK_RISK_WEIGHT == 0.25
    assert RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT["population_std_correction"] == 0
    assert RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT[
        "within_member_candidate_standardization"
    ] is False
    constant_only = aggregate_risk_adjusted_rank_scores(torch.ones(5, 4))
    assert torch.equal(constant_only, torch.ones(4))


def _effect_loss(
    *,
    success: list[float],
    terminal_event: list[int],
    terminal_goal_progress: list[float],
    scores: list[float],
    variant: str = "full",
) -> dict[str, torch.Tensor]:
    score = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    output = {
        "candidate_rank_logit": score,
        "success_logit": torch.zeros(4, requires_grad=True),
        "object_delta_mean": torch.zeros(4, core.OBJECT_DELTA_DIM, requires_grad=True),
        "object_delta_log_scale": torch.zeros(
            4, core.OBJECT_DELTA_DIM, requires_grad=True
        ),
    }
    batch = {
        "success": torch.tensor(success),
        "state": torch.zeros(4, core.STATE_DIM),
        "post_event_id": torch.zeros(4, dtype=torch.long),
        "terminal_max_event_id": torch.tensor(terminal_event, dtype=torch.long),
        "terminal_event_mask": torch.ones(4),
        "terminal_goal_progress": torch.tensor(terminal_goal_progress),
        "terminal_goal_progress_mask": torch.ones(4),
        "object_delta": torch.zeros(4, core.OBJECT_DELTA_DIM),
        "object_delta_mask": torch.ones(4),
        "action_available": torch.ones(4),
        "logical_group": ["piper|clean|one"] * 4,
    }
    _total, pieces = _candidate_rank_loss(
        output,
        batch,
        torch.ones(4),
        ablation_variant=variant,
    )
    return pieces


def test_mixed_success_uses_group_listwise_success_probability_mass() -> None:
    good = _effect_loss(
        success=[0, 1, 0, 0],
        terminal_event=[4, 0, 3, 2],
        terminal_goal_progress=[1000.0, -1000.0, 10.0, 5.0],
        scores=[0.0, 5.0, 0.0, 0.0],
    )
    bad = _effect_loss(
        success=[0, 1, 0, 0],
        terminal_event=[4, 0, 3, 2],
        terminal_goal_progress=[1000.0, -1000.0, 10.0, 5.0],
        scores=[5.0, 0.0, 0.0, 0.0],
    )
    assert good["group_listwise_success_mass_balanced_rank"] < bad[
        "group_listwise_success_mass_balanced_rank"
    ]
    assert good["all_failure_dense_soft_listwise_balanced_rank"] == 0.0
    assert good["dense_rank_effective_weight"] == DENSE_FAILURE_RANK_WEIGHT


def test_all_failure_dense_target_is_true_lexicographic_terminal_value() -> None:
    # Candidate 0 has an extreme geometric value, but candidate 1 reached the
    # later terminal event.  No fixed 100/10 scalar can reverse that ordering.
    good = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 2, 1, 1],
        terminal_goal_progress=[1000.0, -1000.0, 0.0, 0.0],
        scores=[0.0, 5.0, 0.0, 0.0],
    )
    bad = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 2, 1, 1],
        terminal_goal_progress=[1000.0, -1000.0, 0.0, 0.0],
        scores=[5.0, 0.0, 0.0, 0.0],
    )
    assert good["all_failure_dense_soft_listwise_balanced_rank"] < bad[
        "all_failure_dense_soft_listwise_balanced_rank"
    ]
    assert torch.allclose(
        good["candidate_ranking_balanced_rank"],
        DENSE_ONLY_RANK_WEIGHT
        * good["all_failure_dense_soft_listwise_balanced_rank"],
    )
    assert good["dense_rank_effective_weight"] == DENSE_ONLY_RANK_WEIGHT
    no_object = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 1, 1, 1],
        terminal_goal_progress=[0.0, 1000.0, 2.0, 1.0],
        scores=[0.0, 5.0, 0.0, 0.0],
        variant="no_object_effect",
    )
    assert no_object["all_failure_dense_soft_listwise_balanced_rank"] == 0.0
    assert no_object["all_failure_uninformative_groups_in_batch"] == 1
    no_object_event_difference = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 2, 1, 1],
        terminal_goal_progress=[0.0, 1000.0, 2.0, 1.0],
        scores=[0.0, 5.0, 0.0, 0.0],
        variant="no_object_effect",
    )
    assert no_object_event_difference[
        "all_failure_dense_soft_listwise_balanced_rank"
    ] > 0.0


def test_all_failure_without_candidate_consequence_difference_has_no_rank_loss() -> None:
    pieces = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[2, 2, 2, 2],
        terminal_goal_progress=[0.3, 0.3, 0.3, 0.3],
        scores=[0.0, 5.0, -2.0, 1.0],
    )
    assert pieces["all_failure_dense_soft_listwise_balanced_rank"] == 0.0
    assert pieces["all_failure_dense_groups_in_batch"] == 0
    assert pieces["all_failure_uninformative_groups_in_batch"] == 1


def test_effect_bootstrap_uses_one_plus_poisson_for_every_mixed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "logical_group": "piper|clean|mixed",
            "success": float(index == 1),
            "success_mask": 1.0,
        }
        for index in range(4)
    ]
    rows += [
        {
            "logical_group": "piper|clean|failure",
            "success": 0.0,
            "success_mask": 1.0,
        }
        for _index in range(4)
    ]
    monkeypatch.setattr(
        core,
        "logical_group_bootstrap_weights",
        lambda groups, *, members, seed: np.zeros((members, len(groups)), np.float32),
    )
    weights, audit = effect_preserving_group_bootstrap_weights(
        rows, members=5, seed=17
    )
    assert weights.shape == (5, 8)
    assert all(item["positive_rows_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["negative_rows_with_nonzero_weight"] > 0 for item in audit)
    assert np.all(weights[:, :4] == 1.0)
    assert np.all(weights[:, 4:] == 0.0)
    assert all(item["mixed_success_groups_with_nonzero_weight"] == 1 for item in audit)
    assert all(item["mixed_success_groups_total"] == 1 for item in audit)
    assert all(item["mixed_weight_minimum"] == 1 for item in audit)
    assert all(item["deterministic_mixed_group_repairs"] == 0 for item in audit)


def test_proper_bootstrap_repairs_only_missing_binary_outcome_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "logical_group": "piper|clean|mixed",
            "success": float(index == 1),
            "success_mask": 1.0,
        }
        for index in range(4)
    ]
    rows += [
        {
            "logical_group": "piper|clean|failure",
            "success": 0.0,
            "success_mask": 1.0,
        }
        for _index in range(4)
    ]
    base = np.zeros((5, len(rows)), dtype=np.float32)
    base[1:, :4] = 2.0
    base[1:, 4:] = 1.0
    monkeypatch.setattr(
        core,
        "logical_group_bootstrap_weights",
        lambda groups, *, members, seed: base.copy(),
    )
    weights, audit = proper_outcome_preserving_group_bootstrap_weights(
        rows, members=5, seed=17
    )
    assert weights.shape == (5, 8)
    assert np.all(weights[0, :4] == 1.0)
    assert np.all(weights[0, 4:] == 0.0)
    assert np.array_equal(weights[1:], base[1:])
    assert audit[0]["deterministic_mixed_group_repairs"] == 1
    assert all(
        item["deterministic_mixed_group_repairs"] == 0 for item in audit[1:]
    )
    assert all(item["positive_rows_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["negative_rows_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["positive_groups_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["negative_groups_with_nonzero_weight"] > 0 for item in audit)


def test_dense_only_bootstraps_keep_real_rank_and_single_class_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for group_index in range(2):
        rows.extend(
            {
                "logical_group": f"piper|clean|dense-only-{group_index}",
                "success": 0.0,
                "success_mask": 1.0,
                "terminal_max_event_id": int(index == group_index),
                "terminal_event_mask": 1.0,
                "terminal_goal_progress": float(index) / 10.0,
                "terminal_goal_progress_mask": 1.0,
            }
            for index in range(4)
        )
    original_success = [row["success"] for row in rows]
    monkeypatch.setattr(
        core,
        "logical_group_bootstrap_weights",
        lambda groups, *, members, seed: np.zeros((members, len(groups)), np.float32),
    )
    rank_weights, rank_audit = effect_preserving_group_bootstrap_weights(
        rows, members=5, seed=23
    )
    assert np.all(np.count_nonzero(rank_weights, axis=1) == 4)
    assert all(item["mixed_success_groups_total"] == 0 for item in rank_audit)
    assert all(item["informative_dense_groups_total"] == 2 for item in rank_audit)
    assert all(
        item["rank_supervision_groups_with_nonzero_weight"] == 1
        for item in rank_audit
    )
    assert all(item["deterministic_rank_group_repairs"] == 1 for item in rank_audit)

    proper_weights, proper_audit = proper_outcome_preserving_group_bootstrap_weights(
        rows, members=5, seed=23
    )
    assert np.all(np.count_nonzero(proper_weights, axis=1) == 4)
    assert all(item["positive_class_present"] is False for item in proper_audit)
    assert all(item["negative_class_present"] is True for item in proper_audit)
    assert all(item["positive_rows_with_nonzero_weight"] == 0 for item in proper_audit)
    assert all(item["negative_rows_with_nonzero_weight"] == 4 for item in proper_audit)
    assert all(item["deterministic_outcome_group_repairs"] == 1 for item in proper_audit)
    assert all(item["success_labels_synthesized"] == 0 for item in proper_audit)
    assert [row["success"] for row in rows] == original_success


def test_proper_bootstrap_restores_separate_real_classes_without_mixed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for group, success in (
        ("piper|clean|all-failure", 0.0),
        ("piper|clean|all-success", 1.0),
    ):
        rows.extend(
            {
                "logical_group": group,
                "success": success,
                "success_mask": 1.0,
            }
            for _candidate in range(4)
        )
    original_success = [row["success"] for row in rows]
    monkeypatch.setattr(
        core,
        "logical_group_bootstrap_weights",
        lambda groups, *, members, seed: np.zeros((members, len(groups)), np.float32),
    )
    weights, audit = proper_outcome_preserving_group_bootstrap_weights(
        rows, members=5, seed=29
    )
    assert np.all(np.count_nonzero(weights, axis=1) == 8)
    assert all(item["mixed_success_groups_total"] == 0 for item in audit)
    assert all(item["positive_rows_with_nonzero_weight"] == 4 for item in audit)
    assert all(item["negative_rows_with_nonzero_weight"] == 4 for item in audit)
    assert all(item["deterministic_outcome_group_repairs"] == 2 for item in audit)
    assert all(item["success_labels_synthesized"] == 0 for item in audit)
    assert [row["success"] for row in rows] == original_success


def test_dense_goal_progress_uses_frozen_soft_target_inside_max_event() -> None:
    assert DENSE_GOAL_PROGRESS_TEMPERATURE_METERS == 0.02
    scores = torch.zeros(4, requires_grad=True)
    event = torch.tensor([2.0, 2.0, 1.0, 0.0])
    progress = torch.tensor([0.0, 0.02, 1000.0, 1000.0])
    loss = _dense_soft_listwise_loss(
        scores, event, progress, ablation_variant="full"
    )
    loss.backward()
    expected_target = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    assert torch.allclose(
        scores.grad,
        torch.tensor(
            [
                0.25 - expected_target[0],
                0.25 - expected_target[1],
                0.25,
                0.25,
            ]
        ),
    )
    no_object_scores = torch.tensor([0.0, 0.0, 100.0, 100.0])
    no_object = _dense_soft_listwise_loss(
        no_object_scores,
        event,
        progress,
        ablation_variant="no_object_effect",
    )
    expected_no_object = torch.logsumexp(no_object_scores, dim=0) - no_object_scores[
        :2
    ].mean()
    assert torch.allclose(no_object, expected_no_object)


def test_macro_balanced_rank_batches_include_mixed_without_group_duplicates() -> None:
    rows: list[dict[str, object]] = []
    weights: dict[str, float] = {}
    specifications = []
    for ordinal, (body, condition, event) in enumerate(
        (
            ("piper", "clean", 0),
            ("piper", "randomized", 1),
            ("franka", "clean", 2),
            ("franka", "randomized", 3),
        )
    ):
        specifications.append((f"{body}|{condition}|mixed-{ordinal}", body, condition, event, True))
    for ordinal in range(8):
        body = ("piper", "franka")[ordinal % 2]
        condition = ("clean", "randomized")[(ordinal // 2) % 2]
        specifications.append(
            (f"{body}|{condition}|dense-{ordinal}", body, condition, ordinal % 4, False)
        )
    for group, body, _condition, event, mixed in specifications:
        weights[group] = 1.0
        for candidate in range(4):
            rows.append(
                {
                    "logical_group": group,
                    "body": body,
                    "candidate_index": candidate,
                    "current_event_id": event,
                    "success": float(mixed and candidate == 0),
                    "success_mask": 1.0,
                    "terminal_max_event_id": candidate,
                    "terminal_event_mask": 1.0,
                    "terminal_goal_progress": candidate / 10.0,
                    "terminal_goal_progress_mask": 1.0,
                }
            )
    uninformative_group = "piper|clean|dense-uninformative"
    weights[uninformative_group] = 1.0
    for candidate in range(4):
        rows.append(
            {
                "logical_group": uninformative_group,
                "body": "piper",
                "candidate_index": candidate,
                "current_event_id": 0,
                "success": 0.0,
                "success_mask": 1.0,
                "terminal_max_event_id": 1,
                "terminal_event_mask": 1.0,
                "terminal_goal_progress": 0.25,
                "terminal_goal_progress_mask": 1.0,
            }
        )
    sampler = MacroBalancedRankDecisionBatchSampler(
        rows,
        batch_size=32,
        seed=19,
        positive_group_weight=weights,
        ablation_variant="full",
    )
    assert len(sampler) == 2
    assert uninformative_group not in sampler.decisions
    for batch in sampler:
        groups = [str(rows[index]["logical_group"]) for index in batch[::4]]
        assert len(groups) == len(set(groups))
        assert any("|mixed-" in group for group in groups)
        assert all(
            [int(rows[index]["candidate_index"]) for index in batch[offset : offset + 4]]
            == [0, 1, 2, 3]
            for offset in range(0, len(batch), 4)
        )


def test_macro_rank_sampler_can_train_from_informative_dense_only() -> None:
    rows: list[dict[str, object]] = []
    weights: dict[str, float] = {}
    for group_index in range(3):
        group = f"piper|clean|dense-only-{group_index}"
        weights[group] = 1.0
        for candidate in range(4):
            rows.append(
                {
                    "logical_group": group,
                    "body": "piper",
                    "candidate_index": candidate,
                    "current_event_id": 0,
                    "success": 0.0,
                    "success_mask": 1.0,
                    "terminal_max_event_id": int(candidate == group_index % 4),
                    "terminal_event_mask": 1.0,
                    "terminal_goal_progress": candidate / 10.0,
                    "terminal_goal_progress_mask": 1.0,
                }
            )
    tie_group = "piper|clean|dense-only-tie"
    weights[tie_group] = 1.0
    for candidate in range(4):
        rows.append(
            {
                "logical_group": tie_group,
                "body": "piper",
                "candidate_index": candidate,
                "current_event_id": 0,
                "success": 0.0,
                "success_mask": 1.0,
                "terminal_max_event_id": 1,
                "terminal_event_mask": 1.0,
                "terminal_goal_progress": 0.2,
                "terminal_goal_progress_mask": 1.0,
            }
        )
    sampler = MacroBalancedRankDecisionBatchSampler(
        rows,
        batch_size=8,
        seed=31,
        positive_group_weight=weights,
        ablation_variant="full",
    )
    assert sampler.mixed_groups == []
    assert len(sampler.dense_groups) == 3
    assert tie_group not in sampler.decisions
    observed = []
    for batch in sampler:
        groups = [str(rows[index]["logical_group"]) for index in batch[::4]]
        assert batch
        assert len(groups) == len(set(groups))
        assert all(sampler.kinds[group] == "dense" for group in groups)
        assert all(
            [int(rows[index]["candidate_index"]) for index in batch[offset:offset + 4]]
            == [0, 1, 2, 3]
            for offset in range(0, len(batch), 4)
        )
        observed.extend(groups)
    assert set(observed) == set(weights) - {tie_group}


class _FixedRankModel(torch.nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.ablation_variant = variant

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = torch.tensor([0.0, 3.0, 2.0, 1.0], device=batch["candidate_index"].device)
        return {"candidate_rank_logit": values[batch["candidate_index"].long()]}


def _ranking_rows() -> list[dict[str, object]]:
    rows = []
    for group, success, terminal, goal_progress in (
        (
            "piper|clean|mixed",
            [0, 1, 0, 0],
            [4, 0, 3, 2],
            [0.0, 0.1, 0.2, 0.3],
        ),
        (
            "piper|clean|failure",
            [0, 0, 0, 0],
            [0, 3, 2, 1],
            [0.0, 0.1, 0.2, 0.3],
        ),
        (
            "piper|clean|failure-tie",
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            [0.1, 0.1, 0.1, 0.1],
        ),
    ):
        for candidate in range(4):
            rows.append(
                {
                    "logical_group": group,
                    "body": "piper",
                    "candidate_index": np.int64(candidate),
                    "success": np.float32(success[candidate]),
                    "terminal_max_event_id": np.int64(terminal[candidate]),
                    "terminal_stage_progress": np.float32(
                        1.0 if success[candidate] else terminal[candidate] / 4.0
                    ),
                    "terminal_goal_distance": np.float32(
                        1.0 - goal_progress[candidate]
                    ),
                    "terminal_goal_progress": np.float32(
                        goal_progress[candidate]
                    ),
                }
            )
    return rows


def test_ranking_evaluation_separates_success_change_from_dense_progress() -> None:
    loader = torch.utils.data.DataLoader(
        core.TransitionDataset(_ranking_rows(), {"piper": 0}),
        batch_size=8,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    result = evaluate_candidate_ranking(
        _FixedRankModel("full"), loader, torch.device("cpu")
    )
    assert result["mixed_success_decisions"] == 1
    assert result["mixed_success_selection_accuracy"] == 1.0
    assert result["mixed_success_pairwise_accuracy"] == 1.0
    assert result["dense_progress_decisions"] == 1
    assert result["dense_uninformative_decisions"] == 1
    assert result["dense_progress_selection_accuracy"] == 1.0
    assert result["dense_progress_pairwise_accuracy"] == 1.0
    success_only = evaluate_candidate_ranking(
        _FixedRankModel("success_only"), loader, torch.device("cpu")
    )
    assert success_only["dense_progress_decisions"] == 0
    assert success_only["dense_progress_pairwise_accuracy"] is None


def test_checkpoint_selection_prefers_mixed_success_before_dense_diagnostics() -> None:
    base = {
        "macro_one_deviation_branch_success_gain": 0.1,
        "macro_mixed_success_pairwise_accuracy": 0.8,
        "macro_dense_progress_selection_accuracy": 0.5,
        "macro_dense_progress_pairwise_accuracy": 0.5,
    }
    stronger_mixed = {
        **base,
        "macro_mixed_success_selection_accuracy": 0.8,
    }
    stronger_dense_only = {
        **base,
        "macro_mixed_success_selection_accuracy": 0.7,
        "macro_dense_progress_selection_accuracy": 1.0,
        "macro_dense_progress_pairwise_accuracy": 1.0,
    }
    assert candidate_checkpoint_selection_key(
        stronger_mixed, 1.0, 100
    ) < candidate_checkpoint_selection_key(stronger_dense_only, 0.1, 100)


def test_checkpoint_selection_uses_dense_evidence_when_mixed_is_absent() -> None:
    weak_dense = {
        "macro_one_deviation_branch_success_gain": 0.0,
        "macro_mixed_success_selection_accuracy": None,
        "macro_mixed_success_pairwise_accuracy": None,
        "macro_dense_progress_selection_accuracy": 0.5,
        "macro_dense_progress_pairwise_accuracy": 0.5,
    }
    strong_dense = {
        **weak_dense,
        "macro_dense_progress_selection_accuracy": 0.8,
        "macro_dense_progress_pairwise_accuracy": 0.7,
    }
    assert candidate_checkpoint_selection_key(
        strong_dense, 100.0, 3000
    ) < candidate_checkpoint_selection_key(weak_dense, 0.1, 100)


def test_calibration_guard_selects_rank_only_inside_proper_one_se_set() -> None:
    records = [
        {
            "step": 100,
            "mean_member_strict_proper_score": 1.0,
            "conservative_strict_proper_standard_error": 0.1,
            "selection_key": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100),
            "ensemble_candidate_ranking": {
                "mixed_success_decisions": 1,
                "dense_progress_decisions": 0,
            },
        },
        {
            "step": 200,
            "mean_member_strict_proper_score": 1.08,
            "conservative_strict_proper_standard_error": 0.9,
            "selection_key": (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200),
            "ensemble_candidate_ranking": {
                "mixed_success_decisions": 1,
                "dense_progress_decisions": 0,
            },
        },
        {
            "step": 300,
            "mean_member_strict_proper_score": 1.2,
            "conservative_strict_proper_standard_error": 0.9,
            "selection_key": (-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300),
            "ensemble_candidate_ranking": {
                "mixed_success_decisions": 1,
                "dense_progress_decisions": 0,
            },
        },
    ]
    selected, audit = select_calibration_guarded_checkpoint(records)
    assert selected["step"] == 200
    assert audit["comparative_validation_evidence"] is True
    assert audit["eligible_steps"] == [100, 200]
    assert audit["eligible_threshold"] == pytest.approx(1.1)
    assert audit["heldout_rows_used"] == 0


def test_calibration_guard_without_comparative_evidence_uses_strict_proper_step(
) -> None:
    records = [
        {
            "step": step,
            "mean_member_strict_proper_score": score,
            "conservative_strict_proper_standard_error": 0.1,
            "selection_key": rank_key,
            "ensemble_candidate_ranking": {
                "mixed_success_decisions": 0,
                "dense_progress_decisions": 0,
            },
        }
        for step, score, rank_key in (
            (100, 1.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100)),
            (200, 1.0, (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200)),
            (300, 1.1, (-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300)),
        )
    ]
    selected, audit = select_calibration_guarded_checkpoint(records)
    assert selected["step"] == 100
    assert audit["comparative_validation_evidence"] is False
    assert audit["eligible_steps"] == [100]
    assert audit["selected_score"] == pytest.approx(1.0)


def _checkpoint_selection_record(
    step: int,
    *,
    mixed_support: int = 1,
    dense_support: int = 2,
) -> dict[str, object]:
    return {
        "step": step,
        "mean_member_strict_proper_score": 1.0,
        "conservative_strict_proper_standard_error": 0.1,
        "selection_key": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, step],
        "ensemble_candidate_ranking": {
            "mixed_success_decisions": mixed_support,
            "dense_progress_decisions": dense_support,
        },
    }


@pytest.mark.parametrize(
    "field",
    ("mean_member_strict_proper_score", "standard_error", "selection_key"),
)
def test_calibration_guard_rejects_nonfinite_selection_values(field: str) -> None:
    record = _checkpoint_selection_record(100)
    if field == "standard_error":
        record["conservative_strict_proper_standard_error"] = float("inf")
    elif field == "selection_key":
        record["selection_key"][5] = float("nan")
    else:
        record[field] = float("nan")
    with pytest.raises(FiveBodyContractError, match="violates the proper/rank contract"):
        select_calibration_guarded_checkpoint([record])


def test_calibration_guard_rejects_selection_key_step_mismatch() -> None:
    record = _checkpoint_selection_record(100)
    record["selection_key"][-1] = 200
    with pytest.raises(FiveBodyContractError, match="violates the proper/rank contract"):
        select_calibration_guarded_checkpoint([record])


def test_calibration_guard_rejects_duplicate_steps() -> None:
    with pytest.raises(FiveBodyContractError, match="steps are not unique"):
        select_calibration_guarded_checkpoint(
            [
                _checkpoint_selection_record(100),
                _checkpoint_selection_record(100),
            ]
        )


@pytest.mark.parametrize(
    "support_name", ("mixed_success_decisions", "dense_progress_decisions")
)
def test_calibration_guard_rejects_comparative_support_drift(
    support_name: str,
) -> None:
    first = _checkpoint_selection_record(100)
    second = _checkpoint_selection_record(200)
    second["ensemble_candidate_ranking"][support_name] += 1
    with pytest.raises(FiveBodyContractError, match="support changed across steps"):
        select_calibration_guarded_checkpoint([first, second])


def test_ensemble_seeds_must_be_five_distinct_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert validate_ensemble_seeds([11, 12, 13, 14, 15]) == (
        11,
        12,
        13,
        14,
        15,
    )
    with pytest.raises(FiveBodyContractError, match="five distinct integers"):
        validate_ensemble_seeds([11, 12, 13, 14, 14])
    with pytest.raises(FiveBodyContractError, match="five distinct integers"):
        validate_ensemble_seeds([11, 12, 13, 14])
    with pytest.raises(FiveBodyContractError, match="five distinct integers"):
        validate_ensemble_seeds([11, 12, 13, 14, True])
    monkeypatch.setattr(
        trainer_entry,
        "parse_args",
        lambda: argparse.Namespace(ensemble_seeds=[11, 12, 13, 14, 14]),
    )
    monkeypatch.setattr(
        trainer_entry,
        "load_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("binding was opened before ensemble seed validation")
        ),
    )
    with pytest.raises(FiveBodyContractError, match="five distinct integers"):
        trainer_entry.main()


def test_ablation_variants_change_only_declared_score_features() -> None:
    assert MODEL_FAMILY == "terminal_consequence_utility_shared_event_head_v10"
    batch = _model_batch(torch.full((4,), 5.0 / 15.0))
    success_only = EffectAlignedSharedEventHead("success_only").eval()(batch)
    torch.testing.assert_close(
        success_only["candidate_rank_logit"],
        torch.sigmoid(success_only["success_logit"]),
    )
    no_time = EffectAlignedSharedEventHead("no_time_duration").eval()(batch)
    duration_start, duration_stop = CANDIDATE_RANK_FEATURE_SCHEMA[
        "next_event_advance_rate"
    ]
    assert torch.count_nonzero(
        no_time["candidate_rank_features"][:, duration_start:duration_stop]
    ) == 0
    no_time_pair = _model_batch(torch.full((2,), 5.0 / 15.0))
    no_time_pair["state"][1] = no_time_pair["state"][0]
    no_time_pair["actions"][1] = no_time_pair["actions"][0]
    no_time_pair["event_age_seconds"] = torch.tensor([0.0, 9.0])
    no_time_pair["remaining_action_budget"] = torch.tensor([5.0, 200.0])
    no_time_pair_output = EffectAlignedSharedEventHead(
        "no_time_duration"
    ).eval()(no_time_pair)
    terminal_start = CANDIDATE_RANK_FEATURE_SCHEMA[
        "terminal_expected_stage_progress"
    ][0]
    terminal_stop = CANDIDATE_RANK_FEATURE_SCHEMA[
        "terminal_goal_progress_uncertainty_risk"
    ][1]
    torch.testing.assert_close(
        no_time_pair_output["candidate_rank_features"][0, terminal_start:terminal_stop],
        no_time_pair_output["candidate_rank_features"][1, terminal_start:terminal_stop],
    )
    no_object = EffectAlignedSharedEventHead("no_object_effect").eval()(batch)
    object_feature_names = (
        "short_goal_progress_benefit",
        "short_goal_progress_uncertainty_risk",
        "terminal_goal_progress_benefit",
        "terminal_goal_progress_uncertainty_risk",
    )
    for name in object_feature_names:
        start, stop = CANDIDATE_RANK_FEATURE_SCHEMA[name]
        assert torch.count_nonzero(
            no_object["candidate_rank_features"][:, start:stop]
        ) == 0
    full = EffectAlignedSharedEventHead("full").eval()(batch)
    assert torch.count_nonzero(
        full["candidate_rank_features"][:, duration_start:duration_stop]
    ) > 0
    for name in object_feature_names:
        start, stop = CANDIDATE_RANK_FEATURE_SCHEMA[name]
        values = full["candidate_rank_features"][:, start:stop]
        assert bool(((values >= 0.0) & (values <= 1.0)).all())
        if name.endswith("uncertainty_risk"):
            assert torch.count_nonzero(values) > 0
    assert set(ABLATION_VARIANTS) == {
        "success_only", "no_time_duration", "no_object_effect", "full"
    }
    assert ablation_contract("no_object_effect")[
        "object_effect_loss_and_rank_target_enabled"
    ] is False
    assert ablation_contract("success_only")[
        "terminal_horizon_context_enabled"
    ] is True
    components = {
        "post_event_macro_error_ratio": 1.0,
        "next_event_macro_error_ratio": 1.0,
        "observed_duration_mae_ratio": 1.0,
        "success_brier_ratio": 1.0,
        "object_rmse_ratio": 1.0,
    }
    assert set(ablation_selection_components(components, "success_only")) == {
        "success_brier_ratio"
    }
    assert "observed_duration_mae_ratio" not in ablation_selection_components(
        components, "no_time_duration"
    )
    assert "object_rmse_ratio" not in ablation_selection_components(
        components, "no_object_effect"
    )
    assert checkpoint_candidate_rank_contract("no_time_duration")[
        "dt_has_numeric_score_path"
    ] is False
    assert summary_candidate_rank_contract("success_only")[
        "pairwise_rank_loss_enabled"
    ] is False
    assert summary_candidate_rank_contract("full")[
        "group_listwise_success_mass_loss_enabled"
    ] is True
    assert summary_candidate_rank_contract("full")[
        "dt_has_numeric_score_path"
    ] is True
    full_contract = checkpoint_candidate_rank_contract("full")
    assert full_contract["direct_transitioned_or_clock_hidden_rank_path"] is False
    assert full_contract["rank_inputs_are_detached_consequence_predictions"] is True
    assert full_contract["utility_rank_loss_updates_semantic_action_transition"] is False
    assert full_contract["utility_rank_loss_updates_consequence_predictors"] is False
    assert full_contract["semantic_comparative_loss_updates_terminal_predictors"] is True
    assert full_contract[
        "semantic_comparative_gradient_budget_relative_to_active_union_proper"
    ] == SEMANTIC_COMPARATIVE_GRADIENT_BUDGET
    assert full_contract["semantic_gradient_scale_cap"] == SEMANTIC_GRADIENT_SCALE_CAP
    assert full_contract["semantic_comparative_gradient_budget_scope"] == (
        "single_active_union_semantic_action_transition_terminal_trunk_and_location_heads"
    )
    assert full_contract["semantic_comparative_scale_heads_excluded"] is True
    assert full_contract["semantic_comparative_gradient_cap_applications"] == 1
    assert full_contract["terminal_context_fusion"] == (
        "bounded_horizon_conditioned_film_residual_trunk"
    )
    assert full_contract[
        "terminal_candidate_relative_predictions_condition_on_horizon"
    ] is True
    assert full_contract["terminal_film_modulation_bound"] == (
        TERMINAL_FILM_MODULATION_BOUND
    )
    assert full_contract["dense_only_listwise_weight"] == DENSE_ONLY_RANK_WEIGHT
    assert full_contract["world_and_utility_gradient_clipping_are_separate"] is True
    assert full_contract["checkpoint_selection_calibration_guard"] == (
        "source_body_condition_macro_seed_clustered_strict_proper_score_"
        "one_standard_error"
    )
    assert full_contract["strict_proper_components"] == [
        "success_binary_nll",
        "terminal_event_categorical_nll_weight_0.5",
        "terminal_event_ordinal_ranked_probability_score_weight_0.25",
        "terminal_goal_student_t3_nll_weight_0.5",
    ]
    assert full_contract["terminal_stage_progress_loss"] == (
        "terminal_event_cdf_ranked_probability_score_weight_0.25"
    )
    assert full_contract["raw_world_frame_object_axes_in_rank_input"] is False
    assert full_contract["cross_feature_layer_normalization"] is False
    assert full_contract["feature_schema"] == {
        name: list(bounds) for name, bounds in CANDIDATE_RANK_FEATURE_SCHEMA.items()
    }


def _complete_ablation_audit() -> dict[str, object]:
    manifests = {}
    for body in BODIES:
        groups = []
        for condition in ablation.trainer.CONDITIONS:
            for query in ablation.QUERY_INDICES:
                for ordinal in range(ablation.SEEDS_PER_CONDITION_QUERY):
                    groups.append(
                        {
                            "condition": condition,
                            "root_query_index": query,
                            "requested_seed": 2026081000 + ordinal,
                        }
                    )
        manifests[body] = {"groups": groups}
    return {"manifests": manifests}


def _ablation_validation_metrics(offset: float) -> dict[str, object]:
    return {
        "candidate_ranking": {
            "macro_one_deviation_branch_success_gain": 0.1 + offset,
            "macro_selected_success_rate": 0.5 + offset,
            "macro_oracle_success_rate": 0.8,
            "pairwise_accuracy": 0.6 + offset,
        },
        "success_brier": 0.2,
        "success_auroc": 0.7,
        "post_event": {"macro_f1": 0.5, "accuracy": 0.6},
        "next_event": {"macro_f1": 0.4, "accuracy": 0.5},
        "observed_duration_mae": 0.3,
        "observed_duration_nll": 0.4,
        "object_rmse": 0.05,
        "object_nll": 0.1,
    }


def _ablation_fold_summary(body: str, variant: str, offset: float) -> dict[str, object]:
    trainer_sha = sha256_file(Path(ablation.trainer.__file__).resolve())
    direct_rank = variant != "success_only"
    return {
        "status": "source_only_checkpoint_selection_complete",
        "held_out_body": body,
        "source_bodies": [item for item in BODIES if item != body],
        "ablation": ablation_contract(variant),
        "candidate_rank_contract": summary_candidate_rank_contract(variant),
        "trainer_file_sha256": trainer_sha,
        "rank_supervision_available": direct_rank,
        "candidate_rank_parameters_received_direct_supervision": direct_rank,
        "synthetic_success_labels": 0,
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "rank_aggregation": ablation.trainer.risk_adjusted_rank_ensemble_contract(),
            "selected_step": 3000,
            "heldout_rows_used": 0,
        },
        "training_budget": {
            "steps_per_member": 3000,
            "eval_every_steps": 100,
            "batch_size_rows": 64,
            "learning_rate": 3e-4,
            "ensemble_members": 5,
        },
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "heldout_group_npz_opened": 0,
        "preflight": {"split_unit": "body_condition_requested_seed_all_queries"},
        "members": [
            {
                "member": member,
                "seed": seed,
                "best_step": 3000,
                "trainer_file_sha256": trainer_sha,
                "source_validation": _ablation_validation_metrics(offset),
            }
            for member, seed in enumerate(ablation.ENSEMBLE_SEEDS)
        ],
    }


def test_ablation_entry_requires_exact_full_8000_branches() -> None:
    audit = _complete_ablation_audit()
    receipt = ablation.validate_complete_inventory(audit)
    assert receipt["decisions"] == 2000
    assert receipt["branches"] == 8000
    audit["manifests"]["piper"]["groups"].pop()
    with pytest.raises(ablation.AblationError, match="exactly 400"):
        ablation.validate_complete_inventory(audit)


def test_ablation_entry_freezes_same_budget_for_all_20_runs(tmp_path: Path) -> None:
    commands = [
        ablation.fold_command(
            python_executable="python3",
            binding=tmp_path / "binding.json",
            binding_sha256="a" * 64,
            output=tmp_path / variant / body,
            held_out_body=body,
            variant=variant,
        )
        for variant in ablation.VARIANTS
        for body in BODIES
    ]
    assert len(commands) == 20
    for command in commands:
        assert command[command.index("--steps") + 1] == "3000"
        assert command[command.index("--eval-every") + 1] == "100"
        assert command[command.index("--batch-size") + 1] == "64"
        assert command[command.index("--split-seed") + 1] == "20260901"
        assert command[command.index("--ensemble-seeds") + 1 :] == [
            str(seed) for seed in ablation.ENSEMBLE_SEEDS
        ]


def test_ablation_entry_reports_every_fold_macro_and_prediction_metric() -> None:
    summaries = {
        variant: {
            body: _ablation_fold_summary(body, variant, 0.01 * variant_index)
            for body in BODIES
        }
        for variant_index, variant in enumerate(ablation.VARIANTS)
    }
    result = ablation.aggregate_variants(summaries)
    for variant in ablation.VARIANTS:
        assert len(result[variant]["folds"]) == 5
        assert set(result[variant]["equal_fold_macro"]) == set(ablation.METRICS)
        assert result[variant]["equal_fold_macro"][
            "one_deviation_branch_oracle_success_rate"
        ] == 0.8
    assert result["full"]["equal_fold_macro"][
        "one_deviation_best_of_4_success_gain"
    ] > result["success_only"]["equal_fold_macro"][
        "one_deviation_best_of_4_success_gain"
    ]
    heldout = {
        variant: {
            body: {
                "held_out_body": body,
                "metrics": {
                    name: 0.1 + 0.01 * variant_index
                    for name in ablation.POSTHOC_ENSEMBLE_METRICS
                },
                "uncertainty_risk_coverage": {
                    endpoint: {
                        "support": 10,
                        "error_kind": f"{endpoint}_error",
                        "uncertainty_kind": f"{endpoint}_uncertainty",
                        "aurc": 0.2,
                        "full_coverage_risk": 0.3,
                        "error_uncertainty_spearman": 0.4,
                        "risk_at_coverage": [
                            {
                                "coverage": coverage,
                                "retained": max(1, int(10 * coverage)),
                                "risk": 0.1 + coverage,
                            }
                            for coverage in ablation.RISK_COVERAGE_LEVELS
                        ],
                    }
                    for endpoint in (
                        "rank_selected_failure",
                        "rank_oracle_regret",
                        "success",
                        "post_event",
                        "next_event",
                        "duration",
                        "object",
                        "recovery",
                    )
                },
            }
            for body in BODIES
        }
        for variant_index, variant in enumerate(ablation.VARIANTS)
    }
    heldout_result = ablation.aggregate_posthoc_heldout(heldout)
    for variant in ablation.VARIANTS:
        assert len(heldout_result[variant]["folds"]) == 5
        assert set(heldout_result[variant]["equal_fold_macro"]) == set(
            ablation.POSTHOC_ENSEMBLE_METRICS
        )


def test_ablation_posthoc_heldout_uses_frozen_five_member_rank_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    variant = "full"
    heldout = "franka"
    for ordinal, group in enumerate(audit["manifests"][heldout]["groups"]):
        group["group_id"] = (
            f"{group['condition']}|seed={group['requested_seed']}|query={ordinal}"
        )
    summary = _ablation_fold_summary(heldout, variant, 0.0)
    for member, item in enumerate(summary["members"]):
        model = EffectAlignedSharedEventHead(variant).eval()
        checkpoint_path = tmp_path / f"ablation-member-{member}.pt"
        torch.save(
            {
                "format": ablation.trainer.FORMAT,
                "model": model.state_dict(),
                "member": member,
                "seed": ablation.ENSEMBLE_SEEDS[member],
                "held_out_body": heldout,
                "ablation": ablation_contract(variant),
                "candidate_rank_contract": checkpoint_candidate_rank_contract(
                    variant
                ),
                "heldout_rows_used_for_training_normalization_or_selection": 0,
                "rank_supervision_available": True,
                "candidate_rank_parameters_received_direct_supervision": True,
                "synthetic_success_labels": 0,
                "trainer_file_sha256": sha256_file(
                    Path(ablation.trainer.__file__).resolve()
                ),
                "ensemble_common_selection_step": 3000,
            },
            checkpoint_path,
        )
        item["checkpoint"] = str(checkpoint_path)
        item["checkpoint_sha256"] = sha256_file(checkpoint_path)
    monkeypatch.setattr(ablation, "DECISIONS_PER_BODY", 4)
    result = ablation.evaluate_posthoc_heldout_fold(
        summary,
        audit,
        held_out_body=heldout,
        variant=variant,
        device=torch.device("cpu"),
    )
    assert result["heldout_decisions"] == 4
    assert result["heldout_branches"] == 16
    assert result["heldout_labels_used_for_training_checkpoint_or_variant_selection"] is False
    assert set(result["metrics"]) == set(ablation.POSTHOC_ENSEMBLE_METRICS)
    assert result["candidate_metric_aggregation"] == (
        ablation.trainer.RISK_ADJUSTED_RANK_ENSEMBLE_CONTRACT
    )
    assert result["prediction_metric_aggregation"] == (
        "five_frozen_members_mixed_in_probability_or_density_space_then_scored"
    )
    assert result["prediction_support"]["complete_four_candidate_decisions"] == 4
    assert "rank_oracle_regret" in result["uncertainty_risk_coverage"]
