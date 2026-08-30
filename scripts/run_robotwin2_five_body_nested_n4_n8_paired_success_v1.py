#!/usr/bin/env python3
"""Run a shared-raw16 actor/N4/N8 paired RoboTwin2 study.

For every policy query this runner draws one ordered set of sixteen proposals
from the frozen actor using the frozen two-instruction-by-eight-flow roster.
A deterministic, outcome/event/critic-blind greedy
farthest-point order in canonical first-five-action effect space is frozen once.
The first four entries form N4 and the first eight form N8, so N4 is an exact
ordered prefix of N8 and raw proposal zero is candidate zero in every arm.

Actor, nested-N4 and nested-N8 are executed as three fresh rollouts from the
same deterministic reset.  Their initial raw16 set and both nested pools are
bound by one commitment before any arm executes.  Later policy queries use the
same conditional construction, but naturally occur at arm-specific states
after the policies diverge.  This estimates the paired policy-level effect of
expanding the selectable pool; it is not a per-query counterfactual after the
first divergent action.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_robotwin2_five_body_paired_success_v1 as formal
import run_robotwin2_five_body_postformal_candidate_pool_v1 as pool_runner


collector = formal.collector
shared_head = formal.shared_head
analytic_event = formal.analytic_event
FORMAT = "etsf_robotwin2_five_body_nested_n4_n8_execution_v2"
PAIR_FORMAT = "etsf_robotwin2_actor_nested_n4_n8_paired_execution_v2"
CONTRACT_FORMAT = "etsf_robotwin2_nested_n4_n8_execution_contract_v1"
OUTCOME_FORMAT = "etsf_robotwin2_nested_n4_n8_outcomes_v2"
REPORT_FORMAT = "etsf_robotwin2_nested_n4_n8_report_v2"
COMPLETION_FORMAT = "etsf_robotwin2_nested_n4_n8_completion_receipt_v2"
NESTED_POOL_AUDIT_FORMAT = "etsf_robotwin2_shared_raw16_nested_n4_n8_audit_v1"
INITIAL_COMMITMENT_FORMAT = "etsf_robotwin2_initial_raw16_nested_n4_n8_commitment_v1"
METHOD_START_FORMAT = "etsf_robotwin2_nested_method_start_v1"
METHOD_RESUME_FORMAT = "etsf_robotwin2_nested_method_noninformative_resume_v1"
METHOD_RESULT_FORMAT = "etsf_robotwin2_nested_method_result_v1"
METHOD_FAILURE_FORMAT = "etsf_robotwin2_nested_method_failure_v1"
BENCHMARK = formal.BENCHMARK
TASK = formal.TASK
BODIES = formal.BODIES
CONDITIONS = formal.CONDITIONS
# This prospective postformal study uses a fresh, previously untouched reset
# block.  Reusing the formal N4 seeds after their outcomes were inspected would
# make the stronger shared-raw16 N4/N8 comparison exploratory by construction.
SEED_BASE = 2026091000
SEED_COUNT = 100
RAW_PROPOSAL_COUNT = 16
N4_CANDIDATE_COUNT = 4
N8_CANDIDATE_COUNT = 8
METHOD_ACTOR = "actor_baseline"
METHOD_N4 = "etsf_nested_best_of_4_from_raw16"
METHOD_N8 = "etsf_nested_best_of_8_from_raw16"
METHODS = (METHOD_ACTOR, METHOD_N4, METHOD_N8)
ACTION_EXEC_STEPS = formal.ACTION_EXEC_STEPS
ACTOR_DATASET_FPS = formal.ACTOR_DATASET_FPS
PLANNED_DT_SECONDS = formal.PLANNED_DT_SECONDS
NATIVE_EE_DIM = formal.NATIVE_EE_DIM
STAGE_DENOMINATOR = formal.STAGE_DENOMINATOR
EVENT_SPEC_SHA256 = formal.EVENT_SPEC_SHA256
REFERENCE_PREREGISTRATION_SHA256 = formal.PREREGISTRATION_SHA256
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260909


def nested_evaluation_protocol() -> dict[str, Any]:
    """Return the frozen, outcome-blind protocol for the nested comparison."""

    formal_seeds = set(range(formal.SEED_BASE, formal.SEED_BASE + formal.SEED_COUNT))
    nested_seeds = set(range(SEED_BASE, SEED_BASE + SEED_COUNT))
    if formal_seeds & nested_seeds:
        raise RuntimeError("nested evaluation seeds overlap the inspected formal study")
    base = {
        "format": "etsf_robotwin2_nested_n4_n8_prospective_protocol_v1",
        "evaluation_seed_base": SEED_BASE,
        "evaluation_seed_count": SEED_COUNT,
        "formal_seed_block_reused": False,
        "formal_seed_block": [formal.SEED_BASE, formal.SEED_BASE + formal.SEED_COUNT],
        "seed_block_selected_before_any_nested_rollout_outcome": True,
        "balanced_body_condition_cells": len(BODIES) * len(CONDITIONS),
        "bootstrap_unit": (
            "requested_seed_cluster_with_all_selected_body_condition_rows_kept_together"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "bootstrap_seed_derivation": {
            "overall_comparisons": "base_plus_comparison_offset_0_1_2",
            "body_comparisons": (
                "base_plus_10_plus_body_index_times_10_plus_"
                "comparison_offset_0_1_2"
            ),
            "body_condition_comparisons": (
                "base_plus_100_plus_cell_index_times_10_plus_"
                "comparison_offset_0_1_2"
            ),
        },
        "pooled_mcnemar_role": "descriptive_only_due_repeated_requested_seeds",
        "single_body_condition_mcnemar_role": "inferential",
        "fixed_benchmark_body_inference_scope": (
            "new_initial_conditions_on_the_five_fixed_heldout_bodies"
        ),
        "postformal_but_prospective_for_this_nested_estimand": True,
    }
    return {**base, "logical_sha256": canonical_sha256(base)}


canonical_sha256 = formal.canonical_sha256
sha256_file = formal.sha256_file
array_sha256 = formal.array_sha256
atomic_json = formal.atomic_json


def _staged_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.staged")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_create_once_json(
    path: Path, *, label: str
) -> tuple[dict[str, Any] | None, bool]:
    """Read a final/staged create-once artifact without silently replacing it."""

    stage = _staged_path(path)
    final_exists = path.exists()
    stage_exists = stage.exists()
    if (final_exists and (not path.is_file() or path.is_symlink())) or (
        stage_exists and (not stage.is_file() or stage.is_symlink())
    ):
        raise NestedCandidatePoolError(f"{label} is missing, symbolic or non-file")
    if not final_exists and not stage_exists:
        return None, False

    def load(candidate: Path) -> dict[str, Any]:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise NestedCandidatePoolError(
                f"{label} staged/final JSON is incomplete"
            ) from error
        if not isinstance(value, dict):
            raise NestedCandidatePoolError(f"{label} must be a JSON object")
        return value

    final_value = load(path) if final_exists else None
    stage_value = load(stage) if stage_exists else None
    if final_value is not None and stage_value is not None and final_value != stage_value:
        raise NestedCandidatePoolError(f"{label} final/staged artifacts differ")
    return (
        final_value if final_value is not None else stage_value,
        final_value is None and stage_value is not None,
    )


def promote_create_once_json(
    path: Path, value: Mapping[str, Any], *, label: str
) -> str:
    """Persist JSON by fsync + hard-link create; an existing value must match."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = _staged_path(path)
    existing, staged_only = read_create_once_json(path, label=label)
    if existing is None:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        try:
            with stage.open("x", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            stage.chmod(0o444)
            _fsync_directory(path.parent)
        except FileExistsError:
            pass
        existing, staged_only = read_create_once_json(path, label=label)
    if existing != dict(value):
        raise NestedCandidatePoolError(f"{label} create-once value changed")
    if staged_only:
        try:
            os.link(stage, path)
        except FileExistsError:
            current, _staged = read_create_once_json(path, label=label)
            if current != dict(value):
                raise NestedCandidatePoolError(f"{label} link race changed value")
        _fsync_directory(path.parent)
    current, _staged_only = read_create_once_json(path, label=label)
    if current != dict(value) or not path.is_file() or path.is_symlink():
        raise NestedCandidatePoolError(f"{label} final promotion failed")
    if stage.exists():
        stage.unlink()
        _fsync_directory(path.parent)
    return sha256_file(path)
pair_id = formal.pair_id


class NestedCandidatePoolError(RuntimeError):
    """A frozen actor, nested pool, reset, fold, or output changed."""


def existing_separate_n4_n8_comparability_audit() -> dict[str, Any]:
    """State why the two existing independent studies are not a pool-size RCT."""

    return {
        "same_body_condition_requested_seed_schedule": True,
        "same_frozen_actor_candidate_zero_noise_identity": True,
        "each_study_pairs_its_own_actor_baseline": True,
        "formal_n4_construction": "generate_raw4_keep_original_order",
        "current_n8_construction": "generate_raw16_blind_fps_retain8",
        "formal_n4_is_required_subset_of_current_n8": False,
        "shared_cross_study_initial_raw_pool_commitment": False,
        "shared_cross_study_reset_snapshot_commitment": False,
        "direct_strong_causal_pool_size_comparison_authorized": False,
        "valid_separate_claims": [
            "formal_n4_minus_its_paired_actor_baseline",
            "postformal_n8_minus_its_paired_actor_baseline",
        ],
        "invalid_unqualified_claim": "existing_n8_minus_existing_n4_is_pool_size_gain",
    }


def nested_pool_contract() -> dict[str, Any]:
    return {
        "format": NESTED_POOL_AUDIT_FORMAT,
        "raw_proposal_count": RAW_PROPOSAL_COUNT,
        "retained_candidate_counts": [N4_CANDIDATE_COUNT, N8_CANDIDATE_COUNT],
        "single_actor_generation_per_policy_query": True,
        "ordering": (
            "greedy_maximize_minimum_rms_in_flattened_first_five_"
            "canonical_action_effect14_anchor_raw_zero_tie_lowest_raw_index"
        ),
        "n4_is_exact_ordered_prefix_of_n8": True,
        "raw_proposal_zero_is_candidate_zero_for_actor_n4_n8": True,
        "selection_reads_outcomes_events_labels_or_critic_scores": False,
        "raw16_flow_noise_contract": pool_runner.postformal_noise_contract(
            RAW_PROPOSAL_COUNT, RAW_PROPOSAL_COUNT
        ),
        "raw16_instruction_coverage_contract": pool_runner.candidate_pool_contract(
            RAW_PROPOSAL_COUNT, RAW_PROPOSAL_COUNT
        )["instruction_coverage_contract"],
        "critic_scoring_occurs_only_after_both_pool_indices_are_frozen": True,
        "frozen_actor_weights_changed": False,
        "additional_authorization_or_confidence_gate": False,
        "policy_level_estimand": (
            "paired_success_and_stage_progress_delta_n8_minus_n4_from_"
            "the_same_initial_reset"
        ),
        "post_divergence_queries_share_construction_not_state": True,
        "evaluation_protocol": nested_evaluation_protocol(),
    }


def _pairwise_rms(values: np.ndarray) -> np.ndarray:
    rows = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            rows.append(
                float(np.sqrt(np.mean(np.square(values[left] - values[right]))))
            )
    return np.asarray(rows, dtype=np.float64)


def nested_pool_selection_audit(
    *, current_ee: np.ndarray, raw_proposals: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Freeze strict nested N4/N8 prefixes without labels or critic scores."""

    raw = pool_runner.validate_candidates(
        raw_proposals,
        expected_count=RAW_PROPOSAL_COUNT,
        label="shared raw16 proposals",
    )
    effects, embeddings = pool_runner.canonical_effect_embeddings(current_ee, raw)
    n8_indices = pool_runner.greedy_farthest_point_indices(
        embeddings, retain_count=N8_CANDIDATE_COUNT
    )
    n4_indices = n8_indices[:N4_CANDIDATE_COUNT]
    if (
        n4_indices != n8_indices[:N4_CANDIDATE_COUNT]
        or n4_indices[0] != 0
        or len(set(n8_indices)) != N8_CANDIDATE_COUNT
    ):
        raise NestedCandidatePoolError("nested candidate index contract changed")
    n4 = raw[np.asarray(n4_indices, dtype=np.int64)].copy()
    n8 = raw[np.asarray(n8_indices, dtype=np.int64)].copy()
    if not np.array_equal(n4, n8[:N4_CANDIDATE_COUNT]):
        raise NestedCandidatePoolError("N4 is not the exact ordered prefix of N8")
    if not np.array_equal(raw[0], n4[0]) or not np.array_equal(raw[0], n8[0]):
        raise NestedCandidatePoolError("raw proposal zero identity changed")
    raw_distances = _pairwise_rms(embeddings)
    n4_distances = _pairwise_rms(embeddings[n4_indices])
    n8_distances = _pairwise_rms(embeddings[n8_indices])
    base = {
        **nested_pool_contract(),
        "raw_shape": list(raw.shape),
        "raw_ordered_proposals_sha256": array_sha256(raw),
        "raw_proposal_zero_sha256": array_sha256(raw[:1]),
        "canonical_first_five_effects_sha256": array_sha256(effects),
        "canonical_effect_embedding_shape": list(embeddings.shape),
        "ordered_fps_raw_indices_n8": n8_indices,
        "ordered_fps_raw_indices_n4": n4_indices,
        "n4_ordered_candidates_sha256": array_sha256(n4),
        "n8_ordered_candidates_sha256": array_sha256(n8),
        "n4_equals_n8_prefix_sha256": array_sha256(n8[:N4_CANDIDATE_COUNT]),
        "pairwise_effect_rms": {
            "raw16": {
                "minimum": float(raw_distances.min()),
                "median": float(np.median(raw_distances)),
                "maximum": float(raw_distances.max()),
            },
            "nested_n4": {
                "minimum": float(n4_distances.min()),
                "median": float(np.median(n4_distances)),
                "maximum": float(n4_distances.max()),
            },
            "nested_n8": {
                "minimum": float(n8_distances.min()),
                "median": float(np.median(n8_distances)),
                "maximum": float(n8_distances.max()),
            },
        },
    }
    return {
        N4_CANDIDATE_COUNT: n4,
        N8_CANDIDATE_COUNT: n8,
    }, {**base, "audit_sha256": canonical_sha256(base)}


def validate_nested_pool_audit(value: Mapping[str, Any]) -> None:
    unsigned = {key: item for key, item in value.items() if key != "audit_sha256"}
    n4_indices = value.get("ordered_fps_raw_indices_n4")
    n8_indices = value.get("ordered_fps_raw_indices_n8")
    if (
        value.get("format") != NESTED_POOL_AUDIT_FORMAT
        or value.get("raw_proposal_count") != RAW_PROPOSAL_COUNT
        or value.get("retained_candidate_counts")
        != [N4_CANDIDATE_COUNT, N8_CANDIDATE_COUNT]
        or value.get("n4_is_exact_ordered_prefix_of_n8") is not True
        or value.get("raw_proposal_zero_is_candidate_zero_for_actor_n4_n8")
        is not True
        or value.get("selection_reads_outcomes_events_labels_or_critic_scores")
        is not False
        or value.get("critic_scoring_occurs_only_after_both_pool_indices_are_frozen")
        is not True
        or not isinstance(n4_indices, list)
        or not isinstance(n8_indices, list)
        or len(n4_indices) != N4_CANDIDATE_COUNT
        or len(n8_indices) != N8_CANDIDATE_COUNT
        or n4_indices != n8_indices[:N4_CANDIDATE_COUNT]
        or n8_indices[0] != 0
        or len(set(n8_indices)) != N8_CANDIDATE_COUNT
        or any(type(index) is not int or not 0 <= index < RAW_PROPOSAL_COUNT for index in n8_indices)
        or value.get("n4_ordered_candidates_sha256")
        != value.get("n4_equals_n8_prefix_sha256")
        or value.get("audit_sha256") != canonical_sha256(unsigned)
    ):
        raise NestedCandidatePoolError("nested candidate pool audit changed")


def generate_nested_pools(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: Any,
    instruction: str,
    scene_seed: int,
    query_index: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    raw = pool_runner.generate_postformal_flow_candidates(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task=task,
        instruction=instruction,
        scene_seed=scene_seed,
        query_index=query_index,
        candidate_count=RAW_PROPOSAL_COUNT,
        device=device,
    )
    pools, audit = nested_pool_selection_audit(
        current_ee=collector.current_ee_action16(task), raw_proposals=raw
    )
    return np.asarray(raw, dtype=np.float32), pools, audit


def evaluation_schedule() -> list[dict[str, Any]]:
    rotations = (
        list(METHODS),
        [METHOD_N4, METHOD_N8, METHOD_ACTOR],
        [METHOD_N8, METHOD_ACTOR, METHOD_N4],
    )
    rows = []
    for body in BODIES:
        for condition in CONDITIONS:
            for ordinal in range(SEED_COUNT):
                rows.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": SEED_BASE + ordinal,
                        "method_order": list(rotations[ordinal % len(rotations)]),
                    }
                )
    expected = len(BODIES) * len(CONDITIONS) * SEED_COUNT
    if len(rows) != expected:
        raise NestedCandidatePoolError("nested paired schedule cardinality changed")
    return rows


def _commitment_base(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "commitment_sha256"}


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
    instruction: str,
    device: torch.device,
) -> dict[str, Any]:
    """Freeze query-zero raw16 and both nested pools before any arm executes."""

    required_names = set(analytic_event.REQUIRED_OBJECTS)
    # RoboTwin seeds NumPy/Torch during setup but its Python-RNG reset is
    # disabled.  CuRobo fallback paths use Python random, so reset it here and
    # identically in every treatment arm before constructing the fresh scene.
    random.seed(int(seed))
    task = collector._new_task(task_class, task_args, seed, instruction)
    try:
        names, objects = collector.discover_pose_objects(task, required_names)
        reset_snapshot = formal.capture_reset_snapshot(task, names, objects)
        task.scene.step()
        canonical_snapshot = formal.capture_reset_snapshot(task, names, objects)
        raw, pools, audit = generate_nested_pools(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            instruction=instruction,
            scene_seed=seed,
            query_index=0,
            device=device,
        )
        after = formal.capture_reset_snapshot(task, names, objects)
        if canonical_snapshot != after:
            raise NestedCandidatePoolError(
                "initial raw16 generation changed observable simulator state"
            )
        base = {
            "format": INITIAL_COMMITMENT_FORMAT,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "action_exec_steps": ACTION_EXEC_STEPS,
            "planned_dt_seconds": PLANNED_DT_SECONDS,
            "raw_proposal_count": RAW_PROPOSAL_COUNT,
            "raw_candidate_horizon": int(raw.shape[1]),
            "raw_candidate_shape": list(raw.shape),
            "raw_ordered_proposals_sha256": array_sha256(raw),
            "raw_proposal_zero_sha256": array_sha256(raw[:1]),
            "n4_ordered_candidates_sha256": array_sha256(
                pools[N4_CANDIDATE_COUNT]
            ),
            "n8_ordered_candidates_sha256": array_sha256(
                pools[N8_CANDIDATE_COUNT]
            ),
            "nested_pool_audit": audit,
            "reset_snapshot": reset_snapshot,
            "reset_identity_sha256": formal.reset_identity(reset_snapshot),
            "canonical_query_snapshot": canonical_snapshot,
            "canonical_query_identity_sha256": formal.reset_identity(
                canonical_snapshot
            ),
            "query_canonicalization_steps": formal.QUERY_CANONICALIZATION_STEPS,
            "candidate_generation_advanced_simulator": False,
            "frozen_before_any_method_execution": True,
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
    raw: np.ndarray,
    pools: Mapping[int, np.ndarray],
    audit: Mapping[str, Any],
) -> None:
    validate_nested_pool_audit(audit)
    if (
        commitment.get("format") != INITIAL_COMMITMENT_FORMAT
        or commitment.get("heldout_body") != body
        or commitment.get("condition") != condition
        or commitment.get("requested_seed") != seed
        or commitment.get("resolved_seed") != seed
        or commitment.get("raw_proposal_count") != RAW_PROPOSAL_COUNT
        or commitment.get("raw_candidate_shape") != list(raw.shape)
        or commitment.get("raw_ordered_proposals_sha256") != array_sha256(raw)
        or commitment.get("raw_proposal_zero_sha256") != array_sha256(raw[:1])
        or commitment.get("n4_ordered_candidates_sha256")
        != array_sha256(pools[N4_CANDIDATE_COUNT])
        or commitment.get("n8_ordered_candidates_sha256")
        != array_sha256(pools[N8_CANDIDATE_COUNT])
        or commitment.get("nested_pool_audit") != dict(audit)
        or commitment.get("reset_snapshot") != reset_snapshot
        or commitment.get("reset_identity_sha256")
        != formal.reset_identity(reset_snapshot)
        or commitment.get("canonical_query_snapshot") != canonical_query_snapshot
        or commitment.get("canonical_query_identity_sha256")
        != formal.reset_identity(canonical_query_snapshot)
        or commitment.get("candidate_generation_advanced_simulator") is not False
        or commitment.get("frozen_before_any_method_execution") is not True
        or commitment.get("commitment_sha256")
        != canonical_sha256(_commitment_base(commitment))
    ):
        raise NestedCandidatePoolError(
            "method query-zero raw16/pools differ from frozen commitment"
        )


def _method_candidates(
    method: str,
    raw: np.ndarray,
    pools: Mapping[int, np.ndarray],
    audit: Mapping[str, Any],
) -> tuple[np.ndarray, list[int]]:
    if method == METHOD_ACTOR:
        return raw[:1], [0]
    if method == METHOD_N4:
        return pools[N4_CANDIDATE_COUNT], list(
            audit["ordered_fps_raw_indices_n4"]
        )
    if method == METHOD_N8:
        return pools[N8_CANDIDATE_COUNT], list(
            audit["ordered_fps_raw_indices_n8"]
        )
    raise NestedCandidatePoolError(f"unknown nested method {method!r}")


def _score_candidates(
    *,
    method: str,
    ensemble: Sequence[shared_head.EffectAlignedSharedEventHead],
    state: np.ndarray,
    current_ee: np.ndarray,
    candidates: np.ndarray,
    current_event: int,
    event_age_seconds: float,
    remaining_action_budget: int,
    device: torch.device,
) -> dict[str, Any]:
    if method == METHOD_N4:
        return formal.score_candidates(
            ensemble,
            formal.scoring_batch(
                state=state,
                current_ee=current_ee,
                candidates=candidates,
                current_event=current_event,
                event_age_seconds=event_age_seconds,
                remaining_action_budget=remaining_action_budget,
                action_exec_steps=ACTION_EXEC_STEPS,
                dt=1.0 / ACTOR_DATASET_FPS,
                device=device,
            ),
        )
    if method == METHOD_N8:
        return pool_runner.score_candidates(
            ensemble,
            pool_runner.scoring_batch(
                state=state,
                current_ee=current_ee,
                candidates=candidates,
                current_event=current_event,
                event_age_seconds=event_age_seconds,
                remaining_action_budget=remaining_action_budget,
                candidate_count=N8_CANDIDATE_COUNT,
                device=device,
            ),
            candidate_count=N8_CANDIDATE_COUNT,
        )
    raise NestedCandidatePoolError("actor baseline may not call the critic")


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
    max_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    if method not in METHODS:
        raise NestedCandidatePoolError(f"unknown nested method {method!r}")
    required_names = {str(calibration["moving"])}
    anchor = str(calibration.get("anchor", "")).strip()
    if anchor:
        required_names.add(anchor)
    random.seed(int(seed))
    task = collector._new_task(task_class, task_args, seed, instruction)
    decisions = []
    try:
        names, objects = collector.discover_pose_objects(task, required_names)
        initial_poses = collector.read_poses(objects)
        initial_ee = collector.current_ee_action16(task)
        initial_snapshot = formal.capture_reset_snapshot(task, names, objects)
        trajectory = [initial_poses]
        sim_times = [collector._sim_time(task)]
        initial_canonical_snapshot: Mapping[str, Any] | None = None
        query_index = 0
        while not collector._episode_done(task, max_steps):
            task.scene.step()
            collector._append_physical_observation(
                task, objects, trajectory, sim_times
            )
            current_ee = collector.current_ee_action16(task)
            pre_pool_snapshot = formal.capture_reset_snapshot(task, names, objects)
            raw, pools, audit = generate_nested_pools(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                instruction=instruction,
                scene_seed=seed,
                query_index=query_index,
                device=device,
            )
            after_pool_snapshot = formal.capture_reset_snapshot(task, names, objects)
            if after_pool_snapshot != pre_pool_snapshot:
                raise NestedCandidatePoolError(
                    "raw16 generation changed observable simulator state"
                )
            candidates, raw_indices = _method_candidates(method, raw, pools, audit)
            if query_index == 0:
                initial_canonical_snapshot = pre_pool_snapshot
                verify_initial_commitment(
                    initial_commitment,
                    body=body,
                    condition=condition,
                    seed=seed,
                    reset_snapshot=initial_snapshot,
                    canonical_query_snapshot=pre_pool_snapshot,
                    raw=raw,
                    pools=pools,
                    audit=audit,
                )
            current_event_age: float | None = None
            if method == METHOD_ACTOR:
                selected = 0
                score_record = None
            else:
                trajectory_array = np.stack(trajectory).astype(np.float32)
                state, current_event, current_event_age = formal.canonical_state_at(
                    trajectory=trajectory_array,
                    sim_times=np.asarray(sim_times, dtype=np.float64),
                    names=names,
                    ee_action=current_ee,
                    calibration=calibration,
                )
                score_record = _score_candidates(
                    method=method,
                    ensemble=ensemble,
                    state=state,
                    current_ee=current_ee,
                    candidates=candidates,
                    current_event=current_event,
                    event_age_seconds=current_event_age,
                    remaining_action_budget=max_steps
                    - int(getattr(task, "take_action_cnt", 0)),
                    device=device,
                )
                selected = int(score_record["selected_candidate_index"])
            executed = 0
            chunk_start = collector._sim_time(task)
            for action in candidates[selected, :ACTION_EXEC_STEPS]:
                if collector._episode_done(task, max_steps):
                    break
                task.take_action(action, action_type="ee")
                executed += 1
                collector._append_physical_observation(
                    task, objects, trajectory, sim_times
                )
            decisions.append(
                {
                    "query_index": query_index,
                    "raw_proposal_count": RAW_PROPOSAL_COUNT,
                    "raw_ordered_proposals_sha256": array_sha256(raw),
                    "raw_proposal_zero_sha256": array_sha256(raw[:1]),
                    "nested_pool_audit": audit,
                    "selection_pool_candidate_count": len(candidates),
                    "selection_pool_raw_indices": raw_indices,
                    "selection_pool_sha256": array_sha256(candidates),
                    "selected_candidate_index": selected,
                    "selected_raw_proposal_index": raw_indices[selected],
                    "executed_action_count": executed,
                    "planned_chunk_seconds": PLANNED_DT_SECONDS,
                    "physical_sim_seconds": collector._sim_time(task) - chunk_start,
                    "critic_scores": score_record,
                    "event_age_seconds": (
                        None if score_record is None else float(current_event_age)
                    ),
                }
            )
            query_index += 1
        success = bool(getattr(task, "eval_success", False))
        if not success:
            success = bool(task.check_success())
        trajectory_array = np.stack(trajectory).astype(np.float32)
        _predicates, events = collector.derive_predicates_and_events(
            trajectory_array,
            np.asarray(sim_times, dtype=np.float64),
            names,
            success,
            calibration,
        )
        return {
            "method": method,
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "initial_reset_identity_sha256": formal.reset_identity(initial_snapshot),
            "initial_reset_snapshot": initial_snapshot,
            "initial_canonical_query_snapshot": initial_canonical_snapshot,
            "initial_candidate_commitment_sha256": initial_commitment[
                "commitment_sha256"
            ],
            "tracked_object_names": list(names),
            "initial_object_poses": initial_poses.astype(float).tolist(),
            "initial_ee16": initial_ee.astype(float).tolist(),
            "binary_success": int(success),
            "stage_progress": formal.stage_progress(events, success),
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


def _expected_pool_count(method: str) -> int:
    return {
        METHOD_ACTOR: 1,
        METHOD_N4: N4_CANDIDATE_COUNT,
        METHOD_N8: N8_CANDIDATE_COUNT,
    }[method]


def validate_rollout(
    rollout: Mapping[str, Any], *, method: str, expected: Mapping[str, Any]
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
        raise NestedCandidatePoolError(f"{method} rollout identity/outcome changed")
    expected_progress = (
        1.0
        if rollout["binary_success"] == 1
        else rollout["max_event_id"] / float(STAGE_DENOMINATOR)
    )
    if abs(float(rollout.get("stage_progress", -1.0)) - expected_progress) > 1e-9:
        raise NestedCandidatePoolError(f"{method} stage progress changed")
    decisions = rollout.get("decisions")
    if (
        not isinstance(decisions, list)
        or not decisions
        or rollout.get("policy_query_count") != len(decisions)
    ):
        raise NestedCandidatePoolError(f"{method} decision roster changed")
    expected_count = _expected_pool_count(method)
    for query_index, decision in enumerate(decisions):
        audit = decision.get("nested_pool_audit")
        if not isinstance(audit, Mapping):
            raise NestedCandidatePoolError("decision lacks nested pool audit")
        validate_nested_pool_audit(audit)
        raw_indices = decision.get("selection_pool_raw_indices")
        expected_indices = (
            [0]
            if method == METHOD_ACTOR
            else audit[
                "ordered_fps_raw_indices_n4"
                if method == METHOD_N4
                else "ordered_fps_raw_indices_n8"
            ]
        )
        expected_pool_sha = (
            audit["raw_proposal_zero_sha256"]
            if method == METHOD_ACTOR
            else audit[
                "n4_ordered_candidates_sha256"
                if method == METHOD_N4
                else "n8_ordered_candidates_sha256"
            ]
        )
        if (
            decision.get("query_index") != query_index
            or decision.get("raw_proposal_count") != RAW_PROPOSAL_COUNT
            or decision.get("raw_ordered_proposals_sha256")
            != audit.get("raw_ordered_proposals_sha256")
            or decision.get("selection_pool_candidate_count") != expected_count
            or raw_indices != expected_indices
            or decision.get("selection_pool_sha256") != expected_pool_sha
            or type(decision.get("selected_candidate_index")) is not int
            or not 0 <= decision["selected_candidate_index"] < expected_count
            or decision.get("selected_raw_proposal_index")
            != raw_indices[decision["selected_candidate_index"]]
        ):
            raise NestedCandidatePoolError(f"{method} nested decision changed")
        scores = decision.get("critic_scores")
        if method == METHOD_ACTOR:
            if (
                decision["selected_candidate_index"] != 0
                or decision["selected_raw_proposal_index"] != 0
                or scores is not None
                or decision.get("event_age_seconds") is not None
            ):
                raise NestedCandidatePoolError(
                    "actor baseline must execute raw proposal zero without critic"
                )
            continue
        if not isinstance(scores, Mapping):
            raise NestedCandidatePoolError("nested ETSF decision lacks critic scores")
        member_scores = np.asarray(
            scores.get("candidate_rank_score_members"), dtype=np.float64
        )
        recorded = np.asarray(
            scores.get("candidate_rank_score_epistemic_lcb_ensemble"),
            dtype=np.float64,
        )
        if member_scores.shape != (5, expected_count) or recorded.shape != (
            expected_count,
        ):
            raise NestedCandidatePoolError("nested critic score shape changed")
        recomputed_tensor = (
            shared_head.aggregate_risk_adjusted_rank_scores(
                torch.as_tensor(member_scores)
            )
            if method == METHOD_N4
            else pool_runner.aggregate_risk_adjusted_rank_scores(
                torch.as_tensor(member_scores)
            )
        )
        recomputed = recomputed_tensor.cpu().numpy()
        if not np.allclose(recorded, recomputed, atol=1e-6, rtol=0.0):
            raise NestedCandidatePoolError("nested critic score cannot be replayed")
        selected = int(np.argmax(recomputed))
        if (
            scores.get("selected_candidate_index") != selected
            or decision["selected_candidate_index"] != selected
        ):
            raise NestedCandidatePoolError("nested critic selection changed")


def validate_stored_initial_commitment(
    commitment: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    audit = commitment.get("nested_pool_audit")
    if not isinstance(audit, Mapping):
        raise NestedCandidatePoolError("stored commitment lacks nested pool audit")
    validate_nested_pool_audit(audit)
    if (
        commitment.get("format") != INITIAL_COMMITMENT_FORMAT
        or commitment.get("heldout_body") != expected["heldout_body"]
        or commitment.get("condition") != expected["condition"]
        or commitment.get("requested_seed") != expected["requested_seed"]
        or commitment.get("resolved_seed") != expected["requested_seed"]
        or commitment.get("frozen_before_any_method_execution") is not True
        or commitment.get("candidate_generation_advanced_simulator") is not False
        or commitment.get("commitment_sha256")
        != canonical_sha256(_commitment_base(commitment))
        or commitment.get("reset_identity_sha256")
        != formal.reset_identity(commitment.get("reset_snapshot", {}))
        or commitment.get("canonical_query_identity_sha256")
        != formal.reset_identity(commitment.get("canonical_query_snapshot", {}))
    ):
        raise NestedCandidatePoolError("stored initial commitment changed")


def build_method_start(
    expected: Mapping[str, Any],
    *,
    method: str,
    method_ordinal: int,
    attempt_sha256: str,
    commitment_sha256: str,
    execution_contract_logical_sha256: str,
    completed_prefix_result_sha256: Sequence[str],
) -> dict[str, Any]:
    base = {
        "format": METHOD_START_FORMAT,
        "status": "started_once_fixed_order_no_outcome_selection",
        **dict(expected),
        "method": method,
        "method_ordinal": method_ordinal,
        "attempt_sha256": attempt_sha256,
        "commitment_sha256": commitment_sha256,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "completed_prefix_result_sha256": list(completed_prefix_result_sha256),
        "automatic_noninformative_resume_limit": 1,
        "result_or_failure_never_overwritten_or_retried": True,
    }
    return {**base, "method_start_sha256": canonical_sha256(base)}


def validate_method_start(
    value: Mapping[str, Any], expected_value: Mapping[str, Any]
) -> None:
    if dict(value) != dict(expected_value):
        raise NestedCandidatePoolError("nested method start contract changed")


def build_method_result(
    expected: Mapping[str, Any],
    *,
    method: str,
    method_ordinal: int,
    rollout: Mapping[str, Any],
    method_start_sha256: str,
    attempt_sha256: str,
    commitment_sha256: str,
    execution_contract_logical_sha256: str,
    execution_contract_file_sha256: str,
    completed_prefix_result_sha256: Sequence[str],
) -> dict[str, Any]:
    validate_rollout(rollout, method=method, expected=expected)
    base = {
        "format": METHOD_RESULT_FORMAT,
        "status": "complete_create_once_no_retry",
        **dict(expected),
        "method": method,
        "method_ordinal": method_ordinal,
        "method_start_sha256": method_start_sha256,
        "attempt_sha256": attempt_sha256,
        "commitment_sha256": commitment_sha256,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "execution_contract_file_sha256": execution_contract_file_sha256,
        "completed_prefix_result_sha256": list(completed_prefix_result_sha256),
        "rollout": dict(rollout),
        "rollout_sha256": canonical_sha256(rollout),
    }
    return {**base, "method_result_sha256": canonical_sha256(base)}


def validate_method_result(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    method: str,
    method_ordinal: int,
    method_start_sha256: str,
    attempt_sha256: str,
    commitment_sha256: str,
    execution_contract_logical_sha256: str,
    execution_contract_file_sha256: str,
    completed_prefix_result_sha256: Sequence[str],
) -> Mapping[str, Any]:
    rollout = value.get("rollout")
    unsigned = {key: item for key, item in value.items() if key != "method_result_sha256"}
    if (
        value.get("format") != METHOD_RESULT_FORMAT
        or value.get("status") != "complete_create_once_no_retry"
        or any(value.get(key) != expected[key] for key in expected)
        or value.get("method") != method
        or value.get("method_ordinal") != method_ordinal
        or value.get("method_start_sha256") != method_start_sha256
        or value.get("attempt_sha256") != attempt_sha256
        or value.get("commitment_sha256") != commitment_sha256
        or value.get("execution_contract_logical_sha256")
        != execution_contract_logical_sha256
        or value.get("execution_contract_file_sha256")
        != execution_contract_file_sha256
        or value.get("completed_prefix_result_sha256")
        != list(completed_prefix_result_sha256)
        or not isinstance(rollout, Mapping)
        or value.get("rollout_sha256") != canonical_sha256(rollout)
        or value.get("method_result_sha256") != canonical_sha256(unsigned)
    ):
        raise NestedCandidatePoolError("nested method result binding changed")
    validate_rollout(rollout, method=method, expected=expected)
    if rollout.get("initial_candidate_commitment_sha256") != commitment_sha256:
        raise NestedCandidatePoolError("nested method result commitment changed")
    return rollout


def materialize_triplet(
    expected: Mapping[str, Any],
    rollouts: Mapping[str, Mapping[str, Any]],
    *,
    commitment: Mapping[str, Any],
    attempt_sha256: str,
    execution_contract_logical_sha256: str,
    method_result_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(rollouts) != set(METHODS):
        raise NestedCandidatePoolError("triplet must contain actor, N4 and N8")
    if set(method_result_bindings) != set(METHODS):
        raise NestedCandidatePoolError("triplet lacks exact method-result bindings")
    normalized_method_bindings = {}
    for method in METHODS:
        binding = method_result_bindings[method]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"logical_sha256", "file_sha256"}
            or any(
                not isinstance(binding.get(name), str)
                or len(str(binding.get(name))) != 64
                for name in ("logical_sha256", "file_sha256")
            )
        ):
            raise NestedCandidatePoolError("triplet method-result SHA binding changed")
        normalized_method_bindings[method] = dict(binding)
    first = {method: rollouts[method]["decisions"][0] for method in METHODS}
    same_reset = bool(
        len(
            {
                canonical_sha256(rollouts[method]["initial_reset_snapshot"])
                for method in METHODS
            }
        )
        == 1
        and all(
            rollouts[method]["initial_reset_snapshot"]
            == commitment.get("reset_snapshot")
            for method in METHODS
        )
        and all(
            rollouts[method]["initial_canonical_query_snapshot"]
            == commitment.get("canonical_query_snapshot")
            for method in METHODS
        )
        and all(
            rollouts[method]["initial_candidate_commitment_sha256"]
            == commitment.get("commitment_sha256")
            for method in METHODS
        )
    )
    same_initial_raw16 = bool(
        all(
            first[method]["raw_ordered_proposals_sha256"]
            == commitment.get("raw_ordered_proposals_sha256")
            for method in METHODS
        )
        and all(
            first[method]["nested_pool_audit"]
            == commitment.get("nested_pool_audit")
            for method in METHODS
        )
    )
    if not same_reset or not same_initial_raw16:
        raise NestedCandidatePoolError(
            "three methods did not share reset/query-zero raw16 commitment"
        )
    base = {
        "format": PAIR_FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        **dict(expected),
        "attempt_sha256": attempt_sha256,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "initial_candidate_commitment_sha256": commitment.get(
            "commitment_sha256"
        ),
        "method_result_bindings": normalized_method_bindings,
        "same_resolved_reset_actor_n4_n8": same_reset,
        "same_initial_raw16_and_nested_pool_audit": same_initial_raw16,
        "n4_is_exact_ordered_prefix_of_n8": True,
        "rollouts": dict(rollouts),
    }
    return {**base, "pair_sha256": canonical_sha256(base)}


def validate_triplet(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    execution_contract_logical_sha256: str,
    expected_method_result_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if (
        value.get("format") != PAIR_FORMAT
        or value.get("benchmark") != BENCHMARK
        or value.get("task") != TASK
        or value.get("heldout_body") != expected["heldout_body"]
        or value.get("condition") != expected["condition"]
        or value.get("requested_seed") != expected["requested_seed"]
        or value.get("method_order") != expected["method_order"]
        or value.get("same_resolved_reset_actor_n4_n8") is not True
        or value.get("same_initial_raw16_and_nested_pool_audit") is not True
        or value.get("n4_is_exact_ordered_prefix_of_n8") is not True
        or value.get("execution_contract_logical_sha256")
        != execution_contract_logical_sha256
        or not isinstance(value.get("method_result_bindings"), Mapping)
        or set(value["method_result_bindings"]) != set(METHODS)
        or (
            expected_method_result_bindings is not None
            and value.get("method_result_bindings")
            != {key: dict(item) for key, item in expected_method_result_bindings.items()}
        )
        or value.get("pair_sha256")
        != canonical_sha256(
            {key: item for key, item in value.items() if key != "pair_sha256"}
        )
    ):
        raise NestedCandidatePoolError("nested triplet record changed")
    for binding in value["method_result_bindings"].values():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"logical_sha256", "file_sha256"}
            or any(
                not isinstance(binding.get(name), str)
                or len(str(binding.get(name))) != 64
                for name in ("logical_sha256", "file_sha256")
            )
        ):
            raise NestedCandidatePoolError("triplet method-result SHA binding changed")
    rollouts = value.get("rollouts")
    if not isinstance(rollouts, Mapping) or set(rollouts) != set(METHODS):
        raise NestedCandidatePoolError("nested triplet rollout roster changed")
    for method in METHODS:
        validate_rollout(rollouts[method], method=method, expected=expected)
        if (
            rollouts[method].get("initial_candidate_commitment_sha256")
            != value.get("initial_candidate_commitment_sha256")
        ):
            raise NestedCandidatePoolError("triplet commitment binding changed")


def recover_complete_existing_triplet(
    *,
    pair_path: Path,
    attempt_path: Path,
    commitment_path: Path,
    method_starts_dir: Path,
    method_results_dir: Path,
    method_failures_dir: Path,
    identity: str,
    expected: Mapping[str, Any],
    attempt: Mapping[str, Any],
    execution_contract_logical_sha256: str,
    execution_contract_file_sha256: str,
) -> dict[str, Any] | None:
    """Recover a complete pair without executing or repairing a missing method."""

    existing_pair, _pair_staged = read_create_once_json(
        pair_path, label="nested triplet"
    )
    if existing_pair is None:
        return None

    existing_attempt, _attempt_staged = read_create_once_json(
        attempt_path, label="nested triplet attempt"
    )
    if existing_attempt is None or dict(existing_attempt) != dict(attempt):
        raise NestedCandidatePoolError(
            "existing nested triplet lacks its exact attempt"
        )
    promote_create_once_json(
        attempt_path, existing_attempt, label="nested triplet attempt"
    )

    commitment, _commitment_staged = read_create_once_json(
        commitment_path, label="nested initial commitment"
    )
    if commitment is None:
        raise NestedCandidatePoolError(
            "existing nested triplet lacks its initial commitment"
        )
    validate_stored_initial_commitment(commitment, expected)
    promote_create_once_json(
        commitment_path, commitment, label="nested initial commitment"
    )
    commitment_sha = str(commitment["commitment_sha256"])
    attempt_sha = str(attempt["attempt_sha256"])

    rollouts: dict[str, Mapping[str, Any]] = {}
    method_result_bindings: dict[str, dict[str, str]] = {}
    prefix_result_shas: list[str] = []
    for method_ordinal, method in enumerate(expected["method_order"]):
        stem = f"{identity}.{method_ordinal:02d}.{method}"
        start_path = method_starts_dir / f"{stem}.json"
        result_path = method_results_dir / f"{stem}.json"
        method_failure_path = method_failures_dir / f"{stem}.json"
        start_value = build_method_start(
            expected,
            method=method,
            method_ordinal=method_ordinal,
            attempt_sha256=attempt_sha,
            commitment_sha256=commitment_sha,
            execution_contract_logical_sha256=(
                execution_contract_logical_sha256
            ),
            completed_prefix_result_sha256=prefix_result_shas,
        )
        existing_start, _start_staged = read_create_once_json(
            start_path, label="nested method start"
        )
        if existing_start is None:
            raise NestedCandidatePoolError(
                "existing nested triplet lacks a method start"
            )
        validate_method_start(existing_start, start_value)
        promote_create_once_json(
            start_path, existing_start, label="nested method start"
        )

        existing_method_failure, _method_failure_staged = read_create_once_json(
            method_failure_path, label="nested method failure"
        )
        if existing_method_failure is not None:
            raise NestedCandidatePoolError(
                f"existing nested triplet method {method} has a failure"
            )
        result_value, _result_staged = read_create_once_json(
            result_path, label="nested method result"
        )
        if result_value is None:
            raise NestedCandidatePoolError(
                "existing nested triplet lacks a method result"
            )
        rollout = validate_method_result(
            result_value,
            expected,
            method=method,
            method_ordinal=method_ordinal,
            method_start_sha256=start_value["method_start_sha256"],
            attempt_sha256=attempt_sha,
            commitment_sha256=commitment_sha,
            execution_contract_logical_sha256=(
                execution_contract_logical_sha256
            ),
            execution_contract_file_sha256=execution_contract_file_sha256,
            completed_prefix_result_sha256=prefix_result_shas,
        )
        result_file_sha = promote_create_once_json(
            result_path, result_value, label="nested method result"
        )
        result_logical_sha = str(result_value["method_result_sha256"])
        rollouts[method] = rollout
        method_result_bindings[method] = {
            "logical_sha256": result_logical_sha,
            "file_sha256": result_file_sha,
        }
        prefix_result_shas.append(result_logical_sha)

    computed_pair = materialize_triplet(
        expected,
        rollouts,
        commitment=commitment,
        attempt_sha256=attempt_sha,
        execution_contract_logical_sha256=execution_contract_logical_sha256,
        method_result_bindings=method_result_bindings,
    )
    if dict(existing_pair) != computed_pair:
        raise NestedCandidatePoolError(
            "existing nested triplet differs from its method results"
        )
    validate_triplet(
        existing_pair,
        expected,
        execution_contract_logical_sha256=execution_contract_logical_sha256,
        expected_method_result_bindings=method_result_bindings,
    )
    promote_create_once_json(pair_path, existing_pair, label="nested triplet")
    return dict(existing_pair)


def outcome_row(pair: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "benchmark": BENCHMARK,
        "task": TASK,
        "heldout_body": pair["heldout_body"],
        "condition": pair["condition"],
        "requested_seed": pair["requested_seed"],
        "method_order": pair["method_order"],
        "pair_sha256": pair["pair_sha256"],
    }
    for method in METHODS:
        rollout = pair["rollouts"][method]
        result[f"{method}_binary_success"] = rollout["binary_success"]
        result[f"{method}_stage_progress"] = rollout["stage_progress"]
    return result


def _exact_mcnemar_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _comparison_summary(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, seed: int
) -> dict[str, Any]:
    if not rows:
        raise NestedCandidatePoolError("comparison summary cannot be empty")
    left_success = np.asarray(
        [row[f"{left}_binary_success"] for row in rows], dtype=np.float64
    )
    right_success = np.asarray(
        [row[f"{right}_binary_success"] for row in rows], dtype=np.float64
    )
    left_stage = np.asarray(
        [row[f"{left}_stage_progress"] for row in rows], dtype=np.float64
    )
    right_stage = np.asarray(
        [row[f"{right}_stage_progress"] for row in rows], dtype=np.float64
    )
    success_delta = right_success - left_success
    stage_delta = right_stage - left_stage
    selected_cells = sorted(
        {
            (str(row["heldout_body"]), str(row["condition"]))
            for row in rows
        }
    )
    expected_seeds = list(range(SEED_BASE, SEED_BASE + SEED_COUNT))
    indices_by_seed: dict[int, list[int]] = {value: [] for value in expected_seeds}
    for index, row in enumerate(rows):
        requested_seed = row.get("requested_seed")
        if isinstance(requested_seed, bool) or requested_seed not in indices_by_seed:
            raise NestedCandidatePoolError(
                "comparison rows do not use the frozen nested seed roster"
            )
        indices_by_seed[int(requested_seed)].append(index)
    expected_rows_per_cluster = len(selected_cells)
    if expected_rows_per_cluster <= 0 or any(
        len(indices) != expected_rows_per_cluster
        for indices in indices_by_seed.values()
    ):
        raise NestedCandidatePoolError(
            "requested-seed clusters lack complete body-condition coverage"
        )
    for requested_seed, indices in indices_by_seed.items():
        observed_cells = {
            (
                str(rows[index]["heldout_body"]),
                str(rows[index]["condition"]),
            )
            for index in indices
        }
        if observed_cells != set(selected_cells):
            raise NestedCandidatePoolError(
                f"requested-seed cluster {requested_seed} duplicates or loses a cell"
            )
    success_cluster_delta = np.asarray(
        [success_delta[indices].mean() for indices in indices_by_seed.values()],
        dtype=np.float64,
    )
    stage_cluster_delta = np.asarray(
        [stage_delta[indices].mean() for indices in indices_by_seed.values()],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    success_bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    stage_bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        cluster_indices = generator.integers(0, SEED_COUNT, size=SEED_COUNT)
        success_bootstrap[replicate] = success_cluster_delta[
            cluster_indices
        ].mean()
        stage_bootstrap[replicate] = stage_cluster_delta[cluster_indices].mean()
    left_only = int(np.sum((left_success == 1) & (right_success == 0)))
    right_only = int(np.sum((left_success == 0) & (right_success == 1)))
    mcnemar_p = _exact_mcnemar_two_sided(left_only, right_only)
    cell_count = len(selected_cells)
    return {
        "left_method": left,
        "right_method": right,
        "pair_count": len(rows),
        "left_success_rate": float(left_success.mean()),
        "right_success_rate": float(right_success.mean()),
        "paired_success_rate_delta_right_minus_left": float(success_delta.mean()),
        "paired_success_delta_percentile_95_interval": [
            float(value)
            for value in np.quantile(success_bootstrap, [0.025, 0.975])
        ],
        "paired_success_delta_interval_contract": {
            "method": "paired_requested_seed_cluster_percentile_bootstrap",
            "cluster_unit": (
                "requested_seed_with_all_selected_body_condition_rows_kept_together"
            ),
            "cluster_count": SEED_COUNT,
            "rows_per_cluster": expected_rows_per_cluster,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": seed,
            "replacement": True,
            "inference_scope": (
                "new_initial_conditions_on_the_selected_fixed_heldout_bodies"
            ),
        },
        "left_stage_progress_mean": float(left_stage.mean()),
        "right_stage_progress_mean": float(right_stage.mean()),
        "paired_stage_progress_delta_right_minus_left": float(stage_delta.mean()),
        "paired_stage_delta_percentile_95_interval": [
            float(value) for value in np.quantile(stage_bootstrap, [0.025, 0.975])
        ],
        "paired_stage_delta_interval_contract": {
            "method": "paired_requested_seed_cluster_percentile_bootstrap",
            "cluster_unit": (
                "requested_seed_with_all_selected_body_condition_rows_kept_together"
            ),
            "cluster_count": SEED_COUNT,
            "rows_per_cluster": expected_rows_per_cluster,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": seed,
            "replacement": True,
            "inference_scope": (
                "new_initial_conditions_on_the_selected_fixed_heldout_bodies"
            ),
        },
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "mcnemar_contract": {
            "selected_body_condition_cell_count": cell_count,
            "requested_seed_repeated_across_cells": cell_count > 1,
            "repeated_seed_dependence_accounted_for": False,
            "inferentially_valid_for_this_reporting_unit": cell_count == 1,
            "role": "inferential" if cell_count == 1 else "descriptive_only",
            "scope_note": (
                "inferential only for one body-condition cell with distinct seeds; "
                "multi-cell summaries repeat requested seeds"
            ),
        },
    }


def _report_scope(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    return {
        "actor_to_nested_n4": _comparison_summary(
            rows, METHOD_ACTOR, METHOD_N4, seed=seed
        ),
        "actor_to_nested_n8": _comparison_summary(
            rows, METHOD_ACTOR, METHOD_N8, seed=seed + 1
        ),
        "nested_n4_to_nested_n8_primary_pool_size_estimand": _comparison_summary(
            rows, METHOD_N4, METHOD_N8, seed=seed + 2
        ),
    }


def validate_complete_outcome_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require the exact prospective 5-body/2-condition/100-seed roster."""

    expected_schedule = evaluation_schedule()
    if len(rows) != len(expected_schedule):
        raise NestedCandidatePoolError(
            "nested outcomes are not the complete 1000-triplet schedule"
        )
    for index, (row, expected) in enumerate(zip(rows, expected_schedule)):
        if not isinstance(row, Mapping):
            raise NestedCandidatePoolError(
                f"nested outcome row {index} is not an object"
            )
        if any(row.get(key) != expected[key] for key in expected):
            raise NestedCandidatePoolError(
                f"nested outcome row {index} changed the frozen schedule"
            )
        pair_sha = row.get("pair_sha256")
        if (
            not isinstance(pair_sha, str)
            or len(pair_sha) != 64
            or any(character not in "0123456789abcdef" for character in pair_sha)
        ):
            raise NestedCandidatePoolError(
                f"nested outcome row {index} lacks a valid pair SHA-256"
            )
        for method in METHODS:
            success = row.get(f"{method}_binary_success")
            stage = row.get(f"{method}_stage_progress")
            if (
                type(success) is not int
                or success not in (0, 1)
                or isinstance(stage, bool)
                or not isinstance(stage, (int, float, np.integer, np.floating))
                or not math.isfinite(float(stage))
                or not 0.0 <= float(stage) <= 1.0
            ):
                raise NestedCandidatePoolError(
                    f"nested outcome row {index} has an invalid {method} outcome"
                )


def build_outcome_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_contract_logical_sha256: str,
    execution_contract_file_sha256: str,
) -> dict[str, Any]:
    validate_complete_outcome_rows(rows)
    normalized = [dict(row) for row in rows]
    base = {
        "format": OUTCOME_FORMAT,
        "status": "complete_1000_initial_condition_triplets_3000_rollouts",
        "pair_count": len(normalized),
        "rollout_count": len(normalized) * len(METHODS),
        "methods": list(METHODS),
        "rows": normalized,
        "rows_sha256": canonical_sha256(normalized),
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "execution_contract_file_sha256": execution_contract_file_sha256,
        "reference_preregistration_sha256": REFERENCE_PREREGISTRATION_SHA256,
        "nested_evaluation_protocol": nested_evaluation_protocol(),
        "postformal_not_part_of_reference_preregistration": True,
    }
    return {**base, "document_sha256": canonical_sha256(base)}


def build_report(
    rows: Sequence[Mapping[str, Any]], *, outcome_document_sha256: str
) -> dict[str, Any]:
    validate_complete_outcome_rows(rows)
    normalized = [dict(row) for row in rows]
    base = {
        "format": REPORT_FORMAT,
        "status": "complete_shared_raw16_nested_n4_n8_paired_report",
        "outcome_document_sha256": outcome_document_sha256,
        "nested_pool_contract": nested_pool_contract(),
        "nested_evaluation_protocol": nested_evaluation_protocol(),
        "primary_estimand": (
            "paired_policy_success_and_stage_progress_delta_nested_n8_minus_"
            "nested_n4_on_identical_initial_conditions"
        ),
        "overall": _report_scope(normalized, seed=BOOTSTRAP_SEED),
        "by_heldout_body": {
            body: _report_scope(
                [row for row in normalized if row["heldout_body"] == body],
                seed=BOOTSTRAP_SEED + 10 + index * 10,
            )
            for index, body in enumerate(BODIES)
        },
        "by_heldout_body_and_condition": {
            f"{body}|{condition}": _report_scope(
                [
                    row
                    for row in normalized
                    if row["heldout_body"] == body
                    and row["condition"] == condition
                ],
                seed=BOOTSTRAP_SEED + 100 + index * 10,
            )
            for index, (body, condition) in enumerate(
                (body, condition) for body in BODIES for condition in CONDITIONS
            )
        },
    }
    return {**base, "report_sha256": canonical_sha256(base)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--reference-preregistration", type=Path, required=True)
    parser.add_argument("--lobo-fold", action="append", required=True)
    parser.add_argument("--required-supplement-binding-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-exec-steps", type=int, default=ACTION_EXEC_STEPS)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=float, default=ACTOR_DATASET_FPS)
    parser.add_argument("--instruction", default=collector.DEFAULT_INSTRUCTION)
    return parser.parse_args(argv)


def implementation_binding(robotwin_root: Path) -> dict[str, Any]:
    inherited = formal.implementation_binding(robotwin_root)
    paths = [
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(pool_runner) or "").resolve(),
    ]
    return {
        **inherited,
        "nested_runner_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "nested_pool_contract": nested_pool_contract(),
        "formal_n4_or_existing_n8_output_mutated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise NestedCandidatePoolError("nested execution requires remote RTX 4090")
    if args.action_exec_steps != ACTION_EXEC_STEPS or args.max_steps != 200:
        raise NestedCandidatePoolError("action-exec-steps/max-steps must remain 5/200")
    if args.fps != ACTOR_DATASET_FPS:
        raise NestedCandidatePoolError("actor control interval must remain 15 Hz")
    if args.instruction != collector.DEFAULT_INSTRUCTION:
        raise NestedCandidatePoolError("actor instruction changed")
    inputs = (
        args.actor_checkpoint,
        args.vlm_metadata_path,
        args.robotwin_root,
        args.event_spec,
        args.reference_preregistration,
    )
    if any(not path.expanduser().resolve().exists() for path in inputs):
        raise FileNotFoundError("one or more required static inputs are missing")

    random.seed(BOOTSTRAP_SEED)
    np.random.seed(BOOTSTRAP_SEED)
    torch.manual_seed(BOOTSTRAP_SEED)
    robotwin_root = args.robotwin_root.expanduser().resolve()
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(robotwin_root))

    reference = formal.load_preregistration(
        args.reference_preregistration.expanduser().resolve()
    )
    fold_paths = formal.parse_fold_specs(args.lobo_fold)
    folds = {body: formal.inspect_fold(body, fold_paths[body]) for body in BODIES}
    fold_training_regime = formal.inspect_fold_training_regime(
        folds,
        required_supplement_binding_sha256=args.required_supplement_binding_sha256,
    )
    actor_checkpoint = args.actor_checkpoint.expanduser().resolve()
    actor_tree_sha, actor_file_count, actor_size = shared_head.sha256_tree(
        actor_checkpoint
    )
    vlm_metadata = args.vlm_metadata_path.expanduser().resolve()
    vlm_tree_sha, vlm_file_count, vlm_size = shared_head.sha256_tree(vlm_metadata)
    event_spec_path = args.event_spec.expanduser().resolve()
    if sha256_file(event_spec_path) != EVENT_SPEC_SHA256:
        raise NestedCandidatePoolError("event specification differs from training")
    try:
        _event_spec, calibration = analytic_event.load_event_spec(event_spec_path)
    except analytic_event.AnalyticEventSpecError as error:
        raise NestedCandidatePoolError(str(error)) from error

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs_dir = output / "pairs"
    attempts_dir = output / "attempts"
    commitments_dir = output / "initial_commitments"
    failures_dir = output / "failures"
    method_starts_dir = output / "method_starts"
    method_resumes_dir = output / "method_resumes"
    method_results_dir = output / "method_results"
    method_failures_dir = output / "method_failures"
    for directory in (
        pairs_dir,
        attempts_dir,
        commitments_dir,
        failures_dir,
        method_starts_dir,
        method_resumes_dir,
        method_results_dir,
        method_failures_dir,
    ):
        directory.mkdir(exist_ok=True)
    outcome_path = output / "nested_paired_outcomes.json"
    report_path = output / "nested_n4_n8_report.json"
    contract_path = output / "execution_contract.json"
    completion_path = output / "completion_receipt.json"
    schedule = evaluation_schedule()
    contract_base = {
        "format": CONTRACT_FORMAT,
        "runner_format": FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        "bodies": list(BODIES),
        "conditions": list(CONDITIONS),
        "evaluation_seed_base": SEED_BASE,
        "evaluation_seed_count": SEED_COUNT,
        "initial_condition_triplet_count": len(schedule),
        "rollout_count": len(schedule) * len(METHODS),
        "methods": list(METHODS),
        "nested_pool_contract": nested_pool_contract(),
        "same_requested_seed_and_complete_reset_tripled": True,
        "method_order_rotated_before_outcomes": True,
        "method_result_persistence": {
            "state_machine": (
                "fixed_order_create_once_start_result_prefix_then_triplet"
            ),
            "result_write": "deterministic_stage_fsync_hard_link_fsync_directory",
            "existing_result_overwrite_or_retry_allowed": False,
            "automatic_noninformative_resume_limit_per_method": 1,
            "resume_assumption": (
                "process_or_host_interruption_before_create_once_result_is_"
                "noninformative_about_unpersisted_outcome"
            ),
            "exception_or_action_failure_retry_allowed": False,
            "later_method_before_complete_prefix_allowed": False,
        },
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
        "candidate_rank_ensemble_contracts": {
            "nested_n4": shared_head.risk_adjusted_rank_ensemble_contract(),
            "nested_n8": pool_runner.runtime_rank_ensemble_contract(
                N8_CANDIDATE_COUNT
            ),
        },
        "event_spec": str(event_spec_path),
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "reference_preregistration": str(
            args.reference_preregistration.expanduser().resolve()
        ),
        "reference_preregistration_sha256": reference["preregistration_sha256"],
        "nested_evaluation_protocol": nested_evaluation_protocol(),
        "postformal_not_part_of_reference_preregistration": True,
        "action_exec_steps": ACTION_EXEC_STEPS,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "instruction": args.instruction,
        "runtime_binding": implementation_binding(robotwin_root),
        "no_training": True,
        "formal_n4_or_existing_n8_output_mutated": False,
        "official_expert_or_protected_internal_payloads_opened": False,
    }
    contract = {**contract_base, "logical_sha256": canonical_sha256(contract_base)}
    contract_file_sha = promote_create_once_json(
        contract_path, contract, label="nested execution contract"
    )

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
    actor_config.vlm_model_name = str(vlm_metadata)
    actor_config.load_vlm_weights = False
    if (
        actor_config.action_feature is None
        or int(actor_config.action_feature.shape[0]) != NATIVE_EE_DIM
        or actor_config.input_features.get("observation.state") is None
        or int(actor_config.input_features["observation.state"].shape[0])
        != NATIVE_EE_DIM
    ):
        raise NestedCandidatePoolError("frozen actor is not state16/action16 EE")
    policy = SmolVLAPolicy.from_pretrained(
        actor_checkpoint, config=actor_config, local_files_only=True, strict=True
    ).eval().to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=actor_config,
        pretrained_path=str(actor_checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(vlm_metadata)},
        },
    )

    rows = []
    completed = 0
    active_body = None
    ensemble: list[shared_head.EffectAlignedSharedEventHead] = []
    started = time.time()

    def record_completed_pair(
        pair: Mapping[str, Any],
        *,
        body: str,
        expected: Mapping[str, Any],
        identity: str,
    ) -> None:
        nonlocal completed
        rows.append(outcome_row(pair))
        completed += 1
        atomic_json(
            output / "progress.json",
            {
                "format": FORMAT,
                "status": (
                    "running" if completed < len(schedule) else "rollouts_complete"
                ),
                "completed_initial_condition_triplets": completed,
                "completed_rollouts": completed * len(METHODS),
                "total_initial_condition_triplets": len(schedule),
                "last_pair": identity,
                "wall_seconds": time.time() - started,
            },
        )
        print(
            "NESTED_N4_N8_TRIPLET_COMPLETE="
            + json.dumps(
                {
                    "completed": completed,
                    "total": len(schedule),
                    "heldout_body": body,
                    "condition": expected["condition"],
                    "requested_seed": expected["requested_seed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for expected in schedule:
        body = str(expected["heldout_body"])
        identity = pair_id(body, expected["condition"], expected["requested_seed"])
        pair_path = pairs_dir / f"{identity}.json"
        attempt_path = attempts_dir / f"{identity}.json"
        commitment_path = commitments_dir / f"{identity}.json"
        failure_path = failures_dir / f"{identity}.json"
        existing_failure, _failure_staged = read_create_once_json(
            failure_path, label="nested triplet failure"
        )
        if existing_failure is not None:
            raise NestedCandidatePoolError(
                "nested triplet has an immutable prior failure"
            )
        attempt_base = {
            "format": "etsf_robotwin2_nested_n4_n8_attempt_v2",
            "status": "started_once_fixed_method_order_with_bounded_resume",
            "pair_id": identity,
            **dict(expected),
            "execution_contract_logical_sha256": contract["logical_sha256"],
            "execution_contract_file_sha256": contract_file_sha,
            "attempt_number": 1,
        }
        attempt_sha = canonical_sha256(attempt_base)
        attempt = {**attempt_base, "attempt_sha256": attempt_sha}
        pair = recover_complete_existing_triplet(
            pair_path=pair_path,
            attempt_path=attempt_path,
            commitment_path=commitment_path,
            method_starts_dir=method_starts_dir,
            method_results_dir=method_results_dir,
            method_failures_dir=method_failures_dir,
            identity=identity,
            expected=expected,
            attempt=attempt,
            execution_contract_logical_sha256=contract["logical_sha256"],
            execution_contract_file_sha256=contract_file_sha,
        )
        if pair is not None:
            record_completed_pair(
                pair, body=body, expected=expected, identity=identity
            )
            continue
        if body != active_body:
            del ensemble
            gc.collect()
            torch.cuda.empty_cache()
            ensemble = formal.load_ensemble(folds[body], device)
            active_body = body
        promote_create_once_json(
            attempt_path, attempt, label="nested triplet attempt"
        )
        task_args = collector._load_task_args(
            robotwin_root, body, str(expected["condition"])
        )
        task_args["step_lim"] = args.max_steps
        try:
            commitment, commitment_staged = read_create_once_json(
                commitment_path, label="nested initial commitment"
            )
            if commitment is None:
                commitment = prepare_initial_commitment(
                    body=body,
                    condition=str(expected["condition"]),
                    seed=int(expected["requested_seed"]),
                    task_class=task_class,
                    task_args=task_args,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    instruction=args.instruction,
                    device=device,
                )
            validate_stored_initial_commitment(commitment, expected)
            promote_create_once_json(
                commitment_path,
                commitment,
                label="nested initial commitment",
            )
            commitment_sha = str(commitment["commitment_sha256"])
            rollouts: dict[str, Mapping[str, Any]] = {}
            method_result_bindings: dict[str, dict[str, str]] = {}
            prefix_result_shas: list[str] = []
            ordered_methods = list(expected["method_order"])
            for method_ordinal, method in enumerate(ordered_methods):
                stem = f"{identity}.{method_ordinal:02d}.{method}"
                start_path = method_starts_dir / f"{stem}.json"
                resume_path = method_resumes_dir / f"{stem}.json"
                result_path = method_results_dir / f"{stem}.json"
                method_failure_path = method_failures_dir / f"{stem}.json"
                start_value = build_method_start(
                    expected,
                    method=method,
                    method_ordinal=method_ordinal,
                    attempt_sha256=attempt_sha,
                    commitment_sha256=commitment_sha,
                    execution_contract_logical_sha256=contract["logical_sha256"],
                    completed_prefix_result_sha256=prefix_result_shas,
                )
                existing_start, start_staged = read_create_once_json(
                    start_path, label="nested method start"
                )
                start_created_now = existing_start is None
                if existing_start is not None:
                    validate_method_start(existing_start, start_value)
                promote_create_once_json(
                    start_path, start_value, label="nested method start"
                )
                existing_method_failure, _method_failure_staged = (
                    read_create_once_json(
                        method_failure_path, label="nested method failure"
                    )
                )
                if existing_method_failure is not None:
                    raise NestedCandidatePoolError(
                        f"nested method {method} has an immutable prior failure"
                    )
                result_value, result_staged = read_create_once_json(
                    result_path, label="nested method result"
                )
                if result_value is None:
                    for later_ordinal, later_method in enumerate(
                        ordered_methods[method_ordinal + 1 :],
                        start=method_ordinal + 1,
                    ):
                        later_stem = f"{identity}.{later_ordinal:02d}.{later_method}"
                        later_paths = (
                            method_starts_dir / f"{later_stem}.json",
                            method_results_dir / f"{later_stem}.json",
                            method_failures_dir / f"{later_stem}.json",
                        )
                        if any(
                            path.exists() or _staged_path(path).exists()
                            for path in later_paths
                        ):
                            raise NestedCandidatePoolError(
                                "nested method artifacts are not a complete prefix"
                            )
                    if not start_created_now:
                        resume_base = {
                            "format": METHOD_RESUME_FORMAT,
                            "status": "single_automatic_noninformative_resume",
                            "pair_id": identity,
                            **dict(expected),
                            "method": method,
                            "method_ordinal": method_ordinal,
                            "method_start_sha256": start_value[
                                "method_start_sha256"
                            ],
                            "attempt_sha256": attempt_sha,
                            "commitment_sha256": commitment_sha,
                            "resume_number": 1,
                            "resume_trigger": (
                                "missing_result_and_failure_after_process_or_host_"
                                "interruption"
                            ),
                            "noninformative_interruption_assumption": True,
                        }
                        resume_value = {
                            **resume_base,
                            "method_resume_sha256": canonical_sha256(resume_base),
                        }
                        existing_resume, _resume_staged = read_create_once_json(
                            resume_path, label="nested method resume"
                        )
                        if existing_resume is not None:
                            raise NestedCandidatePoolError(
                                "nested method exceeded its one-resume limit"
                            )
                        promote_create_once_json(
                            resume_path, resume_value, label="nested method resume"
                        )
                    try:
                        rollout = execute_rollout(
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
                            max_steps=args.max_steps,
                            device=device,
                        )
                        result_value = build_method_result(
                            expected,
                            method=method,
                            method_ordinal=method_ordinal,
                            rollout=rollout,
                            method_start_sha256=start_value[
                                "method_start_sha256"
                            ],
                            attempt_sha256=attempt_sha,
                            commitment_sha256=commitment_sha,
                            execution_contract_logical_sha256=contract[
                                "logical_sha256"
                            ],
                            execution_contract_file_sha256=contract_file_sha,
                            completed_prefix_result_sha256=prefix_result_shas,
                        )
                        promote_create_once_json(
                            result_path,
                            result_value,
                            label="nested method result",
                        )
                    except Exception as method_error:
                        method_failure_base = {
                            "format": METHOD_FAILURE_FORMAT,
                            "status": "failed_once_no_retry",
                            "pair_id": identity,
                            **dict(expected),
                            "method": method,
                            "method_ordinal": method_ordinal,
                            "method_start_sha256": start_value[
                                "method_start_sha256"
                            ],
                            "attempt_sha256": attempt_sha,
                            "commitment_sha256": commitment_sha,
                            "error_type": type(method_error).__name__,
                            "error_message": str(method_error),
                        }
                        method_failure_value = {
                            **method_failure_base,
                            "method_failure_sha256": canonical_sha256(
                                method_failure_base
                            ),
                        }
                        promote_create_once_json(
                            method_failure_path,
                            method_failure_value,
                            label="nested method failure",
                        )
                        raise
                rollout = validate_method_result(
                    result_value,
                    expected,
                    method=method,
                    method_ordinal=method_ordinal,
                    method_start_sha256=start_value["method_start_sha256"],
                    attempt_sha256=attempt_sha,
                    commitment_sha256=commitment_sha,
                    execution_contract_logical_sha256=contract["logical_sha256"],
                    execution_contract_file_sha256=contract_file_sha,
                    completed_prefix_result_sha256=prefix_result_shas,
                )
                result_file_sha = promote_create_once_json(
                    result_path, result_value, label="nested method result"
                )
                result_logical_sha = str(result_value["method_result_sha256"])
                rollouts[method] = rollout
                method_result_bindings[method] = {
                    "logical_sha256": result_logical_sha,
                    "file_sha256": result_file_sha,
                }
                prefix_result_shas.append(result_logical_sha)
            computed_pair = materialize_triplet(
                expected,
                rollouts,
                commitment=commitment,
                attempt_sha256=attempt_sha,
                execution_contract_logical_sha256=contract["logical_sha256"],
                method_result_bindings=method_result_bindings,
            )
            pair_existing, pair_staged = read_create_once_json(
                pair_path, label="nested triplet"
            )
            if pair_existing is not None and dict(pair_existing) != computed_pair:
                raise NestedCandidatePoolError(
                    "existing nested triplet differs from its method results"
                )
            pair = computed_pair
            if pair.get("attempt_sha256") != attempt_sha:
                raise NestedCandidatePoolError("nested triplet attempt binding changed")
            validate_triplet(
                pair,
                expected,
                execution_contract_logical_sha256=contract["logical_sha256"],
                expected_method_result_bindings=method_result_bindings,
            )
            promote_create_once_json(pair_path, pair, label="nested triplet")
        except Exception as error:
            failure_base = {
                "format": "etsf_robotwin2_nested_n4_n8_attempt_failure_v2",
                "status": "failed_no_automatic_retry",
                "pair_id": identity,
                "attempt_sha256": attempt_sha,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            failure_value = {
                **failure_base,
                "failure_sha256": canonical_sha256(failure_base),
            }
            promote_create_once_json(
                failure_path, failure_value, label="nested triplet failure"
            )
            raise
        record_completed_pair(
            pair, body=body, expected=expected, identity=identity
        )

    document = build_outcome_document(
        rows,
        execution_contract_logical_sha256=contract["logical_sha256"],
        execution_contract_file_sha256=contract_file_sha,
    )
    promote_create_once_json(
        outcome_path, document, label="nested outcome document"
    )
    report = build_report(rows, outcome_document_sha256=document["document_sha256"])
    promote_create_once_json(report_path, report, label="nested paired report")
    completion_base = {
        "format": COMPLETION_FORMAT,
        "status": "complete_1000_triplets_3000_rollouts_frozen",
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": contract_file_sha,
        "outcome_document_sha256": document["document_sha256"],
        "outcome_file_sha256": sha256_file(outcome_path),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": sha256_file(report_path),
        "initial_condition_triplet_count": len(rows),
        "rollout_count": len(rows) * len(METHODS),
        "nested_evaluation_protocol_logical_sha256": (
            nested_evaluation_protocol()["logical_sha256"]
        ),
        "postformal_not_part_of_reference_preregistration": True,
    }
    completion = {
        **completion_base,
        "logical_sha256": canonical_sha256(completion_base),
    }
    promote_create_once_json(
        completion_path, completion, label="nested completion receipt"
    )
    print(
        "NESTED_N4_N8_COMPLETE="
        + json.dumps(
            {
                "triplets": len(rows),
                "rollouts": len(rows) * len(METHODS),
                "outcome": str(outcome_path),
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAT",
    "METHODS",
    "METHOD_ACTOR",
    "METHOD_N4",
    "METHOD_N8",
    "N4_CANDIDATE_COUNT",
    "N8_CANDIDATE_COUNT",
    "NestedCandidatePoolError",
    "RAW_PROPOSAL_COUNT",
    "build_outcome_document",
    "build_report",
    "build_method_result",
    "build_method_start",
    "evaluation_schedule",
    "existing_separate_n4_n8_comparability_audit",
    "materialize_triplet",
    "nested_pool_contract",
    "nested_evaluation_protocol",
    "nested_pool_selection_audit",
    "outcome_row",
    "promote_create_once_json",
    "read_create_once_json",
    "recover_complete_existing_triplet",
    "validate_complete_outcome_rows",
    "validate_nested_pool_audit",
    "validate_method_result",
]
