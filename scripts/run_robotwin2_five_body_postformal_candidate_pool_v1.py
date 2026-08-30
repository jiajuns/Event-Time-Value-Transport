#!/usr/bin/env python3
"""Run a postformal N=8/16 actor-flow candidate-pool paired study.

This experiment is deliberately separate from the preregistered best-of-four
study.  By default every actor query makes exactly N actor proposals using a
two-language flow-noise factorial roster and retains their original order
(N=8 by default, optionally N=16).  The raw16 roster keeps four full iid draws
per language and adds first-five-step and full-chunk XYZ-only conditional
antithetic pairs.  Every individual noise tensor remains exactly marginal
N(0,I), while proposal zero remains bit-exact legacy noise.  One language
condition is the frozen runtime instruction; the other is an exact instruction
from actor training.  A separately budgeted run may explicitly draw more than N
proposals, in which case legacy
proposal zero remains final candidate zero and the remaining N-1 proposals are
selected with deterministic farthest-point sampling in the canonical
executed-prefix effect space fixed by the actor-execution protocol.  Selection
is label/outcome/critic blind.  The
same frozen five-member LOBO critic then scores the retained candidates.

The baseline always executes raw proposal zero, which keeps both the legacy
runtime instruction and legacy flow-noise draw.  Both methods use the same
requested seed and reproduce the same initial reset, raw proposals and retained
ordered pool.  This runner never mutates the formal four-candidate collector or
its output root and must run only as a new postformal remote RTX-4090 job.
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
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_robotwin2_five_body_paired_success_v1 as formal


collector = formal.collector
shared_head = formal.shared_head
analytic_event = formal.analytic_event
FORMAT = "etsf_robotwin2_five_body_postformal_candidate_pool_v2_actor_protocol"
PAIR_FORMAT = (
    "etsf_robotwin2_postformal_actor_flow_pool_paired_execution_v2_actor_protocol"
)
CONTRACT_FORMAT = "etsf_robotwin2_postformal_candidate_pool_contract_v2_actor_protocol"
OUTCOME_FORMAT = "etsf_robotwin2_postformal_candidate_pool_outcomes_v1"
REPORT_FORMAT = "etsf_robotwin2_postformal_candidate_pool_report_v1"
BENCHMARK = formal.BENCHMARK
TASK = formal.TASK
BODIES = formal.BODIES
CONDITIONS = formal.CONDITIONS
SEED_BASE = formal.SEED_BASE
SEED_COUNT = formal.SEED_COUNT
SUPPORTED_CANDIDATE_COUNTS = (8, 16)
PROPOSAL_MULTIPLIER = 2
NATIVE_EE_DIM = formal.NATIVE_EE_DIM
ACTION_EXEC_STEPS = formal.ACTION_EXEC_STEPS
ACTOR_DATASET_FPS = formal.ACTOR_DATASET_FPS
PLANNED_DT_SECONDS = formal.PLANNED_DT_SECONDS
CONDITIONAL_FIRST_FIVE_STEPS = 5
QUERY_CANONICALIZATION_STEPS = formal.QUERY_CANONICALIZATION_STEPS
STAGE_DENOMINATOR = formal.STAGE_DENOMINATOR
EVENT_SPEC_SHA256 = formal.EVENT_SPEC_SHA256
REFERENCE_PREREGISTRATION_SHA256 = formal.PREREGISTRATION_SHA256
REPORT_BOOTSTRAP_REPLICATES = 10_000
REPORT_BOOTSTRAP_SEED = 20260908
# This exact task string is task_index=1068 in the frozen 2,750-episode
# LeRobot actor-training metadata.  It was selected without rollout outcomes
# because it states the same relation without binding color, lid, arm, or
# other appearance attributes that change under randomized evaluation.  The
# value is embedded so execution never reads mutable training metadata.
TRAINING_SEEN_INSTRUCTION_TASK_INDEX = 1068
TRAINING_TASKS_PARQUET_SHA256 = (
    "576e5ff827cc6aae8a283f65e6196e11cd6577d65d99842cebb8aaf63f6dde34"
)
TRAINING_SEEN_INSTRUCTION = (
    "Move the sauce can for condiments to be next to the metal cooking pot"
)
TRAINING_SEEN_INSTRUCTION_SHA256 = hashlib.sha256(
    TRAINING_SEEN_INSTRUCTION.encode("utf-8")
).hexdigest()
SOURCE_TRAIN_NORMALIZED_DIVERSITY_FORMAT = (
    "etsf_source_train_normalized_canonical_effect_diversity_v1"
)
POOL_AUDIT_FORMAT = "etsf_robotwin2_canonical_effect_candidate_pool_audit_v3_actor_protocol"
FLOW_NOISE_CONTRACT_FORMAT = (
    "etsf_postformal_conditional_translation_flow_noise_contract_v2"
)
RAW16_PROPOSAL_COUNT = 16
RAW16_NOISE_PER_INSTRUCTION = 8
NATIVE_XYZ_NOISE_CHANNELS = (0, 1, 2, 8, 9, 10)
CANONICAL_TRANSLATION_EFFECT_CHANNELS = (0, 1, 2, 7, 8, 9)


class PostformalCandidatePoolError(RuntimeError):
    """A postformal pool, paired reset, checkpoint, or receipt changed."""


def configure_actor_execution_protocol(
    protocol: Mapping[str, Any],
    *,
    path: Path,
    file_sha256: str,
    path_root: Path,
) -> dict[str, Any]:
    """Configure formal and postformal consumers from one file binding."""

    try:
        validated = formal.configure_actor_execution_protocol(
            protocol,
            path=path,
            file_sha256=file_sha256,
            path_root=path_root,
        )
    except formal.PairedExecutionError as error:
        raise PostformalCandidatePoolError(str(error)) from error
    global ACTION_EXEC_STEPS, PLANNED_DT_SECONDS
    ACTION_EXEC_STEPS = int(validated["stride"])
    PLANNED_DT_SECONDS = ACTION_EXEC_STEPS / ACTOR_DATASET_FPS
    return dict(validated)


canonical_sha256 = formal.canonical_sha256
sha256_file = formal.sha256_file
array_sha256 = formal.array_sha256
atomic_json = formal.atomic_json
pair_id = formal.pair_id


def method_name(candidate_count: int) -> str:
    validate_candidate_count(candidate_count)
    # Keep the method name valid for the default raw=N path: canonical-effect
    # FPS is an optional oversampling contract, not a property we claim for
    # every run.
    return f"etsf_actor_flow_best_of_{candidate_count}"


def methods(candidate_count: int) -> tuple[str, str]:
    return ("actor_baseline", method_name(candidate_count))


def validate_candidate_count(candidate_count: int) -> int:
    if isinstance(candidate_count, bool) or candidate_count not in SUPPORTED_CANDIDATE_COUNTS:
        raise PostformalCandidatePoolError("candidate count must be exactly 8 or 16")
    return int(candidate_count)


def proposal_count(candidate_count: int, requested: int | None = None) -> int:
    """Resolve a bounded proposal budget; the executable default is raw=N."""

    candidate_count = validate_candidate_count(candidate_count)
    value = candidate_count if requested is None else requested
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < candidate_count
        or value > PROPOSAL_MULTIPLIER * candidate_count
        or value % 2 != 0
    ):
        raise PostformalCandidatePoolError(
            "proposal count must be an even integer in [candidate_count, 2*candidate_count]"
        )
    return int(value)


def validate_candidates(
    candidates: Any, *, expected_count: int, label: str
) -> np.ndarray:
    expected_count = int(expected_count)
    value = np.asarray(candidates, dtype=np.float32)
    if (
        expected_count <= 0
        or value.ndim != 3
        or value.shape[0] != expected_count
        or value.shape[1] < ACTION_EXEC_STEPS
        or value.shape[2] != NATIVE_EE_DIM
        or not np.isfinite(value).all()
    ):
        raise PostformalCandidatePoolError(
            f"{label} must be finite "
            f"[{expected_count},H>={ACTION_EXEC_STEPS},16]"
        )
    return value


def postformal_make_noise(
    config: Any,
    scene_seed: int,
    query_index: int,
    flow_noise_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic independent flow noise while preserving candidate0.

    Formal N=4 keeps the collector's antithetic contract.  This postformal
    study retains the exact legacy candidate-zero draw and creates independent
    standard-normal flow draws rather than deterministic negative pairs.  The
    caller deliberately reuses each draw once across the two language
    conditions so language OOD and flow-noise coverage share the fixed budget.
    """

    if isinstance(flow_noise_index, bool) or not isinstance(flow_noise_index, int):
        raise PostformalCandidatePoolError("flow noise index must be an integer")
    if flow_noise_index < 0:
        raise PostformalCandidatePoolError("flow noise index must be non-negative")
    if flow_noise_index == 0:
        return collector.make_noise(
            config, scene_seed, query_index, flow_noise_index, device
        )
    seed = int(
        (
            20260903
            + int(scene_seed) * 1_000_003
            + int(query_index) * 10_007
            + int(flow_noise_index) * 101
        )
        % (2**63 - 1)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (1, int(config.chunk_size), int(config.max_action_dim)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def flow_noise_construction_roster(
    raw_candidate_count: int,
) -> list[dict[str, Any]]:
    """Return the frozen per-candidate noise construction for both languages."""

    if (
        isinstance(raw_candidate_count, bool)
        or not isinstance(raw_candidate_count, int)
        or raw_candidate_count < 2
        or raw_candidate_count % 2 != 0
    ):
        raise PostformalCandidatePoolError(
            "language/noise candidate roster requires an even count >=2"
        )
    half = raw_candidate_count // 2
    if raw_candidate_count == RAW16_PROPOSAL_COUNT:
        one_language = [
            {"kind": "full_iid", "source_noise_index": 0, "sign": 1},
            {"kind": "full_iid", "source_noise_index": 1, "sign": 1},
            {"kind": "full_iid", "source_noise_index": 2, "sign": 1},
            {"kind": "full_iid", "source_noise_index": 3, "sign": 1},
            {
                "kind": "conditional_first5_xyz",
                "source_noise_index": 4,
                "sign": 1,
            },
            {
                "kind": "conditional_first5_xyz",
                "source_noise_index": 4,
                "sign": -1,
            },
            {
                "kind": "conditional_fullchunk_xyz",
                "source_noise_index": 6,
                "sign": 1,
            },
            {
                "kind": "conditional_fullchunk_xyz",
                "source_noise_index": 6,
                "sign": -1,
            },
        ]
        if len(one_language) != RAW16_NOISE_PER_INSTRUCTION:
            raise PostformalCandidatePoolError("raw16 noise roster cardinality changed")
    else:
        one_language = [
            {"kind": "full_iid", "source_noise_index": index, "sign": 1}
            for index in range(half)
        ]
    return [dict(row) for row in one_language] + [
        dict(row) for row in one_language
    ]


def postformal_rostered_noise(
    config: Any,
    scene_seed: int,
    query_index: int,
    construction: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Materialize one full-iid or conditional XYZ noise tensor.

    For a conditional candidate, coordinates outside the declared XYZ mask
    come from legacy ``z0`` and coordinates inside it come from an independent
    standard-normal draw (or its negative).  Disjoint coordinates therefore
    remain independent standard normals within every candidate: the marginal
    tensor is exactly N(0,I), despite deliberate cross-candidate correlation.
    """

    kind = construction.get("kind")
    source_index = construction.get("source_noise_index")
    sign = construction.get("sign")
    if (
        kind
        not in {
            "full_iid",
            "conditional_first5_xyz",
            "conditional_fullchunk_xyz",
        }
        or isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or source_index < 0
        or sign not in {-1, 1}
    ):
        raise PostformalCandidatePoolError("flow-noise construction is invalid")
    source = postformal_make_noise(
        config, scene_seed, query_index, source_index, device
    )
    if kind == "full_iid":
        if sign != 1:
            raise PostformalCandidatePoolError("full iid noise cannot change sign")
        return source
    if int(config.max_action_dim) <= max(NATIVE_XYZ_NOISE_CHANNELS):
        raise PostformalCandidatePoolError(
            "actor noise tensor lacks the native dual-arm XYZ channels"
        )
    base = postformal_make_noise(config, scene_seed, query_index, 0, device)
    result = base.clone()
    stop = (
        min(CONDITIONAL_FIRST_FIVE_STEPS, int(config.chunk_size))
        if kind == "conditional_first5_xyz"
        else int(config.chunk_size)
    )
    channels = torch.as_tensor(
        NATIVE_XYZ_NOISE_CHANNELS, dtype=torch.long, device=device
    )
    result[:, :stop, channels] = float(sign) * source[:, :stop, channels]
    return result


def candidate_instruction_roster(
    runtime_instruction: str, raw_candidate_count: int
) -> list[str]:
    """Return a two-condition, outcome-blind language/noise factorial roster."""

    if not isinstance(runtime_instruction, str) or not runtime_instruction.strip():
        raise PostformalCandidatePoolError("runtime instruction must be non-empty")
    if (
        isinstance(raw_candidate_count, bool)
        or not isinstance(raw_candidate_count, int)
        or raw_candidate_count < 2
        or raw_candidate_count % 2 != 0
    ):
        raise PostformalCandidatePoolError(
            "language/noise candidate roster requires an even count >=2"
        )
    if runtime_instruction == TRAINING_SEEN_INSTRUCTION:
        raise PostformalCandidatePoolError(
            "runtime and actor-training-seen instruction conditions collapsed"
        )
    half = raw_candidate_count // 2
    return [runtime_instruction] * half + [TRAINING_SEEN_INSTRUCTION] * half


def flow_noise_index_roster(raw_candidate_count: int) -> list[int]:
    """Return source draw indices for the frozen construction roster."""

    return [
        int(row["source_noise_index"])
        for row in flow_noise_construction_roster(raw_candidate_count)
    ]


def generate_postformal_flow_candidates(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: Any,
    instruction: str,
    scene_seed: int,
    query_index: int,
    candidate_count: int,
    device: torch.device,
) -> np.ndarray:
    """Run the frozen actor's real sampling path with independent flow draws."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise PostformalCandidatePoolError("raw candidate count must be an integer")
    if candidate_count < 2 or candidate_count % 2 != 0:
        raise PostformalCandidatePoolError(
            "raw candidate count must be even and at least two"
        )
    raw = collector.raw_policy_input(
        task, list(policy.config.image_features), instruction
    )
    instruction_roster = candidate_instruction_roster(instruction, candidate_count)
    noise_roster = flow_noise_construction_roster(candidate_count)
    processed_by_instruction = {}
    for candidate_instruction in dict.fromkeys(instruction_roster):
        candidate_raw = dict(raw)
        candidate_raw["task"] = candidate_instruction
        processed_by_instruction[candidate_instruction] = preprocessor(candidate_raw)
    candidates = []
    for candidate_instruction, noise_construction in zip(
        instruction_roster, noise_roster, strict=True
    ):
        policy.reset()
        noise = postformal_rostered_noise(
            policy.config,
            scene_seed,
            query_index,
            noise_construction,
            device,
        )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(
                dict(processed_by_instruction[candidate_instruction]), noise=noise
            )
        actions = postprocessor(normalized)
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        candidates.append(collector.normalize_ee_chunk(actions))
    result = np.stack(candidates)
    if not np.any(result[1:] != result[0]):
        raise PostformalCandidatePoolError(
            "independent flow-noise candidates collapsed to candidate zero"
        )
    return result


def canonical_effect_embeddings(
    current_ee: np.ndarray, proposals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return protocol-executed-prefix effects and flattened vectors."""

    current = np.asarray(current_ee, dtype=np.float32)
    if current.shape != (NATIVE_EE_DIM,) or not np.isfinite(current).all():
        raise PostformalCandidatePoolError("current EE state must be finite 16-D")
    value = np.asarray(proposals, dtype=np.float32)
    if (
        value.ndim != 3
        or value.shape[0] < 2
        or value.shape[1] < ACTION_EXEC_STEPS
        or value.shape[2] != NATIVE_EE_DIM
        or not np.isfinite(value).all()
    ):
        raise PostformalCandidatePoolError("proposal roster is invalid")
    effects = np.stack(
        [collector.canonical_action_chunk(current, row)[:ACTION_EXEC_STEPS] for row in value]
    ).astype(np.float32)
    embeddings = effects.reshape(len(effects), -1).astype(np.float64)
    if embeddings.shape != (len(value), ACTION_EXEC_STEPS * collector.CANONICAL_ACTION_DIM):
        raise PostformalCandidatePoolError("canonical effect embedding shape changed")
    if not np.isfinite(embeddings).all():
        raise PostformalCandidatePoolError("canonical effect embedding is non-finite")
    return effects, embeddings


def source_train_normalized_effect_embeddings(
    effects: np.ndarray,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalization_clip: float,
) -> np.ndarray:
    """Replay the shared head's source-train-only action input geometry."""

    value = np.asarray(effects, dtype=np.float32)
    mean = np.asarray(action_mean, dtype=np.float32)
    std = np.asarray(action_std, dtype=np.float32)
    if (
        value.ndim != 3
        or value.shape[0] < 2
        or value.shape[1] != ACTION_EXEC_STEPS
        or value.shape[2] != collector.CANONICAL_ACTION_DIM
        or not np.isfinite(value).all()
        or mean.shape != (collector.CANONICAL_ACTION_DIM,)
        or std.shape != (collector.CANONICAL_ACTION_DIM,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std < 1e-4)
        or not math.isfinite(float(normalization_clip))
        or float(normalization_clip) <= 0.0
    ):
        raise PostformalCandidatePoolError(
            "source-normalized diversity inputs are invalid"
        )
    normalized = np.clip(
        (value - mean[None, None, :]) / std[None, None, :],
        -float(normalization_clip),
        float(normalization_clip),
    )
    embeddings = normalized.reshape(len(value), -1).astype(np.float64)
    if not np.isfinite(embeddings).all():
        raise PostformalCandidatePoolError(
            "source-normalized canonical effect embedding is non-finite"
        )
    return embeddings


def source_train_normalized_translation_effect_embeddings(
    effects: np.ndarray,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    normalization_clip: float,
) -> np.ndarray:
    """Return source-only normalized executed-prefix dual-arm XYZ geometry."""

    full = source_train_normalized_effect_embeddings(
        effects,
        action_mean=action_mean,
        action_std=action_std,
        normalization_clip=normalization_clip,
    ).reshape(len(effects), ACTION_EXEC_STEPS, collector.CANONICAL_ACTION_DIM)
    translation = full[:, :, CANONICAL_TRANSLATION_EFFECT_CHANNELS]
    embeddings = translation.reshape(len(effects), -1).astype(np.float64)
    if embeddings.shape != (
        len(effects),
        ACTION_EXEC_STEPS * len(CANONICAL_TRANSLATION_EFFECT_CHANNELS),
    ) or not np.isfinite(embeddings).all():
        raise PostformalCandidatePoolError(
            "source-normalized translation effect embedding is invalid"
        )
    return embeddings


def greedy_farthest_point_indices(
    embeddings: np.ndarray, *, retain_count: int
) -> list[int]:
    """Select a deterministic label-blind subset anchored at raw proposal zero."""

    values = np.asarray(embeddings, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] == 0
        or not np.isfinite(values).all()
        or isinstance(retain_count, bool)
        or not 1 < int(retain_count) <= len(values)
    ):
        raise PostformalCandidatePoolError("farthest-point input is invalid")
    retain_count = int(retain_count)
    selected = [0]
    remaining = list(range(1, len(values)))
    # RMS has the same ordering as Euclidean distance at a fixed dimension and
    # has a directly auditable canonical-effect scale.
    while len(selected) < retain_count:
        minimum_rms = []
        for raw_index in remaining:
            delta = values[selected] - values[raw_index]
            distance = np.sqrt(np.mean(np.square(delta), axis=1))
            minimum_rms.append(float(distance.min()))
        # remaining is ascending, so NumPy's first-max rule freezes tie breaks.
        position = int(np.argmax(np.asarray(minimum_rms, dtype=np.float64)))
        selected.append(remaining.pop(position))
    if selected[0] != 0 or len(selected) != len(set(selected)):
        raise PostformalCandidatePoolError("farthest-point selection changed identity")
    return selected


def _pairwise_rms(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    rows = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            rows.append(float(np.sqrt(np.mean(np.square(values[left] - values[right])))))
    return np.asarray(rows, dtype=np.float64)


def pool_selection_audit(
    *,
    current_ee: np.ndarray,
    raw_proposals: np.ndarray,
    candidate_count: int,
    raw_proposal_count: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the final pool and an outcome-free reproducibility record."""

    candidate_count = validate_candidate_count(candidate_count)
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    raw = validate_candidates(raw_proposals, expected_count=raw_count, label="raw proposals")
    effects, embeddings = canonical_effect_embeddings(current_ee, raw)
    if raw_count == candidate_count:
        selected_indices = list(range(candidate_count))
        selection_algorithm = "identity_keep_original_actor_proposal_order_no_subset_selection"
    else:
        selected_indices = greedy_farthest_point_indices(
            embeddings, retain_count=candidate_count
        )
        selection_algorithm = (
            "greedy_maximize_minimum_rms_in_flattened_executed_prefix_"
            "canonical_action_effect14_anchor_raw_zero_tie_lowest_raw_index"
        )
    final = raw[np.asarray(selected_indices, dtype=np.int64)].copy()
    if not np.array_equal(final[0], raw[0]):
        raise PostformalCandidatePoolError("legacy candidate zero was not retained at index zero")
    raw_distances = _pairwise_rms(embeddings)
    final_distances = _pairwise_rms(embeddings[selected_indices])
    audit_base = {
        "format": POOL_AUDIT_FORMAT,
        "raw_proposal_count": raw_count,
        "retained_candidate_count": candidate_count,
        "raw_proposal_shape": list(raw.shape),
        "raw_ordered_proposals_sha256": array_sha256(raw),
        "canonical_executed_prefix_effects_sha256": array_sha256(effects),
        "executed_effect_horizon_actions": ACTION_EXEC_STEPS,
        "embedding_shape": list(embeddings.shape),
        "selected_raw_proposal_indices": selected_indices,
        "raw_candidate_instruction_conditions": [
            "runtime_frozen_instruction"
            if index < raw_count // 2
            else f"actor_training_seen_task_index_{TRAINING_SEEN_INSTRUCTION_TASK_INDEX}"
            for index in range(raw_count)
        ],
        "retained_candidate_instruction_conditions": [
            (
                "runtime_frozen_instruction"
                if index < raw_count // 2
                else f"actor_training_seen_task_index_{TRAINING_SEEN_INSTRUCTION_TASK_INDEX}"
            )
            for index in selected_indices
        ],
        "raw_candidate_flow_noise_indices": flow_noise_index_roster(raw_count),
        "raw_candidate_flow_noise_constructions": (
            flow_noise_construction_roster(raw_count)
        ),
        "retained_candidate_flow_noise_indices": [
            flow_noise_index_roster(raw_count)[index] for index in selected_indices
        ],
        "retained_candidate_flow_noise_constructions": [
            flow_noise_construction_roster(raw_count)[index]
            for index in selected_indices
        ],
        "legacy_raw_proposal_zero_is_final_candidate_zero": True,
        "subset_selection_applied": raw_count > candidate_count,
        "selection_algorithm": selection_algorithm,
        "selection_reads_outcomes_events_or_critic_scores": False,
        "raw_pairwise_effect_rms": {
            "minimum": float(raw_distances.min()),
            "median": float(np.median(raw_distances)),
            "maximum": float(raw_distances.max()),
        },
        "retained_pairwise_effect_rms": {
            "minimum": float(final_distances.min()),
            "median": float(np.median(final_distances)),
            "maximum": float(final_distances.max()),
        },
        "retained_ordered_candidates_sha256": array_sha256(final),
    }
    return final, {**audit_base, "audit_sha256": canonical_sha256(audit_base)}


def generate_candidate_pool(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: Any,
    instruction: str,
    scene_seed: int,
    query_index: int,
    candidate_count: int,
    raw_proposal_count: int | None,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    raw = generate_postformal_flow_candidates(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task=task,
        instruction=instruction,
        scene_seed=scene_seed,
        query_index=query_index,
        candidate_count=raw_count,
        device=device,
    )
    return pool_selection_audit(
        current_ee=collector.current_ee_action16(task),
        raw_proposals=raw,
        candidate_count=candidate_count,
        raw_proposal_count=raw_count,
    )


def postformal_noise_contract(
    candidate_count: int, raw_proposal_count: int | None = None
) -> dict[str, Any]:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    indices = list(range(raw_count))
    noise_indices = flow_noise_index_roster(raw_count)
    constructions = flow_noise_construction_roster(raw_count)
    conditional_raw16 = raw_count == RAW16_PROPOSAL_COUNT
    return {
        "format": FLOW_NOISE_CONTRACT_FORMAT,
        "distribution": (
            (
                "four_full_iid_plus_first5_xyz_and_fullchunk_xyz_conditional_"
                "antithetic_pairs_per_instruction_each_candidate_marginal_N_0_I"
            )
            if conditional_raw16
            else (
                "deterministic_independent_standard_normal_flow_draws_paired_across_"
                "two_semantically_equivalent_instruction_conditions_each_marginal_N_0_I"
            )
        ),
        "raw_proposal_indices": indices,
        "raw_proposal_flow_noise_indices": noise_indices,
        "raw_proposal_noise_constructions": constructions,
        "independent_flow_noise_draws": len(set(noise_indices)),
        "same_flow_noise_language_condition_pairs": [
            [index, index + raw_count // 2] for index in range(raw_count // 2)
        ],
        "seed_formula": (
            "(20260903 + scene_seed*1000003 + query_index*10007 + "
            "flow_noise_index*101) mod (2**63-1)"
        ),
        "candidate_zero_legacy_noise_unchanged": True,
        "candidate_zero_uses_formal_collector_make_noise": True,
        "conditional_translation_resampling_enabled": conditional_raw16,
        "conditional_translation_native_xyz_channels": (
            list(NATIVE_XYZ_NOISE_CHANNELS) if conditional_raw16 else []
        ),
        "conditional_translation_prefix_steps": (
            CONDITIONAL_FIRST_FIVE_STEPS if conditional_raw16 else 0
        ),
        "conditional_translation_fullchunk_uses_config_chunk_size": (
            conditional_raw16
        ),
        "raw16_per_instruction_mode_counts": (
            {
                "full_iid": 4,
                "conditional_first5_xyz": 2,
                "conditional_fullchunk_xyz": 2,
            }
            if conditional_raw16
            else None
        ),
        "candidate_marginal_distribution_proof": (
            "disjoint_coordinates_select_independent_standard_normal_z0_or_"
            "signed_znew_so_each_complete_tensor_is_exactly_N_0_I"
            if conditional_raw16
            else "each_complete_tensor_is_one_independent_standard_normal_draw"
        ),
        "candidate_joint_distribution_claimed_iid": False,
        "same_seed_query_and_observation_reproduce_ordered_noise": True,
        "actor_call_count_equals_raw_proposal_count": True,
        "frozen_actor_weights_changed": False,
        "formal_n4_antithetic_contract_changed": False,
        "selection_reads_outcomes_events_or_critic_scores": False,
        "collector_make_noise_sha256": sha256_file(Path(inspect.getsourcefile(collector) or "")),
    }


def candidate_pool_contract(
    candidate_count: int, raw_proposal_count: int | None = None
) -> dict[str, Any]:
    candidate_count = validate_candidate_count(candidate_count)
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    return {
        "candidate_count": candidate_count,
        "raw_proposal_count": raw_count,
        "default_executable_budget": "raw_proposal_count_equals_candidate_count",
        "subset_selection_applied": raw_count > candidate_count,
        "flow_noise_contract": postformal_noise_contract(candidate_count, raw_count),
        "instruction_coverage_contract": {
            "design": (
                "two_instruction_conditions_crossed_with_identical_conditional_"
                "translation_and_full_iid_flow_noise_roster"
                if raw_count == RAW16_PROPOSAL_COUNT
                else "two_instruction_conditions_crossed_with_identical_flow_noise_draws"
            ),
            "runtime_instruction_condition": "execution_contract_instruction",
            "actor_training_seen_condition": TRAINING_SEEN_INSTRUCTION,
            "actor_training_seen_condition_utf8_sha256": (
                TRAINING_SEEN_INSTRUCTION_SHA256
            ),
            "actor_training_seen_task_index": (
                TRAINING_SEEN_INSTRUCTION_TASK_INDEX
            ),
            "frozen_actor_training_tasks_parquet_sha256": (
                TRAINING_TASKS_PARQUET_SHA256
            ),
            "actor_training_seen_source": (
                "frozen_actor_training_lerobot_tasks_parquet_index_value_"
                "selected_for_appearance_agnostic_semantic_equivalence_without_"
                "rollout_outcomes"
            ),
            "raw_candidate_instruction_conditions": [
                "runtime_frozen_instruction"
                if index < raw_count // 2
                else f"actor_training_seen_task_index_{TRAINING_SEEN_INSTRUCTION_TASK_INDEX}"
                for index in range(raw_count)
            ],
            "semantic_task_changed": False,
            "reads_outcomes_events_or_critic_scores": False,
        },
        "candidate_zero": "raw_proposal_zero_retained_as_final_candidate_zero",
        "effect_schema": collector.ACTION_SCHEMA,
        "effect_horizon_actions": ACTION_EXEC_STEPS,
        "canonical_translation_effect_channels": list(
            CANONICAL_TRANSLATION_EFFECT_CHANNELS
        ),
        "selection": (
            "identity_original_actor_order"
            if raw_count == candidate_count
            else "deterministic_greedy_farthest_point_in_flattened_canonical_"
            "effect_space_anchor_zero_tie_lowest_raw_index"
        ),
        "selection_reads_outcomes_events_or_critic_scores": False,
        "additional_confidence_or_authorization_gate": False,
        "actor_call_budget_increased_beyond_raw_proposal_count": False,
        "critic_checkpoint_or_weights_changed": False,
    }


def aggregate_risk_adjusted_rank_scores(member_scores: torch.Tensor) -> torch.Tensor:
    """Extend the frozen five-member aggregation only along candidate axis."""

    if not isinstance(member_scores, torch.Tensor):
        raise TypeError("member rank scores must be a torch.Tensor")
    if member_scores.ndim != 2 or member_scores.shape[0] != 5:
        raise PostformalCandidatePoolError("member rank scores must be [5,N]")
    candidate_count = int(member_scores.shape[1])
    validate_candidate_count(candidate_count)
    if not bool(torch.isfinite(member_scores).all()):
        raise PostformalCandidatePoolError("member rank scores contain non-finite values")
    risk_weight = float(shared_head.EPISTEMIC_RANK_RISK_WEIGHT)
    return member_scores.mean(dim=0) - risk_weight * member_scores.std(
        dim=0, correction=0
    )


def runtime_rank_ensemble_contract(candidate_count: int) -> dict[str, Any]:
    """Declare the frozen N=4 training axis and postformal runtime axis once."""

    candidate_count = validate_candidate_count(candidate_count)
    frozen = shared_head.risk_adjusted_rank_ensemble_contract()
    training_candidate_count = frozen.pop("candidate_count", None)
    if training_candidate_count != formal.CANDIDATE_COUNT:
        raise PostformalCandidatePoolError(
            "shared-head training candidate count changed"
        )
    return {
        **frozen,
        "training_candidate_count": int(training_candidate_count),
        "runtime_candidate_count": candidate_count,
        "formula_and_five_member_axis_unchanged": True,
        "candidate_axis_extension_only": True,
    }


def select_candidate(scores: Sequence[float], *, candidate_count: int) -> int:
    candidate_count = validate_candidate_count(candidate_count)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (candidate_count,) or not np.isfinite(values).all():
        raise PostformalCandidatePoolError(
            f"candidate scores must be finite length {candidate_count}"
        )
    return int(np.argmax(values))


def scoring_batch(
    *,
    state: np.ndarray,
    current_ee: np.ndarray,
    candidates: np.ndarray,
    current_event: int,
    event_age_seconds: float,
    remaining_action_budget: int,
    candidate_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    candidate_count = validate_candidate_count(candidate_count)
    values = validate_candidates(
        candidates, expected_count=candidate_count, label="retained candidates"
    )
    if np.asarray(state).shape != (collector.STATE_DIM,):
        raise PostformalCandidatePoolError("shared critic state must be 27-D")
    if not 0 <= current_event <= STAGE_DENOMINATOR:
        raise PostformalCandidatePoolError("current event id is outside 0..4")
    if not np.isfinite(event_age_seconds) or event_age_seconds < 0.0:
        raise PostformalCandidatePoolError("event age must be finite and non-negative")
    if isinstance(remaining_action_budget, bool) or remaining_action_budget <= 0:
        raise PostformalCandidatePoolError("remaining action budget must be positive")
    event_onehot = np.zeros(STAGE_DENOMINATOR + 1, dtype=np.float32)
    event_onehot[current_event] = 1.0
    if not np.array_equal(np.asarray(state[18:23], dtype=np.float32), event_onehot):
        raise PostformalCandidatePoolError("state event onehot disagrees with event id")
    effects = np.stack(
        [collector.canonical_action_chunk(current_ee, row) for row in values]
    ).astype(np.float32)
    horizon = effects.shape[1]
    mask = np.arange(horizon)[None] < ACTION_EXEC_STEPS
    return {
        "state": torch.as_tensor(
            np.repeat(np.asarray(state, dtype=np.float32)[None], candidate_count, axis=0),
            device=device,
        ),
        "actions": torch.as_tensor(effects, device=device),
        "action_mask": torch.as_tensor(np.repeat(mask, candidate_count, axis=0), device=device),
        "action_available": torch.ones(candidate_count, dtype=torch.bool, device=device),
        "action_schema_id": torch.zeros(candidate_count, dtype=torch.long, device=device),
        "body_id": torch.zeros(candidate_count, dtype=torch.long, device=device),
        "dt": torch.full(
            (candidate_count,), PLANNED_DT_SECONDS, dtype=torch.float32, device=device
        ),
        "current_event_id": torch.full(
            (candidate_count,), current_event, dtype=torch.long, device=device
        ),
        "event_age_seconds": torch.full(
            (candidate_count,), float(event_age_seconds), dtype=torch.float32, device=device
        ),
        "remaining_action_budget": torch.full(
            (candidate_count,),
            float(remaining_action_budget),
            dtype=torch.float32,
            device=device,
        ),
    }


@torch.no_grad()
def score_candidates(
    models: Sequence[shared_head.EffectAlignedSharedEventHead],
    batch: Mapping[str, torch.Tensor],
    *,
    candidate_count: int,
) -> dict[str, Any]:
    candidate_count = validate_candidate_count(candidate_count)
    if len(models) != 5:
        raise PostformalCandidatePoolError("candidate scoring requires five LOBO members")
    names = (
        "candidate_rank_logit",
        "success_logit",
        "post_event_logits",
        "next_event_logits",
        "duration_selected_log_mean",
        "duration_selected_log_scale",
        "terminal_event_logits",
        "terminal_goal_progress_mean",
        "terminal_goal_progress_log_scale",
        "regression_probability",
        "joint_recovery_probability",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for model in models:
        output = model(batch)
        for name in names:
            value = output[name]
            if name == "success_logit":
                value = torch.sigmoid(value)
            elif name in {"post_event_logits", "next_event_logits", "terminal_event_logits"}:
                value = torch.softmax(value, -1)
            collected[name].append(value.detach().cpu().numpy())
    arrays = {name: np.stack(rows) for name, rows in collected.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise PostformalCandidatePoolError("LOBO ensemble produced a non-finite score")
    ranks = arrays["candidate_rank_logit"]
    if ranks.shape != (5, candidate_count):
        raise PostformalCandidatePoolError("LOBO rank score shape changed")
    risk_adjusted = aggregate_risk_adjusted_rank_scores(
        torch.as_tensor(ranks)
    ).cpu().numpy()
    selected = select_candidate(risk_adjusted.tolist(), candidate_count=candidate_count)
    return {
        "selected_candidate_index": selected,
        "candidate_rank_score_epistemic_lcb_ensemble": risk_adjusted.astype(float).tolist(),
        "candidate_rank_score_mean": ranks.mean(axis=0).astype(float).tolist(),
        "candidate_rank_score_raw_candidate_population_std": ranks.std(
            axis=0, ddof=0
        ).astype(float).tolist(),
        "candidate_rank_score_raw_member_candidate_mean": ranks.mean(
            axis=1
        ).astype(float).tolist(),
        "candidate_rank_score_raw_member_candidate_population_std": ranks.std(
            axis=1, ddof=0
        ).astype(float).tolist(),
        "candidate_rank_score_members": ranks.astype(float).tolist(),
        "candidate_success_probability_mean": arrays["success_logit"].mean(
            axis=0
        ).astype(float).tolist(),
        "candidate_post_event_probability_mean": arrays["post_event_logits"].mean(
            axis=0
        ).astype(float).tolist(),
        "candidate_next_event_probability_mean": arrays["next_event_logits"].mean(
            axis=0
        ).astype(float).tolist(),
        "candidate_duration_log_mean_members": arrays[
            "duration_selected_log_mean"
        ].astype(float).tolist(),
        "candidate_duration_log_scale_members": arrays[
            "duration_selected_log_scale"
        ].astype(float).tolist(),
        "candidate_terminal_event_probability_mean": arrays[
            "terminal_event_logits"
        ].mean(axis=0).astype(float).tolist(),
        "candidate_terminal_goal_progress_mean_members": arrays[
            "terminal_goal_progress_mean"
        ].astype(float).tolist(),
        "candidate_terminal_goal_progress_log_scale_members": arrays[
            "terminal_goal_progress_log_scale"
        ].astype(float).tolist(),
        "candidate_regression_probability_mean": arrays[
            "regression_probability"
        ].mean(axis=0).astype(float).tolist(),
        "candidate_joint_recovery_probability_mean": arrays[
            "joint_recovery_probability"
        ].mean(axis=0).astype(float).tolist(),
        "aggregation": {
            "formula": "member_mean_minus_frozen_risk_weight_times_population_std",
            "risk_weight": float(shared_head.EPISTEMIC_RANK_RISK_WEIGHT),
            "candidate_axis_extension_only": True,
        },
    }


def evaluation_schedule(candidate_count: int) -> list[dict[str, Any]]:
    roster = list(methods(candidate_count))
    rows = []
    for body in BODIES:
        for condition in CONDITIONS:
            for ordinal in range(SEED_COUNT):
                rows.append(
                    {
                        "heldout_body": body,
                        "condition": condition,
                        "requested_seed": SEED_BASE + ordinal,
                        "method_order": roster if ordinal % 2 == 0 else list(reversed(roster)),
                    }
                )
    if len(rows) != len(BODIES) * len(CONDITIONS) * SEED_COUNT:
        raise PostformalCandidatePoolError("postformal schedule cardinality changed")
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
    calibration: Mapping[str, Any],
    instruction: str,
    candidate_count: int,
    raw_proposal_count: int,
    device: torch.device,
) -> dict[str, Any]:
    del calibration
    candidate_count = validate_candidate_count(candidate_count)
    task = collector._new_task(task_class, task_args, seed, instruction)
    try:
        names, objects = collector.discover_pose_objects(
            task, set(analytic_event.REQUIRED_OBJECTS)
        )
        reset_snapshot = formal.capture_reset_snapshot(task, names, objects)
        task.scene.step()
        canonical_snapshot = formal.capture_reset_snapshot(task, names, objects)
        candidates, pool_audit = generate_candidate_pool(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            instruction=instruction,
            scene_seed=seed,
            query_index=0,
            candidate_count=candidate_count,
            raw_proposal_count=raw_proposal_count,
            device=device,
        )
        after = formal.capture_reset_snapshot(task, names, objects)
        if canonical_snapshot != after:
            raise PostformalCandidatePoolError(
                "initial candidate-pool generation changed simulator state"
            )
        base = {
            "format": "etsf_robotwin2_postformal_initial_pool_commitment_v1",
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "resolved_seed": seed,
            "action_exec_steps": ACTION_EXEC_STEPS,
            "planned_dt_seconds": PLANNED_DT_SECONDS,
            "candidate_count": candidate_count,
            "raw_proposal_count": proposal_count(candidate_count, raw_proposal_count),
            "candidate_horizon": int(candidates.shape[1]),
            "candidate_shape": list(candidates.shape),
            "ordered_candidate_set_sha256": array_sha256(candidates),
            "candidate_pool_audit": pool_audit,
            "reset_snapshot": reset_snapshot,
            "reset_identity_sha256": formal.reset_identity(reset_snapshot),
            "canonical_query_snapshot": canonical_snapshot,
            "canonical_query_identity_sha256": formal.reset_identity(canonical_snapshot),
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
    pool_audit: Mapping[str, Any],
    candidate_count: int,
    raw_proposal_count: int,
) -> None:
    candidate_count = validate_candidate_count(candidate_count)
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    if (
        commitment.get("format") != "etsf_robotwin2_postformal_initial_pool_commitment_v1"
        or commitment.get("heldout_body") != body
        or commitment.get("condition") != condition
        or commitment.get("requested_seed") != seed
        or commitment.get("resolved_seed") != seed
        or commitment.get("action_exec_steps") != ACTION_EXEC_STEPS
        or commitment.get("planned_dt_seconds") != PLANNED_DT_SECONDS
        or commitment.get("candidate_count") != candidate_count
        or commitment.get("raw_proposal_count") != raw_count
        or commitment.get("candidate_horizon") != int(candidates.shape[1])
        or commitment.get("candidate_shape") != list(candidates.shape)
        or commitment.get("ordered_candidate_set_sha256") != array_sha256(candidates)
        or commitment.get("candidate_pool_audit") != dict(pool_audit)
        or commitment.get("reset_snapshot") != reset_snapshot
        or commitment.get("reset_identity_sha256") != formal.reset_identity(reset_snapshot)
        or commitment.get("canonical_query_snapshot") != canonical_query_snapshot
        or commitment.get("canonical_query_identity_sha256")
        != formal.reset_identity(canonical_query_snapshot)
        or commitment.get("query_canonicalization_steps") != QUERY_CANONICALIZATION_STEPS
        or commitment.get("candidate_generation_advanced_simulator") is not False
        or commitment.get("commitment_sha256") != canonical_sha256(_commitment_base(commitment))
    ):
        raise PostformalCandidatePoolError(
            "paired method initial reset/proposals/pool differ from commitment"
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
    candidate_count: int,
    raw_proposal_count: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    candidate_count = validate_candidate_count(candidate_count)
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    if method not in methods(candidate_count):
        raise PostformalCandidatePoolError(f"unknown postformal method {method!r}")
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
        initial_snapshot = formal.capture_reset_snapshot(task, names, objects)
        trajectory = [initial_poses]
        sim_times = [collector._sim_time(task)]
        initial_identity = formal.reset_identity(initial_snapshot)
        initial_canonical_snapshot: Mapping[str, Any] | None = None
        query_index = 0
        while not collector._episode_done(task, max_steps):
            task.scene.step()
            collector._append_physical_observation(task, objects, trajectory, sim_times)
            current_ee = collector.current_ee_action16(task)
            pre_pool_snapshot = formal.capture_reset_snapshot(task, names, objects)
            candidates, pool_audit = generate_candidate_pool(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                instruction=instruction,
                scene_seed=seed,
                query_index=query_index,
                candidate_count=candidate_count,
                raw_proposal_count=raw_count,
                device=device,
            )
            after_pool_snapshot = formal.capture_reset_snapshot(task, names, objects)
            if after_pool_snapshot != pre_pool_snapshot:
                raise PostformalCandidatePoolError(
                    "candidate-pool generation changed observable simulator state"
                )
            candidate_sha = array_sha256(candidates)
            if candidate_sha != pool_audit.get("retained_ordered_candidates_sha256"):
                raise PostformalCandidatePoolError("candidate-pool audit hash changed")
            if query_index == 0:
                initial_canonical_snapshot = pre_pool_snapshot
                verify_initial_commitment(
                    initial_commitment,
                    body=body,
                    condition=condition,
                    seed=seed,
                    reset_snapshot=initial_snapshot,
                    canonical_query_snapshot=pre_pool_snapshot,
                    candidates=candidates,
                    pool_audit=pool_audit,
                    candidate_count=candidate_count,
                    raw_proposal_count=raw_count,
                )
            current_event_age: float | None = None
            if method == "actor_baseline":
                selected = 0
                score_record = None
            else:
                trajectory_array = np.stack(trajectory).astype(np.float64)
                state, current_event, current_event_age = formal.canonical_state_at(
                    trajectory=trajectory_array,
                    sim_times=np.asarray(sim_times, dtype=np.float64),
                    names=names,
                    ee_action=current_ee,
                    calibration=calibration,
                    success_height_reference_z=collector.success_height_reference_z(
                        task
                    ),
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
                        candidate_count=candidate_count,
                        device=device,
                    ),
                    candidate_count=candidate_count,
                )
                selected = int(score_record["selected_candidate_index"])
            executed = 0
            chunk_start_seconds = collector._sim_time(task)
            for action in candidates[selected, :ACTION_EXEC_STEPS]:
                if collector._episode_done(task, max_steps):
                    break
                task.take_action(action, action_type="ee")
                executed += 1
                collector._append_physical_observation(task, objects, trajectory, sim_times)
            decisions.append(
                {
                    "query_index": query_index,
                    "candidate_set_sha256": candidate_sha,
                    "candidate_count": candidate_count,
                    "raw_proposal_count": raw_count,
                    "candidate_pool_audit": pool_audit,
                    "selected_candidate_index": selected,
                    "selected_raw_proposal_index": pool_audit[
                        "selected_raw_proposal_indices"
                    ][selected],
                    "executed_action_count": executed,
                    "planned_chunk_seconds": PLANNED_DT_SECONDS,
                    "physical_sim_seconds": collector._sim_time(task) - chunk_start_seconds,
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
        trajectory_array = np.stack(trajectory).astype(np.float64)
        _predicates, events = collector.derive_predicates_and_events(
            trajectory_array,
            np.asarray(sim_times, dtype=np.float64),
            names,
            success,
            calibration,
            collector.success_height_reference_z(task),
        )
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


def validate_rollout(
    rollout: Mapping[str, Any],
    *,
    method: str,
    expected: Mapping[str, Any],
    candidate_count: int,
    raw_proposal_count: int,
) -> None:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
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
        raise PostformalCandidatePoolError(f"{method} rollout identity/outcome changed")
    expected_progress = (
        1.0
        if rollout["binary_success"] == 1
        else rollout["max_event_id"] / float(STAGE_DENOMINATOR)
    )
    if abs(float(rollout.get("stage_progress", -1.0)) - expected_progress) > 1e-9:
        raise PostformalCandidatePoolError(f"{method} stage progress changed")
    decisions = rollout.get("decisions")
    if (
        not isinstance(decisions, list)
        or not decisions
        or rollout.get("policy_query_count") != len(decisions)
    ):
        raise PostformalCandidatePoolError(f"{method} decision roster changed")
    for query_index, decision in enumerate(decisions):
        audit = decision.get("candidate_pool_audit")
        if (
            not isinstance(decision, Mapping)
            or decision.get("query_index") != query_index
            or decision.get("candidate_count") != candidate_count
            or decision.get("raw_proposal_count") != raw_count
            or not isinstance(audit, Mapping)
            or audit.get("retained_candidate_count") != candidate_count
            or audit.get("raw_proposal_count") != raw_count
            or audit.get("selection_reads_outcomes_events_or_critic_scores") is not False
            or audit.get("selected_raw_proposal_indices", [None])[0] != 0
            or audit.get("retained_ordered_candidates_sha256")
            != decision.get("candidate_set_sha256")
            or audit.get("audit_sha256")
            != canonical_sha256(
                {key: item for key, item in audit.items() if key != "audit_sha256"}
            )
            or type(decision.get("selected_candidate_index")) is not int
            or not 0 <= decision["selected_candidate_index"] < candidate_count
            or decision.get("selected_raw_proposal_index")
            != audit["selected_raw_proposal_indices"][decision["selected_candidate_index"]]
        ):
            raise PostformalCandidatePoolError(f"{method} candidate-pool record changed")
        scores = decision.get("critic_scores")
        if method == "actor_baseline":
            if (
                decision["selected_candidate_index"] != 0
                or decision["selected_raw_proposal_index"] != 0
                or scores is not None
                or decision.get("event_age_seconds") is not None
            ):
                raise PostformalCandidatePoolError(
                    "baseline must execute legacy raw proposal zero without critic"
                )
            continue
        if not isinstance(scores, Mapping):
            raise PostformalCandidatePoolError("ETSF rollout lacks critic scores")
        raw = np.asarray(scores.get("candidate_rank_score_members"), dtype=np.float64)
        recorded = np.asarray(
            scores.get("candidate_rank_score_epistemic_lcb_ensemble"), dtype=np.float64
        )
        if raw.shape != (5, candidate_count) or recorded.shape != (candidate_count,):
            raise PostformalCandidatePoolError("ETSF critic score shape changed")
        recomputed = aggregate_risk_adjusted_rank_scores(torch.as_tensor(raw)).numpy()
        if not np.allclose(recorded, recomputed, atol=1e-6, rtol=0.0):
            raise PostformalCandidatePoolError("ETSF risk-adjusted score cannot be replayed")
        selected = select_candidate(recomputed.tolist(), candidate_count=candidate_count)
        if (
            scores.get("selected_candidate_index") != selected
            or decision["selected_candidate_index"] != selected
        ):
            raise PostformalCandidatePoolError("ETSF selected candidate changed")


def materialize_pair(
    expected: Mapping[str, Any],
    rollouts: Mapping[str, Mapping[str, Any]],
    *,
    commitment: Mapping[str, Any],
    attempt_sha256: str,
    execution_contract_logical_sha256: str,
    candidate_count: int,
    raw_proposal_count: int,
) -> dict[str, Any]:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    etsf_method = method_name(candidate_count)
    if set(rollouts) != set(methods(candidate_count)):
        raise PostformalCandidatePoolError("pair must contain baseline and postformal ETSF")
    baseline = rollouts["actor_baseline"]
    etsf = rollouts[etsf_method]
    commitment_sha = commitment.get("commitment_sha256")
    same_snapshot = (
        baseline.get("initial_reset_snapshot")
        == etsf.get("initial_reset_snapshot")
        == commitment.get("reset_snapshot")
    )
    same_canonical = (
        baseline.get("initial_canonical_query_snapshot")
        == etsf.get("initial_canonical_query_snapshot")
        == commitment.get("canonical_query_snapshot")
    )
    same_reset = bool(
        baseline.get("tracked_object_names") == etsf.get("tracked_object_names")
        and same_snapshot
        and same_canonical
        and baseline.get("initial_reset_identity_sha256")
        == etsf.get("initial_reset_identity_sha256")
        == commitment.get("reset_identity_sha256")
        and baseline.get("initial_candidate_commitment_sha256")
        == etsf.get("initial_candidate_commitment_sha256")
        == commitment_sha
    )
    if not same_reset:
        raise PostformalCandidatePoolError("paired methods did not use the same reset")
    same_initial_pool = bool(
        baseline["decisions"][0]["candidate_set_sha256"]
        == etsf["decisions"][0]["candidate_set_sha256"]
        == commitment.get("ordered_candidate_set_sha256")
        and baseline["decisions"][0]["candidate_pool_audit"]
        == etsf["decisions"][0]["candidate_pool_audit"]
        == commitment.get("candidate_pool_audit")
    )
    if not same_initial_pool:
        raise PostformalCandidatePoolError("paired initial raw proposals/retained pool changed")
    base = {
        "format": PAIR_FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        **dict(expected),
        "candidate_count": candidate_count,
        "raw_proposal_count": raw_count,
        "attempt_sha256": attempt_sha256,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "initial_candidate_commitment_sha256": commitment_sha,
        "same_resolved_reset": same_reset,
        "same_complete_observable_reset_snapshot": bool(same_snapshot),
        "same_canonical_query0_snapshot": bool(same_canonical),
        "same_initial_raw_proposals_and_retained_pool": same_initial_pool,
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


def validate_pair_record(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    execution_contract_logical_sha256: str,
    candidate_count: int,
    raw_proposal_count: int,
) -> None:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    if (
        value.get("format") != PAIR_FORMAT
        or value.get("benchmark") != BENCHMARK
        or value.get("task") != TASK
        or value.get("heldout_body") != expected["heldout_body"]
        or value.get("condition") != expected["condition"]
        or value.get("requested_seed") != expected["requested_seed"]
        or value.get("method_order") != expected["method_order"]
        or value.get("candidate_count") != candidate_count
        or value.get("raw_proposal_count") != raw_count
        or value.get("same_resolved_reset") is not True
        or value.get("same_complete_observable_reset_snapshot") is not True
        or value.get("same_canonical_query0_snapshot") is not True
        or value.get("same_initial_raw_proposals_and_retained_pool") is not True
        or value.get("execution_contract_logical_sha256")
        != execution_contract_logical_sha256
        or value.get("pair_sha256")
        != canonical_sha256({key: item for key, item in value.items() if key != "pair_sha256"})
    ):
        raise PostformalCandidatePoolError("pair record changed or is incomplete")
    rollouts = value.get("rollouts")
    if not isinstance(rollouts, Mapping) or set(rollouts) != set(methods(candidate_count)):
        raise PostformalCandidatePoolError("pair rollout roster changed")
    for method in methods(candidate_count):
        validate_rollout(
            rollouts[method],
            method=method,
            expected=expected,
            candidate_count=candidate_count,
            raw_proposal_count=raw_count,
        )
        if (
            rollouts[method].get("initial_candidate_commitment_sha256")
            != value.get("initial_candidate_commitment_sha256")
        ):
            raise PostformalCandidatePoolError("rollout commitment binding changed")


def outcome_row(
    pair: Mapping[str, Any], *, candidate_count: int, raw_proposal_count: int
) -> dict[str, Any]:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    etsf_method = method_name(candidate_count)
    baseline = pair["rollouts"]["actor_baseline"]
    etsf = pair["rollouts"][etsf_method]
    return {
        "benchmark": BENCHMARK,
        "task": TASK,
        "heldout_body": pair["heldout_body"],
        "condition": pair["condition"],
        "requested_seed": pair["requested_seed"],
        "method_order": pair["method_order"],
        "candidate_count": candidate_count,
        "raw_proposal_count": raw_count,
        "pair_sha256": pair["pair_sha256"],
        "actor_baseline_binary_success": baseline["binary_success"],
        "actor_baseline_stage_progress": baseline["stage_progress"],
        "etsf_binary_success": etsf["binary_success"],
        "etsf_stage_progress": etsf["stage_progress"],
    }


def build_outcome_document(
    rows: Sequence[Mapping[str, Any]],
    *,
    execution_contract_logical_sha256: str,
    execution_contract_file_sha256: str,
    candidate_count: int,
    raw_proposal_count: int,
) -> dict[str, Any]:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    normalized = [dict(row) for row in rows]
    base = {
        "format": OUTCOME_FORMAT,
        "status": "complete_full_heldout_paired_postformal_outcomes",
        "candidate_count": candidate_count,
        "raw_proposal_count": raw_count,
        "pair_count": len(normalized),
        "rows": normalized,
        "rows_sha256": canonical_sha256(normalized),
        "reference_preregistration_sha256": REFERENCE_PREREGISTRATION_SHA256,
        "postformal_not_part_of_reference_preregistration": True,
        "execution_contract_logical_sha256": execution_contract_logical_sha256,
        "execution_contract_file_sha256": execution_contract_file_sha256,
        "ordered_pair_sha256s_sha256": canonical_sha256(
            [str(row["pair_sha256"]) for row in normalized]
        ),
    }
    return {**base, "document_sha256": canonical_sha256(base)}


def _exact_mcnemar_two_sided(actor_only: int, etsf_only: int) -> float:
    discordant = actor_only + etsf_only
    if discordant == 0:
        return 1.0
    tail = min(actor_only, etsf_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    probability /= 2.0**discordant
    return min(1.0, 2.0 * probability)


def _summary_for_rows(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    baseline_success = np.asarray(
        [row["actor_baseline_binary_success"] for row in rows], dtype=np.float64
    )
    etsf_success = np.asarray([row["etsf_binary_success"] for row in rows], dtype=np.float64)
    baseline_stage = np.asarray(
        [row["actor_baseline_stage_progress"] for row in rows], dtype=np.float64
    )
    etsf_stage = np.asarray([row["etsf_stage_progress"] for row in rows], dtype=np.float64)
    success_delta = etsf_success - baseline_success
    stage_delta = etsf_stage - baseline_stage
    actor_only = int(np.sum(success_delta < 0))
    etsf_only = int(np.sum(success_delta > 0))
    generator = np.random.default_rng(seed)
    success_bootstrap = np.empty(REPORT_BOOTSTRAP_REPLICATES, dtype=np.float64)
    stage_bootstrap = np.empty(REPORT_BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Keep report materialization cheap even for all 1,000 full pairs.
    for start in range(0, REPORT_BOOTSTRAP_REPLICATES, 512):
        stop = min(start + 512, REPORT_BOOTSTRAP_REPLICATES)
        indices = generator.integers(
            0, len(rows), size=(stop - start, len(rows)), endpoint=False
        )
        success_bootstrap[start:stop] = success_delta[indices].mean(axis=1)
        stage_bootstrap[start:stop] = stage_delta[indices].mean(axis=1)
    return {
        "pair_count": len(rows),
        "actor_baseline_success_rate": float(baseline_success.mean()),
        "etsf_success_rate": float(etsf_success.mean()),
        "paired_success_rate_delta": float(success_delta.mean()),
        "paired_success_rate_delta_bootstrap_95_percentile_ci": [
            float(np.quantile(success_bootstrap, 0.025)),
            float(np.quantile(success_bootstrap, 0.975)),
        ],
        "actor_baseline_stage_progress_mean": float(baseline_stage.mean()),
        "etsf_stage_progress_mean": float(etsf_stage.mean()),
        "paired_stage_progress_delta": float(stage_delta.mean()),
        "paired_stage_progress_delta_bootstrap_95_percentile_ci": [
            float(np.quantile(stage_bootstrap, 0.025)),
            float(np.quantile(stage_bootstrap, 0.975)),
        ],
        "actor_only_success_pairs": actor_only,
        "etsf_only_success_pairs": etsf_only,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_two_sided(actor_only, etsf_only),
        "bootstrap_unit": "paired_body_condition_requested_seed_row",
        "bootstrap_replicates": REPORT_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome_document_sha256: str,
    candidate_count: int,
    raw_proposal_count: int,
) -> dict[str, Any]:
    raw_count = proposal_count(candidate_count, raw_proposal_count)
    normalized = [dict(row) for row in rows]
    by_body = {
        body: _summary_for_rows(
            [row for row in normalized if row["heldout_body"] == body],
            seed=REPORT_BOOTSTRAP_SEED + ordinal + 1,
        )
        for ordinal, body in enumerate(BODIES)
    }
    by_body_condition = {
        f"{body}|{condition}": _summary_for_rows(
            [
                row
                for row in normalized
                if row["heldout_body"] == body and row["condition"] == condition
            ],
            seed=REPORT_BOOTSTRAP_SEED + 100 + ordinal,
        )
        for ordinal, (body, condition) in enumerate(
            (body, condition) for body in BODIES for condition in CONDITIONS
        )
    }
    base = {
        "format": REPORT_FORMAT,
        "status": "complete_postformal_paired_candidate_pool_report",
        "candidate_count": candidate_count,
        "raw_proposal_count": raw_count,
        "outcome_document_sha256": outcome_document_sha256,
        "postformal_not_part_of_reference_preregistration": True,
        "candidate_pool_contract": candidate_pool_contract(candidate_count, raw_count),
        "overall": _summary_for_rows(normalized, seed=REPORT_BOOTSTRAP_SEED),
        "by_heldout_body": by_body,
        "by_heldout_body_and_condition": by_body_condition,
    }
    return {**base, "report_sha256": canonical_sha256(base)}


def implementation_binding(
    robotwin_root: Path, *, candidate_count: int, raw_proposal_count: int
) -> dict[str, Any]:
    inherited = formal.implementation_binding(robotwin_root)
    runner_path = Path(__file__).resolve()
    return {
        **inherited,
        "postformal_runner": {
            "path": str(runner_path),
            "sha256": sha256_file(runner_path),
            "size_bytes": runner_path.stat().st_size,
        },
        "postformal_candidate_pool_contract": candidate_pool_contract(
            candidate_count, raw_proposal_count
        ),
        "formal_collector_or_root_mutated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--reference-preregistration", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol-sha256", required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument(
        "--lobo-fold",
        action="append",
        required=True,
        help="Repeat exactly five times as heldout-body=/absolute/fold/root.",
    )
    parser.add_argument(
        "--required-supplement-binding-sha256",
        help=(
            "Require all five folds to use this exact expert-root supplement "
            "binding. The enhanced N=8 study should always pass this option."
        ),
    )
    parser.add_argument("--candidate-count", type=int, choices=SUPPORTED_CANDIDATE_COUNTS, default=8)
    parser.add_argument(
        "--proposal-count",
        type=int,
        default=None,
        help=(
            "Raw actor proposals per query. Defaults to candidate-count; an explicit "
            "larger even value up to 2*candidate-count enables blind FPS subsampling."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-exec-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fps", type=float, default=ACTOR_DATASET_FPS)
    parser.add_argument("--instruction", default=collector.DEFAULT_INSTRUCTION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        execution_protocol = formal.actor_execution.load_execution_protocol_file(
            args.actor_execution_protocol,
            args.actor_execution_protocol_sha256,
        )
    except formal.actor_execution.ActorExecutionProtocolError as error:
        raise PostformalCandidatePoolError(str(error)) from error
    configure_actor_execution_protocol(
        execution_protocol,
        path=args.actor_execution_protocol,
        file_sha256=args.actor_execution_protocol_sha256,
        path_root=args.path_root,
    )
    if args.action_exec_steps is None:
        args.action_exec_steps = ACTION_EXEC_STEPS
    if args.max_steps is None:
        args.max_steps = int(execution_protocol["max_steps"])
    candidate_count = validate_candidate_count(args.candidate_count)
    raw_proposal_count = proposal_count(candidate_count, args.proposal_count)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise PostformalCandidatePoolError("postformal execution requires remote RTX 4090")
    if (
        args.action_exec_steps != ACTION_EXEC_STEPS
        or args.max_steps != execution_protocol["max_steps"]
    ):
        raise PostformalCandidatePoolError(
            "action-exec-steps/max-steps differ from the bound protocol"
        )
    if args.fps != ACTOR_DATASET_FPS:
        raise PostformalCandidatePoolError("actor control interval must remain 15 Hz")
    if args.instruction != collector.DEFAULT_INSTRUCTION:
        raise PostformalCandidatePoolError("actor instruction changed")
    inputs = (
        args.actor_checkpoint,
        args.vlm_metadata_path,
        args.robotwin_root,
        args.event_spec,
        args.reference_preregistration,
        args.actor_execution_protocol,
        args.path_root,
    )
    if any(not path.expanduser().resolve().exists() for path in inputs):
        raise FileNotFoundError("one or more required static inputs are missing")

    random.seed(20260908)
    np.random.seed(20260908)
    torch.manual_seed(20260908)
    robotwin_root = args.robotwin_root.expanduser().resolve()
    os.environ["ASSETS_PATH"] = str(robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(robotwin_root))

    reference_receipt = formal.load_preregistration(
        args.reference_preregistration.expanduser().resolve()
    )
    fold_paths = formal.parse_fold_specs(args.lobo_fold)
    folds = {body: formal.inspect_fold(body, fold_paths[body]) for body in BODIES}
    fold_training_regime = formal.inspect_fold_training_regime(
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
        raise PostformalCandidatePoolError("event specification differs from training")
    try:
        _event_spec, calibration = analytic_event.load_event_spec(event_spec_path)
    except analytic_event.AnalyticEventSpecError as error:
        raise PostformalCandidatePoolError(str(error)) from error

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs_dir = output / "pairs"
    attempts_dir = output / "attempts"
    commitments_dir = output / "initial_commitments"
    failures_dir = output / "failures"
    for directory in (pairs_dir, attempts_dir, commitments_dir, failures_dir):
        directory.mkdir(exist_ok=True)
    outcome_path = output / "paired_outcomes.json"
    report_path = output / "paired_candidate_pool_report.json"
    contract_path = output / "execution_contract.json"
    schedule = evaluation_schedule(candidate_count)
    execution_protocol_binding = formal.actor_execution_protocol_binding()
    try:
        output.relative_to(Path(execution_protocol_binding["path_root"]))
    except ValueError as error:
        raise PostformalCandidatePoolError(
            "postformal output must be contained by the protocol path_root"
        ) from error
    contract_base = {
        "format": CONTRACT_FORMAT,
        "runner_format": FORMAT,
        "benchmark": BENCHMARK,
        "task": TASK,
        "bodies": list(BODIES),
        "conditions": list(CONDITIONS),
        "evaluation_seed_base": SEED_BASE,
        "evaluation_seed_count": SEED_COUNT,
        "pair_count": len(schedule),
        "rollout_count": len(schedule) * 2,
        "methods": list(methods(candidate_count)),
        "candidate_pool_contract": candidate_pool_contract(
            candidate_count, raw_proposal_count
        ),
        "baseline_selector": "legacy_raw_flow_noise_proposal_zero",
        "same_requested_seed_paired": True,
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
        "candidate_rank_ensemble_contract": runtime_rank_ensemble_contract(
            candidate_count
        ),
        "event_spec": str(event_spec_path),
        "event_spec_sha256": EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": sha256_file(
            Path(analytic_event.__file__).resolve()
        ),
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "reference_preregistration": str(args.reference_preregistration.resolve()),
        "reference_preregistration_sha256": reference_receipt["preregistration_sha256"],
        "path_root": execution_protocol_binding["path_root"],
        "actor_execution_protocol": execution_protocol_binding["protocol"],
        "actor_execution_protocol_binding": execution_protocol_binding,
        "actor_execution_protocol_file_sha256": execution_protocol_binding[
            "file_sha256"
        ],
        "postformal_not_part_of_reference_preregistration": True,
        "formal_best_of_four_result_must_be_reported_separately": True,
        "action_exec_steps": ACTION_EXEC_STEPS,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "planned_first_chunk_seconds": PLANNED_DT_SECONDS,
        "instruction": args.instruction,
        "runtime_binding": implementation_binding(
            robotwin_root,
            candidate_count=candidate_count,
            raw_proposal_count=raw_proposal_count,
        ),
        "no_training": True,
        "formal_collector_or_root_mutated": False,
        "official_expert_or_protected_internal_payloads_opened": False,
    }
    contract = {**contract_base, "logical_sha256": canonical_sha256(contract_base)}
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise PostformalCandidatePoolError("existing execution contract differs")
    else:
        atomic_json(contract_path, contract, frozen=True)
    contract_file_sha = sha256_file(contract_path)

    from envs import CONFIGS_PATH  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    module = __import__(f"envs.{TASK}", fromlist=[TASK])
    task_class = getattr(module, TASK)
    device = torch.device("cuda:0")
    actor_config = PreTrainedConfig.from_pretrained(actor_checkpoint, local_files_only=True)
    actor_config.device = str(device)
    actor_config.vlm_model_name = str(vlm_metadata)
    actor_config.load_vlm_weights = False
    if (
        actor_config.action_feature is None
        or int(actor_config.action_feature.shape[0]) != NATIVE_EE_DIM
        or actor_config.input_features.get("observation.state") is None
        or int(actor_config.input_features["observation.state"].shape[0]) != NATIVE_EE_DIM
    ):
        raise PostformalCandidatePoolError("frozen actor is not state16/action16 EE")
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
    for expected in schedule:
        body = str(expected["heldout_body"])
        identity = pair_id(body, expected["condition"], expected["requested_seed"])
        if body != active_body:
            del ensemble
            gc.collect()
            torch.cuda.empty_cache()
            ensemble = formal.load_ensemble(folds[body], device)
            active_body = body
        pair_path = pairs_dir / f"{identity}.json"
        if pair_path.exists():
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            validate_pair_record(
                pair,
                expected,
                execution_contract_logical_sha256=contract["logical_sha256"],
                candidate_count=candidate_count,
                raw_proposal_count=raw_proposal_count,
            )
        else:
            attempt_path = attempts_dir / f"{identity}.json"
            commitment_path = commitments_dir / f"{identity}.json"
            failure_path = failures_dir / f"{identity}.json"
            if attempt_path.exists() or commitment_path.exists() or failure_path.exists():
                raise PostformalCandidatePoolError(
                    "an incomplete/failed attempt exists; no silent automatic retry"
                )
            attempt_base = {
                "format": "etsf_robotwin2_postformal_pool_attempt_v1",
                "status": "started_once_no_automatic_retry",
                "pair_id": identity,
                **dict(expected),
                "candidate_count": candidate_count,
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
                    candidate_count=candidate_count,
                    raw_proposal_count=raw_proposal_count,
                    device=device,
                )
                atomic_json(commitment_path, commitment, frozen=True)
                rollouts = {}
                for selected_method in expected["method_order"]:
                    rollouts[selected_method] = execute_rollout(
                        method=selected_method,
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
                        candidate_count=candidate_count,
                        raw_proposal_count=raw_proposal_count,
                        max_steps=args.max_steps,
                        device=device,
                    )
                pair = materialize_pair(
                    expected,
                    rollouts,
                    commitment=commitment,
                    attempt_sha256=attempt_sha,
                    execution_contract_logical_sha256=contract["logical_sha256"],
                    candidate_count=candidate_count,
                    raw_proposal_count=raw_proposal_count,
                )
                validate_pair_record(
                    pair,
                    expected,
                    execution_contract_logical_sha256=contract["logical_sha256"],
                    candidate_count=candidate_count,
                    raw_proposal_count=raw_proposal_count,
                )
                atomic_json(pair_path, pair, frozen=True)
            except Exception as error:
                failure_base = {
                    "format": "etsf_robotwin2_postformal_pool_attempt_failure_v1",
                    "status": "failed_no_automatic_retry",
                    "pair_id": identity,
                    "attempt_sha256": attempt_sha,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
                if not failure_path.exists():
                    atomic_json(
                        failure_path,
                        {**failure_base, "failure_sha256": canonical_sha256(failure_base)},
                        frozen=True,
                    )
                raise
        rows.append(
            outcome_row(
                pair,
                candidate_count=candidate_count,
                raw_proposal_count=raw_proposal_count,
            )
        )
        completed += 1
        atomic_json(
            output / "progress.json",
            {
                "format": FORMAT,
                "status": "running" if completed < len(schedule) else "rollouts_complete",
                "candidate_count": candidate_count,
                "raw_proposal_count": raw_proposal_count,
                "completed_pairs": completed,
                "completed_rollouts": completed * 2,
                "total_pairs": len(schedule),
                "last_pair": identity,
                "wall_seconds": time.time() - started,
            },
        )
        print(
            "POSTFORMAL_PAIR_COMPLETE="
            + json.dumps(
                {
                    "completed": completed,
                    "total": len(schedule),
                    "candidate_count": candidate_count,
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
        execution_contract_file_sha256=contract_file_sha,
        candidate_count=candidate_count,
        raw_proposal_count=raw_proposal_count,
    )
    if outcome_path.exists():
        if json.loads(outcome_path.read_text(encoding="utf-8")) != document:
            raise PostformalCandidatePoolError("existing outcome document differs")
    else:
        atomic_json(outcome_path, document, frozen=True)
    report = build_report(
        rows,
        outcome_document_sha256=document["document_sha256"],
        candidate_count=candidate_count,
        raw_proposal_count=raw_proposal_count,
    )
    if report_path.exists():
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise PostformalCandidatePoolError("existing postformal report differs")
    else:
        atomic_json(report_path, report, frozen=True)
    completion_path = output / "completion_receipt.json"
    completion_base = {
        "format": "etsf_robotwin2_postformal_candidate_pool_completion_receipt_v1",
        "status": "complete_1000_pairs_2000_rollouts_frozen",
        "candidate_count": candidate_count,
        "raw_proposal_count": raw_proposal_count,
        "execution_contract_logical_sha256": contract["logical_sha256"],
        "execution_contract_file_sha256": contract_file_sha,
        "outcome_document_sha256": document["document_sha256"],
        "outcome_file_sha256": sha256_file(outcome_path),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": sha256_file(report_path),
        "pair_count": len(rows),
        "rollout_count": len(rows) * 2,
        "postformal_not_part_of_reference_preregistration": True,
    }
    completion = {**completion_base, "logical_sha256": canonical_sha256(completion_base)}
    if completion_path.exists():
        if json.loads(completion_path.read_text(encoding="utf-8")) != completion:
            raise PostformalCandidatePoolError("existing completion receipt differs")
    else:
        atomic_json(completion_path, completion, frozen=True)
    print(
        "POSTFORMAL_CANDIDATE_POOL_COMPLETE="
        + json.dumps(
            {
                "candidate_count": candidate_count,
                "pairs": len(rows),
                "rollouts": len(rows) * 2,
                "outcome": str(outcome_path),
                "report": str(report_path),
                "completion_receipt": str(completion_path),
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
    "FLOW_NOISE_CONTRACT_FORMAT",
    "NATIVE_XYZ_NOISE_CHANNELS",
    "POOL_AUDIT_FORMAT",
    "PAIR_FORMAT",
    "PostformalCandidatePoolError",
    "SOURCE_TRAIN_NORMALIZED_DIVERSITY_FORMAT",
    "aggregate_risk_adjusted_rank_scores",
    "build_outcome_document",
    "build_report",
    "candidate_pool_contract",
    "canonical_effect_embeddings",
    "evaluation_schedule",
    "greedy_farthest_point_indices",
    "generate_postformal_flow_candidates",
    "flow_noise_construction_roster",
    "method_name",
    "postformal_make_noise",
    "postformal_rostered_noise",
    "pool_selection_audit",
    "proposal_count",
    "runtime_rank_ensemble_contract",
    "score_candidates",
    "select_candidate",
    "source_train_normalized_effect_embeddings",
    "source_train_normalized_translation_effect_embeddings",
]
