#!/usr/bin/env python3
"""Collect schema-v5 dense OpenVLA counterfactual event branches.

Each group starts from one fixed RoboTwin seed and one fixed instruction.  The
frozen OpenVLA actor proposes a deterministic first chunk plus sampled/blended
alternatives.  Every alternative is executed in a fresh copy of the same scene;
the first chunk is the intervention and deterministic OpenVLA controls the rest
of the episode.  In addition to terminal outcomes, this collector records the
state immediately before and after the intervention and derives event/time
targets from the complete branch trajectory.  Schema v5 additionally records
every deterministic continuation query, its executed action chunk, and the
hidden state after that chunk.  Those transitions expose late-stage event,
regression, and recovery supervision that would otherwise be discarded.

The actor is always frozen.  A post-chunk forward pass supplies both the
post-transition hidden target and, when the episode is still active, the next
deterministic continuation chunk.  Terminal post states are captured for
future-latent supervision but their proposed actions are never executed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from robotwin_development_seed_contract import (
    REGISTRY as DEVELOPMENT_SEED_REGISTRY,
    validate_development_manifest,
)
from openvla_etsf_v7_development_confirmation import (
    validate_preregistration as validate_v7_preregistration,
    validate_preregistered_source_files as validate_v7_source_files,
    validate_seed_manifest as validate_v7_seed_manifest,
)

SCHEMA_VERSION = 5
ACTION_DIM = 14
CHUNK = 25
MAX_STEPS = 200
BODY = "piper_piper_0.6"
DEFAULT_TASK = "move_can_pot"
EVENT_VOCAB = ("e0", "e12", "e3", "e4", "eK")
INTERVENTION = "candidate_first_chunk_then_deterministic_actor"
LANGUAGE_CONTRACT = "same_instruction_for_initial_query_and_all_candidate_branches"
HIDDEN_ANCHOR = "token_before_action_block"
POST_QUERY_ACTION_CONTRACT = "executed_as_next_query_when_nonterminal"
IDENTITY_MANIFEST_NAME = "collection_identity.json"


def explicit_seed_registry(
    *,
    allow_unregistered_seeds: bool,
    fresh_seed_manifest: Path | None,
    development_seed_manifest: Path | None,
    v7_seed_manifest: Path | None = None,
) -> str:
    manifests = (fresh_seed_manifest, development_seed_manifest, v7_seed_manifest)
    if sum(value is not None for value in manifests) > 1:
        raise ValueError("fresh, development and v7 seed manifests are mutually exclusive")
    has_contract = any(value is not None for value in manifests)
    if allow_unregistered_seeds != has_contract:
        raise ValueError(
            "unregistered seeds require exactly one fresh or development manifest"
        )
    if fresh_seed_manifest is not None:
        return "explicit_fresh_confirmation"
    if development_seed_manifest is not None:
        return DEVELOPMENT_SEED_REGISTRY
    if v7_seed_manifest is not None:
        return "explicit_v7_prospective_development"
    return "official_150"


def load_runtime_helpers() -> None:
    """Import simulator-facing helpers only for a real collection run.

    Keeping these imports lazy lets ``--self-test`` validate schema/label logic
    in a small CPU environment that does not have RLinf/OmegaConf installed.
    """
    global atomic_json, derive_events, discover_pose_objects, environment_config
    global generate_candidates, install_hidden_hook, load_official_seeds, model_config
    global predict, raw_state, read_poses, reference_action_scale, reset_with_contract
    global scalar_bool, sha256

    from collect_openvla_etsf_candidate_branches import (
        generate_candidates,
        reference_action_scale,
        reset_with_contract,
    )
    from collect_openvla_etsf_rollouts import (
        ACTION_DIM as runtime_action_dim,
        BODY as runtime_body,
        CHUNK as runtime_chunk,
        MAX_STEPS as runtime_max_steps,
        atomic_json,
        derive_events,
        discover_pose_objects,
        environment_config,
        install_hidden_hook,
        load_official_seeds,
        model_config,
        predict,
        raw_state,
        read_poses,
        scalar_bool,
        sha256,
    )

    runtime_contract = (runtime_action_dim, runtime_chunk, runtime_max_steps, runtime_body)
    local_contract = (ACTION_DIM, CHUNK, MAX_STEPS, BODY)
    if runtime_contract != local_contract:
        raise RuntimeError(
            f"Piper/OpenVLA runtime contract changed: {runtime_contract} != {local_contract}"
        )


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def hidden_anchor(model, capture: dict[str, torch.Tensor]) -> np.ndarray:
    hidden = capture["last_hidden_states"]
    anchor = hidden[:, -model.action_dim * model.num_action_chunks - 1]
    return anchor[0].float().cpu().numpy().astype(np.float16)


def validate_event_sequence(
    event_names: Sequence[str], event_steps: Sequence[int], terminal_step: int
) -> None:
    if len(event_names) != len(event_steps) or not event_names:
        raise ValueError("event names/steps must be non-empty and aligned")
    if len(set(event_names)) != len(event_names):
        raise ValueError("canonical events must not repeat")
    if any(name not in EVENT_VOCAB for name in event_names):
        raise ValueError(f"unknown canonical event: {event_names}")
    steps = np.asarray(event_steps, dtype=np.int64)
    if steps[0] != 0 or event_names[0] != "e0":
        raise ValueError("canonical branch must begin with e0 at step 0")
    if np.any(np.diff(steps) < 0):
        raise ValueError("canonical event steps must be nondecreasing")
    if int(steps[-1]) > int(terminal_step):
        raise ValueError("canonical event occurs after terminal step")


def event_id_at_step(
    event_names: Sequence[str], event_steps: Sequence[int], step: int
) -> int:
    """Return the last canonical event reached at or before ``step``."""
    current = EVENT_VOCAB.index("e0")
    for name, event_step in zip(event_names, event_steps):
        if int(event_step) <= int(step):
            current = EVENT_VOCAB.index(str(name))
    return current


def derive_branch_targets(
    event_names: Sequence[str],
    event_steps: Sequence[int],
    post_chunk_step: int,
    terminal_step: int,
) -> dict[str, int | bool | float]:
    """Derive pre/post event and next-event duration labels.

    Duration is measured from the intervention start (step zero).  When the
    branch never reaches a later canonical event, the episode supplies a
    right-censoring lower bound rather than a fabricated observed duration.
    """
    validate_event_sequence(event_names, event_steps, terminal_step)
    if not 0 <= int(post_chunk_step) <= int(terminal_step):
        raise ValueError("post-chunk step must lie inside the branch")
    future = [
        (str(name), int(step))
        for name, step in zip(event_names, event_steps)
        if int(step) > 0
    ]
    observed = bool(future)
    next_event_id = (
        EVENT_VOCAB.index(future[0][0]) if observed else EVENT_VOCAB.index("e0")
    )
    duration_steps = float(future[0][1]) if observed else float("nan")
    censor_steps = 0 if observed else int(terminal_step)
    return {
        "pre_event_id": EVENT_VOCAB.index("e0"),
        "post_event_id": event_id_at_step(event_names, event_steps, post_chunk_step),
        "next_event_id": next_event_id,
        "post_chunk_step": int(post_chunk_step),
        "duration": duration_steps if observed else float(censor_steps),
        "duration_observed": observed,
        "duration_censored": not observed,
        "next_event_duration_steps": duration_steps,
        "next_event_duration_observed": observed,
        "next_event_duration_censored": not observed,
        "next_event_censor_steps": censor_steps,
    }


def validate_prefix_mask(mask: np.ndarray, executed_length: int) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (CHUNK,):
        raise ValueError(f"invalid first-chunk mask shape: {mask.shape}")
    if not 0 <= int(executed_length) <= CHUNK:
        raise ValueError(f"invalid first-chunk executed length: {executed_length}")
    expected = np.arange(CHUNK) < int(executed_length)
    if not np.array_equal(mask, expected):
        raise ValueError("first-chunk mask is not a contiguous executed prefix")


def run_self_test() -> None:
    reached = derive_branch_targets(
        ["e0", "e12", "e3", "eK"], [0, 7, 31, 50], 25, 50
    )
    assert reached["pre_event_id"] == 0
    assert reached["post_event_id"] == EVENT_VOCAB.index("e12")
    assert reached["next_event_id"] == EVENT_VOCAB.index("e12")
    assert reached["next_event_duration_steps"] == 7.0
    assert reached["next_event_duration_observed"] is True
    assert reached["next_event_duration_censored"] is False

    censored = derive_branch_targets(["e0"], [0], 25, 200)
    assert censored["post_event_id"] == EVENT_VOCAB.index("e0")
    assert censored["next_event_id"] == EVENT_VOCAB.index("e0")
    assert np.isnan(censored["next_event_duration_steps"])
    assert censored["duration"] == 200.0
    assert censored["next_event_duration_observed"] is False
    assert censored["next_event_duration_censored"] is True
    assert censored["next_event_censor_steps"] == 200

    validate_prefix_mask(np.arange(CHUNK) < 11, 11)
    try:
        validate_prefix_mask(np.r_[True, False, True, np.zeros(CHUNK - 3, dtype=bool)], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("non-prefix action mask was accepted")

    branches = [
        {
            "raw_event_names": ["e0", "e1", "eK"],
            "raw_event_steps": np.asarray([0, 7, 50], dtype=np.int32),
            "event_names": ["e0", "e12", "eK"],
            "event_steps": np.asarray([0, 7, 50], dtype=np.int32),
            "trajectory_object_poses": np.r_[
                np.zeros((25, 2, 7), dtype=np.float32),
                np.ones((26, 2, 7), dtype=np.float32),
            ],
            "trajectory_proprio": np.r_[
                np.zeros((25, ACTION_DIM), dtype=np.float32),
                np.ones((26, ACTION_DIM), dtype=np.float32),
            ],
            "query_steps": np.asarray([0, 25], dtype=np.int32),
            "query_post_steps": np.asarray([25, 50], dtype=np.int32),
            "query_hidden": np.stack(
                [np.zeros(4096, dtype=np.float16), np.ones(4096, dtype=np.float16)]
            ),
            "query_post_hidden": np.stack(
                [np.ones(4096, dtype=np.float16), np.full(4096, 2, dtype=np.float16)]
            ),
            "query_actions": np.zeros((2, CHUNK, ACTION_DIM), dtype=np.float32),
            "query_action_mask": np.ones((2, CHUNK), dtype=bool),
        },
        {
            "raw_event_names": ["e0"],
            "raw_event_steps": np.asarray([0], dtype=np.int32),
            "event_names": ["e0"],
            "event_steps": np.asarray([0], dtype=np.int32),
            "trajectory_object_poses": np.r_[
                np.zeros((11, 2, 7), dtype=np.float32),
                np.ones((1, 2, 7), dtype=np.float32),
            ],
            "trajectory_proprio": np.r_[
                np.zeros((11, ACTION_DIM), dtype=np.float32),
                np.ones((1, ACTION_DIM), dtype=np.float32),
            ],
            "query_steps": np.asarray([0], dtype=np.int32),
            "query_post_steps": np.asarray([11], dtype=np.int32),
            "query_hidden": np.zeros((1, 4096), dtype=np.float16),
            "query_post_hidden": np.ones((1, 4096), dtype=np.float16),
            "query_actions": np.zeros((1, CHUNK, ACTION_DIM), dtype=np.float32),
            "query_action_mask": (np.arange(CHUNK) < 11)[None],
        },
    ]
    labels = [
        derive_branch_targets(branch["event_names"], branch["event_steps"], post, terminal)
        for branch, post, terminal in zip(branches, [25, 11], [50, 11])
    ]
    count = len(branches)
    synthetic: dict[str, Any] = {
        "seed": 123,
        "requested_seed": 123,
        "resolved_seed": 123,
        "task": DEFAULT_TASK,
        "body": BODY,
        "instruction": "self test",
        "temperature": 0.7,
        "top_k": 4,
        "preserve_grippers": True,
        "branch_instruction_consistent": True,
        "candidate_names": ["deterministic", "sample_blend_0.5"],
        "object_names": ["can", "pot"],
        "initial_hidden": np.zeros(4096, dtype=np.float16),
        "pre_hidden": np.zeros((count, 4096), dtype=np.float16),
        "post_chunk_hidden": np.ones((count, 4096), dtype=np.float16),
        "candidate_actions": np.zeros((count, CHUNK, ACTION_DIM), dtype=np.float32),
        "source_sampled_actions": np.zeros((count, CHUNK, ACTION_DIM), dtype=np.float32),
        "source_logprobs": np.zeros((count, CHUNK * ACTION_DIM), dtype=np.float32),
        "l2_from_baseline": np.zeros(count, dtype=np.float32),
        "normalized_l2_from_baseline": np.zeros(count, dtype=np.float32),
        "max_abs_from_baseline": np.zeros(count, dtype=np.float32),
        "first_chunk_executed_length": np.asarray([25, 11], dtype=np.int32),
        "first_chunk_action_mask": np.stack(
            [np.arange(CHUNK) < 25, np.arange(CHUNK) < 11]
        ),
        "post_chunk_terminal": np.asarray([False, True]),
        "pre_object_poses": np.zeros((count, 2, 7), dtype=np.float32),
        "post_object_poses": np.ones((count, 2, 7), dtype=np.float32),
        "pre_proprio": np.zeros((count, ACTION_DIM), dtype=np.float32),
        "post_proprio": np.ones((count, ACTION_DIM), dtype=np.float32),
        "success": np.asarray([True, False]),
        "steps": np.asarray([50, 11], dtype=np.int32),
        "queries": np.asarray([1, 0], dtype=np.int32),
        "wall_seconds": np.asarray([1.0, 1.0], dtype=np.float32),
        "branches": branches,
    }
    for key in labels[0]:
        synthetic[key] = np.asarray([label[key] for label in labels])
    with tempfile.TemporaryDirectory(prefix="etsf_event_branch_selftest_") as directory:
        path = Path(directory) / "group.hdf5"
        save_group(path, synthetic)
        audited = validate_group_file(path, 123, count)
        assert audited["success"] == [True, False]
    identity = collection_identity_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "groups": [
                {
                    "index": 0,
                    "seed": 123,
                    "requested_seed": 123,
                    "resolved_seed": 123,
                    "path": "group.hdf5",
                    "candidate_names": synthetic["candidate_names"],
                    "success": [True, False],
                    "steps": [50, 11],
                }
            ],
            "candidate_successes": [1, 0],
        }
    )
    assert "candidate_successes" not in identity
    assert "success" not in identity["groups"][0]
    print("SELF_TEST_COMPLETE=" + json.dumps({"schema_version": SCHEMA_VERSION, "passed": True}))


def evaluate_dense_first_chunk_branch(
    model,
    capture: dict[str, torch.Tensor],
    env,
    seed: int,
    expected_resolved_seed: int,
    fixed_instruction: str,
    first_chunk: torch.Tensor,
    event_spec: dict[str, Any],
    task_name: str,
    required_pose_names: set[str],
) -> dict[str, Any]:
    obs, _, resolved_seed, branch_instruction = reset_with_contract(
        env, seed, fixed_instruction=fixed_instruction
    )
    if resolved_seed != expected_resolved_seed:
        raise RuntimeError(
            f"non-deterministic seed retry: requested {seed}, "
            f"expected {expected_resolved_seed}, got {resolved_seed}"
        )
    subenv = env.venv.envs[0]
    task = subenv.task
    object_names, objects = discover_pose_objects(task, required_pose_names)
    pre_object_poses = read_poses(objects)
    pre_proprio = raw_state(task)

    # This deterministic query is observational and verifies that every branch
    # really begins from the same model state.  Its action is not executed.
    with torch.inference_mode():
        _ = predict(model, obs)
    pre_hidden = hidden_anchor(model, capture)

    trajectory_poses = [pre_object_poses]
    trajectory_proprio = [pre_proprio]
    success = False
    done = False
    steps = 0
    query_steps: list[int] = []
    query_post_steps: list[int] = []
    query_hidden: list[np.ndarray] = []
    query_post_hidden: list[np.ndarray] = []
    query_actions: list[np.ndarray] = []
    query_action_mask: list[np.ndarray] = []
    started = time.time()

    current_hidden = pre_hidden
    chunk = first_chunk[None] if first_chunk.ndim == 2 else first_chunk
    if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1:] != (
        CHUNK,
        ACTION_DIM,
    ):
        raise RuntimeError(f"invalid branch action chunk shape: {tuple(chunk.shape)}")
    post_object_poses: np.ndarray | None = None
    post_proprio: np.ndarray | None = None

    while steps < MAX_STEPS and not done:
        start_step = steps
        query_steps.append(start_step)
        query_hidden.append(current_hidden)
        query_actions.append(chunk[0].float().cpu().numpy().astype(np.float32))
        executed_length = 0
        for action_index in range(chunk.shape[1]):
            action = chunk[:, action_index : action_index + 1]
            obs, _, terminated, truncated, infos = env.step(action, auto_reset=False)
            steps += 1
            executed_length += 1
            trajectory_poses.append(read_poses(objects))
            trajectory_proprio.append(raw_state(task))
            success = success or scalar_bool(infos.get("success", [False]))
            done = scalar_bool(terminated) or scalar_bool(truncated)
            if done or steps >= MAX_STEPS:
                break

        mask = np.arange(CHUNK) < executed_length
        validate_prefix_mask(mask, executed_length)
        query_action_mask.append(mask)
        query_post_steps.append(steps)

        # This single forward pass is both the future-latent target for the
        # executed query and the next deterministic actor query when active.
        with torch.inference_mode():
            next_chunk = predict(model, obs)
        next_hidden = hidden_anchor(model, capture)
        query_post_hidden.append(next_hidden)

        if len(query_steps) == 1:
            post_object_poses = read_poses(objects)
            post_proprio = raw_state(task)
        if done or steps >= MAX_STEPS:
            break
        current_hidden = next_hidden
        chunk = next_chunk

    if not query_steps or post_object_poses is None or post_proprio is None:
        raise RuntimeError("branch executed no action query")
    first_chunk_executed_length = query_post_steps[0] - query_steps[0]
    post_chunk_hidden = query_post_hidden[0]
    continuation_queries = len(query_steps) - 1

    raw_names, raw_steps, event_names, event_steps = derive_events(
        np.stack(trajectory_poses), object_names, success, event_spec, task_name
    )
    targets = derive_branch_targets(
        event_names, event_steps, first_chunk_executed_length, steps
    )
    action_mask = np.arange(CHUNK) < first_chunk_executed_length
    validate_prefix_mask(action_mask, first_chunk_executed_length)
    return {
        "instruction": branch_instruction,
        "resolved_seed": resolved_seed,
        "object_names": object_names,
        "pre_hidden": pre_hidden,
        "post_chunk_hidden": post_chunk_hidden,
        "first_chunk_executed_length": first_chunk_executed_length,
        "first_chunk_action_mask": action_mask,
        "post_chunk_terminal": first_chunk_executed_length == steps,
        "pre_object_poses": pre_object_poses,
        "post_object_poses": post_object_poses,
        "pre_proprio": pre_proprio,
        "post_proprio": post_proprio,
        "raw_event_names": raw_names,
        "raw_event_steps": np.asarray(raw_steps, dtype=np.int32),
        "event_names": event_names,
        "event_steps": np.asarray(event_steps, dtype=np.int32),
        "trajectory_object_poses": np.stack(trajectory_poses),
        "trajectory_proprio": np.stack(trajectory_proprio),
        "query_steps": np.asarray(query_steps, dtype=np.int32),
        "query_post_steps": np.asarray(query_post_steps, dtype=np.int32),
        "query_hidden": np.stack(query_hidden).astype(np.float16),
        "query_post_hidden": np.stack(query_post_hidden).astype(np.float16),
        "query_actions": np.stack(query_actions).astype(np.float32),
        "query_action_mask": np.stack(query_action_mask),
        "success": success,
        "steps": steps,
        "queries": continuation_queries,
        "wall_seconds": time.time() - started,
        **targets,
    }


def validate_group_file(
    path: Path,
    expected_seed: int,
    candidate_count: int,
) -> dict[str, Any]:
    """Audit a completed group before accepting it during resume."""
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(f"schema mismatch in {path}")
        if int(handle.attrs["seed"]) != int(expected_seed):
            raise RuntimeError(f"seed mismatch in {path}")
        if int(handle.attrs.get("candidate_count", -1)) != candidate_count:
            raise RuntimeError(f"candidate-count mismatch in {path}")
        attribute_contract = {
            "body": BODY,
            "intervention": INTERVENTION,
            "language_contract": LANGUAGE_CONTRACT,
            "hidden_anchor": HIDDEN_ANCHOR,
            "post_query_action_contract": POST_QUERY_ACTION_CONTRACT,
        }
        for key, expected in attribute_contract.items():
            if handle.attrs.get(key) != expected:
                raise RuntimeError(f"attribute contract {key} failed in {path}")
        if not bool(handle.attrs.get("branch_instruction_consistent", False)):
            raise RuntimeError(f"fixed-language contract failed in {path}")
        expected_shapes = {
            "initial_hidden": (4096,),
            "pre_hidden": (candidate_count, 4096),
            "post_chunk_hidden": (candidate_count, 4096),
            "candidate_actions": (candidate_count, CHUNK, ACTION_DIM),
            "source_sampled_actions": (candidate_count, CHUNK, ACTION_DIM),
            "source_logprobs": (candidate_count, CHUNK * ACTION_DIM),
            "first_chunk_action_mask": (candidate_count, CHUNK),
            "first_chunk_executed_length": (candidate_count,),
            "post_chunk_terminal": (candidate_count,),
            "post_chunk_step": (candidate_count,),
            "duration": (candidate_count,),
            "duration_observed": (candidate_count,),
            "duration_censored": (candidate_count,),
            "success": (candidate_count,),
            "steps": (candidate_count,),
            "pre_proprio": (candidate_count, ACTION_DIM),
            "post_proprio": (candidate_count, ACTION_DIM),
        }
        for key, shape in expected_shapes.items():
            if key not in handle or handle[key].shape != shape:
                raise RuntimeError(f"invalid {key} shape in {path}: {handle.get(key)}")
        if len(handle["candidate_names"]) != candidate_count:
            raise RuntimeError(f"candidate names missing in {path}")
        object_count = len(handle["object_names"])
        object_names = decode_strings(handle["object_names"][:])
        if not object_names or len(set(object_names)) != len(object_names):
            raise RuntimeError(f"invalid object names in {path}")
        for key in ["pre_object_poses", "post_object_poses"]:
            if handle[key].shape != (candidate_count, object_count, 7):
                raise RuntimeError(f"invalid {key} shape in {path}")
        finite_keys = [
            "initial_hidden",
            "pre_hidden",
            "post_chunk_hidden",
            "candidate_actions",
            "source_sampled_actions",
            "source_logprobs",
            "l2_from_baseline",
            "normalized_l2_from_baseline",
            "max_abs_from_baseline",
            "pre_object_poses",
            "post_object_poses",
            "pre_proprio",
            "post_proprio",
            "duration",
        ]
        if any(not np.isfinite(handle[key][:]).all() for key in finite_keys):
            raise RuntimeError(f"non-finite dense branch data in {path}")
        initial_hidden = handle["initial_hidden"][:]
        if not np.array_equal(
            handle["pre_hidden"][:],
            np.repeat(initial_hidden[None], candidate_count, axis=0),
        ):
            raise RuntimeError(f"branch pre-hidden contract failed in {path}")
        lengths = handle["first_chunk_executed_length"][:].astype(np.int32)
        masks = handle["first_chunk_action_mask"][:].astype(bool)
        terminal_steps = handle["steps"][:].astype(np.int32)
        post_terminal = handle["post_chunk_terminal"][:].astype(bool)
        for mask, length in zip(masks, lengths):
            validate_prefix_mask(mask, int(length))
        if np.any(lengths > terminal_steps) or not np.array_equal(
            post_terminal, lengths == terminal_steps
        ):
            raise RuntimeError(f"invalid post-chunk terminal contract in {path}")
        if "branches" not in handle or len(handle["branches"]) != candidate_count:
            raise RuntimeError(f"branch event groups missing in {path}")
        for index in range(candidate_count):
            branch = handle["branches"][f"candidate_{index:03d}"]
            names = decode_strings(branch["event_names"][:])
            event_steps = branch["event_steps"][:].astype(np.int32)
            terminal_step = int(terminal_steps[index])
            success = bool(handle["success"][index])
            trajectory_poses = branch["object_poses"][:]
            trajectory_proprio = branch["proprio"][:]
            expected_length = terminal_step + 1
            if trajectory_poses.shape != (expected_length, object_count, 7):
                raise RuntimeError(
                    f"invalid trajectory object poses in {path}, candidate {index}: "
                    f"{trajectory_poses.shape}"
                )
            if trajectory_proprio.shape != (expected_length, ACTION_DIM):
                raise RuntimeError(
                    f"invalid trajectory proprio in {path}, candidate {index}: "
                    f"{trajectory_proprio.shape}"
                )
            if not np.isfinite(trajectory_poses).all() or not np.isfinite(
                trajectory_proprio
            ).all():
                raise RuntimeError(
                    f"non-finite trajectory in {path}, candidate {index}"
                )
            query_keys = {
                "query_steps",
                "query_post_steps",
                "query_hidden",
                "query_post_hidden",
                "query_actions",
                "query_action_mask",
            }
            if not query_keys.issubset(branch):
                raise RuntimeError(
                    f"continuation query data missing in {path}, candidate {index}"
                )
            query_steps = branch["query_steps"][:].astype(np.int32)
            query_post_steps = branch["query_post_steps"][:].astype(np.int32)
            query_count = len(query_steps)
            expected_query_shapes = {
                "query_post_steps": (query_count,),
                "query_hidden": (query_count, 4096),
                "query_post_hidden": (query_count, 4096),
                "query_actions": (query_count, CHUNK, ACTION_DIM),
                "query_action_mask": (query_count, CHUNK),
            }
            for key, shape in expected_query_shapes.items():
                if branch[key].shape != shape:
                    raise RuntimeError(
                        f"invalid {key} shape in {path}, candidate {index}: "
                        f"{branch[key].shape} != {shape}"
                    )
            if query_count < 1 or query_steps[0] != 0:
                raise RuntimeError(
                    f"query sequence must start at zero in {path}, candidate {index}"
                )
            if query_post_steps[-1] != terminal_step or not np.array_equal(
                query_steps[1:], query_post_steps[:-1]
            ):
                raise RuntimeError(
                    f"query boundaries are not contiguous in {path}, candidate {index}"
                )
            query_lengths = query_post_steps - query_steps
            if np.any(query_lengths <= 0) or np.any(query_lengths > CHUNK):
                raise RuntimeError(
                    f"query lengths are invalid in {path}, candidate {index}"
                )
            query_masks = branch["query_action_mask"][:].astype(bool)
            for query_mask, query_length in zip(query_masks, query_lengths):
                validate_prefix_mask(query_mask, int(query_length))
            query_finite_keys = [
                "query_hidden",
                "query_post_hidden",
                "query_actions",
            ]
            if any(
                not np.isfinite(branch[key][:]).all() for key in query_finite_keys
            ):
                raise RuntimeError(
                    f"non-finite continuation query data in {path}, candidate {index}"
                )
            query_hidden = branch["query_hidden"][:]
            query_post_hidden = branch["query_post_hidden"][:]
            query_actions = branch["query_actions"][:]
            if not np.array_equal(query_hidden[0], handle["pre_hidden"][index]):
                raise RuntimeError(
                    f"initial query hidden mismatch in {path}, candidate {index}"
                )
            if query_count > 1 and not np.array_equal(
                query_hidden[1:], query_post_hidden[:-1]
            ):
                raise RuntimeError(
                    f"query hidden chain mismatch in {path}, candidate {index}"
                )
            if not np.array_equal(
                query_post_hidden[0], handle["post_chunk_hidden"][index]
            ) or not np.array_equal(
                query_actions[0], handle["candidate_actions"][index]
            ) or not np.array_equal(
                query_masks[0], handle["first_chunk_action_mask"][index]
            ):
                raise RuntimeError(
                    f"first-query compatibility contract failed in {path}, candidate {index}"
                )
            if query_post_steps[0] != int(lengths[index]) or int(
                handle["queries"][index]
            ) != query_count - 1:
                raise RuntimeError(
                    f"query count/first boundary mismatch in {path}, candidate {index}"
                )
            post_step = int(lengths[index])
            if not np.array_equal(
                trajectory_poses[0], handle["pre_object_poses"][index]
            ) or not np.array_equal(
                trajectory_poses[post_step], handle["post_object_poses"][index]
            ):
                raise RuntimeError(
                    f"trajectory/object boundary mismatch in {path}, candidate {index}"
                )
            if not np.array_equal(
                trajectory_proprio[0], handle["pre_proprio"][index]
            ) or not np.array_equal(
                trajectory_proprio[post_step], handle["post_proprio"][index]
            ):
                raise RuntimeError(
                    f"trajectory/proprio boundary mismatch in {path}, candidate {index}"
                )
            if ("eK" in names) != success:
                raise RuntimeError(f"eK/success contract failed in {path}, candidate {index}")
            targets = derive_branch_targets(names, event_steps, int(lengths[index]), terminal_step)
            for key in [
                "pre_event_id",
                "post_event_id",
                "next_event_id",
                "post_chunk_step",
                "duration",
                "duration_observed",
                "duration_censored",
                "next_event_duration_observed",
                "next_event_duration_censored",
                "next_event_censor_steps",
            ]:
                actual = handle[key][index].item()
                if actual != targets[key]:
                    raise RuntimeError(f"inconsistent {key} label in {path}, candidate {index}")
            duration = float(handle["next_event_duration_steps"][index])
            expected_duration = float(targets["next_event_duration_steps"])
            if not (
                (np.isnan(duration) and np.isnan(expected_duration))
                or duration == expected_duration
            ):
                raise RuntimeError(f"inconsistent duration label in {path}, candidate {index}")
        return {
            "seed": int(handle.attrs["seed"]),
            "requested_seed": int(handle.attrs["requested_seed"]),
            "resolved_seed": int(handle.attrs["resolved_seed"]),
            "candidate_names": decode_strings(handle["candidate_names"][:]),
            "success": handle["success"][:].astype(bool).tolist(),
            "steps": handle["steps"][:].astype(int).tolist(),
            "query_transitions": int(
                (handle["queries"][:].astype(np.int64) + 1).sum()
            ),
            "post_event_id": handle["post_event_id"][:].astype(int).tolist(),
            "next_event_duration_observed": handle[
                "next_event_duration_observed"
            ][:].astype(bool).tolist(),
        }


def save_group(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        for key in [
            "seed",
            "requested_seed",
            "resolved_seed",
            "task",
            "body",
            "instruction",
            "temperature",
            "top_k",
            "preserve_grippers",
        ]:
            handle.attrs[key] = record[key]
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["candidate_count"] = len(record["candidate_names"])
        handle.attrs["intervention"] = INTERVENTION
        handle.attrs["language_contract"] = LANGUAGE_CONTRACT
        handle.attrs["branch_instruction_consistent"] = bool(
            record["branch_instruction_consistent"]
        )
        handle.attrs["hidden_anchor"] = HIDDEN_ANCHOR
        handle.attrs["post_query_action_contract"] = POST_QUERY_ACTION_CONTRACT
        handle.create_dataset(
            "candidate_names",
            data=np.asarray(record["candidate_names"], dtype=object),
            dtype=strings,
        )
        handle.create_dataset(
            "object_names",
            data=np.asarray(record["object_names"], dtype=object),
            dtype=strings,
        )
        array_keys = [
            "initial_hidden",
            "pre_hidden",
            "post_chunk_hidden",
            "candidate_actions",
            "source_sampled_actions",
            "source_logprobs",
            "l2_from_baseline",
            "normalized_l2_from_baseline",
            "max_abs_from_baseline",
            "first_chunk_executed_length",
            "first_chunk_action_mask",
            "post_chunk_terminal",
            "post_chunk_step",
            "pre_object_poses",
            "post_object_poses",
            "pre_proprio",
            "post_proprio",
            "pre_event_id",
            "post_event_id",
            "next_event_id",
            "duration",
            "duration_observed",
            "duration_censored",
            "next_event_duration_steps",
            "next_event_duration_observed",
            "next_event_duration_censored",
            "next_event_censor_steps",
            "success",
            "steps",
            "queries",
            "wall_seconds",
        ]
        for key in array_keys:
            value = np.asarray(record[key])
            compression = "gzip" if value.size > 64 else None
            handle.create_dataset(key, data=value, compression=compression)
        branches = handle.create_group("branches")
        for index, branch_record in enumerate(record["branches"]):
            branch = branches.create_group(f"candidate_{index:03d}")
            for prefix in ["raw_event", "event"]:
                branch.create_dataset(
                    f"{prefix}_names",
                    data=np.asarray(branch_record[f"{prefix}_names"], dtype=object),
                    dtype=strings,
                )
                branch.create_dataset(
                    f"{prefix}_steps",
                    data=np.asarray(branch_record[f"{prefix}_steps"], dtype=np.int32),
                )
            branch.create_dataset(
                "object_poses",
                data=np.asarray(
                    branch_record["trajectory_object_poses"], dtype=np.float32
                ),
                compression="gzip",
            )
            branch.create_dataset(
                "proprio",
                data=np.asarray(branch_record["trajectory_proprio"], dtype=np.float32),
                compression="gzip",
            )
            for key, dtype in [
                ("query_steps", np.int32),
                ("query_post_steps", np.int32),
                ("query_hidden", np.float16),
                ("query_post_hidden", np.float16),
                ("query_actions", np.float32),
                ("query_action_mask", bool),
            ]:
                branch.create_dataset(
                    key,
                    data=np.asarray(branch_record[key], dtype=dtype),
                    compression=(
                        "gzip" if np.asarray(branch_record[key]).size > 64 else None
                    ),
                )
        handle.flush()
    os.replace(temporary, path)


def collection_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a label-free collection view safe to inspect before reservation."""

    root_keys = (
        "schema_version",
        "status",
        "collector_seed",
        "task",
        "body",
        "model_path",
        "unnorm_key",
        "requested_seeds",
        "resolved_seeds",
        "seed_registry",
        "fresh_seed_manifest",
        "fresh_seed_manifest_sha256",
        "development_seed_manifest",
        "development_seed_manifest_sha256",
        "v7_seed_manifest",
        "v7_seed_manifest_sha256",
        "v7_preregistration",
        "v7_preregistration_sha256",
        "candidate_count",
        "blends",
        "temperature",
        "top_k",
        "preserve_grippers",
        "intervention",
        "language_contract",
        "event_vocab",
        "event_spec",
        "event_spec_sha256",
        "hidden_dim",
        "hidden_anchor",
        "action_dim",
        "action_chunk",
        "max_steps",
        "trajectory_contract",
        "continuation_query_contract",
        "completed",
    )
    group_keys = (
        "index",
        "seed",
        "requested_seed",
        "resolved_seed",
        "path",
        "candidate_names",
        "status",
    )
    payload = {key: manifest.get(key) for key in root_keys if key in manifest}
    payload["format"] = "etsf_event_branch_collection_identity_v1"
    payload["groups"] = [
        {key: row.get(key) for key in group_keys if key in row}
        for row in manifest.get("groups", [])
        if isinstance(row, Mapping)
    ]
    payload["label_access_contract"] = (
        "identity_only_no_success_steps_event_or_outcome_fields"
    )
    payload["hdf5_sha256_pre_evaluation"] = "not_computed"
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
    if forbidden & set(payload) or any(
        forbidden & set(row) for row in payload["groups"]
    ):
        raise RuntimeError("collection identity payload contains outcome labels")
    return payload


def write_collection_manifests(output: Path, manifest: Mapping[str, Any]) -> None:
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / IDENTITY_MANIFEST_NAME,
        collection_identity_payload(manifest),
    )


def select_seeds(args: argparse.Namespace, seeds_path: Path) -> list[int]:
    if args.seeds_file is not None:
        source = json.loads(args.seeds_file.read_text(encoding="utf-8"))
        if args.seeds_key is not None:
            source = source[args.seeds_key]
        if not isinstance(source, list):
            raise ValueError("seed file selection must be a JSON list")
        seeds = [int(item["seed"] if isinstance(item, dict) else item) for item in source]
    elif args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
    else:
        seeds = load_official_seeds(seeds_path, args.task, args.limit, args.offset)
    official = set(load_official_seeds(seeds_path, args.task, 150, 0))
    invalid = sorted(set(seeds) - official)
    if invalid and not args.allow_unregistered_seeds:
        raise ValueError(f"non-official seeds requested: {invalid}")
    if (
        args.allow_unregistered_seeds
        and args.seeds is None
        and args.seeds_file is None
    ):
        raise ValueError(
            "--allow-unregistered-seeds requires explicit --seeds or --seeds-file"
        )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed selection is empty or contains duplicates")
    return seeds


def main() -> None:
    if "--self-test" in sys.argv[1:]:
        run_self_test()
        return
    load_runtime_helpers()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--seeds-file",
        type=Path,
        help="JSON list, or a shadow split manifest selected with --seeds-key",
    )
    parser.add_argument("--seeds-key", choices=["train", "validation", "test"])
    parser.add_argument("--reference-rollouts", type=Path)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--allow-sampled-grippers", action="store_true")
    parser.add_argument(
        "--allow-unregistered-seeds",
        action="store_true",
        help=(
            "Allow an explicit --seeds list outside the original 150-seed registry. "
            "Use only with one pre-registered fresh or development manifest."
        ),
    )
    parser.add_argument(
        "--fresh-seed-manifest",
        type=Path,
        help=(
            "Frozen reset-only resolved seed manifest. Required with "
            "--allow-unregistered-seeds and hashed into the collection contract."
        ),
    )
    parser.add_argument(
        "--development-seed-manifest",
        type=Path,
        help=(
            "Frozen reset-only development expansion manifest. Mutually exclusive "
            "with --fresh-seed-manifest and recorded as development, never fresh."
        ),
    )
    parser.add_argument(
        "--v7-seed-manifest", type=Path,
        help="Resolved label-free v7 prospective development seed manifest.",
    )
    parser.add_argument(
        "--v7-preregistration", type=Path,
        help="Signed v7 formula/code preregistration frozen before collection.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.seeds is not None and args.seeds_file is not None:
        parser.error("--seeds and --seeds-file are mutually exclusive")
    if args.seeds is None and args.seeds_file is None:
        parser.error(
            "explicit seed selection is required: use --seeds or "
            "--seeds-file/--seeds-key; implicit --limit/--offset selection is disabled"
        )
    try:
        seed_registry = explicit_seed_registry(
            allow_unregistered_seeds=args.allow_unregistered_seeds,
            fresh_seed_manifest=args.fresh_seed_manifest,
            development_seed_manifest=args.development_seed_manifest,
            v7_seed_manifest=args.v7_seed_manifest,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.seeds_key is not None and args.seeds_file is None:
        parser.error("--seeds-key requires --seeds-file")
    if args.task != DEFAULT_TASK:
        parser.error(
            f"schema v5 currently supports only Piper/OpenVLA task {DEFAULT_TASK!r}"
        )
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k == 0 or args.top_k < -1:
        parser.error("--top-k must be -1 or positive")
    if not args.blends or any(not 0 < value <= 1 for value in args.blends):
        parser.error("--blends values must be in (0, 1]")

    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openvla_oft.official import get_model

    event_spec = json.loads(args.event_spec.read_text(encoding="utf-8"))
    if args.task not in event_spec["calibration"] or args.task not in event_spec["chains"]:
        raise KeyError(f"task {args.task!r} is missing from the event specification")
    chain = list(event_spec["chains"][args.task]["chain"])
    if any(event not in EVENT_VOCAB for event in chain):
        raise ValueError(f"event chain is incompatible with schema-v5 vocabulary: {chain}")
    calibration = event_spec["calibration"][args.task]
    required_pose_names = {str(calibration["moving"])}
    if calibration["anchor"]:
        required_pose_names.add(str(calibration["anchor"]))

    args.output.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output / "groups"
    groups_dir.mkdir(exist_ok=True)
    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    seeds = select_seeds(args, seeds_path)
    fresh_seed_manifest_sha256: str | None = None
    fresh_resolved_by_requested: dict[int, int] = {}
    development_seed_manifest_sha256: str | None = None
    development_resolved_by_requested: dict[int, int] = {}
    v7_seed_manifest_sha256: str | None = None
    v7_preregistration_sha256: str | None = None
    if args.fresh_seed_manifest is not None:
        fresh_contract = json.loads(
            args.fresh_seed_manifest.read_text(encoding="utf-8")
        )
        if fresh_contract.get("status") != "fresh_confirmation_preregistered_resolved":
            raise RuntimeError("fresh seed manifest is not frozen/resolved")
        if str(fresh_contract.get("task")) != args.task:
            raise RuntimeError("fresh seed manifest task mismatch")
        fresh_rows = fresh_contract.get("test", [])
        fresh_resolved_by_requested = {
            int(row["requested_seed"]): int(row["resolved_seed"])
            for row in fresh_rows
        }
        if list(fresh_resolved_by_requested) != seeds:
            raise RuntimeError(
                "collector seeds differ from the ordered frozen fresh seed manifest"
            )
        fresh_seed_manifest_sha256 = sha256(args.fresh_seed_manifest)
    if args.development_seed_manifest is not None:
        development_contract = validate_development_manifest(
            args.development_seed_manifest, task=args.task
        )
        if development_contract["requested_seeds"] != seeds:
            raise RuntimeError(
                "collector seeds differ from ordered development expansion manifest"
            )
        development_resolved_by_requested = {
            int(row["requested_seed"]): int(row["resolved_seed"])
            for row in development_contract["rows"]
        }
        development_seed_manifest_sha256 = development_contract["sha256"]
    if (args.v7_seed_manifest is None) != (args.v7_preregistration is None):
        raise RuntimeError("v7 seed manifest and preregistration must be supplied together")
    if args.v7_seed_manifest is not None:
        v7_seed_value = json.loads(args.v7_seed_manifest.read_text(encoding="utf-8"))
        v7_contract = validate_v7_seed_manifest(v7_seed_value, verify_files=True)
        v7_preregistration = json.loads(args.v7_preregistration.read_text(encoding="utf-8"))
        validate_v7_preregistration(v7_preregistration)
        validate_v7_source_files(v7_preregistration)
        if v7_preregistration["seed_manifest_payload_sha256"] != v7_contract[
            "seed_manifest_payload_sha256"
        ]:
            raise RuntimeError("v7 seed/preregistration contract mismatch")
        if v7_contract["requested_seeds"] != seeds:
            raise RuntimeError("collector seeds differ from ordered v7 seed manifest")
        development_resolved_by_requested = dict(zip(
            v7_contract["requested_seeds"], v7_contract["resolved_seeds"]
        ))
        v7_seed_manifest_sha256 = sha256(args.v7_seed_manifest)
        v7_preregistration_sha256 = v7_preregistration["preregistration_sha256"]
    unnorm_key = args.unnorm_key or f"{args.task}_1k"
    event_spec_digest = sha256(args.event_spec)
    resume_contract = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "body": BODY,
        "model_path": str(args.model_path),
        "unnorm_key": unnorm_key,
        "requested_seeds": seeds,
        "seed_registry": seed_registry,
        "fresh_seed_manifest_sha256": fresh_seed_manifest_sha256,
        "development_seed_manifest_sha256": development_seed_manifest_sha256,
        "v7_seed_manifest_sha256": v7_seed_manifest_sha256,
        "v7_preregistration_sha256": v7_preregistration_sha256,
        "candidate_count": 1 + len(args.blends),
        "blends": args.blends,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "preserve_grippers": not args.allow_sampled_grippers,
        "intervention": INTERVENTION,
        "language_contract": LANGUAGE_CONTRACT,
        "event_spec_sha256": event_spec_digest,
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in resume_contract.items():
            if previous_manifest.get(key) != expected:
                raise RuntimeError(
                    f"resume manifest contract mismatch for {key}: "
                    f"{previous_manifest.get(key)!r} != {expected!r}"
                )

    device = torch.device("cuda:0")
    model = (
        get_model(model_config(args.model_path, unnorm_key), torch_dtype=torch.bfloat16)
        .eval()
        .to(device)
    )
    capture = install_hidden_hook(model)
    action_scale = reference_action_scale(args.reference_rollouts, device)
    env = RoboTwinEnv(
        cfg=environment_config(args.robotwin_root, seeds_path, args.task, len(seeds)),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "collecting",
        "collector_seed": 20260826,
        "task": args.task,
        "body": BODY,
        "model_path": str(args.model_path),
        "unnorm_key": unnorm_key,
        "requested_seeds": seeds,
        "seed_registry": seed_registry,
        "fresh_seed_manifest": (
            str(args.fresh_seed_manifest.resolve())
            if args.fresh_seed_manifest is not None
            else None
        ),
        "fresh_seed_manifest_sha256": fresh_seed_manifest_sha256,
        "development_seed_manifest": (
            str(args.development_seed_manifest.resolve())
            if args.development_seed_manifest is not None
            else None
        ),
        "development_seed_manifest_sha256": development_seed_manifest_sha256,
        "v7_seed_manifest": (
            str(args.v7_seed_manifest.resolve()) if args.v7_seed_manifest is not None else None
        ),
        "v7_seed_manifest_sha256": v7_seed_manifest_sha256,
        "v7_preregistration": (
            str(args.v7_preregistration.resolve()) if args.v7_preregistration is not None else None
        ),
        "v7_preregistration_sha256": v7_preregistration_sha256,
        "candidate_count": 1 + len(args.blends),
        "blends": args.blends,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "preserve_grippers": not args.allow_sampled_grippers,
        "intervention": INTERVENTION,
        "language_contract": LANGUAGE_CONTRACT,
        "event_vocab": list(EVENT_VOCAB),
        "event_spec": str(args.event_spec),
        "event_spec_sha256": event_spec_digest,
        "reference_rollouts": str(args.reference_rollouts) if args.reference_rollouts else None,
        "action_scale": action_scale.cpu().numpy().tolist(),
        "hidden_dim": 4096,
        "hidden_anchor": HIDDEN_ANCHOR,
        "action_dim": ACTION_DIM,
        "action_chunk": CHUNK,
        "max_steps": MAX_STEPS,
        "trajectory_contract": {
            "object_poses": "per_step_including_reset_and_terminal",
            "proprio": "per_step_including_reset_and_terminal",
            "length": "terminal_steps_plus_one",
            "purpose": "dynamic_predicates_failure_and_recovery_labels",
        },
        "continuation_query_contract": {
            "query_steps": "inclusive_action_chunk_start_steps",
            "query_post_steps": "exclusive_action_chunk_end_steps",
            "query_hidden": HIDDEN_ANCHOR,
            "query_post_hidden": "same_anchor_after_executed_chunk_including_terminal",
            "query_actions": f"padded_to_{CHUNK}_simulator_actions",
            "query_action_mask": "contiguous_executed_prefix",
            "post_query_action": POST_QUERY_ACTION_CONTRACT,
            "purpose": "late_event_action_conditioned_auxiliary_transitions",
        },
        "groups": [],
    }
    candidate_count = int(manifest["candidate_count"])
    resolved_seeds_seen: set[int] = set()
    write_collection_manifests(args.output, manifest)
    try:
        for group_index, seed in enumerate(seeds):
            path = groups_dir / f"group_{group_index:03d}_seed_{seed}.hdf5"
            if path.exists() and not args.overwrite:
                existing = validate_group_file(path, seed, candidate_count)
                resolved_seed = int(existing["resolved_seed"])
                if (
                    fresh_resolved_by_requested
                    and resolved_seed != fresh_resolved_by_requested[seed]
                ):
                    raise RuntimeError(
                        f"existing fresh scene resolution changed for {seed}: "
                        f"{resolved_seed} != {fresh_resolved_by_requested[seed]}"
                    )
                if (
                    development_resolved_by_requested
                    and resolved_seed != development_resolved_by_requested[seed]
                ):
                    raise RuntimeError(
                        f"existing development scene resolution changed for {seed}: "
                        f"{resolved_seed} != {development_resolved_by_requested[seed]}"
                    )
                if resolved_seed in resolved_seeds_seen:
                    raise RuntimeError(f"duplicate resolved scene {resolved_seed}")
                resolved_seeds_seen.add(resolved_seed)
                manifest["groups"].append(
                    {
                        "index": group_index,
                        "path": path.name,
                        "status": "existing",
                        **existing,
                    }
                )
                continue

            obs, _, resolved_seed, instruction = reset_with_contract(env, seed)
            if fresh_resolved_by_requested and resolved_seed != fresh_resolved_by_requested[seed]:
                raise RuntimeError(
                    f"fresh scene resolution changed for {seed}: "
                    f"{resolved_seed} != {fresh_resolved_by_requested[seed]}"
                )
            if (
                development_resolved_by_requested
                and resolved_seed != development_resolved_by_requested[seed]
            ):
                raise RuntimeError(
                    f"development scene resolution changed for {seed}: "
                    f"{resolved_seed} != {development_resolved_by_requested[seed]}"
                )
            if resolved_seed in resolved_seeds_seen:
                raise RuntimeError(f"requested seed {seed} resolves to duplicate scene {resolved_seed}")
            resolved_seeds_seen.add(resolved_seed)
            generated = generate_candidates(
                model,
                obs,
                capture,
                seed,
                args.blends,
                args.temperature,
                args.top_k,
                not args.allow_sampled_grippers,
                action_scale,
            )
            initial_hidden = generated["initial_hidden"]
            outcomes = [
                evaluate_dense_first_chunk_branch(
                    model,
                    capture,
                    env,
                    seed,
                    resolved_seed,
                    instruction,
                    candidate,
                    event_spec,
                    args.task,
                    required_pose_names,
                )
                for candidate in generated.pop("candidate_actions_tensor")
            ]
            if any(item["resolved_seed"] != resolved_seed for item in outcomes):
                raise RuntimeError(f"branch resolved-seed contract failed for seed {seed}")
            if any(item["instruction"] != instruction for item in outcomes):
                raise RuntimeError(f"branch language changed for seed {seed}")
            object_names = outcomes[0]["object_names"]
            if any(item["object_names"] != object_names for item in outcomes):
                raise RuntimeError(f"tracked object order changed across branches for seed {seed}")
            first_pre_poses = outcomes[0]["pre_object_poses"]
            first_pre_proprio = outcomes[0]["pre_proprio"]
            if any(
                not np.array_equal(item["pre_object_poses"], first_pre_poses)
                or not np.array_equal(item["pre_proprio"], first_pre_proprio)
                for item in outcomes[1:]
            ):
                raise RuntimeError(f"branch pre-state changed for seed {seed}")
            pre_hidden = np.stack([item["pre_hidden"] for item in outcomes])
            if not np.array_equal(
                pre_hidden, np.repeat(initial_hidden[None], candidate_count, axis=0)
            ):
                raise RuntimeError(f"branch pre-hidden changed for seed {seed}")

            record = {
                **generated,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": resolved_seed,
                "task": args.task,
                "body": BODY,
                "instruction": instruction,
                "branch_instruction_consistent": True,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "preserve_grippers": not args.allow_sampled_grippers,
                "object_names": object_names,
                "pre_hidden": pre_hidden,
                "post_chunk_hidden": np.stack(
                    [item["post_chunk_hidden"] for item in outcomes]
                ),
                "branches": outcomes,
            }
            vector_keys = [
                "first_chunk_executed_length",
                "first_chunk_action_mask",
                "post_chunk_terminal",
                "post_chunk_step",
                "pre_object_poses",
                "post_object_poses",
                "pre_proprio",
                "post_proprio",
                "pre_event_id",
                "post_event_id",
                "next_event_id",
                "duration",
                "duration_observed",
                "duration_censored",
                "next_event_duration_steps",
                "next_event_duration_observed",
                "next_event_duration_censored",
                "next_event_censor_steps",
                "success",
                "steps",
                "queries",
                "wall_seconds",
            ]
            for key in vector_keys:
                record[key] = np.asarray([item[key] for item in outcomes])
            save_group(path, record)
            validate_group_file(path, seed, candidate_count)
            item = {
                "index": group_index,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": resolved_seed,
                "path": path.name,
                "candidate_names": record["candidate_names"],
                "success": record["success"].astype(bool).tolist(),
                "steps": record["steps"].astype(int).tolist(),
                "query_transitions": int((record["queries"] + 1).sum()),
                "post_event_id": record["post_event_id"].astype(int).tolist(),
                "next_event_duration_observed": record[
                    "next_event_duration_observed"
                ].astype(bool).tolist(),
                "wall_seconds": float(record["wall_seconds"].sum()),
                "status": "collected",
            }
            manifest["groups"].append(item)
            manifest["completed"] = len(manifest["groups"])
            write_collection_manifests(args.output, manifest)
            print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)
    finally:
        env.venv.close(clear_cache=False)

    manifest["status"] = "complete"
    manifest["completed"] = len(manifest["groups"])
    successes = np.asarray([item["success"] for item in manifest["groups"]], dtype=np.int64)
    manifest["candidate_successes"] = successes.sum(axis=0).tolist()
    manifest["candidate_success_rates"] = successes.mean(axis=0).tolist()
    manifest["groups_with_outcome_variation"] = int(
        sum(len(set(item["success"])) > 1 for item in manifest["groups"])
    )
    manifest["resolved_seeds"] = [int(item["resolved_seed"]) for item in manifest["groups"]]
    manifest["dense_label_counts"] = {
        "branches": int(successes.size),
        "action_conditioned_query_transitions": int(
            sum(int(item["query_transitions"]) for item in manifest["groups"])
        ),
        "duration_observed": int(
            sum(sum(item["next_event_duration_observed"]) for item in manifest["groups"])
        ),
        "duration_censored": int(
            successes.size
            - sum(sum(item["next_event_duration_observed"]) for item in manifest["groups"])
        ),
    }
    write_collection_manifests(args.output, manifest)
    print(
        "COLLECTION_COMPLETE="
        + json.dumps(
            {
                "completed": manifest["completed"],
                "candidate_successes": manifest["candidate_successes"],
                "groups_with_outcome_variation": manifest["groups_with_outcome_variation"],
                "dense_label_counts": manifest["dense_label_counts"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
