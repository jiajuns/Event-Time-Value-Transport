#!/usr/bin/env python3
"""Collect schema-v5 SmolVLA branches with a noise-independent VLM state.

The saved state is the contextualized final state token from SmolVLA's VLM
prefix pass.  That pass fills the prefix KV cache before the flow-matching
denoising loop; it is shared by every explicit-noise candidate for one query.
The 720-D action-expert hidden is deliberately never written by this collector.

The runtime path is intentionally fail-closed.  It currently supports the
audited LeRobot 0.4.4 layout with ``prefix_length == 0`` and requires the VLM
prefix token to be bit-identical across all candidates.  ``--self-test`` only
checks the HDF5/schema invariants on CPU; it is not a model or simulator smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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


SCHEMA_VERSION = 5
ACTION_DIM = 14
AUDITED_CHUNK = 50
MAX_STEPS = 200
BODY = "aloha-agilex"
POLICY = "smolvla"
DEFAULT_TASK = "move_can_pot"
EVENT_VOCAB = ("e0", "e12", "e3", "e4", "eK")
INTERVENTION = "candidate_first_executed_prefix_then_fixed_noise_baseline"
LANGUAGE_CONTRACT = "same_instruction_for_initial_query_and_all_candidate_branches"
SHARED_STATE_ANCHOR = (
    "contextualized_vlm_prefix_final_state_token_before_flow_noise_v1"
)
SHARED_STATE_SOURCE = (
    "policy.model.vlm_with_expert.get_vlm_model().text_model.norm"
)
POST_QUERY_ACTION_CONTRACT = "executed_as_next_query_when_nonterminal"
BASELINE_NAME = "deterministic"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_state_contract(
    *, hidden_dim: int, modeling_sha256: str, bridge_sha256: str
) -> dict[str, Any]:
    """Return a content-addressed state contract shared with training/deployment."""

    base: dict[str, Any] = {
        "policy": POLICY,
        "anchor": SHARED_STATE_ANCHOR,
        "source": SHARED_STATE_SOURCE,
        "hidden_dim": int(hidden_dim),
        "prefix_length": 0,
        "noise_independence": "bit_exact_at_group_intervention_query",
        "modeling_sha256": str(modeling_sha256),
        "bridge_sha256": str(bridge_sha256),
    }
    encoded = json.dumps(
        base, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        **base,
        "calibration_id": hashlib.sha256(encoded).hexdigest(),
    }


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class SharedPrefixStateCapture:
    """Capture the observation-only VLM prefix state used by all candidates.

    In the audited implementation ``embed_prefix`` appends one projected robot
    state token after image/language tokens.  With ``prefix_length == 0`` there
    is no trailing padding, so the last contextualized VLM token is that state
    token.  A non-zero configured prefix length is rejected rather than risking
    the selection of a padding token.
    """

    def __init__(self, module: torch.nn.Module, *, prefix_length: int) -> None:
        if prefix_length != 0:
            raise RuntimeError(
                "shared-prefix hook v1 requires prefix_length=0; padded prefixes "
                "need an explicit valid-token index implementation"
            )
        self.latest: torch.Tensor | None = None
        self.calls = 0
        self.handle = module.register_forward_hook(self._hook)

    @classmethod
    def from_policy(cls, policy: Any) -> "SharedPrefixStateCapture":
        try:
            bridge = policy.model.vlm_with_expert
            norm = bridge.get_vlm_model().text_model.norm
            prefix_length = int(policy.config.prefix_length)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "unsupported SmolVLA layout: cannot locate audited VLM prefix norm"
            ) from error
        if not isinstance(norm, torch.nn.Module):
            raise RuntimeError("audited SmolVLA VLM prefix norm is not a torch module")
        return cls(norm, prefix_length=prefix_length)

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensor = first_tensor(output)
        if tensor is None:
            return
        if tensor.ndim != 3:
            raise RuntimeError(
                f"VLM prefix norm must emit [B,L,D], got {tuple(tensor.shape)}"
            )
        self.calls += 1
        self.latest = tensor.detach().float().cpu()

    def reset(self) -> None:
        self.latest = None
        self.calls = 0

    def consume(self) -> torch.Tensor:
        if self.calls != 1 or self.latest is None:
            raise RuntimeError(
                "shared VLM prefix hook must fire exactly once per action query; "
                f"calls={self.calls}"
            )
        if self.latest.shape[0] != 1 or self.latest.shape[1] < 1:
            raise RuntimeError(
                f"shared VLM prefix must be [1,L,D], got {tuple(self.latest.shape)}"
            )
        state = self.latest[0, -1]
        if state.ndim != 1 or not bool(torch.isfinite(state).all()):
            raise RuntimeError("captured shared VLM state is invalid")
        return state.clone()

    def close(self) -> None:
        self.handle.remove()


def resolve_shared_prefix_capture(policy: Any) -> SharedPrefixStateCapture:
    """Public fail-closed hook resolver used by the 4090 smoke test."""

    return SharedPrefixStateCapture.from_policy(policy)


def noise_seed(scene_seed: int, query_index: int, candidate_index: int) -> int:
    return int(
        (
            20260827
            + int(scene_seed) * 1_000_003
            + int(query_index) * 10_007
            + int(candidate_index) * 101
        )
        % (2**63 - 1)
    )


def make_noise(
    config: Any,
    scene_seed: int,
    query_index: int,
    candidate_index: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed(scene_seed, query_index, candidate_index))
    return torch.randn(
        (1, int(config.chunk_size), int(config.max_action_dim)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def validate_prefix_mask(mask: np.ndarray, executed_length: int, chunk: int) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (chunk,):
        raise ValueError(f"invalid action-mask shape {mask.shape}; expected {(chunk,)}")
    if not 0 <= int(executed_length) <= chunk:
        raise ValueError("executed action length lies outside the chunk")
    if not np.array_equal(mask, np.arange(chunk) < int(executed_length)):
        raise ValueError("action mask is not a contiguous executed prefix")


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
    if event_names[0] != "e0" or int(steps[0]) != 0:
        raise ValueError("canonical event sequence must begin with e0 at step zero")
    if np.any(np.diff(steps) < 0) or int(steps[-1]) > int(terminal_step):
        raise ValueError("canonical event steps are invalid")


def event_id_at_step(
    event_names: Sequence[str], event_steps: Sequence[int], step: int
) -> int:
    current = 0
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
    validate_event_sequence(event_names, event_steps, terminal_step)
    if not 0 <= int(post_chunk_step) <= int(terminal_step):
        raise ValueError("post-chunk step lies outside the branch")
    future = [
        (str(name), int(step))
        for name, step in zip(event_names, event_steps)
        if int(step) > 0
    ]
    observed = bool(future)
    duration = float(future[0][1]) if observed else float(terminal_step)
    return {
        "pre_event_id": 0,
        "post_event_id": event_id_at_step(event_names, event_steps, post_chunk_step),
        "next_event_id": EVENT_VOCAB.index(future[0][0]) if observed else 0,
        "post_chunk_step": int(post_chunk_step),
        "duration": duration,
        "duration_observed": observed,
        "duration_censored": not observed,
        "next_event_duration_steps": float(future[0][1]) if observed else float("nan"),
        "next_event_duration_observed": observed,
        "next_event_duration_censored": not observed,
        "next_event_censor_steps": 0 if observed else int(terminal_step),
    }


def generate_candidates(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    capture: SharedPrefixStateCapture,
    obs: dict[str, Any],
    scene_seed: int,
    query_index: int,
    candidate_count: int,
    device: torch.device,
    image_keys: list[str],
    raw_policy_input_fn: Any,
) -> dict[str, Any]:
    """Generate candidates and prove that their observation state is shared."""

    raw = raw_policy_input_fn(obs, image_keys)
    processed = preprocessor(raw)
    actions: list[torch.Tensor] = []
    shared_states: list[torch.Tensor] = []
    hook_calls: list[int] = []
    elapsed: list[float] = []
    for candidate_index in range(candidate_count):
        capture.reset()
        policy.reset()
        noise = make_noise(
            policy.config, scene_seed, query_index, candidate_index, device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(dict(processed), noise=noise)
        action = postprocessor(normalized)[0].detach().float().cpu()
        elapsed.append(time.perf_counter() - started)
        if action.shape != (
            int(policy.config.chunk_size),
            ACTION_DIM,
        ):
            raise RuntimeError(
                "SmolVLA action contract changed: "
                f"{tuple(action.shape)} != {(int(policy.config.chunk_size), ACTION_DIM)}"
            )
        shared_states.append(capture.consume())
        hook_calls.append(capture.calls)
        actions.append(action)

    state_stack = torch.stack(shared_states)
    reference = state_stack[0:1].expand_as(state_stack)
    state_delta = (state_stack - reference).abs().amax(dim=1)
    if not torch.equal(state_stack, reference):
        raise RuntimeError(
            "candidate noise changed the supposed shared VLM prefix state; "
            f"max_abs_delta={float(state_delta.max())}"
        )
    action_stack = torch.stack(actions)
    action_delta = action_stack - action_stack[0:1]
    if candidate_count > 1 and not bool(
        (action_delta[1:].abs().amax(dim=(1, 2)) > 0).any()
    ):
        raise RuntimeError(
            "flow-noise candidates are identical to the deterministic actor; "
            "the candidate intervention hook is not effective"
        )
    return {
        "actions_tensor": action_stack,
        "shared_state": state_stack[0].numpy().astype(np.float16),
        "candidate_actions": action_stack.numpy().astype(np.float32),
        "candidate_names": [BASELINE_NAME]
        + [f"flow_noise_{index:03d}" for index in range(1, candidate_count)],
        "noise_seeds": np.asarray(
            [noise_seed(scene_seed, query_index, index) for index in range(candidate_count)],
            dtype=np.int64,
        ),
        "shared_state_hook_calls": np.asarray(hook_calls, dtype=np.int16),
        "shared_state_max_abs_delta": state_delta.numpy().astype(np.float32),
        "l2_from_baseline": torch.linalg.vector_norm(
            action_delta.flatten(1), dim=1
        ).numpy().astype(np.float32),
        "normalized_l2_from_baseline": torch.sqrt(
            action_delta.square().mean(dim=(1, 2))
        ).numpy().astype(np.float32),
        "max_abs_from_baseline": action_delta.abs().amax(dim=(1, 2)).numpy().astype(np.float32),
        "elapsed_seconds": np.asarray(elapsed, dtype=np.float32),
    }


def evaluate_branch(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    capture: SharedPrefixStateCapture,
    env: Any,
    seed: int,
    expected_resolved_seed: int,
    fixed_instruction: str,
    first_chunk: torch.Tensor,
    candidate_count: int,
    action_exec_steps: int,
    device: torch.device,
    image_keys: list[str],
    event_spec: dict[str, Any],
    task_name: str,
    required_pose_names: set[str],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    policy.reset()
    obs, _, resolved_seed, instruction = runtime["reset"](
        env, seed, fixed_instruction=fixed_instruction
    )
    if resolved_seed != expected_resolved_seed or instruction != fixed_instruction:
        raise RuntimeError("branch reset changed resolved seed or fixed instruction")
    subenv = env.venv.envs[0]
    task = subenv.task
    object_names, objects = runtime["discover"](task, required_pose_names)
    pre_object_poses = runtime["read_poses"](objects)
    pre_proprio = runtime["raw_state"](task)

    observational = generate_candidates(
        policy,
        preprocessor,
        postprocessor,
        capture,
        obs,
        seed,
        0,
        1,
        device,
        image_keys,
        runtime["raw_policy_input"],
    )
    pre_hidden = observational["shared_state"]
    current_hidden = pre_hidden
    chunk = first_chunk
    if chunk.shape != (int(policy.config.chunk_size), ACTION_DIM):
        raise RuntimeError(f"invalid first action chunk: {tuple(chunk.shape)}")

    trajectory_poses = [pre_object_poses]
    trajectory_proprio = [pre_proprio]
    query_steps: list[int] = []
    query_post_steps: list[int] = []
    query_hidden: list[np.ndarray] = []
    query_post_hidden: list[np.ndarray] = []
    query_actions: list[np.ndarray] = []
    query_action_mask: list[np.ndarray] = []
    success = False
    done = False
    steps = 0
    query_index = 0
    post_object_poses: np.ndarray | None = None
    post_proprio: np.ndarray | None = None
    started = time.time()
    chunk_size = int(policy.config.chunk_size)

    while steps < MAX_STEPS and not done:
        query_steps.append(steps)
        query_hidden.append(current_hidden)
        query_actions.append(chunk.numpy().astype(np.float32))
        executed = 0
        for action in chunk[:action_exec_steps]:
            obs, _, terminated, truncated, infos = env.step(
                action.reshape(1, 1, ACTION_DIM), auto_reset=False
            )
            steps += 1
            executed += 1
            trajectory_poses.append(runtime["read_poses"](objects))
            trajectory_proprio.append(runtime["raw_state"](task))
            success = success or runtime["scalar_bool"](
                infos.get("success", [False])
            )
            done = runtime["scalar_bool"](terminated) or runtime["scalar_bool"](
                truncated
            )
            if done or steps >= MAX_STEPS:
                break
        mask = np.arange(chunk_size) < executed
        validate_prefix_mask(mask, executed, chunk_size)
        query_action_mask.append(mask)
        query_post_steps.append(steps)

        next_query = generate_candidates(
            policy,
            preprocessor,
            postprocessor,
            capture,
            obs,
            seed,
            query_index + 1,
            1,
            device,
            image_keys,
            runtime["raw_policy_input"],
        )
        next_hidden = next_query["shared_state"]
        query_post_hidden.append(next_hidden)
        if query_index == 0:
            post_object_poses = runtime["read_poses"](objects)
            post_proprio = runtime["raw_state"](task)
        query_index += 1
        if done or steps >= MAX_STEPS:
            break
        current_hidden = next_hidden
        chunk = next_query["actions_tensor"][0]

    if not query_steps or post_object_poses is None or post_proprio is None:
        raise RuntimeError("branch did not execute an action query")
    raw_names, raw_steps, event_names, event_steps = runtime["derive_events"](
        np.stack(trajectory_poses), object_names, success, event_spec, task_name
    )
    first_length = int(query_post_steps[0] - query_steps[0])
    targets = derive_branch_targets(event_names, event_steps, first_length, steps)
    return {
        "instruction": instruction,
        "resolved_seed": resolved_seed,
        "object_names": object_names,
        "pre_hidden": pre_hidden,
        "post_chunk_hidden": query_post_hidden[0],
        "first_chunk_executed_length": first_length,
        "first_chunk_action_mask": query_action_mask[0],
        "post_chunk_terminal": first_length == steps,
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
        "queries": len(query_steps) - 1,
        "wall_seconds": time.time() - started,
        **targets,
    }


def save_group(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        for key in (
            "seed",
            "requested_seed",
            "resolved_seed",
            "task",
            "body",
            "policy",
            "instruction",
            "checkpoint",
            "shared_state_modeling_sha256",
            "shared_state_bridge_sha256",
            "shared_state_contract_id",
            "event_spec_sha256",
        ):
            handle.attrs[key] = record[key]
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["candidate_count"] = len(record["candidate_names"])
        handle.attrs["intervention"] = INTERVENTION
        handle.attrs["language_contract"] = LANGUAGE_CONTRACT
        handle.attrs["branch_instruction_consistent"] = bool(
            record["branch_instruction_consistent"]
        )
        handle.attrs["hidden_anchor"] = SHARED_STATE_ANCHOR
        handle.attrs["shared_state_source"] = SHARED_STATE_SOURCE
        handle.attrs["shared_state_noise_independence"] = (
            "bit_exact_across_explicit_noise_candidates"
        )
        handle.attrs["candidate_hidden_forbidden"] = True
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
        array_keys = (
            "initial_hidden",
            "pre_hidden",
            "post_chunk_hidden",
            "candidate_actions",
            "noise_seeds",
            "shared_state_hook_calls",
            "shared_state_max_abs_delta",
            "l2_from_baseline",
            "normalized_l2_from_baseline",
            "max_abs_from_baseline",
            "elapsed_seconds",
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
        )
        for key in array_keys:
            value = np.asarray(record[key])
            handle.create_dataset(
                key,
                data=value,
                compression="gzip" if value.size > 64 else None,
            )
        branches = handle.create_group("branches")
        for index, branch_record in enumerate(record["branches"]):
            branch = branches.create_group(f"candidate_{index:03d}")
            for prefix in ("raw_event", "event"):
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
                data=np.asarray(branch_record["trajectory_object_poses"], dtype=np.float32),
                compression="gzip",
            )
            branch.create_dataset(
                "proprio",
                data=np.asarray(branch_record["trajectory_proprio"], dtype=np.float32),
                compression="gzip",
            )
            for key, dtype in (
                ("query_steps", np.int32),
                ("query_post_steps", np.int32),
                ("query_hidden", np.float16),
                ("query_post_hidden", np.float16),
                ("query_actions", np.float32),
                ("query_action_mask", bool),
            ):
                value = np.asarray(branch_record[key], dtype=dtype)
                branch.create_dataset(
                    key,
                    data=value,
                    compression="gzip" if value.size > 64 else None,
                )
        handle.flush()
    os.replace(temporary, path)


def validate_group_file(
    path: Path,
    expected_seed: int,
    candidate_count: int,
    hidden_dim: int,
    chunk: int,
    *,
    expected_modeling_sha256: str | None = None,
    expected_bridge_sha256: str | None = None,
    expected_event_spec_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail-closed audit for resume and for the CPU synthetic self-test."""

    with h5py.File(path, "r") as handle:
        expected_attrs = {
            "schema_version": SCHEMA_VERSION,
            "seed": expected_seed,
            "candidate_count": candidate_count,
            "body": BODY,
            "policy": POLICY,
            "intervention": INTERVENTION,
            "language_contract": LANGUAGE_CONTRACT,
            "hidden_anchor": SHARED_STATE_ANCHOR,
            "shared_state_source": SHARED_STATE_SOURCE,
            "shared_state_noise_independence": (
                "bit_exact_across_explicit_noise_candidates"
            ),
            "candidate_hidden_forbidden": True,
            "post_query_action_contract": POST_QUERY_ACTION_CONTRACT,
            "branch_instruction_consistent": True,
        }
        for key, expected in expected_attrs.items():
            actual = handle.attrs.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"schema-v5 attribute {key} mismatch in {path}: {actual!r}"
                )
        for digest_key in (
            "shared_state_modeling_sha256",
            "shared_state_bridge_sha256",
            "shared_state_contract_id",
            "event_spec_sha256",
        ):
            digest = str(handle.attrs.get(digest_key, ""))
            if len(digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in digest
            ):
                raise RuntimeError(f"invalid source binding {digest_key} in {path}")
        recorded_state_contract = shared_state_contract(
            hidden_dim=hidden_dim,
            modeling_sha256=str(handle.attrs["shared_state_modeling_sha256"]),
            bridge_sha256=str(handle.attrs["shared_state_bridge_sha256"]),
        )
        if handle.attrs["shared_state_contract_id"] != recorded_state_contract[
            "calibration_id"
        ]:
            raise RuntimeError("shared state contract id does not bind its source hashes")
        for attr_name, expected in (
            ("shared_state_modeling_sha256", expected_modeling_sha256),
            ("shared_state_bridge_sha256", expected_bridge_sha256),
            ("event_spec_sha256", expected_event_spec_sha256),
        ):
            if expected is not None and str(handle.attrs[attr_name]) != str(expected):
                raise RuntimeError(
                    f"completed group {attr_name} differs from current runtime"
                )
        if "candidate_hidden" in handle:
            raise RuntimeError("candidate-specific action-expert hidden is forbidden")
        names = decode_strings(handle["candidate_names"][:])
        if len(names) != candidate_count or names[0] != BASELINE_NAME:
            raise RuntimeError("candidate zero must be the deterministic fallback")
        if len(set(names)) != candidate_count:
            raise RuntimeError("candidate names must be unique")
        expected_shapes = {
            "initial_hidden": (hidden_dim,),
            "pre_hidden": (candidate_count, hidden_dim),
            "post_chunk_hidden": (candidate_count, hidden_dim),
            "candidate_actions": (candidate_count, chunk, ACTION_DIM),
            "noise_seeds": (candidate_count,),
            "shared_state_hook_calls": (candidate_count,),
            "shared_state_max_abs_delta": (candidate_count,),
            "first_chunk_executed_length": (candidate_count,),
            "first_chunk_action_mask": (candidate_count, chunk),
            "success": (candidate_count,),
            "steps": (candidate_count,),
            "queries": (candidate_count,),
        }
        for key, shape in expected_shapes.items():
            if key not in handle or handle[key].shape != shape:
                raise RuntimeError(f"invalid {key} shape in {path}")
        finite_keys = (
            "initial_hidden",
            "pre_hidden",
            "post_chunk_hidden",
            "candidate_actions",
            "shared_state_max_abs_delta",
            "normalized_l2_from_baseline",
            "pre_object_poses",
            "post_object_poses",
            "pre_proprio",
            "post_proprio",
            "duration",
        )
        if any(not np.isfinite(handle[key][:]).all() for key in finite_keys):
            raise RuntimeError("non-finite schema-v5 dense data")
        noise_seeds = handle["noise_seeds"][:].astype(np.int64)
        if len(np.unique(noise_seeds)) != candidate_count:
            raise RuntimeError("candidate flow-noise seeds are not unique")
        candidate_actions = handle["candidate_actions"][:]
        if not np.any(candidate_actions[1:] != candidate_actions[0]):
            raise RuntimeError("candidate action intervention has no effect")
        initial = handle["initial_hidden"][:]
        if not np.array_equal(
            handle["pre_hidden"][:],
            np.repeat(initial[None], candidate_count, axis=0),
        ):
            raise RuntimeError("candidate branches do not share the initial VLM state")
        if not np.array_equal(
            handle["shared_state_hook_calls"][:],
            np.ones(candidate_count, dtype=np.int16),
        ) or not np.array_equal(
            handle["shared_state_max_abs_delta"][:],
            np.zeros(candidate_count, dtype=np.float32),
        ):
            raise RuntimeError("shared-prefix noise-independence proof failed")
        object_count = len(handle["object_names"])
        proprio_dim = handle["pre_proprio"].shape[-1]
        if handle["pre_object_poses"].shape != (candidate_count, object_count, 7):
            raise RuntimeError("pre-object pose shape is invalid")
        if not np.array_equal(
            handle["pre_object_poses"][:],
            np.repeat(handle["pre_object_poses"][0:1], candidate_count, axis=0),
        ) or not np.array_equal(
            handle["pre_proprio"][:],
            np.repeat(handle["pre_proprio"][0:1], candidate_count, axis=0),
        ):
            raise RuntimeError("candidate branches do not share simulator pre-state")
        terminal_steps = handle["steps"][:].astype(np.int32)
        first_lengths = handle["first_chunk_executed_length"][:].astype(np.int32)
        first_masks = handle["first_chunk_action_mask"][:].astype(bool)
        for mask, length in zip(first_masks, first_lengths):
            validate_prefix_mask(mask, int(length), chunk)
        if np.any(first_lengths > terminal_steps) or not np.array_equal(
            handle["post_chunk_terminal"][:].astype(bool),
            first_lengths == terminal_steps,
        ):
            raise RuntimeError("first-query terminal contract is invalid")
        if "branches" not in handle or len(handle["branches"]) != candidate_count:
            raise RuntimeError("candidate trajectory groups are incomplete")
        for index in range(candidate_count):
            branch = handle["branches"][f"candidate_{index:03d}"]
            terminal = int(terminal_steps[index])
            poses = branch["object_poses"][:]
            proprio = branch["proprio"][:]
            if poses.shape != (terminal + 1, object_count, 7):
                raise RuntimeError("branch object trajectory is not step aligned")
            if proprio.shape != (terminal + 1, proprio_dim):
                raise RuntimeError("branch proprio trajectory is not step aligned")
            query_steps = branch["query_steps"][:].astype(np.int32)
            query_post = branch["query_post_steps"][:].astype(np.int32)
            query_count = len(query_steps)
            expected_query_shapes = {
                "query_post_steps": (query_count,),
                "query_hidden": (query_count, hidden_dim),
                "query_post_hidden": (query_count, hidden_dim),
                "query_actions": (query_count, chunk, ACTION_DIM),
                "query_action_mask": (query_count, chunk),
            }
            for key, shape in expected_query_shapes.items():
                if branch[key].shape != shape:
                    raise RuntimeError(f"invalid branch {key} shape")
            if (
                query_count < 1
                or int(query_steps[0]) != 0
                or int(query_post[-1]) != terminal
                or not np.array_equal(query_steps[1:], query_post[:-1])
            ):
                raise RuntimeError("continuation query boundaries are not contiguous")
            lengths = query_post - query_steps
            masks = branch["query_action_mask"][:].astype(bool)
            for mask, length in zip(masks, lengths):
                validate_prefix_mask(mask, int(length), chunk)
            query_hidden = branch["query_hidden"][:]
            query_post_hidden = branch["query_post_hidden"][:]
            if not np.array_equal(query_hidden[0], handle["pre_hidden"][index]):
                raise RuntimeError("initial branch/query shared state mismatch")
            if query_count > 1 and not np.array_equal(
                query_hidden[1:], query_post_hidden[:-1]
            ):
                raise RuntimeError("shared-state continuation chain is broken")
            if not np.array_equal(
                query_post_hidden[0], handle["post_chunk_hidden"][index]
            ) or not np.array_equal(
                branch["query_actions"][0], handle["candidate_actions"][index]
            ) or not np.array_equal(masks[0], first_masks[index]):
                raise RuntimeError("root/first-query compatibility failed")
            if int(query_post[0]) != int(first_lengths[index]) or int(
                handle["queries"][index]
            ) != query_count - 1:
                raise RuntimeError("first-query boundary/count contract failed")
            post_step = int(first_lengths[index])
            if not np.array_equal(
                poses[0], handle["pre_object_poses"][index]
            ) or not np.array_equal(
                poses[post_step], handle["post_object_poses"][index]
            ):
                raise RuntimeError("object trajectory boundaries mismatch")
            if not np.array_equal(
                proprio[0], handle["pre_proprio"][index]
            ) or not np.array_equal(
                proprio[post_step], handle["post_proprio"][index]
            ):
                raise RuntimeError("proprio trajectory boundaries mismatch")
            event_names = decode_strings(branch["event_names"][:])
            event_steps = branch["event_steps"][:].astype(np.int32)
            if ("eK" in event_names) != bool(handle["success"][index]):
                raise RuntimeError("terminal success/event contract failed")
            targets = derive_branch_targets(
                event_names, event_steps, post_step, terminal
            )
            for key in (
                "pre_event_id",
                "post_event_id",
                "next_event_id",
                "post_chunk_step",
                "duration",
                "duration_observed",
                "duration_censored",
            ):
                if handle[key][index].item() != targets[key]:
                    raise RuntimeError(f"dense event label {key} is inconsistent")
        return {
            "seed": int(handle.attrs["seed"]),
            "resolved_seed": int(handle.attrs["resolved_seed"]),
            "success": handle["success"][:].astype(bool).tolist(),
            "steps": terminal_steps.astype(int).tolist(),
            "query_transitions": int(
                (handle["queries"][:].astype(np.int64) + 1).sum()
            ),
        }


def _synthetic_record(hidden_dim: int = 8, chunk: int = 4) -> dict[str, Any]:
    candidate_count = 2
    initial = np.arange(hidden_dim, dtype=np.float16)
    steps = np.asarray([5, 3], dtype=np.int32)
    first_lengths = np.asarray([4, 3], dtype=np.int32)
    branches: list[dict[str, Any]] = []
    event_specs = [
        (["e0", "e12", "e4", "eK"], [0, 1, 3, 5]),
        (["e0", "e12", "e4"], [0, 1, 3]),
    ]
    for index, (terminal, first_length) in enumerate(zip(steps, first_lengths)):
        query_steps = np.asarray([0, 4], dtype=np.int32) if index == 0 else np.asarray([0], dtype=np.int32)
        query_post = np.asarray([4, 5], dtype=np.int32) if index == 0 else np.asarray([3], dtype=np.int32)
        query_hidden = np.stack(
            [initial + offset for offset in range(len(query_steps))]
        ).astype(np.float16)
        query_post_hidden = np.stack(
            [initial + offset + 1 for offset in range(len(query_steps))]
        ).astype(np.float16)
        actions = np.full((len(query_steps), chunk, ACTION_DIM), index, dtype=np.float32)
        masks = np.stack([np.arange(chunk) < int(b - a) for a, b in zip(query_steps, query_post)])
        names, event_steps = event_specs[index]
        trajectory_poses = np.zeros((terminal + 1, 2, 7), dtype=np.float32)
        trajectory_proprio = np.zeros(
            (terminal + 1, ACTION_DIM), dtype=np.float32
        )
        trajectory_poses[1:] = index + 1
        trajectory_proprio[1:] = index + 1
        branches.append(
            {
                "raw_event_names": names,
                "raw_event_steps": np.asarray(event_steps, dtype=np.int32),
                "event_names": names,
                "event_steps": np.asarray(event_steps, dtype=np.int32),
                "trajectory_object_poses": trajectory_poses,
                "trajectory_proprio": trajectory_proprio,
                "query_steps": query_steps,
                "query_post_steps": query_post,
                "query_hidden": query_hidden,
                "query_post_hidden": query_post_hidden,
                "query_actions": actions,
                "query_action_mask": masks,
            }
        )
    labels = [
        derive_branch_targets(
            branch["event_names"], branch["event_steps"], int(first_lengths[index]), int(steps[index])
        )
        for index, branch in enumerate(branches)
    ]
    record: dict[str, Any] = {
        "seed": 123,
        "requested_seed": 123,
        "resolved_seed": 123,
        "task": DEFAULT_TASK,
        "body": BODY,
        "policy": POLICY,
        "instruction": "self test",
        "checkpoint": "/synthetic/smolvla",
        "shared_state_modeling_sha256": "0" * 64,
        "shared_state_bridge_sha256": "1" * 64,
        "shared_state_contract_id": shared_state_contract(
            hidden_dim=hidden_dim,
            modeling_sha256="0" * 64,
            bridge_sha256="1" * 64,
        )["calibration_id"],
        "event_spec_sha256": "2" * 64,
        "branch_instruction_consistent": True,
        "candidate_names": [BASELINE_NAME, "flow_noise_001"],
        "object_names": ["can", "pot"],
        "initial_hidden": initial,
        "pre_hidden": np.repeat(initial[None], candidate_count, axis=0),
        "post_chunk_hidden": np.stack(
            [branch["query_post_hidden"][0] for branch in branches]
        ),
        "candidate_actions": np.stack(
            [branch["query_actions"][0] for branch in branches]
        ),
        "noise_seeds": np.asarray([1, 2], dtype=np.int64),
        "shared_state_hook_calls": np.ones(candidate_count, dtype=np.int16),
        "shared_state_max_abs_delta": np.zeros(candidate_count, dtype=np.float32),
        "l2_from_baseline": np.asarray([0.0, 1.0], dtype=np.float32),
        "normalized_l2_from_baseline": np.asarray([0.0, 1.0], dtype=np.float32),
        "max_abs_from_baseline": np.asarray([0.0, 1.0], dtype=np.float32),
        "elapsed_seconds": np.ones(candidate_count, dtype=np.float32),
        "first_chunk_executed_length": first_lengths,
        "first_chunk_action_mask": np.stack(
            [np.arange(chunk) < length for length in first_lengths]
        ),
        "post_chunk_terminal": first_lengths == steps,
        "pre_object_poses": np.stack(
            [branch["trajectory_object_poses"][0] for branch in branches]
        ),
        "post_object_poses": np.stack(
            [branch["trajectory_object_poses"][length] for branch, length in zip(branches, first_lengths)]
        ),
        "pre_proprio": np.stack(
            [branch["trajectory_proprio"][0] for branch in branches]
        ),
        "post_proprio": np.stack(
            [branch["trajectory_proprio"][length] for branch, length in zip(branches, first_lengths)]
        ),
        "success": np.asarray([True, False]),
        "steps": steps,
        "queries": np.asarray([1, 0], dtype=np.int32),
        "wall_seconds": np.ones(candidate_count, dtype=np.float32),
        "branches": branches,
    }
    for key in labels[0]:
        record[key] = np.asarray([label[key] for label in labels])
    return record


def run_self_test() -> None:
    hidden_dim, chunk = 8, 4
    record = _synthetic_record(hidden_dim, chunk)
    with tempfile.TemporaryDirectory(prefix="smolvla_schema_v5_selftest_") as directory:
        path = Path(directory) / "group.hdf5"
        save_group(path, record)
        result = validate_group_file(path, 123, 2, hidden_dim, chunk)
        assert result["query_transitions"] == 3
    print(
        "SELF_TEST_COMPLETE="
        + json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "shared_state_anchor": SHARED_STATE_ANCHOR,
                "passed": True,
                "gpu_hook_verified": False,
            },
            sort_keys=True,
        )
    )


def select_seeds(args: argparse.Namespace, seeds_path: Path, load_official: Any) -> list[int]:
    if args.seeds_file is not None:
        payload = json.loads(args.seeds_file.read_text(encoding="utf-8"))
        if args.seeds_key:
            payload = payload[args.seeds_key]
        if not isinstance(payload, list):
            raise ValueError("seed-file selection must resolve to a JSON list")
        seeds = [int(row["seed"] if isinstance(row, Mapping) else row) for row in payload]
    elif args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
    else:
        raise ValueError("explicit --seeds or --seeds-file selection is required")
    official = set(load_official(seeds_path, args.task, 150, 0))
    invalid = sorted(set(seeds) - official)
    if invalid:
        raise ValueError(f"non-official seeds requested: {invalid}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed selection is empty or contains duplicates")
    return seeds


def main() -> None:
    if "--self-test" in sys.argv[1:]:
        run_self_test()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--seeds-key", choices=["train", "validation", "test"])
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--action-exec-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.seeds is not None and args.seeds_file is not None:
        parser.error("--seeds and --seeds-file are mutually exclusive")
    if args.seeds_key and args.seeds_file is None:
        parser.error("--seeds-key requires --seeds-file")
    if args.task != DEFAULT_TASK:
        parser.error(f"current audited collector supports only {DEFAULT_TASK!r}")
    if args.candidate_count < 2:
        parser.error("--candidate-count must be at least two")
    for path in (
        args.model_path,
        args.vlm_metadata_path,
        args.rlinf_root,
        args.robotwin_root,
        args.robotwin_code,
        args.event_spec,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "real SmolVLA collection requires the 4090 runtime; use --self-test on CPU"
        )

    random.seed(20260827)
    np.random.seed(20260827)
    torch.manual_seed(20260827)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    from collect_openvla_etsf_rollouts import (
        derive_events,
        discover_pose_objects,
        raw_state,
        read_poses,
        scalar_bool,
    )
    from collect_smolvla_etsf_candidate_branches import (
        environment_config,
        load_official_seeds,
        raw_policy_input,
        reset_with_resolved_seed,
    )

    event_spec = json.loads(args.event_spec.read_text(encoding="utf-8"))
    if args.task not in event_spec.get("calibration", {}) or args.task not in event_spec.get(
        "chains", {}
    ):
        raise RuntimeError("event specification lacks the requested task")
    calibration = event_spec["calibration"][args.task]
    chain = event_spec["chains"][args.task]
    if chain.get("merge_e1_e2") is not True or tuple(
        chain.get("chain", ())
    ) != EVENT_VOCAB:
        raise RuntimeError(
            "SmolVLA schema-v5 currently requires the frozen "
            "e0/e12/e3/e4/eK canonical event chain"
        )
    required_pose_names = {str(calibration["moving"])}
    if calibration.get("anchor"):
        required_pose_names.add(str(calibration["anchor"]))

    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    seeds = select_seeds(args, seeds_path, load_official_seeds)
    device = torch.device("cuda:0")
    config = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
    config.device = str(device)
    config.vlm_model_name = str(args.vlm_metadata_path)
    config.load_vlm_weights = False
    if int(config.chunk_size) != AUDITED_CHUNK:
        raise RuntimeError(
            f"SmolVLA chunk changed: {config.chunk_size} != {AUDITED_CHUNK}"
        )
    if config.action_feature is None or int(config.action_feature.shape[0]) != ACTION_DIM:
        raise RuntimeError("checkpoint is not the audited 14-D ALOHA SmolVLA actor")
    action_exec_steps = args.action_exec_steps or int(config.n_action_steps)
    if not 1 <= action_exec_steps <= int(config.chunk_size):
        parser.error("action-exec-steps lies outside the action chunk")
    policy = SmolVLAPolicy.from_pretrained(
        args.model_path,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.model_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_metadata_path)},
        },
    )
    capture = resolve_shared_prefix_capture(policy)
    norm = policy.model.vlm_with_expert.get_vlm_model().text_model.norm
    hidden_dim = int(norm.weight.numel())
    if hidden_dim != 960:
        raise RuntimeError(f"audited SmolVLM state dimension changed: {hidden_dim} != 960")
    modeling_path = Path(inspect.getsourcefile(policy.model.__class__) or "")
    if not modeling_path.is_file():
        raise RuntimeError("cannot bind collection to the loaded SmolVLA source")
    modeling_digest = sha256(modeling_path)
    bridge_path = Path(
        inspect.getsourcefile(policy.model.vlm_with_expert.__class__) or ""
    )
    if not bridge_path.is_file():
        raise RuntimeError("cannot bind collection to the loaded VLM/expert bridge source")
    bridge_digest = sha256(bridge_path)
    state_contract = shared_state_contract(
        hidden_dim=hidden_dim,
        modeling_sha256=modeling_digest,
        bridge_sha256=bridge_digest,
    )
    event_spec_digest = sha256(args.event_spec)
    image_keys = list(policy.config.image_features)
    runtime = {
        "reset": reset_with_resolved_seed,
        "discover": discover_pose_objects,
        "read_poses": read_poses,
        "raw_state": raw_state,
        "scalar_bool": scalar_bool,
        "derive_events": derive_events,
        "raw_policy_input": raw_policy_input,
    }
    env = RoboTwinEnv(
        cfg=environment_config(
            args.robotwin_root, seeds_path, args.task, len(seeds), MAX_STEPS
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output / "groups"
    groups_dir.mkdir(exist_ok=True)
    resume_contract = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "body": BODY,
        "policy": POLICY,
        "model_path": str(args.model_path),
        "requested_seeds": seeds,
        "candidate_count": args.candidate_count,
        "action_exec_steps": action_exec_steps,
        "event_spec_sha256": event_spec_digest,
        "shared_state_modeling_sha256": modeling_digest,
        "shared_state_bridge_sha256": bridge_digest,
        "shared_state_contract_id": state_contract["calibration_id"],
        "hidden_anchor": SHARED_STATE_ANCHOR,
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in resume_contract.items():
            if previous.get(key) != expected:
                raise RuntimeError(f"resume contract mismatch for {key}")
    manifest: dict[str, Any] = {
        **resume_contract,
        "status": "collecting",
        "collector_seed": 20260827,
        "checkpoint": str(args.model_path),
        "vlm_metadata_path": str(args.vlm_metadata_path),
        "modeling_source": str(modeling_path),
        "bridge_source": str(bridge_path),
        "candidate_generator": "native_smolvla_flow_matching_explicit_fixed_noise",
        "baseline_contract": "candidate_0_fixed_by_scene_seed_and_query_index",
        "intervention": INTERVENTION,
        "language_contract": LANGUAGE_CONTRACT,
        "event_vocab": list(EVENT_VOCAB),
        "event_spec": str(args.event_spec),
        "hidden_dim": hidden_dim,
        "shared_state_contract": {
            **state_contract,
            "position": "final_unpadded_prefix_token_projected_robot_state",
            "flow_noise_dependency": "none_before_denoise_loop",
            "candidate_equality": "bit_exact_at_each_group_intervention_query",
            "candidate_specific_expert_hidden": "forbidden",
        },
        "action_dim": ACTION_DIM,
        "action_chunk": int(config.chunk_size),
        "max_steps": MAX_STEPS,
        "trajectory_contract": {
            "object_poses": "per_step_including_reset_and_terminal",
            "proprio": "per_step_including_reset_and_terminal",
            "purpose": "dynamic_predicates_failure_and_recovery_labels",
        },
        "continuation_query_contract": {
            "query_hidden": SHARED_STATE_ANCHOR,
            "query_post_hidden": "same_anchor_after_executed_chunk_including_terminal",
            "query_actions": f"padded_to_{int(config.chunk_size)}_native_actions",
            "query_action_mask": "contiguous_executed_prefix",
            "post_query_action": POST_QUERY_ACTION_CONTRACT,
        },
        "gpu_hook_smoke": "verified_at_each_group_intervention_query",
        "task_success_claimed": False,
        "groups": [],
    }
    resolved_seen: set[int] = set()
    try:
        for group_index, seed in enumerate(seeds):
            path = groups_dir / f"group_{group_index:03d}_seed_{seed}.hdf5"
            if path.exists() and not args.overwrite:
                existing = validate_group_file(
                    path,
                    seed,
                    args.candidate_count,
                    hidden_dim,
                    int(config.chunk_size),
                    expected_modeling_sha256=modeling_digest,
                    expected_bridge_sha256=bridge_digest,
                    expected_event_spec_sha256=event_spec_digest,
                )
                if existing["resolved_seed"] in resolved_seen:
                    raise RuntimeError("duplicate resolved seed during resume")
                resolved_seen.add(existing["resolved_seed"])
                manifest["groups"].append(
                    {"index": group_index, "path": path.name, "status": "existing", **existing}
                )
                continue

            obs, _, resolved_seed, instruction = reset_with_resolved_seed(env, seed)
            if resolved_seed in resolved_seen:
                raise RuntimeError("two requested seeds resolved to the same scene")
            resolved_seen.add(resolved_seed)
            generated = generate_candidates(
                policy,
                preprocessor,
                postprocessor,
                capture,
                obs,
                seed,
                0,
                args.candidate_count,
                device,
                image_keys,
                raw_policy_input,
            )
            initial_hidden = generated["shared_state"]
            chunks = generated.pop("actions_tensor")
            generated.pop("shared_state")
            outcomes = [
                evaluate_branch(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    capture=capture,
                    env=env,
                    seed=seed,
                    expected_resolved_seed=resolved_seed,
                    fixed_instruction=instruction,
                    first_chunk=chunk,
                    candidate_count=args.candidate_count,
                    action_exec_steps=action_exec_steps,
                    device=device,
                    image_keys=image_keys,
                    event_spec=event_spec,
                    task_name=args.task,
                    required_pose_names=required_pose_names,
                    runtime=runtime,
                )
                for chunk in chunks
            ]
            if any(outcome["instruction"] != instruction for outcome in outcomes):
                raise RuntimeError("candidate branches changed the fixed instruction")
            object_names = outcomes[0]["object_names"]
            if any(outcome["object_names"] != object_names for outcome in outcomes):
                raise RuntimeError("tracked object order changed across branches")
            pre_hidden = np.stack([outcome["pre_hidden"] for outcome in outcomes])
            if not np.array_equal(
                pre_hidden,
                np.repeat(initial_hidden[None], args.candidate_count, axis=0),
            ):
                raise RuntimeError("branch reset changed the shared observation state")
            pre_poses = outcomes[0]["pre_object_poses"]
            pre_proprio = outcomes[0]["pre_proprio"]
            if any(
                not np.array_equal(outcome["pre_object_poses"], pre_poses)
                or not np.array_equal(outcome["pre_proprio"], pre_proprio)
                for outcome in outcomes[1:]
            ):
                raise RuntimeError("branch reset changed simulator pre-state")
            record: dict[str, Any] = {
                **generated,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": resolved_seed,
                "task": args.task,
                "body": BODY,
                "policy": POLICY,
                "instruction": instruction,
                "checkpoint": str(args.model_path),
                "shared_state_modeling_sha256": modeling_digest,
                "shared_state_bridge_sha256": bridge_digest,
                "shared_state_contract_id": state_contract["calibration_id"],
                "event_spec_sha256": event_spec_digest,
                "branch_instruction_consistent": True,
                "object_names": object_names,
                "initial_hidden": initial_hidden,
                "pre_hidden": pre_hidden,
                "post_chunk_hidden": np.stack(
                    [outcome["post_chunk_hidden"] for outcome in outcomes]
                ),
                "branches": outcomes,
            }
            for key in (
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
            ):
                record[key] = np.asarray([outcome[key] for outcome in outcomes])
            save_group(path, record)
            existing = validate_group_file(
                path,
                seed,
                args.candidate_count,
                hidden_dim,
                int(config.chunk_size),
                expected_modeling_sha256=modeling_digest,
                expected_bridge_sha256=bridge_digest,
                expected_event_spec_sha256=event_spec_digest,
            )
            item = {
                "index": group_index,
                "path": path.name,
                "status": "collected",
                **existing,
            }
            manifest["groups"].append(item)
            manifest["completed"] = len(manifest["groups"])
            atomic_json(manifest_path, manifest)
            print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)
    finally:
        capture.close()
        env.venv.close(clear_cache=False)

    manifest["status"] = "complete"
    manifest["completed"] = len(manifest["groups"])
    manifest["resolved_seeds"] = [row["resolved_seed"] for row in manifest["groups"]]
    successes = np.asarray([row["success"] for row in manifest["groups"]], dtype=np.int64)
    manifest["candidate_successes"] = successes.sum(axis=0).tolist()
    manifest["oracle_successes"] = int(successes.max(axis=1).sum())
    manifest["groups_with_outcome_variation"] = int(
        sum(len(set(row["success"])) > 1 for row in manifest["groups"])
    )
    atomic_json(manifest_path, manifest)
    print(
        "COLLECTION_COMPLETE="
        + json.dumps(
            {
                "completed": manifest["completed"],
                "candidate_successes": manifest["candidate_successes"],
                "oracle_successes": manifest["oracle_successes"],
                "task_success_claimed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
