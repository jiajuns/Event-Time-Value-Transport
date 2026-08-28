#!/usr/bin/env python3
"""Development-only schema-v6 dense SmolVLA->Piper branch collector.

The module deliberately has no autonomous CLI execution path.  A launcher must
first provide an independently frozen collection authority and call
``bind_r6f_collection_runtime``.  This prevents the R6f four-step operational
smoke from being misrepresented as authority for a larger data collection.

At every live query the formal checkpoint preprocessor runs once, four native
fixed-noise candidates are generated, and their contextualized 960-D prefix
must be bit-identical.  Actions are copied through the named Aloha/Piper
registry.  Only finite/in-bounds first rows are eligible.  The root creates a
branch group only when at least two candidates are eligible; continuation uses
the lowest eligible *original* candidate index with H=1 at every query.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from collect_smolvla_etsf_event_branches import (
    EVENT_VOCAB,
    SHARED_STATE_ANCHOR,
    derive_branch_targets,
    event_id_at_step,
)
from etsf_schema6_pose_quality import (
    GROUP_NAME as POSE_QUALITY_GROUP,
    derive_interval_supervision_mask,
    derive_pose_quality,
    registry_sha256,
    spec_sha256,
    validate_pose_quality_v6,
    validate_registry,
    validate_spec,
    write_pose_quality_v6,
)
from run_smolvla_piper_r6d_direct_actor_smoke import (
    ACTION_DIM,
    ACTION_EXEC_STEPS,
    ACTOR_ID,
    CHUNK_SIZE,
    EXPECTED_IMAGE_KEYS,
    INSTRUCTION,
    PREFIX_DIM,
    REQUESTED_DEVELOPMENT_SEED,
    _cpu_numpy,
    array_sha256,
    canonical_sha256,
    live_policy_input,
)
from run_smolvla_piper_r6f_feasibility_smoke import (
    CANDIDATE_COUNT,
    FeasibilitySmokeError,
    assess_candidate_first_action,
    fixed_candidate_noise,
)
from verify_smolvla_piper_zero_shot_preflight import ALOHA_FEATURE_NAMES, PIPER_ACTION_SLOTS


SCHEMA_VERSION = 6
FORMAT = "etsf_smolvla_piper_dense_event_branches_schema6_v1"
EVIDENCE_SCOPE = "nonfresh_piper_development_only"
INTERVENTION = "one_root_H1_original_native_candidate_then_lowest_legal_feasibility_H1_continuation"
BASELINE_NAME = "lowest_legal_feasibility_baseline"
MAX_AUTHORIZED_BY_R6F = 4


class DenseBranchContractError(RuntimeError):
    """A schema, runtime, reset, prefix, or feasibility contract violation."""


def bind_r6f_collection_runtime(path: Path) -> dict[str, Any]:
    """Revalidate every R6e/R6f gate; this does not enlarge R6f authority."""

    from freeze_smolvla_piper_schema6_development_collection import (
        load_r6f_lineage_for_collection,
    )

    prereg, r6e, r6c, r6d, seed = load_r6f_lineage_for_collection(path)
    capability = prereg.get("capability_contract", {})
    if capability != {
        "simulation_execution_authorized": True,
        "real_robot_execution_authorized": False,
        "fresh_inputs_allowed": False,
        "fresh_trajectory_or_label_opened": False,
        "performance_evaluation_authorized": False,
        "task_success_claim_authorized": False,
        "transfer_claim_authorized": False,
    }:
        raise DenseBranchContractError("R6f Fresh/no-claim capability changed")
    inherited = prereg["inherited_R6e_contract"]
    if inherited.get("runtime_roots") != r6e.get("runtime_roots"):
        raise DenseBranchContractError("R6e runtime roots changed during R6f binding")
    if prereg.get("explicit_instruction") != INSTRUCTION:
        raise DenseBranchContractError("R6f explicit instruction changed")
    return {
        "r6f_preregistration": {
            "path": str(Path(path).resolve()),
            "logical_sha256": prereg["preregistration_sha256"],
        },
        "runtime_roots": dict(r6e["runtime_roots"]),
        "runtime_source_artifacts": dict(r6e["runtime_source_artifacts"]),
        "r6c_binding": dict(r6e["r6c_binding"]),
        "r6d_binding": dict(r6e["r6d_binding"]),
        "development_seed": dict(seed),
        "model_bundle_sha256": r6e["model_bundle_sha256"],
        "vlm_metadata_bundle_sha256": r6e["vlm_metadata_bundle_sha256"],
        "explicit_instruction": INSTRUCTION,
        "fresh_inputs_allowed": False,
        "collection_execution_authorized_by_this_binding": False,
        "maximum_existing_operational_smoke_steps": MAX_AUTHORIZED_BY_R6F,
    }


def native_action_sha256(value: Any) -> str:
    """Digest the postprocessed native Aloha chunk before named mapping."""

    array = np.asarray(value)
    if array.shape != (CHUNK_SIZE, ACTION_DIM) or array.dtype.kind != "f":
        raise DenseBranchContractError("native action chunk must be float [50,14]")
    return array_sha256(array)


def explicit_named_map_chunk_allowing_rejected_values(value: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Map all rows by name; finite/bounds are assessed only on the H=1 row."""

    source = np.asarray(value)
    if source.shape != (CHUNK_SIZE, ACTION_DIM) or source.dtype.kind != "f":
        raise DenseBranchContractError("native action chunk must be float [50,14]")
    source_by_name = {name: index for index, name in enumerate(ALOHA_FEATURE_NAMES)}
    mapped = np.empty_like(source)
    mapping: list[dict[str, Any]] = []
    for target_index, slot in enumerate(PIPER_ACTION_SLOTS):
        source_index = source_by_name[slot.source_feature_name]
        mapped[:, target_index] = source[:, source_index]
        mapping.append({
            "source_index": source_index,
            "source_feature_name": slot.source_feature_name,
            "target_index": target_index,
            "target_joint_name": slot.target_joint_name,
            "side": slot.side,
            "ordinal": slot.ordinal,
            "numeric_transform": slot.numeric_transform,
        })
    return np.ascontiguousarray(mapped), {
        "mode": "explicit_named_ordinal_angle_preserving_mapping",
        "mapping": mapping,
        "derived_from_equal_14d_width": False,
        "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False,
        "clipping_or_scaling_applied": False,
    }


def generate_candidate_query(
    *,
    policy: Any,
    preprocessor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    postprocessor: Callable[[Any], Any],
    capture: Any,
    observation: Mapping[str, Any],
    bounds: Sequence[Sequence[float]],
    device: Any,
    scene_seed: int,
    query_index: int,
    noise_factory: Callable[[Any, int, int, int, Any], Any] = fixed_candidate_noise,
) -> dict[str, Any]:
    """Generate four candidates from one processed observation, without step."""

    if tuple(getattr(policy.config, "image_features", ())) != EXPECTED_IMAGE_KEYS:
        raise DenseBranchContractError("formal checkpoint camera registry changed")
    raw = live_policy_input(observation, EXPECTED_IMAGE_KEYS)
    interface = dict(raw.pop("_interface"))
    if "drive_target" in interface:
        interface["drive_target"] = _cpu_numpy(interface["drive_target"], dtype=np.float32).tolist()
    processed = preprocessor(raw)
    state = _cpu_numpy(processed.get("observation.state"), dtype=np.float32)
    if state.shape == (ACTION_DIM,):
        state = state[None]
    if state.shape != (1, ACTION_DIM) or not np.all(np.isfinite(state)):
        raise DenseBranchContractError("processed state must be finite [1,14]")
    state_hash = array_sha256(state[0])
    prefixes: list[np.ndarray] = []
    native: list[np.ndarray] = []
    mapped: list[np.ndarray] = []
    feasibility: list[dict[str, Any]] = []
    noise_digests: list[str] = []
    mapping_contract: dict[str, Any] | None = None
    for candidate_index in range(CANDIDATE_COUNT):
        policy.reset()
        capture.reset()
        noise = noise_factory(policy.config, scene_seed, query_index, candidate_index, device)
        noise_digests.append(array_sha256(_cpu_numpy(noise, dtype=np.float32)))
        normalized = policy.predict_action_chunk(dict(processed), noise=noise)
        chunk = _cpu_numpy(postprocessor(normalized), dtype=np.float32)
        if chunk.shape == (1, CHUNK_SIZE, ACTION_DIM):
            chunk = chunk[0]
        if chunk.shape != (CHUNK_SIZE, ACTION_DIM):
            raise DenseBranchContractError("postprocessor output must be [50,14]")
        prefix = _cpu_numpy(capture.consume(), dtype=np.float32)
        if prefix.shape != (PREFIX_DIM,) or not np.all(np.isfinite(prefix)):
            raise DenseBranchContractError("shared prefix must be finite [960]")
        mapped_chunk, contract = explicit_named_map_chunk_allowing_rejected_values(chunk)
        if mapping_contract is None:
            mapping_contract = contract
        elif mapping_contract != contract:
            raise DenseBranchContractError("named mapping changed between candidates")
        prefixes.append(prefix)
        native.append(chunk)
        mapped.append(mapped_chunk)
        feasibility.append(assess_candidate_first_action(
            mapped_chunk[0], bounds, query_index=query_index, candidate_index=candidate_index
        ))
        current_state = _cpu_numpy(processed["observation.state"], dtype=np.float32).reshape(-1)
        if array_sha256(current_state) != state_hash:
            raise DenseBranchContractError("processed observation mutated between candidates")
    reference = prefixes[0]
    if any(not np.array_equal(prefix, reference) for prefix in prefixes[1:]):
        raise DenseBranchContractError("candidate prefix drift at one live query")
    mask = np.asarray([bool(item["accepted"]) for item in feasibility], dtype=bool)
    legal = np.flatnonzero(mask).astype(np.int16)
    return {
        "query_index": int(query_index),
        "hidden": reference,
        "processed_state": state[0],
        "input_interface": interface,
        "native_action_sha256": [native_action_sha256(chunk) for chunk in native],
        "native_actions": np.stack(native).astype(np.float32),
        "mapped_actions": np.stack(mapped).astype(np.float32),
        "noise_sha256": noise_digests,
        "candidate_prefix_sha256": [array_sha256(prefix) for prefix in prefixes],
        "prefix_bit_exact": True,
        "feasibility_mask": mask,
        "feasibility": feasibility,
        "legal_original_candidate_indices": legal,
        "lowest_legal_original_candidate_index": int(legal[0]) if legal.size else -1,
        "mapping_contract": mapping_contract,
    }


def observation_fingerprint(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address reset state, all three RGB views, and language."""

    result = {}
    for key in ("states", "main_images", "wrist_images"):
        value = _cpu_numpy(observation[key])
        result[key] = {"shape": list(value.shape), "sha256": array_sha256(value)}
    descriptions = observation.get("task_descriptions")
    if descriptions is None or len(descriptions) != 1:
        raise DenseBranchContractError("reset observation lacks exactly one instruction")
    instruction = str(descriptions[0])
    result["instruction_utf8_sha256"] = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    return result


def validate_candidate_query(value: Mapping[str, Any]) -> None:
    """Validate query_fn output even when a synthetic/runtime adapter supplies it."""

    hidden = np.asarray(value.get("hidden"))
    state = np.asarray(value.get("processed_state"))
    mapped = np.asarray(value.get("mapped_actions"))
    feasibility = np.asarray(value.get("feasibility_mask"), dtype=bool)
    legal = np.asarray(value.get("legal_original_candidate_indices"), dtype=np.int16)
    prefixes = list(value.get("candidate_prefix_sha256", []))
    native_digests = list(value.get("native_action_sha256", []))
    if hidden.shape != (PREFIX_DIM,) or not np.isfinite(hidden).all():
        raise DenseBranchContractError("query hidden must be finite [960]")
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise DenseBranchContractError("query processed_state must be finite [14]")
    if mapped.shape != (CANDIDATE_COUNT, CHUNK_SIZE, ACTION_DIM):
        raise DenseBranchContractError("query mapped actions must be [4,50,14]")
    if feasibility.shape != (CANDIDATE_COUNT,) or not np.array_equal(legal, np.flatnonzero(feasibility)):
        raise DenseBranchContractError("query feasibility/legal indices disagree")
    if value.get("prefix_bit_exact") is not True or len(prefixes) != CANDIDATE_COUNT or len(set(prefixes)) != 1:
        raise DenseBranchContractError("query does not prove bit-exact four-candidate prefix")
    if len(native_digests) != CANDIDATE_COUNT or any(
        len(str(digest)) != 64 for digest in native_digests
    ):
        raise DenseBranchContractError("query native action digest registry changed")
    expected_lowest = int(legal[0]) if legal.size else -1
    if int(value.get("lowest_legal_original_candidate_index", -2)) != expected_lowest:
        raise DenseBranchContractError("query lowest-legal candidate changed")


def dense_event_targets(
    event_names: Sequence[str], event_steps: Sequence[int], terminal_step: int
) -> dict[str, np.ndarray]:
    """Expand sparse canonical events into per-snapshot/per-transition labels."""

    events = np.asarray(
        [event_id_at_step(event_names, event_steps, step) for step in range(terminal_step + 1)],
        dtype=np.int16,
    )
    next_event = np.zeros(terminal_step, dtype=np.int16)
    duration = np.zeros(terminal_step, dtype=np.float32)
    observed = np.zeros(terminal_step, dtype=bool)
    for start in range(terminal_step):
        future = [
            (str(name), int(step))
            for name, step in zip(event_names, event_steps, strict=True)
            if int(step) > start
        ]
        if future:
            next_event[start] = EVENT_VOCAB.index(future[0][0])
            duration[start] = float(future[0][1] - start)
            observed[start] = True
        else:
            next_event[start] = events[start]
            duration[start] = float(terminal_step - start)
    return {
        "trajectory_event_id": events,
        "transition_next_event_id": next_event,
        "transition_duration_decision_steps": duration,
        "transition_duration_observed": observed,
        "transition_duration_censored": ~observed,
    }


def _validate_snapshot(snapshot: Mapping[str, Any], object_count: int | None = None) -> dict[str, Any]:
    required = {"object_names", "object_poses", "proprio", "telemetry"}
    if set(snapshot) != required:
        raise DenseBranchContractError("snapshot fields changed")
    names = [str(name) for name in snapshot["object_names"]]
    poses = np.asarray(snapshot["object_poses"], dtype=np.float64)
    proprio = np.asarray(snapshot["proprio"], dtype=np.float32)
    if not names or len(set(names)) != len(names) or poses.shape != (len(names), 7):
        raise DenseBranchContractError("snapshot object registry/pose shape changed")
    if object_count is not None and len(names) != object_count:
        raise DenseBranchContractError("snapshot object count changed")
    if proprio.shape != (ACTION_DIM,) or not np.all(np.isfinite(proprio)):
        raise DenseBranchContractError("snapshot proprio must be finite [14]")
    telemetry = dict(snapshot["telemetry"])
    if set(telemetry) != {
        "simulator_timestamp_s", "control_step", "physics_substep_count",
        "reset_generation", "reset_flag", "teleport_flag",
        "simulator_pose_error_flag",
    }:
        raise DenseBranchContractError("pose telemetry fields changed")
    return {"object_names": names, "object_poses": poses, "proprio": proprio, "telemetry": telemetry}


def _same_root_snapshot(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["object_names"] == second["object_names"]
        and np.array_equal(first["object_poses"], second["object_poses"])
        and np.array_equal(first["proprio"], second["proprio"])
    )


def _stack_telemetry(snapshots: Sequence[Mapping[str, Any]], object_count: int) -> dict[str, np.ndarray]:
    scalar_fields = (
        "simulator_timestamp_s", "control_step", "physics_substep_count",
        "reset_generation", "reset_flag",
    )
    result = {
        field: np.asarray([snapshot["telemetry"][field] for snapshot in snapshots])
        for field in scalar_fields
    }
    for field in ("teleport_flag", "simulator_pose_error_flag"):
        value = np.stack([np.asarray(snapshot["telemetry"][field], dtype=bool) for snapshot in snapshots])
        if value.shape != (len(snapshots), object_count):
            raise DenseBranchContractError(f"{field} must align [T,O]")
        result[field] = value
    return result


def collect_dense_group(
    *,
    runtime: Mapping[str, Callable[..., Any]],
    query_fn: Callable[[Mapping[str, Any], int], dict[str, Any]],
    requested_seed: int,
    instruction: str,
    object_registry: Mapping[str, Any],
    pose_quality_spec: Mapping[str, Any],
    event_spec: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    """Collect one reset-matched intervention group using dependency injection."""

    if instruction != INSTRUCTION:
        raise DenseBranchContractError("collector instruction must remain explicit R6e/R6f text")
    if type(max_steps) is not int or max_steps < 1:
        raise DenseBranchContractError("max_steps must be a positive policy-row count")
    canonical_registry = validate_registry(object_registry)
    registry_digest = registry_sha256(canonical_registry)
    canonical_spec = validate_spec(pose_quality_spec, expected_registry_sha256=registry_digest)
    spec_digest = spec_sha256(canonical_spec, expected_registry_sha256=registry_digest)
    root_obs, root_resolved, root_instruction = runtime["reset"](requested_seed, instruction)
    if root_instruction != instruction:
        raise DenseBranchContractError("root reset instruction drift")
    root_snapshot = _validate_snapshot(runtime["snapshot"]())
    if [item["name"] for item in canonical_registry["objects"]] != root_snapshot["object_names"]:
        raise DenseBranchContractError("pose object registry order differs from simulator snapshot")
    root_fingerprint = observation_fingerprint(root_obs)
    root_query = query_fn(root_obs, 0)
    validate_candidate_query(root_query)
    legal = np.asarray(root_query["legal_original_candidate_indices"], dtype=np.int16)
    if legal.size < 2:
        return {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "status": "skipped_fewer_than_two_feasible_root_candidates",
            "requested_seed": requested_seed,
            "resolved_seed": root_resolved,
            "instruction": instruction,
            "root_observation_fingerprint": root_fingerprint,
            "root_query": root_query,
            "eligible_original_candidate_indices": legal,
            "branches": [],
            "fresh_inputs_used": False,
            "performance_or_transfer_claim": False,
        }
    baseline_original = int(legal[0])
    branches: list[dict[str, Any]] = []
    for branch_index, original_index_raw in enumerate(legal.tolist()):
        original_index = int(original_index_raw)
        obs, resolved, observed_instruction = runtime["reset"](requested_seed, instruction)
        branch_snapshot = _validate_snapshot(runtime["snapshot"](), len(root_snapshot["object_names"]))
        if (
            resolved != root_resolved
            or observed_instruction != instruction
            or observation_fingerprint(obs) != root_fingerprint
            or not _same_root_snapshot(root_snapshot, branch_snapshot)
        ):
            raise DenseBranchContractError("branch reset state/object pose/language drift")
        query = query_fn(obs, 0)
        validate_candidate_query(query)
        if (
            not np.array_equal(query["hidden"], root_query["hidden"])
            or query["native_action_sha256"] != root_query["native_action_sha256"]
            or not np.array_equal(query["mapped_actions"], root_query["mapped_actions"])
            or not np.array_equal(query["feasibility_mask"], root_query["feasibility_mask"])
        ):
            raise DenseBranchContractError("branch reset policy prefix/action reproduction drift")
        snapshots = [branch_snapshot]
        queries: list[dict[str, Any]] = []
        success = False
        trajectory_success = [False]
        terminated_seen = False
        truncated_seen = False
        right_censored = False
        censor_reason = ""
        steps = 0
        while steps < max_steps:
            if steps == 0:
                selected = original_index
                selection_role = "root_intervention_original_candidate"
            else:
                selected = int(query["lowest_legal_original_candidate_index"])
                selection_role = "lowest_legal_feasibility_continuation"
                if selected < 0:
                    right_censored = True
                    censor_reason = "all_four_continuation_candidates_infeasible"
                    break
            if not bool(query["feasibility_mask"][selected]):
                raise DenseBranchContractError("selected candidate is infeasible")
            command = np.ascontiguousarray(query["mapped_actions"][selected, 0])
            query_record = {
                **query,
                "selected_original_candidate_index": selected,
                "selection_role": selection_role,
                "executed_action": command,
                "executed_action_mask": np.arange(CHUNK_SIZE) < ACTION_EXEC_STEPS,
            }
            queries.append(query_record)
            obs, terminated, truncated, info = runtime["step"](command.reshape(1, 1, ACTION_DIM))
            steps += 1
            snapshots.append(_validate_snapshot(runtime["snapshot"](), len(root_snapshot["object_names"])))
            success = success or bool(info.get("success", False))
            trajectory_success.append(success)
            terminated_seen = bool(terminated)
            truncated_seen = bool(truncated)
            if terminated_seen or truncated_seen or steps >= max_steps:
                break
            query = query_fn(obs, steps)
            validate_candidate_query(query)
            if np.asarray(query["legal_original_candidate_indices"]).size == 0:
                right_censored = True
                censor_reason = "all_four_continuation_candidates_infeasible"
                # Save the rejected query even though it executes no action.
                queries.append({
                    **query,
                    "selected_original_candidate_index": -1,
                    "selection_role": "right_censored_no_feasible_continuation",
                    "executed_action": np.zeros(ACTION_DIM, dtype=np.float32),
                    "executed_action_mask": np.zeros(CHUNK_SIZE, dtype=bool),
                })
                break
        poses = np.stack([snapshot["object_poses"] for snapshot in snapshots])
        proprio = np.stack([snapshot["proprio"] for snapshot in snapshots])
        telemetry = _stack_telemetry(snapshots, len(root_snapshot["object_names"]))
        quality = derive_pose_quality(
            poses,
            registry=canonical_registry,
            spec=canonical_spec,
            **telemetry,
        )
        if steps:
            starts = np.arange(steps, dtype=np.int64)
            ends = starts + 1
            object_mask, object_reason = derive_interval_supervision_mask(
                pose_quality_valid=quality.valid,
                pose_quality_reason_bitset=quality.reason_bitset,
                reset_generation=quality.reset_generation,
                reset_flag=quality.reset_flag,
                teleport_flag=quality.teleport_flag,
                start_steps=starts,
                end_steps=ends,
            )
        else:
            object_mask = np.empty((0, poses.shape[1]), dtype=bool)
            object_reason = np.empty((0, poses.shape[1]), dtype=np.uint32)
        raw_names, raw_steps, event_names, event_steps = runtime["derive_events"](
            poses, root_snapshot["object_names"], success, event_spec
        )
        targets = derive_branch_targets(event_names, event_steps, min(1, steps), steps)
        dense_targets = dense_event_targets(event_names, event_steps, steps)
        branches.append({
            "branch_index": branch_index,
            "original_candidate_index": original_index,
            "is_feasibility_baseline": original_index == baseline_original,
            "instruction": observed_instruction,
            "resolved_seed": resolved,
            "steps": steps,
            "success": success,
            "trajectory_success": np.asarray(trajectory_success, dtype=bool),
            "terminated": terminated_seen,
            "truncated": truncated_seen,
            "right_censored": right_censored,
            "censor_reason": censor_reason,
            "object_names": root_snapshot["object_names"],
            "object_poses": poses,
            "proprio": proprio,
            "telemetry": telemetry,
            "pose_quality_valid": quality.valid,
            "pose_quality_reason_bitset": quality.reason_bitset,
            "object_delta_supervision_valid": object_mask,
            "object_delta_invalid_reason_bitset": object_reason,
            "raw_event_names": raw_names,
            "raw_event_steps": np.asarray(raw_steps, dtype=np.int32),
            "event_names": event_names,
            "event_steps": np.asarray(event_steps, dtype=np.int32),
            "queries": queries,
            **dense_targets,
            **targets,
        })
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "collected_development_group",
        "evidence_scope": EVIDENCE_SCOPE,
        "requested_seed": requested_seed,
        "resolved_seed": root_resolved,
        "instruction": instruction,
        "root_observation_fingerprint": root_fingerprint,
        "root_object_poses": root_snapshot["object_poses"],
        "root_proprio": root_snapshot["proprio"],
        "root_query": root_query,
        "eligible_original_candidate_indices": legal,
        "baseline_name": BASELINE_NAME,
        "baseline_original_candidate_index": baseline_original,
        "raw_deterministic_baseline_claimed": False,
        "intervention": INTERVENTION,
        "object_registry": canonical_registry,
        "pose_quality_spec": canonical_spec,
        "object_registry_sha256": registry_digest,
        "pose_integrity_spec_sha256": spec_digest,
        "branches": branches,
        "fresh_inputs_used": False,
        "task_success_claimed": False,
        "performance_evaluation_authorized": False,
        "transfer_claim_authorized": False,
    }


def _write_strings(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype("utf-8"))


def _query_arrays(queries: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "query_hidden": np.stack([query["hidden"] for query in queries]).astype(np.float32),
        "query_processed_state": np.stack([query["processed_state"] for query in queries]).astype(np.float32),
        "query_mapped_actions": np.stack([query["mapped_actions"] for query in queries]).astype(np.float32),
        "query_feasibility_mask": np.stack([query["feasibility_mask"] for query in queries]).astype(bool),
        "query_selected_original_candidate_index": np.asarray([query["selected_original_candidate_index"] for query in queries], dtype=np.int16),
        "query_executed_action": np.stack([query["executed_action"] for query in queries]).astype(np.float32),
        "query_executed_action_mask": np.stack([query["executed_action_mask"] for query in queries]).astype(bool),
        "query_native_action_sha256": np.asarray([query["native_action_sha256"] for query in queries], dtype="S64"),
    }


def save_schema6_group(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically write and self-validate a newly collected schema-v6 group."""

    if record.get("status") != "collected_development_group":
        raise DenseBranchContractError("only collected groups may be persisted")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update({
                "format": FORMAT,
                "schema_version": SCHEMA_VERSION,
                "evidence_scope": EVIDENCE_SCOPE,
                "actor_id": ACTOR_ID,
                "instruction": record["instruction"],
                "intervention": INTERVENTION,
                "baseline_name": BASELINE_NAME,
                "baseline_original_candidate_index": int(record["baseline_original_candidate_index"]),
                "raw_deterministic_baseline_claimed": False,
                "fresh_inputs_used": False,
                "task_success_claimed": False,
                "performance_evaluation_authorized": False,
                "transfer_claim_authorized": False,
                "object_registry_sha256": record["object_registry_sha256"],
                "pose_integrity_spec_sha256": record["pose_integrity_spec_sha256"],
            })
            root = handle.create_group("root")
            root.create_dataset("hidden", data=record["root_query"]["hidden"])
            root.create_dataset("processed_state", data=record["root_query"]["processed_state"])
            root.create_dataset("mapped_actions", data=record["root_query"]["mapped_actions"], compression="gzip")
            root.create_dataset("native_action_sha256", data=np.asarray(record["root_query"]["native_action_sha256"], dtype="S64"))
            root.create_dataset("feasibility_mask", data=record["root_query"]["feasibility_mask"])
            root.create_dataset("eligible_original_candidate_indices", data=record["eligible_original_candidate_indices"])
            root.create_dataset("object_poses", data=record["root_object_poses"])
            root.create_dataset("proprio", data=record["root_proprio"])
            root.create_dataset("feasibility_json", data=json.dumps(record["root_query"]["feasibility"], sort_keys=True), dtype=h5py.string_dtype("utf-8"))
            branches = handle.create_group("branches")
            for branch_record in record["branches"]:
                branch = branches.create_group(f"branch_{branch_record['branch_index']:03d}")
                branch.attrs.update({
                    "original_candidate_index": int(branch_record["original_candidate_index"]),
                    "is_feasibility_baseline": bool(branch_record["is_feasibility_baseline"]),
                    "right_censored": bool(branch_record["right_censored"]),
                    "censor_reason": branch_record["censor_reason"],
                    "success_diagnostic_only": bool(branch_record["success"]),
                    "steps": int(branch_record["steps"]),
                })
                branch.create_dataset("object_poses", data=branch_record["object_poses"], compression="gzip")
                branch.create_dataset("proprio", data=branch_record["proprio"], compression="gzip")
                for name, value in _query_arrays(branch_record["queries"]).items():
                    branch.create_dataset(name, data=value, compression="gzip" if value.size > 64 else None)
                branch.create_dataset("object_delta_supervision_valid", data=branch_record["object_delta_supervision_valid"])
                branch.create_dataset("object_delta_invalid_reason_bitset", data=branch_record["object_delta_invalid_reason_bitset"])
                for name in (
                    "trajectory_event_id", "trajectory_success",
                    "transition_next_event_id", "transition_duration_decision_steps",
                    "transition_duration_observed", "transition_duration_censored",
                ):
                    branch.create_dataset(name, data=branch_record[name])
                _write_strings(branch, "raw_event_names", branch_record["raw_event_names"])
                branch.create_dataset("raw_event_steps", data=branch_record["raw_event_steps"])
                _write_strings(branch, "event_names", branch_record["event_names"])
                branch.create_dataset("event_steps", data=branch_record["event_steps"])
                targets = branch.create_group("dense_targets")
                for key in (
                    "pre_event_id", "post_event_id", "next_event_id", "post_chunk_step",
                    "duration", "duration_observed", "duration_censored",
                    "next_event_duration_steps", "next_event_duration_observed",
                    "next_event_duration_censored", "next_event_censor_steps",
                ):
                    targets.attrs[key] = branch_record[key]
                write_pose_quality_v6(
                    branch,
                    registry=record["object_registry"],
                    spec=record["pose_quality_spec"],
                    **branch_record["telemetry"],
                )
            handle.flush()
        validate_schema6_group_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_schema6_group_file(path: Path) -> dict[str, Any]:
    """Revalidate baseline indices, prefix/action shapes, and pose masks."""

    with h5py.File(path, "r") as handle:
        expected_attrs = {
            "format": FORMAT,
            "schema_version": SCHEMA_VERSION,
            "evidence_scope": EVIDENCE_SCOPE,
            "actor_id": ACTOR_ID,
            "instruction": INSTRUCTION,
            "intervention": INTERVENTION,
            "baseline_name": BASELINE_NAME,
            "raw_deterministic_baseline_claimed": False,
            "fresh_inputs_used": False,
            "task_success_claimed": False,
            "performance_evaluation_authorized": False,
            "transfer_claim_authorized": False,
        }
        for key, expected in expected_attrs.items():
            if handle.attrs.get(key) != expected:
                raise DenseBranchContractError(f"schema6 attribute changed: {key}")
        root = handle["root"]
        if root["hidden"].shape != (PREFIX_DIM,) or root["processed_state"].shape != (ACTION_DIM,) or root["mapped_actions"].shape != (CANDIDATE_COUNT, CHUNK_SIZE, ACTION_DIM):
            raise DenseBranchContractError("root dense tensor shapes changed")
        feasibility = root["feasibility_mask"][:].astype(bool)
        eligible = root["eligible_original_candidate_indices"][:].astype(np.int16)
        if not np.array_equal(eligible, np.flatnonzero(feasibility)) or len(eligible) < 2:
            raise DenseBranchContractError("root feasibility/eligible registry mismatch")
        baseline = int(handle.attrs["baseline_original_candidate_index"])
        if baseline != int(eligible[0]):
            raise DenseBranchContractError("baseline is not the lowest legal original index")
        if len(handle["branches"]) != len(eligible):
            raise DenseBranchContractError("branch count does not match eligible candidates")
        for branch_index, original_index in enumerate(eligible.tolist()):
            branch = handle["branches"][f"branch_{branch_index:03d}"]
            if int(branch.attrs["original_candidate_index"]) != original_index:
                raise DenseBranchContractError("branch reindex lost original candidate index")
            query_count = branch["query_hidden"].shape[0]
            expected_shapes = {
                "query_hidden": (query_count, PREFIX_DIM),
                "query_processed_state": (query_count, ACTION_DIM),
                "query_mapped_actions": (query_count, CANDIDATE_COUNT, CHUNK_SIZE, ACTION_DIM),
                "query_feasibility_mask": (query_count, CANDIDATE_COUNT),
                "query_native_action_sha256": (query_count, CANDIDATE_COUNT),
                "query_selected_original_candidate_index": (query_count,),
                "query_executed_action": (query_count, ACTION_DIM),
                "query_executed_action_mask": (query_count, CHUNK_SIZE),
            }
            for name, shape in expected_shapes.items():
                if branch[name].shape != shape:
                    raise DenseBranchContractError(f"invalid branch query shape: {name}")
            if int(branch["query_selected_original_candidate_index"][0]) != original_index:
                raise DenseBranchContractError("root intervention original index changed")
            masks = branch["query_executed_action_mask"][:].astype(bool)
            selected = branch["query_selected_original_candidate_index"][:].astype(int)
            feasibility_rows = branch["query_feasibility_mask"][:].astype(bool)
            for row in range(query_count):
                if selected[row] < 0:
                    if masks[row].any() or feasibility_rows[row].any():
                        raise DenseBranchContractError("censored query is not all-infeasible/no-step")
                elif not feasibility_rows[row, selected[row]] or not np.array_equal(masks[row], np.arange(CHUNK_SIZE) < 1):
                    raise DenseBranchContractError("query executed an infeasible or non-H1 action")
                elif row > 0 and selected[row] != int(np.flatnonzero(feasibility_rows[row])[0]):
                    raise DenseBranchContractError("continuation is not lowest-legal feasibility")
            audit = validate_pose_quality_v6(
                branch,
                expected_registry_sha256=str(handle.attrs["object_registry_sha256"]),
                expected_spec_sha256=str(handle.attrs["pose_integrity_spec_sha256"]),
            )
            quality = branch[POSE_QUALITY_GROUP]
            steps = int(branch.attrs["steps"])
            dense_shapes = {
                "trajectory_event_id": (steps + 1,),
                "trajectory_success": (steps + 1,),
                "transition_next_event_id": (steps,),
                "transition_duration_decision_steps": (steps,),
                "transition_duration_observed": (steps,),
                "transition_duration_censored": (steps,),
            }
            for name, shape in dense_shapes.items():
                if branch[name].shape != shape:
                    raise DenseBranchContractError(f"invalid dense event shape: {name}")
            if not np.array_equal(
                ~branch["transition_duration_observed"][:].astype(bool),
                branch["transition_duration_censored"][:].astype(bool),
            ):
                raise DenseBranchContractError("duration observed/censor masks disagree")
            if steps:
                expected_mask, expected_reason = derive_interval_supervision_mask(
                    pose_quality_valid=quality["pose_quality_valid"][:],
                    pose_quality_reason_bitset=quality["pose_quality_reason_bitset"][:],
                    reset_generation=quality["reset_generation"][:],
                    reset_flag=quality["reset_flag"][:],
                    teleport_flag=quality["teleport_flag"][:],
                    start_steps=np.arange(steps),
                    end_steps=np.arange(steps) + 1,
                )
                if not np.array_equal(branch["object_delta_supervision_valid"][:], expected_mask) or not np.array_equal(branch["object_delta_invalid_reason_bitset"][:], expected_reason):
                    raise DenseBranchContractError("bad pose entered object supervision")
        return {
            "branches": len(eligible),
            "eligible_original_candidate_indices": eligible.astype(int).tolist(),
            "baseline_original_candidate_index": baseline,
            "evidence_scope": EVIDENCE_SCOPE,
        }


__all__ = [
    "BASELINE_NAME",
    "DenseBranchContractError",
    "EVIDENCE_SCOPE",
    "FORMAT",
    "INTERVENTION",
    "SCHEMA_VERSION",
    "bind_r6f_collection_runtime",
    "collect_dense_group",
    "explicit_named_map_chunk_allowing_rejected_values",
    "generate_candidate_query",
    "native_action_sha256",
    "observation_fingerprint",
    "save_schema6_group",
    "validate_candidate_query",
    "validate_schema6_group_file",
]
