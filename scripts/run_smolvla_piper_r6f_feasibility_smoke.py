#!/usr/bin/env python3
"""Run the R6f fixed-candidate feasibility-only SmolVLA/Piper smoke.

For every live observation this runner preprocesses exactly once, performs four
formal SmolVLA forwards with the frozen candidate-0..3 noises, and requires the
captured 960-D observation prefix to be bit-identical across candidates.  It
then selects the lowest candidate index whose *first* explicitly named Piper
action is finite and within all frozen bounds.  This is only a feasibility
baseline: it has no event score, utility, learned ranker, or performance claim.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from run_smolvla_piper_r6d_direct_actor_smoke import (
    ACTION_DIM,
    ACTION_EXEC_STEPS,
    ACTOR_ID,
    CHUNK_SIZE,
    DIRECT_MAX_STEPS,
    EXPECTED_IMAGE_KEYS,
    EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    INSTRUCTION,
    PREFIX_DIM,
    REQUESTED_DEVELOPMENT_SEED,
    DirectActorError,
    _cpu_numpy,
    _load_and_recompute_preregistration,
    _scalar_bool,
    array_sha256,
    atomic_json,
    canonical_sha256,
    file_sha256,
    live_policy_input,
    offload_runtime,
    piper_environment_config,
    reject_fresh_path,
    reset_with_explicit_instruction,
    validate_direct_actor_preregistration,
    validate_loaded_policy_contract,
    validate_piper_step,
    validate_runtime_module_origins,
)
from verify_smolvla_piper_zero_shot_preflight import ALOHA_FEATURE_NAMES, PIPER_ACTION_SLOTS


FORMAT = "smolvla_piper_r6f_fixed_candidate_feasibility_smoke_v1"
PREREGISTRATION_FORMAT = "smolvla_piper_r6f_fixed_candidate_feasibility_preregistration_v1"
R6E_DIRECTORY_NAME = "etsf_smolvla_piper_direct_actor_smoke_r6e_20260828"
R6E_PREREGISTRATION_SHA256 = "7b2509bbcd80b65c1c39d1fb57ef6e2f2b6564eaa6bc9cbf6238b37a2f8cb047"
R6E_PREREGISTRATION_LOGICAL_SHA256 = "12525189aae5a4304bb4753b9676db46e5cfdb30b90056219175cbab8294bfd7"
R6E_RUNNER_SHA256 = "c75e7e59a472e06ec36cef5283861d43fa063f9c0f47682b9cd269941917d383"
CANDIDATE_COUNT = 4
R6E_EXPECTED_UNSAFE_VALUE = -0.0043728
R6E_EXPECTED_UNSAFE_TARGET = "left:joint2"
R6E_EXPECTED_UNSAFE_TARGET_INDEX = 1


class FeasibilitySmokeError(DirectActorError):
    """A fail-closed R6f feasibility contract violation."""


def _json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeasibilitySmokeError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FeasibilitySmokeError(f"{role} must contain an object")
    return value


def bind_r6e_preregistration(
    path: Path,
    *,
    expected_file_sha256: str = R6E_PREREGISTRATION_SHA256,
    expected_logical_sha256: str = R6E_PREREGISTRATION_LOGICAL_SHA256,
    expected_runner_sha256: str = R6E_RUNNER_SHA256,
    expected_directory_name: str = R6E_DIRECTORY_NAME,
    validator: Callable[[Path], dict[str, Any]] = validate_direct_actor_preregistration,
) -> dict[str, Any]:
    """Bind R6e lineage without inventing a receipt for its pre-step failure."""

    prereg_path = reject_fresh_path(path, "R6e preregistration")
    if prereg_path.parent.name != expected_directory_name or prereg_path.name != "direct_actor_preregistration.json":
        raise FeasibilitySmokeError("unexpected R6e preregistration location")
    if file_sha256(prereg_path) != expected_file_sha256:
        raise FeasibilitySmokeError("R6e preregistration file SHA mismatch")
    value = validator(prereg_path)
    if value.get("preregistration_sha256") != expected_logical_sha256:
        raise FeasibilitySmokeError("R6e preregistration logical SHA mismatch")
    runner = value.get("runtime_source_artifacts", {}).get("direct_actor_runner", {})
    runner_path = Path(str(runner.get("path", "")))
    if not runner_path.is_file() or runner.get("sha256") != expected_runner_sha256 or file_sha256(runner_path) != expected_runner_sha256:
        raise FeasibilitySmokeError("R6e runner source binding changed")
    if value.get("execution_contract", {}).get("candidate_index") != 0:
        raise FeasibilitySmokeError("R6e lineage is not the fixed candidate0 baseline")
    return {
        "path": str(prereg_path),
        "file_sha256": expected_file_sha256,
        "logical_sha256": expected_logical_sha256,
        "runner_sha256": expected_runner_sha256,
        "failure_receipt_bound": False,
        "authorization": "lineage_only_not_R6f_execution_authority",
        "expected_external_diagnostic_not_content_authenticated": {
            "query_index": 0,
            "candidate_index": 0,
            "target_index": R6E_EXPECTED_UNSAFE_TARGET_INDEX,
            "target_joint_name": R6E_EXPECTED_UNSAFE_TARGET,
            "reported_value_rounded": R6E_EXPECTED_UNSAFE_VALUE,
            "reported_lower_bound": 0.0,
            "reported_env_steps": 0,
        },
    }


def fixed_candidate_noise(config: Any, scene_seed: int, query_index: int, candidate_index: int, device: Any) -> Any:
    """Use the frozen native SmolVLA noise registry for candidate 0..3."""

    if type(candidate_index) is not int or not 0 <= candidate_index < CANDIDATE_COUNT:
        raise FeasibilitySmokeError("candidate index must be 0..3")
    from collect_smolvla_etsf_event_branches import make_noise

    return make_noise(config, scene_seed, query_index, candidate_index, device)


def explicit_named_map_first_action(source_action: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy a single Aloha row through the frozen named side/ordinal registry."""

    source = np.asarray(source_action)
    if source.shape != (ACTION_DIM,) or source.dtype.kind != "f":
        raise FeasibilitySmokeError("candidate first action must be real float [14]")
    source_by_name = {name: index for index, name in enumerate(ALOHA_FEATURE_NAMES)}
    target = np.empty_like(source)
    mapping = []
    for target_index, slot in enumerate(PIPER_ACTION_SLOTS):
        source_index = source_by_name[slot.source_feature_name]
        target[target_index] = source[source_index]
        mapping.append({
            "source_index": source_index,
            "source_feature_name": slot.source_feature_name,
            "target_index": target_index,
            "target_joint_name": slot.target_joint_name,
            "side": slot.side,
            "ordinal": slot.ordinal,
            "numeric_transform": slot.numeric_transform,
        })
    return np.ascontiguousarray(target), {
        "mode": "explicit_named_ordinal_angle_preserving_mapping",
        "mapping": mapping,
        "derived_from_equal_14d_width": False,
        "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False,
        "clipping_or_scaling_applied": False,
    }


def assess_candidate_first_action(
    action: np.ndarray,
    bounds: Sequence[Sequence[float]],
    *,
    query_index: int,
    candidate_index: int,
) -> dict[str, Any]:
    """Return a JSON-safe acceptance/rejection record without clipping."""

    command = np.asarray(action)
    violations: list[dict[str, Any]] = []
    if command.shape != (ACTION_DIM,) or command.dtype.kind != "f":
        return {
            "accepted": False,
            "reason": "invalid_shape_or_dtype",
            "shape": list(command.shape),
            "dtype": str(command.dtype),
            "violations": [],
        }
    for index, (slot, pair) in enumerate(zip(PIPER_ACTION_SLOTS, bounds, strict=True)):
        value = float(command[index])
        lower, upper = float(pair[0]), float(pair[1])
        if not np.isfinite(value):
            violations.append({
                "target_index": index,
                "target_joint_name": slot.target_joint_name,
                "reason": "nonfinite",
                "value": None,
                "allowed": [lower, upper],
            })
        elif value < lower or value > upper:
            violations.append({
                "target_index": index,
                "target_joint_name": slot.target_joint_name,
                "reason": "outside_bounds",
                "value": value,
                "allowed": [lower, upper],
            })
    if violations:
        return {
            "accepted": False,
            "reason": "nonfinite_or_outside_piper_bounds",
            "violations": violations,
            "clipping_applied": False,
        }
    check = validate_piper_step(command, bounds, step_index=query_index)
    return {
        "accepted": True,
        "reason": None,
        "candidate_index": candidate_index,
        "validation": check,
        "clipping_applied": False,
    }


def run_feasibility_loop(
    *,
    env: Any,
    policy: Any,
    preprocessor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    postprocessor: Callable[[Any], Any],
    capture: Any,
    observation: Mapping[str, Any],
    bounds: Sequence[Sequence[float]],
    device: Any,
    action_converter: Callable[[np.ndarray], Any],
    noise_factory: Callable[[Any, int, int, int, Any], Any] = fixed_candidate_noise,
    max_steps: int = DIRECT_MAX_STEPS,
) -> dict[str, Any]:
    """Generate all four candidates on one processed obs before any H=1 step."""

    if type(max_steps) is not int or not 1 <= max_steps <= DIRECT_MAX_STEPS:
        raise FeasibilitySmokeError("R6f max_steps must be in [1,4]")
    if int(getattr(policy.config, "chunk_size", -1)) != CHUNK_SIZE or int(getattr(policy.config, "max_action_dim", -1)) < ACTION_DIM:
        raise FeasibilitySmokeError("SmolVLA chunk/action contract changed")
    image_keys = tuple(getattr(policy.config, "image_features", ()))
    current = observation
    queries: list[dict[str, Any]] = []
    mapping_contract: dict[str, Any] | None = None
    success = False
    terminated_seen = False
    truncated_seen = False
    no_feasible = False
    for query_index in range(max_steps):
        raw = live_policy_input(current, image_keys)
        interface = dict(raw.pop("_interface"))
        if "drive_target" in interface:
            interface["drive_target"] = _cpu_numpy(
                interface["drive_target"], dtype=np.float32
            ).tolist()
        processed = preprocessor(raw)
        if "observation.state" not in processed:
            raise FeasibilitySmokeError("preprocessor omitted observation.state")
        processed_state = _cpu_numpy(processed["observation.state"], dtype=np.float32)
        if processed_state.shape == (ACTION_DIM,):
            processed_state = processed_state[None]
        if processed_state.shape != (1, ACTION_DIM) or not np.all(np.isfinite(processed_state)):
            raise FeasibilitySmokeError("processed state must be finite [1,14]")
        state_hash = array_sha256(processed_state[0])
        candidate_records: list[dict[str, Any]] = []
        commands: dict[int, np.ndarray] = {}
        prefixes: list[np.ndarray] = []
        for candidate_index in range(CANDIDATE_COUNT):
            policy.reset()
            capture.reset()
            noise = noise_factory(
                policy.config,
                REQUESTED_DEVELOPMENT_SEED,
                query_index,
                candidate_index,
                device,
            )
            noise_cpu = _cpu_numpy(noise, dtype=np.float32)
            normalized = policy.predict_action_chunk(dict(processed), noise=noise)
            postprocessed = _cpu_numpy(postprocessor(normalized), dtype=np.float32)
            prefix = _cpu_numpy(capture.consume(), dtype=np.float32)
            if prefix.shape != (PREFIX_DIM,) or not np.all(np.isfinite(prefix)):
                raise FeasibilitySmokeError("candidate shared prefix must be finite [960]")
            prefixes.append(prefix)
            if postprocessed.shape == (1, CHUNK_SIZE, ACTION_DIM):
                postprocessed = postprocessed[0]
            if postprocessed.shape != (CHUNK_SIZE, ACTION_DIM):
                raise FeasibilitySmokeError("postprocessed candidate must be [50,14]")
            mapped, candidate_mapping = explicit_named_map_first_action(postprocessed[0])
            if mapping_contract is None:
                mapping_contract = candidate_mapping
            elif mapping_contract != candidate_mapping:
                raise FeasibilitySmokeError("named mapping changed between candidates")
            assessment = assess_candidate_first_action(
                mapped,
                bounds,
                query_index=query_index,
                candidate_index=candidate_index,
            )
            if assessment["accepted"]:
                commands[candidate_index] = mapped
            candidate_records.append({
                "candidate_index": candidate_index,
                "noise_sha256": array_sha256(noise_cpu),
                "prefix_sha256": array_sha256(prefix),
                "postprocessed_chunk_sha256": array_sha256(postprocessed),
                "mapped_first_action_sha256": array_sha256(mapped),
                "first_action": [float(value) if np.isfinite(value) else None for value in mapped],
                "feasibility": assessment,
            })
            if array_sha256(_cpu_numpy(processed["observation.state"], dtype=np.float32).reshape(-1)) != state_hash:
                raise FeasibilitySmokeError("processed state mutated between candidates")
        reference = prefixes[0]
        prefix_bit_exact = all(np.array_equal(prefix, reference) for prefix in prefixes[1:])
        if not prefix_bit_exact:
            raise FeasibilitySmokeError("flow noise changed the shared 960D prefix")
        valid_indices = sorted(commands)
        selected_index = valid_indices[0] if valid_indices else None
        query_record: dict[str, Any] = {
            "query_index": query_index,
            "selection_rule": "lowest_candidate_index_with_finite_in_bounds_first_action",
            "selection_uses_event_or_utility_score": False,
            "processed_state": processed_state[0].tolist(),
            "processed_state_sha256": state_hash,
            "shared_prefix": reference.tolist(),
            "candidate_prefix_sha256": [array_sha256(prefix) for prefix in prefixes],
            "prefix_bit_exact_across_all_four_candidates": True,
            "candidate_records": candidate_records,
            "selected_candidate_index": selected_index,
            "input_interface": interface,
        }
        queries.append(query_record)
        if selected_index is None:
            no_feasible = True
            query_record["env_step_performed"] = False
            query_record["halt_reason"] = "no_feasible_candidate_first_action"
            break
        command = commands[selected_index]
        # Recheck immediately before conversion and require the conversion echo
        # to preserve every bit; there is no scaling/clipping path.
        validate_piper_step(command, bounds, step_index=query_index)
        env_action = action_converter(command.copy())
        echoed = _cpu_numpy(env_action)
        if echoed.shape != (1, ACTION_EXEC_STEPS, ACTION_DIM) or not np.array_equal(echoed.reshape(ACTION_DIM), command):
            raise FeasibilitySmokeError("action converter changed selected command; env.step forbidden")
        next_observation, _, terminated, truncated, info = env.step(env_action, auto_reset=False)
        query_record["env_step_performed"] = True
        query_record["action_horizon_per_env_step"] = ACTION_EXEC_STEPS
        success = success or _scalar_bool(info.get("success", [False]))
        terminated_seen = _scalar_bool(terminated)
        truncated_seen = _scalar_bool(truncated)
        current = next_observation
        if terminated_seen or truncated_seen:
            break
    return {
        "queries_performed": len(queries),
        "steps_executed": sum(bool(query.get("env_step_performed")) for query in queries),
        "max_steps": max_steps,
        "action_exec_steps": ACTION_EXEC_STEPS,
        "candidate_count_per_query": CANDIDATE_COUNT,
        "selection_rule": "lowest_candidate_index_with_finite_in_bounds_first_action",
        "feasibility_baseline_only": True,
        "event_or_utility_scoring_performed": False,
        "no_feasible_candidate_halt": no_feasible,
        "stopped_on_termination": terminated_seen,
        "stopped_on_truncation": truncated_seen,
        "success_observed_diagnostic_only": success,
        "all_env_actions_prevalidated": True,
        "silent_clipping_possible": False,
        "mapping_contract": mapping_contract,
        "queries": queries,
    }


def build_feasibility_preregistration(
    *,
    r6e: Mapping[str, Any],
    r6e_preregistration: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Create new R6f authority while retaining every R6e safety gate."""

    output_path = reject_fresh_path(output, "R6f output receipt")
    if output_path.exists():
        raise FileExistsError(output_path)
    runner = Path(__file__).resolve()
    inherited_sections = {
        key: r6e_preregistration[key]
        for key in (
            "r6c_binding", "r6d_binding", "development_seed", "runtime_roots",
            "runtime_source_artifacts", "vlm_metadata_bundle_sha256",
            "model_bundle_sha256", "capability_contract", "mapping_contract",
            "state_contract", "caveats",
        )
    }
    base: dict[str, Any] = {
        "format": PREREGISTRATION_FORMAT,
        "status": "preregistered_R6f_feasibility_simulation_only_not_executed",
        "actor_id": ACTOR_ID,
        "explicit_instruction": INSTRUCTION,
        "source_body": "aloha",
        "target_body": "piper",
        "r6e_lineage": dict(r6e),
        "inherited_R6e_contract": inherited_sections,
        "inherited_R6e_contract_sha256": canonical_sha256(inherited_sections),
        "r6f_runner": {"path": str(runner), "sha256": file_sha256(runner)},
        "output": str(output_path),
        "execution_contract": {
            "max_steps": DIRECT_MAX_STEPS,
            "action_exec_steps": ACTION_EXEC_STEPS,
            "candidate_indices": list(range(CANDIDATE_COUNT)),
            "one_preprocessor_call_per_live_query": True,
            "four_formal_policy_forwards_per_live_query": True,
            "proper_checkpoint_postprocessor_per_candidate": True,
            "prefix_dim": PREFIX_DIM,
            "prefix_bit_exact_across_candidates_required": True,
            "selection_rule": "lowest_candidate_index_with_finite_in_bounds_first_action",
            "no_feasible_candidate_behavior": "zero_env_step_for_query_then_fail_closed_receipt",
            "event_or_utility_scoring_authorized": False,
            "precomputed_chunks_forbidden": True,
        },
        "capability_contract": {
            "simulation_execution_authorized": True,
            "real_robot_execution_authorized": False,
            "fresh_inputs_allowed": False,
            "fresh_trajectory_or_label_opened": False,
            "performance_evaluation_authorized": False,
            "task_success_claim_authorized": False,
            "transfer_claim_authorized": False,
        },
        "interpretation": "candidate feasibility fallback baseline only; not ETSF/event scoring or performance evidence",
    }
    return {**base, "preregistration_sha256": canonical_sha256(base)}


def validate_feasibility_preregistration(path: Path) -> dict[str, Any]:
    prereg_path = reject_fresh_path(path, "R6f preregistration")
    value = _json(prereg_path, "R6f preregistration")
    if set(value) != {
        "format", "status", "actor_id", "explicit_instruction", "source_body",
        "target_body", "r6e_lineage", "inherited_R6e_contract",
        "inherited_R6e_contract_sha256", "r6f_runner", "output",
        "execution_contract", "capability_contract", "interpretation",
        "preregistration_sha256",
    }:
        raise FeasibilitySmokeError("R6f preregistration fields changed")
    recorded = value.get("preregistration_sha256")
    base = {key: item for key, item in value.items() if key != "preregistration_sha256"}
    if recorded != canonical_sha256(base):
        raise FeasibilitySmokeError("R6f logical preregistration SHA mismatch")
    if value.get("format") != PREREGISTRATION_FORMAT or value.get("status") != "preregistered_R6f_feasibility_simulation_only_not_executed":
        raise FeasibilitySmokeError("unexpected R6f preregistration")
    if value.get("actor_id") != ACTOR_ID or value.get("explicit_instruction") != INSTRUCTION or value.get("source_body") != "aloha" or value.get("target_body") != "piper":
        raise FeasibilitySmokeError("R6f identity/language/body contract changed")
    expected_execution = {
        "max_steps": 4,
        "action_exec_steps": 1,
        "candidate_indices": [0, 1, 2, 3],
        "one_preprocessor_call_per_live_query": True,
        "four_formal_policy_forwards_per_live_query": True,
        "proper_checkpoint_postprocessor_per_candidate": True,
        "prefix_dim": 960,
        "prefix_bit_exact_across_candidates_required": True,
        "selection_rule": "lowest_candidate_index_with_finite_in_bounds_first_action",
        "no_feasible_candidate_behavior": "zero_env_step_for_query_then_fail_closed_receipt",
        "event_or_utility_scoring_authorized": False,
        "precomputed_chunks_forbidden": True,
    }
    if value.get("execution_contract") != expected_execution:
        raise FeasibilitySmokeError("R6f execution/selection contract changed")
    expected_capability = {
        "simulation_execution_authorized": True,
        "real_robot_execution_authorized": False,
        "fresh_inputs_allowed": False,
        "fresh_trajectory_or_label_opened": False,
        "performance_evaluation_authorized": False,
        "task_success_claim_authorized": False,
        "transfer_claim_authorized": False,
    }
    if value.get("capability_contract") != expected_capability:
        raise FeasibilitySmokeError("R6f Fresh/capability contract changed")
    if value.get("interpretation") != "candidate feasibility fallback baseline only; not ETSF/event scoring or performance evidence":
        raise FeasibilitySmokeError("R6f interpretation boundary changed")
    runner = value.get("r6f_runner", {})
    runner_path = Path(str(runner.get("path", "")))
    if not runner_path.is_file() or file_sha256(runner_path) != runner.get("sha256") or runner.get("sha256") != file_sha256(Path(__file__)):
        raise FeasibilitySmokeError("R6f runner source changed")
    inherited = value.get("inherited_R6e_contract", {})
    if value.get("inherited_R6e_contract_sha256") != canonical_sha256(inherited):
        raise FeasibilitySmokeError("inherited R6e contract changed")
    reject_fresh_path(Path(value["output"]), "R6f receipt output")
    return value


def load_and_recompute_feasibility_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg = validate_feasibility_preregistration(path)
    r6e_record = prereg["r6e_lineage"]
    r6e = bind_r6e_preregistration(Path(r6e_record["path"]))
    r6e_prereg, r6c, r6d, seed = _load_and_recompute_preregistration(Path(r6e["path"]))
    expected = build_feasibility_preregistration(
        r6e=r6e,
        r6e_preregistration=r6e_prereg,
        output=Path(prereg["output"]),
    )
    if prereg != expected:
        raise FeasibilitySmokeError("R6f preregistration differs from full recomputation")
    return prereg, r6e_prereg, r6c, r6d, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    prereg, r6e_prereg, r6c, r6d, seed = load_and_recompute_feasibility_preregistration(args.preregistration)
    output = reject_fresh_path(Path(prereg["output"]), "R6f output receipt")
    if output.exists():
        raise FileExistsError(output)
    roots = {key: Path(value) for key, value in r6e_prereg["runtime_roots"].items()}
    os.environ["ASSETS_PATH"] = str(roots["robotwin_root"])
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path[:0] = [str(roots["robotwin_code"]), str(roots["rlinf_root"]), str(roots["lerobot_root"] / "src")]

    import torch
    from collect_smolvla_etsf_event_branches import resolve_shared_prefix_capture
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    runtime_origins = validate_runtime_module_origins(
        {
            "rlinf_robotwin_env": importlib.import_module("rlinf.envs.robotwin.robotwin_env"),
            "robotwin_vector_env": importlib.import_module("robotwin.envs.vector_env"),
            "robotwin_base_task": importlib.import_module("envs._base_task"),
            "robotwin_robot_controller": importlib.import_module("envs.robot.robot"),
        },
        r6d["runtime_source_artifacts"],
    )
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise FeasibilitySmokeError("R6f must run on the designated RTX 4090")
    device = torch.device("cuda:0")
    config = PreTrainedConfig.from_pretrained(roots["model_path"], local_files_only=True)
    config.device = str(device)
    config.vlm_model_name = str(roots["vlm_metadata_path"])
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(roots["model_path"], config=config, local_files_only=True, strict=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(roots["model_path"]),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(roots["vlm_metadata_path"])},
        },
    )
    loaded_policy_contract = validate_loaded_policy_contract(
        config=config,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        torch_module=torch,
        device=device,
    )
    capture = resolve_shared_prefix_capture(policy)
    env = None
    try:
        seeds_path = roots["rlinf_root"] / "rlinf/envs/robotwin/seeds/eval_seeds.json"
        frozen_seeds = r6e_prereg["runtime_source_artifacts"]["robotwin_eval_seed_registry"]
        if seeds_path.resolve() != Path(frozen_seeds["path"]).resolve() or file_sha256(seeds_path) != frozen_seeds["sha256"]:
            raise FeasibilitySmokeError("R6e-bound RoboTwin eval seed registry changed")
        env_cfg = piper_environment_config(roots["robotwin_root"], seeds_path, output.parent, step_limit=DIRECT_MAX_STEPS)
        env = RoboTwinEnv(cfg=OmegaConf.create(env_cfg), num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None, record_metrics=True)
        observation, _ = reset_with_explicit_instruction(env, requested_seed=REQUESTED_DEVELOPMENT_SEED, instruction=INSTRUCTION)
        resolved_seed = int(env.venv.envs[0].task.ep_num)
        if resolved_seed != EXPECTED_RESOLVED_DEVELOPMENT_SEED:
            raise FeasibilitySmokeError("resolved development seed changed")
        bounds = r6c["receipt"]["static_contract"]["static_semantics"]["piper_action_bounds"]

        def to_cuda(command: np.ndarray) -> Any:
            return torch.from_numpy(command.reshape(1, 1, ACTION_DIM)).to(device=device, dtype=torch.float32)

        execution = run_feasibility_loop(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            capture=capture,
            observation=observation,
            bounds=bounds,
            device=device,
            action_converter=to_cuda,
            max_steps=DIRECT_MAX_STEPS,
        )
        first_candidate = execution["queries"][0]["candidate_records"][0]
        if execution["steps_executed"] == 0 and execution["no_feasible_candidate_halt"]:
            receipt_status = "completed_R6f_zero_step_no_feasible_candidate_fail_closed"
        elif execution["no_feasible_candidate_halt"]:
            receipt_status = "completed_R6f_stopped_on_no_feasible_candidate"
        else:
            receipt_status = "completed_R6f_feasibility_simulation_interface_smoke"
        result = {
            "format": FORMAT,
            "status": receipt_status,
            "actor_id": ACTOR_ID,
            "source_body": "aloha",
            "target_body": "piper",
            "target_runtime": "RoboTwin_simulation_only",
            "real_robot_execution": False,
            "fresh_inputs_used": False,
            "fresh_trajectory_or_label_opened": False,
            "task_success_claimed": False,
            "performance_evaluation_authorized": False,
            "transfer_claim_authorized": False,
            "event_or_utility_scoring_performed": False,
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "file_sha256": file_sha256(args.preregistration),
                "logical_sha256": prereg["preregistration_sha256"],
            },
            "r6e_lineage": prereg["r6e_lineage"],
            "r6e_candidate0_diagnostic_independently_recomputed": {
                "accepted": first_candidate["feasibility"]["accepted"],
                "first_action": first_candidate["first_action"],
                "feasibility": first_candidate["feasibility"],
                "matches_reported_left_joint2_failure_approximately": any(
                    violation.get("target_joint_name") == R6E_EXPECTED_UNSAFE_TARGET
                    and violation.get("value") is not None
                    and np.isclose(violation["value"], R6E_EXPECTED_UNSAFE_VALUE, rtol=0.0, atol=5e-7)
                    for violation in first_candidate["feasibility"].get("violations", [])
                ),
            },
            "development_seed_contract": seed,
            "environment_contract": {
                "embodiment": ["piper", "piper", 0.6],
                "requested_seed": REQUESTED_DEVELOPMENT_SEED,
                "resolved_seed": resolved_seed,
                "explicit_instruction": INSTRUCTION,
                "scene_seed_and_instruction_strictly_bound": False,
                "center_crop": False,
                "collect_wrist_camera": True,
                "state_is_measured_qpos": False,
                "runtime_module_origins": runtime_origins,
                "eval_seed_registry": dict(frozen_seeds),
            },
            "loaded_policy_contract": loaded_policy_contract,
            "execution": execution,
            "time_contract": {"unit": "policy action row count", "physical_duration_claimed": False},
            "implementation_sha256": file_sha256(Path(__file__)),
            "interpretation": "fixed-candidate feasibility fallback only; no ETSF/event ranking, task performance, transfer, safety, or real-robot claim",
        }
        atomic_json(output, result)
    finally:
        if env is not None:
            offload_runtime(capture, env, policy, torch)
        else:
            capture.close()
            policy.to("cpu")
            torch.cuda.empty_cache()
    print(json.dumps({
        "status": result["status"],
        "steps_executed": execution["steps_executed"],
        "first_selected_candidate_index": execution["queries"][0]["selected_candidate_index"],
        "task_success_claimed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
