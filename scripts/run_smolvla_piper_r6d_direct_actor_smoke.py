#!/usr/bin/env python3
"""Run a frozen, online SmolVLA(Aloha-trained) -> Piper simulation smoke.

This is a four-query direct-actor *interface* smoke, not an evaluation.  Each
query preprocesses the current Piper main/left-wrist/right-wrist RGB and
drive-target state, performs a fresh SmolVLA forward with candidate-0 fixed
noise, postprocesses the chunk, and executes only its first row (H=1).  The
14 columns are copied through an explicit named side/ordinal registry; equal
width is never evidence of embodiment equivalence.  Non-finite, out-of-bound,
or converter-modified commands never reach ``env.step`` and are never clipped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from execute_smolvla_piper_r6c_simulation_smoke import (
    ACTION_DIM,
    ACTOR_ID,
    EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    REQUESTED_DEVELOPMENT_SEED,
    SimulationSmokeError,
    _scalar_bool,
    atomic_json,
    bind_development_seed_manifest,
    bind_r6c_preflight,
    canonical_sha256,
    piper_environment_config,
    reset_with_explicit_instruction,
    validate_piper_step,
    validate_reset_observation,
)
from verify_smolvla_piper_zero_shot_preflight import (
    ALOHA_FEATURE_NAMES,
    PIPER_ACTION_SLOTS,
    array_sha256,
    file_sha256,
    reject_fresh_path,
)


FORMAT = "smolvla_piper_r6d_direct_actor_smoke_v1"
PREREGISTRATION_FORMAT = "smolvla_piper_r6d_direct_actor_preregistration_v1"
R6D_DIRECTORY_NAME = "etsf_smolvla_piper_simulation_smoke_r6d_20260827"
R6D_PREREGISTRATION_SHA256 = "717d2980cd0044fe5bece5d4f1eb2ee792dae066a01a3f5fdb973e958c0b375b"
R6D_PREREGISTRATION_LOGICAL_SHA256 = "ca5b94742690ac8c1616c0d651a87ccbb07e5da56019f7e1eb56bc1a75b2ff03"
R6D_RECEIPT_SHA256 = "cde4f2302f3487539fd1459874994ebc685e4c4e5ec561b1c019b51534c7e14d"
R6D_EXECUTOR_SHA256 = "c65008dce0ce54f45451bfdfc49c58d88d3abb69f4d84961687123ee0d39fdef"
DIRECT_MAX_STEPS = 4
ACTION_EXEC_STEPS = 1
CHUNK_SIZE = 50
PREFIX_DIM = 960
INSTRUCTION = "move the can into the pot"
EXPECTED_IMAGE_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
EXPECTED_STATE_DIMENSION_RESOLUTION = {
    "checkpoint_policy_state_dim": 6,
    "conflict_resolved_for_execution": False,
    "execution_authorized": False,
    "mode": "explicit_forward_only_runtime_shape_probe",
    "normalizer_state_dim": 14,
    "runtime_probe_state_dim": 14,
    "train_env_state_dim": 14,
    "train_policy_state_dim": 6,
}
R6D_RUNTIME_SOURCE_ROLES = frozenset({
    "rlinf_robotwin_env",
    "robotwin_vector_env",
    "robotwin_base_task",
    "robotwin_robot_controller",
})


class DirectActorError(SimulationSmokeError):
    """A fail-closed direct-actor contract violation."""


def _json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DirectActorError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectActorError(f"{role} must contain a JSON object")
    return value


def directory_bundle_sha256(path: Path) -> str:
    """Hash a fully materialized bundle; unresolved HF symlinks are forbidden.

    Formal R6c binds one ordinary-file actor directory.  A Hugging Face cache
    snapshot must first be materialized and independently rebound rather than
    letting a symlink escape the authenticated root.
    """

    root = reject_fresh_path(path, "directory bundle")
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise DirectActorError(f"empty directory bundle: {root}")
    for item in files:
        if item.is_symlink():
            raise DirectActorError(f"symlink forbidden in directory bundle: {item}")
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def validate_runtime_module_origins(
    modules: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Prove imported simulator modules are the preregistered support sources."""

    if set(modules) != set(expected):
        raise DirectActorError("runtime module/source registries differ")
    result: dict[str, Any] = {}
    for role, module in modules.items():
        raw = getattr(module, "__file__", None)
        if not raw:
            raise DirectActorError(f"runtime module has no source file: {role}")
        actual = Path(str(raw)).resolve()
        frozen = Path(str(expected[role].get("path", ""))).resolve()
        if actual != frozen or file_sha256(actual) != expected[role].get("sha256"):
            raise DirectActorError(f"runtime module origin changed: {role}")
        result[role] = {"path": str(actual), "sha256": file_sha256(actual)}
    return result


def validate_loaded_policy_contract(
    *,
    config: Any,
    preprocessor: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    postprocessor: Callable[[Any], Any],
    torch_module: Any,
    device: Any,
) -> dict[str, Any]:
    """Exercise the loaded 14D pre/post path before constructing RoboTwin."""

    state_feature = getattr(config, "robot_state_feature", None)
    action_feature = getattr(config, "action_feature", None)
    if (
        state_feature is None
        or tuple(getattr(state_feature, "shape", ())) != (6,)
        or int(getattr(config, "max_state_dim", -1)) < ACTION_DIM
        or action_feature is None
        or tuple(getattr(action_feature, "shape", ())) != (ACTION_DIM,)
        or int(getattr(config, "max_action_dim", -1)) < ACTION_DIM
        or int(getattr(config, "chunk_size", -1)) != CHUNK_SIZE
        or tuple(getattr(config, "image_features", ())) != EXPECTED_IMAGE_KEYS
    ):
        raise DirectActorError("loaded SmolVLA config differs from the R6c 6/14/14 contract")
    raw: dict[str, Any] = {
        "observation.state": torch_module.zeros(ACTION_DIM, dtype=torch_module.float32),
        "task": INSTRUCTION,
    }
    for key in EXPECTED_IMAGE_KEYS:
        raw[key] = torch_module.zeros((3, 240, 320), dtype=torch_module.float32)
    processed = preprocessor(raw)
    if "observation.state" not in processed:
        raise DirectActorError("loaded preprocessor omitted observation.state")
    state = _cpu_numpy(processed["observation.state"], dtype=np.float32)
    if state.shape == (ACTION_DIM,):
        state = state[None]
    if state.shape != (1, ACTION_DIM) or not np.all(np.isfinite(state)):
        raise DirectActorError("loaded normalizer does not accept runtime state14")
    normalized_action = torch_module.zeros(
        (1, CHUNK_SIZE, ACTION_DIM), device=device, dtype=torch_module.float32
    )
    postprocessed = _cpu_numpy(postprocessor(normalized_action), dtype=np.float32)
    if postprocessed.shape not in {
        (CHUNK_SIZE, ACTION_DIM),
        (1, CHUNK_SIZE, ACTION_DIM),
    } or not np.all(np.isfinite(postprocessed)):
        raise DirectActorError("loaded postprocessor does not produce finite action14 chunks")
    return {
        "checkpoint_declared_state_dim": 6,
        "runtime_preprocessed_state_shape": list(state.shape),
        "checkpoint_action_dim": ACTION_DIM,
        "runtime_postprocessed_action_shape": list(postprocessed.shape),
        "state_dimension_conflict_retained": True,
        "normalizer_state_dim": ACTION_DIM,
    }


def bind_r6d_simulation_receipt(
    preregistration_path: Path,
    receipt_path: Path,
    *,
    expected_preregistration_sha256: str = R6D_PREREGISTRATION_SHA256,
    expected_receipt_sha256: str = R6D_RECEIPT_SHA256,
    expected_directory_name: str = R6D_DIRECTORY_NAME,
    expected_executor_sha256: str = R6D_EXECUTOR_SHA256,
    expected_runtime_source_roles: frozenset[str] = R6D_RUNTIME_SOURCE_ROLES,
) -> dict[str, Any]:
    """Authenticate R6d evidence without treating it as execution authority."""

    prereg_path = reject_fresh_path(preregistration_path, "R6d preregistration")
    receipt_file = reject_fresh_path(receipt_path, "R6d receipt")
    if prereg_path.parent != receipt_file.parent or prereg_path.parent.name != expected_directory_name:
        raise DirectActorError("R6d files must share the exact frozen directory")
    if prereg_path.name != "simulation_preregistration.json" or receipt_file.name != "simulation_receipt.json":
        raise DirectActorError("unexpected R6d artifact filename")
    if file_sha256(prereg_path) != expected_preregistration_sha256:
        raise DirectActorError("R6d preregistration SHA256 mismatch")
    if file_sha256(receipt_file) != expected_receipt_sha256:
        raise DirectActorError("R6d receipt SHA256 mismatch")
    prereg = _json(prereg_path, "R6d preregistration")
    receipt = _json(receipt_file, "R6d receipt")
    if prereg.get("preregistration_sha256") != R6D_PREREGISTRATION_LOGICAL_SHA256:
        raise DirectActorError("R6d logical preregistration SHA256 mismatch")
    if receipt.get("status") != "completed_simulation_interface_smoke":
        raise DirectActorError("R6d receipt is not the completed interface smoke")
    required_false = (
        "real_robot_execution",
        "fresh_inputs_used",
        "fresh_seed_manifest_opened",
        "fresh_trajectory_or_label_opened",
        "policy_forward_performed",
        "task_success_claimed",
        "transfer_claim_authorized",
        "performance_evaluation_authorized",
    )
    if any(receipt.get(key) is not False for key in required_false):
        raise DirectActorError("R6d capability/Fresh boundary changed")
    recorded_prereg = receipt.get("preregistration", {})
    if (
        recorded_prereg.get("file_sha256") != expected_preregistration_sha256
        or recorded_prereg.get("logical_sha256") != R6D_PREREGISTRATION_LOGICAL_SHA256
        or receipt.get("implementation_sha256") != expected_executor_sha256
    ):
        raise DirectActorError("R6d receipt does not bind preregistration/executor")
    environment = receipt.get("environment_contract", {})
    if (
        environment.get("embodiment") != ["piper", "piper", 0.6]
        or environment.get("center_crop") is not False
        or environment.get("collect_wrist_camera") is not True
        or environment.get("action_horizon_per_env_step") != 1
        or environment.get("observation_state_is_measured_qpos") is not False
    ):
        raise DirectActorError("R6d environment interface contract changed")
    execution = receipt.get("execution", {})
    if execution.get("all_steps_prevalidated") is not True or execution.get("silent_clipping_possible") is not False:
        raise DirectActorError("R6d fail-closed action evidence changed")
    runtime_sources = receipt.get("runtime_source_artifacts", {})
    if not isinstance(runtime_sources, dict) or set(runtime_sources) != expected_runtime_source_roles:
        raise DirectActorError("R6d runtime source registry changed")
    for role, record in runtime_sources.items():
        source = reject_fresh_path(Path(record["path"]), f"R6d runtime source {role}")
        if file_sha256(source) != record.get("sha256"):
            raise DirectActorError(f"R6d runtime source changed: {role}")
    return {
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": expected_preregistration_sha256,
        "preregistration_logical_sha256": R6D_PREREGISTRATION_LOGICAL_SHA256,
        "receipt_path": str(receipt_file),
        "receipt_sha256": expected_receipt_sha256,
        "executor_sha256": expected_executor_sha256,
        "runtime_source_artifacts": runtime_sources,
        "authorization": "evidence_only_not_direct_actor_execution_authority",
    }


def explicit_named_map_online_chunk(source_chunk: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Map one postprocessed Aloha chunk using names, never width identity."""

    source = np.asarray(source_chunk)
    if source.shape != (CHUNK_SIZE, ACTION_DIM) or source.dtype.kind not in "f":
        raise DirectActorError("postprocessed Aloha chunk must be real float [50,14]")
    if not np.all(np.isfinite(source)):
        raise DirectActorError("postprocessed Aloha chunk is non-finite")
    source_by_name = {name: index for index, name in enumerate(ALOHA_FEATURE_NAMES)}
    target = np.empty_like(source)
    mapping: list[dict[str, Any]] = []
    for target_index, slot in enumerate(PIPER_ACTION_SLOTS):
        source_index = source_by_name[slot.source_feature_name]
        target[:, target_index] = source[:, source_index]
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
        "identity_inferred_from_equal_dimension": False,
        "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False,
        "clipping_or_scaling_applied": False,
    }


def _cpu_numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(result)


def live_policy_input(observation: Mapping[str, Any], image_keys: Sequence[str]) -> dict[str, Any]:
    """Build the exact three-camera/state14 raw input from the current env row."""

    import torch

    if tuple(image_keys) != EXPECTED_IMAGE_KEYS:
        raise DirectActorError("checkpoint image features differ from frozen Aloha camera registry")
    interface = validate_reset_observation(observation)
    main = _cpu_numpy(observation["main_images"])
    wrists = _cpu_numpy(observation["wrist_images"])
    state = _cpu_numpy(observation["states"], dtype=np.float32)
    descriptions = observation.get("task_descriptions")
    if descriptions is None or len(descriptions) != 1 or str(descriptions[0]) != INSTRUCTION:
        raise DirectActorError("current observation instruction differs from explicit instruction")

    def chw(rgb: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)

    return {
        "observation.state": torch.from_numpy(state[0].copy()),
        EXPECTED_IMAGE_KEYS[0]: chw(main[0]),
        EXPECTED_IMAGE_KEYS[1]: chw(wrists[0, 0]),
        EXPECTED_IMAGE_KEYS[2]: chw(wrists[0, 1]),
        "task": INSTRUCTION,
        "_interface": interface,
    }


def candidate0_noise(config: Any, scene_seed: int, query_index: int, device: Any) -> Any:
    """Create the sole deterministic candidate-0 flow noise for one live query."""

    import torch
    from collect_smolvla_etsf_event_branches import make_noise

    return make_noise(config, scene_seed, query_index, 0, device)


def run_online_actor_loop(
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
    noise_factory: Callable[[Any, int, int, Any], Any] = candidate0_noise,
    max_steps: int = DIRECT_MAX_STEPS,
) -> dict[str, Any]:
    """Dependency-injected online loop used by CUDA runner and CPU tests."""

    if type(max_steps) is not int or not 1 <= max_steps <= DIRECT_MAX_STEPS:
        raise DirectActorError("direct-actor max_steps must be in [1,4]")
    if int(getattr(policy.config, "chunk_size", -1)) != CHUNK_SIZE:
        raise DirectActorError("SmolVLA chunk_size must remain 50")
    if int(getattr(policy.config, "max_action_dim", -1)) < ACTION_DIM:
        raise DirectActorError("SmolVLA max_action_dim cannot represent state/action14")
    image_keys = tuple(getattr(policy.config, "image_features", ()))
    queries: list[dict[str, Any]] = []
    current = observation
    success = False
    terminated_seen = False
    truncated_seen = False
    mapping_contract: dict[str, Any] | None = None
    for query_index in range(max_steps):
        raw = live_policy_input(current, image_keys)
        interface = raw.pop("_interface")
        processed = preprocessor(raw)
        if "observation.state" not in processed:
            raise DirectActorError("preprocessor omitted observation.state")
        processed_state = _cpu_numpy(processed["observation.state"], dtype=np.float32)
        if processed_state.shape == (ACTION_DIM,):
            processed_state = processed_state[None, :]
        if processed_state.shape != (1, ACTION_DIM) or not np.all(np.isfinite(processed_state)):
            raise DirectActorError("processed state must be finite [1,14]")
        policy.reset()
        capture.reset()
        noise = noise_factory(policy.config, REQUESTED_DEVELOPMENT_SEED, query_index, device)
        noise_cpu = _cpu_numpy(noise, dtype=np.float32)
        normalized = policy.predict_action_chunk(dict(processed), noise=noise)
        postprocessed = postprocessor(normalized)
        prefix = _cpu_numpy(capture.consume(), dtype=np.float32)
        if prefix.shape != (PREFIX_DIM,) or not np.all(np.isfinite(prefix)):
            raise DirectActorError("shared VLM prefix must be finite [960]")
        chunk_cpu = _cpu_numpy(postprocessed, dtype=np.float32)
        if chunk_cpu.shape == (1, CHUNK_SIZE, ACTION_DIM):
            chunk_cpu = chunk_cpu[0]
        mapped, query_mapping = explicit_named_map_online_chunk(chunk_cpu)
        if mapping_contract is None:
            mapping_contract = query_mapping
        elif mapping_contract != query_mapping:
            raise DirectActorError("named action mapping changed between queries")
        command = np.ascontiguousarray(mapped[0])
        action_check = validate_piper_step(command, bounds, step_index=query_index)
        env_action = action_converter(command.copy())
        echoed = _cpu_numpy(env_action)
        if echoed.shape != (1, ACTION_EXEC_STEPS, ACTION_DIM) or not np.array_equal(echoed.reshape(ACTION_DIM), command):
            raise DirectActorError("action converter changed command; env.step forbidden")
        next_observation, _, terminated, truncated, info = env.step(env_action, auto_reset=False)
        queries.append({
            "query_index": query_index,
            "candidate_index": 0,
            "candidate_noise_seed_role": "fixed_candidate0_query_specific",
            "noise_sha256": array_sha256(noise_cpu),
            "processed_state": processed_state[0].tolist(),
            "processed_state_sha256": array_sha256(processed_state[0]),
            "shared_prefix": prefix.tolist(),
            "shared_prefix_sha256": array_sha256(prefix),
            "postprocessed_source_chunk_sha256": array_sha256(chunk_cpu),
            "mapped_chunk_sha256": array_sha256(mapped),
            "executed_action": command.tolist(),
            "action_validation": action_check,
            "input_interface": interface,
            "action_horizon_per_env_step": ACTION_EXEC_STEPS,
        })
        success = success or _scalar_bool(info.get("success", [False]))
        terminated_seen = _scalar_bool(terminated)
        truncated_seen = _scalar_bool(truncated)
        current = next_observation
        if terminated_seen or truncated_seen:
            break
    return {
        "queries_performed": len(queries),
        "steps_executed": len(queries),
        "max_steps": max_steps,
        "action_exec_steps": ACTION_EXEC_STEPS,
        "stopped_on_termination": terminated_seen,
        "stopped_on_truncation": truncated_seen,
        "success_observed_diagnostic_only": success,
        "all_actions_prevalidated": True,
        "silent_clipping_possible": False,
        "online_forward_from_current_observation": True,
        "precomputed_action_chunks_used": False,
        "mapping_contract": mapping_contract,
        "queries": queries,
    }


def offload_runtime(capture: Any, env: Any, policy: Any, torch_module: Any) -> None:
    """Release the hook, simulator, actor weights, and CUDA cache in that order."""

    failures: list[BaseException] = []
    for operation in (
        capture.close,
        lambda: env.offload(clear_cache=True) if callable(getattr(env, "offload", None)) else env.close(),
        lambda: policy.to("cpu"),
    ):
        try:
            operation()
        except BaseException as exc:  # still attempt every independent release
            failures.append(exc)
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except BaseException as exc:
        failures.append(exc)
    if failures:
        raise DirectActorError(f"runtime offload failed in {len(failures)} operation(s)") from failures[0]


def build_direct_actor_preregistration(
    *, r6c: Mapping[str, Any], r6d: Mapping[str, Any], seed: Mapping[str, Any],
    rlinf_root: Path, robotwin_root: Path, robotwin_code: Path, lerobot_root: Path,
    model_path: Path, vlm_metadata_path: Path, output: Path,
) -> dict[str, Any]:
    """Build an immutable, independent direct-actor simulation authority."""

    roots = {name: reject_fresh_path(path, name) for name, path in {
        "rlinf_root": rlinf_root, "robotwin_root": robotwin_root,
        "robotwin_code": robotwin_code, "lerobot_root": lerobot_root,
        "model_path": model_path, "vlm_metadata_path": vlm_metadata_path,
    }.items()}
    if any(not path.is_dir() for path in roots.values()):
        raise DirectActorError("all runtime/model roots must be directories")
    actor_directory = Path(r6c["receipt"]["static_contract"]["static_semantics"]["actor_directory"]).resolve()
    if roots["model_path"] != actor_directory:
        raise DirectActorError("model_path differs from the R6c-authenticated actor directory")
    state_resolution = r6c["receipt"]["static_contract"]["static_semantics"].get(
        "state_dimension_resolution"
    )
    if state_resolution != EXPECTED_STATE_DIMENSION_RESOLUTION:
        raise DirectActorError("R6c state6/runtime-normalizer14 resolution changed")
    output_path = reject_fresh_path(output, "direct actor receipt")
    if output_path.exists():
        raise FileExistsError(output_path)
    sources = {
        "direct_actor_runner": Path(__file__).resolve(),
        "r6d_base_executor": Path(__file__).with_name("execute_smolvla_piper_r6c_simulation_smoke.py").resolve(),
        "smolvla_modeling": roots["lerobot_root"] / "src/lerobot/policies/smolvla/modeling_smolvla.py",
        "smolvlm_bridge": roots["lerobot_root"] / "src/lerobot/policies/smolvla/smolvlm_with_expert.py",
        "policy_factory": roots["lerobot_root"] / "src/lerobot/policies/factory.py",
        "shared_prefix_capture": Path(__file__).with_name("collect_smolvla_etsf_event_branches.py").resolve(),
        "robotwin_eval_seed_registry": roots["rlinf_root"] / "rlinf/envs/robotwin/seeds/eval_seeds.json",
    }
    if any(not path.is_file() for path in sources.values()):
        raise DirectActorError("a bound direct-actor runtime source is missing")
    if file_sha256(sources["r6d_base_executor"]) != R6D_EXECUTOR_SHA256:
        raise DirectActorError("local base executor differs from R6d")
    base: dict[str, Any] = {
        "format": PREREGISTRATION_FORMAT,
        "status": "preregistered_direct_actor_simulation_only_not_executed",
        "actor_id": ACTOR_ID,
        "explicit_instruction": INSTRUCTION,
        "source_body": "aloha",
        "target_body": "piper",
        "r6c_binding": {key: r6c[key] for key in ("manifest_path", "manifest_sha256", "receipt_path", "receipt_sha256", "verifier_sha256")},
        "r6d_binding": dict(r6d),
        "development_seed": dict(seed),
        "runtime_roots": {key: str(value) for key, value in roots.items()},
        "runtime_source_artifacts": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in sources.items()},
        "vlm_metadata_bundle_sha256": directory_bundle_sha256(roots["vlm_metadata_path"]),
        "model_bundle_sha256": directory_bundle_sha256(roots["model_path"]),
        "output": str(output_path),
        "execution_contract": {
            "max_steps": DIRECT_MAX_STEPS, "action_exec_steps": ACTION_EXEC_STEPS,
            "candidate_index": 0, "online_policy_forward_each_query": True,
            "precomputed_chunks_forbidden": True, "proper_checkpoint_preprocessor": True,
            "proper_checkpoint_postprocessor": True, "shared_prefix_dim": PREFIX_DIM,
            "processed_state_dim": ACTION_DIM, "embodiment": ["piper", "piper", 0.6],
            "center_crop": False, "collect_wrist_camera": True,
            "checkpoint_declared_state_dim": 6,
            "runtime_normalizer_state_dim": ACTION_DIM,
            "checkpoint_action_dim": ACTION_DIM,
            "state_dimension_conflict_retained": True,
            "materialized_regular_file_bundles_only": True,
        },
        "capability_contract": {
            "simulation_execution_authorized": True, "real_robot_execution_authorized": False,
            "fresh_inputs_allowed": False, "fresh_trajectory_or_label_opened": False,
            "performance_evaluation_authorized": False, "task_success_claim_authorized": False,
            "transfer_claim_authorized": False,
        },
        "mapping_contract": {
            "mode": "explicit_named_ordinal_angle_preserving_mapping",
            "derived_from_equal_14d_width": False, "kinematic_equivalence_claimed": False,
            "physical_equivalence_claimed": False, "clipping_or_scaling_forbidden": True,
        },
        "state_contract": {
            "semantics": "[left drive_target q1..q6,left normalized gripper,right drive_target q1..q6,right normalized gripper]",
            "is_measured_qpos": False,
        },
        "caveats": {
            "scene_seed_and_instruction_strictly_bound": False,
            "instruction_guarantee": "explicit string verified on every current observation in one run",
            "reported_duration": "policy row count not physical time",
            "performance_or_transfer_claim": False,
        },
    }
    return {**base, "preregistration_sha256": canonical_sha256(base)}


def validate_direct_actor_preregistration(path: Path) -> dict[str, Any]:
    prereg_path = reject_fresh_path(path, "direct actor preregistration")
    value = _json(prereg_path, "direct actor preregistration")
    expected_fields = {
        "format", "status", "actor_id", "explicit_instruction", "source_body",
        "target_body", "r6c_binding", "r6d_binding", "development_seed",
        "runtime_roots", "runtime_source_artifacts", "vlm_metadata_bundle_sha256",
        "model_bundle_sha256", "output", "execution_contract",
        "capability_contract", "mapping_contract", "state_contract", "caveats",
        "preregistration_sha256",
    }
    if set(value) != expected_fields:
        raise DirectActorError("direct actor preregistration fields changed")
    base = {key: item for key, item in value.items() if key != "preregistration_sha256"}
    if value.get("preregistration_sha256") != canonical_sha256(base):
        raise DirectActorError("direct actor preregistration logical SHA mismatch")
    if value.get("format") != PREREGISTRATION_FORMAT or value.get("status") != "preregistered_direct_actor_simulation_only_not_executed":
        raise DirectActorError("unexpected direct actor preregistration")
    if (
        value.get("actor_id") != ACTOR_ID
        or value.get("explicit_instruction") != INSTRUCTION
        or value.get("source_body") != "aloha"
        or value.get("target_body") != "piper"
    ):
        raise DirectActorError("direct actor identity/language/body contract changed")
    if set(value.get("r6c_binding", {})) != {
        "manifest_path", "manifest_sha256", "receipt_path", "receipt_sha256", "verifier_sha256"
    }:
        raise DirectActorError("R6c binding fields changed")
    if set(value.get("r6d_binding", {})) != {
        "preregistration_path", "preregistration_sha256", "preregistration_logical_sha256",
        "receipt_path", "receipt_sha256", "executor_sha256", "runtime_source_artifacts",
        "authorization",
    } or value["r6d_binding"].get("authorization") != "evidence_only_not_direct_actor_execution_authority":
        raise DirectActorError("R6d evidence-only binding changed")
    expected_capability = {
        "simulation_execution_authorized": True, "real_robot_execution_authorized": False,
        "fresh_inputs_allowed": False, "fresh_trajectory_or_label_opened": False,
        "performance_evaluation_authorized": False, "task_success_claim_authorized": False,
        "transfer_claim_authorized": False,
    }
    if value.get("capability_contract") != expected_capability:
        raise DirectActorError("direct actor capability/Fresh boundary changed")
    expected_execution = {
        "max_steps": DIRECT_MAX_STEPS, "action_exec_steps": ACTION_EXEC_STEPS,
        "candidate_index": 0, "online_policy_forward_each_query": True,
        "precomputed_chunks_forbidden": True, "proper_checkpoint_preprocessor": True,
        "proper_checkpoint_postprocessor": True, "shared_prefix_dim": PREFIX_DIM,
        "processed_state_dim": ACTION_DIM, "embodiment": ["piper", "piper", 0.6],
        "center_crop": False, "collect_wrist_camera": True,
        "checkpoint_declared_state_dim": 6,
        "runtime_normalizer_state_dim": ACTION_DIM,
        "checkpoint_action_dim": ACTION_DIM,
        "state_dimension_conflict_retained": True,
        "materialized_regular_file_bundles_only": True,
    }
    if value.get("execution_contract") != expected_execution:
        raise DirectActorError("direct actor short-smoke contract changed")
    if value.get("mapping_contract") != {
        "mode": "explicit_named_ordinal_angle_preserving_mapping",
        "derived_from_equal_14d_width": False, "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False, "clipping_or_scaling_forbidden": True,
    }:
        raise DirectActorError("direct actor named mapping contract changed")
    if value.get("state_contract") != {
        "semantics": "[left drive_target q1..q6,left normalized gripper,right drive_target q1..q6,right normalized gripper]",
        "is_measured_qpos": False,
    }:
        raise DirectActorError("direct actor state semantics changed")
    if value.get("caveats") != {
        "scene_seed_and_instruction_strictly_bound": False,
        "instruction_guarantee": "explicit string verified on every current observation in one run",
        "reported_duration": "policy row count not physical time",
        "performance_or_transfer_claim": False,
    }:
        raise DirectActorError("direct actor disclosure contract changed")
    for role, record in value.get("runtime_source_artifacts", {}).items():
        if file_sha256(reject_fresh_path(Path(record["path"]), role)) != record.get("sha256"):
            raise DirectActorError(f"direct actor source changed: {role}")
    roots = value["runtime_roots"]
    if set(roots) != {
        "rlinf_root", "robotwin_root", "robotwin_code", "lerobot_root",
        "model_path", "vlm_metadata_path",
    }:
        raise DirectActorError("direct actor runtime roots changed")
    reject_fresh_path(Path(value["output"]), "direct actor receipt output")
    if directory_bundle_sha256(Path(roots["model_path"])) != value.get("model_bundle_sha256"):
        raise DirectActorError("model bundle changed")
    if directory_bundle_sha256(Path(roots["vlm_metadata_path"])) != value.get("vlm_metadata_bundle_sha256"):
        raise DirectActorError("VLM metadata bundle changed")
    return value


def _load_and_recompute_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prereg = validate_direct_actor_preregistration(path)
    r6c_record, r6d_record, seed_record = prereg["r6c_binding"], prereg["r6d_binding"], prereg["development_seed"]
    r6c = bind_r6c_preflight(Path(r6c_record["manifest_path"]), Path(r6c_record["receipt_path"]))
    r6d = bind_r6d_simulation_receipt(Path(r6d_record["preregistration_path"]), Path(r6d_record["receipt_path"]))
    seed = bind_development_seed_manifest(Path(seed_record["path"]))
    roots = {key: Path(value) for key, value in prereg["runtime_roots"].items()}
    expected = build_direct_actor_preregistration(
        r6c=r6c, r6d=r6d, seed=seed, rlinf_root=roots["rlinf_root"],
        robotwin_root=roots["robotwin_root"], robotwin_code=roots["robotwin_code"],
        lerobot_root=roots["lerobot_root"], model_path=roots["model_path"],
        vlm_metadata_path=roots["vlm_metadata_path"], output=Path(prereg["output"]),
    )
    if prereg != expected:
        raise DirectActorError("preregistration differs from fully recomputed contract")
    return prereg, r6c, r6d, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    prereg, r6c, r6d, seed = _load_and_recompute_preregistration(args.preregistration)
    output = reject_fresh_path(Path(prereg["output"]), "direct actor receipt")
    if output.exists():
        raise FileExistsError(output)
    roots = {key: Path(value) for key, value in prereg["runtime_roots"].items()}
    os.environ["ASSETS_PATH"] = str(roots["robotwin_root"])
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    # The support tree must win both ``robotwin`` and top-level ``envs`` imports.
    # RLinf remains available for ``rlinf.*`` and LeRobot for the actor.
    sys.path[:0] = [str(roots["robotwin_code"]), str(roots["rlinf_root"]), str(roots["lerobot_root"] / "src")]

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from collect_smolvla_etsf_event_branches import resolve_shared_prefix_capture

    runtime_module_origins = validate_runtime_module_origins(
        {
            "rlinf_robotwin_env": importlib.import_module(
                "rlinf.envs.robotwin.robotwin_env"
            ),
            "robotwin_vector_env": importlib.import_module(
                "robotwin.envs.vector_env"
            ),
            "robotwin_base_task": importlib.import_module("envs._base_task"),
            "robotwin_robot_controller": importlib.import_module("envs.robot.robot"),
        },
        r6d["runtime_source_artifacts"],
    )

    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise DirectActorError("direct actor smoke must run on the designated RTX 4090")
    device = torch.device("cuda:0")
    config = PreTrainedConfig.from_pretrained(roots["model_path"], local_files_only=True)
    config.device = str(device)
    config.vlm_model_name = str(roots["vlm_metadata_path"])
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(roots["model_path"], config=config, local_files_only=True, strict=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(roots["model_path"]),
        preprocessor_overrides={"device_processor": {"device": str(device)}, "tokenizer_processor": {"tokenizer_name": str(roots["vlm_metadata_path"])}},
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
        frozen_seeds = prereg["runtime_source_artifacts"]["robotwin_eval_seed_registry"]
        if (
            not seeds_path.is_file()
            or seeds_path.resolve() != Path(frozen_seeds["path"]).resolve()
            or file_sha256(seeds_path) != frozen_seeds["sha256"]
        ):
            raise DirectActorError("RoboTwin evaluation seed registry changed")
        env_cfg = piper_environment_config(roots["robotwin_root"], seeds_path, output.parent, step_limit=DIRECT_MAX_STEPS)
        env = RoboTwinEnv(cfg=OmegaConf.create(env_cfg), num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None, record_metrics=True)
        observation, _ = reset_with_explicit_instruction(env, requested_seed=REQUESTED_DEVELOPMENT_SEED, instruction=INSTRUCTION)
        resolved_seed = int(env.venv.envs[0].task.ep_num)
        if resolved_seed != EXPECTED_RESOLVED_DEVELOPMENT_SEED:
            raise DirectActorError("resolved development seed changed")
        bounds = r6c["receipt"]["static_contract"]["static_semantics"]["piper_action_bounds"]

        def to_cuda(command: np.ndarray) -> Any:
            return torch.from_numpy(command.reshape(1, 1, ACTION_DIM)).to(device=device, dtype=torch.float32)

        execution = run_online_actor_loop(
            env=env, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
            capture=capture, observation=observation, bounds=bounds, device=device,
            action_converter=to_cuda, max_steps=DIRECT_MAX_STEPS,
        )
        result = {
            "format": FORMAT, "status": "completed_direct_actor_simulation_interface_smoke",
            "actor_id": ACTOR_ID, "source_body": "aloha", "target_body": "piper",
            "target_runtime": "RoboTwin_simulation_only", "real_robot_execution": False,
            "fresh_inputs_used": False, "fresh_trajectory_or_label_opened": False,
            "policy_forward_performed_online": True, "task_success_claimed": False,
            "performance_evaluation_authorized": False, "transfer_claim_authorized": False,
            "preregistration": {"path": str(args.preregistration.resolve()), "file_sha256": file_sha256(args.preregistration), "logical_sha256": prereg["preregistration_sha256"]},
            "r6c_binding": prereg["r6c_binding"], "r6d_binding": r6d,
            "development_seed_contract": seed,
            "environment_contract": {
                "embodiment": ["piper", "piper", 0.6], "requested_seed": REQUESTED_DEVELOPMENT_SEED,
                "resolved_seed": resolved_seed, "explicit_instruction": INSTRUCTION,
                "scene_seed_and_instruction_strictly_bound": False, "center_crop": False,
                "collect_wrist_camera": True, "state_is_measured_qpos": False,
                "state_semantics": prereg["state_contract"]["semantics"],
                "runtime_module_origins": runtime_module_origins,
                "eval_seed_registry": dict(frozen_seeds),
            },
            "loaded_policy_contract": loaded_policy_contract,
            "time_contract": {"unit": "policy action row count", "physical_duration_claimed": False},
            "execution": execution, "implementation_sha256": file_sha256(Path(__file__)),
            "interpretation": "online direct-actor simulator interface smoke only; no performance, success, transfer, safety, or real-robot claim",
        }
        atomic_json(output, result)
    finally:
        if env is not None:
            offload_runtime(capture, env, policy, torch)
        else:
            capture.close()
            policy.to("cpu")
            torch.cuda.empty_cache()
    print(json.dumps({"status": "completed_direct_actor_simulation_interface_smoke", "steps_executed": execution["steps_executed"], "task_success_claimed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
