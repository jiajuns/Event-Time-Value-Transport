#!/usr/bin/env python3
"""Bound RoboTwin/Piper runtime factories for Schema6 multi-seed Phase-2.

Both the collection and reset-only factories share the same reset and identity
reader.  Runtime roots, model/VLM bundles, simulator sources, action bounds,
measured-qpos channel and eval-seed registry come from a signed runtime contract;
no server path is hard-coded here.  Heavy simulator/policy imports occur only
after that contract has been validated by the caller.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


RUNTIME_CONTRACT_FORMAT = "etsf_smolvla_piper_schema6_runtime_contract_v2"
RUNTIME_CONTRACT_STATUS = "frozen_runtime_and_identity_channels_before_reset_or_policy"
RUNTIME_ROOT_KEYS = {
    "rlinf_root", "robotwin_root", "robotwin_code", "lerobot_root",
    "model_path", "vlm_metadata_path",
}
RUNTIME_CONTRACT_KEYS = {
    "format", "status", "runtime_roots", "runtime_source_artifacts",
    "eval_seed_registry", "measured_joint_state_channel", "gpu_index",
    "max_episode_steps", "offline_model_loading", "piper_action_bounds",
    "model_tree_sha256", "vlm_metadata_tree_sha256", "reset_scratch_path",
    "test_or_evaluation_execution_authorized",
    "fresh_or_confirmation_inputs_accepted", "runtime_contract_sha256",
}
RUNTIME_SOURCE_KEYS = {
    "rlinf_robotwin_env", "robotwin_vector_env", "robotwin_base_task",
    "robotwin_robot_controller", "robotwin_move_can_pot",
}
DEFAULT_MEASURED_CHANNEL = (
    "task.robot.get_left_arm_real_jointState+"
    "get_right_arm_real_jointState"
)
MEASURED_CHANNELS = {DEFAULT_MEASURED_CHANNEL}
ACTION_DIM = 14
INSTRUCTION = "move the can into the pot"
TASK = "move_can_pot"
SHA_CHARS = frozenset("0123456789abcdef")


class RuntimeAdapterError(RuntimeError):
    """The bound simulator, policy, or identity interface changed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_tree_sha256(path: Path) -> str:
    """Hash every regular file and its relative name; reject mutable indirection."""

    root = path.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeAdapterError(f"runtime bundle is not a real directory: {path}")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise RuntimeAdapterError(f"runtime bundle contains a symlink: {relative}")
        mode = candidate.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeAdapterError(f"runtime bundle contains a non-file: {relative}")
        rows.append({"path": relative, "sha256": file_sha256(candidate)})
    if not rows:
        raise RuntimeAdapterError(f"runtime bundle is empty: {path}")
    return canonical_sha256(rows)


def _safe_scratch(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise RuntimeAdapterError("reset scratch path is invalid")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeAdapterError("reset scratch path must be absolute")
    resolved = path.resolve(strict=False)
    forbidden = {"evaluation", "test", "testing", "fresh", "confirmation"}
    if any(part.casefold() in forbidden for part in resolved.parts):
        raise RuntimeAdapterError("reset scratch path enters a forbidden namespace")
    if path.is_symlink():
        raise RuntimeAdapterError("reset scratch path must not be a symlink")
    return str(resolved)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def validate_runtime_contract(value: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    unsigned = dict(value)
    logical = unsigned.pop("runtime_contract_sha256", None)
    roots = value.get("runtime_roots")
    sources = value.get("runtime_source_artifacts")
    seeds = value.get("eval_seed_registry")
    if (
        set(value) != RUNTIME_CONTRACT_KEYS
        or
        not _is_sha(logical)
        or logical != canonical_sha256(unsigned)
        or value.get("format") != RUNTIME_CONTRACT_FORMAT
        or value.get("status") != RUNTIME_CONTRACT_STATUS
        or not isinstance(roots, Mapping)
        or set(roots) != RUNTIME_ROOT_KEYS
        or not isinstance(sources, Mapping)
        or set(sources) != RUNTIME_SOURCE_KEYS
        or not isinstance(seeds, Mapping)
        or set(seeds) != {"path", "sha256"}
        or value.get("measured_joint_state_channel") not in MEASURED_CHANNELS
        or value.get("gpu_index") != 0
        or type(value.get("max_episode_steps")) is not int
        or value["max_episode_steps"] < 1
        or value.get("offline_model_loading") is not True
        or value.get("test_or_evaluation_execution_authorized") is not False
        or value.get("fresh_or_confirmation_inputs_accepted") is not False
        or not _is_sha(value.get("model_tree_sha256"))
        or not _is_sha(value.get("vlm_metadata_tree_sha256"))
    ):
        raise RuntimeAdapterError("runtime contract schema/scope changed")
    bounds = np.asarray(value.get("piper_action_bounds"), dtype=np.float64)
    if bounds.shape != (ACTION_DIM, 2) or not np.isfinite(bounds).all() or bool((bounds[:, 0] >= bounds[:, 1]).any()):
        raise RuntimeAdapterError("Piper action bounds are invalid")
    decoded_roots = {key: str(Path(str(path)).resolve()) for key, path in roots.items()}
    reset_scratch = _safe_scratch(value.get("reset_scratch_path"))
    if verify_files:
        for key in ("rlinf_root", "robotwin_root", "robotwin_code", "lerobot_root", "model_path", "vlm_metadata_path"):
            if not Path(decoded_roots[key]).is_dir():
                raise RuntimeAdapterError(f"runtime root is unavailable: {key}")
        for role, row in {**dict(sources), "eval_seed_registry": seeds}.items():
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
                raise RuntimeAdapterError(f"runtime artifact fields changed: {role}")
            path = Path(str(row["path"])).resolve()
            if not path.is_file() or path.is_symlink() or file_sha256(path) != row["sha256"]:
                raise RuntimeAdapterError(f"runtime artifact SHA changed: {role}")
        if directory_tree_sha256(Path(decoded_roots["model_path"])) != value["model_tree_sha256"]:
            raise RuntimeAdapterError("SmolVLA model directory tree changed")
        if directory_tree_sha256(Path(decoded_roots["vlm_metadata_path"])) != value["vlm_metadata_tree_sha256"]:
            raise RuntimeAdapterError("VLM metadata directory tree changed")
    return {
        **dict(value), "runtime_roots": decoded_roots,
        "reset_scratch_path": reset_scratch, "runtime_contract_sha256": logical,
    }


def _pose_vector(actor: Any, role: str) -> list[float]:
    try:
        pose = actor.get_pose() if callable(getattr(actor, "get_pose", None)) else actor.pose
        vector = np.r_[np.asarray(pose.p), np.asarray(pose.q)].astype(np.float64)
    except Exception as error:
        raise RuntimeAdapterError(f"cannot read {role} pose") from error
    if vector.shape != (7,) or not np.isfinite(vector).all():
        raise RuntimeAdapterError(f"{role} pose is not finite xyz+quaternion")
    return vector.tolist()


def _finite14(value: Any, role: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        raise RuntimeAdapterError(f"{role} must be a finite 14-vector")
    return np.ascontiguousarray(array)


def measured_joint_state(task: Any, channel: str) -> np.ndarray:
    if channel != DEFAULT_MEASURED_CHANNEL:
        raise RuntimeAdapterError("unbound measured joint-state channel")
    robot = getattr(task, "robot", None)
    left_getter = getattr(robot, "get_left_arm_real_jointState", None)
    right_getter = getattr(robot, "get_right_arm_real_jointState", None)
    if not callable(left_getter) or not callable(right_getter):
        raise RuntimeAdapterError("RoboTwin real left/right joint-state readers are unavailable")
    left = np.asarray(left_getter(), dtype=np.float64)
    right = np.asarray(right_getter(), dtype=np.float64)
    if left.shape != (7,) or right.shape != (7,):
        raise RuntimeAdapterError("RoboTwin real left/right joint states must each be 7-vectors")
    return _finite14(np.concatenate([left, right]), "measured joint state")


def identity_snapshot(task: Any, observation: Mapping[str, Any], measured_channel: str) -> dict[str, Any]:
    from execute_smolvla_piper_r6c_simulation_smoke import validate_reset_observation

    interface = validate_reset_observation(observation)
    return {
        "scene_state": {
            "can_pose": _pose_vector(getattr(task, "can", None), "can"),
            "pot_pose": _pose_vector(getattr(task, "pot", None), "pot"),
        },
        "measured_joint_state": measured_joint_state(task, measured_channel),
        "commanded_drive_target": _finite14(interface["drive_target"], "commanded drive target"),
    }


class DynamicRoboTwinCollectionRuntime:
    """r6j-compatible runtime with per-reset dynamic can/pot identities."""

    def __init__(
        self, *, env: Any, torch_module: Any, device: Any,
        bounds: list[list[float]], measured_channel: str,
        raw_state: Callable[[Any], np.ndarray], derive_events: Callable[..., Any],
        clock_factory: Callable[..., Any], validate_step: Callable[..., Any],
        reset_explicit: Callable[..., Any],
    ) -> None:
        self.env = env
        self.torch = torch_module
        self.device = device
        self.bounds = bounds
        self.measured_channel = measured_channel
        self.raw_state = raw_state
        self.derive_events_fn = derive_events
        self.clock_factory = clock_factory
        self.validate_step = validate_step
        self.reset_explicit = reset_explicit
        self.live_task: Any | None = None
        self.last_observation: Mapping[str, Any] | None = None
        self.objects: list[Any] = []
        self.clock: Any | None = None
        self.env_steps = 0
        self.reset_generation = -1

    def reset(self, seed: int, instruction: str):
        if self.clock is not None:
            self.clock.close()
        observation, _ = self.reset_explicit(
            self.env, requested_seed=seed, instruction=instruction
        )
        subenvs = getattr(getattr(self.env, "venv", None), "envs", None)
        if not isinstance(subenvs, list) or len(subenvs) != 1:
            raise RuntimeAdapterError("RoboTwin must expose exactly one subenv")
        self.live_task = subenvs[0].task
        resolved = int(self.live_task.ep_num)
        descriptions = observation.get("task_descriptions")
        observed = str(descriptions[0]) if descriptions is not None and len(descriptions) == 1 else ""
        self.objects = [getattr(self.live_task, "can", None), getattr(self.live_task, "pot", None)]
        if any(item is None for item in self.objects):
            raise RuntimeAdapterError("live task lacks can/pot actors")
        scene = getattr(self.live_task, "scene", None)
        if scene is None:
            raise RuntimeAdapterError("live task lacks simulator scene")
        self.clock = self.clock_factory(scene, object_count=2)
        self.reset_generation += 1
        self.last_observation = observation
        return observation, resolved, observed

    def task(self):
        if self.live_task is None:
            raise RuntimeAdapterError("task requested before reset")
        return self.live_task

    def identity_snapshot(self):
        if self.live_task is None or self.last_observation is None:
            raise RuntimeAdapterError("identity requested before reset")
        return identity_snapshot(self.live_task, self.last_observation, self.measured_channel)

    def snapshot(self):
        if self.live_task is None or self.clock is None:
            raise RuntimeAdapterError("snapshot requested before reset")
        poses = np.asarray([_pose_vector(item, "object") for item in self.objects], dtype=np.float64)
        proprio = np.asarray(self.raw_state(self.live_task), dtype=np.float32)
        if proprio.shape != (ACTION_DIM,) or not np.isfinite(proprio).all():
            raise RuntimeAdapterError("runtime proprio is not finite state14")
        return {
            "object_names": ["can", "pot"],
            "object_poses": poses,
            "proprio": proprio,
            "telemetry": {
                **self.clock.telemetry(pose_error=np.zeros(2, dtype=bool)),
                "reset_generation": self.reset_generation,
            },
        }

    def step(self, action: Any):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (1, 1, ACTION_DIM):
            raise RuntimeAdapterError("collector action must be [1,1,14]")
        self.validate_step(command.reshape(ACTION_DIM), self.bounds, step_index=self.env_steps)
        device_action = self.torch.from_numpy(command.copy()).to(
            device=self.device, dtype=self.torch.float32
        )
        if not np.array_equal(device_action.detach().cpu().numpy(), command):
            raise RuntimeAdapterError("CUDA action conversion changed command")
        observation, _, terminated, truncated, info = self.env.step(
            device_action, auto_reset=False
        )
        self.last_observation = observation
        self.env_steps += 1
        self.clock.after_policy_step()

        def scalar(value: Any) -> bool:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return bool(np.asarray(value).reshape(-1)[0])

        return observation, scalar(terminated), scalar(truncated), {
            "success": scalar(info.get("success", [False]))
        }

    def derive_events(self, poses, names, success, event_spec):
        return self.derive_events_fn(poses, names, success, event_spec, TASK)

    def mapping(self) -> dict[str, Callable[..., Any]]:
        return {
            "reset": self.reset,
            "identity_snapshot": self.identity_snapshot,
            "task": self.task,
            "snapshot": self.snapshot,
            "step": self.step,
            "derive_events": self.derive_events,
        }

    def close(self) -> None:
        if self.clock is not None:
            self.clock.close()
            self.clock = None


def _load_authority_runtime_contract() -> dict[str, Any]:
    raw_path = os.environ.get("ETSF_SCHEMA6_V2_EXECUTION_AUTHORITY")
    if not raw_path:
        raise RuntimeAdapterError("execution authority environment is absent")
    value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("runtime_contract"), Mapping):
        raise RuntimeAdapterError("execution authority lacks runtime_contract")
    return validate_runtime_contract(value["runtime_contract"])


def _configure_imports(contract: Mapping[str, Any]) -> None:
    roots = contract["runtime_roots"]
    os.environ["ASSETS_PATH"] = roots["robotwin_root"]
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [roots["robotwin_code"], roots["rlinf_root"], str(Path(roots["lerobot_root"]) / "src")]


def _build_resources(
    contract: Mapping[str, Any], *, output_parent: Path, load_policy: bool
) -> dict[str, Any]:
    _configure_imports(contract)
    import torch
    from collect_openvla_etsf_rollouts import derive_events, raw_state
    from execute_smolvla_piper_r6c_simulation_smoke import (
        piper_environment_config,
        reset_with_explicit_instruction,
    )
    from launch_smolvla_piper_schema6_development_collection import DecisionTelemetryClock
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from run_smolvla_piper_r6d_direct_actor_smoke import (
        offload_runtime,
        validate_loaded_policy_contract,
        validate_piper_step,
        validate_runtime_module_origins,
    )

    expected_sources = contract["runtime_source_artifacts"]
    modules = {
        "rlinf_robotwin_env": importlib.import_module("rlinf.envs.robotwin.robotwin_env"),
        "robotwin_vector_env": importlib.import_module("robotwin.envs.vector_env"),
        "robotwin_base_task": importlib.import_module("envs._base_task"),
        "robotwin_robot_controller": importlib.import_module("envs.robot.robot"),
        "robotwin_move_can_pot": importlib.import_module("envs.move_can_pot"),
    }
    validate_runtime_module_origins(modules, expected_sources)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeAdapterError("runtime requires the designated RTX 4090")
    device = torch.device("cuda:0")
    roots = contract["runtime_roots"]
    config = piper_environment_config(
        Path(roots["robotwin_root"]),
        Path(contract["eval_seed_registry"]["path"]),
        output_parent,
        step_limit=int(contract["max_episode_steps"]),
    )
    env = RoboTwinEnv(
        cfg=OmegaConf.create(config), num_envs=1, seed_offset=0,
        total_num_processes=1, worker_info=None, record_metrics=True,
    )
    policy = preprocessor = postprocessor = capture = None
    if load_policy:
        from collect_smolvla_etsf_event_branches import resolve_shared_prefix_capture
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy_config = PreTrainedConfig.from_pretrained(
            roots["model_path"], local_files_only=True
        )
        policy_config.device = str(device)
        policy_config.vlm_model_name = roots["vlm_metadata_path"]
        policy_config.load_vlm_weights = False
        policy = SmolVLAPolicy.from_pretrained(
            roots["model_path"], config=policy_config,
            local_files_only=True, strict=True,
        ).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=roots["model_path"],
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "tokenizer_processor": {"tokenizer_name": roots["vlm_metadata_path"]},
            },
        )
        validate_loaded_policy_contract(
            config=policy_config, preprocessor=preprocessor,
            postprocessor=postprocessor, torch_module=torch, device=device,
        )
        capture = resolve_shared_prefix_capture(policy)
    runtime = DynamicRoboTwinCollectionRuntime(
        env=env,
        torch_module=torch,
        device=device,
        bounds=contract["piper_action_bounds"],
        measured_channel=contract["measured_joint_state_channel"],
        raw_state=raw_state,
        derive_events=derive_events,
        clock_factory=DecisionTelemetryClock,
        validate_step=validate_piper_step,
        reset_explicit=reset_with_explicit_instruction,
    )

    def close() -> None:
        runtime.close()
        if load_policy:
            offload_runtime(capture, env, policy, torch)
        elif callable(getattr(env, "offload", None)):
            env.offload(clear_cache=True)
        else:
            env.close()

    return {
        "runtime": runtime,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "capture": capture,
        "device": device,
        "close": close,
    }


def build_runtime(command: Mapping[str, Any], event_spec: Mapping[str, Any]) -> dict[str, Any]:
    contract = _load_authority_runtime_contract()
    if command.get("split") not in {"adaptation", "validation"}:
        raise RuntimeAdapterError("runtime command split is forbidden")
    if not isinstance(event_spec, Mapping):
        raise RuntimeAdapterError("event spec must be a mapping")
    seed_root = Path(str(command["outputs"]["seed_root"]))
    resources = _build_resources(contract, output_parent=seed_root.parent, load_policy=True)
    from collect_smolvla_piper_schema6_dense_event_branches import generate_candidate_query

    runtime = resources["runtime"]

    def query_fn(observation: Mapping[str, Any], query_index: int):
        return generate_candidate_query(
            policy=resources["policy"],
            preprocessor=resources["preprocessor"],
            postprocessor=resources["postprocessor"],
            capture=resources["capture"],
            observation=observation,
            bounds=contract["piper_action_bounds"],
            device=resources["device"],
            scene_seed=int(command["requested_seed"]),
            query_index=query_index,
        )

    return {
        "runtime": runtime.mapping(),
        "query_fn": query_fn,
        "max_steps": int(contract["max_episode_steps"]),
        "close": resources["close"],
    }


class ResetOnlyAdapter:
    def __init__(self, runtime: DynamicRoboTwinCollectionRuntime, close: Callable[[], None]) -> None:
        self.runtime = runtime
        self._close = close

    def reset_once(self, requested_seed: int, instruction: str) -> dict[str, Any]:
        observation, resolved, observed = self.runtime.reset(requested_seed, instruction)
        if resolved != requested_seed:
            # Stock RoboTwin can internally advance after an unstable setup.  It
            # cannot be selected as the requested identity; expose no retry
            # identity/state and let the preregistered candidate loop advance.
            return {"setup_status": "unstable", "requested_seed": requested_seed}
        snapshot = self.runtime.identity_snapshot()
        return {
            "setup_status": "stable",
            "requested_seed": requested_seed,
            "resolved_seed": resolved,
            "instruction_observed": observed,
            **snapshot,
        }

    def close(self) -> None:
        self._close()


def build_reset_only_adapter(*, plan: Mapping[str, Any], authorization: Mapping[str, Any]) -> ResetOnlyAdapter:
    if not isinstance(authorization.get("runtime_contract"), Mapping):
        raise RuntimeAdapterError(
            "reset authorization lacks signed runtime_contract; v1 SHA-only authority is insufficient"
        )
    contract = validate_runtime_contract(authorization["runtime_contract"])
    resources = _build_resources(
        contract,
        output_parent=Path(contract["reset_scratch_path"]),
        load_policy=False,
    )
    return ResetOnlyAdapter(resources["runtime"], resources["close"])


__all__ = [
    "DEFAULT_MEASURED_CHANNEL", "DynamicRoboTwinCollectionRuntime", "RUNTIME_CONTRACT_FORMAT",
    "RUNTIME_CONTRACT_STATUS", "ResetOnlyAdapter", "RuntimeAdapterError",
    "build_reset_only_adapter", "build_runtime", "identity_snapshot",
    "directory_tree_sha256", "measured_joint_state", "validate_runtime_contract",
]
