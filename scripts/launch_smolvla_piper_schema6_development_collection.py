#!/usr/bin/env python3
"""Launch one frozen non-Fresh Piper schema-v6 development collection on 4090."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from collect_smolvla_piper_schema6_dense_event_branches import (
    bind_r6f_collection_runtime,
    collect_dense_group,
    generate_candidate_query,
    save_schema6_group,
    validate_schema6_group_file,
)
from freeze_smolvla_piper_schema6_development_collection import (
    FIXED_DEVELOPMENT_SEED,
    INSTRUCTION,
    TASK,
    CollectionAuthorityError,
    validate_collection_authority,
)
from materialize_smolvla_piper_schema6_reset_contract import (
    assert_runtime_registry_identity,
    validate_materialized_registry_contract,
)
from run_smolvla_piper_r6d_direct_actor_smoke import (
    ACTION_DIM,
    EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    atomic_json,
    canonical_sha256,
    file_sha256,
    offload_runtime,
    piper_environment_config,
    reject_fresh_path,
    reset_with_explicit_instruction,
    validate_loaded_policy_contract,
    validate_piper_step,
    validate_runtime_module_origins,
)


EXIT_SUCCESS = 0
EXIT_ROOT_INSUFFICIENT = 20
EXIT_FAILURE = 21
RECEIPT_FORMAT = "smolvla_piper_schema6_development_collection_receipt_v1"
MANIFEST_FORMAT = "smolvla_piper_schema6_development_collection_manifest_v1"


class LauncherContractError(RuntimeError):
    """The remote runtime cannot satisfy the frozen collection contract."""


class DecisionTelemetryClock:
    """Count the real RoboTwin ``scene.step`` calls between saved samples.

    The audited SAPIEN 3.0.0b1 scene exposes ``step`` and ``get_timestep`` but
    no cumulative simulation clock.  RoboTwin's frozen TOPP loop invokes that
    Python-visible ``scene.step`` once for every real physics iteration.  A
    per-reset, per-instance wrapper therefore observes the exact calls without
    estimating them from a controller constant or from wall time.
    """

    def __init__(self, scene: Any, *, object_count: int) -> None:
        self.object_count = int(object_count)
        self.control_step = 0
        self._patched_scene = None
        self._original_step = None
        self._instrumented_step = None
        self._install_exact_scene_step_counter(scene)
        self.last_timestamp = self._read_timestamp()
        self.last_counter = self._read_counter()

    def _install_exact_scene_step_counter(self, scene: Any) -> None:
        """Count actual ``scene.step`` calls and sum the simulator timestep.

        SAPIEN's public API exposes the timestep but not a cumulative counter.
        RoboTwin calls the Python wrapper ``Scene.step`` for every physics
        substep, so a reversible instance hook gives an exact counter.  A scene
        that does not permit the audited instance assignment is rejected.
        """

        scene_type = type(scene)
        original = getattr(scene, "step", None)
        timestep_reader = getattr(scene, "get_timestep", None)
        if not callable(original) or not callable(timestep_reader):
            raise LauncherContractError(
                "simulator lacks native clock/counter and an instrumentable scene.step/get_timestep"
            )
        frozen_timestep = float(timestep_reader())
        if not math.isfinite(frozen_timestep) or frozen_timestep <= 0:
            raise LauncherContractError("simulator timestep is invalid")
        counter = {"steps": 0, "since_sample": 0}

        def instrumented(*args, **kwargs):
            dt = float(timestep_reader())
            if not math.isfinite(dt) or dt <= 0:
                raise LauncherContractError("simulator timestep is invalid")
            if dt != frozen_timestep:
                raise LauncherContractError("simulator timestep changed within one reset")
            counter["steps"] += 1
            counter["since_sample"] += 1
            return original(*args, **kwargs)

        try:
            setattr(scene, "step", instrumented)
        except (AttributeError, TypeError) as exc:
            raise LauncherContractError(
                "simulator scene.step cannot be instrumented for exact substep telemetry"
            ) from exc
        if getattr(scene, "step", None) is not instrumented:
            raise LauncherContractError("simulator scene.step instrumentation did not bind")
        self._patched_scene = scene
        self._original_step = original
        self._instrumented_step = instrumented
        self.timestamp_origin = f"instrumented_{scene_type.__module__}.{scene_type.__qualname__}.step_count_times_get_timestep"
        self.counter_origin = f"instrumented_{scene_type.__module__}.{scene_type.__qualname__}.step_call_count"
        self._timestamp = lambda: counter["steps"] * frozen_timestep
        self._counter = lambda: counter["steps"]
        self._since_sample = counter

    def _read_timestamp(self) -> float:
        value = float(self._timestamp())
        if not math.isfinite(value) or value < 0:
            raise LauncherContractError("simulator timestamp is invalid")
        return value

    def _read_counter(self) -> int:
        raw = self._counter()
        if isinstance(raw, bool) or int(raw) != raw or int(raw) < 0:
            raise LauncherContractError("physics-substep counter is invalid")
        return int(raw)

    def after_policy_step(self) -> None:
        timestamp = self._read_timestamp()
        counter = self._read_counter()
        substeps = counter - self.last_counter
        if timestamp <= self.last_timestamp or substeps < 1:
            raise LauncherContractError("simulator clock/counter did not advance after env.step")
        self.control_step += 1
        self.last_timestamp = timestamp
        self.last_counter = counter

    def telemetry(self, *, pose_error: np.ndarray) -> dict[str, Any]:
        errors = np.asarray(pose_error, dtype=bool)
        if errors.shape != (self.object_count,):
            raise LauncherContractError("pose error flags do not align with object registry")
        timestamp = self._read_timestamp()
        counter = self._read_counter()
        substeps = int(self._since_sample["since_sample"])
        if counter != int(self._since_sample["steps"]):
            raise LauncherContractError("instrumented physics counter is internally inconsistent")
        if self.control_step == 0 and (timestamp != 0.0 or counter != 0 or substeps != 0):
            raise LauncherContractError("root telemetry must start at exact zero after reset")
        result = {
            "simulator_timestamp_s": timestamp,
            "control_step": self.control_step,
            "physics_substep_count": substeps,
            "reset_generation": 0,
            "reset_flag": self.control_step == 0,
            "teleport_flag": np.zeros(self.object_count, dtype=bool),
            "simulator_pose_error_flag": errors,
        }
        self._since_sample["since_sample"] = 0
        return result

    def contract(self) -> dict[str, str]:
        return {
            "timestamp_origin": self.timestamp_origin,
            "physics_substep_counter_origin": self.counter_origin,
            "duration_unit": "policy_action_rows_not_physical_time",
            "teleport_flag_origin": "launcher_and_collector_make_no_post_reset_teleport_API_calls",
            "pose_error_origin": "finite_simulator_pose_read_else_fail_closed",
        }

    def close(self) -> None:
        if self._patched_scene is not None:
            if getattr(self._patched_scene, "step", None) is not self._instrumented_step:
                raise LauncherContractError("instrumented scene.step changed before restore")
            setattr(self._patched_scene, "step", self._original_step)
            self._patched_scene = None
            self._original_step = None
            self._instrumented_step = None


def _raw_pose_allow_invalid(value: Any) -> tuple[np.ndarray, bool]:
    try:
        pose = value.get_pose() if callable(getattr(value, "get_pose", None)) else value.pose
        vector = np.r_[np.asarray(pose.p), np.asarray(pose.q)].astype(np.float64)
        if vector.shape != (7,):
            raise ValueError("pose shape")
        return vector, not bool(np.isfinite(vector).all())
    except Exception:
        return np.full(7, np.nan, dtype=np.float64), True


class RoboTwinCollectionRuntime:
    """Strict adapter implementing the collector's reset/snapshot/H=1 API."""

    def __init__(
        self,
        *,
        env: Any,
        torch_module: Any,
        device: Any,
        bounds: list[list[float]],
        registry: Mapping[str, Any],
        event_spec: Mapping[str, Any],
        raw_state: Callable[[Any], np.ndarray],
        derive_events: Callable[..., Any],
    ) -> None:
        self.env = env
        self.torch = torch_module
        self.device = device
        self.bounds = bounds
        self.registry = validate_materialized_registry_contract(registry)
        self.registry_names = [str(item["name"]) for item in self.registry["objects"]]
        self.event_spec = event_spec
        self.raw_state = raw_state
        self.derive_events_fn = derive_events
        self.task = None
        self.objects: list[Any] = []
        self.clock: DecisionTelemetryClock | None = None
        self.env_steps = 0
        self.reset_generation = -1
        self.clock_contracts: list[dict[str, str]] = []
        self.identity_validation_count = 0

    def reset(self, seed: int, instruction: str):
        if self.clock is not None:
            self.clock.close()
        observation, _ = reset_with_explicit_instruction(
            self.env, requested_seed=seed, instruction=instruction
        )
        subenv = self.env.venv.envs[0]
        self.task = subenv.task
        resolved = int(self.task.ep_num)
        descriptions = observation.get("task_descriptions")
        observed = str(descriptions[0]) if descriptions is not None and len(descriptions) == 1 else ""
        try:
            assert_runtime_registry_identity(self.task, self.registry)
        except BaseException as exc:
            raise LauncherContractError(
                "live can/pot identity differs from the frozen reset-only registry"
            ) from exc
        self.identity_validation_count += 1
        self.objects = [getattr(self.task, name) for name in self.registry_names]
        scene = getattr(self.task, "scene", None)
        if scene is None:
            raise LauncherContractError("RoboTwin task lacks simulator scene")
        self.clock = DecisionTelemetryClock(scene, object_count=len(self.objects))
        self.reset_generation += 1
        self.clock_contracts.append(self.clock.contract())
        return observation, resolved, observed

    def snapshot(self) -> dict[str, Any]:
        if self.task is None or self.clock is None:
            raise LauncherContractError("snapshot requested before reset")
        rows = [_raw_pose_allow_invalid(value) for value in self.objects]
        poses = np.stack([row[0] for row in rows])
        errors = np.asarray([row[1] for row in rows], dtype=bool)
        if bool(errors.any()):
            raise LauncherContractError("simulator object pose read is non-finite/invalid")
        proprio = np.asarray(self.raw_state(self.task), dtype=np.float32)
        return {
            "object_names": list(self.registry_names),
            "object_poses": poses,
            "proprio": proprio,
            "telemetry": {
                **self.clock.telemetry(pose_error=errors),
                "reset_generation": self.reset_generation,
            },
        }

    def step(self, action: Any):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (1, 1, ACTION_DIM):
            raise LauncherContractError("collector action must be [1,1,14]")
        validate_piper_step(command.reshape(ACTION_DIM), self.bounds, step_index=self.env_steps)
        env_action = self.torch.from_numpy(command.copy()).to(device=self.device, dtype=self.torch.float32)
        echo = env_action.detach().cpu().numpy()
        if not np.array_equal(echo, command):
            raise LauncherContractError("CUDA action conversion changed command")
        observation, _, terminated, truncated, info = self.env.step(env_action, auto_reset=False)
        self.env_steps += 1
        assert self.clock is not None
        self.clock.after_policy_step()

        def scalar(value: Any) -> bool:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return bool(np.asarray(value).reshape(-1)[0])

        return observation, scalar(terminated), scalar(truncated), {
            "success": scalar(info.get("success", [False]))
        }

    def derive_events(self, poses, names, success, event_spec):
        # This launcher binds the clean Source-r13 collector whose audited
        # derive_events API is exactly four positional arguments and whose
        # task is the module-level move_can_pot contract.
        return self.derive_events_fn(poses, names, success, event_spec)

    def mapping(self) -> dict[str, Callable[..., Any]]:
        return {
            "reset": self.reset,
            "snapshot": self.snapshot,
            "step": self.step,
            "derive_events": self.derive_events,
        }

    def close(self) -> None:
        if self.clock is not None:
            self.clock.close()
            self.clock = None


def _write_manifest_and_receipt(
    *,
    authority: Mapping[str, Any],
    authority_path: Path,
    output: Path,
    status: str,
    exit_code: int,
    group_path: Path | None,
    audit: Mapping[str, Any] | None,
    env_steps: int,
    identity_validation_count: int,
    error: BaseException | None,
    clock_contracts: list[dict[str, str]],
) -> dict[str, Any]:
    output_contract = authority["output_contract"]
    manifest_path = Path(output_contract["manifest"])
    receipt_path = Path(output_contract["receipt"])
    group_record = None
    if group_path is not None:
        group_record = {
            "path": str(group_path),
            "file_sha256": file_sha256(group_path),
            "audit": dict(audit or {}),
        }
    manifest = {
        "format": MANIFEST_FORMAT,
        "status": status,
        "authority_logical_sha256": authority["authority_sha256"],
        "evidence_scope": "nonfresh_piper_development_only",
        "requested_seeds": [FIXED_DEVELOPMENT_SEED],
        "completed_groups": 1 if group_record is not None else 0,
        "reset_identity_validation_count": identity_validation_count,
        "group": group_record,
        "fresh_inputs_used": False,
        "task_success_claimed": False,
        "performance_evaluation_authorized": False,
        "transfer_claim_authorized": False,
        "telemetry_source_binding": authority["telemetry_source_binding"],
        "object_identity_contract": authority["object_identity_contract"],
    }
    atomic_json(manifest_path, manifest)
    receipt_base: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "status": status,
        "exit_code": exit_code,
        "authority": {
            "path": str(authority_path.resolve()),
            "file_sha256": file_sha256(authority_path),
            "logical_sha256": authority["authority_sha256"],
        },
        "manifest": {"path": str(manifest_path), "file_sha256": file_sha256(manifest_path)},
        "group": group_record,
        "environment_steps": env_steps,
        "reset_identity_validation_count": identity_validation_count,
        "clock_contracts": clock_contracts,
        "telemetry_source_binding": authority["telemetry_source_binding"],
        "object_identity_contract": authority["object_identity_contract"],
        "inherited_execution_semantics": authority["inherited_execution_semantics"],
        "duration_semantics": "policy action row count; simulator timestamp is separate telemetry",
        "failure": None if error is None else {
            "type": type(error).__name__,
            "message": str(error),
            "fail_closed": True,
        },
        "fresh_inputs_used": False,
        "real_robot_execution": False,
        "task_success_claimed": False,
        "performance_evaluation_authorized": False,
        "transfer_claim_authorized": False,
    }
    receipt = {**receipt_base, "receipt_logical_sha256": canonical_sha256(receipt_base)}
    atomic_json(receipt_path, receipt)
    return {
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_logical_sha256": receipt["receipt_logical_sha256"],
        "exit_code": exit_code,
        "status": status,
    }


def _release_collection_runtime(
    *, runtime_adapter: RoboTwinCollectionRuntime | None, capture: Any,
    env: Any | None, policy: Any, torch_module: Any,
) -> None:
    """Attempt every release and report cleanup failure as a closed contract."""

    failures: list[BaseException] = []
    if runtime_adapter is not None:
        try:
            runtime_adapter.close()
        except BaseException as exc:
            failures.append(exc)
    try:
        if env is not None:
            offload_runtime(capture, env, policy, torch_module)
        else:
            capture.close()
            policy.to("cpu")
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
    except BaseException as exc:
        failures.append(exc)
    if failures:
        raise LauncherContractError(
            f"collection runtime release failed in {len(failures)} operation(s)"
        ) from failures[0]


def run_authorized_collection(authority_path: Path) -> dict[str, Any]:
    """Validate all content before output creation, then collect exactly one seed."""

    authority, r6f, r6e, r6c, r6d, seed = validate_collection_authority(authority_path)
    collector_runtime_binding = bind_r6f_collection_runtime(
        Path(authority["r6f_lineage"]["path"])
    )
    if (
        collector_runtime_binding["r6f_preregistration"]["logical_sha256"]
        != authority["r6f_lineage"]["logical_sha256"]
        or collector_runtime_binding["runtime_roots"] != r6e["runtime_roots"]
        or collector_runtime_binding["explicit_instruction"] != INSTRUCTION
        or collector_runtime_binding["collection_execution_authorized_by_this_binding"] is not False
    ):
        raise LauncherContractError("collector R6e/R6f runtime binding changed")
    output = reject_fresh_path(Path(authority["output_contract"]["directory"]), "collection output")
    if output.exists():
        raise FileExistsError(output)
    roots = {key: Path(value) for key, value in r6e["runtime_roots"].items()}
    os.environ["ASSETS_PATH"] = str(roots["robotwin_root"])
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path[:0] = [str(roots["robotwin_code"]), str(roots["rlinf_root"]), str(roots["lerobot_root"] / "src")]

    import torch
    from collect_openvla_etsf_rollouts import derive_events, raw_state
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
    task_module = importlib.import_module("envs.move_can_pot")
    task_source = Path(task_module.__file__).resolve()
    frozen_task_source = authority["object_identity_contract"]["move_can_pot_source"]
    if (
        task_source != Path(frozen_task_source["path"]).resolve()
        or file_sha256(task_source) != frozen_task_source["sha256"]
    ):
        raise LauncherContractError("loaded move_can_pot task source differs from authority")
    runtime_origins["robotwin_move_can_pot"] = {
        "path": str(task_source),
        "sha256": file_sha256(task_source),
    }
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise LauncherContractError("collection requires the designated RTX 4090")
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
    loaded_policy = validate_loaded_policy_contract(
        config=config, preprocessor=preprocessor, postprocessor=postprocessor,
        torch_module=torch, device=device,
    )
    capture = resolve_shared_prefix_capture(policy)
    env = None
    runtime_adapter = None
    resources_released = False
    output.mkdir(parents=False, exist_ok=False)
    group_path = Path(authority["output_contract"]["group"])
    try:
        registry = json.loads(Path(authority["input_artifacts"]["object_registry"]["path"]).read_text(encoding="utf-8"))
        pose_spec = json.loads(Path(authority["input_artifacts"]["pose_quality_spec"]["path"]).read_text(encoding="utf-8"))
        event_spec = json.loads(Path(authority["input_artifacts"]["event_spec"]["path"]).read_text(encoding="utf-8"))
        max_steps = int(authority["scope"]["max_episode_steps"])
        seeds_path = roots["rlinf_root"] / "rlinf/envs/robotwin/seeds/eval_seeds.json"
        frozen_seed_registry = r6e["runtime_source_artifacts"]["robotwin_eval_seed_registry"]
        if seeds_path.resolve() != Path(frozen_seed_registry["path"]).resolve() or file_sha256(seeds_path) != frozen_seed_registry["sha256"]:
            raise LauncherContractError("R6e-bound eval seed registry changed")
        env_config = piper_environment_config(
            roots["robotwin_root"], seeds_path, output, step_limit=max_steps
        )
        env = RoboTwinEnv(
            cfg=OmegaConf.create(env_config), num_envs=1, seed_offset=0,
            total_num_processes=1, worker_info=None, record_metrics=True,
        )
        bounds = r6c["receipt"]["static_contract"]["static_semantics"]["piper_action_bounds"]
        runtime_adapter = RoboTwinCollectionRuntime(
            env=env, torch_module=torch, device=device, bounds=bounds,
            registry=registry, event_spec=event_spec,
            raw_state=raw_state, derive_events=derive_events,
        )

        def query_fn(observation: Mapping[str, Any], query_index: int):
            return generate_candidate_query(
                policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                capture=capture, observation=observation, bounds=bounds, device=device,
                scene_seed=FIXED_DEVELOPMENT_SEED, query_index=query_index,
            )

        record = collect_dense_group(
            runtime=runtime_adapter.mapping(), query_fn=query_fn,
            requested_seed=FIXED_DEVELOPMENT_SEED, instruction=INSTRUCTION,
            object_registry=registry, pose_quality_spec=pose_spec,
            event_spec=event_spec, max_steps=max_steps,
        )
        if record["resolved_seed"] != EXPECTED_RESOLVED_DEVELOPMENT_SEED:
            raise LauncherContractError("requested seed resolved to a different scene")
        if record["status"] != "collected_development_group":
            env_steps = runtime_adapter.env_steps
            clock_contracts = [dict(item) for item in runtime_adapter.clock_contracts]
            try:
                _release_collection_runtime(
                    runtime_adapter=runtime_adapter, capture=capture, env=env,
                    policy=policy, torch_module=torch,
                )
            finally:
                resources_released = True
            return _write_manifest_and_receipt(
                authority=authority, authority_path=authority_path, output=output,
                status="completed_root_fewer_than_two_legal_no_group",
                exit_code=EXIT_ROOT_INSUFFICIENT, group_path=None, audit=None,
                env_steps=env_steps,
                identity_validation_count=runtime_adapter.identity_validation_count,
                error=None, clock_contracts=clock_contracts,
            )
        save_schema6_group(group_path, record)
        audit = validate_schema6_group_file(group_path)
        audit = {
            **audit,
            "runtime_module_origins": runtime_origins,
            "loaded_policy_contract": loaded_policy,
            "development_seed_contract": seed,
            "collector_runtime_binding": collector_runtime_binding,
            "reset_identity_validation_count": runtime_adapter.identity_validation_count,
        }
        env_steps = runtime_adapter.env_steps
        clock_contracts = [dict(item) for item in runtime_adapter.clock_contracts]
        try:
            _release_collection_runtime(
                runtime_adapter=runtime_adapter, capture=capture, env=env,
                policy=policy, torch_module=torch,
            )
        finally:
            resources_released = True
        return _write_manifest_and_receipt(
            authority=authority, authority_path=authority_path, output=output,
            status="completed_one_seed_schema6_development_collection",
            exit_code=EXIT_SUCCESS, group_path=group_path, audit=audit,
            env_steps=env_steps,
            identity_validation_count=runtime_adapter.identity_validation_count,
            error=None, clock_contracts=clock_contracts,
        )
    except BaseException as exc:
        env_steps = 0 if runtime_adapter is None else runtime_adapter.env_steps
        clock_contracts = [] if runtime_adapter is None else [
            dict(item) for item in runtime_adapter.clock_contracts
        ]
        failure = exc
        if not resources_released:
            try:
                _release_collection_runtime(
                    runtime_adapter=runtime_adapter, capture=capture, env=env,
                    policy=policy, torch_module=torch,
                )
            except BaseException as cleanup_exc:
                failure = LauncherContractError(
                    f"{type(exc).__name__}: {exc}; runtime cleanup also failed: {cleanup_exc}"
                )
            resources_released = True
        if not Path(authority["output_contract"]["receipt"]).exists():
            return _write_manifest_and_receipt(
                authority=authority, authority_path=authority_path, output=output,
                status="failed_closed_schema6_development_collection",
                exit_code=EXIT_FAILURE,
                group_path=group_path if group_path.is_file() else None,
                audit=None,
                env_steps=env_steps, error=failure,
                identity_validation_count=(
                    0 if runtime_adapter is None
                    else runtime_adapter.identity_validation_count
                ),
                clock_contracts=clock_contracts,
            )
        raise
    finally:
        if not resources_released:
            _release_collection_runtime(
                runtime_adapter=runtime_adapter, capture=capture, env=env,
                policy=policy, torch_module=torch,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    result = run_authorized_collection(args.preregistration)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(int(result["exit_code"]))


if __name__ == "__main__":
    main()
