#!/usr/bin/env python3
"""Reset-only materializer for the two-object Piper schema-v6 pose contract.

This program constructs one fixed non-Fresh development scene and performs one
explicit reset.  It never imports a policy, performs a policy forward, calls
``env.step``, reads reward/success/event labels, or writes a trajectory.  The
only simulator values retained are the stable identities of ``task.can`` and
``task.pot`` and the audited scene timestep.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import numbers
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from etsf_schema6_pose_quality import (
    REGISTRY_FORMAT,
    SPEC_FORMAT,
    registry_sha256,
    spec_sha256,
    validate_registry,
    validate_spec,
)
from run_smolvla_piper_r6d_direct_actor_smoke import (
    EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    REQUESTED_DEVELOPMENT_SEED,
    atomic_json,
    file_sha256,
    piper_environment_config,
    reject_fresh_path,
    reset_with_explicit_instruction,
    validate_runtime_module_origins,
)


TASK = "move_can_pot"
INSTRUCTION = "move the can into the pot"
TRACKED_DEFINITIONS = (
    {
        "name": "can",
        "task_attribute": "can",
        "task_model_id_attribute": "can_id",
        "asset_family": "105_sauce-can",
        "role": "manipulated",
    },
    {
        "name": "pot",
        "task_attribute": "pot",
        "task_model_id_attribute": "pot_id",
        "asset_family": "060_kitchenpot",
        "role": "receptacle",
    },
)
SCENE_TIMESTEP_S = 1.0 / 250.0
WORLD_AABB_M = [[-3.0, 3.0], [-1.0, 2.0], [-0.5, 3.0]]
MAX_PHYSICS_SUBSTEPS = 1000
MAX_STEP_TRANSLATION_M = math.sqrt(sum((upper - lower) ** 2 for lower, upper in WORLD_AABB_M))


class ResetContractError(RuntimeError):
    """The reset-only simulator identity contract cannot be proven."""


def _nonempty_text(value: Any, role: str) -> str:
    if not isinstance(value, str):
        raise ResetContractError(f"{role} must be a string")
    text = value
    if not text or text.strip() != text:
        raise ResetContractError(f"{role} must be a non-empty canonical string")
    return text


def sapien_actor_name(actor: Any) -> str:
    """Read and cross-check SAPIEN's method/property actor-name channels."""

    readings: list[tuple[str, str]] = []
    getter = getattr(actor, "get_name", None)
    if callable(getter):
        readings.append(("get_name", _nonempty_text(getter(), "SAPIEN actor get_name")))
    value = getattr(actor, "name", None)
    if value is not None and not callable(value):
        readings.append(("name", _nonempty_text(value, "SAPIEN actor name")))
    if not readings:
        raise ResetContractError("SAPIEN actor exposes neither get_name() nor name")
    names = {value for _, value in readings}
    if len(names) != 1:
        raise ResetContractError("SAPIEN actor get_name()/name disagree")
    return readings[0][1]


def _model_index(task: Any, attribute: str) -> int:
    value = getattr(task, attribute, None)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ResetContractError(f"task.{attribute} is not an integer model index")
    index = int(value)
    if index < 0:
        raise ResetContractError(f"task.{attribute} is not a non-negative exact integer")
    return index


def stable_sim_actor_id(*, task_attribute: str, actor_name: str) -> str:
    return f"task_attr={task_attribute};sapien_actor_name={actor_name}"


def build_runtime_object_registry(task: Any) -> dict[str, Any]:
    """Build the only accepted can/pot registry from the live reset task."""

    objects: list[dict[str, Any]] = []
    actors: list[Any] = []
    for definition in TRACKED_DEFINITIONS:
        attribute = definition["task_attribute"]
        if not hasattr(task, attribute):
            raise ResetContractError(f"reset task lacks task.{attribute}")
        actor = getattr(task, attribute)
        if actor is None:
            raise ResetContractError(f"reset task task.{attribute} is None")
        actor_name = sapien_actor_name(actor)
        model_index = _model_index(task, definition["task_model_id_attribute"])
        actors.append(actor)
        objects.append({
            "name": definition["name"],
            "stable_sim_actor_id": stable_sim_actor_id(
                task_attribute=attribute, actor_name=actor_name
            ),
            "asset_model_id": f"{definition['asset_family']}/base{model_index}",
            "role": definition["role"],
            "is_static": False,
        })
    if actors[0] is actors[1]:
        raise ResetContractError("task.can and task.pot resolve to the same actor object")
    return validate_materialized_registry_contract({
        "format": REGISTRY_FORMAT,
        "objects": objects,
    })


def validate_materialized_registry_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject registries that do not have the exact runtime-derived layout."""

    registry = validate_registry(value)
    objects = registry["objects"]
    if [item["name"] for item in objects] != ["can", "pot"]:
        raise ResetContractError("registry must track exactly can then pot")
    for item, definition in zip(objects, TRACKED_DEFINITIONS, strict=True):
        prefix = f"task_attr={definition['task_attribute']};sapien_actor_name="
        actor_id = item["stable_sim_actor_id"]
        asset_prefix = f"{definition['asset_family']}/base"
        suffix = item["asset_model_id"][len(asset_prefix):]
        if (
            not actor_id.startswith(prefix)
            or not actor_id[len(prefix):]
            or not item["asset_model_id"].startswith(asset_prefix)
            or not suffix.isdecimal()
            or str(int(suffix)) != suffix
            or item["role"] != definition["role"]
            or item["is_static"] is not False
        ):
            raise ResetContractError(f"registry identity semantics changed for {item['name']}")
    return registry


def assert_runtime_registry_identity(task: Any, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute both identities on every reset and require field equality."""

    recorded = validate_materialized_registry_contract(registry)
    observed = build_runtime_object_registry(task)
    if observed != recorded:
        raise ResetContractError("reset task can/pot identity differs from frozen registry")
    return observed


def build_pose_quality_spec(
    registry: Mapping[str, Any], *, move_can_pot_source: Mapping[str, str]
) -> dict[str, Any]:
    """Build fixed, data-independent workspace/time pose-quality thresholds."""

    canonical_registry = validate_materialized_registry_contract(registry)
    source_path = reject_fresh_path(
        Path(move_can_pot_source["path"]), "move_can_pot source"
    )
    source_sha = str(move_can_pot_source["sha256"])
    if not source_path.is_file() or file_sha256(source_path) != source_sha:
        raise ResetContractError("move_can_pot source binding changed")
    value = {
        "format": SPEC_FORMAT,
        "schema_version": 6,
        "object_registry_sha256": registry_sha256(canonical_registry),
        "pose_layout": {
            "shape_suffix": [7],
            "translation_indices": [0, 1, 2],
            "quaternion_indices": [3, 4, 5, 6],
            "quaternion_order": "wxyz",
            "frame": "simulator_world",
            "translation_unit": "metre",
            "rotation_unit": "radian",
        },
        "time_layout": {
            "timestamp_unit": "second",
            "timestamp_clock": "simulator_monotonic",
            "control_step_semantics": "sample_after_completed_control_step",
            "physics_substep_semantics": "substeps_since_previous_sample_zero_at_reset",
        },
        "thresholds": {
            "world_aabb_m": WORLD_AABB_M,
            "quaternion_norm_abs_tolerance": 1e-3,
            "max_step_translation_m": MAX_STEP_TRANSLATION_M,
            "max_step_rotation_rad": math.pi,
            "static_object_max_step_translation_m": 1e-6,
            "static_object_max_step_rotation_rad": 1e-6,
            "timestamp_step_min_s": SCENE_TIMESTEP_S,
            "timestamp_step_max_s": SCENE_TIMESTEP_S * MAX_PHYSICS_SUBSTEPS,
            "max_physics_substeps_per_control_step": MAX_PHYSICS_SUBSTEPS,
        },
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "source": (
                "RoboTwin/SAPIEN hard development envelope; timestep=1/250 second; "
                "translation bound=world-AABB diagonal; rotation bound=pi; no pose, "
                f"trajectory, event, reward, success or outcome data; move_can_pot.py={source_sha}"
            ),
            "frozen_before_collection": True,
        },
    }
    return validate_spec(
        value, expected_registry_sha256=registry_sha256(canonical_registry)
    )


def _release_reset_environment(env: Any, torch_module: Any) -> None:
    failure: BaseException | None = None
    try:
        if callable(getattr(env, "offload", None)):
            env.offload(clear_cache=True)
        else:
            env.close()
    except BaseException as exc:
        failure = exc
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except BaseException as exc:
        failure = failure or exc
    if failure is not None:
        raise ResetContractError("reset-only environment release failed") from failure


def materialize_reset_contract(*, r6f_preregistration: Path, output_directory: Path) -> dict[str, Any]:
    """Reset once, close the environment, then atomically publish two JSON files."""

    from freeze_smolvla_piper_schema6_development_collection import (
        load_r6f_lineage_for_collection,
    )

    r6f, r6e, _r6c, r6d, _seed = load_r6f_lineage_for_collection(
        r6f_preregistration
    )
    if r6f["explicit_instruction"] != INSTRUCTION:
        raise ResetContractError("R6f instruction changed")
    output = reject_fresh_path(output_directory, "schema6 reset contract output")
    staging = output.with_name(output.name + ".partial")
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    roots = {key: Path(value) for key, value in r6e["runtime_roots"].items()}
    task_source = roots["robotwin_code"] / "envs/move_can_pot.py"
    if not task_source.is_file():
        raise FileNotFoundError(task_source)
    task_source_binding = {
        "path": str(task_source.resolve()),
        "sha256": file_sha256(task_source),
    }
    os.environ["ASSETS_PATH"] = str(roots["robotwin_root"])
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path[:0] = [str(roots["robotwin_code"]), str(roots["rlinf_root"])]

    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    validate_runtime_module_origins(
        {
            "rlinf_robotwin_env": importlib.import_module("rlinf.envs.robotwin.robotwin_env"),
            "robotwin_vector_env": importlib.import_module("robotwin.envs.vector_env"),
            "robotwin_base_task": importlib.import_module("envs._base_task"),
            "robotwin_robot_controller": importlib.import_module("envs.robot.robot"),
        },
        r6d["runtime_source_artifacts"],
    )
    imported_task_source = Path(importlib.import_module("envs.move_can_pot").__file__).resolve()
    if imported_task_source != task_source.resolve() or file_sha256(imported_task_source) != task_source_binding["sha256"]:
        raise ResetContractError("imported move_can_pot.py differs from bound source")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise ResetContractError("reset-only materialization requires the designated RTX 4090")
    seed_path = roots["rlinf_root"] / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    frozen_seed = r6e["runtime_source_artifacts"]["robotwin_eval_seed_registry"]
    if seed_path.resolve() != Path(frozen_seed["path"]).resolve() or file_sha256(seed_path) != frozen_seed["sha256"]:
        raise ResetContractError("R6e-bound eval seed registry changed")
    config = piper_environment_config(
        roots["robotwin_root"], seed_path, output.parent, step_limit=1
    )
    env = RoboTwinEnv(
        cfg=OmegaConf.create(config), num_envs=1, seed_offset=0,
        total_num_processes=1, worker_info=None, record_metrics=False,
    )
    try:
        observation, _reset_info = reset_with_explicit_instruction(
            env, requested_seed=REQUESTED_DEVELOPMENT_SEED, instruction=INSTRUCTION
        )
        task = env.venv.envs[0].task
        resolved = int(task.ep_num)
        if resolved != EXPECTED_RESOLVED_DEVELOPMENT_SEED:
            raise ResetContractError("fixed development seed resolved differently")
        descriptions = observation.get("task_descriptions")
        if descriptions is None or list(descriptions) != [INSTRUCTION]:
            raise ResetContractError("reset instruction differs from explicit text")
        scene = getattr(task, "scene", None)
        timestep_reader = getattr(scene, "get_timestep", None)
        if not callable(timestep_reader) or float(timestep_reader()) != SCENE_TIMESTEP_S:
            raise ResetContractError("scene timestep differs from fixed 1/250 second")
        registry = build_runtime_object_registry(task)
        spec = build_pose_quality_spec(
            registry, move_can_pot_source=task_source_binding
        )
    finally:
        _release_reset_environment(env, torch)

    staging.mkdir(mode=0o755, parents=False, exist_ok=False)
    registry_path = staging / "object_registry.json"
    spec_path = staging / "pose_quality_spec.json"
    atomic_json(registry_path, registry)
    atomic_json(spec_path, spec)
    os.replace(staging, output)
    output.chmod(0o555)
    registry_path = output / registry_path.name
    spec_path = output / spec_path.name
    return {
        "status": "materialized_reset_only_schema6_contract",
        "requested_seed": REQUESTED_DEVELOPMENT_SEED,
        "resolved_seed": EXPECTED_RESOLVED_DEVELOPMENT_SEED,
        "environment_steps": 0,
        "policy_imported_or_forwarded": False,
        "trajectory_or_labels_read": False,
        "fresh_inputs_used": False,
        "object_registry": {
            "path": str(registry_path),
            "file_sha256": file_sha256(registry_path),
            "logical_sha256": registry_sha256(registry),
        },
        "pose_quality_spec": {
            "path": str(spec_path),
            "file_sha256": file_sha256(spec_path),
            "logical_sha256": spec_sha256(
                spec, expected_registry_sha256=registry_sha256(registry)
            ),
        },
        "move_can_pot_source": task_source_binding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6f-preregistration", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_reset_contract(
        r6f_preregistration=args.r6f_preregistration,
        output_directory=args.output_directory,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
