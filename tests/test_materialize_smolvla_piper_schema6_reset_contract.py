from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import launch_smolvla_piper_schema6_development_collection as launcher  # noqa: E402
from etsf_schema6_pose_quality import registry_sha256, validate_spec  # noqa: E402
from materialize_smolvla_piper_schema6_reset_contract import (  # noqa: E402
    MAX_PHYSICS_SUBSTEPS,
    SCENE_TIMESTEP_BINARY32_S,
    SCENE_TIMESTEP_S,
    ResetContractError,
    assert_runtime_registry_identity,
    build_pose_quality_spec,
    build_runtime_object_registry,
    sapien_actor_name,
    validate_scene_timestep,
)
from run_smolvla_piper_r6d_direct_actor_smoke import file_sha256  # noqa: E402


class Actor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.pose = type("Pose", (), {
            "p": np.asarray([0.0, 0.0, 0.5]),
            "q": np.asarray([1.0, 0.0, 0.0, 0.0]),
        })()

    def get_name(self) -> str:
        return self.name


class Scene:
    def get_timestep(self) -> float:
        return SCENE_TIMESTEP_S

    def step(self) -> None:
        pass


class Task:
    def __init__(self) -> None:
        self.can = Actor("can_visual_actor")
        self.pot = Actor("pot_visual_actor")
        self.can_id = 3
        self.pot_id = 8
        self.ep_num = 100101000
        self.scene = Scene()


def test_scene_timestep_accepts_only_exact_binary64_or_binary32_contract() -> None:
    binary64 = validate_scene_timestep(SCENE_TIMESTEP_S)
    assert binary64["accepted_representation"] == "python_binary64_1_div_250"
    assert binary64["broad_numeric_tolerance_used"] is False

    binary32 = validate_scene_timestep(np.float32(SCENE_TIMESTEP_S))
    assert binary32["observed_seconds"] == SCENE_TIMESTEP_BINARY32_S
    assert (
        binary32["accepted_representation"]
        == "ieee754_binary32_roundtrip_1_div_250"
    )

    adjacent = np.nextafter(
        np.float32(SCENE_TIMESTEP_S), np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(ResetContractError, match="observed_hex"):
        validate_scene_timestep(adjacent)
    with pytest.raises(ResetContractError, match="boolean"):
        validate_scene_timestep(True)


def test_registry_is_derived_from_task_attrs_actor_names_and_model_ids() -> None:
    task = Task()
    registry = build_runtime_object_registry(task)
    assert registry["objects"] == [
        {
            "name": "can",
            "stable_sim_actor_id": "task_attr=can;sapien_actor_name=can_visual_actor",
            "asset_model_id": "105_sauce-can/base3",
            "role": "manipulated",
            "is_static": False,
        },
        {
            "name": "pot",
            "stable_sim_actor_id": "task_attr=pot;sapien_actor_name=pot_visual_actor",
            "asset_model_id": "060_kitchenpot/base8",
            "role": "receptacle",
            "is_static": False,
        },
    ]
    assert_runtime_registry_identity(task, registry)
    task.can_id = 4
    with pytest.raises(ResetContractError, match="differs"):
        assert_runtime_registry_identity(task, registry)


def test_actor_name_channels_and_model_index_fail_closed() -> None:
    actor = Actor("property-name")
    actor.get_name = lambda: "method-name"
    with pytest.raises(ResetContractError, match="disagree"):
        sapien_actor_name(actor)
    task = Task()
    task.can_id = 3.0
    with pytest.raises(ResetContractError, match="integer model index"):
        build_runtime_object_registry(task)
    task = Task()
    task.pot = task.can
    task.pot.name = "shared"
    with pytest.raises(ResetContractError, match="same actor"):
        build_runtime_object_registry(task)


def test_pose_spec_is_fixed_data_independent_and_source_bound(tmp_path: Path) -> None:
    source = tmp_path / "move_can_pot.py"
    source.write_text("# synthetic frozen semantics\n", encoding="utf-8")
    registry = build_runtime_object_registry(Task())
    spec = build_pose_quality_spec(
        registry,
        move_can_pot_source={"path": str(source), "sha256": file_sha256(source)},
    )
    validate_spec(spec, expected_registry_sha256=registry_sha256(registry))
    assert spec["thresholds"]["timestamp_step_min_s"] == SCENE_TIMESTEP_S
    assert spec["thresholds"]["timestamp_step_max_s"] == SCENE_TIMESTEP_S * MAX_PHYSICS_SUBSTEPS
    assert spec["threshold_basis"]["thresholds_fit_from_pose_data"] is False
    assert file_sha256(source) in spec["threshold_basis"]["source"]
    source.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ResetContractError, match="source binding"):
        build_pose_quality_spec(
            registry,
            move_can_pot_source={"path": str(source), "sha256": "0" * 64},
        )


def test_launcher_revalidates_identity_on_every_reset_before_snapshot(monkeypatch) -> None:
    task = Task()
    subenv = type("SubEnv", (), {"task": task})()
    env = type("Env", (), {"venv": type("VEnv", (), {"envs": [subenv]})()})()

    def reset(_env, *, requested_seed, instruction):
        return {"task_descriptions": [instruction]}, {}

    monkeypatch.setattr(launcher, "reset_with_explicit_instruction", reset)
    runtime = launcher.RoboTwinCollectionRuntime(
        env=env,
        torch_module=object(),
        device=object(),
        bounds=[[0.0, 1.0]] * 14,
        registry=build_runtime_object_registry(task),
        event_spec={},
        raw_state=lambda _: np.zeros(14, dtype=np.float32),
        derive_events=lambda *args: ([], [], [], []),
    )
    runtime.reset(100101000, "move the can into the pot")
    assert runtime.identity_validation_count == 1
    task.pot_id = 9
    with pytest.raises(launcher.LauncherContractError, match="identity differs"):
        runtime.reset(100101000, "move the can into the pot")
    assert runtime.identity_validation_count == 1


def test_registry_tamper_is_field_exact() -> None:
    task = Task()
    registry = build_runtime_object_registry(task)
    for field, changed_value in (
        ("stable_sim_actor_id", "task_attr=can;sapien_actor_name=fake"),
        ("asset_model_id", "105_sauce-can/base99"),
    ):
        changed = copy.deepcopy(registry)
        changed["objects"][0][field] = changed_value
        with pytest.raises(ResetContractError, match="differs"):
            assert_runtime_registry_identity(task, changed)
