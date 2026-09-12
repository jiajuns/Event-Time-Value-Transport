#!/usr/bin/env python3
"""Collect privileged relation trajectories from public RoboTwin tasks.

The collector observes the public scripted expert at 15 Hz.  It stores only
body-independent scene graphs, physical time, canonical EE effects and labels
derived from simulator state.  Native joint indices are never serialized.
Planner failures are retained as negative trajectories instead of discarded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import sys
import traceback
import types
import zipfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from universal_event.schema import (
    CANONICAL_ACTION_DIM,
    EDGE_FEATURE_DIM,
    EVENT_NAMES,
    MAX_NODES,
    NODE_FEATURE_DIM,
    NODE_TYPES,
    RELATION_NAMES,
    TASK_SPECS,
    TaskSpec,
    schema_sha256,
)


FORMAT = "etsf_robotwin2_universal_event_episode_v1"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
BODY_EMBODIMENT = {
    "aloha-agilex": ["aloha-agilex"],
    "arx-x5": ["ARX-X5", "ARX-X5", 0.6],
    "franka": ["franka-panda", "franka-panda", 0.8],
    "piper": ["piper", "piper", 0.6],
    "ur5": ["ur5-wsg", "ur5-wsg", 0.8],
}


def install_headless_physics_backend(task: Any) -> None:
    """Disable RoboTwin's unused ray-tracing path on compute-only GPUs.

    H100 nodes expose CUDA but no NVIDIA Vulkan graphics ICD.  The universal
    graph collector needs physics poses and canonical EE effects, not pixels.
    This instance-local override therefore preserves SAPIEN physics, collision
    shapes and CuRobo planning while omitting lights, cameras and rendering.
    It deliberately does not patch RoboTwin source files shared by other jobs.
    """
    import sapien
    from sapien.wrapper.actor_builder import ActorBuilder
    from sapien.wrapper.urdf_loader import URDFLoader

    # Geometry helpers and the URDF loader normally attach visual components.
    # Suppress those components process-locally while retaining every collision
    # shape.  LinkBuilder inherits ActorBuilder, so this also covers articulations.
    def ignore_visual(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self

    for name in (
        "add_plane_visual", "add_box_visual", "add_capsule_visual",
        "add_cylinder_visual", "add_sphere_visual", "add_visual_from_file",
    ):
        setattr(ActorBuilder, name, ignore_visual)

    original_build_link = URDFLoader._build_link

    def build_link_without_visuals(self: Any, link: Any, link_builder: Any) -> Any:
        visuals = link.visuals
        link.visuals = []
        try:
            return original_build_link(self, link, link_builder)
        finally:
            link.visuals = visuals

    if not getattr(URDFLoader._build_link, "_etsf_headless", False):
        build_link_without_visuals._etsf_headless = True
        URDFLoader._build_link = build_link_without_visuals

    def static_box(scene: Any, pose: Any, half_size: list[float], name: str) -> Any:
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose(), half_size=half_size,
            material=scene.default_physical_material,
        )
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def setup_scene(self: Any, **kwargs: Any) -> None:
        self.engine = None
        self.renderer = None
        scene_config = sapien.SceneConfig()
        sapien.physx.set_scene_config(scene_config)
        self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        # ``Scene.add_ground`` also creates a render material in SAPIEN 3, so
        # build the collision plane explicitly.
        ground = self.scene.create_actor_builder()
        ground.set_physx_body_type("static")
        ground.add_plane_collision(
            pose=sapien.Pose([0, 0, kwargs.get("ground_height", 0)]),
            material=self.scene.default_physical_material,
        )
        ground.build(name="ground")
        self.direction_light_lst = []
        self.point_light_lst = []
        self.viewer = None

    def create_table_and_wall(
        self: Any, table_xy_bias: list[float] = [0, 0], table_height: float = 0.74
    ) -> None:
        self.table_xy_bias = list(table_xy_bias)
        self.wall_texture = None
        self.table_texture = None
        table_height += self.table_z_bias
        self.wall = static_box(
            self.scene, sapien.Pose([0, 1, 1.5]), [3, 0.6, 1.5], "wall"
        )
        # One collision slab is sufficient for dynamics/planning.  Legs are far
        # from the manipulation workspace and do not affect event trajectories.
        self.table = static_box(
            self.scene,
            sapien.Pose([table_xy_bias[0], table_xy_bias[1], table_height - 0.025]),
            [0.6, 0.35, 0.025],
            "table",
        )

    def load_camera(self: Any, **kwargs: Any) -> None:
        self.cameras = None
        self.scene.step()

    def update_render(self: Any) -> None:
        return None

    task.setup_scene = types.MethodType(setup_scene, task)
    task.create_table_and_wall = types.MethodType(create_table_and_wall, task)
    task.load_camera = types.MethodType(load_camera, task)
    task._update_render = types.MethodType(update_render, task)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialReplayArchive:
    """Read RoboTwin's recorded planner paths without extracting the archive."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with zipfile.ZipFile(self.path) as archive:
            seed_members = [name for name in archive.namelist() if name.endswith("/seed.txt")]
            if len(seed_members) != 1:
                raise ValueError(f"archive must contain exactly one seed.txt: {self.path}")
            self.root = seed_members[0].rsplit("/", 1)[0]
            self.seeds = [int(value) for value in archive.read(seed_members[0]).decode().split()]
            expected = [f"{self.root}/_traj_data/episode{index}.pkl"
                        for index in range(len(self.seeds))]
            missing = [name for name in expected if name not in set(archive.namelist())]
            if missing:
                raise ValueError(f"archive lacks {len(missing)} planner paths")
        self.sha256 = _file_sha256(self.path)

    def paths(self, index: int) -> tuple[int, list[Any], list[Any]]:
        if index < 0 or index >= len(self.seeds):
            raise IndexError(f"replay index {index} outside [0,{len(self.seeds)})")
        member = f"{self.root}/_traj_data/episode{index}.pkl"
        with zipfile.ZipFile(self.path) as archive:
            payload = pickle.loads(archive.read(member))
        if set(payload) != {"left_joint_path", "right_joint_path"}:
            raise ValueError(f"unexpected planner payload keys in {member}")
        return self.seeds[index], payload["left_joint_path"], payload["right_joint_path"]


def _pose7(actor: Any) -> np.ndarray:
    pose = actor.get_pose()
    value = np.asarray([*pose.p, *pose.q], dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("actor pose is not finite xyz+quaternion_wxyz")
    return value


def _functional_pose7(actor: Any, index: int, base_pose: np.ndarray) -> np.ndarray:
    getter = getattr(actor, "get_functional_point", None)
    if not callable(getter):
        raise ValueError("node requests a functional point but actor does not expose one")
    try:
        raw = getter(index, "pose")
    except (TypeError, ValueError):
        raw = getter(index)
    if hasattr(raw, "p"):
        position = np.asarray(raw.p, np.float32).reshape(-1)
        quaternion = np.asarray(getattr(raw, "q", base_pose[3:7]), np.float32).reshape(-1)
        value = np.r_[position[:3], quaternion[:4]]
    else:
        raw_array = np.asarray(raw, np.float32).reshape(-1)
        if len(raw_array) < 3:
            raise ValueError("functional point has fewer than xyz coordinates")
        value = np.r_[raw_array[:3], raw_array[3:7] if len(raw_array) >= 7 else base_pose[3:7]]
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("functional point pose is not finite xyz+quaternion_wxyz")
    return value.astype(np.float32)


def _node_pose7(actor: Any, item: Mapping[str, Any]) -> np.ndarray:
    base = _pose7(actor)
    mode = str(item.get("position_mode", "actor"))
    if mode == "actor":
        return base
    functional = _functional_pose7(actor, int(item.get("functional_point_id", 0)), base)
    if mode == "functional_point":
        return functional
    if mode == "actor_functional_midpoint":
        value = base.copy()
        value[:3] = (base[:3] + functional[:3]) * 0.5
        return value
    raise ValueError(f"unsupported node position mode: {mode}")


def _velocity3(actor: Any, name: str) -> np.ndarray:
    getter = getattr(actor, name, None)
    if not callable(getter):
        return np.zeros(3, np.float32)
    value = np.asarray(getter(), dtype=np.float32).reshape(-1)
    return value[:3] if len(value) >= 3 and np.isfinite(value[:3]).all() else np.zeros(3, np.float32)


def _articulation_fraction(actor: Any) -> float:
    get_qpos, get_limits = getattr(actor, "get_qpos", None), getattr(actor, "get_qlimits", None)
    if not callable(get_qpos) or not callable(get_limits):
        return 0.0
    qpos = np.asarray(get_qpos(), dtype=float).reshape(-1)
    limits = np.asarray(get_limits(), dtype=float).reshape(-1, 2)
    if not len(qpos) or not len(limits):
        return 0.0
    span = max(float(limits[0, 1] - limits[0, 0]), 1e-6)
    return float(np.clip((qpos[0] - limits[0, 0]) / span, 0.0, 1.0))


def _arm_pose(task: Any, side: str) -> np.ndarray:
    value = np.asarray(task.get_arm_pose(side), dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError(f"invalid {side} EE pose")
    return value


def _gripper_value(task: Any, side: str) -> float:
    getter = getattr(task.robot, f"get_{side}_gripper_val")
    return float(getter())


def _gripper_closed(task: Any, side: str) -> bool:
    for owner in (task, getattr(task, "robot", None)):
        fn = getattr(owner, f"is_{side}_gripper_close", None)
        if callable(fn):
            return bool(fn())
    return False


def _name_of(entity: Any) -> str:
    name = getattr(entity, "name", None)
    if name:
        return str(name)
    getter = getattr(entity, "get_name", None)
    return str(getter()) if callable(getter) else ""


def _direct_gripper_contact(task: Any, actor: Any, side: str) -> bool:
    """Return simulator contact evidence for one actor and one gripper side."""
    actor_name = _name_of(actor)
    if not actor_name:
        return False
    robot = task.robot
    gripper_names = set(map(str, getattr(robot, f"{side}_fix_gripper_name", [])))
    for joint_entry in getattr(robot, f"{side}_gripper", []):
        joint = joint_entry[0] if isinstance(joint_entry, (list, tuple)) else joint_entry
        child = getattr(joint, "child_link", None)
        name = _name_of(child)
        if name:
            gripper_names.add(name)
    for contact in task.scene.get_contacts():
        first = _name_of(contact.bodies[0].entity)
        second = _name_of(contact.bodies[1].entity)
        if ((first == actor_name and second in gripper_names)
                or (second == actor_name and first in gripper_names)):
            return True
    return False


def _node_roster(task: Any, spec: TaskSpec) -> list[dict[str, Any]]:
    result = []
    for node in spec.nodes:
        actor = getattr(task, node.attribute, None)
        if actor is None:
            raise ValueError(f"{spec.task} lacks public attribute {node.attribute!r}")
        result.append({"name": node.name, "type": node.node_type, "actor": actor,
                       "movable": node.movable, "articulation": node.articulation,
                       "position_mode": node.position_mode,
                       "functional_point_id": node.functional_point_id})
    result.extend(
        [
            {"name": "left_gripper", "type": "left_gripper", "actor": None,
             "movable": True, "articulation": False,
             "position_mode": "actor", "functional_point_id": 0},
            {"name": "right_gripper", "type": "right_gripper", "actor": None,
             "movable": True, "articulation": False,
             "position_mode": "actor", "functional_point_id": 0},
        ]
    )
    if len(result) > MAX_NODES:
        raise ValueError("task node roster exceeds MAX_NODES")
    return result


class CapturingScene:
    """Transparent scene proxy that samples after successful physics steps."""

    def __init__(self, scene: Any, callback: Any) -> None:
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_callback", callback)
        dt = float(scene.get_timestep())
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("RoboTwin scene timestep is invalid")
        object.__setattr__(self, "timestep", dt)
        object.__setattr__(self, "step_count", 0)

    def step(self) -> Any:
        result = self._scene.step()
        object.__setattr__(self, "step_count", self.step_count + 1)
        self._callback(self.step_count * self.timestep)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_scene", "_callback", "timestep", "step_count"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._scene, name, value)


class EpisodeObserver:
    def __init__(self, task: Any, spec: TaskSpec, sample_hz: float) -> None:
        self.task, self.spec = task, spec
        self.roster = _node_roster(task, spec)
        self.period = 1.0 / sample_hz
        self.next_sample = 0.0
        self.times: list[float] = []
        self.features: list[np.ndarray] = []
        self.ee: list[np.ndarray] = []
        self.closed: list[np.ndarray] = []
        self.direct_contact: list[np.ndarray] = []
        self.success: list[bool] = []
        self.initial_positions: np.ndarray | None = None

    def capture(self, time_s: float, *, force: bool = False) -> None:
        if not force and time_s + 1e-9 < self.next_sample:
            return
        if self.times and time_s <= self.times[-1] + 1e-10:
            return
        node = np.zeros((MAX_NODES, NODE_FEATURE_DIM), np.float32)
        positions = np.zeros((MAX_NODES, 3), np.float32)
        for index, item in enumerate(self.roster):
            node_type = NODE_TYPES.index(item["type"])
            if item["type"] in {"left_gripper", "right_gripper"}:
                side = item["type"].split("_")[0]
                pose = _arm_pose(self.task, side)
                velocity = np.zeros(3, np.float32)
                angular = np.zeros(3, np.float32)
                articulation = 0.0
                grip = _gripper_value(self.task, side)
            else:
                actor = item["actor"]
                pose = _node_pose7(actor, item)
                velocity = _velocity3(actor, "get_velocity")
                angular = _velocity3(actor, "get_angular_velocity")
                articulation = _articulation_fraction(actor) if item["articulation"] else 0.0
                grip = 0.0
            positions[index] = pose[:3]
            node[index, 0:7] = pose
            node[index, 7:10] = velocity
            node[index, 10:13] = angular
            node[index, 13] = float(item["movable"])
            node[index, 14] = float(item["articulation"])
            node[index, 15] = articulation
            node[index, 16] = grip
            node[index, 18] = float(np.linalg.norm(velocity))
            node[index, 19] = float(np.linalg.norm(angular))
            node[index, 21] = float(item["type"] == "object")
            node[index, 22] = float(item["name"] == self.spec.target)
            node[index, 23] = 1.0
        if self.initial_positions is None:
            self.initial_positions = positions.copy()
        node[:, 17] = np.linalg.norm(positions - self.initial_positions, axis=-1)
        node[:, 20] = positions[:, 2] - self.initial_positions[:, 2]
        ee = np.concatenate(
            (_arm_pose(self.task, "left"), [_gripper_value(self.task, "left")],
             _arm_pose(self.task, "right"), [_gripper_value(self.task, "right")])
        ).astype(np.float32)
        closed = np.asarray(
            [_gripper_closed(self.task, "left"), _gripper_closed(self.task, "right")], bool
        )
        moving_item = next((item for item in self.roster if item["name"] == self.spec.moving), None)
        direct_contact = np.zeros(2, bool)
        if moving_item is not None and moving_item["actor"] is not None:
            direct_contact[:] = [
                _direct_gripper_contact(self.task, moving_item["actor"], "left"),
                _direct_gripper_contact(self.task, moving_item["actor"], "right"),
            ]
        try:
            success = bool(self.task.check_success())
        except Exception:
            success = False
        self.times.append(float(time_s))
        self.features.append(node)
        self.ee.append(ee)
        self.closed.append(closed)
        self.direct_contact.append(direct_contact)
        self.success.append(success)
        self.next_sample = time_s + self.period


def edge_features(nodes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = len(nodes)
    result = np.zeros((count, MAX_NODES, MAX_NODES, EDGE_FEATURE_DIM), np.float32)
    for t in range(count):
        position, velocity = nodes[t, :, :3], nodes[t, :, 7:10]
        delta = position[None] - position[:, None]
        relative_velocity = velocity[None] - velocity[:, None]
        result[t, ..., :3] = delta
        result[t, ..., 3] = np.linalg.norm(delta, axis=-1)
        result[t, ..., 4] = np.linalg.norm(delta[..., :2], axis=-1)
        result[t, ..., 5] = delta[..., 2]
        result[t, ..., 6:9] = relative_velocity
        result[t, ..., 9] = mask[:, None]
        result[t, ..., 10] = mask[None, :]
        result[t, ..., 11] = np.abs(delta[..., 2])
    return result


def _goal_arrays(spec: TaskSpec, roster: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    goal = np.zeros((len(RELATION_NAMES), MAX_NODES, MAX_NODES), np.float32)
    mask = np.zeros_like(goal, bool)
    indices = {item["name"]: index for index, item in enumerate(roster)}
    for item in spec.goals:
        r, source, target = RELATION_NAMES.index(item.relation), indices[item.source], indices[item.target]
        goal[r, source, target] = item.value
        mask[r, source, target] = True
    return goal, mask


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs without scipy or embodiment-specific metadata."""
    padded = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def infer_temporal_held(
    object_positions: np.ndarray,
    gripper_positions: np.ndarray,
    closed: np.ndarray,
    gripper_quaternions: np.ndarray | None = None,
    direct_contact: np.ndarray | None = None,
) -> np.ndarray:
    """Infer grasp support from temporal coupling rather than a fixed TCP radius.

    RoboTwin embodiments expose different end-effector reference frames.  For
    example, Aloha's reported TCP remains roughly 0.21 m from a correctly held
    can, so a universal 0.11 m cutoff silently removes every grasp.  Within each
    closed-gripper interval we instead learn its own closest relative offset and
    require evidence that the object is lifted or co-moves with the gripper.
    Empty closes therefore stay negative, while the same rule works across TCP
    conventions and never enters the shared model as a body identifier.
    """
    object_positions = np.asarray(object_positions, np.float32)
    gripper_positions = np.asarray(gripper_positions, np.float32)
    closed = np.asarray(closed, bool)
    if object_positions.shape != gripper_positions.shape or object_positions.shape[-1] != 3:
        raise ValueError("held inference expects matching [time,3] positions")
    if closed.shape != (len(object_positions),):
        raise ValueError("closed mask length does not match positions")
    held = np.zeros(len(closed), bool)
    if len(closed) < 2:
        return held

    relative_world = object_positions - gripper_positions
    distance = np.linalg.norm(relative_world, axis=-1)
    if gripper_quaternions is not None:
        quaternion = np.asarray(gripper_quaternions, np.float32)
        if quaternion.shape != (len(closed), 4):
            raise ValueError("gripper quaternion must have shape [time,4]")
        quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(1e-8)
        conjugate = quaternion.copy()
        conjugate[:, 1:] *= -1
        pure = np.c_[np.zeros(len(closed), np.float32), relative_world]
        relative = _quat_multiply(_quat_multiply(conjugate, pure), quaternion)[:, 1:]
    else:
        # Scalar distance remains rotation invariant for synthetic inputs and
        # older archives that do not expose an EE quaternion.
        relative = distance[:, None]
    contact_support: np.ndarray | None = None
    if direct_contact is not None:
        contact_support = np.asarray(direct_contact, bool)
        if contact_support.shape != (len(closed),):
            raise ValueError("direct contact must have shape [time]")
        # Physics contacts can flicker for one or two sampled frames.  A small
        # temporal dilation retains a grasp without turning indirect co-motion
        # through a basket/cabinet into direct held_by evidence.
        expanded = contact_support.copy()
        for shift in (1, 2):
            expanded[shift:] |= contact_support[:-shift]
            expanded[:-shift] |= contact_support[shift:]
        contact_support = expanded
    object_step = np.r_[0.0, np.linalg.norm(np.diff(object_positions, axis=0), axis=-1)]
    relative_step = np.r_[0.0, np.linalg.norm(np.diff(relative, axis=0), axis=-1)]
    lifted = object_positions[:, 2] >= object_positions[0, 2] + 0.012
    coupled_motion = (object_step >= 0.0015) & (relative_step <= 0.010)

    # One-frame numerical coincidences are not enough to declare a grasp.
    sustained_coupling = coupled_motion.copy()
    if len(closed) >= 3:
        sustained_coupling[1:-1] |= coupled_motion[:-2] & coupled_motion[1:-1]
        sustained_coupling[2:] |= coupled_motion[1:-1] & coupled_motion[2:]

    for start, stop in _contiguous_true_runs(closed):
        run_distance = distance[start:stop]
        if not len(run_distance):
            continue
        # 0.45 m is only a broad impossibility guard.  The actual gate is the
        # run-relative closest TCP offset, which absorbs embodiment geometry.
        closest = float(np.quantile(run_distance, 0.10))
        if closest > 0.45:
            continue
        proximal = distance <= closest + 0.040
        stable = relative_step <= 0.015
        candidate = closed & proximal & stable
        if contact_support is not None:
            candidate &= contact_support
        anchor = candidate & (lifted | sustained_coupling)
        if not anchor[start:stop].any():
            continue
        # Retain only candidate components backed by a physical-motion anchor.
        # A later drop breaks relative-pose stability and terminates held even
        # when the gripper command remains closed.
        for component_start, component_stop in _contiguous_true_runs(candidate[start:stop]):
            absolute_start = start + component_start
            absolute_stop = start + component_stop
            if anchor[absolute_start:absolute_stop].any():
                held[absolute_start:absolute_stop] = True
    return held


def infer_release_drop_states(
    held: np.ndarray,
    closed: np.ndarray,
    object_positions: np.ndarray,
    gripper_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate commanded release from an unintended closed-gripper loss."""
    held = np.asarray(held, bool)
    closed = np.asarray(closed, bool)
    if held.shape != closed.shape or held.ndim != 2:
        raise ValueError("held and closed must be matching [time,arm] arrays")
    length, arms = held.shape
    if gripper_positions.shape != (length, arms, 3):
        raise ValueError("gripper positions must have shape [time,arm,3]")
    released = np.zeros(length, bool)
    dropped = np.zeros(length, bool)
    held_any = held.any(1)
    release_on = drop_on = False
    distances = np.linalg.norm(
        object_positions[:, None, :] - gripper_positions, axis=-1
    )
    for time in range(1, length):
        if held_any[time]:
            release_on = drop_on = False
        else:
            lost = held[time - 1] & ~held[time]
            if lost.any():
                if (~closed[time, lost]).any():
                    release_on, drop_on = True, False
                else:
                    future_stop = min(length, time + 4)
                    future_unheld = not held_any[time:future_stop].any()
                    falls = object_positions[time:future_stop, 2].min(initial=np.inf) < (
                        object_positions[time, 2] - 0.008
                    )
                    separates = any(
                        distances[time:future_stop, arm].max(initial=-np.inf)
                        > distances[time, arm] + 0.015
                        for arm in np.flatnonzero(lost)
                    )
                    if future_unheld and (falls or separates):
                        release_on, drop_on = False, True
            released[time] = release_on
            dropped[time] = drop_on
    return released, dropped


def derive_relations(
    nodes: np.ndarray,
    closed: np.ndarray,
    native_success: np.ndarray,
    roster: list[dict[str, Any]],
    spec: TaskSpec,
    direct_contact: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(nodes)
    labels = np.zeros((length, len(RELATION_NAMES), MAX_NODES, MAX_NODES), np.float32)
    mask = np.zeros_like(labels, bool)
    valid = np.asarray([index < len(roster) for index in range(MAX_NODES)])
    positions = nodes[..., :3]
    distance = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
    near = RELATION_NAMES.index("near")
    for source in np.flatnonzero(valid):
        for target in np.flatnonzero(valid):
            if source == target:
                continue
            mask[:, near, source, target] = True
            labels[:, near, source, target] = distance[:, source, target] <= 0.10

    indices = {item["name"]: index for index, item in enumerate(roster)}
    moving = indices.get(spec.moving, -1)
    target = indices.get(spec.target, -1)
    left, right = indices["left_gripper"], indices["right_gripper"]
    held_index = RELATION_NAMES.index("held_by")
    held = np.zeros((length, 2), bool)
    released = np.zeros(length, bool)
    dropped = np.zeros(length, bool)
    if moving >= 0:
        for arm, gripper in enumerate((left, right)):
            arm_contact = None if direct_contact is None else direct_contact[:, arm]
            held[:, arm] = infer_temporal_held(
                positions[:, moving], positions[:, gripper], closed[:, arm],
                nodes[:, gripper, 3:7], arm_contact,
            )
            mask[:, held_index, moving, gripper] = True
            labels[:, held_index, moving, gripper] = held[:, arm]
            # Re-anchor object/gripper proximity to the TCP convention observed
            # during a physically supported grasp.  This keeps ``approach``
            # meaningful for Aloha (~0.21 m TCP offset) and other embodiments.
            if held[:, arm].any():
                tcp_offset = float(np.median(distance[held[:, arm], moving, gripper]))
                approaching = distance[:, moving, gripper] <= tcp_offset + 0.10
                labels[:, near, moving, gripper] = approaching
                labels[:, near, gripper, moving] = approaching
        lifted_index = RELATION_NAMES.index("lifted")
        released_index = RELATION_NAMES.index("released")
        dropped_index = RELATION_NAMES.index("dropped")
        for relation in (lifted_index, released_index, dropped_index):
            mask[:, relation, moving, moving] = True
        lifted = positions[:, moving, 2] >= positions[0, moving, 2] + 0.02
        released, dropped = infer_release_drop_states(
            held, closed, positions[:, moving], positions[:, (left, right)]
        )
        labels[:, lifted_index, moving, moving] = lifted
        labels[:, released_index, moving, moving] = released

    at_target = np.zeros(length, bool)
    if moving >= 0 and target >= 0:
        xy = np.linalg.norm(positions[:, moving, :2] - positions[:, target, :2], axis=-1)
        dz = positions[:, moving, 2] - positions[:, target, 2]
        geometric = {
            "inside": (xy <= spec.xy_threshold_m) & (np.abs(dz) <= spec.z_threshold_m),
            "on_top_of": (xy <= spec.xy_threshold_m) & (dz >= -0.01) & (dz <= spec.z_threshold_m),
            "supported_by": (xy <= spec.xy_threshold_m) & (np.abs(dz) <= spec.z_threshold_m),
            "away_from": xy > max(spec.xy_threshold_m, 0.12),
        }
        for relation, value in geometric.items():
            index = RELATION_NAMES.index(relation)
            mask[:, index, moving, target] = True
            labels[:, index, moving, target] = value
        goal_relations = {edge.relation for edge in spec.goals}
        location_evidence = [geometric[name] for name in
                             ("inside", "on_top_of", "supported_by")
                             if name in goal_relations]
        if "near" in goal_relations:
            # Goal proximity is defined in task space (XY), not by the generic
            # center-to-center 10 cm radius used for arbitrary node pairs.
            location_evidence.append(xy <= spec.xy_threshold_m)
            labels[:, near, moving, target] = xy <= spec.xy_threshold_m
            labels[:, near, target, moving] = xy <= spec.xy_threshold_m
        if location_evidence:
            at_target = np.logical_or.reduce(location_evidence)

    if moving >= 0:
        dropped_index = RELATION_NAMES.index("dropped")
        # An off-target commanded release is a failed goal relation, but it is
        # not a physical drop.  ``dropped`` specifically means that support was
        # lost while the responsible gripper stayed closed.  Ignore a numerical
        # loss if the object is already physically supported at the task goal.
        labels[:, dropped_index, moving, moving] = dropped & ~at_target

    articulated = next((i for i, item in enumerate(roster) if item["articulation"]), None)
    if articulated is not None:
        index = RELATION_NAMES.index("open")
        mask[:, index, articulated, articulated] = True
        labels[:, index, articulated, articulated] = nodes[:, articulated, 15] >= spec.open_fraction
    if spec.activation_attribute is not None:
        actor_index = 0
        relation = "pressed" if spec.task == "press_stapler" else "activated"
        index = RELATION_NAMES.index(relation)
        mask[:, index, actor_index, actor_index] = True
        labels[:, index, actor_index, actor_index] = native_success
    return labels, mask


def derive_events(relations: np.ndarray, relation_mask: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    length = len(relations)
    events = np.zeros((length, len(EVENT_NAMES)), np.float32)
    rel = {name: relations[:, index] for index, name in enumerate(RELATION_NAMES)}
    rising = {name: np.r_[np.zeros((1, MAX_NODES, MAX_NODES), bool),
                          (rel[name][1:] > 0.5) & (rel[name][:-1] <= 0.5)]
              for name in RELATION_NAMES}
    events[:, EVENT_NAMES.index("approach")] = rising["near"].any((1, 2))
    events[:, EVENT_NAMES.index("grasp")] = rising["held_by"].any((1, 2))
    held = rel["held_by"].any((1, 2))
    moving = np.r_[False, np.linalg.norm(np.diff(nodes[:, :, :3], axis=0), axis=-1).max(1) > 0.006]
    events[:, EVENT_NAMES.index("transport")] = held & moving
    placed = rising["supported_by"] | rising["inside"] | rising["on_top_of"]
    events[:, EVENT_NAMES.index("place")] = placed.any((1, 2))
    events[:, EVENT_NAMES.index("release")] = rising["released"].any((1, 2))
    events[:, EVENT_NAMES.index("drop")] = rising["dropped"].any((1, 2))
    drop_seen = np.maximum.accumulate(events[:, EVENT_NAMES.index("drop")].astype(bool))
    events[:, EVENT_NAMES.index("regrasp")] = drop_seen & rising["held_by"].any((1, 2))
    events[:, EVENT_NAMES.index("open")] = rising["open"].any((1, 2))
    activated = rising["pressed"] | rising["activated"]
    events[:, EVENT_NAMES.index("activate")] = activated.any((1, 2))
    events[:, EVENT_NAMES.index("idle")] = ~events[:, :-1].astype(bool).any(1)
    return events


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack((lw*rw-lx*rx-ly*ry-lz*rz, lw*rx+lx*rw+ly*rz-lz*ry,
                     lw*ry-lx*rz+ly*rw+lz*rx, lw*rz+lx*ry-ly*rx+lz*rw), -1)


def _axis_angle(current: np.ndarray, following: np.ndarray) -> np.ndarray:
    current = current / np.linalg.norm(current, axis=-1, keepdims=True).clip(1e-8)
    following = following / np.linalg.norm(following, axis=-1, keepdims=True).clip(1e-8)
    conjugate = current.copy(); conjugate[..., 1:] *= -1
    relative = _quat_multiply(following, conjugate)
    relative = relative / np.linalg.norm(relative, axis=-1, keepdims=True).clip(1e-8)
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    norm = np.linalg.norm(relative[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, np.clip(relative[..., :1], 0, 1))
    return relative[..., 1:] * np.divide(angle, norm, out=np.full_like(angle, 2.0), where=norm > 1e-8)


def canonical_actions(ee: np.ndarray) -> np.ndarray:
    result = []
    for offset in (0, 8):
        translation = ee[1:, offset:offset+3] - ee[:-1, offset:offset+3]
        rotation = _axis_angle(ee[:-1, offset+3:offset+7], ee[1:, offset+3:offset+7])
        gripper = ee[1:, offset+7:offset+8] - ee[:-1, offset+7:offset+8]
        result.append(np.concatenate((translation, rotation, gripper), -1))
    value = np.concatenate(result, -1).astype(np.float32)
    if value.shape != (len(ee)-1, CANONICAL_ACTION_DIM) or not np.isfinite(value).all():
        raise ValueError("canonical action conversion failed")
    return value


def task_args(robotwin_root: Path, task: str, body: str, condition: str) -> dict[str, Any]:
    config = robotwin_root / "env_cfg" / "task_config"
    with (config / f"demo_{condition}.yml").open() as stream:
        args = yaml.safe_load(stream)
    with (config / "_embodiment_config.yml").open() as stream:
        registry = yaml.safe_load(stream)
    embodiment = list(BODY_EMBODIMENT[body])
    args["embodiment"] = embodiment

    def robot_file(name: str) -> Path:
        return (robotwin_root / str(registry[name]["file_path"])).resolve()

    if len(embodiment) == 1:
        left = right = robot_file(embodiment[0]); args["dual_arm_embodied"] = True
        args["embodiment_name"] = embodiment[0]
    else:
        left, right = robot_file(embodiment[0]), robot_file(embodiment[1])
        args["dual_arm_embodied"] = False; args["embodiment_dis"] = float(embodiment[2])
        args["embodiment_name"] = f"{embodiment[0]}_{embodiment[1]}"
    with (left / "config.yml").open() as stream: left_config = yaml.safe_load(stream)
    with (right / "config.yml").open() as stream: right_config = yaml.safe_load(stream)
    args.update(task_name=task, task_config=f"demo_{condition}", left_robot_file=str(left),
                right_robot_file=str(right), left_embodiment_config=left_config,
                right_embodiment_config=right_config, eval_mode=True, eval_video_log=False,
                collect_data=False, render_freq=0, save_data=False)
    return args


def collect_one(task_class: Any, args: Mapping[str, Any], spec: TaskSpec, seed: int,
                instruction: str, sample_hz: float, scene_backend: str) -> dict[str, Any]:
    task = task_class()
    if scene_backend == "headless-physics":
        install_headless_physics_backend(task)
    observer: EpisodeObserver | None = None
    error: str | None = None
    error_traceback: str | None = None
    try:
        random.seed(seed); np.random.seed(seed)
        task.setup_demo(now_ep_num=seed, seed=seed, is_test=True, **dict(args))
        # Renderer-free initialization can choose the mirrored arm for a
        # symmetric scene even with the same recorded seed.  RoboTwin's five
        # embodiments use the same per-arm joint convention; expose the single
        # recorded path to either arm so the task's current geometric arm choice
        # remains authoritative.  Dual-arm recordings are left unchanged.
        if not bool(args.get("need_plan", True)):
            if not task.left_joint_path and task.right_joint_path:
                task.left_joint_path = copy.deepcopy(task.right_joint_path)
            elif not task.right_joint_path and task.left_joint_path:
                task.right_joint_path = copy.deepcopy(task.left_joint_path)
        task.set_instruction(instruction=instruction)
        observer = EpisodeObserver(task, spec, sample_hz)
        observer.capture(0.0, force=True)
        task.scene = CapturingScene(task.scene, observer.capture)
        try:
            task.play_once()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc(limit=12)
        observer.capture(task.scene.step_count * task.scene.timestep, force=True)
    finally:
        if hasattr(task, "close_env"):
            try: task.close_env(clear_cache=False)
            except Exception: pass
    if observer is None or len(observer.times) < 2:
        detail = f"\n{error_traceback}" if error_traceback else ""
        raise RuntimeError((error or "collector produced fewer than two observations") + detail)
    features = np.stack(observer.features)
    mask = np.zeros((len(features), MAX_NODES), bool); mask[:, :len(observer.roster)] = True
    success = np.maximum.accumulate(np.asarray(observer.success, bool))
    relations, relation_mask = derive_relations(
        features, np.stack(observer.closed), success, observer.roster, spec,
        np.stack(observer.direct_contact),
    )
    goal, goal_mask = _goal_arrays(spec, observer.roster)
    result = {
        "node_features": features,
        "node_types": np.broadcast_to(
            np.asarray([NODE_TYPES.index(item["type"]) for item in observer.roster]
                       + [0] * (MAX_NODES-len(observer.roster)), np.int64),
            (len(features), MAX_NODES),
        ).copy(),
        "node_mask": mask,
        "edge_features": edge_features(features, mask[0]),
        "relations": relations,
        "relation_mask": relation_mask,
        "events": derive_events(relations, relation_mask, features),
        "goal": goal,
        "goal_mask": goal_mask,
        "ee_state16": np.stack(observer.ee),
        "direct_contact2": np.stack(observer.direct_contact),
        "actions14": canonical_actions(np.stack(observer.ee)),
        "sim_times": np.asarray(observer.times, np.float64),
        "success": success,
        "planner_error": error,
        "roster": [{key: item[key] for key in ("name", "type", "movable", "articulation",
                                                "position_mode", "functional_point_id")}
                   for item in observer.roster],
    }
    return result


def write_episode(path: Path, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists(): temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        for key in ("node_features", "node_types", "node_mask", "edge_features", "relations",
                    "relation_mask", "events", "goal", "goal_mask", "ee_state16", "actions14",
                    "direct_contact2", "sim_times", "success"):
            handle.create_dataset(key, data=payload[key], compression="gzip", compression_opts=1)
        for key, value in metadata.items(): handle.attrs[key] = value
        handle.attrs["planner_error"] = payload["planner_error"] or ""
        handle.attrs["node_roster_json"] = _canonical_json(payload["roster"])
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--body", choices=BODIES, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--scene-backend", choices=("headless-physics", "renderer"),
                        default="headless-physics")
    parser.add_argument("--replay-archive", type=Path,
                        help="official RoboTwin zip containing seed.txt and _traj_data")
    parser.add_argument("--replay-index-start", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.episodes <= 0 or not math.isfinite(cli.sample_hz) or cli.sample_hz <= 0:
        raise ValueError("invalid collection size/rate")
    root = cli.robotwin_root.resolve(); output = cli.output.resolve()
    os.environ["ASSETS_PATH"] = str(root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    sys.path.insert(0, str(root))
    from envs import CONFIGS_PATH  # noqa: F401
    module = __import__(f"envs.{cli.task}", fromlist=[cli.task])
    task_class = getattr(module, cli.task)
    spec = TASK_SPECS[cli.task]
    args = task_args(root, cli.task, cli.body, cli.condition)
    replay = OfficialReplayArchive(cli.replay_archive) if cli.replay_archive else None
    if replay is not None and cli.replay_index_start + cli.episodes > len(replay.seeds):
        raise ValueError("requested replay range exceeds official archive")
    combination = output / cli.task / cli.body / cli.condition
    combination.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(cli.episodes):
        replay_index = cli.replay_index_start + offset
        if replay is None:
            seed = cli.seed_start + offset
            episode_args = dict(args)
        else:
            seed, left_path, right_path = replay.paths(replay_index)
            episode_args = dict(args)
            episode_args.update(
                need_plan=False, left_joint_path=left_path, right_joint_path=right_path
            )
        path = combination / f"episode_seed_{seed}.hdf5"
        if path.exists():
            rows.append({"seed": seed, "path": str(path), "status": "existing"}); continue
        try:
            payload = collect_one(task_class, episode_args, spec, seed, spec.instruction,
                                  cli.sample_hz, cli.scene_backend)
            metadata = {
                "format": FORMAT, "schema_sha256": schema_sha256(), "task": cli.task,
                "body": cli.body, "condition": cli.condition, "seed": seed,
                "instruction": spec.instruction, "sample_hz": cli.sample_hz,
                "scene_backend": cli.scene_backend,
                "replay_archive": str(replay.path) if replay is not None else "",
                "replay_archive_sha256": replay.sha256 if replay is not None else "",
                "replay_index": replay_index if replay is not None else -1,
                "native_success": bool(payload["success"][-1]),
            }
            write_episode(path, payload, metadata)
            rows.append({"seed": seed, "path": str(path), "status": "written",
                         "frames": len(payload["sim_times"]),
                         "success": bool(payload["success"][-1]),
                         "planner_error": payload["planner_error"]})
        except Exception as exc:
            rows.append({"seed": seed, "path": str(path), "status": "error",
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=8)})
    receipt = {
        "format": "etsf_robotwin2_universal_collection_receipt_v1",
        "schema_sha256": schema_sha256(), "task": cli.task, "body": cli.body,
        "condition": cli.condition, "seed_start": cli.seed_start,
        "requested_episodes": cli.episodes, "sample_hz": cli.sample_hz,
        "scene_backend": cli.scene_backend,
        "replay_archive": str(replay.path) if replay is not None else None,
        "replay_archive_sha256": replay.sha256 if replay is not None else None,
        "replay_index_start": cli.replay_index_start if replay is not None else None,
        "episodes": rows,
    }
    receipt["logical_sha256"] = _sha256(receipt)
    receipt_path = combination / f"receipt_seed_{cli.seed_start}_{cli.episodes}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    usable = sum(row["status"] in {"written", "existing"} for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(json.dumps({"receipt": str(receipt_path), "written": usable,
                      "errors": errors}, ensure_ascii=False))
    # A setup/configuration failure used to look like a successful Slurm job because
    # every per-episode exception was only written to the receipt.  Keep partial
    # planner failures as data, but never let an entirely empty array cell pass.
    if usable == 0:
        raise RuntimeError(
            f"collection produced zero usable episodes ({errors} errors); "
            f"inspect {receipt_path}"
        )


if __name__ == "__main__":
    main()
