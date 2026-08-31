#!/usr/bin/env python3
"""Frozen SmolVLA Aloha-joint to RoboTwin world-EE policy adapter.

This module is loaded as an XPolicyLab policy.  It deliberately keeps the
SmolVLA actor frozen and converts its native 14-D Aloha joint chunks into the
16-D absolute end-effector convention shared by RoboTwin embodiments:

    left [xyz, qwxyz, gripper] + right [xyz, qwxyz, gripper]

The conversion is analytic forward kinematics.  It contains no learned body
adapter and never updates actor parameters.  The non-active arm is held at its
current target-body pose so single-arm tasks stay single-arm at execution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from XPolicyLab.policy.SmolVLA.model import (
    Model as NativeSmolVLAModel,
    ensure_chw_uint8,
    extract_image,
    raw_observation_to_observation,
    resolve_prompt,
)
from XPolicyLab.utils.process_data import unpack_robot_state


FORMAT = "etsf_robotwin2_frozen_smolvla_aloha_joint_to_world_ee_v1"
NATIVE_ACTION_DIM = 14
ROBOTWIN_EE_ACTION_DIM = 16
LEFT_ARM_JOINTS = tuple(f"fl_joint{index}" for index in range(1, 7))
RIGHT_ARM_JOINTS = tuple(f"fr_joint{index}" for index in range(1, 7))
LEFT_EE_LINK = "fl_link6"
RIGHT_EE_LINK = "fr_link6"


def _sha256_int(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _pose_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [values[4], values[5], values[6], values[3]]
    ).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def _matrix_to_wxyz_pose(matrix: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quat_wxyz = np.asarray(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    if previous is not None and float(np.dot(quat_wxyz, previous)) < 0.0:
        quat_wxyz *= -1.0
    return np.concatenate([matrix[:3, 3], quat_wxyz]).astype(np.float32)


class AlohaJointToWorldEE:
    """Deterministic FK for the exact Aloha embodiment used by RoboTwin."""

    def __init__(self, urdf_path: str | Path, config_path: str | Path):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Aloha URDF is missing: {self.urdf_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Aloha config is missing: {self.config_path}")

        self.robot = URDF.load(str(self.urdf_path))
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        robot_poses = config.get("robot_pose")
        if not isinstance(robot_poses, list) or not robot_poses:
            raise ValueError("Aloha config must contain robot_pose[0].")
        self.world_from_urdf = _pose_matrix(robot_poses[0])

        actuated = {joint.name: joint for joint in self.robot.actuated_joints}
        missing = [
            name
            for name in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
            if name not in actuated
        ]
        if missing:
            raise ValueError(f"Aloha URDF is missing arm joints: {missing}")

        self.lower = np.full(12, -np.inf, dtype=np.float32)
        self.upper = np.full(12, np.inf, dtype=np.float32)
        for index, name in enumerate((*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)):
            limit = actuated[name].limit
            if limit is None:
                continue
            if limit.lower is not None:
                self.lower[index] = float(limit.lower)
            if limit.upper is not None:
                self.upper[index] = float(limit.upper)

        self.home_q14 = np.asarray(
            [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0], dtype=np.float32
        )
        self.home_ee16 = self.convert_chunk(self.home_q14[None])[0]

    def clip_joint_chunk(self, chunk: np.ndarray) -> np.ndarray:
        value = np.asarray(chunk, dtype=np.float32).copy()
        if value.shape[-1] != NATIVE_ACTION_DIM:
            raise ValueError(
                f"SmolVLA native action must be {NATIVE_ACTION_DIM}-D, "
                f"got {value.shape}."
            )
        arm = np.concatenate([value[..., :6], value[..., 7:13]], axis=-1)
        arm = np.clip(arm, self.lower, self.upper)
        value[..., :6] = arm[..., :6]
        value[..., 7:13] = arm[..., 6:]
        value[..., 6] = np.clip(value[..., 6], 0.0, 1.0)
        value[..., 13] = np.clip(value[..., 13], 0.0, 1.0)
        return value

    def convert_chunk(self, chunk: np.ndarray) -> np.ndarray:
        q = self.clip_joint_chunk(chunk)
        flat = q.reshape(-1, NATIVE_ACTION_DIM)
        result = np.empty((flat.shape[0], ROBOTWIN_EE_ACTION_DIM), dtype=np.float32)
        previous_left = None
        previous_right = None
        for row_index, row in enumerate(flat):
            configuration = {
                name: float(value)
                for name, value in zip(LEFT_ARM_JOINTS, row[:6])
            }
            configuration.update(
                {
                    name: float(value)
                    for name, value in zip(RIGHT_ARM_JOINTS, row[7:13])
                }
            )
            self.robot.update_cfg(configuration)
            left = self.world_from_urdf @ self.robot.get_transform(LEFT_EE_LINK)
            right = self.world_from_urdf @ self.robot.get_transform(RIGHT_EE_LINK)
            left_pose = _matrix_to_wxyz_pose(left, previous_left)
            right_pose = _matrix_to_wxyz_pose(right, previous_right)
            previous_left = left_pose[3:].astype(np.float64)
            previous_right = right_pose[3:].astype(np.float64)
            result[row_index] = np.concatenate(
                [left_pose, [row[6]], right_pose, [row[13]]]
            )
        return result.reshape(*q.shape[:-1], ROBOTWIN_EE_ACTION_DIM)


def _current_ee16(observation: Mapping[str, Any], fallback: np.ndarray) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, Mapping):
        return fallback.copy()
    try:
        left_pose = np.asarray(state["left_ee_pose"], dtype=np.float32).reshape(7)
        right_pose = np.asarray(state["right_ee_pose"], dtype=np.float32).reshape(7)
        left_gripper = float(
            np.asarray(state["left_ee_joint_state"], dtype=np.float32).reshape(-1)[0]
        )
        right_gripper = float(
            np.asarray(state["right_ee_joint_state"], dtype=np.float32).reshape(-1)[0]
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return fallback.copy()
    return np.concatenate(
        [left_pose, [left_gripper], right_pose, [right_gripper]]
    ).astype(np.float32)


def _explicit_arm_from_prompt(prompt: str) -> str | None:
    normalized = " ".join(prompt.lower().replace("_", " ").split())
    left = "left arm" in normalized or "left hand" in normalized
    right = "right arm" in normalized or "right hand" in normalized
    if left != right:
        return "left" if left else "right"
    return None


def _motion_selected_arm(chunk: np.ndarray, current_q14: np.ndarray) -> str:
    left_delta = chunk[..., :6] - current_q14[:6]
    right_delta = chunk[..., 7:13] - current_q14[7:13]
    left_energy = float(np.mean(np.square(left_delta)))
    right_energy = float(np.mean(np.square(right_delta)))
    return "left" if left_energy >= right_energy else "right"


def _hold_passive_arm(
    ee_chunk: np.ndarray, current_ee16: np.ndarray, active_arm: str
) -> np.ndarray:
    result = np.asarray(ee_chunk, dtype=np.float32).copy()
    if active_arm == "left":
        result[..., 8:16] = current_ee16[8:16]
    elif active_arm == "right":
        result[..., 0:8] = current_ee16[0:8]
    else:
        raise ValueError(f"Unknown active arm: {active_arm!r}")
    return result


class Model(NativeSmolVLAModel):
    """XPolicyLab policy with a frozen actor and deterministic body adapter."""

    def __init__(self, model_cfg: Mapping[str, Any]):
        requested = dict(model_cfg)
        requested_action_type = str(requested.get("action_type", "ee")).lower()
        if requested_action_type not in {"ee", "endpose"}:
            raise ValueError(
                "Frozen Aloha-to-EE adapter must be deployed with action_type='ee'."
            )

        self.execution_prefix_steps = int(
            requested.get("execution_prefix_steps")
            or requested.get("actions_per_chunk")
            or 5
        )
        if not 1 <= self.execution_prefix_steps <= 50:
            raise ValueError("execution_prefix_steps must be in [1, 50].")
        self.candidate_count = int(requested.get("candidate_count") or 1)
        if self.candidate_count not in {1, 4}:
            raise ValueError("candidate_count must be 1 or 4.")
        self.selection_mode = str(
            requested.get("selection_mode", "actor_candidate0")
        )
        if self.selection_mode != "actor_candidate0":
            raise ValueError(
                "This adapter version only authorizes actor_candidate0. "
                "Use the shared-head policy version for re-ranking."
            )

        robotwin_root = Path(
            requested.get("robotwin_root") or Path(__file__).resolve().parents[3]
        ).expanduser().resolve()
        urdf_path = requested.get("aloha_urdf_path") or (
            robotwin_root
            / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
        )
        aloha_config_path = requested.get("aloha_config_path") or (
            robotwin_root / "assets/embodiments/aloha-agilex/config.yml"
        )

        base_cfg = dict(requested)
        base_cfg["action_type"] = "joint"
        base_cfg["env_cfg_type"] = "aloha_agilex"
        base_cfg["actions_per_chunk"] = self.execution_prefix_steps
        super().__init__(base_cfg)

        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.policy.parameters()):
            raise RuntimeError("Actor freeze assertion failed.")

        self.model_cfg = requested
        self.action_type = "ee"
        self.robot_action_dim_info = {"arm_dim": [7, 7], "ee_dim": [1, 1]}
        self.fk = AlohaJointToWorldEE(urdf_path, aloha_config_path)
        self._virtual_q14: dict[int, np.ndarray] = {}
        self._current_ee16: dict[int, np.ndarray] = {}
        self._case_seed = 0
        self._case_task = str(requested.get("task_name", "unknown"))
        self._query_index = 0
        self._last_selection: dict[int, dict[str, Any]] = {}

        print(
            {
                "format": FORMAT,
                "actor_frozen": True,
                "selection_mode": self.selection_mode,
                "candidate_count": self.candidate_count,
                "execution_prefix_steps": self.execution_prefix_steps,
                "native_action_dim": NATIVE_ACTION_DIM,
                "deployed_action_dim": ROBOTWIN_EE_ACTION_DIM,
            },
            flush=True,
        )

    def prepare_case(self, case_meta: Mapping[str, Any] | None = None):
        case_meta = dict(case_meta or {})
        self._case_seed = int(case_meta.get("seed", 0))
        self._case_task = str(case_meta.get("task_name", self._case_task))
        self._query_index = 0
        return {
            "format": FORMAT,
            "actor_frozen": True,
            "task_name": self._case_task,
            "seed": self._case_seed,
        }

    def _virtual_state(self, env_idx: int) -> np.ndarray:
        return self._virtual_q14.setdefault(env_idx, self.fk.home_q14.copy())

    def _encode_cross_body_observation(
        self, observation: Mapping[str, Any], env_idx: int
    ) -> dict[str, Any]:
        images = {
            "cam_high": ensure_chw_uint8(
                extract_image(
                    observation,
                    ["cam_high", "cam_head", "head_camera", "top_camera"],
                )
            ),
            "cam_left_wrist": ensure_chw_uint8(
                extract_image(
                    observation,
                    ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
                )
            ),
            "cam_right_wrist": ensure_chw_uint8(
                extract_image(
                    observation,
                    ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
                )
            ),
        }
        prompt = resolve_prompt(observation, self.default_prompt)
        self._current_ee16[env_idx] = _current_ee16(
            observation, self.fk.home_ee16
        )
        return {
            "state": self._virtual_state(env_idx).copy(),
            "images": images,
            "prompt": prompt,
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)
        ]
        encoded = [
            self._encode_cross_body_observation(obs, env_idx)
            for obs, env_idx in zip(obs_list, self._latest_env_idx_list)
        ]
        self._latest_payload = encoded[0]
        self._ensure_lerobot_features(self._latest_payload)
        self._latest_payloads = dict(zip(self._latest_env_idx_list, encoded))

    def _infer_native_joint_chunks(
        self, payloads: list[dict[str, Any]], query_seed: int
    ) -> np.ndarray:
        observations = [
            raw_observation_to_observation(payload, self._lerobot_features)
            for payload in payloads
        ]
        observation = self._stack_observations(observations)
        observation = self.preprocessor(observation)
        torch.manual_seed(query_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(query_seed)
        with torch.inference_mode():
            action_tensor = self.policy.predict_action_chunk(observation)
        return self._postprocess_action_chunk(action_tensor)

    def _select_for_env(
        self, env_idx: int, payload: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        query_seed = _sha256_int(
            FORMAT, self._case_task, self._case_seed, self._query_index, env_idx
        )
        candidates = self._infer_native_joint_chunks(
            [payload] * self.candidate_count, query_seed
        )
        candidates = self.fk.clip_joint_chunk(candidates)
        selected_index = 0
        selected_q = candidates[selected_index]
        current_q = self._virtual_state(env_idx)
        active_arm = _explicit_arm_from_prompt(payload["prompt"])
        if active_arm is None:
            active_arm = _motion_selected_arm(selected_q, current_q)
        selected_ee = self.fk.convert_chunk(selected_q)
        selected_ee = _hold_passive_arm(
            selected_ee, self._current_ee16[env_idx], active_arm
        )
        self._last_selection[env_idx] = {
            "query_seed": query_seed,
            "query_index": self._query_index,
            "selected_candidate_index": selected_index,
            "candidate_count": self.candidate_count,
            "active_arm": active_arm,
            "prompt": payload["prompt"],
        }
        return selected_ee, selected_q, active_arm

    @torch.inference_mode()
    def infer_batch_payloads(self, payloads: list[dict[str, Any]]) -> np.ndarray:
        if not payloads:
            raise ValueError("infer_batch_payloads requires at least one payload.")
        if len(payloads) != len(self._latest_env_idx_list):
            raise ValueError(
                "Cross-body inference requires payloads aligned with latest env indices."
            )
        chunks = []
        for env_idx, payload in zip(self._latest_env_idx_list, payloads):
            selected_ee, selected_q, _ = self._select_for_env(env_idx, payload)
            self._virtual_q14[env_idx] = selected_q[-1].copy()
            chunks.append(selected_ee)
        self._query_index += 1
        return np.stack(chunks, axis=0)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        del kwargs
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        else:
            env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        missing = [
            env_idx for env_idx in env_idx_list if env_idx not in self._latest_payloads
        ]
        if missing:
            raise KeyError(f"Missing observations for env_idx: {missing}")
        previous_indices = self._latest_env_idx_list
        self._latest_env_idx_list = list(env_idx_list)
        try:
            raw_batch = self.infer_batch_payloads(
                [self._latest_payloads[env_idx] for env_idx in env_idx_list]
            )
        finally:
            self._latest_env_idx_list = previous_indices
        return [
            unpack_robot_state(
                chunk, "ee", self.robot_action_dim_info, source_type="obs"
            )
            for chunk in raw_batch
        ]

    def get_action(self, **kwargs):
        del kwargs
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    def reset(self):
        if self.policy is not None:
            self.policy.reset()
        self._latest_env_idx_list = [0]
        self._latest_payload = None
        self._latest_payloads = {}
        self._lerobot_features = None
        self._virtual_q14 = {}
        self._current_ee16 = {}
        self._query_index = 0
        self._last_selection = {}


__all__ = ["AlohaJointToWorldEE", "FORMAT", "Model"]
