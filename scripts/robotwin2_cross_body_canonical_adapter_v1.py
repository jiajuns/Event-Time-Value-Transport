#!/usr/bin/env python3
"""Label-free cross-body tensor interfaces for RoboTwin2 actor/critic plumbing.

This module does not infer events, predicates, objects, outcomes or task
success.  It only (1) converts consecutive dual-arm end poses and grippers to
one body-independent 14-D task-space action effect and (2) packs explicitly
provided canonical signals into the reviewed 27-D shared-critic state order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


FORMAT = "etsf_robotwin2_cross_body_canonical_adapter_v1"
STATE_SCHEMA = "dual_ee_object_relative_state_27d_v2"
ACTION_SCHEMA = "dual_ee_se3_gripper_delta_14d_v2"
POSE_CONVENTION = "xyz_plus_quaternion_wxyz"
ACTION_EFFECT14_CHANNELS = (
    "left_delta_x",
    "left_delta_y",
    "left_delta_z",
    "left_delta_axis_angle_x",
    "left_delta_axis_angle_y",
    "left_delta_axis_angle_z",
    "left_delta_gripper",
    "right_delta_x",
    "right_delta_y",
    "right_delta_z",
    "right_delta_axis_angle_x",
    "right_delta_axis_angle_y",
    "right_delta_axis_angle_z",
    "right_delta_gripper",
)
STATE27_CHANNELS = (
    "relative_goal_x",
    "relative_goal_y",
    "relative_goal_z",
    "left_ee_to_object_x",
    "left_ee_to_object_y",
    "left_ee_to_object_z",
    "right_ee_to_object_x",
    "right_ee_to_object_y",
    "right_ee_to_object_z",
    "object_displacement_x",
    "object_displacement_y",
    "object_displacement_z",
    "left_gripper",
    "right_gripper",
    "object_quaternion_w",
    "object_quaternion_x",
    "object_quaternion_y",
    "object_quaternion_z",
    "event_e0",
    "event_e12",
    "event_e3",
    "event_e4",
    "event_eK",
    "predicate_0",
    "predicate_1",
    "predicate_2",
    "predicate_3",
)


class CanonicalAdapterError(ValueError):
    """An explicitly supplied tensor violates the canonical interface."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def contract() -> dict[str, Any]:
    unsigned = {
        "format": FORMAT,
        "state_schema": STATE_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "trainable": False,
        "label_free_geometry_only": True,
        "pose_convention": POSE_CONVENTION,
        "relative_rotation": "q_next_times_conjugate_q_current_shortest_axis_angle",
        "translation_delta_frame": "source_world_frame",
        "action_effect14_channels": list(ACTION_EFFECT14_CHANNELS),
        "state27_channels": list(STATE27_CHANNELS),
        "state27_inputs_are_supplied_not_inferred_by_this_adapter": True,
        "public_expert_hdf5_does_not_supply_complete_state27": True,
        "success_failure_recovery_object_event_labels_generated": False,
    }
    return {**unsigned, "logical_sha256": canonical_sha256(unsigned)}


def _array(value: Any, name: str, trailing: int):
    try:
        import numpy as np
    except ImportError as error:
        raise CanonicalAdapterError("numpy is required for tensor adaptation") from error
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != trailing or not np.isfinite(result).all():
        raise CanonicalAdapterError(f"{name} must be finite with trailing dimension {trailing}")
    return result


def _quaternion_multiply_wxyz(left: Any, right: Any):
    import numpy as np

    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _relative_axis_angle_wxyz(current: Any, following: Any):
    import numpy as np

    current_norm = np.linalg.norm(current, axis=-1, keepdims=True)
    following_norm = np.linalg.norm(following, axis=-1, keepdims=True)
    if np.any(current_norm < 1e-12) or np.any(following_norm < 1e-12):
        raise CanonicalAdapterError("pose quaternion norm is zero")
    current = current / current_norm
    following = following / following_norm
    conjugate = current.copy()
    conjugate[..., 1:] *= -1.0
    relative = _quaternion_multiply_wxyz(following, conjugate)
    relative /= np.linalg.norm(relative, axis=-1, keepdims=True)
    relative = np.where((relative[..., :1] < 0.0), -relative, relative)
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., :1], 0.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1e-10,
    )
    return vector * scale


def task_space_action_effect14(
    left_endpose: Any,
    left_gripper: Any,
    right_endpose: Any,
    right_gripper: Any,
):
    """Return `[T-1,14]` effects from consecutive dual-arm pose samples.

    End poses must be `[T,7]` in `xyz+quaternion_wxyz`; grippers may be `[T]`
    or `[T,1]`.  Translation is a world-frame difference and orientation is
    the shortest relative axis-angle.  No normalization is learned from data.
    """

    import numpy as np

    left = _array(left_endpose, "left_endpose", 7)
    right = _array(right_endpose, "right_endpose", 7)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or left.shape[0] < 2:
        raise CanonicalAdapterError("dual-arm end poses must have equal [T,7], T>=2")

    def gripper(value: Any, name: str):
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 2 and result.shape[1] == 1:
            result = result[:, 0]
        if result.ndim != 1 or result.shape[0] != left.shape[0] or not np.isfinite(result).all():
            raise CanonicalAdapterError(f"{name} must be finite [T] or [T,1]")
        return result

    left_grip = gripper(left_gripper, "left_gripper")
    right_grip = gripper(right_gripper, "right_gripper")
    left_effect = np.concatenate(
        (
            left[1:, :3] - left[:-1, :3],
            _relative_axis_angle_wxyz(left[:-1, 3:], left[1:, 3:]),
            (left_grip[1:] - left_grip[:-1])[:, None],
        ),
        axis=1,
    )
    right_effect = np.concatenate(
        (
            right[1:, :3] - right[:-1, :3],
            _relative_axis_angle_wxyz(right[:-1, 3:], right[1:, 3:]),
            (right_grip[1:] - right_grip[:-1])[:, None],
        ),
        axis=1,
    )
    return np.concatenate((left_effect, right_effect), axis=1).astype(np.float32)


def pack_shared_critic_state27(
    relative_goal3: Any,
    left_ee_to_object3: Any,
    right_ee_to_object3: Any,
    object_displacement3: Any,
    grippers2: Any,
    object_quaternion4: Any,
    event_onehot5: Any,
    predicates4: Any,
):
    """Pack already-derived canonical signals into the frozen 27-D order."""

    import numpy as np

    relative_goal = _array(relative_goal3, "relative_goal3", 3)
    left_relative = _array(left_ee_to_object3, "left_ee_to_object3", 3)
    right_relative = _array(right_ee_to_object3, "right_ee_to_object3", 3)
    displacement = _array(object_displacement3, "object_displacement3", 3)
    grippers = _array(grippers2, "grippers2", 2)
    quaternion = _array(object_quaternion4, "object_quaternion4", 4).copy()
    event_onehot = _array(event_onehot5, "event_onehot5", 5)
    predicates = _array(predicates4, "predicates4", 4)
    quaternion_norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(quaternion_norm < 1e-12):
        raise CanonicalAdapterError("object quaternion norm is zero")
    quaternion /= quaternion_norm
    quaternion = np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    parts = (
        relative_goal,
        left_relative,
        right_relative,
        displacement,
        grippers,
        quaternion,
        event_onehot,
        predicates,
    )
    leading = parts[0].shape[:-1]
    if any(part.shape[:-1] != leading for part in parts[1:]):
        raise CanonicalAdapterError("state27 inputs must have the same leading shape")
    result = np.concatenate(parts, axis=-1).astype(np.float32)
    if result.shape[-1] != 27:
        raise CanonicalAdapterError("internal state27 channel count changed")
    return result


def main() -> int:
    print(json.dumps(contract(), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
