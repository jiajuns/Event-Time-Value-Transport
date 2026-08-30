from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "robotwin2_move_can_pot_analytic_event_spec_v2.py"
CONFIG = (
    ROOT
    / "configs"
    / "robotwin2_move_can_pot_five_body_analytic_event_spec_v2.json"
)
SPEC = importlib.util.spec_from_file_location("success_aligned_event_spec", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
event_spec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(event_spec)


def calibration() -> dict:
    _value, result = event_spec.load_event_spec(CONFIG)
    return result


def pose(x: float, y: float, z: float, roll_deg: float = 0.0) -> list[float]:
    half = math.radians(roll_deg) / 2.0
    return [x, y, z, math.cos(half), math.sin(half), 0.0, 0.0]


def trajectory(rows: list[tuple[list[float], list[float]]]) -> np.ndarray:
    return np.asarray([[can, pot] for can, pot in rows], dtype=np.float64)


def test_frozen_json_sha_and_contract_are_self_consistent() -> None:
    value, result = event_spec.load_event_spec(CONFIG)
    assert value["format"] == event_spec.FORMAT
    assert event_spec.sha256_file(CONFIG) == event_spec.EVENT_SPEC_SHA256
    contract = event_spec.event_contract(result)
    event_spec.validate_event_contract(contract)
    assert contract["event_chain_is_necessary_subset_of_native_success"] is True
    assert contract["eK_is_native_simulator_success"] is True


def test_events_follow_motion_official_position_release_ready_and_success() -> None:
    pot = pose(0.0, 0.0, 0.74)
    poses = trajectory(
        [
            (pose(0.27, 0.10, 0.74), pot),
            (pose(0.25, 0.10, 0.77), pot),
            (pose(0.18, 0.00, 0.77), pot),
            (pose(0.18, 0.00, 0.7405, 90.0), pot),
            (pose(0.18, 0.00, 0.7405, 90.0), pot),
        ]
    )
    predicates, events = event_spec.derive_predicates_and_events(
        poses,
        np.arange(len(poses), dtype=np.float64) / 15.0,
        ["can", "pot"],
        True,
        calibration(),
        0.74,
    )
    assert events.tolist() == [0, 1, 2, 3, 4]
    assert predicates[:, 2].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert predicates[:, 3].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert predicates[:, 4].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("can", "expected_event"),
    [
        (pose(0.0, 0.0, 0.7405, 90.0), 1),
        (pose(0.2, 0.0, 0.7405, 90.0), 1),
        (pose(0.18, 0.035, 0.7405, 90.0), 1),
        (pose(0.18, 0.0, 0.7405, 75.0), 2),
        (pose(0.18, 0.0, 0.7411, 90.0), 2),
    ],
)
def test_official_success_boundaries_keep_public_strictness(
    can: list[float], expected_event: int
) -> None:
    pot = pose(0.0, 0.0, 0.74)
    poses = trajectory([(pose(0.27, 0.10, 0.74), pot), (can, pot)])
    _predicates, events = event_spec.derive_predicates_and_events(
        poses,
        np.asarray([0.0, 0.1]),
        ["can", "pot"],
        False,
        calibration(),
        0.74,
    )
    assert int(events[-1]) == expected_event


def test_left_side_uses_initial_side_sign_and_goal_target() -> None:
    pot = pose(0.0, 0.0, 0.74)
    poses = trajectory(
        [
            (pose(-0.27, 0.10, 0.74), pot),
            (pose(-0.18, 0.0, 0.7405, 90.0), pot),
        ]
    )
    moving, residual = event_spec.goal_vector(
        poses, ["can", "pot"], 1, calibration()
    )
    assert np.allclose(moving, [-0.18, 0.0, 0.7405])
    assert np.allclose(residual, [0.0, 0.0, -0.0005], atol=1e-7)
    _predicates, events = event_spec.derive_predicates_and_events(
        poses,
        np.asarray([0.0, 0.1]),
        ["can", "pot"],
        False,
        calibration(),
        0.74,
    )
    assert events.tolist() == [0, 3]


def test_native_success_must_imply_release_ready_geometry() -> None:
    pot = pose(0.0, 0.0, 0.74)
    poses = trajectory(
        [(pose(0.27, 0.10, 0.74), pot), (pose(0.18, 0.0, 0.77), pot)]
    )
    with pytest.raises(
        event_spec.AnalyticEventSpecError,
        match="native success contradicts",
    ):
        event_spec.derive_predicates_and_events(
            poses,
            np.asarray([0.0, 0.1]),
            ["can", "pot"],
            True,
            calibration(),
            0.74,
        )


def test_randomized_table_height_uses_task_orig_z_not_stable_pot_z() -> None:
    # Public randomized setup can lower the stable pot/table while orig_z was
    # captured before that shift.  Official success still compares to orig_z.
    stable_pot = pose(0.0, 0.0, 0.711)
    poses = trajectory(
        [
            (pose(0.27, 0.10, 0.711), stable_pot),
            (pose(0.18, 0.0, 0.730, 90.0), stable_pot),
        ]
    )
    predicates, events = event_spec.derive_predicates_and_events(
        poses,
        np.asarray([0.0, 0.1]),
        ["can", "pot"],
        True,
        calibration(),
        0.741,
    )
    assert predicates[-1, 3] == 1.0
    assert events.tolist() == [0, 4]


def test_euler_conversion_is_the_exact_transforms3d_public_oracle() -> None:
    import transforms3d as t3d

    quaternion = t3d.euler.euler2quat(
        math.radians(75.0), math.radians(15.0), 0.0
    )
    official_roll, official_pitch, _ = t3d.euler.quat2euler(quaternion)
    observed_roll, observed_pitch = event_spec._roll_pitch_degrees_wxyz(
        np.asarray([quaternion], dtype=np.float64)
    )
    assert observed_roll[0] == math.degrees(official_roll)
    assert observed_pitch[0] == math.degrees(official_pitch)
