from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_robotwin2_staging_to_xpolicylab_ee16_v1 as converter  # noqa: E402


def _jpeg_bytes(value: int) -> bytes:
    try:
        import cv2

        image = np.full((4, 6, 3), value, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        return encoded.tobytes()
    except ImportError:
        return b"opaque-jpeg"


def _source(root: Path, *, frames: int = 4) -> tuple[Path, Path]:
    config = root / "piper_clean"
    data_dir = config / "data"
    instruction_dir = config / "instructions"
    data_dir.mkdir(parents=True)
    instruction_dir.mkdir()
    hdf5_path = data_dir / "episode3.hdf5"

    left = np.zeros((frames, 7), dtype=np.float64)
    right = np.zeros((frames, 7), dtype=np.float64)
    left[:, 0] = np.arange(frames)
    right[:, 1] = np.arange(frames) + 10
    left[:, 3] = np.asarray([2.0, -2.0, 2.0, -2.0])[:frames]
    right[:, 3] = 1.0
    images = np.asarray([_jpeg_bytes(index) for index in range(frames)])
    width = max(len(value) for value in images)

    with h5py.File(hdf5_path, "w") as handle:
        endpose = handle.create_group("endpose")
        endpose.create_dataset("left_endpose", data=left)
        endpose.create_dataset("left_gripper", data=np.arange(frames) % 2)
        endpose.create_dataset("right_endpose", data=right)
        endpose.create_dataset("right_gripper", data=np.ones(frames))
        observation = handle.create_group("observation")
        for camera_name in converter.CAMERA_MAP:
            camera = observation.create_group(camera_name)
            camera.create_dataset("rgb", data=np.asarray(images, dtype=f"S{width}"))

    instruction_path = instruction_dir / "episode3.json"
    instruction_path.write_text(
        json.dumps(
            {
                "seen": ["move can zero", "move can one"],
                "unseen": ["unseen prompt"],
            }
        ),
        encoding="utf-8",
    )
    return hdf5_path, instruction_path


def test_next_frame_ee16_semantics_and_xpolicylab_aliases(tmp_path: Path) -> None:
    source, instruction = _source(tmp_path / "input")
    output = tmp_path / "episode.hdf5"
    result = converter.convert_episode(
        source,
        instruction,
        output,
        source_config="piper_clean",
        episode_id=3,
    )

    assert result["source_horizon"] == 4
    assert result["output_horizon"] == 3
    assert result["action_dim"] == 16
    with h5py.File(output, "r") as handle:
        state_left = handle["state/left_ee_poses"][()]
        action_left = handle["action/left_ee_poses"][()]
        assert state_left.shape == (3, 7)
        assert action_left.shape == (3, 7)
        np.testing.assert_allclose(state_left[:, 0], [0, 1, 2])
        np.testing.assert_allclose(action_left[:, 0], [1, 2, 3])
        np.testing.assert_allclose(state_left[:, 3], 1.0)
        np.testing.assert_allclose(action_left[:, 3], 1.0)
        assert (
            handle["state/left_ee_poses"].id
            == handle["state/left_arm_joint_states"].id
        )
        packed_state = np.concatenate(
            [
                handle["state/left_arm_joint_states"][()],
                handle["state/left_ee_joint_states"][()],
                handle["state/right_arm_joint_states"][()],
                handle["state/right_ee_joint_states"][()],
            ],
            axis=1,
        )
        assert packed_state.shape == (3, 16)
        assert handle["instruction"][()].decode() == "move can one"
        assert int(handle["additional_info/frequency"][()]) == 15
        assert sorted(handle["vision"].keys()) == sorted(
            converter.CAMERA_MAP.values()
        )
        assert not any(
            name in handle
            for name in ("critic", "event", "success", "failure", "object_effect")
        )


def test_same_index_alignment_keeps_last_frame(tmp_path: Path) -> None:
    source, instruction = _source(tmp_path / "input")
    output = tmp_path / "same.hdf5"
    converter.convert_episode(
        source,
        instruction,
        output,
        source_config="piper_clean",
        episode_id=3,
        action_alignment="same",
    )
    with h5py.File(output, "r") as handle:
        assert handle["state/left_ee_poses"].shape == (4, 7)
        np.testing.assert_array_equal(
            handle["state/left_ee_poses"][()],
            handle["action/left_ee_poses"][()],
        )


def test_dataset_layout_and_dimension_config_are_directly_transformable(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "staging"
    _source(input_root)
    xpolicy_root = tmp_path / "RoboTwin"
    robot_dir = xpolicy_root / "env_cfg" / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "_robot_info.json").write_text(
        json.dumps(
            {"dual_franka": {"arm_dim": [7, 7], "ee_dim": [1, 1]}}
        ),
        encoding="utf-8",
    )
    output_root = xpolicy_root / "data"
    args = converter.parse_args(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--xpolicylab-project-root",
            str(xpolicy_root),
        ]
    )
    manifest = converter.convert_dataset(args)

    target = (
        output_root
        / converter.DEFAULT_DATASET_NAME
        / "move_can_pot__piper_clean"
        / converter.DEFAULT_ENV_CFG_TYPE
        / "data"
        / "episode_0000000.hdf5"
    )
    assert target.is_file()
    assert manifest["episode_count"] == 1
    assert manifest["labels_generated"] == {
        "critic": False,
        "event": False,
        "success_failure_recovery": False,
        "object_effect": False,
    }
    env_cfg = xpolicy_root / "env_cfg" / f"{converter.DEFAULT_ENV_CFG_TYPE}.yml"
    assert "robot: dual_franka" in env_cfg.read_text(encoding="utf-8")
    assert "collect_freq: 15" in env_cfg.read_text(encoding="utf-8")
