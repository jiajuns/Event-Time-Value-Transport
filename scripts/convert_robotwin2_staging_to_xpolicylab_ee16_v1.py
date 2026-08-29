#!/usr/bin/env python3
"""Convert staged RoboTwin2 episodes to XPolicyLab trajectory v1.0 EE16.

The actor state and action have one body-independent meaning for every robot:

    left  [x, y, z, qw, qx, qy, qz, gripper]
    right [x, y, z, qw, qx, qy, qz, gripper]

By default ``action[t]`` is the absolute end-effector target at frame ``t+1``
and the final observation is dropped.  This script only converts public expert
actor trajectories.  It never reads trajectory pickle files and never creates
critic, event, success, failure, recovery, or object-effect labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - compressed JPEG input needs no encoder
    cv2 = None


FORMAT = "etsf_robotwin2_xpolicylab_ee16_conversion_v1"
DATA_FORMAT_VERSION = "v1.0"
ACTION_SCHEMA = "dual_absolute_ee_pose7_gripper1_left_then_right_v1"
POSE_ORDER = ("x", "y", "z", "qw", "qx", "qy", "qz")
DEFAULT_DATASET_NAME = "RoboTwin2_move_can_pot_EE16"
DEFAULT_TASK = "move_can_pot"
DEFAULT_ENV_CFG_TYPE = "etsf_ee16_15hz"
DEFAULT_FREQUENCY = 15
CAMERA_MAP = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
EPISODE_RE = re.compile(r"^episode_?(?P<episode>[0-9]+)\.hdf5$")


class EE16ConversionError(RuntimeError):
    """An episode does not satisfy the public EE16 actor contract."""


@dataclass(frozen=True)
class SourceEpisode:
    source_config: str
    episode_id: int
    hdf5_path: Path
    instruction_path: Path


def _matrix(value: Any, name: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and width == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != width:
        raise EE16ConversionError(
            f"{name} must have shape [T,{width}], got {array.shape}"
        )
    if array.shape[0] < 1 or not np.isfinite(array).all():
        raise EE16ConversionError(f"{name} is empty or contains non-finite values")
    return array


def canonicalize_absolute_pose7(value: Any, name: str) -> np.ndarray:
    """Return float32 pose7 with unit, temporally sign-continuous qwxyz."""

    pose = _matrix(value, name, 7).copy()
    quaternion = pose[:, 3:7].astype(np.float64)
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 1e-8):
        raise EE16ConversionError(f"{name} contains a zero quaternion")
    quaternion /= norms[:, None]

    # q and -q encode the same rotation.  Fix the initial hemisphere and then
    # follow the closest representation, eliminating artificial action jumps.
    first_nonzero = np.flatnonzero(np.abs(quaternion[0]) > 1e-12)
    if first_nonzero.size and quaternion[0, first_nonzero[0]] < 0.0:
        quaternion[0] *= -1.0
    for index in range(1, quaternion.shape[0]):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    pose[:, 3:7] = quaternion.astype(np.float32)
    return pose


def _load_instruction_set(path: Path, split: str) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EE16ConversionError(f"invalid instruction JSON: {path}") from error

    if isinstance(payload, str):
        candidates: Iterable[Any] = (payload,)
    elif isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        keys = ("seen", "unseen") if split == "all" else (split,)
        candidates = (
            item
            for key in keys
            for item in (
                payload.get(key, [])
                if isinstance(payload.get(key, []), list)
                else (payload.get(key),)
            )
        )
    else:
        candidates = ()

    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    if not result:
        raise EE16ConversionError(f"no non-empty {split} instruction in {path}")
    return result


def _instruction_for_episode(
    instructions: Sequence[str], episode_id: int, instruction_index: int
) -> str:
    if instruction_index < 0:
        return instructions[episode_id % len(instructions)]
    return instructions[instruction_index % len(instructions)]


def _image_bytes(frame: Any) -> bytes:
    if isinstance(frame, (bytes, np.bytes_)):
        return bytes(frame).rstrip(b"\0")
    array = np.asarray(frame)
    if array.ndim == 0 and array.dtype.kind in {"S", "O"}:
        return bytes(array.item()).rstrip(b"\0")
    if array.ndim == 3:
        if cv2 is None:
            raise EE16ConversionError("OpenCV is required for uncompressed RGB input")
        ok, encoded = cv2.imencode(".jpg", array)
        if not ok:
            raise EE16ConversionError("could not JPEG-encode an RGB frame")
        return encoded.tobytes()
    if array.dtype == np.uint8:
        return array.tobytes().rstrip(b"\0")
    raise EE16ConversionError(f"unsupported RGB frame representation: {array.shape}")


def _write_image_sequence(group: h5py.Group, frames: Any, horizon: int) -> None:
    if len(frames) < horizon:
        raise EE16ConversionError(
            f"camera has {len(frames)} frames but state/action require {horizon}"
        )
    encoded = [_image_bytes(frame) for frame in frames[:horizon]]
    if any(not frame for frame in encoded):
        raise EE16ConversionError("camera contains an empty encoded frame")
    width = max(len(frame) for frame in encoded)
    group.create_dataset("colors", data=np.asarray(encoded, dtype=f"S{width}"))

    if cv2 is not None:
        decoded = cv2.imdecode(
            np.frombuffer(encoded[0], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is not None:
            group.create_dataset("shape", data=np.asarray(decoded.shape, dtype=np.int32))


def _write_arm(
    group: h5py.Group,
    side: str,
    pose: np.ndarray,
    gripper: np.ndarray,
) -> None:
    # left/right_ee_poses are the semantically correct XPolicyLab EE fields.
    # The hard-link aliases are required by the current generic LeRobot-v3
    # converter, which still looks for its historical arm-joint field names.
    pose_dataset = group.create_dataset(f"{side}_ee_poses", data=pose)
    group[f"{side}_arm_joint_states"] = pose_dataset
    group.create_dataset(f"{side}_ee_joint_states", data=gripper)


def convert_episode(
    source_path: Path,
    instruction_path: Path,
    output_path: Path,
    *,
    source_config: str,
    episode_id: int,
    frequency: int = DEFAULT_FREQUENCY,
    action_alignment: str = "next",
    instruction_split: str = "seen",
    instruction_index: int = -1,
) -> dict[str, Any]:
    """Convert one staged public episode and return its small audit summary."""

    if action_alignment not in {"next", "same"}:
        raise ValueError("action_alignment must be 'next' or 'same'")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    instructions = _load_instruction_set(instruction_path, instruction_split)
    instruction = _instruction_for_episode(
        instructions, episode_id, instruction_index
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    try:
        with h5py.File(source_path, "r") as source:
            required = (
                "endpose/left_endpose",
                "endpose/left_gripper",
                "endpose/right_endpose",
                "endpose/right_gripper",
                "observation",
            )
            missing = [name for name in required if name not in source]
            if missing:
                raise EE16ConversionError(f"missing source fields: {missing}")

            left_pose = canonicalize_absolute_pose7(
                source["endpose/left_endpose"][()], "endpose/left_endpose"
            )
            right_pose = canonicalize_absolute_pose7(
                source["endpose/right_endpose"][()], "endpose/right_endpose"
            )
            left_gripper = _matrix(
                source["endpose/left_gripper"][()], "endpose/left_gripper", 1
            )
            right_gripper = _matrix(
                source["endpose/right_gripper"][()], "endpose/right_gripper", 1
            )
            lengths = {
                left_pose.shape[0],
                right_pose.shape[0],
                left_gripper.shape[0],
                right_gripper.shape[0],
            }
            if len(lengths) != 1:
                raise EE16ConversionError("endpose/gripper horizon mismatch")
            source_horizon = lengths.pop()
            if action_alignment == "next" and source_horizon < 2:
                raise EE16ConversionError("next-frame action needs at least two frames")

            if action_alignment == "next":
                state_slice, action_slice = slice(None, -1), slice(1, None)
                horizon = source_horizon - 1
            else:
                state_slice = action_slice = slice(None)
                horizon = source_horizon

            observation = source["observation"]
            missing_cameras = [
                name
                for name in CAMERA_MAP
                if name not in observation or "rgb" not in observation[name]
            ]
            if missing_cameras:
                raise EE16ConversionError(
                    f"three-camera actor input is incomplete: {missing_cameras}"
                )

            with h5py.File(temporary, "w") as output:
                output.attrs["source_format"] = "RoboTwin2_legacy_public_staging"
                output.attrs["source_config"] = source_config
                output.attrs["source_episode_id"] = int(episode_id)
                output.attrs["actor_action_schema"] = ACTION_SCHEMA
                output.attrs["pose_order"] = ",".join(POSE_ORDER)
                output.attrs["action_alignment"] = action_alignment
                output.attrs["critic_labels_generated"] = False
                strings = h5py.string_dtype(encoding="utf-8")
                output.create_dataset(
                    "data_format_version", data=DATA_FORMAT_VERSION, dtype=strings
                )
                output.create_dataset("instruction", data=instruction, dtype=strings)
                output.create_dataset(
                    "instructions",
                    data=np.asarray(instructions, dtype=object),
                    dtype=strings,
                )
                info = output.create_group("additional_info")
                info.create_dataset("frequency", data=np.int32(frequency))
                info.create_dataset("action_dim", data=np.int32(16))

                state = output.create_group("state")
                action = output.create_group("action")
                _write_arm(
                    state,
                    "left",
                    left_pose[state_slice],
                    left_gripper[state_slice],
                )
                _write_arm(
                    state,
                    "right",
                    right_pose[state_slice],
                    right_gripper[state_slice],
                )
                _write_arm(
                    action,
                    "left",
                    left_pose[action_slice],
                    left_gripper[action_slice],
                )
                _write_arm(
                    action,
                    "right",
                    right_pose[action_slice],
                    right_gripper[action_slice],
                )

                vision = output.create_group("vision")
                for source_name, target_name in CAMERA_MAP.items():
                    camera = vision.create_group(target_name)
                    _write_image_sequence(
                        camera, observation[source_name]["rgb"], horizon
                    )

        os.replace(temporary, output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "source_config": source_config,
        "episode_id": episode_id,
        "source_horizon": source_horizon,
        "output_horizon": horizon,
        "action_alignment": action_alignment,
        "action_dim": 16,
        "camera_count": 3,
        "instruction_split": instruction_split,
        "instruction_index": (
            episode_id % len(instructions)
            if instruction_index < 0
            else instruction_index % len(instructions)
        ),
        "output_path": str(output_path),
    }


def _source_config_directories(input_root: Path) -> list[Path]:
    """Find staged config directories from either a staging root or task root."""

    direct = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and (path / "data").is_dir()
    ]
    if direct:
        return sorted(direct)
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_dir()
        and (path / "data").is_dir()
        and (path / "instructions").is_dir()
    )


def discover_episodes(
    input_roots: Path | Sequence[Path],
    source_configs: Sequence[str] | None = None,
) -> list[SourceEpisode]:
    selected = set(source_configs or ())
    result: list[SourceEpisode] = []
    roots = [input_roots] if isinstance(input_roots, Path) else list(input_roots)
    if not roots:
        raise EE16ConversionError("at least one input root is required")
    seen_sources: set[tuple[str, int]] = set()
    for input_root in roots:
        for config_dir in _source_config_directories(input_root):
            if selected and config_dir.name not in selected:
                continue
            data_dir = config_dir / "data"
            instruction_dir = config_dir / "instructions"
            if not instruction_dir.is_dir():
                raise EE16ConversionError(
                    f"missing instruction directory: {instruction_dir}"
                )
            for hdf5_path in sorted(data_dir.glob("*.hdf5")):
                match = EPISODE_RE.fullmatch(hdf5_path.name)
                if match is None:
                    continue
                episode_id = int(match.group("episode"))
                identity = (config_dir.name, episode_id)
                if identity in seen_sources:
                    raise EE16ConversionError(
                        f"duplicate staged episode across input roots: {identity}"
                    )
                seen_sources.add(identity)
                instruction_path = instruction_dir / f"episode{episode_id}.json"
                if not instruction_path.is_file():
                    raise EE16ConversionError(
                        f"missing instruction for {hdf5_path}: {instruction_path}"
                    )
                result.append(
                    SourceEpisode(
                        source_config=config_dir.name,
                        episode_id=episode_id,
                        hdf5_path=hdf5_path,
                        instruction_path=instruction_path,
                    )
                )
    if selected:
        found = {row.source_config for row in result}
        missing = sorted(selected - found)
        if missing:
            raise EE16ConversionError(f"requested source configs not found: {missing}")
    if not result:
        raise EE16ConversionError(f"no staged episodes found under {roots}")
    return sorted(result, key=lambda row: (row.source_config, row.episode_id))


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise EE16ConversionError(f"cannot make output component from {value!r}")
    return result


def install_xpolicylab_env_config(
    project_root: Path, env_cfg_type: str, frequency: int
) -> Path:
    """Install the tiny dimension/frequency config required by the converter."""

    env_cfg_dir = project_root / "env_cfg"
    robot_info = env_cfg_dir / "robot" / "_robot_info.json"
    if not robot_info.is_file():
        raise EE16ConversionError(
            f"XPolicyLab project root is missing env_cfg/robot/_robot_info.json: {project_root}"
        )
    try:
        robots = json.loads(robot_info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EE16ConversionError(f"cannot read {robot_info}") from error
    dual_franka = robots.get("dual_franka") if isinstance(robots, dict) else None
    if dual_franka != {"arm_dim": [7, 7], "ee_dim": [1, 1]}:
        raise EE16ConversionError(
            "XPolicyLab dual_franka dimension carrier is not [7+1,7+1]"
        )

    target = env_cfg_dir / f"{_safe_component(env_cfg_type)}.yml"
    content = (
        "# ETSF canonical EE16 actor data; dual_franka supplies dimensions only.\n"
        "config:\n"
        "  robot: dual_franka\n\n"
        "observation:\n"
        f"  collect_freq: {frequency}\n"
    )
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise EE16ConversionError(f"refusing to overwrite different config: {target}")
        return target
    target.write_text(content, encoding="utf-8")
    return target


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    episodes = discover_episodes(args.input_root, args.source_config)
    config_path = None
    if args.xpolicylab_project_root is not None:
        config_path = install_xpolicylab_env_config(
            args.xpolicylab_project_root, args.env_cfg_type, args.frequency
        )

    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for episode in episodes:
        task_name = _safe_component(f"{args.task}__{episode.source_config}")
        output_dir = (
            args.output_root
            / _safe_component(args.dataset_name)
            / task_name
            / _safe_component(args.env_cfg_type)
            / "data"
        )
        local_index = counters.get(episode.source_config, 0)
        counters[episode.source_config] = local_index + 1
        output_path = output_dir / f"episode_{local_index:07d}.hdf5"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {output_path}")
        rows.append(
            convert_episode(
                episode.hdf5_path,
                episode.instruction_path,
                output_path,
                source_config=episode.source_config,
                episode_id=episode.episode_id,
                frequency=args.frequency,
                action_alignment=args.action_alignment,
                instruction_split=args.instruction_split,
                instruction_index=args.instruction_index,
            )
        )

    manifest = {
        "format": FORMAT,
        "source_roots": [str(path.resolve()) for path in args.input_root],
        "output_root": str(args.output_root.resolve()),
        "dataset_name": args.dataset_name,
        "task": args.task,
        "env_cfg_type": args.env_cfg_type,
        "xpolicylab_env_config": str(config_path) if config_path else None,
        "frequency": args.frequency,
        "action_schema": ACTION_SCHEMA,
        "pose_order": list(POSE_ORDER),
        "state_dim": 16,
        "action_dim": 16,
        "action_alignment": args.action_alignment,
        "camera_map": CAMERA_MAP,
        "episode_count": len(rows),
        "source_config_counts": counters,
        "episodes": rows,
        "labels_generated": {
            "critic": False,
            "event": False,
            "success_failure_recovery": False,
            "object_effect": False,
        },
    }
    manifest_path = (
        args.output_root
        / _safe_component(args.dataset_name)
        / "ee16_conversion_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert staged RoboTwin2 legacy HDF5 to XPolicyLab trajectory v1.0 "
            "with fixed dual-arm absolute EE16 state/action."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        action="append",
        help=(
            "Staging root or task root. Repeat it to combine clean250 and "
            "randomized2500 without copying either tree."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="XPolicyLab DATA_ROOT (normally RoboTwin/data).",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--env-cfg-type", default=DEFAULT_ENV_CFG_TYPE)
    parser.add_argument("--frequency", type=int, default=DEFAULT_FREQUENCY)
    parser.add_argument(
        "--source-config",
        action="append",
        default=[],
        help="Convert only this staged config; repeat for more. Default: all.",
    )
    parser.add_argument(
        "--action-alignment", choices=("next", "same"), default="next"
    )
    parser.add_argument(
        "--instruction-split", choices=("seen", "unseen", "all"), default="seen"
    )
    parser.add_argument(
        "--instruction-index",
        type=int,
        default=-1,
        help="Negative cycles deterministically by source episode; otherwise fixed modulo.",
    )
    parser.add_argument(
        "--xpolicylab-project-root",
        type=Path,
        default=None,
        help="Install env_cfg/<env-cfg-type>.yml so the v3 converter can run directly.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    manifest = convert_dataset(parse_args(argv))
    print(
        json.dumps(
            {
                "format": manifest["format"],
                "episode_count": manifest["episode_count"],
                "source_config_counts": manifest["source_config_counts"],
                "action_schema": manifest["action_schema"],
                "output_root": manifest["output_root"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
