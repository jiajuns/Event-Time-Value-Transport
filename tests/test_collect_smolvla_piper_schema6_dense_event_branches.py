from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_smolvla_piper_schema6_dense_event_branches import (  # noqa: E402
    BASELINE_NAME,
    DenseBranchContractError,
    collect_dense_group,
    generate_candidate_query,
    save_schema6_group,
    validate_schema6_group_file,
)
from etsf_schema6_pose_quality import (  # noqa: E402
    GROUP_NAME,
    REGISTRY_FORMAT,
    SPEC_FORMAT,
    registry_sha256,
)
from run_smolvla_piper_r6d_direct_actor_smoke import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    EXPECTED_IMAGE_KEYS,
    INSTRUCTION,
    PREFIX_DIM,
)
from verify_smolvla_piper_zero_shot_preflight import PIPER_ACTION_SLOTS  # noqa: E402


def _registry() -> dict[str, object]:
    return {
        "format": REGISTRY_FORMAT,
        "objects": [
            {
                "name": "can",
                "stable_sim_actor_id": "scene/can/0",
                "asset_model_id": "asset:can:v1",
                "role": "manipulated",
                "is_static": False,
            },
            {
                "name": "pot",
                "stable_sim_actor_id": "scene/pot/0",
                "asset_model_id": "asset:pot:v1",
                "role": "receptacle",
                "is_static": False,
            },
        ],
    }


def _spec() -> dict[str, object]:
    return {
        "format": SPEC_FORMAT,
        "schema_version": 6,
        "object_registry_sha256": registry_sha256(_registry()),
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
            "world_aabb_m": [[-2.0, 2.0], [-2.0, 2.0], [-0.1, 2.0]],
            "quaternion_norm_abs_tolerance": 1e-3,
            "max_step_translation_m": 0.2,
            "max_step_rotation_rad": math.pi / 2,
            "static_object_max_step_translation_m": 1e-6,
            "static_object_max_step_rotation_rad": 1e-6,
            "timestamp_step_min_s": 0.01,
            "timestamp_step_max_s": 0.2,
            "max_physics_substeps_per_control_step": 16,
        },
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "source": "synthetic simulator contract",
            "frozen_before_collection": True,
        },
    }


def _observation(step: int = 0) -> dict[str, object]:
    state = np.zeros((1, ACTION_DIM), dtype=np.float32)
    state[:, [1, 8]] = 0.1
    state[:, [6, 13]] = 0.5
    state[:, 0] = step * 0.01
    main = np.full((1, 6, 7, 3), step, dtype=np.uint8)
    return {
        "states": torch.from_numpy(state),
        "main_images": torch.from_numpy(main),
        "wrist_images": torch.from_numpy(np.stack([main, main + 1], axis=1)),
        "task_descriptions": [INSTRUCTION],
    }


def _query(mask: list[bool], query_index: int, hidden_offset: float = 0.0) -> dict[str, object]:
    feasible = np.asarray(mask, dtype=bool)
    mapped = np.zeros((4, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    mapped[:, :, [1, 8]] = 0.1
    mapped[:, :, [6, 13]] = 0.5
    for candidate in range(4):
        mapped[candidate, :, 0] = candidate * 0.01
    legal = np.flatnonzero(feasible).astype(np.int16)
    return {
        "query_index": query_index,
        "hidden": np.full(PREFIX_DIM, query_index + hidden_offset, dtype=np.float32),
        "processed_state": np.zeros(ACTION_DIM, dtype=np.float32),
        "input_interface": {},
        "native_action_sha256": [f"{candidate + 1:064x}" for candidate in range(4)],
        "native_actions": mapped.copy(),
        "mapped_actions": mapped,
        "noise_sha256": [f"{candidate + 11:064x}" for candidate in range(4)],
        "candidate_prefix_sha256": ["a" * 64] * 4,
        "prefix_bit_exact": True,
        "feasibility_mask": feasible,
        "feasibility": [
            {"accepted": bool(value), "reason": None if value else "outside_bounds"}
            for value in feasible
        ],
        "legal_original_candidate_indices": legal,
        "lowest_legal_original_candidate_index": int(legal[0]) if legal.size else -1,
        "mapping_contract": {"derived_from_equal_14d_width": False},
    }


class _Runtime:
    def __init__(self, *, drift_reset: bool = False, teleport_after_step: bool = False) -> None:
        self.reset_calls = 0
        self.steps = 0
        self.actions: list[np.ndarray] = []
        self.drift_reset = drift_reset
        self.teleport_after_step = teleport_after_step

    def reset(self, seed: int, instruction: str):
        self.reset_calls += 1
        self.steps = 0
        return _observation(0), seed, instruction

    def snapshot(self):
        poses = np.zeros((2, 7), dtype=np.float64)
        poses[:, 3] = 1.0
        poses[:, 2] = 0.7
        poses[0, 0] = self.steps * 0.01
        poses[1, 0] = 0.5
        if self.drift_reset and self.reset_calls > 1 and self.steps == 0:
            poses[0, 0] = 0.02
        teleport = np.zeros(2, dtype=bool)
        if self.teleport_after_step and self.steps == 1:
            teleport[0] = True
        proprio = np.zeros(ACTION_DIM, dtype=np.float32)
        proprio[[1, 8]] = 0.1
        proprio[[6, 13]] = 0.5
        return {
            "object_names": ["can", "pot"],
            "object_poses": poses,
            "proprio": proprio,
            "telemetry": {
                "simulator_timestamp_s": self.steps * 0.05,
                "control_step": self.steps,
                "physics_substep_count": 0 if self.steps == 0 else 4,
                "reset_generation": 0,
                "reset_flag": self.steps == 0,
                "teleport_flag": teleport,
                "simulator_pose_error_flag": np.zeros(2, dtype=bool),
            },
        }

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        return _observation(self.steps), False, False, {"success": False}

    @staticmethod
    def derive_events(poses, _names, success, _event_spec):
        if len(poses) > 1:
            names, steps = ["e0", "e12"], [0, 1]
        else:
            names, steps = ["e0"], [0]
        if success:
            names.append("eK")
            steps.append(len(poses) - 1)
        return names, steps, names, steps

    def mapping(self):
        return {
            "reset": self.reset,
            "snapshot": self.snapshot,
            "step": self.step,
            "derive_events": self.derive_events,
        }


def _collect(runtime: _Runtime, query_fn, *, max_steps: int = 2):
    return collect_dense_group(
        runtime=runtime.mapping(),
        query_fn=query_fn,
        requested_seed=100101000,
        instruction=INSTRUCTION,
        object_registry=_registry(),
        pose_quality_spec=_spec(),
        event_spec={},
        max_steps=max_steps,
    )


def test_legal_candidates_are_reindexed_but_original_indices_are_preserved() -> None:
    runtime = _Runtime()

    def query(_obs, index):
        return _query([False, True, False, True] if index == 0 else [True, False, True, False], index)

    record = _collect(runtime, query, max_steps=2)
    assert record["eligible_original_candidate_indices"].tolist() == [1, 3]
    assert record["baseline_name"] == BASELINE_NAME
    assert record["baseline_original_candidate_index"] == 1
    assert record["raw_deterministic_baseline_claimed"] is False
    assert [branch["branch_index"] for branch in record["branches"]] == [0, 1]
    assert [branch["original_candidate_index"] for branch in record["branches"]] == [1, 3]
    assert [branch["queries"][0]["selected_original_candidate_index"] for branch in record["branches"]] == [1, 3]
    assert all(branch["queries"][1]["selected_original_candidate_index"] == 0 for branch in record["branches"])
    assert all(action.shape == (1, 1, ACTION_DIM) for action in runtime.actions)


def test_all_infeasible_root_is_skipped_without_any_branch_step() -> None:
    runtime = _Runtime()
    record = _collect(runtime, lambda _obs, index: _query([False] * 4, index))
    assert record["status"] == "skipped_fewer_than_two_feasible_root_candidates"
    assert record["branches"] == []
    assert runtime.actions == []
    assert runtime.reset_calls == 1


def test_all_infeasible_continuation_is_right_censored_without_second_step() -> None:
    runtime = _Runtime()

    def query(_obs, index):
        return _query([True, True, False, False] if index == 0 else [False] * 4, index)

    record = _collect(runtime, query, max_steps=3)
    assert all(branch["steps"] == 1 for branch in record["branches"])
    assert all(branch["right_censored"] for branch in record["branches"])
    assert all(branch["queries"][-1]["selected_original_candidate_index"] == -1 for branch in record["branches"])
    assert len(runtime.actions) == 2


def test_reset_pose_drift_fails_before_branch_action() -> None:
    runtime = _Runtime(drift_reset=True)
    with pytest.raises(DenseBranchContractError, match="reset state/object pose/language drift"):
        _collect(runtime, lambda _obs, index: _query([True, True, False, False], index))
    assert runtime.actions == []


class _Capture:
    def __init__(self) -> None:
        self.value = None

    def reset(self):
        self.value = None

    def consume(self):
        return self.value


class _Policy:
    def __init__(self, capture: _Capture) -> None:
        self.capture = capture
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=32,
            image_features={key: object() for key in EXPECTED_IMAGE_KEYS},
        )

    def reset(self):
        pass

    def predict_action_chunk(self, batch, *, noise):
        candidate = int(noise.reshape(-1)[0].item())
        self.capture.value = torch.full((PREFIX_DIM,), float(candidate == 2))
        chunk = torch.zeros((1, CHUNK_SIZE, ACTION_DIM), dtype=torch.float32)
        chunk[:, :, [1, 8]] = 0.1
        chunk[:, :, [6, 13]] = 0.5
        return chunk


def test_prefix_drift_across_native_candidates_fails_closed() -> None:
    capture = _Capture()
    policy = _Policy(capture)

    def preprocess(raw):
        result = dict(raw)
        result["observation.state"] = raw["observation.state"].reshape(1, ACTION_DIM)
        return result

    bounds = [[slot.lower, slot.upper] for slot in PIPER_ACTION_SLOTS]
    with pytest.raises(DenseBranchContractError, match="prefix drift"):
        generate_candidate_query(
            policy=policy,
            preprocessor=preprocess,
            postprocessor=lambda value: value,
            capture=capture,
            observation=_observation(),
            bounds=bounds,
            device=torch.device("cpu"),
            scene_seed=100101000,
            query_index=0,
            noise_factory=lambda _cfg, _seed, _query, candidate, _device: torch.full((1, CHUNK_SIZE, 32), float(candidate)),
        )


def test_schema6_pose_mask_blocks_bad_pose_from_object_supervision(tmp_path: Path) -> None:
    runtime = _Runtime(teleport_after_step=True)
    record = _collect(runtime, lambda _obs, index: _query([True, True, False, False], index), max_steps=1)
    assert all(not branch["object_delta_supervision_valid"][0, 0] for branch in record["branches"])
    assert all(branch["object_delta_invalid_reason_bitset"][0, 0] != 0 for branch in record["branches"])
    path = tmp_path / "schema6_group.hdf5"
    save_schema6_group(path, record)
    audit = validate_schema6_group_file(path)
    assert audit["eligible_original_candidate_indices"] == [0, 1]
    with h5py.File(path, "r") as handle:
        for branch in handle["branches"].values():
            assert GROUP_NAME in branch
            assert not branch["object_delta_supervision_valid"][0, 0]
