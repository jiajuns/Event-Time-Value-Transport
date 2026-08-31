from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import collect_robotwin2_five_body_ee_candidate_branches_v1 as collector  # noqa: E402
import robotwin2_move_can_pot_analytic_event_spec_v2 as analytic_event  # noqa: E402
import watch_robotwin2_ee16_actor_to_five_body_branches_v1 as watcher  # noqa: E402


class _NoiseConfig:
    chunk_size = 3
    max_action_dim = 4


class _EndPoseRobot:
    def get_left_gripper_val(self) -> float:
        return 0.25

    def get_right_gripper_val(self) -> float:
        return 0.75

    def get_left_tcp_pose(self) -> np.ndarray:
        raise AssertionError("TCP frame must never be read for the actor state")

    def get_right_tcp_pose(self) -> np.ndarray:
        raise AssertionError("TCP frame must never be read for the actor state")


class _EndPoseTask:
    def __init__(self) -> None:
        self.robot = _EndPoseRobot()
        self.calls: list[str] = []

    def get_arm_pose(self, arm: str) -> np.ndarray:
        self.calls.append(arm)
        if arm == "left":
            return np.asarray([1, 2, 3, 1, 0, 0, 0], dtype=np.float64)
        if arm == "right":
            return np.asarray([4, 5, 6, 0, 1, 0, 0], dtype=np.float64)
        raise AssertionError(f"unexpected arm: {arm}")


class _JointRobot(_EndPoseRobot):
    def get_left_arm_jointState(self) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25]

    def get_right_arm_jointState(self) -> list[float]:
        return [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.75]


class _JointTask(_EndPoseTask):
    def __init__(self, arm_tag: str = "left") -> None:
        super().__init__()
        self.robot = _JointRobot()
        self.arm_tag = arm_tag


class _FakeJointToEE:
    def clip_joint_chunk(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).copy()

    def convert_chunk(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        result = np.zeros((len(value), 16), dtype=np.float32)
        result[:, 0:3] = value[:, 0:3]
        result[:, 3] = 1.0
        result[:, 7] = value[:, 6]
        result[:, 8:11] = value[:, 7:10]
        result[:, 11] = 1.0
        result[:, 15] = value[:, 13]
        return result


def test_current_ee_action16_uses_training_endpose_api_and_exact_layout() -> None:
    task = _EndPoseTask()
    value = collector.current_ee_action16(task)
    assert task.calls == ["left", "right"]
    assert value.dtype == np.float32
    np.testing.assert_array_equal(
        value,
        np.asarray(
            [1, 2, 3, 1, 0, 0, 0, 0.25, 4, 5, 6, 0, 1, 0, 0, 0.75],
            dtype=np.float32,
        ),
    )


def test_official_joint14_state_and_single_arm_candidate_contract() -> None:
    task = _JointTask("left")
    current = collector.current_aloha_joint_action14(task)
    np.testing.assert_allclose(
        current,
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.25,
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            0.75,
        ],
    )
    proposed = np.repeat(np.arange(14, dtype=np.float32)[None], 3, axis=0)
    deployed, canonical, active_arm = collector._single_arm_joint_chunk(
        task, proposed, _FakeJointToEE()
    )
    assert active_arm == "left"
    np.testing.assert_array_equal(
        deployed[:, 7:14], np.repeat(current[None, 7:14], 3, axis=0)
    )
    np.testing.assert_array_equal(deployed[:, :7], proposed[:, :7])
    assert canonical.shape == (3, collector.NATIVE_EE_DIM)
    assert collector.state_action_frame_contract("aloha_joint14") == (
        collector.ALOHA_JOINT14_STATE_ACTION_FRAME_CONTRACT
    )
    assert collector.ALOHA_JOINT14_STATE_ACTION_FRAME_CONTRACT[
        "environment_call"
    ].endswith("action_type=qpos)")


def test_state_action_frame_contract_is_explicit_and_rejects_old_tcp_artifacts(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "execute5.json"
    protocol_value = watcher.actor_execution.execution_protocol(5)
    protocol_sha = watcher.actor_execution.write_execution_protocol_file(
        protocol_path, protocol_value
    )
    watcher.configure_execution_protocol(
        protocol_value,
        protocol_path=protocol_path,
        protocol_file_sha256=protocol_sha,
        run_root=tmp_path / "run",
        path_root=tmp_path,
    )
    assert collector.STATE_ACTION_FRAME_CONTRACT == watcher.STATE_ACTION_FRAME_CONTRACT
    assert watcher.STATE_ACTION_FRAME_CONTRACT["runtime_state_api"] == (
        "task.get_arm_pose(left/right)"
    )
    assert watcher.STATE_ACTION_FRAME_CONTRACT["tcp_tool_axis_offset_m_excluded"] == 0.12
    assert watcher.STATE_ACTION_FRAME_CONTRACT["state_and_action_same_frame"] is True
    assert collector.FORMAT == watcher.COLLECTOR_FORMAT
    assert collector.MANIFEST_FORMAT == watcher.MANIFEST_FORMAT
    assert collector.DIAGNOSTIC_FORMAT == watcher.DIAGNOSTIC_FORMAT

    old_tcp_artifact = {
        "format": "etsf_robotwin2_five_body_lobo_training_binding_v1",
        "state_action_frame_contract": {
            "runtime_state_api": "robot.get_left_tcp_pose/get_right_tcp_pose"
        },
    }
    for artifact in ("manifest", "actor authority", "training binding"):
        with pytest.raises(watcher.ContinuationError, match="exact training-aligned"):
            watcher.require_state_action_frame_contract(
                old_tcp_artifact, artifact=artifact
            )
    with pytest.raises(watcher.ContinuationError, match="format"):
        watcher.validate_training_binding_contract(old_tcp_artifact)

    watcher.validate_training_binding_contract(
        {
            "format": watcher.BINDING_FORMAT,
            "state_action_frame_contract": watcher.STATE_ACTION_FRAME_CONTRACT,
            "path_root": str(tmp_path.resolve()),
            "actor_execution_protocol_binding": (
                watcher.require_execution_protocol_binding()
            ),
        }
    )


def test_antithetic_noise_preserves_candidate_zero_and_normal_marginals() -> None:
    device = torch.device("cpu")
    scene_seed, query_index = 19, 7
    values = [
        collector.make_noise(_NoiseConfig, scene_seed, query_index, index, device)
        for index in range(4)
    ]
    legacy_seed = int(
        (20260903 + scene_seed * 1_000_003 + query_index * 10_007) % (2**63 - 1)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(legacy_seed)
    legacy_zero = torch.randn(
        (1, _NoiseConfig.chunk_size, _NoiseConfig.max_action_dim),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    assert torch.equal(values[0], legacy_zero)
    assert torch.equal(values[1], -values[0])
    assert torch.equal(values[3], -values[2])
    assert not torch.equal(values[0], values[2])
    assert collector.CANDIDATE_NOISE_CONTRACT == watcher.CANDIDATE_NOISE_CONTRACT


def test_single_query_resume_preserves_full_manifest_query_universe() -> None:
    requested, declared = collector.resolve_query_contract(
        [30], watcher.ROOT_QUERIES
    )
    assert requested == [30]
    assert declared == list(range(40))
    command = watcher.collector_command(
        {"collector": Path("/code/collector.py")},
        body="piper",
        conditions=("clean",),
        seed_start=2026081050,
        seed_count=1,
        queries=(30,),
    )
    active = command.index("--root-query-indices")
    universe = command.index("--manifest-root-query-indices")
    assert command[active + 1 : universe] == ["30"]
    assert command[universe + 1 : universe + 41] == [
        str(query) for query in range(40)
    ]

    with pytest.raises(collector.BranchCollectionError):
        collector.resolve_query_contract([40], watcher.ROOT_QUERIES)


def test_base_collection_schedule_round_robins_all_body_query_blocks() -> None:
    jobs = watcher.base_collection_jobs()
    assert jobs[: len(watcher.BODIES)] == [
        (body, 0) for body in watcher.COLLECTION_PRIORITY
    ]
    assert len(jobs) == len(watcher.BODIES) * (
        len(watcher.ROOT_QUERIES) // watcher.QUERY_BLOCK_SIZE
    )
    for body in watcher.BODIES:
        assert [block for scheduled_body, block in jobs if scheduled_body == body] == list(
            range(0, len(watcher.ROOT_QUERIES), watcher.QUERY_BLOCK_SIZE)
        )


def test_supplemental_gap_advances_past_unstable_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "piper"
    manifest = {
        "groups": [
            {
                "condition": condition,
                "root_query_index": query,
                "requested_seed": 2026082000 + index,
            }
            for condition in watcher.CONDITIONS
            for query in watcher.ROOT_QUERIES
            for index in range(watcher.TARGET_PER_CONDITION_QUERY)
            if not (condition == "clean" and query == 0 and index == 4)
        ]
    }
    attempts: list[int] = []
    progress_writes: list[dict[str, int]] = []

    monkeypatch.setattr(watcher, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(watcher, "load_manifest", lambda *_args: manifest)
    monkeypatch.setattr(watcher, "finalize_body_manifest", lambda *_args: {"ok": True})
    monkeypatch.setattr(watcher, "load_progress", lambda _body: {
        f"{condition}|{query}": watcher.SUPPLEMENTAL_SEED_START
        for condition in watcher.CONDITIONS
        for query in watcher.ROOT_QUERIES
    })
    monkeypatch.setattr(
        watcher,
        "atomic_json",
        lambda _path, value: progress_writes.append(dict(value)),
    )
    monkeypatch.setattr(watcher, "write_state", lambda *_args, **_kwargs: None)

    def fake_run_collector(*_args, **kwargs):
        seed = kwargs["seed_start"]
        attempts.append(seed)
        if len(attempts) == 1:
            raise watcher.ContinuationError("simulated UnStableError")
        manifest["groups"].append(
            {
                "condition": kwargs["conditions"][0],
                "root_query_index": kwargs["queries"][0],
                "requested_seed": seed,
            }
        )

    monkeypatch.setattr(watcher, "run_collector", fake_run_collector)
    assert watcher.complete_body({}, body) == {"ok": True}
    assert attempts == [watcher.SUPPLEMENTAL_SEED_START, watcher.SUPPLEMENTAL_SEED_START + 1]
    assert progress_writes[-1]["clean|0"] == watcher.SUPPLEMENTAL_SEED_START + 2


def _pose(x: float, quaternion: np.ndarray | None = None) -> np.ndarray:
    quaternion = (
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if quaternion is None
        else np.asarray(quaternion, dtype=np.float32)
    )
    return np.r_[np.asarray([x, 0.0, 0.75], dtype=np.float32), quaternion]


def _trajectory(xs: list[float], quaternion: np.ndarray | None = None) -> np.ndarray:
    rows = []
    for index, x in enumerate(xs):
        can_quaternion = quaternion if index == len(xs) - 1 else None
        rows.append(np.stack((_pose(x, can_quaternion), _pose(0.0))))
    return np.asarray(rows, dtype=np.float32)


def _candidate(index: int, *, horizon: int = 5) -> np.ndarray:
    current = np.asarray(
        [
            0.45, 0.10, 0.80, 1.0, 0.0, 0.0, 0.0, 0.2,
            -0.45, 0.10, 0.80, 1.0, 0.0, 0.0, 0.0, 0.2,
        ],
        dtype=np.float32,
    )
    result = np.repeat(current[None], horizon, axis=0)
    result[:, 0] += (index + 1) * np.linspace(0.002, 0.010, horizon)
    result[:, 8] -= (index + 1) * np.linspace(0.001, 0.005, horizon)
    result[:, 7] = np.clip(result[:, 7] + 0.05 * index, 0.0, 1.0)
    return result


def _root_and_outcomes() -> tuple[dict, list[dict]]:
    current = np.asarray(
        [
            0.45, 0.10, 0.80, 1.0, 0.0, 0.0, 0.0, 0.2,
            -0.45, 0.10, 0.80, 1.0, 0.0, 0.0, 0.0, 0.2,
        ],
        dtype=np.float32,
    )
    prefix = _trajectory([0.30])
    root = {
        "object_names": ["can", "pot"],
        "root_ee_action": current,
        "prefix_trajectory": prefix,
        "prefix_sim_times": np.asarray([0.0], dtype=np.float64),
        "remaining_action_budget": 200,
        "success_height_reference_z": 0.75,
        "candidates": np.stack([_candidate(index) for index in range(4)]),
    }
    quarter_turn_roll = np.asarray(
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32
    )
    specifications = (
        (_trajectory([0.30, 0.28]), [0.0, 0.10], False, 5, None),
        (_trajectory([0.30, 0.18], quarter_turn_roll), [0.0, 0.10], True, 5, None),
        (_trajectory([0.30]), [0.0], False, 0, None),
        (
            _trajectory([0.30, 0.18, 0.18, 0.18], quarter_turn_roll),
            [0.0, 0.05, 0.10, 0.35],
            False,
            5,
            None,
        ),
    )
    outcomes = []
    for trajectory, times, success, executed, error in specifications:
        outcomes.append(
            {
                "trajectory": trajectory,
                "sim_times": np.asarray(times, dtype=np.float64),
                "root_step": 0,
                "post_step": min(1, len(trajectory) - 1),
                "first_executed": executed,
                "success": success,
                "branch_error": error,
                "terminal_stop_reason_id": (
                    0 if success else 2 if error is not None else 1
                ),
            }
        )
    return root, outcomes


def _calibration() -> dict:
    return {
        "moving": "can",
        "anchor": "pot",
        "required_objects": list(analytic_event.REQUIRED_OBJECTS),
        "goal_rule": analytic_event.GOAL_RULE,
        "success_height_reference_rule": (
            analytic_event.SUCCESS_HEIGHT_REFERENCE_RULE
        ),
        "thresholds": analytic_event.THRESHOLDS,
        "event_rules": analytic_event.EVENT_RULES,
    }


def test_materialization_keeps_terminal_endpoints_and_se3_object_effect() -> None:
    root, outcomes = _root_and_outcomes()
    arrays = collector.materialize_group(
        root=root,
        outcomes=outcomes,
        calibration=_calibration(),
        action_exec_steps=5,
    )
    assert arrays["terminal_max_event_id"].tolist() == [1, 4, 0, 3]
    np.testing.assert_allclose(
        arrays["terminal_stage_progress"], [0.25, 1.0, 0.0, 0.75]
    )
    np.testing.assert_allclose(
        arrays["terminal_goal_distance"], [0.10, 0.0, 0.12, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["terminal_goal_progress"], [0.02, 0.12, 0.0, 0.12], atol=1e-6
    )
    np.testing.assert_allclose(arrays["object_delta"][1, :3], [-0.12, 0.0, 0.0])
    np.testing.assert_allclose(
        arrays["object_delta"][1, 3:], [np.pi / 2.0, 0.0, 0.0], atol=1e-5
    )
    np.testing.assert_array_equal(arrays["event_age_seconds"], np.zeros(4))
    np.testing.assert_array_equal(
        arrays["remaining_action_budget"], np.full(4, 200.0)
    )
    assert arrays["terminal_stop_reason_id"].tolist() == [1, 0, 1, 1]
    assert collector.OBJECT_EFFECT_SCHEMA == watcher.OBJECT_EFFECT_SCHEMA
    assert collector.TERMINAL_SUPERVISION_CONTRACT == watcher.TERMINAL_SUPERVISION_CONTRACT
    assert collector.EVENT_AGE_CONTRACT == watcher.EVENT_AGE_CONTRACT
    assert collector.TERMINAL_HORIZON_CONTRACT == watcher.TERMINAL_HORIZON_CONTRACT
    assert (
        collector.BRANCH_ROOT_SNAPSHOT_CONTRACT
        == watcher.BRANCH_ROOT_SNAPSHOT_CONTRACT
    )


def test_event_age_uses_physical_time_since_latest_event_entry() -> None:
    events = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    times = np.asarray([0.0, 0.1, 0.3, 0.5, 0.8, 1.0], dtype=np.float64)
    assert collector.event_age_seconds(events, times, step=4) == pytest.approx(0.5)
    assert collector.event_age_seconds(events, times) == pytest.approx(0.0)


def test_restore_hash_excludes_only_derived_qacc() -> None:
    base = {
        "articulations": {
            "robot": {
                "qpos": [0.1, 0.2],
                "qvel": [0.3, 0.4],
                "qacc": [5.0, -7.0],
                "qf": [0.5, 0.6],
            }
        },
        "simulation_step_count": 31,
    }
    changed_qacc = {
        **base,
        "articulations": {
            "robot": {**base["articulations"]["robot"], "qacc": [0.0, 0.0]}
        },
    }
    assert collector.branch_root_snapshot_sha256(base) != (
        collector.branch_root_snapshot_sha256(changed_qacc)
    )
    assert collector.branch_root_restorable_snapshot_sha256(base) == (
        collector.branch_root_restorable_snapshot_sha256(changed_qacc)
    )
    changed_qvel = {
        **base,
        "articulations": {
            "robot": {**base["articulations"]["robot"], "qvel": [0.3, 0.5]}
        },
    }
    assert collector.branch_root_restorable_snapshot_sha256(base) != (
        collector.branch_root_restorable_snapshot_sha256(changed_qvel)
    )


def _restore_snapshot(root_pose: list[float]) -> dict:
    return {
        "format": "test",
        "articulations": {
            "robot": {
                "root_pose": root_pose,
                "qpos": [0.1, 0.2],
                "qvel": [0.3, 0.4],
                "qacc": [5.0, -7.0],
                "qf": [0.5, 0.6],
            }
        },
        "simulation_step_count": 31,
    }


def test_restore_equivalence_only_tolerates_root_pose_float32_roundtrip() -> None:
    base_pose = [0.0, 0.0, 0.75, 0.0, 0.0, -0.05, -0.9982587695121765]
    base = _restore_snapshot(base_pose)
    arx_roundtrip = _restore_snapshot(
        [*base_pose[:6], -0.9982588887214661]
    )
    assert collector.branch_root_restorable_snapshot_sha256(base) != (
        collector.branch_root_restorable_snapshot_sha256(arx_roundtrip)
    )
    assert collector.branch_root_restorable_snapshots_equal(base, arx_roundtrip)

    piper_pose = [0.0, 0.0, 0.75, 1.0, 0.0, -2.60770320892334e-8, 0.0]
    piper_roundtrip = _restore_snapshot(
        [*piper_pose[:5], -3.3527612686157227e-8, piper_pose[6]]
    )
    assert collector.branch_root_restorable_snapshots_equal(
        _restore_snapshot(piper_pose), piper_roundtrip
    )

    material_pose_change = _restore_snapshot(
        [*base_pose[:6], base_pose[6] + 1e-6]
    )
    assert not collector.branch_root_restorable_snapshots_equal(
        base, material_pose_change
    )

    changed_qpos = _restore_snapshot(base_pose)
    changed_qpos["articulations"]["robot"]["qpos"][1] += 1e-12
    assert not collector.branch_root_restorable_snapshots_equal(base, changed_qpos)


def test_restore_equivalence_rejects_invalid_or_incomplete_root_pose() -> None:
    base = _restore_snapshot([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])
    missing_articulation = _restore_snapshot([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])
    missing_articulation["articulations"] = {}
    assert not collector.branch_root_restorable_snapshots_equal(
        base, missing_articulation
    )

    invalid_shape = _restore_snapshot([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])
    invalid_shape["articulations"]["robot"]["root_pose"] = [0.0] * 6
    assert not collector.branch_root_restorable_snapshots_equal(base, invalid_shape)

    non_finite = _restore_snapshot([0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0])
    non_finite["articulations"]["robot"]["root_pose"][6] = float("nan")
    assert not collector.branch_root_restorable_snapshots_equal(base, non_finite)

    assert collector.BRANCH_ROOT_SNAPSHOT_CONTRACT == (
        watcher.BRANCH_ROOT_SNAPSHOT_CONTRACT
    )
    assert collector.BRANCH_ROOT_SNAPSHOT_CONTRACT[
        "all_non_root_pose_restorable_fields_bit_exact"
    ] is True
    assert collector.BRANCH_ROOT_SNAPSHOT_CONTRACT[
        "post_canonicalization_full_snapshot_bit_exact"
    ] is True


def test_materialization_rejects_action_execution_exceptions() -> None:
    root, outcomes = _root_and_outcomes()
    outcomes[2] = {
        **outcomes[2],
        "branch_error": "RuntimeError: simulator failure",
        "terminal_stop_reason_id": 2,
    }
    with pytest.raises(collector.BranchCollectionError):
        collector.materialize_group(
            root=root,
            outcomes=outcomes,
            calibration=_calibration(),
            action_exec_steps=5,
        )


def test_branch_diagnostics_capture_action_coverage_without_runtime_failures() -> None:
    root, outcomes = _root_and_outcomes()
    arrays = collector.materialize_branch_diagnostics(
        root=root, outcomes=outcomes, action_exec_steps=5
    )
    assert arrays["first_executed"].tolist() == [5, 5, 0, 5]
    assert arrays["branch_error"].tolist() == [False, False, False, False]
    distances = arrays["candidate_action_pairwise_rms"]
    assert distances.shape == (4, 4)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), 0.0)
    assert np.all(distances[np.triu_indices(4, 1)] > 0.0)
    np.testing.assert_allclose(
        arrays["candidate_first_token_translation_norm_m"],
        [[0.002, 0.001], [0.004, 0.002], [0.006, 0.003], [0.008, 0.004]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        arrays["candidate_later_token_translation_norm_median_m"],
        [[0.002, 0.001], [0.004, 0.002], [0.006, 0.003], [0.008, 0.004]],
        atol=1e-7,
    )
    assert np.all(arrays["candidate_first_token_translation_norm_m"] < 0.01)
    assert collector.BRANCH_DIAGNOSTIC_CONTRACT == watcher.BRANCH_DIAGNOSTIC_CONTRACT
    assert watcher.ROOT_QUERIES == tuple(range(40))
