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
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event  # noqa: E402
import watch_robotwin2_ee16_actor_to_five_body_branches_v1 as watcher  # noqa: E402


class _NoiseConfig:
    chunk_size = 3
    max_action_dim = 4


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
        "candidates": np.stack([_candidate(index) for index in range(4)]),
    }
    quarter_turn_z = np.asarray(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
    )
    specifications = (
        (_trajectory([0.30, 0.28]), [0.0, 0.10], False, 5, None),
        (_trajectory([0.30, 0.18], quarter_turn_z), [0.0, 0.10], True, 5, None),
        (_trajectory([0.30]), [0.0], False, 0, None),
        (_trajectory([0.30, 0.18, 0.18, 0.18]), [0.0, 0.05, 0.10, 0.35], False, 5, None),
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
        "goal_rule": analytic_event.GOAL_RULE,
        "thresholds": analytic_event.THRESHOLDS,
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
        arrays["object_delta"][1, 3:], [0.0, 0.0, np.pi / 2.0], atol=1e-5
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
    assert collector.BRANCH_DIAGNOSTIC_CONTRACT == watcher.BRANCH_DIAGNOSTIC_CONTRACT
    assert watcher.ROOT_QUERIES == tuple(range(40))
