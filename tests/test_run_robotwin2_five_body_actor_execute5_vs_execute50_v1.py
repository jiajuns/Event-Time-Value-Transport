from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "actor_execute5_vs_execute50_runner",
    SCRIPTS / "run_robotwin2_five_body_actor_execute5_vs_execute50_v1.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _ee16() -> np.ndarray:
    value = np.zeros(16, dtype=np.float32)
    value[3] = 1.0
    value[11] = 1.0
    return value


def _chunk(query_index: int = 0) -> np.ndarray:
    result = np.repeat(_ee16()[None, None], 50, axis=1)
    result[0, :, 0] = 0.001 * query_index
    result[0, :, 8] = -0.001 * query_index
    return result.astype(np.float32)


def _snap(step: int) -> dict[str, object]:
    return {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "synthetic_physical_step_count": step,
    }


def _commitment(expected: dict[str, object]) -> dict[str, object]:
    candidates = _chunk(0)
    reset = _snap(0)
    canonical = _snap(1)
    base = {
        "format": runner.COMMITMENT_FORMAT,
        "heldout_body": expected["heldout_body"],
        "condition": expected["condition"],
        "requested_seed": expected["requested_seed"],
        "resolved_seed": expected["requested_seed"],
        "frozen_before_any_method_execution": True,
        "candidate_count": 1,
        "candidate_index": 0,
        "candidate_shape": [1, 50, 16],
        "candidate0_chunk_sha256": runner.array_sha256(candidates[0]),
        "candidate0_first_token_sha256": runner.array_sha256(candidates[0, 0]),
        "reset_snapshot": reset,
        "reset_identity_sha256": runner.formal.reset_identity(reset),
        "canonical_query_snapshot": canonical,
        "canonical_query_identity_sha256": runner.formal.reset_identity(canonical),
        "query_canonicalization_steps": runner.QUERY_CANONICALIZATION_STEPS,
        "candidate_generation_advanced_simulator": False,
        "actor_candidate0_native_ee16": True,
    }
    return {**base, "commitment_sha256": runner.canonical_sha256(base)}


def _terminal_audit(success: bool) -> dict[str, object]:
    return {
        "official_terminal_check_success": success,
        "recomputed_terminal_check_success": success,
    }


def _minimal_rollout(
    method: str,
    expected: dict[str, object],
    commitment: dict[str, object],
) -> dict[str, object]:
    chunk = _chunk(0)
    continuity = runner.first_token_continuity(
        live_ee16=_ee16(),
        first_token_ee16=chunk[0, 0],
        previous_executed_token_ee16=None,
        previous_native_next_token_ee16=None,
    )
    decision = {
        "query_index": 0,
        "candidate_count": 1,
        "candidate_index": 0,
        "candidate0_chunk_sha256": runner.array_sha256(chunk[0]),
        "candidate0_first_token_sha256": runner.array_sha256(chunk[0, 0]),
        "native_chunk_steps": 50,
        "protocol_execute_steps": runner.EXECUTION_STEPS[method],
        "query_canonicalization_physical_steps_before_generation": 1,
        "executed_action_count": 1,
        "first_token_continuity": continuity,
        "physical_sim_seconds": 0.01,
        "critic_scores": None,
    }
    return {
        "method": method,
        "heldout_body": expected["heldout_body"],
        "condition": expected["condition"],
        "requested_seed": expected["requested_seed"],
        "resolved_seed": expected["requested_seed"],
        "execution_stride_actions": runner.EXECUTION_STEPS[method],
        "native_chunk_steps": 50,
        "candidate_count": 1,
        "candidate_index": 0,
        "initial_reset_identity_sha256": commitment["reset_identity_sha256"],
        "initial_reset_snapshot": commitment["reset_snapshot"],
        "initial_canonical_query_snapshot": commitment["canonical_query_snapshot"],
        "initial_candidate_commitment_sha256": commitment["commitment_sha256"],
        "initial_candidate0_chunk_sha256": commitment["candidate0_chunk_sha256"],
        "initial_candidate0_first_token_sha256": commitment[
            "candidate0_first_token_sha256"
        ],
        "tracked_object_names": ["can", "pot"],
        "initial_object_poses": [],
        "initial_ee16": _ee16().tolist(),
        "binary_success": 1,
        "latched_eval_success": True,
        "terminal_check_success": True,
        "terminal_native_success_components": _terminal_audit(True),
        "stop_reason": "success",
        "stage_progress": 1.0,
        "max_event_id": 4,
        "initial_goal_distance_m": 1.0,
        "terminal_goal_distance_m": 0.5,
        "goal_progress_m": 0.5,
        "executed_control_steps": 1,
        "physical_sim_seconds": 0.02,
        "sim_timestep_seconds": 0.01,
        "policy_query_count": 1,
        "first_token_continuity_summary": runner.summarize_first_token_continuity(
            [decision]
        ),
        "action_execution_error": None,
        "decisions": [decision],
    }


def test_frozen_roster_is_200_pairs_400_rollouts_and_counterbalanced() -> None:
    schedule = runner.evaluation_schedule()
    assert len(schedule) == 200
    assert Counter(
        (item["heldout_body"], item["condition"]) for item in schedule
    ) == {
        (body, condition): 20
        for body in runner.BODIES
        for condition in runner.CONDITIONS
    }
    assert Counter(tuple(item["method_order"]) for item in schedule) == {
        runner.METHODS: 100,
        tuple(reversed(runner.METHODS)): 100,
    }
    protocol = runner.evaluation_protocol()
    assert protocol["pair_count"] == 200
    assert protocol["rollout_count"] == 400
    assert protocol["candidate_count"] == 1
    assert protocol["candidate_index"] == 0
    assert protocol["critic_loaded_or_called"] is False
    assert protocol["training_performed"] is False
    canonicalization = protocol["query_canonicalization"]
    assert canonicalization["raw_scene_steps_before_every_actor_query"] == 1
    assert canonicalization["formal_actor_action_count_advanced"] is False
    assert canonicalization["physical_time_and_event_age_advanced"] is True
    assert (
        canonicalization[
            "occurs_once_per_policy_query_and_therefore_frequency_is_part_of_"
            "the_execute5_vs_execute50_deployment_protocol"
        ]
        is True
    )
    assert runner.analytic_event.__name__.endswith("analytic_event_spec_v2")
    assert runner.EVENT_SPEC_SHA256 == runner.analytic_event.EVENT_SPEC_SHA256


def test_first_token_continuity_is_quaternion_sign_invariant_effect_frame() -> None:
    reference = _ee16()
    positive = _ee16()
    negative = _ee16()
    negative[3:7] *= -1.0
    negative[11:15] *= -1.0
    positive_metrics = runner.first_token_continuity(
        live_ee16=reference,
        first_token_ee16=positive,
        previous_executed_token_ee16=None,
        previous_native_next_token_ee16=None,
    )
    negative_metrics = runner.first_token_continuity(
        live_ee16=reference,
        first_token_ee16=negative,
        previous_executed_token_ee16=None,
        previous_native_next_token_ee16=None,
    )
    assert positive_metrics["live_state_to_first_token"]["effect14_rms"] == 0.0
    assert negative_metrics["live_state_to_first_token"]["effect14_rms"] == 0.0
    assert negative_metrics["raw_quaternion_component_rms_used"] is False


class _FakeScene:
    timestep_seconds = 0.01

    def __init__(self) -> None:
        self.step_count = 0
        self.sim_seconds = 0.0

    def step(self) -> None:
        self.step_count += 1
        self.sim_seconds += self.timestep_seconds


class _FakeTask:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.take_action_cnt = 0
        self.eval_success = False
        self.orig_z = 0.5
        self.closed = False

    def take_action(self, _action: np.ndarray, *, action_type: str) -> None:
        assert action_type == "ee"
        self.take_action_cnt += 1
        self.scene.step()

    def close_env(self, *, clear_cache: bool) -> None:
        assert clear_cache is False
        self.closed = True


def _install_fake_rollout_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.collector, "_new_task", lambda *_args: _FakeTask())
    monkeypatch.setattr(
        runner.collector,
        "discover_pose_objects",
        lambda _task, _required: (["can", "pot"], [object(), object()]),
    )

    def read_poses(_objects: object) -> np.ndarray:
        result = np.zeros((2, 7), dtype=np.float32)
        result[:, 3] = 1.0
        return result

    monkeypatch.setattr(runner.collector, "read_poses", read_poses)
    monkeypatch.setattr(runner.collector, "current_ee_action16", lambda _task: _ee16())
    monkeypatch.setattr(
        runner.collector, "_sim_time", lambda task: task.scene.sim_seconds
    )

    def append(task: _FakeTask, objects: object, trajectory: list, times: list) -> None:
        now = task.scene.sim_seconds
        assert now > times[-1]
        trajectory.append(read_poses(objects))
        times.append(now)

    monkeypatch.setattr(runner.collector, "_append_physical_observation", append)
    monkeypatch.setattr(
        runner.formal,
        "capture_reset_snapshot",
        lambda task, _names, _objects: _snap(task.scene.step_count),
    )
    monkeypatch.setattr(
        runner,
        "generate_actor_candidate0",
        lambda **kwargs: _chunk(int(kwargs["query_index"])),
    )
    monkeypatch.setattr(
        runner,
        "native_success_components",
        lambda _task: _terminal_audit(False),
    )
    monkeypatch.setattr(
        runner.analytic_event,
        "derive_predicates_and_events",
        lambda trajectory, _times, _names, _success, _calibration, _orig_z: (
            np.zeros((len(trajectory), 1), dtype=np.bool_),
            np.zeros(len(trajectory), dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        runner.analytic_event,
        "goal_vector",
        lambda trajectory, _names, step, _calibration: (
            "can",
            np.asarray(
                [max(0.5, 1.0 - 0.001 * int(step)), 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
    )


def test_execute5_and_execute50_use_native_candidate0_and_exhaust_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rollout_runtime(monkeypatch)
    expected = runner.evaluation_schedule()[0]
    commitment = _commitment(expected)
    common = {
        "body": expected["heldout_body"],
        "condition": expected["condition"],
        "seed": expected["requested_seed"],
        "task_class": object,
        "task_args": {},
        "policy": object(),
        "preprocessor": object(),
        "postprocessor": object(),
        "calibration": {},
        "initial_commitment": commitment,
        "instruction": runner.collector.DEFAULT_INSTRUCTION,
        "max_steps": 200,
        "device": torch.device("cpu"),
    }
    execute5 = runner.execute_rollout(method=runner.METHOD_EXECUTE5, **common)
    execute50 = runner.execute_rollout(method=runner.METHOD_EXECUTE50, **common)
    runner.validate_rollout(execute5, method=runner.METHOD_EXECUTE5, expected=expected)
    runner.validate_rollout(execute50, method=runner.METHOD_EXECUTE50, expected=expected)
    assert execute5["executed_control_steps"] == 200
    assert execute50["executed_control_steps"] == 200
    assert execute5["policy_query_count"] == 40
    assert execute50["policy_query_count"] == 4
    assert {item["candidate_index"] for item in execute5["decisions"]} == {0}
    assert {item["critic_scores"] for item in execute50["decisions"]} == {None}
    assert (
        execute5["initial_candidate0_chunk_sha256"]
        == execute50["initial_candidate0_chunk_sha256"]
        == commitment["candidate0_chunk_sha256"]
    )
    assert execute5["first_token_continuity_summary"][
        "previous_native_next_to_replanned_first_token"
    ]["count"] == 39
    assert execute50["first_token_continuity_summary"][
        "previous_native_next_to_replanned_first_token"
    ]["count"] == 0


def _persist_pair_chain(
    output: Path,
) -> tuple[dict[str, object], str, dict[str, object], dict[str, object]]:
    expected = runner.evaluation_schedule()[0]
    identity = runner.pair_id(
        expected["heldout_body"], expected["condition"], expected["requested_seed"]
    )
    binding_logical = "b" * 64
    binding_file = "c" * 64
    attempt = runner.build_attempt(
        expected,
        binding_logical_sha256=binding_logical,
        binding_file_sha256=binding_file,
    )
    commitment = _commitment(expected)
    paths = runner._artifact_paths(output, identity, expected)
    runner.promote_create_once_json(paths["attempt"], attempt, label="attempt")
    runner.promote_create_once_json(
        paths["commitment"], commitment, label="commitment"
    )
    rollouts = {}
    bindings = {}
    prefix = []
    for ordinal, method in enumerate(expected["method_order"]):
        start = runner.build_method_start(
            expected,
            method=method,
            method_ordinal=ordinal,
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            binding_logical_sha256=binding_logical,
            completed_prefix_result_sha256=prefix,
        )
        runner.promote_create_once_json(
            paths["starts"][method], start, label=f"{method} start"
        )
        rollout = _minimal_rollout(method, expected, commitment)
        result = runner.build_method_result(
            expected,
            method=method,
            method_ordinal=ordinal,
            rollout=rollout,
            method_start_sha256=start["method_start_sha256"],
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            binding_logical_sha256=binding_logical,
            binding_file_sha256=binding_file,
            completed_prefix_result_sha256=prefix,
        )
        file_sha = runner.promote_create_once_json(
            paths["results"][method], result, label=f"{method} result"
        )
        rollouts[method] = rollout
        bindings[method] = {
            "logical_sha256": result["method_result_sha256"],
            "file_sha256": file_sha,
        }
        prefix.append(result["method_result_sha256"])
    pair = runner.materialize_pair(
        expected,
        rollouts,
        attempt_sha256=attempt["attempt_sha256"],
        commitment=commitment,
        binding_logical_sha256=binding_logical,
        binding_file_sha256=binding_file,
        method_result_bindings=bindings,
    )
    runner.promote_create_once_json(paths["pair"], pair, label="pair")
    recovered = runner.load_complete_pair_chain(
        output=output,
        identity=identity,
        expected=expected,
        attempt=attempt,
        binding_logical_sha256=binding_logical,
        binding_file_sha256=binding_file,
    )
    assert recovered == pair
    return expected, identity, attempt, paths


def test_recovery_rejects_self_consistent_pair_outcome_rewrite(tmp_path: Path) -> None:
    expected, identity, attempt, paths = _persist_pair_chain(tmp_path)
    pair_path = paths["pair"]
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    method = runner.METHOD_EXECUTE5
    pair["rollouts"][method]["goal_progress_m"] = 0.4
    pair["rollouts"][method]["terminal_goal_distance_m"] = 0.6
    pair["pair_sha256"] = runner.canonical_sha256(
        {key: value for key, value in pair.items() if key != "pair_sha256"}
    )
    pair_path.chmod(0o644)
    pair_path.write_text(json.dumps(pair, sort_keys=True) + "\n", encoding="utf-8")
    pair_path.chmod(0o444)
    with pytest.raises(runner.ActorDeploymentProtocolError):
        runner.load_complete_pair_chain(
            output=tmp_path,
            identity=identity,
            expected=expected,
            attempt=attempt,
            binding_logical_sha256="b" * 64,
            binding_file_sha256="c" * 64,
        )


def test_existing_pair_missing_method_result_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected, identity, attempt, paths = _persist_pair_chain(tmp_path)
    missing_method = expected["method_order"][0]
    paths["results"][missing_method].unlink()
    called = False

    def forbidden_execute(**_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("rollout must not be called")

    monkeypatch.setattr(runner, "execute_rollout", forbidden_execute)
    with pytest.raises(runner.ActorDeploymentProtocolError):
        runner.load_complete_pair_chain(
            output=tmp_path,
            identity=identity,
            expected=expected,
            attempt=attempt,
            binding_logical_sha256="b" * 64,
            binding_file_sha256="c" * 64,
        )
    assert called is False


def _outcome_rows(*, success_delta: bool, stage_delta: bool) -> list[dict[str, object]]:
    rows = []
    for index, expected in enumerate(runner.evaluation_schedule()):
        row = {
            **expected,
            "benchmark": runner.BENCHMARK,
            "task": runner.TASK,
            "pair_sha256": "a" * 64,
        }
        for method in runner.METHODS:
            success = int(
                success_delta and index == 0 and method == runner.METHOD_EXECUTE50
            )
            stage = 0.25 if (
                stage_delta and index == 0 and method == runner.METHOD_EXECUTE50
            ) else 0.0
            goal_progress = 0.1 if method == runner.METHOD_EXECUTE50 else 0.0
            row[f"{method}_binary_success"] = success
            row[f"{method}_stage_progress"] = stage
            row[f"{method}_terminal_goal_distance_m"] = 1.0 - goal_progress
            row[f"{method}_goal_progress_m"] = goal_progress
            row[f"{method}_live_first_token_effect14_rms_mean"] = 0.01
            row[f"{method}_command_boundary_effect14_rms_mean"] = None
        rows.append(row)
    return rows


@pytest.mark.parametrize(
    ("success_delta", "stage_delta", "selected"),
    (
        (True, True, "binary_success"),
        (False, True, "stage_progress"),
        (False, False, "goal_progress_m"),
    ),
)
def test_report_uses_preregistered_hierarchical_selection_and_cluster_ci(
    monkeypatch: pytest.MonkeyPatch,
    success_delta: bool,
    stage_delta: bool,
    selected: str,
) -> None:
    monkeypatch.setattr(runner, "BOOTSTRAP_REPLICATES", 200)
    report = runner.build_report(
        _outcome_rows(success_delta=success_delta, stage_delta=stage_delta),
        outcome_document_sha256="d" * 64,
    )
    selection = report["primary_hierarchical_selection"]
    assert selection["selected_criterion"] == selected
    assert selection[
        "mcnemar_and_bootstrap_intervals_are_uncertainty_not_sole_gate"
    ] is True
    assert report["overall"]["confidence_interval_contract"]["cluster_count"] == 20
    assert report["overall"]["mcnemar_role"] == "descriptive_only"
    for cell in report["by_body_condition"].values():
        assert cell["mcnemar_role"] == "inferential"
        assert cell["confidence_interval_contract"]["rows_per_cluster"] == 1


def test_native_success_components_matches_public_checker_terms() -> None:
    half = float(np.sqrt(0.5))

    class Pose:
        def __init__(self, p: list[float], q: list[float]) -> None:
            self.p = p
            self.q = q

    task = SimpleNamespace(
        pot=SimpleNamespace(get_pose=lambda: Pose([0.0, 0.0, 0.5], [1, 0, 0, 0])),
        can=SimpleNamespace(
            get_pose=lambda: Pose([-0.1, 0.01, 0.5], [half, half, 0, 0])
        ),
        arm_tag="left",
        orig_z=0.5,
        robot=SimpleNamespace(
            is_left_gripper_open=lambda: True,
            is_right_gripper_open=lambda: True,
        ),
        check_success=lambda: True,
    )
    audit = runner.native_success_components(task)
    assert audit["recomputed_terminal_check_success"] is True
    assert audit["official_terminal_check_success"] is True
    assert all(audit["checks"].values())


def test_terminal_success_audit_uses_exact_public_transforms3d_boundary() -> None:
    import transforms3d as t3d

    quaternion = t3d.euler.euler2quat(
        math.radians(75.0), math.radians(15.0), 0.0
    )
    official_roll, official_pitch, _ = t3d.euler.quat2euler(quaternion)
    observed_roll, observed_pitch = runner._quaternion_roll_pitch_degrees(
        quaternion
    )
    assert observed_roll == math.degrees(official_roll)
    assert observed_pitch == math.degrees(official_pitch)
