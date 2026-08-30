from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "paired_runner", SCRIPTS / "run_robotwin2_five_body_paired_success_v1.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _snapshot() -> dict:
    return {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "tracked_objects": [{"name": "can", "pose_xyz_wxyz": [0.0] * 7}],
        "robot_state": {"synthetic": True},
        "simulator_clock": {
            "physical_step_count": 0,
            "sim_seconds": 0.0,
            "timestep_seconds": 0.01,
        },
        "task_counters": {"take_action_count": 0, "eval_success": False},
    }


def _commitment() -> dict:
    snapshot = _snapshot()
    canonical_snapshot = copy.deepcopy(snapshot)
    canonical_snapshot["simulator_clock"]["physical_step_count"] = 1
    canonical_snapshot["simulator_clock"]["sim_seconds"] = 0.01
    base = {
        "format": "etsf_robotwin2_initial_candidate_commitment_v2",
        "heldout_body": "piper",
        "condition": "clean",
        "requested_seed": runner.SEED_BASE,
        "resolved_seed": runner.SEED_BASE,
        "action_exec_steps": runner.ACTION_EXEC_STEPS,
        "planned_dt_seconds": runner.PLANNED_DT_SECONDS,
        "candidate_count": 4,
        "candidate_horizon": 5,
        "candidate_shape": [4, 5, 16],
        "ordered_candidate_set_sha256": "b" * 64,
        "reset_snapshot": snapshot,
        "reset_identity_sha256": runner.reset_identity(snapshot),
        "canonical_query_snapshot": canonical_snapshot,
        "canonical_query_identity_sha256": runner.reset_identity(
            canonical_snapshot
        ),
        "query_canonicalization_steps": runner.QUERY_CANONICALIZATION_STEPS,
        "candidate_generation_advanced_simulator": False,
    }
    return {**base, "commitment_sha256": runner.canonical_sha256(base)}


def _rollout(method: str, success: int, progress: float) -> dict:
    commitment = _commitment()
    raw_rank = np.asarray([[0.0, 0.0, 1.0, 0.0]] * 5, dtype=np.float64)
    risk_adjusted = runner.shared_head.aggregate_risk_adjusted_rank_scores(
        torch.as_tensor(raw_rank)
    ).numpy()
    critic_scores = {
        "selected_candidate_index": 2,
        "candidate_rank_score_epistemic_lcb_ensemble": risk_adjusted.tolist(),
        "candidate_rank_score_mean": raw_rank.mean(axis=0).tolist(),
        "candidate_rank_score_raw_candidate_population_std": raw_rank.std(
            axis=0, ddof=0
        ).tolist(),
        "candidate_rank_score_raw_member_candidate_mean": raw_rank.mean(axis=1).tolist(),
        "candidate_rank_score_raw_member_candidate_population_std": raw_rank.std(
            axis=1, ddof=0
        ).tolist(),
        "candidate_rank_score_members": raw_rank.tolist(),
    }
    return {
        "method": method,
        "heldout_body": "piper",
        "condition": "clean",
        "requested_seed": runner.SEED_BASE,
        "resolved_seed": runner.SEED_BASE,
        "initial_reset_identity_sha256": commitment["reset_identity_sha256"],
        "initial_reset_snapshot": commitment["reset_snapshot"],
        "initial_canonical_query_snapshot": commitment[
            "canonical_query_snapshot"
        ],
        "initial_candidate_commitment_sha256": commitment["commitment_sha256"],
        "tracked_object_names": ["can"],
        "initial_object_poses": [[0.0] * 7],
        "initial_ee16": [0.0] * 16,
        "binary_success": success,
        "stage_progress": progress,
        "max_event_id": int(progress * 4),
        "executed_control_steps": 5,
        "policy_query_count": 1,
        "action_execution_error": None,
        "decisions": [
            {
                "query_index": 0,
                "candidate_set_sha256": "b" * 64,
                "candidate_count": 4,
                "selected_candidate_index": 0 if method == "actor_baseline" else 2,
                "executed_action_count": 5,
                "event_age_seconds": None if method == "actor_baseline" else 0.0,
                "critic_scores": (
                    None if method == "actor_baseline" else critic_scores
                ),
            }
        ],
    }


def test_schedule_is_exact_five_by_two_by_one_hundred() -> None:
    schedule = runner.evaluation_schedule()
    assert len(schedule) == 1000
    assert schedule[0] == {
        "heldout_body": "aloha-agilex",
        "condition": "clean",
        "requested_seed": 2026090000,
        "method_order": ["actor_baseline", "etsf_best_of_4"],
    }
    assert schedule[1]["method_order"] == ["etsf_best_of_4", "actor_baseline"]
    assert schedule[-1]["heldout_body"] == "ur5"
    assert schedule[-1]["condition"] == "randomized"
    assert schedule[-1]["requested_seed"] == 2026090099


def test_tie_break_is_lowest_candidate_index() -> None:
    assert runner.select_candidate([1.0, 2.0, 2.0, -1.0]) == 1
    assert runner.select_candidate([3.0, 3.0, 3.0, 3.0]) == 0
    with pytest.raises(runner.PairedExecutionError):
        runner.select_candidate([1.0, float("nan"), 0.0, 0.0])


def test_scoring_uses_epistemic_lcb_on_comparable_bounded_utilities() -> None:
    class FixedModel:
        def __init__(self, rank: list[float]):
            self.rank = torch.tensor(rank, dtype=torch.float32)

        def __call__(self, _batch):
            zeros = torch.zeros(4, dtype=torch.float32)
            event_logits = torch.zeros(4, 5, dtype=torch.float32)
            return {
                "candidate_rank_logit": self.rank,
                "success_logit": zeros,
                "post_event_logits": event_logits,
                "next_event_logits": event_logits,
                "duration_selected_log_mean": zeros,
                "duration_selected_log_scale": zeros,
                "terminal_event_logits": event_logits,
                "terminal_goal_progress_mean": zeros,
                "terminal_goal_progress_log_scale": zeros,
                "regression_probability": zeros,
                "joint_recovery_probability": zeros,
            }

    # Candidate 0 has the larger mean only because one member assigns utility
    # one while the other four assign zero.  Candidate 1 has a smaller but
    # unanimous bounded utility, so the frozen epistemic LCB prefers it.
    models = [FixedModel([1.0, 0.15, 0.0, 0.0])] + [
        FixedModel([0.0, 0.15, 0.0, 0.0]) for _ in range(4)
    ]
    scored = runner.score_candidates(models, {})
    assert int(np.argmax(scored["candidate_rank_score_mean"])) == 0
    assert scored["selected_candidate_index"] == 1
    assert int(
        np.argmax(scored["candidate_rank_score_epistemic_lcb_ensemble"])
    ) == 1
    assert "candidate_rank_score_standardized_ensemble" not in scored
    assert len(scored["candidate_rank_score_raw_member_candidate_population_std"]) == 5


def test_array_hash_binds_order_shape_and_values() -> None:
    values = np.arange(4 * 3 * 16, dtype=np.float32).reshape(4, 3, 16)
    assert runner.array_sha256(values) == runner.array_sha256(values.copy())
    assert runner.array_sha256(values) != runner.array_sha256(values[::-1])
    assert runner.array_sha256(values) != runner.array_sha256(values.reshape(4, 6, 8))


def test_stage_progress_uses_max_canonical_event() -> None:
    assert runner.stage_progress(np.array([0, 1, 3, 2]), False) == 0.75
    assert runner.stage_progress(np.array([0, 1]), True) == 1.0


def test_scoring_batch_uses_planned_first_five_tokens_at_actor_15hz() -> None:
    current = np.zeros(16, dtype=np.float32)
    current[3] = 1.0
    current[11] = 1.0
    candidates = np.repeat(current[None, None], 4 * 10, axis=0).reshape(4, 10, 16)
    state = np.zeros(27, dtype=np.float32)
    state[18] = 1.0
    batch = runner.scoring_batch(
        state=state,
        current_ee=current,
        candidates=candidates,
        current_event=0,
        event_age_seconds=0.4,
        remaining_action_budget=180,
        action_exec_steps=5,
        dt=1.0 / runner.ACTOR_DATASET_FPS,
        device=torch.device("cpu"),
    )
    assert batch["action_mask"].shape == (4, 10)
    assert batch["action_mask"][:, :5].all()
    assert not batch["action_mask"][:, 5:].any()
    assert torch.allclose(batch["dt"], torch.full((4,), 5.0 / 15.0))
    assert torch.allclose(batch["event_age_seconds"], torch.full((4,), 0.4))
    assert torch.allclose(
        batch["remaining_action_budget"], torch.full((4,), 180.0)
    )


def test_pair_records_discordance_and_requires_same_initial_candidates() -> None:
    expected = {
        "heldout_body": "piper",
        "condition": "clean",
        "requested_seed": runner.SEED_BASE,
        "method_order": ["actor_baseline", "etsf_best_of_4"],
    }
    rollouts = {
        "actor_baseline": _rollout("actor_baseline", 0, 0.5),
        "etsf_best_of_4": _rollout("etsf_best_of_4", 1, 1.0),
    }
    pair = runner.materialize_pair(
        expected,
        rollouts,
        attempt_sha256="d" * 64,
        commitment=_commitment(),
        execution_contract_logical_sha256="e" * 64,
    )
    assert pair["discordance"] == "etsf_only"
    assert pair["same_resolved_reset"] is True
    assert pair["same_complete_observable_reset_snapshot"] is True
    assert pair["same_canonical_query0_snapshot"] is True
    assert pair["same_initial_candidate_set"] is True
    runner.validate_pair_record(pair, expected)
    corrupted = copy.deepcopy(pair)
    corrupted["rollouts"]["etsf_best_of_4"]["decisions"][0]["critic_scores"][
        "candidate_rank_score_epistemic_lcb_ensemble"
    ][0] += 0.25
    corrupted["pair_sha256"] = runner.canonical_sha256(
        {key: value for key, value in corrupted.items() if key != "pair_sha256"}
    )
    with pytest.raises(runner.PairedExecutionError, match="cannot be replayed"):
        runner.validate_pair_record(corrupted, expected)
    corrupted = copy.deepcopy(pair)
    corrupted["rollouts"]["actor_baseline"]["decisions"][0][
        "selected_candidate_index"
    ] = 1
    corrupted["pair_sha256"] = runner.canonical_sha256(
        {key: value for key, value in corrupted.items() if key != "pair_sha256"}
    )
    with pytest.raises(runner.PairedExecutionError, match="candidate zero"):
        runner.validate_pair_record(corrupted, expected)
    rollouts["etsf_best_of_4"]["decisions"][0]["candidate_set_sha256"] = "c" * 64
    with pytest.raises(runner.PairedExecutionError):
        runner.materialize_pair(
            expected,
            rollouts,
            attempt_sha256="d" * 64,
            commitment=_commitment(),
            execution_contract_logical_sha256="e" * 64,
        )


def test_prepare_and_execute_signatures_bind_the_same_initial_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pose:
        p = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])

    class Object:
        def get_pose(self):
            return Pose()

    class Joint:
        name = "joint"
        drive_target = np.array([0.0])
        drive_velocity_target = np.array([0.0])

    class Entity(Object):
        def get_qpos(self):
            return np.zeros(2)

        def get_qvel(self):
            return np.zeros(2)

        def get_active_joints(self):
            return [Joint()]

    class Robot:
        left_entity = Entity()
        right_entity = Entity()

        def get_left_tcp_pose(self):
            return [0, 0, 0, 1, 0, 0, 0]

        get_right_tcp_pose = get_left_tcp_pose

        def get_left_gripper_val(self):
            return 0.0

        get_right_gripper_val = get_left_gripper_val

        def get_left_arm_jointState(self):
            return [0.0, 0.0]

        get_right_arm_jointState = get_left_arm_jointState

    class RawScene:
        def get_timestep(self):
            return 0.01

        def step(self):
            return None

    class Task:
        def __init__(self):
            self.robot = Robot()
            self.can = Object()
            self.pot = Object()
            self.scene = runner.collector.SimulationClockScene(RawScene())
            self.take_action_cnt = 0
            self.eval_success = False

        def take_action(self, _action, action_type):
            assert action_type == "ee"
            self.scene.step()
            self.take_action_cnt += 1

        def check_success(self):
            return False

        def close_env(self, clear_cache):
            assert clear_cache is False

    current = np.asarray(Robot().get_left_tcp_pose() + [0.0] + Robot().get_right_tcp_pose() + [0.0], dtype=np.float32)
    candidates = np.repeat(current[None, None], 4 * 5, axis=0).reshape(4, 5, 16)
    monkeypatch.setattr(runner.collector, "_new_task", lambda *_args, **_kwargs: Task())
    monkeypatch.setattr(runner.collector, "generate_candidates", lambda **_kwargs: candidates.copy())
    monkeypatch.setattr(
        runner.collector,
        "derive_predicates_and_events",
        lambda poses, sim_times, names, success, calibration: (
            np.zeros((len(poses), 5), dtype=np.float32),
            np.zeros(len(poses), dtype=np.int64),
        ),
    )
    calibration = {
        "moving": "can",
        "anchor": "pot",
        "required_objects": list(runner.analytic_event.REQUIRED_OBJECTS),
        "goal_rule": dict(runner.analytic_event.GOAL_RULE),
        "thresholds": dict(runner.analytic_event.THRESHOLDS),
        "event_rules": dict(runner.analytic_event.EVENT_RULES),
    }
    commitment = runner.prepare_initial_commitment(
        body="piper", condition="clean", seed=runner.SEED_BASE,
        task_class=Task, task_args={}, policy=None, preprocessor=None,
        postprocessor=None, calibration=calibration, instruction="task",
        device=torch.device("cpu"),
    )
    rollout = runner.execute_rollout(
        method="actor_baseline", body="piper", condition="clean",
        seed=runner.SEED_BASE, task_class=Task, task_args={}, policy=None,
        preprocessor=None, postprocessor=None, ensemble=[],
        calibration=calibration, initial_commitment=commitment,
        instruction="task", action_exec_steps=5, max_steps=1,
        dt=1.0 / 15.0, device=torch.device("cpu"),
    )
    assert rollout["initial_candidate_commitment_sha256"] == commitment["commitment_sha256"]
    assert rollout["decisions"][0]["executed_action_count"] == 1


def test_outcome_document_is_directly_accepted_by_existing_evaluator() -> None:
    rows = []
    for expected in runner.evaluation_schedule():
        rows.append(
            {
                "benchmark": runner.BENCHMARK,
                "task": runner.TASK,
                "heldout_body": expected["heldout_body"],
                "condition": expected["condition"],
                "requested_seed": expected["requested_seed"],
                "method_order": expected["method_order"],
                "pair_sha256": runner.canonical_sha256(expected),
                "actor_baseline_binary_success": 0,
                "actor_baseline_stage_progress": 0.0,
                "etsf_best_of_4_binary_success": 0,
                "etsf_best_of_4_stage_progress": 0.0,
            }
        )
    document = runner.build_outcome_document(
        rows,
        execution_contract_logical_sha256="e" * 64,
        execution_contract_file_sha256="f" * 64,
    )
    validated = runner.evaluator.validate_input_document(document)
    assert len(validated["rows"]) == 1000
    assert document["preregistration_sha256"] == runner.PREREGISTRATION_SHA256


def test_fold_specs_require_exactly_five_unique_bodies(tmp_path: Path) -> None:
    specs = []
    for body in runner.BODIES:
        path = tmp_path / body
        path.mkdir()
        specs.append(f"{body}={path}")
    parsed = runner.parse_fold_specs(specs)
    assert set(parsed) == set(runner.BODIES)
    with pytest.raises(runner.PairedExecutionError):
        runner.parse_fold_specs(specs[:-1])


def test_fold_training_regime_binds_one_exact_supplement(tmp_path: Path) -> None:
    supplement_sha = "a" * 64

    def folds(enabled_by_body: dict[str, bool]) -> dict[str, dict]:
        result = {}
        for body in runner.BODIES:
            summary_path = tmp_path / f"{body}-{int(enabled_by_body[body])}.json"
            if enabled_by_body[body]:
                supplement = {
                    "enabled": True,
                    "binding_file_sha256": supplement_sha,
                    "proper_loss_weight": (
                        runner.shared_head.SUPPLEMENT_PROPER_LOSS_WEIGHT
                    ),
                    "usage_contract": dict(
                        runner.shared_head.SUPPLEMENT_USAGE_CONTRACT
                    ),
                    "source_train_groups": 80,
                    "source_train_rows": 320,
                    "heldout_groups_deferred": 20,
                    "rank_or_utility_rows_used": 0,
                    "normalization_rows_used": 0,
                    "source_validation_rows_used": 0,
                    "checkpoint_selection_rows_used": 0,
                    "calibration_rows_used": 0,
                }
            else:
                supplement = {
                    "enabled": False,
                    "binding_file_sha256": None,
                    "source_train_groups": 0,
                    "source_train_rows": 0,
                    "heldout_groups_deferred": 0,
                }
            summary_path.write_text(
                json.dumps({"proper_world_supplement": supplement}),
                encoding="utf-8",
            )
            result[body] = {
                "training_summary": str(summary_path),
                "training_summary_sha256": runner.sha256_file(summary_path),
            }
        return result

    augmented = runner.inspect_fold_training_regime(
        folds({body: True for body in runner.BODIES}),
        required_supplement_binding_sha256=supplement_sha,
    )
    assert augmented["name"] == "c_plus_expert_root_supplement"
    assert augmented["supplement_binding_file_sha256"] == supplement_sha

    mixed = {body: True for body in runner.BODIES}
    mixed[runner.BODIES[-1]] = False
    with pytest.raises(runner.PairedExecutionError, match="mix"):
        runner.inspect_fold_training_regime(folds(mixed))
    with pytest.raises(runner.PairedExecutionError, match="required supplement"):
        runner.inspect_fold_training_regime(
            folds({body: False for body in runner.BODIES}),
            required_supplement_binding_sha256=supplement_sha,
        )


def test_analytic_events_and_state27_goal_are_identical_offline_and_online() -> None:
    spec_path = (
        ROOT / "configs/robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json"
    )
    _spec, calibration = runner.analytic_event.load_event_spec(spec_path)
    poses = np.zeros((6, 2, 7), dtype=np.float32)
    poses[:, 0, 0] = [-0.30, -0.28, -0.18, -0.18, -0.18, -0.18]
    poses[:, :, 2] = 0.74
    poses[:, :, 3] = 1.0
    times = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    predicates, events = runner.collector.derive_predicates_and_events(
        poses, times, ["can", "pot"], False, calibration
    )
    # The high arrival speed at t=0.2 is part of that closed stationary window,
    # so e4 begins only after a full low-speed [0.3, 0.5] interval.
    assert events.tolist() == [0, 1, 2, 2, 2, 3]
    _shifted_predicates, shifted_events = (
        runner.collector.derive_predicates_and_events(
            poses, times + 1.0, ["can", "pot"], False, calibration
        )
    )
    assert shifted_events.tolist() == events.tolist()
    _predicates, terminal = runner.collector.derive_predicates_and_events(
        poses, times, ["can", "pot"], True, calibration
    )
    assert terminal.tolist() == [0, 1, 2, 2, 2, 4]
    ee = np.zeros(16, dtype=np.float32)
    ee[3] = ee[11] = 1.0
    offline = runner.collector._state27(
        poses=poses, names=["can", "pot"], step=5,
        initial_moving_position=poses[0, 0, :3], ee_action=ee,
        event=int(events[-1]), predicates=predicates, calibration=calibration,
    )
    online, online_event, online_event_age = runner.canonical_state_at(
        trajectory=poses, sim_times=times, names=["can", "pot"],
        ee_action=ee, calibration=calibration,
    )
    assert online_event_age == pytest.approx(0.0)
    assert online_event == 3
    np.testing.assert_array_equal(online, offline)
