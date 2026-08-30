from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import collect_robotwin2_five_body_ee_candidate_branches_v1 as base  # noqa: E402
import collect_robotwin2_scripted_expert_root_actor_branches_v1 as collector  # noqa: E402
import materialize_robotwin2_scripted_expert_root_supplement_binding_v1 as materializer  # noqa: E402
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event  # noqa: E402
import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer  # noqa: E402


def _signed(value: dict) -> dict:
    result = dict(value)
    result["logical_sha256"] = trainer.canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return trainer.sha256_file(path)


class _NativeScene:
    def __init__(self, timestep: float = 0.05) -> None:
        self.calls = 0
        self.timestep = float(timestep)

    def get_timestep(self) -> float:
        return self.timestep

    def step(self) -> str:
        self.calls += 1
        return f"native-{self.calls}"


def test_expert_scene_proxy_advances_exactly_one_native_step() -> None:
    native = _NativeScene()
    prior = base.SimulationClockScene(native)
    callbacks: list[tuple[int, int]] = []
    scene = collector.ExpertObservationScene(
        prior,
        callback=lambda: callbacks.append((native.calls, scene.step_count)),
    )

    assert scene.step() == "native-1"
    assert native.calls == 1
    assert scene.step_count == 1
    assert callbacks == [(1, 1)]

    assert scene.step() == "native-2"
    assert native.calls == 2
    assert scene.step_count == 2
    assert callbacks == [(1, 1), (2, 2)]


class _PoseObject:
    def __init__(self, x: float) -> None:
        self.x = float(x)

    def get_pose(self) -> SimpleNamespace:
        return SimpleNamespace(
            p=np.asarray([self.x, 0.0, 0.75], dtype=np.float32),
            q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )


class _ObserverTask:
    def __init__(self) -> None:
        self.scene = base.SimulationClockScene(_NativeScene(timestep=0.1))

    def check_success(self) -> bool:
        return False


def _calibration() -> dict:
    return {
        "moving": "can",
        "anchor": "pot",
        "required_objects": list(analytic_event.REQUIRED_OBJECTS),
        "goal_rule": dict(analytic_event.GOAL_RULE),
        "thresholds": dict(analytic_event.THRESHOLDS),
        "event_rules": dict(analytic_event.EVENT_RULES),
    }


def _fake_snapshot(task: _ObserverTask) -> dict:
    return {
        "task_fields": {
            "take_action_cnt": 37,
            "eval_success": True,
            "plan_success": False,
            "step_lim": 999,
        },
        "simulation_step_count": task.scene.step_count,
        "physical": {"untouched": [1.0, 2.0]},
    }


def test_observer_freezes_only_first_e3_and_e4_and_resets_actor_horizon() -> None:
    task = _ObserverTask()
    can = _PoseObject(0.30)
    pot = _PoseObject(0.0)
    observer = collector.ScriptedRootObserver(
        task=task,
        names=["can", "pot"],
        objects=[can, pot],
        calibration=_calibration(),
        horizon=50,
        snapshot_fn=_fake_snapshot,
    )

    for step, x in enumerate((0.18, 0.18, 0.18), start=1):
        task.scene.step_count = step
        can.x = x
        observer.record_physics_step()
    assert set(observer.roots) == {"e3"}
    assert observer.roots["e3"]["detector_sim_step"] == 1

    task.scene.step_count = 4
    can.x = 0.18
    with pytest.raises(collector._AllRequestedRootsCaptured):
        observer.record_physics_step()
    assert set(observer.roots) == {"e3", "e4"}
    assert observer.roots["e4"]["detector_sim_step"] == 4
    for root in observer.roots.values():
        fields = root["branch_root_snapshot"]["task_fields"]
        assert fields == {
            "take_action_cnt": 0,
            "eval_success": False,
            "plan_success": True,
            "step_lim": 50,
        }
        assert root["branch_root_snapshot"]["physical"] == {
            "untouched": [1.0, 2.0]
        }
        assert root["remaining_action_budget"] == 50


def test_seed_horizon_binding_is_fixed_and_label_blind() -> None:
    expected = dict(zip(collector.PREDEFINED_SEEDS, (10, 25, 50, 100, 200)))
    assert collector.resolve_seed_horizons(collector.PREDEFINED_SEEDS) == expected
    with pytest.raises(collector.ScriptedRootCollectionError):
        collector.resolve_seed_horizons(collector.PREDEFINED_SEEDS[:-1])
    with pytest.raises(collector.ScriptedRootCollectionError):
        collector.resolve_seed_horizons(tuple(reversed(collector.PREDEFINED_SEEDS)))
    assert collector.HORIZON_CONTRACT[
        "candidate_or_terminal_outcomes_used_to_choose_horizon"
    ] is False


def _manifest_header(
    body: str, actor_authority_sha: str, actor_checkpoint_sha: str, event_sha: str
) -> dict:
    calibration = _calibration()
    return {
        "format": trainer.SUPPLEMENT_MANIFEST_FORMAT,
        "collector_format": collector.FORMAT,
        "dataset_repo": trainer.DATASET_REPO,
        "dataset_revision": trainer.DATASET_REVISION,
        "task": trainer.TASK,
        "instruction": trainer.DEFAULT_INSTRUCTION,
        "body": body,
        "conditions": list(trainer.CONDITIONS),
        "target_events": ["e3", "e4"],
        "supplement_role": "expert_event_root_proper_world_model_source_train_only",
        "root_policy": "robotwin_scripted_expert",
        "candidate_and_continuation_policy": (
            "same_frozen_native_actor_as_primary_binding"
        ),
        "candidate_count": trainer.CANDIDATE_COUNT,
        "proper_loss_weight": trainer.SUPPLEMENT_PROPER_LOSS_WEIGHT,
        "usage_contract": dict(trainer.SUPPLEMENT_USAGE_CONTRACT),
        "expert_root_provenance_contract": dict(
            trainer.EXPERT_ROOT_PROVENANCE_CONTRACT
        ),
        "actor_authority_sha256": actor_authority_sha,
        "actor_checkpoint_tree_or_file_sha256": actor_checkpoint_sha,
        "collector_file_sha256": "6" * 64,
        "base_collector_file_sha256": "7" * 64,
        "action_exec_steps": 5,
        "root_selection_contract": dict(collector.ROOT_SELECTION_CONTRACT),
        "horizon_contract": dict(collector.HORIZON_CONTRACT),
        "actor_branch_contract": dict(collector.ACTOR_BRANCH_CONTRACT),
        "event_spec_sha256": trainer.EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": event_sha,
        "analytic_event_contract": analytic_event.event_contract(calibration),
        "schema_adapter": {
            "kind": "analytic_label_free_canonical_v1",
            "trainable": False,
            "labels_or_outcomes_used_to_fit": False,
            "heldout_supervision_allowed": False,
            "state_dim": 27,
            "action_dim": 14,
            "state_schema": trainer.CANONICAL_STATE_SCHEMA,
            "action_schema": trainer.CANONICAL_ACTION_SCHEMA,
            "elapsed_time_unit": "seconds",
            "duration_unit": "seconds",
            "event_names": list(analytic_event.EVENT_NAMES),
            "implementation_sha256": "a" * 64,
        },
        "state27_relative_goal_contract": (
            "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
            "event_labels_and_online_state27_channels_0_2"
        ),
        "physical_time_contract": {
            "source": "counted_successful_sapien_scene_step_calls",
            "simulator_timestep_source": "scene.get_timestep",
            "policy_action_call_count_used_as_time": False,
            "wall_clock_used_as_time": False,
            "dt_semantics": "planned_first_candidate_chunk_seconds",
            "planned_action_steps": 5,
            "actor_control_hz": 15.0,
            "planned_dt_seconds": 5.0 / 15.0,
            "duration_semantics": "simulator_elapsed_seconds_to_event_boundary",
            "zero_elapsed_duration_masked": True,
            "stationary_window_seconds": 0.2,
            "stationary_speed_threshold_m_per_s": 0.01,
        },
        "candidate_action_contract": {
            "critic_observation_time": "before_candidate_execution",
            "planned_action_horizon": 5,
            "action_mask_source": "planned_first_chunk_not_executed_count",
            "executed_action_count_used_for_action_mask": False,
            "executed_action_count_used_for_sim_time_accounting_only": True,
            "planner_status_fail_is_a_valid_action_outcome": True,
            "python_execution_exception_invalidates_complete_decision": True,
        },
        "candidate_noise_contract": trainer.CANDIDATE_NOISE_CONTRACT,
        "terminal_supervision_contract": trainer.TERMINAL_SUPERVISION_CONTRACT,
        "event_age_contract": trainer.EVENT_AGE_CONTRACT,
        "terminal_horizon_contract": trainer.TERMINAL_HORIZON_CONTRACT,
        "branch_root_snapshot_contract": trainer.BRANCH_ROOT_SNAPSHOT_CONTRACT,
        "object_effect_schema": trainer.OBJECT_EFFECT_SCHEMA,
        "branch_diagnostic_contract": trainer.BRANCH_DIAGNOSTIC_CONTRACT,
        "pre_registered_seeds": list(collector.PREDEFINED_SEEDS),
        "pre_registered_horizon_by_seed": {
            str(seed): horizon
            for seed, horizon in collector.resolve_seed_horizons(
                collector.PREDEFINED_SEEDS
            ).items()
        },
    }


def _supplement_manifest(
    body: str, actor_authority_sha: str, actor_checkpoint_sha: str, event_sha: str
) -> dict:
    value = _manifest_header(
        body, actor_authority_sha, actor_checkpoint_sha, event_sha
    )
    groups = []
    horizons = collector.resolve_seed_horizons(collector.PREDEFINED_SEEDS)
    for condition in trainer.CONDITIONS:
        for seed in collector.PREDEFINED_SEEDS:
            for event in (2, 3):
                stem = f"{condition}-seed{seed}-e{event}"
                event_name = {2: "e3", 3: "e4"}[event]
                groups.append(
                    {
                        "group_id": (
                            f"{condition}|seed={seed}|scripted_root={event_name}"
                        ),
                        "condition": condition,
                        "requested_seed": seed,
                        "root_event_id": event,
                        "scripted_root_event_id": event,
                        "scripted_root_event": event_name,
                        "pre_registered_horizon": horizons[seed],
                        "candidate_noise_query_index": event,
                        "path": f"groups/{stem}.npz",
                        "sha256": "1" * 64,
                        "raw_expert_snapshot_sha256": "8" * 64,
                        "branch_root_snapshot_sha256": "2" * 64,
                        "branch_root_restorable_snapshot_sha256": "3" * 64,
                        "canonical_root_snapshot_sha256": "4" * 64,
                        "diagnostic_format": trainer.BRANCH_DIAGNOSTIC_CONTRACT[
                            "format"
                        ],
                        "diagnostics_path": f"groups/{stem}.diagnostics.npz",
                        "diagnostics_sha256": "5" * 64,
                    }
                )
    value["groups"] = groups
    return _signed(value)


def _binding_fixture(tmp_path: Path) -> tuple[Path, str, Path, str, dict[str, Path]]:
    checkpoint_sha = "c" * 64
    actor = _signed(
        {
            "format": trainer.ACTOR_FORMAT,
            "task": trainer.TASK,
            "actors": {
                body: {"checkpoint_sha256": checkpoint_sha}
                for body in trainer.BODIES
            },
        }
    )
    actor_path = tmp_path / "actor-authority.json"
    actor_sha = _write_json(actor_path, actor)
    primary_body_bindings = {}
    primary_manifests = {}
    for body in trainer.BODIES:
        primary_manifest = _signed(
            {
                "groups": [
                    {
                        "condition": condition,
                        "requested_seed": 2026082000,
                    }
                    for condition in trainer.CONDITIONS
                ]
            }
        )
        path = tmp_path / "primary" / body / "manifest.json"
        sha = _write_json(path, primary_manifest)
        primary_body_bindings[body] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha,
        }
        primary_manifests[body] = {
            "groups": primary_manifest["groups"]
        }
    primary = _signed(
        {
            "format": trainer.BINDING_FORMAT,
            "dataset_repo": trainer.DATASET_REPO,
            "dataset_revision": trainer.DATASET_REVISION,
            "task": trainer.TASK,
            "instruction": trainer.DEFAULT_INSTRUCTION,
            "actor_authority": {"path": actor_path.name, "sha256": actor_sha},
            "body_manifests": primary_body_bindings,
        }
    )
    primary_path = tmp_path / "primary-binding.json"
    primary_sha = _write_json(primary_path, primary)

    event_sha = "e" * 64
    supplements = {}
    for body in trainer.BODIES:
        path = tmp_path / "supplement" / body / "manifest.json"
        _write_json(
            path,
            _supplement_manifest(body, actor_sha, checkpoint_sha, event_sha),
        )
        supplements[body] = path
    return primary_path, primary_sha, actor_path, actor_sha, supplements


def test_five_body_materializer_emits_exact_trainer_binding_without_payload_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary_path, primary_sha, actor_path, actor_sha, supplements = (
        _binding_fixture(tmp_path)
    )
    output = tmp_path / "supplement" / "binding.json"
    monkeypatch.setattr(
        np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("binding materializer opened an NPZ payload")
        ),
    )
    monkeypatch.setattr(
        materializer.trainer,
        "load_binding",
        lambda *_args, **_kwargs: {
            "actor": {
                "checkpoint_sha256_by_body": {
                    body: "c" * 64 for body in trainer.BODIES
                }
            }
        },
    )
    binding = materializer.build_binding(
        primary_binding_path=primary_path,
        actor_authority_path=actor_path,
        body_manifest_paths=supplements,
        output_path=output,
    )
    binding_sha = _write_json(output, binding)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary_audit = {
        "binding": primary,
        "binding_file_sha256": primary_sha,
        "manifests": {
            body: {
                "groups": [
                    {
                        "condition": condition,
                        "requested_seed": 2026082000,
                    }
                    for condition in trainer.CONDITIONS
                ]
            }
            for body in trainer.BODIES
        },
        "event_derivation_implementation_sha256": "e" * 64,
        "actor": {
            "checkpoint_sha256_by_body": {
                body: "c" * 64 for body in trainer.BODIES
            }
        },
    }
    audit = trainer.load_supplement_binding(
        output,
        binding_sha,
        primary_audit=primary_audit,
        held_out_body="franka",
    )
    assert set(audit["manifests"]) == set(trainer.BODIES) - {"franka"}
    assert audit["heldout_manifest_binding"]["manifest_file_opened"] == 0
    assert audit["heldout_manifest_binding"]["payload_files_opened"] == 0
    assert binding["actor_authority_sha256"] == actor_sha
    assert binding["primary_binding_file_sha256"] == primary_sha
    assert binding["materializer_provenance"] == {
        "format": materializer.FORMAT,
        "payload_npz_files_opened": 0,
        "complete_decisions": 100,
        "complete_branches": 400,
        "seed_overlap_with_primary": 0,
    }


def test_materializer_rejects_incomplete_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary_path, _primary_sha, actor_path, _actor_sha, supplements = (
        _binding_fixture(tmp_path)
    )
    broken_path = supplements["piper"]
    broken = json.loads(broken_path.read_text(encoding="utf-8"))
    broken.pop("logical_sha256")
    broken["groups"].pop()
    _write_json(broken_path, _signed(broken))
    monkeypatch.setattr(
        materializer.trainer,
        "load_binding",
        lambda *_args, **_kwargs: {
            "actor": {
                "checkpoint_sha256_by_body": {
                    body: "c" * 64 for body in trainer.BODIES
                }
            }
        },
    )
    with pytest.raises(materializer.SupplementBindingError, match="complete 20"):
        materializer.build_binding(
            primary_binding_path=primary_path,
            actor_authority_path=actor_path,
            body_manifest_paths=supplements,
            output_path=tmp_path / "supplement" / "binding.json",
        )


def test_binding_write_is_create_once_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"
    binding = _signed({"format": "test", "value": 1})
    assert materializer.write_binding_create_once(path, binding) is True
    original = path.read_bytes()
    assert materializer.write_binding_create_once(path, binding) is False
    assert path.read_bytes() == original

    changed = _signed({"format": "test", "value": 2})
    with pytest.raises(materializer.SupplementBindingError, match="refusing"):
        materializer.write_binding_create_once(path, changed)
    assert path.read_bytes() == original

    second = tmp_path / "second.json"
    stale = second.with_suffix(".json.partial")
    stale.write_text("different", encoding="utf-8")
    with pytest.raises(materializer.SupplementBindingError, match="partial differs"):
        materializer.write_binding_create_once(second, binding)
    assert not second.exists()
    assert stale.read_text(encoding="utf-8") == "different"


def test_checkpoint_tree_hash_exactly_matches_trainer_authority_hash(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    nested = checkpoint / "weights"
    nested.mkdir()
    (nested / "model.bin").write_bytes(b"weights")
    assert collector.sha256_path(checkpoint) == trainer.sha256_tree(checkpoint)[0]


def test_collector_and_trainer_supplement_constants_are_exactly_aligned() -> None:
    assert collector.MANIFEST_FORMAT == trainer.SUPPLEMENT_MANIFEST_FORMAT
    assert collector.SUPPLEMENT_USAGE_CONTRACT == trainer.SUPPLEMENT_USAGE_CONTRACT
    assert (
        collector.EXPERT_ROOT_PROVENANCE_CONTRACT
        == trainer.EXPERT_ROOT_PROVENANCE_CONTRACT
    )
    assert collector.SUPPLEMENT_PROPER_LOSS_WEIGHT == (
        trainer.SUPPLEMENT_PROPER_LOSS_WEIGHT
    )
    assert collector.PREDEFINED_SEEDS == trainer.SUPPLEMENT_PRE_REGISTERED_SEEDS
    assert collector.HORIZON_SCHEDULE == trainer.SUPPLEMENT_HORIZON_SCHEDULE
    assert (
        collector.ROOT_SELECTION_CONTRACT
        == trainer.SUPPLEMENT_ROOT_SELECTION_CONTRACT
    )
    assert collector.HORIZON_CONTRACT == trainer.SUPPLEMENT_HORIZON_CONTRACT
    assert (
        collector.ACTOR_BRANCH_CONTRACT
        == trainer.SUPPLEMENT_ACTOR_BRANCH_CONTRACT
    )
    assert collector.EXPECTED_FIVE_BODY_DECISIONS == 100
    assert collector.EXPECTED_FIVE_BODY_BRANCHES == 400
