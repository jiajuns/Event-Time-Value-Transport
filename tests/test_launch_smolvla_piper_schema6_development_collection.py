from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from etsf_schema6_pose_quality import REGISTRY_FORMAT  # noqa: E402
import freeze_smolvla_piper_schema6_development_collection as freezer  # noqa: E402
import launch_smolvla_piper_schema6_development_collection as launcher  # noqa: E402
from freeze_smolvla_piper_schema6_development_collection import (  # noqa: E402
    CollectionAuthorityError,
    _project_signed_development_seed,
    _sensitive_path_locations,
    build_collection_authority,
    validate_collection_authority,
)
from launch_smolvla_piper_schema6_development_collection import (  # noqa: E402
    EXIT_FAILURE,
    EXIT_SUCCESS,
    DecisionTelemetryClock,
    LauncherContractError,
    RoboTwinCollectionRuntime,
    _terminate_after_durable_result,
    _write_manifest_and_receipt,
)
from materialize_smolvla_piper_schema6_reset_contract import build_pose_quality_spec  # noqa: E402
from run_smolvla_piper_r6d_direct_actor_smoke import canonical_sha256, file_sha256  # noqa: E402


def _registry() -> dict[str, object]:
    return {
        "format": REGISTRY_FORMAT,
        "objects": [
            {
                "name": "can",
                "stable_sim_actor_id": "task_attr=can;sapien_actor_name=can_actor",
                "asset_model_id": "105_sauce-can/base2",
                "role": "manipulated",
                "is_static": False,
            },
            {
                "name": "pot",
                "stable_sim_actor_id": "task_attr=pot;sapien_actor_name=pot_actor",
                "asset_model_id": "060_kitchenpot/base1",
                "role": "receptacle",
                "is_static": False,
            },
        ],
    }


def _event_spec() -> dict[str, object]:
    return {
        "calibration": {
            "move_can_pot": {
                "moving": "can",
                "anchor": None,
                "centers": [[0, 0, 0]],
            }
        },
        "chains": {
            "move_can_pot": {
                "merge_e1_e2": True,
                "chain": ["e0", "e12", "e3", "e4", "eK"],
            }
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_runtime_calls_the_pinned_four_argument_event_api() -> None:
    calls = []

    def derive(poses, names, success, event_spec):
        calls.append((poses, names, success, event_spec))
        return ["e0"], [0], ["e0"], [0]

    runtime = object.__new__(RoboTwinCollectionRuntime)
    runtime.derive_events_fn = derive
    poses = object()
    names = ["can", "pot"]
    event_spec = _event_spec()
    result = runtime.derive_events(poses, names, False, event_spec)
    assert result == (["e0"], [0], ["e0"], [0])
    assert calls == [(poses, names, False, event_spec)]


def test_signed_seed_projection_contains_only_hash_commitments() -> None:
    legacy = "/sealed/prospective_development_confirmation/seed.json"
    projected = _project_signed_development_seed(
        {
            "path": legacy,
            "sha256": "a" * 64,
            "seed_registry": "explicit_v7_prospective_development",
            "requested_seed": 100101000,
            "expected_resolved_seed": 100101000,
            "fresh_confirmation_eligible": False,
            "label_free": True,
        }
    )
    assert legacy not in json.dumps(projected, sort_keys=True)
    assert projected["legacy_manifest_content_sha256"] == "a" * 64
    assert projected["legacy_path_opened"] is False
    assert projected["legacy_path_dereferenced"] is False
    assert _sensitive_path_locations(projected) == []


def test_signed_seed_projection_rejects_second_or_noncanonical_metadata() -> None:
    seed = {
        "path": "/sealed/prospective_development_confirmation/seed.json",
        "sha256": "a" * 64,
        "seed_registry": "explicit_v7_prospective_development",
        "requested_seed": 100101000,
        "expected_resolved_seed": 100101000,
        "fresh_confirmation_eligible": False,
        "label_free": True,
        "extra_path": "/sealed/fresh/other.json",
    }
    with pytest.raises(CollectionAuthorityError, match="fields changed"):
        _project_signed_development_seed(seed)


def test_real_r6f_r6e_shape_projects_legacy_path_without_dereference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = "/sealed/prospective_development_confirmation/seed.json"
    seed = {
        "path": legacy,
        "sha256": "a" * 64,
        "seed_registry": "explicit_v7_prospective_development",
        "requested_seed": 100101000,
        "expected_resolved_seed": 100101000,
        "fresh_confirmation_eligible": False,
        "label_free": True,
    }
    r6c_record = {
        "manifest_path": "/safe/r6c/manifest.json",
        "manifest_sha256": "1" * 64,
        "receipt_path": "/safe/r6c/receipt.json",
        "receipt_sha256": "2" * 64,
        "verifier_sha256": "3" * 64,
    }
    r6d_record = {
        "preregistration_path": "/safe/r6d/preregistration.json",
        "preregistration_sha256": "4" * 64,
        "preregistration_logical_sha256": "5" * 64,
        "receipt_path": "/safe/r6d/receipt.json",
        "receipt_sha256": "6" * 64,
        "executor_sha256": "7" * 64,
        "runtime_source_artifacts": {},
        "authorization": "evidence_only_not_direct_actor_execution_authority",
    }
    roots = {
        "rlinf_root": "/safe/rlinf",
        "robotwin_root": "/safe/robotwin",
        "robotwin_code": "/safe/robotwin_code",
        "lerobot_root": "/safe/lerobot",
        "model_path": "/safe/model",
        "vlm_metadata_path": "/safe/vlm",
    }
    historical_root = tmp_path / "historical_r6e_code"
    current_root = tmp_path / "current_r6j_code"
    historical_root.mkdir()
    current_root.mkdir()
    historical_sources = {}
    current_sources = {}
    for role in freezer.HISTORICAL_IMPLEMENTATION_ROLES:
        historical_path = historical_root / f"{role}.py"
        current_path = current_root / f"{role}.py"
        payload = f"# byte-identical implementation for {role}\n"
        historical_path.write_text(payload, encoding="utf-8")
        current_path.write_text(payload, encoding="utf-8")
        historical_path.chmod(0o444)
        current_path.chmod(0o444)
        digest = file_sha256(historical_path)
        historical_sources[role] = {"path": str(historical_path), "sha256": digest}
        current_sources[role] = {"path": str(current_path), "sha256": digest}
    r6e = {
        "r6c_binding": r6c_record,
        "r6d_binding": r6d_record,
        "development_seed": seed,
        "runtime_roots": roots,
        "runtime_source_artifacts": historical_sources,
        "vlm_metadata_bundle_sha256": "8" * 64,
        "model_bundle_sha256": "9" * 64,
        "capability_contract": {"fresh_inputs_allowed": False},
        "mapping_contract": {"mode": "explicit_named_ordinal_angle_preserving_mapping"},
        "state_contract": {"is_measured_qpos": False},
        "caveats": {"performance_or_transfer_claim": False},
        "output": "/safe/r6e/output.json",
    }
    r6e["preregistration_sha256"] = canonical_sha256(r6e)
    current_expected_r6e = copy.deepcopy(r6e)
    current_expected_r6e["runtime_source_artifacts"] = current_sources
    current_base = {
        key: value
        for key, value in current_expected_r6e.items()
        if key != "preregistration_sha256"
    }
    current_expected_r6e["preregistration_sha256"] = canonical_sha256(current_base)
    inherited = {
        key: r6e[key]
        for key in (
            "r6c_binding", "r6d_binding", "development_seed", "runtime_roots",
            "runtime_source_artifacts", "vlm_metadata_bundle_sha256",
            "model_bundle_sha256", "capability_contract", "mapping_contract",
            "state_contract", "caveats",
        )
    }
    r6e_binding = {"path": "/safe/r6e/preregistration.json", "sha256": "c" * 64}
    r6f = {
        "r6e_lineage": r6e_binding,
        "inherited_R6e_contract": inherited,
        "inherited_R6e_contract_sha256": canonical_sha256(inherited),
        "preregistration_sha256": "d" * 64,
    }
    bound_r6c = {"bound": "r6c"}
    bound_r6d = {"bound": "r6d"}
    monkeypatch.setattr(freezer, "validate_feasibility_preregistration", lambda _path: copy.deepcopy(r6f))
    monkeypatch.setattr(freezer, "bind_r6e_preregistration", lambda _path: copy.deepcopy(r6e_binding))
    monkeypatch.setattr(freezer, "validate_direct_actor_preregistration", lambda _path: copy.deepcopy(r6e))
    monkeypatch.setattr(freezer, "bind_r6c_preflight", lambda *_paths: bound_r6c)
    monkeypatch.setattr(freezer, "bind_r6d_simulation_receipt", lambda *_paths: bound_r6d)
    monkeypatch.setattr(
        freezer,
        "build_direct_actor_preregistration",
        lambda **_kwargs: copy.deepcopy(current_expected_r6e),
    )

    safe_r6f, safe_r6e, got_r6c, got_r6d, projected = freezer.load_r6f_lineage_for_collection(
        tmp_path / "safe_outer_r6f.json"
    )
    for value in (safe_r6f, safe_r6e, projected):
        assert legacy not in json.dumps(value, sort_keys=True)
        assert _sensitive_path_locations(value) == []
    assert got_r6c is bound_r6c
    assert got_r6d is bound_r6d
    assert projected["legacy_path_opened"] is False
    assert projected["legacy_path_dereferenced"] is False

    current_expected_r6e["model_bundle_sha256"] = "0" * 64
    with pytest.raises(CollectionAuthorityError, match="differs beyond"):
        freezer.load_r6f_lineage_for_collection(tmp_path / "safe_outer_r6f.json")
    current_expected_r6e["model_bundle_sha256"] = r6e["model_bundle_sha256"]
    current_base = {
        key: value
        for key, value in current_expected_r6e.items()
        if key != "preregistration_sha256"
    }
    current_expected_r6e["preregistration_sha256"] = canonical_sha256(current_base)

    historical_runner = Path(historical_sources["direct_actor_runner"]["path"])
    historical_runner.chmod(0o644)
    with pytest.raises(CollectionAuthorityError, match="read-only regular file"):
        freezer.load_r6f_lineage_for_collection(tmp_path / "safe_outer_r6f.json")


def _r6f(path: Path, robotwin_code: Path) -> dict[str, object]:
    return {
        "status": "preregistered_R6f_feasibility_simulation_only_not_executed",
        "explicit_instruction": "move the can into the pot",
        "preregistration_sha256": "a" * 64,
        "inherited_R6e_contract_sha256": "b" * 64,
        "inherited_R6e_contract": {
            "runtime_roots": {"robotwin_code": str(robotwin_code)},
            "development_seed": {
                "requested_seed": 100101000,
                "expected_resolved_seed": 100101000,
                "fresh_confirmation_eligible": False,
                "label_free": True,
            },
            "r6d_binding": {
                "runtime_source_artifacts": {
                    "robotwin_base_task": {"path": "/bound/_base_task.py", "sha256": "c" * 64}
                }
            },
            "mapping_contract": {
                "mode": "explicit_named_ordinal_angle_preserving_mapping",
                "derived_from_equal_14d_width": False,
                "kinematic_equivalence_claimed": False,
                "physical_equivalence_claimed": False,
                "clipping_or_scaling_forbidden": True,
            },
            "state_contract": {
                "semantics": "[left drive_target q1..q6,left normalized gripper,right drive_target q1..q6,right normalized gripper]",
                "is_measured_qpos": False,
            },
            "caveats": {
                "reported_duration": "policy row count not physical time",
                "performance_or_transfer_claim": False,
            },
        },
    }


def _authority_fixture(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    spec_path = tmp_path / "pose_spec.json"
    event_path = tmp_path / "event_spec.json"
    r6f_path = tmp_path / "r6f.json"
    robotwin_code = tmp_path / "robotwin_code"
    task_source = robotwin_code / "envs/move_can_pot.py"
    task_source.parent.mkdir(parents=True)
    task_source.write_text("# frozen synthetic task source\n", encoding="utf-8")
    _write(registry_path, _registry())
    _write(spec_path, build_pose_quality_spec(
        _registry(),
        move_can_pot_source={"path": str(task_source), "sha256": file_sha256(task_source)},
    ))
    _write(event_path, _event_spec())
    _write(r6f_path, {})
    r6f = _r6f(r6f_path, robotwin_code)
    output = tmp_path / "new_output"
    value = build_collection_authority(
        r6f_preregistration=r6f,
        r6f_path=r6f_path,
        object_registry_path=registry_path,
        pose_quality_spec_path=spec_path,
        event_spec_path=event_path,
        output_directory=output,
        max_episode_steps=12,
    )
    authority_path = tmp_path / "authority.json"
    _write(authority_path, value)
    return authority_path, value, r6f, output


def test_new_authority_is_not_an_r6f_smoke_capability_upgrade(tmp_path: Path) -> None:
    path, value, r6f, _ = _authority_fixture(tmp_path)

    def loader(_path):
        return r6f, {}, {}, {}, {}

    validated, *_ = validate_collection_authority(path, r6f_loader=loader)
    assert validated["authority_basis"]["this_is_new_collection_authority"] is True
    assert validated["authority_basis"]["R6f_four_step_smoke_reinterpreted_as_collection_authority"] is False
    assert validated["scope"]["candidate_indices"] == [0, 1, 2, 3]
    assert validated["scope"]["root_minimum_legal_candidates"] == 2
    assert validated["scope"]["root_action_horizon"] == 1
    assert validated["scope"]["continuation_action_horizon"] == 1
    assert validated["scope"]["tracked_pose_objects"] == ["can", "pot"]
    assert validated["scope"]["max_episode_steps"] == 12
    assert validated["capability_contract"]["performance_evaluation_authorized"] is False
    assert validated["object_identity_contract"]["validated_on_every_reset_before_policy_forward_or_action"] is True
    assert validated["object_identity_contract"]["random_clutter_table_wall_excluded"] is True
    assert validated["telemetry_source_binding"] == {
        "robotwin_base_task": {"path": "/bound/_base_task.py", "sha256": "c" * 64},
        "counted_call_site": "BaseTask.gen_sparse_reward_data TOPP loop self.scene.step",
        "counter_mode": "per-reset_instance_bound_scene_step_wrapper",
        "timestamp_mode": "cumulative_scene_step_count_times_scene_get_timestep",
        "physics_substeps_mode": "counted_scene_step_calls_since_previous_snapshot",
    }


def test_authority_tamper_or_existing_output_fails_closed(tmp_path: Path) -> None:
    path, value, r6f, output = _authority_fixture(tmp_path)
    changed = copy.deepcopy(value)
    changed["capability_contract"]["transfer_claim_authorized"] = True
    changed["authority_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "authority_sha256"}
    )
    _write(path, changed)
    with pytest.raises(CollectionAuthorityError, match="capability"):
        validate_collection_authority(path, r6f_loader=lambda _: (r6f, {}, {}, {}, {}))
    changed = copy.deepcopy(value)
    changed["telemetry_source_binding"]["robotwin_base_task"]["sha256"] = "d" * 64
    changed["authority_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "authority_sha256"}
    )
    _write(path, changed)
    with pytest.raises(CollectionAuthorityError, match="full recomputation"):
        validate_collection_authority(path, r6f_loader=lambda _: (r6f, {}, {}, {}, {}))
    _write(path, value)
    output.mkdir()
    with pytest.raises(FileExistsError):
        validate_collection_authority(path, r6f_loader=lambda _: (r6f, {}, {}, {}, {}))


def test_bound_move_can_pot_source_change_invalidates_authority(tmp_path: Path) -> None:
    path, value, r6f, _ = _authority_fixture(tmp_path)
    task_source = Path(value["object_identity_contract"]["move_can_pot_source"]["path"])
    task_source.write_text("# changed after authority freeze\n", encoding="utf-8")
    with pytest.raises(CollectionAuthorityError, match="pose-quality spec differs"):
        validate_collection_authority(path, r6f_loader=lambda _: (r6f, {}, {}, {}, {}))


class _Scene:
    def __init__(self) -> None:
        self.executed = 0

    def get_timestep(self):
        return 0.005

    def step(self):
        self.executed += 1


def test_decision_clock_counts_exact_instance_scene_steps_since_snapshot() -> None:
    scene = _Scene()
    clock = DecisionTelemetryClock(scene, object_count=2)
    initial = clock.telemetry(pose_error=[False, False])
    assert initial["control_step"] == 0
    assert initial["physics_substep_count"] == 0
    assert initial["simulator_timestamp_s"] == 0.0
    for _ in range(7):
        scene.step()
    clock.after_policy_step()
    updated = clock.telemetry(pose_error=[False, True])
    assert updated["control_step"] == 1
    assert updated["physics_substep_count"] == 7
    assert updated["simulator_timestamp_s"] == pytest.approx(0.035)
    assert updated["simulator_pose_error_flag"].tolist() == [False, True]
    assert "instrumented_" in clock.contract()["timestamp_origin"]
    assert clock.telemetry(pose_error=[False, False])["physics_substep_count"] == 0


def test_decision_clock_does_not_fabricate_missing_or_unadvanced_telemetry() -> None:
    with pytest.raises(LauncherContractError, match="scene.step/get_timestep"):
        DecisionTelemetryClock(object(), object_count=1)
    scene = _Scene()
    clock = DecisionTelemetryClock(scene, object_count=1)
    with pytest.raises(LauncherContractError, match="did not advance"):
        clock.after_policy_step()


def test_decision_clock_can_exactly_instrument_public_scene_steps() -> None:
    class StepScene:
        def __init__(self):
            self.executed = 0

        def get_timestep(self):
            return 0.004

        def step(self):
            self.executed += 1

    scene = StepScene()
    clock = DecisionTelemetryClock(scene, object_count=1)
    for _ in range(7):
        scene.step()
    clock.after_policy_step()
    telemetry = clock.telemetry(pose_error=[False])
    assert telemetry["physics_substep_count"] == 7
    assert telemetry["simulator_timestamp_s"] == pytest.approx(0.028)
    assert "instrumented_" in clock.contract()["timestamp_origin"]
    clock.close()
    scene.step()
    assert scene.executed == 8


def test_manifest_receipt_bind_hashes_status_and_exit(tmp_path: Path) -> None:
    authority_path, authority, _, output = _authority_fixture(tmp_path)
    output.mkdir()
    result = _write_manifest_and_receipt(
        authority=authority,
        authority_path=authority_path,
        output=output,
        status="failed_closed_schema6_development_collection",
        exit_code=EXIT_FAILURE,
        group_path=None,
        audit=None,
        env_steps=0,
        identity_validation_count=0,
        error=LauncherContractError("synthetic failure"),
        clock_contracts=[],
    )
    assert result["exit_code"] == EXIT_FAILURE
    assert file_sha256(Path(result["manifest_path"])) == result["manifest_file_sha256"]
    assert file_sha256(Path(result["receipt_path"])) == result["receipt_file_sha256"]
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    logical = receipt.pop("receipt_logical_sha256")
    assert logical == canonical_sha256(receipt)
    assert receipt["failure"]["fail_closed"] is True
    assert receipt["performance_evaluation_authorized"] is False


def test_cli_hard_exit_occurs_only_after_durable_receipt_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path, authority, _, output = _authority_fixture(tmp_path)
    output.mkdir()
    result = _write_manifest_and_receipt(
        authority=authority,
        authority_path=authority_path,
        output=output,
        status="completed_root_fewer_than_two_legal_no_group",
        exit_code=20,
        group_path=None,
        audit=None,
        env_steps=0,
        identity_validation_count=1,
        error=None,
        clock_contracts=[],
    )
    exits: list[int] = []

    class HardExitObserved(BaseException):
        pass

    def fake_hard_exit(code: int) -> None:
        exits.append(code)
        raise HardExitObserved

    terminal_log = tmp_path / "watcher.log"
    with terminal_log.open("w", encoding="utf-8") as stream:
        with monkeypatch.context() as isolated:
            isolated.setattr(launcher, "_PROCESS_HARD_EXIT", fake_hard_exit)
            isolated.setattr(launcher.sys, "stdout", stream)
            with pytest.raises(HardExitObserved):
                _terminate_after_durable_result(
                    result, runtime_release_succeeded=True
                )
    assert exits == [20]
    assert json.loads(terminal_log.read_text(encoding="utf-8")) == result


def test_cli_hard_exit_rejects_receipt_changed_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path, authority, _, output = _authority_fixture(tmp_path)
    output.mkdir()
    result = _write_manifest_and_receipt(
        authority=authority,
        authority_path=authority_path,
        output=output,
        status="failed_closed_schema6_development_collection",
        exit_code=EXIT_FAILURE,
        group_path=None,
        audit=None,
        env_steps=0,
        identity_validation_count=0,
        error=LauncherContractError("synthetic failure"),
        clock_contracts=[],
    )
    receipt_path = Path(result["receipt_path"])
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    exits: list[int] = []

    class HardExitObserved(BaseException):
        pass

    def fake_hard_exit(code: int) -> None:
        exits.append(code)
        raise HardExitObserved

    monkeypatch.setattr(launcher, "_PROCESS_HARD_EXIT", fake_hard_exit)
    with pytest.raises(HardExitObserved):
        _terminate_after_durable_result(
            result, runtime_release_succeeded=False
        )
    assert exits == [EXIT_FAILURE]


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "release_succeeded"),
    [
        ("status", "completed_one_seed_schema6_development_collection", True),
        (None, None, False),
    ],
)
def test_cli_success_contract_violation_hard_exits_failure_without_unwind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str | None,
    changed_value: object,
    release_succeeded: bool,
) -> None:
    authority_path, authority, _, output = _authority_fixture(tmp_path)
    output.mkdir()
    result = _write_manifest_and_receipt(
        authority=authority,
        authority_path=authority_path,
        output=output,
        status="completed_root_fewer_than_two_legal_no_group",
        exit_code=20,
        group_path=None,
        audit=None,
        env_steps=0,
        identity_validation_count=1,
        error=None,
        clock_contracts=[],
    )
    if changed_field is not None:
        result[changed_field] = changed_value
    exits: list[int] = []

    class HardExitObserved(BaseException):
        pass

    def fake_hard_exit(code: int) -> None:
        exits.append(code)
        raise HardExitObserved

    monkeypatch.setattr(launcher, "_PROCESS_HARD_EXIT", fake_hard_exit)
    with pytest.raises(HardExitObserved):
        _terminate_after_durable_result(
            result, runtime_release_succeeded=release_succeeded
        )
    assert exits == [EXIT_FAILURE]


def test_real_hard_exit_bypasses_python_finalizers_after_fsync(
    tmp_path: Path,
) -> None:
    authority_path, authority, _, output = _authority_fixture(tmp_path)
    output.mkdir()
    result = _write_manifest_and_receipt(
        authority=authority,
        authority_path=authority_path,
        output=output,
        status="completed_root_fewer_than_two_legal_no_group",
        exit_code=20,
        group_path=None,
        audit=None,
        env_steps=0,
        identity_validation_count=1,
        error=None,
        clock_contracts=[],
    )
    terminal_log = tmp_path / "hard_exit.log"
    destructor_marker = tmp_path / "destructor_ran"
    source = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import launch_smolvla_piper_schema6_development_collection as launcher
marker = Path({str(destructor_marker)!r})
class NativeFinalizerSentinel:
    def __del__(self):
        marker.write_text('unexpected interpreter finalizer', encoding='utf-8')
sentinel = NativeFinalizerSentinel()
with Path({str(terminal_log)!r}).open('w', encoding='utf-8') as stream:
    sys.stdout = stream
    launcher._terminate_after_durable_result(
        {result!r}, runtime_release_succeeded=True
    )
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 20, completed.stderr
    assert not destructor_marker.exists()
    assert json.loads(terminal_log.read_text(encoding="utf-8")) == result
