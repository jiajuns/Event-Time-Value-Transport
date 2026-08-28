from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from execute_smolvla_piper_r6c_simulation_smoke import (  # noqa: E402
    ACTION_DIM,
    ACTOR_ID,
    EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    MAX_SMOKE_STEPS,
    PREREGISTRATION_FORMAT,
    REQUESTED_DEVELOPMENT_SEED,
    ActionLimitError,
    SimulationSmokeError,
    atomic_json,
    bind_development_seed_manifest,
    bind_r6c_preflight,
    build_simulation_preregistration,
    canonical_sha256,
    execute_validated_steps,
    load_r6c_mapped_candidate,
    piper_environment_config,
    reset_with_explicit_instruction,
    validate_piper_step,
    validate_reset_observation,
    validate_simulation_preregistration,
)
from verify_smolvla_piper_zero_shot_preflight import (  # noqa: E402
    PIPER_ACTION_SLOTS,
    adapt_aloha_source_actions_to_piper_forward_interface,
    array_sha256,
    file_sha256,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bounds() -> list[list[float]]:
    return [[slot.lower, slot.upper] for slot in PIPER_ACTION_SLOTS]


def _valid_chunk() -> np.ndarray:
    chunk = np.zeros((50, ACTION_DIM), dtype=np.float32)
    chunk[:, [1, 8]] = 1.0
    chunk[:, [6, 13]] = 0.5
    return chunk


def _fake_r6c_files(tmp_path: Path):
    directory = tmp_path / "r6c_fixture"
    manifest_path = directory / "preflight_manifest.json"
    receipt_path = directory / "preflight_receipt.json"
    manifest = {
        "capability_contract": {
            "fresh_inputs_allowed": False,
            "environment_step_allowed": False,
            "outcome_inputs_allowed": False,
            "execution_authorized": False,
            "transfer_claim_authorized": False,
            "maximum_authorization": "forward_only",
        }
    }
    _json(manifest_path, manifest)
    recomputed = {
        "format": "fixture",
        "status": "passed_forward_only",
        "actor_id": ACTOR_ID,
        "authorization": "forward_only",
        "environment_execution_authorized": False,
        "transfer_claim_authorized": False,
        "data_blind": True,
        "implementation_sha256": file_sha256(
            ROOT / "scripts" / "verify_smolvla_piper_zero_shot_preflight.py"
        ),
    }
    receipt = {
        **recomputed,
        "manifest_file_sha256": file_sha256(manifest_path),
    }
    _json(receipt_path, receipt)
    return manifest_path, receipt_path, recomputed


def test_r6c_binding_rehashes_manifest_receipt_and_recomputes(tmp_path: Path) -> None:
    manifest_path, receipt_path, recomputed = _fake_r6c_files(tmp_path)
    calls: list[dict[str, object]] = []

    def runner(value):
        calls.append(value)
        return recomputed

    binding = bind_r6c_preflight(
        manifest_path,
        receipt_path,
        expected_manifest_sha256=file_sha256(manifest_path),
        expected_receipt_sha256=file_sha256(receipt_path),
        expected_verifier_sha256=recomputed["implementation_sha256"],
        expected_directory_name="r6c_fixture",
        runner=runner,
    )
    assert binding["manifest_sha256"] == file_sha256(manifest_path)
    assert binding["receipt_sha256"] == file_sha256(receipt_path)
    assert calls == [binding["manifest"]]


def test_r6c_binding_fails_on_tamper_or_capability_upgrade(tmp_path: Path) -> None:
    manifest_path, receipt_path, recomputed = _fake_r6c_files(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capability_contract"]["environment_step_allowed"] = True
    _json(manifest_path, manifest)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_file_sha256"] = file_sha256(manifest_path)
    _json(receipt_path, receipt)
    with pytest.raises(SimulationSmokeError, match="capability boundary"):
        bind_r6c_preflight(
            manifest_path,
            receipt_path,
            expected_manifest_sha256=file_sha256(manifest_path),
            expected_receipt_sha256=file_sha256(receipt_path),
            expected_verifier_sha256=recomputed["implementation_sha256"],
            expected_directory_name="r6c_fixture",
            runner=lambda _: recomputed,
        )


def test_development_seed_contract_is_label_free_and_never_fresh(tmp_path: Path) -> None:
    path = tmp_path / "v7_development_seed_manifest.json"
    value = {
        "format": "etsf_robotwin_v7_development_seed_manifest_v1",
        "status": "preregistered_resolved_label_free",
        "purpose": "independent_prospective_development_confirmation_never_fresh",
        "seed_registry": "explicit_v7_prospective_development",
        "fresh_confirmation_eligible": False,
        "task": "move_can_pot",
        "label_access_contract": (
            "reset_identity_only_no_policy_no_action_no_event_no_success_no_reward"
        ),
        "train": [
            {
                "seed": REQUESTED_DEVELOPMENT_SEED,
                "requested_seed": REQUESTED_DEVELOPMENT_SEED,
                "resolved_seed": EXPECTED_RESOLVED_DEVELOPMENT_SEED,
            }
        ],
    }
    _json(path, value)
    result = bind_development_seed_manifest(path, expected_sha256=file_sha256(path))
    assert result["fresh_confirmation_eligible"] is False
    assert result["label_free"] is True


def test_explicit_mapped_candidate_is_bound_to_r6c_receipt(tmp_path: Path) -> None:
    source = np.stack([_valid_chunk() + i * 0.01 for i in range(4)])
    source[:, :, [6, 13]] = 0.5
    path = tmp_path / "candidates.npy"
    np.save(path, source, allow_pickle=False)
    mapped, mapping_receipt = adapt_aloha_source_actions_to_piper_forward_interface(source)
    hashes = [array_sha256(mapped[index]) for index in range(4)]
    binding = {
        "manifest": {
            "probe_artifacts": {
                "candidate_actions": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            }
        },
        "receipt": {
            "action_mapping_validation": mapping_receipt,
            "candidate_validation": {"candidate_sha256": hashes},
        },
    }
    selected, contract = load_r6c_mapped_candidate(binding, 2)
    assert np.array_equal(selected, source[2])
    assert contract["identity_inferred_from_equal_dimension"] is False
    assert contract["kinematic_equivalence_claimed"] is False


def test_each_action_is_named_bounded_and_never_clipped() -> None:
    action = _valid_chunk()[0]
    result = validate_piper_step(action, _bounds(), step_index=0)
    assert result["clipping_applied"] is False
    assert len(result["slots"]) == 14
    assert [slot["target_joint_name"] for slot in result["slots"]] == [
        slot.target_joint_name for slot in PIPER_ACTION_SLOTS
    ]


class _FakeEnv:
    def __init__(self, terminate_after: int | None = None):
        self.actions: list[np.ndarray] = []
        self.terminate_after = terminate_after

    def step(self, action, *, auto_reset: bool):
        assert auto_reset is False
        self.actions.append(np.asarray(action).copy())
        terminated = self.terminate_after == len(self.actions)
        return {}, 0.0, [terminated], [False], {"success": [False]}


def test_short_smoke_executes_h1_exact_actions_and_stops_on_termination() -> None:
    env = _FakeEnv(terminate_after=2)
    chunk = _valid_chunk()
    result = execute_validated_steps(
        env,
        chunk,
        _bounds(),
        step_limit=MAX_SMOKE_STEPS,
        action_converter=lambda action: action.reshape(1, 1, 14),
    )
    assert result["steps_executed"] == 2
    assert result["stopped_on_termination"] is True
    assert result["silent_clipping_possible"] is False
    assert np.array_equal(env.actions[0].reshape(14), chunk[0])
    assert np.array_equal(env.actions[1].reshape(14), chunk[1])


@pytest.mark.parametrize("failure", ["limit", "nan", "converter_clip"])
def test_invalid_or_changed_action_never_reaches_env_step(failure: str) -> None:
    env = _FakeEnv()
    chunk = _valid_chunk()
    converter = lambda action: action.reshape(1, 1, 14)
    if failure == "limit":
        chunk[0, 4] = 1.2201
    elif failure == "nan":
        chunk[0, 4] = np.nan
    else:
        converter = lambda action: np.minimum(action, 0.25).reshape(1, 1, 14)
    with pytest.raises(ActionLimitError):
        execute_validated_steps(
            env,
            chunk,
            _bounds(),
            step_limit=1,
            action_converter=converter,
        )
    assert env.actions == []


def test_environment_contract_is_dual_piper_with_required_observation_path(
    tmp_path: Path,
) -> None:
    config = piper_environment_config(
        tmp_path / "RoboTwin", tmp_path / "seeds.json", tmp_path, step_limit=4
    )
    assert config["env_type"] == "robotwin"
    assert config["center_crop"] is False
    assert config["task_config"]["embodiment"] == ["piper", "piper", 0.6]
    assert config["task_config"]["camera"]["collect_wrist_camera"] is True
    assert config["task_config"]["collect_data"] is False
    assert config["video_cfg"]["save_video"] is False


def test_preregistration_freezes_independent_simulation_authority_and_caveats(
    tmp_path: Path,
) -> None:
    for name in ("rlinf", "robotwin", "robotwin_code"):
        (tmp_path / name).mkdir()
    rlinf_source = tmp_path / "rlinf/rlinf/envs/robotwin/robotwin_env.py"
    vector_source = tmp_path / "robotwin_code/robotwin/envs/vector_env.py"
    base_task_source = tmp_path / "robotwin_code/envs/_base_task.py"
    robot_source = tmp_path / "robotwin_code/envs/robot/robot.py"
    for path in (rlinf_source, vector_source, base_task_source, robot_source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture {path.name}\n", encoding="utf-8")
    binding = {
        "manifest_path": "/bound/preflight_manifest.json",
        "manifest_sha256": "1" * 64,
        "receipt_path": "/bound/preflight_receipt.json",
        "receipt_sha256": "2" * 64,
    }
    seed = {
        "path": "/bound/v7_seed_manifest.json",
        "sha256": "3" * 64,
        "seed_registry": "explicit_v7_prospective_development",
        "requested_seed": REQUESTED_DEVELOPMENT_SEED,
        "expected_resolved_seed": EXPECTED_RESOLVED_DEVELOPMENT_SEED,
        "fresh_confirmation_eligible": False,
        "label_free": True,
    }
    candidate = {"candidate_sha256": "4" * 64}
    prereg = build_simulation_preregistration(
        binding=binding,
        seed_contract=seed,
        candidate_contract=candidate,
        rlinf_root=tmp_path / "rlinf",
        robotwin_root=tmp_path / "robotwin",
        robotwin_code=tmp_path / "robotwin_code",
        output=tmp_path / "result.json",
        candidate_index=0,
        step_limit=4,
    )
    assert prereg["format"] == PREREGISTRATION_FORMAT
    assert prereg["simulation_capability"]["simulation_execution_authorized"] is True
    assert prereg["simulation_capability"]["real_robot_execution_authorized"] is False
    assert prereg["preflight"]["authorization_remains"] == "forward_only"
    assert prereg["fresh_contract"]["fresh_inputs_allowed"] is False
    assert prereg["observation_contract"]["state_is_measured_qpos"] is False
    assert prereg["language_and_seed_caveat"]["scene_seed_and_instruction_strictly_bound"] is False
    assert prereg["time_contract"]["physical_duration_claimed"] is False
    assert prereg["time_contract"]["one_policy_row_may_expand_to_many_simulator_control_steps"] is True
    assert prereg["action_contract"]["RoboTwin_internal_gripper_clip_present"] is True
    original_controller_sha = prereg["runtime_source_artifacts"][
        "robotwin_robot_controller"
    ]["sha256"]
    path = tmp_path / "preregistration.json"
    atomic_json(path, prereg)
    assert path.stat().st_mode & 0o777 == 0o444
    assert validate_simulation_preregistration(path) == prereg

    tampered = copy.deepcopy(prereg)
    tampered["simulation_capability"]["real_robot_execution_authorized"] = True
    base = {k: v for k, v in tampered.items() if k != "preregistration_sha256"}
    tampered["preregistration_sha256"] = canonical_sha256(base)
    tampered_path = tmp_path / "tampered.json"
    atomic_json(tampered_path, tampered)
    with pytest.raises(SimulationSmokeError, match="simulation-only"):
        validate_simulation_preregistration(tampered_path)

    robot_source.write_text("# changed controller\n", encoding="utf-8")
    rebuilt = build_simulation_preregistration(
        binding=binding,
        seed_contract=seed,
        candidate_contract=candidate,
        rlinf_root=tmp_path / "rlinf",
        robotwin_root=tmp_path / "robotwin",
        robotwin_code=tmp_path / "robotwin_code",
        output=tmp_path / "result.json",
        candidate_index=0,
        step_limit=4,
    )
    assert rebuilt["runtime_source_artifacts"]["robotwin_robot_controller"][
        "sha256"
    ] != original_controller_sha
    assert rebuilt["preregistration_sha256"] != prereg["preregistration_sha256"]


class _ResetSubEnv:
    def __init__(self) -> None:
        self.create_instruction = lambda: "stale instruction"


class _ResetEnv:
    def __init__(self) -> None:
        self.venv = type("Venv", (), {"envs": [_ResetSubEnv()]})()
        self.observed_instruction = None

    def reset(self, *, env_seeds):
        assert env_seeds == [REQUESTED_DEVELOPMENT_SEED]
        self.observed_instruction = self.venv.envs[0].create_instruction()
        return {"task_descriptions": [self.observed_instruction]}, {}


def test_reset_uses_explicit_instruction_and_restores_generator() -> None:
    env = _ResetEnv()
    original = env.venv.envs[0].create_instruction
    observation, _ = reset_with_explicit_instruction(
        env,
        requested_seed=REQUESTED_DEVELOPMENT_SEED,
        instruction="move the can into the pot",
    )
    assert observation["task_descriptions"] == ["move the can into the pot"]
    assert env.venv.envs[0].create_instruction is original


def test_reset_observation_requires_main_and_two_wrist_uint8_images() -> None:
    observation = {
        "states": np.zeros((1, 14), dtype=np.float32),
        "main_images": np.zeros((1, 240, 320, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 240, 320, 3), dtype=np.uint8),
    }
    result = validate_reset_observation(observation)
    assert result["state_shape"] == [1, 14]
    assert result["wrist_images_shape"] == [1, 2, 240, 320, 3]
    broken = dict(observation)
    broken["wrist_images"] = np.zeros((1, 1, 240, 320, 3), dtype=np.uint8)
    with pytest.raises(SimulationSmokeError, match="wrist images"):
        validate_reset_observation(broken)


def test_fresh_paths_and_source_level_clipping_are_forbidden(tmp_path: Path) -> None:
    fresh = tmp_path / "Fresh50" / "preregistration.json"
    fresh.parent.mkdir()
    _json(fresh, {})
    with pytest.raises(Exception, match="Fresh"):
        validate_simulation_preregistration(fresh)
    source = (ROOT / "scripts" / "execute_smolvla_piper_r6c_simulation_smoke.py").read_text(
        encoding="utf-8"
    )
    assert ".clip(" not in source
    assert "torch.clamp(" not in source
