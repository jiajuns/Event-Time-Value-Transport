from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_smolvla_piper_r6d_direct_actor_smoke import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    DIRECT_MAX_STEPS,
    EXPECTED_IMAGE_KEYS,
    INSTRUCTION,
    PREFIX_DIM,
    R6D_PREREGISTRATION_LOGICAL_SHA256,
    DirectActorError,
    bind_r6d_simulation_receipt,
    canonical_sha256,
    directory_bundle_sha256,
    explicit_named_map_online_chunk,
    offload_runtime,
    run_online_actor_loop,
    validate_direct_actor_preregistration,
    validate_loaded_policy_contract,
    validate_runtime_module_origins,
)
from verify_smolvla_piper_zero_shot_preflight import (  # noqa: E402
    ALOHA_FEATURE_NAMES,
    PIPER_ACTION_SLOTS,
    file_sha256,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bounds() -> list[list[float]]:
    return [[slot.lower, slot.upper] for slot in PIPER_ACTION_SLOTS]


def _observation(value: float) -> dict[str, object]:
    state = np.zeros((1, ACTION_DIM), dtype=np.float32)
    state[:, [6, 13]] = 0.5
    state[:, 0] = value
    main = np.full((1, 8, 9, 3), int(value * 10) % 255, dtype=np.uint8)
    wrists = np.stack([main, main + 1], axis=1)
    return {
        "states": torch.from_numpy(state),
        "main_images": torch.from_numpy(main),
        "wrist_images": torch.from_numpy(wrists),
        "task_descriptions": [INSTRUCTION],
    }


class _Capture:
    def __init__(self) -> None:
        self.value: torch.Tensor | None = None
        self.closed = False

    def reset(self) -> None:
        self.value = None

    def consume(self) -> torch.Tensor:
        assert self.value is not None
        return self.value

    def close(self) -> None:
        self.closed = True


class _Policy:
    def __init__(self, capture: _Capture) -> None:
        self.capture = capture
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=32,
            image_features={key: object() for key in EXPECTED_IMAGE_KEYS},
        )
        self.states_seen: list[float] = []

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch, *, noise):
        state = batch["observation.state"]
        scalar = float(state[0, 0])
        self.states_seen.append(scalar)
        prefix = torch.full((PREFIX_DIM,), scalar, dtype=torch.float32)
        self.capture.value = prefix
        chunk = torch.zeros((1, CHUNK_SIZE, ACTION_DIM), dtype=torch.float32)
        chunk[:, :, [6, 13]] = 0.5
        chunk[:, :, 0] = scalar * 0.01
        return chunk


class _Env:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def step(self, action, *, auto_reset: bool):
        assert auto_reset is False
        self.actions.append(np.asarray(action).copy())
        index = len(self.actions)
        return _observation(float(index)), 0.0, [False], [False], {"success": [False]}


def _preprocessor(raw):
    result = dict(raw)
    result["observation.state"] = raw["observation.state"].reshape(1, ACTION_DIM)
    return result


def _noise(_config, scene_seed: int, query_index: int, _device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(scene_seed + query_index * 17)
    return torch.randn((1, CHUNK_SIZE, 32), generator=generator)


def test_online_loop_requeries_current_observation_and_records_cpu_state_prefix() -> None:
    capture = _Capture()
    policy = _Policy(capture)
    env = _Env()
    result = run_online_actor_loop(
        env=env,
        policy=policy,
        preprocessor=_preprocessor,
        postprocessor=lambda value: value,
        capture=capture,
        observation=_observation(0.0),
        bounds=_bounds(),
        device=torch.device("cpu"),
        action_converter=lambda value: value.reshape(1, 1, ACTION_DIM),
        noise_factory=_noise,
        max_steps=DIRECT_MAX_STEPS,
    )
    assert policy.states_seen == [0.0, 1.0, 2.0, 3.0]
    assert result["queries_performed"] == result["steps_executed"] == 4
    assert len(env.actions) == 4
    assert all(action.shape == (1, 1, ACTION_DIM) for action in env.actions)
    assert result["online_forward_from_current_observation"] is True
    assert result["precomputed_action_chunks_used"] is False
    assert len(result["queries"][2]["processed_state"]) == ACTION_DIM
    assert len(result["queries"][2]["shared_prefix"]) == PREFIX_DIM
    assert result["queries"][2]["shared_prefix"][0] == 2.0
    assert len({query["noise_sha256"] for query in result["queries"]}) == 4
    rerun_capture = _Capture()
    rerun = run_online_actor_loop(
        env=_Env(), policy=_Policy(rerun_capture), preprocessor=_preprocessor,
        postprocessor=lambda value: value, capture=rerun_capture,
        observation=_observation(0.0), bounds=_bounds(), device=torch.device("cpu"),
        action_converter=lambda value: value.reshape(1, 1, ACTION_DIM),
        noise_factory=_noise, max_steps=4,
    )
    assert [q["noise_sha256"] for q in rerun["queries"]] == [q["noise_sha256"] for q in result["queries"]]


def test_explicit_mapping_is_named_and_does_not_infer_14d_equivalence() -> None:
    source = np.arange(CHUNK_SIZE * ACTION_DIM, dtype=np.float32).reshape(CHUNK_SIZE, ACTION_DIM)
    mapped, contract = explicit_named_map_online_chunk(source)
    source_index = {name: index for index, name in enumerate(ALOHA_FEATURE_NAMES)}
    for index, slot in enumerate(PIPER_ACTION_SLOTS):
        assert np.array_equal(mapped[:, index], source[:, source_index[slot.source_feature_name]])
    assert contract["identity_inferred_from_equal_dimension"] is False
    assert contract["kinematic_equivalence_claimed"] is False
    assert len(contract["mapping"]) == ACTION_DIM


@pytest.mark.parametrize("bad", [np.nan, np.inf, 999.0])
def test_invalid_action_never_reaches_env_step(bad: float) -> None:
    capture = _Capture()
    policy = _Policy(capture)
    original = policy.predict_action_chunk

    def predict(batch, *, noise):
        chunk = original(batch, noise=noise)
        chunk[0, 0, 0] = bad
        return chunk

    policy.predict_action_chunk = predict
    env = _Env()
    with pytest.raises(Exception, match="non-finite|outside"):
        run_online_actor_loop(
            env=env, policy=policy, preprocessor=_preprocessor,
            postprocessor=lambda value: value, capture=capture,
            observation=_observation(0), bounds=_bounds(), device=torch.device("cpu"),
            action_converter=lambda value: value.reshape(1, 1, ACTION_DIM),
            noise_factory=_noise, max_steps=1,
        )
    assert env.actions == []


def test_converter_mutation_never_reaches_env_step() -> None:
    capture = _Capture()
    env = _Env()
    with pytest.raises(DirectActorError, match="converter changed"):
        run_online_actor_loop(
            env=env, policy=_Policy(capture), preprocessor=_preprocessor,
            postprocessor=lambda value: value, capture=capture,
            observation=_observation(0), bounds=_bounds(), device=torch.device("cpu"),
            action_converter=lambda value: (value + 0.01).reshape(1, 1, ACTION_DIM),
            noise_factory=_noise, max_steps=1,
        )
    assert env.actions == []


def test_instruction_change_blocks_the_next_step() -> None:
    class ChangedInstructionEnv(_Env):
        def step(self, action, *, auto_reset: bool):
            observation, reward, terminated, truncated, info = super().step(
                action, auto_reset=auto_reset
            )
            observation["task_descriptions"] = ["different instruction"]
            return observation, reward, terminated, truncated, info

    capture = _Capture()
    env = ChangedInstructionEnv()
    with pytest.raises(DirectActorError, match="explicit instruction"):
        run_online_actor_loop(
            env=env,
            policy=_Policy(capture),
            preprocessor=_preprocessor,
            postprocessor=lambda value: value,
            capture=capture,
            observation=_observation(0),
            bounds=_bounds(),
            device=torch.device("cpu"),
            action_converter=lambda value: value.reshape(1, 1, ACTION_DIM),
            noise_factory=_noise,
            max_steps=2,
        )
    assert len(env.actions) == 1


def test_loaded_policy_contract_exercises_state14_and_action14() -> None:
    config = SimpleNamespace(
        robot_state_feature=SimpleNamespace(shape=(6,)),
        action_feature=SimpleNamespace(shape=(ACTION_DIM,)),
        max_state_dim=32,
        max_action_dim=32,
        chunk_size=CHUNK_SIZE,
        image_features={key: object() for key in EXPECTED_IMAGE_KEYS},
    )

    def preprocess(raw):
        result = dict(raw)
        result["observation.state"] = raw["observation.state"].reshape(1, ACTION_DIM)
        return result

    result = validate_loaded_policy_contract(
        config=config,
        preprocessor=preprocess,
        postprocessor=lambda value: value,
        torch_module=torch,
        device=torch.device("cpu"),
    )
    assert result["runtime_preprocessed_state_shape"] == [1, ACTION_DIM]
    assert result["runtime_postprocessed_action_shape"] == [1, CHUNK_SIZE, ACTION_DIM]
    config.action_feature = SimpleNamespace(shape=(6,))
    with pytest.raises(DirectActorError, match="6/14/14"):
        validate_loaded_policy_contract(
            config=config,
            preprocessor=preprocess,
            postprocessor=lambda value: value,
            torch_module=torch,
            device=torch.device("cpu"),
        )


def test_runtime_module_origins_are_exact_and_rehashed(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    module = SimpleNamespace(__file__=str(source))
    expected = {"runtime": {"path": str(source.resolve()), "sha256": file_sha256(source)}}
    assert validate_runtime_module_origins({"runtime": module}, expected) == expected
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(DirectActorError, match="origin changed"):
        validate_runtime_module_origins({"runtime": module}, expected)


def test_runtime_offload_releases_capture_env_policy_and_cache() -> None:
    calls: list[object] = []
    capture = SimpleNamespace(close=lambda: calls.append("capture"))
    env = SimpleNamespace(offload=lambda *, clear_cache: calls.append(("env", clear_cache)))
    policy = SimpleNamespace(to=lambda device: calls.append(("policy", device)))
    cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: calls.append("cache"))
    offload_runtime(capture, env, policy, SimpleNamespace(cuda=cuda))
    assert calls == ["capture", ("env", True), ("policy", "cpu"), "cache"]


def _r6d_fixture(tmp_path: Path):
    directory = tmp_path / "r6d_fixture"
    source = directory / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n", encoding="utf-8")
    prereg_path = directory / "simulation_preregistration.json"
    receipt_path = directory / "simulation_receipt.json"
    _write_json(prereg_path, {"preregistration_sha256": R6D_PREREGISTRATION_LOGICAL_SHA256})
    receipt = {
        "status": "completed_simulation_interface_smoke",
        "real_robot_execution": False,
        "fresh_inputs_used": False,
        "fresh_seed_manifest_opened": False,
        "fresh_trajectory_or_label_opened": False,
        "policy_forward_performed": False,
        "task_success_claimed": False,
        "transfer_claim_authorized": False,
        "performance_evaluation_authorized": False,
        "preregistration": {
            "file_sha256": file_sha256(prereg_path),
            "logical_sha256": R6D_PREREGISTRATION_LOGICAL_SHA256,
        },
        "implementation_sha256": "executor-fixture",
        "environment_contract": {
            "embodiment": ["piper", "piper", 0.6],
            "center_crop": False,
            "collect_wrist_camera": True,
            "action_horizon_per_env_step": 1,
            "observation_state_is_measured_qpos": False,
        },
        "execution": {"all_steps_prevalidated": True, "silent_clipping_possible": False},
        "runtime_source_artifacts": {"fixture": {"path": str(source), "sha256": file_sha256(source)}},
    }
    _write_json(receipt_path, receipt)
    return prereg_path, receipt_path, source


def test_r6d_binding_is_evidence_only_and_rehashes_sources(tmp_path: Path) -> None:
    prereg, receipt, source = _r6d_fixture(tmp_path)
    result = bind_r6d_simulation_receipt(
        prereg, receipt,
        expected_preregistration_sha256=file_sha256(prereg),
        expected_receipt_sha256=file_sha256(receipt),
        expected_directory_name="r6d_fixture",
        expected_executor_sha256="executor-fixture",
        expected_runtime_source_roles=frozenset({"fixture"}),
    )
    assert result["authorization"] == "evidence_only_not_direct_actor_execution_authority"
    source.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(DirectActorError, match="runtime source changed"):
        bind_r6d_simulation_receipt(
            prereg, receipt,
            expected_preregistration_sha256=file_sha256(prereg),
            expected_receipt_sha256=file_sha256(receipt),
            expected_directory_name="r6d_fixture",
            expected_executor_sha256="executor-fixture",
            expected_runtime_source_roles=frozenset({"fixture"}),
        )


def test_prereg_tamper_capability_or_bundle_change_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "model"
    metadata = tmp_path / "metadata"
    model.mkdir()
    metadata.mkdir()
    (model / "weights").write_bytes(b"model")
    (metadata / "config.json").write_text("{}", encoding="utf-8")
    other_roots = {}
    for name in ("rlinf_root", "robotwin_root", "robotwin_code", "lerobot_root"):
        root = tmp_path / name
        root.mkdir()
        other_roots[name] = str(root)
    runner = ROOT / "scripts" / "run_smolvla_piper_r6d_direct_actor_smoke.py"
    value = {
        "format": "smolvla_piper_r6d_direct_actor_preregistration_v1",
        "status": "preregistered_direct_actor_simulation_only_not_executed",
        "actor_id": "smolvla_robotwin_aloha-trained__piper-zero-shot",
        "explicit_instruction": INSTRUCTION,
        "source_body": "aloha",
        "target_body": "piper",
        "r6c_binding": {
            "manifest_path": "/bound/r6c/preflight_manifest.json",
            "manifest_sha256": "1" * 64,
            "receipt_path": "/bound/r6c/preflight_receipt.json",
            "receipt_sha256": "2" * 64,
            "verifier_sha256": "3" * 64,
        },
        "r6d_binding": {
            "preregistration_path": "/bound/r6d/simulation_preregistration.json",
            "preregistration_sha256": "4" * 64,
            "preregistration_logical_sha256": "5" * 64,
            "receipt_path": "/bound/r6d/simulation_receipt.json",
            "receipt_sha256": "6" * 64,
            "executor_sha256": "7" * 64,
            "runtime_source_artifacts": {},
            "authorization": "evidence_only_not_direct_actor_execution_authority",
        },
        "development_seed": {"fresh_confirmation_eligible": False},
        "runtime_roots": {
            **other_roots,
            "model_path": str(model),
            "vlm_metadata_path": str(metadata),
        },
        "runtime_source_artifacts": {"runner": {"path": str(runner), "sha256": file_sha256(runner)}},
        "model_bundle_sha256": directory_bundle_sha256(model),
        "vlm_metadata_bundle_sha256": directory_bundle_sha256(metadata),
        "output": str(tmp_path / "receipt.json"),
        "execution_contract": {
            "max_steps": 4,
            "action_exec_steps": 1,
            "candidate_index": 0,
            "online_policy_forward_each_query": True,
            "precomputed_chunks_forbidden": True,
            "proper_checkpoint_preprocessor": True,
            "proper_checkpoint_postprocessor": True,
            "shared_prefix_dim": 960,
            "processed_state_dim": 14,
            "embodiment": ["piper", "piper", 0.6],
            "center_crop": False,
            "collect_wrist_camera": True,
            "checkpoint_declared_state_dim": 6,
            "runtime_normalizer_state_dim": 14,
            "checkpoint_action_dim": 14,
            "state_dimension_conflict_retained": True,
            "materialized_regular_file_bundles_only": True,
        },
        "capability_contract": {
            "simulation_execution_authorized": True,
            "real_robot_execution_authorized": False,
            "fresh_inputs_allowed": False,
            "fresh_trajectory_or_label_opened": False,
            "performance_evaluation_authorized": False,
            "task_success_claim_authorized": False,
            "transfer_claim_authorized": False,
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
            "scene_seed_and_instruction_strictly_bound": False,
            "instruction_guarantee": "explicit string verified on every current observation in one run",
            "reported_duration": "policy row count not physical time",
            "performance_or_transfer_claim": False,
        },
    }
    value["preregistration_sha256"] = canonical_sha256(value)
    path = tmp_path / "prereg.json"
    _write_json(path, value)
    validate_direct_actor_preregistration(path)
    changed = copy.deepcopy(value)
    changed["capability_contract"]["performance_evaluation_authorized"] = True
    changed["preregistration_sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "preregistration_sha256"})
    _write_json(path, changed)
    with pytest.raises(DirectActorError, match="capability"):
        validate_direct_actor_preregistration(path)
    _write_json(path, value)
    (metadata / "config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(DirectActorError, match="metadata bundle changed"):
        validate_direct_actor_preregistration(path)


def test_directory_bundle_forbids_fresh_paths(tmp_path: Path) -> None:
    directory = tmp_path / "Fresh_labels"
    directory.mkdir()
    (directory / "x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Fresh"):
        directory_bundle_sha256(directory)
