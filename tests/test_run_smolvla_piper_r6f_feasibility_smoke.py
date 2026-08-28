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
    ACTOR_ID,
    CHUNK_SIZE,
    EXPECTED_IMAGE_KEYS,
    INSTRUCTION,
    PREFIX_DIM,
    canonical_sha256,
)
from run_smolvla_piper_r6f_feasibility_smoke import (  # noqa: E402
    FeasibilitySmokeError,
    R6E_EXPECTED_UNSAFE_TARGET,
    assess_candidate_first_action,
    bind_r6e_preregistration,
    build_feasibility_preregistration,
    explicit_named_map_first_action,
    run_feasibility_loop,
    validate_feasibility_preregistration,
)
from verify_smolvla_piper_zero_shot_preflight import (  # noqa: E402
    ALOHA_FEATURE_NAMES,
    PIPER_ACTION_SLOTS,
    file_sha256,
)


def _bounds() -> list[list[float]]:
    return [[slot.lower, slot.upper] for slot in PIPER_ACTION_SLOTS]


def _observation(value: float = 0.0) -> dict[str, object]:
    state = np.zeros((1, ACTION_DIM), dtype=np.float32)
    state[:, [6, 13]] = 0.5
    state[:, 0] = value
    main = np.full((1, 7, 8, 3), int(value) % 255, dtype=np.uint8)
    wrists = np.stack([main, main + 1], axis=1)
    return {
        "states": torch.from_numpy(state),
        "main_images": torch.from_numpy(main),
        "wrist_images": torch.from_numpy(wrists),
        "task_descriptions": [INSTRUCTION],
    }


class _Capture:
    def __init__(self, *, mismatch_candidate: int | None = None) -> None:
        self.value: torch.Tensor | None = None
        self.mismatch_candidate = mismatch_candidate

    def reset(self) -> None:
        self.value = None

    def consume(self) -> torch.Tensor:
        assert self.value is not None
        return self.value


class _Policy:
    def __init__(self, capture: _Capture, first_rows: list[np.ndarray]) -> None:
        self.capture = capture
        self.first_rows = first_rows
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=32,
            image_features={key: object() for key in EXPECTED_IMAGE_KEYS},
        )
        self.forwards: list[tuple[float, int]] = []

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch, *, noise):
        candidate = int(float(noise.reshape(-1)[0]))
        state = float(batch["observation.state"][0, 0])
        self.forwards.append((state, candidate))
        prefix_value = state
        if candidate == self.capture.mismatch_candidate:
            prefix_value += 1.0
        self.capture.value = torch.full((PREFIX_DIM,), prefix_value)
        chunk = torch.zeros((1, CHUNK_SIZE, ACTION_DIM), dtype=torch.float32)
        chunk[:, :, [6, 13]] = 0.5
        chunk[0, 0] = torch.from_numpy(self.first_rows[candidate])
        return chunk


class _Env:
    total_steps = 0

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def step(self, action, *, auto_reset: bool):
        assert auto_reset is False
        type(self).total_steps += 1
        self.actions.append(np.asarray(action).copy())
        return _observation(float(len(self.actions))), 0.0, [False], [False], {"success": [False]}


def _valid(value: float = 0.0) -> np.ndarray:
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[[1, 8]] = max(value, 0.1)
    action[[6, 13]] = 0.5
    return action


def _noise(_config, _seed, _query, candidate, _device):
    return torch.full((1, CHUNK_SIZE, 32), float(candidate))


def _run(rows: list[np.ndarray], *, max_steps: int = 1, mismatch: int | None = None, converter=None):
    calls = {"preprocessor": 0}

    def preprocess(raw):
        calls["preprocessor"] += 1
        result = dict(raw)
        result["observation.state"] = raw["observation.state"].reshape(1, ACTION_DIM)
        return result

    capture = _Capture(mismatch_candidate=mismatch)
    policy = _Policy(capture, rows)
    env = _Env()
    result = run_feasibility_loop(
        env=env,
        policy=policy,
        preprocessor=preprocess,
        postprocessor=lambda value: value,
        capture=capture,
        observation=_observation(),
        bounds=_bounds(),
        device=torch.device("cpu"),
        action_converter=converter or (lambda value: value.reshape(1, 1, ACTION_DIM)),
        noise_factory=_noise,
        max_steps=max_steps,
    )
    return result, env, policy, calls


def test_same_processed_observation_generates_four_and_selects_lowest_feasible() -> None:
    unsafe = _valid()
    unsafe[1] = -0.0043728
    result, env, policy, calls = _run([unsafe, _valid(0.2), _valid(0.3), _valid(0.4)], max_steps=2)
    assert result["steps_executed"] == 2
    assert calls["preprocessor"] == 2
    assert policy.forwards == [
        (0.0, 0), (0.0, 1), (0.0, 2), (0.0, 3),
        (1.0, 0), (1.0, 1), (1.0, 2), (1.0, 3),
    ]
    assert [query["selected_candidate_index"] for query in result["queries"]] == [1, 1]
    rejected = result["queries"][0]["candidate_records"][0]["feasibility"]
    assert rejected["accepted"] is False
    assert rejected["violations"][0]["target_joint_name"] == R6E_EXPECTED_UNSAFE_TARGET
    assert result["queries"][0]["prefix_bit_exact_across_all_four_candidates"] is True
    assert len(set(result["queries"][0]["candidate_prefix_sha256"])) == 1
    assert len(env.actions) == 2 and all(action.shape == (1, 1, ACTION_DIM) for action in env.actions)


def test_candidate0_is_selected_when_it_is_feasible_without_event_scoring() -> None:
    result, env, _, _ = _run([_valid(0.1), _valid(0.2), _valid(0.3), _valid(0.4)])
    assert result["queries"][0]["selected_candidate_index"] == 0
    assert result["event_or_utility_scoring_performed"] is False
    assert np.array_equal(env.actions[0].reshape(ACTION_DIM), _valid(0.1))


def test_all_candidates_invalid_yields_zero_step_fail_closed_result() -> None:
    rows = []
    for index in range(4):
        row = _valid()
        row[1] = -0.1 - index
        rows.append(row)
    result, env, _, calls = _run(rows, max_steps=4)
    assert result["queries_performed"] == 1
    assert result["steps_executed"] == 0
    assert result["no_feasible_candidate_halt"] is True
    assert result["queries"][0]["env_step_performed"] is False
    assert env.actions == []
    assert calls["preprocessor"] == 1


def test_prefix_mismatch_aborts_before_env_step() -> None:
    before = _Env.total_steps
    with pytest.raises(FeasibilitySmokeError, match="shared 960D prefix"):
        _run([_valid(), _valid(), _valid(), _valid()], mismatch=2)
    assert _Env.total_steps == before


def test_fixed_noise_registry_is_reproducible_and_candidate_specific() -> None:
    from run_smolvla_piper_r6f_feasibility_smoke import fixed_candidate_noise

    config = SimpleNamespace(chunk_size=CHUNK_SIZE, max_action_dim=32)
    first = fixed_candidate_noise(config, 100101000, 0, 2, torch.device("cpu"))
    repeated = fixed_candidate_noise(config, 100101000, 0, 2, torch.device("cpu"))
    other = fixed_candidate_noise(config, 100101000, 0, 3, torch.device("cpu"))
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)


def test_nonfinite_rejection_is_json_safe_and_other_candidate_can_run() -> None:
    nonfinite = _valid()
    nonfinite[0] = np.nan
    result, env, _, _ = _run([nonfinite, _valid(), _valid(), _valid()])
    record = result["queries"][0]["candidate_records"][0]
    assert record["first_action"][0] is None
    assert record["feasibility"]["violations"][0]["reason"] == "nonfinite"
    assert result["queries"][0]["selected_candidate_index"] == 1
    json.dumps(result, allow_nan=False)
    assert len(env.actions) == 1


def test_converter_mutation_blocks_env_step() -> None:
    with pytest.raises(FeasibilitySmokeError, match="converter changed"):
        _run(
            [_valid(), _valid(), _valid(), _valid()],
            converter=lambda value: (value + 0.01).reshape(1, 1, ACTION_DIM),
        )


def test_first_action_mapping_is_explicit_named_not_14d_identity_claim() -> None:
    source = np.arange(ACTION_DIM, dtype=np.float32)
    mapped, contract = explicit_named_map_first_action(source)
    source_by_name = {name: index for index, name in enumerate(ALOHA_FEATURE_NAMES)}
    for index, slot in enumerate(PIPER_ACTION_SLOTS):
        assert mapped[index] == source[source_by_name[slot.source_feature_name]]
    assert contract["derived_from_equal_14d_width"] is False
    assert contract["kinematic_equivalence_claimed"] is False


def test_r6e_binding_is_prereg_lineage_only_not_fabricated_receipt(tmp_path: Path) -> None:
    directory = tmp_path / "r6e_fixture"
    path = directory / "direct_actor_preregistration.json"
    directory.mkdir()
    path.write_text("{}", encoding="utf-8")
    runner = ROOT / "scripts" / "run_smolvla_piper_r6d_direct_actor_smoke.py"
    logical = "a" * 64
    value = {
        "preregistration_sha256": logical,
        "runtime_source_artifacts": {
            "direct_actor_runner": {"path": str(runner), "sha256": file_sha256(runner)}
        },
        "execution_contract": {"candidate_index": 0},
    }
    binding = bind_r6e_preregistration(
        path,
        expected_file_sha256=file_sha256(path),
        expected_logical_sha256=logical,
        expected_runner_sha256=file_sha256(runner),
        expected_directory_name="r6e_fixture",
        validator=lambda _: value,
    )
    assert binding["failure_receipt_bound"] is False
    assert binding["authorization"] == "lineage_only_not_R6f_execution_authority"
    assert "expected_external_diagnostic_not_content_authenticated" in binding


def _inherited() -> dict[str, object]:
    return {
        "r6c_binding": {"manifest_sha256": "1" * 64},
        "r6d_binding": {"receipt_sha256": "2" * 64},
        "development_seed": {"fresh_confirmation_eligible": False},
        "runtime_roots": {"root": "/bound"},
        "runtime_source_artifacts": {"source": {"sha256": "3" * 64}},
        "vlm_metadata_bundle_sha256": "4" * 64,
        "model_bundle_sha256": "5" * 64,
        "capability_contract": {"fresh_inputs_allowed": False},
        "mapping_contract": {"derived_from_equal_14d_width": False},
        "state_contract": {"is_measured_qpos": False},
        "caveats": {"performance_or_transfer_claim": False},
    }


def test_prereg_freezes_feasibility_only_contract_and_rejects_tamper(tmp_path: Path) -> None:
    r6e = {
        "path": "/bound/r6e/direct_actor_preregistration.json",
        "file_sha256": "6" * 64,
        "logical_sha256": "7" * 64,
        "runner_sha256": "8" * 64,
        "failure_receipt_bound": False,
        "authorization": "lineage_only_not_R6f_execution_authority",
        "expected_external_diagnostic_not_content_authenticated": {},
    }
    value = build_feasibility_preregistration(
        r6e=r6e,
        r6e_preregistration=_inherited(),
        output=tmp_path / "receipt.json",
    )
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    validated = validate_feasibility_preregistration(path)
    assert validated["execution_contract"]["event_or_utility_scoring_authorized"] is False
    assert validated["capability_contract"]["performance_evaluation_authorized"] is False
    changed = copy.deepcopy(value)
    changed["execution_contract"]["selection_rule"] = "event_score"
    changed["preregistration_sha256"] = canonical_sha256({key: item for key, item in changed.items() if key != "preregistration_sha256"})
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FeasibilitySmokeError, match="selection contract"):
        validate_feasibility_preregistration(path)
