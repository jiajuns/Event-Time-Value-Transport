from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_evaluation400_external_executor_v3 as supervisor  # noqa: E402
import run_smolvla_piper_evaluation400_condition_v3 as runner  # noqa: E402


def signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**dict(base), field: runner.canonical_sha256(base)}


def request(condition: str) -> dict[str, Any]:
    position = 0 if condition == "baseline" else 1
    base = {
        "format": runner.REQUEST_FORMAT,
        "status": "write_ahead_before_condition_popen",
        "plan_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "claim_sha256": "3" * 64,
        "pair_id": "4" * 64,
        "ordinal": 0,
        "requested_seed": 101,
        "resolved_seed": 101,
        "initial_scene_state_sha256": "5" * 64,
        "initial_measured_joint_state_sha256": "6" * 64,
        "initial_commanded_drive_target_sha256": "7" * 64,
        "attempt": 0,
        "pair_identity_sha256": "8" * 64,
        "condition": condition,
        "condition_ordinal": position,
        "condition_order": ["baseline", "etsf"],
        "shared_snapshot_sha256": "9" * 64,
        "candidate_count": 4,
        "candidate_generation_contract_sha256": "a" * 64,
        "postfreeze_identity_or_order_change_authorized": False,
        "outcome_visible_before_condition_start": False,
    }
    return signed(base, "request_sha256")


class FakeBackend:
    max_steps = 200
    schema6_execution_authority_file_sha256 = "c" * 64
    schema6_runtime_contract_sha256 = "d" * 64
    continuation_policy_sha256 = "b" * 64

    def __init__(self) -> None:
        self.selector_calls = 0
        self.steps: list[int] = []

    def reset(self, requested_seed: int):
        assert requested_seed == 101
        return {"observation": 0}, {
            "resolved_seed": 101,
            "initial_scene_state_sha256": "5" * 64,
            "initial_measured_joint_state_sha256": "6" * 64,
            "initial_commanded_drive_target_sha256": "7" * 64,
        }

    def query(self, observation: Any, query_index: int):
        native = [str(index + 1) * 64 for index in range(4)]
        legal = np.asarray(
            [False, True, True, True] if query_index == 0 else [True, True, False, False]
        )
        actions = np.zeros((4, 50, 14), dtype=np.float32)
        for index in range(4):
            actions[index, 0, 0] = index
        return {
            "native_action_sha256": native,
            "feasibility_mask": legal,
            "mapped_actions": actions,
            "lowest_legal_original_candidate_index": int(np.flatnonzero(legal)[0]),
        }

    def select_etsf(self, query: Mapping[str, Any]):
        self.selector_calls += 1
        return 2, {
            "event_model_members_called": 5,
            "uncertainty_gate_applied": True,
            "selected_candidate_index": 2,
            "score_contract": (
                "five_member_adjusted_source_composite_candidate_rank_score_margin"
            ),
            "source_rank_score_contract_sha256": [str(index + 1) * 64 for index in range(5)],
            "source_rank_numeric_contract": runner.SOURCE_RANK_NUMERIC_CONTRACT,
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "formal190_target_outcome_calibrated_acceptance_margin": True,
        }

    def step(self, action: np.ndarray):
        selected = int(action.reshape(-1)[0])
        self.steps.append(selected)
        done = len(self.steps) == 3
        return {"observation": len(self.steps)}, done, False, {"success": done}

    def snapshot(self):
        return {"step": len(self.steps)}

    def close(self):
        pass


def test_baseline_never_calls_event_selector_and_continuation_is_lowest_legal() -> None:
    backend = FakeBackend()
    result, trajectory, continuation = runner.execute_condition(
        request=request("baseline"), backend=backend
    )
    assert backend.selector_calls == 0
    assert backend.steps == [1, 0, 0]
    assert result["selected_candidate_index"] == 1
    assert result["task_success"] is True
    assert json.loads(trajectory)["predicted_success_used_as_outcome"] is False
    assert json.loads(continuation)["all_postroot_steps_lowest_legal"] is True


def test_etsf_calls_exact_five_member_guard_and_only_changes_root() -> None:
    backend = FakeBackend()
    result, trajectory, _continuation = runner.execute_condition(
        request=request("etsf"), backend=backend
    )
    assert backend.selector_calls == 1
    assert backend.steps == [2, 0, 0]
    assert result["selected_candidate_index"] == 2
    proof = json.loads(trajectory)["selector_proof"]
    assert proof["event_model_members_called"] == 5
    assert proof["uncertainty_gate_applied"] is True


def test_two_conditions_share_exact_root_registry_and_snapshot() -> None:
    baseline, _, _ = runner.execute_condition(
        request=request("baseline"), backend=FakeBackend()
    )
    etsf, _, _ = runner.execute_condition(request=request("etsf"), backend=FakeBackend())
    assert baseline["shared_snapshot_sha256"] == etsf["shared_snapshot_sha256"]
    assert baseline["candidate_registry_sha256"] == etsf["candidate_registry_sha256"]
    assert baseline["ordered_candidate_sha256"] == etsf["ordered_candidate_sha256"]
    assert baseline["candidate_legal"] == etsf["candidate_legal"]


def test_publish_result_is_consumable_by_supervisor_validator(tmp_path: Path) -> None:
    req = request("baseline")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(req, sort_keys=True), encoding="utf-8")
    request_path.chmod(0o444)
    backend = FakeBackend()
    result, trajectory, continuation = runner.execute_condition(
        request=req, backend=backend
    )
    root = tmp_path / "result"
    runner.publish_result(
        output_root=root,
        request_path=request_path,
        request=req,
        result=result,
        trajectory=trajectory,
        continuation=continuation,
    )
    audited = supervisor.validate_runner_result(
        root, request_path=request_path, request=req
    )
    assert audited["value"]["task_success"] is True


def test_request_rejects_bool_numeric_and_identity_drift() -> None:
    value = request("baseline")
    value["requested_seed"] = False
    value["request_sha256"] = runner.canonical_sha256(
        {key: child for key, child in value.items() if key != "request_sha256"}
    )
    with pytest.raises(runner.ConditionRunnerError, match="contract changed"):
        runner.validate_request(value)


def test_etsf_fails_if_five_member_or_uncertainty_proof_missing() -> None:
    backend = FakeBackend()
    backend.select_etsf = lambda _query: (2, {  # type: ignore[method-assign]
        "event_model_members_called": 4,
        "uncertainty_gate_applied": True,
    })
    with pytest.raises(runner.ConditionRunnerError, match="five-member"):
        runner.execute_condition(request=request("etsf"), backend=backend)


def test_physical_object_denormalization_and_deployment_scales_are_exact() -> None:
    import torch

    output = {
        "duration_selected_log_mean": torch.tensor([1.0]),
        "duration_selected_log_scale": torch.tensor([0.2]),
        "object_delta_mean": torch.tensor([[2.0, -1.0]]),
        "object_delta_log_scale": torch.tensor([[0.0, 0.5]]),
    }
    transformed = runner.apply_deployment_prediction_scales(
        output,
        object_mean=torch.tensor([0.1, 0.2]),
        object_std=torch.tensor([0.5, 2.0]),
        duration_scale_multiplier=3.0,
        object_scale_multiplier=4.0,
    )
    assert torch.allclose(transformed["object_mean"], torch.tensor([[1.1, -1.8]]))
    assert torch.allclose(
        transformed["object_log_scale"],
        output["object_delta_log_scale"]
        + torch.log(torch.tensor([0.5, 2.0]))
        + runner.math.log(4.0),
    )
    assert torch.allclose(
        transformed["duration_log_scale"],
        output["duration_selected_log_scale"] + runner.math.log(3.0),
    )
    with pytest.raises(runner.ConditionRunnerError, match="duration deployment scale"):
        runner.apply_deployment_prediction_scales(
            output,
            object_mean=torch.tensor([0.1, 0.2]),
            object_std=torch.tensor([0.5, 2.0]),
            duration_scale_multiplier=False,
            object_scale_multiplier=1.0,
        )


def test_preflight_requires_every_real_inference_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Complete:
        pass

    runtime = Complete()
    dense = Complete()
    trainer = Complete()
    for target, names in (
        (
            runtime,
            (
                "validate_runtime_contract", "_load_authority_runtime_contract",
                "_build_resources",
            ),
        ),
        (dense, ("generate_candidate_query", "validate_candidate_query")),
        (
            trainer,
            (
                "_load_torch", "validate_source_checkpoint",
                "_validate_source_rank_score_contract", "object_normalization",
                "array_bundle_sha256", "SmolVLAPiperAdapter",
                "DetachedConditionalRecoveryAdapter", "reconstruct_pose_predicates",
            ),
        ),
    ):
        for name in names:
            setattr(target, name, lambda *args, **kwargs: None)
    modules = iter((runtime, dense, trainer))
    monkeypatch.setattr(runner, "load_bound_module", lambda *_args, **_kwargs: next(modules))
    monkeypatch.setattr(
        runner, "validate_full_horizon_runtime_binding",
        lambda *_args, **_kwargs: {"runtime_contract_sha256": "4" * 64},
    )
    args = type("Args", (), {
        "simulator_implementation": Path("sim.py"),
        "simulator_implementation_file_sha256": "1" * 64,
        "dense_collector_implementation": Path("dense.py"),
        "dense_collector_implementation_file_sha256": "2" * 64,
        "adapter_trainer_implementation": Path("trainer.py"),
        "adapter_trainer_implementation_file_sha256": "3" * 64,
    })()
    assert runner.preflight(args)["simulator_steps"] == 0

    delattr(trainer, "object_normalization")
    modules = iter((runtime, dense, trainer))
    monkeypatch.setattr(runner, "load_bound_module", lambda *_args, **_kwargs: next(modules))
    with pytest.raises(runner.ConditionRunnerError, match="API is incomplete"):
        runner.preflight(args)


def test_full_horizon_runtime_binding_rejects_199_steps(tmp_path: Path) -> None:
    runtime_value = {
        "runtime_contract_sha256": "9" * 64,
        "max_episode_steps": 200,
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(
        json.dumps({"runtime_contract": runtime_value}), encoding="utf-8"
    )
    authority_path.chmod(0o444)
    authority_file_sha = runner.file_sha256(authority_path)

    class Runtime:
        @staticmethod
        def validate_runtime_contract(value):
            assert value == runtime_value
            return dict(value)

    def make_args(max_steps: int):
        interface_base = {
            "format": "etsf_smolvla_piper_condition_runner_runtime_contract_v3",
            "status": "externally_reviewed_condition_runner_interface",
            "interface_version": 3,
            "mode": "execute-condition-v3",
            "request_format": runner.REQUEST_FORMAT,
            "result_format": runner.RESULT_FORMAT,
            "condition_runner_implementation_sha256": "1" * 64,
            "simulator_implementation_sha256": "2" * 64,
            "visible_device_contract": "exact_gpu_uuid_as_cuda_visible_devices_and_cuda0",
            "pair_attempt": 0,
            "candidate_count": 4,
            "condition_names": ["baseline", "etsf"],
            "schema6_execution_authority_file_sha256": authority_file_sha,
            "schema6_runtime_contract_sha256": "9" * 64,
            "max_episode_steps": max_steps,
        }
        interface = signed(interface_base, "runtime_contract_sha256")
        path = tmp_path / f"interface-{max_steps}.json"
        path.write_text(json.dumps(interface), encoding="utf-8")
        path.chmod(0o444)
        return type("Args", (), {
            "runtime_contract": path,
            "runtime_contract_file_sha256": runner.file_sha256(path),
            "condition_runner_source_file_sha256": "1" * 64,
            "simulator_implementation_file_sha256": "2" * 64,
            "schema6_execution_authority": authority_path,
            "schema6_execution_authority_file_sha256": authority_file_sha,
        })()

    accepted = runner.validate_full_horizon_runtime_binding(make_args(200), Runtime())
    assert accepted["max_episode_steps"] == 200
    with pytest.raises(runner.ConditionRunnerError, match="interface changed"):
        runner.validate_full_horizon_runtime_binding(make_args(199), Runtime())
