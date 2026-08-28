from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_evaluation400_external_executor_v4 as executor  # noqa: E402
import run_smolvla_piper_evaluation400_condition_v4 as runner  # noqa: E402
import smolvla_piper_evaluation400_audit_contract_v1 as audit  # noqa: E402


CORE_SHA = hashlib.sha256(b"evaluation400-v4-core").hexdigest()
PARENT_SHA = hashlib.sha256(b"evaluation400-v3-core").hexdigest()
RUNTIME_SHA = hashlib.sha256(b"runtime-200-step").hexdigest()
SNAPSHOT_SHA = hashlib.sha256(b"shared-reset-snapshot").hexdigest()


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


OBSERVER_AUTHORITY = runner.build_causal_observer_authority(
    observer_core_file_sha256=sha("observer-core-file"),
    observer_checkpoint_file_sha256=sha("observer-core-checkpoint"),
    training_contract_sha256=sha("observer-training-contract"),
    actor_adapter_set_sha256=sha("observer-adapter-set"),
    actor_adapter_checkpoint_set_sha256=sha("observer-adapter-checkpoint-set"),
    calibration_sha256=sha("observer-calibration"),
    deployment_sha256=sha("observer-deployment"),
    promotion_evidence_sha256=sha("observer-promotion-evidence"),
)


def root_authority() -> dict[str, Any]:
    return {
        "five_member_checkpoint_file_sha256": [
            sha(f"checkpoint-{index}") for index in range(5)
        ],
        "source_rank_score_contract_sha256": [
            sha(f"rank-contract-{index}") for index in range(5)
        ],
        "source_rank_member_authority_sha256": sha("member-authority"),
        "source_rank_numeric_contract": audit.SOURCE_RANK_NUMERIC_CONTRACT,
        "calibration_sha256": sha("calibration"),
        "ensemble_manifest_sha256": sha("ensemble"),
        "deployment_uncertainty_contract_sha256": sha("uncertainty-contract"),
        "deployment_uncertainty_implementation_file_sha256": sha(
            "uncertainty-implementation"
        ),
        "canonical_event_spec_file_sha256": sha("event-spec"),
        "schema6_runtime_contract_sha256": RUNTIME_SHA,
    }


def root_predictions() -> dict[str, np.ndarray]:
    return {
        "post_event_logits": np.zeros((5, 3, 4), dtype=np.float32),
        "next_event_logits": np.ones((5, 3, 4), dtype=np.float32),
        "duration_log_mean": np.full((5, 3), 0.5, dtype=np.float32),
        "duration_log_scale": np.full((5, 3), -1.0, dtype=np.float32),
        "success_logit": np.linspace(-1, 1, 15, dtype=np.float32).reshape(5, 3),
        "object_mean": np.zeros((5, 3, 3), dtype=np.float32),
        "object_log_scale": np.full((5, 3, 3), -2.0, dtype=np.float32),
    }


TENSOR_COMMITMENT = audit.build_root_tensor_commitment(root_predictions())


def make_pair(ordinal: int) -> dict[str, Any]:
    return executor.build_pair_spec(
        pair_id=sha(f"pair-{ordinal}"),
        pair_ordinal=ordinal,
        shared_snapshot_sha256=SNAPSHOT_SHA,
    )


def prepare_root(
    pair: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordered = [sha(f"candidate-{index}") for index in range(4)]
    legal = [False, True, True, True]
    registry_sha = audit.canonical_sha256(
        {
            "pair_id": pair["pair_id"],
            "candidate_count": 4,
            "ordered_candidate_sha256": ordered,
            "candidate_legal": legal,
        }
    )
    root = audit.build_root_precommit(
        protocol_core_v4_sha256=CORE_SHA,
        parent_v3_core_sha256=PARENT_SHA,
        pair_id=pair["pair_id"],
        pair_ordinal=pair["pair_ordinal"],
        shared_snapshot_sha256=pair["shared_snapshot_sha256"],
        ordered_candidate_sha256=ordered,
        candidate_legal=legal,
        candidate_registry_sha256=registry_sha,
        prediction_candidate_indices=[1, 2, 3],
        tensor_commitment=TENSOR_COMMITMENT,
        authority=root_authority(),
    )
    decision = runner.build_root_decision_input(
        root_precommit=root,
        fallback_candidate_index=1,
        mean_success_probability=[0.4, 0.8, 0.5],
        mean_composite_rank_score=[0.0, 0.5, 1.0],
        structured_uncertainty=[0.1, 0.2, 0.9],
        success_margin_threshold=0.1,
        composite_margin_threshold=0.2,
        maximum_global_uncertainty=0.6,
        maximum_pair_uncertainty=0.5,
    )
    return root, decision


class MemoryBackend:
    max_steps = 200

    def __init__(self, pair: Mapping[str, Any], condition: str) -> None:
        self.pair = pair
        self.condition = condition
        self.step_calls = 0
        self.reset_calls = 0
        self.recovery_prediction_steps: list[int] = []

    def reset(self, pair_id: str) -> tuple[dict[str, int], dict[str, str]]:
        self.reset_calls += 1
        if pair_id != self.pair["pair_id"]:
            raise AssertionError("test backend received the wrong pair")
        return {"step": 0}, {
            "pair_id": pair_id,
            "shared_snapshot_sha256": self.pair["shared_snapshot_sha256"],
        }

    def query(self, observation: Mapping[str, int], step_index: int) -> dict[str, Any]:
        assert observation["step"] == step_index
        observer_receipt = runner.build_causal_observer_receipt(
            authority_sha256=OBSERVER_AUTHORITY["authority_sha256"],
            pair_id=self.pair["pair_id"],
            condition_id=self.condition,
            step_index=step_index,
            input_receipt_sha256=sha(
                f"observer-input-{self.pair['pair_id']}-{self.condition}-{step_index}"
            ),
            current_event_id=0 if step_index == 0 else 1,
            current_predicates={name: False for name in runner.PREDICATE_NAMES},
            confidence=0.9,
        )
        return {
            "ordered_candidate_sha256": [
                sha(f"candidate-{index}") for index in range(4)
            ],
            "candidate_legal": [False, True, True, True],
            "lowest_legal_original_candidate_index": 1,
            "mapped_actions": ["illegal", "baseline", "candidate-2", "candidate-3"],
            "pre_action_snapshot_sha256": sha(
                f"{self.pair['pair_id']}-{self.condition}-step-{step_index}"
            ),
            "causal_observer_receipt": observer_receipt,
        }

    def recovery_prediction(
        self, query: Mapping[str, Any], chosen_candidate_index: int,
    ) -> dict[str, Any]:
        assert chosen_candidate_index == 1
        self.recovery_prediction_steps.append(self.step_calls)
        return {
            "member_recovery_logits": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "conditional_recovery_temperature": 0.7,
            "calibration_sha256": sha("calibration"),
            "source_rank_member_authority_sha256": sha("member-authority"),
        }

    def step(self, action: Any) -> tuple[dict[str, int], bool, bool, dict[str, bool]]:
        self.step_calls += 1
        done = self.step_calls == 2
        success = done and self.condition != "baseline"
        return {"step": self.step_calls}, done, False, {"success": success}

    def target_trace(self) -> dict[str, Any]:
        success = self.condition != "baseline"
        return {
            "event_names": ["e0", "e1"],
            "event_steps": [0, 1],
            "terminal_step": 2,
            "terminal_success": success,
            "trajectory_success": [False, False, success],
            "right_censored": False,
            "object_trace": [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
        }


def dense_targets(
    event_names: list[str], event_steps: list[int], terminal_step: int,
) -> dict[str, np.ndarray]:
    assert event_names == ["e0", "e1"]
    assert event_steps == [0, 1]
    assert terminal_step == 2
    return {
        "trajectory_event_id": np.asarray([0, 1, 1], dtype=np.int64),
        "transition_next_event_id": np.asarray([1, 1], dtype=np.int64),
        "transition_duration_decision_steps": np.asarray([1.0, 1.0]),
        "transition_duration_observed": np.asarray([True, False], dtype=bool),
        "transition_duration_censored": np.asarray([False, True], dtype=bool),
    }


def recovery_targets(
    events: np.ndarray, *, right_censored: bool,
) -> dict[str, np.ndarray]:
    assert len(events) == 3
    assert right_censored is False
    return {
        "regress": np.asarray([False, False], dtype=bool),
        "recovery": np.asarray([0, 0], dtype=np.int64),
        "recovery_observed": np.asarray([False, False], dtype=bool),
    }


def object_target(
    object_trace: Any, *, start_step: int, end_step: int,
) -> dict[str, Any]:
    assert start_step == 0 and end_step == 1
    assert len(object_trace) == 2
    return {
        "object_target": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "object_observed": True,
    }


def single_condition_material(
    condition: str = "etsf",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pair = make_pair(0)
    root, decision = prepare_root(pair)
    root_ack = runner.build_root_broker_ack(
        root, ledger_event_sha256=sha("root-ledger-event"), ledger_event_index=1
    )
    request = runner.build_condition_request(
        protocol_core_v4_sha256=CORE_SHA,
        pair_id=pair["pair_id"],
        pair_ordinal=0,
        condition_id=condition,
        shared_snapshot_sha256=pair["shared_snapshot_sha256"],
        root_prediction_commit_sha256=root["commit_sha256"],
        root_ack_sha256=root_ack["ack_sha256"],
        schema6_runtime_contract_sha256=RUNTIME_SHA,
        causal_observer_authority_sha256=OBSERVER_AUTHORITY["authority_sha256"],
    )
    return pair, root, decision, root_ack | {"_request": request}


def test_four_condition_root_algebra_is_exact() -> None:
    _pair, root, decision, _ack = single_condition_material()
    selected = {
        condition: runner.select_condition_root(
            condition, decision, root_precommit=root
        )[0]
        for condition in runner.CONDITION_NAMES
    }
    assert selected == {
        "baseline": 1,
        "success_only_guarded": 2,
        "composite_rank_ungated": 3,
        "etsf": 1,
    }


def test_root_ack_fails_before_backend_reset() -> None:
    pair, root, decision, material = single_condition_material()
    backend = MemoryBackend(pair, "etsf")
    with pytest.raises(runner.ConditionRunnerV4Error):
        runner.execute_condition_v4(
            request=material["_request"],
            backend=backend,
            root_precommit=root,
            root_ack={},
            decision_input=decision,
            recovery_broker=lambda value: {},
            evaluator_public_key_raw=b"x" * 32,
            dense_event_targets_fn=dense_targets,
            recovery_targets_fn=recovery_targets,
            object_target_fn=object_target,
            causal_observer_authority=OBSERVER_AUTHORITY,
        )
    assert backend.reset_calls == 0


def test_condition_runner_commits_recovery_before_continuation_and_seals_target() -> None:
    private_key, public_key = audit.generate_x25519_keypair()
    pair, root, decision, material = single_condition_material()
    request = material.pop("_request")
    backend = MemoryBackend(pair, "etsf")
    committed_steps: list[int] = []

    def broker(commitment: Mapping[str, Any]) -> dict[str, Any]:
        audit.validate_recovery_pre_step_commitment(commitment)
        committed_steps.append(commitment["step_index"])
        return audit.build_broker_ack(
            commitment, ledger_event_sha256=sha("recovery-ledger-step-1")
        )

    result = runner.execute_condition_v4(
        request=request,
        backend=backend,
        root_precommit=root,
        root_ack=material,
        decision_input=decision,
        recovery_broker=broker,
        evaluator_public_key_raw=public_key,
        dense_event_targets_fn=dense_targets,
        recovery_targets_fn=recovery_targets,
        object_target_fn=object_target,
        causal_observer_authority=OBSERVER_AUTHORITY,
    )
    runner.validate_condition_result(
        result,
        request=request,
        root_precommit=root,
        root_ack=material,
        decision_input=decision,
    )
    assert committed_steps == [1]
    assert backend.recovery_prediction_steps == [1]
    assert backend.step_calls == 2
    assert "task_success" not in result
    assert all("success" not in row for row in result["steps"])
    assert not {"target_sha256", "plaintext_sha256"} & set(
        result["target_envelope"]
    )
    incomplete = audit.build_terminal_completeness(
        terminal_receipt_sha256=sha("terminal"), complete_condition_count=1599
    )
    with pytest.raises(audit.AuditContractError):
        audit.open_target_envelope(
            result["target_envelope"],
            evaluator_private_key_raw=private_key,
            terminal_completeness=incomplete,
        )


def test_wrong_recovery_ack_step_stops_before_continuation_action() -> None:
    _private_key, public_key = audit.generate_x25519_keypair()
    pair, root, decision, material = single_condition_material()
    request = material.pop("_request")
    backend = MemoryBackend(pair, "etsf")

    def wrong_step_ack(commitment: Mapping[str, Any]) -> dict[str, Any]:
        ack = audit.build_broker_ack(
            commitment, ledger_event_sha256=sha("wrong-step-event")
        )
        base = {key: value for key, value in ack.items() if key != "ack_sha256"}
        base["step_index"] = commitment["step_index"] + 1
        return {**base, "ack_sha256": audit.canonical_sha256(base)}

    with pytest.raises(audit.AuditContractError, match="wrong step|binding changed"):
        runner.execute_condition_v4(
            request=request,
            backend=backend,
            root_precommit=root,
            root_ack=material,
            decision_input=decision,
            recovery_broker=wrong_step_ack,
            evaluator_public_key_raw=public_key,
            dense_event_targets_fn=dense_targets,
            recovery_targets_fn=recovery_targets,
            object_target_fn=object_target,
            causal_observer_authority=OBSERVER_AUTHORITY,
        )
    assert backend.step_calls == 1


def test_observer_without_promoted_frozen_authority_fails_before_reset() -> None:
    _private_key, public_key = audit.generate_x25519_keypair()
    pair, root, decision, material = single_condition_material()
    request = material.pop("_request")
    backend = MemoryBackend(pair, "etsf")
    monitor_only = copy.deepcopy(OBSERVER_AUTHORITY)
    monitor_only["status"] = "monitor_only_not_integrated"
    monitor_only["promotion_enabled"] = False
    monitor_only["rerank_enabled"] = False
    base = {
        key: value for key, value in monitor_only.items()
        if key != "authority_sha256"
    }
    monitor_only["authority_sha256"] = audit.canonical_sha256(base)
    with pytest.raises(
        runner.ConditionRunnerV4Error, match="not production-rerank authorized"
    ):
        runner.execute_condition_v4(
            request=request,
            backend=backend,
            root_precommit=root,
            root_ack=material,
            decision_input=decision,
            recovery_broker=lambda value: {},
            evaluator_public_key_raw=public_key,
            dense_event_targets_fn=dense_targets,
            recovery_targets_fn=recovery_targets,
            object_target_fn=object_target,
            causal_observer_authority=monitor_only,
        )
    assert backend.reset_calls == 0


def test_worm_ledger_replay_chain_and_identity_fail_closed() -> None:
    ledger = executor.WormLedgerV4(CORE_SHA)
    first = ledger.append("lane_started", artifact_sha256=sha("lane"))
    with pytest.raises(executor.ExecutorV4Error, match="replayed"):
        ledger.append("lane_terminal", artifact_sha256=sha("lane"))
    with pytest.raises(executor.ExecutorV4Error, match="position"):
        ledger.append(
            "condition_started",
            pair_id=sha("pair"),
            pair_ordinal=0,
            condition_id="etsf",
            condition_position=0,
            artifact_sha256=sha("condition"),
        )
    tampered = copy.deepcopy(first)
    tampered["previous_event_sha256"] = sha("wrong-previous")
    base = {key: value for key, value in tampered.items() if key != "event_sha256"}
    tampered["event_sha256"] = audit.canonical_sha256(base)
    with pytest.raises(executor.ExecutorV4Error, match="chain"):
        executor.validate_ledger_event(
            tampered,
            expected_index=0,
            expected_previous_event_sha256=audit.ZERO_SHA256,
        )


def test_exact_400_by_four_end_to_end_and_decryption_gate() -> None:
    private_key, public_key = audit.generate_x25519_keypair()
    pairs = [make_pair(index) for index in range(audit.PAIR_COUNT)]
    lane = executor.Evaluation400ExecutorV4(
        protocol_core_v4_sha256=CORE_SHA,
        schema6_runtime_contract_sha256=RUNTIME_SHA,
        evaluator_public_key_raw=public_key,
        root_preparer=prepare_root,
        backend_factory=lambda pair, condition: MemoryBackend(pair, condition),
        dense_event_targets_fn=dense_targets,
        recovery_targets_fn=recovery_targets,
        object_target_fn=object_target,
        causal_observer_authority=OBSERVER_AUTHORITY,
    )
    terminal, completeness = lane.execute_all(pairs)
    assert terminal["complete_pair_count"] == 400
    assert terminal["complete_condition_count"] == 1600
    assert terminal["root_precommit_count"] == 400
    assert terminal["recovery_pre_step_commit_count"] == 1600
    assert len(lane.condition_results) == len(lane.target_envelopes) == 1600
    assert all("task_success" not in result for result in lane.condition_results)
    with pytest.raises(executor.ExecutorV4Error, match="exact 1600"):
        executor.decrypt_complete_target_envelopes(
            terminal=terminal,
            completeness=completeness,
            target_envelopes=lane.target_envelopes[:-1],
            evaluator_private_key_raw=private_key,
        )
    decoded = executor.decrypt_complete_target_envelopes(
        terminal=terminal,
        completeness=completeness,
        target_envelopes=lane.target_envelopes,
        evaluator_private_key_raw=private_key,
    )
    assert len(decoded) == 1600
    assert decoded[0]["root"]["success"] == 0
    assert decoded[1]["root"]["success"] == 1
    assert lane.ledger.validate() == terminal["ledger_final_event_sha256"]


def test_executor_rejects_duplicate_pair_identity_before_lane_start() -> None:
    _private_key, public_key = audit.generate_x25519_keypair()
    pairs = [make_pair(index) for index in range(audit.PAIR_COUNT)]
    duplicate = copy.deepcopy(pairs[0])
    duplicate["pair_ordinal"] = 1
    base = {
        key: value for key, value in duplicate.items()
        if key != "pair_identity_sha256"
    }
    duplicate["pair_identity_sha256"] = audit.canonical_sha256(base)
    pairs[1] = duplicate
    lane = executor.Evaluation400ExecutorV4(
        protocol_core_v4_sha256=CORE_SHA,
        schema6_runtime_contract_sha256=RUNTIME_SHA,
        evaluator_public_key_raw=public_key,
        root_preparer=prepare_root,
        backend_factory=lambda pair, condition: MemoryBackend(pair, condition),
        dense_event_targets_fn=dense_targets,
        recovery_targets_fn=recovery_targets,
        object_target_fn=object_target,
        causal_observer_authority=OBSERVER_AUTHORITY,
    )
    with pytest.raises(executor.ExecutorV4Error, match="duplicates"):
        lane.execute_all(pairs)
    assert lane.ledger.events == []
