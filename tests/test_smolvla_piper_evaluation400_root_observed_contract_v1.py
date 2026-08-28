from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smolvla_piper_causal_event_observer_v1 as observer  # noqa: E402
import smolvla_piper_evaluation400_root_observed_contract_v1 as root  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


PAIR_ID = sha("pair")
SNAPSHOT_SHA = sha("shared-snapshot")
PRE_ACTION_SHA = sha("pre-action-snapshot")
OBSERVER_AUTHORITY_SHA = sha("realized-observer-authority")
OBSERVER_ADAPTER_CONTRACT_SHA = sha("piper-adapter-contract")
ROOT_PREDICTOR_AUTHORITY_SHA = sha("root-predictor-authority")
ROOT_CALIBRATION_SHA = sha("root-calibration")
ROOT_RANK_CONTRACT_SET_SHA = sha("rank-contract-set")
ROOT_UNCERTAINTY_CONTRACT_SHA = sha("uncertainty-contract")
ROOT_DERIVATION_IMPLEMENTATION_SHA = sha("derivation-code")
DERIVATION_EXPECTED = {
    "expected_root_predictor_authority_sha256": ROOT_PREDICTOR_AUTHORITY_SHA,
    "expected_calibration_sha256": ROOT_CALIBRATION_SHA,
    "expected_source_rank_contract_set_sha256": ROOT_RANK_CONTRACT_SET_SHA,
    "expected_uncertainty_contract_sha256": ROOT_UNCERTAINTY_CONTRACT_SHA,
    "expected_derivation_implementation_file_sha256": (
        ROOT_DERIVATION_IMPLEMENTATION_SHA
    ),
}
ORDERED = [sha(f"candidate-{index}") for index in range(4)]
LEGAL = [False, True, True, True]
CHRONOLOGY = {
    "root_reset_calls": 1,
    "root_policy_query_calls": 1,
    "root_observer_calls": 1,
    "root_world_model_member_calls": 0,
    "simulator_step_calls": 0,
    "condition_started_count": 0,
    "target_read_calls": 0,
}
DERIVATION_CHRONOLOGY = {
    **CHRONOLOGY,
    "root_world_model_member_calls": 5,
}


def signed(base: dict[str, Any], field: str) -> dict[str, Any]:
    return {**base, field: root.canonical_sha256(base)}


def actor_visible_material() -> tuple[dict[str, Any], dict[str, Any]]:
    history = torch.zeros(8, 960, dtype=torch.float32)
    history[0] = torch.arange(960, dtype=torch.float32) / 960.0
    history_mask = torch.tensor([True] + [False] * 7, dtype=torch.bool)
    proprio = torch.linspace(-1.0, 1.0, 14, dtype=torch.float32)
    input_receipt = observer.make_input_receipt(
        history=history,
        history_mask=history_mask,
        proprio=proprio,
        actor_name="piper",
        policy_family="smolvla",
        state_feature_source_sha256=sha("trusted-hidden-hook"),
        current_query_index=0,
        valid_history_steps=1,
    )
    output_base = {
        "format": "etsf_smolvla_piper_causal_observer_query_receipt_v4",
        "status": "actor_visible_promoted_observation_applicable",
        "authority_sha256": OBSERVER_AUTHORITY_SHA,
        "pair_id": PAIR_ID,
        "condition_id": root.ROOT_SCOPE,
        "step_index": 0,
        "input_receipt_sha256": input_receipt["receipt_sha256"],
        "prediction_sha256": sha("real-observer-probability-tensors"),
        "calibration_sha256": sha("observer-calibration"),
        "minimum_joint_confidence": 0.7,
        "current_event_id": 1,
        "current_predicates": {
            "moved": True,
            "lifted": False,
            "near_goal": False,
            "stationary": False,
            "success": False,
        },
        "confidence": 0.9,
        "applicable": True,
        "object_pose_fields_present": False,
        "simulator_privileged_state_read": False,
        "hardcoded_event_fallback_used": False,
    }
    output_receipt = signed(output_base, "receipt_sha256")
    inputs = {
        "actor_name": "piper",
        "policy_family": "smolvla",
        "state_feature_source_sha256": sha("trusted-hidden-hook"),
        "actor_adapter_contract_sha256": OBSERVER_ADAPTER_CONTRACT_SHA,
        "history": history.numpy().copy(),
        "history_mask": history_mask.numpy().copy(),
        "proprio": proprio.numpy().copy(),
        "image_feature": None,
        "image_feature_extractor_file_sha256": None,
        "observer_input_receipt": input_receipt,
    }
    return inputs, output_receipt


def mapped_actions() -> np.ndarray:
    return np.arange(4 * 3 * 14, dtype=np.float32).reshape(4, 3, 14) / 100.0


def observation_material() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray]:
    inputs, receipt = actor_visible_material()
    actions = mapped_actions()
    commitment = root.build_root_observation_commitment(
        pair_id=PAIR_ID,
        pair_ordinal=0,
        shared_snapshot_sha256=SNAPSHOT_SHA,
        pre_action_snapshot_sha256=PRE_ACTION_SHA,
        observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
        observer_actor_adapter_contract_sha256=OBSERVER_ADAPTER_CONTRACT_SHA,
        observer_output_receipt=receipt,
        actor_visible_inputs=inputs,
        ordered_candidate_sha256=ORDERED,
        candidate_legal=LEGAL,
        lowest_legal_original_candidate_index=1,
        mapped_actions=actions,
        chronology=CHRONOLOGY,
    )
    return commitment, inputs, receipt, actions


def validate_observation(
    commitment: dict[str, Any], inputs: dict[str, Any],
    receipt: dict[str, Any], actions: np.ndarray,
) -> str:
    return root.validate_root_observation_commitment(
        commitment,
        expected_pair_id=PAIR_ID,
        expected_pair_ordinal=0,
        expected_shared_snapshot_sha256=SNAPSHOT_SHA,
        expected_pre_action_snapshot_sha256=PRE_ACTION_SHA,
        expected_observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
        expected_observer_actor_adapter_contract_sha256=(
            OBSERVER_ADAPTER_CONTRACT_SHA
        ),
        observer_output_receipt=receipt,
        actor_visible_inputs=inputs,
        ordered_candidate_sha256=ORDERED,
        candidate_legal=LEGAL,
        lowest_legal_original_candidate_index=1,
        mapped_actions=actions,
    )


def raw_predictions() -> dict[str, np.ndarray]:
    candidates = 3
    return {
        "post_event_logits": np.zeros((5, candidates, 5), dtype=np.float32),
        "next_event_logits": np.ones((5, candidates, 5), dtype=np.float32),
        "duration_log_mean": np.full((5, candidates), 0.5, dtype=np.float32),
        "duration_log_scale": np.full((5, candidates), -1.0, dtype=np.float32),
        "success_logit": np.linspace(-1, 1, 15, dtype=np.float32).reshape(5, candidates),
        "object_mean": np.zeros((5, candidates, 3), dtype=np.float32),
        "object_log_scale": np.full((5, candidates, 3), -2.0, dtype=np.float32),
    }


def auxiliary_tensors() -> dict[str, np.ndarray]:
    return {
        "member_calibrated_success_probability": np.linspace(
            0.1, 0.9, 15, dtype=np.float32
        ).reshape(5, 3),
        "member_composite_rank_score": np.linspace(
            -1.0, 1.0, 15, dtype=np.float32
        ).reshape(5, 3),
        "candidate_structured_five_head_uncertainty": np.asarray(
            [0.1, 0.2, 0.3], dtype=np.float32
        ),
    }


def derivation_material():
    observation, inputs, receipt, actions = observation_material()
    raw = raw_predictions()
    auxiliary = auxiliary_tensors()
    commitment = root.build_root_prediction_derivation_commitment(
        observation_commitment=observation,
        raw_predictions=raw,
        auxiliary_tensors=auxiliary,
        root_predictor_authority_sha256=ROOT_PREDICTOR_AUTHORITY_SHA,
        calibration_sha256=ROOT_CALIBRATION_SHA,
        source_rank_contract_set_sha256=ROOT_RANK_CONTRACT_SET_SHA,
        uncertainty_contract_sha256=ROOT_UNCERTAINTY_CONTRACT_SHA,
        derivation_implementation_file_sha256=ROOT_DERIVATION_IMPLEMENTATION_SHA,
        chronology=DERIVATION_CHRONOLOGY,
    )
    return commitment, observation, raw, auxiliary, inputs, receipt, actions


def resign(value: dict[str, Any], field: str) -> dict[str, Any]:
    changed = copy.deepcopy(value)
    base = {name: child for name, child in changed.items() if name != field}
    changed[field] = root.canonical_sha256(base)
    return changed


def test_valid_root_observation_binds_pair_actor_inputs_candidates_and_actions() -> None:
    commitment, inputs, receipt, actions = observation_material()
    assert validate_observation(commitment, inputs, receipt, actions) == commitment[
        "observation_commit_sha256"
    ]
    assert commitment["chronology"] == CHRONOLOGY
    assert commitment["candidate_commitment"]["mapped_action_set_sha256"]
    assert commitment["actor_visible_input_commitment"][
        "direct_event_or_predicate_input_present"
    ] is False


def test_observer_authority_must_bind_the_exact_actor_adapter_contract() -> None:
    commitment, inputs, receipt, actions = observation_material()
    with pytest.raises(root.RootObservedContractError, match="not bound"):
        root.validate_root_observation_commitment(
            commitment,
            expected_pair_id=PAIR_ID,
            expected_pair_ordinal=0,
            expected_shared_snapshot_sha256=SNAPSHOT_SHA,
            expected_pre_action_snapshot_sha256=PRE_ACTION_SHA,
            expected_observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
            expected_observer_actor_adapter_contract_sha256=sha(
                "different-authority-bound-adapter"
            ),
            observer_output_receipt=receipt,
            actor_visible_inputs=inputs,
            ordered_candidate_sha256=ORDERED,
            candidate_legal=LEGAL,
            lowest_legal_original_candidate_index=1,
            mapped_actions=actions,
        )


@pytest.mark.parametrize("forbidden", ["current_event_id", "object_poses", "target_success"])
def test_direct_semantics_pose_or_target_input_is_rejected(forbidden: str) -> None:
    inputs, receipt = actor_visible_material()
    inputs[forbidden] = 1
    with pytest.raises(root.RootObservedContractError, match="fields changed"):
        root.build_root_observation_commitment(
            pair_id=PAIR_ID, pair_ordinal=0,
            shared_snapshot_sha256=SNAPSHOT_SHA,
            pre_action_snapshot_sha256=PRE_ACTION_SHA,
            observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
            observer_actor_adapter_contract_sha256=(
                OBSERVER_ADAPTER_CONTRACT_SHA
            ),
            observer_output_receipt=receipt,
            actor_visible_inputs=inputs,
            ordered_candidate_sha256=ORDERED,
            candidate_legal=LEGAL,
            lowest_legal_original_candidate_index=1,
            mapped_actions=mapped_actions(), chronology=CHRONOLOGY,
        )


def test_low_confidence_or_nonapplicable_root_observer_fails_closed() -> None:
    inputs, receipt = actor_visible_material()
    receipt["confidence"] = 0.6
    receipt = resign(receipt, "receipt_sha256")
    with pytest.raises(root.RootObservedContractError, match="confidence/provenance"):
        root.build_root_observation_commitment(
            pair_id=PAIR_ID, pair_ordinal=0,
            shared_snapshot_sha256=SNAPSHOT_SHA,
            pre_action_snapshot_sha256=PRE_ACTION_SHA,
            observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
            observer_actor_adapter_contract_sha256=(
                OBSERVER_ADAPTER_CONTRACT_SHA
            ),
            observer_output_receipt=receipt,
            actor_visible_inputs=inputs,
            ordered_candidate_sha256=ORDERED, candidate_legal=LEGAL,
            lowest_legal_original_candidate_index=1,
            mapped_actions=mapped_actions(), chronology=CHRONOLOGY,
        )


def test_root_history_must_be_q0_causal_and_match_observer_input_receipt() -> None:
    commitment, inputs, receipt, actions = observation_material()
    tampered = copy.deepcopy(inputs)
    tampered["history"][1, 0] = np.float32(1.0)
    with pytest.raises(root.RootObservedContractError, match="causal/actor-visible"):
        validate_observation(commitment, tampered, receipt, actions)


def test_optional_actor_visible_image_is_content_addressed() -> None:
    inputs, receipt = actor_visible_material()
    image = torch.linspace(0.0, 1.0, 4, dtype=torch.float32)
    extractor_sha = sha("frozen-image-extractor")
    image_receipt = observer.make_image_receipt(
        image, extractor_file_sha256=extractor_sha, frame_query_index=0
    )
    history = torch.from_numpy(inputs["history"])
    history_mask = torch.from_numpy(inputs["history_mask"])
    proprio = torch.from_numpy(inputs["proprio"])
    input_receipt = observer.make_input_receipt(
        history=history, history_mask=history_mask, proprio=proprio,
        actor_name="piper", policy_family="smolvla",
        state_feature_source_sha256=sha("trusted-hidden-hook"),
        current_query_index=0, valid_history_steps=1,
        image_feature_receipt=image_receipt,
    )
    inputs["image_feature"] = image.numpy().copy()
    inputs["image_feature_extractor_file_sha256"] = extractor_sha
    inputs["observer_input_receipt"] = input_receipt
    receipt["input_receipt_sha256"] = input_receipt["receipt_sha256"]
    receipt = resign(receipt, "receipt_sha256")
    commitment = root.build_root_observation_commitment(
        pair_id=PAIR_ID, pair_ordinal=0,
        shared_snapshot_sha256=SNAPSHOT_SHA,
        pre_action_snapshot_sha256=PRE_ACTION_SHA,
        observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
        observer_actor_adapter_contract_sha256=OBSERVER_ADAPTER_CONTRACT_SHA,
        observer_output_receipt=receipt, actor_visible_inputs=inputs,
        ordered_candidate_sha256=ORDERED, candidate_legal=LEGAL,
        lowest_legal_original_candidate_index=1,
        mapped_actions=mapped_actions(), chronology=CHRONOLOGY,
    )
    assert validate_observation(
        commitment, inputs, receipt, mapped_actions()
    ) == commitment["observation_commit_sha256"]
    tampered = copy.deepcopy(inputs)
    tampered["image_feature"][0] = np.nextafter(
        tampered["image_feature"][0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(
        root.RootObservedContractError,
        match="image receipt|causal/actor-visible",
    ):
        validate_observation(commitment, tampered, receipt, mapped_actions())


def test_pair_candidate_order_and_mapped_action_one_ulp_are_externally_bound() -> None:
    commitment, inputs, receipt, actions = observation_material()
    with pytest.raises(
        root.RootObservedContractError, match="provenance|original arrays"
    ):
        root.validate_root_observation_commitment(
            commitment,
            expected_pair_id=sha("other-pair"), expected_pair_ordinal=0,
            expected_shared_snapshot_sha256=SNAPSHOT_SHA,
            expected_pre_action_snapshot_sha256=PRE_ACTION_SHA,
            expected_observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
            expected_observer_actor_adapter_contract_sha256=(
                OBSERVER_ADAPTER_CONTRACT_SHA
            ),
            observer_output_receipt=receipt, actor_visible_inputs=inputs,
            ordered_candidate_sha256=ORDERED, candidate_legal=LEGAL,
            lowest_legal_original_candidate_index=1, mapped_actions=actions,
        )
    reordered = [ORDERED[0], ORDERED[2], ORDERED[1], ORDERED[3]]
    with pytest.raises(root.RootObservedContractError, match="original arrays"):
        root.validate_root_observation_commitment(
            commitment,
            expected_pair_id=PAIR_ID, expected_pair_ordinal=0,
            expected_shared_snapshot_sha256=SNAPSHOT_SHA,
            expected_pre_action_snapshot_sha256=PRE_ACTION_SHA,
            expected_observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
            expected_observer_actor_adapter_contract_sha256=(
                OBSERVER_ADAPTER_CONTRACT_SHA
            ),
            observer_output_receipt=receipt, actor_visible_inputs=inputs,
            ordered_candidate_sha256=reordered, candidate_legal=LEGAL,
            lowest_legal_original_candidate_index=1, mapped_actions=actions,
        )
    changed_actions = actions.copy()
    changed_actions[1, 0, 0] = np.nextafter(
        changed_actions[1, 0, 0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(root.RootObservedContractError, match="original arrays"):
        validate_observation(commitment, inputs, receipt, changed_actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_reset_calls", 0),
        ("root_policy_query_calls", 2),
        ("simulator_step_calls", 1),
        ("condition_started_count", 1),
        ("target_read_calls", 1),
        ("root_observer_calls", 0),
        ("root_world_model_member_calls", 5),
    ],
)
def test_root_observation_chronology_requires_one_observer_and_zero_step_target(
    field: str, value: int,
) -> None:
    inputs, receipt = actor_visible_material()
    chronology = {**CHRONOLOGY, field: value}
    with pytest.raises(root.RootObservedContractError, match="chronology"):
        root.build_root_observation_commitment(
            pair_id=PAIR_ID, pair_ordinal=0,
            shared_snapshot_sha256=SNAPSHOT_SHA,
            pre_action_snapshot_sha256=PRE_ACTION_SHA,
            observer_authority_sha256=OBSERVER_AUTHORITY_SHA,
            observer_actor_adapter_contract_sha256=(
                OBSERVER_ADAPTER_CONTRACT_SHA
            ),
            observer_output_receipt=receipt, actor_visible_inputs=inputs,
            ordered_candidate_sha256=ORDERED, candidate_legal=LEGAL,
            lowest_legal_original_candidate_index=1,
            mapped_actions=mapped_actions(), chronology=chronology,
        )


def test_valid_derivation_recomputes_vectors_from_auxiliary_raw_tensors() -> None:
    commitment, observation, raw, auxiliary, *_rest = derivation_material()
    assert root.validate_root_prediction_derivation_commitment(
        commitment,
        observation_commitment=observation,
        raw_predictions=raw,
        auxiliary_tensors=auxiliary,
        **DERIVATION_EXPECTED,
    ) == commitment["derivation_commit_sha256"]
    expected_success = np.mean(
        auxiliary["member_calibrated_success_probability"],
        axis=0, dtype=np.float64,
    ).astype(np.float32)
    assert commitment["derived_vectors"]["mean_success_probability"] == (
        expected_success.astype(float).tolist()
    )
    assert commitment["chronology"]["root_world_model_member_calls"] == 5


def test_raw_prediction_or_auxiliary_one_ulp_tamper_breaks_derivation() -> None:
    commitment, observation, raw, auxiliary, *_rest = derivation_material()
    changed_raw = copy.deepcopy(raw)
    changed_raw["success_logit"][0, 0] = np.nextafter(
        changed_raw["success_logit"][0, 0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(root.RootObservedContractError, match="original tensors"):
        root.validate_root_prediction_derivation_commitment(
            commitment, observation_commitment=observation,
            raw_predictions=changed_raw, auxiliary_tensors=auxiliary,
            **DERIVATION_EXPECTED,
        )
    changed_auxiliary = copy.deepcopy(auxiliary)
    changed_auxiliary["member_composite_rank_score"][0, 0] = np.nextafter(
        changed_auxiliary["member_composite_rank_score"][0, 0],
        np.float32(np.inf), dtype=np.float32,
    )
    with pytest.raises(root.RootObservedContractError, match="original tensors"):
        root.validate_root_prediction_derivation_commitment(
            commitment, observation_commitment=observation,
            raw_predictions=raw, auxiliary_tensors=changed_auxiliary,
            **DERIVATION_EXPECTED,
        )


def test_resigned_derived_vector_or_wrong_member_count_is_rejected() -> None:
    commitment, observation, raw, auxiliary, *_rest = derivation_material()
    changed = copy.deepcopy(commitment)
    changed["derived_vectors"]["mean_composite_rank_score"][0] += 1.0
    changed = resign(changed, "derivation_commit_sha256")
    with pytest.raises(root.RootObservedContractError, match="original tensors"):
        root.validate_root_prediction_derivation_commitment(
            changed, observation_commitment=observation,
            raw_predictions=raw, auxiliary_tensors=auxiliary,
            **DERIVATION_EXPECTED,
        )
    chronology = {**DERIVATION_CHRONOLOGY, "root_world_model_member_calls": 4}
    with pytest.raises(root.RootObservedContractError, match="member calls"):
        root.build_root_prediction_derivation_commitment(
            observation_commitment=observation,
            raw_predictions=raw, auxiliary_tensors=auxiliary,
            root_predictor_authority_sha256=sha("root-predictor-authority"),
            calibration_sha256=sha("root-calibration"),
            source_rank_contract_set_sha256=sha("rank-contract-set"),
            uncertainty_contract_sha256=sha("uncertainty-contract"),
            derivation_implementation_file_sha256=sha("derivation-code"),
            chronology=chronology,
        )


@pytest.mark.parametrize(
    "field",
    [
        "root_predictor_authority_sha256",
        "calibration_sha256",
        "source_rank_contract_set_sha256",
        "uncertainty_contract_sha256",
        "derivation_implementation_file_sha256",
    ],
)
def test_resigned_derivation_provenance_drift_is_rejected(field: str) -> None:
    commitment, observation, raw, auxiliary, *_rest = derivation_material()
    changed = copy.deepcopy(commitment)
    changed[field] = sha(f"attacker-{field}")
    changed = resign(changed, "derivation_commit_sha256")
    with pytest.raises(root.RootObservedContractError, match="frozen authority"):
        root.validate_root_prediction_derivation_commitment(
            changed,
            observation_commitment=observation,
            raw_predictions=raw,
            auxiliary_tensors=auxiliary,
            **DERIVATION_EXPECTED,
        )
