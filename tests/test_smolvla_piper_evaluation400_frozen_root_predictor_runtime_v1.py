from __future__ import annotations

import copy
import hashlib
import json
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
import smolvla_piper_deployment_uncertainty_v1 as uncertainty  # noqa: E402
import smolvla_piper_evaluation400_frozen_root_predictor_runtime_v1 as runtime  # noqa: E402
import smolvla_piper_evaluation400_root_observed_contract_v1 as root  # noqa: E402


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def signed(base: dict[str, Any], field: str) -> dict[str, Any]:
    return {**base, field: runtime.canonical_sha256(base)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


PAIR_ID = sha("pair")
SNAPSHOT_SHA = sha("shared-snapshot")
PRE_ACTION_SHA = sha("pre-action-snapshot")
OBSERVER_AUTHORITY_SHA = sha("observer-authority")
OBSERVER_ADAPTER_CONTRACT_SHA = sha("piper-adapter")
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
        "prediction_sha256": sha("observer-probabilities"),
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
    return inputs, signed(output_base, "receipt_sha256")


def mapped_actions() -> np.ndarray:
    return np.ascontiguousarray(
        np.arange(4 * 3 * 14, dtype=np.float32).reshape(4, 3, 14) / 100.0
    )


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


def checkpoint_logical(payload: dict[str, Any]) -> str:
    return runtime.canonical_sha256(
        {
            name: value
            for name, value in payload.items()
            if name not in {"state_dict", "checkpoint_logical_sha256"}
        }
    )


def make_frozen_artifacts(path: Path) -> str:
    event_spec = signed(
        {
            "format": runtime.EVENT_SPEC_FORMAT,
            "status": "frozen_before_evaluation400",
            "event_names": list(observer.EXPECTED_EVENTS),
            "history_steps": 8,
            "state_dim": 960,
            "proprio_dim": 14,
            "image_feature_dim": 0,
            "action_horizon": 3,
            "action_dim": 14,
            "object_dim": 3,
            "object_output_space": "physical_delta_xyz_m",
            "feature_order": [
                "masked_history_float64_mean_then_float32",
                "proprio",
                "image_feature_if_frozen",
                "observer_event_onehot",
                "observer_predicates_in_frozen_name_order",
                "mapped_action_flat_c_order",
            ],
            "feature_numeric_contract": (
                "native_float32_actor_visible_q0_plus_candidate_action"
            ),
        },
        "event_spec_sha256",
    )
    write_json(path / "event_spec.json", event_spec)
    calibration = signed(
        {
            "format": runtime.CALIBRATION_FORMAT,
            "status": "formal_validation_frozen_deployment_parameters",
            "event_spec_sha256": event_spec["event_spec_sha256"],
            "post_event_temperature": 1.1,
            "next_event_temperature": 1.2,
            "success_temperature": 1.3,
            "conditional_recovery_temperature": 1.4,
            "duration_scale_multiplier": 1.5,
            "object_scale_multiplier": 1.6,
            "object_error_robust_scale_m": 0.2,
            "duration_and_object_scale_application": (
                "add_log_multiplier_exactly_once_inside_frozen_runtime"
            ),
        },
        "calibration_sha256",
    )
    write_json(path / "calibration.json", calibration)
    uncertainty_contract = signed(
        {
            "format": runtime.UNCERTAINTY_FORMAT,
            "status": "formal_validation_frozen_five_head_root_uncertainty",
            "event_spec_sha256": event_spec["event_spec_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
            "root_included_heads": list(uncertainty.ROOT_INCLUDED_HEADS),
            "root_head_count": uncertainty.ROOT_HEAD_COUNT,
            "root_recovery_policy": uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY,
            "algorithm": (
                "mean_of_five_deployment_dimensionless_head_uncertainties"
            ),
            "implementation_file_sha256": runtime.file_sha256(
                Path(uncertainty.__file__).resolve()
            ),
        },
        "uncertainty_contract_sha256",
    )
    write_json(path / "uncertainty_contract.json", uncertainty_contract)
    input_dim = 960 + 14 + 3 * 14 + 5 + 5
    member_records: list[dict[str, Any]] = []
    checkpoint_file_shas: list[str] = []
    rank_contract_shas: list[str] = []
    for index in range(5):
        torch.manual_seed(100 + index)
        model = runtime.CompactRootPredictorV1(
            input_dim=input_dim, hidden_dim=8, event_count=5, object_dim=3
        )
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        checkpoint = {
            "format": runtime.CHECKPOINT_FORMAT,
            "status": "reference_test_only_frozen_state_dict",
            "model_family": runtime.MODEL_FAMILY,
            "promotion_eligible": False,
            "member_index": index,
            "event_spec_sha256": event_spec["event_spec_sha256"],
            "architecture": "CompactRootPredictorV1",
            "input_dim": input_dim,
            "hidden_dim": 8,
            "event_count": 5,
            "object_dim": 3,
            "state_dict": state,
            "state_dict_sha256": runtime.state_dict_sha256(state),
        }
        checkpoint["checkpoint_logical_sha256"] = checkpoint_logical(checkpoint)
        checkpoint_basename = f"member_{index:02d}.pt"
        checkpoint_path = path / checkpoint_basename
        torch.save(checkpoint, checkpoint_path)
        checkpoint_file_sha = runtime.file_sha256(checkpoint_path)
        rank_contract = signed(
            {
                "format": runtime.RANK_CONTRACT_FORMAT,
                "status": "frozen_source_training_composite_rank",
                "member_index": index,
                "source_checkpoint_file_sha256": checkpoint_file_sha,
                "event_spec_sha256": event_spec["event_spec_sha256"],
                "base_score": "candidate_rank_score",
                "source_action_rank_residual": True,
                "source_action_rank_success_only": False,
                "residual_combination": (
                    "candidate_rank_score_plus_action_rank_residual_div_success_temperature"
                ),
                "event_names": list(observer.EXPECTED_EVENTS),
                "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
                "duration_scale": 10.0 + index,
                "success_temperature": 1.0 + index / 10.0,
                "event_weight": 0.25,
                "duration_weight": 0.05,
                "duration_unit": "decision_steps",
                "numeric_contract": "native_ieee754_float32_training_order",
                "score_is_success_logit": False,
                "score_is_success_probability": False,
            },
            "contract_sha256",
        )
        rank_basename = f"rank_contract_{index:02d}.json"
        rank_path = path / rank_basename
        write_json(rank_path, rank_contract)
        rank_file_sha = runtime.file_sha256(rank_path)
        checkpoint_file_shas.append(checkpoint_file_sha)
        rank_contract_shas.append(rank_contract["contract_sha256"])
        member_records.append(
            {
                "member_index": index,
                "checkpoint_basename": checkpoint_basename,
                "checkpoint_file_sha256": checkpoint_file_sha,
                "checkpoint_logical_sha256": checkpoint[
                    "checkpoint_logical_sha256"
                ],
                "rank_contract_basename": rank_basename,
                "rank_contract_file_sha256": rank_file_sha,
                "rank_contract_sha256": rank_contract["contract_sha256"],
            }
        )
    checkpoint_set_sha = runtime.canonical_sha256(checkpoint_file_shas)
    rank_set_sha = runtime.canonical_sha256(rank_contract_shas)
    runtime_authority = signed(
        {
            "format": runtime.RUNTIME_AUTHORITY_FORMAT,
            "status": (
                "actor_visible_reference_inference_before_synthetic_condition_only"
            ),
            "member_count": 5,
            "model_family": runtime.MODEL_FAMILY,
            "promotion_eligible": False,
            "production_compatibility_status": (
                "must_not_replace_ActionConditionedEventWorldModel_or_PiperAdaptedWorldModel"
            ),
            "event_spec": {
                "basename": "event_spec.json",
                "file_sha256": runtime.file_sha256(path / "event_spec.json"),
                "event_spec_sha256": event_spec["event_spec_sha256"],
            },
            "calibration": {
                "basename": "calibration.json",
                "file_sha256": runtime.file_sha256(path / "calibration.json"),
                "calibration_sha256": calibration["calibration_sha256"],
            },
            "uncertainty_contract": {
                "basename": "uncertainty_contract.json",
                "file_sha256": runtime.file_sha256(
                    path / "uncertainty_contract.json"
                ),
                "uncertainty_contract_sha256": uncertainty_contract[
                    "uncertainty_contract_sha256"
                ],
            },
            "members": member_records,
            "checkpoint_file_set_sha256": checkpoint_set_sha,
            "source_rank_contract_set_sha256": rank_set_sha,
            "runtime_implementation_file_sha256": runtime.file_sha256(
                Path(runtime.__file__).resolve()
            ),
            "root_observed_contract_implementation_file_sha256": (
                runtime.file_sha256(Path(root.__file__).resolve())
            ),
            "deployment_uncertainty_implementation_file_sha256": (
                runtime.file_sha256(Path(uncertainty.__file__).resolve())
            ),
            "online_input_contract": (
                "validated_root_observation_commitment_and_original_actor_visible_tensors_only"
            ),
            "member_call_contract": (
                "exactly_five_members_once_each_vectorized_over_all_legal_candidates"
            ),
            "condition_boundary": (
                "all_validation_and_derivation_complete_before_synthetic_test_condition"
            ),
        },
        "runtime_authority_sha256",
    )
    write_json(path / "runtime_authority.json", runtime_authority)
    roles = {
        "event_spec.json": "event_spec",
        "calibration.json": "calibration",
        "uncertainty_contract.json": "uncertainty_contract",
        "runtime_authority.json": "runtime_authority",
        **{f"member_{index:02d}.pt": "member_checkpoint" for index in range(5)},
        **{
            f"rank_contract_{index:02d}.json": "source_rank_contract"
            for index in range(5)
        },
    }
    inventory = [
        {
            "basename": basename,
            "role": roles[basename],
            "file_sha256": runtime.file_sha256(path / basename),
        }
        for basename in sorted(roles)
    ]
    manifest = signed(
        {
            "format": runtime.MANIFEST_FORMAT,
            "status": "frozen_reference_test_authority_not_deployment_promoted",
            "member_count": 5,
            "artifact_inventory": inventory,
            "runtime_authority": {
                "basename": "runtime_authority.json",
                "file_sha256": runtime.file_sha256(
                    path / "runtime_authority.json"
                ),
                "runtime_authority_sha256": runtime_authority[
                    "runtime_authority_sha256"
                ],
            },
            "checkpoint_file_set_sha256": checkpoint_set_sha,
            "source_rank_contract_set_sha256": rank_set_sha,
        },
        "root_predictor_authority_sha256",
    )
    write_json(path / "authority_manifest.json", manifest)
    return manifest["root_predictor_authority_sha256"]


def load(path: Path, authority_sha: str) -> runtime.FrozenRootPredictorRuntime:
    return runtime.load_frozen_root_predictor_runtime(
        path, expected_root_predictor_authority_sha256=authority_sha
    )


def predict(
    loaded: runtime.FrozenRootPredictorRuntime,
    observation: dict[str, Any], inputs: dict[str, Any],
    receipt: dict[str, Any], actions: np.ndarray,
) -> runtime.FrozenRootPredictionResult:
    return loaded.predict(
        observation,
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


def test_five_frozen_members_execute_and_immediately_validate(tmp_path: Path) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    assert loaded.promotion_eligible is False
    observation, inputs, receipt, actions = observation_material()
    result = predict(loaded, observation, inputs, receipt, actions)
    assert result.member_call_count == 5
    assert set(result.raw_predictions) == {
        "post_event_logits", "next_event_logits", "duration_log_mean",
        "duration_log_scale", "success_logit", "object_mean",
        "object_log_scale",
    }
    assert result.raw_predictions["post_event_logits"].shape == (5, 3, 5)
    assert result.raw_predictions["object_mean"].shape == (5, 3, 3)
    assert result.auxiliary_tensors[
        "member_calibrated_success_probability"
    ].shape == (5, 3)
    assert result.derivation_commitment["chronology"] == {
        **CHRONOLOGY,
        "root_world_model_member_calls": 5,
    }
    assert loaded.validate_prediction_result(
        result, observation_commitment=observation
    ) == result.derivation_commitment["derivation_commit_sha256"]


def test_each_member_is_called_once_over_all_legal_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    counts = [0] * 5
    for index, member in enumerate(loaded._members):
        original = member.model.forward

        def wrapped(features, *, _index=index, _original=original):
            counts[_index] += 1
            assert features.shape[0] == 3
            return _original(features)

        monkeypatch.setattr(member.model, "forward", wrapped)
    observation, inputs, receipt, actions = observation_material()
    predict(loaded, observation, inputs, receipt, actions)
    assert counts == [1, 1, 1, 1, 1]


FROZEN_BASENAMES = [
    "authority_manifest.json",
    "calibration.json",
    "event_spec.json",
    *(f"member_{index:02d}.pt" for index in range(5)),
    *(f"rank_contract_{index:02d}.json" for index in range(5)),
    "runtime_authority.json",
    "uncertainty_contract.json",
]


@pytest.mark.parametrize("basename", FROZEN_BASENAMES)
def test_any_artifact_tamper_after_load_fails_before_prediction(
    tmp_path: Path, basename: str,
) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    path = tmp_path / basename
    path.write_bytes(path.read_bytes() + b"tamper")
    observation, inputs, receipt, actions = observation_material()
    with pytest.raises(runtime.FrozenRootPredictorError, match="changed"):
        predict(loaded, observation, inputs, receipt, actions)


def test_externally_pinned_authority_is_required(tmp_path: Path) -> None:
    make_frozen_artifacts(tmp_path)
    with pytest.raises(runtime.FrozenRootPredictorError, match="externally pinned"):
        load(tmp_path, sha("attacker-self-authority"))


def test_missing_fifth_member_fails_at_load(tmp_path: Path) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    (tmp_path / "member_04.pt").unlink()
    with pytest.raises(runtime.FrozenRootPredictorError, match="missing or extra"):
        load(tmp_path, authority_sha)


def test_low_confidence_observer_fails_before_any_member_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    observation, inputs, receipt, actions = observation_material()
    changed = dict(receipt)
    changed["confidence"] = 0.6
    changed = signed(
        {name: value for name, value in changed.items() if name != "receipt_sha256"},
        "receipt_sha256",
    )
    calls = [0]
    original = loaded._members[0].model.forward

    def wrapped(features):
        calls[0] += 1
        return original(features)

    monkeypatch.setattr(loaded._members[0].model, "forward", wrapped)
    with pytest.raises(runtime.FrozenRootPredictorError, match="before member inference"):
        predict(loaded, observation, inputs, changed, actions)
    assert calls == [0]


@pytest.mark.parametrize("forbidden", ["current_event_id", "object_poses", "target_success"])
def test_direct_event_pose_or_target_actor_input_fails_before_inference(
    tmp_path: Path, forbidden: str,
) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    observation, inputs, receipt, actions = observation_material()
    inputs[forbidden] = 1
    with pytest.raises(runtime.FrozenRootPredictorError, match="before member inference"):
        predict(loaded, observation, inputs, receipt, actions)


def test_mapped_action_one_ulp_tamper_fails_before_inference(tmp_path: Path) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    observation, inputs, receipt, actions = observation_material()
    changed = actions.copy()
    changed[2, 0, 0] = np.nextafter(
        changed[2, 0, 0], np.float32(np.inf), dtype=np.float32
    )
    with pytest.raises(runtime.FrozenRootPredictorError, match="before member inference"):
        predict(loaded, observation, inputs, receipt, changed)


def test_handwritten_derived_vector_even_resigned_is_rejected(tmp_path: Path) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    observation, inputs, receipt, actions = observation_material()
    result = predict(loaded, observation, inputs, receipt, actions)
    changed = copy.deepcopy(result.derivation_commitment)
    changed["derived_vectors"]["mean_success_probability"][0] += 0.1
    unsigned = {
        name: value
        for name, value in changed.items()
        if name != "derivation_commit_sha256"
    }
    changed["derivation_commit_sha256"] = root.canonical_sha256(unsigned)
    forged = runtime.FrozenRootPredictionResult(
        raw_predictions=result.raw_predictions,
        auxiliary_tensors=result.auxiliary_tensors,
        derivation_commitment=changed,
        member_call_count=5,
    )
    with pytest.raises(runtime.FrozenRootPredictorError, match="derived vectors"):
        loaded.validate_prediction_result(forged, observation_commitment=observation)


@pytest.mark.parametrize(
    "field",
    [
        "source_rank_contract_set_sha256",
        "uncertainty_contract_sha256",
        "derivation_implementation_file_sha256",
    ],
)
def test_resigned_runtime_derivation_provenance_drift_fails_closed(
    tmp_path: Path, field: str,
) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    observation, inputs, receipt, actions = observation_material()
    result = predict(loaded, observation, inputs, receipt, actions)
    changed = copy.deepcopy(result.derivation_commitment)
    changed[field] = sha(f"attacker-{field}")
    unsigned = {
        name: value
        for name, value in changed.items()
        if name != "derivation_commit_sha256"
    }
    changed["derivation_commit_sha256"] = root.canonical_sha256(unsigned)
    forged = runtime.FrozenRootPredictionResult(
        raw_predictions=result.raw_predictions,
        auxiliary_tensors=result.auxiliary_tensors,
        derivation_commitment=changed,
        member_call_count=5,
    )
    with pytest.raises(runtime.FrozenRootPredictorError, match="derived vectors"):
        loaded.validate_prediction_result(forged, observation_commitment=observation)


def test_reported_call_count_other_than_five_is_rejected(tmp_path: Path) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    observation, inputs, receipt, actions = observation_material()
    result = predict(loaded, observation, inputs, receipt, actions)
    forged = runtime.FrozenRootPredictionResult(
        raw_predictions=result.raw_predictions,
        auxiliary_tensors=result.auxiliary_tensors,
        derivation_commitment=result.derivation_commitment,
        member_call_count=4,
    )
    with pytest.raises(runtime.FrozenRootPredictorError, match="authority changed"):
        loaded.validate_prediction_result(forged, observation_commitment=observation)


def test_internal_member_inventory_other_than_five_fails_closed(tmp_path: Path) -> None:
    authority_sha = make_frozen_artifacts(tmp_path)
    loaded = load(tmp_path, authority_sha)
    loaded._members = loaded._members[:4]
    observation, inputs, receipt, actions = observation_material()
    with pytest.raises(runtime.FrozenRootPredictorError, match="exactly five"):
        predict(loaded, observation, inputs, receipt, actions)
