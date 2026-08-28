#!/usr/bin/env python3
"""Fail-closed result evaluator for an externally executed evaluation400 v3.

This program never calls a policy or simulator.  It accepts only an exact,
signed, complete 400-pair execution closure and publishes the preregistered
paired-success statistics as a create-once signed receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import numpy as np

import smolvla_piper_paired_success_protocol_v3 as paired_v3
import smolvla_piper_deployment_uncertainty_v1 as deployment_uncertainty


RESULT_FORMAT = "etsf_smolvla_piper_evaluation400_result_receipt_v3"
RESULT_STATUS = "complete_exact_400_signed_no_subset_result"
TERMINAL_FORMAT = "etsf_smolvla_piper_evaluation400_execution_terminal_v3"
TERMINAL_STATUS = "complete_exact_400_external_execution_terminal"
PAIR_FORMAT = "etsf_smolvla_piper_evaluation400_pair_execution_receipt_v3"
PAIR_STATUS = "complete_exact_two_condition_pair"
CONDITION_FORMAT = "etsf_smolvla_piper_evaluation400_condition_execution_receipt_v3"
CONDITION_STATUS = "complete_single_attempt_condition"
LEDGER_FORMAT = "etsf_smolvla_piper_evaluation400_one_shot_ledger_v3"
LEDGER_FINAL_STATE = "COMPLETE_400_RESULT_READY"
CLAIM_FORMAT = "etsf_smolvla_piper_evaluation400_execution_claim_v3"
CLAIM_STATUS = "claimed_once_preoutcome_nonreleasable"
LEDGER_EVENT_FORMAT = "etsf_smolvla_piper_evaluation400_ledger_event_v3"
LEDGER_EVENT_STATUS = "append_only_hash_chained_execution_event"
EXECUTOR_SIGNATURE_CONTEXT = (
    b"ETSF/SmolVLA/Piper/evaluation400-v3/executor-receipt\0"
)
RESULT_SIGNATURE_CONTEXT = (
    b"ETSF/SmolVLA/Piper/evaluation400-v3/result-receipt\0"
)
BOOTSTRAP_GENERATOR = "shake256_rejection_uint16_v1"
BOOTSTRAP_FORMAT = "raw_little_endian_uint16"
CONTINUATION_CONTRACT = "frozen_lowest_legal_feasibility_continuation_v1"
SUCCESS_SOURCE = "simulator_terminal_task_success_exact_bool"
PAIR_COUNT = 400
CONDITION_COUNT = 800
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_SHAPE = (BOOTSTRAP_SAMPLES, PAIR_COUNT)
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
RECEIPT_FIELDS = {
    "format", "status", "signature_algorithm", "statement",
    "executor_signature_ed25519_hex", "receipt_sha256",
}
RECORD_FIELDS = {"path", "file_sha256", "logical_sha256"}
OPAQUE_RECORD_FIELDS = {"path", "file_sha256"}
EXECUTION_ARTIFACT_FIELDS = {
    "runner_result", "stage_launch", "stage_lifecycle", "stage_log",
    "stage_exit", "gpu_idle_before", "gpu_idle_after", "gpu_uuid",
}
RUNNER_RESULT_FORMAT = "etsf_smolvla_piper_evaluation400_condition_runner_result_v3"
RUNNER_RESULT_STATUS = "complete_single_condition_from_bound_snapshot"
STAGE_FORMAT = "etsf_smolvla_piper_evaluation400_condition_stage_v3"
RUNNER_RESULT_FIELDS = {
    "format", "status", "request", "pair_id", "ordinal", "attempt",
    "condition", "condition_ordinal", "shared_snapshot_sha256",
    "candidate_count", "ordered_candidate_sha256", "candidate_legal",
    "candidate_registry_sha256", "selected_candidate_index",
    "schema6_execution_authority_file_sha256",
    "schema6_runtime_contract_sha256", "max_episode_steps",
    "selector_execution_proof", "selector_execution_proof_sha256",
    "selector_score_contract", "source_rank_score_contract_sha256",
    "source_contract_rank_score_is_success_logit",
    "source_contract_rank_score_is_success_probability",
    "formal190_target_outcome_calibrated_acceptance_margin",
    "continuation_contract", "continuation_policy_sha256",
    "continuation_rerank_after_root", "candidate_replacement_count",
    "continuation_proof_sha256", "task_success", "trajectory_artifact",
    "continuation_artifact", "simulator_exit_code", "result_sha256",
}
CONDITION_REQUEST_FIELDS = {
    "format", "status", "plan_sha256", "bundle_sha256", "claim_sha256",
    "pair_id", "ordinal", "requested_seed", "resolved_seed",
    "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
    "initial_commanded_drive_target_sha256", "attempt", "pair_identity_sha256",
    "condition", "condition_ordinal", "condition_order", "shared_snapshot_sha256",
    "candidate_count", "candidate_generation_contract_sha256",
    "postfreeze_identity_or_order_change_authorized",
    "outcome_visible_before_condition_start", "request_sha256",
}
EXACT_STAGE_ENVIRONMENT_KEYS = sorted({
    "PATH", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
})
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)


class Evaluation400ResultError(RuntimeError):
    """The frozen execution/result contract failed closed."""


def _exact_float(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Evaluation400ResultError(f"{role} must be a finite number")
    return float(value)


def _equal_float(left: Any, right: Any, role: str) -> None:
    if not math.isclose(
        _exact_float(left, role), _exact_float(right, role),
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise Evaluation400ResultError(f"{role} changed")


def validate_selector_execution_proof(
    proof: Mapping[str, Any], *, condition: str,
    candidate_legal: Sequence[bool], selected_candidate_index: int,
) -> None:
    """Recompute the frozen baseline or ETSF root decision from signed inputs."""

    if condition == "baseline":
        expected = {
            "selector", "event_model_members_called", "selected_candidate_index",
            "score_contract", "source_rank_score_contract_sha256",
            "source_contract_rank_score_is_success_logit",
            "source_contract_rank_score_is_success_probability",
            "formal190_target_outcome_calibrated_acceptance_margin",
        }
        lowest = next((index for index, legal in enumerate(candidate_legal) if legal), None)
        if (
            set(proof) != expected
            or proof.get("selector") != "lowest_legal_feasibility_root_candidate"
            or type(proof.get("event_model_members_called")) is not int
            or proof["event_model_members_called"] != 0
            or proof.get("selected_candidate_index") != lowest
            or selected_candidate_index != lowest
            or proof.get("score_contract")
            != "lowest_legal_feasibility_root_candidate"
            or proof.get("source_rank_score_contract_sha256") != []
            or proof.get("source_contract_rank_score_is_success_logit") is not False
            or proof.get("source_contract_rank_score_is_success_probability") is not False
            or proof.get("formal190_target_outcome_calibrated_acceptance_margin")
            is not False
        ):
            raise Evaluation400ResultError("baseline selector proof changed")
        return
    if condition != "etsf":
        raise Evaluation400ResultError("unknown condition selector proof")

    expected_outer = {
        "selector", "event_model_members_called", "uncertainty_gate_applied",
        "selector_output_sha256", "selector_decision",
        "selected_candidate_index", "proposed_candidate_index", "score_margin",
        "total_uncertainty", "decision_algebra_sha256", "calibration_sha256",
        "formal190_root_group_ranker_sha256", "score_contract",
        "source_rank_score_contract_sha256",
        "source_rank_numeric_contract",
        "source_contract_rank_score_is_success_logit",
        "source_contract_rank_score_is_success_probability",
        "formal190_target_outcome_calibrated_acceptance_margin",
        "predicted_success_used_as_outcome", "selector_proof_sha256",
    }
    decision = proof.get("selector_decision")
    if (
        set(proof) != expected_outer
        or proof.get("selector")
        != "frozen_five_member_event_world_model_with_uncertainty_abstention"
        or type(proof.get("event_model_members_called")) is not int
        or proof["event_model_members_called"] != 5
        or proof.get("uncertainty_gate_applied") is not True
        or proof.get("score_contract")
        != "five_member_adjusted_source_composite_candidate_rank_score_margin"
        or proof.get("source_contract_rank_score_is_success_logit") is not False
        or proof.get("source_contract_rank_score_is_success_probability") is not False
        or proof.get("formal190_target_outcome_calibrated_acceptance_margin") is not True
        or proof.get("predicted_success_used_as_outcome") is not False
        or proof.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(decision, Mapping)
        or not is_sha(proof.get("selector_proof_sha256"))
        or proof["selector_proof_sha256"]
        != canonical_sha256({key: proof[key] for key in proof if key != "selector_proof_sha256"})
        or proof.get("selector_output_sha256") != decision.get("selector_proof_sha256")
        or decision.get("selector_proof_sha256")
        != canonical_sha256({key: decision[key] for key in decision if key != "selector_proof_sha256"})
    ):
        raise Evaluation400ResultError("ETSF selector envelope changed")
    selector_input = decision.get("selector_input")
    if (
        not isinstance(selector_input, Mapping)
        or decision.get("selector_input_sha256") != canonical_sha256(selector_input)
        or selector_input.get("calibration_sha256") != proof.get("calibration_sha256")
    ):
        raise Evaluation400ResultError("ETSF selector input binding changed")
    uncertainty_record = selector_input.get("deployment_uncertainty_implementation")
    actual_uncertainty_path = Path(deployment_uncertainty.__file__).resolve()
    if uncertainty_record != {
        "path": str(actual_uncertainty_path),
        "file_sha256": hashlib.sha256(actual_uncertainty_path.read_bytes()).hexdigest(),
    }:
        raise Evaluation400ResultError("deployment uncertainty implementation changed")
    indices = np.asarray(selector_input.get("prediction_candidate_indices"))
    legal = selector_input.get("candidate_legal")
    fallback = selector_input.get("fallback_candidate_index")
    predictions = selector_input.get("predictions")
    parameters = selector_input.get("uncertainty_parameters")
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or indices.astype(int).tolist()
        != [index for index, value in enumerate(candidate_legal) if value]
        or len(set(indices.astype(int).tolist())) != len(indices)
        or legal != list(candidate_legal)
        or type(fallback) is not int
        or fallback != next(index for index, value in enumerate(candidate_legal) if value)
        or not isinstance(predictions, Mapping)
        or not isinstance(parameters, Mapping)
        or selector_input.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
    ):
        raise Evaluation400ResultError("ETSF selector candidate input changed")
    try:
        components = deployment_uncertainty.root_components(
            predictions=predictions, parameters=parameters
        )
    except (KeyError, TypeError, deployment_uncertainty.DeploymentUncertaintyError) as error:
        raise Evaluation400ResultError("ETSF uncertainty cannot be recomputed") from error
    recorded_components = decision.get("uncertainty_components")
    if not isinstance(recorded_components, Mapping) or set(recorded_components) != set(components):
        raise Evaluation400ResultError("ETSF uncertainty component inventory changed")
    for name, value in components.items():
        recorded = np.asarray(recorded_components[name], dtype=np.float64)
        if recorded.shape != value.shape or not np.allclose(
            recorded, value, rtol=1e-12, atol=1e-12
        ):
            raise Evaluation400ResultError(f"ETSF uncertainty changed: {name}")
    rank_raw = np.asarray(
        predictions.get("source_contract_rank_score"), dtype=np.float64
    )
    base_rank_raw = np.asarray(
        predictions.get("source_contract_base_rank_score"), dtype=np.float64
    )
    residual_raw = np.asarray(
        predictions.get("source_action_rank_residual"), dtype=np.float64
    )
    rank = np.asarray(
        predictions.get("source_contract_rank_score"), dtype=np.float32
    )
    base_rank = np.asarray(
        predictions.get("source_contract_base_rank_score"), dtype=np.float32
    )
    residual = np.asarray(
        predictions.get("source_action_rank_residual"), dtype=np.float32
    )
    self_reported_temperature_values = decision.get(
        "member_source_rank_success_temperatures"
    )
    self_reported_temperatures_raw = np.asarray(
        self_reported_temperature_values, dtype=np.float64
    )
    self_reported_temperatures = np.asarray(
        self_reported_temperature_values, dtype=np.float32
    )
    recorded_rank_raw = np.asarray(
        decision.get("member_source_contract_rank_scores"), dtype=np.float64
    )
    recorded_base_raw = np.asarray(
        decision.get("member_source_contract_base_rank_scores"), dtype=np.float64
    )
    recorded_residual_raw = np.asarray(
        decision.get("member_source_action_rank_residuals"), dtype=np.float64
    )
    if (
        rank.shape != (5, len(indices))
        or base_rank.shape != rank.shape
        or residual.shape != rank.shape
        or self_reported_temperatures.shape != (5,)
        or not np.isfinite(self_reported_temperatures_raw).all()
        or not np.isfinite(rank).all()
        or not np.isfinite(base_rank).all()
        or not np.isfinite(residual).all()
        or not np.isfinite(self_reported_temperatures).all()
        or not np.array_equal(rank_raw, rank.astype(np.float64))
        or not np.array_equal(base_rank_raw, base_rank.astype(np.float64))
        or not np.array_equal(residual_raw, residual.astype(np.float64))
        or bool((self_reported_temperatures <= 0.0).any())
        or np.asarray(decision.get("member_source_contract_rank_scores")).shape
        != rank.shape
        or np.asarray(
            decision.get("member_source_contract_base_rank_scores")
        ).shape != rank.shape
        or np.asarray(decision.get("member_source_action_rank_residuals")).shape
        != rank.shape
        or not np.array_equal(
            recorded_rank_raw, recorded_rank_raw.astype(np.float32).astype(np.float64)
        )
        or not np.array_equal(
            recorded_base_raw, recorded_base_raw.astype(np.float32).astype(np.float64)
        )
        or not np.array_equal(
            recorded_residual_raw,
            recorded_residual_raw.astype(np.float32).astype(np.float64),
        )
        or not np.array_equal(
            recorded_rank_raw.astype(np.float32),
            rank,
        )
        or not np.array_equal(
            recorded_base_raw.astype(np.float32), base_rank,
        )
        or not np.array_equal(
            recorded_residual_raw.astype(np.float32), residual,
        )
        or not np.array_equal(
            base_rank + residual / self_reported_temperatures[:, None],
            rank,
        )
    ):
        raise Evaluation400ResultError("ETSF composite member scores changed")
    if fallback not in indices:
        raise Evaluation400ResultError("ETSF fallback absent from prediction rows")
    baseline_row = int(np.flatnonzero(indices == fallback)[0])
    mean_rank = rank.astype(np.float64).mean(axis=0)
    margins = mean_rank - mean_rank[baseline_row]
    alternatives = [row for row in range(len(indices)) if int(indices[row]) != fallback]
    proposed_row = min(
        alternatives, key=lambda row: (-float(margins[row]), int(indices[row]))
    )
    proposed = int(indices[proposed_row])
    proposed_margin = float(margins[proposed_row])
    structured = components["structured_five_head"]
    proposed_uncertainty = float(structured[proposed_row])
    baseline_uncertainty = float(structured[baseline_row])
    pair_uncertainty = max(proposed_uncertainty, baseline_uncertainty)
    minimum_margin = _exact_float(
        decision.get("minimum_formal190_composite_margin"), "formal margin"
    )
    maximum_pair = _exact_float(
        decision.get("maximum_formal190_pair_uncertainty"), "formal pair gate"
    )
    maximum_global = _exact_float(
        decision.get("maximum_global_total_uncertainty"), "global uncertainty gate"
    )
    accepted = bool(
        proposed_margin > minimum_margin
        and proposed_uncertainty <= maximum_global
        and pair_uncertainty <= maximum_pair
    )
    recomputed_selected = proposed if accepted else fallback
    algebra = {
        "score_semantics": (
            "mean_member_source_contract_rank_score(candidate)-"
            "mean_member_source_contract_rank_score(lowest_legal_baseline)"
        ),
        "score_is_success_logit": False,
        "score_is_success_probability": False,
        "source_rank_reconstruction": (
            "source_contract_base_rank_score+"
            "source_action_rank_residual/source_success_temperature"
        ),
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_success_temperatures": self_reported_temperatures_raw.tolist(),
        "alternative_set_contract": (
            "all_legal_candidates_except_lowest_legal_baseline"
        ),
        "margin_comparison": "strict_greater_than_formal190_threshold",
        "proposed_candidate_index": proposed,
        "fallback_candidate_index": fallback,
        "score_margin": proposed_margin,
        "minimum_margin": minimum_margin,
        "proposed_uncertainty": proposed_uncertainty,
        "baseline_uncertainty": baseline_uncertainty,
        "pair_uncertainty": pair_uncertainty,
        "maximum_total_uncertainty": maximum_global,
        "maximum_pair_uncertainty": maximum_pair,
        "accepted": accepted,
        "selected_candidate_index": recomputed_selected,
    }
    if (
        decision.get("prediction_candidate_indices") != indices.astype(int).tolist()
        or decision.get("alternative_candidate_indices")
        != [int(indices[row]) for row in alternatives]
        or decision.get("alternative_set_contract")
        != "all_legal_candidates_except_lowest_legal_baseline"
        or decision.get("margin_comparison")
        != "strict_greater_than_formal190_threshold"
        or decision.get("root_recovery_uncertainty_policy")
        != deployment_uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY
        or type(decision.get("root_structured_uncertainty_head_count")) is not int
        or decision["root_structured_uncertainty_head_count"]
        != deployment_uncertainty.ROOT_HEAD_COUNT
        or decision.get("proposed_candidate_index") != proposed
        or decision.get("selected_candidate_index") != recomputed_selected
        or selected_candidate_index != recomputed_selected
        or decision.get("candidate_change_accepted") is not accepted
        or proof.get("proposed_candidate_index") != proposed
        or proof.get("selected_candidate_index") != recomputed_selected
        or decision.get("decision_algebra_sha256") != canonical_sha256(algebra)
        or proof.get("decision_algebra_sha256") != canonical_sha256(algebra)
    ):
        raise Evaluation400ResultError("ETSF proposal/strict gate decision changed")
    for left, right, role in (
        (decision.get("score_margin"), proposed_margin, "composite margin"),
        (proof.get("score_margin"), proposed_margin, "proof composite margin"),
        (decision.get("proposed_uncertainty"), proposed_uncertainty, "proposed uncertainty"),
        (decision.get("baseline_uncertainty"), baseline_uncertainty, "baseline uncertainty"),
        (decision.get("total_uncertainty"), pair_uncertainty, "pair uncertainty"),
        (proof.get("total_uncertainty"), pair_uncertainty, "proof pair uncertainty"),
    ):
        _equal_float(left, right, role)


def validate_selector_proof_against_authority(
    proof: Mapping[str, Any], selector_authority: Mapping[str, Any]
) -> None:
    """Bind every scientific selector input to the paired Formal190 authority."""

    decision = proof.get("selector_decision")
    selector_input = (
        decision.get("selector_input") if isinstance(decision, Mapping) else None
    )
    contracts = selector_authority.get("source_rank_score_contracts")
    contract_shas = selector_authority.get("source_rank_score_contract_sha256")
    member_authority = selector_authority.get("source_rank_member_authority")
    authority_members = (
        member_authority.get("members")
        if isinstance(member_authority, Mapping) else None
    )
    deployment_parameters = selector_authority.get("deployment_parameters")
    formal_thresholds = selector_authority.get("formal190_thresholds")
    if (
        not isinstance(decision, Mapping)
        or not isinstance(selector_input, Mapping)
        or not isinstance(contracts, list)
        or len(contracts) != 5
        or not isinstance(contract_shas, list)
        or len(contract_shas) != 5
        or not isinstance(deployment_parameters, Mapping)
        or set(deployment_parameters) != {
            "post_event_temperature", "next_event_temperature",
            "success_temperature", "conditional_recovery_temperature",
            "duration_scale_multiplier", "object_scale_multiplier",
            "object_error_robust_scale_m",
            "deployment_uncertainty_contract_sha256",
        }
        or not isinstance(formal_thresholds, Mapping)
        or set(formal_thresholds) != {
            "minimum_formal190_composite_margin",
            "maximum_formal190_pair_uncertainty",
            "maximum_global_total_uncertainty", "root_group_ranker_sha256",
        }
        or selector_authority.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(member_authority, Mapping)
        or set(member_authority)
        != {"source_rank_numeric_contract", "members"}
        or member_authority.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(authority_members, list)
        or len(authority_members) != 5
        or selector_authority.get("source_rank_member_authority_sha256")
        != canonical_sha256(member_authority)
    ):
        raise Evaluation400ResultError("paired selector scientific authority changed")
    temperatures: list[float] = []
    source_checkpoint_shas: list[str] = []
    for index, contract in enumerate(contracts):
        if not isinstance(contract, Mapping):
            raise Evaluation400ResultError("paired Source rank contract changed")
        unsigned = dict(contract)
        logical = unsigned.pop("contract_sha256", None)
        temperature = contract.get("success_temperature")
        if (
            not is_sha(logical)
            or logical != canonical_sha256(unsigned)
            or logical != contract_shas[index]
            or not is_sha(contract.get("source_checkpoint_file_sha256"))
            or contract.get("base_score") != "candidate_rank_score"
            or contract.get("source_action_rank_residual") is not True
            or contract.get("source_action_rank_success_only") is not False
            or contract.get("source_rank_numeric_contract")
            != SOURCE_RANK_NUMERIC_CONTRACT
            or contract.get("residual_combination")
            != "candidate_rank_score_plus_action_rank_residual"
        ):
            raise Evaluation400ResultError("paired Source rank contract changed")
        temperatures.append(_exact_float(temperature, "Source rank temperature"))
        if temperatures[-1] <= 0.0:
            raise Evaluation400ResultError("Source rank temperature must be positive")
        source_checkpoint_shas.append(contract["source_checkpoint_file_sha256"])
        authority_row = authority_members[index]
        if (
            not isinstance(authority_row, Mapping)
            or set(authority_row)
            != {
                "member_index", "source_checkpoint_file_sha256",
                "source_rank_score_contract_sha256", "success_temperature",
            }
            or type(authority_row.get("member_index")) is not int
            or authority_row["member_index"] != index
            or authority_row.get("source_checkpoint_file_sha256")
            != contract["source_checkpoint_file_sha256"]
            or authority_row.get("source_rank_score_contract_sha256") != logical
            or _exact_float(
                authority_row.get("success_temperature"),
                "Source rank member authority temperature",
            ) != temperatures[index]
        ):
            raise Evaluation400ResultError(
                "paired Source rank member authority changed"
            )
    if (
        len(set(contract_shas)) != 5
        or len(set(source_checkpoint_shas)) != 5
    ):
        raise Evaluation400ResultError(
            "paired Source rank member identities are not unique"
        )
    raw_decision_temperatures = decision.get(
        "member_source_rank_success_temperatures"
    )
    if (
        not isinstance(raw_decision_temperatures, list)
        or len(raw_decision_temperatures) != 5
    ):
        raise Evaluation400ResultError(
            "Source rank decision temperatures changed: ETSF proof parameters differ"
        )
    decision_temperatures = [
        _exact_float(value, "Source rank decision temperature")
        for value in raw_decision_temperatures
    ]
    if decision_temperatures != temperatures:
        raise Evaluation400ResultError(
            "Source rank decision temperatures changed: ETSF proof parameters differ"
        )
    parameters = selector_input.get("uncertainty_parameters")
    expected_uncertainty_parameters = {
        key: deployment_parameters[key]
        for key in (
            "post_event_temperature", "next_event_temperature",
            "success_temperature", "conditional_recovery_temperature",
            "object_error_robust_scale_m",
        )
    }
    if (
        selector_input.get("selector_authority_sha256")
        != selector_authority.get("selector_authority_sha256")
        or selector_input.get("deployment_parameters") != deployment_parameters
        or selector_input.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or proof.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or selector_input.get("formal190_thresholds") != formal_thresholds
        or parameters != expected_uncertainty_parameters
        or proof.get("calibration_sha256")
        != selector_authority.get("calibration_sha256")
        or proof.get("formal190_root_group_ranker_sha256")
        != formal_thresholds["root_group_ranker_sha256"]
        or proof.get("source_rank_score_contract_sha256") != contract_shas
        or decision.get("member_source_rank_success_temperatures") != temperatures
        or decision.get("minimum_formal190_composite_margin")
        != formal_thresholds["minimum_formal190_composite_margin"]
        or decision.get("maximum_formal190_pair_uncertainty")
        != formal_thresholds["maximum_formal190_pair_uncertainty"]
        or decision.get("maximum_global_total_uncertainty")
        != formal_thresholds["maximum_global_total_uncertainty"]
        or deployment_parameters["deployment_uncertainty_contract_sha256"]
        != selector_authority.get("uncertainty_contract", {}).get(
            "deployment_uncertainty_contract_sha256"
        )
    ):
        raise Evaluation400ResultError(
            "ETSF proof parameters differ from paired Formal190 authority"
        )
    predictions = selector_input.get("predictions")
    if not isinstance(predictions, Mapping):
        raise Evaluation400ResultError("ETSF authority-bound predictions are missing")
    base_raw = np.asarray(
        predictions.get("source_contract_base_rank_score"), dtype=np.float64
    )
    residual_raw = np.asarray(
        predictions.get("source_action_rank_residual"), dtype=np.float64
    )
    composite_raw = np.asarray(
        predictions.get("source_contract_rank_score"), dtype=np.float64
    )
    base_rank = base_raw.astype(np.float32)
    residual = residual_raw.astype(np.float32)
    composite = composite_raw.astype(np.float32)
    temperature32 = np.asarray(temperatures, dtype=np.float32)[:, None]
    if (
        base_rank.shape != composite.shape
        or residual.shape != composite.shape
        or composite.ndim != 2
        or composite.shape[0] != 5
        or not np.isfinite(temperature32).all()
        or bool((temperature32 <= np.float32(0.0)).any())
        or not np.array_equal(base_raw, base_rank.astype(np.float64))
        or not np.array_equal(residual_raw, residual.astype(np.float64))
        or not np.array_equal(composite_raw, composite.astype(np.float64))
        or not np.array_equal(
            base_rank + residual / temperature32,
            composite,
        )
    ):
        raise Evaluation400ResultError(
            "ETSF composite rank differs from authority temperature algebra"
        )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _require_sha(value: Any, role: str) -> str:
    if not is_sha(value):
        raise Evaluation400ResultError(f"{role} must be lowercase SHA-256")
    return str(value)


def _require_int(value: Any, expected: int | None, role: str) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        suffix = "an exact integer" if expected is None else f"exact integer {expected}"
        raise Evaluation400ResultError(f"{role} must be {suffix}")
    return value


def _require_bool(value: Any, expected: bool, role: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise Evaluation400ResultError(f"{role} must be exact boolean {expected}")
    return value


def _decode_hex(value: Any, length: int, role: str) -> bytes:
    if not isinstance(value, str) or len(value) != 2 * length:
        raise Evaluation400ResultError(f"{role} must be {length} bytes in hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise Evaluation400ResultError(f"{role} is invalid hex") from error
    if decoded.hex() != value:
        raise Evaluation400ResultError(f"{role} must be canonical lowercase hex")
    return decoded


def _reject_constant(token: str) -> None:
    raise Evaluation400ResultError(f"non-finite JSON number is forbidden: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise Evaluation400ResultError(f"duplicate JSON key is forbidden: {key}")
        value[key] = child
    return value


def _safe_lexical_path(raw_path: Path, role: str) -> Path:
    expanded = os.path.expanduser(os.fspath(raw_path))
    lexical = Path(os.path.abspath(expanded))
    if not raw_path.is_absolute() or os.fspath(raw_path) != os.fspath(lexical):
        raise Evaluation400ResultError(f"{role} path must be canonical absolute")
    if lexical.suffix.casefold() in HDF_SUFFIXES:
        raise Evaluation400ResultError(f"{role} HDF input is forbidden")
    lowered = lexical.name.casefold()
    if any(token in lowered for token in ("trajectory", "labels", "_label")):
        raise Evaluation400ResultError(f"{role} outcome-container namespace is forbidden")
    return lexical


def secure_read(
    raw_path: Path, expected_file_sha256: str, role: str, *,
    require_frozen: bool = True,
) -> tuple[Path, bytes, str]:
    expected = _require_sha(expected_file_sha256, f"{role} expected file SHA")
    lexical = _safe_lexical_path(raw_path, role)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(lexical.anchor, directory_flags)
    file_fd: int | None = None
    try:
        for component in lexical.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(lexical.name, file_flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise Evaluation400ResultError(f"{role} must be a regular file")
        if require_frozen and before.st_mode & 0o222:
            raise Evaluation400ResultError(f"{role} must be frozen read-only")
        chunks: list[bytes] = []
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(file_fd)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise Evaluation400ResultError(f"{role} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise Evaluation400ResultError(f"{role} short read")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise Evaluation400ResultError(f"{role} file SHA mismatch")
        return lexical, payload, digest
    except OSError as error:
        raise Evaluation400ResultError(
            f"{role} cannot be opened without following symlinks"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def read_json(
    path: Path, expected_file_sha256: str, role: str,
) -> tuple[Path, dict[str, Any], str]:
    if path.suffix.casefold() != ".json":
        raise Evaluation400ResultError(f"{role} must be JSON")
    bound, payload, digest = secure_read(path, expected_file_sha256, role)
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Evaluation400ResultError(f"{role} is invalid strict JSON") from error
    if not isinstance(value, dict):
        raise Evaluation400ResultError(f"{role} must contain an object")
    return bound, value, digest


def _record(value: Any, role: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping) or set(value) != RECORD_FIELDS
        or not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not is_sha(value.get("file_sha256"))
        or not is_sha(value.get("logical_sha256"))
    ):
        raise Evaluation400ResultError(f"{role} descriptor changed")
    return dict(value)


def _descriptor(path: Path, file_sha256: str, logical_sha256: str) -> dict[str, str]:
    return {
        "path": str(path), "file_sha256": file_sha256,
        "logical_sha256": logical_sha256,
    }


def _opaque_record(value: Any, role: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping) or set(value) != OPAQUE_RECORD_FIELDS
        or not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not is_sha(value.get("file_sha256"))
    ):
        raise Evaluation400ResultError(f"{role} opaque descriptor changed")
    return dict(value)


def _verify_canonical_field(
    value: Mapping[str, Any], field: str, role: str
) -> str:
    unsigned = dict(value)
    logical = unsigned.pop(field, None)
    if not is_sha(logical) or logical != canonical_sha256(unsigned):
        raise Evaluation400ResultError(f"{role} logical SHA mismatch")
    return str(logical)


def _path_matches(record_path: str, actual: Path, role: str) -> None:
    if record_path != str(actual):
        raise Evaluation400ResultError(f"{role} descriptor path mismatch")


def _implementation_sha(path: Path, expected: str, role: str) -> str:
    _bound, _payload, digest = secure_read(
        Path(os.path.abspath(os.fspath(path))), expected, role,
        require_frozen=False,
    )
    return digest


def _validate_core_pairs(core: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_top = {
        "format", "status", "post_collection_v3", "development_and_formal190",
        "r7h_target_adapter_lineage", "evaluation400", "deployment",
        "execution_inventory", "authority_policy", "result_protocol",
        "preexecution_capability_receipt", "execution_authorized",
        "protocol_core_sha256",
    }
    if set(core) != expected_top:
        raise Evaluation400ResultError("paired core top-level adapter contract changed")
    evaluation = core.get("evaluation400")
    deployment = core.get("deployment")
    inventory = core.get("execution_inventory")
    authority = core.get("authority_policy")
    if not isinstance(evaluation, Mapping) or not isinstance(deployment, Mapping) \
       or not isinstance(inventory, Mapping) or not isinstance(authority, Mapping):
        raise Evaluation400ResultError("paired core required sections are missing")
    _require_sha(evaluation.get("pair_identity_set_sha256"), "pair identity set")
    _require_int(evaluation.get("pair_count"), PAIR_COUNT, "pair count")
    _require_bool(evaluation.get("only_final_paired_lane"), True, "only paired lane")
    _require_int(
        evaluation.get("additional_reserve400_count"), 0, "additional reserve count"
    )
    _require_bool(
        evaluation.get("postfreeze_seed_candidate_threshold_or_order_change_allowed"),
        False, "postfreeze changes",
    )
    if deployment.get("candidate_count") != 4 \
       or deployment.get("baseline_selector") != "lowest_legal_feasibility_root_candidate" \
       or deployment.get("etsf_selector") != (
           "frozen_five_member_event_world_model_with_uncertainty_abstention"
       ) or deployment.get("fallback") != "baseline":
        raise Evaluation400ResultError("paired deployment selector contract changed")
    for field in ("deployment_binding_sha256", "policy_runtime_action_binding_sha256"):
        _require_sha(deployment.get(field), f"deployment {field}")
    if authority.get("signature_algorithm") != "Ed25519" \
       or authority.get("core_itself_authorizes_execution") is not False:
        raise Evaluation400ResultError("paired authority contract changed")
    for field in (
        "issuer_public_key_sha256", "issuer_identity_sha256",
        "executor_identity_sha256", "result_evaluator_identity_sha256",
    ):
        _require_sha(authority.get(field), f"authority {field}")
    expected_inventory_fields = {
        "attestation", "stack_binding_sha256", "executor_identity_sha256",
        "executor_implementation_file_sha256", "result_evaluator_identity_sha256",
        "result_evaluator_implementation_file_sha256",
        "real_execution_components_complete",
    }
    if set(inventory) != expected_inventory_fields \
       or inventory.get("real_execution_components_complete") is not True:
        raise Evaluation400ResultError("paired execution inventory changed")
    _record(inventory.get("attestation"), "execution inventory attestation")
    for field in expected_inventory_fields - {"attestation", "real_execution_components_complete"}:
        _require_sha(inventory.get(field), f"execution inventory {field}")
    if inventory.get("executor_identity_sha256") != authority.get("executor_identity_sha256") \
       or inventory.get("result_evaluator_identity_sha256") != authority.get(
           "result_evaluator_identity_sha256"
       ):
        raise Evaluation400ResultError("execution inventory identity binding changed")
    pairs = evaluation.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != PAIR_COUNT:
        raise Evaluation400ResultError("paired core must contain exact 400 pairs")
    pair_fields = {
        "ordinal", "pair_id", "target_manifest_global_ordinal", "requested_seed",
        "resolved_seed", "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256", "condition_order", "candidate_count",
    }
    ids: set[str] = set()
    target_ordinals: set[int] = set()
    result: list[dict[str, Any]] = []
    for ordinal, row in enumerate(pairs):
        if not isinstance(row, Mapping) or set(row) != pair_fields:
            raise Evaluation400ResultError(f"core pair {ordinal} fields changed")
        _require_int(row.get("ordinal"), ordinal, f"core pair {ordinal} ordinal")
        pair_id = row.get("pair_id")
        if not is_sha(pair_id) or pair_id in ids:
            raise Evaluation400ResultError(
                "core pair IDs must be unique lowercase SHA-256"
            )
        ids.add(pair_id)
        target_ordinal = _require_int(
            row.get("target_manifest_global_ordinal"), None,
            f"core pair {ordinal} target ordinal",
        )
        if target_ordinal < 0 or target_ordinal in target_ordinals:
            raise Evaluation400ResultError("target ordinals must be nonnegative and unique")
        target_ordinals.add(target_ordinal)
        for field in ("requested_seed", "resolved_seed"):
            seed = _require_int(row.get(field), None, f"core pair {ordinal} {field}")
            if seed < 0:
                raise Evaluation400ResultError("evaluation seed cannot be negative")
        for field in (
            "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
            "initial_commanded_drive_target_sha256",
        ):
            _require_sha(row.get(field), f"core pair {ordinal} {field}")
        expected_order = paired_v3.bridge_v2.paired_condition_order(pair_id)
        if row.get("condition_order") != expected_order:
            raise Evaluation400ResultError("core condition order differs from preregistration")
        _require_int(row.get("candidate_count"), 4, "core candidate count")
        result.append(dict(row))
    return result


def load_paired_closure(
    *, core_path: Path, core_file_sha256: str,
    decision_path: Path, decision_file_sha256: str,
    bundle_path: Path, bundle_file_sha256: str,
    expected_paired_implementation_file_sha256: str,
) -> tuple[
    Path, dict[str, Any], str, Path, dict[str, Any], str,
    Path, dict[str, Any], str, list[dict[str, Any]],
]:
    _implementation_sha(
        Path(paired_v3.__file__), expected_paired_implementation_file_sha256,
        "reviewed paired-v3 adapter implementation",
    )
    core_bound, core, core_file = read_json(core_path, core_file_sha256, "paired core v3")
    try:
        core_logical = paired_v3.validate_core(core)
    except Exception as error:
        raise Evaluation400ResultError("paired core v3 validation failed") from error
    pairs = _validate_core_pairs(core)
    decision_bound, decision, decision_file = read_json(
        decision_path, decision_file_sha256, "paired Ed25519 decision v3"
    )
    try:
        decision_logical = paired_v3.verify_decision(
            decision, core=core, core_file_sha256=core_file
        )
    except Exception as error:
        raise Evaluation400ResultError("paired Ed25519 decision validation failed") from error
    bundle_bound, bundle, bundle_file = read_json(
        bundle_path, bundle_file_sha256, "paired execution bundle v3"
    )
    try:
        bundle_logical = paired_v3.validate_bundle(bundle)
    except Exception as error:
        raise Evaluation400ResultError("paired execution bundle validation failed") from error
    expected_bundle_fields = {
        "format", "status", "protocol_core", "ed25519_decision", "issuer_key_id",
        "issuer_public_key_sha256", "trusted_issuer_attestation_sha256",
        "executor_identity_sha256", "result_evaluator_identity_sha256",
        "execution_inventory", "pair_identity_set_sha256", "deployment_binding_sha256",
        "authorized_pair_count", "additional_reserve400_count", "external_executor_only",
        "protocol_freezer_may_execute", "execution_authorized", "capability_receipt",
        "bundle_sha256",
    }
    if set(bundle) != expected_bundle_fields:
        raise Evaluation400ResultError("paired bundle adapter contract changed")
    core_record = _record(bundle.get("protocol_core"), "bundle core")
    decision_record = _record(bundle.get("ed25519_decision"), "bundle decision")
    _path_matches(core_record["path"], core_bound, "bundle core")
    _path_matches(decision_record["path"], decision_bound, "bundle decision")
    if core_record != _descriptor(core_bound, core_file, core_logical) \
       or decision_record != _descriptor(decision_bound, decision_file, decision_logical):
        raise Evaluation400ResultError("bundle does not bind the supplied core and decision")
    policy = core["authority_policy"]
    evaluation = core["evaluation400"]
    deployment = core["deployment"]
    if (
        bundle.get("issuer_key_id") != policy.get("issuer_key_id")
        or bundle.get("issuer_public_key_sha256") != policy.get("issuer_public_key_sha256")
        or bundle.get("trusted_issuer_attestation_sha256")
        != policy.get("trusted_issuer_attestation_sha256")
        or bundle.get("executor_identity_sha256") != policy.get("executor_identity_sha256")
        or bundle.get("result_evaluator_identity_sha256")
        != policy.get("result_evaluator_identity_sha256")
        or bundle.get("execution_inventory") != core.get("execution_inventory")
        or bundle.get("pair_identity_set_sha256") != evaluation.get("pair_identity_set_sha256")
        or bundle.get("deployment_binding_sha256") != deployment.get("deployment_binding_sha256")
    ):
        raise Evaluation400ResultError("bundle/core authority binding mismatch")
    return (
        core_bound, core, core_file, decision_bound, decision, decision_file,
        bundle_bound, bundle, bundle_file, pairs,
    )


def _receipt_signing_bytes(receipt_format: str, statement: Mapping[str, Any]) -> bytes:
    return (
        EXECUTOR_SIGNATURE_CONTEXT + receipt_format.encode("ascii") + b"\0"
        + canonical_bytes(statement)
    )


def _verify_executor_receipt(
    value: Mapping[str, Any], *, expected_format: str, expected_status: str,
    public_key: Any, role: str,
) -> tuple[dict[str, Any], str]:
    if set(value) != RECEIPT_FIELDS \
       or value.get("format") != expected_format \
       or value.get("status") != expected_status \
       or value.get("signature_algorithm") != "Ed25519":
        raise Evaluation400ResultError(f"{role} envelope changed")
    unsigned = dict(value)
    logical = unsigned.pop("receipt_sha256", None)
    if not is_sha(logical) or logical != canonical_sha256(unsigned):
        raise Evaluation400ResultError(f"{role} logical SHA mismatch")
    statement = value.get("statement")
    if not isinstance(statement, Mapping):
        raise Evaluation400ResultError(f"{role} statement is missing")
    signature = _decode_hex(
        value.get("executor_signature_ed25519_hex"), 64,
        f"{role} executor signature",
    )
    try:
        public_key.verify(signature, _receipt_signing_bytes(expected_format, statement))
    except Exception as error:
        raise Evaluation400ResultError(f"{role} Ed25519 signature failed") from error
    return dict(statement), str(logical)


def _reset_proof(pair: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "pair_id": pair["pair_id"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "initial_scene_state_sha256": pair["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": pair[
            "initial_measured_joint_state_sha256"
        ],
        "initial_commanded_drive_target_sha256": pair[
            "initial_commanded_drive_target_sha256"
        ],
    })


def _candidate_registry_sha(
    pair_id: str, ordered: Sequence[str], legal: Sequence[bool]
) -> str:
    return canonical_sha256({
        "pair_id": pair_id, "candidate_count": 4,
        "ordered_candidate_sha256": list(ordered), "candidate_legal": list(legal),
    })


def _continuation_proof(statement: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "continuation_contract": statement["continuation_contract"],
        "continuation_policy_sha256": statement["continuation_policy_sha256"],
        "continuation_rerank_after_root": statement["continuation_rerank_after_root"],
        "candidate_replacement_count": statement["candidate_replacement_count"],
    })


def _common_execution_binding(
    statement: Mapping[str, Any], *, core: Mapping[str, Any],
    decision: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_nonce_hex: str, dependency_rehash_sha256: str, role: str,
) -> None:
    expected = {
        "protocol_core_sha256": core["protocol_core_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_nonce_hex": execution_nonce_hex,
        "pair_identity_set_sha256": core["evaluation400"]["pair_identity_set_sha256"],
        "deployment_binding_sha256": core["deployment"]["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": core["deployment"][
            "policy_runtime_action_binding_sha256"
        ],
        "preexecution_dependency_rehash_sha256": dependency_rehash_sha256,
    }
    for field, expected_value in expected.items():
        if statement.get(field) != expected_value:
            raise Evaluation400ResultError(f"{role} binding mismatch: {field}")


CONDITION_STATEMENT_FIELDS = {
    "protocol_core_sha256", "decision_sha256", "bundle_sha256",
    "execution_nonce_hex", "pair_identity_set_sha256", "deployment_binding_sha256",
    "policy_runtime_action_binding_sha256", "preexecution_dependency_rehash_sha256",
    "ledger_condition_start_event_sha256",
    "execution_artifacts",
    "global_condition_ordinal", "pair_ordinal", "pair_id",
    "target_manifest_global_ordinal", "requested_seed", "resolved_seed",
    "condition_position", "condition_id", "condition_order", "attempt_index",
    "retry_count", "condition_started", "condition_terminal", "incomplete",
    "excluded", "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
    "initial_commanded_drive_target_sha256", "reset_proof_sha256", "candidate_count",
    "ordered_candidate_sha256", "candidate_legal", "candidate_registry_sha256",
    "continuation_contract", "continuation_policy_sha256",
    "continuation_rerank_after_root", "candidate_replacement_count",
    "continuation_proof_sha256", "selector", "selected_candidate_ordinal",
    "success", "success_source", "predicted_success_used_as_outcome",
}


def _validate_condition_execution_artifacts(
    value: Any, *, statement: Mapping[str, Any], role: str,
    selector_authority: Mapping[str, Any] | None = None,
    runtime_execution_authority: Mapping[str, Any] | None = None,
    target_reset_runtime_contract_sha256: str | None = None,
) -> set[str]:
    if not isinstance(value, Mapping) or set(value) != EXECUTION_ARTIFACT_FIELDS:
        raise Evaluation400ResultError(f"{role} execution-artifact fields changed")
    gpu_uuid = value.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise Evaluation400ResultError(f"{role} GPU UUID is invalid")

    decoded: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    for name in (
        "runner_result", "stage_launch", "stage_lifecycle",
        "gpu_idle_before", "gpu_idle_after",
    ):
        record = _record(value.get(name), f"{role} {name}")
        bound, payload, file_sha = read_json(
            Path(record["path"]), record["file_sha256"], f"{role} {name}"
        )
        _path_matches(record["path"], bound, f"{role} {name}")
        logical_field = {
            "runner_result": "result_sha256",
            "stage_launch": "launch_sha256",
            "stage_lifecycle": "lifecycle_sha256",
            "gpu_idle_before": "audit_sha256",
            "gpu_idle_after": "audit_sha256",
        }[name]
        logical = _verify_canonical_field(payload, logical_field, f"{role} {name}")
        if file_sha != record["file_sha256"] or logical != record["logical_sha256"]:
            raise Evaluation400ResultError(f"{role} {name} descriptor SHA mismatch")
        decoded[name] = (record, payload)

    launch = decoded["stage_launch"][1]
    lifecycle = decoded["stage_lifecycle"][1]
    runner = decoded["runner_result"][1]
    if (
        not isinstance(runtime_execution_authority, Mapping)
        or set(runtime_execution_authority) != {
            "path", "file_sha256", "nested_runtime_contract_sha256",
            "max_episode_steps",
        }
        or not is_sha(target_reset_runtime_contract_sha256)
    ):
        raise Evaluation400ResultError(f"{role} runtime authority is missing")
    _runtime_path, runtime_authority_value, runtime_authority_file_sha = read_json(
        Path(str(runtime_execution_authority["path"])),
        str(runtime_execution_authority["file_sha256"]),
        f"{role} schema6 runtime execution authority",
    )
    nested_runtime = runtime_authority_value.get("runtime_contract")
    if (
        runtime_authority_file_sha != runtime_execution_authority["file_sha256"]
        or not isinstance(nested_runtime, Mapping)
        or nested_runtime.get("runtime_contract_sha256")
        != target_reset_runtime_contract_sha256
        or nested_runtime.get("runtime_contract_sha256")
        != runtime_execution_authority["nested_runtime_contract_sha256"]
        or type(nested_runtime.get("max_episode_steps")) is not int
        or nested_runtime["max_episode_steps"] != 200
        or type(runtime_execution_authority.get("max_episode_steps")) is not int
        or runtime_execution_authority["max_episode_steps"] != 200
    ):
        raise Evaluation400ResultError(
            f"{role} nested runtime config is not the paired 200-step authority"
        )
    expected_launch_fields = {
        "format", "status", "original_command", "command_sha256",
        "executed_command", "fd_mapping", "isolated_python",
        "environment_policy", "environment_keys",
        "forbidden_environment_keys_absent", "gpu_uuid", "device",
        "launch_sha256",
    }
    original = launch.get("original_command")
    executed = launch.get("executed_command")
    mapping = launch.get("fd_mapping")
    forbidden = {"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"}
    if (
        set(launch) != expected_launch_fields
        or launch.get("format") != STAGE_FORMAT
        or launch.get("status") != "fd_bound_guard_passed_immediately_before_popen"
        or launch.get("isolated_python") is not True
        or launch.get("environment_policy")
        != "explicit_allowlist_no_pythonpath_pythonhome_or_ld_preload"
        or launch.get("gpu_uuid") != gpu_uuid or launch.get("device") != "cuda:0"
        or not isinstance(original, list) or len(original) < 2
        or launch.get("command_sha256") != canonical_sha256(original)
        or not isinstance(executed, list) or len(executed) != len(original) + 1
        or executed[1] != "-I"
        or not isinstance(mapping, list) or len(mapping) != 2
        or launch.get("environment_keys") != EXACT_STAGE_ENVIRONMENT_KEYS
        or forbidden & set(launch["environment_keys"])
        or launch.get("forbidden_environment_keys_absent") != sorted(forbidden)
    ):
        raise Evaluation400ResultError(f"{role} FD-bound launch contract changed")
    for index, (row, expected_role) in enumerate(
        zip(mapping, ("runtime_python", "condition_runner"), strict=True)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "role", "source_path", "source_file_sha256", "inherited_fd",
                "executed_path",
            }
            or row.get("role") != expected_role
            or type(row.get("inherited_fd")) is not int or row["inherited_fd"] < 3
            or row.get("executed_path") != f"/proc/self/fd/{row['inherited_fd']}"
            or executed[0 if index == 0 else 2] != row["executed_path"]
            or original[index] != row.get("source_path")
            or _implementation_sha(
                Path(str(row.get("source_path"))),
                str(row.get("source_file_sha256")),
                f"{role} {expected_role}",
            ) != row.get("source_file_sha256")
        ):
            raise Evaluation400ResultError(f"{role} FD mapping changed")

    expected_lifecycle_fields = {
        "popen_attempted", "popen_reached", "process_pid", "process_pgid",
        "process_group_isolated", "returncode", "direct_process_reaped",
        "process_group_reaped", "binding_status", "lifecycle_sha256",
    }
    if (
        set(lifecycle) != expected_lifecycle_fields
        or lifecycle.get("popen_attempted") is not True
        or lifecycle.get("popen_reached") is not True
        or type(lifecycle.get("process_pid")) is not int
        or type(lifecycle.get("process_pgid")) is not int
        or lifecycle["process_pid"] <= 0
        or lifecycle["process_pid"] != lifecycle["process_pgid"]
        or lifecycle.get("process_group_isolated") is not True
        or type(lifecycle.get("returncode")) is not int
        or lifecycle["returncode"] != 0
        or lifecycle.get("direct_process_reaped") is not True
        or lifecycle.get("process_group_reaped") is not True
        or lifecycle.get("binding_status") != "bound_reaped"
    ):
        raise Evaluation400ResultError(f"{role} process lifecycle is unproven")

    if (
        set(runner) != RUNNER_RESULT_FIELDS
        or runner.get("format") != RUNNER_RESULT_FORMAT
        or runner.get("status") != RUNNER_RESULT_STATUS
        or runner.get("pair_id") != statement.get("pair_id")
        or runner.get("ordinal") != statement.get("pair_ordinal")
        or runner.get("condition") != statement.get("condition_id")
        or runner.get("condition_ordinal") != statement.get("condition_position")
        or type(runner.get("attempt")) is not int or runner["attempt"] != 0
        or type(runner.get("simulator_exit_code")) is not int
        or runner["simulator_exit_code"] != 0
        or type(runner.get("task_success")) is not bool
        or runner["task_success"] is not statement.get("success")
        or runner.get("candidate_registry_sha256")
        != statement.get("candidate_registry_sha256")
        or runner.get("ordered_candidate_sha256")
        != statement.get("ordered_candidate_sha256")
        or runner.get("candidate_legal") != statement.get("candidate_legal")
        or runner.get("selected_candidate_index")
        != statement.get("selected_candidate_ordinal")
        or not is_sha(runner.get("schema6_execution_authority_file_sha256"))
        or not is_sha(runner.get("schema6_runtime_contract_sha256"))
        or type(runner.get("max_episode_steps")) is not int
        or runner["max_episode_steps"] != 200
        or runner.get("schema6_execution_authority_file_sha256")
        != runtime_execution_authority["file_sha256"]
        or runner.get("schema6_runtime_contract_sha256")
        != target_reset_runtime_contract_sha256
        or not isinstance(runner.get("selector_execution_proof"), Mapping)
        or not is_sha(runner.get("selector_execution_proof_sha256"))
        or runner["selector_execution_proof_sha256"]
        != canonical_sha256(runner["selector_execution_proof"])
        or runner["selector_execution_proof"].get("score_contract")
        != runner.get("selector_score_contract")
        or runner["selector_execution_proof"].get(
            "source_rank_score_contract_sha256"
        ) != runner.get("source_rank_score_contract_sha256")
        or runner.get("source_contract_rank_score_is_success_logit") is not False
        or runner.get("source_contract_rank_score_is_success_probability") is not False
        or runner["selector_execution_proof"].get(
            "formal190_target_outcome_calibrated_acceptance_margin"
        ) is not runner.get("formal190_target_outcome_calibrated_acceptance_margin")
    ):
        raise Evaluation400ResultError(f"{role} runner result binding changed")
    validate_selector_execution_proof(
        runner["selector_execution_proof"],
        condition=str(runner["condition"]),
        candidate_legal=runner["candidate_legal"],
        selected_candidate_index=runner["selected_candidate_index"],
    )
    if selector_authority is not None:
        proof = runner["selector_execution_proof"]
        if (
            not isinstance(selector_authority, Mapping)
            or runner.get("schema6_execution_authority_file_sha256")
            != selector_authority.get("runtime_execution_authority_sha256")
        ):
            raise Evaluation400ResultError(
                f"{role} runtime/selector authority binding changed"
            )
        if runner["condition"] == "etsf" and (
            proof.get("calibration_sha256")
            != selector_authority.get("calibration_sha256")
            or proof.get("formal190_root_group_ranker_sha256")
            != selector_authority.get("formal190_root_group_ranker_sha256")
            or proof.get("source_rank_score_contract_sha256")
            != selector_authority.get("source_rank_score_contract_sha256")
            or proof.get("selector_decision", {}).get("selector_input", {}).get(
                "deployment_uncertainty_implementation"
            ) != selector_authority.get("deployment_uncertainty_implementation")
        ):
            raise Evaluation400ResultError(
                f"{role} ETSF proof differs from paired selector authority"
            )
        if runner["condition"] == "etsf":
            validate_selector_proof_against_authority(
                proof, selector_authority
            )
    request_record = runner.get("request")
    if not isinstance(request_record, Mapping) or set(request_record) != RECORD_FIELDS:
        raise Evaluation400ResultError(f"{role} runner request descriptor changed")
    request_bound, request, request_file_sha = read_json(
        Path(str(request_record["path"])), request_record["file_sha256"],
        f"{role} runner request",
    )
    _path_matches(request_record["path"], request_bound, f"{role} runner request")
    request_logical = _verify_canonical_field(
        request, "request_sha256", f"{role} runner request"
    )
    if (
        set(request) != CONDITION_REQUEST_FIELDS
        or request_file_sha != request_record["file_sha256"]
        or request_logical != request_record["logical_sha256"]
        or request.get("pair_id") != statement.get("pair_id")
        or request.get("ordinal") != statement.get("pair_ordinal")
        or request.get("condition") != statement.get("condition_id")
        or request.get("condition_ordinal") != statement.get("condition_position")
        or type(request.get("attempt")) is not int or request["attempt"] != 0
        or request.get("outcome_visible_before_condition_start") is not False
        or request.get("postfreeze_identity_or_order_change_authorized") is not False
    ):
        raise Evaluation400ResultError(f"{role} runner request binding changed")
    flag_values: dict[str, str] = {}
    for index, token in enumerate(original[:-1]):
        if isinstance(token, str) and token.startswith("--"):
            flag_values[token] = str(original[index + 1])
    if (
        flag_values.get("--request") != request_record.get("path")
        or flag_values.get("--output-root")
        != str(Path(decoded["runner_result"][0]["path"]).parent)
        or flag_values.get("--device") != "cuda:0"
    ):
        raise Evaluation400ResultError(f"{role} launch/runner path binding changed")

    for name, expected in (("gpu_idle_before", gpu_uuid), ("gpu_idle_after", gpu_uuid)):
        idle = decoded[name][1]
        base = dict(idle)
        base.pop("audit_sha256", None)
        if (
            set(idle) != {"gpu_index", "gpu_name", "gpu_uuid", "checks", "audit_sha256"}
            or idle.get("gpu_uuid") != expected
            or type(idle.get("gpu_index")) is not int or idle["gpu_index"] < 0
            or not isinstance(idle.get("gpu_name"), str) or "4090" not in idle["gpu_name"]
            or type(idle.get("checks")) is not int or idle["checks"] != 2
            or idle.get("audit_sha256") != canonical_sha256(base)
        ):
            raise Evaluation400ResultError(f"{role} {name} contract changed")

    log_record = _opaque_record(value.get("stage_log"), f"{role} stage log")
    exit_record = _opaque_record(value.get("stage_exit"), f"{role} stage exit")
    _log_path, _log_payload, _log_sha = secure_read(
        Path(log_record["path"]), log_record["file_sha256"], f"{role} stage log"
    )
    _exit_path, exit_payload, _exit_sha = secure_read(
        Path(exit_record["path"]), exit_record["file_sha256"], f"{role} stage exit"
    )
    if exit_payload != b"0\n":
        raise Evaluation400ResultError(f"{role} stage exit is not exact zero")

    runner_path = Path(decoded["runner_result"][0]["path"])
    stage_root = Path(decoded["stage_launch"][0]["path"]).parent
    artifact_paths = {
        *(str(Path(decoded[name][0]["path"])) for name in decoded),
        str(Path(log_record["path"])),
        str(Path(exit_record["path"])),
        str(Path(request_record["path"])),
    }
    if (
        len(artifact_paths) != 8
        or runner_path != stage_root / "result" / "condition_result.json"
        or Path(decoded["stage_launch"][0]["path"]) != stage_root / "launch.json"
        or Path(decoded["stage_lifecycle"][0]["path"])
        != stage_root / "lifecycle.json"
        or Path(log_record["path"]) != stage_root / "run.log"
        or Path(exit_record["path"]) != stage_root / "run.exit"
        or Path(decoded["gpu_idle_before"][0]["path"]).parent
        != stage_root.parent
        or Path(decoded["gpu_idle_after"][0]["path"]).parent
        != stage_root.parent
        or Path(decoded["gpu_idle_before"][0]["path"]).name
        != "gpu_idle_before_condition.json"
        or Path(decoded["gpu_idle_after"][0]["path"]).name
        != "gpu_idle_after_condition.json"
        or Path(request_record["path"]).name != "request.json"
    ):
        raise Evaluation400ResultError(f"{role} execution-artifact layout changed")
    before = decoded["gpu_idle_before"][1]
    after = decoded["gpu_idle_after"][1]
    if any(
        before[field] != after[field]
        for field in ("gpu_index", "gpu_name", "gpu_uuid")
    ):
        raise Evaluation400ResultError(
            f"{role} GPU identity changed within condition"
        )
    return artifact_paths


def _validate_condition_statement(
    statement: Mapping[str, Any], *, pair: Mapping[str, Any], position: int,
    core: Mapping[str, Any], decision: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_nonce_hex: str, dependency_rehash_sha256: str,
) -> set[str]:
    ordinal = pair["ordinal"]
    role = f"condition {2 * ordinal + position}"
    if set(statement) != CONDITION_STATEMENT_FIELDS:
        raise Evaluation400ResultError(f"{role} statement fields changed")
    _common_execution_binding(
        statement, core=core, decision=decision, bundle=bundle,
        execution_nonce_hex=execution_nonce_hex,
        dependency_rehash_sha256=dependency_rehash_sha256, role=role,
    )
    _require_sha(
        statement.get("ledger_condition_start_event_sha256"),
        f"{role} ledger start event",
    )
    _require_int(statement.get("global_condition_ordinal"), 2 * ordinal + position, role)
    _require_int(statement.get("pair_ordinal"), ordinal, role)
    for field in (
        "pair_id", "target_manifest_global_ordinal", "requested_seed", "resolved_seed",
        "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256", "condition_order",
    ):
        if statement.get(field) != pair[field]:
            raise Evaluation400ResultError(f"{role} differs from core pair: {field}")
    _require_int(statement.get("condition_position"), position, role)
    condition_id = pair["condition_order"][position]
    if statement.get("condition_id") != condition_id:
        raise Evaluation400ResultError(f"{role} condition order changed")
    _require_int(statement.get("attempt_index"), 1, f"{role} attempt")
    _require_int(statement.get("retry_count"), 0, f"{role} retry")
    for field, expected in (
        ("condition_started", True), ("condition_terminal", True),
        ("incomplete", False), ("excluded", False),
        ("continuation_rerank_after_root", False),
        ("predicted_success_used_as_outcome", False),
    ):
        _require_bool(statement.get(field), expected, f"{role} {field}")
    if statement.get("reset_proof_sha256") != _reset_proof(pair):
        raise Evaluation400ResultError(f"{role} reset proof mismatch")
    _require_int(statement.get("candidate_count"), 4, f"{role} candidate count")
    ordered = statement.get("ordered_candidate_sha256")
    legal = statement.get("candidate_legal")
    if not isinstance(ordered, list) or len(ordered) != 4 \
       or any(not is_sha(value) for value in ordered) or len(set(ordered)) != 4:
        raise Evaluation400ResultError(f"{role} candidate registry is invalid")
    if not isinstance(legal, list) or len(legal) != 4 \
       or any(type(value) is not bool for value in legal) or not any(legal):
        raise Evaluation400ResultError(f"{role} candidate legality is invalid")
    if statement.get("candidate_registry_sha256") != _candidate_registry_sha(
        pair["pair_id"], ordered, legal
    ):
        raise Evaluation400ResultError(f"{role} candidate registry SHA mismatch")
    if statement.get("continuation_contract") != CONTINUATION_CONTRACT:
        raise Evaluation400ResultError(f"{role} continuation contract changed")
    _require_sha(statement.get("continuation_policy_sha256"), f"{role} continuation policy")
    _require_int(
        statement.get("candidate_replacement_count"), 0,
        f"{role} candidate replacement count",
    )
    if statement.get("continuation_proof_sha256") != _continuation_proof(statement):
        raise Evaluation400ResultError(f"{role} continuation proof mismatch")
    selected = _require_int(
        statement.get("selected_candidate_ordinal"), None,
        f"{role} selected candidate",
    )
    if not 0 <= selected < 4 or legal[selected] is not True:
        raise Evaluation400ResultError(f"{role} selected an illegal candidate")
    expected_selector = (
        core["deployment"]["baseline_selector"] if condition_id == "baseline"
        else core["deployment"]["etsf_selector"]
    )
    if statement.get("selector") != expected_selector:
        raise Evaluation400ResultError(f"{role} selector changed")
    if condition_id == "baseline" and selected != legal.index(True):
        raise Evaluation400ResultError("baseline did not select the lowest legal candidate")
    if type(statement.get("success")) is not bool:
        raise Evaluation400ResultError(f"{role} success must be exact boolean")
    if statement.get("success_source") != SUCCESS_SOURCE:
        raise Evaluation400ResultError(f"{role} success source changed")
    selector_authority = core.get("deployment", {}).get("selector_authority")
    if not isinstance(selector_authority, Mapping):
        raise Evaluation400ResultError(
            f"{role} paired selector authority is missing"
        )
    return _validate_condition_execution_artifacts(
        statement.get("execution_artifacts"), statement=statement, role=role,
        selector_authority=selector_authority,
        runtime_execution_authority=core["deployment"].get(
            "runtime_execution_authority"
        ),
        target_reset_runtime_contract_sha256=core["deployment"].get(
            "target_reset_runtime_contract_sha256"
        ),
    )


PAIR_STATEMENT_FIELDS = {
    "protocol_core_sha256", "decision_sha256", "bundle_sha256",
    "execution_nonce_hex", "pair_identity_set_sha256", "deployment_binding_sha256",
    "policy_runtime_action_binding_sha256", "preexecution_dependency_rehash_sha256",
    "ordinal", "pair_id", "target_manifest_global_ordinal", "requested_seed",
    "resolved_seed", "condition_order", "condition_receipts", "reset_proof_sha256",
    "candidate_registry_sha256", "continuation_proof_sha256",
    "ledger_condition_terminal_event_sha256",
    "condition_attempt_count", "complete_condition_count", "retry_count",
    "incomplete", "excluded", "baseline_success", "etsf_success", "success_source",
}


def _validate_pair_statement(
    statement: Mapping[str, Any], *, pair: Mapping[str, Any],
    conditions: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    core: Mapping[str, Any], decision: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_nonce_hex: str, dependency_rehash_sha256: str,
) -> None:
    ordinal = pair["ordinal"]
    role = f"pair {ordinal}"
    if set(statement) != PAIR_STATEMENT_FIELDS:
        raise Evaluation400ResultError(f"{role} statement fields changed")
    _common_execution_binding(
        statement, core=core, decision=decision, bundle=bundle,
        execution_nonce_hex=execution_nonce_hex,
        dependency_rehash_sha256=dependency_rehash_sha256, role=role,
    )
    for field in (
        "ordinal", "pair_id", "target_manifest_global_ordinal", "requested_seed",
        "resolved_seed", "condition_order",
    ):
        if statement.get(field) != pair[field]:
            raise Evaluation400ResultError(f"{role} differs from core: {field}")
    condition_records = statement.get("condition_receipts")
    if not isinstance(condition_records, list) or len(condition_records) != 2:
        raise Evaluation400ResultError(f"{role} must bind exactly two conditions")
    expected_records = [item[0] for item in conditions]
    if condition_records != expected_records:
        raise Evaluation400ResultError(f"{role} condition descriptors changed")
    ledger_condition_terminal = statement.get(
        "ledger_condition_terminal_event_sha256"
    )
    if (
        not isinstance(ledger_condition_terminal, list)
        or len(ledger_condition_terminal) != 2
        or any(not is_sha(value) for value in ledger_condition_terminal)
    ):
        raise Evaluation400ResultError(
            f"{role} condition-terminal ledger bindings changed"
        )
    condition_statements = [item[1] for item in conditions]
    first = condition_statements[0]
    second = condition_statements[1]
    for field in (
        "reset_proof_sha256", "candidate_registry_sha256", "continuation_proof_sha256",
        "ordered_candidate_sha256", "candidate_legal", "continuation_contract",
        "continuation_policy_sha256",
    ):
        if first.get(field) != second.get(field):
            raise Evaluation400ResultError(f"{role} conditions differ at shared root: {field}")
    for field in ("reset_proof_sha256", "candidate_registry_sha256", "continuation_proof_sha256"):
        if statement.get(field) != first.get(field):
            raise Evaluation400ResultError(f"{role} shared proof mismatch: {field}")
    _require_int(statement.get("condition_attempt_count"), 2, role)
    _require_int(statement.get("complete_condition_count"), 2, role)
    _require_int(statement.get("retry_count"), 0, role)
    _require_bool(statement.get("incomplete"), False, role)
    _require_bool(statement.get("excluded"), False, role)
    if statement.get("success_source") != SUCCESS_SOURCE:
        raise Evaluation400ResultError(f"{role} success source changed")
    by_id = {row["condition_id"]: row for row in condition_statements}
    if set(by_id) != {"baseline", "etsf"}:
        raise Evaluation400ResultError(f"{role} condition coverage changed")
    for field, condition_id in (
        ("baseline_success", "baseline"), ("etsf_success", "etsf")
    ):
        if type(statement.get(field)) is not bool \
           or statement[field] is not by_id[condition_id]["success"]:
            raise Evaluation400ResultError(f"{role} success mismatch: {field}")


TERMINAL_STATEMENT_FIELDS = {
    "protocol_core", "ed25519_decision", "execution_bundle", "executor_key_id",
    "executor_public_key_hex", "executor_public_key_sha256",
    "executor_identity_sha256", "execution_nonce_hex", "pair_identity_set_sha256",
    "deployment_binding_sha256", "policy_runtime_action_binding_sha256",
    "preexecution_dependency_rehash_sha256",
    "full_dependency_rehash_after_claim_before_first_condition_started",
    "bootstrap_draws", "bootstrap_format", "bootstrap_shape", "bootstrap_seed",
    "bootstrap_generator", "bootstrap_frozen_before_first_condition_started",
    "execution_claim", "ledger_contract", "ledger_events",
    "pair_receipts", "condition_receipts", "execution_complete",
    "subset_statistics_authorized", "performance_claim_authorized_by_executor",
}


LEDGER_FIELDS = {
    "format", "terminal_state", "ledger_id_sha256", "claim_receipt_sha256",
    "final_event_sha256", "event_count", "claim_count", "claim_release_count",
    "execution_attempt_count", "condition_attempt_count", "retry_count",
    "selective_rerun_count", "pair_exclusion_count", "condition_exclusion_count",
    "incomplete_pair_count", "incomplete_condition_count", "complete_pair_count",
    "complete_condition_count", "claim_before_outcome_read", "one_shot_consumed",
}

CLAIM_STATEMENT_FIELDS = {
    "protocol_core_sha256", "decision_sha256", "bundle_sha256",
    "execution_nonce_hex", "pair_identity_set_sha256", "deployment_binding_sha256",
    "policy_runtime_action_binding_sha256", "ledger_id_sha256", "claim_ordinal",
    "claim_count", "claim_release_count", "claimed_before_any_outcome_read",
    "outcome_or_success_values_read_before_claim", "retry_or_reclaim_authorized",
}

LEDGER_EVENT_STATEMENT_FIELDS = {
    "protocol_core_sha256", "decision_sha256", "bundle_sha256",
    "execution_nonce_hex", "pair_identity_set_sha256", "deployment_binding_sha256",
    "policy_runtime_action_binding_sha256", "preexecution_dependency_rehash_sha256",
    "ledger_id_sha256", "event_index", "event_type", "previous_entry_sha256",
    "pair_ordinal", "global_condition_ordinal", "condition_position", "condition_id",
    "artifact_receipt_sha256", "outcome_or_success_read_before_event",
}

LEDGER_EVENT_COUNT = PAIR_COUNT * 5 + 1


def _validate_terminal_statement(
    statement: Mapping[str, Any], *, core_record: Mapping[str, str],
    decision_record: Mapping[str, str], bundle_record: Mapping[str, str],
    core: Mapping[str, Any], bootstrap_path: Path, bootstrap_file_sha256: str,
) -> tuple[
    str, str, dict[str, str], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    if set(statement) != TERMINAL_STATEMENT_FIELDS:
        raise Evaluation400ResultError("execution terminal statement fields changed")
    if statement.get("protocol_core") != core_record \
       or statement.get("ed25519_decision") != decision_record \
       or statement.get("execution_bundle") != bundle_record:
        raise Evaluation400ResultError("execution terminal paired closure mismatch")
    public_bytes = _decode_hex(
        statement.get("executor_public_key_hex"), 32, "executor public key"
    )
    public_sha = hashlib.sha256(public_bytes).hexdigest()
    if statement.get("executor_public_key_sha256") != public_sha \
       or statement.get("executor_identity_sha256") != public_sha \
       or core["authority_policy"].get("executor_identity_sha256") != public_sha:
        raise Evaluation400ResultError(
            "executor public key is not the identity frozen in paired core"
        )
    if not isinstance(statement.get("executor_key_id"), str) \
       or not statement["executor_key_id"]:
        raise Evaluation400ResultError("executor key ID is missing")
    execution_nonce = statement.get("execution_nonce_hex")
    _decode_hex(execution_nonce, 32, "execution nonce")
    for field, expected in (
        ("pair_identity_set_sha256", core["evaluation400"]["pair_identity_set_sha256"]),
        ("deployment_binding_sha256", core["deployment"]["deployment_binding_sha256"]),
        ("policy_runtime_action_binding_sha256", core["deployment"]["policy_runtime_action_binding_sha256"]),
    ):
        if statement.get(field) != expected:
            raise Evaluation400ResultError(f"terminal binding mismatch: {field}")
    dependency_rehash = _require_sha(
        statement.get("preexecution_dependency_rehash_sha256"),
        "preexecution dependency rehash",
    )
    _require_bool(
        statement.get("full_dependency_rehash_after_claim_before_first_condition_started"),
        True, "terminal dependency rehash ordering",
    )
    bootstrap = _record(statement.get("bootstrap_draws"), "terminal bootstrap draws")
    _path_matches(bootstrap["path"], bootstrap_path, "terminal bootstrap draws")
    if bootstrap["file_sha256"] != bootstrap_file_sha256:
        raise Evaluation400ResultError("terminal bootstrap file SHA mismatch")
    if bootstrap["logical_sha256"] != canonical_sha256({
        "format": BOOTSTRAP_FORMAT, "shape": list(BOOTSTRAP_SHAPE),
        "seed": BOOTSTRAP_SEED, "generator": BOOTSTRAP_GENERATOR,
        "file_sha256": bootstrap_file_sha256,
    }):
        raise Evaluation400ResultError("terminal bootstrap logical SHA mismatch")
    if statement.get("bootstrap_format") != BOOTSTRAP_FORMAT \
       or statement.get("bootstrap_shape") != list(BOOTSTRAP_SHAPE) \
       or statement.get("bootstrap_generator") != BOOTSTRAP_GENERATOR:
        raise Evaluation400ResultError("terminal bootstrap contract changed")
    _require_int(statement.get("bootstrap_seed"), BOOTSTRAP_SEED, "bootstrap seed")
    _require_bool(
        statement.get("bootstrap_frozen_before_first_condition_started"), True,
        "bootstrap pre-outcome freeze",
    )
    claim_record = _record(statement.get("execution_claim"), "execution claim")
    ledger = statement.get("ledger_contract")
    if not isinstance(ledger, Mapping) or set(ledger) != LEDGER_FIELDS \
       or ledger.get("format") != LEDGER_FORMAT \
       or ledger.get("terminal_state") != LEDGER_FINAL_STATE:
        raise Evaluation400ResultError("one-shot ledger terminal contract changed")
    for field in ("ledger_id_sha256", "claim_receipt_sha256", "final_event_sha256"):
        _require_sha(ledger.get(field), f"ledger {field}")
    exact_counts = {
        "event_count": LEDGER_EVENT_COUNT,
        "claim_count": 1, "claim_release_count": 0, "execution_attempt_count": 1,
        "condition_attempt_count": CONDITION_COUNT, "retry_count": 0,
        "selective_rerun_count": 0, "pair_exclusion_count": 0,
        "condition_exclusion_count": 0, "incomplete_pair_count": 0,
        "incomplete_condition_count": 0, "complete_pair_count": PAIR_COUNT,
        "complete_condition_count": CONDITION_COUNT,
    }
    for field, expected in exact_counts.items():
        _require_int(ledger.get(field), expected, f"ledger {field}")
    _require_bool(ledger.get("claim_before_outcome_read"), True, "ledger claim ordering")
    _require_bool(ledger.get("one_shot_consumed"), True, "ledger consumption")
    if ledger.get("claim_receipt_sha256") != claim_record["logical_sha256"]:
        raise Evaluation400ResultError("ledger/claim receipt binding mismatch")
    _require_bool(statement.get("execution_complete"), True, "execution complete")
    _require_bool(statement.get("subset_statistics_authorized"), False, "subset statistics")
    _require_bool(
        statement.get("performance_claim_authorized_by_executor"), False,
        "executor performance claim",
    )
    pair_records = statement.get("pair_receipts")
    condition_records = statement.get("condition_receipts")
    ledger_event_records = statement.get("ledger_events")
    if not isinstance(pair_records, list) or len(pair_records) != PAIR_COUNT \
       or not isinstance(condition_records, list) or len(condition_records) != CONDITION_COUNT \
       or not isinstance(ledger_event_records, list) \
       or len(ledger_event_records) != LEDGER_EVENT_COUNT:
        raise Evaluation400ResultError(
            "terminal receipt/ledger coverage is not exact 400/800/2001"
        )
    return (
        str(execution_nonce), dependency_rehash, claim_record,
        ledger_event_records, pair_records, condition_records,
    )


def _condition_terminal_record(value: Any, ordinal: int, pair: Mapping[str, Any], position: int) -> dict[str, Any]:
    fields = {
        "global_condition_ordinal", "pair_ordinal", "condition_position",
        "condition_id", "path", "file_sha256", "logical_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Evaluation400ResultError(f"terminal condition descriptor {ordinal} changed")
    _require_int(value.get("global_condition_ordinal"), ordinal, "condition descriptor")
    _require_int(value.get("pair_ordinal"), pair["ordinal"], "condition descriptor")
    _require_int(value.get("condition_position"), position, "condition descriptor")
    if value.get("condition_id") != pair["condition_order"][position]:
        raise Evaluation400ResultError("terminal condition descriptor order changed")
    _record({key: value[key] for key in RECORD_FIELDS}, "condition receipt")
    return dict(value)


def _pair_terminal_record(value: Any, ordinal: int, pair: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"ordinal", "pair_id", "path", "file_sha256", "logical_sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Evaluation400ResultError(f"terminal pair descriptor {ordinal} changed")
    _require_int(value.get("ordinal"), ordinal, "pair descriptor")
    if value.get("pair_id") != pair["pair_id"]:
        raise Evaluation400ResultError("terminal pair descriptor identity changed")
    _record({key: value[key] for key in RECORD_FIELDS}, "pair receipt")
    return dict(value)


def _validate_claim_statement(
    statement: Mapping[str, Any], *, core: Mapping[str, Any],
    decision: Mapping[str, Any], bundle: Mapping[str, Any], execution_nonce_hex: str,
) -> str:
    if set(statement) != CLAIM_STATEMENT_FIELDS:
        raise Evaluation400ResultError("execution claim statement fields changed")
    expected = {
        "protocol_core_sha256": core["protocol_core_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_nonce_hex": execution_nonce_hex,
        "pair_identity_set_sha256": core["evaluation400"]["pair_identity_set_sha256"],
        "deployment_binding_sha256": core["deployment"]["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": core["deployment"][
            "policy_runtime_action_binding_sha256"
        ],
    }
    for field, expected_value in expected.items():
        if statement.get(field) != expected_value:
            raise Evaluation400ResultError(f"execution claim binding mismatch: {field}")
    ledger_id = _require_sha(statement.get("ledger_id_sha256"), "claim ledger ID")
    _require_int(statement.get("claim_ordinal"), 0, "claim ordinal")
    _require_int(statement.get("claim_count"), 1, "claim count")
    _require_int(statement.get("claim_release_count"), 0, "claim release count")
    _require_int(
        statement.get("outcome_or_success_values_read_before_claim"), 0,
        "preclaim outcome reads",
    )
    _require_bool(
        statement.get("claimed_before_any_outcome_read"), True,
        "claim before outcome",
    )
    _require_bool(
        statement.get("retry_or_reclaim_authorized"), False,
        "claim retry authority",
    )
    return ledger_id


def _ledger_event_record(
    value: Any, *, event_index: int, expected_type: str,
    pair_ordinal: int | None, global_condition_ordinal: int | None,
) -> dict[str, Any]:
    fields = {
        "event_index", "event_type", "pair_ordinal", "global_condition_ordinal",
        "path", "file_sha256", "logical_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Evaluation400ResultError(f"ledger event descriptor {event_index} changed")
    _require_int(value.get("event_index"), event_index, "ledger event index")
    if value.get("event_type") != expected_type \
       or value.get("pair_ordinal") != pair_ordinal \
       or value.get("global_condition_ordinal") != global_condition_ordinal:
        raise Evaluation400ResultError(f"ledger event descriptor {event_index} order changed")
    _record({key: value[key] for key in RECORD_FIELDS}, "ledger event receipt")
    return dict(value)


def _validate_ledger_event_statement(
    statement: Mapping[str, Any], *, event_index: int, expected_type: str,
    pair: Mapping[str, Any] | None, position: int | None,
    previous_entry_sha256: str, artifact_receipt_sha256: str | None,
    core: Mapping[str, Any], decision: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_nonce_hex: str, dependency_rehash_sha256: str, ledger_id_sha256: str,
) -> None:
    if set(statement) != LEDGER_EVENT_STATEMENT_FIELDS:
        raise Evaluation400ResultError(f"ledger event {event_index} fields changed")
    _common_execution_binding(
        statement, core=core, decision=decision, bundle=bundle,
        execution_nonce_hex=execution_nonce_hex,
        dependency_rehash_sha256=dependency_rehash_sha256,
        role=f"ledger event {event_index}",
    )
    if statement.get("ledger_id_sha256") != ledger_id_sha256:
        raise Evaluation400ResultError(f"ledger event {event_index} ledger ID changed")
    _require_int(statement.get("event_index"), event_index, "ledger event index")
    if statement.get("event_type") != expected_type \
       or statement.get("previous_entry_sha256") != previous_entry_sha256 \
       or statement.get("artifact_receipt_sha256") != artifact_receipt_sha256:
        raise Evaluation400ResultError(f"ledger event {event_index} hash chain changed")
    if pair is None:
        expected_pair = expected_global = expected_position = expected_condition = None
    elif position is None:
        expected_pair = pair["ordinal"]
        expected_global = expected_position = expected_condition = None
    else:
        expected_pair = pair["ordinal"]
        expected_global = 2 * pair["ordinal"] + position
        expected_position = position
        expected_condition = pair["condition_order"][position]
    if (
        statement.get("pair_ordinal") != expected_pair
        or statement.get("global_condition_ordinal") != expected_global
        or statement.get("condition_position") != expected_position
        or statement.get("condition_id") != expected_condition
    ):
        raise Evaluation400ResultError(f"ledger event {event_index} coverage changed")
    before_outcome = expected_type == "condition_started_preoutcome"
    _require_bool(
        statement.get("outcome_or_success_read_before_event"),
        not before_outcome,
        f"ledger event {event_index} outcome ordering",
    )


def validate_execution_ledger(
    *, claim_record: Mapping[str, str], event_records: Sequence[Mapping[str, Any]],
    condition_items: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pair_items: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ledger_contract: Mapping[str, Any], core: Mapping[str, Any],
    decision: Mapping[str, Any], bundle: Mapping[str, Any],
    execution_nonce_hex: str, dependency_rehash_sha256: str, public_key: Any,
) -> dict[str, str]:
    """Open and verify the claim and exact write-ahead hash-chain closure."""

    claim_bound, claim_receipt, claim_file = read_json(
        Path(claim_record["path"]), claim_record["file_sha256"], "execution claim"
    )
    _path_matches(claim_record["path"], claim_bound, "execution claim")
    claim_statement, claim_logical = _verify_executor_receipt(
        claim_receipt, expected_format=CLAIM_FORMAT, expected_status=CLAIM_STATUS,
        public_key=public_key, role="execution claim",
    )
    if claim_file != claim_record["file_sha256"] \
       or claim_logical != claim_record["logical_sha256"]:
        raise Evaluation400ResultError("execution claim descriptor SHA mismatch")
    ledger_id = _validate_claim_statement(
        claim_statement, core=core, decision=decision, bundle=bundle,
        execution_nonce_hex=execution_nonce_hex,
    )
    if ledger_contract.get("ledger_id_sha256") != ledger_id \
       or ledger_contract.get("claim_receipt_sha256") != claim_logical:
        raise Evaluation400ResultError("ledger contract does not bind exact claim")
    if len(event_records) != LEDGER_EVENT_COUNT \
       or len(condition_items) != CONDITION_COUNT or len(pair_items) != PAIR_COUNT:
        raise Evaluation400ResultError("ledger closure is not exact 400/800")
    seen_paths = {claim_record["path"]}
    previous = claim_logical
    final_logical = ""
    event_cursor = 0
    for pair_ordinal in range(PAIR_COUNT):
        pair = core["evaluation400"]["pairs"][pair_ordinal]
        condition_terminal_shas: list[str] = []
        for position in range(2):
            global_ordinal = 2 * pair_ordinal + position
            condition_record, condition_statement = condition_items[global_ordinal]
            for event_type, artifact in (
                ("condition_started_preoutcome", None),
                ("condition_terminal", condition_record["logical_sha256"]),
            ):
                raw_record = event_records[event_cursor]
                record = _ledger_event_record(
                    raw_record, event_index=event_cursor, expected_type=event_type,
                    pair_ordinal=pair_ordinal,
                    global_condition_ordinal=global_ordinal,
                )
                if record["path"] in seen_paths:
                    raise Evaluation400ResultError("ledger receipt path reused")
                seen_paths.add(record["path"])
                bound, receipt, file_sha = read_json(
                    Path(record["path"]), record["file_sha256"],
                    f"ledger event {event_cursor}",
                )
                _path_matches(record["path"], bound, f"ledger event {event_cursor}")
                event_statement, logical = _verify_executor_receipt(
                    receipt, expected_format=LEDGER_EVENT_FORMAT,
                    expected_status=LEDGER_EVENT_STATUS, public_key=public_key,
                    role=f"ledger event {event_cursor}",
                )
                if file_sha != record["file_sha256"] \
                   or logical != record["logical_sha256"]:
                    raise Evaluation400ResultError("ledger event descriptor SHA mismatch")
                _validate_ledger_event_statement(
                    event_statement, event_index=event_cursor,
                    expected_type=event_type, pair=pair, position=position,
                    previous_entry_sha256=previous,
                    artifact_receipt_sha256=artifact,
                    core=core, decision=decision, bundle=bundle,
                    execution_nonce_hex=execution_nonce_hex,
                    dependency_rehash_sha256=dependency_rehash_sha256,
                    ledger_id_sha256=ledger_id,
                )
                if event_type == "condition_started_preoutcome":
                    if condition_statement.get(
                        "ledger_condition_start_event_sha256"
                    ) != logical:
                        raise Evaluation400ResultError(
                            "condition receipt does not bind its preoutcome start event"
                        )
                else:
                    condition_terminal_shas.append(logical)
                previous = logical
                final_logical = logical
                event_cursor += 1
        pair_record, pair_statement = pair_items[pair_ordinal]
        raw_record = event_records[event_cursor]
        record = _ledger_event_record(
            raw_record, event_index=event_cursor, expected_type="pair_terminal",
            pair_ordinal=pair_ordinal, global_condition_ordinal=None,
        )
        if record["path"] in seen_paths:
            raise Evaluation400ResultError("ledger receipt path reused")
        seen_paths.add(record["path"])
        bound, receipt, file_sha = read_json(
            Path(record["path"]), record["file_sha256"],
            f"ledger event {event_cursor}",
        )
        _path_matches(record["path"], bound, f"ledger event {event_cursor}")
        event_statement, logical = _verify_executor_receipt(
            receipt, expected_format=LEDGER_EVENT_FORMAT,
            expected_status=LEDGER_EVENT_STATUS, public_key=public_key,
            role=f"ledger event {event_cursor}",
        )
        if file_sha != record["file_sha256"] or logical != record["logical_sha256"]:
            raise Evaluation400ResultError("ledger pair-terminal descriptor SHA mismatch")
        _validate_ledger_event_statement(
            event_statement, event_index=event_cursor, expected_type="pair_terminal",
            pair=pair, position=None, previous_entry_sha256=previous,
            artifact_receipt_sha256=pair_record["logical_sha256"], core=core,
            decision=decision, bundle=bundle, execution_nonce_hex=execution_nonce_hex,
            dependency_rehash_sha256=dependency_rehash_sha256,
            ledger_id_sha256=ledger_id,
        )
        if pair_statement.get(
            "ledger_condition_terminal_event_sha256"
        ) != condition_terminal_shas:
            raise Evaluation400ResultError(
                "pair receipt does not bind both condition-terminal events"
            )
        previous = logical
        final_logical = logical
        event_cursor += 1
    record = _ledger_event_record(
        event_records[event_cursor], event_index=event_cursor,
        expected_type="execution_terminal", pair_ordinal=None,
        global_condition_ordinal=None,
    )
    if record["path"] in seen_paths:
        raise Evaluation400ResultError("ledger final receipt path reused")
    bound, receipt, file_sha = read_json(
        Path(record["path"]), record["file_sha256"], "ledger final event"
    )
    _path_matches(record["path"], bound, "ledger final event")
    final_statement, final_logical = _verify_executor_receipt(
        receipt, expected_format=LEDGER_EVENT_FORMAT,
        expected_status=LEDGER_EVENT_STATUS, public_key=public_key,
        role="ledger final event",
    )
    if file_sha != record["file_sha256"] or final_logical != record["logical_sha256"]:
        raise Evaluation400ResultError("ledger final event descriptor SHA mismatch")
    _validate_ledger_event_statement(
        final_statement, event_index=event_cursor,
        expected_type="execution_terminal", pair=None, position=None,
        previous_entry_sha256=previous, artifact_receipt_sha256=None,
        core=core, decision=decision, bundle=bundle,
        execution_nonce_hex=execution_nonce_hex,
        dependency_rehash_sha256=dependency_rehash_sha256,
        ledger_id_sha256=ledger_id,
    )
    if event_cursor + 1 != LEDGER_EVENT_COUNT \
       or ledger_contract.get("final_event_sha256") != final_logical:
        raise Evaluation400ResultError("ledger terminal final hash changed")
    return {
        "claim_receipt_sha256": claim_logical,
        "ledger_id_sha256": ledger_id,
        "final_event_sha256": final_logical,
    }


def bootstrap_draw_bytes() -> bytes:
    """Return the protocol-frozen 20k x 400 little-endian uint16 draws."""

    count = BOOTSTRAP_SAMPLES * PAIR_COUNT
    # Rejection avoids modulo bias.  The fixed margin is much larger than the
    # expected number of rejected words and is part of generator v1.
    source_words = count + 65_536
    seed_material = (
        b"ETSF/SmolVLA/Piper/evaluation400-v3/bootstrap-draw-v1\0"
        + BOOTSTRAP_SEED.to_bytes(8, "big")
        + BOOTSTRAP_SAMPLES.to_bytes(8, "big")
        + PAIR_COUNT.to_bytes(8, "big")
    )
    stream = hashlib.shake_256(seed_material).digest(source_words * 2)
    words = np.frombuffer(stream, dtype="<u2")
    accepted = words[words < 65_200]
    if accepted.size < count:  # deterministic invariant, retained fail-closed
        raise Evaluation400ResultError("bootstrap generator v1 rejection margin exhausted")
    draws = np.remainder(accepted[:count], PAIR_COUNT).astype("<u2", copy=False)
    return draws.tobytes(order="C")


def _validate_bootstrap(payload: bytes) -> np.ndarray:
    expected_length = BOOTSTRAP_SAMPLES * PAIR_COUNT * 2
    if len(payload) != expected_length:
        raise Evaluation400ResultError("bootstrap draw artifact byte length changed")
    if payload != bootstrap_draw_bytes():
        raise Evaluation400ResultError("bootstrap draw artifact differs from frozen seed/generator")
    draws = np.frombuffer(payload, dtype="<u2").reshape(BOOTSTRAP_SHAPE)
    if bool(np.any(draws >= PAIR_COUNT)):
        raise Evaluation400ResultError("bootstrap draw index is out of range")
    return draws


def _fraction_value(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator, "denominator": value.denominator,
        "value": float(value),
    }


def exact_mcnemar(n01: int, n10: int) -> Fraction:
    discordant = n01 + n10
    if discordant == 0:
        return Fraction(1, 1)
    tail = min(n01, n10)
    probability = Fraction(
        sum(math.comb(discordant, index) for index in range(tail + 1)),
        2**discordant,
    ) * 2
    return min(Fraction(1, 1), probability)


def _linear_quantile_mean(
    sorted_sums: np.ndarray, quantile_numerator: int, quantile_denominator: int,
) -> Fraction:
    position_numerator = (len(sorted_sums) - 1) * quantile_numerator
    lower, remainder = divmod(position_numerator, quantile_denominator)
    upper = min(lower + 1, len(sorted_sums) - 1)
    interpolated_sum = (
        int(sorted_sums[lower]) * (quantile_denominator - remainder)
        + int(sorted_sums[upper]) * remainder
    )
    return Fraction(interpolated_sum, quantile_denominator * PAIR_COUNT)


def compute_statistics(
    baseline: Sequence[bool], etsf: Sequence[bool], draws: np.ndarray,
) -> dict[str, Any]:
    if len(baseline) != PAIR_COUNT or len(etsf) != PAIR_COUNT:
        raise Evaluation400ResultError("statistics require exact 400 complete pairs")
    if any(type(value) is not bool for value in baseline) \
       or any(type(value) is not bool for value in etsf):
        raise Evaluation400ResultError("statistics accept only exact boolean success")
    baseline_array = np.asarray(baseline, dtype=np.int8)
    etsf_array = np.asarray(etsf, dtype=np.int8)
    differences = etsf_array - baseline_array
    n00 = int(np.sum((baseline_array == 0) & (etsf_array == 0)))
    n01 = int(np.sum((baseline_array == 1) & (etsf_array == 0)))
    n10 = int(np.sum((baseline_array == 0) & (etsf_array == 1)))
    n11 = int(np.sum((baseline_array == 1) & (etsf_array == 1)))
    if n00 + n01 + n10 + n11 != PAIR_COUNT:
        raise Evaluation400ResultError("McNemar cells do not cover exact 400 pairs")
    bootstrap_sums = np.empty(BOOTSTRAP_SAMPLES, dtype=np.int16)
    block = 1000
    for start in range(0, BOOTSTRAP_SAMPLES, block):
        stop = min(start + block, BOOTSTRAP_SAMPLES)
        bootstrap_sums[start:stop] = differences[draws[start:stop]].sum(axis=1)
    bootstrap_sums.sort()
    lower = _linear_quantile_mean(bootstrap_sums, 25_000, 1_000_000)
    upper = _linear_quantile_mean(bootstrap_sums, 975_000, 1_000_000)
    baseline_count = n01 + n11
    etsf_count = n10 + n11
    return {
        "pair_count": PAIR_COUNT,
        "baseline_success_count": baseline_count,
        "etsf_success_count": etsf_count,
        "baseline_success_rate": _fraction_value(Fraction(baseline_count, PAIR_COUNT)),
        "etsf_success_rate": _fraction_value(Fraction(etsf_count, PAIR_COUNT)),
        "success_rate_delta_etsf_minus_baseline": _fraction_value(
            Fraction(etsf_count - baseline_count, PAIR_COUNT)
        ),
        "mcnemar": {
            "cell_definition": {
                "n00": "baseline_fail_etsf_fail", "n01": "baseline_success_etsf_fail",
                "n10": "baseline_fail_etsf_success", "n11": "baseline_success_etsf_success",
            },
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "discordant_count": n01 + n10,
            "test": "exact_two_sided_binomial_on_n01_n10",
            "exact_two_sided_p": _fraction_value(exact_mcnemar(n01, n10)),
        },
        "paired_bootstrap": {
            "unit": "pair_id", "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES, "replacement": True,
            "draw_dtype": BOOTSTRAP_FORMAT, "draw_generator": BOOTSTRAP_GENERATOR,
            "statistic": "mean(etsf_success-baseline_success)",
            "interval": "percentile_95_percent",
            "quantile_method": "linear_hyndman_fan_type_7",
            "lower_quantile_ppm": 25_000, "upper_quantile_ppm": 975_000,
            "lower": _fraction_value(lower), "upper": _fraction_value(upper),
        },
    }


def _result_signing_bytes(statement: Mapping[str, Any]) -> bytes:
    return RESULT_SIGNATURE_CONTEXT + canonical_bytes(statement)


def _load_result_private_key(
    path: Path, expected_file_sha256: str, expected_public_key_sha256: str,
) -> tuple[Any, str, str]:
    bound, payload, file_sha = secure_read(
        path, expected_file_sha256, "result signing private key"
    )
    mode = stat.S_IMODE(os.lstat(bound).st_mode)
    if mode != 0o400:
        raise Evaluation400ResultError("result signing private key must have mode 0400")
    if len(payload) != 32:
        raise Evaluation400ResultError("result signing private key must be raw 32-byte Ed25519")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.from_private_bytes(payload)
        public = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise Evaluation400ResultError("cryptography Ed25519 is required") from error
    public_sha = hashlib.sha256(public).hexdigest()
    if public_sha != _require_sha(
        expected_public_key_sha256, "expected result signer public key SHA"
    ):
        raise Evaluation400ResultError("result signing key differs from external expected key")
    return private_key, public.hex(), file_sha


def validate_result_receipt(
    value: Mapping[str, Any], *, expected_public_key_sha256: str,
) -> str:
    fields = {
        "format", "status", "signature_algorithm", "statement",
        "result_signer_public_key_hex", "result_signature_ed25519_hex",
        "receipt_sha256",
    }
    if set(value) != fields or value.get("format") != RESULT_FORMAT \
       or value.get("status") != RESULT_STATUS \
       or value.get("signature_algorithm") != "Ed25519":
        raise Evaluation400ResultError("result receipt envelope changed")
    unsigned = dict(value)
    logical = unsigned.pop("receipt_sha256", None)
    if not is_sha(logical) or logical != canonical_sha256(unsigned):
        raise Evaluation400ResultError("result receipt logical SHA mismatch")
    public_bytes = _decode_hex(
        value.get("result_signer_public_key_hex"), 32, "result signer public key"
    )
    public_sha = hashlib.sha256(public_bytes).hexdigest()
    if public_sha != _require_sha(expected_public_key_sha256, "expected result signer key"):
        raise Evaluation400ResultError("result receipt signer is not externally expected")
    statement = value.get("statement")
    if not isinstance(statement, Mapping) \
       or statement.get("result_signer_public_key_sha256") != public_sha:
        raise Evaluation400ResultError("result receipt signer statement changed")
    signature = _decode_hex(
        value.get("result_signature_ed25519_hex"), 64, "result signature"
    )
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature, _result_signing_bytes(statement)
        )
    except Exception as error:
        raise Evaluation400ResultError("result receipt Ed25519 signature failed") from error
    return str(logical)


def evaluate_results(
    *, core_path: Path, core_file_sha256: str,
    decision_path: Path, decision_file_sha256: str,
    bundle_path: Path, bundle_file_sha256: str,
    execution_terminal_path: Path, execution_terminal_file_sha256: str,
    bootstrap_draws_path: Path, bootstrap_draws_file_sha256: str,
    expected_paired_implementation_file_sha256: str,
    expected_evaluator_implementation_file_sha256: str,
    result_signing_private_key_path: Path,
    result_signing_private_key_file_sha256: str,
    result_signer_key_id: str, expected_result_signer_public_key_sha256: str,
) -> dict[str, Any]:
    evaluator_implementation_sha = _implementation_sha(
        Path(__file__), expected_evaluator_implementation_file_sha256,
        "reviewed result evaluator implementation",
    )
    (
        core_bound, core, core_file, decision_bound, decision, decision_file,
        bundle_bound, bundle, bundle_file, pairs,
    ) = load_paired_closure(
        core_path=core_path, core_file_sha256=core_file_sha256,
        decision_path=decision_path, decision_file_sha256=decision_file_sha256,
        bundle_path=bundle_path, bundle_file_sha256=bundle_file_sha256,
        expected_paired_implementation_file_sha256=(
            expected_paired_implementation_file_sha256
        ),
    )
    inventory = core["execution_inventory"]
    if evaluator_implementation_sha != inventory[
        "result_evaluator_implementation_file_sha256"
    ]:
        raise Evaluation400ResultError(
            "result evaluator implementation is not the core-frozen inventory build"
        )
    if expected_result_signer_public_key_sha256 != core["authority_policy"][
        "result_evaluator_identity_sha256"
    ]:
        raise Evaluation400ResultError(
            "result signer identity is not the core-authorized result evaluator"
        )
    bootstrap_bound, bootstrap_payload, bootstrap_file = secure_read(
        bootstrap_draws_path, bootstrap_draws_file_sha256, "bootstrap draw artifact"
    )
    draws = _validate_bootstrap(bootstrap_payload)
    terminal_bound, terminal, terminal_file = read_json(
        execution_terminal_path, execution_terminal_file_sha256,
        "external execution terminal",
    )
    raw_statement = terminal.get("statement")
    if not isinstance(raw_statement, Mapping):
        raise Evaluation400ResultError("execution terminal statement is missing")
    executor_public_bytes = _decode_hex(
        raw_statement.get("executor_public_key_hex"), 32, "executor public key"
    )
    if hashlib.sha256(executor_public_bytes).hexdigest() != core[
        "authority_policy"
    ]["executor_identity_sha256"]:
        raise Evaluation400ResultError("terminal executor is not core-authorized")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        executor_public_key = Ed25519PublicKey.from_public_bytes(executor_public_bytes)
    except (ImportError, ModuleNotFoundError) as error:
        raise Evaluation400ResultError("cryptography Ed25519 is required") from error
    terminal_statement, terminal_logical = _verify_executor_receipt(
        terminal, expected_format=TERMINAL_FORMAT, expected_status=TERMINAL_STATUS,
        public_key=executor_public_key, role="execution terminal",
    )
    core_record = _descriptor(core_bound, core_file, core["protocol_core_sha256"])
    decision_record = _descriptor(decision_bound, decision_file, decision["decision_sha256"])
    bundle_record = _descriptor(bundle_bound, bundle_file, bundle["bundle_sha256"])
    (
        execution_nonce, dependency_rehash, claim_record, ledger_event_records,
        terminal_pair_records, terminal_condition_records,
    ) = _validate_terminal_statement(
        terminal_statement, core_record=core_record,
        decision_record=decision_record, bundle_record=bundle_record,
        core=core, bootstrap_path=bootstrap_bound,
        bootstrap_file_sha256=bootstrap_file,
    )
    seen_paths: set[str] = set()
    condition_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for global_ordinal, raw_record in enumerate(terminal_condition_records):
        pair = pairs[global_ordinal // 2]
        position = global_ordinal % 2
        record = _condition_terminal_record(raw_record, global_ordinal, pair, position)
        if record["path"] in seen_paths:
            raise Evaluation400ResultError("condition receipt path reused")
        seen_paths.add(record["path"])
        bound, receipt, file_sha = read_json(
            Path(record["path"]), record["file_sha256"],
            f"condition receipt {global_ordinal}",
        )
        _path_matches(record["path"], bound, f"condition receipt {global_ordinal}")
        condition_statement, logical = _verify_executor_receipt(
            receipt, expected_format=CONDITION_FORMAT,
            expected_status=CONDITION_STATUS, public_key=executor_public_key,
            role=f"condition receipt {global_ordinal}",
        )
        if file_sha != record["file_sha256"] or logical != record["logical_sha256"]:
            raise Evaluation400ResultError("condition receipt descriptor SHA mismatch")
        artifact_paths = _validate_condition_statement(
            condition_statement, pair=pair, position=position, core=core,
            decision=decision, bundle=bundle, execution_nonce_hex=execution_nonce,
            dependency_rehash_sha256=dependency_rehash,
        )
        if seen_paths & artifact_paths:
            raise Evaluation400ResultError("condition execution-artifact path reused")
        seen_paths.update(artifact_paths)
        condition_items.append((record, condition_statement))
    baseline_success: list[bool] = []
    etsf_success: list[bool] = []
    pair_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ordinal, raw_record in enumerate(terminal_pair_records):
        pair = pairs[ordinal]
        record = _pair_terminal_record(raw_record, ordinal, pair)
        if record["path"] in seen_paths:
            raise Evaluation400ResultError("pair/condition receipt path reused")
        seen_paths.add(record["path"])
        bound, receipt, file_sha = read_json(
            Path(record["path"]), record["file_sha256"], f"pair receipt {ordinal}"
        )
        _path_matches(record["path"], bound, f"pair receipt {ordinal}")
        pair_statement, logical = _verify_executor_receipt(
            receipt, expected_format=PAIR_FORMAT, expected_status=PAIR_STATUS,
            public_key=executor_public_key, role=f"pair receipt {ordinal}",
        )
        if file_sha != record["file_sha256"] or logical != record["logical_sha256"]:
            raise Evaluation400ResultError("pair receipt descriptor SHA mismatch")
        pair_conditions = condition_items[2 * ordinal: 2 * ordinal + 2]
        _validate_pair_statement(
            pair_statement, pair=pair, conditions=pair_conditions, core=core,
            decision=decision, bundle=bundle, execution_nonce_hex=execution_nonce,
            dependency_rehash_sha256=dependency_rehash,
        )
        baseline_success.append(pair_statement["baseline_success"])
        etsf_success.append(pair_statement["etsf_success"])
        pair_items.append((record, pair_statement))
    ledger_audit = validate_execution_ledger(
        claim_record=claim_record, event_records=ledger_event_records,
        condition_items=condition_items, pair_items=pair_items,
        ledger_contract=terminal_statement["ledger_contract"], core=core,
        decision=decision, bundle=bundle, execution_nonce_hex=execution_nonce,
        dependency_rehash_sha256=dependency_rehash,
        public_key=executor_public_key,
    )
    statistics = compute_statistics(baseline_success, etsf_success, draws)
    if not isinstance(result_signer_key_id, str) or not result_signer_key_id:
        raise Evaluation400ResultError("result signer key ID is required")
    private_key, result_public_hex, private_key_file_sha = _load_result_private_key(
        result_signing_private_key_path,
        result_signing_private_key_file_sha256,
        expected_result_signer_public_key_sha256,
    )
    result_public_sha = hashlib.sha256(bytes.fromhex(result_public_hex)).hexdigest()
    if result_public_sha != core["authority_policy"][
        "result_evaluator_identity_sha256"
    ]:
        raise Evaluation400ResultError("loaded result signer differs from paired core")
    statement = {
        "input_closure": {
            "protocol_core": core_record,
            "ed25519_decision": decision_record,
            "execution_bundle": bundle_record,
            "execution_terminal": _descriptor(
                terminal_bound, terminal_file, terminal_logical
            ),
            "bootstrap_draws": _descriptor(
                bootstrap_bound, bootstrap_file,
                terminal_statement["bootstrap_draws"]["logical_sha256"],
            ),
        },
        "implementation_binding": {
            "paired_v3_file_sha256": expected_paired_implementation_file_sha256,
            "result_evaluator_file_sha256": evaluator_implementation_sha,
            "paired_adapter_contract": "paired_core_decision_bundle_v3_exact_adapter_v1",
        },
        "execution_binding": {
            "execution_nonce_hex": execution_nonce,
            "executor_key_id": terminal_statement["executor_key_id"],
            "executor_public_key_sha256": terminal_statement[
                "executor_public_key_sha256"
            ],
            "executor_identity_sha256": terminal_statement[
                "executor_identity_sha256"
            ],
            "one_shot_ledger_id_sha256": ledger_audit["ledger_id_sha256"],
            "execution_claim_receipt_sha256": ledger_audit[
                "claim_receipt_sha256"
            ],
            "ledger_final_event_sha256": ledger_audit["final_event_sha256"],
            "pair_identity_set_sha256": core["evaluation400"][
                "pair_identity_set_sha256"
            ],
            "deployment_binding_sha256": core["deployment"][
                "deployment_binding_sha256"
            ],
        },
        "coverage": {
            "required_pair_count": PAIR_COUNT, "complete_pair_count": PAIR_COUNT,
            "required_condition_count": CONDITION_COUNT,
            "complete_condition_count": CONDITION_COUNT,
            "missing_pair_count": 0, "missing_condition_count": 0,
            "retry_count": 0, "incomplete_count": 0, "exclusion_count": 0,
            "subset_statistics_computed": False,
        },
        "statistics": statistics,
        "result_protocol": core["result_protocol"],
        "computation_contract": {
            "binary_success_exact_bool": True,
            "mcnemar_n01": "baseline_success_etsf_fail",
            "mcnemar_n10": "baseline_fail_etsf_success",
            "mcnemar_zero_discordant_p": _fraction_value(Fraction(1, 1)),
            "bootstrap_draw_artifact_preoutcome_frozen": True,
            "bootstrap_quantile_method": "linear_hyndman_fan_type_7",
            "posthoc_exclusion_or_subgroup_selection": False,
        },
        "capability_receipt": {
            "simulator_calls": 0, "policy_calls": 0,
            "trajectory_or_hdf_files_opened": 0,
            "condition_receipts_opened": CONDITION_COUNT,
            "pair_receipts_opened": PAIR_COUNT,
            "execution_terminal_receipts_opened": 1,
            "execution_claim_receipts_opened": 1,
            "ledger_event_receipts_opened": LEDGER_EVENT_COUNT,
            "bootstrap_artifacts_opened": 1,
            "subset_result_publications": 0,
        },
        "result_signer_key_id": result_signer_key_id,
        "result_signer_public_key_sha256": result_public_sha,
        "result_signing_private_key_file_sha256": private_key_file_sha,
    }
    signature = private_key.sign(_result_signing_bytes(statement)).hex()
    base = {
        "format": RESULT_FORMAT, "status": RESULT_STATUS,
        "signature_algorithm": "Ed25519", "statement": statement,
        "result_signer_public_key_hex": result_public_hex,
        "result_signature_ed25519_hex": signature,
    }
    receipt = {**base, "receipt_sha256": canonical_sha256(base)}
    validate_result_receipt(
        receipt, expected_public_key_sha256=expected_result_signer_public_key_sha256
    )
    return receipt


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = _safe_lexical_path(path, "result output")
    if output.suffix.casefold() != ".json":
        raise Evaluation400ResultError("result output must be JSON")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent = output.parent
    parent_metadata = os.lstat(parent)
    if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode):
        raise Evaluation400ResultError("result output parent must be a real directory")
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        os.link(temporary, output, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    mode = stat.S_IMODE(os.lstat(output).st_mode)
    if mode != 0o400:
        raise Evaluation400ResultError("result receipt publication mode is not 0400")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "core", "decision", "bundle", "execution-terminal", "bootstrap-draws",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-file-sha256", required=True)
    parser.add_argument(
        "--expected-paired-v3-implementation-file-sha256", required=True
    )
    parser.add_argument(
        "--expected-evaluator-implementation-file-sha256", required=True
    )
    parser.add_argument("--result-signing-private-key", type=Path, required=True)
    parser.add_argument("--result-signing-private-key-file-sha256", required=True)
    parser.add_argument("--result-signer-key-id", required=True)
    parser.add_argument("--expected-result-signer-public-key-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = evaluate_results(
        core_path=args.core, core_file_sha256=args.core_file_sha256,
        decision_path=args.decision, decision_file_sha256=args.decision_file_sha256,
        bundle_path=args.bundle, bundle_file_sha256=args.bundle_file_sha256,
        execution_terminal_path=args.execution_terminal,
        execution_terminal_file_sha256=args.execution_terminal_file_sha256,
        bootstrap_draws_path=args.bootstrap_draws,
        bootstrap_draws_file_sha256=args.bootstrap_draws_file_sha256,
        expected_paired_implementation_file_sha256=(
            args.expected_paired_v3_implementation_file_sha256
        ),
        expected_evaluator_implementation_file_sha256=(
            args.expected_evaluator_implementation_file_sha256
        ),
        result_signing_private_key_path=args.result_signing_private_key,
        result_signing_private_key_file_sha256=(
            args.result_signing_private_key_file_sha256
        ),
        result_signer_key_id=args.result_signer_key_id,
        expected_result_signer_public_key_sha256=(
            args.expected_result_signer_public_key_sha256
        ),
    )
    write_json_new(args.output, receipt)


if __name__ == "__main__":
    main()
