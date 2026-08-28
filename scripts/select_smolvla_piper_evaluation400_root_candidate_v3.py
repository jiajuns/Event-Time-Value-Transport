#!/usr/bin/env python3
"""Exact formal190-frozen composite-rank selector for evaluation400 v3."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import smolvla_piper_deployment_uncertainty_v1 as deployment_uncertainty


FORMAT = "etsf_smolvla_piper_evaluation400_composite_root_selector_v3"
PRIMARY_SCORE = (
    "five_member_adjusted_source_composite_candidate_rank_score_margin"
)
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)
HEADS = [
    "post_event", "next_event", "duration", "success", "object_effect",
    "recovery", "source_contract_rank_score",
]
SHA_CHARS = frozenset("0123456789abcdef")


class RootSelectorV3Error(RuntimeError):
    """The frozen selector inputs or decision algebra changed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _finite_positive(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RootSelectorV3Error(f"{role} must be finite and positive")
    return float(value)


def _finite_nonnegative(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RootSelectorV3Error(f"{role} must be finite and nonnegative")
    return float(value)


def _metric_value(
    calibration: Mapping[str, Any], head: str, field: str,
) -> float:
    value = calibration.get("metrics", {}).get(head, {}).get(field)
    return _finite_positive(value, f"{head}.{field}")


def select_root_candidate_v3(
    *, predictions: Mapping[str, Any], prediction_candidate_indices: Any,
    candidate_legal: Any, fallback_index: int,
    calibration: Mapping[str, Any], selector_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exactly one root candidate or abstain to the frozen baseline."""

    indices = np.asarray(prediction_candidate_indices)
    legal = np.asarray(candidate_legal)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(indices) < 2
        or len(set(indices.astype(int).tolist())) != len(indices)
        or legal.shape != (4,)
        or legal.dtype != np.bool_
        or type(fallback_index) is not int
        or not 0 <= fallback_index < 4
        or not bool(legal[fallback_index])
        or fallback_index not in indices
        or set(indices.astype(int).tolist())
        != set(np.flatnonzero(legal).astype(int).tolist())
        or any(
            not 0 <= int(value) < 4 or not bool(legal[int(value)])
            for value in indices
        )
    ):
        raise RootSelectorV3Error("legal candidate/index contract changed")
    count = len(indices)
    expected_shapes = {
        "post_event_logits": (5, count, 5),
        "next_event_logits": (5, count, 5),
        "duration_log_mean": (5, count),
        "duration_log_scale": (5, count),
        "success_logit": (5, count),
        "recovery_logit": (5, count),
        "object_mean": None,
        "object_log_scale": None,
        "source_contract_base_rank_score": (5, count),
        "source_action_rank_residual": (5, count),
        "source_contract_rank_score": (5, count),
    }
    source_rank_names = {
        "source_contract_base_rank_score",
        "source_action_rank_residual",
        "source_contract_rank_score",
    }
    arrays: dict[str, np.ndarray] = {}
    for name, shape in expected_shapes.items():
        raw_array = np.asarray(predictions.get(name), dtype=np.float64)
        array = np.asarray(
            predictions.get(name),
            dtype=np.float32 if name in source_rank_names else np.float64,
        )
        if (
            (shape is not None and array.shape != shape)
            or (name == "object_mean" and (array.ndim != 3 or array.shape[:2] != (5, count)))
            or (name == "object_log_scale" and array.shape != arrays.get("object_mean", array).shape)
            or not np.isfinite(array).all()
            or (
                name in source_rank_names
                and not np.array_equal(raw_array, array.astype(np.float64))
            )
        ):
            raise RootSelectorV3Error(f"prediction shape/finite contract changed: {name}")
        arrays[name] = array

    if (
        calibration.get("calibration_sha256")
        != selector_authority.get("calibration_sha256")
        or calibration.get("all_six_heads_support_performance_uncertainty_gate_passed")
        is not True
        or set(calibration.get("head_enabled_for_primary", {}))
        != {"post_event", "next_event", "success", "recovery", "duration", "object_effect"}
        or not all(calibration["head_enabled_for_primary"].values())
    ):
        raise RootSelectorV3Error("six-head calibration authority changed")
    source_rank_contracts = selector_authority.get("source_rank_score_contracts")
    source_rank_contract_shas = selector_authority.get(
        "source_rank_score_contract_sha256"
    )
    if (
        not isinstance(source_rank_contracts, list)
        or len(source_rank_contracts) != 5
        or not isinstance(source_rank_contract_shas, list)
        or len(source_rank_contract_shas) != 5
    ):
        raise RootSelectorV3Error("five Source rank contracts are missing")
    source_temperatures: list[float] = []
    for index, contract in enumerate(source_rank_contracts):
        if not isinstance(contract, Mapping):
            raise RootSelectorV3Error("Source rank contract changed")
        unsigned = dict(contract)
        logical = unsigned.pop("contract_sha256", None)
        temperature = contract.get("success_temperature")
        if (
            not _is_sha(logical)
            or logical != canonical_sha256(unsigned)
            or logical != source_rank_contract_shas[index]
            or contract.get("base_score") != "candidate_rank_score"
            or contract.get("source_action_rank_residual") is not True
            or contract.get("source_action_rank_success_only") is not False
            or contract.get("residual_combination")
            != "candidate_rank_score_plus_action_rank_residual"
        ):
            raise RootSelectorV3Error("Source rank contract authority changed")
        source_temperatures.append(
            _finite_positive(temperature, "Source rank success temperature")
        )
    if selector_authority.get(
        "source_rank_numeric_contract"
    ) != SOURCE_RANK_NUMERIC_CONTRACT:
        raise RootSelectorV3Error("Source rank numeric contract changed")
    member_authority = selector_authority.get("source_rank_member_authority")
    authority_members = (
        member_authority.get("members")
        if isinstance(member_authority, Mapping) else None
    )
    if (
        not isinstance(member_authority, Mapping)
        or set(member_authority)
        != {"source_rank_numeric_contract", "members"}
        or member_authority.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(authority_members, list)
        or len(authority_members) != 5
        or selector_authority.get("source_rank_member_authority_sha256")
        != canonical_sha256(member_authority)
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "member_index", "source_checkpoint_file_sha256",
                "source_rank_score_contract_sha256", "success_temperature",
            }
            or row.get("member_index") != index
            or row.get("source_checkpoint_file_sha256")
            != source_rank_contracts[index].get(
                "source_checkpoint_file_sha256"
            )
            or row.get("source_rank_score_contract_sha256")
            != source_rank_contract_shas[index]
            or isinstance(row.get("success_temperature"), bool)
            or not isinstance(row.get("success_temperature"), (int, float))
            or float(row["success_temperature"]) != source_temperatures[index]
            for index, row in enumerate(authority_members or [])
        )
    ):
        raise RootSelectorV3Error("Source rank member authority changed")
    source_temperatures32 = np.asarray(
        source_temperatures, dtype=np.float32
    )[:, None]
    if (
        not np.isfinite(source_temperatures32).all()
        or bool((source_temperatures32 <= np.float32(0.0)).any())
    ):
        raise RootSelectorV3Error(
            "Source rank temperature is not representable as positive float32"
        )
    reconstructed_rank = (
        arrays["source_contract_base_rank_score"]
        + arrays["source_action_rank_residual"]
        / source_temperatures32
    )
    if reconstructed_rank.dtype != np.float32 or not np.array_equal(
        reconstructed_rank, arrays["source_contract_rank_score"]
    ):
        raise RootSelectorV3Error(
            "Source composite rank does not equal base plus residual/temperature"
        )
    root_ranker = calibration.get("root_group_ranker")
    selected_threshold = (
        root_ranker.get("selected_candidate")
        if isinstance(root_ranker, Mapping)
        else None
    )
    if (
        not isinstance(root_ranker, Mapping)
        or root_ranker.get("enabled_for_primary") is not True
        or root_ranker.get("root_group_ranker_sha256")
        != selector_authority.get("formal190_root_group_ranker_sha256")
        or root_ranker.get("score_is_success_logit") is not False
        or root_ranker.get("score_is_success_probability") is not False
        or not isinstance(selected_threshold, Mapping)
    ):
        raise RootSelectorV3Error("formal190 root-ranker authority changed")
    minimum_margin = _finite_nonnegative(
        selected_threshold.get("minimum_group_relative_composite_rank_score_margin"),
        "formal190 composite margin",
    )
    maximum_pair_uncertainty = _finite_nonnegative(
        selected_threshold.get("maximum_structured_pair_uncertainty"),
        "formal190 pair uncertainty",
    )
    formal_maximum_candidate_uncertainty = _finite_nonnegative(
        selected_threshold.get("maximum_global_candidate_uncertainty"),
        "formal190 global candidate uncertainty",
    )
    abstain = calibration.get("abstain_threshold")
    if not isinstance(abstain, Mapping) or abstain.get("enabled") is not True:
        raise RootSelectorV3Error("global six-head evidence gate is disabled")
    maximum_total_uncertainty = _finite_nonnegative(
        abstain.get("maximum_total_uncertainty"), "global total uncertainty"
    )
    if formal_maximum_candidate_uncertainty != maximum_total_uncertainty:
        raise RootSelectorV3Error(
            "formal190 gain gate did not mirror global candidate uncertainty"
        )
    formal_thresholds = selector_authority.get("formal190_thresholds")
    expected_thresholds = {
        "minimum_formal190_composite_margin": minimum_margin,
        "maximum_formal190_pair_uncertainty": maximum_pair_uncertainty,
        "maximum_global_total_uncertainty": maximum_total_uncertainty,
        "root_group_ranker_sha256": root_ranker["root_group_ranker_sha256"],
    }
    if formal_thresholds != expected_thresholds:
        raise RootSelectorV3Error("Formal190 deployment thresholds changed")
    uncertainty_contract = selector_authority.get("uncertainty_contract")
    if (
        not isinstance(uncertainty_contract, Mapping)
        or uncertainty_contract.get(
            "duration_deployment_scale_applied_before_selector"
        ) is not True
        or uncertainty_contract.get(
            "object_deployment_scale_applied_before_selector"
        ) is not True
        or uncertainty_contract.get(
            "object_predictions_physical_xyz_before_selector"
        ) is not True
    ):
        raise RootSelectorV3Error("uncertainty contract is missing")
    object_robust_scale = _finite_positive(
        uncertainty_contract.get("formal190_object_error_robust_scale_m"),
        "formal190 object robust scale",
    )
    calibrated_robust_scale = calibration.get("metrics", {}).get(
        "object_total_variance", {}
    ).get("deployment_object_error_robust_scale_m")
    if (
        isinstance(calibrated_robust_scale, bool)
        or not isinstance(calibrated_robust_scale, (int, float))
        or float(calibrated_robust_scale) != object_robust_scale
    ):
        raise RootSelectorV3Error(
            "object robust scale is not bound to formal190 calibration"
        )

    uncertainty_parameters = {
        "post_event_temperature": _metric_value(
            calibration, "post_event", "deployment_temperature"
        ),
        "next_event_temperature": _metric_value(
            calibration, "next_event", "deployment_temperature"
        ),
        "success_temperature": _metric_value(
            calibration, "success", "deployment_temperature"
        ),
        "conditional_recovery_temperature": _metric_value(
            calibration, "conditional_recovery", "deployment_temperature"
        ),
        "object_error_robust_scale_m": object_robust_scale,
    }
    deployment_parameters = selector_authority.get("deployment_parameters")
    expected_deployment_parameters = {
        **uncertainty_parameters,
        "duration_scale_multiplier": _metric_value(
            calibration, "duration_lognormal_mixture", "deployment_scale_multiplier"
        ),
        "object_scale_multiplier": _metric_value(
            calibration, "object_total_variance", "deployment_scale_multiplier"
        ),
        "deployment_uncertainty_contract_sha256": uncertainty_contract.get(
            "deployment_uncertainty_contract_sha256"
        ),
    }
    if deployment_parameters != expected_deployment_parameters:
        raise RootSelectorV3Error("Formal190 deployment parameters changed")
    uncertainty_implementation = selector_authority.get(
        "deployment_uncertainty_implementation"
    )
    actual_uncertainty_path = Path(deployment_uncertainty.__file__).resolve()
    actual_uncertainty_sha = hashlib.sha256(
        actual_uncertainty_path.read_bytes()
    ).hexdigest()
    if uncertainty_implementation != {
        "path": str(actual_uncertainty_path),
        "file_sha256": actual_uncertainty_sha,
    }:
        raise RootSelectorV3Error("deployment uncertainty implementation changed")
    try:
        component_arrays = deployment_uncertainty.root_components(
            predictions=arrays, parameters=uncertainty_parameters
        )
    except deployment_uncertainty.DeploymentUncertaintyError as error:
        raise RootSelectorV3Error("deployment uncertainty input changed") from error
    recovery_policy = root_ranker.get("root_recovery_uncertainty_policy")
    if (
        recovery_policy
        != deployment_uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY
        or root_ranker.get("root_structured_uncertainty_head_count")
        != deployment_uncertainty.ROOT_HEAD_COUNT
        or type(root_ranker.get("root_structured_uncertainty_head_count")) is not int
        or uncertainty_contract.get("root_recovery_uncertainty_policy")
        != recovery_policy
        or uncertainty_contract.get("root_structured_uncertainty_head_count")
        != deployment_uncertainty.ROOT_HEAD_COUNT
    ):
        raise RootSelectorV3Error("root recovery applicability policy changed")
    structured = component_arrays["structured_five_head"]
    member_rank = arrays["source_contract_rank_score"]
    mean_rank = member_rank.astype(np.float64).mean(axis=0)
    baseline_row = int(np.flatnonzero(indices == fallback_index)[0])
    margin = mean_rank - mean_rank[baseline_row]
    alternative_rows = [
        row for row in range(count) if int(indices[row]) != fallback_index
    ]
    proposed_row = min(
        alternative_rows,
        key=lambda row: (-float(margin[row]), int(indices[row])),
    )
    proposed = int(indices[proposed_row])
    proposed_margin = float(margin[proposed_row])
    proposed_uncertainty = float(structured[proposed_row])
    baseline_uncertainty = float(structured[baseline_row])
    pair_uncertainty = max(proposed_uncertainty, baseline_uncertainty)
    accepted = bool(
        proposed != fallback_index
        and proposed_margin > minimum_margin
        and proposed_uncertainty <= maximum_total_uncertainty
        and pair_uncertainty <= maximum_pair_uncertainty
    )
    selected = proposed if accepted else fallback_index
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
        "source_rank_success_temperatures": source_temperatures,
        "alternative_set_contract": "all_legal_candidates_except_lowest_legal_baseline",
        "margin_comparison": "strict_greater_than_formal190_threshold",
        "proposed_candidate_index": proposed,
        "fallback_candidate_index": fallback_index,
        "score_margin": proposed_margin,
        "minimum_margin": minimum_margin,
        "proposed_uncertainty": proposed_uncertainty,
        "baseline_uncertainty": baseline_uncertainty,
        "pair_uncertainty": pair_uncertainty,
        "maximum_total_uncertainty": maximum_total_uncertainty,
        "maximum_pair_uncertainty": maximum_pair_uncertainty,
        "accepted": accepted,
        "selected_candidate_index": selected,
    }
    uncertainty_components = {
        name: component_arrays[name].tolist()
        for name in sorted(component_arrays)
    }
    selector_input = {
        "prediction_candidate_indices": indices.astype(int).tolist(),
        "candidate_legal": legal.tolist(),
        "fallback_candidate_index": fallback_index,
        "predictions": {
            name: arrays[name].tolist() for name in sorted(arrays)
        },
        "calibration_sha256": calibration["calibration_sha256"],
        "selector_authority_sha256": selector_authority[
            "selector_authority_sha256"
        ],
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "uncertainty_parameters": uncertainty_parameters,
        "deployment_parameters": deployment_parameters,
        "formal190_thresholds": formal_thresholds,
        "deployment_uncertainty_implementation": uncertainty_implementation,
    }
    base = {
        "selected_candidate_index": selected,
        "proposed_candidate_index": proposed,
        "fallback_candidate_index": fallback_index,
        "score_margin": proposed_margin,
        "total_uncertainty": pair_uncertainty,
        "proposed_uncertainty": proposed_uncertainty,
        "baseline_uncertainty": baseline_uncertainty,
        "minimum_formal190_composite_margin": minimum_margin,
        "maximum_formal190_pair_uncertainty": maximum_pair_uncertainty,
        "maximum_global_total_uncertainty": maximum_total_uncertainty,
        "candidate_change_accepted": accepted,
        "decision_reason": (
            "formal190_composite_margin_and_five_applicable_head_uncertainty_passed"
            if accepted else "fallback_gate"
        ),
        "uncertainty_gate_applied": True,
        "five_member_call_count": 5,
        "prediction_heads_computed": list(HEADS),
        "primary_score_contract": PRIMARY_SCORE,
        "source_rank_score_contract_sha256": list(
            selector_authority["source_rank_score_contract_sha256"]
        ),
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "formal190_target_outcome_calibrated_acceptance_margin": True,
        "calibration_sha256": calibration["calibration_sha256"],
        "formal190_root_group_ranker_sha256": root_ranker[
            "root_group_ranker_sha256"
        ],
        "decision_algebra_sha256": canonical_sha256(algebra),
        "selector_input_sha256": canonical_sha256(selector_input),
        "selector_input": selector_input,
        "prediction_candidate_indices": indices.astype(int).tolist(),
        "alternative_candidate_indices": [
            int(indices[row]) for row in alternative_rows
        ],
        "member_source_contract_rank_scores": member_rank.tolist(),
        "member_source_contract_base_rank_scores": arrays[
            "source_contract_base_rank_score"
        ].tolist(),
        "member_source_action_rank_residuals": arrays[
            "source_action_rank_residual"
        ].tolist(),
        "member_source_rank_success_temperatures": source_temperatures,
        "uncertainty_components": uncertainty_components,
        "root_recovery_uncertainty_policy": recovery_policy,
        "root_structured_uncertainty_head_count": (
            deployment_uncertainty.ROOT_HEAD_COUNT
        ),
        "alternative_set_contract": (
            "all_legal_candidates_except_lowest_legal_baseline"
        ),
        "margin_comparison": "strict_greater_than_formal190_threshold",
    }
    return {**base, "selector_proof_sha256": canonical_sha256(base)}
