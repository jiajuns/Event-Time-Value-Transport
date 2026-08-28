#!/usr/bin/env python3
"""Freeze a fail-closed composite structured-prediction capability manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import calibrate_openvla_etsf_v8_success_inner_cv as success_calibration
import evaluate_openvla_etsf_v8_factual_events as factual_evaluator
import evaluate_openvla_etsf_v8_oof_bridge as structured_bridge
import evaluate_openvla_etsf_v8_structured_heads_arrays as structured_evaluator
import openvla_etsf_structured_event_time_utility as v7_utility
from openvla_etsf_composite_structured_prediction_router import (
    ACTIVE_CAPABILITIES,
    FORMAT,
    INACTIVE_CAPABILITIES,
    validate_composite_activation,
)
from openvla_etsf_duration_hierarchy import canonical_sha256
from openvla_etsf_duration_hierarchy_adapter import (
    load_duration_activation,
    sha256_path,
)
from train_openvla_etsf_v8_structured_adapters import (
    V8_TRAINING_CHECKPOINT_FORMAT,
)


IMPLEMENTATION_FILES = {
    "freeze_openvla_etsf_composite_structured_prediction_activation.py",
    "openvla_etsf_composite_structured_prediction_router.py",
    "openvla_etsf_duration_hierarchy.py",
    "openvla_etsf_duration_hierarchy_adapter.py",
    "evaluate_openvla_etsf_v8_factual_events.py",
    "evaluate_openvla_etsf_v8_oof_bridge.py",
    "evaluate_openvla_etsf_v8_structured_heads_arrays.py",
    "calibrate_openvla_etsf_v8_success_inner_cv.py",
    "openvla_etsf_structured_event_time_utility.py",
}


def _reject_fresh_path(path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot reference Fresh/confirmation")
    return resolved


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    path = _reject_fresh_path(path, role=role)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must contain a JSON object")
    return value


def _signed(value: Mapping[str, Any], key: str, *, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{role} signature mismatch")
    return recorded


def _finite_metric(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} metric is missing") from error
    if not np.isfinite(result):
        raise RuntimeError(f"{name} metric is not finite")
    return result


def _authenticate_factual_result(
    path: Path, *, expected_materialization_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _reject_fresh_path(path, role="R3 factual event result")
    result = _load_json(path, role="R3 factual event result")
    result_sha = _signed(result, "result_sha256", role="R3 factual event result")
    source = result.get("source_materialization")
    frozen = result.get("frozen_factual_state")
    uncertainty = result.get("uncertainty_scope")
    authorization = result.get("authorization")
    immediate = result.get("immediate_event")
    destination = result.get("observed_destination_event")
    if (
        result.get("format") != factual_evaluator.FORMAT
        or result.get("status") != "complete_adaptive_development_only"
        or result.get("evidence_scope")
        != "D250_adaptive_development_only_not_prospective"
        or result.get("logical_groups") != 250
        or result.get("fresh_confirmation_data_or_labels_read") is not False
        or not isinstance(source, Mapping)
        or source.get("materialization_sha256") != expected_materialization_sha256
        or not isinstance(frozen, Mapping)
        or frozen.get("bit_exact_is_accuracy_evidence") is not False
        or frozen.get("accuracy_measured_from_labels_and_logits") is not True
        or not isinstance(uncertainty, Mapping)
        or uncertainty.get("evaluated_quantity")
        != "single_factual_member_composite_aleatoric_score"
        or uncertainty.get("epistemic_uncertainty_available") is not False
        or uncertainty.get("complete_predictive_uncertainty_claimed") is not False
        or not isinstance(authorization, Mapping)
        or any(authorization.get(key) is not False for key in authorization)
        or not isinstance(immediate, Mapping)
        or not isinstance(destination, Mapping)
        or int(immediate.get("support_rows", 0)) <= 0
        or int(destination.get("support_rows", 0)) <= 0
    ):
        raise RuntimeError("R3 factual event evidence contract changed")
    for name, domain in (("next_event", immediate), ("destination", destination)):
        model = domain.get("model")
        risk = domain.get("uncertainty")
        if not isinstance(model, Mapping) or not isinstance(risk, Mapping):
            raise RuntimeError(f"R3 {name} measured diagnostics are missing")
        _finite_metric(model.get("accuracy"), name=f"{name} accuracy")
        _finite_metric(model.get("nll"), name=f"{name} NLL")
        _finite_metric(risk.get("aurc"), name=f"{name} AURC")
    return result, {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "result_sha256": result_sha,
        "materialization_sha256": expected_materialization_sha256,
        "required_sha_fields": [
            "file_sha256",
            "result_sha256",
            "materialization_sha256",
        ],
    }


def _authenticate_adamw_regress(
    *,
    result_path: Path,
    contracts_path: Path,
    expected_materialization_sha256: str,
    expected_factual_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = _reject_fresh_path(result_path, role="R4 AdamW result")
    contracts_path = _reject_fresh_path(contracts_path, role="R4 AdamW contracts")
    result = _load_json(result_path, role="R4 AdamW result")
    result_sha = _signed(result, "result_sha256", role="R4 AdamW result")
    contracts = _load_json(contracts_path, role="R4 AdamW contracts")
    contracts_sha = _signed(
        contracts, "contracts_sha256", role="R4 AdamW contracts"
    )
    adaptive = contracts.get("adaptive_contract")
    bundle = contracts.get("bridge_bundle")
    provenance = contracts.get("bridge_provenance")
    if (
        contracts.get("format") != structured_bridge.OUTPUT_FORMAT
        or not isinstance(adaptive, Mapping)
        or not isinstance(bundle, Mapping)
        or bundle.get("format") != structured_bridge.BRIDGE_FORMAT
        or bundle.get("status") != "authenticated_inputs_rehashed"
        or bundle.get("source_partition") != "development_only"
        or bundle.get("fresh50_inputs_accepted") is not False
        or bundle.get("fresh50_labels_read") is not False
        or bundle.get("adaptive_development_contract_sha256")
        != adaptive.get("contract_sha256")
        or not isinstance(provenance, Mapping)
        or provenance.get("materialization_sha256")
        != expected_materialization_sha256
        or result.get("format") != structured_evaluator.FORMAT
        or result.get("adaptive_development_contract_sha256")
        != adaptive.get("contract_sha256")
        or result.get("evidence_design")
        != "adaptive_current_d250_after_collection_started"
        or result.get("prospective_claim_allowed") is not False
        or result.get("fresh50_inputs_accepted") is not False
        or result.get("fresh50_labels_read") is not False
        or result.get("fresh50_confirmation_authorized") is not False
        or result.get("v7_implementation_changed") is not False
        or result.get("action_selector_authorized") is not False
    ):
        raise RuntimeError("R4 adaptive AdamW evidence binding changed")
    _signed(adaptive, "contract_sha256", role="R4 adaptive contract")
    bundle_sha = _signed(bundle, "bridge_bundle_sha256", role="R4 bridge bundle")
    sources = adaptive.get("source_sha256")
    if not isinstance(sources, Mapping) or sources.get(
        "base_checkpoint"
    ) != expected_factual_checkpoint_sha256:
        raise RuntimeError("R4 factual checkpoint binding changed")
    arrays_path = _reject_fresh_path(
        Path(str(contracts.get("arrays", ""))), role="R4 structured arrays"
    )
    if sha256_path(arrays_path) != contracts.get("arrays_sha256"):
        raise RuntimeError("R4 structured arrays SHA mismatch")
    regress = result.get("domains", {}).get("regress")
    if (
        not isinstance(regress, Mapping)
        or regress.get("status") != "passed"
        or regress.get("passed") is not True
        or result.get("domain_pass", {}).get("regress") is not True
        or regress.get("support_gate") is not True
        or regress.get("weight_provenance_error") is not None
        or regress.get("baseline_provenance_error") is not None
        or regress.get("brier_vs_crossfit_prevalence", {}).get("strict_skill")
        is not True
        or regress.get("nll_vs_crossfit_prevalence", {}).get("strict_skill")
        is not True
        or regress.get("ap_minus_prevalence", {}).get("strict_skill") is not True
        or regress.get("ece_gate") is not True
    ):
        raise RuntimeError("R4 regress did not pass every strict probability gate")
    folds = bundle.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("R4 bridge bundle lacks five AdamW checkpoints")
    checkpoint_rows: list[dict[str, Any]] = []
    for owner, row in enumerate(folds):
        if (
            not isinstance(row, Mapping)
            or row.get("owner_fold_id") != owner
            or row.get("checkpoint_role")
            != "outer_training_only_adapter_checkpoint"
        ):
            raise RuntimeError("R4 AdamW checkpoint owner/role changed")
        checkpoint_path = _reject_fresh_path(
            Path(str(row.get("checkpoint", ""))), role="R4 AdamW checkpoint"
        )
        payload_bytes = checkpoint_path.read_bytes()
        file_sha = hashlib.sha256(payload_bytes).hexdigest()
        if file_sha != row.get("checkpoint_sha256"):
            raise RuntimeError("R4 AdamW checkpoint SHA mismatch")
        checkpoint = torch.load(
            io.BytesIO(payload_bytes), map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("R4 AdamW checkpoint must contain a mapping")
        checkpoint_provenance = checkpoint.get("provenance")
        if (
            checkpoint.get("format") != V8_TRAINING_CHECKPOINT_FORMAT
            or checkpoint.get("optimizer", {}).get("name") != "AdamW"
            or checkpoint.get("all_steps_factual_inputs_bit_exact") is not True
            or checkpoint.get("strict_oof_base_exclusion_eligible") is not True
            or checkpoint.get("fresh_confirmation_data_or_labels_read") is not False
            or checkpoint.get("authorization_guard_changed") is not False
            or not isinstance(checkpoint_provenance, Mapping)
            or checkpoint_provenance.get("outer_fold_id") != owner
            or checkpoint_provenance.get("base_checkpoint_sha256")
            != expected_factual_checkpoint_sha256
            or not isinstance(checkpoint.get("adapter_state_sha256"), str)
            or len(checkpoint["adapter_state_sha256"]) != 64
        ):
            raise RuntimeError("R4 AdamW checkpoint semantic contract changed")
        checkpoint_rows.append(
            {
                "owner_fold_id": owner,
                "path": str(checkpoint_path),
                "file_sha256": file_sha,
                "adapter_state_sha256": checkpoint["adapter_state_sha256"],
            }
        )
    checkpoint_bundle_sha = canonical_sha256(checkpoint_rows)
    return result, {
        "result_path": str(result_path),
        "result_file_sha256": sha256_path(result_path),
        "result_sha256": result_sha,
        "contracts_path": str(contracts_path),
        "contracts_file_sha256": sha256_path(contracts_path),
        "contracts_sha256": contracts_sha,
        "arrays_file_sha256": contracts["arrays_sha256"],
        "bridge_bundle_sha256": bundle_sha,
        "materialization_sha256": expected_materialization_sha256,
        "factual_checkpoint_sha256": expected_factual_checkpoint_sha256,
        "five_checkpoint_bundle_sha256": checkpoint_bundle_sha,
        "checkpoints": checkpoint_rows,
        "required_sha_fields": [
            "result_file_sha256",
            "result_sha256",
            "contracts_file_sha256",
            "contracts_sha256",
            "arrays_file_sha256",
            "bridge_bundle_sha256",
            "materialization_sha256",
            "factual_checkpoint_sha256",
            "five_checkpoint_bundle_sha256",
        ],
    }


def _authenticate_success_inadequacy(
    path: Path, *, expected_materialization_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _reject_fresh_path(path, role="R5 success result")
    result = _load_json(path, role="R5 success result")
    result_sha = _signed(result, "result_sha256", role="R5 success result")
    adequacy = (
        result.get("outer_holdout_evaluation", {})
        .get("pooled_oof", {})
        .get("calibrated_probability_adequacy", {})
    )
    authorization = result.get("authorization")
    contracts = result.get("fold_calibration_contracts")
    if (
        result.get("format") != success_calibration.FORMAT
        or result.get("status") != "complete_adaptive_development_only"
        or result.get("materialization_sha256")
        != expected_materialization_sha256
        or adequacy.get("strict_probability_adequacy") is not False
        or result.get("action_ranking_preserved_within_each_group") is not True
        or result.get("task_success_cannot_change_from_uncalibrated_argmax")
        is not True
        or result.get("outer_holdout_labels_used_for_alpha_selection") is not False
        or result.get("fresh50_inputs_accepted") is not False
        or result.get("fresh50_labels_read") is not False
        or not isinstance(authorization, Mapping)
        or any(authorization.get(key) is not False for key in authorization)
        or not isinstance(contracts, list)
        or len(contracts) != 5
    ):
        raise RuntimeError("R5 success inadequacy evidence changed")
    for owner, contract in enumerate(contracts):
        if not isinstance(contract, Mapping):
            raise RuntimeError("R5 success fold calibration contract is not a mapping")
        unsigned = dict(contract)
        recorded = unsigned.pop("calibration_contract_sha256", None)
        if (
            owner != contract.get("owner_fold_id")
            or recorded != canonical_sha256(unsigned)
            or contract.get("outer_holdout_labels_used_for_alpha_selection")
            is not False
            or contract.get("fresh50_inputs_or_labels_used") is not False
        ):
            raise RuntimeError("R5 success fold calibration contract changed")
    return result, {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "result_sha256": result_sha,
        "materialization_sha256": expected_materialization_sha256,
        "required_sha_fields": [
            "file_sha256",
            "result_sha256",
            "materialization_sha256",
        ],
    }


def freeze_composite_activation(
    *,
    factual_event_result: Path,
    r4_adamw_result: Path,
    r4_adamw_contracts: Path,
    r5_success_result: Path,
    r5_duration_activation: Path,
) -> dict[str, Any]:
    duration_path = _reject_fresh_path(
        r5_duration_activation, role="R5 duration activation"
    )
    duration = load_duration_activation(duration_path)
    duration_evidence = duration["evidence"]
    materialization_sha = duration_evidence["materialization_sha256"]
    factual_checkpoint_sha = duration_evidence["factual_checkpoint_sha256"]
    _, factual_evidence = _authenticate_factual_result(
        factual_event_result,
        expected_materialization_sha256=materialization_sha,
    )
    _, regress_evidence = _authenticate_adamw_regress(
        result_path=r4_adamw_result,
        contracts_path=r4_adamw_contracts,
        expected_materialization_sha256=materialization_sha,
        expected_factual_checkpoint_sha256=factual_checkpoint_sha,
    )
    _, success_evidence = _authenticate_success_inadequacy(
        r5_success_result,
        expected_materialization_sha256=materialization_sha,
    )
    duration_evidence_row = {
        "path": str(duration_path),
        "file_sha256": sha256_path(duration_path),
        "activation_sha256": duration["activation_sha256"],
        "materialization_sha256": materialization_sha,
        "final_hierarchy_contract_sha256": duration[
            "final_hierarchy_contract_sha256"
        ],
        "empirical_registry_contract_sha256": duration[
            "empirical_registry_contract_sha256"
        ],
        "required_sha_fields": [
            "file_sha256",
            "activation_sha256",
            "materialization_sha256",
            "final_hierarchy_contract_sha256",
            "empirical_registry_contract_sha256",
        ],
    }
    root = Path(__file__).resolve().parent
    implementations = {
        filename: sha256_path(root / filename)
        for filename in sorted(IMPLEMENTATION_FILES)
    }
    selector_sha = implementations[
        "openvla_etsf_structured_event_time_utility.py"
    ]
    activation: dict[str, Any] = {
        "format": FORMAT,
        "status": "active_structured_prediction_development_only",
        "evidence_scope": "adaptive_development_only",
        "active": {
            name: {
                "status": "active",
                "prediction_only": True,
                "ranking_input": name in {"next_event", "destination_event"},
            }
            for name in ACTIVE_CAPABILITIES
        },
        "inactive_or_fallback": {
            "success": {
                "status": "inactive",
                "reason": "strict_probability_adequacy_false",
                "ranking_input": False,
            },
            "recovery": {
                "status": "inactive",
                "reason": "strict_conditional_evidence_not_activated",
                "ranking_input": False,
            },
            "object": {
                "status": "fallback_only",
                "reason": "learned_object_output_not_activated",
                "ranking_input": False,
            },
            "total_uncertainty": {
                "status": "unavailable",
                "reason": "single_factual_member_has_aleatoric_only",
                "ranking_input": False,
            },
        },
        "action_selector": {
            "authority": "v7_fixed_parameter_free_selector",
            "implementation": "openvla_etsf_structured_event_time_utility.py",
            "implementation_sha256": selector_sha,
            "format": v7_utility.FORMAT,
            "formula": v7_utility.UTILITY_FORMULA,
            "guard_margin": v7_utility.GUARD_MARGIN,
            "deployment_candidate_count": v7_utility.DEPLOYMENT_CANDIDATE_COUNT,
            "v8_replacement_authorized": False,
            "v8_success_input_allowed": False,
            "v8_regress_input_allowed": False,
            "duration_v2_input_allowed": False,
        },
        "duration_v2_activation": duration,
        "duration_v2_activation_sha256": duration["activation_sha256"],
        "empirical_registry_contract_sha256": duration[
            "empirical_registry_contract_sha256"
        ],
        "empirical_evidence_scope": duration["empirical_evidence_scope"],
        "interface_actor_policy_agnostic": True,
        "transfer_claim_authorized": False,
        "evidence": {
            "factual_event_result": factual_evidence,
            "r4_adamw_regress": regress_evidence,
            "r5_success_inadequacy": success_evidence,
            "r5_duration_activation": duration_evidence_row,
        },
        "implementation_files": implementations,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "fresh50_confirmation_authorized": False,
        "selector_replacement_authorized": False,
        "openvla_gradient_path_allowed": False,
    }
    activation["activation_sha256"] = canonical_sha256(activation)
    validate_composite_activation(activation)
    return activation


def write_composite_activation(path: Path, value: Mapping[str, Any]) -> Path:
    path = _reject_fresh_path(path, role="composite activation output")
    validate_composite_activation(value)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
        return path
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual-event-result", type=Path, required=True)
    parser.add_argument("--r4-adamw-result", type=Path, required=True)
    parser.add_argument("--r4-adamw-contracts", type=Path, required=True)
    parser.add_argument("--r5-success-result", type=Path, required=True)
    parser.add_argument("--r5-duration-activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = freeze_composite_activation(
        factual_event_result=args.factual_event_result,
        r4_adamw_result=args.r4_adamw_result,
        r4_adamw_contracts=args.r4_adamw_contracts,
        r5_success_result=args.r5_success_result,
        r5_duration_activation=args.r5_duration_activation,
    )
    path = write_composite_activation(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(path),
                "activation_sha256": value["activation_sha256"],
                "fresh50_confirmation_authorized": False,
                "selector_replacement_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
