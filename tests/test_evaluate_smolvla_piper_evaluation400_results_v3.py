from __future__ import annotations

import hashlib
import json
import stat
import sys
import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_smolvla_piper_evaluation400_results_v3 as result  # noqa: E402
import smolvla_piper_paired_success_protocol_v3 as paired  # noqa: E402
import select_smolvla_piper_evaluation400_root_candidate_v3 as selector  # noqa: E402
import smolvla_piper_deployment_uncertainty_v1 as uncertainty  # noqa: E402


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bytes(path: Path, payload: bytes, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


def write_json(path: Path, value: Mapping[str, Any]) -> Path:
    return write_bytes(
        path, json.dumps(value, sort_keys=True).encode("utf-8") + b"\n"
    )


def public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def executor_receipt(
    private_key: Ed25519PrivateKey, receipt_format: str, status: str,
    statement: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "format": receipt_format,
        "status": status,
        "signature_algorithm": "Ed25519",
        "statement": dict(statement),
        "executor_signature_ed25519_hex": private_key.sign(
            result._receipt_signing_bytes(receipt_format, statement)
        ).hex(),
    }
    return {**base, "receipt_sha256": result.canonical_sha256(base)}


def record(path: Path, logical_sha: str) -> dict[str, str]:
    return {
        "path": str(path), "file_sha256": file_sha(path),
        "logical_sha256": logical_sha,
    }


def synthetic_selector_proof(
    *, is_etsf: bool, legal: list[bool], selected: int | None = None,
    include_authority: bool = False, runtime_authority_sha256: str = "8" * 64,
) -> Any:
    fallback = legal.index(True)
    if not is_etsf:
        return {
            "selector": "lowest_legal_feasibility_root_candidate",
            "event_model_members_called": 0,
            "selected_candidate_index": fallback,
            "score_contract": "lowest_legal_feasibility_root_candidate",
            "source_rank_score_contract_sha256": [],
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "formal190_target_outcome_calibrated_acceptance_margin": False,
        }
    indices = np.flatnonzero(np.asarray(legal, dtype=bool))
    count = len(indices)
    event = np.zeros((5, count, 5)); event[..., 0] = 20.0
    binary = np.full((5, count), 20.0)
    proposed = selected if selected is not None else int(indices[1])
    rank_row = np.asarray([
        0.0 if int(index) == fallback else (0.5 if int(index) == proposed else 0.2)
        for index in indices
    ], dtype=np.float32)
    ranks = np.tile(rank_row, (5, 1))
    ranker_base = {
        "enabled_for_primary": True,
        "score_is_success_logit": False,
        "score_is_success_probability": False,
        "root_recovery_uncertainty_policy": uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY,
        "root_structured_uncertainty_head_count": 5,
        "selected_candidate": {
            "minimum_group_relative_composite_rank_score_margin": 0.1,
            "maximum_structured_pair_uncertainty": 0.5,
            "maximum_global_candidate_uncertainty": 0.5,
        },
    }
    ranker = {**ranker_base, "root_group_ranker_sha256": result.canonical_sha256(ranker_base)}
    calibration = {
        "metrics": {
            "post_event": {"deployment_temperature": 1.0},
            "next_event": {"deployment_temperature": 1.0},
            "success": {"deployment_temperature": 1.0},
            "conditional_recovery": {"deployment_temperature": 1.0},
            "duration_lognormal_mixture": {
                "deployment_scale_multiplier": 1.0
            },
            "object_total_variance": {
                "deployment_object_error_robust_scale_m": 1.0,
                "deployment_scale_multiplier": 1.0,
            },
        },
        "head_enabled_for_primary": {
            "post_event": True, "next_event": True, "duration": True,
            "success": True, "object_effect": True, "recovery": True,
        },
        "all_six_heads_support_performance_uncertainty_gate_passed": True,
        "root_group_ranker": ranker,
        "abstain_threshold": {"enabled": True, "maximum_total_uncertainty": 0.5},
    }
    calibration["calibration_sha256"] = result.canonical_sha256(calibration)
    uncertainty_path = Path(uncertainty.__file__).resolve()
    contracts = []
    for member_index in range(5):
        contract_base = {
            "source_checkpoint_file_sha256": str(member_index + 1) * 64,
            "base_score": "candidate_rank_score",
            "source_action_rank_residual": True,
            "source_action_rank_success_only": False,
            "source_rank_numeric_contract": selector.SOURCE_RANK_NUMERIC_CONTRACT,
            "residual_combination": (
                "candidate_rank_score_plus_action_rank_residual"
            ),
            "success_temperature": 1.0,
        }
        contracts.append({
            **contract_base,
            "contract_sha256": result.canonical_sha256(contract_base),
        })
    source_rank_member_authority = {
        "source_rank_numeric_contract": selector.SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": member_index,
                "source_checkpoint_file_sha256": contract[
                    "source_checkpoint_file_sha256"
                ],
                "source_rank_score_contract_sha256": contract[
                    "contract_sha256"
                ],
                "success_temperature": contract["success_temperature"],
            }
            for member_index, contract in enumerate(contracts)
        ],
    }
    formal_thresholds = {
        "minimum_formal190_composite_margin": 0.1,
        "maximum_formal190_pair_uncertainty": 0.5,
        "maximum_global_total_uncertainty": 0.5,
        "root_group_ranker_sha256": ranker["root_group_ranker_sha256"],
    }
    authority = {
        "format": "etsf_smolvla_piper_evaluation400_root_selector_authority_v3",
        "status": "frozen_formal190_runtime_bound_composite_selector",
        "implementation": {
            "path": str(Path(selector.__file__).resolve()),
            "file_sha256": file_sha(Path(selector.__file__).resolve()),
        },
        "utility_contract": {
            "primary_score": selector.PRIMARY_SCORE,
            "primary_score_is_success_logit": False,
            "primary_score_is_success_probability": False,
            "scene_relative_candidate_comparison": True,
            "source_action_rank_residual_required": True,
            "source_action_rank_success_only": False,
            "piper_embodiment_adapter_required": True,
            "formal190_target_outcome_calibrated_acceptance_margin": True,
            "structured_heads_enter_primary_utility": False,
            "structured_heads_enter_uncertainty_and_ablation": True,
            "margin_comparison": "strict_greater_than_formal190_threshold",
            "alternative_set_contract": (
                "all_legal_candidates_except_lowest_legal_baseline"
            ),
        },
        "calibration_sha256": calibration["calibration_sha256"],
        "formal190_root_group_ranker_sha256": ranker["root_group_ranker_sha256"],
        "source_rank_score_contract_sha256": [
            contract["contract_sha256"] for contract in contracts
        ],
        "source_rank_score_contracts": contracts,
        "source_rank_numeric_contract": selector.SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": source_rank_member_authority,
        "source_rank_member_authority_sha256": result.canonical_sha256(
            source_rank_member_authority
        ),
        "runtime_execution_authority_sha256": runtime_authority_sha256,
        "uncertainty_contract": {
            "formal190_object_error_robust_scale_m": 1.0,
            "duration_deployment_scale_applied_before_selector": True,
            "object_deployment_scale_applied_before_selector": True,
            "object_predictions_physical_xyz_before_selector": True,
            "root_recovery_uncertainty_policy": uncertainty.ROOT_RECOVERY_UNCERTAINTY_POLICY,
            "root_structured_uncertainty_head_count": 5,
            "deployment_uncertainty_contract_sha256": "e" * 64,
        },
        "deployment_parameters": {
            "post_event_temperature": 1.0,
            "next_event_temperature": 1.0,
            "success_temperature": 1.0,
            "conditional_recovery_temperature": 1.0,
            "duration_scale_multiplier": 1.0,
            "object_scale_multiplier": 1.0,
            "object_error_robust_scale_m": 1.0,
            "deployment_uncertainty_contract_sha256": "e" * 64,
        },
        "formal190_thresholds": formal_thresholds,
        "five_member_checkpoint_sha256": ["f" * 64] * 5,
        "object_source_normalization_sha256": ["0" * 64] * 5,
        "deployment_uncertainty_implementation": {
            "path": str(uncertainty_path), "file_sha256": file_sha(uncertainty_path),
        },
    }
    authority["selector_authority_sha256"] = result.canonical_sha256(authority)
    residual = np.full((5, count), 0.1, dtype=np.float32)
    raw = selector.select_root_candidate_v3(
        predictions={
            "post_event_logits": event, "next_event_logits": event,
            "duration_log_mean": np.zeros((5, count)),
            "duration_log_scale": np.full((5, count), -8.0),
            "success_logit": binary, "recovery_logit": binary,
            "object_mean": np.zeros((5, count, 3)),
            "object_log_scale": np.full((5, count, 3), -8.0),
            "source_contract_rank_score": ranks,
            "source_contract_base_rank_score": ranks - residual,
            "source_action_rank_residual": residual,
        },
        prediction_candidate_indices=indices,
        candidate_legal=np.asarray(legal, dtype=bool), fallback_index=fallback,
        calibration=calibration, selector_authority=authority,
    )
    base = {
        "selector": "frozen_five_member_event_world_model_with_uncertainty_abstention",
        "event_model_members_called": 5, "uncertainty_gate_applied": True,
        "selector_output_sha256": raw["selector_proof_sha256"],
        "selector_decision": raw,
        "selected_candidate_index": raw["selected_candidate_index"],
        "proposed_candidate_index": raw["proposed_candidate_index"],
        "score_margin": raw["score_margin"], "total_uncertainty": raw["total_uncertainty"],
        "decision_algebra_sha256": raw["decision_algebra_sha256"],
        "calibration_sha256": raw["calibration_sha256"],
        "formal190_root_group_ranker_sha256": raw["formal190_root_group_ranker_sha256"],
        "score_contract": selector.PRIMARY_SCORE,
        "source_rank_score_contract_sha256": authority["source_rank_score_contract_sha256"],
        "source_rank_numeric_contract": selector.SOURCE_RANK_NUMERIC_CONTRACT,
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "formal190_target_outcome_calibrated_acceptance_margin": True,
        "predicted_success_used_as_outcome": False,
    }
    proof = {**base, "selector_proof_sha256": result.canonical_sha256(base)}
    return (proof, authority) if include_authority else proof


def test_evaluator_recomputes_selector_components_after_full_resign() -> None:
    legal = [False, True, True, True]
    proof = synthetic_selector_proof(is_etsf=True, legal=legal, selected=2)
    decision = dict(proof["selector_decision"])
    components = dict(decision["uncertainty_components"])
    structured = list(components["structured_five_head"])
    structured[1] += 0.01
    components["structured_five_head"] = structured
    decision["uncertainty_components"] = components
    decision["selector_proof_sha256"] = result.canonical_sha256({
        key: value for key, value in decision.items() if key != "selector_proof_sha256"
    })
    proof["selector_decision"] = decision
    proof["selector_output_sha256"] = decision["selector_proof_sha256"]
    proof["selector_proof_sha256"] = result.canonical_sha256({
        key: value for key, value in proof.items() if key != "selector_proof_sha256"
    })
    with pytest.raises(result.Evaluation400ResultError, match="uncertainty changed"):
        result.validate_selector_execution_proof(
            proof, condition="etsf", candidate_legal=legal,
            selected_candidate_index=2,
        )


def _resign_selector_proof(proof: dict[str, Any]) -> dict[str, Any]:
    decision = proof["selector_decision"]
    decision["selector_input_sha256"] = result.canonical_sha256(
        decision["selector_input"]
    )
    decision["selector_proof_sha256"] = result.canonical_sha256({
        key: value for key, value in decision.items()
        if key != "selector_proof_sha256"
    })
    proof["selector_output_sha256"] = decision["selector_proof_sha256"]
    proof["selector_proof_sha256"] = result.canonical_sha256({
        key: value for key, value in proof.items()
        if key != "selector_proof_sha256"
    })
    return proof


@pytest.mark.parametrize(
    ("field", "persisted"),
    (
        ("source_contract_base_rank_score", "member_source_contract_base_rank_scores"),
        ("source_action_rank_residual", "member_source_action_rank_residuals"),
        ("source_contract_rank_score", "member_source_contract_rank_scores"),
    ),
)
def test_evaluator_rejects_resigned_source_rank_component_tamper(
    field: str, persisted: str,
) -> None:
    legal = [False, True, True, True]
    proof = synthetic_selector_proof(is_etsf=True, legal=legal, selected=2)
    proof = copy.deepcopy(proof)
    predictions = proof["selector_decision"]["selector_input"]["predictions"]
    predictions[field][0][0] += 0.125
    proof["selector_decision"][persisted][0][0] += 0.125
    _resign_selector_proof(proof)
    with pytest.raises(result.Evaluation400ResultError, match="composite member"):
        result.validate_selector_execution_proof(
            proof, condition="etsf", candidate_legal=legal,
            selected_candidate_index=2,
        )


def test_evaluator_rejects_one_float32_ulp_composite_tamper() -> None:
    legal = [False, True, True, True]
    proof = copy.deepcopy(
        synthetic_selector_proof(is_etsf=True, legal=legal, selected=2)
    )
    predictions = proof["selector_decision"]["selector_input"]["predictions"]
    changed = np.nextafter(
        np.float32(predictions["source_contract_rank_score"][0][0]),
        np.float32(np.inf), dtype=np.float32,
    )
    predictions["source_contract_rank_score"][0][0] = float(changed)
    proof["selector_decision"]["member_source_contract_rank_scores"][0][0] = (
        float(changed)
    )
    _resign_selector_proof(proof)
    with pytest.raises(result.Evaluation400ResultError, match="composite member"):
        result.validate_selector_execution_proof(
            proof, condition="etsf", candidate_legal=legal,
            selected_candidate_index=2,
        )


@pytest.mark.parametrize("indices", ([1, 2], [1, 1, 2], [0, 1, 2, 3]))
def test_evaluator_rejects_incomplete_duplicate_or_extra_legal_candidates(
    indices: list[int],
) -> None:
    legal = [False, True, True, True]
    proof = copy.deepcopy(
        synthetic_selector_proof(is_etsf=True, legal=legal, selected=2)
    )
    proof["selector_decision"]["selector_input"][
        "prediction_candidate_indices"
    ] = indices
    _resign_selector_proof(proof)
    with pytest.raises(result.Evaluation400ResultError, match="candidate input"):
        result.validate_selector_execution_proof(
            proof, condition="etsf", candidate_legal=legal,
            selected_candidate_index=2,
        )


@pytest.mark.parametrize("mutation", ("temperature", "robust_scale", "threshold"))
def test_proof_scientific_parameters_must_equal_paired_authority(
    mutation: str,
) -> None:
    proof, authority = synthetic_selector_proof(
        is_etsf=True, legal=[False, True, True, True], selected=2,
        include_authority=True,
    )
    proof = copy.deepcopy(proof)
    selector_input = proof["selector_decision"]["selector_input"]
    if mutation == "temperature":
        proof["selector_decision"][
            "member_source_rank_success_temperatures"
        ][0] = 2.0
    elif mutation == "robust_scale":
        selector_input["uncertainty_parameters"][
            "object_error_robust_scale_m"
        ] = 2.0
        selector_input["deployment_parameters"][
            "object_error_robust_scale_m"
        ] = 2.0
    else:
        selector_input["formal190_thresholds"][
            "minimum_formal190_composite_margin"
        ] = 0.2
    with pytest.raises(
        result.Evaluation400ResultError,
        match="parameters differ",
    ):
        result.validate_selector_proof_against_authority(proof, authority)


@pytest.mark.parametrize(
    "mutation",
    (
        "member_source_checkpoint", "member_temperature", "contract_numeric",
        "member_index_bool", "decision_temperature_bool",
    ),
)
def test_evaluator_rejects_resigned_source_rank_member_authority_tamper(
    mutation: str,
) -> None:
    proof, authority = synthetic_selector_proof(
        is_etsf=True, legal=[False, True, True, True], selected=2,
        include_authority=True,
    )
    proof = copy.deepcopy(proof)
    authority = copy.deepcopy(authority)
    member_authority = authority["source_rank_member_authority"]
    member = member_authority["members"][0]
    if mutation == "member_source_checkpoint":
        member["source_checkpoint_file_sha256"] = "f" * 64
    elif mutation == "member_temperature":
        member["success_temperature"] = 2.0
    elif mutation == "contract_numeric":
        contract = dict(authority["source_rank_score_contracts"][0])
        contract.pop("contract_sha256")
        contract["source_rank_numeric_contract"] = "ieee754_float64_reassociated"
        contract = {
            **contract, "contract_sha256": result.canonical_sha256(contract)
        }
        authority["source_rank_score_contracts"][0] = contract
        authority["source_rank_score_contract_sha256"][0] = contract[
            "contract_sha256"
        ]
        member["source_rank_score_contract_sha256"] = contract[
            "contract_sha256"
        ]
        proof["source_rank_score_contract_sha256"][0] = contract[
            "contract_sha256"
        ]
        proof["selector_decision"]["source_rank_score_contract_sha256"][0] = (
            contract["contract_sha256"]
        )
    elif mutation == "member_index_bool":
        member["member_index"] = False
    else:
        proof["selector_decision"][
            "member_source_rank_success_temperatures"
        ][0] = True
    authority["source_rank_member_authority_sha256"] = result.canonical_sha256(
        member_authority
    )
    authority["selector_authority_sha256"] = result.canonical_sha256({
        key: value for key, value in authority.items()
        if key != "selector_authority_sha256"
    })
    proof["selector_decision"]["selector_input"][
        "selector_authority_sha256"
    ] = authority["selector_authority_sha256"]
    _resign_selector_proof(proof)
    with pytest.raises(
        result.Evaluation400ResultError,
        match="Source rank",
    ):
        result.validate_selector_proof_against_authority(proof, authority)


def test_decision_algebra_sha_is_recomputed() -> None:
    legal = [False, True, True, True]
    proof = copy.deepcopy(
        synthetic_selector_proof(is_etsf=True, legal=legal, selected=2)
    )
    proof["selector_decision"]["decision_algebra_sha256"] = "f" * 64
    proof["decision_algebra_sha256"] = "f" * 64
    _resign_selector_proof(proof)
    with pytest.raises(result.Evaluation400ResultError, match="strict gate"):
        result.validate_selector_execution_proof(
            proof, condition="etsf", candidate_legal=legal,
            selected_candidate_index=2,
        )


def execution_artifacts(
    root: Path, *, pair: Mapping[str, Any], position: int,
    ordered: list[str], legal: list[bool], selected: int, success: bool,
    runtime_authority_sha256: str = "8" * 64,
    runtime_contract_sha256: str = "9" * 64,
) -> dict[str, Any]:
    """Materialize the exact subprocess closure consumed by the evaluator."""
    stage_root = root / "stage"
    output_root = stage_root / "result"
    output_root.mkdir(parents=True)
    request_path = root / "request.json"
    request_base = {
        "format": "etsf_smolvla_piper_evaluation400_condition_request_v3",
        "status": "write_ahead_before_condition_popen",
        "plan_sha256": "1" * 64,
        "bundle_sha256": "2" * 64,
        "claim_sha256": "3" * 64,
        "pair_id": pair["pair_id"],
        "ordinal": pair["ordinal"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "initial_scene_state_sha256": pair["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": pair[
            "initial_measured_joint_state_sha256"
        ],
        "initial_commanded_drive_target_sha256": pair[
            "initial_commanded_drive_target_sha256"
        ],
        "attempt": 0,
        "pair_identity_sha256": "4" * 64,
        "condition": pair["condition_order"][position],
        "condition_ordinal": position,
        "condition_order": list(pair["condition_order"]),
        "shared_snapshot_sha256": "5" * 64,
        "candidate_count": 4,
        "candidate_generation_contract_sha256": "6" * 64,
        "postfreeze_identity_or_order_change_authorized": False,
        "outcome_visible_before_condition_start": False,
    }
    request = {
        **request_base,
        "request_sha256": result.canonical_sha256(request_base),
    }
    write_json(request_path, request)
    trajectory_path = write_bytes(output_root / "trajectory.bin", b"synthetic")
    continuation_path = write_bytes(output_root / "continuation.bin", b"synthetic")
    is_etsf = pair["condition_order"][position] == "etsf"
    selector_proof = synthetic_selector_proof(
        is_etsf=is_etsf, legal=legal, selected=selected,
        runtime_authority_sha256=runtime_authority_sha256,
    )
    selected = selector_proof["selected_candidate_index"]
    runner_base = {
        "format": result.RUNNER_RESULT_FORMAT,
        "status": result.RUNNER_RESULT_STATUS,
        "request": record(request_path, request["request_sha256"]),
        "pair_id": pair["pair_id"],
        "ordinal": pair["ordinal"],
        "attempt": 0,
        "condition": pair["condition_order"][position],
        "condition_ordinal": position,
        "shared_snapshot_sha256": request["shared_snapshot_sha256"],
        "candidate_count": 4,
        "candidate_registry_sha256": result._candidate_registry_sha(
            pair["pair_id"], ordered, legal
        ),
        "schema6_execution_authority_file_sha256": runtime_authority_sha256,
        "schema6_runtime_contract_sha256": runtime_contract_sha256,
        "max_episode_steps": 200,
        "ordered_candidate_sha256": list(ordered),
        "candidate_legal": list(legal),
        "selected_candidate_index": selected,
        "selector_execution_proof": selector_proof,
        "selector_execution_proof_sha256": result.canonical_sha256(selector_proof),
        "selector_score_contract": selector_proof["score_contract"],
        "source_rank_score_contract_sha256": selector_proof[
            "source_rank_score_contract_sha256"
        ],
        "source_contract_rank_score_is_success_logit": False,
        "source_contract_rank_score_is_success_probability": False,
        "formal190_target_outcome_calibrated_acceptance_margin": is_etsf,
        "continuation_contract": result.CONTINUATION_CONTRACT,
        "continuation_policy_sha256": "7" * 64,
        "continuation_rerank_after_root": False,
        "candidate_replacement_count": 0,
        "continuation_proof_sha256": result.canonical_sha256({
            "continuation_contract": result.CONTINUATION_CONTRACT,
            "continuation_policy_sha256": "7" * 64,
            "continuation_rerank_after_root": False,
            "candidate_replacement_count": 0,
        }),
        "task_success": success,
        "trajectory_artifact": {
            "path": str(trajectory_path), "file_sha256": file_sha(trajectory_path)
        },
        "continuation_artifact": {
            "path": str(continuation_path),
            "file_sha256": file_sha(continuation_path),
        },
        "simulator_exit_code": 0,
    }
    runner = {**runner_base, "result_sha256": result.canonical_sha256(runner_base)}
    runner_path = write_json(output_root / "condition_result.json", runner)

    python_path = Path(sys.executable).resolve()
    runner_source = ROOT / "scripts" / "run_smolvla_piper_evaluation400_condition_v3.py"
    original = [
        str(python_path), str(runner_source), "--request", str(request_path),
        "--output-root", str(output_root), "--device", "cuda:0",
    ]
    fd_mapping = [
        {
            "role": role,
            "source_path": str(path),
            "source_file_sha256": file_sha(path),
            "inherited_fd": descriptor,
            "executed_path": f"/proc/self/fd/{descriptor}",
        }
        for role, path, descriptor in (
            ("runtime_python", python_path, 10),
            ("condition_runner", runner_source, 11),
        )
    ]
    launch_base = {
        "format": result.STAGE_FORMAT,
        "status": "fd_bound_guard_passed_immediately_before_popen",
        "original_command": original,
        "command_sha256": result.canonical_sha256(original),
        "executed_command": [
            fd_mapping[0]["executed_path"], "-I",
            fd_mapping[1]["executed_path"], *original[2:],
        ],
        "fd_mapping": fd_mapping,
        "isolated_python": True,
        "environment_policy": (
            "explicit_allowlist_no_pythonpath_pythonhome_or_ld_preload"
        ),
        "environment_keys": [
            "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "LANG", "LC_ALL", "PATH",
            "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED",
            "TRANSFORMERS_OFFLINE",
        ],
        "forbidden_environment_keys_absent": sorted(
            ["PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"]
        ),
        "gpu_uuid": "GPU-synthetic",
        "device": "cuda:0",
    }
    launch = {
        **launch_base, "launch_sha256": result.canonical_sha256(launch_base)
    }
    launch_path = write_json(stage_root / "launch.json", launch)
    lifecycle_base = {
        "popen_attempted": True,
        "popen_reached": True,
        "process_pid": 123,
        "process_pgid": 123,
        "process_group_isolated": True,
        "returncode": 0,
        "direct_process_reaped": True,
        "process_group_reaped": True,
        "binding_status": "bound_reaped",
    }
    lifecycle = {
        **lifecycle_base,
        "lifecycle_sha256": result.canonical_sha256(lifecycle_base),
    }
    lifecycle_path = write_json(stage_root / "lifecycle.json", lifecycle)
    idle_base = {
        "gpu_index": 0,
        "gpu_name": "NVIDIA RTX 4090",
        "gpu_uuid": "GPU-synthetic",
        "checks": 2,
    }
    idle = {**idle_base, "audit_sha256": result.canonical_sha256(idle_base)}
    idle_before = write_json(root / "gpu_idle_before_condition.json", idle)
    idle_after = write_json(root / "gpu_idle_after_condition.json", idle)
    stage_log = write_bytes(stage_root / "run.log", b"")
    stage_exit = write_bytes(stage_root / "run.exit", b"0\n")

    def opaque(path: Path) -> dict[str, str]:
        return {"path": str(path), "file_sha256": file_sha(path)}

    return {
        "runner_result": record(runner_path, runner["result_sha256"]),
        "stage_launch": record(launch_path, launch["launch_sha256"]),
        "stage_lifecycle": record(
            lifecycle_path, lifecycle["lifecycle_sha256"]
        ),
        "stage_log": opaque(stage_log),
        "stage_exit": opaque(stage_exit),
        "gpu_idle_before": record(idle_before, idle["audit_sha256"]),
        "gpu_idle_after": record(idle_after, idle["audit_sha256"]),
        "gpu_uuid": "GPU-synthetic",
    }


def core_pair(ordinal: int) -> dict[str, Any]:
    pair_id = hashlib.sha256(f"evaluation-pair-{ordinal:03d}".encode()).hexdigest()
    digest = lambda role: hashlib.sha256(f"{role}:{ordinal}".encode()).hexdigest()
    return {
        "ordinal": ordinal,
        "pair_id": pair_id,
        "target_manifest_global_ordinal": 10_000 + ordinal,
        "requested_seed": 20_000 + ordinal,
        "resolved_seed": 30_000 + ordinal,
        "initial_scene_state_sha256": digest("scene"),
        "initial_measured_joint_state_sha256": digest("measured"),
        "initial_commanded_drive_target_sha256": digest("commanded"),
        "condition_order": paired.bridge_v2.paired_condition_order(pair_id),
        "candidate_count": 4,
    }


def synthetic_closure(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "synthetic_eval400"
    root.mkdir()
    issuer = Ed25519PrivateKey.from_private_bytes(b"\x11" * 32)
    executor = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32)
    result_signer = Ed25519PrivateKey.from_private_bytes(b"\x33" * 32)
    issuer_public = public_bytes(issuer)
    executor_public = public_bytes(executor)
    result_public = public_bytes(result_signer)
    pairs = [core_pair(index) for index in range(result.PAIR_COUNT)]
    pair_identity_set_sha = result.canonical_sha256(
        [{"ordinal": row["ordinal"], "pair_id": row["pair_id"]} for row in pairs]
    )
    runtime_authority_path = write_json(
        root / "schema6_runtime_execution_authority.json",
        {
            "format": "synthetic_schema6_runtime_execution_authority_v2",
            "runtime_contract": {
                "runtime_contract_sha256": "9" * 64,
                "max_episode_steps": 200,
            },
        },
    )
    runtime_authority_sha = file_sha(runtime_authority_path)
    selector_fixture, selector_authority = synthetic_selector_proof(
        is_etsf=True, legal=[False, True, True, True], selected=2,
        include_authority=True,
        runtime_authority_sha256=runtime_authority_sha,
    )
    deployment_uncertainty_contract_sha = selector_authority[
        "deployment_parameters"
    ]["deployment_uncertainty_contract_sha256"]
    source_rank_contracts = selector_authority[
        "source_rank_score_contract_sha256"
    ]
    core_base: dict[str, Any] = {
        "format": paired.CORE_FORMAT,
        "status": paired.CORE_STATUS,
        "post_collection_v3": {},
        "development_and_formal190": {
            "source_rank_score_contracts": selector_authority[
                "source_rank_score_contracts"
            ],
            "source_rank_member_authority": selector_authority[
                "source_rank_member_authority"
            ],
            "source_rank_member_authority_sha256": selector_authority[
                "source_rank_member_authority_sha256"
            ],
            "deployment_parameters": selector_authority["deployment_parameters"],
            "formal190_thresholds": selector_authority["formal190_thresholds"],
            "formal190_root_group_ranker_sha256": selector_fixture[
                "formal190_root_group_ranker_sha256"
            ],
            "formal190_deployment_uncertainty_contract_sha256": (
                deployment_uncertainty_contract_sha
            ),
            "maximum_total_uncertainty_fixed_point": {
                "coefficient": 5, "decimal_places": 1
            },
            "minimum_composite_margin_fixed_point": {
                "coefficient": 1, "decimal_places": 1
            },
            "maximum_pair_uncertainty_fixed_point": {
                "coefficient": 5, "decimal_places": 1
            },
        },
        "r7h_target_adapter_lineage": {
            "member_count": 5,
            "single_checkpoint_accepted": False,
            "lobo_checkpoint_accepted": False,
            "joint_teacher_accepted": False,
        },
        "evaluation400": {
            "pair_identity_set_sha256": pair_identity_set_sha,
            "pair_count": 400,
            "only_final_paired_lane": True,
            "additional_reserve400_count": 0,
            "postfreeze_seed_candidate_threshold_or_order_change_allowed": False,
            "pairs": pairs,
        },
        "deployment": {
            "deployment_binding_sha256": "a" * 64,
            "policy_runtime_action_binding_sha256": "b" * 64,
            "candidate_count": 4,
            "baseline_selector": "lowest_legal_feasibility_root_candidate",
            "etsf_selector": (
                "frozen_five_member_event_world_model_with_uncertainty_abstention"
            ),
            "fallback": "baseline",
            "runtime_execution_authority": {
                "path": str(runtime_authority_path),
                "file_sha256": runtime_authority_sha,
                "nested_runtime_contract_sha256": "9" * 64,
                "max_episode_steps": 200,
            },
            "selector_authority": selector_authority,
            "selector_authority_sha256": selector_authority[
                "selector_authority_sha256"
            ],
            "formal190_root_group_ranker_sha256": selector_fixture[
                "formal190_root_group_ranker_sha256"
            ],
            "deployment_uncertainty_contract_sha256": (
                deployment_uncertainty_contract_sha
            ),
            "source_rank_score_contract_sha256": source_rank_contracts,
            "source_rank_score_contracts": selector_authority[
                "source_rank_score_contracts"
            ],
            "source_rank_member_authority": selector_authority[
                "source_rank_member_authority"
            ],
            "source_rank_member_authority_sha256": selector_authority[
                "source_rank_member_authority_sha256"
            ],
            "deployment_parameters": selector_authority["deployment_parameters"],
            "formal190_thresholds": selector_authority["formal190_thresholds"],
            "target_reset_runtime_contract_sha256": "9" * 64,
        },
        "execution_inventory": {
            "attestation": {
                "path": str(root / "execution_inventory_attestation.json"),
                "file_sha256": "4" * 64,
                "logical_sha256": "5" * 64,
            },
            "stack_binding_sha256": "6" * 64,
            "executor_identity_sha256": hashlib.sha256(executor_public).hexdigest(),
            "executor_implementation_file_sha256": "7" * 64,
            "result_evaluator_identity_sha256": hashlib.sha256(result_public).hexdigest(),
            "result_evaluator_implementation_file_sha256": file_sha(Path(result.__file__)),
            "real_execution_components_complete": True,
        },
        "authority_policy": {
            "signature_algorithm": "Ed25519",
            "signature_context_utf8": paired.SIGNATURE_CONTEXT[:-1].decode("utf-8"),
            "issuer_key_id": "issuer-synthetic-1",
            "issuer_public_key_hex": issuer_public.hex(),
            "issuer_public_key_sha256": hashlib.sha256(issuer_public).hexdigest(),
            "issuer_identity_sha256": "c" * 64,
            "trusted_issuer_attestation_sha256": "8" * 64,
            "executor_identity_sha256": hashlib.sha256(executor_public).hexdigest(),
            "result_evaluator_identity_sha256": hashlib.sha256(result_public).hexdigest(),
            "authorization_sequence": 1,
            "core_itself_authorizes_execution": False,
        },
        "result_protocol": paired._result_protocol(),
        "preexecution_capability_receipt": {
            "hdf5_files_opened": 0,
            "trajectory_files_opened": 0,
            "prediction_files_opened": 0,
            "label_or_outcome_files_opened": 0,
            "checkpoint_files_hashed_as_opaque_bytes": 10,
            "checkpoint_deserialization_calls": 0,
            "policy_or_simulator_calls": 0,
            "pair_conditions_executed": 0,
        },
        "execution_authorized": False,
    }
    core = {**core_base, "protocol_core_sha256": result.canonical_sha256(core_base)}
    paired.validate_core(core)
    core_path = write_json(root / "paired_core.json", core)
    decision_statement = paired.expected_decision_statement(
        core,
        core_file_sha256=file_sha(core_path),
        decision_nonce_hex=(b"\x44" * 32).hex(),
    )
    decision_base = {
        "format": paired.DECISION_FORMAT,
        "status": paired.DECISION_STATUS,
        "signature_algorithm": "Ed25519",
        "statement": decision_statement,
        "decision_signature_ed25519_hex": issuer.sign(
            paired.decision_signing_bytes(decision_statement)
        ).hex(),
    }
    decision = {
        **decision_base, "decision_sha256": result.canonical_sha256(decision_base)
    }
    decision_path = write_json(root / "decision.json", decision)
    capability = {
        "hdf5_files_opened": 0,
        "trajectory_files_opened": 0,
        "label_or_outcome_files_opened": 0,
        "checkpoint_deserialization_calls": 0,
        "policy_or_simulator_calls": 0,
        "pair_conditions_executed": 0,
    }
    bundle_base = {
        "format": paired.BUNDLE_FORMAT,
        "status": paired.BUNDLE_STATUS,
        "protocol_core": record(core_path, core["protocol_core_sha256"]),
        "ed25519_decision": record(decision_path, decision["decision_sha256"]),
        "issuer_key_id": core["authority_policy"]["issuer_key_id"],
        "issuer_public_key_sha256": core["authority_policy"]["issuer_public_key_sha256"],
        "trusted_issuer_attestation_sha256": core["authority_policy"][
            "trusted_issuer_attestation_sha256"
        ],
        "executor_identity_sha256": core["authority_policy"]["executor_identity_sha256"],
        "result_evaluator_identity_sha256": core["authority_policy"][
            "result_evaluator_identity_sha256"
        ],
        "execution_inventory": core["execution_inventory"],
        "pair_identity_set_sha256": pair_identity_set_sha,
        "deployment_binding_sha256": core["deployment"]["deployment_binding_sha256"],
        "authorized_pair_count": 400,
        "additional_reserve400_count": 0,
        "external_executor_only": True,
        "protocol_freezer_may_execute": False,
        "execution_authorized": True,
        "capability_receipt": capability,
    }
    bundle = {**bundle_base, "bundle_sha256": result.canonical_sha256(bundle_base)}
    paired.validate_bundle(bundle)
    bundle_path = write_json(root / "bundle.json", bundle)
    bootstrap_path = write_bytes(root / "bootstrap_draws.u16", result.bootstrap_draw_bytes())
    bootstrap_logical = result.canonical_sha256({
        "format": result.BOOTSTRAP_FORMAT,
        "shape": list(result.BOOTSTRAP_SHAPE),
        "seed": result.BOOTSTRAP_SEED,
        "generator": result.BOOTSTRAP_GENERATOR,
        "file_sha256": file_sha(bootstrap_path),
    })
    execution_nonce = (b"\x55" * 32).hex()
    dependency_rehash = "d" * 64
    common = {
        "protocol_core_sha256": core["protocol_core_sha256"],
        "decision_sha256": decision["decision_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "execution_nonce_hex": execution_nonce,
        "pair_identity_set_sha256": pair_identity_set_sha,
        "deployment_binding_sha256": core["deployment"]["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": core["deployment"][
            "policy_runtime_action_binding_sha256"
        ],
        "preexecution_dependency_rehash_sha256": dependency_rehash,
    }
    ledger_id = hashlib.sha256(b"synthetic-evaluation400-ledger").hexdigest()
    claim_statement = {
        **{
            field: common[field]
            for field in (
                "protocol_core_sha256", "decision_sha256", "bundle_sha256",
                "execution_nonce_hex", "pair_identity_set_sha256",
                "deployment_binding_sha256", "policy_runtime_action_binding_sha256",
            )
        },
        "ledger_id_sha256": ledger_id,
        "claim_ordinal": 0,
        "claim_count": 1,
        "claim_release_count": 0,
        "claimed_before_any_outcome_read": True,
        "outcome_or_success_values_read_before_claim": 0,
        "retry_or_reclaim_authorized": False,
    }
    claim_receipt = executor_receipt(
        executor, result.CLAIM_FORMAT, result.CLAIM_STATUS, claim_statement
    )
    claim_path = write_json(root / "ledger" / "execution_claim.json", claim_receipt)
    claim_record = record(claim_path, claim_receipt["receipt_sha256"])
    previous_event_sha = claim_receipt["receipt_sha256"]
    ledger_event_records: list[dict[str, Any]] = []

    def append_ledger_event(
        event_type: str, *, pair: Mapping[str, Any] | None,
        position: int | None, artifact_receipt_sha256: str | None,
    ) -> str:
        nonlocal previous_event_sha
        event_index = len(ledger_event_records)
        pair_ordinal = pair["ordinal"] if pair is not None else None
        if pair is not None and position is not None:
            global_ordinal = 2 * pair["ordinal"] + position
            condition_id = pair["condition_order"][position]
        else:
            global_ordinal = None
            condition_id = None
        event_statement = {
            **common,
            "ledger_id_sha256": ledger_id,
            "event_index": event_index,
            "event_type": event_type,
            "previous_entry_sha256": previous_event_sha,
            "pair_ordinal": pair_ordinal,
            "global_condition_ordinal": global_ordinal,
            "condition_position": position,
            "condition_id": condition_id,
            "artifact_receipt_sha256": artifact_receipt_sha256,
            "outcome_or_success_read_before_event": (
                event_type != "condition_started_preoutcome"
            ),
        }
        event_receipt = executor_receipt(
            executor, result.LEDGER_EVENT_FORMAT, result.LEDGER_EVENT_STATUS,
            event_statement,
        )
        event_path = write_json(
            root / "ledger" / f"event_{event_index:04d}.json", event_receipt
        )
        ledger_event_records.append({
            "event_index": event_index,
            "event_type": event_type,
            "pair_ordinal": pair_ordinal,
            "global_condition_ordinal": global_ordinal,
            **record(event_path, event_receipt["receipt_sha256"]),
        })
        previous_event_sha = event_receipt["receipt_sha256"]
        return previous_event_sha

    condition_records: list[dict[str, Any]] = []
    condition_statements: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    for pair in pairs:
        ordinal = pair["ordinal"]
        ordered = [
            hashlib.sha256(f"candidate:{ordinal}:{index}".encode()).hexdigest()
            for index in range(4)
        ]
        legal = [False, True, True, True]
        candidate_registry = result._candidate_registry_sha(
            pair["pair_id"], ordered, legal
        )
        continuation_policy = hashlib.sha256(
            f"continuation:{ordinal}".encode()
        ).hexdigest()
        pair_condition_records: list[dict[str, Any]] = []
        pair_condition_statements: list[dict[str, Any]] = []
        if ordinal < 80:
            success = {"baseline": False, "etsf": False}
        elif ordinal < 160:
            success = {"baseline": True, "etsf": False}
        elif ordinal < 280:
            success = {"baseline": False, "etsf": True}
        else:
            success = {"baseline": True, "etsf": True}
        condition_terminal_event_sha256: list[str] = []
        for position, condition_id in enumerate(pair["condition_order"]):
            condition_start_event_sha256 = append_ledger_event(
                "condition_started_preoutcome", pair=pair, position=position,
                artifact_receipt_sha256=None,
            )
            statement: dict[str, Any] = {
                **common,
                "ledger_condition_start_event_sha256": (
                    condition_start_event_sha256
                ),
                "global_condition_ordinal": 2 * ordinal + position,
                "pair_ordinal": ordinal,
                "pair_id": pair["pair_id"],
                "target_manifest_global_ordinal": pair["target_manifest_global_ordinal"],
                "requested_seed": pair["requested_seed"],
                "resolved_seed": pair["resolved_seed"],
                "condition_position": position,
                "condition_id": condition_id,
                "condition_order": list(pair["condition_order"]),
                "attempt_index": 1,
                "retry_count": 0,
                "condition_started": True,
                "condition_terminal": True,
                "incomplete": False,
                "excluded": False,
                "initial_scene_state_sha256": pair["initial_scene_state_sha256"],
                "initial_measured_joint_state_sha256": pair[
                    "initial_measured_joint_state_sha256"
                ],
                "initial_commanded_drive_target_sha256": pair[
                    "initial_commanded_drive_target_sha256"
                ],
                "reset_proof_sha256": result._reset_proof(pair),
                "candidate_count": 4,
                "ordered_candidate_sha256": ordered,
                "candidate_legal": legal,
                "candidate_registry_sha256": candidate_registry,
                "continuation_contract": result.CONTINUATION_CONTRACT,
                "continuation_policy_sha256": continuation_policy,
                "continuation_rerank_after_root": False,
                "candidate_replacement_count": 0,
                "continuation_proof_sha256": "0" * 64,
                "selector": core["deployment"][
                    "baseline_selector" if condition_id == "baseline" else "etsf_selector"
                ],
                "selected_candidate_ordinal": 1 if condition_id == "baseline" else 2,
                "success": success[condition_id],
                "success_source": result.SUCCESS_SOURCE,
                "predicted_success_used_as_outcome": False,
                "execution_artifacts": execution_artifacts(
                    root / "execution_artifacts"
                    / f"condition_{2 * ordinal + position:03d}",
                    pair=pair,
                    position=position,
                    ordered=ordered,
                    legal=legal,
                    selected=(1 if condition_id == "baseline" else 2),
                    success=success[condition_id],
                    runtime_authority_sha256=runtime_authority_sha,
                    runtime_contract_sha256="9" * 64,
                ),
            }
            statement["continuation_proof_sha256"] = result._continuation_proof(statement)
            receipt = executor_receipt(
                executor, result.CONDITION_FORMAT, result.CONDITION_STATUS, statement
            )
            receipt_path = write_json(
                root / "conditions" / f"condition_{2 * ordinal + position:03d}.json",
                receipt,
            )
            terminal_record = {
                "global_condition_ordinal": 2 * ordinal + position,
                "pair_ordinal": ordinal,
                "condition_position": position,
                "condition_id": condition_id,
                **record(receipt_path, receipt["receipt_sha256"]),
            }
            pair_record = dict(terminal_record)
            condition_records.append(terminal_record)
            condition_statements.append(statement)
            pair_condition_records.append(pair_record)
            pair_condition_statements.append(statement)
            condition_terminal_event_sha256.append(append_ledger_event(
                "condition_terminal", pair=pair, position=position,
                artifact_receipt_sha256=receipt["receipt_sha256"],
            ))
        by_id = {row["condition_id"]: row for row in pair_condition_statements}
        pair_statement = {
            **common,
            "ordinal": ordinal,
            "pair_id": pair["pair_id"],
            "target_manifest_global_ordinal": pair["target_manifest_global_ordinal"],
            "requested_seed": pair["requested_seed"],
            "resolved_seed": pair["resolved_seed"],
            "condition_order": list(pair["condition_order"]),
            "condition_receipts": pair_condition_records,
            "reset_proof_sha256": pair_condition_statements[0]["reset_proof_sha256"],
            "candidate_registry_sha256": candidate_registry,
            "continuation_proof_sha256": pair_condition_statements[0][
                "continuation_proof_sha256"
            ],
            "ledger_condition_terminal_event_sha256": (
                condition_terminal_event_sha256
            ),
            "condition_attempt_count": 2,
            "complete_condition_count": 2,
            "retry_count": 0,
            "incomplete": False,
            "excluded": False,
            "baseline_success": by_id["baseline"]["success"],
            "etsf_success": by_id["etsf"]["success"],
            "success_source": result.SUCCESS_SOURCE,
        }
        pair_receipt = executor_receipt(
            executor, result.PAIR_FORMAT, result.PAIR_STATUS, pair_statement
        )
        pair_path = write_json(
            root / "pairs" / f"pair_{ordinal:03d}.json", pair_receipt
        )
        pair_records.append({
            "ordinal": ordinal,
            "pair_id": pair["pair_id"],
            **record(pair_path, pair_receipt["receipt_sha256"]),
        })
        append_ledger_event(
            "pair_terminal", pair=pair, position=None,
            artifact_receipt_sha256=pair_receipt["receipt_sha256"],
        )
    final_event_sha256 = append_ledger_event(
        "execution_terminal", pair=None, position=None,
        artifact_receipt_sha256=None,
    )
    assert len(ledger_event_records) == result.LEDGER_EVENT_COUNT
    terminal_statement = {
        "protocol_core": record(core_path, core["protocol_core_sha256"]),
        "ed25519_decision": record(decision_path, decision["decision_sha256"]),
        "execution_bundle": record(bundle_path, bundle["bundle_sha256"]),
        "executor_key_id": "executor-synthetic-1",
        "executor_public_key_hex": executor_public.hex(),
        "executor_public_key_sha256": hashlib.sha256(executor_public).hexdigest(),
        "executor_identity_sha256": hashlib.sha256(executor_public).hexdigest(),
        "execution_nonce_hex": execution_nonce,
        "pair_identity_set_sha256": pair_identity_set_sha,
        "deployment_binding_sha256": core["deployment"]["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": core["deployment"][
            "policy_runtime_action_binding_sha256"
        ],
        "preexecution_dependency_rehash_sha256": dependency_rehash,
        "full_dependency_rehash_after_claim_before_first_condition_started": True,
        "bootstrap_draws": record(bootstrap_path, bootstrap_logical),
        "bootstrap_format": result.BOOTSTRAP_FORMAT,
        "bootstrap_shape": list(result.BOOTSTRAP_SHAPE),
        "bootstrap_seed": result.BOOTSTRAP_SEED,
        "bootstrap_generator": result.BOOTSTRAP_GENERATOR,
        "bootstrap_frozen_before_first_condition_started": True,
        "execution_claim": claim_record,
        "ledger_contract": {
            "format": result.LEDGER_FORMAT,
            "terminal_state": result.LEDGER_FINAL_STATE,
            "ledger_id_sha256": ledger_id,
            "claim_receipt_sha256": claim_receipt["receipt_sha256"],
            "final_event_sha256": final_event_sha256,
            "event_count": result.LEDGER_EVENT_COUNT,
            "claim_count": 1,
            "claim_release_count": 0,
            "execution_attempt_count": 1,
            "condition_attempt_count": 800,
            "retry_count": 0,
            "selective_rerun_count": 0,
            "pair_exclusion_count": 0,
            "condition_exclusion_count": 0,
            "incomplete_pair_count": 0,
            "incomplete_condition_count": 0,
            "complete_pair_count": 400,
            "complete_condition_count": 800,
            "claim_before_outcome_read": True,
            "one_shot_consumed": True,
        },
        "ledger_events": ledger_event_records,
        "pair_receipts": pair_records,
        "condition_receipts": condition_records,
        "execution_complete": True,
        "subset_statistics_authorized": False,
        "performance_claim_authorized_by_executor": False,
    }
    terminal = executor_receipt(
        executor, result.TERMINAL_FORMAT, result.TERMINAL_STATUS, terminal_statement
    )
    terminal_path = write_json(root / "execution_terminal.json", terminal)
    result_key_path = write_bytes(root / "result_signer.raw", b"\x33" * 32, mode=0o400)
    kwargs = {
        "core_path": core_path,
        "core_file_sha256": file_sha(core_path),
        "decision_path": decision_path,
        "decision_file_sha256": file_sha(decision_path),
        "bundle_path": bundle_path,
        "bundle_file_sha256": file_sha(bundle_path),
        "execution_terminal_path": terminal_path,
        "execution_terminal_file_sha256": file_sha(terminal_path),
        "bootstrap_draws_path": bootstrap_path,
        "bootstrap_draws_file_sha256": file_sha(bootstrap_path),
        "expected_paired_implementation_file_sha256": file_sha(Path(paired.__file__)),
        "expected_evaluator_implementation_file_sha256": file_sha(Path(result.__file__)),
        "result_signing_private_key_path": result_key_path,
        "result_signing_private_key_file_sha256": file_sha(result_key_path),
        "result_signer_key_id": "result-signer-synthetic-1",
        "expected_result_signer_public_key_sha256": hashlib.sha256(
            result_public
        ).hexdigest(),
    }
    return {
        "kwargs": kwargs,
        "core": core,
        "decision": decision,
        "bundle": bundle,
        "pairs": pairs,
        "terminal": terminal,
        "terminal_statement": terminal_statement,
        "condition_statements": condition_statements,
        "condition_records": condition_records,
        "pair_records": pair_records,
        "ledger_event_records": ledger_event_records,
        "claim_record": claim_record,
        "executor": executor,
        "executor_public": executor_public,
        "result_public_sha": hashlib.sha256(result_public).hexdigest(),
        "root": root,
    }


def test_complete_synthetic_closure_computes_and_signs_exact_result(
    tmp_path: Path,
) -> None:
    fixture = synthetic_closure(tmp_path)
    receipt = result.evaluate_results(**fixture["kwargs"])
    result.validate_result_receipt(
        receipt, expected_public_key_sha256=fixture["result_public_sha"]
    )
    statement = receipt["statement"]
    assert statement["coverage"] == {
        "required_pair_count": 400,
        "complete_pair_count": 400,
        "required_condition_count": 800,
        "complete_condition_count": 800,
        "missing_pair_count": 0,
        "missing_condition_count": 0,
        "retry_count": 0,
        "incomplete_count": 0,
        "exclusion_count": 0,
        "subset_statistics_computed": False,
    }
    statistics = statement["statistics"]
    assert statistics["baseline_success_count"] == 200
    assert statistics["etsf_success_count"] == 240
    assert statistics["success_rate_delta_etsf_minus_baseline"]["value"] == 0.1
    assert statistics["mcnemar"]["n01"] == 80
    assert statistics["mcnemar"]["n10"] == 120
    assert statement["capability_receipt"]["simulator_calls"] == 0
    output = fixture["root"] / "signed_result.json"
    result.write_json_new(output, receipt)
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        result.write_json_new(output, receipt)


def test_bootstrap_artifact_is_seed_bound_and_tamper_fails() -> None:
    payload = result.bootstrap_draw_bytes()
    draws = result._validate_bootstrap(payload)
    assert draws.shape == result.BOOTSTRAP_SHAPE
    tampered = bytearray(payload)
    tampered[0] ^= 1
    with pytest.raises(result.Evaluation400ResultError, match="frozen seed"):
        result._validate_bootstrap(bytes(tampered))


def test_statistics_zero_discordant_mcnemar_is_one() -> None:
    draws = np.frombuffer(result.bootstrap_draw_bytes(), dtype="<u2").reshape(
        result.BOOTSTRAP_SHAPE
    )
    statistics = result.compute_statistics([False] * 400, [False] * 400, draws)
    assert statistics["mcnemar"]["discordant_count"] == 0
    assert statistics["mcnemar"]["exact_two_sided_p"] == {
        "numerator": 1, "denominator": 1, "value": 1.0
    }
    assert statistics["paired_bootstrap"]["lower"]["value"] == 0.0
    assert statistics["paired_bootstrap"]["upper"]["value"] == 0.0


def test_exact_bool_retry_incomplete_and_candidate_contract_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = synthetic_closure(tmp_path)
    pair = fixture["pairs"][0]
    original = fixture["condition_statements"][0]
    kwargs = {
        "pair": pair,
        "position": 0,
        "core": fixture["core"],
        "decision": fixture["decision"],
        "bundle": fixture["bundle"],
        "execution_nonce_hex": fixture["terminal_statement"]["execution_nonce_hex"],
        "dependency_rehash_sha256": fixture["terminal_statement"][
            "preexecution_dependency_rehash_sha256"
        ],
    }
    for field, value, match in (
        ("success", 1, "exact boolean"),
        ("retry_count", 1, "exact integer 0"),
        ("incomplete", True, "exact boolean False"),
        ("excluded", True, "exact boolean False"),
    ):
        changed = dict(original)
        changed[field] = value
        with pytest.raises(result.Evaluation400ResultError, match=match):
            result._validate_condition_statement(changed, **kwargs)


def test_terminal_requires_all_400_800_and_no_ledger_retry(tmp_path: Path) -> None:
    fixture = synthetic_closure(tmp_path)
    terminal = fixture["terminal_statement"]
    core_record = terminal["protocol_core"]
    decision_record = terminal["ed25519_decision"]
    bundle_record = terminal["execution_bundle"]
    bootstrap_path = fixture["kwargs"]["bootstrap_draws_path"]
    bootstrap_sha = fixture["kwargs"]["bootstrap_draws_file_sha256"]
    missing = dict(terminal)
    missing["pair_receipts"] = terminal["pair_receipts"][:-1]
    with pytest.raises(result.Evaluation400ResultError, match="exact 400/800/2001"):
        result._validate_terminal_statement(
            missing, core_record=core_record, decision_record=decision_record,
            bundle_record=bundle_record, core=fixture["core"],
            bootstrap_path=bootstrap_path, bootstrap_file_sha256=bootstrap_sha,
        )
    retried = dict(terminal)
    retried["ledger_contract"] = {
        **terminal["ledger_contract"], "retry_count": 1
    }
    with pytest.raises(result.Evaluation400ResultError, match="exact integer 0"):
        result._validate_terminal_statement(
            retried, core_record=core_record, decision_record=decision_record,
            bundle_record=bundle_record, core=fixture["core"],
            bootstrap_path=bootstrap_path, bootstrap_file_sha256=bootstrap_sha,
        )


def test_executor_signature_and_external_paired_sha_are_mandatory(tmp_path: Path) -> None:
    fixture = synthetic_closure(tmp_path)
    terminal = dict(fixture["terminal"])
    terminal["executor_signature_ed25519_hex"] = "0" * 128
    unsigned = dict(terminal)
    unsigned.pop("receipt_sha256")
    terminal["receipt_sha256"] = result.canonical_sha256(unsigned)
    public_key = fixture["executor"].public_key()
    with pytest.raises(result.Evaluation400ResultError, match="signature failed"):
        result._verify_executor_receipt(
            terminal, expected_format=result.TERMINAL_FORMAT,
            expected_status=result.TERMINAL_STATUS, public_key=public_key,
            role="terminal",
        )
    bad_kwargs = dict(fixture["kwargs"])
    bad_kwargs["expected_paired_implementation_file_sha256"] = "0" * 64
    with pytest.raises(result.Evaluation400ResultError, match="file SHA mismatch"):
        result.evaluate_results(**bad_kwargs)


def test_pair_conditions_must_share_reset_candidates_and_continuation(
    tmp_path: Path,
) -> None:
    fixture = synthetic_closure(tmp_path)
    pair = fixture["pairs"][0]
    condition_records = fixture["condition_records"][:2]
    first = fixture["condition_statements"][0]
    second = dict(fixture["condition_statements"][1])
    second["candidate_registry_sha256"] = "9" * 64
    pair_statement = {
        **{
            field: first[field]
            for field in (
                "protocol_core_sha256", "decision_sha256", "bundle_sha256",
                "execution_nonce_hex", "pair_identity_set_sha256",
                "deployment_binding_sha256", "policy_runtime_action_binding_sha256",
                "preexecution_dependency_rehash_sha256",
            )
        },
        "ordinal": pair["ordinal"],
        "pair_id": pair["pair_id"],
        "target_manifest_global_ordinal": pair["target_manifest_global_ordinal"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "condition_order": pair["condition_order"],
        "condition_receipts": condition_records,
        "reset_proof_sha256": first["reset_proof_sha256"],
        "candidate_registry_sha256": first["candidate_registry_sha256"],
        "continuation_proof_sha256": first["continuation_proof_sha256"],
        "ledger_condition_terminal_event_sha256": ["8" * 64, "9" * 64],
        "condition_attempt_count": 2,
        "complete_condition_count": 2,
        "retry_count": 0,
        "incomplete": False,
        "excluded": False,
        "baseline_success": False,
        "etsf_success": False,
        "success_source": result.SUCCESS_SOURCE,
    }
    with pytest.raises(result.Evaluation400ResultError, match="shared root"):
        result._validate_pair_statement(
            pair_statement, pair=pair,
            conditions=[(condition_records[0], first), (condition_records[1], second)],
            core=fixture["core"], decision=fixture["decision"],
            bundle=fixture["bundle"],
            execution_nonce_hex=fixture["terminal_statement"]["execution_nonce_hex"],
            dependency_rehash_sha256=fixture["terminal_statement"][
                "preexecution_dependency_rehash_sha256"
            ],
        )
