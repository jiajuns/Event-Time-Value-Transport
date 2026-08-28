from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import calibrate_openvla_etsf_v8_success_inner_cv as success_module  # noqa: E402
import evaluate_openvla_etsf_v8_factual_events as factual_module  # noqa: E402
import evaluate_openvla_etsf_v8_oof_bridge as bridge_module  # noqa: E402
import evaluate_openvla_etsf_v8_structured_heads_arrays as arrays_module  # noqa: E402
import freeze_openvla_etsf_composite_structured_prediction_activation as freezer  # noqa: E402
import freeze_openvla_etsf_duration_hierarchy_activation as duration_freezer  # noqa: E402
from openvla_etsf_composite_structured_prediction_router import (  # noqa: E402
    load_composite_activation,
    route_structured_predictions,
    validate_composite_activation,
)
from openvla_etsf_duration_hierarchy import canonical_sha256  # noqa: E402
from openvla_etsf_duration_hierarchy_adapter import (  # noqa: E402
    fit_final_duration_hierarchy,
    sha256_path,
    validate_duration_activation,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_TRAINING_CHECKPOINT_FORMAT,
)


def _write_signed(path: Path, value: dict, key: str) -> dict:
    value[key] = canonical_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    return value


def _duration_activation(root: Path, materialization_sha: str, base_sha: str) -> Path:
    groups = [f"move_can_pot|piper_piper_0.6|{index:03d}" for index in range(250)]
    duration = np.asarray([10.0 + index % 9 for index in range(250)])
    hierarchy = fit_final_duration_hierarchy(
        duration=duration,
        duration_observed=np.ones(250, dtype=bool),
        dense_mask=np.ones(250, dtype=bool),
        current_event_id=np.zeros(250, dtype=np.int64),
        body_id=np.zeros(250, dtype=np.int64),
        logical_group=np.asarray(groups),
        development_groups=groups,
        materialization_groups_sha256="d" * 64,
    )
    metadata = [
        {
            "logical_group_key": group,
            "body": "piper_piper_0.6",
            "body_id": 0,
            "policy": "openvla",
            "policy_id": 0,
        }
        for group in groups
    ]
    registry = duration_freezer._build_empirical_registry(
        metadata,
        development_groups=groups,
        development_groups_sha256="d" * 64,
    )
    implementation_names = {
        "openvla_etsf_duration_hierarchy.py",
        "openvla_etsf_duration_hierarchy_adapter.py",
        "freeze_openvla_etsf_duration_hierarchy_activation.py",
        "evaluate_openvla_etsf_duration_hierarchy_oof.py",
        "openvla_etsf_v8_structured_adapters.py",
        "train_openvla_etsf_v8_structured_adapters.py",
    }
    activation = {
        "format": "etsf_duration_v2_prediction_activation_v1",
        "status": "activated_duration_prediction_only_development",
        "evidence_scope": "adaptive_development_only",
        "permissions": {
            "duration_prediction_adapter": True,
            "actor_control": False,
            "policy_modification": False,
            "reward_or_value": False,
            "candidate_ranking": False,
            "selector": False,
        },
        "interface_actor_policy_agnostic": True,
        "empirical_registry_contract": registry,
        "empirical_registry_contract_sha256": registry["registry_sha256"],
        "empirical_evidence_scope": {
            "policy": "openvla",
            "policy_id": 0,
            "body": "piper_piper_0.6",
            "body_id": 0,
            "one_cell_only": True,
            "cross_body_validated": False,
            "cross_policy_validated": False,
        },
        "transfer_claim_authorized": False,
        "duration_residual_multiplier": 0.375,
        "formula": "baseline+0.375*(frozen_duration_log_mean-baseline)",
        "final_hierarchy_contract": hierarchy,
        "final_hierarchy_contract_sha256": hierarchy["contract_sha256"],
        "development_coverage": {
            "logical_groups": 250,
            "five_holdouts_cover_each_group_exactly_once": True,
            "development_groups_sha256": "d" * 64,
            "materialized_rows": 250,
            "dense_observed_rows": 250,
        },
        "evidence": {
            "factual_checkpoint_sha256": base_sha,
            "factual_state_sha256": "e" * 64,
            "event_spec_sha256": "f" * 64,
            "materialization_sha256": materialization_sha,
            "materialization_file_sha256": "1" * 64,
            "r5_result_sha256": "2" * 64,
            "r5_result_file_sha256": "3" * 64,
            "r5_rows_file_sha256": "4" * 64,
        },
        "source_paths": {
            "materialization_manifest": str((root / "materialization.json").resolve()),
            "r5_result_json": str((root / "duration_result.json").resolve()),
            "r5_rows_npz": str((root / "duration_rows.npz").resolve()),
        },
        "implementation_files": {
            name: sha256_path(SCRIPTS / name) for name in implementation_names
        },
        "authentication_trace": [],
        "source_hdf5_read": False,
        "model_training_performed": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "fresh50_confirmation_authorized": False,
        "selector_authorized": False,
        "prospective_claim_allowed": False,
    }
    activation["activation_sha256"] = canonical_sha256(activation)
    validate_duration_activation(activation)
    path = root / "duration_activation.json"
    path.write_text(json.dumps(activation, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture()
def evidence(tmp_path: Path) -> dict[str, Path]:
    materialization_sha = "a" * 64
    base_sha = "b" * 64
    duration_path = _duration_activation(tmp_path, materialization_sha, base_sha)

    domain = {
        "support_rows": 250,
        "model": {"accuracy": 0.7, "nll": 0.8},
        "uncertainty": {"aurc": 0.2},
    }
    factual = {
        "format": factual_module.FORMAT,
        "status": "complete_adaptive_development_only",
        "evidence_scope": "D250_adaptive_development_only_not_prospective",
        "rows": 500,
        "logical_groups": 250,
        "immediate_event": domain,
        "observed_destination_event": copy.deepcopy(domain),
        "frozen_factual_state": {
            "bit_exact_is_accuracy_evidence": False,
            "accuracy_measured_from_labels_and_logits": True,
        },
        "uncertainty_scope": {
            "evaluated_quantity": "single_factual_member_composite_aleatoric_score",
            "epistemic_uncertainty_available": False,
            "complete_predictive_uncertainty_claimed": False,
        },
        "authorization": {
            "fresh50_confirmation_authorized": False,
            "selector_authorized": False,
            "deployment_authorized": False,
            "policy_success_claim_authorized": False,
        },
        "fresh_confirmation_data_or_labels_read": False,
        "source_materialization": {"materialization_sha256": materialization_sha},
    }
    factual_path = tmp_path / "factual_events.json"
    _write_signed(factual_path, factual, "result_sha256")

    checkpoint_rows = []
    for owner in range(5):
        checkpoint = {
            "format": V8_TRAINING_CHECKPOINT_FORMAT,
            "optimizer": {"name": "AdamW"},
            "all_steps_factual_inputs_bit_exact": True,
            "strict_oof_base_exclusion_eligible": True,
            "fresh_confirmation_data_or_labels_read": False,
            "authorization_guard_changed": False,
            "provenance": {
                "outer_fold_id": owner,
                "base_checkpoint_sha256": base_sha,
            },
            "adapter_state_sha256": f"{owner + 1:x}" * 64,
        }
        path = tmp_path / f"adamw_fold_{owner}.pt"
        torch.save(checkpoint, path)
        checkpoint_rows.append(
            {
                "owner_fold_id": owner,
                "checkpoint_role": "outer_training_only_adapter_checkpoint",
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": sha256_path(path),
            }
        )
    adaptive = {
        "source_sha256": {"base_checkpoint": base_sha},
        "status": "adaptive_development_only",
    }
    adaptive["contract_sha256"] = canonical_sha256(adaptive)
    bundle = {
        "format": bridge_module.BRIDGE_FORMAT,
        "status": "authenticated_inputs_rehashed",
        "source_partition": "development_only",
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "adaptive_development_contract_sha256": adaptive["contract_sha256"],
        "folds": checkpoint_rows,
    }
    bundle["bridge_bundle_sha256"] = canonical_sha256(bundle)
    arrays_path = tmp_path / "structured_arrays.npz"
    np.savez_compressed(arrays_path, regress=np.asarray([0.2, 0.8]))
    contracts = {
        "format": bridge_module.OUTPUT_FORMAT,
        "adaptive_contract": adaptive,
        "bridge_bundle": bundle,
        "bridge_provenance": {"materialization_sha256": materialization_sha},
        "arrays": str(arrays_path.resolve()),
        "arrays_sha256": sha256_path(arrays_path),
    }
    contracts_path = tmp_path / "adamw_contracts.json"
    _write_signed(contracts_path, contracts, "contracts_sha256")
    regress = {
        "status": "passed",
        "passed": True,
        "support_gate": True,
        "weight_provenance_error": None,
        "baseline_provenance_error": None,
        "brier_vs_crossfit_prevalence": {"strict_skill": True},
        "nll_vs_crossfit_prevalence": {"strict_skill": True},
        "ap_minus_prevalence": {"strict_skill": True},
        "ece_gate": True,
    }
    r4 = {
        "format": arrays_module.FORMAT,
        "adaptive_development_contract_sha256": adaptive["contract_sha256"],
        "evidence_design": "adaptive_current_d250_after_collection_started",
        "prospective_claim_allowed": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "fresh50_confirmation_authorized": False,
        "v7_implementation_changed": False,
        "action_selector_authorized": False,
        "domains": {"regress": regress},
        "domain_pass": {"regress": True},
    }
    r4_path = tmp_path / "adamw_result.json"
    _write_signed(r4_path, r4, "result_sha256")

    fold_contracts = []
    for owner in range(5):
        contract = {
            "owner_fold_id": owner,
            "outer_holdout_labels_used_for_alpha_selection": False,
            "fresh50_inputs_or_labels_used": False,
        }
        contract["calibration_contract_sha256"] = canonical_sha256(contract)
        fold_contracts.append(contract)
    success = {
        "format": success_module.FORMAT,
        "status": "complete_adaptive_development_only",
        "materialization_sha256": materialization_sha,
        "fold_calibration_contracts": fold_contracts,
        "outer_holdout_evaluation": {
            "pooled_oof": {
                "calibrated_probability_adequacy": {
                    "strict_probability_adequacy": False
                }
            }
        },
        "action_ranking_preserved_within_each_group": True,
        "task_success_cannot_change_from_uncalibrated_argmax": True,
        "outer_holdout_labels_used_for_alpha_selection": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "authorization": {
            "selector_authorized": False,
            "deployment_authorized": False,
            "fresh50_confirmation_authorized": False,
        },
    }
    success_path = tmp_path / "success_result.json"
    _write_signed(success_path, success, "result_sha256")
    return {
        "factual": factual_path,
        "r4": r4_path,
        "contracts": contracts_path,
        "success": success_path,
        "duration": duration_path,
    }


def _freeze(evidence: dict[str, Path]) -> dict:
    return freezer.freeze_composite_activation(
        factual_event_result=evidence["factual"],
        r4_adamw_result=evidence["r4"],
        r4_adamw_contracts=evidence["contracts"],
        r5_success_result=evidence["success"],
        r5_duration_activation=evidence["duration"],
    )


def test_freeze_capability_boundary_and_immutable_load(
    evidence: dict[str, Path], tmp_path: Path
) -> None:
    activation = _freeze(evidence)
    assert set(activation["active"]) == {
        "next_event",
        "destination_event",
        "aleatoric_uncertainty",
        "regress",
        "duration_v2",
    }
    assert activation["inactive_or_fallback"]["success"]["status"] == "inactive"
    assert activation["action_selector"]["authority"] == (
        "v7_fixed_parameter_free_selector"
    )
    assert activation["action_selector"]["v8_replacement_authorized"] is False
    assert len(activation["evidence"]["r4_adamw_regress"]["checkpoints"]) == 5
    output = tmp_path / "composite_activation.json"
    freezer.write_composite_activation(output, activation)
    assert output.stat().st_mode & 0o222 == 0
    assert load_composite_activation(output)["activation_sha256"] == activation[
        "activation_sha256"
    ]


def test_router_detaches_and_never_exposes_failed_success_or_ranking(
    evidence: dict[str, Path]
) -> None:
    activation = _freeze(evidence)
    frozen = np.asarray([2.0, 3.0])
    routed = route_structured_predictions(
        activation,
        body_registry_contract=activation["duration_v2_activation"][
            "empirical_registry_contract"
        ],
        current_event_id=np.asarray([0, 0]),
        body_id=np.asarray([0, 999]),
        next_event_logits=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        destination_event_logits=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        aleatoric_uncertainty=np.asarray([0.1, 0.5]),
        regress_probability=np.asarray([0.2, 0.8]),
        frozen_duration_log_mean=frozen,
    )
    assert "success" not in routed["active_predictions"]
    assert "regress_probability" not in routed["v7_selector_inputs"]
    np.testing.assert_array_equal(
        routed["v7_selector_inputs"]["duration_selected_log_mean"], frozen
    )
    assert routed["ranking_score_produced"] is False
    assert routed["openvla_gradient_path"] is False
    assert routed["active_predictions"]["duration_v2"][
        "out_of_empirical_body_scope"
    ].tolist() == [False, True]
    with pytest.raises(RuntimeError, match="detached from OpenVLA gradients"):
        route_structured_predictions(
            activation,
            body_registry_contract=activation["duration_v2_activation"][
                "empirical_registry_contract"
            ],
            current_event_id=np.asarray([0]),
            body_id=np.asarray([0]),
            next_event_logits=torch.ones((1, 2), requires_grad=True),
            destination_event_logits=np.ones((1, 2)),
            aleatoric_uncertainty=np.asarray([0.1]),
            regress_probability=np.asarray([0.2]),
            frozen_duration_log_mean=np.asarray([2.0]),
        )


def test_failed_success_or_regress_cannot_be_activated(
    evidence: dict[str, Path], tmp_path: Path
) -> None:
    success = json.loads(evidence["success"].read_text(encoding="utf-8"))
    success["outer_holdout_evaluation"]["pooled_oof"][
        "calibrated_probability_adequacy"
    ]["strict_probability_adequacy"] = True
    success.pop("result_sha256")
    bad_success = tmp_path / "bad_success.json"
    _write_signed(bad_success, success, "result_sha256")
    bad = dict(evidence)
    bad["success"] = bad_success
    with pytest.raises(RuntimeError, match="success inadequacy evidence"):
        _freeze(bad)

    r4 = json.loads(evidence["r4"].read_text(encoding="utf-8"))
    r4["domains"]["regress"]["ece_gate"] = False
    r4.pop("result_sha256")
    bad_r4 = tmp_path / "bad_r4.json"
    _write_signed(bad_r4, r4, "result_sha256")
    bad = dict(evidence)
    bad["r4"] = bad_r4
    with pytest.raises(RuntimeError, match="did not pass every strict"):
        _freeze(bad)


def test_manifest_tamper_and_forbidden_paths_fail_closed(
    evidence: dict[str, Path]
) -> None:
    activation = _freeze(evidence)
    activation["inactive_or_fallback"]["success"]["status"] = "active"
    activation["activation_sha256"] = canonical_sha256(
        {key: value for key, value in activation.items() if key != "activation_sha256"}
    )
    with pytest.raises(RuntimeError, match="capability boundary changed"):
        validate_composite_activation(activation)
    with pytest.raises(RuntimeError, match="Fresh/confirmation"):
        freezer.freeze_composite_activation(
            factual_event_result=Path("/srv/Fresh50/factual.json"),
            r4_adamw_result=evidence["r4"],
            r4_adamw_contracts=evidence["contracts"],
            r5_success_result=evidence["success"],
            r5_duration_activation=evidence["duration"],
        )
