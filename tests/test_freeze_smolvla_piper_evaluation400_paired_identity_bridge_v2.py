from __future__ import annotations

import copy
import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge  # noqa: E402


def signed(value: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(value)
    result[field] = bridge.canonical_sha256(result)
    return result


def source_rank_contract(index: int) -> dict[str, object]:
    return signed(
        {
            "source_checkpoint_file_sha256": f"{index + 1:064x}",
            "success_temperature": 1.0,
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
        },
        "contract_sha256",
    )


def source_rank_member_authority() -> dict[str, object]:
    authority = {
        "source_rank_numeric_contract": bridge.SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": index,
                "source_checkpoint_file_sha256": contract[
                    "source_checkpoint_file_sha256"
                ],
                "source_rank_score_contract_sha256": contract[
                    "contract_sha256"
                ],
                "success_temperature": contract["success_temperature"],
            }
            for index in range(5)
            for contract in [source_rank_contract(index)]
        ],
    }
    return {
        "source_rank_member_authority": authority,
        "source_rank_member_authority_sha256": bridge.canonical_sha256(authority),
    }


def root_selection_oof_evidence() -> dict[str, object]:
    fold_parameters = [
        signed(
            {
                "outer_fold": fold,
                "training_logical_group_count": 152,
                "heldout_logical_group_count": 38,
                "training_logical_group_ids_sha256": f"{fold + 1:064x}",
                "heldout_logical_group_ids_sha256": f"{fold + 6:064x}",
                "post_event_temperature": 1.0,
                "next_event_temperature": 1.0,
                "success_temperature": 1.0,
                "duration_scale_multiplier": 1.0,
                "object_scale_multiplier": 1.0,
                "object_uncertainty_robust_scale_m": 1.0,
                "parameters_complete": True,
                "heldout_labels_used_for_parameters_or_training_quality": False,
            },
            "fold_parameter_sha256",
        )
        for fold in range(5)
    ]
    contract = signed(
        {
            "format": "etsf_smolvla_piper_formal190_complete_root_outer_nesting_v1",
            "status": "complete_outer_heldout_isolation",
            "outer_crossfit_folds": 5,
            "fold_assignment": "lexicographic_logical_group_index_modulo_five",
            "fold_parameters": fold_parameters,
            "outer_heldout_labels_used_for_any_parameter_or_selection": False,
            "same_outer_training_parameters_used_for_training_and_heldout_inference": True,
            "next_duration_observation_masks_used_only_for_parameter_fitting_and_quality": True,
            "object_robust_scale_fit_on_outer_training_groups_only": True,
            "complete_root_pipeline_outer_nesting": True,
            "upstream_predictions_already_group_crossfit": True,
            "evaluation400_outcomes_read": False,
        },
        "root_outer_nesting_contract_sha256",
    )
    decisions = []
    for index in range(190):
        helpful = index < 40
        changed = index < 100
        decisions.append({
            "logical_group_id": f"formal-group-{index:03d}",
            "selection_available": True,
            "changed_from_baseline": changed,
            "selected_candidate_index": 1 if changed else 0,
            "baseline_final_success": 0,
            "selected_final_success": 1 if helpful else 0,
            "paired_gain": 1 if helpful else 0,
            "outer_fold": index % 5,
        })
    draw_values = bridge.np.random.default_rng(20260828).integers(
        0, 190, size=(5000, 190), dtype=bridge.np.uint16
    )
    draw_bytes = bridge.np.ascontiguousarray(
        draw_values.astype("<u2", copy=False)
    ).tobytes(order="C")
    gain_values = bridge.np.asarray(
        [row["paired_gain"] for row in decisions], dtype=bridge.np.float64
    )
    changed_values = bridge.np.asarray(
        [row["changed_from_baseline"] for row in decisions], dtype=bool
    )
    harmful_values = gain_values < 0
    sampled_gain = gain_values[draw_values].mean(axis=1)
    sampled_changed = changed_values[draw_values].sum(axis=1)
    sampled_harmful = harmful_values[draw_values].sum(axis=1)
    draws = signed(
        {
            "algorithm": "numpy_pcg64_fixed_seed_logical_group_indices_v1",
            "seed": 20260828,
            "dtype": "little_endian_uint16",
            "shape": [5000, 190],
            "draws_sha256": bridge.hashlib.sha256(draw_bytes).hexdigest(),
        },
        "descriptor_sha256",
    )
    outer_folds = []
    for fold in range(5):
        heldout = [row for row in decisions if row["outer_fold"] == fold]
        outer_folds.append({
            "outer_fold": fold,
            "training_logical_group_count": 152,
            "heldout_logical_group_count": 38,
            "training_logical_group_ids_sha256": f"{fold + 1:064x}",
            "heldout_logical_group_ids_sha256": f"{fold + 6:064x}",
            "training_global_abstain_threshold": {"enabled": True},
            "training_root_candidate_grid": [{} for _ in range(20)],
            "training_root_candidate_grid_sha256": bridge.canonical_sha256(
                [{} for _ in range(20)]
            ),
            "training_bootstrap_draws": signed(
                {
                    "algorithm": "numpy_pcg64_fixed_seed_logical_group_indices_v1",
                    "seed": 20260828,
                    "dtype": "little_endian_uint16",
                    "shape": [5000, 152],
                    "draws_sha256": "b" * 64,
                },
                "descriptor_sha256",
            ),
            "selected_training_candidate": {
                "minimum_group_relative_composite_rank_score_margin": 0.1,
                "maximum_structured_pair_uncertainty": 0.5,
                "maximum_global_candidate_uncertainty": 0.25,
            },
            "selection_available": True,
            "heldout_outcomes_used_for_training_selection": False,
            "heldout_decisions": heldout,
        })
    return signed(
        {
            "format": "etsf_smolvla_piper_formal190_root_selection_oof_evidence_v1",
            "status": "passed_selection_aware_root_gate",
            "passed_for_primary": True,
            "outer_crossfit_folds": 5,
            "outer_fold_assignment": "lexicographic_logical_group_index_modulo_five",
            "root_selection_nested_within_outer_training_groups": True,
            "global_abstain_threshold_nested_within_outer_training_groups": True,
            "upstream_predictions_already_group_crossfit": True,
            "complete_temperature_scale_and_root_double_nesting": True,
            "scope": "complete_outer_fold_temperature_scale_robust_normalization_quality_threshold_grid_then_once_heldout_evaluation",
            "root_outer_nesting_contract": contract,
            "root_outer_nesting_contract_sha256": contract[
                "root_outer_nesting_contract_sha256"
            ],
            "formal_logical_group_count": 190,
            "stitched_decision_count": 190,
            "unique_stitched_logical_group_count": 190,
            "every_formal_logical_group_scored_exactly_once": True,
            "outer_folds": outer_folds,
            "stitched_group_decisions": decisions,
            "changed_group_count": 100,
            "change_coverage": 100 / 190,
            "helpful_group_count": 40,
            "harmful_group_count": 0,
            "discordant_group_count": 40,
            "selected_success_count": 40,
            "selected_success_rate": 40 / 190,
            "baseline_success_count": 0,
            "baseline_success_rate": 0.0,
            "paired_gain": 40 / 190,
            "paired_gain_group_bootstrap_lcb95": float(
                bridge.np.quantile(sampled_gain, 0.05)
            ),
            "paired_gain_group_bootstrap_ucb95": float(
                bridge.np.quantile(sampled_gain, 0.95)
            ),
            "harmful_rate_among_executed_changes": 0.0,
            "harmful_rate_group_bootstrap_ucb95": float(
                bridge.np.quantile(
                    sampled_harmful / sampled_changed, 0.95
                )
            ),
            "fold_support": [
                {
                    "fold": fold,
                    "logical_groups": 38,
                    "changed_groups": 20,
                    "discordant_groups": 8,
                    "minimum_changed_groups": 10,
                    "minimum_discordant_groups": 4,
                    "support_passed": True,
                }
                for fold in range(5)
            ],
            "minimum_changed_groups": 50,
            "minimum_discordant_groups": 20,
            "maximum_harmful_rate_among_executed_changes": 0.10,
            "paired_gain_lcb_must_be_strictly_positive": True,
            "bootstrap_unit": "logical_group",
            "bootstrap_seed": 20260828,
            "bootstrap_samples": 5000,
            "shared_bootstrap_draws": draws,
            "evaluation400_outcomes_read": False,
        },
        "selection_aware_oof_evidence_sha256",
    )


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def semantic_receipt() -> dict[str, object]:
    return signed(
        {
            "format": "etsf_explicit_instruction_semantics_receipt_v1",
            "task": bridge.TASK,
            "language": "en",
            "instruction": bridge.INSTRUCTION,
            "semantic_frame": {
                "theme": "can",
                "relation": "inside",
                "reference": "pot",
            },
            "source": "protocol_constant_not_episode_info_list",
            "episode_info_list_used": False,
        },
        "receipt_sha256",
    )


def target_row(
    split: str, ordinal: int, global_ordinal: int
) -> dict[str, object]:
    semantic = semantic_receipt()
    row: dict[str, object] = {
        "task": bridge.TASK,
        "actor_id": bridge.ACTOR_ID,
        "target_body": bridge.TARGET_BODY,
        "global_ordinal": global_ordinal,
        "split": split,
        "ordinal": ordinal,
        "stage_role": (
            "direct_actor_only_operational"
            if split == "adaptation" and ordinal < 20
            else "adapter_development"
            if split == "adaptation"
            else "frozen_selection_validation"
            if split == "validation"
            else "sealed_paired_evaluation"
        ),
        "requested_seed": 200_000 + global_ordinal,
        "resolved_seed": 200_000 + global_ordinal,
        "instruction": bridge.INSTRUCTION,
        "instruction_sha256": __import__("hashlib").sha256(
            bridge.INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "instruction_semantics_receipt": semantic,
        "instruction_semantics_receipt_sha256": semantic["receipt_sha256"],
        "initial_scene_state_sha256": f"{global_ordinal + 1:064x}",
        "initial_measured_joint_state_sha256": f"{global_ordinal + 1000:064x}",
        "initial_commanded_drive_target_sha256": f"{global_ordinal + 2000:064x}",
    }
    row["pair_id"] = bridge.canonical_sha256(bridge._row_identity(row))
    return row


def target_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    counts = {
        "adaptation": bridge.ADAPTATION_GROUPS,
        "validation": bridge.VALIDATION_GROUPS,
        "evaluation": bridge.EVALUATION_GROUPS,
    }
    splits: dict[str, list[dict[str, object]]] = {}
    global_ordinal = 0
    for split, count in counts.items():
        splits[split] = [
            target_row(split, ordinal, global_ordinal + ordinal)
            for ordinal in range(count)
        ]
        global_ordinal += count
    requested = [
        row["requested_seed"] for split in counts for row in splits[split]
    ]
    resolved = [
        row["resolved_seed"] for split in counts for row in splits[split]
    ]
    identity_set_sha = bridge.canonical_sha256(
        {"requested": requested, "resolved": resolved}
    )
    attestation = signed(
        {
            "format": bridge.ATTESTATION_FORMAT,
            "status": bridge.ATTESTATION_STATUS,
            "target_role": "selected_requested_and_resolved_target_identities",
            "heldout_identity_set_sha256": "a" * 64,
            "target_identity_set_sha256": identity_set_sha,
            "intersection_count": 0,
            "sensitive_identities_included": False,
        },
        "attestation_sha256",
    )
    manifest = signed(
        {
            "format": bridge.TARGET_MANIFEST_FORMAT,
            "status": bridge.TARGET_MANIFEST_STATUS,
            "task": bridge.TASK,
            "actor_id": bridge.ACTOR_ID,
            "source_body": bridge.SOURCE_BODY,
            "target_body": bridge.TARGET_BODY,
            "purpose": "nonfresh_development_only_no_confirmation_claim",
            "label_access_contract": bridge.LABEL_ACCESS_CONTRACT,
            "instruction_contract": {
                "mode": "one_explicit_protocol_constant_for_every_reset_and_both_conditions",
                "instruction": bridge.INSTRUCTION,
                "same_instruction_for_both_conditions": True,
                "episode_info_list_used": False,
            },
            "splits": splits,
            "provenance": {
                "plan_file_sha256": "1" * 64,
                "plan_sha256": "2" * 64,
                "authorization_file_sha256": "3" * 64,
                "authorization_sha256": "4" * 64,
                "runtime_contract_sha256": "5" * 64,
                "reset_receipt_file_sha256": "6" * 64,
                "reset_receipt_sha256": "7" * 64,
                "resolver_implementation_sha256": "8" * 64,
                "reset_adapter_implementation_sha256": "9" * 64,
            },
            "d250_exclusion": {
                "identity_manifest_file_sha256": "b" * 64,
                "identity_sets_sha256": "c" * 64,
                "intersection_count": 0,
            },
            "heldout_exclusion_attestation": {
                "status": bridge.ATTESTATION_STATUS,
                "heldout_identity_set_sha256": "a" * 64,
                "target_identity_set_sha256": identity_set_sha,
                "intersection_count": 0,
                "sensitive_identities_included": False,
                "attestation_sha256": attestation["attestation_sha256"],
            },
            "capability_receipt": {
                "environment_reset_only": True,
                "environment_step_calls": 0,
                "policy_import_or_forward_calls": 0,
                "labels_or_outcomes_read": False,
                "policy_execution_authorized_by_manifest": False,
            },
        },
        "seed_manifest_sha256",
    )
    return manifest, attestation


def calibration_artifact() -> dict[str, object]:
    enabled = {
        "post_event": True,
        "next_event": True,
        "duration": True,
        "success": True,
        "recovery": True,
        "object_effect": True,
    }
    metric_names = (
        "post_event", "next_event", "success", "conditional_recovery",
        "duration_lognormal_mixture", "object_total_variance",
    )
    metrics = {
        name: {
            "crossfit_folds": 5,
            "crossfit_complete": True,
            "performance_gate_passed": True,
            "metric_weighting": "equal_logical_group",
            "uncertainty_gate": {"passed": True},
            **(
                {"deployment_temperature": 1.0}
                if name in {
                    "post_event", "next_event", "success",
                    "conditional_recovery",
                }
                else {"deployment_scale_multiplier": 1.0}
            ),
            **(
                {"deployment_object_error_robust_scale_m": 1.0}
                if name == "object_total_variance" else {}
            ),
        }
        for name in metric_names
    }
    implementation = Path(bridge.deployment_uncertainty_v1.__file__).resolve()
    member_authority = source_rank_member_authority()
    uncertainty_contract = signed(
        {
            "format": bridge.deployment_uncertainty_v1.FORMAT,
            "status": "frozen_full_formal190_refit_parameters_online_reproducible",
            "shared_implementation_path": str(implementation),
            "shared_implementation_file_sha256": bridge.file_sha256(implementation),
            "performance_gate_uncertainty_source": "five_fold_group_oof_predictions",
            "selector_uncertainty_source": "full_formal190_refit_deployment_parameters",
            "included_root_heads": list(
                bridge.deployment_uncertainty_v1.ROOT_INCLUDED_HEADS
            ),
            "excluded_root_heads": ["recovery"],
            "root_structured_uncertainty_head_count": 5,
            "root_recovery_uncertainty_policy": (
                bridge.deployment_uncertainty_v1.ROOT_RECOVERY_UNCERTAINTY_POLICY
            ),
            "post_event_temperature": 1.0,
            "next_event_temperature": 1.0,
            "success_temperature": 1.0,
            "conditional_recovery_temperature": 1.0,
            "duration_scale_multiplier": 1.0,
            "object_scale_multiplier": 1.0,
            "object_error_robust_scale_m": 1.0,
            "object_prediction_space": "physical_xyz_m",
            "duration_and_object_log_scale_multiplier_application": (
                "add_log_multiplier_exactly_once"
            ),
            "shared_function_input_scale_state": (
                "duration_and_object_deployment_multiplier_already_applied_exactly_once"
            ),
            "uncertainty_range": [0.0, 1.0],
            "formal_root_row_count": 760,
            "evaluation400_outcomes_read": False,
        },
        "deployment_uncertainty_contract_sha256",
    )
    selected = {
        "minimum_group_relative_composite_rank_score_margin": 0.1,
        "maximum_structured_pair_uncertainty": 0.5,
        "maximum_global_candidate_uncertainty": 0.25,
        "changed_group_count": 100,
        "discordant_group_count": 40,
        "paired_gain_group_bootstrap_lcb95": 0.1,
        "harmful_rate_group_bootstrap_ucb95": 0.05,
        "fold_support": [
            {"fold": fold, "support_passed": True} for fold in range(5)
        ],
    }
    oof_evidence = root_selection_oof_evidence()
    root_ranker = signed(
        {
            "format": "etsf_smolvla_piper_formal190_root_group_ranker_v1",
            "status": "enabled_source_composite_primary_ranker",
            "enabled_for_primary": True,
            "formal_logical_group_count": 190,
            "member_count": 5,
            "score_is_success_logit": False,
            "score_is_success_probability": False,
            "source_rank_numeric_contract": bridge.SOURCE_RANK_NUMERIC_CONTRACT,
            **member_authority,
            "factual_success_head_used_only_for_independent_six_head_calibration": True,
            "candidate_grid": [{} for _ in range(20)],
            "selected_candidate": selected,
            "full_formal190_development_metrics_are_in_sample": True,
            "full_formal190_deployment_refit_candidate_available": True,
            "selection_aware_oof_evidence": oof_evidence,
            "selection_aware_oof_evidence_sha256": oof_evidence[
                "selection_aware_oof_evidence_sha256"
            ],
            "primary_activation_requires_selection_aware_oof_evidence": True,
            "primary_gate_components": {
                "all_six_heads_support_performance_uncertainty_gate_passed": True,
                "full_formal190_deployment_candidate_available": True,
                "selection_aware_oof_evidence_passed": True,
            },
            "upstream_predictions_already_group_crossfit": True,
            "complete_temperature_scale_and_root_double_nesting": True,
            "groups": [{} for _ in range(190)],
            "maximum_harmful_rate_among_executed_changes": 0.10,
            "paired_gain_lcb_must_be_strictly_positive": True,
            "zero_gain_lcb_authorizes_only_noninferiority_not_primary": True,
            "root_recovery_uncertainty_policy": uncertainty_contract[
                "root_recovery_uncertainty_policy"
            ],
            "root_structured_uncertainty_head_count": 5,
            "deployment_uncertainty_contract_sha256": uncertainty_contract[
                "deployment_uncertainty_contract_sha256"
            ],
            "shared_uncertainty_implementation_path": str(implementation),
            "shared_uncertainty_implementation_file_sha256": (
                bridge.file_sha256(implementation)
            ),
        },
        "root_group_ranker_sha256",
    )
    return signed(
        {
            "format": bridge.CALIBRATION_FORMAT,
            "status": bridge.CALIBRATION_STATUS,
            "member_count": 5,
            "metrics": metrics,
            "uncertainty_decomposition": {},
            "deployment_root_structured_uncertainty_contract": uncertainty_contract,
            "deployment_uncertainty_contract_sha256": uncertainty_contract[
                "deployment_uncertainty_contract_sha256"
            ],
            "head_enabled_for_primary": enabled,
            "head_performance_gate_protocol": (
                "five_fold_logical_group_crossfit_group_bootstrap_zero_gain_lcb_v1"
            ),
            "all_six_heads_support_performance_uncertainty_gate_passed": True,
            "prediction_contract": {"format": "synthetic_prediction_contract"},
            "success_temperature_fitted_on_validation_only": True,
            "recovery_temperature_fitted_on_validation_only": True,
            "duration_scale_fitted_by_group_crossfit": True,
            "object_scale_fitted_by_group_crossfit": True,
            "recovery_enters_primary_only_if_support_and_calibration_pass": True,
            "abstain_threshold": {
                "status": "frozen_validation_group_bootstrap_lcb",
                "enabled": True,
                "maximum_total_uncertainty": 0.25,
                "minimum_retained_groups": 50,
                "test_or_paired_outcomes_used_for_selection": False,
            },
            "root_group_ranker": root_ranker,
            "root_group_ranker_enabled_for_primary": True,
            "source_rank_numeric_contract": bridge.SOURCE_RANK_NUMERIC_CONTRACT,
            **member_authority,
            "validation_groups": 190,
            "validation_samples": 760,
            "test_artifacts_read": False,
            "test_hdf5_files_opened": 0,
            "fresh_artifacts_read": False,
            "confirmation_artifacts_read": False,
            "paired_development_outcomes_read": False,
            "performance_claim_authorized": False,
        },
        "calibration_sha256",
    )


def head_support_artifact() -> dict[str, object]:
    minimum = {
        "post_event": 10,
        "next_event": 10,
        "duration": 10,
        "success": 50,
        "recovery": 10,
        "object_effect": 50,
    }
    heads = {
        name: {
            "enabled_for_primary": True,
            "support_threshold_met": True,
            "performance_gate_passed": True,
            "uncertainty_gate_passed": True,
            "independent_positive_or_observed_groups": threshold,
            "independent_negative_or_censored_groups": threshold,
            "minimum_required_per_side": threshold,
            "support_source": "synthetic_validation_only",
        }
        for name, threshold in minimum.items()
    }
    heads["recovery"]["all_member_recovery_heads_trained"] = True
    return signed(
        {
            "format": bridge.HEAD_SUPPORT_FORMAT,
            "status": bridge.HEAD_SUPPORT_STATUS,
            "heads": heads,
            "paired_development_outcomes_read": False,
            "sealed_evaluation_reserve_outcomes_read": False,
        },
        "head_support_sha256",
    )


def ensemble_artifact(
    calibration: dict[str, object], head: dict[str, object],
    root_ranker_path: Path,
) -> dict[str, object]:
    members = []
    for index in range(5):
        checkpoint_sha = f"{index + 1:064x}"
        contract = source_rank_contract(index)
        members.append({
            "member_index": index,
            "member_seed": 100 + index,
            "checkpoint_path": f"/immutable/member_{index}.pt",
            "checkpoint_file_sha256": checkpoint_sha,
            "source_rank_score_contract": contract,
            "source_rank_score_contract_sha256": contract["contract_sha256"],
        })
    contract = calibration["deployment_root_structured_uncertainty_contract"]
    root_ranker = calibration["root_group_ranker"]
    root_file_sha = bridge.file_sha256(root_ranker_path)
    return signed(
        {
            "format": bridge.ENSEMBLE_FORMAT,
            "status": bridge.ENSEMBLE_STATUS,
            "member_count": 5,
            "members": members,
            "shared_contract": {"format": "synthetic_shared_contract"},
            "prediction_contract": {"format": "synthetic_prediction_contract"},
            "deployment_root_structured_uncertainty_contract": contract,
            "deployment_uncertainty_contract_sha256": contract[
                "deployment_uncertainty_contract_sha256"
            ],
            "post_event_temperature": 1.0,
            "next_event_temperature": 1.0,
            "success_temperature": 1.0,
            "conditional_recovery_temperature": 1.0,
            "duration_scale_multiplier": 1.0,
            "object_scale_multiplier": 1.0,
            "object_error_robust_scale_m": 1.0,
            "conditional_recovery_semantics": "p(recovery_given_operational_regress)",
            "conditional_recovery_activation_requires_observed_regress": True,
            "head_enabled_for_primary": calibration["head_enabled_for_primary"],
            "all_six_heads_support_performance_uncertainty_gate_passed": True,
            "root_group_ranker": {
                "path": str(root_ranker_path),
                "file_sha256": root_file_sha,
                "logical_sha256": root_ranker["root_group_ranker_sha256"],
                "enabled_for_primary": True,
            },
            "source_rank_numeric_contract": bridge.SOURCE_RANK_NUMERIC_CONTRACT,
            "source_rank_member_authority": calibration[
                "source_rank_member_authority"
            ],
            "source_rank_member_authority_sha256": calibration[
                "source_rank_member_authority_sha256"
            ],
            "maximum_total_uncertainty": 0.25,
            "abstain_threshold_enabled": True,
            "calibration_sha256": calibration["calibration_sha256"],
            "head_support_sha256": head["head_support_sha256"],
            "root_group_ranker_path": str(root_ranker_path),
            "root_group_ranker_file_sha256": root_file_sha,
            "root_group_ranker_sha256": root_ranker[
                "root_group_ranker_sha256"
            ],
            "root_group_ranker_enabled_for_primary": True,
            "validation_identity_set_sha256": "d" * 64,
            "test_artifacts_read": False,
            "test_hdf5_files_opened": 0,
            "fresh_artifacts_read": False,
            "confirmation_artifacts_read": False,
            "paired_development_outcomes_read": False,
        },
        "ensemble_manifest_sha256",
    )


def policy_bridge_artifact() -> dict[str, object]:
    return signed(
        {
            "status": "verified_exact_policy_feature_action_bridge",
            "policy": "smolvla",
            "checkpoint_family": "smolvla_native_event_world_model",
            "bridge_contract_sha256": "1" * 64,
            "runtime_binding_sha256": "2" * 64,
            "state_feature_source_sha256": "3" * 64,
            "state_feature_dimension": 960,
            "state_feature_binding_sha256": "4" * 64,
            "action_mapping": bridge.ACTION_MAPPING,
            "action_mapping_binding_sha256": "5" * 64,
            "policy_row": 0,
            "canonical_event_interface": "canonical_event_id_and_reversible_predicates_v1",
            "canonical_action_effect_interface": "masked_canonical_action_chunk_and_feature_validity_v1",
            "cross_policy_latent_reuse_allowed": False,
        },
        "verification_sha256",
    )


def fixture(tmp_path: Path) -> dict[str, object]:
    manifest, attestation = target_artifacts()
    calibration = calibration_artifact()
    head = head_support_artifact()
    root_ranker_path = write_json(
        tmp_path / "formal190_root_group_ranker.json",
        calibration["root_group_ranker"],
    )
    ensemble = ensemble_artifact(calibration, head, root_ranker_path)
    paths = {
        "target_manifest": write_json(tmp_path / "target_manifest.json", manifest),
        "selected_identity_attestation": write_json(
            tmp_path / "identity_attestation.json", attestation
        ),
        "ensemble_manifest": write_json(
            tmp_path / "ensemble_manifest.json", ensemble
        ),
        "calibration": write_json(tmp_path / "calibration.json", calibration),
        "head_support": write_json(tmp_path / "head_support.json", head),
        "policy_bridge_receipt": write_json(
            tmp_path / "policy_bridge.json", policy_bridge_artifact()
        ),
    }
    receipt = signed(
        {
            "format": bridge.CALIBRATION_RECEIPT_FORMAT,
            "status": bridge.CALIBRATION_RECEIPT_STATUS,
            "input_authority_path": "/synthetic/input_authority.json",
            "input_authority_file_sha256": "6" * 64,
            "input_authority_sha256": "7" * 64,
            "member_count": 5,
            "validation_only": True,
            "shared_contract": {"format": "synthetic_shared_contract"},
            "prediction_contract_sha256": "8" * 64,
            "calibration_path": str(paths["calibration"]),
            "calibration_file_sha256": bridge.file_sha256(paths["calibration"]),
            "calibration_sha256": calibration["calibration_sha256"],
            "head_support_path": str(paths["head_support"]),
            "head_support_file_sha256": bridge.file_sha256(paths["head_support"]),
            "head_support_sha256": head["head_support_sha256"],
            "root_group_ranker_path": str(root_ranker_path),
            "root_group_ranker_file_sha256": bridge.file_sha256(
                root_ranker_path
            ),
            "root_group_ranker_sha256": calibration["root_group_ranker"][
                "root_group_ranker_sha256"
            ],
            "root_group_ranker_enabled_for_primary": True,
            "source_rank_numeric_contract": bridge.SOURCE_RANK_NUMERIC_CONTRACT,
            "source_rank_member_authority": calibration[
                "source_rank_member_authority"
            ],
            "source_rank_member_authority_sha256": calibration[
                "source_rank_member_authority_sha256"
            ],
            "deployment_uncertainty_contract_sha256": calibration[
                "deployment_uncertainty_contract_sha256"
            ],
            "ensemble_manifest_path": str(paths["ensemble_manifest"]),
            "ensemble_manifest_file_sha256": bridge.file_sha256(
                paths["ensemble_manifest"]
            ),
            "ensemble_manifest_sha256": ensemble["ensemble_manifest_sha256"],
            "abstain_threshold_enabled": True,
            "test_artifacts_read": False,
            "test_hdf5_files_opened": 0,
            "fresh_paths_accepted": False,
            "confirmation_artifacts_read": False,
            "paired_development_outcomes_read": False,
            "performance_or_transfer_claim_authorized": False,
            "artifacts_frozen_read_only": True,
        },
        "receipt_sha256",
    )
    paths["calibration_receipt"] = write_json(
        tmp_path / "calibration_receipt.json", receipt
    )
    kwargs: dict[str, object] = {}
    for role, path in paths.items():
        kwargs[f"{role}_path"] = path
        kwargs[f"{role}_file_sha256"] = bridge.file_sha256(path)
    return {"kwargs": kwargs, "paths": paths, "values": {
        "manifest": manifest,
        "attestation": attestation,
        "calibration": calibration,
        "head": head,
        "ensemble": ensemble,
        "receipt": receipt,
    }}


def resign_file(
    data: dict[str, object], role: str, signature: str, fixture_value: dict[str, object]
) -> None:
    paths = fixture_value["paths"]
    kwargs = fixture_value["kwargs"]
    value = signed(data, signature)
    write_json(paths[role], value)
    kwargs[f"{role}_file_sha256"] = bridge.file_sha256(paths[role])


def rewrite_member_authority_closure(
    data: dict[str, object], authority: dict[str, object],
) -> None:
    paths = data["paths"]
    kwargs = data["kwargs"]
    authority_sha = bridge.canonical_sha256(authority)

    calibration = copy.deepcopy(data["values"]["calibration"])
    calibration.pop("calibration_sha256")
    root = copy.deepcopy(calibration["root_group_ranker"])
    root.pop("root_group_ranker_sha256")
    root["source_rank_member_authority"] = copy.deepcopy(authority)
    root["source_rank_member_authority_sha256"] = authority_sha
    root = signed(root, "root_group_ranker_sha256")
    calibration["root_group_ranker"] = root
    calibration["source_rank_member_authority"] = copy.deepcopy(authority)
    calibration["source_rank_member_authority_sha256"] = authority_sha
    calibration = signed(calibration, "calibration_sha256")
    write_json(paths["calibration"], calibration)
    kwargs["calibration_file_sha256"] = bridge.file_sha256(paths["calibration"])

    receipt_original = data["values"]["receipt"]
    root_path = Path(receipt_original["root_group_ranker_path"])
    write_json(root_path, root)
    root_file_sha = bridge.file_sha256(root_path)

    ensemble = copy.deepcopy(data["values"]["ensemble"])
    ensemble.pop("ensemble_manifest_sha256")
    ensemble["calibration_sha256"] = calibration["calibration_sha256"]
    ensemble["root_group_ranker_sha256"] = root["root_group_ranker_sha256"]
    ensemble["root_group_ranker_file_sha256"] = root_file_sha
    ensemble["root_group_ranker"] = {
        "path": str(root_path),
        "file_sha256": root_file_sha,
        "logical_sha256": root["root_group_ranker_sha256"],
        "enabled_for_primary": True,
    }
    ensemble["source_rank_member_authority"] = copy.deepcopy(authority)
    ensemble["source_rank_member_authority_sha256"] = authority_sha
    ensemble = signed(ensemble, "ensemble_manifest_sha256")
    write_json(paths["ensemble_manifest"], ensemble)
    kwargs["ensemble_manifest_file_sha256"] = bridge.file_sha256(
        paths["ensemble_manifest"]
    )

    receipt = copy.deepcopy(receipt_original)
    receipt.pop("receipt_sha256")
    receipt["calibration_file_sha256"] = kwargs["calibration_file_sha256"]
    receipt["calibration_sha256"] = calibration["calibration_sha256"]
    receipt["root_group_ranker_file_sha256"] = root_file_sha
    receipt["root_group_ranker_sha256"] = root["root_group_ranker_sha256"]
    receipt["ensemble_manifest_file_sha256"] = kwargs[
        "ensemble_manifest_file_sha256"
    ]
    receipt["ensemble_manifest_sha256"] = ensemble[
        "ensemble_manifest_sha256"
    ]
    receipt["source_rank_member_authority"] = copy.deepcopy(authority)
    receipt["source_rank_member_authority_sha256"] = authority_sha
    receipt = signed(receipt, "receipt_sha256")
    write_json(paths["calibration_receipt"], receipt)
    kwargs["calibration_receipt_file_sha256"] = bridge.file_sha256(
        paths["calibration_receipt"]
    )


def resign_bridge_closure(value: dict[str, object]) -> dict[str, object]:
    """Re-sign every public hash after a synthetic in-memory authority edit."""
    changed = copy.deepcopy(value)
    changed.pop("bridge_sha256", None)
    deployment = changed["deployment"]
    deployment.pop("deployment_binding_sha256", None)
    selector = deployment["selector_authority"]
    selector.pop("selector_authority_sha256", None)
    selector["selector_authority_sha256"] = bridge.canonical_sha256(selector)
    deployment["selector_authority_sha256"] = selector[
        "selector_authority_sha256"
    ]
    deployment["deployment_binding_sha256"] = bridge.canonical_sha256(deployment)
    changed["bridge_sha256"] = bridge.canonical_sha256(changed)
    return changed


def test_evaluation400_is_the_only_lane_and_preserves_pair_identity(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    audit = bridge.validate_bridge(value)

    assert audit["pair_count"] == 400
    assert audit["execution_authorized"] is False
    assert value["scope"]["additional_reserve400_required"] is False
    assert value["scope"]["additional_reserve400_count"] == 0
    evaluation = data["values"]["manifest"]["splits"]["evaluation"]
    assert [row["pair_id"] for row in value["pairs"]] == [
        row["pair_id"] for row in evaluation
    ]
    assert [row["target_manifest_global_ordinal"] for row in value["pairs"]] == list(
        range(130, 530)
    )
    assert all(row["same_initial_state_for_both_conditions"] for row in value["pairs"])
    assert value["preoutcome_capability_receipt"]["hdf5_files_opened"] == 0
    assert value["preoutcome_capability_receipt"]["labels_or_outcomes_read"] is False
    assert value["preoutcome_capability_receipt"]["pair_conditions_executed"] == 0
    assert value["deployment"]["selector_authority"][
        "source_rank_numeric_contract"
    ] == data["values"]["calibration"]["source_rank_numeric_contract"]
    member_authority = value["deployment"]["source_rank_member_authority"]
    assert member_authority == value["deployment"]["selector_authority"][
        "source_rank_member_authority"
    ]
    assert value["deployment"]["source_rank_member_authority_sha256"] == (
        bridge.canonical_sha256(member_authority)
    )


@pytest.mark.parametrize(
    "tamper",
    ("reorder", "temperature", "source_sha", "contract_sha", "missing_field"),
)
def test_member_authority_tamper_fails_even_after_full_upstream_resign(
    tmp_path: Path, tamper: str,
) -> None:
    data = fixture(tmp_path)
    authority = copy.deepcopy(
        data["values"]["calibration"]["source_rank_member_authority"]
    )
    if tamper == "reorder":
        authority["members"][0], authority["members"][1] = (
            authority["members"][1], authority["members"][0]
        )
    elif tamper == "temperature":
        authority["members"][0]["success_temperature"] = 2.0
    elif tamper == "source_sha":
        authority["members"][0]["source_checkpoint_file_sha256"] = "e" * 64
    elif tamper == "contract_sha":
        authority["members"][0]["source_rank_score_contract_sha256"] = "f" * 64
    else:
        authority["members"][0].pop("success_temperature")
    rewrite_member_authority_closure(data, authority)
    with pytest.raises(
        bridge.Evaluation400BridgeError,
        match="source rank member authority|ensemble",
    ):
        bridge.freeze_bridge(**data["kwargs"])


@pytest.mark.parametrize(
    "role",
    ("calibration", "root_group_ranker", "ensemble_manifest", "calibration_receipt"),
)
def test_member_authority_must_be_identical_at_each_upstream_layer(
    tmp_path: Path, role: str,
) -> None:
    data = fixture(tmp_path)
    value_key = {
        "calibration": "calibration",
        "root_group_ranker": "calibration",
        "ensemble_manifest": "ensemble",
        "calibration_receipt": "receipt",
    }[role]
    artifact = copy.deepcopy(data["values"][value_key])
    signature = {
        "calibration": "calibration_sha256",
        "root_group_ranker": "calibration_sha256",
        "ensemble_manifest": "ensemble_manifest_sha256",
        "calibration_receipt": "receipt_sha256",
    }[role]
    artifact.pop(signature)
    target = artifact
    if role == "root_group_ranker":
        root = copy.deepcopy(artifact["root_group_ranker"])
        root.pop("root_group_ranker_sha256")
        target = root
    authority = copy.deepcopy(target["source_rank_member_authority"])
    authority["members"][0]["success_temperature"] = 2.0
    target["source_rank_member_authority"] = authority
    target["source_rank_member_authority_sha256"] = bridge.canonical_sha256(
        authority
    )
    if role == "root_group_ranker":
        artifact["root_group_ranker"] = signed(
            target, "root_group_ranker_sha256"
        )
    resign_file(artifact, {
        "root_group_ranker": "calibration",
    }.get(role, role), signature, data)
    with pytest.raises(
        bridge.Evaluation400BridgeError,
        match="calibration|ensemble|receipt",
    ):
        bridge.freeze_bridge(**data["kwargs"])


@pytest.mark.parametrize(
    "role",
    ("calibration", "root_group_ranker", "ensemble_manifest", "calibration_receipt"),
)
@pytest.mark.parametrize("mutation", ("missing", "drift"))
def test_upstream_source_rank_numeric_contract_is_a_required_equal_closure(
    tmp_path: Path, role: str, mutation: str,
) -> None:
    data = fixture(tmp_path)
    field = "source_rank_numeric_contract"
    replacement = "ieee754_float64_reassociated"
    if role in {"calibration", "root_group_ranker"}:
        artifact = copy.deepcopy(data["values"]["calibration"])
        artifact.pop("calibration_sha256")
        target = artifact
        if role == "root_group_ranker":
            root = copy.deepcopy(artifact["root_group_ranker"])
            root.pop("root_group_ranker_sha256")
            if mutation == "missing":
                root.pop(field)
            else:
                root[field] = replacement
            artifact["root_group_ranker"] = signed(
                root, "root_group_ranker_sha256"
            )
            target = None
        if target is not None:
            if mutation == "missing":
                target.pop(field)
            else:
                target[field] = replacement
        resign_file(artifact, "calibration", "calibration_sha256", data)
    else:
        signature = (
            "ensemble_manifest_sha256"
            if role == "ensemble_manifest" else "receipt_sha256"
        )
        value_key = "ensemble" if role == "ensemble_manifest" else "receipt"
        artifact = copy.deepcopy(data["values"][value_key])
        artifact.pop(signature)
        if mutation == "missing":
            artifact.pop(field)
        else:
            artifact[field] = replacement
        resign_file(artifact, role, signature, data)
    with pytest.raises(
        bridge.Evaluation400BridgeError,
        match="calibration|ensemble|receipt",
    ):
        bridge.freeze_bridge(**data["kwargs"])


def test_target_pair_id_tamper_fails_even_when_manifest_is_resigned(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    manifest = copy.deepcopy(data["values"]["manifest"])
    manifest.pop("seed_manifest_sha256")
    manifest["splits"]["evaluation"][0]["pair_id"] = "f" * 64
    resign_file(manifest, "target_manifest", "seed_manifest_sha256", data)
    with pytest.raises(bridge.Evaluation400BridgeError, match="row identity"):
        bridge.freeze_bridge(**data["kwargs"])


def test_attestation_must_be_the_one_embedded_by_manifest(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    attestation = copy.deepcopy(data["values"]["attestation"])
    attestation.pop("attestation_sha256")
    attestation["heldout_identity_set_sha256"] = "e" * 64
    resign_file(
        attestation,
        "selected_identity_attestation",
        "attestation_sha256",
        data,
    )
    with pytest.raises(bridge.Evaluation400BridgeError, match="attestation"):
        bridge.freeze_bridge(**data["kwargs"])


def test_five_members_core_heads_and_abstention_are_hard_gates(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    calibration = copy.deepcopy(data["values"]["calibration"])
    calibration.pop("calibration_sha256")
    calibration["abstain_threshold"]["enabled"] = False
    resign_file(calibration, "calibration", "calibration_sha256", data)
    with pytest.raises(
        bridge.Evaluation400BridgeError, match="calibration/abstention"
    ):
        bridge.freeze_bridge(**data["kwargs"])


@pytest.mark.parametrize(
    "mutation",
    (
        "heldout_parameter_leak",
        "fold_index_bool",
        "zero_oof_lcb",
        "duplicate_oof_group",
    ),
)
def test_selection_aware_oof_evidence_tamper_fails_after_full_resign(
    mutation: str,
) -> None:
    calibration = calibration_artifact()
    root = copy.deepcopy(calibration["root_group_ranker"])
    root.pop("root_group_ranker_sha256")
    evidence = copy.deepcopy(root["selection_aware_oof_evidence"])
    evidence.pop("selection_aware_oof_evidence_sha256")
    if mutation in {"heldout_parameter_leak", "fold_index_bool"}:
        contract = copy.deepcopy(evidence["root_outer_nesting_contract"])
        contract.pop("root_outer_nesting_contract_sha256")
        if mutation == "heldout_parameter_leak":
            contract[
                "outer_heldout_labels_used_for_any_parameter_or_selection"
            ] = True
        else:
            parameter = copy.deepcopy(contract["fold_parameters"][0])
            parameter.pop("fold_parameter_sha256")
            parameter["outer_fold"] = False
            contract["fold_parameters"][0] = signed(
                parameter, "fold_parameter_sha256"
            )
        contract = signed(contract, "root_outer_nesting_contract_sha256")
        evidence["root_outer_nesting_contract"] = contract
        evidence["root_outer_nesting_contract_sha256"] = contract[
            "root_outer_nesting_contract_sha256"
        ]
    elif mutation == "zero_oof_lcb":
        evidence["paired_gain_group_bootstrap_lcb95"] = 0.0
    else:
        evidence["stitched_group_decisions"][1]["logical_group_id"] = evidence[
            "stitched_group_decisions"
        ][0]["logical_group_id"]
    evidence = signed(evidence, "selection_aware_oof_evidence_sha256")
    root["selection_aware_oof_evidence"] = evidence
    root["selection_aware_oof_evidence_sha256"] = evidence[
        "selection_aware_oof_evidence_sha256"
    ]
    root = signed(root, "root_group_ranker_sha256")
    calibration.pop("calibration_sha256")
    calibration["root_group_ranker"] = root
    calibration = signed(calibration, "calibration_sha256")
    with pytest.raises(
        bridge.Evaluation400BridgeError, match="root (OOF|outer)"
    ):
        bridge.validate_calibration(calibration)


def test_policy_runtime_or_action_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    policy_path = data["paths"]["policy_bridge_receipt"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["runtime_binding_sha256"] = "e" * 64
    write_json(policy_path, policy)
    data["kwargs"]["policy_bridge_receipt_file_sha256"] = bridge.file_sha256(
        policy_path
    )
    with pytest.raises(bridge.Evaluation400BridgeError, match="logical signature"):
        bridge.freeze_bridge(**data["kwargs"])


def test_expected_file_sha_rejected_before_protocol_construction(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    data["kwargs"]["target_manifest_file_sha256"] = "0" * 64
    with pytest.raises(bridge.Evaluation400BridgeError, match="file SHA mismatch"):
        bridge.freeze_bridge(**data["kwargs"])


def test_condition_order_and_extra_bridge_fields_fail_after_resigning(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    changed = copy.deepcopy(value)
    changed.pop("bridge_sha256")
    changed["pairs"][0]["condition_order"].reverse()
    changed = signed(changed, "bridge_sha256")
    with pytest.raises(bridge.Evaluation400BridgeError, match="identity/order"):
        bridge.validate_bridge(changed)

    changed = copy.deepcopy(value)
    changed.pop("bridge_sha256")
    changed["unexpected"] = True
    changed = signed(changed, "bridge_sha256")
    with pytest.raises(bridge.Evaluation400BridgeError, match="boundary"):
        bridge.validate_bridge(changed)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("deployment_parameters", "success_temperature"),
        ("formal190_thresholds", "minimum_formal190_composite_margin"),
    ),
)
def test_selector_values_cannot_diverge_from_deployment_mirror(
    tmp_path: Path, section: str, field: str,
) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    changed = copy.deepcopy(value)
    changed["deployment"]["selector_authority"][section] = dict(
        changed["deployment"]["selector_authority"][section]
    )
    changed["deployment"]["selector_authority"][section][field] = 1.25
    changed = resign_bridge_closure(changed)
    with pytest.raises(bridge.Evaluation400BridgeError, match="selector authority"):
        bridge.validate_bridge(changed)


def test_selector_threshold_bool_cannot_impersonate_numeric_after_full_resign(
    tmp_path: Path,
) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    changed = copy.deepcopy(value)
    field = "minimum_formal190_composite_margin"
    changed["deployment"]["selector_authority"]["formal190_thresholds"][field] = True
    changed["deployment"]["formal190_thresholds"][field] = True
    changed = resign_bridge_closure(changed)
    with pytest.raises(bridge.Evaluation400BridgeError, match="selector authority"):
        bridge.validate_bridge(changed)


def test_source_rank_contract_temperature_bool_fails_after_full_resign(
    tmp_path: Path,
) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    changed = copy.deepcopy(value)
    contract = changed["deployment"]["source_rank_score_contracts"][0]
    contract.pop("contract_sha256")
    contract["success_temperature"] = True
    contract["contract_sha256"] = bridge.canonical_sha256(contract)
    changed["deployment"]["source_rank_score_contract_sha256"][0] = contract[
        "contract_sha256"
    ]
    changed = resign_bridge_closure(changed)
    with pytest.raises(bridge.Evaluation400BridgeError, match="selector authority"):
        bridge.validate_bridge(changed)


def test_source_rank_numeric_contract_drift_fails_after_full_resign(
    tmp_path: Path,
) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    changed = copy.deepcopy(value)
    changed["deployment"]["selector_authority"][
        "source_rank_numeric_contract"
    ] = "ieee754_float64_reassociated"
    changed = resign_bridge_closure(changed)
    with pytest.raises(bridge.Evaluation400BridgeError, match="selector authority"):
        bridge.validate_bridge(changed)


def test_output_is_create_once_owner_read_only(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    value = bridge.freeze_bridge(**data["kwargs"])
    output = tmp_path / "paired_identity_bridge_v2.json"
    bridge.write_json_new(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        bridge.write_json_new(output, value)


def test_hdf_input_path_is_rejected_without_opening_it(tmp_path: Path) -> None:
    path = tmp_path / "forbidden.hdf5"
    path.write_bytes(b"not opened")
    with pytest.raises(bridge.Evaluation400BridgeError, match="JSON"):
        bridge._read_bound_json(path, "0" * 64, "forbidden")
