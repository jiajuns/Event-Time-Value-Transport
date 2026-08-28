#!/usr/bin/env python3
"""Freeze a label-blind Piper-only dual-provider router-development pool.

This standard-library-only entry point accepts no input files.  Before any
environment reset, policy query, trajectory, HDF5 payload, outcome, or target
label can be read, it deterministically commits 320 new Piper execution-group
candidates in one physical router-development split.  It also freezes the
canonical nested five-fold sizing, per-head support gates, external identity
disjointness roles, and the requirement that both five-member providers make
content-addressed predictions before target labels may open.

The document is a create-once preregistration only.  It authorizes no reset,
collection, training, prediction, label opening, calibration, promotion, or
performance claim, and it does not modify the historical development300 v1
protocol or its Formal190 membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


FORMAT = "etsf_smolvla_piper_schema6_routerdev320_preregistration_v1"
STATUS = "preregistered_label_blind_routerdev320_collection_not_authorized"
PARTITION_FORMAT = "etsf_smolvla_piper_schema6_routerdev320_partition_v1"
SUPPORT_FORMAT = "etsf_smolvla_piper_schema6_routerdev320_support_gates_v1"
NESTED_OOF_FORMAT = "etsf_smolvla_piper_schema6_routerdev320_nested_oof_v1"
DISJOINTNESS_FORMAT = (
    "etsf_smolvla_piper_schema6_routerdev320_external_disjointness_requirements_v1"
)
PREDICTION_BOUNDARY_FORMAT = (
    "etsf_smolvla_piper_schema6_routerdev320_prediction_before_label_v1"
)

TASK = "move_can_pot"
BODY = "piper"
POLICY = "smolvla"
ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
INSTRUCTION = "move the can into the pot"
NAMESPACE = "schema6_piper_dual_provider_routerdev320_20260829_v1"
SPLIT = "dual_provider_router_development"
DEFAULT_SEED_BASE = 2_026_084_500
TOTAL_GROUPS = 320
CANDIDATES_PER_GROUP = 4
FOLD_COUNT = 5
OUTER_HELDOUT_GROUPS = 64
OUTER_TRAINING_GROUPS = 256
INNER_HELDOUT_GROUPS_BY_FOLD = (52, 51, 51, 51, 51)
INNER_TRAINING_GROUPS_BY_FOLD = (204, 205, 205, 205, 205)
MAX_SEED = 2**31 - 1
PRIOR_SINGLE_GROUP_SEED = 100_101_000
LEGACY_DEVELOPMENT300_SEED_MIN = 2_026_082_800
LEGACY_DEVELOPMENT300_SEED_MAX = 2_026_083_099
SOURCE_DENSE260_CANDIDATE_SEED_MIN = 2_026_083_500
SOURCE_DENSE260_CANDIDATE_SEED_MAX = 2_026_083_899
PLANNING_SINGLE_CANDIDATE_SUCCESS_WILSON_LOWER = 0.0981819861
PLANNING_FOUR_CANDIDATE_IID_GROUP_SUCCESS = 0.3385825866
PLANNING_NESTED_INNER_SCOPE_COUNT = 25
PLANNING_FAMILYWISE_FAILURE_BUDGET = 0.05
PLANNING_MINIMUM_INNER_GROUPS = 203
PLANNING_ANALYTICAL_TOTAL_GROUPS = 318
EVENTS = ("e0", "e12", "e3", "e4", "eK")
NEXT_EVENTS = ("e12", "e3", "e4", "eK")
HEADS = (
    "post_event",
    "next_event",
    "duration",
    "success",
    "recovery",
    "object_effect",
)
PROVIDERS = ("body_agnostic_adapter", "body_conditioned_adapter")
EXTERNAL_DISJOINTNESS_ROLES = (
    "provider_training_closure",
    "formal190",
    "evaluation400",
)
SENSITIVE_OUTPUT_TOKENS = (
    "fresh",
    "confirmation",
    "formal",
    "evaluation",
    "adapter_train",
    "adapter-train",
)
SHA_CHARS = frozenset("0123456789abcdef")


class RouterDevelopment320PreregistrationError(RuntimeError):
    """The immutable router-development preregistration is invalid."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= SHA_CHARS
    )


def _ranges_overlap(left_min: int, left_max: int, right_min: int, right_max: int) -> bool:
    return max(left_min, right_min) <= min(left_max, right_max)


def _seed_order(seed_base: int) -> list[int]:
    if isinstance(seed_base, bool) or not isinstance(seed_base, int):
        raise RouterDevelopment320PreregistrationError(
            "seed namespace must use an exact integer base"
        )
    seed_maximum = seed_base + TOTAL_GROUPS - 1
    if (
        seed_base < 0
        or seed_maximum > MAX_SEED
        or seed_base <= PRIOR_SINGLE_GROUP_SEED <= seed_maximum
        or _ranges_overlap(
            seed_base,
            seed_maximum,
            LEGACY_DEVELOPMENT300_SEED_MIN,
            LEGACY_DEVELOPMENT300_SEED_MAX,
        )
        or _ranges_overlap(
            seed_base,
            seed_maximum,
            SOURCE_DENSE260_CANDIDATE_SEED_MIN,
            SOURCE_DENSE260_CANDIDATE_SEED_MAX,
        )
    ):
        raise RouterDevelopment320PreregistrationError(
            "seed namespace is invalid or overlaps a frozen prior namespace"
        )
    candidates = list(range(seed_base, seed_base + TOTAL_GROUPS))
    return sorted(
        candidates,
        key=lambda seed: hashlib.sha256(
            f"{NAMESPACE}:{seed}".encode("ascii")
        ).hexdigest(),
    )


def _support_contract() -> dict[str, Any]:
    return {
        "format": SUPPORT_FORMAT,
        "support_unit": (
            "both_independent_execution_groups_and_independent_semantic_reset_clusters"
        ),
        "required_scopes": [
            "global",
            "every_body_actor_context",
            "every_outer_training_scope",
            "every_inner_training_scope",
        ],
        "body_actor_context_count_in_this_protocol": 1,
        "body_actor_context": {
            "body_id": BODY,
            "actor_id": ACTOR_ID,
            "body_contract_sha256_required_at_identity_freeze": True,
            "actor_contract_id_required_at_identity_freeze": True,
            "actor_contract_sha256_required_at_identity_freeze": True,
        },
        "heads": {
            "post_event": {
                "categories": list(EVENTS),
                "minimum_per_category": 10,
            },
            "next_event": {
                "categories": list(NEXT_EVENTS),
                "minimum_per_category": 10,
                "observed_reached_events_only": True,
            },
            "duration": {
                "categories": ["observed", "censored"],
                "minimum_per_category": 10,
                "right_censoring_aware": True,
            },
            "success": {
                "categories": ["positive", "negative"],
                "minimum_per_category": 50,
                "unobserved_or_right_censored_outcomes_are_not_negative": True,
            },
            "recovery": {
                "categories": ["positive", "negative"],
                "minimum_per_category": 10,
                "applicability": "regress_and_recovery_observed_only",
                "right_censored_nonrecoveries_are_not_negative": True,
            },
            "object_effect": {
                "categories": ["nonzero", "near_zero"],
                "minimum_per_category": 50,
                "observed_valid_object_supervision_only": True,
            },
        },
        "all_six_heads_required_for_route_receipt": True,
        "quota_semantics": (
            "post_collection_activation_gates_never_membership_seed_or_stopping_targets"
        ),
        "insufficient_support_action": (
            "disable_head_and_export_no_route_receipt_no_seed_replacement_or_extension"
        ),
        "sample_count_or_public_success_rate_guarantees_gate_passage": False,
    }


def _nested_oof_contract() -> dict[str, Any]:
    return {
        "format": NESTED_OOF_FORMAT,
        "fold_count": FOLD_COUNT,
        "fold_unit": "semantic_reset_cluster_id",
        "one_execution_group_per_semantic_reset_cluster_required": True,
        "assignment_algorithm": (
            "sort_unique_semantic_reset_cluster_id_then_zero_based_index_mod_5_v1"
        ),
        "plan_materialization_stage": (
            "after_label_blind_exact_reset_identity_freeze_before_collection_or_label_open"
        ),
        "noncanonical_or_caller_selected_fold_plan_accepted": False,
        "outer": {
            "total_groups": TOTAL_GROUPS,
            "total_semantic_reset_clusters": TOTAL_GROUPS,
            "heldout_groups_per_fold": OUTER_HELDOUT_GROUPS,
            "heldout_semantic_reset_clusters_per_fold": OUTER_HELDOUT_GROUPS,
            "training_groups_per_fold": OUTER_TRAINING_GROUPS,
            "training_semantic_reset_clusters_per_fold": OUTER_TRAINING_GROUPS,
            "each_group_and_cluster_heldout_exactly_once": True,
        },
        "inner_within_each_outer_training_scope": {
            "domain_groups": OUTER_TRAINING_GROUPS,
            "domain_semantic_reset_clusters": OUTER_TRAINING_GROUPS,
            "heldout_groups_by_fold": list(INNER_HELDOUT_GROUPS_BY_FOLD),
            "heldout_semantic_reset_clusters_by_fold": list(
                INNER_HELDOUT_GROUPS_BY_FOLD
            ),
            "training_groups_by_fold": list(INNER_TRAINING_GROUPS_BY_FOLD),
            "training_semantic_reset_clusters_by_fold": list(
                INNER_TRAINING_GROUPS_BY_FOLD
            ),
            "minimum_training_fraction_of_full_routerdev": 0.6375,
            "each_outer_training_group_and_cluster_heldout_exactly_once": True,
        },
        "outer_heldout_labels_may_fit_or_select_provider_calibration_or_threshold": False,
        "inner_heldout_labels_may_fit_the_parameter_scored_on_that_inner_fold": False,
        "formal190_or_evaluation400_may_enter_nested_oof": False,
    }


def _sample_size_planning_evidence() -> dict[str, Any]:
    return {
        "format": "etsf_smolvla_piper_schema6_routerdev320_sample_size_planning_v1",
        "status": "planning_evidence_only_not_support_or_performance_guarantee",
        "public_single_candidate_success_wilson_lower_bound": (
            PLANNING_SINGLE_CANDIDATE_SUCCESS_WILSON_LOWER
        ),
        "four_candidate_iid_group_success_probability": (
            PLANNING_FOUR_CANDIDATE_IID_GROUP_SUCCESS
        ),
        "four_candidate_probability_formula": "q=1-(1-p)^4",
        "iid_candidate_assumption_is_verified_by_this_preregistration": False,
        "success_positive_support_target_per_scope": 50,
        "outer_fold_count": FOLD_COUNT,
        "inner_folds_per_outer_fold": FOLD_COUNT,
        "nested_inner_scope_count_for_union_bound": (
            PLANNING_NESTED_INNER_SCOPE_COUNT
        ),
        "familywise_failure_probability_budget": (
            PLANNING_FAMILYWISE_FAILURE_BUDGET
        ),
        "bonferroni_per_inner_scope_failure_budget": 0.002,
        "minimum_inner_scope_groups_from_binomial_union_bound": (
            PLANNING_MINIMUM_INNER_GROUPS
        ),
        "union_bound_at_202_inner_groups": 0.0501255964,
        "union_bound_at_203_inner_groups": 0.0433199824,
        "analytical_minimum_full_groups_at_64pct_inner_fraction": (
            PLANNING_ANALYTICAL_TOTAL_GROUPS
        ),
        "engineering_rounded_full_groups": TOTAL_GROUPS,
        "canonical_routerdev320_minimum_inner_training_groups": min(
            INNER_TRAINING_GROUPS_BY_FOLD
        ),
        "sample_size_or_iid_assumption_guarantees_real_support": False,
        "post_collection_support_is_always_recomputed_from_real_labels": True,
        "insufficient_real_support_action": "disable_head_no_extension_or_replacement",
    }


def _disjointness_contract() -> dict[str, Any]:
    roles = [
        {
            "role": "provider_training_closure",
            "membership": (
                "union_of_shared_core_training_and_adapter_train_internal_selection"
            ),
        },
        {
            "role": "formal190",
            "membership": "frozen_formal190_identity_set_untouched",
        },
        {
            "role": "evaluation400",
            "membership": "frozen_evaluation400_identity_set_untouched",
        },
    ]
    return {
        "format": DISJOINTNESS_FORMAT,
        "required_external_roles": roles,
        "required_role_names_in_canonical_order": list(
            EXTERNAL_DISJOINTNESS_ROLES
        ),
        "candidate_pool_stage": {
            "target_role": "piper_routerdev320_requested_candidate_pool",
            "required_before_reset_authority": True,
            "required_zero_intersection_axes": [
                "requested_seed",
                "semantic_reset_request_sha256",
            ],
            "identity_or_label_values_from_reference_sets_disclosed": False,
        },
        "selected_resolved_stage": {
            "target_role": "piper_routerdev320_selected_resolved_identity_set",
            "required_before_collection_authority": True,
            "required_zero_intersection_axes": [
                "requested_seed",
                "resolved_seed",
                "semantic_reset_cluster_id",
                "execution_group_id",
            ],
            "same_reference_commitments_as_candidate_stage_required": True,
            "identity_or_label_values_from_reference_sets_disclosed": False,
        },
        "external_issuer_or_signature_verification_required": True,
        "self_asserted_boolean_or_self_hash_is_sufficient": False,
        "preregistration_itself_proves_external_disjointness": False,
        "missing_role_or_nonzero_intersection_action": (
            "fail_closed_no_reset_or_collection_authority"
        ),
    }


def _prediction_before_label_contract() -> dict[str, Any]:
    return {
        "format": PREDICTION_BOUNDARY_FORMAT,
        "canonical_provider_ids": list(PROVIDERS),
        "provider_count": len(PROVIDERS),
        "members_per_provider": 5,
        "total_provider_members": 10,
        "provider_pair_requirements": {
            "same_shared_core_lineage": True,
            "same_training_execution_group_ids": True,
            "same_training_semantic_reset_cluster_ids": True,
            "same_member_indices_and_training_seeds": True,
            "distinct_provider_artifact_sha256": True,
            "all_ten_checkpoint_file_sha256_bound": True,
            "provider_training_routerdev_disjointness_reverified": True,
        },
        "ordering": [
            "freeze_complete_two_by_five_provider_pair",
            "freeze_label_blind_router_input_view_and_sample_order",
            "run_all_ten_checkpoint_forwards",
            "freeze_raw_prediction_tensor_set_and_forward_receipt",
            "authorize_router_target_label_materialization",
            "run_fixed_nested_oof_calibrator",
        ],
        "raw_prediction_commitment_required_before_target_label_open": True,
        "prediction_process_accepts_target_label_or_formal_evaluation_path": False,
        "row_membership_or_order_may_change_after_prediction_commitment": False,
        "label_dependent_row_filtering_allowed": False,
        "calibrator_implementation_and_nested_fold_plan_frozen_before_label_open": True,
        "rank_route_included": False,
        "rank_action_selection_fallback": "actor_baseline",
        "routerdev_receipt_scope": (
            "formal190_dependency_candidate_only_not_evaluation_or_deployment_authority"
        ),
    }


def _stopping_contract() -> dict[str, Any]:
    return {
        "collection": {
            "stop_after_exactly_all_preregistered_groups_terminal": TOTAL_GROUPS,
            "early_stop_on_any_label_support_or_metric": False,
            "replace_failed_or_unresolved_seed_inside_this_contract": False,
            "requested_seed_must_resolve_bit_exact_or_protocol_fails_closed": True,
            "all_four_original_candidate_identities_accounted": True,
            "infeasible_candidate_may_be_censored_but_not_replaced": True,
        },
        "extension": {
            "automatic_extension_authorized": False,
            "post_label_seed_addition_authorized": False,
            "post_label_membership_or_fold_change_authorized": False,
            "new_data_requires_new_create_once_preregistration": True,
        },
    }


def build_preregistration(seed_base: int = DEFAULT_SEED_BASE) -> dict[str, Any]:
    ordered_seeds = _seed_order(seed_base)
    instruction_sha = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
    rows: list[dict[str, Any]] = []
    for ordinal, requested_seed in enumerate(ordered_seeds):
        semantic_request_base = {
            "format": "etsf_cross_body_semantic_reset_request_v1",
            "task": TASK,
            "instruction_sha256": instruction_sha,
            "requested_seed": requested_seed,
        }
        semantic_request_sha = canonical_sha256(semantic_request_base)
        identity_base = {
            "namespace": NAMESPACE,
            "task": TASK,
            "body": BODY,
            "policy": POLICY,
            "actor_id": ACTOR_ID,
            "split": SPLIT,
            "global_ordinal": ordinal,
            "split_ordinal": ordinal,
            "requested_seed": requested_seed,
            "expected_resolved_seed": requested_seed,
            "semantic_reset_request_sha256": semantic_request_sha,
        }
        identity_sha = canonical_sha256(identity_base)
        rows.append(
            {
                **identity_base,
                "logical_group_id": f"{NAMESPACE}/{identity_sha}",
                "identity_sha256": identity_sha,
                "resolution_status": (
                    "unresolved_candidate_exact_identity_and_external_disjointness_required"
                ),
            }
        )
    group_ids = [row["logical_group_id"] for row in rows]
    requested_seeds = [int(row["requested_seed"]) for row in rows]
    semantic_requests = [
        str(row["semantic_reset_request_sha256"]) for row in rows
    ]
    partition_base = {
        "format": PARTITION_FORMAT,
        "status": "frozen_label_blind_single_router_development_split",
        "namespace": NAMESPACE,
        "split_counts": {SPLIT: TOTAL_GROUPS},
        "members": {SPLIT: group_ids},
        "requested_seeds": {SPLIT: requested_seeds},
        "semantic_reset_request_sha256": {SPLIT: semantic_requests},
        "all_members_unique": True,
        "all_requested_seeds_unique": True,
        "all_semantic_reset_requests_unique": True,
        "adapter_training_members_included": 0,
        "formal190_members_included": 0,
        "evaluation400_members_included": 0,
    }
    partition = {
        **partition_base,
        "partition_sha256": canonical_sha256(partition_base),
    }
    base = {
        "format": FORMAT,
        "status": STATUS,
        "namespace": NAMESPACE,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "actor_id": ACTOR_ID,
        "instruction": INSTRUCTION,
        "instruction_sha256": instruction_sha,
        "total_groups": TOTAL_GROUPS,
        "semantic_reset_cluster_count_after_exact_identity_freeze": TOTAL_GROUPS,
        "execution_groups_per_semantic_reset_cluster": 1,
        "candidates_per_group": CANDIDATES_PER_GROUP,
        "planned_candidate_accounting_records": TOTAL_GROUPS
        * CANDIDATES_PER_GROUP,
        "physical_split": SPLIT,
        "split_counts": {SPLIT: TOTAL_GROUPS},
        "seed_generation": {
            "algorithm": "sha256_order_of_contiguous_reserved_numeric_namespace_v1",
            "seed_base": seed_base,
            "candidate_count": TOTAL_GROUPS,
            "maximum_seed": seed_base + TOTAL_GROUPS - 1,
            "legacy_development300_seed_range_excluded": [
                LEGACY_DEVELOPMENT300_SEED_MIN,
                LEGACY_DEVELOPMENT300_SEED_MAX,
            ],
            "source_dense260_candidate_seed_range_excluded": [
                SOURCE_DENSE260_CANDIDATE_SEED_MIN,
                SOURCE_DENSE260_CANDIDATE_SEED_MAX,
            ],
            "prior_single_group_seed_excluded": PRIOR_SINGLE_GROUP_SEED,
            "seed_registry_file_read": False,
            "reset_or_scene_state_read": False,
            "exact_resolution_required": True,
        },
        "groups": rows,
        "partition": partition,
        "canonical_nested_oof": _nested_oof_contract(),
        "sample_size_planning_evidence": _sample_size_planning_evidence(),
        "support_gates": _support_contract(),
        "external_disjointness": _disjointness_contract(),
        "prediction_before_label": _prediction_before_label_contract(),
        "stopping_rules": _stopping_contract(),
        "dataset_role_boundary": {
            "adapter_training_groups": 0,
            "adapter_internal_validation_groups": 0,
            "formal190_groups": 0,
            "evaluation400_groups": 0,
            "router_development_groups": TOTAL_GROUPS,
            "may_train_or_select_provider_checkpoint": False,
            "nested_router_fit_requires_separate_authority_after_prediction_commitment": True,
            "preregistration_authorizes_nested_router_fit": False,
            "may_fit_formal_action_selector": False,
            "may_report_evaluation_or_task_success": False,
        },
        "capability": {
            "input_files_accepted": False,
            "target_or_validation_files_read": False,
            "trajectory_files_read": False,
            "hdf5_files_opened": 0,
            "labels_or_outcomes_read": False,
            "environment_reset_authorized": False,
            "policy_query_authorized": False,
            "simulation_execution_authorized": False,
            "real_robot_execution_authorized": False,
            "collection_authorized": False,
            "training_authorized": False,
            "prediction_authorized": False,
            "label_open_authorized": False,
            "calibration_authorized": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "performance_or_transfer_claim_authorized": False,
        },
        "protocol_lineage": {
            "historical_development300_v1_modified": False,
            "historical_development300_v1_artifact_reinterpreted": False,
            "formal190_membership_modified": False,
            "evaluation400_membership_modified": False,
            "new_identity_materializer_and_collection_runner_required": True,
            "cross_body_empirical_claim_authorized": False,
        },
    }
    return {**base, "preregistration_sha256": canonical_sha256(base)}


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RouterDevelopment320PreregistrationError(
            "preregistration must be a mapping"
        )
    document = dict(value)
    signature = document.pop("preregistration_sha256", None)
    if not _is_sha(signature) or signature != canonical_sha256(document):
        raise RouterDevelopment320PreregistrationError(
            "preregistration signature changed"
        )
    seed_generation = document.get("seed_generation")
    if not isinstance(seed_generation, Mapping):
        raise RouterDevelopment320PreregistrationError(
            "seed generation contract is missing"
        )
    seed_base = seed_generation.get("seed_base")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int):
        raise RouterDevelopment320PreregistrationError(
            "seed generation base changed"
        )
    expected = build_preregistration(seed_base)
    if dict(value) != expected:
        raise RouterDevelopment320PreregistrationError(
            "preregistration differs from deterministic contract"
        )
    rows = value["groups"]
    members = value["partition"]["members"][SPLIT]
    requested = [row["requested_seed"] for row in rows]
    semantic_requests = [
        row["semantic_reset_request_sha256"] for row in rows
    ]
    if (
        len(rows) != TOTAL_GROUPS
        or len(members) != TOTAL_GROUPS
        or len(set(members)) != TOTAL_GROUPS
        or members != [row["logical_group_id"] for row in rows]
        or len(set(requested)) != TOTAL_GROUPS
        or len(set(semantic_requests)) != TOTAL_GROUPS
        or set(value["split_counts"]) != {SPLIT}
        or value["split_counts"][SPLIT] != TOTAL_GROUPS
    ):
        raise RouterDevelopment320PreregistrationError(
            "single-split router-development membership changed"
        )
    nested = value["canonical_nested_oof"]
    if (
        nested["outer"]["heldout_groups_per_fold"] * FOLD_COUNT
        != TOTAL_GROUPS
        or nested["outer"]["training_groups_per_fold"]
        != TOTAL_GROUPS - OUTER_HELDOUT_GROUPS
        or sum(
            nested["inner_within_each_outer_training_scope"][
                "heldout_groups_by_fold"
            ]
        )
        != OUTER_TRAINING_GROUPS
        or any(
            train + heldout != OUTER_TRAINING_GROUPS
            for train, heldout in zip(
                nested["inner_within_each_outer_training_scope"][
                    "training_groups_by_fold"
                ],
                nested["inner_within_each_outer_training_scope"][
                    "heldout_groups_by_fold"
                ],
                strict=True,
            )
        )
    ):
        raise RouterDevelopment320PreregistrationError(
            "canonical nested five-fold sizing changed"
        )
    if tuple(
        value["external_disjointness"][
            "required_role_names_in_canonical_order"
        ]
    ) != EXTERNAL_DISJOINTNESS_ROLES:
        raise RouterDevelopment320PreregistrationError(
            "external disjointness roles changed"
        )
    return {
        "status": "verified_label_blind_piper_routerdev320_preregistration",
        "preregistration_sha256": signature,
        "partition_sha256": value["partition"]["partition_sha256"],
        "total_groups": TOTAL_GROUPS,
        "physical_split": SPLIT,
        "candidates_per_group": CANDIDATES_PER_GROUP,
        "outer_heldout_groups_per_fold": OUTER_HELDOUT_GROUPS,
        "outer_training_groups_per_fold": OUTER_TRAINING_GROUPS,
        "minimum_inner_training_groups": min(INNER_TRAINING_GROUPS_BY_FOLD),
        "input_files_read": 0,
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "collection_authorized": False,
        "training_authorized": False,
        "label_open_authorized": False,
        "promotion_authorized": False,
    }


def _output_path(value: Path) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if any(parent.is_symlink() for parent in output.parents):
        raise RouterDevelopment320PreregistrationError(
            "output path contains a symbolic-link parent"
        )
    resolved_parent = output.parent.resolve(strict=True)
    inspected_components = output.parts + resolved_parent.parts
    if any(
        token in component.casefold()
        for component in inspected_components
        for token in SENSITIVE_OUTPUT_TOKENS
    ):
        raise RouterDevelopment320PreregistrationError(
            "output path is in a forbidden sensitive namespace"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    return output


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = _output_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_preregistration(args.seed_base)
    audit = validate_preregistration(document)
    write_json_new(args.output, document)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ACTOR_ID",
    "BODY",
    "CANDIDATES_PER_GROUP",
    "DEFAULT_SEED_BASE",
    "EXTERNAL_DISJOINTNESS_ROLES",
    "FOLD_COUNT",
    "FORMAT",
    "HEADS",
    "INNER_HELDOUT_GROUPS_BY_FOLD",
    "INNER_TRAINING_GROUPS_BY_FOLD",
    "LEGACY_DEVELOPMENT300_SEED_MAX",
    "LEGACY_DEVELOPMENT300_SEED_MIN",
    "MAX_SEED",
    "NAMESPACE",
    "OUTER_HELDOUT_GROUPS",
    "OUTER_TRAINING_GROUPS",
    "PROVIDERS",
    "RouterDevelopment320PreregistrationError",
    "SPLIT",
    "SOURCE_DENSE260_CANDIDATE_SEED_MAX",
    "SOURCE_DENSE260_CANDIDATE_SEED_MIN",
    "TOTAL_GROUPS",
    "build_preregistration",
    "canonical_sha256",
    "validate_preregistration",
    "write_json_new",
]
