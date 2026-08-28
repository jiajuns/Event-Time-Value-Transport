from __future__ import annotations

import ast
import builtins
import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preregister_smolvla_piper_schema6_routerdev320_v1 as prereg  # noqa: E402


def test_default_contract_is_new_piper_only_320_group_single_split() -> None:
    value = prereg.build_preregistration()

    assert value["format"] == prereg.FORMAT
    assert value["namespace"] == prereg.NAMESPACE
    assert value["namespace"] != "schema6_piper_target_development300_20260828_v1"
    assert value["total_groups"] == prereg.TOTAL_GROUPS == 320
    assert value["semantic_reset_cluster_count_after_exact_identity_freeze"] == 320
    assert value["execution_groups_per_semantic_reset_cluster"] == 1
    assert value["candidates_per_group"] == prereg.CANDIDATES_PER_GROUP == 4
    assert value["planned_candidate_accounting_records"] == 1280
    assert value["physical_split"] == prereg.SPLIT
    assert value["split_counts"] == {prereg.SPLIT: 320}

    partition = value["partition"]
    members = partition["members"]
    assert partition["split_counts"] == {prereg.SPLIT: 320}
    assert set(members) == {prereg.SPLIT}
    assert len(members[prereg.SPLIT]) == len(set(members[prereg.SPLIT])) == 320
    assert partition["adapter_training_members_included"] == 0
    assert partition["formal190_members_included"] == 0
    assert partition["evaluation400_members_included"] == 0

    rows = value["groups"]
    assert len(rows) == 320
    assert [row["global_ordinal"] for row in rows] == list(range(320))
    assert [row["split_ordinal"] for row in rows] == list(range(320))
    assert {row["split"] for row in rows} == {prereg.SPLIT}
    assert {row["body"] for row in rows} == {"piper"}
    assert {row["actor_id"] for row in rows} == {prereg.ACTOR_ID}
    assert len({row["requested_seed"] for row in rows}) == 320
    assert all(row["requested_seed"] == row["expected_resolved_seed"] for row in rows)
    assert len({row["semantic_reset_request_sha256"] for row in rows}) == 320
    assert partition["members"][prereg.SPLIT] == [
        row["logical_group_id"] for row in rows
    ]


def test_seed_namespace_is_reserved_and_disjoint_from_legacy_development300() -> None:
    value = prereg.build_preregistration()
    seeds = {row["requested_seed"] for row in value["groups"]}

    assert min(seeds) == prereg.DEFAULT_SEED_BASE
    assert max(seeds) == prereg.DEFAULT_SEED_BASE + 319
    assert max(seeds) < prereg.MAX_SEED
    assert not seeds.intersection(
        range(
            prereg.LEGACY_DEVELOPMENT300_SEED_MIN,
            prereg.LEGACY_DEVELOPMENT300_SEED_MAX + 1,
        )
    )
    assert value["seed_generation"]["legacy_development300_seed_range_excluded"] == [
        prereg.LEGACY_DEVELOPMENT300_SEED_MIN,
        prereg.LEGACY_DEVELOPMENT300_SEED_MAX,
    ]
    assert value["seed_generation"][
        "source_dense260_candidate_seed_range_excluded"
    ] == [
        prereg.SOURCE_DENSE260_CANDIDATE_SEED_MIN,
        prereg.SOURCE_DENSE260_CANDIDATE_SEED_MAX,
    ]
    assert not seeds.intersection(
        range(
            prereg.SOURCE_DENSE260_CANDIDATE_SEED_MIN,
            prereg.SOURCE_DENSE260_CANDIDATE_SEED_MAX + 1,
        )
    )
    assert value["seed_generation"]["seed_registry_file_read"] is False
    assert value["seed_generation"]["reset_or_scene_state_read"] is False


@pytest.mark.parametrize(
    "seed_base",
    [
        True,
        -1,
        prereg.MAX_SEED - prereg.TOTAL_GROUPS + 2,
        prereg.PRIOR_SINGLE_GROUP_SEED,
        prereg.LEGACY_DEVELOPMENT300_SEED_MIN,
        prereg.LEGACY_DEVELOPMENT300_SEED_MAX - prereg.TOTAL_GROUPS + 1,
        prereg.SOURCE_DENSE260_CANDIDATE_SEED_MIN,
        prereg.SOURCE_DENSE260_CANDIDATE_SEED_MAX
        - prereg.TOTAL_GROUPS
        + 1,
    ],
)
def test_invalid_or_overlapping_seed_namespace_fails_closed(seed_base: int) -> None:
    with pytest.raises(
        prereg.RouterDevelopment320PreregistrationError,
        match="seed namespace",
    ):
        prereg.build_preregistration(seed_base)


def test_canonical_nested_five_fold_sizing_is_frozen_for_320_groups() -> None:
    nested = prereg.build_preregistration()["canonical_nested_oof"]

    assert nested["fold_count"] == prereg.FOLD_COUNT == 5
    assert nested["fold_unit"] == "semantic_reset_cluster_id"
    assert nested["assignment_algorithm"] == (
        "sort_unique_semantic_reset_cluster_id_then_zero_based_index_mod_5_v1"
    )
    assert nested["noncanonical_or_caller_selected_fold_plan_accepted"] is False
    assert nested["outer"] == {
        "total_groups": 320,
        "total_semantic_reset_clusters": 320,
        "heldout_groups_per_fold": 64,
        "heldout_semantic_reset_clusters_per_fold": 64,
        "training_groups_per_fold": 256,
        "training_semantic_reset_clusters_per_fold": 256,
        "each_group_and_cluster_heldout_exactly_once": True,
    }
    inner = nested["inner_within_each_outer_training_scope"]
    assert inner["domain_groups"] == inner["domain_semantic_reset_clusters"] == 256
    assert inner["heldout_groups_by_fold"] == [52, 51, 51, 51, 51]
    assert inner["heldout_semantic_reset_clusters_by_fold"] == [52, 51, 51, 51, 51]
    assert inner["training_groups_by_fold"] == [204, 205, 205, 205, 205]
    assert inner["training_semantic_reset_clusters_by_fold"] == [204, 205, 205, 205, 205]
    assert inner["minimum_training_fraction_of_full_routerdev"] == pytest.approx(
        204 / 320
    )
    assert nested["outer_heldout_labels_may_fit_or_select_provider_calibration_or_threshold"] is False
    assert nested["formal190_or_evaluation400_may_enter_nested_oof"] is False


def test_320_sample_size_planning_evidence_is_explicit_and_not_a_guarantee() -> None:
    evidence = prereg.build_preregistration()["sample_size_planning_evidence"]

    assert evidence == {
        "format": "etsf_smolvla_piper_schema6_routerdev320_sample_size_planning_v1",
        "status": "planning_evidence_only_not_support_or_performance_guarantee",
        "public_single_candidate_success_wilson_lower_bound": 0.0981819861,
        "four_candidate_iid_group_success_probability": 0.3385825866,
        "four_candidate_probability_formula": "q=1-(1-p)^4",
        "iid_candidate_assumption_is_verified_by_this_preregistration": False,
        "success_positive_support_target_per_scope": 50,
        "outer_fold_count": 5,
        "inner_folds_per_outer_fold": 5,
        "nested_inner_scope_count_for_union_bound": 25,
        "familywise_failure_probability_budget": 0.05,
        "bonferroni_per_inner_scope_failure_budget": 0.002,
        "minimum_inner_scope_groups_from_binomial_union_bound": 203,
        "union_bound_at_202_inner_groups": 0.0501255964,
        "union_bound_at_203_inner_groups": 0.0433199824,
        "analytical_minimum_full_groups_at_64pct_inner_fraction": 318,
        "engineering_rounded_full_groups": 320,
        "canonical_routerdev320_minimum_inner_training_groups": 204,
        "sample_size_or_iid_assumption_guarantees_real_support": False,
        "post_collection_support_is_always_recomputed_from_real_labels": True,
        "insufficient_real_support_action": "disable_head_no_extension_or_replacement",
    }
    assert evidence["union_bound_at_202_inner_groups"] > 0.05
    assert evidence["union_bound_at_203_inner_groups"] <= 0.05


def test_all_six_head_support_gates_are_per_context_and_fail_closed() -> None:
    support = prereg.build_preregistration()["support_gates"]

    assert support["required_scopes"] == [
        "global",
        "every_body_actor_context",
        "every_outer_training_scope",
        "every_inner_training_scope",
    ]
    assert support["body_actor_context_count_in_this_protocol"] == 1
    assert support["body_actor_context"]["body_id"] == "piper"
    assert support["body_actor_context"]["actor_id"] == prereg.ACTOR_ID
    assert set(support["heads"]) == set(prereg.HEADS)
    assert support["heads"]["post_event"] == {
        "categories": ["e0", "e12", "e3", "e4", "eK"],
        "minimum_per_category": 10,
    }
    assert support["heads"]["next_event"]["categories"] == [
        "e12",
        "e3",
        "e4",
        "eK",
    ]
    assert support["heads"]["duration"]["minimum_per_category"] == 10
    assert support["heads"]["success"]["minimum_per_category"] == 50
    assert support["heads"]["recovery"]["minimum_per_category"] == 10
    assert support["heads"]["object_effect"]["minimum_per_category"] == 50
    assert support["heads"]["recovery"][
        "right_censored_nonrecoveries_are_not_negative"
    ] is True
    assert support["heads"]["success"][
        "unobserved_or_right_censored_outcomes_are_not_negative"
    ] is True
    assert support["all_six_heads_required_for_route_receipt"] is True
    assert "disable_head" in support["insufficient_support_action"]
    assert "no_seed_replacement_or_extension" in support[
        "insufficient_support_action"
    ]
    assert support["sample_count_or_public_success_rate_guarantees_gate_passage"] is False


def test_two_stage_disjointness_covers_training_formal_and_evaluation() -> None:
    value = prereg.build_preregistration()["external_disjointness"]

    assert value["required_role_names_in_canonical_order"] == [
        "provider_training_closure",
        "formal190",
        "evaluation400",
    ]
    assert [row["role"] for row in value["required_external_roles"]] == list(
        prereg.EXTERNAL_DISJOINTNESS_ROLES
    )
    assert value["candidate_pool_stage"] == {
        "target_role": "piper_routerdev320_requested_candidate_pool",
        "required_before_reset_authority": True,
        "required_zero_intersection_axes": [
            "requested_seed",
            "semantic_reset_request_sha256",
        ],
        "identity_or_label_values_from_reference_sets_disclosed": False,
    }
    assert value["selected_resolved_stage"][
        "required_before_collection_authority"
    ] is True
    assert value["selected_resolved_stage"]["required_zero_intersection_axes"] == [
        "requested_seed",
        "resolved_seed",
        "semantic_reset_cluster_id",
        "execution_group_id",
    ]
    assert value["selected_resolved_stage"][
        "same_reference_commitments_as_candidate_stage_required"
    ] is True
    assert value["external_issuer_or_signature_verification_required"] is True
    assert value["self_asserted_boolean_or_self_hash_is_sufficient"] is False
    assert value["preregistration_itself_proves_external_disjointness"] is False


def test_exact_two_by_five_predictions_must_be_frozen_before_labels() -> None:
    boundary = prereg.build_preregistration()["prediction_before_label"]

    assert boundary["canonical_provider_ids"] == [
        "body_agnostic_adapter",
        "body_conditioned_adapter",
    ]
    assert boundary["provider_count"] == 2
    assert boundary["members_per_provider"] == 5
    assert boundary["total_provider_members"] == 10
    assert all(boundary["provider_pair_requirements"].values())
    assert boundary["ordering"] == [
        "freeze_complete_two_by_five_provider_pair",
        "freeze_label_blind_router_input_view_and_sample_order",
        "run_all_ten_checkpoint_forwards",
        "freeze_raw_prediction_tensor_set_and_forward_receipt",
        "authorize_router_target_label_materialization",
        "run_fixed_nested_oof_calibrator",
    ]
    assert boundary[
        "raw_prediction_commitment_required_before_target_label_open"
    ] is True
    assert boundary[
        "prediction_process_accepts_target_label_or_formal_evaluation_path"
    ] is False
    assert boundary["label_dependent_row_filtering_allowed"] is False
    assert boundary[
        "calibrator_implementation_and_nested_fold_plan_frozen_before_label_open"
    ] is True
    assert boundary["rank_route_included"] is False
    assert boundary["rank_action_selection_fallback"] == "actor_baseline"
    assert "not_evaluation_or_deployment_authority" in boundary[
        "routerdev_receipt_scope"
    ]


def test_preregistration_accepts_no_input_files_and_authorizes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("build_preregistration attempted file input")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    value = prereg.build_preregistration()

    assert value["capability"] == {
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
    }
    assert value["dataset_role_boundary"] == {
        "adapter_training_groups": 0,
        "adapter_internal_validation_groups": 0,
        "formal190_groups": 0,
        "evaluation400_groups": 0,
        "router_development_groups": 320,
        "may_train_or_select_provider_checkpoint": False,
        "nested_router_fit_requires_separate_authority_after_prediction_commitment": True,
        "preregistration_authorizes_nested_router_fit": False,
        "may_fit_formal_action_selector": False,
        "may_report_evaluation_or_task_success": False,
    }
    assert value["protocol_lineage"]["historical_development300_v1_modified"] is False
    assert value["protocol_lineage"]["formal190_membership_modified"] is False
    assert value["protocol_lineage"]["cross_body_empirical_claim_authorized"] is False


def test_script_imports_only_python_standard_library_modules() -> None:
    source = (
        SCRIPTS / "preregister_smolvla_piper_schema6_routerdev320_v1.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        str(node.module).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    }
    assert imported_roots <= {
        "argparse",
        "hashlib",
        "json",
        "os",
        "tempfile",
        "pathlib",
        "typing",
    }


def test_build_is_deterministic_and_full_rebuild_validation_passes() -> None:
    first = prereg.build_preregistration()
    second = prereg.build_preregistration()
    assert first == second

    audit = prereg.validate_preregistration(first)
    assert audit == {
        "status": "verified_label_blind_piper_routerdev320_preregistration",
        "preregistration_sha256": first["preregistration_sha256"],
        "partition_sha256": first["partition"]["partition_sha256"],
        "total_groups": 320,
        "physical_split": prereg.SPLIT,
        "candidates_per_group": 4,
        "outer_heldout_groups_per_fold": 64,
        "outer_training_groups_per_fold": 256,
        "minimum_inner_training_groups": 204,
        "input_files_read": 0,
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "collection_authorized": False,
        "training_authorized": False,
        "label_open_authorized": False,
        "promotion_authorized": False,
    }


def test_signature_and_resigned_contract_tampering_both_fail_closed() -> None:
    changed = prereg.build_preregistration()
    changed["groups"][0]["split"] = "formal_target_validation"
    with pytest.raises(
        prereg.RouterDevelopment320PreregistrationError,
        match="signature changed",
    ):
        prereg.validate_preregistration(changed)

    resigned = prereg.build_preregistration()
    resigned["support_gates"]["heads"]["success"]["minimum_per_category"] = 1
    unsigned = dict(resigned)
    unsigned.pop("preregistration_sha256")
    resigned["preregistration_sha256"] = prereg.canonical_sha256(unsigned)
    with pytest.raises(
        prereg.RouterDevelopment320PreregistrationError,
        match="differs from deterministic contract",
    ):
        prereg.validate_preregistration(resigned)


def test_create_once_read_only_json_round_trip(tmp_path: Path) -> None:
    value = prereg.build_preregistration()
    output = tmp_path / "piper_routerdev320_preregistration_v1.json"
    prereg.write_json_new(output, value)

    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        prereg.write_json_new(output, value)


@pytest.mark.parametrize(
    "namespace",
    ["fresh", "confirmation", "formal", "evaluation", "adapter_train", "adapter-train"],
)
def test_sensitive_output_namespaces_are_rejected(
    tmp_path: Path, namespace: str
) -> None:
    forbidden = tmp_path / namespace
    forbidden.mkdir()
    with pytest.raises(
        prereg.RouterDevelopment320PreregistrationError,
        match="forbidden sensitive namespace",
    ):
        prereg.write_json_new(
            forbidden / "piper_routerdev320_preregistration_v1.json",
            prereg.build_preregistration(),
        )


def test_symbolic_link_parent_cannot_bypass_sensitive_namespace(
    tmp_path: Path,
) -> None:
    sensitive = tmp_path / "formal-store"
    sensitive.mkdir()
    alias = tmp_path / "safe-alias"
    alias.symlink_to(sensitive, target_is_directory=True)

    with pytest.raises(
        prereg.RouterDevelopment320PreregistrationError,
        match="symbolic-link parent",
    ):
        prereg.write_json_new(
            alias / "piper_routerdev320_preregistration_v1.json",
            prereg.build_preregistration(),
        )
