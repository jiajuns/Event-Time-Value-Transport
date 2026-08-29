from __future__ import annotations

import ast
import copy
import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as prereg  # noqa: E402


def test_official_move_can_pot_slice_is_exact_and_content_addressed() -> None:
    value = prereg.build_preregistration()
    source = value["official_source_slice"]
    files = source["files"]

    assert source["hf_repo_id"] == "TianxingChen/RoboTwin2.0"
    assert source["hf_repo_type"] == "dataset"
    assert source["hf_repo_revision"] == prereg.HF_REPO_REVISION
    assert source["task_path"] == "dataset/move_can_pot"
    assert len(files) == 11
    assert source["body_condition_archive_count"] == 10
    assert source["demo_clean_archive_count"] == 1
    assert sum(row["size_bytes"] for row in files) == 21_238_835_871
    assert source["total_size_bytes"] == 21_238_835_871
    assert source["total_size_decimal_gb"] == pytest.approx(21.238835871)
    assert source["tree_entry_count"] == 11
    assert source["tree_has_unexpected_entries"] is False

    assert [row["path"] for row in files] == list(prereg.OFFICIAL_FILE_PATHS)
    assert len({row["path"] for row in files}) == 11
    assert len({row["lfs_sha256"] for row in files}) == 11
    assert all(len(row["lfs_sha256"]) == 64 for row in files)
    assert all(len(row["git_blob_oid"]) == 40 for row in files)
    assert all(len(row["xet_hash"]) == 64 for row in files)
    assert all(row["size_bytes"] > 0 for row in files)


def test_exact_five_body_two_condition_inventory_and_demo_boundary() -> None:
    value = prereg.build_preregistration()
    files = value["official_source_slice"]["files"]
    body_rows = [row for row in files if row["role"] == "body_condition_expert_archive"]
    demo_rows = [row for row in files if row["role"] == "demo_clean_reference_archive"]

    assert value["bodies"] == list(prereg.BODIES)
    assert value["source_conditions"] == ["clean_50", "randomized_500"]
    assert len(body_rows) == 10
    assert len(demo_rows) == 1
    assert demo_rows[0]["path"].endswith("/demo_clean.zip")
    assert demo_rows[0]["body"] is None
    assert demo_rows[0]["condition"] == "demo_clean"
    assert demo_rows[0]["lobo_training_or_evaluation_role"] == "excluded_reference_only"
    assert {
        (row["body"], row["condition"])
        for row in body_rows
    } == {
        (body, condition)
        for body in prereg.BODIES
        for condition in prereg.SOURCE_CONDITIONS
    }
    assert {row["declared_episode_count"] for row in body_rows} == {50, 500}
    assert all(row["archive_content_count_verified_without_download"] is False for row in body_rows)


def test_lobo_folds_hold_out_each_body_exactly_once() -> None:
    value = prereg.build_preregistration()
    folds = value["lobo_protocol"]["folds"]

    assert len(folds) == 5
    assert [fold["fold_index"] for fold in folds] == list(range(5))
    assert [fold["heldout_body"] for fold in folds] == list(prereg.BODIES)
    for fold in folds:
        heldout = fold["heldout_body"]
        assert fold["training_bodies"] == [body for body in prereg.BODIES if body != heldout]
        assert len(fold["training_archive_paths"]) == 8
        assert len(fold["heldout_archive_paths"]) == 2
        assert all(f"/{heldout}_" not in path for path in fold["training_archive_paths"])
        assert all(f"/{heldout}_" in path for path in fold["heldout_archive_paths"])
        assert fold["demo_clean_used_for_training_selection_or_evaluation"] is False
        assert fold["heldout_body_data_used_for_training_adapter_calibration_or_selection"] is False


def test_every_fold_uses_same_100_eval_seeds_and_frozen_condition_order() -> None:
    value = prereg.build_preregistration()
    evaluation = value["paired_evaluation_protocol"]
    expected_seeds = list(range(prereg.EVAL_SEED_BASE, prereg.EVAL_SEED_BASE + 100))

    assert evaluation["evaluation_seed_count"] == 100
    assert evaluation["evaluation_seeds"] == expected_seeds
    assert evaluation["condition_order"] == ["clean", "randomized"]
    assert evaluation["heldout_body_order"] == list(prereg.BODIES)
    assert evaluation["same_seed_list_for_every_body_and_condition"] is True
    assert evaluation["same_resolved_reset_for_both_methods"] is True
    assert evaluation["paired_trial_count"] == 1_000
    assert evaluation["planned_rollout_count"] == 2_000
    assert len(evaluation["canonical_condition_seed_schedule"]) == 200
    assert evaluation["canonical_condition_seed_schedule"][0] == {
        "condition_ordinal": 0,
        "condition": "clean",
        "seed_ordinal": 0,
        "requested_seed": prereg.EVAL_SEED_BASE,
        "method_order": ["actor_baseline", "etsf_best_of_4"],
    }
    assert evaluation["canonical_condition_seed_schedule"][100]["condition"] == "randomized"
    assert evaluation["canonical_condition_seed_schedule"][1]["method_order"] == [
        "etsf_best_of_4",
        "actor_baseline",
    ]


def test_actor_baseline_and_etsf_best_of_n_are_strictly_paired() -> None:
    pairing = prereg.build_preregistration()["paired_evaluation_protocol"]["method_pairing"]

    assert pairing["methods"] == ["actor_baseline", "etsf_best_of_4"]
    assert pairing["best_of_n"] == 4
    assert pairing["same_actor_checkpoint"] is True
    assert pairing["same_ordered_candidate_set"] is True
    assert pairing["same_candidate_set_sha256_required"] is True
    assert pairing["actor_baseline_action"] == "candidate_index_0_before_etsf_scoring"
    assert pairing["etsf_action"] == "argmax_frozen_etsf_score_over_same_four_candidates"
    assert pairing["extra_environment_queries_or_rollout_lookahead_allowed"] is False
    assert pairing["method_specific_reset_retry_or_seed_replacement_allowed"] is False
    assert pairing["policy_or_etsf_checkpoint_may_change_after_first_outcome"] is False


def test_primary_metrics_freeze_sr_delta_ci_mcnemar_and_stage_progress() -> None:
    metrics = prereg.build_preregistration()["metrics_protocol"]
    primary = metrics["primary_full_task_success"]
    progress = metrics["supporting_stage_progress"]

    assert primary["outcome"] == "official_simulator_full_task_success_boolean"
    assert primary["paired_key"] == ["heldout_body", "condition", "requested_seed"]
    assert primary["reported_statistics"] == [
        "actor_baseline_sr",
        "etsf_best_of_4_sr",
        "paired_delta_sr",
        "paired_delta_sr_95pct_ci",
        "exact_two_sided_mcnemar_p",
        "discordant_actor_only_success_count",
        "discordant_etsf_only_success_count",
    ]
    assert primary["sr_interval"] == "wilson_score_95pct"
    assert primary["delta_interval"]["method"] == "paired_seed_cluster_percentile_bootstrap"
    assert primary["delta_interval"]["bootstrap_samples"] == 20_000
    assert primary["mcnemar"]["method"] == "exact_two_sided_binomial_on_discordant_pairs"
    assert primary["stage_progress_or_critic_metric_may_replace_failed_full_task_sr"] is False
    assert progress["events"] == ["e0", "e12", "e3", "e4", "eK"]
    assert progress["terminal_max_event_progress"] == {
        "e0": 0.0,
        "e12": 0.25,
        "e3": 0.5,
        "e4": 0.75,
        "eK": 1.0,
    }
    assert progress["role"] == "supporting_endpoint_not_primary_success_substitute"


def test_critic_diagnostics_are_secondary_and_cannot_select_or_rescue() -> None:
    diagnostics = prereg.build_preregistration()["metrics_protocol"]["critic_diagnostics"]

    assert diagnostics["role"] == "secondary_diagnostics_only"
    assert diagnostics["computed_after_all_paired_outcomes_are_frozen"] is True
    assert diagnostics["may_select_checkpoint_threshold_candidate_n_or_route"] is False
    assert diagnostics["may_rescue_failed_primary_success_gate"] is False
    assert diagnostics["metrics"] == [
        "success_brier",
        "success_nll",
        "success_ece",
        "success_auroc_if_both_classes_observed",
        "uncertainty_aurc",
        "post_event_accuracy",
        "next_event_accuracy_on_observed_rows",
        "duration_mae_on_observed_rows",
        "object_effect_error_on_observed_rows",
    ]


def test_public_expert_archives_are_not_reinterpreted_as_failure_supervision() -> None:
    boundary = prereg.build_preregistration()["public_expert_supervision_boundary"]

    assert boundary["public_archives_are_expert_demonstrations"] is True
    assert boundary["archive_contains_verified_failure_labels"] is False
    assert boundary["missing_completion_or_unobserved_tail_is_failure"] is False
    assert boundary["all_unlabelled_rows_are_negative"] is False
    assert boundary["success_failure_critic_may_be_trained_from_this_slice_alone"] is False
    assert boundary["positive_only_or_unknown_outcome_examples_prove_failure_discrimination"] is False
    assert boundary["download_and_content_audit_required_before_any_supervision_use"] is True


def test_preregistration_has_no_operational_authority_or_empirical_claim() -> None:
    value = prereg.build_preregistration()
    capability = value["capability"]

    assert capability == {
        "input_files_accepted": False,
        "network_or_hf_api_access_performed": False,
        "archive_download_authorized": False,
        "archive_downloaded": False,
        "archive_or_pickle_payload_opened": False,
        "training_authorized": False,
        "simulator_reset_authorized": False,
        "policy_query_authorized": False,
        "evaluation_authorized": False,
        "outcomes_read": False,
        "metrics_computed": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
        "cross_embodiment_improvement_claim_authorized": False,
    }
    assert value["status"] == prereg.STATUS
    assert value["empirical_result"] is None


def test_validation_rebuilds_exact_contract_and_rejects_tampering() -> None:
    value = prereg.build_preregistration()
    audit = prereg.validate_preregistration(value)
    assert audit["official_file_count"] == 11
    assert audit["official_total_size_bytes"] == 21_238_835_871
    assert audit["fold_count"] == 5
    assert audit["evaluation_seed_count"] == 100
    assert audit["download_authorized"] is False
    assert audit["evaluation_authorized"] is False

    changed = copy.deepcopy(value)
    changed["official_source_slice"]["files"][0]["size_bytes"] += 1
    unsigned = {key: child for key, child in changed.items() if key != "preregistration_sha256"}
    changed["preregistration_sha256"] = prereg.canonical_sha256(unsigned)
    with pytest.raises(prereg.CrossEmbodimentSlicePreregistrationError):
        prereg.validate_preregistration(changed)


def test_create_once_output_is_read_only_and_existing_path_fails(tmp_path: Path) -> None:
    output = tmp_path / "robotwin2_lobo_prereg.json"
    value = prereg.build_preregistration()

    prereg.write_json_new(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        prereg.write_json_new(output, value)


def test_symlink_output_and_symlink_parent_fail_closed(tmp_path: Path) -> None:
    value = prereg.build_preregistration()
    target = tmp_path / "target.json"
    target.write_text("untouched", encoding="utf-8")
    output = tmp_path / "alias.json"
    output.symlink_to(target)
    with pytest.raises((FileExistsError, prereg.CrossEmbodimentSlicePreregistrationError)):
        prereg.write_json_new(output, value)
    assert target.read_text(encoding="utf-8") == "untouched"

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(prereg.CrossEmbodimentSlicePreregistrationError, match="symbolic-link"):
        prereg.write_json_new(linked_parent / "prereg.json", value)


def test_script_is_standard_library_only_and_has_no_network_or_data_reader() -> None:
    source_path = SCRIPTS / "preregister_robotwin2_move_can_pot_five_body_lobo_v1.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imports <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "os",
        "tempfile",
        "pathlib",
        "typing",
    }
    source = source_path.read_text(encoding="utf-8")
    assert "requests" not in source
    assert "huggingface_hub" not in source
    assert "urlopen(" not in source
    assert "torch.load" not in source
    assert "pickle.load" not in source
    assert "zipfile" not in source
