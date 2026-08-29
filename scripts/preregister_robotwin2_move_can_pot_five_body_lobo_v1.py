#!/usr/bin/env python3
"""Freeze a data-blind five-embodiment RoboTwin2 move_can_pot LOBO study.

The standard-library-only command accepts no dataset, checkpoint, trajectory,
outcome, or metric input.  It records a content-addressed official Hugging Face
slice, five leave-one-body-out folds, and a prospective paired evaluation plan.
It neither contacts Hugging Face nor downloads or opens any archive.  The
create-once JSON grants no training, simulator, evaluation, promotion, or
cross-embodiment performance-claim authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


FORMAT = "etsf_robotwin2_move_can_pot_five_body_lobo_preregistration_v1"
STATUS = "preregistered_data_blind_no_download_training_evaluation_or_claim"
SOURCE_FORMAT = "etsf_robotwin2_move_can_pot_official_hf_slice_v1"
LOBO_FORMAT = "etsf_robotwin2_move_can_pot_five_fold_lobo_v1"
EVALUATION_FORMAT = "etsf_robotwin2_move_can_pot_paired_crossbody_eval_v1"
METRICS_FORMAT = "etsf_robotwin2_move_can_pot_crossbody_metrics_v1"

HF_REPO_ID = "TianxingChen/RoboTwin2.0"
HF_REPO_TYPE = "dataset"
HF_REPO_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
HF_TASK_PATH = "dataset/move_can_pot"
TASK = "move_can_pot"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
SOURCE_CONDITIONS = ("clean_50", "randomized_500")
EVALUATION_CONDITIONS = ("clean", "randomized")
METHODS = ("actor_baseline", "etsf_best_of_4")
BEST_OF_N = 4
EVAL_SEED_BASE = 2_026_090_000
EVAL_SEED_COUNT = 100
BOOTSTRAP_SEED = 2_026_090_200
BOOTSTRAP_SAMPLES = 20_000
EXPECTED_TOTAL_SIZE_BYTES = 21_238_835_871
SHA256_CHARS = frozenset("0123456789abcdef")
SHA1_CHARS = SHA256_CHARS

# Metadata was checked against the official Hub tree endpoint at the immutable
# revision above on 2026-08-30.  ``lfs_sha256`` is the LFS object OID returned
# by the official API; no archive payload was downloaded to create this file.
OFFICIAL_FILES = (
    (
        "dataset/move_can_pot/aloha-agilex_clean_50.zip",
        312_581_958,
        "62b3f5a1fcad7ea5ed8be2c467436e0b02411b77",
        "fbf5231c5be71405364b09ed718cfca1e07a6509f6d1a801f5922516da0ade09",
        "cdfadfb4596651e96d450e767abda1db646bcbd86a33663c729996fd988049ad",
    ),
    (
        "dataset/move_can_pot/aloha-agilex_randomized_500.zip",
        5_561_470_057,
        "34ce3d81f68aa73c4f863a49d98511db5562efb6",
        "761ab72a186941be79082f04c05298753b3ffe3c9fd92bc0c3169d84293d75f6",
        "9c29c14ae1e48d8affd0224e690c6597611141d21e81261704a08aaf5a480927",
    ),
    (
        "dataset/move_can_pot/arx-x5_clean_50.zip",
        218_811_989,
        "eafe155183ecc8ec3052dbcf732c05e6f7c1bc09",
        "8af739367ce74d9b982fc37cd712d6e7674de8d26571df40db3bd7d23f50acae",
        "41dddf73119bf525936c6fc3cf77a7cfa819cf5c9d6e00b6071dd41af3fc6825",
    ),
    (
        "dataset/move_can_pot/arx-x5_randomized_500.zip",
        3_653_570_164,
        "386b51d5bf5f50aa300734cc2ebd6795e7a7e753",
        "716429c14998d86865745333f405f1ddff950400fa3c5734cb1120a803320b59",
        "35d94f338122188b84266f85d52d07e8c1542b7fcfa2baa47a9b769d01df18e3",
    ),
    (
        "dataset/move_can_pot/demo_clean.zip",
        306_859_681,
        "a6773d7642db8084fa00db62fb0be8bb2ba063f0",
        "dddb150282a009fcb0f2c2e97276269747a00358c6800c67d991f8d4c5d2c0e7",
        "43a51dd8312832c4ab915734e92a4ec39fae70db5e1f3e635de7bc74ffeda01c",
    ),
    (
        "dataset/move_can_pot/franka_clean_50.zip",
        183_374_364,
        "6c5c53041e8c86fd0373960c185c5f4103c0e064",
        "2ab00e0b65e5bd9c2fdfd5d5355c3990237070f7b1c58c12f0702cfbc0dd4082",
        "83e44b2facb505d297f06f641363eb19039eb6f6499da71440d50880977c4957",
    ),
    (
        "dataset/move_can_pot/franka_randomized_500.zip",
        3_328_177_600,
        "da96b833fd2fdd9bd652496f6637513a80d2d52d",
        "2bd0dc2d4893326d6d1095565f3370e371160cf503988ef333e4233a43677a82",
        "748e629c0745150e58074eb46d7f7effcde29c9a667138626495bf9877bec10f",
    ),
    (
        "dataset/move_can_pot/piper_clean_50.zip",
        212_427_528,
        "c0d69b4f3e649c8b5d22b5a743776e280766b975",
        "e902b72d8111065080b0432cf2750b9754e8c84f16909376ecb7c85c0fcd0d2f",
        "0c0f2461c993f1eb27e065f14bb821dd30535e98bab7b5e37b4f1b6d3cbc314f",
    ),
    (
        "dataset/move_can_pot/piper_randomized_500.zip",
        3_896_574_298,
        "a4e603750ae70ee5f5fa2c1c76fa3afbc61da00c",
        "3a3b5ae9dc748c2cda1fc143f1eff5141a1b57defdc052790c6480762f91599d",
        "6836a19b1a793455d0004b49e888004a0d57426319758032d162eb45df24216a",
    ),
    (
        "dataset/move_can_pot/ur5_clean_50.zip",
        183_214_565,
        "93f3cbab8fb8511fced5b611a92a4bb3f5ef9c52",
        "400f1c826d264c1d2e04dab34df2fa7681a5e640b37a21209ac8dd0ec6bc36e4",
        "118e3a422eda99e7f2d8e0a7278d222ce1139e494e7842649af21dfab0467103",
    ),
    (
        "dataset/move_can_pot/ur5_randomized_500.zip",
        3_381_773_667,
        "2a556c2296301a809c2a0c48c77ca5aa8cf1480a",
        "7f60dea19c46b94e3ca50ce9a349e33f1a93d0ad8bbba887fd046669178619dd",
        "62b474ac307ff69a83aa833678afb53e270488415cc1b4cc9de880c408aad34f",
    ),
)
OFFICIAL_FILE_PATHS = tuple(row[0] for row in OFFICIAL_FILES)


class CrossEmbodimentSlicePreregistrationError(RuntimeError):
    """The immutable five-body preregistration is malformed or changed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_CHARS


def _is_git_oid(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= SHA1_CHARS


def _body_condition(path: str) -> tuple[str | None, str, int | None]:
    name = path.rsplit("/", 1)[-1]
    if name == "demo_clean.zip":
        return None, "demo_clean", None
    for body in BODIES:
        for condition, count in (("clean_50", 50), ("randomized_500", 500)):
            if name == f"{body}_{condition}.zip":
                return body, condition, count
    raise CrossEmbodimentSlicePreregistrationError(
        "official file does not belong to the frozen five-body slice"
    )


def _official_file_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, size, git_oid, lfs_sha, xet_hash in OFFICIAL_FILES:
        body, condition, count = _body_condition(path)
        demo = body is None
        rows.append(
            {
                "path": path,
                "type": "file",
                "size_bytes": size,
                "git_blob_oid": git_oid,
                "lfs_sha256": lfs_sha,
                "xet_hash": xet_hash,
                "role": (
                    "demo_clean_reference_archive"
                    if demo
                    else "body_condition_expert_archive"
                ),
                "body": body,
                "condition": condition,
                "declared_episode_count": count,
                "archive_content_count_verified_without_download": False,
                "lobo_training_or_evaluation_role": (
                    "excluded_reference_only"
                    if demo
                    else "prospective_fold_membership_only"
                ),
            }
        )
    return rows


def _official_source_slice() -> dict[str, Any]:
    files = _official_file_rows()
    total = sum(row["size_bytes"] for row in files)
    body_files = [row for row in files if row["body"] is not None]
    demo_files = [row for row in files if row["body"] is None]
    if (
        [row["path"] for row in files] != list(OFFICIAL_FILE_PATHS)
        or len(files) != 11
        or len(body_files) != 10
        or len(demo_files) != 1
        or total != EXPECTED_TOTAL_SIZE_BYTES
        or len({row["path"] for row in files}) != len(files)
        or len({row["lfs_sha256"] for row in files}) != len(files)
        or any(
            not _is_sha256(row["lfs_sha256"])
            or not _is_sha256(row["xet_hash"])
            or not _is_git_oid(row["git_blob_oid"])
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] <= 0
            for row in files
        )
    ):
        raise CrossEmbodimentSlicePreregistrationError(
            "official source metadata does not match the frozen slice"
        )
    return {
        "format": SOURCE_FORMAT,
        "status": "official_metadata_frozen_payloads_not_downloaded_or_opened",
        "hf_repo_id": HF_REPO_ID,
        "hf_repo_type": HF_REPO_TYPE,
        "hf_repo_revision": HF_REPO_REVISION,
        "task_path": HF_TASK_PATH,
        "official_tree_api_url": (
            "https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/"
            f"{HF_REPO_REVISION}/dataset/move_can_pot?recursive=true&expand=true"
        ),
        "metadata_checked_utc_date": "2026-08-30",
        "tree_entry_count": len(files),
        "tree_has_unexpected_entries": False,
        "body_condition_archive_count": len(body_files),
        "demo_clean_archive_count": len(demo_files),
        "total_size_bytes": total,
        "total_size_decimal_gb": total / 1_000_000_000,
        "integrity_identity": "immutable_revision_plus_path_plus_lfs_sha256_plus_size",
        "lfs_sha256_is_archive_payload_sha256": True,
        "files": files,
        "payload_bytes_downloaded_or_opened_by_preregistration": 0,
        "archive_member_counts_verified": False,
        "metadata_does_not_prove_supervision_or_outcome_semantics": True,
    }


def _lobo_protocol(files: list[dict[str, Any]]) -> dict[str, Any]:
    body_files = [row for row in files if row["body"] is not None]
    demo_path = next(row["path"] for row in files if row["body"] is None)
    folds: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(BODIES):
        training_bodies = [body for body in BODIES if body != heldout]
        training_paths = [
            row["path"] for row in body_files if row["body"] in training_bodies
        ]
        heldout_paths = [row["path"] for row in body_files if row["body"] == heldout]
        if len(training_paths) != 8 or len(heldout_paths) != 2:
            raise CrossEmbodimentSlicePreregistrationError(
                "LOBO fold does not contain the exact four-body/eight-archive training side"
            )
        folds.append(
            {
                "fold_index": fold_index,
                "heldout_body": heldout,
                "training_bodies": training_bodies,
                "training_archive_paths": training_paths,
                "training_archive_set_sha256": canonical_sha256(training_paths),
                "heldout_archive_paths": heldout_paths,
                "heldout_archive_set_sha256": canonical_sha256(heldout_paths),
                "demo_clean_path": demo_path,
                "demo_clean_used_for_training_selection_or_evaluation": False,
                "heldout_body_data_used_for_training_adapter_calibration_or_selection": False,
                "heldout_public_expert_archives_may_replace_paired_simulator_evaluation": False,
                "heldout_body_identity_must_remain_unseen_until_training_closure": True,
                "fold_training_or_evaluation_authorized": False,
            }
        )
    return {
        "format": LOBO_FORMAT,
        "fold_count": 5,
        "fold_unit": "embodiment_body_id",
        "assignment": "canonical_body_order_each_body_heldout_exactly_once",
        "canonical_body_order": list(BODIES),
        "source_conditions_per_body": list(SOURCE_CONDITIONS),
        "training_body_count_per_fold": 4,
        "training_archive_count_per_fold": 8,
        "heldout_body_count_per_fold": 1,
        "heldout_archive_count_per_fold": 2,
        "folds": folds,
        "same_archives_may_appear_as_training_in_one_fold_and_heldout_in_another": True,
        "cross_fold_checkpoint_or_metric_reuse_allowed": False,
        "demo_clean_global_training_calibration_or_evaluation_use_allowed": False,
    }


def _condition_seed_schedule() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for condition_ordinal, condition in enumerate(EVALUATION_CONDITIONS):
        for seed_ordinal in range(EVAL_SEED_COUNT):
            result.append(
                {
                    "condition_ordinal": condition_ordinal,
                    "condition": condition,
                    "seed_ordinal": seed_ordinal,
                    "requested_seed": EVAL_SEED_BASE + seed_ordinal,
                    "method_order": (
                        list(METHODS)
                        if seed_ordinal % 2 == 0
                        else list(reversed(METHODS))
                    ),
                }
            )
    return result


def _paired_evaluation_protocol() -> dict[str, Any]:
    seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_SEED_COUNT))
    schedule = _condition_seed_schedule()
    return {
        "format": EVALUATION_FORMAT,
        "status": "prospective_paired_protocol_only_execution_not_authorized",
        "heldout_body_order": list(BODIES),
        "condition_order": list(EVALUATION_CONDITIONS),
        "condition_order_may_change_after_outcome_open": False,
        "evaluation_seed_base": EVAL_SEED_BASE,
        "evaluation_seed_count": EVAL_SEED_COUNT,
        "evaluation_seeds": seeds,
        "evaluation_seed_order_sha256": canonical_sha256(seeds),
        "same_seed_list_for_every_body_and_condition": True,
        "requested_seed_must_resolve_exactly": True,
        "same_resolved_reset_for_both_methods": True,
        "seed_replacement_retry_or_extension_after_outcome_allowed": False,
        "canonical_condition_seed_schedule": schedule,
        "canonical_condition_seed_schedule_sha256": canonical_sha256(schedule),
        "method_pairing": {
            "methods": list(METHODS),
            "best_of_n": BEST_OF_N,
            "same_actor_checkpoint": True,
            "same_actor_observation_and_instruction_contract": True,
            "same_ordered_candidate_set": True,
            "same_candidate_set_sha256_required": True,
            "candidate_proposal_randomness_frozen_before_either_action_executes": True,
            "actor_baseline_action": "candidate_index_0_before_etsf_scoring",
            "etsf_action": "argmax_frozen_etsf_score_over_same_four_candidates",
            "etsf_tie_break": "lowest_candidate_index",
            "extra_environment_queries_or_rollout_lookahead_allowed": False,
            "method_specific_reset_retry_or_seed_replacement_allowed": False,
            "policy_or_etsf_checkpoint_may_change_after_first_outcome": False,
            "actor_or_etsf_may_use_official_expert_action_at_evaluation": False,
        },
        "paired_key": ["heldout_body", "condition", "requested_seed"],
        "paired_trial_count": len(BODIES) * len(EVALUATION_CONDITIONS) * len(seeds),
        "planned_rollout_count": (
            len(BODIES) * len(EVALUATION_CONDITIONS) * len(seeds) * len(METHODS)
        ),
        "body_simulator_actor_etsf_event_spec_and_runtime_sha256_required_before_execution": True,
        "all_checkpoints_candidate_n_ties_thresholds_and_metrics_frozen_before_first_outcome": True,
        "failed_or_crashed_pair_action": "retain_pair_as_protocol_failure_no_retry_or_replacement",
        "execution_authorized": False,
    }


def _metrics_protocol() -> dict[str, Any]:
    return {
        "format": METRICS_FORMAT,
        "status": "metrics_preregistered_no_outcomes_read_or_computed",
        "reporting_levels_in_order": [
            "each_heldout_body_by_condition",
            "each_heldout_body_equal_condition_macro",
            "each_condition_equal_body_macro",
            "global_equal_body_condition_macro",
        ],
        "primary_full_task_success": {
            "outcome": "official_simulator_full_task_success_boolean",
            "task_checker_implementation_file_sha256_required_before_execution": True,
            "critic_score_or_stage_progress_is_not_success_ground_truth": True,
            "paired_key": ["heldout_body", "condition", "requested_seed"],
            "reported_statistics": [
                "actor_baseline_sr",
                "etsf_best_of_4_sr",
                "paired_delta_sr",
                "paired_delta_sr_95pct_ci",
                "exact_two_sided_mcnemar_p",
                "discordant_actor_only_success_count",
                "discordant_etsf_only_success_count",
            ],
            "sr_interval": "wilson_score_95pct",
            "delta_interval": {
                "method": "paired_seed_cluster_percentile_bootstrap",
                "cluster_unit": (
                    "requested_seed_with_all_heldout_body_condition_pairs_kept_together"
                ),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "quantiles": [0.025, 0.975],
                "within_cell_pairing_preserved": True,
            },
            "mcnemar": {
                "method": "exact_two_sided_binomial_on_discordant_pairs",
                "zero_discordant_pair_p_value": 1.0,
                "continuity_corrected_asymptotic_test_used": False,
            },
            "prospective_improvement_gate": {
                "global_macro_paired_delta_sr_lcb95_strictly_positive": True,
                "global_exact_mcnemar_p_below": 0.05,
                "each_heldout_body_macro_delta_sr_point_estimate_nonnegative": True,
                "each_condition_macro_delta_sr_point_estimate_nonnegative": True,
                "gate_itself_authorizes_claim_or_deployment": False,
            },
            "stage_progress_or_critic_metric_may_replace_failed_full_task_sr": False,
        },
        "supporting_stage_progress": {
            "role": "supporting_endpoint_not_primary_success_substitute",
            "events": ["e0", "e12", "e3", "e4", "eK"],
            "terminal_max_event_progress": {
                "e0": 0.0,
                "e12": 0.25,
                "e3": 0.5,
                "e4": 0.75,
                "eK": 1.0,
            },
            "event_spec_file_sha256_required_before_execution": True,
            "reported_statistics": [
                "actor_mean_terminal_max_event_progress",
                "etsf_mean_terminal_max_event_progress",
                "paired_progress_delta",
                "paired_progress_delta_95pct_ci",
                "per_stage_reach_rate_by_method",
            ],
            "same_paired_seed_cluster_bootstrap_as_primary": True,
        },
        "critic_diagnostics": {
            "role": "secondary_diagnostics_only",
            "computed_after_all_paired_outcomes_are_frozen": True,
            "may_select_checkpoint_threshold_candidate_n_or_route": False,
            "may_rescue_failed_primary_success_gate": False,
            "metrics": [
                "success_brier",
                "success_nll",
                "success_ece",
                "success_auroc_if_both_classes_observed",
                "uncertainty_aurc",
                "post_event_accuracy",
                "next_event_accuracy_on_observed_rows",
                "duration_mae_on_observed_rows",
                "object_effect_error_on_observed_rows",
            ],
            "undefined_single_class_metrics_reported_as_null_not_zero_or_one": True,
            "diagnostic_missingness_and_applicability_masks_reported": True,
        },
        "multiple_comparison_note": (
            "only_global_macro_full_task_delta_gate_is_primary; body_condition_and_critic_rows_are_supporting"
        ),
    }


def _public_expert_supervision_boundary() -> dict[str, Any]:
    return {
        "public_archives_are_expert_demonstrations": True,
        "archive_contains_verified_failure_labels": False,
        "missing_completion_or_unobserved_tail_is_failure": False,
        "all_unlabelled_rows_are_negative": False,
        "success_failure_critic_may_be_trained_from_this_slice_alone": False,
        "positive_only_or_unknown_outcome_examples_prove_failure_discrimination": False,
        "official_data_generation_success_rate_is_not_per_demo_failure_label": True,
        "filename_declared_episode_count_is_not_payload_count_audit": True,
        "download_and_content_audit_required_before_any_supervision_use": True,
        "per_archive_pickle_or_numpy_payload_must_not_be_opened_by_this_preregistration": True,
        "prospective_allowed_role_after_separate_authority": (
            "training_body_imitation_and_causal_event_representation_only_with_observed_fields_and_masks"
        ),
        "heldout_body_archive_role": "sealed_from_fold_training_and_selection",
        "demo_clean_role": "inventory_reference_only_excluded_from_all_lobo_folds",
    }


def build_preregistration() -> dict[str, Any]:
    source = _official_source_slice()
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "task": TASK,
        "study_scope": "five_embodiment_leave_one_body_out_crossbody_task_success",
        "bodies": list(BODIES),
        "source_conditions": list(SOURCE_CONDITIONS),
        "official_source_slice": source,
        "lobo_protocol": _lobo_protocol(source["files"]),
        "paired_evaluation_protocol": _paired_evaluation_protocol(),
        "metrics_protocol": _metrics_protocol(),
        "public_expert_supervision_boundary": _public_expert_supervision_boundary(),
        "required_future_authorities": [
            "local_archive_materialization_and_payload_audit",
            "per_fold_training_identity_and_checkpoint_freeze",
            "heldout_body_simulator_and_actor_contracts",
            "etsf_best_of_4_checkpoint_and_scoring_contract",
            "canonical_move_can_pot_event_specification",
            "paired_evaluation_execution_authority",
            "outcome_materialization_and_metrics_verifier",
        ],
        "stopping_and_change_control": {
            "all_five_folds_and_all_1000_pairs_required": True,
            "early_stop_on_success_failure_progress_or_critic_metric_allowed": False,
            "post_outcome_body_condition_seed_method_n_metric_or_gate_change_allowed": False,
            "failed_seed_or_body_replacement_allowed": False,
            "new_or_corrected_protocol_requires_new_create_once_preregistration": True,
        },
        "capability": {
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
        },
        "empirical_result": None,
    }
    return {**base, "preregistration_sha256": canonical_sha256(base)}


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CrossEmbodimentSlicePreregistrationError(
            "preregistration must be a mapping"
        )
    document = dict(value)
    digest = document.pop("preregistration_sha256", None)
    if not _is_sha256(digest) or digest != canonical_sha256(document):
        raise CrossEmbodimentSlicePreregistrationError(
            "preregistration canonical SHA changed"
        )
    expected = build_preregistration()
    if dict(value) != expected:
        raise CrossEmbodimentSlicePreregistrationError(
            "preregistration differs from the deterministic reviewed contract"
        )
    source = value["official_source_slice"]
    files = source["files"]
    folds = value["lobo_protocol"]["folds"]
    evaluation = value["paired_evaluation_protocol"]
    if (
        len(files) != 11
        or sum(row["size_bytes"] for row in files) != EXPECTED_TOTAL_SIZE_BYTES
        or len(folds) != len(BODIES)
        or {fold["heldout_body"] for fold in folds} != set(BODIES)
        or len(evaluation["evaluation_seeds"]) != EVAL_SEED_COUNT
        or len(set(evaluation["evaluation_seeds"])) != EVAL_SEED_COUNT
        or evaluation["paired_trial_count"] != 1_000
        or evaluation["planned_rollout_count"] != 2_000
    ):
        raise CrossEmbodimentSlicePreregistrationError(
            "slice, LOBO, or paired evaluation cardinality changed"
        )
    return {
        "status": "verified_data_blind_robotwin2_move_can_pot_five_body_lobo_preregistration",
        "preregistration_sha256": digest,
        "hf_repo_revision": HF_REPO_REVISION,
        "official_file_count": len(files),
        "official_total_size_bytes": EXPECTED_TOTAL_SIZE_BYTES,
        "fold_count": len(folds),
        "evaluation_seed_count": EVAL_SEED_COUNT,
        "paired_trial_count": evaluation["paired_trial_count"],
        "planned_rollout_count": evaluation["planned_rollout_count"],
        "input_files_read": 0,
        "archive_payloads_opened": 0,
        "download_authorized": False,
        "training_authorized": False,
        "evaluation_authorized": False,
        "cross_embodiment_claim_authorized": False,
    }


def _output_path(value: Path) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if any(parent.is_symlink() for parent in output.parents):
        raise CrossEmbodimentSlicePreregistrationError(
            "output path contains a symbolic-link parent"
        )
    output.parent.resolve(strict=True)
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
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_preregistration()
    audit = validate_preregistration(document)
    write_json_new(args.output, document)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BEST_OF_N",
    "BODIES",
    "CrossEmbodimentSlicePreregistrationError",
    "EVAL_SEED_BASE",
    "EVAL_SEED_COUNT",
    "EVALUATION_CONDITIONS",
    "FORMAT",
    "HF_REPO_REVISION",
    "OFFICIAL_FILE_PATHS",
    "SOURCE_CONDITIONS",
    "STATUS",
    "build_preregistration",
    "canonical_sha256",
    "validate_preregistration",
    "write_json_new",
]
