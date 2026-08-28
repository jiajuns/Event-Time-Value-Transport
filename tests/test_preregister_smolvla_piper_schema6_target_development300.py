from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from preregister_smolvla_piper_schema6_target_development300 import (  # noqa: E402
    ADAPTATION_BUCKET_GROUPS,
    CANDIDATES_PER_GROUP,
    DEFAULT_SEED_BASE,
    FORMAL_TARGET_VALIDATION_GROUPS,
    MAX_SEED,
    PRIOR_SINGLE_GROUP_SEED,
    SPLIT_COUNTS,
    TOTAL_GROUPS,
    Development300PreregistrationError,
    build_preregistration,
    validate_preregistration,
    write_json_new,
)


def test_default_partition_is_label_blind_disjoint_80_30_190() -> None:
    preregistration = build_preregistration()

    assert preregistration["total_groups"] == 300
    assert preregistration["candidates_per_group"] == CANDIDATES_PER_GROUP == 4
    assert preregistration["planned_candidate_branches"] == 1200
    assert preregistration["split_counts"] == {
        "adaptation_train": 80,
        "adaptation_internal_validation": 30,
        "formal_target_validation": 190,
    }
    assert preregistration["adaptation_bucket"] == {
        "total": 110,
        "train": 80,
        "internal_validation": 30,
    }
    assert ADAPTATION_BUCKET_GROUPS == 110
    assert FORMAL_TARGET_VALIDATION_GROUPS == 190
    assert sum(SPLIT_COUNTS.values()) == TOTAL_GROUPS

    members = preregistration["partition"]["members"]
    member_sets = {name: set(values) for name, values in members.items()}
    assert {name: len(values) for name, values in member_sets.items()} == SPLIT_COUNTS
    assert member_sets["adaptation_train"].isdisjoint(
        member_sets["adaptation_internal_validation"]
    )
    assert member_sets["adaptation_train"].isdisjoint(
        member_sets["formal_target_validation"]
    )
    assert member_sets["adaptation_internal_validation"].isdisjoint(
        member_sets["formal_target_validation"]
    )
    assert len(set().union(*member_sets.values())) == TOTAL_GROUPS

    seeds = [row["requested_seed"] for row in preregistration["groups"]]
    assert len(seeds) == len(set(seeds)) == TOTAL_GROUPS
    assert PRIOR_SINGLE_GROUP_SEED not in seeds
    assert preregistration["partition"]["evaluation400_members_included"] == 0
    assert preregistration["evaluation_boundary"][
        "evaluation400_group_count_in_development300"
    ] == 0


def test_build_is_deterministic_and_signatures_validate() -> None:
    first = build_preregistration()
    second = build_preregistration()
    assert first == second

    audit = validate_preregistration(first)
    assert audit == {
        "status": "verified_immutable_label_blind_development300_preregistration",
        "preregistration_sha256": first["preregistration_sha256"],
        "partition_sha256": first["partition"]["partition_sha256"],
        "total_groups": 300,
        "candidates_per_group": 4,
        "split_counts": dict(SPLIT_COUNTS),
        "adaptation_bucket_groups": 110,
        "formal_target_validation_groups": 190,
        "input_files_read": 0,
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "collection_authorized": False,
    }


def test_support_quotas_are_activation_gates_not_membership_or_early_stop() -> None:
    preregistration = build_preregistration()
    quotas = preregistration["support_quotas"]
    stopping = preregistration["stopping_rules"]

    assert quotas["quota_semantics"] == (
        "post_collection_activation_gates_not_seed_selection_targets"
    )
    assert quotas["quota_values_may_be_audited_only_after_partition_and_collection_freeze"]
    assert quotas["quota_values_must_not_change_membership_or_seed_order"]
    assert quotas["formal_target_validation"][
        "success_independent_groups_per_side"
    ] == {"positive": 50, "negative": 50}
    assert quotas["formal_target_validation"]["conditional_recovery"] == {
        "versioned_calibrator_v2_supports_recovery": True,
        "independent_lane_minimum_groups_per_class": 10,
        "all_five_recovery_heads_must_be_trained": True,
        "right_censored_nonrecoveries_count_as_negative": False,
        "activation_under_this_contract": False,
    }

    assert stopping["collection"][
        "stop_after_exactly_all_preregistered_groups_terminal"
    ] == 300
    assert stopping["collection"][
        "early_stop_on_success_event_duration_or_recovery_quota"
    ] is False
    assert stopping["adaptation_support_audit"][
        "authorized_only_after_all_300_memberships_and_artifacts_are_frozen"
    ] is True
    assert stopping["formal_target_validation_support_audit"][
        "label_open_authorized_by_this_preregistration"
    ] is False
    assert stopping["formal_target_validation_support_audit"][
        "use_for_training_or_checkpoint_selection"
    ] is False
    assert stopping["extension"]["automatic_extension_authorized"] is False


def test_contract_accepts_no_inputs_and_authorizes_no_data_or_execution() -> None:
    preregistration = build_preregistration()
    assert preregistration["capability"] == {
        "input_files_accepted": False,
        "target_or_validation_files_read": False,
        "trajectory_files_read": False,
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "environment_reset": False,
        "policy_query": False,
        "simulation_execution_authorized": False,
        "real_robot_execution_authorized": False,
        "performance_or_transfer_claim_authorized": False,
    }
    assert preregistration["seed_generation"]["seed_registry_file_read"] is False
    assert preregistration["seed_generation"]["reset_or_scene_state_read"] is False
    assert preregistration["protocol_lineage"] == {
        "frozen_v2_protocol_modified": False,
        "frozen_v2_compatibility_claimed": False,
        "new_collection_and_materialization_contract_required": True,
    }


def test_tampering_membership_or_signature_fails_closed() -> None:
    tampered = build_preregistration()
    tampered["groups"][0]["split"] = "formal_target_validation"
    with pytest.raises(
        Development300PreregistrationError,
        match="signature changed",
    ):
        validate_preregistration(tampered)

    tampered = build_preregistration()
    tampered["preregistration_sha256"] = "0" * 64
    with pytest.raises(
        Development300PreregistrationError,
        match="signature changed",
    ):
        validate_preregistration(tampered)


@pytest.mark.parametrize(
    "seed_base",
    [True, -1, MAX_SEED - TOTAL_GROUPS + 2, PRIOR_SINGLE_GROUP_SEED],
)
def test_invalid_or_overlapping_seed_namespace_fails_closed(seed_base: int) -> None:
    with pytest.raises(Development300PreregistrationError, match="seed namespace"):
        build_preregistration(seed_base)


def test_receipt_is_create_once_read_only_and_round_trips(tmp_path: Path) -> None:
    preregistration = build_preregistration(DEFAULT_SEED_BASE)
    output = tmp_path / "schema6_development300_preregistration.json"
    write_json_new(output, preregistration)

    assert json.loads(output.read_text(encoding="utf-8")) == preregistration
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        write_json_new(output, preregistration)


def test_sensitive_output_namespace_is_rejected(tmp_path: Path) -> None:
    forbidden_parent = tmp_path / "fresh"
    forbidden_parent.mkdir()
    with pytest.raises(
        Development300PreregistrationError,
        match="forbidden namespace",
    ):
        write_json_new(
            forbidden_parent / "schema6_development300_preregistration.json",
            build_preregistration(),
        )
