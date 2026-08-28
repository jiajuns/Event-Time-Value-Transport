#!/usr/bin/env python3
"""Immutable, label-blind preregistration for 300 Piper development groups.

This CPU-only entry point accepts no input file.  It deterministically commits
300 requested seed candidates, a physical 80/30/190 split, support quotas, and
non-adaptive stopping rules before any reset, trajectory, HDF5, or label can be
read.  It does not modify or claim compatibility with the frozen v2 protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


FORMAT = "etsf_smolvla_piper_schema6_target_development300_preregistration_v1"
STATUS = "preregistered_label_blind_seed_candidates_collection_not_authorized"
PARTITION_FORMAT = "etsf_smolvla_piper_schema6_target_development300_partition_v1"
TASK = "move_can_pot"
BODY = "piper"
POLICY = "smolvla"
ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
INSTRUCTION = "move the can into the pot"
NAMESPACE = "schema6_piper_target_development300_20260828_v1"
DEFAULT_SEED_BASE = 2_026_082_800
TOTAL_GROUPS = 300
CANDIDATES_PER_GROUP = 4
SPLIT_COUNTS = {
    "adaptation_train": 80,
    "adaptation_internal_validation": 30,
    "formal_target_validation": 190,
}
ADAPTATION_BUCKET_GROUPS = 110
FORMAL_TARGET_VALIDATION_GROUPS = 190
PRIOR_SINGLE_GROUP_SEED = 100_101_000
MAX_SEED = 2**31 - 1
EVENTS = ("e0", "e12", "e3", "e4", "eK")
NEXT_EVENTS = ("e12", "e3", "e4", "eK")
SENSITIVE_OUTPUT_TOKENS = (
    "fresh",
    "confirmation",
    "evaluation",
)
SHA_CHARS = frozenset("0123456789abcdef")


class Development300PreregistrationError(RuntimeError):
    """The immutable label-blind preregistration contract is invalid."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _seed_order(seed_base: int) -> list[int]:
    if (
        isinstance(seed_base, bool)
        or not isinstance(seed_base, int)
        or seed_base < 0
        or seed_base + TOTAL_GROUPS - 1 > MAX_SEED
        or seed_base <= PRIOR_SINGLE_GROUP_SEED <= seed_base + TOTAL_GROUPS - 1
    ):
        raise Development300PreregistrationError("seed namespace is invalid or overlaps the prior group")
    candidates = list(range(seed_base, seed_base + TOTAL_GROUPS))
    return sorted(
        candidates,
        key=lambda seed: hashlib.sha256(f"{NAMESPACE}:{seed}".encode("ascii")).hexdigest(),
    )


def _split_for_ordinal(global_ordinal: int) -> tuple[str, int]:
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        if global_ordinal < offset + count:
            return split, global_ordinal - offset
        offset += count
    raise Development300PreregistrationError("global ordinal escaped split counts")


def _support_contract() -> dict[str, Any]:
    trainer_events = {
        "post_event_minimum_rows_per_class": {name: 5 for name in EVENTS},
        "observed_next_event_minimum_rows_per_class": {
            name: 5 for name in NEXT_EVENTS
        },
    }
    return {
        "format": "etsf_schema6_target_development300_support_quota_v1",
        "quota_semantics": "post_collection_activation_gates_not_seed_selection_targets",
        "adaptation_train": {
            "groups": 80,
            "outcome_independent_groups": {
                "positive": 5,
                "negative": 5,
                "candidate_outcome_discordant": 5,
            },
            "events": trainer_events,
            "duration_rows": {"observed": 5, "censored": 5},
            "conditional_recovery_independent_groups": {
                "positive": 10,
                "negative": 10,
                "right_censored_nonrecoveries_count_as_negative": False,
            },
        },
        "adaptation_internal_validation": {
            "groups": 30,
            "outcome_independent_groups": {
                "positive": 5,
                "negative": 5,
                "candidate_outcome_discordant": 5,
            },
            "events": trainer_events,
            "duration_rows": {"observed": 5, "censored": 5},
            "conditional_recovery_independent_groups": {
                "positive": 10,
                "negative": 10,
                "right_censored_nonrecoveries_count_as_negative": False,
            },
        },
        "formal_target_validation": {
            "groups": 190,
            "success_independent_groups_per_side": {
                "positive": 50,
                "negative": 50,
            },
            "event_minimum_independent_groups_per_class": {
                "post_event": {name: 10 for name in EVENTS},
                "observed_next_event": {name: 10 for name in NEXT_EVENTS},
            },
            "duration_minimum_independent_groups_per_side": {
                "observed": 10,
                "censored": 10,
            },
            "conditional_recovery": {
                "versioned_calibrator_v2_supports_recovery": True,
                "independent_lane_minimum_groups_per_class": 10,
                "all_five_recovery_heads_must_be_trained": True,
                "right_censored_nonrecoveries_count_as_negative": False,
                "activation_under_this_contract": False,
            },
            "minimum_groups_retained_for_abstention_lcb": 50,
        },
        "quota_values_may_be_audited_only_after_partition_and_collection_freeze": True,
        "quota_values_must_not_change_membership_or_seed_order": True,
    }


def _stopping_contract() -> dict[str, Any]:
    return {
        "format": "etsf_schema6_target_development300_stopping_rules_v1",
        "collection": {
            "stop_after_exactly_all_preregistered_groups_terminal": TOTAL_GROUPS,
            "early_stop_on_success_event_duration_or_recovery_quota": False,
            "replace_failed_or_unresolved_seed_inside_this_contract": False,
            "requested_seed_must_resolve_bit_exact_or_group_fails_closed": True,
            "all_four_candidate_branches_required": True,
        },
        "adaptation_support_audit": {
            "authorized_only_after_all_300_memberships_and_artifacts_are_frozen": True,
            "may_open_only_adaptation_train_and_internal_validation_groups": True,
            "formal_target_validation_opened": False,
            "insufficient_head_action": "disable_head_and_publish_insufficient_support_receipt",
        },
        "formal_target_validation_support_audit": {
            "label_open_authorized_by_this_preregistration": False,
            "requires_separate_authority_after_five_frozen_adapters": True,
            "membership_reassignment_after_open": False,
            "use_for_training_or_checkpoint_selection": False,
            "insufficient_head_action": "disable_head_no_reassignment_no_posthoc_seed_addition",
        },
        "extension": {
            "automatic_extension_authorized": False,
            "new_groups_require_new_create_once_preregistration": True,
            "existing_group_split_may_change": False,
            "individual_group_labels_may_select_future_seed_or_split": False,
        },
    }


def build_preregistration(seed_base: int = DEFAULT_SEED_BASE) -> dict[str, Any]:
    ordered_seeds = _seed_order(seed_base)
    rows: list[dict[str, Any]] = []
    split_ids = {name: [] for name in SPLIT_COUNTS}
    split_seeds = {name: [] for name in SPLIT_COUNTS}
    for global_ordinal, requested_seed in enumerate(ordered_seeds):
        split, split_ordinal = _split_for_ordinal(global_ordinal)
        identity_base: dict[str, Any] = {
            "namespace": NAMESPACE,
            "task": TASK,
            "body": BODY,
            "policy": POLICY,
            "actor_id": ACTOR_ID,
            "split": split,
            "global_ordinal": global_ordinal,
            "split_ordinal": split_ordinal,
            "requested_seed": requested_seed,
            "expected_resolved_seed": requested_seed,
        }
        identity_sha = canonical_sha256(identity_base)
        row = {
            **identity_base,
            "logical_group_id": f"{NAMESPACE}/{identity_sha}",
            "identity_sha256": identity_sha,
            "resolution_status": "unresolved_candidate_exact_resolution_required_before_policy_query",
        }
        rows.append(row)
        split_ids[split].append(row["logical_group_id"])
        split_seeds[split].append(requested_seed)
    partition_base: dict[str, Any] = {
        "format": PARTITION_FORMAT,
        "status": "frozen_label_blind_disjoint_membership",
        "namespace": NAMESPACE,
        "split_counts": dict(SPLIT_COUNTS),
        "adaptation_bucket_groups": ADAPTATION_BUCKET_GROUPS,
        "formal_target_validation_groups": FORMAL_TARGET_VALIDATION_GROUPS,
        "members": split_ids,
        "requested_seeds": split_seeds,
        "all_members_unique": True,
        "all_requested_seeds_unique": True,
        "full_coverage_without_overlap": True,
        "evaluation400_members_included": 0,
    }
    partition = {**partition_base, "partition_sha256": canonical_sha256(partition_base)}
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "namespace": NAMESPACE,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "actor_id": ACTOR_ID,
        "instruction": INSTRUCTION,
        "instruction_sha256": hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest(),
        "total_groups": TOTAL_GROUPS,
        "candidates_per_group": CANDIDATES_PER_GROUP,
        "planned_candidate_branches": TOTAL_GROUPS * CANDIDATES_PER_GROUP,
        "split_counts": dict(SPLIT_COUNTS),
        "adaptation_bucket": {
            "total": ADAPTATION_BUCKET_GROUPS,
            "train": SPLIT_COUNTS["adaptation_train"],
            "internal_validation": SPLIT_COUNTS[
                "adaptation_internal_validation"
            ],
        },
        "formal_target_validation_groups": FORMAL_TARGET_VALIDATION_GROUPS,
        "seed_generation": {
            "algorithm": "sha256_order_of_contiguous_reserved_numeric_namespace_v1",
            "seed_base": seed_base,
            "candidate_count": TOTAL_GROUPS,
            "maximum_seed": seed_base + TOTAL_GROUPS - 1,
            "prior_single_group_seed_excluded": PRIOR_SINGLE_GROUP_SEED,
            "seed_registry_file_read": False,
            "reset_or_scene_state_read": False,
            "exact_resolution_required": True,
        },
        "groups": rows,
        "partition": partition,
        "support_quotas": _support_contract(),
        "stopping_rules": _stopping_contract(),
        "evaluation_boundary": {
            "evaluation400_group_count_in_development300": 0,
            "evaluation400_identity_or_membership_read": False,
            "evaluation400_trajectory_or_label_read": False,
            "evaluation400_collection_authorized": False,
            "production_collection_requires_external_label_blind_disjointness_receipt": True,
        },
        "capability": {
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
        },
        "protocol_lineage": {
            "frozen_v2_protocol_modified": False,
            "frozen_v2_compatibility_claimed": False,
            "new_collection_and_materialization_contract_required": True,
        },
    }
    return {**base, "preregistration_sha256": canonical_sha256(base)}


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Development300PreregistrationError("preregistration must be a mapping")
    document = dict(value)
    signature = document.pop("preregistration_sha256", None)
    if not _is_sha(signature) or signature != canonical_sha256(document):
        raise Development300PreregistrationError("preregistration signature changed")
    seed_generation = document.get("seed_generation")
    if not isinstance(seed_generation, Mapping):
        raise Development300PreregistrationError("seed generation contract is missing")
    expected = build_preregistration(int(seed_generation.get("seed_base", -1)))
    if dict(value) != expected:
        raise Development300PreregistrationError("preregistration differs from deterministic contract")
    rows = value["groups"]
    partitions = value["partition"]["members"]
    all_partition_ids = [identity for split in SPLIT_COUNTS for identity in partitions[split]]
    if (
        len(rows) != TOTAL_GROUPS
        or len(all_partition_ids) != TOTAL_GROUPS
        or len(set(all_partition_ids)) != TOTAL_GROUPS
        or {row["logical_group_id"] for row in rows} != set(all_partition_ids)
        or len({row["requested_seed"] for row in rows}) != TOTAL_GROUPS
    ):
        raise Development300PreregistrationError("partition is not disjoint full coverage")
    return {
        "status": "verified_immutable_label_blind_development300_preregistration",
        "preregistration_sha256": signature,
        "partition_sha256": value["partition"]["partition_sha256"],
        "total_groups": TOTAL_GROUPS,
        "candidates_per_group": CANDIDATES_PER_GROUP,
        "split_counts": dict(SPLIT_COUNTS),
        "adaptation_bucket_groups": ADAPTATION_BUCKET_GROUPS,
        "formal_target_validation_groups": FORMAL_TARGET_VALIDATION_GROUPS,
        "input_files_read": 0,
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "collection_authorized": False,
    }


def _output_path(value: Path) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if any(
        token in component.casefold()
        for component in output.parts
        for token in SENSITIVE_OUTPUT_TOKENS
    ):
        raise Development300PreregistrationError("output path is in a forbidden namespace")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.resolve(strict=True)
    return output


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = _output_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
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
