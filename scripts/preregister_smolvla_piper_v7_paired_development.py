#!/usr/bin/env python3
"""Freeze a label-blind SmolVLA(Aloha-trained)->Piper paired protocol.

The freezer authenticates identity/provenance artifacts only.  It has no
simulator, actor, HDF5, reward, success, event-label, or execution interface.
Fresh/confirmation paths are rejected.  A frozen protocol is an experiment
contract, never execution authorization or evidence of transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_smolvla_piper_v7_paired_development_preregistration_v1"
STATUS = "preregistered_before_target_outcomes_execution_not_authorized"
TARGET_SEED_FORMAT = "etsf_smolvla_piper_paired_development_seed_manifest_v1"
TARGET_SEED_STATUS = "resolved_reset_identity_only_before_policy_execution"
TASK = "move_can_pot"
SOURCE_BODY = "aloha"
TARGET_BODY = "piper_piper_0.6"
TARGET_ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
ADAPTATION_GROUPS = 80
VALIDATION_GROUPS = 50
EVALUATION_GROUPS = 400
TOTAL_GROUPS = ADAPTATION_GROUPS + VALIDATION_GROUPS + EVALUATION_GROUPS
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_SAMPLES = 20_000
MINIMUM_POLICY_CHANGES = 40
HARMFUL_RATE_MAX = 0.10
PRIMARY_ALPHA = 0.05
CONDITION_ORDER_SEED = 20261003
EXPECTED_D250_CANDIDATES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)
TARGET_CANDIDATES = (
    "deterministic",
    "flow_noise_001",
    "flow_noise_002",
    "flow_noise_003",
)
LABEL_ACCESS_CONTRACT = (
    "reset_identity_instruction_and_initial_state_hash_only_no_policy_action_"
    "reward_success_event_or_outcome"
)
FORBIDDEN_LABEL_KEYS = {
    "success",
    "successes",
    "reward",
    "rewards",
    "events",
    "event_id",
    "event_names",
    "event_steps",
    "pre_event_id",
    "post_event_id",
    "next_event_id",
    "next_event_duration_observed",
    "candidate_successes",
    "candidate_success_rates",
    "groups_with_outcome_variation",
    "dense_label_counts",
    "outcome",
    "outcomes",
    "steps",
    "duration",
    "recovery",
    "selected_index",
    "prediction",
    "predictions",
}


class ProtocolError(ValueError):
    """Raised for a fail-closed preregistration violation."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_condition_order(pair_id: str) -> tuple[str, str]:
    """Return the immutable within-pair execution order without labels."""

    if not _is_sha(pair_id):
        raise ProtocolError("pair_id must be a lowercase SHA256")
    digest = hashlib.sha256(f"{pair_id}||{CONDITION_ORDER_SEED}".encode("ascii")).digest()
    return (
        ("direct_smolvla_policy", "v7_event_world_model_selector")
        if digest[-1] & 1 == 0
        else ("v7_event_world_model_selector", "direct_smolvla_policy")
    )


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _reject_sensitive_path(path: str | Path, role: str, *, must_exist: bool = True) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ProtocolError(f"{role} path must be absolute")
    resolved = raw.resolve(strict=False)
    if any(
        token in part.casefold()
        for part in (*raw.parts, *resolved.parts)
        for token in ("fresh", "confirmation")
    ):
        raise ProtocolError(f"{role} must not reference Fresh/confirmation")
    if must_exist and not resolved.is_file():
        raise ProtocolError(f"{role} is not an existing file")
    return resolved


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{role} must contain a JSON object")
    return value


def _signed(value: Mapping[str, Any], key: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise ProtocolError(f"{role} signature mismatch")
    return str(recorded)


def _forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            sensitive_components = {"success", "successes", "reward", "rewards", "outcome", "outcomes", "recovery", "prediction", "predictions"}
            if key in FORBIDDEN_LABEL_KEYS or sensitive_components & set(key.split("_")):
                found.append(path)
            found.extend(_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def _decode_seed_rows(rows: Any, *, split: str, expected: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != expected:
        raise ProtocolError(f"target seed split {split} must contain exactly {expected} rows")
    result: list[dict[str, Any]] = []
    expected_fields = {
        "ordinal",
        "pair_id",
        "requested_seed",
        "resolved_seed",
        "instruction_sha256",
        "instruction_semantics_receipt_sha256",
        "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256",
    }
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ProtocolError(f"target seed {split}[{ordinal}] fields changed")
        requested = row["requested_seed"]
        resolved = row["resolved_seed"]
        if (
            isinstance(requested, bool)
            or isinstance(resolved, bool)
            or not isinstance(requested, int)
            or not isinstance(resolved, int)
            or requested < 0
            or resolved < 0
            or isinstance(row["ordinal"], bool)
            or not isinstance(row["ordinal"], int)
            or row["ordinal"] != ordinal
            or not _is_sha(row["instruction_sha256"])
            or not _is_sha(row["instruction_semantics_receipt_sha256"])
            or not _is_sha(row["initial_scene_state_sha256"])
            or not _is_sha(row["initial_measured_joint_state_sha256"])
            or not _is_sha(row["initial_commanded_drive_target_sha256"])
        ):
            raise ProtocolError(f"target seed {split}[{ordinal}] identity is invalid")
        identity = {
            "task": TASK,
            "actor_id": TARGET_ACTOR_ID,
            "target_body": TARGET_BODY,
            "split": split,
            "ordinal": ordinal,
            "requested_seed": requested,
            "resolved_seed": resolved,
            "instruction_sha256": row["instruction_sha256"],
            "instruction_semantics_receipt_sha256": row[
                "instruction_semantics_receipt_sha256"
            ],
            "initial_scene_state_sha256": row["initial_scene_state_sha256"],
            "initial_measured_joint_state_sha256": row[
                "initial_measured_joint_state_sha256"
            ],
            "initial_commanded_drive_target_sha256": row[
                "initial_commanded_drive_target_sha256"
            ],
        }
        if row["pair_id"] != canonical_sha256(identity):
            raise ProtocolError(f"target seed {split}[{ordinal}] pair_id mismatch")
        result.append(dict(row))
    return result


def validate_d250_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the collector's explicitly label-free D250 identity view."""

    forbidden = _forbidden_keys(value)
    if forbidden:
        raise ProtocolError(f"D250 identity contains forbidden label keys: {forbidden[:3]}")
    if (
        value.get("format") != "etsf_event_branch_collection_identity_v1"
        or value.get("label_access_contract")
        != "identity_only_no_success_steps_event_or_outcome_fields"
        or int(value.get("schema_version", -1)) != 5
        or value.get("task") != TASK
        or value.get("body") != TARGET_BODY
        or int(value.get("candidate_count", -1)) != 4
        or int(value.get("completed", -1)) != 250
        or value.get("seed_registry") != "explicit_v7_prospective_development"
    ):
        raise ProtocolError("D250 identity scope/schema contract changed")
    if value.get("fresh_seed_manifest") not in (None, "") or value.get(
        "fresh_seed_manifest_sha256"
    ) not in (None, ""):
        raise ProtocolError("D250 identity unexpectedly binds a Fresh collection")
    if not _is_sha(value.get("event_spec_sha256")) or not _is_sha(
        value.get("v7_seed_manifest_sha256")
    ) or not _is_sha(value.get("v7_preregistration_sha256")):
        raise ProtocolError("D250 event/V7 SHA binding is incomplete")
    groups = value.get("groups")
    if not isinstance(groups, list) or len(groups) != 250:
        raise ProtocolError("D250 identity must contain exactly 250 groups")
    requested: list[int] = []
    resolved: list[int] = []
    for index, row in enumerate(groups):
        if not isinstance(row, Mapping):
            raise ProtocolError("D250 group identity row is invalid")
        if (
            isinstance(row.get("index"), bool)
            or not isinstance(row.get("index"), int)
            or row.get("index") != index
            or row.get("status") not in {"collected", "existing"}
            or list(row.get("candidate_names", ())) != list(EXPECTED_D250_CANDIDATES)
        ):
            raise ProtocolError(f"D250 group {index} candidate/identity contract changed")
        try:
            raw_requested = row["requested_seed"]
            raw_resolved = row["resolved_seed"]
            if (
                isinstance(raw_requested, bool)
                or isinstance(raw_resolved, bool)
                or not isinstance(raw_requested, int)
                or not isinstance(raw_resolved, int)
                or min(raw_requested, raw_resolved) < 0
            ):
                raise TypeError("seed identities must be non-negative integers")
            requested.append(raw_requested)
            resolved.append(raw_resolved)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"D250 group {index} seed identity is invalid") from exc
    if (
        len(set(requested)) != 250
        or len(set(resolved)) != 250
        or list(map(int, value.get("requested_seeds", ()))) != requested
        or list(map(int, value.get("resolved_seeds", ()))) != resolved
    ):
        raise ProtocolError("D250 requested/resolved seed mirrors changed")
    return {
        "task": TASK,
        "body": TARGET_BODY,
        "schema_version": 5,
        "groups": 250,
        "candidate_names": list(EXPECTED_D250_CANDIDATES),
        "event_spec_sha256": value["event_spec_sha256"],
        "v7_seed_manifest_sha256": value["v7_seed_manifest_sha256"],
        "v7_preregistration_sha256": value["v7_preregistration_sha256"],
        "identity_sets_sha256": canonical_sha256(
            {"requested": requested, "resolved": resolved}
        ),
        "requested": requested,
        "resolved": resolved,
        "labels_read": False,
    }


def validate_target_seed_manifest(
    value: Mapping[str, Any], *, d250_identity_file_sha256: str, d250: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate 530 reset-only target groups and their non-disclosure attestation."""

    _signed(value, "seed_manifest_sha256", "target seed manifest")
    forbidden = _forbidden_keys(value)
    if forbidden:
        raise ProtocolError(f"target seed manifest contains label keys: {forbidden[:3]}")
    expected_root = {
        "format",
        "status",
        "task",
        "actor_id",
        "source_body",
        "target_body",
        "purpose",
        "label_access_contract",
        "splits",
        "d250_exclusion",
        "heldout_exclusion_attestation",
        "instruction_contract",
        "seed_manifest_sha256",
    }
    if set(value) != expected_root:
        raise ProtocolError("target seed manifest fields changed")
    if (
        value["format"] != TARGET_SEED_FORMAT
        or value["status"] != TARGET_SEED_STATUS
        or value["task"] != TASK
        or value["actor_id"] != TARGET_ACTOR_ID
        or value["source_body"] != SOURCE_BODY
        or value["target_body"] != TARGET_BODY
        or value["purpose"] != "nonfresh_development_only_no_confirmation_claim"
        or value["label_access_contract"] != LABEL_ACCESS_CONTRACT
        or value["instruction_contract"]
        != {
            "mode": "explicit_frozen_per_pair_instruction",
            "episode_info_list_used": False,
            "semantic_receipt_required": True,
            "same_instruction_for_both_conditions": True,
        }
    ):
        raise ProtocolError("target seed manifest semantic contract changed")
    raw_splits = value["splits"]
    expected_counts = {
        "adaptation": ADAPTATION_GROUPS,
        "validation": VALIDATION_GROUPS,
        "evaluation": EVALUATION_GROUPS,
    }
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(expected_counts):
        raise ProtocolError("target seed splits changed")
    splits = {
        name: _decode_seed_rows(raw_splits[name], split=name, expected=count)
        for name, count in expected_counts.items()
    }
    all_rows = [row for name in ("adaptation", "validation", "evaluation") for row in splits[name]]
    requested = [int(row["requested_seed"]) for row in all_rows]
    resolved = [int(row["resolved_seed"]) for row in all_rows]
    pair_ids = [str(row["pair_id"]) for row in all_rows]
    if len(set(requested)) != TOTAL_GROUPS or len(set(resolved)) != TOTAL_GROUPS or len(set(pair_ids)) != TOTAL_GROUPS:
        raise ProtocolError("target seed identities must be unique across every split")
    if (set(requested) | set(resolved)) & (set(d250["requested"]) | set(d250["resolved"])):
        raise ProtocolError("target target seeds overlap source D250 identities")
    target_identity_sha = canonical_sha256({"requested": requested, "resolved": resolved})
    exclusion = value["d250_exclusion"]
    expected_exclusion = {
        "identity_manifest_file_sha256": d250_identity_file_sha256,
        "identity_sets_sha256": d250["identity_sets_sha256"],
        "intersection_count": 0,
    }
    if exclusion != expected_exclusion:
        raise ProtocolError("target seed D250 exclusion binding changed")
    attestation = value["heldout_exclusion_attestation"]
    if not isinstance(attestation, Mapping) or attestation != {
        "status": "verified_disjoint_without_disclosing_heldout_identities",
        "heldout_identity_set_sha256": attestation.get("heldout_identity_set_sha256"),
        "target_identity_set_sha256": target_identity_sha,
        "intersection_count": 0,
        "sensitive_identities_included": False,
    } or not _is_sha(attestation.get("heldout_identity_set_sha256")):
        raise ProtocolError("heldout exclusion attestation is invalid")
    return {
        "splits": splits,
        "requested": requested,
        "resolved": resolved,
        "target_identity_sets_sha256": target_identity_sha,
        "heldout_identity_set_sha256": attestation["heldout_identity_set_sha256"],
        "seed_manifest_sha256": value["seed_manifest_sha256"],
        "labels_read": False,
    }


def validate_forward_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("format") != "smolvla_piper_zero_shot_preflight_v2"
        or value.get("status") != "passed_forward_only"
        or value.get("actor_id") != TARGET_ACTOR_ID
        or value.get("authorization") != "forward_only"
        or value.get("environment_execution_authorized") is not False
        or value.get("transfer_claim_authorized") is not False
        or value.get("data_blind") is not True
        or not _is_sha(value.get("implementation_sha256"))
    ):
        raise ProtocolError("SmolVLA->Piper preflight is not the data-blind forward-only receipt")
    candidate = value.get("candidate_validation")
    prefix = value.get("shared_prefix_validation")
    mapping = value.get("action_mapping_validation")
    static = value.get("static_contract")
    if not isinstance(candidate, Mapping):
        raise ProtocolError("SmolVLA candidate validation is missing")
    try:
        max_candidate_delta = float(candidate.get("max_abs_delta_from_candidate0", 0.0))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("SmolVLA candidate delta is invalid") from exc
    if (
        candidate.get("all_candidates_distinct") is not True
        or candidate.get("shape") != [4, 50, 14]
        or candidate.get("piper_limits_satisfied") is not True
        or not isinstance(candidate.get("candidate_sha256"), list)
        or len(candidate["candidate_sha256"]) != 4
        or any(not _is_sha(item) for item in candidate["candidate_sha256"])
        or max_candidate_delta <= 0.0
        or not isinstance(prefix, Mapping)
        or prefix.get("bit_exact_across_candidates") is not True
        or not isinstance(prefix.get("shape"), list)
        or not prefix["shape"]
        or prefix["shape"][0] != 4
        or not _is_sha(prefix.get("shared_prefix_sha256"))
        or not isinstance(mapping, Mapping)
        or mapping.get("identity_inferred_from_equal_dimension") is not False
        or mapping.get("kinematic_equivalence_claimed") is not False
        or mapping.get("physical_equivalence_claimed") is not False
        or mapping.get("execution_authorized") is not False
        or not isinstance(static, Mapping)
        or static.get("authorization_ceiling") != "forward_only"
        or static.get("environment_execution_authorized") is not False
        or static.get("transfer_claim_authorized") is not False
    ):
        raise ProtocolError("SmolVLA candidate/shared-state preflight did not pass")
    return {
        "actor_id": TARGET_ACTOR_ID,
        "authorization_ceiling": "forward_only",
        "environment_execution_authorized": False,
        "transfer_claim_authorized": False,
        "implementation_sha256": value["implementation_sha256"],
    }


def validate_v7_activation(value: Mapping[str, Any]) -> dict[str, Any]:
    activation_sha = _signed(value, "activation_sha256", "V7 composite activation")
    selector = value.get("action_selector")
    inactive = value.get("inactive_or_fallback")
    if (
        value.get("format") != "etsf_composite_structured_prediction_activation_v1"
        or value.get("status") != "active_structured_prediction_development_only"
        or value.get("evidence_scope") != "adaptive_development_only"
        or value.get("transfer_claim_authorized") is not False
        or value.get("fresh50_inputs_accepted") is not False
        or value.get("fresh50_labels_read") is not False
        or not isinstance(selector, Mapping)
        or selector.get("authority") != "v7_fixed_parameter_free_selector"
        or selector.get("v8_replacement_authorized") is not False
        or selector.get("v8_success_input_allowed") is not False
        or selector.get("v8_regress_input_allowed") is not False
        or selector.get("duration_v2_input_allowed") is not False
        or int(selector.get("deployment_candidate_count", -1)) != 4
        or not _is_sha(selector.get("implementation_sha256"))
        or not isinstance(inactive, Mapping)
        or inactive.get("success", {}).get("status") != "inactive"
        or inactive.get("recovery", {}).get("status") != "inactive"
        or inactive.get("object", {}).get("status") != "fallback_only"
        or inactive.get("total_uncertainty", {}).get("status") != "unavailable"
    ):
        raise ProtocolError("V7 activation capability boundary changed")
    return {
        "activation_sha256": activation_sha,
        "selector_implementation_sha256": selector["implementation_sha256"],
        "authority": selector["authority"],
        "success_recovery_object_inputs_allowed": False,
        "transfer_claim_authorized": False,
    }


def _artifact(path: Path, role: str) -> dict[str, str]:
    resolved = _reject_sensitive_path(path, role)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def freeze_protocol(
    *,
    d250_identity_path: Path,
    target_seed_manifest_path: Path,
    forward_preflight_path: Path,
    v7_activation_path: Path,
    event_spec_path: Path,
    forward_preflight_implementation_path: Path,
    v7_utility_path: Path,
    actor_agnostic_plugin_path: Path,
    smolvla_collector_path: Path,
) -> dict[str, Any]:
    """Build the fixed protocol without opening target outcome or HDF data."""

    paths = {
        "d250_collection_identity": d250_identity_path,
        "target_seed_manifest": target_seed_manifest_path,
        "smolvla_piper_forward_preflight": forward_preflight_path,
        "v7_composite_activation": v7_activation_path,
        "event_spec": event_spec_path,
        "smolvla_piper_preflight_implementation": forward_preflight_implementation_path,
        "v7_utility_implementation": v7_utility_path,
        "actor_agnostic_plugin_implementation": actor_agnostic_plugin_path,
        "smolvla_schema5_collector": smolvla_collector_path,
    }
    artifacts = {name: _artifact(path, name) for name, path in paths.items()}
    d250_value = _load_json(Path(artifacts["d250_collection_identity"]["path"]), "D250 identity")
    d250 = validate_d250_identity(d250_value)
    seeds_value = _load_json(Path(artifacts["target_seed_manifest"]["path"]), "target seed manifest")
    seeds = validate_target_seed_manifest(
        seeds_value,
        d250_identity_file_sha256=artifacts["d250_collection_identity"]["sha256"],
        d250=d250,
    )
    preflight = validate_forward_preflight(
        _load_json(Path(artifacts["smolvla_piper_forward_preflight"]["path"]), "forward preflight")
    )
    if artifacts["smolvla_piper_preflight_implementation"]["sha256"] != preflight[
        "implementation_sha256"
    ]:
        raise ProtocolError("forward preflight receipt and implementation SHA differ")
    activation = validate_v7_activation(
        _load_json(Path(artifacts["v7_composite_activation"]["path"]), "V7 activation")
    )
    if artifacts["v7_utility_implementation"]["sha256"] != activation[
        "selector_implementation_sha256"
    ]:
        raise ProtocolError("V7 activation and utility implementation SHA differ")
    event_spec = _load_json(Path(artifacts["event_spec"]["path"]), "event spec")
    if TASK not in event_spec.get("chains", {}) or TASK not in event_spec.get("calibration", {}):
        raise ProtocolError("event spec lacks the target task chain/calibration")
    event_spec_sha = artifacts["event_spec"]["sha256"]
    if d250["event_spec_sha256"] != event_spec_sha:
        raise ProtocolError("D250 and target protocol event-spec SHA differ")

    protocol: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "scope": {
            "task": TASK,
            "source_actor_training_body": SOURCE_BODY,
            "target_execution_body": TARGET_BODY,
            "target_actor_id": TARGET_ACTOR_ID,
            "development_only": True,
            "fresh_or_confirmation_inputs_accepted": False,
            "source_d250_counts_as_target_evidence": False,
        },
        "artifacts": artifacts,
        "identity_bindings": {
            "d250": {key: value for key, value in d250.items() if key not in {"requested", "resolved"}},
            "target_seed_manifest_sha256": seeds["seed_manifest_sha256"],
            "target_identity_sets_sha256": seeds["target_identity_sets_sha256"],
            "heldout_identity_set_sha256": seeds["heldout_identity_set_sha256"],
            "event_spec_sha256": event_spec_sha,
            "labels_or_outcomes_read_by_freezer": False,
        },
        "conditions": {
            "baseline": {
                "name": "direct_smolvla_policy",
                "candidate": "deterministic",
                "candidate_index_by_name_not_position": True,
                "selection_rule": "always_execute_named_deterministic_candidate",
            },
            "plugin": {
                "name": "v7_event_world_model_selector",
                "candidate_names": list(TARGET_CANDIDATES),
                "authority": "v7_fixed_parameter_free_selector",
                "success_recovery_object_heads_used_for_ranking": False,
                "v8_or_duration_v2_selector_replacement_allowed": False,
                "required_contract_guards": [
                    "state_contract_matched",
                    "action_contract_matched",
                    "embodiment_contract_matched",
                    "clock_contract_matched",
                    "predicate_contract_matched",
                ],
                "fallback": "named_deterministic_candidate",
            },
        },
        "paired_execution": {
            "unit": "requested_and_resolved_seed_pair",
            "both_conditions_run_for_every_evaluation_pair": True,
            "same_explicit_instruction_and_initial_scene_hash_required": True,
            "initial_parity_requires_both": [
                "measured_joint_qpos_hash",
                "commanded_drive_target_hash",
            ],
            "actor_observation_state14_semantics": "commanded_drive_target_not_measured_qpos",
            "measured_qpos_selector_input_allowed": False,
            "condition_order": "sha256(ascii(pair_id||20261003))_least_significant_bit",
            "candidate0_noise_and_actor_rng_identical_across_conditions": True,
            "plugin_candidates": "four_native_flow_candidates_with_fixed_per_query_noise_seeds",
            "online_selector_input_cutoff": "before_any_candidate_action_or_environment_step",
            "outcome_based_retry_allowed": False,
            "infrastructure_retry": "at_most_once_same_seed_before_any_outcome_deserialization",
            "broken_pair_gate": "experiment_invalid_if_more_than_2_percent_evaluation_pairs_incomplete",
            "primary_population": "all_400_preregistered_pairs_intention_to_treat_with_worst_case_missing_sensitivity",
        },
        "stages": {
            "adaptation": {
                "groups": ADAPTATION_GROUPS,
                "first_20_groups": (
                    "label_quarantined_direct_actor_only_operational_smoke_then_fixed_prefix_release"
                ),
                "plugin_during_first_20_groups_allowed": False,
                "reason": (
                    "forward-only preflight does not authorize execution and current V7 cannot "
                    "consume the native SmolVLA 960D state directly"
                ),
                "allowed_updates": [
                    "state_adapter",
                    "policy_adapter",
                    "body_action_adapter",
                    "clock_adapter",
                    "nonprivileged_event_observer",
                ],
                "shared_event_core_updates_allowed": False,
            },
            "validation": {
                "groups": VALIDATION_GROUPS,
                "allowed_choices": [
                    "one_adapter_checkpoint_from_preregistered_grid",
                    "one_uncertainty_threshold_from_preregistered_grid",
                    "one_candidate_validity_threshold_from_preregistered_grid",
                ],
                "success_optimized_selector_formula_allowed": False,
                "final_bundle_must_be_frozen_before_evaluation": True,
            },
            "evaluation": {
                "groups": EVALUATION_GROUPS,
                "labels_opened_once_after_all_online_selection_records_are_signed": True,
                "interim_effectiveness_testing": False,
                "hyperparameter_or_threshold_updates_allowed": False,
            },
        },
        "sample_size_basis": {
            "fixed_evaluation_pairs": EVALUATION_GROUPS,
            "planning_alternative": {
                "net_success_delta": 0.05,
                "total_discordant_probability": 0.12,
                "two_sided_alpha": PRIMARY_ALPHA,
                "normal_approximate_power": 0.82,
            },
            "sample_size_reestimation_after_outcomes": False,
            "note": "planning approximation only; inference uses paired bootstrap and exact sign test",
        },
        "online_label_blinding": {
            "allowed_selector_inputs": [
                "current_RGB_or_actor_hidden_observer_state",
                "actor_observation_state14_commanded_drive_target",
                "language_instruction",
                "four_pre_action_candidate_chunks",
                "frozen_event_predictions_and_uncertainty",
            ],
            "forbidden_selector_inputs": [
                "success",
                "reward",
                "future_event",
                "future_object_pose",
                "candidate_outcome",
                "simulator_privileged_object_pose",
                "measured_joint_qpos_unless_a_separately_preregistered_nonprivileged_observer_exposes_it",
                "other_condition_result",
            ],
            "selection_record_signed_before_step": [
                "pair_id",
                "query_index",
                "candidate_action_sha256",
                "observer_input_sha256",
                "prediction_sha256",
                "contract_guard",
                "proposed_index",
                "selected_index",
                "fallback_reason",
            ],
            "evaluation_reservation": "O_EXCL_before_any_evaluation_outcome_or_event_label_read",
        },
        "metrics": {
            "primary_task_success": {
                "estimand": "unconditional_paired_success_mean_plugin_minus_direct_over_400_pairs",
                "bootstrap": {
                    "samples": BOOTSTRAP_SAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "unit": "paired_seed",
                    "ci": 0.95,
                },
                "exact_two_sided_mcnemar_sign_test": True,
                "alpha": PRIMARY_ALPHA,
                "gate": {
                    "delta_strictly_positive": True,
                    "bootstrap_95_lcb_strictly_positive": True,
                    "exact_p_strictly_below": PRIMARY_ALPHA,
                    "minimum_policy_changes": MINIMUM_POLICY_CHANGES,
                    "harmful_rate_over_all_changed_episodes_max": HARMFUL_RATE_MAX,
                    "coverage_min": 0.10,
                },
            },
            "event_prediction": {
                "evaluation_unit": "logical_episode_group_bootstrap_not_transition_iid",
                "next_event": ["accuracy", "macro_f1", "nll", "ece"],
                "observed_destination": ["accuracy", "macro_f1", "nll", "ece"],
                "duration_quantity": "decision_step_duration_not_physical_time",
                "duration": ["mae_log1p_decision_steps", "laplace_nll", "coverage"],
                "baselines_fit_on_adaptation_only": [
                    "event_frequency",
                    "current_event_identity",
                    "event_x_body_decision_step_duration_median",
                ],
                "accuracy_claim_gate": (
                    "group_bootstrap_95_lcb_of_next_event_accuracy_and_nll_skill_above_"
                    "best_baseline_and_all_reported_classes_have_nonzero_evaluation_support"
                ),
            },
            "recovery": {
                "head_status_at_freeze": "inactive",
                "ranking_input_allowed": False,
                "minimum_independent_training_groups_per_class": 50,
                "metrics_if_support_gate_later_passes": [
                    "brier_vs_adaptation_prevalence",
                    "nll_vs_adaptation_prevalence",
                    "average_precision_minus_prevalence",
                    "ece",
                    "unconditional_paired_recovery_rate_delta",
                ],
                "otherwise": "report_support_and_keep_recovery_head_inactive_no_accuracy_claim",
            },
            "success_prediction": {
                "head_status_at_freeze": "inactive",
                "ranking_input_allowed": False,
                "diagnostic_metrics_only": ["brier", "nll", "average_precision", "ece"],
            },
        },
        "adaptive_bias_controls": {
            "one_primary_comparison": "v7_plugin_vs_direct_policy",
            "evaluation_seed_order_and_size_frozen": True,
            "adaptation_is_fixed_prefix_not_outcome_selected": True,
            "validation_bundle_counted_as_model_selection_not_test_evidence": True,
            "evaluation_outcomes_never_used_for_stopping_or_reconfiguration": True,
            "secondary_metrics_do_not_rescue_failed_primary": True,
            "all_exclusions_and_missing_pairs_reported": True,
        },
        "known_runtime_audit": {
            "piper_observation_state14": "commanded_drive_target_not_measured_qpos",
            "measured_qpos_must_be_logged_separately_for_initial_parity": True,
            "policy_step_to_physics": (
                "TOPP_expansion_uses_a_variable_number_of_250Hz_physics_steps"
            ),
            "physical_time_duration_claim_allowed": False,
            "physical_time_claim_requires": [
                "simulator_monotonic_timestamp",
                "per_decision_physics_substep_count",
                "clock_adapter_validation",
            ],
            "explicit_seed_reset_refreshes_episode_info_list": False,
            "instruction_from_episode_info_list_allowed": False,
            "instruction_semantics_receipt_required_per_pair": True,
            "current_v7_native_smolvla_960d_input_allowed": False,
            "required_before_plugin_execution": [
                "validated_960D_state_adapter",
                "validated_50x14_action_policy_adapter",
                "validated_Piper_body_action_adapter",
                "validated_decision_step_clock_adapter",
                "validated_nonprivileged_predicate_adapter",
            ],
        },
        "claim_boundary": {
            "if_primary_and_prediction_gates_pass": (
                "V7-assisted system improves an Aloha-trained SmolVLA actor when executed on Piper, "
                "within this preregistered nonFresh development experiment"
            ),
            "not_authorized_by_this_experiment": [
                "untouched_confirmation_claim",
                "universal_cross_embodiment_claim",
                "event_world_model_itself_transfers_across_bodies",
                "zero_shot_claim_if_any_Piper_outcome_labels_train_or_select_adapters",
            ],
            "event_world_model_cross_body_minimum_future_gate": (
                "freeze a core trained without Piper, hold policy and task fixed across source/target "
                "bodies, permit only preregistered small body/clock adapters, then pass the same paired "
                "success and prediction gates on untouched Piper seeds"
            ),
            "strong_multi_body_claim": (
                "repeat the held-out-body result on at least two target embodiments and multiple tasks"
            ),
        },
        "source_readiness": {
            "d250_identity_bound_without_labels": True,
            "forward_preflight_authorization_ceiling": preflight["authorization_ceiling"],
            "v7_activation_bound": activation["activation_sha256"],
            "all_five_target_contract_guards_currently_empirically_proven": False,
            "execution_authorized_by_this_protocol": False,
            "transfer_claim_authorized_by_this_protocol": False,
        },
    }
    protocol["preregistration_sha256"] = canonical_sha256(protocol)
    return protocol


def validate_frozen_protocol(value: Mapping[str, Any]) -> None:
    _signed(value, "preregistration_sha256", "frozen protocol")
    if (
        value.get("format") != FORMAT
        or value.get("status") != STATUS
        or value.get("scope", {}).get("fresh_or_confirmation_inputs_accepted") is not False
        or value.get("stages", {}).get("evaluation", {}).get("groups") != EVALUATION_GROUPS
        or value.get("metrics", {}).get("primary_task_success", {}).get("alpha") != PRIMARY_ALPHA
        or value.get("source_readiness", {}).get("execution_authorized_by_this_protocol") is not False
        or value.get("source_readiness", {}).get("transfer_claim_authorized_by_this_protocol") is not False
    ):
        raise ProtocolError("frozen protocol semantics changed")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ProtocolError("frozen protocol artifact bindings are missing")
    for name, artifact in artifacts.items():
        if (
            not isinstance(name, str)
            or not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "sha256"}
            or not _is_sha(artifact.get("sha256"))
        ):
            raise ProtocolError("frozen protocol artifact binding is invalid")
        path = _reject_sensitive_path(str(artifact["path"]), f"artifacts.{name}")
        if file_sha256(path) != artifact["sha256"]:
            raise ProtocolError(f"frozen protocol artifact changed: {name}")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d250-identity", type=Path, required=True)
    parser.add_argument("--target-seed-manifest", type=Path, required=True)
    parser.add_argument("--forward-preflight", type=Path, required=True)
    parser.add_argument("--v7-activation", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--forward-preflight-implementation", type=Path, required=True)
    parser.add_argument("--v7-utility", type=Path, required=True)
    parser.add_argument("--actor-agnostic-plugin", type=Path, required=True)
    parser.add_argument("--smolvla-collector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = _reject_sensitive_path(args.output, "output", must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    result = freeze_protocol(
        d250_identity_path=args.d250_identity,
        target_seed_manifest_path=args.target_seed_manifest,
        forward_preflight_path=args.forward_preflight,
        v7_activation_path=args.v7_activation,
        event_spec_path=args.event_spec,
        forward_preflight_implementation_path=args.forward_preflight_implementation,
        v7_utility_path=args.v7_utility,
        actor_agnostic_plugin_path=args.actor_agnostic_plugin,
        smolvla_collector_path=args.smolvla_collector,
    )
    validate_frozen_protocol(result)
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "target_groups": TOTAL_GROUPS,
                "evaluation_pairs": EVALUATION_GROUPS,
                "outcomes_read": False,
                "execution_authorized": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ADAPTATION_GROUPS",
    "EVALUATION_GROUPS",
    "FORMAT",
    "LABEL_ACCESS_CONTRACT",
    "ProtocolError",
    "STATUS",
    "TARGET_SEED_FORMAT",
    "TARGET_SEED_STATUS",
    "TOTAL_GROUPS",
    "VALIDATION_GROUPS",
    "canonical_sha256",
    "file_sha256",
    "freeze_protocol",
    "paired_condition_order",
    "validate_d250_identity",
    "validate_forward_preflight",
    "validate_frozen_protocol",
    "validate_target_seed_manifest",
    "validate_v7_activation",
]
