#!/usr/bin/env python3
"""Strict paired task-success protocol for SmolVLA/Piper event selection.

This module reuses the schema-v6 dense branch collector.  On the collector's
first root query, before any environment step or outcome exists, a frozen event
plugin chooses one of the exact four actor candidates.  The choice is written
with O_EXCL.  The existing collector then executes every feasible root branch
from the same reset and with the same lowest-legal continuation.  Baseline and
plugin task success are read from those actually executed branches, never from
predicted success probabilities.

The production protocol exposes only a development lane.  A separately held
seed authority is represented by count and identity-set hash, not by identities
or outcomes, so this implementation cannot execute that lane.  Paths containing
``fresh`` or ``confirmation`` are rejected, and no existing sensitive artifact
is accepted by the freezer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FORMAT = "etsf_smolvla_piper_paired_task_success_protocol_v1"
AUTHORITY_FORMAT = "etsf_smolvla_piper_paired_task_success_seed_authority_v1"
SELECTION_FORMAT = "etsf_smolvla_piper_preoutcome_selection_v1"
PAIR_RESULT_FORMAT = "etsf_smolvla_piper_paired_task_success_result_v1"
EVALUATION_FORMAT = "etsf_smolvla_piper_paired_task_success_evaluation_v1"
HEAD_SUPPORT_FORMAT = "etsf_smolvla_piper_multitask_head_support_v1"
PRIMARY_UTILITY_FORMAT = "etsf_smolvla_piper_structured_multitask_utility_v1"
TASK = "move_can_pot"
BODY = "piper_piper_0.6"
ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
INSTRUCTION = "move the can into the pot"
CANDIDATE_COUNT = 4
PREFIX_DIM = 960
ACTION_DIM = 14
CHUNK_SIZE = 50
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_SAMPLES = 20_000
ALPHA = 0.05
PRODUCTION_MINIMUM_PAIRS = 400
MINIMUM_DISCORDANT_PAIRS = 20
MINIMUM_EXECUTED_POLICY_CHANGES = 40
MINIMUM_CHANGE_COVERAGE = 0.10
MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES = 0.10
MINIMUM_HEAD_GROUPS_PER_SIDE = {
    "post_event": 10,
    "next_event": 10,
    "duration": 10,
    "success": 50,
    "object_effect": 50,
}
PRIMARY_HEAD_WEIGHTS = {
    "post_event": -1.0,
    "next_event": 1.0,
    "duration": 1.0,
    "success": 0.5,
    "object_effect": 0.5,
}
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
SHA256_CHARS = frozenset("0123456789abcdef")
FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "success",
        "successes",
        "reward",
        "rewards",
        "outcome",
        "outcomes",
        "event",
        "events",
        "selected_index",
        "prediction",
        "predictions",
        "trajectory",
        "trajectories",
    }
)


class PairedSuccessProtocolError(RuntimeError):
    """Fail-closed protocol, selection, collection, or evaluation error."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_CHARS)
    )


def _contains_sensitive_path(path: PurePath) -> bool:
    return any(
        token in component.casefold()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def resolve_existing_file(path: Path, role: str) -> Path:
    raw = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if _contains_sensitive_path(absolute) or raw.is_symlink():
        raise PairedSuccessProtocolError(f"{role} path is sensitive or a symlink")
    resolved = raw.resolve(strict=True)
    if _contains_sensitive_path(resolved) or not resolved.is_file():
        raise PairedSuccessProtocolError(f"{role} must be a safe materialized file")
    return resolved


def resolve_new_path(path: Path, role: str) -> Path:
    raw = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if _contains_sensitive_path(absolute):
        raise PairedSuccessProtocolError(f"{role} path is sensitive")
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(absolute)
    parent = absolute.parent.resolve(strict=True)
    if _contains_sensitive_path(parent) or not parent.is_dir():
        raise PairedSuccessProtocolError(f"{role} parent is invalid")
    return absolute


def load_json(path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PairedSuccessProtocolError(f"{role} must be a materialized file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairedSuccessProtocolError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PairedSuccessProtocolError(f"{role} must contain an object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha256(recorded) or recorded != canonical_sha256(unsigned):
        raise PairedSuccessProtocolError(f"{role} logical signature mismatch")
    return str(recorded)


def _forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw, child in value.items():
            key = str(raw).casefold()
            current = f"{prefix}.{raw}" if prefix else str(raw)
            components = set(key.split("_"))
            # Negative boundary attestations such as ``outcomes_read: false``
            # are part of the authority contract, not leaked labels.  The
            # schema validator below rejects a true/non-boolean value.
            negative_read_attestation = key.endswith("_read") and child is False
            if not negative_read_attestation and (
                key in FORBIDDEN_AUTHORITY_KEYS
                or components & FORBIDDEN_AUTHORITY_KEYS
            ):
                found.append(current)
            found.extend(_forbidden_keys(child, current))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def pair_identity(
    *, requested_seed: int, resolved_seed: int, reset_identity_sha256: str
) -> dict[str, Any]:
    return {
        "task": TASK,
        "body": BODY,
        "actor_id": ACTOR_ID,
        "instruction_utf8_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        "requested_seed": requested_seed,
        "resolved_seed": resolved_seed,
        "reset_identity_sha256": reset_identity_sha256,
    }


def validate_seed_authority(
    value: Mapping[str, Any], *, minimum_pairs: int = PRODUCTION_MINIMUM_PAIRS
) -> dict[str, Any]:
    verify_signed(value, "seed_authority_sha256", "seed authority")
    forbidden = _forbidden_keys(value)
    if forbidden:
        raise PairedSuccessProtocolError(
            f"seed authority contains outcome-like keys: {forbidden[:3]}"
        )
    expected_root = {
        "format",
        "status",
        "task",
        "body",
        "actor_id",
        "instruction",
        "label_access_contract",
        "development",
        "sealed_evaluation_reserve",
        "disjoint_attestation",
        "existing_sensitive_artifacts_read",
        "seed_authority_sha256",
    }
    reserve = value.get("sealed_evaluation_reserve")
    attestation = value.get("disjoint_attestation")
    rows = value.get("development")
    if (
        set(value) != expected_root
        or value.get("format") != AUTHORITY_FORMAT
        or value.get("status") != "reset_identity_only_before_any_actor_or_outcome"
        or value.get("task") != TASK
        or value.get("body") != BODY
        or value.get("actor_id") != ACTOR_ID
        or value.get("instruction") != INSTRUCTION
        or value.get("label_access_contract")
        != "reset_identity_only_no_action_reward_success_event_or_trajectory"
        or value.get("existing_sensitive_artifacts_read") is not False
        or not isinstance(rows, list)
        or len(rows) < minimum_pairs
        or not isinstance(reserve, Mapping)
        or set(reserve)
        != {"count", "identity_set_sha256", "identities_disclosed", "outcomes_read"}
        or not isinstance(reserve.get("count"), int)
        or reserve["count"] < minimum_pairs
        or not _is_sha256(reserve.get("identity_set_sha256"))
        or reserve.get("identities_disclosed") is not False
        or reserve.get("outcomes_read") is not False
        or not isinstance(attestation, Mapping)
        or attestation.get("intersection_count") != 0
        or attestation.get("verified_without_disclosing_reserve_identities") is not True
    ):
        raise PairedSuccessProtocolError("seed authority boundary is invalid")
    exact_fields = {
        "ordinal",
        "pair_id",
        "requested_seed",
        "resolved_seed",
        "reset_identity_sha256",
    }
    decoded: list[dict[str, Any]] = []
    requested: set[int] = set()
    resolved: set[int] = set()
    pair_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != exact_fields:
            raise PairedSuccessProtocolError("development seed row fields changed")
        req, res = row["requested_seed"], row["resolved_seed"]
        if (
            type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or type(req) is not int
            or type(res) is not int
            or min(req, res) < 0
            or not _is_sha256(row["reset_identity_sha256"])
        ):
            raise PairedSuccessProtocolError("development seed identity is invalid")
        identity = pair_identity(
            requested_seed=req,
            resolved_seed=res,
            reset_identity_sha256=row["reset_identity_sha256"],
        )
        if row["pair_id"] != canonical_sha256(identity):
            raise PairedSuccessProtocolError("development pair_id mismatch")
        if req in requested or res in resolved or row["pair_id"] in pair_ids:
            raise PairedSuccessProtocolError("development seed identities are not unique")
        requested.add(req)
        resolved.add(res)
        pair_ids.add(str(row["pair_id"]))
        decoded.append(dict(row))
    development_set_sha = canonical_sha256(
        [
            {
                "pair_id": row["pair_id"],
                "requested_seed": row["requested_seed"],
                "resolved_seed": row["resolved_seed"],
            }
            for row in decoded
        ]
    )
    if attestation.get("development_identity_set_sha256") != development_set_sha:
        raise PairedSuccessProtocolError("development disjoint attestation hash changed")
    if attestation.get("reserve_identity_set_sha256") != reserve["identity_set_sha256"]:
        raise PairedSuccessProtocolError("reserve disjoint attestation hash changed")
    return {
        "development": decoded,
        "development_pairs": len(decoded),
        "development_identity_set_sha256": development_set_sha,
        "reserve_count": reserve["count"],
        "reserve_identity_set_sha256": reserve["identity_set_sha256"],
        "reserve_identities_read": False,
        "reserve_outcomes_read": False,
        "existing_sensitive_artifacts_read": False,
        "seed_authority_sha256": value["seed_authority_sha256"],
    }


def _lookup(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise PairedSuccessProtocolError(f"receipt is missing {dotted}")
        current = current[component]
    return current


def validate_dependency_receipt(spec: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "name",
        "receipt_path",
        "receipt_file_sha256",
        "expected_format",
        "expected_status",
        "logical_sha256_field",
        "required_fields",
        "run_exit_path",
        "run_exit_file_sha256",
    }
    if set(spec) != expected_fields or not isinstance(spec.get("required_fields"), Mapping):
        raise PairedSuccessProtocolError("dependency specification fields changed")
    receipt_path = resolve_existing_file(Path(str(spec["receipt_path"])), "dependency receipt")
    if file_sha256(receipt_path) != spec["receipt_file_sha256"]:
        raise PairedSuccessProtocolError("dependency receipt file SHA mismatch")
    receipt = load_json(receipt_path, "dependency receipt")
    logical_field = str(spec["logical_sha256_field"])
    logical = verify_signed(receipt, logical_field, str(spec["name"]))
    if (
        receipt.get("format") != spec["expected_format"]
        or receipt.get("status") != spec["expected_status"]
    ):
        raise PairedSuccessProtocolError("dependency terminal contract changed")
    for key, expected in spec["required_fields"].items():
        if _lookup(receipt, str(key)) != expected:
            raise PairedSuccessProtocolError(f"dependency required field changed: {key}")
    exit_path = resolve_existing_file(Path(str(spec["run_exit_path"])), "dependency run.exit")
    if (
        file_sha256(exit_path) != spec["run_exit_file_sha256"]
        or exit_path.read_bytes() != b"0\n"
    ):
        raise PairedSuccessProtocolError("dependency run.exit is not exact success")
    return {
        "name": spec["name"],
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": spec["receipt_file_sha256"],
        "receipt_logical_sha256": logical,
        "format": spec["expected_format"],
        "status": spec["expected_status"],
        "run_exit_path": str(exit_path),
        "run_exit_file_sha256": spec["run_exit_file_sha256"],
        "required_fields": dict(spec["required_fields"]),
    }


def validate_dependency_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    verify_signed(value, "dependency_authority_sha256", "dependency authority")
    dependencies = value.get("dependencies")
    if (
        value.get("format") != "etsf_paired_success_dependency_authority_v1"
        or value.get("status") != "all_upstream_training_complete_before_development_execution"
        or not isinstance(dependencies, list)
        or [item.get("name") for item in dependencies if isinstance(item, Mapping)]
        != ["lobo", "schema6", "adapter"]
        or value.get("existing_sensitive_artifacts_read") is not False
    ):
        raise PairedSuccessProtocolError("dependency authority is invalid")
    return {
        "dependencies": [validate_dependency_receipt(item) for item in dependencies],
        "dependency_authority_sha256": value["dependency_authority_sha256"],
        "existing_sensitive_artifacts_read": False,
    }


def validate_head_support(value: Mapping[str, Any]) -> dict[str, Any]:
    verify_signed(value, "head_support_sha256", "head support")
    heads = value.get("heads")
    expected = {"post_event", "next_event", "duration", "success", "object_effect"}
    if (
        value.get("format") != HEAD_SUPPORT_FORMAT
        or value.get("status") != "frozen_from_training_and_validation_only_before_paired_development"
        or value.get("paired_development_outcomes_read") is not False
        or value.get("sealed_evaluation_reserve_outcomes_read") is not False
        or not isinstance(heads, Mapping)
        or set(heads) != expected
    ):
        raise PairedSuccessProtocolError("multitask head support receipt is invalid")
    result: dict[str, Any] = {}
    for name in sorted(expected):
        row = heads[name]
        if not isinstance(row, Mapping) or set(row) != {
            "enabled_for_primary",
            "independent_positive_or_observed_groups",
            "independent_negative_or_censored_groups",
            "minimum_required_per_side",
            "support_source",
        }:
            raise PairedSuccessProtocolError(f"head support fields changed: {name}")
        positive = row["independent_positive_or_observed_groups"]
        negative = row["independent_negative_or_censored_groups"]
        minimum = row["minimum_required_per_side"]
        if (
            type(positive) is not int
            or type(negative) is not int
            or type(minimum) is not int
            or min(positive, negative, minimum) < 0
            or minimum != MINIMUM_HEAD_GROUPS_PER_SIDE[name]
            or not isinstance(row["support_source"], str)
            or not row["support_source"]
        ):
            raise PairedSuccessProtocolError(f"head support counts are invalid: {name}")
        enabled = bool(row["enabled_for_primary"])
        if enabled != (positive >= minimum and negative >= minimum):
            raise PairedSuccessProtocolError(
                f"head {name} enablement does not follow the frozen support gate"
            )
        result[name] = dict(row)
    if not result["post_event"]["enabled_for_primary"] or not result[
        "next_event"
    ]["enabled_for_primary"]:
        raise PairedSuccessProtocolError("primary selector requires supported post/next heads")
    return {
        "heads": result,
        "head_support_sha256": value["head_support_sha256"],
        "paired_development_outcomes_read": False,
        "sealed_evaluation_reserve_outcomes_read": False,
    }


def freeze_protocol(
    *,
    seed_authority_path: Path,
    dependency_authority_path: Path,
    plugin_manifest_path: Path,
    adapter_checkpoint_path: Path,
    collector_implementation_path: Path,
    plugin_implementation_path: Path,
    structured_utility_implementation_path: Path,
    head_support_path: Path,
    maximum_total_uncertainty: float,
    minimum_pairs: int = PRODUCTION_MINIMUM_PAIRS,
) -> dict[str, Any]:
    if not math.isfinite(maximum_total_uncertainty) or maximum_total_uncertainty < 0:
        raise PairedSuccessProtocolError("uncertainty threshold must be finite/non-negative")
    artifacts = {}
    for name, raw in (
        ("seed_authority", seed_authority_path),
        ("dependency_authority", dependency_authority_path),
        ("plugin_manifest", plugin_manifest_path),
        ("adapter_checkpoint", adapter_checkpoint_path),
        ("schema6_collector_implementation", collector_implementation_path),
        ("event_plugin_implementation", plugin_implementation_path),
        ("structured_utility_implementation", structured_utility_implementation_path),
        ("head_support", head_support_path),
    ):
        path = resolve_existing_file(raw, name)
        artifacts[name] = {"path": str(path), "sha256": file_sha256(path)}
    seeds = validate_seed_authority(
        load_json(Path(artifacts["seed_authority"]["path"]), "seed authority"),
        minimum_pairs=minimum_pairs,
    )
    dependencies = validate_dependency_authority(
        load_json(
            Path(artifacts["dependency_authority"]["path"]),
            "dependency authority",
        )
    )
    head_support = validate_head_support(
        load_json(Path(artifacts["head_support"]["path"]), "head support")
    )
    protocol: dict[str, Any] = {
        "format": FORMAT,
        "status": "preregistered_development_before_any_candidate_outcome",
        "scope": {
            "task": TASK,
            "body": BODY,
            "actor_id": ACTOR_ID,
            "instruction": INSTRUCTION,
            "development_only": True,
            "sealed_evaluation_reserve_execution_authorized": False,
            "existing_sensitive_artifacts_accepted": False,
        },
        "artifacts": artifacts,
        "dependencies": dependencies,
        "development_pairs": seeds["development"],
        "development_pair_count": seeds["development_pairs"],
        "development_identity_set_sha256": seeds[
            "development_identity_set_sha256"
        ],
        "sealed_evaluation_reserve": {
            "count": seeds["reserve_count"],
            "identity_set_sha256": seeds["reserve_identity_set_sha256"],
            "identities_read_by_protocol": False,
            "outcomes_read_by_protocol": False,
            "execution_authorized": False,
        },
        "paired_estimand": {
            "unit": "requested_resolved_seed_and_reset_identity",
            "baseline": "lowest_legal_feasibility_root_candidate",
            "plugin": "frozen_event_plugin_guarded_root_candidate",
            "shared": [
                "piper_task",
                "requested_and_resolved_seed",
                "reset_fingerprint",
                "actor_checkpoint",
                "root_observation",
                "four_root_candidates",
                "candidate_noise",
                "lowest_legal_continuation_after_root",
            ],
            "success_source": "simulator_info_success_from_executed_schema6_branch",
            "predicted_success_used_as_outcome": False,
        },
        "preoutcome_selection": {
            "selection_record_must_exist_before_first_environment_step": True,
            "outcomes_visible_to_selector": False,
            "fallback": "lowest_legal_feasibility_root_candidate",
            "maximum_total_uncertainty": float(maximum_total_uncertainty),
            "nonfinite_uncertainty_policy": "abstain_to_baseline",
            "above_threshold_policy": "abstain_to_baseline",
            "abstentions_included_in_unconditional_primary_estimand": True,
        },
        "primary_selector": {
            "format": PRIMARY_UTILITY_FORMAT,
            "base_formula": (
                "z(next_event_expected_progress)-z(post_event_expected_progress)"
                "+z(duration_log_mean)"
            ),
            "optional_terms": "+0.5*z(success_probability)+0.5*z(object_effect_utility)",
            "weights": dict(PRIMARY_HEAD_WEIGHTS),
            "head_support": head_support,
            "success_head_forced_off_when_support_insufficient": True,
            "object_head_forced_off_when_support_insufficient": True,
            "uncertainty_role": "guard_only_aleatoric_plus_epistemic_not_outcome",
            "guard_margin": 0.05,
            "secondary_diagnostics": [
                "ablate_post_event",
                "ablate_next_event",
                "ablate_duration",
                "ablate_success",
                "ablate_object_effect",
                "success_only",
            ],
            "secondary_diagnostics_change_primary_gate": False,
        },
        "statistics": {
            "minimum_preregistered_and_complete_pairs": minimum_pairs,
            "minimum_discordant_pairs": MINIMUM_DISCORDANT_PAIRS,
            "minimum_executed_policy_changes": MINIMUM_EXECUTED_POLICY_CHANGES,
            "minimum_executed_change_coverage": MINIMUM_CHANGE_COVERAGE,
            "maximum_harmful_rate_among_executed_changes": (
                MAXIMUM_HARMFUL_RATE_AMONG_EXECUTED_CHANGES
            ),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "exact_two_sided_mcnemar_sign_test": True,
            "alpha": ALPHA,
            "gate_requires_delta_strictly_positive": True,
            "gate_requires_ci_lower_strictly_positive": True,
            "gate_requires_exact_p_strictly_below_alpha": True,
        },
        "outcomes_or_hdf5_read_during_freeze": False,
        "existing_sensitive_artifacts_read": False,
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    logical = verify_signed(value, "protocol_sha256", "paired success protocol")
    if (
        value.get("format") != FORMAT
        or value.get("status")
        != "preregistered_development_before_any_candidate_outcome"
        or value.get("scope", {}).get("development_only") is not True
        or value.get("scope", {}).get(
            "sealed_evaluation_reserve_execution_authorized"
        )
        is not False
        or value.get("paired_estimand", {}).get("predicted_success_used_as_outcome")
        is not False
        or value.get("primary_selector", {}).get("format") != PRIMARY_UTILITY_FORMAT
        or value.get("primary_selector", {}).get(
            "secondary_diagnostics_change_primary_gate"
        )
        is not False
        or value.get("outcomes_or_hdf5_read_during_freeze") is not False
    ):
        raise PairedSuccessProtocolError("paired success protocol contract changed")
    return logical


def candidate_registry(query: Mapping[str, Any]) -> dict[str, Any]:
    hidden = np.asarray(query.get("hidden"))
    state = np.asarray(query.get("processed_state"))
    actions = np.asarray(query.get("mapped_actions"))
    mask = np.asarray(query.get("feasibility_mask"), dtype=bool)
    legal = np.asarray(query.get("legal_original_candidate_indices"), dtype=np.int64)
    native = list(query.get("native_action_sha256", ()))
    prefixes = list(query.get("candidate_prefix_sha256", ()))
    if (
        hidden.shape != (PREFIX_DIM,)
        or state.shape != (ACTION_DIM,)
        or actions.shape != (CANDIDATE_COUNT, CHUNK_SIZE, ACTION_DIM)
        or mask.shape != (CANDIDATE_COUNT,)
        or not np.array_equal(legal, np.flatnonzero(mask))
        or len(native) != CANDIDATE_COUNT
        or any(not _is_sha256(item) for item in native)
        or len(prefixes) != CANDIDATE_COUNT
        or len(set(prefixes)) != 1
        or query.get("prefix_bit_exact") is not True
    ):
        raise PairedSuccessProtocolError("root candidate registry is invalid")
    return {
        "shared_hidden_sha256": array_sha256(hidden),
        "processed_state_sha256": array_sha256(state),
        "mapped_actions_sha256": array_sha256(actions),
        "native_action_sha256": native,
        "shared_prefix_sha256": prefixes[0],
        "feasibility_mask": mask.astype(bool).tolist(),
        "legal_original_candidate_indices": legal.astype(int).tolist(),
        "registry_sha256": canonical_sha256(
            {
                "hidden": array_sha256(hidden),
                "state": array_sha256(state),
                "actions": array_sha256(actions),
                "native": native,
                "prefix": prefixes[0],
                "mask": mask.astype(bool).tolist(),
            }
        ),
    }


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def structured_multitask_selector_decision(
    *,
    predictions: Mapping[str, Any],
    candidate_valid_mask: Any,
    fallback_index: int,
    event_values: Sequence[float],
    protocol: Mapping[str, Any],
    plugin_manifest_sha256: str,
    adapter_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Compose the frozen primary utility and secondary head ablations.

    The numerical z-score primitive and 0.05 actor guard are imported from the
    existing actor-agnostic structured utility rather than independently tuned.
    Success/object terms are present only when their group-support receipt
    authorizes them.  Aleatoric and epistemic uncertainty never enter the score;
    their sum is the frozen abstention quantity.
    """

    from openvla_etsf_structured_event_time_utility import (
        GUARD_MARGIN,
        within_group_z_numpy,
    )

    validate_protocol(protocol)
    required = {
        "post_event_logits",
        "next_event_logits",
        "duration_selected_log_mean",
        "success_logit",
        "object_effect_utility",
        "aleatoric_uncertainty",
        "epistemic_uncertainty",
    }
    if not isinstance(predictions, Mapping) or set(predictions) != required:
        raise PairedSuccessProtocolError("structured multitask prediction fields changed")
    post = np.asarray(predictions["post_event_logits"], dtype=np.float64)
    nxt = np.asarray(predictions["next_event_logits"], dtype=np.float64)
    duration = np.asarray(predictions["duration_selected_log_mean"], dtype=np.float64)
    success_logit = np.asarray(predictions["success_logit"], dtype=np.float64)
    object_effect = np.asarray(predictions["object_effect_utility"], dtype=np.float64)
    aleatoric = np.asarray(predictions["aleatoric_uncertainty"], dtype=np.float64)
    epistemic = np.asarray(predictions["epistemic_uncertainty"], dtype=np.float64)
    valid = np.asarray(candidate_valid_mask, dtype=bool)
    values = np.asarray(event_values, dtype=np.float64)
    if (
        post.ndim != 2
        or post.shape != nxt.shape
        or post.shape[0] != CANDIDATE_COUNT
        or values.shape != (post.shape[1],)
        or any(
            array.shape != (CANDIDATE_COUNT,)
            for array in (duration, success_logit, object_effect, aleatoric, epistemic, valid)
        )
        or not 0 <= fallback_index < CANDIDATE_COUNT
        or not valid[fallback_index]
        or not all(
            np.isfinite(array).all()
            for array in (post, nxt, duration, success_logit, object_effect, values)
        )
        or any(
            bool((array[np.isfinite(array)] < 0).any())
            for array in (aleatoric, epistemic)
        )
        or not _is_sha256(plugin_manifest_sha256)
        or not _is_sha256(adapter_checkpoint_sha256)
    ):
        raise PairedSuccessProtocolError("structured multitask prediction shapes are invalid")
    selector_contract = protocol.get("primary_selector")
    if (
        not isinstance(selector_contract, Mapping)
        or selector_contract.get("format") != PRIMARY_UTILITY_FORMAT
        or float(selector_contract.get("guard_margin", -1)) != GUARD_MARGIN
        or selector_contract.get("weights") != PRIMARY_HEAD_WEIGHTS
    ):
        raise PairedSuccessProtocolError("primary selector contract changed")
    head_rows = selector_contract["head_support"]["heads"]
    enabled = {
        name: bool(head_rows[name]["enabled_for_primary"])
        for name in PRIMARY_HEAD_WEIGHTS
    }
    post_progress = (_softmax(post) * values).sum(axis=-1)
    next_progress = (_softmax(nxt) * values).sum(axis=-1)
    success_probability = 1.0 / (1.0 + np.exp(-np.clip(success_logit, -40, 40)))
    raw_components_unmasked = {
        "post_event": post_progress,
        "next_event": next_progress,
        "duration": duration,
        "success": success_probability,
        "object_effect": object_effect,
    }
    # Match the existing actor-agnostic utility boundary: invalid alternatives
    # are replaced by fallback values for z-score decomposition, then excluded
    # from proposal selection.  They cannot distort valid-candidate statistics.
    raw_components = {
        name: np.where(valid, value, value[fallback_index])
        for name, value in raw_components_unmasked.items()
    }
    standardized = {
        name: within_group_z_numpy(value) for name, value in raw_components.items()
    }
    weighted = {
        name: (
            float(PRIMARY_HEAD_WEIGHTS[name]) * standardized[name]
            if enabled[name]
            else np.zeros(CANDIDATE_COUNT, dtype=np.float64)
        )
        for name in PRIMARY_HEAD_WEIGHTS
    }
    utility = sum(weighted.values(), np.zeros(CANDIDATE_COUNT, dtype=np.float64))
    masked = np.where(valid, utility, -np.inf)
    proposed = int(np.argmax(masked))
    margin = float(utility[proposed] - utility[fallback_index])
    total_uncertainty = aleatoric + epistemic
    proposed_uncertainty = float(total_uncertainty[proposed])
    threshold = float(protocol["preoutcome_selection"]["maximum_total_uncertainty"])
    reasons: list[str] = []
    if proposed == fallback_index:
        reasons.append("utility_prefers_fallback")
    elif margin < GUARD_MARGIN:
        reasons.append("utility_margin_below_guard")
    if not math.isfinite(proposed_uncertainty):
        reasons.append("nonfinite_uncertainty")
    elif proposed_uncertainty > threshold:
        reasons.append("uncertainty_above_guard")
    changing = proposed != fallback_index
    accepted = changing and not reasons
    selected = proposed if accepted else fallback_index

    secondary: dict[str, dict[str, Any]] = {}
    for name in PRIMARY_HEAD_WEIGHTS:
        score = utility - weighted[name]
        proposal = int(np.argmax(np.where(valid, score, -np.inf)))
        secondary[f"ablate_{name}"] = {
            "available": True,
            "selected_index": proposal,
            "score_sha256": array_sha256(score),
        }
    if enabled["success"]:
        success_score = standardized["success"]
        secondary["success_only"] = {
            "available": True,
            "selected_index": int(np.argmax(np.where(valid, success_score, -np.inf))),
            "score_sha256": array_sha256(success_score),
        }
    else:
        secondary["success_only"] = {
            "available": False,
            "selected_index": fallback_index,
            "reason": "success_head_training_group_support_insufficient",
        }
    component_audit = {
        "enabled": enabled,
        "raw_sha256": {name: array_sha256(value) for name, value in raw_components.items()},
        "standardized_sha256": {
            name: array_sha256(value) for name, value in standardized.items()
        },
        "weighted_sha256": {name: array_sha256(value) for name, value in weighted.items()},
        "primary_utility_sha256": array_sha256(utility),
        "aleatoric_uncertainty_sha256": array_sha256(aleatoric),
        "epistemic_uncertainty_sha256": array_sha256(epistemic),
        "total_uncertainty_sha256": array_sha256(total_uncertainty),
    }
    component_audit["component_audit_sha256"] = canonical_sha256(component_audit)
    return {
        "proposed_index": proposed,
        "selected_index": selected,
        "fallback_index": fallback_index,
        "total_uncertainty": proposed_uncertainty,
        "score_margin": margin,
        "guard_fallback_used": changing and not accepted,
        "fallback_reasons": reasons,
        "plugin_manifest_sha256": plugin_manifest_sha256,
        "adapter_checkpoint_sha256": adapter_checkpoint_sha256,
        "primary_utility_format": PRIMARY_UTILITY_FORMAT,
        "component_audit": component_audit,
        "secondary_diagnostics": secondary,
        "success_head_enabled_for_primary": enabled["success"],
        "object_head_enabled_for_primary": enabled["object_effect"],
        "aleatoric_and_epistemic_used_as_guard_only": True,
    }


def selector_view(query: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy only pre-action fields and make ndarray values read-only."""

    allowed = (
        "hidden",
        "processed_state",
        "mapped_actions",
        "feasibility_mask",
        "legal_original_candidate_indices",
        "lowest_legal_original_candidate_index",
        "native_action_sha256",
        "candidate_prefix_sha256",
        "prefix_bit_exact",
    )
    projected: dict[str, Any] = {}
    for key in allowed:
        value = query[key]
        if isinstance(value, np.ndarray):
            value = value.copy()
            value.flags.writeable = False
        elif isinstance(value, list):
            value = tuple(value)
        projected[key] = value
    return MappingProxyType(projected)


def freeze_preoutcome_selection(
    *,
    pair: Mapping[str, Any],
    query: Mapping[str, Any],
    protocol: Mapping[str, Any],
    selector: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    protocol_sha = validate_protocol(protocol)
    registry = candidate_registry(query)
    legal = registry["legal_original_candidate_indices"]
    if len(legal) < 2:
        decision = {
            "status": "ineligible_fewer_than_two_legal_root_candidates",
            "proposed_index": None,
            "selected_index": None,
            "fallback_index": legal[0] if legal else None,
            "total_uncertainty": None,
            "score_margin": None,
            "guard_fallback_used": False,
            "fallback_reasons": ["insufficient_legal_root_candidates"],
        }
    else:
        raw = selector(selector_view(query))
        exact = {
            "proposed_index",
            "selected_index",
            "fallback_index",
            "total_uncertainty",
            "score_margin",
            "guard_fallback_used",
            "fallback_reasons",
            "plugin_manifest_sha256",
            "adapter_checkpoint_sha256",
            "primary_utility_format",
            "component_audit",
            "secondary_diagnostics",
            "success_head_enabled_for_primary",
            "object_head_enabled_for_primary",
            "aleatoric_and_epistemic_used_as_guard_only",
        }
        if not isinstance(raw, Mapping) or set(raw) != exact:
            raise PairedSuccessProtocolError("plugin selector decision fields changed")
        fallback = int(legal[0])
        proposed, selected = int(raw["proposed_index"]), int(raw["selected_index"])
        uncertainty = float(raw["total_uncertainty"])
        margin = float(raw["score_margin"])
        reasons = list(raw["fallback_reasons"])
        guarded = bool(raw["guard_fallback_used"])
        threshold = float(
            protocol["preoutcome_selection"]["maximum_total_uncertainty"]
        )
        changing = proposed != fallback
        must_abstain_uncertainty = changing and (
            not math.isfinite(uncertainty) or uncertainty > threshold
        )
        if (
            raw["fallback_index"] != fallback
            or proposed not in legal
            or selected not in legal
            or selected not in {proposed, fallback}
            or not _is_sha256(raw["plugin_manifest_sha256"])
            or not _is_sha256(raw["adapter_checkpoint_sha256"])
            or raw["primary_utility_format"] != PRIMARY_UTILITY_FORMAT
            or raw["aleatoric_and_epistemic_used_as_guard_only"] is not True
            or not isinstance(raw["component_audit"], Mapping)
            or not isinstance(raw["secondary_diagnostics"], Mapping)
            or set(raw["secondary_diagnostics"])
            != {
                "ablate_post_event",
                "ablate_next_event",
                "ablate_duration",
                "ablate_success",
                "ablate_object_effect",
                "success_only",
            }
            or bool(raw["success_head_enabled_for_primary"])
            != bool(
                protocol["primary_selector"]["head_support"]["heads"]["success"][
                    "enabled_for_primary"
                ]
            )
            or bool(raw["object_head_enabled_for_primary"])
            != bool(
                protocol["primary_selector"]["head_support"]["heads"][
                    "object_effect"
                ]["enabled_for_primary"]
            )
            or guarded != (changing and selected == fallback)
            or (must_abstain_uncertainty and selected != fallback)
            or (changing and selected == proposed and reasons)
            or (changing and selected == proposed and not math.isfinite(margin))
        ):
            raise PairedSuccessProtocolError("plugin selector violated frozen guard")
        verify_signed(
            raw["component_audit"],
            "component_audit_sha256",
            "utility component audit",
        )
        if must_abstain_uncertainty and not any(
            reason in {"uncertainty_above_guard", "nonfinite_uncertainty"}
            for reason in reasons
        ):
            raise PairedSuccessProtocolError("uncertainty abstention reason is missing")
        decision = {
            "status": "frozen_before_any_environment_step_or_outcome",
            **dict(raw),
            "proposed_index": proposed,
            "selected_index": selected,
            "fallback_index": fallback,
            "total_uncertainty": uncertainty,
            "score_margin": margin,
            "guard_fallback_used": guarded,
            "fallback_reasons": reasons,
            "uncertainty_threshold": threshold,
            "proposed_change": changing,
            "executed_change": selected != fallback,
            "uncertainty_abstention": must_abstain_uncertainty,
        }
    record: dict[str, Any] = {
        "format": SELECTION_FORMAT,
        "status": decision["status"],
        "protocol_sha256": protocol_sha,
        "pair_id": pair["pair_id"],
        "requested_seed": pair["requested_seed"],
        "resolved_seed": pair["resolved_seed"],
        "reset_identity_sha256": pair["reset_identity_sha256"],
        "candidate_registry": registry,
        "decision": decision,
        "environment_steps_before_selection": 0,
        "candidate_outcomes_visible_to_selector": False,
        "success_reward_event_or_trajectory_visible_to_selector": False,
        "selection_record_create_once_before_branch_execution": True,
    }
    record["selection_record_sha256"] = canonical_sha256(record)
    immutable_json_new(output, record)
    return record


class PreOutcomeSelectionHook:
    """Wrap the schema6 query function and freeze the first root decision."""

    def __init__(
        self,
        *,
        query_fn: Callable[[Mapping[str, Any], int], dict[str, Any]],
        selector: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        pair: Mapping[str, Any],
        protocol: Mapping[str, Any],
        selection_output: Path,
    ) -> None:
        self.query_fn = query_fn
        self.selector = selector
        self.pair = pair
        self.protocol = protocol
        self.selection_output = selection_output
        self.selection: dict[str, Any] | None = None
        self.root_registry_sha256: str | None = None
        self.calls = 0

    def __call__(self, observation: Mapping[str, Any], query_index: int) -> dict[str, Any]:
        query = self.query_fn(observation, query_index)
        self.calls += 1
        if query_index == 0:
            registry_sha = candidate_registry(query)["registry_sha256"]
            if self.selection is None:
                self.selection = freeze_preoutcome_selection(
                    pair=self.pair,
                    query=query,
                    protocol=self.protocol,
                    selector=self.selector,
                    output=self.selection_output,
                )
                self.root_registry_sha256 = registry_sha
            elif registry_sha != self.root_registry_sha256:
                raise PairedSuccessProtocolError(
                    "root actor candidates changed between reset-matched branches"
                )
        return query


def collect_paired_group(
    *,
    runtime: Mapping[str, Callable[..., Any]],
    query_fn: Callable[[Mapping[str, Any], int], dict[str, Any]],
    selector: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    pair: Mapping[str, Any],
    protocol: Mapping[str, Any],
    selection_output: Path,
    object_registry: Mapping[str, Any],
    pose_quality_spec: Mapping[str, Any],
    event_spec: Mapping[str, Any],
    max_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Collect one pair with the existing schema6 branch implementation."""

    # Lazy import keeps protocol freezing/evaluation HDF5-blind.
    from collect_smolvla_piper_schema6_dense_event_branches import collect_dense_group

    hook = PreOutcomeSelectionHook(
        query_fn=query_fn,
        selector=selector,
        pair=pair,
        protocol=protocol,
        selection_output=selection_output,
    )
    record = collect_dense_group(
        runtime=runtime,
        query_fn=hook,
        requested_seed=int(pair["requested_seed"]),
        instruction=INSTRUCTION,
        object_registry=object_registry,
        pose_quality_spec=pose_quality_spec,
        event_spec=event_spec,
        max_steps=max_steps,
    )
    if hook.selection is None:
        raise PairedSuccessProtocolError("collector never exposed a root query")
    result = derive_pair_result(record, hook.selection, protocol)
    return record, hook.selection, result


def derive_pair_result(
    record: Mapping[str, Any],
    selection: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    protocol_sha = validate_protocol(protocol)
    selection_sha = verify_signed(
        selection, "selection_record_sha256", "selection record"
    )
    if (
        selection.get("protocol_sha256") != protocol_sha
        or selection.get("candidate_outcomes_visible_to_selector") is not False
        or selection.get("environment_steps_before_selection") != 0
    ):
        raise PairedSuccessProtocolError("selection was not frozen pre-outcome")
    if record.get("status") != "collected_development_group":
        result: dict[str, Any] = {
            "format": PAIR_RESULT_FORMAT,
            "status": "incomplete_noncomparable_pair",
            "protocol_sha256": protocol_sha,
            "selection_record_sha256": selection_sha,
            "pair_id": selection["pair_id"],
            "requested_seed": selection["requested_seed"],
            "resolved_seed": selection["resolved_seed"],
            "reason": str(record.get("status", "collector_incomplete")),
            "task_success_outcome_available": False,
            "predicted_success_used_as_outcome": False,
        }
        result["pair_result_sha256"] = canonical_sha256(result)
        return result
    registry = candidate_registry(record["root_query"])
    if registry != selection["candidate_registry"]:
        raise PairedSuccessProtocolError("selection/root candidate registry mismatch")
    decision = selection["decision"]
    baseline_index = int(record["baseline_original_candidate_index"])
    if decision.get("fallback_index") != baseline_index:
        raise PairedSuccessProtocolError("selection fallback is not schema6 baseline")
    selected_index = int(decision["selected_index"])
    branches = {
        int(branch["original_candidate_index"]): branch for branch in record["branches"]
    }
    if baseline_index not in branches or selected_index not in branches:
        raise PairedSuccessProtocolError("selected branch task outcome was not executed")
    baseline_success = bool(branches[baseline_index]["success"])
    plugin_success = bool(branches[selected_index]["success"])
    secondary_task_success: dict[str, dict[str, Any]] = {}
    for name, diagnostic in decision["secondary_diagnostics"].items():
        if not isinstance(diagnostic, Mapping):
            raise PairedSuccessProtocolError("secondary diagnostic row is invalid")
        diagnostic_index = int(diagnostic["selected_index"])
        if diagnostic_index not in branches:
            raise PairedSuccessProtocolError(
                "secondary diagnostic selected an unexecuted branch"
            )
        secondary_task_success[name] = {
            "available": bool(diagnostic["available"]),
            "selected_index": diagnostic_index,
            "success": bool(branches[diagnostic_index]["success"]),
            "paired_difference_vs_baseline": int(
                bool(branches[diagnostic_index]["success"])
            )
            - int(baseline_success),
        }
        if "reason" in diagnostic:
            secondary_task_success[name]["reason"] = diagnostic["reason"]
    result = {
        "format": PAIR_RESULT_FORMAT,
        "status": "complete_executed_paired_task_success",
        "protocol_sha256": protocol_sha,
        "selection_record_sha256": selection_sha,
        "pair_id": selection["pair_id"],
        "requested_seed": selection["requested_seed"],
        "resolved_seed": selection["resolved_seed"],
        "reset_identity_sha256": selection["reset_identity_sha256"],
        "candidate_registry_sha256": registry["registry_sha256"],
        "baseline_original_candidate_index": baseline_index,
        "plugin_proposed_original_candidate_index": decision["proposed_index"],
        "plugin_selected_original_candidate_index": selected_index,
        "baseline_success": baseline_success,
        "plugin_success": plugin_success,
        "paired_success_difference": int(plugin_success) - int(baseline_success),
        "proposed_change": bool(decision["proposed_change"]),
        "executed_change": bool(decision["executed_change"]),
        "uncertainty_abstention": bool(decision["uncertainty_abstention"]),
        "guard_fallback_used": bool(decision["guard_fallback_used"]),
        "fallback_reasons": list(decision["fallback_reasons"]),
        "total_uncertainty": decision["total_uncertainty"],
        "uncertainty_threshold": decision["uncertainty_threshold"],
        "primary_utility_format": decision["primary_utility_format"],
        "success_head_enabled_for_primary": decision[
            "success_head_enabled_for_primary"
        ],
        "object_head_enabled_for_primary": decision[
            "object_head_enabled_for_primary"
        ],
        "secondary_task_success_diagnostics": secondary_task_success,
        "secondary_diagnostics_change_primary_gate": False,
        "task_success_source": "simulator_info_success_from_executed_schema6_branch",
        "baseline_and_plugin_share_root_candidate_registry": True,
        "predicted_success_used_as_outcome": False,
    }
    result["pair_result_sha256"] = canonical_sha256(result)
    return result


def exact_two_sided_mcnemar(helpful: int, harmful: int) -> float:
    if min(helpful, harmful) < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = helpful + harmful
    if discordant == 0:
        return 1.0
    smaller = min(helpful, harmful)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or samples < 100:
        raise ValueError("paired bootstrap requires nonempty 1D values and >=100 samples")
    rng = np.random.default_rng(seed)
    # Bound peak memory for large production runs while keeping exact frozen draws.
    means = np.empty(samples, dtype=np.float64)
    block = 1000
    for start in range(0, samples, block):
        count = min(block, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    return float(low), float(high)


def evaluate_pair_results(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    protocol_sha = validate_protocol(protocol)
    expected = {row["pair_id"]: row for row in protocol["development_pairs"]}
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        pair_sha = verify_signed(row, "pair_result_sha256", "pair result")
        pair_id = str(row.get("pair_id", ""))
        if (
            pair_id not in expected
            or pair_id in observed
            or row.get("protocol_sha256") != protocol_sha
            or pair_sha != row["pair_result_sha256"]
            or row.get("predicted_success_used_as_outcome") is not False
        ):
            raise PairedSuccessProtocolError("pair result identity/provenance is invalid")
        observed[pair_id] = row
    complete = [
        observed[pair_id]
        for pair_id in expected
        if pair_id in observed
        and observed[pair_id].get("status")
        == "complete_executed_paired_task_success"
    ]
    missing = sorted(set(expected) - set(observed))
    incomplete = sorted(
        pair_id
        for pair_id, row in observed.items()
        if row.get("status") != "complete_executed_paired_task_success"
    )
    for row in complete:
        threshold = float(
            protocol["preoutcome_selection"]["maximum_total_uncertainty"]
        )
        proposed_change = bool(row["proposed_change"])
        uncertainty = float(row["total_uncertainty"])
        must_abstain = proposed_change and (
            not math.isfinite(uncertainty) or uncertainty > threshold
        )
        if (
            float(row["uncertainty_threshold"]) != threshold
            or bool(row["uncertainty_abstention"]) != must_abstain
            or (must_abstain and bool(row["executed_change"]))
            or bool(row["guard_fallback_used"])
            != (proposed_change and not bool(row["executed_change"]))
            or row.get("task_success_source")
            != "simulator_info_success_from_executed_schema6_branch"
            or row.get("primary_utility_format") != PRIMARY_UTILITY_FORMAT
            or row.get("secondary_diagnostics_change_primary_gate") is not False
            or bool(row.get("success_head_enabled_for_primary"))
            != bool(
                protocol["primary_selector"]["head_support"]["heads"]["success"][
                    "enabled_for_primary"
                ]
            )
            or bool(row.get("object_head_enabled_for_primary"))
            != bool(
                protocol["primary_selector"]["head_support"]["heads"][
                    "object_effect"
                ]["enabled_for_primary"]
            )
            or not isinstance(row.get("secondary_task_success_diagnostics"), Mapping)
        ):
            raise PairedSuccessProtocolError("pair result violates uncertainty abstention")
    baseline = np.asarray([bool(row["baseline_success"]) for row in complete], dtype=np.int8)
    plugin = np.asarray([bool(row["plugin_success"]) for row in complete], dtype=np.int8)
    difference = plugin - baseline
    if len(complete):
        ci = paired_bootstrap_ci(
            difference,
            samples=int(protocol["statistics"]["bootstrap_samples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        )
        delta = float(difference.mean())
        baseline_rate = float(baseline.mean())
        plugin_rate = float(plugin.mean())
    else:
        ci = (math.nan, math.nan)
        delta = baseline_rate = plugin_rate = math.nan
    helpful = int(((baseline == 0) & (plugin == 1)).sum())
    harmful = int(((baseline == 1) & (plugin == 0)).sum())
    discordant = helpful + harmful
    proposed_changes = sum(bool(row["proposed_change"]) for row in complete)
    executed_changes = sum(bool(row["executed_change"]) for row in complete)
    abstentions = sum(bool(row["uncertainty_abstention"]) for row in complete)
    harmful_changed = sum(
        bool(row["executed_change"])
        and bool(row["baseline_success"])
        and not bool(row["plugin_success"])
        for row in complete
    )
    coverage = executed_changes / len(complete) if complete else 0.0
    harmful_rate = harmful_changed / executed_changes if executed_changes else 0.0
    p_value = exact_two_sided_mcnemar(helpful, harmful)
    minimum = int(protocol["statistics"]["minimum_preregistered_and_complete_pairs"])
    reasons = []
    if len(expected) < minimum or len(complete) < minimum:
        reasons.append("minimum_complete_pair_gate_failed")
    if missing or incomplete or len(complete) != len(expected):
        reasons.append("intention_to_treat_pair_completeness_gate_failed")
    if discordant < int(protocol["statistics"]["minimum_discordant_pairs"]):
        reasons.append("minimum_discordant_pair_gate_failed")
    if executed_changes < int(
        protocol["statistics"]["minimum_executed_policy_changes"]
    ):
        reasons.append("minimum_executed_policy_change_gate_failed")
    if coverage < float(
        protocol["statistics"]["minimum_executed_change_coverage"]
    ):
        reasons.append("minimum_change_coverage_gate_failed")
    if harmful_rate > float(
        protocol["statistics"]["maximum_harmful_rate_among_executed_changes"]
    ):
        reasons.append("harmful_changed_episode_rate_gate_failed")
    if not math.isfinite(delta) or delta <= 0:
        reasons.append("paired_success_delta_not_positive")
    if not math.isfinite(ci[0]) or ci[0] <= 0:
        reasons.append("paired_bootstrap_ci_lower_not_positive")
    if not p_value < float(protocol["statistics"]["alpha"]):
        reasons.append("exact_mcnemar_not_significant")
    # Worst-case intention-to-treat sensitivity: every absent/incomplete pair is
    # plugin failure and baseline success.  It is reported, never silently dropped.
    unresolved = len(expected) - len(complete)
    worst_case_delta = (
        (float(difference.sum()) - unresolved) / len(expected) if expected else math.nan
    )
    secondary_names = list(protocol["primary_selector"]["secondary_diagnostics"])
    secondary_metrics: dict[str, Any] = {}
    for name in secondary_names:
        diagnostic_rows = [
            row["secondary_task_success_diagnostics"][name]
            for row in complete
            if name in row["secondary_task_success_diagnostics"]
        ]
        available = [row for row in diagnostic_rows if bool(row["available"])]
        values = np.asarray(
            [int(row["paired_difference_vs_baseline"]) for row in available],
            dtype=np.int8,
        )
        secondary_metrics[name] = {
            "available_pairs": len(available),
            "unavailable_pairs": len(diagnostic_rows) - len(available),
            "paired_success_delta": float(values.mean()) if values.size else None,
            "diagnostic_only": True,
            "changes_primary_gate": False,
        }
    evaluation: dict[str, Any] = {
        "format": EVALUATION_FORMAT,
        "status": "task_success_gate_passed" if not reasons else "task_success_gate_failed",
        "protocol_sha256": protocol_sha,
        "estimand": "unconditional_paired_task_success_plugin_minus_baseline",
        "outcome_source": "executed_schema6_simulator_branches",
        "predicted_success_used_as_outcome": False,
        "preregistered_pairs": len(expected),
        "complete_pairs": len(complete),
        "missing_pair_ids": missing,
        "incomplete_pair_ids": incomplete,
        "baseline_success_count": int(baseline.sum()),
        "plugin_success_count": int(plugin.sum()),
        "baseline_success_rate": baseline_rate,
        "plugin_success_rate": plugin_rate,
        "paired_success_delta": delta,
        "paired_bootstrap_ci95": list(ci),
        "worst_case_missing_paired_delta": worst_case_delta,
        "helpful_discordant_pairs": helpful,
        "harmful_discordant_pairs": harmful,
        "discordant_pairs": discordant,
        "exact_two_sided_mcnemar_p": p_value,
        "proposed_change_pairs": proposed_changes,
        "executed_change_pairs": executed_changes,
        "uncertainty_abstention_pairs": abstentions,
        "executed_change_coverage": coverage,
        "harmful_rate_among_executed_changes": harmful_rate,
        "gate_passed": not reasons,
        "gate_reasons": reasons,
        "secondary_head_ablation_and_success_only": secondary_metrics,
        "secondary_diagnostics_change_primary_gate": False,
        "reserve_identities_read": False,
        "reserve_outcomes_read": False,
        "existing_sensitive_artifacts_read": False,
    }
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    return evaluation


def synthetic_protocol(pair_count: int = 120, threshold: float = 0.5) -> dict[str, Any]:
    rows = []
    for ordinal in range(pair_count):
        reset_sha = hashlib.sha256(f"reset-{ordinal}".encode()).hexdigest()
        identity = pair_identity(
            requested_seed=1000 + ordinal,
            resolved_seed=2000 + ordinal,
            reset_identity_sha256=reset_sha,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "pair_id": canonical_sha256(identity),
                "requested_seed": 1000 + ordinal,
                "resolved_seed": 2000 + ordinal,
                "reset_identity_sha256": reset_sha,
            }
        )
    heads = {
        name: {
            "enabled_for_primary": name not in {"success"},
            "independent_positive_or_observed_groups": 120 if name != "success" else 4,
            "independent_negative_or_censored_groups": 120 if name != "success" else 3,
            "minimum_required_per_side": MINIMUM_HEAD_GROUPS_PER_SIDE[name],
            "support_source": "synthetic_training_only",
        }
        for name in PRIMARY_HEAD_WEIGHTS
    }
    head_support = {
        "heads": heads,
        "head_support_sha256": "a" * 64,
    }
    protocol = {
        "format": FORMAT,
        "status": "preregistered_development_before_any_candidate_outcome",
        "scope": {
            "development_only": True,
            "sealed_evaluation_reserve_execution_authorized": False,
        },
        "development_pairs": rows,
        "preoutcome_selection": {"maximum_total_uncertainty": threshold},
        "paired_estimand": {"predicted_success_used_as_outcome": False},
        "primary_selector": {
            "format": PRIMARY_UTILITY_FORMAT,
            "weights": dict(PRIMARY_HEAD_WEIGHTS),
            "guard_margin": 0.05,
            "head_support": head_support,
            "secondary_diagnostics": [
                "ablate_post_event",
                "ablate_next_event",
                "ablate_duration",
                "ablate_success",
                "ablate_object_effect",
                "success_only",
            ],
            "secondary_diagnostics_change_primary_gate": False,
        },
        "statistics": {
            "minimum_preregistered_and_complete_pairs": min(100, pair_count),
            "minimum_discordant_pairs": min(20, max(1, pair_count // 10)),
            "minimum_executed_policy_changes": min(20, max(1, pair_count // 10)),
            "minimum_executed_change_coverage": 0.10,
            "maximum_harmful_rate_among_executed_changes": 0.10,
            "bootstrap_samples": 1000,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": ALPHA,
        },
        "outcomes_or_hdf5_read_during_freeze": False,
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def synthetic_smoke() -> dict[str, Any]:
    protocol = synthetic_protocol()
    results = []
    # 30 helpful, 2 harmful, 40 concordant successes, 48 concordant failures.
    for ordinal, pair in enumerate(protocol["development_pairs"]):
        if ordinal < 30:
            baseline, plugin = False, True
        elif ordinal < 32:
            baseline, plugin = True, False
        elif ordinal < 72:
            baseline = plugin = True
        else:
            baseline = plugin = False
        secondary = {
            name: {
                "available": name != "success_only",
                "selected_index": 1 if name != "success_only" else 0,
                "success": plugin if name != "success_only" else baseline,
                "paired_difference_vs_baseline": (
                    int(plugin) - int(baseline) if name != "success_only" else 0
                ),
            }
            for name in protocol["primary_selector"]["secondary_diagnostics"]
        }
        row = {
            "format": PAIR_RESULT_FORMAT,
            "status": "complete_executed_paired_task_success",
            "protocol_sha256": protocol["protocol_sha256"],
            "pair_id": pair["pair_id"],
            "baseline_success": baseline,
            "plugin_success": plugin,
            "proposed_change": ordinal < 40,
            "executed_change": ordinal < 32,
            "uncertainty_abstention": 32 <= ordinal < 40,
            "guard_fallback_used": 32 <= ordinal < 40,
            "total_uncertainty": 0.75 if 32 <= ordinal < 40 else 0.25,
            "uncertainty_threshold": 0.5,
            "primary_utility_format": PRIMARY_UTILITY_FORMAT,
            "success_head_enabled_for_primary": False,
            "object_head_enabled_for_primary": True,
            "secondary_task_success_diagnostics": secondary,
            "secondary_diagnostics_change_primary_gate": False,
            "task_success_source": "simulator_info_success_from_executed_schema6_branch",
            "predicted_success_used_as_outcome": False,
        }
        row["pair_result_sha256"] = canonical_sha256(row)
        results.append(row)
    evaluation = evaluate_pair_results(protocol, results)
    return {
        "status": "synthetic_smoke_passed",
        "gate_passed": evaluation["gate_passed"],
        "paired_success_delta": evaluation["paired_success_delta"],
        "paired_bootstrap_ci95": evaluation["paired_bootstrap_ci95"],
        "exact_two_sided_mcnemar_p": evaluation["exact_two_sided_mcnemar_p"],
        "uncertainty_abstention_pairs": evaluation["uncertainty_abstention_pairs"],
        "predicted_success_used_as_outcome": False,
        "reserve_outcomes_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("freeze-protocol", "evaluate", "synthetic-smoke"), required=True
    )
    parser.add_argument("--seed-authority", type=Path)
    parser.add_argument("--dependency-authority", type=Path)
    parser.add_argument("--plugin-manifest", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--collector-implementation", type=Path)
    parser.add_argument("--plugin-implementation", type=Path)
    parser.add_argument("--structured-utility-implementation", type=Path)
    parser.add_argument("--head-support", type=Path)
    parser.add_argument("--maximum-total-uncertainty", type=float)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--pair-results", type=Path, nargs="*")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic-smoke":
        print("SYNTHETIC_SMOKE=" + json.dumps(synthetic_smoke(), sort_keys=True))
        return
    if args.output is None:
        raise PairedSuccessProtocolError("mode requires --output")
    output = resolve_new_path(args.output, "output")
    if args.mode == "freeze-protocol":
        required = (
            args.seed_authority,
            args.dependency_authority,
            args.plugin_manifest,
            args.adapter_checkpoint,
            args.collector_implementation,
            args.plugin_implementation,
            args.structured_utility_implementation,
            args.head_support,
            args.maximum_total_uncertainty,
        )
        if any(value is None for value in required):
            raise PairedSuccessProtocolError("freeze-protocol arguments are incomplete")
        protocol = freeze_protocol(
            seed_authority_path=args.seed_authority,
            dependency_authority_path=args.dependency_authority,
            plugin_manifest_path=args.plugin_manifest,
            adapter_checkpoint_path=args.adapter_checkpoint,
            collector_implementation_path=args.collector_implementation,
            plugin_implementation_path=args.plugin_implementation,
            structured_utility_implementation_path=args.structured_utility_implementation,
            head_support_path=args.head_support,
            maximum_total_uncertainty=args.maximum_total_uncertainty,
        )
        immutable_json_new(output, protocol)
        print(
            "FROZEN_PROTOCOL="
            + json.dumps(
                {
                    "path": str(output),
                    "file_sha256": file_sha256(output),
                    "logical_sha256": protocol["protocol_sha256"],
                    "outcomes_or_hdf5_read": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.protocol is None or args.protocol_sha256 is None or args.pair_results is None:
        raise PairedSuccessProtocolError("evaluate arguments are incomplete")
    protocol_path = resolve_existing_file(args.protocol, "protocol")
    if file_sha256(protocol_path) != args.protocol_sha256:
        raise PairedSuccessProtocolError("protocol file SHA mismatch")
    protocol = load_json(protocol_path, "protocol")
    rows = [
        load_json(resolve_existing_file(path, "pair result"), "pair result")
        for path in args.pair_results
    ]
    evaluation = evaluate_pair_results(protocol, rows)
    immutable_json_new(output, evaluation)
    print(
        "PAIRED_SUCCESS_EVALUATION="
        + json.dumps(
            {
                "path": str(output),
                "file_sha256": file_sha256(output),
                "logical_sha256": evaluation["evaluation_sha256"],
                "gate_passed": evaluation["gate_passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ACTOR_ID",
    "AUTHORITY_FORMAT",
    "EVALUATION_FORMAT",
    "FORMAT",
    "PAIR_RESULT_FORMAT",
    "PRIMARY_UTILITY_FORMAT",
    "PairedSuccessProtocolError",
    "PreOutcomeSelectionHook",
    "SELECTION_FORMAT",
    "candidate_registry",
    "canonical_sha256",
    "collect_paired_group",
    "derive_pair_result",
    "evaluate_pair_results",
    "exact_two_sided_mcnemar",
    "freeze_preoutcome_selection",
    "freeze_protocol",
    "pair_identity",
    "paired_bootstrap_ci",
    "selector_view",
    "structured_multitask_selector_decision",
    "synthetic_protocol",
    "synthetic_smoke",
    "validate_dependency_authority",
    "validate_dependency_receipt",
    "validate_head_support",
    "validate_protocol",
    "validate_seed_authority",
]
