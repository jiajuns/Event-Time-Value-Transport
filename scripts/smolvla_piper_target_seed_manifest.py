#!/usr/bin/env python3
"""Pure, label-blind contracts for a non-Fresh Piper target seed registry.

This module deliberately has no simulator imports.  It separates an offline
candidate plan, a reset-only execution receipt, and a final freezer.  The
execution receipt is accepted only after an independently produced attestation
has proved that the *whole candidate pool* is disjoint from the committed
held-out identity set.  The final manifest additionally requires a second
attestation over the selected requested/resolved identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from preregister_smolvla_piper_v7_paired_development import (
    LABEL_ACCESS_CONTRACT,
    SOURCE_BODY,
    TARGET_ACTOR_ID,
    TARGET_BODY,
    TASK,
    ProtocolError,
    canonical_sha256,
    file_sha256,
    validate_d250_identity,
)


PLAN_FORMAT = "etsf_smolvla_piper_target_seed_plan_v2"
PLAN_STATUS = "preregistered_unresolved_reset_execution_not_authorized"
AUTHORIZATION_FORMAT = "etsf_smolvla_piper_target_reset_authorization_v2"
RESET_RECEIPT_FORMAT = "etsf_smolvla_piper_target_reset_receipt_v2"
RESET_RECEIPT_STATUS = "complete_reset_only_no_policy_no_step_no_outcome"
FINAL_FORMAT = "etsf_smolvla_piper_target_seed_manifest_v2"
FINAL_STATUS = "resolved_reset_identity_only_before_policy_execution"
LEGACY_FORMAT = "etsf_smolvla_piper_paired_development_seed_manifest_v1"
LEGACY_STATUS = "resolved_reset_identity_only_before_policy_execution"
ATTESTATION_STATUS = "verified_disjoint_without_disclosing_heldout_identities"

ADAPTATION = 80
VALIDATION = 50
EVALUATION = 400
TOTAL = ADAPTATION + VALIDATION + EVALUATION
CANDIDATE_START = 100_201_000
CANDIDATE_COUNT = 800
CANDIDATE_STEP = 1
INSTRUCTION = "move the can into the pot"

SEMANTICS_RECEIPT_UNSIGNED: dict[str, Any] = {
    "format": "etsf_explicit_instruction_semantics_receipt_v1",
    "task": TASK,
    "language": "en",
    "instruction": INSTRUCTION,
    "semantic_frame": {
        "theme": "can",
        "relation": "inside",
        "reference": "pot",
    },
    "source": "protocol_constant_not_episode_info_list",
    "episode_info_list_used": False,
}


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    if key in result:
        raise ProtocolError(f"refusing to replace existing signature {key}")
    result[key] = canonical_sha256(result)
    return result


def verify_signature(value: Mapping[str, Any], key: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise ProtocolError(f"{role} signature mismatch")
    return str(recorded)


def reject_sensitive_path(path: str | Path, role: str, *, must_exist: bool) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ProtocolError(f"{role} path must be absolute")
    resolved = raw.resolve(strict=False)
    forbidden = ("fresh", "confirmation", "trajectory", "trajectories", "label")
    if any(
        token in part.casefold()
        for part in (*raw.parts, *resolved.parts)
        for token in forbidden
    ):
        raise ProtocolError(f"{role} path is outside the non-Fresh identity-only scope")
    if must_exist and not resolved.is_file():
        raise ProtocolError(f"{role} is not an existing file")
    return resolved


def load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{role} must contain a JSON object")
    return value


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one durable, read-only JSON artifact without overwriting."""

    path = reject_sensitive_path(path, "output", must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Do not unlink: a partial O_EXCL artifact is evidence of a failed run.
        raise


def semantics_receipt() -> dict[str, Any]:
    return signed(SEMANTICS_RECEIPT_UNSIGNED, "receipt_sha256")


def _sha_commitment(value: Any, role: str) -> str:
    if not is_sha(value):
        raise ProtocolError(f"{role} must be a lowercase SHA256")
    return str(value)


def build_plan(
    *,
    d250_identity: Mapping[str, Any],
    d250_identity_file_sha256: str,
    heldout_identity_set_sha256: str,
    upstream_v7_seed_manifest_file_sha256: str,
    upstream_v7_seed_manifest_payload_sha256: str,
    upstream_data_audit_file_sha256: str,
    resolver_implementation_sha256: str,
    reset_adapter_implementation_sha256: str,
    candidate_start: int = CANDIDATE_START,
    candidate_count: int = CANDIDATE_COUNT,
    candidate_step: int = CANDIDATE_STEP,
) -> dict[str, Any]:
    """Freeze candidate order and split assignment before any simulator reset."""

    d250 = validate_d250_identity(d250_identity)
    commitments = {
        "d250_identity_file_sha256": _sha_commitment(
            d250_identity_file_sha256, "D250 identity file"
        ),
        "heldout_identity_set_sha256": _sha_commitment(
            heldout_identity_set_sha256, "heldout identity set"
        ),
        "upstream_v7_seed_manifest_file_sha256": _sha_commitment(
            upstream_v7_seed_manifest_file_sha256, "upstream V7 seed manifest"
        ),
        "upstream_v7_seed_manifest_payload_sha256": _sha_commitment(
            upstream_v7_seed_manifest_payload_sha256, "upstream V7 seed payload"
        ),
        "upstream_data_audit_file_sha256": _sha_commitment(
            upstream_data_audit_file_sha256, "upstream data audit"
        ),
        "resolver_implementation_sha256": _sha_commitment(
            resolver_implementation_sha256, "resolver implementation"
        ),
        "reset_adapter_implementation_sha256": _sha_commitment(
            reset_adapter_implementation_sha256, "reset adapter implementation"
        ),
    }
    if (
        isinstance(candidate_start, bool)
        or isinstance(candidate_count, bool)
        or isinstance(candidate_step, bool)
        or not all(isinstance(item, int) for item in (candidate_start, candidate_count, candidate_step))
        or candidate_start < 0
        or candidate_count < TOTAL
        or candidate_step < 1
    ):
        raise ProtocolError("candidate range must contain at least 530 non-negative integer seeds")
    candidates = [candidate_start + candidate_step * index for index in range(candidate_count)]
    d250_ids = set(d250["requested"]) | set(d250["resolved"])
    if set(candidates) & d250_ids:
        raise ProtocolError("target candidate pool overlaps D250 identities")
    receipt = semantics_receipt()
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "status": PLAN_STATUS,
        "task": TASK,
        "actor_id": TARGET_ACTOR_ID,
        "source_body": SOURCE_BODY,
        "target_body": TARGET_BODY,
        "purpose": "independent_nonfresh_target_development_no_confirmation_claim",
        "label_access_contract": LABEL_ACCESS_CONTRACT,
        "split_contract": {
            "selection_rule": "first_530_stable_resets_in_preregistered_candidate_order",
            "assignment_after_stability_only": [
                {"split": "adaptation", "count": ADAPTATION, "first_20_direct_operational": True},
                {"split": "validation", "count": VALIDATION},
                {"split": "evaluation", "count": EVALUATION},
            ],
            "outcome_dependent_selection_allowed": False,
        },
        "candidate_pool": {
            "start": candidate_start,
            "count": candidate_count,
            "step": candidate_step,
            "requested_seeds": candidates,
            "requested_identity_set_sha256": canonical_sha256(candidates),
            "unstable_setup_handling": "record_identity_only_and_advance_no_internal_seed_retry",
        },
        "instruction_contract": {
            "mode": "one_explicit_protocol_constant_for_every_reset_and_both_conditions",
            "instruction": INSTRUCTION,
            "instruction_sha256": hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest(),
            "semantics_receipt": receipt,
            "episode_info_list_used": False,
        },
        "identity_bindings": {
            **commitments,
            "d250_identity_sets_sha256": d250["identity_sets_sha256"],
            "d250_intersection_count": 0,
            "heldout_identities_disclosed": False,
        },
        "execution_gate": {
            "authorized_by_plan": False,
            "required_authorization_format": AUTHORIZATION_FORMAT,
            "candidate_pool_disjoint_attestation_required_before_environment_construction": True,
            "reset_only": True,
            "environment_step_allowed": False,
            "policy_import_or_forward_allowed": False,
            "reward_success_event_or_outcome_read_allowed": False,
        },
        "finalization_gate": {
            "selected_requested_and_resolved_identity_attestation_required": True,
            "heldout_identity_set_sha256": commitments["heldout_identity_set_sha256"],
            "sensitive_identities_may_be_embedded": False,
        },
    }
    return signed(plan, "plan_sha256")


def validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan_sha = verify_signature(value, "plan_sha256", "target seed plan")
    if (
        value.get("format") != PLAN_FORMAT
        or value.get("status") != PLAN_STATUS
        or value.get("task") != TASK
        or value.get("actor_id") != TARGET_ACTOR_ID
        or value.get("source_body") != SOURCE_BODY
        or value.get("target_body") != TARGET_BODY
        or value.get("label_access_contract") != LABEL_ACCESS_CONTRACT
    ):
        raise ProtocolError("target seed plan scope changed")
    pool = value.get("candidate_pool")
    split = value.get("split_contract")
    instruction = value.get("instruction_contract")
    bindings = value.get("identity_bindings")
    gate = value.get("execution_gate")
    if not all(isinstance(item, Mapping) for item in (pool, split, instruction, bindings, gate)):
        raise ProtocolError("target seed plan is incomplete")
    candidates = pool.get("requested_seeds")
    if (
        not isinstance(candidates, list)
        or len(candidates) < TOTAL
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in candidates)
        or len(set(candidates)) != len(candidates)
        or pool.get("requested_identity_set_sha256") != canonical_sha256(candidates)
    ):
        raise ProtocolError("candidate pool identity changed")
    expected_candidates = [
        int(pool.get("start", -1)) + int(pool.get("step", -1)) * index
        for index in range(int(pool.get("count", -1)))
    ]
    if candidates != expected_candidates:
        raise ProtocolError("candidate pool is not the frozen arithmetic order")
    if split.get("selection_rule") != "first_530_stable_resets_in_preregistered_candidate_order":
        raise ProtocolError("split selection rule changed")
    receipt = instruction.get("semantics_receipt")
    if (
        instruction.get("instruction") != INSTRUCTION
        or instruction.get("episode_info_list_used") is not False
        or not isinstance(receipt, Mapping)
        or verify_signature(receipt, "receipt_sha256", "instruction semantics receipt")
        != receipt.get("receipt_sha256")
        or {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        != SEMANTICS_RECEIPT_UNSIGNED
    ):
        raise ProtocolError("explicit instruction contract changed")
    required_shas = (
        "d250_identity_file_sha256",
        "d250_identity_sets_sha256",
        "heldout_identity_set_sha256",
        "upstream_v7_seed_manifest_file_sha256",
        "upstream_v7_seed_manifest_payload_sha256",
        "upstream_data_audit_file_sha256",
        "resolver_implementation_sha256",
        "reset_adapter_implementation_sha256",
    )
    if any(not is_sha(bindings.get(key)) for key in required_shas):
        raise ProtocolError("plan identity/SHA bindings are incomplete")
    if gate != {
        "authorized_by_plan": False,
        "required_authorization_format": AUTHORIZATION_FORMAT,
        "candidate_pool_disjoint_attestation_required_before_environment_construction": True,
        "reset_only": True,
        "environment_step_allowed": False,
        "policy_import_or_forward_allowed": False,
        "reward_success_event_or_outcome_read_allowed": False,
    }:
        raise ProtocolError("reset-only execution ceiling changed")
    return {
        "plan_sha256": plan_sha,
        "candidates": list(candidates),
        "instruction": instruction["instruction"],
        "semantics_receipt": dict(receipt),
        "bindings": dict(bindings),
    }


def validate_disjoint_attestation(
    value: Mapping[str, Any],
    *,
    heldout_identity_set_sha256: str,
    target_identity_set_sha256: str,
    target_role: str,
) -> str:
    attestation_sha = verify_signature(value, "attestation_sha256", target_role)
    if value != {
        "format": "etsf_private_identity_disjoint_attestation_v1",
        "status": ATTESTATION_STATUS,
        "target_role": target_role,
        "heldout_identity_set_sha256": heldout_identity_set_sha256,
        "target_identity_set_sha256": target_identity_set_sha256,
        "intersection_count": 0,
        "sensitive_identities_included": False,
        "attestation_sha256": attestation_sha,
    }:
        raise ProtocolError(f"{target_role} disjoint attestation is invalid")
    return attestation_sha


def validate_execution_authorization(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    auth_sha = verify_signature(value, "authorization_sha256", "reset authorization")
    decoded = validate_plan(plan)
    bindings = decoded["bindings"]
    permissions = value.get("permissions")
    attestation = value.get("candidate_pool_disjoint_attestation")
    runtime_contract = value.get("runtime_contract")
    expected_fields = {
        "format", "status", "plan_file_sha256", "plan_sha256",
        "resolver_implementation_sha256", "reset_adapter_implementation_sha256",
        "permissions", "runtime_contract", "candidate_pool_disjoint_attestation",
        "authorization_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != AUTHORIZATION_FORMAT
        or value.get("status") != "authorized_reset_only_after_private_disjoint_check"
        or value.get("plan_file_sha256") != plan_file_sha256
        or value.get("plan_sha256") != decoded["plan_sha256"]
        or value.get("resolver_implementation_sha256") != bindings["resolver_implementation_sha256"]
        or value.get("reset_adapter_implementation_sha256")
        != bindings["reset_adapter_implementation_sha256"]
        or permissions
        != {
            "environment_construct_allowed": True,
            "reset_only": True,
            "environment_step_allowed": False,
            "policy_import_or_forward_allowed": False,
            "reward_success_event_or_outcome_read_allowed": False,
        }
        or not isinstance(runtime_contract, Mapping)
        or not isinstance(attestation, Mapping)
    ):
        raise ProtocolError("reset authorization scope/bindings changed")
    validate_disjoint_attestation(
        attestation,
        heldout_identity_set_sha256=bindings["heldout_identity_set_sha256"],
        target_identity_set_sha256=plan["candidate_pool"]["requested_identity_set_sha256"],
        target_role="preregistered_reset_candidate_pool",
    )
    try:
        from smolvla_piper_schema6_runtime_adapter_v2 import validate_runtime_contract

        decoded_runtime = validate_runtime_contract(runtime_contract)
    except Exception as error:
        raise ProtocolError("reset authorization runtime contract is invalid") from error
    return {
        "authorization_sha256": auth_sha,
        "runtime_contract_sha256": decoded_runtime["runtime_contract_sha256"],
        "runtime_contract": decoded_runtime,
    }


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "task",
            "actor_id",
            "target_body",
            "split",
            "ordinal",
            "requested_seed",
            "resolved_seed",
            "instruction_sha256",
            "instruction_semantics_receipt_sha256",
            "initial_scene_state_sha256",
            "initial_measured_joint_state_sha256",
            "initial_commanded_drive_target_sha256",
        )
    }


def validate_reset_receipt(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], plan_file_sha256: str
) -> dict[str, Any]:
    receipt_sha = verify_signature(value, "reset_receipt_sha256", "reset receipt")
    decoded = validate_plan(plan)
    if (
        value.get("format") != RESET_RECEIPT_FORMAT
        or value.get("status") != RESET_RECEIPT_STATUS
        or value.get("plan_file_sha256") != plan_file_sha256
        or value.get("plan_sha256") != decoded["plan_sha256"]
        or value.get("labels_or_outcomes_read") is not False
        or value.get("environment_step_calls") != 0
        or value.get("policy_import_or_forward_calls") != 0
        or not is_sha(value.get("runtime_contract_sha256"))
    ):
        raise ProtocolError("reset receipt capability/scope changed")
    rows = value.get("rows")
    attempts = value.get("attempts")
    if not isinstance(rows, list) or len(rows) != TOTAL or not isinstance(attempts, list):
        raise ProtocolError("reset receipt must contain exactly 530 selected rows")
    expected_candidates = decoded["candidates"]
    attempted = [item.get("requested_seed") for item in attempts if isinstance(item, Mapping)]
    if attempted != expected_candidates[: len(attempted)] or len(attempted) < TOTAL:
        raise ProtocolError("reset attempts did not follow the preregistered candidate order")
    expected_split = ["adaptation"] * ADAPTATION + ["validation"] * VALIDATION + ["evaluation"] * EVALUATION
    expected_ordinals = list(range(ADAPTATION)) + list(range(VALIDATION)) + list(range(EVALUATION))
    for index, (row, split, ordinal) in enumerate(zip(rows, expected_split, expected_ordinals)):
        if not isinstance(row, Mapping):
            raise ProtocolError(f"reset row {index} is invalid")
        if (
            row.get("task") != TASK
            or row.get("actor_id") != TARGET_ACTOR_ID
            or row.get("target_body") != TARGET_BODY
            or row.get("split") != split
            or row.get("ordinal") != ordinal
            or row.get("global_ordinal") != index
            or row.get("instruction") != INSTRUCTION
            or row.get("instruction_semantics_receipt") != decoded["semantics_receipt"]
            or row.get("instruction_sha256")
            != hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
            or row.get("instruction_semantics_receipt_sha256")
            != decoded["semantics_receipt"]["receipt_sha256"]
            or any(
                not is_sha(row.get(key))
                for key in (
                    "initial_scene_state_sha256",
                    "initial_measured_joint_state_sha256",
                    "initial_commanded_drive_target_sha256",
                    "pair_id",
                )
            )
            or row.get("pair_id") != canonical_sha256(_row_identity(row))
        ):
            raise ProtocolError(f"reset row {index} identity/instruction changed")
    requested = [int(row["requested_seed"]) for row in rows]
    resolved = [int(row["resolved_seed"]) for row in rows]
    if len(set(requested)) != TOTAL or len(set(resolved)) != TOTAL:
        raise ProtocolError("selected requested/resolved target identities must be unique")
    return {
        "reset_receipt_sha256": receipt_sha,
        "rows": [dict(row) for row in rows],
        "requested": requested,
        "resolved": resolved,
        "identity_set_sha256": canonical_sha256({"requested": requested, "resolved": resolved}),
    }


def freeze_manifest(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    authorization: Mapping[str, Any],
    authorization_file_sha256: str,
    reset_receipt: Mapping[str, Any],
    reset_receipt_file_sha256: str,
    selected_identity_disjoint_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    decoded_plan = validate_plan(plan)
    decoded_auth = validate_execution_authorization(
        authorization, plan=plan, plan_file_sha256=plan_file_sha256
    )
    decoded = validate_reset_receipt(
        reset_receipt, plan=plan, plan_file_sha256=plan_file_sha256
    )
    if (
        not is_sha(plan_file_sha256)
        or not is_sha(authorization_file_sha256)
        or not is_sha(reset_receipt_file_sha256)
        or reset_receipt.get("authorization_file_sha256") != authorization_file_sha256
        or reset_receipt.get("authorization_sha256") != decoded_auth["authorization_sha256"]
        or reset_receipt.get("runtime_contract_sha256")
        != decoded_auth["runtime_contract_sha256"]
    ):
        raise ProtocolError("reset receipt does not bind the authenticated execution authorization")
    attestation_sha = validate_disjoint_attestation(
        selected_identity_disjoint_attestation,
        heldout_identity_set_sha256=decoded_plan["bindings"]["heldout_identity_set_sha256"],
        target_identity_set_sha256=decoded["identity_set_sha256"],
        target_role="selected_requested_and_resolved_target_identities",
    )
    by_split = {
        split: [row for row in decoded["rows"] if row["split"] == split]
        for split in ("adaptation", "validation", "evaluation")
    }
    manifest: dict[str, Any] = {
        "format": FINAL_FORMAT,
        "status": FINAL_STATUS,
        "task": TASK,
        "actor_id": TARGET_ACTOR_ID,
        "source_body": SOURCE_BODY,
        "target_body": TARGET_BODY,
        "purpose": "nonfresh_development_only_no_confirmation_claim",
        "label_access_contract": LABEL_ACCESS_CONTRACT,
        "instruction_contract": plan["instruction_contract"],
        "splits": by_split,
        "provenance": {
            "plan_file_sha256": plan_file_sha256,
            "plan_sha256": decoded_plan["plan_sha256"],
            "authorization_file_sha256": authorization_file_sha256,
            "authorization_sha256": decoded_auth["authorization_sha256"],
            "runtime_contract_sha256": decoded_auth["runtime_contract_sha256"],
            "reset_receipt_file_sha256": reset_receipt_file_sha256,
            "reset_receipt_sha256": decoded["reset_receipt_sha256"],
            "resolver_implementation_sha256": decoded_plan["bindings"]["resolver_implementation_sha256"],
            "reset_adapter_implementation_sha256": decoded_plan["bindings"]["reset_adapter_implementation_sha256"],
        },
        "d250_exclusion": {
            "identity_manifest_file_sha256": decoded_plan["bindings"]["d250_identity_file_sha256"],
            "identity_sets_sha256": decoded_plan["bindings"]["d250_identity_sets_sha256"],
            "intersection_count": 0,
        },
        "heldout_exclusion_attestation": {
            "status": ATTESTATION_STATUS,
            "heldout_identity_set_sha256": decoded_plan["bindings"]["heldout_identity_set_sha256"],
            "target_identity_set_sha256": decoded["identity_set_sha256"],
            "intersection_count": 0,
            "sensitive_identities_included": False,
            "attestation_sha256": attestation_sha,
        },
        "capability_receipt": {
            "environment_reset_only": True,
            "environment_step_calls": 0,
            "policy_import_or_forward_calls": 0,
            "labels_or_outcomes_read": False,
            "policy_execution_authorized_by_manifest": False,
        },
    }
    return signed(manifest, "seed_manifest_sha256")


def legacy_v1_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project v2 to the exact v1 schema consumed by the existing prereg freezer."""

    verify_signature(manifest, "seed_manifest_sha256", "target seed manifest v2")
    if manifest.get("format") != FINAL_FORMAT or manifest.get("status") != FINAL_STATUS:
        raise ProtocolError("cannot project an unrecognized target manifest")
    projected_splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("adaptation", "validation", "evaluation"):
        projected_splits[split] = []
        for row in manifest["splits"][split]:
            projected_splits[split].append(
                {
                    "ordinal": row["ordinal"],
                    "pair_id": row["pair_id"],
                    "requested_seed": row["requested_seed"],
                    "resolved_seed": row["resolved_seed"],
                    "instruction_sha256": row["instruction_sha256"],
                    "instruction_semantics_receipt_sha256": row["instruction_semantics_receipt_sha256"],
                    "initial_scene_state_sha256": row["initial_scene_state_sha256"],
                    "initial_measured_joint_state_sha256": row["initial_measured_joint_state_sha256"],
                    "initial_commanded_drive_target_sha256": row["initial_commanded_drive_target_sha256"],
                }
            )
    attestation = manifest["heldout_exclusion_attestation"]
    value: dict[str, Any] = {
        "format": LEGACY_FORMAT,
        "status": LEGACY_STATUS,
        "task": TASK,
        "actor_id": TARGET_ACTOR_ID,
        "source_body": SOURCE_BODY,
        "target_body": TARGET_BODY,
        "purpose": "nonfresh_development_only_no_confirmation_claim",
        "label_access_contract": LABEL_ACCESS_CONTRACT,
        "splits": projected_splits,
        "d250_exclusion": manifest["d250_exclusion"],
        "heldout_exclusion_attestation": {
            key: attestation[key]
            for key in (
                "status",
                "heldout_identity_set_sha256",
                "target_identity_set_sha256",
                "intersection_count",
                "sensitive_identities_included",
            )
        },
        "instruction_contract": {
            "mode": "explicit_frozen_per_pair_instruction",
            "episode_info_list_used": False,
            "semantic_receipt_required": True,
            "same_instruction_for_both_conditions": True,
        },
    }
    return signed(value, "seed_manifest_sha256")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="freeze an offline unresolved target seed plan")
    plan.add_argument("--d250-identity", type=Path, required=True)
    plan.add_argument("--heldout-identity-set-sha256", required=True)
    plan.add_argument("--upstream-v7-seed-manifest-file-sha256", required=True)
    plan.add_argument("--upstream-v7-seed-manifest-payload-sha256", required=True)
    plan.add_argument("--upstream-data-audit-file-sha256", required=True)
    plan.add_argument("--resolver-implementation", type=Path, required=True)
    plan.add_argument("--reset-adapter-implementation", type=Path, required=True)
    plan.add_argument("--candidate-start", type=int, default=CANDIDATE_START)
    plan.add_argument("--candidate-count", type=int, default=CANDIDATE_COUNT)
    plan.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze", help="freeze reset receipt after private exclusion check")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--authorization", type=Path, required=True)
    freeze.add_argument("--reset-receipt", type=Path, required=True)
    freeze.add_argument("--selected-identity-disjoint-attestation", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--legacy-v1-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "plan":
        d250_path = reject_sensitive_path(args.d250_identity, "D250 identity", must_exist=True)
        resolver_path = reject_sensitive_path(args.resolver_implementation, "resolver implementation", must_exist=True)
        adapter_path = reject_sensitive_path(args.reset_adapter_implementation, "reset adapter implementation", must_exist=True)
        result = build_plan(
            d250_identity=load_json(d250_path, "D250 identity"),
            d250_identity_file_sha256=file_sha256(d250_path),
            heldout_identity_set_sha256=args.heldout_identity_set_sha256,
            upstream_v7_seed_manifest_file_sha256=args.upstream_v7_seed_manifest_file_sha256,
            upstream_v7_seed_manifest_payload_sha256=args.upstream_v7_seed_manifest_payload_sha256,
            upstream_data_audit_file_sha256=args.upstream_data_audit_file_sha256,
            resolver_implementation_sha256=file_sha256(resolver_path),
            reset_adapter_implementation_sha256=file_sha256(adapter_path),
            candidate_start=args.candidate_start,
            candidate_count=args.candidate_count,
        )
        immutable_json(args.output, result)
    else:
        plan_path = reject_sensitive_path(args.plan, "target seed plan", must_exist=True)
        authorization_path = reject_sensitive_path(
            args.authorization, "reset authorization", must_exist=True
        )
        receipt_path = reject_sensitive_path(args.reset_receipt, "reset receipt", must_exist=True)
        attestation_path = reject_sensitive_path(
            args.selected_identity_disjoint_attestation,
            "selected identity disjoint attestation",
            must_exist=True,
        )
        result = freeze_manifest(
            plan=load_json(plan_path, "target seed plan"),
            plan_file_sha256=file_sha256(plan_path),
            authorization=load_json(authorization_path, "reset authorization"),
            authorization_file_sha256=file_sha256(authorization_path),
            reset_receipt=load_json(receipt_path, "reset receipt"),
            reset_receipt_file_sha256=file_sha256(receipt_path),
            selected_identity_disjoint_attestation=load_json(attestation_path, "selected identity disjoint attestation"),
        )
        immutable_json(args.output, result)
        if args.legacy_v1_output is not None:
            immutable_json(args.legacy_v1_output, legacy_v1_projection(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
