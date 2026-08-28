#!/usr/bin/env python3
"""Resolve a preregistered Piper seed pool using reset-only observations.

The environment adapter is imported and constructed only after the signed plan,
the signed authorization, all implementation hashes, and the private candidate
pool exclusion attestation have passed.  This module has no policy interface
and contains no environment-step call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from preregister_smolvla_piper_v7_paired_development import (
    TARGET_ACTOR_ID,
    TARGET_BODY,
    TASK,
    ProtocolError,
    canonical_sha256,
    file_sha256,
)
from smolvla_piper_target_seed_manifest import (
    ADAPTATION,
    EVALUATION,
    INSTRUCTION,
    RESET_RECEIPT_FORMAT,
    RESET_RECEIPT_STATUS,
    TOTAL,
    VALIDATION,
    _row_identity,
    immutable_json,
    load_json,
    reject_sensitive_path,
    signed,
    validate_execution_authorization,
    validate_plan,
)


FORBIDDEN_RESULT_KEYS = {
    "action",
    "actions",
    "reward",
    "rewards",
    "success",
    "successes",
    "event",
    "events",
    "outcome",
    "outcomes",
    "trajectory",
    "trajectories",
    "policy",
    "prediction",
    "predictions",
}


def _forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).casefold()
            location = f"{prefix}.{key}" if prefix else str(key)
            if name in FORBIDDEN_RESULT_KEYS or set(name.split("_")) & FORBIDDEN_RESULT_KEYS:
                result.append(location)
            result.extend(_forbidden_keys(child, location))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.extend(_forbidden_keys(child, f"{prefix}[{index}]"))
    return result


def _json_numeric(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ProtocolError("reset observation contains non-finite values")
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_numeric(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numeric(child) for child in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ProtocolError("reset observation contains non-finite values")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ProtocolError(f"reset observation contains unsupported value {type(value).__name__}")


def array_sha256(value: Any, *, role: str) -> str:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype="<f8")
    if array.shape != (14,) or not np.all(np.isfinite(array)):
        raise ProtocolError(f"{role} must be a finite 14-vector")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"etsf-array-v1\0")
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def scene_sha256(value: Any) -> str:
    normalized = _json_numeric(value)
    if not isinstance(normalized, Mapping) or set(normalized) != {"can_pose", "pot_pose"}:
        raise ProtocolError("scene state must contain exactly can_pose and pot_pose")
    for key in ("can_pose", "pot_pose"):
        pose = np.asarray(normalized[key], dtype=np.float64)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ProtocolError(f"scene {key} must be a finite xyz+quaternion 7-vector")
    return canonical_sha256(normalized)


def _split_for_global(index: int) -> tuple[str, int, str]:
    if index < ADAPTATION:
        return (
            "adaptation",
            index,
            "direct_actor_only_operational" if index < 20 else "adapter_development",
        )
    if index < ADAPTATION + VALIDATION:
        return "validation", index - ADAPTATION, "frozen_selection_validation"
    return "evaluation", index - ADAPTATION - VALIDATION, "sealed_paired_evaluation"


def resolve_with_adapter(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    authorization: Mapping[str, Any],
    authorization_file_sha256: str,
    reset_once: Callable[[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Collect exactly 530 stable reset identities; never run a policy or action."""

    decoded_plan = validate_plan(plan)
    decoded_auth = validate_execution_authorization(
        authorization, plan=plan, plan_file_sha256=plan_file_sha256
    )
    attempts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    resolved_seen: set[int] = set()
    instruction = decoded_plan["instruction"]
    semantic_receipt = decoded_plan["semantics_receipt"]
    instruction_sha = hashlib.sha256(instruction.encode("utf-8")).hexdigest()

    for requested in decoded_plan["candidates"]:
        if len(rows) == TOTAL:
            break
        raw = reset_once(int(requested), instruction)
        if not isinstance(raw, Mapping):
            raise ProtocolError("reset adapter must return a mapping")
        forbidden = _forbidden_keys(raw)
        if forbidden:
            raise ProtocolError(f"reset adapter exposed forbidden policy/label fields: {forbidden[:3]}")
        setup_status = raw.get("setup_status")
        if setup_status == "unstable":
            attempts.append({"requested_seed": requested, "setup_status": "unstable"})
            continue
        if setup_status != "stable":
            raise ProtocolError("reset adapter setup_status must be stable or unstable")
        if raw.get("requested_seed") != requested:
            raise ProtocolError("reset adapter changed the requested seed")
        resolved = raw.get("resolved_seed")
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
            raise ProtocolError("reset adapter resolved seed is invalid")
        # A conforming adapter performs one exact setup attempt.  This gate catches
        # stock RoboTwin's implicit retry loop, which must not be used here.
        if resolved != requested:
            raise ProtocolError("reset adapter performed an unapproved internal seed retry")
        if resolved in resolved_seen:
            attempts.append(
                {"requested_seed": requested, "resolved_seed": resolved, "setup_status": "duplicate"}
            )
            continue
        if raw.get("instruction_observed") != instruction:
            raise ProtocolError("reset did not preserve the explicit frozen instruction")
        scene_hash = scene_sha256(raw.get("scene_state"))
        measured_hash = array_sha256(raw.get("measured_joint_state"), role="measured joint state")
        commanded_hash = array_sha256(raw.get("commanded_drive_target"), role="commanded drive target")
        global_ordinal = len(rows)
        split, ordinal, stage_role = _split_for_global(global_ordinal)
        row: dict[str, Any] = {
            "task": TASK,
            "actor_id": TARGET_ACTOR_ID,
            "target_body": TARGET_BODY,
            "global_ordinal": global_ordinal,
            "split": split,
            "ordinal": ordinal,
            "stage_role": stage_role,
            "requested_seed": requested,
            "resolved_seed": resolved,
            "instruction": instruction,
            "instruction_sha256": instruction_sha,
            "instruction_semantics_receipt": semantic_receipt,
            "instruction_semantics_receipt_sha256": semantic_receipt["receipt_sha256"],
            "initial_scene_state_sha256": scene_hash,
            "initial_measured_joint_state_sha256": measured_hash,
            "initial_commanded_drive_target_sha256": commanded_hash,
        }
        row["pair_id"] = canonical_sha256(_row_identity(row))
        rows.append(row)
        resolved_seen.add(resolved)
        attempts.append(
            {
                "requested_seed": requested,
                "resolved_seed": resolved,
                "setup_status": "stable_selected",
                "pair_id": row["pair_id"],
            }
        )
    if len(rows) != TOTAL:
        raise ProtocolError(
            f"candidate pool exhausted after {len(rows)} stable unique resets; no partial receipt frozen"
        )
    receipt: dict[str, Any] = {
        "format": RESET_RECEIPT_FORMAT,
        "status": RESET_RECEIPT_STATUS,
        "task": TASK,
        "actor_id": TARGET_ACTOR_ID,
        "target_body": TARGET_BODY,
        "plan_file_sha256": plan_file_sha256,
        "plan_sha256": decoded_plan["plan_sha256"],
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_sha256": decoded_auth["authorization_sha256"],
        "runtime_contract_sha256": decoded_auth["runtime_contract_sha256"],
        "attempts": attempts,
        "rows": rows,
        "split_counts": {"adaptation": ADAPTATION, "validation": VALIDATION, "evaluation": EVALUATION},
        "first_20_adaptation_role": "direct_actor_only_operational",
        "labels_or_outcomes_read": False,
        "environment_step_calls": 0,
        "policy_import_or_forward_calls": 0,
    }
    return signed(receipt, "reset_receipt_sha256")


def _load_factory(path: Path, symbol: str) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location("_etsf_target_reset_adapter", path)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot load reset adapter implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, symbol, None)
    if not callable(factory):
        raise ProtocolError(f"reset adapter lacks callable {symbol}")
    return factory


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--reset-adapter-implementation", type=Path, required=True)
    parser.add_argument("--reset-adapter-factory", default="build_reset_only_adapter")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # Nothing capable of constructing an environment is imported before every
    # offline artifact and implementation binding below has passed.
    plan_path = reject_sensitive_path(args.plan, "target seed plan", must_exist=True)
    auth_path = reject_sensitive_path(args.authorization, "reset authorization", must_exist=True)
    adapter_path = reject_sensitive_path(
        args.reset_adapter_implementation, "reset adapter implementation", must_exist=True
    )
    plan = load_json(plan_path, "target seed plan")
    authorization = load_json(auth_path, "reset authorization")
    decoded = validate_plan(plan)
    if file_sha256(Path(__file__).resolve()) != decoded["bindings"]["resolver_implementation_sha256"]:
        raise ProtocolError("running resolver implementation differs from the frozen plan")
    if file_sha256(adapter_path) != decoded["bindings"]["reset_adapter_implementation_sha256"]:
        raise ProtocolError("reset adapter implementation differs from the frozen plan")
    validate_execution_authorization(
        authorization, plan=plan, plan_file_sha256=file_sha256(plan_path)
    )
    factory = _load_factory(adapter_path, args.reset_adapter_factory)
    adapter = factory(plan=plan, authorization=authorization)
    reset_once = getattr(adapter, "reset_once", None)
    close = getattr(adapter, "close", None)
    if not callable(reset_once) or not callable(close):
        raise ProtocolError("reset adapter must expose reset_once and close")
    try:
        receipt = resolve_with_adapter(
            plan=plan,
            plan_file_sha256=file_sha256(plan_path),
            authorization=authorization,
            authorization_file_sha256=file_sha256(auth_path),
            reset_once=reset_once,
        )
    finally:
        close()
    immutable_json(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
