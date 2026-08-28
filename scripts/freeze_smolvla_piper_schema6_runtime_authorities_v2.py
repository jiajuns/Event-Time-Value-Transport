#!/usr/bin/env python3
"""Create signed Schema6 runtime, reset-only, and Phase-2 authorities.

This freezer is intentionally offline: it hashes existing code/model/runtime
inputs and never imports a policy, constructs an environment, or opens HDF5.
Private held-out identities are represented only by a signed disjointness
attestation supplied by the authorized operator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from preregister_smolvla_piper_schema6_multiseed_collection_v2 import (
    canonical_sha256,
    file_sha256,
    validate_preregistration,
)
from run_smolvla_piper_schema6_multiseed_v2 import (
    EXECUTION_AUTHORITY_FORMAT,
    EXECUTION_AUTHORITY_STATUS,
)
from smolvla_piper_schema6_runtime_adapter_v2 import (
    RUNTIME_CONTRACT_FORMAT,
    RUNTIME_CONTRACT_STATUS,
    DEFAULT_MEASURED_CHANNEL,
    RUNTIME_ROOT_KEYS,
    RUNTIME_SOURCE_KEYS,
    directory_tree_sha256,
    canonical_sha256 as runtime_canonical_sha256,
    validate_runtime_contract,
)
from smolvla_piper_target_seed_manifest import (
    AUTHORIZATION_FORMAT,
    file_sha256 as target_file_sha256,
    canonical_sha256 as target_canonical_sha256,
    validate_disjoint_attestation,
    validate_plan,
)


class AuthorityFreezerError(RuntimeError):
    """An offline authority input is incomplete or mutable."""


def _load_object(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AuthorityFreezerError(f"{role} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuthorityFreezerError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise AuthorityFreezerError(f"{role} must be a JSON object")
    return value


def _real_file(path: str | os.PathLike[str], role: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or value.is_symlink():
        raise AuthorityFreezerError(f"{role} is not a regular non-symlink file")
    return value


def _create_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze_runtime_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    exact = {
        "runtime_roots", "runtime_source_paths", "eval_seed_registry_path",
        "gpu_index", "max_episode_steps",
        "piper_action_bounds", "reset_scratch_path",
    }
    if set(spec) != exact:
        raise AuthorityFreezerError("runtime contract spec fields changed")
    roots = spec["runtime_roots"]
    sources = spec["runtime_source_paths"]
    if not isinstance(roots, Mapping) or set(roots) != RUNTIME_ROOT_KEYS:
        raise AuthorityFreezerError("runtime root spec is incomplete")
    if not isinstance(sources, Mapping) or set(sources) != RUNTIME_SOURCE_KEYS:
        raise AuthorityFreezerError("runtime source spec is incomplete")
    resolved_roots = {key: str(Path(str(value)).resolve()) for key, value in roots.items()}
    source_rows = {}
    for role, raw in sources.items():
        path = _real_file(str(raw), f"runtime source {role}")
        source_rows[role] = {"path": str(path), "sha256": file_sha256(path)}
    registry = _real_file(str(spec["eval_seed_registry_path"]), "eval seed registry")
    contract: dict[str, Any] = {
        "format": RUNTIME_CONTRACT_FORMAT,
        "status": RUNTIME_CONTRACT_STATUS,
        "runtime_roots": resolved_roots,
        "runtime_source_artifacts": source_rows,
        "eval_seed_registry": {"path": str(registry), "sha256": file_sha256(registry)},
        "measured_joint_state_channel": DEFAULT_MEASURED_CHANNEL,
        "gpu_index": spec["gpu_index"],
        "max_episode_steps": spec["max_episode_steps"],
        "offline_model_loading": True,
        "piper_action_bounds": spec["piper_action_bounds"],
        "model_tree_sha256": directory_tree_sha256(Path(resolved_roots["model_path"])),
        "vlm_metadata_tree_sha256": directory_tree_sha256(
            Path(resolved_roots["vlm_metadata_path"])
        ),
        "reset_scratch_path": str(Path(str(spec["reset_scratch_path"])).resolve()),
        "test_or_evaluation_execution_authorized": False,
        "fresh_or_confirmation_inputs_accepted": False,
    }
    contract["runtime_contract_sha256"] = runtime_canonical_sha256(contract)
    return validate_runtime_contract(contract)


def freeze_phase2_authority(
    *, preregistration_path: Path, runner_path: Path, runtime_adapter_path: Path,
    move_can_pot_source_path: Path, runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    prereg_path = _real_file(preregistration_path, "preregistration")
    prereg = _load_object(prereg_path, "preregistration")
    decoded = validate_preregistration(prereg)
    runner = _real_file(runner_path, "runner")
    adapter = _real_file(runtime_adapter_path, "runtime adapter")
    move_source = _real_file(move_can_pot_source_path, "move_can_pot source")
    decoded_runtime = validate_runtime_contract(runtime_contract)
    bound_move = decoded_runtime["runtime_source_artifacts"]["robotwin_move_can_pot"]
    if Path(bound_move["path"]).resolve() != move_source or bound_move["sha256"] != file_sha256(move_source):
        raise AuthorityFreezerError("move_can_pot differs between authority and runtime contract")
    authority: dict[str, Any] = {
        "format": EXECUTION_AUTHORITY_FORMAT,
        "status": EXECUTION_AUTHORITY_STATUS,
        "production_execution_authorized": True,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": file_sha256(prereg_path),
        "preregistration_sha256": decoded["preregistration_sha256"],
        "runner_path": str(runner),
        "runner_file_sha256": file_sha256(runner),
        "runtime_adapter_path": str(adapter),
        "runtime_adapter_file_sha256": file_sha256(adapter),
        "move_can_pot_source_path": str(move_source),
        "move_can_pot_source_file_sha256": file_sha256(move_source),
        "runtime_contract": decoded_runtime,
        "evaluation_commands_authorized": 0,
        "test_inputs_read": False,
        "fresh_inputs_accepted": False,
        "confirmation_inputs_accepted": False,
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    return authority


def freeze_reset_authority(
    *, plan_path: Path, runtime_contract: Mapping[str, Any],
    candidate_pool_disjoint_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    target_plan = _real_file(plan_path, "target reset plan")
    plan = _load_object(target_plan, "target reset plan")
    decoded = validate_plan(plan)
    attestation = dict(candidate_pool_disjoint_attestation)
    validate_disjoint_attestation(
        attestation,
        heldout_identity_set_sha256=decoded["bindings"]["heldout_identity_set_sha256"],
        target_identity_set_sha256=plan["candidate_pool"]["requested_identity_set_sha256"],
        target_role="preregistered_reset_candidate_pool",
    )
    authority: dict[str, Any] = {
        "format": AUTHORIZATION_FORMAT,
        "status": "authorized_reset_only_after_private_disjoint_check",
        "plan_file_sha256": target_file_sha256(target_plan),
        "plan_sha256": decoded["plan_sha256"],
        "resolver_implementation_sha256": decoded["bindings"]["resolver_implementation_sha256"],
        "reset_adapter_implementation_sha256": decoded["bindings"][
            "reset_adapter_implementation_sha256"
        ],
        "permissions": {
            "environment_construct_allowed": True,
            "reset_only": True,
            "environment_step_allowed": False,
            "policy_import_or_forward_allowed": False,
            "reward_success_event_or_outcome_read_allowed": False,
        },
        "runtime_contract": validate_runtime_contract(runtime_contract),
        "candidate_pool_disjoint_attestation": attestation,
    }
    authority["authorization_sha256"] = target_canonical_sha256(authority)
    return authority


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runtime = sub.add_parser("runtime-contract")
    runtime.add_argument("--spec", type=Path, required=True)
    phase2 = sub.add_parser("phase2-authority")
    phase2.add_argument("--preregistration", type=Path, required=True)
    phase2.add_argument("--runner", type=Path, required=True)
    phase2.add_argument("--runtime-adapter", type=Path, required=True)
    phase2.add_argument("--move-can-pot-source", type=Path, required=True)
    phase2.add_argument("--runtime-contract", type=Path, required=True)
    reset = sub.add_parser("reset-authority")
    reset.add_argument("--plan", type=Path, required=True)
    reset.add_argument("--runtime-contract", type=Path, required=True)
    reset.add_argument("--candidate-pool-disjoint-attestation", type=Path, required=True)
    for child in (runtime, phase2, reset):
        child.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "runtime-contract":
        result = freeze_runtime_contract(_load_object(args.spec, "runtime spec"))
    elif args.command == "phase2-authority":
        result = freeze_phase2_authority(
            preregistration_path=args.preregistration,
            runner_path=args.runner,
            runtime_adapter_path=args.runtime_adapter,
            move_can_pot_source_path=args.move_can_pot_source,
            runtime_contract=_load_object(args.runtime_contract, "runtime contract"),
        )
    else:
        result = freeze_reset_authority(
            plan_path=args.plan,
            runtime_contract=_load_object(args.runtime_contract, "runtime contract"),
            candidate_pool_disjoint_attestation=_load_object(
                args.candidate_pool_disjoint_attestation, "candidate pool attestation"
            ),
        )
    _create_once(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuthorityFreezerError", "freeze_phase2_authority", "freeze_reset_authority",
    "freeze_runtime_contract",
]
