#!/usr/bin/env python3
"""Freeze an independent external execution authority for evaluation400.

The freezer consumes only signed JSON.  It authenticates and reconstructs the
pre-outcome identity bridge, then binds a separately supplied independent
decision.  It never imports a policy, opens a checkpoint/HDF/trajectory/label,
or executes either paired condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge_v2


FORMAT = bridge_v2.EXTERNAL_AUTHORITY_FORMAT
STATUS = "authorized_for_external_evaluation400_paired_executor_only"
DECISION_FORMAT = (
    "etsf_smolvla_piper_evaluation400_independent_execution_decision_v2"
)
DECISION_STATUS = "independent_preoutcome_execution_decision_frozen"
PAIR_COUNT = bridge_v2.EVALUATION_GROUPS
MEMBER_COUNT = bridge_v2.MEMBER_COUNT
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
FORBIDDEN_PATH_TOKENS = ("fresh", "confirmation", "trajectory", "label")
BRIDGE_DEPENDENCIES = (
    "target_manifest",
    "selected_identity_attestation",
    "ensemble_manifest",
    "calibration",
    "head_support",
    "calibration_receipt",
    "policy_bridge_receipt",
)
R7H_SHARED_FIELDS = {
    "training_manifest_sha256",
    "split_sha256",
    "source_ensemble_contract_sha256",
    "prediction_contract_sha256",
}


class ExternalAuthorityV2Error(RuntimeError):
    """An authority identity, independence, or deployment binding failed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def file_sha256(path: Path) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise ExternalAuthorityV2Error("HDF input is forbidden")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sensitive(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            return True
        if lowered == "test" or lowered.startswith(("test_", "test-")):
            return True
    return False


def safe_file(path: Path, role: str, *, json_only: bool = True) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if _sensitive(PurePath(lexical)) or lexical.is_symlink():
        raise ExternalAuthorityV2Error(f"{role} path is forbidden or a symlink")
    resolved = lexical.resolve(strict=True)
    if _sensitive(PurePath(resolved)) or not resolved.is_file() or resolved.is_symlink():
        raise ExternalAuthorityV2Error(f"{role} must be a safe materialized file")
    if resolved.suffix.casefold() in HDF_SUFFIXES:
        raise ExternalAuthorityV2Error("HDF input is forbidden")
    if json_only and resolved.suffix.casefold() != ".json":
        raise ExternalAuthorityV2Error(f"{role} must be JSON")
    return resolved


def safe_new_json(path: Path, role: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if _sensitive(PurePath(lexical)) or lexical.suffix.casefold() != ".json":
        raise ExternalAuthorityV2Error(f"{role} path is forbidden or not JSON")
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError(lexical)
    parent = lexical.parent.resolve(strict=True)
    if _sensitive(PurePath(parent)) or not parent.is_dir():
        raise ExternalAuthorityV2Error(f"{role} parent is forbidden")
    return lexical


def read_bound_json(
    path: Path, expected_file_sha256: str, role: str
) -> tuple[Path, dict[str, Any]]:
    if not is_sha(expected_file_sha256):
        raise ExternalAuthorityV2Error(f"{role} expected file SHA is invalid")
    resolved = safe_file(path, role)
    before = file_sha256(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExternalAuthorityV2Error(f"{role} is not valid JSON") from error
    after = file_sha256(resolved)
    if before != after or before != expected_file_sha256:
        raise ExternalAuthorityV2Error(f"{role} file SHA changed")
    if not isinstance(value, dict):
        raise ExternalAuthorityV2Error(f"{role} must contain an object")
    return resolved, value


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise ExternalAuthorityV2Error(f"{role} logical SHA mismatch")
    return str(recorded)


def _strict_bridge_pairs(value: Mapping[str, Any]) -> None:
    scope = value.get("scope")
    pairs = value.get("pairs")
    if (
        not isinstance(scope, Mapping)
        or type(scope.get("pair_count")) is not int
        or scope["pair_count"] != PAIR_COUNT
        or scope.get("target_manifest_evaluation400_is_only_final_paired_lane")
        is not True
        or scope.get("additional_reserve400_required") is not False
        or type(scope.get("additional_reserve400_count")) is not int
        or scope["additional_reserve400_count"] != 0
        or not isinstance(pairs, list)
        or len(pairs) != PAIR_COUNT
    ):
        raise ExternalAuthorityV2Error("identity bridge evaluation400 scope changed")
    for ordinal, row in enumerate(pairs):
        baseline = row.get("baseline_condition") if isinstance(row, Mapping) else None
        etsf = row.get("etsf_condition") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or type(row.get("ordinal")) is not int
            or row["ordinal"] != ordinal
            or type(row.get("target_manifest_global_ordinal")) is not int
            or row["target_manifest_global_ordinal"] != 130 + ordinal
            or type(row.get("requested_seed")) is not int
            or type(row.get("resolved_seed")) is not int
            or not isinstance(baseline, Mapping)
            or not isinstance(etsf, Mapping)
            or type(baseline.get("candidate_count")) is not int
            or baseline["candidate_count"] != bridge_v2.CANDIDATE_COUNT
            or type(etsf.get("candidate_count")) is not int
            or etsf["candidate_count"] != bridge_v2.CANDIDATE_COUNT
            or row.get("outcome_or_trajectory_fields_present") is not False
        ):
            raise ExternalAuthorityV2Error("identity bridge contains non-exact pair fields")


def reconstruct_bound_bridge(
    path: Path, expected_file_sha256: str
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    resolved, value = read_bound_json(path, expected_file_sha256, "identity bridge v2")
    try:
        bridge_v2.validate_bridge(value)
    except bridge_v2.Evaluation400BridgeError as error:
        raise ExternalAuthorityV2Error("identity bridge v2 validation failed") from error
    _strict_bridge_pairs(value)
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        BRIDGE_DEPENDENCIES
    ):
        raise ExternalAuthorityV2Error("identity bridge dependencies changed")
    loaded: dict[str, dict[str, Any]] = {}
    kwargs: dict[str, Any] = {}
    for role in BRIDGE_DEPENDENCIES:
        record = dependencies[role]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "file_sha256", "logical_sha256"}
            or not isinstance(record.get("path"), str)
            or not is_sha(record.get("file_sha256"))
            or not is_sha(record.get("logical_sha256"))
        ):
            raise ExternalAuthorityV2Error(f"identity bridge dependency changed: {role}")
        dependency_path, loaded[role] = read_bound_json(
            Path(record["path"]), record["file_sha256"], role
        )
        kwargs[f"{role}_path"] = dependency_path
        kwargs[f"{role}_file_sha256"] = record["file_sha256"]
    try:
        rebuilt = bridge_v2.freeze_bridge(**kwargs)
    except bridge_v2.Evaluation400BridgeError as error:
        raise ExternalAuthorityV2Error("identity bridge dependencies failed") from error
    if rebuilt != value:
        raise ExternalAuthorityV2Error("identity bridge differs from dependency reconstruction")
    return resolved, value, loaded


def validate_r7h_ensemble(
    ensemble: Mapping[str, Any], *, expected_source_contract_sha256: str
) -> dict[str, Any]:
    if not is_sha(expected_source_contract_sha256):
        raise ExternalAuthorityV2Error("expected r7h source contract SHA is invalid")
    shared = ensemble.get("shared_contract")
    members = ensemble.get("members")
    if (
        ensemble.get("format") != bridge_v2.ENSEMBLE_FORMAT
        or ensemble.get("status") != bridge_v2.ENSEMBLE_STATUS
        or type(ensemble.get("member_count")) is not int
        or ensemble["member_count"] != MEMBER_COUNT
        or not isinstance(shared, Mapping)
        or set(shared) != R7H_SHARED_FIELDS
        or any(not is_sha(shared.get(field)) for field in R7H_SHARED_FIELDS)
        or shared.get("source_ensemble_contract_sha256")
        != expected_source_contract_sha256
        or not isinstance(members, list)
        or len(members) != MEMBER_COUNT
    ):
        raise ExternalAuthorityV2Error("r7h five-member target-adapter ensemble changed")
    return {
        "source_ensemble_contract_sha256": expected_source_contract_sha256,
        "member_checkpoint_sha256": [
            row["checkpoint_file_sha256"] for row in members
        ],
        "ensemble_manifest_sha256": ensemble["ensemble_manifest_sha256"],
    }


def validate_external_decision(
    value: Mapping[str, Any],
    *,
    bridge_file_sha256: str,
    bridge_value: Mapping[str, Any],
    expected_r7h_source_contract_sha256: str,
) -> dict[str, Any]:
    logical = verify_signed(value, "decision_sha256", "independent decision")
    expected_fields = {
        "format",
        "status",
        "authority_issuer",
        "authority_issuer_identity_sha256",
        "decision_nonce_sha256",
        "identity_bridge_file_sha256",
        "identity_bridge_sha256",
        "pair_identity_set_sha256",
        "deployment_binding_sha256",
        "r7h_source_ensemble_contract_sha256",
        "authorized_pair_count",
        "target_manifest_evaluation400_is_only_lane",
        "additional_reserve400_authorized",
        "executor_independent_from_training_selection_and_protocol_freezer",
        "outcomes_or_trajectories_read_before_decision",
        "postfreeze_seed_candidate_or_threshold_change_authorized",
        "external_executor_only",
        "execution_authorized",
        "decision_sha256",
    }
    issuer = value.get("authority_issuer")
    if (
        set(value) != expected_fields
        or value.get("format") != DECISION_FORMAT
        or value.get("status") != DECISION_STATUS
        or not isinstance(issuer, str)
        or not issuer.strip()
        or not is_sha(value.get("authority_issuer_identity_sha256"))
        or not is_sha(value.get("decision_nonce_sha256"))
        or value.get("identity_bridge_file_sha256") != bridge_file_sha256
        or value.get("identity_bridge_sha256") != bridge_value["bridge_sha256"]
        or value.get("pair_identity_set_sha256")
        != bridge_value["pair_identity_set_sha256"]
        or value.get("deployment_binding_sha256")
        != bridge_value["deployment"]["deployment_binding_sha256"]
        or value.get("r7h_source_ensemble_contract_sha256")
        != expected_r7h_source_contract_sha256
        or type(value.get("authorized_pair_count")) is not int
        or value["authorized_pair_count"] != PAIR_COUNT
        or value.get("target_manifest_evaluation400_is_only_lane") is not True
        or value.get("additional_reserve400_authorized") is not False
        or value.get(
            "executor_independent_from_training_selection_and_protocol_freezer"
        )
        is not True
        or value.get("outcomes_or_trajectories_read_before_decision") is not False
        or value.get("postfreeze_seed_candidate_or_threshold_change_authorized")
        is not False
        or value.get("external_executor_only") is not True
        or value.get("execution_authorized") is not True
    ):
        raise ExternalAuthorityV2Error("independent execution decision changed")
    return {
        "path_independent_issuer": issuer.strip(),
        "decision_sha256": logical,
        "authority_issuer_identity_sha256": value[
            "authority_issuer_identity_sha256"
        ],
    }


def freeze_authority(
    *,
    identity_bridge_path: Path,
    identity_bridge_file_sha256: str,
    external_decision_path: Path,
    external_decision_file_sha256: str,
    expected_r7h_source_ensemble_contract_sha256: str,
) -> dict[str, Any]:
    # Validate every caller-supplied SHA before opening the first input.
    for role, value in (
        ("identity bridge", identity_bridge_file_sha256),
        ("external decision", external_decision_file_sha256),
        ("r7h source contract", expected_r7h_source_ensemble_contract_sha256),
    ):
        if not is_sha(value):
            raise ExternalAuthorityV2Error(f"{role} SHA is invalid")
    bridge_path, bridge_value, dependencies = reconstruct_bound_bridge(
        identity_bridge_path, identity_bridge_file_sha256
    )
    r7h = validate_r7h_ensemble(
        dependencies["ensemble_manifest"],
        expected_source_contract_sha256=(
            expected_r7h_source_ensemble_contract_sha256
        ),
    )
    decision_path, decision_value = read_bound_json(
        external_decision_path,
        external_decision_file_sha256,
        "independent external decision",
    )
    decision = validate_external_decision(
        decision_value,
        bridge_file_sha256=identity_bridge_file_sha256,
        bridge_value=bridge_value,
        expected_r7h_source_contract_sha256=(
            expected_r7h_source_ensemble_contract_sha256
        ),
    )
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "identity_bridge": {
            "path": str(bridge_path),
            "file_sha256": identity_bridge_file_sha256,
            "logical_sha256": bridge_value["bridge_sha256"],
            "pair_identity_set_sha256": bridge_value[
                "pair_identity_set_sha256"
            ],
        },
        "external_decision": {
            "path": str(decision_path),
            "file_sha256": external_decision_file_sha256,
            "logical_sha256": decision["decision_sha256"],
            "authority_issuer": decision["path_independent_issuer"],
            "authority_issuer_identity_sha256": decision[
                "authority_issuer_identity_sha256"
            ],
        },
        "deployment": {
            "deployment_binding_sha256": bridge_value["deployment"][
                "deployment_binding_sha256"
            ],
            "ensemble_manifest_sha256": r7h["ensemble_manifest_sha256"],
            "member_count": MEMBER_COUNT,
            "member_checkpoint_sha256": r7h["member_checkpoint_sha256"],
            "head_support_sha256": bridge_value["deployment"][
                "head_support_sha256"
            ],
            "calibration_sha256": bridge_value["deployment"][
                "calibration_sha256"
            ],
            "abstention_contract_sha256": bridge_value["deployment"][
                "abstention_contract_sha256"
            ],
            "r7h_source_ensemble_contract_sha256": r7h[
                "source_ensemble_contract_sha256"
            ],
        },
        "execution_scope": {
            "authorized_pair_count": PAIR_COUNT,
            "target_manifest_evaluation400_is_only_lane": True,
            "additional_reserve400_authorized": False,
            "additional_reserve400_count": 0,
            "baseline_condition": "lowest_legal_feasibility_root_candidate",
            "etsf_condition": (
                "frozen_five_member_event_world_model_with_uncertainty_abstention"
            ),
            "same_initial_state_for_both_conditions_required": True,
            "exact_frozen_condition_order_required": True,
            "postfreeze_seed_candidate_or_threshold_change_authorized": False,
            "external_executor_only": True,
            "protocol_freezer_may_execute": False,
            "execution_authorized": True,
        },
        "preexecution_capability_receipt": {
            "json_files_opened": 9,
            "checkpoint_files_opened": 0,
            "hdf5_files_opened": 0,
            "trajectory_files_opened": 0,
            "label_files_opened": 0,
            "outcomes_read": False,
            "policy_execution_calls": 0,
            "pair_conditions_executed": 0,
        },
    }
    return {**base, "authority_sha256": canonical_sha256(base)}


def validate_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = verify_signed(value, "authority_sha256", "external authority v2")
    expected_fields = {
        "format",
        "status",
        "identity_bridge",
        "external_decision",
        "deployment",
        "execution_scope",
        "preexecution_capability_receipt",
        "authority_sha256",
    }
    identity = value.get("identity_bridge")
    decision = value.get("external_decision")
    deployment = value.get("deployment")
    scope = value.get("execution_scope")
    capability = value.get("preexecution_capability_receipt")
    if (
        set(value) != expected_fields
        or value.get("format") != FORMAT
        or value.get("status") != STATUS
        or not isinstance(identity, Mapping)
        or set(identity)
        != {"path", "file_sha256", "logical_sha256", "pair_identity_set_sha256"}
        or not isinstance(decision, Mapping)
        or set(decision)
        != {
            "path",
            "file_sha256",
            "logical_sha256",
            "authority_issuer",
            "authority_issuer_identity_sha256",
        }
        or not isinstance(deployment, Mapping)
        or set(deployment) != {
            "deployment_binding_sha256", "ensemble_manifest_sha256",
            "member_count", "member_checkpoint_sha256", "head_support_sha256",
            "calibration_sha256", "abstention_contract_sha256",
            "r7h_source_ensemble_contract_sha256",
        }
        or type(deployment.get("member_count")) is not int
        or deployment["member_count"] != MEMBER_COUNT
        or not isinstance(deployment.get("member_checkpoint_sha256"), list)
        or len(deployment["member_checkpoint_sha256"]) != MEMBER_COUNT
        or len(set(deployment["member_checkpoint_sha256"])) != MEMBER_COUNT
        or any(
            not is_sha(candidate)
            for candidate in (
                identity.get("file_sha256"),
                identity.get("logical_sha256"),
                identity.get("pair_identity_set_sha256"),
                decision.get("file_sha256"),
                decision.get("logical_sha256"),
                decision.get("authority_issuer_identity_sha256"),
                deployment.get("deployment_binding_sha256"),
                deployment.get("ensemble_manifest_sha256"),
                deployment.get("head_support_sha256"),
                deployment.get("calibration_sha256"),
                deployment.get("abstention_contract_sha256"),
                deployment.get("r7h_source_ensemble_contract_sha256"),
                *deployment.get("member_checkpoint_sha256", []),
            )
        )
        or not isinstance(scope, Mapping)
        or set(scope) != {
            "authorized_pair_count", "target_manifest_evaluation400_is_only_lane",
            "additional_reserve400_authorized", "additional_reserve400_count",
            "baseline_condition", "etsf_condition",
            "same_initial_state_for_both_conditions_required",
            "exact_frozen_condition_order_required",
            "postfreeze_seed_candidate_or_threshold_change_authorized",
            "external_executor_only", "protocol_freezer_may_execute",
            "execution_authorized",
        }
        or type(scope.get("authorized_pair_count")) is not int
        or scope["authorized_pair_count"] != PAIR_COUNT
        or scope.get("target_manifest_evaluation400_is_only_lane") is not True
        or scope.get("additional_reserve400_authorized") is not False
        or type(scope.get("additional_reserve400_count")) is not int
        or scope["additional_reserve400_count"] != 0
        or scope.get("baseline_condition")
        != "lowest_legal_feasibility_root_candidate"
        or scope.get("etsf_condition")
        != "frozen_five_member_event_world_model_with_uncertainty_abstention"
        or scope.get("same_initial_state_for_both_conditions_required") is not True
        or scope.get("exact_frozen_condition_order_required") is not True
        or scope.get("postfreeze_seed_candidate_or_threshold_change_authorized")
        is not False
        or scope.get("external_executor_only") is not True
        or scope.get("protocol_freezer_may_execute") is not False
        or scope.get("execution_authorized") is not True
        or not isinstance(capability, Mapping)
        or set(capability) != {
            "json_files_opened", "checkpoint_files_opened", "hdf5_files_opened",
            "trajectory_files_opened", "label_files_opened", "outcomes_read",
            "policy_execution_calls", "pair_conditions_executed",
        }
        or type(capability.get("json_files_opened")) is not int
        or capability["json_files_opened"] != 9
        or type(capability.get("checkpoint_files_opened")) is not int
        or capability["checkpoint_files_opened"] != 0
        or type(capability.get("hdf5_files_opened")) is not int
        or capability["hdf5_files_opened"] != 0
        or type(capability.get("trajectory_files_opened")) is not int
        or capability["trajectory_files_opened"] != 0
        or type(capability.get("label_files_opened")) is not int
        or capability["label_files_opened"] != 0
        or capability.get("outcomes_read") is not False
        or type(capability.get("policy_execution_calls")) is not int
        or capability["policy_execution_calls"] != 0
        or type(capability.get("pair_conditions_executed")) is not int
        or capability["pair_conditions_executed"] != 0
    ):
        raise ExternalAuthorityV2Error("external authority v2 contract changed")
    return {
        "authority_sha256": logical,
        "identity_bridge_file_sha256": identity["file_sha256"],
        "identity_bridge_sha256": identity["logical_sha256"],
        "pair_identity_set_sha256": identity["pair_identity_set_sha256"],
        "deployment_binding_sha256": deployment["deployment_binding_sha256"],
        "r7h_source_ensemble_contract_sha256": deployment[
            "r7h_source_ensemble_contract_sha256"
        ],
        "member_checkpoint_sha256": list(
            deployment["member_checkpoint_sha256"]
        ),
        "authorized_pair_count": PAIR_COUNT,
    }


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_new_json(path, "authority output")
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
        temporary.chmod(0o400)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bridge", type=Path, required=True)
    parser.add_argument("--identity-bridge-file-sha256", required=True)
    parser.add_argument("--external-decision", type=Path, required=True)
    parser.add_argument("--external-decision-file-sha256", required=True)
    parser.add_argument(
        "--expected-r7h-source-ensemble-contract-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    value = freeze_authority(
        identity_bridge_path=args.identity_bridge,
        identity_bridge_file_sha256=args.identity_bridge_file_sha256,
        external_decision_path=args.external_decision,
        external_decision_file_sha256=args.external_decision_file_sha256,
        expected_r7h_source_ensemble_contract_sha256=(
            args.expected_r7h_source_ensemble_contract_sha256
        ),
    )
    validate_authority(value)
    write_json_new(args.output, value)
    print(json.dumps({"authority_sha256": value["authority_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
