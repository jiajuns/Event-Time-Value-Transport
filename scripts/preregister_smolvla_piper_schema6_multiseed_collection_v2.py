#!/usr/bin/env python3
"""CPU-only preregistration for the Piper schema-6 multi-seed expansion.

This file does not construct an environment, import a policy, inspect an HDF5
file, or authorize production collection.  It validates the frozen target seed
manifest, selects exactly adaptation80 then validation50 in manifest order,
and emits content-addressed per-seed command contracts for a future v2 runner.

The existing r6j reset/freezer/launcher are intentionally not executed: they
bind one fixed seed and one fixed object registry.  A production v2 runner must
materialize and verify can/pot identity and pose contracts independently after
every seed reset, then account for all four root candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence


FORMAT = "etsf_smolvla_piper_schema6_multiseed_preregistration_v2"
STATUS = "preregistered_cpu_protocol_only_production_not_authorized"
GROUP_RECEIPT_FORMAT = "etsf_smolvla_piper_schema6_multiseed_group_receipt_v2"
GROUP_RECEIPT_STATUS = "complete_four_candidate_schema6_group"
TARGET_FORMAT = "etsf_smolvla_piper_target_seed_manifest_v2"
TARGET_STATUS = "resolved_reset_identity_only_before_policy_execution"
TASK = "move_can_pot"
ACTOR_ID = "smolvla_robotwin_aloha-trained__piper-zero-shot"
TARGET_BODY = "piper_piper_0.6"
INSTRUCTION = "move the can into the pot"
SPLIT_COUNTS = {"adaptation": 80, "validation": 50, "evaluation": 400}
COLLECTED_SPLITS = ("adaptation", "validation")
CANDIDATE_INDICES = (0, 1, 2, 3)
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
HDF_SUFFIXES = (".h5", ".hdf", ".hdf5")
R6J_RUNTIME_ARTIFACTS = (
    "collect_smolvla_piper_schema6_dense_event_branches.py",
    "etsf_schema6_pose_quality.py",
    "freeze_smolvla_piper_schema6_development_collection.py",
    "launch_smolvla_piper_schema6_autonomous_watcher.py",
    "launch_smolvla_piper_schema6_development_collection.py",
    "materialize_smolvla_piper_schema6_reset_contract.py",
    "run_smolvla_piper_r6d_direct_actor_smoke.py",
)
SHA_ALPHABET = frozenset("0123456789abcdef")


class MultiSeedProtocolError(RuntimeError):
    """A frozen multi-seed collection invariant cannot be proved."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA_ALPHABET)
    )


def _contains_sensitive_component(path: PurePath) -> bool:
    return any(
        token in component.casefold()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    """Reject protected lexical components before any filesystem operation."""

    text = os.fspath(value)
    if not text or "\x00" in text:
        raise MultiSeedProtocolError(f"{role} path is empty/invalid")
    path = Path(os.path.abspath(os.path.expanduser(text)))
    if _contains_sensitive_component(PurePath(path)):
        raise MultiSeedProtocolError(f"{role} path contains a forbidden component")
    resolved = path.resolve(strict=False)
    if _contains_sensitive_component(PurePath(resolved)):
        raise MultiSeedProtocolError(f"{role} resolved path contains a forbidden component")
    return resolved


def _regular_read_only_file(value: str | os.PathLike[str], role: str) -> Path:
    path = safe_path(value, role)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise MultiSeedProtocolError(f"{role} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise MultiSeedProtocolError(f"{role} is not a regular file")
    if metadata.st_mode & 0o222:
        raise MultiSeedProtocolError(f"{role} is not frozen read-only")
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise MultiSeedProtocolError(f"{role} must never be an HDF file")
    return path


def _load_json(path: Path, role: str) -> dict[str, Any]:
    if path.suffix.casefold() != ".json":
        raise MultiSeedProtocolError(f"{role} must be JSON metadata")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiSeedProtocolError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise MultiSeedProtocolError(f"{role} must contain a JSON object")
    return value


def _verify_signature(value: Mapping[str, Any], key: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(key, None)
    if not _is_sha256(recorded) or recorded != canonical_sha256(unsigned):
        raise MultiSeedProtocolError(f"{role} logical signature mismatch")
    return str(recorded)


def _audit_embedded_paths(value: Any, role: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _audit_embedded_paths(child, f"{role}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _audit_embedded_paths(child, f"{role}[{index}]")
    elif isinstance(value, str):
        looks_like_path = value.startswith(("/", "./", "../")) or "\\" in value
        if looks_like_path and _contains_sensitive_component(PurePath(value)):
            raise MultiSeedProtocolError(f"{role} embeds a forbidden path")


def _row_pair_identity(row: Mapping[str, Any]) -> dict[str, Any]:
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


def _decode_seed_rows(
    rows: Any, *, split: str, count: int, global_start: int
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != count:
        raise MultiSeedProtocolError(f"target split {split} must contain exactly {count} rows")
    required = {
        "task", "actor_id", "target_body", "global_ordinal", "split", "ordinal",
        "stage_role", "requested_seed", "resolved_seed", "instruction",
        "instruction_sha256", "instruction_semantics_receipt",
        "instruction_semantics_receipt_sha256", "initial_scene_state_sha256",
        "initial_measured_joint_state_sha256",
        "initial_commanded_drive_target_sha256", "pair_id",
    }
    result: list[dict[str, Any]] = []
    instruction_sha = hashlib.sha256(INSTRUCTION.encode("utf-8")).hexdigest()
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise MultiSeedProtocolError(f"target seed {split}[{ordinal}] fields changed")
        row = dict(raw)
        requested = row["requested_seed"]
        resolved = row["resolved_seed"]
        semantic_receipt = row["instruction_semantics_receipt"]
        if (
            row["task"] != TASK
            or row["actor_id"] != ACTOR_ID
            or row["target_body"] != TARGET_BODY
            or row["split"] != split
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or type(row["global_ordinal"]) is not int
            or row["global_ordinal"] != global_start + ordinal
            or type(requested) is not int
            or type(resolved) is not int
            or requested < 0
            or resolved < 0
            or row["instruction"] != INSTRUCTION
            or row["instruction_sha256"] != instruction_sha
            or not isinstance(row["stage_role"], str)
            or not row["stage_role"]
            or not isinstance(semantic_receipt, Mapping)
            or row["instruction_semantics_receipt_sha256"]
            != semantic_receipt.get("receipt_sha256")
            or any(
                not _is_sha256(row[key])
                for key in (
                    "instruction_semantics_receipt_sha256",
                    "initial_scene_state_sha256",
                    "initial_measured_joint_state_sha256",
                    "initial_commanded_drive_target_sha256",
                    "pair_id",
                )
            )
            or row["pair_id"] != canonical_sha256(_row_pair_identity(row))
        ):
            raise MultiSeedProtocolError(f"target seed {split}[{ordinal}] identity changed")
        _verify_signature(semantic_receipt, "receipt_sha256", "instruction semantics receipt")
        result.append(row)
    return result


def validate_target_seed_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate v2 identity metadata; return only adaptation/validation rows."""

    logical_sha = _verify_signature(value, "seed_manifest_sha256", "target seed manifest")
    if (
        value.get("format") != TARGET_FORMAT
        or value.get("status") != TARGET_STATUS
        or value.get("task") != TASK
        or value.get("actor_id") != ACTOR_ID
        or value.get("target_body") != TARGET_BODY
        or value.get("capability_receipt", {}).get("policy_execution_authorized_by_manifest")
        is not False
        or value.get("capability_receipt", {}).get("labels_or_outcomes_read") is not False
    ):
        raise MultiSeedProtocolError("target seed manifest scope/capability changed")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_COUNTS):
        raise MultiSeedProtocolError("target seed manifest splits changed")
    decoded: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split in ("adaptation", "validation", "evaluation"):
        decoded[split] = _decode_seed_rows(
            splits[split], split=split, count=SPLIT_COUNTS[split], global_start=offset
        )
        offset += SPLIT_COUNTS[split]
    all_rows = [row for split in decoded.values() for row in split]
    for key in ("requested_seed", "resolved_seed", "pair_id"):
        values = [row[key] for row in all_rows]
        if len(values) != len(set(values)):
            raise MultiSeedProtocolError(f"target seed manifest {key} is not globally unique")
    return {
        "seed_manifest_sha256": logical_sha,
        "selected_rows": [
            dict(row) for split in COLLECTED_SPLITS for row in decoded[split]
        ],
        "validated_split_counts": dict(SPLIT_COUNTS),
    }


def _validate_event_spec(value: Mapping[str, Any]) -> None:
    chain = value.get("chains", {}).get(TASK, {})
    calibration = value.get("calibration", {}).get(TASK, {})
    if (
        chain.get("merge_e1_e2") is not True
        or chain.get("chain") != ["e0", "e12", "e3", "e4", "eK"]
        or calibration.get("moving") != "can"
        or calibration.get("anchor") not in (None, "", "pot")
    ):
        raise MultiSeedProtocolError("event specification is not canonical move_can_pot")


def _r6j_bindings(code_root: Path) -> dict[str, Any]:
    root = safe_path(code_root, "r6j code root")
    if not root.is_dir():
        raise MultiSeedProtocolError("r6j code root is not a directory")
    artifacts: dict[str, dict[str, str]] = {}
    for name in R6J_RUNTIME_ARTIFACTS:
        path = _regular_read_only_file(root / name, f"r6j artifact {name}")
        if path.parent != root:
            raise MultiSeedProtocolError(f"r6j artifact {name} escaped code root")
        artifacts[name] = {"path": str(path), "sha256": file_sha256(path)}
    closure = canonical_sha256({name: row["sha256"] for name, row in artifacts.items()})
    return {"root": str(root), "artifacts": artifacts, "closure_sha256": closure}


def _assert_expected_sha(actual: str, expected: str, role: str) -> None:
    if not _is_sha256(expected) or actual != expected:
        raise MultiSeedProtocolError(f"{role} SHA256 mismatch")


def _command_for_row(
    *,
    row: Mapping[str, Any],
    output_root: Path,
    runtime_python: Path,
    v2_runner: Path,
    preregistration_path: Path,
    command_bindings: Mapping[str, str],
) -> dict[str, Any]:
    split = str(row["split"])
    ordinal = int(row["ordinal"])
    stem = f"group_{ordinal:03d}_seed_{int(row['requested_seed'])}"
    seed_root = output_root / split / stem
    group_path = seed_root / "schema6_group.hdf5"
    reset_receipt = seed_root / "per_seed_reset_receipt.json"
    group_receipt = seed_root / "completed_group_receipt.json"
    argv = [
        str(runtime_python), str(v2_runner), "collect-one",
        "--preregistration", str(preregistration_path),
        "--split", split, "--ordinal", str(ordinal),
        "--requested-seed", str(row["requested_seed"]),
        "--expected-resolved-seed", str(row["resolved_seed"]),
        "--expected-pair-id", str(row["pair_id"]),
        "--seed-output-root", str(seed_root),
    ]
    command: dict[str, Any] = {
        "split": split,
        "ordinal": ordinal,
        "requested_seed": row["requested_seed"],
        "expected_resolved_seed": row["resolved_seed"],
        "pair_id": row["pair_id"],
        "expected_initial_scene_state_sha256": row["initial_scene_state_sha256"],
        "candidate_original_indices": list(CANDIDATE_INDICES),
        "argv": argv,
        "outputs": {
            "seed_root": str(seed_root),
            "per_seed_reset_receipt": str(reset_receipt),
            "group_hdf5": str(group_path),
            "completed_group_receipt": str(group_receipt),
        },
        "bindings": dict(command_bindings),
    }
    command["command_sha256"] = canonical_sha256(command)
    return command


def build_preregistration(
    *,
    target_seed_manifest_path: Path,
    expected_target_seed_manifest_file_sha256: str,
    r6j_code_root: Path,
    expected_r6j_code_closure_sha256: str,
    event_spec_path: Path,
    expected_event_spec_sha256: str,
    runtime_python_path: Path,
    expected_runtime_python_sha256: str,
    v2_runner_path: Path,
    expected_v2_runner_sha256: str,
    output_root: Path,
    preregistration_path: Path,
    gpu_lock_path: Path,
) -> dict[str, Any]:
    """Build a signed plan.  This function performs no collection operation."""

    manifest_path = _regular_read_only_file(target_seed_manifest_path, "target seed manifest")
    manifest_file_sha = file_sha256(manifest_path)
    _assert_expected_sha(
        manifest_file_sha, expected_target_seed_manifest_file_sha256, "target seed manifest file"
    )
    manifest = _load_json(manifest_path, "target seed manifest")
    _audit_embedded_paths(manifest, "target seed manifest")
    decoded = validate_target_seed_manifest(manifest)

    r6j = _r6j_bindings(r6j_code_root)
    _assert_expected_sha(r6j["closure_sha256"], expected_r6j_code_closure_sha256, "r6j closure")
    event_path = _regular_read_only_file(event_spec_path, "event specification")
    event_sha = file_sha256(event_path)
    _assert_expected_sha(event_sha, expected_event_spec_sha256, "event specification")
    event_value = _load_json(event_path, "event specification")
    _audit_embedded_paths(event_value, "event specification")
    _validate_event_spec(event_value)
    runtime_python = _regular_read_only_file(runtime_python_path, "bound runtime Python")
    runtime_python_sha = file_sha256(runtime_python)
    _assert_expected_sha(runtime_python_sha, expected_runtime_python_sha256, "bound runtime Python")
    v2_runner = _regular_read_only_file(v2_runner_path, "v2 runner")
    v2_runner_sha = file_sha256(v2_runner)
    _assert_expected_sha(v2_runner_sha, expected_v2_runner_sha256, "v2 runner")

    output = safe_path(output_root, "future collection output root")
    prereg = safe_path(preregistration_path, "preregistration output")
    lock = safe_path(gpu_lock_path, "RTX4090 lock")
    if output.exists():
        raise MultiSeedProtocolError("future collection output root already exists (create-once)")
    if prereg.exists():
        raise MultiSeedProtocolError("preregistration already exists (create-once)")
    if prereg == output or output in prereg.parents:
        raise MultiSeedProtocolError(
            "preregistration must be outside the future create-once collection root"
        )
    if prereg.suffix.casefold() != ".json":
        raise MultiSeedProtocolError("preregistration output must be JSON")
    if lock.suffix.casefold() in HDF_SUFFIXES:
        raise MultiSeedProtocolError("RTX4090 lock path cannot be HDF")

    bindings = {
        "target_seed_manifest_file_sha256": manifest_file_sha,
        "target_seed_manifest_sha256": decoded["seed_manifest_sha256"],
        "r6j_code_closure_sha256": r6j["closure_sha256"],
        "event_spec_sha256": event_sha,
        "runtime_python_sha256": runtime_python_sha,
        "v2_runner_sha256": v2_runner_sha,
    }
    commands = [
        _command_for_row(
            row=row, output_root=output, runtime_python=runtime_python,
            v2_runner=v2_runner, preregistration_path=prereg,
            command_bindings=bindings,
        )
        for row in decoded["selected_rows"]
    ]
    value: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "task": TASK,
        "actor_id": ACTOR_ID,
        "target_body": TARGET_BODY,
        "instruction": INSTRUCTION,
        "production_execution_authorized": False,
        "collection_scope": {
            "ordered_splits": list(COLLECTED_SPLITS),
            "adaptation_groups": 80,
            "validation_groups": 50,
            "evaluation_groups_in_target_manifest": 400,
            "evaluation_commands_generated": 0,
            "evaluation_environment_resets_authorized": 0,
            "evaluation_policy_executions_authorized": 0,
            "evaluation_hdf5_files_opened": 0,
            "selection_or_order_depends_on_outcome": False,
        },
        "data_gate": {
            "required_selected_groups": 130,
            "selected_groups": len(commands),
            "requested_seed_unique": True,
            "resolved_seed_unique": True,
            "pair_id_unique": True,
            "manifest_identity_metadata_validated_before_command_generation": True,
        },
        "input_bindings": {
            "target_seed_manifest": {
                "path": str(manifest_path), "file_sha256": manifest_file_sha,
                "logical_sha256": decoded["seed_manifest_sha256"],
            },
            "r6j_runtime_code": r6j,
            "event_spec": {"path": str(event_path), "sha256": event_sha},
            "runtime_python": {"path": str(runtime_python), "sha256": runtime_python_sha},
            "v2_runner": {"path": str(v2_runner), "sha256": v2_runner_sha},
        },
        "per_seed_reset_contract": {
            "requested_seed_must_equal_command": True,
            "resolved_seed_must_equal_manifest": True,
            "initial_scene_state_sha256_must_equal_manifest": True,
            "live_registry_required_after_every_seed_reset": True,
            "fixed_seed_object_registry_reuse_allowed": False,
            "required_objects_in_order": ["can", "pot"],
            "required_asset_families": ["105_sauce-can", "060_kitchenpot"],
            "stable_actor_id_and_model_index_required": True,
            "pose_spec_must_bind_live_registry_sha256": True,
            "branch_resets_must_reproduce_root_identity_registry_pose_and_observation": True,
        },
        "four_candidate_contract": {
            "root_candidate_original_indices": list(CANDIDATE_INDICES),
            "exact_branch_records_required": 4,
            "infeasible_candidate_must_be_signed_nonexecuted_censored_branch": True,
            "infeasible_action_execution_allowed": False,
            "existing_r6j_legal_only_branch_loop_is_production_eligible": False,
        },
        "execution_contract": {
            "create_once": True,
            "detached_process_required": True,
            "exclusive_rtx4090_flock_required": True,
            "gpu_lock_path": str(lock),
            "resume_only_from_signed_complete_prefix": True,
            "completed_group_file_sha256_reverification_required": True,
            "partial_or_unsigned_group_resume_allowed": False,
            "next_command_is_first_unfinished_manifest_order_command": True,
            "outcome_success_reward_event_values_may_control_order_or_resume": False,
        },
        "outputs": {
            "future_collection_root": str(output),
            "preregistration": str(prereg),
        },
        "commands": commands,
        "audit_receipt": {
            "environment_constructed": False,
            "environment_reset_calls": 0,
            "policy_import_or_forward_calls": 0,
            "hdf5_files_opened": 0,
            "protected_path_generated_or_opened": False,
        },
    }
    selected = decoded["selected_rows"]
    if (
        len(commands) != 130
        or len({row["requested_seed"] for row in selected}) != 130
        or len({row["resolved_seed"] for row in selected}) != 130
        or len({row["pair_id"] for row in selected}) != 130
    ):
        raise MultiSeedProtocolError("adaptation80+validation50 data gate failed")
    _audit_embedded_paths(value)
    value["preregistration_sha256"] = canonical_sha256(value)
    return value


def validate_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    logical_sha = _verify_signature(value, "preregistration_sha256", "multiseed preregistration")
    scope = value.get("collection_scope", {})
    commands = value.get("commands")
    if (
        value.get("format") != FORMAT
        or value.get("status") != STATUS
        or value.get("production_execution_authorized") is not False
        or scope.get("ordered_splits") != list(COLLECTED_SPLITS)
        or scope.get("evaluation_commands_generated") != 0
        or scope.get("evaluation_environment_resets_authorized") != 0
        or not isinstance(commands, list)
        or len(commands) != 130
        or [row.get("split") for row in commands]
        != ["adaptation"] * 80 + ["validation"] * 50
        or any(row.get("candidate_original_indices") != list(CANDIDATE_INDICES) for row in commands)
    ):
        raise MultiSeedProtocolError("multiseed preregistration scope changed")
    _audit_embedded_paths(value)
    return {"preregistration_sha256": logical_sha, "commands": [dict(row) for row in commands]}


def validate_completed_prefix(
    preregistration: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return pending commands only after a strict signed, gap-free prefix."""

    decoded = validate_preregistration(preregistration)
    commands = decoded["commands"]
    if len(receipts) > len(commands):
        raise MultiSeedProtocolError("completed receipt prefix is longer than the command plan")
    for index, (receipt, command) in enumerate(zip(receipts, commands, strict=False)):
        receipt_sha = _verify_signature(receipt, "group_receipt_sha256", f"group receipt {index}")
        expected = {
            "format": GROUP_RECEIPT_FORMAT,
            "status": GROUP_RECEIPT_STATUS,
            "preregistration_sha256": decoded["preregistration_sha256"],
            "command_sha256": command["command_sha256"],
            "split": command["split"],
            "ordinal": command["ordinal"],
            "requested_seed": command["requested_seed"],
            "resolved_seed": command["expected_resolved_seed"],
            "pair_id": command["pair_id"],
            "candidate_original_indices": list(CANDIDATE_INDICES),
            "branch_records": 4,
            "per_seed_reset_receipt_sha256": receipt.get("per_seed_reset_receipt_sha256"),
            "object_registry_sha256": receipt.get("object_registry_sha256"),
            "pose_spec_sha256": receipt.get("pose_spec_sha256"),
            "group_file_sha256": receipt.get("group_file_sha256"),
            "group_receipt_sha256": receipt_sha,
        }
        if (
            dict(receipt) != expected
            or any(
                not _is_sha256(receipt.get(key))
                for key in (
                    "per_seed_reset_receipt_sha256", "object_registry_sha256",
                    "pose_spec_sha256", "group_file_sha256",
                )
            )
        ):
            raise MultiSeedProtocolError(f"group receipt {index} is not an exact completed prefix row")
    return commands[len(receipts):]


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    output = safe_path(path, "preregistration output")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output.chmod(0o444)
    except BaseException:
        # Preserve a partial create-once artifact as failure evidence.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "collect-one"))
    parser.add_argument("--target-seed-manifest", type=Path)
    parser.add_argument("--expected-target-seed-manifest-file-sha256")
    parser.add_argument("--r6j-code-root", type=Path)
    parser.add_argument("--expected-r6j-code-closure-sha256")
    parser.add_argument("--event-spec", type=Path)
    parser.add_argument("--expected-event-spec-sha256")
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--expected-runtime-python-sha256")
    parser.add_argument("--v2-runner", type=Path)
    parser.add_argument("--expected-v2-runner-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "collect-one":
        raise MultiSeedProtocolError(
            "production collection is not authorized by this CPU-only phase; "
            "deploy an audited v2 runner first"
        )
    required = {
        name: getattr(args, name)
        for name in (
            "target_seed_manifest", "expected_target_seed_manifest_file_sha256",
            "r6j_code_root", "expected_r6j_code_closure_sha256", "event_spec",
            "expected_event_spec_sha256", "runtime_python",
            "expected_runtime_python_sha256", "v2_runner",
            "expected_v2_runner_sha256", "output_root", "output", "gpu_lock",
        )
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise MultiSeedProtocolError(f"missing preregistration arguments: {missing}")
    result = build_preregistration(
        target_seed_manifest_path=args.target_seed_manifest,
        expected_target_seed_manifest_file_sha256=args.expected_target_seed_manifest_file_sha256,
        r6j_code_root=args.r6j_code_root,
        expected_r6j_code_closure_sha256=args.expected_r6j_code_closure_sha256,
        event_spec_path=args.event_spec,
        expected_event_spec_sha256=args.expected_event_spec_sha256,
        runtime_python_path=args.runtime_python,
        expected_runtime_python_sha256=args.expected_runtime_python_sha256,
        v2_runner_path=args.v2_runner,
        expected_v2_runner_sha256=args.expected_v2_runner_sha256,
        output_root=args.output_root,
        preregistration_path=args.output,
        gpu_lock_path=args.gpu_lock,
    )
    immutable_json(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": result["preregistration_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_INDICES", "FORMAT", "GROUP_RECEIPT_FORMAT",
    "GROUP_RECEIPT_STATUS", "MultiSeedProtocolError", "R6J_RUNTIME_ARTIFACTS",
    "STATUS", "build_preregistration", "canonical_sha256", "file_sha256",
    "immutable_json", "validate_completed_prefix", "validate_preregistration",
    "validate_target_seed_manifest",
]
