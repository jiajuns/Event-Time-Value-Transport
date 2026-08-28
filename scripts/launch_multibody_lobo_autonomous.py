#!/usr/bin/env python3
"""Fail-closed detached watcher for sequential Piper/UR5 LOBO training.

The watcher is deliberately only an orchestrator.  It never imports the LOBO
trainer, never opens an HDF5 file, and never chooses a checkpoint.  It waits
for the already-running SmolVLA source63 watcher to publish its authenticated
terminal receipt and freeze its output tree, acquires the same process lock,
confirms an idle RTX 4090, and then invokes the immutable LOBO trainer first
for Piper and then for UR5.  A failed first stage prevents the second stage
from being launched.

All paths containing ``fresh`` or ``confirmation`` are rejected.  The watcher
root and both external training outputs must be absent at preregistration.
``detach`` creates the watcher root and immutable launch plan synchronously,
before it starts a new-session server-side process, so a second launcher
cannot win a race after the command returns.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence


FORMAT = "etsf_multibody_lobo_autonomous_watcher_v1"
STATE_FORMAT = "etsf_multibody_lobo_autonomous_state_v1"
DETACH_FORMAT = "etsf_multibody_lobo_autonomous_detach_receipt_v1"
SOURCE_BINDING_FORMAT = "etsf_multibody_lobo_source_binding_receipt_v1"
SOURCE_BINDING_STATUS = "bound_native_source_ensemble_for_deployment_rerank"
STAGE_SOURCE_BINDING_FORMAT = "etsf_multibody_lobo_stage_source_binding_v1"
SOURCE_HEADER_TRUST_ROOT = (
    "pinned_source_launch_plan_and_frozen_source_final_receipt_"
    "whose_training_audit_loaded_the_ensemble_and_validated_"
    "its_policy_bridge_header"
)
UPSTREAM_FORMAT = "etsf_smolvla_schema5_source63_native_training_launcher_v1"
UPSTREAM_TERMINAL_STATUS = (
    "complete_source63_native_counterfactual_training_fresh_forbidden"
)
UPSTREAM_FAILURE_STATUS = (
    "failed_closed_source63_native_counterfactual_training_fresh_forbidden"
)
UPSTREAM_TRAINING_STATUS = "complete_verified_source63_counterfactual_training"
LOBO_FORMAT = "etsf_multibody_leave_one_body_out_v1"
LOBO_SPLIT_FORMAT = "etsf_multibody_lobo_frozen_split_v1"
LOBO_TERMINAL_STATUS = (
    "training_and_frozen_target_development_evaluation_complete"
)
TERMINAL_STATUS = "complete_sequential_piper_then_ur5_lobo_training"
FAILURE_STATUS = "failed_closed_sequential_lobo_training"
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
SHA256_CHARS = frozenset("0123456789abcdef")
LOBO_STAGES = (
    ("piper", "piper"),
    ("ur5", "ur5-wsg"),
)
ENSEMBLE_SEEDS = (20260828, 20260829, 20260830, 20260831, 20260832)
SPLIT_SEED = 20260828
TRAINING_STEPS = 3000
EVAL_EVERY = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_CHARS)
    )


def _contains_sensitive_path_component(path: PurePath) -> bool:
    return any(
        token in component.lower()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def reject_sensitive_path_text(value: Any, role: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} path is missing")
    if _contains_sensitive_path_component(PurePath(value)):
        raise ValueError(f"{role} references a forbidden path namespace")


def resolve_existing_path(path: Path, *, role: str, directory: bool) -> Path:
    supplied = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(supplied)))
    if _contains_sensitive_path_component(absolute):
        raise ValueError(f"{role} path is forbidden")
    if supplied.is_symlink():
        raise ValueError(f"{role} must be materialized, not a symlink")
    resolved = supplied.resolve(strict=True)
    if _contains_sensitive_path_component(resolved):
        raise ValueError(f"{role} resolves into a forbidden path")
    mode = resolved.stat().st_mode
    valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not valid:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{role} must be a {kind}")
    return resolved


def resolve_new_path(path: Path, *, role: str) -> Path:
    supplied = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(supplied)))
    if _contains_sensitive_path_component(absolute):
        raise ValueError(f"{role} path is forbidden")
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"{role} already exists: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    if _contains_sensitive_path_component(parent) or not parent.is_dir():
        raise ValueError(f"{role} parent is invalid or forbidden")
    return absolute


def python_contract(path: Path) -> dict[str, Any]:
    invocation = Path(os.path.abspath(os.fspath(path.expanduser())))
    reject_sensitive_path_text(str(invocation), "Python executable")
    resolved = invocation.resolve(strict=True)
    if _contains_sensitive_path_component(resolved) or not resolved.is_file():
        raise ValueError("Python executable is invalid or forbidden")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(resolved)
    return {
        "invocation_path": str(invocation),
        "resolved_path": str(resolved),
        "resolved_sha256": file_sha256(resolved),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_text_new(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_to_claimed_fd(descriptor: int, value: Mapping[str, Any]) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} must be a materialized regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must contain a JSON object")
    return value


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def trainer_implementation_closure(code_root: Path) -> dict[str, dict[str, Any]]:
    scripts = code_root / "scripts"
    scripts_root = scripts.resolve(strict=True)
    queue = [scripts / "train_multibody_leave_one_body_out.py"]
    seen: set[Path] = set()
    while queue:
        path = queue.pop().resolve(strict=True)
        if path in seen:
            continue
        if scripts_root not in path.parents or not path.is_file():
            raise RuntimeError(f"trainer implementation escaped code root: {path}")
        seen.add(path)
        for module in _local_imports(path):
            candidate = scripts / (module.replace(".", "/") + ".py")
            if candidate.is_file():
                queue.append(candidate)
    if not any(path.name == "train_multibody_leave_one_body_out.py" for path in seen):
        raise RuntimeError("LOBO trainer is missing from implementation closure")
    return {
        str(path.relative_to(code_root)): {
            "path": str(path),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(seen)
    }


def verify_file_hash(path: Path, expected: str, *, role: str) -> str:
    if not _is_sha256(expected):
        raise ValueError(f"{role} expected SHA-256 is malformed")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{role} SHA-256 mismatch")
    return actual


def validate_source_launch_plan(
    path: Path, *, expected_file_sha256: str, expected_root: Path, gpu_index: int
) -> dict[str, Any]:
    verify_file_hash(path, expected_file_sha256, role="source63 launch plan")
    plan = load_json(path, role="source63 launch plan")
    unsigned = dict(plan)
    logical = unsigned.pop("static_plan_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise RuntimeError("source63 launch plan internal SHA-256 mismatch")
    if (
        plan.get("format") != UPSTREAM_FORMAT
        or plan.get("status")
        != "static_preflight_complete_waiting_no_manifest_or_hdf5_read"
        or Path(str(plan.get("output_root", ""))).resolve() != expected_root
        or plan.get("gpu_index") != gpu_index
        or plan.get("device") != "cuda"
        or plan.get("fresh_inputs_accepted") is not False
        or plan.get("fresh_labels_read") is not False
        or plan.get("manifest_read_during_static_preflight") is not False
        or plan.get("hdf5_opened_during_static_preflight") is not False
        or plan.get("nonresumable_output") is not True
    ):
        raise RuntimeError("source63 launch plan contract is invalid")
    return {
        "path": str(path),
        "file_sha256": expected_file_sha256,
        "static_plan_sha256": logical,
        "output_root": str(expected_root),
        "gpu_index": gpu_index,
    }


def validate_lobo_split(
    path: Path,
    *,
    expected_file_sha256: str,
    held_out_body: str,
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    verify_file_hash(path, expected_file_sha256, role=f"{held_out_body} split")
    value = load_json(path, role=f"{held_out_body} frozen split")
    unsigned = dict(value)
    logical = unsigned.pop("sha256", None)
    if logical != canonical_sha256(unsigned):
        raise RuntimeError(f"{held_out_body} split internal SHA-256 mismatch")
    lanes = value.get("lanes")
    expected_lanes = {
        "source_train",
        "source_validation",
        "target_development",
        "target_unused_train",
        "sealed_test",
    }
    if (
        value.get("format") != LOBO_SPLIT_FORMAT
        or value.get("held_out_body") != held_out_body
        or value.get("split_seed") != SPLIT_SEED
        or value.get("split_inputs") != dict(input_sha256)
        or value.get("event_spec_sha256") != input_sha256["event_spec"]
        or value.get("labels_used_for_assignment") is not False
        or value.get("checkpoint_selection_lane") != "source_validation"
        or value.get("final_evaluation_lane") != "target_development"
        or value.get("target_development_used_for_checkpoint_selection") is not False
        or value.get("target_unused_train_payload_opened") != 0
        or value.get("sealed_test_group_hdf5_opened") != 0
        or not isinstance(lanes, Mapping)
        or set(lanes) != expected_lanes
    ):
        raise RuntimeError(f"{held_out_body} frozen split contract is invalid")
    all_identities: set[str] = set()
    lane_counts: dict[str, int] = {}
    for name in sorted(expected_lanes):
        lane = lanes[name]
        if not isinstance(lane, Mapping) or not isinstance(lane.get("identities"), list):
            raise RuntimeError(f"{held_out_body} split lane {name} is invalid")
        identities = lane["identities"]
        if (
            identities != sorted(identities)
            or len(identities) != len(set(identities))
            or lane.get("groups") != len(identities)
            or lane.get("identity_sha256") != canonical_sha256(identities)
            or all_identities.intersection(identities)
        ):
            raise RuntimeError(f"{held_out_body} split lane {name} integrity failed")
        all_identities.update(identities)
        lane_counts[name] = len(identities)
    if not all(lane_counts[name] > 0 for name in expected_lanes):
        raise RuntimeError(f"{held_out_body} split contains an empty lane")
    return {
        "path": str(path),
        "file_sha256": expected_file_sha256,
        "logical_sha256": logical,
        "held_out_body": held_out_body,
        "lanes": lane_counts,
        "payload_hdf5_opened": 0,
    }


def build_lobo_command(
    *,
    python_bin: Path,
    trainer: Path,
    held_out_body: str,
    output: Path,
    split: Path,
    split_sha256: str,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    argv = [
        str(python_bin),
        str(trainer),
        "--mode",
        "train",
        "--held-out-body",
        held_out_body,
        "--stage1-root",
        bindings["stage1_root"],
        "--stage1-source-manifest",
        bindings["stage1_source_manifest"],
        "--stage1-source-manifest-sha256",
        bindings["stage1_source_manifest_sha256"],
        "--stage1-target-manifest",
        bindings["stage1_target_manifest"],
        "--stage1-target-manifest-sha256",
        bindings["stage1_target_manifest_sha256"],
        "--event-spec",
        bindings["event_spec"],
        "--event-spec-sha256",
        bindings["event_spec_sha256"],
        "--openvla-schema5-manifest",
        bindings["openvla_schema5_manifest"],
        "--openvla-schema5-manifest-sha256",
        bindings["openvla_schema5_manifest_sha256"],
        "--split-plan",
        str(split),
        "--split-plan-sha256",
        split_sha256,
        "--output",
        str(output),
        "--split-seed",
        str(SPLIT_SEED),
        "--ensemble-seeds",
        *[str(seed) for seed in ENSEMBLE_SEEDS],
        "--steps",
        str(TRAINING_STEPS),
        "--eval-every",
        str(EVAL_EVERY),
        "--batch-size",
        str(BATCH_SIZE),
        "--learning-rate",
        str(LEARNING_RATE),
        "--device",
        "cuda",
    ]
    stage_name = "train_lobo_piper" if held_out_body == "piper" else "train_lobo_ur5"
    return {
        "stage": stage_name,
        "held_out_body": held_out_body,
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "output": str(output),
        "split": str(split),
        "split_file_sha256": split_sha256,
        "device": "cuda",
        "training_steps_per_member": TRAINING_STEPS,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "variants": ["source_body_clock", "body_agnostic"],
        "source_binding_required_after_upstream_terminal": True,
        "lobo_checkpoints_rerank_authorized": False,
        "test_group_hdf5_opened_by_watcher": 0,
    }


def static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    launcher = Path(__file__).resolve(strict=True)
    code_root = resolve_existing_path(args.code_root, role="LOBO code root", directory=True)
    trainer = resolve_existing_path(
        args.trainer or code_root / "scripts" / "train_multibody_leave_one_body_out.py",
        role="LOBO trainer",
        directory=False,
    )
    source_root = resolve_existing_path(
        args.source_training_root, role="source63 training root", directory=True
    )
    source_plan_path = resolve_existing_path(
        args.source_launch_plan or source_root / "launch_plan.json",
        role="source63 launch plan",
        directory=False,
    )
    output_root = resolve_new_path(args.output, role="LOBO watcher output")
    piper_output = resolve_new_path(args.piper_output, role="Piper LOBO output")
    ur5_output = resolve_new_path(args.ur5_output, role="UR5 LOBO output")
    if len({output_root, piper_output, ur5_output}) != 3:
        raise ValueError("watcher and LOBO outputs must be three distinct paths")
    if any(output_root in path.parents for path in (piper_output, ur5_output)):
        raise ValueError("external LOBO outputs must not be nested in watcher output")
    if piper_output in ur5_output.parents or ur5_output in piper_output.parents:
        raise ValueError("Piper and UR5 outputs must not contain each other")
    python = python_contract(args.python_bin)
    input_paths = {
        "stage1_source_manifest": resolve_existing_path(
            args.stage1_source_manifest, role="Stage1 source manifest", directory=False
        ),
        "stage1_target_manifest": resolve_existing_path(
            args.stage1_target_manifest, role="Stage1 target manifest", directory=False
        ),
        "event_spec": resolve_existing_path(
            args.event_spec, role="event spec", directory=False
        ),
        "openvla_schema5_manifest": resolve_existing_path(
            args.openvla_schema5_manifest,
            role="OpenVLA schema5 manifest",
            directory=False,
        ),
    }
    stage1_root = resolve_existing_path(
        args.stage1_root, role="Stage1 root", directory=True
    )
    expected_hashes = {
        "stage1_source_manifest": args.stage1_source_manifest_sha256,
        "stage1_target_manifest": args.stage1_target_manifest_sha256,
        "event_spec": args.event_spec_sha256,
        "openvla_schema5_manifest": args.openvla_schema5_manifest_sha256,
    }
    input_sha256 = {
        name: verify_file_hash(path, expected_hashes[name], role=name)
        for name, path in input_paths.items()
    }
    source_plan = validate_source_launch_plan(
        source_plan_path,
        expected_file_sha256=args.source_launch_plan_sha256,
        expected_root=source_root,
        gpu_index=args.gpu_index,
    )
    piper_split_path = resolve_existing_path(
        args.piper_split, role="Piper frozen split", directory=False
    )
    ur5_split_path = resolve_existing_path(
        args.ur5_split, role="UR5 frozen split", directory=False
    )
    splits = {
        "piper": validate_lobo_split(
            piper_split_path,
            expected_file_sha256=args.piper_split_sha256,
            held_out_body="piper",
            input_sha256=input_sha256,
        ),
        "ur5": validate_lobo_split(
            ur5_split_path,
            expected_file_sha256=args.ur5_split_sha256,
            held_out_body="ur5-wsg",
            input_sha256=input_sha256,
        ),
    }
    implementations = trainer_implementation_closure(code_root)
    bindings = {
        "stage1_root": str(stage1_root),
        **{name: str(path) for name, path in input_paths.items()},
        **{f"{name}_sha256": digest for name, digest in input_sha256.items()},
    }
    commands = [
        build_lobo_command(
            python_bin=Path(python["invocation_path"]),
            trainer=trainer,
            held_out_body="piper",
            output=piper_output,
            split=piper_split_path,
            split_sha256=args.piper_split_sha256,
            bindings=bindings,
        ),
        build_lobo_command(
            python_bin=Path(python["invocation_path"]),
            trainer=trainer,
            held_out_body="ur5-wsg",
            output=ur5_output,
            split=ur5_split_path,
            split_sha256=args.ur5_split_sha256,
            bindings=bindings,
        ),
    ]
    gpu_lock = Path(
        os.path.abspath(
            os.fspath(
                args.gpu_lock
                or Path(f"/tmp/etsf_smolvla_schema5_source63_gpu{args.gpu_index}.lock")
            )
        )
    )
    reject_sensitive_path_text(str(gpu_lock), "shared GPU lock")
    if not gpu_lock.parent.resolve(strict=True).is_dir():
        raise ValueError("shared GPU lock parent is invalid")
    plan: dict[str, Any] = {
        "format": FORMAT,
        "status": "preregistered_waiting_for_authenticated_source63_completion",
        "launcher": {
            "path": str(launcher),
            "sha256": file_sha256(launcher),
            "size": launcher.stat().st_size,
        },
        "code_root": str(code_root),
        "trainer": str(trainer),
        "trainer_implementation_files": implementations,
        "trainer_implementation_bundle_sha256": canonical_sha256(implementations),
        "python_contract": python,
        "source63": source_plan,
        "source63_final_receipt": str(source_root / "final_receipt.json"),
        "source63_failure_receipt": str(source_root / "failure_receipt.json"),
        "source_binding_receipt": str(output_root / "source_binding_receipt.json"),
        "source_binding_materialized_create_once_after_authenticated_terminal": True,
        "source_checkpoint_header_validation_trust_root": (
            SOURCE_HEADER_TRUST_ROOT
        ),
        "lobo_checkpoint_role": "canonical_cross_embodiment_scientific_evaluation_only",
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
        "input_bindings": bindings,
        "input_sha256": input_sha256,
        "splits": splits,
        "output_root": str(output_root),
        "preregistered_outputs": {
            "piper": str(piper_output),
            "ur5": str(ur5_output),
        },
        "commands": commands,
        "execution_order": [command["stage"] for command in commands],
        "gpu_index": args.gpu_index,
        "gpu_lock": str(gpu_lock),
        "poll_seconds": args.poll_seconds,
        "source_timeout_seconds": args.source_timeout_seconds,
        "gpu_timeout_seconds": args.gpu_timeout_seconds,
        "stage_timeout_seconds": args.stage_timeout_seconds,
        "idle_confirmations": args.idle_confirmations,
        "omp_threads": args.omp_threads,
        "output_paths_absent_at_preregistration": True,
        "sequential_failure_stops_later_stages": True,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "watcher_hdf5_imported": False,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
        "nonresumable_output": True,
    }
    plan["static_plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_implementation_unchanged(plan: Mapping[str, Any]) -> None:
    launcher = Path(str(plan["launcher"]["path"]))
    if (
        file_sha256(launcher) != plan["launcher"]["sha256"]
        or launcher.stat().st_size != plan["launcher"]["size"]
    ):
        raise RuntimeError("autonomous watcher implementation changed")
    implementations = trainer_implementation_closure(Path(str(plan["code_root"])))
    if (
        implementations != plan["trainer_implementation_files"]
        or canonical_sha256(implementations)
        != plan["trainer_implementation_bundle_sha256"]
    ):
        raise RuntimeError("LOBO trainer implementation changed")
    if python_contract(Path(str(plan["python_contract"]["invocation_path"]))) != plan[
        "python_contract"
    ]:
        raise RuntimeError("Python executable changed")
    for name, path_text in (
        ("stage1_source_manifest", plan["input_bindings"]["stage1_source_manifest"]),
        ("stage1_target_manifest", plan["input_bindings"]["stage1_target_manifest"]),
        ("event_spec", plan["input_bindings"]["event_spec"]),
        ("openvla_schema5_manifest", plan["input_bindings"]["openvla_schema5_manifest"]),
    ):
        if file_sha256(Path(path_text)) != plan["input_sha256"][name]:
            raise RuntimeError(f"bound LOBO input changed: {name}")
    for name, split in plan["splits"].items():
        if file_sha256(Path(split["path"])) != split["file_sha256"]:
            raise RuntimeError(f"bound {name} split changed")


def _tree_is_read_only_materialized(root: Path) -> bool:
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o222:
        return False
    for path in root.rglob("*"):
        if path.is_symlink() or path.stat().st_mode & 0o222:
            return False
    return True


def validate_source_terminal_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(plan["source63"]["output_root"]))
    receipt_path = root / "final_receipt.json"
    before = file_sha256(receipt_path)
    receipt = load_json(receipt_path, role="source63 final receipt")
    after = file_sha256(receipt_path)
    audit = receipt.get("training_audit")
    ensemble_path_value = audit.get("ensemble_checkpoint") if isinstance(audit, Mapping) else None
    ensemble_sha256 = (
        audit.get("ensemble_checkpoint_sha256") if isinstance(audit, Mapping) else None
    )
    bridge_sha256 = (
        audit.get("policy_feature_action_bridge_sha256")
        if isinstance(audit, Mapping)
        else None
    )
    if (
        before != after
        or receipt.get("format") != UPSTREAM_FORMAT
        or receipt.get("status") != UPSTREAM_TERMINAL_STATUS
        or receipt.get("static_plan_sha256")
        != plan["source63"]["static_plan_sha256"]
        or not isinstance(audit, Mapping)
        or audit.get("status") != UPSTREAM_TRAINING_STATUS
        or audit.get("member_count") != 5
        or audit.get("member_seeds") != list(ENSEMBLE_SEEDS)
        or audit.get("member_training_steps_verified") != [TRAINING_STEPS] * 5
        or audit.get("target_data_read") is not False
        or audit.get("target_labels_read") is not False
        or audit.get("test_labels_used") is not False
        or audit.get("test_hdf_label_datasets_opened") != 0
        or not isinstance(ensemble_path_value, str)
        or not ensemble_path_value
        or not _is_sha256(ensemble_sha256)
        or not _is_sha256(bridge_sha256)
        or receipt.get("target_data_read") is not False
        or receipt.get("target_labels_read") is not False
        or receipt.get("fresh_inputs_accepted") is not False
        or receipt.get("fresh_labels_read") is not False
        or receipt.get("test_labels_used") is not False
        or receipt.get("test_hdf_label_datasets_opened") != 0
        or receipt.get("artifacts_frozen_read_only") is not True
        or not all(
            _is_sha256(receipt.get(field))
            for field in (
                "execution_plan_sha256",
                "snapshot_sha256",
                "artifact_inventory_sha256",
                "initialized_checkpoint_sha256",
            )
        )
        or not _tree_is_read_only_materialized(root)
    ):
        raise RuntimeError("source63 terminal receipt is incomplete or not frozen")
    raw_ensemble_path = Path(ensemble_path_value)
    if raw_ensemble_path.is_symlink() or not raw_ensemble_path.is_file():
        raise RuntimeError("source63 ensemble checkpoint is not materialized")
    ensemble_path = raw_ensemble_path.resolve(strict=True)
    if root.resolve(strict=True) not in ensemble_path.parents:
        raise RuntimeError("source63 ensemble checkpoint escaped its frozen output root")
    ensemble_before = file_sha256(ensemble_path)
    ensemble_after = file_sha256(ensemble_path)
    if ensemble_before != ensemble_after or ensemble_before != ensemble_sha256:
        raise RuntimeError("source63 ensemble checkpoint hash disagrees with terminal receipt")
    state = load_json(root / "launch_state.json", role="source63 terminal state")
    if (
        state.get("status") != UPSTREAM_TERMINAL_STATUS
        or state.get("static_plan_sha256")
        != plan["source63"]["static_plan_sha256"]
        or state.get("target_data_read") is not False
        or state.get("target_labels_read") is not False
        or state.get("test_labels_used") is not False
        or state.get("test_hdf_label_datasets_opened") != 0
    ):
        raise RuntimeError("source63 terminal state disagrees with its receipt")
    return {
        "status": UPSTREAM_TERMINAL_STATUS,
        "final_receipt_path": str(receipt_path),
        "final_receipt_sha256": before,
        "final_receipt_logical_sha256": canonical_sha256(receipt),
        "static_plan_sha256": receipt["static_plan_sha256"],
        "execution_plan_sha256": receipt["execution_plan_sha256"],
        "ensemble_checkpoint": str(ensemble_path),
        "ensemble_checkpoint_sha256": ensemble_before,
        "policy_feature_action_bridge_sha256": bridge_sha256,
        "member_count": 5,
        "member_training_steps_verified": [TRAINING_STEPS] * 5,
        "output_tree_read_only": True,
        "test_hdf_label_datasets_opened": 0,
    }


def _source_binding_payload(
    plan: Mapping[str, Any], source_audit: Mapping[str, Any]
) -> dict[str, Any]:
    source_plan = plan["source63"]
    deployment_checkpoint = {
        "path": source_audit["ensemble_checkpoint"],
        "sha256": source_audit["ensemble_checkpoint_sha256"],
        "policy": "smolvla",
        "checkpoint_family": "smolvla_native_event_world_model",
        "policy_feature_action_bridge_contract_sha256": source_audit[
            "policy_feature_action_bridge_sha256"
        ],
        "source_native_checkpoint": True,
    }
    payload: dict[str, Any] = {
        "format": SOURCE_BINDING_FORMAT,
        "status": SOURCE_BINDING_STATUS,
        "source_training_root": source_plan["output_root"],
        "source_launch_plan": {
            "path": source_plan["path"],
            "file_sha256": source_plan["file_sha256"],
            "logical_sha256": source_plan["static_plan_sha256"],
        },
        "source_final_receipt": {
            "path": source_audit["final_receipt_path"],
            "file_sha256": source_audit["final_receipt_sha256"],
            "logical_sha256": source_audit["final_receipt_logical_sha256"],
            "status": source_audit["status"],
        },
        "source_training_audit_status": UPSTREAM_TRAINING_STATUS,
        "deployment_rerank_checkpoint": deployment_checkpoint,
        "policy_feature_action_bridge_contract_sha256": source_audit[
            "policy_feature_action_bridge_sha256"
        ],
        "checkpoint_header_revalidated_by_lobo_watcher": False,
        "checkpoint_header_validation_trust_root": SOURCE_HEADER_TRUST_ROOT,
        "lobo_checkpoint_role": "canonical_cross_embodiment_scientific_evaluation_only",
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
        "target_data_read": False,
        "target_labels_read": False,
        "test_hdf_label_datasets_opened": 0,
    }
    payload["binding_sha256"] = canonical_sha256(payload)
    return payload


def _verify_bound_json_artifact(
    binding: Mapping[str, Any], *, role: str, logical_field: str
) -> dict[str, Any]:
    path_value = binding.get("path")
    expected_file_sha256 = binding.get("file_sha256")
    expected_logical_sha256 = binding.get("logical_sha256")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not _is_sha256(expected_file_sha256)
        or not _is_sha256(expected_logical_sha256)
    ):
        raise RuntimeError(f"{role} binding is incomplete")
    path = Path(path_value)
    before = file_sha256(path)
    value = load_json(path, role=role)
    after = file_sha256(path)
    if before != after or before != expected_file_sha256:
        raise RuntimeError(f"{role} file SHA changed")
    unsigned = dict(value)
    embedded = unsigned.pop(logical_field, None)
    computed_unsigned = canonical_sha256(unsigned)
    computed_full = canonical_sha256(value)
    if expected_logical_sha256 not in (embedded, computed_unsigned, computed_full):
        raise RuntimeError(f"{role} logical SHA changed")
    return value


def validate_source_binding_receipt(
    path: Path, *, plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise RuntimeError("source binding receipt must be create-once and read-only")
    before = file_sha256(path)
    receipt = load_json(path, role="LOBO source binding receipt")
    after = file_sha256(path)
    unsigned = dict(receipt)
    logical = unsigned.pop("binding_sha256", None)
    source_plan_binding = receipt.get("source_launch_plan")
    source_final_binding = receipt.get("source_final_receipt")
    deployment = receipt.get("deployment_rerank_checkpoint")
    bridge_sha256 = receipt.get("policy_feature_action_bridge_contract_sha256")
    if (
        before != after
        or receipt.get("format") != SOURCE_BINDING_FORMAT
        or receipt.get("status") != SOURCE_BINDING_STATUS
        or logical != canonical_sha256(unsigned)
        or not isinstance(source_plan_binding, Mapping)
        or not isinstance(source_final_binding, Mapping)
        or not isinstance(deployment, Mapping)
        or not _is_sha256(bridge_sha256)
        or receipt.get("source_training_audit_status") != UPSTREAM_TRAINING_STATUS
        or receipt.get("checkpoint_header_revalidated_by_lobo_watcher") is not False
        or receipt.get("checkpoint_header_validation_trust_root")
        != SOURCE_HEADER_TRUST_ROOT
        or receipt.get("lobo_checkpoint_role")
        != "canonical_cross_embodiment_scientific_evaluation_only"
        or receipt.get("lobo_checkpoints_rerank_authorized") is not False
        or receipt.get("deployment_rerank_authority") != "native_source_ensemble_only"
        or receipt.get("target_data_read") is not False
        or receipt.get("target_labels_read") is not False
        or receipt.get("test_hdf_label_datasets_opened") != 0
    ):
        raise RuntimeError("source binding receipt contract is invalid")
    source_plan = _verify_bound_json_artifact(
        source_plan_binding, role="bound source launch plan", logical_field="static_plan_sha256"
    )
    source_final = _verify_bound_json_artifact(
        source_final_binding, role="bound source final receipt", logical_field="receipt_sha256"
    )
    root = Path(str(receipt.get("source_training_root", ""))).resolve(strict=True)
    checkpoint_path_value = deployment.get("path")
    checkpoint_sha256 = deployment.get("sha256")
    if (
        source_plan.get("output_root") != str(root)
        or source_final.get("status") != UPSTREAM_TERMINAL_STATUS
        or source_final_binding.get("status") != UPSTREAM_TERMINAL_STATUS
        or not isinstance(checkpoint_path_value, str)
        or not _is_sha256(checkpoint_sha256)
        or deployment.get("policy") != "smolvla"
        or deployment.get("checkpoint_family") != "smolvla_native_event_world_model"
        or deployment.get("source_native_checkpoint") is not True
        or deployment.get("policy_feature_action_bridge_contract_sha256") != bridge_sha256
    ):
        raise RuntimeError("source binding deployment checkpoint contract is invalid")
    checkpoint_path = Path(checkpoint_path_value)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise RuntimeError("bound source deployment checkpoint is not materialized")
    resolved_checkpoint = checkpoint_path.resolve(strict=True)
    if root not in resolved_checkpoint.parents or file_sha256(resolved_checkpoint) != checkpoint_sha256:
        raise RuntimeError("bound source deployment checkpoint changed")
    if not _tree_is_read_only_materialized(root):
        raise RuntimeError("bound source output tree is no longer frozen")
    if plan is not None:
        expected_source_plan = plan["source63"]
        expected_binding_path = plan.get("source_binding_receipt")
        if expected_binding_path is not None and str(path.resolve(strict=True)) != str(
            expected_binding_path
        ):
            raise RuntimeError("source binding receipt path differs from static plan")
        current_audit = validate_source_terminal_receipt(plan)
        expected = _source_binding_payload(plan, current_audit)
        if receipt != expected or source_plan_binding != {
            "path": expected_source_plan["path"],
            "file_sha256": expected_source_plan["file_sha256"],
            "logical_sha256": expected_source_plan["static_plan_sha256"],
        }:
            raise RuntimeError("source binding receipt differs from current frozen source")
    return {
        "path": str(path.resolve(strict=True)),
        "file_sha256": before,
        "binding_sha256": logical,
        "source_final_receipt_file_sha256": source_final_binding["file_sha256"],
        "source_final_receipt_logical_sha256": source_final_binding["logical_sha256"],
        "policy_feature_action_bridge_contract_sha256": bridge_sha256,
        "deployment_rerank_checkpoint": dict(deployment),
        "checkpoint_header_revalidated_by_lobo_watcher": False,
        "checkpoint_header_validation_trust_root": receipt[
            "checkpoint_header_validation_trust_root"
        ],
        "lobo_checkpoints_rerank_authorized": False,
    }


def materialize_source_binding_receipt(
    output_root: Path,
    *,
    plan: Mapping[str, Any],
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    current = validate_source_terminal_receipt(plan)
    if dict(source_audit) != current:
        raise RuntimeError("source terminal audit changed before binding materialization")
    path = Path(
        str(plan.get("source_binding_receipt", output_root / "source_binding_receipt.json"))
    )
    if path != output_root / "source_binding_receipt.json":
        raise RuntimeError("static source binding receipt path escaped watcher output")
    immutable_json_new(path, _source_binding_payload(plan, current))
    return validate_source_binding_receipt(path, plan=plan)


def bind_lobo_stage_command(
    command: Mapping[str, Any], source_binding: Mapping[str, Any]
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "format": STAGE_SOURCE_BINDING_FORMAT,
        "stage": command["stage"],
        "held_out_body": command["held_out_body"],
        "argv_sha256": command["argv_sha256"],
        "source_binding_receipt": {
            "path": source_binding["path"],
            "file_sha256": source_binding["file_sha256"],
            "binding_sha256": source_binding["binding_sha256"],
        },
        "source_final_receipt_file_sha256": source_binding[
            "source_final_receipt_file_sha256"
        ],
        "source_final_receipt_logical_sha256": source_binding[
            "source_final_receipt_logical_sha256"
        ],
        "deployment_rerank_checkpoint": dict(
            source_binding["deployment_rerank_checkpoint"]
        ),
        "policy_feature_action_bridge_contract_sha256": source_binding[
            "policy_feature_action_bridge_contract_sha256"
        ],
        "checkpoint_header_revalidated_by_lobo_watcher": False,
        "checkpoint_header_validation_trust_root": source_binding[
            "checkpoint_header_validation_trust_root"
        ],
        "lobo_checkpoints_rerank_authorized": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return {**dict(command), "source_binding_contract": contract}


def validate_stage_source_binding_contract(
    command: Mapping[str, Any], *, plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    contract = command.get("source_binding_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("LOBO stage lacks its required source binding contract")
    descriptor = contract.get("source_binding_receipt")
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise RuntimeError("LOBO stage source binding descriptor is invalid")
    source_binding = validate_source_binding_receipt(Path(descriptor["path"]), plan=plan)
    expected = bind_lobo_stage_command(
        {key: value for key, value in command.items() if key != "source_binding_contract"},
        source_binding,
    )["source_binding_contract"]
    if dict(contract) != expected:
        raise RuntimeError("LOBO stage source binding contract changed")
    return source_binding


def wait_for_source_completion(
    plan: Mapping[str, Any],
    *,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    max_polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = Path(str(plan["source63"]["output_root"]))
    final_path = root / "final_receipt.json"
    failure_path = root / "failure_receipt.json"
    started = time.monotonic()
    polls = 0
    while True:
        polls += 1
        state.update(
            {
                "status": "waiting_for_authenticated_source63_terminal_receipt",
                "source63_polls": polls,
                "source63_wait_seconds": time.monotonic() - started,
                "source63_final_receipt_read": False,
                "watcher_hdf5_opened": 0,
                "test_hdf5_opened_by_watcher": 0,
                "last_heartbeat_unix": time.time(),
            }
        )
        atomic_json(state_path, state)
        if failure_path.exists() or failure_path.is_symlink():
            failure = load_json(failure_path, role="source63 failure receipt")
            if failure.get("status") != UPSTREAM_FAILURE_STATUS:
                raise RuntimeError("source63 failure receipt is malformed")
            raise RuntimeError("source63 training failed; LOBO stages will not start")
        if final_path.exists() or final_path.is_symlink():
            audit = validate_source_terminal_receipt(plan)
            state["source63_final_receipt_read"] = True
            state["source63_audit"] = audit
            atomic_json(state_path, state)
            return audit
        if max_polls is not None and polls >= max_polls:
            raise TimeoutError("source63 terminal receipt not available")
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("timed out waiting for source63 training")
        sleep(poll_seconds)


def _gpu_query(gpu_index: int, field: str) -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                f"--query-{field}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"unable to audit GPU {field}") from error
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def gpu_name(gpu_index: int) -> str:
    names = _gpu_query(gpu_index, "gpu=name")
    if len(names) != 1 or "4090" not in names[0]:
        raise RuntimeError(f"LOBO training requires one RTX 4090, found {names!r}")
    return names[0]


def gpu_compute_pids(gpu_index: int) -> list[int]:
    values = _gpu_query(gpu_index, "compute-apps=pid")
    result: list[int] = []
    for value in values:
        if value.lower() in (
            "no running processes found",
            "no running processes found.",
        ):
            continue
        if not value.isdigit():
            raise RuntimeError(f"unexpected GPU PID output: {value}")
        result.append(int(value))
    return sorted(set(result))


def wait_for_idle_4090(
    *,
    gpu_index: int,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    confirmations: int,
    name_fn: Callable[[int], str] = gpu_name,
    pids_fn: Callable[[int], list[int]] = gpu_compute_pids,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    name = name_fn(gpu_index)
    if "4090" not in name:
        raise RuntimeError(f"LOBO training requires RTX 4090, found {name!r}")
    started = time.monotonic()
    checks = 0
    consecutive = 0
    observations: list[dict[str, Any]] = []
    while True:
        pids = pids_fn(gpu_index)
        checks += 1
        consecutive = consecutive + 1 if not pids else 0
        observation = {
            "check": checks,
            "compute_pids": pids,
            "consecutive_idle": consecutive,
            "unix": time.time(),
        }
        observations = (observations + [observation])[-confirmations:]
        audit = {
            "gpu_index": gpu_index,
            "gpu_name": name,
            "checks": checks,
            "required_consecutive_idle": confirmations,
            "consecutive_idle": consecutive,
            "wait_seconds": time.monotonic() - started,
            "observations": observations,
        }
        state["status"] = "waiting_for_exclusive_idle_rtx4090"
        state["gpu_idle_audit"] = audit
        state["last_heartbeat_unix"] = time.time()
        atomic_json(state_path, state)
        if consecutive >= confirmations:
            return audit
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("timed out waiting for an idle RTX 4090")
        sleep(poll_seconds)


def acquire_lock(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"concurrent/stale lock exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def wait_and_acquire_gpu_lock(
    path: Path,
    payload: Mapping[str, Any],
    *,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            acquire_lock(path, payload)
            state["shared_gpu_lock"] = {
                "path": str(path),
                "token": payload["token"],
                "attempts": attempts,
                "acquired": True,
            }
            atomic_json(state_path, state)
            return
        except FileExistsError:
            state.update(
                {
                    "status": "waiting_for_shared_gpu_lock_after_source63_success",
                    "gpu_lock_attempts": attempts,
                    "last_heartbeat_unix": time.time(),
                }
            )
            atomic_json(state_path, state)
            if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
                raise TimeoutError("timed out waiting for shared GPU lock")
            sleep(poll_seconds)


def release_owned_lock(path: Path, token: str) -> None:
    try:
        value = load_json(path, role="shared GPU lock")
        if value.get("token") == token and value.get("pid") == os.getpid():
            path.unlink()
    except (OSError, RuntimeError):
        pass


def _process_group_exists(process_group_id: int) -> bool:
    if (
        isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id <= 0
    ):
        raise ValueError("process group id must be a positive integer")
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(
    process_group_id: int,
    timeout_seconds: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        sleep(0.05)
    return True


def _terminate_process_group(process_group_id: int) -> bool:
    if not _process_group_exists(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    if _wait_process_group_gone(process_group_id, 10.0):
        return True
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return _wait_process_group_gone(process_group_id, 10.0)


def run_subprocess_stage(
    command: Mapping[str, Any],
    *,
    watcher_root: Path,
    environment: Mapping[str, str],
    state: dict[str, Any],
    state_path: Path,
    gpu_index: int,
    poll_seconds: float,
    timeout_seconds: float,
    pids_fn: Callable[[int], list[int]] = gpu_compute_pids,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(command["stage"])
    source_binding_contract = command.get("source_binding_contract")
    if command.get("source_binding_required_after_upstream_terminal") is True:
        validate_stage_source_binding_contract(command)
        if not isinstance(source_binding_contract, Mapping):
            raise RuntimeError("LOBO stage cannot start without a source binding contract")
    stage_root = watcher_root / "stages" / name
    stage_root.mkdir(parents=True)
    log_path = stage_root / "run.log"
    exit_path = stage_root / "run.exit"
    receipt_path = stage_root / "stage_receipt.json"
    output = Path(str(command["output"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"preregistered stage output already exists: {output}")
    started_monotonic = time.monotonic()
    started_unix = time.time()
    process: subprocess.Popen[Any] | None = None
    error: BaseException | None = None
    cleanup_error: BaseException | None = None
    return_code: int | None = None
    process_group_id: int | None = None
    process_group_isolated = False
    if lifecycle is None:
        lifecycle = {}
    lifecycle.update(
        {
            "stage": name,
            "popen_attempted": False,
            "popen_reached": False,
            "process_pid": None,
            "process_reaped": False,
            "process_group_id": None,
            "process_group_isolated": False,
            "process_group_reaped": False,
            "returncode": None,
        }
    )
    running = {
        "format": FORMAT,
        "stage": name,
        "held_out_body": command["held_out_body"],
        "status": "launching",
        "argv": command["argv"],
        "argv_sha256": command["argv_sha256"],
        "output": str(output),
        "log": str(log_path),
        "run_exit": str(exit_path),
        "started_unix": started_unix,
        "test_hdf5_opened_by_watcher": 0,
    }
    if isinstance(source_binding_contract, Mapping):
        running.update(
            {
                "source_binding_contract": dict(source_binding_contract),
                "lobo_checkpoints_rerank_authorized": False,
                "deployment_rerank_checkpoint": dict(
                    source_binding_contract["deployment_rerank_checkpoint"]
                ),
            }
        )
    atomic_json(receipt_path, running)
    with log_path.open("x", encoding="utf-8") as log_handle:
        try:
            lifecycle["popen_attempted"] = True
            process = popen_factory(
                list(command["argv"]),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                close_fds=True,
                start_new_session=True,
            )
            lifecycle.update(
                {
                    "popen_reached": True,
                    "process_pid": process.pid,
                }
            )
            process_group_id = os.getpgid(process.pid)
            process_group_isolated = process_group_id == process.pid
            lifecycle.update(
                {
                    "process_group_id": process_group_id,
                    "process_group_isolated": process_group_isolated,
                }
            )
            if not process_group_isolated:
                raise RuntimeError("LOBO stage did not enter its own process group")
            running.update(
                {
                    "status": "running",
                    "pid": process.pid,
                    "process_group_id": process_group_id,
                    "process_group_isolated": True,
                }
            )
            atomic_json(receipt_path, running)
            state.update(
                {
                    "status": f"running_{name}",
                    "current_stage": name,
                    "stage_pid": process.pid,
                    "last_heartbeat_unix": time.time(),
                }
            )
            atomic_json(state_path, state)
            while process.poll() is None:
                if timeout_seconds > 0 and time.monotonic() - started_monotonic >= timeout_seconds:
                    raise TimeoutError(f"stage timed out: {name}")
                compute_pids = pids_fn(gpu_index)
                foreign = [pid for pid in compute_pids if pid != process.pid]
                if foreign:
                    raise RuntimeError(
                        f"foreign GPU compute process appeared during {name}: {foreign}"
                    )
                state.update(
                    {
                        "last_heartbeat_unix": time.time(),
                        "stage_elapsed_seconds": time.monotonic() - started_monotonic,
                        "stage_log_bytes": log_path.stat().st_size,
                        "stage_gpu_compute_pids": compute_pids,
                        "stage_gpu_exclusive": not foreign,
                    }
                )
                atomic_json(state_path, state)
                sleep(poll_seconds)
            return_code = int(process.returncode)
        except BaseException as caught:
            error = caught
        finally:
            if process is not None:
                try:
                    if error is not None and process.poll() is None:
                        if process_group_isolated and process_group_id is not None:
                            try:
                                os.killpg(process_group_id, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        else:
                            process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if process_group_isolated and process_group_id is not None:
                            try:
                                os.killpg(process_group_id, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait(timeout=10)
                    return_code = process.returncode
                except BaseException as caught:
                    cleanup_error = caught
                process_reaped = isinstance(process.returncode, int)
                process_group_reaped = False
                if process_group_isolated and process_group_id is not None:
                    try:
                        descendants_remained = _process_group_exists(process_group_id)
                        if descendants_remained:
                            if error is None:
                                error = RuntimeError(
                                    f"LOBO stage left live descendant processes: {name}"
                                )
                            if not _terminate_process_group(process_group_id):
                                raise RuntimeError(
                                    f"LOBO stage process group could not be reaped: {name}"
                                )
                        process_group_reaped = not _process_group_exists(
                            process_group_id
                        )
                    except BaseException as caught:
                        if cleanup_error is None:
                            cleanup_error = caught
                lifecycle.update(
                    {
                        "process_reaped": process_reaped,
                        "process_group_reaped": process_group_reaped,
                        "returncode": process.returncode,
                    }
                )
                if (
                    not process_reaped or not process_group_reaped
                ) and cleanup_error is None:
                    cleanup_error = RuntimeError(
                        f"LOBO stage process tree could not be proven reaped: {name}"
                    )
                if cleanup_error is not None and error is None:
                    error = cleanup_error
    exit_code = return_code if isinstance(return_code, int) else 127
    immutable_text_new(exit_path, f"{exit_code}\n")
    log_path.chmod(0o444)
    result = {
        **running,
        "status": "complete" if exit_code == 0 and error is None else "failed_closed",
        "returncode": exit_code,
        "process_reaped": lifecycle.get("process_reaped") is True,
        "process_group_id": process_group_id,
        "process_group_isolated": process_group_isolated,
        "process_group_reaped": lifecycle.get("process_group_reaped") is True,
        "finished_unix": time.time(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "log_sha256": file_sha256(log_path),
        "log_bytes": log_path.stat().st_size,
        "run_exit_sha256": file_sha256(exit_path),
        "gpu_exclusivity_monitored": True,
    }
    if error is not None:
        result.update({"error_type": type(error).__name__, "error": str(error)})
    if cleanup_error is not None:
        result.update(
            {
                "cleanup_error_type": type(cleanup_error).__name__,
                "cleanup_error": str(cleanup_error),
            }
        )
    atomic_json(receipt_path, result)
    if error is not None:
        raise error
    if exit_code != 0:
        raise RuntimeError(f"stage {name} failed with exit {exit_code}")
    return result


def recursive_artifact_inventory(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact symlink is forbidden: {path}")
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    result: dict[str, Any] = {
        "file_count": len(files),
        "files": files,
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def validate_lobo_output(
    command: Mapping[str, Any], *, input_sha256: Mapping[str, str]
) -> dict[str, Any]:
    source_binding = validate_stage_source_binding_contract(command)
    source_binding_contract = dict(command["source_binding_contract"])
    root = Path(str(command["output"])).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("LOBO output is not a materialized directory")
    summary_path = root / "lobo_training_summary.json"
    summary = load_json(summary_path, role="LOBO training summary")
    protocol = summary.get("protocol")
    metrics = summary.get("target_metrics")
    if (
        summary.get("format") != LOBO_FORMAT
        or summary.get("status") != LOBO_TERMINAL_STATUS
        or summary.get("held_out_body") != command["held_out_body"]
        or summary.get("estimand") != "zero_target_label_leave_one_body_out_transfer"
        or summary.get("target_development_opened_after_all_checkpoint_selection") is not True
        or summary.get("target_unused_train_payload_opened") != 0
        or summary.get("sealed_test_evaluated") is not False
        or summary.get("test_group_hdf5_opened") != 0
        or not isinstance(protocol, Mapping)
        or protocol.get("input_sha256") != dict(input_sha256)
        or protocol.get("checkpoint_selection_split") != "source_validation_only"
        or protocol.get("target_development_used_for_checkpoint_selection") is not False
        or protocol.get("target_unused_train_payload_opened") != 0
        or protocol.get("test_group_hdf5_opened") != 0
        or protocol.get("frozen_split_plan", {}).get("file_sha256")
        != command["split_file_sha256"]
        or not isinstance(metrics, Mapping)
        or not {"source_body_clock", "body_agnostic"}.issubset(metrics)
    ):
        raise RuntimeError("LOBO training summary violates the frozen protocol")
    checkpoint_sha256: dict[str, list[str]] = {}
    for variant in ("source_body_clock", "body_agnostic"):
        selection_path = root / variant / "source_selection_summary.json"
        selection = load_json(selection_path, role=f"{variant} selection summary")
        members = selection.get("members")
        if (
            selection.get("format") != LOBO_FORMAT
            or selection.get("variant") != variant
            or selection.get("held_out_body") != command["held_out_body"]
            or selection.get("checkpoint_selection_split") != "source_validation_only"
            or selection.get("target_development_opened") != 0
            or selection.get("test_group_hdf5_opened") != 0
            or not isinstance(members, list)
            or len(members) != 5
            or [member.get("seed") for member in members] != list(ENSEMBLE_SEEDS)
            or [member.get("member") for member in members] != list(range(5))
        ):
            raise RuntimeError(f"{variant} source selection receipt is invalid")
        hashes: list[str] = []
        for member in members:
            checkpoint = Path(str(member.get("checkpoint", ""))).resolve(strict=True)
            if root not in checkpoint.parents:
                raise RuntimeError("LOBO checkpoint escaped its output root")
            digest = file_sha256(checkpoint)
            if digest != member.get("checkpoint_sha256"):
                raise RuntimeError("LOBO checkpoint hash changed")
            hashes.append(digest)
        if metrics[variant].get("evaluated_checkpoint_sha256") != hashes:
            raise RuntimeError("target metrics were not evaluated with selected checkpoints")
        checkpoint_sha256[variant] = hashes
    inventory = recursive_artifact_inventory(root)
    return {
        "status": LOBO_TERMINAL_STATUS,
        "held_out_body": command["held_out_body"],
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_binding_contract": source_binding_contract,
        "source_binding_receipt_file_sha256": source_binding["file_sha256"],
        "source_binding_sha256": source_binding["binding_sha256"],
        "deployment_rerank_checkpoint": dict(
            source_binding["deployment_rerank_checkpoint"]
        ),
        "policy_feature_action_bridge_contract_sha256": source_binding[
            "policy_feature_action_bridge_contract_sha256"
        ],
        "lobo_checkpoints_rerank_authorized": False,
        "artifact_inventory_sha256": inventory["inventory_sha256"],
        "artifact_file_count": inventory["file_count"],
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
        "test_labels_read_by_watcher": False,
    }


def freeze_tree(
    root: Path, *, exclude_root_files: Sequence[str] = ()
) -> None:
    if not root.exists() or root.is_symlink():
        return
    excluded = set(exclude_root_files)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"cannot freeze symlink artifact: {path}")
        if path.is_file():
            if path.parent == root and path.name in excluded:
                continue
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _verify_frozen_tree_before_terminal_publish(
    root: Path, *, hidden_terminals: Sequence[Path]
) -> None:
    expected_hidden = set(hidden_terminals)
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise RuntimeError("LOBO output root was not frozen before terminal publication")
    observed_hidden: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"frozen LOBO output contains a symlink: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path in expected_hidden:
            observed_hidden.add(path)
            if not path.is_file() or mode != 0:
                raise RuntimeError("LOBO terminal became readable before publication")
        elif path.is_file() and mode != 0o444:
            raise RuntimeError(f"frozen LOBO output file mode changed: {path}")
        elif path.is_dir() and mode != 0o555:
            raise RuntimeError(f"frozen LOBO output directory mode changed: {path}")
    if observed_hidden != expected_hidden:
        raise RuntimeError("one or more hidden LOBO terminal files disappeared")


def _create_hidden_terminal(path: Path, payload: bytes) -> tuple[int, int]:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0)
    created = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return created.st_dev, created.st_ino


def publish_frozen_terminal_receipt(
    root: Path,
    *,
    terminal_name: str,
    receipt: Mapping[str, Any],
    exit_code: int,
) -> None:
    if terminal_name not in ("final_receipt.json", "failure_receipt.json"):
        raise ValueError("unauthorized LOBO terminal receipt name")
    expected_exit_code = {
        "final_receipt.json": 0,
        "failure_receipt.json": 1,
    }[terminal_name]
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code != expected_exit_code
    ):
        raise ValueError("LOBO terminal name and exit code are inconsistent")
    if receipt.get("artifacts_frozen_read_only") is not True:
        raise RuntimeError("published LOBO terminal must claim a frozen tree")
    if terminal_name == "final_receipt.json":
        validate_lobo_success_terminal_receipt(receipt)
    terminal_path = root / terminal_name
    exit_path = root / "run.exit"
    opposite = root / (
        "failure_receipt.json"
        if terminal_name == "final_receipt.json"
        else "final_receipt.json"
    )
    for path in (terminal_path, exit_path, opposite):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    receipt_payload = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    exit_payload = f"{exit_code}\n".encode("ascii")
    created: dict[Path, tuple[int, int]] = {}
    published = False
    try:
        created[terminal_path] = _create_hidden_terminal(
            terminal_path, receipt_payload
        )
        created[exit_path] = _create_hidden_terminal(exit_path, exit_payload)
        freeze_tree(
            root,
            exclude_root_files=(terminal_path.name, exit_path.name),
        )
        _verify_frozen_tree_before_terminal_publish(
            root, hidden_terminals=(terminal_path, exit_path)
        )
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        exit_path.chmod(0o444)
        if stat.S_IMODE(exit_path.stat().st_mode) != 0o444:
            raise RuntimeError("LOBO run.exit publication mode is invalid")
        terminal_path.chmod(0o444)
        published = True
    finally:
        if not published:
            try:
                root.chmod(0o700)
            except OSError:
                pass
            for path, identity in created.items():
                try:
                    current = path.lstat()
                    if (
                        stat.S_ISREG(current.st_mode)
                        and (current.st_dev, current.st_ino) == identity
                    ):
                        path.unlink()
                except (FileNotFoundError, OSError):
                    pass


def prepare_execution(plan: Mapping[str, Any]) -> tuple[Path, str]:
    output = Path(str(plan["output_root"]))
    output.mkdir(mode=0o700)
    token = canonical_sha256(
        {"pid": os.getpid(), "plan": plan["static_plan_sha256"], "time": time.time_ns()}
    )
    immutable_json_new(output / "launch_plan.json", dict(plan))
    acquire_lock(
        output / "launch.lock",
        {
            "format": FORMAT,
            "pid": os.getpid(),
            "token": token,
            "static_plan_sha256": plan["static_plan_sha256"],
        },
    )
    return output, token


def load_prepared_plan(output: Path, token: str) -> dict[str, Any]:
    output = resolve_existing_path(output, role="prepared watcher output", directory=True)
    plan_path = resolve_existing_path(
        output / "launch_plan.json", role="prepared launch plan", directory=False
    )
    plan = load_json(plan_path, role="prepared launch plan")
    unsigned = dict(plan)
    logical = unsigned.pop("static_plan_sha256", None)
    if logical != canonical_sha256(unsigned) or plan.get("output_root") != str(output):
        raise RuntimeError("prepared launch plan integrity failed")
    lock = load_json(output / "launch.lock", role="prepared launch lock")
    if (
        lock.get("format") != FORMAT
        or lock.get("token") != token
        or lock.get("static_plan_sha256") != logical
    ):
        raise RuntimeError("prepared launch claim does not match")
    for path in plan["preregistered_outputs"].values():
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError(f"preregistered output was claimed by another process: {path}")
    return plan


def _validate_stage_process_proof(
    *,
    stage: str,
    result: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> None:
    process_pid = result.get("pid")
    result_process_group_id = result.get("process_group_id")
    lifecycle_process_pid = lifecycle.get("process_pid")
    lifecycle_process_group_id = lifecycle.get("process_group_id")
    result_returncode = result.get("returncode")
    lifecycle_returncode = lifecycle.get("returncode")
    if (
        isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid <= 0
        or result.get("stage") != stage
        or result.get("status") != "complete"
        or isinstance(result_returncode, bool)
        or not isinstance(result_returncode, int)
        or result_returncode != 0
        or result.get("process_reaped") is not True
        or isinstance(result_process_group_id, bool)
        or not isinstance(result_process_group_id, int)
        or result_process_group_id != process_pid
        or result.get("process_group_isolated") is not True
        or result.get("process_group_reaped") is not True
        or lifecycle.get("stage") != stage
        or lifecycle.get("popen_attempted") is not True
        or lifecycle.get("popen_reached") is not True
        or isinstance(lifecycle_process_pid, bool)
        or not isinstance(lifecycle_process_pid, int)
        or lifecycle_process_pid != process_pid
        or lifecycle.get("process_reaped") is not True
        or isinstance(lifecycle_process_group_id, bool)
        or not isinstance(lifecycle_process_group_id, int)
        or lifecycle_process_group_id != process_pid
        or lifecycle.get("process_group_isolated") is not True
        or lifecycle.get("process_group_reaped") is not True
        or isinstance(lifecycle_returncode, bool)
        or not isinstance(lifecycle_returncode, int)
        or lifecycle_returncode != 0
    ):
        raise RuntimeError(
            f"LOBO stage process lifecycle proof is inconsistent: {stage}"
        )


def validate_lobo_success_terminal_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(receipt)
    logical = unsigned.pop("receipt_sha256", None)
    order = receipt.get("execution_order")
    results = receipt.get("stage_results")
    lifecycles = receipt.get("stage_lifecycles")
    if (
        logical != canonical_sha256(unsigned)
        or receipt.get("format") != FORMAT
        or receipt.get("status") != TERMINAL_STATUS
        or order != ["train_lobo_piper", "train_lobo_ur5"]
        or not isinstance(results, Mapping)
        or set(results) != set(order)
        or not isinstance(lifecycles, list)
        or len(lifecycles) != len(order)
        or receipt.get("lobo_checkpoints_rerank_authorized") is not False
        or receipt.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
        or receipt.get("fresh_inputs_accepted") is not False
        or receipt.get("fresh_labels_read") is not False
        or receipt.get("watcher_hdf5_opened") != 0
        or receipt.get("test_hdf5_opened_by_watcher") != 0
        or receipt.get("test_labels_read_by_watcher") is not False
        or receipt.get("target_unused_train_payload_opened") != 0
        or receipt.get("test_group_hdf5_opened") != 0
        or receipt.get("artifacts_frozen_read_only") is not True
    ):
        raise RuntimeError("LOBO success terminal receipt semantics are invalid")
    lifecycle_by_stage: dict[str, Mapping[str, Any]] = {}
    for expected_stage, lifecycle in zip(order, lifecycles):
        if (
            not isinstance(lifecycle, Mapping)
            or lifecycle.get("stage") != expected_stage
            or expected_stage in lifecycle_by_stage
        ):
            raise RuntimeError("LOBO success lifecycle order is invalid")
        lifecycle_by_stage[expected_stage] = lifecycle
    for stage in order:
        result = results.get(stage)
        if not isinstance(result, Mapping):
            raise RuntimeError("LOBO success stage result is invalid")
        artifact_audit = result.get("artifact_audit")
        if (
            not isinstance(artifact_audit, Mapping)
            or artifact_audit.get("status") != LOBO_TERMINAL_STATUS
            or artifact_audit.get("held_out_body") != result.get("held_out_body")
            or artifact_audit.get("lobo_checkpoints_rerank_authorized") is not False
            or artifact_audit.get("target_unused_train_payload_opened") != 0
            or artifact_audit.get("test_group_hdf5_opened") != 0
            or artifact_audit.get("test_labels_read_by_watcher") is not False
        ):
            raise RuntimeError("LOBO success artifact audit is invalid")
        _validate_stage_process_proof(
            stage=stage,
            result=result,
            lifecycle=lifecycle_by_stage[stage],
        )
    return dict(receipt)


def build_final_receipt(
    plan: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    current_source_audit = validate_source_terminal_receipt(plan)
    if dict(source_audit) != current_source_audit:
        raise RuntimeError("source terminal audit changed before final LOBO receipt")
    current_binding = validate_source_binding_receipt(
        Path(str(source_binding["path"])), plan=plan
    )
    if dict(source_binding) != current_binding:
        raise RuntimeError("source binding changed before final LOBO receipt")
    stage_results = state.get("stage_results")
    stage_lifecycles = state.get("stage_lifecycles")
    if not isinstance(stage_results, Mapping) or set(stage_results) != set(
        plan["execution_order"]
    ):
        raise RuntimeError("final LOBO receipt lacks one or more stage results")
    if not isinstance(stage_lifecycles, list) or len(stage_lifecycles) != len(
        plan["execution_order"]
    ):
        raise RuntimeError("final LOBO receipt lacks ordered stage lifecycles")
    lifecycle_by_stage: dict[str, Mapping[str, Any]] = {}
    for expected_stage, lifecycle in zip(plan["execution_order"], stage_lifecycles):
        if (
            not isinstance(lifecycle, Mapping)
            or lifecycle.get("stage") != expected_stage
            or expected_stage in lifecycle_by_stage
        ):
            raise RuntimeError("final LOBO receipt has inconsistent stage lifecycles")
        lifecycle_by_stage[expected_stage] = lifecycle
    for base_command in plan["commands"]:
        expected_contract = bind_lobo_stage_command(
            base_command, current_binding
        )["source_binding_contract"]
        result = stage_results.get(base_command["stage"])
        artifact_audit = result.get("artifact_audit") if isinstance(result, Mapping) else None
        if isinstance(result, Mapping):
            _validate_stage_process_proof(
                stage=base_command["stage"],
                result=result,
                lifecycle=lifecycle_by_stage[base_command["stage"]],
            )
        if (
            not isinstance(result, Mapping)
            or result.get("source_binding_contract") != expected_contract
            or result.get("lobo_checkpoints_rerank_authorized") is not False
            or result.get("deployment_rerank_checkpoint")
            != current_binding["deployment_rerank_checkpoint"]
            or not isinstance(artifact_audit, Mapping)
            or artifact_audit.get("status") != LOBO_TERMINAL_STATUS
            or artifact_audit.get("held_out_body") != base_command["held_out_body"]
            or artifact_audit.get("source_binding_contract") != expected_contract
            or artifact_audit.get("lobo_checkpoints_rerank_authorized") is not False
            or artifact_audit.get("deployment_rerank_checkpoint")
            != current_binding["deployment_rerank_checkpoint"]
        ):
            raise RuntimeError("LOBO stage did not preserve its source binding contract")
    receipt: dict[str, Any] = {
        "format": FORMAT,
        "status": TERMINAL_STATUS,
        "static_plan_sha256": plan["static_plan_sha256"],
        "source63_audit": current_source_audit,
        "source_binding_receipt": dict(current_binding),
        "source_final_receipt_file_sha256": current_binding[
            "source_final_receipt_file_sha256"
        ],
        "source_final_receipt_logical_sha256": current_binding[
            "source_final_receipt_logical_sha256"
        ],
        "deployment_rerank_checkpoint": dict(
            current_binding["deployment_rerank_checkpoint"]
        ),
        "policy_feature_action_bridge_contract_sha256": current_binding[
            "policy_feature_action_bridge_contract_sha256"
        ],
        "checkpoint_header_revalidated_by_lobo_watcher": False,
        "checkpoint_header_validation_trust_root": current_binding[
            "checkpoint_header_validation_trust_root"
        ],
        "lobo_checkpoint_role": "canonical_cross_embodiment_scientific_evaluation_only",
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
        "execution_order": list(plan["execution_order"]),
        "stage_results": dict(stage_results),
        "stage_lifecycles": list(stage_lifecycles),
        "preregistered_outputs": dict(plan["preregistered_outputs"]),
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
        "artifacts_frozen_read_only": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return validate_lobo_success_terminal_receipt(receipt)


def _unreaped_lobo_stage(
    stage_lifecycles: Sequence[Mapping[str, Any]],
) -> str | None:
    for lifecycle in stage_lifecycles:
        if lifecycle.get("popen_attempted") is True and not (
            lifecycle.get("popen_reached") is True
            and lifecycle.get("process_reaped") is True
            and lifecycle.get("process_group_isolated") is True
            and lifecycle.get("process_group_id") == lifecycle.get("process_pid")
            and lifecycle.get("process_group_reaped") is True
        ):
            stage = lifecycle.get("stage")
            return str(stage) if stage is not None else "unknown_lobo_stage"
    return None


def _owned_gpu_lock_release_allowed(
    *, gpu_lock_acquired: bool, stage_lifecycles: Sequence[Mapping[str, Any]]
) -> bool:
    return gpu_lock_acquired and _unreaped_lobo_stage(stage_lifecycles) is None


def execute_prepared(output_arg: Path, token: str) -> dict[str, Any]:
    output = Path(os.path.abspath(os.fspath(output_arg.expanduser())))
    plan = load_prepared_plan(output, token)
    state_path = output / "launch_state.json"
    watcher_exit = output / "run.exit"
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "starting_prepared_autonomous_watcher",
        "pid": os.getpid(),
        "static_plan_sha256": plan["static_plan_sha256"],
        "execution_order": plan["execution_order"],
        "stage_results": {},
        "stage_lifecycles": [],
        "stages_started": [],
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "watcher_hdf5_imported": False,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
    }
    atomic_json(state_path, state)
    gpu_lock = Path(str(plan["gpu_lock"]))
    gpu_token = canonical_sha256(
        {"pid": os.getpid(), "plan": plan["static_plan_sha256"], "scope": "gpu"}
    )
    gpu_payload = {
        "format": FORMAT,
        "pid": os.getpid(),
        "token": gpu_token,
        "static_plan_sha256": plan["static_plan_sha256"],
    }
    gpu_lock_acquired = False
    stage_lifecycles: list[dict[str, Any]] = []
    state["stage_lifecycles"] = stage_lifecycles
    try:
        verify_implementation_unchanged(plan)
        source_audit = wait_for_source_completion(
            plan,
            state=state,
            state_path=state_path,
            poll_seconds=float(plan["poll_seconds"]),
            timeout_seconds=float(plan["source_timeout_seconds"]),
        )
        state["source63_audit"] = source_audit
        source_binding = materialize_source_binding_receipt(
            output, plan=plan, source_audit=source_audit
        )
        state["source_binding_receipt"] = source_binding
        state["lobo_checkpoints_rerank_authorized"] = False
        state["deployment_rerank_checkpoint"] = source_binding[
            "deployment_rerank_checkpoint"
        ]
        atomic_json(state_path, state)
        verify_implementation_unchanged(plan)
        wait_and_acquire_gpu_lock(
            gpu_lock,
            gpu_payload,
            state=state,
            state_path=state_path,
            poll_seconds=float(plan["poll_seconds"]),
            timeout_seconds=float(plan["gpu_timeout_seconds"]),
        )
        gpu_lock_acquired = True
        state["gpu_lock_acquired"] = True
        atomic_json(state_path, state)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(plan["gpu_index"]),
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": str(plan["omp_threads"]),
            }
        )
        for base_command in plan["commands"]:
            source_binding = validate_source_binding_receipt(
                Path(str(source_binding["path"])), plan=plan
            )
            command = bind_lobo_stage_command(base_command, source_binding)
            verify_implementation_unchanged(plan)
            idle = wait_for_idle_4090(
                gpu_index=int(plan["gpu_index"]),
                state=state,
                state_path=state_path,
                poll_seconds=float(plan["poll_seconds"]),
                timeout_seconds=float(plan["gpu_timeout_seconds"]),
                confirmations=int(plan["idle_confirmations"]),
            )
            state["stages_started"].append(command["stage"])
            state["pre_stage_gpu_idle_audit"] = idle
            lifecycle: dict[str, Any] = {"stage": command["stage"]}
            stage_lifecycles.append(lifecycle)
            atomic_json(state_path, state)
            result = run_subprocess_stage(
                command,
                watcher_root=output,
                environment=environment,
                state=state,
                state_path=state_path,
                gpu_index=int(plan["gpu_index"]),
                poll_seconds=float(plan["poll_seconds"]),
                timeout_seconds=float(plan["stage_timeout_seconds"]),
                lifecycle=lifecycle,
            )
            verify_implementation_unchanged(plan)
            artifact_audit = validate_lobo_output(
                command, input_sha256=plan["input_sha256"]
            )
            result["artifact_audit"] = artifact_audit
            stage_receipt = output / "stages" / command["stage"] / "stage_receipt.json"
            atomic_json(stage_receipt, result)
            freeze_tree(Path(command["output"]))
            state["stage_results"][command["stage"]] = result
            state["current_stage"] = None
            state["stage_pid"] = None
            atomic_json(state_path, state)
        state.update(
            {
                "status": "terminal_success_pending_frozen_publication",
                "finished_unix": time.time(),
                "target_unused_train_payload_opened": 0,
                "test_group_hdf5_opened": 0,
                "test_hdf5_opened_by_watcher": 0,
                "test_labels_read_by_watcher": False,
                "lobo_checkpoints_rerank_authorized": False,
                "deployment_rerank_checkpoint": source_binding[
                    "deployment_rerank_checkpoint"
                ],
            }
        )
        atomic_json(state_path, state)
        source_binding = validate_source_binding_receipt(
            Path(str(source_binding["path"])), plan=plan
        )
        final_receipt = build_final_receipt(
            plan,
            state=state,
            source_audit=source_audit,
            source_binding=source_binding,
        )
        validate_lobo_success_terminal_receipt(final_receipt)
        publish_frozen_terminal_receipt(
            output,
            terminal_name="final_receipt.json",
            receipt=final_receipt,
            exit_code=0,
        )
        return final_receipt
    except BaseException as error:
        unreaped_stage = _unreaped_lobo_stage(stage_lifecycles)
        state.update(
            {
                "status": FAILURE_STATUS,
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_unix": time.time(),
                "later_stages_started_after_failure": False,
                "fresh_inputs_accepted": False,
                "fresh_labels_read": False,
                "watcher_hdf5_opened": 0,
                "test_hdf5_opened_by_watcher": 0,
                "test_labels_read_by_watcher": False,
                "stage_lifecycles": stage_lifecycles,
                "unreaped_stage_process": unreaped_stage,
                "gpu_lock_acquired": gpu_lock_acquired,
                "gpu_lock_retained_for_unreaped_stage_process": (
                    unreaped_stage is not None and gpu_lock_acquired
                ),
                "artifacts_frozen_read_only": False,
                "terminal_publication_requires_successful_freeze": (
                    unreaped_stage is None
                ),
            }
        )
        try:
            atomic_json(state_path, state)
            failure: dict[str, Any] = {
                "format": FORMAT,
                "status": FAILURE_STATUS,
                "error_type": type(error).__name__,
                "error": str(error),
                "static_plan_sha256": plan["static_plan_sha256"],
                "stages_started": state["stages_started"],
                "execution_order": plan["execution_order"],
                "later_stages_started_after_failure": False,
                "fresh_inputs_accepted": False,
                "fresh_labels_read": False,
                "watcher_hdf5_opened": 0,
                "test_hdf5_opened_by_watcher": 0,
                "test_labels_read_by_watcher": False,
                "stage_lifecycles": stage_lifecycles,
                "unreaped_stage_process": unreaped_stage,
                "gpu_lock_acquired": gpu_lock_acquired,
                "gpu_lock_retained_for_unreaped_stage_process": (
                    unreaped_stage is not None and gpu_lock_acquired
                ),
                "artifacts_frozen_read_only": unreaped_stage is None,
            }
            failure["receipt_sha256"] = canonical_sha256(failure)
            if unreaped_stage is None:
                for command in plan["commands"]:
                    child = Path(command["output"])
                    if child.exists() and not child.is_symlink():
                        freeze_tree(child)
                publish_frozen_terminal_receipt(
                    output,
                    terminal_name="failure_receipt.json",
                    receipt=failure,
                    exit_code=1,
                )
            else:
                atomic_json(output / "failure_receipt.json", failure)
                if not watcher_exit.exists():
                    immutable_text_new(watcher_exit, "1\n")
        except BaseException:
            pass
        raise
    finally:
        if _owned_gpu_lock_release_allowed(
            gpu_lock_acquired=gpu_lock_acquired,
            stage_lifecycles=stage_lifecycles,
        ):
            release_owned_lock(gpu_lock, gpu_token)


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = static_preflight(args)
    output, token = prepare_execution(plan)
    return execute_prepared(output, token)


def detach(args: argparse.Namespace) -> dict[str, Any]:
    plan = static_preflight(args)
    output = Path(str(plan["output_root"]))
    receipt_path = resolve_new_path(
        args.detach_receipt
        or output.parent / f"{output.name}.detach_receipt.json",
        role="detach receipt",
    )
    daemon_log = resolve_new_path(
        args.detach_log or output.parent / f"{output.name}.launcher.log",
        role="detached watcher log",
    )
    if receipt_path == output or output in receipt_path.parents:
        raise ValueError("detach receipt must remain outside the watcher output tree")
    if daemon_log == output or output in daemon_log.parents:
        raise ValueError("detached log must remain outside the watcher output tree")
    output, token = prepare_execution(plan)
    receipt_descriptor = os.open(
        receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    argv = [
        str(plan["python_contract"]["invocation_path"]),
        str(Path(__file__).resolve()),
        "_run-prepared",
        "--output",
        str(output),
        "--claim-token",
        token,
    ]
    try:
        with daemon_log.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as error:
        failed_receipt: dict[str, Any] = {
            "format": DETACH_FORMAT,
            "status": "failed_before_detached_server_process_start",
            "error_type": type(error).__name__,
            "error": str(error),
            "output_root": str(output),
            "static_plan_sha256": plan["static_plan_sha256"],
            "preregistered_outputs": plan["preregistered_outputs"],
            "outputs_preregistered_before_process_start": True,
            "fresh_inputs_accepted": False,
            "fresh_labels_read": False,
            "watcher_hdf5_opened": 0,
            "test_hdf5_opened_by_watcher": 0,
        }
        failed_receipt["receipt_sha256"] = canonical_sha256(failed_receipt)
        _write_json_to_claimed_fd(receipt_descriptor, failed_receipt)
        raise
    receipt: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detached_server_side_lobo_watcher_started",
        "pid": process.pid,
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "output_root": str(output),
        "daemon_log": str(daemon_log),
        "static_plan_sha256": plan["static_plan_sha256"],
        "preregistered_outputs": plan["preregistered_outputs"],
        "survives_client_disconnect": True,
        "new_os_session": True,
        "outputs_preregistered_before_process_start": True,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_json_to_claimed_fd(receipt_descriptor, receipt)
    print("LOBO_AUTONOMOUS_DETACHED=" + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--code-root", type=Path, default=root)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--source-training-root", type=Path, required=True)
    parser.add_argument("--source-launch-plan", type=Path)
    parser.add_argument("--source-launch-plan-sha256", required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1-source-manifest", type=Path, required=True)
    parser.add_argument("--stage1-source-manifest-sha256", required=True)
    parser.add_argument("--stage1-target-manifest", type=Path, required=True)
    parser.add_argument("--stage1-target-manifest-sha256", required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--event-spec-sha256", required=True)
    parser.add_argument("--openvla-schema5-manifest", type=Path, required=True)
    parser.add_argument("--openvla-schema5-manifest-sha256", required=True)
    parser.add_argument("--piper-split", type=Path, required=True)
    parser.add_argument("--piper-split-sha256", required=True)
    parser.add_argument("--ur5-split", type=Path, required=True)
    parser.add_argument("--ur5-split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--piper-output", type=Path, required=True)
    parser.add_argument("--ur5-output", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--source-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--stage-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--idle-confirmations", type=int, default=2)
    parser.add_argument("--omp-threads", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("preflight", "Validate and print the preregistration without writing outputs."),
        ("run", "Preregister and run the watcher in the foreground."),
        ("detach", "Preregister and launch a new-session server-side watcher."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        add_common_arguments(child)
        if command == "detach":
            child.add_argument("--detach-receipt", type=Path)
            child.add_argument("--detach-log", type=Path)
    prepared = subparsers.add_parser("_run-prepared", help=argparse.SUPPRESS)
    prepared.add_argument("--output", type=Path, required=True)
    prepared.add_argument("--claim-token", required=True)
    args = parser.parse_args()
    if args.command != "_run-prepared" and (
        args.gpu_index < 0
        or not 0 < args.poll_seconds <= 60
        or args.source_timeout_seconds < 0
        or args.gpu_timeout_seconds < 0
        or args.stage_timeout_seconds < 0
        or not 1 <= args.idle_confirmations <= 10
        or args.omp_threads <= 0
    ):
        parser.error("invalid LOBO watcher timing/device arguments")
    if args.command == "_run-prepared" and not _is_sha256(args.claim_token):
        parser.error("prepared claim token must be a SHA-256")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        print("LOBO_AUTONOMOUS_PREFLIGHT=" + json.dumps(static_preflight(args), sort_keys=True))
    elif args.command == "detach":
        detach(args)
    elif args.command == "_run-prepared":
        result = execute_prepared(args.output, args.claim_token)
        print("LOBO_AUTONOMOUS_COMPLETE=" + json.dumps(result, sort_keys=True))
    else:
        result = run(args)
        print("LOBO_AUTONOMOUS_COMPLETE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BATCH_SIZE",
    "ENSEMBLE_SEEDS",
    "EVAL_EVERY",
    "FORMAT",
    "LOBO_STAGES",
    "SPLIT_SEED",
    "TRAINING_STEPS",
    "build_lobo_command",
    "canonical_sha256",
    "detach",
    "file_sha256",
    "load_prepared_plan",
    "prepare_execution",
    "resolve_new_path",
    "run_subprocess_stage",
    "static_preflight",
    "validate_lobo_output",
    "validate_lobo_split",
    "validate_source_launch_plan",
    "validate_source_terminal_receipt",
    "wait_for_idle_4090",
    "wait_for_source_completion",
]
