#!/usr/bin/env python3
"""Detached post-collection pipeline for the Piper Schema6 adapter ensemble.

The watcher starts from an authenticated 130-group collection terminal receipt,
materializes a label-blind 60/20/50 manifest, trains five source-member-matched
adapters, and performs internal-validation-only ensemble calibration.  It then
publishes a request for a *separate* 400-pair execution authority and waits.  It
never executes the paired experiment and never opens target-validation50 or
evaluation400 HDF/labels.
"""

from __future__ import annotations

import argparse
import fcntl
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

import calibrate_smolvla_piper_adapter_ensemble as calibration
import evaluate_smolvla_piper_schema6_target_validation50_ensemble as target_validation
import materialize_smolvla_piper_schema6_training_manifest_v2 as materializer
from launch_smolvla_piper_schema6_autonomous_watcher import (
    SOURCE_MEMBER_SEEDS,
    validate_source_training_summary,
)
from launch_smolvla_piper_schema6_multiseed_v2 import (
    FORMAT as COLLECTION_WATCHER_FORMAT,
    PLAN_FORMAT as COLLECTION_PLAN_FORMAT,
    TERMINAL_STATUS as COLLECTION_TERMINAL_STATUS,
)
from preregister_smolvla_piper_schema6_multiseed_collection_v2 import (
    validate_preregistration,
)


FORMAT = "etsf_smolvla_piper_schema6_post_collection_watcher_v2"
PLAN_FORMAT = "etsf_smolvla_piper_schema6_post_collection_plan_v2"
STATE_FORMAT = "etsf_smolvla_piper_schema6_post_collection_state_v2"
DETACH_FORMAT = "etsf_smolvla_piper_schema6_post_collection_detach_v2"
MEMBER_RECEIPT_FORMAT = "etsf_smolvla_piper_schema6_adapter_member_receipt_v2"
CALIBRATION_INPUT_FORMAT = calibration.INPUT_FORMAT
AUTHORIZATION_REQUEST_FORMAT = (
    "etsf_smolvla_piper_paired400_execution_authorization_request_v1"
)
INDEPENDENT_AUTHORITY_FORMAT = (
    "etsf_smolvla_piper_paired400_independent_execution_authority_v1"
)
HANDOFF_FORMAT = "etsf_smolvla_piper_paired400_external_handoff_v1"
TERMINAL_STATUS = (
    "complete_adapter_ensemble_calibration_independent_paired400_authority_verified"
)
FAILURE_STATUS = "failed_closed_schema6_post_collection_pipeline"
WAITING_STATUS = "waiting_for_independent_paired400_execution_authority"
EXPECTED_GPU_FRAGMENT = "RTX 4090"
MEMBER_COUNT = 5
ADAPTER_STEPS = 3000
ADAPTER_EVAL_EVERY = 50
SENSITIVE_TOKENS = ("fresh", "confirmation")
HDF_SUFFIXES = (".h5", ".hdf", ".hdf5")
SHA_CHARS = frozenset("0123456789abcdef")


class PostCollectionError(RuntimeError):
    """A post-collection authority, stage, or secrecy invariant failed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    if path.suffix.casefold() in HDF_SUFFIXES:
        raise PostCollectionError("post-collection watcher cannot hash HDF bytes")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _sensitive(path: PurePath) -> bool:
    return any(
        token in component.casefold()
        for component in path.parts
        for token in SENSITIVE_TOKENS
    )


def safe_path(value: str | os.PathLike[str], role: str) -> Path:
    text = os.fspath(value)
    if not text or "\0" in text:
        raise PostCollectionError(f"{role} path is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(text)))
    if _sensitive(PurePath(lexical)):
        raise PostCollectionError(f"{role} is in a forbidden namespace")
    resolved = lexical.resolve(strict=False)
    if _sensitive(PurePath(resolved)):
        raise PostCollectionError(f"{role} resolves into a forbidden namespace")
    return resolved


def existing_file(
    value: str | os.PathLike[str], role: str, *, allow_writable: bool = False
) -> Path:
    path = safe_path(value, role)
    if path.is_symlink():
        raise PostCollectionError(f"{role} must not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or resolved.suffix.casefold() in HDF_SUFFIXES:
        raise PostCollectionError(f"{role} is not a safe non-HDF regular file")
    if not allow_writable and metadata.st_mode & 0o222:
        raise PostCollectionError(f"{role} must be frozen read-only")
    return resolved


def existing_directory(
    value: str | os.PathLike[str], role: str, *, frozen: bool = False
) -> Path:
    path = safe_path(value, role)
    if path.is_symlink():
        raise PostCollectionError(f"{role} must not be a symlink")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise PostCollectionError(f"{role} must be a directory")
    if frozen and metadata.st_mode & 0o222:
        raise PostCollectionError(f"{role} must be frozen read-only")
    return resolved


def load_json(path: Path, role: str, *, frozen: bool = True) -> dict[str, Any]:
    path = existing_file(path, role, allow_writable=not frozen)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostCollectionError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise PostCollectionError(f"{role} must contain an object")
    return value


def verify_signed(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise PostCollectionError(f"{role} logical SHA mismatch")
    return str(recorded)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def immutable_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def freeze_tree(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            item = base / name
            if item.is_symlink():
                raise PostCollectionError("output contains a symlink")
            item.chmod(0o444)
        for name in names:
            item = base / name
            if item.is_symlink():
                raise PostCollectionError("output contains a symlink")
            item.chmod(0o555)
        base.chmod(0o555)


def _run_text(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def gpu_audit(
    gpu_index: int,
    run_text: Callable[[Sequence[str]], str] = _run_text,
) -> dict[str, Any]:
    identity = run_text(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=name,uuid",
            "--format=csv,noheader",
        ]
    ).strip().split(",", 1)
    if len(identity) != 2 or EXPECTED_GPU_FRAGMENT not in identity[0].strip():
        raise PostCollectionError("designated GPU is not an RTX 4090")
    raw = run_text(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    pids = sorted({int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()})
    return {
        "gpu_index": gpu_index,
        "name": identity[0].strip(),
        "uuid": identity[1].strip(),
        "compute_pids": pids,
    }


def wait_two_idle(
    gpu_index: int,
    *,
    interval: float,
    run_text: Callable[[Sequence[str]], str] = _run_text,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    consecutive: list[dict[str, Any]] = []
    while True:
        audit = gpu_audit(gpu_index, run_text)
        if audit["compute_pids"]:
            consecutive.clear()
        else:
            if consecutive and audit["uuid"] != consecutive[0]["uuid"]:
                raise PostCollectionError("GPU identity changed while waiting")
            consecutive.append(audit)
            if len(consecutive) == 2:
                return consecutive
        sleep(interval)


def wait_for_ppid1(
    *,
    getppid: Callable[[], int] = os.getppid,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    while getppid() != 1:
        sleep(0.1)


def _is_descendant(pid: int, ancestor: int) -> bool:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().split()[3])
        except (OSError, ValueError, IndexError):
            return False
    return False


def validate_collection_terminal(collection_root: Path) -> dict[str, Any]:
    """Authenticate terminal collection metadata without HDF byte access."""

    root = existing_directory(collection_root, "collection root", frozen=True)
    watcher = root / "_watcher"
    final_path = watcher / "final_receipt.json"
    plan_path = watcher / "static_plan.json"
    exit_path = watcher / "run.exit"
    final = load_json(final_path, "collection terminal receipt")
    plan = load_json(plan_path, "collection static plan")
    final_logical = verify_signed(final, "receipt_sha256", "collection terminal receipt")
    plan_logical = verify_signed(plan, "plan_sha256", "collection static plan")
    if (
        final.get("format") != COLLECTION_WATCHER_FORMAT
        or final.get("status") != COLLECTION_TERMINAL_STATUS
        or final.get("plan_sha256") != plan_logical
        or final.get("completed_groups") != 130
        or final.get("adaptation_groups") != 80
        or final.get("validation_groups") != 50
        or final.get("evaluation_groups") != 0
        or final.get("signed_gap_free_prefix_complete") is not True
        or final.get("test_inputs_read") is not False
        or final.get("fresh_inputs_accepted") is not False
        or final.get("confirmation_inputs_accepted") is not False
        or final.get("artifacts_frozen_read_only") is not True
        or plan.get("format") != COLLECTION_PLAN_FORMAT
        or plan.get("output_root") != str(root)
        or plan.get("command_count") != 130
        or plan.get("ordered_splits") != ["adaptation", "validation"]
        or plan.get("evaluation_commands_authorized") != 0
        or plan.get("test_inputs_read") is not False
        or exit_path.read_bytes() != b"0\n"
    ):
        raise PostCollectionError("collection terminal contract changed")
    prereg_path = existing_file(plan["preregistration_path"], "collection preregistration")
    if file_sha256(prereg_path) != plan.get("preregistration_file_sha256"):
        raise PostCollectionError("collection preregistration file SHA changed")
    preregistration = load_json(prereg_path, "collection preregistration")
    decoded = validate_preregistration(preregistration)
    if (
        decoded["preregistration_sha256"] != plan.get("preregistration_sha256")
        or decoded["preregistration_sha256"] != final.get("preregistration_sha256")
        or preregistration.get("outputs", {}).get("future_collection_root") != str(root)
    ):
        raise PostCollectionError("collection preregistration lineage changed")
    commands = decoded["commands"]
    seed_roots = [
        safe_path(command["outputs"]["seed_root"], f"collection seed root {index}")
        for index, command in enumerate(commands)
    ]
    if (
        len(seed_roots) != 130
        or len(set(seed_roots)) != 130
        or any(root not in item.parents for item in seed_roots)
    ):
        raise PostCollectionError("collection seed-root inventory changed")
    input_bindings = preregistration.get("input_bindings", {})
    target = input_bindings.get("target_seed_manifest", {})
    event = input_bindings.get("event_spec", {})
    collector = input_bindings.get("r6j_runtime_code", {})
    if not all(
        _is_sha(value)
        for value in (
            target.get("file_sha256"),
            target.get("logical_sha256"),
            event.get("sha256"),
            collector.get("closure_sha256"),
        )
    ):
        raise PostCollectionError("collection materialization bindings are incomplete")
    return {
        "root": str(root),
        "final_receipt_path": str(final_path),
        "final_receipt_file_sha256": file_sha256(final_path),
        "final_receipt_sha256": final_logical,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": plan["preregistration_file_sha256"],
        "preregistration_sha256": decoded["preregistration_sha256"],
        "target_seed_manifest": dict(target),
        "event_spec": dict(event),
        "collector_lineage_sha256": collector["closure_sha256"],
        "seed_roots": [str(path) for path in seed_roots],
        "hdf5_files_opened": 0,
        "labels_read": False,
    }


def wait_for_collection_terminal(
    collection_root: Path,
    *,
    interval: float,
    validate: Callable[[Path], dict[str, Any]] = validate_collection_terminal,
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Wait for terminal publication; failure receipt is never treated as data."""

    root = safe_path(collection_root, "collection root")
    while True:
        final = root / "_watcher" / "final_receipt.json"
        if final.exists() or final.is_symlink():
            return validate(root)
        if heartbeat is not None:
            heartbeat("waiting_for_authenticated_collection_terminal_receipt")
        sleep(interval)


def source_member_inventory(source_root: Path) -> dict[str, Any]:
    audit = validate_source_training_summary(source_root)
    root = existing_directory(source_root, "source training root", frozen=True)
    manifest_path = root / "counterfactual_training" / "ensemble_manifest.json"
    manifest = load_json(manifest_path, "source ensemble manifest")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != MEMBER_COUNT:
        raise PostCollectionError("source ensemble does not contain five members")
    inventory = []
    for index, (row, seed, expected_sha) in enumerate(
        zip(members, SOURCE_MEMBER_SEEDS, audit["member_checkpoint_sha256"], strict=True)
    ):
        if not isinstance(row, Mapping) or row.get("seed") != seed:
            raise PostCollectionError("source member order/seed changed")
        raw = Path(str(row.get("path", "")))
        checkpoint = raw if raw.is_absolute() else manifest_path.parent / raw
        checkpoint = existing_file(checkpoint, f"source member {index} checkpoint")
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != row.get("sha256") or checkpoint_sha != expected_sha:
            raise PostCollectionError("source member checkpoint SHA changed")
        inventory.append(
            {"member_index": index, "member_seed": seed, "path": str(checkpoint), "file_sha256": checkpoint_sha}
        )
    return {
        "audit": audit,
        "ensemble_manifest_path": str(manifest_path),
        "ensemble_manifest_file_sha256": file_sha256(manifest_path),
        "members": inventory,
    }


def wait_for_source_terminal(
    source_root: Path,
    *,
    interval: float,
    validate: Callable[[Path], dict[str, Any]] = source_member_inventory,
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root = safe_path(source_root, "source root")
    while True:
        if (root / "failure_receipt.json").exists():
            raise PostCollectionError("source training published a failure receipt")
        if (root / "final_receipt.json").exists():
            return validate(root)
        if heartbeat is not None:
            heartbeat("waiting_for_authenticated_source_terminal_receipt")
        sleep(interval)


def _load_bound_plan(root: Path) -> dict[str, Any]:
    plan = load_json(root / "_watcher" / "static_plan.json", "post-collection plan")
    verify_signed(plan, "plan_sha256", "post-collection plan")
    if plan.get("format") != PLAN_FORMAT or plan.get("output_root") != str(root):
        raise PostCollectionError("post-collection plan scope changed")
    code_bindings = plan.get("code_bindings")
    if not isinstance(code_bindings, Mapping):
        raise PostCollectionError("post-collection code bindings are missing")
    expected = {
        "watcher": Path(__file__).resolve(),
        "materializer": Path(materializer.__file__).resolve(),
        "trainer": Path(plan["trainer_path"]).resolve(),
        "calibrator": Path(calibration.__file__).resolve(),
        "target_validation_evaluator": Path(target_validation.__file__).resolve(),
    }
    for name, path in expected.items():
        binding = code_bindings.get(name)
        if (
            not isinstance(binding, Mapping)
            or Path(str(binding.get("path", ""))).resolve() != path
            or file_sha256(existing_file(path, f"bound {name}", allow_writable=True))
            != binding.get("file_sha256")
        ):
            raise PostCollectionError(f"bound {name} implementation changed")
    return plan


def preregister(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output = safe_path(args.output_root, "output root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise PostCollectionError("output parent is invalid")
    python = existing_file(args.python, "runtime Python", allow_writable=True)
    trainer = existing_file(args.trainer, "adapter trainer", allow_writable=True)
    materializer_path = existing_file(
        args.materializer, "manifest materializer", allow_writable=True
    )
    calibrator = existing_file(args.calibrator, "ensemble calibrator", allow_writable=True)
    target_evaluator = existing_file(
        args.target_validation_evaluator,
        "target-validation50 evaluator",
        allow_writable=True,
    )
    event_spec = existing_file(args.canonical_event_spec, "canonical event spec")
    bindings = (
        (python, args.python_sha256, "runtime Python"),
        (trainer, args.trainer_sha256, "adapter trainer"),
        (materializer_path, args.materializer_sha256, "manifest materializer"),
        (calibrator, args.calibrator_sha256, "ensemble calibrator"),
        (
            target_evaluator,
            args.target_validation_evaluator_sha256,
            "target-validation50 evaluator",
        ),
        (event_spec, args.canonical_event_spec_sha256, "canonical event spec"),
    )
    for path, expected_sha, role in bindings:
        if not _is_sha(expected_sha) or file_sha256(path) != expected_sha:
            raise PostCollectionError(f"{role} file SHA mismatch")
    teacher_binding = None
    if args.canonical_teacher_checkpoint is not None:
        teacher = existing_file(args.canonical_teacher_checkpoint, "canonical teacher")
        if (
            not _is_sha(args.canonical_teacher_checkpoint_sha256)
            or file_sha256(teacher) != args.canonical_teacher_checkpoint_sha256
        ):
            raise PostCollectionError("canonical teacher file SHA mismatch")
        teacher_binding = {"path": str(teacher), "file_sha256": file_sha256(teacher)}
    authorization_path = safe_path(
        args.paired_authorization_path, "independent paired400 authority"
    )
    if authorization_path.exists() or authorization_path.is_symlink():
        raise PostCollectionError(
            "independent paired400 authority must not pre-exist calibration"
        )
    authorization_path.parent.resolve(strict=True)
    output.mkdir(mode=0o755)
    watcher = output / "_watcher"
    watcher.mkdir()
    for name in ("manifest", "members", "calibration", "handoff"):
        (output / name).mkdir()
    code_bindings = {
        "watcher": {"path": str(Path(__file__).resolve()), "file_sha256": file_sha256(Path(__file__).resolve())},
        "materializer": {"path": str(materializer_path), "file_sha256": file_sha256(materializer_path)},
        "trainer": {"path": str(trainer), "file_sha256": file_sha256(trainer)},
        "calibrator": {"path": str(calibrator), "file_sha256": file_sha256(calibrator)},
        "target_validation_evaluator": {
            "path": str(target_evaluator),
            "file_sha256": file_sha256(target_evaluator),
        },
    }
    plan: dict[str, Any] = {
        "format": PLAN_FORMAT,
        "status": "preregistered_waiting_for_authenticated_upstream_terminals",
        "output_root": str(output),
        "collection_root": str(safe_path(args.collection_root, "collection root")),
        "source_root": str(safe_path(args.source_root, "source root")),
        "python_path": str(python),
        "python_file_sha256": file_sha256(python),
        "trainer_path": str(trainer),
        "canonical_event_spec": {"path": str(event_spec), "file_sha256": file_sha256(event_spec)},
        "canonical_teacher": teacher_binding,
        "code_bindings": code_bindings,
        "adapter_member_count": MEMBER_COUNT,
        "adapter_member_seeds": list(SOURCE_MEMBER_SEEDS),
        "adapter_steps": args.adapter_steps,
        "adapter_eval_every": args.adapter_eval_every,
        "gpu_index": args.gpu_index,
        "gpu_lock_path": str(safe_path(args.gpu_lock, "RTX4090 lock")),
        "independent_paired400_authority_path": str(authorization_path),
        "paired400_execution_by_this_watcher_authorized": False,
        "target_validation50_hdf5_open_authorized": False,
        "evaluation400_hdf5_or_label_open_authorized": False,
        "hdf5_files_opened_during_static_preflight": 0,
        "labels_read_during_static_preflight": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    immutable_json(watcher / "static_plan.json", plan)
    atomic_json(
        watcher / "state.json",
        {"format": STATE_FORMAT, "status": plan["status"], "plan_sha256": plan["plan_sha256"]},
    )
    return output, plan


def _state_writer(root: Path, plan: Mapping[str, Any]) -> Callable[[str], None]:
    def update(status: str) -> None:
        atomic_json(
            root / "_watcher" / "state.json",
            {
                "format": STATE_FORMAT,
                "status": status,
                "plan_sha256": plan["plan_sha256"],
                "heartbeat_unix": time.time(),
                "target_validation50_hdf5_files_opened_by_watcher_process": 0,
                "target_validation50_evaluator_progress": "see_frozen_evaluator_attempt_receipts",
                "evaluation400_hdf5_or_label_files_opened": 0,
            },
        )

    return update


def materialize_training_inputs(
    root: Path,
    plan: Mapping[str, Any],
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_root = root / "manifest"
    receipt_path = manifest_root / "schema6_training_manifest_v2_receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path, "training manifest receipt")
        verify_signed(receipt, "receipt_sha256", "training manifest receipt")
        if (
            receipt.get("status") != materializer.COMPLETE_STATUS
            or receipt.get("training_authorized") is not True
            or receipt.get("hdf5_content_files_opened") != 0
            or receipt.get("hdf5_labels_read") is not False
        ):
            raise PostCollectionError("existing training manifest receipt is not complete")
        return receipt
    target = collection["target_seed_manifest"]
    receipt = materializer.aggregate(
        target_seed_manifest_path=Path(str(target["path"])),
        expected_target_manifest_file_sha256=str(target["file_sha256"]),
        expected_target_manifest_logical_sha256=str(target["logical_sha256"]),
        event_spec_sha256=str(collection["event_spec"]["sha256"]),
        collector_lineage_sha256=str(collection["collector_lineage_sha256"]),
        bound_trainer_path=Path(plan["trainer_path"]),
        expected_bound_trainer_sha256=plan["code_bindings"]["trainer"]["file_sha256"],
        collection_roots=[Path(path) for path in collection["seed_roots"]],
        output_directory=manifest_root,
        collection_preregistration_path=Path(collection["preregistration_path"]),
        expected_collection_preregistration_file_sha256=collection[
            "preregistration_file_sha256"
        ],
        expected_collection_preregistration_logical_sha256=collection[
            "preregistration_sha256"
        ],
    )
    if receipt.get("status") != materializer.COMPLETE_STATUS:
        raise PostCollectionError("terminal collection did not materialize 130 inputs")
    return receipt


def _member_receipt_path(root: Path, index: int) -> Path:
    return root / "members" / f"member_{index}" / "final_receipt.json"


def validate_member_receipt(
    path: Path,
    *,
    member_index: int,
    member_seed: int,
    manifest_receipt: Mapping[str, Any],
    source_member: Mapping[str, Any],
) -> dict[str, Any]:
    value = load_json(path, f"adapter member {member_index} receipt")
    verify_signed(value, "receipt_sha256", f"adapter member {member_index} receipt")
    expected_fields = {
        "format", "status", "member_index", "member_seed",
        "source_checkpoint_sha256", "training_manifest_sha256", "split_sha256",
        "source_ensemble_contract_sha256", "summary_path", "summary_file_sha256",
        "summary_sha256", "checkpoint_path", "checkpoint_file_sha256",
        "validation_predictions_path", "validation_predictions_file_sha256",
        "validation_predictions_logical_sha256", "validation_labels_path",
        "validation_labels_file_sha256", "validation_labels_logical_sha256",
        "validation_identity_set_sha256", "validation_lane",
        "duration_target_transform", "next_event_observation_mask",
        "success_target", "recovery_target", "recovery_observation_mask",
        "recovery_shared_transition_stop_gradient",
        "recovery_enters_primary_before_calibration", "recovery_head_trained",
        "object_prediction_space",
        "object_source_normalization_sha256", "object_observed_policy",
        "target_validation50_hdf5_files_opened",
        "sealed_test_labels_opened", "receipt_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != MEMBER_RECEIPT_FORMAT
        or value.get("status") != "complete_frozen_internal_validation_predictions"
        or value.get("member_index") != member_index
        or value.get("member_seed") != member_seed
        or value.get("source_checkpoint_sha256") != source_member["file_sha256"]
        or value.get("training_manifest_sha256")
        != manifest_receipt["trainer_compatible_manifest"]["logical_sha256"]
        or value.get("split_sha256")
        != manifest_receipt["external_split"]["logical_sha256"]
        or value.get("target_validation50_hdf5_files_opened") != 0
        or value.get("sealed_test_labels_opened") != 0
        or value.get("validation_lane")
        != "adaptation_derived_internal_validation_only"
        or value.get("duration_target_transform") != "log1p_decision_steps"
        or value.get("next_event_observation_mask") != "duration_observed"
        or value.get("success_target")
        != "eventual_final_branch_success_repeated_per_transition"
        or value.get("recovery_target")
        != "conditional_recovery_given_operational_regress"
        or value.get("recovery_observation_mask")
        != "recovery_observed_and_regress"
        or value.get("recovery_shared_transition_stop_gradient") is not True
        or value.get("recovery_enters_primary_before_calibration") is not False
        or type(value.get("recovery_head_trained")) is not bool
        or value.get("object_prediction_space") != "physical_delta_xyz_m"
        or not _is_sha(value.get("object_source_normalization_sha256"))
        or value.get("object_observed_policy")
        != "row_enabled_only_if_all_selected_xyz_are_valid"
    ):
        raise PostCollectionError(f"adapter member {member_index} receipt scope changed")
    for role, path_key, sha_key in (
        ("summary", "summary_path", "summary_file_sha256"),
        ("checkpoint", "checkpoint_path", "checkpoint_file_sha256"),
        ("predictions", "validation_predictions_path", "validation_predictions_file_sha256"),
        ("labels", "validation_labels_path", "validation_labels_file_sha256"),
    ):
        artifact = existing_file(value[path_key], f"member {member_index} {role}")
        if file_sha256(artifact) != value[sha_key]:
            raise PostCollectionError(f"adapter member {member_index} {role} SHA changed")
    return value


def _next_attempt(member_root: Path) -> tuple[int, Path, Path]:
    attempts = member_root / "attempts"
    attempts.mkdir(exist_ok=True)
    indices = []
    for path in attempts.glob("attempt_*"):
        suffix = path.name.removeprefix("attempt_")
        if path.is_dir() and suffix.isdecimal():
            indices.append(int(suffix))
    index = max(indices, default=-1) + 1
    stage = attempts / f"attempt_{index:03d}"
    stage.mkdir()
    return index, stage, stage / "adapter"


def run_member_stage(
    *,
    root: Path,
    plan: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
    source: Mapping[str, Any],
    member_index: int,
    poll_interval: float,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    audit_gpu: Callable[[int], dict[str, Any]] = gpu_audit,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    member_seed = int(SOURCE_MEMBER_SEEDS[member_index])
    source_member = source["members"][member_index]
    receipt_path = _member_receipt_path(root, member_index)
    if receipt_path.exists():
        return validate_member_receipt(
            receipt_path,
            member_index=member_index,
            member_seed=member_seed,
            manifest_receipt=manifest_receipt,
            source_member=source_member,
        )
    member_root = receipt_path.parent
    member_root.mkdir(exist_ok=True)
    attempt_index, stage, adapter_output = _next_attempt(member_root)
    command = [
        plan["python_path"],
        plan["trainer_path"],
        "--mode", "train",
        "--source-checkpoint", source_member["path"],
        "--schema6-manifest", manifest_receipt["trainer_compatible_manifest"]["path"],
        "--expected-manifest-split-receipt",
        manifest_receipt["expected_manifest_split_receipt"]["path"],
        "--expected-manifest-split-receipt-file-sha256",
        manifest_receipt["expected_manifest_split_receipt"]["file_sha256"],
        "--canonical-event-spec", plan["canonical_event_spec"]["path"],
        "--output", str(adapter_output),
        "--device", f"cuda:{plan['gpu_index']}",
        "--steps", str(plan["adapter_steps"]),
        "--eval-every", str(plan["adapter_eval_every"]),
        "--training-seed", str(member_seed),
    ]
    if plan.get("canonical_teacher") is not None:
        command.extend(
            ["--canonical-teacher-checkpoint", plan["canonical_teacher"]["path"]]
        )
    launch = {
        "format": MEMBER_RECEIPT_FORMAT,
        "status": "launching_bound_adapter_member",
        "member_index": member_index,
        "member_seed": member_seed,
        "attempt_index": attempt_index,
        "command": command,
        "trainer_file_sha256": plan["code_bindings"]["trainer"]["file_sha256"],
        "source_checkpoint_sha256": source_member["file_sha256"],
        "target_validation50_hdf5_open_authorized": False,
        "evaluation400_hdf5_or_label_open_authorized": False,
    }
    launch["launch_sha256"] = canonical_sha256(launch)
    immutable_json(stage / "launch.json", launch)
    log_path = stage / "run.log"
    process: subprocess.Popen[bytes] | None = None
    foreign: list[int] = []
    with log_path.open("xb") as log:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
        while process.poll() is None:
            current = audit_gpu(plan["gpu_index"])
            foreign = [
                pid
                for pid in current["compute_pids"]
                if not _is_descendant(pid, process.pid)
            ]
            if foreign:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                break
            sleep(poll_interval)
        returncode = process.wait()
    immutable_text(stage / "run.exit", f"{returncode}\n")
    if returncode != 0 or foreign or not adapter_output.is_dir():
        raise PostCollectionError(
            f"adapter member {member_index} attempt failed; resumable attempt retained"
        )
    summary_path = adapter_output / "training_summary.json"
    summary = load_json(summary_path, f"adapter member {member_index} summary", frozen=False)
    summary_logical = verify_signed(
        summary, "summary_sha256", f"adapter member {member_index} summary"
    )
    artifacts = summary.get("validation_artifacts")
    if (
        summary.get("status") != "complete"
        or summary.get("source_checkpoint_sha256") != source_member["file_sha256"]
        or summary.get("schema6_training_manifest_sha256")
        != manifest_receipt["trainer_compatible_manifest"]["logical_sha256"]
        or summary.get("external_split_sha256")
        != manifest_receipt["external_split"]["logical_sha256"]
        or summary.get("sealed_test_groups") != 50
        or summary.get("test_hdf5_files_opened") != 0
        or not isinstance(artifacts, Mapping)
        or artifacts.get("lane") != "adaptation_derived_internal_validation_only"
        or artifacts.get("validation_group_count") != 20
        or artifacts.get("target_validation50_hdf5_files_opened") != 0
        or artifacts.get("sealed_test_labels_opened") != 0
    ):
        raise PostCollectionError(f"adapter member {member_index} summary is incomplete")
    checkpoint = existing_file(summary["best_checkpoint"], "best adapter checkpoint", allow_writable=True)
    predictions = existing_file(artifacts["predictions_path"], "internal validation predictions", allow_writable=True)
    labels = existing_file(artifacts["labels_path"], "internal validation labels", allow_writable=True)
    for path, expected, role in (
        (checkpoint, summary["best_checkpoint_sha256"], "checkpoint"),
        (predictions, artifacts["predictions_file_sha256"], "predictions"),
        (labels, artifacts["labels_file_sha256"], "labels"),
    ):
        if file_sha256(path) != expected:
            raise PostCollectionError(f"adapter member {member_index} {role} SHA changed")
    freeze_tree(adapter_output)
    receipt: dict[str, Any] = {
        "format": MEMBER_RECEIPT_FORMAT,
        "status": "complete_frozen_internal_validation_predictions",
        "member_index": member_index,
        "member_seed": member_seed,
        "source_checkpoint_sha256": source_member["file_sha256"],
        "training_manifest_sha256": manifest_receipt["trainer_compatible_manifest"]["logical_sha256"],
        "split_sha256": manifest_receipt["external_split"]["logical_sha256"],
        "source_ensemble_contract_sha256": source["ensemble_manifest_file_sha256"],
        "summary_path": str(summary_path),
        "summary_file_sha256": file_sha256(summary_path),
        "summary_sha256": summary_logical,
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": file_sha256(checkpoint),
        "validation_predictions_path": str(predictions),
        "validation_predictions_file_sha256": file_sha256(predictions),
        "validation_predictions_logical_sha256": artifacts["predictions_logical_sha256"],
        "validation_labels_path": str(labels),
        "validation_labels_file_sha256": file_sha256(labels),
        "validation_labels_logical_sha256": artifacts["labels_logical_sha256"],
        "validation_identity_set_sha256": artifacts["validation_identity_set_sha256"],
        "validation_lane": artifacts["lane"],
        "duration_target_transform": artifacts["duration_target_transform"],
        "next_event_observation_mask": artifacts["next_event_observation_mask"],
        "success_target": artifacts["success_target"],
        "recovery_target": artifacts["recovery_target"],
        "recovery_observation_mask": artifacts["recovery_observation_mask"],
        "recovery_shared_transition_stop_gradient": artifacts[
            "recovery_shared_transition_stop_gradient"
        ],
        "recovery_enters_primary_before_calibration": artifacts[
            "recovery_enters_primary_utility_or_uncertainty"
        ],
        "recovery_head_trained": artifacts["recovery_head_trained"],
        "object_prediction_space": artifacts["object_prediction_space"],
        "object_source_normalization_sha256": artifacts[
            "object_source_normalization_sha256"
        ],
        "object_observed_policy": artifacts["object_observed_policy"],
        "target_validation50_hdf5_files_opened": 0,
        "sealed_test_labels_opened": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    immutable_json(receipt_path, receipt)
    return receipt


def _reject_internal20_formal_calibration_authority(
    root: Path,
    members: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    raise PostCollectionError(
        "internal-validation20 cannot satisfy the formal 50-group abstention gate; "
        "use the independent target-validation50 evaluator"
    )
    # Kept below only as an unreachable migration reference for old receipts.
    authority_path = root / "calibration" / "validation_input_authority.json"
    label_logical = {row["validation_labels_logical_sha256"] for row in members}
    identities = {row["validation_identity_set_sha256"] for row in members}
    contracts = {
        (
            row["training_manifest_sha256"],
            row["split_sha256"],
            row["source_ensemble_contract_sha256"],
        )
        for row in members
    }
    prediction_contracts = {
        canonical_sha256(
            {
                "duration_target_transform": row["duration_target_transform"],
                "next_event_observation_mask": row["next_event_observation_mask"],
                "success_target": row["success_target"],
                "recovery_target": row["recovery_target"],
                "recovery_observation_mask": row["recovery_observation_mask"],
                "recovery_shared_transition_stop_gradient": row[
                    "recovery_shared_transition_stop_gradient"
                ],
                "recovery_enters_primary_before_calibration": row[
                    "recovery_enters_primary_before_calibration"
                ],
                "recovery_head_trained": row["recovery_head_trained"],
                "object_prediction_space": row["object_prediction_space"],
                "object_source_normalization_sha256": row[
                    "object_source_normalization_sha256"
                ],
                "object_observed_policy": row["object_observed_policy"],
            }
        )
        for row in members
    }
    if (
        len(members) != MEMBER_COUNT
        or len(label_logical) != 1
        or len(identities) != 1
        or len(contracts) != 1
        or len(prediction_contracts) != 1
    ):
        raise PostCollectionError("five adapter members do not share validation/contracts")
    if authority_path.exists():
        authority = load_json(authority_path, "calibration input authority")
        verify_signed(authority, "input_authority_sha256", "calibration input authority")
        return authority_path, authority
    first = members[0]
    shared_values = next(iter(contracts))
    prediction_contract = {
        "duration_target_transform": first["duration_target_transform"],
        "next_event_observation_mask": first["next_event_observation_mask"],
        "success_target": first["success_target"],
        "recovery_target": first["recovery_target"],
        "recovery_observation_mask": first["recovery_observation_mask"],
        "recovery_shared_transition_stop_gradient": first[
            "recovery_shared_transition_stop_gradient"
        ],
        "recovery_enters_primary_before_calibration": first[
            "recovery_enters_primary_before_calibration"
        ],
        "recovery_head_trained": first["recovery_head_trained"],
        "object_prediction_space": first["object_prediction_space"],
        "object_source_normalization_sha256": first[
            "object_source_normalization_sha256"
        ],
        "object_observed_policy": first["object_observed_policy"],
    }
    shared = {
        "training_manifest_sha256": shared_values[0],
        "split_sha256": shared_values[1],
        "source_ensemble_contract_sha256": shared_values[2],
        "prediction_contract_sha256": canonical_sha256(prediction_contract),
    }
    authority: dict[str, Any] = {
        "format": CALIBRATION_INPUT_FORMAT,
        "status": calibration.INPUT_STATUS,
        "lane": "validation_only",
        "member_count": MEMBER_COUNT,
        "shared_contract": shared,
        "prediction_contract": prediction_contract,
        "validation_identity_set_sha256": first["validation_identity_set_sha256"],
        "labels_path": first["validation_labels_path"],
        "labels_file_sha256": first["validation_labels_file_sha256"],
        "members": [
            {
                "member_index": row["member_index"],
                "member_seed": row["member_seed"],
                **shared,
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_file_sha256": row["checkpoint_file_sha256"],
                "validation_predictions_path": row["validation_predictions_path"],
                "validation_predictions_file_sha256": row[
                    "validation_predictions_file_sha256"
                ],
            }
            for row in members
        ],
        "test_artifacts_read": False,
        "fresh_artifacts_read": False,
        "confirmation_artifacts_read": False,
    }
    authority["input_authority_sha256"] = canonical_sha256(authority)
    immutable_json(authority_path, authority)
    return authority_path, authority


def materialize_target_validation50_evaluator_authority(
    root: Path,
    plan: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Authorize the independent evaluator only after all five adapters freeze."""

    path = root / "calibration" / "target_validation50_evaluator_input.json"
    if len(members) != MEMBER_COUNT:
        raise PostCollectionError("target-validation50 evaluator requires five members")
    if path.exists():
        value = load_json(path, "target-validation50 evaluator authority")
        verify_signed(value, "authority_sha256", "target-validation50 evaluator authority")
        return path, value
    authority_members = []
    for index, (member, source_member) in enumerate(
        zip(members, source["members"], strict=True)
    ):
        member_receipt = _member_receipt_path(root, index)
        prediction_contract = {
            "duration_target_transform": member["duration_target_transform"],
            "next_event_observation_mask": member["next_event_observation_mask"],
            "success_target": member["success_target"],
            "recovery_target": member["recovery_target"],
            "recovery_observation_mask": member["recovery_observation_mask"],
            "recovery_shared_transition_stop_gradient": member[
                "recovery_shared_transition_stop_gradient"
            ],
            "recovery_enters_primary_before_calibration": member[
                "recovery_enters_primary_before_calibration"
            ],
            "recovery_head_trained": member["recovery_head_trained"],
            "object_prediction_space": member["object_prediction_space"],
            "object_source_normalization_sha256": member[
                "object_source_normalization_sha256"
            ],
            "object_observed_policy": member["object_observed_policy"],
        }
        authority_members.append(
            {
                "member_index": index,
                "member_seed": member["member_seed"],
                "adapter_checkpoint": {
                    "path": member["checkpoint_path"],
                    "file_sha256": member["checkpoint_file_sha256"],
                },
                "source_checkpoint": {
                    "path": source_member["path"],
                    "file_sha256": source_member["file_sha256"],
                },
                "member_receipt": {
                    "path": str(member_receipt),
                    "file_sha256": file_sha256(member_receipt),
                    "logical_sha256": member["receipt_sha256"],
                },
                "training_manifest_sha256": member["training_manifest_sha256"],
                "split_sha256": member["split_sha256"],
                "source_ensemble_contract_sha256": member[
                    "source_ensemble_contract_sha256"
                ],
                "prediction_contract": prediction_contract,
            }
        )
        if not member_receipt.is_file() or member_receipt.is_symlink():
            raise PostCollectionError("frozen adapter member receipt disappeared")
    authority: dict[str, Any] = {
        "format": target_validation.INPUT_FORMAT,
        "status": target_validation.INPUT_STATUS,
        "trainer_compatible_manifest": dict(
            manifest_receipt["trainer_compatible_manifest"]
        ),
        "expected_manifest_split_receipt": dict(
            manifest_receipt["expected_manifest_split_receipt"]
        ),
        "canonical_event_spec": dict(plan["canonical_event_spec"]),
        "members": authority_members,
        "member_count": MEMBER_COUNT,
        "target_validation_group_count": 50,
        "adapter_training_complete_before_authority": True,
        "target_validation_open_authorized": True,
        "evaluation400_membership_present": False,
        "evaluation400_open_authorized": False,
        "fresh_or_confirmation_open_authorized": False,
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    immutable_json(path, authority)
    return path, authority


def _validated_target_validation50_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json(path, "target-validation50 evaluator receipt")
    verify_signed(receipt, "receipt_sha256", "target-validation50 evaluator receipt")
    if (
        receipt.get("format") != target_validation.RECEIPT_FORMAT
        or receipt.get("status") != target_validation.RECEIPT_STATUS
        or receipt.get("target_validation_groups") != 50
        or receipt.get("target_validation_hdf5_files_opened") != 50
        or receipt.get("target_validation_opened_after_five_adapters_frozen") is not True
        or receipt.get("evaluation400_membership_present") is not False
        or receipt.get("evaluation400_hdf5_or_label_files_opened") != 0
        or receipt.get("fresh_or_confirmation_files_opened") != 0
        or receipt.get("performance_or_transfer_claim_authorized") is not False
    ):
        raise PostCollectionError("target-validation50 evaluator receipt is incomplete")
    authority_path = existing_file(
        receipt["calibration_input_authority_path"],
        "target-validation50 calibration input authority",
    )
    if file_sha256(authority_path) != receipt["calibration_input_authority_file_sha256"]:
        raise PostCollectionError("target-validation50 calibration authority SHA changed")
    return receipt


def run_target_validation50_stage(
    *,
    root: Path,
    plan: Mapping[str, Any],
    evaluator_authority_path: Path,
    poll_interval: float,
) -> dict[str, Any]:
    attempts = root / "calibration" / "target_validation50_attempts"
    attempts.mkdir(exist_ok=True)
    completed = sorted(attempts.glob("attempt_*/result/final_receipt.json"))
    if len(completed) > 1:
        raise PostCollectionError("multiple target-validation50 evaluator completions exist")
    if completed:
        return _validated_target_validation50_receipt(completed[0])
    indices = [
        int(path.name.removeprefix("attempt_"))
        for path in attempts.glob("attempt_*")
        if path.is_dir() and path.name.removeprefix("attempt_").isdecimal()
    ]
    index = max(indices, default=-1) + 1
    stage = attempts / f"attempt_{index:03d}"
    stage.mkdir()
    result = stage / "result"
    command = [
        plan["python_path"],
        plan["code_bindings"]["target_validation_evaluator"]["path"],
        "--input-authority", str(evaluator_authority_path),
        "--input-authority-file-sha256", file_sha256(evaluator_authority_path),
        "--output-root", str(result),
        "--device", f"cuda:{plan['gpu_index']}",
    ]
    launch: dict[str, Any] = {
        "status": "launching_independent_target_validation50_evaluator",
        "attempt_index": index,
        "command": command,
        "evaluator_file_sha256": plan["code_bindings"][
            "target_validation_evaluator"
        ]["file_sha256"],
        "target_validation50_open_authorized": True,
        "evaluation400_open_authorized": False,
    }
    launch["launch_sha256"] = canonical_sha256(launch)
    immutable_json(stage / "launch.json", launch)
    log_path = stage / "run.log"
    foreign: list[int] = []
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )
        while process.poll() is None:
            audit = gpu_audit(plan["gpu_index"])
            foreign = [
                pid
                for pid in audit["compute_pids"]
                if not _is_descendant(pid, process.pid)
            ]
            if foreign:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                break
            time.sleep(poll_interval)
        returncode = process.wait()
    immutable_text(stage / "run.exit", f"{returncode}\n")
    if returncode != 0 or foreign or not (result / "final_receipt.json").exists():
        raise PostCollectionError(
            "target-validation50 evaluator failed; retained attempt is resumable"
        )
    return _validated_target_validation50_receipt(result / "final_receipt.json")


def run_calibration(root: Path, authority_path: Path) -> dict[str, Any]:
    output = root / "calibration" / "result"
    final_path = output / "final_receipt.json"
    if final_path.exists():
        receipt = load_json(final_path, "ensemble calibration receipt")
        verify_signed(receipt, "receipt_sha256", "ensemble calibration receipt")
    else:
        receipt = calibration.run(
            authority_path, file_sha256(authority_path), output
        )
    if (
        receipt.get("status") != calibration.RECEIPT_STATUS
        or receipt.get("member_count") != MEMBER_COUNT
        or receipt.get("validation_only") is not True
        or receipt.get("test_hdf5_files_opened") != 0
        or receipt.get("performance_or_transfer_claim_authorized") is not False
        or receipt.get("abstain_threshold_enabled") is not True
    ):
        raise PostCollectionError("ensemble calibration receipt is incomplete")
    ensemble_manifest = load_json(
        Path(receipt["ensemble_manifest_path"]), "calibrated ensemble manifest"
    )
    enabled = ensemble_manifest.get("head_enabled_for_primary")
    if (
        not isinstance(enabled, Mapping)
        or any(enabled.get(name) is not True for name in ("post_event", "next_event", "duration"))
        or ensemble_manifest.get("abstain_threshold_enabled") is not True
    ):
        raise PostCollectionError(
            "core event/duration heads or uncertainty abstention did not pass target-validation50"
        )
    return receipt


def publish_authorization_request(
    root: Path, calibration_receipt: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = root / "handoff" / "paired400_authorization_request.json"
    if path.exists():
        value = load_json(path, "paired400 authorization request")
        verify_signed(value, "request_sha256", "paired400 authorization request")
        return path, value
    calibration_final = root / "calibration" / "result" / "final_receipt.json"
    request: dict[str, Any] = {
        "format": AUTHORIZATION_REQUEST_FORMAT,
        "status": WAITING_STATUS,
        "requested_pair_count": 400,
        "calibration_receipt_path": str(calibration_final),
        "calibration_receipt_file_sha256": file_sha256(calibration_final),
        "calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
        "independent_authority_required": True,
        "post_collection_orchestrator_may_execute_pairs": False,
        "sealed_identity_payload_requested": False,
        "sealed_outcome_or_label_payload_requested": False,
        "target_validation50_hdf5_files_opened_by_evaluator": 50,
        "target_validation50_hdf5_files_opened_by_watcher_process": 0,
        "evaluation400_hdf5_or_label_files_opened": 0,
        "performance_or_transfer_claim_authorized": False,
    }
    request["request_sha256"] = canonical_sha256(request)
    immutable_json(path, request)
    return path, request


def validate_independent_authority(
    path: Path,
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    authority = load_json(path, "independent paired400 authority")
    logical = verify_signed(
        authority, "authority_sha256", "independent paired400 authority"
    )
    expected_fields = {
        "format", "status", "request_sha256", "calibration_receipt_sha256",
        "authorized_pair_count", "execution_authorized",
        "external_paired_launcher_required",
        "post_collection_orchestrator_may_execute_pairs",
        "sealed_identity_payload_disclosed", "sealed_outcome_or_label_payload_disclosed",
        "target_validation50_hdf5_open_authorized",
        "evaluation400_hdf5_or_label_open_authorized", "authority_sha256",
    }
    if (
        set(authority) != expected_fields
        or authority.get("format") != INDEPENDENT_AUTHORITY_FORMAT
        or authority.get("status")
        != "authorized_for_external_paired400_launcher_only"
        or authority.get("request_sha256") != request["request_sha256"]
        or authority.get("calibration_receipt_sha256")
        != request["calibration_receipt_sha256"]
        or authority.get("authorized_pair_count") != 400
        or authority.get("execution_authorized") is not True
        or authority.get("external_paired_launcher_required") is not True
        or authority.get("post_collection_orchestrator_may_execute_pairs") is not False
        or authority.get("sealed_identity_payload_disclosed") is not False
        or authority.get("sealed_outcome_or_label_payload_disclosed") is not False
        or authority.get("target_validation50_hdf5_open_authorized") is not False
        or authority.get("evaluation400_hdf5_or_label_open_authorized") is not False
    ):
        raise PostCollectionError("independent paired400 authority scope changed")
    return {**authority, "verified_authority_sha256": logical}


def wait_for_independent_authority(
    authority_path: Path,
    *,
    request: Mapping[str, Any],
    interval: float,
    validate: Callable[..., dict[str, Any]] = validate_independent_authority,
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    path = safe_path(authority_path, "independent paired400 authority")
    while True:
        if path.exists() or path.is_symlink():
            return validate(path, request=request)
        if heartbeat is not None:
            heartbeat(WAITING_STATUS)
        sleep(interval)


def serve(root: Path, *, poll_interval: float, idle_interval: float) -> dict[str, Any]:
    wait_for_ppid1()
    plan = _load_bound_plan(root)
    update_state = _state_writer(root, plan)
    collection = wait_for_collection_terminal(
        Path(plan["collection_root"]), interval=poll_interval, heartbeat=update_state
    )
    source = wait_for_source_terminal(
        Path(plan["source_root"]), interval=poll_interval, heartbeat=update_state
    )
    update_state("materializing_authenticated_label_blind_training_manifest")
    manifest_receipt = materialize_training_inputs(root, plan, collection)
    members: list[dict[str, Any]] = []
    target_validation_receipt: dict[str, Any]
    lock_path = safe_path(plan["gpu_lock_path"], "RTX4090 lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="ascii") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        wait_two_idle(plan["gpu_index"], interval=idle_interval)
        for index in range(MEMBER_COUNT):
            update_state(f"training_adapter_member_{index}_of_5")
            members.append(
                run_member_stage(
                    root=root,
                    plan=plan,
                    manifest_receipt=manifest_receipt,
                    source=source,
                    member_index=index,
                    poll_interval=poll_interval,
                )
            )
        update_state("evaluating_frozen_ensemble_on_independent_target_validation50")
        evaluator_authority_path, _evaluator_authority = (
            materialize_target_validation50_evaluator_authority(
                root, plan, manifest_receipt, members, source
            )
        )
        target_validation_receipt = run_target_validation50_stage(
            root=root,
            plan=plan,
            evaluator_authority_path=evaluator_authority_path,
            poll_interval=poll_interval,
        )
        wait_two_idle(plan["gpu_index"], interval=idle_interval)
    update_state("calibrating_five_member_target_validation50_ensemble")
    calibration_authority_path = Path(
        target_validation_receipt["calibration_input_authority_path"]
    )
    calibration_receipt = run_calibration(root, calibration_authority_path)
    request_path, request = publish_authorization_request(root, calibration_receipt)
    update_state(WAITING_STATUS)
    independent = wait_for_independent_authority(
        Path(plan["independent_paired400_authority_path"]),
        request=request,
        interval=poll_interval,
        heartbeat=update_state,
    )
    handoff: dict[str, Any] = {
        "format": HANDOFF_FORMAT,
        "status": TERMINAL_STATUS,
        "plan_sha256": plan["plan_sha256"],
        "calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
        "authorization_request_path": str(request_path),
        "authorization_request_sha256": request["request_sha256"],
        "independent_authority_path": plan["independent_paired400_authority_path"],
        "independent_authority_sha256": independent["verified_authority_sha256"],
        "authorized_pair_count": 400,
        "external_paired_launcher_required": True,
        "paired_pairs_executed_by_this_watcher": 0,
        "target_validation50_hdf5_files_opened_by_evaluator": 50,
        "target_validation50_hdf5_files_opened_by_watcher_process": 0,
        "evaluation400_hdf5_or_label_files_opened": 0,
        "performance_or_transfer_claim_authorized": False,
    }
    handoff["receipt_sha256"] = canonical_sha256(handoff)
    immutable_json(root / "handoff" / "final_receipt.json", handoff)
    immutable_text(root / "handoff" / "run.exit", "0\n")
    update_state(TERMINAL_STATUS)
    freeze_tree(root)
    return handoff


def _detach(
    root: Path,
    plan: Mapping[str, Any],
    *,
    poll_interval: float,
    idle_interval: float,
    resume: bool,
) -> dict[str, Any]:
    command = [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--mode", "serve-existing",
        "--output-root", str(root),
        "--poll-interval", str(poll_interval),
        "--idle-interval", str(idle_interval),
    ]
    log_name = "watcher_resume.log" if resume else "watcher.log"
    mode = "ab" if resume else "xb"
    with (root / "_watcher" / log_name).open(mode) as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    receipt: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detached_new_session_ppid1_required",
        "pid": process.pid,
        "plan_sha256": plan["plan_sha256"],
        "resume": resume,
        "command": command,
    }
    receipt["detach_receipt_sha256"] = canonical_sha256(receipt)
    directory = root / "_watcher" / "detach_receipts"
    directory.mkdir(exist_ok=True)
    immutable_json(directory / f"detach_{time.time_ns()}.json", receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "detach", "detach-resume", "serve-existing"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--python-sha256")
    parser.add_argument("--materializer", type=Path, default=Path(materializer.__file__).resolve())
    parser.add_argument("--materializer-sha256")
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--trainer-sha256")
    parser.add_argument("--calibrator", type=Path, default=Path(calibration.__file__).resolve())
    parser.add_argument("--calibrator-sha256")
    parser.add_argument(
        "--target-validation-evaluator",
        type=Path,
        default=Path(target_validation.__file__).resolve(),
    )
    parser.add_argument("--target-validation-evaluator-sha256")
    parser.add_argument("--canonical-event-spec", type=Path)
    parser.add_argument("--canonical-event-spec-sha256")
    parser.add_argument("--canonical-teacher-checkpoint", type=Path)
    parser.add_argument("--canonical-teacher-checkpoint-sha256")
    parser.add_argument("--paired-authorization-path", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--adapter-steps", type=int, default=ADAPTER_STEPS)
    parser.add_argument("--adapter-eval-every", type=int, default=ADAPTER_EVAL_EVERY)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--idle-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode in {"preflight", "detach"}:
        required = (
            args.collection_root,
            args.source_root,
            args.python,
            args.python_sha256,
            args.materializer_sha256,
            args.trainer,
            args.trainer_sha256,
            args.calibrator_sha256,
            args.target_validation_evaluator_sha256,
            args.canonical_event_spec,
            args.canonical_event_spec_sha256,
            args.paired_authorization_path,
            args.gpu_lock,
        )
        if any(value is None for value in required):
            raise PostCollectionError("new watcher preregistration arguments are incomplete")
        if (
            args.gpu_index < 0
            or args.adapter_steps < 1
            or args.adapter_eval_every < 1
            or args.poll_interval <= 0
            or args.idle_interval <= 0
        ):
            raise PostCollectionError("numeric watcher arguments are invalid")
        root, plan = preregister(args)
        if args.mode == "preflight":
            print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"]}, sort_keys=True))
            return 0
        print(json.dumps(_detach(root, plan, poll_interval=args.poll_interval, idle_interval=args.idle_interval, resume=False), sort_keys=True))
        return 0
    root = existing_directory(args.output_root, "post-collection output root")
    plan = _load_bound_plan(root)
    if args.mode == "detach-resume":
        if (root / "handoff" / "final_receipt.json").exists():
            raise PostCollectionError("terminal post-collection pipeline cannot resume")
        print(json.dumps(_detach(root, plan, poll_interval=args.poll_interval, idle_interval=args.idle_interval, resume=True), sort_keys=True))
        return 0
    try:
        serve(root, poll_interval=args.poll_interval, idle_interval=args.idle_interval)
    except Exception as error:
        failure: dict[str, Any] = {
            "format": FORMAT,
            "status": FAILURE_STATUS,
            "error_type": type(error).__name__,
            "error": str(error),
            "resumable": True,
            "paired_pairs_executed_by_this_watcher": 0,
            "target_validation50_hdf5_files_opened_by_watcher_process": 0,
            "target_validation50_evaluator_progress": "see_evaluator_attempt_receipts",
            "evaluation400_hdf5_or_label_files_opened": 0,
        }
        failure["receipt_sha256"] = canonical_sha256(failure)
        directory = root / "_watcher" / "failures"
        directory.mkdir(exist_ok=True)
        immutable_json(directory / f"failure_{time.time_ns()}.json", failure)
        atomic_json(
            root / "_watcher" / "state.json",
            {"format": STATE_FORMAT, "status": FAILURE_STATUS, "receipt_sha256": failure["receipt_sha256"], "resumable": True},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PostCollectionError", "gpu_audit",
    "publish_authorization_request", "validate_collection_terminal",
    "validate_independent_authority", "wait_for_collection_terminal",
    "wait_for_independent_authority", "wait_for_ppid1", "wait_two_idle",
]
