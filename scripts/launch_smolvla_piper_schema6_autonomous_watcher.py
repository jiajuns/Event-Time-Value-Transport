#!/usr/bin/env python3
"""Fail-closed detached watcher for the Piper schema-v6 one-seed pipeline.

The watcher is intentionally a narrow server-side state machine:

1. wait for the autonomous Piper-then-UR5 LOBO watcher to finish both stages;
   that aggregate receipt itself authenticates the currently selected source63
   terminal run, so this watcher never binds a stale source root directly;
2. wait until the designated RTX 4090 has no compute applications;
3. run the reset-only can/pot identity materializer;
4. freeze a content-addressed one-seed, H=1 collection authority;
5. wait for the 4090 to be idle again and execute that authority.

It never opens an HDF5 source/test file.  It reads only the frozen LOBO
aggregate, its stage logs/JSON summaries, and the transitive source audit
embedded in that aggregate receipt.  Every
accepted filesystem path is rejected lexically and after resolution when any
component contains either forbidden sensitive token.  ``detach`` starts a new
OS session so the watcher survives an SSH client or local-computer shutdown.
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


FORMAT = "etsf_smolvla_piper_schema6_autonomous_watcher_v2"
STATE_FORMAT = "etsf_smolvla_piper_schema6_autonomous_state_v2"
DETACH_FORMAT = "etsf_smolvla_piper_schema6_autonomous_detach_v2"
SOURCE_FORMAT = "etsf_smolvla_schema5_source63_native_training_launcher_v1"
SOURCE_TERMINAL_STATUS = (
    "complete_source63_native_counterfactual_training_fresh_forbidden"
)
SOURCE_TRAINING_STATUS = "complete_verified_source63_counterfactual_training"
SOURCE_MEMBER_SEEDS = (20260828, 20260829, 20260830, 20260831, 20260832)
SOURCE_TRAINING_STEPS = 3000
LOBO_WATCHER_FORMAT = "etsf_multibody_lobo_autonomous_watcher_v1"
LOBO_WATCHER_TERMINAL_STATUS = "complete_sequential_piper_then_ur5_lobo_training"
LOBO_WATCHER_FROZEN_STATE_STATUS = "terminal_success_pending_frozen_publication"
LOBO_OUTPUT_FORMAT = "etsf_multibody_leave_one_body_out_v1"
LOBO_OUTPUT_TERMINAL_STATUS = (
    "training_and_frozen_target_development_evaluation_complete"
)
LOBO_SOURCE_BINDING_FORMAT = "etsf_multibody_lobo_source_binding_receipt_v1"
LOBO_SOURCE_BINDING_STATUS = "bound_native_source_ensemble_for_deployment_rerank"
LOBO_STAGE_SOURCE_BINDING_FORMAT = "etsf_multibody_lobo_stage_source_binding_v1"
EXPECTED_LOBO_LAUNCHER_SHA256 = (
    "3af8933fa5ccd09e7b06dc1912926510e5a9fb0508b2aee3c9d323adafb71206"
)
EXPECTED_LOBO_STATIC_PLAN_SHA256 = (
    "467091737465220a1733aca1acb91b9f941cd788aa9368fcbb4b5c6ea7859986"
)
EXPECTED_SOURCE_ROOT = Path(
    "/home/user/etsf_smolvla_schema5_native_source_training_r12_20260828"
)
EXPECTED_SOURCE_ENSEMBLE_RELATIVE = Path(
    "counterfactual_training/counterfactual_ensemble.pt"
)
EXPECTED_SOURCE_PLAN_SHA256 = (
    "78835ab43d36e783335a6c7b6bb322147cf41e009ddb2870d9cad4d027493ac6"
)
EXPECTED_SOURCE_STATIC_PLAN_SHA256 = (
    "10ed8ceb1eb2d5374225df247fe078b220414d4994f5d970af8a0c552fa4aac4"
)
EXPECTED_SOURCE_LAUNCHER_SHA256 = (
    "1713fe07a0416ea692cde171061bd739016f4832dc76b0eff7c43904b1c68d57"
)
EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256 = (
    "8e03189d82950a4f04bdf3bb0de76f21ad85c5b5c9ce1dc660e49e07c6e3efcf"
)
LOBO_STAGES = (
    ("train_lobo_piper", "piper"),
    ("train_lobo_ur5", "ur5-wsg"),
)
AUTHORITY_FORMAT = "smolvla_piper_schema6_development_collection_authority_v1"
AUTHORITY_STATUS = "frozen_development_collection_not_started"
COLLECTION_RECEIPT_FORMAT = (
    "smolvla_piper_schema6_development_collection_receipt_v1"
)
COLLECTION_MANIFEST_FORMAT = (
    "smolvla_piper_schema6_development_collection_manifest_v1"
)
COLLECTION_SUCCESS_STATUS = "completed_one_seed_schema6_development_collection"
COLLECTION_EMPTY_STATUS = "completed_root_fewer_than_two_legal_no_group"
TERMINAL_STATUS = "complete_one_seed_schema6_collection_attempt"
FAILURE_STATUS = "failed_closed_schema6_autonomous_watcher"
TERMINAL_PENDING_STATUS = "terminal_success_pending_frozen_publication"
SCHEMA6_EXECUTION_ORDER = (
    "materialize_reset_only_registry",
    "freeze_one_seed_h1_authority",
    "collect_one_seed_h1_schema6",
)
SCHEMA6_STAGE_ACCEPTED_RETURNCODES = {
    "materialize_reset_only_registry": (0,),
    "freeze_one_seed_h1_authority": (0,),
    "collect_one_seed_h1_schema6": (0, 20),
}
DESIGNATED_LOBO_ROOT = Path(
    "/home/user/etsf_multibody_lobo_autonomous_r12_20260828"
)
DESIGNATED_LOBO_OUTPUTS = {
    "piper": Path("/home/user/etsf_multibody_lobo_piper_train_r12_20260828"),
    "ur5-wsg": Path("/home/user/etsf_multibody_lobo_ur5_train_r12_20260828"),
}
DESIGNATED_CODE_ROOT = Path(
    "/home/user/etsf_smolvla_piper_schema6_code_r6j_20260828"
)
DESIGNATED_PYTHON = Path(
    "/home/user/etsf_stage0/.venv_smolvla_robotwin_eval_np126/bin/python"
)
EXPECTED_GPU_FRAGMENT = "RTX 4090"
FIXED_REQUESTED_SEED = 100101000
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
HDF_SUFFIXES = (".hdf5", ".h5", ".hdf")
SHA256_ALPHABET = frozenset("0123456789abcdef")
SCRUBBED_PYTHON_ENVIRONMENT = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PROMPT_MODIFIER",
        "_CE_CONDA",
        "_CE_M",
    }
)
FORCED_PYTHON_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
}
ENTRYPOINTS = (
    "materialize_smolvla_piper_schema6_reset_contract.py",
    "freeze_smolvla_piper_schema6_development_collection.py",
    "launch_smolvla_piper_schema6_development_collection.py",
)
SOURCE_STAGE_NAMES = (
    "initialize_native_dual_reserved_core",
    "train_source63_counterfactual_five_seed",
)


class WatcherContractError(RuntimeError):
    """A required immutable/runtime contract cannot be proved."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_ALPHABET)
    )


def _contains_sensitive_component(path: PurePath) -> bool:
    return any(
        token in component.lower()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def reject_path_text(value: str | os.PathLike[str], role: str) -> Path:
    """Reject forbidden lexical components before any filesystem operation."""

    text = os.fspath(value)
    if not text or "\x00" in text:
        raise WatcherContractError(f"{role} path is empty/invalid")
    path = Path(os.path.abspath(os.path.expanduser(text)))
    if _contains_sensitive_component(PurePath(path)):
        raise WatcherContractError(f"{role} path contains a forbidden component")
    return path


def _audit_embedded_paths(value: Any, role: str = "document") -> None:
    """Reject forbidden absolute/relative path strings embedded in contracts."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _audit_embedded_paths(item, f"{role}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _audit_embedded_paths(item, f"{role}[{index}]")
    elif isinstance(value, str):
        looks_like_path = (
            value.startswith(("/", "./", "../"))
            or "\\" in value
            or ("/" in value and any(token in value.lower() for token in SENSITIVE_PATH_TOKENS))
        )
        if looks_like_path and _contains_sensitive_component(PurePath(value)):
            raise WatcherContractError(f"{role} embeds a forbidden path")


def resolve_existing(
    value: str | os.PathLike[str], *, role: str, directory: bool
) -> Path:
    lexical = reject_path_text(value, role)
    if lexical.is_symlink():
        raise WatcherContractError(f"{role} must not be a symlink")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise WatcherContractError(f"{role} does not exist") from error
    if _contains_sensitive_component(PurePath(resolved)):
        raise WatcherContractError(f"{role} resolves into a forbidden path")
    try:
        metadata = resolved.stat()
    except OSError as error:
        raise WatcherContractError(f"{role} cannot be stat-ed") from error
    matches = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not matches:
        kind = "directory" if directory else "regular file"
        raise WatcherContractError(f"{role} must be a {kind}")
    return resolved


def resolve_new(value: str | os.PathLike[str], *, role: str) -> Path:
    lexical = reject_path_text(value, role)
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError(f"{role} already exists: {lexical}")
    parent = resolve_existing(lexical.parent, role=f"{role} parent", directory=True)
    if parent != lexical.parent.resolve():
        raise WatcherContractError(f"{role} parent resolution changed")
    return lexical


def resolve_future_directory(value: str | os.PathLike[str], *, role: str) -> Path:
    """Validate a fixed watcher root that may be created after preregistration."""

    lexical = reject_path_text(value, role)
    if lexical.exists() or lexical.is_symlink():
        return resolve_existing(lexical, role=role, directory=True)
    parent = resolve_existing(lexical.parent, role=f"{role} parent", directory=True)
    if parent != lexical.parent.resolve():
        raise WatcherContractError(f"{role} parent resolution changed")
    return lexical


def _inside(root: Path, path: Path, *, role: str, directory: bool = False) -> Path:
    resolved_root = resolve_existing(root, role=f"{role} root", directory=True)
    resolved = resolve_existing(path, role=role, directory=directory)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise WatcherContractError(f"{role} escaped its bound root")
    return resolved


def file_sha256(path: Path) -> str:
    safe = reject_path_text(path, "SHA256 input")
    if safe.suffix.lower() in HDF_SUFFIXES:
        raise WatcherContractError("watcher is forbidden from opening HDF5 artifacts")
    digest = hashlib.sha256()
    with safe.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = reject_path_text(path, "atomic JSON output")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json_new(path: Path, value: Mapping[str, Any]) -> None:
    target = resolve_new(path, role="immutable JSON output")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, target)
        _fsync_directory(target.parent)
    except FileExistsError as error:
        raise FileExistsError(target) from error
    finally:
        temporary.unlink(missing_ok=True)


def immutable_text_new(path: Path, value: str) -> None:
    target = resolve_new(path, role="immutable text output")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise FileExistsError(target) from error
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(target.parent)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def load_json(path: Path, *, role: str) -> dict[str, Any]:
    safe = resolve_existing(path, role=role, directory=False)
    if safe.suffix.lower() in HDF_SUFFIXES:
        raise WatcherContractError(f"{role} must not be HDF5")
    try:
        value = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WatcherContractError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise WatcherContractError(f"{role} must contain a JSON object")
    _audit_embedded_paths(value, role)
    return value


def _load_signed_r6f_lineage_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read R6f while treating one old seed path as signed metadata only.

    The canonical R6f/R6e lineage predates the path-namespace isolation rule
    and embeds the old development seed manifest path.  That string is useful
    only as a content-addressed lineage commitment: it must never be resolved,
    stated, opened, or forwarded in a watcher artifact.  This narrowly scoped
    loader therefore permits exactly that JSON location, but only after the
    R6f logical signature, inherited-contract signature, and fixed label-free
    seed scalars have all been proved.  Every other embedded path still uses
    the global fail-closed audit above.
    """

    safe = resolve_existing(path, role="R6f preregistration", directory=False)
    if safe.suffix.lower() in HDF_SUFFIXES:
        raise WatcherContractError("R6f preregistration must not be HDF5")
    try:
        value = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WatcherContractError("R6f preregistration is not valid JSON") from error
    if not isinstance(value, dict):
        raise WatcherContractError("R6f preregistration must contain a JSON object")

    sensitive_locations: list[str] = []

    def audit(item: Any, location: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                audit(child, (*location, str(key)))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                audit(child, (*location, f"[{index}]"))
        elif isinstance(item, str):
            looks_like_path = (
                item.startswith(("/", "./", "../"))
                or "\\" in item
                or ("/" in item and any(token in item.lower() for token in SENSITIVE_PATH_TOKENS))
            )
            if looks_like_path and _contains_sensitive_component(PurePath(item)):
                sensitive_locations.append(".".join(location))

    audit(value, ())
    allowed_location = "inherited_R6e_contract.development_seed.path"
    if sensitive_locations and sensitive_locations != [allowed_location]:
        raise WatcherContractError("R6f preregistration embeds an unapproved forbidden path")

    inherited = value.get("inherited_R6e_contract")
    if not sensitive_locations:
        _audit_embedded_paths(value, "R6f preregistration")
        return value, {
            "format": "etsf_signed_legacy_seed_lineage_projection_v1",
            "legacy_sensitive_path_present": False,
            "legacy_sensitive_path_dereferenced": False,
        }
    if not isinstance(inherited, Mapping):
        raise WatcherContractError("R6f inherited R6e contract is missing")
    base = {key: item for key, item in value.items() if key != "preregistration_sha256"}
    if not _is_sha256(value.get("preregistration_sha256")) or value.get(
        "preregistration_sha256"
    ) != canonical_sha256(base):
        raise WatcherContractError("R6f logical preregistration SHA mismatch")
    if not _is_sha256(value.get("inherited_R6e_contract_sha256")) or value.get(
        "inherited_R6e_contract_sha256"
    ) != canonical_sha256(inherited):
        raise WatcherContractError("R6f inherited R6e contract SHA mismatch")
    seed = inherited.get("development_seed")
    if not isinstance(seed, Mapping) or set(seed) != {
        "path", "sha256", "seed_registry", "requested_seed",
        "expected_resolved_seed", "fresh_confirmation_eligible", "label_free",
    }:
        raise WatcherContractError("R6f signed development seed record changed")
    raw_path = seed.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path.startswith("/")
        or not _contains_sensitive_component(PurePath(raw_path))
        or not _is_sha256(seed.get("sha256"))
        or seed.get("seed_registry") != "explicit_v7_prospective_development"
        or seed.get("requested_seed") != FIXED_REQUESTED_SEED
        or seed.get("expected_resolved_seed") != FIXED_REQUESTED_SEED
        or seed.get("fresh_confirmation_eligible") is not False
        or seed.get("label_free") is not True
    ):
        raise WatcherContractError("R6f signed development seed metadata is invalid")
    audit_record: dict[str, Any] = {
        "format": "etsf_signed_legacy_seed_lineage_projection_v1",
        "legacy_sensitive_path_present": True,
        "legacy_sensitive_path_sha256": hashlib.sha256(raw_path.encode("utf-8")).hexdigest(),
        "legacy_seed_manifest_content_sha256": seed["sha256"],
        "r6f_logical_sha256": value["preregistration_sha256"],
        "inherited_r6e_contract_sha256": value["inherited_R6e_contract_sha256"],
        "legacy_sensitive_path_resolved": False,
        "legacy_sensitive_path_stated": False,
        "legacy_sensitive_path_opened": False,
        "legacy_sensitive_path_dereferenced": False,
    }
    audit_record["projection_audit_sha256"] = canonical_sha256(audit_record)
    return value, audit_record


def _require_read_only(path: Path, role: str) -> None:
    if path.stat().st_mode & 0o222:
        raise WatcherContractError(f"{role} is not frozen read-only")


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def implementation_closure(code_root: Path) -> dict[str, dict[str, Any]]:
    root = resolve_existing(code_root, role="schema6 immutable code root", directory=True)
    scripts = resolve_existing(root / "scripts", role="schema6 scripts", directory=True)
    queue = [scripts / name for name in ENTRYPOINTS]
    seen: set[Path] = set()
    while queue:
        path = resolve_existing(queue.pop(), role="schema6 implementation", directory=False)
        if scripts not in path.parents:
            raise WatcherContractError("schema6 implementation escaped code root")
        if path in seen:
            continue
        _require_read_only(path, "schema6 implementation")
        seen.add(path)
        for module in _local_imports(path):
            candidate = scripts / (module.replace(".", "/") + ".py")
            if candidate.is_file():
                queue.append(candidate)
    names = {path.name for path in seen}
    if not set(ENTRYPOINTS).issubset(names):
        raise WatcherContractError("schema6 implementation closure is incomplete")
    _require_read_only(root, "schema6 code root")
    return {
        str(path.relative_to(root)): {
            "path": str(path),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(seen)
    }


def python_contract(path: Path) -> dict[str, Any]:
    invocation = reject_path_text(path, "Python executable")
    try:
        resolved = invocation.resolve(strict=True)
    except OSError as error:
        raise WatcherContractError("Python executable does not exist") from error
    if _contains_sensitive_component(PurePath(resolved)) or not resolved.is_file():
        raise WatcherContractError("Python executable is invalid")
    if not os.access(resolved, os.X_OK):
        raise WatcherContractError("Python executable is not executable")
    return {
        "invocation_path": str(invocation),
        "resolved_path": str(resolved),
        "resolved_sha256": file_sha256(resolved),
    }


def _validate_logical_sha(document: Mapping[str, Any], field: str, role: str) -> str:
    recorded = document.get(field)
    base = {key: value for key, value in document.items() if key != field}
    if not _is_sha256(recorded) or recorded != canonical_sha256(base):
        raise WatcherContractError(f"{role} logical SHA is invalid")
    return str(recorded)


def _validate_source_stage_receipt(
    source_root: Path, stage_name: str
) -> tuple[dict[str, Any], str]:
    path = _inside(
        source_root,
        source_root / "stage_receipts" / f"{stage_name}.json",
        role=f"source {stage_name} receipt",
    )
    value = load_json(path, role=f"source {stage_name} receipt")
    if (
        value.get("format") != SOURCE_FORMAT
        or value.get("stage") != stage_name
        or value.get("status") != "complete"
        or value.get("returncode") != 0
        or not _is_sha256(value.get("argv_sha256"))
        or canonical_sha256(value.get("argv")) != value.get("argv_sha256")
        or not _is_sha256(value.get("log_sha256"))
    ):
        raise WatcherContractError(f"source {stage_name} did not exit zero cleanly")
    argv = value.get("argv")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise WatcherContractError(f"source {stage_name} argv is invalid")
    if any(Path(item).suffix.lower() in HDF_SUFFIXES for item in argv):
        raise WatcherContractError("source stage names a direct HDF5 input")
    log_path = _inside(
        source_root, Path(str(value.get("log", ""))), role=f"source {stage_name} log"
    )
    if file_sha256(log_path) != value["log_sha256"]:
        raise WatcherContractError(f"source {stage_name} log SHA changed")
    _require_read_only(path, f"source {stage_name} receipt")
    _require_read_only(log_path, f"source {stage_name} log")
    return value, file_sha256(path)


def validate_source_training_summary(source_root: Path) -> dict[str, Any]:
    """Validate source completion without opening any source/test HDF5 file."""

    root = resolve_existing(source_root, role="source training root", directory=True)
    final_path = _inside(root, root / "final_receipt.json", role="source final receipt")
    final = load_json(final_path, role="source final receipt")
    false_fields = (
        "target_data_read",
        "target_labels_read",
        "fresh_inputs_accepted",
        "fresh_labels_read",
        "test_labels_used",
    )
    if (
        final.get("format") != SOURCE_FORMAT
        or final.get("status") != SOURCE_TERMINAL_STATUS
        or any(final.get(field) is not False for field in false_fields)
        or final.get("test_hdf_label_datasets_opened") != 0
        or final.get("artifacts_frozen_read_only") is not True
    ):
        raise WatcherContractError("source terminal receipt contract is invalid")

    plan_path = _inside(root, root / "launch_plan.json", role="source launch plan")
    plan = load_json(plan_path, role="source launch plan")
    if (
        plan.get("format") != SOURCE_FORMAT
        or plan.get("output_root") != str(root)
        or plan.get("static_plan_sha256") != canonical_sha256(
            {key: value for key, value in plan.items() if key != "static_plan_sha256"}
        )
        or plan.get("static_plan_sha256") != final.get("static_plan_sha256")
        or plan.get("fresh_inputs_accepted") is not False
        or plan.get("hdf5_opened_during_static_preflight") is not False
    ):
        raise WatcherContractError("source static-plan summary contract is invalid")

    execution_path = _inside(
        root, root / "execution_plan.json", role="source execution plan"
    )
    execution = load_json(execution_path, role="source execution plan")
    if (
        execution.get("format") != SOURCE_FORMAT
        or execution.get("execution_order") != list(SOURCE_STAGE_NAMES)
        or execution.get("execution_plan_sha256") != canonical_sha256(
            {
                key: value
                for key, value in execution.items()
                if key != "execution_plan_sha256"
            }
        )
        or execution.get("execution_plan_sha256") != final.get("execution_plan_sha256")
        or execution.get("test_hdf_label_datasets_opened") != 0
    ):
        raise WatcherContractError("source execution-plan summary contract is invalid")

    stage_receipts: dict[str, dict[str, Any]] = {}
    stage_receipt_sha: dict[str, str] = {}
    for name in SOURCE_STAGE_NAMES:
        stage_receipts[name], stage_receipt_sha[name] = _validate_source_stage_receipt(
            root, name
        )

    state_path = _inside(root, root / "launch_state.json", role="source terminal state")
    state = load_json(state_path, role="source terminal state")
    if (
        state.get("format") != "etsf_smolvla_schema5_source63_native_training_state_v1"
        or state.get("status") != SOURCE_TERMINAL_STATUS
        or state.get("test_hdf_label_datasets_opened") != 0
        or any(
            state.get("stage_results", {}).get(name, {}).get("returncode") != 0
            for name in SOURCE_STAGE_NAMES
        )
    ):
        raise WatcherContractError("source terminal state does not prove exit=0")

    inventory_path = _inside(
        root, root / "artifact_inventory.json", role="source artifact inventory"
    )
    inventory = load_json(inventory_path, role="source artifact inventory")
    inventory_base = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    if (
        inventory.get("inventory_sha256") != canonical_sha256(inventory_base)
        or inventory.get("inventory_sha256") != final.get("artifact_inventory_sha256")
    ):
        raise WatcherContractError("source artifact inventory summary SHA is invalid")

    initialized = _inside(
        root,
        root / "smolvla_schema5_native_initialized.pt",
        role="source initialized checkpoint",
    )
    if file_sha256(initialized) != final.get("initialized_checkpoint_sha256"):
        raise WatcherContractError("source initialized checkpoint SHA changed")

    audit = final.get("training_audit")
    train_stage_audit = stage_receipts[SOURCE_STAGE_NAMES[1]].get("artifact_audit")
    if not isinstance(audit, Mapping) or audit != train_stage_audit:
        raise WatcherContractError("source final/stage training summaries disagree")
    if (
        audit.get("status") != SOURCE_TRAINING_STATUS
        or audit.get("member_count") != len(SOURCE_MEMBER_SEEDS)
        or audit.get("member_seeds") != list(SOURCE_MEMBER_SEEDS)
        or audit.get("member_training_steps_verified")
        != [SOURCE_TRAINING_STEPS] * len(SOURCE_MEMBER_SEEDS)
        or audit.get("target_data_read") is not False
        or audit.get("target_labels_read") is not False
        or audit.get("test_labels_used") is not False
        or audit.get("test_hdf_label_datasets_opened") != 0
        or audit.get("test_hdf_identity_attrs_opened") != 5
    ):
        raise WatcherContractError("source training summary is incomplete")
    for field in ("member_proof_sha256", "member_training_log_sha256"):
        values = audit.get(field)
        if (
            not isinstance(values, list)
            or len(values) != len(SOURCE_MEMBER_SEEDS)
            or any(not _is_sha256(item) for item in values)
        ):
            raise WatcherContractError(f"source training summary {field} is invalid")

    training_root = _inside(
        root, root / "counterfactual_training", role="source training output", directory=True
    )
    manifest_path = _inside(
        training_root,
        Path(str(audit.get("manifest_path", ""))),
        role="source ensemble manifest",
    )
    if manifest_path != training_root / "ensemble_manifest.json":
        raise WatcherContractError("source ensemble manifest path changed")
    if file_sha256(manifest_path) != audit.get("manifest_sha256"):
        raise WatcherContractError("source ensemble manifest SHA changed")
    manifest = load_json(manifest_path, role="source ensemble manifest")
    members = manifest.get("members")
    if (
        manifest.get("format") != "etsf_counterfactual_ensemble_v1"
        or manifest.get("test_policy")
        != "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
        or not isinstance(members, list)
        or [item.get("seed") for item in members] != list(SOURCE_MEMBER_SEEDS)
    ):
        raise WatcherContractError("source ensemble manifest contract is invalid")
    member_file_sha: list[str] = []
    member_log_sha: list[str] = []
    for index, member in enumerate(members):
        if not isinstance(member, Mapping) or not _is_sha256(member.get("sha256")):
            raise WatcherContractError("source ensemble member record is invalid")
        checkpoint = _inside(
            training_root,
            Path(str(member.get("path", ""))),
            role=f"source member {index} checkpoint",
        )
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != member["sha256"]:
            raise WatcherContractError("source ensemble member checkpoint SHA changed")
        log_path = _inside(
            training_root,
            checkpoint.parent / "train_log.jsonl",
            role=f"source member {index} training log",
        )
        member_file_sha.append(checkpoint_sha)
        member_log_sha.append(file_sha256(log_path))
        _require_read_only(checkpoint, f"source member {index} checkpoint")
        _require_read_only(log_path, f"source member {index} training log")
    if member_log_sha != audit["member_training_log_sha256"]:
        raise WatcherContractError("source member training-log SHA summary changed")

    ensemble_record = manifest.get("ensemble_checkpoint")
    if not isinstance(ensemble_record, Mapping):
        raise WatcherContractError("source ensemble checkpoint record is missing")
    ensemble_path = _inside(
        training_root,
        Path(str(ensemble_record.get("path", ""))),
        role="source ensemble checkpoint",
    )
    ensemble_sha = file_sha256(ensemble_path)
    if (
        ensemble_sha != ensemble_record.get("sha256")
        or ensemble_sha != audit.get("ensemble_checkpoint_sha256")
        or ensemble_path != Path(str(audit.get("ensemble_checkpoint", ""))).resolve()
    ):
        raise WatcherContractError("source ensemble checkpoint summary changed")

    for path, role in (
        (root, "source root"),
        (final_path, "source final receipt"),
        (plan_path, "source launch plan"),
        (execution_path, "source execution plan"),
        (state_path, "source terminal state"),
        (inventory_path, "source artifact inventory"),
        (initialized, "source initialized checkpoint"),
        (manifest_path, "source ensemble manifest"),
        (ensemble_path, "source ensemble checkpoint"),
    ):
        _require_read_only(path, role)

    summary: dict[str, Any] = {
        "status": "verified_source63_terminal_exit_zero_and_summary",
        "source_root": str(root),
        "final_receipt_sha256": file_sha256(final_path),
        "static_plan_sha256": final["static_plan_sha256"],
        "execution_plan_sha256": final["execution_plan_sha256"],
        "artifact_inventory_sha256": inventory["inventory_sha256"],
        "stage_returncodes": {name: 0 for name in SOURCE_STAGE_NAMES},
        "stage_receipt_sha256": stage_receipt_sha,
        "initialized_checkpoint_sha256": file_sha256(initialized),
        "ensemble_manifest_sha256": file_sha256(manifest_path),
        "ensemble_checkpoint_sha256": ensemble_sha,
        "member_checkpoint_sha256": member_file_sha,
        "member_training_log_sha256": member_log_sha,
        "member_seeds": list(SOURCE_MEMBER_SEEDS),
        "member_training_steps": [SOURCE_TRAINING_STEPS] * len(SOURCE_MEMBER_SEEDS),
        "source_exit_zero_proof": "both_recorded_subprocess_returncodes_zero_plus_terminal_receipt",
        "source_hdf5_opened_by_this_watcher": 0,
        "source_test_hdf5_opened_by_this_watcher": 0,
        "source_test_labels_read_by_this_watcher": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def wait_for_source_training(
    source_root: Path,
    *,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = resolve_existing(source_root, role="source training root", directory=True)
    started = time.monotonic()
    while True:
        failure_path = root / "failure_receipt.json"
        final_path = root / "final_receipt.json"
        if failure_path.exists() or failure_path.is_symlink():
            failure = load_json(failure_path, role="source failure receipt")
            raise WatcherContractError(
                "source63 trainer failed closed: "
                f"{failure.get('error_type', 'unknown')}: {failure.get('error', '')}"
            )
        if final_path.is_symlink():
            raise WatcherContractError("source terminal receipt symlink is forbidden")
        publication_frozen = (
            final_path.exists()
            and not final_path.is_symlink()
            and final_path.is_file()
            and root.stat().st_mode & 0o222 == 0
            and final_path.stat().st_mode & 0o222 == 0
        )
        if publication_frozen:
            state.update(
                {
                    "status": "validating_source_terminal_summary_no_hdf5",
                    "last_heartbeat_unix": time.time(),
                    "source_summary_read": True,
                }
            )
            atomic_json(state_path, state)
            return validate_source_training_summary(root)
        state.update(
            {
                "status": (
                    "waiting_for_source63_terminal_freeze_no_hdf5_access"
                    if final_path.exists()
                    else "waiting_for_source63_terminal_receipt_no_hdf5_access"
                ),
                "last_heartbeat_unix": time.time(),
                "source_summary_read": False,
                "source_hdf5_opened": 0,
                "source_test_hdf5_opened": 0,
            }
        )
        atomic_json(state_path, state)
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("timed out waiting for verified source63 training")
        sleep(poll_seconds)


def _read_exact_zero_exit(path: Path, *, role: str) -> str:
    safe = resolve_existing(path, role=role, directory=False)
    try:
        payload = safe.read_bytes()
    except OSError as error:
        raise WatcherContractError(f"{role} is unreadable") from error
    if payload != b"0\n":
        raise WatcherContractError(f"{role} is not exact exit zero")
    return file_sha256(safe)


def _validate_lobo_stage_lifecycle_proof(
    *, stage: str, result: Mapping[str, Any], lifecycle: Mapping[str, Any]
) -> None:
    process_pid = result.get("pid")
    result_group = result.get("process_group_id")
    result_returncode = result.get("returncode")
    lifecycle_pid = lifecycle.get("process_pid")
    lifecycle_group = lifecycle.get("process_group_id")
    lifecycle_returncode = lifecycle.get("returncode")
    if (
        isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid <= 0
        or isinstance(result_group, bool)
        or not isinstance(result_group, int)
        or result_group != process_pid
        or isinstance(result_returncode, bool)
        or not isinstance(result_returncode, int)
        or result_returncode != 0
        or result.get("stage") != stage
        or result.get("status") != "complete"
        or result.get("process_reaped") is not True
        or result.get("process_group_isolated") is not True
        or result.get("process_group_reaped") is not True
        or lifecycle.get("stage") != stage
        or lifecycle.get("popen_attempted") is not True
        or lifecycle.get("popen_reached") is not True
        or isinstance(lifecycle_pid, bool)
        or not isinstance(lifecycle_pid, int)
        or lifecycle_pid != process_pid
        or lifecycle.get("process_reaped") is not True
        or isinstance(lifecycle_group, bool)
        or not isinstance(lifecycle_group, int)
        or lifecycle_group != process_pid
        or lifecycle.get("process_group_isolated") is not True
        or lifecycle.get("process_group_reaped") is not True
        or isinstance(lifecycle_returncode, bool)
        or not isinstance(lifecycle_returncode, int)
        or lifecycle_returncode != 0
    ):
        raise WatcherContractError(
            f"LOBO {stage} process lifecycle proof is invalid"
        )


def _validate_lobo_native_source_binding(
    *,
    lobo_root: Path,
    final: Mapping[str, Any],
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    binding_audit = final.get("source_binding_receipt")
    deployment = final.get("deployment_rerank_checkpoint")
    source_checkpoint = source_audit.get("ensemble_checkpoint")
    source_checkpoint_sha = source_audit.get("ensemble_checkpoint_sha256")
    source_bridge_sha = source_audit.get("policy_feature_action_bridge_sha256")
    if (
        not isinstance(binding_audit, Mapping)
        or not isinstance(deployment, Mapping)
        or not isinstance(source_checkpoint, str)
        or not source_checkpoint
        or not _is_sha256(source_checkpoint_sha)
        or not _is_sha256(source_bridge_sha)
        or deployment.get("path") != source_checkpoint
        or deployment.get("sha256") != source_checkpoint_sha
        or deployment.get("policy") != "smolvla"
        or deployment.get("checkpoint_family")
        != "smolvla_native_event_world_model"
        or deployment.get("policy_feature_action_bridge_contract_sha256")
        != source_bridge_sha
        or deployment.get("source_native_checkpoint") is not True
        or binding_audit.get("deployment_rerank_checkpoint") != deployment
        or binding_audit.get("policy_feature_action_bridge_contract_sha256")
        != source_bridge_sha
        or binding_audit.get("lobo_checkpoints_rerank_authorized") is not False
    ):
        raise WatcherContractError("LOBO native source deployment binding is incomplete")
    binding_path = _inside(
        lobo_root,
        Path(str(binding_audit.get("path", ""))),
        role="LOBO source binding receipt",
    )
    if binding_path != lobo_root / "source_binding_receipt.json":
        raise WatcherContractError("LOBO source binding receipt path changed")
    binding_file_sha = file_sha256(binding_path)
    binding_document = load_json(binding_path, role="LOBO source binding receipt")
    binding_unsigned = dict(binding_document)
    binding_logical_sha = binding_unsigned.pop("binding_sha256", None)
    source_final_binding = binding_document.get("source_final_receipt")
    source_plan_binding = binding_document.get("source_launch_plan")
    if (
        binding_file_sha != binding_audit.get("file_sha256")
        or binding_logical_sha != binding_audit.get("binding_sha256")
        or binding_logical_sha != canonical_sha256(binding_unsigned)
        or binding_document.get("format") != LOBO_SOURCE_BINDING_FORMAT
        or binding_document.get("status") != LOBO_SOURCE_BINDING_STATUS
        or binding_document.get("deployment_rerank_checkpoint") != deployment
        or binding_document.get("policy_feature_action_bridge_contract_sha256")
        != source_bridge_sha
        or binding_document.get("lobo_checkpoints_rerank_authorized") is not False
        or binding_document.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
        or not isinstance(source_plan_binding, Mapping)
        or source_plan_binding.get("file_sha256")
        != EXPECTED_SOURCE_PLAN_SHA256
        or source_plan_binding.get("logical_sha256")
        != EXPECTED_SOURCE_STATIC_PLAN_SHA256
        or not isinstance(source_final_binding, Mapping)
        or source_final_binding.get("path")
        != source_audit.get("final_receipt_path")
        or source_final_binding.get("file_sha256")
        != source_audit.get("final_receipt_sha256")
        or source_final_binding.get("logical_sha256")
        != source_audit.get("final_receipt_logical_sha256")
        or source_final_binding.get("status") != SOURCE_TERMINAL_STATUS
    ):
        raise WatcherContractError("LOBO source binding receipt content changed")
    source_root_value = binding_document.get("source_training_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise WatcherContractError("LOBO source training root binding is missing")
    source_root = resolve_existing(
        Path(source_root_value), role="bound native source root", directory=True
    )
    expected_source_root = resolve_existing(
        EXPECTED_SOURCE_ROOT, role="designated r7h source root", directory=True
    )
    if source_root != expected_source_root:
        raise WatcherContractError("LOBO source binding does not name designated r7h")
    source_plan = resolve_existing(
        Path(str(source_plan_binding.get("path", ""))),
        role="bound r7h source launch plan",
        directory=False,
    )
    if source_plan != source_root / "launch_plan.json":
        raise WatcherContractError("bound r7h source launch plan path changed")
    source_plan_document = load_json(source_plan, role="bound r7h source launch plan")
    source_plan_unsigned = dict(source_plan_document)
    source_plan_logical = source_plan_unsigned.pop("static_plan_sha256", None)
    implementation_files = source_plan_document.get("implementation_files")
    launcher_records = (
        [
            record
            for relative, record in implementation_files.items()
            if isinstance(relative, str)
            and relative.endswith("launch_smolvla_schema5_source63_native_training.py")
            and isinstance(record, Mapping)
        ]
        if isinstance(implementation_files, Mapping)
        else []
    )
    if (
        file_sha256(source_plan) != EXPECTED_SOURCE_PLAN_SHA256
        or source_plan_logical != EXPECTED_SOURCE_STATIC_PLAN_SHA256
        or source_plan_logical != canonical_sha256(source_plan_unsigned)
        or source_plan_document.get("output_root") != str(source_root)
        or source_plan_document.get("implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
        or len(launcher_records) != 1
        or launcher_records[0].get("sha256")
        != EXPECTED_SOURCE_LAUNCHER_SHA256
    ):
        raise WatcherContractError("bound r7h source launch plan changed")
    checkpoint = resolve_existing(
        Path(source_checkpoint), role="bound native source ensemble", directory=False
    )
    source_final = resolve_existing(
        Path(str(source_audit.get("final_receipt_path", ""))),
        role="bound native source final receipt",
        directory=False,
    )
    if (
        source_root not in checkpoint.parents
        or source_root not in source_final.parents
        or file_sha256(checkpoint) != source_checkpoint_sha
        or file_sha256(source_final) != source_audit.get("final_receipt_sha256")
    ):
        raise WatcherContractError("bound native source artifacts changed")
    for path, role in (
        (binding_path, "LOBO source binding receipt"),
        (source_root, "bound native source root"),
        (source_plan, "bound r7h source launch plan"),
        (checkpoint, "bound native source ensemble"),
        (source_final, "bound native source final receipt"),
    ):
        _require_read_only(path, role)
    return {
        "source_training_root": str(source_root),
        "source_launch_plan": {
            "path": str(source_plan),
            "file_sha256": EXPECTED_SOURCE_PLAN_SHA256,
            "logical_sha256": EXPECTED_SOURCE_STATIC_PLAN_SHA256,
        },
        "source_launcher_sha256": EXPECTED_SOURCE_LAUNCHER_SHA256,
        "source_implementation_bundle_sha256": (
            EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
        ),
        "source_binding_receipt_path": str(binding_path),
        "source_binding_receipt_file_sha256": binding_file_sha,
        "source_binding_sha256": binding_logical_sha,
        "deployment_rerank_checkpoint": dict(deployment),
        "policy_feature_action_bridge_sha256": source_bridge_sha,
        "lobo_checkpoints_rerank_authorized": False,
    }


def _validate_lobo_stage_source_binding_contract(
    *,
    stage: str,
    body: str,
    result: Mapping[str, Any],
    artifact_audit: Mapping[str, Any],
    source_binding_audit: Mapping[str, Any],
) -> None:
    contract = result.get("source_binding_contract")
    descriptor = contract.get("source_binding_receipt") if isinstance(contract, Mapping) else None
    if not isinstance(contract, Mapping) or not isinstance(descriptor, Mapping):
        raise WatcherContractError(f"LOBO {stage} source binding contract is missing")
    unsigned = dict(contract)
    logical = unsigned.pop("contract_sha256", None)
    if (
        logical != canonical_sha256(unsigned)
        or contract.get("format") != LOBO_STAGE_SOURCE_BINDING_FORMAT
        or contract.get("stage") != stage
        or contract.get("held_out_body") != body
        or contract.get("argv_sha256") != result.get("argv_sha256")
        or descriptor.get("path")
        != source_binding_audit["source_binding_receipt_path"]
        or descriptor.get("file_sha256")
        != source_binding_audit["source_binding_receipt_file_sha256"]
        or descriptor.get("binding_sha256")
        != source_binding_audit["source_binding_sha256"]
        or contract.get("deployment_rerank_checkpoint")
        != source_binding_audit["deployment_rerank_checkpoint"]
        or contract.get("policy_feature_action_bridge_contract_sha256")
        != source_binding_audit["policy_feature_action_bridge_sha256"]
        or contract.get("lobo_checkpoints_rerank_authorized") is not False
        or artifact_audit.get("source_binding_contract") != contract
    ):
        raise WatcherContractError(f"LOBO {stage} source binding contract changed")


def validate_lobo_terminal_summary(lobo_root: Path) -> dict[str, Any]:
    """Authenticate the aggregate Piper+UR5 LOBO gate without HDF5 access."""

    root = resolve_existing(lobo_root, role="LOBO autonomous root", directory=True)
    designated = reject_path_text(DESIGNATED_LOBO_ROOT, "designated LOBO root")
    if root != designated.resolve(strict=True):
        raise WatcherContractError("LOBO root is not the designated aggregate run")
    final_path = _inside(root, root / "final_receipt.json", role="LOBO final receipt")
    final = load_json(final_path, role="LOBO final receipt")
    receipt_sha = _validate_logical_sha(final, "receipt_sha256", "LOBO final receipt")
    source_binding = final.get("source_binding_receipt")
    deployment_checkpoint = final.get("deployment_rerank_checkpoint")
    if (
        final.get("format") != LOBO_WATCHER_FORMAT
        or final.get("status") != LOBO_WATCHER_TERMINAL_STATUS
        or final.get("static_plan_sha256") != EXPECTED_LOBO_STATIC_PLAN_SHA256
        or final.get("execution_order") != [stage for stage, _body in LOBO_STAGES]
        or final.get("watcher_hdf5_opened") != 0
        or final.get("test_hdf5_opened_by_watcher") != 0
        or final.get("test_labels_read_by_watcher") is not False
        or final.get("target_unused_train_payload_opened") != 0
        or final.get("test_group_hdf5_opened") != 0
        or final.get("artifacts_frozen_read_only") is not True
        or final.get("lobo_checkpoints_rerank_authorized") is not False
        or final.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
        or not isinstance(source_binding, Mapping)
        or source_binding.get("lobo_checkpoints_rerank_authorized") is not False
        or not isinstance(deployment_checkpoint, Mapping)
        or source_binding.get("deployment_rerank_checkpoint")
        != deployment_checkpoint
        or deployment_checkpoint.get("source_native_checkpoint") is not True
        or deployment_checkpoint.get("policy") != "smolvla"
        or not _is_sha256(deployment_checkpoint.get("sha256"))
    ):
        raise WatcherContractError("LOBO terminal aggregate contract is invalid")
    for field in ("fresh_inputs_accepted", "fresh_labels_read"):
        if final.get(field) is not False:
            raise WatcherContractError("LOBO terminal aggregate accepted forbidden inputs")
    source_audit = final.get("source63_audit")
    if (
        not isinstance(source_audit, Mapping)
        or source_audit.get("status") != SOURCE_TERMINAL_STATUS
        or not _is_sha256(source_audit.get("final_receipt_sha256"))
        or not _is_sha256(source_audit.get("final_receipt_logical_sha256"))
        or not _is_sha256(source_audit.get("static_plan_sha256"))
        or not _is_sha256(source_audit.get("execution_plan_sha256"))
        or not isinstance(source_audit.get("ensemble_checkpoint"), str)
        or not source_audit.get("ensemble_checkpoint")
        or not _is_sha256(source_audit.get("ensemble_checkpoint_sha256"))
        or not _is_sha256(
            source_audit.get("policy_feature_action_bridge_sha256")
        )
        or source_audit.get("member_count") != 5
        or source_audit.get("member_training_steps_verified")
        != [SOURCE_TRAINING_STEPS] * 5
        or source_audit.get("output_tree_read_only") is not True
        or source_audit.get("test_hdf_label_datasets_opened") != 0
    ):
        raise WatcherContractError("LOBO receipt lacks its authenticated source63 gate")
    source_binding_audit = _validate_lobo_native_source_binding(
        lobo_root=root,
        final=final,
        source_audit=source_audit,
    )

    plan_path = _inside(root, root / "launch_plan.json", role="LOBO launch plan")
    plan = load_json(plan_path, role="LOBO launch plan")
    if (
        plan.get("format") != LOBO_WATCHER_FORMAT
        or plan.get("output_root") != str(root)
        or plan.get("execution_order") != final["execution_order"]
        or plan.get("launcher", {}).get("sha256") != EXPECTED_LOBO_LAUNCHER_SHA256
        or plan.get("static_plan_sha256") != canonical_sha256(
            {key: value for key, value in plan.items() if key != "static_plan_sha256"}
        )
        or plan.get("static_plan_sha256") != EXPECTED_LOBO_STATIC_PLAN_SHA256
        or plan.get("static_plan_sha256") != final["static_plan_sha256"]
        or plan.get("watcher_hdf5_opened") != 0
        or plan.get("test_hdf5_opened_by_watcher") != 0
        or plan.get("lobo_checkpoints_rerank_authorized") is not False
        or plan.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
    ):
        raise WatcherContractError("LOBO launch plan disagrees with terminal receipt")
    expected_outputs = {
        "piper": str(DESIGNATED_LOBO_OUTPUTS["piper"]),
        "ur5": str(DESIGNATED_LOBO_OUTPUTS["ur5-wsg"]),
    }
    if final.get("preregistered_outputs") != expected_outputs:
        raise WatcherContractError("LOBO aggregate output roots changed")

    watcher_exit_path = _inside(root, root / "run.exit", role="LOBO watcher exit")
    watcher_exit_sha = _read_exact_zero_exit(
        watcher_exit_path, role="LOBO watcher exit"
    )
    state_path = _inside(root, root / "launch_state.json", role="LOBO terminal state")
    state = load_json(state_path, role="LOBO terminal state")
    if (
        state.get("format") != "etsf_multibody_lobo_autonomous_state_v1"
        or state.get("status") != LOBO_WATCHER_FROZEN_STATE_STATUS
        or state.get("execution_order") != final["execution_order"]
        or state.get("stage_results") != final.get("stage_results")
        or state.get("stages_started") != final["execution_order"]
        or state.get("stage_lifecycles") != final.get("stage_lifecycles")
        or state.get("current_stage") is not None
        or state.get("stage_pid") is not None
        or state.get("test_hdf5_opened_by_watcher") != 0
        or state.get("test_labels_read_by_watcher") is not False
        or state.get("target_unused_train_payload_opened") != 0
        or state.get("test_group_hdf5_opened") != 0
    ):
        raise WatcherContractError("LOBO terminal state contract is invalid")

    final_stages = final.get("stage_results")
    if not isinstance(final_stages, Mapping) or set(final_stages) != {
        stage for stage, _body in LOBO_STAGES
    }:
        raise WatcherContractError("LOBO terminal stage results are incomplete")
    final_lifecycles = final.get("stage_lifecycles")
    if (
        not isinstance(final_lifecycles, list)
        or len(final_lifecycles) != len(LOBO_STAGES)
        or [
            lifecycle.get("stage")
            for lifecycle in final_lifecycles
            if isinstance(lifecycle, Mapping)
        ]
        != [stage for stage, _body in LOBO_STAGES]
    ):
        raise WatcherContractError("LOBO terminal lifecycles are incomplete")
    lifecycle_by_stage = {
        str(lifecycle["stage"]): lifecycle for lifecycle in final_lifecycles
    }
    stage_audits: dict[str, Any] = {}
    for stage, body in LOBO_STAGES:
        result = final_stages[stage]
        if not isinstance(result, Mapping):
            raise WatcherContractError(f"LOBO {stage} result is invalid")
        stage_root = _inside(
            root, root / "stages" / stage, role=f"LOBO {stage} root", directory=True
        )
        receipt_path = _inside(
            stage_root, stage_root / "stage_receipt.json", role=f"LOBO {stage} receipt"
        )
        stage_receipt = load_json(receipt_path, role=f"LOBO {stage} receipt")
        if stage_receipt != result or state.get("stage_results", {}).get(stage) != result:
            raise WatcherContractError(f"LOBO {stage} receipt/state/final disagree")
        _validate_lobo_stage_lifecycle_proof(
            stage=stage,
            result=result,
            lifecycle=lifecycle_by_stage[stage],
        )
        argv = result.get("argv")
        if (
            result.get("format") != LOBO_WATCHER_FORMAT
            or result.get("stage") != stage
            or result.get("held_out_body") != body
            or result.get("status") != "complete"
            or result.get("returncode") != 0
            or result.get("lobo_checkpoints_rerank_authorized") is not False
            or result.get("deployment_rerank_checkpoint")
            != deployment_checkpoint
            or not isinstance(argv, list)
            or result.get("argv_sha256") != canonical_sha256(argv)
            or any(Path(str(item)).suffix.lower() in HDF_SUFFIXES for item in argv)
            or not _is_sha256(result.get("run_exit_sha256"))
            or not _is_sha256(result.get("log_sha256"))
        ):
            raise WatcherContractError(f"LOBO {stage} exit/argv contract is invalid")
        exit_path = _inside(
            stage_root,
            Path(str(result.get("run_exit", ""))),
            role=f"LOBO {stage} run.exit",
        )
        log_path = _inside(
            stage_root, Path(str(result.get("log", ""))), role=f"LOBO {stage} log"
        )
        if (
            exit_path != stage_root / "run.exit"
            or log_path != stage_root / "run.log"
            or _read_exact_zero_exit(exit_path, role=f"LOBO {stage} run.exit")
            != result["run_exit_sha256"]
            or file_sha256(log_path) != result["log_sha256"]
        ):
            raise WatcherContractError(f"LOBO {stage} exit/log SHA changed")
        audit = result.get("artifact_audit")
        if isinstance(audit, Mapping):
            _validate_lobo_stage_source_binding_contract(
                stage=stage,
                body=body,
                result=result,
                artifact_audit=audit,
                source_binding_audit=source_binding_audit,
            )
        expected_output = DESIGNATED_LOBO_OUTPUTS[body]
        if (
            result.get("output") != str(expected_output)
            or not isinstance(audit, Mapping)
            or audit.get("status") != LOBO_OUTPUT_TERMINAL_STATUS
            or audit.get("held_out_body") != body
            or not _is_sha256(audit.get("summary_sha256"))
            or not _is_sha256(audit.get("artifact_inventory_sha256"))
            or audit.get("target_unused_train_payload_opened") != 0
            or audit.get("test_group_hdf5_opened") != 0
            or audit.get("test_labels_read_by_watcher") is not False
            or audit.get("lobo_checkpoints_rerank_authorized") is not False
            or audit.get("deployment_rerank_checkpoint")
            != deployment_checkpoint
        ):
            raise WatcherContractError(f"LOBO {stage} artifact audit is invalid")
        external_root = resolve_existing(
            expected_output, role=f"LOBO {body} external output", directory=True
        )
        summary_path = _inside(
            external_root,
            external_root / "lobo_training_summary.json",
            role=f"LOBO {body} training summary",
        )
        if audit.get("summary_path") != str(summary_path):
            raise WatcherContractError(f"LOBO {stage} summary path changed")
        if file_sha256(summary_path) != audit["summary_sha256"]:
            raise WatcherContractError(f"LOBO {stage} summary SHA changed")
        summary = load_json(summary_path, role=f"LOBO {body} training summary")
        if (
            summary.get("format") != LOBO_OUTPUT_FORMAT
            or summary.get("status") != LOBO_OUTPUT_TERMINAL_STATUS
            or summary.get("held_out_body") != body
            or summary.get("estimand")
            != "zero_target_label_leave_one_body_out_transfer"
            or summary.get("target_development_opened_after_all_checkpoint_selection")
            is not True
            or summary.get("target_unused_train_payload_opened") != 0
            or summary.get("sealed_test_evaluated") is not False
            or summary.get("test_group_hdf5_opened") != 0
        ):
            raise WatcherContractError(f"LOBO {stage} training summary is invalid")
        _require_read_only(external_root, f"LOBO {body} external output root")
        _require_read_only(summary_path, f"LOBO {body} training summary")
        _require_read_only(receipt_path, f"LOBO {stage} receipt")
        _require_read_only(exit_path, f"LOBO {stage} run.exit")
        _require_read_only(log_path, f"LOBO {stage} log")
        stage_audits[stage] = {
            "held_out_body": body,
            "returncode": 0,
            "stage_receipt_sha256": file_sha256(receipt_path),
            "run_exit_sha256": result["run_exit_sha256"],
            "log_sha256": result["log_sha256"],
            "summary_sha256": audit["summary_sha256"],
            "artifact_inventory_sha256": audit["artifact_inventory_sha256"],
        }

    for path, role in (
        (root, "LOBO autonomous root"),
        (final_path, "LOBO final receipt"),
        (plan_path, "LOBO launch plan"),
        (state_path, "LOBO terminal state"),
        (watcher_exit_path, "LOBO watcher exit"),
    ):
        _require_read_only(path, role)
    result: dict[str, Any] = {
        "status": "verified_piper_then_ur5_lobo_terminal_exit_zero",
        "lobo_root": str(root),
        "lobo_launcher_sha256": EXPECTED_LOBO_LAUNCHER_SHA256,
        "final_receipt_sha256": file_sha256(final_path),
        "final_receipt_logical_sha256": receipt_sha,
        "static_plan_sha256": final["static_plan_sha256"],
        "source_binding_receipt": dict(source_binding),
        "deployment_rerank_checkpoint": dict(deployment_checkpoint),
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
        "watcher_run_exit_sha256": watcher_exit_sha,
        "execution_order": final["execution_order"],
        "transitive_source63_audit_sha256": canonical_sha256(source_audit),
        "native_source_binding_audit": source_binding_audit,
        "native_source_binding_audit_sha256": canonical_sha256(
            source_binding_audit
        ),
        "stage_audits": stage_audits,
        "watcher_hdf5_opened": 0,
        "test_hdf5_opened_by_watcher": 0,
        "test_labels_read_by_watcher": False,
        "target_unused_train_payload_opened": 0,
        "test_group_hdf5_opened": 0,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def wait_for_lobo_completion(
    lobo_root: Path,
    *,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    root = reject_path_text(lobo_root, "LOBO autonomous root")
    designated = reject_path_text(DESIGNATED_LOBO_ROOT, "designated LOBO root")
    if root != designated:
        raise WatcherContractError("LOBO wait root differs from designated aggregate root")
    started = time.monotonic()
    while True:
        if root.exists() or root.is_symlink():
            materialized = resolve_existing(root, role="LOBO autonomous root", directory=True)
            failure_path = materialized / "failure_receipt.json"
            final_path = materialized / "final_receipt.json"
            if failure_path.exists() or failure_path.is_symlink():
                failure = load_json(failure_path, role="LOBO failure receipt")
                raise WatcherContractError(
                    "LOBO aggregate failed closed: "
                    f"{failure.get('error_type', 'unknown')}: {failure.get('error', '')}"
                )
            watcher_exit = materialized / "run.exit"
            if final_path.is_symlink() or watcher_exit.is_symlink():
                raise WatcherContractError("LOBO terminal publication symlink is forbidden")
            publication_frozen = (
                final_path.exists()
                and not final_path.is_symlink()
                and final_path.is_file()
                and watcher_exit.exists()
                and not watcher_exit.is_symlink()
                and watcher_exit.is_file()
                and materialized.stat().st_mode & 0o222 == 0
                and final_path.stat().st_mode & 0o222 == 0
                and watcher_exit.stat().st_mode & 0o222 == 0
            )
            if publication_frozen:
                state.update(
                    {
                        "status": "validating_piper_then_ur5_lobo_terminal_no_hdf5",
                        "lobo_summary_read": True,
                        "last_heartbeat_unix": time.time(),
                    }
                )
                atomic_json(state_path, state)
                return validate_lobo_terminal_summary(materialized)
        state.update(
            {
                "status": (
                    "waiting_for_piper_then_ur5_lobo_terminal_freeze_no_hdf5_access"
                    if root.exists() and (root / "final_receipt.json").exists()
                    else "waiting_for_piper_then_ur5_lobo_terminal_no_hdf5_access"
                ),
                "lobo_summary_read": False,
                "lobo_watcher_hdf5_opened": 0,
                "lobo_test_hdf5_opened": 0,
                "last_heartbeat_unix": time.time(),
            }
        )
        atomic_json(state_path, state)
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("timed out waiting for Piper/UR5 LOBO aggregate")
        sleep(poll_seconds)


def _run_nvidia_smi(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise WatcherContractError("unable to audit designated GPU") from error


def audit_idle_4090(
    gpu_index: int,
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run_nvidia_smi,
) -> dict[str, Any]:
    identity_command = (
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=index,name,uuid",
        "--format=csv,noheader",
    )
    identity_result = runner(identity_command)
    rows = [row.strip() for row in identity_result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise WatcherContractError("GPU identity query did not return exactly one device")
    fields = [field.strip() for field in rows[0].split(",", 2)]
    if len(fields) != 3 or fields[0] != str(gpu_index) or EXPECTED_GPU_FRAGMENT not in fields[1]:
        raise WatcherContractError("selected GPU is not the designated RTX 4090")
    compute_command = (
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    )
    compute_result = runner(compute_command)
    compute_rows = [row.strip() for row in compute_result.stdout.splitlines() if row.strip()]
    audit: dict[str, Any] = {
        "status": "idle_designated_rtx4090" if not compute_rows else "busy_designated_rtx4090",
        "gpu_index": gpu_index,
        "gpu_name": fields[1],
        "gpu_uuid": fields[2],
        "compute_process_count": len(compute_rows),
        "compute_process_rows_sha256": canonical_sha256(compute_rows),
        "audited_unix": time.time(),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def wait_for_idle_4090(
    *,
    gpu_index: int,
    state: dict[str, Any],
    state_path: Path,
    phase: str,
    poll_seconds: float,
    timeout_seconds: float,
    audit: Callable[[int], dict[str, Any]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    auditor = audit or (lambda index: audit_idle_4090(index))
    started = time.monotonic()
    while True:
        result = auditor(gpu_index)
        if result.get("status") == "idle_designated_rtx4090" and result.get(
            "compute_process_count"
        ) == 0:
            return result
        if result.get("status") != "busy_designated_rtx4090":
            raise WatcherContractError("GPU audit returned an unknown state")
        state.update(
            {
                "status": f"waiting_for_idle_rtx4090_{phase}",
                "last_gpu_audit": result,
                "last_heartbeat_unix": time.time(),
            }
        )
        atomic_json(state_path, state)
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for idle 4090 before {phase}")
        sleep(poll_seconds)


def acquire_lock(path: Path, payload: Mapping[str, Any]) -> None:
    target = reject_path_text(path, "pipeline lock")
    parent = resolve_existing(target.parent, role="pipeline lock parent", directory=True)
    if parent != target.parent.resolve():
        raise WatcherContractError("pipeline lock parent resolution changed")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise WatcherContractError(f"concurrent/stale pipeline lock exists: {target}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(parent)


def release_owned_lock(path: Path, token: str) -> None:
    try:
        value = load_json(path, role="owned pipeline lock")
        if value.get("pid") == os.getpid() and value.get("token") == token:
            path.unlink()
            _fsync_directory(path.parent)
    except (OSError, WatcherContractError):
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


def _wait_process_group_gone(process_group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
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


def _unreaped_schema6_stage(
    lifecycles: Sequence[Mapping[str, Any]],
) -> str | None:
    for lifecycle in lifecycles:
        if lifecycle.get("popen_attempted") is True and not (
            lifecycle.get("popen_reached") is True
            and lifecycle.get("process_reaped") is True
            and lifecycle.get("process_group_isolated") is True
            and lifecycle.get("process_group_id") == lifecycle.get("process_pid")
            and lifecycle.get("process_group_reaped") is True
        ):
            return str(lifecycle.get("stage", "unknown_schema6_stage"))
    return None


def _owned_gpu_lock_release_allowed(
    *, gpu_lock_acquired: bool, stage_lifecycles: Sequence[Mapping[str, Any]]
) -> bool:
    return (
        gpu_lock_acquired
        and _unreaped_schema6_stage(stage_lifecycles) is None
    )


def _argv_sha(argv: Sequence[str]) -> str:
    if any(Path(item).suffix.lower() in HDF_SUFFIXES for item in argv):
        raise WatcherContractError("a pipeline command directly names an HDF5 file")
    for item in argv:
        if item.startswith("/"):
            reject_path_text(item, "pipeline argv path")
    return canonical_sha256(list(argv))


def isolated_stage_environment(
    base: Mapping[str, str], *, gpu_index: int, omp_threads: int
) -> tuple[dict[str, str], dict[str, Any]]:
    environment = {str(key): str(value) for key, value in base.items()}
    removed_present = sorted(
        name for name in SCRUBBED_PYTHON_ENVIRONMENT if name in environment
    )
    for name in SCRUBBED_PYTHON_ENVIRONMENT:
        environment.pop(name, None)
    forced = {
        **FORCED_PYTHON_ENVIRONMENT,
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "OMP_NUM_THREADS": str(omp_threads),
    }
    environment.update(forced)
    if any(name in environment for name in SCRUBBED_PYTHON_ENVIRONMENT):
        raise WatcherContractError("Python isolation environment scrub failed")
    audit: dict[str, Any] = {
        "format": "etsf_isolated_python_subprocess_environment_v1",
        "status": "isolated_explicit_venv_environment",
        "scrubbed_names": sorted(SCRUBBED_PYTHON_ENVIRONMENT),
        "scrubbed_names_present_in_parent": removed_present,
        "forced_environment": forced,
        "pythonpath_inherited": False,
        "pythonhome_inherited": False,
        "virtualenv_or_conda_metadata_inherited": False,
        "unrelated_environment_values_recorded": False,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return environment, audit


def build_commands(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    code = Path(str(plan["code_root"])) / "scripts"
    python = str(plan["python_contract"]["invocation_path"])
    output = Path(str(plan["output_root"]))
    reset = output / "reset_contract"
    authority = output / "collection_authority.json"
    collection = output / "schema6_collection"
    materialize_argv = [
        python,
        str(code / ENTRYPOINTS[0]),
        "--r6f-preregistration",
        str(plan["r6f_preregistration"]),
        "--output-directory",
        str(reset),
    ]
    freeze_argv = [
        python,
        str(code / ENTRYPOINTS[1]),
        "--r6f-preregistration",
        str(plan["r6f_preregistration"]),
        "--object-registry",
        str(reset / "object_registry.json"),
        "--pose-quality-spec",
        str(reset / "pose_quality_spec.json"),
        "--event-spec",
        str(plan["event_spec"]),
        "--output-directory",
        str(collection),
        "--max-episode-steps",
        str(plan["max_episode_steps"]),
        "--output",
        str(authority),
    ]
    collect_argv = [
        python,
        str(code / ENTRYPOINTS[2]),
        "--preregistration",
        str(authority),
    ]
    rows = [
        ("materialize_reset_only_registry", materialize_argv, [0]),
        ("freeze_one_seed_h1_authority", freeze_argv, [0]),
        ("collect_one_seed_h1_schema6", collect_argv, [0, 20]),
    ]
    return [
        {
            "stage": name,
            "argv": argv,
            "argv_sha256": _argv_sha(argv),
            "accepted_returncodes": accepted,
        }
        for name, argv, accepted in rows
    ]


def _last_json_object(log_path: Path) -> dict[str, Any]:
    safe = resolve_existing(log_path, role="stage log", directory=False)
    lines = safe.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            _audit_embedded_paths(value, "stage stdout")
            return value
    raise WatcherContractError("stage log contains no JSON result")


def validate_materializer_output(plan: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    output = Path(str(plan["output_root"])) / "reset_contract"
    root = resolve_existing(output, role="reset-only materialized contract", directory=True)
    entries = sorted(path.name for path in root.iterdir())
    if entries != ["object_registry.json", "pose_quality_spec.json"]:
        raise WatcherContractError("reset-only materializer output files differ")
    registry_path = _inside(root, root / entries[0], role="schema6 object registry")
    spec_path = _inside(root, root / entries[1], role="schema6 pose-quality spec")
    registry = load_json(registry_path, role="schema6 object registry")
    spec = load_json(spec_path, role="schema6 pose-quality spec")
    objects = registry.get("objects")
    if (
        registry.get("format") != "etsf_schema6_object_registry_v1"
        or not isinstance(objects, list)
        or [item.get("name") for item in objects if isinstance(item, Mapping)] != ["can", "pot"]
        or any(
            not str(item.get("stable_sim_actor_id", "")).startswith(
                f"task_attr={name};sapien_actor_name="
            )
            for item, name in zip(objects, ("can", "pot"), strict=True)
        )
        or not str(objects[0].get("asset_model_id", "")).startswith("105_sauce-can/base")
        or not str(objects[1].get("asset_model_id", "")).startswith("060_kitchenpot/base")
    ):
        raise WatcherContractError("reset-only object registry contract is invalid")
    registry_sha = canonical_sha256(registry)
    if (
        spec.get("format") != "etsf_schema6_pose_quality_spec_v1"
        or spec.get("schema_version") != 6
        or spec.get("object_registry_sha256") != registry_sha
        or spec.get("threshold_basis", {}).get("thresholds_fit_from_pose_data") is not False
        or spec.get("threshold_basis", {}).get("frozen_before_collection") is not True
    ):
        raise WatcherContractError("reset-only pose-quality contract is invalid")
    result = _last_json_object(log_path)
    if (
        result.get("status") != "materialized_reset_only_schema6_contract"
        or result.get("requested_seed") != FIXED_REQUESTED_SEED
        or result.get("resolved_seed") != FIXED_REQUESTED_SEED
        or result.get("environment_steps") != 0
        or result.get("policy_imported_or_forwarded") is not False
        or result.get("trajectory_or_labels_read") is not False
        or result.get("fresh_inputs_used") is not False
        or result.get("object_registry", {}).get("path") != str(registry_path)
        or result.get("object_registry", {}).get("file_sha256") != file_sha256(registry_path)
        or result.get("object_registry", {}).get("logical_sha256") != registry_sha
        or result.get("pose_quality_spec", {}).get("path") != str(spec_path)
        or result.get("pose_quality_spec", {}).get("file_sha256") != file_sha256(spec_path)
        or result.get("pose_quality_spec", {}).get("logical_sha256") != canonical_sha256(spec)
    ):
        raise WatcherContractError("materializer stdout/result contract is invalid")
    _require_read_only(root, "reset-only materializer root")
    audit = {
        "status": "verified_reset_only_materialization",
        "environment_steps": 0,
        "policy_imported_or_forwarded": False,
        "trajectory_or_labels_read": False,
        "object_registry_sha256": file_sha256(registry_path),
        "object_registry_logical_sha256": registry_sha,
        "pose_quality_spec_sha256": file_sha256(spec_path),
        "pose_quality_spec_logical_sha256": canonical_sha256(spec),
        "hdf5_opened": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def validate_authority_output(plan: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    output = Path(str(plan["output_root"]))
    authority_path = _inside(output, output / "collection_authority.json", role="schema6 authority")
    authority = load_json(authority_path, role="schema6 authority")
    logical = _validate_logical_sha(authority, "authority_sha256", "schema6 authority")
    scope = authority.get("scope", {})
    capability = authority.get("capability_contract", {})
    expected_reset = output / "reset_contract"
    if (
        authority.get("format") != AUTHORITY_FORMAT
        or authority.get("status") != AUTHORITY_STATUS
        or scope.get("requested_seed") != FIXED_REQUESTED_SEED
        or scope.get("expected_resolved_seed") != FIXED_REQUESTED_SEED
        or scope.get("seed_count") != 1
        or scope.get("candidate_indices") != [0, 1, 2, 3]
        or scope.get("root_action_horizon") != 1
        or scope.get("continuation_action_horizon") != 1
        or scope.get("max_episode_steps") != plan["max_episode_steps"]
        or authority.get("output_contract", {}).get("directory")
        != str(output / "schema6_collection")
        or authority.get("input_artifacts", {}).get("object_registry", {}).get("path")
        != str(expected_reset / "object_registry.json")
        or authority.get("input_artifacts", {}).get("pose_quality_spec", {}).get("path")
        != str(expected_reset / "pose_quality_spec.json")
        or authority.get("input_artifacts", {}).get("event_spec", {}).get("path")
        != str(plan["event_spec"])
        or capability.get("fresh_inputs_allowed") is not False
        or capability.get("fresh_trajectory_or_label_opened") is not False
        or capability.get("performance_evaluation_authorized") is not False
        or capability.get("transfer_claim_authorized") is not False
    ):
        raise WatcherContractError("frozen schema6 authority scope/capability is invalid")
    result = _last_json_object(log_path)
    if (
        result.get("status") != AUTHORITY_STATUS
        or result.get("path") != str(authority_path)
        or result.get("file_sha256") != file_sha256(authority_path)
        or result.get("authority_sha256") != logical
    ):
        raise WatcherContractError("authority freezer stdout/result contract is invalid")
    _require_read_only(authority_path, "schema6 authority")
    audit = {
        "status": "verified_frozen_one_seed_h1_authority",
        "authority_path": str(authority_path),
        "authority_file_sha256": file_sha256(authority_path),
        "authority_logical_sha256": logical,
        "seed_count": 1,
        "requested_seed": FIXED_REQUESTED_SEED,
        "root_action_horizon": 1,
        "continuation_action_horizon": 1,
        "max_episode_steps": plan["max_episode_steps"],
        "fresh_inputs_allowed": False,
        "test_hdf5_opened": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def validate_collection_output(
    plan: Mapping[str, Any], log_path: Path, returncode: int
) -> dict[str, Any]:
    output = Path(str(plan["output_root"]))
    collection = resolve_existing(
        output / "schema6_collection", role="schema6 collection output", directory=True
    )
    receipt_path = _inside(
        collection, collection / "collection_receipt.json", role="schema6 collection receipt"
    )
    manifest_path = _inside(
        collection, collection / "manifest.json", role="schema6 collection manifest"
    )
    receipt = load_json(receipt_path, role="schema6 collection receipt")
    manifest = load_json(manifest_path, role="schema6 collection manifest")
    logical = _validate_logical_sha(
        receipt, "receipt_logical_sha256", "schema6 collection receipt"
    )
    expected_status = COLLECTION_SUCCESS_STATUS if returncode == 0 else COLLECTION_EMPTY_STATUS
    expected_groups = 1 if returncode == 0 else 0
    authority_path = output / "collection_authority.json"
    authority = load_json(authority_path, role="schema6 authority after collection")
    if (
        returncode not in (0, 20)
        or receipt.get("format") != COLLECTION_RECEIPT_FORMAT
        or receipt.get("status") != expected_status
        or receipt.get("exit_code") != returncode
        or receipt.get("authority", {}).get("path") != str(authority_path.resolve())
        or receipt.get("authority", {}).get("file_sha256") != file_sha256(authority_path)
        or receipt.get("authority", {}).get("logical_sha256") != authority.get("authority_sha256")
        or receipt.get("manifest", {}).get("path") != str(manifest_path)
        or receipt.get("manifest", {}).get("file_sha256") != file_sha256(manifest_path)
        or receipt.get("fresh_inputs_used") is not False
        or receipt.get("real_robot_execution") is not False
        or receipt.get("task_success_claimed") is not False
        or receipt.get("performance_evaluation_authorized") is not False
        or receipt.get("transfer_claim_authorized") is not False
        or receipt.get("failure") is not None
        or manifest.get("format") != COLLECTION_MANIFEST_FORMAT
        or manifest.get("status") != expected_status
        or manifest.get("requested_seeds") != [FIXED_REQUESTED_SEED]
        or manifest.get("completed_groups") != expected_groups
        or manifest.get("fresh_inputs_used") is not False
        or manifest.get("task_success_claimed") is not False
        or manifest.get("transfer_claim_authorized") is not False
    ):
        raise WatcherContractError("schema6 collection receipt/manifest contract is invalid")
    group_sha = None
    if returncode == 0:
        group = receipt.get("group")
        if not isinstance(group, Mapping) or not _is_sha256(group.get("file_sha256")):
            raise WatcherContractError("schema6 collected group record is invalid")
        group_path = _inside(
            collection, Path(str(group.get("path", ""))), role="schema6 development group"
        )
        # Development data may be byte-hashed; no source/test HDF is ever opened.
        digest = hashlib.sha256()
        with group_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        group_sha = digest.hexdigest()
        if group_sha != group["file_sha256"] or manifest.get("group") != group:
            raise WatcherContractError("schema6 development group SHA changed")
    elif receipt.get("group") is not None or manifest.get("group") is not None:
        raise WatcherContractError("root-insufficient collection unexpectedly has a group")
    result = _last_json_object(log_path)
    if (
        result.get("exit_code") != returncode
        or result.get("status") != expected_status
        or result.get("receipt_file_sha256") != file_sha256(receipt_path)
        or result.get("receipt_logical_sha256") != logical
        or result.get("manifest_file_sha256") != file_sha256(manifest_path)
    ):
        raise WatcherContractError("collection launcher stdout/result contract is invalid")
    audit = {
        "status": "verified_one_seed_h1_schema6_collection_attempt",
        "collection_status": expected_status,
        "returncode": returncode,
        "completed_groups": expected_groups,
        "receipt_sha256": file_sha256(receipt_path),
        "receipt_logical_sha256": logical,
        "manifest_sha256": file_sha256(manifest_path),
        "development_group_sha256": group_sha,
        "source_hdf5_opened": 0,
        "test_hdf5_opened": 0,
        "performance_claim_authorized": False,
        "transfer_claim_authorized": False,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def run_stage(
    stage: Mapping[str, Any],
    *,
    output_root: Path,
    environment: Mapping[str, str],
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    environment_audit_sha256: str,
    validator: Callable[[Path, int], dict[str, Any]],
    lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(stage["stage"])
    log_path = output_root / "logs" / f"{name}.log"
    exit_path = output_root / "stage_exits" / f"{name}.exit"
    receipt_path = output_root / "stage_receipts" / f"{name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    exit_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or exit_path.exists() or receipt_path.exists():
        raise FileExistsError(f"stage output already exists: {name}")
    argv = list(stage["argv"])
    if not _is_sha256(environment_audit_sha256):
        raise WatcherContractError("stage lacks an isolated-environment audit SHA")
    if _argv_sha(argv) != stage.get("argv_sha256"):
        raise WatcherContractError(f"stage argv changed: {name}")
    started_unix = time.time()
    started_monotonic = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    stage_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    returncode: int | None = None
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
        "status": "launching",
        "pid": None,
        "process_group_id": None,
        "process_group_isolated": False,
        "started_unix": started_unix,
        "argv": argv,
        "argv_sha256": stage["argv_sha256"],
        "accepted_returncodes": list(stage["accepted_returncodes"]),
        "environment_audit_sha256": environment_audit_sha256,
        "log": str(log_path),
        "run_exit": str(exit_path),
    }
    with log_path.open("x", encoding="utf-8") as log_handle:
        try:
            lifecycle["popen_attempted"] = True
            process = subprocess.Popen(
                argv,
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
            running.update(
                {
                    "status": "auditing_process_group",
                    "pid": process.pid,
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
            running.update(
                {
                    "status": "running",
                    "process_group_id": process_group_id,
                    "process_group_isolated": process_group_isolated,
                }
            )
            if not process_group_isolated:
                raise WatcherContractError(
                    f"schema6 stage did not enter its own process group: {name}"
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
                if (
                    timeout_seconds > 0
                    and time.monotonic() - started_monotonic >= timeout_seconds
                ):
                    raise TimeoutError(f"stage timed out: {name}")
                state["last_heartbeat_unix"] = time.time()
                state["stage_elapsed_seconds"] = (
                    time.monotonic() - started_monotonic
                )
                state["stage_log_bytes"] = log_path.stat().st_size
                atomic_json(state_path, state)
                time.sleep(poll_seconds)
            returncode = process.returncode
        except BaseException as error:
            if stage_error is None:
                stage_error = error
        finally:
            if process is not None:
                try:
                    if stage_error is not None and process.poll() is None:
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
                    returncode = process.returncode
                except BaseException as error:
                    cleanup_error = error
                process_reaped = isinstance(process.returncode, int)
                process_group_reaped = False
                if process_group_isolated and process_group_id is not None:
                    try:
                        if _process_group_exists(process_group_id):
                            if stage_error is None:
                                stage_error = WatcherContractError(
                                    f"schema6 stage left live descendant processes: {name}"
                                )
                            if not _terminate_process_group(process_group_id):
                                raise WatcherContractError(
                                    f"schema6 stage process group could not be reaped: {name}"
                                )
                        process_group_reaped = not _process_group_exists(
                            process_group_id
                        )
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
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
                    cleanup_error = WatcherContractError(
                        f"schema6 stage process tree could not be proven reaped: {name}"
                    )
                if cleanup_error is not None and stage_error is None:
                    stage_error = cleanup_error
    log_path.chmod(0o444)
    recorded_returncode = 127 if returncode is None else int(returncode)
    immutable_text_new(exit_path, f"{recorded_returncode}\n")
    result: dict[str, Any] = {
        **running,
        "status": "subprocess_complete_pending_artifact_audit",
        "returncode": recorded_returncode,
        "process_reaped": lifecycle.get("process_reaped") is True,
        "process_group_id": process_group_id,
        "process_group_isolated": process_group_isolated,
        "process_group_reaped": lifecycle.get("process_group_reaped") is True,
        "finished_unix": time.time(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "log_sha256": file_sha256(log_path),
        "log_bytes": log_path.stat().st_size,
        "run_exit_sha256": file_sha256(exit_path),
    }
    if stage_error is None and recorded_returncode in stage["accepted_returncodes"]:
        try:
            result["artifact_audit"] = validator(log_path, recorded_returncode)
            result["status"] = "complete_verified"
        except BaseException as error:
            stage_error = error
    if stage_error is None and recorded_returncode not in stage["accepted_returncodes"]:
        stage_error = WatcherContractError(
            f"stage {name} failed with exit {recorded_returncode}; see {log_path}"
        )
    if stage_error is not None:
        result.update(
            {
                "status": "failed_closed",
                "error_type": type(stage_error).__name__,
                "error": str(stage_error),
            }
        )
    if cleanup_error is not None:
        result.update(
            {
                "cleanup_error_type": type(cleanup_error).__name__,
                "cleanup_error": str(cleanup_error),
            }
        )
    result["receipt_payload_sha256"] = canonical_sha256(result)
    atomic_json(receipt_path, result)
    if stage_error is not None:
        raise stage_error
    return result


def _validate_event_spec(path: Path) -> None:
    value = load_json(path, role="schema6 event specification")
    chain = value.get("chains", {}).get("move_can_pot", {})
    calibration = value.get("calibration", {}).get("move_can_pot", {})
    if (
        chain.get("merge_e1_e2") is not True
        or chain.get("chain") != ["e0", "e12", "e3", "e4", "eK"]
        or calibration.get("moving") != "can"
        or calibration.get("anchor") not in (None, "", "pot")
    ):
        raise WatcherContractError("event specification lacks canonical move_can_pot events")


def static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    lobo = resolve_future_directory(args.lobo_root, role="LOBO autonomous root")
    code = resolve_existing(args.code_root, role="schema6 immutable code root", directory=True)
    designated_lobo = reject_path_text(DESIGNATED_LOBO_ROOT, "designated LOBO root")
    designated_code = reject_path_text(DESIGNATED_CODE_ROOT, "designated code root")
    if lobo != designated_lobo and (
        not lobo.exists() or lobo != designated_lobo.resolve(strict=True)
    ):
        raise WatcherContractError("LOBO root differs from the designated aggregate run")
    if code != designated_code.resolve(strict=True):
        raise WatcherContractError("code root differs from the deployed immutable r6j code")
    r6f = resolve_existing(args.r6f_preregistration, role="R6f preregistration", directory=False)
    event_spec = resolve_existing(args.event_spec, role="schema6 event specification", directory=False)
    output = resolve_new(args.output, role="autonomous watcher output")
    python = python_contract(args.python_bin)
    if Path(python["invocation_path"]) != reject_path_text(DESIGNATED_PYTHON, "designated Python"):
        raise WatcherContractError("Python executable differs from the designated SmolVLA venv")
    _require_read_only(r6f, "R6f preregistration")
    _require_read_only(event_spec, "schema6 event specification")
    _validate_event_spec(event_spec)
    r6f_document, r6f_lineage_projection = _load_signed_r6f_lineage_metadata(r6f)
    if r6f_document.get("status") != "preregistered_R6f_feasibility_simulation_only_not_executed":
        raise WatcherContractError("R6f preregistration status changed")
    implementations = implementation_closure(code)
    watcher_path = resolve_existing(Path(__file__), role="autonomous watcher implementation", directory=False)
    lock = reject_path_text(
        args.gpu_lock
        or Path(f"/tmp/etsf_smolvla_piper_schema6_autonomous_gpu{args.gpu_index}.lock"),
        "GPU pipeline lock",
    )
    resolve_existing(lock.parent, role="GPU pipeline lock parent", directory=True)
    plan: dict[str, Any] = {
        "format": FORMAT,
        "status": "static_preflight_complete_lobo_summary_not_read",
        "lobo_root": str(lobo),
        "lobo_terminal_receipt": str(lobo / "final_receipt.json"),
        "lobo_summary_read_during_static_preflight": False,
        "lobo_watcher_hdf5_opened_during_static_preflight": 0,
        "lobo_test_hdf5_opened_during_static_preflight": 0,
        "code_root": str(code),
        "implementation_files": implementations,
        "implementation_bundle_sha256": canonical_sha256(implementations),
        "watcher_implementation": {
            "path": str(watcher_path),
            "sha256": file_sha256(watcher_path),
            "size": watcher_path.stat().st_size,
        },
        "r6f_preregistration": str(r6f),
        "r6f_preregistration_sha256": file_sha256(r6f),
        "r6f_lineage_projection_audit": r6f_lineage_projection,
        "event_spec": str(event_spec),
        "event_spec_sha256": file_sha256(event_spec),
        "output_root": str(output),
        "python_contract": python,
        "subprocess_environment_contract": {
            "format": "etsf_isolated_python_subprocess_environment_v1",
            "scrubbed_names": sorted(SCRUBBED_PYTHON_ENVIRONMENT),
            "forced_python_environment": dict(FORCED_PYTHON_ENVIRONMENT),
            "pythonpath_inherited": False,
            "pythonhome_inherited": False,
            "virtualenv_or_conda_metadata_inherited": False,
        },
        "gpu_index": args.gpu_index,
        "expected_gpu_name_fragment": EXPECTED_GPU_FRAGMENT,
        "gpu_lock": str(lock),
        "max_episode_steps": args.max_episode_steps,
        "fixed_requested_seed": FIXED_REQUESTED_SEED,
        "seed_count": 1,
        "root_action_horizon": 1,
        "continuation_action_horizon": 1,
        "fresh_paths_accepted": False,
        "lobo_summary_read": False,
        "lobo_watcher_hdf5_opened": 0,
        "lobo_test_hdf5_opened": 0,
        "test_labels_read": False,
        "nonresumable_output": True,
    }
    plan["commands"] = build_commands(plan)
    plan["execution_order"] = [row["stage"] for row in plan["commands"]]
    plan["static_plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_static_bindings(plan: Mapping[str, Any]) -> None:
    implementations = implementation_closure(Path(str(plan["code_root"])))
    if (
        implementations != plan.get("implementation_files")
        or canonical_sha256(implementations) != plan.get("implementation_bundle_sha256")
    ):
        raise WatcherContractError("schema6 implementation changed while watcher ran")
    watcher = plan.get("watcher_implementation", {})
    watcher_path = resolve_existing(
        Path(str(watcher.get("path", ""))), role="autonomous watcher implementation", directory=False
    )
    if file_sha256(watcher_path) != watcher.get("sha256"):
        raise WatcherContractError("autonomous watcher implementation changed while running")
    for path_field, sha_field, role in (
        ("r6f_preregistration", "r6f_preregistration_sha256", "R6f preregistration"),
        ("event_spec", "event_spec_sha256", "schema6 event specification"),
    ):
        path = resolve_existing(Path(str(plan[path_field])), role=role, directory=False)
        if file_sha256(path) != plan[sha_field]:
            raise WatcherContractError(f"{role} changed while watcher ran")
    if python_contract(Path(str(plan["python_contract"]["invocation_path"]))) != plan[
        "python_contract"
    ]:
        raise WatcherContractError("Python executable changed while watcher ran")


def _development_inventory(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        reject_path_text(path, "watcher output artifact")
        if path.is_symlink():
            raise WatcherContractError("watcher output contains a symlink")
        if not path.is_file() or path.name in {
            "launch_state.json",
            "final_receipt.json",
            "failure_receipt.json",
            "artifact_inventory.json",
        }:
            continue
        if path.suffix.lower() in HDF_SUFFIXES:
            # This is the newly collected development group, never source/test.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            digest = file_sha256(path)
        records.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    value: dict[str, Any] = {
        "format": FORMAT,
        "status": "complete_pre_freeze_inventory",
        "file_count": len(records),
        "files": records,
        "source_hdf5_opened": 0,
        "test_hdf5_opened": 0,
    }
    value["inventory_sha256"] = canonical_sha256(value)
    return value


def freeze_tree(
    root: Path, *, exclude_root_files: Sequence[str] = ()
) -> None:
    excluded = set(exclude_root_files)
    for path in sorted(root.rglob("*"), reverse=True):
        reject_path_text(path, "watcher output freeze target")
        if path.is_symlink():
            raise WatcherContractError("cannot freeze a symlink artifact")
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
        raise WatcherContractError(
            "schema6 output root was not frozen before terminal publication"
        )
    observed_hidden: set[Path] = set()
    for path in root.rglob("*"):
        reject_path_text(path, "frozen schema6 output artifact")
        if path.is_symlink():
            raise WatcherContractError("frozen schema6 output contains a symlink")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path in expected_hidden:
            observed_hidden.add(path)
            if not path.is_file() or mode != 0:
                raise WatcherContractError(
                    "schema6 terminal became readable before publication"
                )
        elif path.is_file() and mode != 0o444:
            raise WatcherContractError(
                f"frozen schema6 output file mode changed: {path}"
            )
        elif path.is_dir() and mode != 0o555:
            raise WatcherContractError(
                f"frozen schema6 output directory mode changed: {path}"
            )
    if observed_hidden != expected_hidden:
        raise WatcherContractError(
            "one or more hidden schema6 terminal files disappeared"
        )


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


def _validate_schema6_stage_process_proof(
    *,
    stage: str,
    result: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> None:
    accepted = SCHEMA6_STAGE_ACCEPTED_RETURNCODES.get(stage)
    process_pid = result.get("pid")
    process_group_id = result.get("process_group_id")
    returncode = result.get("returncode")
    lifecycle_pid = lifecycle.get("process_pid")
    lifecycle_group_id = lifecycle.get("process_group_id")
    lifecycle_returncode = lifecycle.get("returncode")
    unsigned_result = dict(result)
    payload_sha256 = unsigned_result.pop("receipt_payload_sha256", None)
    if (
        accepted is None
        or result.get("format") != FORMAT
        or result.get("stage") != stage
        or result.get("status") != "complete_verified"
        or result.get("accepted_returncodes") != list(accepted)
        or isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid <= 0
        or isinstance(returncode, bool)
        or not isinstance(returncode, int)
        or returncode not in accepted
        or result.get("process_reaped") is not True
        or isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id != process_pid
        or result.get("process_group_isolated") is not True
        or result.get("process_group_reaped") is not True
        or not isinstance(result.get("artifact_audit"), Mapping)
        or payload_sha256 != canonical_sha256(unsigned_result)
        or lifecycle.get("stage") != stage
        or lifecycle.get("popen_attempted") is not True
        or lifecycle.get("popen_reached") is not True
        or isinstance(lifecycle_pid, bool)
        or not isinstance(lifecycle_pid, int)
        or lifecycle_pid != process_pid
        or lifecycle.get("process_reaped") is not True
        or isinstance(lifecycle_group_id, bool)
        or not isinstance(lifecycle_group_id, int)
        or lifecycle_group_id != process_pid
        or lifecycle.get("process_group_isolated") is not True
        or lifecycle.get("process_group_reaped") is not True
        or isinstance(lifecycle_returncode, bool)
        or not isinstance(lifecycle_returncode, int)
        or lifecycle_returncode != returncode
    ):
        raise WatcherContractError(
            f"schema6 stage process lifecycle proof is inconsistent: {stage}"
        )


def validate_schema6_success_terminal_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = dict(receipt)
    logical_sha256 = unsigned.pop("receipt_sha256", None)
    order = receipt.get("execution_order")
    results = receipt.get("stage_results")
    lifecycles = receipt.get("stage_lifecycles")
    returncodes = receipt.get("stage_returncodes")
    payload_hashes = receipt.get("stage_receipt_payload_sha256")
    lobo_gate = receipt.get("lobo_gate")
    native_binding = receipt.get("native_source_binding_audit")
    deployment = receipt.get("deployment_rerank_checkpoint")
    source_plan_binding = (
        native_binding.get("source_launch_plan")
        if isinstance(native_binding, Mapping)
        else None
    )
    lobo_source_binding = (
        lobo_gate.get("source_binding_receipt")
        if isinstance(lobo_gate, Mapping)
        else None
    )
    lobo_stage_audits = (
        lobo_gate.get("stage_audits")
        if isinstance(lobo_gate, Mapping)
        else None
    )
    lobo_gate_unsigned = dict(lobo_gate) if isinstance(lobo_gate, Mapping) else {}
    lobo_gate_logical = lobo_gate_unsigned.pop("summary_sha256", None)
    if (
        logical_sha256 != canonical_sha256(unsigned)
        or receipt.get("format") != FORMAT
        or receipt.get("status") != TERMINAL_STATUS
        or order != list(SCHEMA6_EXECUTION_ORDER)
        or not isinstance(results, Mapping)
        or set(results) != set(SCHEMA6_EXECUTION_ORDER)
        or not isinstance(lifecycles, list)
        or len(lifecycles) != len(SCHEMA6_EXECUTION_ORDER)
        or not isinstance(returncodes, Mapping)
        or set(returncodes) != set(SCHEMA6_EXECUTION_ORDER)
        or not isinstance(payload_hashes, Mapping)
        or set(payload_hashes) != set(SCHEMA6_EXECUTION_ORDER)
        or receipt.get("lobo_checkpoints_rerank_authorized") is not False
        or receipt.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
        or receipt.get("lobo_watcher_hdf5_opened") != 0
        or receipt.get("lobo_test_hdf5_opened") != 0
        or receipt.get("test_labels_read") is not False
        or receipt.get("fresh_paths_accepted") is not False
        or receipt.get("real_robot_execution") is not False
        or receipt.get("performance_or_transfer_claim_authorized") is not False
        or receipt.get("artifacts_frozen_read_only") is not True
        or not _is_sha256(receipt.get("static_plan_sha256"))
        or not _is_sha256(receipt.get("artifact_inventory_sha256"))
        or not isinstance(lobo_gate, Mapping)
        or lobo_gate_logical != canonical_sha256(lobo_gate_unsigned)
        or lobo_gate.get("status")
        != "verified_piper_then_ur5_lobo_terminal_exit_zero"
        or lobo_gate.get("lobo_root") != str(DESIGNATED_LOBO_ROOT)
        or lobo_gate.get("lobo_launcher_sha256")
        != EXPECTED_LOBO_LAUNCHER_SHA256
        or lobo_gate.get("static_plan_sha256")
        != EXPECTED_LOBO_STATIC_PLAN_SHA256
        or not _is_sha256(lobo_gate.get("final_receipt_sha256"))
        or not _is_sha256(lobo_gate.get("final_receipt_logical_sha256"))
        or not _is_sha256(lobo_gate.get("watcher_run_exit_sha256"))
        or lobo_gate.get("execution_order")
        != [stage for stage, _body in LOBO_STAGES]
        or lobo_gate.get("lobo_checkpoints_rerank_authorized") is not False
        or lobo_gate.get("deployment_rerank_authority")
        != "native_source_ensemble_only"
        or lobo_gate.get("watcher_hdf5_opened") != 0
        or lobo_gate.get("test_hdf5_opened_by_watcher") != 0
        or lobo_gate.get("test_labels_read_by_watcher") is not False
        or lobo_gate.get("target_unused_train_payload_opened") != 0
        or lobo_gate.get("test_group_hdf5_opened") != 0
        or receipt.get("lobo_audit_sha256") != lobo_gate_logical
        or receipt.get("transitive_source63_audit_sha256")
        != lobo_gate.get("transitive_source63_audit_sha256")
        or not _is_sha256(receipt.get("transitive_source63_audit_sha256"))
        or not isinstance(native_binding, Mapping)
        or native_binding != lobo_gate.get("native_source_binding_audit")
        or receipt.get("native_source_binding_audit_sha256")
        != canonical_sha256(native_binding)
        or receipt.get("native_source_binding_audit_sha256")
        != lobo_gate.get("native_source_binding_audit_sha256")
        or not isinstance(lobo_source_binding, Mapping)
        or lobo_source_binding.get("path")
        != native_binding.get("source_binding_receipt_path")
        or lobo_source_binding.get("file_sha256")
        != native_binding.get("source_binding_receipt_file_sha256")
        or lobo_source_binding.get("binding_sha256")
        != native_binding.get("source_binding_sha256")
        or lobo_source_binding.get("deployment_rerank_checkpoint")
        != deployment
        or lobo_source_binding.get(
            "policy_feature_action_bridge_contract_sha256"
        )
        != native_binding.get("policy_feature_action_bridge_sha256")
        or lobo_source_binding.get("lobo_checkpoints_rerank_authorized")
        is not False
        or not _is_sha256(
            lobo_source_binding.get("source_final_receipt_file_sha256")
        )
        or not _is_sha256(
            lobo_source_binding.get("source_final_receipt_logical_sha256")
        )
        or native_binding.get("source_training_root")
        != str(EXPECTED_SOURCE_ROOT)
        or not isinstance(source_plan_binding, Mapping)
        or source_plan_binding.get("path")
        != str(EXPECTED_SOURCE_ROOT / "launch_plan.json")
        or source_plan_binding.get("file_sha256")
        != EXPECTED_SOURCE_PLAN_SHA256
        or source_plan_binding.get("logical_sha256")
        != EXPECTED_SOURCE_STATIC_PLAN_SHA256
        or native_binding.get("source_launcher_sha256")
        != EXPECTED_SOURCE_LAUNCHER_SHA256
        or native_binding.get("source_implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
        or native_binding.get("lobo_checkpoints_rerank_authorized") is not False
        or not _is_sha256(
            native_binding.get("source_binding_receipt_file_sha256")
        )
        or not _is_sha256(native_binding.get("source_binding_sha256"))
        or not _is_sha256(
            native_binding.get("policy_feature_action_bridge_sha256")
        )
        or not isinstance(deployment, Mapping)
        or deployment != lobo_gate.get("deployment_rerank_checkpoint")
        or deployment != native_binding.get("deployment_rerank_checkpoint")
        or deployment.get("path")
        != str(EXPECTED_SOURCE_ROOT / EXPECTED_SOURCE_ENSEMBLE_RELATIVE)
        or not _is_sha256(deployment.get("sha256"))
        or deployment.get("policy") != "smolvla"
        or deployment.get("checkpoint_family")
        != "smolvla_native_event_world_model"
        or deployment.get("policy_feature_action_bridge_contract_sha256")
        != native_binding.get("policy_feature_action_bridge_sha256")
        or deployment.get("source_native_checkpoint") is not True
        or not isinstance(lobo_stage_audits, Mapping)
        or set(lobo_stage_audits) != {stage for stage, _body in LOBO_STAGES}
    ):
        raise WatcherContractError(
            "schema6 success terminal receipt semantics are invalid"
        )
    for stage, body in LOBO_STAGES:
        audit = lobo_stage_audits.get(stage)
        if (
            not isinstance(audit, Mapping)
            or audit.get("held_out_body") != body
            or isinstance(audit.get("returncode"), bool)
            or not isinstance(audit.get("returncode"), int)
            or audit.get("returncode") != 0
            or any(
                not _is_sha256(audit.get(field))
                for field in (
                    "stage_receipt_sha256",
                    "run_exit_sha256",
                    "log_sha256",
                    "summary_sha256",
                    "artifact_inventory_sha256",
                )
            )
        ):
            raise WatcherContractError(
                f"schema6 terminal LOBO stage audit is invalid: {stage}"
            )
    for expected_stage, lifecycle in zip(SCHEMA6_EXECUTION_ORDER, lifecycles):
        result = results.get(expected_stage)
        if not isinstance(result, Mapping) or not isinstance(lifecycle, Mapping):
            raise WatcherContractError("schema6 terminal stage proof is invalid")
        _validate_schema6_stage_process_proof(
            stage=expected_stage, result=result, lifecycle=lifecycle
        )
        if (
            isinstance(returncodes.get(expected_stage), bool)
            or not isinstance(returncodes.get(expected_stage), int)
            or returncodes.get(expected_stage) != result.get("returncode")
            or not _is_sha256(payload_hashes.get(expected_stage))
            or payload_hashes.get(expected_stage)
            != result.get("receipt_payload_sha256")
        ):
            raise WatcherContractError(
                "schema6 terminal stage summary differs from stage proof"
            )
    collection_result = results[SCHEMA6_EXECUTION_ORDER[-1]]
    if receipt.get("collection_audit") != collection_result.get("artifact_audit"):
        raise WatcherContractError(
            "schema6 terminal collection audit differs from stage proof"
        )
    return dict(receipt)


def publish_frozen_terminal_receipt(
    root: Path,
    *,
    terminal_name: str,
    receipt: Mapping[str, Any],
    exit_code: int,
) -> None:
    expected_exit_code = {
        "final_receipt.json": 0,
        "failure_receipt.json": 1,
    }.get(terminal_name)
    if (
        expected_exit_code is None
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code != expected_exit_code
    ):
        raise ValueError("schema6 terminal name and exit code are inconsistent")
    if receipt.get("artifacts_frozen_read_only") is not True:
        raise WatcherContractError(
            "published schema6 terminal must claim a frozen tree"
        )
    if terminal_name == "final_receipt.json":
        validate_schema6_success_terminal_receipt(receipt)
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
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
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
            root, exclude_root_files=(terminal_path.name, exit_path.name)
        )
        _verify_frozen_tree_before_terminal_publish(
            root, hidden_terminals=(terminal_path, exit_path)
        )
        _fsync_directory(root)
        exit_path.chmod(0o444)
        if stat.S_IMODE(exit_path.stat().st_mode) != 0o444:
            raise WatcherContractError(
                "schema6 run.exit publication mode is invalid"
            )
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


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = static_preflight(args)
    if plan.get("execution_order") != list(SCHEMA6_EXECUTION_ORDER):
        raise WatcherContractError("schema6 execution order differs from protocol")
    output = Path(str(plan["output_root"]))
    output.mkdir(mode=0o700)
    state_path = output / "launch_state.json"
    plan_path = output / "launch_plan.json"
    claim_token = canonical_sha256(
        {"pid": os.getpid(), "plan": plan["static_plan_sha256"], "time": time.time_ns()}
    )
    claim_payload = {
        "format": FORMAT,
        "pid": os.getpid(),
        "token": claim_token,
        "static_plan_sha256": plan["static_plan_sha256"],
    }
    immutable_json_new(plan_path, plan)
    acquire_lock(output / "launch.lock", claim_payload)
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "starting",
        "pid": os.getpid(),
        "static_plan_sha256": plan["static_plan_sha256"],
        "execution_order": list(plan["execution_order"]),
        "stage_results": {},
        "stage_lifecycles": [],
        "stages_started": [],
        "lobo_summary_read": False,
        "lobo_watcher_hdf5_opened": 0,
        "lobo_test_hdf5_opened": 0,
        "test_labels_read": False,
        "fresh_paths_accepted": False,
    }
    atomic_json(state_path, state)
    lock_path = Path(str(plan["gpu_lock"]))
    gpu_token = canonical_sha256(
        {"pid": os.getpid(), "plan": plan["static_plan_sha256"], "scope": "gpu"}
    )
    gpu_lock_payload = {
        "format": FORMAT,
        "pid": os.getpid(),
        "token": gpu_token,
        "static_plan_sha256": plan["static_plan_sha256"],
    }
    gpu_lock_acquired = False
    stage_lifecycles: list[dict[str, Any]] = []
    state["stage_lifecycles"] = stage_lifecycles
    try:
        lobo_audit = wait_for_lobo_completion(
            Path(str(plan["lobo_root"])),
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.lobo_timeout_seconds,
        )
        state["lobo_audit"] = lobo_audit
        state["lobo_summary_read"] = True
        state["status"] = "lobo_verified_waiting_for_idle_4090"
        atomic_json(state_path, state)
        acquire_lock(lock_path, gpu_lock_payload)
        gpu_lock_acquired = True
        state["gpu_lock_acquired"] = True
        atomic_json(state_path, state)
        verify_static_bindings(plan)
        first_idle = wait_for_idle_4090(
            gpu_index=args.gpu_index,
            state=state,
            state_path=state_path,
            phase="reset_only_materialization",
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.gpu_timeout_seconds,
        )
        state["gpu_idle_before_materializer"] = first_idle
        atomic_json(state_path, state)

        environment, environment_audit = isolated_stage_environment(
            os.environ, gpu_index=args.gpu_index, omp_threads=args.omp_threads
        )
        state["subprocess_environment_audit"] = environment_audit
        atomic_json(state_path, state)
        commands = list(plan["commands"])
        validators: list[Callable[[Path, int], dict[str, Any]]] = [
            lambda log, _code: validate_materializer_output(plan, log),
            lambda log, _code: validate_authority_output(plan, log),
            lambda log, code: validate_collection_output(plan, log, code),
        ]
        timeouts = [
            args.materializer_timeout_seconds,
            args.freezer_timeout_seconds,
            args.collection_timeout_seconds,
        ]
        for index in (0, 1):
            verify_static_bindings(plan)
            lifecycle: dict[str, Any] = {"stage": commands[index]["stage"]}
            stage_lifecycles.append(lifecycle)
            state["stages_started"].append(commands[index]["stage"])
            atomic_json(state_path, state)
            result = run_stage(
                commands[index],
                output_root=output,
                environment=environment,
                state=state,
                state_path=state_path,
                poll_seconds=args.poll_seconds,
                timeout_seconds=timeouts[index],
                environment_audit_sha256=environment_audit["audit_sha256"],
                validator=validators[index],
                lifecycle=lifecycle,
            )
            state["stage_results"][commands[index]["stage"]] = result
            state["current_stage"] = None
            atomic_json(state_path, state)

        verify_static_bindings(plan)
        second_idle = wait_for_idle_4090(
            gpu_index=args.gpu_index,
            state=state,
            state_path=state_path,
            phase="one_seed_h1_collection",
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.gpu_timeout_seconds,
        )
        state["gpu_idle_before_collection"] = second_idle
        atomic_json(state_path, state)
        lifecycle = {"stage": commands[2]["stage"]}
        stage_lifecycles.append(lifecycle)
        state["stages_started"].append(commands[2]["stage"])
        atomic_json(state_path, state)
        result = run_stage(
            commands[2],
            output_root=output,
            environment=environment,
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            timeout_seconds=timeouts[2],
            environment_audit_sha256=environment_audit["audit_sha256"],
            validator=validators[2],
            lifecycle=lifecycle,
        )
        state["stage_results"][commands[2]["stage"]] = result
        current_lobo_audit = validate_lobo_terminal_summary(
            Path(str(plan["lobo_root"]))
        )
        if current_lobo_audit != lobo_audit:
            raise WatcherContractError(
                "LOBO/source lineage changed before schema6 terminal publication"
            )
        state.update(
            {
                "status": TERMINAL_PENDING_STATUS,
                "current_stage": None,
                "stage_pid": None,
                "lobo_watcher_hdf5_opened": 0,
                "lobo_test_hdf5_opened": 0,
                "test_labels_read": False,
                "fresh_paths_accepted": False,
                "finished_unix": time.time(),
            }
        )
        atomic_json(state_path, state)
        inventory = _development_inventory(output)
        atomic_json(output / "artifact_inventory.json", inventory)
        final_base: dict[str, Any] = {
            "format": FORMAT,
            "status": TERMINAL_STATUS,
            "static_plan_sha256": plan["static_plan_sha256"],
            "lobo_gate": lobo_audit,
            "lobo_audit_sha256": lobo_audit["summary_sha256"],
            "transitive_source63_audit_sha256": lobo_audit[
                "transitive_source63_audit_sha256"
            ],
            "native_source_binding_audit": lobo_audit[
                "native_source_binding_audit"
            ],
            "native_source_binding_audit_sha256": lobo_audit[
                "native_source_binding_audit_sha256"
            ],
            "deployment_rerank_checkpoint": lobo_audit[
                "deployment_rerank_checkpoint"
            ],
            "lobo_checkpoints_rerank_authorized": False,
            "deployment_rerank_authority": "native_source_ensemble_only",
            "gpu_idle_before_materializer_sha256": first_idle["audit_sha256"],
            "gpu_idle_before_collection_sha256": second_idle["audit_sha256"],
            "subprocess_environment_audit_sha256": environment_audit[
                "audit_sha256"
            ],
            "execution_order": list(plan["execution_order"]),
            "stage_results": dict(state["stage_results"]),
            "stage_lifecycles": list(stage_lifecycles),
            "stage_returncodes": {
                name: state["stage_results"][name]["returncode"]
                for name in plan["execution_order"]
            },
            "stage_receipt_payload_sha256": {
                name: state["stage_results"][name]["receipt_payload_sha256"]
                for name in plan["execution_order"]
            },
            "collection_audit": result["artifact_audit"],
            "artifact_inventory_sha256": inventory["inventory_sha256"],
            "lobo_watcher_hdf5_opened": 0,
            "lobo_test_hdf5_opened": 0,
            "test_labels_read": False,
            "fresh_paths_accepted": False,
            "real_robot_execution": False,
            "performance_or_transfer_claim_authorized": False,
            "artifacts_frozen_read_only": True,
        }
        final = {**final_base, "receipt_sha256": canonical_sha256(final_base)}
        validate_schema6_success_terminal_receipt(final)
        publish_frozen_terminal_receipt(
            output,
            terminal_name="final_receipt.json",
            receipt=final,
            exit_code=0,
        )
        return final
    except BaseException as error:
        unreaped_stage = _unreaped_schema6_stage(stage_lifecycles)
        state.update(
            {
                "status": FAILURE_STATUS,
                "error_type": type(error).__name__,
                "error": str(error),
                "lobo_watcher_hdf5_opened": 0,
                "lobo_test_hdf5_opened": 0,
                "test_labels_read": False,
                "fresh_paths_accepted": False,
                "stage_lifecycles": stage_lifecycles,
                "unreaped_stage_process": unreaped_stage,
                "gpu_lock_acquired": gpu_lock_acquired,
                "gpu_lock_retained_for_unreaped_stage_process": (
                    unreaped_stage is not None and gpu_lock_acquired
                ),
                "artifacts_frozen_read_only": False,
                "finished_unix": time.time(),
            }
        )
        try:
            atomic_json(state_path, state)
            failure_base = {
                "format": FORMAT,
                "status": FAILURE_STATUS,
                "error_type": type(error).__name__,
                "error": str(error),
                "static_plan_sha256": plan["static_plan_sha256"],
                "execution_order": list(plan["execution_order"]),
                "stages_started": list(state.get("stages_started", [])),
                "stage_lifecycles": list(stage_lifecycles),
                "stage_returncodes": {
                    name: value.get("returncode")
                    for name, value in state.get("stage_results", {}).items()
                },
                "lobo_watcher_hdf5_opened": 0,
                "lobo_test_hdf5_opened": 0,
                "test_labels_read": False,
                "fresh_paths_accepted": False,
                "unreaped_stage_process": unreaped_stage,
                "gpu_lock_acquired": gpu_lock_acquired,
                "gpu_lock_retained_for_unreaped_stage_process": (
                    unreaped_stage is not None and gpu_lock_acquired
                ),
                "artifacts_frozen_read_only": unreaped_stage is None,
                "subprocess_environment_audit_sha256": state.get(
                    "subprocess_environment_audit", {}
                ).get("audit_sha256"),
            }
            failure = {**failure_base, "receipt_sha256": canonical_sha256(failure_base)}
            if unreaped_stage is None:
                publish_frozen_terminal_receipt(
                    output,
                    terminal_name="failure_receipt.json",
                    receipt=failure,
                    exit_code=1,
                )
            else:
                atomic_json(output / "failure_receipt.json", failure)
                if not (output / "run.exit").exists():
                    immutable_text_new(output / "run.exit", "1\n")
        except BaseException:
            pass
        raise
    finally:
        if _owned_gpu_lock_release_allowed(
            gpu_lock_acquired=gpu_lock_acquired,
            stage_lifecycles=stage_lifecycles,
        ):
            release_owned_lock(lock_path, gpu_token)


def _common_run_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        str(args.python_bin),
        str(Path(__file__).resolve()),
        "run",
        "--lobo-root",
        str(args.lobo_root),
        "--code-root",
        str(args.code_root),
        "--r6f-preregistration",
        str(args.r6f_preregistration),
        "--event-spec",
        str(args.event_spec),
        "--output",
        str(args.output),
        "--python-bin",
        str(args.python_bin),
        "--gpu-index",
        str(args.gpu_index),
        "--poll-seconds",
        str(args.poll_seconds),
        "--lobo-timeout-seconds",
        str(args.lobo_timeout_seconds),
        "--gpu-timeout-seconds",
        str(args.gpu_timeout_seconds),
        "--materializer-timeout-seconds",
        str(args.materializer_timeout_seconds),
        "--freezer-timeout-seconds",
        str(args.freezer_timeout_seconds),
        "--collection-timeout-seconds",
        str(args.collection_timeout_seconds),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--omp-threads",
        str(args.omp_threads),
    ]
    if args.gpu_lock is not None:
        argv.extend(["--gpu-lock", str(args.gpu_lock)])
    return argv


def detach(args: argparse.Namespace) -> dict[str, Any]:
    plan = static_preflight(args)
    output = Path(str(plan["output_root"]))
    receipt_path = resolve_new(
        args.detach_receipt or output.parent / f"{output.name}.detach_receipt.json",
        role="detached watcher receipt",
    )
    log_path = resolve_new(
        args.detach_log or output.parent / f"{output.name}.watcher.log",
        role="detached watcher log",
    )
    argv = _common_run_argv(args)
    with log_path.open("x", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    receipt_base: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detached_server_side_schema6_watcher_started",
        "pid": process.pid,
        "argv": argv,
        "argv_sha256": _argv_sha(argv),
        "output_root": str(output),
        "daemon_log": str(log_path),
        "static_plan_sha256": plan["static_plan_sha256"],
        "start_new_session": True,
        "stdin_devnull": True,
        "survives_client_disconnect": True,
        "lobo_summary_read_by_detach": False,
        "lobo_watcher_hdf5_opened_by_detach": 0,
        "lobo_test_hdf5_opened_by_detach": 0,
        "fresh_paths_accepted": False,
    }
    receipt = {**receipt_base, "receipt_sha256": canonical_sha256(receipt_base)}
    immutable_json_new(receipt_path, receipt)
    print("SMOLVLA_PIPER_SCHEMA6_DETACHED=" + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lobo-root", type=Path, default=DESIGNATED_LOBO_ROOT)
    parser.add_argument("--code-root", type=Path, default=DESIGNATED_CODE_ROOT)
    parser.add_argument("--r6f-preregistration", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DESIGNATED_PYTHON)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--lobo-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--materializer-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--freezer-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--collection-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--max-episode-steps", type=int, default=4)
    parser.add_argument("--omp-threads", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preflight", "Verify immutable inputs without reading LOBO summary/HDF5."),
        ("run", "Run the autonomous watcher in the foreground."),
        ("detach", "Start the server-side watcher in a new OS session."),
    ):
        subparser = commands.add_parser(name, help=help_text)
        add_common_arguments(subparser)
        if name == "detach":
            subparser.add_argument("--detach-receipt", type=Path)
            subparser.add_argument("--detach-log", type=Path)
    args = parser.parse_args()
    if (
        args.gpu_index != 0
        or not 0 < args.poll_seconds <= 60
        or args.lobo_timeout_seconds < 0
        or args.gpu_timeout_seconds < 0
        or args.materializer_timeout_seconds <= 0
        or args.freezer_timeout_seconds <= 0
        or args.collection_timeout_seconds < 0
        or not 1 <= args.max_episode_steps <= 200
        or args.omp_threads <= 0
    ):
        parser.error("invalid watcher timing/device/scope arguments")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        plan = static_preflight(args)
        print("SMOLVLA_PIPER_SCHEMA6_PREFLIGHT=" + json.dumps(plan, sort_keys=True))
        return
    if args.command == "detach":
        detach(args)
        return
    result = execute(args)
    print("SMOLVLA_PIPER_SCHEMA6_COMPLETE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
