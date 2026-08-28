#!/usr/bin/env python3
"""Two-phase paired-success v3: immutable core, Ed25519 decision, bundle.

The module consumes signed metadata and hashes source/adapter checkpoints as
opaque bytes.  It never opens prediction/label/HDF/trajectory/outcome files,
never deserializes a checkpoint, and never executes a policy or simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

import freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2 as bridge_v2
import launch_smolvla_piper_schema6_post_collection_v3 as post_v3


CORE_FORMAT = "etsf_smolvla_piper_paired_success_protocol_core_v3"
CORE_STATUS = "frozen_preoutcome_core_external_execution_not_authorized"
DECISION_FORMAT = "etsf_smolvla_piper_paired_execution_decision_v3"
DECISION_STATUS = "independent_ed25519_execution_decision"
BUNDLE_FORMAT = "etsf_smolvla_piper_paired_execution_bundle_v3"
BUNDLE_STATUS = "verified_ed25519_authorized_external_execution_bundle"
EXECUTION_INVENTORY_FORMAT = (
    "etsf_smolvla_piper_paired_execution_inventory_attestation_v3"
)
EXECUTION_INVENTORY_STATUS = "externally_reviewed_complete_immutable_execution_stack"
ISSUER_ATTESTATION_FORMAT = "etsf_smolvla_piper_trusted_issuer_allowlist_v3"
ISSUER_ATTESTATION_STATUS = "externally_reviewed_active_ed25519_issuer"
APPROVED_POST_V3_LAUNCHER_SHA256 = (
    "619d434280afe19317dcb29400d1339728775cbe9565a255ea2617c3dc30cd85"
)
# This is deliberately fail-closed until the real executor/result evaluator and
# their complete immutable runtime inventory have passed independent review.
# Synthetic tests monkeypatch this value to the synthetic inventory file SHA.
APPROVED_EXECUTION_INVENTORY_FILE_SHA256: str | None = None
SIGNATURE_CONTEXT = b"ETSF/SmolVLA/Piper/paired-v3/execution-authority\0"
PAIR_COUNT = 400
MEMBER_COUNT = 5
HEAD_NAMES = (
    "post_event", "next_event", "duration", "success", "recovery",
    "object_effect",
)
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_SAMPLES = 20_000
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
FORBIDDEN_PATH_COMPONENTS = frozenset(
    {"fresh", "confirmation", "test", "trajectory", "label", "labels"}
)
SOURCE_FORMAT = "etsf_smolvla_schema5_source63_native_training_launcher_v1"
SOURCE_STATUS = "complete_source63_native_counterfactual_training_fresh_forbidden"
EXPECTED_STAGES = (
    "materialize_development300_v3",
    "train_adapter_member_0",
    "train_adapter_member_1",
    "train_adapter_member_2",
    "train_adapter_member_3",
    "train_adapter_member_4",
    "evaluate_frozen_five_member_ensemble_on_formal190",
    "calibrate_six_head_formal190_ensemble",
)
RECORD_FIELDS = {"path", "file_sha256", "logical_sha256"}
OPAQUE_RECORD_FIELDS = {"path", "file_sha256"}
SOURCE_RANK_NUMERIC_CONTRACT = bridge_v2.SOURCE_RANK_NUMERIC_CONTRACT


class PairedProtocolV3Error(RuntimeError):
    """A v3 lineage, signature, type, or capability boundary failed closed."""


def canonical_bytes(value: Any) -> bytes:
    # Protocol-owned documents contain no floats.  Upstream logical hashes are
    # verified with their historical encoder separately.
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def require_int(value: Any, expected: int, role: str) -> int:
    if type(value) is not int or value != expected:
        raise PairedProtocolV3Error(f"{role} must be exact integer {expected}")
    return value


def _reject_constant(token: str) -> None:
    raise PairedProtocolV3Error(f"non-finite JSON number is forbidden: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise PairedProtocolV3Error(f"duplicate JSON key is forbidden: {key}")
        value[key] = child
    return value


def _sensitive(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if lowered in FORBIDDEN_PATH_COMPONENTS:
            return True
        if lowered.startswith(("fresh_", "fresh-", "confirmation_", "confirmation-",
                               "test_", "test-", "trajectory_", "trajectory-",
                               "label_", "label-")):
            return True
    return False


def _secure_read(
    raw_path: Path, role: str, *, expected_file_sha256: str | None,
    json_only: bool, frozen: bool = True,
) -> tuple[Path, bytes, str]:
    if expected_file_sha256 is not None and not is_sha(expected_file_sha256):
        raise PairedProtocolV3Error(f"{role} expected file SHA is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(raw_path))))
    if _sensitive(PurePath(lexical)):
        raise PairedProtocolV3Error(f"{role} path is in a forbidden namespace")
    if lexical.suffix.casefold() in HDF_SUFFIXES:
        raise PairedProtocolV3Error(f"{role} HDF path is forbidden")
    if json_only and lexical.suffix.casefold() != ".json":
        raise PairedProtocolV3Error(f"{role} must be JSON")
    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(lexical.anchor, flags_dir)
    file_fd: int | None = None
    try:
        for component in lexical.parts[1:-1]:
            next_fd = os.open(component, flags_dir, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(lexical.name, flags_file, dir_fd=directory_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise PairedProtocolV3Error(f"{role} must be a regular file")
        if frozen and metadata.st_mode & 0o222:
            raise PairedProtocolV3Error(f"{role} must be frozen read-only")
        chunks: list[bytes] = []
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        metadata_after = os.fstat(file_fd)
        if (
            (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            != (
                metadata_after.st_dev, metadata_after.st_ino,
                metadata_after.st_size, metadata_after.st_mtime_ns,
            )
            or len(payload) != metadata.st_size
        ):
            raise PairedProtocolV3Error(f"{role} changed while being read")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_file_sha256 is not None and digest != expected_file_sha256:
            raise PairedProtocolV3Error(f"{role} file SHA mismatch")
        return lexical, payload, digest
    except OSError as error:
        raise PairedProtocolV3Error(
            f"{role} cannot be opened without following symlinks"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def read_json(
    path: Path, expected_file_sha256: str, role: str, *, frozen: bool = True
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload, digest = _secure_read(
        path, role, expected_file_sha256=expected_file_sha256,
        json_only=True, frozen=frozen,
    )
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairedProtocolV3Error(f"{role} is invalid strict JSON") from error
    if not isinstance(value, dict):
        raise PairedProtocolV3Error(f"{role} must contain an object")
    return resolved, value, digest


def hash_opaque_file(
    path: Path, expected: str, role: str, *, frozen: bool = True
) -> tuple[Path, str]:
    resolved, _payload, digest = _secure_read(
        path, role, expected_file_sha256=expected, json_only=False, frozen=frozen
    )
    return resolved, digest


def verify_hash_signature(
    value: Mapping[str, Any], field: str, role: str
) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not is_sha(recorded) or recorded != post_v3.canonical_sha256(unsigned):
        raise PairedProtocolV3Error(f"{role} logical SHA mismatch")
    return str(recorded)


def validate_record(value: Any, role: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping) or set(value) != RECORD_FIELDS
        or not isinstance(value.get("path"), str)
        or not is_sha(value.get("file_sha256"))
        or not is_sha(value.get("logical_sha256"))
    ):
        raise PairedProtocolV3Error(f"{role} descriptor changed")
    return dict(value)


def validate_opaque_record(value: Any, role: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping) or set(value) != OPAQUE_RECORD_FIELDS
        or not isinstance(value.get("path"), str)
        or not is_sha(value.get("file_sha256"))
    ):
        raise PairedProtocolV3Error(f"{role} descriptor changed")
    return dict(value)


def _validate_execution_inventory(
    *, path: Path, expected_file_sha256: str, expected_logical_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], str]:
    approved = APPROVED_EXECUTION_INVENTORY_FILE_SHA256
    if approved is None:
        raise PairedProtocolV3Error(
            "real executor/result evaluator inventory has not been independently approved"
        )
    if (
        not is_sha(approved) or expected_file_sha256 != approved
        or not is_sha(expected_logical_sha256)
    ):
        raise PairedProtocolV3Error("execution inventory is not on the trusted allowlist")
    bound_path, inventory, digest = read_json(
        path, expected_file_sha256, "execution inventory attestation"
    )
    logical = verify_hash_signature(
        inventory, "attestation_sha256", "execution inventory attestation"
    )
    expected_fields = {
        "format", "status", "protocol_format", "execution_lane",
        "trusted_issuer_attestation", "executor", "result_evaluator",
        "execution_stack", "component_inventory_complete",
        "real_executor_present", "real_result_evaluator_present",
        "outcome_or_trajectory_files_opened_during_attestation",
        "attestation_sha256",
    }
    lane = inventory.get("execution_lane")
    executor = inventory.get("executor")
    evaluator = inventory.get("result_evaluator")
    stack = inventory.get("execution_stack")
    if (
        set(inventory) != expected_fields
        or inventory.get("format") != EXECUTION_INVENTORY_FORMAT
        or inventory.get("status") != EXECUTION_INVENTORY_STATUS
        or inventory.get("protocol_format") != CORE_FORMAT
        or not isinstance(lane, Mapping)
        or dict(lane) != {
            "pair_count": 400, "only_evaluation400_lane": True,
            "additional_reserve400_count": 0,
        }
        or inventory.get("component_inventory_complete") is not True
        or inventory.get("real_executor_present") is not True
        or inventory.get("real_result_evaluator_present") is not True
        or type(inventory.get("outcome_or_trajectory_files_opened_during_attestation"))
        is not int
        or inventory["outcome_or_trajectory_files_opened_during_attestation"] != 0
        or logical != expected_logical_sha256
        or not isinstance(executor, Mapping)
        or set(executor) != {"identity_sha256", "implementation"}
        or not is_sha(executor.get("identity_sha256"))
        or not isinstance(evaluator, Mapping)
        or set(evaluator) != {"identity_sha256", "implementation"}
        or not is_sha(evaluator.get("identity_sha256"))
        or not isinstance(stack, Mapping)
        or set(stack) != {
            "simulator_implementation", "runtime_contract",
            "collector_implementation", "container_inventory",
        }
    ):
        raise PairedProtocolV3Error("execution inventory contract changed")

    components: dict[str, dict[str, str]] = {}
    component_values = {
        "executor_implementation": executor.get("implementation"),
        "result_evaluator_implementation": evaluator.get("implementation"),
        **dict(stack),
    }
    expected_suffixes = {
        "executor_implementation": ".py",
        "result_evaluator_implementation": ".py",
        "simulator_implementation": ".py",
        "runtime_contract": ".json",
        "collector_implementation": ".py",
        "container_inventory": ".json",
    }
    for role, raw in component_values.items():
        descriptor = validate_opaque_record(raw, role)
        component_path, component_sha = hash_opaque_file(
            Path(descriptor["path"]), descriptor["file_sha256"], role
        )
        if component_path.suffix.casefold() != expected_suffixes[role]:
            raise PairedProtocolV3Error(f"{role} has an unexpected file type")
        components[role] = {
            "path": str(component_path), "file_sha256": component_sha,
        }

    issuer_record = validate_record(
        inventory.get("trusted_issuer_attestation"), "trusted issuer attestation"
    )
    issuer_path, issuer, issuer_file_sha = read_json(
        Path(issuer_record["path"]), issuer_record["file_sha256"],
        "trusted issuer attestation",
    )
    issuer_logical = verify_hash_signature(
        issuer, "attestation_sha256", "trusted issuer attestation"
    )
    if (
        set(issuer) != {
            "format", "status", "protocol_format", "issuer_key_id",
            "issuer_public_key_hex", "issuer_public_key_sha256",
            "issuer_identity_sha256", "allowlist_entry_active",
            "authorization_sequence", "attestation_sha256",
        }
        or issuer.get("format") != ISSUER_ATTESTATION_FORMAT
        or issuer.get("status") != ISSUER_ATTESTATION_STATUS
        or issuer.get("protocol_format") != CORE_FORMAT
        or not isinstance(issuer.get("issuer_key_id"), str)
        or not issuer["issuer_key_id"]
        or not is_sha(issuer.get("issuer_public_key_sha256"))
        or not is_sha(issuer.get("issuer_identity_sha256"))
        or issuer.get("allowlist_entry_active") is not True
        or type(issuer.get("authorization_sequence")) is not int
        or issuer["authorization_sequence"] != 1
        or issuer_file_sha != issuer_record["file_sha256"]
        or issuer_logical != issuer_record["logical_sha256"]
    ):
        raise PairedProtocolV3Error("trusted issuer allowlist attestation changed")
    public_key = _decode_hex(
        issuer.get("issuer_public_key_hex"), 32, "allowlisted issuer Ed25519 public key"
    )
    if hashlib.sha256(public_key).hexdigest() != issuer["issuer_public_key_sha256"]:
        raise PairedProtocolV3Error("trusted issuer public-key fingerprint changed")
    record = {
        "path": str(bound_path), "file_sha256": digest,
        "logical_sha256": logical,
    }
    issuer_binding = {
        **dict(issuer), "path": str(issuer_path), "file_sha256": issuer_file_sha,
    }
    stack_binding_sha256 = canonical_sha256({
        "executor_identity_sha256": executor["identity_sha256"],
        "result_evaluator_identity_sha256": evaluator["identity_sha256"],
        "components": components,
    })
    return dict(inventory), issuer_binding, record, stack_binding_sha256


def _fixed_point(value: Any, role: str) -> dict[str, int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedProtocolV3Error(f"{role} must be finite numeric")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise PairedProtocolV3Error(f"{role} is invalid") from error
    if not decimal.is_finite() or decimal < 0:
        raise PairedProtocolV3Error(f"{role} must be finite and nonnegative")
    normalized = decimal.normalize()
    sign, digits, exponent = normalized.as_tuple()
    if sign:
        raise PairedProtocolV3Error(f"{role} cannot be negative")
    coefficient = int("".join(str(item) for item in digits) or "0")
    if exponent > 0:
        coefficient *= 10**exponent
        places = 0
    else:
        places = -exponent
    if places > 18:
        raise PairedProtocolV3Error(f"{role} exceeds fixed-point precision")
    return {"coefficient": coefficient, "decimal_places": places}


def _validate_post_plan(
    value: Mapping[str, Any], *, path: Path, root: Path,
    expected_launcher_sha256: str,
) -> str:
    logical = verify_hash_signature(value, "plan_sha256", "post v3 static plan")
    expected_fields = {
        "format", "status", "output_root", "source_root", "r9b_root",
        "r9b_final_file_sha256", "r9b_final_sha256", "r9b_static_plan_sha256",
        "development300_collection_root", "development300_terminal",
        "development300_runner_authority", "development300_target_preregistration",
        "development300_identity_authority", "python", "implementations",
        "python_import_closure",
        "canonical_event_spec", "canonical_teacher", "split_profile",
        "required_trainer_group_counts", "adapter_member_count",
        "adapter_member_seeds", "adapter_source_policy",
        "lobo_or_aggregate_checkpoint_authorized", "adapter_steps",
        "adapter_eval_every", "gpu_index", "gpu_lock_path", "formal190_claim_root",
        "formal190_open_authorized_before_five_adapters_frozen",
        "evaluation400_membership_or_label_open_authorized",
        "old_paired400_authority_path_present", "second_reserve400_authorized",
        "hdf5_files_opened_during_preregistration",
        "labels_or_outcomes_read_during_preregistration",
        "create_once_nonresumable", "plan_sha256",
    }
    implementations = value.get("implementations")
    launcher = implementations.get("launcher") if isinstance(implementations, Mapping) else None
    closure = value.get("python_import_closure")
    expected_implementation_roles = {
        "launcher", "materializer", "trainer", "evaluator", "calibrator",
        "identity_bridge_v2", "r9b_watcher",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != post_v3.PLAN_FORMAT
        or value.get("status")
        != "preregistered_waiting_for_exact_r7h_r8e_r9b_and_development300"
        or value.get("output_root") != str(root)
        or path != root / "_watcher" / "static_plan.json"
        or value.get("split_profile") != post_v3.SPLIT_PROFILE
        or value.get("required_trainer_group_counts")
        != {"train": 80, "validation": 30, "test": 190}
        or type(value.get("adapter_member_count")) is not int
        or value["adapter_member_count"] != MEMBER_COUNT
        or value.get("adapter_member_seeds") != list(post_v3.SOURCE_MEMBER_SEEDS)
        or value.get("adapter_source_policy")
        != "one_to_one_native_r7h_individual_members_only"
        or value.get("canonical_teacher") is not None
        or value.get("lobo_or_aggregate_checkpoint_authorized") is not False
        or value.get("formal190_open_authorized_before_five_adapters_frozen") is not False
        or value.get("evaluation400_membership_or_label_open_authorized") is not False
        or value.get("old_paired400_authority_path_present") is not False
        or value.get("second_reserve400_authorized") is not False
        or type(value.get("hdf5_files_opened_during_preregistration")) is not int
        or value["hdf5_files_opened_during_preregistration"] != 0
        or value.get("labels_or_outcomes_read_during_preregistration") is not False
        or value.get("create_once_nonresumable") is not True
        or not isinstance(implementations, Mapping)
        or set(implementations) != expected_implementation_roles
        or not isinstance(launcher, Mapping)
        or set(launcher) != {"path", "file_sha256"}
        or launcher.get("file_sha256") != expected_launcher_sha256
        or not isinstance(closure, Mapping)
        or not closure
    ):
        raise PairedProtocolV3Error("post v3 static plan contract changed")
    implementation_modules: dict[str, dict[str, str]] = {}
    for role in sorted(expected_implementation_roles):
        implementation = implementations[role]
        if (
            not isinstance(implementation, Mapping)
            or set(implementation) != {"path", "file_sha256"}
            or not isinstance(implementation.get("path"), str)
            or not is_sha(implementation.get("file_sha256"))
        ):
            raise PairedProtocolV3Error(
                f"post v3 {role} implementation descriptor changed"
            )
        module_name = Path(str(implementation["path"])).stem
        if not module_name.isidentifier():
            raise PairedProtocolV3Error(
                f"post v3 {role} implementation module name changed"
            )
        previous = implementation_modules.get(module_name)
        normalized = dict(implementation)
        if previous is not None and previous != normalized:
            raise PairedProtocolV3Error(
                "post v3 implementation module binding is ambiguous"
            )
        implementation_modules[module_name] = normalized
    normalized_closure: dict[str, dict[str, str]] = {}
    for module_name, raw_record in closure.items():
        if (
            not isinstance(module_name, str)
            or not module_name.isidentifier()
            or not isinstance(raw_record, Mapping)
            or set(raw_record) != {"path", "file_sha256"}
            or not isinstance(raw_record.get("path"), str)
            or Path(str(raw_record["path"])).stem != module_name
            or not is_sha(raw_record.get("file_sha256"))
        ):
            raise PairedProtocolV3Error(
                "post v3 Python import closure descriptor changed"
            )
        record = dict(raw_record)
        bound_path, bound_sha = hash_opaque_file(
            Path(record["path"]), record["file_sha256"],
            f"post v3 Python import {module_name}", frozen=False,
        )
        if str(bound_path) != record["path"] or bound_sha != record["file_sha256"]:
            raise PairedProtocolV3Error(
                "post v3 Python import closure binding changed"
            )
        normalized_closure[module_name] = record
    if any(
        normalized_closure.get(module_name) != record
        for module_name, record in implementation_modules.items()
    ):
        raise PairedProtocolV3Error(
            "post v3 Python import closure omits an exact implementation binding"
        )
    try:
        rebuilt_closure = post_v3.build_python_import_closure(implementations)
    except Exception as error:
        raise PairedProtocolV3Error(
            "post v3 Python import closure cannot be independently rebuilt"
        ) from error
    if rebuilt_closure != normalized_closure:
        raise PairedProtocolV3Error(
            "post v3 Python import closure is not the complete reviewed closure"
        )
    launcher_path, launcher_sha = hash_opaque_file(
        Path(str(launcher["path"])), expected_launcher_sha256,
        "reviewed post v3 launcher", frozen=False,
    )
    if str(launcher_path) != launcher["path"] or launcher_sha != expected_launcher_sha256:
        raise PairedProtocolV3Error("post v3 launcher binding changed")
    return logical


def _validate_post_terminal(root: Path, value: Mapping[str, Any]) -> str:
    logical = verify_hash_signature(value, "receipt_sha256", "post v3 terminal")
    expected_fields = {
        "format", "status", "plan_sha256", "detach_proof_sha256",
        "execution_order", "stage_results", "adapter_member_count",
        "adapter_member_seeds", "adapter_source_policy",
        "r7h_member_checkpoint_sha256", "r8e_r9b_lineage_sha256",
        "development300_materializer_receipt_sha256",
        "formal190_opened_after_five_frozen_adapters",
        "formal190_labels_opened_before_five_adapters_frozen",
        "calibration_receipt_sha256", "identity_bridge_v2_handoff",
        "formal190_global_one_shot_claim",
        "evaluation400_hdf5_trajectory_or_labels_opened",
        "evaluation400_conditions_executed",
        "old_paired400_authority_waited_or_generated",
        "second_reserve400_created", "gpu_lock_release_sha256",
        "artifacts_frozen_read_only", "terminal_publication",
        "artifact_closure", "artifact_closure_sha256", "receipt_sha256",
    }
    stage_results = value.get("stage_results")
    if (
        set(value) != expected_fields
        or value.get("format") != post_v3.FORMAT
        or value.get("status") != post_v3.TERMINAL_STATUS
        or value.get("execution_order") != list(EXPECTED_STAGES)
        or not isinstance(stage_results, Mapping)
        or set(stage_results) != set(EXPECTED_STAGES)
        or value.get("adapter_member_seeds") != list(post_v3.SOURCE_MEMBER_SEEDS)
        or value.get("adapter_source_policy")
        != "one_to_one_native_r7h_individual_members_only"
        or value.get("old_paired400_authority_waited_or_generated") is not False
        or value.get("second_reserve400_created") is not False
        or value.get("artifacts_frozen_read_only") is not True
        or value.get("terminal_publication")
        != "mode000_then_tree_freeze_verify_then_run_exit0444_then_final_receipt0444_last"
    ):
        raise PairedProtocolV3Error("post v3 terminal validation failed")
    for field, expected in (
        ("adapter_member_count", 5),
        ("formal190_opened_after_five_frozen_adapters", 190),
        ("formal190_labels_opened_before_five_adapters_frozen", 0),
        ("evaluation400_hdf5_trajectory_or_labels_opened", 0),
        ("evaluation400_conditions_executed", 0),
    ):
        require_int(value.get(field), expected, f"post terminal {field}")
    for field in (
        "plan_sha256", "detach_proof_sha256", "r8e_r9b_lineage_sha256",
        "development300_materializer_receipt_sha256",
        "calibration_receipt_sha256", "gpu_lock_release_sha256",
    ):
        if not is_sha(value.get(field)):
            raise PairedProtocolV3Error(f"post terminal SHA changed: {field}")
    source_shas = value.get("r7h_member_checkpoint_sha256")
    if (
        not isinstance(source_shas, list) or len(source_shas) != 5
        or len(set(source_shas)) != 5 or any(not is_sha(item) for item in source_shas)
    ):
        raise PairedProtocolV3Error("post terminal r7h member set changed")
    physical_bindings: list[Mapping[str, Any]] = []
    for name in EXPECTED_STAGES:
        result = stage_results[name]
        lifecycle = result.get("lifecycle") if isinstance(result, Mapping) else None
        if (
            not isinstance(result, Mapping)
            or set(result) != {
                "stage", "returncode", "command_sha256", "launch_file_sha256",
                "lifecycle", "physical_gpu", "log_file_sha256",
                "run_exit_file_sha256", "result_sha256",
            }
            or result.get("stage") != name
            or type(result.get("returncode")) is not int or result["returncode"] != 0
            or not isinstance(lifecycle, Mapping)
            or set(lifecycle) != {
                "popen_attempted", "popen_reached", "process_pid", "process_pgid",
                "process_group_isolated", "returncode", "direct_process_reaped",
                "process_group_reaped", "binding_status", "lifecycle_sha256",
            }
            or lifecycle.get("popen_attempted") is not True
            or lifecycle.get("popen_reached") is not True
            or lifecycle.get("process_group_isolated") is not True
            or lifecycle.get("direct_process_reaped") is not True
            or lifecycle.get("process_group_reaped") is not True
            or lifecycle.get("binding_status") != "bound_reaped"
            or type(lifecycle.get("process_pid")) is not int
            or lifecycle["process_pid"] <= 0
            or type(lifecycle.get("process_pgid")) is not int
            or lifecycle["process_pgid"] != lifecycle["process_pid"]
            or type(lifecycle.get("returncode")) is not int
            or lifecycle["returncode"] != 0
            or verify_hash_signature(lifecycle, "lifecycle_sha256", f"{name} lifecycle")
            != lifecycle["lifecycle_sha256"]
            or any(not is_sha(result.get(field)) for field in (
                "command_sha256", "launch_file_sha256", "log_file_sha256",
                "run_exit_file_sha256", "result_sha256",
            ))
            or result["result_sha256"] != post_v3.canonical_sha256(
                {key: child for key, child in result.items() if key != "result_sha256"}
            )
        ):
            raise PairedProtocolV3Error(f"post v3 stage changed: {name}")
        gpu_stage = name.startswith("train_adapter_member_") or name.startswith(
            "evaluate_frozen_five_member"
        )
        physical = result.get("physical_gpu")
        if gpu_stage:
            if (
                not isinstance(physical, Mapping)
                or set(physical) != {
                    "gpu_index", "gpu_name", "gpu_uuid", "checks", "audit_sha256"
                }
                or type(physical.get("gpu_index")) is not int
                or physical["gpu_index"] < 0
                or not isinstance(physical.get("gpu_uuid"), str)
                or not physical["gpu_uuid"]
                or not is_sha(physical.get("audit_sha256"))
            ):
                raise PairedProtocolV3Error(f"post v3 physical GPU changed: {name}")
            physical_bindings.append(physical)
        elif physical is not None:
            raise PairedProtocolV3Error(f"CPU stage claims a GPU: {name}")
    if len({post_v3.canonical_sha256(row) for row in physical_bindings}) != 1:
        raise PairedProtocolV3Error("post v3 GPU stages do not share one physical device")
    _validate_terminal_artifact_closure(root, value)
    return logical


def _validate_terminal_artifact_closure(
    root: Path, terminal: Mapping[str, Any]
) -> None:
    records = terminal.get("artifact_closure")
    expected_sha = terminal.get("artifact_closure_sha256")
    expected_roles = {
        "static_plan", "detached_worker_proof", "gpu_idle_before_training",
        "gpu_idle_after_formal190", "gpu_lock_release", "materializer_v3_receipt",
        "formal190_evaluator_authority", "formal190_evaluator_receipt",
        "calibration_receipt", "identity_bridge_v2_handoff",
        "formal190_global_one_shot_claim",
        *[f"adapter_member_{index}_receipt" for index in range(5)],
        *[
            f"stage:{stage}:{suffix}"
            for stage in EXPECTED_STAGES for suffix in ("launch", "lifecycle", "log", "exit")
        ],
    }
    if (
        not isinstance(records, list) or not is_sha(expected_sha)
        or post_v3.canonical_sha256(records) != expected_sha
        or len(records) != len(expected_roles)
    ):
        raise PairedProtocolV3Error("post terminal artifact closure changed")
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"role", "path", "file_sha256", "signature", "logical_sha256"}
            or not isinstance(row.get("role"), str) or row["role"] in by_role
            or not is_sha(row.get("file_sha256"))
            or (row.get("signature") is None and row.get("logical_sha256") is not None)
            or (row.get("signature") is not None and (
                not isinstance(row.get("signature"), str)
                or not is_sha(row.get("logical_sha256"))
            ))
        ):
            raise PairedProtocolV3Error("post terminal closure record changed")
        path, payload, _digest = _secure_read(
            Path(str(row["path"])), f"terminal closure {row['role']}",
            expected_file_sha256=str(row["file_sha256"]), json_only=False,
            frozen=True,
        )
        if row["signature"] is not None:
            if path.suffix.casefold() != ".json":
                raise PairedProtocolV3Error("signed closure artifact must be JSON")
            try:
                document = json.loads(
                    payload.decode("utf-8"), object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (UnicodeError, json.JSONDecodeError) as error:
                raise PairedProtocolV3Error("signed closure artifact is invalid JSON") from error
            if (
                not isinstance(document, Mapping)
                or verify_hash_signature(
                    document, str(row["signature"]), f"closure {row['role']}"
                ) != row["logical_sha256"]
            ):
                raise PairedProtocolV3Error("closure logical SHA changed")
        by_role[row["role"]] = row
    if set(by_role) != expected_roles:
        raise PairedProtocolV3Error("post terminal closure role inventory changed")
    expected_inside = {
        "static_plan": root / "_watcher" / "static_plan.json",
        "detached_worker_proof": root / "_watcher" / "detached_worker_proof.json",
        "gpu_lock_release": root / "_watcher" / "gpu_lock_release.json",
        "identity_bridge_v2_handoff": root / "handoff" / "evaluation400_identity_bridge_v2_handoff.json",
    }
    if any(Path(str(by_role[role]["path"])) != path for role, path in expected_inside.items()):
        raise PairedProtocolV3Error("post terminal closure internal path changed")
    if (
        by_role["detached_worker_proof"]["logical_sha256"]
        != terminal["detach_proof_sha256"]
        or by_role["gpu_lock_release"]["logical_sha256"]
        != terminal["gpu_lock_release_sha256"]
        or by_role["materializer_v3_receipt"]["logical_sha256"]
        != terminal["development300_materializer_receipt_sha256"]
        or by_role["calibration_receipt"]["logical_sha256"]
        != terminal["calibration_receipt_sha256"]
    ):
        raise PairedProtocolV3Error("post terminal summary/closure mismatch")
    for name in EXPECTED_STAGES:
        result = terminal["stage_results"][name]
        if (
            by_role[f"stage:{name}:launch"]["file_sha256"] != result["launch_file_sha256"]
            or by_role[f"stage:{name}:lifecycle"]["logical_sha256"]
            != result["lifecycle"]["lifecycle_sha256"]
            or by_role[f"stage:{name}:log"]["file_sha256"] != result["log_file_sha256"]
            or by_role[f"stage:{name}:exit"]["file_sha256"] != result["run_exit_file_sha256"]
        ):
            raise PairedProtocolV3Error(f"post stage closure mismatch: {name}")


def _validate_formal190_claim(
    descriptor: Any, *, terminal_descriptor: Any
) -> dict[str, Any]:
    expected_descriptor_fields = {
        "path", "file_sha256", "logical_sha256", "formal190_identity_sha256",
        "consumed",
    }
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != expected_descriptor_fields
        or descriptor.get("consumed") is not True
        or not is_sha(descriptor.get("file_sha256"))
        or not is_sha(descriptor.get("logical_sha256"))
        or not is_sha(descriptor.get("formal190_identity_sha256"))
        or dict(descriptor) != terminal_descriptor
    ):
        raise PairedProtocolV3Error("formal190 one-shot descriptor changed")
    path, claim, digest = read_json(
        Path(str(descriptor["path"])), descriptor["file_sha256"],
        "formal190 global one-shot claim",
    )
    logical = verify_hash_signature(claim, "claim_sha256", "formal190 global one-shot claim")
    expected_fields = {
        "format", "status", "formal190_identity_sha256",
        "development300_terminal_file_sha256", "development300_terminal_sha256",
        "split_profile", "formal_group_count", "post_v3_plan_sha256",
        "post_v3_output_root", "formal190_authority_may_be_created_once",
        "reopen_from_second_output_authorized", "claimed_unix_ns", "claim_sha256",
    }
    if (
        set(claim) != expected_fields
        or claim.get("format") != post_v3.FORMAL190_CLAIM_FORMAT
        or claim.get("status") != post_v3.FORMAL190_CLAIM_STATUS
        or claim.get("formal190_identity_sha256")
        != descriptor["formal190_identity_sha256"]
        or claim.get("split_profile") != post_v3.SPLIT_PROFILE
        or type(claim.get("formal_group_count")) is not int
        or claim["formal_group_count"] != 190
        or claim.get("formal190_authority_may_be_created_once") is not True
        or claim.get("reopen_from_second_output_authorized") is not False
        or type(claim.get("claimed_unix_ns")) is not int
        or claim["claimed_unix_ns"] <= 0
        or logical != descriptor["logical_sha256"]
        or digest != descriptor["file_sha256"]
    ):
        raise PairedProtocolV3Error("formal190 global one-shot claim changed")
    return dict(descriptor)


def _validate_member(
    value: Mapping[str, Any], *, index: int, expected_seed: int,
) -> str:
    if set(value) != set(post_v3.MEMBER_FIELDS):
        raise PairedProtocolV3Error(f"member {index} receipt fields changed")
    logical = verify_hash_signature(value, "receipt_sha256", f"member {index} receipt")
    counts = value.get("required_trainer_group_counts")
    contract = value.get("prediction_contract")
    source_rank_contract = value.get("source_rank_score_contract")
    expected_contract = {
        "duration_target_transform": "log1p_decision_steps",
        "next_event_observation_mask": "duration_observed",
        "success_target": "eventual_final_branch_success_repeated_per_transition",
        "recovery_target": "conditional_recovery_given_operational_regress",
        "recovery_observation_mask": "recovery_observed_and_regress",
        "recovery_shared_transition_stop_gradient": True,
        "recovery_enters_primary_before_calibration": False,
        "recovery_head_trained": True,
        "object_prediction_space": "physical_delta_xyz_m",
        "object_source_normalization_sha256": (
            contract.get("object_source_normalization_sha256")
            if isinstance(contract, Mapping) else None
        ),
        "object_observed_policy": "row_enabled_only_if_all_selected_xyz_are_valid",
    }
    if (
        value.get("format") != post_v3.MEMBER_FORMAT
        or value.get("status")
        != "complete_frozen_development300_internal_validation_adapter"
        or type(value.get("member_index")) is not int or value["member_index"] != index
        or type(value.get("member_seed")) is not int or value["member_seed"] != expected_seed
        or value.get("split_profile") != post_v3.SPLIT_PROFILE
        or type(value.get("split_profile_version")) is not int
        or value["split_profile_version"] != 3
        or not isinstance(counts, Mapping)
        or dict(counts) != {"train": 80, "validation": 30, "test": 190}
        or any(type(counts.get(k)) is not int for k in counts)
    ):
        raise PairedProtocolV3Error(f"member {index} split/profile changed")
    if (
        value.get("source_checkpoint_role")
        != "native_r7h_individual_source_member"
        or value.get("validation_lane")
        != "adaptation_derived_internal_validation_only"
        or type(value.get("internal_validation_group_count")) is not int
        or value["internal_validation_group_count"] != 30
        or type(value.get("sealed_formal_target_validation_group_count")) is not int
        or value["sealed_formal_target_validation_group_count"] != 190
        or not isinstance(contract, Mapping) or dict(contract) != expected_contract
        or not is_sha(contract.get("object_source_normalization_sha256"))
        or type(value.get("formal_target_validation_hdf5_files_opened_before_five_adapters_frozen")) is not int
        or value["formal_target_validation_hdf5_files_opened_before_five_adapters_frozen"] != 0
        or type(value.get("formal_target_validation_labels_opened_before_five_adapters_frozen")) is not int
        or value["formal_target_validation_labels_opened_before_five_adapters_frozen"] != 0
        or value.get("formal_target_validation_release_condition")
        != "external_authority_after_all_five_adapter_checkpoints_are_frozen"
        or value.get("lobo_or_aggregate_checkpoint_used") is not False
        or not isinstance(source_rank_contract, Mapping)
        or source_rank_contract.get("source_checkpoint_file_sha256")
        != value.get("source_checkpoint_sha256")
        or source_rank_contract.get("source_contract_rank_score_is_success_logit")
        is not False
        or source_rank_contract.get(
            "source_contract_rank_score_is_success_probability"
        ) is not False
        or isinstance(source_rank_contract.get("success_temperature"), bool)
        or not isinstance(
            source_rank_contract.get("success_temperature"), (int, float)
        )
        or not Decimal(str(source_rank_contract["success_temperature"])).is_finite()
        or Decimal(str(source_rank_contract["success_temperature"])) <= 0
        or value.get("source_rank_score_contract_sha256")
        != source_rank_contract.get("contract_sha256")
        or verify_hash_signature(
            source_rank_contract,
            "contract_sha256",
            f"member {index} source rank score contract",
        )
        != value.get("source_rank_score_contract_sha256")
    ):
        raise PairedProtocolV3Error(f"member {index} recovery/formal190 semantics changed")
    for field in (
        "source_checkpoint_sha256", "training_manifest_sha256", "split_sha256",
        "source_ensemble_contract_sha256", "checkpoint_file_sha256",
        "validation_identity_set_sha256", "stage_result_sha256",
    ):
        if not is_sha(value.get(field)):
            raise PairedProtocolV3Error(f"member {index} SHA field changed: {field}")
    return logical


def _validate_evaluator_authority(
    value: Mapping[str, Any], members: Sequence[Mapping[str, Any]]
) -> str:
    logical = verify_hash_signature(value, "authority_sha256", "formal190 evaluator authority")
    expected_fields = {
        "format", "status", "trainer_compatible_manifest",
        "expected_manifest_split_receipt", "canonical_event_spec", "members",
        "member_count", "target_validation_group_count",
        "adapter_training_complete_before_authority",
        "target_validation_open_authorized", "evaluation400_membership_present",
        "evaluation400_open_authorized", "fresh_or_confirmation_open_authorized",
        "source_rank_numeric_contract",
        "authority_sha256",
    }
    rows = value.get("members")
    if (
        set(value) != expected_fields
        or value.get("format") != post_v3.evaluator.INPUT_FORMAT
        or value.get("status") != post_v3.evaluator.INPUT_STATUS
        or type(value.get("member_count")) is not int or value["member_count"] != 5
        or type(value.get("target_validation_group_count")) is not int
        or value["target_validation_group_count"] != 190
        or value.get("adapter_training_complete_before_authority") is not True
        or value.get("target_validation_open_authorized") is not True
        or value.get("evaluation400_membership_present") is not False
        or value.get("evaluation400_open_authorized") is not False
        or value.get("fresh_or_confirmation_open_authorized") is not False
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(rows, list) or len(rows) != 5
    ):
        raise PairedProtocolV3Error("formal190 evaluator authority changed")
    for index, (row, member) in enumerate(zip(rows, members, strict=True)):
        expected_row_fields = {
            "member_index", "member_seed", "adapter_checkpoint",
            "source_checkpoint", "member_receipt", "training_manifest_sha256",
            "split_sha256", "source_ensemble_contract_sha256",
            "prediction_contract", "source_rank_score_contract",
            "source_rank_score_contract_sha256",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
            or type(row.get("member_index")) is not int or row["member_index"] != index
            or type(row.get("member_seed")) is not int
            or row["member_seed"] != member["member_seed"]
            or row.get("training_manifest_sha256") != member["training_manifest_sha256"]
            or row.get("split_sha256") != member["split_sha256"]
            or row.get("source_ensemble_contract_sha256")
            != member["source_ensemble_contract_sha256"]
            or row.get("prediction_contract") != member["prediction_contract"]
            or row.get("source_rank_score_contract")
            != member["source_rank_score_contract"]
            or row.get("source_rank_score_contract", {}).get(
                "source_rank_numeric_contract"
            ) != value.get("source_rank_numeric_contract")
            or row.get("source_rank_score_contract_sha256")
            != member["source_rank_score_contract_sha256"]
            or row.get("adapter_checkpoint")
            != {"path": member["checkpoint_path"], "file_sha256": member["checkpoint_file_sha256"]}
            or row.get("source_checkpoint")
            != {"path": member["source_checkpoint_path"], "file_sha256": member["source_checkpoint_sha256"]}
            or row.get("member_receipt", {}).get("logical_sha256")
            != member["receipt_sha256"]
        ):
            raise PairedProtocolV3Error(f"formal190 authority member {index} changed")
    return logical


def _validate_formal_receipt(value: Mapping[str, Any]) -> str:
    logical = verify_hash_signature(value, "receipt_sha256", "formal190 evaluator receipt")
    expected_fields = {
        "format", "status", "input_authority_path", "input_authority_file_sha256",
        "input_authority_sha256", "target_validation_groups",
        "target_validation_samples", "target_validation_hdf5_files_opened",
        "target_validation_opened_after_five_adapters_frozen",
        "calibration_input_authority_path",
        "calibration_input_authority_file_sha256",
        "calibration_input_authority_sha256",
        "source_rank_score_contract_sha256s",
        "source_rank_numeric_contract",
        "evaluation400_membership_present",
        "evaluation400_hdf5_or_label_files_opened",
        "fresh_or_confirmation_files_opened",
        "performance_or_transfer_claim_authorized", "receipt_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != post_v3.evaluator.RECEIPT_FORMAT
        or value.get("status") != post_v3.evaluator.RECEIPT_STATUS
        or type(value.get("target_validation_groups")) is not int
        or value["target_validation_groups"] != 190
        or type(value.get("target_validation_samples")) is not int
        or value["target_validation_samples"] <= 0
        or type(value.get("target_validation_hdf5_files_opened")) is not int
        or value["target_validation_hdf5_files_opened"] != 190
        or value.get("target_validation_opened_after_five_adapters_frozen") is not True
        or value.get("evaluation400_membership_present") is not False
        or type(value.get("evaluation400_hdf5_or_label_files_opened")) is not int
        or value["evaluation400_hdf5_or_label_files_opened"] != 0
        or type(value.get("fresh_or_confirmation_files_opened")) is not int
        or value["fresh_or_confirmation_files_opened"] != 0
        or value.get("performance_or_transfer_claim_authorized") is not False
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not isinstance(
            value.get("source_rank_score_contract_sha256s"), list
        )
        or len(value["source_rank_score_contract_sha256s"]) != MEMBER_COUNT
        or any(
            not is_sha(item)
            for item in value["source_rank_score_contract_sha256s"]
        )
    ):
        raise PairedProtocolV3Error("formal190 evaluator receipt changed")
    return logical


def _validate_calibration_authority(
    value: Mapping[str, Any], members: Sequence[Mapping[str, Any]]
) -> str:
    logical = verify_hash_signature(
        value, "input_authority_sha256", "formal190 calibration authority"
    )
    expected_fields = {
        "format", "status", "lane", "member_count", "shared_contract",
        "prediction_contract", "validation_identity_set_sha256", "labels_path",
        "labels_file_sha256", "members", "test_artifacts_read",
        "fresh_artifacts_read", "confirmation_artifacts_read",
        "source_rank_numeric_contract",
        "input_authority_sha256",
    }
    shared = value.get("shared_contract")
    prediction = value.get("prediction_contract")
    rows = value.get("members")
    if (
        set(value) != expected_fields
        or value.get("format") != post_v3.calibrator.INPUT_FORMAT
        or value.get("status") != post_v3.calibrator.INPUT_STATUS
        or value.get("lane") != "validation_only"
        or type(value.get("member_count")) is not int or value["member_count"] != 5
        or not isinstance(shared, Mapping)
        or set(shared) != {
            "training_manifest_sha256", "split_sha256",
            "source_ensemble_contract_sha256", "prediction_contract_sha256",
        }
        or any(not is_sha(shared.get(field)) for field in shared)
        or not isinstance(prediction, Mapping)
        or canonical_sha256(prediction) != shared.get("prediction_contract_sha256")
        or prediction.get("recovery_target")
        != "conditional_recovery_given_operational_regress"
        or prediction.get("recovery_observation_mask")
        != "recovery_observed_and_regress"
        or prediction.get("recovery_shared_transition_stop_gradient") is not True
        or prediction.get("recovery_enters_primary_before_calibration") is not False
        or prediction.get("recovery_head_trained") is not True
        or value.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or not is_sha(value.get("validation_identity_set_sha256"))
        or not is_sha(value.get("labels_file_sha256"))
        or value.get("test_artifacts_read") is not False
        or value.get("fresh_artifacts_read") is not False
        or value.get("confirmation_artifacts_read") is not False
        or not isinstance(rows, list) or len(rows) != 5
    ):
        raise PairedProtocolV3Error("formal190 calibration authority changed")
    row_fields = {
        "member_index", "member_seed", "training_manifest_sha256", "split_sha256",
        "source_ensemble_contract_sha256", "prediction_contract_sha256",
        "checkpoint_path", "checkpoint_file_sha256",
        "validation_predictions_path", "validation_predictions_file_sha256",
        "source_rank_score_contract", "source_rank_score_contract_sha256",
    }
    for index, (row, member) in enumerate(zip(rows, members, strict=True)):
        if (
            not isinstance(row, Mapping) or set(row) != row_fields
            or type(row.get("member_index")) is not int or row["member_index"] != index
            or type(row.get("member_seed")) is not int
            or row["member_seed"] != member["member_seed"]
            or any(row.get(field) != shared.get(field) for field in shared)
            or row.get("checkpoint_path") != member["checkpoint_path"]
            or row.get("checkpoint_file_sha256") != member["checkpoint_file_sha256"]
            or row.get("source_rank_score_contract")
            != member["source_rank_score_contract"]
            or row.get("source_rank_score_contract", {}).get(
                "source_rank_numeric_contract"
            ) != value.get("source_rank_numeric_contract")
            or row.get("source_rank_score_contract_sha256")
            != member["source_rank_score_contract_sha256"]
            or not is_sha(row.get("validation_predictions_file_sha256"))
        ):
            raise PairedProtocolV3Error(f"formal190 calibration member {index} changed")
    return logical


def _validate_six_heads(
    calibration_value: Mapping[str, Any], head_value: Mapping[str, Any],
    ensemble_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        calibration = bridge_v2.validate_calibration(calibration_value)
        head = bridge_v2.validate_head_support(head_value)
        ensemble = bridge_v2.validate_ensemble_manifest(
            ensemble_value, calibration=calibration, head=head
        )
    except bridge_v2.Evaluation400BridgeError as error:
        raise PairedProtocolV3Error("formal190 calibration deployment failed") from error
    enabled = calibration_value.get("head_enabled_for_primary")
    heads = head_value.get("heads")
    recovery = heads.get("recovery") if isinstance(heads, Mapping) else None
    if (
        not isinstance(enabled, Mapping) or set(enabled) != set(HEAD_NAMES)
        or any(enabled.get(name) is not True for name in HEAD_NAMES)
        or not isinstance(heads, Mapping) or set(heads) != set(HEAD_NAMES)
        or not isinstance(recovery, Mapping)
        or recovery.get("support_threshold_met") is not True
        or recovery.get("all_member_recovery_heads_trained") is not True
        or calibration_value.get("recovery_temperature_fitted_on_validation_only") is not True
        or ensemble_value.get("conditional_recovery_semantics")
        != "p(recovery_given_operational_regress)"
        or ensemble_value.get("conditional_recovery_activation_requires_observed_regress") is not True
    ):
        raise PairedProtocolV3Error("all six formal190 heads including recovery are required")
    for name, row in heads.items():
        if not isinstance(row, Mapping):
            raise PairedProtocolV3Error(f"head support row changed: {name}")
        for field in (
            "independent_positive_or_observed_groups",
            "independent_negative_or_censored_groups", "minimum_required_per_side",
        ):
            if type(row.get(field)) is not int or row[field] < 0:
                raise PairedProtocolV3Error(f"head support numeric changed: {name}.{field}")
        if (
            row["independent_positive_or_observed_groups"] < row["minimum_required_per_side"]
            or row["independent_negative_or_censored_groups"] < row["minimum_required_per_side"]
            or row.get("enabled_for_primary") is not True
        ):
            raise PairedProtocolV3Error(f"head support insufficient: {name}")
    return calibration, head, ensemble


def _load_bridge(path: Path, expected_sha: str) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    bridge_path, value, _ = read_json(path, expected_sha, "evaluation400 identity bridge")
    try:
        bridge_v2.validate_bridge(value)
    except bridge_v2.Evaluation400BridgeError as error:
        raise PairedProtocolV3Error("evaluation400 bridge failed") from error
    dependencies = value.get("dependencies")
    expected_roles = {
        "target_manifest", "selected_identity_attestation", "ensemble_manifest",
        "calibration", "head_support", "calibration_receipt", "policy_bridge_receipt",
    }
    if not isinstance(dependencies, Mapping) or set(dependencies) != expected_roles:
        raise PairedProtocolV3Error("evaluation400 bridge dependency closure changed")
    loaded: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    files: dict[str, str] = {}
    for role in expected_roles:
        record = validate_record(dependencies[role], f"bridge {role}")
        bound_path, loaded[role], digest = read_json(
            Path(record["path"]), record["file_sha256"], f"bridge {role}"
        )
        paths[role] = bound_path
        files[role] = digest
    try:
        target = bridge_v2.validate_target_manifest(loaded["target_manifest"])
        if value.get("target_reset_runtime_contract_sha256") != target.get(
            "target_reset_runtime_contract_sha256"
        ):
            raise bridge_v2.Evaluation400BridgeError(
                "target reset runtime contract binding changed"
            )
        bridge_v2.validate_selected_identity_attestation(
            loaded["selected_identity_attestation"], decoded_manifest=target
        )
        calibration = bridge_v2.validate_calibration(loaded["calibration"])
        head = bridge_v2.validate_head_support(loaded["head_support"])
        ensemble = bridge_v2.validate_ensemble_manifest(
            loaded["ensemble_manifest"], calibration=calibration, head=head
        )
        bridge_v2.validate_calibration_receipt(
            loaded["calibration_receipt"], paths=paths, files=files,
            calibration=calibration, head=head, ensemble=ensemble,
        )
        bridge_v2.validate_policy_bridge(loaded["policy_bridge_receipt"])
    except bridge_v2.Evaluation400BridgeError as error:
        raise PairedProtocolV3Error("evaluation400 bridge dependency validation failed") from error
    pairs = value["pairs"]
    evaluation = target["evaluation"]
    if len(evaluation) != PAIR_COUNT:
        raise PairedProtocolV3Error("evaluation400 is not the unique 400-pair lane")
    for index, (pair, row) in enumerate(zip(pairs, evaluation, strict=True)):
        if (
            pair.get("ordinal") != index
            or pair.get("pair_id") != row.get("pair_id")
            or pair.get("requested_seed") != row.get("requested_seed")
            or pair.get("resolved_seed") != row.get("resolved_seed")
            or pair.get("initial_scene_state_sha256")
            != row.get("initial_scene_state_sha256")
            or pair.get("condition_order")
            != bridge_v2.paired_condition_order(str(row.get("pair_id")))
        ):
            raise PairedProtocolV3Error("evaluation400 pair identity differs from target manifest")
    return bridge_path, value, loaded


def _validate_handoff(
    value: Mapping[str, Any], *, terminal: Mapping[str, Any], plan_sha: str,
) -> str:
    logical = verify_hash_signature(value, "handoff_sha256", "post v3 handoff")
    expected_fields = {
        "format", "status", "post_v3_plan_sha256", "lineage",
        "identity_bridge_v2", "adapter_member_count",
        "adapter_member_receipt_sha256", "split_profile",
        "required_trainer_group_counts",
        "formal190_opened_by_independent_evaluator_after_five_frozen_adapters",
        "formal190_opened_by_watcher_process",
        "formal190_labels_opened_before_five_adapters_frozen",
        "evaluation400_membership_present",
        "evaluation400_hdf5_trajectory_or_labels_opened",
        "evaluation400_conditions_executed",
        "old_paired400_authority_waited_or_generated", "second_reserve400_created",
        "lobo_or_aggregate_checkpoint_used_for_adapter_training",
        "performance_or_transfer_claim_authorized", "handoff_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != post_v3.HANDOFF_FORMAT
        or value.get("status") != "ready_for_external_preoutcome_identity_bridge_v2_freeze"
        or value.get("post_v3_plan_sha256") != plan_sha
        or type(value.get("adapter_member_count")) is not int
        or value["adapter_member_count"] != 5
        or value.get("split_profile") != post_v3.SPLIT_PROFILE
        or value.get("required_trainer_group_counts")
        != {"train": 80, "validation": 30, "test": 190}
        or type(value.get("formal190_opened_by_independent_evaluator_after_five_frozen_adapters")) is not int
        or value["formal190_opened_by_independent_evaluator_after_five_frozen_adapters"] != 190
        or type(value.get("formal190_opened_by_watcher_process")) is not int
        or value["formal190_opened_by_watcher_process"] != 0
        or type(value.get("formal190_labels_opened_before_five_adapters_frozen")) is not int
        or value["formal190_labels_opened_before_five_adapters_frozen"] != 0
        or value.get("evaluation400_membership_present") is not False
        or type(value.get("evaluation400_hdf5_trajectory_or_labels_opened")) is not int
        or value["evaluation400_hdf5_trajectory_or_labels_opened"] != 0
        or type(value.get("evaluation400_conditions_executed")) is not int
        or value["evaluation400_conditions_executed"] != 0
        or value.get("old_paired400_authority_waited_or_generated") is not False
        or value.get("second_reserve400_created") is not False
        or value.get("lobo_or_aggregate_checkpoint_used_for_adapter_training") is not False
        or value.get("performance_or_transfer_claim_authorized") is not False
    ):
        raise PairedProtocolV3Error("post v3 handoff contract changed")
    terminal_handoff = terminal.get("identity_bridge_v2_handoff")
    if not isinstance(terminal_handoff, Mapping) or terminal_handoff.get("logical_sha256") != logical:
        raise PairedProtocolV3Error("post terminal does not bind this handoff")
    return logical


def _result_protocol() -> dict[str, Any]:
    return {
        "required_complete_pairs": PAIR_COUNT,
        "binary_success_values": [0, 1],
        "success_rate_difference": "mean(etsf_success-baseline_success)",
        "confidence_interval": {"method": "paired_percentile_bootstrap", "level_ppm": 950000},
        "paired_bootstrap": {
            "unit": "pair_id", "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES, "replacement": True,
            "lower_quantile_ppm": 25000, "upper_quantile_ppm": 975000,
        },
        "mcnemar": {
            "cells": ["n00", "n01", "n10", "n11"],
            "test": "exact_two_sided_binomial_on_n01_n10",
        },
        "posthoc_seed_candidate_threshold_or_subgroup_selection_allowed": False,
    }


def freeze_core(
    *, post_plan_path: Path, post_plan_file_sha256: str,
    post_terminal_path: Path, post_terminal_file_sha256: str,
    post_handoff_path: Path, post_handoff_file_sha256: str,
    member_receipts: Sequence[tuple[Path, str]],
    identity_bridge_path: Path, identity_bridge_file_sha256: str,
    execution_inventory_attestation_path: Path,
    execution_inventory_attestation_file_sha256: str,
    execution_inventory_attestation_sha256: str,
    expected_post_launcher_sha256: str,
    runtime_execution_authority_path: Path,
    runtime_execution_authority_file_sha256: str,
    selector_implementation_path: Path,
    selector_implementation_file_sha256: str,
) -> dict[str, Any]:
    _require_crypto()
    if (
        len(member_receipts) != MEMBER_COUNT
        or not is_sha(expected_post_launcher_sha256)
        or expected_post_launcher_sha256 != APPROVED_POST_V3_LAUNCHER_SHA256
    ):
        raise PairedProtocolV3Error("core external bindings are incomplete")
    inventory, issuer, inventory_record, stack_binding_sha256 = (
        _validate_execution_inventory(
            path=execution_inventory_attestation_path,
            expected_file_sha256=execution_inventory_attestation_file_sha256,
            expected_logical_sha256=execution_inventory_attestation_sha256,
        )
    )
    runtime_authority_path, runtime_authority, runtime_authority_sha = read_json(
        runtime_execution_authority_path,
        runtime_execution_authority_file_sha256,
        "schema6 runtime execution authority",
    )
    runtime_contract = runtime_authority.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise PairedProtocolV3Error(
            "schema6 runtime execution authority omits nested runtime contract"
        )
    nested_runtime_contract_sha = verify_hash_signature(
        runtime_contract,
        "runtime_contract_sha256",
        "schema6 nested runtime contract",
    )
    if (
        type(runtime_contract.get("max_episode_steps")) is not int
        or runtime_contract["max_episode_steps"] != 200
    ):
        raise PairedProtocolV3Error(
            "schema6 nested runtime contract must be exact 200 steps"
        )
    selector_path, selector_sha = hash_opaque_file(
        selector_implementation_path,
        selector_implementation_file_sha256,
        "evaluation400 root selector implementation",
    )
    if selector_path.suffix.casefold() != ".py":
        raise PairedProtocolV3Error("root selector implementation must be Python")
    public_key = _decode_hex(
        issuer["issuer_public_key_hex"], 32, "issuer Ed25519 public key"
    )
    root = Path(os.path.abspath(os.path.expanduser(os.fspath(post_terminal_path)))).parent
    plan_path, plan, _ = read_json(
        post_plan_path, post_plan_file_sha256, "post v3 static plan"
    )
    plan_sha = _validate_post_plan(
        plan, path=plan_path, root=root,
        expected_launcher_sha256=expected_post_launcher_sha256,
    )
    terminal_path, terminal, _ = read_json(
        post_terminal_path, post_terminal_file_sha256, "post v3 terminal"
    )
    terminal_sha = _validate_post_terminal(root, terminal)
    if terminal.get("plan_sha256") != plan_sha:
        raise PairedProtocolV3Error("post terminal/static plan mismatch")
    handoff_path, handoff, _ = read_json(
        post_handoff_path, post_handoff_file_sha256, "post v3 handoff"
    )
    handoff_sha = _validate_handoff(handoff, terminal=terminal, plan_sha=plan_sha)
    terminal_handoff = terminal["identity_bridge_v2_handoff"]
    if (
        terminal_path != root / "final_receipt.json"
        or handoff_path != root / "handoff" / "evaluation400_identity_bridge_v2_handoff.json"
        or terminal_handoff.get("path") != str(handoff_path)
        or terminal_handoff.get("file_sha256") != post_handoff_file_sha256
    ):
        raise PairedProtocolV3Error("post terminal/handoff path closure changed")

    member_values: list[dict[str, Any]] = []
    member_records: list[dict[str, Any]] = []
    for index, ((path, expected_sha), expected_seed) in enumerate(
        zip(member_receipts, post_v3.SOURCE_MEMBER_SEEDS, strict=True)
    ):
        bound_path, member, digest = read_json(path, expected_sha, f"member {index} receipt")
        logical = _validate_member(member, index=index, expected_seed=expected_seed)
        if bound_path != root / "members" / f"member_{index}" / "final_receipt.json":
            raise PairedProtocolV3Error("member receipt is outside exact post root")
        source_path, source_sha = hash_opaque_file(
            Path(str(member["source_checkpoint_path"])),
            member["source_checkpoint_sha256"], f"r7h source member {index}",
        )
        adapter_path, adapter_sha = hash_opaque_file(
            Path(str(member["checkpoint_path"])),
            member["checkpoint_file_sha256"], f"target adapter member {index}",
        )
        member_values.append(member)
        member_records.append({
            "member_index": index, "member_seed": expected_seed,
            "receipt": {"path": str(bound_path), "file_sha256": digest, "logical_sha256": logical},
            "source_checkpoint": {"path": str(source_path), "file_sha256": source_sha},
            "adapter_checkpoint": {"path": str(adapter_path), "file_sha256": adapter_sha},
            "recovery_head_trained": True,
            "source_rank_score_contract_sha256": member[
                "source_rank_score_contract_sha256"
            ],
            "object_source_normalization_sha256": member[
                "prediction_contract"
            ]["object_source_normalization_sha256"],
        })
    if (
        len({row["source_checkpoint"]["file_sha256"] for row in member_records}) != 5
        or len({row["adapter_checkpoint"]["file_sha256"] for row in member_records}) != 5
        or handoff.get("adapter_member_receipt_sha256")
        != [row["receipt"]["logical_sha256"] for row in member_records]
        or terminal.get("r7h_member_checkpoint_sha256")
        != [row["source_checkpoint"]["file_sha256"] for row in member_records]
    ):
        raise PairedProtocolV3Error("r7h/target five-member lineage changed")

    lineage = handoff.get("lineage")
    expected_lineage_fields = {
        "r7h_source_final", "r7h_member_checkpoint_sha256", "r7h_member_seed",
        "r8e_root", "r8e_final", "r8e_summary_sha256", "r9b_final",
        "development300_terminal", "materializer_v3_receipt",
        "formal190_evaluator_authority", "formal190_evaluator_receipt",
        "formal190_global_one_shot_claim",
    }
    if not isinstance(lineage, Mapping) or set(lineage) != expected_lineage_fields:
        raise PairedProtocolV3Error("post v3 handoff lineage missing")
    formal190_claim = _validate_formal190_claim(
        lineage["formal190_global_one_shot_claim"],
        terminal_descriptor=terminal.get("formal190_global_one_shot_claim"),
    )
    claim_path = Path(str(formal190_claim["path"]))
    _claim_path, claim_value, _claim_file_sha = read_json(
        claim_path, formal190_claim["file_sha256"], "formal190 one-shot claim cross-check"
    )
    if (
        claim_value.get("post_v3_plan_sha256") != plan_sha
        or claim_value.get("post_v3_output_root") != str(root)
        or claim_value.get("development300_terminal_file_sha256")
        != plan.get("development300_terminal", {}).get("file_sha256")
        or claim_value.get("development300_terminal_sha256")
        != plan.get("development300_terminal", {}).get("logical_sha256")
    ):
        raise PairedProtocolV3Error("formal190 one-shot claim lineage mismatch")
    source_final_record = validate_record(lineage.get("r7h_source_final"), "r7h source final")
    source_final_path, source_final, _ = read_json(
        Path(source_final_record["path"]), source_final_record["file_sha256"],
        "r7h source final",
    )
    source_final_logical = verify_hash_signature(source_final, "receipt_sha256", "r7h source final")
    if (
        source_final.get("format") != SOURCE_FORMAT or source_final.get("status") != SOURCE_STATUS
        or source_final_logical != source_final_record["logical_sha256"]
        or lineage.get("r7h_member_seed") != list(post_v3.SOURCE_MEMBER_SEEDS)
        or lineage.get("r7h_member_checkpoint_sha256")
        != [row["source_checkpoint"]["file_sha256"] for row in member_records]
        or any(source_final.get(field) is not False for field in (
            "target_data_read", "target_labels_read", "fresh_inputs_accepted",
            "fresh_labels_read", "test_labels_used",
        ))
        or type(source_final.get("test_hdf_label_datasets_opened")) is not int
        or source_final["test_hdf_label_datasets_opened"] != 0
    ):
        raise PairedProtocolV3Error("r7h source terminal lineage changed")

    evaluator_authority_record = validate_record(
        lineage.get("formal190_evaluator_authority"), "formal190 evaluator authority"
    )
    eval_auth_path, eval_auth, _ = read_json(
        Path(evaluator_authority_record["path"]),
        evaluator_authority_record["file_sha256"], "formal190 evaluator authority",
    )
    eval_auth_sha = _validate_evaluator_authority(eval_auth, member_values)
    if eval_auth_sha != evaluator_authority_record["logical_sha256"]:
        raise PairedProtocolV3Error("formal190 evaluator authority logical SHA changed")
    formal_record = validate_record(
        lineage.get("formal190_evaluator_receipt"), "formal190 evaluator receipt"
    )
    formal_path, formal_receipt, _ = read_json(
        Path(formal_record["path"]), formal_record["file_sha256"],
        "formal190 evaluator receipt",
    )
    formal_sha = _validate_formal_receipt(formal_receipt)
    if (
        formal_sha != formal_record["logical_sha256"]
        or formal_receipt.get("input_authority_path") != str(eval_auth_path)
        or formal_receipt.get("input_authority_file_sha256")
        != evaluator_authority_record["file_sha256"]
        or formal_receipt.get("input_authority_sha256") != eval_auth_sha
        or formal_receipt.get("source_rank_score_contract_sha256s")
        != [
            member["source_rank_score_contract_sha256"]
            for member in member_values
        ]
    ):
        raise PairedProtocolV3Error("formal190 evaluator authority/receipt mismatch")
    calibration_authority_path, calibration_authority, _ = read_json(
        Path(str(formal_receipt["calibration_input_authority_path"])),
        str(formal_receipt["calibration_input_authority_file_sha256"]),
        "formal190 calibration authority",
    )
    calibration_authority_sha = _validate_calibration_authority(
        calibration_authority, member_values
    )
    if calibration_authority_sha != formal_receipt.get(
        "calibration_input_authority_sha256"
    ):
        raise PairedProtocolV3Error("formal190 calibration authority logical SHA changed")

    bridge_path, identity_bridge, bridge_dependencies = _load_bridge(
        identity_bridge_path, identity_bridge_file_sha256
    )
    target_reset_runtime_contract_sha256 = identity_bridge.get(
        "target_reset_runtime_contract_sha256"
    )
    if (
        not is_sha(target_reset_runtime_contract_sha256)
        or nested_runtime_contract_sha != target_reset_runtime_contract_sha256
    ):
        raise PairedProtocolV3Error(
            "schema6 runtime authority differs from target reset runtime contract"
        )
    identity_handoff = handoff.get("identity_bridge_v2")
    produced = identity_handoff.get("produced_dependencies") if isinstance(identity_handoff, Mapping) else None
    if not isinstance(produced, Mapping) or set(produced) != {
        "ensemble_manifest", "calibration", "head_support", "calibration_receipt"
    }:
        raise PairedProtocolV3Error("post handoff calibration closure changed")
    for role in produced:
        descriptor = validate_record(produced[role], f"handoff {role}")
        bridge_descriptor = identity_bridge["dependencies"][role]
        if descriptor != bridge_descriptor:
            raise PairedProtocolV3Error(f"handoff/bridge dependency mismatch: {role}")
    calibration_value = bridge_dependencies["calibration"]
    head_value = bridge_dependencies["head_support"]
    ensemble_value = bridge_dependencies["ensemble_manifest"]
    calibration, head, ensemble = _validate_six_heads(
        calibration_value, head_value, ensemble_value
    )
    calibration_receipt = bridge_dependencies["calibration_receipt"]
    if (
        calibration_receipt.get("input_authority_path")
        != str(calibration_authority_path)
        or calibration_receipt.get("input_authority_file_sha256")
        != formal_receipt.get("calibration_input_authority_file_sha256")
        or calibration_receipt.get("input_authority_sha256")
        != formal_receipt.get("calibration_input_authority_sha256")
        or calibration_receipt.get("validation_only") is not True
    ):
        raise PairedProtocolV3Error("formal190 evaluator/calibrator handoff changed")
    threshold = _fixed_point(
        calibration_value["abstain_threshold"]["maximum_total_uncertainty"],
        "formal190 abstention threshold",
    )
    if ensemble_value.get("members") != [
        {
            "member_index": index,
            "member_seed": member["member_seed"],
            "checkpoint_path": member["checkpoint_path"],
            "checkpoint_file_sha256": member["checkpoint_file_sha256"],
            "source_rank_score_contract": member[
                "source_rank_score_contract"
            ],
            "source_rank_score_contract_sha256": member[
                "source_rank_score_contract_sha256"
            ],
        }
        for index, member in enumerate(member_values)
    ]:
        raise PairedProtocolV3Error("calibrated ensemble differs from five member v3 receipts")

    bridge_deployment = identity_bridge.get("deployment")
    bridge_selector = (
        bridge_deployment.get("selector_authority")
        if isinstance(bridge_deployment, Mapping) else None
    )
    if (
        not isinstance(bridge_selector, Mapping)
        or bridge_selector.get("calibration_sha256")
        != calibration["calibration_sha256"]
        or bridge_selector.get("formal190_root_group_ranker_sha256")
        != calibration["root_group_ranker_sha256"]
        or bridge_selector.get("source_rank_score_contract_sha256")
        != [
            member["source_rank_score_contract_sha256"]
            for member in member_values
        ]
        or bridge_selector.get("source_rank_score_contracts")
        != [
            member["source_rank_score_contract"] for member in member_values
        ]
        or bridge_selector.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or bridge_selector.get("source_rank_member_authority")
        != calibration["source_rank_member_authority"]
        or bridge_selector.get("source_rank_member_authority_sha256")
        != calibration["source_rank_member_authority_sha256"]
        or bridge_selector.get("deployment_parameters") != {
            **calibration["deployment_parameters"],
            "deployment_uncertainty_contract_sha256": calibration[
                "deployment_uncertainty_contract_sha256"
            ],
        }
        or bridge_selector.get("formal190_thresholds") != {
            "minimum_formal190_composite_margin": calibration[
                "minimum_group_relative_composite_rank_score_margin"
            ],
            "maximum_formal190_pair_uncertainty": calibration[
                "maximum_structured_pair_uncertainty"
            ],
            "maximum_global_total_uncertainty": calibration[
                "maximum_total_uncertainty"
            ],
            "root_group_ranker_sha256": calibration[
                "root_group_ranker_sha256"
            ],
        }
    ):
        raise PairedProtocolV3Error("identity bridge selector calibration changed")
    bridge_uncertainty = bridge_selector.get("uncertainty_contract")
    uncertainty_implementation = bridge_selector.get(
        "deployment_uncertainty_implementation"
    )
    if (
        not isinstance(bridge_uncertainty, Mapping)
        or not isinstance(uncertainty_implementation, Mapping)
        or set(uncertainty_implementation) != {"path", "file_sha256"}
        or bridge_uncertainty.get("deployment_uncertainty_contract_sha256")
        != calibration["deployment_uncertainty_contract_sha256"]
    ):
        raise PairedProtocolV3Error("identity bridge uncertainty binding changed")
    selector_authority_base = {
        "format": "etsf_smolvla_piper_evaluation400_root_selector_authority_v3",
        "status": "frozen_formal190_runtime_bound_composite_selector",
        "implementation": {
            "path": str(selector_path), "file_sha256": selector_sha,
        },
        "utility_contract": {
            "primary_score": (
                "five_member_adjusted_source_composite_candidate_rank_score_margin"
            ),
            "primary_score_is_success_logit": False,
            "primary_score_is_success_probability": False,
            "scene_relative_candidate_comparison": True,
            "source_action_rank_residual_required": True,
            "source_action_rank_success_only": False,
            "piper_embodiment_adapter_required": True,
            "formal190_target_outcome_calibrated_acceptance_margin": True,
            "structured_heads_enter_primary_utility": False,
            "structured_heads_enter_uncertainty_and_ablation": True,
            "margin_comparison": "strict_greater_than_formal190_threshold",
            "alternative_set_contract": (
                "all_legal_candidates_except_lowest_legal_baseline"
            ),
        },
        "uncertainty_contract": {
            **dict(bridge_uncertainty),
            "calibration_scale_exact": True,
            "aleatoric_and_epistemic_guard_only": True,
        },
        "runtime_execution_authority_sha256": runtime_authority_sha,
        "deployment_uncertainty_implementation": dict(
            uncertainty_implementation
        ),
        "five_member_checkpoint_sha256": [
            member["checkpoint_file_sha256"] for member in member_values
        ],
        "calibration_sha256": calibration["calibration_sha256"],
        "source_rank_score_contract_sha256": [
            member["source_rank_score_contract_sha256"]
            for member in member_values
        ],
        "source_rank_score_contracts": [
            dict(member["source_rank_score_contract"])
            for member in member_values
        ],
        "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": dict(
            bridge_selector["source_rank_member_authority"]
        ),
        "source_rank_member_authority_sha256": bridge_selector[
            "source_rank_member_authority_sha256"
        ],
        "deployment_parameters": dict(
            bridge_selector["deployment_parameters"]
        ),
        "formal190_thresholds": dict(
            bridge_selector["formal190_thresholds"]
        ),
        "object_source_normalization_sha256": [
            member["prediction_contract"][
                "object_source_normalization_sha256"
            ]
            for member in member_values
        ],
        "formal190_root_group_ranker_sha256": calibration[
            "root_group_ranker_sha256"
        ],
    }
    selector_authority = {
        **selector_authority_base,
        "selector_authority_sha256": post_v3.canonical_sha256(
            selector_authority_base
        ),
    }

    pair_rows = [{
        "ordinal": row["ordinal"], "pair_id": row["pair_id"],
        "target_manifest_global_ordinal": row["target_manifest_global_ordinal"],
        "requested_seed": row["requested_seed"], "resolved_seed": row["resolved_seed"],
        "initial_scene_state_sha256": row["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": row["initial_measured_joint_state_sha256"],
        "initial_commanded_drive_target_sha256": row["initial_commanded_drive_target_sha256"],
        "condition_order": list(row["condition_order"]),
        "candidate_count": 4,
    } for row in identity_bridge["pairs"]]
    base: dict[str, Any] = {
        "format": CORE_FORMAT, "status": CORE_STATUS,
        "post_collection_v3": {
            "reviewed_launcher_sha256": expected_post_launcher_sha256,
            "static_plan": {"path": str(plan_path), "file_sha256": post_plan_file_sha256, "logical_sha256": plan_sha},
            "terminal": {"path": str(terminal_path), "file_sha256": post_terminal_file_sha256, "logical_sha256": terminal_sha},
            "handoff": {"path": str(handoff_path), "file_sha256": post_handoff_file_sha256, "logical_sha256": handoff_sha},
        },
        "development_and_formal190": {
            "split_profile": "development300_v3",
            "group_counts": {"train": 80, "internal_validation": 30, "formal_validation": 190},
            "formal190_evaluator_authority": evaluator_authority_record,
            "formal190_evaluator_receipt": formal_record,
            "formal190_global_one_shot_claim": formal190_claim,
            "formal190_calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
            "formal190_calibration_sha256": calibration["calibration_sha256"],
            "formal190_head_support_sha256": head["head_support_sha256"],
            "formal190_ensemble_manifest_sha256": ensemble["ensemble_manifest_sha256"],
            "formal190_root_group_ranker_sha256": calibration[
                "root_group_ranker_sha256"
            ],
            "formal190_deployment_uncertainty_contract_sha256": calibration[
                "deployment_uncertainty_contract_sha256"
            ],
            "source_rank_score_contract_sha256": [
                member["source_rank_score_contract_sha256"]
                for member in member_values
            ],
            "source_rank_score_contracts": [
                dict(member["source_rank_score_contract"])
                for member in member_values
            ],
            "source_rank_member_authority": dict(
                selector_authority["source_rank_member_authority"]
            ),
            "source_rank_member_authority_sha256": selector_authority[
                "source_rank_member_authority_sha256"
            ],
            "deployment_parameters": dict(
                selector_authority["deployment_parameters"]
            ),
            "formal190_thresholds": dict(
                selector_authority["formal190_thresholds"]
            ),
            "all_six_heads_primary": list(HEAD_NAMES),
            "maximum_total_uncertainty_fixed_point": threshold,
            "minimum_composite_margin_fixed_point": _fixed_point(
                calibration[
                    "minimum_group_relative_composite_rank_score_margin"
                ],
                "formal190 composite rank margin",
            ),
            "maximum_pair_uncertainty_fixed_point": _fixed_point(
                calibration["maximum_structured_pair_uncertainty"],
                "formal190 pair uncertainty",
            ),
            "object_error_robust_scale_m_fixed_point": _fixed_point(
                calibration["object_error_robust_scale_m"],
                "formal190 object robust scale",
            ),
        },
        "r7h_target_adapter_lineage": {
            "source_terminal": {"path": str(source_final_path), **{k: source_final_record[k] for k in ("file_sha256", "logical_sha256")}},
            "member_count": 5, "members": member_records,
            "single_checkpoint_accepted": False, "lobo_checkpoint_accepted": False,
            "joint_teacher_accepted": False,
        },
        "evaluation400": {
            "identity_bridge": {"path": str(bridge_path), "file_sha256": identity_bridge_file_sha256, "logical_sha256": identity_bridge["bridge_sha256"]},
            "pair_identity_set_sha256": identity_bridge["pair_identity_set_sha256"],
            "pair_count": 400, "only_final_paired_lane": True,
            "additional_reserve400_count": 0,
            "postfreeze_seed_candidate_threshold_or_order_change_allowed": False,
            "pairs": pair_rows,
        },
        "deployment": {
            "deployment_binding_sha256": identity_bridge["deployment"]["deployment_binding_sha256"],
            "policy_runtime_action_binding_sha256": canonical_sha256({
                key: identity_bridge["deployment"][key]
                for key in (
                    "bridge_contract_sha256", "runtime_binding_sha256",
                    "state_feature_binding_sha256", "action_mapping_binding_sha256",
                )
            }),
            "candidate_count": 4,
            "baseline_selector": "lowest_legal_feasibility_root_candidate",
            "etsf_selector": "frozen_five_member_event_world_model_with_uncertainty_abstention",
            "fallback": "baseline",
            "runtime_execution_authority": {
                "path": str(runtime_authority_path),
                "file_sha256": runtime_authority_sha,
                "nested_runtime_contract_sha256": nested_runtime_contract_sha,
                "max_episode_steps": 200,
            },
            "target_reset_runtime_contract_sha256": (
                target_reset_runtime_contract_sha256
            ),
            "selector_authority": selector_authority,
            "selector_authority_sha256": selector_authority[
                "selector_authority_sha256"
            ],
            "formal190_root_group_ranker_sha256": calibration[
                "root_group_ranker_sha256"
            ],
            "deployment_uncertainty_contract_sha256": calibration[
                "deployment_uncertainty_contract_sha256"
            ],
            "source_rank_score_contract_sha256": [
                member["source_rank_score_contract_sha256"]
                for member in member_values
            ],
            "source_rank_score_contracts": [
                dict(member["source_rank_score_contract"])
                for member in member_values
            ],
            "source_rank_member_authority": dict(
                selector_authority["source_rank_member_authority"]
            ),
            "source_rank_member_authority_sha256": selector_authority[
                "source_rank_member_authority_sha256"
            ],
            "deployment_parameters": dict(
                selector_authority["deployment_parameters"]
            ),
            "formal190_thresholds": dict(
                selector_authority["formal190_thresholds"]
            ),
        },
        "execution_inventory": {
            "attestation": inventory_record,
            "stack_binding_sha256": stack_binding_sha256,
            "executor_identity_sha256": inventory["executor"]["identity_sha256"],
            "executor_implementation_file_sha256": inventory["executor"][
                "implementation"
            ]["file_sha256"],
            "result_evaluator_identity_sha256": inventory["result_evaluator"][
                "identity_sha256"
            ],
            "result_evaluator_implementation_file_sha256": inventory[
                "result_evaluator"
            ]["implementation"]["file_sha256"],
            "real_execution_components_complete": True,
        },
        "authority_policy": {
            "signature_algorithm": "Ed25519",
            "signature_context_utf8": SIGNATURE_CONTEXT[:-1].decode("utf-8"),
            "issuer_key_id": issuer["issuer_key_id"],
            "issuer_public_key_hex": public_key.hex(),
            "issuer_public_key_sha256": hashlib.sha256(public_key).hexdigest(),
            "issuer_identity_sha256": issuer["issuer_identity_sha256"],
            "trusted_issuer_attestation_sha256": issuer["attestation_sha256"],
            "executor_identity_sha256": inventory["executor"]["identity_sha256"],
            "result_evaluator_identity_sha256": inventory["result_evaluator"][
                "identity_sha256"
            ],
            "authorization_sequence": 1,
            "core_itself_authorizes_execution": False,
        },
        "result_protocol": _result_protocol(),
        "preexecution_capability_receipt": {
            "hdf5_files_opened": 0, "trajectory_files_opened": 0,
            "prediction_files_opened": 0, "label_or_outcome_files_opened": 0,
            "checkpoint_files_hashed_as_opaque_bytes": 10,
            "checkpoint_deserialization_calls": 0, "policy_or_simulator_calls": 0,
            "pair_conditions_executed": 0,
        },
        "execution_authorized": False,
    }
    return {**base, "protocol_core_sha256": canonical_sha256(base)}


def validate_core(value: Mapping[str, Any]) -> str:
    logical = verify_hash_signature(value, "protocol_core_sha256", "protocol core v3")
    evaluation = value.get("evaluation400")
    lineage = value.get("r7h_target_adapter_lineage")
    authority = value.get("authority_policy")
    inventory = value.get("execution_inventory")
    capability = value.get("preexecution_capability_receipt")
    deployment = value.get("deployment")
    development = value.get("development_and_formal190")
    selector = (
        deployment.get("selector_authority")
        if isinstance(deployment, Mapping) else None
    )
    if (
        set(value) != {
            "format", "status", "post_collection_v3",
            "development_and_formal190", "r7h_target_adapter_lineage",
            "evaluation400", "deployment", "execution_inventory",
            "authority_policy", "result_protocol",
            "preexecution_capability_receipt", "execution_authorized",
            "protocol_core_sha256",
        }
        or value.get("format") != CORE_FORMAT or value.get("status") != CORE_STATUS
        or value.get("execution_authorized") is not False
        or not isinstance(deployment, Mapping)
        or set(deployment) != {
            "deployment_binding_sha256",
            "policy_runtime_action_binding_sha256", "candidate_count",
            "baseline_selector", "etsf_selector", "fallback",
            "runtime_execution_authority", "selector_authority",
            "selector_authority_sha256",
            "formal190_root_group_ranker_sha256",
            "deployment_uncertainty_contract_sha256",
            "source_rank_score_contract_sha256",
            "source_rank_score_contracts", "source_rank_member_authority",
            "source_rank_member_authority_sha256", "deployment_parameters",
            "formal190_thresholds", "target_reset_runtime_contract_sha256",
        }
        or type(deployment.get("candidate_count")) is not int
        or deployment["candidate_count"] != 4
        or deployment.get("baseline_selector")
        != "lowest_legal_feasibility_root_candidate"
        or deployment.get("fallback") != "baseline"
        or not isinstance(
            deployment.get("runtime_execution_authority"), Mapping
        )
        or set(deployment["runtime_execution_authority"])
        != {
            "path", "file_sha256", "nested_runtime_contract_sha256",
            "max_episode_steps",
        }
        or not is_sha(
            deployment["runtime_execution_authority"].get("file_sha256")
        )
        or type(
            deployment["runtime_execution_authority"].get("max_episode_steps")
        ) is not int
        or deployment["runtime_execution_authority"]["max_episode_steps"] != 200
        or not is_sha(deployment.get("target_reset_runtime_contract_sha256"))
        or deployment["runtime_execution_authority"].get(
            "nested_runtime_contract_sha256"
        ) != deployment.get("target_reset_runtime_contract_sha256")
        or not isinstance(selector, Mapping)
        or set(selector) != {
            "format", "status", "implementation", "utility_contract",
            "uncertainty_contract", "runtime_execution_authority_sha256",
            "deployment_uncertainty_implementation",
            "five_member_checkpoint_sha256", "calibration_sha256",
            "source_rank_score_contract_sha256",
            "source_rank_score_contracts", "source_rank_numeric_contract",
            "source_rank_member_authority",
            "source_rank_member_authority_sha256",
            "deployment_parameters",
            "formal190_thresholds",
            "object_source_normalization_sha256",
            "formal190_root_group_ranker_sha256",
            "selector_authority_sha256",
        }
        or verify_hash_signature(
            selector, "selector_authority_sha256", "selector authority"
        ) != deployment.get("selector_authority_sha256")
        or selector.get("runtime_execution_authority_sha256")
        != deployment["runtime_execution_authority"]["file_sha256"]
        or selector.get("formal190_root_group_ranker_sha256")
        != deployment.get("formal190_root_group_ranker_sha256")
        or selector.get("source_rank_score_contract_sha256")
        != deployment.get("source_rank_score_contract_sha256")
        or selector.get("source_rank_score_contracts")
        != deployment.get("source_rank_score_contracts")
        or selector.get("source_rank_numeric_contract")
        != SOURCE_RANK_NUMERIC_CONTRACT
        or selector.get("source_rank_member_authority")
        != deployment.get("source_rank_member_authority")
        or selector.get("source_rank_member_authority_sha256")
        != deployment.get("source_rank_member_authority_sha256")
        or selector.get("deployment_parameters")
        != deployment.get("deployment_parameters")
        or selector.get("formal190_thresholds")
        != deployment.get("formal190_thresholds")
        or not isinstance(development, Mapping)
        or development.get("source_rank_score_contracts")
        != deployment.get("source_rank_score_contracts")
        or development.get("source_rank_member_authority")
        != deployment.get("source_rank_member_authority")
        or development.get("source_rank_member_authority_sha256")
        != deployment.get("source_rank_member_authority_sha256")
        or development.get("deployment_parameters")
        != deployment.get("deployment_parameters")
        or development.get("formal190_thresholds")
        != deployment.get("formal190_thresholds")
        or development.get("formal190_root_group_ranker_sha256")
        != deployment.get("formal190_root_group_ranker_sha256")
        or development.get(
            "formal190_deployment_uncertainty_contract_sha256"
        ) != deployment.get("deployment_uncertainty_contract_sha256")
        or not is_sha(deployment.get("formal190_root_group_ranker_sha256"))
        or not is_sha(deployment.get("deployment_uncertainty_contract_sha256"))
        or not isinstance(deployment.get("deployment_parameters"), Mapping)
        or set(deployment["deployment_parameters"]) != {
            "post_event_temperature", "next_event_temperature",
            "success_temperature", "conditional_recovery_temperature",
            "duration_scale_multiplier", "object_scale_multiplier",
            "object_error_robust_scale_m",
            "deployment_uncertainty_contract_sha256",
        }
        or not isinstance(deployment.get("formal190_thresholds"), Mapping)
        or set(deployment["formal190_thresholds"]) != {
            "minimum_formal190_composite_margin",
            "maximum_formal190_pair_uncertainty",
            "maximum_global_total_uncertainty",
            "root_group_ranker_sha256",
        }
        or not is_sha(
            deployment["deployment_parameters"].get(
                "deployment_uncertainty_contract_sha256"
            )
        )
        or deployment["formal190_thresholds"].get("root_group_ranker_sha256")
        != deployment.get("formal190_root_group_ranker_sha256")
        or _fixed_point(
            deployment["formal190_thresholds"].get(
                "maximum_global_total_uncertainty"
            ),
            "core global uncertainty threshold",
        ) != development.get("maximum_total_uncertainty_fixed_point")
        or selector.get("uncertainty_contract", {}).get(
            "deployment_uncertainty_contract_sha256"
        ) != deployment.get("deployment_uncertainty_contract_sha256")
        or not isinstance(evaluation, Mapping)
        or type(evaluation.get("pair_count")) is not int or evaluation["pair_count"] != 400
        or evaluation.get("only_final_paired_lane") is not True
        or type(evaluation.get("additional_reserve400_count")) is not int
        or evaluation["additional_reserve400_count"] != 0
        or not isinstance(evaluation.get("pairs"), list) or len(evaluation["pairs"]) != 400
        or not isinstance(lineage, Mapping)
        or type(lineage.get("member_count")) is not int or lineage["member_count"] != 5
        or lineage.get("single_checkpoint_accepted") is not False
        or lineage.get("lobo_checkpoint_accepted") is not False
        or lineage.get("joint_teacher_accepted") is not False
        or not isinstance(inventory, Mapping)
        or set(inventory) != {
            "attestation", "stack_binding_sha256", "executor_identity_sha256",
            "executor_implementation_file_sha256",
            "result_evaluator_identity_sha256",
            "result_evaluator_implementation_file_sha256",
            "real_execution_components_complete",
        }
        or not isinstance(inventory.get("attestation"), Mapping)
        or set(inventory["attestation"]) != RECORD_FIELDS
        or any(not is_sha(inventory.get(field)) for field in (
            "stack_binding_sha256", "executor_identity_sha256",
            "executor_implementation_file_sha256",
            "result_evaluator_identity_sha256",
            "result_evaluator_implementation_file_sha256",
        ))
        or inventory.get("real_execution_components_complete") is not True
        or not isinstance(authority, Mapping)
        or set(authority) != {
            "signature_algorithm", "signature_context_utf8", "issuer_key_id",
            "issuer_public_key_hex", "issuer_public_key_sha256",
            "issuer_identity_sha256", "trusted_issuer_attestation_sha256",
            "executor_identity_sha256", "result_evaluator_identity_sha256",
            "authorization_sequence", "core_itself_authorizes_execution",
        }
        or authority.get("signature_algorithm") != "Ed25519"
        or authority.get("signature_context_utf8")
        != SIGNATURE_CONTEXT[:-1].decode("utf-8")
        or not isinstance(authority.get("issuer_key_id"), str)
        or not authority["issuer_key_id"]
        or not is_sha(authority.get("issuer_public_key_sha256"))
        or not is_sha(authority.get("issuer_identity_sha256"))
        or type(authority.get("authorization_sequence")) is not int
        or authority["authorization_sequence"] != 1
        or authority.get("core_itself_authorizes_execution") is not False
        or authority.get("executor_identity_sha256")
        != inventory.get("executor_identity_sha256")
        or authority.get("result_evaluator_identity_sha256")
        != inventory.get("result_evaluator_identity_sha256")
        or not is_sha(authority.get("trusted_issuer_attestation_sha256"))
        or not isinstance(capability, Mapping)
        or set(capability) != {
            "hdf5_files_opened", "trajectory_files_opened",
            "prediction_files_opened", "label_or_outcome_files_opened",
            "checkpoint_files_hashed_as_opaque_bytes",
            "checkpoint_deserialization_calls", "policy_or_simulator_calls",
            "pair_conditions_executed",
        }
        or any(type(capability.get(field)) is not int or capability[field] != 0 for field in (
            "hdf5_files_opened", "trajectory_files_opened", "prediction_files_opened",
            "label_or_outcome_files_opened", "checkpoint_deserialization_calls",
            "policy_or_simulator_calls", "pair_conditions_executed",
        ))
        or type(capability.get("checkpoint_files_hashed_as_opaque_bytes")) is not int
        or capability["checkpoint_files_hashed_as_opaque_bytes"] != 10
        or value.get("result_protocol") != _result_protocol()
    ):
        raise PairedProtocolV3Error("protocol core v3 boundary changed")
    contracts = deployment["source_rank_score_contracts"]
    contract_shas = deployment["source_rank_score_contract_sha256"]
    try:
        member_authority = bridge_v2.validate_source_rank_member_authority(
            deployment.get("source_rank_member_authority"),
            deployment.get("source_rank_member_authority_sha256"),
            role="protocol core Source rank member authority",
        )
    except bridge_v2.Evaluation400BridgeError as error:
        raise PairedProtocolV3Error(
            "source rank member authority closure changed"
        ) from error
    if (
        not isinstance(contracts, list)
        or not isinstance(contract_shas, list)
        or len(contracts) != MEMBER_COUNT
        or len(contract_shas) != MEMBER_COUNT
        or len(set(contract_shas)) != MEMBER_COUNT
    ):
        raise PairedProtocolV3Error("source rank contract closure changed")
    for index, contract in enumerate(contracts):
        if (
            not isinstance(contract, Mapping)
            or verify_hash_signature(
                contract,
                "contract_sha256",
                f"core source rank contract {index}",
            ) != contract_shas[index]
            or isinstance(contract.get("success_temperature"), bool)
            or not isinstance(contract.get("success_temperature"), (int, float))
            or Decimal(str(contract["success_temperature"])) <= 0
            or not Decimal(str(contract["success_temperature"])).is_finite()
        ):
            raise PairedProtocolV3Error("source rank contract closure changed")
        authority_row = member_authority["source_rank_member_authority"][
            "members"
        ][index]
        if (
            authority_row["member_index"] != index
            or authority_row["source_checkpoint_file_sha256"]
            != contract.get("source_checkpoint_file_sha256")
            or authority_row["source_rank_score_contract_sha256"]
            != contract_shas[index]
            or Decimal(str(authority_row["success_temperature"]))
            != Decimal(str(contract["success_temperature"]))
        ):
            raise PairedProtocolV3Error(
                "source rank member authority closure changed"
            )
    parameters = deployment["deployment_parameters"]
    for field in (
        "post_event_temperature", "next_event_temperature",
        "success_temperature", "conditional_recovery_temperature",
        "duration_scale_multiplier", "object_scale_multiplier",
        "object_error_robust_scale_m",
    ):
        numeric = parameters.get(field)
        if (
            isinstance(numeric, bool)
            or not isinstance(numeric, (int, float))
            or not Decimal(str(numeric)).is_finite()
            or Decimal(str(numeric)) <= 0
        ):
            raise PairedProtocolV3Error("deployment parameter closure changed")
    thresholds = deployment["formal190_thresholds"]
    if (
        _fixed_point(
            thresholds.get("minimum_formal190_composite_margin"),
            "core composite margin",
        ) != development.get("minimum_composite_margin_fixed_point")
        or _fixed_point(
            thresholds.get("maximum_formal190_pair_uncertainty"),
            "core pair uncertainty",
        ) != development.get("maximum_pair_uncertainty_fixed_point")
    ):
        raise PairedProtocolV3Error("formal190 threshold closure changed")
    public_key = _decode_hex(
        authority.get("issuer_public_key_hex"), 32, "issuer Ed25519 public key"
    )
    if hashlib.sha256(public_key).hexdigest() != authority["issuer_public_key_sha256"]:
        raise PairedProtocolV3Error("protocol core issuer key fingerprint changed")
    return logical


def _require_crypto() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except (ImportError, ModuleNotFoundError) as error:
        raise PairedProtocolV3Error(
            "cryptography Ed25519 is required; SHA fallback is forbidden"
        ) from error
    return Ed25519PublicKey


def _decode_hex(value: Any, length: int, role: str) -> bytes:
    if not isinstance(value, str) or len(value) != length * 2:
        raise PairedProtocolV3Error(f"{role} must be {length} bytes in lowercase hex")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise PairedProtocolV3Error(f"{role} is invalid hex") from error
    if decoded.hex() != value:
        raise PairedProtocolV3Error(f"{role} must use canonical lowercase hex")
    return decoded


def decision_signing_bytes(statement: Mapping[str, Any]) -> bytes:
    return SIGNATURE_CONTEXT + canonical_bytes(statement)


def expected_decision_statement(
    core: Mapping[str, Any], *, core_file_sha256: str, decision_nonce_hex: str
) -> dict[str, Any]:
    validate_core(core)
    if not is_sha(core_file_sha256):
        raise PairedProtocolV3Error("protocol core file SHA is invalid")
    _decode_hex(decision_nonce_hex, 32, "decision nonce")
    policy = core["authority_policy"]
    evaluation = core["evaluation400"]
    deployment = core["deployment"]
    inventory = core["execution_inventory"]
    return {
        "protocol_core_file_sha256": core_file_sha256,
        "protocol_core_sha256": core["protocol_core_sha256"],
        "issuer_key_id": policy["issuer_key_id"],
        "issuer_public_key_sha256": policy["issuer_public_key_sha256"],
        "issuer_identity_sha256": policy["issuer_identity_sha256"],
        "trusted_issuer_attestation_sha256": policy[
            "trusted_issuer_attestation_sha256"
        ],
        "executor_identity_sha256": policy["executor_identity_sha256"],
        "result_evaluator_identity_sha256": policy[
            "result_evaluator_identity_sha256"
        ],
        "execution_inventory_file_sha256": inventory["attestation"][
            "file_sha256"
        ],
        "execution_inventory_sha256": inventory["attestation"]["logical_sha256"],
        "execution_stack_binding_sha256": inventory["stack_binding_sha256"],
        "executor_implementation_file_sha256": inventory[
            "executor_implementation_file_sha256"
        ],
        "result_evaluator_implementation_file_sha256": inventory[
            "result_evaluator_implementation_file_sha256"
        ],
        "decision_nonce_hex": decision_nonce_hex,
        "authorization_sequence": 1,
        "pair_identity_set_sha256": evaluation["pair_identity_set_sha256"],
        "deployment_binding_sha256": deployment["deployment_binding_sha256"],
        "policy_runtime_action_binding_sha256": deployment[
            "policy_runtime_action_binding_sha256"
        ],
        "authorized_pair_count": 400,
        "evaluation400_is_only_lane": True,
        "additional_reserve400_authorized": False,
        "postfreeze_changes_authorized": False,
        "outcomes_or_trajectories_read_before_decision": False,
        "external_executor_only": True,
        "execution_authorized": True,
    }


def verify_decision(
    value: Mapping[str, Any], *, core: Mapping[str, Any], core_file_sha256: str
) -> str:
    _require_crypto()
    logical = verify_hash_signature(value, "decision_sha256", "Ed25519 decision")
    if set(value) != {
        "format", "status", "signature_algorithm", "statement",
        "decision_signature_ed25519_hex", "decision_sha256",
    } or value.get("format") != DECISION_FORMAT or value.get("status") != DECISION_STATUS \
       or value.get("signature_algorithm") != "Ed25519":
        raise PairedProtocolV3Error("Ed25519 decision envelope changed")
    statement = value.get("statement")
    if not isinstance(statement, Mapping):
        raise PairedProtocolV3Error("Ed25519 decision statement missing")
    nonce = statement.get("decision_nonce_hex")
    if dict(statement) != expected_decision_statement(
        core, core_file_sha256=core_file_sha256, decision_nonce_hex=str(nonce)
    ):
        raise PairedProtocolV3Error("Ed25519 decision does not bind exact core")
    signature = _decode_hex(
        value.get("decision_signature_ed25519_hex"), 64, "Ed25519 signature"
    )
    public_bytes = _decode_hex(
        core["authority_policy"]["issuer_public_key_hex"], 32,
        "issuer Ed25519 public key",
    )
    try:
        public_key = _require_crypto().from_public_bytes(public_bytes)
        public_key.verify(signature, decision_signing_bytes(statement))
    except Exception as error:
        raise PairedProtocolV3Error("Ed25519 decision signature verification failed") from error
    return logical


def freeze_bundle(
    *, core_path: Path, core_file_sha256: str,
    decision_path: Path, decision_file_sha256: str,
) -> dict[str, Any]:
    core_bound, core, _ = read_json(core_path, core_file_sha256, "protocol core v3")
    core_logical = validate_core(core)
    post_binding = core["post_collection_v3"]
    lineage = core["r7h_target_adapter_lineage"]
    evaluation = core["evaluation400"]
    policy = core["authority_policy"]
    inventory = core["execution_inventory"]
    try:
        rebuilt = freeze_core(
            post_plan_path=Path(post_binding["static_plan"]["path"]),
            post_plan_file_sha256=post_binding["static_plan"]["file_sha256"],
            post_terminal_path=Path(post_binding["terminal"]["path"]),
            post_terminal_file_sha256=post_binding["terminal"]["file_sha256"],
            post_handoff_path=Path(post_binding["handoff"]["path"]),
            post_handoff_file_sha256=post_binding["handoff"]["file_sha256"],
            member_receipts=[
                (Path(row["receipt"]["path"]), row["receipt"]["file_sha256"])
                for row in lineage["members"]
            ],
            identity_bridge_path=Path(evaluation["identity_bridge"]["path"]),
            identity_bridge_file_sha256=evaluation["identity_bridge"]["file_sha256"],
            execution_inventory_attestation_path=Path(
                inventory["attestation"]["path"]
            ),
            execution_inventory_attestation_file_sha256=inventory[
                "attestation"
            ]["file_sha256"],
            execution_inventory_attestation_sha256=inventory["attestation"][
                "logical_sha256"
            ],
            expected_post_launcher_sha256=post_binding["reviewed_launcher_sha256"],
            runtime_execution_authority_path=Path(
                core["deployment"]["runtime_execution_authority"]["path"]
            ),
            runtime_execution_authority_file_sha256=core["deployment"][
                "runtime_execution_authority"
            ]["file_sha256"],
            selector_implementation_path=Path(
                core["deployment"]["selector_authority"]["implementation"][
                    "path"
                ]
            ),
            selector_implementation_file_sha256=core["deployment"][
                "selector_authority"
            ]["implementation"]["file_sha256"],
        )
    except (KeyError, TypeError) as error:
        raise PairedProtocolV3Error("protocol core reconstruction inputs changed") from error
    if rebuilt != core:
        raise PairedProtocolV3Error("protocol core differs from dependency reconstruction")
    decision_bound, decision, _ = read_json(
        decision_path, decision_file_sha256, "Ed25519 execution decision"
    )
    decision_logical = verify_decision(
        decision, core=core, core_file_sha256=core_file_sha256
    )
    base = {
        "format": BUNDLE_FORMAT, "status": BUNDLE_STATUS,
        "protocol_core": {"path": str(core_bound), "file_sha256": core_file_sha256, "logical_sha256": core_logical},
        "ed25519_decision": {"path": str(decision_bound), "file_sha256": decision_file_sha256, "logical_sha256": decision_logical},
        "issuer_key_id": core["authority_policy"]["issuer_key_id"],
        "issuer_public_key_sha256": core["authority_policy"]["issuer_public_key_sha256"],
        "trusted_issuer_attestation_sha256": core["authority_policy"][
            "trusted_issuer_attestation_sha256"
        ],
        "executor_identity_sha256": core["authority_policy"]["executor_identity_sha256"],
        "result_evaluator_identity_sha256": core["authority_policy"][
            "result_evaluator_identity_sha256"
        ],
        "execution_inventory": dict(core["execution_inventory"]),
        "pair_identity_set_sha256": core["evaluation400"]["pair_identity_set_sha256"],
        "deployment_binding_sha256": core["deployment"]["deployment_binding_sha256"],
        "authorized_pair_count": 400, "additional_reserve400_count": 0,
        "external_executor_only": True, "protocol_freezer_may_execute": False,
        "execution_authorized": True,
        "capability_receipt": {
            "hdf5_files_opened": 0, "trajectory_files_opened": 0,
            "label_or_outcome_files_opened": 0, "checkpoint_deserialization_calls": 0,
            "policy_or_simulator_calls": 0, "pair_conditions_executed": 0,
        },
    }
    return {**base, "bundle_sha256": canonical_sha256(base)}


def validate_bundle(value: Mapping[str, Any]) -> str:
    logical = verify_hash_signature(value, "bundle_sha256", "execution bundle v3")
    capability = value.get("capability_receipt")
    if (
        set(value) != {
            "format", "status", "protocol_core", "ed25519_decision",
            "issuer_key_id", "issuer_public_key_sha256",
            "trusted_issuer_attestation_sha256", "executor_identity_sha256",
            "result_evaluator_identity_sha256", "execution_inventory",
            "pair_identity_set_sha256", "deployment_binding_sha256",
            "authorized_pair_count", "additional_reserve400_count",
            "external_executor_only", "protocol_freezer_may_execute",
            "execution_authorized", "capability_receipt", "bundle_sha256",
        }
        or value.get("format") != BUNDLE_FORMAT or value.get("status") != BUNDLE_STATUS
        or not isinstance(value.get("protocol_core"), Mapping)
        or set(value["protocol_core"]) != RECORD_FIELDS
        or not isinstance(value.get("ed25519_decision"), Mapping)
        or set(value["ed25519_decision"]) != RECORD_FIELDS
        or not isinstance(value.get("execution_inventory"), Mapping)
        or value["execution_inventory"].get("real_execution_components_complete") is not True
        or any(not is_sha(value.get(field)) for field in (
            "issuer_public_key_sha256", "trusted_issuer_attestation_sha256",
            "executor_identity_sha256", "result_evaluator_identity_sha256",
            "pair_identity_set_sha256", "deployment_binding_sha256",
        ))
        or type(value.get("authorized_pair_count")) is not int
        or value["authorized_pair_count"] != 400
        or type(value.get("additional_reserve400_count")) is not int
        or value["additional_reserve400_count"] != 0
        or value.get("external_executor_only") is not True
        or value.get("protocol_freezer_may_execute") is not False
        or value.get("execution_authorized") is not True
        or not isinstance(capability, Mapping)
        or set(capability) != {
            "hdf5_files_opened", "trajectory_files_opened",
            "label_or_outcome_files_opened", "checkpoint_deserialization_calls",
            "policy_or_simulator_calls", "pair_conditions_executed",
        }
        or any(type(capability.get(field)) is not int or capability[field] != 0
               for field in capability)
    ):
        raise PairedProtocolV3Error("execution bundle v3 boundary changed")
    return logical


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if _sensitive(PurePath(output)) or output.suffix.casefold() != ".json":
        raise PairedProtocolV3Error("output path is forbidden")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent = output.parent.resolve(strict=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    core = sub.add_parser("freeze-core")
    for name in ("post-plan", "post-terminal", "post-handoff", "identity-bridge"):
        core.add_argument(f"--{name}", type=Path, required=True)
        core.add_argument(f"--{name}-file-sha256", required=True)
    core.add_argument("--member-receipt", nargs=2, action="append", required=True)
    core.add_argument("--execution-inventory-attestation", type=Path, required=True)
    core.add_argument("--execution-inventory-attestation-file-sha256", required=True)
    core.add_argument("--execution-inventory-attestation-sha256", required=True)
    core.add_argument("--expected-post-launcher-sha256", required=True)
    core.add_argument("--runtime-execution-authority", type=Path, required=True)
    core.add_argument("--runtime-execution-authority-file-sha256", required=True)
    core.add_argument("--selector-implementation", type=Path, required=True)
    core.add_argument("--selector-implementation-file-sha256", required=True)
    core.add_argument("--output", type=Path, required=True)
    bundle = sub.add_parser("freeze-bundle")
    bundle.add_argument("--core", type=Path, required=True)
    bundle.add_argument("--core-file-sha256", required=True)
    bundle.add_argument("--decision", type=Path, required=True)
    bundle.add_argument("--decision-file-sha256", required=True)
    bundle.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "freeze-core":
        value = freeze_core(
            post_plan_path=args.post_plan,
            post_plan_file_sha256=args.post_plan_file_sha256,
            post_terminal_path=args.post_terminal,
            post_terminal_file_sha256=args.post_terminal_file_sha256,
            post_handoff_path=args.post_handoff,
            post_handoff_file_sha256=args.post_handoff_file_sha256,
            member_receipts=[(Path(path), sha) for path, sha in args.member_receipt],
            identity_bridge_path=args.identity_bridge,
            identity_bridge_file_sha256=args.identity_bridge_file_sha256,
            execution_inventory_attestation_path=args.execution_inventory_attestation,
            execution_inventory_attestation_file_sha256=(
                args.execution_inventory_attestation_file_sha256
            ),
            execution_inventory_attestation_sha256=(
                args.execution_inventory_attestation_sha256
            ),
            expected_post_launcher_sha256=args.expected_post_launcher_sha256,
            runtime_execution_authority_path=args.runtime_execution_authority,
            runtime_execution_authority_file_sha256=(
                args.runtime_execution_authority_file_sha256
            ),
            selector_implementation_path=args.selector_implementation,
            selector_implementation_file_sha256=(
                args.selector_implementation_file_sha256
            ),
        )
        validate_core(value)
    else:
        value = freeze_bundle(
            core_path=args.core, core_file_sha256=args.core_file_sha256,
            decision_path=args.decision,
            decision_file_sha256=args.decision_file_sha256,
        )
        validate_bundle(value)
    write_json_new(args.output, value)


if __name__ == "__main__":
    main()
