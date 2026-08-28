#!/usr/bin/env python3
"""Freeze paired-v3 execution keys and a production execution inventory.

This tool is pre-outcome and metadata-only.  It never opens HDF, trajectory,
label, outcome, checkpoint, policy output, or simulator output files.  It does
not authorize execution and does not change paired-v3's production allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence


CORE_FORMAT = "etsf_smolvla_piper_paired_success_protocol_core_v3"
INVENTORY_FORMAT = "etsf_smolvla_piper_paired_execution_inventory_attestation_v3"
INVENTORY_STATUS = "externally_reviewed_complete_immutable_execution_stack"
ISSUER_FORMAT = "etsf_smolvla_piper_trusted_issuer_allowlist_v3"
ISSUER_STATUS = "externally_reviewed_active_ed25519_issuer"
KEY_MANIFEST_FORMAT = "etsf_smolvla_piper_paired_execution_key_manifest_v3"
KEY_MANIFEST_STATUS = "generated_keys_frozen_execution_not_authorized"
PAIR_COUNT = 400
SHA_CHARS = frozenset("0123456789abcdef")
HDF_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5"})
FORBIDDEN_COMPONENTS = frozenset(
    {"fresh", "confirmation", "test", "trajectory", "label", "labels", "outcome", "outcomes"}
)


class ExecutionInventoryPreregistrationError(RuntimeError):
    """An immutable key, component, type, or namespace gate failed closed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _reject_constant(token: str) -> None:
    raise ExecutionInventoryPreregistrationError(
        f"non-finite JSON number is forbidden: {token}"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionInventoryPreregistrationError(
                f"duplicate JSON key is forbidden: {key}"
            )
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionInventoryPreregistrationError(
            f"{role} is invalid strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise ExecutionInventoryPreregistrationError(f"{role} must contain an object")
    return value


def _forbidden(path: PurePath) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if lowered in FORBIDDEN_COMPONENTS:
            return True
        if lowered.startswith(
            (
                "fresh_", "fresh-", "confirmation_", "confirmation-",
                "test_", "test-", "trajectory_", "trajectory-",
                "label_", "label-", "outcome_", "outcome-",
            )
        ):
            return True
    return False


def _canonical_path(path: Path, role: str) -> Path:
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if str(path) != str(lexical):
        raise ExecutionInventoryPreregistrationError(
            f"{role} path must be canonical absolute"
        )
    if _forbidden(PurePath(lexical)) or lexical.suffix.casefold() in HDF_SUFFIXES:
        raise ExecutionInventoryPreregistrationError(f"{role} path is forbidden")
    return lexical


def _open_directory(path: Path, role: str) -> int:
    lexical = _canonical_path(path, role)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(lexical.anchor, flags)
    try:
        for component in lexical.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def secure_read(
    path: Path,
    expected_file_sha256: str,
    role: str,
    *,
    suffix: str | None = None,
    strict_json: bool = False,
) -> tuple[Path, bytes, str]:
    lexical = _canonical_path(path, role)
    if not is_sha(expected_file_sha256):
        raise ExecutionInventoryPreregistrationError(f"{role} expected SHA is invalid")
    if suffix is not None and lexical.suffix.casefold() != suffix:
        raise ExecutionInventoryPreregistrationError(f"{role} suffix changed")
    directory_fd = _open_directory(lexical.parent, f"{role} parent")
    file_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        file_fd = os.open(lexical.name, flags, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ExecutionInventoryPreregistrationError(f"{role} must be a regular file")
        if before.st_mode & 0o222:
            raise ExecutionInventoryPreregistrationError(f"{role} must be frozen read-only")
        chunks: list[bytes] = []
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(file_fd)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(payload) != before.st_size
        ):
            raise ExecutionInventoryPreregistrationError(f"{role} changed while read")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_file_sha256:
            raise ExecutionInventoryPreregistrationError(f"{role} file SHA mismatch")
        if strict_json:
            strict_json_bytes(payload, role)
        return lexical, payload, digest
    except OSError as error:
        raise ExecutionInventoryPreregistrationError(
            f"{role} cannot be opened without following symlinks"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes, mode: int, role: str) -> Path:
    lexical = _canonical_path(path, role)
    parent_fd = _open_directory(lexical.parent, f"{role} parent")
    descriptor: int | None = None
    try:
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(lexical.name, flags, mode, dir_fd=parent_fd)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.fsync(parent_fd)
        return lexical
    except OSError as error:
        raise ExecutionInventoryPreregistrationError(
            f"{role} must be create-once in a symlink-free parent"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def write_json_new(path: Path, value: Mapping[str, Any], role: str) -> Path:
    payload = json_output_bytes(value)
    return _write_new(path, payload, 0o444, role)


def json_output_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"


def _require_crypto() -> tuple[Any, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise ExecutionInventoryPreregistrationError(
            "cryptography Ed25519 is required"
        ) from error
    return (Ed25519PrivateKey, (Ed25519PublicKey, serialization))


def _public_key(
    path: Path, expected_file_sha256: str, role: str
) -> tuple[dict[str, str], bytes]:
    _private, public_support = _require_crypto()
    public_type, _serialization = public_support
    bound, payload, digest = secure_read(path, expected_file_sha256, role)
    if len(payload) != 32:
        raise ExecutionInventoryPreregistrationError(
            f"{role} must be an exact 32-byte raw Ed25519 public key"
        )
    try:
        public_type.from_public_bytes(payload)
    except ValueError as error:
        raise ExecutionInventoryPreregistrationError(f"{role} is invalid") from error
    identity = hashlib.sha256(payload).hexdigest()
    return {
        "path": str(bound),
        "file_sha256": digest,
        "public_key_hex": payload.hex(),
        "identity_sha256": identity,
    }, payload


def _component(
    path: Path, expected_file_sha256: str, role: str, suffix: str
) -> dict[str, str]:
    bound, _payload, digest = secure_read(
        path,
        expected_file_sha256,
        role,
        suffix=suffix,
        strict_json=suffix == ".json",
    )
    return {"path": str(bound), "file_sha256": digest}


def validate_issuer_attestation(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    logical = unsigned.pop("attestation_sha256", None)
    if (
        set(value)
        != {
            "format", "status", "protocol_format", "issuer_key_id",
            "issuer_public_key_hex", "issuer_public_key_sha256",
            "issuer_identity_sha256", "allowlist_entry_active",
            "authorization_sequence", "attestation_sha256",
        }
        or value.get("format") != ISSUER_FORMAT
        or value.get("status") != ISSUER_STATUS
        or value.get("protocol_format") != CORE_FORMAT
        or not isinstance(value.get("issuer_key_id"), str)
        or not value["issuer_key_id"]
        or not is_sha(value.get("issuer_public_key_sha256"))
        or value.get("issuer_identity_sha256") != value.get("issuer_public_key_sha256")
        or value.get("allowlist_entry_active") is not True
        or type(value.get("authorization_sequence")) is not int
        or value["authorization_sequence"] != 1
        or logical != canonical_sha256(unsigned)
    ):
        raise ExecutionInventoryPreregistrationError(
            "trusted issuer attestation contract changed"
        )
    public_hex = value.get("issuer_public_key_hex")
    if not isinstance(public_hex, str) or len(public_hex) != 64:
        raise ExecutionInventoryPreregistrationError("issuer public key hex changed")
    try:
        public_bytes = bytes.fromhex(public_hex)
    except ValueError as error:
        raise ExecutionInventoryPreregistrationError("issuer public key hex is invalid") from error
    if public_bytes.hex() != public_hex or hashlib.sha256(public_bytes).hexdigest() != value[
        "issuer_public_key_sha256"
    ]:
        raise ExecutionInventoryPreregistrationError("issuer fingerprint changed")
    return str(logical)


def validate_inventory(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    logical = unsigned.pop("attestation_sha256", None)
    lane = value.get("execution_lane")
    executor = value.get("executor")
    evaluator = value.get("result_evaluator")
    stack = value.get("execution_stack")
    if (
        set(value)
        != {
            "format", "status", "protocol_format", "execution_lane",
            "trusted_issuer_attestation", "executor", "result_evaluator",
            "execution_stack", "component_inventory_complete",
            "real_executor_present", "real_result_evaluator_present",
            "outcome_or_trajectory_files_opened_during_attestation",
            "attestation_sha256",
        }
        or value.get("format") != INVENTORY_FORMAT
        or value.get("status") != INVENTORY_STATUS
        or value.get("protocol_format") != CORE_FORMAT
        or not isinstance(lane, Mapping)
        or dict(lane) != {
            "pair_count": PAIR_COUNT,
            "only_evaluation400_lane": True,
            "additional_reserve400_count": 0,
        }
        or not isinstance(executor, Mapping)
        or set(executor) != {"identity_sha256", "implementation"}
        or not is_sha(executor.get("identity_sha256"))
        or not isinstance(evaluator, Mapping)
        or set(evaluator) != {"identity_sha256", "implementation"}
        or not is_sha(evaluator.get("identity_sha256"))
        or not isinstance(stack, Mapping)
        or set(stack)
        != {
            "simulator_implementation", "runtime_contract",
            "collector_implementation", "container_inventory",
        }
        or value.get("component_inventory_complete") is not True
        or value.get("real_executor_present") is not True
        or value.get("real_result_evaluator_present") is not True
        or type(value.get("outcome_or_trajectory_files_opened_during_attestation"))
        is not int
        or value["outcome_or_trajectory_files_opened_during_attestation"] != 0
        or logical != canonical_sha256(unsigned)
    ):
        raise ExecutionInventoryPreregistrationError("execution inventory contract changed")
    descriptor_values = [
        executor.get("implementation"), evaluator.get("implementation"), *stack.values()
    ]
    if any(
        not isinstance(row, Mapping)
        or set(row) != {"path", "file_sha256"}
        or not isinstance(row.get("path"), str)
        or not is_sha(row.get("file_sha256"))
        for row in descriptor_values
    ):
        raise ExecutionInventoryPreregistrationError("component descriptor changed")
    issuer = value.get("trusted_issuer_attestation")
    if (
        not isinstance(issuer, Mapping)
        or set(issuer) != {"path", "file_sha256", "logical_sha256"}
        or not isinstance(issuer.get("path"), str)
        or not is_sha(issuer.get("file_sha256"))
        or not is_sha(issuer.get("logical_sha256"))
    ):
        raise ExecutionInventoryPreregistrationError("issuer descriptor changed")
    return str(logical)


def generate_keys(output_directory: Path) -> dict[str, Any]:
    private_type, public_support = _require_crypto()
    _public_type, serialization = public_support
    output = _canonical_path(output_directory, "key output directory")
    parent_fd = _open_directory(output.parent, "key output parent")
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as error:
        raise ExecutionInventoryPreregistrationError(
            "key output directory must be create-once"
        ) from error
    finally:
        os.close(parent_fd)
    keys: dict[str, dict[str, str]] = {}
    for role in ("issuer", "executor", "result_signer"):
        private_key = private_type.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_path = _write_new(
            output / f"{role}_ed25519_private.raw",
            private_bytes,
            0o400,
            f"{role} private key",
        )
        public_path = _write_new(
            output / f"{role}_ed25519_public.raw",
            public_bytes,
            0o444,
            f"{role} public key",
        )
        keys[role] = {
            "role": role,
            "public_key_path": str(public_path),
            "public_key_file_sha256": hashlib.sha256(public_bytes).hexdigest(),
            "public_key_hex": public_bytes.hex(),
            "identity_sha256": hashlib.sha256(public_bytes).hexdigest(),
        }
    if len({row["identity_sha256"] for row in keys.values()}) != 3:
        raise ExecutionInventoryPreregistrationError("generated public keys are not distinct")
    base: dict[str, Any] = {
        "format": KEY_MANIFEST_FORMAT,
        "status": KEY_MANIFEST_STATUS,
        "protocol_format": CORE_FORMAT,
        "keys": keys,
        "private_key_count": 3,
        "public_key_count": 3,
        "execution_authorized": False,
        "outcome_hdf_trajectory_or_label_files_opened": 0,
    }
    manifest = {**base, "manifest_sha256": canonical_sha256(base)}
    manifest_path = write_json_new(
        output / "public_key_manifest.json", manifest, "public key manifest"
    )
    manifest_file_sha = hashlib.sha256(json_output_bytes(manifest)).hexdigest()
    secure_read(
        manifest_path,
        manifest_file_sha,
        "published public key manifest",
        suffix=".json",
        strict_json=True,
    )
    output.chmod(0o500)
    return {
        "output_directory": str(output),
        "public_manifest": {
            "path": str(manifest_path),
            "file_sha256": manifest_file_sha,
            "logical_sha256": manifest["manifest_sha256"],
        },
        "execution_authorized": False,
    }


def preregister(
    *,
    issuer_key_id: str,
    issuer_public_key: Path,
    issuer_public_key_file_sha256: str,
    executor_public_key: Path,
    executor_public_key_file_sha256: str,
    result_signer_public_key: Path,
    result_signer_public_key_file_sha256: str,
    executor_implementation: Path,
    executor_implementation_file_sha256: str,
    result_evaluator_implementation: Path,
    result_evaluator_implementation_file_sha256: str,
    simulator_implementation: Path,
    simulator_implementation_file_sha256: str,
    collector_implementation: Path,
    collector_implementation_file_sha256: str,
    runtime_contract: Path,
    runtime_contract_file_sha256: str,
    container_inventory: Path,
    container_inventory_file_sha256: str,
    condition_runner_binding: str,
    issuer_attestation_output: Path,
    inventory_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(issuer_key_id, str) or not issuer_key_id.strip():
        raise ExecutionInventoryPreregistrationError("issuer key ID is required")
    if condition_runner_binding not in {"shared", "distinct"}:
        raise ExecutionInventoryPreregistrationError(
            "condition runner binding must be shared or distinct"
        )
    issuer_key, issuer_bytes = _public_key(
        issuer_public_key, issuer_public_key_file_sha256, "issuer public key"
    )
    executor_key, _executor_bytes = _public_key(
        executor_public_key, executor_public_key_file_sha256, "executor public key"
    )
    result_key, _result_bytes = _public_key(
        result_signer_public_key,
        result_signer_public_key_file_sha256,
        "result signer public key",
    )
    if len(
        {
            issuer_key["identity_sha256"],
            executor_key["identity_sha256"],
            result_key["identity_sha256"],
        }
    ) != 3:
        raise ExecutionInventoryPreregistrationError("issuer/executor/result keys must differ")

    components = {
        "executor": _component(
            executor_implementation,
            executor_implementation_file_sha256,
            "executor implementation",
            ".py",
        ),
        "result_evaluator": _component(
            result_evaluator_implementation,
            result_evaluator_implementation_file_sha256,
            "result evaluator implementation",
            ".py",
        ),
        "simulator": _component(
            simulator_implementation,
            simulator_implementation_file_sha256,
            "simulator condition runner",
            ".py",
        ),
        "collector": _component(
            collector_implementation,
            collector_implementation_file_sha256,
            "collector condition runner",
            ".py",
        ),
        "runtime": _component(
            runtime_contract,
            runtime_contract_file_sha256,
            "runtime contract",
            ".json",
        ),
        "container": _component(
            container_inventory,
            container_inventory_file_sha256,
            "container inventory",
            ".json",
        ),
    }
    runners_same = components["simulator"] == components["collector"]
    if (condition_runner_binding == "shared") is not runners_same:
        raise ExecutionInventoryPreregistrationError(
            "condition runner shared/distinct declaration does not match path and SHA"
        )
    issuer_base: dict[str, Any] = {
        "format": ISSUER_FORMAT,
        "status": ISSUER_STATUS,
        "protocol_format": CORE_FORMAT,
        "issuer_key_id": issuer_key_id.strip(),
        "issuer_public_key_hex": issuer_bytes.hex(),
        "issuer_public_key_sha256": issuer_key["identity_sha256"],
        "issuer_identity_sha256": issuer_key["identity_sha256"],
        "allowlist_entry_active": True,
        "authorization_sequence": 1,
    }
    issuer_attestation = {
        **issuer_base, "attestation_sha256": canonical_sha256(issuer_base)
    }
    validate_issuer_attestation(issuer_attestation)
    issuer_output = _canonical_path(
        issuer_attestation_output, "issuer attestation output"
    )
    inventory_path = _canonical_path(inventory_output, "execution inventory output")
    if issuer_output == inventory_path:
        raise ExecutionInventoryPreregistrationError("output files must be distinct")
    for path in (issuer_output, inventory_path):
        if path.exists() or path.is_symlink():
            raise ExecutionInventoryPreregistrationError("outputs are create-once")
    issuer_path = write_json_new(
        issuer_output, issuer_attestation, "trusted issuer attestation"
    )
    issuer_file_sha = hashlib.sha256(json_output_bytes(issuer_attestation)).hexdigest()
    secure_read(
        issuer_path,
        issuer_file_sha,
        "published trusted issuer attestation",
        suffix=".json",
        strict_json=True,
    )
    issuer_record = {
        "path": str(issuer_path),
        "file_sha256": issuer_file_sha,
        "logical_sha256": issuer_attestation["attestation_sha256"],
    }
    inventory_base: dict[str, Any] = {
        "format": INVENTORY_FORMAT,
        "status": INVENTORY_STATUS,
        "protocol_format": CORE_FORMAT,
        "execution_lane": {
            "pair_count": PAIR_COUNT,
            "only_evaluation400_lane": True,
            "additional_reserve400_count": 0,
        },
        "trusted_issuer_attestation": issuer_record,
        "executor": {
            "identity_sha256": executor_key["identity_sha256"],
            "implementation": components["executor"],
        },
        "result_evaluator": {
            "identity_sha256": result_key["identity_sha256"],
            "implementation": components["result_evaluator"],
        },
        "execution_stack": {
            "simulator_implementation": components["simulator"],
            "runtime_contract": components["runtime"],
            "collector_implementation": components["collector"],
            "container_inventory": components["container"],
        },
        "component_inventory_complete": True,
        "real_executor_present": True,
        "real_result_evaluator_present": True,
        "outcome_or_trajectory_files_opened_during_attestation": 0,
    }
    inventory = {
        **inventory_base, "attestation_sha256": canonical_sha256(inventory_base)
    }
    validate_inventory(inventory)
    write_json_new(inventory_path, inventory, "execution inventory attestation")
    inventory_file_sha = hashlib.sha256(json_output_bytes(inventory)).hexdigest()
    secure_read(
        inventory_path,
        inventory_file_sha,
        "published execution inventory attestation",
        suffix=".json",
        strict_json=True,
    )
    return issuer_attestation, inventory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keys = commands.add_parser("generate-keys")
    keys.add_argument("--output-directory", type=Path, required=True)
    freeze = commands.add_parser("preregister")
    freeze.add_argument("--issuer-key-id", required=True)
    for role in ("issuer", "executor", "result-signer"):
        freeze.add_argument(f"--{role}-public-key", type=Path, required=True)
        freeze.add_argument(f"--{role}-public-key-file-sha256", required=True)
    for role in (
        "executor", "result-evaluator", "simulator", "collector",
    ):
        freeze.add_argument(f"--{role}-implementation", type=Path, required=True)
        freeze.add_argument(f"--{role}-implementation-file-sha256", required=True)
    for role in ("runtime-contract", "container-inventory"):
        freeze.add_argument(f"--{role}", type=Path, required=True)
        freeze.add_argument(f"--{role}-file-sha256", required=True)
    freeze.add_argument(
        "--condition-runner-binding", choices=("shared", "distinct"), required=True
    )
    freeze.add_argument("--issuer-attestation-output", type=Path, required=True)
    freeze.add_argument("--inventory-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "generate-keys":
        result = generate_keys(args.output_directory)
    else:
        issuer, inventory = preregister(
            issuer_key_id=args.issuer_key_id,
            issuer_public_key=args.issuer_public_key,
            issuer_public_key_file_sha256=args.issuer_public_key_file_sha256,
            executor_public_key=args.executor_public_key,
            executor_public_key_file_sha256=args.executor_public_key_file_sha256,
            result_signer_public_key=args.result_signer_public_key,
            result_signer_public_key_file_sha256=args.result_signer_public_key_file_sha256,
            executor_implementation=args.executor_implementation,
            executor_implementation_file_sha256=args.executor_implementation_file_sha256,
            result_evaluator_implementation=args.result_evaluator_implementation,
            result_evaluator_implementation_file_sha256=(
                args.result_evaluator_implementation_file_sha256
            ),
            simulator_implementation=args.simulator_implementation,
            simulator_implementation_file_sha256=args.simulator_implementation_file_sha256,
            collector_implementation=args.collector_implementation,
            collector_implementation_file_sha256=args.collector_implementation_file_sha256,
            runtime_contract=args.runtime_contract,
            runtime_contract_file_sha256=args.runtime_contract_file_sha256,
            container_inventory=args.container_inventory,
            container_inventory_file_sha256=args.container_inventory_file_sha256,
            condition_runner_binding=args.condition_runner_binding,
            issuer_attestation_output=args.issuer_attestation_output,
            inventory_output=args.inventory_output,
        )
        result = {
            "issuer_attestation_sha256": issuer["attestation_sha256"],
            "execution_inventory_sha256": inventory["attestation_sha256"],
            "execution_authorized": False,
            "paired_v3_allowlist_modified": False,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
