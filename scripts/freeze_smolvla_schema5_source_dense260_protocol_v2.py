#!/usr/bin/env python3
"""Offline trust and partition protocol for Source dense260 v2.

This program has no simulator, policy, HDF, trajectory, label, training, or
collection capability.  It can:

* freeze a create-once public issuer/role/source-lineage registry;
* prepare an aggregate-only, unsigned disjointness payload from an owner-only
  identity view (signing is deliberately external);
* verify an Ed25519 OpenSSH SSHSIG detached signature on the reset manifest and
  on all six aggregate attestations; and
* freeze the 100/80/80 dense260 partition and its 8/5/200 collection contract.

The registry is a bootstrap trust root.  Every consumer must supply its
externally reviewed file SHA256 and logical registry SHA256.  A canonical SHA
is content addressing, not an issuer signature.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_NAMESPACE = "schema5_aloha_source_dense260_20260829_v2"
REGISTRY_FORMAT = "etsf_source_dense260_role_issuer_registry_v2"
REGISTRY_SPEC_FORMAT = "etsf_source_dense260_role_issuer_registry_spec_v2"
REGISTRY_STATUS = "frozen_public_trust_root_collection_not_authorized"
CANDIDATE_FORMAT = "etsf_smolvla_schema5_source_dense260_reset_manifest_v2"
CANDIDATE_STATUS = "complete_400_reset_identity_only_no_policy_action_or_label"
PRIVATE_VIEW_FORMAT = "etsf_source_dense260_private_reference_identity_view_v2"
PRIVATE_VIEW_STATUS = "private_identity_only_attestation_material"
ATTESTATION_FORMAT = "etsf_source_dense260_identity_disjoint_attestation_v2"
ATTESTATION_STATUS = "verified_three_axis_disjoint_aggregate_only"
FREEZE_FORMAT = "etsf_smolvla_schema5_source_dense260_preregistration_v2"
FREEZE_STATUS = "frozen_signed_identity_partition_collection_not_authorized"
SIGNATURE_NAMESPACE = "etsf-source-dense260-v2"

TASK = "move_can_pot"
BODY = "aloha-agilex"
POLICY = "smolvla"
TARGET_ROLE = "source_dense260_reset_candidate_pool"
REFERENCE_ROLES = (
    "official150",
    "source63",
    "prior_development",
    "piper_development300",
    "formal_target_validation",
    "evaluation400",
)
AXES = ("requested_seed", "resolved_seed", "reset_identity")
CANDIDATE_START = 2_026_083_500
CANDIDATE_COUNT = 400
CANDIDATE_STEP = 1
SELECTED_GROUPS = 260
SPLIT_COUNTS = {"train": 100, "calibration": 80, "validation": 80}
EVENTS = ("e0", "e12", "e3", "e4", "eK")
MAX_SEED = 2**31 - 1
SHA_CHARS = frozenset("0123456789abcdef")
PRINCIPAL_PATTERN = re.compile(r"[A-Za-z0-9._@+:-]{1,128}\Z")
FORBIDDEN_DATA_SUFFIXES = frozenset({".h5", ".hdf", ".hdf5", ".npz", ".pt", ".pth"})

RESET_IDENTITY_CONTRACT = {
    "format": "etsf_cross_body_semantic_reset_identity_v2",
    "hash_algorithm": "sha256",
    "canonicalization": "json_sort_keys_compact_ascii_no_nan",
    "payload_fields": [
        "format",
        "task",
        "instruction_semantics_receipt_sha256",
        "initial_scene_state_sha256",
    ],
    "initial_scene_state_contract": (
        "etsf_move_can_pot_can_pose_pot_pose_float64_canonical_json_v1"
    ),
    "capture_boundary": "after_reset_before_policy_action_reward_event_or_label",
    "body_policy_joint_drive_requested_and_resolved_seed_excluded": True,
    "v1_identity_accepted": False,
}
RESET_IDENTITY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        RESET_IDENTITY_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()

ROLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "official150": {
        "role_namespace": "robotwin_move_can_pot_official150_v1",
        "logical_group_count": 150,
        "membership_semantics": "exact_official_success_seed_registry_membership",
        "sources": (
            (
                "official150_registry",
                "rlinf_robotwin_eval_seed_registry_unversioned",
                "move_can_pot/success_seeds/official150",
                150,
            ),
        ),
    },
    "source63": {
        "role_namespace": "smolvla_schema5_source63_20260828_v1",
        "logical_group_count": 63,
        "membership_semantics": "exact_source63_requested_resolved_reset_groups",
        "sources": (
            (
                "source63_schema5_manifest",
                "etsf_smolvla_schema5_collection_manifest_v5",
                "smolvla_schema5_source63_20260828_v1",
                63,
            ),
        ),
    },
    "prior_development": {
        "role_namespace": "robotwin_prior_development150_plus_v7d250_union_v2",
        "logical_group_count": 400,
        "membership_semantics": (
            "ordered_disjoint_union_development150_then_v7_development250"
        ),
        "sources": (
            (
                "development150_manifest",
                "etsf_robotwin_development_seed_manifest_v1",
                "explicit_development_expansion",
                150,
            ),
            (
                "v7_development250_manifest",
                "etsf_robotwin_v7_development_seed_manifest_v1",
                "explicit_v7_prospective_development",
                250,
            ),
        ),
    },
    "piper_development300": {
        "role_namespace": "schema6_piper_target_development300_20260828_v1",
        "logical_group_count": 300,
        "membership_semantics": "all_development300_identity_resolution_rows",
        "sources": (
            (
                "piper_development300_identity_receipt",
                (
                    "etsf_smolvla_piper_schema6_development300_"
                    "identity_resolution_receipt_v1"
                ),
                "schema6_piper_target_development300_20260828_v1",
                300,
            ),
        ),
    },
    "formal_target_validation": {
        "role_namespace": (
            "schema6_piper_target_development300_20260828_v1/"
            "formal_target_validation190"
        ),
        "logical_group_count": 190,
        "membership_semantics": (
            "exact_formal190_subset_of_piper_development300_no_outcome_access"
        ),
        "sources": (
            (
                "piper_development300_formal190_identity_receipt",
                (
                    "etsf_smolvla_piper_schema6_development300_"
                    "identity_resolution_receipt_v1"
                ),
                (
                    "schema6_piper_target_development300_20260828_v1/"
                    "formal_target_validation"
                ),
                190,
            ),
        ),
    },
    "evaluation400": {
        "role_namespace": "smolvla_piper_target_seed_manifest_v2/evaluation400",
        "logical_group_count": 400,
        "membership_semantics": (
            "exact_evaluation400_identity_rows_no_outcome_or_trajectory_access"
        ),
        "sources": (
            (
                "piper_target_manifest_v2_evaluation400",
                "etsf_smolvla_piper_target_seed_manifest_v2",
                "smolvla_piper_target_seed_manifest_v2/evaluation",
                400,
            ),
        ),
    },
}

REFERENCE_CAPABILITY = {
    "environment_reset_calls": 0,
    "environment_step_calls": 0,
    "policy_import_or_forward_calls": 0,
    "action_generation_or_execution_calls": 0,
    "trajectory_or_hdf_files_opened": 0,
    "reward_success_event_outcome_or_label_fields_read": 0,
    "identity_only_receipts_used": True,
}

CANDIDATE_CAPABILITY = {
    "candidate_rows": CANDIDATE_COUNT,
    "maximum_environment_reset_calls": CANDIDATE_COUNT,
    "one_reset_invocation_per_candidate": True,
    "implicit_seed_retry_allowed": False,
    "environment_step_calls": 0,
    "policy_import_or_forward_calls": 0,
    "action_generation_or_execution_calls": 0,
    "trajectory_or_hdf_files_opened": 0,
    "reward_success_event_outcome_or_label_fields_read": 0,
    "selection_or_early_stop_before_all_400_receipts": False,
}


class Dense260ProtocolV2Error(RuntimeError):
    """An offline trust, signature, identity, or partition invariant failed."""


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _strict_int(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _signed_content(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise Dense260ProtocolV2Error("logical SHA field already exists")
    result[field] = canonical_sha256(result)
    return result


def _verify_content_sha(
    value: Mapping[str, Any], field: str, role: str
) -> str:
    unsigned = dict(value)
    recorded = unsigned.pop(field, None)
    if not _is_sha(recorded) or recorded != canonical_sha256(unsigned):
        raise Dense260ProtocolV2Error(f"{role} logical SHA mismatch")
    return str(recorded)


def _strict_json(raw: bytes, role: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise Dense260ProtocolV2Error(f"{role} is not strict JSON") from None
    if not isinstance(value, dict):
        raise Dense260ProtocolV2Error(f"{role} must contain one object")
    return value


def _safe_regular(path: Path, role: str, suffixes: set[str]) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute() or raw.suffix.casefold() not in suffixes:
        raise Dense260ProtocolV2Error(f"{role} path or suffix is invalid")
    if raw.suffix.casefold() in FORBIDDEN_DATA_SUFFIXES:
        raise Dense260ProtocolV2Error(f"{role} data type is forbidden")
    current = Path(raw.anchor)
    for component in raw.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise Dense260ProtocolV2Error(f"{role} path contains a symlink")
    try:
        metadata = raw.stat()
    except OSError:
        raise Dense260ProtocolV2Error(f"{role} is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise Dense260ProtocolV2Error(f"{role} is not a regular file")
    return raw.absolute()


def _read_bytes(path: Path, role: str, suffixes: set[str]) -> tuple[bytes, str]:
    source = _safe_regular(path, role, suffixes)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError:
        raise Dense260ProtocolV2Error(f"{role} could not be opened safely") from None
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1 << 20))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if (
            remaining
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise Dense260ProtocolV2Error(f"{role} changed while being read")
        raw = b"".join(chunks)
        return raw, hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def _read_json(
    path: Path, role: str, *, canonical_file: bool
) -> tuple[dict[str, Any], bytes, str]:
    raw, digest = _read_bytes(path, role, {".json"})
    value = _strict_json(raw, role)
    if canonical_file and raw != canonical_bytes(value) + b"\n":
        raise Dense260ProtocolV2Error(f"{role} is not canonical JSON bytes")
    return value, raw, digest


def _read_private_json(
    path: Path, role: str
) -> tuple[dict[str, Any], bytes, str]:
    source = _safe_regular(path, role, {".json"})
    metadata = source.stat()
    if (
        metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
    ):
        raise Dense260ProtocolV2Error(
            f"{role} must be an owner-only 0400 regular file"
        )
    return _read_json(source, role, canonical_file=True)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.suffix.casefold() != ".json":
        raise Dense260ProtocolV2Error("output must be an absolute JSON path")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir() or path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = canonical_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Dense260ProtocolV2Error("output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _decode_ed25519_public_key(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or value.strip() != value:
        raise Dense260ProtocolV2Error("issuer public key is invalid")
    parts = value.split()
    if len(parts) != 2 or parts[0] != "ssh-ed25519":
        raise Dense260ProtocolV2Error("issuer key must be bare ssh-ed25519")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except (ValueError, base64.binascii.Error):
        raise Dense260ProtocolV2Error("issuer Ed25519 key is malformed") from None

    def take_string(data: bytes, offset: int) -> tuple[bytes, int]:
        if offset + 4 > len(data):
            raise Dense260ProtocolV2Error("issuer Ed25519 key blob is truncated")
        length = int.from_bytes(data[offset : offset + 4], "big")
        start = offset + 4
        end = start + length
        if end > len(data):
            raise Dense260ProtocolV2Error("issuer Ed25519 key blob is truncated")
        return data[start:end], end

    algorithm, offset = take_string(blob, 0)
    public, offset = take_string(blob, offset)
    if algorithm != b"ssh-ed25519" or len(public) != 32 or offset != len(blob):
        raise Dense260ProtocolV2Error("issuer Ed25519 key blob has wrong shape")
    return value, hashlib.sha256(blob).hexdigest()


def _source_record_template(role: str) -> tuple[tuple[Any, ...], ...]:
    return tuple(ROLE_TEMPLATES[role]["sources"])


def _validate_role_contract(value: Mapping[str, Any], role: str) -> dict[str, Any]:
    template = ROLE_TEMPLATES[role]
    expected_fields = {
        "reference_role",
        "role_namespace",
        "logical_group_count",
        "membership_semantics",
        "source_records",
        "identity_view_extractor_file_sha256",
        "reset_identity_contract_sha256",
    }
    records = value.get("source_records")
    if (
        set(value) != expected_fields
        or value.get("reference_role") != role
        or value.get("role_namespace") != template["role_namespace"]
        or value.get("logical_group_count") != template["logical_group_count"]
        or value.get("membership_semantics") != template["membership_semantics"]
        or value.get("reset_identity_contract_sha256")
        != RESET_IDENTITY_CONTRACT_SHA256
        or not _is_sha(value.get("identity_view_extractor_file_sha256"))
        or not isinstance(records, list)
        or len(records) != len(_source_record_template(role))
    ):
        raise Dense260ProtocolV2Error(f"{role} source-lineage contract changed")
    decoded: list[dict[str, Any]] = []
    record_fields = {
        "source_id",
        "source_format",
        "source_namespace",
        "logical_group_count",
        "file_sha256",
        "logical_sha256",
    }
    for record, expected in zip(records, _source_record_template(role), strict=True):
        if not isinstance(record, Mapping):
            raise Dense260ProtocolV2Error(f"{role} source record is invalid")
        source_id, source_format, source_namespace, count = expected
        if (
            set(record) != record_fields
            or record.get("source_id") != source_id
            or record.get("source_format") != source_format
            or record.get("source_namespace") != source_namespace
            or record.get("logical_group_count") != count
            or not _is_sha(record.get("file_sha256"))
            or not _is_sha(record.get("logical_sha256"))
        ):
            raise Dense260ProtocolV2Error(f"{role} source record changed")
        decoded.append(dict(record))
    return {**dict(value), "source_records": decoded}


def build_registry(spec: Mapping[str, Any], ssh_keygen_path: Path) -> dict[str, Any]:
    expected_spec_fields = {
        "format",
        "status",
        "protocol_namespace",
        "issuers",
        "role_source_contracts",
    }
    if (
        set(spec) != expected_spec_fields
        or spec.get("format") != REGISTRY_SPEC_FORMAT
        or spec.get("status") != "reviewed_public_keys_and_source_lineage"
        or spec.get("protocol_namespace") != PROTOCOL_NAMESPACE
    ):
        raise Dense260ProtocolV2Error("registry spec scope changed")
    verifier = _safe_regular(ssh_keygen_path, "ssh-keygen verifier", {""})
    issuers = spec.get("issuers")
    if not isinstance(issuers, list) or len(issuers) < 2:
        raise Dense260ProtocolV2Error("registry requires distinct source and reference issuers")
    decoded_issuers: list[dict[str, Any]] = []
    assignments: dict[tuple[str, str], str] = {}
    issuer_ids: set[str] = set()
    fingerprints: set[str] = set()
    issuer_fields = {
        "issuer_id",
        "principal",
        "public_key",
        "authorized_payloads",
    }
    for issuer in issuers:
        if not isinstance(issuer, Mapping) or set(issuer) != issuer_fields:
            raise Dense260ProtocolV2Error("issuer registry row schema changed")
        issuer_id = issuer.get("issuer_id")
        principal = issuer.get("principal")
        if (
            not isinstance(issuer_id, str)
            or not PRINCIPAL_PATTERN.fullmatch(issuer_id)
            or not isinstance(principal, str)
            or not PRINCIPAL_PATTERN.fullmatch(principal)
            or issuer_id in issuer_ids
        ):
            raise Dense260ProtocolV2Error("issuer identity is invalid or duplicated")
        public_key, fingerprint = _decode_ed25519_public_key(issuer.get("public_key"))
        if fingerprint in fingerprints:
            raise Dense260ProtocolV2Error("issuer Ed25519 keys must be distinct")
        authorizations = issuer.get("authorized_payloads")
        if not isinstance(authorizations, list) or not authorizations:
            raise Dense260ProtocolV2Error("issuer has no authorized payload")
        normalized_authorizations: list[dict[str, str]] = []
        for item in authorizations:
            if not isinstance(item, Mapping) or set(item) != {"payload_kind", "role"}:
                raise Dense260ProtocolV2Error("issuer authorization schema changed")
            kind, role = item.get("payload_kind"), item.get("role")
            valid = (
                kind == "candidate_reset_manifest" and role == TARGET_ROLE
            ) or (
                kind == "reference_identity_attestation" and role in REFERENCE_ROLES
            )
            key = (str(kind), str(role))
            if not valid or key in assignments:
                raise Dense260ProtocolV2Error("issuer authorization is invalid or duplicated")
            assignments[key] = issuer_id
            normalized_authorizations.append({"payload_kind": str(kind), "role": str(role)})
        issuer_ids.add(issuer_id)
        fingerprints.add(fingerprint)
        decoded_issuers.append(
            {
                "issuer_id": issuer_id,
                "principal": principal,
                "public_key": public_key,
                "public_key_blob_sha256": fingerprint,
                "authorized_payloads": normalized_authorizations,
            }
        )
    required_assignments = {("candidate_reset_manifest", TARGET_ROLE)} | {
        ("reference_identity_attestation", role) for role in REFERENCE_ROLES
    }
    if set(assignments) != required_assignments:
        raise Dense260ProtocolV2Error("issuer assignments do not cover the exact seven roles")
    source_issuer = assignments[("candidate_reset_manifest", TARGET_ROLE)]
    if any(
        assignments[("reference_identity_attestation", role)] == source_issuer
        for role in REFERENCE_ROLES
    ):
        raise Dense260ProtocolV2Error(
            "candidate reset issuer must be independent from every reference issuer"
        )
    raw_contracts = spec.get("role_source_contracts")
    if not isinstance(raw_contracts, list) or len(raw_contracts) != len(REFERENCE_ROLES):
        raise Dense260ProtocolV2Error("role source contracts are incomplete")
    by_role: dict[str, dict[str, Any]] = {}
    for item in raw_contracts:
        if not isinstance(item, Mapping) or item.get("reference_role") not in REFERENCE_ROLES:
            raise Dense260ProtocolV2Error("role source contract is invalid")
        role = str(item["reference_role"])
        if role in by_role:
            raise Dense260ProtocolV2Error("role source contract is duplicated")
        by_role[role] = _validate_role_contract(item, role)
    if set(by_role) != set(REFERENCE_ROLES):
        raise Dense260ProtocolV2Error("role source contract coverage changed")
    base = {
        "format": REGISTRY_FORMAT,
        "status": REGISTRY_STATUS,
        "protocol_namespace": PROTOCOL_NAMESPACE,
        "reset_identity_contract": RESET_IDENTITY_CONTRACT,
        "reset_identity_contract_sha256": RESET_IDENTITY_CONTRACT_SHA256,
        "signature_contract": {
            "format": "openssh_sshsig_ed25519_detached_v1",
            "namespace": SIGNATURE_NAMESPACE,
            "verifier_path": str(verifier),
            "verifier_file_sha256": file_sha256(verifier),
            "allowed_signers_materialized_ephemerally": True,
            "private_key_generation_or_access_by_protocol": False,
        },
        "issuers": decoded_issuers,
        "issuer_assignments": [
            {
                "payload_kind": kind,
                "role": role,
                "issuer_id": assignments[(kind, role)],
            }
            for kind, role in sorted(assignments)
        ],
        "role_source_contracts": [by_role[role] for role in REFERENCE_ROLES],
        "capability": {
            "public_keys_only": True,
            "private_keys_generated_or_opened": False,
            "environment_reset_or_step_calls": 0,
            "policy_or_action_calls": 0,
            "trajectory_hdf_or_label_files_opened": 0,
            "collection_authorized": False,
        },
        "trust_boundary": {
            "registry_requires_external_file_sha256_pin": True,
            "canonical_sha_is_not_an_issuer_signature": True,
            "signer_independence_is_key_separation_not_third_party_identity_proof": True,
        },
    }
    return _signed_content(base, "registry_sha256")


def validate_registry(value: Mapping[str, Any], *, verify_executable: bool) -> dict[str, Any]:
    logical = _verify_content_sha(value, "registry_sha256", "registry")
    expected_registry_fields = {
        "format",
        "status",
        "protocol_namespace",
        "reset_identity_contract",
        "reset_identity_contract_sha256",
        "signature_contract",
        "issuers",
        "issuer_assignments",
        "role_source_contracts",
        "capability",
        "trust_boundary",
        "registry_sha256",
    }
    if (
        set(value) != expected_registry_fields
        or value.get("format") != REGISTRY_FORMAT
        or value.get("status") != REGISTRY_STATUS
        or value.get("protocol_namespace") != PROTOCOL_NAMESPACE
        or value.get("reset_identity_contract") != RESET_IDENTITY_CONTRACT
        or value.get("reset_identity_contract_sha256")
        != RESET_IDENTITY_CONTRACT_SHA256
    ):
        raise Dense260ProtocolV2Error("registry scope or reset identity contract changed")
    signature = value.get("signature_contract")
    if not isinstance(signature, Mapping) or set(signature) != {
        "format",
        "namespace",
        "verifier_path",
        "verifier_file_sha256",
        "allowed_signers_materialized_ephemerally",
        "private_key_generation_or_access_by_protocol",
    }:
        raise Dense260ProtocolV2Error("registry signature contract changed")
    if (
        signature.get("format") != "openssh_sshsig_ed25519_detached_v1"
        or signature.get("namespace") != SIGNATURE_NAMESPACE
        or signature.get("allowed_signers_materialized_ephemerally") is not True
        or signature.get("private_key_generation_or_access_by_protocol") is not False
        or not _is_sha(signature.get("verifier_file_sha256"))
    ):
        raise Dense260ProtocolV2Error("registry signature verifier scope changed")
    if verify_executable:
        verifier = _safe_regular(Path(str(signature["verifier_path"])), "ssh-keygen verifier", {""})
        if file_sha256(verifier) != signature["verifier_file_sha256"]:
            raise Dense260ProtocolV2Error("ssh-keygen verifier file SHA changed")
    issuers = value.get("issuers")
    assignments = value.get("issuer_assignments")
    contracts = value.get("role_source_contracts")
    if not isinstance(issuers, list) or not isinstance(assignments, list) or not isinstance(contracts, list):
        raise Dense260ProtocolV2Error("registry inventories are incomplete")
    issuer_map: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    for issuer in issuers:
        if not isinstance(issuer, Mapping) or set(issuer) != {
            "issuer_id",
            "principal",
            "public_key",
            "public_key_blob_sha256",
            "authorized_payloads",
        }:
            raise Dense260ProtocolV2Error("registry issuer record changed")
        key, fingerprint = _decode_ed25519_public_key(issuer.get("public_key"))
        issuer_id, principal = issuer.get("issuer_id"), issuer.get("principal")
        if (
            not isinstance(issuer_id, str)
            or not PRINCIPAL_PATTERN.fullmatch(issuer_id)
            or not isinstance(principal, str)
            or not PRINCIPAL_PATTERN.fullmatch(principal)
            or issuer.get("public_key_blob_sha256") != fingerprint
            or issuer_id in issuer_map
            or fingerprint in fingerprints
        ):
            raise Dense260ProtocolV2Error("registry issuer identity changed")
        issuer_map[issuer_id] = {**dict(issuer), "public_key": key}
        fingerprints.add(fingerprint)
    observed_assignments: dict[tuple[str, str], str] = {}
    for item in assignments:
        if not isinstance(item, Mapping) or set(item) != {"payload_kind", "role", "issuer_id"}:
            raise Dense260ProtocolV2Error("registry issuer assignment changed")
        key = (str(item["payload_kind"]), str(item["role"]))
        issuer_id = str(item["issuer_id"])
        if key in observed_assignments or issuer_id not in issuer_map:
            raise Dense260ProtocolV2Error("registry issuer assignment is duplicated or unknown")
        observed_assignments[key] = issuer_id
    required = {("candidate_reset_manifest", TARGET_ROLE)} | {
        ("reference_identity_attestation", role) for role in REFERENCE_ROLES
    }
    if set(observed_assignments) != required:
        raise Dense260ProtocolV2Error("registry issuer assignment coverage changed")
    for key, issuer_id in observed_assignments.items():
        authorized = issuer_map[issuer_id].get("authorized_payloads")
        if (
            not isinstance(authorized, list)
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"payload_kind", "role"}
                for row in authorized
            )
            or {tuple(sorted(row.items())) for row in authorized} != {
            tuple(sorted({"payload_kind": kind, "role": role}.items()))
            for (kind, role), assigned in observed_assignments.items()
            if assigned == issuer_id
            }
        ):
            raise Dense260ProtocolV2Error("registry issuer authorization mirror changed")
    source_issuer = observed_assignments[("candidate_reset_manifest", TARGET_ROLE)]
    if any(
        observed_assignments[("reference_identity_attestation", role)] == source_issuer
        for role in REFERENCE_ROLES
    ):
        raise Dense260ProtocolV2Error("registry lost source/reference signer separation")
    role_contracts: dict[str, dict[str, Any]] = {}
    for item in contracts:
        if not isinstance(item, Mapping) or item.get("reference_role") not in REFERENCE_ROLES:
            raise Dense260ProtocolV2Error("registry role source contract changed")
        role = str(item["reference_role"])
        if role in role_contracts:
            raise Dense260ProtocolV2Error("registry role source contract duplicated")
        role_contracts[role] = _validate_role_contract(item, role)
    if set(role_contracts) != set(REFERENCE_ROLES):
        raise Dense260ProtocolV2Error("registry role source coverage changed")
    if value.get("capability") != {
        "public_keys_only": True,
        "private_keys_generated_or_opened": False,
        "environment_reset_or_step_calls": 0,
        "policy_or_action_calls": 0,
        "trajectory_hdf_or_label_files_opened": 0,
        "collection_authorized": False,
    }:
        raise Dense260ProtocolV2Error("registry capability changed")
    if value.get("trust_boundary") != {
        "registry_requires_external_file_sha256_pin": True,
        "canonical_sha_is_not_an_issuer_signature": True,
        "signer_independence_is_key_separation_not_third_party_identity_proof": True,
    }:
        raise Dense260ProtocolV2Error("registry trust boundary changed")
    return {
        "registry_sha256": logical,
        "signature_contract": dict(signature),
        "issuers": issuer_map,
        "assignments": observed_assignments,
        "role_contracts": role_contracts,
    }


def verify_detached_sshsig(
    *,
    registry: Mapping[str, Any],
    payload_kind: str,
    role: str,
    message: bytes,
    signature_path: Path,
) -> dict[str, str]:
    decoded = validate_registry(registry, verify_executable=True)
    issuer_id = decoded["assignments"].get((payload_kind, role))
    if issuer_id is None:
        raise Dense260ProtocolV2Error("payload role has no authorized issuer")
    issuer = decoded["issuers"][issuer_id]
    signature_raw, signature_sha = _read_bytes(
        signature_path, "detached SSHSIG", {".sig"}
    )
    if (
        len(signature_raw) > 64 * 1024
        or not signature_raw.startswith(b"-----BEGIN SSH SIGNATURE-----\n")
        or not signature_raw.rstrip().endswith(b"-----END SSH SIGNATURE-----")
    ):
        raise Dense260ProtocolV2Error("detached SSHSIG armor is invalid")
    verifier = decoded["signature_contract"]
    with tempfile.TemporaryDirectory(prefix="etsf_dense260_verify_") as temporary:
        root = Path(temporary)
        allowed = root / "allowed_signers"
        allowed.write_text(
            f"{issuer['principal']} {issuer['public_key']}\n",
            encoding="ascii",
        )
        allowed.chmod(0o600)
        try:
            completed = subprocess.run(
                [
                    str(verifier["verifier_path"]),
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed),
                    "-I",
                    str(issuer["principal"]),
                    "-n",
                    SIGNATURE_NAMESPACE,
                    "-s",
                    str(signature_path),
                ],
                input=message,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            )
        except (OSError, subprocess.TimeoutExpired):
            raise Dense260ProtocolV2Error("Ed25519 SSHSIG verifier is unavailable") from None
    if completed.returncode != 0:
        raise Dense260ProtocolV2Error("Ed25519 detached signature verification failed")
    return {
        "issuer_id": str(issuer_id),
        "principal": str(issuer["principal"]),
        "public_key_blob_sha256": str(issuer["public_key_blob_sha256"]),
        "signature_file_sha256": signature_sha,
    }


def reset_identity_sha256(
    *, task: str, instruction_semantics_receipt_sha256: str, initial_scene_state_sha256: str
) -> str:
    if (
        task != TASK
        or not _is_sha(instruction_semantics_receipt_sha256)
        or not _is_sha(initial_scene_state_sha256)
    ):
        raise Dense260ProtocolV2Error("reset identity v2 components are invalid")
    return canonical_sha256(
        {
            "format": RESET_IDENTITY_CONTRACT["format"],
            "task": task,
            "instruction_semantics_receipt_sha256": instruction_semantics_receipt_sha256,
            "initial_scene_state_sha256": initial_scene_state_sha256,
        }
    )


def _set_commitment(values: Sequence[int | str]) -> str:
    return canonical_sha256(sorted(set(values)))


def validate_candidate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = _verify_content_sha(value, "manifest_sha256", "candidate reset manifest")
    expected_fields = {
        "format",
        "status",
        "protocol_namespace",
        "task",
        "body",
        "policy",
        "candidate_range",
        "reset_identity_contract",
        "reset_identity_contract_sha256",
        "source_authority",
        "capability",
        "rows",
        "candidate_identity_counts",
        "candidate_identity_sets_sha256",
        "manifest_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != CANDIDATE_FORMAT
        or value.get("status") != CANDIDATE_STATUS
        or value.get("protocol_namespace") != PROTOCOL_NAMESPACE
        or value.get("task") != TASK
        or value.get("body") != BODY
        or value.get("policy") != POLICY
        or value.get("candidate_range")
        != {"start": CANDIDATE_START, "count": CANDIDATE_COUNT, "step": CANDIDATE_STEP}
        or value.get("reset_identity_contract") != RESET_IDENTITY_CONTRACT
        or value.get("reset_identity_contract_sha256")
        != RESET_IDENTITY_CONTRACT_SHA256
        or value.get("capability") != CANDIDATE_CAPABILITY
    ):
        raise Dense260ProtocolV2Error("candidate reset manifest v2 scope changed or v1 mixed in")
    authority = value.get("source_authority")
    authority_fields = {
        "reset_authority_file_sha256",
        "reset_authority_sha256",
        "reset_materializer_file_sha256",
        "reset_adapter_file_sha256",
        "runtime_contract_file_sha256",
        "runtime_contract_sha256",
        "runtime_source_closure_sha256",
    }
    if (
        not isinstance(authority, Mapping)
        or set(authority) != authority_fields
        or any(not _is_sha(authority.get(field)) for field in authority_fields)
    ):
        raise Dense260ProtocolV2Error("candidate reset source authority is incomplete")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != CANDIDATE_COUNT:
        raise Dense260ProtocolV2Error("candidate reset manifest must contain 400 rows")
    stable: list[Mapping[str, Any]] = []
    row_fields = {
        "ordinal",
        "requested_seed",
        "status",
        "resolved_seed",
        "instruction_semantics_receipt_sha256",
        "initial_scene_state_sha256",
        "reset_identity_sha256",
        "intent_sha256",
        "receipt_sha256",
    }
    for ordinal, row in enumerate(rows):
        requested = CANDIDATE_START + ordinal * CANDIDATE_STEP
        if (
            not isinstance(row, Mapping)
            or set(row) != row_fields
            or row.get("ordinal") != ordinal
            or row.get("requested_seed") != requested
            or not _is_sha(row.get("intent_sha256"))
            or not _is_sha(row.get("receipt_sha256"))
        ):
            raise Dense260ProtocolV2Error("candidate reset row identity changed")
        if row.get("status") == "stable_reset_identity_observed_v2":
            resolved = row.get("resolved_seed")
            semantics = row.get("instruction_semantics_receipt_sha256")
            scene = row.get("initial_scene_state_sha256")
            if (
                not _strict_int(resolved)
                or int(resolved) > MAX_SEED
                or not _is_sha(semantics)
                or not _is_sha(scene)
                or row.get("reset_identity_sha256")
                != reset_identity_sha256(
                    task=TASK,
                    instruction_semantics_receipt_sha256=str(semantics),
                    initial_scene_state_sha256=str(scene),
                )
            ):
                raise Dense260ProtocolV2Error("candidate reset identity v2 cannot be recomputed")
            stable.append(row)
        elif row.get("status") == "reset_identity_unavailable_failed_closed_v2":
            if any(
                row.get(field) is not None
                for field in (
                    "resolved_seed",
                    "instruction_semantics_receipt_sha256",
                    "initial_scene_state_sha256",
                    "reset_identity_sha256",
                )
            ):
                raise Dense260ProtocolV2Error("unavailable candidate fabricated reset identity")
        else:
            raise Dense260ProtocolV2Error("candidate reset row status is invalid")
    axes: dict[str, list[int | str]] = {
        "requested_seed": [int(row["requested_seed"]) for row in rows],
        "resolved_seed": [int(row["resolved_seed"]) for row in stable],
        "reset_identity": [str(row["reset_identity_sha256"]) for row in stable],
    }
    counts = {axis: len(set(axes[axis])) for axis in AXES}
    commitments = {axis: _set_commitment(axes[axis]) for axis in AXES}
    if (
        value.get("candidate_identity_counts") != counts
        or value.get("candidate_identity_sets_sha256") != commitments
    ):
        raise Dense260ProtocolV2Error("candidate identity commitments changed")
    return {
        "manifest_sha256": logical,
        "source_authority_sha256": canonical_sha256(dict(authority)),
        "rows": [dict(row) for row in rows],
        "identity_counts": counts,
        "identity_sets_sha256": commitments,
        "axes": axes,
    }


def validate_private_reference_view(
    value: Mapping[str, Any], *, role_contract: Mapping[str, Any]
) -> dict[str, Any]:
    _verify_content_sha(value, "view_sha256", "private reference identity view")
    role = str(role_contract["reference_role"])
    expected_fields = {
        "format",
        "status",
        "protocol_namespace",
        "reference_role",
        "role_namespace",
        "source_lineage_sha256",
        "reset_identity_contract_sha256",
        "rows",
        "capability",
        "view_sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("format") != PRIVATE_VIEW_FORMAT
        or value.get("status") != PRIVATE_VIEW_STATUS
        or value.get("protocol_namespace") != PROTOCOL_NAMESPACE
        or value.get("reference_role") != role
        or value.get("role_namespace") != role_contract["role_namespace"]
        or value.get("source_lineage_sha256") != canonical_sha256(role_contract)
        or value.get("reset_identity_contract_sha256")
        != RESET_IDENTITY_CONTRACT_SHA256
        or value.get("capability") != REFERENCE_CAPABILITY
    ):
        raise Dense260ProtocolV2Error("private reference view scope or v2 identity changed")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != role_contract["logical_group_count"]:
        raise Dense260ProtocolV2Error("private reference view group count changed")
    axes: dict[str, list[int | str]] = {axis: [] for axis in AXES}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "requested_seed",
            "resolved_seed",
            "reset_identity_sha256",
        }:
            raise Dense260ProtocolV2Error("private reference identity row changed")
        requested, resolved, reset_identity = (
            row.get("requested_seed"),
            row.get("resolved_seed"),
            row.get("reset_identity_sha256"),
        )
        if (
            not _strict_int(requested)
            or not _strict_int(resolved)
            or int(requested) > MAX_SEED
            or int(resolved) > MAX_SEED
            or not _is_sha(reset_identity)
        ):
            raise Dense260ProtocolV2Error("private reference identity is invalid")
        axes["requested_seed"].append(int(requested))
        axes["resolved_seed"].append(int(resolved))
        axes["reset_identity"].append(str(reset_identity))
    if len(set(axes["requested_seed"])) != len(rows):
        raise Dense260ProtocolV2Error("private reference requested identities duplicate")
    return {
        "axes": axes,
        "counts": {axis: len(set(axes[axis])) for axis in AXES},
        "commitments": {axis: _set_commitment(axes[axis]) for axis in AXES},
    }


def prepare_attestation_payload(
    *,
    registry: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_file_sha256: str,
    private_reference_view: Mapping[str, Any],
    reference_role: str,
    issuer_id: str,
) -> dict[str, Any]:
    decoded_registry = validate_registry(registry, verify_executable=False)
    if reference_role not in REFERENCE_ROLES:
        raise Dense260ProtocolV2Error("reference role is not frozen")
    expected_issuer = decoded_registry["assignments"][(
        "reference_identity_attestation",
        reference_role,
    )]
    if issuer_id != expected_issuer:
        raise Dense260ProtocolV2Error("attestation issuer is not authorized for role")
    if not _is_sha(candidate_manifest_file_sha256):
        raise Dense260ProtocolV2Error("candidate manifest file SHA is invalid")
    candidate = validate_candidate_manifest(candidate_manifest)
    role_contract = decoded_registry["role_contracts"][reference_role]
    reference = validate_private_reference_view(
        private_reference_view, role_contract=role_contract
    )
    intersections = {
        axis: len(set(candidate["axes"][axis]) & set(reference["axes"][axis]))
        for axis in AXES
    }
    if intersections != {axis: 0 for axis in AXES}:
        raise Dense260ProtocolV2Error("candidate and reference identities intersect")
    base = {
        "format": ATTESTATION_FORMAT,
        "status": ATTESTATION_STATUS,
        "protocol_namespace": PROTOCOL_NAMESPACE,
        "registry_sha256": decoded_registry["registry_sha256"],
        "issuer_id": issuer_id,
        "reference_role": reference_role,
        "role_namespace": role_contract["role_namespace"],
        "logical_group_count": role_contract["logical_group_count"],
        "role_source_contract_sha256": canonical_sha256(role_contract),
        "candidate_binding": {
            "file_sha256": candidate_manifest_file_sha256,
            "manifest_sha256": candidate["manifest_sha256"],
            "source_authority_sha256": candidate["source_authority_sha256"],
            "identity_counts": candidate["identity_counts"],
            "identity_sets_sha256": candidate["identity_sets_sha256"],
        },
        "reference_identity_counts": reference["counts"],
        "reference_identity_sets_sha256": reference["commitments"],
        "intersection_counts": intersections,
        "reset_identity_contract_sha256": RESET_IDENTITY_CONTRACT_SHA256,
        "sensitive_identity_values_included": False,
        "source_paths_included": False,
        "only_aggregate_commitments_disclosed": True,
        "environment_policy_action_hdf_or_label_access_by_attestor": False,
    }
    return _signed_content(base, "payload_sha256")


def validate_attestation_payload(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
) -> dict[str, Any]:
    logical = _verify_content_sha(value, "payload_sha256", "attestation payload")
    decoded_registry = validate_registry(registry, verify_executable=False)
    expected_fields = {
        "format",
        "status",
        "protocol_namespace",
        "registry_sha256",
        "issuer_id",
        "reference_role",
        "role_namespace",
        "logical_group_count",
        "role_source_contract_sha256",
        "candidate_binding",
        "reference_identity_counts",
        "reference_identity_sets_sha256",
        "intersection_counts",
        "reset_identity_contract_sha256",
        "sensitive_identity_values_included",
        "source_paths_included",
        "only_aggregate_commitments_disclosed",
        "environment_policy_action_hdf_or_label_access_by_attestor",
        "payload_sha256",
    }
    role = value.get("reference_role")
    if role not in REFERENCE_ROLES:
        raise Dense260ProtocolV2Error("attestation role is invalid")
    role = str(role)
    role_contract = decoded_registry["role_contracts"][role]
    expected_issuer = decoded_registry["assignments"][(
        "reference_identity_attestation",
        role,
    )]
    candidate_binding = value.get("candidate_binding")
    expected_candidate = {
        "file_sha256": candidate_file_sha256,
        "manifest_sha256": candidate["manifest_sha256"],
        "source_authority_sha256": candidate["source_authority_sha256"],
        "identity_counts": candidate["identity_counts"],
        "identity_sets_sha256": candidate["identity_sets_sha256"],
    }
    counts, commitments = (
        value.get("reference_identity_counts"),
        value.get("reference_identity_sets_sha256"),
    )
    if (
        set(value) != expected_fields
        or value.get("format") != ATTESTATION_FORMAT
        or value.get("status") != ATTESTATION_STATUS
        or value.get("protocol_namespace") != PROTOCOL_NAMESPACE
        or value.get("registry_sha256") != decoded_registry["registry_sha256"]
        or value.get("issuer_id") != expected_issuer
        or value.get("role_namespace") != role_contract["role_namespace"]
        or value.get("logical_group_count") != role_contract["logical_group_count"]
        or value.get("role_source_contract_sha256") != canonical_sha256(role_contract)
        or candidate_binding != expected_candidate
        or not isinstance(counts, Mapping)
        or set(counts) != set(AXES)
        or any(not _strict_int(counts.get(axis), 1) for axis in AXES)
        or counts.get("requested_seed") != role_contract["logical_group_count"]
        or any(int(counts[axis]) > role_contract["logical_group_count"] for axis in AXES)
        or not isinstance(commitments, Mapping)
        or set(commitments) != set(AXES)
        or any(not _is_sha(commitments.get(axis)) for axis in AXES)
        or value.get("intersection_counts") != {axis: 0 for axis in AXES}
        or value.get("reset_identity_contract_sha256")
        != RESET_IDENTITY_CONTRACT_SHA256
        or value.get("sensitive_identity_values_included") is not False
        or value.get("source_paths_included") is not False
        or value.get("only_aggregate_commitments_disclosed") is not True
        or value.get("environment_policy_action_hdf_or_label_access_by_attestor")
        is not False
    ):
        raise Dense260ProtocolV2Error("attestation payload scope, source, or candidate binding changed")
    return {"reference_role": role, "issuer_id": str(expected_issuer), "payload_sha256": logical}


def _select_and_partition(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_resolved: set[int] = set()
    for row in rows:
        if len(selected) == SELECTED_GROUPS:
            break
        if row["status"] != "stable_reset_identity_observed_v2":
            continue
        resolved = int(row["resolved_seed"])
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        selected.append(
            {
                "selection_ordinal": len(selected),
                "candidate_ordinal": int(row["ordinal"]),
                "requested_seed": int(row["requested_seed"]),
                "resolved_seed": resolved,
                "reset_identity_sha256": str(row["reset_identity_sha256"]),
            }
        )
    if len(selected) != SELECTED_GROUPS:
        raise Dense260ProtocolV2Error("fewer than 260 resolved-unique candidate rows")
    if len({row["reset_identity_sha256"] for row in selected}) != SELECTED_GROUPS:
        raise Dense260ProtocolV2Error("first 260 resolved-unique rows duplicate reset identity v2")
    ordered = sorted(
        selected,
        key=lambda row: canonical_sha256(
            {
                "protocol_namespace": PROTOCOL_NAMESPACE,
                "requested_seed": row["requested_seed"],
                "resolved_seed": row["resolved_seed"],
                "reset_identity_sha256": row["reset_identity_sha256"],
            }
        ),
    )
    result: list[dict[str, Any]] = []
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        for split_ordinal, row in enumerate(ordered[offset : offset + count]):
            split_order_sha = canonical_sha256(
                {
                    "protocol_namespace": PROTOCOL_NAMESPACE,
                    "requested_seed": row["requested_seed"],
                    "resolved_seed": row["resolved_seed"],
                    "reset_identity_sha256": row["reset_identity_sha256"],
                }
            )
            result.append(
                {
                    "global_ordinal": len(result),
                    "split": split,
                    "split_ordinal": split_ordinal,
                    **row,
                    "split_order_sha256": split_order_sha,
                    "group_id": canonical_sha256(
                        {
                            "task": TASK,
                            "body": BODY,
                            "policy": POLICY,
                            "requested_seed": row["requested_seed"],
                            "resolved_seed": row["resolved_seed"],
                            "reset_identity_sha256": row["reset_identity_sha256"],
                        }
                    ),
                }
            )
        offset += count
    for axis, field in {
        "requested_seed": "requested_seed",
        "resolved_seed": "resolved_seed",
        "reset_identity": "reset_identity_sha256",
    }.items():
        split_sets = {
            split: {row[field] for row in result if row["split"] == split}
            for split in SPLIT_COUNTS
        }
        names = list(split_sets)
        if sum(map(len, split_sets.values())) != SELECTED_GROUPS or any(
            split_sets[names[left]] & split_sets[names[right]]
            for left in range(len(names))
            for right in range(left + 1, len(names))
        ):
            raise Dense260ProtocolV2Error(f"selected {axis} split isolation failed")
    return result


def build_freeze(
    *,
    registry: Mapping[str, Any],
    registry_file_sha256: str,
    expected_registry_sha256: str,
    candidate_manifest: Mapping[str, Any],
    candidate_raw: bytes,
    candidate_file_sha256: str,
    candidate_signature_path: Path,
    attestations: Sequence[tuple[Mapping[str, Any], bytes, str, Path]],
) -> dict[str, Any]:
    decoded_registry = validate_registry(registry, verify_executable=True)
    if (
        not _is_sha(registry_file_sha256)
        or decoded_registry["registry_sha256"] != expected_registry_sha256
    ):
        raise Dense260ProtocolV2Error("externally pinned registry binding changed")
    candidate = validate_candidate_manifest(candidate_manifest)
    if (
        not _is_sha(candidate_file_sha256)
        or hashlib.sha256(candidate_raw).hexdigest() != candidate_file_sha256
        or candidate_raw != canonical_bytes(candidate_manifest) + b"\n"
    ):
        raise Dense260ProtocolV2Error("candidate manifest file binding or canonical bytes changed")
    candidate_signature = verify_detached_sshsig(
        registry=registry,
        payload_kind="candidate_reset_manifest",
        role=TARGET_ROLE,
        message=candidate_raw,
        signature_path=candidate_signature_path,
    )
    if len(attestations) != len(REFERENCE_ROLES):
        raise Dense260ProtocolV2Error("exactly six detached attestations are required")
    records: dict[str, dict[str, Any]] = {}
    for value, raw, file_digest, signature_path in attestations:
        if hashlib.sha256(raw).hexdigest() != file_digest or raw != canonical_bytes(value) + b"\n":
            raise Dense260ProtocolV2Error("attestation payload file binding changed")
        decoded = validate_attestation_payload(
            value,
            registry=registry,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
        )
        role = decoded["reference_role"]
        if role in records:
            raise Dense260ProtocolV2Error("attestation role duplicated")
        signature = verify_detached_sshsig(
            registry=registry,
            payload_kind="reference_identity_attestation",
            role=role,
            message=raw,
            signature_path=signature_path,
        )
        if signature["issuer_id"] != decoded["issuer_id"]:
            raise Dense260ProtocolV2Error("attestation signature issuer mismatch")
        records[role] = {
            "payload": dict(value),
            "payload_file_sha256": file_digest,
            **signature,
        }
    if set(records) != set(REFERENCE_ROLES):
        raise Dense260ProtocolV2Error("attestation role coverage is incomplete")
    groups = _select_and_partition(candidate["rows"])
    base = {
        "format": FREEZE_FORMAT,
        "status": FREEZE_STATUS,
        "protocol_namespace": PROTOCOL_NAMESPACE,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "registry_binding": {
            "file_sha256": registry_file_sha256,
            "registry_sha256": decoded_registry["registry_sha256"],
            "reset_identity_contract_sha256": RESET_IDENTITY_CONTRACT_SHA256,
        },
        "candidate_binding": {
            "file_sha256": candidate_file_sha256,
            "manifest_sha256": candidate["manifest_sha256"],
            "source_authority_sha256": candidate["source_authority_sha256"],
            **candidate_signature,
        },
        "signed_reference_attestations": [records[role] for role in REFERENCE_ROLES],
        "partition_contract": {
            "candidate_range": {
                "start": CANDIDATE_START,
                "count": CANDIDATE_COUNT,
                "step": CANDIDATE_STEP,
            },
            "selection": (
                "first_260_stable_resolved_unique_then_require_reset_identity_v2_unique"
            ),
            "assignment": "canonical_sha256_order_v2",
            "split_counts": dict(SPLIT_COUNTS),
            "requested_resolved_reset_identity_split_overlap": 0,
            "labels_or_policy_outputs_used": False,
        },
        "groups": groups,
        "collection_contract": {
            "schema_version": 5,
            "event_vocab": list(EVENTS),
            "candidate_count": 8,
            "action_exec_steps": 5,
            "max_steps": 200,
            "action_chunk": 50,
            "action_dim": 14,
            "root_candidate_state_bit_exact_required": True,
            "event_transition_duration_outcome_recovery_object_and_uncertainty_supervision_required": True,
        },
        "capability": {
            "private_keys_generated_or_opened": False,
            "environment_reset_or_step_calls": 0,
            "policy_or_action_calls": 0,
            "trajectory_hdf_or_label_files_opened": 0,
            "collection_authorized": False,
            "training_authorized": False,
            "performance_or_cross_body_claim_authorized": False,
        },
    }
    return _signed_content(base, "preregistration_sha256")


def _load_pinned_registry(
    path: Path, expected_file_sha256: str, expected_registry_sha256: str
) -> tuple[dict[str, Any], bytes, str]:
    registry, raw, digest = _read_json(path, "issuer registry", canonical_file=True)
    if digest != expected_file_sha256:
        raise Dense260ProtocolV2Error("issuer registry external file SHA changed")
    decoded = validate_registry(registry, verify_executable=True)
    if decoded["registry_sha256"] != expected_registry_sha256:
        raise Dense260ProtocolV2Error("issuer registry external logical SHA changed")
    return registry, raw, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    registry = commands.add_parser("freeze-registry")
    registry.add_argument("--spec", type=Path, required=True)
    registry.add_argument("--ssh-keygen", type=Path, required=True)
    registry.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare-attestation")
    prepare.add_argument("--registry", type=Path, required=True)
    prepare.add_argument("--registry-file-sha256", required=True)
    prepare.add_argument("--registry-sha256", required=True)
    prepare.add_argument("--candidate-manifest", type=Path, required=True)
    prepare.add_argument("--candidate-file-sha256", required=True)
    prepare.add_argument("--candidate-signature", type=Path, required=True)
    prepare.add_argument("--private-reference-view", type=Path, required=True)
    prepare.add_argument("--reference-role", choices=REFERENCE_ROLES, required=True)
    prepare.add_argument("--issuer-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--registry-file-sha256", required=True)
    freeze.add_argument("--registry-sha256", required=True)
    freeze.add_argument("--candidate-manifest", type=Path, required=True)
    freeze.add_argument("--candidate-file-sha256", required=True)
    freeze.add_argument("--candidate-signature", type=Path, required=True)
    freeze.add_argument("--attestation-payload", action="append", type=Path, required=True)
    freeze.add_argument("--attestation-signature", action="append", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-registry":
        spec, _, _ = _read_json(args.spec, "registry spec", canonical_file=False)
        value = build_registry(spec, args.ssh_keygen)
        _write_new(args.output, value)
        public = {"registry_sha256": value["registry_sha256"], "collection_authorized": False}
    elif args.command == "prepare-attestation":
        registry, _, _ = _load_pinned_registry(
            args.registry, args.registry_file_sha256, args.registry_sha256
        )
        candidate, candidate_raw, candidate_digest = _read_json(
            args.candidate_manifest, "candidate manifest", canonical_file=True
        )
        if candidate_digest != args.candidate_file_sha256:
            raise Dense260ProtocolV2Error("candidate manifest expected file SHA changed")
        verify_detached_sshsig(
            registry=registry,
            payload_kind="candidate_reset_manifest",
            role=TARGET_ROLE,
            message=candidate_raw,
            signature_path=args.candidate_signature,
        )
        private, _, _ = _read_private_json(
            args.private_reference_view, "private reference identity view"
        )
        value = prepare_attestation_payload(
            registry=registry,
            candidate_manifest=candidate,
            candidate_manifest_file_sha256=candidate_digest,
            private_reference_view=private,
            reference_role=args.reference_role,
            issuer_id=args.issuer_id,
        )
        _write_new(args.output, value)
        public = {
            "reference_role": args.reference_role,
            "payload_sha256": value["payload_sha256"],
            "external_detached_signature_required": True,
        }
    else:
        registry, _, registry_digest = _load_pinned_registry(
            args.registry, args.registry_file_sha256, args.registry_sha256
        )
        candidate, candidate_raw, candidate_digest = _read_json(
            args.candidate_manifest, "candidate manifest", canonical_file=True
        )
        if candidate_digest != args.candidate_file_sha256:
            raise Dense260ProtocolV2Error("candidate manifest expected file SHA changed")
        if len(args.attestation_payload) != len(args.attestation_signature):
            raise Dense260ProtocolV2Error("attestation payload/signature counts differ")
        attestations = []
        for index, (payload_path, signature_path) in enumerate(
            zip(args.attestation_payload, args.attestation_signature, strict=True)
        ):
            payload, raw, digest = _read_json(
                payload_path, f"attestation payload {index}", canonical_file=True
            )
            attestations.append((payload, raw, digest, signature_path))
        value = build_freeze(
            registry=registry,
            registry_file_sha256=registry_digest,
            expected_registry_sha256=args.registry_sha256,
            candidate_manifest=candidate,
            candidate_raw=candidate_raw,
            candidate_file_sha256=candidate_digest,
            candidate_signature_path=args.candidate_signature,
            attestations=attestations,
        )
        _write_new(args.output, value)
        public = {
            "preregistration_sha256": value["preregistration_sha256"],
            "groups": SELECTED_GROUPS,
            "collection_authorized": False,
        }
    print(json.dumps(public, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTESTATION_FORMAT",
    "CANDIDATE_CAPABILITY",
    "CANDIDATE_COUNT",
    "CANDIDATE_FORMAT",
    "CANDIDATE_START",
    "CANDIDATE_STATUS",
    "Dense260ProtocolV2Error",
    "PRIVATE_VIEW_FORMAT",
    "PRIVATE_VIEW_STATUS",
    "PROTOCOL_NAMESPACE",
    "REFERENCE_CAPABILITY",
    "REFERENCE_ROLES",
    "REGISTRY_SPEC_FORMAT",
    "RESET_IDENTITY_CONTRACT",
    "RESET_IDENTITY_CONTRACT_SHA256",
    "ROLE_TEMPLATES",
    "SPLIT_COUNTS",
    "TARGET_ROLE",
    "build_freeze",
    "build_registry",
    "canonical_bytes",
    "canonical_sha256",
    "prepare_attestation_payload",
    "reset_identity_sha256",
    "validate_attestation_payload",
    "validate_candidate_manifest",
    "validate_private_reference_view",
    "validate_registry",
    "verify_detached_sshsig",
]
