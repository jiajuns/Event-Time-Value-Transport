#!/usr/bin/env python3
"""Private, aggregate-only identity-set commitments and disjoint attestations.

Private identities are parsed only in this local process.  Published artifacts
contain aggregate counts/commitments and the exact disjoint-attestation object
accepted by ``smolvla_piper_target_seed_manifest.validate_disjoint_attestation``;
they never contain an identity value or input path.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from smolvla_piper_target_seed_manifest import (
    ATTESTATION_STATUS,
    canonical_sha256,
    validate_disjoint_attestation,
)


PRIVATE_SET_FORMAT = "etsf_private_identity_set_v1"
PRIVATE_SET_STATUS = "private_local_attestation_material_only"
COMMITMENT_FORMAT = "etsf_private_identity_set_commitment_v1"
COMMITMENT_STATUS = "committed_without_disclosing_private_identities"
ATTESTATION_FORMAT = "etsf_private_identity_disjoint_attestation_v1"
HELDOUT_ROLE = "heldout"
CANDIDATE_ROLE = "candidate_pool"
SELECTED_ROLE = "selected_requested_and_resolved"
CANDIDATE_TARGET_ROLE = "preregistered_reset_candidate_pool"
SELECTED_TARGET_ROLE = "selected_requested_and_resolved_target_identities"
SHA_CHARS = frozenset("0123456789abcdef")
MAX_PRIVATE_IDENTITIES = 10_000_000


class PrivateIdentityAttestationError(RuntimeError):
    """A private-set invariant failed; messages deliberately omit values."""


def _strict_json(raw: bytes, invalid_message: str) -> Any:
    def object_without_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_nonstandard_constant(_: str) -> None:
        raise ValueError

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_without_duplicate_keys,
            parse_constant=reject_nonstandard_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise PrivateIdentityAttestationError(invalid_message) from None


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _private_input_bytes(path: Path, role: str) -> bytes:
    """Read one owner-only, frozen, non-symlink file without path disclosure."""

    del role  # Role is intentionally not interpolated into low-level failures.
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError:
        raise PrivateIdentityAttestationError(
            "private identity input is unavailable or unsafe"
        ) from None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size < 2
        ):
            raise PrivateIdentityAttestationError(
                "private identity input must be a frozen owner-only regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if (
            remaining != 0
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise PrivateIdentityAttestationError(
                "private identity input changed during attestation"
            )
        return b"".join(chunks)
    except PrivateIdentityAttestationError:
        raise
    except OSError:
        raise PrivateIdentityAttestationError(
            "private identity input could not be read safely"
        ) from None
    finally:
        os.close(descriptor)


def _decode_private_set(path: Path, expected_role: str) -> dict[str, Any]:
    raw = _private_input_bytes(path, expected_role)
    value = _strict_json(raw, "private identity input is not strict JSON")
    if not isinstance(value, dict):
        raise PrivateIdentityAttestationError(
            "private identity input schema is invalid"
        )
    common = {
        "format": PRIVATE_SET_FORMAT,
        "status": PRIVATE_SET_STATUS,
        "role": expected_role,
    }
    expected_fields = (
        {"format", "status", "role", "requested", "resolved"}
        if expected_role == SELECTED_ROLE
        else {"format", "status", "role", "identities"}
    )
    if set(value) != expected_fields or any(value.get(key) != item for key, item in common.items()):
        raise PrivateIdentityAttestationError(
            "private identity input schema is invalid"
        )
    return value


def _identity_list(value: Any, role: str) -> list[int]:
    del role
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_PRIVATE_IDENTITIES
        or any(
            isinstance(identity, bool)
            or not isinstance(identity, int)
            or identity < 0
            for identity in value
        )
        or len(set(value)) != len(value)
    ):
        raise PrivateIdentityAttestationError(
            "private identity list is invalid or duplicated"
        )
    return list(value)


def load_single_set(path: Path, role: str) -> list[int]:
    value = _decode_private_set(path, role)
    return _identity_list(value["identities"], role)


def load_selected_set(path: Path) -> tuple[list[int], list[int]]:
    value = _decode_private_set(path, SELECTED_ROLE)
    requested = _identity_list(value["requested"], "selected requested")
    resolved = _identity_list(value["resolved"], "selected resolved")
    if len(requested) != len(resolved):
        raise PrivateIdentityAttestationError(
            "selected requested/resolved identity counts differ"
        )
    return requested, resolved


def heldout_identity_set_sha256(identities: Sequence[int]) -> str:
    """Order-independent heldout set commitment."""

    return canonical_sha256(sorted(identities))


def candidate_identity_set_sha256(identities: Sequence[int]) -> str:
    """Exact compatibility with target plan candidate-pool commitment."""

    return canonical_sha256(list(identities))


def selected_identity_set_sha256(
    requested: Sequence[int], resolved: Sequence[int]
) -> str:
    """Exact compatibility with target reset-receipt identity commitment."""

    return canonical_sha256(
        {"requested": list(requested), "resolved": list(resolved)}
    )


def _signed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise PrivateIdentityAttestationError(
            "aggregate output signature field already exists"
        )
    result[field] = canonical_sha256(result)
    return result


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir() or path.exists() or path.is_symlink():
            raise PrivateIdentityAttestationError(
                "aggregate output target is unavailable or already exists"
            )
        descriptor = os.open(
            os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400
        )
    except PrivateIdentityAttestationError:
        raise
    except OSError:
        raise PrivateIdentityAttestationError(
            "aggregate output target is unavailable or already exists"
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        raise PrivateIdentityAttestationError(
            "aggregate output could not be committed safely"
        ) from None


def _load_commitment(path: Path) -> dict[str, Any]:
    raw = _private_input_bytes(path, "heldout commitment")
    value = _strict_json(raw, "heldout commitment is not strict JSON")
    if not isinstance(value, dict):
        raise PrivateIdentityAttestationError("heldout commitment is invalid")
    unsigned = dict(value)
    recorded = unsigned.pop("commitment_sha256", None)
    if (
        set(value)
        != {
            "format",
            "status",
            "identity_role",
            "identity_count",
            "identity_set_sha256",
            "sensitive_identities_included",
            "commitment_sha256",
        }
        or value.get("format") != COMMITMENT_FORMAT
        or value.get("status") != COMMITMENT_STATUS
        or value.get("identity_role") != HELDOUT_ROLE
        or isinstance(value.get("identity_count"), bool)
        or not isinstance(value.get("identity_count"), int)
        or value.get("identity_count", 0) < 1
        or not _is_sha(value.get("identity_set_sha256"))
        or value.get("sensitive_identities_included") is not False
        or not _is_sha(recorded)
        or recorded != canonical_sha256(unsigned)
    ):
        raise PrivateIdentityAttestationError("heldout commitment is invalid")
    return value


def commit_heldout(private_heldout_path: Path, output_path: Path) -> dict[str, Any]:
    identities = load_single_set(private_heldout_path, HELDOUT_ROLE)
    receipt = _signed(
        {
            "format": COMMITMENT_FORMAT,
            "status": COMMITMENT_STATUS,
            "identity_role": HELDOUT_ROLE,
            "identity_count": len(identities),
            "identity_set_sha256": heldout_identity_set_sha256(identities),
            "sensitive_identities_included": False,
        },
        "commitment_sha256",
    )
    _immutable_json(output_path, receipt)
    return receipt


def _validate_heldout_against_commitment(
    private_heldout_path: Path, commitment_path: Path
) -> tuple[set[int], dict[str, Any]]:
    identities = load_single_set(private_heldout_path, HELDOUT_ROLE)
    commitment = _load_commitment(commitment_path)
    if (
        commitment["identity_count"] != len(identities)
        or commitment["identity_set_sha256"]
        != heldout_identity_set_sha256(identities)
    ):
        raise PrivateIdentityAttestationError(
            "private heldout set does not match its aggregate commitment"
        )
    return set(identities), commitment


def _attestation(
    *,
    target_role: str,
    heldout_sha256: str,
    target_sha256: str,
) -> dict[str, Any]:
    value = _signed(
        {
            "format": ATTESTATION_FORMAT,
            "status": ATTESTATION_STATUS,
            "target_role": target_role,
            "heldout_identity_set_sha256": heldout_sha256,
            "target_identity_set_sha256": target_sha256,
            "intersection_count": 0,
            "sensitive_identities_included": False,
        },
        "attestation_sha256",
    )
    # Prove exact downstream compatibility before publication.
    validate_disjoint_attestation(
        value,
        heldout_identity_set_sha256=heldout_sha256,
        target_identity_set_sha256=target_sha256,
        target_role=target_role,
    )
    return value


def attest_candidate_pool(
    *,
    private_heldout_path: Path,
    heldout_commitment_path: Path,
    private_candidate_pool_path: Path,
    expected_target_identity_set_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if not _is_sha(expected_target_identity_set_sha256):
        raise PrivateIdentityAttestationError(
            "expected target identity-set commitment is invalid"
        )
    heldout, commitment = _validate_heldout_against_commitment(
        private_heldout_path, heldout_commitment_path
    )
    candidates = load_single_set(private_candidate_pool_path, CANDIDATE_ROLE)
    target_sha = candidate_identity_set_sha256(candidates)
    if target_sha != expected_target_identity_set_sha256:
        raise PrivateIdentityAttestationError(
            "candidate-pool identity-set commitment does not match"
        )
    if heldout.intersection(candidates):
        raise PrivateIdentityAttestationError(
            "private identity sets are not disjoint"
        )
    value = _attestation(
        target_role=CANDIDATE_TARGET_ROLE,
        heldout_sha256=commitment["identity_set_sha256"],
        target_sha256=target_sha,
    )
    _immutable_json(output_path, value)
    return value


def attest_selected_requested_resolved(
    *,
    private_heldout_path: Path,
    heldout_commitment_path: Path,
    private_selected_path: Path,
    expected_target_identity_set_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if not _is_sha(expected_target_identity_set_sha256):
        raise PrivateIdentityAttestationError(
            "expected target identity-set commitment is invalid"
        )
    heldout, commitment = _validate_heldout_against_commitment(
        private_heldout_path, heldout_commitment_path
    )
    requested, resolved = load_selected_set(private_selected_path)
    target_sha = selected_identity_set_sha256(requested, resolved)
    if target_sha != expected_target_identity_set_sha256:
        raise PrivateIdentityAttestationError(
            "selected identity-set commitment does not match"
        )
    if heldout.intersection(requested) or heldout.intersection(resolved):
        raise PrivateIdentityAttestationError(
            "private identity sets are not disjoint"
        )
    value = _attestation(
        target_role=SELECTED_TARGET_ROLE,
        heldout_sha256=commitment["identity_set_sha256"],
        target_sha256=target_sha,
    )
    _immutable_json(output_path, value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commit = commands.add_parser("commit-heldout")
    commit.add_argument("--private-heldout", type=Path, required=True)
    commit.add_argument("--output", type=Path, required=True)

    candidate = commands.add_parser("attest-candidate-pool")
    candidate.add_argument("--private-heldout", type=Path, required=True)
    candidate.add_argument("--heldout-commitment", type=Path, required=True)
    candidate.add_argument("--private-candidate-pool", type=Path, required=True)
    candidate.add_argument("--expected-target-identity-set-sha256", required=True)
    candidate.add_argument("--output", type=Path, required=True)

    selected = commands.add_parser("attest-selected-requested-resolved")
    selected.add_argument("--private-heldout", type=Path, required=True)
    selected.add_argument("--heldout-commitment", type=Path, required=True)
    selected.add_argument("--private-selected", type=Path, required=True)
    selected.add_argument("--expected-target-identity-set-sha256", required=True)
    selected.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "commit-heldout":
        value = commit_heldout(args.private_heldout, args.output)
        public = {
            "identity_count": value["identity_count"],
            "identity_set_sha256": value["identity_set_sha256"],
            "commitment_sha256": value["commitment_sha256"],
        }
    elif args.command == "attest-candidate-pool":
        value = attest_candidate_pool(
            private_heldout_path=args.private_heldout,
            heldout_commitment_path=args.heldout_commitment,
            private_candidate_pool_path=args.private_candidate_pool,
            expected_target_identity_set_sha256=args.expected_target_identity_set_sha256,
            output_path=args.output,
        )
        public = {
            "intersection_count": 0,
            "attestation_sha256": value["attestation_sha256"],
        }
    else:
        value = attest_selected_requested_resolved(
            private_heldout_path=args.private_heldout,
            heldout_commitment_path=args.heldout_commitment,
            private_selected_path=args.private_selected,
            expected_target_identity_set_sha256=args.expected_target_identity_set_sha256,
            output_path=args.output,
        )
        public = {
            "intersection_count": 0,
            "attestation_sha256": value["attestation_sha256"],
        }
    print(json.dumps(public, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_TARGET_ROLE",
    "COMMITMENT_FORMAT",
    "HELDOUT_ROLE",
    "PRIVATE_SET_FORMAT",
    "PRIVATE_SET_STATUS",
    "PrivateIdentityAttestationError",
    "SELECTED_TARGET_ROLE",
    "attest_candidate_pool",
    "attest_selected_requested_resolved",
    "candidate_identity_set_sha256",
    "commit_heldout",
    "heldout_identity_set_sha256",
    "load_selected_set",
    "load_single_set",
    "selected_identity_set_sha256",
]
