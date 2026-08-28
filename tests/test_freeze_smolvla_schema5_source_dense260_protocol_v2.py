from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_smolvla_schema5_source_dense260_protocol_v2.py"
)
SPEC = importlib.util.spec_from_file_location("dense260_protocol_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


def _content_signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = dense.canonical_sha256(result)
    return result


def _public_key(public_hex: str) -> str:
    algorithm = b"ssh-ed25519"
    public = bytes.fromhex(public_hex)
    blob = (
        struct.pack(">I", len(algorithm))
        + algorithm
        + struct.pack(">I", len(public))
        + public
    )
    return "ssh-ed25519 " + base64.b64encode(blob).decode("ascii")


SOURCE_PUBLIC_KEY = _public_key(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)
REFERENCE_PUBLIC_KEY = _public_key(
    "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
)


def _fake_verifier(tmp_path: Path) -> Path:
    path = tmp_path / "ssh-keygen"
    path.write_text(
        """#!/bin/sh
sig=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = '-s' ]; then
    shift
    sig="$1"
  fi
  shift
done
cat >/dev/null
grep -q 'ACCEPT' "$sig"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _signature(tmp_path: Path, name: str, *, accept: bool = True) -> Path:
    path = tmp_path / f"{name}.sig"
    decision = "ACCEPT" if accept else "REJECT"
    path.write_text(
        "-----BEGIN SSH SIGNATURE-----\n"
        + decision
        + "\n-----END SSH SIGNATURE-----\n",
        encoding="ascii",
    )
    return path


def _role_contract(role: str) -> dict[str, Any]:
    template = dense.ROLE_TEMPLATES[role]
    records = []
    for source_id, source_format, source_namespace, count in template["sources"]:
        records.append(
            {
                "source_id": source_id,
                "source_format": source_format,
                "source_namespace": source_namespace,
                "logical_group_count": count,
                "file_sha256": dense.canonical_sha256(
                    {"role": role, "source": source_id, "binding": "file"}
                ),
                "logical_sha256": dense.canonical_sha256(
                    {"role": role, "source": source_id, "binding": "logical"}
                ),
            }
        )
    return {
        "reference_role": role,
        "role_namespace": template["role_namespace"],
        "logical_group_count": template["logical_group_count"],
        "membership_semantics": template["membership_semantics"],
        "source_records": records,
        "identity_view_extractor_file_sha256": dense.canonical_sha256(
            {"identity_extractor_for": role}
        ),
        "reset_identity_contract_sha256": dense.RESET_IDENTITY_CONTRACT_SHA256,
    }


def _registry_spec() -> dict[str, Any]:
    return {
        "format": dense.REGISTRY_SPEC_FORMAT,
        "status": "reviewed_public_keys_and_source_lineage",
        "protocol_namespace": dense.PROTOCOL_NAMESPACE,
        "issuers": [
            {
                "issuer_id": "source-reset-issuer-v1",
                "principal": "source-reset@example.test",
                "public_key": SOURCE_PUBLIC_KEY,
                "authorized_payloads": [
                    {
                        "payload_kind": "candidate_reset_manifest",
                        "role": dense.TARGET_ROLE,
                    }
                ],
            },
            {
                "issuer_id": "reference-custodian-v1",
                "principal": "reference-custodian@example.test",
                "public_key": REFERENCE_PUBLIC_KEY,
                "authorized_payloads": [
                    {
                        "payload_kind": "reference_identity_attestation",
                        "role": role,
                    }
                    for role in dense.REFERENCE_ROLES
                ],
            },
        ],
        "role_source_contracts": [
            _role_contract(role) for role in dense.REFERENCE_ROLES
        ],
    }


def _registry(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    verifier = _fake_verifier(tmp_path)
    return dense.build_registry(_registry_spec(), verifier), verifier


def _candidate_manifest() -> dict[str, Any]:
    semantics = dense.canonical_sha256({"fixed_instruction_semantics": "can-inside-pot"})
    rows = []
    for ordinal in range(dense.CANDIDATE_COUNT):
        requested = dense.CANDIDATE_START + ordinal
        scene = dense.canonical_sha256({"scene": ordinal, "can": [0.0], "pot": [1.0]})
        rows.append(
            {
                "ordinal": ordinal,
                "requested_seed": requested,
                "status": "stable_reset_identity_observed_v2",
                "resolved_seed": requested,
                "instruction_semantics_receipt_sha256": semantics,
                "initial_scene_state_sha256": scene,
                "reset_identity_sha256": dense.reset_identity_sha256(
                    task="move_can_pot",
                    instruction_semantics_receipt_sha256=semantics,
                    initial_scene_state_sha256=scene,
                ),
                "intent_sha256": dense.canonical_sha256({"intent": ordinal}),
                "receipt_sha256": dense.canonical_sha256({"receipt": ordinal}),
            }
        )
    axes = {
        "requested_seed": [row["requested_seed"] for row in rows],
        "resolved_seed": [row["resolved_seed"] for row in rows],
        "reset_identity": [row["reset_identity_sha256"] for row in rows],
    }
    base = {
        "format": dense.CANDIDATE_FORMAT,
        "status": dense.CANDIDATE_STATUS,
        "protocol_namespace": dense.PROTOCOL_NAMESPACE,
        "task": "move_can_pot",
        "body": "aloha-agilex",
        "policy": "smolvla",
        "candidate_range": {
            "start": dense.CANDIDATE_START,
            "count": dense.CANDIDATE_COUNT,
            "step": 1,
        },
        "reset_identity_contract": copy.deepcopy(dense.RESET_IDENTITY_CONTRACT),
        "reset_identity_contract_sha256": dense.RESET_IDENTITY_CONTRACT_SHA256,
        "source_authority": {
            field: dense.canonical_sha256({"source_authority_field": field})
            for field in (
                "reset_authority_file_sha256",
                "reset_authority_sha256",
                "reset_materializer_file_sha256",
                "reset_adapter_file_sha256",
                "runtime_contract_file_sha256",
                "runtime_contract_sha256",
                "runtime_source_closure_sha256",
            )
        },
        "capability": copy.deepcopy(dense.CANDIDATE_CAPABILITY),
        "rows": rows,
        "candidate_identity_counts": {
            axis: len(set(values)) for axis, values in axes.items()
        },
        "candidate_identity_sets_sha256": {
            axis: dense.canonical_sha256(sorted(set(values)))
            for axis, values in axes.items()
        },
    }
    return _content_signed(base, "manifest_sha256")


def _private_view(
    registry: dict[str, Any], role: str, *, overlap_axis: str | None = None
) -> dict[str, Any]:
    decoded = dense.validate_registry(registry, verify_executable=False)
    contract = decoded["role_contracts"][role]
    candidate = _candidate_manifest()
    candidate_row = candidate["rows"][0]
    count = contract["logical_group_count"]
    offset = list(dense.REFERENCE_ROLES).index(role) * 10_000
    rows = []
    for index in range(count):
        requested = 10_000_000 + offset + index
        resolved = 20_000_000 + offset + index
        reset_identity = dense.canonical_sha256({"reference": role, "row": index})
        if index == 0 and overlap_axis == "requested_seed":
            requested = candidate_row["requested_seed"]
        elif index == 0 and overlap_axis == "resolved_seed":
            resolved = candidate_row["resolved_seed"]
        elif index == 0 and overlap_axis == "reset_identity":
            reset_identity = candidate_row["reset_identity_sha256"]
        rows.append(
            {
                "requested_seed": requested,
                "resolved_seed": resolved,
                "reset_identity_sha256": reset_identity,
            }
        )
    base = {
        "format": dense.PRIVATE_VIEW_FORMAT,
        "status": dense.PRIVATE_VIEW_STATUS,
        "protocol_namespace": dense.PROTOCOL_NAMESPACE,
        "reference_role": role,
        "role_namespace": contract["role_namespace"],
        "source_lineage_sha256": dense.canonical_sha256(contract),
        "reset_identity_contract_sha256": dense.RESET_IDENTITY_CONTRACT_SHA256,
        "rows": rows,
        "capability": copy.deepcopy(dense.REFERENCE_CAPABILITY),
    }
    return _content_signed(base, "view_sha256")


def _attestations(
    registry: dict[str, Any], candidate: dict[str, Any], candidate_file_sha: str
) -> list[dict[str, Any]]:
    return [
        dense.prepare_attestation_payload(
            registry=registry,
            candidate_manifest=candidate,
            candidate_manifest_file_sha256=candidate_file_sha,
            private_reference_view=_private_view(registry, role),
            reference_role=role,
            issuer_id="reference-custodian-v1",
        )
        for role in dense.REFERENCE_ROLES
    ]


def test_registry_freezes_exact_roles_keys_and_create_once(tmp_path: Path) -> None:
    registry, verifier = _registry(tmp_path)
    decoded = dense.validate_registry(registry, verify_executable=True)

    assert set(decoded["role_contracts"]) == set(dense.REFERENCE_ROLES)
    assert decoded["assignments"][("candidate_reset_manifest", dense.TARGET_ROLE)] == (
        "source-reset-issuer-v1"
    )
    assert all(
        decoded["assignments"][("reference_identity_attestation", role)]
        == "reference-custodian-v1"
        for role in dense.REFERENCE_ROLES
    )
    assert registry["signature_contract"]["verifier_file_sha256"] == hashlib.sha256(
        verifier.read_bytes()
    ).hexdigest()
    output = tmp_path / "registry.json"
    dense._write_new(output, registry)
    assert output.stat().st_mode & 0o200 == 0
    with pytest.raises(FileExistsError):
        dense._write_new(output, registry)


def test_registry_rejects_same_key_or_incomplete_role_authority(tmp_path: Path) -> None:
    spec = _registry_spec()
    spec["issuers"][1]["public_key"] = SOURCE_PUBLIC_KEY
    with pytest.raises(dense.Dense260ProtocolV2Error, match="keys must be distinct"):
        dense.build_registry(spec, _fake_verifier(tmp_path))

    spec = _registry_spec()
    spec["issuers"][1]["authorized_payloads"].pop()
    with pytest.raises(dense.Dense260ProtocolV2Error, match="exact seven roles"):
        dense.build_registry(spec, _fake_verifier(tmp_path))


def test_candidate_recomputes_v2_identity_and_rejects_v1_mix() -> None:
    candidate = _candidate_manifest()
    decoded = dense.validate_candidate_manifest(candidate)
    assert decoded["identity_counts"] == {axis: 400 for axis in dense.AXES}

    old = copy.deepcopy(candidate)
    old["reset_identity_contract"]["format"] = "etsf_cross_body_semantic_reset_identity_v1"
    old = _content_signed(old, "manifest_sha256")
    with pytest.raises(dense.Dense260ProtocolV2Error, match="v1 mixed"):
        dense.validate_candidate_manifest(old)

    old = copy.deepcopy(candidate)
    old["format"] = "etsf_smolvla_schema5_source_dense260_reset_candidate_manifest_v1"
    old = _content_signed(old, "manifest_sha256")
    with pytest.raises(dense.Dense260ProtocolV2Error, match="v1 mixed"):
        dense.validate_candidate_manifest(old)


@pytest.mark.parametrize("axis", dense.AXES)
def test_attestor_fails_closed_on_each_identity_axis(
    tmp_path: Path, axis: str
) -> None:
    registry, _ = _registry(tmp_path)
    candidate = _candidate_manifest()
    candidate_sha = hashlib.sha256(dense.canonical_bytes(candidate) + b"\n").hexdigest()
    view = _private_view(registry, "official150", overlap_axis=axis)

    with pytest.raises(dense.Dense260ProtocolV2Error, match="identities intersect"):
        dense.prepare_attestation_payload(
            registry=registry,
            candidate_manifest=candidate,
            candidate_manifest_file_sha256=candidate_sha,
            private_reference_view=view,
            reference_role="official150",
            issuer_id="reference-custodian-v1",
        )


def test_attestation_is_aggregate_only_and_binds_registry_source_candidate(
    tmp_path: Path,
) -> None:
    registry, _ = _registry(tmp_path)
    candidate = _candidate_manifest()
    candidate_sha = hashlib.sha256(dense.canonical_bytes(candidate) + b"\n").hexdigest()
    view = _private_view(registry, "official150")
    payload = dense.prepare_attestation_payload(
        registry=registry,
        candidate_manifest=candidate,
        candidate_manifest_file_sha256=candidate_sha,
        private_reference_view=view,
        reference_role="official150",
        issuer_id="reference-custodian-v1",
    )
    decoded_registry = dense.validate_registry(registry, verify_executable=False)

    assert payload["registry_sha256"] == registry["registry_sha256"]
    assert payload["role_source_contract_sha256"] == dense.canonical_sha256(
        decoded_registry["role_contracts"]["official150"]
    )
    assert payload["candidate_binding"]["file_sha256"] == candidate_sha
    assert payload["intersection_counts"] == {axis: 0 for axis in dense.AXES}
    assert "rows" not in json.dumps(payload, sort_keys=True)
    assert payload["sensitive_identity_values_included"] is False
    assert payload["source_paths_included"] is False


def test_identity_view_file_requires_owner_only_0400(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    view = _private_view(registry, "official150")
    path = tmp_path / "identity_view.json"
    path.write_bytes(dense.canonical_bytes(view) + b"\n")
    path.chmod(0o600)

    with pytest.raises(dense.Dense260ProtocolV2Error, match="owner-only 0400"):
        dense._read_private_json(path, "identity view")
    path.chmod(0o400)
    decoded, _, _ = dense._read_private_json(path, "identity view")
    assert decoded == view


def test_detached_verifier_accepts_only_pinned_authorized_path(
    tmp_path: Path,
) -> None:
    registry, verifier = _registry(tmp_path)
    message = b"canonical-payload\n"
    accepted = _signature(tmp_path, "accepted")
    rejected = _signature(tmp_path, "rejected", accept=False)

    receipt = dense.verify_detached_sshsig(
        registry=registry,
        payload_kind="candidate_reset_manifest",
        role=dense.TARGET_ROLE,
        message=message,
        signature_path=accepted,
    )
    assert receipt["issuer_id"] == "source-reset-issuer-v1"
    with pytest.raises(dense.Dense260ProtocolV2Error, match="verification failed"):
        dense.verify_detached_sshsig(
            registry=registry,
            payload_kind="candidate_reset_manifest",
            role=dense.TARGET_ROLE,
            message=message,
            signature_path=rejected,
        )

    verifier.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    with pytest.raises(dense.Dense260ProtocolV2Error, match="verifier file SHA changed"):
        dense.verify_detached_sshsig(
            registry=registry,
            payload_kind="candidate_reset_manifest",
            role=dense.TARGET_ROLE,
            message=message,
            signature_path=accepted,
        )


def test_real_openssh_rejects_malformed_detached_signature(tmp_path: Path) -> None:
    verifier = Path("/usr/bin/ssh-keygen")
    if not verifier.is_file():
        pytest.skip("system OpenSSH verifier is unavailable")
    registry = dense.build_registry(_registry_spec(), verifier)
    malformed = _signature(tmp_path, "malformed", accept=False)

    with pytest.raises(dense.Dense260ProtocolV2Error, match="verification failed"):
        dense.verify_detached_sshsig(
            registry=registry,
            payload_kind="candidate_reset_manifest",
            role=dense.TARGET_ROLE,
            message=b"canonical-payload\n",
            signature_path=malformed,
        )


def test_freezer_requires_six_signed_authorized_attestations(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    registry_raw = dense.canonical_bytes(registry) + b"\n"
    registry_file_sha = hashlib.sha256(registry_raw).hexdigest()
    candidate = _candidate_manifest()
    candidate_raw = dense.canonical_bytes(candidate) + b"\n"
    candidate_file_sha = hashlib.sha256(candidate_raw).hexdigest()
    candidate_signature = _signature(tmp_path, "candidate")
    payloads = _attestations(registry, candidate, candidate_file_sha)
    attestation_inputs = []
    for index, payload in enumerate(payloads):
        raw = dense.canonical_bytes(payload) + b"\n"
        attestation_inputs.append(
            (
                payload,
                raw,
                hashlib.sha256(raw).hexdigest(),
                _signature(tmp_path, f"att_{index}"),
            )
        )
    result = dense.build_freeze(
        registry=registry,
        registry_file_sha256=registry_file_sha,
        expected_registry_sha256=registry["registry_sha256"],
        candidate_manifest=candidate,
        candidate_raw=candidate_raw,
        candidate_file_sha256=candidate_file_sha,
        candidate_signature_path=candidate_signature,
        attestations=attestation_inputs,
    )

    assert len(result["groups"]) == 260
    assert {split: sum(row["split"] == split for row in result["groups"])
            for split in dense.SPLIT_COUNTS} == dense.SPLIT_COUNTS
    assert result["collection_contract"]["candidate_count"] == 8
    assert result["collection_contract"]["action_exec_steps"] == 5
    assert result["collection_contract"]["max_steps"] == 200
    assert result["capability"]["collection_authorized"] is False
    assert len(result["signed_reference_attestations"]) == 6
    assert result["preregistration_sha256"] == dense.canonical_sha256(
        {key: value for key, value in result.items() if key != "preregistration_sha256"}
    )

    with pytest.raises(dense.Dense260ProtocolV2Error, match="exactly six"):
        dense.build_freeze(
            registry=registry,
            registry_file_sha256=registry_file_sha,
            expected_registry_sha256=registry["registry_sha256"],
            candidate_manifest=candidate,
            candidate_raw=candidate_raw,
            candidate_file_sha256=candidate_file_sha,
            candidate_signature_path=candidate_signature,
            attestations=attestation_inputs[:-1],
        )


def test_freezer_rejects_rebound_candidate_or_unauthorized_issuer(
    tmp_path: Path,
) -> None:
    registry, _ = _registry(tmp_path)
    candidate = _candidate_manifest()
    candidate_file_sha = hashlib.sha256(
        dense.canonical_bytes(candidate) + b"\n"
    ).hexdigest()
    payload = _attestations(registry, candidate, candidate_file_sha)[0]
    decoded_candidate = dense.validate_candidate_manifest(candidate)

    rebound = copy.deepcopy(payload)
    rebound["candidate_binding"]["file_sha256"] = "0" * 64
    rebound = _content_signed(rebound, "payload_sha256")
    with pytest.raises(dense.Dense260ProtocolV2Error, match="binding changed"):
        dense.validate_attestation_payload(
            rebound,
            registry=registry,
            candidate=decoded_candidate,
            candidate_file_sha256=candidate_file_sha,
        )

    unauthorized = copy.deepcopy(payload)
    unauthorized["issuer_id"] = "source-reset-issuer-v1"
    unauthorized = _content_signed(unauthorized, "payload_sha256")
    with pytest.raises(dense.Dense260ProtocolV2Error, match="binding changed"):
        dense.validate_attestation_payload(
            unauthorized,
            registry=registry,
            candidate=decoded_candidate,
            candidate_file_sha256=candidate_file_sha,
        )


def test_no_private_key_generation_or_online_capability_is_present() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ssh-keygen\", \"-t\"" not in source
    assert "-Y\",\n                    \"sign\"" not in source
    assert "h5py" not in source
    assert "torch" not in source
    assert "RoboTwinEnv" not in source
