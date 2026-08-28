from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import attest_private_target_identity_disjoint_v1 as attestor  # noqa: E402
from smolvla_piper_target_seed_manifest import (  # noqa: E402
    canonical_sha256,
    validate_disjoint_attestation,
)


def private_file(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o400)
    return path


def single(role: str, identities: list[int]) -> dict[str, object]:
    return {
        "format": attestor.PRIVATE_SET_FORMAT,
        "status": attestor.PRIVATE_SET_STATUS,
        "role": role,
        "identities": identities,
    }


def selected(requested: list[int], resolved: list[int]) -> dict[str, object]:
    return {
        "format": attestor.PRIVATE_SET_FORMAT,
        "status": attestor.PRIVATE_SET_STATUS,
        "role": attestor.SELECTED_ROLE,
        "requested": requested,
        "resolved": resolved,
    }


def commit(tmp_path: Path, heldout_values: list[int]) -> tuple[Path, Path, dict[str, object]]:
    heldout = private_file(
        tmp_path / "private-heldout.json",
        single(attestor.HELDOUT_ROLE, heldout_values),
    )
    commitment_path = tmp_path / "heldout-commitment.json"
    receipt = attestor.commit_heldout(heldout, commitment_path)
    return heldout, commitment_path, receipt


def test_commit_heldout_is_order_independent_aggregate_only_and_immutable(
    tmp_path: Path,
) -> None:
    secrets = [987_654_321, 123_456_789]
    heldout, commitment_path, receipt = commit(tmp_path, secrets)
    assert receipt == {
        "format": attestor.COMMITMENT_FORMAT,
        "status": attestor.COMMITMENT_STATUS,
        "identity_role": attestor.HELDOUT_ROLE,
        "identity_count": 2,
        "identity_set_sha256": canonical_sha256(sorted(secrets)),
        "sensitive_identities_included": False,
        "commitment_sha256": receipt["commitment_sha256"],
    }
    unsigned = dict(receipt)
    assert unsigned.pop("commitment_sha256") == canonical_sha256(unsigned)
    assert stat.S_IMODE(commitment_path.stat().st_mode) == 0o400
    serialized = commitment_path.read_text(encoding="utf-8")
    assert "identities" not in json.loads(serialized)
    assert all(str(secret) not in serialized for secret in secrets)

    reordered = private_file(
        tmp_path / "private-heldout-reordered.json",
        single(attestor.HELDOUT_ROLE, list(reversed(secrets))),
    )
    reordered_receipt = attestor.commit_heldout(
        reordered, tmp_path / "heldout-commitment-reordered.json"
    )
    assert reordered_receipt["identity_set_sha256"] == receipt["identity_set_sha256"]
    assert heldout.exists()


def test_candidate_attestation_exactly_matches_target_manifest_validator(
    tmp_path: Path,
) -> None:
    heldout, commitment_path, commitment = commit(
        tmp_path, [900_000_001, 900_000_002]
    )
    candidates = [101, 103, 107]
    private_candidates = private_file(
        tmp_path / "private-candidates.json",
        single(attestor.CANDIDATE_ROLE, candidates),
    )
    target_sha = canonical_sha256(candidates)
    output = tmp_path / "candidate-attestation.json"
    value = attestor.attest_candidate_pool(
        private_heldout_path=heldout,
        heldout_commitment_path=commitment_path,
        private_candidate_pool_path=private_candidates,
        expected_target_identity_set_sha256=target_sha,
        output_path=output,
    )
    assert set(value) == {
        "format",
        "status",
        "target_role",
        "heldout_identity_set_sha256",
        "target_identity_set_sha256",
        "intersection_count",
        "sensitive_identities_included",
        "attestation_sha256",
    }
    assert validate_disjoint_attestation(
        value,
        heldout_identity_set_sha256=commitment["identity_set_sha256"],
        target_identity_set_sha256=target_sha,
        target_role="preregistered_reset_candidate_pool",
    ) == value["attestation_sha256"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert "identities" not in json.loads(output.read_text(encoding="utf-8"))


def test_selected_requested_resolved_attestation_uses_reset_receipt_hash(
    tmp_path: Path,
) -> None:
    heldout, commitment_path, commitment = commit(
        tmp_path, [800_000_001, 800_000_002]
    )
    requested = [201, 202, 203]
    resolved = [301, 302, 303]
    private_selected = private_file(
        tmp_path / "private-selected.json", selected(requested, resolved)
    )
    target_sha = canonical_sha256(
        {"requested": requested, "resolved": resolved}
    )
    output = tmp_path / "selected-attestation.json"
    value = attestor.attest_selected_requested_resolved(
        private_heldout_path=heldout,
        heldout_commitment_path=commitment_path,
        private_selected_path=private_selected,
        expected_target_identity_set_sha256=target_sha,
        output_path=output,
    )
    assert validate_disjoint_attestation(
        value,
        heldout_identity_set_sha256=commitment["identity_set_sha256"],
        target_identity_set_sha256=target_sha,
        target_role="selected_requested_and_resolved_target_identities",
    ) == value["attestation_sha256"]
    published = json.loads(output.read_text(encoding="utf-8"))
    assert "requested" not in published
    assert "resolved" not in published


@pytest.mark.parametrize("selected_lane", [False, True])
def test_overlap_fails_closed_without_identity_value_in_error_or_output(
    tmp_path: Path, selected_lane: bool
) -> None:
    secret = 987_654_321
    heldout, commitment_path, _ = commit(tmp_path, [secret, 987_654_322])
    output = tmp_path / "must-not-exist.json"
    if selected_lane:
        private_target = private_file(
            tmp_path / "private-selected.json",
            selected([401, secret], [501, 502]),
        )
        target_sha = attestor.selected_identity_set_sha256([401, secret], [501, 502])
        call = lambda: attestor.attest_selected_requested_resolved(
            private_heldout_path=heldout,
            heldout_commitment_path=commitment_path,
            private_selected_path=private_target,
            expected_target_identity_set_sha256=target_sha,
            output_path=output,
        )
    else:
        private_target = private_file(
            tmp_path / "private-candidates.json",
            single(attestor.CANDIDATE_ROLE, [401, secret]),
        )
        target_sha = attestor.candidate_identity_set_sha256([401, secret])
        call = lambda: attestor.attest_candidate_pool(
            private_heldout_path=heldout,
            heldout_commitment_path=commitment_path,
            private_candidate_pool_path=private_target,
            expected_target_identity_set_sha256=target_sha,
            output_path=output,
        )
    with pytest.raises(attestor.PrivateIdentityAttestationError) as caught:
        call()
    assert str(secret) not in str(caught.value)
    assert not output.exists()


def test_hash_mismatch_fails_without_echoing_either_hash(tmp_path: Path) -> None:
    heldout, commitment_path, _ = commit(tmp_path, [701, 702])
    candidates = [801, 802]
    private_candidates = private_file(
        tmp_path / "private-candidates.json",
        single(attestor.CANDIDATE_ROLE, candidates),
    )
    actual = attestor.candidate_identity_set_sha256(candidates)
    expected = "f" * 64
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(attestor.PrivateIdentityAttestationError) as caught:
        attestor.attest_candidate_pool(
            private_heldout_path=heldout,
            heldout_commitment_path=commitment_path,
            private_candidate_pool_path=private_candidates,
            expected_target_identity_set_sha256=expected,
            output_path=output,
        )
    assert actual not in str(caught.value)
    assert expected not in str(caught.value)
    assert not output.exists()


@pytest.mark.parametrize("mode", [0o600, 0o500, 0o444])
def test_private_inputs_require_exact_owner_read_only_mode(
    tmp_path: Path, mode: int
) -> None:
    secret = 765_432_109
    path = private_file(
        tmp_path / f"unsafe-{mode:o}.json",
        single(attestor.HELDOUT_ROLE, [secret]),
    )
    path.chmod(mode)
    with pytest.raises(attestor.PrivateIdentityAttestationError) as caught:
        attestor.commit_heldout(path, tmp_path / "must-not-exist.json")
    assert str(secret) not in str(caught.value)
    assert str(path) not in str(caught.value)


def test_cli_stdout_contains_only_public_aggregates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    secrets = [654_321_987, 654_321_988]
    heldout = private_file(
        tmp_path / "private-heldout.json",
        single(attestor.HELDOUT_ROLE, secrets),
    )
    assert attestor.main(
        [
            "commit-heldout",
            "--private-heldout",
            str(heldout),
            "--output",
            str(tmp_path / "heldout-commitment.json"),
        ]
    ) == 0
    public = json.loads(capsys.readouterr().out)
    assert set(public) == {
        "identity_count",
        "identity_set_sha256",
        "commitment_sha256",
    }
    assert public["identity_count"] == len(secrets)
    assert all(str(secret) not in json.dumps(public) for secret in secrets)


def test_schema_and_duplicate_failures_are_sanitized(tmp_path: Path) -> None:
    secret = 543_210_987
    malformed = single(attestor.HELDOUT_ROLE, [secret, secret])
    malformed["unexpected"] = secret
    path = private_file(tmp_path / "malformed.json", malformed)
    with pytest.raises(attestor.PrivateIdentityAttestationError) as caught:
        attestor.commit_heldout(path, tmp_path / "must-not-exist.json")
    assert str(secret) not in str(caught.value)
    assert not (tmp_path / "must-not-exist.json").exists()


def test_duplicate_json_key_is_rejected_without_echoing_private_value(
    tmp_path: Path,
) -> None:
    secret = 432_109_876
    path = tmp_path / "duplicate-key.json"
    path.write_text(
        "{"
        f'"format":"{attestor.PRIVATE_SET_FORMAT}",'
        f'"status":"{attestor.PRIVATE_SET_STATUS}",'
        f'"role":"{attestor.HELDOUT_ROLE}",'
        f'"identities":[{secret}],"identities":[1]'
        "}\n",
        encoding="utf-8",
    )
    path.chmod(0o400)
    with pytest.raises(attestor.PrivateIdentityAttestationError) as caught:
        attestor.commit_heldout(path, tmp_path / "must-not-exist.json")
    assert str(secret) not in str(caught.value)
    assert not (tmp_path / "must-not-exist.json").exists()
