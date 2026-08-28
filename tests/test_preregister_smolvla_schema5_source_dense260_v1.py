from __future__ import annotations

import copy
import importlib.util
import json
import stat
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "preregister_smolvla_schema5_source_dense260_v1.py"
)
SPEC = importlib.util.spec_from_file_location("source_dense260", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


def _signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = dense.canonical_sha256(result)
    return result


def _set_sha(values: list[int | str]) -> str:
    return dense.canonical_sha256(sorted(set(values)))


def _reset_manifest(
    replacements: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal in range(dense.CANDIDATE_COUNT):
        requested = dense.CANDIDATE_START + ordinal * dense.CANDIDATE_STEP
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "requested_seed": requested,
            "status": "stable_reset_identity_observed",
            "resolved_seed": requested,
            "reset_identity_sha256": dense.canonical_sha256(
                {"reset_identity_for_requested_seed": requested}
            ),
        }
        if replacements and ordinal in replacements:
            row.update(replacements[ordinal])
        rows.append(row)
    stable = [
        row for row in rows if row["status"] == "stable_reset_identity_observed"
    ]
    axes = {
        "requested_seed": [row["requested_seed"] for row in rows],
        "resolved_seed": [row["resolved_seed"] for row in stable],
        "reset_identity": [row["reset_identity_sha256"] for row in stable],
    }
    value = {
        "format": dense.RESET_MANIFEST_FORMAT,
        "status": dense.RESET_MANIFEST_STATUS,
        "namespace": dense.NAMESPACE,
        "task": dense.TASK,
        "body": dense.BODY,
        "policy": dense.POLICY,
        "candidate_range": {
            "start": dense.CANDIDATE_START,
            "count": dense.CANDIDATE_COUNT,
            "step": dense.CANDIDATE_STEP,
        },
        "reset_identity_contract": copy.deepcopy(dense.RESET_IDENTITY_CONTRACT),
        "capability": copy.deepcopy(dense.RESET_CAPABILITY),
        "rows": rows,
        "candidate_identity_counts": {
            axis: len(set(values)) for axis, values in axes.items()
        },
        "candidate_identity_sets_sha256": {
            axis: _set_sha(values) for axis, values in axes.items()
        },
    }
    return _signed(value, "manifest_sha256")


def _attestation(
    role: str, candidate_commitments: dict[str, str]
) -> dict[str, Any]:
    value = {
        "format": dense.ATTESTATION_FORMAT,
        "status": dense.ATTESTATION_STATUS,
        "reference_role": role,
        "target_role": dense.TARGET_ROLE,
        "reference_identity_sets_sha256": {
            axis: dense.canonical_sha256({"reference": role, "axis": axis})
            for axis in dense.AXES
        },
        "target_identity_sets_sha256": candidate_commitments,
        "intersection_counts": {axis: 0 for axis in dense.AXES},
        "sensitive_identities_included": False,
        "only_aggregate_commitments_disclosed": True,
    }
    return _signed(value, "attestation_sha256")


def _inputs(
    manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], str]]]:
    reset = manifest or _reset_manifest()
    commitments = reset["candidate_identity_sets_sha256"]
    attestations = [
        (
            _attestation(role, commitments),
            dense.canonical_sha256({"attestation_file_for": role}),
        )
        for role in sorted(dense.REQUIRED_REFERENCE_ROLES)
    ]
    return reset, attestations


def _build(
    manifest: dict[str, Any] | None = None,
    attestations: list[tuple[dict[str, Any], str]] | None = None,
) -> dict[str, Any]:
    reset, defaults = _inputs(manifest)
    return dense.build_preregistration(
        reset_manifest=reset,
        reset_manifest_file_sha256=dense.canonical_sha256(
            {"reset_manifest_file": reset["manifest_sha256"]}
        ),
        attestations=defaults if attestations is None else attestations,
    )


def test_freezes_exact_contract_and_sha_ordered_partition() -> None:
    result = _build()
    summary = dense.validate_preregistration(result)

    assert summary == {
        "status": "verified_source_dense260_label_free_preregistration",
        "preregistration_sha256": result["preregistration_sha256"],
        "groups": 260,
        "split_counts": {"train": 100, "calibration": 80, "validation": 80},
        "candidate_count": 8,
        "action_exec_steps": 5,
        "max_steps": 200,
        "external_reference_roles": sorted(dense.REQUIRED_REFERENCE_ROLES),
        "hdf5_files_opened": 0,
        "labels_or_outcomes_read": False,
        "collection_authorized": False,
    }
    assert result["collection_contract"]["candidate_count"] == 8
    assert result["collection_contract"]["action_exec_steps"] == 5
    assert result["collection_contract"]["max_steps"] == 200
    assert result["capability"]["environment_reset_by_this_freezer"] is False
    assert result["capability"]["collection_authorized"] is False
    assert result["preregistration_sha256"] == dense.canonical_sha256(
        {key: value for key, value in result.items() if key != "preregistration_sha256"}
    )


def test_first_resolved_unique_rule_skips_unavailable_and_duplicate_seed() -> None:
    manifest = _reset_manifest(
        {
            1: {"resolved_seed": dense.CANDIDATE_START},
            2: {
                "status": "reset_identity_unavailable_failed_closed",
                "resolved_seed": None,
                "reset_identity_sha256": None,
            },
        }
    )
    result = _build(manifest)
    candidate_ordinals = {row["candidate_ordinal"] for row in result["groups"]}

    assert {1, 2}.isdisjoint(candidate_ordinals)
    assert max(candidate_ordinals) == 261
    assert result["selection_audit"]["skipped_before_or_after_selection"] == {
        "reset_identity_unavailable": 1,
        "duplicate_resolved_seed": 1,
        "after_first_260_unique": 138,
    }


def test_duplicate_reset_identity_in_first_260_fails_without_substitution() -> None:
    first_reset = dense.canonical_sha256(
        {"reset_identity_for_requested_seed": dense.CANDIDATE_START}
    )
    manifest = _reset_manifest({1: {"reset_identity_sha256": first_reset}})

    with pytest.raises(
        dense.SourceDense260PreregistrationError,
        match="first 260 resolved-unique candidates contain duplicate",
    ):
        _build(manifest)


def test_every_axis_is_unique_and_pairwise_disjoint_between_splits() -> None:
    result = _build()
    for axis, field in {
        "requested_seed": "requested_seed",
        "resolved_seed": "resolved_seed",
        "reset_identity": "reset_identity_sha256",
    }.items():
        split_sets = {
            split: {
                row[field] for row in result["groups"] if row["split"] == split
            }
            for split in dense.SPLIT_COUNTS
        }
        assert sum(map(len, split_sets.values())) == dense.SELECTED_GROUPS
        names = list(split_sets)
        assert all(
            split_sets[names[left]].isdisjoint(split_sets[names[right]])
            for left in range(len(names))
            for right in range(left + 1, len(names))
        ), axis


@pytest.mark.parametrize(
    "change", ["extra_field", "nonzero_capability", "identity_recipe"]
)
def test_reset_contract_tampering_fails_closed(change: str) -> None:
    manifest = _reset_manifest()
    if change == "extra_field":
        manifest["rows"][0]["supervision"] = "forbidden"
    elif change == "nonzero_capability":
        manifest["capability"]["environment_step_calls"] = 1
    else:
        manifest["reset_identity_contract"]["payload_fields"].append(
            "requested_seed"
        )
    manifest = _signed(manifest, "manifest_sha256")

    with pytest.raises(dense.SourceDense260PreregistrationError):
        _build(manifest)


def test_manifest_signature_tampering_fails_closed() -> None:
    manifest = _reset_manifest()
    manifest["rows"][0]["resolved_seed"] += 1

    with pytest.raises(
        dense.SourceDense260PreregistrationError, match="signature mismatch"
    ):
        _build(manifest)


def test_nonzero_attestation_or_missing_role_fails_closed() -> None:
    reset, attestations = _inputs()
    bad = copy.deepcopy(attestations)
    bad[0][0]["intersection_counts"]["resolved_seed"] = 1
    bad[0] = (_signed(bad[0][0], "attestation_sha256"), bad[0][1])
    with pytest.raises(dense.SourceDense260PreregistrationError):
        _build(reset, bad)
    with pytest.raises(dense.SourceDense260PreregistrationError):
        _build(reset, attestations[:-1])


def test_attestation_cannot_disclose_identities_or_rebind_candidate_pool() -> None:
    reset, attestations = _inputs()
    disclosed = copy.deepcopy(attestations)
    disclosed[0][0]["sensitive_identities_included"] = True
    disclosed[0] = (
        _signed(disclosed[0][0], "attestation_sha256"),
        disclosed[0][1],
    )
    with pytest.raises(dense.SourceDense260PreregistrationError):
        _build(reset, disclosed)

    rebound = copy.deepcopy(attestations)
    rebound[0][0]["target_identity_sets_sha256"]["requested_seed"] = "0" * 64
    rebound[0] = (_signed(rebound[0][0], "attestation_sha256"), rebound[0][1])
    with pytest.raises(dense.SourceDense260PreregistrationError):
        _build(reset, rebound)


def test_fewer_than_260_unique_resolved_seeds_fails_closed() -> None:
    replacements = {
        ordinal: {
            "resolved_seed": dense.CANDIDATE_START + ordinal % 259
        }
        for ordinal in range(dense.CANDIDATE_COUNT)
    }
    manifest = _reset_manifest(replacements)

    with pytest.raises(
        dense.SourceDense260PreregistrationError, match="fewer than 260"
    ):
        _build(manifest)


def test_output_is_create_once_read_only_and_roundtrips(tmp_path: Path) -> None:
    result = _build()
    output = tmp_path / "dense260.json"
    dense.write_json_new(output, result)

    assert stat.S_IMODE(output.stat().st_mode) & stat.S_IWUSR == 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        dense.write_json_new(output, result)


@pytest.mark.parametrize("marker", ["protected", "fresh", "confirmation"])
def test_sensitive_path_is_rejected_before_file_access(
    tmp_path: Path, marker: str
) -> None:
    unsafe = tmp_path / marker / "missing.json"

    with pytest.raises(
        dense.SourceDense260PreregistrationError,
        match="forbidden protected namespace",
    ):
        dense.freeze_from_paths(
            reset_manifest_path=unsafe,
            attestation_paths=[tmp_path / f"att_{index}.json" for index in range(6)],
            output_path=tmp_path / "dense260.json",
        )


def test_signed_output_tampering_still_fails_structural_validation() -> None:
    result = _build()
    result["collection_contract"]["action_exec_steps"] = 50
    result = _signed(result, "preregistration_sha256")

    with pytest.raises(
        dense.SourceDense260PreregistrationError,
        match="immutable dense collection contract",
    ):
        dense.validate_preregistration(result)
