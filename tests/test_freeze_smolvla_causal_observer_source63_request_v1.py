from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import freeze_smolvla_causal_observer_source63_request_v1 as freezer  # noqa: E402


STATE_SOURCE_SHA = "a" * 64
TASK = "move_can_pot"
BODY = "aloha-agilex"
POLICY = "smolvla"


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return freezer.file_sha256(path)


def _make_fixture(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    groups_root = root / "groups"
    groups_root.mkdir()
    train = list(range(1000, 1044))
    validation = list(range(2000, 2014))
    test = list(range(3000, 3005))
    ordered = [*train, *validation, *test]
    split = {
        "format": freezer.SPLIT_FORMAT,
        "status": freezer.SPLIT_STATUS,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "split_unit": "requested_seed_logical_group",
        "train": train,
        "validation": validation,
        "test": test,
        "test_policy": "development_holdout_not_confirmatory_not_opened_by_trainer",
        "fresh_inputs_allowed": False,
        "fresh_trajectory_or_label_opened": False,
    }
    event_spec = {
        "format": "synthetic_event_spec_v1",
        "calibration": {TASK: {"synthetic": True}},
    }
    event_path = root / "event_spec.json"
    event_sha = _write_json(event_path, event_spec)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(ordered):
        name = f"group_{index:03d}_seed_{seed}.hdf5"
        # Test fixtures intentionally do not exist.  Successful freezing is
        # therefore proof that those five paths were not even statted.
        if seed not in test:
            (groups_root / name).write_bytes(f"opaque-hdf-container-{seed}".encode())
        rows.append(
            {
                "index": index,
                "seed": seed,
                "resolved_seed": seed + 10000,
                "path": name,
                "status": "collected",
                # Outcome metadata exists but the freezer never projects or
                # branches on it.
                "success": bool(index % 2),
                "steps": 7 + index,
            }
        )
    manifest = {
        "status": "complete",
        "schema_version": 5,
        "task": TASK,
        "body": BODY,
        "policy": POLICY,
        "requested_seeds": ordered,
        "resolved_seeds": [seed + 10000 for seed in ordered],
        "completed": 63,
        "candidate_count": 4,
        "hidden_dim": 960,
        "action_dim": 14,
        "action_chunk": 50,
        "event_vocab": list(freezer.EXPECTED_EVENTS),
        "event_spec_sha256": event_sha,
        "shared_state_contract": {"calibration_id": STATE_SOURCE_SHA},
        "groups": rows,
    }
    split_path = root / "split.json"
    manifest_path = root / "manifest.json"
    return {
        "root": root,
        "groups_root": groups_root,
        "split": split,
        "split_path": split_path,
        "split_sha": _write_json(split_path, split),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha": _write_json(manifest_path, manifest),
        "event_path": event_path,
        "event_sha": event_sha,
        "train": train,
        "validation": validation,
        "test": test,
    }


def _freeze(fixture: dict[str, Any], output: Path, **overrides: Any) -> dict[str, Any]:
    arguments = {
        "schema5_manifest": fixture["manifest_path"],
        "schema5_manifest_sha256": fixture["manifest_sha"],
        "frozen_split": fixture["split_path"],
        "frozen_split_sha256": fixture["split_sha"],
        "event_spec": fixture["event_path"],
        "event_spec_sha256": fixture["event_sha"],
        "output": output,
    }
    arguments.update(overrides)
    return freezer.freeze_source63_request(**arguments)


def test_exact_request_tail_calibration_and_test_zero_contact(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    output = tmp_path / "observer_request.json"
    summary = _freeze(fixture, output)

    request = json.loads(output.read_text(encoding="utf-8"))
    audit_path = freezer.audit_output_path(output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert set(request) == freezer.REQUEST_FIELDS
    assert request["format"] == freezer.REQUEST_FORMAT
    assert request["status"] == freezer.REQUEST_STATUS
    logical = dict(request)
    request_sha = logical.pop("request_sha256")
    assert request_sha == freezer.canonical_sha256(logical)
    assert request["actors"][0]["state_feature_source_sha256"] == STATE_SOURCE_SHA
    assert request["sources"][0]["manifest_logical_sha256"] == (
        freezer.canonical_sha256(fixture["manifest"])
    )

    by_split = {
        name: [int(row["logical_group_id"].rsplit("/", 1)[1]) for row in rows]
        for name, rows in request["splits"].items()
    }
    assert by_split == {
        "train": fixture["train"][:-10],
        "calibration": fixture["train"][-10:],
        "validation": fixture["validation"],
    }
    assert not set(fixture["test"]) & set(
        by_split["train"] + by_split["calibration"] + by_split["validation"]
    )
    assert audit["split_freeze"]["algorithm"] == freezer.SPLIT_ALGORITHM
    assert audit["split_freeze"]["excluded_original_test_requested_seed_ids"] == (
        fixture["test"]
    )
    access = audit["data_access_audit"]
    assert access["selected_development_hdf_file_sha256_computed"] == 58
    assert access["hdf5_library_imported"] is False
    assert access["hdf5_container_opened_count"] == 0
    assert access["original_test_group_paths_resolved"] is False
    assert access["original_test_group_files_statted"] is False
    assert access["original_test_group_files_opened"] == 0
    assert access["original_test_group_files_hashed"] == 0
    assert audit["audit_sha256"] == freezer.canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )
    assert summary["request_file_sha256"] == freezer.file_sha256(output)
    assert summary["split_counts"] == {
        "train": 34,
        "calibration": 10,
        "validation": 14,
        "excluded_test": 5,
    }


def test_request_is_byte_reproducible_and_explicit_count_is_fixed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    first = tmp_path / "first_request.json"
    second = tmp_path / "second_request.json"
    _freeze(fixture, first, calibration_count=7)
    _freeze(fixture, second, calibration_count=7)
    assert first.read_bytes() == second.read_bytes()
    request = json.loads(first.read_text(encoding="utf-8"))
    assert len(request["splits"]["train"]) == 37
    assert len(request["splits"]["calibration"]) == 7
    assert [
        int(row["logical_group_id"].rsplit("/", 1)[1])
        for row in request["splits"]["calibration"]
    ] == fixture["train"][-7:]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("manifest", "0" * 64),
        ("split", "1" * 64),
        ("event", "2" * 64),
    ),
)
def test_expected_hash_mismatch_fails_before_group_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: str,
) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    calls: list[Path] = []
    monkeypatch.setattr(
        freezer,
        "hash_opaque_selected_group",
        lambda path: calls.append(path) or "f" * 64,
    )
    overrides = {
        "schema5_manifest_sha256": fixture["manifest_sha"],
        "frozen_split_sha256": fixture["split_sha"],
        "event_spec_sha256": fixture["event_sha"],
    }
    overrides[
        {
            "manifest": "schema5_manifest_sha256",
            "split": "frozen_split_sha256",
            "event": "event_spec_sha256",
        }[field]
    ] = bad_value
    with pytest.raises(freezer.Source63ObserverRequestError):
        _freeze(fixture, tmp_path / "request.json", **overrides)
    assert calls == []


def test_seed_mismatch_and_insufficient_support_fail_before_hdf_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    calls: list[Path] = []
    monkeypatch.setattr(
        freezer,
        "hash_opaque_selected_group",
        lambda path: calls.append(path) or "f" * 64,
    )
    altered = copy.deepcopy(fixture["manifest"])
    altered["groups"][3]["seed"] += 999
    fixture["manifest_sha"] = _write_json(fixture["manifest_path"], altered)
    with pytest.raises(freezer.Source63ObserverRequestError, match="seed/identity"):
        _freeze(fixture, tmp_path / "request.json")
    assert calls == []

    fixture = _make_fixture(tmp_path / "safe_source_two")
    with pytest.raises(freezer.Source63ObserverRequestError, match="insufficient"):
        _freeze(fixture, tmp_path / "request_two.json", calibration_count=44)
    assert calls == []


def test_actor_state_source_is_manifest_calibration_id(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    altered = copy.deepcopy(fixture["manifest"])
    altered["shared_state_contract"]["calibration_id"] = "not-a-sha"
    fixture["manifest_sha"] = _write_json(fixture["manifest_path"], altered)
    with pytest.raises(freezer.Source63ObserverRequestError, match="calibration_id"):
        _freeze(fixture, tmp_path / "request.json")


def test_selected_symlink_and_protected_path_are_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    first_name = fixture["manifest"]["groups"][0]["path"]
    selected = fixture["groups_root"] / first_name
    target = fixture["groups_root"] / "opaque_target.hdf5"
    selected.rename(target)
    selected.symlink_to(target.name)
    with pytest.raises(freezer.Source63ObserverRequestError, match="symbolic link"):
        _freeze(fixture, tmp_path / "request.json")

    protected = _make_fixture(tmp_path / "safe_source_two")
    protected["manifest"]["groups"][0]["path"] = "formal_target_group.hdf5"
    protected["manifest_sha"] = _write_json(
        protected["manifest_path"], protected["manifest"]
    )
    with pytest.raises(freezer.Source63ObserverRequestError, match="protected"):
        _freeze(protected, tmp_path / "request_two.json")


def test_output_is_exclusive_and_cli_contract_is_required(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path / "safe_source")
    output = tmp_path / "request.json"
    _freeze(fixture, output)
    with pytest.raises(freezer.Source63ObserverRequestError, match="already exists"):
        _freeze(fixture, output)
    parser = freezer.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--schema5-manifest", str(fixture["manifest_path"])])
