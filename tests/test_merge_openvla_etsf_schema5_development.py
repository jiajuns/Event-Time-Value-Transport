from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import merge_openvla_etsf_schema5_development as merge  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common() -> dict:
    return {
        "schema_version": 5,
        "status": "complete",
        "task": "move_can_pot",
        "body": "piper",
        "model_path": "/models/openvla",
        "unnorm_key": "robotwin",
        "temperature": 0.7,
        "top_k": 4,
        "preserve_grippers": True,
        "intervention": "first_action_chunk_only_then_closed_loop_requery",
        "language_contract": merge.LANGUAGE_CONTRACT,
        "event_vocab": ["e0", "e1", "e2", "e3", "eK"],
        "event_spec_sha256": "e" * 64,
        "hidden_dim": 8,
        "hidden_anchor": "last_prompt_token",
        "action_dim": 7,
        "action_chunk": 5,
        "max_steps": 200,
        "trajectory_contract": "full_candidate_branch",
        "continuation_query_contract": "schema5_every_query",
    }


def _collection(
    root: Path,
    *,
    count: int,
    seed_start: int,
    candidate_count: int,
    blends: list[float],
    registry: str | None,
) -> dict:
    group_root = root / "groups"
    group_root.mkdir(parents=True)
    rows = []
    seeds = []
    for index in range(count):
        seed = seed_start + index
        seeds.append(seed)
        path = group_root / f"group_{index:03d}_seed_{seed}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = 5
            handle.attrs["requested_seed"] = seed
            handle.attrs["resolved_seed"] = seed
            handle.attrs["task"] = "move_can_pot"
            handle.attrs["body"] = "piper"
            handle.attrs["candidate_count"] = candidate_count
            handle.attrs["language_contract"] = merge.LANGUAGE_CONTRACT
            handle.attrs["branch_instruction_consistent"] = True
        rows.append(
            {
                "path": path.name,
                "requested_seed": seed,
                "resolved_seed": seed,
                "sha256": _sha(path),
            }
        )
    manifest = {
        **_common(),
        "seed_registry": registry,
        "candidate_count": candidate_count,
        "blends": blends,
        "completed": count,
        "requested_seeds": seeds,
        "resolved_seeds": seeds,
        "groups": rows,
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _fake_expansion(tmp_path: Path, manifest: dict) -> dict:
    official = tmp_path / "official.json"
    official.write_text(
        json.dumps({"move_can_pot": {"success_seeds": list(range(150))}}),
        encoding="utf-8",
    )
    return {
        "path": str(tmp_path / "development.json"),
        "sha256": "d" * 64,
        "requested_seeds": manifest["requested_seeds"],
        "resolved_seeds": manifest["resolved_seeds"],
        "official_seed_registry": {"path": str(official), "sha256": _sha(official)},
        "fresh_seed_manifest": {"path": str(tmp_path / "fresh.json"), "sha256": "f" * 64},
        "label_access_contract": "reset_identity_only",
    }


def test_merge_accepts_legacy_registry_and_mixed_four_five_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = tmp_path / "old100"
    new = tmp_path / "new150"
    output = tmp_path / "combined250"
    _collection(
        old,
        count=100,
        seed_start=0,
        candidate_count=4,
        blends=[0.25, 0.5, 0.75],
        registry=None,
    )
    new_manifest = _collection(
        new,
        count=150,
        seed_start=1000,
        candidate_count=5,
        blends=[0.25, 0.5, 0.75, 1.0],
        registry=merge.DEVELOPMENT_REGISTRY,
    )
    expansion = _fake_expansion(tmp_path, new_manifest)
    monkeypatch.setattr(
        merge,
        "_development_expansion_contract",
        lambda root, manifest: expansion,
    )

    result = merge.merge_development_roots(old, new, output)

    assert result["completed"] == 250
    assert result["candidate_contract"]["candidate_count_histogram"] == {
        "4": 100,
        "5": 150,
    }
    assert result["source_collections"][0]["seed_registry_audit"].startswith(
        "legacy_missing_registry"
    )
    assert result["source_collections"][0]["candidate_contract"]["candidate_count"] == 4
    assert result["source_collections"][1]["candidate_contract"]["candidate_count"] == 5
    for index in (0, 99, 100, 249):
        row = result["groups"][index]
        destination = output / "groups" / row["path"]
        assert destination.stat().st_ino == Path(row["source_path"]).stat().st_ino
        assert _sha(destination) == row["sha256"]


def test_merge_rejects_changed_candidate_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = tmp_path / "old100"
    new = tmp_path / "new150"
    _collection(
        old,
        count=100,
        seed_start=0,
        candidate_count=4,
        blends=[0.25, 0.5, 0.75],
        registry=None,
    )
    new_manifest = _collection(
        new,
        count=150,
        seed_start=1000,
        candidate_count=5,
        blends=[0.25, 0.5, 0.75],
        registry=merge.DEVELOPMENT_REGISTRY,
    )
    monkeypatch.setattr(
        merge,
        "_development_expansion_contract",
        lambda root, manifest: _fake_expansion(tmp_path, new_manifest),
    )
    with pytest.raises(RuntimeError, match="candidate intervention contract"):
        merge.merge_development_roots(old, new, tmp_path / "combined250")
