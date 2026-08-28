from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from collect_openvla_etsf_event_branches import (  # noqa: E402
    collection_identity_payload,
    explicit_seed_registry,
)
from preregister_robotwin_development_expansion_seeds import (  # noqa: E402
    build_manifest,
    select_reset_unique_scenes,
)
from robotwin_development_seed_contract import (  # noqa: E402
    CANDIDATE_FORMAT,
    CANDIDATE_STATUS,
    EXPECTED_COUNT,
    MANIFEST_STATUS,
    PURPOSE,
    REGISTRY,
    expand_candidate_requested_seeds,
    validate_development_manifest,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _source_files(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    candidate = {
        "format": CANDIDATE_FORMAT,
        "schema_version": 1,
        "status": CANDIDATE_STATUS,
        "task": "move_can_pot",
        "purpose": PURPOSE,
        "candidate_seed_range": {"start": 100100276, "count": 200, "step": 1},
        "selection_rule": "first_150_reset_unique_in_frozen_range_order",
        "freeze_rule": "freeze_before_reset_and_policy_access",
    }
    candidate_path = tmp_path / "candidates.json"
    official_path = tmp_path / "official.json"
    fresh_path = tmp_path / "fresh.json"
    _json(candidate_path, candidate)
    _json(
        official_path,
        {"move_can_pot": {"success_seeds": list(range(150))}},
    )
    _json(
        fresh_path,
        {
            "status": "fresh_confirmation_preregistered_resolved",
            "task": "move_can_pot",
            "requested_seeds": list(range(100100196, 100100246)),
            "resolved_seeds": list(range(200100196, 200100246)),
        },
    )
    return candidate_path, official_path, fresh_path, candidate


def _development_manifest(tmp_path: Path) -> tuple[Path, dict]:
    candidate_path, official_path, fresh_path, candidate = _source_files(tmp_path)
    candidates = expand_candidate_requested_seeds(candidate)

    def resolver(seed: int) -> int:
        # One deliberate reset collision proves selection is resolved-identity based.
        return candidates[0] if seed == candidates[1] else seed

    selected, audit = select_reset_unique_scenes(
        candidates,
        resolver=resolver,
        excluded=set(range(150))
        | set(range(100100196, 100100246))
        | set(range(200100196, 200100246)),
    )
    assert len(selected) == EXPECTED_COUNT
    assert len(audit) == EXPECTED_COUNT + 1
    manifest = build_manifest(
        task="move_can_pot",
        selected=selected,
        audit=audit,
        candidate_path=candidate_path,
        official_path=official_path,
        fresh_path=fresh_path,
        candidate=candidate,
    )
    path = tmp_path / "development.json"
    _json(path, manifest)
    return path, manifest


def test_compact_range_reset_unique_manifest_and_registry_contract(tmp_path: Path) -> None:
    manifest_path, _ = _development_manifest(tmp_path)
    audited = validate_development_manifest(manifest_path, task="move_can_pot")
    assert len(audited["rows"]) == 150
    assert audited["seed_registry"] == REGISTRY
    assert audited["requested_seeds"][0] == 100100276
    assert audited["requested_seeds"][1] == 100100278
    assert explicit_seed_registry(
        allow_unregistered_seeds=True,
        fresh_seed_manifest=None,
        development_seed_manifest=manifest_path,
    ) == REGISTRY
    with pytest.raises(ValueError, match="mutually exclusive"):
        explicit_seed_registry(
            allow_unregistered_seeds=True,
            fresh_seed_manifest=tmp_path / "fresh.json",
            development_seed_manifest=manifest_path,
        )
    identity = collection_identity_payload(
        {
            "schema_version": 5,
            "seed_registry": REGISTRY,
            "fresh_seed_manifest": None,
            "fresh_seed_manifest_sha256": None,
            "development_seed_manifest": str(manifest_path),
            "development_seed_manifest_sha256": audited["sha256"],
            "groups": [],
        }
    )
    assert identity["seed_registry"] == REGISTRY
    assert identity["fresh_seed_manifest"] is None
    assert identity["development_seed_manifest_sha256"] == audited["sha256"]


def test_overlap_and_source_tampering_fail_closed(tmp_path: Path) -> None:
    manifest_path, manifest = _development_manifest(tmp_path)
    fresh_path = Path(manifest["fresh_seed_manifest"])
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    fresh["requested_seeds"][0] = manifest["requested_seeds"][0]
    _json(fresh_path, fresh)
    with pytest.raises(RuntimeError, match="source artifact SHA256"):
        validate_development_manifest(manifest_path, task="move_can_pot")

    # Rebuild and sign against the changed exclusion source: overlap itself is rejected.
    candidate_path = Path(manifest["candidate_manifest"])
    official_path = Path(manifest["official_seed_registry"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    overlap_manifest = build_manifest(
        task="move_can_pot",
        selected=manifest["train"],
        audit=manifest["audit"],
        candidate_path=candidate_path,
        official_path=official_path,
        fresh_path=fresh_path,
        candidate=candidate,
    )
    overlap_path = tmp_path / "overlap.json"
    _json(overlap_path, overlap_manifest)
    with pytest.raises(RuntimeError, match="overlap official/fresh"):
        validate_development_manifest(overlap_path, task="move_can_pot")


def test_launcher_dry_run_uses_development_not_fresh_and_five_candidates(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _development_manifest(tmp_path)
    prerequisites = {}
    for name in ("model", "rlinf", "robotwin", "robotwin_code"):
        path = tmp_path / name
        path.mkdir()
        prerequisites[name] = path
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {})
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    output = tmp_path / "development_collection"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "launch_openvla_etsf_development_expansion_v5.py"),
            "--development-seed-manifest", str(manifest_path),
            "--model-path", str(prerequisites["model"]),
            "--rlinf-root", str(prerequisites["rlinf"]),
            "--robotwin-root", str(prerequisites["robotwin"]),
            "--robotwin-code", str(prerequisites["robotwin_code"]),
            "--event-spec", str(event_spec),
            "--python-bin", str(venv_python),
            "--output", str(output),
            "--wait-timeout-seconds", "0",
            "--poll-seconds", "0.01",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(
        completed.stdout.removeprefix("DEVELOPMENT_EXPANSION_DRY_RUN=")
    )
    command = audit["command"]
    assert command[0] == str(venv_python.absolute())
    assert command[0] != str(venv_python.resolve())
    assert "--development-seed-manifest" in command
    assert "--fresh-seed-manifest" not in command
    assert audit["contract"]["seed_registry"] == REGISTRY
    blends = command[command.index("--blends") + 1 : command.index("--temperature")]
    assert blends == ["0.25", "0.5", "0.75", "1.0"]
    assert audit["contract"]["fresh_confirmation_eligible"] is False
    assert not output.exists()
