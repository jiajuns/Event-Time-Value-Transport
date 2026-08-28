from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_paired_success_development as watcher  # noqa: E402
from smolvla_piper_paired_success_protocol import (  # noqa: E402
    PAIR_RESULT_FORMAT,
    canonical_sha256,
    file_sha256,
    synthetic_protocol,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _signed(value: dict, field: str) -> dict:
    value[field] = canonical_sha256(value)
    return value


def test_forbidden_namespace_is_rejected_without_opening(tmp_path: Path) -> None:
    path = tmp_path / "fresh_lane" / "artifact.json"
    with pytest.raises(watcher.WatcherError, match="forbidden"):
        watcher.existing_file(path, "artifact")


def test_rtx4090_identity_and_two_consecutive_idle_audits() -> None:
    outputs = iter(
        [
            "NVIDIA GeForce RTX 4090, GPU-a\n", "122\n",
            "NVIDIA GeForce RTX 4090, GPU-a\n", "\n",
            "NVIDIA GeForce RTX 4090, GPU-a\n", "\n",
        ]
    )
    sleeps: list[float] = []
    audits = watcher.wait_two_idle(
        0,
        interval=0.01,
        timeout=10,
        run_text=lambda _command: next(outputs),
        sleep=sleeps.append,
    )
    assert [row["compute_pids"] for row in audits] == [[], []]
    assert len(sleeps) == 2
    bad = iter(["NVIDIA A100, GPU-b\n", "\n"])
    with pytest.raises(watcher.WatcherError, match="not an RTX 4090"):
        watcher.gpu_audit(0, lambda _command: next(bad))


def _collection(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "collection"
    root.mkdir()
    protocol = synthetic_protocol(pair_count=2)
    for pair in protocol["development_pairs"]:
        pair_id = pair["pair_id"]
        selection = _signed(
            {
                "pair_id": pair_id,
                "protocol_sha256": protocol["protocol_sha256"],
                "environment_steps_before_selection": 0,
                "candidate_outcomes_visible_to_selector": False,
                "success_reward_event_or_trajectory_visible_to_selector": False,
            },
            "selection_record_sha256",
        )
        result = _signed(
            {
                "format": PAIR_RESULT_FORMAT,
                "pair_id": pair_id,
                "protocol_sha256": protocol["protocol_sha256"],
                "predicted_success_used_as_outcome": False,
            },
            "pair_result_sha256",
        )
        _write(root / f"{pair_id}.selection.json", selection)
        _write(root / f"{pair_id}.result.json", result)
    rows = [
        {
            "pair_id": pair["pair_id"],
            "pair_result_path": str(root / f"{pair['pair_id']}.result.json"),
            "pair_result_file_sha256": file_sha256(root / f"{pair['pair_id']}.result.json"),
            "selection_path": str(root / f"{pair['pair_id']}.selection.json"),
            "selection_file_sha256": file_sha256(root / f"{pair['pair_id']}.selection.json"),
        }
        for pair in protocol["development_pairs"]
    ]
    manifest = _signed(
        {
            "format": watcher.MANIFEST_FORMAT,
            "status": watcher.MANIFEST_STATUS,
            "protocol_sha256": protocol["protocol_sha256"],
            "lane": "development",
            "task": watcher.TASK,
            "body": watcher.BODY,
            "actor_id": watcher.ACTOR_ID,
            "pair_count": 2,
            "pairs": rows,
            "task_success_source": "simulator_info_success_from_executed_schema6_branch",
            "predicted_success_used_as_outcome": False,
            "sealed_evaluation_reserve_executed": False,
            "reserve_identities_read": False,
            "reserve_outcomes_read": False,
            "existing_sensitive_artifacts_read": False,
            "test_hdf5_opened": 0,
        },
        "manifest_sha256",
    )
    _write(root / "development_collection_manifest.json", manifest)
    return root, protocol


def test_collection_binds_preoutcome_selection_and_executed_result(tmp_path: Path) -> None:
    root, protocol = _collection(tmp_path)
    manifest, results = watcher.validate_collection_manifest(root, protocol)
    assert manifest["predicted_success_used_as_outcome"] is False
    assert len(results) == 2
    selection = Path(manifest["pairs"][0]["selection_path"])
    changed = json.loads(selection.read_text())
    changed["environment_steps_before_selection"] = 1
    changed["selection_record_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "selection_record_sha256"}
    )
    _write(selection, changed)
    manifest["pairs"][0]["selection_file_sha256"] = file_sha256(selection)
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write(root / "development_collection_manifest.json", manifest)
    with pytest.raises(watcher.WatcherError, match="not frozen"):
        watcher.validate_collection_manifest(root, protocol)


def test_preregister_claims_new_output_and_freezes_fixed_executor_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = tmp_path / "executor.py"
    executor.write_text("# synthetic executor\n", encoding="utf-8")
    protocol = synthetic_protocol(pair_count=2)
    protocol["scope"].update(
        {"task": watcher.TASK, "body": watcher.BODY, "actor_id": watcher.ACTOR_ID}
    )
    audit = {
        "path": str(tmp_path / "protocol.json"),
        "file_sha256": "a" * 64,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol": protocol,
    }
    monkeypatch.setattr(watcher, "revalidate_protocol", lambda *_args: audit)
    args = argparse.Namespace(
        output_root=str(tmp_path / "new_output"),
        protocol=audit["path"],
        protocol_file_sha256=audit["file_sha256"],
        executor=str(executor),
        executor_sha256=file_sha256(executor),
        python=str(Path(sys.executable).resolve()),
        gpu_index=0,
        lock_path=str(tmp_path / "gpu.lock"),
    )
    root, plan = watcher.preregister(args)
    assert (root / "static_plan.json").is_file()
    assert (root / "collection").is_dir()
    assert plan["command"][2:4] == ["--mode", "execute-development"]
    with pytest.raises(FileExistsError):
        watcher.preregister(args)


def test_read_only_freeze_is_recursive(tmp_path: Path) -> None:
    root = tmp_path / "output"
    child = root / "stage"
    child.mkdir(parents=True)
    artifact = child / "run.exit"
    artifact.write_text("0\n", encoding="ascii")
    watcher.freeze_tree_read_only(root)
    assert artifact.stat().st_mode & 0o222 == 0
    assert root.stat().st_mode & 0o222 == 0
