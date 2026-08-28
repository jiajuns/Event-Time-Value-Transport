from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_openvla_etsf_action_signal_oof import (  # noqa: E402
    PRIMARY_VARIANT,
    DiagnosticGroup,
    _embedded_oof_folds,
    load_schema5_development,
    run_oof,
    sha256,
)
from openvla_etsf_counterfactual_oof import make_oof_folds  # noqa: E402


def synthetic_groups() -> list[DiagnosticGroup]:
    groups = []
    for index in range(100):
        rng = np.random.default_rng(1000 + index)
        actions = np.zeros((4, 5, 3), dtype=np.float32)
        actions[0] = -0.5
        actions[1] = 1.0
        actions[2] = -1.0
        actions[3] = 0.5
        actions += rng.normal(0.0, 0.01, actions.shape).astype(np.float32)
        success = np.asarray([0, 1, 0, 1], dtype=np.float32)
        distance = np.sqrt(
            np.mean(np.square(actions - actions[0:1]), axis=(1, 2))
        ).astype(np.float32)
        groups.append(
            DiagnosticGroup(
                logical_key=f"move_can_pot|piper|{2000 + index}",
                seed=2000 + index,
                path=f"group_{index}.hdf5",
                hidden=rng.normal(size=12).astype(np.float32),
                actions=actions,
                success=success,
                candidate_distance=distance,
                candidate_names=(
                    "deterministic",
                    "candidate_positive",
                    "candidate_negative",
                    "candidate_positive_small",
                ),
                baseline_index=0,
            )
        )
    return groups


def test_strict_oof_detects_learnable_action_signal() -> None:
    result = run_oof(synthetic_groups())
    assert result["fold_protocol"]["every_group_predicted_once"]
    assert not result["fold_protocol"]["train_holdout_group_leakage"]
    assert [row["training_groups"] for row in result["fold_audits"]] == [80] * 5
    assert [row["oof_holdout_groups"] for row in result["fold_audits"]] == [20] * 5
    primary = result["metrics"][PRIMARY_VARIANT]
    assert primary["actor_baseline_success_rate"] == 0.0
    assert primary["oracle_success_rate"] == 1.0
    assert primary["selected_success_rate"] > 0.95
    assert primary["helpful_changes"] > 95
    assert primary["harmful_changes"] == 0
    assert primary["within_group_success_pair_accuracy"] > 0.95


def test_embedded_fold_fallback_matches_shared_oof_assignment() -> None:
    keys = [group.logical_key for group in synthetic_groups()]
    embedded = _embedded_oof_folds(keys)
    shared = make_oof_folds(keys)
    for embedded_fold, shared_fold in zip(embedded["folds"], shared["folds"]):
        assert embedded_fold["training_groups"] == shared_fold["training_groups"]
        assert embedded_fold["oof_holdout_groups"] == shared_fold["oof_holdout_groups"]


def _write_schema5_root(root: Path) -> list[Path]:
    group_root = root / "groups"
    group_root.mkdir(parents=True)
    strings = h5py.string_dtype("utf-8")
    rows = []
    paths = []
    for index in range(100):
        path = group_root / f"group_{index:03d}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = 5
            handle.attrs["task"] = "move_can_pot"
            handle.attrs["body"] = "piper"
            handle.attrs["resolved_seed"] = 3000 + index
            handle.attrs["branch_instruction_consistent"] = True
            handle.create_dataset("initial_hidden", data=np.full(8, index, np.float32))
            actions = np.asarray(
                [
                    np.full((3, 2), -0.5),
                    np.full((3, 2), 0.5),
                ],
                dtype=np.float32,
            )
            handle.create_dataset("candidate_actions", data=actions)
            handle.create_dataset("success", data=np.asarray([0, 1], np.float32))
            handle.create_dataset(
                "candidate_names",
                data=np.asarray(["deterministic", "candidate"], dtype=object),
                dtype=strings,
            )
            handle.create_dataset(
                "normalized_l2_from_baseline", data=np.asarray([0.0, 1.0])
            )
        rows.append({"path": path.name, "sha256": sha256(path)})
        paths.append(path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 5,
                "language_contract": (
                    "same_instruction_for_initial_query_and_all_candidate_branches"
                ),
                "groups": rows,
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_schema5_loader_is_read_only_and_records_file_sha(tmp_path: Path) -> None:
    root = tmp_path / "train100"
    paths = _write_schema5_root(root)
    before = {path: sha256(path) for path in paths}
    groups, audit = load_schema5_development(root)
    after = {path: sha256(path) for path in paths}
    assert len(groups) == 100
    assert audit["groups"] == 100
    assert not audit["fresh_confirmation_labels_read"]
    assert before == after


def test_fresh_confirmation_registry_is_rejected_before_group_access(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fresh50"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 5,
                "seed_registry": "explicit_fresh_confirmation",
                "fresh_seed_manifest_sha256": "a" * 64,
                "language_contract": (
                    "same_instruction_for_initial_query_and_all_candidate_branches"
                ),
                "groups": [{}] * 100,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fresh confirmation"):
        load_schema5_development(root)


def test_group_path_may_not_escape_schema5_root(tmp_path: Path) -> None:
    root = tmp_path / "train100"
    root.mkdir()
    outside = tmp_path / "outside.hdf5"
    outside.touch()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 5,
                "language_contract": (
                    "same_instruction_for_initial_query_and_all_candidate_branches"
                ),
                "groups": [{"path": str(outside)}] * 100,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="escapes the schema-v5 root"):
        load_schema5_development(root)
