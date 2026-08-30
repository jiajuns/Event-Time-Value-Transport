from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_robotwin2_stable_seed_roster_v1 as roster  # noqa: E402


def _attempt(seed: int, stable: bool = True) -> dict[str, object]:
    failed = False
    cells = []
    for index, (body, condition) in enumerate(roster.CELL_ORDER):
        if failed:
            cells.append(
                {
                    "cell_index": index,
                    "body": body,
                    "condition": condition,
                    "status": "not_attempted_after_first_setup_failure",
                    "reason": "label_outcome_blind_short_circuit",
                }
            )
        elif not stable and index == 3:
            failed = True
            cells.append(
                {
                    "cell_index": index,
                    "body": body,
                    "condition": condition,
                    "status": "setup_failed",
                    "error_type": "UnStableError",
                    "error_message": "unstable",
                }
            )
        else:
            cells.append(
                {
                    "cell_index": index,
                    "body": body,
                    "condition": condition,
                    "status": "setup_succeeded_and_closed",
                }
            )
    return {
        "candidate_seed": seed,
        "cells": cells,
        "all_ten_setup_cells_stable": stable,
        "actor_inference_calls": 0,
        "task_action_calls": 0,
        "label_or_outcome_reads": 0,
    }


def _valid_roster() -> dict[str, object]:
    prereg = roster.preregistration_document()
    attempts = [_attempt(roster.CANDIDATE_SEED_START, stable=False)]
    attempts.extend(
        _attempt(seed)
        for seed in range(
            roster.CANDIDATE_SEED_START + 1,
            roster.CANDIDATE_SEED_START + 1 + roster.SELECTED_SEED_COUNT,
        )
    )
    selected = [int(value["candidate_seed"]) for value in attempts[1:]]
    unsigned = {
        "format": roster.ROSTER_FORMAT,
        "status": "complete_first_twenty_common_stable_seeds",
        "task": roster.TASK,
        "preregistration": "/tmp/preregistration.json",
        "preregistration_file_sha256": "a" * 64,
        "preregistration_logical_sha256": prereg["logical_sha256"],
        "body_order": list(roster.BODIES),
        "condition_order": list(roster.CONDITIONS),
        "candidate_attempt_count": len(attempts),
        "stable_candidate_count_observed": len(selected),
        "selected_seeds": selected,
        "pair_count": 200,
        "rollout_count_for_two_methods": 400,
        "attempts": attempts,
        "actor_inference_calls": 0,
        "task_action_calls": 0,
        "label_or_outcome_reads": 0,
    }
    return {**unsigned, "logical_sha256": roster.canonical_sha256(unsigned)}


def test_preregistration_uses_fresh_2026104000_interval() -> None:
    document = roster.preregistration_document()
    roster.validate_preregistration(document)
    assert document["candidate_seed_start_inclusive"] == 2026104000
    assert document["candidate_seed_stop_exclusive"] == 2026105000
    assert document["selected_seed_count"] == 20
    assert document["disjointness_contract"] == {
        "prior_failed_actor_v1_seed_block_reused": False,
        "prior_nonformal_probe_2026103000_block_reused": False,
        "fresh_candidate_interval_starts_at": 2026104000,
    }


def test_task_args_replace_stale_yaml_robot_files(tmp_path: Path) -> None:
    config_root = tmp_path / "env_cfg" / "task_config"
    robot_root = tmp_path / "robots" / "piper"
    config_root.mkdir(parents=True)
    robot_root.mkdir(parents=True)
    (config_root / "demo_clean.yml").write_text(
        yaml.safe_dump(
            {
                "left_robot_file": "/stale/left",
                "right_robot_file": "/stale/right",
            }
        ),
        encoding="utf-8",
    )
    (config_root / "_embodiment_config.yml").write_text(
        yaml.safe_dump({"piper": {"file_path": "robots/piper"}}),
        encoding="utf-8",
    )
    (robot_root / "config.yml").write_text("robot: piper\n", encoding="utf-8")

    arguments = roster._task_args(tmp_path, "piper", "clean")

    assert arguments["left_robot_file"] == str(robot_root.resolve())
    assert arguments["right_robot_file"] == str(robot_root.resolve())
    assert "/stale/" not in json.dumps(arguments)


def test_probe_short_circuits_after_first_setup_failure() -> None:
    calls = []

    def probe(body: str, condition: str, seed: int) -> None:
        calls.append((body, condition, seed))
        if len(calls) == 4:
            raise RuntimeError("setup failed")

    attempt = roster.probe_candidate(2026104000, probe)
    assert len(calls) == 4
    assert attempt["all_ten_setup_cells_stable"] is False
    assert [cell["status"] for cell in attempt["cells"]] == [
        "setup_succeeded_and_closed",
        "setup_succeeded_and_closed",
        "setup_succeeded_and_closed",
        "setup_failed",
        *(["not_attempted_after_first_setup_failure"] * 6),
    ]
    assert attempt["actor_inference_calls"] == 0
    assert attempt["task_action_calls"] == 0
    assert attempt["label_or_outcome_reads"] == 0


def test_roster_selects_exact_first_twenty_and_binds_file_sha(tmp_path: Path) -> None:
    value = _valid_roster()
    validated = roster.validate_stable_seed_roster(
        value, expected_preregistration_file_sha256="a" * 64
    )
    assert validated["selected_seeds"][0] == 2026104001
    assert validated["selected_seeds"][-1] == 2026104020

    path = tmp_path / "roster.json"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    file_sha = roster.sha256_file(path)
    assert roster.load_stable_seed_roster_file(
        path,
        file_sha,
        expected_preregistration_file_sha256="a" * 64,
    ) == value
    with pytest.raises(roster.StableSeedRosterError, match="file SHA"):
        roster.load_stable_seed_roster_file(path, "b" * 64)

    tampered = copy.deepcopy(value)
    tampered["selected_seeds"] = list(reversed(tampered["selected_seeds"]))
    unsigned = dict(tampered)
    unsigned.pop("logical_sha256")
    tampered["logical_sha256"] = roster.canonical_sha256(unsigned)
    with pytest.raises(roster.StableSeedRosterError, match="selection changed"):
        roster.validate_stable_seed_roster(tampered)
