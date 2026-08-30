from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "shared_head_upgrade_watcher",
    SCRIPTS / "watch_robotwin2_postformal_shared_head_upgrade_v1.py",
)
assert SPEC and SPEC.loader
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        robotwin_python=tmp_path / "robotwin-python",
        training_python=tmp_path / "training-python",
        system_python=tmp_path / "python3",
        code_root=tmp_path / "code",
        actor_checkpoint=tmp_path / "actor",
        actor_authority=tmp_path / "actor-authority.json",
        vlm_metadata=tmp_path / "vlm",
        robotwin_root=tmp_path / "robotwin",
        event_spec=tmp_path / "event.json",
        supplement_root=tmp_path / "supplement",
        supplement_binding=tmp_path / "supplement" / "binding.json",
        primary_binding=tmp_path / "primary-binding.json",
        primary_branches_root=tmp_path / "primary-branches",
        materialization_receipt=tmp_path / "materialization.json",
        augmented_lobo_root=tmp_path / "augmented-lobo",
        augmented_lobo_state=tmp_path / "augmented-lobo.state.json",
        augmented_lobo_run_exit=tmp_path / "augmented-lobo.run.exit",
        augmented_n4_root=tmp_path / "augmented-n4",
        augmented_n8_root=tmp_path / "augmented-n8",
        metrics_preregistration=tmp_path / "metrics.json",
        lerobot_root=tmp_path / "lerobot",
        lerobot_site=tmp_path / "lerobot-site",
        robotwin_eval_site=tmp_path / "robotwin-eval-site",
        etsf_site=tmp_path / "etsf-site",
        poll_seconds=30.0,
        expected_gpu_uuid=watcher.EXPECTED_GPU_UUID,
    )


def test_commands_bind_complete_supplement_and_nested_candidate_study(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    supplement_sha = "a" * 64
    collector = watcher.supplement_collector_command(args, "piper")
    assert collector[collector.index("--body") + 1] == "piper"
    assert "--seeds" not in collector
    action_index = collector.index("--action-exec-steps")
    assert collector[collector.index("--conditions") + 1 : action_index] == [
        "clean",
        "randomized",
    ]

    materializer = watcher.materializer_command(args)
    declarations = [
        materializer[index + 1]
        for index, value in enumerate(materializer)
        if value == "--body-manifest"
    ]
    assert len(declarations) == 5
    assert {value.split("=", 1)[0] for value in declarations} == set(watcher.BODIES)

    lobo = watcher.lobo_command(args, supplement_sha)
    assert lobo[lobo.index("--supplement-binding-sha256") + 1] == supplement_sha
    assert lobo[lobo.index("--supplement-binding") + 1] == str(
        args.supplement_binding
    )

    n4 = watcher.paired_n4_command(args, supplement_sha)
    n8 = watcher.paired_n8_command(args, supplement_sha)
    nested = watcher.nested_n4_n8_command(args, supplement_sha)
    for command in (n4, n8, nested):
        assert command[
            command.index("--required-supplement-binding-sha256") + 1
        ] == supplement_sha
        assert sum(value == "--lobo-fold" for value in command) == 5
    assert "--candidate-count" not in n4
    assert n8[n8.index("--candidate-count") + 1] == "8"
    assert n8[n8.index("--proposal-count") + 1] == "16"
    assert nested[nested.index("--output") + 1] == str(args.augmented_n8_root)
    assert "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py" in nested[1]
    assert "--candidate-count" not in nested
    assert watcher.N8_RETAINED_CANDIDATE_COUNT == 8
    assert watcher.N8_RAW_PROPOSAL_COUNT == 16


def test_runtime_environment_includes_explicit_etsf_dependency_site(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    environment = watcher.runtime_environment(args)
    python_paths = environment["PYTHONPATH"].split(":")
    assert str(args.etsf_site) in python_paths
    assert python_paths.index(str(args.etsf_site)) > python_paths.index(
        str(args.robotwin_eval_site)
    )


def test_supplement_completion_is_exact_design_not_just_group_count(
    tmp_path: Path,
) -> None:
    roster = watcher.supplement_reserve_roster("franka")
    selected = {
        row["slot_key"]: row["ordered_requested_seeds"][1] for row in roster
    }
    groups = [
        {
            "condition": row["condition"],
            "horizon_slot": row["horizon_slot"],
            "requested_seed": selected[row["slot_key"]],
            "scripted_root_event": event,
        }
        for row in roster
        for event in watcher.TARGET_EVENTS
    ]
    attempts = []
    for row in roster:
        rejected_seed = row["ordered_requested_seeds"][0]
        selected_seed = selected[row["slot_key"]]
        attempts.extend(
            (
                {
                    "attempt_id": (
                        f"{row['slot_key']}|requested_seed={rejected_seed}"
                    ),
                    "status": "rejected_before_actor_outcomes",
                    "actor_candidate_outcomes_executed_before_selection": False,
                },
                {
                    "attempt_id": (
                        f"{row['slot_key']}|requested_seed={selected_seed}"
                    ),
                    "status": "complete",
                    "selected_before_actor_candidate_outcomes": True,
                    "actor_candidate_outcomes_executed_before_selection": False,
                },
            )
        )
    value = {
        "format": watcher.SUPPLEMENT_MANIFEST_FORMAT,
        "body": "franka",
        "conditions": list(watcher.CONDITIONS),
        "collection_status": "complete",
        "reserve_roster": roster,
        "pre_registered_seeds": [
            seed for row in roster for seed in row["ordered_requested_seeds"]
        ],
        "selected_seed_by_slot": selected,
        "groups": groups,
        "attempts": attempts,
    }
    value["logical_sha256"] = watcher.canonical_sha256(value)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert watcher.supplement_manifest_complete(path, "franka") is True

    value["groups"][-1]["scripted_root_event"] = "e3"
    unsigned = dict(value)
    unsigned.pop("logical_sha256")
    value["logical_sha256"] = watcher.canonical_sha256(unsigned)
    path.write_text(json.dumps(value), encoding="utf-8")
    assert watcher.supplement_manifest_complete(path, "franka") is False


def test_reserve_rosters_are_body_local_and_disjoint() -> None:
    by_body = {
        body: {
            seed
            for row in watcher.supplement_reserve_roster(body)
            for seed in row["ordered_requested_seeds"]
        }
        for body in watcher.BODIES
    }
    assert all(len(seeds) == 160 for seeds in by_body.values())
    for index, body in enumerate(watcher.BODIES):
        for other in watcher.BODIES[index + 1 :]:
            assert by_body[body].isdisjoint(by_body[other])


def test_upstream_gate_requires_complete_bound_ablation(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    value = {
        "format": watcher.UPSTREAM_FORMAT,
        "status": "complete",
        "summary": str(summary),
        "summary_file_sha256": watcher.sha256_file(summary),
    }
    watcher.validate_upstream_state(value)
    with pytest.raises(watcher.SharedHeadUpgradeError):
        watcher.validate_upstream_state({**value, "status": "waiting"})
    with pytest.raises(watcher.SharedHeadUpgradeError):
        watcher.validate_upstream_state(
            {**value, "summary_file_sha256": "0" * 64}
        )
