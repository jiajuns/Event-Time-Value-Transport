from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_schema6_multiseed_v2 as watcher  # noqa: E402
import run_smolvla_piper_schema6_multiseed_v2 as runner  # noqa: E402
from preregister_smolvla_piper_schema6_multiseed_collection_v2 import (  # noqa: E402
    FORMAT,
    STATUS,
    canonical_sha256,
    file_sha256,
    validate_completed_prefix,
)
from resolve_smolvla_piper_target_reset_only import array_sha256, scene_sha256  # noqa: E402


INSTRUCTION = "move the can into the pot"


def _prereg(root: Path) -> dict:
    commands = []
    for global_index in range(130):
        split = "adaptation" if global_index < 80 else "validation"
        ordinal = global_index if split == "adaptation" else global_index - 80
        seed = 1000 + global_index
        seed_root = root / split / f"group_{ordinal:03d}_seed_{seed}"
        command = {
            "split": split,
            "ordinal": ordinal,
            "requested_seed": seed,
            "expected_resolved_seed": seed,
            "pair_id": f"{global_index + 1:064x}",
            "expected_initial_scene_state_sha256": "a" * 64,
            "candidate_original_indices": [0, 1, 2, 3],
            "argv": ["/bound/python", "/bound/runner.py", "collect-one"],
            "outputs": {
                "seed_root": str(seed_root),
                "per_seed_reset_receipt": str(seed_root / "per_seed_reset_receipt.json"),
                "group_hdf5": str(seed_root / "schema6_group.hdf5"),
                "completed_group_receipt": str(seed_root / "completed_group_receipt.json"),
            },
            "bindings": {},
        }
        command["command_sha256"] = canonical_sha256(command)
        commands.append(command)
    value = {
        "format": FORMAT,
        "status": STATUS,
        "production_execution_authorized": False,
        "collection_scope": {
            "ordered_splits": ["adaptation", "validation"],
            "evaluation_commands_generated": 0,
            "evaluation_environment_resets_authorized": 0,
        },
        "execution_contract": {"gpu_lock_path": str(root.parent / "gpu.lock")},
        "outputs": {"future_collection_root": str(root)},
        "commands": commands,
    }
    value["preregistration_sha256"] = canonical_sha256(value)
    return value


def _identity() -> tuple[dict, dict]:
    raw = {
        "scene_state": {
            "can_pose": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
            "pot_pose": [0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        },
        "measured_joint_state": np.arange(14, dtype=float),
        "commanded_drive_target": np.arange(14, dtype=float) + 0.5,
    }
    target = {
        "instruction": INSTRUCTION,
        "resolved_seed": 1000,
        "initial_scene_state_sha256": scene_sha256(raw["scene_state"]),
        "initial_measured_joint_state_sha256": array_sha256(
            raw["measured_joint_state"], role="measured"
        ),
        "initial_commanded_drive_target_sha256": array_sha256(
            raw["commanded_drive_target"], role="commanded"
        ),
    }
    return raw, target


def _apis(counter: dict):
    registry = {
        "format": "fake_registry_v1",
        "objects": [
            {"name": "can", "asset": "105_sauce-can/base0"},
            {"name": "pot", "asset": "060_kitchenpot/base0"},
        ],
    }

    def collect_dense_group(**kwargs):
        for _ in range(4):
            kwargs["runtime"]["reset"](kwargs["requested_seed"], kwargs["instruction"])
        mask = np.asarray([True, False, True, True])
        return {
            "status": "collected_development_group",
            "resolved_seed": kwargs["requested_seed"],
            "root_query": {
                "feasibility_mask": mask,
                "native_action_sha256": [f"{index + 1:x}" * 64 for index in range(4)],
            },
            "branches": [
                {"branch_index": branch_index, "original_candidate_index": original}
                for branch_index, original in enumerate([0, 2, 3])
            ],
        }

    def save_group(path, _record):
        with h5py.File(path, "w") as handle:
            handle.attrs["fake_legacy"] = True

    collector = {
        "collect_dense_group": collect_dense_group,
        "save_schema6_group": save_group,
        "validate_schema6_group_file": lambda _path: {"branches": 3},
    }
    registry_api = {
        "build_runtime_object_registry": lambda _task: registry,
        "assert_runtime_registry_identity": lambda _task, expected: counter.__setitem__(
            "registry_checks", counter.get("registry_checks", 0) + int(expected == registry)
        ),
        "build_pose_quality_spec": lambda value, move_can_pot_source: {
            "format": "fake_pose_v1",
            "registry_sha": canonical_sha256(value),
            "source_sha": move_can_pot_source["sha256"],
        },
        "registry_sha256": canonical_sha256,
        "spec_sha256": lambda value, expected_registry_sha256: canonical_sha256(value),
    }
    return collector, registry_api


def _make_writable(root: Path) -> None:
    for directory, names, files in os.walk(root):
        Path(directory).chmod(0o755)
        for name in names:
            (Path(directory) / name).chmod(0o755)
        for name in files:
            (Path(directory) / name).chmod(0o644)


def test_dependency_injected_core_verifies_every_reset_and_accounts_all_four() -> None:
    temp = Path(tempfile.mkdtemp(prefix="schema6_phase2_core_", dir="/tmp"))
    try:
        output = temp / "collection"
        prereg = _prereg(output)
        command = prereg["commands"][0]
        identity, target = _identity()
        command["expected_initial_scene_state_sha256"] = target[
            "initial_scene_state_sha256"
        ]
        command["command_sha256"] = canonical_sha256(
            {key: value for key, value in command.items() if key != "command_sha256"}
        )
        prereg["preregistration_sha256"] = canonical_sha256(
            {key: value for key, value in prereg.items() if key != "preregistration_sha256"}
        )
        counter = {"resets": 0}

        def reset(seed, instruction):
            counter["resets"] += 1
            return {"observation": True}, seed, instruction

        runtime = {
            "reset": reset,
            "identity_snapshot": lambda: identity,
            "task": lambda: "fake-live-task",
        }
        collector, registry_api = _apis(counter)
        receipt = runner.collect_one_core(
            preregistration=prereg,
            command=command,
            target_row=target,
            runtime=runtime,
            query_fn=lambda *_args: {},
            event_spec={"event": "fake"},
            move_can_pot_source={"path": "/bound/move_can_pot.py", "sha256": "b" * 64},
            max_steps=3,
            collector_api=collector,
            registry_api=registry_api,
        )
        assert counter["resets"] == 5
        assert counter["registry_checks"] == 4
        assert receipt["branch_records"] == 4
        group = Path(command["outputs"]["group_hdf5"])
        with h5py.File(group, "r") as handle:
            accounting = handle["candidate_accounting_v2"]
            assert accounting["original_candidate_index"][:].tolist() == [0, 1, 2, 3]
            assert accounting["executed"][:].tolist() == [True, False, True, True]
            assert accounting["right_censored"][:].tolist() == [False, True, False, False]
        assert len(validate_completed_prefix(prereg, [receipt])) == 129
    finally:
        if temp.exists():
            _make_writable(temp)
            shutil.rmtree(temp)


def test_identity_mismatch_stops_before_collector_or_policy_query() -> None:
    identity, target = _identity()
    changed = dict(identity)
    changed["commanded_drive_target"] = np.zeros(14)
    with pytest.raises(runner.Phase2RunnerError, match="commanded"):
        runner.validate_reset_identity(changed, target)


def test_gap_free_prefix_rejects_later_partial_seed_directory() -> None:
    temp = Path(tempfile.mkdtemp(prefix="schema6_phase2_prefix_", dir="/tmp"))
    try:
        output = temp / "collection"
        prereg = _prereg(output)
        later = Path(prereg["commands"][5]["outputs"]["seed_root"])
        later.mkdir(parents=True)
        with pytest.raises(watcher.Phase2WatcherError, match="gap|partial"):
            watcher.validate_completed_prefix_files(prereg)
    finally:
        if temp.exists():
            _make_writable(temp)
            shutil.rmtree(temp)


def test_preflight_without_execution_authority_fails_before_output_claim() -> None:
    temp = Path(tempfile.mkdtemp(prefix="schema6_phase2_preflight_", dir="/tmp"))
    try:
        output = temp / "collection"
        prereg = _prereg(output)
        path = temp / "preregistration.json"
        path.write_text(json.dumps(prereg, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(runner.Phase2RunnerError, match="authority dependency is absent"):
            watcher.static_preflight(
                preregistration_path=path,
                expected_preregistration_file_sha256=file_sha256(path),
                execution_authority_path=None,
                output_root=output,
                gpu_index=0,
                gpu_lock_path=temp / "gpu.lock",
            )
        assert not output.exists()
    finally:
        if temp.exists():
            _make_writable(temp)
            shutil.rmtree(temp)


def test_two_idle_samples_and_ppid1_are_required_without_timeout() -> None:
    outputs = iter(
        [
            "NVIDIA GeForce RTX 4090, GPU-a\n", "77\n",
            "NVIDIA GeForce RTX 4090, GPU-a\n", "\n",
            "NVIDIA GeForce RTX 4090, GPU-a\n", "\n",
        ]
    )
    sleeps = []
    audits = watcher.wait_two_idle_forever(
        0,
        interval=0.01,
        run_text=lambda _command: next(outputs),
        sleep=sleeps.append,
    )
    assert len(audits) == 2
    parents = iter([19, 7, 1])
    ppid_sleeps = []
    watcher.wait_for_ppid1(getppid=lambda: next(parents), sleep=ppid_sleeps.append)
    assert len(ppid_sleeps) == 2


def test_infeasible_branch_cannot_be_reported_as_executed() -> None:
    record = {
        "status": "collected_development_group",
        "root_query": {
            "feasibility_mask": np.asarray([True, False, True, True]),
            "native_action_sha256": [f"{index + 1:x}" * 64 for index in range(4)],
        },
        "branches": [
            {"branch_index": 0, "original_candidate_index": 0},
            {"branch_index": 1, "original_candidate_index": 1},
            {"branch_index": 2, "original_candidate_index": 2},
            {"branch_index": 3, "original_candidate_index": 3},
        ],
    }
    with pytest.raises(runner.Phase2RunnerError, match="accounting"):
        runner.four_candidate_accounting(record)
