from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smolvla_piper_schema6_runtime_adapter_v2 as adapter  # noqa: E402


class Pose:
    p = np.asarray([0.1, 0.2, 0.3])
    q = np.asarray([1.0, 0.0, 0.0, 0.0])


class Actor:
    def get_pose(self):
        return Pose()


def observation() -> dict:
    return {
        "states": np.arange(14, dtype=np.float32)[None, :],
        "main_images": np.zeros((1, 2, 3, 3), dtype=np.uint8),
        "wrist_images": np.zeros((1, 2, 2, 3, 3), dtype=np.uint8),
        "task_descriptions": [adapter.INSTRUCTION],
    }


def test_identity_snapshot_uses_bound_measured_and_commanded_channels() -> None:
    robot = types.SimpleNamespace(
        get_left_arm_real_jointState=lambda: np.arange(7, dtype=np.float64) + 0.5,
        get_right_arm_real_jointState=lambda: np.arange(7, dtype=np.float64) + 7.5,
    )
    task = types.SimpleNamespace(
        can=Actor(), pot=Actor(), robot=robot,
    )
    result = adapter.identity_snapshot(
        task, observation(), adapter.DEFAULT_MEASURED_CHANNEL
    )
    assert set(result) == {"scene_state", "measured_joint_state", "commanded_drive_target"}
    assert result["measured_joint_state"].shape == (14,)
    assert result["commanded_drive_target"].tolist() == list(map(float, range(14)))
    assert set(result["scene_state"]) == {"can_pose", "pot_pose"}


def test_build_runtime_returns_exact_runner_interface_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    runtime_mapping = {
        name: (lambda *_args, **_kwargs: None)
        for name in ("reset", "identity_snapshot", "task", "snapshot", "step", "derive_events")
    }
    fake_runtime = types.SimpleNamespace(mapping=lambda: runtime_mapping)
    contract = {
        "max_episode_steps": 17,
        "piper_action_bounds": [[-1.0, 1.0]] * 14,
    }
    monkeypatch.setattr(adapter, "_load_authority_runtime_contract", lambda: contract)
    monkeypatch.setattr(
        adapter,
        "_build_resources",
        lambda contract, output_parent, load_policy: {
            "runtime": fake_runtime, "policy": "policy", "preprocessor": "pre",
            "postprocessor": "post", "capture": "capture", "device": "cuda:0",
            "close": lambda: calls.setdefault("closed", True),
        },
    )
    collector = types.ModuleType("collect_smolvla_piper_schema6_dense_event_branches")

    def query(**kwargs):
        calls["query"] = kwargs
        return {"candidate": True}

    collector.generate_candidate_query = query
    monkeypatch.setitem(sys.modules, collector.__name__, collector)
    command = {
        "split": "adaptation", "requested_seed": 123,
        "outputs": {"seed_root": "/tmp/schema6_runtime_synthetic/group_000"},
    }
    built = adapter.build_runtime(command, {"format": "synthetic_event_spec"})
    assert set(built) == {"runtime", "query_fn", "max_steps", "close"}
    assert set(built["runtime"]) == {
        "reset", "identity_snapshot", "task", "snapshot", "step", "derive_events"
    }
    assert built["max_steps"] == 17
    assert built["query_fn"]({"obs": True}, 4) == {"candidate": True}
    assert calls["query"]["scene_seed"] == 123
    assert calls["query"]["query_index"] == 4
    built["close"]()
    assert calls["closed"] is True


def test_reset_factory_reuses_runtime_without_loading_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}
    contract = {"runtime_contract_sha256": "a" * 64, "reset_scratch_path": "/tmp/schema6_reset_synthetic"}

    class Runtime:
        def reset(self, seed, instruction):
            return observation(), seed, instruction

        def identity_snapshot(self):
            return {
                "scene_state": {"can_pose": [0.0] * 7, "pot_pose": [0.0] * 7},
                "measured_joint_state": np.zeros(14),
                "commanded_drive_target": np.zeros(14),
            }

    monkeypatch.setattr(adapter, "validate_runtime_contract", lambda value: contract)

    def resources(value, *, output_parent, load_policy):
        calls.update(value=value, output_parent=output_parent, load_policy=load_policy)
        return {"runtime": Runtime(), "close": lambda: calls.setdefault("closed", True)}

    monkeypatch.setattr(adapter, "_build_resources", resources)
    built = adapter.build_reset_only_adapter(
        plan={"plan": True}, authorization={"runtime_contract": contract}
    )
    result = built.reset_once(7, adapter.INSTRUCTION)
    assert result["resolved_seed"] == 7
    assert result["setup_status"] == "stable"
    assert calls["load_policy"] is False
    assert calls["output_parent"] == Path(contract["reset_scratch_path"])
    built.close()
    assert calls["closed"] is True


def test_reset_factory_marks_implicit_seed_retry_unstable_without_exposing_retry() -> None:
    runtime = types.SimpleNamespace(
        reset=lambda seed, instruction: (observation(), seed + 1, instruction),
        identity_snapshot=lambda: pytest.fail("identity must not be read after retry"),
    )
    built = adapter.ResetOnlyAdapter(runtime, lambda: None)
    result = built.reset_once(9, adapter.INSTRUCTION)
    assert result == {"setup_status": "unstable", "requested_seed": 9}
