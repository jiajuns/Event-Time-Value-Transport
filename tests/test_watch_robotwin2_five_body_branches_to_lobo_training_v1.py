from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_robotwin2_five_body_branches_to_lobo_training_v1 as watcher  # noqa: E402


def _core(path: Path) -> np.ndarray:
    count, horizon = 4, 5
    actions = np.zeros((count, horizon, 14), dtype=np.float32)
    actions[1, :, 0] = 1.0
    actions[2, :, 1] = 2.0
    actions[3, :, 2] = -3.0
    arrays: dict[str, np.ndarray] = {}
    for name in watcher.REQUIRED_ARRAYS:
        if name == "state":
            arrays[name] = np.zeros((count, 27), dtype=np.float32)
        elif name == "actions":
            arrays[name] = actions
        elif name == "action_mask":
            arrays[name] = np.ones((count, horizon), dtype=bool)
        elif name == "object_delta":
            arrays[name] = np.zeros((count, 6), dtype=np.float32)
        elif name == "candidate_index":
            arrays[name] = np.arange(count, dtype=np.int64)
        elif name == "dt":
            arrays[name] = np.full(count, 5.0 / 15.0, dtype=np.float32)
        elif name in watcher.INTEGER_ARRAYS:
            arrays[name] = np.zeros(count, dtype=np.int64)
        else:
            arrays[name] = np.zeros(count, dtype=np.float32)
    np.savez(path, **arrays)
    return actions


def _pairwise(actions: np.ndarray) -> np.ndarray:
    first = actions[:, None, :5, :]
    second = actions[None, :, :5, :]
    return np.sqrt(np.mean(np.square(first - second), axis=(2, 3))).astype(np.float32)


def test_diagnostic_values_and_action_rms_are_fully_replayed(tmp_path: Path) -> None:
    core_path = tmp_path / "group.npz"
    actions = _core(core_path)
    decision = watcher.validate_decision_npz(
        core_path, watcher.sha256_file(core_path)
    )
    diagnostic_path = tmp_path / "group.diagnostics.npz"
    np.savez(
        diagnostic_path,
        first_executed=np.asarray([5, 4, 3, 0], dtype=np.int64),
        branch_error=np.asarray([False, False, True, False], dtype=bool),
        candidate_action_pairwise_rms=_pairwise(actions),
    )
    watcher.validate_diagnostic_npz(
        diagnostic_path,
        watcher.sha256_file(diagnostic_path),
        decision["candidate_action_pairwise_rms"],
    )

    invalid = _pairwise(actions)
    invalid[0, 1] = np.nan
    np.savez(
        diagnostic_path,
        first_executed=np.asarray([-1, 6, 3, 0], dtype=np.int64),
        branch_error=np.asarray([False, False, True, False], dtype=bool),
        candidate_action_pairwise_rms=invalid,
    )
    with pytest.raises(watcher.LoboWatcherError):
        watcher.validate_diagnostic_npz(
            diagnostic_path,
            watcher.sha256_file(diagnostic_path),
            decision["candidate_action_pairwise_rms"],
        )
