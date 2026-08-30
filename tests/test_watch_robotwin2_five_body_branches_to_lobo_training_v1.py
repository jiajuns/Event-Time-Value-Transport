from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_robotwin2_five_body_branches_to_lobo_training_v1 as watcher  # noqa: E402


STRICT_PROPER_SELECTION_RULE = (
    "minimize_source_body_condition_macro_proper_score_then_"
    "maximize_rank_within_one_standard_error"
)


@pytest.fixture
def valid_fold_summary(
    tmp_path: Path,
) -> tuple[Path, str, str, Path]:
    held_out_body = "franka"
    binding_sha256 = "b" * 64
    trainer_sha256 = "c" * 64
    selected_step = 3000
    members = []
    for member, seed in enumerate(watcher.ENSEMBLE_SEEDS):
        checkpoint = tmp_path / f"member_{member:02d}.pt"
        checkpoint.write_bytes(f"checkpoint-{member}".encode())
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": selected_step,
                "trainer_file_sha256": trainer_sha256,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": watcher.sha256_file(checkpoint),
                "source_validation": {},
            }
        )
    summary = {
        "status": "source_only_checkpoint_selection_complete",
        "held_out_body": held_out_body,
        "source_bodies": [body for body in watcher.BODIES if body != held_out_body],
        "heldout_group_npz_opened": 0,
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "event_spec_sha256": watcher.EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": "d" * 64,
        "preflight": {
            "binding_file_sha256": binding_sha256,
            "event_derivation_implementation_sha256": "d" * 64,
        },
        "trainer_file_sha256": trainer_sha256,
        "rank_supervision_available": True,
        "candidate_rank_parameters_received_direct_supervision": True,
        "synthetic_success_labels": 0,
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "rank_aggregation": {
                "format": "etsf_bounded_utility_epistemic_lcb_ensemble_v1"
            },
            "selected_step": selected_step,
            "selected_ensemble_candidate_ranking": {},
            "strict_proper_selection": {
                "rule": STRICT_PROPER_SELECTION_RULE,
                "selected_step": selected_step,
                "heldout_rows_used": 0,
            },
        },
        "members": members,
    }
    summary_path = tmp_path / "training_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path, held_out_body, binding_sha256, summary_path


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
        elif name == "remaining_action_budget":
            arrays[name] = np.full(count, 150, dtype=np.float32)
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
        branch_error=np.asarray([False, False, False, False], dtype=bool),
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


def test_fold_summary_accepts_strict_proper_selection_contract(
    valid_fold_summary: tuple[Path, str, str, Path],
) -> None:
    fold_path, held_out_body, binding_sha256, _summary_path = valid_fold_summary

    result = watcher.summarize_fold(fold_path, held_out_body, binding_sha256)

    assert result["ensemble_common_selection_step"] == 3000
    assert result["heldout_labels_used_for_training_normalization_or_selection"] is False


def test_fold_summary_requires_exact_augmented_binding_when_requested(
    valid_fold_summary: tuple[Path, str, str, Path],
) -> None:
    fold_path, held_out_body, binding_sha256, summary_path = valid_fold_summary
    supplement_sha = "e" * 64
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["proper_world_supplement"] = {
        "enabled": True,
        "binding_file_sha256": supplement_sha,
        "proper_loss_weight": 0.25,
        "source_train_groups": 80,
        "source_train_rows": 320,
        "heldout_groups_deferred": 20,
        "source_validation_groups": 0,
        "rank_or_utility_rows_used": 0,
        "normalization_rows_used": 0,
        "source_validation_rows_used": 0,
        "checkpoint_selection_rows_used": 0,
        "calibration_rows_used": 0,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = watcher.summarize_fold(
        fold_path,
        held_out_body,
        binding_sha256,
        supplement_sha,
    )
    assert result["proper_world_supplement"]["binding_file_sha256"] == supplement_sha
    with pytest.raises(watcher.LoboWatcherError, match="violates outer-LOBO"):
        watcher.summarize_fold(
            fold_path,
            held_out_body,
            binding_sha256,
            "f" * 64,
        )


@pytest.mark.parametrize(
    ("tampered_field", "tampered_value"),
    [
        ("rule", "maximize_rank_without_proper_calibration"),
        ("selected_step", 2900),
        ("heldout_rows_used", 1),
    ],
)
def test_fold_summary_rejects_tampered_strict_proper_selection(
    valid_fold_summary: tuple[Path, str, str, Path],
    tampered_field: str,
    tampered_value: object,
) -> None:
    fold_path, held_out_body, binding_sha256, summary_path = valid_fold_summary
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ensemble_checkpoint_selection"]["strict_proper_selection"][
        tampered_field
    ] = tampered_value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(watcher.LoboWatcherError, match="violates outer-LOBO"):
        watcher.summarize_fold(fold_path, held_out_body, binding_sha256)
