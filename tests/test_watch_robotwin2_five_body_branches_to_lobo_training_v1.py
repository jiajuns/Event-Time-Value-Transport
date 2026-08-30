from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _valid_supplement_receipt(supplement_sha: str) -> dict[str, object]:
    return {
        "enabled": True,
        "binding_file_sha256": supplement_sha,
        "proper_loss_weight": 0.25,
        "rank_loss_weight": 0.25,
        "rank_or_utility_loss_weight": 0.25,
        "source_train_groups": 90,
        "source_train_rows": 360,
        "heldout_groups_deferred": 30,
        "source_validation_groups": 30,
        "source_validation_rows": 120,
        "source_validation_body": "arx-x5",
        "source_validation_body_selection": (
            "label_blind_sha256_ordered_five_body_cycle_successor_derangement"
        ),
        "source_validation_assignment_uses_labels": False,
        "rank_or_utility_rows_used": 360,
        "rank_or_utility_groups_with_real_comparative_supervision": 90,
        "semantic_comparative_rows_used": 0,
        "normalization_rows_used": 0,
        "baseline_fit_rows_used": 0,
        "source_validation_rows_used": 120,
        "proper_checkpoint_selection_rows_authorized": 120,
        "proper_checkpoint_selection_weight": 0.25,
        "checkpoint_selection_rows_used": 120,
        "checkpoint_selection_use": (
            "strict_proper_only_primary_plus_fixed_0.25_supplement"
        ),
        "rank_selection_rows_authorized": 0,
        "rank_selection_rows_used": 0,
        "calibration_diagnostic_rows_authorized": 120,
        "calibration_diagnostic_rows_used": 120,
        "calibration_rows_used": 0,
        "calibration_fit": False,
        "proper_validation_primary_reset_overlap": 0,
        "heldout_group_npz_opened": 0,
        "heldout_group_payload_bytes_read": 0,
        "heldout_group_payload_deserialized": 0,
        "heldout_manifest_file_opened": 0,
        "heldout_manifest_bytes_read": 0,
    }


def _add_valid_augmented_selection(summary: dict[str, object]) -> None:
    step = int(summary["ensemble_checkpoint_selection"]["selected_step"])
    combined_score = 1.0 + 0.25 * 2.0
    combined_se = float(np.sqrt(0.1**2 + (0.25 * 0.2) ** 2))
    selection_key = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, step]
    record = {
        "step": step,
        "selection_key": selection_key,
        "ensemble_candidate_ranking": {},
        "mean_member_diagnostic_multitask_score": 0.4,
        "mean_member_primary_strict_proper_score": 1.0,
        "mean_member_supplement_strict_proper_score": 2.0,
        "supplement_strict_proper_weight": 0.25,
        "mean_member_strict_proper_score": combined_score,
        "primary_conservative_strict_proper_standard_error": 0.1,
        "supplement_conservative_strict_proper_standard_error": 0.2,
        "conservative_strict_proper_standard_error": combined_se,
        "strict_proper_standard_error_combination": (
            watcher.SUPPLEMENT_STRICT_PROPER_SE_COMBINATION
        ),
    }
    summary["ensemble_checkpoint_selection"].update(
        {
            "strict_proper_score": watcher.SUPPLEMENT_STRICT_PROPER_SCORE,
            "supplement_validation_never_used_for_rank_comparison": True,
            "calibration_diagnostics_only_no_parameter_fit": True,
            "selected_key": selection_key,
            "selected_ensemble_candidate_ranking": {},
            "selected_mean_member_diagnostic_multitask_score": 0.4,
            "strict_proper_selection": {
                "rule": STRICT_PROPER_SELECTION_RULE,
                "comparative_validation_evidence": False,
                "best_score": combined_score,
                "conservative_one_standard_error": combined_se,
                "eligible_threshold": combined_score + combined_se,
                "eligible_steps": [step],
                "selected_step": step,
                "selected_score": combined_score,
                "heldout_rows_used": 0,
            },
            "evaluated_common_steps": [record],
        }
    )


def test_fold_summary_requires_exact_augmented_binding_when_requested(
    valid_fold_summary: tuple[Path, str, str, Path],
) -> None:
    fold_path, held_out_body, binding_sha256, summary_path = valid_fold_summary
    supplement_sha = "e" * 64
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["proper_world_supplement"] = _valid_supplement_receipt(
        supplement_sha
    )
    _add_valid_augmented_selection(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = watcher.summarize_fold(
        fold_path,
        held_out_body,
        binding_sha256,
        supplement_sha,
    )
    assert result["proper_world_supplement"]["binding_file_sha256"] == supplement_sha
    assert result["proper_world_supplement_fold_receipt_contract"] == (
        watcher.SUPPLEMENT_FOLD_RECEIPT_CONTRACT
    )
    proper_validation = result["proper_world_supplement_proper_validation"]
    assert proper_validation["groups"] == 30
    assert proper_validation["rows"] == 120
    assert proper_validation["rank_selection_rows_used"] == 0
    assert proper_validation["calibration_fit"] is False
    assert proper_validation["selected_combined_strict_proper_evidence"][
        "mean_member_strict_proper_score"
    ] == pytest.approx(1.5)
    with pytest.raises(watcher.LoboWatcherError, match="violates outer-LOBO"):
        watcher.summarize_fold(
            fold_path,
            held_out_body,
            binding_sha256,
            "f" * 64,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("source_train_groups", 120),
        ("source_validation_groups", 0),
        ("source_validation_body", "aloha-agilex"),
        ("source_validation_rows_used", 0),
        ("checkpoint_selection_rows_used", 0),
        ("rank_selection_rows_used", 1),
        ("calibration_rows_used", 1),
        ("calibration_fit", True),
    ],
)
def test_fold_summary_rejects_old_or_leaking_supplement_split_receipt(
    valid_fold_summary: tuple[Path, str, str, Path],
    field: str,
    invalid_value: object,
) -> None:
    fold_path, held_out_body, binding_sha256, summary_path = valid_fold_summary
    supplement_sha = "e" * 64
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    supplement = _valid_supplement_receipt(supplement_sha)
    supplement[field] = invalid_value
    summary["proper_world_supplement"] = supplement
    _add_valid_augmented_selection(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(watcher.LoboWatcherError, match="violates outer-LOBO"):
        watcher.summarize_fold(
            fold_path,
            held_out_body,
            binding_sha256,
            supplement_sha,
        )


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("strict_proper_score",), "primary_only"),
        (
            ("evaluated_common_steps", 0, "mean_member_strict_proper_score"),
            9.0,
        ),
        (
            (
                "evaluated_common_steps",
                0,
                "strict_proper_standard_error_combination",
            ),
            "sum_standard_errors",
        ),
        (("strict_proper_selection", "selected_score"), 9.0),
    ],
)
def test_fold_summary_rejects_tampered_augmented_proper_selection_evidence(
    valid_fold_summary: tuple[Path, str, str, Path],
    path: tuple[object, ...],
    invalid_value: object,
) -> None:
    fold_path, held_out_body, binding_sha256, summary_path = valid_fold_summary
    supplement_sha = "e" * 64
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["proper_world_supplement"] = _valid_supplement_receipt(
        supplement_sha
    )
    _add_valid_augmented_selection(summary)
    cursor: object = summary["ensemble_checkpoint_selection"]
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = invalid_value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(watcher.LoboWatcherError):
        watcher.summarize_fold(
            fold_path,
            held_out_body,
            binding_sha256,
            supplement_sha,
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


def _attempt_args(tmp_path: Path) -> SimpleNamespace:
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# immutable trainer\n", encoding="utf-8")
    training_python = tmp_path / "training-python"
    training_python.write_text("", encoding="utf-8")
    binding = tmp_path / "binding.json"
    binding.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "lobo-output"
    output_root.mkdir()
    return SimpleNamespace(
        output_root=output_root,
        training_python=training_python,
        trainer=trainer,
        binding=binding,
        supplement_binding=None,
        supplement_binding_sha256=None,
    )


def _write_attempt_summary(attempt: dict[str, object]) -> None:
    output = Path(attempt["training_output"])
    output.mkdir()
    members = []
    for member, seed in enumerate(watcher.ENSEMBLE_SEEDS):
        checkpoint = output / f"member_{member:02d}.pt"
        checkpoint.write_bytes(f"checkpoint-{member}".encode())
        members.append(
            {
                "member": member,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": watcher.sha256_file(checkpoint),
            }
        )
    (output / "training_summary.json").write_text(
        json.dumps({"members": members}), encoding="utf-8"
    )
    Path(attempt["log"]).write_text("training complete\n", encoding="utf-8")


def test_interrupted_fold_keeps_attempt_and_starts_new_attempt(
    tmp_path: Path,
) -> None:
    args = _attempt_args(tmp_path)
    binding_sha = "b" * 64
    first = watcher.create_fold_attempt(args, "franka", binding_sha)
    Path(first["training_output"]).mkdir()
    (Path(first["training_output"]) / "partial.pt").write_bytes(b"partial")

    second = watcher.create_fold_attempt(args, "franka", binding_sha)
    assert first["manifest"]["attempt_ordinal"] == 1
    assert second["manifest"]["attempt_ordinal"] == 2
    assert Path(first["directory"]).is_dir()
    assert (Path(first["training_output"]) / "partial.pt").is_file()
    history = watcher.validate_fold_attempt_history(
        args, "franka", binding_sha
    )
    assert [row["manifest"]["attempt_ordinal"] for row in history] == [1, 2]


def test_resigned_attempt_tamper_is_rejected_before_retry(tmp_path: Path) -> None:
    args = _attempt_args(tmp_path)
    binding_sha = "b" * 64
    attempt = watcher.create_fold_attempt(args, "franka", binding_sha)
    manifest_path = Path(attempt["directory"]) / "attempt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trainer_file_sha256"] = "d" * 64
    unsigned = dict(manifest)
    unsigned.pop("logical_sha256")
    manifest["logical_sha256"] = watcher.canonical_sha256(unsigned)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(watcher.LoboWatcherError, match="manifest changed"):
        watcher.create_fold_attempt(args, "franka", binding_sha)


def test_post_rename_interruption_recovers_promotion_receipt(
    tmp_path: Path,
) -> None:
    args = _attempt_args(tmp_path)
    binding_sha = "b" * 64
    attempt = watcher.create_fold_attempt(args, "franka", binding_sha)
    _write_attempt_summary(attempt)
    rebind = watcher.rebind_fold_summary_for_atomic_promotion(
        attempt, args.output_root / "outer_lobo_franka"
    )
    final_output = args.output_root / "outer_lobo_franka"
    os.rename(Path(attempt["training_output"]), final_output)

    history = watcher.validate_fold_attempt_history(
        args, "franka", binding_sha
    )
    assert history[-1]["promotion"] is None
    promotion = watcher.recover_missing_promotion_receipt(history, final_output)
    assert promotion["promotion_receipt_recovered_after_interruption"] is True
    assert promotion["summary_rebinding_logical_sha256"] == rebind[
        "logical_sha256"
    ]
    final_summary = json.loads(
        (final_output / "training_summary.json").read_text(encoding="utf-8")
    )
    assert all(
        Path(member["checkpoint"]).is_relative_to(final_output)
        and Path(member["checkpoint"]).is_file()
        for member in final_summary["members"]
    )
    watcher.validate_fold_attempt_history(args, "franka", binding_sha)


def test_complete_attempt_is_atomically_promoted_and_bound(tmp_path: Path) -> None:
    args = _attempt_args(tmp_path)
    binding_sha = "b" * 64
    attempt = watcher.create_fold_attempt(args, "piper", binding_sha)
    _write_attempt_summary(attempt)
    final_output = args.output_root / "outer_lobo_piper"
    rebinding = watcher.rebind_fold_summary_for_atomic_promotion(
        attempt, final_output
    )
    promotion = watcher.promote_fold_attempt(attempt, final_output)
    assert not Path(attempt["training_output"]).exists()
    assert final_output.is_dir()
    assert promotion["summary_rebinding_logical_sha256"] == rebinding[
        "logical_sha256"
    ]
    history = watcher.validate_fold_attempt_history(args, "piper", binding_sha)
    assert history[-1]["promotion"]["logical_sha256"] == promotion[
        "logical_sha256"
    ]
