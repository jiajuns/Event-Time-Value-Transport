from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openvla_etsf_counterfactual_oof import (  # noqa: E402
    EXPECTED_GROUPS,
    FIXED_TRAINING_STEPS,
    make_oof_folds,
    reduce_oof_predictions,
    validate_oof_folds,
)
import train_openvla_etsf_counterfactual_oof as oof_trainer  # noqa: E402
from train_openvla_etsf_counterfactual_oof import _event_context  # noqa: E402


def logical_keys() -> list[str]:
    return [f"move_can_pot|piper|{1000 + index}" for index in range(EXPECTED_GROUPS)]


def logical_keys_250() -> list[str]:
    return [f"move_can_pot|piper|{5000 + index}" for index in range(250)]


def raw_predictions(manifest: dict, helpful_groups: int) -> list[dict]:
    owner = {
        key: int(fold["fold_id"])
        for fold in manifest["folds"]
        for key in fold["oof_holdout_groups"]
    }
    rows = []
    for index, key in enumerate(sorted(owner)):
        success = np.asarray(
            [0.0, 1.0 if index < helpful_groups else 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        logits = np.asarray(
            [
                [-1.0, 2.0, -1.1, -1.2],
                [-0.8, 2.2, -1.0, -1.3],
                [-1.2, 1.8, -0.9, -1.1],
            ]
        )
        rows.append(
            {
                "logical_key": key,
                "fold_id": owner[key],
                "member_success_logits": logits,
                "member_event_progress": np.zeros_like(logits),
                "member_normalized_duration": np.zeros_like(logits),
                "member_aleatoric": np.full_like(logits, 0.01),
                "success": success,
                "steps": np.asarray([100.0, 80.0, 100.0, 100.0]),
                "candidate_distance": np.asarray([0.0, 0.1, 0.2, 0.3]),
                "baseline_index": 0,
                "candidate_names": [
                    "deterministic",
                    "sample_blend_0.250",
                    "sample_blend_0.500",
                    "sample_blend_0.750",
                ],
            }
        )
    return rows


def test_label_free_folds_are_deterministic_balanced_and_leak_free() -> None:
    first = make_oof_folds(logical_keys())
    second = make_oof_folds(list(reversed(logical_keys())))
    assert first == second
    audit = validate_oof_folds(first, logical_keys())
    assert audit == {"development_groups": 100, "unique_oof_groups": 100}
    assert [fold["training_group_count"] for fold in first["folds"]] == [80] * 5
    assert [fold["oof_holdout_group_count"] for fold in first["folds"]] == [20] * 5
    assert FIXED_TRAINING_STEPS == 100
    assert first["training_steps"] == 100
    assert first["training_step_rationale"] == (
        "constant_group_exposure_scaled_from_pre_oof_retry2b_"
        "100groups_100steps_before_any_expanded_oof_prediction"
    )


def test_expanded_250_folds_are_balanced_and_reduction_uses_all_groups() -> None:
    manifest = make_oof_folds(logical_keys_250())
    audit = validate_oof_folds(manifest, logical_keys_250())
    assert audit == {"development_groups": 250, "unique_oof_groups": 250}
    assert manifest["expected_groups"] == 250
    assert manifest["training_steps"] == 250
    assert [fold["training_group_count"] for fold in manifest["folds"]] == [200] * 5
    assert [fold["oof_holdout_group_count"] for fold in manifest["folds"]] == [50] * 5
    selection = reduce_oof_predictions(
        raw_predictions(manifest, helpful_groups=50), manifest
    )
    assert selection["oof_prediction_groups"] == 250
    assert selection["authorization"]["total_oof_groups"] == 250
    assert selection["authorization"]["authorized"]


def test_fold_manifest_rejects_same_group_in_two_holdouts() -> None:
    manifest = make_oof_folds(logical_keys())
    duplicate = manifest["folds"][0]["oof_holdout_groups"][0]
    replaced = manifest["folds"][1]["oof_holdout_groups"][0]
    manifest["folds"][1]["oof_holdout_groups"][0] = duplicate
    manifest["folds"][1]["training_groups"].remove(duplicate)
    manifest["folds"][1]["training_groups"].append(replaced)
    manifest.pop("preregistration_sha256")
    with pytest.raises(RuntimeError, match="multiple OOF holdouts"):
        validate_oof_folds(manifest, logical_keys())


def test_oof_authorizes_only_with_broad_helpful_headroom() -> None:
    manifest = make_oof_folds(logical_keys())
    selection = reduce_oof_predictions(raw_predictions(manifest, helpful_groups=20), manifest)
    assert selection["oof_prediction_groups"] == 100
    assert selection["authorization"]["authorized"]
    assert selection["authorization"]["helpful_changes"] == 20
    assert selection["authorization"]["harmful_changes"] == 0
    assert selection["authorization"]["fresh_confirmation_allowed"]
    assert selection["guard"]["enabled"]


def test_oof_cannot_create_headroom_and_fails_closed() -> None:
    manifest = make_oof_folds(logical_keys())
    selection = reduce_oof_predictions(raw_predictions(manifest, helpful_groups=1), manifest)
    assert not selection["authorization"]["authorized"]
    assert not selection["authorization"]["fresh_confirmation_allowed"]
    assert not selection["guard"]["enabled"]
    assert "insufficient_total_oracle_headroom" in selection["authorization"][
        "rejection_reasons"
    ]


def test_fifth_expansion_candidate_is_training_only_not_authorization() -> None:
    manifest = make_oof_folds(logical_keys())
    rows = raw_predictions(manifest, helpful_groups=0)
    for row in rows:
        row["member_success_logits"] = np.concatenate(
            [row["member_success_logits"], np.full((3, 1), 12.0)], axis=1
        )
        for key in (
            "member_event_progress",
            "member_normalized_duration",
            "member_aleatoric",
        ):
            row[key] = np.concatenate([row[key], np.zeros((3, 1))], axis=1)
        row["success"] = np.append(row["success"], 1.0)
        row["steps"] = np.append(row["steps"], 50.0)
        row["candidate_distance"] = np.append(row["candidate_distance"], 0.4)
        row["candidate_names"].append("sample_blend_1.000")
    selection = reduce_oof_predictions(rows, manifest)
    assert selection["authorization"]["oracle_headroom_groups"] == 0
    assert not selection["authorization"]["authorized"]
    assert selection["candidate_authorization_contract"] == {
        "deployment_candidate_names": [
            "deterministic",
            "sample_blend_0.250",
            "sample_blend_0.500",
            "sample_blend_0.750",
        ],
        "training_only_extra_candidates": ["sample_blend_1.000"],
        "calibration_scoring_guard_use_deployment_candidates_only": True,
    }
    assert "exact_sign_p_fails_familywise_threshold" in selection["authorization"][
        "rejection_reasons"
    ]


def test_raw_prediction_must_come_from_unique_owner_fold() -> None:
    manifest = make_oof_folds(logical_keys())
    rows = raw_predictions(manifest, helpful_groups=20)
    rows[0]["fold_id"] = (rows[0]["fold_id"] + 1) % 5
    with pytest.raises(RuntimeError, match="heldout fold"):
        reduce_oof_predictions(rows, manifest)


def test_oof_trainer_canonicalizes_factual_policy_path(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text('{"calibration": {}}', encoding="utf-8")
    _, _, _, policy_to_id = _event_context(
        event_spec,
        {
            "object_names": ["can"],
            "body_to_id": {"piper": 0},
            "policy_to_id": {
                "/home/user/checkpoints/openvla-oft-7b-robotwin": 0
            },
        },
    )
    assert policy_to_id == {"openvla": 0}


def test_raw_oof_uses_action_adjusted_logit_and_uncertainty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state, strict=True):
            assert strict

        def to(self, device):
            return self

        def eval(self):
            return self

    group = SimpleNamespace(
        candidate_names=[
            "deterministic",
            "sample_blend_0.250",
            "sample_blend_0.500",
            "sample_blend_0.750",
        ],
        logical_key="move_can_pot|piper|99",
        success=np.asarray([0.0, 1.0, 0.0, 0.0]),
        steps=np.asarray([100.0, 50.0, 100.0, 100.0]),
        candidate_distance=np.asarray([0.0, 0.2, 0.3, 0.4]),
    )
    output = {
        "success_logit": torch.zeros(4),
        "next_event_logits": torch.zeros(4, 5),
        "duration_selected_log_mean": torch.zeros(4),
        "aleatoric_uncertainty": torch.full((4,), 0.1),
    }
    monkeypatch.setattr(oof_trainer, "ActionConditionedEventWorldModel", FakeModel)
    monkeypatch.setattr(
        oof_trainer.torch,
        "load",
        lambda *args, **kwargs: {"model": {}, "duration_scale": 10.0},
    )
    monkeypatch.setattr(oof_trainer, "collate_groups", lambda *args, **kwargs: {})
    monkeypatch.setattr(oof_trainer, "move_batch", lambda batch, device: batch)
    monkeypatch.setattr(oof_trainer, "forward_model", lambda model, batch: output)
    monkeypatch.setattr(
        oof_trainer,
        "counterfactual_success_logit",
        lambda model, prediction, batch: torch.tensor([0.0, 4.0, -1.0, -2.0]),
    )
    monkeypatch.setattr(
        oof_trainer,
        "counterfactual_aleatoric_uncertainty",
        lambda model, prediction, batch: torch.tensor([0.2, 0.7, 0.3, 0.4]),
    )
    rows = oof_trainer.raw_oof_predictions(
        member_paths=[tmp_path / "member.pt"],
        groups=[group],
        fold_id=2,
        config=SimpleNamespace(num_events=5),
        object_mean=np.zeros(1, dtype=np.float32),
        object_std=np.ones(1, dtype=np.float32),
        device=torch.device("cpu"),
    )
    assert rows[0]["member_success_logits"].tolist() == [
        [0.0, 4.0, -1.0, -2.0]
    ]
    assert np.allclose(rows[0]["member_aleatoric"], [[0.2, 0.7, 0.3, 0.4]])


def test_raw_oof_schema5_emits_full_heldout_structured_prediction_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state, strict=True):
            assert strict

        def to(self, device):
            return self

        def eval(self):
            return self

    group = SimpleNamespace(
        schema_version=5,
        candidate_count=2,
        candidate_names=["deterministic", "sample_blend_0.250"],
        logical_key="move_can_pot|piper|101",
        success=np.asarray([0.0, 1.0]),
        steps=np.asarray([100.0, 50.0]),
        candidate_distance=np.asarray([0.0, 0.2]),
    )
    sample_count = 3
    batch = {
        "candidate_names": ["deterministic", "sample_blend_0.250", "continuation_0"],
        "terminal_mask": torch.tensor([1, 1, 0], dtype=torch.bool),
        "structured_mask": torch.ones(sample_count, dtype=torch.bool),
        "dense_mask": torch.ones(sample_count, dtype=torch.bool),
        "duration_observed": torch.tensor([0, 1, 1], dtype=torch.bool),
        "current_event_id": torch.tensor([0, 0, 1]),
        "clock_event_id": torch.tensor([0, 0, 1]),
        "next_event_id": torch.tensor([0, 1, 2]),
        "next_reached_event_id": torch.tensor([0, 1, 2]),
        "body_id": torch.zeros(sample_count, dtype=torch.long),
        "policy_id": torch.zeros(sample_count, dtype=torch.long),
        "duration": torch.tensor([100.0, 5.0, 4.0]),
        "success": torch.tensor([0.0, 1.0, 0.0]),
        "outcome_id": torch.tensor([0, 1, 0]),
        "trajectory_regress": torch.tensor([0, 0, 1], dtype=torch.bool),
        "trajectory_recovery": torch.tensor([0, 0, 1], dtype=torch.bool),
        "object_delta": torch.tensor([[0.0], [1.0], [-1.0]]),
        "post_predicates": torch.zeros(sample_count, 5),
    }
    output = {
        "success_logit": torch.zeros(sample_count),
        "next_event_logits": torch.zeros(sample_count, 5),
        "next_reached_event_logits": torch.zeros(sample_count, 5),
        "post_predicate_logits": torch.zeros(sample_count, 5),
        "duration_selected_log_mean": torch.zeros(sample_count),
        "duration_selected_log_scale": torch.zeros(sample_count),
        "reach_logit": torch.zeros(sample_count),
        "object_delta_mean": torch.tensor([[0.0], [1.0], [-1.0]]),
        "object_delta_log_scale": torch.zeros(sample_count, 1),
        "outcome_logits": torch.zeros(sample_count, 3),
        "aleatoric_uncertainty": torch.full((sample_count,), 0.1),
    }
    monkeypatch.setattr(oof_trainer, "ActionConditionedEventWorldModel", FakeModel)
    monkeypatch.setattr(
        oof_trainer.torch,
        "load",
        lambda *args, **kwargs: {"model": {}, "duration_scale": 10.0},
    )
    monkeypatch.setattr(oof_trainer, "collate_groups", lambda *args, **kwargs: batch)
    monkeypatch.setattr(oof_trainer, "move_batch", lambda value, device: value)
    monkeypatch.setattr(oof_trainer, "forward_model", lambda model, value: output)
    monkeypatch.setattr(
        oof_trainer,
        "counterfactual_success_logit",
        lambda model, prediction, value: torch.tensor([0.0, 4.0, 1.0]),
    )
    monkeypatch.setattr(
        oof_trainer,
        "counterfactual_aleatoric_uncertainty",
        lambda model, prediction, value: torch.tensor([0.2, 0.7, 0.3]),
    )
    rows = oof_trainer.raw_oof_predictions(
        member_paths=[tmp_path / "member.pt"],
        groups=[group],
        fold_id=2,
        config=SimpleNamespace(
            num_events=5,
            recovery_supervised=False,
            predicate_names=("moved", "lifted", "near_goal", "stationary", "success"),
        ),
        object_mean=np.asarray([2.0], dtype=np.float32),
        object_std=np.asarray([3.0], dtype=np.float32),
        device=torch.device("cpu"),
    )
    structured = rows[0]["structured_predictions"]
    assert structured["format"] == "etsf_oof_structured_prediction_row_v1"
    assert structured["member_next_event_logits"].shape == (1, 3, 5)
    assert structured["member_duration_log_mean"].shape == (1, 3)
    assert structured["member_object_delta_mean"].shape == (1, 3, 1)
    assert structured["terminal_mask"].tolist() == [True, True, False]
    assert structured["object_delta"].reshape(-1).tolist() == [2.0, 5.0, -1.0]
