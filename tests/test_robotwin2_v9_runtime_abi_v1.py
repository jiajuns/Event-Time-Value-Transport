from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_robotwin2_five_body_paired_success_v1 as paired  # noqa: E402
import run_robotwin2_five_body_postformal_candidate_pool_v1 as postformal  # noqa: E402


def _write_real_v10_fold(root: Path) -> Path:
    trainer = paired.shared_head
    fold_root = root / "outer_lobo_franka"
    fold_root.mkdir()
    trainer_sha = paired.sha256_file(Path(trainer.__file__).resolve())
    event_sha = paired.sha256_file(Path(paired.analytic_event.__file__).resolve())
    selected_step = 100
    source_bodies = [body for body in paired.BODIES if body != "franka"]
    members = []
    for member in range(5):
        seed = 20260901 + member
        torch.manual_seed(seed)
        model = trainer.EffectAlignedSharedEventHead().eval()
        checkpoint_path = fold_root / f"member_{member}.pt"
        torch.save(
            {
                "format": trainer.FORMAT,
                "held_out_body": "franka",
                "source_bodies": source_bodies,
                "body_adapter": "single_shared_row_zero_heldout_parameters",
                "body_to_id_source_only": {body: 0 for body in source_bodies},
                "canonical_state_schema": trainer.CANONICAL_STATE_SCHEMA,
                "canonical_action_schema": trainer.CANONICAL_ACTION_SCHEMA,
                "event_age_contract": trainer.event_age_contract(),
                "terminal_horizon_contract": trainer.terminal_horizon_contract(),
                "state_action_frame_contract": trainer.state_action_frame_contract(),
                "model_family": trainer.MODEL_FAMILY,
                "candidate_rank_contract": trainer.checkpoint_candidate_rank_contract(
                    "full"
                ),
                "ablation": trainer.ablation_contract("full"),
                "heldout_rows_used_for_training_normalization_or_selection": 0,
                "rank_supervision_available": True,
                "candidate_rank_parameters_received_direct_supervision": True,
                "synthetic_success_labels": 0,
                "rank_supervision_mode": "informative_dense_only",
                "action_stem_count": 1,
                "member": member,
                "seed": seed,
                "event_spec_sha256": paired.EVENT_SPEC_SHA256,
                "event_derivation_implementation_sha256": event_sha,
                "trainer_file_sha256": trainer_sha,
                "ensemble_common_selection_step": selected_step,
                "model": model.state_dict(),
            },
            checkpoint_path,
        )
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": selected_step,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": paired.sha256_file(checkpoint_path),
                "trainer_file_sha256": trainer_sha,
            }
        )
    summary = {
        "format": trainer.FORMAT,
        "status": "source_only_checkpoint_selection_complete",
        "held_out_body": "franka",
        "source_bodies": source_bodies,
        "body_adapter": "single_shared_row_zero_heldout_parameters",
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "heldout_specific_trainable_parameters": 0,
        "actor_frozen": True,
        "state_action_frame_contract": trainer.state_action_frame_contract(),
        "event_spec_sha256": paired.EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": event_sha,
        "candidate_rank_contract": trainer.summary_candidate_rank_contract("full"),
        "event_age_contract": trainer.event_age_contract(),
        "terminal_horizon_contract": trainer.terminal_horizon_contract(),
        "ablation": trainer.ablation_contract("full"),
        "trainer_file_sha256": trainer_sha,
        "rank_supervision_available": True,
        "candidate_rank_parameters_received_direct_supervision": True,
        "synthetic_success_labels": 0,
        "rank_supervision_mode": "informative_dense_only",
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "rank_aggregation": trainer.risk_adjusted_rank_ensemble_contract(),
            "selected_step": selected_step,
            "heldout_rows_used": 0,
        },
        "members": members,
    }
    (fold_root / "training_summary.json").write_text(
        json.dumps(summary, sort_keys=True), encoding="utf-8"
    )
    return fold_root


def _current_ee() -> np.ndarray:
    current = np.zeros(16, dtype=np.float32)
    current[[3, 11]] = 1.0
    current[[7, 15]] = 0.25
    return current


def _candidates(count: int) -> np.ndarray:
    current = _current_ee()
    values = np.repeat(current[None, None], count * 7, axis=0).reshape(
        count, 7, 16
    )
    ramp = np.linspace(0.001, 0.007, 7, dtype=np.float32)
    for index in range(count):
        values[index, :, 0] += ramp * (index + 1)
        values[index, :, 8] -= ramp * (index + 1) / 2.0
    return values


def _state27() -> np.ndarray:
    state = np.zeros(27, dtype=np.float32)
    state[18] = 1.0
    return state


def test_real_v10_checkpoint_loads_and_scores_n4_and_n8_runner_batches(
    tmp_path: Path,
) -> None:
    fold_root = _write_real_v10_fold(tmp_path)
    inspected = paired.inspect_fold("franka", fold_root)
    models = paired.load_ensemble(inspected, torch.device("cpu"))
    assert len(models) == 5
    assert all(not model.training for model in models)

    n4_batch = paired.scoring_batch(
        state=_state27(),
        current_ee=_current_ee(),
        candidates=_candidates(4),
        current_event=0,
        event_age_seconds=0.25,
        remaining_action_budget=100,
        action_exec_steps=5,
        dt=1.0 / 15.0,
        device=torch.device("cpu"),
    )
    n4 = paired.score_candidates(models, n4_batch)
    assert len(n4["candidate_rank_score_members"]) == 5
    assert len(n4["candidate_rank_score_mean"]) == 4
    assert 0 <= n4["selected_candidate_index"] < 4

    n8_batch = postformal.scoring_batch(
        state=_state27(),
        current_ee=_current_ee(),
        candidates=_candidates(8),
        current_event=0,
        event_age_seconds=0.25,
        remaining_action_budget=100,
        candidate_count=8,
        device=torch.device("cpu"),
    )
    n8 = postformal.score_candidates(models, n8_batch, candidate_count=8)
    assert len(n8["candidate_rank_score_members"]) == 5
    assert len(n8["candidate_rank_score_mean"]) == 8
    assert 0 <= n8["selected_candidate_index"] < 8
    assert postformal.runtime_rank_ensemble_contract(8)[
        "epistemic_risk_weight"
    ] == 0.25
