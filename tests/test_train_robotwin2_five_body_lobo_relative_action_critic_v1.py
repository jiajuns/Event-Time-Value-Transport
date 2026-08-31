from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import robotwin2_relative_action_critic_adapter_v1 as adapter  # noqa: E402
import train_multibody_canonical_event_world_model as core  # noqa: E402
import train_robotwin2_five_body_lobo_relative_action_critic_v1 as rac  # noqa: E402
import train_robotwin2_five_body_lobo_shared_event_head_v1 as shared  # noqa: E402


def _decision_rows(
    body: str,
    suffix: str,
    *,
    horizon: int = 5,
) -> list[dict[str, object]]:
    # c2/c3 are a real lexicographic tie.  Every other unordered pair is
    # informative: 4 success-tier pairs and one stage-tier pair.
    keys = (
        (0.0, 1, 0.00),
        (0.0, 2, 0.00),
        (1.0, 2, 0.10),
        (1.0, 2, 0.10 + rac.GOAL_EQUALITY_TOLERANCE / 2.0),
    )
    state = np.zeros(core.STATE_DIM, dtype=np.float32)
    state[18] = 1.0
    rows = []
    for candidate, (success, stage, goal) in enumerate(keys):
        actions = np.full(
            (horizon + candidate % 2, core.ACTION_DIM),
            candidate + 0.25,
            dtype=np.float32,
        )
        rows.append(
            {
                "logical_group": f"{body}|clean|{suffix}",
                "body": body,
                "condition": "clean",
                "candidate_index": np.int64(candidate),
                "state": state.copy(),
                "actions": actions,
                "action_mask": np.ones(actions.shape[0], dtype=bool),
                "action_available": np.float32(1.0),
                "action_schema_id": np.int64(0),
                "dt": np.float32(1.0 / 3.0),
                "current_event_id": np.int64(0),
                "event_age_seconds": np.float32(0.0),
                "remaining_action_budget": np.float32(175.0),
                "success_mask": np.float32(1.0),
                "success": np.float32(success),
                "terminal_event_mask": np.float32(1.0),
                "terminal_max_event_id": np.int64(stage),
                "terminal_goal_progress_mask": np.float32(1.0),
                "terminal_goal_progress": np.float32(goal),
            }
        )
    return rows


def _collection() -> rac.PreferencePairCollection:
    rows = [
        *_decision_rows(rac.BODIES[0], "a", horizon=4),
        *_decision_rows(rac.BODIES[1], "b", horizon=7),
    ]
    return rac.build_preference_pairs(
        rows, source_bodies=rac.BODIES[:2], stream="unit_source"
    )


def _runtime_batch(candidate_count: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(99 + candidate_count)
    state = torch.zeros(candidate_count, core.STATE_DIM)
    state[:, 18] = 1.0
    return {
        "state": state,
        "actions": torch.randn(
            candidate_count, 6, core.ACTION_DIM, generator=generator
        ),
        "action_mask": torch.ones(candidate_count, 6, dtype=torch.bool),
        "action_available": torch.ones(candidate_count, dtype=torch.bool),
        "action_schema_id": torch.zeros(candidate_count, dtype=torch.long),
        "body_id": torch.zeros(candidate_count, dtype=torch.long),
        "dt": torch.full((candidate_count,), 1.0 / 3.0),
        "current_event_id": torch.zeros(candidate_count, dtype=torch.long),
        "event_age_seconds": torch.zeros(candidate_count),
        "remaining_action_budget": torch.full((candidate_count,), 175.0),
    }


def _small_model() -> rac.MatchedRelativeActionCritic:
    return rac.MatchedRelativeActionCritic(
        rac.RACConfig(
            model_dim=16,
            transformer_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
    )


def test_real_branch_lexicographic_preference_and_ties() -> None:
    rows = _decision_rows(rac.BODIES[0], "hierarchy")
    assert rac.lexicographic_preference(rows[0], rows[1]) == (0, "stage")
    assert rac.lexicographic_preference(rows[1], rows[2]) == (0, "success")
    assert rac.lexicographic_preference(rows[2], rows[3]) == (None, None)
    goal = dict(rows[3])
    goal["terminal_goal_progress"] = 0.2
    assert rac.lexicographic_preference(goal, rows[2]) == (1, "goal")


def test_pair_builder_is_symmetric_balanced_and_never_labels_ties() -> None:
    collection = _collection()
    audit = collection.audit
    assert len(collection.examples) == 20
    assert audit["labeled_unordered_pairs"] == 10
    assert audit["tied_unordered_pairs_excluded"] == 2
    assert audit["positive_ordered_pairs"] == audit["negative_ordered_pairs"] == 10
    assert audit["unordered_pair_tier_counts"] == {
        "success": 8,
        "stage": 2,
        "goal": 0,
    }
    assert audit["synthetic_or_tie_labels"] == 0
    assert audit["heldout_rows_used"] == 0
    assert all(best == (2, 3) for best in collection.best_candidates.values())
    orientations = {
        (item.logical_group, item.left_candidate, item.right_candidate): item.label
        for item in collection.examples
    }
    for (group, left, right), label in orientations.items():
        assert orientations[group, right, left] == 1.0 - label
    assert not any(
        {item.left_candidate, item.right_candidate} == {2, 3}
        for item in collection.examples
    )


def test_pair_builder_fails_closed_on_heldout_missing_labels_or_context_drift() -> None:
    rows = [
        *_decision_rows(rac.BODIES[0], "a"),
        *_decision_rows(rac.BODIES[1], "b"),
    ]
    with pytest.raises(rac.RelativeActionCriticError, match="non-source body"):
        rac.build_preference_pairs(
            rows, source_bodies=rac.BODIES[:1], stream="heldout_leak"
        )
    missing = [dict(row) for row in rows]
    missing[0]["terminal_goal_progress_mask"] = 0.0
    with pytest.raises(rac.RelativeActionCriticError, match="complete real branch"):
        rac.build_preference_pairs(
            missing, source_bodies=rac.BODIES[:2], stream="missing_label"
        )
    drift = [dict(row) for row in rows]
    drift[1]["event_age_seconds"] = 1.0
    with pytest.raises(rac.RelativeActionCriticError, match="one pre-action"):
        rac.build_preference_pairs(
            drift, source_bodies=rac.BODIES[:2], stream="context_drift"
        )


def test_pair_collation_pads_variable_horizons_without_changing_masks() -> None:
    collection = _collection()
    batch = rac.collate_preference_pairs(collection.examples[:4])
    assert batch["action_i"].shape[0] == 4
    assert batch["action_i"].shape[-1] == core.ACTION_DIM
    assert batch["action_i_mask"].dtype == torch.bool
    assert torch.all(batch["action_i_mask"].sum(1) > 0)
    assert torch.all(batch["action_j_mask"].sum(1) > 0)


def test_model_has_independent_action_stems_antisymmetric_logits_and_gradients() -> None:
    collection = _collection()
    batch = rac.collate_preference_pairs(collection.examples[:6])
    model = _small_model().eval()
    i_parameters = {id(item) for item in model.action_i_encoder.parameters()}
    j_parameters = {id(item) for item in model.action_j_encoder.parameters()}
    difference_parameters = {
        id(item) for item in model.action_difference_encoder.parameters()
    }
    assert not (i_parameters & j_parameters)
    assert not (i_parameters & difference_parameters)
    assert not (j_parameters & difference_parameters)
    logits = model(batch)
    swapped = dict(batch)
    swapped["action_i"], swapped["action_j"] = batch["action_j"], batch["action_i"]
    swapped["action_i_mask"], swapped["action_j_mask"] = (
        batch["action_j_mask"],
        batch["action_i_mask"],
    )
    torch.testing.assert_close(model(swapped), -logits, atol=1e-6, rtol=0.0)
    model.train()
    loss = rac.pairwise_focal_bce_with_logits(model(batch), batch["label"])
    loss.backward()
    assert all(
        any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in module.parameters())
        for module in (
            model.action_i_encoder,
            model.action_j_encoder,
            model.action_difference_encoder,
            model.state_encoder,
            model.context_encoder,
            model.classifier,
        )
    )


def test_focal_gamma_zero_is_exact_bce_and_invalid_labels_fail_closed() -> None:
    logits = torch.tensor([-2.0, 0.0, 1.5])
    labels = torch.tensor([0.0, 1.0, 1.0])
    observed = rac.pairwise_focal_bce_with_logits(logits, labels, gamma=0.0)
    expected = F.binary_cross_entropy_with_logits(logits, labels)
    torch.testing.assert_close(observed, expected)
    with pytest.raises(rac.RelativeActionCriticError, match="inputs are invalid"):
        rac.pairwise_focal_bce_with_logits(logits, torch.tensor([0.0, 0.5, 1.0]))
    zero = rac.pairwise_focal_bce_with_logits(
        logits.requires_grad_(), labels, sample_weight=torch.zeros_like(labels)
    )
    assert zero.shape == () and float(zero.detach()) == 0.0
    zero.backward()
    assert logits.grad is not None


def test_complete_pair_sampler_and_bootstrap_preserve_decision_mass() -> None:
    collection = _collection()
    sampler = rac.CompletePreferenceGroupBatchSampler(
        collection, batch_size_pairs=12, seed=4
    )
    for indices in sampler:
        groups = defaultdict(list)
        for index in indices:
            groups[collection.examples[index].logical_group].append(index)
        for group, observed in groups.items():
            assert len(observed) == collection.group_pair_counts[group]
    weights, audit = rac.group_bootstrap_weights(collection, seed=14)
    assert len(audit) == 5
    raw = rac.collate_preference_pairs(collection.examples)
    for member in range(5):
        pair_weights = rac.pair_sample_weights(
            raw,
            member=member,
            group_weights=weights,
            group_pair_counts=collection.group_pair_counts,
            device=torch.device("cpu"),
        )
        for group in collection.group_pair_counts:
            indices = [
                index
                for index, value in enumerate(raw["logical_group"])
                if value == group
            ]
            assert float(pair_weights[indices].sum()) == pytest.approx(
                weights[group][member]
            )


def test_preference_metrics_recover_lexicographic_best_set() -> None:
    collection = _collection()
    labels = np.asarray([item.label for item in collection.examples])
    logits = np.where(labels > 0.5, 12.0, -12.0)[None]
    metrics = rac.preference_metrics_from_member_logits(logits, collection)
    assert metrics["pair_accuracy"] == 1.0
    assert metrics["decision_best_set_accuracy"] == 1.0
    assert metrics["pair_bce"] < 1e-4
    assert metrics["heldout_rows_used"] == 0


@pytest.mark.parametrize("candidate_count", (4, 8))
def test_runtime_adapter_scores_every_pair_and_is_permutation_equivariant(
    candidate_count: int,
) -> None:
    model = _small_model().eval()
    batch = _runtime_batch(candidate_count)
    output = adapter.score_candidates(
        [model] * 5, batch, candidate_count=candidate_count
    )
    assert output["rac_unordered_pairs_evaluated_per_member"] == candidate_count * (
        candidate_count - 1
    ) // 2
    assert np.asarray(output["candidate_rank_score_members"]).shape == (
        5,
        candidate_count,
    )
    matrices = np.asarray(output["rac_pair_probability_matrix_members"])
    assert matrices.shape == (5, candidate_count, candidate_count)
    np.testing.assert_allclose(
        matrices + matrices.transpose(0, 2, 1), 1.0, atol=1e-6
    )
    assert output["relative_action_critic_runtime_contract"][
        "single_elimination_bracket_used"
    ] is False

    permutation = torch.arange(candidate_count - 1, -1, -1)
    permuted = {
        key: value.index_select(0, permutation)
        if isinstance(value, torch.Tensor) and value.shape[:1] == (candidate_count,)
        else value
        for key, value in batch.items()
    }
    changed = adapter.score_candidates(
        [model] * 5, permuted, candidate_count=candidate_count
    )
    original_scores = np.asarray(output["candidate_rank_score_members"])
    changed_scores = np.asarray(changed["candidate_rank_score_members"])
    inverse = np.argsort(permutation.numpy())
    np.testing.assert_allclose(original_scores, changed_scores[:, inverse], atol=1e-6)


def test_runtime_adapter_fails_closed_on_non_n4_n8_or_context_drift() -> None:
    batch = _runtime_batch(4)
    with pytest.raises(adapter.RelativeActionCriticAdapterError, match="N4 or N8"):
        adapter.runtime_contract(6)
    drift = dict(batch)
    drift["event_age_seconds"] = batch["event_age_seconds"].clone()
    drift["event_age_seconds"][1] = 1.0
    with pytest.raises(adapter.RelativeActionCriticAdapterError, match="one event_age"):
        adapter.all_pair_batch(drift, candidate_count=4)


def test_rac_contract_matches_official_inputs_and_matched_budget() -> None:
    contract = rac.rac_contract()
    assert contract["official_reference"]["arxiv"] == "2605.01194v2"
    assert contract["official_reference"]["training_loss"] == "binary_focal_loss"
    assert contract["body_or_condition_identity_input"] is False
    assert contract["tie_policy"] == "exclude_unordered_pair_no_synthetic_label"
    assert contract["runtime_candidate_counts"] == [4, 8]
    assert contract["ensemble_members"] == 5
    assert contract["steps_per_member_default"] == 3000


def test_cli_defaults_to_five_by_3000_and_exposes_bce_ablation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = [
        "rac-trainer",
        "--mode",
        "preflight",
        "--binding",
        "binding.json",
        "--binding-sha256",
        "0" * 64,
        "--held-out-body",
        rac.BODIES[0],
    ]
    monkeypatch.setattr(sys, "argv", required)
    args = rac.parse_args()
    assert args.steps == 3000
    assert tuple(args.ensemble_seeds) == rac.DEFAULT_ENSEMBLE_SEEDS
    assert args.focal_gamma == 2.0
    monkeypatch.setattr(sys, "argv", [*required, "--focal-gamma", "0"])
    assert rac.parse_args().focal_gamma == 0.0


def _write_fold(tmp_path: Path) -> Path:
    heldout = rac.BODIES[0]
    sources = [body for body in rac.BODIES if body != heldout]
    model = _small_model().eval()
    config = model.config
    normalization = {
        "format": "etsf_rac_source_train_only_normalization_v1",
        "action": {
            "mean": [0.0] * core.ACTION_DIM,
            "std": [1.0] * core.ACTION_DIM,
            "canonical_action_schema_id": 0,
        },
        "state": {
            "mean": [0.0] * core.STATE_DIM,
            "std": [1.0] * core.STATE_DIM,
            "continuous_channels": list(range(18)),
            "binary_channels_unchanged": list(range(18, core.STATE_DIM)),
        },
        "source_rows": 8,
        "supplement_rows_used": 0,
        "heldout_rows_used": 0,
    }
    normalization["logical_sha256"] = rac.canonical_sha256(normalization)
    protocol = {"logical_sha256": "1" * 64}
    protocol_binding = {"protocol_logical_sha256": "1" * 64}
    members = []
    for member, seed in enumerate(rac.DEFAULT_ENSEMBLE_SEEDS):
        path = tmp_path / f"member_{member}.pt"
        torch.save(
            {
                "format": rac.FORMAT,
                "model_family": rac.MODEL_FAMILY,
                "model": model.state_dict(),
                "config": rac.dataclasses.asdict(config),
                "member": member,
                "seed": seed,
                "step": 100,
                "held_out_body": heldout,
                "source_bodies": sources,
                "rac_contract": rac.rac_contract(),
                "canonical_state_schema": shared.CANONICAL_STATE_SCHEMA,
                "canonical_action_schema": shared.CANONICAL_ACTION_SCHEMA,
                "state_action_frame_contract": shared.state_action_frame_contract(),
                "actor_execution_protocol": protocol,
                "actor_execution_protocol_binding": protocol_binding,
                "actor_execution_protocol_file_sha256": "2" * 64,
                "normalization": normalization,
                "heldout_rows_used_for_training_normalization_or_selection": 0,
                "synthetic_or_tie_labels": 0,
            },
            path,
        )
        members.append(
            {
                "member": member,
                "seed": seed,
                "best_step": 100,
                "checkpoint": str(path),
                "checkpoint_sha256": rac.sha256_file(path),
            }
        )
    summary = {
        "format": rac.SUMMARY_FORMAT,
        "status": "source_only_rac_checkpoint_selection_complete",
        "model_family": rac.MODEL_FAMILY,
        "held_out_body": heldout,
        "source_bodies": sources,
        "rac_contract": rac.rac_contract(),
        "actor_execution_protocol": protocol,
        "actor_execution_protocol_binding": protocol_binding,
        "actor_execution_protocol_file_sha256": "2" * 64,
        "checkpoint_selection": {"selected_step": 100, "heldout_rows_used": 0},
        "members": members,
        "heldout_rows_used_for_training_normalization_or_selection": 0,
        "all_checkpoints_selected_before_any_heldout_payload_open": True,
    }
    summary["logical_sha256"] = rac.canonical_sha256(summary)
    path = tmp_path / "training_summary.json"
    path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return path


def test_fold_loader_and_score_form_a_complete_checkpoint_runtime_loop(
    tmp_path: Path,
) -> None:
    summary = _write_fold(tmp_path)
    models, receipt = adapter.load_fold_ensemble(
        summary,
        device=torch.device("cpu"),
        expected_held_out_body=rac.BODIES[0],
    )
    assert len(models) == 5
    assert receipt["heldout_payloads_or_labels_opened"] == 0
    result = adapter.score_candidates(
        models, _runtime_batch(4), candidate_count=4
    )
    assert 0 <= result["selected_candidate_index"] < 4


def test_fold_loader_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    summary_path = _write_fold(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = Path(summary["members"][0]["checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(adapter.RelativeActionCriticAdapterError, match="missing or changed"):
        adapter.load_fold_ensemble(summary_path, device=torch.device("cpu"))
