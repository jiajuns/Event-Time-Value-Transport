from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import robotwin2_actor_execution_protocol_v1 as actor_execution
import robotwin2_wcm_future_latent_baseline_v1 as wcm
import train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1 as trainer


def labelled_batch(rows: int = 8, horizon: int = 6) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260914)
    return {
        "state": torch.randn(rows, wcm.STATE_DIM, generator=generator),
        "actions": torch.randn(
            rows, horizon, wcm.ACTION_DIM, generator=generator
        ),
        "action_mask": torch.ones(rows, horizon, dtype=torch.bool),
        "action_available": torch.ones(rows),
        "action_schema_id": torch.zeros(rows, dtype=torch.long),
        "body_id": torch.zeros(rows, dtype=torch.long),
        "event_age_seconds": torch.arange(rows, dtype=torch.float32),
        "remaining_action_budget": torch.full((rows,), 50.0),
        "dt": torch.full((rows,), 0.05),
        "terminal_max_event_id": torch.arange(rows) % wcm.EVENT_COUNT,
        "success": (torch.arange(rows) % 2).float(),
        "success_mask": torch.ones(rows),
        "terminal_stage_progress": (torch.arange(rows) % 5).float() / 4.0,
        "terminal_event_mask": torch.ones(rows),
        "terminal_goal_progress": 0.01
        * torch.randn(rows, generator=generator),
        "terminal_goal_progress_mask": torch.ones(rows),
        "object_delta": 0.01
        * torch.randn(rows, wcm.OBJECT_EFFECT_DIM, generator=generator),
        "object_delta_mask": torch.ones(rows),
    }


def runtime_batch(candidate_count: int) -> dict[str, object]:
    batch: dict[str, object] = labelled_batch(candidate_count)
    for name in (
        "terminal_max_event_id",
        "success",
        "success_mask",
        "terminal_stage_progress",
        "terminal_event_mask",
        "terminal_goal_progress",
        "terminal_goal_progress_mask",
        "object_delta",
        "object_delta_mask",
    ):
        batch.pop(name)
    batch["candidate_index"] = torch.arange(candidate_count)
    batch["logical_group"] = ["same-root"] * candidate_count
    return batch


def normalization_receipt(model: wcm.WCMFutureLatentBaseline) -> dict[str, object]:
    value: dict[str, object] = {
        "format": "etsf_wcm_matched_primary_source_normalization_v1",
        "canonical_state_schema": wcm.STATE_SCHEMA,
        "canonical_action_schema": wcm.ACTION_SCHEMA,
        "state_continuous_channels": list(range(18)),
        "state_binary_channels_unchanged": list(range(18, wcm.STATE_DIM)),
        "state_mean": model.state_mean.tolist(),
        "state_std": model.state_std.tolist(),
        "action_mean": model.action_mean.tolist(),
        "action_std": model.action_std.tolist(),
        "primary_source_train_rows": 64,
        "supplement_rows_used": 0,
        "validation_rows_used": 0,
        "heldout_rows_used": 0,
    }
    value["logical_sha256"] = wcm.canonical_sha256(value)
    return value


def checkpoint_value(
    model: wcm.WCMFutureLatentBaseline, *, member: int, seed: int
) -> dict[str, object]:
    protocol = actor_execution.execution_protocol(5)
    file_sha = "a" * 64
    binding = {
        "format": actor_execution.FILE_BINDING_FORMAT,
        "path_root": "/frozen/actor-protocol",
        "path": "execute5.json",
        "file_sha256": file_sha,
        "protocol_logical_sha256": protocol["logical_sha256"],
        "protocol": protocol,
    }
    return {
        "format": wcm.CHECKPOINT_FORMAT,
        "model_family": wcm.MODEL_FAMILY,
        "config": dict(vars(model.config)),
        "model": model.state_dict(),
        "member": member,
        "seed": seed,
        "step": 3000,
        "held_out_body": "ur5",
        "source_bodies": list(wcm.BODIES[:-1]),
        "canonical_state_schema": wcm.STATE_SCHEMA,
        "canonical_action_schema": wcm.ACTION_SCHEMA,
        "state_action_frame_contract": copy.deepcopy(
            wcm.STATE_ACTION_FRAME_CONTRACT
        ),
        "event_spec_sha256": wcm.EVENT_SPEC_SHA256,
        "actor_execution_protocol": protocol,
        "actor_execution_protocol_binding": binding,
        "actor_execution_protocol_file_sha256": file_sha,
        "primary_binding_file_sha256": "b" * 64,
        "supplement_binding_file_sha256": "c" * 64,
        "heldout_rows_used_for_training_normalization_or_selection": 0,
        "trainable_parameter_count": wcm.count_trainable_parameters(model),
        "rank_score_contract": copy.deepcopy(wcm.RANK_SCORE_CONTRACT),
        "trainer_file_sha256": "d" * 64,
        "preflight_logical_sha256": "e" * 64,
        "normalization": normalization_receipt(model),
        "validation": {
            "step": 3000,
            "checkpoint_selection_uses_only_source_strict_proper": True,
            "source_validation": {"source_validation_only": True},
            "supplement_source_validation": {
                "source_validation_only": True
            },
        },
    }


def test_parameter_budget_is_matched_to_v13() -> None:
    receipt = wcm.parameter_budget_receipt(wcm.WCMFutureLatentBaseline())
    assert receipt["trainable_parameters"] == 221_558
    assert receipt["v13_trainable_parameter_reference"] == 223_287
    assert 0.95 <= receipt["ratio_to_v13"] <= 1.05


def test_real_runtime_forward_is_label_free_and_action_conditioned() -> None:
    torch.manual_seed(5)
    model = wcm.WCMFutureLatentBaseline()
    batch = runtime_batch(4)
    batch["state"][:] = batch["state"][0]
    batch["event_age_seconds"][:] = batch["event_age_seconds"][0]
    output = model(batch)
    assert output["candidate_rank_logit"].shape == (4,)
    assert "target_future_latent" not in output
    assert not torch.equal(
        output["predicted_future_latent"][0],
        output["predicted_future_latent"][1],
    )


def test_joint_loss_backpropagates_through_action_dynamics_and_target_encoder() -> None:
    model = wcm.WCMFutureLatentBaseline()
    batch = labelled_batch()
    output = model(batch)
    loss, pieces = wcm.compute_wcm_loss(
        model,
        output,
        batch,
        sample_weight=torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(pieces) >= {
        "latent_mse",
        "success_binary_nll",
        "value_diagonal_gaussian_nll",
        "sigreg",
        "variance_covariance",
    }
    assert model.action_sequence_encoder.weight_ih_l0.grad is not None
    assert model.future_target_encoder[0].weight.grad is not None
    assert model.success_head.weight.grad is not None
    assert torch.isfinite(model.future_target_encoder[0].weight.grad).all()


def test_zero_bootstrap_weight_cannot_leak_target_labels() -> None:
    model = wcm.WCMFutureLatentBaseline()
    batch = labelled_batch()
    output = model(batch)
    loss, pieces = wcm.compute_wcm_loss(
        model, output, batch, sample_weight=torch.zeros(8)
    )
    assert float(loss.detach()) == 0.0
    assert all(float(value.detach()) == 0.0 for value in pieces.values())


def test_masked_branch_effect_never_enters_latent_target_or_regularizers() -> None:
    torch.manual_seed(11)
    model = wcm.WCMFutureLatentBaseline()
    first = labelled_batch()
    first["object_delta_mask"][4:] = 0.0
    changed = {key: value.clone() for key, value in first.items()}
    changed["object_delta"][4:] = 1000.0
    first_loss, _ = wcm.compute_wcm_loss(model, model(first), first)
    changed_loss, _ = wcm.compute_wcm_loss(model, model(changed), changed)
    torch.testing.assert_close(first_loss, changed_loss, rtol=0.0, atol=0.0)


class _FixedMember(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values))

    def forward(self, batch: object) -> dict[str, torch.Tensor]:
        return {"candidate_rank_logit": self.values}


@pytest.mark.parametrize("candidate_count", [4, 8])
def test_runtime_scorer_is_outcome_blind_and_supports_n4_n8(
    candidate_count: int,
) -> None:
    values = [0.0] * candidate_count
    values[-1] = 1.0
    ensemble = wcm.WCMFutureLatentEnsemble(
        [_FixedMember(values) for _ in range(5)]  # type: ignore[list-item]
    )
    batch = runtime_batch(candidate_count)
    result = wcm.score_candidate_pool(
        ensemble, batch, candidate_count=candidate_count
    )
    assert result["selected_candidate_index"] == candidate_count - 1
    assert result["candidate_rank_score_members"].shape == (5, candidate_count)
    assert result["rank_score_contract"][
        "candidate_outcomes_or_labels_read_at_inference"
    ] is False
    contaminated = dict(batch)
    contaminated["success"] = torch.ones(candidate_count)
    with pytest.raises(wcm.WCMBaselineError, match="outcome"):
        wcm.score_candidate_pool(
            ensemble, contaminated, candidate_count=candidate_count
        )


def test_runtime_tie_break_is_lowest_candidate_index() -> None:
    ensemble = wcm.WCMFutureLatentEnsemble(
        [_FixedMember([1.0] * 4) for _ in range(5)]  # type: ignore[list-item]
    )
    result = wcm.score_candidate_pool(ensemble, runtime_batch(4), candidate_count=4)
    assert result["selected_candidate_index"] == 0


def test_five_member_checkpoint_roundtrip_and_protocol_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = wcm.WCMFutureLatentBaseline()
    paths = []
    for member in range(5):
        path = tmp_path / f"member_{member}.pt"
        torch.save(checkpoint_value(model, member=member, seed=100 + member), path)
        paths.append(path)
    ensemble, receipts = wcm.load_five_member_ensemble(paths)
    assert len(ensemble.models) == 5
    assert [receipt["member"] for receipt in receipts] == list(range(5))

    changed = checkpoint_value(model, member=0, seed=100)
    changed["actor_execution_protocol"] = actor_execution.execution_protocol(50)
    tampered = tmp_path / "tampered.pt"
    torch.save(changed, tampered)
    with pytest.raises(wcm.WCMBaselineError, match="binding disagrees"):
        wcm.load_member_checkpoint(tampered)


def test_primary_normalization_is_source_only_and_shared_body() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "body": "aloha-agilex" if index < 4 else "arx-x5",
                "logical_group": f"g{index // 4}",
                "state": np.arange(wcm.STATE_DIM, dtype=np.float32) + index,
                "actions": np.full((3, wcm.ACTION_DIM), index, dtype=np.float32),
                "action_mask": np.ones(3, dtype=bool),
                "action_available": 1.0,
                "action_schema_id": 0,
            }
        )
    receipt = trainer.fit_primary_source_normalization(rows, held_out_body="ur5")
    assert receipt["primary_source_train_rows"] == 8
    assert receipt["supplement_rows_used"] == 0
    assert receipt["validation_rows_used"] == 0
    assert receipt["heldout_rows_used"] == 0
    assert receipt["state_mean"][18:] == [0.0] * 9
    assert receipt["state_std"][18:] == [1.0] * 9
    with pytest.raises(trainer.WCMTrainingError, match="held-out"):
        trainer.fit_primary_source_normalization(
            [{**rows[0], "body": "ur5"}], held_out_body="ur5"
        )


def test_source_validation_aggregates_real_mask_support_across_uneven_batches() -> None:
    model = wcm.WCMFutureLatentBaseline()
    batch = labelled_batch()
    batch["object_delta_mask"][-1] = 0.0
    rows = [
        {name: value[index] for name, value in batch.items()}
        for index in range(8)
    ]
    result = trainer.evaluate_model(
        model, DataLoader(rows, batch_size=5, shuffle=False), torch.device("cpu")
    )
    assert result["source_validation_only"] is True
    assert result["row_count"] == 8
    assert result["real_label_support"][
        "object_effect_diagonal_gaussian_nll"
    ] == 7.0
    assert result["real_label_support"]["latent_mse"] == 7.0
    assert np.isfinite(result["strict_proper_selection_score"])


def test_training_row_router_never_materializes_heldout_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_group = {"tag": "train"}
    validation_group = {"tag": "validation"}
    heldout_group = {"tag": "heldout"}
    calls: list[list[str]] = []

    monkeypatch.setattr(
        trainer.source,
        "source_group_split",
        lambda *args, **kwargs: ([train_group], [validation_group], [heldout_group]),
    )

    def materialize(groups: list[dict[str, str]], *, held_out_body: str) -> list:
        tags = [group["tag"] for group in groups]
        assert "heldout" not in tags
        calls.append(tags)
        return []

    monkeypatch.setattr(trainer, "materialize_primary_rows", materialize)
    result = trainer._training_rows_and_receipts(
        {}, None, held_out_body="ur5", split_seed=1
    )
    assert calls == [["train"], ["validation"]]
    assert result["train_rows"] == []
    assert result["validation_rows"] == []


def test_condition_restoration_uses_only_manifest_group_identity() -> None:
    groups = [
        {
            "body": "aloha-agilex",
            "condition": "clean",
            "group_id": "seed-1",
        }
    ]
    rows = [
        {"logical_group": "aloha-agilex|clean|seed-1", "candidate_index": index}
        for index in range(4)
    ]
    result = trainer._conditioned_rows(rows, groups, supplement=False)
    assert {row["condition"] for row in result} == {"clean"}
    assert all("success" not in row for row in result)
