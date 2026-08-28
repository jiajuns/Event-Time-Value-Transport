from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_multibody_canonical_event_world_model as core  # noqa: E402
from train_multibody_leave_one_body_out import (  # noqa: E402
    body_mapping,
    build_frozen_split_plan,
    evaluate_lobo_ensemble,
    evaluate_source_only_baseline,
    fit_source_only_action_normalization,
    materialize_source_rows,
    materialize_target_development_rows,
    reserve_zero_target_clock_row,
    run_synthetic_smoke,
    strict_leave_one_body_out_split,
    verify_frozen_split_plan,
)


def _descriptors() -> list[core.GroupDescriptor]:
    rows = []
    for body in ("aloha-agilex", "ARX-X5", "piper", "ur5-wsg"):
        for seed in range(20):
            rows.append(
                core.GroupDescriptor(
                    source="synthetic",
                    body=body,
                    policy="synthetic",
                    task="move_can_pot",
                    seed=seed,
                    path=Path(f"{body}_{seed}.hdf5"),
                )
            )
    return rows


def _rows(count: int = 20, *, body: str = "piper") -> list[dict[str, object]]:
    batch = core.synthetic_batch(count)
    batch["next_event_mask"].fill_(1)
    batch["duration_observed"].fill_(1)
    batch["success"] = (torch.arange(count) % 2).float()
    result = []
    for index in range(count):
        row = {key: value[index].numpy() for key, value in batch.items()}
        row.update(
            {
                "logical_group": f"{body}-g{index}",
                "body": body,
                "policy": "synthetic-policy",
                "task": "move_can_pot" if index % 2 else "lift_pot",
            }
        )
        result.append(row)
    return result


def test_strict_lobo_excludes_target_from_fit_and_keeps_test_sealed() -> None:
    first = strict_leave_one_body_out_split(
        _descriptors(), held_out_body="piper", split_seed=17
    )
    second = strict_leave_one_body_out_split(
        list(reversed(_descriptors())), held_out_body="piper", split_seed=17
    )
    assert {row.logical_group for row in first.source_train} == {
        row.logical_group for row in second.source_train
    }
    assert all(row.body != "piper" for row in first.source_train)
    assert all(row.body != "piper" for row in first.source_validation)
    assert all(row.body == "piper" for row in first.target_development)
    assert all(row.body == "piper" for row in first.target_unused_train)
    memberships = [
        {row.logical_group for row in lane} for lane in first.lanes().values()
    ]
    for index, left in enumerate(memberships):
        for right in memberships[index + 1 :]:
            assert not left & right
    assert set().union(*memberships) == {
        row.logical_group for row in _descriptors()
    }


@pytest.mark.parametrize("held_out", ["piper", "ur5-wsg"])
def test_frozen_plan_is_signed_and_recomputed(held_out: str, tmp_path: Path) -> None:
    split = strict_leave_one_body_out_split(
        _descriptors(), held_out_body=held_out, split_seed=9
    )
    audit = {
        "input_sha256": {"manifest": "a" * 64},
        "event_spec_sha256": "b" * 64,
    }
    plan = build_frozen_split_plan(
        split, held_out_body=held_out, split_seed=9, binding_audit=audit
    )
    path = tmp_path / "lobo_split.json"
    core.atomic_json(path, plan)
    receipt = verify_frozen_split_plan(path, core.sha256_file(path), plan)
    assert receipt["verified_against_current_inputs"] is True
    assert plan["labels_used_for_assignment"] is False
    assert plan["target_unused_train_payload_opened"] == 0
    assert plan["sealed_test_group_hdf5_opened"] == 0
    changed = dict(plan)
    changed["split_seed"] = 10
    with pytest.raises(ValueError, match="differs from label-free recomputation"):
        verify_frozen_split_plan(path, core.sha256_file(path), changed)
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        verify_frozen_split_plan(path, "0" * 64, plan)


def test_payload_boundaries_never_open_target_before_selection() -> None:
    split = strict_leave_one_body_out_split(
        _descriptors(), held_out_body="piper", split_seed=11
    )
    calls: list[list[str]] = []

    def loader(
        descriptors: list[core.GroupDescriptor] | tuple[core.GroupDescriptor, ...],
        _event_spec: object,
    ) -> list[dict[str, object]]:
        calls.append([row.body for row in descriptors])
        return [
            {"body": row.body, "logical_group": row.logical_group}
            for row in descriptors
        ]

    train, validation = materialize_source_rows(
        split, {}, "piper", loader=loader
    )
    assert train and validation
    assert len(calls) == 2
    assert all(body != "piper" for call in calls for body in call)
    with pytest.raises(RuntimeError, match="cannot open before checkpoint selection"):
        materialize_target_development_rows(
            split, {}, "piper", checkpoints_selected=False, loader=loader
        )
    assert len(calls) == 2
    target = materialize_target_development_rows(
        split, {}, "piper", checkpoints_selected=True, loader=loader
    )
    assert target and all(row["body"] == "piper" for row in target)
    assert len(calls) == 3
    # target_unused_train and sealed_test are never passed to the loader.
    assert set(calls[-1]) == {"piper"}


def test_source_normalization_marks_unseen_openvla_schema_identity() -> None:
    rows = []
    for schema in (0, 1):
        for value in (1.0, 3.0):
            rows.append(
                {
                    "actions": np.full((3, 14), value + schema, dtype=np.float32),
                    "action_mask": np.asarray([True, True, False]),
                    "action_available": np.float32(1.0),
                    "action_schema_id": np.int64(schema),
                    "logical_group": f"s{schema}-{value}",
                }
            )
    receipt = fit_source_only_action_normalization(rows)
    mean, std = core.action_normalization_arrays(receipt)
    assert receipt["observed_source_schema_ids"] == [0, 1]
    assert receipt["unseen_schema_ids"] == [2]
    assert receipt["schemas"]["openvla"]["transfer_status"] == (
        "unseen_source_schema_frozen_identity"
    )
    assert np.array_equal(mean[2], np.zeros(14, dtype=np.float32))
    assert np.array_equal(std[2], np.ones(14, dtype=np.float32))
    assert receipt["held_out_rows_used"] == 0


def test_body_ablation_and_reserved_target_clock_row_are_auditable() -> None:
    source = _rows(12, body="aloha-agilex") + _rows(12, body="ARX-X5")
    conditioned, target_id = body_mapping(source, "piper", "source_body_clock")
    agnostic, no_target_id = body_mapping(source, "piper", "body_agnostic")
    assert len(set(conditioned.values())) == 3
    assert target_id == conditioned["piper"]
    assert set(agnostic.values()) == {0}
    assert no_target_id is None

    torch.manual_seed(3)
    model = core.MultibodyCanonicalEventWorldModel(
        core.ModelConfig(body_count=3, dropout=0.0)
    )
    hook = reserve_zero_target_clock_row(model, target_id)
    batch = core.synthetic_batch(10)
    batch["body_id"] %= 2  # source rows only
    loss, _ = core.compute_multitask_loss(model(batch), batch)
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    assert torch.equal(
        model.clock.body_beta.weight[target_id],
        torch.zeros_like(model.clock.body_beta.weight[target_id]),
    )
    assert hook is not None
    hook.remove()


def test_target_ensemble_metrics_calibration_uncertainty_and_baseline() -> None:
    rows = _rows(20, body="piper")
    mapping = {"piper": 0}
    models = []
    for seed in range(5):
        torch.manual_seed(seed)
        models.append(
            core.MultibodyCanonicalEventWorldModel(
                core.ModelConfig(body_count=1, dropout=0.0)
            ).eval()
        )
    metrics = evaluate_lobo_ensemble(
        models, rows, mapping, torch.device("cpu"), batch_size=7
    )
    assert metrics["split"] == "frozen_target_development_only"
    assert metrics["post_event"]["macro_f1"] is not None
    assert metrics["post_event"]["ece_10bin"] is not None
    assert metrics["next_event"]["macro_f1"] is not None
    assert metrics["observed_duration_mae"] is not None
    assert metrics["success_auroc_status"] == "available"
    assert metrics["success_brier"] is not None
    assert metrics["success_ece_10bin"] is not None
    assert metrics["object_rmse"] is not None
    assert metrics["object_support"] == 20
    assert metrics["duration_uncertainty"]["mean_epistemic_std"] is not None
    assert metrics["object_uncertainty"]["mean_epistemic_std"] > 0

    source = _rows(20, body="aloha-agilex")
    baseline = core.fit_train_baselines(source)
    baseline_metrics = evaluate_source_only_baseline(baseline, rows)
    assert baseline_metrics["target_rows_used_to_fit"] == 0
    assert baseline_metrics["object_rmse"] is not None
    assert baseline_metrics["observed_duration_mae"] is not None


def test_forbidden_split_plan_namespace_is_rejected(tmp_path: Path) -> None:
    forbidden = tmp_path / "confirmation" / "plan.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden path token"):
        verify_frozen_split_plan(forbidden, "0" * 64, {})


def test_cpu_synthetic_smoke_and_cli() -> None:
    result = run_synthetic_smoke()
    assert result["status"] == "synthetic_smoke_passed"
    assert result["reserved_target_clock_gradient_is_zero"] is True
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "train_multibody_leave_one_body_out.py"),
            "--mode",
            "synthetic-smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SYNTHETIC_SMOKE=" in completed.stdout
    assert "synthetic_smoke_passed" in completed.stdout
