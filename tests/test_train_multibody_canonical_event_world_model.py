from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import h5py


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_multibody_canonical_event_world_model import (  # noqa: E402
    CANONICAL_EVENTS,
    GroupDescriptor,
    InputBinding,
    ModelConfig,
    MultibodyCanonicalEventWorldModel,
    TransitionDataset,
    action_normalization_arrays,
    body_alias_receipt,
    canonical_body_name,
    canonical_event_id,
    canonical_event_name,
    censored_lognormal_loss,
    compute_multitask_loss,
    collate_rows,
    ensemble_predict,
    evaluate_train_only_baselines,
    evaluate_validation_model,
    fit_train_baselines,
    fit_train_action_normalization,
    logical_group_bootstrap_weights,
    load_schema5_rows,
    run_synthetic_smoke,
    scan_schema5_groups,
    synthetic_batch,
    strict_group_split,
    validation_selection_key,
    validation_selection_score,
    verify_input_bindings,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(stratum: int, seed: int) -> GroupDescriptor:
    bodies = ("aloha-agilex", "ARX-X5")
    return GroupDescriptor(
        source="synthetic",
        body=bodies[stratum],
        policy=f"policy{stratum}",
        task="move_can_pot",
        seed=seed,
        path=Path(f"group_{stratum}_{seed}.hdf5"),
        raw_body=bodies[stratum],
    )


def test_five_event_mapping_merges_raw_e1_e2() -> None:
    assert CANONICAL_EVENTS == ("e0", "e12", "e3", "e4", "eK")
    assert canonical_event_name("e1") == "e12"
    assert canonical_event_name("e2") == "e12"
    assert canonical_event_id("e1") == canonical_event_id("e2") == 1
    with pytest.raises(ValueError, match="unknown event"):
        canonical_event_name("e_bad")


def test_body_alias_is_explicit_audited_and_unknown_fails_closed() -> None:
    assert canonical_body_name("piper") == "piper"
    assert canonical_body_name("piper_piper_0.6") == "piper"
    descriptors = [
        GroupDescriptor(
            source="stage1_target",
            body="piper",
            raw_body="piper",
            policy="robotwin_scripted",
            task="move_can_pot",
            seed=1,
            path=Path("stage1.hdf5"),
        ),
        GroupDescriptor(
            source="openvla_schema5",
            body="piper",
            raw_body="piper_piper_0.6",
            policy="openvla",
            task="move_can_pot",
            seed=2,
            path=Path("schema5.hdf5"),
        ),
    ]
    receipt = body_alias_receipt(descriptors)
    assert receipt["raw_to_canonical"] == {
        "piper": "piper",
        "piper_piper_0.6": "piper",
    }
    assert receipt["canonical_body_ids"] == ["piper"]
    assert len(receipt["sha256"]) == 64
    with pytest.raises(ValueError, match="unknown body identity"):
        canonical_body_name("piper-v2-guess")


def test_missing_action_rows_are_exactly_zero_and_stems_are_separate() -> None:
    torch.manual_seed(3)
    model = MultibodyCanonicalEventWorldModel(
        ModelConfig(body_count=5, dropout=0.0)
    ).eval()
    batch = synthetic_batch(5)
    output = model(batch)
    unavailable = ~batch["action_available"].bool()
    assert torch.equal(
        output["action_effect"][unavailable],
        torch.zeros(int(unavailable.sum()), 96),
    )

    # The same numeric chunk is interpreted by independent robot-specific stems.
    common = synthetic_batch(3)
    common["actions"][:] = common["actions"][0]
    common["action_mask"][:] = common["action_mask"][0]
    common["action_available"][:] = 1
    common["action_schema_id"] = torch.tensor([0, 1, 2])
    effects = model(common)["action_effect"]
    assert not torch.allclose(effects[0], effects[1])
    assert not torch.allclose(effects[1], effects[2])


def test_unavailable_rows_cannot_supervise_object_or_recovery_heads() -> None:
    torch.manual_seed(5)
    model = MultibodyCanonicalEventWorldModel(
        ModelConfig(body_count=4, dropout=0.0)
    ).eval()
    batch = synthetic_batch(10)
    output = model(batch)
    first, pieces = compute_multitask_loss(output, batch)
    unavailable = ~batch["action_available"].bool()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["object_delta"][unavailable] = 1_000_000.0
    changed["recovery"][unavailable] = 1.0 - changed["recovery"][unavailable]
    second, changed_pieces = compute_multitask_loss(output, changed)
    assert torch.equal(pieces["object"], changed_pieces["object"])
    assert torch.equal(pieces["recovery"], changed_pieces["recovery"])
    assert torch.equal(first, second)
    assert int(pieces["object_supervised_rows"]) == 6
    assert int(pieces["recovery_supervised_rows"]) == 6


def _normalization_row(
    schema: int, value: float, group: str, *, available: bool = True
) -> dict[str, object]:
    actions = np.full((4, 14), value, dtype=np.float32)
    mask = np.asarray([True, True, False, False]) if available else np.zeros(4, dtype=bool)
    return {
        "actions": actions,
        "action_mask": mask,
        "action_available": np.float32(available),
        "action_schema_id": np.int64(schema if available else -1),
        "logical_group": group,
    }


def _full_rows_from_synthetic(batch: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows = []
    bodies = ("aloha-agilex", "ARX-X5", "piper", "ur5-wsg")
    for index in range(len(batch["state"])):
        row = {key: value[index].numpy() for key, value in batch.items()}
        row.update(
            {
                "logical_group": f"group-{index}",
                "body": bodies[index % len(bodies)],
                "policy": "synthetic",
                "task": "move_can_pot",
            }
        )
        rows.append(row)
    return rows


def test_action_normalization_is_train_only_per_schema_and_restorable() -> None:
    train_rows = []
    for schema in range(3):
        train_rows.extend(
            [
                _normalization_row(schema, 1.0 + schema, f"g{schema}a"),
                _normalization_row(schema, 3.0 + schema, f"g{schema}b"),
            ]
        )
    train_rows.append(_normalization_row(-1, 999_999.0, "missing", available=False))
    # This row represents validation and is deliberately not passed to the fit.
    validation_row = _normalization_row(0, 1_000_000.0, "validation")
    receipt = fit_train_action_normalization(train_rows)
    mean, std = action_normalization_arrays(receipt)
    assert np.allclose(mean[:, 0], [2.0, 3.0, 4.0])
    assert np.allclose(std[:, 0], [1.0, 1.0, 1.0])
    assert receipt["validation_rows_used"] == 0
    assert receipt["test_rows_used"] == 0
    assert receipt["unavailable_train_rows_excluded"] == 1
    assert receipt["schemas"]["aloha"]["valid_action_steps"] == 4
    contaminated, _ = action_normalization_arrays(
        fit_train_action_normalization(train_rows + [validation_row])
    )
    assert not np.allclose(contaminated[0], mean[0])

    first = MultibodyCanonicalEventWorldModel(
        ModelConfig(body_count=4, dropout=0.0)
    ).eval()
    first.action.set_normalization(torch.from_numpy(mean), torch.from_numpy(std))
    restored = MultibodyCanonicalEventWorldModel(
        ModelConfig(body_count=4, dropout=0.0)
    ).eval()
    restored.load_state_dict(first.state_dict(), strict=True)
    assert torch.equal(restored.action.action_mean, first.action.action_mean)
    assert torch.equal(restored.action.action_std, first.action.action_std)
    batch = synthetic_batch(5)
    assert torch.equal(
        restored(batch)["action_effect"], first(batch)["action_effect"]
    )
    unavailable = ~batch["action_available"].bool()
    assert torch.equal(
        restored(batch)["action_effect"][unavailable],
        torch.zeros(int(unavailable.sum()), 96),
    )


def test_validation_metrics_baselines_support_and_single_class_auc() -> None:
    batch = synthetic_batch(20)
    # Guarantee support for every validation metric and all three action schemas.
    batch["next_event_mask"][:] = 1
    batch["duration_observed"][:] = 1
    batch["success"] = (torch.arange(20) % 2).float()
    rows = _full_rows_from_synthetic(batch)
    train_baselines = fit_train_baselines(rows)
    baseline_metrics = evaluate_train_only_baselines(train_baselines, rows)
    assert train_baselines["source_split"] == "train_only"
    assert train_baselines["validation_rows_used"] == 0
    assert baseline_metrics["post_event"]["support"] == 20
    assert baseline_metrics["next_event"]["support"] == 20
    assert baseline_metrics["observed_duration_support"] == 20
    assert baseline_metrics["object_support"] > 0

    body_to_id = {
        body: index
        for index, body in enumerate(sorted({str(row["body"]) for row in rows}))
    }
    loader = torch.utils.data.DataLoader(
        TransitionDataset(rows, body_to_id),
        batch_size=7,
        shuffle=False,
        collate_fn=collate_rows,
    )
    model = MultibodyCanonicalEventWorldModel(
        ModelConfig(body_count=len(body_to_id), dropout=0.0)
    ).eval()
    metrics = evaluate_validation_model(model, loader, torch.device("cpu"))
    assert metrics["split"] == "validation_only"
    assert metrics["post_event"]["accuracy"] is not None
    assert metrics["post_event"]["macro_f1"] is not None
    assert metrics["next_event"]["accuracy"] is not None
    assert metrics["observed_duration_mae"] is not None
    assert metrics["observed_duration_nll"] is not None
    assert metrics["success_brier"] is not None
    assert metrics["success_auroc_status"] == "available"
    assert metrics["object_rmse"] is not None
    assert metrics["object_nll"] is not None
    score, components = validation_selection_score(metrics, baseline_metrics)
    assert np.isfinite(score)
    assert len(components) == 5

    single_class = {key: value.clone() for key, value in batch.items()}
    single_class["success"].zero_()
    single_rows = _full_rows_from_synthetic(single_class)
    single_loader = torch.utils.data.DataLoader(
        TransitionDataset(single_rows, body_to_id),
        batch_size=20,
        collate_fn=collate_rows,
    )
    single_metrics = evaluate_validation_model(
        model, single_loader, torch.device("cpu")
    )
    assert single_metrics["success_auroc"] is None
    assert single_metrics["success_auroc_status"] == "unavailable_single_class"


def test_validation_selection_key_uses_primary_ties_and_earlier_step() -> None:
    metrics = {
        "next_event": {"macro_f1": 0.6},
        "success_brier": 0.2,
        "observed_duration_nll": 1.0,
        "object_nll": 0.5,
    }
    assert validation_selection_key(metrics, 0.9, 100) < validation_selection_key(
        metrics, 1.0, 1
    )
    assert validation_selection_key(metrics, 0.9, 100) < validation_selection_key(
        metrics, 0.9, 200
    )


def test_censored_duration_loss_is_finite_and_differentiable() -> None:
    mean = torch.tensor([2.0, 3.0], requires_grad=True)
    log_scale = torch.tensor([-0.5, 0.2], requires_grad=True)
    loss = censored_lognormal_loss(
        mean,
        log_scale,
        torch.tensor([8.0, 30.0]),
        torch.tensor([1.0, 0.0]),
    ).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert log_scale.grad is not None and torch.isfinite(log_scale.grad).all()


def test_group_split_is_stratified_deterministic_and_label_free() -> None:
    descriptors = [
        _descriptor(stratum, seed)
        for stratum in range(2)
        for seed in range(100, 120)
    ]
    first = strict_group_split(descriptors, split_seed=17)
    second = strict_group_split(list(reversed(descriptors)), split_seed=17)
    for name in ("train", "validation", "test"):
        first_ids = {row.logical_group for row in first[name]}
        second_ids = {row.logical_group for row in second[name]}
        assert first_ids == second_ids
        assert {row.stratum for row in first[name]} == {
            "aloha-agilex|policy0|move_can_pot",
            "ARX-X5|policy1|move_can_pot",
        }
    assert not (
        {row.logical_group for row in first["train"]}
        & {row.logical_group for row in first["test"]}
    )


def test_group_bootstrap_is_five_member_and_constant_within_group() -> None:
    groups = ["g0", "g0", "g1", "g2", "g2", "g2"]
    first = logical_group_bootstrap_weights(groups, members=5, seed=9)
    second = logical_group_bootstrap_weights(groups, members=5, seed=9)
    assert first.shape == (5, len(groups))
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 0], first[:, 1])
    assert np.array_equal(first[:, 3], first[:, 4])
    assert np.array_equal(first[:, 4], first[:, 5])
    assert np.all(first.sum(1) > 0)
    with pytest.raises(ValueError, match="exactly five"):
        logical_group_bootstrap_weights(groups, members=3, seed=9)


def test_ensemble_reports_epistemic_disagreement() -> None:
    batch = synthetic_batch(5)
    models = []
    for seed in range(5):
        torch.manual_seed(seed)
        models.append(
            MultibodyCanonicalEventWorldModel(
                ModelConfig(body_count=5, dropout=0.0)
            ).eval()
        )
    output = ensemble_predict(models, batch)
    assert output["epistemic_components"].shape == (5, 4)
    assert output["epistemic_uncertainty"].shape == (5,)
    assert torch.isfinite(output["epistemic_uncertainty"]).all()
    assert bool((output["epistemic_uncertainty"] > 0).any())


def _make_binding(tmp_path: Path) -> InputBinding:
    stage1 = tmp_path / "stage1"
    stage1.mkdir(parents=True)
    source = stage1 / "source_manifest.json"
    source.write_text(json.dumps({"entries": []}), encoding="utf-8")
    target = stage1 / "target_manifest.csv"
    target.write_text("task,embodiment,seed,path,valid_rollout\n", encoding="utf-8")
    event = tmp_path / "event_spec.json"
    event.write_text(
        json.dumps({"calibration": {name: {} for name in (
            "adjust_bottle",
            "handover_block",
            "move_can_pot",
            "place_container_plate",
            "beat_block_hammer",
            "lift_pot",
        )}}),
        encoding="utf-8",
    )
    groups = tmp_path / "branches" / "groups"
    groups.mkdir(parents=True)
    (groups / "group_0.hdf5").write_bytes(b"not-opened-by-preflight")
    manifest = groups.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "status": "complete",
                "task": "move_can_pot",
                "body": "piper_piper_0.6",
                "model_path": "openvla",
                "event_spec_sha256": _sha256(event),
                "groups": [{"path": "group_0.hdf5", "seed": 1}],
            }
        ),
        encoding="utf-8",
    )
    return InputBinding(
        stage1_root=stage1,
        stage1_source_manifest=source,
        stage1_source_manifest_sha256=_sha256(source),
        stage1_target_manifest=target,
        stage1_target_manifest_sha256=_sha256(target),
        event_spec=event,
        event_spec_sha256=_sha256(event),
        openvla_schema5_manifest=manifest,
        openvla_schema5_manifest_sha256=_sha256(manifest),
    )


def test_preflight_binds_sha_and_does_not_open_group_payload(tmp_path: Path) -> None:
    binding = _make_binding(tmp_path)
    audit = verify_input_bindings(binding)
    assert audit["schema5_groups"] == 1
    assert audit["schema5_group_hdf5_opened"] == 0
    assert audit["sealed_test_group_hdf5_opened"] == 0
    groups = scan_schema5_groups(binding)
    assert groups[0].raw_body == "piper_piper_0.6"
    assert groups[0].body == "piper"
    changed = dataclasses.replace(
        binding, openvla_schema5_manifest_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_input_bindings(changed)


def test_schema5_loader_materializes_canonical_action_transition(tmp_path: Path) -> None:
    path = tmp_path / "group_0.hdf5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "object_names", data=np.asarray(["can"], dtype=object), dtype=string_dtype
        )
        handle.create_dataset("success", data=np.asarray([False]))
        branch = handle.create_group("branches/candidate_000")
        poses = np.zeros((4, 1, 7), dtype=np.float32)
        poses[:, 0, 3] = 1.0
        poses[:, 0, 0] = [0.0, 0.02, 0.06, 0.10]
        branch.create_dataset("object_poses", data=poses)
        branch.create_dataset("query_steps", data=np.asarray([0], dtype=np.int32))
        branch.create_dataset("query_post_steps", data=np.asarray([2], dtype=np.int32))
        branch.create_dataset(
            "query_actions", data=np.ones((1, 25, 14), dtype=np.float32)
        )
        mask = np.zeros((1, 25), dtype=bool)
        mask[:, :2] = True
        branch.create_dataset("query_action_mask", data=mask)
    descriptor = GroupDescriptor(
        source="openvla_schema5",
        body="piper",
        policy="openvla",
        task="move_can_pot",
        seed=1,
        path=path,
    )
    calibration = {
        "moving": "can",
        "anchor": "",
        "centers": [[0.10, 0.0, 0.0]],
        "offset": [0.0, 0.0, 0.0],
        "delta_move": 0.03,
        "delta_z": 0.02,
        "tau_d": 0.01,
        "tau_motion": 0.005,
        "stationary_steps": 2,
    }
    rows = load_schema5_rows(descriptor, calibration)
    assert len(rows) == 1
    assert rows[0]["state"].shape == (27,)
    assert rows[0]["actions"].shape == (25, 14)
    assert rows[0]["action_schema_id"] == 2
    assert rows[0]["action_available"] == 1
    assert rows[0]["object_delta"].shape == (6,)


def test_forbidden_namespace_is_rejected_before_content_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "Fresh_dataset"
    forbidden.mkdir()
    binding = _make_binding(tmp_path / "allowed")
    binding = dataclasses.replace(binding, stage1_root=forbidden)
    with pytest.raises(ValueError, match="forbidden path token"):
        verify_input_bindings(binding)


def test_cpu_synthetic_smoke_and_cli() -> None:
    result = run_synthetic_smoke()
    assert result["status"] == "synthetic_smoke_passed"
    assert result["members"] == 5
    assert result["missing_action_effect_is_zero"] is True
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "train_multibody_canonical_event_world_model.py"),
            "--mode",
            "synthetic-smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SYNTHETIC_SMOKE=" in completed.stdout
    assert "synthetic_smoke_passed" in completed.stdout
