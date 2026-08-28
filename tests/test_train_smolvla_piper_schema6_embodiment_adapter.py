from __future__ import annotations

import copy
import json
import math
import sys
import types
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import initialize_smolvla_schema5_native_event_core as initializer
import train_openvla_etsf_counterfactual as source_training
import train_smolvla_piper_schema6_embodiment_adapter as adapter_trainer
from openvla_etsf_event_world_model import ActionConditionedEventWorldModel, EventWorldModelConfig
from train_smolvla_piper_schema6_embodiment_adapter import (
    ACTION_DIM,
    CAUSAL_HISTORY_MAX_STEPS,
    CanonicalTeacher,
    DEVELOPMENT300_V3_PROFILE,
    HISTORICAL_V2_PROFILE,
    MANIFEST_FORMAT,
    STATE_DIM,
    AdapterContractError,
    DetachedConditionalRecoveryAdapter,
    IdentityLowRankDiagonalActionAdapter,
    LogicalGroupBatchSampler,
    LogicalGroupEqualizedSampler,
    ResidualLowRankStateAdapter,
    SmolVLAPiperAdapter,
    SupportGate,
    adapter_checkpoint_selection_key,
    canonical_sha256,
    collate,
    collate_ranking_groups,
    conditional_recovery_group_support,
    conditional_recovery_loss,
    compute_loss,
    configure_determinism,
    export_internal_validation_artifacts,
    file_sha256,
    final_branch_success_targets,
    derive_conditional_recovery_targets,
    freeze_group_split,
    group_weighted_ranking_loss,
    paired_group_bootstrap_interval,
    reconstruct_pose_predicates,
    read_train_and_internal_validation_groups,
    scan_manifest,
    split_supervision_support,
    validate_group_count_gate,
    validate_production_source_rank_config,
    validate_supervision_support,
    validate_source_checkpoint,
    schema6_causal_history_at_query,
    schema6_causal_history_application_contract,
)


def dual_reserved_checkpoint(tmp_path: Path) -> dict:
    digest = "1" * 64
    checkpoint = initializer._build_payload(
        event_spec=tmp_path / "event.json",
        event_spec_sha256=digest,
        source_manifest=tmp_path / "manifest.json",
        source_manifest_sha256=digest,
        source_split=tmp_path / "split.json",
        source_split_sha256=digest,
        modeling_sha256=digest,
        bridge_sha256=digest,
        initialization_seed=17,
    )
    config = EventWorldModelConfig.from_dict(checkpoint["config"])
    model = ActionConditionedEventWorldModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    # Real source-only training may update source identity rows while the cold
    # source_identity_rows receipt remains an initialization record.
    with torch.no_grad():
        model.action_encoder.body_embedding.weight[0].add_(0.125)
        model.action_encoder.policy_embedding.weight[0].sub_(0.25)
    rows = source_training.validate_reserved_target_rows(checkpoint, config)
    proof = source_training.reserved_rows_source_only_proof(
        model,
        rows,
        source_training_steps=3,
        source_training_groups=2,
        input_pretrained_checkpoint_sha256="2" * 64,
        action_normalization=None,
    )
    checkpoint["model"] = model.state_dict()
    checkpoint["contract"]["object_names"] = ["can"]
    checkpoint["contract"]["causal_history_contract"] = (
        source_training.causal_history_contract()
    )
    checkpoint["reserved_target_rows_source_only_proof"] = proof
    checkpoint["contract"]["reserved_target_rows_source_only_proof"] = proof
    return checkpoint


def test_dual_proof_selects_smolvla_zero_never_reserved_openvla(tmp_path: Path) -> None:
    checkpoint = dual_reserved_checkpoint(tmp_path)
    audit = validate_source_checkpoint(checkpoint)
    assert audit["target_body_row"] == 1
    assert audit["policy_row"] == 0
    assert audit["reserved_openvla_policy_row"] == 1
    tampered = copy.deepcopy(checkpoint)
    tampered["contract"]["policy_to_id"] = {"smolvla": 1, "__reserved__openvla": 0}
    with pytest.raises(AdapterContractError, match="policy registry"):
        validate_source_checkpoint(tampered)
    missing_history = copy.deepcopy(checkpoint)
    del missing_history["contract"]["causal_history_contract"]
    with pytest.raises(AdapterContractError, match="causal hidden-history"):
        validate_source_checkpoint(missing_history)


def test_adapters_are_exact_identity_and_policy_rows_stay_bit_exact() -> None:
    torch.manual_seed(3)
    state_adapter = ResidualLowRankStateAdapter(3)
    action_adapter = IdentityLowRankDiagonalActionAdapter(2)
    state = torch.randn(2, STATE_DIM)
    actions = torch.randn(2, 4, ACTION_DIM)
    assert torch.equal(state_adapter(state), state)
    assert torch.equal(action_adapter(actions), actions)

    config = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        semantic_dim=96,
        action_hidden_dim=16,
        transition_hidden_dim=24,
        clock_hidden_dim=8,
        object_delta_dim=6,
        num_bodies=2,
        num_policies=2,
        structured_events=True,
        dropout=0,
    )
    model = SmolVLAPiperAdapter(ActionConditionedEventWorldModel(config), state_rank=3, action_rank=2)
    model.train()
    assert model.training is True and model.core.training is False
    before = model.core.action_encoder.policy_embedding.weight.detach().clone()
    with torch.no_grad():
        model.core.action_encoder.policy_embedding.weight.add_(7)
        model.core.next_event_head.weight.add_(5)
        model.core.action_encoder.body_embedding.weight[1].add_(1)
    audit = model.enforce_and_verify_frozen_core()
    assert torch.equal(model.core.action_encoder.policy_embedding.weight, before)
    assert audit["all_core_tensors_except_piper_body_row_bit_exact"] is True
    assert not torch.equal(
        model.core.action_encoder.body_embedding.weight[1],
        model._source_state["action_encoder.body_embedding.weight"][1],
    )
    assert model.trainable_parameter_audit()["policy_rows_trainable"] == []


def test_schema6_causal_history_is_prefix_only_padded_and_branch_isolated() -> None:
    application = schema6_causal_history_application_contract()
    assert application["max_history_steps"] == CAUSAL_HISTORY_MAX_STEPS
    assert application["post_hidden_target_supervised"] is False
    assert application["oracle_or_fabricated_post_hidden_allowed"] is False
    assert application["cross_branch_or_group_history_allowed"] is False
    first = np.arange(12 * STATE_DIM, dtype=np.float32).reshape(12, STATE_DIM)
    second = first + np.float32(100_000)
    future_changed = first.copy()
    future_changed[6:] *= np.float32(-7)

    history, mask = schema6_causal_history_at_query(first, 5)
    changed, changed_mask = schema6_causal_history_at_query(future_changed, 5)
    other, other_mask = schema6_causal_history_at_query(second, 5)

    assert mask.tolist() == [True] * 6 + [False] * 2
    assert np.array_equal(history, changed)
    assert np.array_equal(mask, changed_mask)
    assert np.array_equal(history[:6], first[:6])
    assert np.array_equal(history[6:], np.zeros((2, STATE_DIM), dtype=np.float32))
    assert np.array_equal(other[:6], second[:6])
    assert np.array_equal(other_mask, mask)
    assert not np.array_equal(other[:6], history[:6])


def test_piper_root_history_is_t1_encoder_bit_exact_and_collate_is_strict() -> None:
    torch.manual_seed(20260828)
    config = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        semantic_dim=16,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=4,
        object_delta_dim=6,
        num_bodies=2,
        num_policies=2,
        structured_events=True,
        dropout=0,
    )
    core = ActionConditionedEventWorldModel(config).eval()
    root = np.random.default_rng(11).normal(size=(1, STATE_DIM)).astype(np.float32)
    history, mask = schema6_causal_history_at_query(root, 0)
    direct = core.encode_state(torch.from_numpy(root))
    causal = core.encode_state(
        torch.from_numpy(history[None]), torch.from_numpy(mask[None])
    )
    assert torch.equal(direct, causal)

    row = _root_row("causal-root")
    row["state"] = history
    row["history_mask"] = mask
    batch = collate([row])
    assert batch["state"].shape == (1, CAUSAL_HISTORY_MAX_STEPS, STATE_DIM)
    assert batch["history_mask"].sum().item() == 1

    malformed = copy.deepcopy(row)
    malformed["history_mask"] = np.asarray(
        [True, False, True, False, False, False, False, False]
    )
    with pytest.raises(AdapterContractError, match="prefix mask"):
        collate([malformed])


def test_manifest_scan_and_split_do_not_open_hdf_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    groups = []
    for index in range(5):
        path = tmp_path / f"group_{index}.hdf5"
        path.write_bytes(f"opaque-{index}".encode())
        groups.append(
            {
                "logical_group_id": f"piper/smolvla/task/{index}",
                "requested_seed": index,
                "resolved_seed": index + 100,
                "task": "move_can_pot",
                "body": "piper",
                "policy": "smolvla",
                "path": path.name,
                "file_sha256": file_sha256(path),
            }
        )
    manifest = {
        "format": MANIFEST_FORMAT,
        "status": "complete",
        "groups": groups,
        "fresh_inputs_used": False,
        "sealed_test_labels_disclosed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("h5py.File", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HDF opened")))
    _, descriptors = scan_manifest(manifest_path.resolve())
    split = freeze_group_split(descriptors, seed=9, validation_fraction=0.2, test_fraction=0.2)
    assert split["hdf5_files_opened_before_split_freeze"] == 0
    assert len(split["train"]) == 3 and len(split["validation"]) == len(split["test"]) == 1
    train, validation, test = map(set, (split["train"], split["validation"], split["test"]))
    assert not (train & validation or train & test or validation & test)


def _development300_authority(
    root: Path,
    *,
    receipt_profile: str | None = None,
    split_profile: str | None = None,
    receipt_counts: dict[str, int] | None = None,
    split_counts: dict[str, int] | None = None,
) -> tuple[Path, Path, dict[str, object], list[adapter_trainer.GroupDescriptor]]:
    root.mkdir()
    profile = DEVELOPMENT300_V3_PROFILE
    identities = [f"piper/smolvla/move_can_pot/{index:03d}" for index in range(300)]
    adaptation = identities[:110]
    formal = identities[110:]
    partition = {
        "format": adapter_trainer.TARGET_PARTITION_FORMAT_V3,
        "status": "frozen_from_target_seed_manifest_before_hdf_access",
        "split_profile": profile.name,
        "required_group_counts": {
            "adaptation": 110,
            "formal_target_validation": 190,
        },
        "adaptation": adaptation,
        "validation": formal,
        "evaluation": [],
        "evaluation_groups_included": 0,
        "hdf5_files_opened_before_partition_freeze": 0,
        "labels_read": False,
    }
    partition["partition_sha256"] = canonical_sha256(partition)
    partition_path = root / "p.json"
    partition_path.write_text(json.dumps(partition), encoding="utf-8")

    split = {
        "format": adapter_trainer.EXTERNAL_SPLIT_FORMAT_V3,
        "status": "frozen_label_blind_before_hdf_access",
        "split_profile": split_profile or profile.name,
        "required_trainer_group_counts": (
            split_counts or profile.required_trainer_group_counts
        ),
        "algorithm": "development300_label_blind_frozen_membership_v3",
        "seed": 20260828,
        "train": adaptation[:80],
        "validation": adaptation[80:],
        "test": formal,
        "source_partition_sha256": partition["partition_sha256"],
        "target_validation_used_for_training_or_internal_validation": False,
        "evaluation_groups_included": 0,
        "hdf5_files_opened_before_split_freeze": 0,
        "labels_read": False,
    }
    split["split_sha256"] = canonical_sha256(split)
    split_path = root / "s.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")

    manifest = {
        "format": MANIFEST_FORMAT,
        "status": "complete",
        "groups": [],
        "fresh_inputs_used": False,
        "sealed_test_labels_disclosed": False,
        "expected_external_split_sha256": split["split_sha256"],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = root / "m.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    trainer_path = Path(adapter_trainer.__file__).resolve()
    expected = {
        "format": adapter_trainer.EXPECTED_MANIFEST_SPLIT_FORMAT_V3,
        "status": "complete_external_manifest_and_split_expectations",
        "split_profile": receipt_profile or profile.name,
        "trainer_compatible_manifest": {
            "path": str(manifest_path.resolve()),
            "file_sha256": file_sha256(manifest_path),
            "logical_sha256": manifest["manifest_sha256"],
        },
        "target_partition": {
            "path": str(partition_path.resolve()),
            "file_sha256": file_sha256(partition_path),
            "logical_sha256": partition["partition_sha256"],
        },
        "external_split": {
            "path": str(split_path.resolve()),
            "file_sha256": file_sha256(split_path),
            "logical_sha256": split["split_sha256"],
        },
        "bound_trainer_implementation": {
            "path": str(trainer_path),
            "file_sha256": file_sha256(trainer_path),
        },
        "required_trainer_group_counts": (
            receipt_counts or profile.required_trainer_group_counts
        ),
        "direct_bound_trainer_execution_authorized": True,
        "hdf5_content_files_opened": 0,
        "labels_read": False,
    }
    expected["expected_receipt_sha256"] = canonical_sha256(expected)
    expected_path = root / "r.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    descriptors = [
        adapter_trainer.GroupDescriptor(
            logical_group_id=logical,
            requested_seed=index,
            resolved_seed=10_000 + index,
            task="move_can_pot",
            body="piper",
            policy="smolvla",
            path=root / f"opaque_{index:03d}.hdf5",
            file_sha256="a" * 64,
        )
        for index, logical in enumerate(identities)
    ]
    return expected_path, manifest_path, manifest, descriptors


def test_development300_v3_receipt_selects_exact_profile_without_opening_sealed_hdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path, manifest_path, manifest, descriptors = _development300_authority(
        tmp_path / "authority"
    )
    original_open = Path.open

    def guarded_open(self: Path, *args: object, **kwargs: object):
        if self.suffix.casefold() in {".h5", ".hdf", ".hdf5"}:
            raise AssertionError("sealed HDF was opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    split, audit = adapter_trainer.validate_external_split_authority(
        expected_receipt_path=receipt_path,
        expected_receipt_file_sha256=file_sha256(receipt_path),
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        descriptors=descriptors,
    )
    assert (len(split["train"]), len(split["validation"]), len(split["test"])) == (
        80,
        30,
        190,
    )
    assert audit["split_profile"] == "development300_v3"
    assert audit["required_trainer_group_counts"] == {
        "train": 80,
        "validation": 30,
        "test": 190,
    }
    assert audit["sealed_test_group_count"] == 190
    assert audit["formal_target_validation_hdf5_files_opened_before_five_adapters_frozen"] == 0
    assert audit["formal_target_validation_labels_opened_before_five_adapters_frozen"] == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"receipt_profile": "historical_v2"}, "receipt scope changed"),
        (
            {"receipt_counts": {"train": 80, "validation": 30, "test": 189}},
            "receipt scope changed",
        ),
        ({"split_profile": "historical_v2"}, "partition/split scope changed"),
        (
            {"split_counts": {"train": 80, "validation": 30, "test": 189}},
            "partition/split scope changed",
        ),
    ],
)
def test_development300_profile_or_count_tamper_fails_closed(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    receipt_path, manifest_path, manifest, descriptors = _development300_authority(
        tmp_path / "authority", **overrides
    )
    with pytest.raises(AdapterContractError, match=message):
        adapter_trainer.validate_external_split_authority(
            expected_receipt_path=receipt_path,
            expected_receipt_file_sha256=file_sha256(receipt_path),
            manifest_path=manifest_path.resolve(),
            manifest=manifest,
            descriptors=descriptors,
        )


def test_development300_training_access_boundary_never_dereferences_190_sealed_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, descriptors = _development300_authority(tmp_path / "authority")
    split = {
        "train": [item.logical_group_id for item in descriptors[:80]],
        "validation": [item.logical_group_id for item in descriptors[80:110]],
        "test": [item.logical_group_id for item in descriptors[110:]],
    }
    sealed = set(split["test"])
    opened: list[str] = []

    def fake_read_group(descriptor: adapter_trainer.GroupDescriptor, *args: object, **kwargs: object):
        if descriptor.logical_group_id in sealed:
            raise AssertionError("formal target-validation group was dereferenced")
        opened.append(descriptor.logical_group_id)
        return [_root_row(descriptor.logical_group_id)], {
            "logical_group_id": descriptor.logical_group_id,
            "candidates": [],
        }

    monkeypatch.setattr(adapter_trainer, "_read_group", fake_read_group)
    train_rows, validation_rows, train_pairs, validation_pairs = (
        read_train_and_internal_validation_groups(
            split=split,
            descriptors=descriptors,
            calibration_by_task={"move_can_pot": {}},
            object_delta_dim=6,
            object_names=["can"],
            include_canonical_state=False,
        )
    )
    assert len(opened) == 110
    assert not set(opened) & sealed
    assert (len(train_rows), len(validation_rows)) == (80, 30)
    assert (len(train_pairs), len(validation_pairs)) == (80, 30)


def test_sensitive_paths_fail_closed(tmp_path: Path) -> None:
    forbidden = tmp_path / "Fresh" / "manifest.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}")
    with pytest.raises(AdapterContractError, match="Fresh/confirmation"):
        scan_manifest(forbidden.resolve())


def test_optional_teacher_fails_closed_on_incomplete_contract() -> None:
    with pytest.raises(AdapterContractError, match="teacher contract"):
        CanonicalTeacher({}, event_spec_sha256="1" * 64)


def _root_row(group: str) -> dict[str, object]:
    state = np.zeros((CAUSAL_HISTORY_MAX_STEPS, STATE_DIM), dtype=np.float32)
    history_mask = np.zeros(CAUSAL_HISTORY_MAX_STEPS, dtype=bool)
    history_mask[0] = True
    return {
        "state": state,
        "history_mask": history_mask,
        "actions": np.zeros((2, ACTION_DIM), dtype=np.float32),
        "proprio": np.zeros(ACTION_DIM, dtype=np.float32),
        "current_predicates": np.zeros(5, dtype=np.float32),
        "duration": np.float32(1),
        "success": np.float32(0),
        "regress": False,
        "recovery": np.float32(0),
        "recovery_observed": False,
        "object_delta": np.zeros(6, dtype=np.float32),
        "current_event_id": 0,
        "post_event_id": 0,
        "next_event_id": 1,
        "action_mask": np.asarray([True, False]),
        "duration_observed": True,
        "object_mask": np.ones(6, dtype=bool),
        "logical_group_id": group,
        "causal_branch_id": f"synthetic/{group}/candidate_000",
        "causal_query_index": 0,
    }


def test_pose_predicates_are_reconstructed_not_inferred_from_event_ordinal() -> None:
    poses = np.zeros((2, 2, 7), dtype=np.float32)
    poses[:, :, 3] = 1
    poses[1, 0, 2] = 0.2
    calibration = {
        "moving": "can",
        "anchor": "",
        "centers": [[10.0, 10.0, 10.0]],
        "delta_move": 1.0,
        "delta_z": 0.1,
        "tau_d": 0.01,
        "tau_motion": 0.001,
        "stationary_steps": 2,
    }
    predicates = reconstruct_pose_predicates(
        poses, ["can", "pot"], False, calibration, np.asarray([0, 1])
    )
    assert predicates[1].tolist() == [0, 1, 0, 0, 0]
    with pytest.raises(AdapterContractError, match="pose reconstruction"):
        reconstruct_pose_predicates(
            poses, ["can", "pot"], False, calibration, np.asarray([0, 2])
        )


def test_dense_success_target_is_eventual_branch_outcome_not_terminal_pulse() -> None:
    success = final_branch_success_targets(
        np.asarray([False, False, False, True]), final_success=True, steps=3
    )
    assert success.tolist() == [1.0, 1.0, 1.0]
    failure = final_branch_success_targets(
        np.asarray([False, False, False]), final_success=False, steps=2
    )
    assert failure.tolist() == [0.0, 0.0]
    with pytest.raises(AdapterContractError, match="monotone diagnostic"):
        final_branch_success_targets(
            np.asarray([False, True, False]), final_success=False, steps=2
        )


def test_conditional_recovery_targets_require_persistent_regress_and_observation() -> None:
    recovered = derive_conditional_recovery_targets(
        np.asarray([3, 2, 2, 2, 3, 3, 3]), right_censored=False
    )
    assert recovered["regress"].tolist() == [True, False, False, False, False, False]
    assert recovered["recovery"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert recovered["recovery_observed"].tolist() == [True, False, False, False, False, False]

    flicker = derive_conditional_recovery_targets(
        np.asarray([3, 2, 3, 3]), right_censored=False
    )
    assert not bool(flicker["regress"].any())

    censored = derive_conditional_recovery_targets(
        np.asarray([3, 2, 2, 2]), right_censored=True
    )
    assert bool(censored["regress"][0])
    assert not bool(censored["recovery_observed"][0])


def test_conditional_recovery_support_requires_independent_positive_and_negative_groups() -> None:
    rows: list[dict[str, object]] = []
    for label in (1.0, 0.0):
        for index in range(10):
            row = _root_row(f"recovery-{int(label)}-{index}")
            row.update(
                regress=True,
                recovery=np.float32(label),
                recovery_observed=True,
            )
            rows.append(row)
    enabled = conditional_recovery_group_support(rows)
    assert enabled["enabled"] is True
    assert enabled["positive_independent_groups"] == 10
    assert enabled["negative_independent_groups"] == 10

    disabled = conditional_recovery_group_support(rows[1:])
    assert disabled["enabled"] is False
    assert disabled["status"] == "disabled_insufficient_independent_group_support"


def test_conditional_recovery_head_stops_shared_transition_gradient() -> None:
    adapter = DetachedConditionalRecoveryAdapter(4)
    transition = torch.randn(6, 4, requires_grad=True)
    batch = {
        "regress": torch.ones(6, dtype=torch.bool),
        "recovery_observed": torch.ones(6, dtype=torch.bool),
        "recovery": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32),
    }
    loss, logits = conditional_recovery_loss(
        adapter, transition, batch, enabled=True
    )
    assert logits.shape == (6,)
    loss.backward()
    assert transition.grad is None
    assert adapter.head.weight.grad is not None
    assert adapter.parameter_audit()["shared_transition_stop_gradient"] is True
    with pytest.raises(AdapterContractError, match="final branch success"):
        final_branch_success_targets(
            np.asarray([False, False]), final_success=True, steps=1
        )


def test_censored_placeholder_never_supervises_next_reached_event() -> None:
    generator = torch.Generator().manual_seed(19)
    output = {
        "next_event_logits": torch.randn(2, 5, generator=generator),
        "next_reached_event_logits": torch.randn(2, 5, generator=generator),
        "duration_selected_log_mean": torch.randn(2, generator=generator),
        "duration_selected_log_scale": torch.zeros(2),
        "success_logit": torch.randn(2, generator=generator),
        "object_delta_mean": torch.randn(2, 6, generator=generator),
        "object_delta_log_scale": torch.zeros(2, 6),
        "semantic": torch.randn(2, 96, generator=generator),
    }
    batch = {
        "post_event_id": torch.tensor([1, 2]),
        "next_event_id": torch.tensor([3, 0]),
        "duration": torch.tensor([2.0, 5.0]),
        "duration_observed": torch.tensor([True, False]),
        "success": torch.tensor([1.0, 0.0]),
        "object_delta": torch.zeros(2, 6),
        "object_mask": torch.ones(2, 6, dtype=torch.bool),
    }
    _, first = compute_loss(
        output, batch, object_mean=torch.zeros(6), object_std=torch.ones(6)
    )
    changed = {**batch, "next_event_id": torch.tensor([3, 4])}
    _, second = compute_loss(
        output, changed, object_mean=torch.zeros(6), object_std=torch.ones(6)
    )
    assert torch.equal(first["next_event"], second["next_event"])
    expected = torch.nn.functional.cross_entropy(
        output["next_reached_event_logits"][:1], torch.tensor([3])
    )
    assert torch.allclose(first["next_event"], expected)


def test_group_weighted_ranking_uses_final_outcome_and_baseline_anchor() -> None:
    groups = [
        {
            "logical_group_id": "g0",
            "candidates": [
                {"original_candidate_index": 0, "is_baseline": True, "final_success": 0, "root_row": _root_row("g0")},
                {"original_candidate_index": 2, "is_baseline": False, "final_success": 1, "root_row": _root_row("g0")},
            ],
        },
        {
            "logical_group_id": "g1",
            "candidates": [
                {"original_candidate_index": 1, "is_baseline": True, "final_success": 1, "root_row": _root_row("g1")},
                {"original_candidate_index": 3, "is_baseline": False, "final_success": 0, "root_row": _root_row("g1")},
            ],
        },
    ]
    batch = collate_ranking_groups(groups)
    good_pair, good_list, audit = group_weighted_ranking_loss(
        torch.tensor([-2.0, 2.0, 2.0, -2.0]), batch
    )
    bad_pair, bad_list, _ = group_weighted_ranking_loss(
        torch.tensor([2.0, -2.0, -2.0, 2.0]), batch
    )
    assert good_pair < bad_pair
    assert good_list < bad_list
    assert audit == {
        "groups": 2,
        "discordant_groups": 2,
        "pairwise_informative_groups": 2,
    }


def _candidate(
    group: str,
    index: int,
    *,
    baseline: bool,
    outcome: int,
    effect: float,
    base_logit: float,
) -> dict[str, object]:
    row = _root_row(group)
    actions = np.asarray(row["actions"]).copy()
    actions[0, 0] = effect
    actions[0, 1] = base_logit
    row["actions"] = actions
    return {
        "original_candidate_index": index,
        "is_baseline": baseline,
        "final_success": outcome,
        "root_row": row,
    }


def _group_rank_adapter() -> SmolVLAPiperAdapter:
    config = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=8,
        object_delta_dim=6,
        num_bodies=2,
        num_policies=2,
        structured_events=True,
        action_rank_residual=True,
        action_rank_success_only=False,
        dropout=0,
    )
    synthetic_source = {
        "duration_scale": 25.0,
        "contract": {
            "action_rank_optimization": {
                "freeze_factual_core": False,
                "trainable_parameter_names": [
                    "semantic.bridge.0.weight",
                    "next_event_head.weight",
                    "success_head.weight",
                    "clock_cell.candidate.weight",
                    "action_rank_head.0.weight",
                ],
            }
        },
    }
    rank_contract = adapter_trainer.source_rank_score_contract(
        synthetic_source,
        config,
        source_checkpoint_file_sha256="a" * 64,
    )
    model = SmolVLAPiperAdapter(
        ActionConditionedEventWorldModel(config),
        state_rank=2,
        action_rank=2,
        source_rank_contract=rank_contract,
    )

    def relative_rank(
        self: ActionConditionedEventWorldModel,
        semantic: torch.Tensor,
        action_effect: torch.Tensor,
        baseline_action_effect: torch.Tensor,
    ) -> torch.Tensor:
        assert semantic.shape == action_effect.shape == baseline_action_effect.shape
        return action_effect[:, 0] - baseline_action_effect[:, 0]

    def controlled_forward(
        self: SmolVLAPiperAdapter, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        signal = batch["actions"][:, 0, 0]
        action_effect = signal.new_zeros((len(signal), config.semantic_dim))
        action_effect[:, 0] = signal
        semantic = signal.new_zeros((len(signal), config.semantic_dim))
        return {
            "success_logit": batch["actions"][:, 0, 1],
            "semantic": semantic,
            "action_effect": action_effect,
            "next_event_logits": signal.new_zeros((len(signal), config.num_events)),
            "next_reached_event_logits": signal.new_zeros(
                (len(signal), config.num_events)
            ),
            "duration_selected_log_mean": signal.new_zeros(len(signal)),
            "duration_selected_log_scale": signal.new_zeros(len(signal)),
            "object_delta_mean": signal.new_zeros(
                (len(signal), config.object_delta_dim)
            ),
            "object_delta_log_scale": signal.new_zeros(
                (len(signal), config.object_delta_dim)
            ),
            "transition": semantic,
        }

    model.core.relative_action_rank_logit = types.MethodType(relative_rank, model.core)
    model.forward = types.MethodType(controlled_forward, model)
    return model


def _scores_by_identity(
    model: SmolVLAPiperAdapter, groups: list[dict[str, object]]
) -> dict[tuple[str, int], tuple[float, float, float]]:
    batch = collate_ranking_groups(groups)
    output = model.predict_grouped_candidates(batch)
    return {
        (logical_id, int(candidate_index)): (
            float(output["base_success_logit"][row]),
            float(output["action_rank_residual"][row]),
            float(output["source_contract_rank_score"][row]),
        )
        for row, (logical_id, candidate_index) in enumerate(
            zip(
                batch["ranking_logical_group_id"],
                batch["ranking_candidate_index"].tolist(),
            )
        )
    }


def test_grouped_candidate_prediction_is_permutation_consistent_and_group_local() -> None:
    model = _group_rank_adapter()
    assert model._source_rank_contract["source_rank_numeric_contract"] == (
        adapter_trainer.SOURCE_RANK_NUMERIC_CONTRACT
    )
    group0 = {
        "logical_group_id": "g-rank-0",
        "candidates": [
            _candidate("g-rank-0", 0, baseline=True, outcome=0, effect=10, base_logit=1),
            _candidate("g-rank-0", 2, baseline=False, outcome=1, effect=12, base_logit=0),
        ],
    }
    group1 = {
        "logical_group_id": "g-rank-1",
        "candidates": [
            _candidate("g-rank-1", 1, baseline=True, outcome=0, effect=-5, base_logit=0),
            _candidate("g-rank-1", 3, baseline=False, outcome=1, effect=-3, base_logit=0),
        ],
    }
    canonical = _scores_by_identity(model, [group0, group1])
    permuted = _scores_by_identity(
        model,
        [
            {**group1, "candidates": list(reversed(group1["candidates"]))},
            {**group0, "candidates": list(reversed(group0["candidates"]))},
        ],
    )
    assert canonical == permuted
    assert canonical[("g-rank-0", 0)][1] == pytest.approx(0)
    assert canonical[("g-rank-0", 2)][1] == pytest.approx(2)
    assert canonical[("g-rank-1", 1)][1] == pytest.approx(0)
    # This must be relative to -5 in g-rank-1, never to g-rank-0's baseline 10.
    assert canonical[("g-rank-1", 3)][1] == pytest.approx(2)
    # The base success head prefers candidate 0, but the exact Source composite
    # rank score plus residual flips candidate selection without pretending to
    # be a calibrated success logit.
    assert canonical[("g-rank-0", 0)][0] > canonical[("g-rank-0", 2)][0]
    assert canonical[("g-rank-0", 0)][2] < canonical[("g-rank-0", 2)][2]
    assert canonical[("g-rank-0", 0)][2] == pytest.approx(1.125)
    assert canonical[("g-rank-0", 2)][2] == pytest.approx(2.125)
    assert model.enforce_and_verify_frozen_core()[
        "all_core_tensors_except_piper_body_row_bit_exact"
    ] is True


def test_grouped_candidate_rank_algebra_is_bit_exact_float32_and_rejects_float64() -> None:
    model = _group_rank_adapter()
    group = {
        "logical_group_id": "g-rank-numeric",
        "candidates": [
            _candidate(
                "g-rank-numeric", 0, baseline=True, outcome=0,
                effect=0.1, base_logit=0.1,
            ),
            _candidate(
                "g-rank-numeric", 1, baseline=False, outcome=1,
                effect=0.3, base_logit=0.2,
            ),
        ],
    }
    batch = collate_ranking_groups([group])
    output = model.predict_grouped_candidates(batch)
    base = output["source_contract_base_rank_score"].detach().numpy()
    residual = output["action_rank_residual"].detach().numpy()
    composite = output["source_contract_rank_score"].detach().numpy()
    temperature = np.float32(
        model._source_rank_contract["success_temperature"]
    )
    assert base.dtype == residual.dtype == composite.dtype == np.float32
    assert np.array_equal(composite, base + residual / temperature)

    float64_batch = {
        key: (value.double() if isinstance(value, torch.Tensor) and value.is_floating_point() else value)
        for key, value in batch.items()
    }
    with pytest.raises(AdapterContractError, match="native IEEE float32"):
        model.predict_grouped_candidates(float64_batch)


def test_validation_selection_uses_source_composite_rank_not_base_success() -> None:
    model = _group_rank_adapter()
    group = {
        "logical_group_id": "g-validation-rank",
        "candidates": [
            _candidate(
                "g-validation-rank", 0, baseline=True, outcome=0,
                effect=10, base_logit=1,
            ),
            _candidate(
                "g-validation-rank", 2, baseline=False, outcome=1,
                effect=12, base_logit=0,
            ),
        ],
    }
    dense_batch = collate([_root_row("g-validation-rank")])
    metrics = adapter_trainer.validation_metrics(
        model,
        DetachedConditionalRecoveryAdapter(8),
        [dense_batch],
        [group],
        torch.device("cpu"),
        torch.zeros(6),
        torch.ones(6),
        recovery_enabled=False,
        pairwise_weight=1.0,
        listwise_weight=1.0,
    )
    assert metrics["model_success_rate"] == pytest.approx(1.0)
    assert metrics["paired_success_gain"] == pytest.approx(1.0)
    assert metrics["paired_details"][0]["selected_original_candidate_index"] == 2
    assert metrics["candidate_score_is_success_logit"] is False
    assert metrics["candidate_score_is_success_probability"] is False
    scores = metrics["paired_details"][0]["candidate_scores"]
    assert scores[0]["base_success_logit"] > scores[1]["base_success_logit"]
    assert scores[0]["source_contract_rank_score"] < scores[1]["source_contract_rank_score"]


def test_grouped_candidate_prediction_rejects_bad_anchor_group_and_rank_head() -> None:
    model = _group_rank_adapter()
    candidates = [
        _candidate("g-strict", 1, baseline=True, outcome=0, effect=0, base_logit=0),
        _candidate("g-strict", 3, baseline=False, outcome=1, effect=1, base_logit=0),
    ]
    batch = collate_ranking_groups(
        [{"logical_group_id": "g-strict", "candidates": candidates}]
    )
    crossed = dict(batch)
    crossed["logical_group_id"] = ["g-strict", "other-group"]
    with pytest.raises(AdapterContractError, match="logical ids"):
        model.predict_grouped_candidates(crossed)

    bad_anchor = [
        _candidate("g-anchor", 2, baseline=True, outcome=0, effect=0, base_logit=0),
        _candidate("g-anchor", 1, baseline=False, outcome=1, effect=1, base_logit=0),
    ]
    with pytest.raises(AdapterContractError, match="lowest legal"):
        collate_ranking_groups(
            [{"logical_group_id": "g-anchor", "candidates": bad_anchor}]
        )

    model.core.action_rank_head = None
    with pytest.raises(AdapterContractError, match="Source action-rank residual head"):
        model.predict_grouped_candidates(batch)

    disabled = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_rank_residual=False,
        action_rank_success_only=False,
    )
    with pytest.raises(AdapterContractError, match="action_rank_residual=true"):
        validate_production_source_rank_config(disabled)
    rank_only = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_rank_residual=True,
        action_rank_success_only=True,
    )
    with pytest.raises(AdapterContractError, match="action_rank_success_only=false"):
        validate_production_source_rank_config(rank_only)

    tampered_contract = dict(model._source_rank_contract)
    tampered_contract["duration_scale"] = True
    tampered_contract["contract_sha256"] = canonical_sha256(
        {key: value for key, value in tampered_contract.items() if key != "contract_sha256"}
    )
    model._source_rank_contract = tampered_contract
    model.core.action_rank_head = torch.nn.Linear(16, 1)
    with pytest.raises(AdapterContractError, match="rank-score contract changed"):
        model.predict_grouped_candidates(batch)


def test_paired_group_bootstrap_and_checkpoint_selection_are_preregistered() -> None:
    differences = [1] * 12 + [0] * 12 + [-1] * 6
    first = paired_group_bootstrap_interval(differences)
    second = paired_group_bootstrap_interval(differences)
    assert first == second
    assert first["group_count"] == 30
    assert first["resampling_unit"] == "logical_group"
    assert first["within_group_candidate_pairs_treated_as_independent"] is False
    assert first["point_gain"] == pytest.approx(sum(differences) / 30)

    higher_lcb = {
        "paired_success_gain_lcb": 0.01,
        "paired_success_gain": 0.1,
        "model_success_rate": 0.5,
        "dense_loss": 2.0,
    }
    lower_lcb_higher_point = {
        "paired_success_gain_lcb": 0.0,
        "paired_success_gain": 0.5,
        "model_success_rate": 0.9,
        "dense_loss": 0.1,
    }
    assert adapter_checkpoint_selection_key(higher_lcb, 2) < adapter_checkpoint_selection_key(
        lower_lcb_higher_point, 1
    )


def test_dense_sampler_equalizes_logical_groups_and_cycles_long_trajectories() -> None:
    rows = [_root_row("short")] + [_root_row("long") for _ in range(5)]
    first = LogicalGroupEqualizedSampler(rows, seed=31)
    second = LogicalGroupEqualizedSampler(rows, seed=31)
    epochs_first = [list(iter(first)) for _ in range(5)]
    epochs_second = [list(iter(second)) for _ in range(5)]
    assert epochs_first == epochs_second
    assert all(len(indices) == 2 for indices in epochs_first)
    assert all(sum(index == 0 for index in indices) == 1 for indices in epochs_first)
    assert {index for indices in epochs_first for index in indices if index != 0} == set(
        range(1, 6)
    )
    audit = first.audit()
    assert audit["samples_per_epoch"] == 2
    assert audit["minimum_rows_per_group"] == 1
    assert audit["maximum_rows_per_group"] == 5
    assert audit["trajectory_length_changes_group_sampling_mass"] is False
    assert audit["repeated_success_targets_change_group_sampling_mass"] is False


def test_dense_validation_metrics_are_equal_group_not_equal_row() -> None:
    model = _group_rank_adapter()
    short = _root_row("dense-short")
    short["post_event_id"] = 0
    long_rows = []
    for _ in range(3):
        row = _root_row("dense-long")
        row["post_event_id"] = 1
        long_rows.append(row)
    rows = [short, *long_rows]
    loader = torch.utils.data.DataLoader(
        adapter_trainer.RowDataset(rows),
        batch_sampler=LogicalGroupBatchSampler(rows),
        collate_fn=collate,
    )
    group = {
        "logical_group_id": "g-validation-equal-weight",
        "candidates": [
            _candidate(
                "g-validation-equal-weight", 0, baseline=True, outcome=0,
                effect=0, base_logit=0,
            ),
            _candidate(
                "g-validation-equal-weight", 1, baseline=False, outcome=1,
                effect=1, base_logit=0,
            ),
        ],
    }
    metrics = adapter_trainer.validation_metrics(
        model,
        DetachedConditionalRecoveryAdapter(8),
        loader,
        [group],
        torch.device("cpu"),
        torch.zeros(6),
        torch.ones(6),
        recovery_enabled=False,
        pairwise_weight=1.0,
        listwise_weight=1.0,
    )
    # Argmax is event 0: short group is 100% correct, long group 0% correct.
    # Equal-row aggregation would be 0.25; the preregistered group metric is 0.5.
    assert metrics["post_event_accuracy"] == pytest.approx(0.5)
    assert metrics["dense_metric_aggregation"] == "equal_logical_group"
    assert metrics["dense_validation_logical_groups"] == 2


def test_conditional_recovery_validation_is_equal_group_not_equal_row() -> None:
    model = _group_rank_adapter()
    short = _root_row("recovery-short")
    short.update(regress=True, recovery=np.float32(1), recovery_observed=True)
    long_rows = []
    for _ in range(3):
        row = _root_row("recovery-long")
        row.update(regress=True, recovery=np.float32(0), recovery_observed=True)
        long_rows.append(row)
    rows = [short, *long_rows]
    loader = torch.utils.data.DataLoader(
        adapter_trainer.RowDataset(rows),
        batch_sampler=LogicalGroupBatchSampler(rows),
        collate_fn=collate,
    )
    recovery = DetachedConditionalRecoveryAdapter(8)
    with torch.no_grad():
        recovery.head.weight.zero_()
        recovery.head.bias.fill_(math.log(4.0))  # p=0.8
    metrics = adapter_trainer.evaluate_conditional_recovery_adapter(
        model=model,
        recovery_adapter=recovery,
        loader=loader,
        device=torch.device("cpu"),
    )
    expected = (-math.log(0.8) - math.log(0.2)) / 2
    assert metrics["binary_nll"] == pytest.approx(expected)
    assert metrics["metric_aggregation"] == (
        "equal_logical_group_with_observed_recovery"
    )


def test_support_gate_is_parameterized_and_defaults_are_formal() -> None:
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    with pytest.raises(AdapterContractError, match="group support gate"):
        validate_group_count_gate(split, SupportGate())
    low_gate = SupportGate(
            min_train_groups=1,
            min_validation_groups=1,
            min_test_groups=1,
            min_outcome_groups=0,
            min_discordant_groups=0,
            min_event_rows=0,
            min_duration_rows=0,
            min_object_rows=0,
            min_candidate_index_groups=0,
        )
    assert validate_group_count_gate(split, low_gate)["total"] == {
        "train": 1, "validation": 1, "test": 1
    }
    with pytest.raises(AdapterContractError, match="cannot lower support gates"):
        low_gate.require_formal_minimums()


def test_supervision_gate_requires_discordant_outcomes_and_all_events() -> None:
    rows = []
    for event in range(5):
        observed = _root_row("g")
        observed["post_event_id"] = event
        observed["next_event_id"] = event
        observed["duration_observed"] = True
        rows.append(observed)
        censored = _root_row("g")
        censored["post_event_id"] = event
        censored["next_event_id"] = (event + 1) % 5
        censored["duration_observed"] = False
        rows.append(censored)
    paired = [{
        "logical_group_id": "g",
        "candidates": [
            {"original_candidate_index": 0, "is_baseline": True, "final_success": 0, "root_row": _root_row("g")},
            {"original_candidate_index": 1, "is_baseline": False, "final_success": 1, "root_row": _root_row("g")},
            {"original_candidate_index": 2, "is_baseline": False, "final_success": 0, "root_row": _root_row("g")},
            {"original_candidate_index": 3, "is_baseline": False, "final_success": 0, "root_row": _root_row("g")},
        ],
    }]
    support = split_supervision_support(rows, paired)
    gate = SupportGate(
        min_train_groups=0,
        min_validation_groups=0,
        min_test_groups=0,
        min_outcome_groups=1,
        min_discordant_groups=1,
        min_event_rows=1,
        min_duration_rows=1,
        min_object_rows=1,
        min_candidate_index_groups=1,
    )
    validate_supervision_support(support, gate, split_name="synthetic")
    assert support["event_next_observed_rows"] == {
        "e0": 1, "e12": 1, "e3": 1, "e4": 1, "eK": 1
    }
    # e0 is the initial state, never an observed future canonical milestone.
    support["event_next_observed_rows"]["e0"] = 0
    validate_supervision_support(support, gate, split_name="synthetic")


def test_internal_validation_export_denormalizes_object_distribution_and_binds_spaces(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    config = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        semantic_dim=96,
        action_hidden_dim=16,
        transition_hidden_dim=24,
        clock_hidden_dim=8,
        object_delta_dim=6,
        num_bodies=2,
        num_policies=2,
        structured_events=True,
        dropout=0,
    )
    model = SmolVLAPiperAdapter(
        ActionConditionedEventWorldModel(config), state_rank=3, action_rank=2
    )
    rows = [_root_row(f"g-export-{index}") for index in range(20)]
    recovery_adapter = DetachedConditionalRecoveryAdapter(config.semantic_dim)
    recovery_support = conditional_recovery_group_support(rows)
    recovery_fit = {
        "status": "disabled_insufficient_independent_group_support",
        "trained": False,
        "validation_metrics": {
            "status": "disabled_independent_group_support",
            "enters_primary_utility_or_uncertainty": False,
        },
    }
    mean = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.4, -0.5])
    std = torch.tensor([2.0, 0.5, 1.5, 3.0, 0.25, 4.0])
    raw_prediction = model(collate(rows))
    receipt = export_internal_validation_artifacts(
        split_profile=HISTORICAL_V2_PROFILE,
        model=model,
        recovery_adapter=recovery_adapter,
        recovery_fit=recovery_fit,
        recovery_training_support=recovery_support,
        recovery_validation_support=recovery_support,
        rows=rows,
        device=torch.device("cpu"),
        batch_size=4,
        object_mean=mean,
        object_std=std,
        output=tmp_path,
    )
    with np.load(receipt["predictions_path"], allow_pickle=False) as archive:
        assert archive["recovery_logit"].shape == (20,)
        assert np.allclose(
            archive["object_mean"][0],
            (raw_prediction["object_delta_mean"][0] * std + mean).detach().numpy(),
        )
        assert np.allclose(
            archive["object_log_scale"][0],
            (raw_prediction["object_delta_log_scale"][0] + std.log()).detach().numpy(),
        )
    assert receipt["duration_target_transform"] == "log1p_decision_steps"
    assert receipt["split_profile"] == "historical_v2"
    assert receipt["validation_group_count"] == 20
    assert receipt["sealed_formal_target_validation_group_count"] == 50
    assert receipt["next_event_observation_mask"] == "duration_observed"
    assert receipt["success_target"].startswith("eventual_final_branch_success")
    assert receipt["object_prediction_space"] == "physical_delta_xyz_m"
    assert len(receipt["object_source_normalization_sha256"]) == 64
    assert receipt["recovery_head_trained"] is False
    assert receipt["recovery_enters_primary_utility_or_uncertainty"] is False
    assert receipt["recovery_calibration_required_before_activation"] is True


def test_training_seed_controls_adapter_initialization_and_audit_is_exact() -> None:
    configure_determinism(123)
    first = ResidualLowRankStateAdapter(3)
    configure_determinism(123)
    second = ResidualLowRankStateAdapter(3)
    assert torch.equal(first.down.weight, second.down.weight)

    config = EventWorldModelConfig(
        state_input_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        semantic_dim=96,
        action_hidden_dim=16,
        transition_hidden_dim=24,
        clock_hidden_dim=8,
        object_delta_dim=6,
        num_bodies=2,
        num_policies=2,
        structured_events=True,
        dropout=0,
    )
    model = SmolVLAPiperAdapter(ActionConditionedEventWorldModel(config), state_rank=3, action_rank=2)
    assert model.trainable_parameter_audit()["parameter_tensor_count"] == 8
    model.clock_beta.requires_grad_(False)
    with pytest.raises(AdapterContractError, match="trainable parameter set changed"):
        model.trainable_parameter_audit()


def test_manifest_seed_types_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "group.hdf5"
    path.write_bytes(b"opaque")
    group = {
        "logical_group_id": "piper/smolvla/task/0",
        "requested_seed": True,
        "resolved_seed": 1,
        "task": "move_can_pot",
        "body": "piper",
        "policy": "smolvla",
        "path": path.name,
        "file_sha256": file_sha256(path),
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "status": "complete",
        "groups": [group, {**group, "logical_group_id": "g1", "requested_seed": 2, "resolved_seed": 2}, {**group, "logical_group_id": "g2", "requested_seed": 3, "resolved_seed": 3}],
        "fresh_inputs_used": False,
        "sealed_test_labels_disclosed": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AdapterContractError, match="non-negative integer"):
        scan_manifest(manifest_path.resolve())


def test_formal_cli_has_no_internal_split_seed_or_fraction_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trainer", "--mode", "train", "--split-seed", "7",
            "--validation-fraction", "0.2", "--test-fraction", "0.2",
        ],
    )
    with pytest.raises(SystemExit):
        adapter_trainer.parse_args()
