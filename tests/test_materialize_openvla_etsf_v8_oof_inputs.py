from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from materialize_openvla_etsf_v8_oof_inputs import (  # noqa: E402
    V8_COLLECTION_SOURCE_FORMAT,
    V8_BASE_EXCLUSION_FORMAT,
    V8_HOLDOUT_INPUT_FORMAT,
    V8_OWNER_MANIFEST_FORMAT,
    _synthetic_group,
    base_exclusion_status,
    build_outer_fold_payloads,
    cpu_materializer_smoke,
    fit_outer_training_repairs,
    load_frozen_factual_context,
    logical_group_list_sha256,
    materialize_group_records,
    normalize_oof_owner_folds,
    preregister_v8_owner_manifest,
    reject_fresh_sources,
    sha256_path,
    validate_development_collection_contract,
    validate_v7_collection_trust_roots,
)
from openvla_etsf_counterfactual_oof import canonical_sha256  # noqa: E402
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_OBJECT_MODE,
    module_state_sha256,
)
from train_openvla_etsf_v8_structured_adapters import (  # noqa: E402
    V8_TRAINING_INPUT_FORMAT,
    train_v8_payload,
    structured_payload_sha256,
    validate_v8_training_payload,
)
from train_openvla_etsf_counterfactual import GroupDescriptor  # noqa: E402
from openvla_etsf_v7_development_confirmation import (  # noqa: E402
    SEED_CANDIDATE_FORMAT,
    canonical_sha256 as v7_canonical_sha256,
    make_preregistration as make_v7_preregistration,
    make_seed_manifest as make_v7_seed_manifest,
)


def _config() -> EventWorldModelConfig:
    return EventWorldModelConfig(
        state_input_dim=12,
        action_dim=4,
        proprio_dim=3,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=6,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=True,
        dropout=0.0,
    )


def _context(config: EventWorldModelConfig | None = None) -> dict:
    config = config or _config()
    torch.manual_seed(22)
    model = ActionConditionedEventWorldModel(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "model": model,
        "config": config,
        "checkpoint_path": "/synthetic/factual.pt",
        "checkpoint_sha256": "a" * 64,
        "event_spec_path": "/synthetic/event_spec.json",
        "event_spec_sha256": "b" * 64,
        "object_mean": np.zeros(config.object_delta_dim, dtype=np.float32),
        "object_std": np.ones(config.object_delta_dim, dtype=np.float32),
        "label_derivation_contract": {
            "format": "synthetic_label_contract",
            "label_derivation_sha256": "c" * 64,
        },
    }


def _groups(
    count: int,
    *,
    config: EventWorldModelConfig,
    offset: float = 0.0,
    prefix: str = "train",
) -> list:
    return [
        _synthetic_group(
            f"move_can_pot|piper|{prefix}{index:03d}",
            config=config,
            duration_offset=offset + index,
        )
        for index in range(count)
    ]


def _fold(train: list, holdout: list, fold_id: int = 0) -> dict:
    return {
        "outer_fold_id": fold_id,
        "training_groups": sorted(group.logical_key for group in train),
        "oof_holdout_groups": sorted(group.logical_key for group in holdout),
    }


def _synthetic_oof_manifest(keys: list[str]) -> dict:
    keys = sorted(keys)
    folds = []
    width = len(keys) // 5
    ordered = sorted(
        keys,
        key=lambda key: __import__("hashlib").sha256(
            f"20260827|{key}".encode("utf-8")
        ).hexdigest(),
    )
    for fold_id in range(5):
        holdout = sorted(ordered[fold_id * width : (fold_id + 1) * width])
        folds.append(
            {
                "fold_id": fold_id,
                "training_groups": sorted(set(keys) - set(holdout)),
                "oof_holdout_groups": holdout,
            }
        )
    value = {
        "format": "etsf_counterfactual_five_fold_oof_v1",
        "status": "preregistered",
        "split_seed": 20260827,
        "development_groups": keys,
        "folds": folds,
    }
    value["preregistration_sha256"] = canonical_sha256(value)
    return value


def _collection_fixture(tmp_path: Path) -> tuple[Path, list[GroupDescriptor], dict, str]:
    root = tmp_path / "v7_development"
    groups_dir = root / "groups"
    groups_dir.mkdir(parents=True)
    task, body, policy = "move_can_pot", "piper_piper_0.6", "openvla"
    event_spec_path = root / "event_spec.json"
    event_spec_path.write_text(
        json.dumps({"calibration": {"move_can_pot": {}}}), encoding="utf-8"
    )
    checkpoint_path = root / "factual.pt"
    checkpoint_path.write_bytes(b"synthetic-frozen-factual-checkpoint")
    event_sha = sha256_path(event_spec_path)
    names = [
        "deterministic",
        "sample_blend_0.250",
        "sample_blend_0.500",
        "sample_blend_0.750",
    ]
    descriptors = []
    rows = []
    source_files = []
    strings = h5py.string_dtype(encoding="utf-8")
    seeds = list(range(1001, 1251))
    for index, seed in enumerate(seeds):
        path = groups_dir / f"group_{index:03d}_seed_{seed}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = 5
            handle.attrs["task"] = task
            handle.attrs["body"] = body
            handle.attrs["seed"] = seed
            handle.attrs["requested_seed"] = seed
            handle.attrs["resolved_seed"] = seed
            handle.attrs["candidate_count"] = 4
            handle.create_dataset(
                "candidate_names",
                data=np.asarray(names, dtype=object),
                dtype=strings,
            )
        logical_key = f"{task}|{body}|{seed}"
        row = {
            "index": index,
            "seed": seed,
            "requested_seed": seed,
            "resolved_seed": seed,
            "path": path.name,
            "candidate_names": names,
            "status": "collected",
        }
        rows.append(row)
        descriptors.append(
                GroupDescriptor(
                    path=str(path.resolve()),
                    schema_version=5,
                    logical_key=logical_key,
                    seed=seed,
                    requested_seed=seed,
                    task=task,
                    body=body,
                    policy=policy,
                    metadata={"task": task, "body": body, "policy": policy},
                )
        )
        source_files.append(
            {
                "logical_key": logical_key,
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
                "schema_version": 5,
            }
        )
    common = {
        "schema_version": 5,
        "status": "complete",
        "completed": len(seeds),
        "task": task,
        "body": body,
        "policy": policy,
        "model_path": "/models/OpenVLA",
        "seed_registry": "explicit_v7_prospective_development",
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
        "event_spec_sha256": event_sha,
        "candidate_count": 4,
        "groups": rows,
    }
    candidate = {
        "format": SEED_CANDIDATE_FORMAT,
        "status": "preregistered_unresolved_label_free",
        "candidate_seed_range": {"start": 1001, "count": 250, "step": 1},
    }
    candidate["candidate_payload_sha256"] = v7_canonical_sha256(candidate)
    exclusions = {}
    for name, values in (
        ("official150", list(range(1, 151))),
        ("development150", list(range(201, 351))),
        ("fresh50", list(range(401, 451))),
    ):
        exclusions[name] = {
            "path": f"/frozen/{name}.json",
            "sha256": name[0] * 64,
            "requested_seeds": values,
            "resolved_seeds": values,
            "identity_sets_sha256": v7_canonical_sha256(
                {"requested": values, "resolved": values}
            ),
        }
    selected = [
        {"seed": seed, "requested_seed": seed, "resolved_seed": seed}
        for seed in seeds
    ]
    seed_manifest = make_v7_seed_manifest(
        selected=selected,
        audit=[
            {
                "requested_seed": seed,
                "resolved_seed": seed,
                "decision": "selected",
            }
            for seed in seeds
        ],
        sources=exclusions,
        candidate={"payload": candidate},
    )
    seed_path = root / "v7_seed_manifest.json"
    seed_path.write_text(json.dumps(seed_manifest), encoding="utf-8")
    prereg = make_v7_preregistration(
        seed_manifest=seed_manifest,
        source_contract={
            "seed_manifest": str(seed_path.resolve()),
            "seed_manifest_file_sha256": sha256_path(seed_path),
            "pretrained": str(checkpoint_path.resolve()),
            "pretrained_sha256": sha256_path(checkpoint_path),
            "event_spec": str(event_spec_path.resolve()),
            "event_spec_sha256": event_sha,
            "event_spec": str(event_spec_path.resolve()),
            "pretrained": str(checkpoint_path.resolve()),
            "pretrained_sha256": sha256_path(checkpoint_path),
        },
        task_calibration_sha256="b" * 64,
    )
    prereg_path = root / "v7_preregistration.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    common.update(
        {
            "requested_seeds": seeds,
            "resolved_seeds": seeds,
            "v7_seed_manifest": str(seed_path.resolve()),
            "v7_seed_manifest_sha256": sha256_path(seed_path),
            "v7_preregistration": str(prereg_path.resolve()),
            "v7_preregistration_sha256": prereg["preregistration_sha256"],
        }
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(common), encoding="utf-8")
    identity = {
        **common,
        "format": "etsf_event_branch_collection_identity_v1",
        "label_access_contract": "identity_only_no_success_steps_event_or_outcome_fields",
    }
    identity_path = root / "collection_identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    v7_trust_root = validate_v7_collection_trust_roots(identity)
    oof = {
        "source_contract": {
            "format": V8_COLLECTION_SOURCE_FORMAT,
            "data_root": str(root.resolve()),
            "collector_manifest": str(manifest_path.resolve()),
            "collector_manifest_sha256": sha256_path(manifest_path),
            "collection_identity": str(identity_path.resolve()),
            "collection_identity_sha256": sha256_path(identity_path),
            "development_group_files": source_files,
            "event_spec_sha256": event_sha,
            "fresh_seed_manifest": None,
            "fresh_labels_read": False,
            "v7_trust_root": v7_trust_root,
        }
    }
    return root, descriptors, oof, event_sha


def test_owner_fold_normalization_is_identity_only_and_disjoint() -> None:
    keys = [f"task|body|{index:03d}" for index in range(10)]
    manifest = _synthetic_oof_manifest(keys)
    folds = normalize_oof_owner_folds(
        manifest, keys, require_formal_size=False
    )
    assert len(folds) == 5
    owners = [key for fold in folds for key in fold["oof_holdout_groups"]]
    assert sorted(owners) == sorted(keys)
    assert len(set(owners)) == len(keys)
    for fold in folds:
        assert not set(fold["training_groups"]) & set(fold["oof_holdout_groups"])
        assert fold["training_groups_sha256"] == logical_group_list_sha256(
            fold["training_groups"]
        )
    changed = _synthetic_oof_manifest(keys)
    changed["folds"][0]["oof_holdout_groups"][0] = "changed"
    with pytest.raises(RuntimeError, match="missing or invalid"):
        normalize_oof_owner_folds(changed, keys, require_formal_size=False)
    malicious = _synthetic_oof_manifest(keys)
    first = malicious["folds"][0]["oof_holdout_groups"][0]
    second = malicious["folds"][1]["oof_holdout_groups"][0]
    malicious["folds"][0]["oof_holdout_groups"][0] = second
    malicious["folds"][1]["oof_holdout_groups"][0] = first
    for fold in malicious["folds"]:
        holdout = sorted(fold["oof_holdout_groups"])
        fold["oof_holdout_groups"] = holdout
        fold["training_groups"] = sorted(set(keys) - set(holdout))
    malicious["preregistration_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in malicious.items()
            if key != "preregistration_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="fixed SHA256 split"):
        normalize_oof_owner_folds(malicious, keys, require_formal_size=False)


def test_old100_overlap_is_explicitly_unproven() -> None:
    holdout = ["task|body|old001", "task|body|new001"]
    result = base_exclusion_status(
        checkpoint_sha256="a" * 64,
        holdout_groups=holdout,
        exclusion_contract=None,
        legacy_old100_groups=["task|body|old001"],
    )
    assert result["status"] == "unproven_development_only"
    assert result["reason"] == "legacy_old100_factual_training_overlap_not_excluded"
    assert result["legacy_old100_holdout_groups"] == ["task|body|old001"]


def test_self_hashed_base_exclusion_cannot_prove_clean_holdout() -> None:
    holdout = ["task|body|new001"]
    contract = {
        "format": V8_BASE_EXCLUSION_FORMAT,
        "base_checkpoint_sha256": "a" * 64,
        "excluded_groups": holdout,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    result = base_exclusion_status(
        checkpoint_sha256="a" * 64,
        holdout_groups=holdout,
        exclusion_contract=contract,
    )
    assert result["status"] == "unproven_development_only"
    assert result["reason"] == (
        "standalone_self_hashed_exclusion_is_not_an_authoritative_trust_root"
    )
    mismatch = dict(contract)
    mismatch["base_checkpoint_sha256"] = "b" * 64
    mismatch["contract_sha256"] = canonical_sha256(
        {key: value for key, value in mismatch.items() if key != "contract_sha256"}
    )
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        base_exclusion_status(
            checkpoint_sha256="a" * 64,
            holdout_groups=holdout,
            exclusion_contract=mismatch,
        )


def test_checkpoint_and_v7_bound_seed_contract_proves_new_d250_exclusion() -> None:
    checkpoint_sha = "a" * 64
    checkpoint_contract = {
        "source_manifest_sha256": "b" * 64,
        "event_spec_sha256": "c" * 64,
        "train_seeds": list(range(100100000, 100100100)),
        "validation_seeds": list(range(100100100, 100100125)),
        "sealed_test_seeds": list(range(100100125, 100100150)),
    }
    targets = [
        f"move_can_pot|piper_piper_0.6|{seed}"
        for seed in range(100101000, 100101250)
    ]
    trust = {
        "status": "signed_v7_seed_and_preregistration_verified",
        "frozen_factual_checkpoint_sha256": checkpoint_sha,
        "seed_manifest_payload_sha256": "d" * 64,
        "preregistration_sha256": "e" * 64,
    }
    result = base_exclusion_status(
        checkpoint_sha256=checkpoint_sha,
        holdout_groups=targets[:50],
        exclusion_contract=None,
        checkpoint_contract=checkpoint_contract,
        authorized_target_groups=targets,
        v7_trust_root=trust,
    )
    assert result["status"] == "proven"
    assert result["base_seed_count"] == 150
    assert result["authorized_target_group_count"] == 250
    assert result["legacy_old100_holdout_groups"] == []


def test_checkpoint_bound_seed_overlap_and_incomplete_contract_fail_closed() -> None:
    checkpoint_sha = "a" * 64
    targets = [
        f"move_can_pot|piper_piper_0.6|{seed}"
        for seed in range(100101000, 100101250)
    ]
    trust = {
        "status": "signed_v7_seed_and_preregistration_verified",
        "frozen_factual_checkpoint_sha256": checkpoint_sha,
        "seed_manifest_payload_sha256": "d" * 64,
        "preregistration_sha256": "e" * 64,
    }
    overlap_contract = {
        "source_manifest_sha256": "b" * 64,
        "event_spec_sha256": "c" * 64,
        "train_seeds": [100101000, *range(100100001, 100100100)],
        "validation_seeds": list(range(100100100, 100100125)),
        "sealed_test_seeds": list(range(100100125, 100100150)),
    }
    result = base_exclusion_status(
        checkpoint_sha256=checkpoint_sha,
        holdout_groups=targets[:50],
        exclusion_contract=None,
        checkpoint_contract=overlap_contract,
        authorized_target_groups=targets,
        v7_trust_root=trust,
    )
    assert result["status"] == "unproven_development_only"
    assert result["overlap_seeds"] == [100101000]

    incomplete = dict(overlap_contract)
    incomplete["train_seeds"] = incomplete["train_seeds"][:-1]
    with pytest.raises(RuntimeError, match="does not cover 150"):
        base_exclusion_status(
            checkpoint_sha256=checkpoint_sha,
            holdout_groups=targets[:50],
            exclusion_contract=None,
            checkpoint_contract=incomplete,
            authorized_target_groups=targets,
            v7_trust_root=trust,
        )


def test_outer_training_repairs_use_schema5_candidates_and_continuations() -> None:
    config = _config()
    train = _groups(2, config=config)
    repairs = fit_outer_training_repairs(
        train,
        object_mean=np.zeros(3, dtype=np.float32),
        object_std=np.ones(3, dtype=np.float32),
    )
    # Four candidates plus one continuation per logical group.
    assert repairs["training_dense_support"] == 10
    assert repairs["training_observed_duration_support"] == 8
    assert len(repairs["duration_contract"]["duration_baseline_contract_sha256"]) == 64
    assert len(
        repairs["object_fallback_contract"]["object_fallback_contract_sha256"]
    ) == 64
    assert repairs["object_fallback_contract"]["learned_object_output_authorized"] is False


def test_materialized_record_preserves_real_schema5_masks_and_frozen_outputs() -> None:
    config = _config()
    context = _context(config)
    group = _groups(1, config=config)[0]
    repairs = fit_outer_training_repairs(
        [group],
        object_mean=context["object_mean"],
        object_std=context["object_std"],
    )
    before = module_state_sha256(context["model"])
    records, audit = materialize_group_records(
        [group],
        split_role="outer_training",
        outer_fold_id=0,
        model=context["model"],
        duration_contract=repairs["duration_contract"],
        object_fallback_normalized=repairs["object_fallback_normalized"],
        object_mean=context["object_mean"],
        object_std=context["object_std"],
        device="cpu",
    )
    record = records[0]
    batch = record["batch"]
    assert len(batch["terminal_mask"]) == 5
    assert int(batch["terminal_mask"].sum()) == 4
    assert int(batch["structured_mask"].sum()) == 5
    assert int(batch["trajectory_regress"].sum()) == 3
    assert int(batch["trajectory_recovery"].sum()) == 1
    assert batch["terminal_mask"][-1].item() is False
    assert batch["structured_mask"][-1].item() is True
    assert record["factual_outputs"]["transition"].shape == (5, config.semantic_dim)
    assert record["factual_outputs"]["transition"].requires_grad is False
    assert record["factual_outputs"]["duration_selected_log_mean"].requires_grad is False
    assert record["factual_outputs"]["next_event_logits"].shape == (5, 5)
    assert record["factual_outputs"]["next_reached_event_logits"].shape == (5, 5)
    assert record["factual_outputs"]["aleatoric_uncertainty"].shape == (5,)
    assert bool(torch.isfinite(record["factual_outputs"]["aleatoric_uncertainty"]).all())
    assert record["total_uncertainty_status"].endswith("requires_ensemble_fail_closed")
    assert batch["current_event_id"].tolist() == [0, 1, 1, 0, 1]
    assert batch["next_event_id"].shape == (5,)
    assert batch["next_reached_event_id"].shape == (5,)
    assert batch["object_delta_physical"].shape == (5, 3)
    assert record["object_delta_physical"].shape == (5, 3)
    assert batch["object_pose_quality_valid"] is None
    assert batch["object_pose_quality_status"].endswith("fail_closed")
    assert audit["factual_state_bit_exact"] is True
    assert module_state_sha256(context["model"]) == before


def test_extreme_holdout_labels_cannot_change_train_fitted_contracts() -> None:
    config = _config()
    context = _context(config)
    train = _groups(4, config=config)
    first_holdout = _groups(
        1, config=config, offset=1_000.0, prefix="holdout"
    )
    second_holdout = _groups(
        1, config=config, offset=1_000_000.0, prefix="holdout"
    )
    first = build_outer_fold_payloads(
        fold=_fold(train, first_holdout),
        training_groups=train,
        holdout_groups=first_holdout,
        context=context,
        device="cpu",
    )
    second = build_outer_fold_payloads(
        fold=_fold(train, second_holdout),
        training_groups=train,
        holdout_groups=second_holdout,
        context=context,
        device="cpu",
    )
    first_provenance = first["training_payload"]["provenance"]
    second_provenance = second["training_payload"]["provenance"]
    assert (
        first_provenance["duration_baseline_contract_sha256"]
        == second_provenance["duration_baseline_contract_sha256"]
    )
    assert (
        first_provenance["object_fallback_contract_sha256"]
        == second_provenance["object_fallback_contract_sha256"]
    )
    assert (
        first_provenance["duration_laplace_scale_contract_sha256"]
        == second_provenance["duration_laplace_scale_contract_sha256"]
    )
    assert all(
        record["logical_group_key"] not in {group.logical_key for group in first_holdout}
        for record in first["training_payload"]["batches"]
    )


def test_outer_fold_payload_is_directly_accepted_by_v8_trainer() -> None:
    config = _config()
    context = _context(config)
    train = _groups(4, config=config)
    holdout = _groups(1, config=config, prefix="holdout")
    result = build_outer_fold_payloads(
        fold=_fold(train, holdout),
        training_groups=train,
        holdout_groups=holdout,
        context=context,
        legacy_old100_groups=[holdout[0].logical_key],
        device="cpu",
    )
    payload = result["training_payload"]
    config_value, records, provenance = validate_v8_training_payload(payload)
    assert config_value.transition_dim == config.semantic_dim
    assert len(records) == 4
    assert provenance["target_outer_fold_labels_used"] is False
    assert provenance["base_target_outer_fold_exclusion_status"] == (
        "unproven_development_only"
    )
    assert result["holdout_payload"]["format"] == V8_HOLDOUT_INPUT_FORMAT
    checkpoint = train_v8_payload(payload, epochs=1, device="cpu")
    assert checkpoint["strict_oof_base_exclusion_eligible"] is False
    assert checkpoint["all_steps_factual_inputs_bit_exact"] is True


def test_factual_checkpoint_loader_validates_event_spec_and_freezes_state(
    tmp_path: Path,
) -> None:
    config = _config()
    model = ActionConditionedEventWorldModel(config).eval()
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(
        json.dumps({"calibration": {"move_can_pot": {"synthetic": True}}}),
        encoding="utf-8",
    )
    contract = {
        "event_spec_sha256": sha256_path(event_spec),
        "object_names": ["can"],
        "body_to_id": {"piper": 0},
        "policy_to_id": {"openvla": 0},
        "predicate_contract": {"derivation": "derive_atomic_predicates_v1"},
    }
    checkpoint_path = tmp_path / "factual.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "contract": contract,
            "normalization": {
                "object_delta_mean": np.zeros(3, dtype=np.float32),
                "object_delta_std": np.ones(3, dtype=np.float32),
            },
        },
        checkpoint_path,
    )
    context = load_frozen_factual_context(
        checkpoint_path, event_spec, device="cpu"
    )
    assert context["checkpoint_sha256"] == sha256_path(checkpoint_path)
    assert context["event_spec_sha256"] == sha256_path(event_spec)
    assert all(not parameter.requires_grad for parameter in context["model"].parameters())
    assert context["model"].training is False
    assert len(
        context["label_derivation_contract"]["label_derivation_sha256"]
    ) == 64
    changed_event_spec = tmp_path / "changed_event_spec.json"
    changed_event_spec.write_text(
        json.dumps({"calibration": {"move_can_pot": {"synthetic": False}}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="event-spec SHA mismatch"):
        load_frozen_factual_context(
            checkpoint_path, changed_event_spec, device="cpu"
        )


def test_complete_collection_manifest_identity_and_hdf5_sha_are_bound(
    tmp_path: Path,
) -> None:
    root, descriptors, oof, event_sha = _collection_fixture(tmp_path)
    audit = validate_development_collection_contract(
        data_inputs=[root],
        descriptors=descriptors,
        event_spec_sha256=event_sha,
        oof_manifest=oof,
    )
    assert audit["status"] == "complete_schema5_signed_source_verified"
    assert audit["seed_registry"] == "explicit_v7_prospective_development"
    assert audit["policy"] == "openvla"
    assert len(audit["groups"]) == 250
    assert len(audit["groups_sha256"]) == 64
    assert audit["labels_used_for_owner_split"] is False


def test_v8_owner_preregister_entrypoint_binds_real_collection_fields(
    tmp_path: Path,
) -> None:
    root, descriptors, _, event_sha = _collection_fixture(tmp_path)
    output = tmp_path / "v8_owner_manifest.json"
    result = preregister_v8_owner_manifest(
        data_root=root,
        checkpoint_path=root / "factual.pt",
        event_spec_path=root / "event_spec.json",
        output_path=output,
    )
    assert result["format"] == V8_OWNER_MANIFEST_FORMAT
    assert result["timing_scope"] == (
        "adaptive_development_only_designed_after_v7_collection_started"
    )
    assert result["prospective_claim_for_v8"] is False
    assert result["source_contract"]["event_spec_sha256"] == event_sha
    assert result["source_contract"]["v7_trust_root"][
        "fresh_exclusion_artifact_read"
    ] is False
    folds = normalize_oof_owner_folds(
        result, [descriptor.logical_key for descriptor in descriptors]
    )
    assert len(folds) == 5
    assert output.is_file()


def test_collection_v7_seed_trust_root_mutation_fails_closed(tmp_path: Path) -> None:
    root, descriptors, oof, event_sha = _collection_fixture(tmp_path)
    seed_path = root / "v7_seed_manifest.json"
    seed_path.write_text(seed_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="seed-manifest SHA changed"):
        validate_development_collection_contract(
            data_inputs=[root],
            descriptors=descriptors,
            event_spec_sha256=event_sha,
            oof_manifest=oof,
        )


def test_collecting_or_unbound_collection_fails_closed(tmp_path: Path) -> None:
    root, descriptors, oof, event_sha = _collection_fixture(tmp_path)
    manifest_path = root / "manifest.json"
    identity_path = root / "collection_identity.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    manifest["status"] = "collecting"
    identity["status"] = "collecting"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_development_collection_contract(
            data_inputs=[root],
            descriptors=descriptors,
            event_spec_sha256=event_sha,
            oof_manifest=oof,
        )

    root2, descriptors2, oof2, event_sha2 = _collection_fixture(tmp_path / "second")
    del oof2["source_contract"]["collection_identity_sha256"]
    with pytest.raises(RuntimeError, match="SHA binding"):
        validate_development_collection_contract(
            data_inputs=[root2],
            descriptors=descriptors2,
            event_spec_sha256=event_sha2,
            oof_manifest=oof2,
        )


def test_collection_event_spec_and_hdf5_digest_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    root, descriptors, oof, event_sha = _collection_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="event-spec SHA mismatch"):
        validate_development_collection_contract(
            data_inputs=[root],
            descriptors=descriptors,
            event_spec_sha256="9" * 64,
            oof_manifest=oof,
        )
    group_path = Path(descriptors[0].path)
    with h5py.File(group_path, "a") as handle:
        handle.attrs["post_signature_mutation"] = True
    with pytest.raises(RuntimeError, match="HDF5 SHA/path mismatch"):
        validate_development_collection_contract(
            data_inputs=[root],
            descriptors=descriptors,
            event_spec_sha256=event_sha,
            oof_manifest=oof,
        )


def test_fresh_sources_are_rejected_before_scan(tmp_path: Path) -> None:
    fresh = tmp_path / "Fresh50_confirmation"
    fresh.mkdir()
    with pytest.raises(RuntimeError, match="Fresh confirmation"):
        reject_fresh_sources([fresh])


def test_cpu_materializer_smoke_is_gpu_free_and_leakage_safe() -> None:
    result = cpu_materializer_smoke(seed=31)
    assert result["status"] == "passed"
    assert result["device"] == "cpu"
    assert result["cuda_used"] is False
    assert result["factual_state_bit_exact"] is True
    assert result["target_outer_fold_labels_used"] is False
    assert result["extreme_holdout_duration_did_not_change_baseline"] is True
    assert result["learned_object_output_authorized"] is False
