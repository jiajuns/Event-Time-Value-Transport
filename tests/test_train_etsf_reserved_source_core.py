from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from prepare_etsf_transfer_source_core import expand_source_core  # noqa: E402
from train_etsf_reserved_source_core import (  # noqa: E402
    assert_reserved_row_absent,
    load_expanded_source_core,
    make_retraining_proof,
    reserved_row_is_bit_exact,
    restore_reserved_row,
    source_parameters_changed,
    validate_source_contract,
    validation_selection_score,
)


def _source_checkpoint(path: Path) -> None:
    torch.save(
        {
            "model": {
                "action_encoder.policy_embedding.weight": torch.arange(
                    8, dtype=torch.float32
                ).reshape(1, 8),
                "action_encoder.body_embedding.weight": torch.arange(
                    8, dtype=torch.float32
                ).reshape(1, 8),
                "shared.weight": torch.ones(2, 2),
            },
            "config": {
                "state_input_dim": 16,
                "action_dim": 14,
                "proprio_dim": 14,
                "object_delta_dim": 3,
                "num_policies": 1,
                "num_bodies": 1,
                "event_names": ["e0", "e12", "e3", "e4", "eK"],
                "structured_events": True,
            },
            "contract": {
                "source_manifest_sha256": "a" * 64,
                "object_names": ["can"],
                "policy_to_id": {"openvla": 0},
                "body_to_id": {"piper": 0},
                "train_seeds": [1, 2],
                "validation_seeds": [3],
                "sealed_test_seeds": [4],
            },
        },
        path,
    )


def _expanded(tmp_path: Path, *, axis: str = "policy") -> tuple[Path, Path, Path]:
    source = tmp_path / "source.pt"
    expanded = tmp_path / "expanded.pt"
    manifest = tmp_path / "manifest.json"
    split = tmp_path / "split.json"
    manifest.write_text('{"source":true}', encoding="utf-8")
    split.write_text('{"train":[1,2],"validation":[3],"test":[4]}', encoding="utf-8")
    _source_checkpoint(source)
    target = "smolvla" if axis == "policy" else "aloha"
    expand_source_core(
        source,
        expanded,
        axis=axis,
        target_name=target,
        source_manifest=manifest,
        source_split=split,
    )
    return expanded, manifest, split


def _cache() -> dict[str, object]:
    return {
        "hidden_dim": 16,
        "action_dim": 14,
        "proprio_dim": 14,
        "object_delta_dim": 3,
        "events": ["e0", "e12", "e3", "e4", "eK"],
        "source_manifest_sha256": "a" * 64,
        "policy_to_id": {"openvla": 0},
        "body_to_id": {"piper": 0},
        "arrays": {
            "policy_id": np.asarray([0, 0, 0], dtype=np.int64),
            "body_id": np.asarray([0, 0, 0], dtype=np.int64),
        },
    }


@pytest.mark.parametrize("axis", ["policy", "embodiment"])
def test_load_and_validate_exact_source_expansion(tmp_path: Path, axis: str) -> None:
    expanded, manifest, split = _expanded(tmp_path, axis=axis)
    payload, audit = load_expanded_source_core(
        expanded,
        source_manifest=manifest,
        source_split=split,
    )
    # The source checkpoint fixture uses a placeholder digest.  Bind it to the
    # fixture manifest for the cache-contract portion of this test.
    payload["contract"]["source_manifest_sha256"] = "a" * 64
    contract = validate_source_contract(
        payload,
        _cache(),
        {"train": [1, 2], "validation": [3], "test": [4]},
    )
    assert audit["ready_for_protocol_freeze"] is False
    assert contract["axis"] == axis
    assert contract["target_row"] == 1
    assert contract["source_transition_count"] == 3


def test_load_rejects_different_source_split(tmp_path: Path) -> None:
    expanded, manifest, _ = _expanded(tmp_path)
    different = tmp_path / "different.json"
    different.write_text('{"train":[2]}', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source_split_path"):
        load_expanded_source_core(
            expanded,
            source_manifest=manifest,
            source_split=different,
        )


def test_validate_rejects_reserved_id_in_source_cache(tmp_path: Path) -> None:
    expanded, manifest, split = _expanded(tmp_path)
    payload, _ = load_expanded_source_core(
        expanded, source_manifest=manifest, source_split=split
    )
    payload["contract"]["source_manifest_sha256"] = "a" * 64
    cache = _cache()
    cache["arrays"]["policy_id"] = np.asarray([0, 1], dtype=np.int64)  # type: ignore[index]
    with pytest.raises(ValueError, match="reserved target row"):
        validate_source_contract(
            payload,
            cache,
            {"train": [1, 2], "validation": [3], "test": [4]},
        )


def test_batch_guard_is_fail_closed() -> None:
    contract = {"batch_id_field": "policy_id", "target_row": 1}
    assert_reserved_row_absent({"policy_id": torch.tensor([0, 0])}, contract)
    with pytest.raises(RuntimeError, match="reserved target row"):
        assert_reserved_row_absent({"policy_id": torch.tensor([0, 1])}, contract)


def test_adamw_decay_is_repaired_bit_exact() -> None:
    config = EventWorldModelConfig(
        state_input_dim=8,
        action_dim=2,
        proprio_dim=2,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=8,
        clock_hidden_dim=4,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=2,
        metadata_dim=4,
        dropout=0.0,
    )
    model = ActionConditionedEventWorldModel(config)
    name = "action_encoder.policy_embedding.weight"
    parameter = dict(model.named_parameters())[name]
    reference = parameter[1].detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    parameter[0].sum().backward()
    optimizer.step()
    assert not torch.equal(parameter[1], reference)
    restore_reserved_row(
        model, parameter_name=name, target_row=1, reference=reference
    )
    assert reserved_row_is_bit_exact(
        model.state_dict(),
        parameter_name=name,
        target_row=1,
        reference=reference,
    )


def test_change_detection_excludes_reserved_row() -> None:
    before = {
        "embedding": torch.zeros(2, 3),
        "shared": torch.zeros(2, 2),
    }
    only_reserved = copy.deepcopy(before)
    only_reserved["embedding"][1, 0] = 1
    assert not source_parameters_changed(
        before, only_reserved, parameter_name="embedding", target_row=1
    )
    source_changed = copy.deepcopy(before)
    source_changed["embedding"][0, 0] = 1
    assert source_parameters_changed(
        before, source_changed, parameter_name="embedding", target_row=1
    )
    shared_changed = copy.deepcopy(before)
    shared_changed["shared"][0, 0] = 1
    assert source_parameters_changed(
        before, shared_changed, parameter_name="embedding", target_row=1
    )


def test_proof_has_exact_fail_closed_fields(tmp_path: Path) -> None:
    expanded, manifest, split = _expanded(tmp_path)
    proof = make_retraining_proof(
        expanded_path=expanded,
        source_manifest=manifest,
        source_split=split,
        training_steps=25,
        training_groups=2,
    )
    assert set(proof) == {
        "format",
        "status",
        "input_expanded_checkpoint_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_split_path",
        "source_split_sha256",
        "source_training_steps",
        "source_training_groups",
        "target_data_read",
        "target_labels_read",
        "reserved_row_used_in_source_batches",
        "shared_core_retrained",
    }
    assert proof["target_data_read"] is False
    assert proof["target_labels_read"] is False
    assert proof["reserved_row_used_in_source_batches"] is False


def test_validation_score_requires_event_duration_evidence() -> None:
    metrics = {
        "reach_brier": 0.1,
        "success_brier": 0.2,
        "event_macro_f1": 0.7,
        "future_semantic_cosine": 0.8,
        "duration_observed_mae_steps": 3.0,
        "object_delta_mae": 0.01,
        "relative_transition_macro_f1": 0.6,
        "predicate_macro_f1": 0.7,
        "next_reached_event_observed_macro_f1": 0.5,
    }
    assert math_is_finite(
        validation_selection_score(metrics, duration_scale=10.0, object_scale=0.1)
    )
    metrics["duration_observed_mae_steps"] = None
    with pytest.raises(RuntimeError, match="lacks observed"):
        validation_selection_score(metrics, duration_scale=10.0, object_scale=0.1)


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))
