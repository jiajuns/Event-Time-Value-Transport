from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from initialize_smolvla_schema5_native_event_core import (  # noqa: E402
    BODY_EMBEDDING,
    POLICY_EMBEDDING,
    make_config,
)
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
)
from train_openvla_etsf_counterfactual import (  # noqa: E402
    assert_reserved_ids_absent,
    assert_reserved_target_rows_bit_exact,
    install_source_action_normalization,
    reserved_rows_source_only_proof,
    resolve_source_action_normalization,
    restore_reserved_target_rows,
    source_train_action_statistics,
    tensor_sha256,
    validate_reserved_rows_source_only_proof,
    validate_reserved_target_rows,
)


def _payload() -> tuple[dict, ActionConditionedEventWorldModel]:
    torch.manual_seed(41)
    config = make_config()
    model = ActionConditionedEventWorldModel(config).cpu()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    contract = {
        "body_to_id": {"aloha-agilex": 0, "__reserved__piper": 1},
        "policy_to_id": {"smolvla": 0, "__reserved__openvla": 1},
        "reserved_target_rows": {
            "body": {
                "mapping_name": "body_to_id",
                "identity": "piper",
                "id": 1,
                "row": 1,
                "parameter": BODY_EMBEDDING,
                "tensor_sha256": tensor_sha256(state[BODY_EMBEDDING][1]),
                "initializer": "bit_exact_clone_source_row_data_blind_v1",
                "source_identity": "aloha-agilex",
                "source_row": 0,
            },
            "policy": {
                "mapping_name": "policy_to_id",
                "identity": "openvla",
                "id": 1,
                "row": 1,
                "parameter": POLICY_EMBEDDING,
                "tensor_sha256": tensor_sha256(state[POLICY_EMBEDDING][1]),
                "initializer": "bit_exact_clone_source_row_data_blind_v1",
                "source_identity": "smolvla",
                "source_row": 0,
            },
        },
        "action_normalization": {
            "status": "identity_placeholder_unfitted",
            "action_mean": [0.0] * config.action_dim,
            "action_std": [1.0] * config.action_dim,
        },
    }
    return {
        "model": state,
        "config": config.to_dict(),
        "contract": contract,
    }, model


def _group(
    key: str,
    values: list[float],
    mask: list[bool],
    *,
    continuation_values: list[float] | None = None,
    continuation_mask: list[bool] | None = None,
) -> SimpleNamespace:
    action_dim = 14
    continuation = None
    if continuation_values is not None:
        assert continuation_mask is not None
        continuation = {
            "action_chunks": np.asarray(continuation_values, dtype=np.float32)[
                None, :, None
            ]
            * np.ones((1, len(continuation_values), action_dim), dtype=np.float32),
            "action_mask": np.asarray(continuation_mask, dtype=bool)[None],
        }
    return SimpleNamespace(
        logical_key=key,
        actions=np.asarray(values, dtype=np.float32)[None, :, None]
        * np.ones((1, len(values), action_dim), dtype=np.float32),
        action_mask=np.asarray(mask, dtype=bool)[None],
        continuation=continuation,
    )


def test_adamw_weight_decay_changes_both_unused_rows_then_restore_is_bit_exact() -> None:
    payload, model = _payload()
    rows = validate_reserved_target_rows(payload)
    assert rows is not None
    before = {
        axis: model.state_dict()[row.parameter][row.row].clone()
        for axis, row in rows.items()
    }
    optimizer = torch.optim.AdamW(
        [
            model.action_encoder.body_embedding.weight,
            model.action_encoder.policy_embedding.weight,
        ],
        lr=0.1,
        weight_decay=0.2,
    )
    loss = model.action_encoder.body_embedding(torch.tensor([0])).sum()
    loss = loss + model.action_encoder.policy_embedding(torch.tensor([0])).sum()
    loss.backward()
    optimizer.step()
    assert all(
        not torch.equal(model.state_dict()[row.parameter][row.row], before[axis])
        for axis, row in rows.items()
    )

    restore_reserved_target_rows(model, rows)
    assert_reserved_target_rows_bit_exact(model, rows)
    assert all(
        torch.equal(model.state_dict()[row.parameter][row.row], before[axis])
        for axis, row in rows.items()
    )


def test_reserved_target_id_in_either_source_batch_axis_fails_closed() -> None:
    payload, _ = _payload()
    rows = validate_reserved_target_rows(payload)
    assert rows is not None
    assert_reserved_ids_absent(
        {"body_id": torch.tensor([0, 0]), "policy_id": torch.tensor([0, 0])},
        rows,
    )
    with pytest.raises(RuntimeError, match="reserved target body id"):
        assert_reserved_ids_absent(
            {"body_id": torch.tensor([0, 1]), "policy_id": torch.tensor([0, 0])},
            rows,
        )
    with pytest.raises(RuntimeError, match="reserved target policy id"):
        assert_reserved_ids_absent(
            {"body_id": torch.tensor([0, 0]), "policy_id": torch.tensor([1, 0])},
            rows,
        )


def test_reserved_row_tensor_sha_tampering_is_rejected() -> None:
    payload, _ = _payload()
    tampered = copy.deepcopy(payload)
    tampered["contract"]["reserved_target_rows"]["policy"][
        "tensor_sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="parameter/tensor SHA"):
        validate_reserved_target_rows(tampered)


def test_checkpoint_without_reserved_rows_keeps_legacy_behavior() -> None:
    payload, model = _payload()
    payload["contract"].pop("reserved_target_rows")
    payload["contract"].pop("action_normalization")
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rows = validate_reserved_target_rows(payload)
    assert rows is None
    assert resolve_source_action_normalization(payload, [], model.config) is None
    assert_reserved_ids_absent({}, rows)
    restore_reserved_target_rows(model, rows)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_cold_action_normalization_uses_only_valid_source_train_actions() -> None:
    payload, model = _payload()
    train_group = _group(
        "source|aloha|1",
        [1.0, 2.0, 9999.0],
        [True, True, False],
        continuation_values=[3.0, -9999.0],
        continuation_mask=[True, False],
    )
    proof = resolve_source_action_normalization(
        payload, [train_group], model.config
    )
    assert proof is not None
    assert proof["valid_action_sample_count"] == 3
    assert proof["candidate_valid_action_sample_count"] == 2
    assert proof["continuation_valid_action_sample_count"] == 1
    assert proof["validation_groups_used"] == 0
    assert proof["sealed_test_groups_used"] == 0
    np.testing.assert_array_equal(proof["action_mean"], np.full(14, 2.0))
    np.testing.assert_allclose(
        proof["action_std"], np.full(14, np.std([1.0, 2.0, 3.0])), rtol=1e-6
    )
    install_source_action_normalization(model, proof)
    assert torch.equal(model.action_encoder.action_mean, torch.full((14,), 2.0))

    # An extreme validation-shaped object is deliberately not an argument to
    # the statistic function; recomputation remains identical.
    _validation_group = _group("source|aloha|2", [1e8], [True])
    assert source_train_action_statistics([train_group], 14) == proof


def test_saved_source_only_proof_binds_both_final_row_shas_and_model() -> None:
    payload, model = _payload()
    rows = validate_reserved_target_rows(payload)
    assert rows is not None
    action_normalization = resolve_source_action_normalization(
        payload, [_group("source|aloha|1", [1.0, 2.0], [True, True])], model.config
    )
    install_source_action_normalization(model, action_normalization)
    proof = reserved_rows_source_only_proof(
        model,
        rows,
        source_training_steps=7,
        source_training_groups=1,
        input_pretrained_checkpoint_sha256="a" * 64,
        action_normalization=action_normalization,
    )
    assert proof is not None
    assert proof["target_data_read"] is False
    assert proof["target_labels_read"] is False
    assert proof["reserved_rows_used_in_source_batches"] is False
    assert proof["reserved_rows_unchanged_during_source_training"] is True
    for axis in ("body", "policy"):
        assert proof["rows"][axis]["initial_tensor_sha256"] == proof["rows"][axis][
            "final_tensor_sha256"
        ]
    saved = copy.deepcopy(payload)
    saved["model"] = model.state_dict()
    saved["reserved_target_rows_source_only_proof"] = proof
    saved["contract"]["reserved_target_rows_source_only_proof"] = proof
    validate_reserved_rows_source_only_proof(saved, rows)
