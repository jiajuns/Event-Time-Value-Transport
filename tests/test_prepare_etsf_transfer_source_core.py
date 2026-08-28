from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_etsf_transfer_source_core import (  # noqa: E402
    expand_source_core,
    file_sha256,
    verify_expansion,
    verify_source_retraining,
)


def _checkpoint(path: Path, *, policies: int = 1, bodies: int = 1) -> None:
    torch.save(
        {
            "model": {
                "action_encoder.policy_embedding.weight": torch.arange(
                    policies * 4, dtype=torch.float32
                ).reshape(policies, 4),
                "action_encoder.body_embedding.weight": torch.arange(
                    bodies * 4, dtype=torch.float32
                ).reshape(bodies, 4),
                "semantic.gru.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            },
            "config": {
                "state_input_dim": 8,
                "num_policies": policies,
                "num_bodies": bodies,
            },
            "contract": {
                "policy_to_id": {f"source_policy_{index}": index for index in range(policies)},
                "body_to_id": {f"source_body_{index}": index for index in range(bodies)},
                "event_spec_sha256": "a" * 64,
            },
            "normalization": {"action_mean": [0.0] * 14},
        },
        path,
    )


def _source_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "source_manifest.json"
    split = tmp_path / "source_split.json"
    manifest.write_text('{"groups":100}', encoding="utf-8")
    split.write_text('{"train":[1,2]}', encoding="utf-8")
    return manifest, split


@pytest.mark.parametrize(
    ("axis", "target", "embedding", "count", "mapping"),
    [
        (
            "policy",
            "smolvla",
            "action_encoder.policy_embedding.weight",
            "num_policies",
            "policy_to_id",
        ),
        (
            "embodiment",
            "aloha-agilex",
            "action_encoder.body_embedding.weight",
            "num_bodies",
            "body_to_id",
        ),
    ],
)
def test_expand_preserves_parent_and_adds_only_reserved_row(
    tmp_path: Path,
    axis: str,
    target: str,
    embedding: str,
    count: str,
    mapping: str,
) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    _checkpoint(source, policies=2, bodies=2)
    before = torch.load(source, map_location="cpu", weights_only=True)
    audit = expand_source_core(
        source,
        output,
        axis=axis,
        target_name=target,
        source_manifest=source_manifest,
        source_split=source_split,
    )
    after = torch.load(output, map_location="cpu", weights_only=True)
    assert audit["status"] == "vocabulary_preparation_requires_source_retraining"
    assert audit["target_data_read"] is False
    assert audit["target_labels_read"] is False
    assert audit["shared_parent_tensors_preserved_bit_exact"] is True
    assert audit["ready_for_protocol_freeze"] is False
    assert after["config"][count] == before["config"][count] + 1
    assert after["contract"][mapping][f"__reserved__{target}"] == before["config"][count]
    assert torch.equal(after["model"][embedding][:-1], before["model"][embedding])
    expected = before["model"][embedding].to(torch.float64).mean(0).to(torch.float32)
    assert torch.equal(after["model"][embedding][-1], expected)
    for name, tensor in before["model"].items():
        if name != embedding:
            assert torch.equal(after["model"][name], tensor)
    assert verify_expansion(source, output) == audit


def test_verify_rejects_changed_shared_core(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    _checkpoint(source)
    expand_source_core(
        source,
        output,
        axis="policy",
        target_name="smolvla",
        source_manifest=source_manifest,
        source_split=source_split,
    )
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["model"]["semantic.gru.weight"][0, 0] += 1
    torch.save(payload, output)
    with pytest.raises(ValueError, match="shared core tensor changed"):
        verify_expansion(source, output)


def test_expand_safely_loads_numpy_normalization_from_authoritative_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    _checkpoint(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["normalization"] = {
        "object_delta_mean": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "object_delta_std": np.asarray([1.0, 1.1, 1.2], dtype=np.float32),
    }
    torch.save(payload, source)
    audit = expand_source_core(
        source,
        output,
        axis="policy",
        target_name="smolvla",
        source_manifest=source_manifest,
        source_split=source_split,
    )
    assert audit["status"] == "vocabulary_preparation_requires_source_retraining"
    assert verify_expansion(source, output) == audit


def test_expand_rejects_existing_target_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    _checkpoint(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["contract"]["policy_to_id"] = {"smolvla": 0}
    torch.save(payload, source)
    with pytest.raises(ValueError, match="already registered"):
        expand_source_core(
            source,
            output,
            axis="policy",
            target_name="smolvla",
            source_manifest=source_manifest,
            source_split=source_split,
        )
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        expand_source_core(
            source,
            output,
            axis="policy",
            target_name="another",
            source_manifest=source_manifest,
            source_split=source_split,
        )


def test_source_only_retraining_proof_is_required_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    expanded = tmp_path / "expanded.pt"
    retrained = tmp_path / "retrained.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    _checkpoint(source)
    expand_source_core(
        source,
        expanded,
        axis="policy",
        target_name="smolvla",
        source_manifest=source_manifest,
        source_split=source_split,
    )
    payload = torch.load(expanded, map_location="cpu", weights_only=True)
    payload["model"]["semantic.gru.weight"][0, 0] += 0.25
    payload["reserved_source_retraining"] = {
        "format": "etsf_reserved_source_core_retraining_v1",
        "status": "complete_source_only",
        "input_expanded_checkpoint_sha256": file_sha256(expanded),
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_split_path": str(source_split.resolve()),
        "source_split_sha256": file_sha256(source_split),
        "source_training_steps": 100,
        "source_training_groups": 100,
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_row_used_in_source_batches": False,
        "shared_core_retrained": True,
    }
    torch.save(payload, retrained)
    audit = verify_source_retraining(
        expanded,
        retrained,
        source_manifest=source_manifest,
        source_split=source_split,
    )
    assert audit["status"] == "source_core_ready_for_protocol_freeze"
    assert audit["ready_for_protocol_freeze"] is True
    assert audit["reserved_target_row_unchanged"] is True


def test_source_retraining_rejects_reserved_row_use(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    expanded = tmp_path / "expanded.pt"
    retrained = tmp_path / "retrained.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    _checkpoint(source)
    expand_source_core(
        source,
        expanded,
        axis="policy",
        target_name="smolvla",
        source_manifest=source_manifest,
        source_split=source_split,
    )
    payload = torch.load(expanded, map_location="cpu", weights_only=True)
    payload["model"]["semantic.gru.weight"][0, 0] += 0.25
    payload["reserved_source_retraining"] = {
        "format": "etsf_reserved_source_core_retraining_v1",
        "status": "complete_source_only",
        "input_expanded_checkpoint_sha256": file_sha256(expanded),
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_split_path": str(source_split.resolve()),
        "source_split_sha256": file_sha256(source_split),
        "source_training_steps": 10,
        "source_training_groups": 1,
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_row_used_in_source_batches": True,
        "shared_core_retrained": True,
    }
    torch.save(payload, retrained)
    with pytest.raises(ValueError, match="proof is invalid"):
        verify_source_retraining(
            expanded,
            retrained,
            source_manifest=source_manifest,
            source_split=source_split,
        )


def test_source_retraining_rejects_manifest_different_from_preparation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    expanded = tmp_path / "expanded.pt"
    retrained = tmp_path / "retrained.pt"
    source_manifest, source_split = _source_inputs(tmp_path)
    different_manifest = tmp_path / "different_source_manifest.json"
    different_manifest.write_text('{"groups":99}', encoding="utf-8")
    _checkpoint(source)
    expand_source_core(
        source,
        expanded,
        axis="policy",
        target_name="smolvla",
        source_manifest=source_manifest,
        source_split=source_split,
    )
    payload = torch.load(expanded, map_location="cpu", weights_only=True)
    payload["model"]["semantic.gru.weight"][0, 0] += 0.25
    payload["reserved_source_retraining"] = {
        "format": "etsf_reserved_source_core_retraining_v1",
        "status": "complete_source_only",
        "input_expanded_checkpoint_sha256": file_sha256(expanded),
        "source_manifest_path": str(different_manifest.resolve()),
        "source_manifest_sha256": file_sha256(different_manifest),
        "source_split_path": str(source_split.resolve()),
        "source_split_sha256": file_sha256(source_split),
        "source_training_steps": 100,
        "source_training_groups": 99,
        "target_data_read": False,
        "target_labels_read": False,
        "reserved_row_used_in_source_batches": False,
        "shared_core_retrained": True,
    }
    torch.save(payload, retrained)
    with pytest.raises(ValueError, match="proof is invalid"):
        verify_source_retraining(
            expanded,
            retrained,
            source_manifest=different_manifest,
            source_split=source_split,
        )
