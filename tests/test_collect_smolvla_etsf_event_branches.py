from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from collect_smolvla_etsf_event_branches import (  # noqa: E402
    SharedPrefixStateCapture,
    _synthetic_record,
    resolve_shared_prefix_capture,
    save_group,
    shared_state_contract,
    validate_group_file,
)
from openvla_etsf_event_world_model import EventWorldModelConfig  # noqa: E402
from train_openvla_etsf_counterfactual import (  # noqa: E402
    load_descriptor_groups,
    read_group,
    read_group_descriptor,
)


class _FakeTextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.Identity()


class _FakeBridge:
    def __init__(self) -> None:
        self.text_model = _FakeTextModel()

    def get_vlm_model(self):
        return SimpleNamespace(text_model=self.text_model)


def _fake_policy(prefix_length: int = 0):
    bridge = _FakeBridge()
    return SimpleNamespace(
        config=SimpleNamespace(prefix_length=prefix_length),
        model=SimpleNamespace(vlm_with_expert=bridge),
    )


def test_shared_prefix_capture_selects_final_state_token() -> None:
    policy = _fake_policy()
    capture = resolve_shared_prefix_capture(policy)
    prefix = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5)
    try:
        output = policy.model.vlm_with_expert.get_vlm_model().text_model.norm(prefix)
        assert torch.equal(output, prefix)
        assert capture.calls == 1
        assert torch.equal(capture.consume(), prefix[0, -1])
    finally:
        capture.close()


def test_shared_prefix_capture_rejects_padded_or_ambiguous_calls() -> None:
    with pytest.raises(RuntimeError, match="prefix_length=0"):
        resolve_shared_prefix_capture(_fake_policy(prefix_length=128))

    norm = torch.nn.Identity()
    capture = SharedPrefixStateCapture(norm, prefix_length=0)
    try:
        norm(torch.zeros(1, 2, 4))
        norm(torch.zeros(1, 2, 4))
        with pytest.raises(RuntimeError, match="exactly once"):
            capture.consume()
    finally:
        capture.close()


def test_schema_v5_synthetic_group_forbids_candidate_expert_hidden(
    tmp_path: Path,
) -> None:
    path = tmp_path / "group.hdf5"
    save_group(path, _synthetic_record())
    result = validate_group_file(path, 123, 2, 8, 4)
    assert result["query_transitions"] == 3

    with pytest.raises(RuntimeError, match="differs from current runtime"):
        validate_group_file(
            path,
            123,
            2,
            8,
            4,
            expected_modeling_sha256="f" * 64,
        )
    with pytest.raises(RuntimeError, match="differs from current runtime"):
        validate_group_file(
            path,
            123,
            2,
            8,
            4,
            expected_event_spec_sha256="e" * 64,
        )

    with h5py.File(path, "r+") as handle:
        handle.create_dataset("candidate_hidden", data=torch.zeros(2, 720).numpy())
    with pytest.raises(RuntimeError, match="candidate-specific"):
        validate_group_file(path, 123, 2, 8, 4)


def test_schema_v5_group_is_consumable_by_structured_counterfactual_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "group.hdf5"
    save_group(path, _synthetic_record())
    config = EventWorldModelConfig(
        state_input_dim=8,
        action_dim=14,
        proprio_dim=14,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=6,
        object_delta_dim=6,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=True,
        dropout=0.0,
    )
    calibrations = {
        "move_can_pot": {
            "moving": "can",
            "anchor": "pot",
            "centers": [[1.0, 1.0, 1.0]],
            "offset": [0.0, 0.0, 0.0],
            "delta_move": 0.05,
            "delta_z": 0.1,
            "tau_d": 0.15,
            "tau_motion": 0.03,
            "stationary_steps": 2,
        }
    }
    group = read_group(
        path,
        {},
        config,
        ["can", "pot"],
        calibrations=calibrations,
        regression_persistence_steps=2,
        expected_event_spec_sha256="2" * 64,
    )
    assert group.schema_version == 5
    assert group.hidden.shape == (2, 8)
    assert group.actions.shape == (2, 4, 14)
    assert group.continuation is not None
    assert group.continuation["hidden_t"].shape == (1, 8, 8)
    assert group.continuation["history_mask"].shape == (1, 8)
    assert group.continuation["history_mask"][0].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert (group.continuation["hidden_t"][0, 2:] == 0).all()
    assert group.state_contract == shared_state_contract(
        hidden_dim=8,
        modeling_sha256="0" * 64,
        bridge_sha256="1" * 64,
    )

    descriptor = read_group_descriptor(path, {})
    with pytest.raises(RuntimeError, match="unknown policy 'smolvla'"):
        load_descriptor_groups(
            [descriptor],
            config,
            ["can", "pot"],
            {"aloha-agilex": 0},
            {"openvla": 0},
            calibrations=calibrations,
            expected_event_spec_sha256="2" * 64,
        )

    with pytest.raises(RuntimeError, match="event-spec provenance mismatch"):
        read_group(
            path,
            {},
            config,
            ["can", "pot"],
            calibrations=calibrations,
            regression_persistence_steps=2,
            expected_event_spec_sha256="e" * 64,
        )


def test_counterfactual_loader_rechecks_shared_state_and_source_provenance(
    tmp_path: Path,
) -> None:
    config = EventWorldModelConfig(
        state_input_dim=8,
        action_dim=14,
        proprio_dim=14,
        semantic_dim=8,
        action_hidden_dim=8,
        transition_hidden_dim=12,
        clock_hidden_dim=6,
        object_delta_dim=6,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=True,
        dropout=0.0,
    )
    calibrations = {
        "move_can_pot": {
            "moving": "can",
            "anchor": "pot",
            "centers": [[1.0, 1.0, 1.0]],
            "offset": [0.0, 0.0, 0.0],
            "delta_move": 0.05,
            "delta_z": 0.1,
            "tau_d": 0.15,
            "tau_motion": 0.03,
            "stationary_steps": 2,
        }
    }

    shared_path = tmp_path / "tampered_shared.hdf5"
    save_group(shared_path, _synthetic_record())
    with h5py.File(shared_path, "r+") as handle:
        handle["pre_hidden"][1, 0] += 1
    with pytest.raises(RuntimeError, match="do not share the root state"):
        read_group(
            shared_path,
            {},
            config,
            ["can", "pot"],
            calibrations=calibrations,
            expected_event_spec_sha256="2" * 64,
        )

    source_path = tmp_path / "tampered_source.hdf5"
    save_group(source_path, _synthetic_record())
    with h5py.File(source_path, "r+") as handle:
        handle.attrs["shared_state_modeling_sha256"] = "not-a-sha"
    with pytest.raises(RuntimeError, match="source hash"):
        read_group(
            source_path,
            {},
            config,
            ["can", "pot"],
            calibrations=calibrations,
            expected_event_spec_sha256="2" * 64,
        )

    event_path = tmp_path / "tampered_event.hdf5"
    save_group(event_path, _synthetic_record())
    with h5py.File(event_path, "r+") as handle:
        handle["branches/candidate_000/event_steps"][1] = 2
    with pytest.raises(RuntimeError, match="event/predicate provenance mismatch"):
        read_group(
            event_path,
            {},
            config,
            ["can", "pot"],
            calibrations=calibrations,
            expected_event_spec_sha256="2" * 64,
        )


def test_schema_v5_self_test_runs_without_robotwin_or_gpu() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "collect_smolvla_etsf_event_branches.py"),
            "--self-test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SELF_TEST_COMPLETE=" in result.stdout
    assert '"gpu_hook_verified": false' in result.stdout
