from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_etsf_transfer_asset_preflight import (  # noqa: E402
    file_sha256,
    validate_preflight,
)


def _preflight(tmp_path: Path) -> dict[str, object]:
    artifact_path = tmp_path / "actor_or_code.bin"
    artifact_path.write_bytes(b"content-addressed-fixture")
    artifact = {
        "path": str(artifact_path.resolve()),
        "sha256": file_sha256(artifact_path),
    }
    return {
        "format": "etsf_transfer_asset_preflight_v1",
        "status": "ready_unconfounded_schema5",
        "study_id": "policy_transfer_piper_v1",
        "axis": "policy",
        "source_domain": {
            "policy": "openvla",
            "embodiment": "piper",
            "actor_artifact": artifact,
        },
        "target_domain": {
            "policy": "smolvla",
            "embodiment": "piper",
            "actor_artifact": artifact,
        },
        "tasks": ["move_can_pot"],
        "contracts": {
            "schema_version": 5,
            "event_spec_sha256": "a" * 64,
            "source_state_contract_sha256": "b" * 64,
            "target_state_contract_sha256": "c" * 64,
            "source_action_effect_contract_sha256": "d" * 64,
            "target_action_effect_contract_sha256": "e" * 64,
        },
        "capabilities": {
            "source_schema5_collector": artifact,
            "target_schema5_collector": artifact,
            "deployable_observer": {
                "mode": "actor_hidden_observer",
                "artifact": artifact,
            },
            "privileged_pose_upper_bound_available": True,
        },
        "access": {
            "openvla_confirmation_labels_read": False,
            "gpu_job_started": False,
        },
    }


def test_policy_transfer_requires_same_embodiment(tmp_path: Path) -> None:
    value = _preflight(tmp_path)
    validate_preflight(value)
    mixed = copy.deepcopy(value)
    mixed["target_domain"]["embodiment"] = "aloha-agilex"  # type: ignore[index]
    with pytest.raises(ValueError, match="identical embodiment"):
        validate_preflight(mixed)


def test_embodiment_transfer_requires_same_policy(tmp_path: Path) -> None:
    value = _preflight(tmp_path)
    value["axis"] = "embodiment"
    value["target_domain"]["policy"] = "openvla"  # type: ignore[index]
    value["target_domain"]["embodiment"] = "aloha-agilex"  # type: ignore[index]
    validate_preflight(value)
    value["target_domain"]["policy"] = "smolvla"  # type: ignore[index]
    with pytest.raises(ValueError, match="identical policy"):
        validate_preflight(value)


def test_preflight_requires_schema5_and_nonprivileged_observer(tmp_path: Path) -> None:
    value = _preflight(tmp_path)
    value["contracts"]["schema_version"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="schema-v5"):
        validate_preflight(value)
    value = _preflight(tmp_path)
    value["capabilities"]["deployable_observer"]["mode"] = (  # type: ignore[index]
        "privileged_simulator_pose_upper_bound"
    )
    with pytest.raises(ValueError, match="actor-hidden or RGB"):
        validate_preflight(value)


def test_preflight_rejects_changed_artifact(tmp_path: Path) -> None:
    value = _preflight(tmp_path)
    actor_path = Path(value["source_domain"]["actor_artifact"]["path"])  # type: ignore[index]
    actor_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_preflight(value)
