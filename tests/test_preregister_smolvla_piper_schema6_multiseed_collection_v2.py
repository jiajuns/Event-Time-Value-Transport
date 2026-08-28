from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preregister_smolvla_piper_schema6_multiseed_collection_v2 as protocol  # noqa: E402


def _write_frozen(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def _write_json_frozen(path: Path, value: object) -> Path:
    return _write_frozen(
        path, (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    )


def _signed(value: dict[str, object], key: str) -> dict[str, object]:
    result = dict(value)
    result[key] = protocol.canonical_sha256(result)
    return result


def _semantic_receipt() -> dict[str, object]:
    return _signed(
        {
            "format": "etsf_explicit_instruction_semantics_receipt_v1",
            "task": protocol.TASK,
            "instruction": protocol.INSTRUCTION,
        },
        "receipt_sha256",
    )


def _manifest() -> dict[str, object]:
    semantic = _semantic_receipt()
    instruction_sha = hashlib.sha256(protocol.INSTRUCTION.encode()).hexdigest()
    splits: dict[str, list[dict[str, object]]] = {}
    global_ordinal = 0
    for split, count in protocol.SPLIT_COUNTS.items():
        rows: list[dict[str, object]] = []
        for ordinal in range(count):
            row: dict[str, object] = {
                "task": protocol.TASK,
                "actor_id": protocol.ACTOR_ID,
                "target_body": protocol.TARGET_BODY,
                "global_ordinal": global_ordinal,
                "split": split,
                "ordinal": ordinal,
                "stage_role": "identity_only_reset",
                "requested_seed": 100_201_000 + global_ordinal,
                "resolved_seed": 200_201_000 + global_ordinal,
                "instruction": protocol.INSTRUCTION,
                "instruction_sha256": instruction_sha,
                "instruction_semantics_receipt": semantic,
                "instruction_semantics_receipt_sha256": semantic["receipt_sha256"],
                "initial_scene_state_sha256": hashlib.sha256(
                    f"scene-{global_ordinal}".encode()
                ).hexdigest(),
                "initial_measured_joint_state_sha256": hashlib.sha256(
                    f"measured-{global_ordinal}".encode()
                ).hexdigest(),
                "initial_commanded_drive_target_sha256": hashlib.sha256(
                    f"drive-{global_ordinal}".encode()
                ).hexdigest(),
            }
            row["pair_id"] = protocol.canonical_sha256(
                protocol._row_pair_identity(row)
            )
            rows.append(row)
            global_ordinal += 1
        splits[split] = rows
    return _signed(
        {
            "format": protocol.TARGET_FORMAT,
            "status": protocol.TARGET_STATUS,
            "task": protocol.TASK,
            "actor_id": protocol.ACTOR_ID,
            "source_body": "aloha_agilex",
            "target_body": protocol.TARGET_BODY,
            "purpose": "nonfresh_development_only_no_confirmation_claim",
            "label_access_contract": {"labels_read": False},
            "instruction_contract": {"instruction": protocol.INSTRUCTION},
            "splits": splits,
            "provenance": {},
            "d250_exclusion": {},
            "heldout_exclusion_attestation": {},
            "capability_receipt": {
                "environment_reset_only": True,
                "environment_step_calls": 0,
                "policy_import_or_forward_calls": 0,
                "labels_or_outcomes_read": False,
                "policy_execution_authorized_by_manifest": False,
            },
        },
        "seed_manifest_sha256",
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    manifest = _write_json_frozen(tmp_path / "target_manifest.json", _manifest())
    event = _write_json_frozen(
        tmp_path / "event_spec.json",
        {
            "chains": {
                protocol.TASK: {
                    "merge_e1_e2": True,
                    "chain": ["e0", "e12", "e3", "e4", "eK"],
                }
            },
            "calibration": {
                protocol.TASK: {"moving": "can", "anchor": "", "tau_d": 0.02}
            },
        },
    )
    code = tmp_path / "r6j_code"
    code.mkdir()
    artifact_shas: dict[str, str] = {}
    for name in protocol.R6J_RUNTIME_ARTIFACTS:
        path = _write_frozen(code / name, f"# frozen {name}\n".encode())
        artifact_shas[name] = protocol.file_sha256(path)
    runtime = _write_frozen(tmp_path / "bound_python", b"#!/bin/sh\n")
    runner = _write_frozen(tmp_path / "multiseed_runner_v2.py", b"# protocol runner\n")
    output = tmp_path / "collection_output"
    prereg = tmp_path / "preregistration.json"
    kwargs: dict[str, object] = {
        "target_seed_manifest_path": manifest,
        "expected_target_seed_manifest_file_sha256": protocol.file_sha256(manifest),
        "r6j_code_root": code,
        "expected_r6j_code_closure_sha256": protocol.canonical_sha256(artifact_shas),
        "event_spec_path": event,
        "expected_event_spec_sha256": protocol.file_sha256(event),
        "runtime_python_path": runtime,
        "expected_runtime_python_sha256": protocol.file_sha256(runtime),
        "v2_runner_path": runner,
        "expected_v2_runner_sha256": protocol.file_sha256(runner),
        "output_root": output,
        "preregistration_path": prereg,
        "gpu_lock_path": tmp_path / "rtx4090.lock",
    }
    return {"kwargs": kwargs, "manifest": manifest, "output": output, "prereg": prereg}


def test_builds_exact_adaptation80_validation50_outcome_blind_plan(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    value = protocol.build_preregistration(**fixture["kwargs"])
    commands = value["commands"]
    assert len(commands) == 130
    assert [row["split"] for row in commands] == ["adaptation"] * 80 + ["validation"] * 50
    assert [row["ordinal"] for row in commands[:80]] == list(range(80))
    assert [row["ordinal"] for row in commands[80:]] == list(range(50))
    assert all(row["candidate_original_indices"] == [0, 1, 2, 3] for row in commands)
    assert value["collection_scope"]["selection_or_order_depends_on_outcome"] is False
    assert value["production_execution_authorized"] is False
    protocol.validate_preregistration(value)


def test_evaluation_has_no_command_seed_or_output_and_no_hdf_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    evaluation_seed = str(100_201_000 + 130)
    touched_hdf: list[Path] = []
    original_open = Path.open

    def guarded_open(self: Path, *args: object, **kwargs: object):
        if self.suffix.casefold() in protocol.HDF_SUFFIXES:
            touched_hdf.append(self)
            raise AssertionError("HDF access is forbidden during CPU preregistration")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    value = protocol.build_preregistration(**fixture["kwargs"])
    encoded_commands = json.dumps(value["commands"], sort_keys=True)
    assert evaluation_seed not in encoded_commands
    assert value["collection_scope"]["evaluation_commands_generated"] == 0
    assert value["collection_scope"]["evaluation_environment_resets_authorized"] == 0
    assert value["collection_scope"]["evaluation_hdf5_files_opened"] == 0
    assert touched_hdf == []


def test_each_command_requires_seed_local_registry_and_pose_contract(
    tmp_path: Path,
) -> None:
    value = protocol.build_preregistration(**_fixture(tmp_path)["kwargs"])
    reset = value["per_seed_reset_contract"]
    assert reset["live_registry_required_after_every_seed_reset"] is True
    assert reset["fixed_seed_object_registry_reuse_allowed"] is False
    assert reset["required_objects_in_order"] == ["can", "pot"]
    assert reset["required_asset_families"] == ["105_sauce-can", "060_kitchenpot"]
    reset_paths = [row["outputs"]["per_seed_reset_receipt"] for row in value["commands"]]
    assert len(reset_paths) == len(set(reset_paths)) == 130


def test_manifest_pair_mutation_fails_even_after_outer_resigning(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["manifest"]
    path.chmod(0o644)
    value = json.loads(path.read_text())
    value["splits"]["adaptation"][0]["resolved_seed"] += 7
    value.pop("seed_manifest_sha256")
    value["seed_manifest_sha256"] = protocol.canonical_sha256(value)
    _write_json_frozen(path, value)
    fixture["kwargs"]["expected_target_seed_manifest_file_sha256"] = protocol.file_sha256(path)
    with pytest.raises(protocol.MultiSeedProtocolError, match="identity changed"):
        protocol.build_preregistration(**fixture["kwargs"])


def test_target_manifest_sha_is_a_hard_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["kwargs"]["expected_target_seed_manifest_file_sha256"] = "0" * 64
    with pytest.raises(protocol.MultiSeedProtocolError, match="manifest file SHA256 mismatch"):
        protocol.build_preregistration(**fixture["kwargs"])


def test_protected_path_is_rejected_before_read(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    protected = tmp_path / "fresh_hidden" / "target_manifest.json"
    protected.parent.mkdir()
    fixture["kwargs"]["target_seed_manifest_path"] = protected
    with pytest.raises(protocol.MultiSeedProtocolError, match="forbidden component"):
        protocol.build_preregistration(**fixture["kwargs"])


def test_protected_embedded_path_is_never_accepted_or_dereferenced(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["manifest"]
    path.chmod(0o644)
    value = json.loads(path.read_text())
    value["unexpected_lineage"] = "/tmp/confirmation_hidden/seed.json"
    value.pop("seed_manifest_sha256")
    value["seed_manifest_sha256"] = protocol.canonical_sha256(value)
    _write_json_frozen(path, value)
    fixture["kwargs"]["expected_target_seed_manifest_file_sha256"] = protocol.file_sha256(path)
    with pytest.raises(protocol.MultiSeedProtocolError, match="embeds a forbidden path"):
        protocol.build_preregistration(**fixture["kwargs"])


def _completed_receipt(prereg: dict[str, object], index: int) -> dict[str, object]:
    command = prereg["commands"][index]
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return _signed(
        {
            "format": protocol.GROUP_RECEIPT_FORMAT,
            "status": protocol.GROUP_RECEIPT_STATUS,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "command_sha256": command["command_sha256"],
            "split": command["split"],
            "ordinal": command["ordinal"],
            "requested_seed": command["requested_seed"],
            "resolved_seed": command["expected_resolved_seed"],
            "pair_id": command["pair_id"],
            "candidate_original_indices": [0, 1, 2, 3],
            "branch_records": 4,
            "per_seed_reset_receipt_sha256": digest(f"reset-{index}"),
            "object_registry_sha256": digest(f"registry-{index}"),
            "pose_spec_sha256": digest(f"pose-{index}"),
            "group_file_sha256": digest(f"group-{index}"),
        },
        "group_receipt_sha256",
    )


def test_resume_accepts_only_signed_gap_free_completed_prefix(tmp_path: Path) -> None:
    prereg = protocol.build_preregistration(**_fixture(tmp_path)["kwargs"])
    receipts = [_completed_receipt(prereg, 0), _completed_receipt(prereg, 1)]
    pending = protocol.validate_completed_prefix(prereg, receipts)
    assert pending[0]["command_sha256"] == prereg["commands"][2]["command_sha256"]
    gap = [_completed_receipt(prereg, 1)]
    with pytest.raises(protocol.MultiSeedProtocolError, match="exact completed prefix"):
        protocol.validate_completed_prefix(prereg, gap)
    forged = [_completed_receipt(prereg, 0)]
    forged[0]["group_file_sha256"] = "f" * 64
    with pytest.raises(protocol.MultiSeedProtocolError, match="signature mismatch"):
        protocol.validate_completed_prefix(prereg, forged)


def test_create_once_and_collect_one_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["output"].mkdir()
    with pytest.raises(protocol.MultiSeedProtocolError, match="create-once"):
        protocol.build_preregistration(**fixture["kwargs"])
    with pytest.raises(protocol.MultiSeedProtocolError, match="production collection is not authorized"):
        protocol.main(["collect-one"])
