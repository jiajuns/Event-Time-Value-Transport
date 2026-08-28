from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from preregister_smolvla_piper_v7_paired_development import (  # noqa: E402
    EXPECTED_D250_CANDIDATES,
    ProtocolError,
    canonical_sha256,
    validate_d250_identity,
    validate_target_seed_manifest,
)
from resolve_smolvla_piper_target_reset_only import resolve_with_adapter  # noqa: E402
from smolvla_piper_schema6_runtime_adapter_v2 import (  # noqa: E402
    RUNTIME_CONTRACT_FORMAT,
    RUNTIME_CONTRACT_STATUS,
    RUNTIME_ROOT_KEYS,
    RUNTIME_SOURCE_KEYS,
    DEFAULT_MEASURED_CHANNEL,
    directory_tree_sha256,
)
from smolvla_piper_target_seed_manifest import (  # noqa: E402
    ADAPTATION,
    AUTHORIZATION_FORMAT,
    EVALUATION,
    INSTRUCTION,
    TOTAL,
    VALIDATION,
    build_plan,
    freeze_manifest,
    legacy_v1_projection,
    reject_sensitive_path,
    signed,
    validate_plan,
    validate_reset_receipt,
)


SHA = "a" * 64
HELDOUT_SHA = "b" * 64


def d250_identity() -> dict[str, object]:
    groups = [
        {
            "index": index,
            "status": "collected",
            "requested_seed": index,
            "resolved_seed": index,
            "candidate_names": list(EXPECTED_D250_CANDIDATES),
        }
        for index in range(250)
    ]
    return {
        "format": "etsf_event_branch_collection_identity_v1",
        "schema_version": 5,
        "task": "move_can_pot",
        "body": "piper_piper_0.6",
        "candidate_count": 4,
        "completed": 250,
        "seed_registry": "explicit_v7_prospective_development",
        "label_access_contract": "identity_only_no_success_steps_event_or_outcome_fields",
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
        "event_spec_sha256": "1" * 64,
        "v7_seed_manifest_sha256": "2" * 64,
        "v7_preregistration_sha256": "3" * 64,
        "groups": groups,
        "requested_seeds": list(range(250)),
        "resolved_seeds": list(range(250)),
    }


def plan() -> dict[str, object]:
    return build_plan(
        d250_identity=d250_identity(),
        d250_identity_file_sha256=SHA,
        heldout_identity_set_sha256=HELDOUT_SHA,
        upstream_v7_seed_manifest_file_sha256="c" * 64,
        upstream_v7_seed_manifest_payload_sha256="d" * 64,
        upstream_data_audit_file_sha256="e" * 64,
        resolver_implementation_sha256="f" * 64,
        reset_adapter_implementation_sha256="9" * 64,
        candidate_start=1000,
        candidate_count=TOTAL,
    )


def disjoint(target_sha: str, role: str) -> dict[str, object]:
    return signed(
        {
            "format": "etsf_private_identity_disjoint_attestation_v1",
            "status": "verified_disjoint_without_disclosing_heldout_identities",
            "target_role": role,
            "heldout_identity_set_sha256": HELDOUT_SHA,
            "target_identity_set_sha256": target_sha,
            "intersection_count": 0,
            "sensitive_identities_included": False,
        },
        "attestation_sha256",
    )


def runtime_contract(tmp_path: Path) -> dict[str, object]:
    roots = {}
    for key in RUNTIME_ROOT_KEYS:
        root = tmp_path / "runtime" / key
        root.mkdir(parents=True, exist_ok=True)
        roots[key] = str(root)
    for key in ("model_path", "vlm_metadata_path"):
        (Path(roots[key]) / "bound.bin").write_bytes(key.encode("ascii"))
    sources = {}
    for index, key in enumerate(sorted(RUNTIME_SOURCE_KEYS)):
        path = tmp_path / "sources" / f"{key}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# bound {index}\n", encoding="utf-8")
        sources[key] = {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    registry = tmp_path / "eval_seed_registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    result = {
        "format": RUNTIME_CONTRACT_FORMAT,
        "status": RUNTIME_CONTRACT_STATUS,
        "runtime_roots": roots,
        "runtime_source_artifacts": sources,
        "eval_seed_registry": {
            "path": str(registry),
            "sha256": __import__("hashlib").sha256(registry.read_bytes()).hexdigest(),
        },
        "measured_joint_state_channel": DEFAULT_MEASURED_CHANNEL,
        "gpu_index": 0,
        "max_episode_steps": 8,
        "offline_model_loading": True,
        "piper_action_bounds": [[-1.0, 1.0]] * 14,
        "model_tree_sha256": directory_tree_sha256(Path(roots["model_path"])),
        "vlm_metadata_tree_sha256": directory_tree_sha256(Path(roots["vlm_metadata_path"])),
        "reset_scratch_path": f"/tmp/schema6_reset_synthetic_{tmp_path.name}",
        "test_or_evaluation_execution_authorized": False,
        "fresh_or_confirmation_inputs_accepted": False,
    }
    result["runtime_contract_sha256"] = canonical_sha256(result)
    return result


def authorization(value: dict[str, object], tmp_path: Path) -> dict[str, object]:
    decoded = validate_plan(value)
    return signed(
        {
            "format": AUTHORIZATION_FORMAT,
            "status": "authorized_reset_only_after_private_disjoint_check",
            "plan_file_sha256": SHA,
            "plan_sha256": decoded["plan_sha256"],
            "resolver_implementation_sha256": "f" * 64,
            "reset_adapter_implementation_sha256": "9" * 64,
            "candidate_pool_disjoint_attestation": disjoint(
                value["candidate_pool"]["requested_identity_set_sha256"],
                "preregistered_reset_candidate_pool",
            ),
            "runtime_contract": runtime_contract(tmp_path),
            "permissions": {
                "environment_construct_allowed": True,
                "reset_only": True,
                "environment_step_allowed": False,
                "policy_import_or_forward_allowed": False,
                "reward_success_event_or_outcome_read_allowed": False,
            },
        },
        "authorization_sha256",
    )


def reset_once(seed: int, instruction: str) -> dict[str, object]:
    assert instruction == INSTRUCTION
    offset = float(seed) / 100000.0
    return {
        "setup_status": "stable",
        "requested_seed": seed,
        "resolved_seed": seed,
        "instruction_observed": instruction,
        "scene_state": {
            "can_pose": [offset, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            "pot_pose": [0.1, offset, 0.8, 1.0, 0.0, 0.0, 0.0],
        },
        "measured_joint_state": np.arange(14, dtype=np.float64) + offset,
        "commanded_drive_target": np.arange(14, dtype=np.float64) - offset,
    }


def receipt(value: dict[str, object], tmp_path: Path) -> dict[str, object]:
    return resolve_with_adapter(
        plan=value,
        plan_file_sha256=SHA,
        authorization=authorization(value, tmp_path),
        authorization_file_sha256="8" * 64,
        reset_once=reset_once,
    )


def test_plan_is_offline_fixed_and_does_not_disclose_heldout_identities() -> None:
    value = plan()
    decoded = validate_plan(value)
    assert len(decoded["candidates"]) == TOTAL == 530
    assert value["execution_gate"]["authorized_by_plan"] is False
    assert value["instruction_contract"]["instruction"] == INSTRUCTION
    assert value["instruction_contract"]["semantics_receipt"]["episode_info_list_used"] is False
    serialized = json.dumps(value).casefold()
    assert "heldout_requested" not in serialized
    assert "heldout_resolved" not in serialized
    assert "trajectory" not in serialized


@pytest.mark.parametrize("name", ["Fresh", "confirmation", "trajectory", "labels"])
def test_all_protocol_paths_reject_sensitive_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ProtocolError, match="non-Fresh"):
        reject_sensitive_path(tmp_path / name / "value.json", "test", must_exist=False)


def test_authorization_candidate_attestation_is_a_hard_gate(tmp_path: Path) -> None:
    value = plan()
    auth = authorization(value, tmp_path)
    changed = copy.deepcopy(auth)
    changed["candidate_pool_disjoint_attestation"]["intersection_count"] = 1
    changed.pop("authorization_sha256")
    changed = signed(changed, "authorization_sha256")
    with pytest.raises(ProtocolError, match="candidate_pool|signature"):
        resolve_with_adapter(
            plan=value,
            plan_file_sha256=SHA,
            authorization=changed,
            authorization_file_sha256="8" * 64,
            reset_once=lambda *_: pytest.fail("reset must not run before the gate"),
        )


def test_reset_receipt_freezes_instruction_semantics_and_three_state_hashes(tmp_path: Path) -> None:
    value = plan()
    result = receipt(value, tmp_path)
    decoded = validate_reset_receipt(result, plan=value, plan_file_sha256=SHA)
    assert len(decoded["rows"]) == TOTAL
    assert result["split_counts"] == {
        "adaptation": ADAPTATION,
        "validation": VALIDATION,
        "evaluation": EVALUATION,
    }
    assert all(row["stage_role"] == "direct_actor_only_operational" for row in result["rows"][:20])
    assert result["rows"][0]["instruction"] == INSTRUCTION
    assert result["rows"][0]["instruction_semantics_receipt"]["semantic_frame"] == {
        "theme": "can",
        "relation": "inside",
        "reference": "pot",
    }
    assert result["environment_step_calls"] == 0
    assert result["policy_import_or_forward_calls"] == 0


def test_resolver_rejects_stock_style_internal_seed_retry(tmp_path: Path) -> None:
    value = plan()

    def retry(seed: int, instruction: str) -> dict[str, object]:
        result = reset_once(seed, instruction)
        result["resolved_seed"] = seed + 1
        return result

    with pytest.raises(ProtocolError, match="internal seed retry"):
        resolve_with_adapter(
            plan=value,
            plan_file_sha256=SHA,
            authorization=authorization(value, tmp_path),
            authorization_file_sha256="8" * 64,
            reset_once=retry,
        )


def test_final_freezer_requires_selected_identity_attestation_and_projects_v1(tmp_path: Path) -> None:
    value = plan()
    reset = receipt(value, tmp_path)
    requested = [row["requested_seed"] for row in reset["rows"]]
    resolved = [row["resolved_seed"] for row in reset["rows"]]
    target_sha = canonical_sha256({"requested": requested, "resolved": resolved})
    manifest = freeze_manifest(
        plan=value,
        plan_file_sha256=SHA,
        authorization=authorization(value, tmp_path),
        authorization_file_sha256="8" * 64,
        reset_receipt=reset,
        reset_receipt_file_sha256="7" * 64,
        selected_identity_disjoint_attestation=disjoint(
            target_sha, "selected_requested_and_resolved_target_identities"
        ),
    )
    assert manifest["capability_receipt"]["policy_execution_authorized_by_manifest"] is False
    assert manifest["splits"]["adaptation"][0]["instruction"] == INSTRUCTION
    legacy = legacy_v1_projection(manifest)
    d250 = validate_d250_identity(d250_identity())
    decoded = validate_target_seed_manifest(
        legacy,
        d250_identity_file_sha256=SHA,
        d250=d250,
    )
    assert len(decoded["requested"]) == TOTAL

    bad = disjoint(target_sha, "selected_requested_and_resolved_target_identities")
    bad["sensitive_identities_included"] = True
    bad.pop("attestation_sha256")
    bad = signed(bad, "attestation_sha256")
    with pytest.raises(ProtocolError, match="attestation"):
        freeze_manifest(
            plan=value,
            plan_file_sha256=SHA,
            authorization=authorization(value, tmp_path),
            authorization_file_sha256="8" * 64,
            reset_receipt=reset,
            reset_receipt_file_sha256="7" * 64,
            selected_identity_disjoint_attestation=bad,
        )


def test_resolver_source_has_no_environment_step_or_policy_import_surface() -> None:
    source = (SCRIPTS / "resolve_smolvla_piper_target_reset_only.py").read_text(encoding="utf-8")
    forbidden_call = "." + "step" + "("
    assert forbidden_call not in source
    assert "transformers" not in source
    assert "torch" not in source
