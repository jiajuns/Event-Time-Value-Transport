from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_smolvla_piper_r6d_direct_actor_smoke import (  # noqa: E402
    ACTION_DIM,
    ACTOR_ID,
    INSTRUCTION,
    PREFIX_DIM,
    canonical_sha256,
)
from run_smolvla_piper_r6f_feasibility_smoke import (  # noqa: E402
    FORMAT,
    R6E_EXPECTED_UNSAFE_TARGET,
    R6E_EXPECTED_UNSAFE_VALUE,
    assess_candidate_first_action,
    explicit_named_map_first_action,
)
from verify_smolvla_piper_r6f_feasibility_receipt import (  # noqa: E402
    FeasibilityReceiptVerificationError,
    load_completed_feasibility_preregistration,
    verify_feasibility_receipt,
)
import verify_smolvla_piper_r6f_feasibility_receipt as verifier_module  # noqa: E402
from verify_smolvla_piper_zero_shot_preflight import (  # noqa: E402
    PIPER_ACTION_SLOTS,
    array_sha256,
    file_sha256,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _bounds() -> list[list[float]]:
    return [[float(slot.lower), float(slot.upper)] for slot in PIPER_ACTION_SLOTS]


def _valid_action(value: float) -> np.ndarray:
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[[1, 8]] = value
    action[[6, 13]] = 0.5
    return action


def _query(query_index: int) -> dict[str, Any]:
    processed = np.arange(ACTION_DIM, dtype=np.float32) / 100.0 + query_index
    prefix = np.arange(PREFIX_DIM, dtype=np.float32) / 1000.0 + query_index
    raw = np.arange(ACTION_DIM, dtype=np.float32) / 20.0 + query_index
    unsafe = _valid_action(0.2)
    unsafe[1] = np.float32(R6E_EXPECTED_UNSAFE_VALUE)
    actions = [unsafe, _valid_action(0.2), _valid_action(0.3), _valid_action(0.4)]
    prefix_sha = array_sha256(prefix)
    records = []
    for candidate_index, action in enumerate(actions):
        records.append(
            {
                "candidate_index": candidate_index,
                "noise_sha256": _digest(f"noise-{query_index}-{candidate_index}"),
                "prefix_sha256": prefix_sha,
                "postprocessed_chunk_sha256": _digest(
                    f"chunk-{query_index}-{candidate_index}"
                ),
                "mapped_first_action_sha256": array_sha256(action),
                "first_action": action.tolist(),
                "feasibility": assess_candidate_first_action(
                    action,
                    _bounds(),
                    query_index=query_index,
                    candidate_index=candidate_index,
                ),
            }
        )
    return {
        "query_index": query_index,
        "selection_rule": "lowest_candidate_index_with_finite_in_bounds_first_action",
        "selection_uses_event_or_utility_score": False,
        "processed_state": processed.tolist(),
        "processed_state_sha256": array_sha256(processed),
        "shared_prefix": prefix.tolist(),
        "candidate_prefix_sha256": [prefix_sha] * 4,
        "prefix_bit_exact_across_all_four_candidates": True,
        "candidate_records": records,
        "selected_candidate_index": 1,
        "input_interface": {
            "drive_target": raw.tolist(),
            "state_shape": [1, ACTION_DIM],
            "state_sha256": array_sha256(raw.reshape(1, ACTION_DIM)),
            "main_image_shape": [1, 240, 320, 3],
            "main_image_sha256": _digest(f"main-{query_index}"),
            "wrist_images_shape": [1, 2, 240, 320, 3],
            "wrist_images_sha256": _digest(f"wrists-{query_index}"),
        },
        "env_step_performed": True,
        "action_horizon_per_env_step": 1,
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    prereg_path = tmp_path / "feasibility_preregistration.json"
    receipt_path = tmp_path / "feasibility_receipt.json"
    _write_json(prereg_path, {"fixture": True})
    runner_path = ROOT / "scripts" / "run_smolvla_piper_r6f_feasibility_smoke.py"
    source_records: dict[str, dict[str, str]] = {}
    for role in (
        "rlinf_robotwin_env",
        "robotwin_vector_env",
        "robotwin_base_task",
        "robotwin_robot_controller",
        "robotwin_eval_seed_registry",
    ):
        path = tmp_path / f"{role}.txt"
        path.write_text(role, encoding="utf-8")
        source_records[role] = {"path": str(path), "sha256": file_sha256(path)}
    logical_sha = _digest("logical-preregistration")
    r6e_lineage = {
        "path": str(tmp_path / "r6e" / "direct_actor_preregistration.json"),
        "file_sha256": _digest("r6e-file"),
        "logical_sha256": _digest("r6e-logical"),
        "runner_sha256": _digest("r6e-runner"),
        "failure_receipt_bound": False,
        "authorization": "lineage_only_not_R6f_execution_authority",
        "expected_external_diagnostic_not_content_authenticated": {
            "query_index": 0,
            "candidate_index": 0,
            "target_index": 1,
            "target_joint_name": "left:joint2",
            "reported_value_rounded": R6E_EXPECTED_UNSAFE_VALUE,
            "reported_lower_bound": 0.0,
            "reported_env_steps": 0,
        },
    }
    prereg = {
        "output": str(receipt_path),
        "preregistration_sha256": logical_sha,
        "r6e_lineage": r6e_lineage,
        "r6f_runner": {"path": str(runner_path), "sha256": file_sha256(runner_path)},
        "inherited_R6e_contract": {"runtime_source_artifacts": source_records},
    }
    r6c = {
        "receipt": {
            "static_contract": {
                "static_semantics": {"piper_action_bounds": _bounds()}
            }
        }
    }
    seed = {
        "path": str(tmp_path / "development_seed.json"),
        "sha256": _digest("development-seed"),
        "seed_registry": "explicit_v7_prospective_development",
        "requested_seed": 100101000,
        "expected_resolved_seed": 100101000,
        "fresh_confirmation_eligible": False,
        "label_free": True,
    }
    queries = [_query(index) for index in range(4)]
    first = queries[0]["candidate_records"][0]
    _, mapping = explicit_named_map_first_action(
        np.zeros(ACTION_DIM, dtype=np.float32)
    )
    receipt = {
        "format": FORMAT,
        "status": "completed_R6f_feasibility_simulation_interface_smoke",
        "actor_id": ACTOR_ID,
        "source_body": "aloha",
        "target_body": "piper",
        "target_runtime": "RoboTwin_simulation_only",
        "real_robot_execution": False,
        "fresh_inputs_used": False,
        "fresh_trajectory_or_label_opened": False,
        "task_success_claimed": False,
        "performance_evaluation_authorized": False,
        "transfer_claim_authorized": False,
        "event_or_utility_scoring_performed": False,
        "preregistration": {
            "path": str(prereg_path.resolve()),
            "file_sha256": file_sha256(prereg_path),
            "logical_sha256": logical_sha,
        },
        "r6e_lineage": r6e_lineage,
        "r6e_candidate0_diagnostic_independently_recomputed": {
            "accepted": first["feasibility"]["accepted"],
            "first_action": first["first_action"],
            "feasibility": first["feasibility"],
            "matches_reported_left_joint2_failure_approximately": True,
        },
        "development_seed_contract": seed,
        "environment_contract": {
            "embodiment": ["piper", "piper", 0.6],
            "requested_seed": 100101000,
            "resolved_seed": 100101000,
            "explicit_instruction": INSTRUCTION,
            "scene_seed_and_instruction_strictly_bound": False,
            "center_crop": False,
            "collect_wrist_camera": True,
            "state_is_measured_qpos": False,
            "runtime_module_origins": {
                role: source_records[role]
                for role in (
                    "rlinf_robotwin_env",
                    "robotwin_vector_env",
                    "robotwin_base_task",
                    "robotwin_robot_controller",
                )
            },
            "eval_seed_registry": source_records["robotwin_eval_seed_registry"],
        },
        "loaded_policy_contract": {
            "checkpoint_declared_state_dim": 6,
            "runtime_preprocessed_state_shape": [1, 14],
            "checkpoint_action_dim": 14,
            "runtime_postprocessed_action_shape": [1, 50, 14],
            "state_dimension_conflict_retained": True,
            "normalizer_state_dim": 14,
        },
        "execution": {
            "queries_performed": 4,
            "steps_executed": 4,
            "max_steps": 4,
            "action_exec_steps": 1,
            "candidate_count_per_query": 4,
            "selection_rule": "lowest_candidate_index_with_finite_in_bounds_first_action",
            "feasibility_baseline_only": True,
            "event_or_utility_scoring_performed": False,
            "no_feasible_candidate_halt": False,
            "stopped_on_termination": False,
            "stopped_on_truncation": True,
            "success_observed_diagnostic_only": False,
            "all_env_actions_prevalidated": True,
            "silent_clipping_possible": False,
            "mapping_contract": mapping,
            "queries": queries,
        },
        "time_contract": {
            "unit": "policy action row count",
            "physical_duration_claimed": False,
        },
        "implementation_sha256": file_sha256(runner_path),
        "interpretation": (
            "fixed-candidate feasibility fallback only; no ETSF/event ranking, "
            "task performance, transfer, safety, or real-robot claim"
        ),
    }
    _write_json(receipt_path, receipt)

    def loader(_path: Path):
        return prereg, {}, r6c, {
            "runtime_source_artifacts": {
                role: source_records[role]
                for role in (
                    "rlinf_robotwin_env",
                    "robotwin_vector_env",
                    "robotwin_base_task",
                    "robotwin_robot_controller",
                )
            }
        }, seed

    return {
        "prereg_path": prereg_path,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "loader": loader,
    }


def _verify(fixture: dict[str, Any]) -> dict[str, Any]:
    return verify_feasibility_receipt(
        fixture["prereg_path"],
        fixture["receipt_path"],
        expected_receipt_sha256=file_sha256(fixture["receipt_path"]),
        preregistration_loader=fixture["loader"],
    )


def test_completed_preregistration_loader_is_read_only_after_receipt_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "preregistration.json"
    receipt_path = tmp_path / "receipt.json"
    _write_json(prereg_path, {"fixture": True})
    _write_json(receipt_path, {"already": "materialized"})
    inherited_keys = (
        "r6c_binding",
        "r6d_binding",
        "development_seed",
        "runtime_roots",
        "runtime_source_artifacts",
        "vlm_metadata_bundle_sha256",
        "model_bundle_sha256",
        "capability_contract",
        "mapping_contract",
        "state_contract",
        "caveats",
    )
    r6e_prereg = {key: {"key": key} for key in inherited_keys}
    inherited = {key: r6e_prereg[key] for key in inherited_keys}
    r6e = {
        "path": str(tmp_path / "r6e_preregistration.json"),
        "file_sha256": _digest("r6e-file"),
        "logical_sha256": _digest("r6e-logical"),
        "runner_sha256": _digest("r6e-runner"),
        "failure_receipt_bound": False,
        "authorization": "lineage_only_not_R6f_execution_authority",
        "expected_external_diagnostic_not_content_authenticated": {},
    }
    prereg = {
        "output": str(receipt_path),
        "r6e_lineage": r6e,
        "inherited_R6e_contract": inherited,
        "inherited_R6e_contract_sha256": canonical_sha256(inherited),
    }
    r6c, r6d, seed = {"r6c": True}, {"r6d": True}, {"seed": True}
    monkeypatch.setattr(
        verifier_module,
        "validate_feasibility_preregistration",
        lambda path: prereg,
    )
    monkeypatch.setattr(
        verifier_module,
        "bind_r6e_preregistration",
        lambda path: r6e,
    )
    monkeypatch.setattr(
        verifier_module,
        "_load_and_recompute_preregistration",
        lambda path: (r6e_prereg, r6c, r6d, seed),
    )
    assert load_completed_feasibility_preregistration(prereg_path) == (
        prereg,
        r6e_prereg,
        r6c,
        r6d,
        seed,
    )
    assert receipt_path.is_file()


def test_verifies_complete_four_by_four_receipt_and_reports_evidence_ceiling(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _verify(fixture)
    assert result["status"] == "passed_complete_R6f_receipt_internal_consistency"
    assert result["verified_execution"] == {
        "queries": 4,
        "candidates_per_query": 4,
        "candidate_records": 16,
        "steps_executed": 4,
        "action_horizon": 1,
        "lowest_legal_selection_recomputed": True,
        "candidate0_R6e_diagnostic_recomputed": True,
        "fresh_used": False,
        "performance_or_transfer_claim": False,
    }
    hashes = result["hash_evidence"]
    assert hashes["mapped_first_action_arrays_recomputed"] == 16
    assert hashes["candidate_prefix_hash_links_recomputed"] == 16
    assert hashes["postprocessed_chunks_independently_recomputed"] is False
    assert hashes["noise_tensors_independently_recomputed"] is False
    assert result["receipt"]["logical_sha256"] == canonical_sha256(
        fixture["receipt"]
    )


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    [
        (
            "processed-state",
            lambda value: value["execution"]["queries"][0]["processed_state"].__setitem__(0, 9.0),
            "processed_state_sha256",
        ),
        (
            "prefix",
            lambda value: value["execution"]["queries"][0]["shared_prefix"].__setitem__(0, 9.0),
            "candidate_prefix_sha256",
        ),
        (
            "mapped-action",
            lambda value: value["execution"]["queries"][0]["candidate_records"][1]["first_action"].__setitem__(0, 0.25),
            "mapped_first_action_sha256",
        ),
        (
            "selection",
            lambda value: value["execution"]["queries"][0].__setitem__("selected_candidate_index", 2),
            "lowest legal",
        ),
        (
            "horizon",
            lambda value: value["execution"]["queries"][0].__setitem__("action_horizon_per_env_step", 2),
            "H=1",
        ),
        (
            "claim",
            lambda value: value.__setitem__("task_success_claimed", True),
            "capability boundary",
        ),
        (
            "candidate0-diagnostic",
            lambda value: value["r6e_candidate0_diagnostic_independently_recomputed"].__setitem__("matches_reported_left_joint2_failure_approximately", False),
            "match flag",
        ),
        (
            "chunk-digest",
            lambda value: value["execution"]["queries"][0]["candidate_records"][0].__setitem__("postprocessed_chunk_sha256", "bad"),
            "postprocessed_chunk_sha256",
        ),
        (
            "duplicate-noise",
            lambda value: value["execution"]["queries"][0]["candidate_records"][1].__setitem__("noise_sha256", value["execution"]["queries"][0]["candidate_records"][0]["noise_sha256"]),
            "reuses a candidate noise",
        ),
        (
            "raw-state",
            lambda value: value["execution"]["queries"][0]["input_interface"]["drive_target"].__setitem__(0, 8.0),
            "state_sha256",
        ),
        (
            "missing-query",
            lambda value: value["execution"]["queries"].pop(),
            "exactly four queries",
        ),
        (
            "missing-final-time-limit",
            lambda value: value["execution"].__setitem__("stopped_on_truncation", False),
            "time-limit-truncated",
        ),
        (
            "implementation",
            lambda value: value.__setitem__("implementation_sha256", "0" * 64),
            "authenticated runner",
        ),
    ],
)
def test_internal_tamper_fails_even_when_mutated_file_sha_is_supplied(
    tmp_path: Path, name: str, mutate, match: str
) -> None:
    fixture = _fixture(tmp_path)
    value = copy.deepcopy(fixture["receipt"])
    mutate(value)
    _write_json(fixture["receipt_path"], value)
    with pytest.raises(FeasibilityReceiptVerificationError, match=match):
        _verify(fixture)


def test_recomputed_feasibility_rejects_semantic_tamper_with_updated_action_hash(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    value = copy.deepcopy(fixture["receipt"])
    record = value["execution"]["queries"][0]["candidate_records"][1]
    record["first_action"][1] = -0.25
    action = np.asarray(record["first_action"], dtype=np.float32)
    record["mapped_first_action_sha256"] = array_sha256(action)
    _write_json(fixture["receipt_path"], value)
    with pytest.raises(
        FeasibilityReceiptVerificationError, match="bounds recomputation"
    ):
        _verify(fixture)


def test_external_receipt_sha_is_mandatory_and_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(FeasibilityReceiptVerificationError, match="file SHA mismatch"):
        verify_feasibility_receipt(
            fixture["prereg_path"],
            fixture["receipt_path"],
            expected_receipt_sha256="0" * 64,
            preregistration_loader=fixture["loader"],
        )


def test_strict_json_rejects_nan_even_with_current_file_sha(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    text = fixture["receipt_path"].read_text(encoding="utf-8")
    fixture["receipt_path"].write_text(
        text.replace("0.0", "NaN", 1), encoding="utf-8"
    )
    with pytest.raises(FeasibilityReceiptVerificationError, match="strict JSON"):
        _verify(fixture)


def test_fresh_named_receipt_path_is_rejected_before_loading(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fresh_dir = tmp_path / "Fresh50"
    fresh_dir.mkdir()
    fresh_receipt = fresh_dir / "feasibility_receipt.json"
    fresh_receipt.write_bytes(fixture["receipt_path"].read_bytes())
    with pytest.raises(Exception, match="Fresh path"):
        verify_feasibility_receipt(
            fixture["prereg_path"],
            fresh_receipt,
            expected_receipt_sha256=file_sha256(fresh_receipt),
            preregistration_loader=fixture["loader"],
        )


def test_preregistration_file_and_logical_hashes_are_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = copy.deepcopy(fixture["receipt"])
    value["preregistration"]["logical_sha256"] = "0" * 64
    _write_json(fixture["receipt_path"], value)
    with pytest.raises(FeasibilityReceiptVerificationError, match="logical SHA"):
        _verify(fixture)


def test_candidate0_expected_joint_and_value_are_not_replaceable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    value = copy.deepcopy(fixture["receipt"])
    record = value["execution"]["queries"][0]["candidate_records"][0]
    action = np.asarray(record["first_action"], dtype=np.float32)
    action[1] = -0.02
    record["first_action"] = action.tolist()
    record["mapped_first_action_sha256"] = array_sha256(action)
    record["feasibility"] = assess_candidate_first_action(
        action, _bounds(), query_index=0, candidate_index=0
    )
    diagnostic = value["r6e_candidate0_diagnostic_independently_recomputed"]
    diagnostic["first_action"] = record["first_action"]
    diagnostic["feasibility"] = record["feasibility"]
    _write_json(fixture["receipt_path"], value)
    with pytest.raises(
        FeasibilityReceiptVerificationError,
        match=f"R6e {R6E_EXPECTED_UNSAFE_TARGET}",
    ):
        _verify(fixture)
