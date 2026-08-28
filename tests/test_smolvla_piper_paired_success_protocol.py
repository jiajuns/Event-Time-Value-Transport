from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from smolvla_piper_paired_success_protocol import (  # noqa: E402
    ACTOR_ID,
    AUTHORITY_FORMAT,
    FORMAT,
    HEAD_SUPPORT_FORMAT,
    INSTRUCTION,
    MINIMUM_HEAD_GROUPS_PER_SIDE,
    PAIR_RESULT_FORMAT,
    PRIMARY_HEAD_WEIGHTS,
    PRIMARY_UTILITY_FORMAT,
    PairedSuccessProtocolError,
    PreOutcomeSelectionHook,
    canonical_sha256,
    candidate_registry,
    derive_pair_result,
    evaluate_pair_results,
    exact_two_sided_mcnemar,
    file_sha256,
    freeze_preoutcome_selection,
    freeze_protocol,
    pair_identity,
    structured_multitask_selector_decision,
    synthetic_protocol,
    synthetic_smoke,
    validate_dependency_receipt,
    validate_head_support,
    validate_seed_authority,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _query(*, action_shift: float = 0.0) -> dict:
    actions = np.zeros((4, 50, 14), dtype=np.float32)
    for index in range(4):
        actions[index] = index + action_shift
    prefix = "a" * 64
    return {
        "hidden": np.arange(960, dtype=np.float32),
        "processed_state": np.arange(14, dtype=np.float32),
        "mapped_actions": actions,
        "feasibility_mask": np.asarray([True, True, True, True]),
        "legal_original_candidate_indices": np.arange(4, dtype=np.int16),
        "lowest_legal_original_candidate_index": 0,
        "native_action_sha256": [f"{index + 1:x}" * 64 for index in range(4)],
        "candidate_prefix_sha256": [prefix] * 4,
        "prefix_bit_exact": True,
    }


def _head_support(*, success_enabled: bool = False, object_enabled: bool = True) -> dict:
    heads = {}
    for name in PRIMARY_HEAD_WEIGHTS:
        enabled = True
        if name == "success":
            enabled = success_enabled
        if name == "object_effect":
            enabled = object_enabled
        minimum = MINIMUM_HEAD_GROUPS_PER_SIDE[name]
        count = minimum + 10 if enabled else max(0, minimum - 1)
        heads[name] = {
            "enabled_for_primary": enabled,
            "independent_positive_or_observed_groups": count,
            "independent_negative_or_censored_groups": count,
            "minimum_required_per_side": minimum,
            "support_source": "training_only_group_counts",
        }
    value = {
        "format": HEAD_SUPPORT_FORMAT,
        "status": "frozen_from_training_and_validation_only_before_paired_development",
        "heads": heads,
        "paired_development_outcomes_read": False,
        "sealed_evaluation_reserve_outcomes_read": False,
    }
    value["head_support_sha256"] = canonical_sha256(value)
    return value


def _protocol(*, threshold: float = 0.5, success_enabled: bool = False) -> dict:
    protocol = synthetic_protocol(pair_count=120, threshold=threshold)
    support = _head_support(success_enabled=success_enabled)
    protocol["primary_selector"]["head_support"] = {
        "heads": support["heads"],
        "head_support_sha256": support["head_support_sha256"],
    }
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256")
    protocol["protocol_sha256"] = canonical_sha256(unsigned)
    return protocol


def _predictions(*, high_uncertainty: bool = False) -> dict:
    # Event/time/object jointly prefer candidate 1.  Success alone prefers 2.
    post = np.zeros((4, 3), dtype=np.float64)
    nxt = np.zeros((4, 3), dtype=np.float64)
    nxt[1, 2] = 6.0
    duration = np.asarray([0.0, 3.0, 0.0, 0.0])
    success = np.asarray([-2.0, -2.0, 9.0, -2.0])
    object_effect = np.asarray([0.0, 2.0, 0.0, 0.0])
    epistemic = np.asarray([0.1, 0.6 if high_uncertainty else 0.1, 0.1, 0.1])
    return {
        "post_event_logits": post,
        "next_event_logits": nxt,
        "duration_selected_log_mean": duration,
        "success_logit": success,
        "object_effect_utility": object_effect,
        "aleatoric_uncertainty": np.asarray([0.1] * 4),
        "epistemic_uncertainty": epistemic,
    }


def _selector(protocol: dict, *, high_uncertainty: bool = False):
    def select(view):
        assert "success" not in view
        return structured_multitask_selector_decision(
            predictions=_predictions(high_uncertainty=high_uncertainty),
            candidate_valid_mask=view["feasibility_mask"],
            fallback_index=view["lowest_legal_original_candidate_index"],
            event_values=[0.0, 1.0, 2.0],
            protocol=protocol,
            plugin_manifest_sha256="b" * 64,
            adapter_checkpoint_sha256="c" * 64,
        )

    return select


def test_primary_multitask_utility_disables_unsupported_success_head() -> None:
    protocol = _protocol(success_enabled=False)
    decision = _selector(protocol)(_query())
    assert decision["primary_utility_format"] == PRIMARY_UTILITY_FORMAT
    assert decision["success_head_enabled_for_primary"] is False
    assert decision["proposed_index"] == 1
    assert decision["selected_index"] == 1
    assert decision["secondary_diagnostics"]["success_only"] == {
        "available": False,
        "selected_index": 0,
        "reason": "success_head_training_group_support_insufficient",
    }
    assert decision["aleatoric_and_epistemic_used_as_guard_only"] is True


def test_high_epistemic_plus_aleatoric_uncertainty_abstains_to_baseline() -> None:
    protocol = _protocol(threshold=0.5)
    decision = _selector(protocol, high_uncertainty=True)(_query())
    assert decision["proposed_index"] == 1
    assert decision["selected_index"] == 0
    assert decision["guard_fallback_used"] is True
    assert decision["total_uncertainty"] == pytest.approx(0.7)
    assert "uncertainty_above_guard" in decision["fallback_reasons"]


def test_selection_is_create_once_before_any_step_and_root_candidates_are_reproduced(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    pair = protocol["development_pairs"][0]
    step_count = {"value": 0}

    def query_fn(_observation, _index):
        assert step_count["value"] == 0
        return _query()

    output = tmp_path / "selection.json"
    hook = PreOutcomeSelectionHook(
        query_fn=query_fn,
        selector=_selector(protocol),
        pair=pair,
        protocol=protocol,
        selection_output=output,
    )
    hook({}, 0)
    assert output.is_file()
    frozen = json.loads(output.read_text())
    assert frozen["environment_steps_before_selection"] == 0
    assert frozen["candidate_outcomes_visible_to_selector"] is False
    hook({}, 0)
    with pytest.raises(PairedSuccessProtocolError, match="changed between"):
        hook.query_fn = lambda _observation, _index: _query(action_shift=0.25)
        hook({}, 0)
    with pytest.raises(FileExistsError):
        freeze_preoutcome_selection(
            pair=pair,
            query=_query(),
            protocol=protocol,
            selector=_selector(protocol),
            output=output,
        )


def test_executed_branch_success_not_prediction_becomes_pair_outcome(tmp_path: Path) -> None:
    protocol = _protocol()
    pair = protocol["development_pairs"][0]
    selection = freeze_preoutcome_selection(
        pair=pair,
        query=_query(),
        protocol=protocol,
        selector=_selector(protocol),
        output=tmp_path / "selection.json",
    )
    branches = [
        {"original_candidate_index": index, "success": index == 1}
        for index in range(4)
    ]
    record = {
        "status": "collected_development_group",
        "root_query": _query(),
        "baseline_original_candidate_index": 0,
        "branches": branches,
    }
    result = derive_pair_result(record, selection, protocol)
    assert result["baseline_success"] is False
    assert result["plugin_success"] is True
    assert result["paired_success_difference"] == 1
    assert result["task_success_source"] == "simulator_info_success_from_executed_schema6_branch"
    assert result["predicted_success_used_as_outcome"] is False
    assert set(result["secondary_task_success_diagnostics"]) == {
        "ablate_post_event",
        "ablate_next_event",
        "ablate_duration",
        "ablate_success",
        "ablate_object_effect",
        "success_only",
    }


def test_synthetic_gate_uses_unconditional_pairs_ci_exact_test_and_abstention() -> None:
    smoke = synthetic_smoke()
    assert smoke["gate_passed"] is True
    assert smoke["paired_success_delta"] == pytest.approx(28 / 120)
    assert smoke["paired_bootstrap_ci95"][0] > 0
    assert smoke["exact_two_sided_mcnemar_p"] < 0.05
    assert smoke["uncertainty_abstention_pairs"] == 8
    assert smoke["predicted_success_used_as_outcome"] is False
    assert exact_two_sided_mcnemar(30, 2) == pytest.approx(
        smoke["exact_two_sided_mcnemar_p"]
    )


def test_missing_pairs_fail_intention_to_treat_and_sample_gate() -> None:
    protocol = _protocol()
    rows = []
    for pair in protocol["development_pairs"][:10]:
        row = {
            "format": PAIR_RESULT_FORMAT,
            "status": "complete_executed_paired_task_success",
            "protocol_sha256": protocol["protocol_sha256"],
            "pair_id": pair["pair_id"],
            "baseline_success": False,
            "plugin_success": True,
            "proposed_change": True,
            "executed_change": True,
            "uncertainty_abstention": False,
            "guard_fallback_used": False,
            "total_uncertainty": 0.2,
            "uncertainty_threshold": 0.5,
            "primary_utility_format": PRIMARY_UTILITY_FORMAT,
            "success_head_enabled_for_primary": False,
            "object_head_enabled_for_primary": True,
            "secondary_task_success_diagnostics": {
                name: {
                    "available": name != "success_only",
                    "selected_index": 1 if name != "success_only" else 0,
                    "success": name != "success_only",
                    "paired_difference_vs_baseline": 1 if name != "success_only" else 0,
                }
                for name in protocol["primary_selector"]["secondary_diagnostics"]
            },
            "secondary_diagnostics_change_primary_gate": False,
            "task_success_source": "simulator_info_success_from_executed_schema6_branch",
            "predicted_success_used_as_outcome": False,
        }
        row["pair_result_sha256"] = canonical_sha256(row)
        rows.append(row)
    evaluation = evaluate_pair_results(protocol, rows)
    assert evaluation["gate_passed"] is False
    assert "minimum_complete_pair_gate_failed" in evaluation["gate_reasons"]
    assert "intention_to_treat_pair_completeness_gate_failed" in evaluation["gate_reasons"]
    assert evaluation["worst_case_missing_paired_delta"] < 0


def _seed_authority(count: int = 4) -> dict:
    rows = []
    for ordinal in range(count):
        reset = f"{ordinal + 1:x}" * 64
        identity = pair_identity(
            requested_seed=ordinal + 10,
            resolved_seed=ordinal + 20,
            reset_identity_sha256=reset,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "pair_id": canonical_sha256(identity),
                "requested_seed": ordinal + 10,
                "resolved_seed": ordinal + 20,
                "reset_identity_sha256": reset,
            }
        )
    development_sha = canonical_sha256(
        [
            {
                "pair_id": row["pair_id"],
                "requested_seed": row["requested_seed"],
                "resolved_seed": row["resolved_seed"],
            }
            for row in rows
        ]
    )
    reserve_sha = "e" * 64
    value = {
        "format": AUTHORITY_FORMAT,
        "status": "reset_identity_only_before_any_actor_or_outcome",
        "task": "move_can_pot",
        "body": "piper_piper_0.6",
        "actor_id": ACTOR_ID,
        "instruction": INSTRUCTION,
        "label_access_contract": "reset_identity_only_no_action_reward_success_event_or_trajectory",
        "development": rows,
        "sealed_evaluation_reserve": {
            "count": count,
            "identity_set_sha256": reserve_sha,
            "identities_disclosed": False,
            "outcomes_read": False,
        },
        "disjoint_attestation": {
            "development_identity_set_sha256": development_sha,
            "reserve_identity_set_sha256": reserve_sha,
            "intersection_count": 0,
            "verified_without_disclosing_reserve_identities": True,
        },
        "existing_sensitive_artifacts_read": False,
    }
    value["seed_authority_sha256"] = canonical_sha256(value)
    return value


def test_seed_authority_discloses_no_reserve_identity_or_outcome() -> None:
    authority = _seed_authority()
    audit = validate_seed_authority(authority, minimum_pairs=4)
    assert audit["development_pairs"] == 4
    assert audit["reserve_count"] == 4
    assert audit["reserve_identities_read"] is False
    changed = json.loads(json.dumps(authority))
    changed["development"][0]["success"] = True
    changed["seed_authority_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "seed_authority_sha256"}
    )
    with pytest.raises(PairedSuccessProtocolError, match="outcome-like"):
        validate_seed_authority(changed, minimum_pairs=4)


def test_head_support_uses_frozen_per_head_minimums() -> None:
    support = _head_support(success_enabled=False)
    audit = validate_head_support(support)
    assert audit["heads"]["post_event"]["minimum_required_per_side"] == 10
    assert audit["heads"]["next_event"]["enabled_for_primary"] is True
    assert audit["heads"]["duration"]["enabled_for_primary"] is True
    assert audit["heads"]["success"]["minimum_required_per_side"] == 50
    assert audit["heads"]["success"]["enabled_for_primary"] is False
    assert audit["heads"]["object_effect"]["minimum_required_per_side"] == 50
    support["heads"]["success"]["enabled_for_primary"] = True
    support["head_support_sha256"] = canonical_sha256(
        {key: value for key, value in support.items() if key != "head_support_sha256"}
    )
    with pytest.raises(PairedSuccessProtocolError, match="enablement"):
        validate_head_support(support)


def test_dependency_receipt_binds_file_logical_signature_zero_fields_and_run_exit(
    tmp_path: Path,
) -> None:
    receipt = {
        "format": "dependency_v1",
        "status": "complete",
        "test_hdf5_opened": 0,
        "artifacts_frozen_read_only": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = tmp_path / "dependency.json"
    exit_path = tmp_path / "run.exit"
    _write(receipt_path, receipt)
    exit_path.write_text("0\n", encoding="ascii")
    spec = {
        "name": "adapter",
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": file_sha256(receipt_path),
        "expected_format": "dependency_v1",
        "expected_status": "complete",
        "logical_sha256_field": "receipt_sha256",
        "required_fields": {
            "test_hdf5_opened": 0,
            "artifacts_frozen_read_only": True,
        },
        "run_exit_path": str(exit_path),
        "run_exit_file_sha256": file_sha256(exit_path),
    }
    assert validate_dependency_receipt(spec)["status"] == "complete"
    exit_path.write_text("1\n", encoding="ascii")
    spec["run_exit_file_sha256"] = file_sha256(exit_path)
    with pytest.raises(PairedSuccessProtocolError, match="not exact success"):
        validate_dependency_receipt(spec)
