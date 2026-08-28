from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import select_smolvla_piper_evaluation400_root_candidate_v3 as selector  # noqa: E402
import smolvla_piper_deployment_uncertainty_v1 as uncertainty  # noqa: E402


def calibration() -> dict:
    selected = {
        "minimum_group_relative_composite_rank_score_margin": 0.125,
        "maximum_structured_pair_uncertainty": 0.5,
        "maximum_global_candidate_uncertainty": 0.5,
    }
    ranker_base = {
        "enabled_for_primary": True,
        "score_semantics": (
            "source_contract_rank_score_minus_same_group_lowest_legal_baseline_then_five_member_mean"
        ),
        "score_is_success_logit": False,
        "score_is_success_probability": False,
        "root_recovery_uncertainty_policy": (
            "excluded_at_initial_e0_without_observed_operational_regress"
        ),
        "root_structured_uncertainty_head_count": 5,
        "selected_candidate": selected,
    }
    ranker = {
        **ranker_base,
        "root_group_ranker_sha256": selector.canonical_sha256(ranker_base),
    }
    value = {
        "metrics": {
            "post_event": {"deployment_temperature": 1.0},
            "next_event": {"deployment_temperature": 1.0},
            "success": {"deployment_temperature": 1.0},
            "conditional_recovery": {"deployment_temperature": 1.0},
            "duration_lognormal_mixture": {
                "deployment_scale_multiplier": 1.0
            },
            "object_total_variance": {
                "deployment_object_error_robust_scale_m": 1.0,
                "deployment_scale_multiplier": 1.0,
            },
        },
        "head_enabled_for_primary": {
            "post_event": True, "next_event": True, "duration": True,
            "success": True, "object_effect": True, "recovery": True,
        },
        "all_six_heads_support_performance_uncertainty_gate_passed": True,
        "root_group_ranker": ranker,
        "abstain_threshold": {"enabled": True, "maximum_total_uncertainty": 0.5},
    }
    value["calibration_sha256"] = selector.canonical_sha256(value)
    return value


def authority(calibration_value: dict) -> dict:
    contracts = []
    for index in range(5):
        base = {
            "source_checkpoint_file_sha256": str(index + 1) * 64,
            "base_score": "candidate_rank_score",
            "source_action_rank_residual": True,
            "source_action_rank_success_only": False,
            "residual_combination": (
                "candidate_rank_score_plus_action_rank_residual"
            ),
            "success_temperature": 1.0,
        }
        contracts.append({**base, "contract_sha256": selector.canonical_sha256(base)})
    thresholds = {
        "minimum_formal190_composite_margin": 0.125,
        "maximum_formal190_pair_uncertainty": 0.5,
        "maximum_global_total_uncertainty": 0.5,
        "root_group_ranker_sha256": calibration_value["root_group_ranker"][
            "root_group_ranker_sha256"
        ],
    }
    member_authority = {
        "source_rank_numeric_contract": selector.SOURCE_RANK_NUMERIC_CONTRACT,
        "members": [
            {
                "member_index": index,
                "source_checkpoint_file_sha256": contract[
                    "source_checkpoint_file_sha256"
                ],
                "source_rank_score_contract_sha256": contract[
                    "contract_sha256"
                ],
                "success_temperature": contract["success_temperature"],
            }
            for index, contract in enumerate(contracts)
        ],
    }
    value = {
        "calibration_sha256": calibration_value["calibration_sha256"],
        "formal190_root_group_ranker_sha256": calibration_value[
            "root_group_ranker"
        ]["root_group_ranker_sha256"],
        "source_rank_score_contract_sha256": [
            contract["contract_sha256"] for contract in contracts
        ],
        "source_rank_score_contracts": contracts,
        "source_rank_numeric_contract": selector.SOURCE_RANK_NUMERIC_CONTRACT,
        "source_rank_member_authority": member_authority,
        "source_rank_member_authority_sha256": selector.canonical_sha256(
            member_authority
        ),
        "uncertainty_contract": {
            "formal190_object_error_robust_scale_m": 1.0,
            "duration_deployment_scale_applied_before_selector": True,
            "object_deployment_scale_applied_before_selector": True,
            "object_predictions_physical_xyz_before_selector": True,
            "root_recovery_uncertainty_policy": (
                "excluded_at_initial_e0_without_observed_operational_regress"
            ),
            "root_structured_uncertainty_head_count": 5,
            "deployment_uncertainty_contract_sha256": "e" * 64,
        },
        "deployment_parameters": {
            "post_event_temperature": 1.0,
            "next_event_temperature": 1.0,
            "success_temperature": 1.0,
            "conditional_recovery_temperature": 1.0,
            "duration_scale_multiplier": 1.0,
            "object_scale_multiplier": 1.0,
            "object_error_robust_scale_m": 1.0,
            "deployment_uncertainty_contract_sha256": "e" * 64,
        },
        "formal190_thresholds": thresholds,
        "deployment_uncertainty_implementation": {
            "path": str(Path(uncertainty.__file__).resolve()),
            "file_sha256": __import__("hashlib").sha256(
                Path(uncertainty.__file__).read_bytes()
            ).hexdigest(),
        },
    }
    value["selector_authority_sha256"] = selector.canonical_sha256(value)
    return value


def predictions(*, uncertain: bool = False) -> dict[str, np.ndarray]:
    event = np.zeros((5, 4, 5), dtype=np.float64)
    binary = np.zeros((5, 4), dtype=np.float64)
    log_scale = np.full((5, 4), -8.0, dtype=np.float64)
    object_log_scale = np.full((5, 4, 3), -8.0, dtype=np.float64)
    if not uncertain:
        event[..., 0] = 20.0
        binary[...] = 20.0
    else:
        log_scale[...] = 3.0
        object_log_scale[...] = 3.0
    rank = np.tile(
        np.asarray([0.0, 0.5, 0.2, 0.1], dtype=np.float32), (5, 1)
    )
    residual = np.full((5, 4), 0.1, dtype=np.float32)
    return {
        "post_event_logits": event.copy(),
        "next_event_logits": event.copy(),
        "duration_log_mean": np.zeros((5, 4)),
        "duration_log_scale": log_scale,
        "success_logit": binary.copy(),
        "recovery_logit": binary.copy(),
        "object_mean": np.zeros((5, 4, 3)),
        "object_log_scale": object_log_scale,
        "source_contract_rank_score": rank,
        "source_contract_base_rank_score": rank - residual,
        "source_action_rank_residual": residual,
    }


def test_formal190_composite_margin_accepts_and_proves_algebra() -> None:
    calibrated = calibration()
    result = selector.select_root_candidate_v3(
        predictions=predictions(),
        prediction_candidate_indices=np.arange(4),
        candidate_legal=np.ones(4, dtype=bool),
        fallback_index=0,
        calibration=calibrated,
        selector_authority=authority(calibrated),
    )
    assert result["selected_candidate_index"] == 1
    assert result["proposed_candidate_index"] == 1
    assert result["score_margin"] == pytest.approx(0.5)
    assert result["candidate_change_accepted"] is True
    assert result["source_contract_rank_score_is_success_logit"] is False
    assert result["decision_algebra_sha256"]


def test_six_head_uncertainty_abstains_to_lowest_legal() -> None:
    calibrated = calibration()
    result = selector.select_root_candidate_v3(
        predictions=predictions(uncertain=True),
        prediction_candidate_indices=np.arange(4),
        candidate_legal=np.ones(4, dtype=bool),
        fallback_index=0,
        calibration=calibrated,
        selector_authority=authority(calibrated),
    )
    assert result["proposed_candidate_index"] == 1
    assert result["selected_candidate_index"] == 0
    assert result["candidate_change_accepted"] is False


def test_source_rank_float32_training_order_probe_and_random_matrix() -> None:
    calibrated = calibration()
    frozen_authority = authority(calibrated)
    rng = np.random.default_rng(20260828)
    for base, residual in (
        (
            np.full((5, 4), np.float32(0.1), dtype=np.float32),
            np.full((5, 4), np.float32(0.2), dtype=np.float32),
        ),
        (
            rng.normal(size=(5, 4)).astype(np.float32),
            rng.normal(size=(5, 4)).astype(np.float32),
        ),
    ):
        values = predictions()
        composite = base + residual / np.float32(1.0)
        values["source_contract_base_rank_score"] = base
        values["source_action_rank_residual"] = residual
        values["source_contract_rank_score"] = composite
        result = selector.select_root_candidate_v3(
            predictions=values,
            prediction_candidate_indices=np.arange(4),
            candidate_legal=np.ones(4, dtype=bool),
            fallback_index=0,
            calibration=calibrated,
            selector_authority=frozen_authority,
        )
        assert result["source_rank_numeric_contract"] == (
            selector.SOURCE_RANK_NUMERIC_CONTRACT
        )
        assert np.array_equal(
            np.asarray(
                result["member_source_contract_rank_scores"], dtype=np.float32
            ),
            composite,
        )
    assert np.float32(0.1) + np.float32(0.2) != 0.3


def test_source_rank_one_float32_ulp_tamper_fails_closed() -> None:
    calibrated = calibration()
    values = predictions()
    rank = values["source_contract_rank_score"]
    rank[0, 1] = np.nextafter(rank[0, 1], np.float32(np.inf), dtype=np.float32)
    with pytest.raises(selector.RootSelectorV3Error, match="base plus residual"):
        selector.select_root_candidate_v3(
            predictions=values,
            prediction_candidate_indices=np.arange(4),
            candidate_legal=np.ones(4, dtype=bool),
            fallback_index=0,
            calibration=calibrated,
            selector_authority=authority(calibrated),
        )


def test_missing_composite_score_or_disabled_six_head_gate_fails_closed() -> None:
    calibrated = calibration()
    broken = predictions()
    broken.pop("source_contract_rank_score")
    with pytest.raises(selector.RootSelectorV3Error, match="source_contract_rank_score"):
        selector.select_root_candidate_v3(
            predictions=broken,
            prediction_candidate_indices=np.arange(4),
            candidate_legal=np.ones(4, dtype=bool),
            fallback_index=0,
            calibration=calibrated,
            selector_authority=authority(calibrated),
        )


def test_margin_equal_to_formal_threshold_strictly_falls_back() -> None:
    calibrated = calibration()
    values = predictions()
    values["source_contract_rank_score"][:, 1] = 0.125
    values["source_contract_rank_score"][:, 2:] = 0.0
    values["source_contract_base_rank_score"] = (
        values["source_contract_rank_score"]
        - values["source_action_rank_residual"]
    )
    result = selector.select_root_candidate_v3(
        predictions=values,
        prediction_candidate_indices=np.arange(4),
        candidate_legal=np.ones(4, dtype=bool),
        fallback_index=0,
        calibration=calibrated,
        selector_authority=authority(calibrated),
    )
    assert result["proposed_candidate_index"] == 1
    assert result["score_margin"] == pytest.approx(0.125)
    assert result["selected_candidate_index"] == 0
    assert result["candidate_change_accepted"] is False
    calibrated["head_enabled_for_primary"]["recovery"] = False
    with pytest.raises(selector.RootSelectorV3Error, match="six-head"):
        selector.select_root_candidate_v3(
            predictions=predictions(),
            prediction_candidate_indices=np.arange(4),
            candidate_legal=np.ones(4, dtype=bool),
            fallback_index=0,
            calibration=calibrated,
            selector_authority=authority(calibrated),
        )
