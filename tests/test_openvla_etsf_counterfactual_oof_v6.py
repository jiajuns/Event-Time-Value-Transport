from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import openvla_etsf_counterfactual_oof_v6 as v6  # noqa: E402


def keys() -> list[str]:
    return [f"move_can_pot|piper|{index:03d}" for index in range(250)]


def manifest() -> dict:
    return v6.make_nested_oof_manifest(
        keys(), source_contract={"manifest_sha256": "a" * 64}, semantic_dim=96
    )


def test_nested_manifest_has_real_leak_free_inner_crossfit() -> None:
    value = manifest()
    audit = v6.validate_nested_oof_manifest(value, keys())
    assert audit == {
        "development_groups": 250,
        "outer_training_groups": 200,
        "outer_holdout_groups": 50,
        "inner_training_groups": 150,
        "inner_holdout_groups": 50,
    }
    outer_seen = []
    for outer in value["outer_folds"]:
        outer_holdout = set(outer["oof_holdout_groups"])
        inner_seen = []
        for inner in outer["inner_folds"]:
            assert not outer_holdout.intersection(inner["training_groups"])
            assert not outer_holdout.intersection(inner["selection_holdout_groups"])
            inner_seen.extend(inner["selection_holdout_groups"])
        assert set(inner_seen) == set(outer["training_groups"])
        assert len(inner_seen) == len(set(inner_seen)) == 200
        outer_seen.extend(outer["oof_holdout_groups"])
    assert set(outer_seen) == set(keys())
    assert len(outer_seen) == len(set(outer_seen)) == 250


def test_manifest_freezes_low_capacity_success_only_and_fresh_forbidden() -> None:
    value = manifest()
    model = value["model_contract"]
    assert model["factual_core"] == "frozen_bit_exact"
    assert model["trainable_parameter_names"] == ["action_rank_head.0.weight"]
    assert model["trainable_parameter_count"] == 192
    assert model["event_duration_object_terms_in_rank_score"] is False
    assert value["selector_contract"]["score_candidates"] == ["success_only"]
    assert value["fresh_confirmation"] == {
        "inputs_accepted": False,
        "data_or_labels_read": False,
        "authorization_possible": False,
        "policy": "forbidden_even_if_development_gate_passes",
    }


def test_formal_v6_rejects_non_192_parameter_head() -> None:
    with pytest.raises(ValueError, match="semantic_dim=96"):
        v6.make_nested_oof_manifest(
            keys(), source_contract={"manifest_sha256": "a" * 64}, semantic_dim=8
        )


def test_manifest_tampering_fails_closed() -> None:
    value = manifest()
    changed = copy.deepcopy(value)
    changed["training_contract"]["weight_decay"] = 0.0
    with pytest.raises(RuntimeError, match="signature"):
        v6.validate_nested_oof_manifest(changed, keys())
    changed = copy.deepcopy(value)
    changed["model_contract"]["trainable_parameter_names"].append(
        "transition.0.weight"
    )
    changed.pop("preregistration_sha256")
    changed["preregistration_sha256"] = v6.canonical_sha256(changed)
    with pytest.raises(RuntimeError, match="allowlist"):
        v6.validate_nested_oof_manifest(changed, keys())


def row(key: str, scores: list[float], success: list[float]) -> dict:
    return {
        "logical_key": key,
        "success_only_scores": np.asarray(scores),
        "success": np.asarray(success),
        "baseline_index": 0,
    }


def test_inner_guard_selection_and_outer_application_are_separate(monkeypatch) -> None:
    monkeypatch.setattr(v6, "BOOTSTRAP_SAMPLES", 100)
    inner = []
    for index in range(40):
        # Twenty helpful changes and no harmful changes make one inner guard
        # eligible without any outer label access.
        success = [0.0, 1.0] if index < 20 else [0.0, 0.0]
        inner.append(row(f"inner-{index}", [0.0, 0.3], success))
    selected = v6.select_inner_guard(inner)
    assert selected["enabled"] is True
    assert selected["gain_margin"] in v6.GUARD_MARGIN_THRESHOLDS
    outer = [row("outer", [0.0, 0.4], [1.0, 0.0])]
    decisions = v6.apply_outer_policy(outer, selected)
    assert decisions[0]["changed"] is True
    assert decisions[0]["success_delta"] == -1.0
    # Outer harm is reported, never fed back into threshold selection.
    assert selected["reason"] == "selected_on_inner_crossfit_only"


def test_ineligible_inner_guard_forces_baseline_on_outer(monkeypatch) -> None:
    monkeypatch.setattr(v6, "BOOTSTRAP_SAMPLES", 100)
    inner = [row(f"inner-{index}", [0.0, 0.3], [1.0, 0.0]) for index in range(40)]
    selected = v6.select_inner_guard(inner)
    assert selected["enabled"] is False
    decisions = v6.apply_outer_policy(
        [row("outer", [0.0, 10.0], [0.0, 1.0])], selected
    )
    assert decisions[0]["selected_index"] == 0
    assert decisions[0]["changed"] is False


def test_unregistered_outer_threshold_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="unregistered"):
        v6.apply_outer_policy(
            [row("outer", [0.0, 1.0], [0.0, 1.0])],
            {"enabled": True, "gain_margin": 0.123},
        )
