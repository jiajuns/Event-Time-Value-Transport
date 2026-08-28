from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plan_smolvla_piper_schema6_support import (  # noqa: E402
    aggregate_group_bounds,
    balanced_recovery_probability,
    build_plan,
    canonical_sha256,
    group_category_probabilities,
    joint_group_support_probability,
    write_json_new,
)


def test_source63_aggregate_bounds_are_exact_without_group_label_access() -> None:
    bounds = aggregate_group_bounds(
        groups=63, candidates=4, successes=34, failures=218
    )
    assert bounds == {
        "positive_groups": {"minimum": 9, "maximum": 34},
        "negative_groups": {"minimum": 55, "maximum": 63},
        "discordant_groups": {"minimum": 1, "maximum": 34},
    }


def test_fixed_validation50_success_gate_requires_every_group_discordant() -> None:
    categories = group_category_probabilities(34 / 252, 4)
    exact = joint_group_support_probability(
        50, categories, min_positive=50, min_negative=50
    )
    assert exact == pytest.approx(categories["discordant"] ** 50, rel=1e-12)
    assert exact < 2e-18


def test_default_plan_requires_expansion_and_does_not_impute_unknown_heads() -> None:
    plan = build_plan()
    assert plan["label_access"] == {
        "dataset_paths_accepted": False,
        "hdf5_opened": 0,
        "target_labels_read": False,
        "validation_labels_read": False,
        "evaluation_labels_read": False,
        "aggregate_counts_only": True,
    }
    assert plan["decision"]["training_authorized"] is False
    assert plan["decision"]["preregister_expansion_required"] is True
    assert plan["sample_size_plan"]["current_v2_physical_groups"] == 130
    assert plan["sample_size_plan"]["current_v2_adaptation_bucket_groups"] == 80
    assert plan["sample_size_plan"][
        "current_v2_training_groups_inside_adaptation"
    ] == 60
    assert plan["sample_size_plan"][
        "current_v2_internal_validation_groups_inside_adaptation"
    ] == 20
    assert plan["sample_size_plan"][
        "additional_groups_for_current_v2_physical_split"
    ] == 129
    assert plan["sample_size_plan"][
        "iid_minimum_target_validation_groups_for_success_support"
    ]["groups"] == 134
    assert plan["sample_size_plan"][
        "illustrative_minimum_internal_groups_for_balanced_exclusive_recovery"
    ]["groups"] == 30
    assert plan["sample_size_plan"]["iid_recommended_total_groups"] == 244
    assert plan["sample_size_plan"]["additional_groups_for_iid_recommended_plan"] == 243
    assert plan["sample_size_plan"][
        "wilson95_lower_rate_minimum_target_validation_groups_for_success_support"
    ]["groups"] == 177
    assert plan["sample_size_plan"][
        "wilson95_lower_rate_recommended_total_groups"
    ] == 287
    assert plan["sample_size_plan"][
        "operational_rounded_preregistration_target_groups"
    ] == 300
    assert plan["split_support"]["adaptation_internal_validation"][
        "event_all_classes_probability_identified"
    ] is False
    assert plan["split_support"]["adaptation_internal_validation"][
        "duration_observed_and_censored_probability_identified"
    ] is False
    assert plan["split_support"]["adaptation_internal_validation"][
        "conditional_recovery_probability_identified"
    ] is False
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256")
    assert recorded == canonical_sha256(unsigned)


def test_internal20_recovery_gate_is_weak_even_in_balanced_binary_scenario() -> None:
    assert balanced_recovery_probability(20, 10) == pytest.approx(
        184756 / 2**20
    )
    assert balanced_recovery_probability(20, 10) < 0.18


def test_invalid_aggregate_fails_before_any_artifact_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_plan(source_failures=217)
    assert not list(tmp_path.iterdir())


def test_receipt_is_create_once_and_read_only(tmp_path: Path) -> None:
    plan = build_plan()
    output = tmp_path / "support_plan.json"
    write_json_new(output, plan)
    assert json.loads(output.read_text(encoding="utf-8")) == plan
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        write_json_new(output, plan)
