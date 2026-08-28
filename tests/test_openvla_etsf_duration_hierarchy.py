from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "openvla_etsf_duration_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("duration_hierarchy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
duration_hierarchy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(duration_hierarchy)


def _training_arrays() -> dict[str, np.ndarray]:
    # exact eligible: event=3,body=3, n=20
    # event fallback: event=0 has n=20, but exact 0:0 has only n=3
    # body fallback: body=2 has n=20, but its two events each have n=10
    events = np.asarray([0] * 3 + [0] * 17 + [1] * 10 + [2] * 10 + [3] * 20)
    bodies = np.asarray([0] * 3 + [1] * 17 + [2] * 20 + [3] * 20)
    count = len(events)
    return {
        "duration": np.arange(1, count + 1, dtype=np.float64),
        "duration_observed": np.ones(count, dtype=bool),
        "current_event_id": events,
        "body_id": bodies,
        "logical_group": np.asarray([f"group-{index:03d}" for index in range(count)]),
        "split_role": np.asarray(["outer_training"] * count),
    }


def _fit(arrays: dict[str, np.ndarray] | None = None) -> dict:
    return duration_hierarchy.fit_duration_hierarchy(
        **(arrays or _training_arrays()), owner_fold_id=2
    )


def test_fixed_lookup_uses_only_sources_with_support_at_least_twenty() -> None:
    contract = _fit()
    applied = duration_hierarchy.apply_duration_hierarchy(
        contract,
        current_event_id=np.asarray([3, 0, 1, 99]),
        body_id=np.asarray([3, 0, 2, 99]),
        expected_training_logical_groups_sha256=contract[
            "outer_training_logical_groups_sha256"
        ],
    )
    assert applied["source_kind"].tolist() == [
        "event_body",
        "event",
        "body",
        "global",
    ]
    assert applied["source_key"].tolist() == ["3:3", "0", "2", "global"]
    assert applied["source_support"].tolist() == [20, 20, 20, 60]
    assert applied["minimum_applied_source_support"] == 20
    assert applied["target_fold_labels_used_for_fit"] is False
    assert contract["current_event_field"] == "current_event_id"
    assert contract["clock_event_proxy_allowed"] is False


def test_sparse_exact_cell_is_recorded_but_never_applied() -> None:
    contract = _fit()
    sparse = contract["sources"]["event_body"]["0:0"]
    assert sparse["support"] == 3
    assert sparse["eligible"] is False
    applied = duration_hierarchy.apply_duration_hierarchy(
        contract, current_event_id=[0], body_id=[0]
    )
    assert applied["source_kind"].tolist() == ["event"]
    assert applied["source_support"].tolist() == [20]


def test_fit_rejects_outer_holdout_rows_and_insufficient_global_support() -> None:
    arrays = _training_arrays()
    arrays["split_role"][7] = "outer_holdout"
    with pytest.raises(RuntimeError, match="non-outer-training"):
        _fit(arrays)

    count = 19
    with pytest.raises(RuntimeError, match="fixed support of 20"):
        duration_hierarchy.fit_duration_hierarchy(
            duration=np.ones(count),
            duration_observed=np.ones(count, dtype=bool),
            current_event_id=np.zeros(count, dtype=np.int64),
            body_id=np.zeros(count, dtype=np.int64),
            logical_group=[f"g{index}" for index in range(count)],
            split_role=["outer_training"] * count,
            owner_fold_id=0,
        )


def test_unknown_ids_use_global_but_invalid_ids_fail_closed() -> None:
    contract = _fit()
    applied = duration_hierarchy.apply_duration_hierarchy(
        contract, current_event_id=[777], body_id=[888]
    )
    assert applied["source_kind"].tolist() == ["global"]
    with pytest.raises(ValueError, match="non-negative integer ids"):
        duration_hierarchy.apply_duration_hierarchy(
            contract, current_event_id=[-1], body_id=[0]
        )
    with pytest.raises(ValueError, match="aligned"):
        duration_hierarchy.apply_duration_hierarchy(
            contract, current_event_id=[0, 1], body_id=[0]
        )


def test_training_group_hash_and_contract_signature_are_fail_closed() -> None:
    contract = _fit()
    with pytest.raises(RuntimeError, match="owner-training group SHA mismatch"):
        duration_hierarchy.apply_duration_hierarchy(
            contract,
            current_event_id=[3],
            body_id=[3],
            expected_training_logical_groups_sha256="0" * 64,
        )

    tampered = dict(contract)
    tampered["outer_training_logical_groups"] = list(
        contract["outer_training_logical_groups"]
    ) + ["injected-holdout"]
    with pytest.raises(RuntimeError, match="signature mismatch"):
        duration_hierarchy.validate_duration_hierarchy_contract(tampered)

    sparse_enabled = {
        **contract,
        "sources": {
            **contract["sources"],
            "event_body": {
                **contract["sources"]["event_body"],
                "0:0": {
                    **contract["sources"]["event_body"]["0:0"],
                    "eligible": True,
                },
            },
        },
    }
    sparse_enabled["contract_sha256"] = duration_hierarchy.canonical_sha256(
        {
            key: value
            for key, value in sparse_enabled.items()
            if key != "contract_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="source provenance is invalid"):
        duration_hierarchy.validate_duration_hierarchy_contract(sparse_enabled)

    source_hash_changed = {
        **contract,
        "sources": {
            **contract["sources"],
            "global": {
                **contract["sources"]["global"],
                "source_training_logical_groups_sha256": "f" * 64,
            },
        },
    }
    source_hash_changed["contract_sha256"] = duration_hierarchy.canonical_sha256(
        {
            key: value
            for key, value in source_hash_changed.items()
            if key != "contract_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="source provenance is invalid"):
        duration_hierarchy.validate_duration_hierarchy_contract(source_hash_changed)


def test_serialization_and_fit_are_stable_under_input_permutation() -> None:
    arrays = _training_arrays()
    first = _fit(arrays)
    order = np.arange(len(arrays["duration"]))[::-1]
    second = _fit({key: value[order] for key, value in arrays.items()})
    assert first["contract_sha256"] == second["contract_sha256"]
    serialized = duration_hierarchy.serialize_duration_hierarchy(first)
    assert serialized == duration_hierarchy.serialize_duration_hierarchy(second)
    restored = duration_hierarchy.deserialize_duration_hierarchy(serialized)
    assert restored == first


def test_logical_group_alignment_is_mandatory() -> None:
    arrays = _training_arrays()
    arrays["logical_group"] = arrays["logical_group"][:-1]
    with pytest.raises(ValueError, match="aligned"):
        _fit(arrays)
