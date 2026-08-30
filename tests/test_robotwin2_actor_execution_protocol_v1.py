from __future__ import annotations

import ast
import copy
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "robotwin2_actor_execution_protocol_v1.py"
SPEC = importlib.util.spec_from_file_location("actor_execution_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def resign(value: dict) -> dict:
    changed = copy.deepcopy(value)
    changed.pop("logical_sha256", None)
    changed["logical_sha256"] = protocol.canonical_sha256(changed)
    return changed


def assert_prefix_mask(mask: list[bool], expected_steps: int) -> None:
    assert len(mask) == protocol.NATIVE_CHUNK_STEPS
    assert all(type(item) is bool for item in mask)
    assert mask == [
        index < expected_steps for index in range(protocol.NATIVE_CHUNK_STEPS)
    ]


def test_module_is_independent_of_all_production_modules() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "collections",
        "copy",
        "hashlib",
        "json",
        "types",
        "typing",
    }


@pytest.mark.parametrize(
    ("stride", "digest", "method", "protocol_id"),
    [
        (
            5,
            "6a3193c79a6f5f88738559711916b68fcfd795f1944bf5fe0cbae31736e98ce8",
            protocol.METHOD_EXECUTE5,
            "execute5_replan",
        ),
        (
            50,
            "4d9401a2777b9f42b6036cf6ba18729b4fa30c1d88d8f8e11174c47bc9cc65d4",
            protocol.METHOD_EXECUTE50,
            "execute50_native",
        ),
    ],
)
def test_protocol_identities_are_literal_canonical_and_strictly_validated(
    stride: int, digest: str, method: str, protocol_id: str
) -> None:
    value = protocol.execution_protocol(stride)
    assert value["format"] == protocol.FORMAT
    assert value["protocol_id"] == protocol_id
    assert value["actor_report_method"] == method
    assert value["logical_sha256"] == digest
    unsigned = dict(value)
    unsigned.pop("logical_sha256")
    assert protocol.canonical_sha256(unsigned) == digest
    assert protocol.validate_execution_protocol(
        value,
        expected_stride=stride,
        expected_actor_report_method=method,
    ) == value


def test_protocol_constants_freeze_shared_environment_semantics() -> None:
    for stride in protocol.LEGAL_STRIDES:
        value = protocol.execution_protocol(stride)
        assert value["task"] == "move_can_pot"
        assert value["bodies"] == [
            "aloha-agilex",
            "arx-x5",
            "franka",
            "piper",
            "ur5",
        ]
        assert value["conditions"] == ["clean", "randomized"]
        assert value["max_steps"] == 200
        assert type(value["max_steps"]) is int
        assert value["fps"] == 15
        assert type(value["fps"]) is int
        assert value["native_chunk_steps"] == 50
        assert value["candidate_count"] == 4
        assert value["supplement_horizons"] == [10, 25, 50, 100, 200]
        assert value["planning_contract"] == {
            "planned_steps": "min(stride,remaining_action_budget,horizon)",
            "planned_dt_seconds": "planned_steps/fps",
            "action_mask": (
                "length_native_chunk_steps_boolean_prefix_true_for_planned_steps"
            ),
            "mask_uses_planned_not_observed_executed_steps": True,
        }


def test_execute5_primary_grid_budgets_allocation_and_masks() -> None:
    value = protocol.execution_protocol(5)
    assert value["stride"] == 5
    assert value["query_indices"] == list(range(40))
    assert value["target_per_condition_query"] == 5
    assert value["primary_remaining_action_budgets"] == list(
        range(200, 0, -5)
    )
    assert len(value["primary_query_schedule"]) == 40
    for query, row in enumerate(value["primary_query_schedule"]):
        remaining = 200 - 5 * query
        assert row["query_index"] == query
        assert row["remaining_action_budget"] == remaining
        assert row["horizon"] == remaining
        assert row["planned_steps"] == 5
        assert row["planned_dt_seconds"] == pytest.approx(5 / 15)
        assert_prefix_mask(row["action_mask"], 5)


def test_execute50_primary_grid_budgets_allocation_and_masks() -> None:
    value = protocol.execution_protocol(50)
    assert value["stride"] == 50
    assert value["query_indices"] == [0, 1, 2, 3]
    assert value["target_per_condition_query"] == 50
    assert value["primary_remaining_action_budgets"] == [200, 150, 100, 50]
    assert len(value["primary_query_schedule"]) == 4
    for query, row in enumerate(value["primary_query_schedule"]):
        remaining = 200 - 50 * query
        assert row["query_index"] == query
        assert row["remaining_action_budget"] == remaining
        assert row["horizon"] == remaining
        assert row["planned_steps"] == 50
        assert row["planned_dt_seconds"] == pytest.approx(50 / 15)
        assert_prefix_mask(row["action_mask"], 50)


@pytest.mark.parametrize("stride", [5, 50])
def test_each_protocol_keeps_exactly_8000_five_body_branches(stride: int) -> None:
    value = protocol.execution_protocol(stride)
    accounting = value["branch_accounting"]
    expected = (
        len(value["bodies"])
        * len(value["conditions"])
        * len(value["query_indices"])
        * value["target_per_condition_query"]
        * value["candidate_count"]
    )
    assert expected == protocol.EXPECTED_TOTAL_BRANCHES == 8_000
    assert accounting["body_count"] == 5
    assert accounting["condition_count"] == 2
    assert accounting["query_count"] == len(value["query_indices"])
    assert accounting["target_per_condition_query"] == value[
        "target_per_condition_query"
    ]
    assert accounting["candidate_count"] == 4
    assert accounting["groups_per_body"] == 400
    assert accounting["branches_per_body"] == 1_600
    assert accounting["five_body_total_branches"] == 8_000


@pytest.mark.parametrize(
    ("stride", "expected_steps"),
    [
        (5, [5, 5, 5, 5, 5]),
        (50, [10, 25, 50, 50, 50]),
    ],
)
def test_supplement_horizons_materialize_min_rule_and_action_masks(
    stride: int, expected_steps: list[int]
) -> None:
    value = protocol.execution_protocol(stride)
    assert len(value["supplement_root_schedule"]) == 5
    for horizon, steps, frozen in zip(
        protocol.SUPPLEMENT_HORIZONS,
        expected_steps,
        value["supplement_root_schedule"],
        strict=True,
    ):
        live = protocol.supplement_action_plan(value, horizon)
        assert frozen["remaining_action_budget"] == horizon
        assert frozen["horizon"] == horizon
        assert frozen["planned_steps"] == steps
        assert frozen["planned_dt_seconds"] == pytest.approx(steps / 15)
        assert_prefix_mask(frozen["action_mask"], steps)
        for key in (
            "remaining_action_budget",
            "horizon",
            "planned_steps",
            "planned_dt_seconds",
            "action_mask",
        ):
            assert live[key] == frozen[key]


@pytest.mark.parametrize(
    ("stride", "remaining", "horizon", "expected_steps"),
    [
        (5, 200, 200, 5),
        (5, 3, 200, 3),
        (5, 200, 2, 2),
        (5, 4, 3, 3),
        (50, 200, 10, 10),
        (50, 25, 200, 25),
        (50, 30, 12, 12),
        (50, 200, 200, 50),
    ],
)
def test_action_plan_uses_min_stride_remaining_horizon_exactly(
    stride: int, remaining: int, horizon: int, expected_steps: int
) -> None:
    value = protocol.execution_protocol(stride)
    plan = protocol.action_plan(
        value,
        remaining_action_budget=remaining,
        horizon=horizon,
    )
    assert plan["execution_protocol_logical_sha256"] == value["logical_sha256"]
    assert plan["stride"] == stride
    assert plan["remaining_action_budget"] == remaining
    assert plan["horizon"] == horizon
    assert plan["planned_steps"] == min(stride, remaining, horizon)
    assert plan["planned_steps"] == expected_steps
    assert plan["planned_dt_seconds"] == pytest.approx(expected_steps / 15)
    assert_prefix_mask(plan["action_mask"], expected_steps)


@pytest.mark.parametrize("stride", [5, 50])
def test_primary_action_plan_matches_every_frozen_schedule_row(stride: int) -> None:
    value = protocol.execution_protocol(stride)
    for frozen in value["primary_query_schedule"]:
        live = protocol.primary_action_plan(value, frozen["query_index"])
        assert live["execution_protocol_logical_sha256"] == value["logical_sha256"]
        assert live["stride"] == stride
        for key, expected in frozen.items():
            assert live[key] == expected


def test_actor_report_method_mapping_is_exact_and_bidirectional() -> None:
    assert dict(protocol.ACTOR_REPORT_METHOD_BY_STRIDE) == {
        5: "actor_candidate0_execute5_replan",
        50: "actor_candidate0_execute50_native",
    }
    assert dict(protocol.STRIDE_BY_ACTOR_REPORT_METHOD) == {
        "actor_candidate0_execute5_replan": 5,
        "actor_candidate0_execute50_native": 50,
    }
    for stride, method in protocol.ACTOR_REPORT_METHOD_BY_STRIDE.items():
        assert protocol.execution_protocol_for_actor_report_method(method) == (
            protocol.execution_protocol(stride)
        )
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.execution_protocol_for_actor_report_method("execute5")


def test_protocol_results_and_validation_results_do_not_share_mutable_state() -> None:
    first = protocol.execution_protocol(5)
    first["query_indices"].append(999)
    first["primary_query_schedule"][0]["action_mask"][0] = False
    second = protocol.execution_protocol(5)
    assert second["query_indices"] == list(range(40))
    assert second["primary_query_schedule"][0]["action_mask"][0] is True
    validated = protocol.validate_execution_protocol(second)
    validated["supplement_horizons"].clear()
    assert protocol.validate_execution_protocol(second)["supplement_horizons"] == [
        10,
        25,
        50,
        100,
        200,
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("fps"),
        lambda value: value.__setitem__("extra", True),
        lambda value: value.__setitem__("fps", 15.0),
        lambda value: value.__setitem__("stride", 50),
        lambda value: value.__setitem__(
            "actor_report_method", "actor_candidate0_execute50_native"
        ),
        lambda value: value["query_indices"].reverse(),
        lambda value: value["primary_remaining_action_budgets"].__setitem__(0, 199),
        lambda value: value["primary_query_schedule"][0].__setitem__(
            "planned_dt_seconds", 0.0
        ),
        lambda value: value["primary_query_schedule"][0]["action_mask"].__setitem__(
            5, True
        ),
        lambda value: value["supplement_root_schedule"][0].__setitem__(
            "planned_steps", 4
        ),
        lambda value: value["branch_accounting"].__setitem__(
            "five_body_total_branches", 7999
        ),
    ],
)
def test_strict_validator_rejects_even_resigned_semantic_or_type_drift(
    mutation,
) -> None:
    changed = protocol.execution_protocol(5)
    mutation(changed)
    changed = resign(changed)
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.validate_execution_protocol(changed)


def test_strict_validator_rejects_bad_digest_non_mapping_and_wrong_expectation(
) -> None:
    value = protocol.execution_protocol(5)
    bad_digest = copy.deepcopy(value)
    bad_digest["logical_sha256"] = "0" * 64
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.validate_execution_protocol(bad_digest)
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.validate_execution_protocol([])
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.validate_execution_protocol(value, expected_stride=50)
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.validate_execution_protocol(
            value,
            expected_actor_report_method=protocol.METHOD_EXECUTE50,
        )
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.validate_execution_protocol(
            value, expected_actor_report_method="execute5"
        )


@pytest.mark.parametrize("invalid_stride", [True, 0, 1, 10, 25, 5.0, "5"])
def test_only_plain_integer_execute5_and_execute50_strides_are_legal(
    invalid_stride,
) -> None:
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.execution_protocol(invalid_stride)


@pytest.mark.parametrize(
    ("remaining", "horizon"),
    [
        (0, 10),
        (201, 10),
        (True, 10),
        (5.0, 10),
        (5, 0),
        (5, 201),
        (5, False),
        (5, 10.0),
    ],
)
def test_action_plan_rejects_terminal_out_of_range_or_non_integer_inputs(
    remaining, horizon
) -> None:
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.action_plan(
            protocol.execution_protocol(5),
            remaining_action_budget=remaining,
            horizon=horizon,
        )


@pytest.mark.parametrize("query", [-1, 40, True, 1.0, "1"])
def test_primary_plan_rejects_queries_outside_execute5_grid(query) -> None:
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.primary_action_plan(protocol.execution_protocol(5), query)


@pytest.mark.parametrize("horizon", [-1, 0, 5, 15, 49, 201, True, 10.0])
def test_supplement_plan_accepts_only_the_five_frozen_horizons(horizon) -> None:
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.supplement_action_plan(protocol.execution_protocol(50), horizon)


def test_supplement_plan_applies_remaining_budget_before_large_horizon() -> None:
    plan = protocol.supplement_action_plan(
        protocol.execution_protocol(50),
        200,
        remaining_action_budget=7,
    )
    assert plan["planned_steps"] == 7
    assert plan["planned_dt_seconds"] == pytest.approx(7 / 15)
    assert_prefix_mask(plan["action_mask"], 7)


def test_canonical_hash_is_key_order_invariant_and_rejects_nonfinite_json() -> None:
    assert protocol.canonical_sha256({"a": 1, "b": [2, 3]}) == (
        protocol.canonical_sha256({"b": [2, 3], "a": 1})
    )
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.canonical_sha256({"value": math.nan})
    with pytest.raises(protocol.ActorExecutionProtocolError):
        protocol.canonical_sha256({"value": math.inf})
