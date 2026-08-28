from __future__ import annotations

import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from actor_agnostic_structured_event_time_plugin import (  # noqa: E402
    ActorAgnosticStructuredEventTimePlugin,
    DEFAULT_REQUIRED_CONTRACTS,
)


def predictions(*, candidates: int = 7, events: int = 3) -> dict[str, torch.Tensor]:
    logits = torch.zeros(2, candidates, events, dtype=torch.float64)
    duration = torch.arange(candidates, dtype=torch.float64).expand(2, -1).clone()
    uncertainty = torch.full((2, candidates), 0.05, dtype=torch.float64)
    return {
        "next_reached_event_logits": logits.clone(),
        "next_event_logits": logits.clone(),
        "duration_selected_log_mean": duration,
        "total_uncertainty": uncertainty,
    }


def contracts(**updates: bool) -> dict[str, bool]:
    value = {name: True for name in DEFAULT_REQUIRED_CONTRACTS}
    value.update(updates)
    return value


def plugin() -> ActorAgnosticStructuredEventTimePlugin:
    return ActorAgnosticStructuredEventTimePlugin(
        event_values_registry={
            "task_a": (0.0, 0.4, 1.0),
            "task_nonordinal": (0.5, -1.0, 0.25),
        },
        maximum_total_uncertainty=0.25,
    )


def run(
    model: ActorAgnosticStructuredEventTimePlugin,
    values: dict[str, torch.Tensor],
    *,
    slots=(5, 2, 4, 1),
    fallback=2,
    valid: torch.Tensor | None = None,
    contract: dict[str, bool] | None = None,
):
    if valid is None:
        valid = torch.ones_like(values["duration_selected_log_mean"], dtype=torch.bool)
    return model(
        values,
        task="task_a",
        deployment_slots=slots,
        fallback_slot=fallback,
        candidate_valid_mask=valid,
        contract_guard=contracts() if contract is None else contract,
    )


def test_module_has_zero_parameters_buffers_and_explicit_task_registry() -> None:
    model = plugin()
    assert sum(parameter.numel() for parameter in model.parameters()) == 0
    assert sum(buffer.numel() for buffer in model.buffers()) == 0
    assert model.state_dict() == {}
    result = run(model, predictions())
    assert result["trainable_parameter_count"] == 0
    assert result["buffer_count"] == 0
    assert result["event_values"] == (0.0, 0.4, 1.0)
    assert len(result["event_values_registry_sha256"]) == 64
    assert isinstance(model.event_values_registry, MappingProxyType)
    with pytest.raises(TypeError):
        model.event_values_registry["task_a"] = (0.0, 1.0)  # type: ignore[index]


def test_deployment_plugin_detaches_actor_prediction_graph() -> None:
    value = predictions()
    for tensor in value.values():
        tensor.requires_grad_(True)
    result = run(plugin(), value)
    assert result["utility"].requires_grad is False
    assert result["proposed_uncertainty"].requires_grad is False


def test_explicit_native_fallback_is_moved_to_local_candidate_zero() -> None:
    result = run(plugin(), predictions())
    assert result["deployment_slots_requested"] == (5, 2, 4, 1)
    assert result["local_to_actor_slot"] == (2, 5, 4, 1)
    assert result["fallback_actor_slot"] == 2
    assert result["fallback_local_slot"] == 0
    assert result["selected_actor_slot"].tolist() == [5, 5]
    assert result["accepted"].tolist() == [True, True]


def test_only_four_declared_slots_participate_even_with_more_candidates() -> None:
    value = predictions(candidates=8)
    value["duration_selected_log_mean"][:, 7] = 1e9
    result = run(plugin(), value, slots=(0, 1, 2, 3), fallback=0)
    assert result["local_to_actor_slot"] == (0, 1, 2, 3)
    assert result["selected_actor_slot"].tolist() == [3, 3]
    assert result["utility"].shape == (2, 4)


def test_openvla_and_simulated_smolvla_prediction_mappings_have_parity() -> None:
    base = predictions()
    openvla = {**base, "openvla_hidden_4096": torch.randn(2, 4096)}
    smolvla = {**base, "smolvla_shared_prefix_960": torch.randn(2, 960)}
    left = run(plugin(), openvla)
    right = run(plugin(), smolvla)
    for key in (
        "utility",
        "selected_actor_slot",
        "proposed_actor_slot",
        "score_margin",
        "accepted",
    ):
        assert torch.equal(left[key], right[key])
    for key in (
        "destination_expected_progress",
        "immediate_next_event_expected_progress",
        "duration_selected_log_mean",
        "destination_z",
        "immediate_next_event_z",
        "duration_z",
    ):
        assert torch.equal(left["decomposition"][key], right["decomposition"][key])


def test_candidate_permutation_is_equivariant_in_actor_slot_space() -> None:
    value = predictions()
    original = run(plugin(), value)
    permutation = [3, 5, 0, 2, 1, 4, 6]
    permuted = {
        key: tensor[:, permutation, ...]
        if tensor.ndim == 3
        else tensor[:, permutation]
        for key, tensor in value.items()
    }
    old_to_new = {old: new for new, old in enumerate(permutation)}
    result = run(
        plugin(),
        permuted,
        slots=tuple(old_to_new[slot] for slot in (5, 2, 4, 1)),
        fallback=old_to_new[2],
    )
    selected_old_slots = [permutation[index] for index in result["selected_actor_slot"].tolist()]
    assert selected_old_slots == original["selected_actor_slot"].tolist()
    assert torch.equal(result["utility"], original["utility"])


def test_invalid_and_nan_alternatives_are_excluded_with_reason_codes() -> None:
    value = predictions()
    value["duration_selected_log_mean"][:, 1] = 10.0
    valid = torch.ones_like(value["duration_selected_log_mean"], dtype=torch.bool)
    valid[:, 5] = False  # Highest-duration declared slot.
    value["next_event_logits"][:, 4, 0] = torch.nan
    result = run(plugin(), value, valid=valid)
    assert result["selected_actor_slot"].tolist() == [1, 1]
    assert result["input_sanitized_mask"][:, 2].tolist() == [True, True]
    assert all("invalid_candidate_excluded" in row for row in result["reason_codes"])
    assert all("candidate_invalid" in row[1] for row in result["candidate_reason_codes"])
    assert all(
        "candidate_prediction_nonfinite" in row[2]
        for row in result["candidate_reason_codes"]
    )


def test_invalid_or_nonfinite_fallback_is_rejected_as_no_safe_action() -> None:
    value = predictions()
    valid = torch.ones_like(value["duration_selected_log_mean"], dtype=torch.bool)
    valid[:, 2] = False
    with pytest.raises(RuntimeError, match="fallback candidate is invalid"):
        run(plugin(), value, valid=valid)
    value = predictions()
    value["duration_selected_log_mean"][:, 2] = torch.nan
    with pytest.raises(RuntimeError, match="fallback prediction is non-finite"):
        run(plugin(), value)


def test_uncertainty_guard_falls_back_and_emits_reason() -> None:
    value = predictions()
    value["total_uncertainty"][:, 5] = 0.5
    result = run(plugin(), value)
    assert result["proposed_actor_slot"].tolist() == [5, 5]
    assert result["selected_actor_slot"].tolist() == [2, 2]
    assert result["accepted"].tolist() == [False, False]
    assert all("uncertainty_above_guard" in row for row in result["reason_codes"])
    value["total_uncertainty"][:, 5] = torch.nan
    result = run(plugin(), value)
    assert all("nonfinite_uncertainty" in row for row in result["reason_codes"])


def test_contract_guard_is_explicit_and_missing_or_false_is_fail_closed() -> None:
    missing = contracts()
    missing.pop("clock_contract_matched")
    result = run(plugin(), predictions(), contract=missing)
    assert result["selected_actor_slot"].tolist() == [2, 2]
    assert all(
        "contract_missing:clock_contract_matched" in row
        for row in result["reason_codes"]
    )
    failed = contracts(predicate_contract_matched=False)
    result = run(plugin(), predictions(), contract=failed)
    assert result["selected_actor_slot"].tolist() == [2, 2]
    assert all(
        "contract_failed:predicate_contract_matched" in row
        for row in result["reason_codes"]
    )
    with pytest.raises(TypeError, match="explicit mapping"):
        value = predictions()
        plugin()(
            value,
            task="task_a",
            deployment_slots=(0, 1, 2, 3),
            fallback_slot=0,
            candidate_valid_mask=torch.ones_like(
                value["duration_selected_log_mean"], dtype=torch.bool
            ),
            contract_guard=None,  # type: ignore[arg-type]
        )


def test_unknown_task_and_mismatched_event_registry_fail_closed() -> None:
    value = predictions()
    model = plugin()
    valid = torch.ones_like(value["duration_selected_log_mean"], dtype=torch.bool)
    with pytest.raises(KeyError, match="unknown task"):
        model(
            value,
            task="not_registered",
            deployment_slots=(0, 1, 2, 3),
            fallback_slot=0,
            candidate_valid_mask=valid,
            contract_guard=contracts(),
        )
    wrong = ActorAgnosticStructuredEventTimePlugin(
        event_values_registry={"task_a": (0.0, 1.0)},
        maximum_total_uncertainty=1.0,
    )
    with pytest.raises(ValueError, match="event_values length"):
        run(wrong, value)


@pytest.mark.parametrize(
    "slots,fallback,match",
    [
        ((0, 1, 2), 0, "four unique"),
        ((0, 1, 1, 2), 0, "four unique"),
        ((0, 1, 2, 99), 0, "outside"),
        ((0, 1, 2, 3), 4, "one of the four"),
    ],
)
def test_deployment_slot_contract_rejects_malformed_layout(
    slots, fallback, match
) -> None:
    with pytest.raises(ValueError, match=match):
        run(plugin(), predictions(), slots=slots, fallback=fallback)


def test_complete_decomposition_and_fallback_reason_are_returned() -> None:
    value = predictions()
    value["duration_selected_log_mean"][:, 2] = 100.0
    result = run(plugin(), value)
    assert result["selected_actor_slot"].tolist() == [2, 2]
    assert all("utility_prefers_fallback" in row for row in result["reason_codes"])
    assert {
        "destination_expected_progress",
        "immediate_next_event_expected_progress",
        "duration_selected_log_mean",
        "destination_z",
        "immediate_next_event_z",
        "duration_z",
        "utility",
    }.issubset(result["decomposition"])
