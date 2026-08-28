from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openvla_etsf_structured_event_time_utility import (  # noqa: E402
    BASELINE_INDEX,
    DEPLOYMENT_CANDIDATE_COUNT,
    GUARD_MARGIN,
    UTILITY_FORMULA,
    V7StructuredEventTimeUtility,
    Z_STD_EPS,
    guarded_candidate_selection_numpy,
    guarded_candidate_selection_torch,
    structured_event_time_utility_from_predictions,
    structured_event_time_utility_numpy,
    structured_event_time_utility_torch,
    within_group_z_numpy,
    within_group_z_torch,
)


ARRAY_KEYS = (
    "destination_expected_progress",
    "immediate_next_event_expected_progress",
    "duration_selected_log_mean",
    "destination_z",
    "immediate_next_event_z",
    "duration_z",
    "utility",
)


def random_inputs(
    *, batch: int = 3, candidates: int = 4, events: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260827)
    destination = generator.normal(size=(batch, candidates, events))
    immediate = generator.normal(size=(batch, candidates, events))
    duration = generator.normal(size=(batch, candidates))
    # Explicitly non-uniform and non-monotonic: callers, not the module, own
    # cross-task event semantics.
    event_values = np.asarray([0.1, 0.8, -0.2, 1.4, 0.3])
    return destination, immediate, duration, event_values


def test_numpy_and_torch_decompositions_match_float64() -> None:
    destination, immediate, duration, event_values = random_inputs()
    numpy_result = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    torch_result = structured_event_time_utility_torch(
        torch.from_numpy(destination),
        torch.from_numpy(immediate),
        torch.from_numpy(duration),
        event_values=torch.from_numpy(event_values),
    )
    assert numpy_result["formula"] == torch_result["formula"] == UTILITY_FORMULA
    assert numpy_result["deployment_candidate_count"] == 4
    assert numpy_result["trainable_parameter_count"] == 0
    assert torch_result["trainable_parameter_count"] == 0
    np.testing.assert_array_equal(numpy_result["event_values"], event_values)
    np.testing.assert_array_equal(torch_result["event_values"].numpy(), event_values)
    for key in ARRAY_KEYS:
        np.testing.assert_allclose(
            numpy_result[key], torch_result[key].numpy(), rtol=1e-12, atol=1e-12
        )
    np.testing.assert_allclose(
        numpy_result["utility"],
        numpy_result["destination_z"]
        - numpy_result["immediate_next_event_z"]
        + numpy_result["duration_z"],
    )


def test_zero_parameter_module_matches_function_and_requires_event_values() -> None:
    destination, immediate, duration, event_values = random_inputs(batch=1)
    module = V7StructuredEventTimeUtility()
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    assert sum(buffer.numel() for buffer in module.buffers()) == 0
    result = module(
        torch.from_numpy(destination),
        torch.from_numpy(immediate),
        torch.from_numpy(duration),
        event_values=event_values,
    )
    expected = structured_event_time_utility_torch(
        torch.from_numpy(destination),
        torch.from_numpy(immediate),
        torch.from_numpy(duration),
        event_values=event_values,
    )
    assert torch.equal(result["utility"], expected["utility"])
    with pytest.raises(TypeError, match="event_values"):
        module(
            torch.from_numpy(destination),
            torch.from_numpy(immediate),
            torch.from_numpy(duration),
        )


def test_training_only_fifth_candidate_is_excluded_before_z_scoring() -> None:
    destination, immediate, duration, event_values = random_inputs(candidates=5)
    original = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    destination[..., 4, :] = 1e6
    immediate[..., 4, :] = -1e6
    duration[..., 4] = 1e9
    changed = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    assert original["utility"].shape[-1] == DEPLOYMENT_CANDIDATE_COUNT
    for key in ARRAY_KEYS:
        np.testing.assert_array_equal(original[key], changed[key])


def test_candidate_permutation_is_equivariant_within_deployment_group() -> None:
    destination, immediate, duration, event_values = random_inputs()
    original = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = structured_event_time_utility_numpy(
        destination[:, permutation],
        immediate[:, permutation],
        duration[:, permutation],
        event_values=event_values,
    )
    for key in ARRAY_KEYS:
        np.testing.assert_allclose(permuted[key], original[key][:, permutation])


def test_event_vocabulary_permutation_with_explicit_values_is_invariant() -> None:
    destination, immediate, duration, event_values = random_inputs()
    original = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    permutation = np.asarray([3, 0, 4, 1, 2])
    permuted = structured_event_time_utility_numpy(
        destination[..., permutation],
        immediate[..., permutation],
        duration,
        event_values=event_values[permutation],
    )
    for key in ARRAY_KEYS:
        np.testing.assert_allclose(permuted[key], original[key], atol=1e-14)


def test_positive_affine_component_rescaling_does_not_change_utility() -> None:
    destination, immediate, duration, event_values = random_inputs()
    original = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    transformed = structured_event_time_utility_numpy(
        destination,
        immediate,
        7.5 * duration - 19.0,
        event_values=3.25 * event_values + 41.0,
    )
    np.testing.assert_allclose(
        original["utility"], transformed["utility"], rtol=1e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        original["destination_z"], transformed["destination_z"], atol=2e-12
    )
    np.testing.assert_allclose(
        original["immediate_next_event_z"],
        transformed["immediate_next_event_z"],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        original["duration_z"], transformed["duration_z"], atol=2e-12
    )


def test_zero_variance_terms_contribute_exactly_zero() -> None:
    destination = np.zeros((2, 4, 3))
    immediate = np.zeros((2, 4, 3))
    duration = np.full((2, 4), 7.0)
    result = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=[4.0, -2.0, 9.0]
    )
    for key in ("destination_z", "immediate_next_event_z", "duration_z", "utility"):
        np.testing.assert_array_equal(result[key], np.zeros((2, 4)))
    torch_z = within_group_z_torch(torch.full((2, 4), 7.0))
    numpy_z = within_group_z_numpy(np.full((2, 4), 7.0))
    assert torch.equal(torch_z, torch.zeros_like(torch_z))
    np.testing.assert_array_equal(numpy_z, np.zeros((2, 4)))


def test_near_constant_std_at_or_below_frozen_epsilon_contributes_zero() -> None:
    assert Z_STD_EPS == 1e-8
    # Population std is sqrt(5)*1e-9/2 < 1e-8.
    values = np.asarray([[1.0, 1.0 + 1e-9, 1.0 - 1e-9, 1.0 + 2e-9]])
    numpy_z = within_group_z_numpy(values)
    torch_z = within_group_z_torch(torch.from_numpy(values))
    np.testing.assert_array_equal(numpy_z, np.zeros((1, 4)))
    assert torch.equal(torch_z, torch.zeros_like(torch_z))

    active = np.asarray([[1.0, 1.0 + 1e-6, 1.0 - 1e-6, 1.0 + 2e-6]])
    assert bool((within_group_z_numpy(active) != 0.0).any())


def test_nonconstant_z_has_population_zero_mean_and_unit_variance() -> None:
    values = np.asarray([[1.0, 2.0, 4.0, 9.0], [-3.0, 0.0, 2.0, 8.0]])
    numpy_z = within_group_z_numpy(values)
    torch_z = within_group_z_torch(torch.from_numpy(values))
    np.testing.assert_allclose(numpy_z.mean(axis=-1), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.mean(numpy_z**2, axis=-1), 1.0, atol=1e-15)
    np.testing.assert_allclose(numpy_z, torch_z.numpy(), atol=1e-15)


def test_fixed_guard_uses_only_candidate_zero_and_exact_margin_contract() -> None:
    utility = np.asarray(
        [
            [0.0, 0.0499, -2.0, -3.0],
            [0.0, 0.0501, -2.0, -3.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    numpy_result = guarded_candidate_selection_numpy(utility)
    torch_result = guarded_candidate_selection_torch(torch.from_numpy(utility))
    assert numpy_result["baseline_index"] == BASELINE_INDEX == 0
    assert numpy_result["guard_margin"] == GUARD_MARGIN == 0.05
    np.testing.assert_array_equal(numpy_result["selected_index"], [0, 1, 0])
    np.testing.assert_array_equal(numpy_result["accepted"], [False, True, False])
    for key in ("proposed_index", "selected_index", "score_margin", "accepted"):
        np.testing.assert_array_equal(numpy_result[key], torch_result[key].numpy())
    with pytest.raises(TypeError):
        guarded_candidate_selection_numpy(utility, fallback_index=2)
    with pytest.raises(TypeError):
        guarded_candidate_selection_torch(
            torch.from_numpy(utility), minimum_score_margin=0.0
        )


def test_guard_is_equivariant_to_alternative_candidate_permutation() -> None:
    utility = np.asarray([[0.0, 0.4, -0.2, 1.2]])
    original = guarded_candidate_selection_numpy(utility)
    permutation = np.asarray([0, 3, 1, 2])
    permuted = guarded_candidate_selection_numpy(utility[:, permutation])
    assert int(permutation[permuted["selected_index"]][0]) == int(
        original["selected_index"][0]
    )
    assert bool(permuted["accepted"][0]) is True


def test_plugin_mapping_adapter_reads_predictions_but_not_labels() -> None:
    destination, immediate, duration, event_values = random_inputs(batch=1)

    class ExplodesIfRead:
        def __array__(self, *_: object, **__: object) -> np.ndarray:
            raise AssertionError("label was read")

    class AuditedPredictions(dict[str, object]):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.read: list[str] = []

        def __getitem__(self, key: str) -> object:
            self.read.append(key)
            return super().__getitem__(key)

    predictions = AuditedPredictions(
        next_reached_event_logits=torch.from_numpy(destination),
        next_event_logits=torch.from_numpy(immediate),
        duration_selected_log_mean=torch.from_numpy(duration),
        success=ExplodesIfRead(),
        labels=ExplodesIfRead(),
    )
    result = structured_event_time_utility_from_predictions(
        predictions, event_values=event_values
    )
    assert result["utility"].shape == (1, 4)
    assert predictions.read == [
        "next_reached_event_logits",
        "next_event_logits",
        "duration_selected_log_mean",
    ]
    forbidden = {"success", "label", "labels", "target", "outcome"}
    for function in (
        structured_event_time_utility_numpy,
        structured_event_time_utility_torch,
        structured_event_time_utility_from_predictions,
        guarded_candidate_selection_numpy,
        guarded_candidate_selection_torch,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_event_values_are_mandatory_explicit_and_may_be_nonmonotonic() -> None:
    destination, immediate, duration, event_values = random_inputs(batch=1)
    with pytest.raises(TypeError, match="event_values"):
        structured_event_time_utility_numpy(destination, immediate, duration)
    with pytest.raises((TypeError, ValueError)):
        structured_event_time_utility_numpy(
            destination, immediate, duration, event_values=None
        )
    result = structured_event_time_utility_numpy(
        destination, immediate, duration, event_values=event_values
    )
    assert np.isfinite(result["utility"]).all()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("too_few_candidates", "at least four"),
        ("duration_shape", "duration log-mean"),
        ("event_values", "length-E"),
        ("nonfinite", "finite"),
    ],
)
def test_invalid_contracts_fail_closed(mutation: str, match: str) -> None:
    destination, immediate, duration, event_values = random_inputs(batch=1)
    if mutation == "too_few_candidates":
        destination = destination[:, :3]
        immediate = immediate[:, :3]
        duration = duration[:, :3]
    elif mutation == "duration_shape":
        duration = duration[:, :3]
    elif mutation == "event_values":
        event_values = event_values[:3]
    elif mutation == "nonfinite":
        destination[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        structured_event_time_utility_numpy(
            destination, immediate, duration, event_values=event_values
        )
