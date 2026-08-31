from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "nested_n4_n8_runner",
    SCRIPTS / "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _current() -> np.ndarray:
    value = np.zeros(16, dtype=np.float32)
    value[6] = 1.0
    value[14] = 1.0
    return value


def _raw16() -> np.ndarray:
    rows = []
    for index in range(runner.RAW_PROPOSAL_COUNT):
        chunk = np.repeat(_current()[None], 8, axis=0)
        chunk[:, 0] = 0.01 * index
        chunk[:, 1] = 0.002 * (index**2)
        chunk[:, 8] = -0.007 * index
        chunk[:, 9] = 0.001 * (index**2)
        chunk[:, 7] = (index % 3) / 2.0
        chunk[:, 15] = (index % 5) / 4.0
        rows.append(chunk)
    return np.stack(rows).astype(np.float32)


def _normalizer(
    body: str | None = None, *, first_dimension_std: float = 1.0
) -> dict[str, object]:
    action_std = [1.0] * runner.collector.CANONICAL_ACTION_DIM
    action_std[0] = first_dimension_std
    base = {
        "format": runner.SOURCE_ACTION_NORMALIZER_FORMAT,
        "heldout_body": body or runner.BODIES[0],
        "canonical_action_schema": runner.collector.ACTION_SCHEMA,
        "normalization_fit_scope": "four_source_bodies_train_only",
        "heldout_rows_used": 0,
        "checkpoint_action_normalization_sha256": "a" * 64,
        "action_mean": [0.0] * runner.collector.CANONICAL_ACTION_DIM,
        "action_std": action_std,
        "normalization_clip": float(
            runner.shared_head.CROSS_BODY_STANDARDIZED_INPUT_CLIP
        ),
        "five_member_normalizers_bit_exact_equal": True,
        "selection_reads_heldout_labels_outcomes_or_critic_scores": False,
    }
    return {**base, "logical_sha256": runner.canonical_sha256(base)}


def _nested(
    normalizer: dict[str, object] | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    return runner.nested_pool_selection_audit(
        current_ee=_current(),
        raw_proposals=_raw16(),
        source_action_normalizer=normalizer or _normalizer(),
    )


def test_existing_independent_runs_do_not_authorize_pool_size_gain() -> None:
    audit = runner.existing_separate_n4_n8_comparability_audit()
    assert audit["same_body_condition_requested_seed_schedule"] is True
    assert audit["same_frozen_actor_candidate_zero_noise_identity"] is True
    assert audit["formal_n4_is_required_subset_of_current_n8"] is False
    assert audit["shared_cross_study_initial_raw_pool_commitment"] is False
    assert audit["direct_strong_causal_pool_size_comparison_authorized"] is False


def test_raw16_blind_fps_produces_exact_nested_prefix_and_candidate_zero() -> None:
    pools, audit = _nested()
    n4 = pools[runner.N4_CANDIDATE_COUNT]
    n8 = pools[runner.N8_CANDIDATE_COUNT]
    assert np.array_equal(n4, n8[: runner.N4_CANDIDATE_COUNT])
    assert np.array_equal(n4[0], _raw16()[0])
    assert audit["ordered_fps_raw_indices_n4"] == audit[
        "ordered_fps_raw_indices_n8"
    ][: runner.N4_CANDIDATE_COUNT]
    assert audit["ordered_fps_raw_indices_n8"][0] == 0
    assert audit[
        "selection_reads_outcomes_events_labels_or_critic_scores"
    ] is False
    assert audit["format"] == runner.NESTED_POOL_AUDIT_FORMAT
    assert audit["executed_effect_horizon_actions"] == runner.ACTION_EXEC_STEPS
    assert audit["ordered_selection_metrics_n8"] == list(
        runner.MIXED_SELECTION_METRIC_ORDER
    )
    assert audit["ordered_selection_metrics_n4"] == list(
        runner.MIXED_SELECTION_METRIC_ORDER[: runner.N4_CANDIDATE_COUNT]
    )
    assert audit["raw16_flow_noise_contract_sha256"] == runner.canonical_sha256(
        runner.nested_pool_contract()["raw16_flow_noise_contract"]
    )
    assert audit["source_action_normalizer"] == _normalizer()
    assert audit["source_action_normalizer_logical_sha256"] == _normalizer()[
        "logical_sha256"
    ]
    runner.validate_nested_pool_audit(audit)

    legacy = copy.deepcopy(audit)
    legacy.pop("state_action_frame_contract")
    legacy["audit_sha256"] = runner.canonical_sha256(
        {key: value for key, value in legacy.items() if key != "audit_sha256"}
    )
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_nested_pool_audit(legacy)

    tampered = copy.deepcopy(audit)
    tampered["ordered_fps_raw_indices_n4"] = list(reversed(
        tampered["ordered_fps_raw_indices_n4"]
    ))
    tampered["audit_sha256"] = runner.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "audit_sha256"}
    )
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_nested_pool_audit(tampered)

    tampered_noise = copy.deepcopy(audit)
    tampered_noise["raw16_flow_noise_contract_sha256"] = "f" * 64
    tampered_noise["audit_sha256"] = runner.canonical_sha256(
        {
            key: value
            for key, value in tampered_noise.items()
            if key != "audit_sha256"
        }
    )
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_nested_pool_audit(tampered_noise)


def test_execute50_binding_propagates_effect_horizon_but_not_first5_noise(
    tmp_path: Path,
) -> None:
    def bind(stride: int) -> None:
        protocol = runner.formal.actor_execution.execution_protocol(stride)
        path = tmp_path / f"actor_execution_protocol_{stride}.json"
        path.write_text(
            json.dumps(protocol, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        runner.configure_actor_execution_protocol(
            protocol,
            path=path,
            file_sha256=runner.sha256_file(path),
            path_root=tmp_path,
        )

    try:
        bind(50)
        assert runner.ACTION_EXEC_STEPS == 50
        assert runner.formal.ACTION_EXEC_STEPS == 50
        assert runner.pool_runner.ACTION_EXEC_STEPS == 50
        raw = np.repeat(_raw16()[:, :1], 50, axis=1)
        pools, audit = runner.nested_pool_selection_audit(
            current_ee=_current(),
            raw_proposals=raw,
            source_action_normalizer=_normalizer(),
        )
        assert pools[runner.N8_CANDIDATE_COUNT].shape == (8, 50, 16)
        assert audit["executed_effect_horizon_actions"] == 50
        assert audit["canonical_effect_embedding_shape"] == [16, 50 * 14]
        noise = runner.pool_runner.postformal_noise_contract(8, 16)
        assert noise["conditional_translation_prefix_steps"] == 5
        assert runner.nested_pool_contract()["raw16_flow_noise_contract"] == noise
    finally:
        bind(5)


def test_source_normalized_fps_reverses_raw_unit_dominance_toward_translation() -> None:
    effects = np.zeros((16, runner.ACTION_EXEC_STEPS, 14), dtype=np.float32)
    # Raw physical units prefer one-radian rotation over two-centimetre motion.
    effects[1, :, 3] = 1.0
    effects[2, :, 0] = 0.02
    for index in range(3, 16):
        effects[index, :, 13] = 0.001 * index
    raw_embeddings = effects.reshape(16, -1)
    raw_indices = runner.pool_runner.greedy_farthest_point_indices(
        raw_embeddings, retain_count=2
    )
    assert raw_indices == [0, 1]

    std = np.ones(14, dtype=np.float32)
    std[0] = 0.005
    normalized = runner.pool_runner.source_train_normalized_effect_embeddings(
        effects,
        action_mean=np.zeros(14, dtype=np.float32),
        action_std=std,
        normalization_clip=float(
            runner.shared_head.CROSS_BODY_STANDARDIZED_INPUT_CLIP
        ),
    )
    normalized_indices = runner.pool_runner.greedy_farthest_point_indices(
        normalized, retain_count=2
    )
    assert normalized_indices == [0, 2]
    assert effects[normalized_indices[1], :, 0].mean() == pytest.approx(0.02)

    translation = (
        runner.pool_runner.source_train_normalized_translation_effect_embeddings(
            effects,
            action_mean=np.zeros(14, dtype=np.float32),
            action_std=std,
            normalization_clip=float(
                runner.shared_head.CROSS_BODY_STANDARDIZED_INPUT_CLIP
            ),
        )
    )
    translation_indices = runner.pool_runner.greedy_farthest_point_indices(
        translation, retain_count=2
    )
    assert translation_indices == [0, 2]


def test_mixed_metric_order_preserves_opposite_translation_then_full_coverage() -> None:
    translation = np.zeros((16, 2), dtype=np.float64)
    full = np.zeros((16, 3), dtype=np.float64)
    translation[1] = [10.0, 0.0]
    translation[2] = [-10.0, 0.0]
    full[:, :2] = translation
    # Raw 3 has no translation difference but provides a large orientation/
    # gripper-like direction.  It must enter only at the declared full slot.
    full[3, 2] = 100.0
    selected = runner.mixed_metric_farthest_point_indices(translation, full)
    assert selected[:4] == [0, 1, 2, 3]
    assert runner.MIXED_SELECTION_METRIC_ORDER == (
        "anchor_candidate_zero",
        "translation",
        "translation",
        "full",
        "translation",
        "translation",
        "translation",
        "full",
    )
    assert runner.nested_pool_contract()[
        "selection_reads_outcomes_events_labels_or_critic_scores"
    ] is False


def test_fold_fails_closed_when_one_member_normalizer_differs(
    tmp_path: Path,
) -> None:
    def write_checkpoint(path: Path, std0: float) -> None:
        mean = np.zeros(14, dtype=np.float32)
        std = np.ones(14, dtype=np.float32)
        std[0] = std0
        normalization_base = {
            "format": "etsf_five_body_canonical_train_only_normalization_v2",
            "canonical_action_schema": runner.collector.ACTION_SCHEMA,
            "canonical_action_schema_id": 0,
            "schema": {
                "mean": mean.astype(float).tolist(),
                "std": std.astype(float).tolist(),
            },
            "heldout_rows_used": 0,
        }
        normalization = {
            **normalization_base,
            "sha256": runner.canonical_sha256(normalization_base),
        }
        torch.save(
            {
                "action_normalization": normalization,
                "model": {
                    "action.action_mean": torch.from_numpy(mean[None]),
                    "action.action_std": torch.from_numpy(std[None]),
                },
            },
            path,
        )

    members = []
    for member in range(5):
        path = tmp_path / f"member-{member}.pt"
        write_checkpoint(path, 1.0)
        members.append({"member": member, "checkpoint": str(path)})
    fold = {"heldout_body": runner.BODIES[0], "members": members}
    binding = runner.inspect_fold_source_action_normalizer(fold)
    assert binding["five_member_normalizers_bit_exact_equal"] is True
    assert binding["heldout_rows_used"] == 0
    runner.validate_source_action_normalizer(
        binding, expected_heldout_body=runner.BODIES[0]
    )

    write_checkpoint(Path(members[-1]["checkpoint"]), 2.0)
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="five bootstrap members",
    ):
        runner.inspect_fold_source_action_normalizer(fold)


def test_audit_and_commitment_normalizer_sha_tampering_fails_closed() -> None:
    _pools, audit = _nested()
    changed_audit = copy.deepcopy(audit)
    changed_audit["source_action_normalizer"][
        "checkpoint_action_normalization_sha256"
    ] = "b" * 64
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_nested_pool_audit(changed_audit)

    expected = runner.evaluation_schedule()[0]
    reset = {"format": "etsf_robotwin2_observable_reset_snapshot_v2", "kind": "reset"}
    canonical = {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "kind": "canonical",
    }
    commitment_base = {
        "format": runner.INITIAL_COMMITMENT_FORMAT,
        "heldout_body": expected["heldout_body"],
        "condition": expected["condition"],
        "requested_seed": expected["requested_seed"],
        "resolved_seed": expected["requested_seed"],
        "nested_pool_audit": audit,
        "nested_pool_contract_sha256": runner.canonical_sha256(
            runner.nested_pool_contract()
        ),
        "raw16_flow_noise_contract_sha256": audit[
            "raw16_flow_noise_contract_sha256"
        ],
        "source_action_normalizer_logical_sha256": audit[
            "source_action_normalizer_logical_sha256"
        ],
        "reset_snapshot": reset,
        "reset_identity_sha256": runner.formal.reset_identity(reset),
        "canonical_query_snapshot": canonical,
        "canonical_query_identity_sha256": runner.formal.reset_identity(canonical),
        "candidate_generation_advanced_simulator": False,
        "frozen_before_any_method_execution": True,
    }
    commitment = {
        **commitment_base,
        "commitment_sha256": runner.canonical_sha256(commitment_base),
    }
    authority_sha = str(audit["source_action_normalizer_logical_sha256"])
    runner.validate_stored_initial_commitment(
        commitment,
        expected,
        expected_source_action_normalizer_logical_sha256=authority_sha,
    )
    changed_commitment = copy.deepcopy(commitment)
    changed_commitment["source_action_normalizer_logical_sha256"] = "c" * 64
    changed_commitment["commitment_sha256"] = runner.canonical_sha256(
        {
            key: value
            for key, value in changed_commitment.items()
            if key != "commitment_sha256"
        }
    )
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_stored_initial_commitment(
            changed_commitment,
            expected,
            expected_source_action_normalizer_logical_sha256=authority_sha,
        )

    changed_noise_commitment = copy.deepcopy(commitment)
    changed_noise_commitment["raw16_flow_noise_contract_sha256"] = "d" * 64
    changed_noise_commitment["commitment_sha256"] = runner.canonical_sha256(
        {
            key: value
            for key, value in changed_noise_commitment.items()
            if key != "commitment_sha256"
        }
    )
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.validate_stored_initial_commitment(
            changed_noise_commitment,
            expected,
            expected_source_action_normalizer_logical_sha256=authority_sha,
        )


def test_self_consistent_replacement_commitment_normalizer_is_rejected() -> None:
    expected = runner.evaluation_schedule()[0]
    _pools, authoritative_audit = _nested()
    _changed_pools, changed_audit = _nested(
        _normalizer(first_dimension_std=2.0)
    )
    reset = {"format": "etsf_robotwin2_observable_reset_snapshot_v2", "kind": "reset"}
    canonical = {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "kind": "canonical",
    }
    replacement_base = {
        "format": runner.INITIAL_COMMITMENT_FORMAT,
        "heldout_body": expected["heldout_body"],
        "condition": expected["condition"],
        "requested_seed": expected["requested_seed"],
        "resolved_seed": expected["requested_seed"],
        "nested_pool_audit": changed_audit,
        "nested_pool_contract_sha256": runner.canonical_sha256(
            runner.nested_pool_contract()
        ),
        "raw16_flow_noise_contract_sha256": changed_audit[
            "raw16_flow_noise_contract_sha256"
        ],
        "source_action_normalizer_logical_sha256": changed_audit[
            "source_action_normalizer_logical_sha256"
        ],
        "reset_snapshot": reset,
        "reset_identity_sha256": runner.formal.reset_identity(reset),
        "canonical_query_snapshot": canonical,
        "canonical_query_identity_sha256": runner.formal.reset_identity(canonical),
        "candidate_generation_advanced_simulator": False,
        "frozen_before_any_method_execution": True,
    }
    replacement = {
        **replacement_base,
        "commitment_sha256": runner.canonical_sha256(replacement_base),
    }
    runner.validate_nested_pool_audit(changed_audit)
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="stored initial commitment changed",
    ):
        runner.validate_stored_initial_commitment(
            replacement,
            expected,
            expected_source_action_normalizer_logical_sha256=str(
                authoritative_audit["source_action_normalizer_logical_sha256"]
            ),
        )


def test_schedule_uses_same_complete_seed_roster_and_rotates_three_arms() -> None:
    schedule = runner.evaluation_schedule()
    assert len(schedule) == len(runner.BODIES) * len(runner.CONDITIONS) * runner.SEED_COUNT
    for body in runner.BODIES:
        for condition in runner.CONDITIONS:
            rows = [
                row
                for row in schedule
                if row["heldout_body"] == body and row["condition"] == condition
            ]
            assert [row["requested_seed"] for row in rows] == [
                runner.SEED_BASE + index for index in range(runner.SEED_COUNT)
            ]
            assert all(set(row["method_order"]) == set(runner.METHODS) for row in rows)
            first_counts = Counter(row["method_order"][0] for row in rows)
            assert max(first_counts.values()) - min(first_counts.values()) <= 1


def _complete_outcome_rows() -> list[dict[str, object]]:
    rows = []
    for expected in runner.evaluation_schedule():
        row = {
            **expected,
            "pair_sha256": runner.canonical_sha256(expected),
        }
        for method in runner.METHODS:
            row[f"{method}_binary_success"] = 0
            row[f"{method}_stage_progress"] = 0.25
        rows.append(row)
    return rows


def test_complete_outcome_roster_rejects_an_entire_missing_cell() -> None:
    rows = _complete_outcome_rows()
    runner.validate_complete_outcome_rows(rows)
    incomplete = [
        row
        for row in rows
        if not (
            row["heldout_body"] == runner.BODIES[-1]
            and row["condition"] == runner.CONDITIONS[-1]
        )
    ]
    with pytest.raises(
        runner.NestedCandidatePoolError, match="complete 1000-triplet schedule"
    ):
        runner.validate_complete_outcome_rows(incomplete)


def test_nested_protocol_declares_all_bootstrap_seed_offsets() -> None:
    protocol = runner.nested_evaluation_protocol()
    assert protocol["formal_seed_block_reused"] is False
    assert protocol["bootstrap_seed_base"] == runner.BOOTSTRAP_SEED
    assert set(protocol["bootstrap_seed_derivation"]) == {
        "overall_comparisons",
        "body_comparisons",
        "body_condition_comparisons",
    }


def test_overall_interval_clusters_all_ten_cells_by_requested_seed() -> None:
    rows = _complete_outcome_rows()
    summary = runner._comparison_summary(
        rows,
        runner.METHOD_N4,
        runner.METHOD_N8,
        seed=runner.BOOTSTRAP_SEED,
    )
    contract = summary["paired_success_delta_interval_contract"]
    assert contract["cluster_count"] == runner.SEED_COUNT
    assert contract["rows_per_cluster"] == len(runner.BODIES) * len(
        runner.CONDITIONS
    )
    assert summary["mcnemar_contract"]["role"] == "descriptive_only"


def _decision(
    method: str,
    *,
    query_index: int = 0,
    normalizer: dict[str, object] | None = None,
) -> dict[str, object]:
    pools, audit = _nested(normalizer)
    if method == runner.METHOD_ACTOR:
        candidates = _raw16()[:1]
        indices = [0]
        scores = None
        selected = 0
    else:
        count = (
            runner.N4_CANDIDATE_COUNT
            if method == runner.METHOD_N4
            else runner.N8_CANDIDATE_COUNT
        )
        candidates = pools[count]
        indices = list(
            audit[
                "ordered_fps_raw_indices_n4"
                if method == runner.METHOD_N4
                else "ordered_fps_raw_indices_n8"
            ]
        )
        members = np.zeros((5, count), dtype=np.float64)
        members[:, count - 1] = 2.0
        aggregate = (
            runner.shared_head.aggregate_risk_adjusted_rank_scores(
                torch.as_tensor(members)
            )
            if method == runner.METHOD_N4
            else runner.pool_runner.aggregate_risk_adjusted_rank_scores(
                torch.as_tensor(members)
            )
        ).numpy()
        selected = int(np.argmax(aggregate))
        scores = {
            "candidate_rank_score_members": members.tolist(),
            "candidate_rank_score_epistemic_lcb_ensemble": aggregate.tolist(),
            "selected_candidate_index": selected,
        }
    return {
        "query_index": query_index,
        "raw_proposal_count": runner.RAW_PROPOSAL_COUNT,
        "raw_ordered_proposals_sha256": audit["raw_ordered_proposals_sha256"],
        "raw_proposal_zero_sha256": audit["raw_proposal_zero_sha256"],
        "nested_pool_audit": audit,
        "raw16_flow_noise_contract_sha256": audit[
            "raw16_flow_noise_contract_sha256"
        ],
        "source_action_normalizer_logical_sha256": audit[
            "source_action_normalizer_logical_sha256"
        ],
        "selection_pool_candidate_count": len(candidates),
        "selection_pool_raw_indices": indices,
        "selection_pool_sha256": runner.array_sha256(candidates),
        "selected_candidate_index": selected,
        "selected_raw_proposal_index": indices[selected],
        "critic_scores": scores,
        "event_age_seconds": None if scores is None else 0.0,
    }


def _rollout(method: str, commitment_sha: str) -> dict[str, object]:
    source_action_normalizer_logical_sha256 = str(_normalizer()["logical_sha256"])
    return {
        "method": method,
        "heldout_body": runner.BODIES[0],
        "condition": runner.CONDITIONS[0],
        "requested_seed": runner.SEED_BASE,
        "initial_reset_snapshot": {"format": "reset", "seed": runner.SEED_BASE},
        "initial_canonical_query_snapshot": {"format": "canonical", "step": 1},
        "initial_candidate_commitment_sha256": commitment_sha,
        "source_action_normalizer_logical_sha256": (
            source_action_normalizer_logical_sha256
        ),
        "binary_success": 1 if method == runner.METHOD_N8 else 0,
        "stage_progress": 1.0 if method == runner.METHOD_N8 else 0.5,
        "max_event_id": 4 if method == runner.METHOD_N8 else 2,
        "action_execution_error": None,
        "policy_query_count": 1,
        "decisions": [_decision(method)],
    }


def test_triplet_materialization_requires_one_reset_and_initial_raw16() -> None:
    _pools, audit = _nested()
    commitment = {
        "commitment_sha256": "c" * 64,
        "reset_snapshot": {"format": "reset", "seed": runner.SEED_BASE},
        "canonical_query_snapshot": {"format": "canonical", "step": 1},
        "raw_ordered_proposals_sha256": audit["raw_ordered_proposals_sha256"],
        "nested_pool_audit": audit,
        "raw16_flow_noise_contract_sha256": audit[
            "raw16_flow_noise_contract_sha256"
        ],
        "source_action_normalizer_logical_sha256": audit[
            "source_action_normalizer_logical_sha256"
        ],
    }
    expected = runner.evaluation_schedule()[0]
    rollouts = {
        method: _rollout(method, commitment["commitment_sha256"])
        for method in runner.METHODS
    }
    pair = runner.materialize_triplet(
        expected,
        rollouts,
        commitment=commitment,
        attempt_sha256="a" * 64,
        execution_contract_logical_sha256="b" * 64,
        source_action_normalizer_logical_sha256=str(
            audit["source_action_normalizer_logical_sha256"]
        ),
        method_result_bindings={
            method: {
                "logical_sha256": f"{index + 1:x}" * 64,
                "file_sha256": f"{index + 4:x}" * 64,
            }
            for index, method in enumerate(runner.METHODS)
        },
    )
    assert pair["same_resolved_reset_actor_n4_n8"] is True
    assert pair["same_initial_raw16_and_nested_pool_audit"] is True
    assert pair["n4_is_exact_ordered_prefix_of_n8"] is True

    broken = copy.deepcopy(rollouts)
    broken[runner.METHOD_N4]["decisions"][0][
        "raw_ordered_proposals_sha256"
    ] = "0" * 64
    with pytest.raises(runner.NestedCandidatePoolError):
        runner.materialize_triplet(
            expected,
            broken,
            commitment=commitment,
            attempt_sha256="a" * 64,
            execution_contract_logical_sha256="b" * 64,
            source_action_normalizer_logical_sha256=str(
                audit["source_action_normalizer_logical_sha256"]
            ),
            method_result_bindings={
                method: {
                    "logical_sha256": f"{index + 1:x}" * 64,
                    "file_sha256": f"{index + 4:x}" * 64,
                }
                for index, method in enumerate(runner.METHODS)
            },
        )


def test_rollout_validator_replays_n4_and_n8_selection_without_new_gate() -> None:
    expected = runner.evaluation_schedule()[0]
    for method in runner.METHODS:
        rollout = _rollout(method, "c" * 64)
        runner.validate_rollout(
            rollout,
            method=method,
            expected=expected,
            expected_source_action_normalizer_logical_sha256=str(
                _normalizer()["logical_sha256"]
            ),
        )
    assert runner.nested_pool_contract()[
        "additional_authorization_or_confidence_gate"
    ] is False


def test_query_after_zero_cannot_switch_to_another_self_consistent_normalizer() -> None:
    expected = runner.evaluation_schedule()[0]
    authority_sha = str(_normalizer()["logical_sha256"])
    rollout = _rollout(runner.METHOD_N4, "c" * 64)
    rollout["decisions"].append(
        _decision(
            runner.METHOD_N4,
            query_index=1,
            normalizer=_normalizer(first_dimension_std=2.0),
        )
    )
    rollout["policy_query_count"] = 2
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="nested decision changed",
    ):
        runner.validate_rollout(
            rollout,
            method=runner.METHOD_N4,
            expected=expected,
            expected_source_action_normalizer_logical_sha256=authority_sha,
        )


def test_method_result_is_create_once_and_replay_validated(tmp_path: Path) -> None:
    expected = runner.evaluation_schedule()[0]
    method = runner.METHOD_ACTOR
    start = runner.build_method_start(
        expected,
        method=method,
        method_ordinal=0,
        attempt_sha256="a" * 64,
        commitment_sha256="c" * 64,
        execution_contract_logical_sha256="b" * 64,
        completed_prefix_result_sha256=[],
    )
    result = runner.build_method_result(
        expected,
        method=method,
        method_ordinal=0,
        rollout=_rollout(method, "c" * 64),
        method_start_sha256=start["method_start_sha256"],
        attempt_sha256="a" * 64,
        commitment_sha256="c" * 64,
        execution_contract_logical_sha256="b" * 64,
        execution_contract_file_sha256="d" * 64,
        source_action_normalizer_logical_sha256=str(
            _normalizer()["logical_sha256"]
        ),
        completed_prefix_result_sha256=[],
    )
    path = tmp_path / "result.json"
    first_file_sha = runner.promote_create_once_json(
        path, result, label="test method result"
    )
    assert first_file_sha == runner.sha256_file(path)
    loaded, staged_only = runner.read_create_once_json(
        path, label="test method result"
    )
    assert staged_only is False
    rollout = runner.validate_method_result(
        loaded,
        expected,
        method=method,
        method_ordinal=0,
        method_start_sha256=start["method_start_sha256"],
        attempt_sha256="a" * 64,
        commitment_sha256="c" * 64,
        execution_contract_logical_sha256="b" * 64,
        execution_contract_file_sha256="d" * 64,
        source_action_normalizer_logical_sha256=str(
            _normalizer()["logical_sha256"]
        ),
        completed_prefix_result_sha256=[],
    )
    assert rollout["method"] == method
    changed = copy.deepcopy(result)
    changed["rollout"]["binary_success"] = 1
    with pytest.raises(runner.NestedCandidatePoolError, match="create-once value"):
        runner.promote_create_once_json(
            path, changed, label="test method result"
        )


def _persist_complete_triplet(
    tmp_path: Path, *, tamper_embedded_actor_outcome: bool = False
) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    expected = runner.evaluation_schedule()[0]
    identity = runner.pair_id(
        expected["heldout_body"], expected["condition"], expected["requested_seed"]
    )
    logical_contract_sha = "b" * 64
    file_contract_sha = "d" * 64
    attempt_base = {
        "format": "etsf_robotwin2_nested_n4_n8_attempt_v2",
        "status": "started_once_fixed_method_order_with_bounded_resume",
        "pair_id": identity,
        **expected,
        "execution_contract_logical_sha256": logical_contract_sha,
        "execution_contract_file_sha256": file_contract_sha,
        "attempt_number": 1,
    }
    attempt = {
        **attempt_base,
        "attempt_sha256": runner.canonical_sha256(attempt_base),
    }
    reset_snapshot = {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "kind": "reset",
    }
    canonical_snapshot = {
        "format": "etsf_robotwin2_observable_reset_snapshot_v2",
        "kind": "canonical_query",
    }
    _pools, audit = _nested()
    commitment_base = {
        "format": runner.INITIAL_COMMITMENT_FORMAT,
        "heldout_body": expected["heldout_body"],
        "condition": expected["condition"],
        "requested_seed": expected["requested_seed"],
        "resolved_seed": expected["requested_seed"],
        "nested_pool_audit": audit,
        "nested_pool_contract_sha256": runner.canonical_sha256(
            runner.nested_pool_contract()
        ),
        "raw16_flow_noise_contract_sha256": audit[
            "raw16_flow_noise_contract_sha256"
        ],
        "source_action_normalizer_logical_sha256": audit[
            "source_action_normalizer_logical_sha256"
        ],
        "raw_ordered_proposals_sha256": audit["raw_ordered_proposals_sha256"],
        "reset_snapshot": reset_snapshot,
        "reset_identity_sha256": runner.formal.reset_identity(reset_snapshot),
        "canonical_query_snapshot": canonical_snapshot,
        "canonical_query_identity_sha256": runner.formal.reset_identity(
            canonical_snapshot
        ),
        "candidate_generation_advanced_simulator": False,
        "frozen_before_any_method_execution": True,
    }
    commitment = {
        **commitment_base,
        "commitment_sha256": runner.canonical_sha256(commitment_base),
    }
    paths = {
        "pair_path": tmp_path / "pairs" / f"{identity}.json",
        "attempt_path": tmp_path / "attempts" / f"{identity}.json",
        "commitment_path": tmp_path / "initial_commitments" / f"{identity}.json",
        "method_starts_dir": tmp_path / "method_starts",
        "method_results_dir": tmp_path / "method_results",
        "method_failures_dir": tmp_path / "method_failures",
    }
    runner.promote_create_once_json(
        paths["attempt_path"], attempt, label="test attempt"
    )
    runner.promote_create_once_json(
        paths["commitment_path"], commitment, label="test commitment"
    )

    rollouts = {}
    bindings = {}
    prefix_result_shas = []
    for method_ordinal, method in enumerate(expected["method_order"]):
        stem = f"{identity}.{method_ordinal:02d}.{method}"
        start = runner.build_method_start(
            expected,
            method=method,
            method_ordinal=method_ordinal,
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            execution_contract_logical_sha256=logical_contract_sha,
            completed_prefix_result_sha256=prefix_result_shas,
        )
        start_path = paths["method_starts_dir"] / f"{stem}.json"
        runner.promote_create_once_json(start_path, start, label="test method start")
        rollout = _rollout(method, commitment["commitment_sha256"])
        rollout["initial_reset_snapshot"] = reset_snapshot
        rollout["initial_canonical_query_snapshot"] = canonical_snapshot
        result = runner.build_method_result(
            expected,
            method=method,
            method_ordinal=method_ordinal,
            rollout=rollout,
            method_start_sha256=start["method_start_sha256"],
            attempt_sha256=attempt["attempt_sha256"],
            commitment_sha256=commitment["commitment_sha256"],
            execution_contract_logical_sha256=logical_contract_sha,
            execution_contract_file_sha256=file_contract_sha,
            source_action_normalizer_logical_sha256=str(
                audit["source_action_normalizer_logical_sha256"]
            ),
            completed_prefix_result_sha256=prefix_result_shas,
        )
        result_path = paths["method_results_dir"] / f"{stem}.json"
        result_file_sha = runner.promote_create_once_json(
            result_path, result, label="test method result"
        )
        rollouts[method] = rollout
        bindings[method] = {
            "logical_sha256": result["method_result_sha256"],
            "file_sha256": result_file_sha,
        }
        prefix_result_shas.append(result["method_result_sha256"])

    pair = runner.materialize_triplet(
        expected,
        rollouts,
        commitment=commitment,
        attempt_sha256=attempt["attempt_sha256"],
        execution_contract_logical_sha256=logical_contract_sha,
        source_action_normalizer_logical_sha256=str(
            audit["source_action_normalizer_logical_sha256"]
        ),
        method_result_bindings=bindings,
    )
    if tamper_embedded_actor_outcome:
        pair["rollouts"][runner.METHOD_ACTOR]["binary_success"] = 1
        pair["rollouts"][runner.METHOD_ACTOR]["stage_progress"] = 1.0
        pair["pair_sha256"] = runner.canonical_sha256(
            {key: value for key, value in pair.items() if key != "pair_sha256"}
        )
    runner.promote_create_once_json(paths["pair_path"], pair, label="test pair")
    context = {
        "identity": identity,
        "expected": expected,
        "attempt": attempt,
        "execution_contract_logical_sha256": logical_contract_sha,
        "execution_contract_file_sha256": file_contract_sha,
        "source_action_normalizer_logical_sha256": str(
            audit["source_action_normalizer_logical_sha256"]
        ),
    }
    return context, pair, paths


def test_existing_pair_embedded_outcome_must_match_method_result(
    tmp_path: Path,
) -> None:
    context, _pair, paths = _persist_complete_triplet(
        tmp_path, tamper_embedded_actor_outcome=True
    )
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="differs from its method results",
    ):
        runner.recover_complete_existing_triplet(**paths, **context)


def test_recovery_rejects_changed_raw16_noise_contract_binding(
    tmp_path: Path,
) -> None:
    context, _pair, paths = _persist_complete_triplet(tmp_path)
    commitment_path = paths["commitment_path"]
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    commitment["raw16_flow_noise_contract_sha256"] = "e" * 64
    commitment["commitment_sha256"] = runner.canonical_sha256(
        {
            key: value
            for key, value in commitment.items()
            if key != "commitment_sha256"
        }
    )
    commitment_path.chmod(0o644)
    commitment_path.write_text(
        json.dumps(commitment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commitment_path.chmod(0o444)
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="stored initial commitment changed",
    ):
        runner.recover_complete_existing_triplet(**paths, **context)


def test_existing_pair_with_missing_method_result_fails_without_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _pair, paths = _persist_complete_triplet(tmp_path)
    expected = context["expected"]
    first_method = expected["method_order"][0]
    identity = context["identity"]
    missing = paths["method_results_dir"] / f"{identity}.00.{first_method}.json"
    missing.unlink()
    rollout_called = False

    def forbidden_rollout(**_kwargs: object) -> dict[str, object]:
        nonlocal rollout_called
        rollout_called = True
        raise AssertionError("existing pair recovery must not execute a rollout")

    monkeypatch.setattr(runner, "execute_rollout", forbidden_rollout)
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="lacks a method result",
    ):
        runner.recover_complete_existing_triplet(**paths, **context)
    assert rollout_called is False


def _oracle_candidate_result(
    candidate_index: int,
    *,
    commitment_sha: str = "1" * 64,
    reset_sha: str = "2" * 64,
    pool_sha: str = "3" * 64,
) -> dict[str, object]:
    raw_indices = [0, 3, 5, 7, 9, 11, 13, 15]
    success = int(candidate_index in {1, 7})
    failed_stage = [0.0, 1.0, 0.25, 0.5, 0.75, 0.25, 0.5, 1.0]
    return runner.build_query0_oracle_candidate_result(
        candidate_index=candidate_index,
        raw_proposal_index=raw_indices[candidate_index],
        initial_candidate_commitment_sha256=commitment_sha,
        paired_reset_sha256=reset_sha,
        shared_raw8_candidate_pool_sha256=pool_sha,
        binary_success=success,
        stage_progress=failed_stage[candidate_index],
        goal_progress=float(candidate_index),
    )


def _oracle_group(expected: dict[str, object]) -> dict[str, object]:
    return runner.build_query0_oracle_group(
        heldout_body=str(expected["heldout_body"]),
        condition=str(expected["condition"]),
        requested_seed=int(expected["requested_seed"]),
        pair_sha256="0" * 64,
        initial_candidate_commitment_sha256="1" * 64,
        paired_reset_sha256="2" * 64,
        shared_raw8_candidate_pool_sha256="3" * 64,
        selected_index_n4=1,
        selected_index_n8=7,
        candidate_results=[_oracle_candidate_result(index) for index in range(8)],
    )


def test_query0_oracle_candidate_contract_rejects_self_signed_false_truth() -> None:
    valid = _oracle_candidate_result(1)
    runner.validate_query0_oracle_candidate_result(
        valid,
        candidate_index=1,
        initial_candidate_commitment_sha256="1" * 64,
        paired_reset_sha256="2" * 64,
        shared_raw8_candidate_pool_sha256="3" * 64,
    )

    changed = copy.deepcopy(valid)
    changed["stage_progress"] = 0.5
    changed["result_sha256"] = runner.canonical_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="candidate result is invalid",
    ):
        runner.validate_query0_oracle_candidate_result(
            changed,
            candidate_index=1,
            initial_candidate_commitment_sha256="1" * 64,
            paired_reset_sha256="2" * 64,
            shared_raw8_candidate_pool_sha256="3" * 64,
        )

    with pytest.raises(runner.NestedCandidatePoolError):
        runner.build_query0_oracle_candidate_result(
            candidate_index=1,
            raw_proposal_index=3,
            initial_candidate_commitment_sha256="1" * 64,
            paired_reset_sha256="2" * 64,
            shared_raw8_candidate_pool_sha256="3" * 64,
            binary_success=1,
            stage_progress=1.0,
            goal_progress=float("inf"),
        )


def test_query0_oracle_group_requires_all_eight_distinct_nested_candidates() -> None:
    valid = _oracle_group(runner.evaluation_schedule()[0])
    runner.validate_query0_oracle_group(valid)
    assert len(valid["candidate_results"]) == runner.N8_CANDIDATE_COUNT

    duplicated = [
        _oracle_candidate_result(index) for index in range(runner.N8_CANDIDATE_COUNT)
    ]
    changed = copy.deepcopy(duplicated[1])
    changed["raw_proposal_index"] = duplicated[0]["raw_proposal_index"]
    changed["result_sha256"] = runner.canonical_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    duplicated[1] = changed
    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="not one nested N8 candidate pool",
    ):
        runner.build_query0_oracle_group(
            heldout_body=runner.BODIES[0],
            condition=runner.CONDITIONS[0],
            requested_seed=runner.SEED_BASE,
            pair_sha256="0" * 64,
            initial_candidate_commitment_sha256="1" * 64,
            paired_reset_sha256="2" * 64,
            shared_raw8_candidate_pool_sha256="3" * 64,
            selected_index_n4=1,
            selected_index_n8=7,
            candidate_results=duplicated,
        )


def test_query0_oracle_truth_requires_exact_complete_formal_schedule() -> None:
    groups = [_oracle_group(expected) for expected in runner.evaluation_schedule()]
    truth = runner.build_query0_oracle_truth_document(
        nested_completion_logical_sha256="4" * 64,
        nested_outcome_document_sha256="5" * 64,
        groups=groups,
    )
    assert truth["group_count"] == 1000
    assert truth["candidate_rollout_count"] == 8000
    assert truth["status"] == runner.ORACLE_TRUTH_STATUS
    assert truth["logical_sha256"] == runner.canonical_sha256(
        {key: value for key, value in truth.items() if key != "logical_sha256"}
    )

    with pytest.raises(
        runner.NestedCandidatePoolError,
        match="oracle truth document is incomplete",
    ):
        runner.build_query0_oracle_truth_document(
            nested_completion_logical_sha256="4" * 64,
            nested_outcome_document_sha256="5" * 64,
            groups=groups[:-1],
        )


def test_nested_runner_collects_all_eight_real_branches_from_one_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, pair, paths = _persist_complete_triplet(tmp_path)
    commitment = json.loads(paths["commitment_path"].read_text(encoding="utf-8"))
    actor = pair["rollouts"][runner.METHOD_ACTOR]
    actor["initial_reset_identity_sha256"] = "2" * 64
    pair["pair_sha256"] = runner.canonical_sha256(
        {key: value for key, value in pair.items() if key != "pair_sha256"}
    )
    n8 = pair["rollouts"][runner.METHOD_N8]["decisions"][0]
    calls = []

    def fake_candidate(**kwargs: object) -> dict[str, object]:
        index = int(kwargs["candidate_index"])
        calls.append(index)
        success = actor["binary_success"] if index == 0 else int(index == 7)
        stage = actor["stage_progress"] if index == 0 else (
            1.0 if success else 0.25
        )
        return runner.build_query0_oracle_candidate_result(
            candidate_index=index,
            raw_proposal_index=n8["selection_pool_raw_indices"][index],
            initial_candidate_commitment_sha256=commitment["commitment_sha256"],
            paired_reset_sha256="2" * 64,
            shared_raw8_candidate_pool_sha256=n8["selection_pool_sha256"],
            binary_success=success,
            stage_progress=stage,
            goal_progress=float(index),
        )

    monkeypatch.setattr(
        runner, "execute_query0_oracle_candidate_rollout", fake_candidate
    )
    group = runner.execute_query0_oracle_group(
        expected=context["expected"],
        pair=pair,
        initial_commitment=commitment,
        task_class=object(),
        task_args={},
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        calibration={},
        source_action_normalizer=_normalizer(),
        instruction="test",
        max_steps=100,
        device=torch.device("cpu"),
    )
    assert calls == list(range(runner.N8_CANDIDATE_COUNT))
    assert group["candidate_results"][0]["binary_success"] == actor[
        "binary_success"
    ]
    assert group["selected_index_n4"] == pair["rollouts"][runner.METHOD_N4][
        "decisions"
    ][0]["selected_candidate_index"]
    runner.validate_query0_oracle_group(group)
