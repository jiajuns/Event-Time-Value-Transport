from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_smolvla_per_head_body_mode_selection_v1 as mode  # noqa: E402


NAMESPACE = "schema5_aloha_source_dense260_20260829_v2/calibration"
TASK = "move_can_pot"
INSTRUCTION_SHA = hashlib.sha256(b"move can into pot").hexdigest()
RESET_CONTRACT_SHA = hashlib.sha256(b"reset-identity-v2").hexdigest()
EVENT_CONTRACT_SHA = hashlib.sha256(b"canonical-event-contract").hexdigest()
OBJECT_CONTRACT_SHA = hashlib.sha256(b"object-state-contract").hexdigest()
CONTEXT = {
    "task": TASK,
    "body_id": "aloha-agilex",
    "body_contract_sha256": hashlib.sha256(b"aloha-body-contract").hexdigest(),
    "policy_id": "smolvla",
    "actor_id": "smolvla-aloha-source-dense260",
    "actor_contract_sha256": hashlib.sha256(b"smolvla-actor-contract").hexdigest(),
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resign(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    base = {key: child for key, child in value.items() if key != field}
    return {**base, field: mode.canonical_sha256(base)}


def identities(groups: int = 100) -> list[dict[str, Any]]:
    rows = []
    for group_index in range(groups):
        semantic = digest(f"semantic-reset-{group_index:03d}")
        identity = {
            "namespace": NAMESPACE,
            "task": TASK,
            "instruction_semantics_sha256": INSTRUCTION_SHA,
            "body_id": CONTEXT["body_id"],
            "body_contract_sha256": CONTEXT["body_contract_sha256"],
            "policy_id": CONTEXT["policy_id"],
            "actor_id": CONTEXT["actor_id"],
            "actor_contract_sha256": CONTEXT["actor_contract_sha256"],
            "requested_seed": 2_026_083_500 + group_index,
            "resolved_seed": 2_026_083_500 + group_index,
            "semantic_reset_cluster_id": semantic,
        }
        execution = mode.execution_group_id(identity)
        for ordinal in range(2):
            rows.append(
                {
                    "sample_id": mode.execution_sample_id(execution, ordinal),
                    "logical_group_id": f"dense260/calibration/{group_index:03d}",
                    "semantic_reset_cluster_id": semantic,
                    "execution_group_id": execution,
                    "group_row_ordinal": ordinal,
                    **identity,
                    "support_category_by_head": {
                        "post_event": mode.EVENT_VOCAB[
                            (group_index + ordinal) % len(mode.EVENT_VOCAB)
                        ],
                        "next_event": mode.EVENT_VOCAB[1:][
                            (group_index + ordinal) % len(mode.EVENT_VOCAB[1:])
                        ],
                        "duration": "observed" if ordinal == 0 else "censored",
                        "success": "positive" if ordinal == 0 else "negative",
                        "recovery": "positive" if ordinal == 0 else "negative",
                        "object_effect": "nonzero" if ordinal == 0 else "near_zero",
                    },
                }
            )
    return sorted(
        rows, key=lambda row: (row["execution_group_id"], row["group_row_ordinal"])
    )


def variants(scope: str) -> list[dict[str, Any]]:
    if scope == mode.PURE_CLOCK_SCOPE:
        names = (mode.PURE_REFERENCE_VARIANT, mode.PURE_CANDIDATE_VARIANT)
        adapter_scope = "clock_only_non_clock_frozen"
    else:
        names = (mode.FULL_REFERENCE_VARIANT, mode.FULL_CANDIDATE_VARIANT)
        adapter_scope = "full_body_adapter"
    seeds = [101, 102, 103, 104, 105]
    shared = [digest(f"shared-{index}") for index in range(5)]
    frozen_nonclock = [digest(f"nonclock-{index}") for index in range(5)]
    result = []
    for variant_index, name in enumerate(names):
        nonclock = (
            list(frozen_nonclock)
            if scope == mode.PURE_CLOCK_SCOPE
            else [digest(f"{name}-nonclock-{index}") for index in range(5)]
        )
        result.append(
            {
                "variant_id": name,
                "body_mode": (
                    "body_agnostic" if variant_index == 0 else "body_conditioned"
                ),
                "adapter_scope": adapter_scope,
                "member_seeds": seeds,
                "member_checkpoint_file_sha256": [
                    digest(f"{name}-checkpoint-{index}") for index in range(5)
                ],
                "shared_core_checkpoint_sha256": shared,
                "non_clock_state_sha256": nonclock,
                "clock_state_sha256": [
                    digest(f"{name}-clock-{index}") for index in range(5)
                ],
                "clock_contract_sha256": digest(f"{name}-clock-contract"),
            }
        )
    return result


def plan_for(scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_ids = sorted({row["execution_group_id"] for row in rows})
    return mode.build_selection_plan(
        protocol_namespace=NAMESPACE,
        task=TASK,
        instruction_semantics_sha256=INSTRUCTION_SHA,
        reset_identity_contract_sha256=RESET_CONTRACT_SHA,
        event_contract_sha256=EVENT_CONTRACT_SHA,
        object_state_contract_sha256=OBJECT_CONTRACT_SHA,
        dataset_format="etsf_smolvla_schema5_source_dense260_preregistration_v2",
        dataset_file_sha256=digest("dense260-file"),
        dataset_logical_sha256=digest("dense260-logical"),
        partition_sha256=digest("dense260-partition"),
        lane="calibration80_synthetic_expanded_for_contract_test",
        lane_group_count=len(group_ids),
        lane_execution_group_set_sha256=mode.canonical_sha256(group_ids),
        execution_contexts=[CONTEXT],
        variant_scope=scope,
        variants=variants(scope),
        bootstrap_draws_sha256=digest("shared-bootstrap-draws"),
    )


def oof_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({row["execution_group_id"] for row in rows})
    result = []
    for fold_index in range(mode.FOLD_COUNT):
        heldout = [
            group for index, group in enumerate(groups) if index % mode.FOLD_COUNT == fold_index
        ]
        train = sorted(set(groups) - set(heldout))
        result.append(
            {
                "fold_index": fold_index,
                "training_execution_group_ids": train,
                "heldout_execution_group_ids": heldout,
                "training_execution_group_ids_sha256": mode.canonical_sha256(train),
                "heldout_execution_group_ids_sha256": mode.canonical_sha256(heldout),
            }
        )
    return result


def variant_views(
    plan: Mapping[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sample = [row["sample_id"] for row in rows]
    logical = [row["logical_group_id"] for row in rows]
    execution = [row["execution_group_id"] for row in rows]
    order_sha = mode.canonical_sha256(
        {
            "sample_id": sample,
            "logical_group_id": logical,
            "execution_group_id": execution,
        }
    )
    masks = {
        head: [row["support_category_by_head"][head] is not None for row in rows]
        for head in mode.HEADS
    }
    result = []
    for variant_index, variant in enumerate(plan["variants"]):
        outputs = {}
        for head in mode.HEADS:
            if plan["variant_scope"] == mode.PURE_CLOCK_SCOPE and head != "duration":
                outputs[head] = digest(f"pure-shared-output-{head}")
            else:
                outputs[head] = digest(
                    f"{variant['variant_id']}-output-{head}-{variant_index}"
                )
        result.append(
            {
                "variant_id": variant["variant_id"],
                "member_checkpoint_file_sha256": variant[
                    "member_checkpoint_file_sha256"
                ],
                "sample_id": sample,
                "logical_group_id": logical,
                "execution_group_id": execution,
                "sample_order_sha256": order_sha,
                "applicable_masks": masks,
                "applicable_mask_set_sha256": mode.canonical_sha256(masks),
                "head_output_sha256": outputs,
            }
        )
    return result


def head_evidence(plan: Mapping[str, Any]) -> dict[str, Any]:
    reference = plan["variant_selection_contract"]["reference_variant_id"]
    candidate = plan["variant_selection_contract"]["candidate_variant_id"]
    eligible = set(plan["variant_selection_contract"]["eligible_heads"])
    result = {}
    for head in mode.HEADS:
        gates = {
            variant: {
                "baseline_performance_gate_passed": True,
                "uncertainty_gate_passed": True,
                "per_fold_support_passed": True,
                "oof_metric_evidence_sha256": digest(f"{head}-{variant}-metric"),
                "uncertainty_evidence_sha256": digest(
                    f"{head}-{variant}-uncertainty"
                ),
                "per_fold_support_evidence_sha256": digest(
                    f"{head}-{variant}-per-fold-support"
                ),
                "deployment_calibration_parameter_sha256": digest(
                    f"{head}-{variant}-deployment-calibration"
                ),
            }
            for variant in (reference, candidate)
        }
        if head in eligible:
            comparison = {
                "status": "evaluated_paired_oof",
                "reference_variant_id": reference,
                "candidate_variant_id": candidate,
                "paired_gain_lcb95": 0.02,
                "harmful_rate_ucb95": 0.05,
                "shared_bootstrap_draws_sha256": plan["statistical_contract"][
                    "bootstrap_draws_sha256"
                ],
                "paired_unit": (
                    "semantic_reset_cluster_if_reused_else_execution_group"
                ),
                "identical_sample_group_order_and_mask": True,
                "selection_labels_used_only_in_oof_training_folds": True,
            }
        else:
            comparison = {
                "status": "not_applicable_pure_clock_non_duration",
                "reason": "clock_dependency_graph_excludes_this_head",
                "reference_variant_id": reference,
                "candidate_variant_id": candidate,
            }
        result[head] = {
            "variant_gates": gates,
            "paired_comparison": comparison,
        }
    return result


def components(scope: str = mode.PURE_CLOCK_SCOPE) -> dict[str, Any]:
    rows = identities()
    plan = plan_for(scope, rows)
    return {
        "plan": plan,
        "rows": rows,
        "folds": oof_folds(rows),
        "views": variant_views(plan, rows),
        "heads": head_evidence(plan),
    }


def build_evidence(parts: Mapping[str, Any]) -> dict[str, Any]:
    return mode.build_calibration_evidence(
        plan=parts["plan"],
        identity_rows=parts["rows"],
        folds=parts["folds"],
        variant_views=parts["views"],
        head_evidence=parts["heads"],
    )


def test_pure_clock_only_selects_duration_and_never_promotes() -> None:
    parts = components()
    evidence = build_evidence(parts)
    receipt = mode.build_selection_receipt(parts["plan"], evidence)
    assert (
        mode.validate_selection_receipt(receipt, parts["plan"], evidence)
        == receipt["receipt_sha256"]
    )
    assert receipt["head_decisions"]["duration"]["selected_variant_id"] == (
        mode.PURE_CANDIDATE_VARIANT
    )
    for head in set(mode.HEADS) - {"duration"}:
        assert receipt["head_decisions"][head]["selection_status"] == (
            "fixed_reference_pure_clock_invariant"
        )
        assert receipt["head_decisions"][head]["selected_variant_id"] == (
            mode.PURE_REFERENCE_VARIANT
        )
    assert receipt["all_six_heads_enabled"] is True
    assert receipt["system_fallback"] == (
        "actor_baseline_required_rank_route_not_authorized"
    )
    assert receipt["rank_route_contract"]["rank_route_authorized"] is False
    assert receipt["rank_route_contract"][
        "rank_action_selection_must_fallback_to_actor_baseline"
    ] is True
    assert receipt["capability"][
        "selected_variant_ids_are_provisional_non_executable"
    ] is True
    assert receipt["capability"]["metric_truth_recomputed_by_this_schema"] is False
    assert receipt["capability"]["runtime_route_exported"] is False
    duration_gate = evidence["head_evidence"]["duration"]["variant_gates"][
        mode.PURE_CANDIDATE_VARIANT
    ]
    assert receipt["head_decisions"]["duration"][
        "selected_deployment_calibration_parameter_sha256"
    ] == duration_gate["deployment_calibration_parameter_sha256"]
    assert parts["plan"]["capability"]["promotion_authorized"] is False
    assert evidence["capability"]["promotion_authorized"] is False
    assert receipt["capability"]["promotion_authorized"] is False


def test_full_body_adapter_has_explicit_names_and_all_heads_are_eligible() -> None:
    parts = components(mode.FULL_BODY_ADAPTER_SCOPE)
    evidence = build_evidence(parts)
    receipt = mode.build_selection_receipt(parts["plan"], evidence)
    assert [row["variant_id"] for row in parts["plan"]["variants"]] == [
        mode.FULL_REFERENCE_VARIANT,
        mode.FULL_CANDIDATE_VARIANT,
    ]
    assert all(
        row["selection_status"] == "selected_body_conditioned_candidate"
        and row["selected_variant_id"] == mode.FULL_CANDIDATE_VARIANT
        for row in receipt["head_decisions"].values()
    )
    wrong = variants(mode.PURE_CLOCK_SCOPE)
    with pytest.raises(mode.BodyModeSelectionError, match="variant id"):
        mode.build_selection_plan(
            protocol_namespace=NAMESPACE,
            task=TASK,
            instruction_semantics_sha256=INSTRUCTION_SHA,
            reset_identity_contract_sha256=RESET_CONTRACT_SHA,
            event_contract_sha256=EVENT_CONTRACT_SHA,
            object_state_contract_sha256=OBJECT_CONTRACT_SHA,
            dataset_format="synthetic",
            dataset_file_sha256=digest("file"),
            dataset_logical_sha256=digest("logical"),
            partition_sha256=digest("partition"),
            lane="calibration",
            lane_group_count=100,
            lane_execution_group_set_sha256=digest("groups"),
            execution_contexts=[CONTEXT],
            variant_scope=mode.FULL_BODY_ADAPTER_SCOPE,
            variants=wrong,
            bootstrap_draws_sha256=digest("draws"),
        )


def test_pure_clock_rejects_non_duration_output_or_nonclock_state_change() -> None:
    parts = components()
    parts["views"][1]["head_output_sha256"]["success"] = digest("changed-success")
    with pytest.raises(mode.BodyModeSelectionError, match="non-duration"):
        build_evidence(parts)

    rows = identities()
    changed = variants(mode.PURE_CLOCK_SCOPE)
    changed[1]["non_clock_state_sha256"][0] = digest("changed-nonclock")
    with pytest.raises(mode.BodyModeSelectionError, match="non-clock"):
        mode.build_selection_plan(
            protocol_namespace=NAMESPACE,
            task=TASK,
            instruction_semantics_sha256=INSTRUCTION_SHA,
            reset_identity_contract_sha256=RESET_CONTRACT_SHA,
            event_contract_sha256=EVENT_CONTRACT_SHA,
            object_state_contract_sha256=OBJECT_CONTRACT_SHA,
            dataset_format="synthetic",
            dataset_file_sha256=digest("file"),
            dataset_logical_sha256=digest("logical"),
            partition_sha256=digest("partition"),
            lane="calibration",
            lane_group_count=100,
            lane_execution_group_set_sha256=mode.canonical_sha256(
                sorted({row["execution_group_id"] for row in rows})
            ),
            execution_contexts=[CONTEXT],
            variant_scope=mode.PURE_CLOCK_SCOPE,
            variants=changed,
            bootstrap_draws_sha256=digest("draws"),
        )


def test_variants_require_identical_sample_group_order_and_masks() -> None:
    parts = components()
    parts["views"][1]["sample_id"][0], parts["views"][1]["sample_id"][1] = (
        parts["views"][1]["sample_id"][1],
        parts["views"][1]["sample_id"][0],
    )
    with pytest.raises(mode.BodyModeSelectionError, match="sample/group/order"):
        build_evidence(parts)

    parts = components()
    parts["views"][1]["applicable_masks"]["recovery"][0] = False
    parts["views"][1]["applicable_mask_set_sha256"] = mode.canonical_sha256(
        parts["views"][1]["applicable_masks"]
    )
    with pytest.raises(mode.BodyModeSelectionError, match="applicable masks"):
        build_evidence(parts)


def test_three_layer_identity_and_five_fold_exact_once_fail_closed() -> None:
    parts = components()
    parts["rows"][0]["actor_contract_sha256"] = digest("different-actor")
    with pytest.raises(mode.BodyModeSelectionError, match="context"):
        build_evidence(parts)

    parts = components()
    first = parts["folds"][0]["heldout_execution_group_ids"][0]
    displaced = parts["folds"][1]["heldout_execution_group_ids"][0]
    parts["folds"][1]["heldout_execution_group_ids"][0] = first
    parts["folds"][1]["heldout_execution_group_ids"].sort()
    all_groups = sorted({row["execution_group_id"] for row in parts["rows"]})
    for index in (0, 1):
        heldout = parts["folds"][index]["heldout_execution_group_ids"]
        parts["folds"][index]["training_execution_group_ids"] = sorted(
            set(all_groups) - set(heldout)
        )
        parts["folds"][index]["training_execution_group_ids_sha256"] = (
            mode.canonical_sha256(
                parts["folds"][index]["training_execution_group_ids"]
            )
        )
        parts["folds"][index]["heldout_execution_group_ids_sha256"] = (
            mode.canonical_sha256(heldout)
        )
    assert displaced not in parts["folds"][1]["heldout_execution_group_ids"]
    with pytest.raises(mode.BodyModeSelectionError, match="more than once"):
        build_evidence(parts)


def test_support_is_recomputed_and_insufficient_head_is_disabled() -> None:
    parts = components()
    for row in parts["rows"]:
        row["support_category_by_head"]["post_event"] = "e0"
    evidence = build_evidence(parts)
    assert evidence["head_support"]["heads"]["post_event"][
        "support_gate_passed"
    ] is False
    receipt = mode.build_selection_receipt(parts["plan"], evidence)
    assert receipt["head_decisions"]["post_event"] == {
        "head": "post_event",
        "support_gate_passed": False,
        "reference_variant_gate_passed": False,
        "candidate_variant_gate_passed": False,
        "selection_status": "head_disabled",
        "selected_variant_id": None,
        "reason": "insufficient_independent_group_support",
        "paired_gain_lcb95": None,
        "harmful_rate_ucb95": None,
        "selected_deployment_calibration_parameter_sha256": None,
        "actor_baseline_fallback_required": True,
    }
    assert receipt["all_six_heads_enabled"] is False
    assert "actor_baseline_required" in receipt["system_fallback"]


def test_next_event_e0_is_rejected_and_proper_loss_gain_need_only_be_finite() -> None:
    parts = components()
    parts["rows"][0]["support_category_by_head"]["next_event"] = "e0"
    with pytest.raises(mode.BodyModeSelectionError, match="next_event support category"):
        build_evidence(parts)

    parts = components()
    # Proper/decision loss differences are not probabilities and need not lie
    # in [-1, 1].  A finite positive LCB remains structurally valid.
    parts["heads"]["duration"]["paired_comparison"]["paired_gain_lcb95"] = 1.25
    evidence = build_evidence(parts)
    decision = mode.build_selection_receipt(parts["plan"], evidence)[
        "head_decisions"
    ]["duration"]
    assert decision["selected_variant_id"] == mode.PURE_CANDIDATE_VARIANT
    assert decision["paired_gain_lcb95"] == 1.25


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("uncertainty", "candidate_performance_or_uncertainty_gate_failed"),
        ("gain", "paired_gain_lcb_not_strictly_positive"),
        ("harm", "harmful_rate_ucb_exceeds_limit"),
    ],
)
def test_candidate_gate_failures_deterministically_fallback(
    mutation: str, reason: str
) -> None:
    parts = components()
    duration = parts["heads"]["duration"]
    candidate = mode.PURE_CANDIDATE_VARIANT
    if mutation == "uncertainty":
        duration["variant_gates"][candidate]["uncertainty_gate_passed"] = False
    elif mutation == "gain":
        duration["paired_comparison"]["paired_gain_lcb95"] = 0.0
    else:
        duration["paired_comparison"]["harmful_rate_ucb95"] = 0.100001
    evidence = build_evidence(parts)
    decision = mode.build_selection_receipt(parts["plan"], evidence)[
        "head_decisions"
    ]["duration"]
    assert decision["selection_status"] == "fallback_body_agnostic_reference"
    assert decision["selected_variant_id"] == mode.PURE_REFERENCE_VARIANT
    assert decision["reason"] == reason


def test_per_fold_support_failure_falls_back_and_selected_calibration_is_bound() -> None:
    parts = components()
    candidate = mode.PURE_CANDIDATE_VARIANT
    reference = mode.PURE_REFERENCE_VARIANT
    parts["heads"]["duration"]["variant_gates"][candidate][
        "per_fold_support_passed"
    ] = False
    evidence = build_evidence(parts)
    decision = mode.build_selection_receipt(parts["plan"], evidence)[
        "head_decisions"
    ]["duration"]
    assert decision["selection_status"] == "fallback_body_agnostic_reference"
    assert decision["selected_variant_id"] == reference
    assert decision["selected_deployment_calibration_parameter_sha256"] == evidence[
        "head_evidence"
    ]["duration"]["variant_gates"][reference][
        "deployment_calibration_parameter_sha256"
    ]


def test_failed_reference_disables_head_instead_of_selecting_candidate() -> None:
    parts = components()
    parts["heads"]["duration"]["variant_gates"][mode.PURE_REFERENCE_VARIANT][
        "baseline_performance_gate_passed"
    ] = False
    evidence = build_evidence(parts)
    decision = mode.build_selection_receipt(parts["plan"], evidence)[
        "head_decisions"
    ]["duration"]
    assert decision["selection_status"] == "head_disabled"
    assert decision["selected_variant_id"] is None
    assert decision["actor_baseline_fallback_required"] is True


def test_bootstrap_binding_and_no_promotion_are_not_self_asserted() -> None:
    parts = components()
    evidence = build_evidence(parts)
    changed = copy.deepcopy(evidence)
    changed["bootstrap"]["draws_sha256"] = digest("posthoc-draws")
    changed = resign(changed, "evidence_sha256")
    with pytest.raises(mode.BodyModeSelectionError, match="bootstrap"):
        mode.validate_calibration_evidence(changed, parts["plan"])

    receipt = mode.build_selection_receipt(parts["plan"], evidence)
    promoted = copy.deepcopy(receipt)
    promoted["capability"]["promotion_authorized"] = True
    promoted = resign(promoted, "receipt_sha256")
    with pytest.raises(mode.BodyModeSelectionError, match="deterministic"):
        mode.validate_selection_receipt(promoted, parts["plan"], evidence)

    promoted_plan = copy.deepcopy(parts["plan"])
    promoted_plan["capability"]["promotion_authorized"] = True
    promoted_plan = resign(promoted_plan, "plan_sha256")
    with pytest.raises(mode.BodyModeSelectionError, match="no-promotion"):
        mode.validate_selection_plan(promoted_plan)

    rank_routed = copy.deepcopy(parts["plan"])
    rank_routed["rank_route_contract"]["rank_route_authorized"] = True
    rank_routed = resign(rank_routed, "plan_sha256")
    with pytest.raises(mode.BodyModeSelectionError, match="rank route"):
        mode.validate_selection_plan(rank_routed)
