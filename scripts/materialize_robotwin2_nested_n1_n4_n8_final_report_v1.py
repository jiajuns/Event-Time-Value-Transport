#!/usr/bin/env python3
"""Materialize one fail-closed five-body N1/N4/N8 transfer report.

The materializer replays the completed nested SHA chain and every one of the
1,000 triplet records.  Selected closed-loop arms are sufficient for paired
success/progress metrics, but never for candidate-oracle regret.  Oracle
metrics are enabled only by a separate complete query-zero intervention
artifact containing eight independently executed candidates for every
body/condition/seed cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import evaluate_robotwin2_five_body_lobo_n1_n4_n8_oracle_v1 as evaluator


FORMAT = "etsf_robotwin2_nested_n1_n4_n8_final_report_input_v1"
ORACLE_TRUTH_FORMAT = "etsf_robotwin2_nested_query0_oracle_truth_v1"
ORACLE_TRUTH_STATUS = "complete_1000_query0_groups_8000_candidate_rollouts"
ORACLE_GROUP_FORMAT = "etsf_robotwin2_nested_query0_oracle_group_v1"
ORACLE_RESULT_FORMAT = "etsf_robotwin2_nested_query0_candidate_result_v1"

NESTED_CONTRACT_FORMAT = (
    "etsf_robotwin2_nested_n4_n8_execution_contract_v2_actor_protocol"
)
NESTED_OUTCOME_FORMAT = "etsf_robotwin2_nested_n4_n8_outcomes_v2"
NESTED_REPORT_FORMAT = "etsf_robotwin2_nested_n4_n8_report_v2"
NESTED_COMPLETION_FORMAT = "etsf_robotwin2_nested_n4_n8_completion_receipt_v2"
NESTED_PAIR_FORMAT = (
    "etsf_robotwin2_actor_nested_n4_n8_paired_execution_v3_actor_protocol"
)
ACTOR_AUTHORITY_FORMAT = (
    "etsf_robotwin2_frozen_native_actor_authority_v3_actor_execution_protocol"
)

METHOD_ACTOR = "actor_baseline"
METHOD_N4 = "etsf_nested_best_of_4_from_raw16"
METHOD_N8 = "etsf_nested_best_of_8_from_raw16"
METHODS = (METHOD_ACTOR, METHOD_N4, METHOD_N8)
EXPECTED_PAIRS = len(evaluator.BODIES) * len(evaluator.CONDITIONS) * evaluator.SEED_COUNT
EXPECTED_CANDIDATE_ROLLOUTS = EXPECTED_PAIRS * 8
ORACLE_CONTINUATION_POLICY = (
    "force_query0_nested_n8_candidate_then_actor_candidate0_at_all_future_queries"
)
SHA_CHARS = frozenset("0123456789abcdef")


class FinalReportMaterializationError(RuntimeError):
    """A nested, actor, fold, or oracle artifact is incomplete or changed."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalReportMaterializationError(
            "artifact is not finite canonical JSON"
        ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FinalReportMaterializationError(f"file is missing or symbolic: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= SHA_CHARS
    ):
        raise FinalReportMaterializationError(f"{label} must be a lowercase SHA-256")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalReportMaterializationError(f"{label} must be a real file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FinalReportMaterializationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise FinalReportMaterializationError(f"{label} must be an object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def verify_named_sha(value: Mapping[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    declared = _sha(unsigned.pop(field, None), f"{label} {field}")
    if canonical_sha256(unsigned) != declared:
        raise FinalReportMaterializationError(f"{label} {field} mismatch")
    return declared


def pair_id(body: str, condition: str, seed: int) -> str:
    return f"{body.replace('/', '_')}__{condition}__seed_{seed}"


def expected_schedule() -> list[tuple[str, str, int]]:
    return [
        (body, condition, evaluator.SEED_BASE + ordinal)
        for body in evaluator.BODIES
        for condition in evaluator.CONDITIONS
        for ordinal in range(evaluator.SEED_COUNT)
    ]


def _stage(value: Any, success: Any, label: str) -> float:
    if type(success) is not int or success not in (0, 1):
        raise FinalReportMaterializationError(f"{label} success is invalid")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportMaterializationError(f"{label} stage is invalid")
    result = float(value)
    if result not in evaluator.STAGE_SUPPORT or (result == 1.0) != bool(success):
        raise FinalReportMaterializationError(
            f"{label} success and stage progress disagree"
        )
    return result


def validate_nested_sha_chain(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract_path = root / "execution_contract.json"
    outcome_path = root / "nested_paired_outcomes.json"
    report_path = root / "nested_n4_n8_report.json"
    completion_path = root / "completion_receipt.json"
    contract = read_json(contract_path, "nested execution contract")
    outcomes = read_json(outcome_path, "nested outcomes")
    report = read_json(report_path, "nested report")
    completion = read_json(completion_path, "nested completion")
    contract_sha = verify_named_sha(contract, "logical_sha256", "nested contract")
    outcome_sha = verify_named_sha(outcomes, "document_sha256", "nested outcomes")
    report_sha = verify_named_sha(report, "report_sha256", "nested report")
    completion_sha = verify_named_sha(
        completion, "logical_sha256", "nested completion"
    )
    if (
        contract.get("format") != NESTED_CONTRACT_FORMAT
        or contract.get("initial_condition_triplet_count") != EXPECTED_PAIRS
        or contract.get("rollout_count") != EXPECTED_PAIRS * 3
        or contract.get("methods") != list(METHODS)
        or outcomes.get("format") != NESTED_OUTCOME_FORMAT
        or outcomes.get("status")
        != "complete_1000_initial_condition_triplets_3000_rollouts"
        or outcomes.get("pair_count") != EXPECTED_PAIRS
        or outcomes.get("rollout_count") != EXPECTED_PAIRS * 3
        or outcomes.get("methods") != list(METHODS)
        or outcomes.get("execution_contract_logical_sha256") != contract_sha
        or outcomes.get("execution_contract_file_sha256") != sha256_file(contract_path)
        or report.get("format") != NESTED_REPORT_FORMAT
        or report.get("status") != "complete_shared_raw16_nested_n4_n8_paired_report"
        or report.get("outcome_document_sha256") != outcome_sha
        or completion.get("format") != NESTED_COMPLETION_FORMAT
        or completion.get("status") != "complete_1000_triplets_3000_rollouts_frozen"
        or completion.get("execution_contract_logical_sha256") != contract_sha
        or completion.get("execution_contract_file_sha256") != sha256_file(contract_path)
        or completion.get("outcome_document_sha256") != outcome_sha
        or completion.get("outcome_file_sha256") != sha256_file(outcome_path)
        or completion.get("report_sha256") != report_sha
        or completion.get("report_file_sha256") != sha256_file(report_path)
        or completion.get("initial_condition_triplet_count") != EXPECTED_PAIRS
        or completion.get("rollout_count") != EXPECTED_PAIRS * 3
        or not isinstance(outcomes.get("rows"), list)
        or outcomes.get("rows_sha256") != canonical_sha256(outcomes["rows"])
    ):
        raise FinalReportMaterializationError("nested completion SHA chain changed")
    completion = dict(completion)
    completion["_logical_sha256"] = completion_sha
    return contract, outcomes, completion


def actor_provenance(
    authority_path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    authority = read_json(authority_path, "actor authority")
    verify_named_sha(authority, "logical_sha256", "actor authority")
    actors = authority.get("actors")
    if (
        authority.get("format") != ACTOR_AUTHORITY_FORMAT
        or authority.get("one_universal_actor_for_all_five_bodies") is not True
        or not isinstance(actors, Mapping)
        or set(actors) != set(evaluator.BODIES)
        or not isinstance(authority.get("upstream_training_state_file_sha256"), str)
    ):
        raise FinalReportMaterializationError(
            "actor authority does not prove five-body actor exposure"
        )
    checkpoint_shas = {
        item.get("checkpoint_sha256")
        for item in actors.values()
        if isinstance(item, Mapping)
    }
    checkpoint = contract.get("actor_checkpoint_tree_sha256")
    if len(checkpoint_shas) != 1 or checkpoint_shas != {checkpoint}:
        raise FinalReportMaterializationError(
            "nested actor checkpoint differs from actor authority"
        )
    return {
        "checkpoint_sha256": _sha(checkpoint, "nested actor checkpoint"),
        "training_data_receipt_sha256": _sha(
            authority["upstream_training_state_file_sha256"],
            "actor training receipt",
        ),
        "training_bodies": list(evaluator.BODIES),
    }


def critic_folds(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds = contract.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != set(evaluator.BODIES):
        raise FinalReportMaterializationError("nested contract lacks five LOBO folds")
    result = []
    for body in evaluator.BODIES:
        fold = folds[body]
        if not isinstance(fold, Mapping):
            raise FinalReportMaterializationError(f"{body} fold is invalid")
        members = fold.get("members")
        sources = sorted(candidate for candidate in evaluator.BODIES if candidate != body)
        if (
            fold.get("heldout_body") != body
            or sorted(fold.get("source_bodies", [])) != sources
            or not isinstance(members, list)
            or len(members) != 5
            or fold.get("training_summary_sha256") is None
        ):
            raise FinalReportMaterializationError(f"{body} LOBO fold changed")
        member_shas = []
        for index, member in enumerate(members):
            if not isinstance(member, Mapping) or member.get("member") != index:
                raise FinalReportMaterializationError(f"{body} member order changed")
            member_shas.append(_sha(member.get("checkpoint_sha256"), "critic member"))
        result.append(
            {
                "heldout_body": body,
                "source_supervision_bodies": sources,
                "selection_bodies": sources,
                "normalizer_fit_bodies": sources,
                "target_labeled_group_count": 0,
                "target_adapter": evaluator.TARGET_ADAPTER,
                "checkpoint_sha256": canonical_sha256(member_shas),
                "training_receipt_sha256": _sha(
                    fold["training_summary_sha256"], f"{body} training summary"
                ),
            }
        )
    return result


def policy_rows(
    root: Path,
    outcomes: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    rows = outcomes["rows"]
    schedule = expected_schedule()
    if len(rows) != len(schedule):
        raise FinalReportMaterializationError("nested outcomes are not 1000 pairs")
    fold_sha = {str(fold["heldout_body"]): str(fold["checkpoint_sha256"]) for fold in folds}
    normalized = []
    pair_evidence: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row, identity in zip(rows, schedule, strict=True):
        body, condition, seed = identity
        if (
            not isinstance(row, Mapping)
            or row.get("heldout_body") != body
            or row.get("condition") != condition
            or row.get("requested_seed") != seed
        ):
            raise FinalReportMaterializationError("nested outcome schedule changed")
        path = root / "pairs" / f"{pair_id(body, condition, seed)}.json"
        pair = read_json(path, f"nested pair {identity}")
        pair_sha = verify_named_sha(pair, "pair_sha256", f"nested pair {identity}")
        rollouts = pair.get("rollouts")
        if (
            pair.get("format") != NESTED_PAIR_FORMAT
            or row.get("pair_sha256") != pair_sha
            or not isinstance(rollouts, Mapping)
            or set(rollouts) != set(METHODS)
            or pair.get("same_resolved_reset_actor_n4_n8") is not True
            or pair.get("same_initial_raw16_and_nested_pool_audit") is not True
            or pair.get("n4_is_exact_ordered_prefix_of_n8") is not True
        ):
            raise FinalReportMaterializationError(f"nested pair changed: {identity}")
        reset_shas = {rollouts[method].get("initial_reset_identity_sha256") for method in METHODS}
        first = {method: rollouts[method].get("decisions", [None])[0] for method in METHODS}
        if (
            len(reset_shas) != 1
            or None in reset_shas
            or any(not isinstance(first[method], Mapping) for method in METHODS)
            or first[METHOD_ACTOR].get("selected_candidate_index") != 0
            or first[METHOD_N4].get("selection_pool_candidate_count") != 4
            or first[METHOD_N8].get("selection_pool_candidate_count") != 8
            or first[METHOD_N4].get("selection_pool_raw_indices")
            != first[METHOD_N8].get("selection_pool_raw_indices", [])[:4]
            or first[METHOD_N4].get("selection_pool_sha256")
            != first[METHOD_N8].get("nested_pool_audit", {}).get(
                "n4_ordered_candidates_sha256"
            )
        ):
            raise FinalReportMaterializationError(
                f"nested reset/prefix evidence changed: {identity}"
            )
        values: dict[str, Any] = {
            "heldout_body": body,
            "condition": condition,
            "requested_seed": seed,
            "paired_reset_sha256": _sha(next(iter(reset_shas)), "paired reset"),
            "shared_raw8_candidate_pool_sha256": _sha(
                first[METHOD_N8].get("selection_pool_sha256"), "shared N8 pool"
            ),
            "critic_checkpoint_sha256": fold_sha[body],
        }
        for method, target in (
            (METHOD_ACTOR, "actor_n1"),
            (METHOD_N4, "critic_n4"),
            (METHOD_N8, "critic_n8"),
        ):
            rollout = rollouts[method]
            success = rollout.get("binary_success")
            stage = _stage(rollout.get("stage_progress"), success, f"{identity}/{method}")
            if (
                row.get(f"{method}_binary_success") != success
                or float(row.get(f"{method}_stage_progress", -1.0)) != stage
            ):
                raise FinalReportMaterializationError(
                    f"outcome row differs from pair: {identity}/{method}"
                )
            values[f"{target}_binary_success"] = success
            values[f"{target}_stage_progress"] = stage
        normalized.append(values)
        pair_evidence[identity] = {
            "pair_sha256": pair_sha,
            "initial_candidate_commitment_sha256": pair.get(
                "initial_candidate_commitment_sha256"
            ),
            "paired_reset_sha256": values["paired_reset_sha256"],
            "shared_raw8_candidate_pool_sha256": values[
                "shared_raw8_candidate_pool_sha256"
            ],
            "n8_raw_indices": list(first[METHOD_N8]["selection_pool_raw_indices"]),
            "selected_index_n1": 0,
            "selected_index_n4": first[METHOD_N4].get("selected_candidate_index"),
            "selected_index_n8": first[METHOD_N8].get("selected_candidate_index"),
            "actor_n1_binary_success": values["actor_n1_binary_success"],
            "actor_n1_stage_progress": values["actor_n1_stage_progress"],
        }
    return normalized, pair_evidence


def _oracle_result(
    raw: Any,
    *,
    candidate_index: int,
    raw_index: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise FinalReportMaterializationError("oracle candidate result is invalid")
    fields = {
        "format",
        "candidate_index",
        "raw_proposal_index",
        "initial_candidate_commitment_sha256",
        "paired_reset_sha256",
        "shared_raw8_candidate_pool_sha256",
        "continuation_policy",
        "binary_success",
        "stage_progress",
        "goal_progress",
        "action_execution_error",
        "result_sha256",
    }
    if set(raw) != fields:
        raise FinalReportMaterializationError("oracle result fields changed")
    verify_named_sha(raw, "result_sha256", "oracle candidate result")
    success = raw.get("binary_success")
    stage = _stage(raw.get("stage_progress"), success, "oracle candidate")
    goal = raw.get("goal_progress")
    if (
        raw.get("format") != ORACLE_RESULT_FORMAT
        or raw.get("candidate_index") != candidate_index
        or raw.get("raw_proposal_index") != raw_index
        or raw.get("initial_candidate_commitment_sha256")
        != evidence["initial_candidate_commitment_sha256"]
        or raw.get("paired_reset_sha256") != evidence["paired_reset_sha256"]
        or raw.get("shared_raw8_candidate_pool_sha256")
        != evidence["shared_raw8_candidate_pool_sha256"]
        or raw.get("continuation_policy") != ORACLE_CONTINUATION_POLICY
        or raw.get("action_execution_error") is not None
        or isinstance(goal, bool)
        or not isinstance(goal, (int, float))
        or not math.isfinite(float(goal))
    ):
        raise FinalReportMaterializationError("oracle candidate binding changed")
    return {"success": success, "stage": stage, "goal": float(goal)}


def oracle_groups(
    path: Path,
    *,
    completion: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    pairs: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    truth = read_json(path, "nested oracle truth")
    verify_named_sha(truth, "logical_sha256", "nested oracle truth")
    groups = truth.get("groups")
    if (
        truth.get("format") != ORACLE_TRUTH_FORMAT
        or truth.get("status") != ORACLE_TRUTH_STATUS
        or truth.get("nested_completion_logical_sha256")
        != completion["_logical_sha256"]
        or truth.get("nested_outcome_document_sha256")
        != outcomes["document_sha256"]
        or truth.get("group_count") != EXPECTED_PAIRS
        or truth.get("candidate_rollout_count") != EXPECTED_CANDIDATE_ROLLOUTS
        or not isinstance(groups, list)
        or len(groups) != EXPECTED_PAIRS
        or truth.get("groups_sha256") != canonical_sha256(groups)
    ):
        raise FinalReportMaterializationError("oracle truth is incomplete or unbound")
    fold_sha = {str(fold["heldout_body"]): str(fold["checkpoint_sha256"]) for fold in folds}
    normalized = []
    for raw, identity in zip(groups, expected_schedule(), strict=True):
        if not isinstance(raw, Mapping):
            raise FinalReportMaterializationError("oracle group is invalid")
        fields = {
            "format",
            "heldout_body",
            "condition",
            "requested_seed",
            "decision_group_id",
            "pair_sha256",
            "initial_candidate_commitment_sha256",
            "paired_reset_sha256",
            "shared_raw8_candidate_pool_sha256",
            "selected_index_n1",
            "selected_index_n4",
            "selected_index_n8",
            "candidate_results",
            "group_sha256",
        }
        if set(raw) != fields:
            raise FinalReportMaterializationError("oracle group fields changed")
        verify_named_sha(raw, "group_sha256", "oracle group")
        body, condition, seed = identity
        evidence = pairs[identity]
        candidate_results = raw.get("candidate_results")
        if (
            raw.get("format") != ORACLE_GROUP_FORMAT
            or raw.get("heldout_body") != body
            or raw.get("condition") != condition
            or raw.get("requested_seed") != seed
            or raw.get("decision_group_id") != f"query0|{body}|{condition}|{seed}"
            or raw.get("pair_sha256") != evidence["pair_sha256"]
            or any(raw.get(key) != evidence[key] for key in (
                "initial_candidate_commitment_sha256",
                "paired_reset_sha256",
                "shared_raw8_candidate_pool_sha256",
                "selected_index_n1",
                "selected_index_n4",
                "selected_index_n8",
            ))
            or not isinstance(candidate_results, list)
            or len(candidate_results) != 8
        ):
            raise FinalReportMaterializationError("oracle group identity changed")
        results = [
            _oracle_result(
                result,
                candidate_index=index,
                raw_index=evidence["n8_raw_indices"][index],
                evidence=evidence,
            )
            for index, result in enumerate(candidate_results)
        ]
        if (
            results[0]["success"] != evidence["actor_n1_binary_success"]
            or results[0]["stage"] != evidence["actor_n1_stage_progress"]
        ):
            raise FinalReportMaterializationError(
                "oracle candidate zero disagrees with paired actor N1"
            )
        normalized.append(
            {
                "heldout_body": body,
                "condition": condition,
                "requested_seed": seed,
                "decision_group_id": raw["decision_group_id"],
                "shared_raw8_candidate_pool_sha256": evidence[
                    "shared_raw8_candidate_pool_sha256"
                ],
                "critic_checkpoint_sha256": fold_sha[body],
                "candidate_binary_success": [item["success"] for item in results],
                "candidate_stage_progress": [item["stage"] for item in results],
                "candidate_goal_progress": [item["goal"] for item in results],
                "selected_index_n1": raw["selected_index_n1"],
                "selected_index_n4": raw["selected_index_n4"],
                "selected_index_n8": raw["selected_index_n8"],
            }
        )
    return normalized


def build_materialization(
    *,
    nested_root: Path,
    actor_authority_path: Path,
    oracle_truth_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = nested_root.expanduser().resolve()
    contract, outcomes, completion = validate_nested_sha_chain(root)
    actor = actor_provenance(actor_authority_path.expanduser().resolve(), contract)
    folds = critic_folds(contract)
    rows, pairs = policy_rows(root, outcomes, folds)
    oracle = None
    if oracle_truth_path is not None:
        oracle = oracle_groups(
            oracle_truth_path.expanduser().resolve(),
            completion=completion,
            outcomes=outcomes,
            folds=folds,
            pairs=pairs,
        )
    base = {
        "format": FORMAT,
        "benchmark": evaluator.BENCHMARK,
        "task": evaluator.TASK,
        "nested_root": str(root),
        "nested_completion_logical_sha256": completion["_logical_sha256"],
        "nested_outcome_document_sha256": outcomes["document_sha256"],
        "actor_provenance": actor,
        "critic_folds": folds,
        "policy_rows": rows,
        "oracle_evidence": {
            "status": (
                "complete_sealed_query0_candidate_truth"
                if oracle is not None
                else "unavailable_no_candidate_truth_artifact"
            ),
            "candidate_truth_artifact_present": oracle is not None,
            "candidate_rollouts_observed": (
                EXPECTED_CANDIDATE_ROLLOUTS if oracle is not None else 0
            ),
            "oracle_groups": oracle,
        },
    }
    materialization = {**base, "logical_sha256": canonical_sha256(base)}
    if oracle is None:
        report = evaluator.build_policy_only_report(
            input_document_sha256=materialization["logical_sha256"],
            actor_provenance=actor,
            critic_folds=folds,
            policy_rows=rows,
            oracle_unavailability={
                "reason": (
                    "completed nested arms contain only selected policy outcomes; "
                    "unexecuted candidate counterfactuals are absent"
                ),
                "candidate_truth_artifact_present": False,
                "candidate_rollouts_observed": 0,
            },
        )
    else:
        evaluator_base = {
            "format": evaluator.FORMAT,
            "benchmark": evaluator.BENCHMARK,
            "task": evaluator.TASK,
            "actor_provenance": actor,
            "critic_folds": folds,
            "policy_rows": rows,
            "oracle_groups": oracle,
        }
        evaluator_input = {
            **evaluator_base,
            "document_sha256": evaluator.canonical_sha256(evaluator_base),
        }
        report = evaluator.build_report(evaluator_input)
        report_base = dict(report)
        report_base.pop("report_sha256")
        report_base["materialization_logical_sha256"] = materialization[
            "logical_sha256"
        ]
        report = {
            **report_base,
            "report_sha256": evaluator.canonical_sha256(report_base),
        }
    return materialization, report


def write_create_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    output = path.expanduser().resolve()
    payload = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.suffix != ".json" or output.is_symlink():
        raise FinalReportMaterializationError(f"{label} must be one real .json path")
    if output.exists():
        if not output.is_file() or output.read_text(encoding="utf-8") != payload:
            raise FinalReportMaterializationError(f"existing {label} changed")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".create", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-root", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument("--oracle-truth", type=Path)
    parser.add_argument("--output-input", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    materialization, report = build_materialization(
        nested_root=args.nested_root,
        actor_authority_path=args.actor_authority,
        oracle_truth_path=args.oracle_truth,
    )
    write_create_once(args.output_input, materialization, "materialized input")
    write_create_once(args.output_report, report, "final report")
    print(
        "FINAL_N1_N4_N8_REPORT="
        + json.dumps(
            {
                "input": str(args.output_input.expanduser().resolve()),
                "report": str(args.output_report.expanduser().resolve()),
                "oracle_evidence_sufficient": report[
                    "oracle_branch_diagnostic"
                ]["evidence_sufficient"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
