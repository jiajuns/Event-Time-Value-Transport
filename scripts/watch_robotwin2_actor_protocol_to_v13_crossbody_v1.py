#!/usr/bin/env python3
"""Continue the complete v13 cross-body pipeline after actor protocol selection.

This remote, standard-library watcher waits for the authenticated 200-pair
execute-5/execute-50 comparison, freezes its prospectively defined winner as
one immutable actor-execution protocol, runs the complete 8,000-branch primary
collection, then invokes the protocol-bound supplement/LOBO/N1-N4-N8 watcher.
v13 keeps the v12 competing-risks model and adds source-only causal-stratum
proper-loss balancing.  It never selects a protocol from critic metrics and
never changes the selected stride downstream.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import robotwin2_actor_execution_protocol_v1 as actor_execution


FORMAT = "etsf_robotwin2_actor_protocol_to_v13_crossbody_watcher_v1"
SELECTION_FORMAT = "etsf_robotwin2_actor_execution_protocol_selection_receipt_v1"
ACTOR_REPORT_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_report_v1"
ACTOR_OUTCOME_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_outcomes_v1"
ACTOR_BINDING_FORMAT = (
    "etsf_robotwin2_actor_deployment_protocol_binding_v2_stable_roster"
)
ACTOR_COMPLETION_FORMAT = "etsf_robotwin2_actor_execute5_vs_execute50_completion_v1"
ACTOR_REPORT_STATUS = (
    "complete_five_body_two_condition_paired_actor_deployment_report"
)
ACTOR_COMPLETION_STATUS = "complete_200_pairs_400_rollouts_frozen"
ACTOR_PAIR_COUNT = 200
ACTOR_ROLLOUT_COUNT = 400
ORDERED_SELECTION_CRITERIA = (
    "binary_success",
    "stage_progress",
    "goal_progress_m",
)
PRIMARY_WATCHER = "watch_robotwin2_ee16_actor_to_five_body_branches_v1.py"
DOWNSTREAM_WATCHER = "watch_robotwin2_postformal_shared_head_upgrade_v1.py"
FINAL_REPORT_MATERIALIZER = (
    "materialize_robotwin2_nested_n1_n4_n8_final_report_v1.py"
)
FINAL_REPORT_FORMAT = "etsf_robotwin2_five_body_lobo_n1_n4_n8_oracle_report_v1"
FINAL_REPORT_POLICY_ONLY_STATUS = (
    "complete_policy_transfer_metrics_oracle_unavailable_fail_closed"
)


class CrossbodyContinuationError(RuntimeError):
    """The actor evidence, selected protocol, or child pipeline changed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CrossbodyContinuationError(f"required real file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CrossbodyContinuationError(f"{label} is missing or symbolic")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CrossbodyContinuationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CrossbodyContinuationError(f"{label} must be a JSON object")
    return value


def verify_named_logical_sha(
    value: Mapping[str, Any], field: str, label: str
) -> None:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != canonical_sha256(unsigned):
        raise CrossbodyContinuationError(f"{label} {field} mismatch")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_once_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    payload = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CrossbodyContinuationError(f"existing {label} changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".create-{os.getpid()}")
    try:
        with temporary.open("x+b") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise CrossbodyContinuationError(f"racing {label} changed")
    finally:
        temporary.unlink(missing_ok=True)


def validate_actor_completion(
    actor_root: Path, guardian_state_path: Path
) -> dict[str, Any] | None:
    """Return authenticated report evidence, or None while still running."""

    try:
        guardian = read_json(guardian_state_path, "actor guardian state")
    except CrossbodyContinuationError:
        return None
    if guardian.get("status") == "failed":
        raise CrossbodyContinuationError(
            f"actor comparison guardian failed: {guardian.get('error')}"
        )
    completion_path = actor_root / "run.complete.json"
    if guardian.get("status") != "complete" or not completion_path.is_file():
        return None

    completion = read_json(completion_path, "actor completion receipt")
    report_path = actor_root / "paired_report.json"
    outcome_path = actor_root / "paired_outcomes.json"
    binding_path = actor_root / "immutable_deployment_binding.json"
    binding = read_json(binding_path, "actor immutable deployment binding")
    report = read_json(report_path, "actor paired report")
    outcome = read_json(outcome_path, "actor paired outcomes")
    verify_named_logical_sha(completion, "logical_sha256", "actor completion")
    verify_named_logical_sha(report, "report_sha256", "actor report")
    verify_named_logical_sha(outcome, "document_sha256", "actor outcomes")
    verify_named_logical_sha(binding, "logical_sha256", "actor binding")
    if (
        completion.get("format") != ACTOR_COMPLETION_FORMAT
        or completion.get("status") != ACTOR_COMPLETION_STATUS
        or completion.get("pair_count") != ACTOR_PAIR_COUNT
        or completion.get("rollout_count") != ACTOR_ROLLOUT_COUNT
        or completion.get("binding_file_sha256") != sha256_file(binding_path)
        or completion.get("binding_logical_sha256")
        != binding.get("logical_sha256")
        or completion.get("outcome_file_sha256") != sha256_file(outcome_path)
        or completion.get("report_file_sha256") != sha256_file(report_path)
        or completion.get("outcome_document_sha256")
        != outcome.get("document_sha256")
        or completion.get("report_sha256") != report.get("report_sha256")
        or report.get("format") != ACTOR_REPORT_FORMAT
        or report.get("status") != ACTOR_REPORT_STATUS
        or report.get("outcome_document_sha256")
        != outcome.get("document_sha256")
        or outcome.get("format") != ACTOR_OUTCOME_FORMAT
        or outcome.get("status") != "complete_200_pairs_400_rollouts"
        or binding.get("format") != ACTOR_BINDING_FORMAT
    ):
        raise CrossbodyContinuationError("actor completion SHA chain changed")
    selection = report.get("primary_hierarchical_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("ordered_criteria") != list(ORDERED_SELECTION_CRITERIA)
        or selection.get("rule")
        != (
            "compare paired success first; only if exactly tied compare paired "
            "stage progress; only if still tied compare paired goal progress"
        )
        or selection.get("preferred_protocol")
        not in {
            actor_execution.METHOD_EXECUTE5,
            actor_execution.METHOD_EXECUTE50,
            "tie",
        }
        or isinstance(
            selection.get("selected_delta_execute50_minus_execute5"), bool
        )
        or not isinstance(
            selection.get("selected_delta_execute50_minus_execute5"),
            (int, float),
        )
    ):
        raise CrossbodyContinuationError("actor hierarchical selection changed")
    rows = outcome.get("rows")
    if not isinstance(rows, list) or len(rows) != ACTOR_PAIR_COUNT:
        raise CrossbodyContinuationError("actor outcomes do not contain 200 pairs")
    if outcome.get("rows_sha256") != canonical_sha256(rows):
        raise CrossbodyContinuationError("actor outcome row SHA changed")
    methods = (
        actor_execution.METHOD_EXECUTE5,
        actor_execution.METHOD_EXECUTE50,
    )
    observed_by_cell: dict[tuple[str, str], set[int]] = {}
    metric_deltas = {metric: [] for metric in ORDERED_SELECTION_CRITERIA}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CrossbodyContinuationError("actor outcome row is invalid")
        body = row.get("heldout_body")
        condition = row.get("condition")
        seed = row.get("requested_seed")
        rollouts = row.get("rollouts")
        if (
            body not in actor_execution.BODIES
            or condition not in actor_execution.CONDITIONS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(rollouts, Mapping)
            or set(rollouts) != set(methods)
        ):
            raise CrossbodyContinuationError("actor outcome identity/method changed")
        cell = (str(body), str(condition))
        observed_by_cell.setdefault(cell, set())
        if seed in observed_by_cell[cell]:
            raise CrossbodyContinuationError("actor outcome seed is duplicated")
        observed_by_cell[cell].add(seed)
        for metric in ORDERED_SELECTION_CRITERIA:
            left = rollouts[methods[0]].get(metric)
            right = rollouts[methods[1]].get(metric)
            if (
                isinstance(left, bool)
                or isinstance(right, bool)
                or not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
                or not math.isfinite(float(left))
                or not math.isfinite(float(right))
            ):
                raise CrossbodyContinuationError("actor outcome metric is invalid")
            metric_deltas[metric].append(float(right) - float(left))
    expected_cells = {
        (body, condition)
        for body in actor_execution.BODIES
        for condition in actor_execution.CONDITIONS
    }
    seed_sets = list(observed_by_cell.values())
    if (
        set(observed_by_cell) != expected_cells
        or any(len(seeds) != 20 for seeds in seed_sets)
        or any(seeds != seed_sets[0] for seeds in seed_sets[1:])
    ):
        raise CrossbodyContinuationError("actor outcome body/condition roster changed")
    selected_metric = ORDERED_SELECTION_CRITERIA[-1]
    selected_delta = math.fsum(metric_deltas[selected_metric]) / len(rows)
    for metric in ORDERED_SELECTION_CRITERIA[:-1]:
        delta = math.fsum(metric_deltas[metric]) / len(rows)
        if abs(delta) > 0.0:
            selected_metric = metric
            selected_delta = delta
            break
    computed_preferred = (
        methods[1]
        if selected_delta > 0.0
        else methods[0]
        if selected_delta < 0.0
        else "tie"
    )
    if (
        selection.get("selected_criterion") != selected_metric
        or not math.isclose(
            float(selection.get("selected_delta_execute50_minus_execute5")),
            selected_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or selection.get("preferred_protocol") != computed_preferred
    ):
        raise CrossbodyContinuationError(
            "actor report selection does not replay from complete outcomes"
        )
    return {
        "completion_path": str(completion_path),
        "completion_file_sha256": sha256_file(completion_path),
        "completion_logical_sha256": completion["logical_sha256"],
        "report_path": str(report_path),
        "report_file_sha256": sha256_file(report_path),
        "report_sha256": report["report_sha256"],
        "outcome_path": str(outcome_path),
        "outcome_file_sha256": sha256_file(outcome_path),
        "outcome_document_sha256": outcome["document_sha256"],
        "binding_path": str(binding_path),
        "binding_file_sha256": sha256_file(binding_path),
        "primary_hierarchical_selection": dict(selection),
    }


def materialize_selected_protocol(
    evidence: Mapping[str, Any], *, run_root: Path, path_root: Path
) -> tuple[Path, str, dict[str, Any]]:
    selection = evidence["primary_hierarchical_selection"]
    preferred = selection["preferred_protocol"]
    if preferred == "tie":
        raise CrossbodyContinuationError(
            "actor comparison is tied; no protocol may be selected automatically"
        )
    selected_method = preferred
    protocol = actor_execution.execution_protocol_for_actor_report_method(
        selected_method
    )
    protocol_path = run_root / "actor_execution_protocol.json"
    protocol_file_sha256 = actor_execution.write_execution_protocol_file(
        protocol_path, protocol
    )
    binding = actor_execution.execution_protocol_file_binding(
        protocol_path,
        protocol_file_sha256,
        path_root=path_root,
        expected_actor_report_method=selected_method,
    )
    base = {
        "format": SELECTION_FORMAT,
        "selected_before_primary_collection_or_shared_head_training": True,
        "selection_source": "complete_200_pair_actor_execute5_vs_execute50_report",
        "ordered_criteria": list(ORDERED_SELECTION_CRITERIA),
        "reported_preferred_protocol": preferred,
        "tie_behavior": "fail_closed_without_downstream_collection_or_training",
        "selected_actor_report_method": selected_method,
        "selected_stride": protocol["stride"],
        "selection_implementation_path": str(Path(__file__).resolve()),
        "selection_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "actor_evidence": dict(evidence),
        "actor_execution_protocol_binding": binding,
    }
    receipt = {**base, "logical_sha256": canonical_sha256(base)}
    create_once_json(
        run_root / "actor_execution_protocol_selection_receipt.json",
        receipt,
        "actor protocol selection receipt",
    )
    return protocol_path, protocol_file_sha256, receipt


def primary_command(
    args: argparse.Namespace, protocol_path: Path, protocol_sha256: str
) -> list[str]:
    return [
        str(args.robotwin_python),
        str(args.code_root / PRIMARY_WATCHER),
        "--actor-execution-protocol",
        str(protocol_path),
        "--actor-execution-protocol-sha256",
        protocol_sha256,
        "--path-root",
        str(args.path_root),
        "--run-root",
        str(args.run_root),
    ]


def downstream_command(
    args: argparse.Namespace, protocol_path: Path, protocol_sha256: str
) -> list[str]:
    root = args.run_root
    return [
        str(args.system_python),
        str(args.code_root / DOWNSTREAM_WATCHER),
        "--upstream-state",
        str(root / "watcher_state.json"),
        "--upstream-kind",
        "primary_collection",
        "--code-root",
        str(args.code_root),
        "--code-commit-marker",
        args.code_commit_marker,
        "--primary-branches-root",
        str(root / "primary_branches"),
        "--primary-binding",
        str(root / "primary_training_binding.json"),
        "--actor-execution-protocol",
        str(protocol_path),
        "--actor-execution-protocol-sha256",
        protocol_sha256,
        "--path-root",
        str(args.path_root),
        "--actor-authority",
        str(root / "actor_authority.json"),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--materialization-receipt",
        str(args.materialization_receipt),
        "--robotwin-root",
        str(args.robotwin_root),
        "--vlm-metadata",
        str(args.vlm_metadata),
        "--event-spec",
        str(args.event_spec),
        "--metrics-preregistration",
        str(args.metrics_preregistration),
        "--supplement-root",
        str(root / "supplement_branches"),
        "--supplement-binding",
        str(root / "supplement_binding.json"),
        "--augmented-lobo-root",
        str(root / "lobo_v13"),
        "--augmented-lobo-state",
        str(root / "lobo_v13.watcher_state.json"),
        "--augmented-lobo-run-exit",
        str(root / "lobo_v13.run.exit"),
        "--augmented-n4-root",
        str(root / "formal_n4"),
        "--augmented-n8-root",
        str(root / "nested_n1_n4_n8"),
        "--state",
        str(root / "v13_crossbody.watcher_state.json"),
        "--run-exit",
        str(root / "v13_crossbody.run.exit"),
        "--lock",
        str(root / "v13_crossbody.lock"),
        "--log-root",
        str(root / "logs"),
    ]


def final_report_command(args: argparse.Namespace) -> list[str]:
    root = args.run_root
    nested = root / "nested_n1_n4_n8"
    return [
        str(args.system_python),
        str(args.code_root / FINAL_REPORT_MATERIALIZER),
        "--nested-root",
        str(nested),
        "--actor-authority",
        str(root / "actor_authority.json"),
        "--output-input",
        str(nested / "crossbody_final_report_input.json"),
        "--output-report",
        str(nested / "crossbody_final_report.json"),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-comparison-root", type=Path, required=True)
    parser.add_argument("--actor-guardian-state", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-commit-marker", required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--vlm-metadata", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--metrics-preregistration", type=Path, required=True)
    parser.add_argument(
        "--robotwin-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/RoboTwin2/bin/python"),
    )
    parser.add_argument(
        "--system-python", type=Path, default=Path("/usr/bin/python3")
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "actor_comparison_root",
        "actor_guardian_state",
        "code_root",
        "path_root",
        "run_root",
        "state",
        "lock",
        "actor_checkpoint",
        "materialization_receipt",
        "robotwin_root",
        "vlm_metadata",
        "event_spec",
        "metrics_preregistration",
        "robotwin_python",
        "system_python",
    ):
        raw = getattr(args, name).expanduser()
        if raw.is_symlink() and name not in {"robotwin_python", "system_python"}:
            raise CrossbodyContinuationError(f"{name} may not be symbolic")
        setattr(args, name, raw.resolve())
    if (
        len(args.code_commit_marker) != 40
        or any(character not in "0123456789abcdef" for character in args.code_commit_marker)
        or args.poll_seconds <= 0
        or not args.path_root.is_dir()
        or args.run_root.parent != args.path_root
        or args.run_root == args.path_root
        or not args.code_root.is_dir()
    ):
        raise CrossbodyContinuationError("continuation paths/commit/poll contract is invalid")
    for name in (
        "actor_checkpoint",
        "materialization_receipt",
        "robotwin_root",
        "vlm_metadata",
        "event_spec",
        "metrics_preregistration",
        "robotwin_python",
        "system_python",
    ):
        if not getattr(args, name).exists():
            raise CrossbodyContinuationError(f"required input is missing: {name}")
    for name in (PRIMARY_WATCHER, DOWNSTREAM_WATCHER, FINAL_REPORT_MATERIALIZER):
        path = args.code_root / name
        if path.is_symlink() or not path.is_file():
            raise CrossbodyContinuationError(f"deployed code is missing {name}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = normalize_args(parse_args(argv))
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = args.lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise CrossbodyContinuationError("another continuation watcher is active") from error

    def write_state(status: str, **extra: Any) -> None:
        atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "actor_comparison_root": str(args.actor_comparison_root),
                "run_root": str(args.run_root),
                "code_root": str(args.code_root),
                "code_commit_marker": args.code_commit_marker,
                **extra,
            },
        )

    evidence = None
    while evidence is None:
        evidence = validate_actor_completion(
            args.actor_comparison_root, args.actor_guardian_state
        )
        if evidence is None:
            write_state("waiting_for_authenticated_200_pair_actor_protocol_report")
            time.sleep(args.poll_seconds)
    args.run_root.mkdir(parents=False, exist_ok=True)
    protocol_path, protocol_sha256, receipt = materialize_selected_protocol(
        evidence,
        run_root=args.run_root,
        path_root=args.path_root,
    )
    write_state(
        "actor_protocol_frozen",
        actor_execution_protocol_binding=receipt[
            "actor_execution_protocol_binding"
        ],
        selection_receipt=str(
            args.run_root / "actor_execution_protocol_selection_receipt.json"
        ),
    )

    primary = primary_command(args, protocol_path, protocol_sha256)
    primary_log = args.run_root / "primary_collection.log"
    with primary_log.open("a", encoding="utf-8") as stream:
        write_state("running_complete_primary_8000_branch_collection", command=primary)
        result = subprocess.run(
            primary,
            cwd=args.robotwin_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise CrossbodyContinuationError(
            f"primary collection watcher exited {result.returncode}"
        )

    downstream = downstream_command(args, protocol_path, protocol_sha256)
    downstream_log = args.run_root / "v13_crossbody_pipeline.log"
    with downstream_log.open("a", encoding="utf-8") as stream:
        write_state(
            "running_supplement_v13_lobo_and_nested_n1_n4_n8",
            command=downstream,
        )
        result = subprocess.run(
            downstream,
            cwd=args.code_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise CrossbodyContinuationError(
            f"v13 cross-body watcher exited {result.returncode}"
        )
    final_state = read_json(
        args.run_root / "v13_crossbody.watcher_state.json",
        "v13 cross-body final state",
    )
    if final_state.get("status") != "complete":
        raise CrossbodyContinuationError("v13 cross-body watcher returned without completion")
    final_report = final_report_command(args)
    final_report_log = args.run_root / "final_crossbody_report.log"
    with final_report_log.open("a", encoding="utf-8") as stream:
        write_state(
            "materializing_n1_n4_n8_crossbody_final_report",
            command=final_report,
        )
        result = subprocess.run(
            final_report,
            cwd=args.code_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise CrossbodyContinuationError(
            f"final cross-body report materializer exited {result.returncode}"
        )
    final_report_path = (
        args.run_root / "nested_n1_n4_n8" / "crossbody_final_report.json"
    )
    final_report_value = read_json(final_report_path, "final cross-body report")
    verify_named_logical_sha(
        final_report_value, "report_sha256", "final cross-body report"
    )
    oracle = final_report_value.get("oracle_branch_diagnostic")
    if (
        final_report_value.get("format") != FINAL_REPORT_FORMAT
        or final_report_value.get("status") != FINAL_REPORT_POLICY_ONLY_STATUS
        or not isinstance(oracle, Mapping)
        or oracle.get("evidence_sufficient") is not False
        or oracle.get("oracle_regret_reported") is not False
        or oracle.get("oracle_regret") is not None
    ):
        raise CrossbodyContinuationError(
            "final cross-body report did not fail closed without oracle truth"
        )
    write_state(
        "complete",
        actor_execution_protocol_binding=receipt[
            "actor_execution_protocol_binding"
        ],
        downstream_state=str(args.run_root / "v13_crossbody.watcher_state.json"),
        downstream_state_file_sha256=sha256_file(
            args.run_root / "v13_crossbody.watcher_state.json"
        ),
        nested_report=final_state.get("nested_actor_n4_n8_report"),
        final_crossbody_report=str(final_report_path),
        final_crossbody_report_file_sha256=sha256_file(final_report_path),
        final_crossbody_report_sha256=final_report_value["report_sha256"],
        oracle_evidence_sufficient=False,
    )
    lock_stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
