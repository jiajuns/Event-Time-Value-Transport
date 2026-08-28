#!/usr/bin/env python3
"""Safety-gated remote launcher for formal schema-v5 counterfactual tuning.

The launcher is deliberately non-resumable because the current counterfactual
trainer has no resume contract.  A partial output is refused.  A cryptographically
consistent completed output is skipped.  Use ``--dry-run`` for CPU-only preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import torch


SCHEMA_VERSION = 5
EXPECTED_GROUPS = 100
SEEDS = (20260827, 20260828, 20260829)
LANGUAGE_CONTRACT = "same_instruction_for_initial_query_and_all_candidate_branches"
INTERVENTION = "candidate_first_chunk_then_deterministic_actor"
SCORING_GRID_VERSION = "validation_scoring_grid_v1"
GUARD_GRID_VERSION = "validation_guard_quantile_grid_v1"
SCORING_SELECTION_RULE = (
    "eligible_if_proposals_coverage_lcb_pass_then_lexicographic_"
    "lcb90_policy_success_mean_delta_conservative_grid_order"
)
MEMBER_SELECTION_RULE = (
    "validation_only_pure_success_pair_lcb90_then_top1_uplift_then_event_then_total_v2"
)
SCORING_CANDIDATE_IDS = (
    "success_only",
    "success_distance",
    "progress_light",
    "progress",
    "progress_clock",
    "full_light",
    "full",
)
SCORING_WEIGHTS = {
    "success_only": (0.0, 0.0, 0.0),
    "success_distance": (0.0, 0.0, 0.02),
    "progress_light": (0.10, 0.0, 0.0),
    "progress": (0.25, 0.0, 0.0),
    "progress_clock": (0.25, 0.05, 0.0),
    "full_light": (0.10, 0.02, 0.02),
    "full": (0.25, 0.05, 0.02),
}
DEFAULT_PYTHON = Path(
    "/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python"
)
DEFAULT_CODE_ROOT = Path("/home/user/etsf_event_world_model_code_20260827")
DEFAULT_DATA = Path("/home/user/etsf_openvla_event_branches_v5_train100_20260827")
DEFAULT_FACTUAL_ROOT = Path(
    "/home/user/etsf_openvla_structured_event_world_model_move_can_pot_"
    "sealed_schema3_20260827"
)
DEFAULT_EVENT_SPEC = Path("/home/user/etsf_stage2_run_20260825/event_spec.json")
DEFAULT_OUTPUT = Path(
    "/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry1_20260827"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_recorded_path(recorded: str, anchor: Path) -> Path:
    path = Path(recorded).expanduser()
    if path.is_file():
        return path.resolve()
    portable = anchor / path.name
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(f"recorded artifact unavailable: {path} or {portable}")


def collector_ready(root: Path) -> tuple[bool, str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False, "collector manifest is absent"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"collector manifest is not readable yet: {error}"
    if manifest.get("status") != "complete":
        return False, f"collector status={manifest.get('status')!r}"
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("collector completed with an obsolete schema")
    if int(manifest.get("completed", -1)) != EXPECTED_GROUPS:
        raise RuntimeError("completed collector does not contain exactly 100 groups")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or len(groups) != EXPECTED_GROUPS:
        raise RuntimeError("completed collector group manifest is not exactly 100 rows")
    return True, "complete"


def wait_for_collector(
    root: Path, timeout_seconds: float, poll_seconds: float
) -> None:
    started = time.monotonic()
    last_reason = "not checked"
    while True:
        ready, last_reason = collector_ready(root)
        if ready:
            return
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"schema-v5 collector was not ready within {timeout_seconds}s: {last_reason}"
            )
        remaining = timeout_seconds - elapsed
        delay = min(poll_seconds, remaining, 60.0)
        print(
            "WAITING_FOR_SCHEMA_V5="
            + json.dumps(
                {"root": str(root), "reason": last_reason, "elapsed_seconds": elapsed},
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(delay, 0.01))


def audit_collector(root: Path, event_spec_sha256: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "completed": EXPECTED_GROUPS,
        "candidate_count": 4,
        "language_contract": LANGUAGE_CONTRACT,
        "intervention": INTERVENTION,
        "event_spec_sha256": event_spec_sha256,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"collector contract mismatch for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    registry = manifest.get("seed_registry")
    if registry is None:
        fresh_fields = {
            "fresh_seed_manifest",
            "fresh_seed_manifest_sha256",
        }
        if fresh_fields & set(manifest):
            raise RuntimeError(
                "legacy collector provenance is ambiguous because fresh fields are present"
            )
        registry_provenance = "legacy_missing_pending_factual_train_crosscheck"
    elif registry == "official_150":
        if manifest.get("fresh_seed_manifest") not in (None, "") or manifest.get(
            "fresh_seed_manifest_sha256"
        ) not in (None, ""):
            raise RuntimeError("official collector unexpectedly binds a fresh manifest")
        registry_provenance = "explicit_official_150"
    else:
        raise RuntimeError(
            f"collector seed_registry is not accepted for train100: {registry!r}"
        )
    groups = manifest.get("groups")
    requested = [int(value) for value in manifest.get("requested_seeds", [])]
    resolved = [int(value) for value in manifest.get("resolved_seeds", [])]
    if (
        not isinstance(groups, list)
        or len(groups) != EXPECTED_GROUPS
        or len(requested) != EXPECTED_GROUPS
        or len(resolved) != EXPECTED_GROUPS
        or len(set(requested)) != EXPECTED_GROUPS
        or len(set(resolved)) != EXPECTED_GROUPS
    ):
        raise RuntimeError("collector seed/group cardinality contract failed")
    manifested_paths: list[Path] = []
    manifested_resolved: list[int] = []
    for item in groups:
        if not isinstance(item, Mapping) or not item.get("path"):
            raise RuntimeError("invalid collector group row")
        path = root / "groups" / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        manifested_paths.append(path.resolve())
        manifested_resolved.append(int(item["resolved_seed"]))
    if manifested_resolved != resolved:
        raise RuntimeError("collector group order differs from resolved seed manifest")
    discovered = sorted((root / "groups").glob("*.hdf5"))
    if len(discovered) != EXPECTED_GROUPS or {
        path.resolve() for path in discovered
    } != set(manifested_paths):
        raise RuntimeError("collector has missing or extra HDF5 group files")
    for path, expected_seed in zip(manifested_paths, resolved):
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise RuntimeError(f"non-v5 group in formal train root: {path}")
            if int(handle.attrs.get("resolved_seed", -1)) != expected_seed:
                raise RuntimeError(f"group resolved seed mismatch: {path}")
            if int(handle.attrs.get("candidate_count", -1)) != 4:
                raise RuntimeError(f"group candidate count mismatch: {path}")
            if str(handle.attrs.get("language_contract", "")) != LANGUAGE_CONTRACT:
                raise RuntimeError(f"group language contract mismatch: {path}")
            if not bool(handle.attrs.get("branch_instruction_consistent", False)):
                raise RuntimeError(f"branch language inconsistency: {path}")
    return {
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "schema_version": SCHEMA_VERSION,
        "groups": EXPECTED_GROUPS,
        "candidate_count": 4,
        "requested_seeds": requested,
        "resolved_seeds": resolved,
        "seed_registry": registry,
        "seed_registry_provenance": registry_provenance,
        "event_spec_sha256": event_spec_sha256,
        "labels_read_by_launcher": False,
        "hdf5_access": "identity_attrs_only",
    }


def discover_factual_summaries(root: Path) -> list[Path]:
    expected = [
        (root / f"seed_{seed}" / "training_summary.json").resolve()
        for seed in SEEDS
    ]
    discovered = {
        path.resolve() for path in root.glob("seed_*/training_summary.json")
    }
    extras = sorted(discovered - set(expected))
    missing = [path for path in expected if path not in discovered]
    if extras or missing:
        raise RuntimeError(
            "factual summaries must be exactly the frozen seed members "
            f"{list(SEEDS)}; missing={missing}, extras={extras}"
        )
    return expected


def factual_members_ready(root: Path) -> tuple[bool, str]:
    """Cheap readiness check used while the three-seed factual queue runs."""

    expected = [
        (root / f"seed_{seed}" / "training_summary.json").resolve()
        for seed in SEEDS
    ]
    discovered = {
        path.resolve() for path in root.glob("seed_*/training_summary.json")
    }
    extras = sorted(discovered - set(expected))
    if extras:
        raise RuntimeError(
            f"unexpected factual seed summaries under {root}: {extras}"
        )
    missing = [path for path in expected if path not in discovered]
    if missing:
        return False, f"factual summaries={len(discovered)}/3; next_missing={missing[0]}"
    for path in expected:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return False, f"factual summary is not readable yet: {path}: {error}"
        status = summary.get("status")
        if status != "training_complete":
            return False, f"factual status={status!r}: {path}"
    return True, "complete"


def wait_for_factual_members(
    root: Path, timeout_seconds: float, poll_seconds: float
) -> None:
    started = time.monotonic()
    while True:
        ready, reason = factual_members_ready(root)
        if ready:
            return
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"structured factual ensemble was not ready within "
                f"{timeout_seconds}s: {reason}"
            )
        remaining = timeout_seconds - elapsed
        delay = min(poll_seconds, remaining, 60.0)
        print(
            "WAITING_FOR_STRUCTURED_FACTUAL="
            + json.dumps(
                {"root": str(root), "reason": reason, "elapsed_seconds": elapsed},
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(delay, 0.01))


def query_other_compute_pids(gpu_index: int = 0) -> list[int]:
    """Return compute PIDs on the selected GPU, excluding this launcher.

    ``torch.cuda.get_device_name`` may create a CUDA context for the launcher
    itself, so its PID is explicitly ignored.  Any unparsable ``nvidia-smi``
    output is treated as a hard safety failure rather than as an idle device.
    """

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to audit GPU compute processes with nvidia-smi") from error
    pids: set[int] = set()
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("no running processes"):
            continue
        try:
            pid = int(line)
        except ValueError as error:
            raise RuntimeError(f"unrecognized nvidia-smi compute PID row: {line!r}") from error
        if pid > 0 and pid != os.getpid():
            pids.add(pid)
    return sorted(pids)


def wait_for_gpu_idle(
    timeout_seconds: float,
    poll_seconds: float,
    *,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """Wait until factual training has released the target GPU."""

    started = time.monotonic()
    checks = 0
    while True:
        checks += 1
        active = query_other_compute_pids(gpu_index)
        elapsed = time.monotonic() - started
        if not active:
            return {
                "gpu_index": gpu_index,
                "other_compute_pids": [],
                "checks": checks,
                "waited_seconds": elapsed,
                "status": "idle_before_counterfactual_launch",
            }
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"GPU {gpu_index} remained occupied for {timeout_seconds}s; "
                f"other compute PIDs={active}"
            )
        remaining = timeout_seconds - elapsed
        delay = min(poll_seconds, remaining, 60.0)
        print(
            "WAITING_FOR_GPU_RELEASE="
            + json.dumps(
                {
                    "gpu_index": gpu_index,
                    "other_compute_pids": active,
                    "elapsed_seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(max(delay, 0.01))


def audit_factual_summaries(
    root: Path, event_spec_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    shared_contract: Mapping[str, Any] | None = None
    for expected_seed, summary_path in zip(SEEDS, discover_factual_summaries(root)):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "training_complete":
            raise RuntimeError(f"factual member is incomplete: {summary_path}")
        if summary.get("sealed_test_evaluated") is not False:
            raise RuntimeError(f"factual member touched sealed test: {summary_path}")
        contract = summary.get("contract")
        if not isinstance(contract, Mapping) or contract.get("event_mode") != "structured":
            raise RuntimeError(f"absolute-event factual member is forbidden: {summary_path}")
        if int(contract.get("training_seed", -1)) != expected_seed:
            raise RuntimeError(
                f"factual training_seed does not match directory seed {expected_seed}: "
                f"{summary_path}"
            )
        if str(contract.get("event_spec_sha256", "")) != event_spec_sha256:
            raise RuntimeError(f"factual event-spec mismatch: {summary_path}")
        shared_view = {
            key: value for key, value in contract.items() if key != "training_seed"
        }
        if shared_contract is None:
            shared_contract = shared_view
        elif json.dumps(shared_contract, sort_keys=True) != json.dumps(
            shared_view, sort_keys=True
        ):
            raise RuntimeError("three factual members do not share one data/split contract")
        score = float(summary.get("best_validation_selection_score", math.inf))
        if not math.isfinite(score):
            raise RuntimeError(f"factual member has no finite best score: {summary_path}")
        checkpoint = resolve_recorded_path(str(summary.get("checkpoint", "")), summary_path.parent)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config") if isinstance(payload, Mapping) else None
        checkpoint_contract = payload.get("contract") if isinstance(payload, Mapping) else None
        if not isinstance(config, Mapping) or config.get("structured_events") is not True:
            raise RuntimeError(f"best checkpoint is not structured: {checkpoint}")
        if not isinstance(checkpoint_contract, Mapping) or checkpoint_contract.get(
            "event_mode"
        ) != "structured":
            raise RuntimeError(f"checkpoint contract is not structured: {checkpoint}")
        if int(checkpoint_contract.get("training_seed", -1)) != expected_seed:
            raise RuntimeError(
                f"checkpoint training_seed does not match directory seed: {checkpoint}"
            )
        if json.dumps(checkpoint_contract, sort_keys=True) != json.dumps(contract, sort_keys=True):
            raise RuntimeError(f"summary/checkpoint contract mismatch: {checkpoint}")
        checkpoint_score = float(payload.get("best_score", math.inf))
        if not math.isfinite(checkpoint_score) or not math.isclose(
            checkpoint_score, score, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError(f"summary/checkpoint best score mismatch: {checkpoint}")
        rows.append(
            {
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "best_validation_selection_score": score,
                "best_step": int(summary.get("best_step", -1)),
                "training_seed": expected_seed,
                "structured_events": True,
                "status": "training_complete",
            }
        )
    assert shared_contract is not None
    rows.sort(
        key=lambda row: (
            row["best_validation_selection_score"], row["checkpoint"]
        )
    )
    return rows, rows[0]


def crosscheck_training_seeds(
    collector: Mapping[str, Any], factual_contract: Mapping[str, Any]
) -> dict[str, Any]:
    train_values = [int(seed) for seed in factual_contract.get("train_seeds", [])]
    train = set(train_values)
    validation = {int(seed) for seed in factual_contract.get("validation_seeds", [])}
    sealed = {int(seed) for seed in factual_contract.get("sealed_test_seeds", [])}
    requested_values = [int(seed) for seed in collector["requested_seeds"]]
    resolved_values = [int(seed) for seed in collector["resolved_seeds"]]
    requested = set(requested_values)
    resolved = set(resolved_values)
    if (
        len(train_values) != EXPECTED_GROUPS
        or len(train) != EXPECTED_GROUPS
        or len(requested_values) != EXPECTED_GROUPS
        or len(resolved_values) != EXPECTED_GROUPS
        or requested != train
        or resolved != train
    ):
        raise RuntimeError("schema-v5 train100 is not exactly the factual training split")
    if train & validation or train & sealed or validation & sealed:
        raise RuntimeError("factual train/validation/sealed split contains leakage")
    if (requested | resolved) & validation or (requested | resolved) & sealed:
        raise RuntimeError("schema-v5 train root includes development holdout/test seeds")
    provenance = str(collector.get("seed_registry_provenance", ""))
    if provenance not in {
        "explicit_official_150",
        "legacy_missing_pending_factual_train_crosscheck",
    }:
        raise RuntimeError("collector seed provenance was not classified safely")
    return {
        "status": (
            "legacy_schema_v5_provenance_accepted_after_exact_factual_crosscheck"
            if provenance == "legacy_missing_pending_factual_train_crosscheck"
            else "explicit_official_150_crosschecked"
        ),
        "requested_seed_set_equals_factual_train": True,
        "resolved_seed_set_equals_factual_train": True,
        "factual_train_seed_count": len(train),
        "factual_validation_seed_count": len(validation),
        "factual_sealed_seed_count": len(sealed),
        "development_split_overlap": False,
        "order_matches_factual_train_list": (
            requested_values == train_values and resolved_values == train_values
        ),
    }


def build_command(args: argparse.Namespace, selected: Mapping[str, Any]) -> list[str]:
    trainer = args.trainer.resolve()
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    # Keep the virtual-environment launcher path intact.  Resolving this
    # symlink executes the base interpreter without the venv site-packages.
    python_bin = args.python_bin.expanduser().absolute()
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    return [
        str(python_bin),
        str(trainer),
        "--data",
        str(args.data.resolve()),
        "--pretrained",
        str(selected["checkpoint"]),
        "--output",
        str(args.output.resolve()),
        "--event-spec",
        str(args.event_spec.resolve()),
        "--object-names",
        "can",
        "--seeds",
        *(str(seed) for seed in SEEDS),
        "--device",
        "cuda",
        "--amp",
        "bf16",
        "--steps",
        "3000",
        "--groups-per-batch",
        "8",
        "--learning-rate",
        "1e-4",
        "--eval-every",
        "100",
        "--num-workers",
        "2",
        "--distance-weight",
        "0.02",
        "--pairwise-weight",
        "0.75",
        "--listwise-weight",
        "0.5",
        "--group-centered-weight",
        "1.0",
        "--baseline-contrast-weight",
        "1.5",
        "--guard-min-groups",
        "10",
        "--guard-min-coverage",
        "0.10",
        "--guard-min-lcb",
        "0.0",
        "--guard-max-harmful-rate",
        "0.10",
        "--min-relative-support",
        "5",
        "--min-recovery-support",
        "5",
    ]


def validate_complete_output(
    output: Path, selected_checkpoint_sha256: str, expected_data_manifest_sha256: str
) -> dict[str, Any] | None:
    if not output.exists():
        return None
    entries = list(output.iterdir()) if output.is_dir() else []
    if not entries:
        return None
    manifest_path = output / "ensemble_manifest.json"
    audit_path = output / "launch_audit.json"
    if not manifest_path.is_file() or not audit_path.is_file():
        raise RuntimeError(
            "counterfactual output is partial/conflicting and trainer has no resume; refusing"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "launcher_complete":
        raise RuntimeError("counterfactual launch audit is incomplete; safe resume is unavailable")
    if str(audit.get("selected_factual", {}).get("checkpoint_sha256", "")) != (
        selected_checkpoint_sha256
    ):
        raise RuntimeError("existing launch audit selected a different factual checkpoint")
    if manifest.get("format") != "etsf_counterfactual_ensemble_v1":
        raise RuntimeError("existing output has an unexpected ensemble format")
    contract = manifest.get("contract")
    config = manifest.get("config")
    if not isinstance(contract, Mapping) or not isinstance(config, Mapping):
        raise RuntimeError("existing ensemble lacks config/contract")
    if config.get("structured_events") is not True:
        raise RuntimeError("existing ensemble is an absolute-event model")
    if str(contract.get("pretrained_sha256", "")) != selected_checkpoint_sha256:
        raise RuntimeError("existing ensemble used a different factual checkpoint")
    if str(audit.get("data", {}).get("manifest_sha256", "")) != expected_data_manifest_sha256:
        raise RuntimeError("existing ensemble used a different schema-v5 dataset")
    if int(contract.get("schema_counts", {}).get("5", 0)) != EXPECTED_GROUPS:
        raise RuntimeError("existing ensemble was not trained on exactly 100 schema-v5 groups")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(SEEDS):
        raise RuntimeError("existing ensemble does not contain three members")
    if [int(member.get("seed", -1)) for member in members if isinstance(member, Mapping)] != list(SEEDS):
        raise RuntimeError("existing ensemble member seeds differ from the frozen launcher seeds")
    if manifest.get("test_policy") != (
        "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
    ):
        raise RuntimeError("existing ensemble lacks the sealed-data holdout contract")
    aggregate = manifest.get("ensemble_checkpoint")
    if not isinstance(aggregate, Mapping) or not aggregate.get("path"):
        raise RuntimeError("existing ensemble lacks aggregate checkpoint provenance")
    aggregate_path = resolve_recorded_path(str(aggregate["path"]), output)
    if sha256(aggregate_path) != str(aggregate.get("sha256", "")):
        raise RuntimeError("existing aggregate checkpoint SHA256 mismatch")
    scoring_selection = manifest.get("scoring_selection")
    scoring = manifest.get("scoring")
    guard = manifest.get("guard")
    if not all(
        isinstance(value, Mapping)
        for value in (scoring_selection, scoring, guard)
    ):
        raise RuntimeError("existing ensemble lacks frozen scoring/guard audit")
    if scoring_selection.get("grid_version") != SCORING_GRID_VERSION:
        raise RuntimeError("existing ensemble used an unregistered scoring search")
    scoring_candidates = scoring_selection.get("candidates", [])
    if (
        not isinstance(scoring_candidates, list)
        or len(scoring_candidates) != 7
        or not all(isinstance(row, Mapping) for row in scoring_candidates)
    ):
        raise RuntimeError("existing ensemble scoring grid is incomplete")
    expected_candidate_ids = list(SCORING_CANDIDATE_IDS)
    if [row.get("candidate_id") for row in scoring_candidates] != expected_candidate_ids:
        raise RuntimeError("existing ensemble scoring grid ids/order differ")
    for row in scoring_candidates:
        expected_weights = SCORING_WEIGHTS[str(row["candidate_id"])]
        actual_weights = tuple(
            float(row.get(key, float("nan")))
            for key in (
                "event_weight",
                "duration_weight",
                "candidate_distance_weight",
            )
        )
        if actual_weights != expected_weights:
            raise RuntimeError("existing ensemble scoring grid weights differ")
    if scoring_selection.get("selection_rule") != SCORING_SELECTION_RULE:
        raise RuntimeError("existing ensemble scoring selection rule differs")
    if scoring_selection.get("minimum_proposals") != 10:
        raise RuntimeError("existing ensemble scoring gate does not require ten proposals")
    if float(scoring_selection.get("minimum_coverage", -1.0)) != 0.10:
        raise RuntimeError("existing ensemble scoring coverage floor differs")
    if float(scoring_selection.get("minimum_lcb90", -1.0)) != 0.0:
        raise RuntimeError("existing ensemble scoring LCB floor differs")
    if scoring.get("candidate_id") != scoring_selection.get("selected_candidate_id"):
        raise RuntimeError("existing ensemble selected scoring id is inconsistent")
    selected_rows = [
        row
        for row in scoring_candidates
        if row.get("candidate_id") == scoring.get("candidate_id")
    ]
    if len(selected_rows) != 1 or any(
        float(scoring.get(key, float("nan"))) != float(selected_rows[0].get(key, float("nan")))
        for key in ("event_weight", "duration_weight", "candidate_distance_weight")
    ):
        raise RuntimeError("existing ensemble scoring weights do not match selected audit row")
    scoring_contract = contract.get("scoring_selection_contract")
    if not isinstance(scoring_contract, Mapping) or scoring_contract.get(
        "selection_data"
    ) != "validation_only_no_sealed_test":
        raise RuntimeError("existing ensemble lacks validation-only scoring contract")
    if scoring_contract.get("grid_version") != SCORING_GRID_VERSION:
        raise RuntimeError("existing ensemble scoring contract version differs")
    if scoring_contract.get("selection_rule") != SCORING_SELECTION_RULE:
        raise RuntimeError("existing ensemble scoring contract rule differs")
    if scoring_contract.get("guard_grid_version") != GUARD_GRID_VERSION:
        raise RuntimeError("existing ensemble guard contract version differs")
    if scoring_contract.get("grid_candidate_ids") != expected_candidate_ids:
        raise RuntimeError("existing ensemble scoring contract grid differs")
    ranking_contract = contract.get("counterfactual_ranking_contract")
    if ranking_contract is not None:
        if not isinstance(ranking_contract, Mapping):
            raise RuntimeError("existing ensemble ranking contract is invalid")
        if ranking_contract.get("member_selection_data") != (
            "validation_only_no_sealed_test"
        ) or ranking_contract.get("member_selection_rule") != MEMBER_SELECTION_RULE:
            raise RuntimeError("existing ensemble member selection contract differs")
        expected_rank_weights = {
            "pairwise": 0.75,
            "listwise": 0.5,
            "group_centered": 1.0,
            "baseline_contrast": 1.5,
        }
        if ranking_contract.get("loss_weights") != expected_rank_weights:
            raise RuntimeError("existing ensemble ranking loss weights differ")
        expected_rank_semantics = {
            "pairwise_target": (
                "success_changing_candidate_pairs_only_terminal_steps_excluded"
            ),
            "listwise_target": (
                "softmax_2x_binary_success_uniform_within_outcome_"
                "terminal_steps_excluded_normalized_by_log_candidate_count"
            ),
            "duration_supervision": (
                "dedicated_duration_head_only_not_ranking_utility"
            ),
            "validation_metrics": {
                "member_selection_primary": "pure_success_pair_lcb90",
                "pure_success_pair": (
                    "all_unordered_within_group_pairs_with_different_binary_success"
                ),
                "baseline_changing_pair": (
                    "success_changing_candidates_vs_deterministic_fallback"
                ),
                "legacy_pairwise_alias": "pure_success_pair",
            },
        }
        for key, expected in expected_rank_semantics.items():
            if ranking_contract.get(key) != expected:
                raise RuntimeError(
                    f"existing ensemble ranking semantic {key} differs"
                )
        cardinality = ranking_contract.get("candidate_cardinality")
        if not isinstance(cardinality, Mapping) or any(
            cardinality.get(key) != expected
            for key, expected in {
                "variable_candidate_count_supported": True,
                "minimum_candidates_per_group": 2,
                "unique_baseline_name": "deterministic",
                "baseline_index": 0,
                "pairwise_reduction": "mean_pairs_then_mean_groups",
                "listwise_reduction": (
                    "cross_entropy_div_log_C_then_mean_groups"
                ),
            }.items()
        ):
            raise RuntimeError("existing ensemble candidate-cardinality contract differs")
        if ranking_contract.get("action_sensitivity") != {
            "enabled": True,
            "architecture": "baseline_relative_action_effect_residual_v1",
            "inputs": [
                "action_effect_minus_deterministic_action_effect",
                "shared_semantic_times_action_effect_delta",
            ],
            "baseline_residual": 0.0,
            "absolute_success_supervision": "base_world_success_logit_only",
            "ranking_gradient": "residual_branch_with_base_score_stop_gradient",
            "deployment": "predict_candidates_adds_residual_to_success_logit",
            "event_time_object_heads": "unchanged",
        }:
            raise RuntimeError("existing ensemble action-sensitivity contract differs")
    if guard.get("minimum_guarded_groups") != 10:
        raise RuntimeError("existing ensemble guard does not require ten proposals")
    if float(guard.get("minimum_coverage", -1.0)) != 0.10:
        raise RuntimeError("existing ensemble guard coverage floor differs")
    if float(guard.get("minimum_lcb", -1.0)) != 0.0:
        raise RuntimeError("existing ensemble guard LCB floor differs")
    if float(guard.get("maximum_harmful_rate", -1.0)) != 0.10:
        raise RuntimeError("existing ensemble guard harmful-rate ceiling differs")
    if guard.get("grid_version") != GUARD_GRID_VERSION:
        raise RuntimeError("existing ensemble used an unregistered guard search")
    threshold_candidates = guard.get("threshold_candidates")
    if not isinstance(threshold_candidates, list) or len(threshold_candidates) > 9:
        raise RuntimeError("existing ensemble guard grid is incomplete or unbounded")
    aggregate_payload = torch.load(
        aggregate_path, map_location="cpu", weights_only=False
    )
    for key in ("scoring", "scoring_selection", "guard"):
        if aggregate_payload.get(key) != manifest.get(key):
            raise RuntimeError(f"existing manifest/checkpoint {key} mismatch")
    for member in members:
        if not isinstance(member, Mapping) or not member.get("path"):
            raise RuntimeError("existing member provenance is invalid")
        member_path = resolve_recorded_path(str(member["path"]), output)
        if sha256(member_path) != str(member.get("sha256", "")):
            raise RuntimeError("existing member checkpoint SHA256 mismatch")
        if ranking_contract is not None:
            member_payload = torch.load(
                member_path, map_location="cpu", weights_only=False
            )
            if member_payload.get("best_selection_rule") != MEMBER_SELECTION_RULE:
                raise RuntimeError("existing member checkpoint selection rule differs")
            selection_key = member_payload.get("best_selection_key")
            if not isinstance(selection_key, list) or len(selection_key) != 4:
                raise RuntimeError(
                    "existing member checkpoint lacks validation selection key"
                )
    return {
        "status": "already_complete_skip",
        "ensemble_manifest": str(manifest_path.resolve()),
        "ensemble_manifest_sha256": sha256(manifest_path),
        "aggregate_checkpoint": str(aggregate_path),
        "aggregate_checkpoint_sha256": sha256(aggregate_path),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    event_spec = args.event_spec.resolve()
    if not event_spec.is_file():
        raise FileNotFoundError(event_spec)
    event_spec_digest = sha256(event_spec)
    wait_for_collector(args.data.resolve(), args.wait_timeout_seconds, args.poll_seconds)
    collector = audit_collector(args.data.resolve(), event_spec_digest)
    wait_for_factual_members(
        args.factual_root.resolve(),
        args.wait_timeout_seconds,
        args.poll_seconds,
    )
    factual_rows, selected = audit_factual_summaries(
        args.factual_root.resolve(), event_spec_digest
    )
    selected_payload = torch.load(
        Path(selected["checkpoint"]), map_location="cpu", weights_only=False
    )
    factual_contract = selected_payload["contract"]
    collector["seed_registry_audit"] = crosscheck_training_seeds(
        collector, factual_contract
    )
    command = build_command(args, selected)
    return {
        "format": "etsf_counterfactual_v5_launch_audit_v1",
        "status": "preflight_complete",
        "resume_supported": False,
        "partial_output_policy": "refuse_no_safe_resume_in_trainer",
        "data": collector,
        "event_spec": {"path": str(event_spec), "sha256": event_spec_digest},
        "factual_candidates": factual_rows,
        "selection_rule": "minimum_best_validation_selection_score",
        "selected_factual": selected,
        "counterfactual_member_seeds": list(SEEDS),
        "runtime": {"device": "cuda", "amp": "bf16", "expected_gpu": "4090"},
        "trainer": str(args.trainer.resolve()),
        "output": str(args.output.resolve()),
        "command": command,
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "sealed_labels_read": False,
    }


def parse_args() -> argparse.Namespace:
    local_trainer = Path(__file__).resolve().parent / "train_openvla_etsf_counterfactual.py"
    default_python = DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else Path(sys.executable)
    default_trainer = (
        DEFAULT_CODE_ROOT / "scripts/train_openvla_etsf_counterfactual.py"
        if (DEFAULT_CODE_ROOT / "scripts/train_openvla_etsf_counterfactual.py").is_file()
        else local_trainer
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--factual-root", type=Path, default=DEFAULT_FACTUAL_ROOT)
    parser.add_argument("--event-spec", type=Path, default=DEFAULT_EVENT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trainer", type=Path, default=default_trainer)
    parser.add_argument("--python-bin", type=Path, default=default_python)
    parser.add_argument("--wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-wait-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.wait_timeout_seconds < 0
        or args.gpu_wait_timeout_seconds < 0
        or not 0 < args.poll_seconds <= 60
        or not 0 < args.gpu_poll_seconds <= 60
    ):
        raise ValueError(
            "wait timeouts must be non-negative and poll intervals in (0,60]"
        )
    audit = preflight(args)
    complete = validate_complete_output(
        args.output.resolve(),
        audit["selected_factual"]["checkpoint_sha256"],
        audit["data"]["manifest_sha256"],
    )
    if complete is not None:
        print("COUNTERFACTUAL_TRAINING_SKIP=" + json.dumps(complete, sort_keys=True))
        return
    if args.dry_run:
        print("COUNTERFACTUAL_V5_DRY_RUN=" + json.dumps(audit, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("formal launch requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "4090" not in gpu_name:
        raise RuntimeError(f"formal launch requires RTX 4090, found {gpu_name!r}")
    gpu_idle_audit = wait_for_gpu_idle(
        args.gpu_wait_timeout_seconds,
        args.gpu_poll_seconds,
        gpu_index=0,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "launch.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("launch lock already exists; concurrent/resume launch refused") from error
    with os.fdopen(lock_descriptor, "w", encoding="utf-8") as handle:
        handle.write(audit["command_sha256"] + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    audit["status"] = "launching_nonresumable"
    audit["runtime"]["gpu_name"] = gpu_name
    audit["runtime"]["gpu_idle_audit"] = gpu_idle_audit
    atomic_json(output / "launch_audit.json", audit)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
    try:
        subprocess.run(audit["command"], check=True, env=environment)
        audit["status"] = "launcher_complete"
        atomic_json(output / "launch_audit.json", audit)
        complete = validate_complete_output(
            output,
            audit["selected_factual"]["checkpoint_sha256"],
            audit["data"]["manifest_sha256"],
        )
        assert complete is not None
        print("COUNTERFACTUAL_TRAINING_COMPLETE=" + json.dumps(complete, sort_keys=True))
    except BaseException:
        audit["status"] = "failed_nonresumable_manual_new_output_required"
        atomic_json(output / "launch_audit.json", audit)
        raise


if __name__ == "__main__":
    main()
