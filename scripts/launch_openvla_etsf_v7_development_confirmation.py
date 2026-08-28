#!/usr/bin/env python3
"""Serial server orchestrator: label-free seeds -> prereg -> collect -> evaluate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from launch_openvla_etsf_counterfactual_oof_v6 import require_exclusive_idle_gpu
from openvla_etsf_v7_development_confirmation import (
    EXPECTED_GROUPS, RESULT_FORMAT, canonical_sha256, sha256,
    validate_preregistration, validate_seed_manifest,
)


FORMAT = "etsf_openvla_v7_prospective_server_launch_v1"
RECOVERY_FORMAT = "etsf_openvla_v7_resolved_seed_recovery_v1"
RECOVERY_STAGES = ("preregister", "collect", "evaluate")
RESOLVED_SEED_MARKER = "V7_SEEDS_PREREGISTERED="


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise RuntimeError(f"expected JSON object: {path}")
    return value


def _argv_sha(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), separators=(",", ":")).encode()).hexdigest()


def _validate_signed_plan(plan: Mapping[str, Any]) -> None:
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", "")
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError("v7 original launch plan signature changed")


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "candidates": args.candidates,
        "official150_manifest": args.official150_manifest,
        "development150_manifest": args.development150_manifest,
        "fresh50_manifest": args.fresh50_manifest,
        "official_seed_registry": args.official_seed_registry,
        "pretrained": args.pretrained,
        "event_spec": args.event_spec,
        "seed_resolver": args.seed_resolver,
        "collector": args.collector,
        "evaluator": args.evaluator,
        "python_bin": args.python_bin,
    }


def _validate_resolved_seed_marker(
    log_path: Path, *, seeds_path: Path, seed_contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not log_path.is_file():
        raise RuntimeError("v7 recovery requires the original resolve log")
    markers = [
        line[len(RESOLVED_SEED_MARKER) :]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(RESOLVED_SEED_MARKER)
    ]
    if len(markers) != 1:
        raise RuntimeError("v7 recovery requires exactly one resolved-seed completion marker")
    try:
        marker = json.loads(markers[0])
    except json.JSONDecodeError as error:
        raise RuntimeError("v7 resolved-seed completion marker is malformed") from error
    if not isinstance(marker, Mapping):
        raise RuntimeError("v7 resolved-seed completion marker must be an object")
    if (
        Path(str(marker.get("output", ""))).expanduser().resolve()
        != seeds_path.resolve()
        or int(marker.get("groups", -1)) != EXPECTED_GROUPS
        or marker.get("labels_read") is not False
        or marker.get("payload_sha256")
        != seed_contract.get("seed_manifest_payload_sha256")
    ):
        raise RuntimeError("v7 resolved-seed completion marker changed provenance")
    return dict(marker)


def build_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.state_root.absolute(); python = str(args.python_bin.expanduser().absolute())
    seeds, prereg, result = root / "v7_seed_manifest.json", root / "v7_preregistration.json", root / "v7_result.json"
    resolve = [python, str(args.seed_resolver.absolute()), "--candidates", str(args.candidates.absolute()),
        "--official150-manifest", str(args.official150_manifest.absolute()),
        "--development150-manifest", str(args.development150_manifest.absolute()),
        "--fresh50-manifest", str(args.fresh50_manifest.absolute()),
        "--official-seed-registry", str(args.official_seed_registry.absolute()),
        "--rlinf-root", str(args.rlinf_root.absolute()), "--robotwin-root", str(args.robotwin_root.absolute()),
        "--robotwin-code", str(args.robotwin_code.absolute()), "--output", str(seeds)]
    register = [python, str(args.evaluator.absolute()), "preregister", "--seed-manifest", str(seeds),
        "--pretrained", str(args.pretrained.absolute()), "--event-spec", str(args.event_spec.absolute()),
        "--actor-model-path", str(args.model_path.absolute()),
        "--output", str(prereg)]
    collect = [python, str(args.collector.absolute()), "--model-path", str(args.model_path.absolute()),
        "--rlinf-root", str(args.rlinf_root.absolute()), "--robotwin-root", str(args.robotwin_root.absolute()),
        "--robotwin-code", str(args.robotwin_code.absolute()), "--event-spec", str(args.event_spec.absolute()),
        "--output", str(args.collection_output.absolute()), "--task", "move_can_pot",
        "--seeds-file", str(seeds), "--seeds-key", "train", "--allow-unregistered-seeds",
        "--v7-seed-manifest", str(seeds), "--v7-preregistration", str(prereg),
        "--blends", "0.25", "0.5", "0.75", "--temperature", "0.7", "--top-k", "4"]
    evaluate = [python, str(args.evaluator.absolute()), "evaluate", "--seed-manifest", str(seeds),
        "--pretrained", str(args.pretrained.absolute()), "--event-spec", str(args.event_spec.absolute()),
        "--actor-model-path", str(args.model_path.absolute()),
        "--preregistration", str(prereg), "--data", str(args.collection_output.absolute()),
        "--output", str(result)]
    return [{"stage": name, "argv": argv, "argv_sha256": _argv_sha(argv), "uses_gpu": gpu}
            for name, argv, gpu in (("resolve_seeds", resolve, False), ("preregister", register, False),
                                    ("collect", collect, True), ("evaluate", evaluate, False))]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.state_root.exists() or args.collection_output.exists():
        raise FileExistsError("v7 state and collection outputs must both be brand new")
    files = _source_paths(args)
    directories = {"model_path": args.model_path, "rlinf_root": args.rlinf_root,
                   "robotwin_root": args.robotwin_root, "robotwin_code": args.robotwin_code}
    for name, path in files.items():
        if not path.expanduser().absolute().is_file(): raise FileNotFoundError(f"{name}: {path}")
    for name, path in directories.items():
        if not path.expanduser().absolute().exists(): raise FileNotFoundError(f"{name}: {path}")
    plan = {"format": FORMAT, "status": "preflight_complete", "serial_execution": True,
        "state_root": str(args.state_root.absolute()), "collection_output": str(args.collection_output.absolute()),
        "fresh_confirmation_inputs_accepted": False, "expected_groups": EXPECTED_GROUPS,
        "source_files": {name: {"path": str(path.expanduser().absolute()),
                                 "sha256": sha256(path.expanduser().absolute())}
                         for name, path in files.items()},
        "commands": build_commands(args),
        "terminal_status": "complete_independent_development_fresh_forbidden"}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def recovery_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Authorize only the narrow resolver-teardown recovery case.

    The original launch state and log remain immutable evidence.  Recovery is
    rejected unless the resolver produced a fully valid label-free manifest,
    printed its completion receipt, and no later-stage output exists.
    """

    root = args.state_root.expanduser().absolute()
    collection = args.collection_output.expanduser().absolute()
    if not root.is_dir():
        raise FileNotFoundError("v7 recovery requires an existing state root")
    plan_path = root / "launch_plan.json"
    state_path = root / "launch_state.json"
    if not plan_path.is_file() or not state_path.is_file():
        raise RuntimeError("v7 recovery requires the original plan and failed state")
    original_plan = _json(plan_path)
    original_state = _json(state_path)
    _validate_signed_plan(original_plan)
    if (
        original_plan.get("format") != FORMAT
        or original_plan.get("status") != "preflight_complete"
        or original_plan.get("fresh_confirmation_inputs_accepted") is not False
        or int(original_plan.get("expected_groups", -1)) != EXPECTED_GROUPS
        or Path(str(original_plan.get("state_root", ""))).resolve() != root
        or Path(str(original_plan.get("collection_output", ""))).resolve() != collection
    ):
        raise RuntimeError("v7 original launch plan is not recovery-compatible")
    expected_commands = build_commands(args)
    if original_plan.get("commands") != expected_commands:
        raise RuntimeError("v7 recovery arguments differ from the original launch plan")

    original_sources = original_plan.get("source_files")
    if not isinstance(original_sources, Mapping) or set(original_sources) != set(
        _source_paths(args)
    ):
        raise RuntimeError("v7 original source-file contract changed")
    for name, raw_path in _source_paths(args).items():
        path = raw_path.expanduser().absolute()
        row = original_sources[name]
        if not isinstance(row, Mapping):
            raise RuntimeError(f"v7 original source-file row is malformed: {name}")
        if (
            not path.is_file()
            or Path(str(row.get("path", ""))).resolve() != path.resolve()
            or row.get("sha256") != sha256(path)
        ):
            raise RuntimeError(f"v7 source file changed before recovery: {name}")

    # ``status`` is the sole plan field intentionally advanced by execute();
    # every immutable plan/provenance field must still mirror bit-for-bit.
    for key, value in original_plan.items():
        if key == "status":
            continue
        if original_state.get(key) != value:
            raise RuntimeError(f"v7 failed state no longer mirrors original plan: {key}")
    if (
        original_state.get("status") != "failed_closed_no_fresh_authorization"
        or original_state.get("current_stage") != "resolve_seeds"
        or original_state.get("last_completed_stage") not in (None, "")
        or original_state.get("stage_results") != {}
        or original_state.get("fresh_confirmation_labels_read") is not False
        or original_state.get("fresh50_confirmation_authorized") not in (None, False)
        or original_state.get("automatic_fresh_launch") not in (None, False)
        or original_state.get("error_type") != "RuntimeError"
        or "v7 stage resolve_seeds failed" not in str(original_state.get("error", ""))
    ):
        raise RuntimeError("v7 recovery accepts only a failed resolve-seeds stage")

    seeds_path = root / "v7_seed_manifest.json"
    if not seeds_path.is_file():
        raise RuntimeError("v7 failed resolver did not leave a seed manifest")
    seeds = _json(seeds_path)
    # This is intentionally independent of the old subprocess return code and
    # old launch state: every signed source file and exclusion identity is read
    # and revalidated here before any labels can be collected.
    seed_contract = validate_seed_manifest(seeds, verify_files=True)
    resolve_log = root / "logs" / "resolve_seeds.log"
    marker = _validate_resolved_seed_marker(
        resolve_log, seeds_path=seeds_path, seed_contract=seed_contract
    )

    forbidden_outputs = (
        root / "v7_preregistration.json",
        root / "v7_result.json",
        root / "v7_fixed_predictions.pt",
        root / "v7_fresh50_authorization.json",
        root / "v7_recovery_plan.json",
        root / "v7_recovery_state.json",
        root / "logs" / "preregister.log",
        root / "logs" / "collect.log",
        root / "logs" / "evaluate.log",
    )
    forbidden = tuple(
        path
        for output in forbidden_outputs
        for path in (output, output.with_suffix(output.suffix + ".partial"))
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing or collection.exists():
        raise FileExistsError(
            "v7 recovery requires prereg/result/token/collection to be absent: "
            + ", ".join(existing + ([str(collection)] if collection.exists() else []))
        )

    stages = [
        dict(stage)
        for stage in expected_commands
        if str(stage.get("stage")) in RECOVERY_STAGES
    ]
    if [str(stage["stage"]) for stage in stages] != list(RECOVERY_STAGES):
        raise RuntimeError("v7 recovery stage order changed")
    recovery = {
        "format": RECOVERY_FORMAT,
        "status": "recovery_preflight_complete",
        "scope": "validated_resolved_seeds_then_preregister_collect_evaluate_only",
        "state_root": str(root),
        "collection_output": str(collection),
        "fresh_confirmation_inputs_accepted": False,
        "fresh_confirmation_labels_read": False,
        "automatic_fresh_launch": False,
        "original_failure": {
            "launch_plan": str(plan_path),
            "launch_plan_file_sha256": sha256(plan_path),
            "launch_plan_payload_sha256": original_plan["plan_sha256"],
            "launch_state": str(state_path),
            "launch_state_file_sha256": sha256(state_path),
            "status": original_state["status"],
            "error_type": original_state["error_type"],
            "error": original_state["error"],
            "failed_stage": "resolve_seeds",
            "resolve_log": str(resolve_log),
            "resolve_log_sha256": sha256(resolve_log),
        },
        "recovered_seed_manifest": {
            "path": str(seeds_path),
            "file_sha256": sha256(seeds_path),
            "payload_sha256": seed_contract["seed_manifest_payload_sha256"],
            "requested_count": len(seed_contract["requested_seeds"]),
            "resolved_count": len(seed_contract["resolved_seeds"]),
            "completion_marker": marker,
            "validated_with_verify_files": True,
            "labels_read": False,
        },
        "commands": stages,
        # This launcher contains the recovery implementation and is already in
        # evaluator._implementation_files(), so preregistration content-addresses
        # the exact recovery logic that authorized collection.
        "recovery_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "terminal_status": "complete_independent_development_fresh_forbidden",
    }
    recovery["recovery_plan_sha256"] = canonical_sha256(recovery)
    return recovery


def validate_stage(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    root = args.state_root.absolute(); seeds_path = root / "v7_seed_manifest.json"
    if stage == "resolve_seeds":
        value = _json(seeds_path); validate_seed_manifest(value, verify_files=True)
        return {"artifact": str(seeds_path), "sha256": sha256(seeds_path)}
    if stage == "preregister":
        path = root / "v7_preregistration.json"; value = _json(path); validate_preregistration(value)
        return {"artifact": str(path), "sha256": sha256(path),
                "preregistration_sha256": value["preregistration_sha256"]}
    if stage == "collect":
        path = args.collection_output.absolute() / "manifest.json"; value = _json(path)
        if value.get("status") != "complete" or int(value.get("completed", -1)) != EXPECTED_GROUPS:
            raise RuntimeError("v7 collection incomplete")
        if value.get("seed_registry") != "explicit_v7_prospective_development" or int(
            value.get("candidate_count", -1)
        ) != 4 or any(value.get(k) not in (None, "") for k in (
            "fresh_seed_manifest", "fresh_seed_manifest_sha256")):
            raise RuntimeError("v7 collection provenance changed")
        return {"artifact": str(path), "sha256": sha256(path), "groups": EXPECTED_GROUPS}
    if stage == "evaluate":
        path = root / "v7_result.json"; value = _json(path)
        unsigned = dict(value); recorded = unsigned.pop("result_sha256", "")
        if recorded != canonical_sha256(unsigned) or value.get("format") != RESULT_FORMAT:
            raise RuntimeError("v7 result signature changed")
        authorized = value.get("authorization", {}).get("fresh50_confirmation_authorized")
        if authorized is not (value.get("metrics", {}).get("development_gate_pass") is True):
            raise RuntimeError("v7 result authorization does not mirror signed gate")
        token_path = root / "v7_fresh50_authorization.json"; token = _json(token_path)
        token_unsigned = dict(token); token_recorded = token_unsigned.pop("authorization_sha256", "")
        if token_recorded != canonical_sha256(token_unsigned):
            raise RuntimeError("v7 fresh authorization token signature changed")
        if token.get("fresh50_confirmation_authorized") is not authorized:
            raise RuntimeError("v7 fresh authorization token/result mismatch")
        if token.get("result_file_sha256") != sha256(path) or token.get(
            "preregistration_sha256"
        ) != value.get("preregistration_sha256"):
            raise RuntimeError("v7 authorization token provenance changed")
        return {"artifact": str(path), "sha256": sha256(path),
                "development_gate_pass": value.get("metrics", {}).get("development_gate_pass") is True,
                "fresh50_confirmation_authorized": authorized,
                "authorization_token": str(token_path),
                "authorization_token_sha256": sha256(token_path)}
    raise AssertionError(stage)


def run_stage(stage: Mapping[str, Any], args: argparse.Namespace, env: Mapping[str, str]):
    log = args.state_root.absolute() / "logs" / f"{stage['stage']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        done = subprocess.run(stage["argv"], stdout=handle, stderr=subprocess.STDOUT,
                              env=dict(env), check=False)
    if done.returncode: raise RuntimeError(f"v7 stage {stage['stage']} failed; see {log}")
    return {"status": "complete", "log": str(log), "log_sha256": sha256(log),
            **validate_stage(str(stage["stage"]), args)}


def execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    root = args.state_root.absolute(); root.mkdir(parents=True, exist_ok=False)
    state = {**plan, "status": "running", "stage_results": {},
             "fresh_confirmation_labels_read": False}
    atomic_json(root / "launch_plan.json", plan); atomic_json(root / "launch_state.json", state)
    env = os.environ.copy(); env.update({"PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                                        "OMP_NUM_THREADS": "8", "CUDA_VISIBLE_DEVICES": str(args.gpu_index)})
    try:
        for stage in plan["commands"]:
            name = str(stage["stage"]); state["current_stage"] = name
            if stage["uses_gpu"]:
                state.setdefault("gpu_idle_audits", {})[name] = require_exclusive_idle_gpu(args.gpu_index)
            atomic_json(root / "launch_state.json", state)
            state["stage_results"][name] = run_stage(stage, args, env)
            state["last_completed_stage"] = name; state["current_stage"] = None
            atomic_json(root / "launch_state.json", state)
        state["status"] = "complete_independent_development_fresh_forbidden"
        state["fresh50_confirmation_authorized"] = state["stage_results"]["evaluate"][
            "fresh50_confirmation_authorized"
        ]
        state["automatic_fresh_launch"] = False
        atomic_json(root / "launch_state.json", state); return state
    except BaseException as error:
        state["status"] = "failed_closed_no_fresh_authorization"
        state["error_type"] = type(error).__name__; state["error"] = str(error)
        atomic_json(root / "launch_state.json", state); raise


def execute_recovery(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Resume after seed resolution without mutating the original failure record."""

    root = args.state_root.expanduser().absolute()
    unsigned = dict(plan)
    recorded = unsigned.pop("recovery_plan_sha256", "")
    if recorded != canonical_sha256(unsigned) or plan.get("format") != RECOVERY_FORMAT:
        raise RuntimeError("v7 recovery plan signature changed")
    # Close the preflight/execution TOCTOU window by recomputing the complete
    # recovery contract before creating any recovery artifact.  This repeats
    # the original-plan, command, source-file, failed-state, log-marker, seed,
    # and forbidden-output checks rather than only checking a subset of hashes.
    try:
        revalidated_plan = recovery_preflight(args)
    except BaseException as error:
        raise RuntimeError("v7 recovery provenance changed after preflight") from error
    if revalidated_plan.get("recovery_plan_sha256") != plan.get(
        "recovery_plan_sha256"
    ):
        raise RuntimeError("v7 recovery provenance changed after preflight")
    original_state = root / "launch_state.json"
    original_plan = root / "launch_plan.json"
    original_log = root / "logs" / "resolve_seeds.log"
    seeds_path = root / "v7_seed_manifest.json"
    failure = plan.get("original_failure")
    recovered = plan.get("recovered_seed_manifest")
    implementation = plan.get("recovery_implementation")
    if not all(isinstance(value, Mapping) for value in (failure, recovered, implementation)):
        raise RuntimeError("v7 recovery provenance is incomplete")
    assert isinstance(failure, Mapping)
    assert isinstance(recovered, Mapping)
    assert isinstance(implementation, Mapping)
    if (
        not original_state.is_file()
        or sha256(original_state) != failure.get("launch_state_file_sha256")
        or not original_plan.is_file()
        or sha256(original_plan) != failure.get("launch_plan_file_sha256")
        or not original_log.is_file()
        or sha256(original_log) != failure.get("resolve_log_sha256")
        or not seeds_path.is_file()
        or sha256(seeds_path) != recovered.get("file_sha256")
        or Path(str(implementation.get("path", ""))).resolve() != Path(__file__).resolve()
        or sha256(Path(__file__).resolve()) != implementation.get("sha256")
    ):
        raise RuntimeError("v7 recovery provenance changed after preflight")
    seed_contract = validate_seed_manifest(_json(seeds_path), verify_files=True)
    if seed_contract.get("seed_manifest_payload_sha256") != recovered.get(
        "payload_sha256"
    ):
        raise RuntimeError("v7 recovered seed manifest changed after preflight")
    original_state_sha = sha256(original_state)
    original_plan_sha = sha256(original_plan)
    original_log_sha = sha256(original_log)
    state = {
        **plan,
        "status": "recovery_running",
        "stage_results": {},
        "fresh_confirmation_labels_read": False,
    }
    recovery_plan_path = root / "v7_recovery_plan.json"
    recovery_state_path = root / "v7_recovery_state.json"
    if recovery_plan_path.exists() or recovery_state_path.exists():
        raise FileExistsError("v7 recovery plan/state already exists")
    atomic_json(recovery_plan_path, plan)
    atomic_json(recovery_state_path, state)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "8",
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "ETSF_V7_RECOVERY_PLAN_SHA256": str(plan["recovery_plan_sha256"]),
        }
    )
    try:
        for stage in plan["commands"]:
            name = str(stage["stage"])
            if name not in RECOVERY_STAGES:
                raise RuntimeError(f"v7 recovery refuses stage: {name}")
            state["current_stage"] = name
            if stage["uses_gpu"]:
                state.setdefault("gpu_idle_audits", {})[name] = require_exclusive_idle_gpu(
                    args.gpu_index
                )
            atomic_json(recovery_state_path, state)
            state["stage_results"][name] = run_stage(stage, args, env)
            state["last_completed_stage"] = name
            state["current_stage"] = None
            atomic_json(recovery_state_path, state)
        if (
            sha256(original_state) != original_state_sha
            or sha256(original_plan) != original_plan_sha
            or sha256(original_log) != original_log_sha
        ):
            raise RuntimeError("v7 recovery mutated original failure provenance")
        state["status"] = "complete_independent_development_fresh_forbidden"
        state["fresh50_confirmation_authorized"] = state["stage_results"]["evaluate"][
            "fresh50_confirmation_authorized"
        ]
        state["automatic_fresh_launch"] = False
        state["original_failure_provenance_preserved"] = True
        atomic_json(recovery_state_path, state)
        return state
    except BaseException as error:
        state["status"] = "failed_closed_no_fresh_authorization"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        state["automatic_fresh_launch"] = False
        atomic_json(recovery_state_path, state)
        raise


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent; parser = argparse.ArgumentParser()
    for name in ("candidates", "official150_manifest", "development150_manifest", "fresh50_manifest",
                 "official_seed_registry", "model_path", "rlinf_root", "robotwin_root", "robotwin_code",
                 "event_spec", "pretrained", "collection_output", "state_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--seed-resolver", type=Path, default=here / "preregister_robotwin_v7_development_confirmation.py")
    parser.add_argument("--collector", type=Path, default=here / "collect_openvla_etsf_event_branches.py")
    parser.add_argument("--evaluator", type=Path, default=here / "evaluate_openvla_etsf_v7_development_confirmation.py")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recover-resolved-seeds",
        action="store_true",
        help=(
            "Resume only after an independently valid seed manifest survived a "
            "resolve-stage teardown failure."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recover_resolved_seeds:
        plan = recovery_preflight(args)
        if args.dry_run:
            print("V7_RECOVERY_DRY_RUN=" + json.dumps(plan, sort_keys=True))
            return
        execute_recovery(args, plan)
        return
    plan = preflight(args)
    if args.dry_run:
        print("V7_DRY_RUN=" + json.dumps(plan, sort_keys=True))
        return
    execute(args, plan)


if __name__ == "__main__": main()
