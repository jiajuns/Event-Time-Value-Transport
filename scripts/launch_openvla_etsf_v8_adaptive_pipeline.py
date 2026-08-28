#!/usr/bin/env python3
"""Wait for signed v7 completion, then run the adaptive v8 OOF pipeline.

This launcher is deliberately independent from the v7 process.  It never
opens Fresh50, never authorizes a selector, and never mutates the v7 code or
artifacts.  It runs owner preregistration, leakage-audited materialization,
factual event diagnostics, two independently trained probability-head
optimizers, and their five-fold evaluation bridges on a remote worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_openvla_v8_adaptive_remote_pipeline_v1"
TERMINAL_STATUS = "complete_adaptive_development_only_no_fresh"
V7_TERMINAL_STATUS = "complete_independent_development_fresh_forbidden"
V7_FORMATS = {
    "etsf_openvla_v7_prospective_server_launch_v1": "original",
    "etsf_openvla_v7_resolved_seed_recovery_v1": "recovery",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _script(code_root: Path, name: str) -> Path:
    path = (code_root / "scripts" / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _reject_fresh_path(path: Path, *, role: str) -> Path:
    """Reject Fresh/confirmation tokens anywhere in an output path."""

    resolved = path.resolve()
    if any(
        token in part.lower()
        for part in resolved.parts
        for token in ("fresh", "confirmation")
    ):
        raise RuntimeError(f"{role} cannot be named as Fresh/confirmation")
    return resolved


def _validate_signed_payload(value: Mapping[str, Any], key: str, *, name: str) -> None:
    unsigned = dict(value)
    digest = unsigned.pop(key, None)
    if not isinstance(digest, str) or digest != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} signature mismatch")


def _gpu_compute_pids(gpu_index: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_index),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            result.append(int(line))
    return sorted(set(result))


def _wait_for_gpu_idle(
    *,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    gpu_index: int,
) -> None:
    while True:
        pids = _gpu_compute_pids(gpu_index)
        state["gpu_wait"] = {"compute_pids": pids, "gpu_index": gpu_index}
        state["last_heartbeat_unix"] = time.time()
        atomic_json(state_path, state)
        if not pids:
            return
        time.sleep(poll_seconds)


def _run_stage(
    *,
    stage: str,
    argv: Sequence[str],
    state: dict[str, Any],
    state_path: Path,
    logs_dir: Path,
    code_root: Path,
    child_env: Mapping[str, str],
) -> None:
    log = logs_dir / f"{stage}.log"
    state["current_stage"] = stage
    state["last_heartbeat_unix"] = time.time()
    state.setdefault("commands", []).append(
        {
            "stage": stage,
            "argv": list(argv),
            "argv_sha256": canonical_sha256(list(argv)),
            "log": str(log),
        }
    )
    atomic_json(state_path, state)
    _validate_implementation_files(state, stage=stage)
    with log.open("xb") as handle:
        completed = subprocess.run(
            list(argv),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=code_root,
            env=dict(child_env),
        )
    if completed.returncode != 0:
        raise RuntimeError(f"v8 stage {stage} failed; see {log}")
    state["last_completed_stage"] = stage
    state["current_stage"] = None
    state["stage_logs"] = {
        **state.get("stage_logs", {}),
        stage: {"path": str(log), "sha256": sha256_path(log)},
    }
    atomic_json(state_path, state)


def _validate_implementation_files(
    state: Mapping[str, Any], *, stage: str
) -> None:
    implementations = state.get("implementation_files")
    if not isinstance(implementations, Mapping) or not implementations:
        raise RuntimeError("implementation hash contract is missing")
    for name, record in implementations.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"implementation record is invalid: {name}")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file() or sha256_path(path) != record.get("sha256"):
            raise RuntimeError(
                f"implementation changed before stage {stage}: {name}"
            )


def _validate_v7_completion(
    *, state_path: Path, result_path: Path, data_root: Path
) -> dict[str, Any]:
    state = load_json(state_path)
    execution_kind = V7_FORMATS.get(str(state.get("format", "")))
    if (
        execution_kind is None
        or state.get("terminal_status") != V7_TERMINAL_STATUS
        or state.get("status") != V7_TERMINAL_STATUS
        or state.get("current_stage") is not None
        or state.get("last_completed_stage") != "evaluate"
        or state.get("automatic_fresh_launch") is not False
        or state.get("fresh_confirmation_labels_read") is not False
    ):
        raise RuntimeError("v7 state is not a completed no-Fresh run")
    result = load_json(result_path)
    _validate_signed_payload(result, "result_sha256", name="v7 result")
    evaluate_record = state.get("stage_results", {}).get("evaluate", {})
    collect_record = state.get("stage_results", {}).get("collect", {})
    preregister_record = state.get("stage_results", {}).get("preregister", {})
    result_authorized = result.get("authorization", {}).get(
        "fresh50_confirmation_authorized"
    )
    development_gate_pass = result.get("metrics", {}).get("development_gate_pass")
    if (
        Path(str(evaluate_record.get("artifact", ""))).resolve()
        != result_path.resolve()
        or evaluate_record.get("sha256") != sha256_path(result_path)
        or evaluate_record.get("status") != "complete"
        or result_authorized not in (True, False)
        or result_authorized is not (development_gate_pass is True)
        or state.get("fresh50_confirmation_authorized") is not result_authorized
        or evaluate_record.get("fresh50_confirmation_authorized")
        is not result_authorized
        or evaluate_record.get("development_gate_pass")
        is not (development_gate_pass is True)
        or preregister_record.get("preregistration_sha256")
        != result.get("preregistration_sha256")
    ):
        raise RuntimeError("v7 state does not bind the supplied result")
    if (
        result.get("format")
        != "etsf_openvla_v7_prospective_development_result_v1"
        or result.get("status") != "complete_development_only"
        or result.get("fresh_confirmation_labels_read") is not False
        or result.get("authorization", {}).get("automatic_fresh_launch") is not False
    ):
        raise RuntimeError("v7 result changed development/Fresh semantics")
    manifest_path = Path(str(result.get("collection_manifest", ""))).resolve()
    if manifest_path != (data_root / "manifest.json").resolve():
        raise RuntimeError("v7 result points to a different collection")
    if sha256_path(manifest_path) != result.get("collection_manifest_sha256"):
        raise RuntimeError("v7 result collection SHA mismatch")
    if (
        Path(str(collect_record.get("artifact", ""))).resolve() != manifest_path
        or collect_record.get("sha256") != result.get("collection_manifest_sha256")
        or collect_record.get("status") != "complete"
        or int(collect_record.get("groups", -1)) != 250
    ):
        raise RuntimeError("v7 state does not bind the collection")
    preregistration_path = Path(str(preregister_record.get("artifact", ""))).resolve()
    if (
        not preregistration_path.is_file()
        or preregister_record.get("sha256") != sha256_path(preregistration_path)
        or preregister_record.get("status") != "complete"
    ):
        raise RuntimeError("v7 state does not bind the preregistration artifact")
    preregistration = load_json(preregistration_path)
    _validate_signed_payload(
        preregistration, "preregistration_sha256", name="v7 preregistration"
    )
    if preregistration["preregistration_sha256"] != result.get(
        "preregistration_sha256"
    ) or (
        preregistration.get("format")
        != "etsf_openvla_v7_prospective_development_confirmation_v1"
        or preregistration.get("status") != "preregistered_before_labels"
    ):
        raise RuntimeError("v7 preregistration and result disagree")
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or int(manifest.get("completed", -1)) != 250
        or int(manifest.get("schema_version", -1)) != 5
        or int(manifest.get("candidate_count", -1)) != 4
        or manifest.get("seed_registry")
        != "explicit_v7_prospective_development"
        or manifest.get("fresh_seed_manifest") not in (None, "")
        or manifest.get("fresh_seed_manifest_sha256") not in (None, "")
    ):
        raise RuntimeError("v7 collection is not a complete independent schema5 source")
    return {
        "v7_execution_kind": execution_kind,
        "v7_state_file_sha256": sha256_path(state_path),
        "v7_result_sha256": sha256_path(result_path),
        "v7_result_payload_sha256": result["result_sha256"],
        "collection_manifest_sha256": result["collection_manifest_sha256"],
        "v7_development_gate_pass": bool(
            result.get("metrics", {}).get("development_gate_pass", False)
        ),
        "fresh50_read_or_launched": False,
    }


def _validate_factual_output(
    *, result_path: Path, materialization_manifest: Path
) -> dict[str, Any]:
    result = load_json(result_path)
    _validate_signed_payload(result, "result_sha256", name="factual result")
    authorization = result.get("authorization")
    source = result.get("source_materialization")
    if (
        result.get("format") != "etsf_v8_factual_event_oof_diagnostics_v1"
        or result.get("status") != "complete_adaptive_development_only"
        or result.get("evidence_scope")
        != "D250_adaptive_development_only_not_prospective"
        or result.get("fresh_confirmation_data_or_labels_read") is not False
        or not isinstance(authorization, Mapping)
        or any(
            authorization.get(key) is not False
            for key in (
                "fresh50_confirmation_authorized",
                "selector_authorized",
                "deployment_authorized",
                "policy_success_claim_authorized",
            )
        )
        or not isinstance(source, Mapping)
    ):
        raise RuntimeError("factual result changed adaptive/no-Fresh semantics")
    manifest = load_json(materialization_manifest)
    _validate_signed_payload(
        manifest, "materialization_sha256", name="v8 materialization"
    )
    assert isinstance(source, Mapping)
    if (
        Path(str(source.get("path", ""))).resolve()
        != materialization_manifest.resolve()
        or source.get("file_sha256") != sha256_path(materialization_manifest)
        or source.get("materialization_sha256")
        != manifest.get("materialization_sha256")
    ):
        raise RuntimeError("factual result is not bound to this materialization")
    return {
        "path": str(result_path.resolve()),
        "file_sha256": sha256_path(result_path),
        "result_sha256": result["result_sha256"],
        "status": result["status"],
        "fresh50_read_or_authorized": False,
    }


def _validate_bridge_output(
    *, output_dir: Path, materialization_manifest: Path
) -> dict[str, Any]:
    result_path = output_dir / "structured_heads_evaluation.json"
    contracts_path = output_dir / "structured_heads_contracts.json"
    result = load_json(result_path)
    contracts = load_json(contracts_path)
    _validate_signed_payload(result, "result_sha256", name="bridge result")
    _validate_signed_payload(contracts, "contracts_sha256", name="bridge contracts")
    adaptive_contract = contracts.get("adaptive_contract")
    bridge_bundle = contracts.get("bridge_bundle")
    provenance = contracts.get("bridge_provenance")
    input_contract = contracts.get("input_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (adaptive_contract, bridge_bundle, provenance, input_contract)
    ):
        raise RuntimeError("bridge output provenance is incomplete")
    assert isinstance(adaptive_contract, Mapping)
    assert isinstance(bridge_bundle, Mapping)
    assert isinstance(provenance, Mapping)
    assert isinstance(input_contract, Mapping)
    _validate_signed_payload(
        adaptive_contract, "contract_sha256", name="bridge adaptive contract"
    )
    _validate_signed_payload(
        bridge_bundle, "bridge_bundle_sha256", name="bridge bundle"
    )
    arrays_path = Path(str(contracts.get("arrays", ""))).resolve()
    materialization = load_json(materialization_manifest)
    _validate_signed_payload(
        materialization, "materialization_sha256", name="v8 materialization"
    )
    if (
        contracts.get("format")
        != "etsf_v8_authenticated_oof_evaluation_output_v1"
        or result.get("format")
        != "etsf_v8_structured_heads_array_evaluation_v1"
        or result.get("status")
        not in (
            "all_structured_domains_passed",
            "fail_closed_one_or_more_domains",
        )
        or result.get("development_only") is not True
        or result.get("prospective_claim_allowed") is not False
        or result.get("fresh50_inputs_accepted") is not False
        or result.get("fresh50_labels_read") is not False
        or result.get("fresh50_confirmation_authorized") is not False
        or result.get("action_selector_authorized") is not False
        or (
            result.get("status") == "all_structured_domains_passed"
        ) is not (result.get("all_domain_pass") is True)
        or input_contract.get("fresh50_inputs_accepted") is not False
        or input_contract.get("fresh50_labels_read") is not False
        or provenance.get("fresh50_inputs_accepted") is not False
        or provenance.get("fresh50_labels_read") is not False
        or provenance.get("prospective_claim_allowed") is not False
        or result.get("adaptive_development_contract_sha256")
        != adaptive_contract.get("contract_sha256")
        or provenance.get("materialization_sha256")
        != materialization.get("materialization_sha256")
        or not arrays_path.is_file()
        or contracts.get("arrays_sha256") != sha256_path(arrays_path)
    ):
        raise RuntimeError("bridge output changed signed adaptive/no-Fresh contract")
    return {
        "path": str(output_dir.resolve()),
        "result_file_sha256": sha256_path(result_path),
        "result_sha256": result["result_sha256"],
        "contracts_file_sha256": sha256_path(contracts_path),
        "contracts_sha256": contracts["contracts_sha256"],
        "status": result["status"],
        "all_domain_pass": result.get("all_domain_pass") is True,
        "fresh50_read_or_authorized": False,
    }


def _publish_terminal_state(
    *,
    summary_path: Path,
    state_path: Path,
    summary: Mapping[str, Any],
    state: dict[str, Any],
) -> None:
    """Publish two terminal files without leaving a complete summary on failure."""

    atomic_json(summary_path, summary)
    terminal_state = {**state, **summary, "current_stage": None}
    try:
        atomic_json(state_path, terminal_state)
    except BaseException:
        # This new launcher-owned summary is authoritative only after the state
        # update succeeds, so retract it if terminal publication fails.
        summary_path.unlink(missing_ok=True)
        raise
    state.clear()
    state.update(terminal_state)


def _base_identity_sha(materialization_manifest: Path) -> str:
    manifest = load_json(materialization_manifest)
    _validate_signed_payload(
        manifest, "materialization_sha256", name="v8 materialization"
    )
    values = {
        str(row.get("base_exclusion_audit", {}).get("base_identity_contract_sha256", ""))
        for row in manifest.get("folds", [])
        if isinstance(row, Mapping)
    }
    if len(values) != 1:
        raise RuntimeError("five folds do not share one proven base identity contract")
    value = next(iter(values))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("base identity contract SHA is invalid")
    if any(
        row.get("base_exclusion_audit", {}).get("status") != "proven"
        for row in manifest["folds"]
    ):
        raise RuntimeError("v8 factual base exclusion is not proven for all folds")
    return value


def execute(args: argparse.Namespace) -> None:
    code_root = args.code_root.resolve()
    output_root = _reject_fresh_path(args.output_root, role="v8 adaptive output")
    # A venv's ``bin/python`` is commonly a symlink to the system interpreter.
    # Resolving that symlink silently drops the venv and therefore its packages.
    python_bin = Path(os.path.abspath(os.fspath(args.python_bin)))
    if output_root.exists():
        raise FileExistsError(output_root)
    for path in (
        args.v7_state,
        args.checkpoint,
        args.event_spec,
        args.python_bin,
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise RuntimeError(f"python interpreter is not executable: {python_bin}")
    scripts = {
        "materializer": _script(code_root, "materialize_openvla_etsf_v8_oof_inputs.py"),
        "trainer": _script(code_root, "train_openvla_etsf_v8_structured_adapters.py"),
        "bridge": _script(code_root, "evaluate_openvla_etsf_v8_oof_bridge.py"),
        "factual_events": _script(code_root, "evaluate_openvla_etsf_v8_factual_events.py"),
    }
    output_root.mkdir(parents=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()
    state_path = output_root / "pipeline_state.json"
    plan = {
        "format": FORMAT,
        "status": "waiting_for_v7",
        "code_root": str(code_root),
        "implementation_files": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in scripts.items()
        },
        "v7_state": str(args.v7_state.resolve()),
        "v7_result": str(args.v7_result.resolve()),
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_path(args.checkpoint.resolve()),
        "event_spec": str(args.event_spec.resolve()),
        "event_spec_sha256": sha256_path(args.event_spec.resolve()),
        "python_bin": str(python_bin),
        "gpu_index": int(args.gpu_index),
        "optimizer_candidates": {
            "lbfgs_convex": {"max_iter": 100},
            "adamw_fixed": {"epochs": 10, "learning_rate": 0.001},
        },
        "adaptive_development_only": True,
        "prospective_claim_allowed": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
        "automatic_fresh_launch": False,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    atomic_json(output_root / "pipeline_plan.json", plan)
    state: dict[str, Any] = {
        **plan,
        "status": "waiting_for_v7",
        "current_stage": None,
        "last_completed_stage": None,
        "commands": [],
    }
    atomic_json(state_path, state)
    try:
        while True:
            v7_state = load_json(args.v7_state.resolve())
            status = v7_state.get("status")
            if status == V7_TERMINAL_STATUS:
                break
            if status not in (
                "running",
                "recovery_running",
                "recovery_preflight_complete",
            ):
                raise RuntimeError(f"v7 entered nonrecoverable status: {status}")
            state["v7_observed_status"] = status
            state["v7_observed_stage"] = v7_state.get("current_stage")
            state["last_heartbeat_unix"] = time.time()
            atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
        for path in (args.v7_result, args.data):
            if not path.resolve().exists():
                raise FileNotFoundError(path.resolve())
        state["v7_completion_audit"] = _validate_v7_completion(
            state_path=args.v7_state.resolve(),
            result_path=args.v7_result.resolve(),
            data_root=args.data.resolve(),
        )
        state["status"] = "waiting_for_gpu_idle"
        atomic_json(state_path, state)
        _wait_for_gpu_idle(
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            gpu_index=args.gpu_index,
        )
        state["status"] = "running_adaptive_development"
        atomic_json(state_path, state)

        python = str(python_bin)
        child_env = dict(os.environ)
        child_env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(code_root / "scripts"),
                "OMP_NUM_THREADS": "4",
            }
        )
        owner = output_root / "v8_owner_manifest.json"
        materialized = output_root / "materialized_oof"
        materialization_manifest = materialized / "materialization_manifest.json"
        _run_stage(
            stage="preregister_owner",
            argv=[
                python,
                str(scripts["materializer"]),
                "--preregister-owner-manifest",
                "--data",
                str(args.data.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--event-spec",
                str(args.event_spec.resolve()),
                "--output",
                str(owner),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            code_root=code_root,
            child_env=child_env,
        )
        _run_stage(
            stage="materialize_oof",
            argv=[
                python,
                str(scripts["materializer"]),
                "--data",
                str(args.data.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--event-spec",
                str(args.event_spec.resolve()),
                "--oof-manifest",
                str(owner),
                "--output",
                str(materialized),
                "--device",
                "cuda",
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            code_root=code_root,
            child_env=child_env,
        )
        base_identity_sha = _base_identity_sha(materialization_manifest)
        state["base_identity_contract_sha256"] = base_identity_sha
        _run_stage(
            stage="evaluate_factual_events",
            argv=[
                python,
                str(scripts["factual_events"]),
                "--materialization-manifest",
                str(materialization_manifest),
                "--output",
                str(output_root / "factual_event_evaluation.json"),
            ],
            state=state,
            state_path=state_path,
            logs_dir=logs_dir,
            code_root=code_root,
            child_env=child_env,
        )
        factual_audit = _validate_factual_output(
            result_path=output_root / "factual_event_evaluation.json",
            materialization_manifest=materialization_manifest,
        )
        state["factual_event_evaluation_audit"] = factual_audit
        atomic_json(state_path, state)

        mode_outputs: dict[str, Any] = {}
        for mode in ("lbfgs_convex", "adamw_fixed"):
            checkpoints: list[Path] = []
            for fold in range(5):
                checkpoint_output = output_root / mode / f"fold_{fold}.pt"
                argv = [
                    python,
                    str(scripts["trainer"]),
                    "--input",
                    str(materialized / f"fold_{fold}_train.pt"),
                    "--materialization-manifest",
                    str(materialization_manifest),
                    "--outer-fold-id",
                    str(fold),
                    "--output",
                    str(checkpoint_output),
                    "--device",
                    "cuda",
                ]
                if mode == "lbfgs_convex":
                    argv.extend(["--optimizer-mode", "lbfgs", "--lbfgs-max-iter", "100"])
                else:
                    argv.extend(
                        [
                            "--optimizer-mode",
                            "adamw",
                            "--epochs",
                            "10",
                            "--learning-rate",
                            "0.001",
                        ]
                    )
                _run_stage(
                    stage=f"train_{mode}_fold_{fold}",
                    argv=argv,
                    state=state,
                    state_path=state_path,
                    logs_dir=logs_dir,
                    code_root=code_root,
                    child_env=child_env,
                )
                checkpoints.append(checkpoint_output)
            bridge_output = output_root / f"evaluation_{mode}"
            bridge_argv = [python, str(scripts["bridge"])]
            for checkpoint_path in checkpoints:
                bridge_argv.extend(["--checkpoint", str(checkpoint_path)])
            for fold in range(5):
                bridge_argv.extend(
                    ["--holdout", str(materialized / f"fold_{fold}_holdout.pt")]
                )
            bridge_argv.extend(
                [
                    "--materialization-manifest",
                    str(materialization_manifest),
                    "--base-identity-contract-sha256",
                    base_identity_sha,
                    "--output",
                    str(bridge_output),
                ]
            )
            _run_stage(
                stage=f"evaluate_{mode}",
                argv=bridge_argv,
                state=state,
                state_path=state_path,
                logs_dir=logs_dir,
                code_root=code_root,
                child_env=child_env,
            )
            bridge_audit = _validate_bridge_output(
                output_dir=bridge_output,
                materialization_manifest=materialization_manifest,
            )
            mode_outputs[mode] = {
                "checkpoints": [
                    {"path": str(path), "sha256": sha256_path(path)}
                    for path in checkpoints
                ],
                "evaluation": str(bridge_output),
                "evaluation_audit": bridge_audit,
            }
        summary = {
            "format": FORMAT,
            "status": TERMINAL_STATUS,
            "v7_completion_audit": state["v7_completion_audit"],
            "owner_manifest": str(owner),
            "owner_manifest_sha256": sha256_path(owner),
            "materialization_manifest": str(materialization_manifest),
            "materialization_manifest_sha256": sha256_path(
                materialization_manifest
            ),
            "factual_event_evaluation": str(
                output_root / "factual_event_evaluation.json"
            ),
            "factual_event_evaluation_sha256": sha256_path(
                output_root / "factual_event_evaluation.json"
            ),
            "factual_event_evaluation_audit": factual_audit,
            "optimizer_candidates": mode_outputs,
            "automatic_optimizer_selection": False,
            "selection_reason": (
                "retain_both_development_oof_results_for_evidence_based_audit"
            ),
            "prospective_claim_allowed": False,
            "selector_authorized": False,
            "fresh50_inputs_accepted": False,
            "fresh50_labels_read": False,
            "automatic_fresh_launch": False,
        }
        summary["summary_sha256"] = canonical_sha256(summary)
        state["last_completed_stage"] = "evaluate_adamw_fixed"
        _publish_terminal_state(
            summary_path=output_root / "pipeline_summary.json",
            state_path=state_path,
            summary=summary,
            state=state,
        )
    except BaseException as error:
        state["status"] = "failed_closed_no_fresh"
        state["error_type"] = type(error).__name__
        state["error"] = str(error)
        state["fresh50_inputs_accepted"] = False
        state["fresh50_labels_read"] = False
        state["automatic_fresh_launch"] = False
        atomic_json(state_path, state)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-state", type=Path, required=True)
    parser.add_argument("--v7-result", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 5.0 <= args.poll_seconds <= 60.0:
        raise ValueError("poll-seconds must be in [5,60]")
    if args.gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    execute(args)


if __name__ == "__main__":
    main()
