from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_counterfactual_v5 as launcher  # noqa: E402
from launch_openvla_etsf_counterfactual_v5 import (  # noqa: E402
    DEFAULT_EVENT_SPEC,
    DEFAULT_OUTPUT,
    audit_factual_summaries,
    collector_ready,
    factual_members_ready,
    query_other_compute_pids,
    sha256,
    wait_for_gpu_idle,
)


def _event_spec(path: Path) -> str:
    path.write_text(
        json.dumps(
            {
                "chains": {"move_can_pot": {"chain": ["e0", "e12", "e3", "e4", "eK"]}},
                "calibration": {"move_can_pot": {"moving": "can"}},
            }
        ),
        encoding="utf-8",
    )
    return sha256(path)


def _collector(
    root: Path,
    event_digest: str,
    count: int = 100,
    *,
    seed_registry: str | None = "official_150",
) -> list[int]:
    groups = root / "groups"
    groups.mkdir(parents=True)
    seeds = list(range(1000, 1000 + count))
    rows = []
    for index, seed in enumerate(seeds):
        path = groups / f"group_{index:03d}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = 5
            handle.attrs["resolved_seed"] = seed
            handle.attrs["candidate_count"] = 4
            handle.attrs["language_contract"] = (
                "same_instruction_for_initial_query_and_all_candidate_branches"
            )
            handle.attrs["branch_instruction_consistent"] = True
        rows.append(
            {
                "path": path.name,
                "resolved_seed": seed,
                "status": "collected",
            }
        )
    manifest = {
        "status": "complete",
        "schema_version": 5,
        "completed": count,
        "candidate_count": 4,
        "language_contract": (
            "same_instruction_for_initial_query_and_all_candidate_branches"
        ),
        "intervention": "candidate_first_chunk_then_deterministic_actor",
        "event_spec_sha256": event_digest,
        "requested_seeds": seeds,
        "resolved_seeds": seeds,
        "groups": rows,
    }
    if seed_registry is not None:
        manifest["seed_registry"] = seed_registry
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return seeds


def _factual_members(
    root: Path,
    event_digest: str,
    train_seeds: list[int],
    scores: list[float],
    structured: bool = True,
) -> None:
    shared_contract = {
        "event_mode": "structured" if structured else "absolute",
        "event_spec_sha256": event_digest,
        "train_seeds": train_seeds,
        "validation_seeds": [2000, 2001],
        "sealed_test_seeds": [3000, 3001],
    }
    for seed, score in zip([20260827, 20260828, 20260829], scores):
        contract = {**shared_contract, "training_seed": seed}
        directory = root / f"seed_{seed}"
        directory.mkdir(parents=True)
        checkpoint = directory / "event_world_model_best.pt"
        torch.save(
            {
                "model": {},
                "config": {"structured_events": structured},
                "contract": contract,
                "best_score": score,
                "best_step": 10,
            },
            checkpoint,
        )
        summary = {
            "status": "training_complete",
            "sealed_test_evaluated": False,
            "best_validation_selection_score": score,
            "best_step": 10,
            "checkpoint": str(checkpoint.resolve()),
            "contract": contract,
        }
        (directory / "training_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )


def _command(
    tmp_path: Path,
    data: Path,
    factual: Path,
    event_spec: Path,
    output: Path,
    *,
    python_bin: Path | None = None,
) -> list[str]:
    python_bin = Path(sys.executable) if python_bin is None else python_bin
    return [
        sys.executable,
        str(SCRIPTS / "launch_openvla_etsf_counterfactual_v5.py"),
        "--data",
        str(data),
        "--factual-root",
        str(factual),
        "--event-spec",
        str(event_spec),
        "--output",
        str(output),
        "--trainer",
        str(SCRIPTS / "train_openvla_etsf_counterfactual.py"),
        "--python-bin",
        str(python_bin),
        "--wait-timeout-seconds",
        "0",
        "--poll-seconds",
        "0.01",
        "--dry-run",
    ]


def test_trainer_argv_preserves_venv_python_symlink(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    digest = _event_spec(event_spec)
    data = tmp_path / "v5_train100"
    seeds = _collector(data, digest)
    factual = tmp_path / "factual"
    _factual_members(factual, digest, seeds, [0.8, 0.3, 0.5])
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())

    completed = subprocess.run(
        _command(
            tmp_path,
            data,
            factual,
            event_spec,
            tmp_path / "output",
            python_bin=venv_python,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout.removeprefix("COUNTERFACTUAL_V5_DRY_RUN="))
    assert audit["command"][0] == str(venv_python.absolute())
    assert audit["command"][0] != str(venv_python.resolve())


def test_cpu_dry_run_selects_minimum_structured_checkpoint(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    digest = _event_spec(event_spec)
    data = tmp_path / "v5_train100"
    seeds = _collector(data, digest)
    factual = tmp_path / "factual"
    _factual_members(factual, digest, seeds, [0.8, 0.3, 0.5])
    output = tmp_path / "formal_output"
    completed = subprocess.run(
        _command(tmp_path, data, factual, event_spec, output),
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "COUNTERFACTUAL_V5_DRY_RUN="
    assert completed.stdout.startswith(prefix)
    audit = json.loads(completed.stdout[len(prefix) :])
    assert audit["selected_factual"]["best_validation_selection_score"] == 0.3
    assert "seed_20260828" in audit["selected_factual"]["checkpoint"]
    assert audit["data"]["groups"] == 100
    assert audit["runtime"] == {
        "device": "cuda",
        "amp": "bf16",
        "expected_gpu": "4090",
    }
    assert audit["counterfactual_member_seeds"] == [20260827, 20260828, 20260829]
    assert sorted(row["training_seed"] for row in audit["factual_candidates"]) == [
        20260827,
        20260828,
        20260829,
    ]
    assert "--device" in audit["command"] and "bf16" in audit["command"]
    guard_index = audit["command"].index("--guard-min-groups")
    assert audit["command"][guard_index + 1] == "10"
    coverage_index = audit["command"].index("--guard-min-coverage")
    assert audit["command"][coverage_index + 1] == "0.10"
    harmful_index = audit["command"].index("--guard-max-harmful-rate")
    assert audit["command"][harmful_index + 1] == "0.10"
    distance_index = audit["command"].index("--distance-weight")
    assert audit["command"][distance_index + 1] == "0.02"
    centered_index = audit["command"].index("--group-centered-weight")
    assert audit["command"][centered_index + 1] == "1.0"
    contrast_index = audit["command"].index("--baseline-contrast-weight")
    assert audit["command"][contrast_index + 1] == "1.5"
    assert not output.exists()


def test_legacy_v5_missing_registry_requires_exact_factual_train_crosscheck(
    tmp_path: Path,
) -> None:
    event_spec = tmp_path / "event_spec.json"
    digest = _event_spec(event_spec)
    data = tmp_path / "legacy_v5_train100"
    seeds = _collector(data, digest, seed_registry=None)
    factual = tmp_path / "factual"
    _factual_members(factual, digest, seeds, [0.8, 0.3, 0.5])
    completed = subprocess.run(
        _command(
            tmp_path,
            data,
            factual,
            event_spec,
            tmp_path / "output",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "COUNTERFACTUAL_V5_DRY_RUN="
    audit = json.loads(completed.stdout[len(prefix) :])
    assert audit["data"]["seed_registry"] is None
    assert audit["data"]["seed_registry_audit"]["status"] == (
        "legacy_schema_v5_provenance_accepted_after_exact_factual_crosscheck"
    )

    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fresh_seed_manifest"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="ambiguous"):
        launcher.audit_collector(data, digest)
    manifest.pop("fresh_seed_manifest")
    manifest["seed_registry"] = "explicit_fresh_confirmation"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="not accepted"):
        launcher.audit_collector(data, digest)


def test_rejects_partial_collector_and_absolute_factual(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    digest = _event_spec(event_spec)
    partial = tmp_path / "partial"
    _collector(partial, digest, count=99)
    with pytest.raises(RuntimeError, match="exactly 100"):
        collector_ready(partial)

    data = tmp_path / "complete"
    seeds = _collector(data, digest)
    factual = tmp_path / "absolute"
    ready, reason = factual_members_ready(factual)
    assert ready is False and "factual summaries=0/3" in reason
    _factual_members(factual, digest, seeds, [0.1, 0.2, 0.3], structured=False)
    assert factual_members_ready(factual) == (True, "complete")
    with pytest.raises(RuntimeError, match="absolute-event"):
        audit_factual_summaries(factual, digest)


def test_factual_readiness_waits_for_frozen_members_and_rejects_extra_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "factual"
    expected = root / "seed_20260827"
    expected.mkdir(parents=True)
    (expected / "training_summary.json").write_text(
        json.dumps({"status": "training_complete"}), encoding="utf-8"
    )
    ready, reason = factual_members_ready(root)
    assert ready is False
    assert "1/3" in reason and "seed_20260828" in reason

    extra = root / "seed_7"
    extra.mkdir()
    (extra / "training_summary.json").write_text(
        json.dumps({"status": "training_complete"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unexpected factual seed"):
        factual_members_ready(root)


def test_gpu_idle_wait_excludes_launcher_pid_and_times_out_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="123\n456\n", stderr=""
    )
    monkeypatch.setattr(launcher.os, "getpid", lambda: 123)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: completed)
    assert query_other_compute_pids() == [456]

    states = iter([[456], []])
    monkeypatch.setattr(launcher, "query_other_compute_pids", lambda index: next(states))
    audit = wait_for_gpu_idle(1.0, 0.01)
    assert audit["checks"] == 2
    assert audit["status"] == "idle_before_counterfactual_launch"

    monkeypatch.setattr(launcher, "query_other_compute_pids", lambda index: [456])
    with pytest.raises(RuntimeError, match="remained occupied"):
        wait_for_gpu_idle(0.0, 0.01)


def test_default_event_spec_points_to_real_stage2_run() -> None:
    assert DEFAULT_EVENT_SPEC == Path(
        "/home/user/etsf_stage2_run_20260825/event_spec.json"
    )
    assert DEFAULT_OUTPUT == Path(
        "/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry1_20260827"
    )


def test_rejects_factual_training_seed_directory_mismatch(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    digest = _event_spec(event_spec)
    factual = tmp_path / "factual"
    _factual_members(factual, digest, list(range(100)), [0.1, 0.2, 0.3])
    summary_path = factual / "seed_20260828" / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["contract"]["training_seed"] = 7
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="training_seed does not match directory"):
        audit_factual_summaries(factual, digest)


def test_existing_conflicting_output_is_refused_without_resume(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    digest = _event_spec(event_spec)
    data = tmp_path / "v5_train100"
    seeds = _collector(data, digest)
    factual = tmp_path / "factual"
    _factual_members(factual, digest, seeds, [0.8, 0.3, 0.5])
    output = tmp_path / "formal_output"
    output.mkdir()
    (output / "partial.log").write_text("interrupted", encoding="utf-8")
    completed = subprocess.run(
        _command(tmp_path, data, factual, event_spec, output),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "no resume" in completed.stderr
    assert not (output / "launch_audit.json").exists()


def test_completed_output_requires_frozen_scoring_and_formal_guard_doors(
    tmp_path: Path,
) -> None:
    output = tmp_path / "complete"
    output.mkdir()
    factual_sha = "a" * 64
    data_sha = "b" * 64
    candidate_ids = list(launcher.SCORING_CANDIDATE_IDS)
    weights = launcher.SCORING_WEIGHTS
    candidates = [
        {
            "candidate_id": candidate_id,
            "event_weight": weights[candidate_id][0],
            "duration_weight": weights[candidate_id][1],
            "candidate_distance_weight": weights[candidate_id][2],
        }
        for candidate_id in candidate_ids
    ]
    selection = {
        "grid_version": launcher.SCORING_GRID_VERSION,
        "selection_rule": launcher.SCORING_SELECTION_RULE,
        "minimum_proposals": 10,
        "minimum_coverage": 0.10,
        "minimum_lcb90": 0.0,
        "selected_candidate_id": "full",
        "candidates": candidates,
    }
    scoring = {
        **candidates[-1],
        "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
    }
    guard = {
        "enabled": False,
        "grid_version": "validation_guard_quantile_grid_v1",
        "minimum_guarded_groups": 10,
        "minimum_coverage": 0.10,
        "minimum_lcb": 0.0,
        "maximum_harmful_rate": 0.10,
        "threshold_candidates": [],
    }
    contract = {
        "pretrained_sha256": factual_sha,
        "schema_counts": {"5": 100},
        "scoring_selection_contract": {
            "grid_version": launcher.SCORING_GRID_VERSION,
            "selection_data": "validation_only_no_sealed_test",
            "grid_candidate_ids": candidate_ids,
            "selection_rule": launcher.SCORING_SELECTION_RULE,
            "guard_grid_version": launcher.GUARD_GRID_VERSION,
        },
        "counterfactual_ranking_contract": {
            "member_selection_data": "validation_only_no_sealed_test",
            "member_selection_rule": launcher.MEMBER_SELECTION_RULE,
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
                "candidate_cardinality": {
                    "variable_candidate_count_supported": True,
                    "minimum_candidates_per_group": 2,
                    "unique_baseline_name": "deterministic",
                    "baseline_index": 0,
                    "pairwise_reduction": "mean_pairs_then_mean_groups",
                    "listwise_reduction": (
                        "cross_entropy_div_log_C_then_mean_groups"
                    ),
                    "train_count_histogram": {"4": 70, "5": 105},
                    "validation_count_histogram": {"4": 15, "5": 22},
                },
                "action_sensitivity": {
                    "enabled": True,
                    "architecture": "baseline_relative_action_effect_residual_v1",
                    "inputs": [
                        "action_effect_minus_deterministic_action_effect",
                        "shared_semantic_times_action_effect_delta",
                    ],
                    "baseline_residual": 0.0,
                    "absolute_success_supervision": "base_world_success_logit_only",
                    "ranking_gradient": (
                        "residual_branch_with_base_score_stop_gradient"
                    ),
                    "deployment": (
                        "predict_candidates_adds_residual_to_success_logit"
                    ),
                    "event_time_object_heads": "unchanged",
                },
            "loss_weights": {
                "pairwise": 0.75,
                "listwise": 0.5,
                "group_centered": 1.0,
                "baseline_contrast": 1.5,
            },
        },
    }
    aggregate_path = output / "counterfactual_ensemble.pt"
    torch.save(
        {"scoring": scoring, "scoring_selection": selection, "guard": guard},
        aggregate_path,
    )
    members = []
    for seed in launcher.SEEDS:
        member_path = output / f"counterfactual_seed_{seed}.pt"
        torch.save(
            {
                "seed": seed,
                "best_selection_rule": launcher.MEMBER_SELECTION_RULE,
                "best_selection_key": [0.1, 0.0, -1.0, -2.0],
            },
            member_path,
        )
        members.append(
            {"path": str(member_path), "sha256": sha256(member_path), "seed": seed}
        )
    manifest = {
        "format": "etsf_counterfactual_ensemble_v1",
        "ensemble_checkpoint": {
            "path": str(aggregate_path),
            "sha256": sha256(aggregate_path),
        },
        "members": members,
        "config": {"structured_events": True},
        "contract": contract,
        "test_policy": (
            "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
        ),
        "scoring": scoring,
        "scoring_selection": selection,
        "guard": guard,
    }
    manifest_path = output / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output / "launch_audit.json").write_text(
        json.dumps(
            {
                "status": "launcher_complete",
                "selected_factual": {"checkpoint_sha256": factual_sha},
                "data": {"manifest_sha256": data_sha},
            }
        ),
        encoding="utf-8",
    )
    assert launcher.validate_complete_output(output, factual_sha, data_sha)[
        "status"
    ] == "already_complete_skip"

    manifest["guard"]["minimum_guarded_groups"] = 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ten proposals"):
        launcher.validate_complete_output(output, factual_sha, data_sha)
