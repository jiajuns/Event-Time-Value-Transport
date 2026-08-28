from __future__ import annotations

import hashlib
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

import launch_openvla_etsf_progress_v5 as progress_launcher  # noqa: E402
from launch_openvla_etsf_progress_v5 import (  # noqa: E402
    counterfactual_ready,
    sha256,
)
from launch_openvla_etsf_counterfactual_v5 import (  # noqa: E402
    GUARD_GRID_VERSION,
    SCORING_CANDIDATE_IDS,
    SCORING_GRID_VERSION,
    SCORING_SELECTION_RULE,
    SCORING_WEIGHTS,
)
from train_openvla_etsf_counterfactual import (  # noqa: E402
    make_group_splits,
    scan_group_descriptors,
)


LANGUAGE_CONTRACT = "same_instruction_for_initial_query_and_all_candidate_branches"
INTERVENTION = "candidate_first_chunk_then_deterministic_actor"
SEEDS = [20260827, 20260828, 20260829]


def _event_spec(path: Path) -> str:
    path.write_text(
        json.dumps(
            {
                "calibration": {
                    "move_can_pot": {
                        "moving": "can",
                        "anchor": "pot",
                        "offset": [0.0, 0.0, 0.0],
                        "delta_move": 0.05,
                        "delta_z": 0.05,
                        "tau_d": 0.06,
                        "tau_motion": 0.02,
                        "stationary_steps": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return sha256(path)


def _collector(root: Path, event_digest: str) -> None:
    groups = root / "groups"
    groups.mkdir(parents=True)
    rows = []
    seeds = list(range(5000, 5100))
    for index, seed in enumerate(seeds):
        path = groups / f"group_{index:03d}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema_version"] = 5
            handle.attrs["task"] = "move_can_pot"
            handle.attrs["body"] = "piper"
            handle.attrs["policy"] = "openvla"
            handle.attrs["resolved_seed"] = seed
            handle.attrs["candidate_count"] = 4
            handle.attrs["language_contract"] = LANGUAGE_CONTRACT
            handle.attrs["branch_instruction_consistent"] = True
        rows.append({"path": path.name, "resolved_seed": seed, "status": "collected"})
    manifest = {
        "status": "complete",
        "schema_version": 5,
        "completed": 100,
        "candidate_count": 4,
        "seed_registry": "official_150",
        "language_contract": LANGUAGE_CONTRACT,
        "intervention": INTERVENTION,
        "event_spec_sha256": event_digest,
        "requested_seeds": seeds,
        "resolved_seeds": seeds,
        "groups": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _factual(root: Path, event_digest: str) -> str:
    selected_sha = ""
    for index, seed in enumerate(SEEDS):
        directory = root / f"seed_{seed}"
        directory.mkdir(parents=True)
        contract = {
            "event_mode": "structured",
            "event_spec_sha256": event_digest,
            "training_seed": seed,
            "train_seeds": list(range(5000, 5100)),
            "validation_seeds": [6000],
            "sealed_test_seeds": [7000],
        }
        score = [0.3, 0.1, 0.2][index]
        checkpoint = directory / "event_world_model_best.pt"
        torch.save(
            {
                "model": {},
                "config": {"structured_events": True},
                "contract": contract,
                "best_score": score,
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
        if seed == 20260828:
            selected_sha = sha256(checkpoint)
    return selected_sha


def _counterfactual(
    root: Path,
    data: Path,
    selected_factual_sha: str,
    event_digest: str,
) -> None:
    root.mkdir(parents=True)
    descriptors = scan_group_descriptors([data])
    splits = make_group_splits(descriptors)
    split = {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
        "test_policy": "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened",
    }
    (root / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    members = []
    for seed in SEEDS:
        path = root / f"member_{seed}.pt"
        path.write_bytes(f"member {seed}".encode())
        members.append({"path": str(path.resolve()), "sha256": sha256(path), "seed": seed})
    contract = {
        "pretrained_sha256": selected_factual_sha,
        "event_spec_sha256": event_digest,
        "schema_counts": {"5": 100},
        "train_groups": splits["train"],
        "validation_groups": splits["validation"],
        "sealed_test_groups": splits["test"],
        "group_files": [
            {
                "logical_key": descriptor.logical_key,
                "schema_version": 5,
                "path": descriptor.path,
                "sha256": sha256(Path(descriptor.path)),
            }
            for descriptor in descriptors
        ],
        "scoring_selection_contract": {
            "selection_data": "validation_only_no_sealed_test",
            "grid_version": SCORING_GRID_VERSION,
            "selection_rule": SCORING_SELECTION_RULE,
            "guard_grid_version": GUARD_GRID_VERSION,
            "grid_candidate_ids": list(SCORING_CANDIDATE_IDS),
        },
    }
    scoring_candidates = [
        {
            "candidate_id": candidate_id,
            "event_weight": SCORING_WEIGHTS[candidate_id][0],
            "duration_weight": SCORING_WEIGHTS[candidate_id][1],
            "candidate_distance_weight": SCORING_WEIGHTS[candidate_id][2],
        }
        for candidate_id in SCORING_CANDIDATE_IDS
    ]
    scoring_selection = {
        "grid_version": SCORING_GRID_VERSION,
        "candidates": scoring_candidates,
        "selection_rule": SCORING_SELECTION_RULE,
        "minimum_proposals": 10,
        "minimum_coverage": 0.10,
        "minimum_lcb90": 0.0,
        "selected_candidate_id": "success_only",
    }
    scoring = {**scoring_candidates[0], "event_values": [0, 0.25, 0.5, 0.75, 1]}
    guard = {
        "minimum_guarded_groups": 10,
        "minimum_coverage": 0.10,
        "minimum_lcb": 0.0,
        "maximum_harmful_rate": 0.10,
        "grid_version": GUARD_GRID_VERSION,
        "threshold_candidates": [],
    }
    aggregate = root / "counterfactual_ensemble.pt"
    torch.save(
        {
            "scoring": scoring,
            "scoring_selection": scoring_selection,
            "guard": guard,
        },
        aggregate,
    )
    ensemble = {
        "format": "etsf_counterfactual_ensemble_v1",
        "ensemble_checkpoint": {
            "path": str(aggregate.resolve()),
            "sha256": sha256(aggregate),
        },
        "members": members,
        "config": {"structured_events": True},
        "contract": contract,
        "scoring": scoring,
        "scoring_selection": scoring_selection,
        "guard": guard,
        "test_policy": "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened",
    }
    (root / "ensemble_manifest.json").write_text(
        json.dumps(ensemble), encoding="utf-8"
    )
    audit = {
        "status": "launcher_complete",
        "selected_factual": {"checkpoint_sha256": selected_factual_sha},
        "data": {"manifest_sha256": sha256(data / "manifest.json")},
    }
    (root / "launch_audit.json").write_text(json.dumps(audit), encoding="utf-8")


def _command(
    data: Path,
    factual: Path,
    counterfactual: Path,
    event_spec: Path,
    output: Path,
    *,
    python_bin: Path | None = None,
) -> list[str]:
    python_bin = Path(sys.executable) if python_bin is None else python_bin
    return [
        sys.executable,
        str(SCRIPTS / "launch_openvla_etsf_progress_v5.py"),
        "--data",
        str(data),
        "--factual-root",
        str(factual),
        "--counterfactual-root",
        str(counterfactual),
        "--event-spec",
        str(event_spec),
        "--output",
        str(output),
        "--trainer",
        str(SCRIPTS / "train_openvla_etsf_progress_baseline.py"),
        "--python-bin",
        str(python_bin),
        "--wait-timeout-seconds",
        "0",
        "--poll-seconds",
        "0.01",
        "--dry-run",
    ]


def test_all_trainer_argv_preserve_venv_python_symlink(tmp_path: Path) -> None:
    event_spec = tmp_path / "event_spec.json"
    event_digest = _event_spec(event_spec)
    data = tmp_path / "v5_train100"
    _collector(data, event_digest)
    factual = tmp_path / "factual"
    selected_sha = _factual(factual, event_digest)
    counterfactual = tmp_path / "counterfactual"
    _counterfactual(counterfactual, data, selected_sha, event_digest)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())

    completed = subprocess.run(
        _command(
            data,
            factual,
            counterfactual,
            event_spec,
            tmp_path / "output",
            python_bin=venv_python,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout.removeprefix("PROGRESS_BASELINE_V5_DRY_RUN="))
    expected = str(venv_python.absolute())
    assert all(row["argv"][0] == expected for row in audit["commands"])
    assert expected != str(venv_python.resolve())


def test_dry_run_builds_absolute_train_validation_views_and_six_runs(
    tmp_path: Path,
) -> None:
    event_spec = tmp_path / "event_spec.json"
    event_digest = _event_spec(event_spec)
    data = tmp_path / "v5_train100"
    _collector(data, event_digest)
    factual = tmp_path / "factual"
    selected_sha = _factual(factual, event_digest)
    counterfactual = tmp_path / "counterfactual"
    _counterfactual(counterfactual, data, selected_sha, event_digest)
    output = tmp_path / "progress"
    completed = subprocess.run(
        _command(data, factual, counterfactual, event_spec, output),
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "PROGRESS_BASELINE_V5_DRY_RUN="
    assert completed.stdout.startswith(prefix)
    audit = json.loads(completed.stdout[len(prefix) :])
    assert audit["split_views"]["counts"] == {
        "train": 70,
        "validation": 15,
        "test": 15,
    }
    assert audit["run_count"] == 6
    assert [(row["variant"], row["seed"]) for row in audit["commands"]] == [
        *[("direct", seed) for seed in SEEDS],
        *[("latent_future", seed) for seed in SEEDS],
    ]
    train_view = audit["split_views"]["views"]["train"]["payload"]
    validation_view = audit["split_views"]["views"]["validation"]["payload"]
    assert len(train_view["groups"]) == 70
    assert len(validation_view["groups"]) == 15
    assert all(Path(row["path"]).is_absolute() for row in train_view["groups"])
    sealed_keys = set(audit["split_views"]["split_manifest"]["payload"]["test"])
    assert not sealed_keys & set(train_view["logical_keys"])
    assert not sealed_keys & set(validation_view["logical_keys"])
    assert all("success" not in row for row in train_view["groups"])
    assert audit["sealed_internal_15"]["labels_read"] is False
    descriptor_by_key = {
        descriptor.logical_key: descriptor
        for descriptor in scan_group_descriptors([data])
    }
    serialized_audit = json.dumps(audit)
    for key in sealed_keys:
        assert str(Path(descriptor_by_key[key].path).resolve()) not in serialized_audit
    assert not output.exists()
    output.mkdir()
    (output / "partial.txt").write_text("interrupted", encoding="utf-8")
    conflict = subprocess.run(
        _command(data, factual, counterfactual, event_spec, output),
        check=False,
        capture_output=True,
        text=True,
    )
    assert conflict.returncode != 0
    assert "safe resume is unavailable" in conflict.stderr


def test_counterfactual_wait_state_and_conflicting_output_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "counterfactual"
    root.mkdir()
    assert counterfactual_ready(root)[0] is False
    (root / "launch_audit.json").write_text(
        json.dumps({"status": "failed_nonresumable_manual_new_output_required"}),
        encoding="utf-8",
    )
    try:
        counterfactual_ready(root)
    except RuntimeError as error:
        assert "prerequisite failed" in str(error)
    else:
        raise AssertionError("failed prerequisite was accepted")


def test_complete_suite_verifies_split_command_and_checkpoint_mirrors(
    tmp_path: Path,
) -> None:
    output = tmp_path / "complete"
    output.mkdir()
    split = output / "split_views" / "split_manifest.json"
    split.parent.mkdir()
    split.write_text(json.dumps({"train": ["train"], "validation": ["val"]}))
    split_digest = sha256(split)
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(json.dumps({"calibration": {}}))
    train_view = output / "split_views" / "train"
    validation_view = output / "split_views" / "validation"
    train_view.mkdir()
    validation_view.mkdir()
    (train_view / "manifest.json").write_text(json.dumps({"split": "train"}))
    (validation_view / "manifest.json").write_text(
        json.dumps({"split": "validation"})
    )
    commands = []
    for variant in progress_launcher.VARIANTS:
        for seed in progress_launcher.SEEDS:
            run_output = output / variant / f"seed_{seed}"
            argv = [
                "python",
                "trainer.py",
                "--train-data",
                str(train_view),
                "--validation-data",
                str(validation_view),
                "--event-spec",
                str(event_spec),
                "--split-manifest",
                str(split),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--output",
                str(run_output),
                "--device",
                "cuda",
                "--steps",
                "2000",
                "--latent-dim",
                "64",
                "--action-hidden-dim",
                "48",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.0003",
                "--latent-weight",
                "0.5",
                "--evaluation-interval",
                "50",
            ]
            command = {
                "variant": variant,
                "seed": seed,
                "output": str(run_output),
                "argv": argv,
                "argv_sha256": hashlib.sha256(
                    json.dumps(argv, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            commands.append(command)
            config = {
                "variant": variant,
                "hidden_dim": 4096,
                "action_dim": 14,
                "latent_dim": 64,
                "action_hidden_dim": 48,
                "projection_seed": seed,
            }
            contract = {
                "training_seed": seed,
                "candidate_policy_diagnostics": "validation_only",
                "success_supervision": "terminal_eK_progress_target_only",
                "success_loss": False,
                "checkpoint_selection": "validation_progress_mae_only",
                "model_config": config,
                "optimization": {
                    "steps": 2000,
                    "batch_size": 64,
                    "learning_rate": 0.0003,
                    "latent_weight": 0.5,
                    "evaluation_interval": 50,
                },
                "event_spec": str(event_spec.resolve()),
                "event_spec_sha256": sha256(event_spec),
                "split_audit": {
                    "split_manifest": str(split.resolve()),
                    "split_manifest_sha256": split_digest,
                },
                "train_roots": [
                    {
                        "root": str(train_view.resolve()),
                        "manifest_sha256": sha256(train_view / "manifest.json"),
                    }
                ],
                "validation_roots": [
                    {
                        "root": str(validation_view.resolve()),
                        "manifest_sha256": sha256(
                            validation_view / "manifest.json"
                        ),
                    }
                ],
            }
            run_output.mkdir(parents=True)
            checkpoint = run_output / f"openvla_etsf_progress_{variant}.pt"
            torch.save(
                {
                    "format": "etsf_scalar_progress_baseline_v1",
                    "model": {"weight": torch.ones(1)},
                    "config": config,
                    "contract": contract,
                },
                checkpoint,
            )
            validation = {
                "policy_diagnostics_included": True,
                "progress_mae": 0.1,
                "progress_rmse": 0.2,
                "baseline_success_rate": 0.3,
                "selected_success_rate": 0.4,
                "oracle_success_rate": 0.5,
                "candidate_success_auc_scope": "first_query_candidates_only",
            }
            summary = {
                "format": "etsf_scalar_progress_baseline_v1",
                "status": "training_complete",
                "variant": variant,
                "training_seed": seed,
                "config": config,
                "contract": contract,
                "train": {"policy_diagnostics_included": False},
                "validation": validation,
                "training": {
                    "history": [{"step": 50, "validation": validation}],
                    "best_validation": validation,
                    "selection": {
                        "data": "validation_only",
                        "metric": "progress_mae",
                        "mode": "min",
                        "best_step": 50,
                        "best_value": 0.1,
                    },
                },
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
            }
            (run_output / f"progress_{variant}_summary.json").write_text(
                json.dumps(summary)
            )
    suite = progress_launcher.summarize_completed_runs(
        output, commands, split_digest
    )
    (output / "progress_baseline_suite_summary.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True)
    )
    audit = {
        "format": progress_launcher.FORMAT,
        "status": "launcher_complete",
        "commands": commands,
        "split_views": {
            "split_manifest": {"sha256": split_digest},
            "views": {
                "train": {
                    "path": str(train_view / "manifest.json"),
                    "sha256": sha256(train_view / "manifest.json"),
                },
                "validation": {
                    "path": str(validation_view / "manifest.json"),
                    "sha256": sha256(validation_view / "manifest.json"),
                },
            },
        },
        "sealed_internal_15": {
            "labels_read": False,
            "hdf5_paths_passed_to_progress": False,
            "actor_oracle_evaluation": False,
        },
    }
    (output / "launch_audit.json").write_text(json.dumps(audit))
    assert progress_launcher.validate_complete_output(
        output, commands, split_digest
    )["status"] == "already_complete_skip"

    first = commands[0]
    first_summary_path = Path(first["output"]) / "progress_direct_summary.json"
    first_summary = json.loads(first_summary_path.read_text())
    first_checkpoint = Path(first_summary["checkpoint"])
    payload = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
    payload["contract"] = {**payload["contract"], "success_loss": True}
    torch.save(payload, first_checkpoint)
    first_summary["checkpoint_sha256"] = sha256(first_checkpoint)
    first_summary_path.write_text(json.dumps(first_summary))
    with pytest.raises(RuntimeError, match="checkpoint/summary mirror"):
        progress_launcher.summarize_completed_runs(output, commands, split_digest)
