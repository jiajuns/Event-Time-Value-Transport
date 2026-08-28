from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_openvla_etsf_event_world_model import (  # noqa: E402
    CACHE_SCHEMA,
    collate_transitions,
    derive_atomic_predicates,
    dynamic_event_ids,
    event_transition_target,
    load_or_build_cache,
    lognormal_nll,
    masked_weighted_cross_entropy,
    read_rollout_descriptors,
    relative_transition_ids,
)
from verify_openvla_etsf_factual_run import (  # noqa: E402
    sha256 as verifier_sha256,
    verify_completed_run,
)


def _write_rollout_episode(path: Path, seed: int, success: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["seed"] = seed
        handle.attrs["success"] = success
        handle.attrs["steps"] = 2
        handle.attrs["body"] = "aloha-agilex"
        handle.attrs["model_path"] = "synthetic-openvla"
        handle.create_dataset("query_steps", data=np.asarray([0], dtype=np.int32))
        handle.create_dataset(
            "hidden", data=np.full((1, 8), seed / 1000, dtype=np.float16)
        )
        handle.create_dataset(
            "terminal_hidden", data=np.full(8, seed / 900, dtype=np.float16)
        )
        handle.create_dataset(
            "action_chunks", data=np.full((1, 2, 2), seed / 100, dtype=np.float32)
        )
        handle.create_dataset(
            "object_names", data=np.asarray(["can"], dtype=object), dtype=string_dtype
        )
        poses = np.zeros((3, 1, 7), dtype=np.float32)
        poses[:, :, 3] = 1.0
        poses[:, 0, 0] = [0.0, 0.01, 0.02]
        handle.create_dataset("object_poses", data=poses)
        handle.create_dataset("proprio", data=np.zeros((3, 3), dtype=np.float32))
        handle.create_dataset(
            "event_names",
            data=np.asarray(["e0", "e12"], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset("event_steps", data=np.asarray([0, 2], dtype=np.int32))


def test_event_transition_and_right_censor_targets() -> None:
    names = ["e0", "e12", "eK"]
    steps = [0, 43, 95]
    ids = {"e0": 0, "e12": 1, "eK": 2}

    # The next event is supervised under the frozen continuation even when it
    # lies beyond the first 25-step action chunk.
    assert event_transition_target(0, 25, 95, names, steps, ids) == (0, 1, 43.0, True)
    assert event_transition_target(25, 50, 95, names, steps, ids) == (0, 1, 18.0, True)
    # Partial terminal chunks retain their true horizon and can reach eK.
    assert event_transition_target(75, 95, 95, names, steps, ids) == (1, 2, 20.0, True)

    # If no later event occurs, the terminal horizon is the censor bound.
    assert event_transition_target(50, 75, 95, ["e0"], [0], ids) == (
        0,
        0,
        45.0,
        False,
    )


def test_lognormal_censored_likelihood_has_finite_gradients() -> None:
    mean = torch.tensor([2.0, 3.0], requires_grad=True)
    log_scale = torch.tensor([-0.5, 0.0], requires_grad=True)
    duration = torch.tensor([8.0, 25.0])
    observed = torch.tensor([1.0, 0.0])
    loss = lognormal_nll(mean, log_scale, duration, observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert log_scale.grad is not None and torch.isfinite(log_scale.grad).all()


def test_collate_right_pads_semantic_histories() -> None:
    common = {
        "action_chunks": torch.zeros(2, 1),
        "action_mask": torch.ones(2, dtype=torch.bool),
        "proprio": torch.zeros(1),
        "current_event_id": torch.tensor(0),
        "next_event_id": torch.tensor(0),
        "reach": torch.tensor(0.0),
        "duration": torch.tensor(2.0),
        "duration_observed": torch.tensor(0.0),
        "success": torch.tensor(0.0),
        "body_id": torch.tensor(0),
        "policy_id": torch.tensor(0),
        "horizon": torch.tensor(2.0),
        "object_delta": torch.zeros(3),
    }
    first = {
        **common,
        "hidden_t": torch.ones(1, 4),
        "next_hidden": torch.ones(2, 4),
    }
    second = {
        **common,
        "hidden_t": torch.ones(3, 4),
        "next_hidden": torch.ones(4, 4),
    }
    batch = collate_transitions([first, second])
    assert batch["hidden_t"].shape == (2, 3, 4)
    assert batch["history_mask"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["next_hidden"].shape == (2, 4, 4)
    assert batch["next_history_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_dynamic_predicates_capture_progress_and_regression() -> None:
    # can moves, lifts, reaches a goal at x=0.10, then moves away and drops.
    poses = torch.zeros(7, 2, 7).numpy()
    poses[:, :, 3] = 1.0
    poses[:, 0, 0] = [0.0, 0.03, 0.07, 0.10, 0.10, 0.20, 0.20]
    poses[:, 0, 2] = [0.0, 0.0, 0.04, 0.04, 0.04, 0.0, 0.0]
    calibration = {
        "moving": "can",
        "anchor": None,
        "centers": [[0.10, 0.0, 0.04]],
        "offset": [0.0, 0.0, 0.0],
        "delta_move": 0.05,
        "delta_z": 0.03,
        "tau_d": 0.015,
        "tau_motion": 0.04,
        "stationary_steps": 2,
    }
    predicates = derive_atomic_predicates(
        poses, ["can", "pot"], False, calibration
    )
    assert predicates.shape == (7, 5)
    assert predicates[2, :2].tolist() == [1.0, 1.0]
    assert predicates[3, 2] == 1.0
    assert predicates[4, 3] == 1.0
    assert predicates[5, 1:4].tolist() == [0.0, 0.0, 0.0]

    event_to_id = {"e0": 0, "e12": 1, "e3": 2, "e4": 3, "eK": 4}
    dynamic = dynamic_event_ids(predicates, event_to_id)
    assert dynamic.tolist() == [0, 0, 1, 2, 3, 1, 1]
    relative = relative_transition_ids(dynamic[:-1], dynamic[1:])
    assert relative.tolist() == [0, 1, 1, 1, 3, 0]


def test_success_predicate_is_terminal_only() -> None:
    poses = torch.zeros(3, 1, 7).numpy()
    poses[:, :, 3] = 1.0
    calibration = {
        "moving": "can",
        "anchor": None,
        "centers": [[1.0, 0.0, 0.0]],
        "delta_move": 1.0,
        "delta_z": 1.0,
        "tau_d": 0.01,
        "tau_motion": 0.01,
        "stationary_steps": 2,
    }
    predicates = derive_atomic_predicates(poses, ["can"], True, calibration)
    assert predicates[:, 4].tolist() == [0.0, 0.0, 1.0]


def test_masked_cross_entropy_is_zero_with_no_observed_targets() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    loss = masked_weighted_cross_entropy(
        logits,
        torch.tensor([0, 1, 2]),
        sample_mask=torch.zeros(3),
    )
    assert loss == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_masked_cross_entropy_is_zero_when_all_class_weights_are_unsupported() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    with torch.no_grad():
        logits[0, 3] = -torch.inf
    loss = masked_weighted_cross_entropy(
        logits,
        torch.tensor([0, 1, 2]),
        class_weight=torch.zeros(4),
        sample_mask=torch.ones(3),
    )
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_sealed_broken_episode_is_not_opened_or_cached(tmp_path: Path) -> None:
    data = tmp_path / "rollouts"
    episodes = data / "episodes"
    seeds = [101, 102, 103, 104]
    for seed in seeds[:3]:
        _write_rollout_episode(
            episodes / f"episode_{seed}.hdf5", seed, success=seed == 102
        )

    # This is a valid HDF5 container but an intentionally invalid episode: it
    # has neither transition inputs nor usable labels.  Training can complete
    # only if the sealed container is never opened as HDF5.
    broken = episodes / "episode_104.hdf5"
    with h5py.File(broken, "w") as handle:
        handle.attrs["seed"] = 104
        handle.create_dataset(
            "success", data=np.asarray(["corrupt-label"], dtype=h5py.string_dtype())
        )

    data.mkdir(parents=True, exist_ok=True)
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(
        json.dumps(
            {
                "calibration": {
                    "beat_block_hammer": {
                        "moving": "can",
                        "anchor": "",
                        "centers": [[1.0, 0.0, 0.0]],
                        "offset": [0.0, 0.0, 0.0],
                        "delta_move": 0.005,
                        "delta_z": 1.0,
                        "tau_d": 0.01,
                        "tau_motion": 0.01,
                        "stationary_steps": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "beat_block_hammer",
        "body": "aloha-agilex",
        "model_path": "synthetic-openvla",
        "event_spec": str(event_spec),
        "event_spec_sha256": verifier_sha256(event_spec),
        "requested_seeds": seeds,
        # Outcome/step summaries are intentionally present, as in the real
        # collector, but descriptor-first splitting must not use them.
        "episodes": [
            {
                "index": index,
                "seed": seed,
                "path": f"episode_{seed}.hdf5",
                "success": seed == 102,
                "steps": 2 if seed != 104 else "corrupt",
            }
            for index, seed in enumerate(seeds)
        ],
    }
    (data / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {"train": [101, 102], "validation": [103], "test": [104]}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "training"
    command = [
        sys.executable,
        str(SCRIPTS / "train_openvla_etsf_event_world_model.py"),
        "--data",
        str(data),
        "--output",
        str(output),
        "--split-manifest",
        str(split_path),
        "--event-mode",
        "structured",
        "--event-spec",
        str(event_spec),
        "--device",
        "cpu",
        "--amp",
        "off",
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--eval-every",
        "1",
        "--save-every",
        "1",
        "--num-workers",
        "0",
    ]
    completed = subprocess.run(
        command,
        cwd=SCRIPTS.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    cache = torch.load(
        output / "query_transitions.pt", map_location="cpu", weights_only=False
    )
    assert cache["schema_version"] == CACHE_SCHEMA
    assert cache["event_spec_sha256"] == verifier_sha256(event_spec)
    assert cache["predicate_names"] == [
        "moved",
        "lifted",
        "near_goal",
        "stationary",
        "success",
    ]
    assert set(cache["arrays"]["seed"].tolist()) == {101, 102, 103}
    assert {row["seed"] for row in cache["episodes"]} == {101, 102, 103}
    assert cache["loaded_episode_seeds"] == [101, 102, 103]
    assert cache["sealed_test_files"][0]["seed"] == 104
    assert cache["sealed_test_files"][0]["path"] == str(broken.resolve())
    assert cache["sealed_test_files"][0]["sha256"]
    assert "success" not in cache["sealed_test_files"][0]
    assert "steps" not in cache["sealed_test_files"][0]

    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["sealed_test_episode_datasets_opened"] == 0
    assert audit["sealed_test_transition_count"] == "unknown_not_loaded"
    assert audit["loaded_train_validation_episodes"] == 3
    summary = json.loads(
        (output / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "training_complete"
    assert summary["sealed_test_evaluated"] is False
    assert summary["contract"]["event_mode"] == "structured"
    assert summary["contract"]["training_seed"] == 20260827
    assert summary["contract"]["predicate_contract"]["task_calibration"] == {
        "moving": "can",
        "anchor": "",
        "centers": [[1.0, 0.0, 0.0]],
        "offset": [0.0, 0.0, 0.0],
        "delta_move": 0.005,
        "delta_z": 1.0,
        "tau_d": 0.01,
        "tau_motion": 0.01,
        "stationary_steps": 2,
    }

    parsed_manifest, descriptors = read_rollout_descriptors(data)
    event_spec.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="event spec digest differs"):
        load_or_build_cache(
            data,
            output / "query_transitions.pt",
            ["e0", "e12", "e3", "e4", "eK"],
            ["can"],
            False,
            manifest=parsed_manifest,
            episode_descriptors=descriptors[:3],
            sealed_test_descriptors=descriptors[3:],
            split_seeds={"train": [101, 102], "validation": [103], "test": [104]},
            event_spec_path=event_spec,
            require_predicates=True,
        )


def test_pre_split_cache_schema_is_explicitly_rejected(tmp_path: Path) -> None:
    data = tmp_path / "rollouts"
    episodes = data / "episodes"
    episodes.mkdir(parents=True)
    seeds = [1, 2, 3]
    for seed in seeds:
        (episodes / f"episode_{seed}.hdf5").write_bytes(b"identity-only")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "requested_seeds": seeds,
        "episodes": [
            {"index": index, "seed": seed, "path": f"episode_{seed}.hdf5"}
            for index, seed in enumerate(seeds)
        ],
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    parsed_manifest, descriptors = read_rollout_descriptors(data)
    cache_path = tmp_path / "old_cache.pt"
    torch.save({"schema_version": CACHE_SCHEMA - 1}, cache_path)
    split = {"train": [1], "validation": [2], "test": [3]}
    with pytest.raises(RuntimeError, match="contract mismatch.*rebuild-cache"):
        load_or_build_cache(
            data,
            cache_path,
            ["e0", "e12", "e3", "e4", "eK"],
            ["can"],
            False,
            manifest=parsed_manifest,
            episode_descriptors=descriptors[:2],
            sealed_test_descriptors=descriptors[2:],
            split_seeds=split,
        )


def test_formal_completion_verifier_binds_seed_split_spec_cache_and_best(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps({"status": "complete", "task": "move_can_pot"}),
        encoding="utf-8",
    )
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(
        json.dumps({"calibration": {"move_can_pot": {"delta_move": 0.05}}}),
        encoding="utf-8",
    )
    split_path = tmp_path / "split.json"
    split = {"train": [1], "validation": [2], "test": [3]}
    split_path.write_text(json.dumps(split), encoding="utf-8")
    event_digest = verifier_sha256(event_spec)
    manifest_digest = verifier_sha256(data / "manifest.json")
    cache_path = tmp_path / "query_transitions_schema3_train_validation_only.pt"
    cache = {
        "schema_version": 3,
        "source_manifest_sha256": manifest_digest,
        "event_spec_sha256": event_digest,
        "task_calibration": {"delta_move": 0.05},
        "predicate_names": ["moved", "lifted", "near_goal", "stationary", "success"],
        "split_seeds": split,
        "loaded_episode_seeds": [1, 2],
        "arrays": {"seed": np.asarray([1, 2], dtype=np.int64)},
        "sealed_test_access": (
            "manifest_identity_and_raw_file_sha256_only_no_episode_hdf5_open"
        ),
        "sealed_test_files": [{"seed": 3, "path": "sealed.hdf5", "sha256": "abc"}],
    }
    torch.save(cache, cache_path)
    contract = {
        "cache_schema": 3,
        "training_seed": 11,
        "event_mode": "structured",
        "source_manifest_sha256": manifest_digest,
        "event_spec_sha256": event_digest,
        "train_seeds": [1],
        "validation_seeds": [2],
        "sealed_test_seeds": [3],
        "predicate_contract": {
            "names": ["moved", "lifted", "near_goal", "stationary", "success"],
            "derivation": "derive_atomic_predicates_v1",
            "source": "simulator_object_poses_at_query_step",
            "event_spec_sha256": event_digest,
            "task_calibration": {"delta_move": 0.05},
            "online_requires_explicit_predicates": True,
            "missing_policy": "error",
        },
    }
    output = tmp_path / "seed_11"
    output.mkdir()
    best_path = output / "event_world_model_best.pt"
    latest_path = output / "event_world_model_latest.pt"
    best_score = 1.25
    torch.save(
        {
            "contract": contract,
            "config": {"structured_events": True},
            "step": 80,
            "best_score": best_score,
        },
        best_path,
    )
    torch.save(
        {
            "contract": contract,
            "config": {"structured_events": True},
            "step": 100,
            "best_step": 80,
            "best_score": best_score,
        },
        latest_path,
    )
    summary_path = output / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "training_complete",
                "steps": 100,
                "requested_steps": 100,
                "stopped_early": False,
                "best_step": 80,
                "best_validation_selection_score": best_score,
                "best_validation": {"event_macro_f1": 0.5},
                "sealed_test_evaluated": False,
                "checkpoint": str(best_path),
                "resume_checkpoint": str(latest_path),
                "contract": contract,
            }
        ),
        encoding="utf-8",
    )
    report = verify_completed_run(
        summary_path,
        seed=11,
        requested_steps=100,
        data_root=data,
        split_manifest=split_path,
        event_spec=event_spec,
        cache_path=cache_path,
    )
    assert report["status"] == "complete_verified"
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        verify_completed_run(
            summary_path,
            seed=12,
            requested_steps=100,
            data_root=data,
            split_manifest=split_path,
            event_spec=event_spec,
            cache_path=cache_path,
        )

    cache["arrays"]["seed"] = np.asarray([1, 2, 3], dtype=np.int64)
    torch.save(cache, cache_path)
    with pytest.raises(RuntimeError, match="sealed test seeds present"):
        verify_completed_run(
            summary_path,
            seed=11,
            requested_steps=100,
            data_root=data,
            split_manifest=split_path,
            event_spec=event_spec,
            cache_path=cache_path,
        )


def test_formal_launcher_is_fail_closed_and_uses_dedicated_schema3_cache() -> None:
    launcher = (
        SCRIPTS / "run_openvla_etsf_event_world_model_ensemble.sh"
    ).read_text(encoding="utf-8")
    assert "query_transitions_schema3_train_validation_only.pt" in launcher
    assert "--event-spec \"$event_spec\"" in launcher
    assert "verify_openvla_etsf_factual_run.py" in launcher
    assert "formal ensemble requires exactly three seeds" in launcher
    assert "grep -q '\"status\"" not in launcher
    assert "Use a new OUTPUT_ROOT; existing files will not be overwritten." in launcher
