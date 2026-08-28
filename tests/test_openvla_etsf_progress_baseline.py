from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_openvla_etsf_progress_baseline import (  # noqa: E402
    ProgressModelConfig,
    ScalarProgressBaseline,
    action_statistics,
    audit_split,
    dynamic_phase_progress,
    evaluate,
    load_counterfactual_root,
    progress_loss,
    sha256,
    train_baseline,
)


CALIBRATION = {
    "moving": "can",
    "anchor": "pot",
    "offset": [0.0, 0.0, 0.0],
    "delta_move": 0.05,
    "delta_z": 0.05,
    "tau_d": 0.06,
    "tau_motion": 0.02,
    "stationary_steps": 2,
}


def _poses(length: int, candidate: int) -> np.ndarray:
    poses = np.zeros((length, 2, 7), dtype=np.float32)
    poses[:, :, 3] = 1.0
    poses[:, 1, 0] = 0.5  # pot
    poses[:, 0, 0] = np.linspace(0.0, 0.5 - 0.01 * candidate, length)
    if candidate == 0:
        poses[:, 0, 0] = np.linspace(0.0, 0.2, length)
    return poses


def _write_root(root: Path, schema: int, seed: int) -> None:
    (root / "groups").mkdir(parents=True)
    manifest = {
        "status": "complete",
        "schema_version": schema,
        "task": "move_can_pot",
        "body": "piper",
        "policy": "openvla",
        "groups": [{"path": f"group_seed_{seed}.hdf5"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path = root / "groups" / f"group_seed_{seed}.hdf5"
    count, chunk, terminal = 2, 4, 8
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(count, 4096)).astype(np.float32)
    post_hidden = (hidden + 0.1).astype(np.float32)
    actions = rng.normal(size=(count, chunk, 14)).astype(np.float32)
    mask = np.ones((count, chunk), dtype=bool)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = schema
        handle.attrs["task"] = "move_can_pot"
        handle.attrs["body"] = "piper"
        handle.attrs["policy"] = "openvla"
        handle.attrs["seed"] = seed
        handle.attrs["resolved_seed"] = seed
        handle.create_dataset("initial_hidden", data=hidden[0])
        handle.create_dataset("pre_hidden", data=hidden)
        handle.create_dataset("post_chunk_hidden", data=post_hidden)
        handle.create_dataset("candidate_actions", data=actions)
        handle.create_dataset("first_chunk_action_mask", data=mask)
        handle.create_dataset("post_chunk_step", data=np.full(count, chunk))
        handle.create_dataset("success", data=np.asarray([0.0, 1.0]))
        handle.create_dataset("steps", data=np.full(count, terminal))
        strings = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "object_names", data=np.asarray(["can", "pot"], dtype=object), dtype=strings
        )
        branches = handle.create_group("branches")
        for candidate in range(count):
            branch = branches.create_group(f"candidate_{candidate:03d}")
            branch.create_dataset("object_poses", data=_poses(terminal + 1, candidate))
            if schema == 5:
                query_hidden = np.stack([hidden[candidate], post_hidden[candidate]])
                query_post_hidden = np.stack(
                    [post_hidden[candidate], post_hidden[candidate] + 0.1]
                )
                query_actions = np.stack(
                    [actions[candidate], actions[candidate] * 0.5]
                )
                query_mask = np.ones((2, chunk), dtype=bool)
                branch.create_dataset("query_steps", data=np.asarray([0, 4]))
                branch.create_dataset("query_post_steps", data=np.asarray([4, 8]))
                branch.create_dataset("query_hidden", data=query_hidden)
                branch.create_dataset("query_post_hidden", data=query_post_hidden)
                branch.create_dataset("query_actions", data=query_actions)
                branch.create_dataset("query_action_mask", data=query_mask)


def test_dynamic_progress_is_bounded_and_reversible() -> None:
    poses = _poses(6, 1)
    near = dynamic_phase_progress(poses, ["can", "pot"], CALIBRATION, False)
    regressed = poses.copy()
    regressed[-1, 0, 0] = 0.0
    back = dynamic_phase_progress(regressed, ["can", "pot"], CALIBRATION, False)
    terminal = dynamic_phase_progress(poses, ["can", "pot"], CALIBRATION, True)
    assert 0.0 <= back < near <= terminal == 1.0


def test_loader_supports_v4_and_v5_continuation_queries(tmp_path: Path) -> None:
    root4, root5 = tmp_path / "v4", tmp_path / "v5"
    _write_root(root4, 4, 10)
    _write_root(root5, 5, 11)
    calibrations = {"move_can_pot": CALIBRATION}
    loaded4 = load_counterfactual_root(root4, calibrations)
    loaded5 = load_counterfactual_root(root5, calibrations)
    assert len(loaded4.examples) == 2
    assert len(loaded5.examples) == 4
    assert [item.query_index for item in loaded5.examples] == [0, 1, 0, 1]
    audit = audit_split(loaded4, loaded5)
    assert audit["sealed_test_policy"] == "not accepted_by_cli_not_loaded_not_evaluated"


def test_manifest_only_split_view_requires_absolute_path_key_and_sha(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_root(source, 5, 31)
    hdf5 = (source / "groups" / "group_seed_31.hdf5").resolve()
    view = tmp_path / "view"
    view.mkdir()
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "train": ["move_can_pot|piper|31"],
                "validation": ["move_can_pot|piper|32"],
                "test": ["move_can_pot|piper|33"],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "format": "etsf_progress_split_view_v1",
        "status": "complete",
        "schema_version": 5,
        "split": "train",
        "task": "move_can_pot",
        "body": "piper",
        "policy": "openvla",
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256(split_manifest),
        "logical_keys": ["move_can_pot|piper|31"],
        "groups": [
            {
                "logical_key": "move_can_pot|piper|31",
                "schema_version": 5,
                "resolved_seed": 31,
                "path": str(hdf5),
                "sha256": sha256(hdf5),
            }
        ],
    }
    manifest = view / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_counterfactual_root(view, {"move_can_pot": CALIBRATION})
    assert loaded.logical_keys == ["move_can_pot|piper|31"]
    payload["groups"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_counterfactual_root(view, {"move_can_pot": CALIBRATION})
    except RuntimeError as error:
        assert "SHA256 mismatch" in str(error)
    else:
        raise AssertionError("tampered split-view HDF5 was accepted")


def test_both_progress_models_and_metrics_run_on_cpu(tmp_path: Path) -> None:
    root = tmp_path / "v5"
    _write_root(root, 5, 12)
    loaded = load_counterfactual_root(root, {"move_can_pot": CALIBRATION})
    mean, std = action_statistics(loaded.examples)
    batch = loaded.examples
    hidden = torch.from_numpy(np.stack([item.hidden for item in batch]))
    actions = torch.from_numpy(np.stack([item.actions for item in batch]))
    mask = torch.from_numpy(np.stack([item.action_mask for item in batch]))
    post_hidden = torch.from_numpy(np.stack([item.post_hidden for item in batch]))
    target = torch.tensor([item.progress for item in batch])
    for variant in ["direct", "latent_future"]:
        model = ScalarProgressBaseline(
            ProgressModelConfig(
                variant=variant,
                latent_dim=8,
                action_hidden_dim=8,
                projection_seed=123,
            ),
            mean,
            std,
        )
        output = model(hidden, actions, mask, post_hidden)
        loss, _ = progress_loss(output, target, latent_weight=0.5)
        loss.backward()
        assert torch.isfinite(loss)
        metrics = evaluate(model, batch, batch_size=4, device=torch.device("cpu"), bootstrap_seed=1)
        assert metrics["examples"] == 4
        assert metrics["groups"] == 1
        assert 0.0 <= metrics["progress_mae"] <= 1.0
        if variant == "latent_future":
            assert "future_latent_cosine" in metrics

    trained, result = train_baseline(
        config=ProgressModelConfig(
            variant="direct", latent_dim=8, action_hidden_dim=8, projection_seed=7
        ),
        train_examples=batch,
        validation_examples=batch,
        device=torch.device("cpu"),
        steps=2,
        batch_size=4,
        learning_rate=1e-3,
        latent_weight=0.5,
        seed=7,
        evaluation_interval=1,
    )
    assert isinstance(trained, ScalarProgressBaseline)
    assert len(result["history"]) == 2
    assert "best_validation" in result


def test_candidate_success_auc_uses_first_query_candidates_only(tmp_path: Path) -> None:
    root = tmp_path / "v5_auc"
    _write_root(root, 5, 13)
    loaded = load_counterfactual_root(root, {"move_can_pot": CALIBRATION})

    class FixedProgress:
        def eval(self) -> None:
            return None

        def __call__(self, hidden, actions, action_mask, post_hidden):
            # Order is candidate0/q0, candidate0/q1, candidate1/q0,
            # candidate1/q1. First-query ranking is perfect; continuations are
            # deliberately inverted and must not change candidate AUC.
            return {
                "progress": torch.tensor(
                    [0.1, 0.99, 0.9, 0.0], device=hidden.device
                )
            }

    metrics = evaluate(
        FixedProgress(),
        loaded.examples,
        batch_size=4,
        device=torch.device("cpu"),
        bootstrap_seed=1,
    )
    assert metrics["candidate_success_auc"] == 1.0
    assert metrics["candidate_success_auc_scope"] == (
        "first_query_candidates_only"
    )


def test_cpu_cli_uses_only_explicit_development_roots(tmp_path: Path) -> None:
    train_root, validation_root = tmp_path / "train", tmp_path / "validation"
    _write_root(train_root, 5, 21)
    _write_root(validation_root, 5, 22)
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(
        json.dumps({"calibration": {"move_can_pot": CALIBRATION}}), encoding="utf-8"
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "train": ["move_can_pot|piper|21"],
                "validation": ["move_can_pot|piper|22"],
                "test": ["this sealed reference must not be resolved or opened"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_openvla_etsf_progress_baseline.py"),
            "--train-data",
            str(train_root),
            "--validation-data",
            str(validation_root),
            "--event-spec",
            str(event_spec),
            "--split-manifest",
            str(split_manifest),
            "--variant",
            "latent_future",
            "--latent-dim",
            "8",
            "--action-hidden-dim",
            "8",
            "--steps",
            "1",
            "--evaluation-interval",
            "1",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PROGRESS_BASELINE_COMPLETE=" in completed.stdout
    summary = json.loads(
        (output / "progress_latent_future_summary.json").read_text(encoding="utf-8")
    )
    assert summary["contract"]["success_supervision"] == (
        "terminal_eK_progress_target_only"
    )
    assert summary["contract"]["success_loss"] is False
    assert summary["contract"]["candidate_policy_diagnostics"] == "validation_only"
    assert summary["train"]["policy_diagnostics_included"] is False
    assert "oracle_success_rate" not in summary["train"]
    assert summary["validation"]["policy_diagnostics_included"] is True
    assert "oracle_success_rate" in summary["validation"]
    assert summary["validation"]["continuation_query_examples"] == 2
    assert summary["validation"]["candidate_success_auc_scope"] == (
        "first_query_candidates_only"
    )
    assert summary["training"]["selection"] == {
        "data": "validation_only",
        "metric": "progress_mae",
        "mode": "min",
        "best_step": 1,
        "best_value": summary["training"]["best_validation"]["progress_mae"],
    }
    checkpoint = torch.load(
        summary["checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint["contract"] == summary["contract"]
    assert checkpoint["config"] == summary["config"]
    assert summary["contract"]["optimization"]["steps"] == 1
    assert summary["limitations"][-1] == "Sealed test data was not loaded or evaluated."
