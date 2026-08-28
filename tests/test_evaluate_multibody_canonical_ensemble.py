from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_multibody_canonical_ensemble import (  # noqa: E402
    EvaluationError,
    authenticate_training_bundle,
    compute_ensemble_metrics,
    equal_weight_lognormal_mixture_median,
    load_validation_only,
    reconstruct_frozen_split,
    run_synthetic_smoke,
)
from train_multibody_canonical_event_world_model import (  # noqa: E402
    FORMAT,
    GroupDescriptor,
    InputBinding,
    ModelConfig,
    MultibodyCanonicalEventWorldModel,
    body_alias_receipt,
    canonical_json_sha256,
    scan_schema5_groups,
    scan_stage1_groups,
    sha256_file,
    split_receipt,
    strict_group_split,
    verify_input_bindings,
)


def _signed(value: dict[str, object], key: str = "sha256") -> dict[str, object]:
    result = dict(value)
    result[key] = canonical_json_sha256(result)
    return result


def _normalization() -> dict[str, object]:
    schemas = {}
    for schema_id, name in enumerate(("aloha", "arx", "openvla")):
        schemas[name] = {
            "schema_id": schema_id,
            "train_rows": 2,
            "train_logical_groups": 2,
            "valid_action_steps": 4,
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
        }
    return _signed(
        {
            "format": "etsf_train_only_action_normalization_v1",
            "source_split": "train_only",
            "validation_rows_used": 0,
            "test_rows_used": 0,
            "unavailable_train_rows_excluded": 0,
            "schemas": schemas,
        }
    )


def _baseline() -> dict[str, object]:
    return _signed(
        {
            "format": "etsf_train_only_validation_baselines_v1",
            "source_split": "train_only",
            "validation_rows_used": 0,
            "test_rows_used": 0,
            "majority_post_event": 0,
            "majority_next_event": 1,
            "duration_median_by_body_event": {},
            "duration_median_by_event": {},
            "duration_global_median": 2.0,
            "empirical_success": 0.5,
            "zero_object_delta": [0.0] * 6,
            "zero_object_scale": [1.0] * 6,
            "support": {
                "post_event_rows": 10,
                "next_event_rows": 10,
                "observed_duration_rows": 10,
                "success_rows": 10,
                "object_rows": 10,
            },
        }
    )


def _write_training_bundle(tmp_path: Path) -> tuple[Path, str, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    normalization = _normalization()
    baseline = _baseline()
    baseline_metrics = {
        "source": "train_only_baselines_evaluated_on_validation",
        "post_event": {"accuracy": 0.2, "macro_f1": 0.1, "support": 10},
        "next_event": {"accuracy": 0.2, "macro_f1": 0.1, "support": 10},
        "observed_duration_mae": 2.0,
        "success_brier": 0.25,
        "success_auroc": 0.5,
        "object_rmse": 1.0,
        "object_nll": 1.5,
    }
    selection_rule = _signed(
        {"format": "etsf_multibody_validation_selection_v1"}
    )
    protocol = {
        "format": FORMAT,
        "sealed_test_group_hdf5_opened": 0,
        "labels_used_for_split": False,
        "test_transition_count": "unknown_not_loaded",
        "test_group_hdf5_opened": 0,
        "action_normalization": normalization,
        "train_only_baselines": baseline,
        "validation_baseline_metrics": baseline_metrics,
        "validation_selection_rule": selection_rule,
    }
    config = ModelConfig(body_count=2, dropout=0.0)
    members = []
    paths = []
    for member in range(5):
        model = MultibodyCanonicalEventWorldModel(config)
        validation = {"selection_score": float(member + 1)}
        checkpoint = {
            "format": FORMAT,
            "model": model.state_dict(),
            "config": {
                "body_count": 2,
                "action_schema_count": 3,
                "semantic_dim": 96,
                "clock_dim": 64,
                "object_delta_dim": 6,
                "dropout": 0.0,
            },
            "contract": protocol,
            "action_normalization": normalization,
            "train_only_baselines": baseline,
            "validation_baseline_metrics": baseline_metrics,
            "validation_selection_rule": selection_rule,
            "member": member,
            "seed": 100 + member,
            "step": 50,
            "validation": validation,
            "selection_score": float(member + 1),
        }
        path = tmp_path / f"member_{member}.pt"
        torch.save(checkpoint, path)
        paths.append(path)
        members.append(
            {
                "member": member,
                "seed": 100 + member,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "best_step": 50,
                "best_validation_selection_score": float(member + 1),
                "best_validation": validation,
            }
        )
    summary = {
        "format": FORMAT,
        "status": "training_complete",
        "members": members,
        "protocol": protocol,
        "sealed_test_evaluated": False,
        "test_group_hdf5_opened": 0,
    }
    summary_path = tmp_path / "training_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, sha256_file(summary_path), paths


def test_bundle_authenticates_all_five_sha_contract_config_and_normalization(
    tmp_path: Path,
) -> None:
    summary, digest, _ = _write_training_bundle(tmp_path)
    value, checkpoints, receipt = authenticate_training_bundle(summary, digest)
    assert value["status"] == "training_complete"
    assert len(checkpoints) == 5
    assert len(receipt["checkpoints"]) == 5
    assert len(receipt["checkpoint_bundle_sha256"]) == 64
    assert receipt["model_config"]["body_count"] == 2


def test_bundle_rejects_checkpoint_byte_tamper_and_normalization_drift(
    tmp_path: Path,
) -> None:
    summary, digest, paths = _write_training_bundle(tmp_path)
    paths[2].write_bytes(paths[2].read_bytes() + b"tampered")
    with pytest.raises(EvaluationError, match="checkpoint SHA-256 mismatch"):
        authenticate_training_bundle(summary, digest)

    summary, _, paths = _write_training_bundle(tmp_path / "second")
    payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    payload["model"]["action.action_mean"][0, 0] = 1.0
    torch.save(payload, paths[0])
    summary_value = json.loads(summary.read_text(encoding="utf-8"))
    summary_value["members"][0]["checkpoint_sha256"] = sha256_file(paths[0])
    summary.write_text(json.dumps(summary_value), encoding="utf-8")
    with pytest.raises(EvaluationError, match="differs from normalization"):
        authenticate_training_bundle(summary, sha256_file(summary))


def _prediction_values() -> dict[str, np.ndarray]:
    rows = 12
    labels = np.arange(rows) % 5
    post = np.full((5, rows, 5), 0.025, dtype=np.float64)
    next_probability = np.full_like(post, 0.025)
    for member in range(5):
        post[member, np.arange(rows), labels] = 0.9
        next_probability[member, np.arange(rows), (labels + (member == 4)) % 5] = 0.9
        post[member] /= post[member].sum(axis=-1, keepdims=True)
        next_probability[member] /= next_probability[member].sum(
            axis=-1, keepdims=True
        )
    success_label = (np.arange(rows) % 2).astype(np.float64)
    success = np.tile(
        np.where(success_label > 0.5, 0.8, 0.2)[None, :], (5, 1)
    )
    success[:, [0, 1, 2]] = 1.0 - success[:, [0, 1, 2]]
    success += np.linspace(-0.04, 0.04, 5)[:, None]
    success = np.clip(success, 0.01, 0.99)
    duration = np.linspace(1.0, 12.0, rows)
    duration_log_mean = np.tile(np.log1p(duration)[None, :], (5, 1))
    duration_log_mean += np.linspace(-0.1, 0.1, 5)[:, None]
    object_label = np.linspace(-1.0, 1.0, rows * 6).reshape(rows, 6)
    object_mean = np.tile(object_label[None, :, :], (5, 1, 1))
    object_mean += np.linspace(-0.1, 0.1, 5)[:, None, None]
    return {
        "body_id": np.arange(rows) % 2,
        "post_label": labels,
        "post_mask": np.ones(rows),
        "next_label": labels,
        "next_mask": np.ones(rows),
        "success_label": success_label,
        "success_mask": np.ones(rows),
        "duration_label": duration,
        "duration_observed": np.ones(rows),
        "duration_mask": np.ones(rows),
        "object_label": object_label,
        "object_mask": np.ones(rows),
        "post_probability": post,
        "next_probability": next_probability,
        "success_probability": success,
        "duration_log_mean": duration_log_mean,
        "duration_log_scale": np.full((5, rows), -2.0),
        "object_mean": object_mean,
        "object_log_scale": np.full((5, rows, 6), -1.0),
    }


def test_synthetic_metrics_cover_global_body_mixtures_and_uncertainty() -> None:
    values = _prediction_values()
    global_metrics = compute_ensemble_metrics(values)
    body_metrics = compute_ensemble_metrics(values, values["body_id"] == 1)
    assert global_metrics["rows"] == 12
    assert body_metrics["rows"] == 6
    for name in ("post_event", "next_event"):
        assert global_metrics[name]["accuracy"] is not None
        assert global_metrics[name]["macro_f1"] is not None
        assert global_metrics[name]["nll"] is not None
        uncertainty = global_metrics[name]["uncertainty"]
        assert len(uncertainty["per_member_disagreement_rate"]) == 5
        assert uncertainty["total"]["mean"] >= uncertainty["aleatoric"]["mean"]
    assert global_metrics["success"]["auroc_status"] == "available"
    assert global_metrics["success"]["brier"] is not None
    assert global_metrics["success"]["ece_10_bin"] is not None
    assert global_metrics["success"]["error_detection"]["status"] == "available"
    assert global_metrics["observed_duration"]["mixture_nll"] is not None
    assert global_metrics["observed_duration"]["mae"] == global_metrics[
        "observed_duration"
    ]["mixture_median_mae"]
    assert global_metrics["observed_duration"]["prediction_diagnostics"][
        "median_solver"
    ]["labels_used"] is False
    assert global_metrics["observed_duration"]["uncertainty"][
        "law_of_total_variance_verified"
    ] is True
    assert global_metrics["object"]["rmse"] is not None
    assert global_metrics["object"]["nll"] is not None
    assert global_metrics["object"]["uncertainty"][
        "law_of_total_variance_verified"
    ] is True


def test_heavy_tail_duration_uses_label_independent_mixture_median_for_mae() -> None:
    values = _prediction_values()
    labels = values["duration_label"].copy()
    # All five components share the same median, while one component has a
    # large log-scale and therefore an enormous arithmetic mean.
    common_log_median = np.log1p(labels)
    values["duration_log_mean"][:] = common_log_median[None, :]
    values["duration_log_scale"][:] = -2.0
    values["duration_log_scale"][0] = 1.5
    prediction, audit = equal_weight_lognormal_mixture_median(
        values["duration_log_mean"], values["duration_log_scale"]
    )
    assert np.allclose(prediction, labels, rtol=1e-10, atol=1e-10)
    assert audit["labels_used"] is False
    assert audit["upper_finite_boundary_rows"] == 0

    metrics = compute_ensemble_metrics(values)["observed_duration"]
    assert metrics["mae"] == metrics["mixture_median_mae"]
    assert metrics["mixture_median_mae"] < 1e-9
    assert metrics["mixture_mean_mae_heavy_tail_diagnostic"] > 100.0
    assert metrics["prediction_diagnostics"]["mixture_mean_heavy_tail_diagnostic"][
        "mean"
    ] > metrics["prediction_diagnostics"]["mixture_median"]["mean"]

    # Changing labels changes the reported loss only; it cannot change the
    # point prediction or its CDF solver audit.
    changed = dict(values)
    changed["duration_label"] = labels + 1000.0
    changed_metrics = compute_ensemble_metrics(changed)["observed_duration"]
    assert changed_metrics["prediction_diagnostics"] == metrics[
        "prediction_diagnostics"
    ]
    assert changed_metrics["mixture_median_mae"] != metrics["mixture_median_mae"]


def test_duration_median_solver_has_finite_physical_boundaries() -> None:
    lower, lower_audit = equal_weight_lognormal_mixture_median(
        np.full((5, 2), -10.0), np.full((5, 2), -2.0)
    )
    assert np.array_equal(lower, np.zeros(2))
    assert lower_audit["lower_zero_boundary_rows"] == 2

    upper, upper_audit = equal_weight_lognormal_mixture_median(
        np.full((5, 1), 1000.0), np.full((5, 1), -2.0)
    )
    assert np.isfinite(upper).all()
    assert upper[0] == np.finfo(np.float64).max
    assert upper_audit["upper_finite_boundary_rows"] == 1
    with pytest.raises(EvaluationError, match="log scale violates"):
        equal_weight_lognormal_mixture_median(
            np.zeros((5, 1)), np.full((5, 1), 3.0)
        )


def test_variance_identities_promote_float32_and_report_numeric_residuals() -> None:
    values = _prediction_values()
    for key in (
        "success_probability",
        "object_mean",
        "object_log_scale",
        "duration_log_mean",
        "duration_log_scale",
    ):
        values[key] = values[key].astype(np.float32)
    metrics = compute_ensemble_metrics(values)
    for family in ("success", "object", "observed_duration"):
        uncertainty = metrics[family]["uncertainty"]
        assert uncertainty["law_of_total_variance_verified"] is True
        audit = uncertainty["law_of_total_variance_audit"]
        assert audit["verified"] is True
        assert audit["member_outputs_promoted_to"].startswith("float64")
        assert audit["max_abs_residual"] <= audit["effective_max_tolerance"]


def test_error_detection_is_explicitly_unavailable_without_both_classes() -> None:
    values = _prediction_values()
    # Every success is classified correctly, so error detection has one class.
    labels = values["success_label"]
    values["success_probability"][:] = np.where(labels > 0.5, 0.9, 0.1)
    metrics = compute_ensemble_metrics(values)
    assert metrics["success"]["error_detection"]["auroc"] is None
    assert (
        metrics["success"]["error_detection"]["status"]
        == "unavailable_single_class"
    )


def _descriptor(name: str, seed: int) -> GroupDescriptor:
    return GroupDescriptor(
        source="synthetic",
        body="piper",
        raw_body="piper",
        policy=name,
        task="move_can_pot",
        seed=seed,
        path=Path(f"{name}_{seed}.hdf5"),
    )


def test_validation_gateway_never_passes_train_or_test_to_loader() -> None:
    splits = {
        "train": [_descriptor("train", 1)],
        "validation": [_descriptor("validation", 2)],
        "test": [_descriptor("test", 3)],
    }
    observed: list[str] = []

    def spy(
        descriptors: list[GroupDescriptor], event_spec: dict[str, object]
    ) -> list[dict[str, object]]:
        observed.extend(item.logical_group for item in descriptors)
        return [{"sentinel": event_spec["sentinel"]}]

    rows = load_validation_only(splits, {"sentinel": 7}, row_loader=spy)
    assert rows == [{"sentinel": 7}]
    assert observed == [splits["validation"][0].logical_group]
    assert not ({item.logical_group for item in splits["test"]} & set(observed))


def _write_identity_only_inputs(tmp_path: Path) -> InputBinding:
    stage1 = tmp_path / "stage1"
    source_root = tmp_path / "stage1_source"
    (source_root / "data").mkdir(parents=True)
    pose_root = stage1 / "source_object_poses" / "move_can_pot" / "aloha-agilex"
    pose_root.mkdir(parents=True)
    (source_root / "seed.txt").write_text("11 12 13\n", encoding="utf-8")
    for index in range(3):
        # Identity reconstruction checks existence only; opening is a test failure.
        (source_root / "data" / f"episode{index}.hdf5").write_bytes(b"sealed")
        (pose_root / f"episode_{index:06d}.npz").write_bytes(b"sealed")
    source_manifest = stage1 / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "task": "move_can_pot",
                        "embodiment": "aloha-agilex",
                        "path": str(source_root),
                        "episodes": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_manifest = stage1 / "target.csv"
    target_manifest.write_text(
        "task,embodiment,seed,path,valid_rollout,success\n", encoding="utf-8"
    )
    tasks = (
        "adjust_bottle",
        "handover_block",
        "move_can_pot",
        "place_container_plate",
        "beat_block_hammer",
        "lift_pot",
    )
    event_spec = tmp_path / "event.json"
    event_spec.write_text(
        json.dumps({"calibration": {task: {} for task in tasks}}), encoding="utf-8"
    )
    group_root = tmp_path / "schema5" / "groups"
    group_root.mkdir(parents=True)
    groups = []
    for index, seed in enumerate((21, 22, 23)):
        path = group_root / f"group_{index}.hdf5"
        path.write_bytes(b"sealed")
        groups.append({"path": path.name, "seed": seed})
    schema_manifest = group_root.parent / "manifest.json"
    schema_manifest.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "status": "complete",
                "task": "move_can_pot",
                "body": "piper_piper_0.6",
                "model_path": "openvla",
                "event_spec_sha256": sha256_file(event_spec),
                "groups": groups,
            }
        ),
        encoding="utf-8",
    )
    return InputBinding(
        stage1_root=stage1,
        stage1_source_manifest=source_manifest,
        stage1_source_manifest_sha256=sha256_file(source_manifest),
        stage1_target_manifest=target_manifest,
        stage1_target_manifest_sha256=sha256_file(target_manifest),
        event_spec=event_spec,
        event_spec_sha256=sha256_file(event_spec),
        openvla_schema5_manifest=schema_manifest,
        openvla_schema5_manifest_sha256=sha256_file(schema_manifest),
    )


def test_split_rebinding_reads_identity_only_and_matches_all_three_shas(
    tmp_path: Path,
) -> None:
    binding = _write_identity_only_inputs(tmp_path)
    descriptors = scan_stage1_groups(binding) + scan_schema5_groups(binding)
    expected_split = strict_group_split(descriptors, split_seed=71)
    protocol = {
        **verify_input_bindings(binding),
        **split_receipt(expected_split),
        "body_alias": body_alias_receipt(descriptors),
        "body_to_id": {"aloha-agilex": 0, "piper": 1},
        "total_groups": len(descriptors),
    }
    splits, receipt = reconstruct_frozen_split(
        binding, split_seed=71, expected_protocol=protocol
    )
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    for name in ("train", "validation", "test"):
        assert receipt["split"][f"{name}_identity_sha256"] == protocol[
            f"{name}_identity_sha256"
        ]
    assert receipt["split"]["sealed_test_group_hdf5_opened"] == 0


def test_cli_cpu_synthetic_smoke_and_help_are_available() -> None:
    smoke = run_synthetic_smoke()
    assert smoke["status"] == "synthetic_smoke_passed"
    assert smoke["members"] == 5
    assert smoke["success_error_detection_status"] == "available"
    smoke_completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "evaluate_multibody_canonical_ensemble.py"),
            "--mode",
            "synthetic-smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SYNTHETIC_SMOKE=" in smoke_completed.stdout
    assert "synthetic_smoke_passed" in smoke_completed.stdout
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "evaluate_multibody_canonical_ensemble.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--training-summary-sha256" in completed.stdout
    assert "--openvla-schema5-manifest" in completed.stdout
