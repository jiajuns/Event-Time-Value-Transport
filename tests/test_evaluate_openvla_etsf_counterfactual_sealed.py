from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from evaluate_openvla_etsf_counterfactual_sealed import (  # noqa: E402
    binary_auc,
    binary_probability_metrics,
    classify_evaluation_protocol,
    diagonal_gaussian_joint_nll,
    equal_weight_mixture_nll,
    exact_two_sided_binomial_p,
    multiclass_prediction_metrics,
    paired_policy_metrics,
    regression_prediction_metrics,
    sha256,
    uncertainty_risk_coverage_metrics,
)
from openvla_etsf_event_world_model import ActionConditionedEventWorldModel  # noqa: E402
from openvla_etsf_event_critic_plugin import EventCriticPlugin  # noqa: E402
from test_train_openvla_etsf_counterfactual import tiny_config, write_group  # noqa: E402


def _event_spec(path: Path) -> str:
    value = {
        "calibration": {
            "move_can_pot": {
                "moving": "can",
                "anchor": "",
                "centers": [[1.0, 0.0, 0.0]],
                "offset": [0.0, 0.0, 0.0],
                "delta_move": 0.05,
                "delta_z": 0.1,
                "tau_d": 0.15,
                "tau_motion": 0.03,
                "stationary_steps": 2,
            }
        }
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256(path)


def _sealed_root(root: Path, config, seeds: list[int], event_digest: str) -> list[dict]:
    groups = root / "groups"
    groups.mkdir(parents=True)
    rows = []
    names = ["deterministic", "candidate_1", "candidate_2"]
    for index, seed in enumerate(seeds):
        path = groups / f"group_{index:03d}.hdf5"
        write_group(path, 5, seed, config)
        with h5py.File(path, "r+") as handle:
            handle.attrs["candidate_count"] = 3
            handle.attrs["intervention"] = (
                "candidate_first_chunk_then_deterministic_actor"
            )
            handle.attrs["language_contract"] = (
                "same_instruction_for_initial_query_and_all_candidate_branches"
            )
            handle.attrs["branch_instruction_consistent"] = True
            handle.attrs["post_query_action_contract"] = (
                "executed_as_next_query_when_nonterminal"
            )
        rows.append(
            {
                "index": index,
                "seed": seed,
                "resolved_seed": seed,
                "path": path.name,
                "candidate_names": names,
                "status": "collected",
            }
        )
    manifest = {
        "status": "complete",
        "schema_version": 5,
        "task": "move_can_pot",
        "body": "piper",
        "policy": "openvla",
        "candidate_count": 3,
        "seed_registry": "official_150",
        "fresh_seed_manifest": None,
        "fresh_seed_manifest_sha256": None,
        "completed": len(seeds),
        "requested_seeds": seeds,
        "resolved_seeds": seeds,
        "intervention": "candidate_first_chunk_then_deterministic_actor",
        "language_contract": (
            "same_instruction_for_initial_query_and_all_candidate_branches"
        ),
        "event_spec_sha256": event_digest,
        "event_vocab": list(config.event_names),
        "hidden_dim": config.state_input_dim,
        "action_dim": config.action_dim,
        "action_chunk": 5,
        "trajectory_contract": {
            "purpose": "dynamic_predicates_failure_and_recovery_labels"
        },
        "continuation_query_contract": {
            "post_query_action": "executed_as_next_query_when_nonterminal",
            "query_action_mask": "contiguous_executed_prefix",
            "purpose": "late_event_action_conditioned_auxiliary_transitions",
        },
        "groups": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return [
        {
            "logical_key": f"move_can_pot|piper|{seed}",
            "schema_version": 5,
            "path": str((groups / f"group_{index:03d}.hdf5").resolve()),
            "sha256": sha256(groups / f"group_{index:03d}.hdf5"),
        }
        for index, seed in enumerate(seeds)
    ]


def _ensemble_manifest(
    directory: Path,
    config,
    sealed_files: list[dict],
    event_spec: Path,
) -> Path:
    models = []
    members = []
    for seed in [1, 2]:
        torch.manual_seed(seed)
        model = ActionConditionedEventWorldModel(config)
        models.append(model.state_dict())
        member_path = directory / f"member_{seed}.pt"
        torch.save({"model": model.state_dict(), "seed": seed}, member_path)
        members.append(
            {
                "path": str(member_path.resolve()),
                "sha256": sha256(member_path),
                "seed": seed,
            }
        )
    train_key = "move_can_pot|piper|100"
    validation_key = "move_can_pot|piper|200"
    sealed_keys = [item["logical_key"] for item in sealed_files]
    event_spec_payload = json.loads(event_spec.read_text(encoding="utf-8"))
    predicate_contract = {
        "names": list(config.predicate_names),
        "derivation": "derive_atomic_predicates_v1",
        "source": "simulator_object_poses_at_query_step",
        "event_spec_sha256": sha256(event_spec),
        "task_calibration": event_spec_payload["calibration"]["move_can_pot"],
        "online_requires_explicit_predicates": True,
        "missing_policy": "error",
    }
    candidate_contract = {
        "baseline_candidate_name": "deterministic",
        "fallback_index": 0,
    }
    contract = {
        "trainer": "schema_v2_to_v5_structured_counterfactual_v3",
        "events": list(config.event_names),
        "object_names": ["can"],
        "body_to_id": {"piper": 0},
        "policy_to_id": {"openvla": 0},
        "train_groups": [train_key],
        "validation_groups": [validation_key],
        "sealed_test_groups": sealed_keys,
        "sealed_test_access": (
            "identity_attrs_and_raw_file_sha256_only_no_label_datasets"
        ),
        "sealed_test_files": sealed_files,
        "event_spec": str(event_spec.resolve()),
        "event_spec_sha256": sha256(event_spec),
        "predicate_contract": predicate_contract,
        "candidate_contract": candidate_contract,
        "group_files": [
            {
                "logical_key": train_key,
                "schema_version": 5,
                "path": "/not/read/train.hdf5",
                "sha256": "train-provenance-only",
            },
            {
                "logical_key": validation_key,
                "schema_version": 5,
                "path": "/not/read/validation.hdf5",
                "sha256": "validation-provenance-only",
            },
            *sealed_files,
        ],
    }
    normalization = {
        "object_delta_mean": [0.0] * config.object_delta_dim,
        "object_delta_std": [1.0] * config.object_delta_dim,
    }
    calibration = {"temperature": 1.5}
    guard = {
        "enabled": True,
        "gain_margin": 0.0,
        "uncertainty_threshold": 100.0,
        "coverage": 1.0,
    }
    scoring = {
        "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "event_weight": 0.25,
        "duration_weight": 0.05,
        "candidate_distance_weight": 0.02,
        "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
    }
    checkpoint = directory / "counterfactual_ensemble.pt"
    torch.save(
        {
            "format": "etsf_counterfactual_ensemble_v1",
            "models": models,
            "member_seeds": [1, 2],
            "config": dataclasses.asdict(config),
            "contract": contract,
            "predicate_contract": predicate_contract,
            "candidate_contract": candidate_contract,
            "normalization": normalization,
            "duration_scale": 20.0,
            "success_calibration": calibration,
            "guard": guard,
            "scoring": scoring,
        },
        checkpoint,
    )
    manifest = {
        "format": "etsf_counterfactual_ensemble_v1",
        "ensemble_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        },
        "members": members,
        "config": dataclasses.asdict(config),
        "contract": contract,
        "predicate_contract": predicate_contract,
        "candidate_contract": candidate_contract,
        "normalization": normalization,
        "duration_scale": 20.0,
        "success_calibration": calibration,
        "guard": guard,
        "scoring": scoring,
        "test_policy": (
            "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
        ),
    }
    path = directory / "ensemble_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_exact_paired_statistics() -> None:
    assert exact_two_sided_binomial_p(0, 0) == 1.0
    assert exact_two_sided_binomial_p(3, 0) == 0.25
    metrics = paired_policy_metrics(
        np.asarray([0, 1, 0, 1]),
        np.asarray([1, 1, 0, 0]),
        bootstrap_seed=1,
    )
    assert metrics["improved_groups"] == 1
    assert metrics["harmed_groups"] == 1
    assert metrics["mcnemar_exact_two_sided_p"] == 1.0


def test_prediction_metric_helpers_report_support_and_calibration() -> None:
    multiclass = multiclass_prediction_metrics(
        np.asarray([0, 0, 1, 2]),
        np.asarray([0, 1, 1, 1]),
        ["stay", "advance", "regress"],
    )
    assert multiclass["count"] == 4
    assert multiclass["per_class"]["regress"]["support"] == 1
    assert multiclass["per_class"]["regress"]["predicted"] == 0
    assert multiclass["macro_f1_supported"] < 1.0
    assert len(multiclass["confusion_matrix_rows_true_columns_predicted"]) == 3
    binary = binary_probability_metrics(
        np.asarray([0.0, 1.0, 1.0]), np.asarray([0.1, 0.8, 0.6])
    )
    assert binary["positive_support"] == 2
    assert binary["auc"] == 1.0
    assert binary["average_precision"] == 1.0
    assert binary["brier"] < 0.1
    regression = regression_prediction_metrics(
        np.asarray([1.0, 3.0]), np.asarray([2.0, 3.0])
    )
    assert regression["mae"] == 0.5


def test_auc_ap_and_aurc_are_tie_aware_without_pairwise_matrix() -> None:
    labels = np.asarray([1.0, 0.0, 1.0, 0.0])
    tied = np.asarray([0.5, 0.5, 0.5, 0.5])
    assert binary_auc(labels, tied) == 0.5
    binary = binary_probability_metrics(labels, tied)
    assert binary["average_precision"] == 0.5
    risk = uncertainty_risk_coverage_metrics(labels, tied, tied)
    assert risk["aurc"] == 0.5
    assert risk["risk_at_coverage"]["0.25"] == 0.5
    assert risk["tie_policy"] == "expected_random_order_within_equal_uncertainty"

    # This would require a multi-gigabyte positive-by-negative matrix in the
    # old implementation; the rank implementation remains linear-memory.
    many_labels = np.tile(np.asarray([0.0, 1.0]), 25_000)
    many_scores = np.arange(len(many_labels), dtype=np.float64)
    assert 0.0 <= float(binary_auc(many_labels, many_scores)) <= 1.0


def test_equal_weight_gaussian_mixture_forms_joint_density_before_mixing() -> None:
    target = torch.zeros((2, 2), dtype=torch.float64)
    mean = torch.asarray([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float64)
    log_scale = torch.zeros_like(mean)
    member_joint = diagonal_gaussian_joint_nll(target, mean, log_scale)
    mixture = equal_weight_mixture_nll(member_joint)
    expected = -torch.log(
        0.5 * torch.exp(-member_joint[0]) + 0.5 * torch.exp(-member_joint[1])
    )
    assert torch.allclose(mixture, expected)
    # Mixing per-dimension averages first is a different, invalid joint model.
    wrong = equal_weight_mixture_nll(member_joint / 2.0)
    assert not torch.allclose(mixture / 2.0, wrong)


def test_fresh_protocol_requires_exact_frozen_50_seed_contract() -> None:
    requested = list(range(1000, 1050))
    resolved = list(range(2000, 2050))
    digest = "a" * 64
    fresh = {
        "status": "fresh_confirmation_preregistered_resolved",
        "task": "move_can_pot",
        "test": [
            {"requested_seed": request, "resolved_seed": resolution}
            for request, resolution in zip(requested, resolved)
        ],
    }
    root = {
        "task": "move_can_pot",
        "seed_registry": "explicit_fresh_confirmation",
        "requested_seeds": requested,
        "resolved_seeds": resolved,
        "fresh_seed_manifest": "/frozen/fresh50.json",
        "fresh_seed_manifest_sha256": digest,
    }
    protocol = classify_evaluation_protocol(root, fresh, digest)
    assert protocol["confirmatory"] is True
    assert protocol["evidence_tier"] == "fresh_confirmatory"
    root["fresh_seed_manifest_sha256"] = "tampered"
    try:
        classify_evaluation_protocol(root, fresh, digest)
    except RuntimeError as error:
        assert "SHA256" in str(error)
    else:
        raise AssertionError("tampered fresh manifest binding was accepted")


def test_structured_manifest_predicate_mirror_is_frozen(tmp_path: Path) -> None:
    config = tiny_config(structured=True)
    event_spec = tmp_path / "event_spec.json"
    _event_spec(event_spec)
    manifest_path = _ensemble_manifest(tmp_path, config, [], event_spec)
    plugin = EventCriticPlugin.from_manifest(manifest_path)
    assert plugin.predicate_contract["derivation"] == "derive_atomic_predicates_v1"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predicate_contract"]["derivation"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="predicate_contract mirror mismatch"):
        EventCriticPlugin.from_manifest(manifest_path)


def test_sealed_evaluator_is_atomic_frozen_and_one_shot(tmp_path: Path) -> None:
    config = tiny_config(structured=True)
    event_spec = tmp_path / "event_spec.json"
    event_digest = _event_spec(event_spec)
    sealed_root = tmp_path / "sealed"
    sealed_files = _sealed_root(sealed_root, config, [300, 301], event_digest)
    ensemble_manifest = _ensemble_manifest(
        tmp_path, config, sealed_files, event_spec
    )
    output = tmp_path / "evaluation"
    command = [
        sys.executable,
        str(SCRIPTS / "evaluate_openvla_etsf_counterfactual_sealed.py"),
        "--ensemble-manifest",
        str(ensemble_manifest),
        "--sealed-data",
        str(sealed_root),
        "--event-spec",
        str(event_spec),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "SEALED_EVALUATION_COMPLETE=" in completed.stdout
    result_path = output / "evaluated_once.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["evaluated_once"] is True
    assert result["split_audit"]["overlap_checks_passed"] is True
    assert result["collection_audit"]["schema_version"] == 5
    assert result["evaluation_protocol"]["evidence_tier"] == "development_holdout"
    assert result["evaluation_protocol"]["confirmatory"] is False
    assert result["frozen_deployment"]["override_flags_available"] is False
    assert result["metrics"]["guarded_selected"]["groups"] == 2
    assert "mcnemar_exact_two_sided_p" in result["metrics"]["guarded_selected"]
    assert "fallback_reason_histogram" in result["metrics"]["guard"]
    assert result["prediction_metrics"]["initial_candidates"]["query_count"] == 6
    initial_prediction = result["prediction_metrics"]["initial_candidates"]
    assert initial_prediction["eligible_counts"]["structured"] == 6
    assert initial_prediction["eligible_counts"]["dense"] == 6
    assert 0 <= initial_prediction["eligible_counts"]["duration_observed"] <= 6
    assert initial_prediction["post_event"]["mixture_nll"] is not None
    assert initial_prediction["reach_probability"]["mixture_log_loss"] is not None
    assert (
        initial_prediction["object_xyz_delta_m"]
        ["normalized_gaussian_mixture_joint_nll"]
        is not None
    )
    assert (
        initial_prediction["future_latent_cosine"]
        ["mean_member_normalized_gaussian_nll_per_dimension"]
        is not None
    )
    assert "terminal_initial_candidates_only" in result["prediction_metrics"]["mask_contract"]
    assert result["prediction_metrics"]["continuation_query_count"] > 0
    assert (
        result["prediction_metrics"]["all_query_transitions"]["post_event"]["count"]
        > result["prediction_metrics"]["initial_candidates"]["post_event"]["count"]
    )
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing overwrite" in second.stderr
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "complete"
