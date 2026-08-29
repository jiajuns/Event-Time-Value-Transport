from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_multibody_canonical_event_world_model as core  # noqa: E402
import robotwin2_move_can_pot_analytic_event_spec_v1 as analytic_event  # noqa: E402
import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as prereg  # noqa: E402
import run_robotwin2_five_body_lobo_offline_ablation_v1 as ablation  # noqa: E402
import verify_robotwin2_move_can_pot_public_materialization_v1 as verifier  # noqa: E402
from train_robotwin2_five_body_lobo_shared_event_head_v1 import (  # noqa: E402
    _effect_aligned_loss,
    ABLATION_VARIANTS,
    ACTOR_FORMAT,
    BINDING_FORMAT,
    BODIES,
    CANONICAL_ACTION_SCHEMA,
    CANONICAL_STATE_SCHEMA,
    CANDIDATE_NOISE_CONTRACT,
    DATASET_REPO,
    DATASET_REVISION,
    EVENT_SPEC_SHA256,
    CANDIDATE_RANK_FEATURE_DIM,
    DENSE_FAILURE_RANK_WEIGHT,
    BRANCH_DIAGNOSTIC_CONTRACT,
    EffectAlignedSharedEventHead,
    MANIFEST_FORMAT,
    MATERIALIZATION_FORMAT,
    OBJECT_EFFECT_SCHEMA,
    PREREGISTRATION_SHA256,
    STANDARDIZED_RANK_ENSEMBLE_CONTRACT,
    TERMINAL_SUPERVISION_CONTRACT,
    SOURCE_EVENT_SAMPLING_HZ,
    TASK,
    FiveBodyContractError,
    ablation_contract,
    ablation_selection_components,
    aggregate_standardized_rank_scores,
    build_preflight_receipt,
    canonical_sha256,
    checkpoint_candidate_rank_contract,
    candidate_checkpoint_selection_key,
    effect_preserving_group_bootstrap_weights,
    evaluate_candidate_ranking,
    load_binding,
    materialize_source_rows,
    sha256_file,
    sha256_tree,
    source_group_split,
    summary_candidate_rank_contract,
)


def _signed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["logical_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha256_file(path)


def _group(path: Path, offset: float) -> None:
    count, horizon = 4, 5
    state = np.full((count, core.STATE_DIM), offset, dtype=np.float32)
    state[:, 18:27] = 0.0
    state[:, 18] = 1.0
    terminal_goal_progress = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    terminal_goal_distance = (
        np.linalg.norm(state[:, 0:3], axis=-1) - terminal_goal_progress
    ).astype(np.float32)
    np.savez(
        path,
        state=state,
        actions=np.full((count, horizon, core.ACTION_DIM), offset, dtype=np.float32),
        action_mask=np.ones((count, horizon), dtype=np.float32),
        current_event_id=np.zeros(count, dtype=np.int64),
        post_event_id=np.asarray([1, 2, 3, 4], dtype=np.int64),
        post_event_mask=np.ones(count, dtype=np.float32),
        next_event_id=np.asarray([1, 2, 3, 4], dtype=np.int64),
        next_event_mask=np.ones(count, dtype=np.float32),
        duration=np.asarray([0, 2, 3, 4], dtype=np.float32),
        duration_observed=np.asarray([0, 1, 1, 1], dtype=np.float32),
        duration_mask=np.asarray([0, 1, 1, 1], dtype=np.float32),
        success=np.asarray([0, 1, 0, 1], dtype=np.float32),
        success_mask=np.ones(count, dtype=np.float32),
        recovery=np.asarray([0, 0, 1, 0], dtype=np.float32),
        recovery_mask=np.ones(count, dtype=np.float32),
        object_delta=np.full((count, core.OBJECT_DELTA_DIM), 0.1, dtype=np.float32),
        object_delta_mask=np.ones(count, dtype=np.float32),
        terminal_max_event_id=np.asarray([1, 4, 3, 4], dtype=np.int64),
        terminal_stage_progress=np.asarray([0.25, 1.0, 0.75, 1.0], dtype=np.float32),
        terminal_goal_distance=terminal_goal_distance,
        terminal_goal_progress=terminal_goal_progress,
        candidate_index=np.arange(count, dtype=np.int64),
        dt=np.full(count, 5.0 / SOURCE_EVENT_SAMPLING_HZ, dtype=np.float32),
    )


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    files = [
        {
            "path": f"dataset/move_can_pot/file-{index}.zip",
            "size_bytes": index + 1,
            "expected_payload_sha256": f"{index + 1:064x}",
            "observed_payload_sha256": f"{index + 1:064x}",
            "size_match": True,
            "payload_sha256_match": True,
            "zip_central_directory_audit": {
                "central_directory_read_only": True,
                "member_payload_bytes_read": 0,
            },
        }
        for index in range(11)
    ]
    materialization_unsigned = {
        "format": MATERIALIZATION_FORMAT,
        "status": verifier.STATUS,
        "materialized": True,
        "no_missing_or_extra_official_task_files": True,
        "all_exact_sizes_verified": True,
        "all_exact_archive_payload_sha256_verified": True,
        "hf_repo_id": DATASET_REPO,
        "hf_repo_revision": DATASET_REVISION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "official_file_count": 11,
        "official_total_size_bytes": sum(range(1, 12)),
        "files": files,
        "read_boundary": {
            "zip_member_payload_bytes_read": 0,
            "archive_extracted": False,
            "pickle_payload_opened_or_deserialized": False,
            "numpy_payload_opened_or_deserialized": False,
            "torch_payload_opened_or_deserialized": False,
        },
        "implementation_binding": {
            "verifier_module": Path(verifier.__file__).name,
            "verifier_file_sha256": verifier._file_sha256(Path(verifier.__file__).resolve()),
            "preregistration_module": Path(prereg.__file__).name,
            "preregistration_module_file_sha256": verifier._file_sha256(
                Path(prereg.__file__).resolve()
            ),
        },
        "authority": {
            "download_completeness_attested": True,
            "training_authorized": False,
            "evaluation_authorized": False,
            "simulator_execution_authorized": False,
            "checkpoint_selection_or_promotion_authorized": False,
            "deployment_authorized": False,
            "cross_embodiment_performance_claim_authorized": False,
        },
    }
    materialization = {
        **materialization_unsigned,
        "materialization_receipt_sha256": verifier.canonical_sha256(
            materialization_unsigned
        ),
    }
    materialization_path = tmp_path / "materialization.json"
    materialization_sha = _write_json(materialization_path, materialization)

    actors = {}
    for index, body in enumerate(BODIES):
        checkpoint_path = tmp_path / f"{body}-actor.ckpt"
        if body == "piper":
            checkpoint_path.mkdir()
            (checkpoint_path / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint_path / "model.safetensors").write_bytes(b"frozen-piper")
            checkpoint_kind = "directory_tree"
            checkpoint_sha256 = sha256_tree(checkpoint_path)[0]
        else:
            checkpoint_path.write_bytes(f"frozen-{body}".encode())
            checkpoint_kind = "file"
            checkpoint_sha256 = sha256_file(checkpoint_path)
        actors[body] = {
            "family": "synthetic-test-native-actor",
            "frozen": True,
            "optimizer_updates_allowed": False,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_sha256": checkpoint_sha256,
            "sampling_contract_sha256": f"{index + 40:064x}",
            "candidate_count": 4,
            "candidate_zero_is_actor_baseline": True,
            "same_ordered_candidate_set_for_baseline_and_etsf": True,
        }
    actor_authority = _signed(
        {
            "format": ACTOR_FORMAT,
            "task": TASK,
            "actors": actors,
        }
    )
    actor_path = tmp_path / "actors.json"
    actor_sha = _write_json(actor_path, actor_authority)

    body_bindings = {}
    for body_index, body in enumerate(BODIES):
        groups = []
        for condition in ("clean", "randomized"):
            for group_index in range(2):
                name = f"{body}-{condition}-{group_index}.npz"
                group_path = tmp_path / name
                _group(group_path, float(body_index + group_index + 1))
                groups.append(
                    {
                        "group_id": name.removesuffix(".npz"),
                        "condition": condition,
                        "requested_seed": body_index * 100 + group_index,
                        "path": name,
                        "sha256": sha256_file(group_path),
                        "diagnostic_format": BRANCH_DIAGNOSTIC_CONTRACT["format"],
                        "diagnostics_path": name.replace(".npz", ".diagnostics.npz"),
                        "diagnostics_sha256": "d" * 64,
                    }
                )
        manifest = _signed(
            {
                "format": MANIFEST_FORMAT,
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "task": TASK,
                "body": body,
                "schema_adapter": {
                    "kind": "analytic_label_free_canonical_v1",
                    "trainable": False,
                    "labels_or_outcomes_used_to_fit": False,
                    "heldout_supervision_allowed": False,
                    "state_dim": core.STATE_DIM,
                    "action_dim": core.ACTION_DIM,
                    "state_schema": CANONICAL_STATE_SCHEMA,
                    "action_schema": CANONICAL_ACTION_SCHEMA,
                    "elapsed_time_unit": "seconds",
                    "duration_unit": "seconds",
                    "event_names": list(core.CANONICAL_EVENTS),
                    "implementation_sha256": f"{body_index + 60:064x}",
                },
                "event_spec_sha256": EVENT_SPEC_SHA256,
                "analytic_event_contract": analytic_event.event_contract(
                    {
                        "moving": "can",
                        "anchor": "pot",
                        "required_objects": list(analytic_event.REQUIRED_OBJECTS),
                        "goal_rule": dict(analytic_event.GOAL_RULE),
                        "thresholds": dict(analytic_event.THRESHOLDS),
                        "event_rules": dict(analytic_event.EVENT_RULES),
                    }
                ),
                "event_derivation_implementation_sha256": "7" * 64,
                "state27_relative_goal_contract": (
                    "same_analytic_initial_side_pot_relative_goal_vector_used_for_"
                    "event_labels_and_online_state27_channels_0_2"
                ),
                "physical_time_contract": {
                    "source": "counted_successful_sapien_scene_step_calls",
                    "simulator_timestep_source": "scene.get_timestep",
                    "policy_action_call_count_used_as_time": False,
                    "wall_clock_used_as_time": False,
                    "dt_semantics": "planned_first_candidate_chunk_seconds",
                    "planned_action_steps": 5,
                    "actor_control_hz": SOURCE_EVENT_SAMPLING_HZ,
                    "planned_dt_seconds": 5.0 / SOURCE_EVENT_SAMPLING_HZ,
                    "duration_semantics": "simulator_elapsed_seconds_to_event_boundary",
                    "zero_elapsed_duration_masked": True,
                    "stationary_source_sampling_hz": SOURCE_EVENT_SAMPLING_HZ,
                    "stationary_window_seconds": analytic_event.THRESHOLDS[
                        "stationary_window_seconds"
                    ],
                    "stationary_speed_threshold_m_per_s": analytic_event.THRESHOLDS[
                        "stationary_speed_m_per_s"
                    ],
                },
                "candidate_action_contract": {
                    "critic_observation_time": "before_candidate_execution",
                    "planned_action_horizon": 5,
                    "action_mask_source": "planned_first_chunk_not_executed_count",
                    "executed_action_count_used_for_action_mask": False,
                    "executed_action_count_used_for_sim_time_accounting_only": True,
                    "zero_step_infeasible_candidate_keeps_failure_and_action_binding": True,
                },
                "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
                "object_effect_schema": OBJECT_EFFECT_SCHEMA,
                "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
                "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
                "groups": groups,
            }
        )
        manifest_path = tmp_path / f"{body}-manifest.json"
        body_bindings[body] = {
            "path": manifest_path.name,
            "sha256": _write_json(manifest_path, manifest),
        }

    binding = _signed(
        {
            "format": BINDING_FORMAT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "event_spec_sha256": EVENT_SPEC_SHA256,
            "candidate_noise_contract": CANDIDATE_NOISE_CONTRACT,
            "terminal_supervision_contract": TERMINAL_SUPERVISION_CONTRACT,
            "object_effect_schema": OBJECT_EFFECT_SCHEMA,
            "branch_diagnostic_contract": BRANCH_DIAGNOSTIC_CONTRACT,
            "heldout_labels_may_train_fit_calibrate_or_select": False,
            "canonical_shared_body_rows": 1,
            "execution_authority": {
                "explicit_user_training_request_recorded": True,
                "public_data_only": True,
                "protected_internal_data_allowed": False,
                "remote_cuda_only": True,
            },
            "materialization_receipt": {
                "path": materialization_path.name,
                "sha256": materialization_sha,
            },
            "actor_authority": {"path": actor_path.name, "sha256": actor_sha},
            "body_manifests": body_bindings,
        }
    )
    binding_path = tmp_path / "binding.json"
    return binding_path, _write_json(binding_path, binding)


def test_preflight_is_five_fold_source_only_and_payload_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("preflight opened a transition NPZ")
    ))
    for heldout in BODIES:
        receipt = build_preflight_receipt(audit, held_out_body=heldout, split_seed=17)
        assert heldout not in receipt["source_bodies"]
        assert len(receipt["source_bodies"]) == 4
        assert receipt["heldout_group_npz_opened"] == 0
        assert receipt["heldout_specific_trainable_parameters"] == 0
        assert receipt["model_body_rows"] == 1
        assert receipt["actor_frozen"] is True
        assert receipt["split_unit"] == "body_condition_requested_seed_all_queries"


def test_source_split_keeps_all_queries_from_one_seed_in_one_lane() -> None:
    manifests = {}
    for body in BODIES:
        groups = []
        for condition in ("clean", "randomized"):
            for seed in (101, 102, 103, 104, 105):
                for query in (0, 10, 20, 30):
                    groups.append(
                        {
                            "condition": condition,
                            "requested_seed": seed,
                            "group_id": f"{condition}-seed{seed}-query{query}",
                        }
                    )
        manifests[body] = {"groups": groups}
    train, validation, _heldout = source_group_split(
        {"manifests": manifests}, held_out_body="franka", split_seed=19
    )
    for body in BODIES:
        if body == "franka":
            continue
        for condition in ("clean", "randomized"):
            train_seeds = {
                row["requested_seed"]
                for row in train
                if row["body"] == body and row["condition"] == condition
            }
            validation_seeds = {
                row["requested_seed"]
                for row in validation
                if row["body"] == body and row["condition"] == condition
            }
            assert train_seeds and validation_seeds
            assert train_seeds.isdisjoint(validation_seeds)
            for seed in train_seeds | validation_seeds:
                train_count = sum(
                    row["requested_seed"] == seed
                    and row["body"] == body
                    and row["condition"] == condition
                    for row in train
                )
                validation_count = sum(
                    row["requested_seed"] == seed
                    and row["body"] == body
                    and row["condition"] == condition
                    for row in validation
                )
                assert (train_count, validation_count) in {(4, 0), (0, 4)}


def test_heldout_payload_is_not_stat_hashed_or_deserialized_in_preflight(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    manifest_path = tmp_path / decoded["body_manifests"]["franka"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in manifest["groups"]:
        (tmp_path / group["path"]).unlink()
    # A strict Franka-heldout preflight consumes only the declared paths and
    # commitments.  Missing target payloads therefore cannot be observed.
    audit = load_binding(binding, digest)
    receipt = build_preflight_receipt(
        audit, held_out_body="franka", split_seed=23
    )
    assert receipt["heldout_group_payload_bytes_read"] == 0
    assert receipt["heldout_group_payload_deserialized"] == 0

    # The same missing files fail once Franka becomes a source body and its
    # source payload boundary is deliberately crossed.
    train, _, _ = source_group_split(audit, held_out_body="piper", split_seed=23)
    franka = [group for group in train if group["body"] == "franka"]
    with pytest.raises(FiveBodyContractError, match="missing/tampered"):
        materialize_source_rows(franka, held_out_body="piper")


def test_source_materialization_never_accepts_heldout_and_model_has_one_body_row(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    train, validation, heldout = source_group_split(
        audit, held_out_body="franka", split_seed=19
    )
    assert all(row["body"] != "franka" for row in train + validation)
    assert all("body" not in row for row in heldout)
    rows = materialize_source_rows(train, held_out_body="franka")
    assert rows and all(row["body"] != "franka" for row in rows)
    mapping = {body: 0 for body in BODIES if body != "franka"}
    dataset = core.TransitionDataset(rows, mapping)
    batch = core.collate_rows([dataset[0], dataset[1]])
    model = core.MultibodyCanonicalEventWorldModel(
        core.ModelConfig(body_count=1, action_schema_count=1, dropout=0.0)
    ).eval()
    assert model.clock.body_beta.weight.shape[0] == 1
    assert model.action.schema_count == 1
    assert set(batch["body_id"].tolist()) == {0}
    with torch.no_grad():
        output = model(batch)
    assert output["success_logit"].shape == (2,)
    with pytest.raises(FiveBodyContractError, match="held-out group"):
        materialize_source_rows(
            [{**audit["manifests"]["franka"]["groups"][0], "body": "franka"}],
            held_out_body="franka",
        )


def test_tampered_or_supervised_adapter_fails_closed(tmp_path: Path) -> None:
    binding, digest = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    manifest_binding = decoded["body_manifests"]["piper"]
    manifest_path = tmp_path / manifest_binding["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_adapter"]["labels_or_outcomes_used_to_fit"] = True
    unsigned = dict(manifest)
    unsigned.pop("logical_sha256")
    manifest["logical_sha256"] = canonical_sha256(unsigned)
    manifest_binding["sha256"] = _write_json(manifest_path, manifest)
    binding_unsigned = dict(decoded)
    binding_unsigned.pop("logical_sha256")
    decoded["logical_sha256"] = canonical_sha256(binding_unsigned)
    digest = _write_json(binding, decoded)
    with pytest.raises(FiveBodyContractError, match="analytic/label-free"):
        load_binding(binding, digest)


def test_incomplete_public_download_receipt_fails_closed(tmp_path: Path) -> None:
    binding, _ = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    receipt_path = tmp_path / decoded["materialization_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "download_incomplete"
    unsigned = dict(receipt)
    unsigned.pop("materialization_receipt_sha256")
    receipt["materialization_receipt_sha256"] = verifier.canonical_sha256(unsigned)
    decoded["materialization_receipt"]["sha256"] = _write_json(receipt_path, receipt)
    unsigned_binding = dict(decoded)
    unsigned_binding.pop("logical_sha256")
    decoded["logical_sha256"] = canonical_sha256(unsigned_binding)
    digest = _write_json(binding, decoded)
    with pytest.raises(FiveBodyContractError, match="not verified"):
        load_binding(binding, digest)


def _model_batch(dt: torch.Tensor) -> dict[str, torch.Tensor]:
    count = len(dt)
    state = torch.zeros(count, core.STATE_DIM)
    state[:, 18] = 1.0
    return {
        "state": state,
        "actions": torch.randn(count, 5, core.ACTION_DIM),
        "action_mask": torch.ones(count, 5, dtype=torch.bool),
        "action_available": torch.ones(count, dtype=torch.bool),
        "action_schema_id": torch.zeros(count, dtype=torch.long),
        "body_id": torch.zeros(count, dtype=torch.long),
        "dt": dt,
        "current_event_id": torch.zeros(count, dtype=torch.long),
    }


def test_rank_gradient_updates_effect_backbone_but_not_clock_or_duration_heads() -> None:
    torch.manual_seed(7)
    model = EffectAlignedSharedEventHead().eval()
    output = model(_model_batch(torch.full((4,), 5.0 / 15.0)))
    output["candidate_rank_logit"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.semantic.parameters())
    assert any(parameter.grad is not None for parameter in model.action.parameters())
    assert any(parameter.grad is not None for parameter in model.transition.parameters())
    assert all(parameter.grad is None for parameter in model.clock.parameters())
    assert all(parameter.grad is None for parameter in model.duration_mean.parameters())
    assert all(parameter.grad is None for parameter in model.duration_scale.parameters())
    assert all(parameter.grad is None for parameter in model.post_event.parameters())
    assert all(parameter.grad is None for parameter in model.success.parameters())


def test_rank_score_has_explicit_numeric_dt_path_through_clock() -> None:
    model = EffectAlignedSharedEventHead().eval()
    linear = torch.nn.Linear(CANDIDATE_RANK_FEATURE_DIM, 1, bias=False)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, core.SEMANTIC_DIM] = 1.0
        model.clock.body_beta.weight.zero_()
        model.clock.base_tau.weight.zero_()
        model.clock.base_tau.bias.zero_()
        model.clock.candidate.weight.zero_()
        model.clock.candidate.bias.zero_()
        model.clock.candidate.bias[0] = 1.0
    model.candidate_rank = linear
    batch = _model_batch(torch.tensor([1.0 / 15.0, 5.0 / 15.0]))
    batch["state"][1] = batch["state"][0]
    batch["actions"][1] = batch["actions"][0]
    output = model(batch)
    assert output["clock_hidden"][0, 0] != output["clock_hidden"][1, 0]
    assert output["candidate_rank_logit"][0] != output["candidate_rank_logit"][1]


def test_rank_ensemble_standardizes_each_member_within_one_decision() -> None:
    scores = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [7.0, 7.0, 7.0, 7.0],
            [0.0, 100.0, 0.0, 0.0],
        ]
    )
    assert int(scores.mean(0).argmax()) == 1
    aggregate = aggregate_standardized_rank_scores(scores)
    assert aggregate.shape == (4,)
    assert int(aggregate.argmax()) == 0
    assert STANDARDIZED_RANK_ENSEMBLE_CONTRACT["population_std_correction"] == 0
    assert STANDARDIZED_RANK_ENSEMBLE_CONTRACT[
        "member_with_std_at_or_below_floor"
    ] == "all_zero_contribution"
    constant_only = aggregate_standardized_rank_scores(torch.ones(5, 4))
    assert torch.equal(constant_only, torch.zeros(4))


def _effect_loss(
    *,
    success: list[float],
    terminal_event: list[int],
    terminal_goal_progress: list[float],
    scores: list[float],
    variant: str = "full",
) -> dict[str, torch.Tensor]:
    score = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    output = {
        "candidate_rank_logit": score,
        "success_logit": torch.zeros(4, requires_grad=True),
        "object_delta_mean": torch.zeros(4, core.OBJECT_DELTA_DIM, requires_grad=True),
        "object_delta_log_scale": torch.zeros(
            4, core.OBJECT_DELTA_DIM, requires_grad=True
        ),
    }
    batch = {
        "success": torch.tensor(success),
        "state": torch.zeros(4, core.STATE_DIM),
        "post_event_id": torch.zeros(4, dtype=torch.long),
        "terminal_max_event_id": torch.tensor(terminal_event, dtype=torch.long),
        "terminal_goal_progress": torch.tensor(terminal_goal_progress),
        "object_delta": torch.zeros(4, core.OBJECT_DELTA_DIM),
        "object_delta_mask": torch.ones(4),
        "action_available": torch.ones(4),
        "logical_group": ["piper|clean|one"] * 4,
    }
    _total, pieces = _effect_aligned_loss(
        output,
        batch,
        torch.ones(4),
        ablation_variant=variant,
    )
    return pieces


def test_mixed_success_uses_group_listwise_success_probability_mass() -> None:
    good = _effect_loss(
        success=[0, 1, 0, 0],
        terminal_event=[4, 0, 3, 2],
        terminal_goal_progress=[1000.0, -1000.0, 10.0, 5.0],
        scores=[0.0, 5.0, 0.0, 0.0],
    )
    bad = _effect_loss(
        success=[0, 1, 0, 0],
        terminal_event=[4, 0, 3, 2],
        terminal_goal_progress=[1000.0, -1000.0, 10.0, 5.0],
        scores=[5.0, 0.0, 0.0, 0.0],
    )
    assert good["group_listwise_success_mass"] < bad[
        "group_listwise_success_mass"
    ]
    assert good["all_failure_dense_listwise"] == 0.0


def test_all_failure_dense_target_is_true_lexicographic_terminal_value() -> None:
    # Candidate 0 has an extreme geometric value, but candidate 1 reached the
    # later terminal event.  No fixed 100/10 scalar can reverse that ordering.
    good = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 2, 1, 1],
        terminal_goal_progress=[1000.0, -1000.0, 0.0, 0.0],
        scores=[0.0, 5.0, 0.0, 0.0],
    )
    bad = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 2, 1, 1],
        terminal_goal_progress=[1000.0, -1000.0, 0.0, 0.0],
        scores=[5.0, 0.0, 0.0, 0.0],
    )
    assert good["all_failure_dense_listwise"] < bad["all_failure_dense_listwise"]
    assert torch.allclose(
        good["candidate_ranking"],
        DENSE_FAILURE_RANK_WEIGHT * good["all_failure_dense_listwise"],
    )
    no_object = _effect_loss(
        success=[0, 0, 0, 0],
        terminal_event=[1, 1, 1, 1],
        terminal_goal_progress=[0.0, 1000.0, 2.0, 1.0],
        scores=[0.0, 5.0, 0.0, 0.0],
        variant="no_object_effect",
    )
    assert no_object["all_failure_dense_listwise"] == 0.0


def test_effect_bootstrap_repairs_missing_mixed_success_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "logical_group": "piper|clean|mixed",
            "success": float(index == 1),
            "success_mask": 1.0,
        }
        for index in range(4)
    ]
    rows += [
        {
            "logical_group": "piper|clean|failure",
            "success": 0.0,
            "success_mask": 1.0,
        }
        for _index in range(4)
    ]
    monkeypatch.setattr(
        core,
        "logical_group_bootstrap_weights",
        lambda groups, *, members, seed: np.zeros((members, len(groups)), np.float32),
    )
    weights, audit = effect_preserving_group_bootstrap_weights(
        rows, members=5, seed=17
    )
    assert weights.shape == (5, 8)
    assert all(item["positive_rows_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["negative_rows_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["mixed_success_groups_with_nonzero_weight"] > 0 for item in audit)
    assert all(item["deterministic_mixed_group_repairs"] == 1 for item in audit)


class _FixedRankModel(torch.nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.ablation_variant = variant

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        values = torch.tensor([0.0, 3.0, 2.0, 1.0], device=batch["candidate_index"].device)
        return {"candidate_rank_logit": values[batch["candidate_index"].long()]}


def _ranking_rows() -> list[dict[str, object]]:
    rows = []
    for group, success, terminal in (
        ("piper|clean|mixed", [0, 1, 0, 0], [4, 0, 3, 2]),
        ("piper|clean|failure", [0, 0, 0, 0], [0, 3, 2, 1]),
    ):
        for candidate in range(4):
            rows.append(
                {
                    "logical_group": group,
                    "body": "piper",
                    "candidate_index": np.int64(candidate),
                    "success": np.float32(success[candidate]),
                    "terminal_max_event_id": np.int64(terminal[candidate]),
                    "terminal_stage_progress": np.float32(
                        1.0 if success[candidate] else terminal[candidate] / 4.0
                    ),
                    "terminal_goal_distance": np.float32(1.0 - candidate / 10.0),
                    "terminal_goal_progress": np.float32(candidate / 10.0),
                }
            )
    return rows


def test_ranking_evaluation_separates_success_change_from_dense_progress() -> None:
    loader = torch.utils.data.DataLoader(
        core.TransitionDataset(_ranking_rows(), {"piper": 0}),
        batch_size=8,
        shuffle=False,
        collate_fn=core.collate_rows,
    )
    result = evaluate_candidate_ranking(
        _FixedRankModel("full"), loader, torch.device("cpu")
    )
    assert result["mixed_success_decisions"] == 1
    assert result["mixed_success_selection_accuracy"] == 1.0
    assert result["mixed_success_pairwise_accuracy"] == 1.0
    assert result["dense_progress_decisions"] == 1
    assert result["dense_progress_selection_accuracy"] == 1.0
    assert result["dense_progress_pairwise_accuracy"] == 1.0
    success_only = evaluate_candidate_ranking(
        _FixedRankModel("success_only"), loader, torch.device("cpu")
    )
    assert success_only["dense_progress_decisions"] == 0
    assert success_only["dense_progress_pairwise_accuracy"] is None


def test_checkpoint_selection_prefers_mixed_success_before_dense_diagnostics() -> None:
    base = {
        "macro_delta_success_rate": 0.1,
        "macro_mixed_success_pairwise_accuracy": 0.8,
        "macro_dense_progress_selection_accuracy": 0.5,
        "macro_dense_progress_pairwise_accuracy": 0.5,
    }
    stronger_mixed = {
        **base,
        "macro_mixed_success_selection_accuracy": 0.8,
    }
    stronger_dense_only = {
        **base,
        "macro_mixed_success_selection_accuracy": 0.7,
        "macro_dense_progress_selection_accuracy": 1.0,
        "macro_dense_progress_pairwise_accuracy": 1.0,
    }
    assert candidate_checkpoint_selection_key(
        stronger_mixed, 1.0, 100
    ) < candidate_checkpoint_selection_key(stronger_dense_only, 0.1, 100)


def test_ablation_variants_change_only_declared_score_features() -> None:
    batch = _model_batch(torch.full((4,), 5.0 / 15.0))
    success_only = EffectAlignedSharedEventHead("success_only").eval()(batch)
    assert torch.equal(
        success_only["candidate_rank_logit"], success_only["success_logit"]
    )
    no_time = EffectAlignedSharedEventHead("no_time_duration").eval()(batch)
    assert torch.count_nonzero(
        no_time["candidate_rank_features"][:, core.SEMANTIC_DIM:]
    ) == 0
    full = EffectAlignedSharedEventHead("full").eval()(batch)
    assert torch.count_nonzero(
        full["candidate_rank_features"][:, core.SEMANTIC_DIM:]
    ) > 0
    assert set(ABLATION_VARIANTS) == {
        "success_only", "no_time_duration", "no_object_effect", "full"
    }
    assert ablation_contract("no_object_effect")[
        "object_effect_loss_and_rank_target_enabled"
    ] is False
    components = {
        "post_event_macro_error_ratio": 1.0,
        "next_event_macro_error_ratio": 1.0,
        "observed_duration_mae_ratio": 1.0,
        "success_brier_ratio": 1.0,
        "object_rmse_ratio": 1.0,
    }
    assert set(ablation_selection_components(components, "success_only")) == {
        "success_brier_ratio"
    }
    assert "observed_duration_mae_ratio" not in ablation_selection_components(
        components, "no_time_duration"
    )
    assert "object_rmse_ratio" not in ablation_selection_components(
        components, "no_object_effect"
    )
    assert checkpoint_candidate_rank_contract("no_time_duration")[
        "dt_has_numeric_score_path"
    ] is False
    assert summary_candidate_rank_contract("success_only")[
        "pairwise_rank_loss_enabled"
    ] is False
    assert summary_candidate_rank_contract("full")[
        "group_listwise_success_mass_loss_enabled"
    ] is True
    assert summary_candidate_rank_contract("full")[
        "dt_has_numeric_score_path"
    ] is True


def _complete_ablation_audit() -> dict[str, object]:
    manifests = {}
    for body in BODIES:
        groups = []
        for condition in ablation.trainer.CONDITIONS:
            for query in ablation.QUERY_INDICES:
                for ordinal in range(ablation.SEEDS_PER_CONDITION_QUERY):
                    groups.append(
                        {
                            "condition": condition,
                            "root_query_index": query,
                            "requested_seed": 2026081000 + ordinal,
                        }
                    )
        manifests[body] = {"groups": groups}
    return {"manifests": manifests}


def _ablation_validation_metrics(offset: float) -> dict[str, object]:
    return {
        "candidate_ranking": {
            "macro_delta_success_rate": 0.1 + offset,
            "macro_selected_success_rate": 0.5 + offset,
            "macro_oracle_success_rate": 0.8,
            "pairwise_accuracy": 0.6 + offset,
        },
        "success_brier": 0.2,
        "success_auroc": 0.7,
        "post_event": {"macro_f1": 0.5, "accuracy": 0.6},
        "next_event": {"macro_f1": 0.4, "accuracy": 0.5},
        "observed_duration_mae": 0.3,
        "observed_duration_nll": 0.4,
        "object_rmse": 0.05,
        "object_nll": 0.1,
    }


def _ablation_fold_summary(body: str, variant: str, offset: float) -> dict[str, object]:
    trainer_sha = sha256_file(Path(ablation.trainer.__file__).resolve())
    return {
        "status": "source_only_checkpoint_selection_complete",
        "held_out_body": body,
        "source_bodies": [item for item in BODIES if item != body],
        "ablation": ablation_contract(variant),
        "candidate_rank_contract": summary_candidate_rank_contract(variant),
        "trainer_file_sha256": trainer_sha,
        "ensemble_checkpoint_selection": {
            "common_step_required_for_all_five_members": True,
            "rank_aggregation": ablation.trainer.standardized_rank_ensemble_contract(),
            "selected_step": 3000,
            "heldout_rows_used": 0,
        },
        "training_budget": {
            "steps_per_member": 3000,
            "eval_every_steps": 100,
            "batch_size_rows": 64,
            "learning_rate": 3e-4,
            "ensemble_members": 5,
        },
        "heldout_labels_used_for_normalization_training_or_selection": False,
        "heldout_group_npz_opened": 0,
        "preflight": {"split_unit": "body_condition_requested_seed_all_queries"},
        "members": [
            {
                "member": member,
                "seed": seed,
                "best_step": 3000,
                "trainer_file_sha256": trainer_sha,
                "source_validation": _ablation_validation_metrics(offset),
            }
            for member, seed in enumerate(ablation.ENSEMBLE_SEEDS)
        ],
    }


def test_ablation_entry_requires_exact_full_8000_branches() -> None:
    audit = _complete_ablation_audit()
    receipt = ablation.validate_complete_inventory(audit)
    assert receipt["decisions"] == 2000
    assert receipt["branches"] == 8000
    audit["manifests"]["piper"]["groups"].pop()
    with pytest.raises(ablation.AblationError, match="exactly 400"):
        ablation.validate_complete_inventory(audit)


def test_ablation_entry_freezes_same_budget_for_all_20_runs(tmp_path: Path) -> None:
    commands = [
        ablation.fold_command(
            python_executable="python3",
            binding=tmp_path / "binding.json",
            binding_sha256="a" * 64,
            output=tmp_path / variant / body,
            held_out_body=body,
            variant=variant,
        )
        for variant in ablation.VARIANTS
        for body in BODIES
    ]
    assert len(commands) == 20
    for command in commands:
        assert command[command.index("--steps") + 1] == "3000"
        assert command[command.index("--eval-every") + 1] == "100"
        assert command[command.index("--batch-size") + 1] == "64"
        assert command[command.index("--split-seed") + 1] == "20260901"
        assert command[command.index("--ensemble-seeds") + 1 :] == [
            str(seed) for seed in ablation.ENSEMBLE_SEEDS
        ]


def test_ablation_entry_reports_every_fold_macro_and_prediction_metric() -> None:
    summaries = {
        variant: {
            body: _ablation_fold_summary(body, variant, 0.01 * variant_index)
            for body in BODIES
        }
        for variant_index, variant in enumerate(ablation.VARIANTS)
    }
    result = ablation.aggregate_variants(summaries)
    for variant in ablation.VARIANTS:
        assert len(result[variant]["folds"]) == 5
        assert set(result[variant]["equal_fold_macro"]) == set(ablation.METRICS)
        assert result[variant]["equal_fold_macro"]["oracle_success_rate"] == 0.8
    assert result["full"]["equal_fold_macro"][
        "best_of_4_delta_success_rate"
    ] > result["success_only"]["equal_fold_macro"][
        "best_of_4_delta_success_rate"
    ]
    heldout = {
        variant: {
            body: {
                "held_out_body": body,
                "metrics": {
                    name: 0.1 + 0.01 * variant_index
                    for name in ablation.POSTHOC_ENSEMBLE_METRICS
                },
                "uncertainty_risk_coverage": {
                    endpoint: {
                        "support": 10,
                        "error_kind": f"{endpoint}_error",
                        "uncertainty_kind": f"{endpoint}_uncertainty",
                        "aurc": 0.2,
                        "full_coverage_risk": 0.3,
                        "error_uncertainty_spearman": 0.4,
                        "risk_at_coverage": [
                            {
                                "coverage": coverage,
                                "retained": max(1, int(10 * coverage)),
                                "risk": 0.1 + coverage,
                            }
                            for coverage in ablation.RISK_COVERAGE_LEVELS
                        ],
                    }
                    for endpoint in (
                        "rank_selected_failure",
                        "rank_oracle_regret",
                        "success",
                        "post_event",
                        "next_event",
                        "duration",
                        "object",
                        "recovery",
                    )
                },
            }
            for body in BODIES
        }
        for variant_index, variant in enumerate(ablation.VARIANTS)
    }
    heldout_result = ablation.aggregate_posthoc_heldout(heldout)
    for variant in ablation.VARIANTS:
        assert len(heldout_result[variant]["folds"]) == 5
        assert set(heldout_result[variant]["equal_fold_macro"]) == set(
            ablation.POSTHOC_ENSEMBLE_METRICS
        )


def test_ablation_posthoc_heldout_uses_frozen_five_member_rank_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    variant = "full"
    heldout = "franka"
    for ordinal, group in enumerate(audit["manifests"][heldout]["groups"]):
        group["group_id"] = (
            f"{group['condition']}|seed={group['requested_seed']}|query={ordinal}"
        )
    summary = _ablation_fold_summary(heldout, variant, 0.0)
    for member, item in enumerate(summary["members"]):
        model = EffectAlignedSharedEventHead(variant).eval()
        checkpoint_path = tmp_path / f"ablation-member-{member}.pt"
        torch.save(
            {
                "format": ablation.trainer.FORMAT,
                "model": model.state_dict(),
                "member": member,
                "seed": ablation.ENSEMBLE_SEEDS[member],
                "held_out_body": heldout,
                "ablation": ablation_contract(variant),
                "candidate_rank_contract": checkpoint_candidate_rank_contract(
                    variant
                ),
                    "heldout_rows_used_for_training_normalization_or_selection": 0,
                    "trainer_file_sha256": sha256_file(
                        Path(ablation.trainer.__file__).resolve()
                    ),
                    "ensemble_common_selection_step": 3000,
            },
            checkpoint_path,
        )
        item["checkpoint"] = str(checkpoint_path)
        item["checkpoint_sha256"] = sha256_file(checkpoint_path)
    monkeypatch.setattr(ablation, "DECISIONS_PER_BODY", 4)
    result = ablation.evaluate_posthoc_heldout_fold(
        summary,
        audit,
        held_out_body=heldout,
        variant=variant,
        device=torch.device("cpu"),
    )
    assert result["heldout_decisions"] == 4
    assert result["heldout_branches"] == 16
    assert result["heldout_labels_used_for_training_checkpoint_or_variant_selection"] is False
    assert set(result["metrics"]) == set(ablation.POSTHOC_ENSEMBLE_METRICS)
    assert result["candidate_metric_aggregation"] == (
        ablation.trainer.STANDARDIZED_RANK_ENSEMBLE_CONTRACT
    )
    assert result["prediction_metric_aggregation"] == (
        "five_frozen_members_mixed_in_probability_or_density_space_then_scored"
    )
    assert result["prediction_support"]["complete_four_candidate_decisions"] == 4
    assert "rank_oracle_regret" in result["uncertainty_risk_coverage"]
