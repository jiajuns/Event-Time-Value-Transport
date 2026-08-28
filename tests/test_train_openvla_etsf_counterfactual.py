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

from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)
from train_openvla_etsf_counterfactual import (  # noqa: E402
    CAUSAL_HISTORY_MAX_STEPS,
    GroupDescriptor,
    canonical_policy_identity,
    canonical_policy_mapping,
    class_weights,
    collate_groups,
    compute_loss,
    configure_action_rank_training,
    counterfactual_aleatoric_uncertainty,
    counterfactual_rank_score,
    counterfactual_success_logit,
    counterfactual_centered_losses,
    fit_success_temperature,
    fixed_causal_hidden_window,
    derive_regression_recovery,
    load_groups,
    load_counterfactual_pretrained_state,
    make_group_splits,
    member_validation_selection_key,
    move_batch,
    predefined_scoring_grid,
    read_group,
    read_split_manifest,
    ranking_losses,
    select_validation_scoring,
    success_pair_ranking_counts,
    tune_guard,
)


def test_fixed_causal_history_padding_future_independence_and_branch_isolation() -> None:
    first = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    second = first + np.float32(10_000)
    future_changed = first.copy()
    future_changed[4:] *= np.float32(-99)

    prefix, mask = fixed_causal_hidden_window(first[:4])
    changed_prefix, changed_mask = fixed_causal_hidden_window(future_changed[:4])
    other_branch, other_mask = fixed_causal_hidden_window(second[:4])

    assert prefix.shape == (CAUSAL_HISTORY_MAX_STEPS, 4)
    assert mask.tolist() == [True] * 4 + [False] * 4
    assert np.array_equal(prefix[:4], first[:4])
    assert np.array_equal(prefix[4:], np.zeros((4, 4), dtype=np.float32))
    assert np.array_equal(prefix, changed_prefix)
    assert np.array_equal(mask, changed_mask)
    assert np.array_equal(other_branch[:4], second[:4])
    assert np.array_equal(other_mask, mask)
    assert not np.any(other_branch[:4] == first[:4])
    long = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    truncated, truncated_mask = fixed_causal_hidden_window(long)
    assert truncated_mask.all()
    assert np.array_equal(truncated, long[-CAUSAL_HISTORY_MAX_STEPS:])


def test_single_root_history_is_encoder_bit_exact() -> None:
    torch.manual_seed(20260828)
    config = tiny_config()
    model = ActionConditionedEventWorldModel(config).eval()
    root = np.random.default_rng(7).normal(
        size=(1, config.state_input_dim)
    ).astype(np.float32)
    history, mask = fixed_causal_hidden_window(root)
    direct = model.encode_state(torch.from_numpy(root))
    causal = model.encode_state(
        torch.from_numpy(history[None]), torch.from_numpy(mask[None])
    )
    assert torch.equal(direct, causal)


def test_requested_seed_split_maps_to_resolved_logical_groups(tmp_path: Path) -> None:
    descriptors = [
        GroupDescriptor(
            path=str(tmp_path / f"group_{resolved}.hdf5"),
            schema_version=5,
            logical_key=f"move_can_pot|aloha-agilex|{resolved}",
            seed=resolved,
            requested_seed=requested,
            task="move_can_pot",
            body="aloha-agilex",
            policy="smolvla",
            metadata={},
        )
        for requested, resolved in ((100, 100), (101, 103), (102, 104))
    ]
    split_path = tmp_path / "requested_split.json"
    split_path.write_text(
        json.dumps(
            {
                "split_unit": "requested_seed_logical_group",
                "train": [100],
                "validation": [101],
                "test": [102],
            }
        ),
        encoding="utf-8",
    )

    split = read_split_manifest(split_path, descriptors)

    assert split == {
        "train": ["move_can_pot|aloha-agilex|100"],
        "validation": ["move_can_pot|aloha-agilex|103"],
        "test": ["move_can_pot|aloha-agilex|104"],
    }


def test_policy_identity_canonicalizes_openvla_checkpoint_path() -> None:
    checkpoint = (
        "/models/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot"
    )
    assert canonical_policy_identity(checkpoint) == "openvla"
    assert canonical_policy_mapping({checkpoint: 0}) == {"openvla": 0}


def test_policy_identity_rejects_alias_collision() -> None:
    with pytest.raises(RuntimeError, match="aliases collide"):
        canonical_policy_mapping({"openvla": 0, "/models/OpenVLA-OFT": 1})


def tiny_config(structured: bool = False) -> EventWorldModelConfig:
    return EventWorldModelConfig(
        state_input_dim=16,
        action_dim=4,
        proprio_dim=3,
        semantic_dim=8,
        action_hidden_dim=7,
        transition_hidden_dim=10,
        clock_hidden_dim=6,
        object_delta_dim=3,
        num_bodies=1,
        num_policies=1,
        metadata_dim=4,
        structured_events=structured,
        dropout=0.0,
    )


def test_factual_checkpoint_upgrade_and_flat_group_prediction_match_candidates() -> None:
    factual_config = tiny_config(structured=True)
    factual = ActionConditionedEventWorldModel(factual_config).eval()
    ranked_config = EventWorldModelConfig.from_dict(
        {**factual_config.to_dict(), "action_rank_residual": True}
    )
    ranked = ActionConditionedEventWorldModel(ranked_config).eval()
    load_counterfactual_pretrained_state(ranked, factual.state_dict())
    with torch.no_grad():
        ranked.action_rank_head[-1].weight.fill_(0.2)

    batch_size, candidates, horizon = 2, 3, 5
    hidden = torch.randn(batch_size, factual_config.state_input_dim)
    actions = torch.randn(
        batch_size, candidates, horizon, factual_config.action_dim
    )
    current_event = torch.tensor([0, 1])
    predicates = torch.zeros(batch_size, factual_config.num_predicates)
    vectorized = ranked.predict_candidates(
        hidden,
        actions,
        current_event_id=current_event,
        current_predicates=predicates,
    )

    flat_output = ranked(
        hidden[:, None]
        .expand(-1, candidates, -1)
        .reshape(batch_size * candidates, -1),
        actions.reshape(batch_size * candidates, horizon, -1),
        current_event_id=current_event[:, None]
        .expand(-1, candidates)
        .reshape(-1),
        current_predicates=predicates[:, None]
        .expand(-1, candidates, -1)
        .reshape(batch_size * candidates, -1),
    )
    grouped = {
        "group_index": torch.arange(batch_size).repeat_interleave(candidates),
        "baseline_mask": torch.tensor(
            [True, False, False, True, False, False]
        ),
    }
    flat_logit = counterfactual_success_logit(ranked, flat_output, grouped)
    flat_uncertainty = counterfactual_aleatoric_uncertainty(
        ranked, flat_output, grouped
    )
    assert torch.allclose(
        flat_logit.reshape(batch_size, candidates),
        vectorized["success_logit"],
        atol=1e-6,
    )
    assert torch.allclose(
        flat_uncertainty.reshape(batch_size, candidates),
        vectorized["aleatoric_uncertainty"],
        atol=1e-6,
    )


def test_frozen_core_ranker_is_low_capacity_and_exposes_only_rank_weight() -> None:
    config = EventWorldModelConfig.from_dict(
        {
            **tiny_config(structured=True).to_dict(),
            "action_rank_residual": True,
            "action_rank_success_only": True,
        }
    )
    model = ActionConditionedEventWorldModel(config)
    audit = configure_action_rank_training(model, freeze_factual_core=True)
    assert audit["trainable_parameter_names"] == ["action_rank_head.0.weight"]
    assert audit["trainable_parameter_count"] == 2 * config.semantic_dim
    assert audit["trainable_parameter_count"] <= 600
    assert audit["factual_core_trainable_parameters"] == 0
    assert all(
        parameter.requires_grad == name.startswith("action_rank_head.")
        for name, parameter in model.named_parameters()
    )


def test_ranking_gradient_only_detaches_shared_state_and_action_features() -> None:
    config = EventWorldModelConfig.from_dict(
        {
            **tiny_config(structured=True).to_dict(),
            "action_rank_residual": True,
            "action_rank_success_only": True,
        }
    )
    model = ActionConditionedEventWorldModel(config)
    configure_action_rank_training(model, freeze_factual_core=True)
    with torch.no_grad():
        model.action_rank_head[0].weight.fill_(0.1)
    success_logit = torch.randn(3, requires_grad=True)
    semantic = torch.randn(3, config.semantic_dim, requires_grad=True)
    action_effect = torch.randn(3, config.semantic_dim, requires_grad=True)
    score = counterfactual_rank_score(
        model,
        {
            "success_logit": success_logit,
            "semantic": semantic,
            "action_effect": action_effect,
        },
        {
            "group_index": torch.zeros(3, dtype=torch.long),
            "baseline_mask": torch.tensor([True, False, False]),
        },
        torch.linspace(0.0, 1.0, config.num_events),
        10.0,
        ranking_gradient_only=True,
    )
    score.sum().backward()
    assert success_logit.grad is None
    assert semantic.grad is None
    assert action_effect.grad is None
    assert model.action_rank_head[0].weight.grad is not None
    assert bool((model.action_rank_head[0].weight.grad.abs() > 0).any())


def test_success_only_rank_score_ignores_fixed_event_and_duration_utility() -> None:
    config = EventWorldModelConfig.from_dict(
        {
            **tiny_config(structured=True).to_dict(),
            "action_rank_residual": True,
            "action_rank_success_only": True,
        }
    )
    model = ActionConditionedEventWorldModel(config)
    common = {
        "success_logit": torch.tensor([-0.5, 0.2, 0.4]),
        "semantic": torch.randn(3, config.semantic_dim),
        "action_effect": torch.randn(3, config.semantic_dim),
    }
    batch = {
        "group_index": torch.zeros(3, dtype=torch.long),
        "baseline_mask": torch.tensor([True, False, False]),
    }
    first = counterfactual_rank_score(
        model,
        {
            **common,
            "next_event_logits": torch.randn(3, config.num_events),
            "duration_selected_log_mean": torch.randn(3),
        },
        batch,
        torch.linspace(0.0, 1.0, config.num_events),
        10.0,
        event_weight=1000.0,
        duration_weight=1000.0,
    )
    second = counterfactual_rank_score(
        model,
        {
            **common,
            "next_event_logits": torch.randn(3, config.num_events) * 100.0,
            "duration_selected_log_mean": torch.randn(3) * 100.0,
        },
        batch,
        torch.linspace(0.0, 1.0, config.num_events),
        10.0,
        event_weight=1000.0,
        duration_weight=1000.0,
    )
    assert torch.equal(first, second)


def write_group(path: Path, schema: int, seed: int, config: EventWorldModelConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count, horizon = 3, 5
    strings = h5py.string_dtype("utf-8")
    success = np.asarray([0, 1, seed % 2], dtype=np.bool_)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = schema
        handle.attrs["seed"] = seed
        handle.attrs["resolved_seed"] = seed
        handle.attrs["task"] = "move_can_pot"
        handle.attrs["body"] = "piper"
        handle.create_dataset("initial_hidden", data=rng.normal(size=config.state_input_dim).astype(np.float16))
        handle.create_dataset(
            "candidate_names",
            data=np.asarray(["deterministic", "candidate_1", "candidate_2"], dtype=object),
            dtype=strings,
        )
        handle.create_dataset(
            "candidate_actions",
            data=rng.normal(size=(count, horizon, config.action_dim)).astype(np.float32),
        )
        handle.create_dataset("success", data=success)
        terminal_steps = np.asarray([100, 50, 80], dtype=np.int32)
        handle.create_dataset("steps", data=terminal_steps)
        handle.create_dataset("normalized_l2_from_baseline", data=np.asarray([0, 0.2, 0.4], dtype=np.float32))
        if schema in (3, 4, 5):
            initial = handle["initial_hidden"][:]
            handle.create_dataset("pre_hidden", data=np.repeat(initial[None], count, axis=0))
            post_hidden_root = rng.normal(
                size=(count, config.state_input_dim)
            ).astype(np.float16)
            handle.create_dataset(
                "post_chunk_hidden",
                data=post_hidden_root,
            )
            handle.create_dataset("first_chunk_action_mask", data=np.ones((count, horizon), dtype=np.bool_))
            handle.create_dataset("first_chunk_executed_length", data=np.full(count, horizon, dtype=np.int32))
            handle.create_dataset("pre_event_id", data=np.zeros(count, dtype=np.int64))
            handle.create_dataset("next_event_id", data=np.asarray([0, 1, 2], dtype=np.int64))
            handle.create_dataset("duration", data=np.asarray([100, 9, 20], dtype=np.float32))
            handle.create_dataset("duration_observed", data=np.asarray([0, 1, 1], dtype=np.bool_))
            handle.create_dataset("object_names", data=np.asarray(["can"], dtype=object), dtype=strings)
            pre_pose = np.zeros((count, 1, 7), dtype=np.float32)
            pre_proprio = np.zeros((count, config.proprio_dim), dtype=np.float32)
            post_pose = pre_pose.copy()
            post_proprio = pre_proprio.copy()
            trajectories: list[tuple[np.ndarray, np.ndarray]] = []
            for candidate, terminal in enumerate(terminal_steps):
                poses = np.zeros((int(terminal) + 1, 1, 7), dtype=np.float32)
                proprio = np.zeros((int(terminal) + 1, config.proprio_dim), dtype=np.float32)
                if candidate == 1:
                    poses[:, 0, 0] = np.linspace(0.0, 1.0, len(poses))
                elif candidate == 2:
                    # Reach/stabilize near goal, remain away for four simulator
                    # states, then return and stabilize: persistent e4->e12->e4.
                    poses[1:5, 0, 0] = 0.9
                    poses[5:9, 0, 0] = 0.5
                    poses[9:, 0, 0] = 0.9
                trajectories.append((poses, proprio))
                post_pose[candidate] = poses[horizon]
                post_proprio[candidate] = proprio[horizon]
            handle.create_dataset("pre_proprio", data=pre_proprio)
            handle.create_dataset("post_proprio", data=post_proprio)
            handle.create_dataset("pre_object_poses", data=pre_pose)
            handle.create_dataset("post_object_poses", data=post_pose)
            if schema >= 4:
                branches = handle.create_group("branches")
                query_counts = []
                for candidate, (poses, proprio) in enumerate(trajectories):
                    branch = branches.create_group(f"candidate_{candidate:03d}")
                    branch.create_dataset("object_poses", data=poses)
                    branch.create_dataset("proprio", data=proprio)
                    event_names = ["e0", "eK"] if success[candidate] else ["e0"]
                    event_steps = [0, int(terminal_steps[candidate])] if success[candidate] else [0]
                    branch.create_dataset(
                        "event_names", data=np.asarray(event_names, dtype=object), dtype=strings
                    )
                    branch.create_dataset("event_steps", data=np.asarray(event_steps, dtype=np.int32))
                    if schema == 5:
                        terminal = int(terminal_steps[candidate])
                        query_steps = np.arange(0, terminal, horizon, dtype=np.int32)
                        query_post_steps = np.minimum(
                            query_steps + horizon, terminal
                        ).astype(np.int32)
                        query_count = len(query_steps)
                        query_counts.append(query_count - 1)
                        query_hidden = np.empty(
                            (query_count, config.state_input_dim), dtype=np.float16
                        )
                        query_post_hidden = rng.normal(
                            size=(query_count, config.state_input_dim)
                        ).astype(np.float16)
                        query_hidden[0] = initial
                        query_post_hidden[0] = post_hidden_root[candidate]
                        if query_count > 1:
                            query_hidden[1:] = query_post_hidden[:-1]
                        query_actions = rng.normal(
                            size=(query_count, horizon, config.action_dim)
                        ).astype(np.float32)
                        query_actions[0] = handle["candidate_actions"][candidate]
                        query_lengths = query_post_steps - query_steps
                        query_masks = (
                            np.arange(horizon)[None] < query_lengths[:, None]
                        )
                        query_masks[0] = handle["first_chunk_action_mask"][candidate]
                        branch.create_dataset("query_steps", data=query_steps)
                        branch.create_dataset("query_post_steps", data=query_post_steps)
                        branch.create_dataset("query_hidden", data=query_hidden)
                        branch.create_dataset("query_post_hidden", data=query_post_hidden)
                        branch.create_dataset("query_actions", data=query_actions)
                        branch.create_dataset("query_action_mask", data=query_masks)
                if schema == 5:
                    handle.create_dataset("queries", data=np.asarray(query_counts, dtype=np.int32))


def make_groups(tmp_path: Path, schemas: list[int]) -> tuple[list, EventWorldModelConfig]:
    config = tiny_config()
    root = tmp_path / "branches"
    for index, schema in enumerate(schemas):
        write_group(root / "groups" / f"group_{index:03d}.hdf5", schema, 100 + index, config)
    groups = load_groups(
        [root], config, ["can"], {"piper": 0}, {"openvla": 0}
    )
    return groups, config


def test_v4_derives_dynamic_predicates_regression_recovery_and_structured_loss(
    tmp_path: Path,
) -> None:
    import dataclasses

    config = tiny_config(structured=True)
    root = tmp_path / "v4"
    write_group(root / "groups" / "group.hdf5", 4, 105, config)
    calibration = {
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
    groups = load_groups(
        [root],
        config,
        ["can"],
        {"piper": 0},
        {"openvla": 0},
        calibrations=calibration,
    )
    group = groups[0]
    assert group.schema_version == 4
    assert group.structured_mask.all()
    assert group.trajectory_regress.tolist() == [False, False, True]
    assert group.trajectory_recovery.tolist() == [False, False, True]
    assert group.outcome_id[2] == config.outcome_names.index("recovery")
    assert group.post_predicates.shape == (3, config.num_predicates)

    trained_config = dataclasses.replace(config, recovery_supervised=True)
    model = ActionConditionedEventWorldModel(trained_config)
    batch = move_batch(
        collate_groups(groups, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        torch.device("cpu"),
    )
    metadata = class_weights(
        groups, trained_config, torch.device("cpu"), min_relative_support=1
    )
    loss, pieces, _ = compute_loss(
        model,
        batch,
        {},
        metadata["success_pos"],
        metadata["event"],
        torch.linspace(0, 1, trained_config.num_events),
        25.0,
        destination_class_weight=metadata["destination"],
        relative_class_weight=metadata["relative"],
        relative_supported=metadata["relative_supported"],
        predicate_pos_weight=metadata["predicate_pos"],
        outcome_class_weight=metadata["outcome"],
    )
    assert all(torch.isfinite(value) for value in pieces.values())
    assert pieces["relative"] > 0
    assert pieces["destination"] > 0
    assert pieces["predicate"] > 0
    loss.backward()
    assert model.relative_transition_head.weight.grad is not None
    assert model.post_predicate_head.weight.grad is not None


def test_put_down_without_phase_drop_is_not_regression() -> None:
    predicates = np.zeros((7, 5), dtype=np.float32)
    predicates[1:, 0] = 1  # cumulative moved keeps phase at e12
    predicates[1:3, 1] = 1  # lifted drops, but dynamic phase does not
    regress, recovery = derive_regression_recovery(
        predicates, np.asarray([0, 1, 1, 1, 1, 1, 1])
    )
    assert not regress and not recovery


def test_short_phase_jitter_is_not_regression() -> None:
    predicates = np.zeros((7, 5), dtype=np.float32)
    regress, recovery = derive_regression_recovery(
        predicates, np.asarray([0, 1, 2, 1, 2, 2, 2])
    )
    assert not regress and not recovery


def test_persistent_phase_drop_and_recovery_requires_persistent_peak() -> None:
    predicates = np.zeros((10, 5), dtype=np.float32)
    regress, recovery = derive_regression_recovery(
        predicates, np.asarray([0, 1, 2, 1, 1, 1, 2, 2, 2, 2])
    )
    assert regress and recovery

    regress, recovery = derive_regression_recovery(
        predicates[:7], np.asarray([0, 1, 2, 1, 1, 1, 2])
    )
    assert regress and not recovery


def test_persistent_phase_drop_can_recover_at_terminal_success() -> None:
    predicates = np.zeros((7, 5), dtype=np.float32)
    predicates[-1, 4] = 1
    regress, recovery = derive_regression_recovery(
        predicates, np.asarray([0, 1, 2, 1, 1, 1, 4])
    )
    assert regress and recovery


def test_v5_continuations_are_dense_auxiliary_not_rank_or_terminal_repeats(
    tmp_path: Path,
) -> None:
    config = tiny_config(structured=True)
    root = tmp_path / "v5"
    write_group(root / "groups" / "group.hdf5", 5, 107, config)
    calibration = {
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
    groups = load_groups(
        [root],
        config,
        ["can"],
        {"piper": 0},
        {"openvla": 0},
        calibrations=calibration,
    )
    group = groups[0]
    assert group.schema_version == 5 and group.continuation is not None
    assert group.history_hidden is not None and group.history_mask is not None
    assert group.post_history_hidden is not None
    assert group.post_history_mask is not None
    assert group.history_hidden.shape == (
        group.candidate_count,
        CAUSAL_HISTORY_MAX_STEPS,
        config.state_input_dim,
    )
    assert np.all(group.history_mask.sum(axis=1) == 1)
    assert np.all(group.post_history_mask.sum(axis=1) == 2)
    assert np.array_equal(group.history_hidden[:, 0], group.hidden)
    assert np.array_equal(group.post_history_hidden[:, 0], group.hidden)
    assert np.array_equal(group.post_history_hidden[:, 1], group.post_hidden)
    expected_auxiliary = sum(int(np.ceil(step / 5)) - 1 for step in [100, 50, 80])
    assert len(group.continuation["duration"]) == expected_auxiliary
    assert config.event_names.index("e4") in group.continuation["next_event_id"]
    assert config.event_names.index("eK") in group.continuation["next_event_id"]

    with_auxiliary = collate_groups(
        groups, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    )
    candidates_only = collate_groups(
        groups,
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        include_auxiliary=False,
    )
    assert int(with_auxiliary["terminal_mask"].sum()) == group.candidate_count
    assert int((with_auxiliary["group_index"] < 0).sum()) == expected_auxiliary
    assert len(candidates_only["success"]) == group.candidate_count
    assert len(with_auxiliary["success"]) == group.candidate_count + expected_auxiliary
    assert with_auxiliary["hidden_t"].shape[1:] == (
        CAUSAL_HISTORY_MAX_STEPS,
        config.state_input_dim,
    )
    assert with_auxiliary["history_mask"][: group.candidate_count].sum(1).tolist() == [
        1,
        1,
        1,
    ]
    assert with_auxiliary["post_history_mask"][: group.candidate_count].sum(1).tolist() == [
        2,
        2,
        2,
    ]

    model = ActionConditionedEventWorldModel(config).eval()
    metadata = class_weights(groups, config, torch.device("cpu"), min_relative_support=1)
    common = (
        {},
        metadata["success_pos"],
        metadata["event"],
        torch.linspace(0, 1, config.num_events),
        25.0,
    )
    extra = {
        "destination_class_weight": metadata["destination"],
        "relative_class_weight": metadata["relative"],
        "relative_supported": metadata["relative_supported"],
        "predicate_pos_weight": metadata["predicate_pos"],
        "outcome_class_weight": metadata["outcome"],
    }
    _, full_pieces, _ = compute_loss(
        model,
        move_batch(with_auxiliary, torch.device("cpu")),
        *common,
        **extra,
    )
    _, candidate_pieces, _ = compute_loss(
        model,
        move_batch(candidates_only, torch.device("cpu")),
        *common,
        **extra,
    )
    for name in (
        "success",
        "outcome",
        "pairwise",
        "listwise",
        "group_centered",
        "baseline_contrast",
    ):
        assert torch.allclose(full_pieces[name], candidate_pieces[name], atol=1e-6)
    assert not torch.allclose(full_pieces["event"], candidate_pieces["event"])


def test_v3_supersedes_v2_and_split_is_group_disjoint(tmp_path: Path) -> None:
    config = tiny_config()
    v2 = tmp_path / "v2"
    v3 = tmp_path / "v3"
    write_group(v2 / "groups" / "same.hdf5", 2, 101, config)
    write_group(v3 / "groups" / "same.hdf5", 3, 101, config)
    for seed in (102, 103):
        write_group(v2 / "groups" / f"group_{seed}.hdf5", 2, seed, config)
    groups = load_groups(
        [v2, v3], config, ["can"], {"piper": 0}, {"openvla": 0}
    )
    assert len(groups) == 3
    assert next(group for group in groups if group.seed == 101).schema_version == 3
    splits = make_group_splits(groups, seed=9, train_fraction=0.34, validation_fraction=0.33)
    assert not (set(splits["train"]) & set(splits["validation"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert set().union(*map(set, splits.values())) == {group.logical_key for group in groups}


def test_v2_only_contributes_weak_terminal_and_rank_losses(tmp_path: Path) -> None:
    groups, config = make_groups(tmp_path, [2])
    batch = move_batch(
        collate_groups(groups, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        torch.device("cpu"),
    )
    model = ActionConditionedEventWorldModel(config)
    metadata = class_weights(groups, config, torch.device("cpu"))
    weights = {name: 1.0 for name in ("success", "outcome", "pairwise", "listwise", "event", "reach", "duration", "object", "latent")}
    loss, pieces, _ = compute_loss(
        model,
        batch,
        weights,
        metadata["success_pos"],
        metadata["event"],
        torch.linspace(0, 1, config.num_events),
        25.0,
    )
    assert pieces["success"] > 0
    assert pieces["pairwise"] > 0
    assert pieces["listwise"] > 0
    for name in ("event", "reach", "duration", "object", "latent"):
        assert pieces[name].item() == 0.0
    loss.backward()
    assert model.success_head.weight.grad is not None
    assert model.next_event_head.weight.grad is None or torch.count_nonzero(
        model.next_event_head.weight.grad
    ) == 0
    assert model.duration_mean_head.weight.grad is None or torch.count_nonzero(
        model.duration_mean_head.weight.grad
    ) == 0


def test_v3_dense_multitask_loss_and_rank_have_finite_gradients(tmp_path: Path) -> None:
    groups, config = make_groups(tmp_path, [3, 3])
    object_rows = np.concatenate([group.object_delta for group in groups])
    mean = object_rows.mean(0).astype(np.float32)
    std = np.maximum(object_rows.std(0), 1e-4).astype(np.float32)
    batch = move_batch(collate_groups(groups, mean, std), torch.device("cpu"))
    model = ActionConditionedEventWorldModel(config)
    metadata = class_weights(groups, config, torch.device("cpu"))
    weights = {name: 1.0 for name in ("success", "outcome", "pairwise", "listwise", "event", "reach", "duration", "object", "latent")}
    loss, pieces, _ = compute_loss(
        model,
        batch,
        weights,
        metadata["success_pos"],
        metadata["event"],
        torch.linspace(0, 1, config.num_events),
        25.0,
    )
    assert all(torch.isfinite(value) for value in pieces.values())
    assert all(pieces[name].abs() > 0 for name in ("event", "reach", "duration", "object", "latent"))
    loss.backward()
    assert model.next_event_head.weight.grad is not None
    assert model.future_latent_mean_head.weight.grad is not None

    scores = torch.tensor([0.0, 2.0, 1.0, 0.0, 1.0, 2.0], requires_grad=True)
    pairwise, listwise, count = ranking_losses(
        scores,
        batch["success"],
        batch["group_index"],
    )
    assert count > 0 and pairwise > 0 and listwise > 0


def test_ranking_losses_use_only_success_changing_pairs() -> None:
    success = torch.tensor([0.0, 0.0, 1.0, 1.0])
    group = torch.zeros(4, dtype=torch.long)
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    swapped_within_outcome = torch.tensor([1.0, 0.0, 3.0, 2.0])

    pairwise, listwise, count = ranking_losses(scores, success, group)
    swapped_pairwise, swapped_listwise, swapped_count = ranking_losses(
        swapped_within_outcome, success, group
    )

    assert count == swapped_count == 4
    assert torch.allclose(pairwise, swapped_pairwise)
    assert torch.allclose(listwise, swapped_listwise)

    homogeneous_success = torch.ones(3)
    homogeneous_group = torch.zeros(3, dtype=torch.long)
    flat = ranking_losses(
        torch.zeros(3), homogeneous_success, homogeneous_group
    )
    spread = ranking_losses(
        torch.tensor([-2.0, 0.0, 2.0]),
        homogeneous_success,
        homogeneous_group,
    )
    assert flat[2] == spread[2] == 0
    assert flat[0].item() == spread[0].item() == 0.0
    assert flat[1] < spread[1]


def test_success_pair_metrics_separate_all_pairs_from_baseline_pairs() -> None:
    counts = success_pair_ranking_counts(
        scores=torch.tensor([0.0, 2.0, 1.0, 3.0]),
        success=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        baseline_mask=torch.tensor([True, False, False, False]),
    )
    assert counts == {
        "pure_success_correct": 3,
        "pure_success_total": 4,
        "baseline_changing_correct": 2,
        "baseline_changing_total": 2,
    }


def test_centered_and_baseline_losses_remove_group_difficulty() -> None:
    success = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    group = torch.tensor([0, 0, 0, 1, 1, 1])
    baseline = torch.tensor([True, False, False, True, False, False])
    correct = torch.tensor([0.0, 2.0, -1.0, 2.0, 0.0, 0.0])
    shifted = correct + torch.tensor([100.0, 100.0, 100.0, -50.0, -50.0, -50.0])
    reversed_score = -correct

    centered, contrast, pairs = counterfactual_centered_losses(
        correct, success, group, baseline
    )
    shifted_centered, shifted_contrast, shifted_pairs = (
        counterfactual_centered_losses(shifted, success, group, baseline)
    )
    wrong_centered, wrong_contrast, _ = counterfactual_centered_losses(
        reversed_score, success, group, baseline
    )
    assert torch.allclose(centered, shifted_centered)
    assert torch.allclose(contrast, shifted_contrast)
    assert pairs == shifted_pairs == 3
    assert centered < wrong_centered
    assert contrast < wrong_contrast


def test_member_selection_prioritizes_validation_within_group_evidence() -> None:
    common = {
        "baseline_success_rate": 0.2,
        "top1_success_rate": 0.2,
        "losses": {"event": 0.1, "total": 0.2},
    }
    weak_ranking = {
        **common,
        "pure_success_pair_lcb90": 0.1,
        "pairwise_lcb90": 0.9,
    }
    strong_ranking = {
        **common,
        "pure_success_pair_lcb90": 0.6,
        "pairwise_lcb90": 0.0,
        "losses": {"event": 5.0, "total": 10.0},
    }
    assert member_validation_selection_key(strong_ranking) > (
        member_validation_selection_key(weak_ranking)
    )


def test_temperature_calibration_and_conservative_guard() -> None:
    logits = np.asarray([[-8, 8, -8, 8], [-7, 7, -7, 7]], dtype=np.float64)
    labels = np.asarray([0, 1, 1, 0], dtype=np.float64)
    calibration = fit_success_temperature(logits, labels)
    assert 0.05 <= calibration["temperature"] <= 20.0
    assert calibration["after"]["brier"] <= calibration["before"]["brier"] + 1e-9

    rows = []
    for index in range(6):
        rows.append(
            {
                "mean_score": np.asarray([0.0, 1.0]),
                "uncertainty": np.asarray([0.1, 0.2]),
                "success": np.asarray([0.0, 1.0]),
                "baseline_index": 0,
            }
        )
    guard = tune_guard(rows, min_guarded_groups=5, minimum_lcb=0.0)
    assert guard["enabled"] is True
    assert guard["validation_policy_success_rate"] == 1.0
    unsafe = [{**row, "success": np.asarray([1.0, 0.0])} for row in rows]
    fallback = tune_guard(unsafe, min_guarded_groups=5, minimum_lcb=0.0)
    assert fallback["enabled"] is False


def test_preregistered_scoring_grid_selects_on_validation_then_tunes_guard() -> None:
    grid = predefined_scoring_grid(0.02)
    assert [candidate["candidate_id"] for candidate in grid] == [
        "success_only",
        "success_distance",
        "progress_light",
        "progress",
        "progress_clock",
        "full_light",
        "full",
    ]
    candidate_rows = []
    for candidate in grid:
        proposes = candidate["candidate_id"] == "progress"
        rows = [
            {
                "mean_score": np.asarray([0.0, 1.0]) if proposes else np.asarray([1.0, 0.0]),
                "uncertainty": np.asarray([0.1, 0.2]),
                "success": np.asarray([0.0, 1.0]),
                "baseline_index": 0,
            }
            for _ in range(12)
        ]
        candidate_rows.append((candidate, rows))
    selected, rows, audit = select_validation_scoring(
        candidate_rows,
        minimum_proposals=10,
        minimum_coverage=0.10,
        minimum_lcb=0.0,
    )
    assert selected["candidate_id"] == "progress"
    assert audit["grid_version"] == "validation_scoring_grid_v1"
    assert audit["selection_pool"] == "pre_guard_evidence_eligible"
    assert len(audit["candidates"]) == len(grid)
    assert sum(
        candidate["passes_pre_guard_evidence_gate"]
        for candidate in audit["candidates"]
    ) == 1

    guard = tune_guard(
        rows,
        min_guarded_groups=10,
        min_coverage=0.10,
        minimum_lcb=0.0,
        max_harmful_rate=0.10,
    )
    assert guard["enabled"] is True
    assert guard["guarded_groups"] >= 10
    assert guard["coverage"] >= 0.10
    assert guard["paired_success_delta_lcb90"] >= 0.0
    assert guard["harmful_rate"] <= 0.10
    assert 1 <= len(guard["threshold_candidates"]) <= 9


def test_one_step_cli_writes_member_calibration_guard_and_ensemble(tmp_path: Path) -> None:
    config = tiny_config(structured=True)
    model = ActionConditionedEventWorldModel(config)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": dataclasses_asdict(config),
            "contract": {"body_to_id": {"piper": 0}, "policy_to_id": {"openvla": 0}},
            "normalization": {
                "object_delta_mean": np.zeros(3, dtype=np.float32),
                "object_delta_std": np.ones(3, dtype=np.float32),
            },
        },
        checkpoint,
    )
    root = tmp_path / "data"
    for index in range(7):
        write_group(root / "groups" / f"group_{index:03d}.hdf5", 5, 200 + index, config)
    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "train_openvla_etsf_counterfactual.py"),
            "--data",
            str(root),
            "--pretrained",
            str(checkpoint),
            "--output",
            str(output),
            "--event-spec",
            str(event_spec),
            "--seeds",
            "11",
            "--device",
            "cpu",
            "--freeze-factual-core",
            "--amp",
            "off",
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--groups-per-batch",
            "2",
            "--num-workers",
            "0",
            "--guard-min-groups",
            "1",
            "--guard-min-lcb",
            "-1",
            "--min-relative-support",
            "1",
            "--min-recovery-support",
            "1",
            "--regression-persistence-steps",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    member = output / "members" / "seed_11" / "event_world_model_counterfactual_best.pt"
    ensemble = output / "counterfactual_ensemble.pt"
    manifest_path = output / "ensemble_manifest.json"
    assert member.is_file() and ensemble.is_file() and manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "etsf_counterfactual_ensemble_v1"
    assert len(manifest["members"]) == 1
    assert manifest["ensemble_checkpoint"]["path"] == str(ensemble.resolve())
    assert "success_calibration" in manifest and "guard" in manifest
    aggregate = torch.load(ensemble, map_location="cpu", weights_only=False)
    assert aggregate["scoring"] == manifest["scoring"]
    assert aggregate["scoring_selection"] == manifest["scoring_selection"]
    assert len(manifest["scoring_selection"]["candidates"]) == 1
    assert manifest["scoring"]["candidate_id"] == manifest["scoring_selection"][
        "selected_candidate_id"
    ]
    assert aggregate["predicate_contract"] == manifest["predicate_contract"]
    assert aggregate["candidate_contract"] == manifest["candidate_contract"]
    assert manifest["contract"]["predicate_contract"] == manifest["predicate_contract"]
    assert manifest["contract"]["candidate_contract"] == {
        "baseline_candidate_name": "deterministic",
        "fallback_index": 0,
    }
    ranking_contract = manifest["contract"]["counterfactual_ranking_contract"]
    assert ranking_contract["member_selection_data"] == (
        "validation_only_no_sealed_test"
    )
    assert ranking_contract["loss_weights"]["group_centered"] == 1.0
    assert ranking_contract["loss_weights"]["baseline_contrast"] == 1.5
    assert ranking_contract["pairwise_target"] == (
        "success_changing_candidate_pairs_only_terminal_steps_excluded"
    )
    assert ranking_contract["listwise_target"] == (
        "softmax_2x_binary_success_uniform_within_outcome_"
        "terminal_steps_excluded_normalized_by_log_candidate_count"
    )
    assert ranking_contract["candidate_cardinality"][
        "variable_candidate_count_supported"
    ] is True
    assert ranking_contract["action_sensitivity"]["architecture"] == (
        "baseline_relative_diagonal_bilinear_2d_linear_v2"
    )
    assert manifest["config"]["action_rank_residual"] is True
    assert manifest["config"]["action_rank_success_only"] is True
    assert manifest["scoring"] == {
        "candidate_id": "success_only",
        "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "event_weight": 0.0,
        "duration_weight": 0.0,
        "candidate_distance_weight": 0.0,
        "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
    }
    assert ranking_contract["validation_metrics"]["member_selection_primary"] == (
        "pure_success_pair_lcb90"
    )
    member_payload = torch.load(member, map_location="cpu", weights_only=False)
    optimization = member_payload["contract"]["action_rank_optimization"]
    assert optimization["freeze_factual_core"] is True
    assert optimization["trainable_parameter_names"] == [
        "action_rank_head.0.weight"
    ]
    assert optimization["trainable_parameter_count"] == 2 * config.semantic_dim
    for name, value in model.state_dict().items():
        assert torch.equal(member_payload["model"][name], value), name
    assert member_payload["best_selection_rule"] == ranking_contract[
        "member_selection_rule"
    ]
    assert len(member_payload["best_selection_key"]) == 4
    member_validation = member_payload["validation"]
    assert member_validation["pairwise_accuracy"] == member_validation[
        "pure_success_pair_accuracy"
    ]
    assert member_validation["pairwise_lcb90"] == member_validation[
        "pure_success_pair_lcb90"
    ]
    assert member_validation["comparable_pairs"] == member_validation[
        "pure_success_comparable_pairs"
    ]
    assert "baseline_changing_pair_accuracy" in member_validation
    assert "baseline_changing_pair_lcb90" in member_validation
    assert "baseline_changing_pairs" in member_validation
    assert manifest["predicate_contract"]["task_calibration"]["delta_move"] == 0.05
    assert manifest["predicate_contract"]["online_requires_explicit_predicates"] is True
    assert manifest["predicate_contract"]["missing_policy"] == "error"
    assert manifest["test_policy"] == (
        "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
    )
    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["regression_persistence_steps"] == 4
    assert audit["contract"]["regression_recovery_label_contract"] == {
        "phase_drop_persistence_simulator_states": 4,
        "regression": "dynamic_phase_below_pre_drop_peak_for_minimum_persistence",
        "recovery": (
            "later_at_or_above_pre_drop_peak_for_minimum_persistence_"
            "or_later_terminal_success_eK"
        ),
        "predicate_downflip_alone_is_regression": False,
    }


def test_sealed_test_with_broken_labels_is_never_loaded(tmp_path: Path) -> None:
    config = tiny_config(structured=True)
    model = ActionConditionedEventWorldModel(config)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": dataclasses_asdict(config),
            "contract": {
                "body_to_id": {"piper": 0},
                "policy_to_id": {"openvla": 0},
            },
            "normalization": {
                "object_delta_mean": np.zeros(3, dtype=np.float32),
                "object_delta_std": np.ones(3, dtype=np.float32),
            },
        },
        checkpoint,
    )
    root = tmp_path / "data"
    valid_seeds = [301, 302, 303, 304]
    for index, seed in enumerate(valid_seeds):
        write_group(
            root / "groups" / f"group_{index:03d}.hdf5", 5, seed, config
        )
    broken_seed = 399
    broken = root / "groups" / "group_999_broken_test.hdf5"
    with h5py.File(broken, "w") as handle:
        handle.attrs["schema_version"] = 5
        handle.attrs["seed"] = broken_seed
        handle.attrs["resolved_seed"] = broken_seed
        handle.attrs["task"] = "move_can_pot"
        handle.attrs["body"] = "piper"
        # A malformed label proves that strict read_group would reject it.
        handle.create_dataset("success", data=np.asarray([[1, 0]], dtype=np.int8))

    calibration = {
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
    with pytest.raises(RuntimeError, match="missing fields"):
        read_group(
            broken,
            {},
            config,
            ["can"],
            calibrations={"move_can_pot": calibration["move_can_pot"]},
        )

    event_spec = tmp_path / "event_spec.json"
    event_spec.write_text(
        json.dumps({"calibration": calibration}), encoding="utf-8"
    )
    logical = lambda seed: f"move_can_pot|piper|{seed}"
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "train": [logical(seed) for seed in valid_seeds[:3]],
                "validation": [logical(valid_seeds[3])],
                "test": [logical(broken_seed)],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "train_openvla_etsf_counterfactual.py"),
            "--data",
            str(root),
            "--pretrained",
            str(checkpoint),
            "--output",
            str(output),
            "--event-spec",
            str(event_spec),
            "--split-manifest",
            str(split_manifest),
            "--seeds",
            "17",
            "--device",
            "cpu",
            "--amp",
            "off",
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--groups-per-batch",
            "2",
            "--num-workers",
            "0",
            "--guard-min-groups",
            "1",
            "--guard-min-lcb",
            "-1",
            "--min-relative-support",
            "1",
            "--min-recovery-support",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    contract = audit["contract"]
    assert contract["sealed_test_groups"] == [logical(broken_seed)]
    assert contract["sealed_test_files"][0]["path"] == str(broken.resolve())
    assert contract["sealed_test_files"][0]["sha256"]
    assert contract["sealed_test_access"].endswith("no_label_datasets")


def dataclasses_asdict(config: EventWorldModelConfig) -> dict:
    # Keep the test independent of implementation helper methods.
    import dataclasses

    return dataclasses.asdict(config)
