#!/usr/bin/env python3
"""Cross-fitted original RGB-CfC + separate state25 value adapter TD(lambda)-AWR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPECTED_OBSERVER_SHA = "0b9638af3dd39a39d2f6fe61c6f671c08176e4316f424c1233d4a59f86bb8b89"


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("features", "labels", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite CfC critic output")

    from cfc_value_umi_s1_20260906.value import (
        fit, lambda_returns, sha256, state_histories, transitions,
    )

    with np.load(args.features, allow_pickle=False) as source:
        features = source["features"]
        states = source["states"]
        episode = source["episode"]
        times = source["time"]
        index = source["index"]
        frame = source["frame"]
        contract = json.loads(str(source["contract_json"].item()))
    if (features.shape != (9649, 111) or states.shape != (9649, 25) or
            not np.isfinite(features).all() or not np.isfinite(states).all() or
            not np.array_equal(index, np.arange(9649)) or sorted(np.unique(episode).tolist()) != list(range(52))):
        raise ValueError("invalid rl_fruit feature contract")
    if contract["format"] != "rl_fruit_original_cfc_features_v1" or contract["checkpoint_sha256"] != EXPECTED_OBSERVER_SHA:
        raise ValueError("wrong original CfC features")
    labels = json.loads(args.labels.read_text())
    outcomes = {int(row["episode_index"]): row["terminal_outcome"] for row in labels["labels"]}
    if (labels["format"] != "rl_fruit_terminal_labels_v1" or len(outcomes) != 52 or
            sorted(outcomes) != list(range(52)) or labels["conversion_mapping_sha256"] != contract["source_hashes"]["conversion_mapping.json"]):
        raise ValueError("terminal labels do not belong to these features")
    transition = transitions(episode, times, outcomes, chunk=50, rate=.1)
    histories = state_histories(states, episode, length=8, stride=3)
    args.output.mkdir(parents=True)
    oof = np.full(len(index), np.nan, np.float32)
    advantages = np.full(len(index), np.nan, np.float32)
    deltas = np.full(len(index), np.nan, np.float32)
    weights = np.ones(len(index), np.float32)
    records = []
    episode_folds = [part.tolist() for part in np.array_split(np.arange(52), 5)]
    for fold, heldout in enumerate(episode_folds):
        train_episodes = [eid for eid in range(52) if eid not in heldout]
        model, values, scale, history = fit(features, histories, episode, transition,
                                            train_episodes, args.steps, 20260909 + fold, args.device)
        heldout_mask = np.isin(episode, heldout)
        valid = heldout_mask & transition["known"]
        fold_advantage = lambda_returns(values, transition) - values
        oof[heldout_mask] = values[heldout_mask]
        advantages[valid] = fold_advantage[valid]
        td = transition["reward"] + np.where(transition["terminal"], 0,
                                              transition["gamma"] * values[transition["next"]]) - values
        deltas[valid] = td[valid]
        weights[valid] = np.clip(np.exp(np.clip(fold_advantage[valid] / (3. * scale), -5., 5.)), .9, 1.1)
        train_mask = np.isin(episode, train_episodes) & transition["known"]
        mse = float(np.mean((values[valid] - transition["mc"][valid]) ** 2))
        baseline = float(transition["mc"][train_mask].mean())
        baseline_mse = float(np.mean((baseline - transition["mc"][valid]) ** 2))
        record = dict(fold=fold, train_episodes=train_episodes, scoring_episodes=heldout,
                      advantage_scale_train_only=scale, heldout_mc_mse=mse,
                      train_constant_baseline_mse=baseline_mse, training_history=history)
        torch.save(dict(format="rl_fruit_original_cfc_state25_value_v1",
                        state_dict={key: value.cpu() for key, value in model.state_dict().items()},
                        feature_contract=contract, feature_dim=111, state_history_length=8, state_stride=3,
                        labels_sha256=sha256(args.labels), features_sha256=sha256(args.features), record=record),
                   args.output / f"value_fold{fold}.pt")
        records.append(record)
        print(json.dumps(dict(stage="fit_original_cfc_fold", **record)), flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(weights).all() or weights.std() < 1e-6:
        raise ValueError("invalid or uniform cross-fitted CfC weights")
    last = np.array([np.flatnonzero(episode == eid)[-1] for eid in np.unique(episode)])
    weights[last] = 1.
    advantages[last] = np.nan
    deltas[last] = np.nan
    final_model, _, _, history = fit(features, histories, episode, transition, list(range(52)),
                                      args.steps, 20261009, args.device)
    torch.save(dict(format="rl_fruit_original_cfc_state25_value_v1",
                    state_dict={key: value.cpu() for key, value in final_model.state_dict().items()},
                    feature_contract=contract, feature_dim=111, state_history_length=8, state_stride=3,
                    scoring_use="not used for current actor weights; weights use held-out-fold critics",
                    training_history=history), args.output / "value_all.pt")
    sidecar = args.output / "event_weights.parquet"
    pd.DataFrame(dict(index=index, episode_index=episode, frame_index=frame, event_weight=weights,
                      value_oof=oof, td_delta_oof=deltas, advantage_td_lambda=advantages,
                      mc_return=transition["mc"], reward_known=transition["known"],
                      endpoint_index=transition["next"], terminal_chunk=transition["terminal"])).to_parquet(sidecar, index=False)
    receipt = dict(format="rl_fruit_crossfit_td_lambda_awr_v1", features_sha256=sha256(args.features),
                   labels_sha256=sha256(args.labels), sidecar_sha256=sha256(sidecar),
                   observer_sha256=contract["checkpoint_sha256"], dataset_source_hashes=contract["source_hashes"],
                   frames=len(index), episodes=52, chunk=50, discount_rate_per_second=.1, td_lambda=.95,
                   terminal_reward={"success": 1., "failure": -1., "unknown": "masked; actor weight=1"},
                   outcomes={name: sum(value == name for value in outcomes.values())
                             for name in ("success", "failure", "unknown")},
                   weight_min=float(weights.min()), weight_max=float(weights.max()),
                   weight_mean=float(weights.mean()), weight_std=float(weights.std()), folds=records,
                   own_episode_critic_leakage=False, shared_rgb_cfc_frozen=True,
                   raw_states_only_in_separate_s1_adapter=True, candidate_actions_in_value=False,
                   online_rl=False, actor_initialization="right_wrist_pure050000", actor_additional_steps=10000,
                   limitations=["Terminal labels are AI-reviewed, not human gold.",
                                "All 52 terminal outcomes are successes, so this learns progress along demonstrations rather than failure discrimination.",
                                "Only logged actions are reweighted; there is no counterfactual action search or online exploration.",
                                "State25 makes this value adapter embodiment-specific; the frozen RGB-CfC shared encoder remains embodiment-independent.",
                                "Clipped low-LR AWR reduces update size but does not guarantee policy improvement."])
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(stage="original_cfc_awr_complete", frames=len(index), folds=len(records),
                          weight_min=receipt["weight_min"], weight_max=receipt["weight_max"],
                          weight_std=receipt["weight_std"])), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
