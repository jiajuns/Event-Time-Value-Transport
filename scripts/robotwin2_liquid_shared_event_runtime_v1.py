#!/usr/bin/env python3
"""Label-free runtime adapter for the source-only v14 Liquid-CfC ensemble."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import collect_robotwin2_five_body_ee_candidate_branches_v1 as collector
import robotwin2_liquid_shared_event_head_v1 as liquid_head
import train_robotwin2_five_body_lobo_shared_event_head_v1 as v13


FORMAT = "etsf_robotwin2_liquid_shared_event_runtime_v1"
SUPPORTED_CANDIDATE_COUNTS = (4, 8, 16)
HISTORY_LENGTH = 32
AUTHORIZED_RUNTIME_BODIES = (
    liquid_head.SOURCE_BODY,
    *liquid_head.TARGET_BODIES,
)


class LiquidRuntimeError(RuntimeError):
    """A frozen v14 runtime or canonical input failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_ensemble(
    summary_path: Path,
    expected_summary_sha256: str,
    *,
    target_body: str,
    device: torch.device | str,
) -> list[liquid_head.LiquidEffectAlignedSharedEventHead]:
    """Load one Aloha-only ensemble without accepting target fit inputs."""

    summary_path = summary_path.expanduser().resolve()
    if not summary_path.is_file() or sha256_file(summary_path) != expected_summary_sha256:
        raise LiquidRuntimeError("liquid training summary is missing or changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status")
        != "source_only_training_complete_targets_still_sealed"
        or summary.get("model_family") != liquid_head.MODEL_FAMILY
        or summary.get("source_body") != liquid_head.SOURCE_BODY
        or summary.get("sealed_target_bodies") != list(liquid_head.TARGET_BODIES)
        or summary.get("target_rows_opened") != 0
        or target_body not in AUTHORIZED_RUNTIME_BODIES
    ):
        raise LiquidRuntimeError("summary is not an authorized sealed-target v14 ensemble")
    members = summary.get("members")
    if not isinstance(members, list) or len(members) != 5:
        raise LiquidRuntimeError("liquid runtime requires exactly five members")
    selected_step = summary.get("selected_step")
    if isinstance(selected_step, bool) or not isinstance(selected_step, int):
        raise LiquidRuntimeError("liquid selected step is invalid")
    contract = summary.get("liquid_contract")
    if not isinstance(contract, Mapping):
        raise LiquidRuntimeError("liquid summary lacks its model contract")
    history_length = contract.get("state_history_length")
    if history_length != HISTORY_LENGTH:
        raise LiquidRuntimeError("liquid runtime requires the frozen K=32 history")
    expected_contract = liquid_head.checkpoint_contract(int(history_length or 0))
    if dict(contract) != expected_contract:
        raise LiquidRuntimeError("liquid summary model contract changed")
    resolved_device = torch.device(device)
    models = []
    for expected_member, item in enumerate(members):
        if not isinstance(item, Mapping):
            raise LiquidRuntimeError("liquid member descriptor is invalid")
        checkpoint_path = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        try:
            checkpoint_path.relative_to(summary_path.parent)
        except ValueError as error:
            raise LiquidRuntimeError("liquid checkpoint escapes training root") from error
        if (
            item.get("member") != expected_member
            or not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != item.get("checkpoint_sha256")
        ):
            raise LiquidRuntimeError("liquid member checkpoint identity changed")
        checkpoint = torch.load(
            checkpoint_path, map_location=resolved_device, weights_only=True
        )
        if (
            checkpoint.get("model_family") != liquid_head.MODEL_FAMILY
            or checkpoint.get("source_body") != liquid_head.SOURCE_BODY
            or checkpoint.get("sealed_target_bodies")
            != list(liquid_head.TARGET_BODIES)
            or checkpoint.get("target_rows_used") != 0
            or checkpoint.get("member") != expected_member
            or checkpoint.get("seed") != item.get("seed")
            or checkpoint.get("step") != selected_step
            or checkpoint.get("liquid_contract") != expected_contract
        ):
            raise LiquidRuntimeError("liquid checkpoint contract changed")
        model = liquid_head.LiquidEffectAlignedSharedEventHead().to(resolved_device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
    return models


def canonical_history_at_runtime(
    *,
    trajectory: np.ndarray,
    sim_times: np.ndarray,
    ee_actions: np.ndarray,
    names: Sequence[str],
    calibration: Mapping[str, Any],
    success_height_reference_z: float,
    history_length: int,
) -> dict[str, Any]:
    """Materialize the same pre-action history used by source collection."""

    if history_length != HISTORY_LENGTH:
        raise LiquidRuntimeError("runtime canonical history is frozen at K=32")
    trajectory = np.asarray(trajectory, dtype=np.float64)
    sim_times = np.asarray(sim_times, dtype=np.float64).reshape(-1)
    ee_actions = np.asarray(ee_actions, dtype=np.float32)
    if (
        trajectory.ndim != 3
        or len(trajectory) != len(sim_times)
        or ee_actions.shape != (len(sim_times), collector.NATIVE_EE_DIM)
    ):
        raise LiquidRuntimeError("runtime trajectory/time/EE history is misaligned")
    predicates, events = collector.derive_predicates_and_events(
        trajectory,
        sim_times,
        names,
        False,
        calibration,
        float(success_height_reference_z),
    )
    moving_name = str(calibration.get("moving", ""))
    if moving_name not in names:
        raise LiquidRuntimeError("runtime history lacks the calibrated moving object")
    moving_index = list(names).index(moving_name)
    history = collector.materialize_liquid_history(
        prefix=trajectory,
        prefix_times=sim_times,
        prefix_ee_actions=ee_actions,
        names=names,
        initial_moving_position=trajectory[0, moving_index, :3],
        predicates=predicates,
        events=events,
        calibration=calibration,
        history_length=history_length,
    )
    return {
        **history,
        "state": history["state_history"][-1].copy(),
        "current_event_id": int(events[-1]),
        "event_age_seconds": collector.event_age_seconds(events, sim_times),
        "current_ee": ee_actions[-1].copy(),
    }


def scoring_batch(
    *,
    canonical_history: Mapping[str, Any],
    native_candidates: np.ndarray,
    remaining_action_budget: int,
    action_exec_steps: int,
    fps: float,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Convert a policy-native candidate set into the v14 tensor ABI."""

    candidates = np.asarray(native_candidates, dtype=np.float32)
    if (
        candidates.ndim != 3
        or candidates.shape[0] not in SUPPORTED_CANDIDATE_COUNTS
        or candidates.shape[2] != collector.NATIVE_EE_DIM
        or not np.isfinite(candidates).all()
    ):
        raise LiquidRuntimeError("native candidates must be finite [N,H,16]")
    if (
        isinstance(remaining_action_budget, bool)
        or not isinstance(remaining_action_budget, int)
        or remaining_action_budget <= 0
        or isinstance(action_exec_steps, bool)
        or not isinstance(action_exec_steps, int)
        or action_exec_steps <= 0
        or not math.isfinite(fps)
        or fps <= 0.0
    ):
        raise LiquidRuntimeError("runtime action budget/timing is invalid")
    state = np.asarray(canonical_history.get("state"), dtype=np.float32)
    state_history = np.asarray(
        canonical_history.get("state_history"), dtype=np.float32
    )
    history_mask = np.asarray(
        canonical_history.get("state_history_mask"), dtype=bool
    )
    history_dt = np.asarray(
        canonical_history.get("state_history_dt"), dtype=np.float32
    )
    current_ee = np.asarray(canonical_history.get("current_ee"), dtype=np.float32)
    current_event = canonical_history.get("current_event_id")
    event_age = canonical_history.get("event_age_seconds")
    if (
        state.shape != (collector.STATE_DIM,)
        or state_history.shape != (HISTORY_LENGTH, collector.STATE_DIM)
        or history_mask.shape != state_history.shape[:1]
        or history_dt.shape != state_history.shape[:1]
        or current_ee.shape != (collector.NATIVE_EE_DIM,)
        or isinstance(current_event, bool)
        or not isinstance(current_event, (int, np.integer))
        or not 0 <= int(current_event) < len(collector.CANONICAL_EVENTS)
        or isinstance(event_age, bool)
        or not isinstance(event_age, (int, float, np.number))
        or not math.isfinite(float(event_age))
        or float(event_age) < 0.0
        or not np.isfinite(state).all()
        or not np.isfinite(state_history).all()
        or not np.isfinite(history_dt).all()
        or not np.isfinite(current_ee).all()
        or not history_mask.any()
        or not history_mask[-1]
        or np.any(history_mask[:-1] & ~history_mask[1:])
        or np.any(history_dt < 0.0)
        or np.any(history_dt[~history_mask] != 0.0)
    ):
        raise LiquidRuntimeError("canonical liquid history is invalid")
    first_valid = int(np.flatnonzero(history_mask)[0])
    if history_dt[first_valid] != 0.0:
        raise LiquidRuntimeError("canonical history first valid timestamp must be zero")
    if not np.allclose(state_history[-1], state, atol=1e-6, rtol=0.0):
        raise LiquidRuntimeError("canonical history does not end at root state")
    expected_event = np.zeros(len(collector.CANONICAL_EVENTS), dtype=np.float32)
    expected_event[int(current_event)] = 1.0
    if not np.array_equal(state[18:23], expected_event):
        raise LiquidRuntimeError("canonical state event one-hot disagrees with event id")
    effects = np.stack(
        [collector.canonical_action_chunk(current_ee, value) for value in candidates]
    ).astype(np.float32)
    candidate_count, horizon = effects.shape[:2]
    planned_steps = min(action_exec_steps, remaining_action_budget, horizon)
    action_mask = np.arange(horizon)[None] < planned_steps
    action_mask = np.repeat(action_mask, candidate_count, axis=0)
    resolved_device = torch.device(device)

    def repeated(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.repeat(value[None], candidate_count, axis=0),
            device=resolved_device,
        )

    return {
        "state": repeated(state),
        "state_history": repeated(state_history),
        "state_history_mask": repeated(history_mask),
        "state_history_dt": repeated(history_dt),
        "actions": torch.as_tensor(effects, device=resolved_device),
        "action_mask": torch.as_tensor(action_mask, device=resolved_device),
        "planned_action_dt": torch.as_tensor(
            action_mask.astype(np.float32) / float(fps), device=resolved_device
        ),
        "action_available": torch.ones(
            candidate_count, dtype=torch.bool, device=resolved_device
        ),
        "action_schema_id": torch.zeros(
            candidate_count, dtype=torch.long, device=resolved_device
        ),
        "body_id": torch.zeros(
            candidate_count, dtype=torch.long, device=resolved_device
        ),
        "dt": torch.full(
            (candidate_count,),
            planned_steps / float(fps),
            dtype=torch.float32,
            device=resolved_device,
        ),
        "current_event_id": torch.full(
            (candidate_count,),
            int(current_event),
            dtype=torch.long,
            device=resolved_device,
        ),
        "event_age_seconds": torch.full(
            (candidate_count,),
            float(event_age),
            dtype=torch.float32,
            device=resolved_device,
        ),
        "remaining_action_budget": torch.full(
            (candidate_count,),
            float(remaining_action_budget),
            dtype=torch.float32,
            device=resolved_device,
        ),
    }


@torch.no_grad()
def score_candidates(
    models: Sequence[liquid_head.LiquidEffectAlignedSharedEventHead],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if len(models) != 5:
        raise LiquidRuntimeError("liquid scoring requires five ensemble members")
    member_rank = torch.stack(
        [model(batch)["candidate_rank_logit"] for model in models]
    )
    if (
        member_rank.ndim != 2
        or member_rank.shape[0] != 5
        or member_rank.shape[1] not in SUPPORTED_CANDIDATE_COUNTS
        or not bool(torch.isfinite(member_rank).all())
    ):
        raise LiquidRuntimeError("liquid member score matrix is invalid")
    aggregate = member_rank.mean(dim=0) - float(
        v13.EPISTEMIC_RANK_RISK_WEIGHT
    ) * member_rank.std(dim=0, correction=0)
    selected = int(torch.argmax(aggregate).item())
    return {
        "format": FORMAT,
        "selected_candidate_index": selected,
        "candidate_score": aggregate.cpu().tolist(),
        "candidate_member_mean": member_rank.mean(dim=0).cpu().tolist(),
        "candidate_epistemic_std": member_rank.std(
            dim=0, correction=0
        ).cpu().tolist(),
        "member_count": 5,
        "candidate_count": int(member_rank.shape[1]),
        "target_labels_read": False,
    }


__all__ = [
    "FORMAT",
    "HISTORY_LENGTH",
    "AUTHORIZED_RUNTIME_BODIES",
    "LiquidRuntimeError",
    "SUPPORTED_CANDIDATE_COUNTS",
    "canonical_history_at_runtime",
    "load_frozen_ensemble",
    "score_candidates",
    "scoring_batch",
]
