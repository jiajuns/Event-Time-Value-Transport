#!/usr/bin/env python3
"""Matched WCM-style future-latent baseline for RoboTwin2 branch ranking.

This module is intentionally independent of the ETSF shared-event head.  It
uses the same canonical 27-D causal state and 14-D candidate action sequence,
but learns an action-conditioned prediction of a finite-horizon consequence
latent.  It is a matched scientific baseline inspired by WCM/LeWM; it is not
an official WCM architecture or weight reproduction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import robotwin2_actor_execution_protocol_v1 as actor_execution
import robotwin2_move_can_pot_analytic_event_spec_v2 as analytic_event


FORMAT = "etsf_robotwin2_wcm_future_latent_baseline_v1"
CHECKPOINT_FORMAT = "etsf_robotwin2_wcm_future_latent_checkpoint_v1"
MODEL_FAMILY = "matched_wcm_style_terminal_consequence_latent_v1"
STATE_SCHEMA = "dual_ee_object_relative_state_27d_v2"
ACTION_SCHEMA = "dual_ee_se3_gripper_delta_14d_v2"
STATE_DIM = 27
ACTION_DIM = 14
EVENT_COUNT = 5
OBJECT_EFFECT_DIM = 6
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CANDIDATE_COUNTS = (4, 8)
STANDARDIZED_INPUT_CLIP = 5.0
GOAL_PROGRESS_SCALE_METERS = 0.02
OBJECT_EFFECT_SCALES = (
    0.02,
    0.02,
    0.02,
    0.25,
    0.25,
    0.25,
)
V13_TRAINABLE_PARAMETER_REFERENCE = 223_287
EPISTEMIC_RISK_WEIGHT = 0.25
EVENT_SPEC_SHA256 = analytic_event.EVENT_SPEC_SHA256
STATE_ACTION_FRAME_CONTRACT = {
    "format": "etsf_robotwin2_native_ee16_state_action_frame_v2",
    "training_state_source": "public_hdf5_endpose_left_right_endpose",
    "runtime_state_api": "task.get_arm_pose(left/right)",
    "runtime_state_pose_semantics": "robot.get_*_ee_pose(is_endpose=False)",
    "native_action_pose_semantics": (
        "same_absolute_world_ee_frame_as_training_endpose"
    ),
    "environment_call": "task.take_action(native_ee16, action_type=ee)",
    "pose_convention": "xyz_plus_quaternion_wxyz",
    "tcp_tool_axis_offset_m_excluded": 0.12,
    "state_and_action_same_frame": True,
}
FUTURE_TARGET_SCHEMA = (
    "terminal_event_e0",
    "terminal_event_e12",
    "terminal_event_e3",
    "terminal_event_e4",
    "terminal_event_eK",
    "terminal_success",
    "terminal_stage_progress",
    "bounded_terminal_goal_progress",
    "bounded_object_effect_dx",
    "bounded_object_effect_dy",
    "bounded_object_effect_dz",
    "bounded_object_effect_drx",
    "bounded_object_effect_dry",
    "bounded_object_effect_drz",
)
FUTURE_TARGET_DIM = len(FUTURE_TARGET_SCHEMA)
VALUE_DIM = 2
RANK_SCORE_CONTRACT = {
    "format": "etsf_wcm_style_success_value_rank_v1",
    "inputs": [
        "predicted_terminal_success_probability",
        "predicted_terminal_stage_progress",
        "predicted_bounded_terminal_goal_progress",
    ],
    "score": (
        "success_probability_plus_0.25_times_clamped_stage_mean_plus_"
        "0.05_times_tanh_goal_mean"
    ),
    "ensemble": "member_mean_minus_0.25_times_member_population_std",
    "candidate_counts": list(CANDIDATE_COUNTS),
    "candidate_outcomes_or_labels_read_at_inference": False,
}


class WCMBaselineError(RuntimeError):
    """A model, loss, checkpoint, or candidate-pool contract failed closed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WCMBaselineError(f"checkpoint is missing or symbolic: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class WCMConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    event_count: int = EVENT_COUNT
    object_effect_dim: int = OBJECT_EFFECT_DIM
    latent_dim: int = 96
    hidden_dim: int = 128
    action_hidden_dim: int = 96
    residual_depth: int = 3
    dropout: float = 0.0
    sigreg_knots: int = 17
    sigreg_projections: int = 256
    sigreg_seed: int = 20260913

    def validate(self) -> None:
        if (
            self.state_dim != STATE_DIM
            or self.action_dim != ACTION_DIM
            or self.event_count != EVENT_COUNT
            or self.object_effect_dim != OBJECT_EFFECT_DIM
            or self.latent_dim < 16
            or self.hidden_dim < 16
            or self.action_hidden_dim < 16
            or self.residual_depth < 1
            or not 0.0 <= self.dropout < 1.0
            or self.sigreg_knots < 3
            or self.sigreg_projections < 8
        ):
            raise WCMBaselineError("WCM baseline configuration is invalid")


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class SketchedIsotropicGaussianRegularizer(torch.nn.Module):
    """Characteristic-function SIGReg over one already-global row batch.

    The Epps--Pulley sketch follows the public WCM/LeWM formulation: random
    unit projections reduce the latent distribution to one-dimensional
    characteristic functions, which are compared with N(0,1).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        knots: int = 17,
        num_projections: int = 256,
        seed: int = 20260913,
    ) -> None:
        super().__init__()
        if latent_dim < 2 or knots < 3 or num_projections < 8:
            raise WCMBaselineError("SIGReg dimensions are invalid")
        points = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        delta = 3.0 / float(knots - 1)
        trapezoid = torch.full((knots,), 2.0 * delta, dtype=torch.float32)
        trapezoid[[0, -1]] = delta
        gaussian_cf = torch.exp(-0.5 * points.square())
        generator = torch.Generator().manual_seed(seed)
        projections = torch.randn(
            latent_dim,
            num_projections,
            generator=generator,
            dtype=torch.float32,
        )
        projections = F.normalize(projections, dim=0)
        self.num_projections = num_projections
        self.register_buffer("points", points)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", trapezoid * gaussian_cf)
        self.register_buffer("projections", projections)

    def forward(
        self,
        latent: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.projections.shape[0]:
            raise WCMBaselineError("SIGReg latent must be [B,D]")
        if latent.shape[0] < 2 or not bool(torch.isfinite(latent).all()):
            raise WCMBaselineError("SIGReg needs at least two finite rows")
        if sample_weight is None:
            normalized_weight = latent.new_full(
                (latent.shape[0],), 1.0 / float(latent.shape[0])
            )
            active_rows = latent.shape[0]
        else:
            if (
                sample_weight.shape != (latent.shape[0],)
                or not bool(torch.isfinite(sample_weight).all())
                or bool((sample_weight < 0).any())
            ):
                raise WCMBaselineError("SIGReg sample weights are invalid")
            denominator = sample_weight.sum()
            active_rows = int((sample_weight > 0).sum().item())
            if float(denominator.detach()) <= 0.0:
                return latent.sum() * 0.0
            if active_rows < 2:
                return latent.sum() * 0.0
            normalized_weight = sample_weight.to(latent) / denominator.to(latent)
        projected = (latent @ self.projections.to(latent)).unsqueeze(-1)
        arguments = projected * self.points.to(latent)
        sketch_weight = normalized_weight[:, None, None]
        real_error = (arguments.cos() * sketch_weight).sum(dim=0) - self.gaussian_cf.to(
            latent
        )
        imaginary_error = (arguments.sin() * sketch_weight).sum(dim=0)
        statistic = (
            (real_error.square() + imaginary_error.square())
            @ self.weights.to(latent)
        ) * active_rows
        return statistic.mean()


def variance_covariance_regularizer(
    latent: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    target_std: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg-style batch variance/covariance protection and diagnostics."""

    if latent.ndim != 2 or latent.shape[0] < 2:
        raise WCMBaselineError("variance/covariance loss needs [B>=2,D]")
    if sample_weight is None:
        normalized_weight = latent.new_full(
            (latent.shape[0],), 1.0 / float(latent.shape[0])
        )
    else:
        if (
            sample_weight.shape != (latent.shape[0],)
            or not bool(torch.isfinite(sample_weight).all())
            or bool((sample_weight < 0).any())
        ):
            raise WCMBaselineError("variance/covariance sample weights are invalid")
        denominator = sample_weight.sum()
        if float(denominator.detach()) <= 0.0:
            zero = latent.sum() * 0.0
            return zero, {
                "variance": zero,
                "covariance": zero,
                "minimum_std": zero,
                "mean_std": zero,
            }
        if int((sample_weight > 0).sum().item()) < 2:
            zero = latent.sum() * 0.0
            return zero, {
                "variance": zero,
                "covariance": zero,
                "minimum_std": zero,
                "mean_std": zero,
            }
        normalized_weight = sample_weight.to(latent) / denominator.to(latent)
    mean = (latent * normalized_weight[:, None]).sum(dim=0, keepdim=True)
    centered = latent - mean
    variance = (centered.square() * normalized_weight[:, None]).sum(dim=0)
    std = torch.sqrt(variance + epsilon)
    variance_loss = F.relu(target_std - std).mean()
    covariance = centered.T @ (centered * normalized_weight[:, None])
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / float(latent.shape[1])
    return variance_loss + covariance_loss, {
        "variance": variance_loss,
        "covariance": covariance_loss,
        "minimum_std": std.min(),
        "mean_std": std.mean(),
    }


class WCMFutureLatentBaseline(torch.nn.Module):
    """Causal context + candidate action -> finite-horizon consequence latent."""

    def __init__(self, config: WCMConfig | None = None) -> None:
        super().__init__()
        self.config = config or WCMConfig()
        self.config.validate()
        dim = self.config.latent_dim
        hidden = self.config.hidden_dim
        self.register_buffer("state_mean", torch.zeros(STATE_DIM))
        self.register_buffer("state_std", torch.ones(STATE_DIM))
        self.register_buffer("action_mean", torch.zeros(ACTION_DIM))
        self.register_buffer("action_std", torch.ones(ACTION_DIM))
        self.register_buffer(
            "object_effect_scales", torch.tensor(OBJECT_EFFECT_SCALES)
        )
        self.state_history_encoder = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM + 3, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, dim),
            torch.nn.LayerNorm(dim),
        )
        self.action_token_encoder = torch.nn.Sequential(
            torch.nn.Linear(ACTION_DIM, self.config.action_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.config.action_hidden_dim, dim),
        )
        self.action_sequence_encoder = torch.nn.GRU(
            input_size=dim,
            hidden_size=dim,
            batch_first=True,
        )
        self.context_action_fusion = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, dim),
            torch.nn.LayerNorm(dim),
        )
        self.dynamics_blocks = torch.nn.ModuleList(
            [
                ResidualMLPBlock(dim, hidden, self.config.dropout)
                for _ in range(self.config.residual_depth)
            ]
        )
        self.future_delta = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim),
        )
        torch.nn.init.normal_(self.future_delta[-1].weight, std=1e-3)
        torch.nn.init.zeros_(self.future_delta[-1].bias)
        self.future_target_encoder = torch.nn.Sequential(
            torch.nn.Linear(FUTURE_TARGET_DIM, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, dim),
            torch.nn.LayerNorm(dim),
        )
        self.terminal_event_head = torch.nn.Linear(dim, EVENT_COUNT)
        self.success_head = torch.nn.Linear(dim, 1)
        self.value_head = torch.nn.Linear(dim, 2 * VALUE_DIM)
        self.object_effect_head = torch.nn.Linear(dim, 2 * OBJECT_EFFECT_DIM)
        self.sigreg = SketchedIsotropicGaussianRegularizer(
            latent_dim=dim,
            knots=self.config.sigreg_knots,
            num_projections=self.config.sigreg_projections,
            seed=self.config.sigreg_seed,
        )

    @torch.no_grad()
    def set_input_normalization(
        self,
        *,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        expected = {
            "state_mean": (state_mean, (STATE_DIM,)),
            "state_std": (state_std, (STATE_DIM,)),
            "action_mean": (action_mean, (ACTION_DIM,)),
            "action_std": (action_std, (ACTION_DIM,)),
        }
        for name, (value, shape) in expected.items():
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise WCMBaselineError(f"{name} normalization is invalid")
        if bool((state_std < 1e-4).any()) or bool((action_std < 1e-4).any()):
            raise WCMBaselineError("normalization std is below the floor")
        self.state_mean.copy_(state_mean.to(self.state_mean))
        self.state_std.copy_(state_std.to(self.state_std))
        self.action_mean.copy_(action_mean.to(self.action_mean))
        self.action_std.copy_(action_std.to(self.action_std))

    def _validate_inputs(self, batch: Mapping[str, Any]) -> None:
        state = batch.get("state")
        actions = batch.get("actions")
        mask = batch.get("action_mask")
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim != 2
            or state.shape[-1] != STATE_DIM
            or not isinstance(actions, torch.Tensor)
            or actions.ndim != 3
            or actions.shape[0] != state.shape[0]
            or actions.shape[-1] != ACTION_DIM
            or not isinstance(mask, torch.Tensor)
            or mask.shape != actions.shape[:2]
            or not bool(torch.isfinite(state).all())
            or not bool(torch.isfinite(actions).all())
        ):
            raise WCMBaselineError("canonical state/action batch shape changed")
        boolean_mask = mask.bool()
        lengths = boolean_mask.sum(dim=1)
        expected_prefix = (
            torch.arange(mask.shape[1], device=mask.device)[None]
            < lengths[:, None]
        )
        if bool((lengths <= 0).any()) or not torch.equal(
            boolean_mask, expected_prefix
        ):
            raise WCMBaselineError("candidate action mask must be one nonempty prefix")
        for name in ("event_age_seconds", "remaining_action_budget", "dt"):
            value = batch.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != state.shape[:1]
                or not bool(torch.isfinite(value).all())
            ):
                raise WCMBaselineError(f"causal history summary {name} is invalid")
        if bool((batch["event_age_seconds"] < 0).any()):
            raise WCMBaselineError("event age cannot be negative")
        if bool((batch["remaining_action_budget"] <= 0).any()) or bool(
            (batch["dt"] <= 0).any()
        ):
            raise WCMBaselineError("horizon/dt must be positive")
        for name, expected in (
            ("action_available", 1),
            ("action_schema_id", 0),
            ("body_id", 0),
        ):
            if name not in batch:
                continue
            value = batch[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != state.shape[:1]
                or not bool((value == expected).all())
            ):
                raise WCMBaselineError(
                    f"matched WCM requires canonical shared {name}={expected}"
                )

    def _context_latent(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        state = (
            (batch["state"] - self.state_mean) / self.state_std
        ).clamp(-STANDARDIZED_INPUT_CLIP, STANDARDIZED_INPUT_CLIP)
        history = torch.stack(
            (
                torch.log1p(batch["event_age_seconds"]),
                torch.log1p(batch["remaining_action_budget"]) / math.log(201.0),
                torch.log1p(batch["dt"]),
            ),
            dim=-1,
        ).to(state)
        return self.state_history_encoder(torch.cat((state, history), dim=-1))

    def _action_latent(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        actions = (
            (batch["actions"] - self.action_mean) / self.action_std
        ).clamp(-STANDARDIZED_INPUT_CLIP, STANDARDIZED_INPUT_CLIP)
        tokens = self.action_token_encoder(actions)
        lengths = batch["action_mask"].bool().sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            tokens,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _sequence, hidden = self.action_sequence_encoder(packed)
        return hidden[-1]

    def build_future_target(self, batch: Mapping[str, Any]) -> torch.Tensor:
        terminal_event = batch.get("terminal_max_event_id")
        success = batch.get("success")
        stage = batch.get("terminal_stage_progress")
        goal = batch.get("terminal_goal_progress")
        object_effect = batch.get("object_delta")
        batch_size = batch["state"].shape[0]
        if (
            not isinstance(terminal_event, torch.Tensor)
            or terminal_event.shape != (batch_size,)
            or not isinstance(success, torch.Tensor)
            or success.shape != (batch_size,)
            or not isinstance(stage, torch.Tensor)
            or stage.shape != (batch_size,)
            or not isinstance(goal, torch.Tensor)
            or goal.shape != (batch_size,)
            or not isinstance(object_effect, torch.Tensor)
            or object_effect.shape != (batch_size, OBJECT_EFFECT_DIM)
            or not bool(torch.isfinite(success).all())
            or not bool(torch.isfinite(stage).all())
            or not bool(torch.isfinite(goal).all())
            or not bool(torch.isfinite(object_effect).all())
        ):
            raise WCMBaselineError("branch future target is missing or invalid")
        if bool(((success < 0) | (success > 1)).any()) or bool(
            ((stage < 0) | (stage > 1)).any()
        ):
            raise WCMBaselineError("success/stage targets are outside [0,1]")
        terminal_event = terminal_event.long()
        if bool(((terminal_event < 0) | (terminal_event >= EVENT_COUNT)).any()):
            raise WCMBaselineError("terminal event target is outside the canonical set")
        event_onehot = F.one_hot(terminal_event, EVENT_COUNT).to(stage)
        bounded_goal = torch.tanh(goal / GOAL_PROGRESS_SCALE_METERS)[:, None]
        bounded_effect = torch.tanh(
            object_effect / self.object_effect_scales.to(object_effect)
        )
        return torch.cat(
            (
                event_onehot,
                success.to(stage)[:, None],
                stage[:, None],
                bounded_goal,
                bounded_effect,
            ),
            dim=-1,
        )

    def _decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.value_head(latent)
        value_mean, value_log_scale = value.chunk(2, dim=-1)
        effect = self.object_effect_head(latent)
        effect_mean, effect_log_scale = effect.chunk(2, dim=-1)
        terminal_event_logits = self.terminal_event_head(latent)
        success_logit = self.success_head(latent).squeeze(-1)
        success_probability = torch.sigmoid(success_logit)
        candidate_rank_logit = (
            success_probability
            + 0.25 * value_mean[:, 0].clamp(0.0, 1.0)
            + 0.05 * torch.tanh(value_mean[:, 1])
        )
        return {
            "terminal_event_logits": terminal_event_logits,
            "success_logit": success_logit,
            "success_probability": success_probability,
            "value_mean": value_mean,
            "value_log_scale": value_log_scale.clamp(-5.0, 2.0),
            "object_effect_mean": effect_mean,
            "object_effect_log_scale": effect_log_scale.clamp(-5.0, 2.0),
            "candidate_rank_logit": candidate_rank_logit,
        }

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        self._validate_inputs(batch)
        context = self._context_latent(batch)
        action = self._action_latent(batch)
        hidden = self.context_action_fusion(torch.cat((context, action), dim=-1))
        for block in self.dynamics_blocks:
            hidden = block(hidden)
        predicted = context + self.future_delta(hidden)
        output = {
            "context_latent": context,
            "candidate_action_latent": action,
            "predicted_future_latent": predicted,
            **self._decode(predicted),
        }
        if "terminal_max_event_id" in batch:
            target_vector = self.build_future_target(batch)
            target_latent = self.future_target_encoder(target_vector)
            output["future_target_vector"] = target_vector
            output["target_future_latent"] = target_latent
            for name, value in self._decode(target_latent).items():
                output[f"target_{name}"] = value
        return output


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or weight.shape != value.shape:
        raise WCMBaselineError("weighted loss rows changed shape")
    if (
        not bool(torch.isfinite(value).all())
        or not bool(torch.isfinite(weight).all())
        or bool((weight < 0).any())
    ):
        raise WCMBaselineError("weighted loss values or weights are invalid")
    denominator = weight.sum()
    if float(denominator.detach()) <= 0.0:
        return value.sum() * 0.0
    return (value * weight).sum() / denominator


def _gaussian_nll(
    mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if mean.shape != log_scale.shape or mean.shape != target.shape:
        raise WCMBaselineError("Gaussian proper-loss shapes changed")
    return 0.5 * ((target - mean) / log_scale.exp()).square() + log_scale


def compute_wcm_loss(
    model: WCMFutureLatentBaseline,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    sample_weight: torch.Tensor | None = None,
    latent_mse_weight: float = 1.0,
    success_weight: float = 1.0,
    value_weight: float = 1.0,
    event_weight: float = 0.5,
    effect_weight: float = 0.5,
    target_decode_weight: float = 0.25,
    sigreg_weight: float = 0.01,
    variance_covariance_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint latent, proper outcome/value/effect, and anti-collapse objective."""

    predicted = output.get("predicted_future_latent")
    target = output.get("target_future_latent")
    if not isinstance(predicted, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise WCMBaselineError("training loss requires future target labels")
    rows = predicted.shape[0]
    weight = (
        torch.ones(rows, device=predicted.device, dtype=predicted.dtype)
        if sample_weight is None
        else sample_weight.to(predicted)
    )
    masks = {}
    for name in (
        "success_mask",
        "terminal_event_mask",
        "terminal_goal_progress_mask",
        "object_delta_mask",
    ):
        mask = batch.get(name)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.shape != (rows,)
            or not bool(torch.isfinite(mask).all())
            or bool(((mask < 0) | (mask > 1)).any())
        ):
            raise WCMBaselineError(f"branch supervision mask {name} is invalid")
        masks[name] = mask.to(predicted)
    complete_target_weight = weight
    for mask in masks.values():
        complete_target_weight = complete_target_weight * mask
    latent_rows = (predicted - target.detach()).square().mean(dim=-1)
    latent_loss = _weighted_mean(latent_rows, complete_target_weight)
    success_target = batch["success"].to(predicted)
    success_mask = masks["success_mask"]
    success_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output["success_logit"], success_target, reduction="none"
        ),
        weight * success_mask,
    )
    value_target = torch.stack(
        (
            batch["terminal_stage_progress"].to(predicted),
            torch.tanh(
                batch["terminal_goal_progress"].to(predicted)
                / GOAL_PROGRESS_SCALE_METERS
            ),
        ),
        dim=-1,
    )
    value_mask = torch.stack(
        (
            masks["terminal_event_mask"],
            masks["terminal_goal_progress_mask"],
        ),
        dim=-1,
    )
    value_rows = (
        _gaussian_nll(
            output["value_mean"], output["value_log_scale"], value_target
        )
        * value_mask
    ).sum(dim=-1) / value_mask.sum(dim=-1).clamp_min(1.0)
    value_loss = _weighted_mean(
        value_rows, weight * (value_mask.sum(dim=-1) > 0).to(weight)
    )
    event_rows = F.cross_entropy(
        output["terminal_event_logits"],
        batch["terminal_max_event_id"].long(),
        reduction="none",
    )
    event_loss = _weighted_mean(
        event_rows, weight * masks["terminal_event_mask"].to(weight)
    )
    effect_target = torch.tanh(
        batch["object_delta"].to(predicted)
        / model.object_effect_scales.to(predicted)
    )
    effect_rows = _gaussian_nll(
        output["object_effect_mean"],
        output["object_effect_log_scale"],
        effect_target,
    ).mean(dim=-1)
    effect_loss = _weighted_mean(
        effect_rows, weight * masks["object_delta_mask"].to(weight)
    )
    target_success_loss = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output["target_success_logit"], success_target, reduction="none"
        ),
        complete_target_weight,
    )
    target_value_rows = (
        _gaussian_nll(
            output["target_value_mean"],
            output["target_value_log_scale"],
            value_target,
        )
        * value_mask
    ).sum(dim=-1) / value_mask.sum(dim=-1).clamp_min(1.0)
    target_decode_loss = target_success_loss + _weighted_mean(
        target_value_rows, complete_target_weight
    )
    sigreg_target = model.sigreg(target, complete_target_weight)
    sigreg_predicted = model.sigreg(predicted, weight)
    sigreg_loss = 0.5 * (sigreg_target + sigreg_predicted)
    vicreg_target, target_vicreg = variance_covariance_regularizer(
        target, sample_weight=complete_target_weight
    )
    vicreg_predicted, predicted_vicreg = variance_covariance_regularizer(
        predicted, sample_weight=weight
    )
    variance_covariance_loss = 0.5 * (vicreg_target + vicreg_predicted)
    total = (
        latent_mse_weight * latent_loss
        + success_weight * success_loss
        + value_weight * value_loss
        + event_weight * event_loss
        + effect_weight * effect_loss
        + target_decode_weight * target_decode_loss
        + sigreg_weight * sigreg_loss
        + variance_covariance_weight * variance_covariance_loss
    )
    if not bool(torch.isfinite(total)):
        raise WCMBaselineError("WCM baseline loss is non-finite")
    return total, {
        "total": total,
        "latent_mse": latent_loss,
        "success_binary_nll": success_loss,
        "value_diagonal_gaussian_nll": value_loss,
        "terminal_event_categorical_nll": event_loss,
        "object_effect_diagonal_gaussian_nll": effect_loss,
        "target_decode_proper": target_decode_loss,
        "sigreg": sigreg_loss,
        "variance_covariance": variance_covariance_loss,
        "target_minimum_std": target_vicreg["minimum_std"],
        "predicted_minimum_std": predicted_vicreg["minimum_std"],
    }


def aggregate_epistemic_lcb(member_scores: torch.Tensor) -> torch.Tensor:
    if (
        member_scores.ndim != 2
        or member_scores.shape[0] != 5
        or member_scores.shape[1] not in CANDIDATE_COUNTS
        or not bool(torch.isfinite(member_scores).all())
    ):
        raise WCMBaselineError("member candidate scores must be finite [5,N4|N8]")
    return member_scores.mean(dim=0) - EPISTEMIC_RISK_WEIGHT * member_scores.std(
        dim=0, correction=0
    )


class WCMFutureLatentEnsemble(torch.nn.Module):
    def __init__(self, models: Sequence[WCMFutureLatentBaseline]) -> None:
        super().__init__()
        if len(models) != 5:
            raise WCMBaselineError("formal WCM baseline ensemble requires five members")
        self.models = torch.nn.ModuleList(models)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        member_scores = torch.stack(
            [model(batch)["candidate_rank_logit"] for model in self.models]
        )
        if member_scores.shape[1] not in CANDIDATE_COUNTS:
            raise WCMBaselineError("runtime candidate pool must be N4 or N8")
        return {
            "candidate_rank_score_members": member_scores,
            "candidate_rank_score_epistemic_lcb_ensemble": (
                aggregate_epistemic_lcb(member_scores)
            ),
        }


def validate_runtime_candidate_batch(
    batch: Mapping[str, Any], *, candidate_count: int
) -> None:
    if candidate_count not in CANDIDATE_COUNTS:
        raise WCMBaselineError("candidate_count must be 4 or 8")
    candidate_index = batch.get("candidate_index")
    if (
        not isinstance(candidate_index, torch.Tensor)
        or candidate_index.shape != (candidate_count,)
        or not torch.equal(
            candidate_index.cpu(), torch.arange(candidate_count)
        )
    ):
        raise WCMBaselineError("runtime candidate order must be exactly 0..N-1")
    groups = batch.get("logical_group")
    if (
        not isinstance(groups, list)
        or len(groups) != candidate_count
        or len(set(str(value) for value in groups)) != 1
    ):
        raise WCMBaselineError("runtime candidates must share one logical root")
    forbidden = {
        "success",
        "success_mask",
        "post_event_id",
        "post_event_mask",
        "next_event_id",
        "next_event_mask",
        "duration",
        "duration_mask",
        "duration_observed",
        "recovery",
        "recovery_mask",
        "terminal_stage_progress",
        "terminal_event_mask",
        "terminal_goal_progress",
        "terminal_goal_progress_mask",
        "terminal_max_event_id",
        "object_delta",
        "object_delta_mask",
    }
    if forbidden & set(batch):
        raise WCMBaselineError("runtime scorer received candidate outcome labels")


@torch.inference_mode()
def score_candidate_pool(
    ensemble: WCMFutureLatentEnsemble,
    batch: Mapping[str, Any],
    *,
    candidate_count: int,
) -> dict[str, Any]:
    validate_runtime_candidate_batch(batch, candidate_count=candidate_count)
    ensemble.eval()
    output = ensemble(batch)
    aggregate = output["candidate_rank_score_epistemic_lcb_ensemble"]
    selected = int(torch.argmax(aggregate).item())
    return {
        "model_family": MODEL_FAMILY,
        "candidate_count": candidate_count,
        "candidate_rank_score_members": output[
            "candidate_rank_score_members"
        ].detach().cpu(),
        "candidate_rank_score_epistemic_lcb_ensemble": aggregate.detach().cpu(),
        "selected_candidate_index": selected,
        "rank_score_contract": dict(RANK_SCORE_CONTRACT),
    }


def _validate_checkpoint_normalization(
    value: Any, model: WCMFutureLatentBaseline
) -> None:
    required = {
        "format",
        "canonical_state_schema",
        "canonical_action_schema",
        "state_continuous_channels",
        "state_binary_channels_unchanged",
        "state_mean",
        "state_std",
        "action_mean",
        "action_std",
        "primary_source_train_rows",
        "supplement_rows_used",
        "validation_rows_used",
        "heldout_rows_used",
        "logical_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise WCMBaselineError("checkpoint normalization receipt fields changed")
    unsigned = dict(value)
    digest = unsigned.pop("logical_sha256", None)
    if (
        digest != canonical_sha256(unsigned)
        or value.get("format")
        != "etsf_wcm_matched_primary_source_normalization_v1"
        or value.get("canonical_state_schema") != STATE_SCHEMA
        or value.get("canonical_action_schema") != ACTION_SCHEMA
        or value.get("state_continuous_channels") != list(range(18))
        or value.get("state_binary_channels_unchanged") != list(range(18, STATE_DIM))
        or type(value.get("primary_source_train_rows")) is not int
        or value["primary_source_train_rows"] <= 0
        or value.get("supplement_rows_used") != 0
        or value.get("validation_rows_used") != 0
        or value.get("heldout_rows_used") != 0
    ):
        raise WCMBaselineError("checkpoint normalization scope changed")
    expected = {
        "state_mean": model.state_mean,
        "state_std": model.state_std,
        "action_mean": model.action_mean,
        "action_std": model.action_std,
    }
    for name, observed in expected.items():
        try:
            recorded = torch.tensor(value[name], dtype=observed.dtype)
        except (TypeError, ValueError) as error:
            raise WCMBaselineError("checkpoint normalization values are invalid") from error
        if recorded.shape != observed.shape or not torch.equal(recorded, observed.cpu()):
            raise WCMBaselineError("checkpoint model/normalization buffers disagree")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_member_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[WCMFutureLatentBaseline, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        checkpoint = torch.load(
            resolved, map_location=map_location, weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise WCMBaselineError(f"cannot load WCM checkpoint: {resolved}") from error
    if not isinstance(checkpoint, Mapping):
        raise WCMBaselineError("WCM checkpoint must be a mapping")
    required = {
        "format",
        "model_family",
        "config",
        "model",
        "member",
        "seed",
        "step",
        "held_out_body",
        "source_bodies",
        "canonical_state_schema",
        "canonical_action_schema",
        "state_action_frame_contract",
        "event_spec_sha256",
        "actor_execution_protocol",
        "actor_execution_protocol_binding",
        "actor_execution_protocol_file_sha256",
        "primary_binding_file_sha256",
        "supplement_binding_file_sha256",
        "heldout_rows_used_for_training_normalization_or_selection",
        "trainable_parameter_count",
        "rank_score_contract",
        "trainer_file_sha256",
        "preflight_logical_sha256",
        "normalization",
        "validation",
    }
    if (
        set(checkpoint) != required
        or checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("model_family") != MODEL_FAMILY
        or checkpoint.get("canonical_state_schema") != STATE_SCHEMA
        or checkpoint.get("canonical_action_schema") != ACTION_SCHEMA
        or checkpoint.get("state_action_frame_contract")
        != STATE_ACTION_FRAME_CONTRACT
        or checkpoint.get("event_spec_sha256") != EVENT_SPEC_SHA256
        or checkpoint.get("heldout_rows_used_for_training_normalization_or_selection")
        != 0
        or checkpoint.get("rank_score_contract") != RANK_SCORE_CONTRACT
        or type(checkpoint.get("member")) is not int
        or not 0 <= checkpoint["member"] < 5
        or checkpoint.get("held_out_body") not in BODIES
        or checkpoint.get("source_bodies")
        != [body for body in BODIES if body != checkpoint.get("held_out_body")]
        or type(checkpoint.get("seed")) is not int
        or type(checkpoint.get("step")) is not int
        or checkpoint["step"] <= 0
        or not all(
            _is_sha256(checkpoint.get(name))
            for name in (
                "actor_execution_protocol_file_sha256",
                "primary_binding_file_sha256",
                "trainer_file_sha256",
                "preflight_logical_sha256",
            )
        )
        or (
            checkpoint.get("supplement_binding_file_sha256") is not None
            and not _is_sha256(checkpoint["supplement_binding_file_sha256"])
        )
    ):
        raise WCMBaselineError("WCM checkpoint contract changed")
    try:
        actor_execution.validate_execution_protocol(
            checkpoint["actor_execution_protocol"]
        )
    except actor_execution.ActorExecutionProtocolError as error:
        raise WCMBaselineError("checkpoint actor protocol is not frozen") from error
    protocol_binding = checkpoint.get("actor_execution_protocol_binding")
    if (
        not isinstance(protocol_binding, Mapping)
        or set(protocol_binding)
        != {
            "format",
            "path_root",
            "path",
            "file_sha256",
            "protocol_logical_sha256",
            "protocol",
        }
        or protocol_binding.get("format") != actor_execution.FILE_BINDING_FORMAT
        or not isinstance(protocol_binding.get("path_root"), str)
        or not protocol_binding["path_root"]
        or not isinstance(protocol_binding.get("path"), str)
        or not protocol_binding["path"]
        or protocol_binding.get("protocol") != checkpoint["actor_execution_protocol"]
        or protocol_binding.get("protocol_logical_sha256")
        != checkpoint["actor_execution_protocol"]["logical_sha256"]
        or protocol_binding.get("file_sha256")
        != checkpoint["actor_execution_protocol_file_sha256"]
    ):
        raise WCMBaselineError("checkpoint actor protocol binding disagrees")
    validation = checkpoint.get("validation")
    source_validation = (
        validation.get("source_validation")
        if isinstance(validation, Mapping)
        else None
    )
    supplement_validation = (
        validation.get("supplement_source_validation")
        if isinstance(validation, Mapping)
        else None
    )
    if (
        not isinstance(validation, Mapping)
        or validation.get("step") != checkpoint["step"]
        or validation.get("checkpoint_selection_uses_only_source_strict_proper")
        is not True
        or not isinstance(source_validation, Mapping)
        or source_validation.get("source_validation_only") is not True
        or (
            supplement_validation is not None
            and (
                not isinstance(supplement_validation, Mapping)
                or supplement_validation.get("source_validation_only") is not True
            )
        )
    ):
        raise WCMBaselineError("checkpoint selection is not proven source-only")
    try:
        config = WCMConfig(**dict(checkpoint["config"]))
    except (TypeError, ValueError) as error:
        raise WCMBaselineError("WCM checkpoint config is invalid") from error
    model = WCMFutureLatentBaseline(config).to(torch.device(map_location))
    model.load_state_dict(checkpoint["model"], strict=True)
    if count_trainable_parameters(model) != checkpoint["trainable_parameter_count"]:
        raise WCMBaselineError("WCM checkpoint parameter count changed")
    parameter_budget_receipt(model)
    _validate_checkpoint_normalization(checkpoint["normalization"], model)
    model.eval()
    return model, dict(checkpoint)


def load_five_member_ensemble(
    paths: Sequence[Path],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[WCMFutureLatentEnsemble, list[dict[str, Any]]]:
    if len(paths) != 5:
        raise WCMBaselineError("exactly five member checkpoint paths are required")
    loaded = [
        load_member_checkpoint(path, map_location=map_location) for path in paths
    ]
    models = [item[0] for item in loaded]
    receipts = [item[1] for item in loaded]
    if (
        [item["member"] for item in receipts] != list(range(5))
        or len({item["held_out_body"] for item in receipts}) != 1
        or len({item["step"] for item in receipts}) != 1
        or len({canonical_sha256(item["source_bodies"]) for item in receipts}) != 1
        or len({item["seed"] for item in receipts}) != 5
        or len({item["primary_binding_file_sha256"] for item in receipts}) != 1
        or len({item["supplement_binding_file_sha256"] for item in receipts}) != 1
        or len({canonical_sha256(item["normalization"]) for item in receipts}) != 1
        or len(
            {
                canonical_sha256(item["actor_execution_protocol"])
                for item in receipts
            }
        )
        != 1
    ):
        raise WCMBaselineError("five WCM members do not form one common-step LOBO fold")
    return WCMFutureLatentEnsemble(models), receipts


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_budget_receipt(model: WCMFutureLatentBaseline) -> dict[str, Any]:
    count = count_trainable_parameters(model)
    ratio = count / float(V13_TRAINABLE_PARAMETER_REFERENCE)
    if not 0.95 <= ratio <= 1.05:
        raise WCMBaselineError(
            "matched WCM parameter count is outside 0.95x--1.05x of v13"
        )
    return {
        "trainable_parameters": count,
        "v13_trainable_parameter_reference": V13_TRAINABLE_PARAMETER_REFERENCE,
        "ratio_to_v13": ratio,
        "accepted_ratio_interval": [0.95, 1.05],
        "matched": True,
    }


__all__ = [
    "ACTION_DIM",
    "ACTION_SCHEMA",
    "BODIES",
    "CANDIDATE_COUNTS",
    "CHECKPOINT_FORMAT",
    "EPISTEMIC_RISK_WEIGHT",
    "EVENT_SPEC_SHA256",
    "FORMAT",
    "FUTURE_TARGET_SCHEMA",
    "MODEL_FAMILY",
    "RANK_SCORE_CONTRACT",
    "STATE_DIM",
    "STATE_SCHEMA",
    "STATE_ACTION_FRAME_CONTRACT",
    "SketchedIsotropicGaussianRegularizer",
    "WCMBaselineError",
    "WCMConfig",
    "WCMFutureLatentBaseline",
    "WCMFutureLatentEnsemble",
    "aggregate_epistemic_lcb",
    "canonical_sha256",
    "compute_wcm_loss",
    "count_trainable_parameters",
    "load_five_member_ensemble",
    "load_member_checkpoint",
    "parameter_budget_receipt",
    "score_candidate_pool",
    "sha256_file",
    "validate_runtime_candidate_batch",
    "variance_covariance_regularizer",
]
