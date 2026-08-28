#!/usr/bin/env python3
"""Detached group-relative success probability and candidate-ranking heads.

The factual transition tensor is immutable input.  The probability head and
ranking head have disjoint parameters and no trainable shared representation.
Relative features are computed within each four-candidate logical group, using
either the deterministic candidate or group mean as the fixed reference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from openvla_etsf_counterfactual_oof import canonical_sha256
from openvla_etsf_v8_structured_adapters import (
    frozen_tensor_mapping_sha256,
    module_state_sha256,
)


FORMAT = "etsf_v9_group_relative_success_ranking_adapter_v1"
TRAINING_FORMAT = "etsf_v9_group_relative_success_ranking_training_v1"
DEPLOYMENT_CANDIDATE_NAMES = (
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
)
RELATIVE_MODES = ("deterministic_delta", "group_centered")
RANKING_OBJECTIVES = ("pairwise_logistic", "listwise_success_cross_entropy")
REGULARIZATION_GRID = (1e-3, 1e-2)
RANKING_LOSS_WEIGHT = 1.0
LBFGS_MAX_ITER = 50
LBFGS_TOLERANCE_GRAD = 1e-7
LBFGS_TOLERANCE_CHANGE = 1e-9
PROBABILITY_CLIP_EPS = 1e-12


@dataclass(frozen=True)
class GroupRelativeAdapterConfig:
    transition_dim: int
    relative_mode: str
    ranking_objective: str
    l2_regularization: float
    ranking_loss_weight: float = RANKING_LOSS_WEIGHT

    def __post_init__(self) -> None:
        if self.transition_dim <= 0:
            raise ValueError("transition_dim must be positive")
        if self.relative_mode not in RELATIVE_MODES:
            raise ValueError("unknown group-relative feature mode")
        if self.ranking_objective not in RANKING_OBJECTIVES:
            raise ValueError("unknown ranking objective")
        if self.l2_regularization not in REGULARIZATION_GRID:
            raise ValueError("regularization is outside the preregistered grid")
        if self.ranking_loss_weight != RANKING_LOSS_WEIGHT:
            raise ValueError("ranking loss weight is fixed, not tunable on D250")

    @property
    def feature_dim(self) -> int:
        return self.transition_dim + len(DEPLOYMENT_CANDIDATE_NAMES)

    @property
    def config_id(self) -> str:
        return (
            f"{self.relative_mode}|{self.ranking_objective}|"
            f"l2={self.l2_regularization:.3g}|rank_weight={self.ranking_loss_weight:.1f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preregistered_config_grid(transition_dim: int) -> tuple[GroupRelativeAdapterConfig, ...]:
    return tuple(
        GroupRelativeAdapterConfig(
            transition_dim=transition_dim,
            relative_mode=relative_mode,
            ranking_objective=ranking_objective,
            l2_regularization=regularization,
        )
        for relative_mode in RELATIVE_MODES
        for ranking_objective in RANKING_OBJECTIVES
        for regularization in REGULARIZATION_GRID
    )


class GroupRelativeSuccessRankingAdapter(nn.Module):
    """Two convex heads over fixed group-relative transition features."""

    def __init__(self, config: GroupRelativeAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.probability_head = nn.Linear(config.feature_dim, 1)
        self.ranking_head = nn.Linear(config.feature_dim, 1)
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_scale", torch.ones(config.feature_dim))
        self.register_buffer("feature_scaler_fitted", torch.tensor(False))

    def _validate_transition(self, transition: torch.Tensor) -> torch.Tensor:
        if (
            not torch.is_tensor(transition)
            or transition.ndim not in (2, 3)
            or transition.shape[-2] != len(DEPLOYMENT_CANDIDATE_NAMES)
            or transition.shape[-1] != self.config.transition_dim
            or not bool(torch.isfinite(transition).all())
            or transition.requires_grad
        ):
            raise ValueError(
                "adapter requires finite detached [groups,4,transition_dim] input"
            )
        return transition.float()

    def raw_relative_features(self, transition: torch.Tensor) -> torch.Tensor:
        transition = self._validate_transition(transition)
        squeezed = transition.ndim == 2
        if squeezed:
            transition = transition.unsqueeze(0)
        if self.config.relative_mode == "deterministic_delta":
            reference = transition[:, :1]
        else:
            reference = transition.mean(dim=1, keepdim=True)
        relative = transition - reference
        identity = torch.eye(
            len(DEPLOYMENT_CANDIDATE_NAMES),
            device=transition.device,
            dtype=transition.dtype,
        ).expand(len(transition), -1, -1)
        result = torch.cat((relative, identity), dim=-1)
        return result.squeeze(0) if squeezed else result

    @torch.no_grad()
    def fit_feature_scaler(self, transition: torch.Tensor) -> None:
        raw = self.raw_relative_features(transition)
        if raw.ndim == 2:
            raw = raw.unsqueeze(0)
        flattened = raw.reshape(-1, raw.shape[-1])
        mean = flattened.mean(dim=0)
        scale = flattened.std(dim=0, unbiased=False).clamp_min(1e-6)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()):
            raise RuntimeError("group-relative feature scaler became non-finite")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)
        self.feature_scaler_fitted.fill_(True)

    def features(self, transition: torch.Tensor) -> torch.Tensor:
        if not bool(self.feature_scaler_fitted):
            raise RuntimeError("feature scaler must be fit on the training groups")
        raw = self.raw_relative_features(transition)
        return (raw - self.feature_mean) / self.feature_scale

    def forward(self, transition: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.features(transition)
        return {
            "success_logit": self.probability_head(feature).squeeze(-1),
            "candidate_ranking_score": self.ranking_head(feature).squeeze(-1),
        }

    def probability_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.probability_head.parameters())

    def ranking_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.ranking_head.parameters())


def _record_success_rows(
    record: Mapping[str, Any], *, transition_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = record.get("batch")
    factual = record.get("factual_outputs")
    if not isinstance(batch, Mapping) or not isinstance(factual, Mapping):
        raise ValueError("group-relative record lacks batch/factual outputs")
    transition = factual.get("transition")
    terminal = batch.get("terminal_mask")
    success = batch.get("success")
    candidate_names = tuple(map(str, batch.get("candidate_names", ())))
    if (
        not torch.is_tensor(transition)
        or transition.ndim != 2
        or transition.shape[1] != transition_dim
        or transition.requires_grad
        or not bool(torch.isfinite(transition).all())
        or not torch.is_tensor(terminal)
        or not torch.is_tensor(success)
        or terminal.shape != (len(transition),)
        or success.shape != (len(transition),)
        or candidate_names[:4] != DEPLOYMENT_CANDIDATE_NAMES
        or len(transition) < 4
        or not torch.equal(
            terminal.bool().cpu(),
            torch.arange(len(transition)).lt(4),
        )
    ):
        raise ValueError("record is not the authenticated four-candidate terminal layout")
    label = success[:4].detach().float()
    if bool(((label != 0.0) & (label != 1.0)).any()):
        raise ValueError("success labels must be binary")
    return transition[:4].detach(), label


def records_to_group_tensors(
    records: Sequence[Mapping[str, Any]], *, transition_dim: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if not records:
        raise ValueError("group-relative training needs records")
    groups = [str(record.get("logical_group_key", "")) for record in records]
    if any(not group for group in groups) or len(groups) != len(set(groups)):
        raise ValueError("logical_group_key values must be nonempty and unique")
    pairs = [
        _record_success_rows(record, transition_dim=transition_dim)
        for record in records
    ]
    transition = torch.stack([pair[0] for pair in pairs])
    labels = torch.stack([pair[1] for pair in pairs])
    if int(labels.sum()) in (0, labels.numel()):
        raise ValueError("success probability head requires both label classes")
    discordant = labels.unsqueeze(2) > labels.unsqueeze(1)
    if not bool(discordant.any()):
        raise ValueError("ranking head requires at least one discordant candidate pair")
    return transition, labels, groups


def pairwise_logistic_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    difference = scores.unsqueeze(2) - scores.unsqueeze(1)
    preferred = labels.unsqueeze(2) > labels.unsqueeze(1)
    if not bool(preferred.any()):
        raise ValueError("pairwise ranking loss has no discordant labels")
    return F.softplus(-difference[preferred]).mean()


def listwise_success_cross_entropy(
    scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    positive = labels.sum(dim=1)
    mixed = (positive > 0) & (positive < labels.shape[1])
    if not bool(mixed.any()):
        raise ValueError("listwise ranking loss has no mixed-success groups")
    target = labels[mixed] / positive[mixed, None]
    return -(target * F.log_softmax(scores[mixed], dim=1)).sum(dim=1).mean()


def adapter_losses(
    adapter: GroupRelativeSuccessRankingAdapter,
    transition: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    output = adapter(transition)
    probability = F.binary_cross_entropy_with_logits(
        output["success_logit"], labels, reduction="mean"
    )
    if adapter.config.ranking_objective == "pairwise_logistic":
        ranking = pairwise_logistic_loss(
            output["candidate_ranking_score"], labels
        )
    else:
        ranking = listwise_success_cross_entropy(
            output["candidate_ranking_score"], labels
        )
    probability_l2 = 0.5 * adapter.config.l2_regularization * (
        adapter.probability_head.weight.square().sum()
    )
    ranking_l2 = 0.5 * adapter.config.l2_regularization * (
        adapter.ranking_head.weight.square().sum()
    )
    probability_objective = probability + probability_l2
    ranking_objective = ranking + ranking_l2
    combined = (
        probability_objective
        + adapter.config.ranking_loss_weight * ranking_objective
    )
    return {
        "unweighted_success_bce": probability,
        "ranking_loss": ranking,
        "probability_l2": probability_l2,
        "ranking_l2": ranking_l2,
        "probability_objective": probability_objective,
        "ranking_objective": ranking_objective,
        "combined_objective": combined,
    }


def _factual_hashes(records: Sequence[Mapping[str, Any]]) -> list[str]:
    result = []
    for record in records:
        factual = record.get("factual_outputs")
        if (
            not isinstance(factual, Mapping)
            or record.get("factual_outputs_require_grad") is not False
            or record.get("factual_outputs_sha256")
            != frozen_tensor_mapping_sha256(factual)
        ):
            raise ValueError("record factual-output authentication failed")
        result.append(frozen_tensor_mapping_sha256(factual))
    return result


def train_group_relative_adapter(
    records: Sequence[Mapping[str, Any]],
    *,
    config: GroupRelativeAdapterConfig,
    device: torch.device | str = "cpu",
) -> tuple[GroupRelativeSuccessRankingAdapter, dict[str, Any]]:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    factual_before = _factual_hashes(records)
    transition, labels, groups = records_to_group_tensors(
        records, transition_dim=config.transition_dim
    )
    transition = transition.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    adapter = GroupRelativeSuccessRankingAdapter(config).to(device)
    adapter.fit_feature_scaler(transition)
    prevalence = float(labels.mean().cpu())
    with torch.no_grad():
        adapter.probability_head.weight.zero_()
        adapter.probability_head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))
        adapter.ranking_head.weight.zero_()
        adapter.ranking_head.bias.zero_()
    optimizer = torch.optim.LBFGS(
        adapter.parameters(),
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0
    loss_trace: list[float] = []

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        losses = adapter_losses(adapter, transition, labels)
        objective = losses["combined_objective"]
        if not bool(torch.isfinite(objective)):
            raise RuntimeError("group-relative objective became non-finite")
        objective.backward()
        closure_evaluations += 1
        loss_trace.append(float(objective.detach().cpu()))
        return objective

    optimizer.step(closure)
    adapter.eval()
    final_losses = adapter_losses(adapter, transition, labels)
    factual_after = _factual_hashes(records)
    if factual_before != factual_after:
        raise RuntimeError("group-relative training mutated frozen factual outputs")
    probability_ids = {id(parameter) for parameter in adapter.probability_parameters()}
    ranking_ids = {id(parameter) for parameter in adapter.ranking_parameters()}
    if probability_ids & ranking_ids:
        raise RuntimeError("probability and ranking heads unexpectedly share parameters")
    audit: dict[str, Any] = {
        "format": TRAINING_FORMAT,
        "config": config.to_dict(),
        "config_id": config.config_id,
        "training_groups": groups,
        "training_groups_sha256": canonical_sha256(
            {"logical_groups": sorted(groups)}
        ),
        "support": int(labels.numel()),
        "positive": int(labels.sum().cpu()),
        "prevalence": prevalence,
        "optimizer": {
            "name": "full_batch_LBFGS",
            "max_iter": LBFGS_MAX_ITER,
            "tolerance_grad": LBFGS_TOLERANCE_GRAD,
            "tolerance_change": LBFGS_TOLERANCE_CHANGE,
            "line_search_fn": "strong_wolfe",
            "closure_evaluations": closure_evaluations,
        },
        "record_order_sha256": hashlib.sha256(
            "\n".join(groups).encode("utf-8")
        ).hexdigest(),
        "loss_trace_sha256": canonical_sha256(loss_trace),
        "final_losses": {
            key: float(value.detach().cpu()) for key, value in final_losses.items()
        },
        "adapter_state_sha256": module_state_sha256(adapter),
        "factual_outputs_bit_exact": True,
        "factual_outputs_sha256": factual_before,
        "probability_and_ranking_parameters_disjoint": True,
        "shared_trainable_representation": False,
        "unweighted_success_bce": True,
        "ranking_loss_weight_fixed_before_v9_D250_rerun": True,
    }
    audit["training_audit_sha256"] = canonical_sha256(audit)
    return adapter, audit


@torch.no_grad()
def predict_group_relative_adapter(
    adapter: GroupRelativeSuccessRankingAdapter,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    _factual_hashes(records)
    transition, labels, groups = records_to_group_tensors(
        records, transition_dim=adapter.config.transition_dim
    )
    adapter = adapter.to(device).eval()
    output = adapter(transition.to(device=device, dtype=torch.float32))
    # Evaluate probabilities in float64 before clipping.  CUDA float32 sigmoid
    # rounds sufficiently positive logits to exactly one, which makes NLL and
    # the signed probability contract ill-defined even though logits are
    # finite.  The fixed clip is metric stability only; ranking uses the
    # independent, unclipped ranking score below.
    probability = torch.sigmoid(
        output["success_logit"].detach().to(dtype=torch.float64).cpu()
    ).clamp(PROBABILITY_CLIP_EPS, 1.0 - PROBABILITY_CLIP_EPS)
    return {
        "success_probability": probability,
        "candidate_ranking_score": output["candidate_ranking_score"]
        .detach()
        .cpu(),
        "success_label": labels.cpu(),
        "logical_groups": groups,
    }


def adapter_protocol_contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": FORMAT,
        "feature_modes": list(RELATIVE_MODES),
        "feature_definition": (
            "frozen_transition_minus_deterministic_or_group_mean_plus_candidate_one_hot"
        ),
        "feature_scaler_fit_scope": "training_groups_only",
        "probability_head": "independent_linear_unweighted_binary_cross_entropy",
        "ranking_head": "independent_linear_pairwise_or_listwise",
        "shared_trainable_representation": False,
        "ranking_objectives": list(RANKING_OBJECTIVES),
        "regularization_grid": list(REGULARIZATION_GRID),
        "ranking_loss_weight": RANKING_LOSS_WEIGHT,
        "optimizer": {
            "name": "full_batch_LBFGS",
            "max_iter": LBFGS_MAX_ITER,
            "tolerance_grad": LBFGS_TOLERANCE_GRAD,
            "tolerance_change": LBFGS_TOLERANCE_CHANGE,
            "line_search_fn": "strong_wolfe",
        },
        "candidate_count": 4,
        "candidate_names": list(DEPLOYMENT_CANDIDATE_NAMES),
        "probability_evaluation_dtype": "float64",
        "probability_clip_epsilon": PROBABILITY_CLIP_EPS,
        "probability_clip_not_used_for_ranking": True,
        "fresh_inputs_or_labels_used": False,
    }
    value["protocol_sha256"] = canonical_sha256(value)
    return value


def serialize_adapter_state(
    adapter: GroupRelativeSuccessRankingAdapter,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(adapter.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        result[name] = {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    return result


def load_serialized_adapter(
    config: GroupRelativeAdapterConfig,
    state: Mapping[str, Mapping[str, Any]],
) -> GroupRelativeSuccessRankingAdapter:
    adapter = GroupRelativeSuccessRankingAdapter(config)
    expected = adapter.state_dict()
    if set(state) != set(expected):
        raise ValueError("serialized adapter state keys changed")
    dtype_by_name = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
    }
    restored = {}
    for name, reference in expected.items():
        record = state[name]
        dtype = dtype_by_name.get(str(record.get("dtype")))
        shape = tuple(record.get("shape", ()))
        if dtype is None or dtype != reference.dtype or shape != tuple(reference.shape):
            raise ValueError("serialized adapter tensor dtype/shape changed")
        tensor = torch.as_tensor(record.get("values"), dtype=dtype)
        if tuple(tensor.shape) != shape or (
            tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("serialized adapter tensor values are invalid")
        restored[name] = tensor
    adapter.load_state_dict(restored, strict=True)
    adapter.eval()
    return adapter


__all__ = [
    "DEPLOYMENT_CANDIDATE_NAMES",
    "GroupRelativeAdapterConfig",
    "GroupRelativeSuccessRankingAdapter",
    "adapter_losses",
    "adapter_protocol_contract",
    "listwise_success_cross_entropy",
    "pairwise_logistic_loss",
    "predict_group_relative_adapter",
    "preregistered_config_grid",
    "records_to_group_tensors",
    "serialize_adapter_state",
    "load_serialized_adapter",
    "train_group_relative_adapter",
]
