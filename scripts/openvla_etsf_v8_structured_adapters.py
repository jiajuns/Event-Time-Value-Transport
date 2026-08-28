#!/usr/bin/env python3
"""Isolated v8 adapters for the frozen ETSF factual world-model outputs.

This module deliberately does not import or mutate the factual world model.  A
caller materialises the factual ``transition`` feature and duration log-mean,
then passes them here.  Every factual tensor is detached at the public module
boundary.  The only trainable parameters belong to three independent binary
probability heads:

* terminal success (failure is its complement);
* structured trajectory regression;
* recovery conditional on an observed regression.

Duration is a fixed, preregistered residual shrinkage around an outer-training
event x body median.  Object displacement is always a supplied robust/zero
fallback and has no trainable parameter in v8.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


V8_ADAPTER_FORMAT = "etsf_v8_detached_structured_adapters_v1"
V8_SCHEMA_VERSION = 5
V8_DURATION_RESIDUAL_MULTIPLIER = 0.375
V8_OBJECT_MODE = "outer_training_robust_or_zero_fallback_no_learned_head"
V8_LOSS_CONTRACT = "independent_unweighted_binary_bce_heads_v1"


@dataclass(frozen=True)
class V8StructuredAdapterConfig:
    transition_dim: int
    duration_residual_multiplier: float = V8_DURATION_RESIDUAL_MULTIPLIER
    schema_version: int = V8_SCHEMA_VERSION
    object_mode: str = V8_OBJECT_MODE

    def __post_init__(self) -> None:
        if self.transition_dim <= 0:
            raise ValueError("transition_dim must be positive")
        if self.schema_version != V8_SCHEMA_VERSION:
            raise ValueError("v8 adapters require schema version 5 supervision")
        if not math.isclose(
            self.duration_residual_multiplier,
            V8_DURATION_RESIDUAL_MULTIPLIER,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("v8 duration residual multiplier is frozen at 0.375")
        if self.object_mode != V8_OBJECT_MODE:
            raise ValueError("v8 object output must remain the robust/zero fallback")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V8StructuredAdapterConfig":
        return cls(
            transition_dim=int(value["transition_dim"]),
            duration_residual_multiplier=float(
                value.get(
                    "duration_residual_multiplier",
                    V8_DURATION_RESIDUAL_MULTIPLIER,
                )
            ),
            schema_version=int(value.get("schema_version", V8_SCHEMA_VERSION)),
            object_mode=str(value.get("object_mode", V8_OBJECT_MODE)),
        )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    # Flatten first because PyTorch cannot reinterpret a scalar tensor as a
    # different element size while it still has zero dimensions.
    detached = value.detach().contiguous().cpu().reshape(-1)
    return detached.view(torch.uint8).numpy().tobytes()


def frozen_tensor_mapping_sha256(
    values: Mapping[str, torch.Tensor],
) -> str:
    """Hash tensor values, shapes and dtypes without following autograd."""

    digest = hashlib.sha256()
    for key in sorted(values):
        value = values[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key} is not a tensor")
        metadata = json.dumps(
            {
                "key": str(key),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = _tensor_bytes(value)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    return frozen_tensor_mapping_sha256(dict(module.state_dict()))


def _finite_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _binary_vector(value: Any, *, name: str, count: int) -> torch.Tensor:
    result = _finite_tensor(value, name=name, shape=(count,))
    if bool(((result != 0) & (result != 1)).any()):
        raise ValueError(f"{name} must be binary")
    return result


def validate_schema5_adapter_batch(
    batch: Mapping[str, Any],
    *,
    expected_count: int | None = None,
) -> dict[str, int]:
    """Validate only the existing schema5 fields consumed by v8.

    Continuation rows are supported: their terminal mask is false, while their
    structured/dense masks and regression/recovery labels remain supervised.
    """

    required = {
        "terminal_mask",
        "structured_mask",
        "dense_mask",
        "duration",
        "duration_observed",
        "success",
        "trajectory_regress",
        "trajectory_recovery",
        "object_delta",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"schema5 adapter batch missing fields: {missing}")
    duration = batch["duration"]
    if not torch.is_tensor(duration) or duration.ndim != 1:
        raise ValueError("duration must be a vector")
    count = int(len(duration))
    if count < 1 or (expected_count is not None and count != expected_count):
        raise ValueError("schema5 adapter batch size mismatch")
    duration = _finite_tensor(duration, name="duration", shape=(count,))
    if bool((duration < 0).any()):
        raise ValueError("duration must be non-negative")

    terminal = _binary_vector(batch["terminal_mask"], name="terminal_mask", count=count).bool()
    structured = _binary_vector(
        batch["structured_mask"], name="structured_mask", count=count
    ).bool()
    dense = _binary_vector(batch["dense_mask"], name="dense_mask", count=count).bool()
    observed = _binary_vector(
        batch["duration_observed"], name="duration_observed", count=count
    ).bool()
    success = _binary_vector(batch["success"], name="success", count=count).bool()
    regress = _binary_vector(
        batch["trajectory_regress"], name="trajectory_regress", count=count
    ).bool()
    recovery = _binary_vector(
        batch["trajectory_recovery"], name="trajectory_recovery", count=count
    ).bool()
    object_delta = batch["object_delta"]
    if not torch.is_tensor(object_delta) or object_delta.ndim != 2:
        raise ValueError("object_delta must have shape [items,coordinates]")
    _finite_tensor(
        object_delta,
        name="object_delta",
        shape=(count, int(object_delta.shape[1])),
    )
    if bool((observed & ~dense).any()):
        raise ValueError("observed duration rows must be dense")
    if bool((regress & ~structured).any()):
        raise ValueError("schema5 trajectory_regress requires structured supervision")
    if bool((recovery & ~(structured & regress)).any()):
        raise ValueError("schema5 recovery supervision requires trajectory_regress")
    return {
        "items": count,
        "terminal": int(terminal.sum()),
        "structured": int(structured.sum()),
        "dense": int(dense.sum()),
        "observed_duration": int(observed.sum()),
        "success_positive": int((success & terminal).sum()),
        "regress_positive": int((regress & structured).sum()),
        "conditional_recovery_support": int((regress & structured).sum()),
        "conditional_recovery_positive": int((recovery & regress & structured).sum()),
    }


def validate_factual_adapter_inputs(
    factual_outputs: Mapping[str, Any],
    *,
    count: int,
    transition_dim: int,
) -> dict[str, torch.Tensor]:
    required = {"transition", "duration_selected_log_mean"}
    missing = sorted(required - set(factual_outputs))
    if missing:
        raise ValueError(f"factual outputs missing fields: {missing}")
    transition = _finite_tensor(
        factual_outputs["transition"],
        name="factual transition",
        shape=(count, transition_dim),
    )
    duration = _finite_tensor(
        factual_outputs["duration_selected_log_mean"],
        name="factual duration_selected_log_mean",
        shape=(count,),
    )
    if duration.device != transition.device:
        raise ValueError("factual transition and duration must share a device")
    return {"transition": transition, "duration_selected_log_mean": duration}


class V8DetachedStructuredAdapters(torch.nn.Module):
    """Three independent probability heads plus non-trainable safe repairs."""

    def __init__(self, config: V8StructuredAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.success_head = torch.nn.Linear(config.transition_dim, 1)
        self.regress_head = torch.nn.Linear(config.transition_dim, 1)
        self.recovery_given_regress_head = torch.nn.Linear(
            config.transition_dim, 1
        )
        self.register_buffer(
            "duration_residual_multiplier",
            torch.tensor(config.duration_residual_multiplier, dtype=torch.float32),
            persistent=True,
        )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        allowed_prefixes = (
            "success_head.",
            "regress_head.",
            "recovery_given_regress_head.",
        )
        names = tuple(name for name, _ in self.named_parameters())
        if not names or any(not name.startswith(allowed_prefixes) for name in names):
            raise RuntimeError("v8 registered a trainable parameter outside adapter heads")
        return names

    def initialize_probability_biases(
        self,
        *,
        success_prevalence: float,
        regress_prevalence: float,
        recovery_given_regress_prevalence: float,
    ) -> None:
        """Zero weights and initialize intercepts to outer-training priors."""

        prevalences = (
            success_prevalence,
            regress_prevalence,
            recovery_given_regress_prevalence,
        )
        if any(not 0.0 < value < 1.0 for value in prevalences):
            raise ValueError("adapter prevalence initialization needs both classes")
        with torch.no_grad():
            for head, prevalence in zip(
                (
                    self.success_head,
                    self.regress_head,
                    self.recovery_given_regress_head,
                ),
                prevalences,
            ):
                head.weight.zero_()
                head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))

    def forward(
        self,
        factual_outputs: Mapping[str, Any],
        *,
        duration_baseline_log1p: torch.Tensor,
        object_fallback: torch.Tensor,
    ) -> dict[str, Any]:
        transition_value = factual_outputs.get("transition")
        if not torch.is_tensor(transition_value) or transition_value.ndim != 2:
            raise ValueError("factual transition must have shape [items,features]")
        count = int(transition_value.shape[0])
        factual = validate_factual_adapter_inputs(
            factual_outputs,
            count=count,
            transition_dim=self.config.transition_dim,
        )
        baseline = _finite_tensor(
            duration_baseline_log1p,
            name="duration_baseline_log1p",
            shape=(count,),
        )
        if baseline.device != factual["transition"].device:
            raise ValueError("duration baseline and factual outputs must share a device")
        fallback = _finite_tensor(object_fallback, name="object_fallback")
        if fallback.ndim == 1:
            fallback = fallback.unsqueeze(0).expand(count, -1)
        elif fallback.ndim != 2 or fallback.shape[0] != count:
            raise ValueError("object_fallback must have shape [coordinates] or [items,coordinates]")
        if fallback.device != factual["transition"].device:
            raise ValueError("object fallback and factual outputs must share a device")

        # Detach at the boundary even when the factual tensors came directly
        # from a trainable model in the same autograd graph.
        transition = factual["transition"].detach()
        factual_duration = factual["duration_selected_log_mean"].detach()
        baseline = baseline.detach()
        fallback = fallback.detach()
        success_logit = self.success_head(transition).squeeze(-1)
        regress_logit = self.regress_head(transition).squeeze(-1)
        recovery_logit = self.recovery_given_regress_head(transition).squeeze(-1)
        success_probability = torch.sigmoid(success_logit)
        regress_probability = torch.sigmoid(regress_logit)
        conditional_recovery_probability = torch.sigmoid(recovery_logit)
        duration_log1p = baseline + self.duration_residual_multiplier.to(
            factual_duration
        ) * (factual_duration - baseline)
        return {
            "success_logit": success_logit,
            "success_probability": success_probability,
            "failure_probability": 1.0 - success_probability,
            "regress_logit": regress_logit,
            "regress_probability": regress_probability,
            "recovery_given_regress_logit": recovery_logit,
            "recovery_given_regress_probability": conditional_recovery_probability,
            "recovery_probability": (
                regress_probability * conditional_recovery_probability
            ),
            "duration_repaired_log1p_mean": duration_log1p,
            "object_delta_point": fallback.clone(),
            "object_prediction_status": V8_OBJECT_MODE,
            "learned_object_output_authorized": False,
        }


def _masked_unweighted_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.any():
        return F.binary_cross_entropy_with_logits(
            logits[mask], labels[mask].to(logits), reduction="mean"
        )
    return logits.sum() * 0.0


def compute_v8_adapter_loss(
    adapters: V8DetachedStructuredAdapters,
    factual_outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    duration_baseline_log1p: torch.Tensor,
    object_fallback: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    transition = factual_outputs.get("transition")
    if not torch.is_tensor(transition) or transition.ndim != 2:
        raise ValueError("factual transition must have shape [items,features]")
    support = validate_schema5_adapter_batch(batch, expected_count=len(transition))
    output = adapters(
        factual_outputs,
        duration_baseline_log1p=duration_baseline_log1p,
        object_fallback=object_fallback,
    )
    terminal = batch["terminal_mask"].bool()
    structured = batch["structured_mask"].bool()
    regress_label = batch["trajectory_regress"].bool()
    recovery_mask = structured & regress_label
    losses = {
        "success_unweighted_bce": _masked_unweighted_bce(
            output["success_logit"], batch["success"], terminal
        ),
        "regress_unweighted_bce": _masked_unweighted_bce(
            output["regress_logit"], batch["trajectory_regress"], structured
        ),
        "recovery_given_regress_unweighted_bce": _masked_unweighted_bce(
            output["recovery_given_regress_logit"],
            batch["trajectory_recovery"],
            recovery_mask,
        ),
    }
    total = sum(losses.values(), output["success_logit"].sum() * 0.0)
    observed = batch["dense_mask"].bool() & batch["duration_observed"].bool()
    duration_mae = (
        torch.abs(
            output["duration_repaired_log1p_mean"][observed]
            - torch.log1p(batch["duration"][observed].to(output["success_logit"]))
        ).mean()
        if observed.any()
        else output["success_logit"].sum().detach() * 0.0
    )
    diagnostics: dict[str, Any] = {
        **support,
        "duration_observed_log1p_mae": duration_mae.detach(),
        "duration_is_fixed_not_an_optimization_loss": True,
        "object_is_fallback_not_an_optimization_loss": True,
        "loss_contract": V8_LOSS_CONTRACT,
        "outputs": output,
    }
    return total, losses, diagnostics


def assert_optimizer_adapter_only(
    optimizer: torch.optim.Optimizer,
    adapters: V8DetachedStructuredAdapters,
) -> None:
    expected = {id(parameter) for parameter in adapters.parameters() if parameter.requires_grad}
    actual_list = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    actual = {id(parameter) for parameter in actual_list}
    if len(actual) != len(actual_list) or actual != expected:
        raise RuntimeError("optimizer parameters must equal the v8 adapter parameters exactly")
    adapters.trainable_parameter_names()


def train_v8_adapter_one_step(
    adapters: V8DetachedStructuredAdapters,
    optimizer: torch.optim.Optimizer,
    factual_outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    duration_baseline_log1p: torch.Tensor,
    object_fallback: torch.Tensor,
    frozen_factual_module: torch.nn.Module | None = None,
    maximum_gradient_norm: float | None = 1.0,
) -> dict[str, Any]:
    """Perform one isolated step and prove factual tensors/state are unchanged."""

    assert_optimizer_adapter_only(optimizer, adapters)
    if maximum_gradient_norm is not None and (
        not math.isfinite(maximum_gradient_norm) or maximum_gradient_norm <= 0
    ):
        raise ValueError("maximum_gradient_norm must be positive or None")
    frozen_inputs = {
        **{
            str(key): value
            for key, value in factual_outputs.items()
            if torch.is_tensor(value)
        },
        "duration_baseline_log1p": duration_baseline_log1p,
        "object_fallback": object_fallback,
    }
    input_sha_before = frozen_tensor_mapping_sha256(frozen_inputs)
    factual_state_before = (
        module_state_sha256(frozen_factual_module)
        if frozen_factual_module is not None
        else None
    )
    adapter_state_before = module_state_sha256(adapters)
    optimizer.zero_grad(set_to_none=True)
    total, losses, diagnostics = compute_v8_adapter_loss(
        adapters,
        factual_outputs,
        batch,
        duration_baseline_log1p=duration_baseline_log1p,
        object_fallback=object_fallback,
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("v8 adapter loss is non-finite")
    total.backward()
    for value in factual_outputs.values():
        if torch.is_tensor(value) and value.is_leaf and value.grad is not None:
            raise RuntimeError("factual output received a gradient across detach boundary")
    if frozen_factual_module is not None and any(
        parameter.grad is not None for parameter in frozen_factual_module.parameters()
    ):
        raise RuntimeError("factual module received a gradient from v8 adapters")
    if maximum_gradient_norm is not None:
        # Keep the three probability tasks optimization-independent: a large
        # gradient in one head must not rescale either of the other heads.
        for head in (
            adapters.success_head,
            adapters.regress_head,
            adapters.recovery_given_regress_head,
        ):
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=maximum_gradient_norm
            )
    optimizer.step()
    input_sha_after = frozen_tensor_mapping_sha256(frozen_inputs)
    if input_sha_after != input_sha_before:
        raise RuntimeError("v8 training mutated frozen factual tensors or fallbacks")
    factual_state_after = (
        module_state_sha256(frozen_factual_module)
        if frozen_factual_module is not None
        else None
    )
    if factual_state_after != factual_state_before:
        raise RuntimeError("v8 training mutated the frozen factual module")
    adapter_state_after = module_state_sha256(adapters)
    return {
        "format": V8_ADAPTER_FORMAT,
        "loss": float(total.detach().cpu()),
        "losses": {
            key: float(value.detach().cpu()) for key, value in losses.items()
        },
        "support": {
            key: value
            for key, value in diagnostics.items()
            if isinstance(value, int) and not isinstance(value, bool)
        },
        "duration_observed_log1p_mae": float(
            diagnostics["duration_observed_log1p_mae"].cpu()
        ),
        "factual_input_sha256_before": input_sha_before,
        "factual_input_sha256_after": input_sha_after,
        "factual_state_sha256_before": factual_state_before,
        "factual_state_sha256_after": factual_state_after,
        "adapter_state_sha256_before": adapter_state_before,
        "adapter_state_sha256_after": adapter_state_after,
        "adapter_parameters_changed": adapter_state_after != adapter_state_before,
        "factual_gradient_isolated": True,
        "learned_object_output_authorized": False,
        "gradient_clip_scope": "independent_per_probability_head",
    }


__all__ = [
    "V8_ADAPTER_FORMAT",
    "V8_DURATION_RESIDUAL_MULTIPLIER",
    "V8_LOSS_CONTRACT",
    "V8_OBJECT_MODE",
    "V8_SCHEMA_VERSION",
    "V8DetachedStructuredAdapters",
    "V8StructuredAdapterConfig",
    "assert_optimizer_adapter_only",
    "compute_v8_adapter_loss",
    "frozen_tensor_mapping_sha256",
    "module_state_sha256",
    "train_v8_adapter_one_step",
    "validate_factual_adapter_inputs",
    "validate_schema5_adapter_batch",
]
