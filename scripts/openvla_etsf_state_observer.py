#!/usr/bin/env python3
"""Non-privileged OpenVLA-hidden observer for structured ETSF inputs.

The observer is a separate supervised artifact.  It predicts the *current*
dynamic event and predicate vector from the policy hidden at a query point; it
does not modify an already frozen event-world-model checkpoint or its scoring
and guard.  Newly trained artifacts are monitor-only until an explicit,
independent calibration/promotion artifact enables reranking.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from openvla_etsf_event_world_model import EventWorldModelConfig


FORMAT = "etsf_state_hidden_event_predicate_observer_v1"
SOURCE = "state_hidden_event_predicate_observer_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_equivalent(left: Any, right: Any) -> bool:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): plain(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(child) for child in value]
        return value

    return json.dumps(plain(left), sort_keys=True) == json.dumps(
        plain(right), sort_keys=True
    )


@dataclass(frozen=True)
class StateObserverConfig:
    state_input_dim: int = 4096
    hidden_dim: int = 96
    event_names: tuple[str, ...] = ("e0", "e12", "e3", "e4", "eK")
    predicate_names: tuple[str, ...] = (
        "moved",
        "lifted",
        "near_goal",
        "stationary",
        "success",
    )
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.state_input_dim < 1 or self.hidden_dim < 4:
            raise ValueError("observer dimensions are invalid")
        if not self.event_names or not self.predicate_names:
            raise ValueError("observer vocabularies must be non-empty")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("observer dropout must lie in [0,1)")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateObserverConfig":
        values = dict(value)
        for key in ("event_names", "predicate_names"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class ObserverPrediction:
    current_event_id: torch.Tensor
    current_event_probability: torch.Tensor
    current_predicates: torch.Tensor
    current_predicate_probability: torch.Tensor
    confidence: torch.Tensor
    valid_for_rerank: torch.Tensor


class StateHiddenEventPredicateObserver(nn.Module):
    """Small supervised current-state classifier with frozen calibration."""

    def __init__(
        self,
        config: StateObserverConfig,
        *,
        contract: Mapping[str, Any],
        calibration: Mapping[str, Any],
        deployment: Mapping[str, Any],
        artifact_sha256: str = "unserialized",
    ) -> None:
        super().__init__()
        self.config = config
        self.contract = dict(contract)
        self.calibration = dict(calibration)
        self.deployment = dict(deployment)
        self.artifact_sha256 = str(artifact_sha256)
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.state_input_dim),
            nn.Linear(config.state_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.event_head = nn.Linear(config.hidden_dim, len(config.event_names))
        self.predicate_head = nn.Linear(
            config.hidden_dim, len(config.predicate_names)
        )
        self._validate_frozen_metadata()

    def _validate_frozen_metadata(self) -> None:
        if self.contract.get("source") != SOURCE:
            raise ValueError("observer contract has an unsupported source")
        if self.contract.get("state_source") != "openvla_hidden_at_query":
            raise ValueError("observer contract has an unsupported hidden source")
        if self.contract.get("label_derivation") != (
            "derive_atomic_predicates_v1_plus_dynamic_event_ids_v1"
        ):
            raise ValueError("observer label derivation contract differs")
        if tuple(self.contract.get("event_names", ())) != self.config.event_names:
            raise ValueError("observer event vocabulary mirror mismatch")
        if tuple(self.contract.get("predicate_names", ())) != self.config.predicate_names:
            raise ValueError("observer predicate vocabulary mirror mismatch")
        digest = str(self.contract.get("event_spec_sha256", ""))
        if len(digest) != 64:
            raise ValueError("observer contract lacks event-spec SHA256")
        temperature = float(self.calibration.get("event_temperature", 0.0))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("observer event temperature must be positive")
        thresholds = self.calibration.get("predicate_thresholds")
        if not isinstance(thresholds, Sequence) or isinstance(
            thresholds, (str, bytes)
        ):
            raise ValueError("observer predicate thresholds must be a sequence")
        if len(thresholds) != len(self.config.predicate_names) or any(
            not 0.0 < float(value) < 1.0 for value in thresholds
        ):
            raise ValueError("observer predicate thresholds are invalid")
        confidence = float(self.calibration.get("minimum_joint_confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("observer confidence gate must lie in [0,1]")
        if self.deployment.get("rerank_enabled") is True:
            digest = self.artifact_sha256
            if len(digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in digest
            ):
                raise ValueError(
                    "observer rerank enablement requires a serialized checkpoint SHA256"
                )
            state_contracts = self.contract.get("state_contracts")
            if not isinstance(state_contracts, Mapping) or not state_contracts:
                raise ValueError(
                    "observer rerank enablement lacks a frozen state representation"
                )
            if self.deployment.get("promotion_status") != (
                "independent_validation_calibrated_and_explicitly_promoted"
            ):
                raise ValueError("observer rerank enablement lacks promotion evidence")
            if self.calibration.get("selection_data") != (
                "independent_observer_calibration_no_world_model_sealed_test"
            ):
                raise ValueError("observer rerank calibration used an invalid split")

    @property
    def rerank_enabled(self) -> bool:
        return self.deployment.get("rerank_enabled") is True

    @property
    def calibration_id(self) -> str:
        value = self.calibration.get("calibration_id")
        return str(value) if value else self.artifact_sha256

    def forward(
        self,
        hidden: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim not in (2, 3) or hidden.shape[-1] != self.config.state_input_dim:
            raise ValueError(
                "observer hidden must be [B,D] or [B,T,D] in the frozen state space"
            )
        if hidden.ndim == 3:
            batch, steps = hidden.shape[:2]
            if history_mask is None:
                history_mask = torch.ones(
                    (batch, steps), dtype=torch.bool, device=hidden.device
                )
            if history_mask.shape != (batch, steps):
                raise ValueError("observer history_mask shape differs from hidden history")
            mask = history_mask.to(device=hidden.device, dtype=torch.bool)
            if bool((~mask.any(1)).any()):
                raise ValueError("every observer history needs a valid hidden state")
            positions = torch.arange(steps, device=hidden.device)[None].expand(
                batch, -1
            )
            last = positions.masked_fill(~mask, -1).amax(1)
            hidden = hidden[torch.arange(batch, device=hidden.device), last]
        elif history_mask is not None:
            expected = (hidden.shape[0], 1)
            if history_mask.shape != expected or bool(
                (~history_mask.to(torch.bool).any(1)).any()
            ):
                raise ValueError(f"single-state observer history_mask must be {expected}")
        feature = self.encoder(hidden.float())
        return {
            "event_logits": self.event_head(feature),
            "predicate_logits": self.predicate_head(feature),
        }

    @torch.inference_mode()
    def observe(
        self,
        hidden: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> ObserverPrediction:
        output = self.forward(hidden, history_mask)
        temperature = float(self.calibration["event_temperature"])
        event_probability = torch.softmax(output["event_logits"] / temperature, -1)
        predicate_probability = torch.sigmoid(output["predicate_logits"])
        thresholds = predicate_probability.new_tensor(
            self.calibration["predicate_thresholds"]
        )
        predicates = (predicate_probability >= thresholds).to(
            predicate_probability.dtype
        )
        event_confidence = event_probability.max(-1).values
        predicate_confidence = torch.maximum(
            predicate_probability, 1.0 - predicate_probability
        ).amin(-1)
        confidence = torch.minimum(event_confidence, predicate_confidence)
        valid = confidence >= float(self.calibration["minimum_joint_confidence"])
        valid = valid & self.rerank_enabled
        return ObserverPrediction(
            current_event_id=event_probability.argmax(-1),
            current_event_probability=event_probability,
            current_predicates=predicates,
            current_predicate_probability=predicate_probability,
            confidence=confidence,
            valid_for_rerank=valid,
        )

    def validate_for_world_model(
        self,
        config: EventWorldModelConfig,
        world_contract: Mapping[str, Any],
        predicate_contract: Mapping[str, Any],
    ) -> None:
        if not config.structured_events:
            raise ValueError("state observer may attach only to a structured checkpoint")
        if self.config.state_input_dim != config.state_input_dim:
            raise ValueError("observer/world-model hidden dimensions differ")
        if self.config.event_names != config.event_names:
            raise ValueError("observer/world-model event vocabularies differ")
        if self.config.predicate_names != config.predicate_names:
            raise ValueError("observer/world-model predicate vocabularies differ")
        expected_event_spec = str(predicate_contract.get("event_spec_sha256", ""))
        if str(self.contract.get("event_spec_sha256", "")) != expected_event_spec:
            raise ValueError("observer/world-model event-spec provenance differs")
        if tuple(predicate_contract.get("names", ())) != config.predicate_names:
            raise ValueError("world-model predicate contract vocabulary differs")
        if self.contract.get("policy_to_id") != world_contract.get("policy_to_id"):
            raise ValueError("observer/world-model policy registration differs")
        if self.contract.get("state_contracts", {}) != world_contract.get(
            "state_contracts", {}
        ):
            raise ValueError("observer/world-model state representations differ")
        observer_bodies = self.contract.get("body_to_id")
        world_bodies = world_contract.get("body_to_id")
        if observer_bodies is not None and observer_bodies != world_bodies:
            raise ValueError("observer/world-model embodiment registration differs")

    def matches_state_provenance(
        self,
        *,
        source: str | None,
        artifact_sha256: str | None,
        calibration_id: str | None,
    ) -> bool:
        return (
            source == SOURCE
            and artifact_sha256 == self.artifact_sha256
            and calibration_id == self.calibration_id
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        verify_sha256: bool = True,
    ) -> "StateHiddenEventPredicateObserver":
        path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported state observer manifest format")
        artifact = manifest.get("checkpoint")
        if not isinstance(artifact, Mapping) or not artifact.get("path"):
            raise ValueError("state observer manifest lacks checkpoint provenance")
        recorded = Path(str(artifact["path"])).expanduser()
        portable = path.parent / recorded.name
        checkpoint_path = recorded if recorded.is_file() else portable
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        digest = sha256(checkpoint_path)
        if verify_sha256 and digest != str(artifact.get("sha256", "")):
            raise ValueError("state observer checkpoint SHA256 mismatch")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("format") != FORMAT:
            raise ValueError("state observer checkpoint format mismatch")
        for key in ("config", "contract", "calibration", "deployment"):
            if not _json_equivalent(manifest.get(key), checkpoint.get(key)):
                raise ValueError(f"state observer manifest/checkpoint {key} mismatch")
        config = StateObserverConfig.from_dict(manifest["config"])
        observer = cls(
            config,
            contract=manifest["contract"],
            calibration=manifest["calibration"],
            deployment=manifest["deployment"],
            artifact_sha256=digest,
        )
        observer.load_state_dict(checkpoint["model"], strict=True)
        observer.to(device).eval()
        for parameter in observer.parameters():
            parameter.requires_grad_(False)
        return observer


def observer_artifact_payload(
    observer: StateHiddenEventPredicateObserver,
) -> dict[str, Any]:
    """Build the mirrored metadata portion used by trainer/tests."""

    return {
        "format": FORMAT,
        "config": dataclasses.asdict(observer.config),
        "contract": dict(observer.contract),
        "calibration": dict(observer.calibration),
        "deployment": dict(observer.deployment),
    }


__all__ = [
    "FORMAT",
    "SOURCE",
    "ObserverPrediction",
    "StateHiddenEventPredicateObserver",
    "StateObserverConfig",
    "observer_artifact_payload",
    "sha256",
]
