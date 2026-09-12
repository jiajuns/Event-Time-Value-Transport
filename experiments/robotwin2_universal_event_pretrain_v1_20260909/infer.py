#!/usr/bin/env python3
"""Policy-agnostic candidate chunk scorer for the universal event model."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

import torch

from universal_event.model import FORMAT, UniversalEventWorldModel, UniversalEventWorldModelConfig


class UniversalEventChunkScorer:
    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if saved.get("format") != FORMAT:
            raise ValueError("not a universal event world-model checkpoint")
        config = UniversalEventWorldModelConfig(**saved["config"])
        self.model = UniversalEventWorldModel(config)
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.to(device).eval(); self.device = torch.device(device)

    @torch.inference_mode()
    def score(self, root: Mapping[str, Any], actions: torch.Tensor,
              action_mask: torch.Tensor, action_dt: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score N candidate canonical action chunks from one observed root.

        ``root`` contains one graph history and one language/goal graph.  It is
        repeated across candidates; only candidate actions differ.  Raw robot
        joint vectors are rejected by the required trailing action dimension 14.
        """
        if actions.ndim != 3 or actions.shape[-1] != 14:
            raise ValueError("candidate actions must be [N,H,14] canonical EE effects")
        count = len(actions)
        batch: dict[str, Any] = {}
        for key, value in root.items():
            if key == "instruction":
                if not isinstance(value, str): raise ValueError("root instruction must be one string")
                batch[key] = [value] * count
            elif isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0).expand(count, *value.shape).to(self.device)
        batch.update(actions=actions.to(self.device), action_mask=action_mask.to(self.device),
                     action_dt=action_dt.to(self.device))
        output = self.model(batch)
        # Goal probability is relation-compositional.  Success is a calibrated
        # residual head; risk explicitly penalizes drop probability.
        drop = torch.sigmoid(output["event_logits"][:, 5])
        utility = .7 * output["goal_probability"] + .3 * torch.sigmoid(output["success_logit"]) - .35 * drop
        return {
            "utility": utility,
            "goal_probability": output["goal_probability"],
            "success_probability": torch.sigmoid(output["success_logit"]),
            "drop_probability": drop,
            "best_index": utility.argmax(),
            "future_relation_probability": torch.sigmoid(output["future_relation_logits"]),
        }


__all__ = ["UniversalEventChunkScorer"]
