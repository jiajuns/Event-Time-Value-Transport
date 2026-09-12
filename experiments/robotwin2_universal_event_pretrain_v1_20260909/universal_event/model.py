"""Goal- and action-conditioned graph/CfC event world model.

This is the shared, embodiment-neutral continuation of Event V4.  V4's
96-dimensional elapsed-time CfC can be loaded exactly.  Robot-specific joint
vectors are intentionally absent: candidate chunks must first be converted to
14-D left/right SE(3)+gripper effects.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .schema import (
    CANONICAL_ACTION_DIM,
    EDGE_FEATURE_DIM,
    EVENT_NAMES,
    MAX_NODES,
    NODE_FEATURE_DIM,
    NODE_TYPES,
    RELATION_NAMES,
)


FORMAT = "etsf_robotwin2_universal_graph_cfc_action_world_model_v1"


@dataclasses.dataclass(frozen=True)
class UniversalEventWorldModelConfig:
    node_feature_dim: int = NODE_FEATURE_DIM
    edge_feature_dim: int = EDGE_FEATURE_DIM
    node_hidden: int = 48
    hidden: int = 96
    graph_layers: int = 3
    language_embed: int = 48
    maximum_language_bytes: int = 192
    action_dim: int = CANONICAL_ACTION_DIM
    dropout: float = 0.1


class ElapsedTimeCfCCell(nn.Module):
    """The exact lightweight CfC parameterization used by Event V4."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proposal = nn.Linear(2 * hidden_size, hidden_size)
        self.time_constant = nn.Linear(2 * hidden_size, hidden_size)
        self.mix = nn.Linear(2 * hidden_size, hidden_size)

    def forward(
        self, value: torch.Tensor, hidden: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        if dt.ndim != 1 or dt.shape[0] != value.shape[0]:
            raise ValueError("CfC dt must be [B]")
        joined = torch.cat((value, hidden), dim=-1)
        target = torch.tanh(self.proposal(joined))
        tau = F.softplus(self.time_constant(joined)) + 1e-3
        decay = torch.exp(-dt.clamp_min(0.0)[:, None] / tau)
        gate = torch.sigmoid(self.mix(joined))
        return gate * (decay * hidden + (1.0 - decay) * target) + (1.0 - gate) * hidden


class ByteLanguageEncoder(nn.Module):
    """Small byte-level language encoder with no task-id lookup table.

    Bytes make the adapter closed-vocabulary-free and allow both English and
    Chinese instructions.  Goal-graph reconstruction keeps language tied to
    relations instead of episode time or a task index.
    """

    PAD = 0
    BOS = 257
    EOS = 258

    def __init__(self, embed_dim: int, hidden: int, maximum_bytes: int) -> None:
        super().__init__()
        self.maximum_bytes = maximum_bytes
        self.embedding = nn.Embedding(259, embed_dim, padding_idx=self.PAD)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def tokenize(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("instruction must be a nonempty string")
            body = list(text.strip().lower().encode("utf-8")[: self.maximum_bytes - 2])
            rows.append([self.BOS, *[value + 1 for value in body], self.EOS])
        width = max(map(len, rows))
        tokens = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        lengths = torch.empty(len(rows), dtype=torch.long, device=device)
        for index, row in enumerate(rows):
            tokens[index, : len(row)] = torch.tensor(row, device=device)
            lengths[index] = len(row)
        return tokens, lengths

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens, lengths = self.tokenize(texts, device)
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _output, hidden = self.gru(packed)
        return self.norm(hidden[-1])


class TypedSceneGraphEncoder(nn.Module):
    def __init__(self, config: UniversalEventWorldModelConfig) -> None:
        super().__init__()
        h = config.node_hidden
        self.type_embedding = nn.Embedding(len(NODE_TYPES), 16)
        self.node = nn.Sequential(
            nn.Linear(config.node_feature_dim + 16, h), nn.LayerNorm(h), nn.SiLU()
        )
        self.messages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * h + config.edge_feature_dim, h),
                    nn.SiLU(),
                    nn.Linear(h, h),
                )
                for _ in range(config.graph_layers)
            ]
        )
        self.updates = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.SiLU())
             for _ in range(config.graph_layers)]
        )
        self.pool = nn.Sequential(nn.Linear(2 * h, config.hidden), nn.LayerNorm(config.hidden), nn.Tanh())

    def forward(
        self,
        node_features: torch.Tensor,
        node_types: torch.Tensor,
        node_mask: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_features.ndim != 3 or node_features.shape[-1] != NODE_FEATURE_DIM:
            raise ValueError("node_features must be [B,N,24]")
        batch, nodes = node_features.shape[:2]
        if nodes != MAX_NODES or node_types.shape != (batch, nodes) or node_mask.shape != (batch, nodes):
            raise ValueError("node graph shape mismatch")
        if edge_features.shape != (batch, nodes, nodes, EDGE_FEATURE_DIM):
            raise ValueError("edge graph shape mismatch")
        mask = node_mask.bool()
        if not bool(mask.any(1).all()):
            raise ValueError("each scene graph needs at least one node")
        safe_types = node_types.clamp(0, len(NODE_TYPES) - 1)
        x = self.node(torch.cat((node_features, self.type_embedding(safe_types)), -1))
        x = x * mask[..., None]
        eye = torch.eye(nodes, dtype=torch.bool, device=x.device)[None]
        pair_mask = mask[:, :, None] & mask[:, None, :] & ~eye
        for message, update in zip(self.messages, self.updates):
            source = x[:, :, None].expand(-1, -1, nodes, -1)
            target = x[:, None, :].expand(-1, nodes, -1, -1)
            msg = message(torch.cat((source, target, edge_features), -1))
            incoming = (msg * pair_mask[..., None]).sum(1)
            incoming = incoming / pair_mask.sum(1).clamp_min(1)[..., None]
            x = (x + update(torch.cat((x, incoming), -1))) * mask[..., None]
        mean = x.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = x.masked_fill(~mask[..., None], -torch.inf).amax(1)
        return self.pool(torch.cat((mean, maximum), -1)), x


class CanonicalActionChunkEncoder(nn.Module):
    """Encode body-independent executed/planned EE effects and their timing."""

    def __init__(self, config: UniversalEventWorldModelConfig) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.action_dim + 1, config.hidden), nn.SiLU(), nn.LayerNorm(config.hidden)
        )
        self.cell = nn.GRUCell(config.hidden, config.hidden)
        self.output = nn.LayerNorm(config.hidden)

    def forward(
        self, actions: torch.Tensor, mask: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != CANONICAL_ACTION_DIM:
            raise ValueError("canonical action chunk must be [B,H,14]")
        if mask.shape != actions.shape[:2] or dt.shape != actions.shape[:2]:
            raise ValueError("action mask/dt shape mismatch")
        if not bool(mask.bool().any(1).all()):
            raise ValueError("every action chunk must contain a valid step")
        value = self.input(torch.cat((actions, dt[..., None]), -1))
        hidden = actions.new_zeros(actions.shape[0], value.shape[-1])
        for step in range(actions.shape[1]):
            proposal = self.cell(value[:, step], hidden)
            hidden = torch.where(mask[:, step, None].bool(), proposal, hidden)
        return self.output(hidden)


class RelationDecoder(nn.Module):
    def __init__(self, node_hidden: int, context_hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * node_hidden + context_hidden, context_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden, len(RELATION_NAMES)),
        )

    def forward(self, nodes: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, count, hidden = nodes.shape
        source = nodes[:, :, None].expand(-1, -1, count, -1)
        target = nodes[:, None, :].expand(-1, count, -1, -1)
        shared = context[:, None, None].expand(-1, count, count, -1)
        # [B,R,N,N] is convenient for goal-graph masks.
        return self.network(torch.cat((source, target, shared), -1)).permute(0, 3, 1, 2)


class UniversalEventWorldModel(nn.Module):
    """Graph state + CfC history + language goal + action chunk -> consequence."""

    def __init__(self, config: UniversalEventWorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or UniversalEventWorldModelConfig()
        if self.config.hidden != 96:
            raise ValueError("hidden=96 is required for exact Event V4 CfC transfer")
        self.graph = TypedSceneGraphEncoder(self.config)
        self.language = ByteLanguageEncoder(
            self.config.language_embed, self.config.hidden, self.config.maximum_language_bytes
        )
        self.goal_graph = nn.Sequential(
            nn.Linear(2 * len(RELATION_NAMES), self.config.hidden),
            nn.SiLU(), nn.LayerNorm(self.config.hidden),
        )
        self.frame_fusion = nn.Sequential(
            nn.Linear(2 * self.config.hidden, self.config.hidden), nn.LayerNorm(self.config.hidden), nn.Tanh()
        )
        self.initial = nn.Linear(self.config.hidden, self.config.hidden)
        self.cfc = ElapsedTimeCfCCell(self.config.hidden)
        self.action = CanonicalActionChunkEncoder(self.config)
        self.transition = nn.Sequential(
            nn.Linear(4 * self.config.hidden, 2 * self.config.hidden),
            nn.SiLU(), nn.Dropout(self.config.dropout),
            nn.Linear(2 * self.config.hidden, self.config.hidden), nn.LayerNorm(self.config.hidden),
        )
        self.node_transition = nn.Sequential(
            nn.Linear(self.config.node_hidden + self.config.hidden, self.config.node_hidden),
            nn.SiLU(), nn.Linear(self.config.node_hidden, self.config.node_hidden),
        )
        self.current_relations = RelationDecoder(
            self.config.node_hidden, self.config.hidden, self.config.dropout
        )
        self.future_relations = RelationDecoder(
            self.config.node_hidden, self.config.hidden, self.config.dropout
        )
        self.node_delta = nn.Sequential(
            nn.Linear(self.config.node_hidden + self.config.hidden, self.config.node_hidden),
            nn.SiLU(), nn.Linear(self.config.node_hidden, 8),
        )
        self.event_head = nn.Linear(self.config.hidden, len(EVENT_NAMES))
        self.language_goal_head = nn.Linear(self.config.hidden, 2 * len(RELATION_NAMES))
        self.success_residual = nn.Sequential(
            nn.Linear(2 * self.config.hidden, self.config.hidden), nn.SiLU(), nn.Linear(self.config.hidden, 1)
        )
        # Kept solely to transfer and regularize the three V4 relation heads.
        self.legacy_relation_heads = nn.ModuleDict(
            {name: nn.Linear(self.config.hidden, 3) for name in
             ("held_by_actor", "supported_by_target", "in_target_region")}
        )

    @staticmethod
    def _goal_summary(goal: torch.Tensor, goal_mask: torch.Tensor) -> torch.Tensor:
        if goal.shape != goal_mask.shape or goal.ndim != 4:
            raise ValueError("goal and goal_mask must be [B,R,N,N]")
        positive = ((goal > 0.5) & goal_mask.bool()).float().sum((2, 3))
        negative = ((goal <= 0.5) & goal_mask.bool()).float().sum((2, 3))
        scale = goal_mask.float().sum((2, 3)).clamp_min(1.0)
        return torch.cat((positive / scale, negative / scale), -1)

    @staticmethod
    def goal_probability(
        relation_logits: torch.Tensor, goal: torch.Tensor, goal_mask: torch.Tensor
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(relation_logits)
        desired = torch.where(goal > 0.5, probabilities, 1.0 - probabilities)
        mask = goal_mask.bool()
        if not bool(mask.flatten(1).any(1).all()):
            raise ValueError("every task goal must constrain at least one relation")
        mean_log = (torch.log(desired.clamp_min(1e-6)) * mask).sum((1, 2, 3))
        mean_log = mean_log / mask.sum((1, 2, 3)).clamp_min(1)
        return torch.exp(mean_log)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        nodes = batch["node_features"]
        node_types = batch["node_types"]
        node_mask = batch["node_mask"]
        edges = batch["edge_features"]
        history_mask = batch["history_mask"].bool()
        history_dt = batch["history_dt"]
        if nodes.ndim != 4:
            raise ValueError("node history must be [B,T,N,F]")
        batch_size, length = nodes.shape[:2]
        if history_mask.shape != (batch_size, length) or history_dt.shape != (batch_size, length):
            raise ValueError("history mask/dt mismatch")
        if not bool(history_mask.any(1).all()):
            raise ValueError("each row needs observed history")
        instructions = list(batch["instruction"])
        if len(instructions) != batch_size:
            raise ValueError("instruction batch mismatch")
        language = self.language(instructions, nodes.device)
        goal_summary = self._goal_summary(batch["goal"], batch["goal_mask"])
        goal_features = self.goal_graph(goal_summary)

        flat_scene, flat_nodes = self.graph(
            nodes.flatten(0, 1), node_types.flatten(0, 1),
            node_mask.flatten(0, 1), edges.flatten(0, 1),
        )
        frames = flat_scene.reshape(batch_size, length, -1)
        frame_nodes = flat_nodes.reshape(batch_size, length, MAX_NODES, -1)
        fused = self.frame_fusion(torch.cat((frames, goal_features[:, None].expand(-1, length, -1)), -1))
        hidden = nodes.new_zeros(batch_size, self.config.hidden)
        seen = torch.zeros(batch_size, dtype=torch.bool, device=nodes.device)
        for step in range(length):
            proposal = self.cfc(fused[:, step], hidden, torch.where(history_mask[:, step], history_dt[:, step], 0.0))
            proposal = torch.where(seen[:, None], proposal, torch.tanh(self.initial(fused[:, step])))
            hidden = torch.where(history_mask[:, step, None], proposal, hidden)
            # Do not mutate a condition tensor retained by ``torch.where`` for
            # backward; an in-place OR changes its autograd version on the next
            # recurrent step.
            seen = seen | history_mask[:, step]
        last_index = history_mask.long().sum(1) - 1
        last_nodes = frame_nodes[torch.arange(batch_size, device=nodes.device), last_index]

        action = self.action(
            batch["actions"], batch["action_mask"].bool(), batch["action_dt"]
        )
        future = hidden + self.transition(torch.cat((hidden, action, language, goal_features), -1))
        future_nodes = last_nodes + self.node_transition(
            torch.cat((last_nodes, future[:, None].expand(-1, MAX_NODES, -1)), -1)
        )
        current_relation_logits = self.current_relations(last_nodes, hidden)
        future_relation_logits = self.future_relations(future_nodes, future)
        goal_probability = self.goal_probability(
            future_relation_logits, batch["goal"], batch["goal_mask"]
        )
        base_logit = torch.logit(goal_probability.clamp(1e-5, 1.0 - 1e-5))
        learned_residual = 2.0 * torch.tanh(
            self.success_residual(torch.cat((future, language), -1)).squeeze(-1)
        )
        node_delta = self.node_delta(
            torch.cat((last_nodes, future[:, None].expand(-1, MAX_NODES, -1)), -1)
        )
        return {
            "history_features": hidden,
            "language_features": language,
            "goal_features": goal_features,
            "action_features": action,
            "future_features": future,
            "current_relation_logits": current_relation_logits,
            "future_relation_logits": future_relation_logits,
            "future_node_delta": node_delta,
            "event_logits": self.event_head(future),
            "language_goal_logits": self.language_goal_head(language).reshape(
                batch_size, 2, len(RELATION_NAMES)
            ),
            "goal_probability": goal_probability,
            "success_logit": base_logit + learned_residual,
            "legacy_relations": {
                name: head(hidden) for name, head in self.legacy_relation_heads.items()
            },
        }

    def load_event_v4_initialization(self, checkpoint: str | Path) -> dict[str, Any]:
        """Load only shape- and meaning-compatible Event V4 parameters."""

        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if saved.get("format") not in {"umi_relational_graph_cfc_v3", "umi_relational_evidence_cfc_v4"}:
            raise ValueError("checkpoint is not relational Event V3/V4")
        state = saved.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("Event V4 checkpoint has no model state")
        copied: list[str] = []
        own = self.state_dict()
        for suffix in ("proposal.weight", "proposal.bias", "time_constant.weight",
                       "time_constant.bias", "mix.weight", "mix.bias"):
            source, target = f"cfc.{suffix}", f"cfc.{suffix}"
            if source not in state or state[source].shape != own[target].shape:
                raise ValueError(f"Event V4 CfC tensor mismatch: {source}")
            own[target].copy_(state[source])
            copied.append(target)
        mappings = {
            "held_by_actor": "held_by_actor",
            "supported_by_target": "supported_by_target",
            "in_target_region": "in_target_region",
        }
        for target_name, source_name in mappings.items():
            for parameter in ("weight", "bias"):
                candidates = (
                    f"current_heads.{source_name}.{parameter}",
                    f"relation_heads.{source_name}.{parameter}",
                )
                source = next((name for name in candidates if name in state), None)
                target = f"legacy_relation_heads.{target_name}.{parameter}"
                if source is not None and state[source].shape == own[target].shape:
                    own[target].copy_(state[source])
                    copied.append(target)
        self.load_state_dict(own, strict=True)
        return {
            "source_format": saved["format"],
            "source_selected_step": saved.get("selected_step"),
            "copied_tensors": copied,
            "copied_tensor_count": len(copied),
            "policy": "only_exact_shape_and_semantic_match",
        }


def parameter_inventory(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(value.numel() for value in model.parameters()),
        "trainable": sum(value.numel() for value in model.parameters() if value.requires_grad),
    }
