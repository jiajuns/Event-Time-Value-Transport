"""YOLO-assisted role graph -> GNN -> relations -> goal-conditioned evaluation.

No joint/action vectors, frame ordinal, video identity, task outcome, or labels enter
the network. Goal predicates are task inputs, not observed relations. This is an
observer, not an action-conditioned Q function or an embodiment action adapter.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .factorized_events import HEADS
from .factorized_observer import CausalWindows, load_dataset
from .semantic_cues import sha256
from .relational_graph import GRAPH_VERSION, NODE_FEATURES, EDGE_FEATURES

# A format bump is deliberate: the Event-State and Event-Value heads below are
# freshly initialized and therefore no CfC-era or relation-only checkpoint can
# be presented as an Event Observer result.
FORMAT = "umi_role_graph_event_observer_v2"
RELATIONS = {
    "held_by_actor": ("unheld", "held", "unknown"),
    "supported_by_target": ("unsupported", "supported", "unknown"),
    "in_target_region": ("outside", "inside", "unknown"),
}
GOAL_CLASSES = ("not_satisfied", "satisfied", "unknown")
# -1 means unconstrained, NOT an observed unknown. Region is not 3D containment.
GOALS = {
    "place_on": (0, 1, 1),
    "maintain_hold": (1, -1, -1),
    "release_outside": (0, 0, 0),
}
GRAPH_INPUTS = ("node_features", "node_images_uint8", "node_mask", "edge_features", "bindings")
EVENT_STATES = ("approach", "grasp", "lift", "transport", "align", "place", "unknown")


class EventValueCritic(nn.Module):
    """Value sidecar for semi-Markov event targets.

    It intentionally receives an Event-State representation rather than a
    simulator reward, task outcome, action, or future frame.  The πRL adapter
    owns the SMDP target and optimizer; keeping this module in the observer
    makes the visual contract explicit and keeps YOLOE frozen.
    """

    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                     nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, event_state):
        return self.network(event_state).squeeze(-1)


def goal_vector(name_or_values):
    values = GOALS[name_or_values] if isinstance(name_or_values, str) else tuple(name_or_values)
    if len(values) != len(RELATIONS) or any(v not in (-1, 0, 1) for v in values) or all(v == -1 for v in values):
        raise ValueError("goal must constrain at least one known relation; unknown/3D containment are not goals")
    return values


def goal_targets(relations, goals=None):
    """Conservative conjunction supervised from independently observed facts.

    A known contradiction proves non-satisfaction even when other facts are missing.
    Missing annotation (-1) is different from an explicitly observed unknown (2).
    """
    goals = np.asarray(list(GOALS.values()) if goals is None else goals, dtype=np.int64)
    output = np.full((len(relations), len(goals)), -1, np.int64)
    for j, goal in enumerate(goals):
        goal_vector(goal)
        needed = goal >= 0
        values = relations[:, needed]
        expected = goal[needed]
        contradict = ((values >= 0) & (values < 2) & (values != expected)).any(1)
        success = (values == expected).all(1)
        uncertain = (values == 2).any(1) & ~contradict
        output[contradict, j] = 0
        output[success, j] = 1
        output[uncertain, j] = 2
    return output


def attach_relation_labels(arrays, annotation_csv=None):
    labels = np.full((len(arrays["labels"]), len(RELATIONS)), -1, np.int64)
    # Existing holding labels were reviewed separately, not inferred from YOLO.
    labels[:, 0] = arrays["labels"][:, list(HEADS).index("holding")]
    if annotation_csv is None:
        return labels
    uids = set(arrays["attempt_uid"].tolist())
    occupied = np.zeros_like(labels, dtype=bool)
    with Path(annotation_csv).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = row["relation"]
            if name not in RELATIONS or name == "held_by_actor" or row["attempt_uid"] not in uids:
                raise ValueError("unknown relation/source or attempt to replace holding annotations")
            if row["reviewed"].lower() != "true":
                raise ValueError("relation supervision must have an explicit review")
            start, end = float(row["start_s"]), float(row["end_s"])
            if not np.isfinite([start, end]).all() or not 0 <= start < end:
                raise ValueError("invalid relation interval")
            column = list(RELATIONS).index(name)
            label = RELATIONS[name].index(row["label"])
            chosen = ((arrays["attempt_uid"] == row["attempt_uid"]) &
                      (arrays["elapsed_s"] >= start - 1e-7) & (arrays["elapsed_s"] < end - 1e-7))
            if not chosen.any() or occupied[chosen, column].any():
                raise ValueError("empty/overlapping relation annotation interval")
            labels[chosen, column] = label
            occupied[chosen, column] = True
    return labels


def load_relational_data(data, graph, annotations=None):
    arrays, contract = load_dataset(Path(data))
    with np.load(graph, allow_pickle=False) as archive:
        graphs = {key: archive[key] for key in GRAPH_INPUTS}
        graph_contract = json.loads(str(archive["contract_json"].item()))
        for key in ("attempt_uid", "query_id"):
            if not np.array_equal(archive[key], arrays[key]):
                raise ValueError("graph/data identity or ordering mismatch")
        if not np.allclose(archive["elapsed_s"], arrays["elapsed_s"], atol=1e-6, rtol=0):
            raise ValueError("graph/data timestamp mismatch")
    if (graph_contract.get("format") != GRAPH_VERSION or graph_contract.get("causal") is not True or
            graph_contract.get("labels_used_as_features") is not False or
            graph_contract.get("future_interpolation") is not False or
            graph_contract.get("node_features") != list(NODE_FEATURES) or
            graph_contract.get("edge_features") != list(EDGE_FEATURES)):
        raise ValueError("incompatible or noncausal graph contract")
    valid_source_hashes = {sha256(Path(data)), contract.get("split_override", {}).get("source_cache_sha256")}
    if graph_contract.get("source_data_sha256") not in valid_source_hashes:
        raise ValueError("graph cache was built from a different visual dataset")
    n, nodes = graphs["node_mask"].shape
    if n != len(arrays["labels"]) or graphs["node_mask"].dtype != np.bool_:
        raise ValueError("invalid graph node mask")
    if graphs["node_features"].shape[:2] != (n, nodes) or graphs["edge_features"].shape[:3] != (n, nodes, nodes):
        raise ValueError("invalid graph geometry shape")
    if graphs["node_images_uint8"].shape[:3] != (n, nodes, 3) or graphs["node_images_uint8"].dtype != np.uint8:
        raise ValueError("invalid graph ROI images")
    if graphs["bindings"].shape != (n, 3) or np.any((graphs["bindings"] < -1) | (graphs["bindings"] >= nodes)):
        raise ValueError("invalid role bindings")
    for key in ("node_features", "edge_features"):
        if not np.isfinite(graphs[key]).all():
            raise ValueError("nonfinite graph features")
    for row, binding in enumerate(graphs["bindings"]):
        valid = binding >= 0
        if np.any(~graphs["node_mask"][row, binding[valid]]):
            raise ValueError("role is bound to an invisible/padded node")
    arrays.update(graphs)
    arrays["relation_labels"] = attach_relation_labels(arrays, annotations)
    arrays["goal_labels"] = goal_targets(arrays["relation_labels"])
    contract = dict(contract, relational_graph=graph_contract,
                    graph_cache_sha256=sha256(Path(graph)),
                    relation_annotations_sha256=sha256(Path(annotations)) if annotations else None)
    return arrays, contract


class RelationalWindows(CausalWindows):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        index = result["index"]
        history = self.histories[index]
        for key in GRAPH_INPUTS:
            values = self.arrays[key]
            padded = np.full((self.window, *values.shape[1:]), -1 if key == "bindings" else 0, values.dtype)
            padded[:len(history)] = values[history]
            result[key] = torch.from_numpy(padded)
        result["relation_labels"] = torch.from_numpy(self.arrays["relation_labels"][index])
        result["goal_labels"] = torch.from_numpy(self.arrays["goal_labels"][index])
        return result


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in ("images", "dt", "mask", *GRAPH_INPUTS)}


class RelationalObserver(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=96, graph_hidden=48, graph_layers=2,
                 forecast_horizons_s=(.3, .6), use_graph=True):
        super().__init__()
        self.model_spec = dict(node_dim=node_dim, edge_dim=edge_dim, hidden=hidden,
                               graph_hidden=graph_hidden, graph_layers=graph_layers,
                               forecast_horizons_s=list(forecast_horizons_s), use_graph=use_graph)
        # Keep spatial layout: global average pooling alone loses tray-boundary detail.
        self.global_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(48 * 16, 64), nn.SiLU())
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, 2, 2), nn.GroupNorm(3, 12), nn.SiLU(),
            nn.Conv2d(12, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(96, 32), nn.SiLU())
        self.node_encoder = nn.Sequential(nn.Linear(node_dim + 32, graph_hidden), nn.LayerNorm(graph_hidden), nn.SiLU())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(graph_hidden * 2 + edge_dim, graph_hidden), nn.SiLU(),
                          nn.Linear(graph_hidden, graph_hidden)) for _ in range(graph_layers)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(graph_hidden * 2, graph_hidden), nn.LayerNorm(graph_hidden), nn.SiLU())
            for _ in range(graph_layers)])
        self.missing_role = nn.Parameter(torch.zeros(3, graph_hidden))
        self.fusion = nn.Sequential(nn.Linear(64 + graph_hidden * 4 + 3, hidden), nn.LayerNorm(hidden), nn.Tanh())
        # The observer is intentionally GNN-only.  A masked mean summarizes the
        # causal graph history and the last valid graph preserves the current
        # relation state; this MLP fuses the two without a recurrent/CfC cell.
        self.history_fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        # Role-Graph Event Observer outputs. Relation/event heads below remain
        # auxiliary supervision; they are not declared to be causal evidence.
        self.event_state_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.event_state_head = nn.Linear(hidden, len(EVENT_STATES))
        self.event_progress_head = nn.Linear(hidden, 1)
        self.event_boundary_head = nn.Linear(hidden, 1)
        self.event_uncertainty_head = nn.Linear(hidden, 1)
        self.event_value_critic = EventValueCritic(hidden)
        self.event_heads = nn.ModuleDict({name: nn.Linear(hidden, len(HEADS[name])) for name in ("phase", "transition")})
        self.relation_heads = nn.ModuleDict({name: nn.Linear(hidden, 3) for name in RELATIONS})
        self.forecast_heads = nn.ModuleDict({f"h{k}_{name}": nn.Linear(hidden, 3)
                                            for k, _ in enumerate(forecast_horizons_s) for name in RELATIONS})
        # A goal is a composable relation specification, never the video's outcome.
        self.goal_encoder = nn.Sequential(nn.Linear(9, 32), nn.SiLU())
        self.goal_head = nn.Sequential(nn.Linear(hidden + 32 + 9, 64), nn.SiLU(), nn.Linear(64, 3))

    def encode_graph(self, node_features, node_images_uint8, node_mask, edge_features, bindings):
        batch, nodes, _ = node_features.shape
        node_mask = node_mask.bool()
        roi = self.roi_encoder(node_images_uint8.float().reshape(batch * nodes, 3, *node_images_uint8.shape[-2:]) / 255.)
        x = self.node_encoder(torch.cat((roi.reshape(batch, nodes, -1), node_features), -1))
        x = x * node_mask[..., None]
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        pair_mask &= ~torch.eye(nodes, dtype=torch.bool, device=x.device)[None]
        for message, update in zip(self.messages, self.updates):
            left = x[:, :, None].expand(-1, -1, nodes, -1)
            right = x[:, None, :].expand(-1, nodes, -1, -1)
            messages = message(torch.cat((left, right, edge_features), -1))
            pooled = (messages * pair_mask[..., None]).sum(2) / pair_mask.sum(2).clamp_min(1)[..., None]
            x = (x + update(torch.cat((x, pooled), -1))) * node_mask[..., None]
        valid_bindings = bindings >= 0
        selected = torch.gather(x, 1, bindings.clamp_min(0)[..., None].expand(-1, -1, x.shape[-1]))
        observed = torch.gather(node_mask, 1, bindings.clamp_min(0))
        valid_bindings &= observed
        selected = torch.where(valid_bindings[..., None], selected, self.missing_role[None])
        pooled = x.sum(1) / node_mask.sum(1).clamp_min(1)[:, None]
        return torch.cat((selected.flatten(1), pooled, valid_bindings.float()), -1)

    def forward(self, images, dt, mask, node_features, node_images_uint8, node_mask, edge_features,
                bindings, *, goals=None, return_features=False):
        if images.ndim != 6 or images.shape[2:4] != (1, 3):
            raise ValueError("v3 public video input is single-camera [B,T,1,3,H,W]")
        batch, length = images.shape[:2]
        if mask.shape != (batch, length) or mask.dtype != torch.bool or not bool(mask.any(1).all()):
            raise ValueError("invalid causal history mask")
        if dt.shape != mask.shape or not bool(torch.isfinite(dt[mask]).all()) or bool((dt[mask] < 0).any()):
            raise ValueError("invalid time intervals")
        # Time intervals remain validated for dataset/ABI auditing, but are not
        # consumed by a temporal network. Invalid frames are excluded by mask.
        rgb = torch.where(mask[..., None, None, None, None], images.float(), 0.)
        global_features = self.global_encoder(rgb.reshape(batch * length, 3, *images.shape[-2:]) / 255.)
        graph = self.encode_graph(
            node_features.flatten(0, 1), node_images_uint8.flatten(0, 1),
            node_mask.flatten(0, 1), edge_features.flatten(0, 1), bindings.flatten(0, 1))
        if not self.model_spec["use_graph"]:
            graph = torch.zeros_like(graph)
        inputs = self.fusion(torch.cat((global_features, graph), -1)).reshape(batch, length, -1)
        valid = mask[..., None]
        pooled = (inputs * valid).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        last_index = torch.where(mask, torch.arange(length, device=mask.device)[None], -1).max(1).values
        last = inputs[torch.arange(batch, device=images.device), last_index]
        hidden = self.history_fusion(torch.cat((last, pooled), -1))
        event_state = self.event_state_encoder(hidden)
        relations = {name: head(hidden) for name, head in self.relation_heads.items()}
        probabilities = torch.cat([relations[name].softmax(-1) for name in RELATIONS], -1)
        goals = torch.as_tensor(list(GOALS.values()) if goals is None else goals, device=hidden.device, dtype=torch.long)
        if goals.ndim != 2 or goals.shape[1] != 3 or bool(((goals < -1) | (goals > 1)).any()) or bool((goals == -1).all(1).any()):
            raise ValueError("invalid relation goal specification")
        goal_features = self.goal_encoder(F.one_hot(goals + 1, num_classes=3).float().flatten(1))
        count = len(goals)
        goal_logits = self.goal_head(torch.cat((hidden[:, None].expand(-1, count, -1),
                                                probabilities[:, None].expand(-1, count, -1),
                                                goal_features[None].expand(batch, -1, -1)), -1))
        result = dict(events={name: head(hidden) for name, head in self.event_heads.items()},
                      relations=relations, goals=goal_logits,
                      forecast={key: head(hidden) for key, head in self.forecast_heads.items()},
                      event_state=event_state,
                      event_posterior=self.event_state_head(event_state),
                      event_progress=self.event_progress_head(event_state).squeeze(-1).sigmoid(),
                      event_boundary_logit=self.event_boundary_head(event_state).squeeze(-1),
                      event_uncertainty=F.softplus(self.event_uncertainty_head(event_state).squeeze(-1)),
                      event_value=self.event_value_critic(event_state))
        if return_features:
            result["history_features"] = hidden
        return result


def load_observer(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != FORMAT or saved.get("labels_are_model_inputs") is not False:
        raise ValueError("not a versioned YOLO-assisted GNN checkpoint")
    if saved.get("relation_schema") != {key: list(value) for key, value in RELATIONS.items()}:
        raise ValueError("relation schema mismatch")
    model = RelationalObserver(**saved["model_spec"])
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, saved


def confirm_relations(times, probabilities, *, confidence=.6, frames=3):
    """Confirm facts causally. Unknown clears evidence; no future smoothing."""
    if not 0 <= confidence <= 1 or frames < 2:
        raise ValueError("invalid confidence/confirmation length")
    n = len(times)
    stable = np.full((n, len(RELATIONS)), 2, np.int64)
    previous = np.full(len(RELATIONS), 2, np.int64)
    counts = np.zeros(len(RELATIONS), np.int64)
    starts = np.zeros(len(RELATIONS), float)
    for i, timestamp in enumerate(times):
        if i and timestamp <= times[i - 1]:
            raise ValueError("timestamps must increase")
        if i and timestamp - times[i - 1] > .151:
            counts[:] = 0
        for j, name in enumerate(RELATIONS):
            p = probabilities[name][i]
            label = int(p.argmax()) if p.max() >= confidence else 2
            if label == 2:
                counts[j] = 0
            else:
                if label != previous[j] or counts[j] == 0:
                    counts[j] = 0
                    starts[j] = timestamp
                counts[j] += 1
                if counts[j] >= frames and timestamp - starts[j] >= .18 - 1e-7:
                    stable[i, j] = label
            previous[j] = label
    return stable


def publish_goal(stable_relations, goal_probabilities, goal, *, confidence=.6):
    """A high goal-network score cannot override contradictory/unknown relations.

    The result is goal satisfaction evidence, not a calibrated value or advantage.
    """
    goal = np.asarray(goal_vector(goal))
    needed = goal >= 0
    values = stable_relations[:, needed]
    expected = goal[needed]
    contradiction = ((values < 2) & (values != expected)).any(1)
    satisfied = (values == expected).all(1)
    predicted = goal_probabilities.argmax(1)
    confident = goal_probabilities.max(1) >= confidence
    result = np.full(len(values), 2, np.int64)
    result[contradiction] = 0
    result[satisfied & confident & (predicted == 1)] = 1
    return result
