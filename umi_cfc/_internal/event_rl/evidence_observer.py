"""V4: causal graph beliefs, visual evidence, and masked-history imagination.

Missing evidence does not erase beliefs. Published confirmation is separate from
latent inference, and neither is a calibrated RL value. No joint/action inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .relational_observer import (
    RelationalObserver, RelationalWindows, GRAPH_INPUTS, RELATIONS, GOALS,
    load_relational_data, goal_targets, confirm_relations, publish_goal,
)
from .relational_visual import VISUAL_VERSION, PAIR_FEATURE_NAMES as PAIR_FEATURES
from .semantic_cues import sha256

FORMAT = 'umi_yolo_relational_evidence_gnn_v1'
VISUAL_INPUTS = ('pair_images_uint8', 'pair_features')


def load_evidence_data(data, graph, visual, annotations=None, evidence=None):
    arrays, contract = load_relational_data(data, graph, annotations)
    with np.load(visual, allow_pickle=False) as z:
        vc = json.loads(str(z['contract_json'].item()))
        for key in ('attempt_uid', 'query_id'):
            if not np.array_equal(z[key], arrays[key]):
                raise ValueError('visual/data identities or order mismatch')
        if not np.allclose(z['elapsed_s'], arrays['elapsed_s'], atol=1e-6, rtol=0):
            raise ValueError('visual/data time mismatch')
        for key in VISUAL_INPUTS:
            arrays[key] = z[key]
    if (vc.get('format') != VISUAL_VERSION or vc.get('causal') is not True or
            vc.get('future_interpolation') is not False or vc.get('labels_used_as_inputs') is not False or
            vc.get('actions_used') is not False):
        raise ValueError('invalid visual cache contract')
    if vc.get('pair_feature_names') != list(PAIR_FEATURES):
        raise ValueError('visual feature order mismatch')
    allowed = {sha256(Path(data)), contract.get('split_override', {}).get('source_cache_sha256')}
    if vc.get('source_data_sha256') not in allowed or vc.get('source_graph_sha256') != sha256(Path(graph)):
        raise ValueError('visual cache belongs to different data/graph')
    n = len(arrays['query_id'])
    if arrays['pair_images_uint8'].shape != (n, 2, 3, 96, 96) or arrays['pair_images_uint8'].dtype != np.uint8:
        raise ValueError('invalid pair images')
    if arrays['pair_features'].shape != (n, 2, len(PAIR_FEATURES)) or not np.isfinite(arrays['pair_features']).all():
        raise ValueError('invalid pair features')
    arrays['observability_labels'] = np.full((n, 3), -1, np.int64)
    arrays['current_relation_labels'] = np.full((n, 3), -1, np.int64)
    arrays['segmentation_labels'] = np.full((n, 2, 32, 32), -1, np.int8)
    if evidence is not None:
        annotation = json.loads(Path(evidence).read_text())
        if annotation.get('format') != 'umi_visual_evidence_annotations_v1':
            raise ValueError('invalid evidence annotation format')
        keys = list(zip(arrays['attempt_uid'].astype(str), arrays['query_id'].astype(int)))
        index = dict(zip(keys, range(n)))
        if len(index) != n:
            raise ValueError('duplicate source identity')
        seen = set()
        for record in annotation['records']:
            key = (record['attempt_uid'], record['query_id'])
            if key not in index or key in seen or not isinstance(key[1], int):
                raise ValueError('invalid/duplicate evidence identity')
            seen.add(key); i = index[key]
            if arrays['split'][i] not in ('train', 'validation'):
                raise ValueError('test/inference cannot supply training annotations')
            for j, name in enumerate(RELATIONS):
                value = record['relations'].get(name, {})
                state, observable = value.get('state', -1), value.get('observable', -1)
                if state not in (-1, 0, 1, 2) or observable not in (-1, 0, 1):
                    raise ValueError('invalid relation/evidence class')
                if observable == 1 and state not in (0, 1):
                    raise ValueError('observable relation requires a known current state')
                arrays['observability_labels'][i, j] = observable
                arrays['current_relation_labels'][i, j] = state
                # Only direct current evidence can correct the old coarse labels.
                if observable == 1:
                    arrays['relation_labels'][i, j] = state
                inferred = value.get('inferred_state', -1)
                if inferred not in (-1, 0, 1, 2):
                    raise ValueError('invalid inferred state')
                # Inferred annotations are audit notes, never training ground truth.
                # Imagination is supervised by hiding genuinely observed frames.
            for channel, role in enumerate(('object', 'target_surface')):
                points = record.get('polygons', {}).get(role)
                if points is None:
                    continue  # No annotation is NOT an empty/background mask.
                points = np.asarray(points, float)
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
                    raise ValueError('invalid normalized polygon')
                mask = np.zeros((32, 32), np.uint8)
                cv2.fillPoly(mask, [np.rint(points * 31).astype(np.int32)], 1)
                arrays['segmentation_labels'][i, channel] = mask
        arrays['goal_labels'] = goal_targets(arrays['relation_labels'])
    contract = dict(contract, relational_visual=vc, visual_cache_sha256=sha256(Path(visual)),
                    evidence_annotations_sha256=sha256(Path(evidence)) if evidence else None)
    return arrays, contract


class EvidenceWindows(RelationalWindows):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        index = result['index']; history = self.histories[index]
        for key in VISUAL_INPUTS:
            values = self.arrays[key]
            padded = np.zeros((self.window, *values.shape[1:]), values.dtype)
            padded[:len(history)] = values[history]
            result[key] = torch.from_numpy(padded)
        for key in ('observability_labels', 'current_relation_labels', 'segmentation_labels'):
            result[key] = torch.from_numpy(self.arrays[key][index])
        return result


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in ('images', 'dt', 'mask', *GRAPH_INPUTS, *VISUAL_INPUTS)}


class EvidenceObserver(RelationalObserver):
    def __init__(self, node_dim, edge_dim, hidden=96, graph_hidden=48, graph_layers=2,
                 forecast_horizons_s=(.3, .6), use_graph=True, use_evidence=True):
        super().__init__(node_dim, edge_dim, hidden, graph_hidden, graph_layers, forecast_horizons_s, use_graph)
        # This variant performs explicit evidence/prior updates below and does
        # not use the base observer's non-recurrent history pooling module.
        del self.history_fusion
        self.model_spec['use_evidence'] = use_evidence
        self.pair_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, 2, 2), nn.GroupNorm(3, 12), nn.SiLU(),
            nn.Conv2d(12, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(384, 32), nn.SiLU())
        self.surface_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, 2, 2), nn.GroupNorm(3, 12), nn.SiLU(),
            nn.Conv2d(12, 24, 3, 2, 1), nn.GroupNorm(4, 24), nn.SiLU())
        self.surface_head = nn.Conv2d(24, 2, 1)
        extra = 64 + 2 * len(PAIR_FEATURES) + 24 * 4 * 4
        self.fusion = nn.Sequential(nn.Linear(64 + graph_hidden * 4 + 3 + extra, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.current_heads = nn.ModuleDict({name: nn.Linear(hidden, 3) for name in RELATIONS})
        self.evidence_head = nn.Sequential(nn.Linear(hidden, 48), nn.SiLU(), nn.Linear(48, 3))
        # Prior carries latent state through missing observations, conditioned on
        # the LAST SEEN visual motion, not a hidden robot command or future image.
        self.prior_transition = nn.Sequential(nn.Linear(hidden + 2 * len(PAIR_FEATURES), hidden),
                                              nn.Tanh(), nn.Linear(hidden, hidden))

    def forward(self, images, dt, mask, node_features, node_images_uint8, node_mask,
                edge_features, bindings, pair_images_uint8, pair_features, *, goals=None,
                return_features=False, observation_mask=None):
        if images.ndim != 6 or images.shape[2:4] != (1, 3):
            raise ValueError('expected single-camera video')
        b, t = images.shape[:2]
        if mask.shape != (b, t) or mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError('invalid history mask')
        if dt.shape != mask.shape or not torch.isfinite(dt[mask]).all() or (dt[mask] < 0).any():
            raise ValueError('invalid dt')
        if observation_mask is None:
            observation_mask = mask
        if observation_mask.shape != mask.shape or observation_mask.dtype != torch.bool or (observation_mask & ~mask).any():
            raise ValueError('invalid observation mask')
        visible = observation_mask
        rgb = torch.where(visible[..., None, None, None, None], images.float(), 0.)
        flat_rgb = rgb.reshape(b * t, 3, *images.shape[-2:]) / 255.
        global_features = self.global_encoder(flat_rgb)
        graph = self.encode_graph(node_features.flatten(0, 1), node_images_uint8.flatten(0, 1),
                                  node_mask.flatten(0, 1), edge_features.flatten(0, 1), bindings.flatten(0, 1))
        if not self.model_spec['use_graph']:
            graph = torch.zeros_like(graph)
        graph = torch.where(visible.flatten()[:, None], graph, 0.)
        pairs = torch.where(visible[..., None, None, None, None], pair_images_uint8.float(), 0.)
        pair_latent = self.pair_encoder(pairs.reshape(b * t * 2, 3, *pairs.shape[-2:]) / 255.).reshape(b * t, -1)
        motion = torch.where(visible[..., None, None], pair_features.float(), 0.).reshape(b, t, -1)
        surface = self.surface_encoder(flat_rgb)
        surface_latent = F.adaptive_avg_pool2d(surface, (4, 4)).flatten(1)
        inputs = self.fusion(torch.cat((global_features, graph, pair_latent, motion.flatten(0, 1), surface_latent), -1)).reshape(b, t, -1)
        evidence_sequence = self.evidence_head(inputs)
        hidden = inputs.new_zeros(b, self.model_spec['hidden'])
        prior = hidden
        last_motion = motion.new_zeros(b, motion.shape[-1])
        for k in range(t):
            delta = torch.where(mask[:, k], dt[:, k], 0.).clamp(max=1.)
            prior_k = hidden + delta[:, None] * self.prior_transition(torch.cat((hidden, last_motion), -1))
            # A visible observation is already a graph-conditioned state.  It
            # directly corrects the motion prior; no CfC/recurrent observation
            # update is used in the GNN-only observer.
            candidate = inputs[:, k]
            reliability = evidence_sequence[:, k].sigmoid().mean(-1)
            if not self.model_spec['use_evidence']:
                reliability = torch.ones_like(reliability)
            reliability = reliability * visible[:, k]
            combined = reliability[:, None] * candidate + (1. - reliability[:, None]) * prior_k
            hidden = torch.where(mask[:, k, None], combined, hidden)
            prior = torch.where(mask[:, k, None], prior_k, prior)
            last_motion = torch.where(visible[:, k, None], motion[:, k], last_motion)
        last = torch.where(mask, torch.arange(t, device=mask.device)[None], -1).max(1).values
        row = torch.arange(b, device=mask.device)
        current = inputs[row, last]
        relations = {name: head(hidden) for name, head in self.relation_heads.items()}
        probabilities = torch.cat([relations[name].softmax(-1) for name in RELATIONS], -1)
        specs = torch.as_tensor(list(GOALS.values()) if goals is None else goals, device=hidden.device, dtype=torch.long)
        if specs.ndim != 2 or specs.shape[1] != 3 or ((specs < -1) | (specs > 1)).any() or (specs == -1).all(1).any():
            raise ValueError('invalid goal specification')
        gf = self.goal_encoder(F.one_hot(specs + 1, 3).float().flatten(1))
        count = len(specs)
        goal_logits = self.goal_head(torch.cat((hidden[:, None].expand(-1, count, -1),
            probabilities[:, None].expand(-1, count, -1), gf[None].expand(b, -1, -1)), -1))
        event_state = self.event_state_encoder(hidden)
        result = dict(relations=relations, goals=goal_logits,
            events={name: head(hidden) for name, head in self.event_heads.items()},
            forecast={name: head(hidden) for name, head in self.forecast_heads.items()},
            # The visual evidence branch exposes the same Event Observer ABI
            # as the GNN-only branch. These heads are trained only after event
            # labels and Event-SMDP returns are supplied; legacy V4 weights are
            # deliberately incompatible with this format.
            event_state=event_state,
            event_posterior=self.event_state_head(event_state),
            event_progress=self.event_progress_head(event_state).squeeze(-1).sigmoid(),
            event_boundary_logit=self.event_boundary_head(event_state).squeeze(-1),
            event_uncertainty=F.softplus(self.event_uncertainty_head(event_state).squeeze(-1)),
            event_value=self.event_value_critic(event_state),
            current_relations={name: head(current) for name, head in self.current_heads.items()},
            prior_relations={name: head(prior) for name, head in self.relation_heads.items()},
            observability=evidence_sequence[row, last],
            segmentation=F.interpolate(self.surface_head(surface).reshape(b, t, 2, *surface.shape[-2:])[row, last],
                                       (32, 32), mode='bilinear', align_corners=False))
        if return_features:
            result['history_features'] = hidden
        return result


def load_observer(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != FORMAT or saved.get('labels_are_model_inputs') is not False:
        raise ValueError('not a versioned v4 observer')
    if saved.get('relation_schema') != {k: list(v) for k, v in RELATIONS.items()}:
        raise ValueError('incompatible relation schema')
    model = EvidenceObserver(**saved['model_spec'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved


def publish_evidence(times, beliefs, current, observability, goal_probabilities, *, use_evidence=True, confidence=.6):
    """Return both inferred goals and evidence-confirmed goals. No belief reset.

    Current confident contradictory evidence overrides history in the published
    facts immediately. Positive confirmation still requires three observations.
    Missing visual evidence removes confirmation, not temporal inference.
    """
    stable_belief = confirm_relations(times, beliefs, confidence=confidence)
    inferred = np.stack([publish_goal(stable_belief, goal_probabilities[:, j], goal, confidence=confidence)
                         for j, goal in enumerate(GOALS.values())], -1)
    if not use_evidence:
        return stable_belief, inferred, inferred.copy()
    if observability.shape != (len(times), 3) or not np.isfinite(observability).all() or np.any((observability < 0) | (observability > 1)):
        raise ValueError('invalid evidence probabilities')
    observed = {}
    for j, name in enumerate(RELATIONS):
        p = np.asarray(current[name]).copy()
        unsupported = observability[:, j] < confidence
        p[unsupported] = (0., 0., 1.)
        observed[name] = p
    stable_observed = confirm_relations(times, observed, confidence=confidence)
    facts = np.full_like(stable_belief, 2)
    for j, name in enumerate(RELATIONS):
        p = observed[name]
        state = p.argmax(-1)
        direct = (p.max(-1) >= confidence) & (state < 2)
        agrees = (stable_observed[:, j] == stable_belief[:, j]) & (stable_observed[:, j] < 2)
        facts[agrees, j] = stable_observed[agrees, j]
        # A directly observed contrary fact can immediately refute success.
        contradiction = direct & (state != stable_belief[:, j])
        facts[contradiction, j] = state[contradiction]
    confirmed = np.stack([publish_goal(facts, goal_probabilities[:, j], goal, confidence=confidence)
                          for j, goal in enumerate(GOALS.values())], -1)
    # An unstable contradictory observation must never establish a new success.
    stable_goal = np.stack([publish_goal(stable_observed, goal_probabilities[:, j], goal, confidence=confidence)
                           for j, goal in enumerate(GOALS.values())], -1)
    confirmed[(confirmed == 1) & (stable_goal != 1)] = 2
    return stable_belief, inferred, confirmed
