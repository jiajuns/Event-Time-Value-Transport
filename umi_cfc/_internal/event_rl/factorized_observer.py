"""RGB + optional-ablatable YOLO cues -> causal CfC -> factorized event heads.

No current/previous ground-truth labels, actions, joints, episode IDs, outcomes,
absolute time or time-to-end are model inputs. Unknown is a supervised class;
unannotated targets use -1 and never enter the forward pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .factorized_events import CHECKPOINT_FORMAT, DATA_FORMAT, HEADS, SCHEMA
from .semantic_cues import feature_names
from .cfc import VideoEventCfCCell


def load_dataset(path: Path) -> tuple[dict, dict]:
    names = ("images_uint8", "semantic_features", "labels", "attempt_uid", "query_id", "elapsed_s", "group_uid", "split")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in names}  # No robot arrays are deserialized.
        contract = json.loads(str(archive["contract_json"].item()))
    if contract.get("format") != DATA_FORMAT or contract.get("schema") != SCHEMA:
        raise ValueError("dataset/schema version mismatch; legacy five/six events are incompatible")
    n = len(arrays["attempt_uid"])
    size, cameras = contract["image_size"], contract["cameras"]
    dim = len(feature_names(cameras))
    if n == 0 or arrays["images_uint8"].shape != (n, len(cameras), 3, size, size) or arrays["images_uint8"].dtype != np.uint8:
        raise ValueError("invalid RGB dataset shape/dtype")
    if arrays["semantic_features"].shape != (n, dim) or not np.isfinite(arrays["semantic_features"]).all():
        raise ValueError("invalid semantic features")
    if arrays["labels"].shape != (n, len(HEADS)) or arrays["labels"].dtype.kind not in "iu":
        raise ValueError("invalid labels shape/dtype")
    for column, classes in enumerate(HEADS.values()):
        if np.any((arrays["labels"][:, column] < -1) | (arrays["labels"][:, column] >= len(classes))):
            raise ValueError("label out of schema range")
    for name in ("attempt_uid", "group_uid", "split", "query_id", "elapsed_s"):
        if arrays[name].shape != (n,):
            raise ValueError("invalid identity/time shape")
    if arrays["query_id"].dtype.kind not in "iu" or not np.isfinite(arrays["elapsed_s"]).all() or np.any(arrays["elapsed_s"] < 0):
        raise ValueError("invalid query/time values")
    if not set(arrays["split"].tolist()) <= {"train", "validation", "test", "inference"}:
        raise ValueError("invalid split")
    for group in np.unique(arrays["group_uid"]):
        if len(np.unique(arrays["split"][arrays["group_uid"] == group])) != 1:
            raise ValueError("source group crosses splits")
    for uid in np.unique(arrays["attempt_uid"]):
        selected = np.flatnonzero(arrays["attempt_uid"] == uid)
        if any(len(np.unique(arrays[key][selected])) != 1 for key in ("group_uid", "split")):
            raise ValueError("episode crosses split/group")
        query = arrays["query_id"][selected]
        if len(np.unique(query)) != len(query) or np.any(np.diff(arrays["elapsed_s"][selected[np.argsort(query)]]) <= 0):
            raise ValueError("invalid episode chronology")
    return arrays, contract


class CausalWindows(Dataset):
    """Classify each anchor using only its past/current video; right-padded."""
    def __init__(self, arrays: dict, window: int, indices=None):
        if window <= 0:
            raise ValueError("window must be positive")
        self.arrays, self.window = arrays, window
        self.indices = np.arange(len(arrays["attempt_uid"])) if indices is None else np.asarray(indices)
        self.histories = {}
        for uid in np.unique(arrays["attempt_uid"]):
            selected = np.flatnonzero(arrays["attempt_uid"] == uid)
            selected = selected[np.argsort(arrays["query_id"][selected])]
            for pos, index in enumerate(selected):
                self.histories[int(index)] = selected[max(0, pos + 1 - window):pos + 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        history = self.histories[index]
        size = len(history)
        images = np.zeros((self.window, *self.arrays["images_uint8"].shape[1:]), dtype=np.uint8)
        cues = np.zeros((self.window, self.arrays["semantic_features"].shape[-1]), dtype=np.float32)
        dt = np.zeros(self.window, dtype=np.float32)
        mask = np.arange(self.window) < size
        images[:size] = self.arrays["images_uint8"][history]
        cues[:size] = self.arrays["semantic_features"][history]
        dt[1:size] = np.diff(self.arrays["elapsed_s"][history])
        return dict(images=torch.from_numpy(images), cues=torch.from_numpy(cues),
                    dt=torch.from_numpy(dt), mask=torch.from_numpy(mask),
                    labels=torch.as_tensor(self.arrays["labels"][index], dtype=torch.long), index=index)


class FactorizedEventObserver(nn.Module):
    def __init__(self, cameras: int, cue_dim: int, hidden: int = 96, use_cues: bool = True,
                 forecast_horizons_s=()):
        super().__init__()
        if min(cameras, cue_dim, hidden) <= 0:
            raise ValueError("dimensions must be positive")
        self.model_spec = dict(cameras=cameras, cue_dim=cue_dim, hidden=hidden, use_cues=use_cues)
        if forecast_horizons_s:
            self.model_spec['forecast_horizons_s']=list(forecast_horizons_s)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.cue_projection = nn.Sequential(nn.Linear(cue_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(64 * cameras + 32, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.initial = nn.Linear(hidden, hidden)
        self.cfc = VideoEventCfCCell(hidden)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, len(classes)) for name, classes in HEADS.items()})
        self.forecast_heads = nn.ModuleDict({f'h{k}_{name}':nn.Linear(hidden,len(classes))
                                            for k,_ in enumerate(forecast_horizons_s)
                                            for name,classes in HEADS.items()})
        self.register_buffer("cue_mean", torch.zeros(cue_dim))
        self.register_buffer("cue_std", torch.ones(cue_dim))

    def fit_normalization(self, cues: np.ndarray):
        if cues.ndim != 2 or len(cues) == 0 or not np.isfinite(cues).all():
            raise ValueError("finite training cues required")
        self.cue_mean.copy_(torch.as_tensor(cues.mean(axis=0), dtype=torch.float32))
        self.cue_std.copy_(torch.as_tensor(np.maximum(cues.std(axis=0), 1e-4), dtype=torch.float32))

    def forward(self, images, cues, dt, mask, *, return_features=False):
        if images.ndim != 6 or images.shape[2:4] != (self.model_spec["cameras"], 3):
            raise ValueError("images must be [B,T,C,3,H,W]")
        batch, length, views, channels, height, width = images.shape
        if (cues.shape != (batch, length, self.model_spec["cue_dim"]) or dt.shape != (batch, length) or
                mask.shape != (batch, length) or mask.dtype != torch.bool or not bool(mask.any(dim=1).all())):
            raise ValueError("invalid temporal input shapes/mask")
        if (not bool(torch.isfinite(cues[mask]).all()) or not bool(torch.isfinite(dt[mask]).all()) or
                bool((dt[mask] < 0).any()) or not bool(torch.isfinite(images[mask]).all())):
            raise ValueError("valid inputs must be finite with nonnegative dt")
        if bool(((images[mask] < 0) | (images[mask] > 255)).any()):
            raise ValueError("RGB values must be in [0,255]")
        rgb = torch.where(mask[..., None, None, None, None], images.float(), 0.) / 255.
        visual = self.encoder(rgb.reshape(batch * length * views, channels, height, width)).reshape(batch, length, -1)
        normalized = torch.where(mask[..., None], cues, self.cue_mean)
        normalized = (normalized - self.cue_mean) / self.cue_std
        cue_features = self.cue_projection(normalized)
        if not self.model_spec["use_cues"]:
            cue_features = torch.zeros_like(cue_features)
        inputs = self.fusion(torch.cat((visual, cue_features), dim=-1))
        dt = torch.where(mask, dt, 0.)
        hidden = inputs.new_zeros(batch, self.model_spec["hidden"])
        seen = torch.zeros(batch, dtype=torch.bool, device=inputs.device)
        for index in range(length):
            candidate = self.cfc(inputs[:, index], hidden, dt[:, index])
            candidate = torch.where(seen[:, None], candidate, torch.tanh(self.initial(inputs[:, index])))
            hidden = torch.where(mask[:, index, None], candidate, hidden)
            seen = seen | mask[:, index]
        result = {name: head(hidden) for name, head in self.heads.items()}
        if self.forecast_heads:
            result['forecast']={name:head(hidden) for name,head in self.forecast_heads.items()}
        if return_features:
            result["history_features"] = hidden
        return result


def model_inputs(batch: dict, device) -> dict:
    return {name: batch[name].to(device) for name in ("images", "cues", "dt", "mask")}


def classification_loss(logits: dict, labels, active_heads, weights=None):
    losses = []
    for name in active_heads:
        target = labels[:, list(HEADS).index(name)]
        valid = target >= 0
        if bool(valid.any()):
            losses.append(F.cross_entropy(logits[name][valid], target[valid],
                                          weight=None if weights is None else weights[name]))
    if not losses:
        raise ValueError("batch has no supervised labels")
    return torch.stack(losses).mean()


def load_observer(path: Path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)  # trusted local artifacts only
    if saved.get("format") != CHECKPOINT_FORMAT or saved.get("schema") != SCHEMA:
        raise ValueError("incompatible event observer checkpoint")
    model = FactorizedEventObserver(**saved["model_spec"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved
