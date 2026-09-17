"""Event-SMDP targets for the Role-Graph Event Observer value sidecar.

This is intentionally separate from visual/event auxiliary supervision.  The
targets use only online rewards, terminal flags, reviewed/oracle event IDs, and
the next Event Value bootstrap; neither task outcome nor future RGB frame is a
model input.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def event_smdp_value_loss(
    event_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    event_ids: torch.Tensor,
    *,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return critic loss, one target per event, and its start indices.

    Inputs represent one rollout: values/rewards/event IDs are `[T]`, while
    `dones` is `[T + 1]`.  An event is a contiguous run of the same nonnegative
    ID.  A terminal event has no bootstrap.  Bootstrap values are detached so
    every target trains only its event-start value.
    """
    steps = rewards.numel()
    if (event_values.shape != (steps + 1,) or dones.shape != (steps + 1,) or
            event_ids.shape != (steps,) or not 0 < gamma <= 1):
        raise ValueError("invalid Event-SMDP sequence shapes or gamma")
    starts, targets = [], []
    begin = 0
    while begin < steps:
        if event_ids[begin] < 0:
            begin += 1
            continue
        label, end = event_ids[begin], begin + 1
        while end < steps and event_ids[end] == label and not dones[end]:
            end += 1
        duration = end - begin
        discounts = rewards.new_tensor(gamma).pow(torch.arange(duration, device=rewards.device))
        total = (discounts * rewards[begin:end]).sum()
        if not bool(dones[end]):
            total = total + gamma ** duration * event_values[end].detach()
        starts.append(begin)
        targets.append(total)
        begin = end
    if not starts:
        return event_values.sum() * 0., event_values.new_empty(0), torch.empty(0, dtype=torch.long, device=event_values.device)
    index = torch.tensor(starts, dtype=torch.long, device=event_values.device)
    target = torch.stack(targets)
    return F.smooth_l1_loss(event_values[index], target), target, index
