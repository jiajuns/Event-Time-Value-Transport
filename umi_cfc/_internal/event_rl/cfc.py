"""Shared elapsed-time CfC cell, independent of legacy event/value schemas."""
import torch
from torch import nn
from torch.nn import functional as F


class VideoEventCfCCell(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proposal = nn.Linear(2 * hidden_size, hidden_size)
        self.time_constant = nn.Linear(2 * hidden_size, hidden_size)
        self.mix = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, value, hidden, dt):
        joined = torch.cat((value, hidden), dim=-1)
        target = torch.tanh(self.proposal(joined))
        tau = F.softplus(self.time_constant(joined)) + 1e-3
        decay = torch.exp(-dt[:, None] / tau)
        gate = torch.sigmoid(self.mix(joined))
        return gate * (decay * hidden + (1 - decay) * target) + (1 - gate) * hidden
