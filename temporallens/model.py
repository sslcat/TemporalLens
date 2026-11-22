"""Dense classification with one frame or a centered five-frame receptive field."""

from itertools import groupby

import torch
from torch import nn
from torch.nn import functional as F


class MotionHead(nn.Module):
    def __init__(self, temporal):
        super().__init__()
        self.temporal = temporal
        self.project = nn.Conv1d(12 if temporal else 2, 32, 1)
        self.readout = nn.Sequential(
            nn.ReLU(), nn.Conv1d(32, 32, 1), nn.ReLU(), nn.Conv1d(32, 5, 1)
        )

    def forward(self, x):
        center = (x.transpose(1, 2) - 0.5) * 2
        if self.temporal:
            window = F.pad(center, (2, 2), mode="replicate").unfold(2, 5, 1)
            relative = (window - center.unsqueeze(-1)) * 32
            offsets = relative.permute(0, 1, 3, 2).flatten(1, 2)
            center = torch.cat((center, offsets), dim=1)
        return self.readout(self.project(center)).transpose(1, 2)


def segments(ids, minimum=3):
    result = []
    start = 0
    for label, group in groupby(map(int, ids)):
        end = start + sum(1 for _ in group)
        if label and end - start >= minimum:
            result.append((label, start, end))
        start = end
    return result
