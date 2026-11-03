"""The one 1D-CNN used by every pipeline (raw signal, VMD modes, UVMD modes, end-to-end UVMD).

Only ``in_channels`` changes: K for a K-mode decomposition, 1 for the raw signal. Everything else is identical, so
differences in accuracy come from the input representation, not from the classifier.

Input  (B, in_channels, T)   a window, or its K modes, as float32
Output (B, n_classes)        logits

The first layer is a BatchNorm over the input channels. It is part of the classifier (learned affine + running
statistics, identical in every pipeline), and it keeps the modes, which have different energies, on a common scale
without any non-differentiable preprocessing in front of the end-to-end model.
"""
from __future__ import annotations

import torch.nn as nn


class Fault1DCNN(nn.Module):
    def __init__(self, in_channels, n_classes, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.in_channels, self.n_classes = in_channels, n_classes
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, w, 15, padding=7), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(w, 2 * w, 9, padding=4), nn.BatchNorm1d(2 * w), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(2 * w, 4 * w, 5, padding=2), nn.BatchNorm1d(4 * w), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(4 * w, 4 * w, 3, padding=1), nn.BatchNorm1d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
