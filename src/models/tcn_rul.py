"""
Temporal Convolutional Network (TCN) for battery Remaining Useful Life (RUL).

Iter-3 §3.13 — secondary deep-learning baseline against the audited XGBoost RUL.

The TCN architecture (Bai, Kolter & Koltun, 2018, arXiv:1803.01271) replaces
LSTM's sequential recurrence with stacked dilated *causal* 1-D convolutions.
This addresses the LSTM convergence pathologies documented in §3.11.5:
  - Parallelisable across the time axis (no recurrence bottleneck)
  - Receptive field grows exponentially with depth via dilation
  - No vanishing/exploding gradients
  - Weight-norm + residual connections give stable training on medium-scale corpora

Battery-RUL precedent:
- Bai et al. 2018 (arXiv:1803.01271) — TCN architectural foundation
- TCN-LSTM hybrid for SoH+RUL — J. Energy Storage 2024 [10.1016/j.est.2023.110...]
- Cycle-based MLP/GRU/LSTM/TCN bake-off — Sci. Reports 2025
  (https://www.nature.com/articles/s41598-025-20995-7) — TCN ties MLP, both beat LSTM
- TCN with spatial attention for RUL — Sci. Reports 2025
  (https://www.nature.com/articles/s41598-025-17610-0)

Output reuses `BatterySequenceDataset` from `src.models.lstm_rul` so the same
training data flows through both architectures (apples-to-apples comparison).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_rul import BatterySequenceDataset
from src.utils.config import TCN_CONFIG


# =============================================================================
# Building blocks (Bai et al. 2018 TCN — github.com/locuslab/TCN reference)
# =============================================================================

class _CausalChomp(nn.Module):
    """Slice off the right-side padding so the convolution is strictly causal:
    the output at time `t` depends only on inputs at times `≤ t`."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp == 0:
            return x
        return x[:, :, : -self.chomp].contiguous()


class TemporalBlock(nn.Module):
    """One TCN residual block: two stacked dilated causal Conv1D layers with
    weight norm, ReLU, dropout, and a 1x1 residual projection if needed."""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int,
                 padding: int, dropout: float):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp1 = _CausalChomp(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size,
                                           padding=padding, dilation=dilation))
        self.chomp2 = _CausalChomp(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection if channel count changes
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu_out = nn.ReLU()

        # Bai 2018 init recommendation
        nn.init.normal_(self.conv1.weight, 0.0, 0.01)
        nn.init.normal_(self.conv2.weight, 0.0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially-increasing dilation."""

    def __init__(self, n_inputs: int, num_channels: list[int],
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        for i, n_out in enumerate(num_channels):
            dilation = 2 ** i
            n_in = n_inputs if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(n_in, n_out, kernel_size,
                                        dilation=dilation, padding=padding,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, channels, time)
        return self.network(x)


# =============================================================================
# Public model
# =============================================================================

class BatteryTCN(nn.Module):
    """
    TCN regressor for battery RUL.

    Input shape   : (batch, seq_len, n_features)   ← matches `load_battery_sequences`
    Conv1D shape  : (batch, n_features, seq_len)   ← internal transpose
    Output shape  : (batch,)                       ← scalar RUL

    Layout: input → transpose → TemporalConvNet → take last timestep → FC head.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        num_channels = num_channels or TCN_CONFIG["num_channels"]
        kernel_size = kernel_size or TCN_CONFIG["kernel_size"]
        dropout = dropout if dropout is not None else TCN_CONFIG["dropout"]

        self.tcn = TemporalConvNet(
            n_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @property
    def receptive_field(self) -> int:
        """Cycles back from the prediction point that the TCN can attend to."""
        kernel = self.tcn.network[0].conv1.kernel_size[0]
        n_layers = len(self.tcn.network)
        # 2 conv layers per block, each contributing (kernel-1)*dilation
        return 1 + 2 * (kernel - 1) * (2 ** n_layers - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x_t = x.transpose(1, 2)
        out = self.tcn(x_t)
        # Take the last time-step's representation → FC → scalar
        last = out[:, :, -1]
        return self.fc(last).squeeze(-1)


def create_dataloaders(
    train_sequences, train_targets, val_sequences, val_targets,
    *, batch_size: int | None = None,
):
    """Same-shape DataLoaders as LSTM — reusable for apples-to-apples training."""
    batch_size = batch_size or TCN_CONFIG["batch_size"]
    train_ds = BatterySequenceDataset(train_sequences, train_targets)
    val_ds   = BatterySequenceDataset(val_sequences,   val_targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl
