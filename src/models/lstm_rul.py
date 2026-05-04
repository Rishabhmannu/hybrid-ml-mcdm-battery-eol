"""
LSTM model for Remaining Useful Life (RUL) time-series forecasting.
Captures temporal degradation patterns in battery cycling data.

References:
- Jin et al. (2025), Energy — LSTM+XGBoost with Binary Firefly feature selection
- Zhao et al. (2024), Scientific Reports — AT-CNN-BiLSTM, R2>0.9910
- Rout et al. (2025), Scientific Reports — LSTM achieves 0.9982 accuracy
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import LSTM_CONFIG


class BatterySequenceDataset(Dataset):
    """Wraps pre-built (n_windows, seq_len, n_features) tensors."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class BatteryLSTM(nn.Module):
    """
    LSTM regressor for battery RUL prediction.
    Input  → stacked LSTM (default 2 layers, hidden=128) → FC head → scalar RUL.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        hidden_size = hidden_size or LSTM_CONFIG["hidden_size"]
        num_layers = num_layers or LSTM_CONFIG["num_layers"]
        dropout = dropout or LSTM_CONFIG["dropout"]

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


def create_dataloaders(
    train_sequences, train_targets, val_sequences, val_targets,
    *, batch_size: int | None = None,
):
    """Create train and validation DataLoaders from pre-built sequences."""
    batch_size = batch_size or LSTM_CONFIG["batch_size"]
    train_ds = BatterySequenceDataset(train_sequences, train_targets)
    val_ds = BatterySequenceDataset(val_sequences, val_targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl
