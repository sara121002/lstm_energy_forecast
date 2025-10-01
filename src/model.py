from __future__ import annotations
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.1, bidirectional: bool = False, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        dir_mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * dir_mult, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)             # (B, T, H*D)
        last = out[:, -1, :]              # (B, H*D)
        yhat = self.head(last)            # (B, horizon)
        return yhat