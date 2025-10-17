import torch.nn as nn
import torch

class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, pred_len=15, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, pred_len)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        preds = self.fc(last)
        return preds
