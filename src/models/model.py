import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    LSTM-based forecaster for rack inlet temperature.
    Inputs: sequences of features (ambient_C, it_load, rack_inlet_C, cooling_setpoint_C)
    Outputs: predicted next N minutes of rack inlet temperature.
    """
    def __init__(self, n_features, hidden_size=64, num_layers=2, pred_len=15):
        super(LSTMForecaster, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, sequence_length, n_features)
        Returns:
            preds: Tensor of shape (batch_size, pred_len)
        """
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        preds = self.fc(last_timestep)
        return preds
