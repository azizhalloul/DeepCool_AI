import torch
from src.models.lstm import LSTMForecaster

def test_lstm_forward():
    model = LSTMForecaster(n_features=4, hidden_size=32, num_layers=1, pred_len=10)
    x = torch.randn(2, 20, 4)
    y = model(x)
    assert y.shape == (2, 10)
