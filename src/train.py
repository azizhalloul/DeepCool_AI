import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import TimeSeriesDataset
from src.models.model import LSTMForecaster
import os

def train(run_csv='data/raw/run1.csv', epochs=10, batch_size=32, out_file='experiments/model_final.pt'):
    os.makedirs('experiments', exist_ok=True)
    print(f"Loading data from {run_csv}...")
    df = pd.read_csv(run_csv)

    ds = TimeSeriesDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMForecaster(n_features=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Starting training for {epochs} epochs...")
    for ep in range(epochs):
        total_loss = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = F.mse_loss(preds, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs} | Loss: {total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), out_file)
    print(f"✅ Model saved to {out_file}")

if __name__ == "__main__":
    train()
