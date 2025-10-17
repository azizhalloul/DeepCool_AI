import numpy as np
from torch.utils.data import Dataset
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_len=60, pred_len=15, features=None):
        if features is None:
            features = ['ambient_C','it_load','rack_inlet_C','cooling_setpoint_C']
        self.df = df.reset_index(drop=True)
        self.arr = self.df[features].values.astype('float32')
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)

    def __len__(self):
        return max(0, len(self.arr) - self.input_len - self.pred_len + 1)

    def __getitem__(self, idx):
        x = self.arr[idx: idx + self.input_len]
        y = self.arr[idx + self.input_len: idx + self.input_len + self.pred_len, 2]  # target: rack_inlet_C
        return x, y

def load_csv(path: str):
    return pd.read_csv(path, parse_dates=['timestamp'])
