import torch
import numpy as np

def sequence_mse(preds: torch.Tensor, target: torch.Tensor):
    return torch.mean((preds - target) ** 2)

def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)
