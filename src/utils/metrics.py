import torch
from typing import Dict, List


def MAE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def RMSE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def MAPE(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return torch.mean(torch.abs((pred - target) / (torch.abs(target) + epsilon))) * 100.0


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, metrics: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, float]:
    out = {}
    for m in metrics:
        k = m.lower()
        if k == 'mae':
            out['mae'] = MAE(pred, target).item()
        elif k == 'rmse':
            out['rmse'] = RMSE(pred, target).item()
        elif k == 'mape':
            out['mape'] = MAPE(pred, target).item()
    return out
