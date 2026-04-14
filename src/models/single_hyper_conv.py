import torch
import torch.nn as nn
from .kan_linear import KANLinear


class SingleHyperConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_kan: bool = True,
                 dropout: float = 0.1, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.use_kan = use_kan
        self.proj = (
            KANLinear(in_dim, out_dim, grid_size, spline_order) if use_kan
            else nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """接收预计算好的 float32 归一化超图矩阵 A (N,N)。"""
        with torch.amp.autocast('cuda', enabled=False):
            x = torch.einsum('nm,bmc->bnc', A, x.float())
        x = self.proj(x)
        return self.dropout(x)
