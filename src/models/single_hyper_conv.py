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
        """超图聚合 + 投影。

        Args:
            x: (B, N, D)
            A: (N, N) 静态 或 (B, N, N) batch 动态
        """
        with torch.amp.autocast('cuda', enabled=False):
            xf = x.float()
            if A.dim() == 2:
                xf = torch.einsum('nm,bmc->bnc', A, xf)
            else:
                xf = torch.bmm(A, xf)
        x = self.proj(xf)
        return self.dropout(x)
