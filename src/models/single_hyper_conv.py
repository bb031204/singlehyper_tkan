import torch
import torch.nn as nn
from ..graph.hypergraph_utils import normalized_hypergraph_matrix
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

    def forward(self, x: torch.Tensor, H: torch.Tensor, W: torch.Tensor):
        A = normalized_hypergraph_matrix(H.to(x.device), W.to(x.device))
        x = torch.einsum('nm,bmc->bnc', A, x)
        x = self.proj(x)
        return self.dropout(x)
