import torch
import torch.nn as nn
from ..graph.hypergraph_utils import normalized_hypergraph_matrix
from .single_hyper_conv import SingleHyperConv
from .tkan import TKANLayer


class SingleHyperTKAN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 32,
                 hidden_channels: int = 32, tkan_hidden: int = 32,
                 sub_kan_layers: int = 3, use_kan: bool = True,
                 dropout: float = 0.1, pred_steps: int = 12,
                 grid_size: int = 5, spline_order: int = 3,
                 bypass_spatial: bool = False):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_dim = output_dim
        self.bypass_spatial = bypass_spatial

        self.in_proj = nn.Linear(input_dim, d_model)

        if not bypass_spatial:
            self.spatial = SingleHyperConv(
                d_model, hidden_channels, use_kan=use_kan,
                dropout=dropout, grid_size=grid_size, spline_order=spline_order)
        else:
            self.spatial_bypass = (
                nn.Linear(d_model, hidden_channels) if d_model != hidden_channels
                else nn.Identity()
            )

        self.temporal = TKANLayer(
            hidden_channels, tkan_hidden, sub_layers=sub_kan_layers,
            grid_size=grid_size, spline_order=spline_order,
            return_sequences=False)

        self.head = nn.Sequential(
            nn.Linear(tkan_hidden, tkan_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tkan_hidden, pred_steps * output_dim)
        )

        self._cached_A = None

    def forward(self, x, H, W, output_length: int = 12):
        B, T, N, _ = x.shape
        x = self.in_proj(x)

        if not self.bypass_spatial:
            if self._cached_A is None or self._cached_A.shape[0] != N:
                self._cached_A = normalized_hypergraph_matrix(
                    H.to(x.device), W.to(x.device))
            A = self._cached_A

            spatial_out = []
            for t in range(T):
                xt = x[:, t, :, :]
                yt = self.spatial(xt, A)
                spatial_out.append(yt)
            x = torch.stack(spatial_out, dim=1)
        else:
            x = self.spatial_bypass(x)

        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        h = self.temporal(x)
        y = self.head(h).view(B, N, self.pred_steps, self.output_dim)
        y = y.permute(0, 2, 1, 3).contiguous()
        if output_length != self.pred_steps:
            y = y[:, :output_length]
        return y

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
