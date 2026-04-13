import torch
import torch.nn as nn


class KANLinear(nn.Module):
    """轻量版 KANLinear：线性项 + 逐维非线性映射（可替换为更完整样条版）。"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.zeros(out_features))
        self.act = nn.SiLU()

    def forward(self, x):
        base = self.linear(x)
        return base + self.act(base) * self.gate
