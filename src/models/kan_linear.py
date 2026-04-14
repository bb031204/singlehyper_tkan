import torch
import torch.nn as nn


class KANLinear(nn.Module):
    """轻量版 KAN Linear 层：base linear + learnable grid-based 非线性映射。

    与标准 KAN (Kolmogorov-Arnold Networks) 的关系：
    - 标准 KAN 对每条边学习一个完整的 univariate B-spline 函数，
      需要 in_features * out_features 组独立样条参数
    - 本轻量版简化为：base linear + 逐输出维度的 grid-based learnable
      activation（用 RBF 插值近似样条），参数量远小于完整版
    - grid_size 控制非线性映射的精细度（对应 B-spline 控制点数量）
    - spline_order 保留为接口参数，便于后续升级为完整 B-spline 实现
    - 初始化时 spline 权重为零，模型启动接近 Linear；训练中逐步
      学习非线性成分——这是 KAN 论文推荐的渐进式训练策略
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.base_linear = nn.Linear(in_features, out_features)

        grid = torch.linspace(-2.0, 2.0, grid_size)
        self.register_buffer('grid', grid)
        self.spline_weight = nn.Parameter(torch.zeros(out_features, grid_size))
        self.spline_scale = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        # 强制 float32 避免 AMP float16 下 exp() 溢出导致 NaN
        orig_dtype = base.dtype
        base_f = base.float()
        diff = base_f.unsqueeze(-1) - self.grid.float()
        rbf = torch.exp(-0.5 * diff ** 2)
        spline = (rbf * self.spline_weight.float()).sum(-1)
        out = base_f + spline * self.spline_scale.float()
        return out.to(orig_dtype)
