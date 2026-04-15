"""窗口动态语义特征：从当前输入窗口提取每个站点的统计特征（纯 tensor 运算，支持 GPU）。"""
import torch
import torch.nn.functional as F


def build_window_dynamic_features(x: torch.Tensor) -> torch.Tensor:
    """从当前 batch 的输入窗口提取每站点动态统计特征。

    Args:
        x: (B, T, N, C) 原始输入（投影前的观测值）

    Returns:
        feat: (B, N, D) 每个站点的窗口动态特征向量
              D = C * 8  (mean, std, min, max, slope, delta, diff_mean, diff_std)
    """
    B, T, N, C = x.shape

    # (B, T, N, C) → 在时间维上统计
    x_mean = x.mean(dim=1)                                  # (B, N, C)
    x_std = x.std(dim=1, unbiased=False).clamp(min=1e-8)    # (B, N, C)
    x_min = x.min(dim=1).values                              # (B, N, C)
    x_max = x.max(dim=1).values                              # (B, N, C)

    # 首末差值
    delta = x[:, -1, :, :] - x[:, 0, :, :]                  # (B, N, C)

    # 线性趋势斜率: least-squares slope over T steps
    t_idx = torch.arange(T, dtype=x.dtype, device=x.device)  # (T,)
    t_mean = t_idx.mean()
    t_var = ((t_idx - t_mean) ** 2).sum()
    # x_centered: (B, T, N, C)
    slope = ((t_idx.view(1, T, 1, 1) - t_mean) * (x - x_mean.unsqueeze(1))).sum(dim=1) / (t_var + 1e-8)

    # 一阶差分统计
    diff = x[:, 1:, :, :] - x[:, :-1, :, :]                 # (B, T-1, N, C)
    diff_mean = diff.mean(dim=1)                              # (B, N, C)
    diff_std = diff.std(dim=1, unbiased=False).clamp(min=1e-8)

    # 拼接: 8 个统计量 × C 通道 → D = 8C
    feat = torch.cat([x_mean, x_std, x_min, x_max, slope, delta, diff_mean, diff_std], dim=-1)
    return feat  # (B, N, 8C)


def dynamic_semantic_similarity(feat: torch.Tensor, mode: str = 'cosine') -> torch.Tensor:
    """计算 batch 内每个样本的站点间动态语义相似度。

    Args:
        feat: (B, N, D)
        mode: 'cosine' | 'dot'

    Returns:
        sim: (B, N, N) 值域 [0, 1]
    """
    if mode == 'cosine':
        feat_norm = F.normalize(feat, dim=-1, eps=1e-8)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # (B, N, N)
    else:
        sim = torch.bmm(feat, feat.transpose(1, 2))
        sim = sim / (feat.shape[-1] ** 0.5)

    # clamp 到 [0, 1]
    sim = sim.clamp(min=0.0, max=1.0)
    return sim
