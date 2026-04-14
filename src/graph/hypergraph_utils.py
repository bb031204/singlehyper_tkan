import torch


@torch.no_grad()
def normalized_hypergraph_matrix(H: torch.Tensor, W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}

    利用对角矩阵稀疏性，避免构造完整 NxN 对角阵再做稠密乘法。
    全程 float32 + 禁用 autocast，防止 float16 下矩阵链乘溢出。
    """
    with torch.amp.autocast('cuda', enabled=False):
        H = H.float()
        W = W.float()

        De_inv = 1.0 / (H.sum(0) + eps)            # (E,)
        Dv_inv_sqrt = (H * W.unsqueeze(0)).sum(1)   # (N,)
        Dv_inv_sqrt = torch.pow(Dv_inv_sqrt + eps, -0.5)

        # 等价于 Dv^{-1/2} @ H @ W_diag @ De^{-1} @ H^T @ Dv^{-1/2}
        # 但只做逐元素缩放 + 一次矩阵乘法，不构造 NxN 对角阵
        HW = H * (W * De_inv).unsqueeze(0)          # (N, E) — H 列缩放
        A = HW @ H.t()                               # (N, N) — 唯一的稠密乘法
        A = Dv_inv_sqrt.unsqueeze(1) * A * Dv_inv_sqrt.unsqueeze(0)

        return A
