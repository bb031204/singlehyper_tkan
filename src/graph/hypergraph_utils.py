import torch


@torch.no_grad()
def normalized_hypergraph_matrix(H: torch.Tensor, W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}

    支持两种输入：
      - W: (E,)   → 返回 (N, N) 静态归一化矩阵
      - W: (B, E) → 返回 (B, N, N) batch 动态归一化矩阵

    全程 float32 + 禁用 autocast，防止 float16 下矩阵链乘溢出。
    """
    with torch.amp.autocast('cuda', enabled=False):
        H = H.float()
        W = W.float()

        De_inv = 1.0 / (H.sum(0) + eps)  # (E,)

        if W.dim() == 1:
            Dv_inv_sqrt = (H * W.unsqueeze(0)).sum(1)
            Dv_inv_sqrt = torch.pow(Dv_inv_sqrt + eps, -0.5)

            HW = H * (W * De_inv).unsqueeze(0)
            A = HW @ H.t()
            A = Dv_inv_sqrt.unsqueeze(1) * A * Dv_inv_sqrt.unsqueeze(0)
            return A  # (N, N)
        else:
            # W: (B, E) → batch 归一化
            B = W.shape[0]
            Ht = H.t()  # (E, N)

            # (B, E) * (E,) → (B, E)
            W_scaled = W * De_inv.unsqueeze(0)
            # H: (N, E), W_scaled: (B, E) → HW: (B, N, E)
            HW = H.unsqueeze(0) * W_scaled.unsqueeze(1)
            # (B, N, E) @ (E, N) → (B, N, N)
            A = torch.bmm(HW, Ht.unsqueeze(0).expand(B, -1, -1))

            # Dv: (B, N)
            Dv = (H.unsqueeze(0) * W.unsqueeze(1)).sum(2)
            Dv_inv_sqrt = torch.pow(Dv + eps, -0.5)
            A = Dv_inv_sqrt.unsqueeze(2) * A * Dv_inv_sqrt.unsqueeze(1)
            return A  # (B, N, N)
