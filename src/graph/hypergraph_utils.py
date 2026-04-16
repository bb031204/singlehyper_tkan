import torch


@torch.no_grad()
def normalized_hypergraph_matrix(H: torch.Tensor, W: torch.Tensor,
                                 cache: dict = None,
                                 eps: float = 1e-6) -> torch.Tensor:
    """A = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}

    支持两种输入：
      - W: (E,)   → 返回 (N, N) 静态归一化矩阵
      - W: (B, E) → 返回 (B, N, N) batch 动态归一化矩阵

    cache: 可选预计算缓存 {'De_inv': (E,), 'Ht': (E, N), 'H': (N, E)}，
           由 precompute_hypergraph_cache() 生成，避免每次重复计算。
    """
    with torch.amp.autocast('cuda', enabled=False):
        H = H.float()
        W = W.float()

        if cache is not None:
            De_inv = cache['De_inv']
            Ht = cache['Ht']
        else:
            De_inv = 1.0 / (H.sum(0) + eps)
            Ht = H.t()

        if W.dim() == 1:
            Dv_inv_sqrt = (H * W.unsqueeze(0)).sum(1)
            Dv_inv_sqrt = torch.pow(Dv_inv_sqrt + eps, -0.5)

            HW = H * (W * De_inv).unsqueeze(0)
            A = HW @ Ht
            A = Dv_inv_sqrt.unsqueeze(1) * A * Dv_inv_sqrt.unsqueeze(0)
            return A

        B = W.shape[0]
        W_scaled = W * De_inv.unsqueeze(0)
        HW = H.unsqueeze(0) * W_scaled.unsqueeze(1)
        A = torch.bmm(HW, Ht.unsqueeze(0).expand(B, -1, -1))

        Dv = (H.unsqueeze(0) * W.unsqueeze(1)).sum(2)
        Dv_inv_sqrt = torch.pow(Dv + eps, -0.5)
        A = Dv_inv_sqrt.unsqueeze(2) * A * Dv_inv_sqrt.unsqueeze(1)
        return A


@torch.no_grad()
def precompute_hypergraph_cache(H: torch.Tensor, eps: float = 1e-6) -> dict:
    """预计算只依赖 H 的中间量，供 normalized_hypergraph_matrix 复用。"""
    H = H.float()
    De_inv = 1.0 / (H.sum(0) + eps)
    Ht = H.t().contiguous()
    return {'De_inv': De_inv, 'Ht': Ht}
