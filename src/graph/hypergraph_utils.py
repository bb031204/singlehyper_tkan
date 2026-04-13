import torch


def normalized_hypergraph_matrix(H: torch.Tensor, W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """A = Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}"""
    H = H.float()
    W = W.float()
    De = torch.sum(H, dim=0) + eps
    Dv = torch.sum(H * W.unsqueeze(0), dim=1) + eps
    Dv_inv_sqrt = torch.diag(torch.pow(Dv, -0.5))
    De_inv = torch.diag(1.0 / De)
    W_diag = torch.diag(W)
    A = Dv_inv_sqrt @ H @ W_diag @ De_inv @ H.t() @ Dv_inv_sqrt
    return A
