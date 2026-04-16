import torch
import torch.nn as nn
from ..graph.hypergraph_utils import normalized_hypergraph_matrix, precompute_hypergraph_cache
from ..graph.dynamic_semantic_features import (
    build_window_dynamic_features, dynamic_semantic_similarity)
from .single_hyper_conv import SingleHyperConv
from .tkan import TKANLayer


class DynamicEdgeWeighter(nn.Module):
    """在地理候选边内部，用当前窗口动态特征修正超边权重。

    对每条超边 e（中心节点 = edge index），计算边内节点与中心节点的
    动态语义相似度均值，作为修正项：
        W_dynamic[b, e] = W_static[e] * (1 + lambda * mean_sim[b, e])
    """

    def __init__(self, lam: float = 0.3, similarity: str = 'cosine',
                 normalize_sim: bool = True):
        super().__init__()
        self.lam = lam
        self.similarity = similarity
        self.normalize_sim = normalize_sim

    @torch.no_grad()
    def forward(self, x_raw: torch.Tensor, W: torch.Tensor,
                edge_tensors: dict) -> torch.Tensor:
        """
        Args:
            x_raw: (B, T, N, C) 原始观测（投影前）
            W: (E,) 静态边权
            edge_tensors: 预缓存的边结构（members, centers, member_edge_ids, counts）

        Returns:
            W_dynamic: (B, E)
        """
        feat = build_window_dynamic_features(x_raw)  # (B, N, D)
        B, N, D = feat.shape
        E = W.shape[0]

        edge_members = edge_tensors['members']
        edge_centers = edge_tensors['centers']
        member_edge_ids = edge_tensors['member_edge_ids']
        counts = edge_tensors['counts']  # (1, E)

        center_feat = feat[:, edge_centers, :]          # (B, E, D)
        member_feat = feat[:, edge_members, :]          # (B, total_members, D)
        center_for_members = center_feat[:, member_edge_ids, :]

        if self.similarity == 'cosine':
            sim = nn.functional.cosine_similarity(
                member_feat, center_for_members, dim=-1)
        else:
            sim = (member_feat * center_for_members).sum(-1) / (D ** 0.5)

        sim = sim.clamp(min=0.0, max=1.0)

        mean_sim = torch.zeros(B, E, device=feat.device, dtype=feat.dtype)
        mean_sim.scatter_add_(1, member_edge_ids.unsqueeze(0).expand(B, -1), sim)
        mean_sim = mean_sim / counts.clamp(min=1.0)

        if self.normalize_sim:
            sim_min = mean_sim.min(dim=1, keepdim=True).values
            sim_max = mean_sim.max(dim=1, keepdim=True).values
            mean_sim = (mean_sim - sim_min) / (sim_max - sim_min + 1e-8)

        W_dynamic = W.unsqueeze(0) * (1.0 + self.lam * mean_sim)
        return W_dynamic


class SingleHyperTKAN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 32,
                 hidden_channels: int = 32, tkan_hidden: int = 32,
                 sub_kan_layers: int = 3, use_kan: bool = True,
                 dropout: float = 0.1, pred_steps: int = 12,
                 grid_size: int = 5, spline_order: int = 3,
                 bypass_spatial: bool = False,
                 dynamic_semantic_cfg: dict = None):
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

        # 动态语义加权模块
        ds = dynamic_semantic_cfg or {}
        self.use_dynamic_semantic = ds.get('enabled', False)
        if self.use_dynamic_semantic:
            self.dynamic_weighter = DynamicEdgeWeighter(
                lam=ds.get('semantic_weight_lambda', 0.3),
                similarity=ds.get('similarity', 'cosine'),
                normalize_sim=ds.get('normalize_dynamic_similarity', True),
            )

        self._cached_A_static = None
        self._hypergraph_cache = None
        self._edge_tensors = None
        self._w_dyn_accumulator = None

    def register_edges(self, edges: list, device: torch.device):
        """将候选边列表转换为 GPU tensor 缓存，避免每次 forward 重新构建。"""
        members = []
        offsets = [0]
        centers = []
        for e_idx, nodes in enumerate(edges):
            members.extend(nodes)
            offsets.append(offsets[-1] + len(nodes))
            centers.append(e_idx)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
        E = len(edges)
        edge_sizes = offsets_t[1:] - offsets_t[:-1]
        edge_ids = torch.arange(E, device=device)
        member_edge_ids = edge_ids.repeat_interleave(edge_sizes)
        self._edge_tensors = {
            'members': torch.tensor(members, dtype=torch.long, device=device),
            'centers': torch.tensor(centers, dtype=torch.long, device=device),
            'offsets': offsets_t,
            'member_edge_ids': member_edge_ids,
            'counts': edge_sizes.float().unsqueeze(0),  # (1, E)
        }

    def forward(self, x, H, W, output_length: int = 12, x_raw: torch.Tensor = None):
        """
        Args:
            x:   (B, T, N, F) 模型输入（含 context 拼接后的特征）
            H:   (N, E) 超图关联矩阵
            W:   (E,) 静态超边权重
            x_raw: (B, T, N, C) 原始观测（仅气象变量，不含 context），
                   用于动态语义特征提取。若 None 则退化为静态图。
        """
        B, T, N, _ = x.shape
        x = self.in_proj(x)

        if not self.bypass_spatial:
            use_dynamic = (self.use_dynamic_semantic
                           and x_raw is not None
                           and self._edge_tensors is not None)

            if use_dynamic:
                W_dyn = self.dynamic_weighter(
                    x_raw, W.to(x.device), self._edge_tensors)
                w_min, w_max = float(W_dyn.min()), float(W_dyn.max())
                w_mean, w_std = float(W_dyn.mean()), float(W_dyn.std())
                if self._w_dyn_accumulator is None:
                    self._w_dyn_accumulator = {
                        'min': w_min, 'max': w_max,
                        'sum': w_mean, 'sum_sq': w_mean ** 2, 'count': 1,
                    }
                else:
                    acc = self._w_dyn_accumulator
                    acc['min'] = min(acc['min'], w_min)
                    acc['max'] = max(acc['max'], w_max)
                    acc['sum'] += w_mean
                    acc['sum_sq'] += w_mean ** 2
                    acc['count'] += 1
                # 预缓存只依赖 H 的中间量
                H_dev = H.to(x.device)
                if self._hypergraph_cache is None:
                    self._hypergraph_cache = precompute_hypergraph_cache(H_dev)
                A = normalized_hypergraph_matrix(
                    H_dev, W_dyn, cache=self._hypergraph_cache)  # (B, N, N)
            else:
                if self._cached_A_static is None or self._cached_A_static.shape[-1] != N:
                    self._cached_A_static = normalized_hypergraph_matrix(
                        H.to(x.device), W.to(x.device))
                A = self._cached_A_static  # (N, N)

            # 合并 B*T 消除 Python for 循环，一次完成所有时间步的空间卷积
            x_flat = x.reshape(B * T, N, -1)  # (B*T, N, D)
            if A.dim() == 2:
                A_exp = A  # (N, N) 静态，einsum 自动广播
            else:
                # (B, N, N) → (B*T, N, N)：每个样本的 T 个时间步共享同一个 A
                A_exp = A.unsqueeze(1).expand(B, T, N, N).reshape(B * T, N, N)
            x_flat = self.spatial(x_flat, A_exp)
            x = x_flat.reshape(B, T, N, -1)
        else:
            x = self.spatial_bypass(x)

        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        h = self.temporal(x)
        y = self.head(h).view(B, N, self.pred_steps, self.output_dim)
        y = y.permute(0, 2, 1, 3).contiguous()
        if output_length != self.pred_steps:
            y = y[:, :output_length]
        return y

    def get_w_dynamic_stats(self):
        """返回本轮累积的 W_dynamic 统计并重置累积器。"""
        acc = self._w_dyn_accumulator
        if acc is None or acc['count'] == 0:
            return None
        n = acc['count']
        avg_mean = acc['sum'] / n
        avg_std = (acc['sum_sq'] / n - avg_mean ** 2) ** 0.5 if n > 1 else 0.0
        stats = {
            'min': acc['min'],
            'max': acc['max'],
            'mean': avg_mean,
            'std_of_batch_means': abs(avg_std),
            'num_batches': n,
        }
        self._w_dyn_accumulator = None
        return stats

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
