import torch
import torch.nn as nn
from ..graph.hypergraph_utils import normalized_hypergraph_matrix
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
    def forward(self, x_raw: torch.Tensor, H: torch.Tensor, W: torch.Tensor,
                edge_members: torch.Tensor, edge_centers: torch.Tensor,
                edge_offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (B, T, N, C) 原始观测（投影前）
            H: (N, E) 关联矩阵
            W: (E,) 静态边权
            edge_members: (total_members,) 所有超边成员的 flat 索引
            edge_centers: (E,) 每条边的中心节点
            edge_offsets: (E+1,) CSR 偏移量

        Returns:
            W_dynamic: (B, E)
        """
        feat = build_window_dynamic_features(x_raw)  # (B, N, D)
        B, N, D = feat.shape
        E = W.shape[0]

        # 计算中心节点特征: (B, E, D)
        center_feat = feat[:, edge_centers, :]

        # 对每条边，计算边内成员与中心的局部语义相似度均值
        # 用向量化方式避免 python 循环
        member_feat = feat[:, edge_members, :]  # (B, total_members, D)

        # 中心特征按边展开到 member 级别
        edge_ids = torch.arange(E, device=feat.device)
        member_edge_ids = edge_ids.repeat_interleave(
            edge_offsets[1:] - edge_offsets[:-1])  # (total_members,)
        center_for_members = center_feat[:, member_edge_ids, :]  # (B, total_members, D)

        # cosine similarity per member
        if self.similarity == 'cosine':
            sim = nn.functional.cosine_similarity(
                member_feat, center_for_members, dim=-1)  # (B, total_members)
        else:
            sim = (member_feat * center_for_members).sum(-1) / (D ** 0.5)

        sim = sim.clamp(min=0.0, max=1.0)

        # scatter mean: 按边聚合
        mean_sim = torch.zeros(B, E, device=feat.device, dtype=feat.dtype)
        counts = (edge_offsets[1:] - edge_offsets[:-1]).float().unsqueeze(0)  # (1, E)
        mean_sim.scatter_add_(1, member_edge_ids.unsqueeze(0).expand(B, -1), sim)
        mean_sim = mean_sim / counts.clamp(min=1.0)

        if self.normalize_sim:
            # 每个样本内归一化到 [0, 1]
            sim_min = mean_sim.min(dim=1, keepdim=True).values
            sim_max = mean_sim.max(dim=1, keepdim=True).values
            mean_sim = (mean_sim - sim_min) / (sim_max - sim_min + 1e-8)

        W_dynamic = W.unsqueeze(0) * (1.0 + self.lam * mean_sim)  # (B, E)
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
        self._edge_tensors = None

    def register_edges(self, edges: list, device: torch.device):
        """将候选边列表转换为 GPU tensor 缓存，避免每次 forward 重新构建。"""
        members = []
        offsets = [0]
        centers = []
        for e_idx, nodes in enumerate(edges):
            members.extend(nodes)
            offsets.append(offsets[-1] + len(nodes))
            centers.append(e_idx)  # 每条边的中心节点 = 边索引（与构图逻辑一致）
        self._edge_tensors = {
            'members': torch.tensor(members, dtype=torch.long, device=device),
            'centers': torch.tensor(centers, dtype=torch.long, device=device),
            'offsets': torch.tensor(offsets, dtype=torch.long, device=device),
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
                    x_raw,
                    H.to(x.device), W.to(x.device),
                    self._edge_tensors['members'],
                    self._edge_tensors['centers'],
                    self._edge_tensors['offsets'],
                )
                A = normalized_hypergraph_matrix(H.to(x.device), W_dyn)  # (B, N, N)
            else:
                if self._cached_A_static is None or self._cached_A_static.shape[-1] != N:
                    self._cached_A_static = normalized_hypergraph_matrix(
                        H.to(x.device), W.to(x.device))
                A = self._cached_A_static  # (N, N)

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
