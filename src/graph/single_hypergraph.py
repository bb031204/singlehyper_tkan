import os
import logging
import numpy as np
import torch
from .station_statistics import build_station_statistics, semantic_similarity
from .geo_similarity import geo_similarity_from_position

logger = logging.getLogger('SingleHyperTKAN')


def _edge_from_percentile(row_sim, center, percentile, min_size, max_size):
    """自适应局部百分位构边：每个站点根据自身相似度分布确定阈值。

    percentile=99.5 表示取该站相似度排名前 0.5% 的邻居。
    """
    mask = np.ones(len(row_sim), dtype=bool)
    mask[center] = False
    local_threshold = np.percentile(row_sim[mask], percentile)

    idx = np.where(row_sim >= local_threshold)[0].tolist()
    if center not in idx:
        idx.append(center)

    order = np.argsort(-row_sim)
    if len(idx) < min_size:
        for j in order:
            if int(j) not in idx:
                idx.append(int(j))
            if len(idx) >= min_size:
                break
    if len(idx) > max_size:
        idx = [int(i) for i in order[:max_size].tolist()]
        if center not in idx:
            idx[-1] = center
    return sorted(set(idx))


def build_single_hypergraph(train_x: np.ndarray, position: np.ndarray, cfg: dict):
    gcfg = cfg['graph']['single_hypergraph']
    alpha = float(gcfg['alpha'])
    min_size = int(gcfg['min_hyperedge_size'])
    max_size = int(gcfg['max_hyperedge_size'])

    p_geo = float(gcfg.get('geo_percentile', 99.5))
    p_sem = float(gcfg.get('sem_percentile', 99.5))
    p_fusion = float(gcfg.get('fusion_percentile', 99.5))

    sem_feat = build_station_statistics(train_x)
    s_sem = semantic_similarity(sem_feat, gcfg.get('semantic_similarity', 'cosine'))
    s_geo = geo_similarity_from_position(position)
    s_fusion = alpha * s_sem + (1.0 - alpha) * s_geo

    N = s_fusion.shape[0]

    # 打印相似度矩阵分位数分布
    pcts = [50, 75, 90, 95, 99, 99.5]
    for name, mat in [('Geo', s_geo), ('Semantic', s_sem), ('Fusion', s_fusion)]:
        vals = mat[np.triu_indices(N, k=1)]
        qs = np.percentile(vals, pcts)
        q_str = '  '.join(f'P{p}={v:.4f}' for p, v in zip(pcts, qs))
        logger.info(f"  [{name}] similarity distribution: {q_str}")

    logger.info(f"  Adaptive percentile: geo={p_geo}  sem={p_sem}  fusion={p_fusion}")

    # 三种边均使用自适应局部百分位构边
    geo_edges = [_edge_from_percentile(s_geo[i], i, p_geo, min_size, max_size)
                 for i in range(N)]
    sem_edges = [_edge_from_percentile(s_sem[i], i, p_sem, min_size, max_size)
                 for i in range(N)]

    edges = []
    weights = []
    for i in range(N):
        e = _edge_from_percentile(s_fusion[i], i, p_fusion, min_size, max_size)
        edges.append(e)
        sub = s_fusion[i, e]
        weights.append(float(np.mean(sub)))

    E = len(edges)
    H = np.zeros((N, E), dtype=np.float32)
    for e_idx, nodes in enumerate(edges):
        H[nodes, e_idx] = 1.0

    W = np.array(weights, dtype=np.float32)

    def _edge_stats(elist):
        sizes = [len(e) for e in elist]
        return {'min': int(min(sizes)), 'max': int(max(sizes)),
                'mean': float(np.mean(sizes))}

    stats = {
        'num_nodes': N,
        'num_edges': E,
        'edge_size_min': int(min(len(e) for e in edges)),
        'edge_size_max': int(max(len(e) for e in edges)),
        'edge_size_mean': float(np.mean([len(e) for e in edges])),
        'geo_edge': _edge_stats(geo_edges),
        'sem_edge': _edge_stats(sem_edges),
        'static_W_min': float(W.min()),
        'static_W_max': float(W.max()),
        'static_W_mean': float(W.mean()),
        'static_W_std': float(W.std()),
    }
    return torch.from_numpy(H), torch.from_numpy(W), stats, edges


def build_or_load_single_hypergraph(train_x: np.ndarray, position: np.ndarray, cfg: dict):
    gcfg = cfg['graph']['single_hypergraph']
    os.makedirs(gcfg['cache_dir'], exist_ok=True)
    N = position.shape[0]
    pg = gcfg.get('geo_percentile', 99.5)
    ps = gcfg.get('sem_percentile', 99.5)
    pf = gcfg.get('fusion_percentile', 99.5)
    cache_name = (f"single_{cfg['meta']['element']}_N{N}"
                  f"_a{gcfg['alpha']}_pg{pg}_ps{ps}_pf{pf}.npz")
    cache_path = os.path.join(gcfg['cache_dir'], cache_name)

    if gcfg.get('use_cache', True) and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        H = torch.from_numpy(data['H'].astype(np.float32))
        W = torch.from_numpy(data['W'].astype(np.float32))
        stats = dict(data['stats'].item())
        edges = [list(map(int, e)) for e in data['edges']]
        return H, W, stats, edges

    H, W, stats, edges = build_single_hypergraph(train_x, position, cfg)
    np.savez(cache_path, H=H.numpy(), W=W.numpy(),
             stats=np.array(stats, dtype=object),
             edges=np.array(edges, dtype=object))
    return H, W, stats, edges
