import os
import numpy as np
import torch
from .station_statistics import build_station_statistics, semantic_similarity
from .geo_similarity import geo_similarity_from_position


def _edge_from_threshold(row_sim, center, threshold, min_size, max_size):
    idx = np.where(row_sim >= threshold)[0].tolist()
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
    threshold = float(gcfg['threshold'])
    min_size = int(gcfg['min_hyperedge_size'])
    max_size = int(gcfg['max_hyperedge_size'])

    sem_feat = build_station_statistics(train_x)
    s_sem = semantic_similarity(sem_feat, gcfg.get('semantic_similarity', 'cosine'))
    s_geo = geo_similarity_from_position(position)
    s_fusion = alpha * s_sem + (1.0 - alpha) * s_geo

    N = s_fusion.shape[0]
    edges = []
    weights = []
    for i in range(N):
        e = _edge_from_threshold(s_fusion[i], i, threshold, min_size, max_size)
        edges.append(e)
        sub = s_fusion[i, e]
        weights.append(float(np.mean(sub)))

    E = len(edges)
    H = np.zeros((N, E), dtype=np.float32)
    for e_idx, nodes in enumerate(edges):
        H[nodes, e_idx] = 1.0

    W = np.array(weights, dtype=np.float32)
    stats = {
        'num_nodes': N,
        'num_edges': E,
        'edge_size_min': int(min(len(e) for e in edges)),
        'edge_size_max': int(max(len(e) for e in edges)),
        'edge_size_mean': float(np.mean([len(e) for e in edges]))
    }
    return torch.from_numpy(H), torch.from_numpy(W), stats, edges


def build_or_load_single_hypergraph(train_x: np.ndarray, position: np.ndarray, cfg: dict):
    gcfg = cfg['graph']['single_hypergraph']
    os.makedirs(gcfg['cache_dir'], exist_ok=True)
    N = position.shape[0]
    cache_name = f"single_{cfg['meta']['element']}_N{N}_a{gcfg['alpha']}_t{gcfg['threshold']}.npz"
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
