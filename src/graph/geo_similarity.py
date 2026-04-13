import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def haversine_distance_matrix(lonlat: np.ndarray) -> np.ndarray:
    """输入 [N,2] (lon,lat)，返回 [N,N] 距离矩阵(km)。"""
    a = lonlat.astype(np.float64)
    lon = np.radians(a[:, 0])
    lat = np.radians(a[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    h = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2)
    return (6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))).astype(np.float32)


def geo_similarity_from_position(position: np.ndarray, sigma: float = 500.0) -> np.ndarray:
    d = haversine_distance_matrix(position)
    s = np.exp(-d / max(sigma, 1e-6))
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s.astype(np.float32)


def plot_geo_similarity_stats(position: np.ndarray, save_dir: str,
                              sigma: float = 500.0) -> dict:
    """可视化地理相似度分布统计并返回统计字典。"""
    os.makedirs(save_dir, exist_ok=True)
    d = haversine_distance_matrix(position)
    s = geo_similarity_from_position(position, sigma)

    upper_idx = np.triu_indices_from(d, k=1)
    upper_d = d[upper_idx]
    upper_s = s[upper_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(upper_d.ravel(), bins=50, color='steelblue', edgecolor='white')
    axes[0].set_title('Station Pairwise Distance (km)')
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Count')

    axes[1].hist(upper_s.ravel(), bins=50, color='coral', edgecolor='white')
    axes[1].set_title('Geographic Similarity Distribution')
    axes[1].set_xlabel('Similarity')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geo_similarity_stats.png'), dpi=150)
    plt.close()

    return {
        'distance_min': float(upper_d.min()),
        'distance_max': float(upper_d.max()),
        'distance_mean': float(upper_d.mean()),
        'similarity_min': float(upper_s.min()),
        'similarity_max': float(upper_s.max()),
        'similarity_mean': float(upper_s.mean()),
    }
