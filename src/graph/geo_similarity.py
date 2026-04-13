import numpy as np


def haversine_distance_matrix(lonlat: np.ndarray) -> np.ndarray:
    """输入 [N,2] (lon,lat) 或 (lat,lon) 近似处理"""
    a = lonlat.astype(np.float64)
    lon = np.radians(a[:, 0])
    lat = np.radians(a[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
    return (6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))).astype(np.float32)


def geo_similarity_from_position(position: np.ndarray, sigma: float = 500.0) -> np.ndarray:
    d = haversine_distance_matrix(position)
    s = np.exp(-d / max(sigma, 1e-6))
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s.astype(np.float32)
