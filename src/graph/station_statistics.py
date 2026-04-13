import numpy as np
from sklearn.preprocessing import StandardScaler


def build_station_statistics(train_x: np.ndarray) -> np.ndarray:
    """从训练集构建站点统计特征，train_x: [S,T,N,F] 或 [T,N,F]"""
    if train_x.ndim == 4:
        data = train_x.reshape(-1, train_x.shape[2], train_x.shape[3])  # [S*T,N,F]
    elif train_x.ndim == 3:
        data = train_x
    else:
        raise ValueError(f"Unexpected train_x shape: {train_x.shape}")

    N = data.shape[1]
    feats = []
    for i in range(N):
        v = data[:, i, :].reshape(-1)
        dv = np.diff(v) if v.shape[0] > 1 else np.array([0.0])
        f = [
            np.mean(v), np.std(v), np.min(v), np.max(v), np.median(v),
            np.max(v) - np.min(v), np.mean(dv), np.std(dv)
        ]
        feats.append(f)
    feats = np.array(feats, dtype=np.float32)
    return StandardScaler().fit_transform(feats)


def semantic_similarity(features: np.ndarray, mode: str = 'cosine') -> np.ndarray:
    x = features
    if mode == 'cosine':
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        s = (x @ x.T) / (n @ n.T)
    elif mode == 'pearson':
        s = np.corrcoef(x)
    elif mode == 'euclidean':
        d = np.sqrt(np.maximum(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1), 0.0))
        s = np.exp(-d / (np.std(d) + 1e-8))
    else:
        raise ValueError(mode)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s.astype(np.float32)
