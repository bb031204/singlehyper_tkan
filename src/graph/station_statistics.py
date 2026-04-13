import numpy as np
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler


def build_station_statistics(train_x: np.ndarray) -> np.ndarray:
    """从训练集构建站点统计特征。

    train_x: [S,T,N,F] 或 [T,N,F]
    返回标准化后的特征矩阵 [N, num_features]

    特征包括：mean, std, min, max, median, range,
    一阶差分均值, 一阶差分标准差, skew, kurtosis, 周期性振幅
    """
    if train_x.ndim == 4:
        data = train_x.reshape(-1, train_x.shape[2], train_x.shape[3])
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
            np.mean(v),
            np.std(v),
            np.min(v),
            np.max(v),
            np.median(v),
            np.max(v) - np.min(v),
            np.mean(dv),
            np.std(dv),
        ]
        if v.shape[0] > 2:
            f.append(float(sp_stats.skew(v)))
            f.append(float(sp_stats.kurtosis(v)))
        else:
            f.extend([0.0, 0.0])
        period = 24
        if v.shape[0] >= period * 2:
            n_full = v.shape[0] // period * period
            reshaped = v[:n_full].reshape(-1, period)
            hourly_mean = reshaped.mean(axis=0)
            f.append(float(np.std(hourly_mean)))
        else:
            f.append(0.0)
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
