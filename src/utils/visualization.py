import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

ELEMENT_UNITS = {
    'Temperature': '°C',
    'Cloud': 'fraction',
    'Humidity': '%',
    'Wind': 'm/s',
}

# 配色
_COLOR_GT   = '#2166AC'
_COLOR_PRED = '#D6604D'
_COLOR_FILL = '#FDDBC7'
_COLOR_HOR  = '#4DAF4A'


def plot_loss_curve(train_losses, val_losses, save_path: str,
                    title: str = 'Loss Curve'):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    if val_losses:
        best_val = min(val_losses)
        best_ep = val_losses.index(best_val) + 1
        plt.axvline(x=best_ep, color='g', linestyle='--', alpha=0.5,
                     label=f'Best Epoch: {best_ep}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------ helpers
def _select_representative_samples(pred: np.ndarray, target: np.ndarray,
                                   num: int) -> List[int]:
    B = pred.shape[0]
    mae = np.mean(np.abs(pred - target), axis=(1, 2, 3))
    order = np.argsort(mae)
    sel = set()
    sel.add(order[0])
    if B > 1:
        sel.add(order[max(1, B // 10)])
    if B > 2:
        sel.add(order[B // 2])
    if B > 3:
        sel.add(order[min(B - 1, int(B * 0.9))])
    i = 0
    while len(sel) < num and i < B:
        sel.add(order[i]); i += 1
    return sorted(list(sel))[:num]


def _select_diverse_stations(pred: np.ndarray, target: np.ndarray,
                             num: int) -> List[int]:
    N = pred.shape[2]
    mae = np.mean(np.abs(pred - target), axis=(0, 1, 3))
    order = np.argsort(mae)
    sel = set()
    sel.add(order[0])
    if N > 1:
        sel.add(order[max(1, N // 10)])
    if N > 2:
        sel.add(order[N // 2])
    if N > 3:
        sel.add(order[min(N - 1, int(N * 0.75))])
    i = 0
    while len(sel) < num and i < N:
        sel.add(order[i]); i += 1
    return sorted(list(sel))[:num]


def _auto_unit(values: np.ndarray, element: str) -> Tuple[np.ndarray, str]:
    """如果数据是开尔文温度则自动转摄氏度，返回 (转换后值, 单位)。"""
    mean_v = np.mean(values)
    if 180 < mean_v < 350:
        return values - 273.15, '°C'
    return values, ELEMENT_UNITS.get(element, '')


# -------------------------------------------------------- main prediction plot
def plot_predictions(pred: np.ndarray, target: np.ndarray, save_path: str,
                     num_samples: int = 4, num_stations: int = 4,
                     horizon_steps: Optional[List[int]] = None,
                     element: str = 'Temperature'):
    """绘制预测结果：主拼图 + 综合分析图（与 hyper_kan 生成逻辑一致）。

    pred / target: (B, T, N, F)  或 (B, T, N)
    """
    if pred.ndim == 3:
        pred = pred[..., np.newaxis]
        target = target[..., np.newaxis]

    B, T, N, F = pred.shape
    num_samples = min(num_samples, B)
    num_stations = min(num_stations, N)
    if horizon_steps is None:
        horizon_steps = [3, 6, 12] if T >= 12 else [T // 3, T // 2, T]

    sample_idx = _select_representative_samples(pred, target, num_samples)
    station_idx = _select_diverse_stations(pred, target, num_stations)

    pred_p, unit = _auto_unit(pred, element)
    target_p, _  = _auto_unit(target, element)
    overall_mae = np.mean(np.abs(pred_p - target_p))

    # ==================== 图 1：样本 × 站点 拼图 ====================
    fig, axes = plt.subplots(num_samples, num_stations,
                             figsize=(5.5 * num_stations, 4 * num_samples))
    if num_samples == 1 and num_stations == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = axes.reshape(1, -1)
    elif num_stations == 1:
        axes = axes.reshape(-1, 1)

    ts = np.arange(1, T + 1)
    for ri, si in enumerate(sample_idx):
        for ci, sj in enumerate(station_idx):
            ax = axes[ri, ci]
            ps = pred_p[si, :, sj, 0]
            gs = target_p[si, :, sj, 0]

            ax.fill_between(ts, gs, ps, alpha=0.2, color=_COLOR_FILL,
                            label='Error')
            ax.plot(ts, gs, color=_COLOR_GT, linewidth=2.2, marker='o',
                    markersize=3.5, label='Ground Truth', zorder=3)
            ax.plot(ts, ps, color=_COLOR_PRED, linewidth=2.2, marker='s',
                    markersize=3, linestyle='--', label='Prediction', zorder=3)

            for h in horizon_steps:
                if h <= T:
                    ax.axvline(x=h, color=_COLOR_HOR, linestyle=':',
                               alpha=0.4, linewidth=1)

            sub_mae = np.mean(np.abs(ps - gs))
            y_label = f'{element} ({unit})' if unit else element
            ax.set_xlabel('Forecast Hour', fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'Sample {si+1}, Station {sj+1}  |  '
                         f'MAE={sub_mae:.2f}{unit}',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best', framealpha=0.8)
            ax.grid(True, alpha=0.2, linestyle='-')
            ax.set_xticks(ts)
            y_lo = min(ps.min(), gs.min())
            y_hi = max(ps.max(), gs.max())
            margin = max((y_hi - y_lo) * 0.15, 0.5)
            ax.set_ylim(y_lo - margin, y_hi + margin)

    fig.suptitle(
        f'SingleHyperTKAN {element} Prediction  '
        f'(Overall MAE={overall_mae:.3f}{unit})',
        fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Predictions plot saved to {save_path}")

    # ==================== 图 2：综合分析 6 面板 ====================
    analysis_path = save_path.replace('.png', '_analysis.png')
    fig2 = plt.figure(figsize=(18, 12))
    gs_layout = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    flat_p = pred_p[:, :, :, 0].flatten()
    flat_g = target_p[:, :, :, 0].flatten()
    n_pts = len(flat_p)
    if n_pts > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_pts, 50000, replace=False)
        fp, fg = flat_p[idx], flat_g[idx]
    else:
        fp, fg = flat_p, flat_g

    # (a) 散点图
    ax1 = fig2.add_subplot(gs_layout[0, 0])
    ax1.scatter(fg, fp, alpha=0.08, s=3, c='#4393C3', rasterized=True)
    vmin = min(fg.min(), fp.min()); vmax = max(fg.max(), fp.max())
    ax1.plot([vmin, vmax], [vmin, vmax], 'r-', linewidth=1.5,
             label='y = x (Perfect)')
    ax1.set_xlabel(f'Ground Truth ({unit})')
    ax1.set_ylabel(f'Prediction ({unit})')
    ax1.set_title('(a) Prediction vs Ground Truth', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal', adjustable='box')

    # (b) 误差分布
    ax2 = fig2.add_subplot(gs_layout[0, 1])
    errors = fp - fg
    ax2.hist(errors, bins=80, color='#92C5DE', edgecolor='#4393C3',
             alpha=0.8, density=True)
    ax2.axvline(x=0, color='red', linewidth=1.5, linestyle='--')
    mean_err = np.mean(flat_p - flat_g)
    std_err = np.std(flat_p - flat_g)
    ax2.axvline(x=mean_err, color=_COLOR_PRED, linewidth=1.2, linestyle='-',
                label=f'Mean={mean_err:.3f}')
    ax2.set_xlabel(f'Prediction Error ({unit})')
    ax2.set_ylabel('Density')
    ax2.set_title(f'(b) Error Distribution (std={std_err:.3f})',
                  fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)

    # (c) 各步 MAE
    ax3 = fig2.add_subplot(gs_layout[0, 2])
    h_mae = [np.mean(np.abs(pred_p[:, t, :, 0] - target_p[:, t, :, 0]))
             for t in range(T)]
    colors_c = ['#D6604D' if (t+1) in horizon_steps else '#92C5DE'
                for t in range(T)]
    ax3.bar(range(1, T+1), h_mae, color=colors_c, edgecolor='white',
            linewidth=0.5)
    ax3.set_xlabel('Forecast Step (hour)')
    ax3.set_ylabel(f'MAE ({unit})')
    ax3.set_title('(c) MAE by Forecast Horizon', fontweight='bold')
    ax3.set_xticks(range(1, T+1))
    ax3.grid(True, alpha=0.2, axis='y')
    for t in range(T):
        if (t+1) in horizon_steps:
            ax3.text(t+1, h_mae[t]+0.02, f'{h_mae[t]:.3f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    # (d) 站点级 MAE 箱线图
    ax4 = fig2.add_subplot(gs_layout[1, 0])
    st_mae = np.mean(np.abs(pred_p[:, :, :, 0] - target_p[:, :, :, 0]),
                     axis=(0, 1))
    bp = ax4.boxplot(st_mae, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='#92C5DE', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax4.set_ylabel(f'MAE ({unit})')
    ax4.set_title(f'(d) Station-level MAE Distribution (N={N})',
                  fontweight='bold')
    ax4.set_xticklabels(['All Stations'])
    ax4.grid(True, alpha=0.2, axis='y')
    q25, med, q75 = np.percentile(st_mae, [25, 50, 75])
    ax4.text(1.25, med,
             f'Median: {med:.3f}\nQ25: {q25:.3f}\nQ75: {q75:.3f}',
             fontsize=9, va='center')

    # (e) Best case / (f) Hard case
    pair_mae = np.mean(np.abs(pred_p[:, :, :, 0] - target_p[:, :, :, 0]),
                       axis=1)
    best_si, best_sj = np.unravel_index(np.argmin(pair_mae), pair_mae.shape)
    worst_si, worst_sj = np.unravel_index(np.argmax(pair_mae), pair_mae.shape)

    for panel, (s_i, s_j, label) in enumerate([
        (best_si, best_sj, 'Best'),
        (worst_si, worst_sj, 'Hard'),
    ]):
        ax = fig2.add_subplot(gs_layout[1, 1 + panel])
        p_s = pred_p[s_i, :, s_j, 0]
        g_s = target_p[s_i, :, s_j, 0]
        ax.fill_between(ts, g_s, p_s, alpha=0.2, color=_COLOR_FILL)
        ax.plot(ts, g_s, color=_COLOR_GT, linewidth=2.5, marker='o',
                markersize=5, label='Ground Truth', zorder=3)
        ax.plot(ts, p_s, color=_COLOR_PRED, linewidth=2.5, marker='s',
                markersize=4, linestyle='--', label='Prediction', zorder=3)
        case_mae = pair_mae[s_i, s_j]
        tag = chr(ord('e') + panel)
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel(f'{element} ({unit})')
        ax.set_title(f'({tag}) {label} Case (MAE={case_mae:.3f}{unit})',
                     fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        ax.set_xticks(ts)

    rmse = np.sqrt(np.mean((pred_p - target_p) ** 2))
    fig2.suptitle(
        f'SingleHyperTKAN {element} Prediction Analysis  |  '
        f'Overall MAE={overall_mae:.3f}{unit}  RMSE={rmse:.3f}{unit}',
        fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Analysis plot saved to {analysis_path}")


def plot_step_metrics(step_metrics: dict, save_path: str,
                      metric_name: str = 'MAE'):
    """绘制每个预测步的指标柱状图。"""
    steps = sorted(step_metrics.keys(), key=lambda x: int(x))
    values = [step_metrics[s] for s in steps]
    labels = [str(s) for s in steps]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color='steelblue', edgecolor='white')
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    plt.title(f'Per-Step {metric_name}')
    plt.xlabel('Prediction Step')
    plt.ylabel(metric_name)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
