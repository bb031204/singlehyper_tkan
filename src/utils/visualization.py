import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, val_losses, save_path: str,
                    title: str = 'Loss Curve'):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions(predictions: np.ndarray, targets: np.ndarray,
                     save_path: str, num_samples: int = 5):
    """绘制预测值 vs 真实值对比图。

    predictions/targets: [S, T, N, C] 或 [S, T, N]
    """
    if predictions.ndim == 4:
        predictions = predictions[..., 0]
        targets = targets[..., 0]

    S, T, N = predictions.shape
    num_samples = min(num_samples, S)
    station_idx = N // 2

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples),
                             squeeze=False)
    for i in range(num_samples):
        ax = axes[i, 0]
        ax.plot(range(1, T + 1), targets[i, :, station_idx],
                'b-o', markersize=3, label='Actual')
        ax.plot(range(1, T + 1), predictions[i, :, station_idx],
                'r--s', markersize=3, label='Predicted')
        ax.set_title(f'Sample {i + 1}, Station {station_idx}')
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_step_metrics(step_metrics: dict, save_path: str,
                      metric_name: str = 'MAE'):
    """绘制每个预测步的指标柱状图。

    step_metrics: {'1': value, '2': value, ...}
    """
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
