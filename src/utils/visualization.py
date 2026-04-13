import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, val_losses, save_path: str, title: str = 'Loss Curve'):
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
