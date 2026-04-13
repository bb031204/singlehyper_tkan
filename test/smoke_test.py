"""最小化冒烟测试：检查关键模块可导入与前向计算。"""
import torch
from src.models.single_hyper_tkan_model import SingleHyperTKAN


def main():
    B, T, N, F = 2, 12, 16, 4
    x = torch.randn(B, T, N, F)
    H = torch.eye(N)
    W = torch.ones(N)
    model = SingleHyperTKAN(input_dim=F, output_dim=1, d_model=16, hidden_channels=16, tkan_hidden=16, pred_steps=12)
    y = model(x, H, W, output_length=12)
    assert y.shape == (B, 12, N, 1), f"Unexpected output shape: {y.shape}"
    print('smoke test passed')


if __name__ == '__main__':
    main()
