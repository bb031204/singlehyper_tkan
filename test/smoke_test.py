"""最小化冒烟测试：检查关键模块可导入与前向计算。"""
import torch
from src.models.single_hyper_tkan_model import SingleHyperTKAN


def main():
    B, T, N, F = 2, 12, 16, 4
    x = torch.randn(B, T, N, F)
    H = torch.eye(N)
    W = torch.ones(N)

    model = SingleHyperTKAN(
        input_dim=F, output_dim=1, d_model=16, hidden_channels=16,
        tkan_hidden=16, pred_steps=12, grid_size=5, spline_order=3)
    y = model(x, H, W, output_length=12)
    assert y.shape == (B, 12, N, 1), f"Unexpected output shape: {y.shape}"
    print('smoke test passed: normal mode')

    model_bypass = SingleHyperTKAN(
        input_dim=F, output_dim=1, d_model=16, hidden_channels=16,
        tkan_hidden=16, pred_steps=12, bypass_spatial=True)
    y2 = model_bypass(x, H, W, output_length=12)
    assert y2.shape == (B, 12, N, 1), f"Unexpected bypass shape: {y2.shape}"
    print('smoke test passed: bypass_spatial mode')

    print(f'Model params: {model.get_num_parameters():,}')


if __name__ == '__main__':
    main()
