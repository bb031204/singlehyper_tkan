"""最小化冒烟测试：检查关键模块可导入与前向计算。"""
import torch
from src.models.single_hyper_tkan_model import SingleHyperTKAN


def main():
    B, T, N, F = 2, 12, 16, 4
    C_weather = 1
    x = torch.randn(B, T, N, F)
    x_raw = torch.randn(B, T, N, C_weather)
    H = torch.eye(N)
    W = torch.ones(N)

    # 静态模式
    model = SingleHyperTKAN(
        input_dim=F, output_dim=1, d_model=16, hidden_channels=16,
        tkan_hidden=16, pred_steps=12, grid_size=5, spline_order=3)
    y = model(x, H, W, output_length=12)
    assert y.shape == (B, 12, N, 1), f"Unexpected output shape: {y.shape}"
    print('smoke test passed: static mode')

    # 动态语义模式
    ds_cfg = {'enabled': True, 'semantic_weight_lambda': 0.3,
              'similarity': 'cosine', 'normalize_dynamic_similarity': True}
    model_dyn = SingleHyperTKAN(
        input_dim=F, output_dim=1, d_model=16, hidden_channels=16,
        tkan_hidden=16, pred_steps=12, grid_size=5, spline_order=3,
        dynamic_semantic_cfg=ds_cfg)
    edges = [[i, (i+1) % N, (i+2) % N] for i in range(N)]
    model_dyn.register_edges(edges, torch.device('cpu'))
    y_dyn = model_dyn(x, H, W, output_length=12, x_raw=x_raw)
    assert y_dyn.shape == (B, 12, N, 1), f"Unexpected dynamic shape: {y_dyn.shape}"
    print('smoke test passed: dynamic semantic mode')

    # bypass 模式
    model_bypass = SingleHyperTKAN(
        input_dim=F, output_dim=1, d_model=16, hidden_channels=16,
        tkan_hidden=16, pred_steps=12, bypass_spatial=True)
    y2 = model_bypass(x, H, W, output_length=12)
    assert y2.shape == (B, 12, N, 1), f"Unexpected bypass shape: {y2.shape}"
    print('smoke test passed: bypass_spatial mode')

    print(f'Model params: {model.get_num_parameters():,}')


if __name__ == '__main__':
    main()
