"""SingleHyper-TKAN 预测脚本。"""
import os
import sys
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[sys.argv.index('--gpu') + 1])

import argparse
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm

from src.data import (load_pkl_data, load_position_data, SpatioTemporalDataset,
                      DataPreprocessor, apply_element_settings,
                      validate_dataset_selection, truncate_stations,
                      subsample_data)
from src.graph import build_or_load_single_hypergraph
from src.models import SingleHyperTKAN
from src.utils import (setup_logger, compute_metrics, get_latest_checkpoint,
                       plot_predictions, plot_step_metrics)


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_latest_experiment(base_dir: str):
    if not os.path.exists(base_dir):
        return None
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def main(args):
    config = load_config(args.config)
    if not validate_dataset_selection(config):
        raise ValueError('dataset_selection必须且仅能有一个true')
    config = apply_element_settings(config)

    logger = setup_logger('SingleHyperTKAN-Predict', 'INFO',
                          output_dir=None, console=True, file=False)

    exp_dir = args.output_dir or infer_latest_experiment(
        config['output']['base_dir'])
    if exp_dir is None:
        raise FileNotFoundError('未找到实验输出目录，请先训练')

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = get_latest_checkpoint(os.path.join(exp_dir, 'checkpoints'))
    if ckpt is None:
        raise FileNotFoundError('未找到checkpoint')

    logger.info(f'Using experiment dir: {exp_dir}')
    logger.info(f'Using checkpoint: {ckpt}')

    preprocessor = DataPreprocessor.load(
        os.path.join(exp_dir, 'preprocessor.pkl'))

    context_features = config['data'].get('context_features', {})
    context_feature_mask = [
        context_features.get('use_longitude', True),
        context_features.get('use_latitude', True),
        context_features.get('use_altitude', True),
        context_features.get('use_year', True),
        context_features.get('use_month', True),
        context_features.get('use_day', True),
        context_features.get('use_hour', True),
        context_features.get('use_region', True),
    ]

    train_data = load_pkl_data(
        config['data']['train_path'], config['data']['use_context'],
        config['data']['context_dim'], config['data']['use_dim4'],
        context_feature_mask)
    test_data = load_pkl_data(
        config['data']['test_path'], config['data']['use_context'],
        config['data']['context_dim'], config['data']['use_dim4'],
        context_feature_mask)
    position = load_position_data(config['data']['position_path'])

    # --- 站点截取 ---
    num_stations = config['data'].get('num_stations')
    if num_stations:
        train_data = truncate_stations(train_data, num_stations)
        test_data = truncate_stations(test_data, num_stations)
        if position is not None and position.shape[0] > num_stations:
            position = position[:num_stations]

    # --- 测试集采样 ---
    test_ratio = config['data'].get('test_sample_ratio', 1.0)
    if test_ratio < 1.0:
        test_data = subsample_data(test_data, test_ratio)

    # 用已 fit 的 preprocessor 变换数据（保证与训练时一致）
    train_data_processed = preprocessor.transform(train_data)
    test_data = preprocessor.transform(test_data)

    H, W, _ = build_or_load_single_hypergraph(
        train_data_processed['x'], position, config)

    dataset = SpatioTemporalDataset(
        x=test_data['x'], y=test_data.get('y'),
        context=test_data.get('context'),
        input_window=config['data']['input_window'],
        output_window=config['data']['output_window'],
        stride=1, concat_context=config['data']['use_context'])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['data']['batch_size'], shuffle=False)

    weather_dim = train_data['x'].shape[-1]
    ctx_dim = (train_data['context'].shape[-1]
               if (config['data']['use_context']
                   and train_data.get('context') is not None) else 0)
    input_dim = weather_dim + ctx_dim

    ablation = config.get('ablation', {})
    model = SingleHyperTKAN(
        input_dim=input_dim,
        output_dim=config['model']['output_projection']['output_dim'],
        d_model=config['model']['input_projection']['d_model'],
        hidden_channels=config['model']['spatial']['hidden_channels'],
        tkan_hidden=config['model']['temporal']['hidden_size'],
        sub_kan_layers=config['model']['temporal']['sub_kan_layers'],
        use_kan=(config['model']['use_kan']
                 and not ablation.get('disable_kan', False)),
        dropout=config['model']['temporal']['dropout'],
        pred_steps=config['data']['output_window'],
        grid_size=config['model']['temporal'].get('kan_grid_size', 5),
        spline_order=config['model']['temporal'].get('kan_spline_order', 3),
        bypass_spatial=ablation.get('disable_single_hypergraph', False),
    )

    device = args.device or config['meta']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    model = model.to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    all_inputs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting', colour='green'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            p = model(x, H.to(device), W.to(device),
                      output_length=y.shape[1])
            all_inputs.append(x.cpu().numpy())
            all_preds.append(p.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    inputs = np.concatenate(all_inputs, 0)
    preds = np.concatenate(all_preds, 0)
    targets = np.concatenate(all_targets, 0)

    preds_inv = preprocessor.inverse_transform(preds)
    targets_inv = preprocessor.inverse_transform(targets)

    pred_t = torch.from_numpy(preds_inv).float()
    target_t = torch.from_numpy(targets_inv).float()
    overall = compute_metrics(
        pred_t, target_t, metrics=config['evaluation']['metrics'])
    by_step = {}
    for t in range(pred_t.shape[1]):
        step_m = compute_metrics(
            pred_t[:, t:t + 1], target_t[:, t:t + 1], metrics=['mae', 'rmse'])
        by_step[str(t + 1)] = step_m

    np.savez(os.path.join(exp_dir, 'predictions.npz'),
             inputs=inputs, predictions=preds_inv, targets=targets_inv)
    with open(os.path.join(exp_dir, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'overall': overall, 'by_step': by_step}, f,
                  indent=2, ensure_ascii=False)
    with open(os.path.join(exp_dir, 'test_summary.txt'), 'w',
              encoding='utf-8') as f:
        f.write('SingleHyperTKAN Test Summary\n')
        f.write('=' * 40 + '\n')
        f.write(f"Overall: {json.dumps(overall, indent=2)}\n\n")
        f.write("Per-step MAE:\n")
        for k in sorted(by_step.keys(), key=int):
            f.write(f"  Step {k}: {by_step[k]}\n")

    # --- 可视化 ---
    if config['evaluation'].get('visualize', True):
        plot_predictions(
            preds_inv, targets_inv,
            os.path.join(exp_dir, 'predictions_plot.png'),
            num_samples=config['evaluation'].get('num_samples', 5))
        step_mae_dict = {k: v['mae'] for k, v in by_step.items()}
        plot_step_metrics(
            step_mae_dict,
            os.path.join(exp_dir, 'step_mae.png'), 'MAE')
        logger.info("Saved prediction visualizations")

    logger.info(f"Overall metrics: {overall}")
    for k in sorted(by_step.keys(), key=int):
        logger.info(f"  Step {k}: MAE={by_step[k]['mae']:.4f}")
    logger.info(f"Saved predictions and metrics to: {exp_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SingleHyper-TKAN Prediction')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    main(args)
