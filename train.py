"""SingleHyper-TKAN 训练脚本。"""
import os
import sys
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[sys.argv.index('--gpu') + 1])

import argparse
import random
import yaml
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

from src.data import (load_pkl_data, load_position_data, create_data_loaders,
                      DataPreprocessor, apply_element_settings,
                      validate_dataset_selection, sample_stations,
                      subsample_data)
from src.graph import build_or_load_single_hypergraph, plot_geo_similarity_stats
from src.models import SingleHyperTKAN
from src.training import Trainer
from src.utils import setup_logger


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config, input_dim):
    ablation = config.get('ablation', {})
    ds_cfg = config.get('graph', {}).get('dynamic_semantic', {})
    return SingleHyperTKAN(
        input_dim=input_dim,
        output_dim=config['model']['output_projection']['output_dim'],
        d_model=config['model']['input_projection']['d_model'],
        hidden_channels=config['model']['spatial']['hidden_channels'],
        tkan_hidden=config['model']['temporal']['hidden_size'],
        sub_kan_layers=config['model']['temporal']['sub_kan_layers'],
        use_kan=config['model']['use_kan'] and not ablation.get('disable_kan', False),
        dropout=config['model']['temporal']['dropout'],
        pred_steps=config['data']['output_window'],
        grid_size=config['model']['temporal'].get('kan_grid_size', 5),
        spline_order=config['model']['temporal'].get('kan_spline_order', 3),
        bypass_spatial=ablation.get('disable_single_hypergraph', False),
        dynamic_semantic_cfg=ds_cfg,
    )


def main(args):
    config = load_config(args.config)

    if args.gpu is not None:
        config['meta']['gpu'] = args.gpu
    if args.device is not None:
        config['meta']['device'] = args.device
    if args.dataset is not None:
        for key in config['dataset_selection']:
            config['dataset_selection'][key] = False
        config['dataset_selection'][args.dataset] = True

    if not validate_dataset_selection(config):
        raise ValueError('dataset_selection必须且仅能有一个true')
    config = apply_element_settings(config)

    set_seed(
        config['meta']['seed'],
        config['reproducibility']['deterministic'],
        config['reproducibility'].get('benchmark', False),
    )

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(
        config['output']['base_dir'], f"{ts}_{config['meta']['element']}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config_snapshot.yaml'), 'w',
              encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    logger = setup_logger(
        'SingleHyperTKAN', config['output']['logging']['level'],
        output_dir=output_dir,
        console=config['output']['logging']['console'],
        file=config['output']['logging']['file'],
        append_mode=bool(args.resume))

    logger.info('=' * 60)
    logger.info('SingleHyperTKAN Training')
    logger.info('=' * 60)

    # context 通道真实顺序: 0=year 1=month 2=day 3=hour 4=region 5=altitude 6=latitude 7=longitude
    context_features = config['data'].get('context_features', {})
    context_feature_mask = [
        context_features.get('use_year', False),       # ch0
        context_features.get('use_month', True),       # ch1
        context_features.get('use_day', True),         # ch2
        context_features.get('use_hour', True),        # ch3
        context_features.get('use_region', False),     # ch4
        context_features.get('use_altitude', True),    # ch5
        context_features.get('use_latitude', True),    # ch6
        context_features.get('use_longitude', True),   # ch7
    ]

    train_data = load_pkl_data(
        config['data']['train_path'], config['data']['use_context'],
        config['data']['context_dim'], config['data']['use_dim4'],
        context_feature_mask)
    val_data = load_pkl_data(
        config['data']['val_path'], config['data']['use_context'],
        config['data']['context_dim'], config['data']['use_dim4'],
        context_feature_mask)
    test_data = load_pkl_data(
        config['data']['test_path'], config['data']['use_context'],
        config['data']['context_dim'], config['data']['use_dim4'],
        context_feature_mask)
    position = load_position_data(config['data']['position_path'])

    # --- 站点随机采样（与 hyper_kan 一致，seed 固定可复现） ---
    num_stations = config['data'].get('num_stations')
    if num_stations:
        seed = config['meta']['seed']
        total_before = position.shape[0] if position is not None else '?'
        train_data, position, sel_idx = sample_stations(
            train_data, position, num_stations, seed)
        val_data, _, _ = sample_stations(val_data, None, num_stations, seed)
        test_data, _, _ = sample_stations(test_data, None, num_stations, seed)
        logger.info(f"Station sampling: {total_before} -> {num_stations} "
                    f"(seed={seed}, idx range {sel_idx[0]}~{sel_idx[-1]})")

    # --- 样本采样 ---
    for name, d, key in [
        ('train', train_data, 'train_sample_ratio'),
        ('val', val_data, 'val_sample_ratio'),
        ('test', test_data, 'test_sample_ratio'),
    ]:
        ratio = config['data'].get(key, 1.0)
        if ratio < 1.0:
            sampled = subsample_data(d, ratio)
            d.update(sampled)
            logger.info(f"Subsampled {name} data with ratio={ratio}")

    logger.info(f"train_x shape: {train_data['x'].shape}")
    logger.info(f"val_x shape:   {val_data['x'].shape}")
    logger.info(f"test_x shape:  {test_data['x'].shape}")

    preprocessor = DataPreprocessor(
        kelvin_to_celsius=config['data']['kelvin_to_celsius'],
        normalize=config['data']['normalize'],
        scaler_type=config['data']['scaler_type'],
        context_dim=(train_data['context'].shape[-1]
                     if train_data.get('context') is not None else 0),
    )
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.transform(val_data)
    test_data = preprocessor.transform(test_data)
    preprocessor.save(os.path.join(output_dir, 'preprocessor.pkl'))

    H, W, graph_stats, edges = build_or_load_single_hypergraph(
        train_data['x'], position, config)

    # ---- 训练前：打印超图详细统计 ----
    logger.info("=" * 60)
    logger.info("Hypergraph Statistics (before training)")
    logger.info("-" * 40)
    logger.info(f"  Nodes: {graph_stats['num_nodes']}, Edges: {graph_stats['num_edges']}")
    # 融合超图
    logger.info(f"  [Fusion]  edge size  min={graph_stats['edge_size_min']}  "
                f"max={graph_stats['edge_size_max']}  "
                f"mean={graph_stats['edge_size_mean']:.2f}")
    # 地理超图
    if 'geo_edge' in graph_stats:
        g = graph_stats['geo_edge']
        logger.info(f"  [Geo]     edge size  min={g['min']}  max={g['max']}  mean={g['mean']:.2f}")
    # 语义超图
    if 'sem_edge' in graph_stats:
        s = graph_stats['sem_edge']
        logger.info(f"  [Semantic] edge size min={s['min']}  max={s['max']}  mean={s['mean']:.2f}")
    # 静态权重（优先用 graph_stats 中的值，旧缓存无此字段时从 W tensor 计算）
    w_min = graph_stats.get('static_W_min', float(W.min()))
    w_max = graph_stats.get('static_W_max', float(W.max()))
    w_mean = graph_stats.get('static_W_mean', float(W.float().mean()))
    w_std = graph_stats.get('static_W_std', float(W.float().std()))
    logger.info(f"  [Static W] min={w_min:.4f}  max={w_max:.4f}  "
                f"mean={w_mean:.4f}  std={w_std:.4f}")
    logger.info("=" * 60)

    if config['evaluation'].get('visualize', True) and position is not None:
        geo_stats = plot_geo_similarity_stats(position, output_dir)
        logger.info(f"Geo similarity stats: {json.dumps(geo_stats)}")

    train_loader, val_loader, _ = create_data_loaders(
        train_data, val_data, test_data,
        input_window=config['data']['input_window'],
        output_window=config['data']['output_window'],
        batch_size=config['data']['batch_size'],
        num_workers=config['meta']['num_workers'],
        shuffle_train=config['data']['shuffle_train'],
        stride=1,
        concat_context=config['data']['use_context'],
    )

    weather_dim = train_data['x'].shape[-1]
    ctx_dim = (train_data['context'].shape[-1]
               if (config['data']['use_context']
                   and train_data.get('context') is not None) else 0)
    input_dim = weather_dim + ctx_dim

    model = build_model(config, input_dim)
    device = config['meta']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    ds_cfg = config.get('graph', {}).get('dynamic_semantic', {})
    if ds_cfg.get('enabled', False):
        model.register_edges(edges, torch.device(device))
        logger.info("Dynamic semantic weighting enabled")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    opt_cfg = config['training']['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg['betas']))

    sch_cfg = config['training']['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sch_cfg['mode'], factor=sch_cfg['factor'],
        patience=sch_cfg['patience'], min_lr=sch_cfg['min_lr'])

    loss_fn = nn.L1Loss()

    trainer = Trainer(
        model, train_loader, val_loader, optimizer, scheduler, loss_fn,
        H, W, device, config, preprocessor=preprocessor,
        output_dir=output_dir, weather_dim=weather_dim)
    trainer.train(resume_from=args.resume, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SingleHyper-TKAN Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset_selection, e.g. Temperature')
    args = parser.parse_args()
    main(args)
