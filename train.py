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

from src.data import load_pkl_data, load_position_data, create_data_loaders, DataPreprocessor
from src.data import apply_element_settings, validate_dataset_selection
from src.graph import build_or_load_single_hypergraph
from src.models import SingleHyperTKAN
from src.training import Trainer
from src.utils import setup_logger


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config, input_dim):
    return SingleHyperTKAN(
        input_dim=input_dim,
        output_dim=config['model']['output_projection']['output_dim'],
        d_model=config['model']['input_projection']['d_model'],
        hidden_channels=config['model']['spatial']['hidden_channels'],
        tkan_hidden=config['model']['temporal']['hidden_size'],
        sub_kan_layers=config['model']['temporal']['sub_kan_layers'],
        use_kan=config['model']['use_kan'],
        dropout=config['model']['temporal']['dropout'],
        pred_steps=config['data']['output_window'],
    )


def main(args):
    config = load_config(args.config)
    if args.gpu is not None:
        config['meta']['gpu'] = args.gpu
    if not validate_dataset_selection(config):
        raise ValueError('dataset_selection必须且仅能有一个true')
    config = apply_element_settings(config)

    set_seed(config['meta']['seed'], config['reproducibility']['deterministic'])

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['output']['base_dir'], f"{ts}_{config['meta']['element']}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config_snapshot.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    logger = setup_logger('SingleHyperTKAN', config['output']['logging']['level'], output_dir=output_dir,
                          console=config['output']['logging']['console'], file=config['output']['logging']['file'],
                          append_mode=bool(args.resume))

    logger.info('=' * 60)
    logger.info('SingleHyperTKAN Training')
    logger.info('=' * 60)

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

    train_data = load_pkl_data(config['data']['train_path'], config['data']['use_context'], config['data']['context_dim'], config['data']['use_dim4'], context_feature_mask)
    val_data = load_pkl_data(config['data']['val_path'], config['data']['use_context'], config['data']['context_dim'], config['data']['use_dim4'], context_feature_mask)
    test_data = load_pkl_data(config['data']['test_path'], config['data']['use_context'], config['data']['context_dim'], config['data']['use_dim4'], context_feature_mask)
    position = load_position_data(config['data']['position_path'])

    preprocessor = DataPreprocessor(
        kelvin_to_celsius=config['data']['kelvin_to_celsius'],
        normalize=config['data']['normalize'],
        scaler_type=config['data']['scaler_type'],
        context_dim=train_data['context'].shape[-1] if train_data.get('context') is not None else 0,
    )
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.transform(val_data)
    test_data = preprocessor.transform(test_data)
    preprocessor.save(os.path.join(output_dir, 'preprocessor.pkl'))

    H, W, graph_stats = build_or_load_single_hypergraph(train_data['x'], position, config)
    logger.info(f"Graph stats: {json.dumps(graph_stats, ensure_ascii=False)}")

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
    ctx_dim = train_data['context'].shape[-1] if (config['data']['use_context'] and train_data.get('context') is not None) else 0
    input_dim = weather_dim + ctx_dim

    model = build_model(config, input_dim)
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    opt_cfg = config['training']['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'], betas=tuple(opt_cfg['betas']))

    sch_cfg = config['training']['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sch_cfg['mode'], factor=sch_cfg['factor'], patience=sch_cfg['patience'], min_lr=sch_cfg['min_lr'])

    loss_fn = nn.L1Loss()
    device = config['meta']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, loss_fn, H, W, device, config, preprocessor=preprocessor, output_dir=output_dir)
    trainer.train(resume_from=args.resume, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SingleHyper-TKAN Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()
    main(args)
