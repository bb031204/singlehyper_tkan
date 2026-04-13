"""
数据加载器 - 对齐 hyper_kan 风格
"""
import pickle
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_pkl_data(file_path: str,
                  use_context: bool = True,
                  context_dim: int = 8,
                  use_dim4: bool = True,
                  context_feature_mask: Optional[List[bool]] = None) -> Dict[str, np.ndarray]:
    """加载PKL格式数据。"""
    logger.info(f"Loading data from {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        x = data.get('x', None)
        y = data.get('y', None)
        context = data.get('context', None)
        position = data.get('position', None)
    elif isinstance(data, (list, tuple)):
        if len(data) == 3:
            x, y, context = data
            position = None
        elif len(data) == 2:
            x, y = data
            context = None
            position = None
        else:
            raise ValueError(f"Unexpected tuple length: {len(data)}")
    elif isinstance(data, np.ndarray):
        x = data
        y = None
        context = None
        position = None
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if x is None:
        raise ValueError("Data must contain 'x' field")

    x = np.array(x)
    if y is not None:
        y = np.array(y)
    if context is not None:
        context = np.array(context)
    if position is not None:
        position = np.array(position)

    if use_context and context is not None:
        if context.ndim == 2:
            T = x.shape[0]
            context = np.tile(context[np.newaxis, :, :], (T, 1, 1))

        if context_feature_mask is not None and len(context_feature_mask) > 0:
            mask_len = min(len(context_feature_mask), context.shape[-1])
            selected_indices = [i for i in range(mask_len) if context_feature_mask[i]]
            if len(selected_indices) > 0:
                context = context[..., selected_indices]
                logger.info(f"Context features selected: {len(selected_indices)}/{mask_len}")
            else:
                logger.warning("No context features selected by mask, disable context")
                context = None
        elif context.shape[-1] > context_dim:
            context = context[..., :context_dim]

    if not use_dim4 and x.ndim >= 3 and x.shape[-1] > 3:
        x = x[..., :3]
        if y is not None and y.ndim >= 3 and y.shape[-1] > 3:
            y = y[..., :3]

    logger.info(f"Data loaded - x: {x.shape}")
    if y is not None:
        logger.info(f"              y: {y.shape}")
    if context is not None:
        logger.info(f"              context: {context.shape}")
    if position is not None:
        logger.info(f"              position: {position.shape}")

    return {
        'x': x,
        'y': y,
        'context': context if use_context else None,
        'position': position,
    }


def load_position_data(file_path: str) -> np.ndarray:
    """加载站点位置数据。"""
    try:
        logger.info(f"Loading position data from {file_path}")
        with open(file_path, 'rb') as f:
            position = pickle.load(f)

        if isinstance(position, dict):
            if 'lonlat' in position:
                position = position['lonlat']
            elif 'position' in position:
                position = position['position']
            else:
                logger.error(f"Unexpected keys in position dict: {position.keys()}")
                return None

        position = np.array(position)
        if position.ndim == 1 and len(position) % 2 == 0:
            position = position.reshape(-1, 2)

        if position.ndim != 2:
            return None
        if position.shape[1] > 2:
            position = position[:, :2]
        if position.shape[1] != 2:
            return None

        logger.info(f"Position loaded: {position.shape}")
        return position
    except Exception as e:
        logger.error(f"Error loading position data: {e}")
        return None


def save_pkl_data(data: Any, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
