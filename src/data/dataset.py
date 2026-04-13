"""
PyTorch Dataset for Spatio-Temporal Data
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SpatioTemporalDataset(Dataset):
    def __init__(self,
                 x: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 context: Optional[np.ndarray] = None,
                 input_window: int = 12,
                 output_window: int = 12,
                 stride: int = 1,
                 concat_context: bool = True):
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.concat_context = concat_context

        if x.ndim == 4:
            self.is_prebuilt = True
            if context is not None and concat_context:
                x = np.concatenate([x, context], axis=-1)
            self.samples_x = torch.from_numpy(x).float()
            if y is not None:
                self.samples_y = torch.from_numpy(y).float()
            else:
                self.samples_y = self.samples_x[:, -output_window:, :, :]
            self.num_samples = self.samples_x.shape[0]
        else:
            self.is_prebuilt = False
            if context is not None and concat_context:
                self.x = np.concatenate([x, context], axis=-1)
            else:
                self.x = x
            self.y = y if y is not None else x
            total_window = input_window + output_window
            T = self.x.shape[0]
            self.num_samples = max(0, (T - total_window) // stride + 1)
            if self.num_samples == 0:
                raise ValueError(f"Insufficient time steps: T={T}, required={total_window}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_prebuilt:
            return {'x': self.samples_x[idx], 'y': self.samples_y[idx]}

        start_idx = idx * self.stride
        end_input = start_idx + self.input_window
        end_output = end_input + self.output_window
        x_window = self.x[start_idx:end_input]
        y_window = self.y[end_input:end_output]
        return {'x': torch.from_numpy(x_window).float(), 'y': torch.from_numpy(y_window).float()}


def create_data_loaders(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    input_window: int,
    output_window: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
    stride: int = 1,
    concat_context: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SpatioTemporalDataset(
        x=train_data['x'], y=train_data.get('y', None), context=train_data.get('context', None),
        input_window=input_window, output_window=output_window, stride=stride, concat_context=concat_context
    )
    val_dataset = SpatioTemporalDataset(
        x=val_data['x'], y=val_data.get('y', None), context=val_data.get('context', None),
        input_window=input_window, output_window=output_window, stride=stride, concat_context=concat_context
    )
    test_dataset = SpatioTemporalDataset(
        x=test_data['x'], y=test_data.get('y', None), context=test_data.get('context', None),
        input_window=input_window, output_window=output_window, stride=stride, concat_context=concat_context
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    logger.info(f"Train/Val/Test samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
    return train_loader, val_loader, test_loader
