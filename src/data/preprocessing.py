"""数据预处理模块。"""
import numpy as np
import pickle
import logging
from typing import Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, kelvin_to_celsius: bool = False, normalize: bool = True, scaler_type: str = 'standard', context_dim: int = 0):
        self.kelvin_to_celsius = kelvin_to_celsius
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.context_dim = context_dim
        self.fitted = False

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
            self.context_scaler = StandardScaler() if context_dim > 0 else None
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            self.context_scaler = MinMaxScaler() if context_dim > 0 else None
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

    def _k2c(self, data: np.ndarray) -> np.ndarray:
        return data - 273.15

    def fit(self, train_data: Dict[str, np.ndarray]):
        x = train_data['x']
        context = train_data.get('context', None)
        if self.kelvin_to_celsius:
            x = self._k2c(x)
        if self.normalize:
            x2d = x.reshape(-1, x.shape[-1])
            self.scaler.fit(x2d)
            if context is not None and self.context_scaler is not None:
                c2d = context.reshape(-1, context.shape[-1])
                self.context_scaler.fit(c2d)
        self.fitted = True
        return self

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
        out = {}
        for key in ['x', 'y']:
            if key in data and data[key] is not None:
                v = data[key].copy()
                if self.kelvin_to_celsius:
                    v = self._k2c(v)
                if self.normalize:
                    sh = v.shape
                    v = self.scaler.transform(v.reshape(-1, sh[-1])).reshape(sh)
                out[key] = v
        if 'context' in data and data['context'] is not None:
            c = data['context'].copy()
            if self.normalize and self.context_scaler is not None:
                sh = c.shape
                c = self.context_scaler.transform(c.reshape(-1, sh[-1])).reshape(sh)
            out['context'] = c
        if 'position' in data:
            out['position'] = data['position']
        return out

    def fit_transform(self, train_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.fit(train_data)
        return self.transform(train_data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
        out = data.copy()
        if self.normalize:
            sh = out.shape
            out = self.scaler.inverse_transform(out.reshape(-1, sh[-1])).reshape(sh)
        if self.kelvin_to_celsius:
            out = out + 273.15
        return out

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(path: str) -> 'DataPreprocessor':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = DataPreprocessor()
        obj.__dict__.update(state)
        return obj
