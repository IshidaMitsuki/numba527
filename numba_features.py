"""numba_features.py – 旧 features() を NumPy/Numba 版で再実装"""
from numba import njit
import numpy as np

from numba_core import (
    find_drop_y, place_piece, clear_lines, aggregate_features,
    MATRIX_W, MATRIX_H,
)

@njit(cache=True)
def evaluate(cols: np.ndarray, shape: np.ndarray, x_off: int):
    """旧 core.features() + evaluate() を 1 関数に統合。"""
    y = find_drop_y(cols, shape, x_off)
    new_cols = cols.copy()
    place_piece(new_cols, shape, x_off, y)
    lines = clear_lines(new_cols)
    feat = aggregate_features(new_cols, y, lines, len(shape))
    return feat