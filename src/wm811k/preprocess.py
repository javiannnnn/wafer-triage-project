from collections.abc import Iterable

import numpy as np

WAFER_VALUES = (0, 1, 2)  # background, good die, defective die


def _require_2d(wafer: np.ndarray) -> None:
    if wafer.ndim != 2 or wafer.size == 0:
        raise ValueError(f"wafer map must be a non-empty 2D array, got shape {wafer.shape}")


def resize_map(wafer: np.ndarray, size: int = 64) -> np.ndarray:
    """Nearest-neighbour resize, so values stay within the wafer alphabet."""
    _require_2d(wafer)
    rows = (np.arange(size) * wafer.shape[0] // size).astype(int)
    cols = (np.arange(size) * wafer.shape[1] // size).astype(int)
    return wafer[np.ix_(rows, cols)]


def to_channels(wafer: np.ndarray) -> np.ndarray:
    """One-hot encode a wafer map into float32 channels of shape (3, H, W)."""
    _require_2d(wafer)
    if not np.isin(wafer, WAFER_VALUES).all():
        raise ValueError("wafer map values must be in {0, 1, 2}")
    return np.stack([wafer == v for v in WAFER_VALUES]).astype(np.float32)


def resize_all(wafers: Iterable[np.ndarray], size: int = 64) -> np.ndarray:
    """Resize many wafer maps into one uint8 array of shape (N, size, size)."""
    return np.stack([resize_map(w, size) for w in wafers])
