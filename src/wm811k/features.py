import numpy as np

N_RINGS = 10
N_SECTORS = 8


def _defect_density(bins: np.ndarray, is_defect: np.ndarray, n_bins: int) -> np.ndarray:
    """Share of defective dies per bin, given the bin and defect flag of every die."""
    dies = np.bincount(bins, minlength=n_bins)
    defects = np.bincount(bins[is_defect], minlength=n_bins)
    return defects / np.maximum(dies, 1)


def extract_features(wafer: np.ndarray) -> np.ndarray:
    """Radial and angular defect density plus overall defect ratio, for any map size.

    Returns N_RINGS + N_SECTORS + 1 values.
    """
    h, w = wafer.shape
    dy = (np.arange(h) - (h - 1) / 2)[:, None] / (h / 2)
    dx = (np.arange(w) - (w - 1) / 2)[None, :] / (w / 2)
    ring = np.minimum((np.hypot(dy, dx) * N_RINGS).astype(int), N_RINGS - 1)
    angle = np.arctan2(dy, dx)  # (-pi, pi]
    sector = np.minimum(((angle + np.pi) / (2 * np.pi) * N_SECTORS).astype(int), N_SECTORS - 1)

    on_wafer = wafer > 0
    is_defect = (wafer == 2)[on_wafer]
    return np.concatenate(
        [
            _defect_density(ring[on_wafer], is_defect, N_RINGS),
            _defect_density(sector[on_wafer], is_defect, N_SECTORS),
            [is_defect.sum() / max(is_defect.size, 1)],
        ]
    )
