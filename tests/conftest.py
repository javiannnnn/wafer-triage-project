import numpy as np
import pandas as pd
import pytest


def _raw_label(value):
    """LSWMD stores labels as 1x1 nested arrays, and unlabeled rows as empty arrays."""
    return np.array([[value]]) if value is not None else np.array([])


@pytest.fixture
def raw_lswmd_path(tmp_path):
    """A tiny pickle in the exact raw LSWMD layout, including the `trianTestLabel` typo."""
    rows = [
        (np.ones((4, 5), dtype=np.uint8), 20.0, "lot1", 1.0, "Training", "Center"),
        (np.ones((3, 3), dtype=np.uint8), 9.0, "lot1", 2.0, "Test", "none"),
        (np.zeros((6, 2), dtype=np.uint8), 12.0, "lot2", 1.0, None, None),
    ]
    df = pd.DataFrame(
        {
            "waferMap": [r[0] for r in rows],
            "dieSize": [r[1] for r in rows],
            "lotName": [r[2] for r in rows],
            "waferIndex": [r[3] for r in rows],
            "trianTestLabel": [_raw_label(r[4]) for r in rows],
            "failureType": [_raw_label(r[5]) for r in rows],
        }
    )
    path = tmp_path / "lswmd.pkl"
    df.to_pickle(path)
    return path
