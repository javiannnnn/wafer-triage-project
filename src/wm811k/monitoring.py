from collections import deque

import numpy as np
import pandas as pd

COLUMNS = ["die_count", "defect_ratio", "label"]


def summarize_wafer(wafer: np.ndarray) -> dict[str, float]:
    """Cheap per-wafer statistics that drift monitoring compares against the training data."""
    die_count = int((wafer > 0).sum())
    defects = int((wafer == 2).sum())
    return {"die_count": die_count, "defect_ratio": defects / die_count if die_count else 0.0}


class PredictionWindow:
    """The most recent `maxlen` prediction records (summary statistics plus predicted label)."""

    def __init__(self, maxlen: int):
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        self._records: deque[dict] = deque(maxlen=maxlen)

    def add(self, record: dict) -> None:
        self._records.append(record)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._records), columns=COLUMNS)
