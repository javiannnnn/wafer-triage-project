from dataclasses import dataclass

import numpy as np
import pandas as pd

from wm811k.data import encode_labels
from wm811k.preprocess import resize_all
from wm811k.splits import Splits


@dataclass(frozen=True)
class ExperimentData:
    """Everything the training arms need, indexed positionally.

    maps: (N, S, S) uint8 resized wafer maps. labels: (N,) class index, -1 when unlabeled.
    """

    maps: np.ndarray
    labels: np.ndarray
    splits: Splits


def build_experiment_data(df: pd.DataFrame, splits: Splits, map_size: int) -> ExperimentData:
    return ExperimentData(
        maps=resize_all(df["wafer_map"], map_size),
        labels=encode_labels(df["label"]),
        splits=splits,
    )
