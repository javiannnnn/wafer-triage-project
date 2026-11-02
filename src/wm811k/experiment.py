import json
from dataclasses import dataclass
from pathlib import Path

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


def save_experiment_data(data: ExperimentData, path: str | Path, meta: dict) -> None:
    """Write a compressed cache; `meta` records the config that shaped it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        maps=data.maps,
        labels=data.labels,
        meta=json.dumps(meta),
        **{name: getattr(data.splits, name) for name in Splits.__dataclass_fields__},
    )


def load_experiment_data(path: str | Path) -> tuple[ExperimentData, dict]:
    with np.load(path) as cache:
        splits = Splits(**{name: cache[name] for name in Splits.__dataclass_fields__})
        data = ExperimentData(maps=cache["maps"], labels=cache["labels"], splits=splits)
        return data, json.loads(str(cache["meta"]))
