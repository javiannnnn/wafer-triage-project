from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


class DataLoadError(Exception):
    pass


def _unwrap(cell: np.ndarray) -> str | None:
    """Raw label cells are 1x1 nested arrays; unlabeled rows hold an empty array."""
    return str(cell[0][0]) if cell.size else None


def load_lswmd(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"LSWMD file not found: {path}")
    try:
        raw = pd.read_pickle(path)
    except Exception as exc:
        raise DataLoadError(
            f"Could not unpickle {path} with pandas {pd.__version__}: {exc}"
        ) from exc

    return pd.DataFrame(
        {
            "wafer_map": raw["waferMap"].map(lambda m: np.asarray(m, dtype=np.uint8)),
            "lot": raw["lotName"].astype(str),
            "die_size": raw["dieSize"].astype(int),
            "label": raw["failureType"].map(_unwrap),
            "split_hint": raw["trianTestLabel"].map(_unwrap),
        }
    )


def labeled_mask(df: pd.DataFrame) -> pd.Series:
    return df["label"].notna()


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Class indices into LABELS; unlabeled rows become -1."""
    index = {label: i for i, label in enumerate(LABELS)}
    return labels.map(index).fillna(-1).astype(np.int64).to_numpy()
