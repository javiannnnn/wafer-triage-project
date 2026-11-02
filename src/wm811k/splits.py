from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wm811k.data import labeled_mask


@dataclass(frozen=True)
class Splits:
    """Positional row indices into the DataFrame the splits were made from."""

    train_labeled: np.ndarray
    train_unlabeled: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _group_holdout(labels: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Positions of a class-stratified, group-disjoint hold-out of roughly `frac` of the rows."""
    folds = StratifiedGroupKFold(n_splits=round(1 / frac), shuffle=True, random_state=seed)
    _, held_out = next(folds.split(np.zeros(len(labels)), labels, groups))
    return held_out


def make_splits(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> Splits:
    labeled = np.flatnonzero(labeled_mask(df))
    labels = df["label"].to_numpy()[labeled]
    lots = df["lot"].to_numpy()[labeled]

    test_pos = _group_holdout(labels, lots, test_frac, seed)
    rest_pos = np.setdiff1d(np.arange(len(labeled)), test_pos)
    val_pos = rest_pos[
        _group_holdout(labels[rest_pos], lots[rest_pos], val_frac / (1 - test_frac), seed)
    ]
    train_pos = np.setdiff1d(rest_pos, val_pos)

    held_out_lots = set(lots[test_pos]) | set(lots[val_pos])
    unlabeled = np.flatnonzero(~labeled_mask(df))
    train_unlabeled = unlabeled[~df["lot"].iloc[unlabeled].isin(held_out_lots).to_numpy()]

    return Splits(
        train_labeled=labeled[train_pos],
        train_unlabeled=train_unlabeled,
        val=labeled[val_pos],
        test=labeled[test_pos],
    )


def labeled_subset(
    train_labeled: np.ndarray, labels: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Class-stratified subset of `train_labeled` (`labels[i]` is the label of `train_labeled[i]`).

    Keeps at least one sample per class. Subsets for the same seed are nested across fractions.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels):
        members = rng.permutation(train_labeled[labels == cls])
        chosen.append(members[: max(1, round(fraction * len(members)))])
    return np.sort(np.concatenate(chosen))
