import numpy as np
import pandas as pd
import pytest

from wm811k.splits import labeled_subset, make_splits

CLASSES = ["Center", "Donut", "none"]


@pytest.fixture
def df():
    """60 lots x 6 wafers: 4 labeled (cycling through CLASSES) and 2 unlabeled each."""
    rows = []
    for lot in range(60):
        for w in range(6):
            label = CLASSES[(lot + w) % 3] if w < 4 else None
            rows.append({"lot": f"lot{lot}", "label": label})
    return pd.DataFrame(rows)


def _lots(df, idx):
    return set(df["lot"].iloc[idx])


def test_no_lot_in_two_splits(df):
    s = make_splits(df, val_frac=0.1, test_frac=0.2, seed=0)

    train = _lots(df, np.concatenate([s.train_labeled, s.train_unlabeled]))
    assert not train & _lots(df, s.val)
    assert not train & _lots(df, s.test)
    assert not _lots(df, s.val) & _lots(df, s.test)


def test_unlabeled_only_in_train_unlabeled(df):
    s = make_splits(df, val_frac=0.1, test_frac=0.2, seed=0)

    assert df["label"].iloc[s.train_unlabeled].isna().all()
    for held_out in (s.train_labeled, s.val, s.test):
        assert df["label"].iloc[held_out].notna().all()


def test_unlabeled_from_held_out_lots_are_dropped(df):
    s = make_splits(df, val_frac=0.1, test_frac=0.2, seed=0)

    held_out_lots = _lots(df, s.val) | _lots(df, s.test)
    assert not _lots(df, s.train_unlabeled) & held_out_lots


def test_split_is_deterministic_for_seed(df):
    a = make_splits(df, val_frac=0.1, test_frac=0.2, seed=3)
    b = make_splits(df, val_frac=0.1, test_frac=0.2, seed=3)

    assert np.array_equal(a.test, b.test) and np.array_equal(a.train_labeled, b.train_labeled)


def test_split_sizes_follow_fractions(df):
    s = make_splits(df, val_frac=0.1, test_frac=0.2, seed=0)
    n_labeled = df["label"].notna().sum()

    assert abs(len(s.test) / n_labeled - 0.2) < 0.05
    assert abs(len(s.val) / n_labeled - 0.1) < 0.05


def _train_labels(df, seed=0):
    s = make_splits(df, val_frac=0.1, test_frac=0.2, seed=seed)
    return s.train_labeled, df["label"].to_numpy()[s.train_labeled]


def test_subset_keeps_every_class_at_least_once(df):
    idx, labels = _train_labels(df)

    subset = labeled_subset(idx, labels, fraction=0.01, seed=0)

    assert set(df["label"].iloc[subset]) == set(CLASSES)


def test_subsets_are_nested(df):
    idx, labels = _train_labels(df)

    subsets = [set(labeled_subset(idx, labels, f, seed=0)) for f in (0.01, 0.05, 0.10, 0.25)]

    for smaller, larger in zip(subsets, subsets[1:], strict=False):
        assert smaller <= larger


def test_fraction_one_returns_all(df):
    idx, labels = _train_labels(df)

    assert np.array_equal(np.sort(labeled_subset(idx, labels, 1.0, seed=0)), np.sort(idx))


def test_subset_differs_across_seeds(df):
    idx, labels = _train_labels(df)

    a = labeled_subset(idx, labels, 0.25, seed=0)
    b = labeled_subset(idx, labels, 0.25, seed=1)

    assert not np.array_equal(a, b)
