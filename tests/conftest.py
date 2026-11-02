import numpy as np
import pandas as pd
import pytest


def _raw_label(value):
    """LSWMD stores labels as 1x1 nested arrays, and unlabeled rows as empty arrays."""
    return np.array([[value]]) if value is not None else np.array([])


def _write_raw_lswmd(path, rows):
    """rows: (wafer_map, die_size, lot, wafer_index, split_hint, label) tuples."""
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
    df.to_pickle(path)
    return path


@pytest.fixture
def raw_lswmd_path(tmp_path):
    """A tiny pickle in the exact raw LSWMD layout, including the `trianTestLabel` typo."""
    rows = [
        (np.ones((4, 5), dtype=np.uint8), 20.0, "lot1", 1.0, "Training", "Center"),
        (np.ones((3, 3), dtype=np.uint8), 9.0, "lot1", 2.0, "Test", "none"),
        (np.zeros((6, 2), dtype=np.uint8), 12.0, "lot2", 1.0, None, None),
    ]
    return _write_raw_lswmd(tmp_path / "lswmd.pkl", rows)


@pytest.fixture
def big_raw_path(tmp_path):
    """60 lots x 6 wafers (4 labeled over 3 classes, 2 unlabeled): enough to split by lot."""
    rng = np.random.default_rng(0)
    classes = ["Center", "Donut", "none"]
    rows = [
        (
            rng.integers(0, 3, size=(12, 12), dtype=np.uint8),
            144.0,
            f"lot{lot}",
            float(w),
            "Training" if w < 4 else None,
            classes[(lot + w) % 3] if w < 4 else None,
        )
        for lot in range(60)
        for w in range(6)
    ]
    return _write_raw_lswmd(tmp_path / "big.pkl", rows)


@pytest.fixture
def tiny_data():
    """180 random 16x16 maps over 3 classes: 80 train-labeled, 20 val, 20 test, 60 unlabeled."""
    from wm811k.experiment import ExperimentData
    from wm811k.splits import Splits

    rng = np.random.default_rng(0)
    maps = rng.integers(0, 3, size=(180, 16, 16), dtype=np.uint8)
    labels = np.concatenate([np.arange(120) % 3, np.full(60, -1)]).astype(np.int64)
    # make classes learnable: class k has a bright stripe on row k
    for i in range(120):
        maps[i, labels[i], :] = 2
    splits = Splits(
        train_labeled=np.arange(80),
        train_unlabeled=np.arange(120, 180),
        val=np.arange(80, 100),
        test=np.arange(100, 120),
    )
    return ExperimentData(maps=maps, labels=labels, splits=splits)


@pytest.fixture
def tiny_cfg(tmp_path):
    from wm811k.config import load_config

    db = (tmp_path / "mlflow.db").as_posix()
    return load_config(
        [
            f"mlflow.tracking_uri=sqlite:///{db}",
            "train.steps=6",
            "train.eval_every=3",
            "train.batch_size=16",
            "gbm.n_estimators=5",
        ]
    )
