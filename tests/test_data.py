import numpy as np
import pytest

from wm811k.data import LABELS, DataLoadError, labeled_mask, load_lswmd


def test_unlabeled_rows_have_none_label(raw_lswmd_path):
    df = load_lswmd(raw_lswmd_path)

    assert list(df["label"]) == ["Center", "none", None]
    assert list(labeled_mask(df)) == [True, True, False]


def test_trianTestLabel_typo_normalized(raw_lswmd_path):
    df = load_lswmd(raw_lswmd_path)

    assert list(df["split_hint"]) == ["Training", "Test", None]


def test_wafer_map_is_2d_uint8(raw_lswmd_path):
    df = load_lswmd(raw_lswmd_path)

    assert all(m.ndim == 2 and m.dtype == np.uint8 for m in df["wafer_map"])


def test_output_columns_and_types(raw_lswmd_path):
    df = load_lswmd(raw_lswmd_path)

    assert list(df.columns) == ["wafer_map", "lot", "die_size", "label", "split_hint"]
    assert df["die_size"].dtype == int
    assert set(df["label"].dropna()) <= set(LABELS)


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(DataLoadError, match="does-not-exist.pkl"):
        load_lswmd(tmp_path / "does-not-exist.pkl")


def test_corrupt_pickle_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")

    with pytest.raises(DataLoadError, match="bad.pkl"):
        load_lswmd(bad)
