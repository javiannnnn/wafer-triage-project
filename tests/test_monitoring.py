import numpy as np
import pytest

from wm811k.monitoring import PredictionWindow, summarize_wafer


def test_summarize_wafer_counts_dies_and_defect_ratio():
    wafer = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)

    summary = summarize_wafer(wafer)

    assert summary == {"die_count": 4, "defect_ratio": 0.5}


def test_summarize_wafer_all_background_has_zero_ratio_not_nan():
    assert summarize_wafer(np.zeros((3, 3), dtype=np.uint8)) == {
        "die_count": 0,
        "defect_ratio": 0.0,
    }


def test_window_keeps_only_the_most_recent_records():
    window = PredictionWindow(maxlen=3)

    for i in range(5):
        window.add({"die_count": i, "defect_ratio": 0.0, "label": "none"})

    assert list(window.frame()["die_count"]) == [2, 3, 4]


def test_empty_window_gives_an_empty_frame_with_the_expected_columns():
    frame = PredictionWindow(maxlen=3).frame()

    assert frame.empty
    assert list(frame.columns) == ["die_count", "defect_ratio", "label"]


def test_window_rejects_non_positive_size():
    with pytest.raises(ValueError):
        PredictionWindow(maxlen=0)
