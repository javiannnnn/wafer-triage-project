import numpy as np
import pandas as pd
import pytest
import torch

from wm811k.api.predictor import Predictor
from wm811k.data import LABELS
from wm811k.drift import check_drift, inflate_defects, reference_frame, simulate_drift
from wm811k.models import build_backbone
from wm811k.monitoring import COLUMNS


def _frame(seed, n=600, defect_shift=0.0, label_probs=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "die_count": rng.normal(3000, 100, n).astype(int),
            "defect_ratio": np.clip(rng.normal(0.02 + defect_shift, 0.01, n), 0, 1),
            "label": rng.choice(["none", "Center", "Donut"], n, p=label_probs),
        }
    )


def test_same_distribution_not_drifted():
    result = check_drift(_frame(0), _frame(1))

    assert not result.drifted and result.n_current == 600
    assert result.share_drifted_features == 0.0


def test_same_distribution_not_drifted_with_a_large_reference():
    # the served reference has 5,000 rows; past 1,000 Evidently switches from p-value tests to
    # distances, so the threshold must keep meaning the same thing at every reference size
    result = check_drift(_frame(0, n=5000), _frame(1, n=500))

    assert not result.drifted and result.share_drifted_features == 0.0


def test_shifted_defect_ratio_drifted_with_a_large_reference():
    result = check_drift(_frame(0, n=5000), _frame(1, n=500, defect_shift=0.1))

    assert result.drifted


def test_shifted_defect_ratio_drifted():
    result = check_drift(_frame(0), _frame(1, defect_shift=0.1))

    assert result.drifted and result.share_drifted_features > 0
    assert "<" in result.html and len(result.html) > 1000


def test_shifted_label_mix_drifted():
    result = check_drift(_frame(0), _frame(1, label_probs=(0.3, 0.4, 0.3)))

    assert result.drifted


def test_below_min_samples_not_drifted_and_no_error():
    result = check_drift(_frame(0), _frame(1, n=20, defect_shift=0.5), min_samples=200)

    assert not result.drifted and result.n_current == 20
    assert "20" in result.note and "200" in result.note
    assert result.html


def test_empty_current_window_is_not_drifted():
    result = check_drift(_frame(0), _frame(1, n=0))

    assert not result.drifted and result.n_current == 0


def test_inflate_defects_raises_defect_ratio_and_keeps_background_and_values():
    rng = np.random.default_rng(0)
    maps = rng.integers(0, 3, size=(20, 16, 16), dtype=np.uint8)

    shifted = inflate_defects(maps, rate=0.5, seed=0)

    assert ((shifted == 0) == (maps == 0)).all()
    assert set(np.unique(shifted)) <= {0, 1, 2}
    assert (shifted == 2).sum() > (maps == 2).sum()
    assert ((maps == 2) <= (shifted == 2)).all()  # existing defects are never removed
    assert (maps == 0).sum() == (shifted == 0).sum()


def test_inflate_defects_is_deterministic_and_leaves_its_input_alone():
    maps = np.ones((5, 8, 8), dtype=np.uint8)

    a = inflate_defects(maps, rate=0.3, seed=1)
    b = inflate_defects(maps, rate=0.3, seed=1)

    assert np.array_equal(a, b) and (maps == 1).all()


@pytest.fixture
def predictor():
    torch.manual_seed(0)
    return Predictor(build_backbone(num_classes=9), LABELS, map_size=16)


def test_reference_frame_has_one_row_per_map_with_the_monitoring_columns(predictor):
    maps = np.random.default_rng(0).integers(0, 3, size=(30, 16, 16), dtype=np.uint8)

    frame = reference_frame(predictor, maps)

    assert list(frame.columns) == COLUMNS and len(frame) == 30
    assert set(frame["label"]) <= set(LABELS)


def test_simulate_drift_reports_no_drift_before_and_drift_after_the_shift():
    posted = []
    window = []

    def post(wafer_map):
        posted.append(wafer_map)
        window.append(np.mean(np.array(wafer_map) == 2))

    def drifted():
        return len(window) > 0 and np.mean(window) > 0.5

    normal = np.zeros((4, 3, 3), dtype=np.uint8)
    shifted = np.full((6, 3, 3), 2, dtype=np.uint8)

    outcome = simulate_drift(post, drifted, normal, shifted)

    assert outcome == {"before_shift": False, "after_shift": True}
    assert len(posted) == 10 and posted[0] == normal[0].tolist()
