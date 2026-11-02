import numpy as np
import pytest

from wm811k.features import extract_features


@pytest.mark.parametrize("shape", [(5, 5), (64, 64), (30, 90)])
def test_features_fixed_length_for_any_map_size(shape):
    wafer = np.random.default_rng(0).integers(0, 3, size=shape, dtype=np.uint8)

    assert extract_features(wafer).shape == extract_features(np.ones((9, 9), dtype=np.uint8)).shape


def test_features_all_background_map_is_finite():
    assert np.isfinite(extract_features(np.zeros((20, 20), dtype=np.uint8))).all()


def test_center_defects_raise_inner_density_above_outer():
    wafer = np.ones((41, 41), dtype=np.uint8)
    wafer[18:23, 18:23] = 2

    features = extract_features(wafer)

    assert features[0] > features[9]  # first radial ring vs last
