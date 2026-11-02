import numpy as np
import pytest

from wm811k.preprocess import resize_map, to_channels


@pytest.mark.parametrize("shape", [(25, 27), (300, 202), (1, 1), (5, 400)])
def test_resize_output_shape_for_various_inputs(shape):
    wafer = np.random.default_rng(0).integers(0, 3, size=shape, dtype=np.uint8)

    assert resize_map(wafer, size=64).shape == (64, 64)


def test_resize_preserves_value_set():
    wafer = np.array([[0, 1], [2, 1]], dtype=np.uint8)

    assert set(np.unique(resize_map(wafer, size=8))) == {0, 1, 2}


def test_all_background_map_ok():
    resized = resize_map(np.zeros((30, 30), dtype=np.uint8), size=16)

    assert resized.shape == (16, 16)
    assert not resized.any()


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((2, 2, 2)), np.zeros((0, 4))])
def test_non_2d_or_empty_raises_value_error(bad):
    with pytest.raises(ValueError):
        resize_map(bad, size=8)


def test_to_channels_is_one_hot():
    wafer = np.random.default_rng(1).integers(0, 3, size=(8, 8), dtype=np.uint8)

    channels = to_channels(wafer)

    assert channels.shape == (3, 8, 8)
    assert channels.dtype == np.float32
    assert (channels.sum(axis=0) == 1).all()


def test_to_channels_rejects_values_outside_wafer_alphabet():
    with pytest.raises(ValueError, match="0, 1, 2"):
        to_channels(np.array([[0, 3]], dtype=np.uint8))
