import numpy as np
import pytest

import wm811k.pipeline as pipeline
from wm811k.config import load_config
from wm811k.experiment import load_experiment_data, save_experiment_data
from wm811k.pipeline import prepare_experiment_data


@pytest.fixture
def cfg(big_raw_path, tmp_path):
    return load_config(
        [
            f"data.path={big_raw_path.as_posix()}",
            f"data.cache={(tmp_path / 'cache.npz').as_posix()}",
            "data.map_size=16",
        ]
    )


def test_save_then_load_roundtrips_experiment_data(tiny_data, tmp_path):
    path = tmp_path / "d.npz"

    save_experiment_data(tiny_data, path, meta={"map_size": 16})
    loaded, meta = load_experiment_data(path)

    assert np.array_equal(loaded.maps, tiny_data.maps)
    assert np.array_equal(loaded.labels, tiny_data.labels)
    assert np.array_equal(loaded.splits.test, tiny_data.splits.test)
    assert np.array_equal(loaded.splits.train_unlabeled, tiny_data.splits.train_unlabeled)
    assert meta == {"map_size": 16}


def test_prepare_builds_cache_then_reuses_it_without_the_pickle(cfg, monkeypatch):
    first = prepare_experiment_data(cfg)

    def must_not_load(*_):
        raise AssertionError("the raw pickle should not be read when the cache is valid")

    monkeypatch.setattr(pipeline, "load_lswmd", must_not_load)
    second = prepare_experiment_data(cfg)

    assert np.array_equal(first.maps, second.maps)
    assert first.maps.shape[1:] == (16, 16)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("data", "map_size", 8), ("split", "seed", 1), ("split", "test_frac", 0.25)],
)
def test_cache_is_rebuilt_when_the_config_that_shaped_it_changes(cfg, section, key, value):
    from omegaconf import OmegaConf

    prepare_experiment_data(cfg)
    changed = OmegaConf.merge(cfg, {section: {key: value}})

    prepare_experiment_data(changed)

    _, meta = load_experiment_data(cfg.data.cache)
    assert meta[key] == value
