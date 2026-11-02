from pathlib import Path

from omegaconf import DictConfig

from wm811k.data import load_lswmd
from wm811k.experiment import (
    ExperimentData,
    build_experiment_data,
    load_experiment_data,
    save_experiment_data,
)
from wm811k.splits import make_splits


def _cache_meta(cfg: DictConfig) -> dict:
    """The config values that shape the cached arrays; a mismatch means the cache is stale."""
    return {
        "map_size": cfg.data.map_size,
        "seed": cfg.split.seed,
        "val_frac": cfg.split.val_frac,
        "test_frac": cfg.split.test_frac,
    }


def prepare_experiment_data(cfg: DictConfig) -> ExperimentData:
    """Load the cached arrays if they match the config, else rebuild them from the raw pickle."""
    cache, meta = Path(cfg.data.cache), _cache_meta(cfg)
    if cache.is_file():
        data, cached_meta = load_experiment_data(cache)
        if cached_meta == meta:
            return data

    df = load_lswmd(cfg.data.path)
    splits = make_splits(df, cfg.split.val_frac, cfg.split.test_frac, cfg.split.seed)
    data = build_experiment_data(df, splits, cfg.data.map_size)
    save_experiment_data(data, cache, meta)
    return data
