from wm811k.config import load_config


def test_load_config_defaults():
    cfg = load_config()

    assert cfg.data.map_size == 64
    assert list(cfg.budgets) == [0.01, 0.05, 0.10, 0.25, 1.0]
    assert list(cfg.seeds) == [0, 1, 2]


def test_override_applies():
    cfg = load_config(["data.map_size=32"])

    assert cfg.data.map_size == 32
