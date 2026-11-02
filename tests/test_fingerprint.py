from omegaconf import OmegaConf

from wm811k.config import load_config
from wm811k.tracking import config_fingerprint


def _with(cfg, **sections):
    return OmegaConf.merge(cfg, sections)


def test_fingerprint_is_stable_for_the_same_config():
    assert config_fingerprint(load_config(), ("train",)) == config_fingerprint(
        load_config(), ("train",)
    )


def test_fingerprint_changes_when_a_training_setting_of_its_sections_changes():
    base = load_config()

    changed = _with(base, train={"steps": 2000})

    assert config_fingerprint(base, ("train",)) != config_fingerprint(changed, ("train",))


def test_fingerprint_ignores_settings_that_do_not_affect_training():
    base = load_config()

    other = _with(
        base,
        train={"num_workers": 4},
        simclr={"checkpoint": "elsewhere.pt"},
        mlflow={"tracking_uri": "sqlite:///other.db"},
        budgets=[1.0],
        seeds=[9],
        data={"cache": "other.npz", "path": "other.pkl"},
    )

    assert config_fingerprint(base, ("train", "simclr")) == config_fingerprint(
        other, ("train", "simclr")
    )


def test_fingerprint_ignores_sections_the_arm_does_not_use():
    base = load_config()

    changed = _with(base, gbm={"n_estimators": 7}, fixmatch={"mu": 9})

    assert config_fingerprint(base, ("train",)) == config_fingerprint(changed, ("train",))
    assert config_fingerprint(base, ("gbm",)) != config_fingerprint(changed, ("gbm",))


def test_fingerprint_changes_with_the_data_split_and_map_size():
    base = load_config()

    assert config_fingerprint(base, ("train",)) != config_fingerprint(
        _with(base, split={"seed": 1}), ("train",)
    )
    assert config_fingerprint(base, ("train",)) != config_fingerprint(
        _with(base, data={"map_size": 32}), ("train",)
    )
