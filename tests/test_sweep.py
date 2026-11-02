import mlflow
import pytest
from omegaconf import OmegaConf

from wm811k.sweep import ARMS, run_sweep


@pytest.fixture
def sweep_cfg(tiny_cfg, tmp_path):
    return OmegaConf.merge(
        tiny_cfg,
        {
            "budgets": [1.0],
            "seeds": [0],
            "simclr": {"steps": 2, "batch_size": 16, "checkpoint": str(tmp_path / "enc.pt")},
        },
    )


def _run_count(cfg):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    return len(mlflow.search_runs([experiment.experiment_id], output_format="list"))


def test_unknown_arm_raises_value_error_listing_options(sweep_cfg, tiny_data):
    with pytest.raises(ValueError, match="gbm"):
        run_sweep(sweep_cfg, tiny_data, ["nonsense"])


def test_sweep_runs_every_arm_once_and_skips_completed_runs_on_rerun(sweep_cfg, tiny_data):
    run_sweep(sweep_cfg, tiny_data, list(ARMS))
    after_first = _run_count(sweep_cfg)

    run_sweep(sweep_cfg, tiny_data, list(ARMS))

    # 4 arms x 1 budget x 1 seed, plus the SimCLR pretraining run
    assert after_first == 5
    assert _run_count(sweep_cfg) == after_first


def test_sweep_covers_every_budget_and_seed(sweep_cfg, tiny_data):
    cfg = OmegaConf.merge(sweep_cfg, {"budgets": [0.5, 1.0], "seeds": [0, 1]})

    run_sweep(cfg, tiny_data, ["gbm"])

    assert _run_count(cfg) == 4


def test_changing_a_training_setting_reruns_instead_of_skipping(sweep_cfg, tiny_data):
    run_sweep(sweep_cfg, tiny_data, ["gbm"])
    changed = OmegaConf.merge(sweep_cfg, {"gbm": {"n_estimators": 7}})

    run_sweep(changed, tiny_data, ["gbm"])

    assert _run_count(sweep_cfg) == 2


def test_settings_an_arm_does_not_use_do_not_rerun_it(sweep_cfg, tiny_data):
    run_sweep(sweep_cfg, tiny_data, ["gbm"])
    changed = OmegaConf.merge(sweep_cfg, {"train": {"steps": 99}, "fixmatch": {"mu": 5}})

    run_sweep(changed, tiny_data, ["gbm"])

    assert _run_count(sweep_cfg) == 1


def test_after_run_fires_once_per_new_run_and_not_for_skipped_ones(sweep_cfg, tiny_data):
    calls = []
    cfg = OmegaConf.merge(sweep_cfg, {"budgets": [0.5, 1.0]})

    run_sweep(cfg, tiny_data, ["gbm"], after_run=lambda: calls.append(1))
    run_sweep(cfg, tiny_data, ["gbm"], after_run=lambda: calls.append(1))

    assert len(calls) == 2
