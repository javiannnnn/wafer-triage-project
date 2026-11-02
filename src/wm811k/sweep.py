import itertools
from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig

from wm811k.arms import fixmatch, simclr, supervised
from wm811k.experiment import ExperimentData
from wm811k.tracking import completed_run_keys


@dataclass(frozen=True)
class SweepArm:
    """A runnable arm and the (arm, model) tags its MLflow runs carry."""

    arm: str
    model: str
    title: str
    run: Callable[[DictConfig, ExperimentData, float, int], dict]


ARMS = {
    "gbm": SweepArm(supervised.ARM, supervised.MODEL_GBM, "GBM on features", supervised.train_gbm),
    "cnn": SweepArm(supervised.ARM, supervised.MODEL_CNN, "CNN, supervised", supervised.train_cnn),
    "simclr": SweepArm(simclr.ARM, simclr.MODEL, "SimCLR + fine-tune", simclr.run_simclr_arm),
    "fixmatch": SweepArm(fixmatch.ARM, fixmatch.MODEL, "FixMatch", fixmatch.train_fixmatch),
}

SERIES_BY_RUN = {(arm.arm, arm.model): name for name, arm in ARMS.items()}


def run_sweep(cfg: DictConfig, data: ExperimentData, arms: list[str]) -> None:
    """Run every arm x label budget x seed, skipping combinations already finished in MLflow."""
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {sorted(ARMS)}")

    done = completed_run_keys(cfg)
    for name, fraction, seed in itertools.product(arms, cfg.budgets, cfg.seeds):
        arm = ARMS[name]
        if (arm.arm, arm.model, str(fraction), str(seed)) not in done:
            arm.run(cfg, data, fraction, seed)
