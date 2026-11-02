import itertools
from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig

from wm811k.arms import fixmatch, simclr, supervised
from wm811k.experiment import ExperimentData
from wm811k.tracking import completed_run_keys, config_fingerprint


@dataclass(frozen=True)
class SweepArm:
    """A runnable arm and the (arm, model) tags its MLflow runs carry."""

    arm: str
    model: str
    title: str
    sections: tuple[str, ...]
    run: Callable[[DictConfig, ExperimentData, float, int], dict]


ARMS = {
    "gbm": SweepArm(
        supervised.ARM,
        supervised.MODEL_GBM,
        "GBM on features",
        supervised.SECTIONS_GBM,
        supervised.train_gbm,
    ),
    "cnn": SweepArm(
        supervised.ARM,
        supervised.MODEL_CNN,
        "CNN, supervised",
        supervised.SECTIONS_CNN,
        supervised.train_cnn,
    ),
    "simclr": SweepArm(
        simclr.ARM, simclr.MODEL, "SimCLR + fine-tune", simclr.SECTIONS, simclr.run_simclr_arm
    ),
    "fixmatch": SweepArm(
        fixmatch.ARM, fixmatch.MODEL, "FixMatch", fixmatch.SECTIONS, fixmatch.train_fixmatch
    ),
}

SERIES_BY_RUN = {(arm.arm, arm.model): name for name, arm in ARMS.items()}


def run_sweep(
    cfg: DictConfig,
    data: ExperimentData,
    arms: list[str],
    after_run: Callable[[], None] | None = None,
) -> None:
    """Run every arm x label budget x seed, skipping runs already finished with the same config.

    `after_run` is called after each run that actually ran, e.g. to copy progress somewhere safe.
    """
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; choose from {sorted(ARMS)}")

    done = completed_run_keys(cfg)
    for name, fraction, seed in itertools.product(arms, cfg.budgets, cfg.seeds):
        arm = ARMS[name]
        key = (arm.arm, arm.model, str(fraction), str(seed), config_fingerprint(cfg, arm.sections))
        if key not in done:
            arm.run(cfg, data, fraction, seed)
            if after_run is not None:
                after_run()
