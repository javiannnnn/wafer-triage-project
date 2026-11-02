from collections.abc import Callable

import mlflow
import numpy as np
from omegaconf import DictConfig

from wm811k.experiment import ExperimentData
from wm811k.splits import labeled_subset
from wm811k.tracking import start_run
from wm811k.training import seed_everything


def run_logged(
    cfg: DictConfig,
    data: ExperimentData,
    *,
    arm: str,
    model: str,
    fraction: float,
    seed: int,
    fit: Callable[[np.ndarray], dict[str, float]],
) -> dict[str, float]:
    """Pick the label budget, seed, and run `fit(train_idx)` inside a tagged MLflow run."""
    seed_everything(seed)
    train_labeled = data.splits.train_labeled
    train_idx = labeled_subset(train_labeled, data.labels[train_labeled], fraction, seed)
    tags = {"arm": arm, "model": model, "fraction": str(fraction), "seed": str(seed)}
    with start_run(cfg, run_name=f"{arm}-{model}-{fraction}-{seed}", tags=tags):
        mlflow.log_param("n_labeled", len(train_idx))
        metrics = fit(train_idx)
        mlflow.log_metrics(metrics)
    return metrics
