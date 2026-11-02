from collections.abc import Callable

import mlflow
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig

from wm811k.experiment import ExperimentData
from wm811k.splits import labeled_subset
from wm811k.tracking import config_fingerprint, start_run
from wm811k.training import seed_everything


def run_logged(
    cfg: DictConfig,
    data: ExperimentData,
    *,
    arm: str,
    model: str,
    fraction: float,
    seed: int,
    sections: tuple[str, ...],
    fit: Callable[[np.ndarray], dict[str, float]],
) -> dict[str, float]:
    """Pick the label budget, seed, and run `fit(train_idx)` inside a tagged MLflow run.

    `sections` names the config sections the arm depends on; they go into the run's `config` tag.
    """
    seed_everything(seed)
    train_labeled = data.splits.train_labeled
    train_idx = labeled_subset(train_labeled, data.labels[train_labeled], fraction, seed)
    tags = {
        "arm": arm,
        "model": model,
        "fraction": str(fraction),
        "seed": str(seed),
        "config": config_fingerprint(cfg, sections),
    }
    with start_run(cfg, run_name=f"{arm}-{model}-{fraction}-{seed}", tags=tags):
        mlflow.log_param("n_labeled", len(train_idx))
        metrics = fit(train_idx)
        mlflow.log_metrics(metrics)
    return metrics


def log_model_if_full_budget(model: nn.Module, fraction: float) -> None:
    """Attach the trained model to the active run when it used every label.

    Only full-label runs are candidates for serving, so partial-budget runs stay lightweight.
    """
    if fraction == 1.0:
        mlflow.pytorch.log_model(model.cpu(), name="model", serialization_format="pickle")
