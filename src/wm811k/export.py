import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from wm811k.data import LABELS
from wm811k.experiment import ExperimentData

SAMPLE_FILE = "eval_sample.npz"

NEURAL_SERIES = ("cnn", "simclr", "fixmatch")  # the arms whose full-label runs log a model


def best_full_budget_run(results: pd.DataFrame) -> pd.Series:
    """The full-label neural run with the best validation macro-F1 (never chosen on test)."""
    candidates = results[(results["fraction"] == 1.0) & results["series"].isin(NEURAL_SERIES)]
    if candidates.empty:
        raise ValueError("no full-label neural run found; run the sweep with budget 1.0 first")
    return candidates.loc[candidates["val_macro_f1"].idxmax()]


def register_run(cfg: DictConfig, run_id: str) -> int:
    """Add the run's logged model to the MLflow registry and return the new version number."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    version = mlflow.register_model(f"runs:/{run_id}/model", cfg.mlflow.model_name)
    return int(version.version)


def export_model(cfg: DictConfig, run_id: str, out_dir: str | Path) -> Path:
    """Write the run's weights (a state_dict for `build_backbone`), label names and metrics."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location="cpu")
    size = cfg.data.map_size

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "weights.pt")
    (out / "labels.json").write_text(json.dumps(LABELS))
    run = mlflow.get_run(run_id)
    metadata = {
        "run_id": run_id,
        "map_size": size,
        "arm": run.data.tags["arm"],
        "model": run.data.tags["model"],
        **run.data.metrics,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out


def write_eval_sample(
    data: ExperimentData, out_dir: str | Path, per_class_cap: int = 100, seed: int = 0
) -> Path:
    """Save a small class-balanced sample of the test split next to the model, for the CI gate."""
    rng = np.random.default_rng(seed)
    test = data.splits.test
    chosen = np.concatenate(
        [
            rng.permutation(test[data.labels[test] == label])[:per_class_cap]
            for label in np.unique(data.labels[test])
        ]
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SAMPLE_FILE
    np.savez_compressed(path, maps=data.maps[chosen], labels=data.labels[chosen])
    return path
