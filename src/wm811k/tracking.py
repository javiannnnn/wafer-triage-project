import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from omegaconf import DictConfig, OmegaConf


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, prefix=f"{name}."))
        elif isinstance(value, list):
            flat[name] = str(value)
        else:
            flat[name] = value
    return flat


def _ensure_experiment(cfg: DictConfig) -> None:
    """Create the experiment with the configured artifact root the first time it is needed."""
    name = cfg.mlflow.experiment_name
    if MlflowClient(cfg.mlflow.tracking_uri).get_experiment_by_name(name) is None:
        root = Path(cfg.mlflow.artifact_root).resolve().as_uri()
        MlflowClient(cfg.mlflow.tracking_uri).create_experiment(name, artifact_location=root)
    mlflow.set_experiment(name)


@contextmanager
def start_run(cfg: DictConfig, run_name: str, tags: dict[str, str]) -> Iterator[Run]:
    """Point MLflow at the configured store/experiment and open a run with the config logged."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _ensure_experiment(cfg)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(flatten_config(OmegaConf.to_container(cfg, resolve=True)))
        yield run


RUN_KEY_TAGS = ("arm", "model", "fraction", "seed", "config")
# settings that change where things are written or how fast they run, never what is learned
FINGERPRINT_EXCLUDED = {"train": {"num_workers"}, "simclr": {"checkpoint"}}


def config_fingerprint(cfg: DictConfig, sections: tuple[str, ...]) -> str:
    """Short hash of the settings that determine a run's result: split, map size, `sections`."""
    settings = {"map_size": cfg.data.map_size, "split": OmegaConf.to_container(cfg.split)}
    for name in sections:
        section = OmegaConf.to_container(cfg[name])
        for key in FINGERPRINT_EXCLUDED.get(name, ()):
            section.pop(key, None)
        settings[name] = section
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]



def completed_run_keys(cfg: DictConfig) -> set[tuple[str, ...]]:
    """(arm, model, fraction, seed) of every finished run in the configured experiment."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        return set()
    runs = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        output_format="list",
    )
    return {
        tuple(run.data.tags[tag] for tag in RUN_KEY_TAGS)
        for run in runs
        if all(tag in run.data.tags for tag in RUN_KEY_TAGS)
    }
