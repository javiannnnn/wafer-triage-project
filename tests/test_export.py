import json

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch

from wm811k.arms.supervised import train_cnn
from wm811k.export import (
    best_full_budget_run,
    export_model,
    register_run,
    write_eval_sample,
)
from wm811k.models import build_backbone
from wm811k.results import collect_results


def _results(rows):
    return pd.DataFrame(rows, columns=["series", "fraction", "val_macro_f1", "run_id"])


def test_best_run_is_picked_by_validation_not_test_among_full_budget_neural_runs():
    results = _results(
        [
            ("gbm", 1.0, 0.99, "gbm-run"),  # no model artifact: never a candidate
            ("cnn", 0.5, 0.95, "partial-run"),  # not a full-label run
            ("cnn", 1.0, 0.60, "cnn-run"),
            ("simclr", 1.0, 0.70, "simclr-run"),
        ]
    )

    assert best_full_budget_run(results)["run_id"] == "simclr-run"


def test_best_run_raises_when_no_full_budget_neural_run_exists():
    with pytest.raises(ValueError, match="full-label"):
        best_full_budget_run(_results([("gbm", 1.0, 0.9, "g"), ("cnn", 0.1, 0.5, "c")]))


@pytest.fixture
def trained_run(tiny_data, tiny_cfg):
    train_cnn(tiny_cfg, tiny_data, fraction=1.0, seed=0)
    return best_full_budget_run(collect_results(tiny_cfg))


def test_export_writes_loadable_weights_with_labels_and_metadata(
    trained_run, tiny_cfg, tmp_path
):
    out = export_model(tiny_cfg, trained_run["run_id"], tmp_path / "exported")

    model = build_backbone(num_classes=9)
    model.load_state_dict(torch.load(out / "weights.pt"))
    assert model.eval()(torch.zeros(1, 3, 16, 16)).shape == (1, 9)
    assert json.loads((out / "labels.json").read_text())[0] == "Center"
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["run_id"] == trained_run["run_id"]
    assert metadata["map_size"] == tiny_cfg.data.map_size
    assert metadata["test_macro_f1"] == pytest.approx(trained_run["test_macro_f1"])


def test_register_run_adds_a_model_version_to_the_registry(trained_run, tiny_cfg):
    version = register_run(tiny_cfg, trained_run["run_id"])

    assert version == 1
    client = mlflow.MlflowClient(tiny_cfg.mlflow.tracking_uri)
    assert client.get_registered_model(tiny_cfg.mlflow.model_name).name == "wafer-classifier"


def test_eval_sample_covers_every_test_class_with_a_per_class_cap(tiny_data, tmp_path):
    path = write_eval_sample(tiny_data, tmp_path, per_class_cap=3, seed=0)

    with np.load(path) as sample:
        labels, maps = sample["labels"], sample["maps"]
    test_labels = tiny_data.labels[tiny_data.splits.test]
    assert set(labels) == set(test_labels)
    assert all((labels == c).sum() <= 3 for c in set(labels))
    assert maps.shape[1:] == tiny_data.maps.shape[1:]


def test_eval_sample_comes_only_from_the_test_split_with_matching_labels(tiny_data, tmp_path):
    path = write_eval_sample(tiny_data, tmp_path, per_class_cap=100, seed=0)

    with np.load(path) as sample:
        maps, labels = sample["maps"], sample["labels"]
    test_idx = tiny_data.splits.test
    label_of = {
        wafer.tobytes(): label
        for wafer, label in zip(tiny_data.maps[test_idx], tiny_data.labels[test_idx], strict=True)
    }
    assert len(maps) == len(test_idx)
    pairs = zip(maps, labels, strict=True)
    assert all(label_of[wafer.tobytes()] == label for wafer, label in pairs)


def test_eval_sample_is_deterministic_for_a_seed(tiny_data, tmp_path):
    a = write_eval_sample(tiny_data, tmp_path / "a", per_class_cap=3, seed=1)
    b = write_eval_sample(tiny_data, tmp_path / "b", per_class_cap=3, seed=1)

    with np.load(a) as first, np.load(b) as second:
        assert np.array_equal(first["maps"], second["maps"])
