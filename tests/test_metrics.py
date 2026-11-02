import mlflow
import pytest

from wm811k.config import load_config
from wm811k.metrics import evaluate
from wm811k.tracking import flatten_config, start_run

LABELS = ["a", "b", "c"]


def test_macro_f1_perfect_is_one():
    metrics = evaluate(["a", "b", "c"], ["a", "b", "c"], LABELS)

    assert metrics["macro_f1"] == 1.0


def test_all_classes_present_in_output_keys():
    metrics = evaluate(["a", "b"], ["a", "b"], LABELS)

    assert {f"recall_{c}" for c in LABELS} <= set(metrics)


def test_class_absent_from_predictions_gives_zero_recall_not_error():
    metrics = evaluate(["a", "b", "b"], ["a", "a", "a"], LABELS)

    assert metrics["recall_b"] == 0.0
    assert metrics["recall_a"] == 1.0


def test_class_absent_from_truth_gives_zero_recall_not_error():
    metrics = evaluate(["a", "a"], ["a", "c"], LABELS)

    assert metrics["recall_c"] == 0.0


def test_flatten_config_uses_dotted_keys():
    flat = flatten_config({"data": {"map_size": 64}, "seeds": [0, 1]})

    assert flat == {"data.map_size": 64, "seeds": "[0, 1]"}


def test_start_run_logs_params_and_tags(tmp_path):
    db = (tmp_path / "mlflow.db").as_posix()
    cfg = load_config(
        [
            f"mlflow.tracking_uri=sqlite:///{db}",
            f"mlflow.artifact_root={(tmp_path / 'mlartifacts').as_posix()}",
        ]
    )

    with start_run(cfg, run_name="smoke", tags={"arm": "test"}) as run:
        mlflow.log_metric("macro_f1", 0.5)

    finished = mlflow.get_run(run.info.run_id)
    assert finished.data.params["data.map_size"] == "64"
    assert finished.data.tags["arm"] == "test"
    assert finished.data.metrics["macro_f1"] == pytest.approx(0.5)
