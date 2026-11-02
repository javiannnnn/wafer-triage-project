import mlflow

from wm811k.arms.supervised import train_cnn, train_gbm


def _only_run(cfg):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    assert len(runs) == 1
    return runs[0]


def test_train_cnn_logs_tagged_run_with_test_metrics(tiny_data, tiny_cfg):
    metrics = train_cnn(tiny_cfg, tiny_data, fraction=0.5, seed=0)

    run = _only_run(tiny_cfg)
    assert run.data.tags["arm"] == "supervised" and run.data.tags["model"] == "cnn"
    assert run.data.tags["fraction"] == "0.5" and run.data.tags["seed"] == "0"
    assert run.data.metrics["test_macro_f1"] == metrics["test_macro_f1"]


def test_train_gbm_logs_tagged_run_with_test_metrics(tiny_data, tiny_cfg):
    metrics = train_gbm(tiny_cfg, tiny_data, fraction=1.0, seed=0)

    run = _only_run(tiny_cfg)
    assert run.data.tags["model"] == "gbm"
    assert 0.0 <= metrics["test_macro_f1"] <= 1.0
