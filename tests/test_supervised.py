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


def test_full_label_runs_log_a_loadable_model_and_partial_runs_do_not(tiny_data, tiny_cfg):
    import mlflow
    import pytest

    train_cnn(tiny_cfg, tiny_data, fraction=1.0, seed=0)
    train_cnn(tiny_cfg, tiny_data, fraction=0.5, seed=0)

    mlflow.set_tracking_uri(tiny_cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(tiny_cfg.mlflow.experiment_name)
    runs = {
        r.data.tags["fraction"]: r.info.run_id
        for r in mlflow.search_runs([experiment.experiment_id], output_format="list")
    }
    assert mlflow.pytorch.load_model(f"runs:/{runs['1.0']}/model") is not None
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow.pytorch.load_model(f"runs:/{runs['0.5']}/model")
