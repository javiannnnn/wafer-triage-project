import threading
import time

import mlflow
import numpy as np
import pandas as pd
import pytest
import uvicorn

import wm811k.cli as cli
from wm811k.api.app import create_app
from wm811k.arms.supervised import train_cnn
from wm811k.cli import main, split_cli_args
from wm811k.config import load_config
from wm811k.gate import SAMPLE_FILE
from wm811k.monitoring import COLUMNS
from wm811k.pipeline import prepare_experiment_data


def test_split_cli_args_separates_arm_names_from_config_overrides():
    arms, overrides = split_cli_args(["gbm", "cnn", "data.map_size=32", "budgets=[0.1,1.0]"])

    assert arms == ["gbm", "cnn"]
    assert overrides == ["data.map_size=32", "budgets=[0.1,1.0]"]


def test_split_cli_args_with_no_arms_means_all_arms():
    arms, overrides = split_cli_args(["seeds=[0]"])

    assert arms == ["gbm", "cnn", "simclr", "fixmatch"]
    assert overrides == ["seeds=[0]"]


def _gate_args(model_dir, floor):
    maps = np.random.default_rng(0).integers(0, 3, size=(30, 16, 16), dtype=np.uint8)
    np.savez_compressed(model_dir / SAMPLE_FILE, maps=maps, labels=np.arange(30) % 3)
    return ["gate", f"export_dir={model_dir.as_posix()}", f"gate.min_macro_f1={floor}"]


def test_gate_command_exits_with_a_message_when_the_gate_fails(model_dir):
    with pytest.raises(SystemExit) as exit_info:
        main(_gate_args(model_dir, 1.5))

    assert "quality gate failed" in str(exit_info.value)


def test_gate_command_passes_quietly_when_the_gate_holds(model_dir, capsys):
    main(_gate_args(model_dir, 0.0))

    assert "quality gate passed" in capsys.readouterr().out


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit, match="usage"):
        main(["nonsense"])


def _data_overrides(big_raw_path, tmp_path):
    return [
        f"data.path={big_raw_path.as_posix()}",
        f"data.cache={(tmp_path / 'cache.npz').as_posix()}",
        "data.map_size=16",
    ]


def _run_args(tiny_cfg, big_raw_path, tmp_path):
    return [
        f"mlflow.tracking_uri={tiny_cfg.mlflow.tracking_uri}",
        f"mlflow.artifact_root={tiny_cfg.mlflow.artifact_root}",
        *_data_overrides(big_raw_path, tmp_path),
        "train.steps=6",
        "train.eval_every=3",
        "train.batch_size=16",
    ]


def _train_full_label_run(args):
    cfg = load_config(args)
    train_cnn(cfg, prepare_experiment_data(cfg), fraction=1.0, seed=0)


def test_export_command_writes_weights_and_a_gate_sample_and_registers_the_model(
    tiny_cfg, big_raw_path, tmp_path, capsys
):
    args = _run_args(tiny_cfg, big_raw_path, tmp_path)
    _train_full_label_run(args)
    out = tmp_path / "exported"

    main(["export", *args, f"export_dir={out.as_posix()}"])

    assert (out / "weights.pt").is_file() and (out / "metadata.json").is_file()
    assert (out / SAMPLE_FILE).is_file()
    assert "gate.min_macro_f1" in capsys.readouterr().out
    client = mlflow.MlflowClient(tiny_cfg.mlflow.tracking_uri)
    versions = client.search_model_versions(f"name='{tiny_cfg.mlflow.model_name}'")
    assert len(versions) == 1


def test_a_failed_export_does_not_leave_a_registered_model_version(
    tiny_cfg, big_raw_path, tmp_path, monkeypatch
):
    args = _run_args(tiny_cfg, big_raw_path, tmp_path)
    _train_full_label_run(args)

    def broken_export(*_):
        raise OSError("model artifact is not reachable")

    monkeypatch.setattr(cli, "export_model", broken_export)
    with pytest.raises(OSError):
        main(["export", *args, f"export_dir={(tmp_path / 'out').as_posix()}"])

    assert mlflow.MlflowClient(tiny_cfg.mlflow.tracking_uri).search_registered_models() == []


def test_reference_command_writes_reference_csv_next_to_the_model(
    big_raw_path, model_dir, tmp_path
):
    overrides = _data_overrides(big_raw_path, tmp_path)

    main(["reference", f"export_dir={model_dir.as_posix()}", *overrides])

    reference = pd.read_csv(model_dir / "reference.csv")
    assert list(reference.columns) == COLUMNS and len(reference) > 0


def test_simulate_drift_command_detects_the_shift_end_to_end(
    big_raw_path, model_dir, tmp_path, capsys
):
    overrides = _data_overrides(big_raw_path, tmp_path)
    main(["reference", f"export_dir={model_dir.as_posix()}", *overrides])
    app = create_app(model_dir, flag_path=tmp_path / "retrain.flag", drift_min_samples=15)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]

    try:
        main(
            [
                "simulate-drift",
                f"drift.url=http://127.0.0.1:{port}",
                "drift.normal_samples=20",
                "drift.shifted_samples=20",
                "drift.shift_rate=0.9",
                *overrides,
            ]
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    output = capsys.readouterr().out
    assert "before shift: drifted=False" in output
    assert "after shift: drifted=True" in output
