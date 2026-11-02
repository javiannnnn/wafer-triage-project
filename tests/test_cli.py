from wm811k.cli import split_cli_args


def test_split_cli_args_separates_arm_names_from_config_overrides():
    arms, overrides = split_cli_args(["gbm", "cnn", "data.map_size=32", "budgets=[0.1,1.0]"])

    assert arms == ["gbm", "cnn"]
    assert overrides == ["data.map_size=32", "budgets=[0.1,1.0]"]


def test_split_cli_args_with_no_arms_means_all_arms():
    arms, overrides = split_cli_args(["seeds=[0]"])

    assert arms == ["gbm", "cnn", "simclr", "fixmatch"]
    assert overrides == ["seeds=[0]"]


def _gate_args(tmp_path, f1):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(f'{{"test_macro_f1": {f1}}}')
    return ["gate", f"gate.metadata_file={metadata.as_posix()}", "gate.min_macro_f1=0.4"]


def test_gate_command_exits_with_a_message_when_the_gate_fails(tmp_path):
    import pytest

    from wm811k.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(_gate_args(tmp_path, 0.1))

    assert "quality gate failed" in str(exit_info.value)


def test_gate_command_passes_quietly_when_the_gate_holds(tmp_path, capsys):
    from wm811k.cli import main

    main(_gate_args(tmp_path, 0.9))

    assert "quality gate passed" in capsys.readouterr().out


def test_unknown_command_exits_with_usage():
    import pytest

    from wm811k.cli import main

    with pytest.raises(SystemExit, match="usage"):
        main(["nonsense"])


def test_export_command_exports_the_best_full_label_run(tiny_data, tiny_cfg, tmp_path):
    from wm811k.arms.supervised import train_cnn
    from wm811k.cli import main

    train_cnn(tiny_cfg, tiny_data, fraction=1.0, seed=0)
    out = tmp_path / "exported"

    main(
        [
            "export",
            f"mlflow.tracking_uri={tiny_cfg.mlflow.tracking_uri}",
            f"export_dir={out.as_posix()}",
            "data.map_size=16",
        ]
    )

    assert (out / "weights.pt").is_file() and (out / "metadata.json").is_file()


def _data_overrides(big_raw_path, tmp_path):
    return [
        f"data.path={big_raw_path.as_posix()}",
        f"data.cache={(tmp_path / 'cache.npz').as_posix()}",
        "data.map_size=16",
    ]


def test_reference_command_writes_reference_csv_next_to_the_model(
    big_raw_path, model_dir, tmp_path
):
    import pandas as pd

    from wm811k.cli import main
    from wm811k.monitoring import COLUMNS

    overrides = _data_overrides(big_raw_path, tmp_path)
    main(["reference", f"export_dir={model_dir.as_posix()}", *overrides])

    reference = pd.read_csv(model_dir / "reference.csv")
    assert list(reference.columns) == COLUMNS and len(reference) > 0


def test_simulate_drift_command_detects_the_shift_end_to_end(
    big_raw_path, model_dir, tmp_path, capsys
):
    import threading
    import time

    import uvicorn

    from wm811k.api.app import create_app
    from wm811k.cli import main

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
