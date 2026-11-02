import mlflow
import pandas as pd
import pytest

from wm811k.results import collect_results, plot_label_efficiency, summarize
from wm811k.tracking import start_run


@pytest.fixture
def logged_cfg(tiny_cfg):
    """MLflow store holding gbm/cnn runs at two budgets x two seeds, plus a pretrain run."""
    f1 = {("gbm", 0.1): 0.2, ("gbm", 1.0): 0.4, ("cnn", 0.1): 0.3, ("cnn", 1.0): 0.7}
    for (model, fraction), base in f1.items():
        for seed in (0, 1):
            tags = {
                "arm": "supervised",
                "model": model,
                "fraction": str(fraction),
                "seed": str(seed),
            }
            with start_run(tiny_cfg, f"{model}-{fraction}-{seed}", tags):
                mlflow.log_metrics({"test_macro_f1": base + 0.01 * seed, "val_macro_f1": base})
    with start_run(tiny_cfg, "pretrain", {"arm": "simclr", "stage": "pretrain"}):
        pass
    return tiny_cfg


def test_collect_results_has_one_row_per_run_and_skips_pretraining(logged_cfg):
    results = collect_results(logged_cfg)

    assert len(results) == 8
    assert set(results["series"]) == {"gbm", "cnn"}
    assert results["fraction"].dtype == float and results["seed"].dtype == int
    assert {"test_macro_f1", "val_macro_f1", "run_id"} <= set(results.columns)


def test_collect_results_is_empty_for_a_fresh_store(tiny_cfg):
    assert collect_results(tiny_cfg).empty


def test_summarize_reports_mean_std_and_count_per_series_and_budget(logged_cfg):
    summary = summarize(collect_results(logged_cfg), metric="test_macro_f1")

    row = summary[(summary["series"] == "cnn") & (summary["fraction"] == 1.0)].iloc[0]
    assert row["mean"] == pytest.approx(0.705)
    assert row["std"] == pytest.approx(pd.Series([0.7, 0.71]).std())
    assert row["count"] == 2


def test_summarize_std_is_zero_for_a_single_seed():
    results = pd.DataFrame(
        {"series": ["gbm"], "fraction": [0.1], "seed": [0], "test_macro_f1": [0.5]}
    )

    assert summarize(results, "test_macro_f1")["std"].iloc[0] == 0.0


def test_plot_label_efficiency_writes_a_png(logged_cfg, tmp_path):
    summary = summarize(collect_results(logged_cfg), "test_macro_f1")
    out = tmp_path / "curve.png"

    plot_label_efficiency(summary, out)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_spread_labels_fans_a_cluster_out_around_its_center_keeping_order():
    from wm811k.results import spread_labels

    spread = spread_labels([0.50, 0.51, 0.52, 0.90], min_gap=0.05)

    assert spread[1] - spread[0] >= 0.05 - 1e-9
    assert spread[2] - spread[1] >= 0.05 - 1e-9
    assert sum(spread[:3]) / 3 == pytest.approx(0.51)
    assert spread[3] == pytest.approx(0.90)


def test_spread_labels_keeps_each_label_aligned_with_its_own_point():
    from wm811k.results import spread_labels

    spread = spread_labels([0.72, 0.60, 0.72], min_gap=0.05)

    assert spread[1] == pytest.approx(0.60)  # the isolated label does not move
    assert abs(spread[0] - spread[2]) >= 0.05 - 1e-9


def test_spread_labels_leaves_well_separated_labels_alone():
    from wm811k.results import spread_labels

    assert spread_labels([0.1, 0.5, 0.9]) == pytest.approx([0.1, 0.5, 0.9])


def test_spread_labels_handles_no_labels():
    from wm811k.results import spread_labels

    assert spread_labels([]) == []
