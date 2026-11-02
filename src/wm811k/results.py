from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from omegaconf import DictConfig

from wm811k.sweep import ARMS, SERIES_BY_RUN

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# categorical slots 1-4 in fixed order, one per arm; validated with the dataviz palette script
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_MARKERS = ["o", "s", "^", "D"]
MIN_LABEL_GAP = 0.05


def collect_results(cfg: DictConfig) -> pd.DataFrame:
    """One row per finished sweep run: series, fraction, seed and every logged metric."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(
        [experiment.experiment_id], filter_string="attributes.status = 'FINISHED'"
    )
    if runs.empty or "tags.fraction" not in runs:
        return pd.DataFrame()

    runs = runs[runs["tags.fraction"].notna()]
    identities = zip(runs["tags.arm"], runs["tags.model"], strict=True)
    results = pd.DataFrame(
        {
            "series": [SERIES_BY_RUN[identity] for identity in identities],
            "fraction": runs["tags.fraction"].astype(float).to_numpy(),
            "seed": runs["tags.seed"].astype(int).to_numpy(),
        }
    )
    for column in runs.columns[runs.columns.str.startswith("metrics.")]:
        results[column.removeprefix("metrics.")] = runs[column].to_numpy()
    return results


def summarize(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean, std (0 for a single seed) and run count of `metric` per series and label budget."""
    summary = (
        results.groupby(["series", "fraction"])[metric].agg(["mean", "std", "count"]).reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def spread_labels(positions: list[float], min_gap: float = MIN_LABEL_GAP) -> list[float]:
    """Move label heights as little as possible (least squares) so neighbours are `min_gap` apart.

    Equal gaps are subtracted, the heights are made non-decreasing by pooling adjacent
    violators, and the gaps are added back: crowded labels fan out around their own centre.
    """
    if not positions:
        return []
    order = np.argsort(positions)
    ranks = np.arange(len(positions))
    shifted = np.asarray(positions, dtype=float)[order] - ranks * min_gap

    blocks: list[list[float]] = []  # [mean, size]
    for value in shifted:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (mean_b, size_b), (mean_a, size_a) = blocks.pop(), blocks.pop()
            size = size_a + size_b
            blocks.append([(mean_a * size_a + mean_b * size_b) / size, size])
    fitted = np.concatenate([np.full(int(size), mean) for mean, size in blocks])

    spread = np.empty(len(positions))
    spread[order] = fitted + ranks * min_gap
    return spread.tolist()


def plot_label_efficiency(summary: pd.DataFrame, path: str | Path) -> None:
    """Mean test macro-F1 vs share of labels, one series per arm, with a +/- 1 std band."""
    fig = Figure(figsize=(7.5, 4.6), dpi=200, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)

    fractions = sorted(summary["fraction"].unique())
    end_points = []
    for index, (name, arm) in enumerate(ARMS.items()):
        series = summary[summary["series"] == name].sort_values("fraction")
        if series.empty:
            continue
        color, marker = SERIES_COLORS[index], SERIES_MARKERS[index]
        ax.fill_between(
            series["fraction"],
            series["mean"] - series["std"],
            series["mean"] + series["std"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            series["fraction"],
            series["mean"],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=arm.title,
        )
        end_points.append((arm.title, series["fraction"].iloc[-1], series["mean"].iloc[-1]))

    label_heights = spread_labels([y for _, _, y in end_points])
    for (title, x, _), height in zip(end_points, label_heights, strict=True):
        ax.annotate(
            title,
            (x, height),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(fractions))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(min(fractions) * 0.8, max(fractions) * 3.2)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_xlabel("Share of labeled training maps", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Test macro-F1 (mean, +/- 1 std over seeds)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Label efficiency on WM-811K", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
