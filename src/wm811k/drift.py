from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from wm811k.api.predictor import Predictor
from wm811k.monitoring import COLUMNS, summarize_wafer

DRIFT_SHARE = 0.3  # of the 3 monitored columns, one drifting is enough to call the window drifted
P_VALUE_THRESHOLD = 0.001  # strict per column: large windows make the default 0.05 too jumpy
NUMERICAL = ["die_count", "defect_ratio"]
CATEGORICAL = ["label"]


@dataclass(frozen=True)
class DriftResult:
    drifted: bool
    share_drifted_features: float
    html: str
    n_current: int
    note: str = ""


def _drifted_share(report: dict) -> float:
    for metric in report["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return float(metric["value"]["share"])
    raise ValueError("Evidently report has no DriftedColumnsCount metric")


def check_drift(
    reference: pd.DataFrame, current: pd.DataFrame, min_samples: int = 200
) -> DriftResult:
    """Compare recent predictions against the reference; too few samples is never an alarm."""
    if len(current) < min_samples:
        note = (
            f"only {len(current)} predictions in the window; "
            f"at least {min_samples} are needed to test for drift"
        )
        return DriftResult(False, 0.0, f"<p>{note}</p>", len(current), note)

    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE, threshold=P_VALUE_THRESHOLD)])
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    share = _drifted_share(snapshot.dict())
    return DriftResult(
        drifted=share >= DRIFT_SHARE,
        share_drifted_features=share,
        html=snapshot.get_html_str(as_iframe=False),
        n_current=len(current),
    )


def write_retrain_flag(path: Path, result: DriftResult) -> None:
    """Leave a marker file telling the retraining pipeline that the model needs attention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"drift detected: {result.share_drifted_features:.0%} of monitored features drifted "
        f"over {result.n_current} predictions\n"
    )


def reference_frame(predictor: Predictor, maps: np.ndarray) -> pd.DataFrame:
    """What healthy traffic looks like: summary statistics and predicted label of good maps."""
    predicted = predictor.predict_batch(maps).argmax(axis=1)
    rows = [
        {**summarize_wafer(wafer), "label": predictor.labels[index]}
        for wafer, index in zip(maps, predicted, strict=True)
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def inflate_defects(maps: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Simulate a process shift: each good die turns defective with probability `rate`."""
    flipped = (maps == 1) & (np.random.default_rng(seed).random(maps.shape) < rate)
    shifted = maps.copy()
    shifted[flipped] = 2
    return shifted


def simulate_drift(
    post: Callable[[list[list[int]]], object],
    drifted: Callable[[], bool],
    normal_maps: np.ndarray,
    shifted_maps: np.ndarray,
) -> dict[str, bool]:
    """Send normal traffic then shifted traffic, asking the service for its verdict after each."""
    for wafer in normal_maps:
        post(wafer.tolist())
    before_shift = drifted()
    for wafer in shifted_maps:
        post(wafer.tolist())
    return {"before_shift": before_shift, "after_shift": drifted()}
