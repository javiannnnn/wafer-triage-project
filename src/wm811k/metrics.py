from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, recall_score

from wm811k.data import LABELS


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], labels: list[str]) -> dict[str, float]:
    """Macro-F1 and per-class recall over all `labels`; absent classes score 0.0."""
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        **{f"recall_{label}": float(r) for label, r in zip(labels, recalls, strict=True)},
    }


def evaluate_encoded(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """`evaluate` for class indices into LABELS."""
    names = np.array(LABELS)
    return evaluate(names[y_true], names[y_pred], LABELS)
