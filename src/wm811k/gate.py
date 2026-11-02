from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from wm811k.api.predictor import Predictor
from wm811k.export import SAMPLE_FILE
from wm811k.metrics import evaluate_encoded


def sample_macro_f1(model_dir: Path) -> float:
    """Macro-F1 of the exported weights on the held-out sample committed next to them."""
    predictor = Predictor.from_dir(model_dir)
    with np.load(Path(model_dir) / SAMPLE_FILE) as sample:
        maps, labels = sample["maps"], sample["labels"]
    predicted = predictor.predict_batch(maps).argmax(axis=1)
    return evaluate_encoded(labels, predicted)["macro_f1"]


def check_gate(cfg: DictConfig) -> bool:
    """Re-score the committed model on the committed sample; missing files fail the gate."""
    model_dir = Path(cfg.export_dir)
    if not (model_dir / "weights.pt").is_file() or not (model_dir / SAMPLE_FILE).is_file():
        return False
    return sample_macro_f1(model_dir) >= cfg.gate.min_macro_f1
