import numpy as np
from omegaconf import DictConfig
from xgboost import XGBClassifier

from wm811k.arms.common import run_logged
from wm811k.data import LABELS
from wm811k.experiment import ExperimentData
from wm811k.features import extract_features
from wm811k.metrics import evaluate_encoded
from wm811k.models import build_backbone
from wm811k.training import class_weights, fit_and_evaluate, get_device

ARM = "supervised"
MODEL_CNN = "cnn"
MODEL_GBM = "gbm"


def _feature_matrix(data: ExperimentData, idx: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(m) for m in data.maps[idx]])


def train_cnn(cfg: DictConfig, data: ExperimentData, fraction: float, seed: int) -> dict:
    def fit(train_idx: np.ndarray) -> dict[str, float]:
        model = build_backbone(num_classes=len(LABELS))
        return fit_and_evaluate(model, data, train_idx, cfg, get_device())

    return run_logged(cfg, data, arm=ARM, model=MODEL_CNN, fraction=fraction, seed=seed, fit=fit)


def train_gbm(cfg: DictConfig, data: ExperimentData, fraction: float, seed: int) -> dict:
    def fit(train_idx: np.ndarray) -> dict[str, float]:
        labels = data.labels[train_idx]
        model = XGBClassifier(
            n_estimators=cfg.gbm.n_estimators,
            max_depth=cfg.gbm.max_depth,
            learning_rate=cfg.gbm.learning_rate,
            tree_method="hist",
            random_state=seed,
        )
        model.fit(
            _feature_matrix(data, train_idx),
            labels,
            sample_weight=class_weights(labels).numpy()[labels],
        )

        def score(idx: np.ndarray) -> dict[str, float]:
            return evaluate_encoded(data.labels[idx], model.predict(_feature_matrix(data, idx)))

        return {
            "val_macro_f1": score(data.splits.val)["macro_f1"],
            **{f"test_{k}": v for k, v in score(data.splits.test).items()},
        }

    return run_logged(cfg, data, arm=ARM, model=MODEL_GBM, fraction=fraction, seed=seed, fit=fit)
