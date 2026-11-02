import numpy as np
import pytest
import torch

from wm811k.models import build_backbone
from wm811k.training import (
    WaferDataset,
    class_weights,
    fit_and_evaluate,
    make_loader,
    predict,
)


def test_class_weights_rarer_class_is_heavier_and_mean_is_one():
    weights = class_weights(np.array([0, 0, 0, 0, 1]), n_classes=3)

    assert weights[1] > weights[0]
    assert weights[[0, 1]].mean().item() == pytest.approx(1.0)


def test_class_weights_handles_absent_class():
    weights = class_weights(np.array([0, 1]), n_classes=3)

    assert torch.isfinite(weights).all()


def test_dataset_item_is_one_hot_tensor_and_label():
    maps = np.random.default_rng(0).integers(0, 3, size=(5, 8, 8), dtype=np.uint8)
    labels = np.arange(5)

    x, y = WaferDataset(maps, labels)[2]

    assert x.shape == (3, 8, 8) and x.dtype == torch.float32 and y == 2


def test_predict_returns_one_class_index_per_sample(tiny_data):
    loader = make_loader(tiny_data.maps[:20], tiny_data.labels[:20], batch_size=8, shuffle=False)

    preds = predict(build_backbone(num_classes=9), loader, torch.device("cpu"))

    assert preds.shape == (20,) and preds.min() >= 0 and preds.max() < 9


def test_fit_and_evaluate_returns_val_and_test_metrics(tiny_data, tiny_cfg):
    metrics = fit_and_evaluate(
        build_backbone(num_classes=9),
        tiny_data,
        tiny_data.splits.train_labeled,
        tiny_cfg,
        torch.device("cpu"),
    )

    assert {"val_macro_f1", "test_macro_f1", "test_recall_Center"} <= set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_fit_and_evaluate_learns_separable_classes(tiny_data, tiny_cfg):
    from omegaconf import OmegaConf

    from wm811k.training import seed_everything

    cfg = OmegaConf.merge(tiny_cfg, {"train": {"steps": 150, "eval_every": 50}})
    seed_everything(0)

    metrics = fit_and_evaluate(
        build_backbone(num_classes=9),
        tiny_data,
        tiny_data.splits.train_labeled,
        cfg,
        torch.device("cpu"),
    )

    # three present classes out of nine -> a perfect macro-F1 would be 1/3
    assert metrics["test_macro_f1"] > 0.25


def test_select_best_checkpoint_restores_the_best_weights(tiny_data, tiny_cfg):
    from omegaconf import OmegaConf

    from wm811k.training import select_best_checkpoint

    cfg = OmegaConf.merge(tiny_cfg, {"train": {"steps": 3, "eval_every": 1}})
    model = build_backbone(num_classes=9)
    initial = model.conv1.weight.detach().clone()
    calls = []

    def wrecking_step():
        calls.append(1)
        if len(calls) > 1:  # step 1 leaves the model alone; later steps always predict class 8
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
                model.fc.bias[8] = 1000.0

    select_best_checkpoint(model, tiny_data, cfg, torch.device("cpu"), wrecking_step)

    assert len(calls) == 3
    assert torch.equal(model.conv1.weight, initial)
