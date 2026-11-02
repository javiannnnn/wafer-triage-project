import mlflow
import numpy as np
import torch

from wm811k.arms.fixmatch import select_pseudo_labels, train_fixmatch, unlabeled_pool


def test_pseudo_label_mask_respects_threshold():
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    mask, labels = select_pseudo_labels(logits, threshold=0.95)

    assert mask.tolist() == [True, False]
    assert labels[0].item() == 0


def test_no_confident_samples_gives_empty_mask_not_error():
    mask, labels = select_pseudo_labels(torch.zeros(4, 3), threshold=0.95)

    assert not mask.any() and labels.shape == (4,)


def test_unlabeled_pool_hides_budget_labels_but_reuses_the_rest(tiny_data):
    train_idx = tiny_data.splits.train_labeled[:10]

    pool = unlabeled_pool(tiny_data, train_idx)

    assert not set(pool) & set(train_idx)
    assert set(tiny_data.splits.train_labeled[10:]) <= set(pool)
    assert set(tiny_data.splits.train_unlabeled) <= set(pool)
    assert len(pool) == len(np.unique(pool))


def test_train_fixmatch_logs_tagged_run_and_returns_metrics(tiny_data, tiny_cfg):
    metrics = train_fixmatch(tiny_cfg, tiny_data, fraction=0.5, seed=0)

    assert {"val_macro_f1", "test_macro_f1"} <= set(metrics)
    mlflow.set_tracking_uri(tiny_cfg.mlflow.tracking_uri)
    run = mlflow.search_runs(
        [mlflow.get_experiment_by_name(tiny_cfg.mlflow.experiment_name).experiment_id],
        output_format="list",
    )[0]
    assert run.data.tags["arm"] == "fixmatch"
