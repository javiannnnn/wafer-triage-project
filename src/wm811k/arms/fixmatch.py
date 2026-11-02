import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wm811k.arms.common import run_logged
from wm811k.augment import wafer_augment, weak_augment
from wm811k.data import LABELS
from wm811k.experiment import ExperimentData
from wm811k.models import build_backbone
from wm811k.training import (
    autocast,
    class_weights,
    cycle_batches,
    get_device,
    make_loader,
    select_best_checkpoint,
)

ARM = "fixmatch"
HIDDEN_LABEL = -1


def select_pseudo_labels(
    logits_weak: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (confident-enough mask, predicted class) for the weakly augmented unlabeled batch."""
    confidence, labels = F.softmax(logits_weak.float(), dim=1).max(dim=1)
    return confidence >= threshold, labels


def unlabeled_pool(data: ExperimentData, train_idx: np.ndarray) -> np.ndarray:
    """Train maps whose labels this run may not use: the truly unlabeled plus the hidden labeled."""
    hidden = np.setdiff1d(data.splits.train_labeled, train_idx)
    return np.concatenate([data.splits.train_unlabeled, hidden])


def train_fixmatch(cfg: DictConfig, data: ExperimentData, fraction: float, seed: int) -> dict:
    def fit(train_idx: np.ndarray) -> dict[str, float]:
        f, t = cfg.fixmatch, cfg.train
        device = get_device()
        model = build_backbone(num_classes=len(LABELS)).to(device)
        labels = data.labels[train_idx]
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights(labels).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)

        labeled = cycle_batches(
            make_loader(data.maps[train_idx], labels, t.batch_size, True, t.num_workers)
        )
        pool = unlabeled_pool(data, train_idx)
        unlabeled = cycle_batches(
            make_loader(
                data.maps[pool],
                np.full(len(pool), HIDDEN_LABEL),
                t.batch_size * f.mu,
                True,
                t.num_workers,
            )
        )

        def train_step() -> None:
            x_l, y_l = next(labeled)
            x_u, _ = next(unlabeled)
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            with autocast(device):
                batch = torch.cat([weak_augment(x_l), weak_augment(x_u), wafer_augment(x_u)])
                logits_l, logits_weak, logits_strong = model(batch).split(
                    [len(x_l), len(x_u), len(x_u)]
                )
                confident, pseudo = select_pseudo_labels(logits_weak.detach(), f.threshold)
                per_sample = F.cross_entropy(logits_strong.float(), pseudo, reduction="none")
                unsupervised = (per_sample * confident).mean()
                loss = criterion(logits_l.float(), y_l) + f.lambda_u * unsupervised
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return select_best_checkpoint(model, data, cfg, device, train_step)

    return run_logged(cfg, data, arm=ARM, model="resnet18", fraction=fraction, seed=seed, fit=fit)
