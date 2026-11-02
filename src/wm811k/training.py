import copy
import random
from collections.abc import Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from wm811k.data import LABELS
from wm811k.experiment import ExperimentData
from wm811k.metrics import evaluate_encoded
from wm811k.preprocess import to_channels

N_CLASSES = len(LABELS)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device: torch.device):
    """bfloat16 mixed precision on GPU, a no-op on CPU."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


class WaferDataset(Dataset):
    """Resized uint8 maps -> one-hot float tensors; labels pass through unchanged."""

    def __init__(self, maps: np.ndarray, labels: np.ndarray):
        self.maps, self.labels = maps, labels

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(to_channels(self.maps[i])), int(self.labels[i])


def make_loader(
    maps: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        WaferDataset(maps, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle and len(maps) > batch_size,
    )


def cycle_batches(loader: DataLoader) -> Iterator:
    """Iterate over the loader forever, reshuffling on every pass."""
    while True:
        yield from loader


def class_weights(labels: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Square-root inverse-frequency weights, scaled to mean 1 over the classes present."""
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights[counts > 0].mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    for x, _ in loader:
        with autocast(device):
            batches.append(model(x.to(device)).argmax(dim=1).cpu())
    return torch.cat(batches).numpy()


def evaluate_split(
    model: nn.Module, data: ExperimentData, idx: np.ndarray, cfg: DictConfig, device: torch.device
) -> dict[str, float]:
    loader = make_loader(data.maps[idx], data.labels[idx], cfg.train.batch_size, shuffle=False)
    return evaluate_encoded(data.labels[idx], predict(model, loader, device))


def select_best_checkpoint(
    model: nn.Module,
    data: ExperimentData,
    cfg: DictConfig,
    device: torch.device,
    train_step: Callable[[], None],
) -> dict[str, float]:
    """Call `train_step` cfg.train.steps times, validating every cfg.train.eval_every steps.

    Restores the weights with the best val macro-F1, then returns `val_macro_f1` and that
    checkpoint's test metrics, prefixed `test_`.
    """
    t = cfg.train
    best_f1, best_state = -1.0, None
    for step in range(1, t.steps + 1):
        model.train()
        train_step()
        if step % t.eval_every == 0 or step == t.steps:
            val_f1 = evaluate_split(model, data, data.splits.val, cfg, device)["macro_f1"]
            if val_f1 > best_f1:
                best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test = evaluate_split(model, data, data.splits.test, cfg, device)
    return {"val_macro_f1": best_f1, **{f"test_{k}": v for k, v in test.items()}}


def fit_and_evaluate(
    model: nn.Module,
    data: ExperimentData,
    train_idx: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """Supervised training with class-weighted cross-entropy on `train_idx`."""
    t = cfg.train
    model.to(device)
    labels = data.labels[train_idx]
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    loader = make_loader(
        data.maps[train_idx], labels, t.batch_size, shuffle=True, num_workers=t.num_workers
    )
    batches = cycle_batches(loader)

    def train_step() -> None:
        x, y = next(batches)
        with autocast(device):
            loss = criterion(model(x.to(device)), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return select_best_checkpoint(model, data, cfg, device, train_step)
