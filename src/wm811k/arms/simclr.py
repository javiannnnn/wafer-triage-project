from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from wm811k.arms.common import run_logged
from wm811k.augment import wafer_augment
from wm811k.data import LABELS
from wm811k.experiment import ExperimentData
from wm811k.models import build_backbone
from wm811k.tracking import start_run
from wm811k.training import (
    autocast,
    cycle_batches,
    fit_and_evaluate,
    get_device,
    make_loader,
    seed_everything,
)

ARM = "simclr"
MODEL = "finetune"
PROJECTION_DIM = 128


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR loss: each embedding must pick out its other view among the 2N - 1 candidates."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    similarity = (z @ z.T / temperature).fill_diagonal_(float("-inf"))
    positives = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(similarity, positives)


def pretrain_simclr(cfg: DictConfig, data: ExperimentData) -> Path:
    """Self-supervised pretraining on every training map, labeled or not. Returns the checkpoint."""
    s = cfg.simclr
    seed_everything(cfg.split.seed)
    device = get_device()
    rows = np.concatenate([data.splits.train_labeled, data.splits.train_unlabeled])
    loader = make_loader(
        data.maps[rows], data.labels[rows], s.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    encoder = build_backbone().to(device)
    head = nn.Sequential(
        nn.Linear(encoder.out_dim, 512), nn.ReLU(), nn.Linear(512, PROJECTION_DIM)
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()],
        lr=s.lr,
        weight_decay=cfg.train.weight_decay,
    )

    with start_run(cfg, run_name="simclr-pretrain", tags={"arm": ARM, "stage": "pretrain"}):
        mlflow.log_param("n_pretrain_rows", len(rows))
        encoder.train()
        head.train()
        for step, (x, _) in zip(range(1, s.steps + 1), cycle_batches(loader), strict=False):
            x = x.to(device)
            with autocast(device):
                z1 = head(encoder(wafer_augment(x)))
                z2 = head(encoder(wafer_augment(x)))
                loss = nt_xent_loss(z1.float(), z2.float(), s.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == s.steps:
                mlflow.log_metric("pretrain_loss", loss.item(), step=step)

        checkpoint = Path(s.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), checkpoint)
        mlflow.log_param("checkpoint", str(checkpoint))
    return checkpoint


def load_encoder(model: nn.Module, checkpoint: Path) -> None:
    """Load pretrained encoder weights into a classifier, leaving its fresh `fc` head untouched."""
    result = model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    if result.unexpected_keys:
        raise ValueError(f"checkpoint {checkpoint} has unexpected keys: {result.unexpected_keys}")


def finetune(
    cfg: DictConfig, data: ExperimentData, checkpoint: Path, fraction: float, seed: int
) -> dict:
    """Fine-tune the pretrained encoder with the same loop and budget as the supervised CNN."""

    def fit(train_idx: np.ndarray) -> dict[str, float]:
        model = build_backbone(num_classes=len(LABELS))
        load_encoder(model, checkpoint)
        return fit_and_evaluate(model, data, train_idx, cfg, get_device())

    return run_logged(cfg, data, arm=ARM, model=MODEL, fraction=fraction, seed=seed, fit=fit)


def run_simclr_arm(cfg: DictConfig, data: ExperimentData, fraction: float, seed: int) -> dict:
    """Fine-tune from the configured checkpoint, pretraining it first if it does not exist yet."""
    checkpoint = Path(cfg.simclr.checkpoint)
    if not checkpoint.is_file():
        pretrain_simclr(cfg, data)
    return finetune(cfg, data, checkpoint, fraction, seed)
