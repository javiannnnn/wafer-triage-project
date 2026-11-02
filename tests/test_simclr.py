import mlflow
import pytest
import torch

from wm811k.arms.simclr import finetune, load_encoder, nt_xent_loss, pretrain_simclr
from wm811k.models import build_backbone


def test_nt_xent_is_lower_for_matching_pairs_than_shuffled_pairs():
    z = torch.randn(16, 8)

    matched = nt_xent_loss(z, z + 0.01 * torch.randn_like(z), temperature=0.2)
    shuffled = nt_xent_loss(z, z[torch.randperm(16)], temperature=0.2)

    assert matched < shuffled


def test_nt_xent_is_symmetric():
    a, b = torch.randn(8, 4), torch.randn(8, 4)

    assert torch.isclose(nt_xent_loss(a, b, 0.2), nt_xent_loss(b, a, 0.2))


def test_pretrain_saves_loadable_encoder_and_logs_run(tiny_data, tiny_cfg, tmp_path):
    ckpt = tmp_path / "enc.pt"
    cfg = _with(tiny_cfg, steps=4, batch_size=16, checkpoint=str(ckpt))

    path = pretrain_simclr(cfg, tiny_data)

    assert path == ckpt and ckpt.is_file()
    build_backbone().load_state_dict(torch.load(ckpt))
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    runs = mlflow.search_runs(
        [mlflow.get_experiment_by_name(cfg.mlflow.experiment_name).experiment_id],
        output_format="list",
    )
    assert runs[0].data.tags["arm"] == "simclr" and runs[0].data.tags["stage"] == "pretrain"


def test_pretrain_uses_only_train_rows(tiny_data, tiny_cfg, tmp_path, monkeypatch):
    seen = []
    import wm811k.arms.simclr as simclr

    real = simclr.make_loader

    def spy(maps, *args, **kwargs):
        seen.append(len(maps))
        return real(maps, *args, **kwargs)

    monkeypatch.setattr(simclr, "make_loader", spy)
    cfg = _with(tiny_cfg, steps=2, batch_size=16, checkpoint=str(tmp_path / "e.pt"))

    pretrain_simclr(cfg, tiny_data)

    train_rows = len(tiny_data.splits.train_labeled) + len(tiny_data.splits.train_unlabeled)
    assert seen == [train_rows]


def test_finetune_logs_simclr_arm_tagged_run(tiny_data, tiny_cfg, tmp_path):
    ckpt = tmp_path / "enc.pt"
    torch.save(build_backbone().state_dict(), ckpt)

    metrics = finetune(tiny_cfg, tiny_data, ckpt, fraction=0.5, seed=0)

    assert "test_macro_f1" in metrics
    mlflow.set_tracking_uri(tiny_cfg.mlflow.tracking_uri)
    run = mlflow.search_runs(
        [mlflow.get_experiment_by_name(tiny_cfg.mlflow.experiment_name).experiment_id],
        output_format="list",
    )[0]
    assert run.data.tags["arm"] == "simclr" and run.data.tags["model"] == "finetune"


def test_load_encoder_copies_weights_into_classifier(tmp_path):
    ckpt = tmp_path / "enc.pt"
    encoder = build_backbone()
    torch.save(encoder.state_dict(), ckpt)
    classifier = build_backbone(num_classes=9)

    load_encoder(classifier, ckpt)

    assert torch.equal(classifier.conv1.weight, encoder.conv1.weight)


def test_load_encoder_rejects_checkpoint_with_unexpected_keys(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"not_a_layer": torch.zeros(1)}, ckpt)

    with pytest.raises(ValueError, match="not_a_layer"):
        load_encoder(build_backbone(num_classes=9), ckpt)


def _with(cfg, **simclr_overrides):
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(cfg, {"simclr": simclr_overrides})
    return merged
