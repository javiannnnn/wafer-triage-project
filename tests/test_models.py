import torch

from wm811k.models import build_backbone


def test_classifier_forward_shape():
    model = build_backbone(num_classes=9)

    assert model(torch.zeros(4, 3, 16, 16)).shape == (4, 9)


def test_feature_extractor_exposes_out_dim():
    encoder = build_backbone()

    assert encoder(torch.zeros(4, 3, 16, 16)).shape == (4, encoder.out_dim)
