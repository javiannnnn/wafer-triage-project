import torch.nn as nn
from torchvision.models import resnet18


def build_backbone(num_classes: int | None = None) -> nn.Module:
    """ResNet-18 with a 3x3 stem (keeping the stem max-pool) for small 3-channel wafer maps.

    With `num_classes=None` it is a feature extractor whose output width is `out_dim`.
    """
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    out_dim = net.fc.in_features
    net.fc = nn.Identity() if num_classes is None else nn.Linear(out_dim, num_classes)
    net.out_dim = out_dim
    return net
