import json
from pathlib import Path

import numpy as np
import torch

from wm811k.models import build_backbone
from wm811k.preprocess import resize_map, to_channels


class Predictor:
    """The exported classifier: raw wafer map in, label and class probabilities out."""

    def __init__(self, model: torch.nn.Module, labels: list[str], map_size: int):
        self.model = model.eval()
        self.labels = labels
        self.map_size = map_size

    @classmethod
    def from_dir(cls, model_dir: Path) -> "Predictor":
        weights = Path(model_dir) / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        labels = json.loads((Path(model_dir) / "labels.json").read_text())
        map_size = json.loads((Path(model_dir) / "metadata.json").read_text())["map_size"]
        model = build_backbone(num_classes=len(labels))
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        return cls(model, labels, map_size)

    @torch.no_grad()
    def predict(self, wafer: np.ndarray) -> tuple[str, dict[str, float]]:
        channels = to_channels(resize_map(wafer, self.map_size))
        probabilities = self.model(torch.from_numpy(channels)[None]).softmax(dim=1)[0].tolist()
        by_label = dict(zip(self.labels, probabilities, strict=True))
        return max(by_label, key=by_label.get), by_label
