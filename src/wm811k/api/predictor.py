import json
from pathlib import Path

import numpy as np
import torch

from wm811k.models import build_backbone
from wm811k.preprocess import to_channels


class Predictor:
    """The exported classifier. Maps must already be resized to `map_size`."""

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
    def predict_batch(self, maps: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Class probabilities, shape (N, n_labels), for resized maps of shape (N, S, S)."""
        chunks = []
        for start in range(0, len(maps), batch_size):
            batch = np.stack([to_channels(m) for m in maps[start : start + batch_size]])
            chunks.append(self.model(torch.from_numpy(batch)).softmax(dim=1).numpy())
        return np.concatenate(chunks)

    def predict(self, resized_map: np.ndarray) -> tuple[str, dict[str, float]]:
        probabilities = self.predict_batch(resized_map[None])[0]
        by_label = dict(zip(self.labels, probabilities.tolist(), strict=True))
        return max(by_label, key=by_label.get), by_label
