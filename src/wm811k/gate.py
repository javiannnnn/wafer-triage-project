import json
from pathlib import Path

from omegaconf import DictConfig


def passes_gate(metrics: dict[str, float], min_macro_f1: float) -> bool:
    """A model passes when its test macro-F1 is present and at least the configured floor."""
    return metrics.get("test_macro_f1", float("-inf")) >= min_macro_f1


def check_gate(cfg: DictConfig) -> bool:
    """Apply the quality gate to the committed metadata of the exported model."""
    metadata_file = Path(cfg.gate.metadata_file)
    if not metadata_file.is_file():
        return False
    return passes_gate(json.loads(metadata_file.read_text()), cfg.gate.min_macro_f1)
