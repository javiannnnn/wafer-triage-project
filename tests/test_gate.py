import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from wm811k.api.predictor import Predictor
from wm811k.gate import SAMPLE_FILE, check_gate, sample_macro_f1
from wm811k.metrics import evaluate_encoded
from wm811k.models import build_backbone


@pytest.fixture
def gated_model_dir(model_dir):
    """The random 16x16 model plus a committed-style held-out sample covering three classes."""
    rng = np.random.default_rng(0)
    maps = rng.integers(0, 3, size=(45, 16, 16), dtype=np.uint8)
    labels = np.arange(45) % 3
    np.savez_compressed(model_dir / SAMPLE_FILE, maps=maps, labels=labels)
    return model_dir, maps, labels


def _cfg(model_dir, floor):
    return OmegaConf.create({"export_dir": str(model_dir), "gate": {"min_macro_f1": floor}})


def test_sample_macro_f1_is_the_models_score_on_the_committed_sample(gated_model_dir):
    model_dir, maps, labels = gated_model_dir
    predicted = Predictor.from_dir(model_dir).predict_batch(maps).argmax(axis=1)

    assert sample_macro_f1(model_dir) == evaluate_encoded(labels, predicted)["macro_f1"]


def test_gate_passes_when_the_floor_is_met(gated_model_dir):
    model_dir, _, _ = gated_model_dir

    assert check_gate(_cfg(model_dir, 0.0)) is True


def test_gate_passes_exactly_at_the_models_own_score(gated_model_dir):
    model_dir, _, _ = gated_model_dir

    assert check_gate(_cfg(model_dir, sample_macro_f1(model_dir))) is True


def test_gate_fails_when_the_floor_is_not_met(gated_model_dir):
    model_dir, _, _ = gated_model_dir

    assert check_gate(_cfg(model_dir, sample_macro_f1(model_dir) + 0.01)) is False


def test_gate_notices_when_the_weights_are_swapped_for_worse_ones(gated_model_dir):
    model_dir, _, _ = gated_model_dir
    floor = sample_macro_f1(model_dir)
    for seed in range(40):  # look for another random initialisation that scores lower
        torch.manual_seed(1000 + seed)
        torch.save(build_backbone(num_classes=9).state_dict(), model_dir / "weights.pt")
        if sample_macro_f1(model_dir) < floor:
            break
    else:
        pytest.skip("no lower-scoring random model found")

    assert check_gate(_cfg(model_dir, floor)) is False


def test_gate_fails_closed_when_the_sample_is_missing(gated_model_dir):
    model_dir, _, _ = gated_model_dir
    (model_dir / SAMPLE_FILE).unlink()

    assert check_gate(_cfg(model_dir, 0.0)) is False


def test_gate_fails_closed_when_there_is_no_model(tmp_path):
    assert check_gate(_cfg(tmp_path / "nowhere", 0.0)) is False
