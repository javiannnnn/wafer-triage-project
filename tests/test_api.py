import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from wm811k.api.app import create_app
from wm811k.data import LABELS
from wm811k.models import build_backbone


@pytest.fixture
def model_dir(tmp_path):
    """A randomly initialised model in the exported layout, for 16x16 inputs."""
    torch.manual_seed(0)
    torch.save(build_backbone(num_classes=9).state_dict(), tmp_path / "weights.pt")
    (tmp_path / "labels.json").write_text(json.dumps(LABELS))
    (tmp_path / "metadata.json").write_text(json.dumps({"map_size": 16}))
    return tmp_path


@pytest.fixture
def client(model_dir):
    return TestClient(create_app(model_dir))


def _wafer(h=20, w=20, seed=0):
    return np.random.default_rng(seed).integers(0, 3, size=(h, w)).tolist()


def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_predict_valid_map_returns_label_and_probabilities(client):
    response = client.post("/predict", json={"wafer_map": _wafer()})

    body = response.json()
    assert response.status_code == 200
    assert body["label"] in LABELS
    assert set(body["probabilities"]) == set(LABELS)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["probabilities"][body["label"]] == max(body["probabilities"].values())


@pytest.mark.parametrize("shape", [(1, 1), (30, 90), (512, 512)])
def test_predict_accepts_any_map_size_up_to_the_limit(client, shape):
    assert client.post("/predict", json={"wafer_map": _wafer(*shape)}).status_code == 200


def test_predict_is_deterministic(client):
    a = client.post("/predict", json={"wafer_map": _wafer()}).json()
    b = client.post("/predict", json={"wafer_map": _wafer()}).json()

    assert a == b


@pytest.mark.parametrize(
    "wafer_map",
    [
        [[0, 1, 3]],  # value outside {0, 1, 2}
        [[0, 1, -1]],
        [],  # empty payload
        [[]],  # empty row
        [[0, 1], [1]],  # ragged rows
        [[0] * 2 for _ in range(513)],  # too many rows
        [[0] * 513],  # too many columns
    ],
    ids=["bad-value", "negative", "empty", "empty-row", "ragged", "too-tall", "too-wide"],
)
def test_predict_rejects_invalid_maps_with_422(client, wafer_map):
    assert client.post("/predict", json={"wafer_map": wafer_map}).status_code == 422


def test_predict_rejects_missing_or_non_list_payload_with_422(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"wafer_map": "abc"}).status_code == 422
    assert client.post("/predict", content=b"not json").status_code == 422


def test_each_prediction_is_recorded_in_the_monitoring_window(model_dir):
    app = create_app(model_dir)
    client = TestClient(app)

    client.post("/predict", json={"wafer_map": _wafer(seed=1)})
    client.post("/predict", json={"wafer_map": _wafer(seed=2)})
    client.post("/predict", json={"wafer_map": [[3]]})  # rejected: must not be recorded

    frame = app.state.window.frame()
    assert len(frame) == 2
    assert set(frame["label"]) <= set(LABELS)


def test_missing_model_directory_fails_at_startup_with_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights.pt"):
        create_app(tmp_path)
