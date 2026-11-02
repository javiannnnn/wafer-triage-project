import numpy as np
import pytest
from fastapi.testclient import TestClient

from wm811k.api.app import create_app
from wm811k.api.predictor import Predictor
from wm811k.data import LABELS
from wm811k.drift import inflate_defects, reference_frame


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


def _maps(n=60, seed=0):
    return np.random.default_rng(seed).integers(0, 3, size=(n, 16, 16), dtype=np.uint8)


def _post_all(client, maps):
    for m in maps:
        assert client.post("/predict", json={"wafer_map": m.tolist()}).status_code == 200


def _app_with_reference(model_dir, tmp_path, maps):
    reference_frame(Predictor.from_dir(model_dir), maps).to_csv(
        model_dir / "reference.csv", index=False
    )
    flag = tmp_path / "retrain.flag"
    return create_app(model_dir, flag_path=flag, drift_min_samples=50), flag


def test_drift_report_without_a_reference_is_503_with_instructions(client):
    response = client.get("/drift-report")

    assert response.status_code == 503 and "reference" in response.json()["detail"]


def test_drift_report_for_in_distribution_traffic_does_not_raise_the_flag(model_dir, tmp_path):
    maps = _maps()
    app, flag = _app_with_reference(model_dir, tmp_path, maps)
    client = TestClient(app)
    _post_all(client, maps)

    response = client.get("/drift-report")

    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert response.headers["x-drift-detected"] == "false"
    assert not flag.exists()


def test_drift_report_for_shifted_traffic_raises_the_retrain_flag(model_dir, tmp_path):
    maps = _maps()
    app, flag = _app_with_reference(model_dir, tmp_path, maps)
    client = TestClient(app)
    _post_all(client, inflate_defects(maps, rate=1.0, seed=0))

    response = client.get("/drift-report")

    assert response.headers["x-drift-detected"] == "true"
    assert flag.is_file() and "drift" in flag.read_text().lower()


def test_drift_report_with_too_few_predictions_is_not_an_alarm(model_dir, tmp_path):
    maps = _maps()
    app, flag = _app_with_reference(model_dir, tmp_path, maps)
    client = TestClient(app)
    _post_all(client, inflate_defects(maps[:5], rate=1.0, seed=0))

    response = client.get("/drift-report")

    assert response.status_code == 200 and response.headers["x-drift-detected"] == "false"
    assert not flag.exists()
