import threading
import time
import urllib.error

import numpy as np
import pytest
import uvicorn

from wm811k.api.app import create_app
from wm811k.api.predictor import Predictor
from wm811k.drift import inflate_defects, reference_frame
from wm811k.http_client import get_drift_detected, post_json


@pytest.fixture
def live_url(model_dir, tmp_path):
    """The service running on a real local port, with a reference and a low drift threshold."""
    maps = np.random.default_rng(0).integers(0, 3, size=(60, 16, 16), dtype=np.uint8)
    reference_frame(Predictor.from_dir(model_dir), maps).to_csv(
        model_dir / "reference.csv", index=False
    )
    app = create_app(model_dir, flag_path=tmp_path / "retrain.flag", drift_min_samples=40)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", maps
    server.should_exit = True
    thread.join(timeout=10)


def test_post_json_returns_the_status_code(live_url):
    url, maps = live_url

    assert post_json(url, "/predict", {"wafer_map": maps[0].tolist()}) == 200


def test_post_json_raises_on_a_rejected_payload(live_url):
    url, _ = live_url

    with pytest.raises(urllib.error.HTTPError) as error:
        post_json(url, "/predict", {"wafer_map": [[7]]})

    assert error.value.code == 422


def test_get_drift_detected_tracks_the_traffic(live_url):
    url, maps = live_url
    for wafer in maps:
        post_json(url, "/predict", {"wafer_map": wafer.tolist()})
    assert get_drift_detected(url) is False

    for wafer in inflate_defects(maps, rate=1.0, seed=0):
        post_json(url, "/predict", {"wafer_map": wafer.tolist()})
    assert get_drift_detected(url) is True
