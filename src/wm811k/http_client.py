import json
import urllib.request

TIMEOUT_SECONDS = 60


def post_json(base_url: str, path: str, payload: dict) -> int:
    """POST `payload` as JSON and return the status code; HTTP errors raise `HTTPError`."""
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.status


def get_drift_detected(base_url: str) -> bool:
    """Ask the service's drift report for its verdict on the current prediction window."""
    with urllib.request.urlopen(f"{base_url}/drift-report", timeout=TIMEOUT_SECONDS) as response:
        return response.headers["X-Drift-Detected"] == "true"
