import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from wm811k.api.predictor import Predictor
from wm811k.api.schemas import PredictRequest, PredictResponse
from wm811k.drift import check_drift, write_retrain_flag
from wm811k.monitoring import PredictionWindow, summarize_wafer
from wm811k.preprocess import resize_map

DEFAULT_MODEL_DIR = "models/best_model"
DEFAULT_FLAG_PATH = "artifacts/retrain.flag"
WINDOW_SIZE = 5000
DRIFT_MIN_SAMPLES = 200
REFERENCE_FILE = "reference.csv"


def create_app(
    model_dir: Path,
    window_size: int = WINDOW_SIZE,
    flag_path: Path = Path(DEFAULT_FLAG_PATH),
    drift_min_samples: int = DRIFT_MIN_SAMPLES,
) -> FastAPI:
    app = FastAPI(title="WM-811K wafer defect classifier")
    predictor = Predictor.from_dir(model_dir)
    reference_path = Path(model_dir) / REFERENCE_FILE
    reference = pd.read_csv(reference_path) if reference_path.is_file() else None
    app.state.window = PredictionWindow(window_size)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        resized = resize_map(np.array(request.wafer_map, dtype=np.uint8), predictor.map_size)
        label, probabilities = predictor.predict(resized)
        app.state.window.add({**summarize_wafer(resized), "label": label})
        return PredictResponse(label=label, probabilities=probabilities)

    @app.get("/drift-report", response_class=HTMLResponse)
    def drift_report() -> HTMLResponse:
        if reference is None:
            raise HTTPException(
                status_code=503,
                detail=f"{REFERENCE_FILE} not found next to the model; "
                "run `python -m wm811k.cli reference`",
            )
        result = check_drift(reference, app.state.window.frame(), drift_min_samples)
        if result.drifted:
            write_retrain_flag(flag_path, result)
        return HTMLResponse(result.html, headers={"X-Drift-Detected": str(result.drifted).lower()})

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory: WM811K_MODEL_DIR and WM811K_RETRAIN_FLAG configure the service."""
    return create_app(
        Path(os.environ.get("WM811K_MODEL_DIR", DEFAULT_MODEL_DIR)),
        flag_path=Path(os.environ.get("WM811K_RETRAIN_FLAG", DEFAULT_FLAG_PATH)),
    )
