import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI

from wm811k.api.predictor import Predictor
from wm811k.api.schemas import PredictRequest, PredictResponse
from wm811k.monitoring import PredictionWindow, summarize_wafer

DEFAULT_MODEL_DIR = "models/best_model"
WINDOW_SIZE = 5000


def create_app(model_dir: Path, window_size: int = WINDOW_SIZE) -> FastAPI:
    app = FastAPI(title="WM-811K wafer defect classifier")
    predictor = Predictor.from_dir(model_dir)
    app.state.window = PredictionWindow(window_size)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        wafer = np.array(request.wafer_map, dtype=np.uint8)
        label, probabilities = predictor.predict(wafer)
        app.state.window.add({**summarize_wafer(wafer), "label": label})
        return PredictResponse(label=label, probabilities=probabilities)

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory: the model directory comes from WM811K_MODEL_DIR."""
    return create_app(Path(os.environ.get("WM811K_MODEL_DIR", DEFAULT_MODEL_DIR)))
