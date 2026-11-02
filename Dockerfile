# Inference service: FastAPI + the exported wafer-defect classifier (CPU only).
# Build after exporting a model:  python -m wm811k.cli export && python -m wm811k.cli reference

FROM python:3.11-slim AS builder

ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install poetry==2.4.1

WORKDIR /app
COPY pyproject.toml poetry.lock ./
# serving needs the main group only; the train and dev groups stay out of the image
RUN poetry install --only main --no-root


FROM python:3.11-slim AS runtime

# OpenMP runtime used by torch and scikit-learn
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY models/best_model ./models/best_model

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    WM811K_MODEL_DIR=/app/models/best_model \
    WM811K_RETRAIN_FLAG=/app/artifacts/retrain.flag

RUN useradd --create-home appuser && mkdir -p /app/artifacts && chown appuser /app/artifacts
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=4)"

CMD ["uvicorn", "wm811k.api.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8080"]
