import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()


def test_entrypoint_is_a_real_uvicorn_factory():
    module_path, factory = re.search(r'"(wm811k\.[\w.]+):(\w+)"', DOCKERFILE).groups()

    assert callable(getattr(importlib.import_module(module_path), factory))
    assert "--factory" in DOCKERFILE


def test_environment_variables_match_the_ones_the_app_reads():
    app_source = (ROOT / "src" / "wm811k" / "api" / "app.py").read_text()
    read_by_app = set(re.findall(r'environ\.get\("(WM811K_\w+)"', app_source))
    set_in_dockerfile = set(re.findall(r"(WM811K_\w+)=", DOCKERFILE))

    assert read_by_app and read_by_app <= set_in_dockerfile


def test_image_installs_only_the_main_dependencies_and_exposes_the_health_route():
    assert "--only main" in DOCKERFILE
    assert "/health" in DOCKERFILE and "EXPOSE 8080" in DOCKERFILE


def test_dockerignore_keeps_the_dataset_and_run_history_out_of_the_build_context():
    ignored = (ROOT / ".dockerignore").read_text().split()

    assert {"LSWMD.pkl", "mlflow.db", "mlartifacts", ".venv"} <= set(ignored)
