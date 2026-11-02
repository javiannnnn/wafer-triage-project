import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ONLY = ["mlflow", "xgboost", "matplotlib", "hydra", "omegaconf"]


def test_the_api_imports_without_any_training_only_package():
    """The Docker image installs only the main dependency group; the API must not need more."""
    blocked = ", ".join(f"{name!r}" for name in TRAINING_ONLY)
    code = (
        f"import sys\nfor m in [{blocked}]: sys.modules[m] = None\n"
        "import wm811k.api.app\nprint('ok')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
    )

    assert result.stdout.strip() == "ok", result.stderr


def test_the_training_cli_imports_without_evidently():
    """Colab has no evidently; `sweep`, `results` and `export` must not need it."""
    blocked = ["evidently"]  # the only serving package Colab lacks: mlflow brings fastapi itself
    code = "\n".join(
        [
            "import sys",
            f"for m in {blocked!r}: sys.modules[m] = None",
            "import wm811k.cli",
            "print('ok')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
    )

    assert result.stdout.strip() == "ok", result.stderr


def test_training_only_packages_are_not_main_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    main = " ".join(pyproject["project"]["dependencies"]).lower()

    for name in ("mlflow", "xgboost", "matplotlib", "hydra-core"):
        assert name not in main, f"{name} should be in the train group"
    assert "mlflow" in pyproject["tool"]["poetry"]["group"]["train"]["dependencies"]
