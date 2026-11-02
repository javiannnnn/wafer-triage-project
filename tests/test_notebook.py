import json
import re
from pathlib import Path

from wm811k.cli import COMMANDS
from wm811k.config import load_config
from wm811k.sweep import ARMS

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_run.ipynb"


def _source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


def test_notebook_sweeps_every_arm_through_the_resumable_helper():
    swept = re.findall(r"sweep_arm\(['\"](\w+)['\"]\)", _source())

    assert set(swept) == set(ARMS)


def test_notebook_saves_progress_after_every_run_and_even_when_a_run_fails():
    source = _source()

    assert "after_run=save_progress" in source
    assert "finally" in source


def test_notebook_calls_only_existing_cli_commands():
    commands = set(re.findall(r"wm811k\(\s*['\"]([\w-]+)['\"]", _source()))

    assert {"results", "export", "reference"} <= commands
    assert commands <= set(COMMANDS)


def test_notebook_exports_the_model_on_colab_so_no_mlflow_artifacts_must_be_moved():
    source = _source()

    assert source.index("wm811k('export')") < source.index("wm811k('reference')")
    assert "models/best_model" in source


def test_notebook_overrides_are_valid_config_keys():
    keys = re.findall(r'f?"([a-z_]+\.[a-z_]+|artifacts_dir|export_dir)=', _source())

    assert keys
    cfg = load_config()
    for key in keys:
        node = cfg
        for part in key.split("."):
            assert part in node, f"{key} is not a config key"
            node = node[part]
