import json
import re
from pathlib import Path

from wm811k.cli import COMMANDS
from wm811k.sweep import ARMS

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_run.ipynb"


def _code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_notebook_calls_only_existing_cli_commands_and_arms():
    pattern = r"wm811k\(\s*['\"](\w+)['\"](?:,\s*['\"](\w+)['\"])?"
    calls = re.findall(pattern, "\n".join(_code_cells()))

    assert calls, "notebook never calls the CLI"
    for command, arm in calls:
        assert command in COMMANDS
        assert arm == "" or arm in ARMS


def test_notebook_sweeps_every_arm():
    swept = re.findall(r"wm811k\('sweep',\s*'(\w+)'", "\n".join(_code_cells()))

    assert set(swept) == set(ARMS)


def test_notebook_overrides_are_valid_config_keys():
    from wm811k.config import load_config

    source = "\n".join(_code_cells())
    keys = re.findall(r'f?"([a-z_]+\.[a-z_]+|artifacts_dir)=', source)

    assert keys
    cfg = load_config()
    for key in keys:
        node = cfg
        for part in key.split("."):
            assert part in node, f"{key} is not a config key"
            node = node[part]
