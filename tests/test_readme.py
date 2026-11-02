import re
from pathlib import Path

from wm811k.cli import COMMANDS
from wm811k.config import load_config

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def _cli_commands(text):
    return set(re.findall(r"wm811k\.cli ([a-z][a-z-]*)", text))


def test_extractor_finds_commands_in_prose_and_code_blocks():
    assert _cli_commands("run `python -m wm811k.cli bogus-cmd` then wm811k.cli sweep gbm") == {
        "bogus-cmd",
        "sweep",
    }


def test_every_cli_command_the_readme_mentions_exists():
    mentioned = _cli_commands(README)

    assert mentioned == set(COMMANDS)  # and every command is documented


def test_config_keys_the_readme_tells_users_to_set_exist():
    cfg = load_config()

    assert "min_macro_f1" in cfg.gate
    for key in ("budgets", "seeds", "train"):
        assert key in cfg
