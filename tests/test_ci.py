import re
from pathlib import Path

import yaml

from wm811k.cli import COMMANDS
from wm811k.config import load_config

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
WORKFLOW_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _run_steps(job):
    return [step["run"] for step in WORKFLOW["jobs"][job]["steps"] if "run" in step]


def test_ci_lints_tests_and_applies_the_quality_gate():
    commands = "\n".join(_run_steps("test"))

    assert "ruff check" in commands and "pytest" in commands
    assert "wm811k.cli gate" in commands


def test_every_cli_command_ci_uses_exists():
    used = set(re.findall(r"wm811k\.cli (\S+)", WORKFLOW_TEXT))

    assert used and used <= set(COMMANDS)


def test_model_dependent_steps_are_skipped_until_a_model_is_committed():
    cfg = load_config()
    steps = WORKFLOW["jobs"]["test"]["steps"] + WORKFLOW["jobs"]["docker"]["steps"]
    model_dependent = ("wm811k.cli gate", "docker build")
    guarded = [s for s in steps if any(c in s.get("run", "") for c in model_dependent)]

    assert guarded
    for step in guarded:
        assert "hashFiles" in step["if"] and cfg.export_dir in step["if"]


def test_the_gate_step_waits_for_both_the_weights_and_the_committed_sample():
    gate_step = next(
        s for s in WORKFLOW["jobs"]["test"]["steps"] if "wm811k.cli gate" in s.get("run", "")
    )

    assert "weights.pt" in gate_step["if"] and "eval_sample.npz" in gate_step["if"]


def test_docker_job_smoke_tests_health_and_predict():
    commands = "\n".join(_run_steps("docker"))

    assert "docker build" in commands and "docker run" in commands
    assert "/health" in commands and "/predict" in commands
