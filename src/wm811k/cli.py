import sys
from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig

from wm811k.config import load_config
from wm811k.pipeline import prepare_experiment_data
from wm811k.results import collect_results, plot_label_efficiency, summarize
from wm811k.sweep import ARMS, run_sweep


def split_cli_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split `argv` into arm names and `key=value` config overrides; no arms means all arms."""
    overrides = [a for a in argv if "=" in a]
    arms = [a for a in argv if "=" not in a]
    return arms or list(ARMS), overrides


def _prepare(cfg: DictConfig, arms: list[str]) -> None:
    data = prepare_experiment_data(cfg)
    print(f"cache ready at {cfg.data.cache}: {data.maps.shape[0]} maps")


def _sweep(cfg: DictConfig, arms: list[str]) -> None:
    run_sweep(cfg, prepare_experiment_data(cfg), arms)


def _results(cfg: DictConfig, arms: list[str]) -> None:
    results = collect_results(cfg)
    out = Path(cfg.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(results, "test_macro_f1")
    results.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "results.csv", index=False)
    plot_label_efficiency(summary, out / "label_efficiency.png")
    print(summary.to_string(index=False))


COMMANDS: dict[str, Callable[[DictConfig, list[str]], None]] = {
    "prepare": _prepare,
    "sweep": _sweep,
    "results": _results,
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        sys.exit("usage: python -m wm811k.cli {prepare|sweep|results} [arms...] [key=value...]")
    arms, overrides = split_cli_args(argv[1:])
    COMMANDS[argv[0]](load_config(overrides), arms)


if __name__ == "__main__":
    main()
