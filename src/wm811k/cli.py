import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from wm811k.api.predictor import Predictor
from wm811k.config import load_config
from wm811k.drift import inflate_defects, reference_frame, simulate_drift
from wm811k.export import best_full_budget_run, export_model, register_run, write_eval_sample
from wm811k.gate import check_gate, sample_macro_f1
from wm811k.http_client import get_drift_detected, post_json
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


def _export(cfg: DictConfig, arms: list[str]) -> None:
    best = best_full_budget_run(collect_results(cfg))
    # export first: a model that cannot be loaded must not end up in the registry
    out = export_model(cfg, best["run_id"], cfg.export_dir)
    write_eval_sample(prepare_experiment_data(cfg), out)
    version = register_run(cfg, best["run_id"])
    print(f"registered {cfg.mlflow.model_name} v{version}; exported {best['series']} run to {out}")
    print(
        f"sample macro-F1 {sample_macro_f1(out):.4f}: "
        "set gate.min_macro_f1 just under it in conf/config.yaml"
    )


def _gate(cfg: DictConfig, arms: list[str]) -> None:
    if not check_gate(cfg):
        sys.exit(
            f"quality gate failed: {cfg.export_dir} must contain weights.pt and eval_sample.npz "
            f"and score sample macro-F1 >= {cfg.gate.min_macro_f1}"
        )
    print("quality gate passed")


def _reference(cfg: DictConfig, arms: list[str]) -> None:
    data = prepare_experiment_data(cfg)
    rng = np.random.default_rng(cfg.split.seed)
    size = min(cfg.drift.reference_size, len(data.splits.val))
    chosen = rng.choice(data.splits.val, size=size, replace=False)
    reference = reference_frame(Predictor.from_dir(cfg.export_dir), data.maps[chosen])
    path = Path(cfg.export_dir) / "reference.csv"
    reference.to_csv(path, index=False)
    print(f"wrote {len(reference)} reference rows to {path}")


def _simulate_drift(cfg: DictConfig, arms: list[str]) -> None:
    data = prepare_experiment_data(cfg)
    n_normal, n_shifted = cfg.drift.normal_samples, cfg.drift.shifted_samples
    chosen = np.random.default_rng(cfg.split.seed).choice(
        data.splits.test, size=n_normal + n_shifted, replace=False
    )
    normal = data.maps[chosen[:n_normal]]
    shifted = inflate_defects(data.maps[chosen[n_normal:]], cfg.drift.shift_rate, cfg.split.seed)
    url = cfg.drift.url
    outcome = simulate_drift(
        post=lambda wafer: post_json(url, "/predict", {"wafer_map": wafer}),
        drifted=lambda: get_drift_detected(url),
        normal_maps=normal,
        shifted_maps=shifted,
    )
    print(f"before shift: drifted={outcome['before_shift']}")
    print(f"after shift: drifted={outcome['after_shift']}")


COMMANDS: dict[str, Callable[[DictConfig, list[str]], None]] = {
    "prepare": _prepare,
    "sweep": _sweep,
    "results": _results,
    "export": _export,
    "gate": _gate,
    "reference": _reference,
    "simulate-drift": _simulate_drift,
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        sys.exit(f"usage: python -m wm811k.cli {{{'|'.join(COMMANDS)}}} [arms...] [key=value...]")
    arms, overrides = split_cli_args(argv[1:])
    COMMANDS[argv[0]](load_config(overrides), arms)


if __name__ == "__main__":
    main()
