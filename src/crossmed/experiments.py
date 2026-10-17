from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .evaluate import evaluate_experiment
from .train import train_experiment


def load_experiment_matrix(path="configs/experiment_matrix.yaml"):
    """Return {experiment_name: absolute_config_path}."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    experiments = matrix.get("experiments", {})
    if not experiments:
        raise ValueError(f"No experiments defined in {path}")
    return {name: str((path.parent / config).resolve()) for name, config in experiments.items()}


def aggregate_results(matrix_path="configs/experiment_matrix.yaml",
                      output_path=None):
    """Combine available evaluation JSON files into a presentation-ready CSV."""
    configs = load_experiment_matrix(matrix_path)
    rows = []
    for experiment, config_path in configs.items():
        with Path(config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        project_root = Path(config_path).parent.parent
        output_dir = Path(config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        evaluation_path = output_dir / "evaluation.json"
        if not evaluation_path.exists():
            continue
        with evaluation_path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
        for dataset, metrics in evaluation.items():
            row = {"experiment": experiment, "dataset": dataset}
            for metric in ("n", "accuracy", "balanced_accuracy", "precision",
                           "recall_sensitivity", "specificity", "f1", "roc_auc"):
                row[metric] = metrics.get(metric)
            rows.append(row)
    table = pd.DataFrame(rows)
    matrix_path = Path(matrix_path).resolve()
    destination = Path(output_path) if output_path else matrix_path.parent.parent / "results/model_comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table


def run_experiment_matrix(matrix_path="configs/experiment_matrix.yaml",
                          selected=None, stage="all", skip_completed=True):
    """Train/evaluate selected experiments and return the aggregate table."""
    configs = load_experiment_matrix(matrix_path)
    selected = list(selected) if selected else list(configs)
    unknown = set(selected) - set(configs)
    if unknown:
        raise ValueError(f"Unknown experiments {sorted(unknown)}; choose from {sorted(configs)}")

    for name in selected:
        config_path = configs[name]
        with Path(config_path).open(encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle)
        project_root = Path(config_path).parent.parent
        output_dir = Path(raw_config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        checkpoint = output_dir / "best.pt"
        evaluation = output_dir / "evaluation.json"
        print(f"\n=== {name} ===")
        if stage in {"all", "train"}:
            if skip_completed and checkpoint.exists():
                print(f"Skipping training; checkpoint exists: {checkpoint}")
            else:
                train_experiment(config_path)
        if stage in {"all", "evaluate"}:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Train {name} first; missing {checkpoint}")
            if skip_completed and evaluation.exists():
                print(f"Skipping evaluation; results exist: {evaluation}")
            else:
                evaluate_experiment(config_path, checkpoint)
    return aggregate_results(matrix_path)


def main():
    parser = argparse.ArgumentParser(description="Run the CrossMed experiment matrix.")
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--experiments", nargs="*", help="Names to run; omit for all")
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages")
    args = parser.parse_args()
    table = run_experiment_matrix(args.matrix, args.experiments, args.stage, not args.force)
    if table.empty:
        print("No evaluation results are available yet.")
    else:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
