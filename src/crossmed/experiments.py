from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .evaluate import evaluate_experiment
from .train import train_experiment
from .utils import release_gpu_memory


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
                          selected=None, stage="all", skip_completed=True,
                          cleanup_between_runs=False, strict_missing=False):
    """Train/evaluate selected experiments and return the aggregate table."""
    configs = load_experiment_matrix(matrix_path)
    selected = list(selected) if selected else list(configs)
    unknown = set(selected) - set(configs)
    if unknown:
        raise ValueError(f"Unknown experiments {sorted(unknown)}; choose from {sorted(configs)}")

    status_rows = []
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
        try:
            if stage in {"all", "train"}:
                if skip_completed and checkpoint.exists():
                    print(f"Skipping training; checkpoint exists: {checkpoint}")
                    status_rows.append({"experiment": name, "stage": "train",
                                        "status": "checkpoint_exists", "path": str(checkpoint)})
                else:
                    train_experiment(config_path)
                    status_rows.append({"experiment": name, "stage": "train",
                                        "status": "trained", "path": str(checkpoint)})
            if stage in {"all", "evaluate"}:
                if not checkpoint.exists():
                    message = f"Skipping {name}; checkpoint not found: {checkpoint}"
                    if strict_missing:
                        raise FileNotFoundError(message)
                    print(message)
                    status_rows.append({"experiment": name, "stage": "evaluate",
                                        "status": "missing_checkpoint", "path": str(checkpoint)})
                    continue
                if skip_completed and evaluation.exists():
                    print(f"Skipping evaluation; results exist: {evaluation}")
                    status_rows.append({"experiment": name, "stage": "evaluate",
                                        "status": "evaluation_exists", "path": str(evaluation)})
                else:
                    evaluate_experiment(config_path, checkpoint)
                    status_rows.append({"experiment": name, "stage": "evaluate",
                                        "status": "evaluated", "path": str(evaluation)})
        finally:
            if cleanup_between_runs:
                # Intended for sequential local/notebook execution on a
                # memory-limited GPU. CARC jobs normally run one model each.
                release_gpu_memory()
    if stage == "train":
        return pd.DataFrame(status_rows)
    return aggregate_results(matrix_path)


def main():
    parser = argparse.ArgumentParser(description="Run the CrossMed experiment matrix.")
    parser.add_argument("--matrix", default="configs/experiment_matrix.yaml")
    parser.add_argument("--experiments", nargs="*", help="Names to run; omit for all")
    parser.add_argument("--stage", choices=["all", "train", "evaluate"], default="all")
    parser.add_argument("--force", action="store_true", help="Rerun completed stages")
    parser.add_argument("--cleanup-between-runs", action="store_true",
                        help="Unload CUDA cache between sequential models")
    parser.add_argument("--strict-missing", action="store_true",
                        help="Fail instead of skipping an untrained experiment")
    args = parser.parse_args()
    table = run_experiment_matrix(args.matrix, args.experiments, args.stage,
                                  not args.force, args.cleanup_between_runs,
                                  args.strict_missing)
    if table.empty:
        print("No evaluation results are available yet.")
    else:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
