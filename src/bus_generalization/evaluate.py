from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import make_loader
from .metrics import bootstrap_ci, classification_metrics, selective_metrics
from .models import build_model
from .train import predict
from .utils import load_config, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config); seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint or Path(cfg["output_dir"]) / "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(**cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    data_cfg, eval_cfg = cfg["data"], cfg["evaluation"]
    manifest_dir = Path(data_cfg["manifests_dir"])
    sets = {
        "BUS-BRA_in_domain": manifest_dir / "busbra_test.csv",
        "BrEaST_external": manifest_dir / "breast_test.csv",
        "BUSI_external": manifest_dir / "busi_test.csv",
    }
    results, prediction_frames = {}, []
    for name, manifest in sets.items():
        loader = make_loader(manifest, data_cfg["image_size"], data_cfg["eval_batch_size"], data_cfg["num_workers"])
        y, p, ids = predict(model, loader, device)
        metrics = classification_metrics(y, p, eval_cfg["threshold"])
        metrics["confidence_intervals_95"] = bootstrap_ci(y, p, eval_cfg["threshold"], eval_cfg["bootstrap_samples"], cfg["seed"])
        metrics["selective_evaluation"] = selective_metrics(y, p, eval_cfg["abstention_coverages"], eval_cfg["threshold"])
        results[name] = metrics
        prediction_frames.append(pd.DataFrame({"dataset": name, "sample_id": ids, "label": y,
                                                "malignant_probability": p, "prediction": (p >= eval_cfg["threshold"]).astype(int)}))
    base = results["BUS-BRA_in_domain"]
    for name in ("BrEaST_external", "BUSI_external"):
        results[name]["generalization_gap"] = {
            metric: results[name][metric] - base[metric]
            for metric in ("balanced_accuracy", "f1", "roc_auc")
        }
    output = Path(cfg["output_dir"])
    save_json(results, output / "evaluation.json")
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    print(pd.DataFrame(results).T[["n", "accuracy", "balanced_accuracy", "f1", "roc_auc"]])


if __name__ == "__main__":
    main()

