from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import make_loader
from .metrics import classification_metrics
from .models import build_model
from .utils import load_config, save_json, seed_everything


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    labels, probabilities, ids = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
        labels.extend(batch["label"].tolist())
        ids.extend(batch["sample_id"])
    return np.asarray(labels), np.asarray(probabilities), ids


def train_experiment(config="configs/baseline.yaml"):
    """Train one configured experiment and return its history and checkpoint path."""
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = cfg["data"]
    manifests = Path(paths["manifests_dir"])
    train_manifest, val_manifest = manifests / "busbra_train.csv", manifests / "busbra_val.csv"
    train_loader = make_loader(train_manifest, paths["image_size"], paths["train_batch_size"],
                               paths["num_workers"], True, paths.get("root", "data"))
    val_loader = make_loader(val_manifest, paths["image_size"], paths["eval_batch_size"],
                             paths["num_workers"], data_root=paths.get("root", "data"))

    model = build_model(**cfg["model"]).to(device)
    train_cfg = cfg["training"]
    weights = None
    if train_cfg["use_class_weights"]:
        counts = pd.read_csv(train_manifest).label.value_counts().reindex([0, 1]).values
        weights = torch.tensor(len(counts) and counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"] and device.type == "cuda")
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    best, stale, history = -float("inf"), 0, []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train(); losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=train_cfg["amp"] and device.type == "cuda"):
                loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(loss.item())
        y, p, _ = predict(model, val_loader, device)
        metrics = classification_metrics(y, p)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row); print(row)
        monitored = metrics[train_cfg["monitor"]]
        if monitored is not None and monitored > best:
            best, stale = monitored, 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch}, output / "best.pt")
        else:
            stale += 1
            if stale >= train_cfg["patience"]:
                break
    save_json({"history": history, "best_validation_score": best}, output / "training_history.json")
    return {"history": history, "best_validation_score": best,
            "checkpoint_path": str(output / "best.pt")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    train_experiment(args.config)


if __name__ == "__main__":
    main()

