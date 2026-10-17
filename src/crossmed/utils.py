from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Config files live in <project>/configs; resolve project paths from there.
    project_root = path.parent.parent
    for key in ("root", "manifests_dir"):
        value = Path(config["data"][key])
        if not value.is_absolute():
            config["data"][key] = str(project_root / value)
    output = Path(config["output_dir"])
    if not output.is_absolute():
        config["output_dir"] = str(project_root / output)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)

