from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_col(frame, candidates, required=True):
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Could not find any of {candidates}; columns are {list(frame.columns)}")
    return None


def _resolve_image(root, value):
    value = str(value).strip()
    direct = root / value
    if direct.is_file():
        return direct.resolve()
    matches = list(root.rglob(Path(value).name))
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {value!r}; found {len(matches)}")
    return matches[0].resolve()


def _map_label(value):
    text = str(value).strip().lower()
    if text in {"0", "benign", "b", "negative"} or "benign" in text:
        return 0
    if text in {"1", "malignant", "m", "positive"} or "malignant" in text:
        return 1
    return None


def prepare_busbra(root: Path, metadata: Path):
    frame = pd.read_csv(metadata)
    image_col = _find_col(frame, ["image", "image_name", "filename", "file_name", "path"])
    label_col = _find_col(frame, ["pathology", "classification", "label"])
    patient_col = _find_col(frame, ["patient_id", "patient", "case_id", "case"], required=False)
    out = pd.DataFrame({
        "image_path": frame[image_col].map(lambda x: str(_resolve_image(root, x))),
        "label": frame[label_col].map(_map_label),
    }).dropna(subset=["label"])
    out["label"] = out.label.astype(int)
    out["sample_id"] = frame.loc[out.index, image_col].astype(str).values
    out["patient_id"] = frame.loc[out.index, patient_col].astype(str).values if patient_col else ""
    out["dataset"] = "BUS-BRA"
    return out.reset_index(drop=True), patient_col is not None


def prepare_breast(root: Path, metadata: Path):
    frame = pd.read_excel(metadata)
    image_col = _find_col(frame, ["filename(s) of ultrasound b-mode image(s)", "image", "filename", "file_name"])
    label_col = _find_col(frame, ["classification", "pathology", "label"])
    rows = []
    for idx, row in frame.iterrows():
        label = _map_label(row[label_col])
        if label is None:
            continue
        # Some cells contain more than one image filename.
        names = [x.strip() for x in re.split(r"[;,\n]", str(row[image_col])) if x.strip()]
        for name in names:
            rows.append({"image_path": str(_resolve_image(root, name)), "label": label,
                         "sample_id": name, "patient_id": str(idx), "dataset": "BrEaST"})
    return pd.DataFrame(rows)


def prepare_busi(root: Path):
    rows = []
    for class_name, label in (("benign", 0), ("malignant", 1)):
        folder = root / class_name
        if not folder.is_dir():
            matches = [p for p in root.rglob(class_name) if p.is_dir()]
            if len(matches) != 1:
                raise ValueError(f"Could not uniquely locate BUSI {class_name!r} directory under {root}")
            folder = matches[0]
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES and "_mask" not in path.stem.lower():
                rows.append({"image_path": str(path.resolve()), "label": label,
                             "sample_id": path.name, "patient_id": path.stem, "dataset": "BUSI"})
    return pd.DataFrame(rows)


def split_source(frame, seed, allow_image_split=False):
    has_groups = frame.patient_id.astype(str).str.len().gt(0).all()
    if not has_groups and not allow_image_split:
        raise ValueError("No BUS-BRA patient ID column found. Use --allow-image-split only after checking leakage risk.")
    if has_groups:
        first = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(first.split(frame, frame.label, frame.patient_id))
        train_val = frame.iloc[train_val_idx].reset_index(drop=True)
        second = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1)
        train_idx, val_idx = next(second.split(train_val, train_val.label, train_val.patient_id))
        return train_val.iloc[train_idx], train_val.iloc[val_idx], frame.iloc[test_idx]
    train_val, test = train_test_split(frame, test_size=0.2, stratify=frame.label, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.125, stratify=train_val.label, random_state=seed + 1)
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Create standardized manifests for all three datasets.")
    parser.add_argument("--busbra-root", type=Path, required=True)
    parser.add_argument("--busbra-csv", type=Path, required=True)
    parser.add_argument("--breast-root", type=Path, required=True)
    parser.add_argument("--breast-xlsx", type=Path, required=True)
    parser.add_argument("--busi-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=566)
    parser.add_argument("--allow-image-split", action="store_true")
    args = parser.parse_args()

    source, _ = prepare_busbra(args.busbra_root, args.busbra_csv)
    train, val, test = split_source(source, args.seed, args.allow_image_split)
    manifests = {
        "busbra_train.csv": train,
        "busbra_val.csv": val,
        "busbra_test.csv": test,
        "breast_test.csv": prepare_breast(args.breast_root, args.breast_xlsx),
        "busi_test.csv": prepare_busi(args.busi_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in manifests.items():
        frame.to_csv(args.output_dir / name, index=False)
        print(f"{name}: n={len(frame)}, benign={(frame.label == 0).sum()}, malignant={(frame.label == 1).sum()}")


if __name__ == "__main__":
    main()

