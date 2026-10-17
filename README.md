# Cross-Dataset Breast Ultrasound Generalization

This repository implements the experiment in the CSCI 566 pre-proposal: train a binary benign-versus-malignant classifier on **BUS-BRA (Brazil)**, then evaluate the unchanged checkpoint on a held-out BUS-BRA test set and two external datasets: **BrEaST-Lesions-USG (Poland)** and **BUSI (Egypt)**.

This is a research benchmark, not a clinical diagnostic system.

## Experimental design

| Dataset | Use | Included classes | Masks used? |
|---|---|---|---|
| BUS-BRA | train, validation, in-domain test | benign / malignant | No |
| BrEaST-Lesions-USG | locked external test 1 | benign / malignant; normal excluded | No |
| BUSI | locked external test 2 | benign / malignant; normal excluded | No |

The external datasets must never be used for model selection, threshold selection, early stopping, or normalization-statistic fitting. The baseline converts grayscale scans to three channels and uses ImageNet normalization because the backbone is ImageNet-pretrained.

## Repository layout

```text
configs/baseline.yaml             experiment settings
src/bus_generalization/prepare.py dataset adapters and leakage-safe splitting
src/bus_generalization/data.py    PyTorch dataset and transforms
src/bus_generalization/models.py  ResNet-18 and EfficientNet-B0
src/bus_generalization/train.py   training and early stopping
src/bus_generalization/evaluate.py three-dataset evaluation and abstention
src/bus_generalization/metrics.py clinical classification metrics and CIs
tests/                             small unit tests
```

## 1. Install

```bash
cd breast-us-generalization
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## 2. Download and arrange data

Download each dataset from its official/provider page. Keep the raw files outside version control:

```text
data/raw/
  busbra/
    bus_data.csv
    ... images ...
  breast/
    BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx
    ... images and masks ...
  busi/
    benign/
    malignant/
    normal/
```

The adapters tolerate nested directories and several common filename-column spellings. They deliberately exclude any file containing `_mask` from BUSI. Run `bus-prepare --help` if your download uses different root paths.

## 3. Create manifests

```bash
bus-prepare \
  --busbra-root data/raw/busbra \
  --busbra-csv data/raw/busbra/bus_data.csv \
  --breast-root data/raw/breast \
  --breast-xlsx data/raw/breast/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx \
  --busi-root data/raw/busi
```

BUS-BRA is split approximately 70/10/20 using patient groups. The command stops if it cannot identify a patient column; inspect the metadata before considering `--allow-image-split`, because multiple scans from one patient must not cross splits.

Before training, verify the printed counts against the expected binary cohorts: BUS-BRA 1,875; BrEaST 252 cases (the number of image rows may differ if a case has multiple images); BUSI 647.

## 4. Train and evaluate

```bash
bus-train --config configs/baseline.yaml
bus-evaluate --config configs/baseline.yaml
```

Outputs are written under `outputs/baseline_resnet18/`: the best checkpoint, training history, per-image predictions, aggregate metrics, 95% bootstrap intervals, generalization gaps, and confidence-based selective evaluation.

## Recommended experiment matrix

Run at least three random seeds for each backbone:

1. ResNet-18 (primary baseline).
2. EfficientNet-B0 (architecture robustness check).
3. Optional ablation without ImageNet pretraining.

Report mean and standard deviation across seeds for ROC-AUC, balanced accuracy, sensitivity, specificity, and F1. Accuracy alone is insufficient because benign and malignant counts are imbalanced. Report external minus in-domain performance as the generalization gap. Treat abstention as a secondary analysis and show performance versus retained coverage, rather than only one favorable confidence cutoff.

## Reproducibility checklist

- Freeze external test sets before training.
- Split BUS-BRA by patient, not image.
- Do not feed masks or mask-derived crops into the baseline.
- Apply identical deterministic evaluation preprocessing to all datasets.
- Save the exact manifests used for every run.
- Fit any later calibration method only on BUS-BRA validation data.
- Document dataset licenses and required citations in the final report.

