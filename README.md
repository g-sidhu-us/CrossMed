# CrossMed: Cross-Dataset Medical Image Generalization

This repository implements the experiment in the CSCI 566 pre-proposal: train a binary benign-versus-malignant classifier on **BUS-BRA (Brazil)**, then evaluate the unchanged checkpoint on a held-out BUS-BRA test set and two external datasets: **BrEaST-Lesions-USG (Poland)** and **BUSI (Egypt)**.

This is a research benchmark, not a clinical diagnostic system.

## Experimental design

| Dataset | Use | Included classes | Masks used? |
|---|---|---|---|
| BUS-BRA | train, validation, in-domain test | benign / malignant | No |
| BrEaST-Lesions-USG | locked external test 1 | benign / malignant; normal excluded | No |
| BUSI | locked external test 2 | benign / malignant; normal excluded | No |

The external datasets must never be used for model selection, threshold selection, early stopping, or normalization-statistic fitting. The baseline converts grayscale scans to three channels and uses ImageNet normalization because the backbone is ImageNet-pretrained.

For BrEaST, `Image_filename` links each row to its scan, `CaseID` is retained as the patient identifier, and `Classification` provides the binary label. `Mask_tumor_filename` and `Mask_other_filename` are deliberately never passed to the baseline model. Clinical and sonographic fields are retained in the manifest for later subgroup analysis only.

## Repository layout

```text
configs/baseline.yaml             experiment settings
src/crossmed/prepare.py dataset adapters and leakage-safe splitting
src/crossmed/data.py    PyTorch dataset and transforms
src/crossmed/models.py  ResNet-18, EfficientNet-B0, and ViT-B/16
src/crossmed/train.py   training and early stopping
src/crossmed/evaluate.py three-dataset evaluation and abstention
src/crossmed/metrics.py clinical classification metrics and CIs
tests/                             small unit tests
notebooks/                         stepwise preparation, training, and evaluation
ARCHITECTURE.md                    model designs and comparison protocol
```

## 1. Install

```bash
cd crossmed
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

## 2. Team dataset setup

The datasets are not included in this repository. Each team member must download
the same three datasets independently and place them under the standardized
folder names `busbra`, `breast`, and `busi`. Do not rename individual image or
mask files.

Keep the raw files outside version control:

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

The exact image subfolders from the original downloads may remain nested under
these three dataset folders; the preparation code searches them recursively.
Every team member should therefore have the same project-relative data paths,
even when the absolute project location differs between local machines or CARC.

The repository `.gitignore` excludes `data/raw/`, so `git add` will not copy or
upload the medical images. Generated CSV manifests under `data/manifests/` use
paths relative to `data/` and can be shared through Git after they are verified.

The adapters tolerate nested directories and several common filename-column spellings. They deliberately exclude any file containing `_mask` from BUSI. Run `crossmed-prepare --help` if your download uses different root paths.

## 3. Create manifests

```bash
crossmed-prepare \
  --data-root data \
  --busbra-root data/raw/busbra \
  --busbra-metadata data/raw/busbra/bus_data.csv \
  --breast-root data/raw/breast \
  --breast-xlsx data/raw/breast/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx \
  --busi-root data/raw/busi
```

BUS-BRA is split approximately 70/10/20 using patient groups. The command stops if it cannot identify a patient column; inspect the metadata before considering `--allow-image-split`, because multiple scans from one patient must not cross splits.

Manifest image paths are stored relative to `data/` (for example,
`raw/busbra/images/example.png`) and resolved using `data.root` from the YAML
configuration. This makes manifests portable across teammates and CARC paths.

Before training, verify the printed counts against the expected binary cohorts: BUS-BRA 1,875; BrEaST 252 cases (the number of image rows may differ if a case has multiple images); BUSI 647.

## 4. Train and evaluate

```bash
crossmed-train --config configs/baseline.yaml
crossmed-evaluate --config configs/baseline.yaml
```

Outputs are written under `outputs/baseline_resnet18/`: the best checkpoint, training history, per-image predictions, aggregate metrics, 95% bootstrap intervals, generalization gaps, and confidence-based selective evaluation.

All primary experiments use a maximum of 100 epochs, a minimum of 40 epochs,
and early-stopping patience of 10 based on validation ROC-AUC. The best
checkpoint is tracked from epoch 1, but training cannot stop before completing
epoch 40.

## Recommended experiment matrix

The experiments are organized in stages so backbone training and classifier
depth are not changed at the same time.

### Stage 1: select a fully trainable ResNet

| Experiment | Backbone | Backbone training | Head |
|---|---|---|---|
| R18 fine-tuned | ResNet-18 | Fully trainable | `Dropout -> Linear(2)` |
| R50 fine-tuned | ResNet-50 | Fully trainable | `Dropout -> Linear(2)` |
| R101 fine-tuned | ResNet-101 | Fully trainable | `Dropout -> Linear(2)` |

Run these first. They isolate backbone depth while holding the head and training
strategy constant.

### Stage 2: isolate weight freezing

For each ResNet, compare its Stage 1 run against the corresponding frozen model
with the same simple head. The only intended changed variable is whether the
pretrained feature weights are updated.

### Stage 3: test classifier depth

Compare the simple head with the notebook MLP head (`features -> 256 -> 2`)
under both weight conditions. This completes a 2 x 2 experiment for each
ResNet: frozen/simple, frozen/MLP, fine-tuned/simple, and fine-tuned/MLP.

The complete registry is:

| Experiment | Backbone | Backbone training | Classifier head | Configuration |
|---|---|---|---|---|
| Baseline | ResNet-18 | Fully trainable | Dropout + Linear(2) | `baseline.yaml` |
| Fine-tuned R50 | ResNet-50 | Fully trainable | Simple | `resnet50_finetuned.yaml` |
| Fine-tuned R101 | ResNet-101 | Fully trainable | Simple | `resnet101_finetuned.yaml` |
| Fine-tuned-MLP R18 | ResNet-18 | Fully trainable | 256-unit MLP | `resnet18_finetuned_mlp.yaml` |
| Fine-tuned-MLP R50 | ResNet-50 | Fully trainable | 256-unit MLP | `resnet50_finetuned_mlp.yaml` |
| Fine-tuned-MLP R101 | ResNet-101 | Fully trainable | 256-unit MLP | `resnet101_finetuned_mlp.yaml` |
| Frozen-simple R18 | ResNet-18 | Frozen | Simple | `resnet18_frozen_simple.yaml` |
| Frozen-simple R50 | ResNet-50 | Frozen | Simple | `resnet50_frozen_simple.yaml` |
| Frozen-simple R101 | ResNet-101 | Frozen | Simple | `resnet101_frozen_simple.yaml` |
| Frozen-MLP R18/R50/R101 | ResNet family | Frozen | 256-unit MLP | existing frozen configs |
| Fine-tuned-simple ViT | ViT-B/16 | Fully trainable | Simple | `vit.yaml` |
| Fine-tuned-MLP ViT | ViT-B/16 | Fully trainable | 256-unit MLP | `vit_finetuned_mlp.yaml` |
| Frozen-simple ViT | ViT-B/16 | Frozen | Simple | `vit_frozen_simple.yaml` |
| Frozen-MLP ViT | ViT-B/16 | Frozen | 256-unit MLP | `vit_frozen.yaml` |

For each ResNet depth, the controlled comparison is:

| Feature weights | Simple head | 256-unit MLP head |
|---|---|---|
| Frozen | Yes | Yes |
| Fully trainable | Yes | Yes |

The same four-condition comparison is used for ViT-B/16. For ViT, “feature
weights” means the patch projection, positional embedding, CLS token processing,
and Transformer encoder weights.

Run at least three seeds for the final selected experiments. The frozen
ResNet sequence tests whether larger pretrained feature extractors generalize
better when only the classifier is learned. The frozen ViT experiment uses its
768-dimensional CLS representation rather than global average pooling.

Run the complete matrix one configuration at a time:

```bash
crossmed-train --config configs/baseline.yaml
crossmed-evaluate --config configs/baseline.yaml

crossmed-train --config configs/resnet50_finetuned.yaml
crossmed-evaluate --config configs/resnet50_finetuned.yaml

crossmed-train --config configs/resnet101_finetuned.yaml
crossmed-evaluate --config configs/resnet101_finetuned.yaml
```

For easier execution, all runs are registered in
`configs/experiment_matrix.yaml`. Run the complete suite with:

```bash
crossmed-run
```

Run only selected experiments:

```bash
crossmed-run --experiments resnet18_finetuned resnet50_finetuned resnet101_finetuned
```

Run training and evaluation separately:

```bash
crossmed-run --stage train --experiments resnet50_finetuned
crossmed-run --stage evaluate --experiments resnet50_finetuned
```

Existing checkpoints and evaluation files are skipped automatically. Use
`--force` only when a completed run should be replaced. Every evaluation pass
updates `results/model_comparison.csv`, which can be loaded directly for the
report or presentation.

The training stage returns a checkpoint-status table, not performance metrics.
Metrics appear only after evaluation, preventing older `evaluation.json` files
from being mistaken for the model that was just trained.

Each training run saves its best validation checkpoint to the experiment's
`best.pt`. Saved checkpoints can be evaluated later without retraining:

```bash
crossmed-run --stage evaluate --experiments resnet101_finetuned
```

For a local machine that runs several models sequentially, enable explicit GPU
cleanup between models:

```bash
crossmed-run --cleanup-between-runs
```

This deletes unreferenced objects, runs garbage collection, and clears cached
CUDA allocations before loading the next model. It is optional because CARC
should normally run one experiment per independent job.

Checkpoint behavior is controlled explicitly:

- Normal run: an existing `best.pt` is treated as a completed training result
  and is not retrained.
- `--stage evaluate`: load the existing checkpoint and produce test results.
- During list evaluation, experiments without `best.pt` are reported and
  skipped; trained experiments continue. Add `--strict-missing` to fail instead.
- `--force`: start fresh training and replace that experiment's checkpoint and
  results. Use this after intentionally changing its configuration.
- For independent repetitions, change the seed and output directory so earlier
  runs remain available instead of being overwritten.

## Classifier architectures

### Shared preprocessing

![Shared image preprocessing pipeline](docs/images/preprocessing_pipeline.svg)

This preprocessing occurs before every architecture. The augmentation block is
used only during training; validation and all test datasets bypass it.

### Fully trainable baseline

![Fully trainable ResNet-18 baseline](docs/images/resnet18_finetuned.svg)

CrossMed keeps the existing simple head as the primary baseline:

```text
trainable pretrained backbone -> Dropout(0.2) -> Linear(2)
```

It also reproduces the architecture from the supplied project notebook:

### Frozen ResNet experiments

![Frozen ResNet with MLP classifier](docs/images/resnet_frozen_mlp.svg)

```text
frozen pretrained CNN -> global average pooling -> Linear(256) -> ReLU
-> BatchNorm -> Dropout(0.2) -> Linear(2)
```

Run the notebook-style ResNet-50 architecture with:

```bash
crossmed-train --config configs/notebook_architecture.yaml
crossmed-evaluate --config configs/notebook_architecture.yaml
```

The notebook's VGG16, ResNet50, ResNet101, and EfficientNetB0 backbones are
available as `vgg16`, `resnet50`, `resnet101`, and `efficientnet_b0`. See
`ARCHITECTURE.md` for layer details, PyTorch/Keras equivalences, and the fair
head-ablation protocol.

### Frozen ViT experiment

![Frozen ViT-B/16 with MLP classifier](docs/images/vit_frozen_mlp.svg)

An additional fully fine-tuned ViT configuration remains available as an
optional extension:

```bash
crossmed-train --config configs/vit.yaml
crossmed-evaluate --config configs/vit.yaml
```

## Notebook workflow

Install Jupyter support and register the environment:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name crossmed --display-name "Python (CrossMed)"
jupyter lab
```

Open the three notebooks in numerical order:

1. `01_prepare_data.ipynb` creates and audits manifests.
2. `02_train_models.ipynb` selects and trains one or more experiments.
3. `03_evaluate_models.ipynb` loads saved checkpoints and compares results.

They call the same public functions as the CLI:

```python
from crossmed.prepare import prepare_all
from crossmed.train import train_experiment
from crossmed.evaluate import evaluate_experiment
from crossmed.experiments import run_experiment_matrix, aggregate_results
```

Edit only the path cell in notebook 01. The generated manifests and YAML configurations are then shared by the CLI and all notebooks.

The ViT configuration starts at batch size 8 because it uses substantially more GPU memory than ResNet-18. If CUDA runs out of memory, reduce `train_batch_size` to 4 or 2. Automatic mixed precision is already enabled. ViT-B/16 expects 224 x 224 inputs, matching this configuration.

Report mean and standard deviation across seeds for ROC-AUC, balanced accuracy, sensitivity, specificity, and F1. Accuracy alone is insufficient because benign and malignant counts are imbalanced. Report external minus in-domain performance as the generalization gap. Treat abstention as a secondary analysis and show performance versus retained coverage, rather than only one favorable confidence cutoff.

## Reproducibility checklist

- Freeze external test sets before training.
- Split BUS-BRA by patient, not image.
- Do not feed masks or mask-derived crops into the baseline.
- Apply identical deterministic evaluation preprocessing to all datasets.
- Save the exact manifests used for every run.
- Fit any later calibration method only on BUS-BRA validation data.
- Document dataset licenses and required citations in the final report.

