from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probability, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    result = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(recall_score(y_true, prediction, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    return result


def bootstrap_ci(y_true, probability, threshold=0.5, samples=1000, seed=566):
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("balanced_accuracy", "f1", "roc_auc")}
    for _ in range(samples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        score = classification_metrics(y_true[idx], probability[idx], threshold)
        for name in values:
            values[name].append(score[name])
    return {
        name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for name, v in values.items() if v
    }


def selective_metrics(y_true, probability, coverages, threshold=0.5):
    confidence = np.maximum(probability, 1.0 - probability)
    rows = []
    for coverage in coverages:
        keep = max(1, int(np.ceil(len(y_true) * coverage)))
        idx = np.argsort(-confidence)[:keep]
        row = classification_metrics(np.asarray(y_true)[idx], np.asarray(probability)[idx], threshold)
        row.update({"coverage": float(coverage), "abstained": int(len(y_true) - keep)})
        rows.append(row)
    return rows

