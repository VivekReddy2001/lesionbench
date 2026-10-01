"""Evaluation metrics that stay honest under class imbalance.

Plain accuracy rewards a model for predicting the majority class (nevi are
67% of HAM10000), so the headline metrics here are balanced accuracy (mean
per-class recall) and macro-F1. Expected calibration error (ECE) measures
whether a predicted probability of 0.8 is right about 80% of the time.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def evaluate(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(axis=1)
    classes = list(range(n_classes))
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", labels=classes, zero_division=0)),
        "ece": expected_calibration_error(probs, labels),
        "per_class_recall": recall_score(labels, pred, average=None, labels=classes, zero_division=0).tolist(),
    }
    try:
        out["macro_auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=classes))
    except ValueError:  # a class missing from ``labels``
        out["macro_auroc"] = float("nan")
    return out
