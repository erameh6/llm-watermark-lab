"""Statistical evaluation helpers for watermark detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def classification_metrics(labels, predictions, scores) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0:
        return {}
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    # Exclude sentinel scores used for texts too short to detect; keep them in
    # operational confusion-matrix metrics as undetected samples.
    scored = np.isfinite(scores) & (scores > -999999)
    auc = (float(roc_auc_score(labels[scored], scores[scored]))
           if len(np.unique(labels[scored])) == 2 else float("nan"))
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc_zscore": auc,
        "true_positive_rate": float(tp / (tp + fn)) if tp + fn else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def bootstrap_rate_ci(values, iterations: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap interval for a binary rate."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(iterations, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
