import numpy as np
from dataclasses import dataclass


@dataclass
class OutlierThresholds:
    low_signal_std: float = 10.0
    artifact_mean: float = 100.0
    artifact_std: float = 15.0
    artifact_relative_std: float = 0.10


def classify_set(y_set, thresholds=None):
    if thresholds is None:
        thresholds = OutlierThresholds()

    std_total = float(np.std(y_set))
    if std_total < thresholds.low_signal_std:
        return "low_signal"

    for c in range(y_set.shape[1]):
        col = y_set[:, c]
        m = abs(float(np.mean(col)))
        s = float(np.std(col))
        if m > thresholds.artifact_mean and s < thresholds.artifact_std:
            return "fem_artifact"
        if m > thresholds.artifact_mean and (s / max(m, 1e-12)) < thresholds.artifact_relative_std:
            return "fem_artifact"

    return "normal"


def classify_holdout(y_hold_per_set, thresholds=None):
    n_sets = y_hold_per_set.shape[0]
    labels = np.empty(n_sets, dtype=object)
    for i in range(n_sets):
        labels[i] = classify_set(y_hold_per_set[i], thresholds)
    return labels


def split_by_category(predictions, y_true, labels):
    cats = {"normal": [], "low_signal": [], "fem_artifact": []}
    for i, lab in enumerate(labels):
        cats[lab].append(i)
    out = {}
    for cat, idx in cats.items():
        if not idx:
            out[cat] = (np.empty((0, *predictions.shape[1:])),
                        np.empty((0, *y_true.shape[1:])), np.array([]))
        else:
            idx = np.array(idx)
            out[cat] = (predictions[idx], y_true[idx], idx)
    return out
