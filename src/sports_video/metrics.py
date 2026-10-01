from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import LABELS


def head_metrics(y_true, y_pred, label_name: str):
    names = LABELS[label_name]
    valid = np.asarray(y_true) >= 0
    if not valid.any():
        return {"n": 0, "status": "no verified labels"}
    true = np.asarray(y_true)[valid]
    pred = np.asarray(y_pred)[valid]
    ids = list(range(len(names)))
    report = classification_report(true, pred, labels=ids, target_names=names,
                                   output_dict=True, zero_division=0)
    return {
        "n": int(valid.sum()),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": {name: report[name] for name in names},
        "confusion_matrix": confusion_matrix(true, pred, labels=ids).tolist(),
    }
