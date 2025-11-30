# common/metrics.py

"""
Metrics and reporting utilities shared across subtasks.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_basic_metrics(
    y_true,
    y_pred,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute accuracy + Precision/Recall/F1 with a given averaging scheme.

    average = "macro"  -> all classes equal (good for SemEval leaderboard)
    average = "weighted" -> accounts for class imbalance
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {
        "accuracy": acc,
        f"{average}_precision": prec,
        f"{average}_recall": rec,
        f"{average}_f1": f1,
    }


def full_classification_report(
    y_true,
    y_pred,
    target_names: Optional[List[str]] = None,
    digits: int = 3,
) -> str:
    """
    Wrapper around sklearn.classification_report that returns a string.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=digits,
        zero_division=0,
    )


def confusion_matrix_array(y_true, y_pred) -> np.ndarray:
    """Return the raw confusion-matrix array."""
    return confusion_matrix(y_true, y_pred)
