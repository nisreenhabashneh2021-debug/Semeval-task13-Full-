# task_a/src/eval_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import numpy as np
import pandas as pd

from common.metrics import compute_classification_metrics, classification_report_df
from common.plotting import plot_confusion_matrix, plot_metric_bars


def evaluate_and_log(
    y_true,
    y_pred,
    run_name: str,
    results_dir: str | Path,
):
    """
    Compute metrics, save JSON + plots, and return the metric dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_classification_metrics(y_true, y_pred)
    report_df = classification_report_df(y_true, y_pred)

    # Save metrics JSON
    with open(results_dir / f"{run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report
    report_df.to_csv(results_dir / f"{run_name}_classification_report.csv")

    # Plots
    plot_metric_bars(
        metrics,
        title=f"{run_name} metrics",
        save_path=str(results_dir / f"{run_name}_metrics.png"),
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        title=f"{run_name} confusion matrix",
        save_path=str(results_dir / f"{run_name}_cm.png"),
    )

    return metrics
