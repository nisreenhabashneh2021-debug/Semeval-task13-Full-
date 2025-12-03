
# task_a/experiments/run_ensemble.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from common.data_utils import load_parquet_splits, extract_xgb_features
from common.metrics import compute_classification_metrics
from common.plotting import plot_confusion_matrix, plot_metric_bars


DATA_DIR = Path("task_a/data/raw")
RESULTS_DIR = Path("task_a/results/logs")


def main():
    train, val, test = load_parquet_splits(DATA_DIR)

    # Load trained models from previous runs
    tfidf_model = joblib.load("task_a/results/tfidf_baseline_model.joblib")
    xgb_model = joblib.load("task_a/results/xgb_model.joblib")

    X_val_text = val["code"].astype(str)
    X_val_feats = extract_xgb_features(val, text_col="code")
    y_val = val["label"].astype(int).to_numpy()

    tfidf_proba = tfidf_model.predict_proba(X_val_text)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_val_feats)[:, 1]

    # Simple average
    ensemble_proba = 0.5 * tfidf_proba + 0.5 * xgb_proba
    y_pred = (ensemble_proba >= 0.5).astype(int)

    metrics = compute_classification_metrics(y_val, y_pred)
    print("Ensemble metrics:", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric_bars(
        metrics, title="Ensemble metrics", save_path=RESULTS_DIR / "ensemble_metrics.png"
    )
    plot_confusion_matrix(
        y_val,
        y_pred,
        title="Ensemble confusion matrix",
        save_path=RESULTS_DIR / "ensemble_cm.png",
    )


if __name__ == "__main__":
    main()


