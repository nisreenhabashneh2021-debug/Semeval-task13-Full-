# ASK_NAME = "task_c"
from task_c.src.models import build_tfidf_svm

"""
Train a classical TF–IDF baseline and save:
  - validation metrics to results/logs/
  - test predictions to results/submissions/

Works for any of {task_a, task_b, task_c} by changing TASK_NAME.
"""

from __future__ import annotations
import os
import json

import numpy as np
import pandas as pd

# ---- CHANGE THIS PER FOLDER ----
TASK_NAME = "task_c"   # "task_a", "task_b", or "task_c"

# -------------------------------------------------------------------
# Make project root importable (so we can import common.*)
# -------------------------------------------------------------------
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.data_utils import load_splits, get_text_and_labels
from common.metrics import compute_basic_metrics, full_classification_report
from task_c.src.models import build_tfidf_svm  # <- change to task_a/src/... if needed
# For a pure template, you can also dynamically import, but this is simpler.

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, TASK_NAME, "results")
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
SUB_DIR     = os.path.join(RESULTS_DIR, "submissions")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)


def main():
    print(f"=== Running TF–IDF baseline for {TASK_NAME} ===")

    # 1) Load data
    train_df, val_df, test_df = load_splits(TASK_NAME, base_dir=PROJECT_ROOT)
    print("Train shape:", train_df.shape)
    print("Val   shape:", val_df.shape)
    print("Test  shape:", test_df.shape)

    X_train, y_train = get_text_and_labels(train_df)
    X_val,   y_val   = get_text_and_labels(val_df)

    # 2) Build model (here: SVM; you can swap to build_tfidf_logreg())
    clf = build_tfidf_svm()
    print("Fitting TF–IDF + LinearSVC...")
    clf.fit(X_train, y_train)

    # 3) Evaluate on validation
    val_pred = clf.predict(X_val)

    metrics_macro = compute_basic_metrics(y_val, val_pred, average="macro")
    metrics_weighted = compute_basic_metrics(y_val, val_pred, average="weighted")

    print("\n=== Validation metrics (macro) ===")
    for k, v in metrics_macro.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Validation metrics (weighted) ===")
    for k, v in metrics_weighted.items():
        print(f"{k}: {v:.4f}")

    report = full_classification_report(y_val, val_pred, digits=3)
    print("\n=== Classification report ===")
    print(report)

    # 4) Save metrics & report
    metrics_all = {
        "macro": metrics_macro,
        "weighted": metrics_weighted,
    }

    metrics_path = os.path.join(LOG_DIR, f"{TASK_NAME}_tfidf_baseline_metrics.json")
    report_path  = os.path.join(LOG_DIR, f"{TASK_NAME}_tfidf_baseline_report.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved report  to {report_path}")

    # 5) Predict on test and save submission
    X_test = test_df["code"].astype(str).values
    test_pred = clf.predict(X_test)

    sub_df = pd.DataFrame({
        "row_id": np.arange(len(test_df)),
        "label": test_pred,
    })

    sub_path = os.path.join(SUB_DIR, f"{TASK_NAME}_submission_tfidf_svm.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")


if __name__ == "__main__":
    main()
