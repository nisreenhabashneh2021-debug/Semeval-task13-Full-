https://github.com/nisreenhabashneh2021-debug/Semeval-task13-Full-/invitations
import os
import subprocess
# task_x/experiments/run_ensemble.py

"""
Simple ensemble script.

Given multiple CSV submissions for the same task:

  results/submissions/task_b_submission_tfidf_svm.csv
  results/submissions/task_b_submission_bert-base-uncased.csv
  results/submissions/task_b_submission_microsoft_codebert-base.csv

…this script:
  - loads them
  - checks alignment on row_id
  - applies majority vote
  - writes a new ensemble CSV
"""

from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ---- CHANGE THIS PER FOLDER ----
TASK_NAME = "task_b"

# List of submission filenames (relative to submissions/ folder)
SUBMISSION_FILES = [
    f"{TASK_NAME}_submission_tfidf_svm.csv",
    f"{TASK_NAME}_submission_bert-base-uncased.csv",
    # f"{TASK_NAME}_submission_microsoft_codebert-base.csv",
]


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SUB_DIR = os.path.join(PROJECT_ROOT, TASK_NAME, "results", "submissions")
os.makedirs(SUB_DIR, exist_ok=True)


def main():
    print(f"=== Building ensemble for {TASK_NAME} ===")
    sub_dfs = []

    for fname in SUBMISSION_FILES:
        path = os.path.join(SUB_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing submission file: {path}")
        df = pd.read_csv(path).sort_values("row_id").reset_index(drop=True)
        sub_dfs.append(df)
        print(f"Loaded {fname} with shape {df.shape}")

    # Sanity checks
    num_models = len(sub_dfs)
    n = len(sub_dfs[0])
    for df in sub_dfs[1:]:
        assert len(df) == n, "All submissions must have same length."
        assert (df["row_id"].values == sub_dfs[0]["row_id"].values).all(), \
            "row_id mismatch across submissions."

    # Build matrix of shape (num_models, N)
    pred_matrix = np.vstack([df["label"].values for df in sub_dfs])
    # Majority vote along axis=0
    ensemble_labels = stats.mode(pred_matrix, axis=0, keepdims=False).mode

    ensemble_df = pd.DataFrame({
        "row_id": sub_dfs[0]["row_id"].values,
        "label": ensemble_labels,
    })

    out_name = f"{TASK_NAME}_submission_ensemble.csv"
    out_path = os.path.join(SUB_DIR, out_name)
    ensemble_df.to_csv(out_path, index=False)
    print(f"Saved ensemble submission to {out_path}")


if __name__ == "__main__":
    main()
