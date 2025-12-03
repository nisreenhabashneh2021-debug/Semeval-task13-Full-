#!/usr/bin/env python3
"""
evaluate_all.py

Collects metrics from Task A, B, C (baselines + transformers + ensembles)
and prints a single comparison table.

Expected metric JSON files (use whatever subset you actually have):

Task A:
    task_a/results/logs/tfidf_baseline_metrics.json
    task_a/results/logs/xgb_baseline_metrics.json            (optional)
    task_a/results/logs/transformer_graphcodebert_metrics.json
    task_a/results/logs/ensemble_metrics.json                (optional)

Task B:
    task_b/results/logs/task_b_tfidf_svm_metrics.json
    task_b/results/logs/transformer_codebert_metrics.json    (or codet5)

Task C:
    task_c/results/logs/task_c_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_metrics(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    rows = []

    # -------- Task A --------
    task = "Task A"
    base = Path("task_a/results/logs")

    files_a = {
        "TFIDF-LogReg": base / "tfidf_baseline_metrics.json",
        "XGBoost": base / "xgb_baseline_metrics.json",
        "Transformer-GraphCodeBERT": base / "transformer_graphcodebert_metrics.json",
        "Ensemble": base / "ensemble_metrics.json",
    }

    for name, p in files_a.items():
        metrics = load_metrics(p)
        if metrics is None:
            continue
        rows.append(
            {
                "task": task,
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
            }
        )

    # -------- Task B --------
    task = "Task B"
    base = Path("task_b/results/logs")

    files_b = {
        "TFIDF+SVM": base / "task_b_tfidf_svm_metrics.json",
        "Transformer-CodeBERT": base / "transformer_codebert_metrics.json",
        "Transformer-CodeT5": base / "transformer_codet5_metrics.json",
    }

    for name, p in files_b.items():
        metrics = load_metrics(p)
        if metrics is None:
            continue
        rows.append(
            {
                "task": task,
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
            }
        )

    # -------- Task C --------
    task = "Task C"
    base = Path("task_c/results/logs")

    files_c = {
        "Token-Transformer": base / "task_c_metrics.json",
    }

    for name, p in files_c.items():
        metrics = load_metrics(p)
        if metrics is None:
            continue
        rows.append(
            {
                "task": task,
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "macro_precision": None,
                "macro_recall": None,
            }
        )

    # ---------- Summary table ----------
    if not rows:
        print("No metrics JSON files found. Run the experiment scripts first.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["task", "macro_f1"], ascending=[True, False])

    print("\n================ Model Comparison ================")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if x is not None else "NA"))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "all_tasks_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Saved combined metrics to {out_path}")
    

if __name__ == "__main__":
    main()
