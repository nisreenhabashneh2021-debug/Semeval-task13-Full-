python task_b/experiments/run_transformer.py
# task_B/experiments/run_transformer.py

"""
Train a transformer model (e.g., BERT or CodeBERT) for a given subtask.

Outputs:
  - validation metrics in results/logs/
  - test predictions CSV in results/submissions/

This script is intentionally simple: no fancy sampling or scheduling.
"""

from __future__ import annotations
import os
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ---- CHANGE THIS PER FOLDER ----
TASK_NAME  = "task_b"             # "task_a", "task_b", or "task_c"
MODEL_NAME = "bert-base-uncased"  # or "microsoft/codebert-base", "Salesforce/codet5-small", ...


# -------------------------------------------------------------------
# Imports from project
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.data_utils import load_splits
from common.metrics import compute_basic_metrics, full_classification_report


RESULTS_DIR = os.path.join(PROJECT_ROOT, TASK_NAME, "results")
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
SUB_DIR     = os.path.join(RESULTS_DIR, "submissions")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Dataset wrapper
# -------------------------------------------------------------------
@dataclass
class CodeDataset(Dataset):
    df: pd.DataFrame
    tokenizer: AutoTokenizer
    text_col: str = "code"
    label_col: str | None = "label"
    max_length: int = 256

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx][self.text_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.label_col is not None:
            item["labels"] = torch.tensor(int(self.df.iloc[idx][self.label_col]))
        return item


def main():
    print(f"=== Transformer training for {TASK_NAME} using {MODEL_NAME} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load data
    train_df, val_df, test_df = load_splits(TASK_NAME, base_dir=PROJECT_ROOT)
    num_labels = train_df["label"].nunique()
    print("Num labels:", num_labels)

    # 2) Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    ).to(device)

    train_ds = CodeDataset(train_df, tokenizer)
    val_ds   = CodeDataset(val_df, tokenizer)
    test_ds  = CodeDataset(test_df, tokenizer, label_col=None)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) TrainingArguments (tweak as needed)
    out_dir = os.path.join(PROJECT_ROOT, TASK_NAME, f"{MODEL_NAME.replace('/', '_')}_ckpt")
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )

    # 4) Metric function for Trainer
    from sklearn.metrics import f1_score, accuracy_score

    def compute_metrics_trainer(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_trainer,
    )

    # 5) Train
    trainer.train()

    # 6) Evaluate on validation with our helper metrics
    val_output = trainer.predict(val_ds)
    val_logits = val_output.predictions
    val_labels = val_output.label_ids
    val_pred   = np.argmax(val_logits, axis=-1)

    metrics_macro = compute_basic_metrics(val_labels, val_pred, average="macro")
    metrics_weighted = compute_basic_metrics(val_labels, val_pred, average="weighted")
    report = full_classification_report(val_labels, val_pred, digits=3)

    metrics_all = {"macro": metrics_macro, "weighted": metrics_weighted}
    metrics_path = os.path.join(LOG_DIR, f"{TASK_NAME}_transformer_{MODEL_NAME.replace('/', '_')}_metrics.json")
    report_path  = os.path.join(LOG_DIR, f"{TASK_NAME}_transformer_{MODEL_NAME.replace('/', '_')}_report.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report  to {report_path}")

    # 7) Predict on test
    test_output = trainer.predict(test_ds)
    test_logits = test_output.predictions
    test_pred   = np.argmax(test_logits, axis=-1)

    sub_df = pd.DataFrame({
        "row_id": np.arange(len(test_df)),
        "label": test_pred,
    })
    sub_path = os.path.join(SUB_DIR, f"{TASK_NAME}_submission_{MODEL_NAME.replace('/', '_')}.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")


if __name__ == "__main__":
    main()
