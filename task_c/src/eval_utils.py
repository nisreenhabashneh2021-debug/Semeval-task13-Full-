# task_c/src/eval_utils.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

IGNORE_ID = -100


def evaluate_token_model(trainer, val_df, tokenizer, max_length=512):
    """
    Returns dictionary of metrics + prints a token-level report.
    """

    val_ds = trainer.eval_dataset
    preds_output = trainer.predict(val_ds)

    logits = preds_output.predictions
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=-1)

    mask = labels != IGNORE_ID
    true_vals = labels[mask]
    pred_vals = preds[mask]

    metrics = {
        "accuracy": accuracy_score(true_vals, pred_vals),
        "macro_f1": f1_score(true_vals, pred_vals, average="macro"),
        "report": classification_report(true_vals, pred_vals),
    }

    print("\n=== Task C Evaluation Report ===")
    print(metrics["report"])
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])

    return metrics
