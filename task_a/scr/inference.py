# task_a/src/inference.py

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from common.data_utils import extract_xgb_features
from .models import TfidfLogRegModel, XGBoostModel


def _predict_transformer_logits(
    model,
    tokenizer,
    df: pd.DataFrame,
    max_length: int = 256,
) -> np.ndarray:
    model.eval()
    all_logits = []
    device = next(model.parameters()).device

    for text in df["code"].astype(str).tolist():
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        all_logits.append(logits[0])
    return np.stack(all_logits, axis=0)


def generate_submission_from_model(
    model_type: Literal["tfidf", "xgb", "transformer"],
    model,
    test_df: pd.DataFrame,
    output_csv: str | Path,
    tokenizer=None,
    max_length: int = 256,
):
    """
    Given a trained model and test DataFrame with 'code' column, write a
    submission.csv with columns ['row_id', 'label'] for Subtask A.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "tfidf":
        y_pred = model.predict(test_df["code"].astype(str))
    elif model_type == "xgb":
        X_test = extract_xgb_features(test_df, text_col="code")
        y_pred = model.predict(X_test)
    elif model_type == "transformer":
        logits = _predict_transformer_logits(
            model,
            tokenizer,
            test_df,
            max_length=max_length,
        )
        y_pred = logits.argmax(axis=-1)
    else:
        raise ValueError(f"Unknown model_type={model_type}")

    sub = pd.DataFrame(
        {
            "row_id": np.arange(len(test_df)),
            "label": y_pred.astype(int),
        }
    )
    sub.to_csv(output_csv, index=False)
    return sub
