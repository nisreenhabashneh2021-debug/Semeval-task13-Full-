# task_c/src/inference.py

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

IGNORE_ID = -100


def predict_code_origin(model_path, df, max_length=512):
    """
    Inference for Subtask C (token classification).
    Returns token predictions per sequence.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    all_pred_tokens = []

    for code in df["code"].astype(str):
        enc = tokenizer(
            code,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        with torch.no_grad():
            logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits

        preds = logits.argmax(dim=-1).numpy()[0]
        offsets = enc["offset_mapping"][0].numpy()

        token_preds = []
        for pred, (s, e) in zip(preds, offsets):
            if s == 0 and e == 0:
                continue
            token_preds.append((s, e, int(pred)))

        all_pred_tokens.append(token_preds)

    return all_pred_tokens
