# task_b/src/train_utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from common.metrics import compute_classification_metrics
from .labels import NUM_LABELS
from .models import (
    TfidfSVMModel,
    TfidfSVMConfig,
    build_transformer_model,
)


# ===================== Dataset for transformers =====================


class CodeDataset(Dataset):
    """
    Simple dataset for 'code' column + integer labels 0..10.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        df = df.copy()
        df["label"] = df["label"].astype(int)

        self.codes = df["code"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx: int):
        code = self.codes[idx]
        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ====================== TF-IDF + SVM training =======================


def train_tfidf_svm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Optional[TfidfSVMConfig] = None,
):
    """
    Train TF-IDF + LinearSVC baseline and return model + val metrics.
    """
    model = TfidfSVMModel(cfg)

    X_train = train_df["code"].astype(str)
    y_train = train_df["label"].astype(int)

    X_val = val_df["code"].astype(str)
    y_val = val_df["label"].astype(int)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = compute_classification_metrics(y_val, y_pred, average="macro")
    return model, metrics


# ===================== Transformer training =========================


@dataclass
class TransformerTrainConfig:
    model_name: str = "Salesforce/codet5-small"   # same as notebook default
    max_length: int = 256
    num_labels: int = NUM_LABELS
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    logging_steps: int = 100
    output_dir: str = "task_b/results/transformer_codet5"
    seed: int = 42


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Optional[TransformerTrainConfig] = None,
):
    """
    Train a multi-class transformer (e.g., CodeT5-small) on Task B.
    """
    cfg = cfg or TransformerTrainConfig()

    model, tokenizer = build_transformer_model(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )

    train_dataset = CodeDataset(train_df, tokenizer, max_length=cfg.max_length)
    val_dataset = CodeDataset(val_df, tokenizer, max_length=cfg.max_length)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=cfg.seed,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return compute_classification_metrics(labels, preds, average="macro")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, tokenizer
