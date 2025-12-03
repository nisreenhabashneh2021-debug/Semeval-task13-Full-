# task_a/src/train_utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    Trainer,
    TrainingArguments,
)

from common.data_utils import extract_xgb_features
from common.metrics import compute_classification_metrics
from .models import (
    TfidfLogRegModel,
    TfidfLogRegConfig,
    XGBoostModel,
    XGBConfig,
    build_transformer_model,
)


# ---------------------- Transformer dataset ----------------------


class CodeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.codes = df["code"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, idx: int):
        text = self.codes[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -------------------------- Baselines ----------------------------


def train_tfidf_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Optional[TfidfLogRegConfig] = None,
):
    model = TfidfLogRegModel(cfg)

    X_train = train_df["code"].astype(str)
    y_train = train_df["label"].astype(int)

    X_val = val_df["code"].astype(str)
    y_val = val_df["label"].astype(int)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = compute_classification_metrics(y_val, y_pred)
    return model, metrics


def train_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Optional[XGBConfig] = None,
):
    model = XGBoostModel(cfg)

    X_train = extract_xgb_features(train_df, text_col="code")
    y_train = train_df["label"].astype(int).to_numpy()

    X_val = extract_xgb_features(val_df, text_col="code")
    y_val = val_df["label"].astype(int).to_numpy()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = compute_classification_metrics(y_val, y_pred)
    return model, metrics


# ---------------------- Transformer training ---------------------


@dataclass
class TransformerTrainConfig:
    model_name: str = "microsoft/graphcodebert-base"
    max_length: int = 256
    num_labels: int = 2
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    logging_steps: int = 100
    output_dir: str = "task_a/results/transformer"
    seed: int = 42


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Optional[TransformerTrainConfig] = None,
):
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
        return compute_classification_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, tokenizer

