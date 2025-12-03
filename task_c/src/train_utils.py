# task_c/src/train_utils.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification

from .models import build_token_classifier


IGNORE_ID = -100


class CodeOriginDataset(Dataset):
    """
    Converts each row with a single label into token-level labels:
    first subword gets the label, the rest get -100.
    """

    def __init__(self, df, tokenizer, max_length=512):
        self.codes = df["code"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = int(self.labels[idx])

        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        word_ids = enc.word_ids()
        previous_word = None
        token_labels = []

        for w in word_ids:
            if w is None:
                token_labels.append(IGNORE_ID)
            elif w != previous_word:
                token_labels.append(label)
            else:
                token_labels.append(IGNORE_ID)
            previous_word = w

        enc.pop("offset_mapping")
        enc["labels"] = torch.tensor(token_labels)
        return enc


@dataclass
class TransformerTrainConfig:
    model_name: str = "microsoft/graphcodebert-base"
    max_length: int = 512
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    output_dir: str = "task_c/results/transformer"
    seed: int = 42


def train_token_classifier(train_df, val_df, cfg: TransformerTrainConfig):
    """
    Full training pipeline for token-based code origin detection.
    """

    num_labels = train_df["label"].nunique()
    model, tokenizer = build_token_classifier(cfg.model_name, num_labels)

    train_ds = CodeOriginDataset(train_df, tokenizer, cfg.max_length)
    val_ds = CodeOriginDataset(val_df, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        logging_steps=100,
        metric_for_best_model="eval_loss",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(pred_obj):
        logits = pred_obj.predictions
        labels = pred_obj.label_ids

        preds = np.argmax(logits, axis=-1)

        mask = labels != IGNORE_ID
        true_vals = labels[mask]
        pred_vals = preds[mask]

        if len(true_vals) == 0:
            return {"f1": 0, "acc": 0}

        from sklearn.metrics import accuracy_score, f1_score

        return {
            "acc": accuracy_score(true_vals, pred_vals),
            "f1": f1_score(true_vals, pred_vals, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, tokenizer
