#Transformer Encoder (GraphCodeBERT)

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "microsoft/graphcodebert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

#dataset


class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, codes, labels=None):
        self.codes = codes
        self.labels = labels

    def __getitem__(self, idx):
        item = tokenizer(
            self.codes[idx],
            truncation=True,
            padding="max_length",
            max_length=256    # FAST, recommended for training
        )
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.codes)

  import os
os.environ["WANDB_DISABLED"] = "true"

# 1) Subsample for faster training
train_small = train.sample(50000, random_state=42).reset_index(drop=True)
val_small   = validation.sample(10000, random_state=42).reset_index(drop=True)

# 2) Build datasets from the small subsets
train_ds = CodeDataset(train_small["code"].tolist(), train_small["label"].tolist())
val_ds   = CodeDataset(val_small["code"].tolist(), val_small["label"].tolist())

from transformers import TrainingArguments, Trainer
import torch

# 3) Training arguments (fast + safe)
training_args = TrainingArguments(
    output_dir="./codebert",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=200,
    save_strategy="epoch",
    fp16=True if torch.cuda.is_available() else False,
)

# 4) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# 5) Train
trainer.train()


import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Fix the validation subset ONCE (do NOT resample later)
val_small = validation.sample(10000, random_state=42).reset_index(drop=True)

# Ground-truth labels (aligned with val_small)
y_val = val_small["label"].values

# 2) Build dataset from this exact subset (for CodeBERT)
val_ds = CodeDataset(
    val_small["code"].tolist(),
    val_small["label"].tolist()
)

# 3) Get logits from CodeBERT for this subset
pred_output = trainer.predict(val_ds)
logits = pred_output.predictions          # shape: (N, 2)

# 4) Convert logits → probabilities with softmax
#    probs[:, 0] = P(human), probs[:, 1] = P(AI)
probs = softmax(logits, axis=1)
codebert_prob = probs[:, 1]              # P(AI | code)

# 5) Convert probabilities → predicted labels (threshold 0.5)
codebert_pred_labels = (codebert_prob >= 0.5).astype(int)

# 6) Evaluation metrics
print("=== CodeBERT Validation Report ===")
print(classification_report(y_val, codebert_pred_labels, digits=4))

print("Macro F1:", f1_score(y_val, codebert_pred_labels, average="macro"))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_val, codebert_pred_labels))

# 7) Build dataframe for visualization (perfectly aligned)
val_df_viz = pd.DataFrame({
    "y_true": y_val,
    "codebert_prob": codebert_prob
})

