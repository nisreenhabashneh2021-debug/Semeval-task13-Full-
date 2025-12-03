# task_c/experiments/run_transformer.py

from pathlib import Path
import json
import pandas as pd

from task_c.src.train_utils import TransformerTrainConfig, train_token_classifier
from task_c.src.eval_utils import evaluate_token_model


def main():
    data_dir = Path("task_c/data/raw")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")

    cfg = TransformerTrainConfig(
        model_name="microsoft/graphcodebert-base",
        max_length=512,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        output_dir="task_c/results/transformer",
    )

    trainer, tokenizer = train_token_classifier(train_df, val_df, cfg)

    metrics = evaluate_token_model(trainer, val_df, tokenizer)

    # Save model
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save metrics
    logs_dir = Path("task_c/results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(logs_dir / "task_c_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
