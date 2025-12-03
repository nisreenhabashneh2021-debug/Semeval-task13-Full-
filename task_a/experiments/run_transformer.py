# task_a/experiments/run_transformer.py

from __future__ import annotations

from pathlib import Path

from common.data_utils import load_parquet_splits
from task_a.src.train_utils import TransformerTrainConfig, train_transformer
from task_a.src.inference import generate_submission_from_model


DATA_DIR = Path("task_a/data/raw")
SUBMISSION_DIR = Path("task_a/results/submissions")


def main():
    train, val, test = load_parquet_splits(DATA_DIR)

    cfg = TransformerTrainConfig(
        model_name="microsoft/graphcodebert-base",
        num_labels=2,
        output_dir="task_a/results/transformer_graphcodebert",
        num_train_epochs=2,
    )

    trainer, tokenizer = train_transformer(train, val, cfg)

    # Save final model
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Generate submission on test set
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    generate_submission_from_model(
        model_type="transformer",
        model=trainer.model,
        tokenizer=tokenizer,
        test_df=test,
        max_length=cfg.max_length,
        output_csv=SUBMISSION_DIR / "submission_transformer_graphcodebert.csv",
    )


if __name__ == "__main__":
    main()
