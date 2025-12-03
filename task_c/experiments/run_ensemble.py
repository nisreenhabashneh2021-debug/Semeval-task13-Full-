# task_c/experiments/run_ensemble.py
#
# Example ensemble: average logits of 2 trained models.
# Useful for boosting token-level F1.

import torch
import numpy as np
from pathlib import Path
import pandas as pd

from task_c.src.inference import predict_code_origin

def main():
    data_dir = Path("task_c/data/raw")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    model_paths = [
        "task_c/results/transformer",            # model 1
        "task_c/results/transformer_alt"         # model 2 (if you train twice)
    ]

    all_preds = []

    for mp in model_paths:
        preds = predict_code_origin(mp, test_df)
        all_preds.append(preds)

    # Simple ensemble: majority vote per token index
    final = []
    for seq_preds in zip(*all_preds):
        out = [p for p in seq_preds]
        # majority vote at each token index
        # flatten all predictions
        ensemble_seq = []
        for i in range(len(out[0])):
            votes = [seq[i][2] for seq in out]
            winner = max(set(votes), key=votes.count)
            ensemble_seq.append((out[0][i][0], out[0][i][1], winner))
        final.append(ensemble_seq)

    print("Ensemble complete. Tokens per seq:", len(final[0]))


if __name__ == "__main__":
    main()
