# common/data_utils.py

"""
Utility functions for loading data and preparing features
for all SemEval-2026 Task 13 subtasks (A, B, C).

Assumes the following folder layout:

semeval-2026-task13-full/
    task_a/
        data/
            raw/
                task_a_train.parquet
                task_a_val.parquet
                task_a_test.parquet
    task_b/
        data/raw/...
    task_c/
        data/raw/...

You can change file names in `get_default_paths` if needed.
"""

from __future__ import annotations
import os
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def get_default_paths(task_name: str, base_dir: str = ".") -> Tuple[str, str, str]:
    """
    Build default train/val/test parquet paths for a given task.

    Example:
        get_default_paths("task_b") ->
            ("task_b/data/raw/task_b_train.parquet",
             "task_b/data/raw/task_b_val.parquet",
             "task_b/data/raw/task_b_test.parquet")
    """
    task_name = task_name.lower()
    raw_dir = os.path.join(base_dir, task_name, "data", "raw")

    train_path = os.path.join(raw_dir, f"{task_name}_train.parquet")
    val_path   = os.path.join(raw_dir, f"{task_name}_val.parquet")
    test_path  = os.path.join(raw_dir, f"{task_name}_test.parquet")

    return train_path, val_path, test_path


def load_parquet(path: str) -> pd.DataFrame:
    """Tiny wrapper around pandas.read_parquet with a nicer error."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def load_splits(
    task_name: str,
    base_dir: str = ".",
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train / val / test DataFrames for a task.

    If paths are not given, use the default pattern from get_default_paths.
    """
    if train_path is None or val_path is None or test_path is None:
        train_path_d, val_path_d, test_path_d = get_default_paths(task_name, base_dir)
        train_path = train_path or train_path_d
        val_path   = val_path   or val_path_d
        test_path  = test_path  or test_path_d

    train_df = load_parquet(train_path)
    val_df   = load_parquet(val_path)
    test_df  = load_parquet(test_path)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------

def get_text_and_labels(
    df: pd.DataFrame,
    text_col: str = "code",
    label_col: str = "label",
):
    """
    Return numpy arrays (X, y) from a DataFrame.

    X: array of strings (code)
    y: array of labels (ints)
    """
    X = df[text_col].astype(str).values
    y = df[label_col].values
    return X, y


def train_val_split(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.1,
    random_state: int = 42,
):
    """
    Quick train/val split from a single DataFrame (if you only have train.parquet).

    Returns:
        train_df, val_df
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
