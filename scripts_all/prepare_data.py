#!/usr/bin/env python3
"""
prepare_data.py

Copies the original SemEval Task 13 parquet files into the repo
layout we are using:

    task_a/data/raw/*.parquet
    task_b/data/raw/*.parquet
    task_c/data/raw/*.parquet

You only need to adjust DATA_ROOT to match your local / Kaggle path.
"""

from pathlib import Path
import shutil


# TODO: change this path to where your Kaggle data lives
DATA_ROOT = Path("/kaggle/input/semeval-2026-task13")  # or Path("data_raw_semeval")


TASK_DIRS = {
    "task_a": Path("task_a/data/raw"),
    "task_b": Path("task_b/data/raw"),
    "task_c": Path("task_c/data/raw"),
}

PARQUET_NAMES = [
    "train.parquet",
    "validation.parquet",
    "test.parquet",
]


def copy_for_task(task: str):
    src_base = DATA_ROOT / task   # e.g. /kaggle/.../task_a
    dst_base = TASK_DIRS[task]

    dst_base.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Preparing data for {task} ===")
    for name in PARQUET_NAMES:
        src = src_base / name
        dst = dst_base / name

        if not src.exists():
            print(f"⚠  Missing source file: {src}")
            continue

        if dst.exists():
            print(f"✔  Already exists: {dst}")
        else:
            print(f"→ Copying {src} → {dst}")
            shutil.copy(src, dst)


def main():
    for task in TASK_DIRS.keys():
        copy_for_task(task)

    print("\n Data prepared for all tasks.")


if __name__ == "__main__":
    main()
