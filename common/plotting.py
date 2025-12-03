# common/plotting.py

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Sequence[int] | None = None,
    label_names: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names if label_names is not None else labels,
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax


def plot_label_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
    title: str = "Label Distribution",
    save_path: str | None = None,
):
    counts = df[label_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel(label_col)
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax


def plot_metric_bars(
    metrics_dict: dict[str, float],
    title: str = "Metrics",
    save_path: str | None = None,
):
    names = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax
