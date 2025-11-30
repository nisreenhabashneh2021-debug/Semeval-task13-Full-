# task_x/src/models.py
"""
Model builders for SemEval-2026 Task 13.

Provides:
    - build_tfidf_linear_model  (TF–IDF + Linear model)
    - build_transformer_model   (HuggingFace transformer)

These are intentionally generic so they work for Tasks A, B, and C.
"""

from __future__ import annotations
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# -------------------------------------------------------------------
# Classical baselines
# -------------------------------------------------------------------

def build_tfidf_logreg() -> Pipeline:
    """
    TF–IDF (char + word n-grams) + Logistic Regression.

    Good starting baseline for binary / multi-class classification.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=200_000,
    )
    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        class_weight="balanced",
    )
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("logreg", clf),
    ])
    return pipe


def build_tfidf_svm() -> Pipeline:
    """
    TF–IDF + LinearSVC (often stronger for multi-class like Subtask B).
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        max_features=200_000,
    )
    clf = LinearSVC()
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("svm", clf),
    ])
    return pipe


# -------------------------------------------------------------------
# Transformer models
# -------------------------------------------------------------------

def load_transformer(
    model_name: str,
    num_labels: int,
):
    """
    Load a transformer model + tokenizer for sequence classification.

    Example:
        tokenizer, model = load_transformer("bert-base-uncased", num_labels=11)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model
