# task_c/src/models.py

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


def build_token_classifier(model_name: str, num_labels: int):
    """
    Creates tokenizer + token classification model (CodeBERT / GraphCodeBERT).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model, tokenizer
