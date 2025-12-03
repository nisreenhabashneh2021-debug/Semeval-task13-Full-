# task_b/src/__init__.py

from .labels import FAMILIES, NUM_LABELS, label2name, name2label
from .models import (
    TfidfSVMModel,
    TfidfSVMConfig,
    build_transformer_model,
)
from .train_utils import (
    TransformerTrainConfig,
    train_tfidf_svm,
    train_transformer,
)
from .eval_utils import evaluate_and_log
from .inference import generate_submission_from_model

# task_b/src/labels.py

# Author families (11 classes) for Subtask B
FAMILIES = [
    "human",
    "deepseek-ai",
    "qwen",
    "01-ai",
    "bigcode",
    "gemma",
    "phi",
    "meta-llama",
    "ibm-granite",
    "mistral",
    "openai",
]

label2name = {i: name for i, name in enumerate(FAMILIES)}
name2label = {name: i for i, name in enumerate(FAMILIES)}

NUM_LABELS = len(FAMILIES)
