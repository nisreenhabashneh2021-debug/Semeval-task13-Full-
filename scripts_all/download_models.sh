#!/usr/bin/env bash
set -e

python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

MODELS = [
    # Task A/B (sequence classification)
    "microsoft/graphcodebert-base",
    "microsoft/codebert-base",
    "bert-base-uncased",
    "Salesforce/codet5-small",
    # Task C (token classification – same checkpoints)
    "microsoft/graphcodebert-base",
]

for name in MODELS:
    print("Downloading:", name)
    AutoTokenizer.from_pretrained(name)
    # Try sequence head
    try:
        AutoModelForSequenceClassification.from_pretrained(name)
    except Exception:
        pass
    # Try token head
    try:
        AutoModelForTokenClassification.from_pretrained(name)
    except Exception:
        pass

print(" Finished downloading all required models.")
EOF
