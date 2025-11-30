
# SemEval-2026 Task 13 – Full System (Subtasks A, B, C)

This repository contains our complete, modular solution for **SemEval-2026 Task 13**, covering:

- **Subtask A — Binary Machine-Generated Code Detection**
- **Subtask B — Multi-Class LLM Family Authorship Detection**
- **Subtask C — Hybrid + Adversarial Code Classification**

The project includes:

- TF–IDF + classical ML baselines  
- Transformer-based models (e.g., **BERT**, **CodeBERT**, **CodeT5-small**)  
- Shared utilities for data loading, metrics, and plotting  
- Experiment scripts for training, evaluation, and ensembling  
- Inference pipelines for generating competition-ready submission files  

All three subtasks follow the **same structure and workflow**.

---

## 📌 Task Overview

### 🔹 Subtask A — Binary Machine-Generated Code Detection

**Goal:**  
Given a code snippet, determine whether it is:

- **Human-written**, or  
- **Machine-generated**

**Training Languages:** C++, Python, Java  
**Domains:** Algorithmic (e.g., LeetCode-style problems), with evaluation including unseen languages and domains.  

**Dataset sizes (approx.):**

- Train: 500K samples (238K human, 262K machine-generated)  
- Validation: 100K samples  

**Official metric:**  
- **Macro F1-score** (used for leaderboard ranking)

---

### 🔹 Subtask B — Multi-Class LLM Family Authorship Detection

**Goal:**  
Given a code snippet, predict its **author family**:

- **Human**  
- One of **10 LLM families**:
  - DeepSeek-AI  
  - Qwen  
  - 01-ai  
  - BigCode  
  - Gemma  
  - Phi  
  - Meta-LLaMA  
  - IBM-Granite  
  - Mistral  
  - OpenAI  

**Dataset sizes (approx.):**

- Train: 500K samples (highly imbalanced: many human samples, fewer per LLM family)  
- Validation: 100K samples  

**Official metric:**  
- **Macro F1-score**

---

### 🔹 Subtask C — Hybrid + Adversarial Code Classification

**Goal:**  
Classify each code snippet into one of **four** categories:

1. **Human-written**  
2. **Machine-generated**  
3. **Hybrid** – partially written or completed by an LLM  
4. **Adversarial** – generated to mimic human style (e.g., via adversarial prompts or RLHF)  

**Dataset sizes (approx.):**

- Train: 900K samples  
- Validation: 200K samples  

**Official metric:**  
- **Macro F1-score**

semeval-2026-task13-full/
│
├── common/
│   ├── __init__.py
│   ├── data_utils.py       # Shared data loading utilities
│   ├── metrics.py          # Accuracy, Macro-F1, weighted-F1, reports
│   └── plotting.py         # Confusion matrices & visualizations
│
├── task_a/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── models.py        # TF-IDF, classical ML models, transformers
│   │   ├── train_utils.py   # Training loops & dataloaders
│   │   ├── eval_utils.py    # Evaluation helpers
│   │   └── inference.py     # Submission CSV generator
│   ├── experiments/
│   │   ├── run_tfidf_baseline.py
│   │   ├── run_transformer.py
│   │   └── run_ensemble.py
│   ├── configs/
│   ├── results/
│   │   ├── logs/
│   │   ├── plots/
│   │   └── submissions/
│   └── data/
│       ├── raw/             # Parquet files (NOT uploaded to GitHub)
│       └── processed/
│
├── task_b/                  # Same layout as task_a
│
├── task_c/                  # Same layout as task_a
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_analysis.ipynb
│
├── scripts/
│   ├── download_models.sh
│   ├── prepare_data.py
│   ├── evaluate_all.py
│
├── requirements.txt
├── .gitignore
└── README.md
