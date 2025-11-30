# SemEval-2026 Task 13 – Full System Solution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Complete solution for **SemEval-2026 Task 13: Machine-Generated Code Detection and Attribution**, covering all three subtasks with modular, extensible implementations.

## 📋 Overview

This repository provides a comprehensive framework for detecting and classifying machine-generated code:

- **Subtask A** — Binary Machine-Generated Code Detection
- **Subtask B** — Multi-Class LLM Family Authorship Detection
- **Subtask C** — Hybrid + Adversarial Code Classification

### Features

- 🔧 **Modular Architecture** — Reusable components across all subtasks
- 🤖 **Multiple Model Types** — TF-IDF baselines, BERT, CodeBERT, CodeT5-small
- 📊 **Comprehensive Evaluation** — Accuracy, Macro-F1, confusion matrices, detailed reports
- 🔄 **Reproducible Experiments** — Consistent structure and configuration-driven experiments
- 📈 **Visualization Tools** — Built-in plotting and analysis utilities

## 🗂️ Project Structure


semeval-2026-task13-full/
│
├── common/                      # Shared utilities across all tasks
│   ├── _init_.py
│   ├── data_utils.py           # Data loading and preprocessing
│   ├── metrics.py              # Evaluation metrics (Accuracy, Macro-F1)
│   └── plotting.py             # Visualization utilities
│
├── task_a/                      # Subtask A: Binary Detection
│   ├── src/
│   │   ├── _init_.py
│   │   ├── models.py           # Model implementations
│   │   ├── train_utils.py      # Training loops and utilities
│   │   ├── eval_utils.py       # Evaluation functions
│   │   └── inference.py        # Submission generation
│   ├── experiments/
│   │   ├── run_tfidf_baseline.py
│   │   ├── run_transformer.py
│   │   └── run_ensemble.py
│   ├── configs/                # Configuration files (YAML/JSON)
│   ├── results/
│   │   ├── logs/               # Training logs
│   │   ├── plots/              # Visualizations
│   │   └── submissions/        # Competition submissions
│   └── data/
│       ├── raw/                # Original datasets
│       └── processed/          # Preprocessed data
│
├── task_b/                      # Subtask B: Multi-Class Attribution
│   └── [Same structure as task_a]
│
├── task_c/                      # Subtask C: Hybrid Classification
│   └── [Same structure as task_a]
│
├── notebooks/                   # Jupyter notebooks for analysis
├── scripts/                     # Utility scripts
│
├── requirements.txt
├── .gitignore
└── README.md
