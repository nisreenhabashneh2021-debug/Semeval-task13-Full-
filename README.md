# SemEval-2026 Task 13 – Full System (Subtasks A, B, C)
This repository contains our complete solution for **SemEval-2026 Task 13**, covering:

Subtask A — Binary Machine-Generated Code Detection

Subtask B — Multi-Class LLM Family Authorship Detection

Subtask C — Hybrid + Adversarial Code Classification 

The project is fully modular and includes TF-IDF baselines, transformer-based models (BERT, CodeBERT, CodeT5-small), custom utilities, experiment scripts, and inference pipelines.  
All experiments follow the same structure across Tasks A, B, and C.
semeval-2026-task13-full/
│
├── common/
│ ├── init.py
│ ├── data_utils.py # Shared data loading utilities
│ ├── metrics.py # Accuracy, Macro-F1, reports
│ └── plotting.py # Confusion matrix, visualizations
│
├── task_a/
│ ├── src/
│ │ ├── init.py
│ │ ├── models.py # TF-IDF, transformers
│ │ ├── train_utils.py # Train loop, collators, losses
│ │ ├── eval_utils.py # Evaluation + metrics
│ │ └── inference.py # Submission CSV generation
│ ├── experiments/
│ │ ├── run_tfidf_baseline.py
│ │ ├── run_transformer.py
│ │ └── run_ensemble.py
│ ├── configs/ # YAML/JSON config files
│ ├── results/
│ │ ├── logs/
│ │ ├── plots/
│ │ └── submissions/
│ └── data/
│ ├── raw/
│ └── processed/
│
├── task_b/ # Same layout as task_a
├── task_c/ # Same layout as task_a
│
├── notebooks/ # Jupyter notebooks, analysis, error inspection
├── scripts/ # Utility scripts (downloading, preprocessing)
│
├── requirements.txt
├── .gitignore
└── README.md