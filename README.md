SemEval-2026 Task 13 — Full System (Subtasks A, B, C)

This repository contains our **complete end-to-end system** for **SemEval-2026 Task 13: Detecting Machine-Generated Code**.  
The solution covers **all subtasks**:

- **Subtask A — Binary Code Generation Detection**  
  Classify code as *Human-written* or *Machine-generated*.

- **Subtask B — Multi-Class Authorship Detection**  
  Identify whether the code was written by a human or one of **10 LLM families**.

- **Subtask C — Hybrid & Adversarial Code Detection**  
  Predict whether a snippet is *Human*, *Machine*, *Hybrid*, or *Adversarial*.

Our system includes:
- simple baselines
- Transformer models
- ensemble models
- GPU-optimized training loops  
- Logging, metrics, and plotting utilities  
- Fully modular folder structure for scalable experimentation  


```

## 📁 Repository Structure

```
semeval-2026-task13-full/
│
├── common/                     # Shared utilities
│   ├── __init__.py
│   ├── data_utils.py           # Data loading, parquet readers
│   ├── metrics.py              # Accuracy, Macro-F1, reports
│   └── plotting.py             # Confusion matrices, plots
│
├── task_a/                     # Subtask A modules
│   ├── src/
│   │   ├── __init__.py
│   │   ├── models.py          
│   │   ├── train_utils.py      # Training loop, collators
│   │   ├── eval_utils.py       # Evaluation + metrics
│   │   └── inference.py        # Submission CSV generation
│   ├── experiments/
│   │   ├── run_tfidf_baseline.py
│   │   ├── run_transformer.py
│   │   └── run_ensemble.py
│   ├
│   ├
│   │   
│   │   
│   │   
│   └── data/
│       ├── raw/
│       └── processed/
│
├── task_b/                     # Subtask B (same structure as A)
├── task_c/                     # Subtask C (same structure as A)
│
├── notebook_important/
│   ├── EDA.ipynb
│   └── model_analysis.ipynb
│
├── scripts_all/
│   ├── download_models.sh
│   ├── prepare_data.py
│   └── evaluate_all.py
│
├── requirements.txt
├── .gitignore
└── README.md
```














