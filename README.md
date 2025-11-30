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


## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for transformer models)

### Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/semeval-2026-task13-full.git
cd semeval-2026-task13-full


2. Install dependencies:
bash
pip install -r requirements.txt


3. Download and prepare the data:
bash
python scripts/download_data.py
python scripts/preprocess_data.py


## 💻 Usage

### Running Experiments

Each subtask follows the same experimental structure:

#### TF-IDF Baseline
bash
python task_a/experiments/run_tfidf_baseline.py --config task_a/configs/tfidf_config.yaml


#### Transformer Models
bash
python task_a/experiments/run_transformer.py \
    --model bert-base-uncased \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5


#### Ensemble Methods
bash
python task_a/experiments/run_ensemble.py --config task_a/configs/ensemble_config.yaml


### Generating Submissions

bash
python task_a/src/inference.py \
    --model_path task_a/results/best_model.pt \
    --test_data data/test.csv \
    --output task_a/results/submissions/submission.csv


## 📊 Model Performance

| Subtask | Model | Accuracy | Macro-F1 |
|---------|-------|----------|----------|
| A | TF-IDF Baseline | TBD | TBD |
| A | CodeBERT | TBD | TBD |
| B | CodeT5-small | TBD | TBD |
| C | Ensemble | TBD | TBD |

## 🔧 Configuration

Each experiment can be configured using YAML files in the `configs/` directory:

yaml
model:
  name: "microsoft/codebert-base"
  max_length: 512

training:
  epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 500

data:
  train_path: "data/processed/train.csv"
  val_path: "data/processed/val.csv"


## 📈 Results and Visualization

Results are automatically saved to the `results/` directory for each task:
- Training logs in `logs/`
- Plots and confusion matrices in `plots/`
- Submission files in `submissions/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SemEval-2026 Task 13 organizers
- Hugging Face for transformer implementations
- The open-source NLP community

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).

## 📚 Citation

If you use this code in your research, please cite:

bibtex
@inproceedings{yourname2026semeval,
  title={Your System Description for SemEval-2026 Task 13},
  author={Your Name},
  booktitle={Proceedings of SemEval-2026},
  year={2026}
}


---

**Note:** This is a work in progress. Results and implementations will be updated as experiments are completed.