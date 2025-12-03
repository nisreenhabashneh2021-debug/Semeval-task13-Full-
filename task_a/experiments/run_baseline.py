#TF-IDF (char n-grams) + Logistic Regression
# Core libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Visualization style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

#Extra imports for baseline

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Pipeline & model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

#load data
train = pd.read_parquet("train.parquet")
validation = pd.read_parquet("validation.parquet")
test = pd.read_parquet("test.parquet")

# Features and labels
X_train = train["code"].astype(str).values
y_train = train["label"].values

X_val = validation["code"].astype(str).values
y_val = validation["label"].values

print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))

#Build TF-IDF + Logistic Regression pipeline

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Optional: handle slight imbalance via class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)

baseline_clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",          # character n-grams
        ngram_range=(3, 5),       # 3-5 char n-grams
        min_df=5,                 # ignore very rare n-grams
        max_features=200_000,     # adjust based on RAM
    )),
    ("logreg", LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight=class_weight_dict,
        solver="lbfgs"
    ))
])
#train 
%%time
baseline_clf.fit(X_train, y_train)

#Evaluate on validation set
# Predictions
y_val_pred = baseline_clf.predict(X_val)
y_val_proba = baseline_clf.predict_proba(X_val)[:, 1]

print("=== Classification report (Validation) ===")
print(classification_report(y_val, y_val_pred, digits=4))

print("=== F1-score (macro) ===")
print(f1_score(y_val, y_val_pred, average="macro"))

print("=== ROC-AUC ===")
print(roc_auc_score(y_val, y_val_proba))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_val, y_val_pred))


#XGBoost

#step 1: Feature Engineering for OOD Robustness

#We create language-agnostic features that generalize across unseen languages:

import numpy as np
import re
from collections import Counter
import math

def calc_entropy(s):
    if not s:
        return 0.0
    probs = [freq/len(s) for freq in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def extract_features(code):
    lines = code.split("\n")
    num_lines = len(lines)
    num_chars = len(code)
    num_tokens = len(code.split())
    
    avg_line_len = num_chars / max(1, num_lines)
    line_lengths = [len(l) for l in lines]
    
    indent_levels = [len(re.match(r'\s*', l).group()) for l in lines]

    return {
        "num_chars": num_chars,
        "num_tokens": num_tokens,
        "num_lines": num_lines,
        "avg_line_len": avg_line_len,
        "line_len_var": np.var(line_lengths),
        "indent_var": np.var(indent_levels),
        "entropy": calc_entropy(code),

        # structural counts
        "brace_count": code.count("{") + code.count("}"),
        "paren_count": code.count("(") + code.count(")"),
        "bracket_count": code.count("[") + code.count("]"),
        "semicolon_count": code.count(";"),
        "comma_count": code.count(","),

        # keyword counts (works across most languages)
        "for_count": code.count("for"),
        "if_count": code.count("if"),
        "while_count": code.count("while"),
        "return_count": code.count("return"),
    }

#Apply feature extraction to train/val/test

train_feats = train["code"].apply(extract_features).apply(pd.Series)
val_feats   = validation["code"].apply(extract_features).apply(pd.Series)
test_feats  = test["code"].apply(extract_features).apply(pd.Series)

print(train_feats.head())

#step 2: XGBoost Model (Strong OOD Baseline)

#This model excels on unseen languages because it uses language-agnostic statistics.
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss"
)

xgb.fit(train_feats, train["label"])
xgb_val_pred_proba = xgb.predict_proba(val_feats)[:, 1]
xgb_val_pred = (xgb_val_pred_proba >= 0.5).astype(int)

#Evalute

print(classification_report(validation["label"], xgb_val_pred, digits=4))
print("Macro F1:", f1_score(validation["label"], xgb_val_pred, average="macro"))
