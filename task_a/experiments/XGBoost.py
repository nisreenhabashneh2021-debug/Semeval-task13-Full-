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
