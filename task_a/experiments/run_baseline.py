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

