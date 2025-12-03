weighted average ensemble

#Step 1: Get XGBoost probs on the same validation subset

# 0) Fix the validation subset ONCE
val_small = validation.sample(10000, random_state=42).reset_index(drop=True)
y_val = val_small["label"].values

# 1) CodeBERT probs on this exact val_small
val_ds = CodeDataset(
    val_small["code"].tolist(),
    val_small["label"].tolist()
)
pred_output = trainer.predict(val_ds)
logits = pred_output.predictions
codebert_prob = softmax(logits, axis=1)[:, 1]   # shape (10000,)

# 2) XGBoost probs on this exact SAME val_small
val_small_feats = val_small["code"].apply(extract_features).apply(pd.Series)
xgb_val_prob = xgb.predict_proba(val_small_feats)[:, 1]  # shape (10000,)


#Step 2: Grid-search ensemble weight and threshold

from sklearn.metrics import f1_score

best_f1 = 0.0
best_w = None
best_th = None

for w in np.linspace(0, 1, 21):
    ens_prob = w * codebert_prob + (1 - w) * xgb_val_prob

    for th in np.linspace(0.2, 0.8, 25):
        ens_pred = (ens_prob >= th).astype(int)
        f1 = f1_score(y_val, ens_pred, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_w = w
            best_th = th

print("Best ensemble weight w:", best_w)
print("Best threshold:", best_th)
print("Best macro F1 on validation:", best_f1)


#Evaluate

from sklearn.metrics import classification_report, f1_score, confusion_matrix

# We already computed ±:
# codebert_prob
# xgb_val_prob
# best_w, best_th
# y_val

ens_val_prob = best_w * codebert_prob + (1 - best_w) * xgb_val_prob
ens_val_pred = (ens_val_prob >= best_th).astype(int)

print("=== ENSEMBLE VALIDATION RESULTS ===")
print(classification_report(y_val, ens_val_pred, digits=4))

print("Macro F1 (Ensemble):", f1_score(y_val, ens_val_pred, average="macro"))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, ens_val_pred))
