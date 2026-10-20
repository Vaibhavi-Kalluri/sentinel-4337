from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path("data/experimental/real_background")
SPLITS = ROOT / "splits"
REPORTS = Path("reports/real_background_benchmark")
MODELS = Path("models")

REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

META_PATH = Path("data/processed/multichain_feature_manifest.json")
manifest = json.loads(META_PATH.read_text(encoding="utf-8"))
FEATURES = manifest["candidate_model_features"]

METADATA_COLUMNS = {
    "label", "source_id", "modification_type", "userop_hash",
    "sender", "paymaster", "nonce", "success", "chain", "dt",
    "actual_gas_used", "actual_gas_cost_eth",
    "destination_addresses", "method_selectors", "split",
}

def load_split(name):
    path = SPLITS / f"{name}.csv"
    df = pd.read_csv(path)

    required = set(FEATURES) | {
        "label", "source_id", "chain", "modification_type"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name}: labels must be 0/1")

    # Explicitly select features; never feed metadata to the model.
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: non-finite model features")

    return df, X, df["label"].astype(int)

train_df, X_train, y_train = load_split("train")
val_df, X_val, y_val = load_split("validation")
test_df, X_test, y_test = load_split("test")

# Ensure paired source operations never cross split boundaries.
train_ids = set(train_df["source_id"])
val_ids = set(val_df["source_id"])
test_ids = set(test_df["source_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

# Each source ID must have both its original and modified row in one split.
for name, df in [
    ("train", train_df), ("validation", val_df), ("test", test_df)
]:
    counts = df.groupby("source_id").agg(
        rows=("label", "size"),
        labels=("label", "nunique"),
    )
    assert counts["rows"].eq(2).all(), f"{name}: source pair incomplete"
    assert counts["labels"].eq(2).all(), f"{name}: missing paired label"

model = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=4337,
    n_jobs=-1,
)

start = time.perf_counter()
model.fit(X_train, y_train)
fit_seconds = time.perf_counter() - start

val_prob = model.predict_proba(X_val)[:, 1]

# Choose the threshold using validation F1 only.
precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
if len(thresholds) == 0:
    raise RuntimeError("Could not derive validation thresholds")

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / np.maximum(precision[:-1] + recall[:-1], 1e-12)
)
best_idx = int(np.argmax(f1_values))
threshold = float(thresholds[best_idx])

# Evaluate the held-out test set only after threshold selection.
start = time.perf_counter()
test_prob = model.predict_proba(X_test)[:, 1]
inference_seconds = time.perf_counter() - start
test_pred = (test_prob >= threshold).astype(int)

cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

metrics = {
    "threshold_selected_on_validation": threshold,
    "validation_best_f1": float(f1_values[best_idx]),
    "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "test_average_precision_pr_auc": float(average_precision_score(y_test, test_prob)),
    "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
    "test_false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    "test_confusion_matrix_labels_0_1": cm.tolist(),
    "test_counts": {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    },
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "feature_count": int(len(FEATURES)),
    "features": FEATURES,
    "fit_seconds": fit_seconds,
    "test_inference_seconds_total": inference_seconds,
    "test_inference_ms_per_row": float(
        1000 * inference_seconds / max(len(X_test), 1)
    ),
    "classification_report": classification_report(
        y_test, test_pred, output_dict=True, zero_division=0
    ),
    "limitations": [
        "Label 1 represents controlled synthetic feature perturbation, not verified maliciousness.",
        "Label 0 represents unmodified real-background records, not confirmed benign operations.",
        "This is not evidence of real-world malicious UserOperation detection accuracy.",
        "The benchmark modifies aggregate features without reconstructing transaction traces.",
        "Metrics depend on the chosen perturbation design and should not be presented as deployment performance.",
    ],
}

# Evaluate each injected perturbation type separately.
test_detail = test_df[[
    "source_id", "chain", "label", "modification_type"
]].copy()
test_detail["score"] = test_prob
test_detail["prediction"] = test_pred
test_detail.to_csv(REPORTS / "test_predictions.csv", index=False)

scenario_rows = []
for scenario, group in test_detail[test_detail["label"] == 1].groupby(
    "modification_type"
):
    scenario_rows.append({
        "modification_type": scenario,
        "n": int(len(group)),
        "recall": float((group["prediction"] == 1).mean()),
        "mean_score": float(group["score"].mean()),
    })
pd.DataFrame(scenario_rows).to_csv(
    REPORTS / "test_by_perturbation_type.csv", index=False
)

chain_rows = []
for chain, group in test_detail.groupby("chain"):
    yt = group["label"].to_numpy()
    yp = group["prediction"].to_numpy()
    probs = group["score"].to_numpy()
    chain_cm = confusion_matrix(yt, yp, labels=[0, 1])
    ctn, cfp, cfn, ctp = chain_cm.ravel()
    chain_rows.append({
        "chain": chain,
        "n": int(len(group)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "average_precision": float(average_precision_score(yt, probs)),
        "roc_auc": float(roc_auc_score(yt, probs)),
        "false_positive_rate": float(cfp / (cfp + ctn)) if cfp + ctn else 0.0,
    })
pd.DataFrame(chain_rows).to_csv(
    REPORTS / "test_by_chain.csv", index=False
)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
importance.to_csv(REPORTS / "feature_importance.csv", index=False)

joblib.dump(model, MODELS / "sentinel_real_background_injected_rf.joblib")
(REPORTS / "report.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("\n=== REAL-BACKGROUND INJECTED BENCHMARK ===")
print("Task: distinguish unmodified records from synthetic feature perturbations")
print(f"Features: {len(FEATURES)}")
print(f"Rows train/validation/test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"Validation-selected threshold: {threshold:.6f}")
print(f"Test precision: {metrics['test_precision']:.4f}")
print(f"Test recall:    {metrics['test_recall']:.4f}")
print(f"Test F1:        {metrics['test_f1']:.4f}")
print(f"Test PR-AUC:    {metrics['test_average_precision_pr_auc']:.4f}")
print(f"Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
print(f"Test FPR:       {metrics['test_false_positive_rate']:.4f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print(f"\nFit time: {fit_seconds:.2f}s")
print(f"Test inference: {metrics['test_inference_ms_per_row']:.4f} ms/row")
print("\nTop 10 feature importances:")
print(importance.head(10).to_string(index=False))
print("\nRecall by perturbation type:")
print(pd.DataFrame(scenario_rows).to_string(index=False))
print("\nMetrics by chain:")
print(pd.DataFrame(chain_rows).to_string(index=False))
print(f"\nReports saved to: {REPORTS}")
