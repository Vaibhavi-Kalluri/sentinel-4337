from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path("data/experimental/real_background/splits")
OUT = Path("reports/real_background_benchmark/ablation")
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads(
    Path("data/processed/multichain_feature_manifest.json").read_text()
)
FULL = manifest["candidate_model_features"]

# These groups are derived from the approved feature names.
INDICATORS = [
    c for c in FULL
    if c.startswith("has_")
]
COUNTS = [
    c for c in FULL
    if not c.startswith("has_")
    and not c.startswith("log1p_")
    and c not in {"delegatecall_count"}
]
CONFIGS = {
    "count_only": COUNTS,
    "indicator_only": INDICATORS,
    "full_17_features": FULL,
}

def load(name):
    df = pd.read_csv(ROOT / f"{name}.csv")
    X = df[FULL].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite feature values in {name}")
    return df, X, df["label"].astype(int)

train_df, X_train_full, y_train = load("train")
val_df, X_val_full, y_val = load("validation")
test_df, X_test_full, y_test = load("test")

# Ensure the grouped split remains leakage-safe.
ids = [
    set(d["source_id"])
    for d in (train_df, val_df, test_df)
]
assert ids[0].isdisjoint(ids[1])
assert ids[0].isdisjoint(ids[2])
assert ids[1].isdisjoint(ids[2])

def metrics(y, prob, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }

def choose_threshold_at_fpr(y, prob, max_fpr=0.05):
    # Evaluate every distinct score threshold on validation data.
    candidates = np.unique(np.r_[0.0, prob, 1.0000001])
    eligible = []
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        m = metrics(y, prob, pred)
        if m["fpr"] <= max_fpr:
            eligible.append((m["recall"], m["precision"], threshold, m))

    if not eligible:
        return None, None

    # Maximise recall, then precision; prefer the higher threshold on ties.
    eligible.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, threshold, m = eligible[-1]
    return float(threshold), m

rows = []
all_predictions = []

for name, features in CONFIGS.items():
    if not features:
        print(f"Skipping {name}: no features selected.")
        continue

    print(f"\nTraining {name}: {len(features)} features")
    start = time.perf_counter()

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=4337,
        n_jobs=-1,
    )
    model.fit(X_train_full[features], y_train)
    fit_seconds = time.perf_counter() - start

    val_prob = model.predict_proba(X_val_full[features])[:, 1]
    threshold, val_metrics = choose_threshold_at_fpr(
        y_val.to_numpy(), val_prob, max_fpr=0.05
    )

    if threshold is None:
        print(f"{name}: no threshold met validation FPR <= 5%.")
        rows.append({
            "configuration": name,
            "feature_count": len(features),
            "status": "no_threshold_met_validation_fpr",
            "fit_seconds": fit_seconds,
        })
        continue

    test_prob = model.predict_proba(X_test_full[features])[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = metrics(y_test.to_numpy(), test_prob, test_pred)

    model_path = OUT / f"{name}.joblib"
    joblib.dump(model, model_path)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUT / f"{name}_importance.csv", index=False)

    row = {
        "configuration": name,
        "feature_count": len(features),
        "features": features,
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fit_seconds": fit_seconds,
        "status": "ok",
    }
    rows.append(row)

    part = test_df[["source_id", "chain", "label", "modification_type"]].copy()
    part["configuration"] = name
    part["score"] = test_prob
    part["prediction"] = test_pred
    all_predictions.append(part)

    print(f"Validation threshold: {threshold:.6f}")
    print(f"Validation FPR:       {val_metrics['fpr']:.4f}")
    print(f"Test FPR:             {test_metrics['fpr']:.4f}")
    print(f"Test recall:          {test_metrics['recall']:.4f}")
    print(f"Test precision:       {test_metrics['precision']:.4f}")
    print(f"Test F1:              {test_metrics['f1']:.4f}")
    print(f"Test PR-AUC:          {test_metrics['pr_auc_average_precision']:.4f}")
    print(f"Test ROC-AUC:         {test_metrics['roc_auc']:.4f}")

(OUT / "results.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)

if all_predictions:
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)

    summary = pd.DataFrame([
        {
            "configuration": r["configuration"],
            "feature_count": r["feature_count"],
            "validation_fpr": r["validation_metrics"]["fpr"],
            "threshold": r["threshold"],
            **{
                f"test_{k}": v
                for k, v in r["test_metrics"].items()
            },
        }
        for r in rows if r.get("status") == "ok"
    ])
    summary.to_csv(OUT / "summary.csv", index=False)
    print("\n=== ABLATION SUMMARY ===")
    print(summary.to_string(index=False))

print(f"\nSaved experiment outputs to {OUT}")
print(
    "\nReminder: this evaluates synthetic feature perturbations, "
    "not verified malicious UserOperations."
)
