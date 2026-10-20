from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupKFold

ROOT = Path("data/experimental/real_background")
OUT = Path("reports/phase6/chain_holdout")
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads(
    Path("data/processed/multichain_feature_manifest.json").read_text()
)
FEATURES = manifest["candidate_model_features"]
df = pd.read_csv(ROOT / "real_background_injected_benchmark.csv")

required = set(FEATURES) | {
    "label", "source_id", "chain", "modification_type"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

X = df[FEATURES].apply(pd.to_numeric, errors="raise")
if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite features found")

# Verify original/modified pairs remain intact.
pair_check = df.groupby("source_id").agg(
    rows=("label", "size"),
    labels=("label", "nunique"),
    chains=("chain", "nunique"),
)
assert pair_check["rows"].eq(2).all()
assert pair_check["labels"].eq(2).all()
assert pair_check["chains"].eq(1).all()

results = []
predictions = []

for heldout in sorted(df["chain"].unique()):
    train_mask = df["chain"].ne(heldout).to_numpy()
    test_mask = ~train_mask

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    # Select threshold from grouped cross-validation on the five training chains.
    # Both versions of a source operation are assigned to the same fold.
    cv = GroupKFold(n_splits=5)
    oof_scores = np.zeros(len(train_df), dtype=float)

    for fit_idx, val_idx in cv.split(
        X_train, y_train, groups=train_df["source_id"]
    ):
        fold_model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4337,
            n_jobs=-1,
        )
        fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_scores[val_idx] = fold_model.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

    # A training-only threshold targeting approximately 5% FPR on
    # cross-validated unmodified examples. No held-out-chain labels used.
    negatives = oof_scores[y_train.to_numpy() == 0]
    threshold = float(np.quantile(negatives, 0.95))

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

    start = time.perf_counter()
    scores = model.predict_proba(X_test)[:, 1]
    inference_seconds = time.perf_counter() - start
    pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    row = {
        "heldout_chain": heldout,
        "train_chains": sorted(train_df["chain"].unique()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "threshold_from_training_oof_negative_95th_percentile": threshold,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix_labels_0_1": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fit_seconds": fit_seconds,
        "inference_ms_per_row": 1000 * inference_seconds / len(test_df),
    }
    results.append(row)

    part = test_df[["source_id", "chain", "label", "modification_type"]].copy()
    part["score"] = scores
    part["prediction"] = pred
    part["threshold"] = threshold
    predictions.append(part)

    print(
        f"{heldout:12s} F1={row['f1']:.4f} "
        f"Recall={row['recall']:.4f} "
        f"FPR={row['false_positive_rate']:.4f} "
        f"PR-AUC={row['pr_auc_average_precision']:.4f}"
    )

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "metrics_by_heldout_chain.csv", index=False)
pd.concat(predictions, ignore_index=True).to_csv(
    OUT / "predictions.csv", index=False
)

summary = {
    "experiment": "Leave-one-chain-out evaluation",
    "features": FEATURES,
    "threshold_policy": (
        "For each held-out chain, threshold is the 95th percentile of "
        "out-of-fold scores on unmodified examples from the other five chains."
    ),
    "metrics_macro_mean": {
        col: float(result_df[col].mean())
        for col in [
            "precision", "recall", "f1", "pr_auc_average_precision",
            "roc_auc", "false_positive_rate"
        ]
    },
    "results": results,
    "limitations": [
        "Label 1 denotes a controlled synthetic feature perturbation, not a verified attack.",
        "Label 0 denotes an unmodified real-background record, not verified benignness.",
        "This measures cross-chain generalization to the designed perturbations only.",
        "The 5% training threshold target is not guaranteed to produce 5% FPR on a held-out chain.",
        "This experiment does not establish real-world malicious-operation detection accuracy."
    ]
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

print("\nMacro mean across held-out chains:")
print(result_df[
    ["precision", "recall", "f1", "pr_auc_average_precision",
     "roc_auc", "false_positive_rate"]
].mean().round(4).to_string())
print(f"\nReports saved to {OUT}")
