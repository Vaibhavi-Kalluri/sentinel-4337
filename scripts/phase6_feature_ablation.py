from pathlib import Path
import json
import time

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
from sklearn.model_selection import GroupKFold

ROOT = Path("data/experimental/real_background")
OUT = Path("reports/phase6/feature_ablation")
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.loads(
    Path("data/processed/multichain_feature_manifest.json").read_text()
)
ALL_FEATURES = manifest["candidate_model_features"]
df = pd.read_csv(ROOT / "real_background_injected_benchmark.csv")

experiments = {
    "full_17_features": ALL_FEATURES,
    "without_value_transfer_counts": [
        f for f in ALL_FEATURES
        if f not in {
            "value_transfer_trace_count",
            "log1p_value_transfer_trace_count",
        }
    ],
    "without_has_paymaster": [
        f for f in ALL_FEATURES if f != "has_paymaster"
    ],
}

required = set(ALL_FEATURES) | {
    "label", "source_id", "chain", "modification_type"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

X_all = df[ALL_FEATURES].apply(pd.to_numeric, errors="raise")
if not np.isfinite(X_all.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite features found")

# Confirm every source has exactly one original and one perturbed record.
pairs = df.groupby("source_id").agg(
    rows=("label", "size"),
    labels=("label", "nunique"),
    chains=("chain", "nunique"),
)
assert pairs["rows"].eq(2).all(), "Unexpected number of rows per source"
assert pairs["labels"].eq(2).all(), "Each source must have labels 0 and 1"
assert pairs["chains"].eq(1).all(), "A source appears in multiple chains"

results = []
prediction_frames = []

for experiment_name, features in experiments.items():
    print(f"\n{'=' * 65}")
    print(f"EXPERIMENT: {experiment_name} ({len(features)} features)")
    print(f"{'=' * 65}")

    X = X_all[features]

    for heldout in sorted(df["chain"].unique()):
        train_mask = df["chain"].ne(heldout).to_numpy()
        test_mask = ~train_mask

        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        X_train = X.loc[train_mask].reset_index(drop=True)
        X_test = X.loc[test_mask].reset_index(drop=True)
        y_train = train_df["label"].astype(int)
        y_test = test_df["label"].astype(int)

        # Choose threshold using grouped OOF predictions from training chains.
        # All records belonging to the same source stay in one fold.
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
            fold_model.fit(
                X_train.iloc[fit_idx],
                y_train.iloc[fit_idx],
            )
            oof_scores[val_idx] = fold_model.predict_proba(
                X_train.iloc[val_idx]
            )[:, 1]

        negatives = oof_scores[y_train.to_numpy() == 0]
        threshold = float(np.quantile(negatives, 0.95))

        # Fit final model on all non-held-out chains.
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

        # Save every held-out operation's score and prediction.
        # source_id lets us compare the same source across feature sets.
        prediction_frames.append(pd.DataFrame({
            "experiment": experiment_name,
            "source_id": test_df["source_id"].to_numpy(),
            "chain": test_df["chain"].to_numpy(),
            "label": y_test.to_numpy(),
            "modification_type": test_df["modification_type"].to_numpy(),
            "score": scores,
            "prediction": pred,
            "threshold": threshold,
        }))

        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        row = {
            "experiment": experiment_name,
            "n_features": len(features),
            "heldout_chain": heldout,
            "threshold": threshold,
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
            "pr_auc": average_precision_score(y_test, scores),
            "roc_auc": roc_auc_score(y_test, scores),
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fit_seconds": fit_seconds,
            "inference_ms_per_row":
                1000 * inference_seconds / len(test_df),
        }
        results.append(row)

        print(
            f"{heldout:12s} "
            f"F1={row['f1']:.4f} "
            f"Recall={row['recall']:.4f} "
            f"FPR={row['fpr']:.4f} "
            f"PR-AUC={row['pr_auc']:.4f} "
            f"threshold={threshold:.4f}"
        )

# Save aggregate metrics.
result_df = pd.DataFrame(results)
metrics_path = OUT / "metrics_by_experiment_and_chain.csv"
result_df.to_csv(metrics_path, index=False)

metric_cols = [
    "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr"
]
macro = (
    result_df.groupby("experiment")[metric_cols]
    .mean()
    .reset_index()
)
macro_path = OUT / "macro_metrics.csv"
macro.to_csv(macro_path, index=False)

# Save per-operation predictions across every experiment and held-out chain.
prediction_df = pd.concat(prediction_frames, ignore_index=True)
prediction_path = OUT / "predictions.csv"
prediction_df.to_csv(prediction_path, index=False)

summary = {
    "experiment": "Feature ablation with leave-one-chain-out evaluation",
    "feature_sets": experiments,
    "threshold_policy": (
        "For each held-out chain and feature set, the threshold is the "
        "95th percentile of training-chain grouped out-of-fold scores "
        "on unmodified records. Held-out labels are not used for threshold "
        "selection."
    ),
    "prediction_output": str(prediction_path),
    "prediction_columns": [
        "experiment",
        "source_id",
        "chain",
        "label",
        "modification_type",
        "score",
        "prediction",
        "threshold",
    ],
    "macro_mean_metrics": macro.to_dict(orient="records"),
    "limitations": [
        "Label 1 is a controlled synthetic feature perturbation, not a verified attack.",
        "Label 0 is an unmodified real-background record, not verified benignness.",
        "Feature ablation tests sensitivity to feature removal, not causality.",
        "Thresholds target 5% FPR on training OOF negatives; this is not guaranteed on held-out chains.",
        "Results do not establish real-world malicious-operation detection accuracy.",
    ],
}
summary_path = OUT / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("\nMACRO MEAN ACROSS HELD-OUT CHAINS")
print(macro.round(4).to_string(index=False))

print("\nOUTPUT FILES")
print(f"Detailed metrics:       {metrics_path}")
print(f"Macro metrics:          {macro_path}")
print(f"Per-operation predictions: {prediction_path}")
print(f"Summary:                {summary_path}")
print(f"Prediction rows:        {len(prediction_df):,}")
print(
    "Rows per experiment:",
    prediction_df["experiment"].value_counts().to_dict(),
)
