#!/usr/bin/env python3
"""Train and evaluate Sentinel's first synthetic Random Forest baseline."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
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

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/synthetic/splits"
OUTPUT_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

METADATA_COLUMNS = {
    "scenario_id",
    "generation_batch",
    "scenario_type",
    "label",
    "label_source",
}


def load_splits():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")

    feature_columns = [
        c for c in train.columns if c not in METADATA_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No model features found")

    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing features: {sorted(missing)}")

        if frame[feature_columns].isna().any().any():
            raise ValueError(f"{name} contains missing model feature values")

        if not set(frame["label"].unique()).issubset({0, 1}):
            raise ValueError(f"{name} has non-binary labels")

    if not (
        train["generation_batch"].is_unique
        or True
    ):
        raise ValueError("Unexpected batch validation failure")

    # Verify batches are disjoint across all splits.
    batch_sets = {
        name: set(frame["generation_batch"])
        for name, frame in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]
    }
    names = list(batch_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if batch_sets[left] & batch_sets[right]:
                raise ValueError(f"Batch leakage between {left} and {right}")

    return train, validation, test, feature_columns


def choose_threshold(y_true, probabilities):
    """Choose threshold on validation only, maximizing F1."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    # The final precision/recall point has no corresponding threshold.
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["benign_like", "attack_like"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    train, validation, test, feature_columns = load_splits()

    X_train = train[feature_columns]
    y_train = train["label"].astype(int)
    X_validation = validation[feature_columns]
    y_validation = validation["label"].astype(int)
    X_test = test[feature_columns]
    y_test = test["label"].astype(int)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=4337,
        n_jobs=-1,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    threshold = choose_threshold(y_validation, validation_probabilities)

    # Test set is used only after model fitting and threshold selection.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    validation_metrics = evaluate(
        y_validation, validation_probabilities, threshold
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "sentinel_rf_synthetic.joblib"
    joblib.dump(model, model_path)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = REPORT_DIR / "synthetic_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    test_predictions = (test_probabilities >= threshold).astype(int)
    predictions_path = REPORT_DIR / "synthetic_test_predictions.csv"
    pd.DataFrame({
        "scenario_type": test["scenario_type"],
        "label": y_test,
        "predicted_label": test_predictions,
        "attack_like_probability": test_probabilities,
    }).to_csv(predictions_path, index=False)

    report = {
        "experiment": "Sentinel 4337 synthetic Random Forest baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "synthetic; labels generated by scenario rules",
        "scikit_learn_version": sklearn.__version__,
        "random_seed": 4337,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "feature_columns": feature_columns,
        "metadata_excluded": sorted(METADATA_COLUMNS),
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "validation_metrics_at_selected_threshold": validation_metrics,
        "test_metrics_at_validation_selected_threshold": test_metrics,
        "confusion_matrix_order": [
            ["true benign_like", "predicted benign_like/attack_like"],
            ["true attack_like", "predicted benign_like/attack_like"],
        ],
        "limitations": [
            "Synthetic labels are not verified real-world maliciousness labels.",
            "Results do not establish real blockchain attack detection performance.",
            "Threshold was selected on validation data only.",
            "The test set was not used to fit the model or select the threshold.",
        ],
        "artifacts": {
            "model": str(model_path.relative_to(ROOT)),
            "feature_importance": str(importance_path.relative_to(ROOT)),
            "test_predictions": str(predictions_path.relative_to(ROOT)),
        },
    }

    report_path = REPORT_DIR / "synthetic_rf_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== VALIDATION-SELECTED THRESHOLD ===")
    print(f"Threshold: {threshold:.4f}")
    print("\n=== HELD-OUT TEST METRICS ===")
    for metric in [
        "precision",
        "recall",
        "f1",
        "average_precision_pr_auc",
        "roc_auc",
    ]:
        print(f"{metric}: {test_metrics[metric]:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("\nTop 10 feature importances:")
    print(importance.head(10).to_string(index=False))
    print("\nSaved model:", model_path)
    print("Saved report:", report_path)
    print("Saved feature importances:", importance_path)
    print("Saved predictions:", predictions_path)


if __name__ == "__main__":
    main()
