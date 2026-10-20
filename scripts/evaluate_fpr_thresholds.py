"""Evaluate validation-selected thresholds under a false-positive-rate target."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from joblib import load

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/synthetic/e1_splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET_FPR = 0.05


def metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold_at_fpr(y, probabilities, target_fpr):
    """Choose the lowest threshold meeting the target FPR on validation.

    Among thresholds satisfying the FPR constraint, the lowest threshold
    maximizes recall. The test set is not used for threshold selection.
    """
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    feasible = []

    for threshold in candidates:
        result = metrics(y, probabilities, threshold)
        if result["false_positive_rate"] <= target_fpr:
            feasible.append((result["recall"], threshold, result))

    if not feasible:
        # Always-negative classifier: explicit safe fallback.
        threshold = float(np.nextafter(np.max(probabilities), np.inf))
        return threshold

    # Maximize recall; for ties, choose the higher threshold.
    feasible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return float(feasible[0][1])


def main():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    validation = pd.read_csv(SPLIT_DIR / "validation.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")

    y_val = validation["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    model_files = {
        "E0_features_only": MODEL_DIR / "e0_features_only_e1_split.joblib",
        "E1_with_context": MODEL_DIR / "e1_with_context_e1_split.joblib",
    }

    results = {
        "target_validation_fpr": TARGET_FPR,
        "threshold_selected_on": "validation only",
        "test_used_for_threshold_selection": False,
        "dataset_type": "synthetic_simulation",
        "models": {},
        "limitations": [
            "Synthetic labels are not verified maliciousness labels.",
            "The validation FPR target may not hold on unseen data.",
            "The test set is small in terms of independent sessions.",
        ],
    }

    for name, path in model_files.items():
        bundle = load(path)
        model = bundle["model"]
        features = bundle["features"]

        val_prob = model.predict_proba(validation[features])[:, 1]
        test_prob = model.predict_proba(test[features])[:, 1]

        threshold = choose_threshold_at_fpr(
            y_val, val_prob, TARGET_FPR
        )

        val_metrics = metrics(y_val, val_prob, threshold)
        test_metrics = metrics(y_test, test_prob, threshold)

        results["models"][name] = {
            "feature_count": len(features),
            "previous_f1_threshold": float(bundle["threshold"]),
            "validation": val_metrics,
            "test": test_metrics,
            "test_pr_auc_average_precision": float(
                average_precision_score(y_test, test_prob)
            ),
            "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        }

    output = REPORT_DIR / "e1_fpr_threshold_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")

    print(json.dumps(results, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
