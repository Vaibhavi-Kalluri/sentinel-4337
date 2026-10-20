"""Compare E0-style and E1 contextual Random Forests on identical E1 splits."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
SPLIT_DIR = ROOT / "data/synthetic/e1_splits"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_contextual_dataset import MODEL_COLUMNS, MODEL_FEATURES, CONTEXT_FEATURES

LABEL = "label"
METADATA = {
    "scenario_id", "generation_batch", "scenario_type", "label",
    "label_source", "session_id", "operation_index", "wallet_id",
    "destination_id", "selector_id", "timestamp_seconds", "value_eth",
}


def select_threshold(y_true, probabilities):
    """Choose F1-maximizing threshold on validation only."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    if len(thresholds) == 0:
        return 0.5

    # precision/recall contain one more point than thresholds.
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best = int(np.argmax(f1_values))
    return float(thresholds[best])


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold_selected_on_validation": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "false_positive_rate": float(fp / max(fp + tn, 1)),
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv")
        for name in ["train", "validation", "test"]
    }
    train, validation, test = (
        splits["train"], splits["validation"], splits["test"]
    )

    # Strictly define feature sets; never infer model inputs from all columns.
    feature_sets = {
        "E0_features_only": list(MODEL_FEATURES),
        "E1_with_context": list(MODEL_COLUMNS),
    }

    # Verify no session is shared across splits.
    session_sets = {
        name: set(part["session_id"])
        for name, part in splits.items()
    }
    names = list(session_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if session_sets[left] & session_sets[right]:
                raise ValueError(f"Session leakage between {left} and {right}")

    for name, part in splits.items():
        missing = set(MODEL_COLUMNS + [LABEL]) - set(part.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if part[MODEL_COLUMNS].isna().any().any():
            raise ValueError(f"{name} has missing model feature values")
        if not set(part[LABEL].unique()).issubset({0, 1}):
            raise ValueError(f"{name} has invalid labels")

    y_train = train[LABEL].astype(int)
    y_val = validation[LABEL].astype(int)
    y_test = test[LABEL].astype(int)

    results = {
        "experiment": "paired_synthetic_E0_vs_E1",
        "dataset_type": "synthetic_simulation",
        "test_used_for_threshold_selection": False,
        "same_train_validation_test_records_for_both_models": True,
        "session_grouped_split": True,
        "labels_are_verified_real_world_maliciousness": False,
        "models": {},
        "split_sizes": {
            name: int(len(part)) for name, part in splits.items()
        },
        "session_counts": {
            name: int(part["session_id"].nunique())
            for name, part in splits.items()
        },
        "limitations": [
            "All records and labels are synthetic.",
            "Scenario-conditioned context may create artificial predictive signals.",
            "The results do not establish real-world detection performance.",
            "PR-AUC depends on the synthetic class prevalence.",
        ],
    }

    prediction_frame = test[
        ["session_id", "scenario_type", LABEL]
    ].copy().reset_index(drop=True)

    for model_name, features in feature_sets.items():
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4337,
            n_jobs=-1,
        )
        model.fit(train[features], y_train)

        val_probabilities = model.predict_proba(validation[features])[:, 1]
        threshold = select_threshold(y_val, val_probabilities)

        # Test data is evaluated only after threshold selection is complete.
        test_probabilities = model.predict_proba(test[features])[:, 1]
        metrics = evaluate(y_test, test_probabilities, threshold)

        per_scenario = {}
        for scenario in sorted(test["scenario_type"].unique()):
            mask = test["scenario_type"].to_numpy() == scenario
            scenario_labels = y_test.to_numpy()[mask]
            scenario_predictions = (
                test_probabilities[mask] >= threshold
            ).astype(int)
            per_scenario[scenario] = {
                "rows": int(mask.sum()),
                "precision": float(precision_score(
                    scenario_labels, scenario_predictions,
                    zero_division=0
                )),
                "recall": float(recall_score(
                    scenario_labels, scenario_predictions,
                    zero_division=0
                )),
                "f1": float(f1_score(
                    scenario_labels, scenario_predictions,
                    zero_division=0
                )),
                "false_positives": int(
                    ((scenario_labels == 0) & (scenario_predictions == 1)).sum()
                ),
                "false_negatives": int(
                    ((scenario_labels == 1) & (scenario_predictions == 0)).sum()
                ),
            }

        metrics["feature_count"] = len(features)
        metrics["per_scenario"] = per_scenario
        metrics["top_10_features"] = sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(
                    features, model.feature_importances_
                )
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:10]

        results["models"][model_name] = metrics
        prediction_frame[f"{model_name}_probability"] = test_probabilities
        prediction_frame[f"{model_name}_prediction"] = (
            test_probabilities >= threshold
        ).astype(int)

        model_path = MODEL_DIR / f"{model_name.lower()}_e1_split.joblib"
        joblib.dump(
            {
                "model": model,
                "features": features,
                "threshold": threshold,
                "experiment": "paired_synthetic_E0_vs_E1",
            },
            model_path,
        )
        print(f"\n========== {model_name} ==========")
        print(json.dumps({
            key: metrics[key]
            for key in [
                "threshold_selected_on_validation",
                "precision", "recall", "f1",
                "pr_auc_average_precision", "roc_auc",
                "false_positive_rate", "confusion_matrix",
            ]
        }, indent=2))
        print("Top features:")
        for item in metrics["top_10_features"]:
            print(f"  {item['feature']}: {item['importance']:.4f}")
        print("\nPer-scenario test metrics:")
        print(json.dumps(per_scenario, indent=2))
        print(f"Saved model: {model_path}")

    report_path = REPORT_DIR / "e1_contextual_comparison.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    prediction_path = REPORT_DIR / "e1_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    print(f"\nSaved report: {report_path}")
    print(f"Saved test predictions: {prediction_path}")


if __name__ == "__main__":
    main()
