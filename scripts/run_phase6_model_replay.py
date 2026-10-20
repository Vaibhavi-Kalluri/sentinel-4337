#!/usr/bin/env python3
"""Phase 6: reproducible model-response replay.

Original rows are real-background features without verified benign labels.
Perturbations are controlled feature edits, not reconstructed transactions
or independently verified malicious operations.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/sentinel_real_background_injected_rf.joblib"
DATA_PATH = ROOT / "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
OUT_DIR = ROOT / "reports/phase6"
PREDICTIONS_PATH = OUT_DIR / "model_replay_predictions.csv"
SUMMARY_PATH = OUT_DIR / "model_replay_summary.json"

PERTURBATIONS = [
    "approval_perturbation",
    "transfer_perturbation",
    "destination_perturbation",
    "multi_call_perturbation",
    "combined_perturbation",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed data: {DATA_PATH}")

    print("Loading model and processed real-background features...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("Model does not expose feature_names_in_; refusing to guess feature order.")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Processed CSV is missing model features: {missing}")

    base = df[feature_names].copy()
    for col in feature_names:
        base[col] = pd.to_numeric(base[col], errors="raise")

    if base.isna().any().any():
        raise ValueError("Model feature matrix contains missing values.")
    if len(base) == 0:
        raise ValueError("Input dataset is empty.")

    print(f"Background rows: {len(base):,}")
    print(f"Model feature count: {len(feature_names)}")
    print(f"Classes: {list(model.classes_)}")

    if 1 not in list(model.classes_):
        raise ValueError("Expected class 1 for controlled-perturbation scoring.")
    positive_col = list(model.classes_).index(1)

    records = []
    start = time.perf_counter()

    def add_scores(frame, source_rows, scenario):
        scores = model.predict_proba(frame[feature_names])[:, positive_col]
        for i, score in enumerate(scores):
            records.append({
                "source_row": int(source_rows[i]),
                "scenario": scenario,
                "score_for_perturbation_class": float(score),
                "threshold_0_5_flag": bool(score >= 0.5),
                "label_status": (
                    "unlabelled_real_background"
                    if scenario == "original_background"
                    else "controlled_feature_perturbation_not_verified_attack"
                ),
            })

    source_ids = np.arange(len(base), dtype=np.int64)
    add_scores(base, source_ids, "original_background")

    perturbation_definitions = {
        "approval_perturbation": [
            ("approve_trace_count", 1, 3),
        ],
        "transfer_perturbation": [
            ("transfer_from_trace_count", 1, 3),
        ],
        "destination_perturbation": [
            ("unique_destinations", 1, 3),
        ],
        "multi_call_perturbation": [
            ("trace_count", 1, 3),
        ],
        "combined_perturbation": [
            ("approve_trace_count", 1, 3),
            ("value_transfer_trace_count", 1, 3),
            ("trace_count", 1, 3),
        ],
    }

    rng = np.random.default_rng(4337)

    for scenario in PERTURBATIONS:
        altered = base.copy()
        for column, low, high in perturbation_definitions[scenario]:
            if column not in altered.columns:
                raise ValueError(f"Required perturbation column missing: {column}")

            # Use a reproducible per-row increment in [1, 3].
            increments = rng.integers(low, high + 1, size=len(altered))
            altered[column] = altered[column].to_numpy() + increments

        # Recalculate the associated log1p features after count changes.
        log_pairs = {
            "trace_count": "log1p_trace_count",
            "unique_destinations": "log1p_unique_destinations",
            "approve_trace_count": "log1p_approve_trace_count",
            "transfer_from_trace_count": "log1p_transfer_from_trace_count",
            "value_transfer_trace_count": "log1p_value_transfer_trace_count",
            "errored_trace_count": "log1p_errored_trace_count",
        }
        for count_col, log_col in log_pairs.items():
            if count_col in altered.columns and log_col in altered.columns:
                altered[log_col] = np.log1p(altered[count_col].clip(lower=0))

        # Update binary indicators for counts that were perturbed.
        indicators = {
            "approve_trace_count": "has_approval",
            "transfer_from_trace_count": "has_transfer_from",
            "value_transfer_trace_count": "has_value_transfer",
        }
        for count_col, flag_col in indicators.items():
            if count_col in altered.columns and flag_col in altered.columns:
                altered[flag_col] = (
                    altered[count_col].to_numpy() > 0
                ).astype(int)

        add_scores(altered, source_ids, scenario)
        print(f"Scored {scenario}: {len(altered):,} rows")

    elapsed = time.perf_counter() - start
    result = pd.DataFrame.from_records(records)
    result.to_csv(PREDICTIONS_PATH, index=False)

    pivot = result.pivot(
        index="source_row",
        columns="scenario",
        values="score_for_perturbation_class",
    )
    summary = {
        "title": "Sentinel Phase 6 model-response replay",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "input_path": str(DATA_PATH.relative_to(ROOT)),
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "background_rows": len(base),
        "perturbation_types": PERTURBATIONS,
        "prediction_rows": len(result),
        "threshold_for_descriptive_flag_rate": 0.5,
        "elapsed_seconds": round(elapsed, 3),
        "scenarios": {},
        "limitations": [
            "Original real-background rows have no verified benign labels.",
            "Perturbation labels describe controlled feature edits, not verified attacks.",
            "These are feature-space edits, not reconstructed or executed UserOperations.",
            "A score is not a calibrated probability of maliciousness.",
            "The 0.5 flag rate is descriptive, not a validated operating threshold.",
            "This experiment does not measure real-world precision, recall, or attack detection rate.",
        ],
    }

    original = pivot["original_background"]
    for scenario in ["original_background"] + PERTURBATIONS:
        scores = pivot[scenario]
        entry = {
            "count": int(scores.notna().sum()),
            "mean_score": round(float(scores.mean()), 6),
            "median_score": round(float(scores.median()), 6),
            "p90_score": round(float(scores.quantile(0.90)), 6),
            "p95_score": round(float(scores.quantile(0.95)), 6),
            "fraction_score_ge_0_5": round(float((scores >= 0.5).mean()), 6),
        }
        if scenario != "original_background":
            delta = scores - original
            entry["paired_mean_score_delta_vs_original"] = round(float(delta.mean()), 6)
            entry["fraction_score_increased_vs_original"] = round(float((delta > 0).mean()), 6)
            entry["fraction_score_decreased_vs_original"] = round(float((delta < 0).mean()), 6)
        summary["scenarios"][scenario] = entry

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n===== RESULTS =====")
    for scenario, values in summary["scenarios"].items():
        print(
            f"{scenario:28s} "
            f"median={values['median_score']:.4f} "
            f"mean={values['mean_score']:.4f} "
            f"score>=0.5={values['fraction_score_ge_0_5']:.1%}"
        )
        if "paired_mean_score_delta_vs_original" in values:
            print(
                f"  paired mean delta="
                f"{values['paired_mean_score_delta_vs_original']:+.4f}; "
                f"score increased on "
                f"{values['fraction_score_increased_vs_original']:.1%} of rows"
            )

    print(f"\nPredictions: {PREDICTIONS_PATH}")
    print(f"Summary:     {SUMMARY_PATH}")
    print(f"Total prediction rows: {len(result):,}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
