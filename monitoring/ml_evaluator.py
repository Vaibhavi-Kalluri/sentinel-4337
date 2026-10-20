#!/usr/bin/env python3
"""Inference-only Sentinel 4337 Random Forest evaluator.

Uses the existing trained model. Does not train, submit transactions,
or enforce policy on-chain.
"""

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/sentinel_real_background_injected_rf.joblib")

FEATURES = [
    "trace_count",
    "unique_destinations",
    "approve_trace_count",
    "transfer_from_trace_count",
    "value_transfer_trace_count",
    "errored_trace_count",
    "has_approval",
    "has_transfer_from",
    "has_value_transfer",
    "has_trace_errors",
    "has_paymaster",
    "log1p_trace_count",
    "log1p_unique_destinations",
    "log1p_approve_trace_count",
    "log1p_transfer_from_trace_count",
    "log1p_value_transfer_trace_count",
    "log1p_errored_trace_count",
]

COUNT_FEATURES = [
    "trace_count",
    "unique_destinations",
    "approve_trace_count",
    "transfer_from_trace_count",
    "value_transfer_trace_count",
    "errored_trace_count",
]

FLAG_FEATURES = [
    "has_approval",
    "has_transfer_from",
    "has_value_transfer",
    "has_trace_errors",
    "has_paymaster",
]


def extract_features(operation: dict) -> dict:
    """Validate supplied trace-derived features and create log features.

    This function intentionally does not invent missing trace data.
    """
    missing = [name for name in COUNT_FEATURES + FLAG_FEATURES
               if name not in operation]
    if missing:
        raise ValueError(
            "Missing trace-derived features: " + ", ".join(missing)
        )

    features = {}

    for name in COUNT_FEATURES:
        value = operation[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a numeric count")
        if not math.isfinite(value) or value < 0 or not float(value).is_integer():
            raise ValueError(f"{name} must be a non-negative integer")
        features[name] = int(value)

    for name in FLAG_FEATURES:
        value = operation[name]
        if value not in (True, False, 0, 1):
            raise ValueError(f"{name} must be boolean or 0/1")
        features[name] = int(value)

    # Keep flag fields consistent with the corresponding counts.
    consistency = {
        "has_approval": "approve_trace_count",
        "has_transfer_from": "transfer_from_trace_count",
        "has_value_transfer": "value_transfer_trace_count",
        "has_trace_errors": "errored_trace_count",
    }
    for flag, count in consistency.items():
        if features[flag] != int(features[count] > 0):
            raise ValueError(
                f"{flag} conflicts with {count}; check feature extraction"
            )

    features["log1p_trace_count"] = math.log1p(
        features["trace_count"]
    )
    features["log1p_unique_destinations"] = math.log1p(
        features["unique_destinations"]
    )
    features["log1p_approve_trace_count"] = math.log1p(
        features["approve_trace_count"]
    )
    features["log1p_transfer_from_trace_count"] = math.log1p(
        features["transfer_from_trace_count"]
    )
    features["log1p_value_transfer_trace_count"] = math.log1p(
        features["value_transfer_trace_count"]
    )
    features["log1p_errored_trace_count"] = math.log1p(
        features["errored_trace_count"]
    )

    return {name: features[name] for name in FEATURES}


def evaluate(operation: dict, model=None) -> dict:
    if not isinstance(operation, dict):
        raise ValueError("Input JSON must be an object")

    features = extract_features(operation)

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    expected = getattr(model, "n_features_in_", None)
    if expected != len(FEATURES):
        raise ValueError(
            f"Model expects {expected} features, "
            f"but evaluator supplies {len(FEATURES)}"
        )

    feature_frame = pd.DataFrame([features], columns=FEATURES)
    score = float(model.predict_proba(feature_frame)[0, 1])

    # This is a model score, NOT a calibrated probability of an attack.
    return {
        "sender": operation.get("sender"),
        "chain": operation.get("chain"),
        "model": MODEL_PATH.name,
        "feature_count": len(FEATURES),
        "features": features,
        "risk_score": round(score, 6),
        "score_interpretation": (
            "Random Forest score for controlled synthetic perturbation "
            "patterns; not a calibrated attack probability."
        ),
        "recommendation": "REVIEW" if score >= 0.5 else "NO_ML_ALERT",
        "recommendation_is_advisory": True,
        "enforces_on_chain": False,
        "limitations": [
            "Requires trace-derived features from a trusted extractor.",
            "Trained/evaluated using controlled feature perturbations.",
            "Not validated as a real-world maliciousness probability.",
            "The ML result does not replace SentinelPolicy enforcement.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the existing Sentinel Random Forest."
    )
    parser.add_argument("input", type=Path, help="Input JSON feature file")
    args = parser.parse_args()

    try:
        operation = json.loads(args.input.read_text(encoding="utf-8"))
        result = evaluate(operation)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
