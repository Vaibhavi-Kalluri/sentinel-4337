from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

DATA = Path(
    "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
)
MODEL = Path("models/e0_features_only_e1_split.joblib")
OUT = Path("reports/real_data_audit")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
bundle = joblib.load(MODEL)

if not isinstance(bundle, dict) or "features" not in bundle:
    raise ValueError("Unexpected model bundle format; inspect the saved model first.")

features = bundle["features"]

# Find the estimator in the saved bundle without assuming its key name.
estimators = [
    (key, value)
    for key, value in bundle.items()
    if hasattr(value, "predict_proba")
]
if len(estimators) != 1:
    print("Bundle keys:", list(bundle.keys()))
    print("Probability estimators found:", [key for key, _ in estimators])
    raise ValueError("Could not uniquely identify the saved classifier.")

model_key, model = estimators[0]
missing = sorted(set(features) - set(df.columns))
if missing:
    raise ValueError(f"Missing required model features: {missing}")

X = df[features].apply(pd.to_numeric, errors="raise")
if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Model features contain NaN or infinite values.")

# Locate the positive-class probability explicitly.
classes = list(model.classes_)
if 1 not in classes:
    raise ValueError(f"Expected positive class 1; found classes {classes}")

positive_index = classes.index(1)
scores = model.predict_proba(X)[:, positive_index]

# Scores are model outputs, not calibrated probabilities of real-world attacks.
scored = df.copy()
scored["e0_synthetic_model_score"] = scores
scored["score_band"] = pd.cut(
    scores,
    bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
    labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    right=False,
)
scored["score_band"] = scored["score_band"].astype(str)

scored.to_csv(OUT / "scored_operations.csv", index=False)

def summarize(group):
    s = group["e0_synthetic_model_score"]
    return pd.Series({
        "operations": len(group),
        "mean_score": s.mean(),
        "median_score": s.median(),
        "p90_score": s.quantile(0.90),
        "p95_score": s.quantile(0.95),
        "p99_score": s.quantile(0.99),
        "score_ge_0_5": int((s >= 0.5).sum()),
        "score_ge_0_8": int((s >= 0.8).sum()),
        "score_ge_0_5_pct": 100 * (s >= 0.5).mean(),
        "score_ge_0_8_pct": 100 * (s >= 0.8).mean(),
    })

by_chain = scored.groupby("chain", observed=True).apply(
    summarize, include_groups=False
).reset_index()
by_chain.to_csv(OUT / "scores_by_chain.csv", index=False)

if "dt" in scored.columns:
    by_day_chain = scored.groupby(["dt", "chain"], observed=True).apply(
        summarize, include_groups=False
    ).reset_index()
    by_day_chain.to_csv(OUT / "scores_by_day_and_chain.csv", index=False)

band_counts = (
    scored.groupby(["chain", "score_band"], observed=False)
    .size().rename("operations").reset_index()
)
band_counts.to_csv(OUT / "score_bands_by_chain.csv", index=False)

# Outcome is reported separately as operational metadata only, never as a safety label.
if "success" in scored.columns:
    outcome = scored.groupby(["chain", "success"], observed=True).agg(
        operations=("e0_synthetic_model_score", "size"),
        mean_model_score=("e0_synthetic_model_score", "mean"),
        median_model_score=("e0_synthetic_model_score", "median"),
    ).reset_index()
    outcome.to_csv(OUT / "scores_by_chain_and_execution_outcome.csv", index=False)

manifest = {
    "dataset": str(DATA),
    "model": str(MODEL),
    "model_estimator_key": model_key,
    "rows_scored": len(scored),
    "feature_count": len(features),
    "features": list(features),
    "chains": sorted(scored["chain"].dropna().unique().tolist()),
    "score_definition": (
        "Positive-class score from a random forest trained on synthetic data. "
        "Not calibrated or validated as a real-world attack probability."
    ),
    "label_limitations": (
        "Real data has no verified maliciousness labels. Execution success is "
        "not a maliciousness label. No real-world precision, recall, or accuracy "
        "is calculated."
    ),
    "score_thresholds_are_descriptive_only": True,
}
(OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2))

print(f"Scored {len(scored):,} operations using {len(features)} features.")
print(f"Model estimator key: {model_key}")
print("\nScore summary:")
print(scored["e0_synthetic_model_score"].describe(
    percentiles=[0.5, 0.9, 0.95, 0.99]
).round(4).to_string())
print("\nBy chain:")
print(by_chain.round(4).to_string(index=False))
print(f"\nReports saved under: {OUT}")
