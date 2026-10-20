from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REAL_PATH = Path(
    "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
)
TRAIN_PATH = Path("data/synthetic/splits/train.csv")
OUT = Path("reports/real_data_audit")
OUT.mkdir(parents=True, exist_ok=True)

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(TRAIN_PATH)

manifest_path = Path("data/processed/multichain_feature_manifest.json")
manifest = json.loads(manifest_path.read_text())
features = manifest["candidate_model_features"]

missing_real = sorted(set(features) - set(real.columns))
missing_synthetic = sorted(set(features) - set(synthetic.columns))
if missing_real or missing_synthetic:
    raise ValueError(
        f"Feature mismatch. Real missing={missing_real}; "
        f"synthetic missing={missing_synthetic}"
    )

def compare(a, b):
    a = pd.to_numeric(a, errors="raise").dropna()
    b = pd.to_numeric(b, errors="raise").dropna()
    ks = ks_2samp(a, b)
    return {
        "synthetic_median": a.median(),
        "real_median": b.median(),
        "synthetic_mean": a.mean(),
        "real_mean": b.mean(),
        "synthetic_zero_pct": 100 * (a == 0).mean(),
        "real_zero_pct": 100 * (b == 0).mean(),
        "ks_statistic": ks.statistic,
        "ks_p_value": ks.pvalue,
        "synthetic_n": len(a),
        "real_n": len(b),
    }

rows = []
for feature in features:
    result = compare(synthetic[feature], real[feature])
    rows.append({"scope": "all_real", "chain": "ALL", "feature": feature, **result})

    for chain, group in real.groupby("chain"):
        result = compare(synthetic[feature], group[feature])
        rows.append({"scope": "by_chain", "chain": chain, "feature": feature, **result})

report = pd.DataFrame(rows)
report.to_csv(OUT / "synthetic_to_real_feature_drift.csv", index=False)

overall = report[report["scope"] == "all_real"].copy()
overall = overall.sort_values("ks_statistic", ascending=False)

print("========== LARGEST OVERALL FEATURE DRIFTS ==========")
print(overall[
    ["feature", "synthetic_median", "real_median",
     "synthetic_zero_pct", "real_zero_pct", "ks_statistic"]
].head(12).round(4).to_string(index=False))

print("\n========== LARGEST DRIFTS BY CHAIN ==========")
for chain in sorted(real["chain"].unique()):
    part = report[report["chain"] == chain].sort_values(
        "ks_statistic", ascending=False
    )
    print(f"\n{chain}")
    print(part[
        ["feature", "synthetic_median", "real_median", "ks_statistic"]
    ].head(5).round(4).to_string(index=False))

print(f"\nFull report: {OUT / 'synthetic_to_real_feature_drift.csv'}")
print("Interpretation: KS statistics describe distribution differences, not attack accuracy.")
