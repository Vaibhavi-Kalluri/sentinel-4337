from pathlib import Path
import pandas as pd

path = Path(
    "data/processed/"
    "multichain_userop_features_20250401_20250414_processed.csv"
)
out = Path("reports/phase6/feature_audit")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(path)

features = [
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
]

missing = [c for c in ["chain", *features] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

rows = []
for chain, g in df.groupby("chain"):
    for feature in features:
        s = g[feature].dropna()
        rows.append({
            "chain": chain,
            "feature": feature,
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "p95": s.quantile(0.95),
            "zero_fraction": (s == 0).mean(),
        })

summary = pd.DataFrame(rows)
summary.to_csv(out / "original_features_by_chain.csv", index=False)

for feature in features:
    print(f"\n=== {feature} ===")
    print(
        summary[summary["feature"] == feature][
            ["chain", "mean", "median", "p95", "zero_fraction"]
        ].round(3).to_string(index=False)
    )

print(f"\nSaved: {out / 'original_features_by_chain.csv'}")
