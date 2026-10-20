from pathlib import Path
import pandas as pd

pred_path = Path("reports/phase6/chain_holdout/predictions.csv")
feat_path = Path(
    "data/processed/"
    "multichain_userop_features_20250401_20250414_processed.csv"
)
out_dir = Path("reports/phase6/error_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(pred_path)
feat = pd.read_csv(feat_path)

pred["source_id"] = pred["source_id"].astype(str).str.lower()
feat["userop_hash"] = feat["userop_hash"].astype(str).str.lower()

# Only original, unmodified benchmark records.
original = pred[pred["label"] == 0].copy()

# Join original operations to their real, unmodified feature rows.
merged = original.merge(
    feat,
    left_on=["source_id", "chain"],
    right_on=["userop_hash", "chain"],
    how="left",
    validate="many_to_one",
    indicator=True,
)

print("=== JOIN AUDIT ===")
print("Original prediction rows:", len(original))
print("Matched rows:", int((merged["_merge"] == "both").sum()))
print("Unmatched rows:", int((merged["_merge"] != "both").sum()))
print("Unique source IDs:", original["source_id"].nunique())
print("Unique feature hashes:", feat["userop_hash"].nunique())

if (merged["_merge"] != "both").any():
    print("\nSample unmatched rows:")
    print(
        merged.loc[merged["_merge"] != "both", ["source_id", "chain"]]
        .head(10)
        .to_string(index=False)
    )

merged = merged[merged["_merge"] == "both"].copy()
merged["outcome"] = merged["prediction"].map({
    1: "false_positive",
    0: "true_negative",
})

numeric = [
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
    "actual_gas_used",
    "actual_gas_cost_eth",
]

numeric = [c for c in numeric if c in merged.columns]

# Summaries by chain and outcome.
summary = merged.groupby(["chain", "outcome"])[numeric].agg(
    ["count", "median", "mean"]
)
summary.to_csv(out_dir / "original_features_by_error_outcome.csv")

print("\n=== FEATURE COMPARISON: FALSE POSITIVE VS TRUE NEGATIVE ===")
for chain in sorted(merged["chain"].dropna().unique()):
    sub = merged[merged["chain"] == chain]
    print(f"\n--- {chain} ---")

    for feature in numeric:
        fp = sub.loc[sub["outcome"] == "false_positive", feature].dropna()
        tn = sub.loc[sub["outcome"] == "true_negative", feature].dropna()

        if len(fp) == 0 or len(tn) == 0:
            continue

        print(
            f"{feature:30s} "
            f"FP median={fp.median():.4g}, mean={fp.mean():.4g} | "
            f"TN median={tn.median():.4g}, mean={tn.mean():.4g}"
        )

print(f"\nSaved summary: {out_dir / 'original_features_by_error_outcome.csv'}")
