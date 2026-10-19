import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
MANIFEST = ROOT / "data/processed/multichain_feature_manifest.json"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
manifest = json.loads(MANIFEST.read_text())

print("\n========== DATASET OVERVIEW ==========")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Unique operation hashes: {df['userop_hash'].nunique():,}")
print(f"Unique senders: {df['sender'].nunique():,}")
print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")
print(f"Chains: {sorted(df['chain'].dropna().unique())}")
print(f"Missing cells: {int(df.isna().sum().sum())}")

print("\n========== OPERATIONS PER CHAIN ==========")
chain_summary = df.groupby("chain").agg(
    operations=("userop_hash", "count"),
    unique_senders=("sender", "nunique"),
    success_rate=("success", "mean"),
    mean_trace_count=("trace_count", "mean"),
    median_trace_count=("trace_count", "median"),
    mean_gas_used=("actual_gas_used", "mean"),
    approval_operations=("has_approval", "sum"),
    transfer_from_operations=("has_transfer_from", "sum"),
    value_transfer_operations=("has_value_transfer", "sum"),
    operations_with_errors=("has_trace_errors", "sum"),
    paymaster_operations=("has_paymaster", "sum"),
).reset_index()

for col in ["success_rate"]:
    chain_summary[col] = (chain_summary[col] * 100).round(2)

print(chain_summary.to_string(index=False))
chain_summary.to_csv(OUT / "multichain_by_chain.csv", index=False)

print("\n========== OPERATIONS PER DAY ==========")
daily = df.groupby(["dt", "chain"]).agg(
    operations=("userop_hash", "count"),
    success_rate=("success", "mean"),
    mean_trace_count=("trace_count", "mean"),
).reset_index()
daily["success_rate"] = (daily["success_rate"] * 100).round(2)
print(daily.head(18).to_string(index=False))
daily.to_csv(OUT / "multichain_by_day_and_chain.csv", index=False)

print("\n========== FEATURE DISTRIBUTIONS ==========")
features = [
    "trace_count",
    "unique_destinations",
    "approve_trace_count",
    "transfer_from_trace_count",
    "value_transfer_trace_count",
    "errored_trace_count",
    "actual_gas_used",
    "actual_gas_cost_eth",
]
available = [c for c in features if c in df.columns]
print(df[available].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(6).to_string())
df[available].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(
    OUT / "multichain_feature_summary.csv"
)

print("\n========== OUTCOME COUNTS (NOT MALICIOUSNESS LABELS) ==========")
print(df["success"].value_counts(dropna=False).to_string())
print("\n========== INDICATOR COUNTS ==========")
for col in [
    "has_approval",
    "has_transfer_from",
    "has_value_transfer",
    "has_trace_errors",
    "has_paymaster",
]:
    if col in df.columns:
        print(f"{col}: {int(df[col].sum()):,} / {len(df):,} "
              f"({df[col].mean() * 100:.2f}%)")

print("\n========== DATA LIMITATIONS ==========")
print("Verified maliciousness labels available:",
      manifest.get("source_maliciousness_labels_available"))
print("Execution success used as maliciousness label:",
      manifest.get("success_is_maliciousness_label"))
print("Sampling is balanced by chain and day; it is not proportional to chain activity.")

print("\nReports saved under reports/:")
for name in [
    "multichain_by_chain.csv",
    "multichain_by_day_and_chain.csv",
    "multichain_feature_summary.csv",
]:
    print(f"  {name}")
