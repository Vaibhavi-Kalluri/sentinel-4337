from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

print("\n========== 1. EXTREME GAS VALUES ==========")
gas_cols = ["actual_gas_used", "actual_gas_cost_eth"]

for col in gas_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"\n{col}")
    print(df[col].describe(percentiles=[.5, .9, .95, .99, .999]).to_string())
    print("\nTop 10 observations:")
    cols = [c for c in [
        "dt", "chain", "userop_hash", "sender",
        "success", col, "trace_count", "unique_destinations"
    ] if c in df.columns]
    print(df.nlargest(10, col)[cols].to_string(index=False))

print("\n========== 2. GAS DISTRIBUTION BY CHAIN ==========")
gas_by_chain = df.groupby("chain").agg(
    operations=("userop_hash", "count"),
    gas_median=("actual_gas_used", "median"),
    gas_p95=("actual_gas_used", lambda s: s.quantile(.95)),
    gas_p99=("actual_gas_used", lambda s: s.quantile(.99)),
    gas_max=("actual_gas_used", "max"),
    cost_median_eth=("actual_gas_cost_eth", "median"),
    cost_p99_eth=("actual_gas_cost_eth", lambda s: s.quantile(.99)),
).round(8)
print(gas_by_chain.to_string())
gas_by_chain.to_csv(OUT / "gas_quality_by_chain.csv")

print("\n========== 3. SENDERS SHARED ACROSS CHAINS ==========")
sender_chains = df.groupby("sender")["chain"].nunique()
print("Unique senders:", len(sender_chains))
print("Senders appearing on multiple chains:", int((sender_chains > 1).sum()))
print("Senders appearing on 3+ chains:", int((sender_chains >= 3).sum()))

print("\nTop senders by operation count:")
sender_counts = df.groupby("sender").agg(
    operations=("userop_hash", "count"),
    chains=("chain", "nunique"),
).sort_values("operations", ascending=False)
print(sender_counts.head(15).to_string())
sender_counts.head(100).to_csv(OUT / "top_sender_activity.csv")

print("\n========== 4. OPERATION HASH OVERLAP ACROSS CHAINS ==========")
hash_chain_count = df.groupby("userop_hash")["chain"].nunique()
print("Hashes appearing on multiple chains:", int((hash_chain_count > 1).sum()))
print("Duplicate chain/hash pairs:", int(df.duplicated(["chain", "userop_hash"]).sum()))

print("\n========== 5. SUCCESS AND FAILURE BY CHAIN ==========")
outcomes = pd.crosstab(df["chain"], df["success"], margins=True)
print(outcomes.to_string())

print("\n========== 6. HIGH TRACE COUNT CHECK ==========")
trace_cutoff = df["trace_count"].quantile(.99)
high_trace = df[df["trace_count"] >= trace_cutoff]
print(f"99th percentile trace count: {trace_cutoff}")
print("Rows at or above cutoff:", len(high_trace))
print("\nBy chain:")
print(high_trace["chain"].value_counts().to_string())

print("\n========== 7. FINISHED ==========")
print("Reports saved in reports/:")
print("  gas_quality_by_chain.csv")
print("  top_sender_activity.csv")
print("Reminder: execution failure and unusual activity are not proof of maliciousness.")
