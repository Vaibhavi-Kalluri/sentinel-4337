from pathlib import Path
import pandas as pd

src = Path("reports/phase6/chain_holdout/predictions.csv")
out = Path("reports/phase6/error_analysis")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)

required = {
    "source_id", "chain", "label",
    "modification_type", "score", "prediction"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["label"] = df["label"].astype(int)
df["prediction"] = df["prediction"].astype(int)

# 1. Discover the modification categories in the saved data.
print("\n=== MODIFICATION TYPES ===")
print(df.groupby(["label", "modification_type"]).size().to_string())

# 2. Errors by chain and ground-truth benchmark label.
rows = []
for (chain, label), g in df.groupby(["chain", "label"]):
    n = len(g)
    correct = (g["prediction"] == g["label"]).sum()
    rows.append({
        "chain": chain,
        "label": label,
        "n": n,
        "predicted_positive": int(g["prediction"].sum()),
        "correct": int(correct),
        "error_rate": float((g["prediction"] != g["label"]).mean()),
        "mean_score": float(g["score"].mean()),
        "median_score": float(g["score"].median()),
        "p95_score": float(g["score"].quantile(0.95)),
    })

by_class = pd.DataFrame(rows)
by_class.to_csv(out / "errors_by_chain_and_label.csv", index=False)

print("\n=== ERRORS BY CHAIN AND LABEL ===")
print(by_class.round(4).to_string(index=False))

# 3. Detection performance for each injected perturbation type.
positive = df[df["label"] == 1].copy()
rows = []

for (chain, kind), g in positive.groupby(["chain", "modification_type"]):
    tp = int((g["prediction"] == 1).sum())
    fn = int((g["prediction"] == 0).sum())
    rows.append({
        "chain": chain,
        "perturbation": kind,
        "n": len(g),
        "recall": tp / len(g) if len(g) else 0.0,
        "false_negatives": fn,
        "mean_score": float(g["score"].mean()),
        "median_score": float(g["score"].median()),
    })

by_perturbation = pd.DataFrame(rows)
by_perturbation.to_csv(
    out / "recall_by_chain_and_perturbation.csv", index=False
)

print("\n=== RECALL BY CHAIN AND PERTURBATION ===")
print(by_perturbation.round(4).to_string(index=False))

# 4. False positives among unmodified original records.
negative = df[df["label"] == 0].copy()
fp_rows = []

for chain, g in negative.groupby("chain"):
    fp = int(g["prediction"].sum())
    tn = len(g) - fp
    fp_rows.append({
        "chain": chain,
        "original_records": len(g),
        "false_positives": fp,
        "true_negatives": tn,
        "fpr": fp / len(g) if len(g) else 0.0,
        "mean_score": float(g["score"].mean()),
        "median_score": float(g["score"].median()),
        "p95_score": float(g["score"].quantile(0.95)),
    })

fp_df = pd.DataFrame(fp_rows)
fp_df.to_csv(out / "false_positives_by_chain.csv", index=False)

print("\n=== FALSE POSITIVES BY CHAIN ===")
print(fp_df.round(4).to_string(index=False))

print(f"\nReports saved to: {out}")
