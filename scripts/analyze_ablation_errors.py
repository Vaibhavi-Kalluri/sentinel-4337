from pathlib import Path
import pandas as pd

ROOT = Path("reports/real_background_benchmark/ablation")
df = pd.read_csv(ROOT / "test_predictions.csv")

# Only compare the two strongest configurations.
df = df[df["configuration"].isin(
    ["count_only", "full_17_features"]
)].copy()

df["correct"] = df["label"].eq(df["prediction"])
df["false_positive"] = (
    df["label"].eq(0) & df["prediction"].eq(1)
)
df["false_negative"] = (
    df["label"].eq(1) & df["prediction"].eq(0)
)

# Chain-specific metrics.
chain_rows = []
for (model, chain), g in df.groupby(["configuration", "chain"]):
    negatives = g[g["label"] == 0]
    positives = g[g["label"] == 1]
    chain_rows.append({
        "model": model,
        "chain": chain,
        "n": len(g),
        "false_positives": int(g["false_positive"].sum()),
        "FPR": (
            g["false_positive"].sum() / len(negatives)
            if len(negatives) else 0
        ),
        "false_negatives": int(g["false_negative"].sum()),
        "recall": (
            1 - g["false_negative"].sum() / len(positives)
            if len(positives) else 0
        ),
    })

chain = pd.DataFrame(chain_rows)
chain.to_csv(ROOT / "chain_error_analysis.csv", index=False)

# Recall by perturbation type.
attack = df[df["label"] == 1]
scenario = (
    attack.groupby(["configuration", "modification_type"])
    .agg(
        n=("label", "size"),
        missed=("false_negative", "sum"),
        mean_score=("score", "mean"),
    )
    .reset_index()
)
scenario["recall"] = 1 - scenario["missed"] / scenario["n"]
scenario.to_csv(ROOT / "perturbation_error_analysis.csv", index=False)

print("\n=== CHAIN-SPECIFIC ERRORS ===")
print(chain.round(4).to_string(index=False))

print("\n=== PERTURBATION-SPECIFIC ERRORS ===")
print(scenario.round(4).to_string(index=False))

print(f"\nSaved tables to {ROOT}")
