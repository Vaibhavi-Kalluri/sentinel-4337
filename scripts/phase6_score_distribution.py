from pathlib import Path
import numpy as np
import pandas as pd

src = Path("reports/phase6/chain_holdout/predictions.csv")
out = Path("reports/phase6/score_distribution")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
required = {"chain", "label", "score"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

rows = []
for chain, group in df.groupby("chain"):
    for label, part in group.groupby("label"):
        scores = part["score"].astype(float)
        rows.append({
            "chain": chain,
            "label": int(label),
            "class_name": (
                "unmodified_background" if label == 0
                else "synthetic_perturbation"
            ),
            "n": len(scores),
            "mean": scores.mean(),
            "median": scores.median(),
            "p05": scores.quantile(0.05),
            "p25": scores.quantile(0.25),
            "p75": scores.quantile(0.75),
            "p95": scores.quantile(0.95),
            "fraction_ge_0_5": (scores >= 0.5).mean(),
            "fraction_ge_0_7": (scores >= 0.7).mean(),
            "fraction_ge_0_9": (scores >= 0.9).mean(),
        })

result = pd.DataFrame(rows)
result.to_csv(out / "score_quantiles_by_chain_and_label.csv", index=False)

for chain in sorted(df["chain"].unique()):
    g = result[result["chain"] == chain].set_index("class_name")
    benign = g.loc["unmodified_background"]
    perturbed = g.loc["synthetic_perturbation"]

    print(f"\n{chain.upper()}")
    print(
        f"Original scores:   median={benign['median']:.3f}, "
        f"p95={benign['p95']:.3f}, "
        f"mean={benign['mean']:.3f}"
    )
    print(
        f"Perturbed scores:  median={perturbed['median']:.3f}, "
        f"p05={perturbed['p05']:.3f}, "
        f"median={perturbed['median']:.3f}, "
        f"mean={perturbed['mean']:.3f}"
    )

print(f"\nSaved: {out / 'score_quantiles_by_chain_and_label.csv'}")
