from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

src = Path("reports/phase6/chain_holdout/predictions.csv")
out = Path("reports/phase6/error_analysis")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(src)
rng_seed = 4337
rows = []

for chain, g in df.groupby("chain"):
    # Split by source ID, so all versions of a source stay together.
    source_ids = g["source_id"].drop_duplicates().to_numpy()
    cal_ids, test_ids = train_test_split(
        source_ids, test_size=0.5, random_state=rng_seed
    )

    cal = g[g["source_id"].isin(cal_ids)]
    test = g[g["source_id"].isin(test_ids)]

    cal_neg = cal.loc[cal["label"] == 0, "score"]
    test_neg = test.loc[test["label"] == 0]
    test_pos = test.loc[test["label"] == 1]

    if len(cal_neg) == 0 or len(test_neg) == 0 or len(test_pos) == 0:
        raise ValueError(f"Missing class for chain {chain}")

    # Threshold chosen on calibration negatives only.
    threshold = float(cal_neg.quantile(0.95))

    fp = int((test_neg["score"] >= threshold).sum())
    tn = len(test_neg) - fp
    tp = int((test_pos["score"] >= threshold).sum())
    fn = len(test_pos) - tp

    rows.append({
        "chain": chain,
        "calibration_sources": len(cal_ids),
        "test_sources": len(test_ids),
        "threshold": threshold,
        "test_fpr": fp / len(test_neg),
        "test_injected_recall": tp / len(test_pos),
        "test_precision": tp / (tp + fp) if tp + fp else 0.0,
        "test_fp": fp,
        "test_fn": fn,
        "test_negative_n": len(test_neg),
        "test_positive_n": len(test_pos),
    })

result = pd.DataFrame(rows).sort_values("chain")
print(result.round(4).to_string(index=False))

path = out / "calibration_validation.csv"
result.to_csv(path, index=False)
print(f"\nSaved: {path}")
print(
    "\nCaveat: this validates thresholds on held-out source groups, "
    "but the underlying model was trained in the existing chain-holdout "
    "experiment. Results remain specific to the injected benchmark."
)
