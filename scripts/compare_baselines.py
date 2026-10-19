import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic/splits"
REPORT = ROOT / "reports"
REPORT.mkdir(exist_ok=True)

test = pd.read_csv(DATA / "test.csv")
pred = pd.read_csv(REPORT / "synthetic_test_predictions.csv")

assert len(test) == len(pred)
assert test["label"].equals(pred["label"])
assert test["scenario_type"].equals(pred["scenario_type"])

y = test["label"].astype(int)
rf_probability = pred["attack_like_probability"].astype(float)
rf_prediction = pred["predicted_label"].astype(int)

# Transparent heuristic: flag an operation if it contains any of
# these transfer/approval signals. This is intentionally simple,
# not a recommended production security policy.
rule_prediction = (
    (test["has_value_transfer"] > 0)
    | (test["has_transfer_from"] > 0)
    | (test["has_approval"] > 0)
).astype(int)

def summarize(name, truth, predicted, score=None):
    tn, fp, fn, tp = confusion_matrix(
        truth, predicted, labels=[0, 1]
    ).ravel()

    result = {
        "model": name,
        "precision": float(precision_score(
            truth, predicted, zero_division=0
        )),
        "recall": float(recall_score(
            truth, predicted, zero_division=0
        )),
        "f1": float(f1_score(
            truth, predicted, zero_division=0
        )),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn))
            if (fp + tn) else None,
    }

    if score is not None:
        result["average_precision_pr_auc"] = float(
            average_precision_score(truth, score)
        )
        result["roc_auc"] = float(roc_auc_score(truth, score))

    return result

results = [
    summarize(
        "Random Forest (validation-selected threshold)",
        y, rf_prediction, rf_probability
    ),
    summarize(
        "Simple approval/transfer rule",
        y, rule_prediction
    ),
]

print("\n=== SYNTHETIC TEST-SET BASELINE COMPARISON ===")
print(pd.DataFrame(results).to_string(index=False))

print("\n=== RULE RESULTS BY SCENARIO ===")
for scenario, indices in test.groupby("scenario_type").groups.items():
    truth = y.loc[indices]
    predicted = rule_prediction.loc[indices]
    print(
        f"{scenario:28} "
        f"n={len(indices):3} "
        f"recall={recall_score(truth, predicted, zero_division=0):.3f} "
        f"false_positives={((
            (truth == 0) & (predicted == 1)
        ).sum())}"
    )

out = REPORT / "synthetic_baseline_comparison.json"
out.write_text(json.dumps(results, indent=2) + "\n")
print(f"\nSaved: {out.relative_to(ROOT)}")
print("\nReminder: these results measure agreement with synthetic scenario labels,")
print("not verified real-world maliciousness.")
