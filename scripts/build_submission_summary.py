from pathlib import Path
import json
import pandas as pd

root = Path("reports/real_background_benchmark")
ablation = root / "ablation"

results = json.loads((ablation / "results.json").read_text())
rows = []
for item in results:
    if item.get("status") != "ok":
        continue
    m = item["test_metrics"]
    rows.append({
        "Model": item["configuration"],
        "Features": item["feature_count"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "PR-AUC": m["pr_auc_average_precision"],
        "ROC-AUC": m["roc_auc"],
        "FPR": m["fpr"],
    })

table = pd.DataFrame(rows)
lines = [
    "# Sentinel 4337: Experimental Results",
    "",
    "## Experimental setup",
    "",
    "- Source background: 42,000 UserOperation feature records across six chains.",
    "- Injected benchmark: 84,000 rows, consisting of 42,000 original records and 42,000 controlled feature perturbations.",
    "- Split: 60/20/20 by source ID, stratified within each chain. Original and perturbed copies remain in the same split.",
    "- Model: Random Forest, 400 trees, minimum leaf size 2, balanced class weights.",
    "- Thresholds: selected on validation data; test set used for final evaluation.",
    "- Positive label: controlled synthetic feature perturbation, not verified maliciousness.",
    "",
    "## Ablation at validation FPR target of 5%",
    "",
    "| " + " | ".join(table.columns) + " |",
    "| " + " | ".join(["---"] * len(table.columns)) + " |",
    *[
        "| " + " | ".join(
            f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            for v in row
        ) + " |"
        for row in table.itertuples(index=False, name=None)
    ],
    "",
    "## Error analysis",
    "",
    "The full 17-feature model's test FPR is approximately 13% on Base,",
    "compared with lower rates on the other sampled chains. This indicates",
    "that aggregate performance can hide chain-specific errors.",
    "",
    "Multi-call perturbations are the hardest category for the full model:",
    "381 of 1,681 test examples were missed (recall approximately 77.33%).",
    "",
    "## Limitations",
    "",
    "These results evaluate detection of controlled feature-space perturbations.",
    "They do not measure real-world attack detection accuracy. Original records",
    "are not confirmed benign, and positive examples are not verified attacks.",
    "The benchmark does not reconstruct modified transaction traces. More",
    "independently labelled security examples and realistic attack simulations",
    "are required before claiming production readiness.",
    "",
    "## Current conclusion",
    "",
    "The full feature model is the strongest tested configuration by F1 and",
    "ranking metrics, while the count-only model is a simpler competitive",
    "baseline. The indicator-only model loses substantial recall. The next",
    "research priority is independently labelled validation, not further",
    "optimization on this injected benchmark.",
]
out = root / "SUBMISSION_RESULTS.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Created {out}")
print(table.to_string(index=False))
