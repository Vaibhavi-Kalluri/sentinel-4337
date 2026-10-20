import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
OUT = ROOT / "reports/phase6/operation_metadata_audit.json"

df = pd.read_csv(DATA)
required = {
    "chain", "userop_hash", "destination_addresses",
    "method_selectors", "success"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

def parse_list(value):
    if pd.isna(value):
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []

destination_counts = Counter()
selector_counts = Counter()
per_chain = {}

for _, row in df.iterrows():
    destinations = parse_list(row["destination_addresses"])
    selectors = parse_list(row["method_selectors"])

    destination_counts.update(
        str(x).lower() for x in destinations if x
    )
    selector_counts.update(
        str(x).lower() for x in selectors if x
    )

for chain, group in df.groupby("chain"):
    per_chain[str(chain)] = {
        "operations": int(len(group)),
        "unique_senders": int(group["sender"].nunique())
            if "sender" in group else None,
        "success_rate": round(float(group["success"].mean()), 4),
        "operations_with_metadata": int(
            (
                group["destination_addresses"].notna()
                & group["method_selectors"].notna()
            ).sum()
        ),
    }

def top(counter, n=25):
    return [
        {"value": value, "occurrences": count}
        for value, count in counter.most_common(n)
    ]

# Known ERC-20 selectors are useful for reconnaissance, not verdicts.
known_selectors = {
    "0x095ea7b3": "approve(address,uint256)",
    "0xa9059cbb": "transfer(address,uint256)",
    "0x23b872dd": "transferFrom(address,address,uint256)",
    "0x70a08231": "balanceOf(address)",
    "0x18160ddd": "totalSupply()",
}

selector_rows = []
for selector, count in selector_counts.most_common():
    selector_rows.append({
        "selector": selector,
        "occurrences": count,
        "known_signature": known_selectors.get(selector),
    })

report = {
    "dataset": str(DATA.relative_to(ROOT)),
    "operations": int(len(df)),
    "unique_operation_hashes": int(df["userop_hash"].nunique()),
    "unique_destinations_across_lists": len(destination_counts),
    "unique_selectors_across_lists": len(selector_counts),
    "chains": per_chain,
    "top_destinations": top(destination_counts),
    "top_selectors": selector_rows[:100],
    "known_selector_occurrences": {
        selector: {
            "signature": signature,
            "occurrences": selector_counts[selector],
        }
        for selector, signature in known_selectors.items()
    },
    "limitations": [
        "Selector recognition does not establish intent or maliciousness.",
        "Unknown selectors may be legitimate custom functions.",
        "Destination frequency alone is not a safety label.",
        "This report does not verify contract code or transaction traces.",
        "Execution success is not a maliciousness label.",
    ],
}

OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Saved: {OUT}")
print(f"Operations: {len(df):,}")
print(f"Unique destinations: {len(destination_counts):,}")
print(f"Unique selectors: {len(selector_counts):,}")

print("\nTop selectors:")
for item in report["top_selectors"][:20]:
    signature = item["known_signature"] or "unclassified"
    print(f"{item['selector']:12} {item['occurrences']:7,}  {signature}")

print("\nKnown selector counts:")
for selector, item in report["known_selector_occurrences"].items():
    print(f"{selector} {item['signature']}: {item['occurrences']:,}")
