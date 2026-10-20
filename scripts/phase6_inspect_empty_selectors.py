import ast
import json
from pathlib import Path

import pandas as pd

SOURCE = Path(
    "data/processed/"
    "multichain_userop_features_20250401_20250414_processed.csv"
)
OUT = Path("reports/phase6/empty_selector_examples.json")

df = pd.read_csv(SOURCE)

def parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            result = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return ["__PARSE_ERROR__"]
    return result if isinstance(result, list) else ["__NON_LIST__"]

selectors = df["method_selectors"].map(parse_list)

mask = selectors.map(
    lambda items: any(str(x).strip().lower() == "0x" for x in items)
)

selected = df.loc[mask].copy()
selected["parsed_selectors"] = selectors.loc[mask].map(
    lambda x: [str(v) for v in x]
)

print("Rows containing literal 0x:", len(selected))
print("\nCounts by chain:")
print(selected.groupby("chain").size().to_string())

print("\nExamples by chain:")
examples = []
for chain, group in selected.groupby("chain"):
    print(f"\n--- {chain}: {len(group)} rows ---")
    for _, row in group.head(5).iterrows():
        selectors_value = row["parsed_selectors"]
        destinations = parse_list(row.get("destination_addresses"))
        example = {
            "chain": str(chain),
            "userop_hash": str(row.get("userop_hash", "")),
            "sender": str(row.get("sender", "")),
            "success": str(row.get("success", "")),
            "parsed_selectors": selectors_value,
            "destination_addresses_sample": destinations[:5],
            "trace_count": str(row.get("trace_count", "")),
        }
        examples.append(example)
        print(json.dumps(example, ensure_ascii=False))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "source": str(SOURCE),
    "rows_with_literal_0x": int(len(selected)),
    "counts_by_chain": {
        str(k): int(v)
        for k, v in selected.groupby("chain").size().items()
    },
    "examples": examples,
    "limitations": [
        "A literal 0x value alone does not prove calldata was empty.",
        "Processed metadata is not independent verification of raw transactions.",
        "No maliciousness labels are inferred."
    ]
}, indent=2) + "\n")

print("\nSaved:", OUT)
