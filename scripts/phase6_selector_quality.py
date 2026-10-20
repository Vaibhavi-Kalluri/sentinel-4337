import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SOURCE = Path(
    "data/processed/"
    "multichain_userop_features_20250401_20250414_processed.csv"
)
OUT = Path("reports/phase6/selector_quality_audit.json")

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

known = {
    "0x095ea7b3": "approve(address,uint256)",
    "0xa9059cbb": "transfer(address,uint256)",
    "0x23b872dd": "transferFrom(address,address,uint256)",
    "0x70a08231": "balanceOf(address)",
    "0x18160ddd": "totalSupply()",
}

overall = Counter()
by_chain = defaultdict(Counter)
selector_frequency = Counter()
chain_selector_frequency = defaultdict(Counter)
parse_errors = Counter()
empty_selector_ops = Counter()
known_selector_ops = Counter()
no_selector_ops = Counter()
metadata_lengths = Counter()

for _, row in df.iterrows():
    chain = str(row["chain"])
    selectors = parse_list(row.get("method_selectors"))
    destinations = parse_list(row.get("destination_addresses"))

    if "__PARSE_ERROR__" in selectors:
        parse_errors[chain] += 1
        continue

    normalized = [
        str(s).strip().lower()
        for s in selectors
        if s is not None
    ]

    overall["operations"] += 1
    overall["selector_entries"] += len(normalized)
    overall["destination_entries"] += len(destinations)
    metadata_lengths[str(len(normalized))] += 1

    if not normalized or all(s in ("", "0x", "0x0") for s in normalized):
        no_selector_ops[chain] += 1

    if any(s == "0x" for s in normalized):
        empty_selector_ops[chain] += 1

    for selector in normalized:
        selector_frequency[selector] += 1
        chain_selector_frequency[chain][selector] += 1

        if selector in known:
            known_selector_ops[chain] += 1

        by_chain[chain]["selector_entries"] += 1

    by_chain[chain]["operations"] += 1
    by_chain[chain]["destination_entries"] += len(destinations)
    by_chain[chain]["unique_senders"] = df.loc[
        df["chain"].astype(str) == chain, "sender"
    ].nunique()

chain_summary = {}

for chain in sorted(df["chain"].astype(str).unique()):
    chain_summary[chain] = {
        "operations": int(by_chain[chain]["operations"]),
        "selector_entries": int(by_chain[chain]["selector_entries"]),
        "destination_entries": int(by_chain[chain]["destination_entries"]),
        "unique_senders": int(by_chain[chain]["unique_senders"]),
        "operations_with_literal_empty_0x": int(empty_selector_ops[chain]),
        "operations_with_no_or_empty_selector": int(no_selector_ops[chain]),
        "operations_containing_known_selector": int(known_selector_ops[chain]),
        "top_10_selectors": [
            {"selector": s, "occurrences": n}
            for s, n in chain_selector_frequency[chain].most_common(10)
        ],
    }

report = {
    "source": str(SOURCE),
    "total_source_rows": int(len(df)),
    "operations_audited": int(overall["operations"]),
    "selector_entries": int(overall["selector_entries"]),
    "destination_entries": int(overall["destination_entries"]),
    "parse_error_rows_by_chain": dict(parse_errors),
    "operations_with_literal_empty_0x_by_chain": dict(empty_selector_ops),
    "operations_with_no_or_empty_selector_by_chain": dict(no_selector_ops),
    "operations_containing_known_selector_by_chain": dict(known_selector_ops),
    "selector_list_length_distribution": dict(metadata_lengths),
    "top_30_selectors": [
        {
            "selector": s,
            "occurrences": n,
            "known_signature": known.get(s),
        }
        for s, n in selector_frequency.most_common(30)
    ],
    "chains": chain_summary,
    "known_selectors": known,
    "limitations": [
        "An unknown selector is not evidence of maliciousness.",
        "A known selector does not establish the operation's intent.",
        "Counts refer to selector entries, not necessarily unique operations.",
        "Metadata quality does not establish ground-truth security labels.",
        "This audit does not independently verify on-chain transaction traces.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")

print("Saved:", OUT)
print("Source rows:", len(df))
print("Audited rows:", overall["operations"])
print("Selector entries:", overall["selector_entries"])
print("Destination entries:", overall["destination_entries"])
print("Parse-error rows:", sum(parse_errors.values()))
print("\nTop selectors:")
for item in report["top_30_selectors"][:15]:
    print(
        item["selector"],
        item["occurrences"],
        item["known_signature"] or "unclassified"
    )
print("\nPer-chain summary:")
for chain, details in chain_summary.items():
    print(
        chain,
        "operations =", details["operations"],
        "empty 0x =", details["operations_with_literal_empty_0x"],
        "no/empty selector =",
        details["operations_with_no_or_empty_selector"],
        "known selector operations =",
        details["operations_containing_known_selector"],
    )
