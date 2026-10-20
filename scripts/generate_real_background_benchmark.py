from pathlib import Path
import json
import numpy as np
import pandas as pd

SEED = 4337

SOURCE = Path(
    "data/processed/"
    "multichain_userop_features_20250401_20250414_processed.csv"
)
OUT = Path("data/experimental/real_background")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
df = pd.read_csv(SOURCE)

required = {
    "userop_hash", "chain", "trace_count", "unique_destinations",
    "approve_trace_count", "transfer_from_trace_count",
    "value_transfer_trace_count", "errored_trace_count",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df["userop_hash"].isna().any():
    raise ValueError("Source contains missing UserOperation hashes")

if df["userop_hash"].astype(str).duplicated().any():
    raise ValueError("Expected unique source UserOperation hashes")

base = df.copy()
base["source_id"] = base["userop_hash"].astype(str)
base["label"] = 0
base["modification_type"] = "unmodified_real_background"

modified = df.copy()
modified["source_id"] = modified["userop_hash"].astype(str)

scenario_names = np.array([
    "approval_perturbation",
    "transfer_perturbation",
    "destination_perturbation",
    "multi_call_perturbation",
    "combined_perturbation",
])
modified["modification_type"] = rng.choice(
    scenario_names, size=len(modified), replace=True
)
modified["label"] = 1

# Every selected perturbation adds at least one count.
# These are controlled feature-space changes, not reconstructed transactions.
def add_positive_increment(mask, column, low=1, high=4):
    idx = modified.index[mask]
    if len(idx) == 0:
        return
    increments = rng.integers(low, high, size=len(idx))
    modified.loc[idx, column] = (
        pd.to_numeric(modified.loc[idx, column]).to_numpy() + increments
    )

types = modified["modification_type"]

add_positive_increment(
    types.eq("approval_perturbation"), "approve_trace_count"
)
add_positive_increment(
    types.eq("transfer_perturbation"), "transfer_from_trace_count"
)
add_positive_increment(
    types.eq("destination_perturbation"), "unique_destinations"
)
add_positive_increment(
    types.eq("multi_call_perturbation"), "trace_count"
)

combined_mask = types.eq("combined_perturbation")
add_positive_increment(combined_mask, "approve_trace_count")
add_positive_increment(combined_mask, "value_transfer_trace_count")
add_positive_increment(combined_mask, "trace_count")

# Keep all count-derived indicators consistent.
indicator_map = {
    "has_approval": "approve_trace_count",
    "has_transfer_from": "transfer_from_trace_count",
    "has_value_transfer": "value_transfer_trace_count",
    "has_trace_errors": "errored_trace_count",
}
for indicator, count_column in indicator_map.items():
    if indicator in modified.columns:
        modified[indicator] = (
            pd.to_numeric(modified[count_column]) > 0
        ).astype(int)

# Recompute log1p features from their source counts.
log_map = {
    "log1p_trace_count": "trace_count",
    "log1p_unique_destinations": "unique_destinations",
    "log1p_approve_trace_count": "approve_trace_count",
    "log1p_transfer_from_trace_count": "transfer_from_trace_count",
    "log1p_value_transfer_trace_count": "value_transfer_trace_count",
    "log1p_errored_trace_count": "errored_trace_count",
}
for log_column, count_column in log_map.items():
    if log_column in modified.columns:
        modified[log_column] = np.log1p(
            pd.to_numeric(modified[count_column])
        )

combined = pd.concat([base, modified], ignore_index=True)
combined["source_id"] = combined["source_id"].astype(str)
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Pairing and guaranteed-change assertions.
assert len(combined) == 2 * len(df)
assert combined.groupby("source_id").size().eq(2).all()
assert combined.groupby("source_id")["label"].nunique().eq(2).all()

base_by_id = base.set_index("source_id")
mod_by_id = modified.set_index("source_id")

intended_feature = {
    "approval_perturbation": "approve_trace_count",
    "transfer_perturbation": "transfer_from_trace_count",
    "destination_perturbation": "unique_destinations",
    "multi_call_perturbation": "trace_count",
    "combined_perturbation": "trace_count",
}

for scenario, feature in intended_feature.items():
    ids = modified.loc[
        modified["modification_type"].eq(scenario), "source_id"
    ]
    delta = (
        pd.to_numeric(mod_by_id.loc[ids, feature]).to_numpy()
        - pd.to_numeric(base_by_id.loc[ids, feature]).to_numpy()
    )
    assert (delta > 0).all(), (
        f"{scenario} has non-positive changes in {feature}"
    )

out_path = OUT / "real_background_injected_benchmark.csv"
combined.to_csv(out_path, index=False)

manifest = {
    "seed": SEED,
    "source_file": str(SOURCE),
    "source_rows": int(len(df)),
    "benchmark_rows": int(len(combined)),
    "unique_source_ids": int(combined["source_id"].nunique()),
    "labels": {
        "0": "unmodified real-background operation",
        "1": "controlled synthetic feature perturbation",
    },
    "perturbation_counts": {
        str(k): int(v)
        for k, v in modified["modification_type"].value_counts().items()
    },
    "perturbation_design": {
        "approval_perturbation": "Add 1-3 to approve_trace_count",
        "transfer_perturbation": "Add 1-3 to transfer_from_trace_count",
        "destination_perturbation": "Add 1-3 to unique_destinations",
        "multi_call_perturbation": "Add 1-3 to trace_count",
        "combined_perturbation": (
            "Add 1-3 to approval count, value-transfer count, "
            "and trace count"
        ),
    },
    "important_limitations": [
        "Positive labels mean synthetic feature perturbation, not verified maliciousness.",
        "This is a feature-space benchmark, not reconstructed UserOperations.",
        "Raw destination_addresses and method_selectors are not rewritten.",
        "Do not use raw identifiers, metadata, or source_id as model features.",
        "Execution success is not a maliciousness label.",
        "The original real-background operations are not confirmed benign.",
        "Results measure detection of these injected perturbations only.",
    ],
}

(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print("Benchmark regenerated successfully.")
print(f"Source operations: {len(df):,}")
print(f"Benchmark rows: {len(combined):,}")
print("Labels:")
print(combined["label"].value_counts().sort_index().to_string())
print("\nPerturbation types:")
print(modified["modification_type"].value_counts().to_string())
print(f"\nSaved: {out_path}")
print(f"Manifest: {OUT / 'manifest.json'}")
