from pathlib import Path
import json
import numpy as np
import pandas as pd

SEED = 4337
ROOT = Path("data/experimental/real_background")
SOURCE = ROOT / "real_background_injected_benchmark.csv"
OUT = ROOT / "splits"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)
required = {"source_id", "chain", "label", "modification_type"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

group_check = df.groupby("source_id").agg(
    rows=("label", "size"),
    labels=("label", "nunique"),
    chains=("chain", "nunique"),
)
if not group_check["rows"].eq(2).all():
    raise ValueError("Every source_id must have exactly two rows")
if not group_check["labels"].eq(2).all():
    raise ValueError("Every source_id must have labels 0 and 1")
if not group_check["chains"].eq(1).all():
    raise ValueError("A source_id appears in multiple chains")

rng = np.random.default_rng(SEED)
assignments = {}

# Split source IDs within each chain: 60% train, 20% validation, 20% test.
for chain, group in df.groupby("chain"):
    ids = group["source_id"].drop_duplicates().to_numpy().copy()
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)

    for sid in ids[:n_train]:
        assignments[str(sid)] = "train"
    for sid in ids[n_train:n_train + n_val]:
        assignments[str(sid)] = "validation"
    for sid in ids[n_train + n_val:]:
        assignments[str(sid)] = "test"

df["split"] = df["source_id"].astype(str).map(assignments)
if df["split"].isna().any():
    raise ValueError("Some rows did not receive a split")

source_sets = {
    name: set(df.loc[df["split"].eq(name), "source_id"])
    for name in ("train", "validation", "test")
}
assert source_sets["train"].isdisjoint(source_sets["validation"])
assert source_sets["train"].isdisjoint(source_sets["test"])
assert source_sets["validation"].isdisjoint(source_sets["test"])

summary = {}
for name in ("train", "validation", "test"):
    part = df[df["split"].eq(name)].copy()
    part.drop(columns=["split"]).to_csv(OUT / f"{name}.csv", index=False)
    summary[name] = {
        "rows": int(len(part)),
        "source_ids": int(part["source_id"].nunique()),
        "labels": {
            str(k): int(v)
            for k, v in part["label"].value_counts().sort_index().items()
        },
        "chains": {
            str(k): int(v)
            for k, v in part.groupby("chain")["source_id"].nunique().items()
        },
    }

manifest = {
    "seed": SEED,
    "strategy": "60/20/20 source-grouped split, stratified by chain",
    "pair_leakage": False,
    "splits": summary,
    "limitation": (
        "Label 1 indicates a synthetic feature perturbation, not verified "
        "maliciousness. The task evaluates detectability of injected changes."
    ),
}
(OUT / "split_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print(json.dumps(manifest, indent=2))
