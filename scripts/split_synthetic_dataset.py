#!/usr/bin/env python3
"""Create deterministic, scenario-stratified, batch-grouped dataset splits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/synthetic/synthetic_userop_features.csv"
OUTPUT = ROOT / "data/synthetic/splits"

METADATA_COLUMNS = {
    "scenario_id",
    "generation_batch",
    "scenario_type",
    "label",
    "label_source",
}


def create_splits(df, seed=4337):
    required = METADATA_COLUMNS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["generation_batch"].isna().any():
        raise ValueError("generation_batch contains missing values")

    rng = np.random.default_rng(seed)
    assignments = {"train": [], "validation": [], "test": []}

    # Split each scenario's batches separately to guarantee scenario coverage.
    for scenario in sorted(df["scenario_type"].unique()):
        scenario_df = df[df["scenario_type"] == scenario]
        batches = scenario_df["generation_batch"].unique().tolist()

        if len(batches) < 3:
            raise ValueError(
                f"Scenario {scenario!r} needs at least 3 batches; got {len(batches)}"
            )

        rng.shuffle(batches)

        n_test = max(1, round(len(batches) * 0.20))
        n_validation = max(1, round(len(batches) * 0.20))
        n_train = len(batches) - n_test - n_validation

        if n_train < 1:
            raise ValueError(f"Not enough batches for scenario {scenario!r}")

        assignments["train"].extend(batches[:n_train])
        assignments["validation"].extend(
            batches[n_train:n_train + n_validation]
        )
        assignments["test"].extend(batches[n_train + n_validation:])

    splits = {}
    for name, batches in assignments.items():
        splits[name] = (
            df[df["generation_batch"].isin(batches)]
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

    # Guard against accidental leakage and missing scenario coverage.
    batch_sets = {
        name: set(part["generation_batch"])
        for name, part in splits.items()
    }

    names = list(batch_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = batch_sets[left] & batch_sets[right]
            if overlap:
                raise AssertionError(
                    f"Batch leakage between {left} and {right}: {overlap}"
                )

    expected_scenarios = set(df["scenario_type"].unique())
    for name, part in splits.items():
        if set(part["scenario_type"].unique()) != expected_scenarios:
            raise ValueError(f"{name} split is missing one or more scenarios")
        if set(part["label"].unique()) != {0, 1}:
            raise ValueError(f"{name} split is missing one or more labels")

    return splits


def main():
    df = pd.read_csv(INPUT)
    splits = create_splits(df)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Synthetic prototype evaluation only",
        "seed": 4337,
        "split_method": (
            "Scenario-stratified generation-batch split; batches are kept intact"
        ),
        "metadata_excluded_from_model_features": sorted(METADATA_COLUMNS),
        "limitations": [
            "Labels are generated from synthetic scenario rules.",
            "Results do not establish real-world malicious-operation detection.",
            "Batch separation does not prove generalization to unseen attack types.",
            "Real UserOperation success/failure is not a maliciousness label.",
        ],
        "splits": {},
    }

    for name, part in splits.items():
        path = OUTPUT / f"{name}.csv"
        part.to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "batches": int(part["generation_batch"].nunique()),
            "scenarios": {
                str(k): int(v)
                for k, v in part["scenario_type"].value_counts().sort_index().items()
            },
            "labels": {
                str(k): int(v)
                for k, v in part["label"].value_counts().sort_index().items()
            },
            "file": str(path.relative_to(ROOT)),
        }
        print(
            f"{name}: {len(part)} rows, "
            f"{part['generation_batch'].nunique()} batches"
        )
        print(part["scenario_type"].value_counts().sort_index().to_string())

    manifest_path = OUTPUT / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSaved split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
