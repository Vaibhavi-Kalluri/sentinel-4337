"""Split E1 contextual data by independent session, never by individual row."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/synthetic/e1_contextual_dataset.csv"
DEFAULT_OUTPUT = ROOT / "data/synthetic/e1_splits"

METADATA_COLUMNS = [
    "scenario_id",
    "generation_batch",
    "scenario_type",
    "label",
    "label_source",
    "session_id",
    "operation_index",
    "wallet_id",
    "destination_id",
    "selector_id",
    "timestamp_seconds",
    "value_eth",
]


def split_by_session(df, seed=4337):
    required = {"session_id", "scenario_type", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["session_id"].isna().any():
        raise ValueError("session_id cannot be missing")
    if df["session_id"].nunique() < 5:
        raise ValueError("At least five sessions are required")

    sessions = np.array(sorted(df["session_id"].unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = int(round(n * 0.60))
    n_val = int(round(n * 0.20))
    # Assign the remainder to test to account for rounding.
    train_sessions = set(sessions[:n_train])
    val_sessions = set(sessions[n_train:n_train + n_val])
    test_sessions = set(sessions[n_train + n_val:])

    if not train_sessions or not val_sessions or not test_sessions:
        raise ValueError("Each split must contain at least one session")

    splits = {
        "train": df[df["session_id"].isin(train_sessions)].copy(),
        "validation": df[df["session_id"].isin(val_sessions)].copy(),
        "test": df[df["session_id"].isin(test_sessions)].copy(),
    }

    expected_scenarios = set(df["scenario_type"].unique())
    expected_labels = set(df["label"].unique())

    for name, part in splits.items():
        if set(part["scenario_type"].unique()) != expected_scenarios:
            raise ValueError(f"{name} split does not contain every scenario")
        if set(part["label"].unique()) != expected_labels:
            raise ValueError(f"{name} split does not contain every label")

    # Verify that sessions are isolated.
    session_sets = {
        name: set(part["session_id"].unique())
        for name, part in splits.items()
    }
    names = list(session_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if not session_sets[left].isdisjoint(session_sets[right]):
                raise ValueError(f"Session leakage between {left} and {right}")

    return splits


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=4337)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    splits = split_by_session(df, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        path = args.output_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        print(
            f"{name}: {len(part):,} rows, "
            f"{part['session_id'].nunique()} sessions, "
            f"{part['scenario_type'].nunique()} scenarios"
        )
        print(part["label"].value_counts().sort_index().to_dict())

    manifest = {
        "dataset_type": "synthetic_contextual_simulation",
        "seed": args.seed,
        "input": str(args.input),
        "split_unit": "session_id",
        "split_ratios_approximate": {
            "train": 0.60,
            "validation": 0.20,
            "test": 0.20,
        },
        "session_counts": {
            name: int(part["session_id"].nunique())
            for name, part in splits.items()
        },
        "row_counts": {
            name: int(len(part)) for name, part in splits.items()
        },
        "metadata_excluded_from_model_features": METADATA_COLUMNS,
        "limitations": [
            "All records and labels are synthetic.",
            "Sessions are independent simulation units, not real wallets or chain sessions.",
            "A session-grouped split does not demonstrate generalization to real attacks.",
        ],
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
