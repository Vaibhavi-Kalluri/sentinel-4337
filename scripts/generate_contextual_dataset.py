"""Generate a synthetic mixed-session dataset for the E1 contextual experiment.

This is a simulation, not verified blockchain transaction data.
History features use only earlier operations within the same session.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_synthetic_dataset import (
    MODEL_FEATURES,
    SCENARIOS,
    generate_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/synthetic/e1_contextual_dataset.csv"
DEFAULT_MANIFEST = ROOT / "data/synthetic/e1_contextual_manifest.json"

CONTEXT_FEATURES = [
    "dest_seen_before",
    "selector_seen_before",
    "recent_operation_count",
    "value_zscore",
]

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

MODEL_COLUMNS = MODEL_FEATURES + CONTEXT_FEATURES


def generate_contextual_dataset(
    samples_per_scenario=1000,
    batches_per_scenario=30,
    sessions=30,
    seed=4337,
):
    if sessions < 5:
        raise ValueError("Use at least 5 independent sessions.")
    if batches_per_scenario < 3:
        raise ValueError("Use at least 3 batches per scenario.")
    if sessions < batches_per_scenario:
        raise ValueError(
            "Use at least as many sessions as batches per scenario."
        )

    rng = np.random.default_rng(seed)

    # Preserve E0 feature-generation logic and labels.
    df = generate_dataset(
        samples_per_scenario=samples_per_scenario,
        seed=seed,
        batches_per_scenario=batches_per_scenario,
    ).reset_index(drop=True)

    # Spread every scenario across the mixed sessions.
    # A session will contain examples from all six scenarios.
    df["session_id"] = ""
    for scenario in SCENARIOS:
        indices = df.index[df["scenario_type"] == scenario].to_numpy()
        indices = rng.permutation(indices)
        for position, idx in enumerate(indices):
            df.at[idx, "session_id"] = (
                f"session_{position % sessions:03d}"
            )

    # Assign chronological order independently inside each session.
    df["operation_index"] = 0
    df["timestamp_seconds"] = 0
    for session_id, indices in df.groupby("session_id", sort=True).groups.items():
        ordered = list(indices)
        rng.shuffle(ordered)

        # Simulated timestamps: positive time gaps, no future information.
        gaps = rng.integers(1, 3601, size=len(ordered))
        timestamps = np.cumsum(gaps)

        for position, idx in enumerate(ordered):
            df.at[idx, "operation_index"] = position
            df.at[idx, "timestamp_seconds"] = int(timestamps[position])

    # Synthetic wallet reuse across different scenario types is deliberate.
    # Contexts reset at session boundaries to avoid train/test history leakage.
    df["wallet_id"] = ""
    df["destination_id"] = ""
    df["selector_id"] = ""
    df["value_eth"] = 0.0

    for session_id, indices in df.groupby("session_id", sort=True).groups.items():
        indices = sorted(indices, key=lambda i: df.at[i, "operation_index"])

        wallet_count = max(8, min(30, len(indices) // 5))
        wallet_ids = [f"{session_id}_wallet_{i:03d}" for i in range(wallet_count)]

        # Wallet assignment does not directly use the label.
        wallet_for_index = {
            idx: wallet_ids[int(rng.integers(0, wallet_count))]
            for idx in indices
        }

        for idx in indices:
            scenario = df.at[idx, "scenario_type"]
            df.at[idx, "wallet_id"] = wallet_for_index[idx]

            # Use a shared pool so destinations/selectors can recur.
            destination_pool_size = 18
            selector_pool_size = 10

            # Scenario-conditioned activity is part of the simulation
            # hypothesis and must not be interpreted as real-world evidence.
            if scenario == "destination_substitution":
                dest = int(rng.integers(10, destination_pool_size + 10))
            else:
                dest = int(rng.integers(0, destination_pool_size))

            if scenario == "unlimited_approval":
                selector = 0
            elif scenario == "module_install_transfer":
                selector = int(rng.integers(1, selector_pool_size))
            else:
                selector = int(rng.integers(0, selector_pool_size))

            df.at[idx, "destination_id"] = f"{session_id}_dest_{dest:03d}"
            df.at[idx, "selector_id"] = f"selector_{selector:03d}"

            # Values are synthetic. No claim is made that these are ETH values
            # observed on-chain.
            if scenario in {
                "drain_sequence",
                "destination_substitution",
                "module_install_transfer",
            }:
                value = float(rng.lognormal(mean=-0.2, sigma=1.0))
            else:
                value = float(rng.lognormal(mean=-2.0, sigma=0.8))
            df.at[idx, "value_eth"] = value

    # Build context strictly from previous operations by the same wallet
    # within the same session.
    df["dest_seen_before"] = 0
    df["selector_seen_before"] = 0
    df["recent_operation_count"] = 0
    df["value_zscore"] = 0.0

    for session_id, indices in df.groupby("session_id", sort=True).groups.items():
        ordered = sorted(indices, key=lambda i: df.at[i, "operation_index"])
        wallet_history = {}

        for idx in ordered:
            wallet = df.at[idx, "wallet_id"]
            destination = df.at[idx, "destination_id"]
            selector = df.at[idx, "selector_id"]
            value = float(df.at[idx, "value_eth"])
            timestamp = int(df.at[idx, "timestamp_seconds"])

            history = wallet_history.setdefault(
                wallet, {"destinations": set(), "selectors": set(), "events": []}
            )
            prior_events = history["events"]

            df.at[idx, "dest_seen_before"] = int(
                destination in history["destinations"]
            )
            df.at[idx, "selector_seen_before"] = int(
                selector in history["selectors"]
            )

            # Previous operations in the preceding hour, not including current.
            recent = [
                event for event in prior_events
                if 0 <= timestamp - event["timestamp"] <= 3600
            ]
            df.at[idx, "recent_operation_count"] = len(recent)

            # Past-only z-score. Zero until there are enough historical values.
            prior_values = [event["value"] for event in prior_events]
            if len(prior_values) >= 3:
                mean = float(np.mean(prior_values))
                std = float(np.std(prior_values))
                if std > 1e-9:
                    z = (value - mean) / std
                    df.at[idx, "value_zscore"] = float(
                        np.clip(z, -10.0, 10.0)
                    )

            # Update history only AFTER deriving this operation's features.
            history["destinations"].add(destination)
            history["selectors"].add(selector)
            history["events"].append({
                "timestamp": timestamp,
                "value": value,
            })

    df["session_id"] = df["session_id"].astype(str)
    df["operation_index"] = df["operation_index"].astype(int)
    df["timestamp_seconds"] = df["timestamp_seconds"].astype(int)

    # Shuffle row order to ensure training does not depend on CSV ordering.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df[MODEL_COLUMNS + METADATA_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-scenario", type=int, default=1000)
    parser.add_argument("--batches-per-scenario", type=int, default=30)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    df = generate_contextual_dataset(
        samples_per_scenario=args.samples_per_scenario,
        batches_per_scenario=args.batches_per_scenario,
        sessions=args.sessions,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    manifest = {
        "dataset_type": "synthetic_contextual_simulation",
        "seed": args.seed,
        "rows": len(df),
        "sessions": int(df["session_id"].nunique()),
        "scenarios": SCENARIOS,
        "base_model_features": MODEL_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "all_model_features": MODEL_COLUMNS,
        "metadata_excluded_from_model_features": METADATA_COLUMNS,
        "label_source": "synthetic_simulation",
        "real_transactions_included": False,
        "history_policy": (
            "Past operations only; history resets at each independent session."
        ),
        "limitations": [
            "All records and context are synthetic.",
            "Labels are scenario-defined, not verified maliciousness labels.",
            "Scenario-conditioned synthetic activity can create artificial signals.",
            "Results cannot establish real-world detection performance.",
            "The dataset does not represent the true prevalence of attacks.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved dataset: {args.output}")
    print(f"Saved manifest: {args.manifest}")
    print(f"Rows: {len(df):,}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Model features: {len(MODEL_COLUMNS)}")
    print("\nScenario counts:")
    print(df["scenario_type"].value_counts().sort_index().to_string())
    print("\nScenarios per session (minimum, maximum):")
    per_session = df.groupby("session_id")["scenario_type"].nunique()
    print(f"{per_session.min()}, {per_session.max()}")


if __name__ == "__main__":
    main()
