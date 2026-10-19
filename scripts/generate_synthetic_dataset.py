"""Generate reproducible synthetic ERC-4337 feature records.

These are simulated patterns, not real transactions or verified
maliciousness labels. They must not support claims of real-world accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/synthetic/synthetic_userop_features.csv"
DEFAULT_MANIFEST = ROOT / "data/synthetic/synthetic_dataset_manifest.json"

SCENARIOS = (
    "benign_like",
    "unlimited_approval",
    "drain_sequence",
    "module_install_transfer",
    "destination_substitution",
    "malicious_delegation",
)

MODEL_FEATURES = [
    "trace_count",
    "unique_destinations",
    "approve_trace_count",
    "transfer_from_trace_count",
    "value_transfer_trace_count",
    "errored_trace_count",
    "has_approval",
    "has_transfer_from",
    "has_value_transfer",
    "has_trace_errors",
    "has_paymaster",
    "log1p_trace_count",
    "log1p_unique_destinations",
    "log1p_approve_trace_count",
    "log1p_transfer_from_trace_count",
    "log1p_value_transfer_trace_count",
    "log1p_errored_trace_count",
]

METADATA_COLUMNS = [
    "scenario_id",
    "generation_batch",
    "scenario_type",
    "label",
    "label_source",
]

COUNT_FEATURES = [
    "trace_count",
    "unique_destinations",
    "approve_trace_count",
    "transfer_from_trace_count",
    "value_transfer_trace_count",
    "errored_trace_count",
]

BINARY_FEATURES = [
    "has_approval",
    "has_transfer_from",
    "has_value_transfer",
    "has_trace_errors",
    "has_paymaster",
]


def _sample_int(rng, low, high, size):
    return rng.integers(low, high, size=size)


def _generate_scenario_batch(scenario, n, seed, batch_id):
    """Generate one scenario batch using its own deterministic RNG seed."""
    rng = np.random.default_rng(seed)

    trace_count = _sample_int(rng, 2, 45, n)
    unique_destinations = np.minimum(
        _sample_int(rng, 1, 12, n), trace_count
    )
    approvals = _sample_int(rng, 0, 2, n)
    transfer_from = _sample_int(rng, 0, 2, n)
    value_transfers = _sample_int(rng, 0, 4, n)
    errors = rng.binomial(1, 0.04, n)
    has_paymaster = rng.binomial(1, 0.75, n)

    if scenario == "benign_like":
        approvals = rng.binomial(1, 0.22, n)
        transfer_from = rng.binomial(1, 0.18, n)
        value_transfers = rng.binomial(1, 0.35, n)

    elif scenario == "unlimited_approval":
        approvals = _sample_int(rng, 1, 4, n)
        trace_count += _sample_int(rng, 1, 5, n)

    elif scenario == "drain_sequence":
        transfer_from = _sample_int(rng, 2, 7, n)
        value_transfers = _sample_int(rng, 2, 7, n)
        unique_destinations = np.maximum(
            unique_destinations, _sample_int(rng, 2, 7, n)
        )
        trace_count += _sample_int(rng, 4, 18, n)

    elif scenario == "module_install_transfer":
        # Proxy patterns only: no explicit module-install feature exists yet.
        trace_count += _sample_int(rng, 3, 12, n)
        unique_destinations = np.maximum(
            unique_destinations, _sample_int(rng, 2, 6, n)
        )
        value_transfers = np.maximum(
            value_transfers, rng.binomial(1, 0.7, n)
        )

    elif scenario == "destination_substitution":
        # Indirect patterns only: expected-vs-actual destination is absent.
        unique_destinations = np.maximum(
            unique_destinations, _sample_int(rng, 2, 6, n)
        )
        value_transfers = np.maximum(
            value_transfers, rng.binomial(1, 0.6, n)
        )

    elif scenario == "malicious_delegation":
        # Indirect patterns only: no usable delegation feature is present.
        trace_count += _sample_int(rng, 2, 10, n)
        unique_destinations = np.maximum(
            unique_destinations, _sample_int(rng, 2, 5, n)
        )

    unique_destinations = np.minimum(unique_destinations, trace_count)
    approvals = np.minimum(approvals, trace_count)
    transfer_from = np.minimum(transfer_from, trace_count)
    value_transfers = np.minimum(value_transfers, trace_count)
    errors = np.minimum(errors, trace_count)

    frame = pd.DataFrame({
        "trace_count": trace_count,
        "unique_destinations": unique_destinations,
        "approve_trace_count": approvals,
        "transfer_from_trace_count": transfer_from,
        "value_transfer_trace_count": value_transfers,
        "errored_trace_count": errors,
        "has_approval": (approvals > 0).astype(int),
        "has_transfer_from": (transfer_from > 0).astype(int),
        "has_value_transfer": (value_transfers > 0).astype(int),
        "has_trace_errors": (errors > 0).astype(int),
        "has_paymaster": has_paymaster,
    })

    for column in COUNT_FEATURES:
        frame[f"log1p_{column}"] = np.log1p(
            frame[column].astype(float)
        )

    label = 0 if scenario == "benign_like" else 1
    frame["scenario_id"] = [
        f"{batch_id}_row{i:04d}" for i in range(n)
    ]
    frame["generation_batch"] = batch_id
    frame["scenario_type"] = scenario
    frame["label"] = label
    frame["label_source"] = "synthetic_simulation"

    return frame


def generate_dataset(
    samples_per_scenario=1000,
    seed=4337,
    batches_per_scenario=30,
):
    """Generate deterministic rows partitioned into independent seed batches."""
    if samples_per_scenario < 10:
        raise ValueError("samples_per_scenario must be at least 10")
    if batches_per_scenario < 3:
        raise ValueError("batches_per_scenario must be at least 3")
    if batches_per_scenario > samples_per_scenario:
        raise ValueError("batches_per_scenario cannot exceed samples_per_scenario")

    base_size, remainder = divmod(
        samples_per_scenario, batches_per_scenario
    )
    frames = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for batch_index in range(batches_per_scenario):
            batch_size = base_size + int(batch_index < remainder)
            # Each scenario/batch pair gets a distinct reproducible seed.
            batch_seed = (
                seed
                + scenario_index * 1_000_003
                + batch_index * 9_176
            )
            batch_id = f"{scenario}_batch{batch_index:03d}"

            frames.append(
                _generate_scenario_batch(
                    scenario, batch_size, batch_seed, batch_id
                )
            )

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    return dataset[MODEL_FEATURES + METADATA_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-scenario", type=int, default=1000)
    parser.add_argument("--batches-per-scenario", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4337)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    dataset = generate_dataset(
        samples_per_scenario=args.samples_per_scenario,
        seed=args.seed,
        batches_per_scenario=args.batches_per_scenario,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    manifest = {
        "dataset_type": "synthetic_simulation",
        "seed": args.seed,
        "samples_per_scenario": args.samples_per_scenario,
        "batches_per_scenario": args.batches_per_scenario,
        "rows": len(dataset),
        "scenario_counts": {
            str(k): int(v)
            for k, v in dataset["scenario_type"].value_counts().items()
        },
        "label_definition": {
            "0": "benign_like simulated scenario",
            "1": "attack_like simulated scenario",
        },
        "label_source": "synthetic_simulation",
        "source_maliciousness_labels_available": False,
        "real_transactions_included": False,
        "model_features": MODEL_FEATURES,
        "metadata_excluded_from_model_features": METADATA_COLUMNS,
        "limitations": [
            "All records are synthetic, not verified transactions.",
            "Labels reflect simulation rules, not ground-truth maliciousness.",
            "Metrics cannot establish real-world detection performance.",
            "Some attack types are only indirectly represented by current features.",
            "Approvals and transfers can be benign.",
            "Batch separation reduces row-level leakage but does not test unseen attack families.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved dataset: {args.output}")
    print(f"Saved manifest: {args.manifest}")
    print(f"Rows: {len(dataset):,}")
    print(f"Model features: {len(MODEL_FEATURES)}")
    print(f"Unique generation batches: {dataset['generation_batch'].nunique()}")
    print("\nScenario counts:")
    print(dataset["scenario_type"].value_counts().sort_index().to_string())
    print("\nLabel counts:")
    print(dataset["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
