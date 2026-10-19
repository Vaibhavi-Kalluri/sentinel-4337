from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/real/multichain_userop_features_20250401_20250414.csv"
OUTPUT_DIR = ROOT / "data/processed"
OUTPUT = OUTPUT_DIR / "multichain_userop_features_20250401_20250414_processed.csv"
MANIFEST = OUTPUT_DIR / "multichain_feature_manifest.json"

NUMERIC_FEATURES = [
    "trace_count",
    "unique_destinations",
    "approve_trace_count",
    "transfer_from_trace_count",
    "value_transfer_trace_count",
    "errored_trace_count",
    "actual_gas_used",
    "actual_gas_cost_eth",
]

REQUIRED_COLUMNS = [
    "dt",
    "chain",
    "userop_hash",
    "sender",
    "paymaster",
    "nonce",
    "success",
    *NUMERIC_FEATURES,
]

SKEWED_FEATURES = NUMERIC_FEATURES.copy()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")

    df = pd.read_csv(INPUT)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Normalize identifiers for reliable validation.
    df["dt"] = pd.to_datetime(df["dt"], errors="raise").dt.strftime("%Y-%m-%d")
    df["chain"] = df["chain"].astype(str).str.strip().str.lower()
    df["userop_hash"] = df["userop_hash"].astype(str).str.strip().str.lower()

    if df[["dt", "chain", "userop_hash"]].isna().any().any():
        raise ValueError("Missing date, chain, or operation hash.")

    if df["userop_hash"].eq("").any():
        raise ValueError("Empty operation hash found.")

    if df.duplicated(["chain", "userop_hash"]).any():
        raise ValueError("Duplicate chain + operation hash pairs found.")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any():
            raise ValueError(f"Missing numeric values in {col}")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite numeric values in {col}")
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}")

    success_text = df["success"].astype(str).str.strip().str.lower()
    if not success_text.isin(["true", "false"]).all():
        raise ValueError("Unexpected values in success column.")
    df["success"] = success_text.map({"true": True, "false": False})

    df["has_approval"] = (df["approve_trace_count"] > 0).astype("int8")
    df["has_transfer_from"] = (
        df["transfer_from_trace_count"] > 0
    ).astype("int8")
    df["has_value_transfer"] = (
        df["value_transfer_trace_count"] > 0
    ).astype("int8")
    df["has_trace_errors"] = (
        df["errored_trace_count"] > 0
    ).astype("int8")

    paymaster = df["paymaster"].fillna("").astype(str).str.strip().str.lower()
    df["has_paymaster"] = (
        paymaster.ne("")
        & paymaster.ne("0x0000000000000000000000000000000000000000")
    ).astype("int8")

    for col in SKEWED_FEATURES:
        df[f"log1p_{col}"] = np.log1p(df[col].astype(float))

    # Exclude features that have no variation in this dataset.
    candidate_features = [
        *NUMERIC_FEATURES,
        "has_approval",
        "has_transfer_from",
        "has_value_transfer",
        "has_trace_errors",
        "has_paymaster",
        *[f"log1p_{col}" for col in SKEWED_FEATURES],
    ]

    constant_features = [
        col for col in candidate_features if df[col].nunique(dropna=False) <= 1
    ]
    candidate_features = [
        col for col in candidate_features if col not in constant_features
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    chain_counts = df.groupby("chain").size().sort_index().to_dict()
    daily_counts = df.groupby("dt").size().sort_index().to_dict()
    chain_day_counts = {
        f"{chain}|{dt}": int(count)
        for (chain, dt), count in df.groupby(["chain", "dt"]).size().items()
    }

    manifest = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "output_file": str(OUTPUT.relative_to(ROOT)),
        "rows": int(len(df)),
        "unique_chain_operation_pairs": int(
            df[["chain", "userop_hash"]].drop_duplicates().shape[0]
        ),
        "chains": sorted(df["chain"].unique().tolist()),
        "date_start": df["dt"].min(),
        "date_end": df["dt"].max(),
        "operations_per_chain": {k: int(v) for k, v in chain_counts.items()},
        "operations_per_day": {k: int(v) for k, v in daily_counts.items()},
        "operations_per_chain_per_day": chain_day_counts,
        "source_maliciousness_labels_available": False,
        "success_is_maliciousness_label": False,
        "constant_candidate_features_excluded": constant_features,
        "candidate_model_features": candidate_features,
        "notes": [
            "Real on-chain observations have no verified maliciousness labels.",
            "Execution success indicates outcome, not maliciousness.",
            "Raw input and the original Optimism processed output are not modified.",
            "Log-transformed features supplement rather than replace raw counts.",
            "Sampling coverage should be checked before claiming representativeness.",
            "These features alone do not establish that an operation is malicious.",
        ],
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print("Multichain feature preparation complete.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Chains: {', '.join(sorted(df['chain'].unique()))}")
    print(f"Date range: {df['dt'].min()} to {df['dt'].max()}")
    print(f"Unique chain-operation pairs: {df[['chain', 'userop_hash']].drop_duplicates().shape[0]}")
    print(f"Candidate features: {len(candidate_features)}")
    print(f"Constant candidate features excluded: {constant_features}")
    print(f"Output: {OUTPUT.relative_to(ROOT)}")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print("\nIndicator counts:")
    for col in [
        "has_approval",
        "has_transfer_from",
        "has_value_transfer",
        "has_trace_errors",
        "has_paymaster",
    ]:
        print(f"{col}: {int(df[col].sum())}")


if __name__ == "__main__":
    main()
