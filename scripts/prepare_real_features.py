from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/real/optimism_userop_features_20250409.csv"
OUTPUT_DIR = ROOT / "data/processed"
OUTPUT = OUTPUT_DIR / "optimism_userop_features_20250409_processed.csv"
MANIFEST = OUTPUT_DIR / "feature_manifest.json"

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
    "userop_hash",
    "chain",
    "sender",
    "paymaster",
    "nonce",
    "success",
    *NUMERIC_FEATURES,
]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")

    df = pd.read_csv(INPUT)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["userop_hash"].duplicated().any():
        raise ValueError("Duplicate operation hashes found.")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any():
            raise ValueError(f"Missing numeric values in {col}")
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}")

    # The source's success flag describes execution outcome,
    # not whether an operation is malicious.
    success_text = df["success"].astype(str).str.lower()
    if not success_text.isin(["true", "false"]).all():
        raise ValueError("Unexpected values in success column.")
    df["success"] = success_text.map({"true": True, "false": False})

    # Keep the original counts and create additional model candidates.
    df["has_approval"] = (
        df["approve_trace_count"] > 0
    ).astype("int8")

    df["has_transfer_from"] = (
        df["transfer_from_trace_count"] > 0
    ).astype("int8")

    df["has_value_transfer"] = (
        df["value_transfer_trace_count"] > 0
    ).astype("int8")

    df["has_trace_errors"] = (
        df["errored_trace_count"] > 0
    ).astype("int8")

    df["has_paymaster"] = (
        df["paymaster"].fillna("").astype(str).str.strip().str.lower()
        .ne("")
        & df["paymaster"].fillna("").astype(str).str.lower()
        .ne("0x0000000000000000000000000000000000000000")
    ).astype("int8")

    # Compress heavy-tailed counts and gas usage without dropping rows.
    skewed_features = [
        "trace_count",
        "unique_destinations",
        "approve_trace_count",
        "transfer_from_trace_count",
        "value_transfer_trace_count",
        "errored_trace_count",
        "actual_gas_used",
        "actual_gas_cost_eth",
    ]

    for col in skewed_features:
        df[f"log1p_{col}"] = (
            __import__("numpy").log1p(df[col].astype(float))
        )

    # Delegatecall count is constant in this sample, so it is not
    # included as a candidate model feature.
    candidate_features = [
        *NUMERIC_FEATURES,
        "has_approval",
        "has_transfer_from",
        "has_value_transfer",
        "has_trace_errors",
        "has_paymaster",
        *[f"log1p_{col}" for col in skewed_features],
    ]

    # Retain all source fields and all rows for traceability.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    manifest = {
        "input_file": str(INPUT.relative_to(ROOT)),
        "output_file": str(OUTPUT.relative_to(ROOT)),
        "rows": int(len(df)),
        "unique_operation_hashes": int(df["userop_hash"].nunique()),
        "chain_values": sorted(df["chain"].dropna().unique().tolist()),
        "date_scope": "2025-04-09",
        "source_label_available": False,
        "success_is_maliciousness_label": False,
        "constant_feature_excluded": {
            "delegatecall_count": (
                "Constant zero in this sample; reconsider with broader data."
            )
        },
        "candidate_model_features": candidate_features,
        "notes": [
            "No maliciousness labels are inferred.",
            "Raw input is not modified.",
            "Log-transformed columns supplement rather than replace raw counts.",
            "Candidate features require further validation before model training.",
        ],
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print("Feature preparation complete.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique operations: {df['userop_hash'].nunique()}")
    print(f"Output: {OUTPUT.relative_to(ROOT)}")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print("\nNew indicator counts:")
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
