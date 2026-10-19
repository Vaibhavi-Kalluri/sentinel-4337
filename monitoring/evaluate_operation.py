#!/usr/bin/env python3
"""Prototype rule-based evaluator for Sentinel 4337 UserOperations.

This is a development prototype, not a trained ML model or a security guarantee.
It analyses decoded operation metadata; it does not submit transactions.
"""

import argparse
import json
from pathlib import Path


def evaluate(operation: dict) -> dict:
    """Return extracted features, risk score, reasons, and recommendation."""
    required = ("sender", "target", "selector", "value_eth", "gas_limit")
    missing = [key for key in required if key not in operation]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    value_eth = float(operation["value_eth"])
    gas_limit = int(operation["gas_limit"])
    selector = str(operation["selector"]).lower()
    is_new_target = bool(operation.get("is_new_target", False))
    is_approved_target = bool(operation.get("is_approved_target", False))

    if value_eth < 0 or gas_limit < 0:
        raise ValueError("value_eth and gas_limit must be non-negative")

    risk = 0
    reasons = []

    # These are illustrative rules, not empirically calibrated risk weights.
    if not is_approved_target:
        risk += 40
        reasons.append("Target is not allowlisted by the supplied policy context")

    if selector in {"0x095ea7b3", "0xa22cb465"}:
        risk += 35
        reasons.append("Approval or operator-approval selector detected")

    if is_new_target:
        risk += 15
        reasons.append("Target is new to this wallet")

    if value_eth >= 1:
        risk += 10
        reasons.append("Native value transfer is at least 1 ETH")

    risk = min(risk, 100)

    # Fail closed for a target the policy context says is unapproved.
    # Other recommendations are advisory and do not enforce on-chain.
    if not is_approved_target:
        recommendation = "FLAG"
    elif risk >= 60:
        recommendation = "FLAG"
    else:
        recommendation = "ALLOW"

    return {
        "sender": operation["sender"],
        "target": operation["target"],
        "features": {
            "selector": selector,
            "value_eth": value_eth,
            "gas_limit": gas_limit,
            "is_new_target": is_new_target,
            "is_approved_target": is_approved_target,
        },
        "risk_score": risk,
        "reasons": reasons or ["No prototype risk rules triggered"],
        "recommendation": recommendation,
        "model_type": "prototype_rules_not_ml",
        "enforces_on_chain": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a decoded Sentinel UserOperation JSON file."
    )
    parser.add_argument("input", type=Path, help="Path to a JSON operation")
    args = parser.parse_args()

    try:
        operation = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(operation, dict):
            raise ValueError("Input JSON must be an object")
        result = evaluate(operation)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
