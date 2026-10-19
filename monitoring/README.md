# Sentinel 4337: Prototype Operation Evaluator

## Overview

This directory contains an initial Python prototype for evaluating decoded
ERC-4337 UserOperation metadata. It extracts basic features, assigns a
rule-based risk score, explains triggered rules, and returns an advisory
recommendation.

This is an early prototype, not a trained machine-learning model or a
complete transaction security system.

## Current Features

- Validates required input fields.
- Extracts the selector, ETH value, gas limit, target novelty, and supplied
  target-approval status.
- Assigns illustrative risk points using predefined rules.
- Returns a risk score, reasons, and an ALLOW or FLAG recommendation.
- Rejects missing fields and negative ETH values or gas limits.

## Risk Rules

| Rule | Risk points |
|---|---:|
| Target is not approved | 40 |
| Approval or operator-approval selector | 35 |
| Target is new to the wallet | 15 |
| ETH value is at least 1 ETH | 10 |

The score is capped at 100. The current recommendation is FLAG when the
target is unapproved or the score is at least 60; otherwise, it is ALLOW.

These weights and the threshold are illustrative and have not been
empirically calibrated.

## Input Format

Example operation:

    {
      "sender": "0x1111111111111111111111111111111111111111",
      "target": "0x0165878A594ca255338adfa4d48449f69242Eb8F",
      "selector": "0x55241077",
      "value_eth": 0,
      "gas_limit": 100000,
      "is_new_target": false,
      "is_approved_target": true
    }

The input must be a decoded JSON object. The approval and novelty fields
are supplied by the caller; the evaluator does not query the blockchain.

## Run the Evaluator

From the repository root:

    python monitoring/evaluate_operation.py monitoring/sample_allowed.json

Other examples:

    python monitoring/evaluate_operation.py monitoring/sample_unapproved.json
    python monitoring/evaluate_operation.py monitoring/sample_approval.json

## Run Automated Tests

    cd monitoring
    python -m unittest -v test_evaluate_operation.py
    cd ..

## Current Limitations

- Uses hand-written rules, not machine learning.
- Does not retrieve or decode live UserOperations.
- Does not query wallet history or target allowlists.
- Does not submit, block, or enforce transactions on-chain.
- Does not provide a security guarantee; a flagged operation is not
  necessarily malicious, and an allowed operation is not necessarily safe.
- Has not been validated against a representative real-world dataset.

## Planned Work

1. Add more unit and edge-case tests.
2. Build a reproducible dataset and feature-extraction pipeline.
3. Connect target-approval context to trusted on-chain policy data.
4. Train and evaluate baseline machine-learning models.
5. Compare model predictions with policy enforcement.
6. Measure detection performance, false positives, and latency.
7. Investigate zero-knowledge proofs as a later extension.
