# Sentinel 4337: Experimental Results

## Experimental setup

- Source background: 42,000 UserOperation feature records across six chains.
- Injected benchmark: 84,000 rows, consisting of 42,000 original records and 42,000 controlled feature perturbations.
- Split: 60/20/20 by source ID, stratified within each chain. Original and perturbed copies remain in the same split.
- Model: Random Forest, 400 trees, minimum leaf size 2, balanced class weights.
- Thresholds: selected on validation data; test set used for final evaluation.
- Positive label: controlled synthetic feature perturbation, not verified maliciousness.

## Ablation at validation FPR target of 5%

| Model | Features | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count_only | 6.0000 | 0.9510 | 0.9268 | 0.9387 | 0.9863 | 0.9855 | 0.0477 |
| indicator_only | 5.0000 | 0.9214 | 0.5208 | 0.6655 | 0.8137 | 0.7859 | 0.0444 |
| full_17_features | 17.0000 | 0.9497 | 0.9337 | 0.9416 | 0.9875 | 0.9867 | 0.0494 |

## Error analysis

The full 17-feature model's test FPR is approximately 13% on Base,
compared with lower rates on the other sampled chains. This indicates
that aggregate performance can hide chain-specific errors.

Multi-call perturbations are the hardest category for the full model:
381 of 1,681 test examples were missed (recall approximately 77.33%).

## Limitations

These results evaluate detection of controlled feature-space perturbations.
They do not measure real-world attack detection accuracy. Original records
are not confirmed benign, and positive examples are not verified attacks.
The benchmark does not reconstruct modified transaction traces. More
independently labelled security examples and realistic attack simulations
are required before claiming production readiness.

## Current conclusion

The full feature model is the strongest tested configuration by F1 and
ranking metrics, while the count-only model is a simpler competitive
baseline. The indicator-only model loses substantial recall. The next
research priority is independently labelled validation, not further
optimization on this injected benchmark.
