"""Local research-demo decision layer. Not on-chain enforcement or ZK."""
from __future__ import annotations


def decide(*, approved_target: bool, ml_score: float | None, review_threshold: float = 0.4466300357476273, high_threshold: float = 0.8) -> dict:
    if type(approved_target) is not bool:
        raise ValueError('approved_target must be a boolean')
    if not 0 < review_threshold < high_threshold <= 1:
        raise ValueError('Invalid thresholds')
    if ml_score is not None and (type(ml_score) not in (int, float) or not 0 <= ml_score <= 1):
        raise ValueError('ml_score must be between 0 and 1')
    if not approved_target:
        return dict(tier='HIGH', decision='BLOCK', reason='Destination not approved: deterministic local policy preview.', gate_required=False, simulated=True)
    if ml_score is None:
        return dict(tier='UNKNOWN', decision='REVIEW', reason='ML unavailable; fail closed to manual review in this demo.', gate_required=False, simulated=True)
    if ml_score >= high_threshold:
        return dict(tier='HIGH', decision='GATE', reason='High model score: require demo confirmation before local approval.', gate_required=True, simulated=True)
    if ml_score >= review_threshold:
        return dict(tier='MEDIUM', decision='REVIEW', reason='Model score exceeds validation-selected review threshold.', gate_required=False, simulated=True)
    return dict(tier='LOW', decision='ALLOW', reason='Approved destination and score below review threshold.', gate_required=False, simulated=True)
