import unittest
import numpy as np

from monitoring.ml_evaluator import FEATURES, evaluate, extract_features


class FakeModel:
    n_features_in_ = 17
    classes_ = [0, 1]

    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURES
        return np.array([[0.2, 0.8]])

    def predict(self, frame):
        return [1]


def valid_operation():
    return {
        "sender": "0x1111111111111111111111111111111111111111",
        "trace_count": 6,
        "unique_destinations": 3,
        "approve_trace_count": 0,
        "transfer_from_trace_count": 0,
        "value_transfer_trace_count": 0,
        "errored_trace_count": 0,
        "has_approval": False,
        "has_transfer_from": False,
        "has_value_transfer": False,
        "has_trace_errors": False,
        "has_paymaster": False,
    }


class MLEvaluatorTests(unittest.TestCase):
    def test_extracts_all_17_features(self):
        features = extract_features(valid_operation())
        self.assertEqual(len(features), 17)
        self.assertEqual(list(features), FEATURES)

    def test_derives_log_features(self):
        features = extract_features(valid_operation())
        self.assertAlmostEqual(features["log1p_trace_count"], 1.945910149)

    def test_rejects_missing_trace_features(self):
        operation = valid_operation()
        del operation["trace_count"]
        with self.assertRaisesRegex(ValueError, "Missing trace-derived features"):
            extract_features(operation)

    def test_rejects_negative_counts(self):
        operation = valid_operation()
        operation["trace_count"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            extract_features(operation)

    def test_rejects_inconsistent_flags(self):
        operation = valid_operation()
        operation["approve_trace_count"] = 1
        with self.assertRaisesRegex(ValueError, "conflicts"):
            extract_features(operation)

    def test_returns_advisory_score(self):
        result = evaluate(valid_operation(), model=FakeModel())
        self.assertEqual(result["risk_score"], 0.8)
        self.assertEqual(result["recommendation"], "REVIEW")
        self.assertTrue(result["recommendation_is_advisory"])
        self.assertFalse(result["enforces_on_chain"])


if __name__ == "__main__":
    unittest.main()
