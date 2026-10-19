import unittest

from evaluate_operation import evaluate


class TestEvaluateOperation(unittest.TestCase):
    def setUp(self):
        self.operation = {
            "sender": "0x1111111111111111111111111111111111111111",
            "target": "0x0165878A594ca255338adfa4d48449f69242Eb8F",
            "selector": "0x55241077",
            "value_eth": 0,
            "gas_limit": 100000,
            "is_new_target": False,
            "is_approved_target": True,
        }

    def test_approved_normal_operation(self):
        result = evaluate(self.operation)
        self.assertEqual(result["risk_score"], 0)
        self.assertEqual(result["recommendation"], "ALLOW")

    def test_unapproved_target_is_flagged(self):
        operation = {**self.operation, "is_approved_target": False}
        result = evaluate(operation)
        self.assertEqual(result["risk_score"], 40)
        self.assertEqual(result["recommendation"], "FLAG")

    def test_new_unapproved_target_is_flagged(self):
        operation = {
            **self.operation,
            "is_approved_target": False,
            "is_new_target": True,
        }
        result = evaluate(operation)
        self.assertEqual(result["risk_score"], 55)
        self.assertEqual(result["recommendation"], "FLAG")

    def test_approval_selector_adds_risk(self):
        operation = {**self.operation, "selector": "0x095ea7b3"}
        result = evaluate(operation)
        self.assertEqual(result["risk_score"], 35)
        self.assertEqual(result["recommendation"], "ALLOW")

    def test_negative_value_is_rejected(self):
        operation = {**self.operation, "value_eth": -1}
        with self.assertRaises(ValueError):
            evaluate(operation)

    def test_missing_required_field_is_rejected(self):
        operation = dict(self.operation)
        del operation["sender"]
        with self.assertRaises(ValueError):
            evaluate(operation)


if __name__ == "__main__":
    unittest.main()
