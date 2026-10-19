import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/multichain_userop_features_20250401_20250414_processed.csv"
MANIFEST = ROOT / "data/processed/multichain_feature_manifest.json"


class TestMultichainPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA.exists() or not MANIFEST.exists():
            raise FileNotFoundError(
                "Processed data or manifest missing. Run "
                "'python scripts/prepare_multichain_features.py' first."
            )
        cls.df = pd.read_csv(DATA)
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_expected_row_count(self):
        self.assertEqual(len(self.df), 42000)
        self.assertEqual(self.manifest["rows"], 42000)

    def test_unique_chain_operation_pairs(self):
        self.assertFalse(self.df.duplicated(["chain", "userop_hash"]).any())
        self.assertEqual(
            self.df[["chain", "userop_hash"]].drop_duplicates().shape[0],
            42000,
        )

    def test_expected_chains_and_balanced_sample(self):
        expected = {
            "base", "op", "polynomial",
            "soneium", "unichain", "worldchain",
        }
        self.assertEqual(set(self.df["chain"].unique()), expected)
        self.assertTrue((self.df["chain"].value_counts() == 7000).all())

    def test_no_missing_values(self):
        self.assertFalse(self.df.isna().any().any())

    def test_constant_features_excluded(self):
        excluded = set(self.manifest["constant_candidate_features_excluded"])
        candidates = set(self.manifest["candidate_model_features"])

        self.assertIn("delegatecall_count", excluded)
        self.assertIn("log1p_delegatecall_count", excluded)
        self.assertNotIn("delegatecall_count", candidates)
        self.assertNotIn("log1p_delegatecall_count", candidates)

    def test_unvalidated_gas_features_excluded_from_model(self):
        candidates = set(self.manifest["candidate_model_features"])

        excluded_gas_features = {
            "actual_gas_used",
            "actual_gas_cost_eth",
            "log1p_actual_gas_used",
            "log1p_actual_gas_cost_eth",
        }

        self.assertTrue(excluded_gas_features.isdisjoint(candidates))

        # Keep the original measurements available for later investigation.
        for column in ("actual_gas_used", "actual_gas_cost_eth"):
            self.assertIn(column, self.df.columns)

    def test_no_maliciousness_label_claim(self):
        self.assertFalse(
            self.manifest["source_maliciousness_labels_available"]
        )
        self.assertFalse(
            self.manifest["success_is_maliciousness_label"]
        )

    def test_date_range(self):
        self.assertEqual(self.manifest["date_start"], "2025-04-01")
        self.assertEqual(self.manifest["date_end"], "2025-04-14")

    def test_success_is_boolean(self):
        self.assertTrue(self.df["success"].isin([True, False]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
